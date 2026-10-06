"""Real aria2 versus native receivers on an identical, preloaded catalogue.

Run on a host with rootless network namespaces and existing ERA5 variants:
python bench/external_catalog_compare.py run VARIANTS ROOT ARIA2 --reps 3
Placements use either full public copies or sparse copies with closed holders. Discovery and source
registration precede timing; caches in every receiver start empty. Kernel TCP
shaping is shared by the native data port and the plain HTTP mirror port.
"""
import argparse
import asyncio
import copy
import faulthandler
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aiohttp
import numpy as np
import xarray as xr
from aiohttp import web

from sim.e_hetero import VAR, build_sat, verify_downloaded_chunks
from sim.netemu import EmuSwarm, ensure_router
from sim.simulate import fresh_client
from zarrswarm import codec, scan
from zarrswarm.node import Node, merge_view
from zarrswarm.store import http


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


async def serve(ip, mapping):
    files = json.loads(Path(mapping).read_text())
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=90))
    async def get(req):
        key = req.match_info.get('peer', '')
        key = (key + '/' if key else '') + req.match_info['cid']
        path = files.get(key)
        if path is None:
            raise web.HTTPNotFound()
        if isinstance(path, dict):
            # Reuse the production outbound tunnel. Both clients pay for the same
            # holder and relay links; aria2 sees an ordinary whole-file HTTP GET.
            async with session.post(path['url'], json={'keys': [path['key']]}) as r:
                if r.status != 200:
                    return web.Response(status=r.status)
                n = struct.unpack('>I', await r.content.readexactly(4))[0]
                if n != path['size']:
                    raise web.HTTPBadGateway(text='relay chunk length')
                resp = web.StreamResponse(headers={'Content-Length': str(n), 'Accept-Ranges': 'none'})
                await resp.prepare(req)
                while n:
                    data = await r.content.readexactly(min(n, 256 << 10))
                    await resp.write(data)
                    n -= len(data)
                await resp.write_eof()
                return resp
        return web.FileResponse(path)
    app = web.Application()
    app.add_routes([web.get('/{cid}', get)])
    app.add_routes([web.get('/{peer}/{cid}', get)])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, ip, 7020).start()
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await session.close()


def trace_receiver(n, home):
    """Keep phase timings in RAM; diagnostic writes must not stall the query."""
    events = []
    started = time.monotonic()
    def wrap(label, fn):
        def measured(*args, **kwargs):
            t = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                events.append({'phase': label, 'start_s': t - started,
                               'seconds': time.monotonic() - t})
        return measured
    n._verify_store = wrap('verify_store', n._verify_store)
    n._index = wrap('index', n._index)
    n._flush = wrap('flush', n._flush)
    codec.decode = wrap('decode', codec.decode)
    codec.vcid_of = wrap('value_hash', codec.vcid_of)
    for name in ('write_bytes', 'write_text', 'replace', 'mkdir', 'stat'):
        fn = getattr(Path, name)
        def measured(path, *args, _fn=fn, _name=name, **kwargs):
            if str(path).startswith(str(home)):
                return wrap('file_' + _name, _fn)(path, *args, **kwargs)
            return _fn(path, *args, **kwargs)
        setattr(Path, name, measured)
    trace = aiohttp.TraceConfig()
    async def begin(session, ctx, params):
        ctx.started = time.monotonic()
    async def end(session, ctx, params):
        events.append({'phase': 'http_headers', 'url': str(params.url),
                       'start_s': ctx.started - started, 'seconds': time.monotonic() - ctx.started})
    trace.on_request_start.append(begin)
    trace.on_request_end.append(end)
    trace.freeze()
    n.session.trace_configs.append(trace)
    return events


def validate(path, cid, vcid, docs, axis):
    raw = Path(path).read_bytes()
    assert scan.cid_of(raw) == cid, ('byte identity', path)
    got = codec.vcid_of(codec.decode(docs, raw), axis)
    assert codec.same_vcid(got, vcid), ('value identity', path)
    return len(raw)


async def probe_inbound(cat_path):
    cat = json.loads(Path(cat_path).read_text())
    async def blocked(ip, port):
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(ip, port), 0.3)
        except asyncio.TimeoutError:
            return
        writer.close()
        await writer.wait_closed()
        raise AssertionError(('private source reachable from receiver', ip, port))
    await asyncio.gather(*(blocked(p['private_ip'], port) for p in cat['placement'].values()
                           if p['nat'] for port in (7000, 7001, 7020)))


async def aria_download(n, cat, reg, home, aria2):
    v, gid = n.views[cat['grid']], cat['grid']
    keys, info = await n.region_keys(gid, dict(reg, var=VAR), 'bytes')
    records, files, lines = {}, {}, []
    for i, key in enumerate(keys):
        b = v['best'][key]
        groups = {}
        for p, cid, size in b['src']:
            groups.setdefault(cid, []).append((p, size))
        # Mirrors of a single aria2 download must have identical bytes.
        cid = max(sorted(groups), key=lambda c: sum(cat['rates'][p] for p, _ in groups[c]))
        peers = sorted(groups[cid], key=lambda x: (-cat['rates'][x[0]], x[0]))
        name, layout, _ = scan.split_key(key)
        encoding = v['pinfo'][peers[0][0]][name]['layouts'][layout]
        docs = encoding['docs']
        # Array extents differ between hourly and 6-hourly copies of identical chunks.
        assert all(v['pinfo'][p][name]['layouts'][layout]['fid'] == encoding['fid'] for p, _ in peers)
        out = f'{i:05d}-{cid}'
        files[key] = str(home / out)
        records[out] = (cid, b['vcid'], docs, v['arrays'][name]['taxis'])
        urls = '\t'.join(cat['http'][p] + '/' + cid for p, _ in peers)
        lines.append(urls + '\n  out=' + out)
    coverage = __import__('zarrswarm.jlps', fromlist=['region_coverage']).region_coverage(
        VAR, v['arrays'], v['best'], keys, info['request']['g_lo'], info['request']['g_hi'],
        reg.get('isel'), tuple(info['lattice']))
    assert coverage['missing_samples'] == 0, coverage
    input_path = home / 'input.txt'
    input_path.write_text('\n'.join(lines) + '\n')
    cmd = [aria2, '--no-conf=true', '--no-netrc=true', '--all-proxy=', '--file-allocation=none',
           '--auto-file-renaming=false', '--allow-overwrite=false', '--max-concurrent-downloads=40',
           '--split=' + str(cat.get('aria_split', 4)), '--max-connection-per-server=4', '--min-split-size=1M', '--uri-selector=adaptive',
           '--connect-timeout=10', '--timeout=60', '--max-tries=3', '--enable-rpc=true',
           '--rpc-listen-port=7021', '--max-download-result=10000', '--summary-interval=0',
           '--console-log-level=warn', '--dir=' + str(home), '--input-file=' + str(input_path)]
    pending, seen = [], set()
    start = time.monotonic()
    with open(home / 'aria2.log', 'w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2), trust_env=False) as session:
                async def rpc(method, args):
                    async with session.post('http://127.0.0.1:7021/jsonrpc', json={
                        'jsonrpc': '2.0', 'id': 'benchmark', 'method': method, 'params': args}) as r:
                        result = await r.json()
                        if 'error' in result:
                            raise RuntimeError(result['error'])
                        return result['result']
                while len(seen) < len(records):
                    if time.monotonic() - start > 1800:
                        raise TimeoutError('aria2 deadline')
                    try:
                        stopped = await rpc('aria2.tellStopped', [0, 10000])
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if proc.poll() is not None:
                            # RPC disappears on a clean exit; verify all remaining files then.
                            assert proc.returncode == 0, (proc.returncode, (home / 'aria2.log').read_text()[-2000:])
                            stopped = [{'status': 'complete', 'files': [{'path': str(home / out)}]}
                                       for out in records if out not in seen]
                        else:
                            await asyncio.sleep(0.03)
                            continue
                    for item in stopped:
                        assert item['status'] == 'complete', item
                        out = Path(item['files'][0]['path']).name
                        if out in seen:
                            continue
                        assert out in records, out
                        seen.add(out)
                        pending.append(asyncio.create_task(asyncio.to_thread(validate, home / out, *records[out])))
                    await asyncio.sleep(0.03)
                transfer_s = time.monotonic() - start
                sizes = await asyncio.gather(*pending)
        finally:
            if proc.poll() is None:
                proc.terminate()
            await asyncio.to_thread(proc.wait, 10)
    return files, {'coverage': coverage, 'bytes': sum(sizes), 'done_keys': keys,
                   'state': 'done', 'transfer_s': transfer_s, 'cover': info}


async def receive(cat_path, arm, reg_json, home, ip, aria2, out):
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=4))
    cat, reg = json.loads(Path(cat_path).read_text()), json.loads(reg_json)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    n = Node(home / 'native', host=ip, port=7005, ctl_port=7006)
    await n.start()
    v = copy.deepcopy(cat['view'])
    v.update(ts=time.time() + 86400, addrs=cat['addresses'])
    n.views[cat['grid']], n.hints = v, dict(cat['rates'])
    n.xt1 = False  # compare transfers of the same stored chunk bytes
    read_runner = None
    events = trace_receiver(n, home)
    lag = []
    async def heartbeat():
        while True:
            t = time.monotonic()
            await asyncio.sleep(0.1)
            lag.append(max(0.0, time.monotonic() - t - 0.1))
    heartbeat_task = asyncio.create_task(heartbeat())
    stack_path = Path('/tmp') / f'zt-query-stacks-{os.getpid()}.txt'
    stack = stack_path.open('w')
    try:
        faulthandler.dump_traceback_later(30, repeat=True, file=stack)
        started = time.monotonic()
        cpu_started = time.process_time()
        if arm == 'aria2-min':
            files, job = await aria_download(n, cat, reg, home, aria2)
        else:
            jid = await n.start_job({'grid': cat['grid'], 'region': dict(reg, var=VAR),
                                     'cover': 'jlps' if arm == 'native-jlps' else 'bytes'})
            while n.jobs[jid]['state'] == 'running':
                if time.monotonic() - started > 1800:
                    raise TimeoutError('native deadline')
                await asyncio.sleep(0.03)
            job = n.jobs[jid]
            files = n.local.get(cat['grid'], {}).get('files', {})
        seconds = time.monotonic() - started
        cpu_seconds = time.process_time() - cpu_started
        faulthandler.cancel_dump_traceback_later()
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        diagnostic = {'events': list(events), 'max_event_loop_lag_s': max(lag, default=0),
                      'cpu_seconds': cpu_seconds}
        stack.flush()
        diagnostic['stacks'] = stack_path.read_text()
        assert job['state'] == 'done' and job['coverage']['missing_samples'] == 0, job
        # Identical post-timing source comparison, using only receiver-local payloads.
        async def read(req):
            d = await req.json()
            if d['grid'] != cat['grid'] or d['key'] not in files:
                raise web.HTTPNotFound()
            return web.Response(body=await asyncio.to_thread(Path(files[d['key']]).read_bytes))
        app = web.Application()
        app.add_routes([web.post('/api/read', read)])
        read_runner = web.AppRunner(app, access_log=None)
        await read_runner.setup()
        await web.TCPSite(read_runner, '127.0.0.1', 7022).start()
        await asyncio.to_thread(verify_downloaded_chunks, 'http://127.0.0.1:7022', cat['grid'], v,
                                job['done_keys'], cat['truth'])
        dump(out, {'arm': arm, 'seconds': seconds, 'job': job, 'value_check': 'exact',
                   'python_receiver_cpu_s': cpu_seconds,
                   'diagnostic': diagnostic,
                   'catalogue_sha256': hashlib.sha256(Path(cat_path).read_bytes()).hexdigest()})
    finally:
        faulthandler.cancel_dump_traceback_later()
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        stack.close()
        stack_path.unlink(missing_ok=True)
        if read_runner:
            await read_runner.cleanup()
        await n.stop()


def run(a):
    ensure_router()
    root, variants = Path(a.root).resolve(), Path(a.variants).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(ZT_KEEP_LOGS='1', ZT_SCAN_WORKERS='2', ZT_HASH_CACHE=str(root / 'hashcache'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    source = Path(__file__).resolve().parents[1]
    truth = variants / 'V1-arco-1h.zarr'
    with xr.open_zarr(truth, consolidated=False) as ds:
        times = ds.time.values
        day, first, last = str(times[24])[:10], str(times[0])[:19], str(times[-1])[:19]
    queries = {'day_maps': {'t0': day, 't1': day, 'step': 3600},
               'point_series': {'t0': first, 't1': last, 'step': 3600,
                                'isel': {'latitude': [300, 301], 'longitude': [500, 501]}},
               'month_6h': {'t0': first, 't1': last, 'step': 21600}}
    if a.smoke:
        queries = {'smoke': {'t0': first, 't1': str(times[3])[:19], 'step': 3600}}
    paths = sorted(variants.glob('V*.zarr'))
    assert len(paths) == 5, paths
    # Scan each existing representation once; never copy the month of data.
    scans = {p.stem: scan.scan(p, root / 'scan', workers=2) for p in paths}
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
                'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                'aria2_version': subprocess.check_output([a.aria2, '--version'], text=True).splitlines()[0],
                'aria2_sha256': hashlib.sha256(Path(a.aria2).read_bytes()).hexdigest(),
                'estimator': codec.ESTIMATOR, 'smoke': a.smoke, 'placement': a.placement,
                'cycles': a.cycles, 'nat': a.placement == 'partial-nat',
                'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((source / 'zarrswarm').glob('*.py')) + [Path(__file__).resolve()]},
                'protocol': ('22 sparse stations and 11 window mirrors, 30% NAT, blocked inbound ports, shared outbound relays; ' if a.placement == 'partial-nat' else 'Two full public holders per layout; ') + 'preloaded common catalogue and declared rates; '
                            'fresh receiver caches; kernel TCP shaping; stored bytes; four verification workers; '
                            'minimum-byte cover for aria2; source-value checks outside query timing.'}
    rows = []
    output = root / ('smoke.json' if a.smoke else 'results.json')
    for rep in range(a.rep_start, 1 if a.smoke else a.reps):
        partial = a.placement == 'partial-nat'
        sw = EmuSwarm(root / ('smoke' if a.smoke else f'rep{rep}'), 36 if partial else 11,
                      3 if partial else 1, 0.3 if partial else 0, seed=5100 + rep,
                      rate_median=1e6, rate_range=(0.2e6, 5e6), client_rate=30e6)
        servers = []
        try:
            mans, addresses, rates, mirrors, placement = {}, {}, {}, {}, {}
            if partial:
                holders = build_sat(sw, {p.stem: p for p in paths}, random.Random(6100 + rep),
                                    json.loads(Path(a.traces).read_text()), 22)
                assignment = {peer: sw.root / 'rep' / peer for peer in holders}
            else:
                ordered = paths * 2
                random.Random(6100 + rep).shuffle(ordered)
                assignment = dict(zip([p for p in sw.nodes if not p.startswith('boot')], ordered))
            relay_maps = {}
            for peer, path in assignment.items():
                if not partial:
                    http(sw.ctl(peer), 'POST', '/api/seed', {'path': str(path)}, timeout=900)
                st = http(sw.ctl(peer), 'GET', '/api/status')
                subgrids = (scan.scan(path, root / 'scan', workers=2) if partial else scans[path.stem])['subgrids']
                gid, man = next((g, m) for g, m in subgrids.items() if VAR in m['arrays'])
                nid, node = st['id'], sw.nodes[peer]
                mans[nid], addresses[nid], rates[nid] = man, st['addr'], sw.meta[peer]['rate']
                mirrors[nid] = f'http://{node.ip}:7020'
                placement[peer] = {'node': nid, 'private_ip': node.ip, 'layout': holders[peer][0] if partial else path.stem,
                                   'replica': holders[peer] if partial else 'full', **sw.meta[peer]}
                if sw.meta[peer]['nat']:
                    relay = st['addr'].split('/r/')[0]
                    maps = relay_maps.setdefault(relay, {})
                    for key, ent in man['chunks'].items():
                        maps[nid + '/' + ent[0]] = {'url': st['addr'] + '/cb/' + gid,
                                                   'key': key, 'size': ent[2]}
                    mirrors[nid] = relay.replace(':7000', ':7020') + '/' + nid
                    iface = f'n{int(node.ip.split(".")[2]) * 250 + int(node.ip.split(".")[3]) - 1}'
                    # Drop inbound data traffic on the veth; outbound relay connections
                    # and loopback reads remain usable. Control access is setup-only.
                    subprocess.run(['nsenter', '-t', str(node.proc.pid), '-n', 'tc', 'qdisc', 'add',
                                    'dev', iface, 'clsact'], check=True)
                    for port in (7000, 7020):
                        subprocess.run(['nsenter', '-t', str(node.proc.pid), '-n', 'tc', 'filter', 'add',
                            'dev', iface, 'ingress', 'protocol', 'ip', 'pref', str(port), 'flower',
                            'ip_proto', 'tcp', 'dst_port', str(port), 'action', 'drop'], check=True)
                        try:
                            with socket.create_connection((node.ip, port), timeout=0.25):
                                raise AssertionError(('NAT bypass', peer, port))
                        except socket.timeout:
                            pass
                    # The setup controller alone may reach the administrative port.
                    subprocess.run(['nsenter', '-t', str(node.proc.pid), '-n', 'tc', 'filter', 'add',
                        'dev', iface, 'ingress', 'protocol', 'ip', 'pref', '1', 'flower', 'ip_proto',
                        'tcp', 'src_ip', '10.0.255.254', 'dst_port', '7001', 'action', 'pass'], check=True)
                    subprocess.run(['nsenter', '-t', str(node.proc.pid), '-n', 'tc', 'filter', 'add',
                        'dev', iface, 'ingress', 'protocol', 'ip', 'pref', '7001', 'flower', 'ip_proto',
                        'tcp', 'dst_port', '7001', 'action', 'drop'], check=True)
                    placement[peer]['inbound_data_blocked'] = True
                    placement[peer]['control_controller_only'] = True
                    continue
                mapping = sw.root / f'{peer}-files.json'
                dump(mapping, {ent[0]: man['files'][key] for key, ent in man['chunks'].items()})
                log = open(sw.root / f'{peer}-http.log', 'w')
                servers.append(subprocess.Popen(['nsenter', '-t', str(node.proc.pid), '-n', sys.executable,
                    str(Path(__file__).resolve()), 'serve', node.ip, str(mapping)], stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True))
                log.close()
            for relay, files in relay_maps.items():
                boot = next(node for node in sw.nodes.values() if f'http://{node.ip}:7000' == relay)
                mapping = sw.root / f'relay-{boot.ip}-files.json'
                dump(mapping, files)
                log = open(sw.root / f'relay-{boot.ip}-http.log', 'w')
                servers.append(subprocess.Popen(['nsenter', '-t', str(boot.proc.pid), '-n', sys.executable,
                    str(Path(__file__).resolve()), 'serve', boot.ip, str(mapping)], stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True))
                log.close()
            v = merge_view(gid, mans, 'receiver')
            cat_path = sw.root / 'catalogue.json'
            dump(cat_path, {'grid': gid, 'view': v, 'addresses': addresses, 'rates': rates,
                            'http': mirrors, 'truth': str(truth), 'placement': placement,
                            'aria_split': 1 if partial else 4})
            time.sleep(1)
            if partial:
                client = fresh_client(sw, 'inbound_check', 'maxflow')
                try:
                    subprocess.run(['nsenter', '-t', str(sw.nodes[client].proc.pid), '-n', sys.executable,
                                    str(Path(__file__).resolve()), 'probe', str(cat_path)], check=True, timeout=30)
                finally:
                    sw.kill(client)
                    shutil.rmtree(sw.root / f'h_{client}', ignore_errors=True)
                def check_tunnel(peer):
                    key, ent = min(((k, e) for k, e in mans[peer]['chunks'].items()
                                    if scan.split_key(k)[0] == VAR), key=lambda item: item[1][2])
                    raw = http(mirrors[peer], 'GET', '/' + ent[0], raw=True, timeout=180)
                    assert scan.cid_of(raw) == ent[0] and len(raw) == ent[2], peer
                    return {'node': peer, 'cid': ent[0], 'bytes': len(raw)}
                private = [d['node'] for d in placement.values() if d['nat']]
                with ThreadPoolExecutor(4) as executor:
                    preflight = list(executor.map(check_tunnel, private))
                dump(sw.root / 'nat_preflight.json', {'blocked_ports': [7000, 7001, 7020],
                                                     'probe_from_receiver': True,
                                                     'verified_tunnels': preflight})
                metadata.setdefault('nat_preflights', {})[rep] = len(preflight)
            def relay_bytes():
                return sum(http(sw.ctl(p), 'GET', '/api/status')['served']['relayed_bytes']
                           for p in sw.nodes if p.startswith('boot'))
            for cycle, query, reg in [(c, q, r) for c in range(a.cycles) for q, r in queries.items()]:
                if not a.smoke and query not in a.queries.split(','):
                    continue
                arms = ['native-jlps', 'native-min', 'aria2-min']
                qi = list(queries).index(query)
                shift = (rep + qi + (cycle if a.rotate_cycles else 0)) % len(arms)
                for order, arm in enumerate(arms[shift:] + arms[:shift]):
                    tag = f'{rep}_{cycle}_{query}_{arm}'
                    client = fresh_client(sw, tag, 'maxflow')
                    home, result = sw.root / f'receiver_{tag}', sw.root / f'{tag}.json'
                    try:
                        before_relay = relay_bytes() if partial else 0
                        cmd = ['nsenter', '-t', str(sw.nodes[client].proc.pid), '-n', sys.executable,
                               str(Path(__file__).resolve()), 'receive', str(cat_path), arm, json.dumps(reg),
                               str(home), sw.nodes[client].ip, a.aria2, str(result)]
                        with open(sw.root / f'{tag}.log', 'w') as log:
                            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2100)
                        row = json.loads(result.read_text()) | {'rep': rep, 'cycle': cycle, 'query': query, 'order': order}
                        if partial:
                            row['relay_bytes_delta'] = relay_bytes() - before_relay
                        rows.append(row)
                        print(json.dumps({k: row[k] for k in ('rep', 'query', 'arm', 'seconds', 'value_check')}), flush=True)
                        dump(output, {'metadata': metadata, 'rows': rows})
                    finally:
                        sw.kill(client)
                        shutil.rmtree(sw.root / f'h_{client}', ignore_errors=True)
                        # Only this run's fresh receiver files are temporary.
                        shutil.rmtree(home, ignore_errors=True)
        finally:
            try:
                for proc in servers:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=10)
            finally:
                sw.stop()
                if partial:
                    shutil.rmtree(sw.root / 'rep', ignore_errors=True)
    metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
    dump(output, {'metadata': metadata, 'rows': rows})


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'serve':
        asyncio.run(serve(*args))
    elif mode == 'receive':
        asyncio.run(receive(*args))
    elif mode == 'probe':
        asyncio.run(probe_inbound(*args))
    elif mode == 'run':
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument('variants')
        ap.add_argument('root')
        ap.add_argument('aria2')
        ap.add_argument('--reps', type=int, default=3)
        ap.add_argument('--rep-start', type=int, default=0)
        ap.add_argument('--queries', default='day_maps,point_series,month_6h')
        ap.add_argument('--smoke', action='store_true', help='one small query, all three real receiver arms')
        ap.add_argument('--placement', choices=['public', 'partial-nat'], default='public')
        ap.add_argument('--traces', default='bench/sat_traces_meteor.json')
        ap.add_argument('--cycles', type=int, default=1)
        ap.add_argument('--rotate-cycles', action='store_true')
        run(ap.parse_args(args))
    else:
        raise ValueError(mode)
