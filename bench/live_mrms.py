"""Live public MRMS ingestion; local node instances, not a station deployment.

Keep source timestamps and decoded float64 values. Two stores use different
chunking/codecs; the result records the configured scan/subscription intervals.
"""
import argparse
import asyncio
import gzip
import hashlib
import importlib.metadata
import json
import os
import resource
import socket
import sys
import time
import traceback
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import gribberish
import numcodecs
import numpy as np
import xarray as xr
from zarr.codecs import ZstdCodec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import e_hetero
from zarrswarm import codec, jlps, node as nodemod
from zarrswarm.node import Node
from zarrswarm.scan import chunk_extent, split_key, tstride

VAR = "precipitation_rate"
BASE = "https://noaa-mrms-pds.s3.amazonaws.com/"
PRODUCT = "CONUS/PrecipRate_00.00/"
CROP = (slice(1000, 2024), slice(2500, 3524))


def now():
    return datetime.now(timezone.utc).isoformat()


def listing():
    prefix = PRODUCT + datetime.now(timezone.utc).strftime("%Y%m%d/")
    url = BASE + "?" + urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "max-keys": 1000})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache"}), timeout=30) as response:
        root = ET.fromstring(response.read())
        server_date = response.headers.get("Date")
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    rows = [{n: v.findtext("s:" + n, namespaces=ns) for n in ("Key", "LastModified", "Size", "ETag")}
            for v in root.findall("s:Contents", ns)]
    return rows, {"observed_utc": now(), "http_date": server_date, "url": url, "objects": len(rows),
                  "latest": rows[-1] if rows else None}


def decode(row, work):
    started, cpu = time.monotonic(), time.process_time()
    with urllib.request.urlopen(BASE + row["Key"], timeout=30) as response:
        payload = response.read()
    assert len(payload) == int(row["Size"])
    raw = gzip.decompress(payload)
    message = gribberish.parse_grib_message(raw, 0)
    meta = gribberish.parse_grib_message_metadata(raw, 0)
    values = message.data().reshape(meta.grid_shape)[CROP].copy()
    assert values.dtype == np.dtype("f8") and values.shape == (1024, 1024)
    x, y = meta.xy()
    ts = np.datetime64(meta.forecast_date.replace(tzinfo=None), "s")
    assert str(ts).replace("-", "").replace("T", "-").replace(":", "") in row["Key"]
    dataset = xr.Dataset({VAR: (("time", "latitude", "longitude"), values[None], {"units": meta.units})},
                         coords={"time": [ts], "latitude": y[CROP[0]], "longitude": x[CROP[1]]})
    name = Path(row["Key"]).name
    (work / "sources" / name).write_bytes(payload)
    np.save(work / "sources" / (name + ".npy"), values, allow_pickle=False)
    return dataset, dict(row, source_url=BASE + row["Key"], fetched_utc=now(), time=str(ts),
                        grib_sha256=hashlib.sha256(payload).hexdigest(),
                        crop_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        decode_fetch_s=time.monotonic() - started, decode_fetch_cpu_s=time.process_time() - cpu,
                        minimum=float(values.min()), maximum=float(values.max()),
                        nonzero_cells=int(np.count_nonzero(values)))


def store(dataset, path, temporal, initial=False):
    if initial:
        encoding = {VAR: {"chunks": (4 if temporal else 1, 512, 512),
                         **({"compressors": [ZstdCodec(level=3)]} if temporal else
                            {"compressor": numcodecs.Blosc("lz4", 5, numcodecs.Blosc.SHUFFLE)})}}
        dataset.to_zarr(path, encoding=encoding, consolidated=False, zarr_format=3 if temporal else 2)
    else:
        dataset.to_zarr(path, append_dim="time", consolidated=False)


def cached_keys(client, grid, epoch):
    local = client.local.get(grid)
    if not local:
        return None, []
    t = local["grid"]["time"]
    g = (epoch - t["rphase"]) // t["dt"]
    keys, best = [], {}
    for key, ent in list(local["chunks"].items()):
        name, layout, co = split_key(key)
        if name != VAR:
            continue
        a = local["arrays"][name]
        li, axis = a["layouts"][layout], a["taxis"]
        lo, hi = chunk_extent(li, axis, co[axis], ent[3])
        if lo <= g < hi and (g - lo) % tstride(li)[0] == 0:
            keys.append(key)
            best[key] = {"src": [(client.ident.id, ent[0], ent[2])], "nv": ent[3]}
    lattice = nodemod._lattice(local["arrays"][VAR], t)
    coverage = jlps.region_coverage(VAR, local["arrays"], best, keys, g, g + 1, {}, lattice)
    return local, keys if coverage["covered_samples"] == 1 else []


async def wait_cached(client, grid, epoch, work, committed):
    while True:
        local, keys = cached_keys(client, grid, epoch)
        if keys:
            record = dict(cached_utc=now(), commit_to_cache_s=time.monotonic() - committed,
                          keys=keys, layouts=sorted({split_key(k)[1] for k in keys}))
            await asyncio.to_thread(e_hetero.verify_downloaded_chunks,
                f"http://127.0.0.1:{client.ctl_port}", grid, local, keys, work / "maps.zarr")
            record["received_value_check"] = "exact numerical equality on received payloads"
            return record
        if time.monotonic() - committed > 300:
            raise TimeoutError("subscriber did not receive a complete latest map")
        await asyncio.sleep(0.25)


async def main(args):
    global CROP
    CROP = tuple(slice(i, i + 1024) for i in args.crop)
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(4))
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert not (work / "live_mrms_v5.json").exists(), "use a fresh run directory"
    (work / "sources").mkdir(exist_ok=True)
    e_hetero.VAR = VAR
    result = {"started_utc": now(), "status": "running", "rows": [], "polls": [], "protocol": {
        "source": BASE + PRODUCT, "initial_fields": 2, "future_updates": args.updates,
        "listing_poll_s": 10, "rescan_s": nodemod.RESCAN_EVERY, "follow_s": nodemod.FOLLOW_EVERY,
        "crop_indices": [[s.start, s.stop] for s in CROP], "dtype": "float64 unchanged after gribberish decoding",
        "scope": "One process with three node instances on one shared host; public live ingestion, loopback replication.",
        "timing": "Latest-map availability in the subscriber's existing cache; validation never fetches missing source data."},
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "xarray", "zarr", "gribberish")},
        "source_sha256": {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in list((Path(__file__).parents[1] / "zarrswarm").glob("*.py")) + [Path(__file__)]}}
    save = lambda: (work / "live_mrms_v5.json").write_text(json.dumps(result, indent=2) + "\n")
    save()  # protocol is recorded before listing or fetching fields
    nodes, ports = [], iter(range(14000, 30000))
    def port():
        for candidate in ports:
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", candidate))
                    return candidate
                except OSError:
                    continue
        raise RuntimeError("no free port")
    started, cpu_start = time.monotonic(), time.process_time()
    try:
        rows, poll = await asyncio.to_thread(listing)
        result["polls"].append(poll)
        initial = [await asyncio.to_thread(decode, row, work) for row in rows[-2:]]
        data = xr.concat([d for d, _ in initial], dim="time")
        assert np.diff(data.time.values.astype("datetime64[s]").astype("i8")).tolist() == [120]
        for temporal in (False, True):
            await asyncio.to_thread(store, data, work / ("series.zarr" if temporal else "maps.zarr"), temporal, True)
        boot_port = port()
        a = Node(work / "node_maps", port=boot_port, ctl_port=port())
        b = Node(work / "node_series", port=port(), ctl_port=port(), bootstrap=[a.addr])
        client = Node(work / "node_client", port=port(), ctl_port=port(), bootstrap=[a.addr])
        for node in (a, b, client):
            nodes.append(node)
            await node.start()
        grids = [(await node.add_seed(str(work / name)))["subgrids"] for node, name in
                 ((a, "maps.zarr"), (b, "series.zarr"))]
        assert list(grids[0]) == list(grids[1])
        grid = next(iter(grids[0]))
        sub = {"id": "live", "link": "zs://" + grid, "vars": [VAR], "last_s": 240}
        client.subs[sub["id"]] = sub
        await client.follow_once(sub)
        last_time = data.time.values[-1].astype("datetime64[s]").astype("i8")
        warm = await wait_cached(client, grid, int(last_time), work, time.monotonic())
        result.update(grid=grid, initial=[r for _, r in initial], initialized_utc=now(), warm=warm)
        save()
        seen = {row["Key"] for row in rows}
        while len(result["rows"]) < args.updates:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("live observation deadline")
            rows, poll = await asyncio.to_thread(listing)
            result["polls"].append(poll)
            fresh = [row for row in rows if row["Key"] not in seen]
            if not fresh:
                save()
                await asyncio.sleep(10)
                continue
            row = fresh[0]
            seen.add(row["Key"])
            dataset, record = await asyncio.to_thread(decode, row, work)
            epoch = int(dataset.time.values[0].astype("datetime64[s]").astype("i8"))
            assert epoch - last_time == 120, "do not normalize or skip source timestamps"
            last_time = epoch
            record["first_observed_utc"] = poll["observed_utc"]
            for temporal in (False, True):
                await asyncio.to_thread(store, dataset, work / ("series.zarr" if temporal else "maps.zarr"), temporal)
            committed = time.monotonic()
            record["committed_utc"] = now()
            result["rows"].append(record)
            save()
            record.update(await wait_cached(client, grid, epoch, work, committed))
            result["jobs"] = [{k: v for k, v in job.items() if k != "done_keys"} for job in client.jobs.values()]
            save()
            print(json.dumps(record), flush=True)
        result["status"] = "complete"
    except BaseException:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        for node in reversed(nodes):
            await node.stop()
        result.update(finished_utc=now(), wall_s=time.monotonic() - started,
                      total_process_cpu_s=time.process_time() - cpu_start,
                      peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      served={node.home.name: node.served for node in nodes})
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--crop", nargs=2, type=int, default=[1000, 2500], metavar=("LAT_INDEX", "LON_INDEX"))
    args = parser.parse_args()
    asyncio.run(main(args))
