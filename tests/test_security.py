"""Regression tests for review findings: poisoning, infinite retry, control-API guard, probe SSRF, DHT input."""
import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zarr_torrent as zt
from zarr_torrent.node import Node
from zarr_torrent.store import http, wait_job
_ports = iter(range(10000, 30000))


def port():
    for p in _ports:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free test port")


def make(path, values, times):
    xr.Dataset({"sec": (("time", "y", "x"), values)},
               coords={"time": times, "y": np.arange(4.0), "x": np.arange(5.0)}).to_zarr(
        path, encoding={"sec": {"chunks": (12, 4, 5)}}, consolidated=False)
    return str(path)


@pytest.fixture
def net(tmp_path):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(60)
    bp = port()
    nodes = {"boot": Node(tmp_path / "boot", port=bp, ctl_port=port())}
    run(nodes["boot"].start())
    for n in ("h1", "h2", "evil", "client"):
        nodes[n] = Node(tmp_path / n, port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
        run(nodes[n].start())
    yield nodes, run, tmp_path
    for n in nodes.values():
        run(n.stop())
    loop.call_soon_threadsafe(loop.stop)


def ctl(n):
    return f"http://127.0.0.1:{n.ctl_port}"


def test_opposing_slice_poisoning_is_rejected_at_receiver(net):
    from zarr_torrent.node import merge_view
    nodes, run, tmp = net
    times = pd.date_range("2020-01-01", periods=24, freq="h")
    good = np.tile(np.arange(20, dtype="f4").reshape(1, 4, 5), (24, 1, 1))
    bad = good.copy()
    bad[:6] += 1
    bad[6:12] -= 1
    for n, data in (("h1", good), ("evil", bad)):
        run(nodes[n].add_seed(make(tmp / f"{n}_shift.zarr", data, times)))
    grid = next(g for g, m in nodes["h1"].local.items() if "sec" in m["arrays"])
    honest, evil = nodes["h1"].local[grid], nodes["evil"].local[grid]
    key = sorted(k for k in honest["chunks"] if k.startswith("sec@"))[0]
    raw = Path(evil["files"][key]).read_bytes()
    # Advertise the honest identity alongside the attacker's genuine byte hash.
    evil["chunks"][key][1] = honest["chunks"][key][1]
    ep = nodes["evil"].ident.id
    view = merge_view(grid, {nodes["h1"].ident.id: honest, ep: evil}, "client")
    assert nodes["client"]._verify_store(view, key, ep, raw) is None


def test_legacy_cached_ids_are_recomputed_without_removing_data(net):
    from zarr_torrent import codec
    nodes, run, tmp = net
    values = np.tile(np.arange(20, dtype="f4").reshape(1, 4, 5), (24, 1, 1))
    times = pd.date_range("2020-01-01", periods=24, freq="h")
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": make(tmp / "migration.zarr", values, times)})["link"]
    client = nodes["client"]
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(client)).sec.values, values)
    run(client.stop())
    blobs = {}
    for path in (client.home / "cache").glob("*.json"):
        manifest = json.loads(path.read_text())
        for key, entry in manifest["chunks"].items():
            if key.startswith("sec@"):
                blobs[entry[0]] = client._cas(entry[0]).read_bytes()
                entry[1] = "L2:" + "a" * 32 + ":[[0,1]]"
        path.write_text(json.dumps(manifest))
    assert blobs
    run(client.start())
    for manifest in client.caches.values():
        for key, entry in manifest["chunks"].items():
            if key.startswith("sec@"):
                assert entry[1].startswith("L3:") and codec.same_vcid(entry[1], entry[1])
    assert all(client._cas(cid).read_bytes() == raw for cid, raw in blobs.items())


def test_identity_groups_are_independent_of_announcement_order(net):
    from itertools import permutations
    from zarr_torrent.codec import vcid_of
    from zarr_torrent.node import merge_view
    nodes, run, tmp = net
    good = np.tile(np.arange(20, dtype="f4").reshape(1, 4, 5), (24, 1, 1))
    times = pd.date_range("2020-01-01", periods=24, freq="h")
    run(nodes["h1"].add_seed(make(tmp / "groups.zarr", good, times)))
    grid = next(g for g, m in nodes["h1"].local.items() if "sec" in m["arrays"])
    import copy
    manifests = {}
    for i, offset in enumerate((0, .2, .4)):
        m = copy.deepcopy(nodes["h1"].local[grid])
        for k, ent in m["chunks"].items():
            if k.startswith("sec@"):
                ent[1] = vcid_of(good[:12] + offset, 0)
        manifests[str(i)] = m
    results = [merge_view(grid, dict(order), "client")["best"] for order in permutations(manifests.items())]
    # Holder lists are compared as sets; the elected identities and conflict decisions must be identical.
    normalize = lambda v: {k: (b["vcid"], frozenset(b["src"]), b.get("contested")) for k, b in v.items()}
    assert all(normalize(v) == normalize(results[0]) for v in results)


def test_majority_beats_conflicting_replica_and_tampered_bytes_are_rejected(net):
    nodes, run, tmp = net
    times = pd.date_range("2020-01-01", periods=48, freq="h")
    good = np.random.default_rng(1).random((48, 4, 5)).astype("float32")
    p1, p2 = make(tmp / "h1.zarr", good, times), make(tmp / "h2.zarr", good, times)
    pe = make(tmp / "evil.zarr", good + 1000, times)  # same structure, different values (conflicting replica)
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": p1})["link"]
    http(ctl(nodes["h2"]), "POST", "/api/seed", {"path": p2})
    http(ctl(nodes["evil"]), "POST", "/api/seed", {"path": pe})
    ds = zt.open_dataset(link, ctl=ctl(nodes["client"]))
    np.testing.assert_array_equal(ds.sec.values, good)
    # the conflicting holder may serve the (identical) coordinate chunks, never a data chunk (960 B each)
    assert nodes["evil"].served["bytes"] < 500, nodes["evil"].served


def test_one_liar_against_one_honest_holder_is_not_served(net):
    """1 vs 1 with equally complete chunks: no majority decides, so the chunk is not served at all (an arbitrary
    tie-break let a liar win by grinding its hash); a trusted publisher settles it."""
    nodes, run, tmp = net
    times = pd.date_range("2020-02-01", periods=24, freq="h")
    good = np.random.default_rng(4).random((24, 4, 5)).astype("float32")
    p1 = make(tmp / "honest.zarr", good, times)
    pe = make(tmp / "liar.zarr", good + 1000, times)
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": p1})["link"]
    http(ctl(nodes["evil"]), "POST", "/api/seed", {"path": pe})
    grid = link.removeprefix("zt://")
    jid = http(ctl(nodes["client"]), "POST", "/api/download", {"grid": grid, "region": {"var": "sec"}})["job"]
    job = wait_job(ctl(nodes["client"]), jid)
    assert job["done"] == 0 and job["missing"] == job["total"] == 2, job  # both chunks contested, none accepted
    nodes["client"].trusted.add(nodes["h1"].ident.id)
    nodes["client"].views.clear()
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(nodes["client"])).sec.values, good)


def test_tampered_seed_file_is_dropped_not_retried_forever(net):
    nodes, run, tmp = net
    times = pd.date_range("2021-01-01", periods=24, freq="h")
    vals = np.random.default_rng(2).random((24, 4, 5)).astype("float32")
    p = make(tmp / "only.zarr", vals, times)
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": p})["link"]
    for f in Path(p, "sec").rglob("*"):  # corrupt every chunk after it was hashed and announced
        if f.is_file() and f.name != "zarr.json":
            f.write_bytes(b"\x00" * f.stat().st_size)
    t0 = time.time()
    ds = zt.open_dataset(link, ctl=ctl(nodes["client"]))
    with pytest.raises(Exception, match="download failed"):
        ds.sec.values
    assert time.time() - t0 < 30


def test_control_api_requires_local_client_header(net):
    nodes, _, _ = net
    req = urllib.request.Request(ctl(nodes["h1"]) + "/api/status")  # no X-Zt-Client (what a web page sends)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 403
    req = urllib.request.Request(ctl(nodes["h1"]) + "/api/status", headers={"X-Zt-Client": "1", "Host": "evil.com"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 403


def test_probe_cannot_target_arbitrary_urls(net):
    nodes, _, _ = net
    url = f"http://127.0.0.1:{nodes['boot'].port}/probe"
    body = json.dumps({"url": f"http://127.0.0.1:{nodes['boot'].ctl_port}/api/status"}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400


def test_dht_ignores_malformed_contacts_and_records(net):
    nodes, _, _ = net
    d = nodes["boot"].dht
    n0 = len(d.contacts())
    d.add({"id": "zz" * 20, "addr": "http://x"})
    d.add({"id": "ab" * 20, "addr": "file:///etc/passwd"})
    assert len(d.contacts()) == n0
    assert d.handle({"op": "find_node", "target": "not-hex"}).get("nodes") is None
    assert d.put_local("zz", {"t": "peer"}) is False


def test_trusted_publisher_beats_sybil_majority(net, tmp_path):
    """Two colluding peers outvote one honest seeder, unless the client trusts the honest publisher's key."""
    nodes, run, tmp = net
    times = pd.date_range("2022-01-01", periods=24, freq="h")
    good = np.random.default_rng(3).random((24, 4, 5)).astype("float32")
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": make(tmp / "t_h1.zarr", good, times)})["link"]
    for n in ("h2", "evil"):  # Sybil pair with forged values
        http(ctl(nodes[n]), "POST", "/api/seed", {"path": make(tmp / f"t_{n}.zarr", good + 7, times)})
    plain = zt.open_dataset(link, ctl=ctl(nodes["client"]))
    assert np.array_equal(plain.sec.values, good + 7)  # majority wins without a trust anchor
    trusting = Node(tmp_path / "trusting", port=port(), ctl_port=port(), trust=[nodes["h1"].ident.pk],
                    bootstrap=[f"http://127.0.0.1:{nodes['boot'].port}"])
    run(trusting.start())
    try:
        ds = zt.open_dataset(link, ctl=ctl(trusting))
        np.testing.assert_array_equal(ds.sec.values, good)
    finally:
        run(trusting.stop())


def test_pushdown_point_series_is_exact_and_cheaper(net):
    nodes, run, tmp = net
    times = pd.date_range("2024-01-01", periods=96, freq="h")
    vals = np.random.default_rng(5).random((96, 4, 5)).astype("float32")
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": make(tmp / "pd.zarr", vals, times)})["link"]
    ds = zt.open_dataset(link, ctl=ctl(nodes["client"]), chunking={"time": -1, "y": 1, "x": 1})
    np.testing.assert_array_equal(ds.sec.isel(y=2, x=3).values, vals[:, 2, 3])
    st = http(ctl(nodes["client"]), "GET", "/api/status")["pushdown"]
    assert st["remote"] >= 1 and st["bytes"] <= 96 * 4 * 2  # a few hundred bytes instead of whole chunks
    assert zt.store_of(ds).stats["pushdown"] >= 1


def test_optimistic_pushdown_catches_a_lying_holder(net):
    nodes, run, tmp = net
    times = pd.date_range("2024-02-01", periods=48, freq="h")
    vals = np.random.default_rng(6).random((48, 4, 5)).astype("float32") + 1
    link = http(ctl(nodes["evil"]), "POST", "/api/seed", {"path": make(tmp / "lie.zarr", vals, times)})["link"]
    nodes["evil"].cheat = True                     # signs zeros instead of the real slice
    ds = zt.open_dataset(link, ctl=ctl(nodes["client"]), chunking={"time": -1, "y": 1, "x": 1})
    np.testing.assert_array_equal(ds.sec.isel(y=1, x=1).values, vals[:, 1, 1])  # audit replaced the lie
    st = http(ctl(nodes["client"]), "GET", "/api/status")
    assert nodes["evil"].ident.id in st["blacklist"]
    f = st["fraud"][-1]
    assert f["peer"] == nodes["evil"].ident.id and f["claimed_h"] != f["true_h"]
    # the stored receipt is a transferable proof: anyone can check the holder signed the wrong hash
    from zarr_torrent.common import verify, cjson
    body = cjson({"g": link.removeprefix("zt://"), "k": f["key"], "cid": f["cid"], "sel": f["sel"], "h": f["claimed_h"]})
    assert verify(f["pk"], f["sig"], body)


def test_closed_network_key(tmp_path):
    """Private network: members with the key see and download everything; a node or a plain HTTP client
    without it (or with a wrong key) gets 403 on every data-port path, DHT included."""
    import urllib.error
    import urllib.request
    from zarr_torrent import cli
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(60)
    bp = port()
    boot_home = tmp_path / "boot"
    cli.main(["init", "--home", str(boot_home), "--bootstrap-node", "--public-host", "127.0.0.1", "--port", str(bp),
              "--private"])
    invite = cli._config(boot_home)["network"]
    _, burl, key = cli.parse_ztnet(invite)
    assert key and invite.endswith("?k=" + key) and burl == f"http://127.0.0.1:{bp}"
    assert (boot_home / "config.toml").stat().st_mode & 0o077 == 0  # secret: owner-only
    boot = Node(boot_home, port=bp, ctl_port=port(), network_key=key, relay_server=True)
    run(boot.start())
    mk = lambda n, k, **kw: Node(tmp_path / n, port=port(), ctl_port=port(), bootstrap=[burl], network_key=k, **kw)
    a = mk("a", key, relay=burl, host="127.0.0.2")  # relay must use the bound address and carry the key
    member, outsider, wrong = mk("m", key), mk("o", None), mk("w", "nope")
    for n in (a, member, outsider, wrong):
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    vals = np.arange(48 * 6, dtype="f4").reshape(48, 2, 3)
    p_ = str(tmp_path / "d.zarr")
    xr.Dataset({"v": (("time", "y", "x"), vals)}, coords={"time": pd.date_range("2021-01-01", periods=48, freq="h"),
                                                          "y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]}).to_zarr(p_, consolidated=False)
    link = http(ctl(a), "POST", "/api/seed", {"path": p_})["link"]
    assert "/r/" in a.addr
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(member)).v.values, vals)
    for n in (outsider, wrong):
        assert run(n.find_peers(link.removeprefix("zt://"))) == {}  # cannot even look it up in the DHT
    grid = link.removeprefix("zt://")
    for path, body in (("/whoami", None), (f"/m/{grid}", None), (f"/mh/{grid}", None), ("/dht", b"{}"),
                       (f"/cb/{grid}", b"{}"), ("/relay/attach", None),
                       (f"/r/{a.ident.id}/m/{grid}", None)):
        for hdr in ({}, {"X-Zt-Net": "nope"}):
            req = urllib.request.Request(f"http://127.0.0.1:{bp}{path}",
                                         data=body, headers=hdr, method="POST" if body else "GET")
            try:
                urllib.request.urlopen(req, timeout=5)
                raise AssertionError(f"{path} served without key")
            except urllib.error.HTTPError as e:
                assert e.code == 403, (path, e.code)
    # the key never travels: a member's header is a MAC bound to method, path and time; a captured header opens
    # only its own request, and only for NET_SKEW seconds
    def get(path, hdr):
        try:
            return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{bp}{path}", headers=hdr), timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code
    now = int(time.time())
    captured = {"X-Zt-Net": f"{now}:{member._net_mac(now, 'GET', '/whoami')}"}
    assert key not in captured["X-Zt-Net"]
    assert get("/whoami", captured) == 200
    assert get(f"/mh/{grid}", captured) == 403  # replayed on another path
    old = now - 10 * 60
    assert get("/whoami", {"X-Zt-Net": f"{old}:{member._net_mac(old, 'GET', '/whoami')}"}) == 403  # expired
    assert get("/whoami", {"X-Zt-Net": key}) == 403  # the bare key is no longer a credential
    for n in (a, member, outsider, wrong, boot):
        run(n.stop())


def test_malformed_requests_get_4xx_not_500(tmp_path):
    """Data port = trust boundary: junk bodies, wrong types, traversal keys never produce a 5xx or a stream cut
    mid-body, and the node keeps serving."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(60)
    n = Node(tmp_path / "h", port=port(), ctl_port=port(), relay_server=True)
    run(n.start())
    xr.Dataset({"v": (("time", "x"), np.zeros((48, 3), "f4"))},
               coords={"time": pd.date_range("2020", periods=48, freq="h"), "x": [0., 1, 2]}).to_zarr(
        tmp_path / "d.zarr", consolidated=False)
    g = http(f"http://127.0.0.1:{n.ctl_port}", "POST", "/api/seed", {"path": str(tmp_path / "d.zarr")})["link"][5:]
    junk = [None, 1, "x", [], {}, {"keys": None}, {"keys": [1, 2]}, {"keys": ["../../etc/passwd"]}, {"items": None},
            {"items": [["k", [[0, 1]]]]}, {"items": [{"key": 5}]}, {"op": 5}, {"op": "find_value", "key": None},
            {"addr": 5}, b"\x00\xff", b"{", b"[" * 5000]
    for path in ("/dht", f"/cb/{g}", "/cb/nope", f"/qb/{g}", "/probe", "/r/" + "0" * 40 + "/cb/x"):
        for body in junk:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{n.port}{path}", data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read()  # a 200 must be a complete body (IncompleteRead would raise here)
            except urllib.error.HTTPError as e:
                assert e.code < 500, (path, body, e.code, e.read()[:200])
    assert http(f"http://127.0.0.1:{n.ctl_port}", "GET", "/api/status")["grids"][g]["chunks"] > 0
    run(n.stop())


def test_a_longer_fake_array_does_not_win_a_tie(net):
    """1 vs 1 on a growing dataset: the honest copy has 18 hours, the liar announces 24 with other values. More valid
    time steps used to win the tie on the last chunk; now the more complete value must agree with the rival on the
    rival's cells, so neither chunk is served."""
    nodes, run, tmp = net
    times = pd.date_range("2020-03-01", periods=24, freq="h")
    good = np.random.default_rng(6).random((24, 4, 5)).astype("float32")
    p1 = make(tmp / "honest.zarr", good[:18], times[:18])
    pe = make(tmp / "longer.zarr", good + 1000, times)
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": p1})["link"]
    http(ctl(nodes["evil"]), "POST", "/api/seed", {"path": pe})
    grid = link.removeprefix("zt://")
    job = wait_job(ctl(nodes["client"]), http(ctl(nodes["client"]), "POST", "/api/download",
                                              {"grid": grid, "region": {"var": "sec"}})["job"])
    assert job["done"] == 0, {kk: job.get(kk) for kk in ("total", "done", "missing", "failed")}  # chunk 0: plain tie; chunk 1: longer but inconsistent


def test_growing_copy_wins_over_its_older_prefix(net):
    """Honest growth: the newer copy has 6 more hours and agrees with the older one on the older one's hours."""
    nodes, run, tmp = net
    times = pd.date_range("2020-04-01", periods=24, freq="h")
    new = np.random.default_rng(7).random((24, 4, 5)).astype("float32")
    p_old, p_new = make(tmp / "old.zarr", new[:18], times[:18]), make(tmp / "new.zarr", new, times)
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": p_old})["link"]
    http(ctl(nodes["h2"]), "POST", "/api/seed", {"path": p_new})
    grid = link.removeprefix("zt://")
    job = wait_job(ctl(nodes["client"]), http(ctl(nodes["client"]), "POST", "/api/download",
                                              {"grid": grid, "region": {"var": "sec"}})["job"])
    assert job["done"] == 2 and job["missing"] == 0, job
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(nodes["client"])).sec.values, new)


def test_manifest_heads_reject_rollback_and_expose_equivocation(tmp_path):
    """Versioned heads: an older version is refused; two different roots under one version are kept as a
    transferable proof and the publisher drops out of every view."""
    n = Node(tmp_path / "n", port=port(), ctl_port=port())
    head = lambda seq, root: {"seq": seq, "root": root}
    assert n._head_ok("p", "g", head(5, "r5"), "pk", "s5", b"h5")
    assert not n._head_ok("p", "g", head(4, "r4"), "pk", "s4", b"h4")  # rollback
    assert n._head_ok("p", "g", head(5, "r5"), "pk", "s5", b"h5")  # same version again: fine
    assert "p" not in n.bad
    assert not n._head_ok("p", "g", head(5, "rX"), "pk", "sX", b"hX")  # equivocation
    assert "p" in n.bad and n.fraud[-1]["kind"] == "equivocation" and n.fraud[-1]["sigs"] == ["s5", "sX"]
    assert n._head_ok("q", "g", {"root": "r"}, "pk", "s", b"h")  # unversioned (older) peers still accepted


def test_publisher_equivocation_is_detected_end_to_end(net):
    nodes, run, tmp = net
    times = pd.date_range("2020-05-01", periods=24, freq="h")
    vals = np.random.default_rng(9).random((24, 4, 5)).astype("float32")
    p = make(tmp / "eq.zarr", vals, times)
    link = http(ctl(nodes["h1"]), "POST", "/api/seed", {"path": p})["link"]
    grid = link.removeprefix("zt://")
    http(ctl(nodes["client"]), "GET", f"/api/view/{grid}?refresh=1")
    seen = nodes["client"].seen_heads[(nodes["h1"].ident.id, grid)]
    h1 = nodes["h1"]
    import shutil
    make(tmp / "eq2.zarr", vals + 1, times)  # other content ...
    shutil.rmtree(p)
    shutil.copytree(tmp / "eq2.zarr", p)
    http(ctl(h1), "POST", "/api/seed", {"path": p})
    rebuild = lambda: (h1._capt.pop(grid, None), h1._capt_t.pop(grid, None), run(h1._capt_for(grid)))[2]
    root_new = rebuild()[3]
    h1._heads[grid] = {"seq": seen["seq"], "root": root_new}  # ... signed under the version the client already saw
    assert rebuild()[3] == root_new != seen["root"]
    http(ctl(nodes["client"]), "GET", f"/api/view/{grid}?refresh=1")
    assert h1.ident.id in nodes["client"].bad
