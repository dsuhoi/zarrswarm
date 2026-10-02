"""Operational features: growing (appended) datasets are picked up automatically."""
import asyncio
import socket
import threading
import time

import numpy as np
import pandas as pd
import xarray as xr

import zarr_torrent as zt
from zarr_torrent import node as nodemod
from zarr_torrent.node import Node
from zarr_torrent.scan import split_key
from zarr_torrent.store import http


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def swarm(tmp_path, n_clients=1):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(60)
    bp = port()
    seed = Node(tmp_path / "s", port=bp, ctl_port=port(), relay_server=True)
    run(seed.start())
    clients = [Node(tmp_path / f"c{i}", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
               for i in range(n_clients)]
    for c in clients:
        run(c.start())
    return run, seed, clients


def feed(times, seed_=0):
    v = np.random.default_rng(seed_).random((len(times), 3, 4)).astype("f4")
    return xr.Dataset({"t2m": (("time", "y", "x"), v)}, coords={"time": times, "y": np.arange(3.0), "x": np.arange(4.0)})


def wait(cond, secs=20):
    t = time.time()
    while time.time() - t < secs:
        if cond():
            return True
        time.sleep(0.2)
    return False


def test_appended_time_steps_are_picked_up(tmp_path, monkeypatch):
    monkeypatch.setattr(nodemod, "RESCAN_EVERY", 0.2)
    run, seed, (cli,) = swarm(tmp_path)
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    p = tmp_path / "feed.zarr"
    day1 = feed(pd.date_range("2024-01-01", periods=24, freq="h"), 1)
    day1.to_zarr(p, encoding={"t2m": {"chunks": (6, 3, 4)}}, consolidated=False)
    link = http(ctl(seed), "POST", "/api/seed", {"path": str(p)})["link"]
    assert zt.open_dataset(link, ctl=ctl(cli)).sizes["time"] == 24
    day2 = feed(pd.date_range("2024-01-02", periods=24, freq="h"), 2)
    day2.to_zarr(p, append_dim="time", consolidated=False)  # the operational feed grows
    grid = link.removeprefix("zt://")
    assert wait(lambda: http(ctl(cli), "GET", f"/api/view/{grid}?refresh=1")["gmax"] -
                http(ctl(cli), "GET", f"/api/view/{grid}")["gmin"] == 48)
    got = zt.open_dataset(link, ctl=ctl(cli)).t2m.values
    np.testing.assert_array_equal(got, np.concatenate([day1.t2m.values, day2.t2m.values]))
    # a chunk rewritten in place (preliminary -> final values), metadata untouched: the periodic full rescan
    grid_chunks = lambda: next(sd["chunks"] for (pp, _), sd in seed.seeds.items() if pp == str(p.resolve()))
    k0 = min((k for k in grid_chunks() if k.startswith("t2m@")), key=lambda k: split_key(k)[2])  # first 6 h
    old_cid = grid_chunks()[k0][0]
    fixed = day1.t2m.values.copy()
    fixed[:6] += 100
    day1.assign(t2m=(("time", "y", "x"), fixed)).isel(time=slice(0, 6)).drop_vars(["y", "x"]).to_zarr(
        p, region={"time": slice(0, 6)}, consolidated=False)
    assert grid_chunks()[k0][0] == old_cid  # metadata unchanged: the cheap check alone does not see it
    assert wait(lambda: grid_chunks()[k0][0] != old_cid)  # ... the 10th-round full rescan does
    for n in (cli, seed):
        run(n.stop())


def test_follow_keeps_newest_window_of_a_growing_feed(tmp_path, monkeypatch):
    """Subscription on a name: the newest 12 h are downloaded; when the feed grows (and the name is re-pointed
    at nothing new - same grid), the next catch-up fetches exactly the new window, nothing older."""
    monkeypatch.setattr(nodemod, "RESCAN_EVERY", 0.2)
    monkeypatch.setattr(nodemod, "FOLLOW_EVERY", 0.3)
    run, seed, (cli,) = swarm(tmp_path)
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    p = tmp_path / "feed.zarr"
    d1 = feed(pd.date_range("2024-01-01", periods=48, freq="h"), 1)
    d1.to_zarr(p, encoding={"t2m": {"chunks": (6, 3, 4)}}, consolidated=False)
    link = http(ctl(seed), "POST", "/api/seed", {"path": str(p)})["link"]
    name = http(ctl(seed), "POST", "/api/name", {"name": "feed", "target": link})["link"]
    grid = link.removeprefix("zt://")
    r = http(ctl(cli), "POST", "/api/follow", {"link": name, "vars": ["t2m"], "last_s": 12 * 3600})
    assert r["jobs"]
    have = lambda: {k for k in cli.local.get(grid, {"files": {}})["files"] if k.startswith("t2m@")}
    assert wait(lambda: len(have()) == 2)  # 12 h of 6-hourly chunks = the newest two
    first = set(have())
    d2 = feed(pd.date_range("2024-01-03", periods=12, freq="h"), 2)
    d2.to_zarr(p, append_dim="time", consolidated=False)
    assert wait(lambda: len(have()) == 4, 30)  # the new 12 h arrive; nothing older is fetched
    assert first < have()
    got = zt.open_dataset(link, ctl=ctl(cli)).t2m.sel(time=slice("2024-01-03", None)).values
    np.testing.assert_array_equal(got, d2.t2m.values)
    assert [s["id"] for s in http(ctl(cli), "GET", "/api/follows")] == [r["id"]]
    run(cli.stop())  # subscriptions survive a restart
    cli2 = Node(tmp_path / "c0", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{seed.port}"])
    run(cli2.start())
    assert list(cli2.subs) == [r["id"]]
    assert http(ctl(cli2), "POST", "/api/unfollow", {"id": r["id"]})["ok"] and not cli2.subs
    for n in (cli2, seed):
        run(n.stop())


def test_cache_limit_evicts_least_recently_used(tmp_path):
    """Downloaded chunks over the limit are evicted LRU (down to 90 %); recently used ones and the user's own
    seeded files stay; evicted data is simply downloaded again when needed."""
    run, seed, (cli,) = swarm(tmp_path)
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    p = tmp_path / "d.zarr"
    d = feed(pd.date_range("2024-01-01", periods=96, freq="h"), 3)
    d.to_zarr(p, encoding={"t2m": {"chunks": (12, 3, 4)}}, consolidated=False)
    link = http(ctl(seed), "POST", "/api/seed", {"path": str(p)})["link"]
    grid = link.removeprefix("zt://")
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(cli)).t2m.values, d.t2m.values)  # all 8 chunks
    keys = sorted((k for k in cli.caches[grid]["chunks"] if k.startswith("t2m@")),
                  key=lambda k: split_key(k)[2])
    size = sum(e[2] for e in cli.caches[grid]["chunks"].values())
    now = time.time()
    for i, k in enumerate(keys):  # day order = use order: the oldest days are least recently used
        cli.used[(grid, k)] = now - 10_000 + i
    async def on_loop(f, *a, **kw):  # evict mutates node state: run it on the node's loop, like _evict_loop
        return f(*a, **kw)
    freed = run(on_loop(cli.evict, int(size * 0.6), keep_recent_s=60))
    left = [k for k in keys if k in cli.caches[grid]["chunks"]]
    assert freed > 0 and left == keys[len(keys) - len(left):], left  # oldest went first
    assert sum(e[2] for e in cli.caches[grid]["chunks"].values()) <= 0.6 * size
    assert all(cli._cas(cli.local[grid]["chunks"][k][0]).exists() for k in left)
    assert not any(k in cli.local[grid]["files"] for k in keys if k not in left)
    assert run(on_loop(seed.evict, 0)) == 0 and p.exists()  # the seeder's own files are not a cache
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(cli)).t2m.values, d.t2m.values)  # re-downloaded
    for n in (cli, seed):
        run(n.stop())


def test_pause_and_resume_download(tmp_path):
    """Pause stops a running job after the current batches (fetched chunks stay); resume restarts the same job
    id, counting what is already here as done, and finishes with exact values."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(60)
    bp = port()
    seed = Node(tmp_path / "s", port=bp, ctl_port=port(), rate=300e3)  # slow uplink: the job takes seconds
    run(seed.start())
    cli = Node(tmp_path / "c", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    run(cli.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    v = np.random.default_rng(5).random((24 * 16, 32, 32)).astype("f4")  # 16 chunks x 96 KB, incompressible
    ds = xr.Dataset({"t2m": (("time", "y", "x"), v)},
                    coords={"time": pd.date_range("2024-01-01", periods=len(v), freq="h"),
                            "y": np.arange(32.0), "x": np.arange(32.0)})
    ds.to_zarr(tmp_path / "d.zarr", encoding={"t2m": {"chunks": (24, 32, 32), "compressors": None}},
               consolidated=False)
    link = http(ctl(seed), "POST", "/api/seed", {"path": str(tmp_path / "d.zarr")})["link"]
    grid = link.removeprefix("zt://")
    jid = http(ctl(cli), "POST", "/api/download", {"grid": grid, "region": {"var": "t2m"}})["job"]
    job = lambda: http(ctl(cli), "GET", f"/api/job/{jid}")
    assert wait(lambda: job()["done"] >= 2)
    http(ctl(cli), "POST", f"/api/pause/{jid}")
    assert wait(lambda: job()["state"] == "paused"), {k: job()[k] for k in ("state", "done", "total", "bytes", "probed") if k in job()}
    done_at_pause = job()["done"]
    time.sleep(1.0)
    assert job()["done"] == done_at_pause < job()["total"]  # nothing moves while paused
    http(ctl(cli), "POST", f"/api/resume/{jid}")
    assert wait(lambda: job()["state"] == "done", 60), job()
    j = job()
    assert j["done"] == j["total"] == 16 and j["bytes"] < 16 * 24 * 32 * 32 * 4 * 1.5  # no full re-download
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(cli)).t2m.values, v)
    for n in (cli, seed):
        run(n.stop())


def test_replicas_with_different_time_steps_form_one_swarm(tmp_path):
    """Hourly and 6-hourly copies of one field (different chunking, format, extent) share ONE link. A 6-hourly
    view is complete over the union (hourly replica subsampled where it is the only holder); the default
    hourly view takes hourly data where it exists and the 6-hourly samples elsewhere; a 6 h region download
    uses both replicas; every value matches the source field."""
    run, seed, (b, cli) = swarm(tmp_path, 2)
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    hours = pd.date_range("2024-01-01", "2024-01-06T23:00", freq="h")
    truth = feed(hours, 7)  # hourly ground truth
    r1 = truth.sel(time=slice("2024-01-01", "2024-01-03"))  # hourly, days 1-3, zarr v3, daily chunks
    r2 = truth.sel(time=slice("2024-01-02", None)).isel(time=slice(None, None, 6))  # 6-hourly, days 2-6, v2
    r1.to_zarr(tmp_path / "h.zarr", encoding={"t2m": {"chunks": (24, 3, 4)}}, consolidated=False, zarr_format=3)
    r2.to_zarr(tmp_path / "s.zarr", encoding={"t2m": {"chunks": (4, 3, 2)}}, consolidated=False, zarr_format=2)
    l1 = http(ctl(seed), "POST", "/api/seed", {"path": str(tmp_path / "h.zarr")})["link"]
    l2 = http(ctl(b), "POST", "/api/seed", {"path": str(tmp_path / "s.zarr")})["link"]
    assert l1 == l2  # one identity: time step is a layout property
    six = zt.open_dataset(l1, ctl=ctl(cli), step="6h")
    want6 = truth.isel(time=slice(None, None, 6))
    np.testing.assert_array_equal(six.time.values, want6.time.values)
    np.testing.assert_array_equal(six.t2m.values, want6.t2m.values)
    hourly = zt.open_dataset(l1, ctl=ctl(cli))
    np.testing.assert_array_equal(hourly.time.values, truth.time.values[:len(hourly.time)])
    got = hourly.t2m.sel(time=slice("2024-01-01", "2024-01-03")).values
    np.testing.assert_array_equal(got, r1.t2m.values)
    late = hourly.t2m.sel(time=slice("2024-01-04", None))
    have = late.dropna("time", how="all")  # only the 6-hourly samples exist after day 3
    np.testing.assert_array_equal(have.time.values, r2.time.values[r2.time.values >= np.datetime64("2024-01-04")])
    np.testing.assert_array_equal(have.values, truth.t2m.sel(time=have.time).values)
    # region download at 6 h over the union: the planner may use both replicas
    grid = l1.removeprefix("zt://")
    j = http(ctl(cli), "POST", "/api/download", {"grid": grid, "region": {"var": "t2m", "step": 21600}})["job"]
    assert wait(lambda: http(ctl(cli), "GET", f"/api/job/{j}")["state"] == "done")
    job = http(ctl(cli), "GET", f"/api/job/{j}")
    assert job["missing"] == 0 and job["cover"]["lattice"] == [6, 0], job
    for n in (cli, b, seed):
        run(n.stop())


def test_identity_modes_values_vs_bytes(tmp_path, monkeypatch):
    """Ablation used in the evaluation. Value identity: same values in different encodings (chunks, codec) and
    at different time steps share one grid. Byte identity (IPFS/BitTorrent-like): only identical encodings do -
    including single-step chunks of hourly and 6-hourly copies, whose bytes are identical."""
    from zarr_torrent import scan as scanmod
    hours = pd.date_range("2024-01-01", periods=48, freq="h")
    d = feed(hours, 9)
    d.to_zarr(tmp_path / "a.zarr", encoding={"t2m": {"chunks": (1, 3, 4)}}, consolidated=False, zarr_format=2)
    d.isel(time=slice(None, None, 6)).to_zarr(tmp_path / "b.zarr", encoding={"t2m": {"chunks": (1, 3, 4)}},
                                              consolidated=False, zarr_format=2)  # same bytes per step, 6 h
    d.to_zarr(tmp_path / "c.zarr", encoding={"t2m": {"chunks": (24, 3, 2)}}, consolidated=False, zarr_format=3)
    gid = lambda p: {g for g, sg in scanmod.scan(p)["subgrids"].items() if "t2m" in sg["arrays"]}
    monkeypatch.setattr(scanmod, "BYTE_IDENTITY", False)
    assert gid(tmp_path / "a.zarr") == gid(tmp_path / "b.zarr") == gid(tmp_path / "c.zarr")
    monkeypatch.setattr(scanmod, "BYTE_IDENTITY", True)
    assert gid(tmp_path / "a.zarr") == gid(tmp_path / "b.zarr") != gid(tmp_path / "c.zarr")


def test_int_replicas_v2_v3_merge_without_masking_zeros(tmp_path):
    """int16 replicas written as zarr v2 (fill None) and v3 (mandatory fill 0) are one value family, and a view
    assembled from the v3 layout does not turn every zero into a missing value."""
    run, seed, (b, cli) = swarm(tmp_path, 2)
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    t = pd.date_range("2000-01-01", periods=20, freq="2s")
    v = np.random.default_rng(3).integers(0, 4, (20, 4, 6)).astype("i2")  # many zeros
    ds = xr.Dataset({"bold": (("time", "z", "x"), v)}, coords={"time": t, "z": np.arange(4.0), "x": np.arange(6.0)})
    ds.isel(time=slice(0, 12)).to_zarr(tmp_path / "a.zarr", encoding={"bold": {"chunks": (1, 4, 6)}}, zarr_format=2,
                                      consolidated=False)
    ds.to_zarr(tmp_path / "b.zarr", encoding={"bold": {"chunks": (20, 2, 3)}}, zarr_format=3, consolidated=False)
    l1 = http(ctl(seed), "POST", "/api/seed", {"path": str(tmp_path / "a.zarr")})["link"]
    l2 = http(ctl(b), "POST", "/api/seed", {"path": str(tmp_path / "b.zarr")})["link"]
    assert l1 == l2
    got = zt.open_dataset(l1, ctl=ctl(cli)).bold
    assert got.sizes["time"] == 20  # 2 s step recovered from single-volume chunks, not the 1 s quantum
    np.testing.assert_array_equal(got.values, v)
    for n in (cli, b, seed):
        run(n.stop())


def test_stale_cached_family_is_not_served_for_a_new_view(tmp_path):
    """The client's cache holds a variable in one value family (float64, from an earlier swarm); the swarm now
    serves the same grid as float32. Reads must return the new values, never the cached bytes decoded with the new
    dtype (seen on a rerun: 'cannot reshape array of size 720 into shape (1440,)')."""
    run, seed, (cli, s2) = swarm(tmp_path, 2)
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    times = pd.date_range("2024-01-01", periods=12, freq="h")
    old = feed(times, 1)
    old = old.assign(t2m=old.t2m.astype("f8"))
    p1 = tmp_path / "old.zarr"
    old.to_zarr(p1, encoding={"t2m": {"chunks": (6, 3, 4)}}, consolidated=False)
    link = http(ctl(seed), "POST", "/api/seed", {"path": str(p1)})["link"]
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(cli)).t2m.values, old.t2m.values)  # now cached
    new = feed(times, 2)  # same grid, other values, float32
    p2 = tmp_path / "new.zarr"
    new.to_zarr(p2, encoding={"t2m": {"chunks": (6, 3, 4)}}, consolidated=False)
    http(ctl(seed), "POST", "/api/unseed", {"path": str(p1)})
    link2 = http(ctl(seed), "POST", "/api/seed", {"path": str(p2)})["link"]
    assert link2 == link
    http(ctl(s2), "POST", "/api/seed", {"path": str(p2)})  # two holders of the new family outvote the stale cache
    grid = link.removeprefix("zt://")
    assert wait(lambda: http(ctl(cli), "GET", f"/api/view/{grid}?refresh=1")["arrays"]["t2m"]["vfid"]
                != next(iter(cli.caches.values()))["arrays"]["t2m"]["vfid"])
    got = zt.open_dataset(link, ctl=ctl(cli)).t2m.values
    np.testing.assert_array_equal(got, new.t2m.values)
    for n in (cli, seed, s2):
        run(n.stop())
