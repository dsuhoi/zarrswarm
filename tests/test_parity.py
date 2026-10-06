"""ZTP-EC: a volunteer keeps only XOR parity (1/k storage); after the only full seeder dies, chunks that no
live peer holds are restored from parity + surviving stripe members, verified and exact."""
import asyncio
import socket
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import zarr_torrent as zt
from zarr_torrent.node import Node
from zarr_torrent.store import http, open_views, wait_job
_ports = iter(range(10000, 30000))


def port():
    # Avoid collisions with outgoing benchmark sockets and ports chosen earlier in this test.
    for p in _ports:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free test port")


def test_parity_restores_chunks_nobody_holds(tmp_path):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(120)
    bp = port()
    boot = Node(tmp_path / "boot", port=bp, ctl_port=port())
    run(boot.start())
    mk = lambda n: Node(tmp_path / n, port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    a, b, vol, cli = mk("a"), mk("b"), mk("vol"), mk("cli")
    for n in (a, b, vol, cli):
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    times = pd.date_range("2021-01-01", periods=240, freq="h")
    vals = np.random.default_rng(7).random((240, 3, 4)).astype("float32")
    ds = xr.Dataset({"pv": (("time", "y", "x"), vals)}, coords={"time": times, "y": np.arange(3.0), "x": np.arange(4.0)})
    enc = {"pv": {"chunks": (24, 3, 4)}}
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    ds.to_zarr(pa, encoding=enc, consolidated=False)
    ds.to_zarr(pb, encoding=enc, consolidated=False)
    import zarr  # B lacks days 0 and 5
    zb = zarr.open_array(str(Path(pb) / "pv"), mode="r")
    for day in (0, 5):
        (Path(pb) / "pv" / zb.metadata.encode_chunk_key((day, 0, 0))).unlink()
    link = http(ctl(a), "POST", "/api/seed", {"path": pa})["link"]
    http(ctl(b), "POST", "/api/seed", {"path": pb})
    grid = link.removeprefix("zt://")
    r = http(ctl(vol), "POST", "/api/parity", {"grid": grid, "var": "pv", "k": 5, "drop": True})
    assert 2 <= r["stripes"] <= 4  # absolute alignment + interleaving (D = ceil(10/k)) may split into 3-4
    st = http(ctl(vol), "GET", "/api/status")["grids"][grid]
    assert st["chunks"] <= r["stripes"] + 2  # only parity (+ coordinates) kept
    run(a.stop())  # the only holder of days 0 and 5 disappears
    jid = http(ctl(cli), "POST", "/api/download", {"grid": grid, "region": {"var": "pv"}})["job"]
    job = wait_job(ctl(cli), jid)
    assert job.get("restored") == 2 and job["state"] == "done", job
    got = zt.open_dataset(link, ctl=ctl(cli)).pv.values
    np.testing.assert_array_equal(got, vals)
    for n in (b, vol, cli, boot):
        run(n.stop())


def test_two_losses_in_one_stripe_need_two_volunteers_rows(tmp_path):
    """RS(k, m): two volunteers with different rows rebuild TWO lost members of the same stripe."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(120)
    bp = port()
    boot = Node(tmp_path / "boot", port=bp, ctl_port=port())
    run(boot.start())
    mk = lambda n: Node(tmp_path / n, port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    a, b, v1, v2, cli = mk("a"), mk("b"), mk("v1"), mk("v2"), mk("cli")
    for n in (a, b, v1, v2, cli):
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    times = pd.date_range("2021-03-01", periods=24 * 8, freq="h")
    vals = np.random.default_rng(9).random((24 * 8, 3, 4)).astype("float32")
    ds = xr.Dataset({"rs": (("time", "y", "x"), vals)}, coords={"time": times, "y": np.arange(3.0), "x": np.arange(4.0)})
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    for p_ in (pa, pb):
        ds.to_zarr(p_, encoding={"rs": {"chunks": (24, 3, 4)}}, consolidated=False)
    import zarr
    zb = zarr.open_array(str(Path(pb) / "rs"), mode="r")
    for day in (2, 3):  # two neighbours: same consecutive stripe (d=1)
        (Path(pb) / "rs" / zb.metadata.encode_chunk_key((day, 0, 0))).unlink()
    link = http(ctl(a), "POST", "/api/seed", {"path": pa})["link"]
    http(ctl(b), "POST", "/api/seed", {"path": pb})
    grid = link.removeprefix("zt://")
    for row, vol in enumerate((v1, v2)):
        http(ctl(vol), "POST", "/api/parity", {"grid": grid, "var": "rs", "k": 4, "d": 1, "row": row, "drop": True})
    run(a.stop())
    jid = http(ctl(cli), "POST", "/api/download", {"grid": grid, "region": {"var": "rs"}})["job"]
    job = wait_job(ctl(cli), jid)
    assert job.get("restored") == 2 and job["state"] == "done", job
    np.testing.assert_array_equal(zt.open_dataset(link, ctl=ctl(cli)).rs.values, vals)
    for n in (b, v1, v2, cli, boot):
        run(n.stop())


def test_eight_rs_volunteers_rebuild_everything_after_the_only_seeder_dies(tmp_path):
    """MDS property end-to-end: 8 volunteers x 1/8 storage (distinct Cauchy rows, interleaved stripes)
    replace the only full copy."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(120)
    bp = port()
    boot = Node(tmp_path / "boot", port=bp, ctl_port=port())
    run(boot.start())
    mk = lambda n: Node(tmp_path / n, port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    seed, cli, vols = mk("seed"), mk("cli"), [mk(f"v{j}") for j in range(8)]
    for n in [seed, cli] + vols:
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    times = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
    vals = np.random.default_rng(1).random((len(times), 6, 8)).astype("float32")
    p = str(tmp_path / "d.zarr")
    xr.Dataset({"x": (("time", "y", "z"), vals)}, coords={"time": times, "y": np.arange(6.0), "z": np.arange(8.0)}
               ).to_zarr(p, encoding={"x": {"chunks": (24, 6, 8)}}, consolidated=False)
    link = http(ctl(seed), "POST", "/api/seed", {"path": p})["link"]
    grid = link.removeprefix("zt://")
    for j, v in enumerate(vols):
        http(ctl(v), "POST", "/api/parity", {"grid": grid, "var": "x", "k": 8, "row": j, "drop": True})
    run(seed.stop())
    job = wait_job(ctl(cli), http(ctl(cli), "POST", "/api/download", {"grid": grid, "region": {"var": "x"}})["job"])
    assert job["state"] == "done" and job.get("restored") == 30, job
    got = zt.open_dataset(link, ctl=ctl(cli)).x.sel(time=slice(times[0], times[-1])).values
    np.testing.assert_array_equal(got, vals)
    for n in [cli, boot] + vols:
        run(n.stop())


def test_value_parity_spans_heterogeneous_layouts(tmp_path):
    """Value-level stripes: members are canonical chunks assembled from ANY layout, so replicas chunked
    differently still protect each other. A (24 h chunks) dies; B (48 h chunks) lacks days 4-5; value parity
    held by volunteers rebuilds days 4-5 from B's other chunks."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(180)
    bp = port()
    boot = Node(tmp_path / "boot", port=bp, ctl_port=port())
    run(boot.start())
    mk = lambda n: Node(tmp_path / n, port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    a, b, cli, vols = mk("a"), mk("b"), mk("cli"), [mk(f"v{j}") for j in range(2)]
    for n in [a, b, cli] + vols:
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    times = pd.date_range("2021-05-01", periods=24 * 10, freq="h")
    vals = np.random.default_rng(11).random((240, 3, 4)).astype("float32")
    ds = xr.Dataset({"hv": (("time", "y", "x"), vals)}, coords={"time": times, "y": np.arange(3.0), "x": np.arange(4.0)})
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    ds.to_zarr(pa, encoding={"hv": {"chunks": (24, 3, 4)}}, consolidated=False)
    ds.to_zarr(pb, encoding={"hv": {"chunks": (48, 3, 4)}}, consolidated=False)
    import zarr
    zb = zarr.open_array(str(Path(pb) / "hv"), mode="r")
    (Path(pb) / "hv" / zb.metadata.encode_chunk_key((2, 0, 0))).unlink()  # days 4-5 only in A
    link = http(ctl(a), "POST", "/api/seed", {"path": pa})["link"]
    http(ctl(b), "POST", "/api/seed", {"path": pb})
    grid = link.removeprefix("zt://")
    rows = set()
    for vol in vols:  # no explicit row: each volunteer must pick one the swarm does not publish yet
        r = http(ctl(vol), "POST", "/api/parity", {"grid": grid, "var": "hv", "k": 5, "drop": True})
        assert r["kind"] == "v"
        rows.add(r["row"])
    assert len(rows) == len(vols), rows
    run(a.stop())
    job = wait_job(ctl(cli), http(ctl(cli), "POST", "/api/download", {"grid": grid, "region": {"var": "hv"}})["job"])
    assert job["state"] == "done" and job.get("restored", 0) >= 2, job
    got = zt.open_dataset(link, ctl=ctl(cli)).hv.sel(time=slice(times[0], times[-1])).values
    np.testing.assert_array_equal(got, vals)
    for n in [b, cli, boot] + vols:
        run(n.stop())


def test_value_parity_repairs_with_members_from_another_decoder(tmp_path):
    """Two providers decoded the same integer codes differently (B is off by a code-dependent fraction of a step:
    no bit in common, same lattice codes). A dies and days 4-5 existed only at A. The volunteers' value stripes are
    coded over lattice codes, so B's members (other bits) still rebuild days 4-5; with raw float bytes the algebra
    produced garbage that the vcid check rejected."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(180)
    bp = port()
    boot = Node(tmp_path / "boot", port=bp, ctl_port=port())
    run(boot.start())
    mk = lambda n: Node(tmp_path / n, port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    a, b, cli, vols = mk("a"), mk("b"), mk("cli"), [mk(f"v{j}") for j in range(2)]
    for n in [a, b, cli] + vols:
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    times = pd.date_range("2021-05-01", periods=24 * 10, freq="h")
    step = 2.0 ** -9
    k = np.random.default_rng(5).integers(0, 4000, (240, 30, 40))  # tiles of 1200 values: a lattice is estimable
    va = (250 + k * step).astype("float32")
    vb = (250 + k * step + np.where(k % 3 == 0, 0.27, -0.13) * step).astype("float32")
    assert not np.any(va == vb)
    mkds = lambda x: xr.Dataset({"hv": (("time", "y", "x"), x)},
                                coords={"time": times, "y": np.arange(30.0), "x": np.arange(40.0)})
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    mkds(va).to_zarr(pa, encoding={"hv": {"chunks": (24, 30, 40)}}, consolidated=False)
    mkds(vb).to_zarr(pb, encoding={"hv": {"chunks": (48, 30, 40)}}, consolidated=False)
    import zarr
    zb = zarr.open_array(str(Path(pb) / "hv"), mode="r")
    (Path(pb) / "hv" / zb.metadata.encode_chunk_key((2, 0, 0))).unlink()  # days 4-5 only at A
    link = http(ctl(a), "POST", "/api/seed", {"path": pa})["link"]
    assert http(ctl(b), "POST", "/api/seed", {"path": pb})["link"] == link
    grid = link.removeprefix("zt://")
    for vol in vols:
        r = http(ctl(vol), "POST", "/api/parity", {"grid": grid, "var": "hv", "k": 5, "drop": True})
        assert r["code_stripes"] == r["stripes"] > 0, r
    run(a.stop())
    job = wait_job(ctl(cli), http(ctl(cli), "POST", "/api/download", {"grid": grid, "region": {"var": "hv"}})["job"])
    assert job["state"] == "done" and job.get("restored", 0) >= 2, repr({kk: job.get(kk) for kk in ("state", "total", "done", "missing", "failed", "restored")}) + repr(cli.pd_stats.get("restore_fail"))
    got = zt.open_dataset(link, ctl=ctl(cli)).hv.sel(time=slice(times[0], times[-1])).values
    assert np.all(np.abs(got.astype("f8") - va) < step / 2)  # one faithful decoding of the same codes
    for n in [b, cli, boot] + vols:
        run(n.stop())
