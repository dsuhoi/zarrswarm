"""Localhost benchmark: software overhead of the swarm path vs reading zarr from local disk.

3 seeders (A: Jan-Aug, B: May-Dec, C: whole year in another chunk layout) + a fresh client.
Usage: python bench/bench_local.py [--days 365] [--ny 64] [--nx 128]
"""
import argparse
import asyncio
import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import zarrswarm as zs
from zarrswarm.node import Node
from zarrswarm.store import http


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def dataset(times, ny, nx):
    s = times.values.astype("datetime64[h]").astype("int64").astype("float32")
    y, x = np.meshgrid(np.linspace(0, 3, ny, dtype="float32"), np.linspace(0, 6, nx, dtype="float32"), indexing="ij")
    t2m = 280 + 10 * np.sin(s[:, None, None] / 24 / 58) * np.cos(y)[None] + 3 * np.sin(s[:, None, None] / 24 * 6.28 + x[None])
    return xr.Dataset({"t2m": (("time", "lat", "lon"), t2m.astype("float32"))},
                      coords={"time": times, "lat": np.linspace(-80, 80, ny), "lon": np.linspace(0, 359, nx)})


def timed(label, fn, out):
    t0 = time.perf_counter()
    r = fn()
    out[label] = round(time.perf_counter() - t0, 3)
    print(f"{label:<42} {out[label]:8.3f} s", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nx", type=int, default=128)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ztbench"))
    times = pd.date_range("2020-01-01", periods=24 * a.days, freq="h")
    full = dataset(times, a.ny, a.nx)
    nb = full.t2m.nbytes
    print(f"dataset {full.t2m.shape} float32 = {nb / 1e6:.0f} MB raw  ({tmp})")
    res = {"raw_MB": round(nb / 1e6, 1)}
    cut1, cut2 = int(len(times) * 8 / 12) // 24 * 24, int(len(times) * 4 / 12) // 24 * 24
    pa, pb, pc = str(tmp / "a.zarr"), str(tmp / "b.zarr"), str(tmp / "c.zarr")
    timed("write replicas", lambda: (
        full.isel(time=slice(0, cut1)).to_zarr(pa, encoding={"t2m": {"chunks": (24, a.ny, a.nx)}}, consolidated=False),
        full.isel(time=slice(cut2, None)).to_zarr(pb, encoding={"t2m": {"chunks": (24, a.ny, a.nx)}}, consolidated=False),
        full.to_zarr(pc, encoding={"t2m": {"chunks": (168, a.ny // 2, a.nx // 2)}}, consolidated=False)), res)
    disk = sum(f.stat().st_size for f in Path(pa).rglob("*") if f.is_file())
    res["replica_A_disk_MB"] = round(disk / 1e6, 1)

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(600)
    bp = port()
    boot = f"http://127.0.0.1:{bp}"
    nodes = {"boot": Node(tmp / "h_boot", port=bp, ctl_port=port(), relay_server=True)}
    run(nodes["boot"].start())
    for n in ("a", "b", "c", "client"):
        nodes[n] = Node(tmp / f"h_{n}", port=port(), ctl_port=port(), bootstrap=[boot])
        run(nodes[n].start())
    ctl = lambda n: f"http://127.0.0.1:{nodes[n].ctl_port}"
    link = None
    for n, p in (("a", pa), ("b", pb), ("c", pc)):
        link = timed(f"scan+seed {n}", lambda: http(ctl(n), "POST", "/api/seed", {"path": p})["link"], res)

    timed("baseline: xr.open_zarr(A, local disk).load()", lambda: xr.open_zarr(pa, consolidated=False).t2m.load(), res)
    ds = timed("zs.open_dataset (DHT lookup + 3 manifests)", lambda: zs.open_dataset(link, ctl=ctl("client")), res)
    v = timed("swarm: full t2m load (cold, 3 peers)", lambda: ds.t2m.load(), res)
    assert np.array_equal(v.values, full.t2m.values)
    res["swarm_cold_MBps_raw"] = round(nb / 1e6 / res["swarm: full t2m load (cold, 3 peers)"], 1)
    job = [j for j in http(ctl("client"), "GET", "/api/status")["jobs"]]
    st = http(ctl("client"), "GET", "/api/status")
    res["client_bw_est_MBps"] = {p[:8]: round(b / 1e6, 1) for p, b in st["bw"].items()}
    ds2 = zs.open_dataset(link, ctl=ctl("client"))
    timed("swarm: full t2m load (warm cache)", lambda: ds2.t2m.load(), res)
    ts = zs.open_dataset(link, ctl=ctl("client"), chunking={"time": -1, "lat": 1, "lon": 1})
    pt = timed("re-chunked view: point time series (warm)", lambda: ts.t2m.isel(lat=10, lon=20).load(), res)
    assert np.array_equal(pt.values, full.t2m.isel(lat=10, lon=20).values)
    # fresh client for prefetch / progressive
    fresh = Node(tmp / "h_fresh", port=port(), ctl_port=port(), bootstrap=[boot])
    run(fresh.start())
    fctl = f"http://127.0.0.1:{fresh.ctl_port}"
    ds3 = zs.open_dataset(link, ctl=fctl)
    r = timed("progressive mean to 1% (cold)", lambda: zs.progressive_mean(ds3, "t2m", rel_err=0.01), res)
    res["progressive"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    res["true_mean"] = float(full.t2m.mean())
    fresh2 = Node(tmp / "h_fresh2", port=port(), ctl_port=port(), bootstrap=[boot])
    run(fresh2.start())
    ds4 = zs.open_dataset(link, ctl=f"http://127.0.0.1:{fresh2.ctl_port}")
    timed("prefetch whole t2m (cold, one plan)", lambda: zs.prefetch(ds4.t2m), res)
    print(json.dumps(res, indent=1))
    for n in list(nodes.values()) + [fresh, fresh2]:
        run(n.stop())


if __name__ == "__main__":
    main()
