"""E7: optimistic pushdown vs whole-chunk transfer for point / box / full-map queries over map-chunked data.
Seeders emulate a 5 MB/s uplink. python bench/bench_pushdown.py [--days 120]"""
import argparse
import asyncio
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarr_torrent as zt  # noqa: E402
from zarr_torrent.node import Node  # noqa: E402
from zarr_torrent.store import http  # noqa: E402


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--rate", type=float, default=5e6)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ztpd", dir=Path("~/.cache").expanduser()))
    times = pd.date_range("2020-01-01", periods=24 * a.days, freq="h")
    lat, lon = np.linspace(-90, 90, 91), np.arange(0, 360, 2.0)
    h = np.arange(len(times))[:, None, None]
    f = (280 + 20 * np.cos(np.deg2rad(lat))[None, :, None] + 5 * np.sin(h / 24 * 6.28 + np.deg2rad(lon)[None, None, :])
         + np.random.default_rng(0).normal(0, 0.5, (len(times), 91, 180))).astype("float32")
    p = str(tmp / "maps.zarr")
    xr.Dataset({"t2m": (("time", "lat", "lon"), f)}, coords={"time": times, "lat": lat, "lon": lon}).to_zarr(
        p, encoding={"t2m": {"chunks": (24, 91, 180)}}, consolidated=False)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(600)
    bp = port()
    seeds = []
    for i in range(2):
        n = Node(tmp / f"s{i}", port=bp if i == 0 else port(), ctl_port=port(), rate=a.rate,
                 bootstrap=[] if i == 0 else [f"http://127.0.0.1:{bp}"])
        run(n.start())
        seeds.append(n)
    link = None
    for n in seeds:
        link = http(f"http://127.0.0.1:{n.ctl_port}", "POST", "/api/seed", {"path": p})["link"]
    res = {"days": a.days, "chunk_MB_raw": round(24 * 91 * 180 * 4 / 1e6, 2), "seed_uplink_MBps": a.rate / 1e6}
    queries = {"point": dict(lat=slice(40, 41), lon=slice(30, 31)),
               "box10x10": dict(lat=slice(40, 58), lon=slice(30, 48)),
               "full_map_1day": None}
    for q, sel in queries.items():
        for mode in ("whole_chunks", "pushdown", "pushdown_trusted"):
            trust = [n.ident.pk for n in seeds] if mode == "pushdown_trusted" else []
            c = Node(tmp / f"c_{q}_{mode}", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"],
                     trust=trust)
            run(c.start())
            ctl = f"http://127.0.0.1:{c.ctl_port}"
            chunking = {"time": -1, "lat": 1, "lon": 1} if sel is not None else None
            ds = zt.open_dataset(link, ctl=ctl, chunking=chunking, pushdown=mode.startswith("pushdown"))
            t0 = time.perf_counter()
            if sel is None:
                v = ds.t2m.sel(time="2020-02-01").values
                ref = f[31 * 24:32 * 24]
            else:
                v = ds.t2m.sel(**sel).values
                li = np.nonzero((lat >= sel["lat"].start) & (lat <= sel["lat"].stop))[0]
                lo = np.nonzero((lon >= sel["lon"].start) & (lon <= sel["lon"].stop))[0]
                ref = f[:, li][:, :, lo]
            dt = time.perf_counter() - t0
            st = http(ctl, "GET", "/api/status")
            chunk_bytes = sum(j["bytes"] for j in st["jobs"]) if st["jobs"] else 0
            moved = sum(st["served"].values()) if False else None
            got_bytes = st["pushdown"]["bytes"] + sum(x["bytes"] for x in st["grids"].values())
            res[f"{q}/{mode}"] = {"s": round(dt, 2), "MB_in": round(got_bytes / 1e6, 2),
                                  "audits": st["pushdown"]["audits"], "exact": bool(np.array_equal(v, ref))}
            print(q, mode, res[f"{q}/{mode}"], flush=True)
            run(c.stop())
    print(json.dumps(res, indent=1))
    Path("bench/results_pushdown.json").write_text(json.dumps(res, indent=1))
    for n in seeds:
        run(n.stop())


if __name__ == "__main__":
    main()
