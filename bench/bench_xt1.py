"""E8: value-level transport codec (xt1) on a slow link, real ERA5 (WeatherBench2 replica R1).
python bench/bench_xt1.py [--rate 2e6]"""
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
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarrswarm as zt  # noqa: E402
from zarrswarm.node import Node  # noqa: E402
from zarrswarm.store import http, open_views, wait_job  # noqa: E402


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=2e6)
    ap.add_argument("--replica", default="~/.cache/zt_real/wb2/wb2_R1_v3.zarr")
    a = ap.parse_args()
    rep = str(Path(a.replica).expanduser())
    tmp = Path(tempfile.mkdtemp(prefix="ztxt", dir=Path("~/.cache").expanduser()))
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(900)
    bp = port()
    seed = Node(tmp / "seed", port=bp, ctl_port=port(), rate=a.rate)
    run(seed.start())
    link = http(f"http://127.0.0.1:{seed.ctl_port}", "POST", "/api/seed", {"path": rep})["link"]
    src = xr.open_zarr(rep, consolidated=False)
    res = {"rate_MBps": a.rate / 1e6}
    for mode in ("plain", "xt1"):
        c = Node(tmp / f"c_{mode}", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
        c.xt1 = mode == "xt1"
        run(c.start())
        ctl = f"http://127.0.0.1:{c.ctl_port}"
        t0 = time.perf_counter()
        wire = 0
        for grid, view in open_views(link, ctl):
            for var in [n for n in view.arrays if n not in view.v["grid"]["dims"]]:
                jid = http(ctl, "POST", "/api/download", {"grid": grid, "region": {"var": var}})["job"]
                job = wait_job(ctl, jid, every=0.2)
                wire += job["bytes"]
        dt = time.perf_counter() - t0
        ds = zt.open_dataset(link, ctl=ctl)
        exact = all(np.array_equal(ds[v].sel(time=src.time).transpose(*src[v].dims).values, src[v].values)
                    for v in src.data_vars)
        res[mode] = {"s": round(dt, 1), "wire_MB": round(wire / 1e6, 1), "exact": exact,
                     "xt1_chunks_served": seed.served.get("xt1_chunks", 0)}
        print(mode, res[mode], flush=True)
        run(c.stop())
    res["speedup"] = round(res["plain"]["s"] / res["xt1"]["s"], 2)
    print(json.dumps(res, indent=1))
    Path("bench/results_xt1.json").write_text(json.dumps(res, indent=1))
    run(seed.stop())


if __name__ == "__main__":
    main()
