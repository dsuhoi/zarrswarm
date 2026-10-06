"""VAS (stratified + sequential Neyman) vs low-discrepancy ratio estimator: chunks needed to reach rel_err,
and empirical CI coverage over trials. Heteroscedastic data: calm background + a stormy period.

python bench/bench_vas.py [--trials 10] [--rel-err 0.0005]
"""
import argparse
import asyncio
import json
import socket
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarrswarm as zs  # noqa: E402
from zarrswarm.node import Node  # noqa: E402
from zarrswarm.store import http  # noqa: E402


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def data(days):
    times = pd.date_range("2020-01-01", periods=24 * days, freq="h")
    d = np.arange(len(times)) / 24.0
    lat, lon = np.linspace(-80, 80, 46), np.linspace(0, 358, 90)
    storm = np.exp(-((d - days * 0.55) / (days * 0.06)) ** 2)  # stormy episode
    amp = 1 + 25 * storm
    rng = np.random.default_rng(0)
    slow = np.cumsum(rng.normal(size=len(times))) * 0.05
    f = (280 + slow[:, None, None] + amp[:, None, None] * np.sin(d[:, None, None] * 0.7 + lat[None, :, None] / 9)
         * np.cos(lon[None, None, :] / 13)).astype("float32")
    return xr.Dataset({"t2m": (("time", "lat", "lon"), f)}, coords={"time": times, "lat": lat, "lon": lon})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--rel-err", type=float, default=0.0005)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ztvas", dir=Path("~/.cache").expanduser()))
    ds = data(a.days)
    true = float(ds.t2m.astype("float64").mean())
    p = str(tmp / "d.zarr")
    ds.to_zarr(p, encoding={"t2m": {"chunks": (24, 23, 45)}}, consolidated=False)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(300)
    bp = port()
    seeder = Node(tmp / "seed", port=bp, ctl_port=port())
    run(seeder.start())
    link = http(f"http://127.0.0.1:{seeder.ctl_port}", "POST", "/api/seed", {"path": p})["link"]
    out = {"true_mean": true, "rel_err": a.rel_err, "vdc_ratio": [], "vas": []}
    for trial in range(a.trials):
        for method in ("vdc_ratio", "vas"):
            c = Node(tmp / f"c_{method}_{trial}", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
            run(c.start())
            dsc = zs.open_dataset(link, ctl=f"http://127.0.0.1:{c.ctl_port}")
            if method == "vas":
                r = zs.progressive_mean_vas(dsc, "t2m", rel_err=a.rel_err, seed=trial)
            else:
                r = zs.progressive_mean(dsc, "t2m", rel_err=a.rel_err)
            r["covered"] = bool(abs(r["mean"] - true) <= r["ci95"]) if r.get("mean") is not None else None
            r["abs_err"] = abs(r["mean"] - true) if r.get("mean") is not None else None
            out[method].append(r)
            print(method, trial, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            run(c.stop())
    for m in ("vdc_ratio", "vas"):
        rs = out[m]
        out[m + "_summary"] = {"chunks_mean": float(np.mean([r["chunks"] for r in rs])), "of": rs[0]["of"],
                               "coverage": float(np.mean([r["covered"] for r in rs])),
                               "abs_err_mean": float(np.mean([r["abs_err"] for r in rs])),
                               "MB_mean": float(np.mean([r["bytes"] for r in rs]) / 1e6)}
    print(json.dumps({k: v for k, v in out.items() if k.endswith("summary") or k in ("true_mean", "rel_err")}, indent=1))
    run(seeder.stop())


if __name__ == "__main__":
    main()
