"""Download the union of the ERA5-like replicas through a local zt node and verify every value.

python bench/verify_real.py zt://<grid> --ctl http://127.0.0.1:17902 [--vars t2m,u10,v10] [--time A:B]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zarr_torrent as zt  # noqa: E402
from make_era5like import field  # noqa: E402
from zarr_torrent.store import http, keys_for, open_view, wait_job  # noqa: E402

COVER = {"t2m": ("2020-01-01", "2020-03-15 23:00"), "u10": ("2020-01-01", "2020-02-29 23:00"),
         "v10": ("2020-02-15", "2020-03-15 23:00")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("link")
    ap.add_argument("--ctl", default="http://127.0.0.1:7882")
    ap.add_argument("--vars", default="t2m,u10,v10")
    ap.add_argument("--time")
    ap.add_argument("--cover", default="jlps")
    a = ap.parse_args()
    vars_ = a.vars.split(",")
    t0, t1 = (a.time.split("/") if a.time else (None, None))
    grid, view = open_view(a.link, a.ctl)
    out = {"grid": grid}
    t_start = time.time()
    for v in vars_:
        jid = http(a.ctl, "POST", "/api/download", {"grid": grid, "cover": a.cover, "label": f"verify {v}",
                                                    "region": {"var": v, "t0": t0, "t1": t1}})["job"]
        job = wait_job(a.ctl, jid, every=0.5)
        out[v] = {"state": job["state"], "chunks": job["done"], "total": job["total"], "failed": job.get("failed", 0), "missing": job["missing"],
                  "MB": round(job["bytes"] / 1e6, 1), "s": round(job["t1"] - job["t0"], 2),
                  "MBps": round(job["bytes"] / 1e6 / max(job["t1"] - job["t0"], 1e-3), 2),
                  "per_peer_MB": {p[:8]: round(b / 1e6, 1) for p, b in job["per_peer"].items()},
                  "cover": job.get("cover")}
        print(v, json.dumps(out[v]), flush=True)
    ds = zt.open_dataset(a.link, ctl=a.ctl)
    if t0 or t1:
        ds = ds.sel(time=slice(t0, t1))
    times = pd.DatetimeIndex(ds.time.values)
    for v in vars_:
        got = ds[v].values
        lo, hi = COVER[v]
        m = (times >= lo) & (times <= hi)
        ref = field(v, times[m])
        out[v]["values_equal"] = bool(np.array_equal(got[m], ref))
        out[v]["nan_outside"] = bool(np.isnan(got[~m]).all())
    out["total_s"] = round(time.time() - t_start, 2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
