"""Metadata overhead of registering a dataset of N chunks: scan time, manifest size (prolly-tree pages, compressed),
hash cache on disk, resident memory of a real node holding it, and what goes into the DHT. Synthetic quantized
float32 data, one small chunk per time step, so N is large while the data stay small; per-chunk costs are what scale.

python bench/bench_metadata.py [--n 1000 10000 100000] [--out bench/metadata.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from procswarm import ProcSwarm  # noqa: E402
from zarr_torrent import capt  # noqa: E402
from zarr_torrent.scan import scan  # noqa: E402
from zarr_torrent.store import http  # noqa: E402


def rss(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--out", default="bench/metadata.json")
    ap.add_argument("--work", default="~/.cache/zt_meta")
    a = ap.parse_args()
    root = Path(a.work).expanduser()
    import shutil
    root.mkdir(parents=True, exist_ok=False)  # refuse to erase any previous experiment
    rows = []
    for n in a.n:
        p = root / f"d{n}.zarr"
        k = np.random.default_rng(n).integers(0, 4000, (n, 16, 16))
        ds = xr.Dataset({"t2m": (("time", "y", "x"), (250 + k * 2.0 ** -9).astype("f4"))},
                        coords={"time": pd.date_range("2000-01-01", periods=n, freq="h"),
                                "y": np.arange(16.0), "x": np.arange(16.0)})
        ds.to_zarr(p, encoding={"t2m": {"chunks": (1, 16, 16)}}, consolidated=False)
        data_bytes = du(p / "t2m")
        t = time.time()
        res = scan(p, root / f"scan{n}")
        t_scan = time.time() - t
        sg = next(iter(res["subgrids"].values()))
        root_hash, pages = capt.build(sg["chunks"])
        man = sum(len(b) for b in pages.values())
        sw = ProcSwarm(root / f"sw{n}", 2, 1, 0.0, seed=1, relay_rate=None)
        pid = sw.nodes["p00"].proc.pid
        time.sleep(2)
        idle = rss(pid)
        t = time.time()
        http(sw.ctl("p00"), "POST", "/api/seed", {"path": str(p)}, timeout=3600)
        t_seed = time.time() - t
        time.sleep(2)
        held = rss(pid)
        st = http(sw.ctl("p00"), "GET", "/api/status")
        row = {"chunks": n, "data_MB": round(data_bytes / 1e6, 1), "scan_s": round(t_scan, 1),
               "scan_ms_per_chunk": round(1e3 * t_scan / n, 2), "seed_s": round(t_seed, 1),
               "manifest_MB": round(man / 1e6, 2), "manifest_B_per_chunk": round(man / n, 1), "pages": len(pages),
               "hash_cache_B_per_chunk": round(du(root / f"scan{n}") / n, 1),
               "node_idle_MB": round(idle, 1), "node_holding_MB": round(held, 1),
               "node_B_per_chunk": round((held - idle) * 1024 * 1024 / n, 1),
               "dht_records": st.get("dht_values")}
        rows.append(row)
        print(json.dumps(row), flush=True)
        sw.stop()
    json.dump(rows, open(a.out, "w"), indent=1)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
