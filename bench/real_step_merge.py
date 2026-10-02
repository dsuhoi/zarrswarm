"""Real-data check of time-quantum identity: hourly ARCO-ERA5 and 6-hourly WeatherBench2 (public GCS copies of
the same ERA5 field, 0.25 deg, 721x1440) join ONE swarm, and a 6-hourly view over their union equals the cloud
originals bit for bit.

python bench/real_step_merge.py WORKDIR [--out bench/real_step_merge.json]
"""
import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarr_torrent as zt  # noqa: E402
from zarr_torrent.node import Node  # noqa: E402
from zarr_torrent.scan import scan  # noqa: E402
from zarr_torrent.store import http  # noqa: E402

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2 = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
VAR = "2m_temperature"


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def fetch(url, t0, t1, out: Path):
    if out.exists():
        return
    ds = xr.open_zarr(url, consolidated=True, chunks=None)[[VAR]].sel(time=slice(t0, t1))
    enc = {VAR: {"chunks": (1, 721, 1440)}}  # the originals' layout
    for v in ds.variables.values():
        v.encoding.pop("chunks", None)
        v.encoding.pop("preferred_chunks", None)
    ds.load().to_zarr(out, encoding=enc, consolidated=False, zarr_format=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--out", default="bench/real_step_merge.json")
    a = ap.parse_args()
    w = Path(a.work).expanduser()
    w.mkdir(parents=True, exist_ok=True)
    t = time.time()
    fetch(ARCO, "2020-01-01T00", "2020-01-02T23", w / "arco_1h.zarr")      # 48 hourly steps
    fetch(WB2, "2020-01-02T00", "2020-01-05T18", w / "wb2_6h.zarr")        # 16 six-hourly steps
    fetch_s = time.time() - t
    s1, s2 = scan(w / "arco_1h.zarr"), scan(w / "wb2_6h.zarr")
    g1 = {g for g, sg in s1["subgrids"].items() if VAR in sg["arrays"]}
    g2 = {g for g, sg in s2["subgrids"].items() if VAR in sg["arrays"]}
    lay1 = [l for sg in s1["subgrids"].values() for l in sg["arrays"].get(VAR, {}).get("layouts", {})]
    lay2 = [l for sg in s2["subgrids"].values() for l in sg["arrays"].get(VAR, {}).get("layouts", {})]

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(300)
    bp = port()
    nodes = [Node(w / "n0", port=bp, ctl_port=port(), relay_server=True)]
    nodes += [Node(w / f"n{i}", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"]) for i in (1, 2)]
    for n in nodes:
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{n.ctl_port}"
    l1 = http(ctl(nodes[0]), "POST", "/api/seed", {"path": str(w / "arco_1h.zarr")})["link"]
    l2 = http(ctl(nodes[1]), "POST", "/api/seed", {"path": str(w / "wb2_6h.zarr")})["link"]
    t = time.time()
    six = zt.open_dataset(l1, ctl=ctl(nodes[2]), step="6h")[VAR].sel(time=slice("2020-01-01", "2020-01-05T18")).load()
    get_s = time.time() - t
    ref = xr.open_zarr(WB2, consolidated=True, chunks=None)[VAR].sel(time=six.time.values).load()
    exact = bool(np.array_equal(six.values, ref.values))
    arco_only = six.sel(time=slice("2020-01-01", "2020-01-01T18"))  # only the hourly replica has day 1
    res = {"same_link": l1 == l2, "grid_ids_arco": sorted(g1), "grid_ids_wb2": sorted(g2),
           "layouts_arco": lay1, "layouts_wb2": lay2, "view_steps": int(six.sizes["time"]),
           "day1_from_hourly_replica": int(arco_only.sizes["time"]), "values_exact_vs_cloud": exact,
           "fetch_s": round(fetch_s, 1), "swarm_read_s": round(get_s, 2),
           "per_peer": {p[:8]: b for j in http(ctl(nodes[2]), "GET", "/api/status")["jobs"]
                        for p, b in j["per_peer"].items()}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    for n in nodes:
        run(n.stop())


if __name__ == "__main__":
    main()
