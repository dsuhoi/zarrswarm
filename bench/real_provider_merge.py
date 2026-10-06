"""Two independent providers of the same measurements: ERA5 2 m temperature from Google (ARCO, Zarr) and from NCAR
(ds633.0 on AWS, NetCDF). Their floats differ (different GRIB decoders): bit-identical in ~2% of the values, never
more than half a packing step apart.

Sites: A = ARCO copy, B = NCAR copy (the site declares its local name VAR_2T as 2m_temperature and keeps float64
coordinates), C = a second ARCO copy. For each identity mode (exact values vs quantization lattice) and holder set,
a fresh client downloads the day; we report how many holders are usable per chunk, missing chunks, and how far the
received values are from each original.

python bench/real_provider_merge.py WORKDIR [--out bench/real_provider_merge.json]
"""
import argparse
import asyncio
import json
import socket
import sys
import threading
from pathlib import Path

import fsspec
import numcodecs
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarr_torrent as zt  # noqa: E402
from zarr_torrent import codec  # noqa: E402
from zarr_torrent.node import Node  # noqa: E402
from zarr_torrent.store import http, wait_job  # noqa: E402

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
NCAR = "https://nsf-ncar-era5.s3.amazonaws.com/e5.oper.an.sfc/202001/e5.oper.an.sfc.128_167_2t.ll025sc.2020010100_2020013123.nc"
VAR, T0, T1 = "2m_temperature", "2020-01-01T00", "2020-01-01T23"


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def fetch(w: Path):
    enc = {VAR: {"chunks": (1, 721, 1440), "compressor": numcodecs.Blosc("lz4", 5, numcodecs.Blosc.SHUFFLE)}}
    if not (w / "arco.zarr").exists():
        a = xr.open_zarr(ARCO, consolidated=True, chunks=None)[[VAR]].sel(time=slice(T0, T1)).load()
        for v in a.variables.values():
            v.encoding.clear()
        a.to_zarr(w / "arco.zarr", encoding=enc, consolidated=False, zarr_format=2)
    if not (w / "ncar.zarr").exists():
        d = xr.open_dataset(fsspec.open(NCAR, block_size=8 << 20).open(), engine="h5netcdf", chunks=None)
        n = d[["VAR_2T"]].sel(time=slice(T0, T1)).load().rename({"VAR_2T": VAR})  # the site's declared name
        n = n.drop_vars([c for c in n.coords if c not in ("time", "latitude", "longitude")])
        for v in n.variables.values():
            v.encoding.clear()
        n[VAR].attrs = {k: v for k, v in n[VAR].attrs.items() if k in ("long_name", "units")}
        n.to_zarr(w / "ncar.zarr", encoding=enc, consolidated=False, zarr_format=2)


def run_case(w: Path, mode: str, holders: list[str], trusted_a=False):
    codec.VALUE_ID = mode
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(300)
    root = w / (f"run_{mode}_{'-'.join(holders)}" + ("_trusted_A" if trusted_a else ""))
    import shutil
    shutil.rmtree(root, ignore_errors=True)  # fresh nodes: caches of an earlier run (other code, other ids) must not leak in
    bp = port()
    nodes = {"boot": Node(root / "boot", port=bp, ctl_port=port())}
    for n in holders + ["client"]:
        nodes[n] = Node(root / n, port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    for n in nodes.values():
        run(n.start())
    if trusted_a:
        nodes["client"].trusted.add(nodes["A"].ident.id)
    ctl = lambda n: f"http://127.0.0.1:{nodes[n].ctl_port}"
    src = {"A": w / "arco.zarr", "B": w / "ncar.zarr", "C": w / "arco.zarr"}
    links = {h: http(ctl(h), "POST", "/api/seed", {"path": str(src[h])})["link"] for h in holders}
    grid = links[holders[0]].removeprefix("zt://")
    v = http(ctl("client"), "GET", f"/api/view/{grid}?refresh=1")
    view = run(nodes["client"].view(grid, refresh=True))
    keys = [k for k in view["best"] if k.startswith(VAR + "@")]
    usable = [len(view["best"][k]["src"]) for k in keys]
    contested = sum(1 for k in keys if view["best"][k].get("contested"))
    jid = http(ctl("client"), "POST", "/api/download", {"grid": grid, "region": {"var": VAR}})["job"]
    job = wait_job(ctl("client"), jid)
    res = {"mode": mode, "holders": holders, "trusted_a": trusted_a, "state": job["state"],
           "sample_coverage": job.get("coverage"), "estimator": codec.ESTIMATOR,
           "one_link": len(set(links.values())) == 1, "chunks": len(keys),
           "usable_holders_per_chunk": round(float(np.mean(usable)), 2) if usable else 0, "contested": contested,
           "done": job["done"], "missing": job["missing"], "per_peer_MB": {p[:6]: round(b / 1e6, 1) for p, b in job["per_peer"].items()}}
    if job["done"]:
        got = zt.open_dataset(links[holders[0]], ctl=ctl("client"))[VAR].values.astype("f8")
        for name, path in (("arco", w / "arco.zarr"), ("ncar", w / "ncar.zarr")):
            ref = xr.open_zarr(path, consolidated=False)[VAR].values.astype("f8")
            ok = np.isfinite(got)
            res[f"max_abs_diff_vs_{name}"] = float(np.abs(got[ok] - ref[ok]).max()) if ok.any() else None
    for n in nodes.values():
        run(n.stop())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--out", default="bench/real_provider_merge.json")
    a = ap.parse_args()
    w = Path(a.work).expanduser()
    w.mkdir(parents=True, exist_ok=True)
    fetch(w)
    rows = []
    for mode in ("exact", "lattice"):
        for holders in (["A", "B"], ["A", "B", "C"]):
            r = run_case(w, mode, holders)
            rows.append(r)
            print(json.dumps(r), flush=True)
    r = run_case(w, "exact", ["A", "B"], trusted_a=True)
    rows.append(r)
    print(json.dumps(r), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
