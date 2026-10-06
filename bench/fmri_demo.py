"""Generality check on medical imaging: a real fMRI BOLD run (OpenNeuro ds000102, sub-01, flanker run 1:
64x64x40 voxels x 146 volumes, TR = 2 s, int16) shared as heterogeneous replicas, like weather fields.

Replicas (each on its own node, each holding part of the run):
  R1 per-volume chunks (1,40,64,64), LZ4, zarr v2             - volumes 0..99
  R2 voxel time-series chunks (146,8,16,16), Zstd, zarr v3     - whole run
  R3 temporally subsampled copy (TR 4 s, every 2nd volume), per-volume chunks - volumes 60..145
Queries by a fresh client: one volume, a region-of-interest time series (8x16x16 voxels, all volumes), and the whole
run at TR 4 s. Every answer is compared with the original NIfTI. With int16 data, exact value identity suffices; the
lattice layer is exercised separately on an EMULATED scaled-float decode (float32 vs float64 arithmetic) - mainstream
NIfTI readers (nibabel, SimpleITK) were found to agree bit for bit, so no real reader disagreement is claimed.

python bench/fmri_demo.py WORKDIR [--out bench/fmri_demo.json]
"""
import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import nibabel as nib
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import ZstdCodec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarrswarm as zs  # noqa: E402
from zarrswarm.codec import same_vcid, vcid_of  # noqa: E402
from zarrswarm.node import Node  # noqa: E402
from zarrswarm.store import http, wait_job  # noqa: E402

URL = "https://s3.amazonaws.com/openneuro.org/ds000102/sub-01/func/sub-01_task-flanker_run-1_bold.nii.gz"


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def load(w: Path) -> xr.Dataset:
    f = w / "sub01_run1.nii.gz"
    if not f.exists():
        import urllib.request
        urllib.request.urlretrieve(URL, f)
    im = nib.load(str(f))
    tr = float(im.header.get_zooms()[3])
    data = np.asarray(im.dataobj).astype("i2").transpose(3, 2, 1, 0)  # (t, z, y, x)
    t = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(data.shape[0]) * tr, unit="s")
    return xr.Dataset({"bold": (("time", "z", "y", "x"), data, {"long_name": "BOLD signal", "units": "a.u."})},
                      coords={"time": t, "z": np.arange(data.shape[1], dtype="f8") * 4.0,
                              "y": np.arange(data.shape[2], dtype="f8") * 3.0, "x": np.arange(data.shape[3], dtype="f8") * 3.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--out", default="bench/fmri_demo.json")
    a = ap.parse_args()
    w = Path(a.work).expanduser()
    w.mkdir(parents=True, exist_ok=True)
    ds = load(w)
    lz4 = numcodecs.Blosc("lz4", 5, numcodecs.Blosc.SHUFFLE)
    reps = {"R1": (ds.isel(time=slice(0, 100)), {"bold": {"chunks": (1, 40, 64, 64), "compressor": lz4}}, 2),
            "R2": (ds, {"bold": {"chunks": (146, 8, 16, 16), "compressors": [ZstdCodec(level=3)]}}, 3),
            "R3": (ds.isel(time=slice(60, None, 2)), {"bold": {"chunks": (1, 40, 64, 64), "compressor": lz4}}, 2)}
    for name, (d, enc, fmt) in reps.items():
        p = w / f"{name}.zarr"
        if not p.exists():
            d.to_zarr(p, encoding=enc, consolidated=False, zarr_format=fmt)
    import shutil
    for d in w.glob("n_*"):  # fresh node homes: no cache from an earlier run
        shutil.rmtree(d)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(300)
    bp = port()
    nodes = {"boot": Node(w / "n_boot", port=bp, ctl_port=port())}
    for n in list(reps) + ["client"]:
        nodes[n] = Node(w / f"n_{n}", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
    for n in nodes.values():
        run(n.start())
    ctl = lambda n: f"http://127.0.0.1:{nodes[n].ctl_port}"
    links = {n: http(ctl(n), "POST", "/api/seed", {"path": str(w / f"{n}.zarr")})["link"] for n in reps}
    res = {"one_link": len(set(links.values())) == 1, "layouts": {}, "queries": {}}
    grid = next(iter(links.values())).removeprefix("zs://")
    v = http(ctl("client"), "GET", f"/api/view/{grid}?refresh=1")
    res["layouts"] = sorted(v["arrays"]["bold"]["layouts"])
    truth = ds.bold.values
    link = links["R1"]
    queries = {"one volume (t=120, only in R2/R3)": ({"t0": "2000-01-01T00:04:00", "t1": "2000-01-01T00:04:00", "step": 2},
                                                   lambda x: x.sel(time="2000-01-01T00:04:00")),
               "ROI time series 8x16x16, all 146 volumes": ({"step": 2, "isel": {"z": [16, 24], "y": [16, 32], "x": [16, 32]}},
                                                            lambda x: x.isel(z=slice(16, 24), y=slice(16, 32), x=slice(16, 32))),
               "whole run at TR 4 s": ({"step": 4}, lambda x: x.isel(time=slice(0, None, 2)))}
    for qn, (reg, sel) in queries.items():
        c = Node(w / f"n_q{len(res['queries'])}", port=port(), ctl_port=port(), bootstrap=[f"http://127.0.0.1:{bp}"])
        run(c.start())
        cc = f"http://127.0.0.1:{c.ctl_port}"
        t = time.time()
        jid = http(cc, "POST", "/api/download", {"grid": grid, "region": dict(reg, var="bold")})["job"]
        job = wait_job(cc, jid)
        step = "4s" if reg["step"] == 4 else None
        view = zs.open_dataset(link, ctl=cc, step=step).bold
        got = (view if step else sel(view)).values  # the 4 s view is already subsampled
        ref = sel(ds.bold).values
        res["queries"][qn] = {"seconds": round(time.time() - t, 2), "chunks": job["done"], "missing": job["missing"],
                              "MB": round(job["bytes"] / 1e6, 2), "layouts": (job.get("cover") or {}).get("layouts"),
                              "exact": bool(np.array_equal(got, ref))}
        print(qn, json.dumps(res["queries"][qn]), flush=True)
        run(c.stop())
    for n in nodes.values():
        run(n.stop())
    # lattice layer on an EMULATED scaled decode (scanner rescale slope/intercept; float32 vs float64 arithmetic)
    slope, inter = 0.0038157, 12.5
    f64 = (truth.astype("f8") * slope + inter).astype("f4")
    f32 = truth.astype("f4") * np.float32(slope) + np.float32(inter)
    res["emulated_scaled_decode"] = {
        "bit_identical_values": round(float((f64 == f32).mean()), 4),
        "volumes_equivalent_lattice": int(sum(same_vcid(vcid_of(f64[i:i + 1], 0), vcid_of(f32[i:i + 1], 0)) for i in range(146))),
        "volumes_exact_equal": int(sum(np.array_equal(f64[i], f32[i]) for i in range(146))),
        "one_step_shift_kept_apart": not same_vcid(vcid_of(f64[5:6], 0), vcid_of(f64[5:6] + np.float32(slope), 0))}
    print(json.dumps(res["emulated_scaled_decode"]))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
