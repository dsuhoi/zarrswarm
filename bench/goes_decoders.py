"""Real sensor data, several decoders: GOES-16 ABI L1b radiances (CONUS, band 13) are stored as 14-bit counts packed
in int16 (_Unsigned) with scale_factor/add_offset. Three common ways to read the same file:
  xarray (CF decoding, netCDF4 engine), netCDF4-python (auto mask and scale), and a float64 pipeline (h5py raw
  counts, scale and offset applied in double precision, stored as float32).
Per tile: fraction of bit-identical values and whether lattice identities (codec.vcid_of / same_vcid) agree; the
same tile with one count changed must not agree.

python bench/goes_decoders.py DIR_WITH_NC_FILES [--out bench/goes_decoders.json]
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import netCDF4
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zarrswarm.codec import lattice_of, same_vcid, vcid_of  # noqa: E402

TILE = 250


def decoders(f):
    xa = xr.open_dataset(f, engine="netcdf4", mask_and_scale=True)["Rad"].values
    with netCDF4.Dataset(f) as d:
        nc = np.ma.filled(d["Rad"][:].astype("f4"), np.nan)
    with h5py.File(f) as h:
        v = h["Rad"]
        raw = v[...].view("u2") if v.attrs.get("_Unsigned", b"").astype(str) == "true" else v[...]
        fill = int(np.asarray(v.attrs["_FillValue"]).view("u2")[0])
        f64 = raw.astype("f8") * float(v.attrs["scale_factor"][0]) + float(v.attrs["add_offset"][0])
        f64[raw == fill] = np.nan
    return {"xarray": xa.astype("f4"), "netCDF4": nc, "float64 pipeline": f64.astype("f4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="bench/goes_decoders.json")
    a = ap.parse_args()
    files = sorted(Path(a.dir).expanduser().glob("OR_ABI-L1b-Rad*.nc"))
    res = {"files": [f.name for f in files], "pairs": {}}
    for f in files:
        d = decoders(f)
        names = list(d)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p = res["pairs"].setdefault(f"{names[i]} vs {names[j]}",
                                            {"tiles": 0, "bit_equal": 0.0, "lattice_equal": 0, "one_count_changed_merged": 0})
                A, B = d[names[i]], d[names[j]]
                for y in range(0, A.shape[0], TILE):
                    for x in range(0, A.shape[1], TILE):
                        ta, tb = A[y:y + TILE, x:x + TILE], B[y:y + TILE, x:x + TILE]
                        fin = np.isfinite(ta)
                        if fin.sum() < 2:
                            continue
                        p["tiles"] += 1
                        p["bit_equal"] += float(np.mean(ta[fin] == tb[fin]))
                        va = vcid_of(ta)
                        p["lattice_equal"] += same_vcid(va, vcid_of(tb))
                        s = lattice_of(ta)[0]
                        t2 = ta.astype("f8").copy()
                        k = np.flatnonzero(fin)[len(np.flatnonzero(fin)) // 2]
                        t2.flat[k] += s
                        p["one_count_changed_merged"] += same_vcid(va, vcid_of(t2.astype("f4")))
        res.setdefault("step", float(lattice_of(d["xarray"])[0]))
    for p in res["pairs"].values():
        p["bit_equal"] = round(p["bit_equal"] / max(p["tiles"], 1), 4)
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
