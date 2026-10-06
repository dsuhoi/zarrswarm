"""Where do copies of one sensed product disagree in their bits? Same bytes through independent decoders, and the
same product through different pipelines. GRIB2 products (NOAA MRMS radar mosaics, GFS analysis fields) are decoded
by ecCodes (ECMWF) and gribberish (an independent Rust implementation); a third path applies the decimal scale in
float32 arithmetic (value = (R + X 2^E) * 10^-D, as a lightweight ingestor might). For every pair: fraction of
bit-identical float32 values and of 256x256 tiles with equivalent lattice identity. ERA5 (two providers) and GOES-16
(three decoders) come from bench/lattice_controls.json and bench/goes_decoders.json.

python bench/decoder_survey.py GRIB_DIR [--out bench/decoder_survey.json]
"""
import argparse
import json
import sys
from pathlib import Path

import eccodes
import gribberish
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zarr_torrent.codec import lattice_of, same_vcid, vcid_of  # noqa: E402

TILE = 256


def decode(path):
    b = path.read_bytes()
    gb = gribberish.parse_grib_message(b, 0).data().ravel()
    with open(path, "rb") as fh:
        h = eccodes.codes_grib_new_from_file(fh)
        ec = eccodes.codes_get_values(h)
        nx, ny = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
        E, D, R = (eccodes.codes_get(h, k) for k in ("binaryScaleFactor", "decimalScaleFactor", "referenceValue"))
        eccodes.codes_release(h)
    X = np.round((ec * 10.0 ** D - R) / 2.0 ** E)  # packed integers (exact for these packings)
    f32 = ((np.float32(R) + X.astype("f4") * np.float32(2.0 ** E)) * np.float32(10.0 ** -D)).astype("f4")
    shape = (ny, nx)
    return {"ecCodes": ec.astype("f4").reshape(shape), "gribberish": gb.astype("f4").reshape(shape),
            "float32 scaling": f32.reshape(shape)}


def compare(A, B):
    tiles = eq = changed_merged = 0
    for y in range(0, A.shape[0], TILE):
        for x in range(0, A.shape[1], TILE):
            ta, tb = A[y:y + TILE, x:x + TILE], B[y:y + TILE, x:x + TILE]
            if np.isfinite(ta).sum() < 2 or np.unique(ta[np.isfinite(ta)]).size < 2:
                continue  # empty or constant tile (e.g. no precipitation): nothing to estimate
            tiles += 1
            va = vcid_of(ta)
            eq += same_vcid(va, vcid_of(tb))
            changed = ta.astype("f8").copy()
            indices = np.flatnonzero(np.isfinite(ta))
            changed.flat[indices[len(indices) // 2]] += lattice_of(ta)[0]
            changed_merged += same_vcid(va, vcid_of(changed.astype(ta.dtype)))
    fin = np.isfinite(A)
    return {"bit_equal": round(float(np.mean(A[fin] == B[fin])), 4), "tiles": tiles, "lattice_equal": eq,
            "one_count_changed_merged": changed_merged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="bench/decoder_survey.json")
    ap.add_argument("--lattice-controls", default="bench/lattice_controls.json")
    ap.add_argument("--goes", default="bench/goes_decoders.json")
    a = ap.parse_args()
    res = {}
    for f in sorted(Path(a.dir).expanduser().glob("*.grib2")):
        d = decode(f)
        res[f.stem] = {"ecCodes vs gribberish": compare(d["ecCodes"], d["gribberish"]),
                       "ecCodes vs float32 scaling": compare(d["ecCodes"], d["float32 scaling"])}
        print(f.stem, json.dumps(res[f.stem]), flush=True)
    c = json.load(open(a.lattice_controls))
    res["ERA5 2m temperature, ARCO vs NCAR (providers)"] = {"bit_equal": c["mean_bit_equal"], "tiles": c["tiles"],
                                                             "lattice_equal": c["providers_equivalent"]}
    g = json.load(open(a.goes))["pairs"]
    for k, v in g.items():
        res[f"GOES-16 ABI L1b C13, {k}"] = {"bit_equal": v["bit_equal"], "tiles": v["tiles"], "lattice_equal": v["lattice_equal"]}
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
