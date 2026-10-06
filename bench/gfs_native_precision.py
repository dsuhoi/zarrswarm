"""Native decoder precision on the SAME retained GFS sample; no new selection/fetch.

python bench/gfs_native_precision.py --sample bench/revalidation/gfs_pipelines_v7.json --out RESULT
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import eccodes
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zarrswarm import codec


def native_field(work, row):
    """Load the pinned original payloads without narrowing the ecCodes output."""
    tag = f"{row['short_name']}_f{row['lead_hours']:03d}"
    gp, nc = work / (tag + ".grib2"), work / (tag + ".nc")
    for p, key in ((gp, "GRIB"), (nc, "NetCDF")):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row["payload_sha256"][key]
    with gp.open("rb") as fh:
        h = eccodes.codes_grib_new_from_file(fh)
        try:
            shape = (eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni"))
            v = eccodes.codes_get_values(h).reshape(shape)
            lat = eccodes.codes_get_array(h, "latitudes").reshape(shape)[:, 0]
            lon = eccodes.codes_get_array(h, "longitudes").reshape(shape)[0]
            decimal = 10.0 ** -eccodes.codes_get(h, "decimalScaleFactor")
            step = 2.0 ** eccodes.codes_get(h, "binaryScaleFactor") * decimal
            origin = eccodes.codes_get(h, "referenceValue") * decimal
            assert np.isclose(step, row["source_quantum"], rtol=1e-14, atol=0)
        finally:
            eccodes.codes_release(h)
    with xr.open_dataset(nc, engine="h5netcdf") as ds:
        name = next(n for n in ds.data_vars if ds[n].ndim >= 2)
        b = ds[name].squeeze(drop=True)
        ys = [np.flatnonzero(lat == x).item() for x in b.latitude.values]
        xs = [np.flatnonzero(lon == x).item() for x in b.longitude.values]
        a, b = v[np.ix_(ys, xs)], b.values
    assert a.dtype == np.dtype("f8") and b.dtype == np.dtype("f4") and a.shape == b.shape == (64, 128)
    return a, b, lat[ys], lon[xs], (step, origin, .45 * step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--packing", action="store_true", help="Also use the original GRIB step/origin, budget 0.45 steps")
    args = ap.parse_args()
    sample = json.loads(args.sample.read_text())
    work = Path(sample["work"])
    out = {"started_utc": datetime.now(timezone.utc).isoformat(), "sample_sha256":
           hashlib.sha256(args.sample.read_bytes()).hexdigest(), "sample": str(args.sample),
           "protocol": "Same eight pinned fields and original payloads; native ecCodes float64 versus NetCDF float32. No target-dtype conversion. Coordinate selection only. One true GRIB-quantum change per tile.",
           "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                             for p in ("bench/gfs_native_precision.py", "zarrswarm/codec.py")}, "fields": []}
    if args.packing:
        protocol = ROOT / "bench/revalidation/packing_protocol_v8.json"
        out["packing_protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
        out["packing_scope"] = "Caller-provided original GRIB reference/scales; 0.45-step decoder budget. Numeric API only, no new station or transport deployment."
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for row in sample["fields"]:
        tag = f"{row['short_name']}_f{row['lead_hours']:03d}"
        a, b, _, _, (step, origin, _) = native_field(work, row)
        matches = changes = packed_matches = packed_changes = packed_eligible = 0
        packing = (step, origin, .45 * step)
        for y in range(0, 64, 32):
            for x in range(0, 128, 32):
                ta, tb = a[y:y + 32, x:x + 32], b[y:y + 32, x:x + 32]
                changed = ta.copy()
                changed[0, 0] += row["source_quantum"]
                va = codec.vcid_of(ta)
                matches += codec.same_vcid(va, codec.vcid_of(tb))
                changes += codec.same_vcid(va, codec.vcid_of(changed))
                if args.packing:
                    try:
                        pa = codec.vcid_of(ta, packing=packing)
                        pb = codec.vcid_of(tb, packing=packing)
                        pc = codec.vcid_of(changed, packing=packing)
                    except ValueError:
                        continue  # out-of-contract tiles are reported, not silently pooled by another rule
                    packed_eligible += 1
                    packed_matches += codec.same_vcid(pa, pb)
                    packed_changes += codec.same_vcid(pa, pc)
        codec.VALUE_ID = "exact"
        try:
            exact_match = codec.same_vcid(codec.vcid_of(a), codec.vcid_of(b))
        finally:
            codec.VALUE_ID = "lattice"
        out["fields"].append({"short_name": row["short_name"], "lead_hours": row["lead_hours"],
                              "dtypes": [str(a.dtype), str(b.dtype)], "units": row["units"],
                              "cells": a.size, "numerically_equal_cells": int(np.count_nonzero(a == b)),
                              "float32_normalized_equal_cells": int(np.count_nonzero(a.astype("f4") == b)),
                              "exact_identity_match": exact_match, "tiles": 8, "matching_tiles": matches,
                              "accepted_source_count_changes": changes, "source_quantum": row["source_quantum"],
                              "max_abs_error": float(np.max(np.abs(a - b))),
                              "max_error_in_source_quanta": float(np.max(np.abs(a - b)) / row["source_quantum"])})
        if args.packing:
            out["fields"][-1]["trusted_packing"] = {
                "source_step": step, "source_origin": origin, "max_error": .45 * step,
                "eligible_tiles": packed_eligible, "matching_tiles": packed_matches,
                "accepted_source_count_changes": packed_changes,
                "max_residual_in_source_steps": [float(np.max(np.abs(vv.astype("f8") -
                    (origin + np.rint((vv.astype("f8") - origin) / step) * step))) / step) for vv in (a, b)]}
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(tag, out["fields"][-1], flush=True)
    out["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
