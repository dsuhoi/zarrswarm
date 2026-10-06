"""Working L3 estimator and shared-quantizer controls on four-variable provider fields.

Source fields are cached without changing values. Shared quantizers use the full ARCO field's estimated step
and, optionally, its phase: these baselines receive common parameters that independent L3 announcements do not.
This compares identity rules, not network protocols or a live station pipeline.
"""
import argparse
import calendar
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zarrswarm.codec import ESTIMATOR, lattice_of, same_vcid, vcid_of
from lattice_probe import ARCO, NCAR, PAIRS

TIMES = ("2020-01-03T18", "2020-01-15T06", "2020-01-28T00")


def quantize(values, step, origin):
    return np.rint((values.astype("f8") - origin) / step).astype("i8")


def compare(a, b):
    assert a.dtype == b.dtype == np.dtype("float32")
    assert np.isfinite(a).all() and np.isfinite(b).all()
    step, phase, _ = lattice_of(a)
    rules = {"exact": (None, None), "L3": (None, None)}
    rules.update({f"shared-{origin}-{factor}step": (step * factor, offset)
                  for origin, offset in (("zero", 0.0), ("ARCO-phase", phase)) for factor in (1, 2, 4)})
    out = {name: {"accepted_provider_tiles": 0, "accepted_changed_tiles": 0} for name in rules}
    tiles = step_matches = 0
    for y in range(0, a.shape[0], 91):
        for x in range(0, a.shape[1], 180):
            ta, tb = a[y:y + 91, x:x + 180], b[y:y + 91, x:x + 180]
            tile_step = lattice_of(ta)[0]
            step_matches += abs(tile_step - step) <= 1e-5 * step
            changed = ta.copy()
            changed.flat[changed.size // 2] += step
            assert changed.flat[changed.size // 2] != ta.flat[ta.size // 2]
            va = vcid_of(ta)
            for name, (width, origin) in rules.items():
                if name == "L3":
                    positive, negative = same_vcid(va, vcid_of(tb)), same_vcid(va, vcid_of(changed))
                elif name == "exact":
                    positive, negative = ta.tobytes() == tb.tobytes(), ta.tobytes() == changed.tobytes()
                else:
                    qa = quantize(ta, width, origin)
                    positive = np.array_equal(qa, quantize(tb, width, origin))
                    negative = np.array_equal(qa, quantize(changed, width, origin))
                out[name]["accepted_provider_tiles"] += int(positive)
                out[name]["accepted_changed_tiles"] += int(negative)
            tiles += 1
    return {"tiles": tiles, "ARCO_field_step": step, "ARCO_field_phase": phase,
            "tile_step_matches_field": step_matches,
            "exact_cell_fraction": float(np.mean(a.view("u4") == b.view("u4"))),
            "max_provider_difference_in_field_steps": float(np.max(np.abs(a.astype("f8") - b)) / step),
            "rules": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", nargs="+", default=TIMES)
    args = ap.parse_args()
    work = Path(args.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    result = {"estimator": ESTIMATOR, "started_utc": datetime.now(timezone.utc).isoformat(),
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "xarray", "h5netcdf", "h5py")},
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("zarrswarm/codec.py", "bench/provider_identity_controls.py",
                                             "bench/lattice_probe.py")},
              "protocol": "Four survey variables; 91x180 tiles; change one cell by one full-field ARCO step. Shared quantizers receive ARCO field parameters. L3 fits each tile independently. Exact identity compares float32 bits.",
              "times": args.times,
              "fields": {}}
    arco = xr.open_zarr(ARCO, consolidated=True, chunks=None)
    for var, code in PAIRS.items():
        months = {}
        for ts in args.times:
            dt = datetime.fromisoformat(ts)
            months.setdefault((dt.year, dt.month), []).append(ts)
        for (year, month), times in months.items():
            ym = f"{year:04d}{month:02d}"
            last = calendar.monthrange(year, month)[1]
            url = f"{NCAR.rsplit('/', 2)[0]}/{ym}/e5.oper.an.sfc.{code}.ll025sc.{ym}0100_{ym}{last:02d}23.nc"
            with fsspec.open(url, block_size=8 << 20).open() as fh, \
                    xr.open_dataset(fh, engine="h5netcdf", chunks=None) as ncar:
                name = list(ncar.data_vars)[0]
                assert all(np.array_equal(arco[c].values.astype("f8"), ncar[c].values.astype("f8"))
                           for c in ("latitude", "longitude"))
                for ts in times:
                    arrays, hashes = [], {}
                    for source, field in (("ARCO", arco[var]), ("NCAR", ncar[name])):
                        p = work / f"{var}_{ts.replace(':', '-')}_{source}.npy"
                        if not p.exists():
                            np.save(p, field.sel(time=np.datetime64(ts)).values, allow_pickle=False)
                        hashes[source] = hashlib.sha256(p.read_bytes()).hexdigest()
                        arrays.append(np.load(p, allow_pickle=False))
                    assert arrays[0].shape == arrays[1].shape == (721, 1440)
                    r = compare(*arrays)
                    r.update(source_urls={"ARCO": ARCO, "NCAR": url}, field_sha256=hashes,
                             dtypes=[str(a.dtype) for a in arrays])
                    result["fields"][f"{var}@{ts}"] = r
                    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
                    print(var, ts, json.dumps(r["rules"]), flush=True)
    arco.close()


if __name__ == "__main__":
    main()
