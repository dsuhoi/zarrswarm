"""Lattice identity as the system computes it (codec.vcid_of / same_vcid), per chunk-sized tile, on two independent
providers of one day of ERA5 2 m temperature (bench/real_provider_merge.py WORKDIR: arco.zarr, ncar.zarr), plus
quantized negative controls that must NOT merge:
  * one code changed in 1 / 10 / 100 random cells of a tile (+-1 step),
  * the whole tile shifted by one step,
  * the same tile one hour later (other measurements, same lattice).
Also: does the step estimated per tile equal the step of the whole field (a sparse or smooth tile could yield a
coarser step)?

python bench/lattice_controls.py WORKDIR [--out bench/lattice_controls.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zarrswarm.codec import ESTIMATOR, lattice_of, same_vcid, vcid_of  # noqa: E402

VAR, TILE = "2m_temperature", (91, 180)


def tiles(field):
    ny, nx = field.shape
    for y in range(0, ny, TILE[0]):
        for x in range(0, nx, TILE[1]):
            yield field[y:y + TILE[0], x:x + TILE[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--out", default="bench/lattice_controls.json")
    a = ap.parse_args()
    w = Path(a.work).expanduser()
    A = xr.open_zarr(w / "arco.zarr", consolidated=False)[VAR].values
    B = xr.open_zarr(w / "ncar.zarr", consolidated=False)[VAR].values
    rng = np.random.default_rng(0)
    n = {"tiles": 0, "providers_same": 0, "bits_equal": 0.0, "step_eq_field": 0}
    neg = {"1 cell +-1 step": 0, "10 cells": 0, "100 cells": 0, "shift one step": 0, "next hour": 0}
    for t in range(A.shape[0]):
        s_field = lattice_of(A[t])[0]
        for ta, tb, tn in zip(tiles(A[t]), tiles(B[t]), tiles(A[(t + 1) % A.shape[0]])):
            n["tiles"] += 1
            va = vcid_of(ta)
            n["providers_same"] += same_vcid(va, vcid_of(tb))
            n["bits_equal"] += float(np.mean(ta == tb))
            s = lattice_of(ta)[0]
            n["step_eq_field"] += abs(s - s_field) <= 1e-5 * s_field
            for name, m in (("1 cell +-1 step", 1), ("10 cells", 10), ("100 cells", 100)):
                x = ta.astype("f8").copy().ravel()
                idx = rng.choice(x.size, m, replace=False)
                x[idx] += rng.choice([-1, 1], m) * s
                neg[name] += same_vcid(va, vcid_of(x.reshape(ta.shape).astype(ta.dtype)))
            neg["shift one step"] += same_vcid(va, vcid_of((ta.astype("f8") + s).astype(ta.dtype)))
            neg["next hour"] += same_vcid(va, vcid_of(tn))
    n["bits_equal"] = round(n["bits_equal"] / n["tiles"], 4)
    res = {"tiles": n["tiles"], "providers_equivalent": n["providers_same"], "mean_bit_equal": n["bits_equal"],
           "tile_step_equals_field_step": n["step_eq_field"], "false_merges": neg, "estimator": ESTIMATOR,
           "multi_slice": {}}
    for length in (2, 6, 24):
        total = equivalent = cancelled = 0
        for start in range(0, len(A) - length + 1, length):
            for y in range(0, A.shape[1], TILE[0]):
                for x in range(0, A.shape[2], TILE[1]):
                    ta, tb = (arr[start:start + length, y:y + TILE[0], x:x + TILE[1]] for arr in (A, B))
                    va = vcid_of(ta, 0)
                    total += 1
                    equivalent += same_vcid(va, vcid_of(tb, 0))
                    changed = ta.copy()
                    changed[0] += lattice_of(ta[0])[0]
                    changed[1] -= lattice_of(ta[1])[0]
                    cancelled += same_vcid(va, vcid_of(changed, 0))
        res["multi_slice"][str(length)] = {"tiles": total, "providers_equivalent": equivalent,
                                            "opposing_shift_false_merges": cancelled}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
