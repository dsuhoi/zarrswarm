"""E1/E2 under two emulations of the same swarm: process-level (token-bucket uplinks and delays inside the node,
sim/procswarm.py) and kernel-level (every node in its own network namespace, links shaped by tc netem/tbf,
sim/netemu.py). Same placements (seeded by repetition), same queries. Prints median seconds per configuration,
the paired speedups of lattice+JLPS, and whether both emulations rank the configurations alike.

python bench/compare_emulators.py PROC.json EMU.json [--out bench/emulators.json] [--reps N]
"""
import argparse
import json
import statistics as st

import numpy as np

CONF = {("values", "jlps"): "lattice+JLPS", ("values", "bytes"): "lattice+min-bytes", ("bytes", "jlps"): "bytes+JLPS"}
QUERIES = ("map_day_1h", "series_point_1h", "period_6h")


def table(path, reps=None):
    t = {}
    for r in json.load(open(path))["rows"]:
        if r.get("seconds") is not None and r.get("coverage", 0) == 1 and not r.get("missing", 0) and \
                r.get("state", "done") == "done" and r.get("value_check") == "exact" and \
                not r.get("sample_coverage", {}).get("missing_samples", 0) and \
                (reps is None or r["rep"] < reps):
            t[(r["query"], CONF[(r["mode"], r["select"])], r["rep"])] = r["seconds"]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proc")
    ap.add_argument("emu")
    ap.add_argument("--out", default="bench/emulators.json")
    ap.add_argument("--reps", type=int, help="only repetitions 0..N-1 (the placements both runs share)")
    a = ap.parse_args()
    res = {}
    for name, path in (("process", a.proc), ("kernel", a.emu)):
        t = table(path, a.reps)
        reps = sorted({k[2] for k in t})
        out = {}
        for q in QUERIES:
            times = {c: [t[(q, c, r)] for r in reps if (q, c, r) in t] for c in CONF.values()}
            med = {c: st.median(v) for c, v in times.items() if v}
            ratios = {b: [t[(q, b, r)] / t[(q, "lattice+JLPS", r)] for r in reps
                         if (q, b, r) in t and (q, "lattice+JLPS", r) in t]
                      for b in ("bytes+JLPS", "lattice+min-bytes")}
            out[q] = {"median_s": {c: round(med[c], 2) if c in med else None for c in CONF.values()},
                      "speedup_vs": {b: round(float(np.median(v)), 2) if v else None for b, v in ratios.items()},
                      "complete_runs": {c: len(v) for c, v in times.items()},
                      "complete_pairs": {b: len(v) for b, v in ratios.items()},
                      "rank": sorted(med, key=med.get), "n": len(reps)}
        res[name] = out
    for q in QUERIES:
        p, k = res["process"][q], res["kernel"][q]
        print(f"{q:16} process {p['median_s']}  kernel {k['median_s']}")
        print(f"{'':16} speedup process {p['speedup_vs']}  kernel {k['speedup_vs']}  same ranking: {p['rank'] == k['rank']}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
