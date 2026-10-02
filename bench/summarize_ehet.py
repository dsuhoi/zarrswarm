"""Summarise sim/e_hetero.py results for the paper: per query and configuration, median / min / max seconds, median
MB and completeness over repetitions -> CSV (paper/figs/plots/e1e2.csv) + a printed table.

python bench/summarize_ehet.py sim/results_ehet_meteor.json [--csv paper/figs/plots/e1e2.csv]
"""
import argparse
import json
import statistics as st

CONF = {("values", "jlps"): "lattice_jlps", ("values", "bytes"): "lattice_minbytes", ("bytes", "jlps"): "bytes_jlps"}
QUERIES = ("map_day_1h", "series_point_1h", "period_6h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--csv", default="paper/figs/plots/e1e2.csv")
    a = ap.parse_args()
    rows = json.load(open(a.results))["rows"]
    out = {}
    failed = [r for r in rows if r.get("seconds") is None]
    if failed:
        print(f"{len(failed)} failed queries (no answer within the deadline):", [(r["query"], r["mode"], r["rep"]) for r in failed])
    for r in rows:
        if r.get("seconds") is None:
            continue
        c = CONF.get((r["mode"], r.get("select", "jlps")))
        if c:
            out.setdefault((r["query"], c), []).append(r)
    lines = ["query,qi," + ",".join(f"{c}_s,{c}_lo,{c}_hi" for c in CONF.values())]
    for qi, q in enumerate(QUERIES):
        cells = []
        for c in CONF.values():
            xs = out.get((q, c), [])
            t = [x["seconds"] for x in xs]
            med = st.median(t) if t else float("nan")
            cells += [f"{med:.2f}", f"{med - min(t):.2f}" if t else "nan", f"{max(t) - med:.2f}" if t else "nan"]
            if t:
                print(f"{q:16} {c:17} median {med:7.1f}s  [{min(t):.1f}, {max(t):.1f}]  "
                      f"MB {st.median(x['bytes'] for x in xs) / 1e6:7.0f}  complete {min(x['coverage'] for x in xs):.3f}"
                      f"  holders {st.median(x['swarm_holders'] for x in xs):.0f}  n={len(t)}")
        lines.append(f"{q},{qi}," + ",".join(cells))
    open(a.csv, "w").write("\n".join(lines) + "\n")
    print("->", a.csv)


if __name__ == "__main__":
    main()
