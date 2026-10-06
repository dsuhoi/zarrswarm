"""Summarise sim/multisite.py: per configuration and query, median [min, max] seconds, median MB, layouts, worst
error vs the source -> printed table + LaTeX rows for the paper.

python bench/summarize_multisite.py sim/results_multisite.json
"""
import json
import statistics as st
import sys

CONF = [("cloud", None, "cloud bucket (GCS)"), ("mirror", None, "one mirror (public layout)"),
        ("http", None, "HTTP mirrors with a catalog"),
        ("bytes", "jlps", "byte identity, best swarm (oracle)"), ("swarm", "bytes", "lattice, min bytes"),
        ("swarm", "jlps", "lattice + JLPS")]
QUERIES = ("map_day_1h", "series_point_1h", "week_6h")

rows = json.load(open(sys.argv[1]))["rows"]
for ph, sel, name in CONF:
    cells = []
    for q in QUERIES:
        candidates = [r for r in rows if r["phase"] == ph and r.get("select") == sel and r["query"] == q]
        xs = [r for r in candidates if r.get("coverage", 1) == 1 and not r.get("missing", 0)
              and r.get("state", "done") == "done" and r.get("max_abs_err") == 0]
        if len(xs) != len(candidates):
            print(f"{name} {q}: {len(candidates) - len(xs)} incomplete or unverified answers excluded")
        if not xs:
            cells.append("---")
            continue
        t = [x["seconds"] for x in xs]
        err = max((x["max_abs_err"] for x in xs if isinstance(x.get("max_abs_err"), float)), default=None)
        mb = st.median(x["MB"] for x in xs) if "MB" in xs[0] else None
        miss = sum(x.get("missing", 0) for x in xs)
        print(f"{name:36} {q:16} {st.median(t):6.2f}s [{min(t):.2f},{max(t):.2f}]  MB {mb}  err {err}  missing {miss}"
              f"  n={len(t)}  {sorted({str(x.get('layouts')) for x in xs})}")
        cells.append(f"{st.median(t):.1f}" + (f" ({mb:.0f})" if mb is not None else ""))
    print("  LaTeX:", name, "&", " & ".join(cells), r"\\")
