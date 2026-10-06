"""Holder-count sweep (sim/e_hetero.py --peers N): median completion time per identity and query vs N
-> bench/speed_vs_peers.csv.

python bench/summarize_sweep.py sim/results_sweep_12.json sim/results_sweep_24.json ...
"""
import json
import statistics as st
import sys

cols = {("values", "map_day_1h"): "lat_map", ("bytes", "map_day_1h"): "byt_map",
        ("values", "period_6h"): "lat_6h", ("bytes", "period_6h"): "byt_6h"}
lines = ["peers," + ",".join(cols.values()) + ",hold_lat,hold_byt"]
for f in sys.argv[1:]:
    d = json.load(open(f))
    rows = d["rows"]
    cell = {c: st.median(r["seconds"] for r in rows if (r["mode"], r["query"]) == k) for k, c in cols.items()}
    hold = {m: st.median(r["swarm_holders"] for r in rows if r["mode"] == m and r["query"] == "map_day_1h")
            for m in ("values", "bytes")}
    lines.append(f"{d['args']['peers']}," + ",".join(f"{cell[c]:.2f}" for c in cols.values())
                 + f",{hold['values']:.0f},{hold['bytes']:.0f}")
    print(lines[-1])
open("bench/speed_vs_peers.csv", "w").write("\n".join(lines) + "\n")
