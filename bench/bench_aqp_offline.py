"""Offline (network-free) comparison of anytime estimators on exact chunk means of a heteroscedastic field.
python bench/bench_aqp_offline.py [--trials 2000] [--rel-err 0.0005]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zarrswarm.aqp import VAS, RatioVDC, vdc_order
from bench_vas import data

ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=2000)
ap.add_argument("--rel-err", type=float, default=0.0005); ap.add_argument("--days", type=int, default=120)
ap.add_argument("--strata", type=int, default=8); ap.add_argument("--pilot", type=int, default=5); ap.add_argument("--min-h", type=int, default=5)
a = ap.parse_args()
f = data(a.days).t2m.values.astype("float64")                       # (time, 46, 90), chunks (24, 23, 45)
T, Y, X = f.shape
cm = f.reshape(T // 24, 24, 2, 23, 2, 45).mean(axis=(1, 3, 5))       # chunk means (days, 2, 2)
means = cm.reshape(-1)                                                # time-major order of chunks
true = means.mean(); N = len(means); rng = np.random.default_rng(0)
res = {}
for name in ("ratio_vdc", "vas"):
    cov, used = 0, []
    for t in range(a.trials):
        if name == "ratio_vdc":
            est = RatioVDC(N)
            for i in vdc_order(N, int(rng.integers(N))):
                est.add(means[i] * 1035, 1035)
                if est.done(a.rel_err):
                    break
        else:
            strata = np.array_split(np.arange(N), a.strata)
            perm = [rng.permutation(s) for s in strata]; ptr = [0] * a.strata
            est = VAS([len(s) for s in strata], pilot=a.pilot, min_h=a.min_h)
            for h in est.pilot_batch():
                est.add(h, means[perm[h][ptr[h]]]); ptr[h] += 1
            while not est.done(a.rel_err):
                for h in est.next_batch(8):
                    est.add(h, means[perm[h][ptr[h]]]); ptr[h] += 1
        m, hw, n = est.estimate()
        cov += abs(m - true) <= hw; used.append(n)
    res[name] = {"chunks_mean": round(float(np.mean(used)), 1), "chunks_p90": int(np.percentile(used, 90)), "of": N,
                 "coverage": round(cov / a.trials, 3)}
print(json.dumps({"rel_err": a.rel_err, **res}, indent=1))
