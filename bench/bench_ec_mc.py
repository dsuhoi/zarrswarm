"""Monte-Carlo availability of replication vs Cauchy-RS volunteers at EQUAL extra storage, under independent
node failures - the placement model of the simulator (seeders hold contiguous time windows), thousands of trials.
Decodability rule = the one implemented and tested in zarrswarm (a stripe with m lost members is rebuilt iff
>= m of its parity rows survive; tests/test_parity.py checks it end-to-end).

python bench/bench_ec_mc.py [--trials 3000]
"""
import argparse
import json

import numpy as np


def stripes(n, k, d):
    """list of member index arrays for stride-d stripes over n chunks (d=1: consecutive)."""
    out = {}
    for c in range(n):
        q, r = divmod(c, k * d)
        i, s = divmod(r, d)
        out.setdefault((q, s), []).append(c)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seeders", type=int, default=12)
    ap.add_argument("--budget", type=int, default=1, help="extra storage in full replicas")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    N, k, B = a.days, a.k, a.budget
    res = {}
    for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        acc = {m: [] for m in ("none", "replicas", "rs_consecutive", "rs_interleaved")}
        for _ in range(a.trials):
            # seeders: random contiguous windows (5..40% of the period), as in sim/simulate.py
            hold = np.zeros((a.seeders, N), bool)
            for s_ in range(a.seeders):
                span = max(1, int(N * rng.uniform(0.05, 0.4)))
                st = rng.integers(0, N - span + 1)
                hold[s_, st:st + span] = True
            exists = hold.any(0)
            alive_s = rng.random(a.seeders) > p
            have = hold[alive_s].any(0)
            acc["none"].append(have[exists].mean())
            # B full replicas on B volunteers
            rep_alive = (rng.random(B) > p).any()
            acc["replicas"].append(1.0 if rep_alive else have[exists].mean())
            # 8B RS rows on 8B volunteers (each row = 1/k of the data)
            rows_alive = int((rng.random(k * B) > p).sum())
            for name, d in (("rs_consecutive", 1), ("rs_interleaved", max(1, -(-N // k)))):
                got = have.copy()
                for mem in stripes(N, k, d):
                    mem = [c for c in mem if exists[c]]
                    lost = [c for c in mem if not have[c]]
                    if lost and len(lost) <= rows_alive:
                        got[lost] = True
                acc[name].append(got[exists].mean())
        res[p] = {m: round(float(np.mean(v)), 3) for m, v in acc.items()}
        print(p, res[p], flush=True)
    print(json.dumps({"budget_replicas": B, "k": k, "days": N, "seeders": a.seeders, "trials": a.trials,
                      "availability": res}, indent=1))


if __name__ == "__main__":
    main()
