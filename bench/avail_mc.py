"""Usable redundancy and availability on the real ground-station placement (bench/sat_traces.json).

Stations hold the hours they received, encoded by their own software (alternating hourly maps / tiles, as in
sim/e_hetero.py build_sat); downstream mirrors hold contiguous windows of re-encoded variants. For every hour:
holders usable under VALUE identity (all) vs BYTE identity (one swarm per encoding; the client picks the best one).
Availability of the month under independent site failures (probability p) and under regional outages (a random
longitude band of width w degrees down), with and without value-level RS parity volunteers (k data members per
stripe, interleaved over the month; each volunteer holds one row = 1/k of the data).

python bench/avail_mc.py [--trials 2000 --mirrors 12 --out bench/avail_mc.json]
"""
import argparse
import json
import random

import numpy as np

ENCS = ("maps-lz4", "tiles-zstd")
MIRROR_ENCS = ("6h-lz4", "ts-zstd")


def placement(tr, n_mirrors, rng, hours):
    sites = []
    for i, s in enumerate(tr["stations"]):
        m = np.zeros(hours, bool)
        m[[h for h in s["hours"] if h < hours]] = True
        sites.append({"enc": ENCS[i % 2], "lon": s["lon"], "hold": m, "kind": "station"})
    for _ in range(n_mirrors):
        enc = rng.choice(MIRROR_ENCS)
        span = int(hours * rng.uniform(0.2, 0.6))
        lo = rng.randint(0, hours - span)
        m = np.zeros(hours, bool)
        if enc == "6h-lz4":
            m[lo:lo + span:6] = True  # 6-hourly copy: every 6th hour
        else:
            m[lo:lo + span] = True
        sites.append({"enc": enc, "lon": rng.uniform(-180, 180), "hold": m, "kind": "mirror"})
    return sites


def byte_swarms(sites):
    """6-hourly lz4 maps have the same bytes per step as hourly lz4 maps (sec. 3): one byte swarm."""
    fam = {"maps-lz4": "lz4maps", "6h-lz4": "lz4maps", "tiles-zstd": "tiles", "ts-zstd": "ts"}
    out = {}
    for i, s in enumerate(sites):
        out.setdefault(fam[s["enc"]], []).append(i)
    return out


def avail(hold, alive):
    """fraction of hours held by at least one live site"""
    return float((hold[alive].any(0)).mean()) if alive.any() else 0.0


def with_parity(have, k, rows_alive):
    """value-level RS over the month: stripes of k hours interleaved with stride D = hours/k; a stripe with m
    missing members is rebuilt iff m <= surviving parity rows (sec. 3.3)."""
    H = len(have)
    D = -(-H // k)
    got = have.copy()
    for s in range(D):
        mem = [s + i * D for i in range(k) if s + i * D < H]
        miss = [h for h in mem if not have[h]]
        if miss and len(miss) <= rows_alive:
            got[miss] = True
    return float(got.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="bench/sat_traces.json")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--mirrors", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--volunteers", type=int, default=4, help="parity volunteers (4 x 1/8 = +0.5 replica)")
    ap.add_argument("--out", default="bench/avail_mc.json")
    a = ap.parse_args()
    tr = json.load(open(a.traces))
    hours = tr["days"] * 24
    rng = random.Random(0)
    sites = placement(tr, a.mirrors, rng, hours)
    hold = np.stack([s["hold"] for s in sites])
    swarms = byte_swarms(sites)
    rep_val = hold.sum(0)
    rep_byte = np.max([hold[idx].sum(0) for idx in swarms.values()], axis=0)
    red = {"sites": len(sites), "byte_swarms": {k: len(v) for k, v in swarms.items()},
           "holders_per_hour_value": {"median": float(np.median(rep_val)), "min": int(rep_val.min())},
           "holders_per_hour_best_byte_swarm": {"median": float(np.median(rep_byte)), "min": int(rep_byte.min())}}
    print(json.dumps(red))
    nprng = np.random.default_rng(1)
    res = {"redundancy": red, "independent": {}, "regional": {}}
    for p in (0.5, 0.7, 0.8, 0.9, 0.95):
        acc = {"value": [], "byte_best": [], "value+rs": [], "byte_best+rs": []}
        for _ in range(a.trials):
            alive = nprng.random(len(sites)) > p
            rows = int((nprng.random(a.volunteers) > p).sum())
            hv = hold[alive].any(0) if alive.any() else np.zeros(hours, bool)
            acc["value"].append(float(hv.mean()))
            acc["value+rs"].append(with_parity(hv, a.k, rows))
            best = max(swarms.values(), key=lambda idx: avail(hold[idx], alive[idx]))
            hb = hold[best][alive[best]].any(0) if alive[best].any() else np.zeros(hours, bool)
            acc["byte_best"].append(float(hb.mean()))
            acc["byte_best+rs"].append(with_parity(hb, a.k, rows))  # generous: byte parity of the chosen swarm
        res["independent"][p] = {m: round(float(np.mean(v)), 4) for m, v in acc.items()}
        print("p", p, res["independent"][p], flush=True)
    for w in (30, 60, 90, 120):
        acc = {"value": [], "byte_best": []}
        for _ in range(a.trials):
            c = nprng.uniform(-180, 180)
            down = np.array([abs((s["lon"] - c + 180) % 360 - 180) < w / 2 for s in sites])
            alive = ~down
            acc["value"].append(avail(hold, alive))
            acc["byte_best"].append(max(avail(hold[idx], alive[idx]) for idx in swarms.values()))
        res["regional"][w] = {m: round(float(np.mean(v)), 4) for m, v in acc.items()}
        print("band", w, res["regional"][w], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
