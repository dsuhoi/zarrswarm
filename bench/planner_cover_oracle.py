"""Joint cover/holder controls on small fixed graphs, with exhaustive integral optima.

This checks the planner cost model, not measured network time. Every inclusion-minimal complete cover is
enumerated; removing an unnecessary chunk cannot increase the optimum with nonnegative transfer costs.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import statistics
import sys
from functools import reduce
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zarr_torrent import jlps as solver, plan
from zarr_torrent.scan import chunk_g


def fixture(seed):
    rng = random.Random(seed)
    peers = ["p0", "p1", "p2"]
    bw = {p: rng.choice((0.5, 1, 2, 5)) * 1e6 for p in peers}
    if seed % 5 == 0:
        bw["p0"] = 1e12
    via = {p: "relay" for p in peers[1:]} if seed % 3 == 0 else {}
    rb = {"relay": rng.choice((0.5, 1, 3)) * 1e6} if via else {}
    arrays = {"v": {"dims": ["time"], "taxis": 0, "layouts": {}}}
    best = {}
    for name, width in (("fine", 1), ("coarse", rng.choice((2, 3, 4)))):
        li = {"chunks": [width], "phase": rng.randrange(width), "docs": {".zarray": {"shape": [8]}}}
        arrays["v"]["layouts"][name] = li
        for c in range(8):
            lo, hi = chunk_g(li, 0, c)
            if hi <= 2 or lo >= 6:
                continue
            size = rng.randint(2, 10) * 1e5 * width
            holders = sorted(rng.sample(peers, rng.randint(1, 3)))
            best[f"v@{name}/{c}"] = {"src": [(p, "cid", size) for p in holders]}
    return arrays, best, bw, via, rb, rng.choice((0.5, 1, 3, 30)) * 1e6


def span(assignment, chunks, bw, via, rb, receiver):
    load, relay = {}, {}
    received = 0.0
    for key, peer in assignment.items():
        size = chunks[key][0]
        if bw[peer] < 1e11:
            received += size
        load[peer] = load.get(peer, 0.0) + size
        if peer in via:
            r = via[peer]
            relay[r] = relay.get(r, 0.0) + size
    return max([0.0, received / receiver] + [v / bw[p] for p, v in load.items()]
               + [v / rb[r] for r, v in relay.items()])


def minimal_covers(tcs):
    entries = list(tcs.items())
    masks = [sum(1 << (t - 2) for t in range(tc["a"], tc["b"])) for _, tc in entries]
    covers = []
    for bits in range(1, 1 << len(entries)):
        chosen = [i for i in range(len(entries)) if bits >> i & 1]
        union = lambda indices: reduce(int.__or__, (masks[i] for i in indices), 0)
        if union(chosen) == 15 and all(union([j for j in chosen if j != i]) != 15 for i in chosen):
            covers.append([entries[i][0] for i in chosen])
    assert covers
    return covers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "Run with PYTHONHASHSEED=0"
    settings = [("default", 24, 0.5, 0.1, 0.05), ("1 iteration", 1, 0.5, 0.1, 0.05),
                ("8 iterations", 8, 0.5, 0.1, 0.05), ("epsilon 0.1", 24, 0.1, 0.1, 0.05),
                ("epsilon 1", 24, 1, 0.1, 0.05), ("zero slack", 24, 0.5, 0, 0.05),
                ("slack 0.25", 24, 0.5, 0.25, 0.05), ("zero overhead", 24, 0.5, 0.1, 0),
                ("200ms overhead", 24, 0.5, 0.1, 0.2)]
    result = {"protocol": f"{args.cases} seeded three-peer graphs; two layouts; four requested samples; complete catalogue; exhaustive cover and holder assignment; identical graph and cost model for each comparison.",
              "hash_seed": 0, "source_sha256": {}, "rows": [], "summary": {}}
    root = Path(__file__).resolve().parents[1]
    for name in ("zarr_torrent/jlps.py", "zarr_torrent/plan.py", "zarr_torrent/scan.py", "bench/planner_cover_oracle.py"):
        result["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for seed in range(args.cases):
        arrays, best, bw, via, rb, receiver = fixture(seed)
        tcs = solver.time_chunks("v", arrays, best, 2, 6)
        covers = minimal_covers(tcs)
        local = frozenset(p for p in bw if bw[p] >= 1e11)
        for label, iterations, epsilon, slack, overhead in settings:
            solver.CLIENT_BW, solver.SLACK, solver.CHUNK_OVERHEAD_S = receiver, slack, overhead
            rates = sorted(b for b in bw.values() if b < 1e11)
            charge = overhead * rates[len(rates) // 2]
            chunks = {k: (size + charge, hs) for tc in tcs.values() for k, (size, hs) in tc["tiles"].items()}
            optimum = float("inf")
            for cover in covers:
                keys = [k for tc in cover for k in tcs[tc]["tiles"]]
                for holders in itertools.product(*(chunks[k][1] for k in keys)):
                    optimum = min(optimum, span(dict(zip(keys, holders)), chunks, bw, via, rb, receiver))
            keys, assignment, predicted, info = solver.jlps("v", arrays, best, 2, 6, bw, iters=iterations,
                                                          eps=epsilon, via=via, via_bw=rb)
            selected = {k: p for p, ks in assignment.items() for k in ks}
            assert set(keys) == set(selected)
            actual = span(selected, chunks, bw, via, rb, receiver)
            assert actual <= predicted + 1e-9 and predicted <= actual * 1.002 + 1e-9
            assert solver.region_coverage("v", arrays, best, keys, 2, 6)["missing_samples"] == 0
            byte_keys = solver.bytes_greedy_cover("v", arrays, best, 2, 6)
            _, byte_prediction = plan.plan({k: chunks[k] for k in byte_keys}, bw, via=via,
                                                       via_bw=rb, client_bw=receiver, local=local)
            assert predicted <= (1 + slack) * byte_prediction + 1e-7, (seed, label, predicted, byte_prediction)
            assert actual >= optimum - 1e-7
            ratio = actual / optimum if optimum else 1.0
            result["rows"].append({"seed": seed, "setting": label, "optimum_s": optimum,
                                   "JLPS_model_s": actual, "JLPS_prediction_s": predicted,
                                   "reference_model_s": byte_prediction,
                                   "ratio_to_optimum": ratio, "candidate_covers": info["covers"],
                                   "complete_minimal_covers": len(covers)})
    for label, *_ in settings:
        ratios = [r["ratio_to_optimum"] for r in result["rows"] if r["setting"] == label]
        result["summary"][label] = {"cases": len(ratios), "median_ratio": statistics.median(ratios),
                                    "max_ratio": max(ratios), "above_1_2": sum(r > 1.2 for r in ratios)}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
