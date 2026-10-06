"""JLPS - Joint Layout-and-Peer Selection.

Given a request over the absolute time interval [g_lo, g_hi) of one variable, several chunk layouts present in
the swarm (different time chunking / phase / spatial tiling), their per-chunk holders and peer bandwidths,
choose WHICH source chunks to read (an interval cover of the time axis, possibly mixing layouts) and FROM WHOM,
minimising the download makespan. Read amplification (coarse chunks cover more than requested) trades off
against bandwidth (coarse layouts may sit on fast peers).

  oracle   : exact min-price interval cover by DP over the arrangement of all layout boundaries,
             price(time-chunk) = sum over its spatial tiles of size * min_{holder} lambda_p
  MW loop  : lambda_p *= 1 + eps * load_p / (b_p * T_ref)   (Garg-Koenemann style packing prices)
  finish   : every distinct cover produced is scored with the exact parametric max-flow (plan.py);
             the best one is returned together with its peer assignment.
Cost is independent of the spatial size and of the number of elements: O(iters * (#time-chunks * m)).
"""
from collections import defaultdict

from . import plan as planmod
from .scan import chunk_g, split_key


CHUNK_OVERHEAD_S = float(__import__("os").environ.get("ZT_CHUNK_OVERHEAD_MS", "50")) / 1e3
# the receiver's own rate (download + value verification, stored bytes/s): one more shared bottleneck (sec. 3.2)
CLIENT_BW = float(__import__("os").environ.get("ZT_CLIENT_MBPS", "30")) * 1e6
SLACK = float(__import__("os").environ.get("ZT_JLPS_SLACK", "0.1"))  # covers this close in makespan count as ties


def complete_for(li: dict, lattice: tuple[int, int]) -> bool:
    """A layout serves a request on lattice (S, O) (samples g = O mod S, in grid quanta) iff every requested
    sample is one of its samples: its stride divides S and the offsets agree (6 h from 1 h: yes; 1 h from 6 h: no)."""
    S, O = lattice
    s, o = li.get("stride", 1), li.get("soff", 0)
    return S % s == 0 and (O - o) % s == 0


def time_chunks(var: str, arrays: dict, best: dict, g_lo: int, g_hi: int, isel: dict | None = None,
                lattice: tuple[int, int] = (1, 0)) -> dict:
    """(layout, c) -> {"a": start g, "b": end g, "tiles": {key: (size, holders)}} overlapping [g_lo, g_hi) and,
    for non-time dims, the index boxes in `isel` ({dim: [i0, i1)}); only layouts complete for the request lattice."""
    a = arrays[var]
    T = a["taxis"]
    box = {a["dims"].index(d): (int(r[0]), int(r[1])) for d, r in (isel or {}).items() if d in a["dims"]}
    out = {}
    for k, b in best.items():
        n, lay, co = split_key(k)
        if n != var or lay not in a["layouts"] or not b["src"]:  # parity-restorable keys are added later
            continue
        li = a["layouts"][lay]
        if not complete_for(li, lattice):
            continue
        s, e = chunk_g(li, T, co[T])
        if b.get("nv"):
            e = min(e, s + b["nv"] * li.get("stride", 1))
        if e <= g_lo or s >= g_hi:
            continue
        if any(co[i] * li["chunks"][i] >= hi or (co[i] + 1) * li["chunks"][i] <= lo for i, (lo, hi) in box.items()):
            continue
        tc = out.setdefault((lay, co[T]), {"a": max(s, g_lo), "b": min(e, g_hi), "tiles": {}})
        tc["b"] = min(tc["b"], e, g_hi)
        tc["tiles"][k] = (b["src"][0][2], tuple(sorted(p for p, _, _ in b["src"])))
    # One tile is not a global map. Only complete spatial families may cover a time interval.
    return {key: tc for key, tc in out.items()
            if len(tc["tiles"]) == _required_tiles(a, a["layouts"][key[0]], isel)}


def _required_tiles(a, li, isel=None):
    docs = li.get("docs", {})
    meta = docs.get("zarr.json", docs.get(".zarray", {}))
    shape = meta.get("shape", li["chunks"])
    dims = a.get("dims", [str(i) for i in range(len(shape))])
    count = 1
    for i, (sz, cs) in enumerate(zip(shape, li["chunks"])):
        if i == a["taxis"]:
            continue
        lo, hi = (isel or {}).get(dims[i], (0, sz))
        if not 0 <= lo < hi <= sz:
            raise ValueError(f"invalid selection for {dims[i]}: {(lo, hi)}")
        count *= (hi - 1) // cs - lo // cs + 1
    return count


def region_coverage(var, arrays, best, keys, g_lo, g_hi, isel=None, lattice=(1, 0)):
    """Count requested time samples covered by complete spatial families, using valid lengths, not plan status.

    Complementary partial tiles from different layouts are conservatively reported incomplete.
    """
    a = arrays[var]
    selected = {k: dict(best[k], src=best[k].get("src") or [("local", "", 0)])
                for k in set(keys) if k in best and split_key(k)[0] == var}
    if a["taxis"] is None:
        counts = defaultdict(int)
        for k in selected:
            _, lay, _ = split_key(k)
            counts[lay] += 1
        covered = any(n == _required_tiles(a, a["layouts"][lay], isel) for lay, n in counts.items())
        return {"requested_samples": 1, "covered_samples": int(covered), "missing_samples": int(not covered)}
    S, O = lattice
    if S <= 0 or g_hi < g_lo:
        raise ValueError("invalid query interval or sample step")
    lo, hi = -(-(g_lo - O) // S), -(-(g_hi - O) // S)
    ivs = sorted((-(-(tc["a"] - O) // S), -(-(tc["b"] - O) // S))
                 for tc in time_chunks(var, arrays, selected, g_lo, g_hi, isel, lattice).values())
    covered, end = 0, lo
    for x, y in ivs:
        covered += max(0, y - max(end, x))
        end = max(end, y)
    requested = max(0, hi - lo)
    return {"requested_samples": requested, "covered_samples": covered, "missing_samples": requested - covered}


def _dp_cover(tcs: dict, g_lo: int, g_hi: int, price, lattice: tuple[int, int] = (1, 0)) -> list:
    """Min-price cover of the requested samples in [g_lo, g_hi) by intervals: a cell is free when no interval covers
    it or when it holds no sample of the request lattice (S, O) - e.g. the second between two 2 s volumes, which a
    single-volume chunk [g, g+1) leaves open but nobody asked for."""
    pts = sorted({g_lo, g_hi} | {t["a"] for t in tcs.values()} | {t["b"] for t in tcs.values()})
    idx = {x: i for i, x in enumerate(pts)}
    INF = float("inf")
    dist, back = [INF] * len(pts), [None] * len(pts)
    dist[0] = 0.0
    ivs = sorted(((t["a"], t["b"], key, price(t)) for key, t in tcs.items()), key=lambda x: x[1])
    covered = [False] * (len(pts) - 1)
    ending = defaultdict(list)  # interval end index -> intervals (was a scan of all intervals per point: O(P*I))
    for a, b, key, c in ivs:
        for i in range(idx[a], idx[b]):
            covered[i] = True
        ending[idx[b]].append((a, key, c))
    S, O = lattice
    needed = [-(-(pts[i] - O) // S) * S + O < pts[i + 1] for i in range(len(pts) - 1)]
    for i, nd in enumerate(needed):
        covered[i] = covered[i] and nd  # a cell without requested samples is as free as a hole
    for j in range(1, len(pts)):
        if not covered[j - 1] and dist[j - 1] < dist[j]:  # hole in every replica: skip for free
            dist[j], back[j] = dist[j - 1], (j - 1, None)
        for a, key, c in ending[j]:
            for i in range(idx[a], j):  # enter the interval anywhere inside it
                if dist[i] + c < dist[j]:
                    dist[j], back[j] = dist[i] + c, (i, key)
        if not covered[j - 1] and dist[j - 1] < dist[j]:
            dist[j], back[j] = dist[j - 1], (j - 1, None)
    chosen, j = [], len(pts) - 1
    while j > 0 and back[j] is not None:
        i, key = back[j]
        if key is not None:
            chosen.append(key)
        j = i
    return chosen


def jlps(var: str, arrays: dict, best: dict, g_lo: int, g_hi: int, bw: dict, iters: int = 24, eps: float = 0.5,
         via: dict | None = None, via_bw: dict | None = None, isel: dict | None = None,
         lattice: tuple[int, int] = (1, 0)) -> tuple[list[str], dict, float, dict]:
    """Returns (keys, assignment {peer: [keys]}, makespan estimate, info)."""
    tcs = time_chunks(var, arrays, best, g_lo, g_hi, isel, lattice)
    if not tcs:
        return [], {}, 0.0, {"covers": 0}
    byte_cover = tuple(sorted(_dp_cover(tcs, g_lo, g_hi,
        lambda t: sum(sz for sz, _ in t["tiles"].values()), lattice)))
    local = frozenset(p for p, b in bw.items() if b >= 1e11)
    # every chunk also costs a request, a disk read and a verification: charge it as bytes at the median rate, so
    # a cover of hundreds of small reads spread over many peers is not mistaken for a fast one
    rates = sorted(b for b in bw.values() if b < 1e11)
    ovh = CHUNK_OVERHEAD_S * (rates[len(rates) // 2] if rates else 0.0)
    tcs = {key: dict(t, tiles={k: (sz + ovh, hs) for k, (sz, hs) in t["tiles"].items()}) for key, t in tcs.items()}
    lam = {p: 1.0 / bw[p] for p in bw}
    holders = {p for t in tcs.values() for _, hs in t["tiles"].values() for p in hs}
    total_bw = sum(bw[p] for p in holders - local) or 1.0

    def price(t):
        return sum(sz * min(0.0 if p in local else lam[p] for p in hs) for sz, hs in t["tiles"].values())

    covers, seen = [byte_cover], {byte_cover}
    t_ref = None
    for _ in range(iters):
        cover = tuple(sorted(_dp_cover(tcs, g_lo, g_hi, price, lattice)))
        if cover not in seen:
            seen.add(cover)
            covers.append(cover)
        load = defaultdict(float)
        for key in cover:
            for sz, hs in tcs[key]["tiles"].values():
                load[min(hs, key=lambda p: 0.0 if p in local else lam[p])] += sz
        tot = sum(l for p, l in load.items() if p not in local)
        t_ref = t_ref or max(tot / total_bw, 1e-9)
        for p, l in load.items():
            if p not in local:
                lam[p] *= 1 + eps * l / (bw[p] * t_ref)
    def lower_bound(cover):  # no schedule beats all of its holders sending at full rate at once
        vol, hs = 0.0, set()
        for key in cover:
            for sz, h in tcs[key]["tiles"].values():
                if not local.intersection(h):
                    vol += sz
                    hs.update(h)
        cap = sum(bw[p] for p in hs if (via or {}).get(p) not in (via_bw or {})) + \
            sum((via_bw or {})[r] for r in {(via or {}).get(p) for p in hs} if r in (via_bw or {}))
        return vol / max(min(cap, CLIENT_BW) if CLIENT_BW else cap, 1e-9)
    best_T, scored = float("inf"), []
    for lb, cover in sorted((lower_bound(c), c) for c in covers):
        if lb >= best_T * (1 + SLACK):  # every remaining cover is provably outside the slack
            break
        chunks = {k: v for key in cover for k, v in tcs[key]["tiles"].items()}
        asg, T = planmod.plan(chunks, bw, via=via, via_bw=via_bw, client_bw=CLIENT_BW or None,
                              local=local)
        best_T = min(best_T, T)
        scored.append((T, sum(sz for sz, _ in chunks.values()), cover, asg))
    # predictions within SLACK of the best are indistinguishable under rate noise; reading fewer bytes from fewer
    # peers then has the shorter tail (a point series spread over 25 stations: 12.2 s planned, 57 s measured,
    # against 10 s for one time-series chunk planned at 12.3 s)
    T_, _, best_cover, best_asg = min((x for x in scored if x[0] <= best_T * (1 + SLACK)), key=lambda x: (x[1], x[0]))
    best_T = T_
    keys = [k for key in best_cover for k in tcs[key]["tiles"]]
    return keys, best_asg, best_T, {"covers": len(covers), "layouts": sorted({lay for lay, _ in best_cover})}


def bytes_greedy_cover(var: str, arrays: dict, best: dict, g_lo: int, g_hi: int, isel: dict | None = None,
                       lattice: tuple[int, int] = (1, 0)) -> list[str]:
    """Baseline (what a bandwidth-blind client does): min read amplification = min total bytes."""
    tcs = time_chunks(var, arrays, best, g_lo, g_hi, isel, lattice)
    cover = _dp_cover(tcs, g_lo, g_hi, lambda t: sum(sz for sz, _ in t["tiles"].values()), lattice)
    return [k for key in cover for k in tcs[key]["tiles"]]


if __name__ == "__main__":
    # self-check: exact layout "A" (24 h chunks) only on a slow peer; coarse "B" (168 h) on two fast peers.
    # request = 24 h -> bytes-greedy picks A (1 unit, slow: 1/0.1 = 10 s); JLPS picks B (7 units / 2.0 = 3.5 s).
    arrays = {"v": {"taxis": 0, "layouts": {"A": {"chunks": [24], "phase": 0}, "B": {"chunks": [168], "phase": 0}}}}
    best = {"v@A/7": {"src": [("slow", "c", 1.0)]}, "v@B/1": {"src": [("f1", "c", 7.0), ("f2", "c", 7.0)]}}
    bw = {"slow": 0.1, "f1": 1.0, "f2": 1.0}
    assert bytes_greedy_cover("v", arrays, best, 168, 192) == ["v@A/7"]
    keys, asg, T, info = jlps("v", arrays, best, 168, 192, bw)
    assert keys == ["v@B/1"] and T < 10, (keys, T)  # fractional LP bound 3.5 s vs 10 s for the exact layout
    # mixed cover: A covers [0,48) cheaply on a fast peer, only B covers [48,168)
    best2 = {"v@A/0": {"src": [("f1", "c", 1.0)]}, "v@A/1": {"src": [("f1", "c", 1.0)]},
             "v@B/0": {"src": [("f2", "c", 7.0)]}}
    keys, _, _, info = jlps("v", arrays, best2, 0, 168, {"f1": 1.0, "f2": 1.0})
    assert "v@B/0" in keys, keys
    print("jlps ok", info)
