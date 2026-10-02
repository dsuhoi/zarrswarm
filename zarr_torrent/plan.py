"""Makespan-optimal multi-source chunk assignment.

Chunks are grouped by their *availability signature* (the exact set of peers holding
them). Because replicas cover contiguous coordinate ranges, the number of signatures is
tiny compared with the number of chunks, so the flow network has O(#signatures + #peers)
nodes regardless of request size. Binary search on the makespan T with a max-flow
feasibility test gives the optimal fractional schedule; rounding loses < 1 chunk per peer.
"""
from collections import defaultdict, deque

INF = float("inf")


def maxflow(cap: dict, s, t) -> tuple[float, dict]:
    """Dinic (level graph + blocking flow) on array adjacency. cap: {u: {v: capacity}}.
    Returns (value, flow[u][v]). Edmonds-Karp augmented one path per BFS and dominated planning time on
    requests spanning thousands of time chunks (0.44 s per call x hundreds of calls)."""
    ids: dict = {}
    for u, vs in cap.items():
        ids.setdefault(u, len(ids))
        for v in vs:
            ids.setdefault(v, len(ids))
    if s not in ids or t not in ids:
        return 0.0, defaultdict(lambda: defaultdict(float))
    n = len(ids)
    to, res, head, nxt = [], [], [-1] * n, []

    def add(u, v, c):
        for a_, b_, c_ in ((u, v, c), (v, u, 0.0)):
            to.append(b_)
            res.append(c_)
            nxt.append(head[a_])
            head[a_] = len(to) - 1
    orig = []
    for u, vs in cap.items():
        for v, c in vs.items():
            orig.append((u, v, len(to)))
            add(ids[u], ids[v], float(c))
    S, T = ids[s], ids[t]
    EPS = 1e-9
    total = 0.0
    while True:
        level = [-1] * n
        level[S] = 0
        q = deque([S])
        while q:
            u = q.popleft()
            e = head[u]
            while e != -1:
                if res[e] > EPS and level[to[e]] < 0:
                    level[to[e]] = level[u] + 1
                    q.append(to[e])
                e = nxt[e]
        if level[T] < 0:
            break
        it = head[:]
        while True:  # blocking flow by iterative DFS
            stack, path = [S], []
            pushed = 0.0
            while stack:
                u = stack[-1]
                if u == T:
                    f = min(res[e] for e in path)
                    for e in path:
                        res[e] -= f
                        res[e ^ 1] += f
                    pushed = f
                    break
                e = it[u]
                while e != -1 and not (res[e] > EPS and level[to[e]] == level[u] + 1):
                    e = nxt[e]
                it[u] = e
                if e == -1:
                    stack.pop()
                    if path:
                        path.pop()
                        it[stack[-1]] = nxt[it[stack[-1]]] if stack else -1
                    continue
                stack.append(to[e])
                path.append(e)
            if pushed <= EPS:
                break
            total += pushed
    flow = defaultdict(lambda: defaultdict(float))
    for u, v, e in orig:
        f = res[e ^ 1]  # flow on an edge = residual of its reverse
        flow[u][v] += f
        flow[v][u] -= f
    return total, flow


def _greedy(chunks, bw):
    load = defaultdict(float)
    out = defaultdict(list)
    for k, (size, peers) in sorted(chunks.items(), key=lambda kv: -kv[1][0]):
        p = min(peers, key=lambda p: (load[p] + size) / bw[p])
        load[p] += size
        out[p].append(k)
    return dict(out), max((load[p] / bw[p] for p in load), default=0.0)


def plan(chunks: dict, bw: dict, max_groups: int = 400, via: dict | None = None,
         via_bw: dict | None = None, client_bw: float | None = None, local: frozenset = frozenset()) -> tuple[dict, float]:
    """chunks: {key: (size_bytes, peers)}; bw: {peer: bytes/s}.
    via: {peer: bottleneck id} for peers sharing an upstream (e.g. NAT'd peers behind one relay),
    via_bw: {bottleneck id: bytes/s}. client_bw: the receiver's own rate (download + verification by value): every
    byte from a remote peer passes it, so it is one more shared bottleneck; peers in `local` (the receiver itself)
    bypass it. Returns ({peer: [keys]}, estimated makespan seconds)."""
    via, via_bw = via or {}, via_bw or {}
    groups = defaultdict(list)
    for k, (size, peers) in chunks.items():
        groups[frozenset(peers)].append((k, size))
    if not groups:
        return {}, 0.0
    if len(groups) > max_groups:  # ponytail: pathological fragmentation -> LPT greedy
        return _greedy(chunks, bw)
    glist = list(groups.items())
    vol = [sum(s for _, s in ks) for _, ks in glist]
    total = float(sum(vol))
    peers = set().union(*groups)

    def solve(T):
        cap = defaultdict(dict)
        for i, (g, _) in enumerate(glist):
            cap["s"][("g", i)] = float(vol[i])
            for p in g:
                cap[("g", i)][("p", p)] = INF
        sink = "c" if client_bw else "t"
        if client_bw:
            cap["c"]["t"] = client_bw * T
        for p in peers:
            r = via.get(p)
            dst = "t" if p in local else sink
            if r is not None and r in via_bw:
                cap[("p", p)][("r", r)] = bw[p] * T
                cap[("r", r)][dst] = via_bw[r] * T
            else:
                cap[("p", p)][dst] = bw[p] * T
        return maxflow(cap, "s", "t")

    lo = total / (sum(bw[p] for p in peers if via.get(p) not in via_bw) +
                  sum(via_bw[r] for r in {via.get(p) for p in peers} if r in via_bw))  # aggregate-rate bound
    hi = total / min([bw[p] for p in peers] + [via_bw[r] for r in {via.get(p) for p in peers} if r in via_bw])
    if client_bw:
        remote = sum(v for (g, _), v in zip(glist, vol) if not (g & local))
        lo = max(lo, remote / client_bw)
        hi = max(hi, total / client_bw + lo)
    while hi - lo > 1e-3 * hi:  # relative precision; was 40 fixed halvings
        mid = (lo + hi) / 2
        if solve(mid)[0] >= total * (1 - 1e-9):
            hi = mid
        else:
            lo = mid
    _, f = solve(hi)
    out = defaultdict(list)
    for i, (g, ks) in enumerate(glist):
        quota = {p: f[("g", i)][("p", p)] for p in g}
        for k, s in sorted(ks, key=lambda x: -x[1]):
            p = max(quota, key=quota.get)
            out[p].append(k)
            quota[p] -= s
    asg = _repair(out, chunks, bw, via, via_bw)
    # the INTEGRAL schedule's makespan, not the fractional bound `hi`: covers are compared by it, and whole chunks on
    # slow peers made a station-heavy cover look 1.5x faster than it ran (8.4 s planned, 12-19 s measured)
    return asg, max(hi, makespan_of(asg, chunks, bw, via, via_bw, client_bw, local))


def makespan_of(asg, chunks, bw, via=None, via_bw=None, client_bw=None, local=frozenset()) -> float:
    """Finish time of an integral assignment: slowest peer, shared relay, or the receiver."""
    via, via_bw = via or {}, via_bw or {}
    load, rload, remote = defaultdict(float), defaultdict(float), 0.0
    for p, ks in asg.items():
        for k in ks:
            load[p] += chunks[k][0]
            if via.get(p) in via_bw:
                rload[via[p]] += chunks[k][0]
            if p not in local:
                remote += chunks[k][0]
    t = max([load[p] / bw[p] for p in load] + [rload[r] / via_bw[r] for r in rload] + [0.0])
    return max(t, remote / client_bw) if client_bw else t


def _repair(out, chunks, bw, via, via_bw, max_moves: int = 2000):
    """Rounding the fractional flow can leave a split chunk on a slow peer (fuzz: up to 2.1x the integer optimum
    on tiny instances). Local search: move a chunk off the bottleneck (peer or shared relay) to another holder
    while the makespan drops. Chunks with equal (size, holders) are interchangeable, so one move is tried per
    such bucket: cost per step ~ buckets x holders x peers, independent of the chunk count.
    ponytail: first-improvement single moves (no swaps), bounded steps; exact ILP only if this ever matters."""
    rel = lambda p: via.get(p) if via.get(p) in via_bw else None
    load, rload = defaultdict(float), defaultdict(float)
    bucket = defaultdict(lambda: defaultdict(list))  # peer -> (size, holders) -> keys
    for p, ks in out.items():
        for k in ks:
            size, holders = chunks[k]
            bucket[p][(size, tuple(holders))].append(k)
            load[p] += size
            if rel(p):
                rload[rel(p)] += size
    fin = lambda p: max(load[p] / bw[p], rload[rel(p)] / via_bw[rel(p)] if rel(p) else 0.0)

    def makespan():
        return max(fin(p) for p in load if load[p] > 0)

    def shift(src, q, size, sign):
        load[src] -= sign * size
        load[q] += sign * size
        if rel(src):
            rload[rel(src)] -= sign * size
        if rel(q):
            rload[rel(q)] += sign * size
    for _ in range(max_moves):
        cur = makespan()
        worst = max((p for p in load if load[p] > 0), key=fin)
        srcs = [p for p in bucket if p == worst or (rel(worst) and rel(p) == rel(worst))]
        best = None
        for src in srcs:
            for (size, holders), ks in bucket[src].items():
                if not ks:
                    continue
                for q in holders:
                    if q == src:
                        continue
                    shift(src, q, size, 1)
                    new = makespan()
                    shift(src, q, size, -1)
                    if new < cur - 1e-12 and (best is None or new < best[0]):
                        best = (new, src, q, (size, holders))
        if best is None:
            break
        _, src, q, sig = best
        bucket[q][sig].append(bucket[src][sig].pop())
        shift(src, q, sig[0], 1)
    return {p: [k for ks in b.values() for k in ks] for p, b in bucket.items() if any(b.values())}

if __name__ == "__main__":
    # self-check: A is fast and holds everything, B slow and holds half, C only a quarter
    ch = {i: (1, ("A", "B") if i < 50 else ("A",) if i < 75 else ("A", "C")) for i in range(100)}
    bw = {"A": 2.0, "B": 1.0, "C": 1.0}
    asg, T = plan(ch, bw)
    # optimum: B takes x from first 50, C takes y from last 25, A rest: (100-x-y)/2 = x = y -> x=y=25
    assert abs(T - 25.0) < 2e-3 * 25.0, T  # binary search to 1e-3 relative
    assert sorted(len(v) for v in asg.values()) == [25, 25, 50], {p: len(v) for p, v in asg.items()}
    assert sorted(k for v in asg.values() for k in v) == list(range(100))
    # two peers behind one relay of capacity 1 are together no better than one peer
    ch = {i: (1, ("A", "B")) for i in range(40)}
    asg, T = plan(ch, {"A": 1.0, "B": 1.0}, via={"A": "r", "B": "r"}, via_bw={"r": 1.0})
    assert abs(T - 40.0) < 2e-3 * 40.0, T
    print("plan ok, makespan", round(T, 3))
