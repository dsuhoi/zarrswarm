"""Planner (max-flow, relay bottlenecks) and JLPS self-checks, plus a region download through the API."""
import runpy


def test_plan_selfcheck():
    runpy.run_module("zarrswarm.plan", run_name="__main__")


def test_jlps_selfcheck():
    runpy.run_module("zarrswarm.jlps", run_name="__main__")


def test_aqp_vas_coverage():
    runpy.run_module("zarrswarm.aqp", run_name="__main__")


def test_capt_selfcheck():
    runpy.run_module("zarrswarm.capt", run_name="__main__")


def test_parity_selfcheck():
    runpy.run_module("zarrswarm.parity", run_name="__main__")


def test_gf256_reed_solomon_selfcheck():
    runpy.run_module("zarrswarm.gf", run_name="__main__")


def test_plan_against_brute_force():
    """Random small instances (with shared relays) vs exhaustive integer optimum: every chunk goes to one of its
    holders, the reported makespan is that of the returned schedule, and rounding + local repair stays close
    to the optimum."""
    import itertools
    import random
    from zarrswarm.plan import plan
    ratios = []
    for trial in range(600):
        rnd = random.Random(trial)
        P = [f"p{i}" for i in range(rnd.randint(1, 3))]
        bw = {p: rnd.choice([0.5, 1, 2, 5]) for p in P}
        via, via_bw = {}, {}
        if len(P) > 1 and rnd.random() < 0.4:
            via, via_bw = {p: "r" for p in rnd.sample(P, 2)}, {"r": rnd.choice([0.5, 1, 3])}
        ch = {k: (rnd.randint(1, 5), tuple(rnd.sample(P, rnd.randint(1, len(P))))) for k in range(rnd.randint(1, 7))}
        asg, T = plan(ch, bw, via=via, via_bw=via_bw)
        got = {k: p for p, ks in asg.items() for k in ks}
        assert sorted(got) == sorted(ch) and all(got[k] in ch[k][1] for k in ch)

        def span(a):
            load, rl = {}, {}
            for k, p in a.items():
                load[p] = load.get(p, 0) + ch[k][0]
                if p in via:
                    rl[via[p]] = rl.get(via[p], 0) + ch[k][0]
            return max([load[p] / bw[p] for p in load] + [rl[r] / via_bw[r] for r in rl])
        keys = list(ch)
        opt = min(span(dict(zip(keys, c))) for c in itertools.product(*[ch[k][1] for k in keys]))
        assert opt - 1e-9 <= T <= span(got) * (1 + 2e-3) + 1e-9  # T is the makespan of the schedule returned
        ratios.append(span(got) / opt)
    assert max(ratios) < 1.6 and sum(r > 1.2 for r in ratios) <= 3, (max(ratios), sum(r > 1.2 for r in ratios))


def test_plan_is_invariant_to_catalogue_order():
    """The same catalogue formerly produced 0.90 or 0.52 s after reversing chunk entries."""
    from zarrswarm.plan import plan
    chunks = {"a": (450000, ("p0", "p1")), "b": (450000, ("p2",)),
              "c": (650000, ("p0", "p1", "p2"))}
    bw = {"p0": 1e6, "p1": 0.5e6, "p2": 5e6}
    expected = plan(chunks, bw, client_bw=3e6)
    permuted = {k: (size, tuple(reversed(holders)))
                for k, (size, holders) in reversed(list(chunks.items()))}
    assert plan(permuted, dict(reversed(list(bw.items()))), client_bw=3e6) == expected


def test_idle_holder_survives_until_assigned_worker_starts(tmp_path, monkeypatch):
    """An initially idle holder must stay available to hedge a stalled primary."""
    import asyncio
    import struct
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from zarrswarm import node as mod

    async def run():
        # Match _download's set construction and put the unassigned holder first.
        holders = ['holder-a', 'holder-b']
        idle, primary = list({p for p in holders})
        payload, key = b'x' * 512, 'v@1/0'
        n = mod.Node(tmp_path / 'receiver')
        n.xt1 = False
        n.hints = {p: 1e6 for p in holders}
        calls, cancelled = [], asyncio.Event()

        @asynccontextmanager
        async def post(url, **kwargs):
            peer = url.split('/')[2]
            calls.append(peer)
            async def readexactly(size):
                if size == 4:
                    return struct.pack('>I', len(payload))
                if peer == primary:
                    await asyncio.Event().wait()  # live request, no completed chunk
                return payload
            try:
                yield SimpleNamespace(status=200, content=SimpleNamespace(readexactly=readexactly))
            finally:
                if peer == primary:
                    cancelled.set()

        async def bandwidth(*args):
            return {primary: 1e9, idle: 1e6}, {}, {}
        n.session = SimpleNamespace(post=post)
        n._bw_model = bandwidth
        n._verify_store = lambda *args: ('cid', len(payload))
        n._index = lambda *args: 'verified-path'
        monkeypatch.setattr(mod.planmod, 'plan', lambda *args, **kwargs: ({primary: [key]}, 0.001))
        v = {'arrays': {}, 'best': {key: {'src': [(p, 'cid', len(payload)) for p in holders]}},
             'addrs': {p: 'http://' + p for p in holders}}
        result = await asyncio.wait_for(n._download('g', v, [key], None), timeout=1)
        assert result == {key: 'verified-path'}
        assert calls == [primary, idle]
        assert cancelled.is_set()

    asyncio.run(run())


def test_completed_workers_are_removed_from_wait_set(tmp_path, monkeypatch):
    """An idle worker finishing must not turn verification into an event-loop spin."""
    import asyncio
    import struct
    import time
    import aiohttp
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from zarrswarm import node as mod

    async def run():
        payload, key = b'x' * 512, 'v@1/0'
        n = mod.Node(tmp_path / 'receiver')
        n.xt1 = False
        n.hints = {'a': 1e6, 'b': 1e6}
        primary, alternative = list({p for p in n.hints})
        monkeypatch.setitem(n._download.__func__.__globals__, 'PEER_MAX_ERRORS', 1)
        monkeypatch.setattr(mod.planmod, 'plan', lambda *args, **kwargs: ({primary: [key]}, 0.001))
        @asynccontextmanager
        async def post(url, **kwargs):
            if url.split('/')[2] == primary:
                raise aiohttp.ClientError('permanent holder failure')
            async def readexactly(size):
                return struct.pack('>I', len(payload)) if size == 4 else payload
            yield SimpleNamespace(status=200, content=SimpleNamespace(readexactly=readexactly))
        async def bandwidth(*args):
            return n.hints, {}, {}
        def verify(*args):
            time.sleep(0.08)
            return 'cid', len(payload)
        n.session = SimpleNamespace(post=post)
        n._bw_model, n._verify_store = bandwidth, verify
        n._index = lambda *args: 'verified-path'
        waits, real_wait = [], asyncio.wait
        async def wait(tasks, **kwargs):
            waits.append(len(tasks))
            return await real_wait(tasks, **kwargs)
        monkeypatch.setattr(mod.asyncio, 'wait', wait)
        v = {'arrays': {}, 'best': {key: {'src': [(p, 'cid', len(payload)) for p in n.hints]}},
             'addrs': {p: 'http://' + p for p in n.hints}}
        assert await n._download('g', v, [key], None) == {key: 'verified-path'}
        assert len(waits) <= 9, len(waits)  # at most one retirement per worker plus watcher

    asyncio.run(run())


def test_lattice_value_identity():
    """Copies of the same quantized measurements decoded by different software (deviation a deterministic function
    of the code, < half a step) share one value id; shifts by a step or a unit offset, and genuinely different data,
    do not; non-float data keep exact identity; float32 vs float64 coordinates of one axis coincide."""
    import numpy as np
    from zarrswarm.codec import same_vcid, vcid_of
    from zarrswarm.scan import coord_id
    rng = np.random.default_rng(0)
    for trial in range(40):
        k = rng.integers(0, 50000, (3, 40, 50))
        R, s = 200 + rng.random() * 50, 2.0 ** int(rng.integers(-12, -1))
        dev = np.random.default_rng(trial).uniform(-0.4, 0.4, 50000)
        a, b = R + k * s, R + k * s + dev[k] * s
        va = vcid_of(a, 0)
        assert same_vcid(va, vcid_of(b, 0))
        assert not same_vcid(va, vcid_of(a + s, 0)) and not same_vcid(va, vcid_of(a - 273.15, 0))
        c = a.copy()
        c[1, 3, 4] += 2 * s  # one value genuinely different
        assert not same_vcid(va, vcid_of(c, 0))
    ints = rng.integers(0, 9, (4, 4))
    assert vcid_of(ints) == vcid_of(ints.copy()) and not vcid_of(ints).startswith("L:")
    lat = np.arange(90, -90.25, -0.25)
    assert coord_id(lat.astype("f4")) == coord_id(lat) != coord_id(lat + 0.25)
