"""Planner (max-flow, relay bottlenecks) and JLPS self-checks, plus a region download through the API."""
import runpy


def test_plan_selfcheck():
    runpy.run_module("zarr_torrent.plan", run_name="__main__")


def test_jlps_selfcheck():
    runpy.run_module("zarr_torrent.jlps", run_name="__main__")


def test_aqp_vas_coverage():
    runpy.run_module("zarr_torrent.aqp", run_name="__main__")


def test_capt_selfcheck():
    runpy.run_module("zarr_torrent.capt", run_name="__main__")


def test_parity_selfcheck():
    runpy.run_module("zarr_torrent.parity", run_name="__main__")


def test_gf256_reed_solomon_selfcheck():
    runpy.run_module("zarr_torrent.gf", run_name="__main__")


def test_plan_against_brute_force():
    """Random small instances (with shared relays) vs exhaustive integer optimum: every chunk goes to one of its
    holders, the reported makespan is that of the returned schedule, and rounding + local repair stays close
    to the optimum."""
    import itertools
    import random
    from zarr_torrent.plan import plan
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


def test_lattice_value_identity():
    """Copies of the same quantized measurements decoded by different software (deviation a deterministic function
    of the code, < half a step) share one value id; shifts by a step or a unit offset, and genuinely different data,
    do not; non-float data keep exact identity; float32 vs float64 coordinates of one axis coincide."""
    import numpy as np
    from zarr_torrent.codec import same_vcid, vcid_of
    from zarr_torrent.scan import coord_id
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
