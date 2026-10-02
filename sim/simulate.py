"""zarr-torrent swarm simulator: N real nodes (real DHT, relay, planner, HTTP) in one process with emulated
links (upload cap + latency), NAT'd peers behind relays, partial replicas, churn.

Experiments
  E1 strategies : makespan + load decentralization (Gini, max share, #contributors) for
                  maxflow (ours) | rarest (BitTorrent-like) | random | single (one main source)
  E2 availability: kill a fraction f of peers (incl. bootstrap/relay nodes), fresh client from a
                  surviving node: DHT lookup success, chunk availability, download success/time
  E3 mid-download churn: kill 30% of the serving peers while a download runs; must still complete
  E4 DHT load  : records stored per node, RPCs served per node (decentralization of the control plane)

python sim/simulate.py --peers 24 --out sim/results.json
"""
import argparse
import asyncio
import json
import math
import os
import random
import shutil
import socket
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# many nodes share one process here: keep per-node caches small
os.environ.setdefault("ZT_DECODED_MB", "32")
os.environ.setdefault("ZT_PAGE_CACHE_MB", "32")
import zarr_torrent as zt  # noqa: E402
from zarr_torrent.node import Node  # noqa: E402
from zarr_torrent.store import http, keys_for, open_view, wait_job  # noqa: E402

LAT, LON = np.linspace(90, -90, 91), np.arange(0, 360, 2.0)


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def field(times):
    h = times.values.astype("datetime64[h]").astype("int64").astype("float64")[:, None, None]
    lat, lon = np.deg2rad(LAT)[None, :, None], np.deg2rad(LON)[None, None, :]
    return (288 - 40 * np.sin(lat) ** 2 + 4 * np.cos(lat) * np.cos(2 * np.pi * h / 24 + lon)
            + np.sin(3 * lon + h / 37)).astype("float32")


def gini(xs):
    xs = sorted(x for x in xs)
    n, s = len(xs), sum(xs)
    if n == 0 or s == 0:
        return 0.0
    return sum((2 * i - n + 1) * x for i, x in enumerate(xs)) / (n * s)


class Swarm:
    def __init__(self, root: Path, n: int, n_boot: int, nat_frac: float, seed: int, strategy="maxflow"):
        self.root, self.rng = root, random.Random(seed)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.nodes: dict[str, Node] = {}
        self.meta: dict[str, dict] = {}
        boots = []
        for i in range(n_boot):  # several public bootstrap+relay nodes: no single point of failure
            p = port()
            name = f"boot{i}"
            self._start(name, dict(port=p, relay_server=True, bootstrap=boots[:], strategy=strategy,
                                   rate=self._rate(), latency=self._lat()))
            boots.append(f"http://127.0.0.1:{p}")
        self.boots = boots
        for i in range(n - n_boot):
            nat = self.rng.random() < nat_frac
            relay = self.rng.choice(boots) if nat else None
            self._start(f"p{i:02d}", dict(port=port(), bootstrap=self.rng.sample(boots, min(2, len(boots))),
                                          relay=relay, strategy=strategy, rate=self._rate(), latency=self._lat()))

    def _rate(self):  # heterogeneous uplinks, lognormal around 4 MB/s (0.5 .. 30 MB/s)
        return float(np.clip(self.rng.lognormvariate(math.log(4e6), 0.9), 0.5e6, 30e6))

    def _lat(self):
        return self.rng.uniform(0.002, 0.04)

    def run(self, coro, timeout=600):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def _start(self, name, kw):
        n = Node(self.root / f"h_{name}", ctl_port=port(), **kw)
        self.run(n.start())
        self.nodes[name] = n
        self.meta[name] = {"rate": kw["rate"], "nat": bool(kw.get("relay")), "alive": True}

    def ctl(self, name):
        return f"http://127.0.0.1:{self.nodes[name].ctl_port}"

    def kill(self, name):
        self.run(self.nodes[name].stop())
        self.meta[name]["alive"] = False

    def alive(self):
        return [n for n, m in self.meta.items() if m["alive"]]

    def stop(self):
        for n in self.alive():
            try:
                self.kill(n)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        shutil.rmtree(self.root, ignore_errors=True)
        # release per-node caches (decoded chunks, manifest pages) before the next swarm is built
        self.nodes.clear()
        time.sleep(0.2)
        try:
            self.loop.close()
        except RuntimeError:
            pass
        import gc
        gc.collect()


def make_replicas(sw: Swarm, days: int, replicas_per_peer: float, seed: int,
                  layouts=((24, 91, 180), (48, 46, 90))) -> tuple[str, pd.DatetimeIndex, dict]:
    """Every non-bootstrap peer holds a random contiguous window (5..40% of the period) in one of two layouts."""
    rng = random.Random(seed)
    times = pd.date_range("2020-01-01", periods=24 * days, freq="h")
    data = field(times)
    link, holders = None, {}
    for name in [n for n in sw.nodes if not n.startswith("boot")]:
        if rng.random() > replicas_per_peer:
            continue
        span = max(24, int(len(times) * rng.uniform(0.05, 0.4)) // 24 * 24)
        start = rng.randrange(0, len(times) - span + 1, 24)
        layout = layouts[rng.randrange(len(layouts))]
        p = sw.root / "data" / f"{name}.zarr"
        xr.Dataset({"t2m": (("time", "lat", "lon"), data[start:start + span])},
                   coords={"time": times[start:start + span], "lat": LAT, "lon": LON}).to_zarr(
            p, encoding={"t2m": {"chunks": layout}}, consolidated=False)
        link = http(sw.ctl(name), "POST", "/api/seed", {"path": str(p)})["link"]
        holders[name] = (start, span, layout)
    # guarantee full coverage with two extra seeders covering halves (so availability is well-defined)
    return link, times, holders


def fresh_client(sw: Swarm, tag: str, strategy: str) -> str:
    name = f"client_{tag}"
    sw._start(name, dict(port=port(), bootstrap=sw.rng.sample([b for b in sw.boots], 1),
                         relay=None, strategy=strategy, rate=None, latency=0.0))
    return name


def served_snapshot(sw):
    out = {}
    for n in sw.alive():
        try:
            out[n] = http(sw.ctl(n), "GET", "/api/status", timeout=10)["served"]["bytes"]
        except Exception:
            pass
    return out


def download(sw, client, link, t_limit=900):
    grid, view = open_view(link, sw.ctl(client))
    keys = keys_for(view, ["t2m"])
    t0 = time.perf_counter()
    jid = http(sw.ctl(client), "POST", "/api/download", {"grid": grid, "keys": keys, "label": "sim"})["job"]
    job = wait_job(sw.ctl(client), jid, every=0.2)
    return job, time.perf_counter() - t0, len(keys)


def e1_strategies(args, root):
    rows = []
    for strategy in args.e1_strategies:
        for rep in range(args.reps):
            sw = Swarm(root / f"e1_{strategy}_{rep}", args.peers, args.boot, args.nat, seed=100 + rep, strategy=strategy)
            try:
                link, times, holders = make_replicas(sw, args.days, args.replica_frac, seed=200 + rep)
                before = served_snapshot(sw)
                c = fresh_client(sw, "e1", strategy)
                job, secs, nkeys = download(sw, c, link)
                after = served_snapshot(sw)
                up = {n: after.get(n, 0) - before.get(n, 0) for n in after if n != c}
                contrib = [b for b in up.values() if b > 0]
                tot = sum(up.values())
                # correctness: whole union equals the generator where covered
                ds = zt.open_dataset(link, ctl=sw.ctl(c))
                got = ds.t2m.values
                ref = field(pd.DatetimeIndex(ds.time.values))
                ok = bool(np.array_equal(got[~np.isnan(got)], ref[~np.isnan(got)]))
                diag = {}
                pr, ac = job.get("pred") or {}, job.get("actual") or {}
                for kind in ("direct", "relay"):
                    r_ = [ac[p]["bytes"] / max(ac[p]["busy_s"], 1e-3) / max(pr[p]["bw"], 1)
                          for p in pr if p in ac and ac[p]["bytes"] and (pr[p]["via"] is not None) == (kind == "relay")]
                    diag[f"bw_ratio_{kind}"] = round(float(np.median(r_)), 2) if r_ else None
                ends = [a_["end_s"] for a_ in ac.values()]
                diag["T_err"] = round(max(ends) / job["plan_T"], 2) if ends and job.get("plan_T") else None
                diag["phases"] = job.get("phases")
                rows.append({"strategy": strategy, "rep": rep, "seconds": round(secs, 2), "chunks": job["done"], **diag,
                             "missing": job["missing"], "MB": round(job["bytes"] / 1e6, 1),
                             "MBps": round(job["bytes"] / 1e6 / secs, 2), "plan_T": job.get("plan_T"),
                             "gini_upload": round(gini(up.values()), 3),
                             "max_share": round(max(up.values()) / tot, 3) if tot else None,
                             "contributors": len(contrib), "seeders": len(holders), "values_ok": ok})
                print("E1", rows[-1], flush=True)
            finally:
                sw.stop()
    return rows


def e2_availability(args, root):
    rows = []
    for f in (0.0, 0.2, 0.4, 0.6):
        for rep in range(args.reps):
            sw = Swarm(root / f"e2_{f}_{rep}", args.peers, args.boot, args.nat, seed=300 + rep)
            try:
                link, times, holders = make_replicas(sw, args.days, args.replica_frac, seed=400 + rep)
                grid = link.removeprefix("zt://")
                full_view = http(sw.ctl(next(iter(holders))), "GET", f"/api/view/{grid}?refresh=1")
                n_total = sum(li["n"] for li in full_view["arrays"]["t2m"]["layouts"].values())
                victims = sw.rng.sample(list(sw.nodes), int(f * len(sw.nodes)))
                for v in victims:
                    sw.kill(v)
                survivors = sw.alive()
                # fresh client joins through any surviving bootstrap (or any survivor if all bootstraps died)
                entry = [b for b, n in zip(sw.boots, [f"boot{i}" for i in range(args.boot)]) if n in survivors] or \
                        [f"http://127.0.0.1:{sw.nodes[n].port}" for n in survivors if not sw.meta[n]["nat"]]
                c = f"client_e2"
                sw._start(c, dict(port=port(), bootstrap=entry[:1], strategy="maxflow", rate=None, latency=0.0))
                t0 = time.perf_counter()
                try:
                    v = http(sw.ctl(c), "GET", f"/api/view/{grid}?refresh=1", timeout=60)
                    lookup_ok, n_avail, npeers = True, sum(li["n"] for li in v["arrays"]["t2m"]["layouts"].values()), v["npeers"]
                except Exception:
                    lookup_ok, n_avail, npeers = False, 0, 0
                lookup_s = time.perf_counter() - t0
                job, secs = ({"done": 0, "missing": 0, "state": "no-peers"}, 0.0)
                if lookup_ok:
                    job, secs, _ = download(sw, c, link)
                # ground truth: which source chunks are still held by some survivor
                surv_holders = {h for h in holders if h in survivors}
                rows.append({"kill_frac": f, "rep": rep, "killed": len(victims),
                             "bootstraps_alive": sum(f"boot{i}" in survivors for i in range(args.boot)),
                             "seeders_alive": len(surv_holders), "seeders": len(holders),
                             "dht_lookup_ok": lookup_ok, "lookup_s": round(lookup_s, 2), "peers_found": npeers,
                             "chunk_availability": round(n_avail / max(n_total, 1), 3),
                             "downloaded": job["done"], "download_state": job["state"], "seconds": round(secs, 2)})
                print("E2", rows[-1], flush=True)
            finally:
                sw.stop()
    return rows


def e3_churn(args, root):
    rows = []
    for rep in range(args.reps):
        sw = Swarm(root / f"e3_{rep}", args.peers, args.boot, args.nat, seed=500 + rep)
        try:
            link, times, holders = make_replicas(sw, args.days, args.replica_frac, seed=600 + rep)
            c = fresh_client(sw, "e3", "maxflow")
            grid, view = open_view(link, sw.ctl(c))
            keys = keys_for(view, ["t2m"])
            jid = http(sw.ctl(c), "POST", "/api/download", {"grid": grid, "keys": keys, "label": "churn"})["job"]
            time.sleep(1.0)
            victims = sw.rng.sample(sorted(holders), max(1, int(0.3 * len(holders))))
            for v in victims:
                sw.kill(v)
            job = wait_job(sw.ctl(c), jid, every=0.2)
            # chunks that only the victims held are legitimately lost
            v2 = http(sw.ctl(c), "GET", f"/api/view/{grid}?refresh=1")
            rows.append({"rep": rep, "killed_mid_download": len(victims), "state": job["state"],
                         "done": job["done"], "total": job["total"], "missing_at_plan": job["missing"],
                         "local_after": v2["local"], "seconds": round(job["t1"] - job["t0"], 2)})
            print("E3", rows[-1], flush=True)
        finally:
            sw.stop()
    return rows


def e4_dht_load(args, root):
    sw = Swarm(root / "e4", args.peers, args.boot, args.nat, seed=700)
    try:
        link, _, holders = make_replicas(sw, args.days, args.replica_frac, seed=800)
        stats = {n: http(sw.ctl(n), "GET", "/api/status") for n in sw.alive()}
        vals = [s["dht_values"] for s in stats.values()]
        rpcs = [s["served"]["dht_rpcs"] for s in stats.values()]
        # every node must find the grid
        found = 0
        for n in sw.alive():
            try:
                v = http(sw.ctl(n), "GET", f"/api/peers/{link.removeprefix('zt://')}", timeout=60)
                found += bool(v["peers"])
            except Exception:
                pass
        row = {"nodes": len(stats), "dht_values_gini": round(gini(vals), 3), "dht_values_max": max(vals),
               "dht_values_median": statistics.median(vals), "dht_rpc_gini": round(gini(rpcs), 3),
               "dht_rpc_max_share": round(max(rpcs) / max(sum(rpcs), 1), 3),
               "lookup_success": f"{found}/{len(stats)}",
               "contacts_median": statistics.median(s["contacts"] for s in stats.values())}
        print("E4", row, flush=True)
        return [row]
    finally:
        sw.stop()


def e5_jlps(args, root):
    """Exact layout (24 h chunks) on slow peers vs coarse layout (168 h) on fast peers.
    Same requests downloaded by fresh clients with cover='bytes' (min read amplification) and cover='jlps'."""
    rows = []
    for rep in range(args.reps):
        sw = Swarm(root / f"e5_{rep}", 12, 2, 0.0, seed=900 + rep)
        try:
            times = pd.date_range("2020-01-01", periods=24 * 28, freq="h")
            data = field(times)
            names = [n for n in sw.nodes if not n.startswith("boot")]
            slow, fast = names[:6], names[6:10]
            link = None
            for grp, rate, layout in ((slow, 0.6e6, (24, 91, 180)), (fast, 15e6, (168, 91, 180))):
                for n in grp:
                    sw.nodes[n].rate = rate
                    p = sw.root / "data" / f"{n}.zarr"
                    xr.Dataset({"t2m": (("time", "lat", "lon"), data)}, coords={"time": times, "lat": LAT, "lon": LON}
                               ).to_zarr(p, encoding={"t2m": {"chunks": layout}}, consolidated=False)
                    link = http(sw.ctl(n), "POST", "/api/seed", {"path": str(p)})["link"]
            grid = link.removeprefix("zt://")
            rng = random.Random(rep)
            windows = [(d, d + 1) for d in rng.sample(range(1, 26), 4)] + [(2, 21)]
            for d0, d1 in windows:
                t0, t1 = str(times[24 * d0].date()), str(times[24 * d1].date())
                res = {}
                for cover in ("bytes", "jlps"):
                    c = f"client_{cover}_{d0}"
                    sw._start(c, dict(port=port(), bootstrap=sw.boots[:1], strategy="maxflow", rate=None, latency=0.0))
                    http(sw.ctl(c), "GET", f"/api/view/{grid}?refresh=1")
                    jid = http(sw.ctl(c), "POST", "/api/download", {"grid": grid, "cover": cover,
                                                                     "region": {"var": "t2m", "t0": t0, "t1": t1}})["job"]
                    job = wait_job(sw.ctl(c), jid, every=0.1)
                    ds = zt.open_dataset(link, ctl=sw.ctl(c))
                    sub = ds.t2m.sel(time=slice(t0, t1)).values
                    ref = field(pd.date_range(t0, pd.Timestamp(t1) + pd.Timedelta("23h"), freq="h"))
                    res[cover] = {"s": round(job["t1"] - job["t0"], 2), "MB": round(job["bytes"] / 1e6, 1),
                                  "layouts": job["cover"].get("layouts"), "ok": bool(np.array_equal(sub, ref))}
                    sw.kill(c)
                rows.append({"rep": rep, "days": d1 - d0 + 1, **{f"{k}_{m}": v for m, r in res.items() for k, v in r.items()},
                             "speedup": round(res["bytes"]["s"] / max(res["jlps"]["s"], 1e-3), 2)})
                print("E5", rows[-1], flush=True)
        finally:
            sw.stop()
    return rows


def e6_loss(args, root):
    """Lossy links: every node stalls a fraction of incoming data-port requests. Completion and time."""
    rows = []
    for loss in (0.0, 0.2, 0.4):
        for rep in range(args.reps):
            sw = Swarm(root / f"e6_{loss}_{rep}", args.peers, args.boot, args.nat, seed=1000 + rep)
            try:
                link, times, holders = make_replicas(sw, args.days, args.replica_frac, seed=1100 + rep)
                for n in sw.nodes.values():
                    n.loss = loss
                c = fresh_client(sw, "e6", "maxflow")
                t0 = time.perf_counter()
                job, secs, nkeys = download(sw, c, link)
                ds = zt.open_dataset(link, ctl=sw.ctl(c))
                got = ds.t2m.values
                ok = bool(np.array_equal(got[~np.isnan(got)], field(pd.DatetimeIndex(ds.time.values))[~np.isnan(got)]))
                rows.append({"loss": loss, "rep": rep, "state": job["state"], "done": job["done"], "total": job["total"],
                             "failed": job.get("failed", 0), "missing": job["missing"], "seconds": round(secs, 1),
                             "phases": job.get("phases"),
                             "values_ok": ok})
                print("E6", rows[-1], flush=True)
            finally:
                sw.stop()
    return rows


def e9_parity(args, root):
    """Availability per storage byte: +2 full replicas vs +16 parity volunteers (k=8, i.e. the same 2 replicas
    of storage), different k per volunteer for stripe diversity. Then a fraction of ALL nodes dies."""
    rows = []
    B = args.e9_budget  # extra storage budget in replicas: B full copies vs 8*B RS rows (k=8)
    for mode in args.e9_modes:
        for f in args.e9_kills:
            for rep in range(args.reps):
                sw = Swarm(root / f"e9_{mode}_{f}_{rep}", args.peers, args.boot, args.nat, seed=1200 + rep)
                try:
                    link, times, holders = make_replicas(sw, args.days, args.replica_frac * 0.6, seed=1300 + rep,
                                                         layouts=((24, 91, 180),))  # isolate the coding effect
                    grid = link.removeprefix("zt://")
                    vols = [n for n in sw.nodes if not n.startswith("boot") and n not in holders]
                    if mode == "replicas":
                        for n in vols[:B]:
                            jid = http(sw.ctl(n), "POST", "/api/download", {"grid": grid, "region": {"var": "t2m"}})["job"]
                            wait_job(sw.ctl(n), jid)
                    elif mode.startswith("rs"):
                        # 16 volunteers x one distinct Cauchy-RS row each (k=8): 2 replicas of storage in total,
                        # any 8 surviving pieces of a stripe rebuild it
                        for j, n in enumerate(vols[:8 * B]):
                            body = {"grid": grid, "var": "t2m", "k": 8, "drop": True, "row": j}
                            if mode == "rs_consecutive":
                                body["d"] = 1
                            t_ = time.time()
                            r_ = http(sw.ctl(n), "POST", "/api/parity", body)
                            print(f"  parity vol {j}: {r_} {time.time() - t_:.1f}s", flush=True)
                    stored = sum(http(sw.ctl(n), "GET", "/api/status")["grids"].get(grid, {}).get("bytes", 0)
                                 for n in sw.alive())
                    covered = set()  # time steps that exist somewhere before the failures
                    for st_, sp_, _ in holders.values():
                        covered.update(range(st_, st_ + sp_))
                    total = len(covered)
                    victims = sw.rng.sample([n for n in sw.nodes if not n.startswith("boot")], int(f * (len(sw.nodes) - args.boot)))
                    for vv in victims:
                        sw.kill(vv)
                    c = fresh_client(sw, "e9", "maxflow")
                    t_ = time.time()
                    jid = http(sw.ctl(c), "POST", "/api/download", {"grid": grid, "region": {"var": "t2m"}})["job"]
                    job = wait_job(sw.ctl(c), jid)
                    print(f"  download {time.time() - t_:.1f}s restored={job.get('restored')}", flush=True)
                    ds = zt.open_dataset(link, ctl=sw.ctl(c))
                    t0h = times.values[0].astype("datetime64[h]").astype("int64")
                    got = set()
                    for d0 in range(0, len(times), 24):  # per day: an unrecoverable chunk only costs its own day
                        try:
                            sub = ds.t2m.sel(time=slice(times[d0], times[min(d0 + 23, len(times) - 1)]))
                            ok = ~np.isnan(sub.values).all(axis=(1, 2))
                            hs = sub.time.values.astype("datetime64[h]").astype("int64") - t0h
                            got |= {int(h) for h, o in zip(hs, ok) if o}
                        except OSError:
                            pass
                    got &= covered
                    pdst = http(sw.ctl(c), "GET", "/api/status")["pushdown"]
                    rows.append({"mode": mode, "kill": f, "rep": rep, "stored_MB": round(stored / 1e6, 1),
                                 "restore_fail": pdst.get("restore_fail"), "loss_hist": pdst.get("loss_hist"),
                                 "steps_total": total, "steps_got": len(got), "restored": job.get("restored", 0),
                                 "availability": round(len(got) / max(total, 1), 3)})
                    print("E9", rows[-1], flush=True)
                finally:
                    sw.stop()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peers", type=int, default=24)
    ap.add_argument("--boot", type=int, default=3)
    ap.add_argument("--nat", type=float, default=0.3)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--replica-frac", type=float, default=0.8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", default="e1,e2,e3,e4")
    ap.add_argument("--e9-budget", type=int, default=2)
    ap.add_argument("--e1-strategies", nargs="*", default=["maxflow", "rarest", "random", "single"])
    ap.add_argument("--e9-kills", type=float, nargs="*", default=[0.4, 0.6])
    ap.add_argument("--e9-modes", nargs="*", default=["none", "replicas", "rs_consecutive", "rs_interleaved"])
    ap.add_argument("--out", default="sim/results.json")
    a = ap.parse_args()
    base = Path(os.environ.get("ZT_SIM_DIR", "~/.cache/zt_sim")).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ztsim", dir=base))
    res = {"config": vars(a)}
    try:
        for e in a.only.split(","):
            res[e] = {"e1": e1_strategies, "e2": e2_availability, "e3": e3_churn, "e4": e4_dht_load,
                      "e5": e5_jlps, "e6": e6_loss, "e9": e9_parity}[e](a, root)
            Path(a.out).write_text(json.dumps(res, indent=1))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
