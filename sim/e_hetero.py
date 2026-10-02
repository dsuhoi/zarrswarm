"""E-het: heterogeneous replicas of the SAME values (bench/make_variants.py: 1 h / 6 h, map / time-series chunks,
lz4 / zstd / gzip, zarr v2 / v3) spread over a swarm; value identity (ours) vs byte identity (IPFS/BitTorrent-like
ablation: same engine, but a swarm is one byte representation and the client must pick one).

Every peer holds a hard-linked time window (whole chunks) of one variant. Queries: day of global hourly maps,
point time series over the whole period (hourly), the whole period at 6 h. Per (mode, query, rep): seconds, bytes,
missing fraction, holders usable / used, exact values vs the source.

python sim/e_hetero.py VARIANTS_DIR [--peers 48 --reps 3 --out sim/results_ehet.json]
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate as S  # noqa: E402
import zarr_torrent as zt  # noqa: E402
from zarr_torrent import scan as scanmod  # noqa: E402
from zarr_torrent.store import http, wait_job  # noqa: E402

VAR = "2m_temperature"


def window_replica(src: Path, dst: Path, c_lo: int, c_hi: int):
    """dst = time chunks [c_lo, c_hi) of src: metadata rewritten, chunk files hard-linked (identical bytes)."""
    g = zarr.open_group(str(src), mode="r")
    fmt = g.metadata.zarr_format
    dst.mkdir(parents=True)
    for f in ("zarr.json", ".zgroup", ".zattrs"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    tarr = g["time"]
    ct = g[VAR].chunks[0]
    n = g[VAR].shape[0]
    s0, s1 = c_lo * ct, min(c_hi * ct, n)
    for name, a in g.arrays():
        (dst / name).mkdir()
        meta_files = ["zarr.json"] if fmt == 3 else [".zarray", ".zattrs"]
        dims = a.metadata.dimension_names if fmt == 3 else a.attrs.get("_ARRAY_DIMENSIONS")
        timed = dims and dims[0] == "time"
        for f in meta_files:
            if not (src / name / f).exists():
                continue
            doc = json.loads((src / name / f).read_text())
            if timed and f in ("zarr.json", ".zarray"):
                doc["shape"][0] = s1 - s0
                if name == "time":  # coordinate: one chunk holding exactly the window
                    if fmt == 3:
                        doc["chunk_grid"]["configuration"]["chunk_shape"] = [s1 - s0]
                    else:
                        doc["chunks"] = [s1 - s0]
            (dst / name / f).write_text(json.dumps(doc))
        if name == "time":
            out = zarr.open_array(str(dst / name), mode="r+")
            out[:] = np.asarray(tarr[s0:s1])
            continue
        grid = [-(-s // c) for s, c in zip(a.shape, a.chunks)]
        for idx in np.ndindex(*grid):
            if timed and not (c_lo <= idx[0] < c_hi):
                continue
            sk = a.metadata.encode_chunk_key(idx)
            dk = a.metadata.encode_chunk_key((idx[0] - c_lo,) + idx[1:] if timed else idx)
            sp, dp = src / name / sk, dst / name / dk
            if sp.exists():
                dp.parent.mkdir(parents=True, exist_ok=True)
                os.link(sp, dp)


def subset_replica(src: Path, dst: Path, times: set[int]):
    """dst = src with only the time chunks in `times` present (a station's sparse ingest archive): metadata and
    coordinates as-is, absent chunks simply missing (the node advertises only what exists)."""
    g = zarr.open_group(str(src), mode="r")
    dst.mkdir(parents=True)
    for f in ("zarr.json", ".zgroup", ".zattrs"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    for name, a in g.arrays():
        dims = a.metadata.dimension_names if g.metadata.zarr_format == 3 else a.attrs.get("_ARRAY_DIMENSIONS")
        timed = name != "time" and dims and dims[0] == "time"
        for p in (src / name).rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(src / name)
            if timed and p.name not in ("zarr.json", ".zarray", ".zattrs"):
                idx = [int(x) for x in str(rel).replace("c/", "").replace("/", ".").split(".")]
                if idx[0] not in times:
                    continue
            (dst / name / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(p, dst / name / rel)


def seed_all(sw, holders):
    """Every node scans and announces its replica at once (setup only: nothing is measured here)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(holders) or 1) as ex:
        list(ex.map(lambda peer: http(sw.ctl(peer), "POST", "/api/seed", {"path": str(sw.root / "rep" / peer)}), holders))


def build_sat(sw, variants: dict, rng, traces: dict, n_stations: int, mirrors=("V2-wb2-6h", "V3-ts-1h"),
              station_enc=("V1-arco-1h", "V5-tiles-1h"), frac=(0.2, 0.6)):
    """Sensing placement: the first n_stations peers are ground stations holding exactly the hours they received
    (real SatNOGS stations x real orbits, bench/sat_traces.py), encoded by their own software (hourly maps or
    tiles); the other peers are downstream mirrors with contiguous windows of re-encoded variants."""
    holders = {}
    peers = [p for p in sw.nodes if not p.startswith("boot")]
    for i, peer in enumerate(peers):
        dst = sw.root / "rep" / peer
        if i < n_stations and i < len(traces["stations"]):
            var = station_enc[i % len(station_enc)]
            subset_replica(variants[var], dst, set(traces["stations"][i]["hours"]))
            holders[peer] = (var, "station", len(traces["stations"][i]["hours"]))
        else:
            var = rng.choice(mirrors)
            g = zarr.open_group(str(variants[var]), mode="r")
            nch = -(-g[VAR].shape[0] // g[VAR].chunks[0])
            span = max(1, round(nch * rng.uniform(*frac)))
            lo = rng.randint(0, nch - span)
            window_replica(variants[var], dst, lo, lo + span)
            holders[peer] = (var, lo, lo + span)
    seed_all(sw, holders)
    return holders


def build(sw, variants: dict, rng, frac=(0.2, 0.6)):
    """Each non-bootstrap peer seeds a window of one variant; returns {peer: (variant, c_lo, c_hi)}."""
    holders = {}
    names = sorted(variants)
    for peer in [p for p in sw.nodes if not p.startswith("boot")]:
        var = rng.choice(names)
        g = zarr.open_group(str(variants[var]), mode="r")
        nch = -(-g[VAR].shape[0] // g[VAR].chunks[0])
        span = max(1, round(nch * rng.uniform(*frac)))
        lo = rng.randint(0, nch - span)
        dst = sw.root / "rep" / peer
        window_replica(variants[var], dst, lo, lo + span)
        holders[peer] = (var, lo, lo + span)
    seed_all(sw, holders)
    return holders


def requested_samples(view: dict, region: dict) -> set[int]:
    """Grid quanta the query asks for: its time window on its step lattice. The window is the query's own t0/t1
    (every query names it): clipping it to the chosen swarm's extent would let a byte swarm that holds part of the
    month report a complete answer."""
    t = view["grid"]["time"]
    S = int(region["step"]) // t["dt"]
    lo, hi = -10 ** 18, 10 ** 18
    sec = lambda x: int(np.datetime64(x, "s").astype("int64"))
    if region.get("t0"):
        lo = max(lo, -(-(sec(region["t0"]) - t["rphase"]) // t["dt"]))
    if region.get("t1"):
        end = sec(str(np.datetime64(region["t1"], "D") + 1)) if len(region["t1"]) == 10 else sec(region["t1"]) + 1
        hi = min(hi, -(-(end - t["rphase"]) // t["dt"]))
    first = -(-lo // S) * S
    return set(range(first, hi, S))


def obtained_samples(view: dict, keys: list[str], S: int) -> set[int]:
    """Lattice samples (step S) whose data arrived: union of the time samples of the received chunks
    (spatial tiles of a chunk family count once they are all in - approximated per time chunk)."""
    got = set()
    for k in keys:
        n, lay, co = scanmod.split_key(k)
        li = view["arrays"][n]["layouts"].get(lay)
        if li is None:
            continue
        s, _ = scanmod.tstride(li)
        a, b = scanmod.chunk_g(li, 0, co[0])
        got.update(range(a, b, s))  # the chunk's own samples; the caller intersects with the requested lattice
    return got


def run_query(sw, tag, link, region, cover="jlps", tries=3, deadline=1800):
    """run_query_once with a fresh client per attempt: a client that fails to start or to answer is replaced, and
    the attempt is counted. A job that outlives `deadline` is not retried: it is reported as censored (TimeoutError)."""
    for attempt in range(tries):
        try:
            job, secs, cov = run_query_once(sw, f"{tag}a{attempt}", link, region, cover, deadline)
            job["attempts"] = attempt + 1
            return job, secs, cov
        except TimeoutError:
            raise
        except Exception as e:
            print(f"query {tag} attempt {attempt}: {type(e).__name__} {e}", flush=True)
            if attempt == tries - 1:
                raise


def run_query_once(sw, tag, link, region, cover="jlps", deadline=1800):
    """Fresh client (empty cache) -> one download job; seconds, job, completeness of the answer."""
    client = S.fresh_client(sw, tag, "maxflow")
    time.sleep(1.0)
    ctl = sw.ctl(client)
    grids = http(ctl, "GET", "/api/resolve?link=" + link)["grid"].split("+")
    grid = next(g for g in grids if VAR in http(ctl, "GET", f"/api/view/{g}?refresh=1")["arrays"])
    view = http(ctl, "GET", f"/api/view/{grid}")
    t = time.time()
    try:
        jid = http(ctl, "POST", "/api/download", {"grid": grid, "region": dict(region, var=VAR),
                                                   "cover": cover, "label": "q"}, timeout=400)["job"]
        wait_job(ctl, jid, deadline=deadline)
    except Exception:
        sw.kill(client)
        raise
    secs = time.time() - t
    job = http(ctl, "GET", f"/api/job/{jid}")
    full_view = http(ctl, "GET", f"/api/view/{grid}")  # layouts incl. stride info
    want = requested_samples(view, region)
    got = obtained_samples(full_view, job["done_keys"], int(region["step"]) // view["grid"]["time"]["dt"]) & want
    sw.kill(client)  # a finished client would seed what it fetched (uncapped) to the next query: remove it
    return job, secs, len(got) / max(len(want), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants")
    ap.add_argument("--peers", type=int, default=48)
    ap.add_argument("--boot", type=int, default=3)
    ap.add_argument("--nat", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep-start", type=int, default=0, help="resume: skip repetitions already done")
    ap.add_argument("--modes", default="values,bytes")
    ap.add_argument("--out", default="sim/results_ehet.json")
    ap.add_argument("--placement", default="random", choices=["random", "sat"])
    ap.add_argument("--procs", action="store_true", help="one real `zt node` process per peer (no shared GIL)")
    ap.add_argument("--emu", action="store_true", help="kernel-level emulation: netns + tc netem/tbf (sim/netemu.py)")
    ap.add_argument("--rate-mbps", type=float, default=4.0, help="median peer uplink (lognormal, clipped x0.2..x5)")
    ap.add_argument("--covers", default="jlps", help="values mode: also run e.g. 'jlps,bytes' (E2)")
    ap.add_argument("--queries", default="map_day_1h,series_point_1h,period_6h")
    ap.add_argument("--traces", default="bench/sat_traces.json")
    ap.add_argument("--stations", type=int, default=24, help="sat placement: how many peers are ground stations")
    a = ap.parse_args()
    vdir = Path(a.variants).expanduser()
    variants = {p.stem: p for p in sorted(vdir.glob("V*.zarr"))}
    ds_truth = __import__("xarray").open_zarr(variants[next(k for k in variants if "V1" in k)], consolidated=False)
    times = ds_truth.time.values
    day = str(times[24])[:10]
    first, last = str(times[0])[:19], str(times[-1])[:19]  # the whole period every variant was cut from
    lat_i, lon_i = 300, 500  # an arbitrary grid point for the time-series query
    queries = {
        "map_day_1h": {"t0": day, "t1": day, "step": 3600},
        "series_point_1h": {"t0": first, "t1": last, "step": 3600,
                            "isel": {"latitude": [lat_i, lat_i + 1], "longitude": [lon_i, lon_i + 1]}},
        "period_6h": {"t0": first, "t1": last, "step": 21600},
    }
    queries = {q: queries[q] for q in a.queries.split(",")}
    root = Path(os.environ.get("ZT_SIM_ROOT", "~/.cache/zt_sim")).expanduser()
    rows = []
    for rep in range(a.rep_start, a.reps):
        for mode in a.modes.split(","):
            scanmod.BYTE_IDENTITY = mode == "bytes"
            rng = random.Random(1000 + rep)  # same placement for both modes within a rep
            if a.emu:
                from netemu import EmuSwarm
                sw = EmuSwarm(root / f"ehet_{mode}_{rep}", a.peers, a.boot, a.nat, seed=100 + rep,
                              env={"ZT_IDENTITY": mode}, rate_median=a.rate_mbps * 1e6,
                              rate_range=(a.rate_mbps * 0.2e6, a.rate_mbps * 5e6))
            elif a.procs:
                from procswarm import ProcSwarm
                sw = ProcSwarm(root / f"ehet_{mode}_{rep}", a.peers, a.boot, a.nat, seed=100 + rep,
                               env={"ZT_IDENTITY": mode}, rate_median=a.rate_mbps * 1e6,
                               rate_range=(a.rate_mbps * 0.2e6, a.rate_mbps * 5e6))
            else:
                sw = S.Swarm(root / f"ehet_{mode}_{rep}", a.peers, a.boot, a.nat, seed=100 + rep)
            try:
                holders = build(sw, variants, rng) if a.placement == "random" else \
                    build_sat(sw, variants, rng, json.load(open(a.traces)), a.stations)
                links = {}
                for peer, (var, lo, hi) in holders.items():
                    st = http(sw.ctl(peer), "GET", "/api/status")
                    g = next(s["grid"] for s in st["seeds"] if VAR in s["arrays"])
                    links.setdefault(g, []).append(peer)
                time.sleep(2)
                for (qn, reg), cover in [(q, c) for q in queries.items()
                                         for c in (a.covers.split(",") if mode == "values" else ["jlps"])]:
                    # value identity: one swarm. byte identity: the client picks the swarm that covers the most
                    # of its query (oracle, generous to the baseline), ties -> more holders
                    cands = sorted(links, key=lambda g: -len(links[g]))
                    best = None
                    for i, g in enumerate(cands):
                        # the oracle needs the best byte swarm, not how slow the others are: once one result exists,
                        # a later candidate gets three times its time (at least 2 min) and is otherwise just worse
                        dl = 1800 if best is None else max(120.0, 3 * best["seconds"])
                        try:
                            job, secs, cov = run_query(sw, f"{mode}{qn[:3]}{cover[:2]}{i}", "zt://" + g, reg, cover,
                                                       deadline=dl)
                        except Exception as e:
                            print(f"candidate {g[:8]} of {qn}: {type(e).__name__} {e}", flush=True)
                            if best is None and i == len(cands) - 1:
                                best = {"mode": mode, "select": cover, "rep": rep, "query": qn, "failed": str(e)[:200],
                                        "seconds": None, "swarms": len(links)}
                            continue
                        cand = {"mode": mode, "select": cover, "rep": rep, "query": qn, "swarm_holders": len(links[g]),
                                "swarms": len(links), "seconds": round(secs, 2), "bytes": job["bytes"],
                                "chunks": job["done"], "total": job["total"], "missing": job["missing"],
                                "coverage": round(cov, 3), "peers_used": len(job["per_peer"]),
                                "plan_T": job.get("plan_T"), "est_T": (job.get("cover") or {}).get("est_T"),
                                "attempts": job.get("attempts", 1),
                                "phases": job.get("phases"),
                                "peer_detail": {p[:8]: {"pred_MB": round((job.get("pred") or {}).get(p, {}).get("bytes", 0) / 1e6, 2),
                                                        "bw_MBps": round((job.get("pred") or {}).get(p, {}).get("bw", 0) / 1e6, 2),
                                                        "act_MB": round(x["bytes"] / 1e6, 2), "end_s": x["end_s"],
                                                        "via": bool((job.get("pred") or {}).get(p, {}).get("via"))}
                                                for p, x in (job.get("actual") or {}).items()},
                                "cover": job.get("cover", {}).get("layouts")}
                        if best is None or (cand["coverage"], -cand["seconds"]) > (best["coverage"], -best["seconds"]):
                            best = cand
                        if mode == "values":
                            break
                    rows.append(best)
                    print("EHET", json.dumps(best), flush=True)
            finally:
                sw.stop()
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
