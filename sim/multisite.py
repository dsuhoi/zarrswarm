"""E7: real multi-site run over the Internet. Holders run on remote sites (each node behind NAT, attached through its
own reverse SSH tunnel to a bootstrap/relay on this host); a fresh client on this host issues each query. No emulated
links: every byte crosses a real WAN path. Same values everywhere (ERA5 2 m temperature from ARCO, each site encodes
its own copies with bench/make_variants.py), compared against the public cloud copy read directly.

Phases (identity mode = how remote nodes are started):
  mirror  : lattice, only the first holder (one HTTP mirror holding everything in the public layout)
  swarm   : lattice, all holders; covers jlps and bytes (min-bytes)
  bytes   : byte identity, all holders; the client tries every byte swarm and keeps the best (oracle)
  cloud   : xarray reads the region straight from the ARCO bucket (Google Cloud Storage)

python sim/multisite.py sim/multisite.toml [--reps 3 --out sim/results_multisite.json]
"""
import argparse
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate as S  # noqa: E402
import zarr_torrent as zt  # noqa: E402
from e_hetero import obtained_samples, requested_samples  # noqa: E402
from procswarm import ProcSwarm  # noqa: E402
from zarr_torrent.store import http, wait_job  # noqa: E402

VAR = "2m_temperature"
ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


class LocalSide(ProcSwarm):  # this host: bootstrap/relay + clients, no emulated latency or caps
    def _lat(self):
        return 0.0


class Tunnel:
    """One reverse tunnel per site: the remote port equals the bootstrap port here, so the relay URL a remote node
    announces (http://127.0.0.1:BP/r/<id>) is valid on both ends."""

    def __init__(self, host: str, bp: int, log: Path):
        self.args = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
                     "-o", "ExitOnForwardFailure=yes", "-R", f"{bp}:127.0.0.1:{bp}", host]
        self.log, self.proc = log, None

    def start(self, tries=25):
        for _ in range(tries):
            self.proc = subprocess.Popen(self.args, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            time.sleep(25)
            if self.proc.poll() is None:
                return self
        raise RuntimeError(f"tunnel: {self.log.read_text()[-400:]}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class Remote:
    """One holder node on a remote site, alive as long as its ssh session (pty: SIGHUP kills the process group)."""

    def __init__(self, site: dict, name: str, data: str, boot_port: int, env: dict, log: Path):
        self.name, self.site, self.log = name, site, log
        r = random.Random(name + str(time.time()))
        port, ctl = (r.randint(20000, 60000) for _ in range(2))
        tport = boot_port  # the site's tunnel
        py, src = site["python"], site.get("src", "~/zt_src")
        exports = " ".join(f"{k}={v}" for k, v in dict(env, PYTHONPATH=site.get("pythonpath", src)).items())
        home = f"{site.get('work', '~/zt_ms')}/h_{name}"
        q = (f"from zarr_torrent.store import http; import json; "
             f"print('GRIDS', json.dumps([s['grid'] for s in http('http://127.0.0.1:{ctl}','GET','/api/status')['seeds']]))")
        # page-cache the replica first: otherwise whichever configuration runs first pays the site's cold disk
        # (measured: 9.3 s vs 2.1 s for the same 56 MB from the same peers)
        cmd = (f"trap 'kill 0' EXIT HUP TERM; cd {src}; export {exports}; rm -rf {home}; "
               f"find {data} -type f -exec cat {{}} + > /dev/null; "
               f"{site.get('nice', 'nice -n 10')} {py} -m zarr_torrent.cli node --home {home} --host 127.0.0.1 "
               f"--port {port} --ctl-port {ctl} --bootstrap http://127.0.0.1:{tport} --relay http://127.0.0.1:{tport} "
               f"> {home}.log 2>&1 & "
               f"for i in $(seq 90); do sleep 2; {py} -m zarr_torrent.cli --ctl http://127.0.0.1:{ctl} seed {data} "
               f">/dev/null 2>&1 && break; done; {py} -c \"{q}\"; echo SEEDED; wait")
        self.args = ["ssh", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
                     site["host"], cmd]
        self.grids = None

    def start(self, tries=25, wait_s=300):
        for _ in range(tries):  # sshd on a loaded host drops some connections at the banner: retry
            self.proc = subprocess.Popen(self.args, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            t = time.time()
            while time.time() - t < wait_s and self.proc.poll() is None:
                m = re.search(r"GRIDS (\[.*?\])", self.log.read_text(errors="replace"))
                if m and "SEEDED" in self.log.read_text(errors="replace"):
                    self.grids = json.loads(m.group(1))
                    return self
                time.sleep(2)
            self.stop()
            time.sleep(20)
        raise RuntimeError(f"{self.name}: no start: {self.log.read_text(errors='replace')[-600:]}")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def truth_of(src: Path, region: dict) -> np.ndarray:
    ds = xr.open_zarr(src, consolidated=False)[VAR]
    return select(ds, region).values


def select(da, region):
    da = da.sel(time=slice(region["t0"], region["t1"]))
    da = da.isel(time=slice(None, None, int(region["step"]) // 3600))
    for d, (lo, hi) in region.get("isel", {}).items():
        da = da.isel({d: slice(lo, hi)})
    return da


def query(side, link, region, cover, truth, tag):
    """Fresh client -> one download job; time, bytes, completeness and the largest deviation from the source."""
    client = S.fresh_client(side, tag, "maxflow")
    try:
        ctl = side.ctl(client)
        grid = None
        for _ in range(30):  # the client learns the swarm through the DHT
            try:
                grids = http(ctl, "GET", "/api/resolve?link=" + link)["grid"].split("+")
                grid = next(g for g in grids if VAR in http(ctl, "GET", f"/api/view/{g}?refresh=1")["arrays"])
                break
            except Exception:
                time.sleep(2)
        t_v = time.time()
        view = http(ctl, "GET", f"/api/view/{grid}")
        t = time.time()
        jid = http(ctl, "POST", "/api/download", {"grid": grid, "region": dict(region, var=VAR), "cover": cover,
                                                   "label": "q"})["job"]
        wait_job(ctl, jid)
        secs = time.time() - t
        job = http(ctl, "GET", f"/api/job/{jid}")
        S_ = int(region["step"]) // view["grid"]["time"]["dt"]
        want = requested_samples(view, region)
        got = obtained_samples(http(ctl, "GET", f"/api/view/{grid}"), job["done_keys"], S_) & want
        err = None
        if job["missing"] == 0:
            ds = zt.open_dataset(link, ctl=ctl, step=None if region["step"] == 3600 else f"{region['step']}s")[VAR]
            got_v = ds.sel(time=slice(region["t0"], region["t1"]))
            for d, (lo, hi) in region.get("isel", {}).items():
                got_v = got_v.isel({d: slice(lo, hi)})
            got_v = got_v.values
            err = float(np.nanmax(np.abs(got_v - truth))) if got_v.shape == truth.shape else f"shape {got_v.shape}"
        return {"seconds": round(secs, 2), "MB": round(job["bytes"] / 1e6, 2), "chunks": job["done"],
                "missing": job["missing"], "coverage": round(len(got) / max(len(want), 1), 3),
                "peers_used": len(job["per_peer"]), "layouts": (job.get("cover") or {}).get("layouts"),
                "plan_T": job.get("plan_T"), "max_abs_err": err, "phases": job.get("phases"), "t_view": round(t - t_v, 2),
                "per_peer_MB": {p[:8]: round(x["bytes"] / 1e6, 2) for p, x in (job.get("actual") or {}).items()}}
    finally:
        side.kill(client)
        shutil.copy(side.root / f"{client}.log", side.root.parent / "logs" / f"{client}.log")


def cloud(region, truth, tries=3):
    for attempt in range(tries):  # a truncated response from the bucket is retried and reported, not fatal
        try:
            t = time.time()
            da = xr.open_zarr(ARCO, consolidated=True, chunks=None)[VAR]
            t_open = time.time() - t
            v = select(da, region).values
            secs = time.time() - t
            return {"seconds": round(secs, 2), "open_s": round(t_open, 2), "attempts": attempt + 1,
                    "max_abs_err": float(np.nanmax(np.abs(v - truth)))}
        except Exception:
            if attempt == tries - 1:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--phases", default="cloud,mirror,swarm,bytes")
    ap.add_argument("--out", default="sim/results_multisite.json")
    ap.add_argument("--queries", help="comma-separated subset of the configured queries")
    a = ap.parse_args()
    cfg = tomllib.load(open(a.config, "rb"))
    queries = {q: r for q, r in cfg["queries"].items() if not a.queries or q in a.queries.split(",")}
    truth_src = Path(cfg["truth"]).expanduser()
    truths = {q: truth_of(truth_src, r) for q, r in queries.items()}
    key = secrets.token_urlsafe(16)  # closed network: only our nodes speak to each other
    root = Path("~/.cache/zt_ms").expanduser()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    out = lambda: json.dump({"config": cfg, "reps": a.reps, "rows": rows}, open(a.out, "w"), indent=1)

    def emit(row):
        rows.append(row)
        print("EMS", json.dumps(row), flush=True)
        out()

    phases = a.phases.split(",")
    if "cloud" in phases:
        for rep in range(a.reps):
            for qn, reg in queries.items():
                emit(dict(cloud(reg, truths[qn]), phase="cloud", query=qn, rep=rep))
    if "http" in phases:  # catalog-driven multi-source HTTP over the sites that hold the public layout
        import http_mirrors as H
        ms = [H.Mirror(cfg["sites"][m["site"]], m["data"], m["first_hour"], m["hours"], logs / f"http_{i}.log").start()
              for i, m in enumerate(cfg["http_mirrors"])]
        try:
            week0 = np.datetime64(cfg["week_start"])
            for rep in range(a.reps):
                for qn, reg in queries.items():
                    secs, per, vals = H.query(ms, reg, week0)
                    err = float(np.nanmax(np.abs(vals - truths[qn]))) if vals.shape == truths[qn].shape else f"shape {vals.shape}"
                    emit({"phase": "http", "query": qn, "rep": rep, "seconds": round(secs, 2), "per_mirror": per,
                          "max_abs_err": err})
        finally:
            for m in ms:
                m.stop()
    for mode, names in (("values", ["mirror", "swarm"]), ("bytes", ["bytes"])):
        names = [n for n in names if n in phases]
        if not names:
            continue
        env = {"ZT_IDENTITY": mode, "ZT_NETWORK_KEY": key}
        side = LocalSide(root / f"local_{mode}", 1, 1, 0.0, seed=7, env=env, relay_rate=None)
        boot_port = side.nodes["boot0"].port
        remotes, tunnels = [], {}
        try:
            holders = cfg["holders"]
            for i, h in enumerate(holders):
                if i == 1 and "mirror" in names:  # mirror phase: the first holder alone
                    for rep in range(a.reps):
                        for qn, reg in queries.items():
                            emit(dict(query(side, "zt://" + remotes[0].grids[0], reg, "jlps", truths[qn],
                                            f"m{rep}{qn[:3]}"), phase="mirror", query=qn, rep=rep, holders=1))
                site = cfg["sites"][h["site"]]
                if h["site"] not in tunnels:
                    tunnels[h["site"]] = Tunnel(site["host"], boot_port, logs / f"tunnel_{h['site']}.log").start()
                remotes.append(Remote(site, f"{h['site']}{i}_{mode}", h["data"], boot_port,
                                      dict(env, **site.get("env", {})), logs / f"{h['site']}{i}_{mode}.log").start())
                print("up", remotes[-1].name, remotes[-1].grids, flush=True)
            if not {"swarm", "bytes"} & set(names):
                continue
            time.sleep(5)
            grids = {}
            for r in remotes:
                grids.setdefault(r.grids[0], []).append(r.name)
            for rep in range(a.reps):
                for qn, reg in queries.items():
                    covers = ["jlps", "bytes"] if mode == "values" else ["jlps"]
                    for cover in (covers if rep % 2 == 0 else covers[::-1]):  # neither always runs first
                        best = None
                        for g in sorted(grids, key=lambda g: -len(grids[g])):
                            r = dict(query(side, "zt://" + g, reg, cover, truths[qn], f"{mode[0]}{rep}{qn[:3]}{cover[:2]}"),
                                     swarm_holders=len(grids[g]), swarms=len(grids))
                            if best is None or (r["coverage"], -r["seconds"]) > (best["coverage"], -best["seconds"]):
                                best = r
                        emit(dict(best, phase="swarm" if mode == "values" else "bytes", select=cover, query=qn,
                                  rep=rep, holders=len(remotes)))
        finally:
            for r in remotes:
                r.stop()
            for t in tunnels.values():
                t.stop()
            side.stop()
    out()


if __name__ == "__main__":
    main()
