"""Baseline for E7: a catalog-driven multi-source HTTP client (as aria2 or a Rucio-style catalog would do). The client
is told which plain HTTP mirrors hold the public layout (hourly maps, Zarr v2) and which hours each holds; it splits
the needed chunk files across the mirrors that hold them (8 concurrent requests per mirror), writes them into a local
Zarr copy and reads the region. No discovery, no verification beyond the catalog: the client trusts it.

Used by sim/multisite.py (--phases http); mirrors are `python -m http.server` on each site behind `ssh -L`.
"""
import asyncio
import random
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp
import numpy as np
import xarray as xr

VAR = "2m_temperature"


class Mirror:
    """A static HTTP server on a remote site, reached through a local port forward (no public port)."""

    def __init__(self, site: dict, data: str, first_hour: int, n_hours: int, log: Path):
        r = random.Random(data + str(time.time()))
        self.rport, self.lport = r.randint(20000, 60000), r.randint(20000, 60000)
        self.first, self.n = first_hour, n_hours
        py = site["python"]
        cmd = f"trap 'kill 0' EXIT HUP TERM; cd {data} && exec {py} -m http.server {self.rport} --bind 127.0.0.1"
        self.args = ["ssh", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
                     "-o", "ExitOnForwardFailure=yes", "-L", f"{self.lport}:127.0.0.1:{self.rport}", site["host"], cmd]
        if site.get("jump"):
            self.args[1:1] = ["-J", site["jump"]]
        self.log, self.proc = log, None
        self.url = f"http://127.0.0.1:{self.lport}"

    def start(self, tries=25):
        import urllib.request
        for _ in range(tries):
            self.proc = subprocess.Popen(self.args, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            for _ in range(30):
                time.sleep(2)
                try:
                    urllib.request.urlopen(self.url + f"/{VAR}/.zarray", timeout=5)
                    return self
                except Exception:
                    if self.proc.poll() is not None:
                        break
            self.stop()
            time.sleep(10)
        raise RuntimeError(f"mirror: {self.log.read_text()[-300:]}")

    def holds(self, t):
        return self.first <= t < self.first + self.n

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


async def _fetch(session, url, dst):
    async with session.get(url) as r:
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(await r.read())


async def _download(mirrors, hours, work: Path, conc=8):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        meta = [".zgroup", ".zattrs", f"{VAR}/.zarray", f"{VAR}/.zattrs"] + \
               [f"{c}/{f}" for c in ("time", "latitude", "longitude") for f in (".zarray", ".zattrs", "0")]
        full = next(m for m in mirrors if m.first == 0)  # metadata and coordinates of the whole week
        await asyncio.gather(*(_fetch(s, f"{full.url}/{k}", work / k) for k in meta), return_exceptions=True)
        queues = {id(m): [] for m in mirrors}
        for i, t in enumerate(hours):  # round robin over the mirrors that hold the hour
            hs = [m for m in mirrors if m.holds(t)]
            queues[id(hs[i % len(hs)])].append(t)
        sems = {id(m): asyncio.Semaphore(conc) for m in mirrors}

        async def one(m, t):
            async with sems[id(m)]:
                await _fetch(s, f"{m.url}/{VAR}/{t - m.first}.0.0", work / VAR / f"{t}.0.0")
        await asyncio.gather(*(one(m, t) for m in mirrors for t in queues[id(m)]))
        return {m.url: len(queues[id(m)]) for m in mirrors}


def query(mirrors, region, t_week0: np.datetime64):
    """Fetch what the region needs, then read it from the local copy. Returns (seconds, chunks per mirror, values)."""
    step = int(region["step"]) // 3600
    t0 = int((np.datetime64(region["t0"]) - t_week0) // np.timedelta64(1, "h"))
    t1 = int((np.datetime64(region["t1"], "h") + (np.timedelta64(23, "h") if len(region["t1"]) == 10 else 0)
              - t_week0) // np.timedelta64(1, "h"))
    hours = list(range(t0, t1 + 1, step))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "c.zarr"
        t = time.time()
        per = asyncio.run(_download(mirrors, hours, work))
        da = xr.open_zarr(work, consolidated=False)[VAR].isel(time=hours)
        for d, (lo, hi) in region.get("isel", {}).items():
            da = da.isel({d: slice(lo, hi)})
        vals = da.values
        return time.time() - t, per, vals
