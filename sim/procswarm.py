"""A swarm of real `zt node` processes on one host (one process per peer: no shared GIL, real sockets), with the
same interface as simulate.Swarm (nodes, meta, ctl, _start, kill, stop). Uplinks are token buckets (--upload-mbps),
latency via ZT_EMU_LATENCY_MS; NAT'd peers attach to a bootstrap relay.
"""
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zarrswarm.store import http  # noqa: E402
from simulate import port  # noqa: E402


class _P:  # what callers read from a node object
    def __init__(self, proc, port, ctl_port):
        self.proc, self.port, self.ctl_port = proc, port, ctl_port


class ProcSwarm:
    def __init__(self, root: Path, n: int, n_boot: int, nat_frac: float, seed: int, strategy="maxflow",
                 env: dict | None = None, rate_median=4e6, rate_range=(0.5e6, 30e6), relay_rate=25e6):
        self.root, self.rng = Path(root), random.Random(seed)
        self.rate_median, self.rate_range = rate_median, rate_range
        shutil.rmtree(self.root, ignore_errors=True)  # a fresh experiment: no node home of an interrupted run
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes, self.meta, self.env, self.strategy = {}, {}, dict(env or {}), strategy
        boots = []
        for i in range(n_boot):
            p = port()
            # bootstrap/relay nodes are public servers, not stations: server-class uplink
            self._start(f"boot{i}", dict(port=p, relay_server=True, bootstrap=boots[:], rate=relay_rate,
                                         latency=self._lat()))
            boots.append(f"http://127.0.0.1:{p}")
        self.boots = boots
        for i in range(n - n_boot):
            nat = self.rng.random() < nat_frac
            self._start(f"p{i:02d}", dict(port=port(), bootstrap=self.rng.sample(boots, min(2, len(boots))),
                                          relay=self.rng.choice(boots) if nat else None, rate=self._rate(),
                                          latency=self._lat()))

    def _rate(self):
        return float(np.clip(self.rng.lognormvariate(math.log(self.rate_median), 0.9), *self.rate_range))

    def _lat(self):
        return self.rng.uniform(0.002, 0.04)

    def _start(self, name, kw):
        cp = port()
        args = [sys.executable, "-m", "zarrswarm.cli", "node", "--home", str(self.root / f"h_{name}"),
                "--host", "127.0.0.1", "--port", str(kw["port"]), "--ctl-port", str(cp)]
        if kw.get("rate"):
            args += ["--upload-mbps", str(kw["rate"] / 1e6)]
        if kw.get("bootstrap"):
            args += ["--bootstrap", *kw["bootstrap"]]
        if kw.get("relay"):
            args += ["--relay", kw["relay"]]
        if kw.get("relay_server"):
            args += ["--relay-server"]
        env = dict(os.environ, **self.env, ZT_EMU_LATENCY_MS=str(kw.get("latency", 0) * 1e3),
                   ZT_EMU_STRATEGY=kw.get("strategy", self.strategy),
                   PYTHONPATH=os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                            os.environ.get("PYTHONPATH")])))
        log = open(self.root / f"{name}.log", "w")
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.nodes[name] = _P(proc, kw["port"], cp)
        self.meta[name] = {"rate": kw["rate"], "nat": bool(kw.get("relay")), "alive": True}
        for _ in range(300):
            try:
                http(self.ctl(name), "GET", "/api/status", None, 2)
                return
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited: {(self.root / f'{name}.log').read_text()[-800:]}")
                time.sleep(0.1)
        raise RuntimeError(f"{name} did not start")

    def ctl(self, name):
        return f"http://127.0.0.1:{self.nodes[name].ctl_port}"

    def kill(self, name):
        p = self.nodes[name].proc
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        self.meta[name]["alive"] = False

    def alive(self):
        return [n for n, m in self.meta.items() if m["alive"]]

    def stop(self):
        for n in self.alive():
            try:
                self.kill(n)
            except Exception:
                pass
        if os.environ.get("ZT_KEEP_LOGS"):  # keep node logs (post-mortem), drop the bulky homes
            for h in self.root.glob("h_*"):
                shutil.rmtree(h, ignore_errors=True)
            return
        shutil.rmtree(self.root, ignore_errors=True)
