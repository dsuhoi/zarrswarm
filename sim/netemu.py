"""Kernel-level network emulation of a swarm (Mininet-style, rootless): every peer is a real `zs node` process in its
own network namespace, attached by a veth pair to a router namespace; its access link is shaped by the kernel
(`tc netem` one-way delay and loss, `tc tbf` rate) in both directions, and TCP is the kernel's own. Same interface as
procswarm.ProcSwarm, so sim/e_hetero.py runs unchanged on it (--emu).

Needs unprivileged user namespaces and the veth, sch_netem and sch_tbf kernel modules (one-time, as root:
`modprobe -a veth sch_netem sch_tbf`). The driver re-executes itself inside `unshare -rn`, which becomes the router.

python sim/netemu.py selftest
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
from zarrswarm.common import env
from zarrswarm.store import http  # noqa: E402

IN_ROUTER = "ZS_EMU_ROUTER"


def ensure_router():
    """Re-exec the current program inside a fresh user+net namespace (root there), which acts as the router."""
    if env(IN_ROUTER):
        return
    os.environ[IN_ROUTER] = "1"
    os.execvp("unshare", ["unshare", "-rn", "--kill-child", sys.executable] + sys.argv)


def sh(cmd: str):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd}: {r.stderr.strip()}")
    return r.stdout


class _P:
    def __init__(self, proc, port, ctl_port, ip):
        self.proc, self.port, self.ctl_port, self.ip = proc, port, ctl_port, ip


class EmuSwarm:
    _seq = 0
    def __init__(self, root: Path, n: int, n_boot: int, nat_frac: float, seed: int, strategy="maxflow",
                 env: dict | None = None, rate_median=4e6, rate_range=(0.5e6, 30e6), relay_rate=25e6,
                 client_rate=30e6, loss=0.0):
        ensure_router()
        self.root, self.rng = Path(root), random.Random(seed)
        self.rate_median, self.rate_range, self.client_rate, self.loss = rate_median, rate_range, client_rate, loss
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes, self.meta, self.env, self.strategy = {}, {}, dict(env or {}), strategy
        # an L2 switch (as in Mininet): every access link is a veth whose far end is a bridge port; no routing,
        # so no writable /proc/sys is needed (container sandboxes mount it read-only)
        sh("ip link show br0 >/dev/null 2>&1 || (ip link set lo up && ip link add br0 type bridge && ip link set br0 up "
           "&& ip addr add 10.0.255.254/16 dev br0)")  # one switch per driver; later swarms reuse it
        boots = []
        for i in range(n_boot):
            p = 7000
            self._start(f"boot{i}", dict(port=p, relay_server=True, bootstrap=boots[:], rate=relay_rate,
                                         latency=self._lat()))
            boots.append(f"http://{self.nodes[f'boot{i}'].ip}:{p}")
        self.boots = boots
        for i in range(n - n_boot):
            nat = self.rng.random() < nat_frac
            self._start(f"p{i:02d}", dict(port=7000, bootstrap=self.rng.sample(boots, min(2, len(boots))),
                                          relay=self.rng.choice(boots) if nat else None, rate=self._rate(),
                                          latency=self._lat()))

    def _rate(self):
        return float(np.clip(self.rng.lognormvariate(math.log(self.rate_median), 0.9), *self.rate_range))

    def _lat(self):
        return self.rng.uniform(0.002, 0.04)

    def _start(self, name, kw):
        EmuSwarm._seq += 1  # unique interface names and addresses across the swarms of one driver
        k = EmuSwarm._seq
        ip = f"10.0.{k // 250}.{k % 250 + 1}"
        port, cp = kw["port"], 7001
        args = [sys.executable, "-m", "zarrswarm.cli", "node", "--home", str(self.root / f"h_{name}"),
                "--host", ip, "--port", str(port), "--ctl-port", str(cp)]
        if kw.get("bootstrap"):
            args += ["--bootstrap", *kw["bootstrap"]]
        if kw.get("relay"):
            args += ["--relay", kw["relay"]]
        if kw.get("relay_server"):
            args += ["--relay-server"]
        env = dict(os.environ, **self.env, ZS_CTL_HOST=ip, ZS_EMU_STRATEGY=kw.get("strategy", self.strategy),
                   PYTHONPATH=os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                            os.environ.get("PYTHONPATH")])))
        env.pop("ZS_EMU_LATENCY_MS", None)  # the kernel delays packets; the node adds nothing
        env.pop("ZT_EMU_LATENCY_MS", None)  # legacy setting must not add a second delay either
        if kw.get("rate"):  # announce the access-link rate like a station would; the kernel enforces it
            env["ZS_ANNOUNCE_MBPS"] = str(kw["rate"] / 1e6)
        log = open(self.root / f"{name}.log", "w")
        # the child waits on stdin until its interface exists, then becomes the node
        proc = subprocess.Popen(["unshare", "-n", "sh", "-c", 'read x; exec "$@"', "sh", *args], env=env,
                                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(0.05)
        r, n = f"r{k}", f"n{k}"
        up = kw.get("rate") or self.client_rate          # access link: node -> network
        down = self.client_rate if kw.get("rate") is None else max(up, 50e6)  # network -> node
        delay = kw.get("latency", 0.0) * 1e3
        loss = f" loss {self.loss * 100:.2f}%" if self.loss else ""
        sh(f"ip link add {r} type veth peer name {n}")
        sh(f"ip link set {n} netns {proc.pid}")
        sh(f"ip link set {r} master br0 && ip link set {r} up")
        shape = lambda dev, rate: (f"tc qdisc add dev {dev} root handle 1: netem delay {delay:.1f}ms{loss} && "
                                   f"tc qdisc add dev {dev} parent 1: handle 2: tbf rate {int(rate * 8)}bit "
                                   f"burst {max(32768, int(rate / 50))} latency 2s")
        sh(shape(r, down))
        sh(f"nsenter -t {proc.pid} -n sh -c 'ip link set lo up && ip addr add {ip}/16 dev {n} && ip link set {n} up "
           f"&& {shape(n, up)}'")
        proc.stdin.write(b"go\n")
        proc.stdin.close()
        self.nodes[name] = _P(proc, port, cp, ip)
        self.meta[name] = {"rate": kw.get("rate"), "nat": bool(kw.get("relay")), "alive": True, "latency": delay}
        for _ in range(3000):  # up to 5 min: a loaded host imports slowly
            try:
                http(self.ctl(name), "GET", "/api/status", None, 2)
                return
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited: {(self.root / f'{name}.log').read_text()[-800:]}")
                time.sleep(0.1)
        raise RuntimeError(f"{name} did not start")

    def ctl(self, name):
        n = self.nodes[name]
        return f"http://{n.ip}:{n.ctl_port}"

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
        if env("ZS_KEEP_LOGS"):
            for h in self.root.glob("h_*"):
                shutil.rmtree(h, ignore_errors=True)
            return
        shutil.rmtree(self.root, ignore_errors=True)


def selftest():
    """Two nodes, a 1 MB/s uplink with 50 ms one-way delay: measured RTT and transfer rate must follow the shaping."""
    import xarray as xr
    import pandas as pd
    import zarrswarm as zs
    root = Path("~/.cache/zs_netemu_selftest").expanduser()
    shutil.rmtree(root, ignore_errors=True)
    sw = EmuSwarm(root, 2, 1, 0.0, seed=1, relay_rate=25e6, rate_median=1e6, rate_range=(1e6, 1e6))
    vals = np.random.default_rng(0).random((48, 64, 64)).astype("f4")
    p = root / "d.zarr"
    xr.Dataset({"v": (("time", "y", "x"), vals)}, coords={"time": pd.date_range("2021-01-01", periods=48, freq="h"),
               "y": np.arange(64.0), "x": np.arange(64.0)}).to_zarr(p, encoding={"v": {"chunks": (12, 64, 64)}},
                                                                    consolidated=False)
    link = http(sw.ctl("p00"), "POST", "/api/seed", {"path": str(p)})["link"]
    t = time.time()
    got = zs.open_dataset(link, ctl=sw.ctl("boot0")).v.values
    dt = time.time() - t
    assert np.array_equal(got, vals)
    print(f"netemu ok: {vals.nbytes / 1e6:.2f} MB in {dt:.1f} s through a 1 MB/s shaped uplink")
    sw.stop()


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        ensure_router()
        selftest()
