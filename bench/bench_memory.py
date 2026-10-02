"""Memory and disk overhead of a node: resident memory (VmRSS, peak VmHWM) of real `zt node` processes when idle,
after seeding a replica, and while a client downloads from it; on-disk state besides the data itself.

python bench/bench_memory.py REPLICA.zarr [--out bench/memory.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate as S  # noqa: E402
from procswarm import ProcSwarm  # noqa: E402
from zarr_torrent.store import http, wait_job  # noqa: E402


def mem(pid):
    st = dict(line.split(":", 1) for line in open(f"/proc/{pid}/status").read().splitlines() if ":" in line)
    return round(int(st["VmRSS"].split()[0]) / 1024, 1), round(int(st["VmHWM"].split()[0]) / 1024, 1)


def du(p, skip=("cas",)):
    return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file() and not set(f.parts) & set(skip)) / 1e6, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replica")
    ap.add_argument("--out", default="bench/memory.json")
    a = ap.parse_args()
    root = Path("~/.cache/zt_mem").expanduser()
    sw = ProcSwarm(root, 2, 1, 0.0, seed=1, relay_rate=None)  # boot0 + p00
    seeder = sw.nodes["p00"].proc.pid
    time.sleep(3)
    res = {"idle_node_MB": mem(seeder)[0]}
    t = time.time()
    link = http(sw.ctl("p00"), "POST", "/api/seed", {"path": str(Path(a.replica).resolve())})["link"]
    res["seed_s"] = round(time.time() - t, 1)
    st = http(sw.ctl("p00"), "GET", "/api/status")
    n_chunks = sum(len(s.get("chunks", {})) if isinstance(s.get("chunks"), dict) else s.get("chunks", 0) for s in st["seeds"])
    res["seeded_chunks"] = n_chunks
    res["replica_GB"] = round(sum(f.stat().st_size for f in Path(a.replica).rglob("*") if f.is_file()) / 1e9, 2)
    res["seeder_after_seed_MB"] = mem(seeder)[0]
    client = S.fresh_client(sw, "mem", "maxflow")
    cpid = sw.nodes[client].proc.pid
    res["client_idle_MB"] = mem(cpid)[0]
    grid = link.removeprefix("zt://").split("+")[0]
    for q, reg in (("day_of_maps", {"t0": "2020-01-02", "t1": "2020-01-02", "step": 3600}),
                   ("month_6h", {"t0": "2020-01-01", "t1": "2020-01-31", "step": 21600})):
        t = time.time()
        jid = http(sw.ctl(client), "POST", "/api/download", {"grid": grid, "region": dict(reg, var="2m_temperature")})["job"]
        job = wait_job(sw.ctl(client), jid)
        res[q] = {"MB": round(job["bytes"] / 1e6, 1), "s": round(time.time() - t, 1), "state": job["state"],
                  "client_rss_MB": mem(cpid)[0], "client_peak_MB": mem(cpid)[1], "seeder_peak_MB": mem(seeder)[1]}
    res["seeder_state_on_disk_MB"] = du(root / "h_p00")
    res["client_state_on_disk_MB"] = du(root / f"h_{client}")
    res["client_cache_MB"] = round(sum(f.stat().st_size for f in (root / f"h_{client}" / "cas").rglob("*") if f.is_file()) / 1e6, 1)
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    sw.stop()


if __name__ == "__main__":
    main()
