"""Pinned native GFS fields through independent node processes; local sockets, no WAN speed claim."""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sim"), str(ROOT / "bench")]
from gfs_native_precision import native_field
from procswarm import ProcSwarm
from simulate import port
import zarrswarm as zt
from zarrswarm.common import h160, verify
from zarrswarm.node import merge_view
from zarrswarm.scan import scan
from zarrswarm.store import http


def manifest(swarm, name, grid):
    status = http(swarm.ctl(name), "GET", "/api/status")
    with urllib.request.urlopen(f"{status['addr']}/m/{grid}", timeout=30) as r:
        body = zlib.decompress(r.read())
        pk, sig = r.headers["X-Zt-Pk"], r.headers["X-Zt-Sig"]
    assert h160(bytes.fromhex(pk)) == status["id"] and verify(pk, sig, body)
    return status["id"], json.loads(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--sample", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run
    sample = json.loads(args.sample.read_text())
    source, mirror, changed, params, units = {}, {}, {}, {}, {}
    reference = datetime.fromisoformat(sample["protocol"]["run"])
    times = [reference + timedelta(hours=h) for h in (0, 6)]
    for row in sample["fields"]:
        a, b, lat, lon, triple = native_field(Path(sample["work"]), row)
        name = row["short_name"]
        source.setdefault(name, []).append(a)
        mirror.setdefault(name, []).append(b)
        bad = a.copy()
        bad[::32, ::32] += triple[0]  # one actual source-count change in each of the 64 paired tiles
        changed.setdefault(name, []).append(bad)
        params.setdefault(name, {"time": {}})["time"][str(int((reference + timedelta(hours=row["lead_hours"])).timestamp()))] = list(triple)
        units[name] = row["units"]
    source, mirror, changed = ({n: np.stack(v) for n, v in d.items()} for d in (source, mirror, changed))
    coords = {"time": np.array([t.replace(tzinfo=None) for t in times], dtype="datetime64[ns]"),
              "latitude": lat, "longitude": lon}
    paths = {}
    for label, data, chunks, fmt in (("publisher", source, (1, 32, 32), 2),
                                     ("mirror", mirror, (1, 32, 32), 2),
                                     ("alternate", mirror, (2, 16, 64), 3),
                                     ("changed", changed, (1, 32, 32), 2)):
        path = args.work / f"{label}.zarr"
        xr.Dataset({n: (("time", "latitude", "longitude"), v, {"units": units[n]}) for n, v in data.items()},
                   coords=coords).to_zarr(path, encoding={n: {"chunks": chunks} for n in data},
                                          consolidated=False, zarr_format=fmt)
        paths[label] = str(path.resolve())
    default = {label: next(iter(scan(paths[label])["subgrids"].values())) for label in ("publisher", "mirror")}
    out = {"started_utc": datetime.now(timezone.utc).isoformat(), "sample": str(args.sample),
           "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
           "protocol_sha256": hashlib.sha256((ROOT / "bench/revalidation/integration_protocol_v9.json").read_bytes()).hexdigest(),
           "scope": "Six independent processes on one host, loopback sockets; retained native ecCodes float64 and Unidata float32, source publisher signs extracted GRIB parameters, 0.45-step common bound. Not a station or WAN throughput test.",
           "layouts": [[1, 32, 32], [2, 16, 64]], "formats": [2, 3],
           "default_shared_value_families": sum(default["publisher"]["arrays"][n]["vfid"] == default["mirror"]["arrays"][n]["vfid"] for n in source),
           "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
               "bench/packing_network.py", "bench/gfs_native_precision.py", "sim/procswarm.py",
               "zarrswarm/codec.py", "zarrswarm/scan.py", "zarrswarm/node.py", "zarrswarm/cli.py")}}
    def save(): args.out.write_text(json.dumps(out, indent=2) + "\n")
    save()
    swarm = ProcSwarm(args.work / "nodes", 1, 1, 0, 91)
    kw = lambda: {"port": port(), "bootstrap": swarm.boots, "rate": 1e9, "latency": 0}
    try:
        swarm._start("publisher", kw())
        authority = http(swarm.ctl("publisher"), "GET", "/api/status")
        swarm.env["ZT_TRUST"] = authority["pk"]
        seeded = http(swarm.ctl("publisher"), "POST", "/api/seed", {"path": paths["publisher"], "packing": params})
        grid, contract = seeded["link"].removeprefix("zt://"), seeded["packing"]
        out.update(grid_id=grid, publisher_pk=authority["pk"], signed_contracts=contract)
        for name in ("mirror", "alternate", "changed", "client"):
            swarm._start(name, kw())
            if name in paths:
                http(swarm.ctl(name), "POST", "/api/seed", {"path": paths[name], "packing": contract})
        mans = dict(manifest(swarm, name, grid) for name in ("publisher", "mirror", "alternate", "changed"))
        view = merge_view(grid, mans, "fresh-client", {authority["id"]})
        original = mans[authority["id"]]
        keys = [k for k in original["chunks"] if k.partition("@")[0] in source]
        changed_id = http(swarm.ctl("changed"), "GET", "/api/status")["id"]
        mirror_id = http(swarm.ctl("mirror"), "GET", "/api/status")["id"]
        out["paired_chunks"] = len(keys)
        out["pooled_native_pairs"] = sum(any(p == mirror_id for p, _, _ in view["best"][k]["src"]) for k in keys)
        out["accepted_source_count_changes"] = sum(any(p == changed_id for p, _, _ in view["best"][k]["src"]) for k in keys)
        assert out["paired_chunks"] == out["pooled_native_pairs"] == 64 and out["accepted_source_count_changes"] == 0
        save()
        swarm.kill("publisher")
        got = zt.open_dataset(seeded["link"], ctl=swarm.ctl("client")).load()
        out["publisher_offline"] = {}
        for name in source:
            assert got[name].shape == source[name].shape
            equal = 0
            for i, t in enumerate(times):
                step, origin, error = params[name]["time"][str(int(t.timestamp()))]
                expected = np.rint((source[name][i] - origin) / step)
                actual = np.rint((got[name].values[i].astype("f8") - origin) / step)
                equal += int(np.count_nonzero(actual == expected))
                assert np.array_equal(actual, expected)
            out["publisher_offline"][name] = {"cells": source[name].size, "matching_source_codes": equal,
                                               "max_abs_difference": float(np.max(np.abs(got[name].values - source[name])))}
        save()
        # Explicit stop flushes the current cache before its persisted contract is inspected.
        swarm.kill("client")
        cache_files = list((swarm.root / "h_client" / "cache").glob("*.json"))
        for name in ("mirror", "alternate", "changed"): swarm.kill(name)
        assert cache_files and all(json.loads(p.read_text())["arrays"][n]["packing"] == contract[n]
                                   for p in cache_files for n in source)
        swarm._start("client", kw())
        cached = zt.open_dataset(seeded["link"], ctl=swarm.ctl("client")).load()
        assert all(np.array_equal(cached[n].values, got[n].values) for n in source)
        out.update(cache_only_restart=True, completed_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({k: out[k] for k in ("paired_chunks", "pooled_native_pairs", "accepted_source_count_changes", "cache_only_restart")}), flush=True)
    except Exception as e:
        out["failure"] = repr(e)
        raise
    finally:
        logs = args.work / "logs"
        logs.mkdir(exist_ok=True)
        for p in swarm.root.glob("*.log"):
            shutil.copyfile(p, logs / p.name)
        swarm.stop()
        out["all_processes_stopped"] = all(n.proc.poll() is not None for n in swarm.nodes.values())
        save()


if __name__ == "__main__":
    main()
