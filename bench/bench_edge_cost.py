"""CPU cost of value identity on a small host: per real chunk, decode + vcid (value hash) vs cid (byte hash) vs the
xt1 transport codec. Reports CPU seconds per GB of decoded values (process_time, single thread).

python bench/bench_edge_cost.py sample  VARIANTS_DIR OUT_DIR [--k 8]   # copy K chunk files + metadata per variant
python bench/bench_edge_cost.py measure OUT_DIR [--json out.json]      # run on the target host
"""
import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VAR = "2m_temperature"


def sample(vdir: Path, out: Path, k: int):
    for v in sorted(vdir.glob("V*.zarr")):
        a = v / VAR
        meta = [f for f in ("zarr.json", ".zarray", ".zattrs") if (a / f).exists()]
        files = sorted(p for p in a.rglob("*") if p.is_file() and p.name not in meta)[:k]
        dst = out / v.stem
        dst.mkdir(parents=True, exist_ok=True)
        for f in meta:
            shutil.copy(a / f, dst / f)
        for i, p in enumerate(files):
            shutil.copy(p, dst / f"chunk{i:03d}")
        print(v.stem, len(files), "chunks", sum(p.stat().st_size for p in files) // 1024, "KB")


def _exact(v):
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(v.tobytes())
    return h.hexdigest()


def measure(out: Path):
    from zarr_torrent import codec
    from zarr_torrent.common import cid_of
    res = {"host": platform.node(), "machine": platform.machine(), "python": platform.python_version(), "variants": {}}
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        docs = {f: json.loads((d / f).read_text()) for f in ("zarr.json", ".zarray", ".zattrs") if (d / f).exists()}
        raws = [p.read_bytes() for p in sorted(d.glob("chunk*"))]
        vals = [codec.decode(docs, r) for r in raws]  # warm-up + decoded sizes
        nb = sum(v.nbytes for v in vals)
        row = {"chunks": len(raws), "stored_MB": round(sum(map(len, raws)) / 1e6, 2), "decoded_MB": round(nb / 1e6, 2)}
        for name, fn, rep in (("decode", lambda: [codec.decode(docs, r) for r in raws], 3),
                              ("vcid", lambda: [codec.vcid_of(v) for v in vals], 3),  # lattice id (default)
                              ("vcid_exact", lambda: [_exact(v) for v in vals], 3),
                              ("cid", lambda: [cid_of(r) for r in raws], 3),
                              ("xt1_encode", lambda: [codec.xt1_encode(v) for v in vals], 1)):
            t = time.process_time()
            for _ in range(rep):
                fn()
            cpu = (time.process_time() - t) / rep
            row[f"{name}_ms_per_chunk"] = round(1e3 * cpu / len(raws), 2)
            row[f"{name}_s_per_GB"] = round(cpu / (nb / 1e9), 2)
        row["verify_s_per_GB"] = round(row["decode_s_per_GB"] + row["vcid_s_per_GB"], 2)  # what value identity adds
        row["verify_exact_s_per_GB"] = round(row["decode_s_per_GB"] + row["vcid_exact_s_per_GB"], 2)
        res["variants"][d.name] = row
        print(d.name, json.dumps(row), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "measure"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(Path(a.paths[0]).expanduser(), Path(a.paths[1]).expanduser(), a.k)
    else:
        r = measure(Path(a.paths[0]).expanduser())
        if a.json:
            json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
