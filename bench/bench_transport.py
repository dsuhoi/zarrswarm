"""Transport-codec study on real ERA5 chunks: how many bytes must cross the wire per chunk, and at what CPU cost.

Every candidate is lossless on values, so the receiver can still verify the chunk by its value-level id (vcid)
after decoding - the property that makes transport re-encoding safe in zarr-torrent.
python bench/bench_transport.py [ZARR] [--chunk 100]
"""
import argparse
import json
import time
from pathlib import Path

import numcodecs
import numpy as np
import xarray as xr
from numcodecs import Blosc, Zstd

BIT = Blosc.BITSHUFFLE


def xor_time(a: np.ndarray) -> np.ndarray:
    """Intra-chunk temporal prediction: step t stored as bits(t) XOR bits(t-1) (lossless, self-contained)."""
    u = a.view(np.uint32) if a.dtype.itemsize == 4 else a.view(np.uint64)
    out = u.copy()
    out[1:] ^= u[:-1]
    return out


def unxor_time(u: np.ndarray, dtype) -> np.ndarray:
    out = u.copy()
    for t in range(1, len(out)):
        out[t] ^= out[t - 1]
    return out.view(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr", nargs="?", default="~/.cache/zt_real/wb2/wb2_oracle.zarr")
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args()
    ds = xr.open_zarr(Path(a.zarr).expanduser(), consolidated=False)
    res = {}
    for var in [v for v in ds.data_vars][:4]:
        arr = ds[var].values.astype("float32")
        T = arr.shape[0] // a.chunk * a.chunk
        blocks = [np.ascontiguousarray(arr[i:i + a.chunk]) for i in range(0, T, a.chunk)][:20]
        raw = sum(b.nbytes for b in blocks)
        cands = {
            "zstd-1 (≈ zarr default)": (lambda b, prev: Zstd(1).encode(b), None),
            "zstd-19": (lambda b, prev: Zstd(19).encode(b), None),
            "blosc-zstd5 bitshuffle": (lambda b, prev: Blosc("zstd", 5, BIT).encode(b), None),
            "xor-time + blosc bitshuffle": (lambda b, prev: Blosc("zstd", 5, BIT).encode(xor_time(b)), None),
            "xor vs prev chunk + blosc": (lambda b, prev: Blosc("zstd", 5, BIT).encode(
                (b.view(np.uint32) ^ prev.view(np.uint32)) if prev is not None and prev.shape == b.shape
                else b.view(np.uint32)), None),
        }
        out = {}
        for name, (enc, _) in cands.items():
            t0 = time.perf_counter()
            encs = [enc(b, blocks[i - 1] if i else None) for i, b in enumerate(blocks)]
            te = time.perf_counter() - t0
            nbytes = sum(len(e) for e in encs)
            out[name] = {"ratio": round(raw / nbytes, 2), "enc_MBps": round(raw / 1e6 / te, 1)}
        # decode speed + losslessness for the xor-time transport codec
        codec = Blosc("zstd", 5, BIT)
        e = [codec.encode(xor_time(b)) for b in blocks]
        t0 = time.perf_counter()
        dec = [unxor_time(np.frombuffer(codec.decode(x), dtype=np.uint32).reshape(b.shape), np.float32)
               for x, b in zip(e, blocks)]
        td = time.perf_counter() - t0
        assert all(np.array_equal(d, b, equal_nan=True) for d, b in zip(dec, blocks))
        out["xor-time + blosc bitshuffle"]["dec_MBps"] = round(raw / 1e6 / td, 1)
        res[var] = out
        print(var, json.dumps(out, ensure_ascii=False))
    Path("bench/results_transport.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
