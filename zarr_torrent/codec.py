"""Decode/encode a single stored chunk using only its array metadata docs.

This is what makes codec-agnostic swarming possible: a chunk fetched from a peer
that compressed it differently is decoded with *that peer's* metadata, checked
against the value-level id (vcid) and re-encoded with the local view metadata.
"""
import copy
import hashlib
import json
import threading

import numpy as np
import zarr
from zarr.core.buffer import cpu
from zarr.storage import MemoryStore


def chunk_shape(docs: dict) -> list[int]:
    if "zarr.json" in docs:
        m = docs["zarr.json"]
        cs = m["chunk_grid"]["configuration"]["chunk_shape"]
        return list(cs)
    return list(docs[".zarray"]["chunks"])


def _one_chunk_array(docs: dict, raw: bytes | None, read_only: bool):
    """Array whose shape == one chunk, holding `raw` as its only chunk."""
    cs = chunk_shape(docs)
    d = {}
    if "zarr.json" in docs:
        m = copy.deepcopy(docs["zarr.json"])
        m["shape"] = cs
        m["attributes"] = {}
        m.pop("storage_transformers", None)
        d["zarr.json"] = cpu.Buffer.from_bytes(json.dumps(m).encode())
    else:
        m = dict(docs[".zarray"], shape=cs)
        d[".zarray"] = cpu.Buffer.from_bytes(json.dumps(m).encode())
    store = MemoryStore(d)
    arr = zarr.open_array(store=store, mode="r" if read_only else "r+")
    key = arr.metadata.encode_chunk_key((0,) * len(cs))
    if raw is not None:
        d[key] = cpu.Buffer.from_bytes(raw)
    return arr, d, key


_tls = threading.local()


def decode(docs: dict, raw: bytes) -> np.ndarray:
    """Decode one stored chunk. The one-chunk array is built once per (thread, metadata) and reused."""
    cache = getattr(_tls, "arrays", None)
    if cache is None:
        cache = _tls.arrays = {}
    k = json.dumps(docs, sort_keys=True)
    hit = cache.get(k)
    if hit is None:
        if len(cache) > 256:
            cache.clear()
        hit = cache[k] = _one_chunk_array(docs, None, read_only=True)
    arr, d, key = hit
    d[key] = cpu.Buffer.from_bytes(raw)
    try:
        return arr[...]
    finally:
        d.pop(key, None)


def encode(docs: dict, values: np.ndarray) -> bytes:
    with zarr.config.set({"array.write_empty_chunks": True}):
        arr, d, key = _one_chunk_array(docs, None, read_only=False)
        arr[...] = values
    return d[key].to_bytes()


VALUE_ID = __import__("os").environ.get("ZT_VALUE_ID", "lattice")  # "lattice" | "exact"
ESTIMATOR = "lattice-v3"  # bump when lattice_of changes: cached value ids of an older estimator are not reused


def lattice_of(x: np.ndarray):
    """Quantization lattice of measured values v = phi + k s. Returns (s, phi, kmed) or None (< 2 distinct finite
    values); kmed is the code of the median DISTINCT value (distinct values of two decodings correspond one to one).

    Two families, tried in order; the id built from them does not depend on which one succeeded, so two decodings that
    land on the same lattice by different routes still agree:
    * linear (scale/offset: NIfTI scl_slope, NetCDF scale_factor, FITS BSCALE, ADC gain): step = the smallest gap
      between distinct values (or 1/d of it), refined by least squares; accepted if every distinct value lies within
      5% of a step of the fitted lattice - decoders that differ by rounding only (float32 vs float64) pass;
    * binary (GRIB R + k 2^E): the coarsest s = 2^e whose codes keep every distinct value distinct, phi the circular
      mean - tolerates decoders that are off by up to ~0.4 s (the ARCO vs NCAR ERA5 case).
    One sort; on sorted distinct values codes are non-decreasing, so injectivity is "no two neighbours share a code"."""
    u = np.unique(x[np.isfinite(x)]).astype("f8")  # sort in the native dtype, widen only the distinct values
    if u.size < 2:
        return None
    g0 = float(np.min(np.diff(u)))
    gaps = np.diff(u)
    for d in range(1, 9):  # linear lattice with a small rounding error
        # codes from neighbouring gaps, not from (u - u0)/g0: in float32 a decimal step (GFS 2 m temperature,
        # 0.01 K near 294 K) shows up as gaps of 0.009979 and 0.010010, and rounding against the smallest gap drifted
        # by half a step over a few thousand codes, so the scale/offset family was rejected
        # the unit is the MEAN one-step gap (float32 rounding alternates 0.1953 / 0.2031 for a 0.2 step): rounding a
        # 93-step gap against the smallest one gave 95. Two least-squares refinements absorb what is left.
        unit = float(np.mean(gaps[gaps < 1.5 * g0])) / d
        q = gaps / unit
        if np.max(np.abs(q - np.round(q))) > 0.25:  # gaps are not whole multiples of this unit: no lattice here
            continue
        for _ in range(3):
            k = np.concatenate(([0.0], np.cumsum(np.round(gaps / unit))))
            if u.size <= 2:
                s_fit, a_fit = unit, u[0]
                break
            km = k.mean()  # least squares in closed form (np.polyfit was the per-slice cost on small tiles)
            s_fit = float(np.dot(k - km, u - u.mean()) / np.dot(k - km, k - km))
            a_fit = float(u.mean() - s_fit * km)
            unit = s_fit
        if s_fit > 0 and np.all(np.abs(u - (a_fit + s_fit * k)) <= 0.05 * s_fit) and np.all(np.diff(k) > 0):
            lin = float(s_fit), float(a_fit), int(k[u.size // 2])
            break
    else:
        lin = None
    bin_ = _binary_lattice(u, g0)
    # Both families may fit. The binary lattice replaces the linear one only if it is coarser AND every value sits
    # clearly inside its cell (<= 0.45 of a step from a lattice point): a decoder whose errors are commensurate with a
    # fraction of the step (+0.27 / -0.13 of it) also lies on a finer linear lattice, and its power-of-two lattice is
    # the faithful one; but on a sparse tile a power of two may be coarser than the true step with values exactly on
    # cell borders, where any decoding error flips a code - there the linear lattice stays.
    if lin and bin_ and bin_[0] > lin[0] * (1 + 1e-6):
        r = (u - bin_[1]) / bin_[0]
        if np.max(np.abs(r - np.round(r))) <= 0.45:
            return bin_
    return lin or bin_


def _binary_lattice(u, g0):
    """Coarsest s = 2^e whose codes keep the sorted distinct values u distinct; phase = circular mean."""
    sub = u[:: max(1, u.size // 4096)]
    # start well above the smallest gap: a second decoder can put two adjacent codes closer than one step
    # (deviations -0.4 s and +0.4 s leave 0.2 s), yet they still round to different codes at step s
    e_hi = int(np.floor(np.log2(g0))) + 4
    for e in range(e_hi, e_hi - 44, -1):
        st = 2.0 ** e
        ang = 2 * np.pi * (sub / st % 1.0)
        phi = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) % 1.0) * st
        if not np.any(np.diff(np.round((u - phi) / st)) == 0):
            return st, float(phi), int(np.round((u[u.size // 2] - phi) / st))
    return None


def vcid_of(values: np.ndarray, taxis: int | None = None) -> str:
    """Value-level id, independent of compressor/filters/byte order/format.

    Floating-point chunks (ZT_VALUE_ID=lattice): per time slice, the quantization lattice (step s, phase phi) is
    estimated from the slice alone and the values become integer codes. The hash covers the codes RELATIVE to their
    median (a phase estimated near 0 by one decoder and near s by another shifts all codes by one: relative codes
    are immune), and the absolute level travels next to it: "L:<hash>:<level>:<step>". Two ids denote the same
    values iff the hashes match and the levels differ by less than a quarter step (same_vcid). The phase is a mean
    over thousands of values, so decoders agree on the level far more tightly than half a step, while a field
    shifted by one step or more (other units, other data) stays apart. Other dtypes, and ZT_VALUE_ID=exact, hash
    the exact values (plain hex id)."""
    v = np.ascontiguousarray(values)
    v = v.astype(v.dtype.newbyteorder("<"), copy=False)
    h = hashlib.blake2b(digest_size=16)
    if VALUE_ID != "lattice" or v.dtype.kind != "f" or v.size == 0:
        h.update(f"{v.dtype.str}|{v.shape}|".encode())
        h.update(v.tobytes())
        return h.hexdigest()
    h.update(f"L|{v.dtype.kind}|{v.shape}|".encode())
    slices = [v] if taxis is None or v.ndim == 0 else [np.take(v, i, axis=taxis) for i in range(v.shape[taxis])]
    levels, steps = [], []
    for sl in slices:
        fin = np.isfinite(sl)
        h.update(np.packbits(fin).tobytes())  # where the missing values are
        lat = lattice_of(sl)
        if lat is None:  # constant (or empty) slice: its exact value
            h.update(b"C" + np.unique(sl[fin]).astype("<f8").tobytes())
            continue
        st, phi, kmed = lat
        rel = np.round((sl[fin].astype("f8") - phi) / st) - kmed
        small = rel.size == 0 or (rel.min() >= -2 ** 31 and rel.max() < 2 ** 31)
        h.update(b"|")
        h.update(rel.astype("<i4" if small else "<i8").tobytes())
        levels.append(phi + kmed * st)
        steps.append(st)
    if not levels:
        return "L:" + h.hexdigest() + ":0:0"
    return f"L:{h.hexdigest()}:{float(np.mean(levels))!r}:{float(np.mean(steps))!r}"


_SENT = np.iinfo(np.int32).min  # code of a missing (non-finite) value


def _slices(v, taxis):
    return [(None, v)] if taxis is None or v.ndim == 0 else [(i, np.take(v, i, axis=taxis)) for i in range(v.shape[taxis])]


def lattice_codes(values: np.ndarray, taxis: int | None = None, exact: bool = True):
    """Decoder-invariant representation for value-level erasure coding: per time slice the integer codes relative to
    the median code (int32, missing = sentinel) and the slice's (step, level). Two decodings with equivalent vcids
    have identical codes, so a stripe coded over codes can be repaired with members from ANY decoder. Returns None
    unless the codes reproduce THESE values bit for bit (unquantized floats, infinities, overflow): the caller then
    codes raw bytes, which is exact but needs bit-identical members. exact=False (a repairer reading a member from
    another decoder, already matched by vcid) skips that check: only the codes are needed."""
    v = np.asarray(values)
    if VALUE_ID != "lattice" or v.dtype.kind != "f" or v.size == 0:
        return None
    codes = np.empty(v.shape, dtype="<i4")
    params = []
    for i, sl in _slices(v, taxis):
        fin = np.isfinite(sl)
        out = np.full(sl.shape, _SENT, dtype=np.int64)
        lat = lattice_of(sl)
        if lat is None:
            u = np.unique(sl[fin])
            params.append(["C", float(u[0]) if u.size else 0.0])
            out[fin] = 0
        else:
            st, phi, kmed = lat
            out[fin] = np.round((sl[fin].astype("f8") - phi) / st) - kmed
            params.append([st, phi + kmed * st])
        if out[fin].size and (out[fin].min() <= _SENT or out[fin].max() > np.iinfo(np.int32).max):
            return None
        if i is None:
            codes[...] = out
        else:
            idx = [slice(None)] * v.ndim
            idx[taxis] = i
            codes[tuple(idx)] = out
    if not exact:
        return codes, params
    back = from_lattice_codes(codes, params, v.dtype, taxis)
    if not np.array_equal(back, v, equal_nan=True) or not np.array_equal(np.isnan(back), np.isnan(v)):
        return None
    return codes, params


def from_lattice_codes(codes: np.ndarray, params: list, dtype, taxis: int | None = None) -> np.ndarray:
    """Values from lattice_codes(): level + code * step per slice, missing -> NaN."""
    c = np.asarray(codes)
    out = np.empty(c.shape, dtype=np.float64)
    for (i, sl), prm in zip(_slices(c, taxis), params):
        vals = np.full(sl.shape, np.nan)
        ok = sl != _SENT
        vals[ok] = prm[1] if prm[0] == "C" else prm[1] + sl[ok].astype("f8") * prm[0]
        if i is None:
            out[...] = vals
        else:
            idx = [slice(None)] * c.ndim
            idx[taxis] = i
            out[tuple(idx)] = vals
    return out.astype(dtype)


def same_vcid(a: str, b: str) -> bool:
    """Do two value ids denote the same values? Exact ids: equality. Lattice ids: same codes, level within 1/4 step."""
    if a == b:
        return True
    if not (a.startswith("L:") and b.startswith("L:")):
        return False
    _, ha, la, sa = a.split(":")
    _, hb, lb, sb = b.split(":")
    sa, sb = float(sa), float(sb)
    return ha == hb and abs(sa - sb) <= 1e-5 * max(sa, sb) and abs(float(la) - float(lb)) <= sa / 4


# ---------------------------------------------------------------- transport codec "xt1" (value-level, lossless)
# Temporal XOR prediction along axis 0 + bitshuffle + zstd. Only used on the wire: the receiver decodes it,
# checks the value-level id (vcid) and re-encodes into its own layout. Header: dtype str | shape (json).
XT1 = b"XT1\0"


def xt1_encode(values: np.ndarray) -> bytes:
    from numcodecs import Blosc
    v = np.ascontiguousarray(values)
    if v.dtype.kind not in "fiu" or v.dtype.itemsize not in (4, 8) or v.ndim == 0:
        raise ValueError("xt1: unsupported dtype/shape")
    u = v.view(f"<u{v.dtype.itemsize}").copy()
    if len(u) > 1:
        u[1:] ^= v.view(f"<u{v.dtype.itemsize}")[:-1]
    hdr = json.dumps({"d": v.dtype.newbyteorder("<").str, "s": list(v.shape)}).encode()
    body = Blosc("zstd", 5, Blosc.BITSHUFFLE).encode(u)
    return XT1 + len(hdr).to_bytes(4, "big") + hdr + bytes(body)


def xt1_decode(b: bytes) -> np.ndarray:
    from numcodecs import Blosc
    n = int.from_bytes(b[4:8], "big")
    h = json.loads(b[8:8 + n])
    dt = np.dtype(h["d"])
    u = np.frombuffer(Blosc("zstd", 5, Blosc.BITSHUFFLE).decode(b[8 + n:]), dtype=f"<u{dt.itemsize}")
    u = u.reshape(h["s"]).copy()
    for t in range(1, len(u)):
        u[t] ^= u[t - 1]
    return u.view(dt)
