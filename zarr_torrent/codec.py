"""Decode/encode a single stored chunk using only its array metadata docs.

This is what makes codec-agnostic swarming possible: a chunk fetched from a peer
that compressed it differently is decoded with *that peer's* metadata, checked
against the value-level id (vcid) and re-encoded with the local view metadata.
"""
import copy
import hashlib
import json
import threading
from functools import lru_cache

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
ESTIMATOR = "lattice-v5"  # per-slice centers, steps and radii; invalidate private and shared scan caches


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


def vcid_of(values: np.ndarray, taxis: int | None = None, *, packing=None) -> str:
    """L3:<code hash>:<per-slice [level, step, code radius] triples>; constant slices use null parameters.

    Each slice is compared separately: opposing level/step changes cannot cancel. The hash includes shape,
    time axis, code widths and separate NaN/+inf/-inf masks. Lattice fitting remains an empirical estimator,
    not a guarantee of recovering source packing. Non-floats and exact mode retain byte-exact value identity.
    packing=(step, origin, max_error), or {"slices": [triples]}, uses caller-trusted source cells.
    It never estimates its parameters; values outside the error budget fail closed. The caller must establish
    metadata provenance and the decoder error bound.
    """
    v = np.ascontiguousarray(values)
    v = v.astype(v.dtype.newbyteorder("<"), copy=False)
    if packing is not None:
        return _packing_vcid(v, taxis, packing)
    h = hashlib.blake2b(digest_size=16)
    if VALUE_ID != "lattice" or v.dtype.kind != "f" or v.size == 0:
        h.update(f"{v.dtype.str}|{v.shape}|".encode())
        h.update(v.tobytes())
        return h.hexdigest()
    if taxis is not None:
        if not -v.ndim <= taxis < v.ndim:
            raise ValueError("time axis outside array dimensions")
        taxis %= v.ndim
    h.update(f"L3|{v.shape}|{taxis}|".encode())
    slices = [v] if taxis is None or v.ndim == 0 else [np.take(v, i, axis=taxis) for i in range(v.shape[taxis])]
    params = []
    for sl in slices:
        fin = np.isfinite(sl)
        for mask in (fin, np.isnan(sl), np.isposinf(sl), np.isneginf(sl)):
            h.update(np.packbits(mask).tobytes())
        lat = lattice_of(sl)
        if lat is None:  # constant (or empty) slice: its exact value
            h.update(b"C" + np.unique(sl[fin]).astype("<f8").tobytes())
            params.append(None)
            continue
        st, phi, kmed = lat
        rel = np.round((sl[fin].astype("f8") - phi) / st) - kmed
        small = rel.size == 0 or (rel.min() >= -2 ** 31 and rel.max() < 2 ** 31)
        if not np.all(np.isfinite(rel)) or np.any(np.abs(rel) >= 2 ** 53):
            # Unrepresentable integer codes cannot be hashed by a narrowing cast.
            h.update(b"E" + sl.astype("<f8").tobytes())
            params.append(None)
            continue
        h.update(b"i4" if small else b"i8")
        h.update(rel.astype("<i4" if small else "<i8").tobytes())
        params.append([phi + kmed * st, st, int(np.max(np.abs(rel)))])
    return f"L3:{h.hexdigest()}:{json.dumps(params, separators=(',', ':'), allow_nan=False)}"


def _packing_vcid(v, taxis, packing):
    """Fixed source cells have unique codes when the certified decoder error is less than half a step."""
    if VALUE_ID != "lattice" or v.dtype.kind != "f":
        raise ValueError("source packing requires floating-point data in lattice mode")
    if taxis is not None:
        if not -v.ndim <= taxis < v.ndim:
            raise ValueError("time axis outside array dimensions")
        taxis %= v.ndim
    if isinstance(packing, dict):
        slices = _slices(v, taxis)
        if set(packing) != {"slices"} or len(packing["slices"]) != len(slices):
            raise ValueError("one source packing triple is required per slice")
        params = [packing_parameters(p) for p in packing["slices"]]
        if params and all(p == params[0] for p in params):
            return _packing_vcid(v, taxis, params[0])
        h = hashlib.blake2b(digest_size=16)
        h.update(f"P2|{v.shape}|{taxis}|".encode())
        for (_, sl), prm in zip(slices, params):
            h.update(bytes.fromhex(_packing_vcid(sl, None, prm)))
        return h.hexdigest()
    step, origin, error = packing_parameters(packing)
    codes = _source_codes(v, (step, origin, error))
    h = hashlib.blake2b(digest_size=16)
    # The budget is a validation policy, not part of the identity. Two valid decoders can have different bounds.
    h.update(f"P1|{v.shape}|{taxis}|{step.hex()}|{origin.hex()}|".encode())
    for mask in (np.isfinite(v), np.isnan(v), np.isposinf(v), np.isneginf(v)):
        h.update(np.packbits(mask).tobytes())
    h.update(codes.astype("<i8").tobytes())
    return h.hexdigest()


def packing_parameters(packing):
    if not isinstance(packing, (tuple, list)) or len(packing) != 3:
        raise ValueError("packing must be (step, origin, max_error)")
    if any(type(x) not in (int, float) for x in packing):
        raise ValueError("packing parameters must be numbers")
    step, origin, error = map(float, packing)
    if not all(np.isfinite(x) for x in (step, origin, error)) or not 0 < error < step / 2:
        raise ValueError("packing needs finite parameters and 0 < max_error < step/2")
    return step, origin, error


def _source_codes(v, packing):
    step, origin, error = packing
    finite = v[np.isfinite(v)].astype("f8")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        codes = np.rint((finite - origin) / step)
        centers = origin + codes * step
        # Refuse a contract that floating-point arithmetic cannot resolve safely; never silently narrow codes.
        roundoff = 8 * np.finfo("f8").eps * np.maximum(np.maximum(np.abs(finite), abs(origin)), np.abs(centers))
        if (not np.all(np.isfinite(codes)) or np.any(np.abs(codes) >= 2 ** 52) or
                np.any(roundoff >= step / 2 - error) or
                np.any(np.abs(finite - centers) + roundoff > error)):
            raise ValueError("values exceed the source packing error budget or numerical precision")
    return codes


_SENT = np.iinfo(np.int32).min  # code of a missing (non-finite) value


def _slices(v, taxis):
    return [(None, v)] if taxis is None or v.ndim == 0 else [(i, np.take(v, i, axis=taxis)) for i in range(v.shape[taxis])]


def lattice_codes(values: np.ndarray, taxis: int | None = None, exact: bool = True, *, packing=None):
    """Decoder-invariant representation for value-level erasure coding: per time slice the integer codes relative to
    the median code (int32, missing = sentinel) and the slice's (step, level). Two decodings with equivalent vcids
    have identical codes, so a stripe coded over codes can be repaired with members from ANY decoder. Returns None
    unless the codes reproduce THESE values bit for bit (unquantized floats, infinities, overflow): the caller then
    codes raw bytes, which is exact but needs bit-identical members. exact=False (a repairer reading a member from
    another decoder, already matched by vcid) skips that check: only the codes are needed.
    A source packing contract supplies absolute codes and parameters instead of fitting them."""
    v = np.asarray(values)
    if VALUE_ID != "lattice" or v.dtype.kind != "f" or v.size == 0:
        return None
    if np.any(np.isinf(v)):
        return None  # the sentinel represents NaN only; infinities require exact byte parity
    codes = np.empty(v.shape, dtype="<i4")
    params = []
    slices = _slices(v, taxis)
    if packing is not None:
        _packing_vcid(v, taxis, packing)  # validate the entire contract before using its codes
        source = packing["slices"] if isinstance(packing, dict) else [packing] * len(slices)
    for j, (i, sl) in enumerate(slices):
        fin = np.isfinite(sl)
        out = np.full(sl.shape, _SENT, dtype=np.int64)
        lat = lattice_of(sl) if packing is None else None
        if packing is not None:
            st, origin, error = packing_parameters(source[j])
            out[fin] = _source_codes(sl, (st, origin, error))
            params.append([st, origin])
        elif lat is None:
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
    if back.tobytes() != v.tobytes():  # numerical equality ignores zero signs and NaN payloads
        return None
    return codes, params


def from_lattice_codes(codes: np.ndarray, params: list, dtype, taxis: int | None = None) -> np.ndarray:
    """Values from lattice_codes(): level + code * step per slice, missing -> NaN."""
    c = np.asarray(codes)
    slices = _slices(c, taxis)
    if len(params) != len(slices):
        raise ValueError("one lattice parameter pair is required per slice")
    out = np.empty(c.shape, dtype=np.float64)
    for (i, sl), prm in zip(slices, params):
        if len(prm) != 2 or not np.isfinite(prm[1]) or (prm[0] != "C" and
                (not np.isfinite(prm[0]) or prm[0] <= 0)):
            raise ValueError("invalid reconstruction parameters")
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


@lru_cache(maxsize=4096)
def vcid_params(vcid: str):
    """Parse the current lattice format; malformed and legacy identifiers fail closed."""
    try:
        version, digest, raw = vcid.split(":", 2)
        if version != "L3" or len(digest) != 32 or any(c not in "0123456789abcdef" for c in digest):
            return None
        params = json.loads(raw)
        if not isinstance(params, list):
            return None
        for p in params:
            if p is not None and (not isinstance(p, list) or len(p) != 3 or
                    any(type(x) not in (int, float) or not np.isfinite(x) for x in p) or p[1] <= 0 or type(p[2]) is not int or not 0 <= p[2] < 2 ** 53):
                return None
        return digest, tuple(None if p is None else tuple(p) for p in params)
    except (ValueError, TypeError, OverflowError):
        return None


def same_vcid(a: str, b: str) -> bool:
    """Matching code hashes and symmetric lattice-center bounds for EVERY slice; exact ids compare literally."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if a.startswith("L") or b.startswith("L"):
        pa, pb = vcid_params(a), vcid_params(b)
        if pa is None or pb is None or pa[0] != pb[0] or len(pa[1]) != len(pb[1]):
            return False
        for x, y in zip(pa[1], pb[1]):
            if x is None or y is None:
                if x != y:
                    return False
            elif abs(x[1] - y[1]) > 1e-5 * min(x[1], y[1]) or (abs(x[0] - y[0]) + max(x[2], y[2]) * abs(x[1] - y[1])) > min(x[1], y[1]) / 4:
                return False
        return True
    return a == b


def agrees_with_prefix(values: np.ndarray, prefix: np.ndarray, taxis: int | None = None,
                       valid_steps: int | None = None) -> bool:
    """A more complete chunk must preserve every known cell of its rival, with a separate tolerance per slice."""
    if values.shape != prefix.shape:
        return False
    if taxis is not None and valid_steps is not None:
        if not 0 < valid_steps <= prefix.shape[taxis]:
            return False
        selection = [slice(None)] * prefix.ndim
        selection[taxis] = slice(0, valid_steps)
        values, prefix = values[tuple(selection)], prefix[tuple(selection)]
    if prefix.dtype.kind != "f":
        return bool(np.array_equal(values, prefix))
    for (_, mine), (_, rival) in zip(_slices(values, taxis), _slices(prefix, taxis)):
        if not np.array_equal(np.isposinf(mine)[np.isinf(rival)], np.isposinf(rival)[np.isinf(rival)]):
            return False
        if not np.all(np.isinf(mine[np.isinf(rival)])):
            return False
        known = np.isfinite(rival)
        lat = lattice_of(rival)
        tol = lat[0] / 2 if lat else 0.0
        if not np.all(np.abs(mine[known].astype("f8") - rival[known].astype("f8")) <= tol):
            return False
    return True


# ---------------------------------------------------------------- transport codec "xt1" (value-level, lossless)
# Temporal XOR prediction along axis 0 + bitshuffle + zstd. Only used on the wire: the receiver decodes it,
# checks the value-level id (vcid) and re-encodes into its own layout. Header: dtype str | shape (json).
XT1 = b"XT1\0"


def xt1_encode(values: np.ndarray) -> bytes:
    from numcodecs import Blosc
    v = np.ascontiguousarray(values)
    if v.dtype.kind not in "fiu" or v.dtype.itemsize not in (4, 8) or v.ndim == 0:
        raise ValueError("xt1: unsupported dtype/shape")
    v = v.astype(v.dtype.newbyteorder("<"), copy=False)
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
