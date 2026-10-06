"""Scan a local Zarr (v2 or v3) directory into coordinate-aware, content-addressed form.

Identities produced:
  grid_id  - DHT key of a *sub-grid*: one variable's non-time dims + their coordinate values + time step
             and phase. A dataset announces all its sub-grids (link zt://g1+g2). Independent of time
             extent, variable set, chunk layout, codecs, zarr format and CF time units.
  vfid     - per variable "value family": name, dims, dtype, fill, non-time shape.
  layout   - chunk shape + time phase, e.g. "24x4x6+0". Several layouts may coexist in a swarm.
  fid      - per (variable, layout) "encoding family": vfid + layout + codec pipeline + format.
  ckey     - canonical chunk key "var@layout/i.j.k"; the time index is absolute:
             g = (t - rphase) / q counts time *quanta* q (1 d, 1 h, 1 min or 1 s: the largest dividing the
             replica's step, stored as grid time "dt"), a replica with step = stride*q holds samples
             m = (g - soff) / stride, and c = (m - phase) // ct. Replicas at 1 h and 6 h (same quantum) share
             one grid: time resolution is a property of the layout, not of the data identity.
  cid      - hash of stored bytes; vcid - hash of decoded values (codec-agnostic).
"""
import json
import os
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

from . import codec
from .common import check_signed, cid_of, cjson, h160

UNIT_SECONDS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
                "minute": 60, "minutes": 60, "min": 60, "mins": 60,
                "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
                "day": 86400, "days": 86400, "d": 86400}
STD_CALENDARS = {None, "standard", "gregorian", "proleptic_gregorian"}
_UNITS_RE = re.compile(
    r"\s*(\w+)\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})"
    r"(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}(?:\.\d*)?))?)?\s*(?:Z|UTC|[+-]00:?00)?\s*$")


def parse_time_units(units: str) -> tuple[int, float]:
    """'hours since 1900-01-01 00:00:00.0' -> (3600, epoch seconds of the reference)."""
    m = _UNITS_RE.match(units or "")
    if not m or m.group(1).lower() not in UNIT_SECONDS:
        raise ValueError(f"unsupported time units: {units!r}")
    y, mo, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
    hh, mm = int(m.group(5) or 0), int(m.group(6) or 0)
    ref = datetime(y, mo, d, hh, mm, tzinfo=timezone.utc).timestamp() + float(m.group(7) or 0)
    return UNIT_SECONDS[m.group(1).lower()], ref


def _dims(arr) -> list[str]:
    md = arr.metadata
    if md.zarr_format == 3:
        dims = md.dimension_names
    else:
        dims = arr.attrs.get("_ARRAY_DIMENSIONS")
    if not dims or any(d is None for d in dims):
        return [f"{arr.basename}_dim{i}" for i in range(arr.ndim)]
    return list(dims)


def _chunks(arr) -> tuple[int, ...]:
    grid = getattr(arr.metadata, "chunk_grid", None)
    return tuple(grid.chunk_shape) if grid is not None else tuple(arr.metadata.chunks)


def _docs(root: Path, name: str, fmt: int) -> dict:
    files = ["zarr.json"] if fmt == 3 else [".zarray", ".zattrs"]
    return {f: json.loads((root / name / f).read_text()) for f in files if (root / name / f).exists()}


def _group_docs(root: Path, fmt: int) -> dict:
    files = ["zarr.json"] if fmt == 3 else [".zgroup", ".zattrs"]
    out = {f: json.loads((root / f).read_text()) for f in files if (root / f).exists()}
    if "zarr.json" in out:
        out["zarr.json"].pop("consolidated_metadata", None)
    return out


def _fill(v) -> str:
    return "null" if v is None else repr(np.asarray(v).item())


def _time_axis(g, arrays: dict) -> dict | None:
    """Detect the (single) CF time coordinate and turn it into a canonical definition."""
    cands = [n for n, a in arrays.items()
             if a.ndim == 1 and _dims(a)[0] == n and " since " in str(a.attrs.get("units", ""))]
    if not cands:
        return None
    name = "time" if "time" in cands else cands[0]
    a = arrays[name]
    cal = a.attrs.get("calendar")
    if cal not in STD_CALENDARS:
        # ponytail: only standard calendars; 360_day/noleap need cftime arithmetic
        raise ValueError(f"calendar {cal!r} not supported yet")
    mult, ref = parse_time_units(a.attrs["units"])
    raw = np.asarray(a[:], dtype="float64")
    if raw.size < 2:
        raise ValueError("time axis needs >= 2 steps to infer its step")  # ponytail
    secs = raw * mult + ref
    step = int(round(secs[1] - secs[0]))
    if step <= 0 or not np.allclose(np.diff(secs), step, rtol=0, atol=1e-3):
        raise ValueError("time axis must be regular")  # ponytail: irregular axes -> index-based family
    q = next(x for x in QUANTA if step % x == 0)  # grid quantum: same for 1 h / 3 h / 6 h replicas
    rphase = int(round(secs[0])) % q
    g0 = int(round((secs[0] - rphase) / q))
    attrs = {k: v for k, v in a.attrs.items() if k not in ("units", "calendar") and not k.startswith("_")}
    return {"name": name, "dt": q, "rphase": rphase, "g0": g0, "n": int(raw.size), "attrs": attrs,
            "stride": step // q}


QUANTA = (86400, 3600, 60, 1)  # time quanta of grid identity, coarsest first
# Ablation for the evaluation: identity by stored bytes (what IPFS/BitTorrent/HTTP mirrors offer). Replicas then
# share a swarm only with the same time step AND the same encoding (chunks, codecs, format) of every variable.
BYTE_IDENTITY = __import__("os").environ.get("ZT_IDENTITY") == "bytes"


def tstride(li: dict) -> tuple[int, int]:
    """(stride, sample offset) of a layout in grid quanta; (1, 0) for replicas at the grid quantum."""
    return li.get("stride", 1), li.get("soff", 0)


def chunk_g(li: dict, T: int, c: int) -> tuple[int, int]:
    """Quanta [start, end) spanned by time chunk c of a layout (its samples: start, start+stride, ...)."""
    s, o = tstride(li)
    ct = li["chunks"][T]
    start = (c * ct + li["phase"]) * s + o
    return start, start + ct * s


def chunk_extent(li: dict, T: int, c: int, nv: int | None = None) -> tuple[int, int]:
    """Quanta [first sample, last sample + 1) actually held by chunk c (nv valid samples; None = full)."""
    s, _ = tstride(li)
    start, end = chunk_g(li, T, c)
    return (start, start + (nv - 1) * s + 1) if nv else (start, end - s + 1)


def chunks_for(li: dict, T: int, g_lo: int, g_hi: int) -> tuple[int, int] | None:
    """Inclusive time chunk range of a layout whose samples fall in quanta [g_lo, g_hi); None if none do."""
    s, o = tstride(li)
    m_lo, m_hi = -(-(g_lo - o) // s), (g_hi - 1 - o) // s
    if m_hi < m_lo:
        return None
    ct, ph = li["chunks"][T], li["phase"]
    return (m_lo - ph) // ct, (m_hi - ph) // ct


def natural_stride(li: dict, T: int) -> tuple[int, int]:
    """(stride, offset) of the samples a layout actually holds. Single-sample time chunks are keyed by quantum
    (stride 1) so that copies at different steps share keys; their real spacing is recovered from where the chunks
    are: the gcd of the gaps between held positions (a 2 s fMRI run on a 1 s quantum -> 2)."""
    s, o = tstride(li)
    if li["chunks"][T] != 1 or s != 1 or not li.get("cov"):
        return s, o
    pos = []
    for c0, c1 in li["cov"]:
        if c1 > c0:
            return 1, 0  # consecutive quanta held: the quantum itself is the step
        pos.append(c0 * li["chunks"][T] + li["phase"])
    import math
    g = 0
    for x, y in zip(pos, pos[1:]):
        g = math.gcd(g, y - x)
    return (g, pos[0] % g) if g > 1 else (1, 0)


def available_layouts(array: dict) -> dict:
    """Unserved layouts must not determine the reader's time step or preferred encoding."""
    active = {lay: li for lay, li in array["layouts"].items()
              if (array["taxis"] is None and li.get("n", 1) > 0) or
                 (array["taxis"] is not None and (li.get("cov") or "cov" not in li))}
    return active or array["layouts"]  # retain metadata when every layout is unserved


def sample_of(li: dict, g: int) -> int | None:
    """Sample index of quantum g in a layout, None if that layout has no sample there."""
    s, o = tstride(li)
    return None if (g - o) % s else (g - o) // s


def coord_id(v: np.ndarray) -> str:
    """Identity of a coordinate axis by its values, not its storage: float32 and float64 copies of the same
    latitudes (0.25 deg) or of inexact steps (0.1 deg, equal to ~1e-8) coincide; values rounded to 1e-6."""
    if v.dtype.kind in "fc":
        return h160(b"f" + np.round(v.astype("f8"), 6).tobytes())
    if v.dtype.kind in "iub":
        return h160(b"i" + v.astype("<i8").tobytes())
    return h160(v.dtype.str.encode() + np.ascontiguousarray(v).tobytes())


def layout_id(chunks, phase, stride: int = 1, soff: int = 0) -> str:
    """Chunk layout id; time phase (and stride/offset for replicas coarser than the grid quantum) are part of
    it because canonical time indices depend on them."""
    return "x".join(map(str, chunks)) + ("" if phase is None else f"+{phase}") + \
        (f"s{stride}o{soff}" if stride > 1 else "")


def split_key(ckey: str) -> tuple[str, str, list[int]]:
    """'t2m@24x4x6+0/3.0.1' -> ('t2m', '24x4x6+0', [3, 0, 1])"""
    head, _, coords = ckey.partition("/")
    name, _, lay = head.partition("@")
    return name, lay, [int(x) for x in coords.split(".") if x]


def grid_id_of(grid: dict) -> str:
    """DHT key of a grid family (informational time attrs excluded)."""
    t = grid.get("time")
    return h160(cjson({k: v for k, v in grid.items() if k != "time"} |
                      {"time": None if not t else {k: t[k] for k in ("name", "dt", "rphase")}}))


def packing_contract(spec, gid: str, name: str, units: str = "") -> dict:
    """Bind an explicit source contract to its field, physical grid and units; never infer it from data."""
    bound = {"version": 1, "grid_id": gid, "variable": name, "units": units}
    if isinstance(spec, dict) and "parameters" in spec:
        if any(spec.get(k) != v for k, v in bound.items()) or set(spec) not in (
                set(bound) | {"parameters"}, set(bound) | {"parameters", "pk", "sig"}):
            raise ValueError("source packing contract belongs to another field, grid or units")
        if "sig" in spec and not check_signed(spec):
            raise ValueError("invalid source packing signature")
        params = spec["parameters"]
        out = dict(spec)
    else:
        params = spec
        out = dict(bound, parameters=params)
    if codec.VALUE_ID != "lattice" or BYTE_IDENTITY:
        raise ValueError("source packing requires value identity in lattice mode")
    if isinstance(params, dict):
        if set(params) != {"time"} or not isinstance(params["time"], dict) or not params["time"]:
            raise ValueError("source packing needs a nonempty UTC-seconds time catalogue")
        for stamp, triple in params["time"].items():
            if not isinstance(stamp, str) or not re.fullmatch(r"-?(0|[1-9][0-9]*)", stamp) or str(int(stamp)) != stamp:
                raise ValueError("packing catalogue times must be canonical integer UTC seconds")
            codec.packing_parameters(triple)
    else:
        codec.packing_parameters(params)
    return out


def packing_id(contract: dict) -> str:
    return h160(cjson({k: v for k, v in contract.items() if k not in ("pk", "sig")}))


def value_family(name, dims, dtype, shape_nt, contract=None):
    return h160(cjson({"name": name, "dims": dims, "dtype": "source-codes-v1" if contract else dtype,
                      "shape_nt": shape_nt} | ({"packing": packing_id(contract)} if contract else {})))


def packing_values(values, array: dict, grid: dict, key: str, nv: int):
    """Logical chunk cells and their authenticated source parameters, excluding storage padding."""
    name, lay, co = split_key(key)
    li, T = array["layouts"][lay], array["taxis"]
    if tuple(values.shape) != tuple(li["chunks"]) or len(co) != values.ndim:
        raise ValueError("source packing chunk shape does not match its layout")
    selection = []
    for axis, (dim, size, c) in enumerate(zip(array["dims"], values.shape, co)):
        length = (nv or size) if axis == T else min(size, grid["dims"][dim] - c * size)
        if (axis != T and c < 0) or not 0 < length <= size:
            raise ValueError("source packing chunk is outside its declared grid")
        selection.append(slice(0, length))
    logical = values[tuple(selection)]
    params = array["packing"]["parameters"]
    if isinstance(params, dict):
        if T is None or not grid.get("time"):
            raise ValueError("time-varying source packing needs a time axis")
        start, _ = chunk_g(li, T, co[T])
        stride, _ = tstride(li)
        tm = grid["time"]
        try:
            params = {"slices": [params["time"][str((start + i * stride) * tm["dt"] + tm["rphase"])]
                                 for i in range(logical.shape[T])]}
        except KeyError as e:
            raise ValueError("source packing catalogue does not cover this chunk") from e
    return logical, params


def value_id(values, array: dict, grid: dict, key: str, nv: int) -> str:
    if "packing" not in array:
        return codec.vcid_of(values, array["taxis"])
    logical, params = packing_values(values, array, grid, key, nv)
    return codec.vcid_of(logical, array["taxis"], packing=params)


def value_codes(values, array: dict, grid: dict, key: str, nv: int, *, exact=True):
    if "packing" not in array:
        return codec.lattice_codes(values, array["taxis"], exact=exact)
    logical, params = packing_values(values, array, grid, key, nv)
    # Source-code parity preserves the source counts, masks and contract, rather than decoder-specific bits.
    lc = codec.lattice_codes(logical, array["taxis"], exact=False, packing=params)
    if lc is None:
        return None
    codes = np.full(values.shape, codec._SENT, dtype="<i4")
    codes[tuple(slice(0, s) for s in logical.shape)] = lc[0]
    pairs = lc[1]
    if array["taxis"] is not None:
        pairs += [[1.0, 0.0]] * (values.shape[array["taxis"]] - logical.shape[array["taxis"]])
    return codes, pairs


def scan(path: str | Path, cache_dir: Path | None = None, workers: int | None = None, *, packing: dict | None = None) -> dict:
    root = Path(path).resolve()
    g = zarr.open_group(str(root), mode="r")
    fmt = g.metadata.zarr_format
    arrays = dict(g.arrays())  # ponytail: flat groups only (the xarray layout)
    t = _time_axis(g, arrays)
    tname = t["name"] if t else None

    dims_sizes, coords = {}, {}
    for n, a in arrays.items():
        for d, s in zip(_dims(a), a.shape):
            if d == tname:
                continue
            if dims_sizes.setdefault(d, s) != s:
                raise ValueError(f"inconsistent size for dim {d}")
        if n != tname and a.ndim == 1 and _dims(a)[0] == n:
            coords[n] = coord_id(np.asarray(a[:]))

    tdef = None if not t else {"name": tname, "dt": t["dt"], "rphase": t["rphase"], "attrs": t["attrs"]}
    stride = t["stride"] if t else 1
    soff = t["g0"] % stride if t else 0
    m0 = (t["g0"] - soff) // stride if t else 0  # first sample index in this replica's own lattice
    coord_names = {n for n, a in arrays.items() if n != tname and a.ndim == 1 and _dims(a)[0] == n}
    if packing is not None and (not isinstance(packing, dict) or set(packing) - (set(arrays) - coord_names - {tname})):
        raise ValueError("source packing must map data variable names to contracts")

    def subgrid_of(n):
        """Sub-grid = the variable's own non-time dims (+ their coordinates) and whether it has time.
        Replicas holding different variable sets (e.g. with/without a level axis) still meet per sub-grid."""
        dims = [d for d in _dims(arrays[n]) if d != tname]
        return {"v": 3, "dims": {d: dims_sizes[d] for d in dims}, "coords": {d: coords[d] for d in dims if d in coords},
                "time": tdef if tname in _dims(arrays[n]) else None}

    ainfo, todo = {}, []
    for n, a in arrays.items():
        if n == tname:
            continue  # synthesized by the view
        dims, cs = _dims(a), _chunks(a)
        taxis = dims.index(tname) if tname in dims else None
        shape_nt = [s for i, s in enumerate(a.shape) if i != taxis]
        spec = (packing or {}).get(n, a.attrs.get("zt_packing"))
        contract = None
        if spec is not None:
            if a.dtype.kind != "f":
                raise ValueError("source packing requires floating-point data")
            contract = packing_contract(spec, grid_id_of(subgrid_of(n)), n, a.attrs.get("units", ""))
        # fill_value is not part of the value family: it only says what an ABSENT chunk would read as (zarr v2 and v3
        # writers default it differently for the same data, e.g. None vs 0 for int16), and holders only ever serve
        # chunks that exist; each layout keeps its own fill in its docs
        vfid = value_family(n, dims, np.dtype(a.dtype).newbyteorder("<").str, shape_nt, contract)
        ct = cs[taxis] if taxis is not None else None
        # a chunk holding ONE time sample is the same object whatever the replica's step: index it by its quantum
        # (stride 1), so hourly and 6-hourly single-step chunks share keys and holders directly
        a_s, a_o, a_m0 = (1, 0, t["g0"]) if ct == 1 else (stride, soff, m0)
        phase = a_m0 % ct if taxis is not None else None
        c0 = (a_m0 - phase) // ct if taxis is not None else 0
        lay = layout_id(cs, phase, a_s if taxis is not None else 1, a_o if taxis is not None else 0)
        docs = _docs(root, n, fmt)
        hdocs = json.loads(json.dumps(docs))
        for m in hdocs.values():
            m.pop("shape", None)
            m.pop("attributes", None)
        hdocs.pop(".zattrs", None)
        fid = h160(cjson({"vfid": vfid, "lay": lay, "fmt": fmt, "docs": hdocs}))
        ainfo[n] = {"vfid": vfid, "dims": dims, "taxis": taxis, "fmt": fmt,
                    "attrs": {k: v for k, v in (a.attrs.asdict() if hasattr(a.attrs, "asdict") else dict(a.attrs)).items()
                              if not k.startswith("_") and k != "zt_packing"},
                    "layouts": {lay: {"fid": fid, "docs": docs, "chunks": list(cs), "phase": phase}
                                | ({"stride": a_s, "soff": a_o} if taxis is not None and a_s > 1 else {})}}
        if contract:
            ainfo[n]["packing"] = contract
        grid_shape = [math.ceil(s / c) for s, c in zip(a.shape, cs)]
        for idx in np.ndindex(*grid_shape):
            rel = f"{n}/{a.metadata.encode_chunk_key(idx)}"
            if not (root / rel).is_file():
                continue
            canon = list(idx)
            nvalid = 0
            if taxis is not None:
                canon[taxis] = c0 + idx[taxis] * (stride if ct == 1 else 1)  # single-step chunks: by quantum
                nvalid = min(ct, t["n"] - idx[taxis] * ct)
            todo.append((f"{n}@{lay}/" + ".".join(map(str, canon)), rel, n, nvalid))

    # hash cache: rel -> [size, mtime_ns, cid, vcid, metadata key]
    hc_path = (cache_dir / f"{h160(str(root).encode())}.{codec.VALUE_ID}.{codec.ESTIMATOR}.json") if cache_dir else None  # ids depend on it
    hc = json.loads(hc_path.read_text()) if hc_path and hc_path.exists() else {}

    # optional cache shared by every node on a host, keyed by file identity: hard-linked replicas (experiments, a site
    # serving one archive under several paths) hash each chunk file once instead of once per node and run
    shared = Path(os.environ["ZT_HASH_CACHE"]) if os.environ.get("ZT_HASH_CACHE") else None
    dkey = {n: h160(cjson([a["layouts"], a["taxis"], a.get("packing"), t, subgrid_of(n)])) for n, a in ainfo.items()}

    def one(item):
        ckey, rel, n, nvalid = item
        st = (root / rel).stat()
        hit = hc.get(rel)
        if hit and len(hit) == 5 and hit[0] == st.st_size and hit[1] == st.st_mtime_ns and hit[4] == dkey[n]:
            return ckey, rel, hit[2], hit[3], st.st_size, st.st_mtime_ns, nvalid
        sp = None
        if shared:
            sp = shared / h160(f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}:{dkey[n]}:{codec.VALUE_ID}:{codec.ESTIMATOR}".encode())
            try:
                cid, vc = json.loads(sp.read_text())
                return ckey, rel, cid, vc, st.st_size, st.st_mtime_ns, nvalid
            except (OSError, ValueError):
                pass
        raw = (root / rel).read_bytes()
        vals = codec.decode(next(iter(ainfo[n]["layouts"].values()))["docs"], raw)
        vc = value_id(vals, ainfo[n], subgrid_of(n), ckey, nvalid)
        cid = cid_of(raw)
        if sp is not None:
            sp.parent.mkdir(parents=True, exist_ok=True)
            tmp = sp.with_name(sp.name + f".{os.getpid()}")
            tmp.write_text(json.dumps([cid, vc]))
            tmp.replace(sp)
        return ckey, rel, cid, vc, st.st_size, st.st_mtime_ns, nvalid

    chunks, files, newhc = {}, {}, {}
    # one decoding thread per core (was 16): peak memory scales with the threads, a 1-vCPU station gains nothing
    with ThreadPoolExecutor(workers or int(os.environ.get("ZT_SCAN_WORKERS", 0)) or min(16, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4)) as ex:
        for ckey, rel, cid, vc, size, mt, nvalid in ex.map(one, todo):
            chunks[ckey] = [cid, vc, size, nvalid]
            files[ckey] = str(root / rel)
            newhc[rel] = [size, mt, cid, vc, dkey[split_key(ckey)[0]]]
    if hc_path:
        hc_path.parent.mkdir(parents=True, exist_ok=True)
        hc_path.write_text(json.dumps(newhc))
    gdocs = _group_docs(root, fmt)
    members: dict[str, dict] = {}
    for n in ainfo:
        if n in coord_names:
            continue
        grid = subgrid_of(n)
        if BYTE_IDENTITY:  # the encoding becomes part of the identity (one swarm per byte representation)
            grid = dict(grid, enc=sorted(li["fid"] for li in ainfo[n]["layouts"].values()))
        sg = members.setdefault(grid_id_of(grid), {"grid": grid, "arrays": set()})
        sg["arrays"].add(n)
        sg["arrays"].update(d for d in grid["dims"] if d in coord_names)  # coordinates travel with their users
    if not members and coord_names:  # coordinates only
        for n in coord_names:
            grid = subgrid_of(n)
            members.setdefault(grid_id_of(grid), {"grid": grid, "arrays": set()})["arrays"].add(n)
    subgrids = {}
    for gid, m in members.items():
        subgrids[gid] = {"grid_id": gid, "grid": m["grid"], "gdocs": gdocs,
                         "arrays": {n: ainfo[n] for n in sorted(m["arrays"])},
                         "chunks": {k: v for k, v in chunks.items() if k.partition("@")[0] in m["arrays"]},
                         "files": {k: v for k, v in files.items() if k.partition("@")[0] in m["arrays"]}}
    return {"path": str(root), "subgrids": subgrids}
