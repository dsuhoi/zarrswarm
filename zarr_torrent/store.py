"""xarray / zarr integration.

    import zarr_torrent as zt
    ds = zt.open_dataset("zt://<grid_id>")                          # union view, native chunks
    ts = zt.open_dataset("zt://<grid_id>", chunking={"time": 8760, "lat": 1, "lon": 1})  # re-chunked view
    sub = ds.t2m.sel(time=slice("2020-01-05", "2020-01-20"))
    zt.prefetch(sub)                                                  # one plan for the whole slice
    sub.mean().compute()

Chunk access modes (chosen per view chunk from swarm metadata only, no data is read to decide):
  direct    view layout == a replica layout (same chunks, time-aligned): stored bytes pass through.
  assembled any other view chunking: the chunk is cut from the cheapest set of source chunks over
            all layouts present in the swarm (cost = estimated bytes), gaps filled layout by layout,
            and handed to zarr uncompressed (it never leaves the process).
  progressive  see progressive_mean(): low-discrepancy download order + anytime estimate.

ZtStore is a read-only zarr v3 Store. Chunk reads are coalesced (5 ms window) into batched
/api/fetch calls to the local node, which plans them across peers (max-flow) and returns paths
into its content-addressed cache.
"""
import asyncio
import base64
import copy
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import aiohttp
import numpy as np
import xarray as xr
import zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import cpu, default_buffer_prototype
from zarr.storage import MemoryStore

from . import codec
from .jlps import complete_for
from .scan import chunk_g, chunks_for, natural_stride, sample_of, split_key, tstride

CTL = os.environ.get("ZT_CTL", "http://127.0.0.1:7882")
BATCH_WINDOW = 0.005
DECODED_CACHE_BYTES = int(os.environ.get("ZT_DECODED_CACHE_MB", "512")) << 20
PUSHDOWN_FRAC = float(os.environ.get("ZT_PUSHDOWN_FRAC", "0.25"))
READAHEAD = int(os.environ.get("ZT_READAHEAD", "4"))  # chunks prefetched ahead of a sequential time scan  # below: ask holders for the hyperslab only
_STORES: "weakref.WeakValueDictionary[str, ZtStore]" = weakref.WeakValueDictionary()
# Assembly calls synchronous zarr (codec decode/encode) which needs zarr's own event loop *and* its default
# executor; running it in that same executor starves it (deadlock). Use a dedicated pool.
_POOL = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) + 4), thread_name_prefix="zt-assemble")


def http(ctl: str, method: str, path: str, body=None, timeout: float = 3600, raw: bool = False):
    req = urllib.request.Request(ctl.rstrip("/") + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Zt-Client": "1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read() if raw else json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: HTTP {e.code}: {e.read().decode(errors='replace')[-2000:]}") from None


def read_chunk(ctl: str, grid: str, key: str, path: str) -> bytes:
    """Stored bytes of a chunk the local node holds: straight from its file when this process can see it,
    otherwise through the node's API (node in a container / another mount namespace)."""
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return http(ctl, "POST", "/api/read", {"grid": grid, "key": key}, raw=True)


def parse_key(docs: dict, rest: str, ndim: int) -> list[int] | None:
    """Chunk coords from a key relative to the array, honoring its chunk key encoding."""
    if "zarr.json" in docs:
        enc = docs["zarr.json"].get("chunk_key_encoding") or {"name": "default"}
        conf = enc.get("configuration") or {}
        if enc.get("name") == "default":
            sep = conf.get("separator", "/")
            if ndim == 0:
                return [] if rest == "c" else None
            if not rest.startswith("c" + sep):
                return None
            rest = rest[2:]
        else:
            sep = conf.get("separator", ".")
    else:
        sep = docs[".zarray"].get("dimension_separator") or "."
    if ndim == 0:
        return [] if rest == "0" else None
    parts = rest.split(sep)
    if len(parts) != ndim or not all(p.isdigit() for p in parts):
        return None
    return [int(p) for p in parts]


def _meta(docs: dict) -> dict:
    return docs["zarr.json"] if "zarr.json" in docs else docs[".zarray"]


def _set_chunks(docs: dict, chunks: list[int]):
    m = _meta(docs)
    if "zarr.json" in docs:
        m["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": list(chunks)}}
    else:
        m["chunks"] = list(chunks)


@lru_cache(maxsize=4096)
def _dtype_fill(docs_json: str):
    arr, _, _ = codec._one_chunk_array(json.loads(docs_json), None, read_only=True)
    return np.dtype(arr.dtype), arr.fill_value


def _fill_from_attr(v):
    """xarray stores float _FillValue attributes of zarr v3 arrays as base64 little-endian bytes."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            raw = base64.b64decode(v)
            return np.frombuffer(raw, dtype=f"<f{len(raw)}")[0].item()
    return v


def _fill_to_attr(v):
    if isinstance(v, float):
        return base64.b64encode(np.float64(v).tobytes()).decode()
    return v


def _json_num(v):
    if isinstance(v, float) and not np.isfinite(v):
        return "NaN" if np.isnan(v) else ("Infinity" if v > 0 else "-Infinity")
    return v


def _intersect(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def _subtract(ivs: list[tuple[int, int]], cut: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for lo, hi in ivs:
        if cut[1] <= lo or cut[0] >= hi:
            out.append((lo, hi))
            continue
        if lo < cut[0]:
            out.append((lo, cut[0]))
        if cut[1] < hi:
            out.append((cut[1], hi))
    return out


class View:
    """Client-side swarm view: time alignment, per-variable view chunking and source planning."""

    def __init__(self, v: dict, chunking: dict | None = None, step: int | None = None):
        """step: time step of the view in seconds (a multiple of the grid quantum). Default: the finest step any
        replica offers; coarser replicas then fill only their own samples (the rest is missing)."""
        self.v, self.fmt, self.t = v, v["fmt"], v["grid"]["time"]
        self.arrays = {n: a for n, a in v["arrays"].items() if a["layouts"]}
        self.G0, self.n_time, self.S, self.O = 0, None, 1, 0
        if self.t:
            tl = [li for a in self.arrays.values() if a["taxis"] is not None for li in a["layouts"].values()]
            if step:
                self.S = int(step) // self.t["dt"]
                if self.S < 1 or int(step) % self.t["dt"]:
                    raise ValueError(f"step {step}s is not a multiple of the grid quantum {self.t['dt']}s")
                offs = [tstride(li)[1] % self.S for li in tl if self.S % tstride(li)[0] == 0]
                self.O = offs[0] if offs else 0
            elif tl:
                nat = [natural_stride(li, a["taxis"]) for a in self.arrays.values() if a["taxis"] is not None
                       for li in a["layouts"].values()]
                self.S, self.O = min(nat)
                self.O %= self.S
        lat = (self.S, self.O)
        # native layout = the most common one among those serving the view's lattice directly
        # native layout = the most common one among those whose REAL sample spacing is the view's step (a single-sample
        # layout is keyed at stride 1 but holds 2 s volumes: judged by natural_stride, not by its key stride)
        self.native = {n: max(a["layouts"], key=lambda l: (a["taxis"] is None or complete_for(a["layouts"][l], lat)
                                                             and natural_stride(a["layouts"][l], a["taxis"])[0] == self.S,
                                                             a["layouts"][l]["n"], l))
                       for n, a in self.arrays.items()}
        if self.t:
            al = [(a["layouts"][self.native[n]]["chunks"][a["taxis"]], a["layouts"][self.native[n]]["phase"])
                  for n, a in self.arrays.items() if a["taxis"] is not None]
            ct, ph = max(set(al), key=al.count) if al else (1, 0)
            if v["gmin"] is None:
                self.n_time = 0
            else:
                j0 = -(-(v["gmin"] - self.O) // self.S)  # first lattice sample >= gmin, aligned to native chunks
                j0 -= (j0 - ph) % ct
                self.G0 = j0 * self.S + self.O
                self.n_time = max(0, -(-(v["gmax"] - self.G0) // self.S))
        self.spec = {n: self._spec(n, a, chunking or {}) for n, a in self.arrays.items()}

    def _g(self, j: int) -> int:
        """View time sample j -> grid quantum."""
        return self.G0 + j * self.S

    def _spec(self, n, a, chunking):
        nat = a["layouts"][self.native[n]]
        shape = list(_meta(nat["docs"])["shape"])
        if a["taxis"] is not None:
            shape[a["taxis"]] = self.n_time
        vch = list(nat["chunks"])
        for d, size in chunking.items():
            if d in a["dims"]:
                i = a["dims"].index(d)
                vch[i] = max(1, min(size if size and size > 0 else shape[i], max(shape[i], 1)))
        direct = None
        for lay in sorted(a["layouts"], key=lambda l: l != self.native[n]):
            li = a["layouts"][lay]
            same_fmt = ("zarr.json" in li["docs"]) == (self.fmt == 3)
            if same_fmt and li["chunks"] == vch and (a["taxis"] is None or (
                    tstride(li)[0] == self.S and complete_for(li, (self.S, self.O))
                    and (sample_of(li, self.G0) - li["phase"]) % vch[a["taxis"]] == 0)):
                direct = lay
                break
        if direct:
            docs = copy.deepcopy(a["layouts"][direct]["docs"])
            li = a["layouts"][direct]
            toff = 0 if a["taxis"] is None else (sample_of(li, self.G0) - li["phase"]) // vch[a["taxis"]]
        else:
            docs, toff = self._plain_docs(nat["docs"], vch, a["dims"]), 0
        _meta(docs)["shape"] = shape
        return {"shape": shape, "vch": vch, "direct": direct, "toff": toff, "docs": docs}

    def _plain_docs(self, base: dict, vch: list[int], dims: list[str]) -> dict:
        """Uncompressed little-endian metadata (in the view's zarr format) for locally assembled chunks.
        Built from scratch so replicas stored as zarr v2 and v3 can be mixed in one view."""
        dt, fill = _dtype_fill(json.dumps(base))
        attrs = dict(base["zarr.json"].get("attributes", {})) if "zarr.json" in base else dict(base.get(".zattrs", {}))
        attrs.pop("_ARRAY_DIMENSIONS", None)
        shape = list(_meta(base)["shape"])
        fv = None if fill is None else np.asarray(fill).item()
        # CF mask value: xarray reads it from the zarr fill_value for v2 but from the _FillValue attribute for v3
        if self.fmt == 2 and "_FillValue" in attrs:
            fv = _fill_from_attr(attrs.pop("_FillValue"))
        elif self.fmt == 2 and "zarr.json" in base:
            # zarr v3 makes fill_value mandatory (0 for int16 by default) without meaning "missing"; a v2 view would
            # turn it into a mask and hide every zero voxel. Only a declared _FillValue carries that meaning.
            fv = None
        elif self.fmt == 3 and "zarr.json" not in base and fv is not None and "_FillValue" not in attrs:
            attrs["_FillValue"] = _fill_to_attr(fv)
        fv = _json_num(fv)
        if self.fmt == 3:
            codecs = [{"name": "bytes", "configuration": {"endian": "little"}}] if dt.itemsize > 1 else [{"name": "bytes"}]
            return {"zarr.json": {"zarr_format": 3, "node_type": "array", "shape": shape, "data_type": dt.name,
                                  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(vch)}},
                                  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                                  "fill_value": fv if fv is not None else 0, "codecs": codecs,
                                  "attributes": attrs, "dimension_names": list(dims)}}
        return {".zarray": {"zarr_format": 2, "shape": shape, "chunks": list(vch), "dtype": dt.newbyteorder("<").str,
                            "compressor": None, "fill_value": fv, "order": "C", "filters": None},
                ".zattrs": dict(attrs, _ARRAY_DIMENSIONS=list(dims))}

    # -------------------------------------------------------------- planning
    def _cost(self, a, li, region) -> float:
        """Estimated seconds: bytes of the source chunks touched / aggregate rate of the layout's holders."""
        per = li["bytes"] / max(li["n"], 1) / max(li.get("rate") or 1.0, 1.0)
        n = 1
        for i, (lo, hi) in enumerate(region):
            cs = li["chunks"][i]
            if i == a["taxis"]:
                cr = chunks_for(li, i, self._g(lo), self._g(hi - 1) + 1)
                n *= 0 if cr is None else cr[1] - cr[0] + 1
            else:
                n *= (hi - 1) // cs - lo // cs + 1
        return n * per

    def sources(self, n: str, region: list[tuple[int, int]]) -> list[tuple[str, tuple, tuple]]:
        """Source chunks covering `region` (view coords) -> [(ckey, src_slices, dst_slices)].
        Layouts are tried cheapest first; each fills the time gaps the previous ones left."""
        a = self.arrays[n]
        T = a["taxis"]
        if any(hi <= lo for lo, hi in region):
            return []
        order = sorted(a["layouts"], key=lambda l: self._cost(a, a["layouts"][l], region))
        out = []
        if T is None:
            return self._emit(n, a, order[0], region, region)
        lat = (self.S, self.O)
        full = [l for l in order if complete_for(a["layouts"][l], lat)]
        coarse = [l for l in order if l not in full]  # hold only some of the view's samples: fill those last
        remaining = [tuple(region[T])]  # view sample intervals still to fill
        for lay in full + coarse:
            li = a["layouts"][lay]
            for c0, c1 in li["cov"]:
                g_a, g_b = chunk_g(li, T, c0)[0], chunk_g(li, T, c1)[1]
                cov = (-(-(g_a - self.G0) // self.S), -(-(g_b - self.G0) // self.S))  # view samples in [g_a, g_b)
                for iv in list(remaining):
                    x = _intersect(iv, cov)
                    if x:
                        sub = list(region)
                        sub[T] = x
                        out += self._emit(n, a, lay, region, sub)
                        if lay in full:  # a coarse layout leaves the samples it lacks missing
                            remaining = _subtract(remaining, x)
            if not remaining:
                break
        return out

    def _emit(self, n, a, lay, region, sub):
        """[(ckey, src slices, dst slices)] reading view region `sub` from layout `lay`. Along time a source
        sample m <-> view sample j when G0 + j*S = m*stride + soff; the slices get a step when the strides
        differ (view 6 h from a 1 h replica: every 6th source sample; view 1 h from a 6 h replica: every 6th
        view sample)."""
        li = a["layouts"][lay]
        T, per_dim = a["taxis"], []
        for i, (lo, hi) in enumerate(sub):
            cs = li["chunks"][i]
            if i != T:
                per_dim.append([(c, slice(max(lo, c * cs) - c * cs, min(hi, (c + 1) * cs) - c * cs),
                                 slice(max(lo, c * cs) - region[i][0], min(hi, (c + 1) * cs) - region[i][0]))
                                for c in range(lo // cs, (hi - 1) // cs + 1)])
                continue
            s_, o_ = tstride(li)
            if self.S % s_ == 0:
                dj = 1
            elif s_ % self.S == 0:
                dj = s_ // self.S
            else:
                per_dim.append([])  # incommensurate steps: nothing to take from this layout
                continue
            # first view sample that is also a source sample (then every dj-th one is)
            j = next((j for j in range(lo, min(hi, lo + dj)) if (self._g(j) - o_) % s_ == 0), None)
            if j is None:
                per_dim.append([])
                continue
            dm = dj * self.S // s_  # source samples advanced per taken view sample
            m0 = (self._g(j) - o_) // s_
            cnt = (hi - 1 - j) // dj + 1  # view samples j, j+dj, ... < hi
            ct, ph = cs, li["phase"]
            items = []
            t = 0
            while t < cnt:
                m = m0 + t * dm
                c = (m - ph) // ct
                base = c * ct + ph
                t_end = min(cnt, t + (base + ct - m + dm - 1) // dm)  # samples of this chunk
                src = slice(m - base, m - base + (t_end - t - 1) * dm + 1, dm)
                dst = slice(j + t * dj - region[i][0], j + (t_end - 1) * dj - region[i][0] + 1, dj)
                items.append((c, src, dst))
                t = t_end
            per_dim.append(items)
        out = []
        for idx in np.ndindex(*[len(r) for r in per_dim]) if per_dim else [()]:
            parts = [per_dim[i][k] for i, k in enumerate(idx)]
            out.append((f"{n}@{lay}/" + ".".join(str(c) for c, _, _ in parts),
                        tuple(s for _, s, _ in parts), tuple(d for _, _, d in parts)))
        return out

    def time_seconds(self) -> np.ndarray | None:
        if not self.t or self.n_time is None:
            return None
        return (self.G0 + np.arange(self.n_time, dtype="int64") * self.S) * self.t["dt"] + self.t["rphase"]

    def time_index(self, t0=None, t1=None) -> tuple[int, int]:
        secs = self.time_seconds()
        if secs is None:
            return 0, 0
        j0 = 0 if t0 is None else int(np.searchsorted(secs, np.datetime64(t0, "s").astype("int64"), "left"))
        if t1 is None:
            j1 = len(secs)
        elif isinstance(t1, str) and len(t1.strip()) == 10:  # date only -> whole day, like pandas/xarray .sel
            j1 = int(np.searchsorted(secs, (np.datetime64(t1.strip(), "D") + 1).astype("datetime64[s]").astype("int64"),
                                     "left"))
        else:
            j1 = int(np.searchsorted(secs, np.datetime64(t1, "s").astype("int64"), "right"))
        return j0, j1

    def region(self, n: str, t0=None, t1=None) -> list[tuple[int, int]]:
        a, shape = self.arrays[n], self.spec[n]["shape"]
        reg = [(0, s) for s in shape]
        if a["taxis"] is not None:
            reg[a["taxis"]] = self.time_index(t0, t1)
        return reg

    # -------------------------------------------------------------- zarr documents
    def docs(self) -> dict[str, bytes]:
        fmt, t = self.fmt, self.t
        out = {f: json.dumps(m).encode() for f, m in self.v["gdocs"].items()}
        if fmt == 3:
            out.setdefault("zarr.json", json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}).encode())
        else:
            out.setdefault(".zgroup", b'{"zarr_format": 2}')
        for n, sp in self.spec.items():
            for f, m in sp["docs"].items():
                out[f"{n}/{f}"] = json.dumps(m).encode()
        secs = self.time_seconds()
        if secs is not None:
            n, name = len(secs), t["name"]
            attrs = dict(t.get("attrs") or {}, units="seconds since 1970-01-01", calendar="proleptic_gregorian")
            if fmt == 3:
                out[f"{name}/zarr.json"] = json.dumps({
                    "zarr_format": 3, "node_type": "array", "shape": [n], "data_type": "int64",
                    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [max(n, 1)]}},
                    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                    "fill_value": 0, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                    "attributes": attrs, "dimension_names": [name]}).encode()
                ckey = f"{name}/c/0"
            else:
                out[f"{name}/.zarray"] = json.dumps({
                    "zarr_format": 2, "shape": [n], "chunks": [max(n, 1)], "dtype": "<i8", "compressor": None,
                    "fill_value": None, "order": "C", "filters": None}).encode()
                out[f"{name}/.zattrs"] = json.dumps(dict(attrs, _ARRAY_DIMENSIONS=[name])).encode()
                ckey = f"{name}/0"
            if n:
                out[ckey] = secs.astype("<i8").tobytes()
        return out


def _slice(data: bytes, br) -> bytes:
    if br is None:
        return data
    if isinstance(br, RangeByteRequest):
        return data[br.start:br.end]
    if isinstance(br, OffsetByteRequest):
        return data[br.offset:]
    if isinstance(br, SuffixByteRequest):
        return data[-br.suffix:]
    raise TypeError(br)


class ZtStore(MemoryStore):
    __hash__ = object.__hash__  # MemoryStore defines __eq__ -> unhashable; we need weak references

    def __init__(self, ctl: str, grid: str, view: View):
        super().__init__({k: cpu.Buffer.from_bytes(v) for k, v in view.docs().items()}, read_only=True)
        self.ctl, self.grid, self.view = ctl, grid, view
        self.recording: set | None = None
        self.stats = {"direct": 0, "assembled": 0, "source_chunks": 0, "pushdown": 0, "readahead": 0}
        self._last_t: dict = {}
        self._ahead: set = set()
        self.pushdown = True
        self._spending: list = []
        self._sflusher: asyncio.Task | None = None
        self._pending: dict[str, list[asyncio.Future]] = {}
        self._flusher: asyncio.Task | None = None
        self._dcache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._dbytes, self._dlock = 0, threading.Lock()
        self.token = os.urandom(8).hex()
        _STORES[self.token] = self

    def _locate(self, key: str):
        name, _, rest = key.partition("/")
        sp = self.view.spec.get(name)
        if sp is None or not rest:
            return None
        coords = parse_key(sp["docs"], rest, len(sp["shape"]))
        if coords is None:
            return None
        return name, sp, coords

    async def get(self, key, prototype=None, byte_range=None):
        prototype = prototype or default_buffer_prototype()
        if key in self._store_dict:
            return await super().get(key, prototype, byte_range)
        loc = self._locate(key)
        if loc is None:
            return None
        name, sp, coords = loc
        a = self.view.arrays[name]
        if self.recording is None:
            self._readahead(name, sp, a["taxis"], coords)
        region = [(c * cs, min((c + 1) * cs, s)) for c, cs, s in zip(coords, sp["vch"], sp["shape"])]
        if sp["direct"]:
            src = list(coords)
            if a["taxis"] is not None:
                src[a["taxis"]] += sp["toff"]
            li = a["layouts"][sp["direct"]]
            if a["taxis"] is None or any(c0 <= src[a["taxis"]] <= c1 for c0, c1 in li["cov"]):
                ck = f"{name}@{sp['direct']}/" + ".".join(map(str, src))
                if self.recording is not None:
                    self.recording.add(ck)
                    return None
                self.stats["direct"] += 1
                data = await self._read(ck)
                return None if data is None else prototype.buffer.from_bytes(_slice(data, byte_range))
            # this time range exists only in other layouts: assemble, then encode like the direct layout
        plan = self.view.sources(name, region)
        if self.recording is not None:
            self.recording.update(k for k, _, _ in plan)
            return None
        if not plan:
            return None
        a = self.view.arrays[name]

        def small(k, src):  # optimistic pushdown pays off when we need a small part of the source chunk
            _, lay, _ = split_key(k)
            cs = a["layouts"][lay]["chunks"]
            frac = np.prod([(s_.stop - s_.start) / c for s_, c in zip(src, cs)]) if cs else 1.0
            return self.pushdown and frac < PUSHDOWN_FRAC

        async def one(k, src):
            if small(k, src):
                self.stats["pushdown"] += 1
                return "arr", await self._slice(k, [[s_.start, s_.stop] for s_ in src])  # bounding box; step below
            return "raw", await self._read(k)
        parts = await asyncio.gather(*(one(k, src) for k, src, _ in plan))
        self.stats["assembled"] += 1
        self.stats["source_chunks"] += len(plan)
        data = await asyncio.get_running_loop().run_in_executor(_POOL, self._assemble, name, sp, plan, parts)
        return prototype.buffer.from_bytes(_slice(data, byte_range))

    def _assemble(self, name, sp, plan, raws) -> bytes:  # raws: [(kind, payload)]
        a = self.view.arrays[name]
        dt, fill = _dtype_fill(json.dumps(sp["docs"]))
        out = np.full(sp["vch"], 0 if fill is None else fill, dtype=dt)
        for (k, src, dst), (kind, raw) in zip(plan, raws):
            _, lay, _ = split_key(k)
            docs = a["layouts"][lay]["docs"]
            if kind == "arr" and raw is not None:  # pushed-down hyperslab (raw little-endian values)
                sdt = _dtype_fill(json.dumps(docs))[0]
                shape = tuple(s_.stop - s_.start for s_ in src)
                box = np.frombuffer(raw, dtype=sdt.newbyteorder("<")).reshape(shape)
                out[dst] = box[tuple(slice(None, None, s_.step) for s_ in src)]
                continue
            if raw is None:  # chunk absent in a covering replica = that replica's fill_value (zarr semantics)
                sfill = _dtype_fill(json.dumps(docs))[1]
                out[dst] = 0 if sfill is None else sfill
                continue
            out[dst] = self._decoded(k, docs, raw)[src]
        if sp["direct"]:
            return codec.encode(sp["docs"], out)
        return out.astype(dt.newbyteorder("<"), copy=False).tobytes()

    def _decoded(self, k: str, docs: dict, raw: bytes) -> np.ndarray:
        """Decoded source chunks are shared by all view chunks cut from them (re-chunked views)."""
        with self._dlock:
            hit = self._dcache.get(k)
            if hit is not None:
                self._dcache.move_to_end(k)
                return hit
        vals = codec.decode(docs, raw)
        with self._dlock:
            self._dcache[k] = vals
            self._dbytes += vals.nbytes
            while self._dbytes > DECODED_CACHE_BYTES and len(self._dcache) > 1:
                _, old = self._dcache.popitem(last=False)
                self._dbytes -= old.nbytes
        return vals

    def _readahead(self, name, sp, T, co):
        """Sequential scans along time (ML training loops): after view chunks c-1 then c, prefetch the source
        chunks of c+1..c+k in the background (any mode: direct or assembled) so next reads hit the cache."""
        if T is None or READAHEAD <= 0:
            return
        key = (name, tuple(c for i, c in enumerate(co) if i != T))
        prev = self._last_t.get(key)
        self._last_t[key] = co[T]
        if prev is None or co[T] != prev + 1:
            return
        keys = []
        nt = -(-sp["shape"][T] // sp["vch"][T])
        for c in range(co[T] + 1, min(co[T] + 1 + READAHEAD, nt)):
            if (key, c) in self._ahead:
                continue
            self._ahead.add((key, c))
            nxt = list(co)
            nxt[T] = c
            region = [(x * cs, min((x + 1) * cs, s_)) for x, cs, s_ in zip(nxt, sp["vch"], sp["shape"])]
            keys += [k for k, _, _ in self.view.sources(name, region)]
        if keys:
            self.stats["readahead"] += len(keys)
            asyncio.get_running_loop().create_task(self._post_download(list(dict.fromkeys(keys))))

    async def _post_download(self, keys):
        try:
            async with aiohttp.ClientSession(headers={"X-Zt-Client": "1"}) as session:
                async with session.post(f"{self.ctl}/api/download", json={"grid": self.grid, "keys": keys,
                                                                           "label": "readahead"}) as r:
                    await r.read()
        except Exception:
            pass

    async def _slice(self, ck: str, sel: list) -> bytes | None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._spending.append((ck, sel, fut))
        if self._sflusher is None or self._sflusher.done():
            self._sflusher = loop.create_task(self._sflush())
        return await fut

    async def _sflush(self):
        await asyncio.sleep(BATCH_WINDOW)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                         headers={"X-Zt-Client": "1"}) as session:
            while self._spending:
                batch, self._spending = self._spending, []
                try:
                    items = [{"key": k, "sel": sel} for k, sel, _ in batch]
                    async with session.post(f"{self.ctl}/api/slices", json={"grid": self.grid, "items": items}) as r:
                        r.raise_for_status()
                        body = await r.read()
                    off = 0
                    for _, _, f in batch:
                        n = int.from_bytes(body[off:off + 4], "big")
                        off += 4
                        if n == 0xFFFFFFFF:
                            f.set_exception(OSError("zt: pushdown slice unavailable")) if not f.done() else None
                            continue
                        if not f.done():
                            f.set_result(body[off:off + n])
                        off += n
                except Exception as e:
                    for _, _, f in batch:
                        if not f.done():
                            f.set_exception(e)

    async def _read(self, ck: str) -> bytes | None:
        path = await self._path(ck)
        if path is None:
            return None  # no replica holds this chunk -> fill_value
        if path == "!":
            raise OSError(f"zt: replicas exist for {ck} but download failed")
        return await asyncio.to_thread(read_chunk, self.ctl, self.grid, ck, path)

    async def exists(self, key):
        return key in self._store_dict or self._locate(key) is not None

    async def _path(self, ck: str):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending.setdefault(ck, []).append(fut)
        if self._flusher is None or self._flusher.done():
            self._flusher = loop.create_task(self._flush())
        return await fut

    async def _flush(self):
        await asyncio.sleep(BATCH_WINDOW)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                         headers={"X-Zt-Client": "1"}) as session:  # no leaked sessions
            await self._drain(session)

    async def _drain(self, session):
        while self._pending:
            batch, self._pending = self._pending, {}
            try:
                # the encoding we will decode with: the node must not hand back a local file of another family
                expect = {}
                for k in batch:
                    n, _, rest = k.partition("@")
                    a = self.view.arrays.get(n)
                    li = a["layouts"].get(rest.partition("/")[0]) if a else None
                    if li:
                        expect[k] = [a.get("vfid"), li.get("fid")]
                async with session.post(f"{self.ctl}/api/fetch", json={"grid": self.grid, "keys": list(batch),
                                                                      "expect": expect}) as r:
                    r.raise_for_status()
                    res = await r.json()
                for k, fs in batch.items():
                    for f in fs:
                        if not f.done():
                            f.set_result(res.get(k))
            except Exception as e:
                for fs in batch.values():
                    for f in fs:
                        if not f.done():
                            f.set_exception(e)


def resolve(link: str, ctl: str | None = None) -> list[str]:
    """zt://g1+g2 or zt://name@pk -> list of sub-grid ids."""
    return http(ctl or CTL, "GET", "/api/resolve?link=" + urllib.parse.quote(link))["grid"].split("+")


def open_view(link: str, ctl: str | None = None, chunking: dict | None = None) -> tuple[str, View]:
    """View of ONE sub-grid (the first one of a composite link)."""
    ctl = ctl or CTL
    grid = resolve(link, ctl)[0]
    return grid, View(http(ctl, "GET", f"/api/view/{grid}?refresh=1"), chunking)


def step_seconds(step) -> int | None:
    """'6h' / '30min' / '1d' / 3600 / numpy/pandas timedelta -> seconds (None stays None)."""
    if step is None:
        return None
    if isinstance(step, (int, float)):
        return int(step)
    import pandas as pd
    return int(pd.Timedelta(step).total_seconds())


def open_views(link: str, ctl: str | None = None, chunking: dict | None = None, step=None) -> list[tuple[str, View]]:
    ctl = ctl or CTL
    return [(g, View(http(ctl, "GET", f"/api/view/{g}?refresh=1"), chunking, step_seconds(step)))
            for g in resolve(link, ctl)]


def open_dataset(link: str, ctl: str | None = None, chunking: dict | None = None, pushdown: bool = True,
                 step=None, **kw) -> xr.Dataset:
    """Open the swarm union of `link` as an xarray Dataset.

    chunking: {dim: size} view chunking (size <= 0 / None = whole axis). Default: the most common
              layout in the swarm (direct pass-through). Any other chunking is assembled on the fly.
    step:     time step of the view ("6h", 3600, ...). Replicas at a finer step serve it by subsampling, so
              hourly and 6-hourly copies both fill a 6-hourly view. Default: the finest step in the swarm
              (coarser replicas then fill only their own samples).
    Remaining kwargs go to xarray.open_zarr (e.g. chunks={} for dask).
    """
    ctl = ctl or CTL
    if zarr.config.get("async.concurrency") < 256:
        zarr.config.set({"async.concurrency": 256})  # let zarr issue whole slices at once -> big batches
    # lazy numpy-backed arrays by default: one zarr read per selection -> one big batch -> one optimal plan.
    # Pass chunks={} / "auto" for dask (then zt.prefetch() first to keep planning global).
    kw.setdefault("chunks", None)
    kw.setdefault("consolidated", False)
    parts, owners = [], {}
    for grid, view in open_views(link, ctl, chunking, step):  # one store per sub-grid, merged on shared coords
        store = ZtStore(ctl, grid, view)
        store.pushdown = pushdown
        d = xr.open_zarr(store, **dict(kw, zarr_format=kw.get("zarr_format", view.fmt)))
        owners.update({v: store.token for v in d.data_vars})
        parts.append(d)
    ds = parts[0] if len(parts) == 1 else xr.merge(parts, join="outer", compat="override", combine_attrs="override")
    ds.attrs["zt_grid"] = "+".join(s.grid for s in (_STORES[t] for t in dict.fromkeys(owners.values())))
    ds.attrs["zt_store"] = next(iter(owners.values()), "")
    ds.attrs["zt_stores"] = json.dumps(owners)
    return ds


def store_of(ds, var: str | None = None) -> ZtStore:
    tok = json.loads(ds.attrs.get("zt_stores", "{}")).get(var) if var else None
    s = _STORES.get(tok or ds.attrs.get("zt_store", ""))
    if s is None:
        raise ValueError("dataset was not opened with zarr_torrent.open_dataset")
    return s


def keys_for(view: View, variables=None, t0=None, t1=None) -> list[str]:
    """Source chunk keys (cheapest layouts) covering variables x [t0, t1]. No data is read."""
    out = []
    for n in view.arrays:
        if variables and n not in variables:
            continue
        out += [k for k, _, _ in view.sources(n, view.region(n, t0, t1))]
    return list(dict.fromkeys(out))


def prefetch(obj, wait: bool = True, label: str = "", progress=None) -> list[str]:
    """Download every source chunk `obj` (a lazily selected Dataset/DataArray) needs, in one plan.

    Keys are discovered by a dry read in which the store records keys and returns fill values.
    ponytail: the dry read allocates the selection in memory; use keys_for() for huge slices.
    """
    stores = list(_STORES.values())
    for s in stores:
        s.recording = set()
    try:
        obj.copy(deep=False).load()
    finally:
        rec = {s: s.recording for s in stores}
        for s in stores:
            s.recording = None
    jobs = [(s.ctl, http(s.ctl, "POST", "/api/download", {"grid": s.grid, "keys": sorted(k), "label": label})["job"])
            for s, k in rec.items() if k]
    if wait:
        for ctl, jid in jobs:
            wait_job(ctl, jid, progress)
    return [j for _, j in jobs]


def wait_job(ctl: str, jid: str, progress=None, every: float = 0.3, deadline: float | None = None) -> dict:
    t0 = time.time()
    while True:
        if deadline and time.time() - t0 > deadline:
            raise TimeoutError(f"job {jid} still running after {deadline:.0f} s")
        job = http(ctl, "GET", f"/api/job/{jid}?since=1000000000", timeout=60)
        if progress:
            progress(job)
        if job["state"] != "running":
            return job
        time.sleep(every)


def progressive_mean(ds, var: str, t0=None, t1=None, rel_err: float = 0.01, z: float = 1.96,
                     min_chunks: int = 8, callback=None) -> dict:
    """Anytime, error-bounded mean of `var` over [t0, t1] while the slice is still downloading.

    Chunks arrive in a low-discrepancy order (progressive job), so each prefix is a
    near-stratified cluster sample. Ratio estimator R = sum(y)/sum(x) over sampled chunks,
    Var(R) ~ (1 - n/N) * s^2 / (n * mean(x)^2), s^2 = sum((y - R x)^2)/(n - 1) (conservative
    for stratified prefixes). The download is cancelled once z*sd <= rel_err*|R|.
    """
    st = store_of(ds, var)
    view = st.view
    a = view.arrays[var]
    plan = {k: src for k, src, _ in view.sources(var, view.region(var, t0, t1))}
    jid = http(st.ctl, "POST", "/api/download", {"grid": st.grid, "keys": list(plan), "order": "progressive",
                                                   "label": f"progressive {var}"})["job"]
    ys, xs, seen, t_start, res = [], [], 0, time.time(), {}
    n = N = 0
    while True:
        job = http(st.ctl, "GET", f"/api/job/{jid}?since={seen}")
        new = job["done_keys"]
        seen += len(new)
        if new:
            paths = http(st.ctl, "POST", "/api/fetch", {"grid": st.grid, "keys": new})
            for k in new:
                p = paths.get(k)
                if not p or p == "!" or k not in plan:
                    continue
                _, lay, _ = split_key(k)
                docs = a["layouts"][lay]["docs"]
                v = codec.decode(docs, read_chunk(st.ctl, st.grid, k, p))[plan[k]].astype("float64")
                _, fill = _dtype_fill(json.dumps(docs))
                ok = np.isfinite(v)
                if fill is not None and np.isfinite(float(fill)):
                    ok &= v != float(fill)
                if ok.any():
                    ys.append(float(v[ok].sum()))
                    xs.append(int(ok.sum()))
        N = max(job["total"] - job["missing"], 1)
        n = len(xs)
        if n >= 2:
            y, x = np.array(ys), np.array(xs)
            R = y.sum() / x.sum()
            s2 = ((y - R * x) ** 2).sum() / (n - 1)
            sd = math.sqrt(max(1 - n / N, 0) * s2 / (n * x.mean() ** 2))
            from .aqp import t_quantile
            res = {"mean": R, "ci95": t_quantile(z, n - 1) * sd, "chunks": n, "of": N, "bytes": job["bytes"],
                   "seconds": time.time() - t_start, "exact": job["state"] != "running" and n >= N}
            if callback:
                callback(res)
            if n >= min_chunks and res["ci95"] <= rel_err * abs(R) and job["state"] == "running":
                http(st.ctl, "POST", f"/api/cancel/{jid}")
                return res
        if job["state"] != "running" and not new:
            return res or {"mean": None, "chunks": n, "of": N, "state": job["state"]}
        time.sleep(0.1)


def _chunk_mean(st, a, k, path, src):
    """(sum, count) of valid values of source chunk k restricted to the request."""
    _, lay, _ = split_key(k)
    docs = a["layouts"][lay]["docs"]
    v = codec.decode(docs, read_chunk(st.ctl, st.grid, k, path))[src].astype("float64")
    _, fill = _dtype_fill(json.dumps(docs))
    ok = np.isfinite(v)
    if fill is not None and np.isfinite(float(fill)):
        ok &= v != float(fill)
    return float(v[ok].sum()), int(ok.sum())


def progressive_mean_vas(ds, var: str, t0=None, t1=None, rel_err: float = 0.01, z: float = 1.96,
                         strata: int | None = None, pilot: int = 5, batch: int = 8, seed: int = 0,
                         callback=None) -> dict:
    """VAS: variance-aware anytime scheduling (stratified sampling + sequential greedy Neyman allocation,
    Satterthwaite-t intervals; see aqp.VAS). Strata are contiguous time blocks, H = clip(N/30, 2, 16)."""
    from .aqp import VAS
    rng = np.random.default_rng(seed)
    st = store_of(ds, var)
    view = st.view
    a = view.arrays[var]
    plan = {k: src for k, src, _ in view.sources(var, view.region(var, t0, t1))}
    T = a["taxis"]
    keys = sorted(plan, key=lambda k: (split_key(k)[2][T] if T is not None else 0, k))
    H = max(1, min(strata or int(np.clip(len(keys) // 30, 2, 16)), len(keys)))
    groups = [list(rng.permutation(np.array(g, dtype=object))) for g in np.array_split(np.array(keys, dtype=object), H)]
    est = VAS([len(g) for g in groups], z=z, pilot=pilot, min_h=pilot)
    ptr = [0] * H
    t_start, nbytes = time.time(), 0

    def fetch(hs):
        nonlocal nbytes
        ks = []
        for h in hs:
            ks.append((h, groups[h][ptr[h]]))
            ptr[h] += 1
        jid = http(st.ctl, "POST", "/api/download", {"grid": st.grid, "keys": [k for _, k in ks],
                                                       "label": f"vas {var}"})["job"]
        nbytes += wait_job(st.ctl, jid, every=0.05)["bytes"]
        paths = http(st.ctl, "POST", "/api/fetch", {"grid": st.grid, "keys": [k for _, k in ks]})
        for h, k in ks:
            p = paths.get(k)
            if p and p != "!":
                y, x = _chunk_mean(st, a, k, p, plan[k])
                if x:
                    est.add(h, y / x)

    fetch(est.pilot_batch())
    while True:
        mean, hw, n = est.estimate()
        res = {"mean": mean, "ci95": hw, "chunks": int(est.taken.sum()), "of": len(keys), "bytes": nbytes,
               "seconds": time.time() - t_start, "exact": bool((est.taken >= est.N_h).all()), "strata": H}
        if callback:
            callback(res)
        if est.done(rel_err):
            return res
        fetch(est.next_batch(batch))
