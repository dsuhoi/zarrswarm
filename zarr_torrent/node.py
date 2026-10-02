"""zt node: DHT participant, chunk server, optional relay, download engine, local control API.

Ports:
  data  (--port)     : /dht, /m/<grid>, /cb/<grid>, /pex/<grid>, relay endpoints. May be public.
  control (--ctl)    : /api/*, bound to 127.0.0.1 only. Used by CLI, TUI and the xarray store.
"""
import asyncio
import glob
import hashlib
import hmac
import json
import os
import random
import struct
import time
import uuid
import zlib
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from aiohttp import web

import numpy as np

from . import capt, codec, jlps as jlpsmod, parity as par, plan as planmod, pushdown as pd
from .common import Identity, check_signed, cid_of, cjson, h160, verify
from .dht import DHT
from .scan import chunk_extent, chunk_g, chunks_for, grid_id_of, natural_stride, scan, split_key, tstride

ANNOUNCE_EVERY = float(os.environ.get("ZT_ANNOUNCE_EVERY", "600"))
VIEW_TTL = 30
CONC_PER_PEER = 4
BATCH_CHUNKS = 64
BATCH_BYTES = 32 << 20
BATCH_SECONDS = 1.0
DEFAULT_BW = 20e6
RELAY_PATHS = ("dht", "m/", "mh/", "mp/", "cb/", "pex/", "whoami", "qb/")
PAGE_CACHE_BYTES = int(os.environ.get("ZT_PAGE_CACHE_MB", "256")) << 20
CAPT_MIN_INTERVAL = 2.0  # s between manifest-tree rebuilds of one grid (heads may lag the index by this much)
AUDIT_RATE = float(os.environ.get("ZT_AUDIT", "0.05"))
PD_MIN_AUDITS = 3
PD_WHOLE_FRAC = float(os.environ.get("ZT_PD_WHOLE_FRAC", "0.25"))  # every untrusted peer is audited on its first results (no free first strike)  # optimistic pushdown: fraction of results re-derived
RELAY_MAX_PEERS = 2000
RELAY_MAX_INFLIGHT = 512
RELAY_MAX_INFLIGHT_PER_PEER = 32
RELAY_MAX_MSG = 72 << 20          # > BATCH_BYTES + framing (legacy single-frame responses)
RELAY_FRAME = int(os.environ.get("ZT_RELAY_FRAME_KB", "4096")) << 10  # cut-through relay: body frame size
MAX_MANIFEST = 1 << 30            # decompressed manifest bound (~2e7 chunks)
MISSING = 0xFFFFFFFF
# lossy WAN links (seen on real clusters: ~50% of connections to one host stalled): fail fast per attempt, retry
T_META = aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=10)  # read timeout = max gap between reads
T_DATA = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
META_TRIES = 3
PEER_MAX_ERRORS = 3
FETCH_ROUNDS = 4
XT1_BELOW = float(os.environ.get("ZT_XT1_BELOW", str(30e6)))  # ask for the xt1 transport codec below this link rate
VIEW_WAIT = 8.0
STALL_FIRST = 30.0  # s: read timeout for a peer we have not measured yet
HINT_EFF = 0.8  # announced link rates vs achieved goodput (measured ~0.77 in the simulator): calibrate first contact
FOLLOW_EVERY = float(os.environ.get("ZT_FOLLOW_EVERY", "300"))  # s between subscription catch-ups
NET_SKEW = 120  # s: tolerated clock difference for closed-network request MACs
RESCAN_EVERY = float(os.environ.get("ZT_RESCAN_EVERY", "60"))  # s between checks of seeded datasets for appends
STALL_MIN = float(os.environ.get("ZT_STALL_MIN", "5.0"))  # s without a byte before a batch counts as stalled


def _frame(hdr: dict, body: bytes = b"") -> bytes:
    h = json.dumps(hdr).encode()
    return struct.pack(">I", len(h)) + h + body


def _unframe(b: bytes) -> tuple[dict, bytes]:
    n = struct.unpack(">I", b[:4])[0]
    return json.loads(b[4:4 + n]), b[4 + n:]


CTL_BIND = os.environ.get("ZT_CTL_HOST", "127.0.0.1")  # emulation testbeds drive nodes over their own address
CTL_HOSTS = ("127.0.0.1", "localhost", "[::1]") + ((CTL_BIND,) if CTL_BIND != "127.0.0.1" else ())


@web.middleware
async def _ctl_guard(request, handler):
    """Control API is local-only and not callable from web pages: browsers cannot add a custom header to a
    cross-origin request without a CORS preflight (which this server never approves), and the Host check
    defeats DNS rebinding."""
    host = request.headers.get("Host", "").rsplit(":", 1)[0]
    if host not in CTL_HOSTS or request.headers.get("X-Zt-Client") != "1":
        raise web.HTTPForbidden(text="zt control API: local clients only (X-Zt-Client header required)")
    return await handler(request)


def parse_link(link: str) -> tuple[str, str | None]:
    """zt://<grid_id>  or  zt://<name>@<pubkey>  ->  (grid_or_name, pubkey|None)."""
    s = link.removeprefix("zt://").strip("/")
    if "@" in s:
        name, pk = s.rsplit("@", 1)
        return name, pk
    return s, None


def _trim():
    """Return freed heap to the OS (glibc only; a no-op elsewhere)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class Node:
    def __init__(self, home: str | Path = "~/.zt", host: str = "127.0.0.1", port: int = 7881,
                 ctl_port: int | None = None, public: str | None = None, bootstrap: list[str] = (),
                 relay: str | None = None, relay_server: bool = False, auto: bool = False,
                 rate: float | None = None, latency: float = 0.0, strategy: str = "maxflow",
                 trust: list[str] = (), loss: float = 0.0, seeds: list[str] = (), network_key: str | None = None,
                 cache_max: int = 0):
        self.home = Path(home).expanduser()
        self.net_key = network_key or None  # closed network: every data-port request must carry it (X-Zt-Net)
        self.cfg_seeds = list(seeds)  # config [[seed]] paths/globs: re-read on every start, not persisted
        self.host, self.port, self.ctl_port = host, port, ctl_port or port + 1
        self.bootstrap_urls, self.relay, self.relay_server = list(bootstrap), relay, relay_server
        self.ident = Identity(self.home)
        if public:
            self.addr, ro = public.rstrip("/"), False
        elif relay:
            self.addr, ro = f"{relay.rstrip('/')}/r/{self.ident.id}", True
        else:
            self.addr, ro = f"http://{host}:{port}", False
        self.ro = ro
        self.auto = auto and not public and not relay
        # emulation / experiments (simulator): upload cap in bytes/s, one-way latency, peer-selection strategy
        self.rate, self.latency, self.strategy = rate, latency, strategy
        self.loss = loss  # emulation: probability that a data-port request stalls (lossy WAN link)
        self.audit_rate = AUDIT_RATE
        self.xt1 = True     # negotiate the value-level transport codec on slow links
        self.cheat = False  # fault injection (tests/simulator): sign deliberately wrong pushdown results
        # trusted publisher pubkeys (hex): their vcid wins over any number of other holders (anti-Sybil anchor)
        self.trusted = {h160(bytes.fromhex(pk)) for pk in (list(trust) + os.environ.get("ZT_TRUST", "").split(",")) if pk}
        self._up_next = 0.0
        self._t_start = time.time()
        self.hints: dict[str, float] = {}
        self.relay_bw: dict[str, float | None] = {}
        self._capt: dict[str, tuple] = {}
        self._capt_t: dict[str, float] = {}
        self._capt_building: dict[str, asyncio.Future] = {}
        self._views_obj: dict[int, object] = {}  # store.View per swarm-view dict (value parity assembly)
        self.page_cache: OrderedDict[str, bytes] = OrderedDict()
        self._page_bytes = 0
        self.page_fetched = 0
        self.decoded = pd.DecodedLRU(int(os.environ.get("ZT_DECODED_MB", "256")) << 20)
        self.bad: set[str] = set()                # peers caught cheating (local blacklist)
        self.seen_heads: dict[tuple[str, str], dict] = {}  # (peer, grid) -> newest verified manifest head
        try:
            self._heads = json.loads((self.home / "heads.json").read_text())  # our own versioned heads
        except (OSError, ValueError):
            self._heads = {}
        self.fraud: list[dict] = []               # signed receipts that contradict verified data (fraud proofs)
        self.rep: dict[str, list[int]] = {}       # peer -> [audits passed, audits failed]
        self.audit_started: dict[str, int] = {}
        self.audit_decided: dict[tuple[str, str], bool] = {}
        self.accepted: dict[str, list[dict]] = {}  # receipts accepted per peer (taint list if it is later caught)
        self.tainted: list[dict] = []
        self.pd_stats = {"items": 0, "remote": 0, "local": 0, "audits": 0, "fallback": 0, "bytes": 0}
        self.served = {"bytes": 0, "chunks": 0, "relayed_bytes": 0, "dht_rpcs": 0}
        self.seeds: dict[str, dict] = {}          # path -> scan result
        self.caches: dict[str, dict] = {}         # grid -> {"grid","gdocs","arrays","chunks"}
        self.local: dict[str, dict] = {}          # grid -> merged local index
        self._mf: dict[str, bytes] = {}           # grid -> signed compressed manifest body
        self.views: dict[str, dict] = {}
        self.peer_manifests: dict[tuple[str, str], tuple[float, dict, str]] = {}
        self._view_memo: dict[str, tuple[tuple, dict]] = {}
        self._lver = 0
        self.bw: dict[str, float] = {}
        self.pending: dict[tuple[str, str], asyncio.Future] = {}
        self.jobs: dict[str, dict] = {}
        self.relayed: dict[str, tuple[web.WebSocketResponse, asyncio.Lock]] = {}
        self.relay_waits: dict[str, asyncio.Future] = {}
        self.relay_owner: dict[str, web.WebSocketResponse] = {}
        self.relay_streams: dict[str, asyncio.Queue] = {}
        self.relay_up = False
        self._dirty: set[str] = set()
        self.names: dict[str, list] = {}
        self.subs: dict[str, dict] = {}  # follow subscriptions: id -> {link, vars, last_s, isel}
        self.cache_max = cache_max        # bytes of downloaded chunks to keep (0 = unlimited), LRU eviction
        self._job_keys: dict[str, tuple] = {}  # job id -> (keys, region view) for resume
        self.used: dict[tuple[str, str], float] = {}  # (grid, key) -> last use (download / client read / serve)
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ lifecycle
    async def start(self):
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=128, limit_per_host=16),
                                             middlewares=(self._net_sign,) if self.net_key else ())
        self.dht = DHT(self.ident, self.addr, self.session, ro=self.ro)
        (self.home / "cas").mkdir(parents=True, exist_ok=True)
        (self.home / "cache").mkdir(parents=True, exist_ok=True)
        for p in sorted((self.home / "cache").glob("*.json")):
            d = json.loads(p.read_text())
            self.caches[p.stem] = d
        data = web.Application(client_max_size=64 << 20, middlewares=[self._net_guard, self._bad_input, self._lossy])
        data.add_routes([web.post("/dht", self.h_dht), web.get("/m/{grid}", self.h_manifest),
                         web.post("/cb/{grid}", self.h_chunks), web.get("/pex/{grid}", self.h_pex),
                         web.post("/qb/{grid}", self.h_pushdown), web.get("/mh/{grid}", self.h_mhead),
                         web.get("/mp/{grid}/{h}", self.h_mpage),
                         web.get("/whoami", self.h_whoami), web.post("/probe", self.h_probe)])
        if self.relay_server:
            data.add_routes([web.get("/relay/attach", self.h_relay_attach),
                             web.route("*", "/r/{nid}/{tail:.*}", self.h_relay_forward)])
        ctl = web.Application(client_max_size=64 << 20, middlewares=[_ctl_guard])
        ctl.add_routes([web.get("/api/status", self.a_status), web.post("/api/seed", self.a_seed),
                        web.post("/api/unseed", self.a_unseed), web.get("/api/resolve", self.a_resolve),
                        web.get("/api/view/{grid}", self.a_view), web.post("/api/fetch", self.a_fetch),
                        web.post("/api/read", self.a_read),
                        web.post("/api/download", self.a_download), web.get("/api/peers/{grid}", self.a_peers),
                        web.post("/api/name", self.a_name), web.get("/api/job/{jid}", self.a_job),
                        web.post("/api/cancel/{jid}", self.a_cancel), web.get("/api/search", self.a_search), web.get("/api/pieces/{grid}", self.a_pieces),
                        web.post("/api/slices", self.a_slices), web.post("/api/parity", self.a_parity),
                        web.post("/api/follow", self.a_follow), web.get("/api/follows", self.a_follows),
                        web.post("/api/unfollow", self.a_unfollow), web.post("/api/pause/{jid}", self.a_pause),
                        web.post("/api/resume/{jid}", self.a_resume)])
        self._runners = []
        for app, host, port in ((data, self.host, self.port), (ctl, CTL_BIND, self.ctl_port)):
            r = web.AppRunner(app, access_log=None, shutdown_timeout=2)
            await r.setup()
            await web.TCPSite(r, host, port).start()
            self._runners.append(r)
        state = self.home / "state.json"
        st = json.loads(state.read_text()) if state.exists() else {}
        paths = list(dict.fromkeys(st.get("seeds", []) + [m for g in self.cfg_seeds
                                                           for m in sorted(glob.glob(os.path.expanduser(g))) or [g]]))
        self.names = st.get("names", {})
        self.subs = st.get("subs", {})
        for p in paths:
            try:
                await self.add_seed(p, persist=False, announce=False)
            except Exception as e:  # dataset moved/removed: keep node running
                print(f"[zt] cannot reseed {p}: {e}")
        self._rebuild()
        if self.auto:
            await self._auto_address()
        if self.relay:
            self._tasks.append(asyncio.create_task(self._relay_client()))
            for _ in range(50):
                if self.relay_up:
                    break
                await asyncio.sleep(0.1)
        await self.dht.bootstrap(self.bootstrap_urls)
        self._tasks += [asyncio.create_task(self._announce_loop()), asyncio.create_task(self._flush_loop()),
                        asyncio.create_task(self._rejoin_loop()), asyncio.create_task(self._rescan_loop()),
                        asyncio.create_task(self._follow_loop()), asyncio.create_task(self._evict_loop())]

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        for ws, _ in list(self.relayed.values()):
            await ws.close()
        self._flush()
        for r in self._runners:
            await r.cleanup()
        await self.session.close()

    # ------------------------------------------------------------------ local index
    def _rebuild(self):
        local = {}
        for src in list(self.seeds.values()) + [dict(c, files=None) for c in self.caches.values()]:
            gid = src.get("grid_id") or grid_id_of(src["grid"])
            L = local.setdefault(gid, {"grid": src["grid"], "gdocs": src["gdocs"], "arrays": {},
                                       "chunks": {}, "files": {}})
            for n, a in src["arrays"].items():
                La = L["arrays"].get(n)
                if La is None:
                    L["arrays"][n] = dict(a, layouts=dict(a["layouts"]))
                elif La["vfid"] == a["vfid"]:
                    for lay, li in a["layouts"].items():
                        La["layouts"].setdefault(lay, li)
            for ck, ent in src["chunks"].items():
                n, lay, _ = split_key(ck)
                La, sa = L["arrays"][n], src["arrays"][n]
                if ck in L["chunks"] or La["vfid"] != sa["vfid"] or La["layouts"][lay]["fid"] != sa["layouts"][lay]["fid"]:
                    continue
                L["chunks"][ck] = ent
                L["files"][ck] = src["files"][ck] if src["files"] else str(self._cas(ent[0]))
        self.local = local
        self._mf.clear()
        self._capt.clear()  # explicit (re)seed: publish immediately, no rate limit
        self._capt_t.clear()
        self._lver += 1

    def _cas(self, cid: str) -> Path:
        return self.home / "cas" / cid[:2] / cid

    async def add_seed(self, path: str, persist=True, announce=True) -> dict:
        res = await asyncio.to_thread(scan, path, self.home / "scan")
        _trim()  # a scan decodes every chunk on 16 threads; glibc keeps those arenas unless asked to return them
        for k in [k for k in self.seeds if k[0] == res["path"]]:
            del self.seeds[k]
        for gid, sg in res["subgrids"].items():
            self.seeds[(res["path"], gid)] = dict(sg, path=res["path"])
        self._rebuild()
        if persist:
            self._save_state()
        if announce:
            await asyncio.gather(*(self.announce(g) for g in res["subgrids"]))
        return res

    @staticmethod
    def _meta_sig(path: str):
        """Cheap change signature of a Zarr store: its metadata files (an append along time rewrites the array
        shape). None if the store is gone."""
        root = Path(path)
        try:
            return tuple(sorted((str(f.relative_to(root)), f.stat().st_mtime_ns, f.stat().st_size)
                                for pat in ("zarr.json", ".zarray", ".zattrs", ".zgroup", "*/zarr.json", "*/.zarray")
                                for f in root.glob(pat)))
        except OSError:
            return None

    def evict(self, limit: int, keep_recent_s: float = 600) -> int:
        """Keep downloaded chunks (the CAS cache) under `limit` bytes: least recently used first (downloaded,
        read by a client or served to a peer), down to 90 %. Never touched: the user's own seeded files, volunteer
        parity, chunks used in the last `keep_recent_s`. Returns bytes freed."""
        refs: dict[str, int] = {}
        total = 0
        for c in self.caches.values():
            for e in c["chunks"].values():
                refs[e[0]] = refs.get(e[0], 0) + 1
                total += e[2] if refs[e[0]] == 1 else 0  # identical bytes are stored once
        if total <= limit:
            return 0
        now = time.time()
        cand = []
        for g, c in self.caches.items():
            for k, e in c["chunks"].items():
                if par.is_parity(k.partition("@")[0]):
                    continue
                t = self.used.get((g, k))
                if t is None:
                    try:
                        t = self._cas(e[0]).stat().st_mtime
                    except OSError:
                        t = 0.0
                if now - t >= keep_recent_s:
                    cand.append((t, g, k))
        freed, target, touched = 0, total - int(0.9 * limit), set()
        for t, g, k in sorted(cand):
            if freed >= target:
                break
            e = self.caches[g]["chunks"].pop(k)
            self.used.pop((g, k), None)
            touched.add(g)
            refs[e[0]] -= 1
            if refs[e[0]] == 0:
                self._cas(e[0]).unlink(missing_ok=True)
                freed += e[2]
        if touched:
            self._dirty |= touched
            self._rebuild()
        return freed

    async def _evict_loop(self):
        while True:
            await asyncio.sleep(30)
            if self.cache_max:
                freed = self.evict(self.cache_max)  # on the loop: it mutates caches/local (no thread races)
                if freed:
                    print(f"[zt] cache over {self.cache_max / 1e9:g} GB: evicted {freed / 1e6:,.1f} MB (LRU)")
                    for g in list(self.caches):
                        await self.announce(g)

    async def _rescan_loop(self):
        """Growing datasets (operational feeds appending time steps) are re-scanned and re-announced without a
        manual `zt seed`: metadata signature every RESCAN_EVERY s; a full (stat-cached) rescan every 10th round
        also catches chunks rewritten in place (e.g. preliminary -> final values)."""
        sigs: dict[str, tuple] = {}
        n = 0
        while True:
            await asyncio.sleep(RESCAN_EVERY)
            n += 1
            for path in sorted({p for p, _ in self.seeds}):
                sig = await asyncio.to_thread(self._meta_sig, path)
                if sig is None:
                    continue
                if sigs.get(path) == sig and n % 10:
                    continue
                first = path not in sigs
                sigs[path] = sig
                if first and n % 10:
                    continue  # baseline taken; the seed itself was scanned at add time
                before = {k: e[:2] for (p_, _), sd in self.seeds.items() if p_ == path for k, e in sd["chunks"].items()}
                try:
                    await self.add_seed(path, persist=False, announce=False)
                except Exception as e:
                    print(f"[zt] rescan {path}: {e}")
                    continue
                after = {k: e[:2] for (p_, _), sd in self.seeds.items() if p_ == path for k, e in sd["chunks"].items()}
                if after != before:
                    for g in {g for p_, g in self.seeds if p_ == path}:
                        await self.announce(g)
                    print(f"[zt] {path}: {len(set(after) - set(before))} new, "
                          f"{sum(1 for k in after if k in before and after[k] != before[k])} changed chunks")

    def _save_heads(self):
        try:
            tmp = self.home / "heads.json.tmp"
            tmp.write_text(json.dumps(self._heads))
            tmp.replace(self.home / "heads.json")
        except OSError:
            pass

    def _save_state(self):
        (self.home / "state.json").write_text(json.dumps({"seeds": sorted({p for p, _ in self.seeds}),
                                                          "names": self.names,
                                                          "subs": {i: {k: v for k, v in s.items() if k != "jobs"}
                                                                   for i, s in self.subs.items()}}))

    def _flush(self):
        for g in list(self._dirty):
            p = self.home / "cache" / f"{g}.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.caches[g]))
            tmp.replace(p)
        self._dirty.clear()

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(5)
            self._flush()

    # ------------------------------------------------------------------ announce / discovery
    async def announce(self, grid: str) -> int:
        now = time.time()
        rec = self.ident.signed({"t": "peer", "grid": grid, "node": self.ident.id, "addr": self.addr, "ts": now,
                                 "bw": self.rate or self.up_estimate()})
        n = await self.dht.store(grid, rec)
        L = self.local.get(grid)
        if L:  # metadata index: one record per tag (variable name, standard_name, long_name)
            ext = local_extent(L)
            t = L["grid"].get("time")
            # inclusive time range (last step held, not the exclusive extent end)
            tr = None if not ext or not t else [ext[0] * t["dt"] + t["rphase"], (ext[1] - 1) * t["dt"] + t["rphase"]]
            data_var = lambda n: not par.is_parity(n) and n not in L["grid"]["dims"]  # coordinate arrays are no data
            vars_ = sorted(n for n in L["arrays"] if data_var(n))
            tags = set(vars_)
            for n_, a in L["arrays"].items():
                if not data_var(n_):
                    continue
                for k in ("standard_name", "long_name"):
                    if isinstance(a.get("attrs", {}).get(k), str):
                        tags.add(a["attrs"][k].lower())
            for tag in tags:
                idx = self.ident.signed({"t": "idx", "tag": tag, "grid": grid, "node": self.ident.id, "vars": vars_,
                                         "tr": tr, "dims": L["grid"]["dims"], "ts": now})
                await self.dht.store(tag_key(tag), idx)
        return n

    def _prune(self):
        """Bound long-running memory: finished jobs, stale peer manifests and views."""
        now = time.time()
        fin = [j for j, x in self.jobs.items() if x["state"] != "running"]
        for j in fin[:-200]:
            self.jobs.pop(j, None)
        for x in list(self.jobs.values())[:-20]:
            if x["state"] != "running":
                x["done_keys"] = []
        for key, hit in list(self.peer_manifests.items()):
            if now - hit[0] > 20 * VIEW_TTL:
                del self.peer_manifests[key]
        for g, v in list(self.views.items()):
            if now - v["ts"] > 3600:
                self.views.pop(g, None)
                self._view_memo.pop(g, None)

    def up_estimate(self) -> float | None:
        """Self-reported uplink hint for planners (bytes/s): configured cap or observed serving rate."""
        el = time.time() - self._t_start
        return round(self.served["bytes"] / el) if el > 60 and self.served["bytes"] > 50e6 else None

    async def search(self, tag: str) -> list[dict]:
        out = {}
        for r in await self.dht.get(tag_key(tag.lower())):
            if r.get("t") != "idx":
                continue
            g = out.setdefault(r["grid"], {"grid": r["grid"], "vars": set(), "seeders": set(), "tr": None,
                                            "dims": r.get("dims")})
            g["vars"].update(r.get("vars", []))
            g["seeders"].add(r["node"])
            if r.get("tr"):
                g["tr"] = r["tr"] if not g["tr"] else [min(g["tr"][0], r["tr"][0]), max(g["tr"][1], r["tr"][1])]
        return [dict(g, vars=sorted(g["vars"]), seeders=len(g["seeders"])) for g in out.values()]

    async def _auto_address(self):
        """STUN-like: learn our external IP from a bootstrap node, ask it to connect back.
        Reachable -> public address; otherwise attach to the bootstrap's relay."""
        for b in self.bootstrap_urls:
            b = b.rstrip("/")
            try:
                async with self.session.get(f"{b}/whoami", timeout=aiohttp.ClientTimeout(total=8)) as r:
                    who = await r.json()
                cand = f"http://{who['ip']}:{self.port}"
                async with self.session.post(f"{b}/probe", json={"port": self.port},
                                             timeout=aiohttp.ClientTimeout(total=15)) as r:
                    pr = await r.json()
            except Exception as e:
                print(f"[zt] auto-address via {b} failed: {e}")
                continue
            if pr.get("id") == self.ident.id:
                self.addr, self.ro = cand, False
            elif who.get("relay"):
                self.relay, self.addr, self.ro = b, f"{b}/r/{self.ident.id}", True
            else:
                continue
            self.dht.contact.update(addr=self.addr, ro=self.ro)
            print(f"[zt] auto address: {self.addr} ({'relay' if self.ro else 'public'})")
            return

    async def _rejoin_loop(self):
        """Flaky or slow bootstrap: keep retrying every 15 s while we know nobody, then announce."""
        while True:
            await asyncio.sleep(15)
            if not self.dht.contacts() and self.bootstrap_urls:
                await self.dht.bootstrap(self.bootstrap_urls)
                if self.dht.contacts():
                    for g in list(self.local):
                        await self.announce(g)

    async def _announce_loop(self):
        while True:
            if not self.dht.contacts() and self.bootstrap_urls:  # lost the network (flaky link): rejoin
                await self.dht.bootstrap(self.bootstrap_urls)
            for g in list(self.local):
                try:
                    await self.announce(g)
                except Exception as e:
                    print(f"[zt] announce {g[:8]} failed: {e}")
            for name, (target, seq) in list(self.names.items()):
                try:
                    await self.publish_name(name, target, seq)
                except Exception as e:
                    print(f"[zt] republish name {name} failed: {e}")
            self._prune()
            await asyncio.sleep(ANNOUNCE_EVERY)

    async def find_peers(self, grid: str, _again: bool = True) -> dict[str, str]:
        if not self.dht.contacts() and self.bootstrap_urls:  # lost/failed bootstrap: rejoin before searching
            await self.dht.bootstrap(self.bootstrap_urls)
        recs = [r for r in await self.dht.get(grid) if r.get("t") == "peer"]
        if not recs and _again and self.bootstrap_urls:  # lossy lookup: one more try through a fresh bootstrap
            await self.dht.bootstrap(self.bootstrap_urls)
            return await self.find_peers(grid, _again=False)
        peers = {r["node"]: r["addr"] for r in recs}
        self.hints.update({r["node"]: float(r["bw"]) for r in recs if r.get("bw")})
        for nid, addr in list(peers.items())[:3]:  # PEX: ask a few peers whom they know
            if nid == self.ident.id:
                continue
            try:
                async with self.session.get(f"{addr}/pex/{grid}", timeout=T_META) as r:
                    for rec in await r.json():
                        if self.dht.put_local(grid, rec):
                            peers.setdefault(rec["node"], rec["addr"])
            except Exception:
                pass
        peers.pop(self.ident.id, None)
        return peers

    async def resolve(self, link: str) -> str:
        name, pk = parse_link(link)
        if pk is None:
            return name
        key = h160(bytes.fromhex(pk) + name.encode())
        recs = [r for r in await self.dht.get(key) if r.get("t") == "name" and r.get("pk") == pk]
        if not recs:
            raise KeyError(f"name {name}@{pk[:8]} not found in DHT")
        return parse_link(max(recs, key=lambda r: r["seq"])["target"])[0]

    # ------------------------------------------------------------------ data-port handlers
    def _net_mac(self, ts: int, method: str, path: str) -> str:
        return hmac.new(self.net_key.encode(), f"{ts}\n{method}\n{path}".encode(), "sha256").hexdigest()

    async def _net_sign(self, req, handler):
        """Client side of a closed network: the key never travels. Each request carries a MAC of its method, path and
        time (it used to carry the key itself, in clear, on every HTTP request)."""
        ts = int(time.time())
        req.headers["X-Zt-Net"] = f"{ts}:{self._net_mac(ts, req.method, req.url.raw_path_qs)}"
        return await handler(req)

    @web.middleware
    async def _net_guard(self, request, handler):
        """Closed network: without a valid MAC under the shared key nothing is served - no chunks, manifests, DHT,
        relay or probe. A MAC binds method, path and a timestamp (+-NET_SKEW s), so a captured header opens nothing
        else and expires."""
        if self.net_key:
            ts, _, mac = request.headers.get("X-Zt-Net", "").partition(":")
            ok = ts.isdigit() and abs(time.time() - int(ts)) <= NET_SKEW and hmac.compare_digest(
                mac.encode(), self._net_mac(int(ts), request.method, request.raw_path).encode())
            if not ok:
                raise web.HTTPForbidden(text="zt: network key required")
        return await handler(request)

    @web.middleware
    async def _bad_input(self, request, handler):
        """Untrusted peers: malformed bodies/paths are the requester's error (400), not a server fault (500 +
        traceback in the log). Handlers index into request JSON directly; this is the one place that maps it."""
        try:
            return await handler(request)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as e:  # JSONDecodeError is a ValueError
            raise web.HTTPBadRequest(text=f"zt: bad request ({type(e).__name__})") from None

    @web.middleware
    async def _lossy(self, request, handler):
        if self.loss and random.random() < self.loss:  # stalled connection: the client's read timeout handles it
            await asyncio.sleep(3600)
        return await handler(request)

    async def _pace(self, n: int, key: str = "bytes"):
        """Token bucket on this node's upstream (emulated link)."""
        self.served[key] += n
        if not self.rate:
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        start = max(now, self._up_next)
        self._up_next = start + n / self.rate
        await asyncio.sleep(self._up_next - now)

    async def h_dht(self, req):
        self.served["dht_rpcs"] += 1
        if self.latency:
            await asyncio.sleep(self.latency)
        try:
            msg = await req.json()
        except Exception:  # truncated body (peer died mid-request) or garbage
            raise web.HTTPBadRequest()
        return web.json_response(self.dht.handle(msg))

    async def h_manifest(self, req):
        g = req.match_info["grid"]
        L = self.local.get(g)
        if not L:
            raise web.HTTPNotFound()
        if g not in self._mf:
            ver = self._lver
            doc = {"node": self.ident.id, "grid": L["grid"], "gdocs": L["gdocs"], "arrays": L["arrays"],
                   "chunks": dict(L["chunks"])}

            def build():  # ~0.4 s for 1.5e5 chunks: keep it off the event loop
                body = cjson(doc)
                sig = self.ident.sign(body)
                return zlib.compress(body, 3), sig, sig[:32]
            # ponytail: whole manifest per change (~45 B/chunk compressed); ETag makes unchanged ones free
            built = await asyncio.to_thread(build)
            if self._lver != ver:  # index changed during the build: serve it once, don't cache a stale one
                comp, sig, etag = built
            else:
                self._mf[g] = built
        comp, sig, etag = self._mf.get(g) or built
        self.served["legacy_manifest"] = self.served.get("legacy_manifest", 0) + 1
        if req.headers.get("If-None-Match") == etag:
            return web.Response(status=304, headers={"ETag": etag})
        return web.Response(body=comp, content_type="application/octet-stream",
                            headers={"X-Zt-Pk": self.ident.pk, "X-Zt-Sig": sig, "ETag": etag})

    async def _capt_for(self, g: str):
        """Signed manifest head + CAPT pages for grid g (rebuilt lazily when the local index changes)."""
        L = self.local.get(g)
        if not L:
            return None
        hit = self._capt.get(g)
        if hit and hit[0] == self._lver:
            return hit
        # single-flight + rate limit: an index that changes on every downloaded chunk must not trigger one
        # O(N) tree build per incoming head request (that was 20+ concurrent builds under load)
        if hit and time.monotonic() - self._capt_t.get(g, 0) < CAPT_MIN_INTERVAL:
            return hit
        inflight = self._capt_building.get(g)
        if inflight:
            return await asyncio.shield(inflight)
        fut = asyncio.get_running_loop().create_future()
        self._capt_building[g] = fut
        try:
            built = await self._capt_build(g, L)
            fut.set_result(built)
            return built
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            self._capt_building.pop(g, None)

    async def _capt_build(self, g: str, L: dict):
        ver, chunks = self._lver, dict(L["chunks"])

        def build():
            root, pages = capt.build(chunks) if chunks else ("", {})
            # versioned, hash-linked heads (a light transparency log): seq grows with every new root (ms clock, so a
            # node that lost its state still moves forward), prev links to the previous root. Clients refuse an
            # older seq (rollback) and keep two signed heads with one seq and different roots as proof of
            # equivocation (a publisher showing different data to different clients)
            last = self._heads.get(g, {"seq": 0, "root": None})
            seq, prev = (last["seq"], last.get("prev")) if root == last["root"] else \
                (max(last["seq"] + 1, int(time.time() * 1000)), last["root"])
            self._heads[g] = {"seq": seq, "root": root, "prev": prev}
            head = cjson({"node": self.ident.id, "grid": L["grid"], "gdocs": L["gdocs"], "arrays": L["arrays"],
                          "root": root, "n": len(chunks), "seq": seq, "prev": prev})
            return ver, zlib.compress(head, 6), self.ident.sign(head), root, pages
        built = await asyncio.to_thread(build)
        self._save_heads()
        prev = self._capt.get(g)
        if prev:  # keep the previous version's pages: clients may still be walking the old root
            built[4].update({h: b for h, b in prev[4].items() if h not in built[4]} if len(prev[4]) < 50_000 else {})
        self._capt[g] = built
        self._capt_t[g] = time.monotonic()
        return built

    async def h_mhead(self, req):
        c = await self._capt_for(req.match_info["grid"])
        if not c:
            raise web.HTTPNotFound()
        _, head, sig, root, _ = c
        if req.headers.get("If-None-Match") == root:
            return web.Response(status=304, headers={"ETag": root})
        self.served["manifest_bytes"] = self.served.get("manifest_bytes", 0) + len(head)
        return web.Response(body=head, content_type="application/octet-stream",
                            headers={"X-Zt-Pk": self.ident.pk, "X-Zt-Sig": sig, "ETag": root})

    async def h_mpage(self, req):
        c = self._capt.get(req.match_info["grid"])
        b = c[4].get(req.match_info["h"]) if c else None
        if b is None:
            raise web.HTTPNotFound()
        self.served["manifest_bytes"] = self.served.get("manifest_bytes", 0) + len(b)
        return web.Response(body=b, content_type="application/octet-stream")

    async def _capt_load(self, addr: str, grid: str, root: str, ranges: list | None = None) -> dict[str, list]:
        """Walk a peer's CAPT level by level, fetching only pages we have not cached (hash-verified).
        ranges: [(lo, hi)] sort-key intervals -> only subtrees that can overlap them are visited."""
        def hit(a, b):  # child key interval [a, b) overlaps some requested range?
            return ranges is None or any(a <= hi_ and (b is None or b > lo_) for lo_, hi_ in ranges)
        out: dict[str, list] = {}
        frontier = [root] if root else []
        while frontier:
            need = [h for h in dict.fromkeys(frontier) if h not in self.page_cache]

            async def get(h):
                async with self.session.get(f"{addr}/mp/{grid}/{h}", timeout=T_META) as r:
                    if r.status != 200:
                        raise aiohttp.ClientError(f"page {h}: {r.status}")
                    b = await r.read()
                if hashlib.blake2b(b, digest_size=16).hexdigest() != h:
                    raise ValueError("page hash mismatch")
                self.page_cache[h] = b
                self._page_bytes += len(b)
                self.page_fetched += len(b)
            await asyncio.gather(*(get(h) for h in need))
            while self._page_bytes > PAGE_CACHE_BYTES and len(self.page_cache) > 1:
                _, old = self.page_cache.popitem(last=False)
                self._page_bytes -= len(old)
            nxt = []
            for h in frontier:
                page = json.loads(zlib.decompress(self.page_cache[h]))
                items = page["items"]
                if page["leaf"]:
                    out.update({k: v for k, v in items if hit(capt.sort_key(k), None) and
                                (ranges is None or any(lo_ <= capt.sort_key(k) <= hi_ for lo_, hi_ in ranges))})
                    continue
                for i, (first, child) in enumerate(items):
                    b = capt.sort_key(items[i + 1][0]) if i + 1 < len(items) else None
                    if hit(capt.sort_key(first), b):
                        nxt.append(child)
            frontier = nxt
        return out

    async def h_chunks(self, req):
        try:
            return await self._h_chunks(req)
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            return web.Response(status=499)  # requester went away (endgame cancel): stop streaming quietly

    async def _h_chunks(self, req):
        g = req.match_info["grid"]
        L = self.local.get(g)
        keys = (await req.json())["keys"][:BATCH_CHUNKS * 4]
        if not all(isinstance(k, str) for k in keys):  # validate before the stream starts (no 500 mid-body)
            raise web.HTTPBadRequest(text="keys: list of strings")
        xt = "xt1" in req.headers.get("X-Zt-Accept", "")
        resp = web.StreamResponse(headers={"Content-Type": "application/octet-stream"})
        if self.latency:
            await asyncio.sleep(self.latency)
        await resp.prepare(req)
        for k in keys:
            f = L["files"].get(k) if L else None
            data = await asyncio.to_thread(Path(f).read_bytes) if f and os.path.exists(f) else None
            if data is not None:
                self.used[(g, k)] = time.time()  # peers want it: keep it cached
            if data is not None and xt and L:
                try:  # value-level transport codec: fewer bytes on slow links, verified by vcid on arrival
                    n, lay, _ = split_key(k)
                    vals = await asyncio.to_thread(self.decoded.get, k, L["arrays"][n]["layouts"][lay]["docs"],
                                                   lambda d=data: d)
                    alt = await asyncio.to_thread(codec.xt1_encode, vals)
                    if len(alt) < len(data):
                        data = alt
                        self.served["xt1_chunks"] = self.served.get("xt1_chunks", 0) + 1
                except Exception:
                    pass
            if data is None:
                await resp.write(struct.pack(">I", MISSING))
            else:
                await self._pace(len(data))
                self.served["chunks"] += 1
                await resp.write(struct.pack(">I", len(data)) + data)
        await resp.write_eof()
        return resp

    async def h_pushdown(self, req):
        try:
            return await self._h_pushdown(req)
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            return web.Response(status=499)

    async def _h_pushdown(self, req):
        """Holder side of optimistic pushdown: cut hyperslabs out of local chunks, sign one receipt per item."""
        g = req.match_info["grid"]
        L = self.local.get(g)
        items = (await req.json())["items"][:pd.MAX_ITEMS]
        if not all(isinstance(it, dict) and isinstance(it.get("key"), str) for it in items):
            raise web.HTTPBadRequest(text="items: list of {key, sel}")
        if self.latency:
            await asyncio.sleep(self.latency)
        resp = web.StreamResponse(headers={"Content-Type": "application/octet-stream", "X-Zt-Pk": self.ident.pk})
        await resp.prepare(req)
        for it in items:
            k, sel = it["key"], it["sel"]
            f = L["files"].get(k) if L else None
            if not f or not os.path.exists(f) or not isinstance(sel, list):
                await resp.write(struct.pack(">I", MISSING))
                continue
            n, lay, _ = split_key(k)
            docs = L["arrays"][n]["layouts"][lay]["docs"]
            cid = L["chunks"][k][0]
            try:
                vals = await asyncio.to_thread(self.decoded.get, k, docs, lambda f=f: Path(f).read_bytes())
                out = pd.cut(vals, sel)
                if self.cheat:
                    out = bytes(len(out))  # a lie with a perfectly valid signature
            except Exception:
                await resp.write(struct.pack(">I", MISSING))
                continue
            sig = self.ident.sign(pd.receipt_body(g, k, cid, sel, out))
            hdr = json.dumps({"cid": cid, "sig": sig}).encode()
            await self._pace(len(out))
            await resp.write(struct.pack(">II", len(out), len(hdr)) + hdr + out)
        await resp.write_eof()
        return resp

    async def slices(self, grid: str, items: list[dict]) -> list[bytes | None]:
        """Coalesce all slices of one chunk into their bounding box (one signed hyperslab per chunk), then cut
        the requested pieces locally. Thousands of per-pixel requests become one request per chunk."""
        groups: dict[str, list[int]] = {}
        for i, it in enumerate(items):
            groups.setdefault(it["key"], []).append(i)
        boxes = []
        for k, idxs in groups.items():
            sels = [items[i]["sel"] for i in idxs]
            boxes.append({"key": k, "sel": [[min(s_[d][0] for s_ in sels), max(s_[d][1] for s_ in sels)]
                                            for d in range(len(sels[0]))]})
        got = await self._slices_core(grid, boxes)
        v = await self.view(grid)
        out: list[bytes | None] = [None] * len(items)
        for box, res in zip(boxes, got):
            if res is None:
                continue
            n, lay, _ = split_key(box["key"])
            dt = _dtype_of(v["arrays"][n]["layouts"][lay]["docs"])
            shape = [hi - lo for lo, hi in box["sel"]]
            arr = np.frombuffer(res, dtype=dt).reshape(shape)
            for i in groups[box["key"]]:
                local = [[lo - blo, hi - blo] for (lo, hi), (blo, _) in zip(items[i]["sel"], box["sel"])]
                out[i] = pd.cut(arr, local)
        return out

    async def _slices_core(self, grid: str, items: list[dict]) -> list[bytes | None]:
        """Requester side: optimistic pushdown with signed receipts and random audits (see pushdown.py)."""
        v = await self.view(grid)
        L = self.local.get(grid, {"files": {}})
        out: list[bytes | None] = [None] * len(items)
        by_peer: dict[str, list[int]] = {}
        # demand aggregation: if the slices asked from one chunk add up to a large part of it, move the whole
        # (vcid-verified) chunk once instead of many signed slices
        frac: dict[str, float] = {}
        for it in items:
            n, lay, _ = split_key(it["key"])
            cs = v["arrays"].get(n, {}).get("layouts", {}).get(lay, {}).get("chunks") or []
            if cs:
                frac[it["key"]] = frac.get(it["key"], 0.0) + float(np.prod([(hi - lo) / c for (lo, hi), c in zip(it["sel"], cs)]))
        whole = [k for k, f_ in frac.items() if f_ > PD_WHOLE_FRAC and not L["files"].get(k)]
        if whole:
            await self.fetch(grid, whole)
            L = self.local.get(grid, {"files": {}})
        for i, it in enumerate(items):
            k, sel = it["key"], it["sel"]
            self.pd_stats["items"] += 1
            f = L["files"].get(k)
            if f and os.path.exists(f):  # we hold the chunk: cut locally
                n, lay, _ = split_key(k)
                vals = await asyncio.to_thread(self.decoded.get, k, L["arrays"][n]["layouts"][lay]["docs"],
                                               lambda f=f: Path(f).read_bytes())
                out[i] = pd.cut(vals, sel)
                self.pd_stats["local"] += 1
                continue
            b = v["best"].get(k)
            cands = [p for p, _, _ in (b["src"] if b else []) if p in v["addrs"] and p not in self.bad]
            if not cands:
                continue
            # prefer trusted holders, then reputation, then measured bandwidth
            p = max(cands, key=lambda q: (q in self.trusted, -self.rep.get(q, [0, 0])[1], self.bw.get(q, 0)))
            by_peer.setdefault(p, []).append(i)

        async def ask(p, idxs):
            try:
                req = {"items": [{"key": items[i]["key"], "sel": items[i]["sel"]} for i in idxs]}
                async with self.session.post(f"{v['addrs'][p]}/qb/{grid}", json=req, timeout=T_DATA) as r:
                    pk = r.headers.get("X-Zt-Pk", "")
                    body = await r.read()
                if h160(bytes.fromhex(pk)) != p:
                    return
                off = 0
                audit_items: dict[str, list] = {}
                for i in idxs:
                    n = struct.unpack(">I", body[off:off + 4])[0]
                    if n == MISSING:
                        off += 4
                        continue
                    n, h = struct.unpack(">II", body[off:off + 8])
                    rc = json.loads(body[off + 8:off + 8 + h])
                    res = body[off + 8 + h:off + 8 + h + n]
                    off += 8 + h + n
                    k, sel = items[i]["key"], items[i]["sel"]
                    want_cid = next((c for q, c, _ in v["best"][k]["src"] if q == p), None)
                    if rc.get("cid") != want_cid or not pd.check_receipt(pk, rc["sig"], grid, k, want_cid, sel, res):
                        continue
                    out[i] = res
                    self.pd_stats["remote"] += 1
                    self.pd_stats["bytes"] += len(res)
                    audit_items.setdefault(k, []).append((sel, rc, res, i))
                    self.accepted.setdefault(p, []).append({"grid": grid, "key": k, "sel": sel, "sig": rc["sig"]})
                    del self.accepted[p][:-10000]
                # chunk-granular audit, decided once per (peer, chunk): one full fetch verifies all its slices
                for k, its in audit_items.items():
                    dec = self.audit_decided.get((p, k))
                    if dec is None:
                        started = self.audit_started.get(p, 0)
                        dec = p not in self.trusted and (started < PD_MIN_AUDITS or random.random() < self.audit_rate)
                        self.audit_decided[(p, k)] = dec
                        if dec:
                            self.audit_started[p] = started + 1
                    if dec:
                        for sel, rc, res, i in its:
                            await self._audit(grid, p, pk, k, sel, rc, res, out, i)
                    if p in self.bad:
                        break
                if p in self.bad:  # rollback: nothing this cheater sent in this batch is kept
                    for i in idxs:
                        out[i] = None
            except Exception as e:
                print(f"[zt] pushdown via {p[:8]} failed: {type(e).__name__} {e}")

        calls = [(p, idxs[i:i + pd.MAX_ITEMS]) for p, idxs in by_peer.items() for i in range(0, len(idxs), pd.MAX_ITEMS)]
        await asyncio.gather(*(ask(p, part) for p, part in calls))
        missing = [i for i, r in enumerate(out) if r is None and v["best"].get(items[i]["key"])]
        if missing:  # fallback: fetch whole (vcid-verified) chunks and cut locally
            self.pd_stats["fallback"] += len(missing)
            paths = await self.fetch(grid, list({items[i]["key"] for i in missing}))
            for i in missing:
                k = items[i]["key"]
                pth = paths.get(k)
                if pth and pth not in ("!", "retry"):
                    n, lay, _ = split_key(k)
                    vals = await asyncio.to_thread(self.decoded.get, k, v["arrays"][n]["layouts"][lay]["docs"],
                                                   lambda pth=pth: Path(pth).read_bytes())
                    out[i] = pd.cut(vals, items[i]["sel"])
        return out

    async def _audit(self, grid, p, pk, k, sel, rc, res, out, i):
        """Re-derive the slice from the full chunk (verified by majority vcid); a mismatch is fraud."""
        self.pd_stats["audits"] += 1
        paths = await self.fetch(grid, [k])
        pth = paths.get(k)
        if not pth or pth in ("!", "retry"):
            return
        n, lay, _ = split_key(k)
        v = await self.view(grid)
        vals = await asyncio.to_thread(self.decoded.get, k, v["arrays"][n]["layouts"][lay]["docs"],
                                       lambda: Path(pth).read_bytes())
        truth = pd.cut(vals, sel)
        r = self.rep.setdefault(p, [0, 0])
        if truth == res:
            r[0] += 1
            return
        r[1] += 1
        self.bad.add(p)
        self.tainted.extend(self.accepted.pop(p, []))  # earlier optimistic results from this peer: recompute
        self.fraud.append({"peer": p, "pk": pk, "grid": grid, "key": k, "sel": sel, "cid": rc["cid"],
                           "sig": rc["sig"], "claimed_h": hashlib.blake2b(res, digest_size=16).hexdigest(),
                           "true_h": hashlib.blake2b(truth, digest_size=16).hexdigest(), "ts": time.time()})
        out[i] = truth
        print(f"[zt] FRAUD: peer {p[:8]} signed a wrong slice of {k}; blacklisted (proof stored)")

    async def a_slices(self, req):
        d = await req.json()
        res = await self.slices(d["grid"], d["items"])
        body = b"".join(struct.pack(">I", MISSING) if r is None else struct.pack(">I", len(r)) + r for r in res)
        return web.Response(body=body, content_type="application/octet-stream")

    async def h_whoami(self, req):
        ip = (req.remote or "").strip("[]")  # never trust X-Forwarded-For on a public port
        return web.json_response({"ip": ip, "id": self.ident.id, "relay": self.relay_server,
                                  "bw": self.rate or self.up_estimate()})

    async def h_probe(self, req):
        """Connect-back check. Only the requester's own address can be probed (no SSRF): the caller
        supplies a port, the host is the observed source address."""
        try:
            port = int((await req.json())["port"])
            assert 1 <= port <= 65535
        except Exception:
            raise web.HTTPBadRequest()
        ip = (req.remote or "").strip("[]")
        host = f"[{ip}]" if ":" in ip else ip
        try:
            async with self.session.get(f"http://{host}:{port}/whoami", timeout=aiohttp.ClientTimeout(total=5),
                                        allow_redirects=False) as r:
                return web.json_response({"id": (await r.json()).get("id")})
        except Exception:
            return web.json_response({"id": None})

    async def h_pex(self, req):
        g = req.match_info["grid"]
        return web.json_response([r for r in self.dht.get_local(g) if r.get("t") == "peer"][:50])

    # ------------------------------------------------------------------ relay (server side)
    async def h_relay_attach(self, req):
        ws = web.WebSocketResponse(max_msg_size=RELAY_MAX_MSG, heartbeat=30)
        await ws.prepare(req)
        nonce = os.urandom(16).hex()
        await ws.send_json({"nonce": nonce})
        try:
            auth = await ws.receive_json(timeout=10)
            assert verify(auth["pk"], auth["sig"], nonce.encode())
        except Exception:
            await ws.close()
            return ws
        nid = h160(bytes.fromhex(auth["pk"]))
        if nid not in self.relayed and len(self.relayed) >= RELAY_MAX_PEERS:
            await ws.close(message=b"relay full")
            return ws
        self.relayed[nid] = (ws, asyncio.Lock())
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    hdr, body = _unframe(msg.data)
                    rid = hdr.get("rid")
                    if hdr.get("d") or hdr.get("end"):  # streamed body frames (cut-through relaying)
                        q = self.relay_streams.get(rid)
                        if q is not None:
                            q.put_nowait(body if hdr.get("d") else (ConnectionError("stream error")
                                                                  if hdr.get("err") else None))
                        continue
                    fut = self.relay_waits.pop(rid, None)
                    if fut and not fut.done():
                        fut.set_result((hdr, body))
        finally:
            if self.relayed.get(nid, (None,))[0] is ws:
                del self.relayed[nid]
            for rid in [r for r, o in self.relay_owner.items() if o is ws]:  # fail fast; only this socket's
                fut = self.relay_waits.pop(rid, None)
                self.relay_owner.pop(rid, None)
                if fut and not fut.done():
                    fut.set_exception(ConnectionError(f"relayed peer {nid[:8]} disconnected"))
                q = self.relay_streams.get(rid)
                if q is not None:
                    q.put_nowait(ConnectionError("relayed peer disconnected"))
        return ws

    async def h_relay_forward(self, req):
        try:
            return await self._h_relay_forward(req)
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            return web.Response(status=499)  # the requester went away mid-stream

    async def _h_relay_forward(self, req):
        nid, tail = req.match_info["nid"], req.match_info["tail"]
        if nid not in self.relayed:
            raise web.HTTPNotFound(text="peer not attached")
        if not tail.startswith(RELAY_PATHS):
            raise web.HTTPForbidden()
        ws, lock = self.relayed[nid]
        inflight = sum(1 for o in self.relay_owner.values() if o is ws)
        if inflight >= RELAY_MAX_INFLIGHT_PER_PEER or len(self.relay_owner) >= RELAY_MAX_INFLIGHT:
            raise web.HTTPServiceUnavailable(text="relay busy")
        rid = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self.relay_waits[rid] = fut
        self.relay_owner[rid] = ws
        self.relay_streams[rid] = asyncio.Queue()
        path = "/" + tail + (f"?{req.query_string}" if req.query_string else "")
        try:
            async with lock:
                await ws.send_bytes(_frame({"rid": rid, "method": req.method, "path": path, "stream": 1},
                                           await req.read()))
            try:
                hdr, body = await asyncio.wait_for(fut, 120)
            except (ConnectionError, asyncio.TimeoutError) as e:
                raise web.HTTPBadGateway(text=str(e))
            if not hdr.get("stream"):  # legacy peer: whole body in one frame
                await self._pace(len(body), "relayed_bytes")
                return web.Response(status=hdr["status"], body=body, content_type=hdr.get("ctype") or None,
                                    headers=hdr.get("headers") or {})
            # cut-through: forward body frames as they arrive (both legs of the path work in parallel)
            resp = web.StreamResponse(status=hdr["status"], headers={**(hdr.get("headers") or {}),
                                                                       "Content-Type": hdr.get("ctype") or
                                                                       "application/octet-stream"})
            await resp.prepare(req)
            q = self.relay_streams[rid]
            while True:
                part = await asyncio.wait_for(q.get(), 120)
                if part is None:
                    break
                if isinstance(part, Exception):
                    raise part  # truncated response: the requester sees an error and retries elsewhere
                await self._pace(len(part), "relayed_bytes")
                await resp.write(part)
            await resp.write_eof()
            return resp
        finally:
            self.relay_waits.pop(rid, None)
            self.relay_owner.pop(rid, None)
            self.relay_streams.pop(rid, None)

    # ------------------------------------------------------------------ relay (client side, NAT'd node)
    async def _relay_client(self):
        while True:
            try:
                async with self.session.ws_connect(self.relay.rstrip("/") + "/relay/attach",
                                                   max_msg_size=256 << 20, heartbeat=30) as ws:
                    nonce = (await ws.receive_json())["nonce"]
                    await ws.send_json({"pk": self.ident.pk, "sig": self.ident.sign(nonce.encode())})
                    self.relay_up, lock = True, asyncio.Lock()
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            asyncio.create_task(self._relay_serve(ws, lock, msg.data))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[zt] relay connection lost: {e}")
            self.relay_up = False
            await asyncio.sleep(3)

    async def _relay_serve(self, ws, lock, data):
        hdr, body = _unframe(data)
        if hdr.get("stream") and hdr["path"].lstrip("/").startswith(RELAY_PATHS):
            return await self._relay_serve_stream(ws, lock, hdr, body)
        out = {"rid": hdr["rid"], "status": 403}
        rb = b""
        if hdr["path"].lstrip("/").startswith(RELAY_PATHS):
            try:
                async with self.session.request(hdr["method"], f"http://127.0.0.1:{self.port}{hdr['path']}",
                                                data=body) as r:
                    rb = await r.read()
                    out.update(status=r.status, ctype=r.content_type,
                               headers={k: v for k, v in r.headers.items() if k.startswith("X-Zt-")})
            except Exception:
                out["status"] = 502
        try:
            async with lock:
                await ws.send_bytes(_frame(out, rb))
        except (ConnectionError, aiohttp.ClientError, RuntimeError):
            pass  # relay link dropped while we were answering; the requester times out and retries

    async def _relay_serve_stream(self, ws, lock, hdr, body):
        rid, started = hdr["rid"], False

        async def send(h, b=b""):
            async with lock:
                await ws.send_bytes(_frame(h, b))
        try:
            async with self.session.request(hdr["method"], f"http://127.0.0.1:{self.port}{hdr['path']}",
                                            data=body) as r:
                await send({"rid": rid, "status": r.status, "ctype": r.content_type, "stream": 1,
                            "headers": {k: v for k, v in r.headers.items() if k.startswith("X-Zt-")}})
                started = True
                # read our own local response fully (cheap, loopback), then ship it over the WAN leg in large
                # frames: the relay forwards frame by frame (pipelined), without many tiny frames that stalled
                # behind the tunnel's flow control in the real WAN test
                whole = await r.read()
                for off in range(0, len(whole), RELAY_FRAME):
                    await send({"rid": rid, "d": 1}, whole[off:off + RELAY_FRAME])
                await send({"rid": rid, "end": 1})
        except (ConnectionError, aiohttp.ClientError, RuntimeError, asyncio.TimeoutError):
            try:
                await send({"rid": rid, "end": 1, "err": 1} if started else {"rid": rid, "status": 502})
            except Exception:
                pass

    # ------------------------------------------------------------------ remote views
    def _head_ok(self, nid: str, grid: str, m: dict, pk: str, sig: str, body: bytes) -> bool:
        """Rollback and equivocation check on a verified manifest head (see _capt_build)."""
        seq = m.get("seq")
        if not isinstance(seq, int):
            return True  # a peer from before versioned heads
        seen = self.seen_heads.get((nid, grid))
        if seen and seq < seen["seq"]:
            return False  # replay of an older version
        if seen and seq == seen["seq"] and m["root"] != seen["root"]:
            self.bad.add(nid)  # two signed heads, one version, two contents: equivocation (a transferable proof)
            self.fraud.append({"peer": nid, "pk": pk, "grid": grid, "kind": "equivocation", "seq": seq,
                               "heads": [seen["body"], body.decode(errors="replace")], "sigs": [seen["sig"], sig]})
            return False
        self.seen_heads[(nid, grid)] = {"seq": seq, "root": m["root"], "body": body.decode(errors="replace"), "sig": sig}
        return True

    async def _manifest(self, nid: str, addr: str, grid: str, refresh: bool = False) -> dict | None:
        if nid in self.bad:
            return None  # caught cheating: out of every view
        hit = self.peer_manifests.get((nid, grid))
        if hit and not refresh and time.time() - hit[0] < VIEW_TTL:
            return hit[1]
        m = await self._manifest_capt(nid, addr, grid, hit)
        if m is not None:
            return m
        try:
            hdrs = {"If-None-Match": hit[2]} if hit else {}
            for attempt in range(META_TRIES):
                try:
                    async with self.session.get(f"{addr}/m/{grid}", headers=hdrs, timeout=T_META) as r:
                        if r.status == 304 and hit:  # unchanged: no transfer, no re-verification
                            self.peer_manifests[(nid, grid)] = (time.time(), hit[1], hit[2])
                            return hit[1]
                        if r.status != 200:
                            return None
                        comp, pk, sig = await r.read(), r.headers.get("X-Zt-Pk", ""), r.headers.get("X-Zt-Sig", "")
                        etag = r.headers.get("ETag", "")
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt == META_TRIES - 1:
                        raise

            def check():
                d = zlib.decompressobj()
                body = d.decompress(comp, MAX_MANIFEST)
                if d.unconsumed_tail:
                    return None  # zip bomb / oversized manifest
                if h160(bytes.fromhex(pk)) != nid or not verify(pk, sig, body):
                    return None
                m = json.loads(body)
                return m if grid_id_of(m["grid"]) == grid else None
            m = await asyncio.to_thread(check)
            if m is None:
                return None
            m["_etag"] = etag
        except Exception:
            return None
        self.peer_manifests[(nid, grid)] = (time.time(), m, etag)
        return m

    async def _manifest_capt(self, nid, addr, grid, hit) -> dict | None:
        """Head (signed, small) + only the CAPT pages we have not seen. None -> caller falls back to /m."""
        try:
            hdrs = {"If-None-Match": hit[2]} if hit else {}
            for attempt in range(META_TRIES):
                try:
                    async with self.session.get(f"{addr}/mh/{grid}", headers=hdrs, timeout=T_META) as r:
                        if r.status == 304 and hit:
                            self.peer_manifests[(nid, grid)] = (time.time(), hit[1], hit[2])
                            return hit[1]
                        if r.status != 200:
                            return None
                        comp, pk, sig, root = await r.read(), r.headers.get("X-Zt-Pk", ""), \
                            r.headers.get("X-Zt-Sig", ""), r.headers.get("ETag", "")
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt == META_TRIES - 1:
                        return None
            d = zlib.decompressobj()
            body = d.decompress(comp, MAX_MANIFEST)
            if d.unconsumed_tail or h160(bytes.fromhex(pk)) != nid or not verify(pk, sig, body):
                return None
            m = json.loads(body)
            if grid_id_of(m["grid"]) != grid or m.get("root", "") != root:
                return None
            if not self._head_ok(nid, grid, m, pk, sig, body):
                return None
            m["chunks"] = await self._capt_load(addr, grid, root)
            if len(m["chunks"]) != m.get("n"):
                return None
            m["_etag"] = root
        except Exception:
            return None
        self.peer_manifests[(nid, grid)] = (time.time(), m, root)
        return m

    async def view(self, grid: str, refresh: bool = False) -> dict:
        v = self.views.get(grid)
        if v and not refresh and time.time() - v["ts"] < VIEW_TTL:
            return v
        peers = await self.find_peers(grid)
        tasks = {asyncio.create_task(self._manifest(n, a, grid, refresh)): n for n, a in peers.items()}
        mans = {}
        if tasks:
            # don't let the slowest (stalled) peer dictate: plan with whoever answered within VIEW_WAIT;
            # stragglers keep running in the background and land in the manifest cache for the next round
            done, pending = await asyncio.wait(tasks, timeout=VIEW_WAIT)
            if not any(t.result() for t in done) and pending:
                more, pending = await asyncio.wait(pending, timeout=120, return_when=asyncio.FIRST_COMPLETED)
                done |= more
            mans = {tasks[t]: t.result() for t in done if t.result()}
        if grid in self.local:
            mans[self.ident.id] = dict(self.local[grid], node=self.ident.id, _etag=f"local{self._lver}")
        if not mans:
            raise KeyError(f"no reachable peers for {grid}")
        key = tuple(sorted((n, m.get("_etag", "")) for n, m in mans.items()))
        memo = self._view_memo.get(grid)
        if memo and memo[0] == key:  # no replica changed: skip the O(#chunks) merge
            v = dict(memo[1])
        else:
            v = await asyncio.to_thread(merge_view, grid, mans, self.ident.id, self.trusted)
            self._view_memo[grid] = (key, v)
            v = dict(v)
        v.update(ts=time.time(), addrs=peers)
        self.views[grid] = v
        return v

    @staticmethod
    def _fits_expect(L: dict, k: str, exp: list) -> bool:
        """Is our local file for k in the (vfid, layout fid) the reader will decode it with?"""
        n, lay, _ = split_key(k)
        La = L.get("arrays", {}).get(n)
        return bool(La) and La["vfid"] == exp[0] and La["layouts"].get(lay, {}).get("fid") == exp[1]

    @staticmethod
    def _fits(L: dict, v: dict, k: str) -> bool:
        """Is our local file for k stored in the encoding the view reads it with (same value family and layout
        encoding)? A seed or cache entry of another family must be transcoded, not passed through."""
        n, lay, _ = split_key(k)
        La, va = L.get("arrays", {}).get(n), v.get("arrays", {}).get(n)
        if not La or not va or lay not in va["layouts"]:
            return True  # nothing to compare against
        return La["vfid"] == va["vfid"] and La["layouts"].get(lay, {}).get("fid") == va["layouts"][lay].get("fid")

    # ------------------------------------------------------------------ download engine
    async def fetch(self, grid: str, keys: list[str], job: dict | None = None,
                    view: dict | None = None, expect: dict | None = None) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        L = self.local.get(grid, {"files": {}})
        now = time.time()
        for k in keys:  # the client is using these: most recently used for cache eviction
            self.used[(grid, k)] = now
        need = []
        v0 = view or self.views.get(grid)
        for k in dict.fromkeys(keys):
            f = L["files"].get(k)
            if f and os.path.exists(f) and (self._fits_expect(L, k, expect[k]) if expect and k in expect
                                            else v0 is None or self._fits(L, v0, k)):
                out[k] = f
            else:
                need.append(k)
        if not need:
            return out
        waits, mine = {}, []
        for k in need:
            fut = self.pending.get((grid, k))
            if fut:
                waits[k] = fut
            else:
                self.pending[(grid, k)] = asyncio.get_running_loop().create_future()
                mine.append(k)
        try:
            if mine:
                ph = job.setdefault("phases", {}) if job is not None else {}
                t_ = time.monotonic()
                v = view or await self.view(grid)
                ph["view"] = round(ph.get("view", 0) + time.monotonic() - t_, 2)
                t_ = time.monotonic()
                res = await self._download(grid, v, mine, job)
                ph["round0"] = round(ph.get("round0", 0) + time.monotonic() - t_, 2)
                # lossy links: retry what failed (peers failed in one round get a fresh chance), with backoff
                for rnd in range(1, FETCH_ROUNDS):
                    left = [k for k in mine if k not in res and k in v["best"]]
                    if not left or (job is not None and job.get("cancel")):
                        break
                    t_ = time.monotonic()
                    await asyncio.sleep(rnd)
                    v = view or await self.view(grid, refresh=True)
                    res.update(await self._download(grid, v, left, job, count=False))
                    ph[f"round{rnd}"] = round(ph.get(f"round{rnd}", 0) + time.monotonic() - t_, 2)
                if job is not None:
                    job["failed"] = job.get("failed", 0) + sum(1 for k in mine if k not in res and k in v["best"])
                cancelled = bool(job and job.get("cancel"))
                for k in mine:  # "!" = replicas exist but every attempt failed (never silently fill)
                    r = res.get(k) or ("retry" if cancelled and k in v["best"] else "!" if k in v["best"] else None)
                    self.pending.pop((grid, k)).set_result(r)
        except Exception as e:
            for k in mine:
                fut = self.pending.pop((grid, k), None)
                if fut and not fut.done():
                    fut.set_exception(e)
            raise
        for k in mine:
            lf = self.local.get(grid, {"files": {}})["files"].get(k)
            if lf and expect and k in expect and not self._fits_expect(self.local.get(grid, {}), k, expect[k]):
                lf = None  # a local file of another family is no answer for this reader
            out[k] = res.get(k) or lf or \
                ("!" if k in v["best"] and not (job and job.get("cancel")) else None)
        retry = []
        for k, fut in waits.items():
            try:
                out[k] = await fut
            except Exception:
                out[k] = "!"  # the owner failed: report, never fill silently
            if out[k] == "retry":  # the owner was cancelled before fetching it: fetch it ourselves
                retry.append(k)
        if retry:
            out.update(await self.fetch(grid, retry))
        return out

    async def _probe_unknown(self, grid: str, v: dict, keys: list[str]) -> int:
        """Measure the holders of `keys` that have no rate yet (one small chunk each). Returns how many."""
        me = self.ident.id
        src = {}
        for k in keys:
            b = v["best"].get(k)
            s = {p: (cid, size) for p, cid, size in (b["src"] if b else []) if p != me and p in v["addrs"]}
            if s:
                src[k] = s
        unknown = sorted({p for s_ in src.values() for p in s_ if p not in self.bw and not self.hints.get(p)})
        if len(unknown) < 2 or len(src) < 3 * len(unknown):
            return 0
        probe_map: dict[str, str] = {}
        for p in unknown:
            cands = [(s_[p][1], k) for k, s_ in src.items() if p in s_ and k not in probe_map]
            if cands:
                probe_map[min(cands)[1]] = p
        await self._probe(grid, v, probe_map, src, None)
        return len(probe_map)

    async def _probe(self, grid: str, v: dict, probe_map: dict, src: dict, job) -> dict:
        """One chunk per unknown peer, concurrently. Once half of them are in, the rest get up to three times that
        long (at least 1 s); a peer still sending is planned at the rate it has at most shown (size / elapsed)."""
        t0 = time.monotonic()
        tasks = {asyncio.create_task(self._download(grid, v, [k], job, count=False, only={k: p}, probe=False)): (k, p)
                 for k, p in probe_map.items()}
        pending, got = set(tasks), {}
        half = max(1, len(tasks) // 2)
        while pending and len(tasks) - len(pending) < half:
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if pending:
            _, pending = await asyncio.wait(pending, timeout=max(1.0, 3 * (time.monotonic() - t0)))
        for t in pending:
            k, p = tasks[t]
            self.bw[p] = src[k][p][1] / max(time.monotonic() - t0, 1e-3)  # an upper bound on what it can do
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t, (k, p) in tasks.items():
            if t not in pending and not t.cancelled() and t.exception() is None:
                got.update(t.result())
                if p not in self.bw:  # too quick to be timed inside the download: still a measurement
                    self.bw[p] = src[k][p][1] / max(time.monotonic() - t0, 1e-3)
        return got

    async def _download(self, grid: str, v: dict, keys: list[str], job: dict | None, count: bool = True,
                        only: dict | None = None, probe: bool = True) -> dict:
        me = self.ident.id
        restorable = [k for k in keys if v["best"].get(k, {}).get("restore") and not any(
            p != me and p in v["addrs"] for p, _, _ in v["best"][k]["src"])]
        restored = {}
        for k in restorable:
            path = await self._restore(grid, v, k)
            if path:
                restored[k] = path
                if job is not None:
                    job["done"] += 1
                    job["done_keys"].append(k)
                    job["restored"] = job.get("restored", 0) + 1
        keys = [k for k in keys if k not in restorable]
        if not keys:
            if job is not None and count:
                job["total"] = job.get("total", 0) + len(restorable)
                job["missing"] = job.get("missing", 0) + len(restorable) - len(restored)
            return restored
        src = {}  # key -> {peer: (cid, size)}
        for k in keys:
            b = v["best"].get(k)
            if b:
                s = {p: (cid, size) for p, cid, size in b["src"] if p != me and p in v["addrs"]}
                if s:
                    src[k] = s
        if job is not None and count:
            job["total"] = job.get("total", 0) + len(keys) + len(restorable)
            job["missing"] = job.get("missing", 0) + len(keys) - len(src) + len(restorable) - len(restored)
        if only:  # probe round: each key from the one peer it was given to
            src = {k: {only[k]: s[only[k]]} for k, s in src.items() if k in only and only[k] in s}
        if not src:
            return restored
        if probe and not only and self.strategy == "maxflow":
            # Peers we know nothing about (no measured rate, no announced hint) would be planned at a default rate;
            # on real paths that made the swarm slower than its best single source (a slow site planned like a fast
            # one). Measure them on real work first: each gets one small chunk of this very request, then the rest
            # is planned with the measured rates (as aria2 ranks mirrors, or BitTorrent probes by unchoking).
            unknown = sorted({p for s in src.values() for p in s if p not in self.bw and not self.hints.get(p)})
            if len(unknown) >= 2 and len(src) >= 3 * len(unknown):
                probe_map: dict[str, str] = {}
                for p in unknown:
                    k = min((s[p][1], k) for k, s in src.items() if p in s and k not in probe_map)[1] \
                        if any(p in s and k not in probe_map for k, s in src.items()) else None
                    if k:
                        probe_map[k] = p
                got = await self._probe(grid, v, probe_map, src, job)
                restored.update(got)
                src = {k: s for k, s in src.items() if k not in got}
                if job is not None:
                    job["probed"] = len(probe_map)
                if not src:
                    return restored
        peers = {p for s in src.values() for p in s}
        bw, via, via_bw = await self._bw_model(peers, v)
        steal = self.strategy == "maxflow"
        est = 0.0
        if self.strategy == "maxflow":
            assign, est = planmod.plan({k: (max(sz for _, sz in s.values()), tuple(s)) for k, s in src.items()},
                                       bw, via=via, via_bw=via_bw, client_bw=jlpsmod.CLIENT_BW or None)
        elif self.strategy == "random":  # uniform random replica per chunk, no rebalancing
            rng = random.Random(len(src))
            assign = {}
            for k, s in src.items():
                assign.setdefault(rng.choice(sorted(s)), []).append(k)
        elif self.strategy == "single":  # classic single-source: the best-covering peer, others only for its gaps
            cover = {p: sum(p in s for s in src.values()) for p in peers}
            top = max(sorted(cover), key=cover.get)
            rng = random.Random(0)
            assign = {}
            for k, s in src.items():
                assign.setdefault(top if top in s else rng.choice(sorted(s)), []).append(k)
        else:  # "rarest": BitTorrent-style rarest-first pull, every peer pulls the rarest chunk it holds
            assign = {}
        rare = deque(sorted(src, key=lambda k: (len(src[k]), k))) if self.strategy == "rarest" else None
        if job is not None:
            job["plan_T"] = round(est, 2)
            # per-peer prediction for model diagnostics: planned bytes and the bandwidth the plan assumed
            job["pred"] = {p: {"bytes": sum(src[k][p][1] for k in ks), "bw": round(bw[p]),
                               "via": via.get(p) if self.strategy == "maxflow" else None}
                           for p, ks in assign.items()}
            t_plan0 = time.monotonic()
        if job is not None and job.get("order") == "progressive":
            assign = {p: progressive_order(ks) for p, ks in assign.items()}
        queues = {p: deque(assign.get(p, [])) for p in peers}
        done, inflight, failed, results, span = set(), {}, set(), {}, {}
        tries: dict[tuple[str, str], int] = {}
        perr: dict[str, int] = {}
        via_load: dict[str, int] = {}
        peer_load: dict[str, int] = {}
        peer_bytes: dict[str, float] = {}  # bytes in flight per peer: its uplink is busy with them
        via_bytes: dict[str, float] = {}  # bytes in flight per relay: its one uplink carries them
        eta: dict[str, float] = {}  # expected arrival of each in-flight chunk (endgame picks the latest)
        t_disp: dict[str, float] = {}  # when it was first requested
        all_done = asyncio.Event()

        def take(p):
            if job is not None and job.get("cancel"):
                return []
            if rare is not None:
                batch, skipped = [], []
                while rare and len(batch) < BATCH_CHUNKS // 4:
                    k = rare.popleft()
                    if k in done:
                        continue
                    (batch if p in src[k] else skipped).append(k)
                rare.extendleft(reversed(skipped))
                return batch
            q, batch, nbytes = queues[p], [], 0
            # ~BATCH_SECONDS of work per request: small on slow peers so stealing/endgame can act on the tail
            cap = min(BATCH_BYTES, max(256 << 10, bw[p] * BATCH_SECONDS / CONC_PER_PEER))
            while q and len(batch) < BATCH_CHUNKS:
                k = q[0]
                if k in done or p not in src[k]:
                    q.popleft()
                    continue
                if batch and nbytes + src[k][p][1] > cap:
                    break
                q.popleft()
                batch.append(k)
                nbytes += src[k][p][1]
            if batch or not steal:
                return batch
            # behind a relay, stolen/duplicated work queues on the relay's shared uplink too: the thief's finish
            # time counts the relay's bytes in flight (a blanket "relay busy -> no stealing" rule disabled straggler
            # mitigation whenever every remote peer sat behind one fast relay: 10.6 s vs 2.1 s on a real WAN run;
            # without any relay term, 7 MB of duplicates on a 1 MB/s relay turned a 13 s plan into 21 s)
            def t_of(extra):
                t = (peer_bytes.get(p, 0.0) + extra) / bw[p]
                r_ = via.get(p)
                return max(t, (via_bytes.get(r_, 0.0) + extra) / via_bw[r_]) if r_ in via_bw else t
            # work stealing: from the peer with the largest estimated remaining time
            others = sorted((o for o in queues if o != p and queues[o]),
                            key=lambda o: -sum(src[k][o][1] for k in queues[o]) / bw[o])
            for o in others:
                # take from the victim's tail only what the thief finishes before the victim would reach it
                # (a slow peer stealing a big chunk became the straggler: 0.38 MB/s thief, 2.3 MB map chunks)
                left = (sum(src[k][o][1] for k in queues[o]) + peer_bytes.get(o, 0.0)) / bw[o]
                stolen, mine = [], 0.0  # t_of counts the thief's own batches (and its relay's) still in flight
                for k in reversed(list(queues[o])[-BATCH_CHUNKS:]):
                    if p not in src[k] or k in done:
                        continue
                    t_thief = t_of(mine + src[k][p][1])
                    if t_thief >= left:
                        break
                    stolen.append(k)
                    mine += src[k][p][1]
                    left -= src[k][o][1] / bw[o]
                for k in stolen:
                    queues[o].remove(k)
                if stolen:
                    return stolen
            # endgame: duplicate the in-flight chunk expected LAST, if this peer would clearly deliver it sooner
            # (any in-flight chunk used to be duplicated: the straggler - a 0.2 MB/s peer on a 2.3 MB map chunk -
            # still set the finish time, 2x the plan on short queries)
            # An overdue chunk (its ETA passed: the holder is slower than we assumed, e.g. a first-contact peer
            # planned at the default rate) is expected to need as long again as it has taken so far.
            now = time.monotonic()
            left = lambda k: max(eta.get(k, now) - now, now - t_disp.get(k, now))
            for k in sorted((k for k, ps in inflight.items() if k not in done and p in src[k] and p not in ps
                             and len(ps) < 2), key=lambda k: -left(k)):
                if left(k) > 1.5 * t_of(src[k][p][1]):
                    return [k]
            return []

        async def worker(p):
            view_arrays = v["arrays"]
            while p not in failed:
                batch = take(p)
                if not batch:
                    if not inflight or (job is not None and job.get("cancel")):
                        return
                    await asyncio.sleep(0.05)  # others still in flight: failures may requeue work to us
                    continue
                for k in batch:
                    inflight.setdefault(k, set()).add(p)
                t0, got = time.monotonic(), 0
                span.setdefault(p, [t0, t0, 0])
                nb_ = 0.0
                try:
                    link_bw = min(bw[p], via_bw.get(via.get(p), float("inf")))  # a relay is part of the path
                    nb_ = sum(src[k][p][1] for k in batch if p in src[k])
                    peer_load[p] = peer_load.get(p, 0) + 1  # this peer's parallel batches share its uplink
                    peer_bytes[p] = peer_bytes.get(p, 0.0) + nb_
                    expect = peer_load[p] * nb_ / max(bw[p], 1e3)
                    r_ = via.get(p)
                    if r_ in via_bw:  # cut-through relay: legs overlap, but its one uplink is shared (queue)
                        via_load[r_] = via_load.get(r_, 0) + 1
                        via_bytes[r_] = via_bytes.get(r_, 0.0) + nb_
                        expect = max(expect, via_load[r_] * nb_ / max(via_bw[r_], 1e3))
                    acc = 0.0
                    for k in batch:  # a batch streams in order: chunk i arrives after chunks 0..i
                        acc += src[k][p][1] if p in src[k] else 0
                        eta[k] = min(eta.get(k, float("inf")), t0 + expect * acc / max(nb_, 1.0))
                        t_disp.setdefault(k, t0)
                    # a stalled connection is detected after ~6x the expected batch time, not after a fixed minute
                    # unmeasured peer (first contact): our bandwidth guess may be far off -> be patient until the
                    # first batch gives a real measurement, then switch to the tight adaptive stall detector
                    known = p in self.bw or p in self.hints or span.get(p, [0, 0, 0])[2]
                    floor = STALL_MIN if known else STALL_FIRST
                    tmo = aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=max(floor, 4 * expect))
                    hdrs = {"X-Zt-Accept": "xt1"} if self.xt1 and link_bw < XT1_BELOW else {}
                    async with self.session.post(f"{v['addrs'][p]}/cb/{grid}", json={"keys": batch},
                                                 timeout=tmo, headers=hdrs) as r:
                        if r.status != 200:
                            raise aiohttp.ClientError(f"status {r.status}")
                        body = None  # direct and (cut-through) relayed responses are both read incrementally
                        off = 0
                        for k in batch:
                            if body is None:
                                n = struct.unpack(">I", await r.content.readexactly(4))[0]
                                if n != MISSING and n != src[k][p][1] and not (hdrs and n <= src[k][p][1] + 4096):
                                    raise aiohttp.ClientError(f"bad length {n} for {k}")  # manifest size (or xt1)
                                data = None if n == MISSING else await r.content.readexactly(n)
                            else:
                                n = struct.unpack(">I", body[off:off + 4])[0]
                                if n != MISSING and n != src[k][p][1] and not (hdrs and n <= src[k][p][1] + 4096):
                                    raise aiohttp.ClientError(f"bad length {n} for {k}")
                                off += 4
                                data = None if n == MISSING else body[off:off + n]
                                off += 0 if n == MISSING else n
                            if data is None or k in done:
                                continue
                            got += len(data)
                            stored = await asyncio.to_thread(self._verify_store, v, k, p, data)
                            if stored and v["best"][k].get("rival") and not await self._rival_ok(v, grid, k, stored[0]):
                                v["best"][k].update(src=[], contested=2)  # completeness alone may not decide
                                src[k] = {}
                                stored = None
                            path = self._index(grid, v, k, *stored) if stored else None
                            if path and k not in done:
                                done.add(k)
                                results[k] = path
                                if len(done) >= len(src):
                                    all_done.set()
                                if job is not None:
                                    job["done"] += 1
                                    job["bytes"] += len(data)
                                    job["per_peer"][p] = job["per_peer"].get(p, 0) + len(data)
                                    job["done_keys"].append(k)
                except Exception as e:
                    perr[p] = perr.get(p, 0) + 1
                    if perr[p] >= PEER_MAX_ERRORS:  # transient WAN errors are retried; persistent ones drop the peer
                        failed.add(p)
                    print(f"[zt] peer {p[:8]} error {perr[p]}/{PEER_MAX_ERRORS}: {type(e).__name__} {e}")
                else:
                    perr[p] = 0
                finally:
                    peer_load[p] = peer_load.get(p, 1) - 1
                    peer_bytes[p] = max(0.0, peer_bytes.get(p, 0.0) - nb_)
                    if via.get(p) in via_load:
                        via_load[via[p]] -= 1
                        via_bytes[via[p]] = max(0.0, via_bytes.get(via[p], 0.0) - nb_)
                    for k in batch:
                        inflight.get(k, set()).discard(p)
                        if not inflight.get(k):
                            inflight.pop(k, None)
                            eta.pop(k, None)
                            t_disp.pop(k, None)
                sp = span[p]
                sp[1], sp[2] = time.monotonic(), sp[2] + got
                if sp[2] and sp[1] - sp[0] > 0.05:  # peer throughput = its bytes / its busy wall time
                    bw[p] = sp[2] / (sp[1] - sp[0])
                # anything this peer did not deliver goes back to the best alternative peer; a peer that failed
                # a key twice (MISSING / wrong bytes) is dropped as a holder of that key -> no infinite retry
                for k in batch:
                    if k not in done and not inflight.get(k):
                        tries[(k, p)] = tries.get((k, p), 0) + 1
                        if tries[(k, p)] >= 2 and p in src[k]:
                            src[k] = {o: x for o, x in src[k].items() if o != p}
                        alts = [o for o in src[k] if o not in failed and o != p] or \
                               ([p] if p not in failed and p in src[k] else [])
                        if alts:
                            # finish time on the alternative, this chunk included (an idle 0.01 MB/s peer has an
                            # empty queue too: by queue alone it won requeued chunks and took minutes)
                            best = min(alts, key=lambda o: (sum(src[x][o][1] for x in queues[o])
                                                            + peer_bytes.get(o, 0.0) + src[k][o][1]) / bw[o])
                            if rare is not None:
                                rare.appendleft(k)
                            elif k not in queues[best]:
                                queues[best].appendleft(k)

        tasks = [asyncio.create_task(worker(p)) for p in peers for _ in range(CONC_PER_PEER)]
        watcher = asyncio.create_task(all_done.wait())
        try:
            await asyncio.wait(tasks + [watcher], return_when=asyncio.FIRST_COMPLETED)
            # BitTorrent-style endgame CANCEL: once every chunk is in, drop the duplicate/straggler requests still in
            # flight instead of waiting for them (a slow relayed peer used to add seconds to finished downloads)
            while not all_done.is_set() and any(not t.done() for t in tasks):
                await asyncio.wait(tasks + [watcher], return_when=asyncio.FIRST_COMPLETED)
                if all(t.done() for t in tasks):
                    break
        finally:  # also when this download itself is cancelled (a probe past its deadline): no orphan workers
            for t in tasks + [watcher]:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, watcher, return_exceptions=True)
        results.update(restored)
        for p, (a0, a1, nb) in span.items():
            if nb and a1 - a0 > 0.05:
                r = nb / (a1 - a0)
                self.bw[p] = r if p not in self.bw else 0.5 * self.bw[p] + 0.5 * r
        if job is not None and "pred" in job:
            job["actual"] = {p: {"bytes": nb, "busy_s": round(a1 - a0, 3), "end_s": round(a1 - t_plan0, 3)}
                             for p, (a0, a1, nb) in span.items()}
        return results

    async def _raw(self, v, grid, key, want_cid, holders) -> bytes | None:
        """Exact stored bytes of `key` with cid `want_cid` (local file or a holder), hash-checked."""
        f = self.local.get(grid, {"files": {}})["files"].get(key)
        if f and os.path.exists(f):
            b = await asyncio.to_thread(Path(f).read_bytes)
            if cid_of(b) == want_cid:
                return b
        for p, cid, _ in holders:
            if cid != want_cid or p not in v["addrs"] or p == self.ident.id:
                continue
            try:
                async with self.session.post(f"{v['addrs'][p]}/cb/{grid}", json={"keys": [key]}, timeout=T_DATA) as r:
                    body = await r.read()
                n = struct.unpack(">I", body[:4])[0]
                if n != MISSING and cid_of(body[4:4 + n]) == want_cid:
                    return body[4:4 + n]
            except Exception:
                continue
        return None

    async def _restore(self, grid: str, v: dict, k: str) -> str | None:
        """Try every stripe family that covers k (a previous family pass may already have rebuilt it)."""
        f = self.local.get(grid, {"files": {}})["files"].get(k)
        if f and os.path.exists(f):
            return f
        failed = v.setdefault("_fam_failed", set())  # a family that cannot be rebuilt now is not retried per key
        for r in v["best"][k]["restore"]:
            if r["fam"] in failed:
                continue
            path = await self._restore_family(grid, v, r["fam"], want=k)
            if path:
                return path
            if not (self.local.get(grid, {"files": {}})["files"].get(k)):
                failed.add(r["fam"])
        return None

    def _rfail(self, why: str):
        f = self.pd_stats.setdefault("restore_fail", {})
        f[why] = f.get(why, 0) + 1

    async def _restore_family(self, grid: str, v: dict, fk: str, want: str) -> str | None:
        """Rebuild every lost member of one stripe from any RS rows (different volunteers) + surviving members;
        each rebuilt chunk is checked against the cid/vcid recorded in the parity header, then stored."""
        f_ = v["fams"][fk]
        if f_["kind"] == "v":
            return await self._restore_value_family(grid, v, f_, want)
        a = v["arrays"][f_["base"]]
        T, K, D, O, lay = a["taxis"], f_["k"], f_["d"], f_["o"], f_["lay"]
        blobs = []
        for row, info in f_["rows"].items():
            for p, cid, _ in info["holders"]:
                b = await self._raw(v, grid, info["pkey"], cid, [(p, cid, 0)])
                if b:
                    blobs.append(b)
                    break
        if not blobs:
            self._rfail("no_parity_row")
            return None
        members = par.header(blobs[0])
        present, missing, keys = {}, [], {}
        for j, mem in enumerate(members):
            if mem[2] < 0:
                continue
            co = list(f_["co"])
            co[T] = par.member_of(f_["co"][T], j, K, D, O)
            mk = f"{f_['base']}@{lay}/" + ".".join(map(str, co))
            keys[j] = mk
            b = await self._raw(v, grid, mk, mem[0], v["best"].get(mk, {}).get("src", []))
            if b is None:
                missing.append(j)
            else:
                present[j] = b
        if not missing:
            self._rfail("nothing_missing")
            return None
        if len(blobs) < len(missing):  # more losses than surviving parity rows (needs k pieces in total)
            self._rfail(f"too_many_losses")
            self.pd_stats.setdefault("loss_hist", {}).setdefault(f"{len(missing)}>{len(blobs)}", 0)
            self.pd_stats["loss_hist"][f"{len(missing)}>{len(blobs)}"] += 1
            return None
        rebuilt = par.restore_many(blobs[:len(missing)], present, missing)
        out = None
        for j, (data, cid, vcid) in rebuilt.items():
            if cid_of(data) != cid:
                continue
            try:
                if not codec.same_vcid(codec.vcid_of(codec.decode(a["layouts"][lay]["docs"], data), a["taxis"]), vcid):
                    continue
            except Exception:
                continue
            path = self._cas(cid)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            mk = keys[j]
            if mk not in v["best"]:
                v["best"][mk] = {"vcid": vcid, "nv": members[j][3], "src": []}
            stored = self._index(grid, v, mk, cid, len(data))
            self.pd_stats["restored"] = self.pd_stats.get("restored", 0) + 1
            if mk == want:
                out = stored
        return out

    async def _fetch_direct(self, grid: str, v: dict, keys: list[str]) -> dict[str, str | None]:
        """Local files, else download - WITHOUT the shared pending-futures registry. Used from inside a running
        fetch (parity restoration), where waiting on the outer fetch's own futures would deadlock."""
        L = self.local.get(grid, {"files": {}})
        out, need = {}, []
        for k in keys:
            f = L["files"].get(k)
            if f and os.path.exists(f):
                out[k] = f
            else:
                need.append(k)
        if need:
            got = await self._download(grid, v, [k for k in need if not v["best"].get(k, {}).get("restore")], None)
            out.update(got)
        return out

    async def _restore_value_family(self, grid: str, v: dict, f_: dict, want: str) -> str | None:
        """Value stripe: surviving members are re-assembled from any layout; the lost ones are rebuilt from
        >= #lost parity rows, checked against their value ids and stored as canonical-layout chunks."""
        a = v["arrays"][f_["base"]]
        T, K, D, O, lay = a["taxis"], f_["k"], f_["d"], f_["o"], f_["lay"]
        li = a["layouts"][lay]
        blobs = []
        for row, info in f_["rows"].items():
            for p, cid, _ in info["holders"]:
                b = await self._raw(v, grid, info["pkey"], cid, [(p, cid, 0)])
                if b:
                    blobs.append(b)
                    break
        if not blobs:
            self._rfail("no_parity_row")
            return None
        members = par.header(blobs[0])
        by_codes = par.mode(blobs[0]) == "codes"
        present, missing, coords = {}, [], {}
        for j, mem in enumerate(members):
            if mem[2] < 0:
                continue
            co = list(f_["co"])
            co[T] = par.member_of(f_["co"][T], j, K, D, O)
            coords[j] = co
            vals = await self._canon_values_tile(grid, v, f_["base"], lay, co)
            lc = codec.lattice_codes(vals, T, exact=False) if by_codes and vals is not None else None
            if vals is None or (by_codes and (lc is None or not codec.same_vcid(codec.vcid_of(vals, T), mem[1]))):
                missing.append(j)  # (a member that does not match its recorded value id cannot help the algebra)
            else:
                present[j] = lc[0].tobytes() if by_codes else \
                    np.ascontiguousarray(vals).astype(vals.dtype.newbyteorder("<"), copy=False).tobytes()
        if not missing:
            self._rfail("nothing_missing")
            return None
        if len(blobs) < len(missing):
            self._rfail("too_many_losses")
            return None
        rebuilt = par.restore_many(blobs[:len(missing)], present, missing)
        dt, _ = _dtype_fill_docs(li["docs"])
        out = None
        for j, (data, _, vcid) in rebuilt.items():
            if by_codes:
                vals = codec.from_lattice_codes(np.frombuffer(data, dtype="<i4").reshape(li["chunks"]), members[j][4], dt, T)
            else:
                vals = np.frombuffer(data, dtype=dt.newbyteorder("<")).reshape(li["chunks"]).astype(dt)
            if not codec.same_vcid(codec.vcid_of(vals, T), vcid):
                self._rfail("vcid_mismatch")
                continue
            enc = await asyncio.to_thread(codec.encode, li["docs"], vals)
            cid = cid_of(enc)
            path = self._cas(cid)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(enc)
            mk = f"{f_['base']}@{lay}/" + ".".join(map(str, coords[j]))
            v["best"].setdefault(mk, {"vcid": vcid, "nv": members[j][3], "src": []})
            stored = self._index(grid, v, mk, cid, len(enc))
            self.pd_stats["restored"] = self.pd_stats.get("restored", 0) + 1
            if mk == want:
                out = stored
        return out

    def _free_row(self, v, var: str, k: int, kind: str) -> int:
        """A Cauchy row of (var, k, kind) nobody in the swarm publishes yet: a duplicated row adds no protection
        (MDS needs DISTINCT rows). One of the 4 lowest free rows by id hash, so volunteers racing on the same
        stale view rarely collide."""
        taken = set()
        for key in v.get("parity", {}):
            b = par.base_of(key.partition("@")[0])
            if b[0] == var and b[1] == k and b[5] == kind:
                taken.add(b[3])
        free = [r for r in range(256 - k) if r not in taken]
        return free[int(self.ident.id[:8], 16) % min(len(free), 4)] if free else int(self.ident.id[:8], 16) % (256 - k)

    async def make_parity(self, grid: str, var: str, k: int, drop: bool = False, layout: str | None = None,
                          d: int | None = None, row: int | None = None, kind: str = "v") -> dict:
        """Volunteer: build XOR parity for every stripe of k time-consecutive chunks of `var`, seed it, and
        optionally drop the data itself (store 1/k while still restoring any single loss per stripe)."""
        v = await self.view(grid, refresh=True)
        a = v["arrays"][var]
        T = a["taxis"]
        if T is None:
            raise ValueError("parity stripes need a time axis")
        # one layout only (default: the one with most chunks) - parity per layout would duplicate the budget
        lay_n = {}
        for key in v["best"]:
            n_, l_, _ = split_key(key)
            if n_ == var:
                lay_n[l_] = lay_n.get(l_, 0) + 1
        layout = layout or max(lay_n, key=lay_n.get)
        if kind == "v":
            return await self._make_value_parity(grid, v, var, layout, k, d, row, drop)
        keys = [key for key in v["best"] if split_key(key)[0] == var and split_key(key)[1] == layout]
        await self.fetch(grid, keys)
        L = self.local[grid]
        cs_ = [split_key(key)[2][T] for key in keys]
        o = min(cs_) if cs_ else 0  # stripes start at the extent start -> ceil(N/k) full stripes, not blocks
        if d is None:  # interleave across the whole extent: a burst loss shorter than D costs <= 1 member/stripe
            d = max(1, -(-(max(cs_) - min(cs_) + 1) // k)) if cs_ else 1
        stripes: dict[tuple, dict[int, str]] = {}
        for key in keys:
            _, lay, co = split_key(key)
            s_, i = par.stripe_of(co[T], k, d, o)
            sp = tuple(c for d, c in enumerate(co) if d != T)
            stripes.setdefault((lay, sp, s_), {})[i] = key
        if row is None:
            row = self._free_row(v, var, k, "p")
        pname = par.parity_name(var, k, d, row, o)
        c = self.caches.setdefault(grid, {"grid": v["grid"], "gdocs": v["gdocs"], "arrays": {}, "chunks": {}})
        made = 0
        for (lay, sp, s_), mem in stripes.items():
            members = []
            for i in range(k):
                key = mem.get(i)
                ent = L["chunks"].get(key) if key else None
                f = L["files"].get(key) if key else None
                b = await asyncio.to_thread(Path(f).read_bytes) if f and os.path.exists(f) else None
                members.append((ent[0], ent[1], ent[3], b) if ent and b is not None else ("", "", 0, None))
            blob = par.encode(members, row=row)
            cid = cid_of(blob)
            path = self._cas(cid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            co = list(sp)
            co.insert(T, s_)
            li = a["layouts"][lay]
            c["arrays"].setdefault(pname, {"vfid": "parity:" + a["vfid"], "dims": a["dims"], "taxis": T, "fmt": a["fmt"],
                                           "attrs": {}, "layouts": {}})["layouts"].setdefault(
                lay, {"fid": "parity", "docs": {}, "chunks": li["chunks"], "phase": li["phase"]}
                | {k_: li[k_] for k_ in ("stride", "soff") if k_ in li})
            c["chunks"][f"{pname}@{lay}/" + ".".join(map(str, co))] = [cid, "", len(blob), 0, par.header(blob)]
            made += 1
        if drop:  # keep only parity: 1/k of the storage
            for key in keys:
                ent = c["chunks"].pop(key, None)
                seeded = any(key in sd["chunks"] for sd in self.seeds.values())
                if ent and not seeded:
                    try:
                        self._cas(ent[0]).unlink()
                    except OSError:
                        pass
        self._dirty.add(grid)
        self._rebuild()
        self._flush()
        await self.announce(grid)
        return {"stripes": made, "k": k, "d": d, "row": row, "parity_bytes": sum(e[2] for kk, e in c["chunks"].items()
                                                            if kk.startswith(pname + "@"))}

    async def _make_value_parity(self, grid, v, var, lay, k, d, row, drop) -> dict:
        """Layout-agnostic stripes: members are canonical chunks (layout `lay`) whose VALUES are assembled from
        whatever layouts the swarm holds; all members have the same byte size (no padding waste)."""
        a = v["arrays"][var]
        T = a["taxis"]
        li = a["layouts"][lay]
        if tstride(li)[0] > 1:  # ponytail: value stripes assume the grid-quantum lattice; byte parity works for any
            raise ValueError("value parity needs a layout at the grid time quantum; use kind='p'")
        ct, ph = li["chunks"][T], li["phase"]
        c_lo, c_hi = (v["gmin"] - ph) // ct, (v["gmax"] - 1 - ph) // ct
        n = c_hi - c_lo + 1
        d = d or max(1, -(-n // k))
        row = row if row is not None else self._free_row(v, var, k, "v")
        o = c_lo
        sp_tiles = [co for co in np.ndindex(*[-(-s_ // cs) for i, (s_, cs) in
                                              enumerate(zip(_meta(li["docs"])["shape"], li["chunks"])) if i != T])]
        pname = par.parity_name(var, k, d, row, o, kind="v")
        c = self.caches.setdefault(grid, {"grid": v["grid"], "gdocs": v["gdocs"], "arrays": {}, "chunks": {}})
        c["arrays"].setdefault(pname, {"vfid": "vparity:" + a["vfid"], "dims": a["dims"], "taxis": T, "fmt": a["fmt"],
                                       "attrs": {}, "layouts": {}})["layouts"].setdefault(
            lay, {"fid": "vparity", "docs": {}, "chunks": li["chunks"], "phase": ph})
        made, n_codes = 0, 0
        n_stripes = -(-n // (k * d)) * d
        for sp in sp_tiles:
            for t in range(n_stripes):
                members = []
                for i in range(k):
                    cc = par.member_of(t, i, k, d, o)
                    if cc > c_hi:
                        members.append(("", "", 0, None, None))
                        continue
                    co = list(sp)
                    co.insert(T, cc)
                    vals = await self._canon_values_tile(grid, v, var, lay, co)
                    if vals is None:
                        members.append(("", "", 0, None, None))
                        continue
                    nv = min(ct, v["gmax"] - (cc * ct + ph))
                    b = np.ascontiguousarray(vals).astype(vals.dtype.newbyteorder("<"), copy=False).tobytes()
                    members.append(("", codec.vcid_of(vals, T), nv, b, codec.lattice_codes(vals, T)))
                if all(m[3] is None for m in members):
                    continue
                # code over lattice codes when they reproduce every member exactly: identical for every decoder of
                # the same measurements, so a repair may assemble members from sites whose floats differ in the bits
                # (raw float bytes made such a repair rebuild garbage that the vcid check then rejected)
                codes = all(m[3] is None or m[4] is not None for m in members)
                n_codes += codes
                members = [(c_, vc, nv_, None if b_ is None else (lc[0].tobytes() if codes else b_),
                            *([lc[1]] if codes and lc else [[]] if codes else []))
                           for c_, vc, nv_, b_, lc in members]
                blob = par.encode(members, row=row, mode="codes" if codes else "bytes")
                cid = cid_of(blob)
                path = self._cas(cid)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
                pco = list(sp)
                pco.insert(T, t)
                c["chunks"][f"{pname}@{lay}/" + ".".join(map(str, pco))] = [cid, "", len(blob), 0, par.header(blob)]
                made += 1
        if drop:  # keep only the value parity (1/k)
            for key in [kk for kk in list(c["chunks"]) if split_key(kk)[0] == var]:
                ent = c["chunks"].pop(key)
                if not any(key in sd["chunks"] for sd in self.seeds.values()):
                    try:
                        self._cas(ent[0]).unlink()
                    except OSError:
                        pass
        self._dirty.add(grid)
        self._rebuild()
        self._flush()
        await self.announce(grid)
        return {"stripes": made, "code_stripes": n_codes, "k": k, "d": d, "row": row, "kind": "v", "layout": lay,
                "parity_bytes": sum(e[2] for kk, e in c["chunks"].items() if kk.startswith(pname + "@"))}

    async def _canon_values_tile(self, grid, v, var, lay, co):
        """Values of canonical chunk `co` (layout `lay`, one spatial tile) assembled from ANY layout in the swarm,
        only if its valid time range is fully covered (partial coverage would break the code algebra); None
        otherwise. Source chunks are fetched vcid-verified."""
        from .store import View
        a = v["arrays"][var]
        T = a["taxis"]
        li = a["layouts"][lay]
        ct, ph = li["chunks"][T], li["phase"]
        g0 = co[T] * ct + ph
        g1 = min(g0 + ct, v["gmax"])
        if g1 <= g0 or not _covers(v.get("covered", {}).get(var, []), g0, g1):
            return None
        vw = self._views_obj.get(id(v))
        if vw is None:
            vw = self._views_obj[id(v)] = View(v)
        shape = list(_meta(li["docs"])["shape"])
        region = []
        for i, (cs, sz) in enumerate(zip(li["chunks"], shape)):
            region.append(((g0 - vw.G0) // vw.S, -(-(g1 - vw.G0) // vw.S)) if i == T else (co[i] * cs, min((co[i] + 1) * cs, sz)))
        plan = vw.sources(var, region)
        if not plan:
            return None
        paths = await self._fetch_direct(grid, v, list(dict.fromkeys(k for k, _, _ in plan)))
        dt, fill = _dtype_fill_docs(li["docs"])
        out = np.full(li["chunks"], 0 if fill is None else fill, dtype=dt)
        for k, src, dst in plan:
            pth = paths.get(k)
            if not pth or pth in ("!", "retry"):
                return None
            _, l2, _ = split_key(k)
            vals = await asyncio.to_thread(self.decoded.get, k, a["layouts"][l2]["docs"], lambda p_=pth: Path(p_).read_bytes())
            out[dst] = vals[src]
        return out

    async def a_parity(self, req):
        d = await req.json()
        return web.json_response(await self.make_parity(d["grid"], d["var"], int(d.get("k", 8)), bool(d.get("drop")),
                                                        d=int(d["d"]) if d.get("d") else None,
                                                        row=int(d["row"]) if d.get("row") is not None else None,
                                                        kind=d.get("kind", "v")))

    async def _bw_model(self, peers, v) -> tuple[dict, dict, dict]:
        """Per-peer bandwidth (measured EWMA > announced hint > median) and shared relay bottlenecks."""
        known = list(self.bw.values())
        dflt = sorted(known)[len(known) // 2] if known else DEFAULT_BW
        bw = {p: self.bw.get(p) or (HINT_EFF * self.hints[p] if self.hints.get(p) else dflt) for p in peers}
        via = {p: v["addrs"][p].split("/r/")[0] for p in peers if "/r/" in v["addrs"].get(p, "")}
        via_bw = {}
        for r in set(via.values()):  # relay uplink is shared by every peer behind it
            if r not in self.relay_bw:
                try:
                    async with self.session.get(f"{r}/whoami", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        self.relay_bw[r] = (await resp.json()).get("bw")
                except Exception:
                    self.relay_bw[r] = None
            via_bw[r] = HINT_EFF * self.relay_bw[r] if self.relay_bw[r] else dflt
        # a relay that also seeds serves its own chunks through the SAME uplink: one shared bottleneck, not two
        for p in peers:
            r = v["addrs"].get(p)
            if r in via_bw:
                via[p] = r
                via_bw[r] = min(via_bw[r], bw[p])
        return bw, via, via_bw

    async def region_view(self, grid: str, var: str, g_lo: int, g_hi: int) -> dict | None:
        """Partial view for one variable over [g_lo, g_hi): manifest heads + only the CAPT subtrees of that
        key range (per layout). First contact with a huge archive costs O(log N + range), not O(N)."""
        peers = await self.find_peers(grid)

        async def one(nid, addr):
            try:
                async with self.session.get(f"{addr}/mh/{grid}", timeout=T_META) as r:
                    if r.status != 200:
                        return None
                    comp, pk, sig, root = await r.read(), r.headers.get("X-Zt-Pk", ""), \
                        r.headers.get("X-Zt-Sig", ""), r.headers.get("ETag", "")
                body = zlib.decompressobj().decompress(comp, MAX_MANIFEST)
                if h160(bytes.fromhex(pk)) != nid or not verify(pk, sig, body):
                    return None
                m = json.loads(body)
                a = m["arrays"].get(var)
                if grid_id_of(m["grid"]) != grid or a is None or a["taxis"] is None or m.get("root") != root:
                    return None
                ranges = []
                for lay, li in a["layouts"].items():
                    nd = len(li["chunks"])
                    cr = chunks_for(li, a["taxis"], g_lo, g_hi)
                    if cr is None:
                        continue
                    c0, c1 = cr
                    lo_co = [0] * nd
                    hi_co = [10 ** 15] * nd
                    lo_co[a["taxis"]], hi_co[a["taxis"]] = c0, c1
                    ranges.append(((var, lay, lo_co), (var, lay, hi_co)))
                m["chunks"] = await self._capt_load(addr, grid, root, ranges)
                m["arrays"] = {var: a}
                return m
            except Exception:
                return None
        tasks = {asyncio.create_task(one(n, a)): n for n, a in peers.items()}
        done, _ = await asyncio.wait(tasks, timeout=VIEW_WAIT * 3) if tasks else (set(), set())
        mans = {tasks[t]: t.result() for t in done if t.result()}
        L = self.local.get(grid)
        if L and var in L["arrays"]:
            mans[self.ident.id] = {"node": self.ident.id, "grid": L["grid"], "gdocs": L["gdocs"],
                                   "arrays": {var: L["arrays"][var]},
                                   "chunks": {k: e for k, e in L["chunks"].items() if split_key(k)[0] == var}}
        if not mans:
            return None
        v = await asyncio.to_thread(merge_view, grid, mans, self.ident.id, self.trusted)
        v.update(ts=time.time(), addrs=peers)
        return v

    async def _grid_time(self, grid: str) -> dict | None:
        """Time definition of a grid from the local index, a cached view, or one small manifest head."""
        if grid in self.local:
            return self.local[grid]["grid"]["time"]
        if grid in self.views:
            return self.views[grid]["grid"]["time"]
        for nid, addr in (await self.find_peers(grid)).items():
            try:
                async with self.session.get(f"{addr}/mh/{grid}", timeout=T_META) as r:
                    if r.status == 200:
                        g = json.loads(zlib.decompressobj().decompress(await r.read(), MAX_MANIFEST))["grid"]
                        if grid_id_of(g) == grid:
                            return g["time"]
            except Exception:
                continue
        return None

    async def region_keys(self, grid: str, reg: dict, cover: str = "jlps") -> tuple[list[str], dict]:
        """Source keys for {var, t0, t1} chosen by JLPS (joint layout+peer) or by min bytes.
        A bounded time window on a grid without a fresh full view uses a CAPT range view (O(range))."""
        var = reg["var"]
        fresh = self.views.get(grid)
        if reg.get("t0") and reg.get("t1") and not (fresh and time.time() - fresh["ts"] < VIEW_TTL):
            t = await self._grid_time(grid)
            if t:
                sec_ = lambda x, unit="s": int(np.datetime64(x, unit).astype("datetime64[s]").astype("int64"))
                t1 = str(reg["t1"]).strip()
                end = sec_(t1, "D") + 86400 if len(t1) == 10 else sec_(t1) + 1
                g_lo = -(-(sec_(reg["t0"]) - t["rphase"]) // t["dt"])
                g_hi = -(-(end - t["rphase"]) // t["dt"])
                rv = await self.region_view(grid, var, g_lo, g_hi)
                if rv and var in rv["arrays"]:
                    lat = _lattice(rv["arrays"][var], t, reg.get("step"), g_lo)
                    keys, info = await self._cover(rv, var, g_lo, g_hi, reg.get("isel") or None, cover, lat)
                    return keys, dict(info, view="capt-range", _view=rv)
        v = await self.view(grid)
        t = v["grid"]["time"]
        if var not in v["arrays"]:
            v = await self.view(grid, refresh=True)  # cached view may predate a new seeder
        if var not in v["arrays"]:
            raise KeyError(f"variable {var!r} not in swarm view")
        isel = reg.get("isel") or None
        if t is None or v["arrays"][var]["taxis"] is None:
            a = v["arrays"][var]
            box = {a["dims"].index(d): r for d, r in (isel or {}).items() if d in a["dims"]}

            def inside(k):
                _, lay, co = split_key(k)
                cs = a["layouts"][lay]["chunks"]
                return all(co[i] * cs[i] < hi and (co[i] + 1) * cs[i] > lo for i, (lo, hi) in box.items())
            return [k for k in v["best"] if split_key(k)[0] == var and inside(k)], {"cover": "all"}
        sec = lambda x, unit="s": int(np.datetime64(x, unit).astype("datetime64[s]").astype("int64"))
        g_lo, g_hi = v["gmin"], v["gmax"]
        if reg.get("t0"):
            g_lo = max(g_lo, -(-(sec(reg["t0"]) - t["rphase"]) // t["dt"]))
        if reg.get("t1"):
            t1 = str(reg["t1"]).strip()
            end = sec(t1, "D") + 86400 if len(t1) == 10 else sec(t1) + 1
            g_hi = min(g_hi, -(-(end - t["rphase"]) // t["dt"]))
        lat = _lattice(v["arrays"][var], t, reg.get("step"), g_lo if reg.get("t0") else None)
        return await self._cover(v, var, g_lo, g_hi, isel, cover, lat)

    async def _cover(self, v, var, g_lo, g_hi, isel, cover, lattice=(1, 0)):
        keys, info = await self._cover_core(v, var, g_lo, g_hi, isel, cover, lattice)
        info = dict(info, lattice=list(lattice))
        # chunks nobody holds any more but a stripe parity can restore (ZTP-EC) join the request as-is
        a = v["arrays"][var]
        extra = []
        for k, b in v["best"].items():  # restorable (parity) and contested (no majority) keys join as-is: the job
            if not (b.get("restore") or b.get("contested")) or b["src"] or split_key(k)[0] != var:  # reports them
                continue
            _, lay, co = split_key(k)
            li = a["layouts"][lay]
            g0, g1 = chunk_extent(li, a["taxis"], co[a["taxis"]])
            if g0 < g_hi and g1 > g_lo:
                extra.append(k)
        if extra:
            info = dict(info, restorable=sum(1 for k in extra if v["best"][k].get("restore")),
                        contested=sum(1 for k in extra if v["best"][k].get("contested")))
        return keys + extra, info

    async def _cover_core(self, v, var, g_lo, g_hi, isel, cover, lattice=(1, 0)):
        me = self.ident.id
        holders = {h for b in v["best"].values() for h, _, _ in b["src"]}
        bw, via, via_bw = await self._bw_model(holders - {me}, v)
        bw[me] = 1e12  # chunks we already hold are free
        if cover == "bytes":
            return jlpsmod.bytes_greedy_cover(var, v["arrays"], v["best"], g_lo, g_hi, isel, lattice), {"cover": "bytes"}
        keys, _, T, info = jlpsmod.jlps(var, v["arrays"], v["best"], g_lo, g_hi, bw, via=via, via_bw=via_bw, isel=isel,
                                        lattice=lattice)
        return keys, dict(info, cover="jlps", est_T=round(T, 3))

    async def _rival_ok(self, v: dict, grid: str, k: str, cid: str) -> bool:
        """The chunk we accepted (cid, in the view's encoding) won its tie by completeness only: it must agree with
        the less complete rival on every cell the rival holds (within half a lattice step), else nobody wins."""
        name, lay, _ = split_key(k)
        rv = v["best"][k]["rival"]
        a = v["arrays"][name]
        mine = await asyncio.to_thread(codec.decode, a["layouts"][lay]["docs"], self._cas(cid).read_bytes())
        for rp, rcid, _ in rv["src"]:
            raw = await self._raw(v, grid, k, rcid, [(rp, rcid, 0)])
            if raw is None:
                continue
            try:
                theirs = await asyncio.to_thread(codec.decode, v["pinfo"][rp][name]["layouts"][lay]["docs"], raw)
            except Exception:
                continue
            if not codec.same_vcid(codec.vcid_of(theirs, a["taxis"]), rv["vcid"]):
                continue  # not even the rival's own value: ask another rival holder
            m = np.isfinite(theirs) if theirs.dtype.kind == "f" else np.ones(theirs.shape, bool)
            parts = rv["vcid"].split(":")
            tol = float(parts[3]) / 2 if len(parts) == 4 else 0.0
            return bool(np.all(np.abs(mine[m].astype("f8") - theirs[m].astype("f8")) <= tol))
        return False  # the rival could not be checked: completeness alone does not decide

    def _verify_store(self, v: dict, k: str, p: str, data: bytes) -> tuple[str, int] | None:
        """Thread: check cid, transcode if the peer uses another encoding (checked by vcid), write CAS."""
        name, lay, _ = split_key(k)
        b = v["best"][k]
        cid = next(c for q, c, _ in b["src"] if q == p)
        va, pa = v["arrays"][name]["layouts"][lay], v["pinfo"][p][name]["layouts"][lay]
        if data[:4] == codec.XT1:  # transport-encoded values: verify by vcid, store in our own encoding
            try:
                vals = codec.xt1_decode(data)
            except Exception:
                return None
            if not codec.same_vcid(codec.vcid_of(vals, v["arrays"][name]["taxis"]), b["vcid"]):
                return None
            data = codec.encode(va["docs"], vals)
            cid = cid_of(data)
            path = self._cas(cid)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}")
                tmp.write_bytes(data)
                tmp.replace(path)
            return cid, len(data)
        if cid_of(data) != cid:
            return None
        # always check the value-level id chosen by holder majority: a single peer cannot poison the swarm
        # by advertising its own (cid, vcid) for bytes that decode to other values
        try:
            vals = codec.decode(pa["docs"], data)
        except Exception:
            return None
        if not codec.same_vcid(codec.vcid_of(vals, v["arrays"][name]["taxis"]), b["vcid"]):
            return None
        if pa["fid"] != va["fid"]:
            data = codec.encode(va["docs"], vals)
            cid = cid_of(data)
        path = self._cas(cid)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}")
            tmp.write_bytes(data)
            tmp.replace(path)
        return cid, len(data)

    def _index(self, grid: str, v: dict, k: str, cid: str, size: int) -> str:
        """Event loop: register a stored chunk in the local cache index (so it is seeded)."""
        name, lay, _ = split_key(k)
        b, va = v["best"][k], v["arrays"][name]
        path = self._cas(cid)
        c = self.caches.setdefault(grid, {"grid": v["grid"], "gdocs": v["gdocs"], "arrays": {}, "chunks": {}})
        ca = c["arrays"].setdefault(name, {kk: va[kk] for kk in ("vfid", "dims", "taxis", "fmt", "attrs")} | {"layouts": {}})
        li = {kk: va["layouts"][lay][kk] for kk in ("fid", "docs", "chunks", "phase", "stride", "soff")
              if kk in va["layouts"][lay]}
        if ca["vfid"] != va["vfid"] or ca["layouts"].setdefault(lay, li)["fid"] != li["fid"]:
            # the cache holds this variable (or layout) in another family, e.g. written by an older version of the
            # swarm or node: the chunk just verified for the current view wins, the stale family is dropped (kept,
            # it was served for the new view's key and decoded with the wrong dtype)
            same_var = ca["vfid"] == va["vfid"]
            for ck in [ck for ck in c["chunks"] if split_key(ck)[0] == name and (not same_var or split_key(ck)[1] == lay)]:
                del c["chunks"][ck]
            if same_var:
                ca["layouts"][lay] = li
            else:
                ca = c["arrays"][name] = {kk: va[kk] for kk in ("vfid", "dims", "taxis", "fmt", "attrs")} | {"layouts": {lay: li}}
            c["chunks"][k] = [cid, b["vcid"], size, b["nv"]]
            self.used[(grid, k)] = time.time()
            self._rebuild()
            self._dirty.add(grid)
            return str(path)
        c["chunks"][k] = [cid, b["vcid"], size, b["nv"]]
        self.used[(grid, k)] = time.time()
        L = self.local.setdefault(grid, {"grid": v["grid"], "gdocs": v["gdocs"], "arrays": {}, "chunks": {}, "files": {}})
        La = L["arrays"].setdefault(name, dict(ca, layouts={}))
        La["layouts"].setdefault(lay, li)
        L["chunks"][k] = c["chunks"][k]
        L["files"][k] = str(path)
        self._mf.pop(grid, None)
        self._lver += 1
        self._dirty.add(grid)
        return str(path)

    # ------------------------------------------------------------------ control API
    async def a_status(self, req):
        return web.json_response({
            "id": self.ident.id, "pk": self.ident.pk, "addr": self.addr, "ro": self.ro, "relay_up": self.relay_up,
            "private": bool(self.net_key), "home": str(self.home),
            "contacts": len(self.dht.contacts()), "relayed": len(self.relayed),
            "seeds": [{"path": p, "grid": g, "chunks": len(s["chunks"]), "arrays": sorted(s["arrays"]),
                       "bytes": sum(e[2] for e in s["chunks"].values())} for (p, g), s in self.seeds.items()],
            "grids": {g: {"chunks": len(L["chunks"]), "bytes": sum(e[2] for e in L["chunks"].values()),
                          "arrays": sorted(L["arrays"])} for g, L in self.local.items()},
            "jobs": [{k: v for k, v in j.items() if k != "done_keys"} for j in list(self.jobs.values())[-50:]],
            "served": self.served, "dht_values": sum(len(v) for v in self.dht.values.values()),
            "page_fetched": self.page_fetched,
            "pushdown": self.pd_stats, "fraud": self.fraud[-20:], "blacklist": sorted(self.bad),
            "tainted": len(self.tainted),
            "bw": {p: round(b) for p, b in self.bw.items()}})

    async def a_seed(self, req):
        res = await self.add_seed((await req.json())["path"])
        sgs = res["subgrids"]
        return web.json_response({"grids": sorted(sgs), "chunks": sum(len(g["chunks"]) for g in sgs.values()),
                                  "arrays": sorted({n for g in sgs.values() for n in g["arrays"]}),
                                  "link": "zt://" + "+".join(sorted(sgs))})

    async def a_unseed(self, req):
        p = str(Path((await req.json())["path"]).resolve())
        for k in [k for k in self.seeds if k[0] == p]:
            del self.seeds[k]
        self._rebuild()
        self._save_state()
        return web.json_response({"ok": True})

    async def a_resolve(self, req):
        try:
            return web.json_response({"grid": await self.resolve(req.query["link"])})
        except KeyError as e:
            raise web.HTTPNotFound(text=str(e))

    async def a_view(self, req):
        try:
            v = await self.view(req.match_info["grid"], refresh=req.query.get("refresh") == "1")
        except KeyError as e:
            raise web.HTTPNotFound(text=str(e))
        # per-layout aggregate holder rate (bytes/s): lets clients price layouts like JLPS (lambda = 1/bw)
        holders = {}
        for k, b in v["best"].items():
            n, lay, _ = split_key(k)
            holders.setdefault((n, lay), set()).update(p for p, _, _ in b["src"])
        known = list(self.bw.values())
        dflt = sorted(known)[len(known) // 2] if known else DEFAULT_BW
        for (n, lay), ps in holders.items():
            v["arrays"][n]["layouts"][lay]["rate"] = sum(
                1e12 if p == self.ident.id else (self.bw.get(p) or self.hints.get(p) or dflt) for p in ps)
        return web.json_response({k: v[k] for k in ("grid", "gdocs", "arrays", "gmin", "gmax", "fmt")} |
                                 {"nchunks": len(v["best"]), "npeers": len(v["addrs"]),
                                  "local": len(self.local.get(v["grid_id"], {}).get("chunks", {}))})

    async def a_fetch(self, req):
        d = await req.json()
        return web.json_response(await self.fetch(d["grid"], d["keys"], expect=d.get("expect")))

    async def a_read(self, req):
        """Bytes of one locally held chunk, for clients that cannot open the node's paths (node in a container,
        other mount namespace). Only files of the local index are served - never an arbitrary path."""
        d = await req.json()
        path = self.local.get(d["grid"], {"files": {}})["files"].get(d["key"])
        if path is None:
            raise web.HTTPNotFound()
        return web.Response(body=await asyncio.to_thread(Path(path).read_bytes))

    async def a_download(self, req):
        d = await req.json()
        try:
            return web.json_response({"job": await self.start_job(d)})
        except Exception as e:
            import traceback
            raise web.HTTPBadRequest(text=f"region cover failed: {e!r}\n{traceback.format_exc()}")

    async def start_job(self, d: dict) -> str:
        """Download job: explicit `keys`, or a `region` {var, t0, t1, isel} resolved by cover selection."""
        info, rview = {}, None
        if "region" in d:  # server-side cover selection (JLPS by default)
            d["keys"], info = await self.region_keys(d["grid"], d["region"], d.get("cover", "jlps"))
            rview = info.pop("_view", None)
            if d.get("cover", "jlps") == "jlps" and self.strategy == "maxflow":
                # JLPS chooses layouts and holders by rate: peers it knows nothing about are measured first, on
                # chunks of this cover, and the cover is chosen again with the measured rates (a slow site planned at
                # the default rate made the swarm slower than its best single source on a real three-site run)
                v = rview or self.views.get(d["grid"])
                probed = await self._probe_unknown(d["grid"], v, d["keys"]) if v else 0
                if probed:
                    d["keys"], info = await self.region_keys(d["grid"], d["region"], "jlps")
                    rview = info.pop("_view", None)
                    info["probed"] = probed
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "grid": d["grid"], "label": d.get("label", ""), "state": "running", "total": 0,
               "done": 0, "bytes": 0, "missing": 0, "per_peer": {}, "t0": time.time(), "t1": None,
               "order": d.get("order", "optimal"), "done_keys": [], "cover": info}
        self.jobs[jid] = job
        self._job_keys[jid] = (d["keys"], rview)
        self._launch(job)
        return jid

    def _launch(self, job: dict):
        """(Re)start a job: keys already present locally count as done, the rest is fetched."""
        grid, (keys, rview) = job["grid"], self._job_keys[job["id"]]
        L = self.local.get(grid, {"files": {}})
        have = [k for k in keys if k in L["files"]]
        job.update(state="running", t1=None, cancel=False, pause=False, total=len(have), done=len(have),
                   missing=0, failed=0, done_keys=list(have))

        async def run():
            try:
                await self.fetch(grid, [k for k in keys if k not in L["files"]], job, view=rview)
                job["state"] = ("paused" if job.get("pause") else "cancelled") if job.get("cancel") \
                    else "partial" if job.get("failed") else "done"
            except Exception as e:
                job["state"] = f"error: {e}"
            job["t1"] = time.time()
            self._flush()
            await self.announce(grid)
        asyncio.create_task(run())

    async def a_pause(self, req):
        job = self.jobs.get(req.match_info["jid"])
        if job and job["state"] == "running":
            job["pause"] = job["cancel"] = True  # workers stop at the next batch; fetched chunks are kept
        return web.json_response({"ok": bool(job)})

    async def a_resume(self, req):
        job = self.jobs.get(req.match_info["jid"])
        if not job or job["id"] not in self._job_keys:
            raise web.HTTPNotFound()
        if job["state"] != "running":
            self._launch(job)
        return web.json_response({"ok": True})

    # ------------------------------------------------------------------ subscriptions (follow a growing feed)
    async def follow_once(self, sub: dict) -> list[str]:
        """Bring the last `last_s` seconds (before the newest step anyone holds) of the subscribed variables up
        to date. The link is re-resolved every time, so a name that moves to a new version is followed."""
        jobs = []
        grids = (await self.resolve(sub["link"])).split("+")
        for g in grids:
            prev = [j for j in sub.get("jobs", []) if self.jobs.get(j, {}).get("state") == "running"]
            if prev:
                return prev  # still catching up
            v = await self.view(g, refresh=True)
            t = v["grid"].get("time")
            if not t or v["gmax"] is None:
                continue
            end = (v["gmax"] - 1) * t["dt"] + t["rphase"]
            iso = lambda x: datetime.fromtimestamp(x, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            for var in sub.get("vars") or [n for n in v["arrays"] if n not in v["grid"]["dims"] and "#" not in n]:
                if var not in v["arrays"]:
                    continue
                region = {"var": var, "t0": iso(end - sub["last_s"] + t["dt"]), "t1": iso(end),
                          "isel": sub.get("isel"), "step": sub.get("step")}
                keys, _ = await self.region_keys(g, region, "jlps")
                if all(k in self.local.get(g, {"files": {}})["files"] for k in keys):
                    continue
                jobs.append(await self.start_job({"grid": g, "keys": keys, "label": f"follow {var} {sub['id']}"}))
        sub["jobs"] = jobs
        return jobs

    async def _follow_loop(self):
        while True:
            for sub in list(self.subs.values()):
                try:
                    await self.follow_once(sub)
                except Exception as e:
                    print(f"[zt] follow {sub['link']}: {e}")
            await asyncio.sleep(FOLLOW_EVERY)

    async def a_follow(self, req):
        d = await req.json()
        sub = {"id": uuid.uuid4().hex[:6], "link": d["link"], "vars": d.get("vars") or None,
               "last_s": float(d["last_s"]), "isel": d.get("isel") or None, "step": d.get("step") or None}
        self.subs[sub["id"]] = sub
        self._save_state()
        jobs = await self.follow_once(sub)
        return web.json_response({"id": sub["id"], "jobs": jobs})

    async def a_follows(self, req):
        return web.json_response([{k: v for k, v in s.items()} for s in self.subs.values()])

    async def a_unfollow(self, req):
        ok = self.subs.pop((await req.json())["id"], None) is not None
        self._save_state()
        return web.json_response({"ok": ok})

    async def a_job(self, req):
        job = self.jobs.get(req.match_info["jid"])
        if not job:
            raise web.HTTPNotFound()
        since = int(req.query.get("since", 0))
        return web.json_response({k: v for k, v in job.items() if k != "done_keys"} |
                                 {"done_keys": job["done_keys"][since:]})

    async def a_cancel(self, req):
        job = self.jobs.get(req.match_info["jid"])
        if job:
            job["cancel"] = True
            job["pause"] = False
            if job["state"] == "paused":  # nothing running: settle it now
                job["state"] = "cancelled"
        return web.json_response({"ok": bool(job)})

    async def a_peers(self, req):
        g = req.match_info["grid"]
        try:
            v = await self.view(g, refresh=True)
        except KeyError as e:
            raise web.HTTPNotFound(text=str(e))
        cov = {}
        for k, b in v["best"].items():
            for p, _, _ in b["src"]:
                cov[p] = cov.get(p, 0) + 1
        return web.json_response({"peers": [{"id": p, "addr": v["addrs"].get(p, self.addr), "chunks": n,
                                             "bw": round(self.bw.get(p, 0))} for p, n in cov.items()]})

    async def a_pieces(self, req):
        """Piece map per variable, like a torrent client's: the time extent split into `width` bins; per bin the
        best holder count of the chunks covering it (0 = nobody has it) and the fraction held locally."""
        g, width = req.match_info["grid"], max(1, min(int(req.query.get("width", 60)), 400))
        try:
            v = await self.view(g)
        except KeyError as e:
            raise web.HTTPNotFound(text=str(e))
        mine = self.local.get(v["grid_id"], {}).get("chunks", {})
        lo, hi = v["gmin"], v["gmax"]
        if lo is None or hi is None:  # no time axis: nothing to map
            return web.json_response({"t0": None, "t1": None, "vars": {}})
        span = max(hi - lo, 1)
        out = {}
        for k, b in v["best"].items():
            n, lay, co = split_key(k)
            a = v["arrays"].get(n)
            if a is None or a.get("taxis") is None:
                continue
            li, T = a["layouts"][lay], a["taxis"]
            t0, t1 = chunk_g(li, T, co[T])
            bins = out.setdefault(n, [[0, 0, 0] for _ in range(width)])  # [holders, local, total]
            for i in range(max(0, (t0 - lo) * width // span), min(width, -(-(t1 - lo) * width // span))):
                bins[i][0] = max(bins[i][0], len(b["src"]))
                bins[i][1] += k in mine
                bins[i][2] += 1
        t = v["grid"].get("time") or {}
        return web.json_response({"t0": lo * t["dt"] + t["rphase"] if t else None,
                                  "t1": (hi - 1) * t["dt"] + t["rphase"] if t else None,  # last step held
                                  "vars": {n: [[h, round(l / max(c, 1), 3)] for h, l, c in bins] for n, bins in out.items()}})

    async def a_search(self, req):
        return web.json_response(await self.search(req.query["tag"]))

    async def publish_name(self, name: str, target: str, seq: int) -> int:
        rec = self.ident.signed({"t": "name", "name": name, "target": target, "seq": seq, "ts": time.time()})
        return await self.dht.store(h160(bytes.fromhex(self.ident.pk) + name.encode()), rec)

    async def a_name(self, req):
        d = await req.json()
        seq = time.time_ns()
        self.names[d["name"]] = [d["target"], seq]  # persisted and republished (records expire after TTL)
        self._save_state()
        n = await self.publish_name(d["name"], d["target"], seq)
        return web.json_response({"link": f"zt://{d['name']}@{self.ident.pk}", "stored_on": n})


def progressive_order(keys: list[str]) -> list[str]:
    """Low-discrepancy (bit-reversal / van der Corput) order over the chunk grid, so that every
    prefix of the download is a near-stratified sample of the selection (anytime estimates)."""
    ks = sorted(keys, key=lambda k: split_key(k)[2])
    n = len(ks)
    if n < 2:
        return ks
    bits = (n - 1).bit_length()
    rank = sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))
    return [ks[i] for i in rank]


def _dtype_of(docs: dict) -> np.dtype:
    arr, _, _ = codec._one_chunk_array(docs, None, read_only=True)
    return np.dtype(arr.dtype).newbyteorder("<")


def _meta(docs: dict) -> dict:
    return docs["zarr.json"] if "zarr.json" in docs else docs[".zarray"]


def _dtype_fill_docs(docs: dict):
    arr, _, _ = codec._one_chunk_array(docs, None, read_only=True)
    return np.dtype(arr.dtype), arr.fill_value


def _lattice(a: dict, t: dict, step_s=None, g_ref: int | None = None) -> tuple[int, int]:
    """Request lattice (S, O) in grid quanta: `step_s` seconds if given (aligned to the requested start or to
    the replicas), else the finest stride any layout of the variable offers."""
    lays = list(a["layouts"].values())
    if step_s:
        S = int(float(step_s)) // t["dt"]
        if S < 1 or int(float(step_s)) % t["dt"]:
            raise ValueError(f"step {step_s}s is not a multiple of the grid quantum {t['dt']}s")
        offs = [li.get("soff", 0) % S for li in lays if S % li.get("stride", 1) == 0]
        return S, (g_ref % S if g_ref is not None and not offs else (offs[0] if offs else 0))
    S, O = min(natural_stride(li, a["taxis"]) for li in lays)
    return S, O % S


def _time_coverage(best: dict, arrays: dict, parsed: dict) -> dict:
    """Merged absolute-time intervals [g0, g1) held by anyone, per variable, over every dense layout (stride 1:
    a coarser replica holds only every stride-th quantum and does not cover the interval between samples)."""
    iv: dict[str, list] = {}
    for k, b in best.items():
        if not b["src"]:
            continue
        n, lay, co = parsed.get(k) or split_key(k)
        a = arrays.get(n)
        if not a or a["taxis"] is None or lay not in a["layouts"]:
            continue
        li = a["layouts"][lay]
        if tstride(li)[0] > 1:
            continue
        iv.setdefault(n, []).append(chunk_extent(li, a["taxis"], co[a["taxis"]], b["nv"] or None))
    out = {}
    for n, xs in iv.items():
        xs.sort()
        m = [list(xs[0])]
        for a0, a1 in xs[1:]:
            if a0 <= m[-1][1]:
                m[-1][1] = max(m[-1][1], a1)
            else:
                m.append([a0, a1])
        out[n] = m
    return out


def _covers(ivs: list, g0: int, g1: int) -> bool:
    return any(a0 <= g0 and g1 <= a1 for a0, a1 in ivs)


def tag_key(tag: str) -> str:
    return h160(("tag:" + tag).encode())


def _ranges(xs: list[int]) -> list[list[int]]:
    out = []
    for x in xs:
        if out and out[-1][1] == x - 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return out


def local_extent(L: dict) -> tuple[int, int] | None:
    """[gmin, gmax) absolute time-step extent of the chunks in a (local or merged) index."""
    gmin = gmax = None
    for k, ent in L["chunks"].items():
        n, lay, co = split_key(k)
        if par.is_parity(n):
            continue
        a = L["arrays"][n]
        if a["taxis"] is None:
            continue
        li = a["layouts"][lay]
        g0, g1 = chunk_extent(li, a["taxis"], co[a["taxis"]], ent[3] or None)
        gmin = g0 if gmin is None else min(gmin, g0)
        gmax = g1 if gmax is None else max(gmax, g1)
    return None if gmin is None else (gmin, gmax)


def merge_view(grid: str, mans: dict[str, dict], me: str, trusted: set = frozenset()) -> dict:
    """Union view over all replicas and layouts: majority value-family per variable, per layout the
    preferred encoding (own > majority), best replica per chunk, per-layout time coverage & sizes."""
    first = mans.get(me) or next(iter(mans.values()))
    # view format: our own replica's if we have one, else the majority over all replicas' arrays (ties -> v3)
    fmts = [a["fmt"] for m in mans.values() for a in m["arrays"].values()]
    own = [a["fmt"] for a in mans[me]["arrays"].values()] if me in mans else []
    fmt = own[0] if own else (max(sorted(set(fmts), reverse=True), key=fmts.count) if fmts else 3)
    if not own:
        first = next((m for m in mans.values() if any(a["fmt"] == fmt for a in m["arrays"].values())), first)
    arrays, pinfo = {}, {p: m["arrays"] for p, m in mans.items()}
    for n in sorted({n for m in mans.values() for n in m["arrays"] if not par.is_parity(n)}):
        vf = [m["arrays"][n]["vfid"] for m in mans.values() if n in m["arrays"]]
        vfid = max(set(vf), key=vf.count)
        cands = [(p, m["arrays"][n]) for p, m in mans.items() if n in m["arrays"] and m["arrays"][n]["vfid"] == vfid]
        base = next((a for p, a in cands if p == me), cands[0][1])
        lays = {}
        for lay in sorted({lay for _, a in cands for lay in a["layouts"]}):
            opts = [(p, a["layouts"][lay]) for p, a in cands if lay in a["layouts"]]
            li = next((li for p, li in opts if p == me), None)
            if li is None:
                fids = [li["fid"] for _, li in opts]
                fid = max(set(fids), key=fids.count)
                li = next(li for _, li in opts if li["fid"] == fid)
            lays[lay] = dict(li, n=0, bytes=0, cov=[])
        arrays[n] = {k: base[k] for k in ("vfid", "dims", "taxis", "fmt")} | {"attrs": base.get("attrs", {}),
                                                                             "layouts": lays}
    cand, parsed = {}, {}
    pcoll: dict[str, dict] = {}
    for p, m in mans.items():
        ok = {n: a["vfid"] == arrays[n]["vfid"] for n, a in m["arrays"].items() if n in arrays}
        for k, e in m["chunks"].items():
            cid, vcid, size, nv = e[:4]
            if par.is_parity(k.partition("@")[0]):  # erasure-stripe parity entry: [cid, "", size, 0, members]
                if len(e) > 4 and isinstance(e[4], list):
                    pc = pcoll.setdefault(k, {"holders": [], "members": e[4]})
                    pc["holders"].append((p, cid, size))
                continue
            sk = parsed.get(k)
            if sk is None:
                sk = parsed[k] = split_key(k)
            n, lay, _ = sk
            if ok.get(n) and lay in arrays[n]["layouts"]:
                cand.setdefault(k, []).append((nv, vcid, cid, size, p))
    best = {}
    for k, cs in cand.items():
        if len(cs) == 1:  # common case: single holder
            c = cs[0]
            best[k] = {"vcid": c[1], "nv": c[0], "src": [(c[4], c[2], c[3])]}
            continue
        # value identity by holder majority (then more valid steps): one lying peer cannot override honest ones.
        # ponytail: Sybil peers can still outvote; add trusted publisher keys when that matters
        score, rep = {}, {}
        for nv, vcid, _, _, p in cs:
            # lattice ids of copies decoded by different software are equivalent, not equal: one group per value
            g = rep.setdefault(vcid, next((r for r in score if codec.same_vcid(r, vcid)), vcid))
            t, h, n = score.get(g, (0, 0, 0))
            score[g] = (t + (p in trusted or p == me), h + 1, max(n, nv))
        ranked = sorted(score, key=lambda x: score[x], reverse=True)
        vc = ranked[0]
        if len(ranked) > 1 and score[ranked[1]] == score[vc]:
            # equally trusted, equally many holders, equally complete, different values: nothing tells the honest
            # value apart (a lexicographic tie-break let a liar win by grinding its hash). Serve nothing for this
            # chunk until a strict majority or a trusted publisher decides; the job reports it missing.
            best[k] = {"vcid": vc, "nv": score[vc][2], "src": [], "contested": len(ranked)}
            continue
        best[k] = {"vcid": vc, "nv": score[vc][2], "src": [(c[4], c[2], c[3]) for c in cs if rep[c[1]] == vc]}
        if len(ranked) > 1 and score[ranked[1]][:2] == score[vc][:2]:
            # decided by completeness alone (growing data: the newer copy has more valid steps). A liar could pad
            # missing cells to win, so the receiver also fetches the rival and accepts only a superset of it
            rv = ranked[1]
            best[k]["rival"] = {"vcid": rv, "src": [(c[4], c[2], c[3]) for c in cs if rep[c[1]] == rv]}
    # chunks whose holders are all gone but whose stripe survives (RS rows from different volunteers +
    # surviving members) are restorable (ZTP-EC). Rows of one stripe form a family.
    fams: dict[str, dict] = {}
    for pk, pc in pcoll.items():
        pname, lay, co = split_key(pk)
        base, K, D, row, O, kind = par.base_of(pname)
        a = arrays.get(base)
        if not a or a["taxis"] is None or lay not in a["layouts"]:
            continue
        fk = f"{base}|{kind}|{K}|{D}|{O}|{lay}|{'.'.join(map(str, co))}"
        f_ = fams.setdefault(fk, {"base": base, "k": K, "d": D, "o": O, "lay": lay, "co": co, "kind": kind,
                                  "members": pc["members"], "rows": {}})
        f_["rows"][row] = {"pkey": pk, "holders": pc["holders"]}
    covered = _time_coverage(best, arrays, parsed)  # var -> merged [g0, g1) intervals over ALL layouts
    for fk, f_ in fams.items():
        a = arrays[f_["base"]]
        li = a["layouts"][f_["lay"]]
        ct, ph = li["chunks"][a["taxis"]], li["phase"]
        for i, mem in enumerate(f_["members"]):
            if not mem or mem[2] < 0:
                continue
            dco = list(f_["co"])
            dco[a["taxis"]] = par.member_of(f_["co"][a["taxis"]], i, f_["k"], f_["d"], f_["o"])
            dk = f"{f_['base']}@{f_['lay']}/" + ".".join(map(str, dco))
            if f_["kind"] == "v":  # value parity: missing only if NO layout covers this time range any more
                g0 = dco[a["taxis"]] * ct + ph
                if _covers(covered.get(f_["base"], []), g0, g0 + mem[3]):
                    continue
            if dk not in best:
                best[dk] = {"vcid": mem[1], "nv": mem[3], "src": [], "restore": [{"fam": fk, "i": i}]}
                parsed[dk] = (f_["base"], f_["lay"], dco)
            elif best[dk].get("restore") is not None:
                best[dk]["restore"].append({"fam": fk, "i": i})
    tcs = {}
    for k, b in best.items():
        n, lay, co = parsed[k]
        li = arrays[n]["layouts"][lay]
        li["n"] += 1
        li["bytes"] += b["src"][0][2] if b["src"] else 0
        if arrays[n]["taxis"] is not None:
            tcs.setdefault((n, lay), set()).add(co[arrays[n]["taxis"]])
    for (n, lay), cs in tcs.items():
        arrays[n]["layouts"][lay]["cov"] = _ranges(sorted(cs))
    ext = local_extent({"arrays": arrays, "chunks": {k: [None, None, None, b["nv"]] for k, b in best.items()}})
    return {"grid_id": grid, "grid": first["grid"], "gdocs": first["gdocs"], "fmt": fmt, "arrays": arrays,
            "pinfo": pinfo, "best": best, "gmin": ext[0] if ext else None, "gmax": ext[1] if ext else None,
            "parity": pcoll, "fams": fams, "covered": covered}
