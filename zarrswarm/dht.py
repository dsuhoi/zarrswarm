"""Kademlia DHT over HTTP/JSON (so it shares the data port and can traverse the relay).

Values are signed records, multi-valued per key (one per publisher):
  {"t": "peer", "grid": <grid_id>, "node": <id>, "addr": <url>, "ts": ...}   key = grid_id
  {"t": "name", "name": ..., "target": <link>, "seq": int, ...}             key = h160(pk + name)
Nodes behind NAT (ro=True) query and publish but are never put into routing tables (BEP 43).
"""
import asyncio
import re
import time

import aiohttp

from .common import check_signed, h160

K = 8
ALPHA = 3
TTL = 3600
MAX_PER_KEY = 2000
MAX_KEYS = 100_000
_ID = re.compile(r"^[0-9a-f]{40}$")


def valid_contact(c) -> bool:
    return (isinstance(c, dict) and isinstance(c.get("id"), str) and bool(_ID.match(c["id"]))
            and isinstance(c.get("addr"), str) and c["addr"].startswith(("http://", "https://")) and len(c["addr"]) < 512)


def dist(a: str, b: str) -> int:
    return int(a, 16) ^ int(b, 16)


class DHT:
    def __init__(self, ident, addr: str, session: aiohttp.ClientSession, ro: bool = False):
        self.ident, self.session = ident, session
        self.contact = {"id": ident.id, "addr": addr, "ro": ro}
        self.buckets: list[list[dict]] = [[] for _ in range(160)]
        self.values: dict[str, dict[str, tuple[dict, float]]] = {}

    # ---- routing table
    def add(self, c: dict | None):
        if not valid_contact(c) or c.get("ro") or c["id"] == self.contact["id"]:
            return
        b = self.buckets[dist(c["id"], self.contact["id"]).bit_length() - 1]
        b[:] = [x for x in b if x["id"] != c["id"]]
        b.append({"id": c["id"], "addr": c["addr"]})
        if len(b) > K:
            b.pop(0)  # ponytail: evict oldest without ping-before-evict (weaker churn/Sybil resistance)

    def remove(self, node_id: str):
        for b in self.buckets:
            b[:] = [x for x in b if x["id"] != node_id]

    def contacts(self) -> list[dict]:
        return [c for b in self.buckets for c in b]

    def closest(self, target: str, n: int = K) -> list[dict]:
        return sorted(self.contacts(), key=lambda c: dist(c["id"], target))[:n]

    # ---- values
    def put_local(self, key: str, rec: dict) -> bool:
        now = time.time()
        if not isinstance(rec, dict) or not check_signed(rec):
            return False
        if not isinstance(key, str) or not _ID.match(key):
            return False
        pub = h160(bytes.fromhex(rec["pk"]))
        if rec.get("t") == "peer":
            if rec.get("node") != pub or rec.get("grid") != key:
                return False
            addr, bw = rec.get("addr"), rec.get("bw")
            if not valid_contact({"id": pub, "addr": addr}) or not (bw is None or isinstance(bw, (int, float))):
                return False
        elif rec.get("t") == "name":
            if key != h160(bytes.fromhex(rec["pk"]) + str(rec.get("name")).encode()):
                return False
        elif rec.get("t") == "idx":
            if rec.get("node") != pub or key != h160(("tag:" + str(rec.get("tag"))).encode()):
                return False
            tr = rec.get("tr")
            if not isinstance(rec.get("grid"), str) or not _ID.match(rec["grid"]) or not isinstance(rec.get("vars"), list) \
                    or not (tr is None or (isinstance(tr, list) and len(tr) == 2 and all(isinstance(x, (int, float)) for x in tr))):
                return False
            pub = f"{pub}:{rec['grid']}"  # one index record per (publisher, grid)
        else:
            return False
        ts = float(rec.get("ts", 0))
        if ts > now + 300 or ts < now - TTL:
            return False
        slot = self.values.get(key)
        if slot is None:
            if len(self.values) >= MAX_KEYS:  # ponytail: global cap, no eviction policy yet
                return False
            slot = self.values[key] = {}
        old = slot.get(pub)
        if old is None and len(slot) >= MAX_PER_KEY:  # existing publishers may always refresh
            self.get_local(key)  # drop expired first
            if len(slot) >= MAX_PER_KEY:
                return False
        if old and rec.get("t") == "name" and old[0].get("seq", 0) > rec.get("seq", 0):
            return True
        if old and old[0].get("ts", 0) > ts:
            return True
        slot[pub] = (rec, ts + TTL)
        return True

    def get_local(self, key: str) -> list[dict]:
        now = time.time()
        slot = self.values.get(key, {})
        for pub in [p for p, (_, exp) in slot.items() if exp < now]:
            del slot[pub]
        return [r for r, _ in slot.values()]

    # ---- rpc
    def handle(self, msg: dict) -> dict:
        if not isinstance(msg, dict):
            return {"me": self.contact}
        self.add(msg.get("from"))
        op, out = msg.get("op"), {"me": self.contact}
        if op in ("find_node", "find_value", "store") and not (
                isinstance(msg.get("target", msg.get("key")), str) and _ID.match(msg.get("target", msg.get("key")))):
            return out
        if op == "find_node":
            out["nodes"] = self.closest(msg["target"])
        elif op == "find_value":
            out["nodes"] = self.closest(msg["target"])
            out["values"] = self.get_local(msg["target"])
        elif op == "store":
            out["ok"] = self.put_local(msg["key"], msg["rec"])
        return out

    async def rpc(self, c: dict, msg: dict, tries: int = 2) -> dict | None:
        res = None
        for attempt in range(tries):  # one retry: lossy links drop single connections
            try:
                async with self.session.post(c["addr"].rstrip("/") + "/dht", json={**msg, "from": self.contact},
                                             timeout=aiohttp.ClientTimeout(total=None, sock_connect=4, sock_read=5)) as r:
                    if r.status != 200:
                        raise aiohttp.ClientError(r.status)
                    res = await r.json()
                break
            except Exception:
                if attempt == tries - 1:
                    if c.get("id"):
                        self.remove(c["id"])
                    return None
        if not isinstance(res, dict):
            return None
        self.add(res.get("me"))
        return res

    async def lookup(self, target: str, values: bool = False) -> tuple[list[dict], list[dict]]:
        short = {c["id"]: c for c in self.closest(target)}
        queried, found = set(), {}
        for r in self.get_local(target) if values else []:
            found[(r["pk"], r.get("ts"))] = r
        while True:
            top = sorted(short.values(), key=lambda c: dist(c["id"], target))[:K]
            batch = [c for c in top if c["id"] not in queried][:ALPHA]
            if not batch:
                break
            op = "find_value" if values else "find_node"
            res = await asyncio.gather(*(self.rpc(c, {"op": op, "target": target}) for c in batch))
            for c, r in zip(batch, res):
                queried.add(c["id"])
                if not r:
                    short.pop(c["id"], None)
                    continue
                for n in r.get("nodes", []) if isinstance(r.get("nodes"), list) else []:
                    if valid_contact(n) and n["id"] != self.contact["id"]:
                        short.setdefault(n["id"], {"id": n["id"], "addr": n["addr"]})
                for v in r.get("values", []) if isinstance(r.get("values"), list) else []:
                    if self.put_local(target, v):
                        found[(v["pk"], v.get("ts"))] = v
        top = [c for c in sorted(short.values(), key=lambda c: dist(c["id"], target)) if c["id"] in queried][:K]
        return top, list(found.values())

    async def store(self, key: str, rec: dict) -> int:
        self.put_local(key, rec)
        nodes, _ = await self.lookup(key)
        res = await asyncio.gather(*(self.rpc(n, {"op": "store", "key": key, "rec": rec}) for n in nodes))
        return sum(1 for r in res if r and r.get("ok"))

    async def get(self, key: str) -> list[dict]:
        await self.lookup(key, values=True)
        return self.get_local(key)

    async def bootstrap(self, urls: list[str]):
        for u in urls:
            await self.rpc({"addr": u}, {"op": "ping"})
        await self.lookup(self.contact["id"])
