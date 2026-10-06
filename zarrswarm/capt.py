"""CAPT - Coordinate-Addressed Prolly Tree for chunk manifests.

Leaves are manifest entries ordered by array coordinates (var, layout, time index, space index...).
Page boundaries are content-defined (a leaf ends a page when H(key) mod Q == 0; inner levels use the
child's first key), so an append or a local change rewrites only O(1) pages per level. Pages are addressed
by their hash, so a client that cached an older tree fetches only the pages it has not seen: O(delta * log N)
instead of the whole manifest. The signed root travels in the small manifest head.
"""
import hashlib
import json
import zlib

from .scan import split_key

Q = 64  # expected fan-out


def sort_key(ckey: str):
    n, lay, co = split_key(ckey)
    return (n, lay, co)


def _h(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()


def _boundary(key: str) -> bool:
    return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % Q == 0


def _page(obj) -> tuple[str, bytes]:
    b = zlib.compress(json.dumps(obj, separators=(",", ":")).encode(), 6)
    return _h(b), b


def build(entries: dict[str, list]) -> tuple[str, dict[str, bytes]]:
    """entries: ckey -> value. Returns (root hash, {page hash: compressed page})."""
    pages: dict[str, bytes] = {}
    level = [[k, entries[k]] for k in sorted(entries, key=sort_key)]
    leaf = True
    while True:
        groups, cur = [], []
        for item in level:
            cur.append(item)
            if _boundary(item[0]):
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        nxt = []
        for g in groups:
            h, b = _page({"leaf": leaf, "items": g})
            pages[h] = b
            nxt.append([g[0][0], h])  # inner entry: (first key, child hash)
        if len(nxt) == 1:
            return nxt[0][1], pages
        level, leaf = nxt, False


def load(root: str, get_page, cache: dict) -> tuple[dict[str, list], int]:
    """Rebuild the full entry map from `root`, fetching only pages missing from `cache`.
    get_page(hash) -> compressed bytes. Returns (entries, bytes fetched)."""
    fetched = 0
    out: dict[str, list] = {}
    stack = [root]
    while stack:
        h = stack.pop()
        b = cache.get(h)
        if b is None:
            b = get_page(h)
            if _h(b) != h:
                raise ValueError(f"page {h} hash mismatch")
            cache[h] = b
            fetched += len(b)
        page = json.loads(zlib.decompress(b))
        if page["leaf"]:
            out.update({k: v for k, v in page["items"]})
        else:
            stack.extend(child for _, child in page["items"])
    return out, fetched


def load_range(root: str, get_page, cache: dict, lo, hi) -> tuple[dict[str, list], int]:
    """Entries with lo <= sort_key(key) <= hi only. Inner pages keep each child's first key, so a child
    [first_i, first_{i+1}) is visited only if it can overlap [lo, hi]: O(log N + pages in range)."""
    fetched, out, stack = 0, {}, [root]
    while stack:
        h = stack.pop()
        b = cache.get(h)
        if b is None:
            b = get_page(h)
            if _h(b) != h:
                raise ValueError(f"page {h} hash mismatch")
            cache[h] = b
            fetched += len(b)
        page = json.loads(zlib.decompress(b))
        items = page["items"]
        if page["leaf"]:
            out.update({k: v for k, v in items if lo <= sort_key(k) <= hi})
            continue
        for i, (first, child) in enumerate(items):
            nxt = sort_key(items[i + 1][0]) if i + 1 < len(items) else None
            if sort_key(first) > hi or (nxt is not None and nxt <= lo):
                continue
            stack.append(child)
    return out, fetched


if __name__ == "__main__":
    import os
    import time
    # self-check + incremental-sync measurement: 146k chunks, then one more day appended to every variable
    def entries(days):
        return {f"v{i}@24x181x360+0/{c}.0.0": [os.urandom(16).hex(), os.urandom(16).hex(), 250000, 24]
                for i in range(10) for c in range(days)}
    base = entries(14600)
    t0 = time.perf_counter()
    root1, pages1 = build(base)
    tb = time.perf_counter() - t0
    full = sum(len(b) for b in pages1.values())
    cache = {}
    got, f1 = load(root1, pages1.__getitem__, cache)
    assert got == base and f1 == full
    grown = dict(base)
    grown.update({f"v{i}@24x181x360+0/14600.0.0": ["n" * 32, "m" * 32, 250000, 24] for i in range(10)})
    root2, pages2 = build(grown)
    got2, f2 = load(root2, pages2.__getitem__, cache)
    assert got2 == grown
    print(f"CAPT: {len(base)} chunks, {len(pages1)} pages, build {tb:.2f}s, full tree {full / 1e6:.2f} MB; "
          f"after appending 10 chunks the client fetched {f2 / 1e3:.1f} KB ({100 * f2 / full:.2f}% of full)")
    assert f2 < 0.05 * full
    # range read: one month (30 daily chunks) of one variable out of 40 years x 10 variables
    lo, hi = ("v3", "24x181x360+0", [7000, 0, 0]), ("v3", "24x181x360+0", [7029, 0, 0])
    part, f3 = load_range(root1, pages1.__getitem__, {}, lo, hi)
    assert len(part) == 30 and all(lo <= sort_key(k) <= hi for k in part)
    print(f"CAPT range read: 30 of {len(base)} entries, fetched {f3 / 1e3:.1f} KB "
          f"({100 * f3 / full:.2f}% of the full manifest)")
    assert f3 < 0.02 * full
