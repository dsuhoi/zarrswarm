"""Erasure stripes for availability (ZTP-EC) with Cauchy Reed-Solomon over GF(256).

A stripe = k data chunks of one (variable, layout, spatial tile). Each volunteer stores ONE distinct parity row j
(1/k of the data); any k surviving pieces - data members or parity rows held by different volunteers - rebuild
every member (MDS). The parity header carries every member's cid/vcid/length/nvalid, so a rebuilt chunk is
verified exactly like a downloaded one.

Interleaving: replicas usually hold *contiguous* time windows, so a dead seeder erases a burst of neighbours.
Stripes take members with stride D: stripe (q, s) = {c = q*k*D + i*D + s, i = 0..k-1}; a burst shorter than D
hits at most one member per stripe. Published as pseudo-variable "<var>#p<k>x<D>r<row>"; its time coordinate
is q*D + s. Views and the metadata index ignore '#' variables.
"""
import json

import numpy as np

from . import gf

MAGIC = b"ZTR1"


def is_parity(name: str) -> bool:
    return "#p" in name or "#v" in name


def parity_name(var: str, k: int, d: int = 1, row: int = 0, o: int = 0, kind: str = "p") -> str:
    """kind "p": parity over stored bytes of one layout; "v": over canonical-grid VALUES (layout-agnostic)."""
    return f"{var}#{kind}{k}x{d}o{o}r{row}"


def base_of(name: str) -> tuple[str, int, int, int, int, str]:
    """-> (var, k, D, row, offset o, kind)."""
    kind = "v" if "#v" in name else "p"
    var, _, rest = name.partition("#" + kind)
    k, _, rest = rest.partition("x")
    d, _, rest = rest.partition("o")
    o, _, r = rest.partition("r")
    return var, int(k), int(d or 1), int(r or 0), int(o or 0), kind


def stripe_of(c: int, k: int, d: int, o: int = 0) -> tuple[int, int]:
    """data time index c -> (stripe time coordinate q*D + s, member index i). o = extent start shared by
    everyone who reads the same swarm view, so stripes are full (no half-empty blocks from absolute alignment)."""
    q, r = divmod(c - o, k * d)
    i, s = divmod(r, d)
    return q * d + s, i


def member_of(t: int, i: int, k: int, d: int, o: int = 0) -> int:
    """stripe time coordinate t = q*D + s, member i -> data time index."""
    q, s = divmod(t, d)
    return q * k * d + i * d + s + o


def encode(members: list[tuple], row: int = 0, mode: str = "bytes") -> bytes:
    """Parity row `row` over the stripe (members padded to the longest; absent members = zeros).
    members: [(cid, vcid, nvalid, payload bytes or None[, per-slice lattice params])] in stripe order.
    mode "codes": payloads are lattice codes (codec.lattice_codes), "bytes": raw bytes."""
    k = len(members)
    L = max((len(b) for _, _, _, b, *_x in members if b is not None), default=0)
    pad = [np.frombuffer(b.ljust(L, b"\0"), dtype=np.uint8) if b is not None else np.zeros(L, np.uint8)
           for _, _, _, b, *_x in members]
    body = gf.encode_row(row, pad, k) if L else np.zeros(0, np.uint8)
    hdr = json.dumps({"m": [[c, v, (len(b) if b is not None else -1), nv] + list(x) for c, v, nv, b, *x in members],
                      "row": row, "mode": mode}).encode()
    return MAGIC + len(hdr).to_bytes(4, "big") + hdr + body.tobytes()


def _parse(blob: bytes):
    n = int.from_bytes(blob[4:8], "big")
    return json.loads(blob[8:8 + n]), np.frombuffer(blob[8 + n:], dtype=np.uint8)


def header(blob: bytes) -> list:
    return _parse(blob)[0]["m"]


def mode(blob: bytes) -> str:
    return _parse(blob)[0].get("mode", "bytes")


def restore_many(blobs: list[bytes], present: dict[int, bytes], missing: list[int]) -> dict[int, tuple[bytes, str, str]]:
    """Rebuild members `missing` of one stripe from parity rows (`blobs`, distinct rows) and the present members'
    stored bytes. Needs len(blobs) >= len(missing). Returns {i: (bytes, cid, vcid)}."""
    heads = [_parse(b) for b in blobs]
    if not heads:
        raise ValueError("no parity rows")
    m = heads[0][0]["m"]
    k = len(m)
    L = len(heads[0][1])
    if any(h["m"] != m or h.get("mode", "bytes") != heads[0][0].get("mode", "bytes") or len(body) != L
           for h, body in heads) or len({h["row"] for h, _ in heads}) != len(heads):
        raise ValueError("inconsistent stripe headers or duplicate rows")
    data = {i: np.frombuffer(b.ljust(L, b"\0"), dtype=np.uint8) for i, b in present.items()}
    for i, (_, _, ln, _, *_x) in enumerate(m):  # members that never existed were encoded as zeros
        if ln < 0 and i not in missing:
            data.setdefault(i, np.zeros(L, np.uint8))
    rows = {h["row"]: body for h, body in heads}
    got = gf.decode(k, data, rows, list(missing))
    return {i: (got[i].tobytes()[:m[i][2]], m[i][0], m[i][1]) for i in missing}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = 8
    data = [rng.bytes(int(rng.integers(50, 200))) for _ in range(k)]
    mem = [(f"c{i}", f"v{i}", 24, b) for i, b in enumerate(data)]
    blobs = [encode(mem, row=j) for j in range(4)]
    for lost in ([0], [3, 5], [1, 2, 6], [0, 4, 6, 7]):
        got = restore_many(blobs[:len(lost)], {i: b for i, b in enumerate(data) if i not in lost}, lost)
        assert all(got[i][0] == data[i] and got[i][1] == f"c{i}" for i in lost)
    for c in range(1000):
        for kk, d, o in ((5, 1, 0), (4, 7, 3), (8, 13, 417)):
            t, i = stripe_of(c + o, kk, d, o)
            assert member_of(t, i, kk, d, o) == c + o
    print(f"RS parity ok: any {k} of {k}+4 pieces; one row = {len(blobs[0])} B for {sum(map(len, data))} B data")
