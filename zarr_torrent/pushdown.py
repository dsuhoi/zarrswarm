"""Optimistic pushdown (ZTP-OP): the holder cuts the requested hyperslab out of a chunk and signs a receipt;
the requester accepts optimistically and audits a random fraction by re-deriving the slice from the full,
vcid-verified chunk. A mismatching signed receipt is a transferable fraud proof.

Receipt = sign(pk_holder, cjson({"g": grid, "k": key, "cid": cid, "sel": sel, "h": blake2b(result)}))
"""
import hashlib
from collections import OrderedDict
from threading import Lock

import numpy as np

from . import codec
from .common import cjson, verify

MAX_ITEMS = 256


def receipt_body(grid: str, key: str, cid: str, sel: list, result: bytes) -> bytes:
    return cjson({"g": grid, "k": key, "cid": cid, "sel": sel,
                  "h": hashlib.blake2b(result, digest_size=16).hexdigest()})


def check_receipt(pk: str, sig: str, grid: str, key: str, cid: str, sel: list, result: bytes) -> bool:
    return verify(pk, sig, receipt_body(grid, key, cid, sel, result))


def to_slices(sel: list) -> tuple:
    return tuple(slice(int(lo), int(hi)) for lo, hi in sel)


def cut(values: np.ndarray, sel: list) -> bytes:
    """Canonical result encoding: C-order, little-endian raw values of the hyperslab."""
    v = np.ascontiguousarray(values[to_slices(sel)])
    return v.astype(v.dtype.newbyteorder("<"), copy=False).tobytes()


class DecodedLRU:
    """Holder-side cache: a burst of point queries against one chunk decodes it once."""

    def __init__(self, max_bytes: int = 256 << 20):
        self.d: OrderedDict[str, np.ndarray] = OrderedDict()
        self.n, self.max, self.lock = 0, max_bytes, Lock()

    def get(self, key: str, docs: dict, raw_fn) -> np.ndarray:
        with self.lock:
            hit = self.d.get(key)
            if hit is not None:
                self.d.move_to_end(key)
                return hit
        vals = codec.decode(docs, raw_fn())
        with self.lock:
            self.d[key] = vals
            self.n += vals.nbytes
            while self.n > self.max and len(self.d) > 1:
                _, old = self.d.popitem(last=False)
                self.n -= old.nbytes
        return vals
