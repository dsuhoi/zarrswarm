"""GF(256) Cauchy Reed-Solomon over byte vectors (numpy, table-driven). Systematic MDS code: k data members,
any number of distinct parity rows j; ANY k surviving pieces (data or parity) rebuild the stripe.

Row j (0 <= j < 256 - k) of the Cauchy matrix: C[j, i] = 1 / (x_j XOR y_i), x_j = k + j, y_i = i.
"""
import numpy as np

_EXP = np.zeros(512, dtype=np.uint8)
_LOG = np.zeros(256, dtype=np.int32)
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
_EXP[255:510] = _EXP[:255]


def mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_EXP[_LOG[a] + _LOG[b]])


def inv(a: int) -> int:
    return int(_EXP[255 - _LOG[a]])


def mul_vec(c: int, v: np.ndarray) -> np.ndarray:
    """c * v (elementwise over GF(256)) for a uint8 vector v."""
    if c == 0:
        return np.zeros_like(v)
    out = _EXP[(_LOG[v] + _LOG[c]) % 255].astype(np.uint8)
    out[v == 0] = 0
    return out


def coef(j: int, i: int, k: int) -> int:
    return inv((k + j) ^ i)


def encode_row(j: int, members: list[np.ndarray], k: int) -> np.ndarray:
    """Parity row j over k equal-length uint8 members (absent members = zeros)."""
    acc = np.zeros_like(members[0])
    for i, d in enumerate(members):
        acc ^= mul_vec(coef(j, i, k), d)
    return acc


def _solve(A: list[list[int]], B: list[np.ndarray]) -> list[np.ndarray]:
    """Gauss-Jordan over GF(256): A x = B, A n x n (list of lists), B list of n uint8 vectors."""
    n = len(A)
    A = [row[:] for row in A]
    B = [b.copy() for b in B]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        B[c], B[p] = B[p], B[c]
        iv = inv(A[c][c])
        A[c] = [mul(iv, x) for x in A[c]]
        B[c] = mul_vec(iv, B[c])
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x ^ mul(f, y) for x, y in zip(A[r], A[c])]
                B[r] ^= mul_vec(f, B[c])
    return B


def decode(k: int, data: dict[int, np.ndarray], parity: dict[int, np.ndarray], missing: list[int]) -> dict[int, np.ndarray]:
    """Rebuild data members `missing` from surviving data members and parity rows (need len(parity) >= len(missing))."""
    rows = sorted(parity)[:len(missing)]
    if len(rows) < len(missing):
        raise ValueError("not enough pieces: need k survivors")
    A = [[coef(j, i, k) for i in missing] for j in rows]
    B = []
    for j in rows:
        rhs = parity[j].copy()
        for i, d in data.items():
            rhs ^= mul_vec(coef(j, i, k), d)
        B.append(rhs)
    return dict(zip(missing, _solve(A, B)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k, L = 8, 1000
    data = [rng.integers(0, 256, L, dtype=np.uint8) for _ in range(k)]
    par = {j: encode_row(j, data, k) for j in range(6)}
    for trial in range(200):  # any k survivors out of k data + 6 parity rebuild everything
        lost = sorted(rng.choice(k, size=int(rng.integers(1, 7)), replace=False).tolist())
        rows = sorted(rng.choice(6, size=len(lost), replace=False).tolist())
        got = decode(k, {i: data[i] for i in range(k) if i not in lost}, {j: par[j] for j in rows}, lost)
        assert all(np.array_equal(got[i], data[i]) for i in lost)
    print("gf256 cauchy RS ok: any k of k+6 pieces rebuild the stripe")
