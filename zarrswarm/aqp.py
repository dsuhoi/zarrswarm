"""Anytime estimators for progressive downloads (pure, network-free; shared by the online code and benchmarks).

RatioVDC : randomized systematic (van der Corput order with a random cyclic start) + ratio estimator.
VAS      : stratified by time blocks, pilot per stratum, sequential greedy Neyman allocation,
           Satterthwaite degrees of freedom + Student-t quantile, minimum n_h before stopping.
"""
import math

import numpy as np


def t_quantile(z: float, df: float) -> float:
    """Cornish-Fisher expansion of the Student-t quantile around the normal one (good for df >= 3)."""
    if not math.isfinite(df) or df > 1e6:
        return z
    df = max(df, 1.0)
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2) \
        + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3)


def vdc_order(n: int, shift: int = 0) -> list[int]:
    """Bit-reversal permutation of range(n) applied after a cyclic shift (randomized systematic design)."""
    if n < 2:
        return list(range(n))
    bits = (n - 1).bit_length()
    base = sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))
    return [(i + shift) % n for i in base]


class RatioVDC:
    def __init__(self, N: int, z: float = 1.96, min_n: int = 8):
        self.N, self.z, self.min_n = N, z, min_n
        self.ys, self.xs = [], []

    def add(self, y: float, x: int):
        if x:
            self.ys.append(y)
            self.xs.append(x)

    def estimate(self) -> tuple[float, float, int]:
        n = len(self.xs)
        if n < 2:
            return float("nan"), float("inf"), n
        y, x = np.array(self.ys), np.array(self.xs, dtype=float)
        R = y.sum() / x.sum()
        s2 = ((y - R * x) ** 2).sum() / (n - 1)
        sd = math.sqrt(max(1 - n / self.N, 0) * s2 / (n * x.mean() ** 2))
        return R, t_quantile(self.z, n - 1) * sd, n

    def done(self, rel_err: float) -> bool:
        R, hw, n = self.estimate()
        return n >= self.min_n and hw <= rel_err * abs(R)


class VAS:
    def __init__(self, sizes: list[int], z: float = 1.96, pilot: int = 5, min_h: int = 5):
        self.N_h = np.array(sizes, dtype=float)
        self.W = self.N_h / self.N_h.sum()
        self.z, self.pilot, self.min_h = z, pilot, min_h
        self.m = [[] for _ in sizes]           # chunk-level means per stratum
        self.taken = np.zeros(len(sizes), dtype=int)

    def pilot_batch(self) -> list[int]:
        out = []
        for h in range(len(self.N_h)):
            k = int(min(self.pilot, self.N_h[h]))
            out += [h] * k
        self.taken += np.bincount(out, minlength=len(self.N_h)) if out else 0
        return out

    def add(self, h: int, mean: float):
        self.m[h].append(mean)

    def _s2(self) -> np.ndarray:
        s2 = np.array([np.var(m, ddof=1) if len(m) > 1 else np.nan for m in self.m])
        pooled = np.nanmax(s2) if np.isfinite(s2).any() else 1.0
        return np.where(np.isfinite(s2), s2, pooled)  # unknown spread -> worst seen (conservative)

    def estimate(self) -> tuple[float, float, int]:
        n_h = np.array([len(m) for m in self.m], dtype=float)
        if (n_h == 0).any():
            return float("nan"), float("inf"), int(n_h.sum())
        ybar = np.array([np.mean(m) for m in self.m])
        s2 = self._s2()
        f = np.clip(self.taken / self.N_h, 0, 1)
        a = self.W ** 2 * (1 - f) / n_h
        var = float((a * s2).sum())
        den = float(((a * s2) ** 2 / np.maximum(n_h - 1, 1)).sum())
        df = var ** 2 / den if den > 0 else float("inf")
        return float((self.W * ybar).sum()), t_quantile(self.z, df) * math.sqrt(max(var, 0.0)), int(n_h.sum())

    def done(self, rel_err: float) -> bool:
        if (self.taken >= self.N_h).all():
            return True
        mean, hw, _ = self.estimate()
        enough = all(len(m) >= min(self.min_h, self.N_h[h]) for h, m in enumerate(self.m))
        return enough and hw <= rel_err * abs(mean)

    def next_batch(self, size: int) -> list[int]:
        """Greedy Neyman: strata with the largest marginal variance reduction W_h^2 s_h^2 (1/n - 1/(n+1)),
        topping up strata below min_h first."""
        s2 = self._s2()
        n = np.maximum(self.taken.astype(float), 1)
        out = []
        for _ in range(size):
            room = self.taken < self.N_h
            if not room.any():
                break
            short = room & (self.taken < self.min_h)
            if short.any():
                h = int(np.argmax(np.where(short, self.W, -1)))
            else:
                gain = np.where(room, self.W ** 2 * s2 * (1 / n - 1 / (n + 1)), -1.0)
                h = int(np.argmax(gain))
            out.append(h)
            self.taken[h] += 1
            n[h] += 1
        return out


if __name__ == "__main__":
    # offline self-check: coverage on a heteroscedastic population of chunk means must be near nominal
    rng = np.random.default_rng(1)
    pop = np.concatenate([rng.normal(0, 1, 400), rng.normal(0, 12, 80)])  # calm + stormy block
    true = pop.mean()
    cov, used = 0, []
    for trial in range(200):
        strata = np.array_split(np.arange(len(pop)), 8)
        perm = [rng.permutation(s) for s in strata]
        v = VAS([len(s) for s in strata], pilot=5)
        ptr = [0] * 8
        for h in v.pilot_batch():
            v.add(h, pop[perm[h][ptr[h]]])
            ptr[h] += 1
        while not v.done(0.02 * 10):
            for h in v.next_batch(8):
                v.add(h, pop[perm[h][ptr[h]]])
                ptr[h] += 1
        m, hw, n = v.estimate()
        cov += abs(m - true) <= hw
        used.append(n)
    print(f"VAS coverage {cov / 200:.3f}, mean chunks {np.mean(used):.0f}/{len(pop)}")
    assert cov / 200 > 0.87
