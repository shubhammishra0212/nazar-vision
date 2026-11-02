"""The memory bank of good-washer patches: coreset selection and nearest-neighbour distance (numpy only)."""
import time

import numpy as np


def kcenter_coreset(x, k, proj_dim=128, seed=0, log=print):
    """Greedy k-center: repeatedly add the patch that is FARTHEST from everything picked so far.
    The result covers all kinds of good patches with ~10 % of them. Distances are computed on a random
    128-d projection (much faster, nearly the same ordering)."""
    n = len(x)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    p = x @ (rng.standard_normal((x.shape[1], proj_dim)).astype(np.float32) / np.sqrt(proj_dim))
    sq = np.einsum("ij,ij->i", p, p)
    chosen = np.empty(k, np.int64)
    chosen[0] = int(rng.integers(n))
    gap = np.full(n, np.inf, np.float32)
    t0 = time.time()
    for i in range(1, k):
        c = chosen[i - 1]
        np.minimum(gap, sq - 2 * (p @ p[c]) + sq[c], out=gap)
        chosen[i] = int(np.argmax(gap))
        if log and i % 4000 == 0:
            log(f"    coreset {i}/{k}  {time.time() - t0:.0f} s")
    return chosen


class Memory:
    def __init__(self, bank):
        self.bank = np.ascontiguousarray(bank, dtype=np.float32)
        self.bank_sq = np.einsum("ij,ij->i", self.bank, self.bank)

    def __len__(self):
        return len(self.bank)

    def nearest(self, q, chunk=8192):
        """Euclidean distance from each query row to its nearest bank row, in bank chunks (bounded memory)."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        best = np.full(len(q), np.inf, np.float32)
        for i in range(0, len(self.bank), chunk):
            d = self.bank_sq[i:i + chunk][None, :] - 2.0 * (q @ self.bank[i:i + chunk].T)
            np.minimum(best, d.min(1), out=best)
        best += np.einsum("ij,ij->i", q, q)
        return np.sqrt(np.maximum(best, 0.0))
