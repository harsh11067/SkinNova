"""Bootstrap 95% CIs (1,000 resamples) for any per-sample metric function."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(metric, *arrays, n: int = 1000, seed: int = 3407, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    point = float(metric(*arrays))
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        vals.append(metric(*[a[idx] for a in arrays]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"value": point, "ci95": [float(lo), float(hi)], "n": int(m)}
