"""Bootstrap utilities and simple interval estimates."""
import math

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def binomial_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def bootstrap_mean_ci(values, n_boot=1000, seed=6304, alpha=0.05):
    """Percentile CI for the mean of per-unit values (units = questions or pairs)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot=1000, seed=6304, alpha=0.05):
    """CI for mean(a - b) where a[i], b[i] belong to the same unit (question/pair)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    return bootstrap_mean_ci(a - b, n_boot=n_boot, seed=seed, alpha=alpha)


def bootstrap_indices(n_units, n_boot, seed=6304):
    """Shared resample indices so several metrics/conditions use the same resampled units."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(n_boot, n_units))


def ci_from_indices(values, idx, alpha=0.05):
    v = np.asarray(values, dtype=float)
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
