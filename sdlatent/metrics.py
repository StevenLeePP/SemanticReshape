from __future__ import annotations

import numpy as np


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    return float(np.mean((ref - est) ** 2) / (np.mean(ref**2) + 1e-12))


def db(x: float) -> float:
    return float(10.0 * np.log10(max(x, 1e-12)))
