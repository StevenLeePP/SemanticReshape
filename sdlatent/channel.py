from __future__ import annotations

import numpy as np


def signal_power(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x * x) + 1e-12)


def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add real AWGN using measured signal power."""
    pwr = signal_power(x)
    noise_var = pwr / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(noise_var), size=x.shape)


def bpsk_awgn(symbols: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """AWGN for unit-power BPSK symbols."""
    noise_var = 1.0 / (10.0 ** (snr_db / 10.0))
    return symbols + rng.normal(0.0, np.sqrt(noise_var), size=symbols.shape)
