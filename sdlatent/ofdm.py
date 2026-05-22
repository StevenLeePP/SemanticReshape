from __future__ import annotations

import numpy as np


def real_to_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size % 2:
        x = np.concatenate([x, np.zeros(1, dtype=np.float32)])
    return x[0::2] + 1j * x[1::2]


def complex_to_real(z: np.ndarray, original_len: int) -> np.ndarray:
    out = np.empty(z.size * 2, dtype=np.float32)
    out[0::2] = np.real(z).astype(np.float32)
    out[1::2] = np.imag(z).astype(np.float32)
    return out[:original_len]


def ofdm_awgn_multipath(
    x_real: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    n_fft: int = 256,
    cp: int = 32,
    taps: int = 6,
    perfect_csi: bool = True,
) -> np.ndarray:
    """Simple OFDM transmission with multipath and one-tap equalization.

    This is an intermediate realism check, not a full standard-compliant PHY.
    """
    x_complex = real_to_complex(x_real)
    pad = (-x_complex.size) % n_fft
    if pad:
        x_complex = np.concatenate([x_complex, np.zeros(pad, dtype=np.complex64)])
    grid = x_complex.reshape(-1, n_fft)
    time_symbols = np.fft.ifft(grid, axis=1) * np.sqrt(n_fft)
    tx = np.concatenate([time_symbols[:, -cp:], time_symbols], axis=1).reshape(-1)

    h = (rng.normal(size=taps) + 1j * rng.normal(size=taps)).astype(np.complex64)
    decay = np.exp(-np.arange(taps) / max(taps / 3.0, 1.0))
    h = h * decay
    h = h / np.sqrt(np.sum(np.abs(h) ** 2) + 1e-12)
    rx = np.convolve(tx, h, mode="full")[: tx.size]

    pwr = float(np.mean(np.abs(rx) ** 2) + 1e-12)
    noise_var = pwr / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (rng.normal(size=rx.shape) + 1j * rng.normal(size=rx.shape))
    rx = rx + noise

    rx_symbols = rx.reshape(-1, n_fft + cp)[:, cp:]
    rx_grid = np.fft.fft(rx_symbols, axis=1) / np.sqrt(n_fft)
    if perfect_csi:
        H = np.fft.fft(np.concatenate([h, np.zeros(n_fft - taps, dtype=np.complex64)]))
        rx_grid = rx_grid / (H[None, :] + 1e-8)
    rx_complex = rx_grid.reshape(-1)[: x_complex.size - pad if pad else x_complex.size]
    return complex_to_real(rx_complex, len(x_real))
