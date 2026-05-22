from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import bpsk_awgn, signal_power


def _bits_for_classes(num_classes: int) -> int:
    return int(np.ceil(np.log2(num_classes)))


def _id_to_bits(shape_id: int, width: int) -> np.ndarray:
    return np.array([(shape_id >> i) & 1 for i in range(width - 1, -1, -1)], dtype=np.int8)


def _bits_to_id(bits: np.ndarray) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b > 0)
    return out


@dataclass
class DecodeResult:
    shape_id: int
    confidence: float


class CompactDigitalHeader:
    """Separate conventional BPSK metadata channel.

    This models the fragile "traditional info tells tensor dimensions" path.
    """

    name = "digital_header"

    def __init__(self, num_classes: int, repeat: int = 1):
        self.num_classes = num_classes
        self.width = _bits_for_classes(num_classes)
        self.repeat = int(repeat)

    @property
    def overhead_symbols(self) -> int:
        return self.width * self.repeat

    def transmit_decode(self, shape_id: int, snr_db: float, rng: np.random.Generator) -> DecodeResult:
        bits = _id_to_bits(shape_id, self.width)
        symbols = 2.0 * np.repeat(bits, self.repeat).astype(np.float32) - 1.0
        rx = bpsk_awgn(symbols, snr_db, rng)
        rx = rx.reshape(self.width, self.repeat)
        soft = rx.mean(axis=1)
        decoded = _bits_to_id(soft > 0.0)
        margin = float(np.min(np.abs(soft)))
        if decoded >= self.num_classes:
            decoded = -1
        return DecodeResult(decoded, margin)


class PrefixBpskHeader:
    """Overwrite the first latent samples with a repeated BPSK shape ID."""

    name = "prefix_bpsk_in_latent"

    def __init__(self, num_classes: int, repeat: int = 32, amplitude: float = 1.0):
        self.num_classes = num_classes
        self.width = _bits_for_classes(num_classes)
        self.repeat = int(repeat)
        self.amplitude = float(amplitude)

    @property
    def used_samples(self) -> int:
        return self.width * self.repeat

    def embed(self, x: np.ndarray, shape_id: int) -> np.ndarray:
        y = np.array(x, copy=True)
        bits = _id_to_bits(shape_id, self.width)
        symbols = 2.0 * np.repeat(bits, self.repeat).astype(np.float32) - 1.0
        scale = np.sqrt(signal_power(x))
        y[: self.used_samples] = self.amplitude * scale * symbols
        return y

    def decode(self, y: np.ndarray) -> DecodeResult:
        rx = y[: self.used_samples].reshape(self.width, self.repeat)
        soft = rx.mean(axis=1)
        decoded = _bits_to_id(soft > 0.0)
        margin = float(np.min(np.abs(soft)))
        if decoded >= self.num_classes:
            decoded = -1
        return DecodeResult(decoded, margin)

    def remove(self, y: np.ndarray, shape_id: int) -> np.ndarray:
        # Prefix overwrite is not reversible without a copy of the original host.
        return np.array(y, copy=True)


class SpreadSpectrumID:
    """Additive direct-sequence spread-spectrum shape watermark.

    A full-length pseudo-random codeword represents each tensor shape. The
    receiver only needs the flattened vector length and the shared seed.
    """

    name = "spread_spectrum_shape_id"

    def __init__(self, num_classes: int, length: int, alpha: float = 0.08, seed: int = 2026):
        self.num_classes = num_classes
        self.length = int(length)
        self.alpha = float(alpha)
        rng = np.random.default_rng(seed)
        codes = rng.choice([-1.0, 1.0], size=(num_classes, self.length)).astype(np.float32)
        codes = codes / np.linalg.norm(codes, axis=1, keepdims=True)
        self.codes = codes

    @property
    def overhead_symbols(self) -> int:
        return 0

    def embed(self, x: np.ndarray, shape_id: int) -> np.ndarray:
        scale = np.sqrt(signal_power(x)) * np.sqrt(self.length)
        return x + self.alpha * scale * self.codes[shape_id]

    def decode(self, y: np.ndarray) -> DecodeResult:
        scores = self.codes @ y.reshape(-1).astype(np.float32)
        order = np.argsort(scores)
        best = int(order[-1])
        second = float(scores[order[-2]]) if self.num_classes > 1 else 0.0
        return DecodeResult(best, float(scores[best] - second))

    def remove(self, y: np.ndarray, shape_id: int) -> np.ndarray:
        scale = np.sqrt(signal_power(y)) * np.sqrt(self.length)
        return y - self.alpha * scale * self.codes[shape_id]


class SpreadSpectrumShapeFields:
    """Spread-spectrum watermark for numeric (H, W, C) shape fields.

    This avoids treating shape recovery as a small classification problem. The
    payload is 8-bit H, 8-bit W, 8-bit C, and CRC8, for 32 embedded bits.
    """

    name = "spread_spectrum_shape_fields"

    def __init__(self, length: int, alpha: float = 0.08, seed: int = 2028, num_bits: int = 32):
        self.length = int(length)
        self.alpha = float(alpha)
        self.num_bits = int(num_bits)
        if self.length < self.num_bits:
            raise ValueError("latent length must be at least the number of payload bits")
        rng = np.random.default_rng(seed)
        bounds = np.linspace(0, self.length, self.num_bits + 1, dtype=np.int64)
        self.slices = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(self.num_bits)]
        self.codes = []
        for sl in self.slices:
            block_len = sl.stop - sl.start
            code = rng.choice([-1.0, 1.0], size=block_len).astype(np.float32)
            code = code / (np.linalg.norm(code) + 1e-12)
            self.codes.append(code)

    @property
    def overhead_symbols(self) -> int:
        return 0

    def embed_bits(self, x: np.ndarray, bits: np.ndarray) -> np.ndarray:
        y = np.array(x, copy=True)
        scale = np.sqrt(signal_power(x))
        bits = np.asarray(bits[: self.num_bits], dtype=np.uint8)
        for bit, sl, code in zip(bits, self.slices, self.codes):
            symbol = 1.0 if int(bit) else -1.0
            block_len = sl.stop - sl.start
            y[sl] += symbol * self.alpha * scale * np.sqrt(block_len) * code
        return y

    def decode_bits(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        bits = []
        margins = []
        y = y.reshape(-1).astype(np.float32)
        for sl, code in zip(self.slices, self.codes):
            score = float(np.dot(y[sl], code))
            bits.append(1 if score >= 0.0 else 0)
            margins.append(abs(score))
        return np.array(bits, dtype=np.uint8), float(np.min(margins))

    def embed_shape(self, x: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        from .ldpc import shape_to_payload

        return self.embed_bits(x, shape_to_payload(shape))

    def decode_shape(self, y: np.ndarray) -> DecodeResult:
        from .ldpc import payload_to_shape

        bits, confidence = self.decode_bits(y)
        shape = payload_to_shape(bits)
        if shape is None:
            return DecodeResult(-1, confidence)
        # Store the numeric shape in a dynamic attribute for callers that need it.
        result = DecodeResult(0, confidence)
        result.shape = shape  # type: ignore[attr-defined]
        return result

    def remove_shape(self, y: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        from .ldpc import shape_to_payload

        bits = shape_to_payload(shape)
        cleaned = np.array(y, copy=True)
        scale = np.sqrt(signal_power(y))
        for bit, sl, code in zip(bits, self.slices, self.codes):
            symbol = 1.0 if int(bit) else -1.0
            block_len = sl.stop - sl.start
            cleaned[sl] -= symbol * self.alpha * scale * np.sqrt(block_len) * code
        return cleaned


class SpreadSpectrumShapeComponents:
    """Recover H, W, C as separate semantic-watermarked components.

    This is a middle ground between a full shape-ID codebook and fragile raw
    bit fields. It composes independent codebooks for H, W, and C, so the
    receiver can represent many shape combinations without enumerating every
    full tensor shape as a class.
    """

    name = "spread_spectrum_shape_components"

    def __init__(
        self,
        length: int,
        h_values: list[int],
        w_values: list[int],
        c_values: list[int],
        alpha: float = 0.08,
        seed: int = 2029,
    ):
        self.length = int(length)
        self.values = [list(h_values), list(w_values), list(c_values)]
        self.alpha = float(alpha)
        bounds = np.linspace(0, self.length, 4, dtype=np.int64)
        self.slices = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(3)]
        rng = np.random.default_rng(seed)
        self.codes: list[np.ndarray] = []
        for values, sl in zip(self.values, self.slices):
            block_len = sl.stop - sl.start
            codes = rng.choice([-1.0, 1.0], size=(len(values), block_len)).astype(np.float32)
            codes = codes / np.linalg.norm(codes, axis=1, keepdims=True)
            self.codes.append(codes)

    @property
    def overhead_symbols(self) -> int:
        return 0

    def embed_shape(self, x: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        y = np.array(x, copy=True)
        scale = np.sqrt(signal_power(x))
        for value, values, sl, codes in zip(shape, self.values, self.slices, self.codes):
            if value not in values:
                raise ValueError(f"value {value} not in allowed component set {values}")
            idx = values.index(value)
            block_len = sl.stop - sl.start
            y[sl] += self.alpha * scale * np.sqrt(block_len) * codes[idx]
        return y

    def decode_shape(self, y: np.ndarray) -> DecodeResult:
        decoded = []
        margins = []
        y = y.reshape(-1).astype(np.float32)
        for values, sl, codes in zip(self.values, self.slices, self.codes):
            scores = codes @ y[sl]
            order = np.argsort(scores)
            best = int(order[-1])
            second = float(scores[order[-2]]) if len(values) > 1 else 0.0
            decoded.append(values[best])
            margins.append(float(scores[best] - second))
        result = DecodeResult(0, float(min(margins)))
        result.shape = tuple(decoded)  # type: ignore[attr-defined]
        return result

    def remove_shape(self, y: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        cleaned = np.array(y, copy=True)
        scale = np.sqrt(signal_power(y) / (1.0 + self.alpha**2))
        for value, values, sl, codes in zip(shape, self.values, self.slices, self.codes):
            idx = values.index(value)
            block_len = sl.stop - sl.start
            cleaned[sl] -= self.alpha * scale * np.sqrt(block_len) * codes[idx]
        return cleaned


class RobustSpreadSpectrumShapeComponents:
    """Interleaved, diversity-combined component watermark.

    Design choices:
    - random interleaving spreads each component over the whole latent, which
      helps against OFDM frequency-selective fades;
    - multiple replicas per component provide diversity without increasing the
      total watermark energy;
    - normalized correlation reduces sensitivity to local latent power;
    - removal estimates the host power by compensating the known watermark
      energy, reducing high-SNR PSNR loss.
    """

    name = "robust_spread_spectrum_shape_components"

    def __init__(
        self,
        length: int,
        h_values: list[int],
        w_values: list[int],
        c_values: list[int],
        alpha: float = 0.08,
        seed: int = 2030,
        replicas: int = 4,
    ):
        self.length = int(length)
        self.values = [list(h_values), list(w_values), list(c_values)]
        self.alpha = float(alpha)
        self.replicas = int(replicas)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.length)
        component_indices = np.array_split(perm, 3)
        self.indices: list[list[np.ndarray]] = []
        self.codes: list[list[np.ndarray]] = []
        for values, comp_idx in zip(self.values, component_indices):
            replica_indices = [np.asarray(idx, dtype=np.int64) for idx in np.array_split(comp_idx, self.replicas)]
            replica_codes = []
            for idx in replica_indices:
                codes = rng.choice([-1.0, 1.0], size=(len(values), idx.size)).astype(np.float32)
                codes = codes - codes.mean(axis=1, keepdims=True)
                codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
                replica_codes.append(codes)
            self.indices.append(replica_indices)
            self.codes.append(replica_codes)

    @property
    def overhead_symbols(self) -> int:
        return 0

    def _host_scale(self, x: np.ndarray) -> float:
        return float(np.sqrt(signal_power(x)))

    def _rx_host_scale(self, y: np.ndarray) -> float:
        return float(np.sqrt(signal_power(y) / (1.0 + self.alpha**2)))

    def embed_shape(self, x: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        y = np.array(x, copy=True)
        scale = self._host_scale(x)
        for value, values, idx_list, code_list in zip(shape, self.values, self.indices, self.codes):
            if value not in values:
                raise ValueError(f"value {value} not in allowed component set {values}")
            value_idx = values.index(value)
            for idx, codes in zip(idx_list, code_list):
                # Energy per chip remains alpha^2 * P, split over interleaved replicas.
                y[idx] += self.alpha * scale * np.sqrt(idx.size) * codes[value_idx]
        return y

    def decode_shape(self, y: np.ndarray) -> DecodeResult:
        y = y.reshape(-1).astype(np.float32)
        decoded = []
        margins = []
        for values, idx_list, code_list in zip(self.values, self.indices, self.codes):
            scores = np.zeros(len(values), dtype=np.float64)
            for idx, codes in zip(idx_list, code_list):
                block = y[idx].astype(np.float32)
                block = block - block.mean()
                denom = float(np.linalg.norm(block) + 1e-12)
                # Normalized correlation is more stable after OFDM equalization.
                scores += (codes @ block) / denom
            order = np.argsort(scores)
            best = int(order[-1])
            second = float(scores[order[-2]]) if len(values) > 1 else 0.0
            decoded.append(values[best])
            margins.append(float(scores[best] - second))
        result = DecodeResult(0, float(min(margins)))
        result.shape = tuple(decoded)  # type: ignore[attr-defined]
        return result

    def remove_shape(self, y: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        cleaned = np.array(y, copy=True)
        scale = self._rx_host_scale(y)
        for value, values, idx_list, code_list in zip(shape, self.values, self.indices, self.codes):
            value_idx = values.index(value)
            for idx, codes in zip(idx_list, code_list):
                cleaned[idx] -= self.alpha * scale * np.sqrt(idx.size) * codes[value_idx]
        return cleaned


class ProjectionQIMID:
    """M-ary dither/QIM code in one random projection of the latent vector."""

    name = "projection_qim_shape_id"

    def __init__(self, num_classes: int, length: int, delta: float = 0.75, seed: int = 2027):
        self.num_classes = int(num_classes)
        self.length = int(length)
        self.delta = float(delta)
        rng = np.random.default_rng(seed)
        direction = rng.standard_normal(self.length).astype(np.float32)
        self.direction = direction / (np.linalg.norm(direction) + 1e-12)

    @property
    def overhead_symbols(self) -> int:
        return 0

    def _target_projection(self, projection: float, shape_id: int, scale: float) -> float:
        step = self.delta * scale
        period = step * self.num_classes
        offset = (shape_id + 0.5) * step
        k = np.round((projection - offset) / period)
        return float(k * period + offset)

    def embed(self, x: np.ndarray, shape_id: int) -> np.ndarray:
        scale = np.sqrt(signal_power(x))
        projection = float(np.dot(x, self.direction))
        target = self._target_projection(projection, shape_id, scale)
        return x + (target - projection) * self.direction

    def decode(self, y: np.ndarray) -> DecodeResult:
        scale = np.sqrt(signal_power(y))
        step = self.delta * scale
        period = step * self.num_classes
        projection = float(np.dot(y, self.direction))
        residue = projection % period
        decoded = int(np.floor(residue / step))
        centers = (np.arange(self.num_classes) + 0.5) * step
        dist = np.abs(residue - centers)
        return DecodeResult(decoded, float(step / 2.0 - np.min(dist)))

    def remove(self, y: np.ndarray, shape_id: int) -> np.ndarray:
        # QIM is not exactly invertible because the host projection was quantized.
        return np.array(y, copy=True)
