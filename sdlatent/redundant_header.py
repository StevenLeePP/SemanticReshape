from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import signal_power


def _id_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(width - 1, -1, -1)], dtype=np.uint8)


def _bits_to_id(bits: np.ndarray) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


@dataclass(frozen=True)
class HeaderDecode:
    shape_id: int
    margin: float


class RepetitionCodedShapeHeader:
    """Very-low-rate conventional metadata code.

    This is intentionally simple and conservative: a fixed-width shape ID is
    mapped to BPSK bits, each bit is repeated to fill the requested symbol
    budget, and the receiver uses soft majority combining. A practical LDPC,
    polar, or tail-biting convolutional code at the same rate should be at
    least competitive with this baseline for such a tiny payload.
    """

    name = "repetition_coded_shape_header"

    def __init__(
        self,
        num_classes: int,
        total_symbols: int,
        metadata_bits: int = 3,
        amplitude: float = 1.0,
    ):
        self.num_classes = int(num_classes)
        self.total_symbols = int(total_symbols)
        self.metadata_bits = int(metadata_bits)
        self.amplitude = float(amplitude)
        if self.metadata_bits < int(np.ceil(np.log2(max(self.num_classes, 2)))):
            raise ValueError("metadata_bits is too small for num_classes")
        if self.total_symbols < self.metadata_bits:
            raise ValueError("total_symbols must be at least metadata_bits")

        counts = np.full(self.metadata_bits, self.total_symbols // self.metadata_bits, dtype=np.int64)
        counts[: self.total_symbols % self.metadata_bits] += 1
        self.counts = counts

    @property
    def overhead_symbols(self) -> int:
        return self.total_symbols

    @property
    def effective_rate(self) -> float:
        return self.metadata_bits / self.total_symbols

    def encode(self, shape_id: int, host_power: float = 1.0) -> np.ndarray:
        bits = _id_to_bits(int(shape_id), self.metadata_bits)
        scale = self.amplitude * np.sqrt(host_power)
        chunks = []
        for bit, count in zip(bits, self.counts):
            symbol = 1.0 if int(bit) else -1.0
            chunks.append(np.full(int(count), scale * symbol, dtype=np.float32))
        return np.concatenate(chunks)

    def decode(self, rx: np.ndarray) -> HeaderDecode:
        rx = np.asarray(rx, dtype=np.float32).reshape(-1)
        bits = []
        margins = []
        offset = 0
        for count in self.counts:
            block = rx[offset : offset + int(count)]
            offset += int(count)
            soft = float(np.mean(block))
            bits.append(1 if soft >= 0.0 else 0)
            margins.append(abs(soft))
        shape_id = _bits_to_id(np.array(bits, dtype=np.uint8))
        if shape_id >= self.num_classes:
            shape_id = -1
        return HeaderDecode(shape_id, float(min(margins)))


class RandomCodebookShapeHeader:
    """Low-rate ML-decoded BPSK codebook for a shape ID.

    This represents an optimistic conventional-coded metadata side channel:
    every admissible shape ID receives a long random BPSK codeword, and the
    receiver picks the maximum-correlation codeword.
    """

    name = "random_codebook_shape_header"

    def __init__(
        self,
        num_classes: int,
        total_symbols: int,
        metadata_bits: int = 3,
        amplitude: float = 1.0,
        seed: int = 2206,
    ):
        self.num_classes = int(num_classes)
        self.total_symbols = int(total_symbols)
        self.metadata_bits = int(metadata_bits)
        self.amplitude = float(amplitude)
        rng = np.random.default_rng(seed)
        codes = rng.choice([-1.0, 1.0], size=(self.num_classes, self.total_symbols)).astype(np.float32)
        codes = codes - codes.mean(axis=1, keepdims=True)
        codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
        self.codes = codes

    @property
    def overhead_symbols(self) -> int:
        return self.total_symbols

    @property
    def effective_rate(self) -> float:
        return self.metadata_bits / self.total_symbols

    def encode(self, shape_id: int, host_power: float = 1.0) -> np.ndarray:
        scale = self.amplitude * np.sqrt(host_power) * np.sqrt(self.total_symbols)
        return (scale * self.codes[int(shape_id)]).astype(np.float32)

    def decode(self, rx: np.ndarray) -> HeaderDecode:
        rx = np.asarray(rx, dtype=np.float32).reshape(-1)
        rx = rx - rx.mean()
        denom = float(np.linalg.norm(rx) + 1e-12)
        scores = (self.codes @ rx) / denom
        order = np.argsort(scores)
        best = int(order[-1])
        second = float(scores[order[-2]]) if self.num_classes > 1 else 0.0
        return HeaderDecode(best, float(scores[best] - second))


def add_awgn_with_reference_power(
    symbols: np.ndarray,
    snr_db: float,
    reference_power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise_var = float(reference_power) / (10.0 ** (float(snr_db) / 10.0))
    return np.asarray(symbols, dtype=np.float32) + rng.normal(0.0, np.sqrt(noise_var), size=symbols.shape)


def host_power_from_latent(latent: np.ndarray) -> float:
    return signal_power(np.asarray(latent, dtype=np.float32).reshape(-1))
