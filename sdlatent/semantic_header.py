from __future__ import annotations

import numpy as np

from .channel import signal_power


class SemanticShapeHeader:
    """Analog semantic header for shape metadata.

    H/W/C are mapped to short spread-spectrum analog codewords and concatenated
    with the image semantic latent. This baseline spends explicit semantic
    symbols but does not distort image-latent coefficients like a watermark.
    """

    name = "semantic_shape_header"

    def __init__(
        self,
        h_values: list[int],
        w_values: list[int],
        c_values: list[int],
        chips_per_component: int = 512,
        amplitude: float = 1.0,
        seed: int = 2040,
    ):
        self.values = [list(h_values), list(w_values), list(c_values)]
        self.chips_per_component = int(chips_per_component)
        self.amplitude = float(amplitude)
        rng = np.random.default_rng(seed)
        self.codes: list[np.ndarray] = []
        for values in self.values:
            codes = rng.choice([-1.0, 1.0], size=(len(values), self.chips_per_component)).astype(np.float32)
            codes = codes - codes.mean(axis=1, keepdims=True)
            codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
            self.codes.append(codes)

    @property
    def overhead_symbols(self) -> int:
        return 3 * self.chips_per_component

    def encode(self, shape: tuple[int, int, int], host_power: float = 1.0) -> np.ndarray:
        scale = np.sqrt(host_power)
        chunks = []
        for value, values, codes in zip(shape, self.values, self.codes):
            if value not in values:
                raise ValueError(f"value {value} not in allowed semantic-header set {values}")
            idx = values.index(value)
            chunks.append(self.amplitude * scale * np.sqrt(self.chips_per_component) * codes[idx])
        return np.concatenate(chunks).astype(np.float32)

    def decode(self, header_rx: np.ndarray) -> tuple[tuple[int, int, int] | None, float]:
        header_rx = np.asarray(header_rx, dtype=np.float32).reshape(-1)
        if header_rx.size < self.overhead_symbols:
            return None, float("-inf")
        decoded = []
        margins = []
        offset = 0
        for values, codes in zip(self.values, self.codes):
            block = header_rx[offset : offset + self.chips_per_component]
            offset += self.chips_per_component
            block = block - block.mean()
            denom = float(np.linalg.norm(block) + 1e-12)
            scores = (codes @ block) / denom
            order = np.argsort(scores)
            best = int(order[-1])
            second = float(scores[order[-2]]) if len(values) > 1 else 0.0
            decoded.append(values[best])
            margins.append(float(scores[best] - second))
        return tuple(decoded), float(min(margins))

    def concatenate(self, image_latent: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        header = self.encode(shape, signal_power(image_latent))
        return np.concatenate([header, image_latent.astype(np.float32)])

    def split(self, rx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rx = np.asarray(rx, dtype=np.float32).reshape(-1)
        return rx[: self.overhead_symbols], rx[self.overhead_symbols :]
