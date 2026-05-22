from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShapeSpec:
    h: int
    w: int
    c: int

    @property
    def size(self) -> int:
        return self.h * self.w * self.c

    @property
    def label(self) -> str:
        return f"{self.h}x{self.w}x{self.c}"


def make_default_shape_book() -> list[ShapeSpec]:
    """Ambiguous shapes with the same flattened length.

    The equal product intentionally removes the easy "infer from length" answer.
    """
    return [
        ShapeSpec(16, 16, 128),
        ShapeSpec(16, 32, 64),
        ShapeSpec(32, 16, 64),
        ShapeSpec(32, 32, 32),
        ShapeSpec(8, 64, 64),
        ShapeSpec(64, 8, 64),
        ShapeSpec(8, 32, 128),
        ShapeSpec(64, 16, 32),
    ]


def _smooth2d(x: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        x = (
            x
            + np.roll(x, 1, axis=0)
            + np.roll(x, -1, axis=0)
            + np.roll(x, 1, axis=1)
            + np.roll(x, -1, axis=1)
        ) / 5.0
    return x


def sample_latent(shape: ShapeSpec, rng: np.random.Generator) -> np.ndarray:
    """Generate a plausible neural latent tensor and flatten it.

    This is not meant to mimic one specific model exactly. It gives us a
    non-white host signal with spatial and channel correlations so watermark
    detection is tested against realistic host interference.
    """
    tensor = rng.standard_normal((shape.h, shape.w, shape.c)).astype(np.float32)
    tensor = _smooth2d(tensor, passes=3)
    channel_scale = rng.lognormal(mean=0.0, sigma=0.35, size=(1, 1, shape.c)).astype(np.float32)
    tensor = tensor * channel_scale
    tensor = tensor - tensor.mean()
    tensor = tensor / (tensor.std() + 1e-6)
    return tensor.reshape(-1).astype(np.float32)
