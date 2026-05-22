from __future__ import annotations

import numpy as np


def latent_smoothness_score(vec: np.ndarray, shape: tuple[int, int, int]) -> float:
    h, w, c = shape
    x = vec.reshape(h, w, c).astype(np.float32)
    score = 0.0
    count = 0
    if h > 1:
        score += float(np.mean((x[1:, :, :] - x[:-1, :, :]) ** 2))
        count += 1
    if w > 1:
        score += float(np.mean((x[:, 1:, :] - x[:, :-1, :]) ** 2))
        count += 1
    return score / max(count, 1)


def choose_shape_by_smoothness(vec: np.ndarray, candidates: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return min(candidates, key=lambda shape: latent_smoothness_score(vec, shape))


def choose_shape_by_length(vec: np.ndarray, candidates: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    matches = [shape for shape in candidates if shape[0] * shape[1] * shape[2] == vec.size]
    if len(matches) == 1:
        return matches[0]
    return None
