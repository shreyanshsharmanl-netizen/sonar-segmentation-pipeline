"""Synthetic side-scan sonar waterfall images with exact per-pixel labels.

This is a simplified physical model, not a calibrated sonar simulator:
- texture differs by seabed class (sand ripples, rock, mud);
- echo strength falls with range from the nadir;
- objects return a bright highlight and cast an acoustic shadow away from
  the sonar, whose length grows with range;
- multiplicative speckle noise is applied last.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

CLASSES = ["water_column", "sand", "rock", "mud", "object", "shadow"]
NUM_CLASSES = len(CLASSES)
WATER, SAND, ROCK, MUD, OBJECT, SHADOW = range(NUM_CLASSES)


@dataclass(frozen=True)
class Config:
    size: int = 128
    speckle_looks: float = 4.0  # lower values mean heavier speckle
    max_objects: int = 3


def _smooth_noise(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    coarse = rng.standard_normal((size // scale + 2, size // scale + 2))
    return zoom(coarse, size / (size // scale), order=3)[:size, :size]


def _seabed(rng: np.random.Generator, size: int) -> np.ndarray:
    """Patchy seabed labels from thresholded smooth noise."""
    field = _smooth_noise(rng, size, rng.integers(32, 56))
    low, high = np.quantile(field, sorted(rng.uniform(0.2, 0.8, 2)))
    labels = np.full((size, size), SAND, np.int64)
    labels[field < low] = MUD
    labels[field > high] = ROCK
    return labels


def _texture(rng: np.random.Generator, labels: np.ndarray) -> np.ndarray:
    size = labels.shape[0]
    y, x = np.mgrid[:size, :size]
    angle = rng.uniform(0, np.pi)
    ripples = np.sin((x * np.cos(angle) + y * np.sin(angle)) * rng.uniform(0.5, 0.9))
    rock = np.abs(gaussian_filter(rng.standard_normal((size, size)), 1.5)) * 6
    echo = np.zeros((size, size), np.float32)
    echo[labels == SAND] = (0.34 + 0.13 * ripples)[labels == SAND]
    echo[labels == ROCK] = (0.42 + 0.34 * rock)[labels == ROCK]
    echo[labels == MUD] = 0.16
    return echo


def _add_objects(rng, labels, echo, centre, starts, config) -> None:
    size = config.size
    for _ in range(rng.integers(0, config.max_objects + 1)):
        y = rng.integers(8, size - 8)
        x = rng.integers(8, size - 8)
        if abs(x - centre) < starts[y] + 6:
            continue
        half_h, half_w = rng.integers(2, 6, 2)
        yy, xx = np.ogrid[:size, :size]
        body = ((yy - y) / half_h) ** 2 + ((xx - x) / half_w) ** 2 <= 1
        direction = 1 if x > centre else -1
        length = int(4 + abs(x - centre) * rng.uniform(0.12, 0.28))
        far = x + direction * (half_w + length)
        left, right = sorted((x, far))
        shadow = np.zeros_like(body)
        shadow[max(y - half_h, 0) : y + half_h + 1, max(left, 0) : min(right, size)] = True
        shadow &= ~body
        echo[shadow] *= 0.08
        labels[shadow] = SHADOW
        echo[body] = 1.0
        labels[body] = OBJECT


def generate(seed: int, config: Config = Config()) -> tuple[np.ndarray, np.ndarray]:
    """Return (image in [0, 1], labels), both square. Same seed, same sample."""
    rng = np.random.default_rng(seed)
    size = config.size
    labels = _seabed(rng, size)
    echo = _texture(rng, labels)

    centre = size // 2 + rng.integers(-6, 7)
    width = rng.integers(7, 15)
    wobble = gaussian_filter(rng.standard_normal(size), 12) * 25
    starts = np.clip(width + wobble, 3, size // 4).astype(int)
    _add_objects(rng, labels, echo, centre, starts, config)

    range_from_nadir = np.abs(np.arange(size) - centre) / (size / 2)
    echo = echo * (1.0 - 0.45 * range_from_nadir)[None, :]
    for row in range(size):
        low, high = centre - starts[row], centre + starts[row]
        labels[row, low:high] = WATER
        echo[row, low:high] = 0.025
    speckle = rng.gamma(config.speckle_looks, 1 / config.speckle_looks, (size, size))
    return np.clip(echo * speckle, 0, 1).astype(np.float32), labels
