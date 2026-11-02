"""Classical baseline: Gaussian blur plus per-class intensity bands.

Band edges are fitted on training data, so the U-Net must beat something
reasonable rather than random guessing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .simulate import NUM_CLASSES


def fit(images: list[np.ndarray], labels: list[np.ndarray]) -> np.ndarray:
    """Mean blurred intensity per class."""
    totals, counts = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
    for image, label in zip(images, labels, strict=True):
        blurred = gaussian_filter(image, 1.5)
        totals += np.bincount(label.ravel(), blurred.ravel(), NUM_CLASSES)
        counts += np.bincount(label.ravel(), minlength=NUM_CLASSES)
    return totals / np.maximum(counts, 1)


def predict(image: np.ndarray, centres: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(image, 1.5)
    return np.abs(blurred[..., None] - centres).argmin(-1)
