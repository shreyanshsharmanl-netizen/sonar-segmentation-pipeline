"""Confusion-matrix metrics for multi-class segmentation."""

import numpy as np


def confusion(pred: np.ndarray, truth: np.ndarray, classes: int) -> np.ndarray:
    return np.bincount(classes * truth.ravel() + pred.ravel(), minlength=classes**2).reshape(
        classes, classes
    )


def iou_per_class(matrix: np.ndarray) -> np.ndarray:
    hits = np.diag(matrix).astype(float)
    union = matrix.sum(0) + matrix.sum(1) - hits
    return np.divide(hits, union, out=np.full(len(hits), np.nan), where=union > 0)
