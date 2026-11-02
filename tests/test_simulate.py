import numpy as np

from sonarseg.metrics import confusion, iou_per_class
from sonarseg.simulate import NUM_CLASSES, OBJECT, SHADOW, WATER, Config, generate


def test_same_seed_same_sample():
    first, second = generate(5), generate(5)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_shapes_and_ranges():
    image, labels = generate(1, Config(size=128))
    assert image.shape == labels.shape == (128, 128)
    assert 0 <= image.min() and image.max() <= 1
    assert labels.max() < NUM_CLASSES


def test_water_column_is_darker_than_seabed():
    image, labels = generate(2)
    assert image[labels == WATER].mean() < image[labels != WATER].mean()


def test_objects_are_bright_with_dark_shadows():
    for seed in range(50):
        image, labels = generate(seed)
        if (labels == OBJECT).any() and (labels == SHADOW).any():
            assert image[labels == OBJECT].mean() > image[labels == SHADOW].mean() * 3
            return
    raise AssertionError("no sample contained an object and shadow")


def test_iou_known_values():
    truth = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    assert iou_per_class(confusion(pred, truth, 2)).tolist() == [0.5, 2 / 3]


def test_absent_class_is_nan_not_zero():
    assert np.isnan(iou_per_class(confusion(np.zeros(4, int), np.zeros(4, int), 2))[1])
