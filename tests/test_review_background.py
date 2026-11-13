"""Does a frame look like the reference frame?  (Review's "no bubble here" test.)"""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.controller.review_background import (
    MIN_DARK_AREA, BackgroundCheck, looks_like_reference,
)

SHAPE = (120, 160)


def _background(seed: int = 0, noise: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    shading = 1.0 + 0.2 * xx / SHAPE[1]  # uneven lighting, the same on every frame
    return (2000.0 * shading * (1 + noise * rng.standard_normal(SHAPE))).astype(np.float32)


def _disk(radius: float, depth: float, centre=(60, 80), soft: bool = False) -> np.ndarray:
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    d = np.hypot(yy - centre[0], xx - centre[1])
    if soft:
        return 1.0 - depth * np.exp(-(d / radius) ** 2)
    return np.where(d <= radius, 1.0 - depth, 1.0)


@pytest.fixture
def reference() -> np.ndarray:
    return _background(seed=1)


class TestLooksLikeReference:
    def test_a_frame_like_the_reference(self, reference):
        assert looks_like_reference(_background(seed=2), reference)

    def test_a_dark_bubble_is_not_background(self, reference):
        assert not looks_like_reference(_background(seed=2) * _disk(10, 0.6), reference)

    def test_a_soft_blurred_bubble_is_not_background(self, reference):
        """Blood: a wide, soft dark spot the engine's flattened view hardly shows."""
        frame = _background(seed=2) * _disk(30, 0.35, soft=True)
        assert not looks_like_reference(frame, reference)

    def test_flicker_of_the_whole_frame_is_background(self, reference):
        assert looks_like_reference(0.8 * _background(seed=2), reference)

    def test_a_bright_flash_is_no_dark_bubble(self, reference):
        frame = _background(seed=2) * (2.0 - _disk(25, 0.8))  # the laser's plasma glow
        assert looks_like_reference(frame, reference)

    def test_a_speck_of_debris_is_background(self, reference):
        radius = np.sqrt(MIN_DARK_AREA / np.pi) * 0.6
        assert looks_like_reference(_background(seed=2) * _disk(radius, 0.8), reference)

    def test_a_bright_bubble_with_bright_polarity(self, reference):
        frame = _background(seed=2) * (2.0 - _disk(10, 0.6))
        assert not looks_like_reference(frame, reference, polarity="bright")
        assert looks_like_reference(frame, reference, polarity="dark")

    def test_shapes_must_match(self, reference):
        with pytest.raises(ValueError):
            looks_like_reference(np.ones((10, 10), np.float32), reference)


class TestBackgroundCheck:
    def test_reads_each_frame_once(self, reference):
        frames = {0: _background(seed=2), 1: _background(seed=3) * _disk(10, 0.6)}
        reads: list[int] = []

        def read(i):
            reads.append(i)
            return frames[i]

        check = BackgroundCheck()
        key = ("source", 0)
        assert check.looks_empty(key, 0, read, lambda: reference) is True
        assert check.looks_empty(key, 1, read, lambda: reference) is False
        assert check.looks_empty(key, 0, read, lambda: reference) is True
        assert reads == [0, 1]

    def test_another_source_starts_over(self, reference):
        check = BackgroundCheck()
        check.looks_empty(("a",), 0, lambda i: reference, lambda: reference)
        reads: list[int] = []
        check.looks_empty(("b",), 0, lambda i: reads.append(i) or reference, lambda: reference)
        assert reads == [0]

    def test_unknown_without_a_reference_or_on_errors(self, reference):
        check = BackgroundCheck()
        assert check.looks_empty(("a",), 0, lambda i: reference, lambda: None) is None

        def broken(_i):
            raise OSError("unreadable")
        assert check.looks_empty(("b",), 0, broken, lambda: reference) is None
