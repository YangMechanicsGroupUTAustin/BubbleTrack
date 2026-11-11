"""Automatic reference-frame choice on synthetic recordings (abnormal first frame, flash, bubble)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from _edge_synth import Bubble, Scene, render
from bubbletrack.model.edge import choose_reference_frame

H, W = 150, 240
YY, XX = np.mgrid[0:H, 0:W].astype(float)


def _glow(rc: float, cc: float, sigma: float) -> np.ndarray:
    return np.exp(-((YY - rc) ** 2 + (XX - cc) ** 2) / (2 * sigma ** 2))


@lru_cache(maxsize=None)
def _frame(i: int, radius: float) -> np.ndarray:
    bubbles = (Bubble(rc=75.0, cc=120.0, r=radius),) if radius > 0 else ()
    img, _ = render(Scene(h=H, w=W, bubbles=bubbles, static=1, seed=i))
    return img


def make_sequence(n: int, *, flash_at: int | None, bubble_from: int, abnormal_first: bool = True) -> list:
    """uint16 frames: optional abnormal frame 0 (brighter + glow), flash frame, growing bubble."""
    frames = []
    for i in range(n):
        radius = 8.0 + 5.0 * (i - bubble_from) if i >= bubble_from else 0.0
        img = _frame(i, radius).astype(float)
        if i == 0 and abnormal_first:
            img = img * 1.2 * (1 + 0.4 * _glow(75, 20, 40))          # exposure + afterglow
        if i == flash_at:
            img = img * 1.12 + 40000 * _glow(75, 120, 12)             # plasma flash
        frames.append(np.clip(img, 0, 65504).astype(np.uint16))
    return frames


def _choose(frames, **kw):
    return choose_reference_frame(lambda i: frames[i], len(frames), **kw)


def test_abnormal_first_frame_then_flash_picks_frame_2():
    """Frame 1 abnormal, frame 2 clean, frame 3 flash, bubble after: 0-based index 1."""
    assert _choose(make_sequence(10, flash_at=2, bubble_from=3)) == 1


def test_last_clean_frame_before_a_late_flash_is_chosen():
    assert _choose(make_sequence(10, flash_at=6, bubble_from=7)) == 5


def test_bubble_without_a_flash_is_an_event():
    assert _choose(make_sequence(10, flash_at=None, bubble_from=4)) == 3


def test_recording_that_starts_with_a_bubble_has_no_reference():
    assert _choose(make_sequence(10, flash_at=None, bubble_from=0, abnormal_first=False)) is None


def test_quiet_window_returns_its_last_frame():
    frames = make_sequence(12, flash_at=None, bubble_from=99)
    assert _choose(frames, search=6) == 5


def test_unreadable_frames_are_skipped():
    frames = make_sequence(10, flash_at=2, bubble_from=3)

    def read(i):
        if i == 0:
            raise OSError("corrupt file")
        return frames[i]
    assert choose_reference_frame(read, len(frames)) == 1


def test_colour_frames_are_supported():
    frames = [np.dstack([f] * 3) for f in make_sequence(10, flash_at=2, bubble_from=3)]
    assert _choose(frames) == 1


def test_degenerate_arguments():
    assert choose_reference_frame(lambda i: None, 0) is None
    with pytest.raises(ValueError):
        choose_reference_frame(lambda i: None, -1)
    with pytest.raises(ValueError):
        choose_reference_frame(lambda i: None, 5, search=0)
