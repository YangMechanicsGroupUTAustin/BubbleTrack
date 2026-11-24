"""The recording's black level (edge-1.7.0): estimate, and Auto unchanged by a constant offset.

An export may add a constant to every pixel (the HPV-X TIFF16 export adds 655 counts to the raw
.dat).  Auto removes the recording's darkest level before dividing by the reference, so an
offset copy of a recording must give the same results as the recording itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from _edge_synth import Bubble, Scene, render
from bubbletrack.model import edge_pipeline
from bubbletrack.model.edge import EdgeParams, choose_reference_frame, detect_edge
from bubbletrack.model.edge.black import (
    frame_dark_level,
    gray_pair,
    recording_black_level,
    sample_frames,
    with_black,
)
from bubbletrack.model.edge.engine import engine_view
from bubbletrack.model.edge.normalize import to_gray_frame
from bubbletrack.model.teach.features import compute_features

OFFSET = 655  # the HPV-X TIFF16 export's offset
ROI = ((1, 150), (1, 240))


def _offset(frame: np.ndarray, c: int = OFFSET) -> np.ndarray:
    """*frame* as an export that adds *c* would write it (clipped at the dtype's top)."""
    top = np.iinfo(frame.dtype).max
    return np.minimum(frame.astype(np.int64) + c, top).astype(frame.dtype)


def _recording(n: int = 6) -> list[np.ndarray]:
    """uint16 frames of a dark-speckled background: bubble-free frame 0, then a growing bubble."""
    frames = []
    for i in range(n):
        bubbles = (Bubble(rc=75.0, cc=120.0, r=12.0 + 6.0 * i),) if i else ()
        img, _ = render(Scene(h=150, w=240, bg=9000.0, bubbles=bubbles, static=2, speckle=0.7, seed=i))
        frames.append(img)
    return frames


# -- the estimate ------------------------------------------------------------------------ #

def test_dark_level_is_a_low_percentile_of_the_unsaturated_values():
    img = np.arange(1000, 11000, dtype=np.uint16).reshape(100, 100)
    img[:5] = 65535                                        # saturated rows say nothing about black
    assert frame_dark_level(img) == pytest.approx(np.percentile(img[5:], 0.1))


def test_a_burned_in_label_of_zeros_is_not_black():
    img = _offset(_recording(1)[0])
    img[-20:, -60:] = 0                                    # the export's info stamp
    assert frame_dark_level(img) >= OFFSET


def test_true_black_that_reaches_zero_counts():
    img = (np.arange(256, dtype=np.uint8).repeat(40))[:10000].reshape(100, 100) // 4
    assert frame_dark_level(img) == 0.0                    # zeros next to 1, 2, 3 ...: real black


def test_the_level_moves_with_an_offset():
    frames = _recording()
    plain = recording_black_level(lambda i: frames[i], len(frames))
    shifted = recording_black_level(lambda i: _offset(frames[i]), len(frames))
    assert plain > 0
    assert shifted == pytest.approx(plain + OFFSET, abs=1e-3)


def test_the_darkest_frame_sets_the_level():
    frames = _recording()
    frames[3] = (frames[3] // 8).astype(np.uint16)
    lows = [frame_dark_level(f) for f in frames]
    assert recording_black_level(lambda i: frames[i], len(frames)) == min(lows) == lows[3]


def test_frames_are_sampled_evenly_with_both_ends():
    assert sample_frames(10, 256) == list(range(10))
    picked = sample_frames(10_000, 256)
    assert len(picked) == 256 and picked[0] == 0 and picked[-1] == 9999
    assert picked == sorted(set(picked))
    assert sample_frames(0, 256) == []


def test_unreadable_frames_are_skipped():
    frames = _recording(3)

    def read(i):
        if i == 1:
            raise OSError("broken file")
        return frames[i]

    assert recording_black_level(read, 3) == min(frame_dark_level(frames[0]), frame_dark_level(frames[2]))
    assert recording_black_level(lambda i: (_ for _ in ()).throw(OSError("x")), 4) == 0.0
    with pytest.raises(ValueError, match="n_frames"):
        recording_black_level(lambda i: frames[i], -1)


def test_a_raw_dat_recording_keeps_black_zero_without_reading_a_frame():
    def read(i):
        raise AssertionError("a .dat frame was read")

    assert edge_pipeline.recording_black_level(read, 256, "C:/data/Jin_14_37_03.DAT") == 0.0
    frames = _recording(2)
    assert edge_pipeline.recording_black_level(lambda i: frames[i], 2, "Jin_14_37_03") > 0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "655"])
def test_black_level_is_validated(bad):
    with pytest.raises(ValueError, match="black_level"):
        EdgeParams(black_level=bad)


def test_black_level_is_stored_as_float():
    assert EdgeParams(black_level=np.float32(655)).black_level == 655.0
    assert isinstance(EdgeParams(black_level=655).black_level, float)


def test_with_black_shifts_values_and_full_scale():
    gray = to_gray_frame(np.array([[0, 100, 700, 65535]], np.uint16))
    out = with_black(gray, 655.0)
    np.testing.assert_array_equal(out.values, [[0, 0, 45, 64880]])
    assert out.full_scale == 65535 - 655
    assert with_black(gray, 0.0) is gray
    with pytest.raises(ValueError, match="reference shape"):
        gray_pair(np.zeros((4, 4), np.uint16), np.zeros((4, 5), np.uint16), 0.0)


# -- Auto with the black level ------------------------------------------------------------- #

def _fit(frame, ref, black):
    return detect_edge(frame, *ROI, reference=ref, params=EdgeParams(black_level=black))


def test_an_offset_copy_gives_the_same_auto_result():
    frames = _recording()
    ref, frame = frames[0], frames[4]
    d = recording_black_level(lambda i: frames[i], len(frames))
    d_off = recording_black_level(lambda i: _offset(frames[i]), len(frames))
    plain, shifted = _fit(frame, ref, d), _fit(_offset(frame), _offset(ref), d_off)
    assert plain.accepted and shifted.accepted
    assert shifted.radius == pytest.approx(plain.radius, abs=1e-3)
    assert (shifted.rc, shifted.cc) == pytest.approx((plain.rc, plain.cc), abs=1e-3)
    assert shifted.confidence == pytest.approx(plain.confidence, abs=1e-4)
    assert shifted.provenance["params"]["black_level"] == pytest.approx(d + OFFSET, abs=1e-3)


def test_without_the_black_level_an_offset_changes_what_auto_sees():
    # why edge-1.7.0: frame / reference assumes 0 = no light, so the offset lifts the bubble's inside
    frames = _recording()
    ref, frame = frames[0], frames[4]
    d = recording_black_level(lambda i: frames[i], len(frames))
    view = engine_view(frame, *ROI, reference=ref, params=EdgeParams(black_level=d))
    raw_off = engine_view(_offset(frame), *ROI, reference=_offset(ref), params=EdgeParams())
    fixed = engine_view(_offset(frame), *ROI, reference=_offset(ref), params=EdgeParams(black_level=d + OFFSET))
    inside = view < 0.3
    assert inside.sum() > 1000
    assert float(np.median(raw_off[inside] - view[inside])) > 0.02
    np.testing.assert_allclose(fixed, view, atol=1e-4)


def test_the_reference_choice_ignores_an_offset():
    # bubble-free frames 0-2, then a growing bubble (as in test_edge_reference)
    frames = []
    for i in range(8):
        bubbles = (Bubble(rc=75.0, cc=120.0, r=8.0 + 5.0 * (i - 3)),) if i >= 3 else ()
        frames.append(render(Scene(h=150, w=240, bg=9000.0, bubbles=bubbles, static=1, seed=i))[0])
    shifted = [_offset(f) for f in frames]
    d = recording_black_level(lambda i: frames[i], len(frames))
    d_off = recording_black_level(lambda i: shifted[i], len(shifted))
    plain = choose_reference_frame(lambda i: frames[i], len(frames), black_level=d)
    assert plain == 2
    assert choose_reference_frame(lambda i: shifted[i], len(shifted), black_level=d_off) == plain


def test_ml_assist_features_ignore_an_offset():
    frames = _recording()
    d = recording_black_level(lambda i: frames[i], len(frames))
    a = compute_features(frames[3], *ROI, reference=frames[0], black_level=d).values
    b = compute_features(_offset(frames[3]), *ROI, reference=_offset(frames[0]), black_level=d + OFFSET).values
    np.testing.assert_allclose(a, b, atol=1e-3)
