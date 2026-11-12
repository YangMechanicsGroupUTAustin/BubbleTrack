"""Teach examples: sparse per-frame labels, strokes, verified frames, JSON round trip.

Pure numpy (no scikit-learn): these run everywhere, CI included.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bubbletrack.model.teach import (
    BACKGROUND,
    BUBBLE,
    ExampleConfig,
    MLExamples,
    circle_masks,
    stroke_pixels,
    training_pixels,
)

SHAPE = (120, 160)


def _px(*pairs):
    return np.array(pairs, dtype=int)


def _as_set(pixels):
    return {tuple(p) for p in pixels.tolist()}


# --------------------------------------------------------------------------- #
#  MLExamples
# --------------------------------------------------------------------------- #

def test_empty_examples():
    ex = MLExamples()
    assert ex.is_empty
    assert ex.frames == ()
    assert ex.to_dict()["strokes"] == {} and ex.to_dict()["verified"] == {}


def test_paint_returns_a_new_object_and_leaves_the_old_one():
    ex = MLExamples()
    ex2 = ex.paint(3, BUBBLE, _px((10, 20), (10, 21)))
    assert ex.is_empty and not ex2.is_empty
    assert ex2.frames == (3,)
    assert ex2.strokes[3].bubble == frozenset({(10, 20), (10, 21)})
    with pytest.raises(TypeError):
        ex2.strokes[4] = None                 # read-only mapping


def test_later_paint_moves_pixels_to_the_new_label():
    ex = MLExamples().paint(0, BUBBLE, _px((5, 5), (5, 6))).paint(0, BACKGROUND, _px((5, 6), (9, 9)))
    assert ex.strokes[0].bubble == frozenset({(5, 5)})
    assert ex.strokes[0].background == frozenset({(5, 6), (9, 9)})


def test_erase_and_clear():
    ex = MLExamples().paint(1, BUBBLE, _px((5, 5), (5, 6))).paint(1, BACKGROUND, _px((7, 7)))
    ex = ex.erase(1, _px((5, 6), (7, 7)))
    assert ex.strokes[1].bubble == frozenset({(5, 5)}) and ex.strokes[1].background == frozenset()
    assert ex.erase(1, _px((5, 5))).is_empty               # nothing left: the frame disappears
    assert ex.verify(1, (50.0, 60.0, 20.0)).clear(1).is_empty


def test_verify_and_unverify():
    ex = MLExamples().verify(7, (50.5, 60.0, 21.0)).verify(2, (40.0, 41.0, 10.0))
    assert ex.frames == (2, 7)
    assert ex.verified[7] == (50.5, 60.0, 21.0)
    assert ex.unverify(7).frames == (2,)
    assert ex.unverify(99) == ex


def test_with_verified_replaces_all_circles_and_keeps_strokes():
    ex = MLExamples().verify(1, (10.0, 10.0, 5.0)).paint(2, BUBBLE, _px((3, 3)))
    synced = ex.with_verified({4: (40.0, 41.0, 12.0), 5: (50.0, 51.0, 13.0)})
    assert dict(synced.verified) == {4: (40.0, 41.0, 12.0), 5: (50.0, 51.0, 13.0)}
    assert synced.strokes == ex.strokes
    assert ex.with_verified({}).frames == (2,)
    with pytest.raises(ValueError):
        ex.with_verified({3: (1.0, 1.0, 0.0)})


@pytest.mark.parametrize("bad", [(1.0, 2.0, 0.0), (1.0, 2.0, -3.0), (np.nan, 2.0, 3.0), (1.0, 2.0)])
def test_verify_rejects_bad_circles(bad):
    with pytest.raises(ValueError):
        MLExamples().verify(0, bad)


def test_paint_validates_its_input():
    ex = MLExamples()
    with pytest.raises(ValueError):
        ex.paint(0, "foam", _px((1, 1)))
    with pytest.raises(ValueError):
        ex.paint(-1, BUBBLE, _px((1, 1)))
    with pytest.raises(ValueError):
        ex.paint(0, BUBBLE, _px((0, 1)))            # 1-based: row 0 does not exist
    with pytest.raises(ValueError):
        ex.paint(0, BUBBLE, np.array([1, 2, 3]))


def test_json_round_trip_is_exact_and_plain():
    ex = (MLExamples().paint(4, BUBBLE, _px((10, 20), (11, 20))).paint(4, BACKGROUND, _px((90, 100)))
          .verify(9, (50.25, 61.5, 22.75)))
    text = json.dumps(ex.to_dict())                    # plain JSON, no pickle
    back = MLExamples.from_dict(json.loads(text))
    assert back == ex
    assert back.to_dict() == ex.to_dict()


@pytest.mark.parametrize("payload", [
    {"version": 99, "strokes": {}, "verified": {}},
    {"version": 1, "strokes": {"x": {}}, "verified": {}},
    {"version": 1, "strokes": {}, "verified": {"3": [1.0, 2.0]}},
    {"version": 1, "strokes": {"3": {"bubble": [[0, 5]]}}, "verified": {}},
    "not a dict",
])
def test_from_dict_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        MLExamples.from_dict(payload)


# --------------------------------------------------------------------------- #
#  Strokes
# --------------------------------------------------------------------------- #

def test_stroke_is_a_thin_band_along_the_polyline():
    px = stroke_pixels([(20.0, 10.0), (20.0, 60.0)], radius=2, shape=SHAPE)
    assert px.dtype.kind == "i" and px.shape[1] == 2
    rows, cols = px[:, 0], px[:, 1]
    assert rows.min() >= 18 and rows.max() <= 22              # 1-based rows 18..22
    assert cols.min() >= 8 and cols.max() <= 62
    assert 200 <= len(px) <= 300                               # ~5 px wide x 51 px long


def test_single_point_stroke_is_a_dot_and_strokes_are_clipped():
    dot = stroke_pixels([(50.0, 50.0)], radius=3, shape=SHAPE)
    assert 20 <= len(dot) <= 40
    edge = stroke_pixels([(1.0, 1.0), (1.0, 500.0)], radius=1, shape=SHAPE)
    assert edge[:, 0].min() >= 1 and edge[:, 1].max() <= SHAPE[1]
    assert len(stroke_pixels([], radius=2, shape=SHAPE)) == 0


# --------------------------------------------------------------------------- #
#  Automatic examples from a verified circle
# --------------------------------------------------------------------------- #

def test_circle_masks_follow_the_configured_margins():
    cfg = ExampleConfig(inner=0.8, ring=(1.25, 2.0), far=2.5)
    bub, bg = circle_masks((60.0, 80.0, 20.0), SHAPE, cfg)      # 1-based centre -> 0-based (59, 79)
    assert bub.shape == SHAPE and bg.shape == SHAPE
    assert bub[59, 79] and bub[59, 79 + 15] and not bub[59, 79 + 17]
    assert not bg[59, 79 + 22] and bg[59, 79 + 30] and not bg[59, 79 + 45]   # gap, ring, between
    assert bg[0, 0]                                                       # far background
    assert not (bub & bg).any()


def test_circle_masks_without_far_background():
    bub, bg = circle_masks((60.0, 80.0, 20.0), SHAPE, ExampleConfig(far=None))
    assert not bg[0, 0] and bg[59, 79 + 30]


def test_training_pixels_are_capped_deterministic_and_zero_based():
    cfg = ExampleConfig(max_per_class=500)
    ex = MLExamples().verify(2, (60.0, 80.0, 20.0))
    b1, g1 = training_pixels(ex, 2, SHAPE, cfg)
    b2, g2 = training_pixels(ex, 2, SHAPE, cfg)
    assert len(b1) == 500 and len(g1) == 500
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(g1, g2)
    d = np.hypot(b1[:, 0] - 59, b1[:, 1] - 79)
    assert d.max() < 16.5
    assert training_pixels(ex, 3, SHAPE, cfg)[0].shape == (0, 2)       # nothing for an unlabelled frame


def test_strokes_override_the_automatic_examples():
    cfg = ExampleConfig(max_per_class=100_000)
    # the user says a pixel in the automatic background ring is bubble (and vice versa)
    ex = (MLExamples().verify(0, (60.0, 80.0, 20.0))
          .paint(0, BUBBLE, _px((60, 110))).paint(0, BACKGROUND, _px((60, 80))))
    bub, bg = (_as_set(a) for a in training_pixels(ex, 0, SHAPE, cfg))
    assert (59, 109) in bub and (59, 109) not in bg
    assert (59, 79) in bg and (59, 79) not in bub


def test_stroke_pixels_are_kept_before_sampling_the_automatic_ones():
    cfg = ExampleConfig(max_per_class=50)
    strokes = [(60 + k, 70) for k in range(-5, 5)]
    ex = MLExamples().verify(0, (60.0, 80.0, 20.0)).paint(0, BUBBLE, np.array(strokes))
    bub, _ = training_pixels(ex, 0, SHAPE, cfg)
    assert len(bub) == 50
    assert {(r - 1, c - 1) for r, c in strokes} <= _as_set(bub)


def test_pixels_outside_the_frame_are_ignored():
    ex = MLExamples().paint(0, BUBBLE, _px((5, 5), (500, 500)))
    bub, _ = training_pixels(ex, 0, SHAPE, ExampleConfig())
    assert bub.tolist() == [[4, 4]]
