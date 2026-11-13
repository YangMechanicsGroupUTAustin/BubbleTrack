"""Teach pixel classifier: training from examples, probability maps, determinism, novelty.

Needs scikit-learn (not a project dependency yet): skipped where it is missing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from _edge_synth import Bubble, Scene, render  # noqa: E402
from bubbletrack.model.teach import (  # noqa: E402
    BACKGROUND,
    BUBBLE,
    FeatureConfig,
    MLConfig,
    MLError,
    MLExamples,
    stroke_pixels,
    train_model,
)

H, W = 120, 160
FULL = ((1, H), (1, W))
CIRCLES = {0: (60.0, 80.0, 30.0), 1: (58.0, 84.0, 18.0), 2: (62.0, 78.0, 24.0)}   # 0-based centres


@pytest.fixture(scope="module")
def frames():
    out, ref = {}, None
    for i, (rc, cc, r) in CIRCLES.items():
        img, info = render(Scene(h=H, w=W, bubbles=(Bubble(rc=rc, cc=cc, r=r),), speckle=0.3, seed=10 + i))
        out[i] = img
        ref = info["ref"] if ref is None else ref
    return out, ref


def _verified(*frames_):
    ex = MLExamples()
    for i in frames_:
        rc, cc, r = CIRCLES[i]
        ex = ex.verify(i, (rc + 1, cc + 1, r))           # 1-based, like the app
    return ex


def _disk(shape, rc, cc, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.hypot(yy - rc, xx - cc) < r


def test_model_from_verified_frames_finds_the_bubble_in_a_new_frame(frames):
    imgs, ref = frames
    model = train_model(_verified(0, 1), imgs.__getitem__, *FULL, reference=ref)
    p = model.probability(imgs[2])
    assert p.shape == (H, W) and p.dtype == np.float32
    assert 0.0 <= p.min() and p.max() <= 1.0
    rc, cc, r = CIRCLES[2]
    assert p[_disk(p.shape, rc, cc, 0.7 * r)].mean() > 0.85
    assert p[~_disk(p.shape, rc, cc, 1.5 * r)].mean() < 0.1
    assert model.info.frames == (0, 1)
    assert model.info.n_bubble > 0 and model.info.n_background > 0


def test_training_is_deterministic_and_survives_a_json_round_trip(frames):
    imgs, ref = frames
    ex = _verified(0, 1)
    again = MLExamples.from_dict(json.loads(json.dumps(ex.to_dict())))
    p1 = train_model(ex, imgs.__getitem__, *FULL, reference=ref).probability(imgs[2])
    p2 = train_model(again, imgs.__getitem__, *FULL, reference=ref).probability(imgs[2])
    np.testing.assert_array_equal(p1, p2)


def test_a_few_strokes_are_enough_on_a_clean_scene(frames):
    imgs, ref = frames
    rc, cc, r = CIRCLES[0]
    inside = stroke_pixels([(rc + 1, cc + 1 - 0.5 * r), (rc + 1, cc + 1 + 0.5 * r)], radius=2, shape=(H, W))
    outside = stroke_pixels([(10.0, 10.0), (10.0, 150.0), (110.0, 150.0)], radius=2, shape=(H, W))
    ex = MLExamples().paint(0, BUBBLE, inside).paint(0, BACKGROUND, outside)
    p = train_model(ex, imgs.__getitem__, *FULL, reference=ref).probability(imgs[2])
    rc2, cc2, r2 = CIRCLES[2]
    assert p[_disk(p.shape, rc2, cc2, 0.6 * r2)].mean() > 0.7
    assert p[~_disk(p.shape, rc2, cc2, 1.6 * r2)].mean() < 0.2


def test_half_resolution_prediction_is_upsampled_to_the_roi(frames):
    imgs, ref = frames
    cfg = MLConfig(features=FeatureConfig(half_resolution=True))
    model = train_model(_verified(0, 1), imgs.__getitem__, (11, 110), (21, 150), reference=ref, config=cfg)
    p = model.probability(imgs[2])
    assert p.shape == (100, 130)
    rc, cc, r = CIRCLES[2]
    assert p[_disk(p.shape, rc - 10, cc - 20, 0.7 * r)].mean() > 0.8


def test_training_needs_both_classes(frames):
    imgs, ref = frames
    only_bubble = MLExamples().paint(0, BUBBLE, np.array([[60, 80], [61, 80]]))
    with pytest.raises(MLError):
        train_model(only_bubble, imgs.__getitem__, *FULL, reference=ref)
    with pytest.raises(MLError):
        train_model(MLExamples(), imgs.__getitem__, *FULL, reference=ref)


def test_frame_of_the_wrong_size_is_rejected(frames):
    imgs, ref = frames
    model = train_model(_verified(0, 1), imgs.__getitem__, *FULL, reference=ref)
    with pytest.raises(ValueError):
        model.probability(np.zeros((50, 50), np.uint16))


def test_novelty_is_low_on_the_same_kind_of_frame_and_high_on_another(frames):
    imgs, ref = frames
    model = train_model(_verified(0, 1), imgs.__getitem__, *FULL, reference=ref)
    same = model.predict(imgs[2]).novelty
    rng = np.random.default_rng(0)
    other = (rng.random((H, W)) * 60000).astype(np.uint16)          # nothing like the training frames
    assert 0.0 <= same < 0.05
    assert model.predict(other).novelty > max(4 * same, 0.08)
