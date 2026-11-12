"""Teach features: a fixed multiscale filter bank of the frame and of the engine's view.

Pure numpy / scikit-image (no scikit-learn): these run everywhere, CI included.
"""

from __future__ import annotations

import numpy as np
import pytest

from _edge_synth import Bubble, Scene, render
from bubbletrack.model.teach import FeatureConfig, compute_features, feature_names

H, W = 120, 160
FULL = ((1, H), (1, W))


@pytest.fixture(scope="module")
def scene():
    img, info = render(Scene(h=H, w=W, bubbles=(Bubble(rc=60.0, cc=80.0, r=25.0),), seed=3))
    return img, info["ref"]


def test_default_stack_has_both_channels_at_full_resolution(scene):
    img, ref = scene
    cfg = FeatureConfig()
    stack = compute_features(img, *FULL, reference=ref, config=cfg)
    names = feature_names(cfg)
    assert stack.values.shape == (H, W, len(names))
    assert stack.values.dtype == np.float32 and np.isfinite(stack.values).all()
    assert stack.step == 1 and stack.roi_shape == (H, W)
    assert any(n.startswith("raw") for n in names) and any(n.startswith("view") for n in names)
    assert len(set(names)) == len(names)


def test_channels_and_texture_are_configurable(scene):
    img, ref = scene
    raw_only = FeatureConfig(channels=("raw",), texture=False)
    stack = compute_features(img, *FULL, reference=ref, config=raw_only)
    assert stack.values.shape[-1] == len(feature_names(raw_only)) == 2 * raw_only.num_sigma
    with pytest.raises(ValueError):
        FeatureConfig(channels=("raw", "colour"))
    with pytest.raises(ValueError):
        FeatureConfig(channels=())


def test_half_resolution_halves_the_grid_but_keeps_the_roi_shape(scene):
    img, ref = scene
    stack = compute_features(img, *FULL, reference=ref, config=FeatureConfig(half_resolution=True))
    assert stack.step == 2 and stack.roi_shape == (H, W)
    assert stack.values.shape[:2] == (H // 2, W // 2)


def test_features_cover_the_roi_only(scene):
    img, ref = scene
    stack = compute_features(img, (11, 100), (21, 140), reference=ref, config=FeatureConfig())
    assert stack.values.shape[:2] == (90, 120) and stack.roi_shape == (90, 120)


def test_no_reference_still_gives_the_same_feature_set(scene):
    img, ref = scene
    with_ref = compute_features(img, *FULL, reference=ref, config=FeatureConfig())
    without = compute_features(img, *FULL, reference=None, config=FeatureConfig())
    assert without.values.shape == with_ref.values.shape
    assert np.isfinite(without.values).all()


def test_features_are_deterministic_and_do_not_touch_the_input(scene):
    img, ref = scene
    before = img.copy()
    a = compute_features(img, *FULL, reference=ref, config=FeatureConfig())
    b = compute_features(img, *FULL, reference=ref, config=FeatureConfig())
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(img, before)


def test_raw_channel_ignores_a_global_brightness_change(scene):
    """Flicker: the same scene 30 % brighter gives (nearly) the same raw features."""
    img, _ = scene
    cfg = FeatureConfig(channels=("raw",))
    a = compute_features(img.astype(np.float32), *FULL, reference=None, config=cfg).values
    b = compute_features(img.astype(np.float32) * 1.3, *FULL, reference=None, config=cfg).values
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)


def test_view_channel_separates_bubble_from_background(scene):
    img, ref = scene
    cfg = FeatureConfig(channels=("view",), texture=False)
    stack = compute_features(img, *FULL, reference=ref, config=cfg)
    first = stack.values[..., 0]                      # finest-scale smoothed view
    assert np.median(first[5:20, 5:20]) == pytest.approx(1.0, abs=0.1)
    assert np.median(first[55:65, 60:70]) < 0.5
