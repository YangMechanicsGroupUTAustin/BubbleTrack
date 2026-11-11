"""Tests for the scikit-image compatibility wrapper (model/skimage_compat.py)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from bubbletrack.model import skimage_compat
from bubbletrack.model.skimage_compat import remove_small_objects_compat


def _mask_with_blobs(sizes: list[int]) -> np.ndarray:
    """Return a bool mask holding one horizontal bar per entry of *sizes*.

    Bars are separated by empty rows so each one is its own component.
    """
    width = max(sizes) + 2
    mask = np.zeros((2 * len(sizes) + 1, width), dtype=bool)
    for i, size in enumerate(sizes):
        mask[2 * i + 1, 1:1 + size] = True
    return mask


def _component_sizes(mask: np.ndarray) -> list[int]:
    return sorted(int(row.sum()) for row in mask if row.any())


class TestOldSemantics:
    """The wrapper must keep ``min_size`` semantics: remove objects < min_size."""

    def test_object_of_exactly_min_size_is_kept(self):
        mask = _mask_with_blobs([5, 6, 7])
        out = remove_small_objects_compat(mask, 6)
        assert _component_sizes(out) == [6, 7]

    def test_object_one_below_min_size_is_removed(self):
        mask = _mask_with_blobs([5])
        out = remove_small_objects_compat(mask, 6)
        assert not out.any()

    @pytest.mark.parametrize("min_size", [0, 1])
    def test_tiny_min_size_removes_nothing(self, min_size):
        mask = _mask_with_blobs([1, 2])
        out = remove_small_objects_compat(mask, min_size)
        np.testing.assert_array_equal(out, mask)

    def test_does_not_mutate_input(self):
        mask = _mask_with_blobs([3, 9])
        before = mask.copy()
        remove_small_objects_compat(mask, 5)
        np.testing.assert_array_equal(mask, before)

    def test_emits_no_deprecation_warning(self):
        mask = _mask_with_blobs([3, 9])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            remove_small_objects_compat(mask, 5)


def test_detection_routes_every_call_through_the_wrapper():
    """No direct remove_small_objects call may remain in detection.py."""
    import inspect

    from bubbletrack.model import detection

    source = inspect.getsource(detection)
    assert "morphology.remove_small_objects" not in source
    assert "remove_small_objects_compat(" in source


class TestApiSelection:
    """The wrapper picks ``max_size`` when the installed version has it."""

    def test_uses_max_size_minus_one_on_new_api(self, monkeypatch):
        calls = []

        def new_api(ar, max_size=64, connectivity=1, *, out=None):
            calls.append({"max_size": max_size})
            return ar.copy()

        monkeypatch.setattr(skimage_compat.morphology, "remove_small_objects", new_api)
        remove_small_objects_compat(np.zeros((3, 3), bool), 40)
        assert calls == [{"max_size": 39}]

    def test_new_api_never_gets_negative_max_size(self, monkeypatch):
        calls = []

        def new_api(ar, max_size=64, connectivity=1, *, out=None):
            calls.append(max_size)
            return ar.copy()

        monkeypatch.setattr(skimage_compat.morphology, "remove_small_objects", new_api)
        remove_small_objects_compat(np.zeros((3, 3), bool), 0)
        assert calls == [0]

    def test_uses_min_size_on_old_api(self, monkeypatch):
        calls = []

        def old_api(ar, min_size=64, connectivity=1, *, out=None):
            calls.append({"min_size": min_size})
            return ar.copy()

        monkeypatch.setattr(skimage_compat.morphology, "remove_small_objects", old_api)
        remove_small_objects_compat(np.zeros((3, 3), bool), 40)
        assert calls == [{"min_size": 40}]

    def test_deprecated_signature_with_both_params_prefers_max_size(self, monkeypatch):
        """scikit-image 0.26 keeps ``min_size`` (deprecated) next to ``max_size``."""
        calls = []

        def transitional_api(ar, min_size=None, connectivity=1, *, max_size=64, out=None):
            calls.append({"min_size": min_size, "max_size": max_size})
            return ar.copy()

        monkeypatch.setattr(
            skimage_compat.morphology, "remove_small_objects", transitional_api,
        )
        remove_small_objects_compat(np.zeros((3, 3), bool), 40)
        assert calls == [{"min_size": None, "max_size": 39}]
