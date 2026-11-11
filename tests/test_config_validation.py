"""config.json robustness: hand-edited or stale values must never crash a launch.

Every persisted key is validated on load: unambiguous values are coerced,
numbers are clamped to the UI's ranges, and anything else falls back to the
default (the key is dropped, so AppState's default applies) with a warning.
"""

from __future__ import annotations

import json
import logging
import math

import pytest

from bubbletrack.model import config
from bubbletrack.model.config import PERSIST_KEYS, load_config, save_config
from bubbletrack.model.state import AppState, update_state


@pytest.fixture(autouse=True)
def config_file(tmp_path, monkeypatch):
    """Redirect CONFIG_DIR / CONFIG_FILE to a temp directory (never ~/.bubbletrack)."""
    config_dir = tmp_path / ".bubbletrack"
    path = config_dir / "config.json"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", path)
    return path


def _write(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _load(path, **data) -> dict:
    _write(path, data)
    return load_config()


class TestWrongTypesNoLongerCrash:
    """The reported values that crashed every launch."""

    def test_numeric_string_is_coerced(self, config_file):
        assert _load(config_file, img_thr="0.5") == {"img_thr": 0.5}

    @pytest.mark.parametrize("key", ["fps", "gaussian_sigma", "um2px", "img_thr"])
    def test_null_falls_back_to_default(self, config_file, key):
        assert key not in _load(config_file, **{key: None})

    def test_fractional_int_falls_back_to_default(self, config_file):
        assert "rmax_fit_length" not in _load(config_file, rmax_fit_length=4.5)

    def test_whole_float_for_int_field_is_coerced(self, config_file):
        loaded = _load(config_file, rmax_fit_length=9.0, removing_factor="85")
        assert loaded == {"rmax_fit_length": 9, "removing_factor": 85}
        assert isinstance(loaded["rmax_fit_length"], int)

    @pytest.mark.parametrize("value", [True, "abc", [1], {"a": 1}, math.nan, math.inf])
    def test_non_numbers_fall_back(self, config_file, value):
        assert "img_thr" not in _load(config_file, img_thr=value)

    def test_bad_values_give_default_state(self, config_file):
        loaded = _load(
            config_file, img_thr="0.5", fps=None, gaussian_sigma=None, rmax_fit_length=4.5,
        )
        state = update_state(AppState(), **loaded)
        defaults = AppState()
        assert state.img_thr == 0.5
        assert state.fps == defaults.fps
        assert state.gaussian_sigma == defaults.gaussian_sigma
        assert state.rmax_fit_length == defaults.rmax_fit_length


class TestRanges:
    @pytest.mark.parametrize(("key", "raw", "expected"), [
        ("img_thr", 5, 1.0),
        ("img_thr", -0.2, 0.0),
        ("removing_factor", -20, 0),
        ("removing_factor", 250, 100),
        ("gaussian_sigma", 99.0, 20.0),
        ("gaussian_sigma", -1.0, 0.0),
        ("clahe_clip", 100.0, 40.0),
        ("closing_radius", 31, 30),
        ("opening_radius", -3, 0),
        ("area_max_regions", 0, 1),
        ("area_max_regions", 50, 20),
        ("um2px", 1e6, 9999.0),
        ("um2px", 0.0001, 0.001),
        ("fps", 1e15, 99_999e6),
        ("rmax_fit_length", 1, 3),
        ("rmax_fit_length", 5000, 999),
    ])
    def test_out_of_range_is_clamped(self, config_file, key, raw, expected):
        assert _load(config_file, **{key: raw}) == {key: expected}

    @pytest.mark.parametrize("key", ["fps", "um2px"])
    @pytest.mark.parametrize("raw", [0, -5.0])
    def test_non_positive_physical_scale_falls_back(self, config_file, key, raw):
        assert key not in _load(config_file, **{key: raw})

    def test_even_rmax_fit_length_is_kept(self, config_file):
        """The UI accepts even windows and the export handles them."""
        assert _load(config_file, rmax_fit_length=10) == {"rmax_fit_length": 10}


class TestDetectionMode:
    @pytest.mark.parametrize("mode", ["intensity", "gradient", "area"])
    def test_valid_modes(self, config_file, mode):
        assert _load(config_file, detection_mode=mode) == {"detection_mode": mode}

    def test_case_and_whitespace_are_normalised(self, config_file):
        assert _load(config_file, detection_mode=" Gradient ") == {"detection_mode": "gradient"}

    @pytest.mark.parametrize("mode", ["bogus", "", 3, None])
    def test_unknown_mode_falls_back(self, config_file, mode):
        assert "detection_mode" not in _load(config_file, detection_mode=mode)


class TestEdgesAndFlags:
    def test_four_flags_become_a_tuple_of_bools(self, config_file):
        loaded = _load(config_file, bubble_cross_edges=[1, 0, 0, True])
        assert loaded == {"bubble_cross_edges": (True, False, False, True)}

    @pytest.mark.parametrize("edges", [[True], [True] * 5, "TRDL", None, [2, 0, 0, 0], 1])
    def test_malformed_edges_fall_back(self, config_file, edges):
        assert "bubble_cross_edges" not in _load(config_file, bubble_cross_edges=edges)

    @pytest.mark.parametrize(("raw", "expected"), [(True, True), (0, False), (1, True)])
    def test_temporal_refine(self, config_file, raw, expected):
        assert _load(config_file, temporal_refine=raw) == {"temporal_refine": expected}

    @pytest.mark.parametrize("raw", ["yes", 2, None])
    def test_temporal_refine_invalid(self, config_file, raw):
        assert "temporal_refine" not in _load(config_file, temporal_refine=raw)


class TestGradientBoundary:
    @pytest.mark.parametrize(("raw", "expected"), [
        (0.0, 0.0), (0.5, 0.5), (1.0, 1.0),  # fractions, as the UI stores them
        (2, 0.5), (3, 1.0),                  # legacy ring stage 1..3, read like the UI
        (-1.0, 0.0), (9.0, 1.0),
    ])
    def test_values(self, config_file, raw, expected):
        assert _load(config_file, gradient_boundary=raw) == {"gradient_boundary": expected}


class TestWarningsAndCoverage:
    def test_rejected_and_clamped_values_are_logged(self, config_file, caplog):
        with caplog.at_level(logging.WARNING, logger="bubbletrack.model.config"):
            _load(config_file, img_thr=5, fps=None, detection_mode="bogus", um2px=3.2)
        text = caplog.text
        assert "img_thr" in text
        assert "fps" in text
        assert "detection_mode" in text
        assert "um2px" not in text  # valid values are silent

    def test_every_persisted_key_has_a_validator(self):
        assert set(config._VALIDATORS) == set(PERSIST_KEYS)

    def test_defaults_pass_validation_unchanged(self, config_file, caplog):
        with caplog.at_level(logging.WARNING, logger="bubbletrack.model.config"):
            save_config(AppState())
            loaded = load_config()
        defaults = AppState()
        assert loaded == {key: getattr(defaults, key) for key in PERSIST_KEYS}
        assert caplog.text == ""
