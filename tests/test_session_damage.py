"""Hand-edited or damaged session files: an invalid setting is ignored (and named),
damaged results refuse the file; nothing may escape the load or break the window."""

from __future__ import annotations

import json
import logging
import sys
import traceback

import numpy as np
import pytest

from bubbletrack.model.config import check_session_settings
from bubbletrack.model.session import load_session

_BAD = [None, "x", -1, 1e308, True, [], {}, [1, "a"], ["x"] * 4]


class TestSettings:
    def test_invalid_values_are_dropped_and_named(self):
        clean, invalid = check_session_settings({
            "img_thr": "x", "closing_radius": 10 ** 400, "gridx": [1, None], "edge_seed": [3],
            "detection_mode": "EDGE", "reference_frame": None, "fps": 5e6, "ml_assist": True})
        assert sorted(invalid) == ["closing_radius", "edge_seed", "gridx", "img_thr"]
        assert clean == {"detection_mode": "edge", "reference_frame": None, "fps": 5e6,
                         "ml_assist": True}  # unchecked keys pass through

    def test_valid_values_keep_their_meaning(self):
        clean, invalid = check_session_settings({
            "gridx": [1, 250], "gridy": [3, 400], "image_no": 4, "reference_frame": 2,
            "edge_seed": [10, 20.5], "bubble_cross_edges": [True, False, False, True]})
        assert not invalid
        assert clean == {"gridx": (1, 250), "gridy": (3, 400), "image_no": 4, "reference_frame": 2,
                         "edge_seed": (10.0, 20.5), "bubble_cross_edges": (True, False, False, True)}


class TestDamagedResults:
    @pytest.mark.parametrize("key, value", [
        ("radius", {"a": 1}), ("radius", 5), ("radius", [[1, 2], [3, 4]]),
        ("circle_fit_par", [1, 2]), ("fit_flags", [1, 2]), ("fit_flags", "abc"),
        ("fit_locked", ["x", "y"]), ("fit_source", [99, 1]), ("fit_source", [1e308, 1]),
        ("fit_metrics", [1]), ("fit_confidence", 3), ("folder_path", [1]), ("image_names", "a.tif"),
    ])
    def test_refuse_the_file(self, tmp_path, key, value):
        path = tmp_path / "s.brt"
        path.write_text(json.dumps({"version": "3.3", key: value}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(str(path))

    def test_infinity_is_damage_not_a_radius(self, tmp_path):
        path = tmp_path / "s.brt"
        path.write_text('{"version": "3.3", "radius": [20.0, Infinity]}', encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(str(path))


def test_a_damaged_session_never_breaks_the_window(gui, make_sequence, tmp_path, monkeypatch, caplog):
    """Every kind of key gets wrong-typed values: settings, ROI, Auto setup, results, paths."""
    from PyQt6.QtWidgets import QMessageBox

    window, ctl = gui
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(ctl.session_ctrl, "_locate_folder", lambda _f: "")
    monkeypatch.setattr(ctl.session_ctrl, "confirm_leave", lambda: True)
    ctl.file_ctrl.on_folder_selected(str(make_sequence("s", 4, r_min=20, r_max=30)))
    ctl._get_state().radius[:] = [20.0, 22.0, 24.0, 26.0]
    good = tmp_path / "good.brt"
    assert ctl.save_session_file(str(good))
    data = json.loads(good.read_text(encoding="utf-8"))
    slot_errors = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda t, v, tb: slot_errors.append("".join(traceback.format_exception(t, v, tb))))
    caplog.set_level(logging.ERROR)
    keys = ("img_thr", "closing_radius", "detection_mode", "edge_definition", "gridx", "image_no",
            "edge_seed", "reference_frame", "physical_set", "radius", "fit_source", "folder_path")
    for key in keys:
        for bad in _BAD:
            path = tmp_path / "bad.brt"
            path.write_text(json.dumps({**data, key: bad}), encoding="utf-8")
            caplog.clear()
            ctl.load_session_file(str(path))
            for i in range(len(ctl._get_state().images or ())):
                ctl.file_ctrl.on_frame_changed(i)
            errors = [r.getMessage() for r in caplog.records if "Session load failed" not in r.getMessage()]
            assert not errors and not slot_errors, (key, bad, errors, slot_errors[:1])
    assert ctl.load_session_file(str(good))  # and the good file still loads
    np.testing.assert_allclose(ctl._get_state().radius, [20.0, 22.0, 24.0, 26.0])
