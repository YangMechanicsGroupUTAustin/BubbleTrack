"""Tests for session persistence (model/session.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bubbletrack.model.session import (
    SESSION_VERSION,
    load_session,
    save_session,
)
from bubbletrack.model.state import AppState, update_state


class TestSaveSession:
    """Verify save_session writes the expected .brt JSON."""

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "test.brt")
        state = AppState()
        save_session(path, state)
        assert (tmp_path / "test.brt").exists()

    def test_output_is_valid_json(self, tmp_path):
        path = str(tmp_path / "test.brt")
        save_session(path, AppState())
        data = json.loads((tmp_path / "test.brt").read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_version_field(self, tmp_path):
        path = str(tmp_path / "test.brt")
        save_session(path, AppState())
        data = json.loads((tmp_path / "test.brt").read_text(encoding="utf-8"))
        assert data["version"] == SESSION_VERSION

    def test_saves_modified_parameters(self, tmp_path):
        state = update_state(
            AppState(),
            folder_path="/some/path",
            image_no=42,
            img_thr=0.75,
            gridx=(10, 200),
            gridy=(20, 300),
            removing_factor=60,
            bubble_cross_edges=(True, False, True, False),
            gaussian_sigma=1.5,
            clahe_clip=2.5,
            closing_radius=3,
            opening_radius=4,
            um2px=4.0,
            fps=500_000.0,
            rmax_fit_length=9,
        )
        path = str(tmp_path / "params.brt")
        save_session(path, state)
        data = json.loads((tmp_path / "params.brt").read_text(encoding="utf-8"))
        assert data["folder_path"] == "/some/path"
        assert data["image_no"] == 42
        assert data["img_thr"] == 0.75
        assert data["gridx"] == [10, 200]
        assert data["gridy"] == [20, 300]
        assert data["removing_factor"] == 60
        assert data["bubble_cross_edges"] == [True, False, True, False]
        assert data["gaussian_sigma"] == 1.5
        assert data["clahe_clip"] == 2.5
        assert data["closing_radius"] == 3
        assert data["opening_radius"] == 4
        assert data["um2px"] == 4.0
        assert data["fps"] == 500_000.0
        assert data["rmax_fit_length"] == 9

    def test_saves_numpy_results(self, tmp_path):
        state = AppState().with_results_initialized(5)
        # Set some radius values
        state.radius[0] = 10.0
        state.radius[2] = 25.0
        state.circle_fit_par[0] = [50, 60]
        path = str(tmp_path / "results.brt")
        save_session(path, state)
        data = json.loads((tmp_path / "results.brt").read_text(encoding="utf-8"))
        assert data["radius"][0] == 10.0
        assert data["radius"][2] == 25.0
        assert data["circle_fit_par"][0] == [50.0, 60.0]

    def test_saves_none_results(self, tmp_path):
        path = str(tmp_path / "empty.brt")
        save_session(path, AppState())
        data = json.loads((tmp_path / "empty.brt").read_text(encoding="utf-8"))
        assert data["radius"] is None
        assert data["circle_fit_par"] is None


class TestLoadSession:
    """Verify load_session reads and reconstructs data correctly."""

    def test_round_trip(self, tmp_path):
        state = update_state(
            AppState(),
            folder_path="/test/path",
            image_no=5,
            img_thr=0.6,
            gridx=(10, 200),
            gridy=(20, 300),
            removing_factor=70,
            bubble_cross_edges=(True, True, False, False),
            gaussian_sigma=2.0,
            um2px=5.0,
            fps=750_000.0,
        )
        path = str(tmp_path / "roundtrip.brt")
        save_session(path, state)
        loaded = load_session(path)
        assert loaded["folder_path"] == "/test/path"
        assert loaded["image_no"] == 5
        assert loaded["img_thr"] == 0.6
        assert loaded["gridx"] == (10, 200)
        assert loaded["gridy"] == (20, 300)
        assert loaded["removing_factor"] == 70
        assert loaded["bubble_cross_edges"] == (True, True, False, False)
        assert isinstance(loaded["bubble_cross_edges"], tuple)
        assert isinstance(loaded["gridx"], tuple)
        assert loaded["gaussian_sigma"] == 2.0
        assert loaded["um2px"] == 5.0
        assert loaded["fps"] == 750_000.0

    def test_reconstructs_numpy_arrays(self, tmp_path):
        state = AppState().with_results_initialized(3)
        state.radius[0] = 15.0
        state.radius[1] = 20.0
        state.circle_fit_par[0] = [100, 200]
        path = str(tmp_path / "numpy.brt")
        save_session(path, state)
        loaded = load_session(path)
        assert isinstance(loaded["radius"], np.ndarray)
        assert loaded["radius"].dtype == np.float64
        np.testing.assert_array_almost_equal(loaded["radius"][:2], [15.0, 20.0])
        assert isinstance(loaded["circle_fit_par"], np.ndarray)
        np.testing.assert_array_almost_equal(
            loaded["circle_fit_par"][0], [100.0, 200.0],
        )

    def test_none_results_stay_none(self, tmp_path):
        path = str(tmp_path / "none.brt")
        save_session(path, AppState())
        loaded = load_session(path)
        assert loaded["radius"] is None
        assert loaded["circle_fit_par"] is None

    def test_rejects_old_version(self, tmp_path):
        path = tmp_path / "old.brt"
        path.write_text(
            json.dumps({"version": "1.0", "folder_path": "/x"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="too old"):
            load_session(str(path))

    def test_rejects_invalid_json(self, tmp_path):
        path = tmp_path / "bad.brt"
        path.write_text("not json {{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid session file"):
            load_session(str(path))

    def test_rejects_non_object_json(self, tmp_path):
        path = tmp_path / "array.brt"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="not a JSON object"):
            load_session(str(path))

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(str(tmp_path / "nonexistent.brt"))

    def test_can_apply_to_appstate(self, tmp_path):
        """Loaded session data can create a valid AppState via update_state."""
        original = update_state(
            AppState(),
            folder_path="/data",
            img_thr=0.8,
            fps=100_000.0,
        )
        path = str(tmp_path / "apply.brt")
        save_session(path, original)
        loaded = load_session(path)
        loaded.pop("version", None)
        loaded.pop("image_names", None)  # checked on restore, not a state field
        loaded.pop("source_fingerprint", None)
        restored = update_state(AppState(), **loaded)
        assert restored.folder_path == "/data"
        assert restored.img_thr == 0.8
        assert restored.fps == 100_000.0

    def test_round_trip_with_results_applied(self, tmp_path):
        """Full round-trip including results arrays applied to state."""
        state = update_state(
            AppState().with_results_initialized(4),
            folder_path="/test",
            image_no=2,
        )
        state.radius[1] = 33.0
        state.circle_fit_par[1] = [50, 60]
        path = str(tmp_path / "full.brt")
        save_session(path, state)
        loaded = load_session(path)
        loaded.pop("version", None)
        loaded.pop("image_names", None)  # checked on restore, not a state field
        loaded.pop("source_fingerprint", None)
        restored = update_state(AppState(), **loaded)
        assert restored.folder_path == "/test"
        assert restored.image_no == 2
        assert isinstance(restored.radius, np.ndarray)
        assert restored.radius[1] == 33.0


def test_session_json_is_one_key_per_line_and_strict(tmp_path):
    """Compact per-frame arrays (the file halves at 10k frames), readable keys, no NaN."""
    import math

    import pytest

    from bubbletrack.model.session import session_json

    text = session_json({"version": "3.3", "radius": [1.5, None], "flags": [["a"], []]})
    assert text.splitlines() == ['{', '  "version": "3.3",', '  "radius": [1.5,null],',
                                 '  "flags": [["a"],[]]', '}']
    assert json.loads(text)["radius"] == [1.5, None]
    with pytest.raises(ValueError):
        session_json({"radius": [math.nan]})
