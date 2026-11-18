"""Results, choices and cached pixels belong to one recording.

Regression tests for the 2026-11-16 review: a session restored never-confirmed
frame rate / scale as set; another recording kept the user-picked reference
frame and seed; a session restored onto different files of the same count;
the display cache served the previous video's frames; an interrupted save
truncated the previous session file.
"""

from __future__ import annotations

import json
import os
import shutil
import unicodedata

import numpy as np
import pytest

from bubbletrack.model.session import images_differ, load_session, save_session
from bubbletrack.model.state import AppState, update_state


@pytest.fixture
def fitted(gui, make_sequence):
    window, controller = gui
    controller.file_ctrl.on_folder_selected(str(make_sequence("shot", 12)))
    state = controller._get_state()
    state.radius[:] = np.linspace(20, 40, 12)
    state.circle_fit_par[:] = 50.0
    return window, controller


class TestConfirmedPhysicalValues:
    def test_unconfirmed_values_stay_unconfirmed_after_a_session(self, fitted, tmp_path):
        window, controller = fitted
        assert not controller.export_ctrl.physical_set()
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        assert controller.load_session_file(str(path))
        assert not controller.export_ctrl.physical_set()
        assert window.status_bar.scale_text() == "Scale: not set"
        assert controller.export_ctrl.make_dialog().unset_fields() == {"fps", "scale"}

    def test_one_confirmed_value_is_restored_alone(self, fitted, tmp_path):
        _, controller = fitted
        controller.export_ctrl.make_dialog()._scale_spin.setValue(2.5)
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        assert controller.load_session_file(str(path))
        assert controller.export_ctrl.make_dialog().unset_fields() == {"fps"}

    def test_a_session_from_before_the_mark_loads_as_set(self, fitted, tmp_path):
        _, controller = fitted
        path = tmp_path / "old.brt"
        assert controller.save_session_file(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["physical_set"]
        data["version"] = "3.2"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert controller.load_session_file(str(path))
        assert controller.export_ctrl.physical_set()


class TestAnotherRecording:
    def test_the_picked_reference_and_seed_do_not_carry_over(self, gui, make_sequence):
        _, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
        controller._set_state(update_state(controller._get_state(), reference_frame=1,
                                           edge_seed=(40.0, 50.0)))
        controller.file_ctrl.on_folder_selected(str(make_sequence("b", 6)))
        state = controller._get_state()
        assert state.reference_frame is None and state.edge_seed is None

    def test_the_display_cache_is_emptied(self, gui, make_sequence):
        _, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
        cache = controller.file_ctrl._cache
        cache.put("video:3:stale", (np.zeros((4, 4)),))
        controller.file_ctrl.on_folder_selected(str(make_sequence("b", 6)))
        assert cache.get("video:3:stale") is None

    def test_cached_references_are_forgotten(self, gui, make_sequence):
        from bubbletrack.controller import edge_support

        _, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
        edge_support._reference_cache[("stale",)] = 3
        controller.file_ctrl.on_folder_selected(str(make_sequence("b", 6)))
        assert ("stale",) not in edge_support._reference_cache


class TestSessionImages:
    def test_results_are_not_restored_onto_other_files(self, fitted, tmp_path):
        window, controller = fitted
        folder = controller._get_state().folder_path
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        # The same number of frames, other files (another shot of the camera)
        other = tmp_path / "other_shot"
        other.mkdir()
        for i, name in enumerate(sorted(os.listdir(folder))):
            shutil.copy(os.path.join(folder, name), other / f"x{i:03d}{os.path.splitext(name)[1]}")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["folder_path"] = str(other)
        path.write_text(json.dumps(data), encoding="utf-8")
        assert controller.load_session_file(str(path))
        state = controller._get_state()
        assert np.all(state.radius < 0)  # nothing restored
        assert "image names differ" in window.header.status_text()

    def test_the_same_files_restore(self, fitted, tmp_path):
        _, controller = fitted
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        assert controller.load_session_file(str(path))
        np.testing.assert_allclose(controller._get_state().radius, np.linspace(20, 40, 12))

    def test_names_match_across_unicode_spellings(self):
        nfc = unicodedata.normalize("NFC", "Ölçüm_001.tif")
        nfd = unicodedata.normalize("NFD", nfc)
        assert nfc != nfd
        assert images_differ([nfd], [f"/data/{nfc}"]) is None
        assert images_differ(["a.tif"], ["/data/b.tif"]) is not None
        assert images_differ(None, ["/data/b.tif"]) is None  # sessions before 3.3

    def test_a_video_session_stores_no_names(self, tmp_path):
        state = AppState(video_path="/v/clip.avi", images=("video:0", "video:1"))
        path = tmp_path / "v.brt"
        save_session(str(path), state)
        assert load_session(str(path))["image_names"] is None


class TestAtomicSave:
    def test_an_interrupted_save_keeps_the_previous_file(self, tmp_path, monkeypatch):
        from bubbletrack.model import atomic_write

        path = tmp_path / "s.brt"
        save_session(str(path), AppState(folder_path="/first"))
        before = path.read_bytes()

        def fail(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(atomic_write.os, "replace", fail)
        with pytest.raises(OSError):
            save_session(str(path), AppState(folder_path="/second"))
        assert path.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["s.brt"]  # no temp file left
