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


class TestRunResults:
    """What one Fit All run writes into a frame it measured."""

    def test_a_classic_result_carries_no_earlier_auto_meta(self, fitted):
        _, controller = fitted
        state = controller._get_state()
        state.fit_confidence[3] = 0.99
        state.fit_flags[3] = frozenset({"no_reference"})
        xy = np.array([[10.0, 10.0], [12.0, 10.0], [11.0, 12.0]])
        controller.auto_ctrl._on_auto_frame_done(3, 30.0, xy, None, 50.0, 50.0, 1)
        assert state.radius[3] == 30.0
        assert np.isnan(state.fit_confidence[3]) and state.fit_flags[3] == frozenset()

    def test_a_frame_that_cannot_be_read_keeps_its_result(self, fitted):
        _, controller = fitted
        state = controller._get_state()
        auto = controller.auto_ctrl
        auto._on_worker_frame_error(auto._worker, 4, "file is locked")
        auto._on_auto_frame_done(4, -1.0, np.empty((0, 2)), None, np.nan, np.nan, 0)
        assert state.radius[4] == pytest.approx(np.linspace(20, 40, 12)[4])
        # A frame with no result before stays empty
        state.radius[5] = -1.0
        auto._on_worker_frame_error(auto._worker, 5, "file is locked")
        auto._on_auto_frame_done(5, -1.0, np.empty((0, 2)), None, np.nan, np.nan, 0)
        assert state.radius[5] == -1.0


class TestAutoTune:
    """Classic Auto Tune applies its result only where it was started."""

    def _result(self):
        from bubbletrack.model.autotune import AutoTuneResult
        from bubbletrack.model.quality import FitQuality

        quality = FitQuality(score=0.9, n_edge_points=50, edge_density=1.0, rms_residual=0.5,
                             sector_coverage=1.0, radius=30.0)
        return AutoTuneResult(threshold=0.44, removing_factor=14, quality=quality,
                              candidates_evaluated=10)

    @pytest.fixture
    def classic(self, gui, make_sequence):
        window, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
        controller._set_state(update_state(controller._get_state(), detection_mode="intensity",
                                           img_thr=0.58, removing_factor=85, gaussian_sigma=3.0))
        return window, controller

    def test_a_frame_change_during_the_tune_applies_nothing(self, classic):
        window, controller = classic
        pretune = controller.pretune_ctrl
        pretune._autotune_target = pretune._tune_target()
        controller.file_ctrl.on_frame_changed(3)
        pretune._on_autotune_finished(self._result())
        assert controller._get_state().img_thr == 0.58
        assert "nothing applied" in window.header.status_text()

    def test_a_locked_frame_keeps_its_result(self, classic):
        from bubbletrack.model.provenance import FitSource

        _, controller = classic
        state = controller._get_state()
        state.radius[0] = 47.0
        state.fit_source[0] = FitSource.MANUAL
        state.fit_locked[0] = True
        pretune = controller.pretune_ctrl
        pretune._autotune_target = pretune._tune_target()
        pretune._on_autotune_finished(self._result())
        state = controller._get_state()
        assert state.img_thr == 0.44  # the tuned values are applied ...
        assert state.radius[0] == 47.0 and state.fit_source[0] == FitSource.MANUAL  # ... not stored

    def test_loading_another_recording_cancels_the_tune(self, classic, make_sequence):
        _, controller = classic
        pretune = controller.pretune_ctrl
        cancelled = []

        class FakeWorker:
            class _Sig:
                def disconnect(self, _slot):
                    pass
            result_ready = _Sig()
            progress = _Sig()

            def isRunning(self):  # noqa: N802 (Qt name)
                return True

            def cancel(self):
                cancelled.append(True)

            def wait(self):
                pass

        pretune._autotune_worker = FakeWorker()
        controller.file_ctrl.on_folder_selected(str(make_sequence("b", 6)))
        assert cancelled == [True] and pretune._autotune_worker is None


class TestFolderNotes:
    def test_a_tiff_stack_and_left_out_files_are_reported(self, gui, tmp_path):
        import cv2

        window, controller = gui
        folder = tmp_path / "stack"
        folder.mkdir()
        pages = [np.full((40, 50), 200, np.uint8) for _ in range(4)]
        assert cv2.imwritemulti(str(folder / "stack.tif"), pages)
        cv2.imwrite(str(folder / "extra.png"), pages[0])
        controller.file_ctrl.on_folder_selected(str(folder))
        text = window.header.status_text()
        assert text.startswith("Ready — ")
        assert "stack.tif has 4 pages" in text and "Ignored 1 .png" in text

    def test_a_plain_folder_just_says_ready(self, gui, make_sequence):
        window, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("a", 4)))
        assert window.header.status_text() == "Ready"


class TestRefusedRestore:
    def test_no_choice_of_the_other_recording_is_restored(self, fitted, tmp_path):
        """Results refused (other files): neither the picked reference and seed nor the
        confirmed frame rate and scale of that recording may apply here."""
        _, controller = fitted
        controller._set_state(update_state(controller._get_state(), reference_frame=3,
                                           edge_seed=(50.5, 60.5),
                                           physical_set=frozenset({"fps", "scale"})))
        path = tmp_path / "s.brt"
        assert controller.save_session_file(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        data["image_names"] = [f"other_{i}.tif" for i in range(len(data["image_names"]))]
        path.write_text(json.dumps(data), encoding="utf-8")
        assert controller.load_session_file(str(path))
        state = controller._get_state()
        assert state.reference_frame is None and state.edge_seed is None
        assert not controller.export_ctrl.physical_set()

    def test_auto_tune_applies_nothing_after_a_method_switch(self, gui, make_sequence):
        """Switching to Auto during a classic tune made its fit take the Edge path,
        which erased a locked manual fix."""
        from bubbletrack.model.autotune import AutoTuneResult
        from bubbletrack.model.provenance import FitSource
        from bubbletrack.model.quality import FitQuality

        _, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("a", 6)))
        controller._set_state(update_state(controller._get_state(), detection_mode="intensity"))
        state = controller._get_state()
        state.radius[0], state.fit_source[0], state.fit_locked[0] = 77.0, FitSource.MANUAL, True
        pretune = controller.pretune_ctrl
        pretune._autotune_target = pretune._tune_target()
        controller._set_state(update_state(controller._get_state(), detection_mode="edge"))
        quality = FitQuality(score=0.9, n_edge_points=50, edge_density=1.0, rms_residual=0.5,
                             sector_coverage=1.0, radius=30.0)
        pretune._on_autotune_finished(AutoTuneResult(0.44, 14, quality, 10))
        state = controller._get_state()
        assert state.radius[0] == 77.0 and state.fit_source[0] == FitSource.MANUAL


def test_a_refused_log_rollover_keeps_appending(tmp_path, monkeypatch):
    """Two instances on Windows: the rename is refused; records must not be lost."""
    import logging

    from bubbletrack import logging_config

    logger = logging.getLogger("bubbletrack")
    handlers, level = list(logger.handlers), logger.level
    try:
        logging_config.setup_logging(log_dir=tmp_path)
        handler = next(h for h in logger.handlers if isinstance(h, logging_config._SharedRotatingFileHandler))
        handler.maxBytes = 200

        def refuse(*_a, **_k):
            raise PermissionError(32, "being used by another process")

        monkeypatch.setattr(logging_config.logging.handlers.RotatingFileHandler, "rotate", refuse)
        for i in range(50):
            logger.warning("record %d", i)
        handler.flush()
        text = (tmp_path / "bubbletrack.log").read_text(encoding="utf-8")
        assert "record 0" in text and "record 49" in text
    finally:
        for h in [h for h in logger.handlers if h not in handlers]:
            logger.removeHandler(h)
            h.close()
        logger.setLevel(level)
