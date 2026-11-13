"""ExportController v3.1: fit-result columns, remembered folder/format, default names.

The window is replaced by light fakes and the exports are called with a
path (the Export dialog supplies it); the config is isolated so nothing
touches ~/.bubbletrack.
"""

from __future__ import annotations

import csv
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat

from bubbletrack.controller import edge_support
from bubbletrack.controller import export_controller as ec
from bubbletrack.event_bus import EventBus
from bubbletrack.model import config
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState

STAMP = r"\d{8}_\d{6}"


@pytest.fixture(autouse=True)
def _config(isolated_config):
    return isolated_config


class _Panel:
    def __init__(self, fmt: str = "mat") -> None:
        self.fmt = fmt
        self.statuses: list[tuple[str, bool]] = []
        self._status = SimpleNamespace(setStyleSheet=lambda _style: None)

    def get_format(self) -> str:
        return self.fmt

    def get_fps(self) -> float:
        return 1e6

    def get_scale(self) -> float:
        return 3.2

    def get_fit_length(self) -> int:
        return 11

    def set_status(self, text: str, success: bool = True) -> None:
        self.statuses.append((text, success))


def _radius(n: int = 30) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=float)
    return 100.0 - 0.05 * (t - 15.3) ** 2


def _make(*, panel=None, n: int = 30, **state_fields):
    state = AppState(images=tuple(f"img_{i:03d}.tif" for i in range(n)),
                     **state_fields).with_results_initialized(n)
    np.copyto(state.radius, _radius(n))
    state.circle_fit_par[:] = 50.0
    state.fit_source[:] = FitSource.AUTO
    state.fit_source[3] = FitSource.MANUAL
    state.fit_confidence[:] = 0.8
    state.fit_locked[3] = True
    state.fit_flags[4] = frozenset({"fragmented"})
    holder = {"state": state}
    window = SimpleNamespace(
        status_bar=SimpleNamespace(update_scale=lambda _scale: None),
        radius_chart=None,
    )
    ctrl = ec.ExportController(EventBus(), lambda: holder["state"],
                               lambda s: holder.update(state=s), window)
    return ctrl, panel or _Panel()


class TestFitResultsReachTheFiles:
    def test_pixel_csv_has_the_new_columns(self, monkeypatch, tmp_path):
        out = tmp_path / "r.csv"
        ctrl, panel = _make(panel=_Panel("csv"))
        ctrl.export_pixel(panel, str(out))
        assert panel.statuses[-1][1], panel.statuses
        with open(out, newline="", encoding="utf-8") as f:
            header, *rows = list(csv.reader(f))
        for column in ("Source", "Confidence", "Locked", "Flags", "R_Circle_px"):
            assert column in header
        row4 = dict(zip(header, rows[3]))
        assert (row4["Source"], row4["Locked"]) == ("MANUAL", "1")
        assert dict(zip(header, rows[4]))["Flags"] == "fragmented"

    def test_pixel_mat_has_the_new_variables(self, monkeypatch, tmp_path):
        out = tmp_path / "r.mat"
        ctrl, panel = _make()
        ctrl.export_pixel(panel, str(out))
        data = loadmat(str(out), simplify_cells=True)
        assert data["FitSource"][3] == FitSource.MANUAL
        assert data["FitLocked"][3] == 1
        assert "FitMetrics" in data

    def test_rof_t_mat_has_confidence_and_radius_definition(self, monkeypatch, tmp_path):
        out = tmp_path / "rt.mat"
        ctrl, panel = _make(detection_mode="area")
        ctrl.export_physical(panel, str(out))
        data = loadmat(str(out), simplify_cells=True)
        assert len(data["Confidence"]) == len(data["R"])
        assert data["BubbleTrackMeta"]["radius_definition"] == (
            "area-equivalent (dark regions)")

    def test_mismatched_field_is_dropped_not_fatal(self, monkeypatch, tmp_path, caplog):
        out = tmp_path / "r.csv"
        ctrl, panel = _make(panel=_Panel("csv"))
        state = ctrl.state
        from bubbletrack.model.state import update_state
        ctrl.state = update_state(state, fit_flags=[frozenset()] * 3)  # stale length
        ctrl.export_pixel(panel, str(out))
        assert panel.statuses[-1][1], panel.statuses
        with open(out, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert "Flags" not in header and "Source" in header
        assert "fit_flags" in caplog.text

    def test_edge_metadata_records_the_used_reference_frame(self, monkeypatch, tmp_path):
        monkeypatch.setattr(edge_support, "reference_index", lambda _state: 6)
        out = tmp_path / "r.mat"
        ctrl, panel = _make(detection_mode="edge", edge_seed=(10.5, 20.5))
        ctrl.export_pixel(panel, str(out))
        meta = loadmat(str(out), simplify_cells=True)["BubbleTrackMeta"]
        assert meta["reference_frame"] == 7
        assert meta["reference_frame_origin"] == "auto"
        assert meta["edge_seed_used"]

    def test_unresolvable_reference_frame_does_not_block_export(self, monkeypatch, tmp_path):
        def _boom(_state):
            raise OSError("image gone")

        monkeypatch.setattr(edge_support, "reference_index", _boom)
        out = tmp_path / "r.mat"
        ctrl, panel = _make(detection_mode="edge")
        ctrl.export_pixel(panel, str(out))
        assert panel.statuses[-1][1], panel.statuses
        assert out.exists()


class TestRememberedFolderAndFormat:
    def test_successful_export_is_remembered(self, tmp_path):
        out = tmp_path / "exports" / "r.csv"
        out.parent.mkdir()
        ctrl, panel = _make(panel=_Panel("csv"))
        ctrl.export_pixel(panel, str(out))
        assert config.get_meta("last_export_dir") == str(out.parent)
        assert config.get_meta("last_export_format") == "csv"

    def test_next_dialog_starts_in_the_remembered_folder(self, tmp_path):
        remembered = tmp_path / "remembered"
        remembered.mkdir()
        config.set_meta(last_export_dir=str(remembered))
        ctrl, _ = _make(folder_path=str(tmp_path / "src"))
        assert os.path.dirname(ctrl.default_path("pixel", "mat")) == str(remembered)

    def test_missing_remembered_folder_falls_back_to_the_source(self, tmp_path):
        config.set_meta(last_export_dir=str(tmp_path / "gone"))
        source = tmp_path / "shot_042"
        ctrl, _ = _make(folder_path=str(source))
        assert os.path.dirname(ctrl.default_path("physical", "mat")) == str(source)

    def test_video_source_folder_is_the_fallback(self, tmp_path):
        video = tmp_path / "videos" / "run7.avi"
        ctrl, _ = _make(video_path=str(video))
        assert os.path.dirname(ctrl.default_path("pixel", "mat")) == str(video.parent)

    def test_missing_path_and_failure_remember_nothing(self, tmp_path):
        ctrl, panel = _make()
        assert ctrl.export_pixel(panel, "") is None  # no file chosen
        bad = tmp_path / "no_such_dir" / "r.mat"
        assert ctrl.export_pixel(panel, str(bad)) is None  # write fails
        assert not panel.statuses[-1][1]
        assert config.get_meta("last_export_dir") is None
        assert config.get_meta("last_export_format") is None

    def test_report_remembers_the_folder_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ec, "generate_report", lambda *a, **k: None)
        ctrl, panel = _make()
        ctrl.export_report(panel, str(tmp_path / "report.pdf"))
        assert config.get_meta("last_export_dir") == str(tmp_path)
        assert config.get_meta("last_export_format") is None


class TestDefaultName:
    @pytest.mark.parametrize("fields, units, pattern", [
        ({"folder_path": "/data/shot_042"}, "pixel", rf"shot_042_R_data_{STAMP}\.mat"),
        ({"video_path": "/v/run7.avi"}, "physical", rf"run7_RofT_data_{STAMP}\.mat"),
        ({}, "pixel", rf"R_data_{STAMP}\.mat"),
    ])
    def test_name_is_source_stem_plus_timestamp(self, fields, units, pattern):
        ctrl, _ = _make(**fields)
        name = os.path.basename(ctrl.default_path(units, "mat"))
        assert re.fullmatch(pattern, name), name

    def test_extension_follows_the_format(self):
        ctrl, _ = _make()
        assert ctrl.default_path("pixel", "xlsx").endswith(".xlsx")

    def test_unsafe_characters_are_replaced(self):
        assert ec.source_stem(AppState(video_path="/v/a:b*c?.avi")) == "a_b_c_"


class TestFormatPreselection:
    def test_saved_format_is_preselected_in_the_dialog(self, qapp):
        config.set_meta(last_export_format="xlsx")
        ctrl, _ = _make()
        ctrl.w = None  # the dialog's parent
        assert ctrl.make_dialog().get_format() == "xlsx"

    def test_no_saved_format_starts_with_mat(self, qapp):
        ctrl, _ = _make()
        ctrl.w = None
        assert ctrl.make_dialog().get_format() == "mat"

    def test_gui_starts_with_the_saved_format(self, qapp, isolated_config):
        from bubbletrack.controller.controller import AppController
        from bubbletrack.ui.main_window import MainWindow

        config.set_meta(last_export_format="xlsx")
        window = MainWindow()
        controller = AppController(window)
        try:
            assert controller.export_ctrl.make_dialog().get_format() == "xlsx"
        finally:
            window.close()
