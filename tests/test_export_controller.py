"""Tests for ExportController wiring (status messages, export arguments).

The Qt window is replaced by light fakes and the save dialog is patched, so
these tests exercise only the controller's own logic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bubbletrack.controller import export_controller as ec
from bubbletrack.event_bus import EventBus
from bubbletrack.model.state import AppState

AMBER = "#f59e0b"


class _FakeLabel:
    def __init__(self) -> None:
        self.style = ""

    def setStyleSheet(self, style: str) -> None:  # noqa: N802 (Qt API)
        self.style = style


class _FakePanel:
    """Stand-in for PostProcessingPanel."""

    def __init__(self, fmt: str = "mat", fps: float = 1e6,
                 scale: float = 3.2, fit_length: int = 11) -> None:
        self._fmt = fmt
        self._fps = fps
        self._scale = scale
        self._fit_length = fit_length
        self._status = _FakeLabel()
        self.statuses: list[tuple[str, bool]] = []

    def get_format(self) -> str:
        return self._fmt

    def get_fps(self) -> float:
        return self._fps

    def get_scale(self) -> float:
        return self._scale

    def get_fit_length(self) -> int:
        return self._fit_length

    def set_status(self, text: str, success: bool = True) -> None:
        self.statuses.append((text, success))
        self._status.setStyleSheet(
            f"color:{'#10b981' if success else '#ef4444'}; font-size:11px;"
        )


def _make_controller(monkeypatch, save_path: str, radius: np.ndarray,
                     panel: _FakePanel):
    n = len(radius)
    state = AppState(
        folder_path="",
        images=tuple(f"img_{i:04d}.tif" for i in range(n)),
    ).with_results_initialized(n)
    np.copyto(state.radius, radius)
    state.circle_fit_par[:] = 50.0
    holder = {"state": state}

    window = SimpleNamespace(
        left_panel=SimpleNamespace(post_processing=panel),
        status_bar=SimpleNamespace(update_scale=lambda _scale: None),
    )
    monkeypatch.setattr(
        ec, "QFileDialog",
        SimpleNamespace(getSaveFileName=lambda *a, **k: (save_path, "")),
    )
    ctrl = ec.ExportController(
        EventBus(),
        lambda: holder["state"],
        lambda s: holder.update(state=s),
        window,
    )
    return ctrl


class TestRofTStatus:
    def test_clean_export_shows_green_status(self, monkeypatch, tmp_path):
        t = np.arange(1, 31, dtype=float)
        radius = 100.0 - 0.05 * (t - 15) ** 2
        panel = _FakePanel()
        ctrl = _make_controller(monkeypatch, str(tmp_path / "a.mat"), radius, panel)

        ctrl.on_export_rof_t()

        text, success = panel.statuses[-1]
        assert success
        assert text == "Exported: a.mat"
        assert AMBER not in panel._status.style

    def test_warning_is_shown_in_amber(self, monkeypatch, tmp_path):
        radius = np.array([100.0, 90.0, 80.0, 70.0, 60.0])  # peak at frame 1
        panel = _FakePanel()
        out = tmp_path / "b.mat"
        ctrl = _make_controller(monkeypatch, str(out), radius, panel)

        ctrl.on_export_rof_t()

        assert out.exists()
        text, _success = panel.statuses[-1]
        assert text.startswith("Exported with warning: ")
        assert "boundary" in text
        assert AMBER in panel._status.style

    def test_csv_passes_panel_values_and_shows_warning(self, monkeypatch, tmp_path):
        radius = np.array([100.0, 90.0, 80.0, 70.0, 60.0])  # peak at frame 1
        panel = _FakePanel(fmt="csv", fps=2e5, scale=1.5, fit_length=7)
        out = tmp_path / "d.csv"
        ctrl = _make_controller(monkeypatch, str(out), radius, panel)
        calls = []
        real_export_csv = ec.export_csv

        def _spy(*args, **kwargs):
            calls.append(kwargs)
            return real_export_csv(*args, **kwargs)

        monkeypatch.setattr(ec, "export_csv", _spy)

        ctrl.on_export_rof_t()

        assert out.exists()
        assert calls[0]["fps"] == 2e5
        assert calls[0]["um2px"] == 1.5
        assert calls[0]["rmax_fit_length"] == 7
        text, _success = panel.statuses[-1]
        assert text.startswith("Exported with warning: ")
        assert AMBER in panel._status.style

    def test_failure_is_shown_in_red(self, monkeypatch, tmp_path):
        radius = np.array([-1.0, 10.0, -1.0])
        panel = _FakePanel()
        ctrl = _make_controller(monkeypatch, str(tmp_path / "c.mat"), radius, panel)

        ctrl.on_export_rof_t()

        text, success = panel.statuses[-1]
        assert not success
        assert "Not enough valid frames" in text


def _smooth_radius(n: int = 30) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=float)
    return 100.0 - 0.05 * (t - n / 2) ** 2


class TestReport:
    def test_report_uses_panel_fps_and_scale(self, monkeypatch, tmp_path):
        panel = _FakePanel(fps=5e5, scale=2.5)
        out = tmp_path / "report.pdf"
        ctrl = _make_controller(monkeypatch, str(out), _smooth_radius(), panel)
        calls = []
        monkeypatch.setattr(
            ec, "generate_report",
            lambda *args, **kwargs: calls.append(kwargs),
        )

        ctrl.on_generate_report()

        assert calls[0]["fps"] == pytest.approx(5e5)
        assert calls[0]["um2px"] == pytest.approx(2.5)
        assert panel.statuses[-1] == ("Report saved: report.pdf", True)


class TestExportMetadata:
    """Every export carries provenance built from state + panel values."""

    def _check_meta(self, meta: dict, panel: _FakePanel) -> None:
        import bubbletrack

        assert meta["fps"] == pytest.approx(panel.get_fps())
        assert meta["um2px"] == pytest.approx(panel.get_scale())
        assert meta["rmax_fit_length"] == panel.get_fit_length()
        assert meta["detection_mode"] == "intensity"
        assert meta["app_version"] == bubbletrack.__version__
        assert meta["exported_at"]

    def test_pixel_mat_export(self, monkeypatch, tmp_path):
        from scipy.io import loadmat

        panel = _FakePanel(fmt="mat", fps=5e5, scale=2.5, fit_length=7)
        out = tmp_path / "r.mat"
        ctrl = _make_controller(monkeypatch, str(out), _smooth_radius(), panel)

        ctrl.on_export_r_data()

        meta = loadmat(str(out), simplify_cells=True)["BubbleTrackMeta"]
        self._check_meta(meta, panel)

    def test_rof_t_mat_export(self, monkeypatch, tmp_path):
        from scipy.io import loadmat

        panel = _FakePanel(fmt="mat", fps=5e5, scale=2.5, fit_length=7)
        out = tmp_path / "rt.mat"
        ctrl = _make_controller(monkeypatch, str(out), _smooth_radius(), panel)

        ctrl.on_export_rof_t()

        meta = loadmat(str(out), simplify_cells=True)["BubbleTrackMeta"]
        self._check_meta(meta, panel)

    def test_csv_export_writes_sidecar(self, monkeypatch, tmp_path):
        import json

        panel = _FakePanel(fmt="csv", fps=5e5, scale=2.5, fit_length=7)
        ctrl = _make_controller(
            monkeypatch, str(tmp_path / "run.csv"), _smooth_radius(), panel,
        )

        ctrl.on_export_rof_t()

        meta = json.loads((tmp_path / "run.meta.json").read_text(encoding="utf-8"))
        self._check_meta(meta, panel)

    def test_xlsx_pixel_export_writes_metadata_sheet(self, monkeypatch, tmp_path):
        from openpyxl import load_workbook

        panel = _FakePanel(fmt="xlsx", fps=5e5, scale=2.5, fit_length=7)
        out = tmp_path / "run.xlsx"
        ctrl = _make_controller(monkeypatch, str(out), _smooth_radius(), panel)

        ctrl.on_export_r_data()

        rows = list(load_workbook(str(out))["Metadata"].iter_rows(values_only=True))
        self._check_meta(dict(rows[1:]), panel)
