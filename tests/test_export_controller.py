"""Tests for ExportController wiring (status messages, export arguments).

The Qt window is replaced by light fakes and the save dialog is patched, so
these tests exercise only the controller's own logic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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

    def test_failure_is_shown_in_red(self, monkeypatch, tmp_path):
        radius = np.array([-1.0, 10.0, -1.0])
        panel = _FakePanel()
        ctrl = _make_controller(monkeypatch, str(tmp_path / "c.mat"), radius, panel)

        ctrl.on_export_rof_t()

        text, success = panel.statuses[-1]
        assert not success
        assert "Not enough valid frames" in text
