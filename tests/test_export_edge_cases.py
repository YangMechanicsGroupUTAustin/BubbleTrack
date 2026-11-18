"""Export and input edge cases found in the 2026-11-16 review."""

from __future__ import annotations

import numpy as np
import scipy.io

from bubbletrack.model.export import export_csv, export_r_data
from bubbletrack.model.folder_scan import natural_sort_key
from bubbletrack.model.report import generate_report, pdf_text
from bubbletrack.model.state import AppState, update_state


def _outline(n_points: int, r: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([50 + r * np.sin(t), 60 + r * np.cos(t)])


class TestCircleXY:
    def _load(self, tmp_path, edges):
        n = len(edges)
        path = tmp_path / "r.mat"
        export_r_data(str(path), np.full(n, 20.0), np.full((n, 2), 50.0), edges)
        return scipy.io.loadmat(str(path))["CircleXY"]

    def test_equal_length_outlines_stay_one_cell_per_frame(self, tmp_path):
        cells = self._load(tmp_path, [_outline(40, 20), _outline(40, 21), _outline(40, 22)])
        assert cells.shape == (1, 3)
        assert cells[0, 1].shape == (40, 2)
        np.testing.assert_allclose(cells[0, 1], _outline(40, 21))

    def test_one_frame(self, tmp_path):
        cells = self._load(tmp_path, [_outline(12, 5)])
        assert cells.shape == (1, 1) and cells[0, 0].shape == (12, 2)

    def test_unequal_lengths_and_missing_outlines(self, tmp_path):
        cells = self._load(tmp_path, [_outline(10, 5), None, _outline(7, 6)])
        assert cells.shape == (1, 3)
        assert cells[0, 0].shape == (10, 2) and cells[0, 1].size == 0


def test_csv_rows_skip_nan_radii(tmp_path):
    radius = np.array([20.0, np.nan, -1.0, 22.0])
    path = tmp_path / "r.csv"
    export_csv(str(path), radius, np.full((4, 2), 50.0))
    rows = [line for line in path.read_text(encoding="utf-8").splitlines()
            if line[:1].isdigit()]
    assert [row.split(",")[0] for row in rows] == ["1", "4"]
    assert not any("nan" in row.lower() for row in rows)


def test_natural_sort_survives_other_digit_characters():
    names = ["shot1²2_010.png", "shot1²2_002.png", "f10.tif", "f2.tif"]
    assert sorted(names, key=natural_sort_key) == [
        "f2.tif", "f10.tif", "shot1²2_002.png", "shot1²2_010.png"]


def test_the_report_accepts_any_folder_name(tmp_path):
    state = update_state(AppState(), folder_path="D:/数据/实验_气泡 — Ölçüm").with_results_initialized(3)
    state.radius[:] = [10.0, 12.0, 11.0]
    path = tmp_path / "report.pdf"
    generate_report(str(path), state)
    assert path.stat().st_size > 1000
    assert pdf_text("实验 — µm") == "?? - µm"


class TestRmaxGap:
    def test_failed_frames_inside_the_fit_window_are_reported(self):
        from bubbletrack.model.rmax import locate_rmax

        t = np.arange(1, 61, dtype=float)
        radius = 100.0 - 0.05 * (t - 30.4) ** 2
        clean = locate_rmax(radius, 11)
        assert not any("bridges the gap" in w for w in clean.warnings)
        radius[[28, 29, 30]] = -1.0  # frames 29-31 unmeasured, next to the peak
        gap = locate_rmax(radius, 11)
        assert any("3 frames between frames" in w and "bridges the gap" in w
                   for w in gap.warnings)
