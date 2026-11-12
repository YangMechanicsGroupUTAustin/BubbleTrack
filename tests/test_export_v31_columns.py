"""v3.1 per-frame fit results in the pixel exports (MAT, CSV, XLSX).

Source, confidence, lock, flags and every shape metric are exported next to
the radius.  The new columns are appended after the existing ones and only
appear when the caller passes them, so old callers get identical files.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy.io import loadmat

from bubbletrack.model.export import (
    export_csv,
    export_excel,
    export_r_data,
    fit_result_arrays,
)
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import FIT_METRIC_KEYS, AppState

OLD_HEADER = [
    "Frame", "Time_s", "Radius_px", "Center_Row_px", "Center_Col_px",
    "Radius_um", "Center_Row_um", "Center_Col_um", "Region_Count",
    "Time_from_Rmax_s",
]
FIT_HEADER = [
    "Source", "Confidence", "Locked", "Flags",
    "R_Circle_px", "Area_px2", "Circularity", "Solidity",
    "Fragment_Count", "Fragment_Area_px2", "Shoulder_Width_px",
]

N = 4
RADIUS = np.array([10.0, -1.0, 20.0, 30.0])
CENTERS = np.array([[50.0, 60.0], [np.nan, np.nan], [51.0, 61.0], [52.0, 62.0]])
REGIONS = np.array([1, 0, 1, 2])


def _fit_kwargs() -> dict:
    metrics = {key: np.full(N, np.nan) for key in FIT_METRIC_KEYS}
    metrics["r_circle"][:] = [10.5, np.nan, 20.25, 30.0]
    metrics["area_px"][:] = [314.0, np.nan, 1256.5, 2827.0]
    metrics["circularity"][:] = [0.95, np.nan, 0.9, 0.85]
    metrics["fragment_count"][:] = [0.0, np.nan, 2.0, 1.0]
    return {
        "fit_source": np.array(
            [FitSource.AUTO, FitSource.NONE, FitSource.MANUAL, FitSource.NEIGHBOR],
            dtype=np.int8,
        ),
        "fit_confidence": np.array([0.91234, np.nan, 0.4, 0.75]),
        "fit_locked": np.array([False, False, True, False]),
        "fit_flags": [frozenset(), frozenset(), frozenset({"fragmented", "clipped"}),
                      frozenset({"low_contrast"})],
        "fit_metrics": metrics,
    }


def _read_csv(path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _cellstr(values) -> list[str]:
    """MATLAB cellstr from loadmat(simplify_cells=True): '' loads as an empty array."""
    return [v if isinstance(v, str) else "" for v in np.atleast_1d(values)]


class TestCsvColumns:
    def test_columns_are_appended_after_the_existing_ones(self, tmp_path):
        out = tmp_path / "r.csv"
        export_csv(str(out), RADIUS, CENTERS, fps=1e6, um2px=3.2,
                   region_count=REGIONS, **_fit_kwargs())
        assert _read_csv(out)[0] == OLD_HEADER + FIT_HEADER

    def test_values(self, tmp_path):
        out = tmp_path / "r.csv"
        export_csv(str(out), RADIUS, CENTERS, **_fit_kwargs())
        header, *rows = _read_csv(out)
        first, third, fourth = (dict(zip(header, row)) for row in rows)
        assert len(rows) == 3  # the unprocessed frame 2 is skipped, as before
        assert first["Source"] == "AUTO"
        assert first["Confidence"] == "0.9123"
        assert first["Locked"] == "0"
        assert first["Flags"] == ""
        assert first["R_Circle_px"] == "10.5000"
        assert first["Fragment_Count"] == "0"
        assert first["Solidity"] == "nan"
        assert third["Source"] == "MANUAL"
        assert third["Locked"] == "1"
        assert third["Flags"] == "clipped|fragmented"  # sorted, pipe-joined
        assert third["Fragment_Count"] == "2"
        assert fourth["Source"] == "NEIGHBOR"

    def test_old_callers_get_the_old_header(self, tmp_path):
        out = tmp_path / "r.csv"
        export_csv(str(out), RADIUS, CENTERS, fps=1e6, um2px=3.2, region_count=REGIONS)
        assert _read_csv(out)[0] == OLD_HEADER

    def test_only_the_given_fields_add_columns(self, tmp_path):
        out = tmp_path / "r.csv"
        kwargs = _fit_kwargs()
        export_csv(str(out), RADIUS, CENTERS, fit_source=kwargs["fit_source"],
                   fit_flags=kwargs["fit_flags"])
        assert _read_csv(out)[0][-2:] == ["Source", "Flags"]

    def test_unknown_metric_keys_are_appended_after_the_known_ones(self, tmp_path):
        out = tmp_path / "r.csv"
        metrics = {"zz_custom": np.ones(N), "area_px": np.ones(N)}
        export_csv(str(out), RADIUS, CENTERS, fit_metrics=metrics)
        assert _read_csv(out)[0][-2:] == ["Area_px2", "zz_custom"]

    def test_length_mismatch_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="fit_confidence"):
            export_csv(str(tmp_path / "r.csv"), RADIUS, CENTERS,
                       fit_confidence=np.ones(N + 1))

    def test_fit_fields_are_keyword_only(self, tmp_path):
        with pytest.raises(TypeError):
            export_csv(str(tmp_path / "r.csv"), RADIUS, CENTERS,  # type: ignore[misc]
                       None, None, None, 11, None, np.zeros(N, dtype=np.int8))


class TestExcelColumns:
    def test_types(self, tmp_path):
        from openpyxl import load_workbook

        out = str(tmp_path / "r.xlsx")
        export_excel(out, RADIUS, CENTERS, fps=1e6, um2px=3.2,
                     region_count=REGIONS, **_fit_kwargs())
        ws = load_workbook(out).active
        header = [cell.value for cell in ws[1]]
        assert header == OLD_HEADER + FIT_HEADER
        rows = [dict(zip(header, (c.value for c in row)))
                for row in ws.iter_rows(min_row=2)]
        first, third = rows[0], rows[1]
        assert first["Source"] == "AUTO"
        assert first["Confidence"] == pytest.approx(0.91234)
        assert first["Locked"] == 0 and isinstance(first["Locked"], int)
        assert first["Solidity"] is None  # NaN stays an empty cell
        assert first["Area_px2"] == pytest.approx(314.0)
        assert third["Flags"] == "clipped|fragmented"
        assert third["Locked"] == 1
        assert third["Fragment_Count"] == pytest.approx(2.0)

    def test_numeric_columns_stay_numeric(self, tmp_path):
        from openpyxl import load_workbook

        out = str(tmp_path / "r.xlsx")
        export_excel(out, RADIUS, CENTERS, **_fit_kwargs())
        ws = load_workbook(out).active
        header = [cell.value for cell in ws[1]]
        text_columns = {"Source", "Flags"}
        for row in ws.iter_rows(min_row=2):
            for name, cell in zip(header, row):
                if name in text_columns or cell.value is None:
                    continue
                assert cell.data_type == "n", (name, cell.value)


class TestMatVariables:
    def test_all_variables_round_trip(self, tmp_path):
        out = str(tmp_path / "r.mat")
        export_r_data(out, RADIUS, CENTERS, [None] * N, REGIONS, **_fit_kwargs())
        data = loadmat(out, simplify_cells=True)

        np.testing.assert_array_equal(data["FitSource"], [1, 0, 2, 4])
        names = _cellstr(data["FitSourceNames"])
        assert names == [s.name for s in FitSource]
        assert [names[code] for code in data["FitSource"]] == [
            "AUTO", "NONE", "MANUAL", "NEIGHBOR"]
        np.testing.assert_allclose(data["FitConfidence"], [0.91234, np.nan, 0.4, 0.75])
        np.testing.assert_array_equal(data["FitLocked"], [0, 0, 1, 0])
        assert _cellstr(data["FitFlags"]) == ["", "", "clipped|fragmented", "low_contrast"]
        metrics = data["FitMetrics"]
        assert set(metrics) == set(FIT_METRIC_KEYS)
        np.testing.assert_allclose(metrics["r_circle"], [10.5, np.nan, 20.25, 30.0])
        # Existing variables are untouched.
        np.testing.assert_allclose(data["Radius"], RADIUS)
        np.testing.assert_array_equal(data["RegionCount"], REGIONS)

    def test_row_vectors_like_radius(self, tmp_path):
        out = str(tmp_path / "r.mat")
        export_r_data(out, RADIUS, CENTERS, [None] * N, **_fit_kwargs())
        data = loadmat(out)
        for key in ("FitSource", "FitConfidence", "FitLocked", "FitFlags"):
            assert data[key].shape == (1, N), key
        assert data["FitLocked"].dtype == np.uint8  # MATLAB logical

    def test_old_callers_get_no_new_variables(self, tmp_path):
        out = str(tmp_path / "r.mat")
        export_r_data(out, RADIUS, CENTERS, [None] * N, REGIONS)
        data = loadmat(out)
        for key in ("FitSource", "FitSourceNames", "FitConfidence", "FitLocked",
                    "FitFlags", "FitMetrics"):
            assert key not in data

    def test_fit_fields_are_keyword_only(self, tmp_path):
        with pytest.raises(TypeError):
            export_r_data(str(tmp_path / "r.mat"), RADIUS, CENTERS,  # type: ignore[misc]
                          [None] * N, REGIONS, None, np.zeros(N, dtype=np.int8))


class TestFitResultArrays:
    def test_state_fields_feed_every_exporter(self, tmp_path):
        state = AppState().with_results_initialized(N)
        np.copyto(state.radius, RADIUS)
        state.circle_fit_par[:] = CENTERS
        state.fit_source[:] = FitSource.AUTO
        state.fit_locked[2] = True
        kwargs = fit_result_arrays(state)
        assert set(kwargs) == {"fit_source", "fit_confidence", "fit_locked",
                               "fit_flags", "fit_metrics"}

        out = tmp_path / "r.csv"
        export_csv(str(out), state.radius, state.circle_fit_par, **kwargs)
        header, *rows = _read_csv(out)
        assert header[-len(FIT_HEADER):] == FIT_HEADER
        assert [dict(zip(header, r))["Locked"] for r in rows] == ["0", "1", "0"]

    def test_empty_state_gives_no_columns(self):
        assert all(v is None for v in fit_result_arrays(AppState()).values())
