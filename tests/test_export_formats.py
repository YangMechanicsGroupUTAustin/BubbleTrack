"""Tests for CSV and Excel export functions."""

import csv
import os

import numpy as np
import pytest

from bubbletrack.model.export import export_csv, export_excel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_data():
    """Return (radius, centers) with some unprocessed frames."""
    radius = np.array([10.0, -1.0, 20.0, 30.0, -1.0])
    centers = np.array([
        [50.0, 60.0],
        [0.0, 0.0],
        [51.0, 61.0],
        [52.0, 62.0],
        [0.0, 0.0],
    ])
    return radius, centers


@pytest.fixture()
def full_data():
    """Return (radius, centers, region_count) exercising every column."""
    radius = np.array([12.34567, -1.0, 20.5])
    centers = np.array([[100.98765, 200.12345], [0.0, 0.0], [51.5, 61.25]])
    region_count = np.array([2, 0, 1])
    return radius, centers, region_count


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------

class TestExportCSV:
    def test_creates_file(self, tmp_path, sample_data):
        radius, centers = sample_data
        out = str(tmp_path / "test.csv")
        export_csv(out, radius, centers)
        assert os.path.exists(out)

    def test_skips_unprocessed_frames(self, tmp_path, sample_data):
        radius, centers = sample_data
        out = str(tmp_path / "test.csv")
        export_csv(out, radius, centers)

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 3 valid frames (indices 0, 2, 3)
        assert len(rows) == 4
        assert rows[0][0] == "Frame"
        # Frame numbers should be 1, 3, 4 (1-indexed)
        assert rows[1][0] == "1"
        assert rows[2][0] == "3"
        assert rows[3][0] == "4"

    def test_basic_columns_no_fps_no_scale(self, tmp_path, sample_data):
        radius, centers = sample_data
        out = str(tmp_path / "test.csv")
        export_csv(out, radius, centers)

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ["Frame", "Radius_px", "Center_Row_px", "Center_Col_px"]

    def test_with_fps(self, tmp_path, sample_data):
        radius, centers = sample_data
        out = str(tmp_path / "test.csv")
        export_csv(out, radius, centers, fps=1e6)

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)

        assert "Time_s" in header
        # Time for frame 0 at 1 MHz should be 0.0
        time_idx = header.index("Time_s")
        assert float(first_row[time_idx]) == pytest.approx(0.0)

    def test_with_scale(self, tmp_path, sample_data):
        radius, centers = sample_data
        out = str(tmp_path / "test.csv")
        export_csv(out, radius, centers, um2px=3.2)

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)

        assert "Radius_um" in header
        assert "Center_Row_um" in header
        assert "Center_Col_um" in header
        # Radius_um for first valid frame: 10.0 * 3.2 = 32.0
        r_um_idx = header.index("Radius_um")
        assert float(first_row[r_um_idx]) == pytest.approx(32.0)

    def test_with_fps_and_scale(self, tmp_path, sample_data):
        radius, centers = sample_data
        out = str(tmp_path / "test.csv")
        export_csv(out, radius, centers, fps=1000.0, um2px=2.0)

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = [
            "Frame", "Time_s",
            "Radius_px", "Center_Row_px", "Center_Col_px",
            "Radius_um", "Center_Row_um", "Center_Col_um",
            "Time_from_Rmax_s",
        ]
        assert header == expected

    def test_all_invalid_produces_header_only(self, tmp_path):
        radius = np.array([-1.0, -1.0, -1.0])
        centers = np.zeros((3, 2))
        out = str(tmp_path / "empty.csv")
        export_csv(out, radius, centers)

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 1  # header only

    def test_radius_precision(self, tmp_path):
        radius = np.array([12.34567])
        centers = np.array([[100.98765, 200.12345]])
        out = str(tmp_path / "precision.csv")
        export_csv(out, radius, centers)

        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)

        # 4 decimal places
        assert row[1] == "12.3457"
        assert row[2] == "100.9877"

    def test_exact_bytes_all_columns(self, tmp_path, full_data):
        """Pin the CSV byte format so refactors cannot change it."""
        radius, centers, region_count = full_data
        out = tmp_path / "pinned.csv"
        export_csv(
            str(out), radius, centers,
            fps=3000.0, um2px=3.2, region_count=region_count,
        )
        # Time_from_Rmax_s is appended last; Rmax is the discrete max at
        # frame 3 (only two valid frames, too few for a parabola).
        expected = (
            b"Frame,Time_s,Radius_px,Center_Row_px,Center_Col_px,"
            b"Radius_um,Center_Row_um,Center_Col_um,Region_Count,"
            b"Time_from_Rmax_s\r\n"
            b"1,0.00000000,12.3457,100.9877,200.1234,"
            b"39.5061,323.1605,640.3950,2,-0.00066667\r\n"
            b"3,0.00066667,20.5000,51.5000,61.2500,"
            b"65.6000,164.8000,196.0000,1,0.00000000\r\n"
        )
        assert out.read_bytes() == expected

    def test_exact_bytes_without_fps(self, tmp_path, full_data):
        """Without fps there is no time column at all."""
        radius, centers, region_count = full_data
        out = tmp_path / "pinned.csv"
        export_csv(str(out), radius, centers, um2px=3.2, region_count=region_count)
        expected = (
            b"Frame,Radius_px,Center_Row_px,Center_Col_px,"
            b"Radius_um,Center_Row_um,Center_Col_um,Region_Count\r\n"
            b"1,12.3457,100.9877,200.1234,39.5061,323.1605,640.3950,2\r\n"
            b"3,20.5000,51.5000,61.2500,65.6000,164.8000,196.0000,1\r\n"
        )
        assert out.read_bytes() == expected


class TestTimeFromRmax:
    """Time_from_Rmax_s uses the same Rmax location as the RofT .mat export."""

    @staticmethod
    def _peak_data():
        t = np.arange(1, 41, dtype=float)
        radius = 100.0 - 0.07 * (t - 20.4) ** 2
        radius[[3, 30]] = -1.0  # failed frames are skipped as rows
        centers = np.full((40, 2), 50.0)
        return radius, centers

    def _read_csv(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return rows[0], rows[1:]

    def test_column_appended_last_when_fps_given(self, tmp_path):
        radius, centers = self._peak_data()
        out = str(tmp_path / "t.csv")
        export_csv(out, radius, centers, fps=1e6, um2px=2.0)
        header, _ = self._read_csv(out)
        assert header == [
            "Frame", "Time_s",
            "Radius_px", "Center_Row_px", "Center_Col_px",
            "Radius_um", "Center_Row_um", "Center_Col_um",
            "Time_from_Rmax_s",
        ]

    def test_no_column_without_fps(self, tmp_path):
        radius, centers = self._peak_data()
        out = str(tmp_path / "t.csv")
        export_csv(out, radius, centers, um2px=2.0)
        header, _ = self._read_csv(out)
        assert "Time_from_Rmax_s" not in header

    def test_matches_rof_t_mat_time(self, tmp_path):
        from scipy.io import loadmat

        from bubbletrack.model.export import export_rof_t_data

        radius, centers = self._peak_data()
        fps, fit_length = 2e5, 9
        csv_path = str(tmp_path / "t.csv")
        mat_path = str(tmp_path / "t.mat")
        export_csv(csv_path, radius, centers, fps=fps, rmax_fit_length=fit_length)
        ok, _ = export_rof_t_data(mat_path, radius, 1.0, fps, fit_length)
        assert ok

        header, rows = self._read_csv(csv_path)
        col = header.index("Time_from_Rmax_s")
        csv_t = np.array([float(r[col]) for r in rows])
        frames = np.array([int(r[0]) for r in rows])

        mat = loadmat(mat_path)
        rmax_frame = mat["RmaxTimeLoc"].item() * fps  # 1-based frame position
        assert rmax_frame == pytest.approx(20.4, abs=1e-6)
        np.testing.assert_allclose(csv_t, (frames - rmax_frame) / fps, atol=1e-8)
        # Every CSV time also appears in the MAT t vector.
        mat_t = mat["t"].ravel()
        for value in csv_t:
            assert np.min(np.abs(mat_t - value)) < 1e-8

    def test_fit_length_changes_rmax_location(self, tmp_path):
        radius = np.array(
            [50.0, 70.0, 85.0, 95.0, 99.0, 100.0, 97.0, 90.0, 80.0, 65.0, 45.0],
        )
        centers = np.zeros((11, 2))
        a, b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
        export_csv(a, radius, centers, fps=1.0, rmax_fit_length=3)
        export_csv(b, radius, centers, fps=1.0, rmax_fit_length=5)
        assert self._read_csv(a)[1] != self._read_csv(b)[1]

    def test_returns_rmax_warnings(self, tmp_path, full_data):
        radius, centers, _ = full_data
        msg = export_csv(str(tmp_path / "w.csv"), radius, centers, fps=1.0)
        assert "boundary" in msg
        assert export_csv(str(tmp_path / "n.csv"), radius, centers) == ""

    def test_excel_column_is_numeric(self, tmp_path):
        from openpyxl import load_workbook

        radius, centers = self._peak_data()
        out = str(tmp_path / "t.xlsx")
        msg = export_excel(out, radius, centers, fps=1e6)
        assert msg == ""
        ws = load_workbook(out).active
        header = [cell.value for cell in ws[1]]
        assert header[-1] == "Time_from_Rmax_s"
        col = header.index("Time_from_Rmax_s") + 1
        cells = [row[col - 1] for row in ws.iter_rows(min_row=2)]
        assert all(cell.data_type == "n" for cell in cells)
        frames = [row[0].value for row in ws.iter_rows(min_row=2)]
        expected = (np.array(frames) - 20.4) / 1e6
        np.testing.assert_allclose([c.value for c in cells], expected, atol=1e-12)

    def test_all_invalid_with_fps_is_header_only(self, tmp_path):
        out = str(tmp_path / "e.csv")
        msg = export_csv(out, np.array([-1.0, -1.0]), np.zeros((2, 2)), fps=1e6)
        assert msg == ""
        header, rows = self._read_csv(out)
        assert header[-1] == "Time_from_Rmax_s"
        assert rows == []


# ---------------------------------------------------------------------------
# Excel tests
# ---------------------------------------------------------------------------

class TestExportExcel:
    def test_creates_file(self, tmp_path, sample_data):
        radius, centers = sample_data
        out = str(tmp_path / "test.xlsx")
        export_excel(out, radius, centers)
        assert os.path.exists(out)

    def test_skips_unprocessed_frames(self, tmp_path, sample_data):
        from openpyxl import load_workbook

        radius, centers = sample_data
        out = str(tmp_path / "test.xlsx")
        export_excel(out, radius, centers)

        wb = load_workbook(out)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))

        # Header + 3 valid frames
        assert len(rows) == 4
        assert rows[0][0] == "Frame"
        assert rows[1][0] == 1
        assert rows[2][0] == 3
        assert rows[3][0] == 4

    def test_data_cells_are_numeric(self, tmp_path, full_data):
        """Data cells must be real numbers, not text, so Excel can plot them."""
        from openpyxl import load_workbook

        radius, centers, region_count = full_data
        out = str(tmp_path / "numeric.xlsx")
        export_excel(
            out, radius, centers,
            fps=3000.0, um2px=3.2, region_count=region_count,
        )

        ws = load_workbook(out).active
        header = [cell.value for cell in ws[1]]
        data_rows = list(ws.iter_rows(min_row=2))
        assert len(data_rows) == 2
        for row in data_rows:
            for cell in row:
                assert cell.data_type == "n", (header[cell.column - 1], cell.value)

        first = dict(zip(header, (cell.value for cell in data_rows[0])))
        assert first["Frame"] == 1
        assert isinstance(first["Frame"], int)
        assert first["Time_s"] == pytest.approx(0.0)
        assert first["Radius_px"] == pytest.approx(12.34567)
        assert first["Center_Row_px"] == pytest.approx(100.98765)
        assert first["Center_Col_px"] == pytest.approx(200.12345)
        assert first["Radius_um"] == pytest.approx(12.34567 * 3.2)
        assert first["Center_Row_um"] == pytest.approx(100.98765 * 3.2)
        assert first["Center_Col_um"] == pytest.approx(200.12345 * 3.2)
        assert first["Region_Count"] == 2
        assert isinstance(first["Region_Count"], int)

        second = dict(zip(header, (cell.value for cell in data_rows[1])))
        assert second["Frame"] == 3
        assert second["Time_s"] == pytest.approx(2 / 3000.0)

    def test_non_finite_values_become_empty_cells(self, tmp_path):
        """Excel has no NaN: such cells are left empty, not written as NaN."""
        import zipfile

        from openpyxl import load_workbook

        radius = np.array([10.0, 20.0])
        centers = np.array([[np.nan, 60.0], [51.0, 61.0]])
        out = str(tmp_path / "nan.xlsx")
        export_excel(out, radius, centers)

        sheet_xml = zipfile.ZipFile(out).read("xl/worksheets/sheet1.xml")
        assert b"<v></v>" not in sheet_xml
        assert b"nan" not in sheet_xml.lower()
        ws = load_workbook(out).active
        assert ws.cell(row=2, column=3).value is None  # Center_Row_px
        assert ws.cell(row=2, column=4).value == pytest.approx(60.0)

    def test_sheet_title(self, tmp_path, sample_data):
        from openpyxl import load_workbook

        radius, centers = sample_data
        out = str(tmp_path / "test.xlsx")
        export_excel(out, radius, centers)

        wb = load_workbook(out)
        assert wb.active.title == "Bubble Radius Data"

    def test_with_fps_and_scale(self, tmp_path, sample_data):
        from openpyxl import load_workbook

        radius, centers = sample_data
        out = str(tmp_path / "test.xlsx")
        export_excel(out, radius, centers, fps=1000.0, um2px=2.0)

        wb = load_workbook(out)
        ws = wb.active
        header = [cell.value for cell in ws[1]]

        expected = [
            "Frame", "Time_s",
            "Radius_px", "Center_Row_px", "Center_Col_px",
            "Radius_um", "Center_Row_um", "Center_Col_um",
            "Time_from_Rmax_s",
        ]
        assert header == expected

    def test_all_invalid_produces_header_only(self, tmp_path):
        from openpyxl import load_workbook

        radius = np.array([-1.0, -1.0, -1.0])
        centers = np.zeros((3, 2))
        out = str(tmp_path / "empty.xlsx")
        export_excel(out, radius, centers)

        wb = load_workbook(out)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

META = {
    "detection_mode": "gradient",
    "img_thr": 0.45,
    "removing_factor": np.int64(30),
    "gridx": (10, 200),
    "bubble_cross_edges": (False, True, False, False),
    "folder_path": "C:/data/run1",
    "video_path": None,
}


class TestCsvMetadata:
    def test_writes_json_sidecar(self, tmp_path, sample_data):
        import json

        radius, centers = sample_data
        out = tmp_path / "run.csv"
        export_csv(str(out), radius, centers, fps=1e6, metadata=META)

        sidecar = tmp_path / "run.meta.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta == {
            "detection_mode": "gradient",
            "img_thr": 0.45,
            "removing_factor": 30,
            "gridx": [10, 200],
            "bubble_cross_edges": [False, True, False, False],
            "folder_path": "C:/data/run1",
            "video_path": None,
        }

    def test_csv_itself_is_unchanged(self, tmp_path, sample_data):
        radius, centers = sample_data
        plain, meta = tmp_path / "plain.csv", tmp_path / "meta.csv"
        export_csv(str(plain), radius, centers, fps=1e6)
        export_csv(str(meta), radius, centers, fps=1e6, metadata=META)
        assert meta.read_bytes() == plain.read_bytes()

    def test_no_sidecar_without_metadata(self, tmp_path, sample_data):
        radius, centers = sample_data
        export_csv(str(tmp_path / "run.csv"), radius, centers)
        assert list(tmp_path.iterdir()) == [tmp_path / "run.csv"]


class TestExcelMetadata:
    def test_writes_metadata_sheet(self, tmp_path, sample_data):
        import json

        from openpyxl import load_workbook

        radius, centers = sample_data
        out = str(tmp_path / "run.xlsx")
        export_excel(out, radius, centers, fps=1e6, metadata=META)

        wb = load_workbook(out)
        assert wb.sheetnames == ["Bubble Radius Data", "Metadata"]
        assert wb.active.title == "Bubble Radius Data"
        rows = list(wb["Metadata"].iter_rows(values_only=True))
        assert rows[0] == ("Key", "Value")
        values = dict(rows[1:])
        assert values["detection_mode"] == "gradient"
        assert values["img_thr"] == pytest.approx(0.45)
        assert values["removing_factor"] == 30
        assert json.loads(values["gridx"]) == [10, 200]
        assert json.loads(values["bubble_cross_edges"]) == [False, True, False, False]
        assert values["video_path"] is None

    def test_no_metadata_sheet_by_default(self, tmp_path, sample_data):
        from openpyxl import load_workbook

        radius, centers = sample_data
        out = str(tmp_path / "run.xlsx")
        export_excel(out, radius, centers)
        assert load_workbook(out).sheetnames == ["Bubble Radius Data"]
