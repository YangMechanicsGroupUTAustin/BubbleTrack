"""Tests for data export functions."""

import os

import numpy as np
import pytest
from scipy.io import loadmat

from bubbletrack.model.export import (
    build_export_metadata,
    export_r_data,
    export_rof_t_data,
    locate_rmax,
    safe_loadmat,
)


class TestExportRData:
    def test_creates_mat_file(self, tmp_path):
        radius = np.array([10.0, 20.0, 30.0])
        centers = np.array([[50, 60], [51, 61], [52, 62]], dtype=float)
        edges = [np.array([[1, 2], [3, 4]]), None, np.array([[5, 6]])]
        out = str(tmp_path / "radius_pixel.mat")
        export_r_data(out, radius, centers, edges)
        assert os.path.exists(out)

    def test_round_trip(self, tmp_path):
        radius = np.array([10.0, 20.0, 30.0])
        centers = np.array([[50, 60], [51, 61], [52, 62]], dtype=float)
        edges = [None, None, None]
        out = str(tmp_path / "radius_pixel.mat")
        export_r_data(out, radius, centers, edges)
        data = loadmat(out)
        np.testing.assert_array_almost_equal(data["Radius"].ravel(), radius)
        np.testing.assert_array_almost_equal(data["CircleFitPar"], centers)


class TestExportRofTData:
    def _make_parabola(self, n=50, peak=25):
        """Synthetic radius array with a clear parabolic peak."""
        t = np.arange(1, n + 1, dtype=float)
        return 100.0 - 0.5 * (t - peak) ** 2

    def test_basic_export(self, tmp_path):
        R = self._make_parabola()
        out = str(tmp_path / "radius_time_physical.mat")
        ok, msg = export_rof_t_data(out, R, 3.2, 1e6, 11)
        assert ok
        assert msg == ""
        assert os.path.exists(out)

    def test_rmax_at_t_zero(self, tmp_path):
        R = self._make_parabola()
        out = str(tmp_path / "radius_time_physical.mat")
        export_rof_t_data(out, R, 3.2, 1e6, 11)
        data = loadmat(out)
        t = data["t"].ravel()
        # t=0 should be at the peak
        assert 0.0 in t

    # Since b9a054c a peak at the sequence boundary no longer aborts the
    # export: the fit window is clamped and the discrete maximum is used.
    def test_left_boundary_falls_back_to_discrete_max(self, tmp_path):
        R = np.array([100.0, 90.0, 80.0])  # peak at index 0
        out = str(tmp_path / "x.mat")
        ok, _ = export_rof_t_data(out, R, 1.0, 1.0, 5)
        assert ok
        data = loadmat(out)
        assert data["RmaxAll"].item() == pytest.approx(100.0)
        np.testing.assert_allclose(data["t"].ravel(), [0.0, 1.0, 2.0])

    def test_right_boundary_falls_back_to_discrete_max(self, tmp_path):
        R = np.array([80.0, 90.0, 100.0])  # peak at last index
        out = str(tmp_path / "x.mat")
        ok, _ = export_rof_t_data(out, R, 1.0, 1.0, 5)
        assert ok
        data = loadmat(out)
        assert data["RmaxAll"].item() == pytest.approx(100.0)
        np.testing.assert_allclose(data["t"].ravel(), [-2.0, -1.0, 0.0])

    def test_physical_units(self, tmp_path):
        R = self._make_parabola(n=50, peak=25)
        um2px, fps = 2.0, 1000.0
        out = str(tmp_path / "radius_time_physical.mat")
        export_rof_t_data(out, R, um2px, fps, 11)
        data = loadmat(out)
        rmax = float(np.asarray(data["RmaxAll"]).flat[0])
        # Rmax in pixels is 100.0, so in um it should be 200.0
        assert abs(rmax - 200.0) < 1.0

    def test_rmax_fit_with_invalid_gaps(self, tmp_path):
        """Radius array with invalid (-1) entries should still export using valid values."""
        # Peak at index 4 (frame 5) = 30.0, window [4,5,6] all valid
        R = np.array([-1.0, 10.0, -1.0, 20.0, 30.0, 25.0, 15.0])
        out = str(tmp_path / "gaps.mat")
        ok, msg = export_rof_t_data(out, R, 1.0, 1.0, 3)
        assert ok, f"Expected success but got: {msg}"
        data = loadmat(out)
        rmax = float(np.asarray(data["RmaxAll"]).flat[0])
        # Rmax should be near 30.0 (the peak among valid frames)
        assert rmax >= 30.0

    def test_stores_export_parameters(self, tmp_path):
        """fps, um2px and the fit length travel with the data."""
        out = str(tmp_path / "params.mat")
        ok, _ = export_rof_t_data(out, self._make_parabola(), 3.2, 2.5e5, 9)
        assert ok
        data = loadmat(out)
        assert data["fps"].shape == (1, 1)
        assert data["fps"].item() == pytest.approx(2.5e5)
        assert data["um2px"].item() == pytest.approx(3.2)
        assert data["RmaxFitLength"].item() == 9
        assert "BubbleTrackMeta" not in data  # only with metadata

    def test_invalid_frames_exported_as_nan(self, tmp_path):
        """Failed frames (-1, 0, NaN) become NaN in R but keep their t."""
        R = np.array([-1.0, 10.0, np.nan, 20.0, 30.0, 25.0, 15.0, 0.0])
        invalid = [0, 2, 7]
        out = str(tmp_path / "nan.mat")
        ok, _ = export_rof_t_data(out, R, 3.2, 1.0, 3)
        assert ok
        data = loadmat(out)
        t = data["t"].ravel()
        r = data["R"].ravel()
        loc = data["RmaxTimeLoc"].item()  # fps = 1, so this is in frames

        assert len(t) == len(r)
        assert not np.isnan(t).any()
        assert np.isnan(r).sum() == len(invalid)
        assert (r[~np.isnan(r)] > 0).all()
        frame_times = np.arange(1, len(R) + 1) - loc
        for k in invalid:
            idx = int(np.argmin(np.abs(t - frame_times[k])))
            assert t[idx] == pytest.approx(frame_times[k])
            assert np.isnan(r[idx])

    def test_vertex_outside_fit_window_falls_back_to_discrete_max(self, tmp_path):
        """A parabola whose vertex leaves its own window must be rejected."""
        R = np.array([
            32.4, 32.15, 31.92, 31.72, 31.38, 75.42, 73.91,
            72.16, 70.15, 67.92, 65.35, 62.0, 58.0, 54.0,
        ])
        out = str(tmp_path / "vertex.mat")
        ok, _ = export_rof_t_data(out, R, 1.0, 1.0, 11)
        assert ok
        data = loadmat(out)
        # Discrete max is 75.42 at frame 6; the window is frames [1, 11].
        assert data["RmaxAll"].item() == pytest.approx(75.42)
        assert data["RmaxTimeLoc"].item() == pytest.approx(6.0)
        t = data["t"].ravel()
        np.testing.assert_allclose(t, np.arange(1, 15) - 6.0)
        np.testing.assert_allclose(data["R"].ravel(), R)

    def test_rmax_fit_all_invalid(self, tmp_path):
        """All invalid radius values should return failure."""
        R = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        ok, msg = export_rof_t_data(str(tmp_path / "bad.mat"), R, 1.0, 1.0, 3)
        assert not ok
        assert "Not enough valid frames" in msg

    def test_rmax_fit_too_few_valid(self, tmp_path):
        """Only 2 valid frames should return failure (need >= 3)."""
        R = np.array([-1.0, 10.0, -1.0, 20.0, -1.0])
        ok, msg = export_rof_t_data(str(tmp_path / "few.mat"), R, 1.0, 1.0, 3)
        assert not ok
        assert "Not enough valid frames" in msg


class TestLocateRmax:
    """Pure Rmax location helper shared by the MAT and CSV/XLSX exports."""

    def test_parabola_peak_is_fitted(self):
        t = np.arange(1, 51, dtype=float)
        R = 100.0 - 0.5 * (t - 25.3) ** 2
        est = locate_rmax(R, 11)
        assert est.fitted
        assert est.peak_frame == 25
        assert est.window == (20, 30)
        assert est.frame == pytest.approx(25.3)
        assert est.radius == pytest.approx(100.0)

    def test_vertex_outside_window_uses_discrete_max(self):
        R = np.array([
            32.4, 32.15, 31.92, 31.72, 31.38, 75.42, 73.91,
            72.16, 70.15, 67.92, 65.35, 62.0, 58.0, 54.0,
        ])
        est = locate_rmax(R, 11)
        assert not est.fitted
        assert est.window == (1, 11)
        assert est.frame == 6.0
        assert est.radius == pytest.approx(75.42)

    def test_window_is_clamped_at_boundary(self):
        est = locate_rmax(np.array([100.0, 90.0, 80.0]), 5)
        assert est.window == (1, 3)
        assert est.frame == 1.0
        assert est.radius == pytest.approx(100.0)

    def test_invalid_frames_are_ignored(self):
        R = np.array([-1.0, 10.0, np.nan, 20.0, 0.0, 30.0, 25.0])
        est = locate_rmax(R, 1)
        assert est.peak_frame == 6
        assert est.radius == pytest.approx(30.0)

    def test_even_lengths_give_distinct_windows(self):
        t = np.arange(1, 51, dtype=float)
        R = 100.0 - 0.5 * (t - 25) ** 2
        assert locate_rmax(R, 4).window == (24, 26)
        assert locate_rmax(R, 5).window == (23, 27)
        assert locate_rmax(R, 6).window == (23, 27)

    def test_no_valid_frames_raises(self):
        with pytest.raises(ValueError, match="valid"):
            locate_rmax(np.array([-1.0, 0.0, np.nan]), 11)

    def test_does_not_mutate_input(self):
        R = np.array([-1.0, 10.0, 30.0, 20.0, -1.0])
        before = R.copy()
        locate_rmax(R, 3)
        np.testing.assert_array_equal(R, before)


class TestRmaxBoundaryWarning:
    """A peak near the edge of the valid data still exports, with a warning."""

    @staticmethod
    def _smooth_peak(n, peak):
        t = np.arange(1, n + 1, dtype=float)
        return 100.0 - 0.05 * (t - peak) ** 2

    def test_peak_at_first_frame_warns(self, tmp_path):
        ok, msg = export_rof_t_data(
            str(tmp_path / "x.mat"), np.array([100.0, 90.0, 80.0]), 1.0, 1.0, 5,
        )
        assert ok
        assert "boundary" in msg
        assert "start" in msg
        assert "Rmax" in msg and "t=0" in msg and "inaccurate" in msg

    def test_peak_at_last_frame_warns(self, tmp_path):
        ok, msg = export_rof_t_data(
            str(tmp_path / "x.mat"), np.array([80.0, 90.0, 100.0]), 1.0, 1.0, 5,
        )
        assert ok
        assert "boundary" in msg
        assert "end" in msg

    def test_peak_near_first_valid_frame_warns(self, tmp_path):
        """Leading failed frames move the boundary to the first valid frame."""
        R = self._smooth_peak(30, 5.0)
        R[:2] = -1.0  # first valid frame is 3; peak at 5 is 2 frames away
        ok, msg = export_rof_t_data(str(tmp_path / "x.mat"), R, 1.0, 1.0, 11)
        assert ok
        assert "boundary" in msg
        assert "start" in msg
        assert "end" not in msg

    def test_peak_half_window_from_boundary_does_not_warn(self, tmp_path):
        R = self._smooth_peak(30, 6.0)  # frames 1..5 fill the left half-window
        ok, msg = export_rof_t_data(str(tmp_path / "x.mat"), R, 1.0, 1.0, 11)
        assert ok
        assert msg == ""

    def test_both_boundaries_are_reported_together(self):
        est = locate_rmax(np.array([99.0, 100.0, 99.5]), 11)
        assert len(est.warnings) == 1
        assert "boundary (start and end of the valid frames)" in est.warnings[0]

    def test_locate_rmax_reports_boundary_warning(self):
        est = locate_rmax(np.array([100.0, 99.0, 98.0]), 5)
        assert len(est.warnings) == 1
        assert "boundary" in est.warnings[0]
        assert locate_rmax(self._smooth_peak(30, 15.0), 11).warnings == ()


class TestRmaxOutlierWarning:
    """A lone spurious Rmax frame is flagged but still chosen as the peak."""

    @staticmethod
    def _smooth_peak(n=30, peak=15.0):
        t = np.arange(1, n + 1, dtype=float)
        return 100.0 - 0.05 * (t - peak) ** 2

    def test_spike_warns_but_peak_is_unchanged(self, tmp_path):
        R = self._smooth_peak()
        R[21] = 130.0  # frame 22: a single spurious detection
        est = locate_rmax(R, 11)
        assert est.peak_frame == 22  # peak selection is not altered
        assert len(est.warnings) == 1
        assert "frame 22" in est.warnings[0]
        assert "median" in est.warnings[0]

        ok, msg = export_rof_t_data(str(tmp_path / "x.mat"), R, 1.0, 1.0, 11)
        assert ok
        assert "median" in msg

    def test_threshold_is_more_than_ten_percent(self):
        R = np.full(30, 100.0)
        R[14] = 109.0
        assert locate_rmax(R, 11).warnings == ()
        R[14] = 111.0
        assert len(locate_rmax(R, 11).warnings) == 1

    def test_invalid_neighbours_are_skipped(self):
        R = np.full(30, 100.0)
        R[14] = 105.0          # 5% above its valid neighbours: no warning
        R[[13, 15]] = -1.0     # failed frames right next to the peak
        assert locate_rmax(R, 11).warnings == ()

    def test_combined_with_boundary_warning(self, tmp_path):
        R = np.array([130.0, 100.0, 99.0, 98.0, 97.0, 96.0])
        ok, msg = export_rof_t_data(str(tmp_path / "x.mat"), R, 1.0, 1.0, 5)
        assert ok
        assert "boundary" in msg
        assert "median" in msg

    def test_threshold_constant_is_used(self, monkeypatch):
        import bubbletrack.model.export as export_mod

        R = np.full(30, 100.0)
        R[14] = 105.0
        assert locate_rmax(R, 11).warnings == ()
        monkeypatch.setattr(export_mod, "RMAX_OUTLIER_TOLERANCE", 0.01)
        assert len(locate_rmax(R, 11).warnings) == 1


class TestRmaxFitWindow:
    """The fit window uses ``half = (L - 1) // 2`` frames on each side."""

    # Asymmetric peak at frame 6: 3-point and 5-point fits disagree.
    R = np.array([50.0, 70.0, 85.0, 95.0, 99.0, 100.0, 97.0, 90.0, 80.0, 65.0, 45.0])

    def _rmax(self, tmp_path, fit_length):
        out = str(tmp_path / f"L{fit_length}.mat")
        ok, msg = export_rof_t_data(out, self.R, 1.0, 1.0, fit_length)
        assert ok, msg
        data = loadmat(out)
        return data["RmaxAll"].item(), data["RmaxTimeLoc"].item()

    def test_even_length_uses_one_point_fewer(self, tmp_path):
        assert self._rmax(tmp_path, 4) == pytest.approx(self._rmax(tmp_path, 3))
        assert self._rmax(tmp_path, 6) == pytest.approx(self._rmax(tmp_path, 5))

    def test_even_lengths_give_different_windows(self, tmp_path):
        rmax4, loc4 = self._rmax(tmp_path, 4)
        rmax6, loc6 = self._rmax(tmp_path, 6)
        assert rmax4 != pytest.approx(rmax6)
        assert loc4 != pytest.approx(loc6)


META = {
    "detection_mode": "intensity",
    "img_thr": 0.45,
    "removing_factor": np.int64(30),
    "gridx": (10, 200),
    "bubble_cross_edges": (False, True, False, False),
    "folder_path": "C:/data/run1",
    "video_path": None,
    "app_version": "3.0.0",
}


class TestMatMetadata:
    """Optional provenance metadata saved as a BubbleTrackMeta struct."""

    @staticmethod
    def _pixel_export(path, **kwargs):
        radius = np.array([10.0, 20.0, 30.0])
        centers = np.array([[50, 60], [51, 61], [52, 62]], dtype=float)
        export_r_data(path, radius, centers, [None, None, None], **kwargs)

    def test_r_data_saves_metadata_struct(self, tmp_path):
        out = str(tmp_path / "r.mat")
        self._pixel_export(out, metadata=META)
        data = loadmat(out, simplify_cells=True)
        meta = data["BubbleTrackMeta"]
        assert meta["detection_mode"] == "intensity"
        assert meta["img_thr"] == pytest.approx(0.45)
        assert meta["removing_factor"] == 30
        np.testing.assert_array_equal(meta["gridx"], [10, 200])
        np.testing.assert_array_equal(meta["bubble_cross_edges"], [0, 1, 0, 0])
        assert meta["folder_path"] == "C:/data/run1"
        assert meta["app_version"] == "3.0.0"
        # Existing variables are untouched.
        for key in ("Radius", "CircleFitPar", "CircleXY"):
            assert key in data

    def test_r_data_without_metadata_has_no_struct(self, tmp_path):
        out = str(tmp_path / "r.mat")
        self._pixel_export(out)
        assert "BubbleTrackMeta" not in loadmat(out)

    def test_rof_t_saves_metadata_struct(self, tmp_path):
        t = np.arange(1, 31, dtype=float)
        R = 100.0 - 0.05 * (t - 15) ** 2
        out = str(tmp_path / "rt.mat")
        ok, _ = export_rof_t_data(out, R, 3.2, 1e6, 11, metadata=META)
        assert ok
        data = loadmat(out, simplify_cells=True)
        assert data["BubbleTrackMeta"]["detection_mode"] == "intensity"
        assert data["fps"] == pytest.approx(1e6)
        for key in ("t", "R", "RmaxAll", "RmaxTimeLoc"):
            assert key in data

    def test_metadata_is_not_mutated(self, tmp_path):
        meta = dict(META)
        snapshot = dict(meta)
        self._pixel_export(str(tmp_path / "r.mat"), metadata=meta)
        assert meta == snapshot

    def test_metadata_is_keyword_only(self, tmp_path):
        radius = np.array([10.0, 20.0, 30.0])
        centers = np.zeros((3, 2))
        with pytest.raises(TypeError):
            export_r_data(  # type: ignore[misc]
                str(tmp_path / "r.mat"), radius, centers, [None] * 3, None, META,
            )


class TestBuildExportMetadata:
    """Provenance dict assembled from the app state plus panel values."""

    @staticmethod
    def _state():
        from bubbletrack.model.state import AppState

        return AppState(
            folder_path="C:/data/run1",
            detection_mode="gradient",
            img_thr=0.42,
            removing_factor=77,
            gaussian_sigma=1.5,
            clahe_clip=2.0,
            opening_radius=2,
            closing_radius=3,
            bubble_cross_edges=(True, False, False, True),
            gridx=(5, 300),
            gridy=(7, 400),
            fps=1.0,   # stale state defaults: the panel values must win
            um2px=1.0,
        )

    def test_contains_detection_settings_and_panel_values(self):
        from datetime import datetime

        stamp = datetime(2026, 11, 11, 9, 30, 0)
        meta = build_export_metadata(
            self._state(), fps=2e5, um2px=3.2, rmax_fit_length=9,
            timestamp=stamp,
        )
        expected = {
            "detection_mode": "gradient",
            "img_thr": 0.42,
            "removing_factor": 77,
            "gaussian_sigma": 1.5,
            "clahe_clip": 2.0,
            "opening_radius": 2,
            "closing_radius": 3,
            "bubble_cross_edges": [True, False, False, True],
            "gridx": [5, 300],
            "gridy": [7, 400],
            "folder_path": "C:/data/run1",
            "fps": 2e5,
            "um2px": 3.2,
            "rmax_fit_length": 9,
            "exported_at": "2026-11-11T09:30:00",
        }
        for key, value in expected.items():
            assert meta[key] == value, key

    def test_app_version_and_iso_timestamp(self):
        from datetime import datetime

        import bubbletrack

        meta = build_export_metadata(
            self._state(), fps=1e6, um2px=3.2, rmax_fit_length=11,
        )
        assert meta["app_version"] == bubbletrack.__version__
        parsed = datetime.fromisoformat(meta["exported_at"])
        assert parsed.tzinfo is not None  # local time with UTC offset

    def test_version_falls_back_to_package_metadata(self, monkeypatch):
        import importlib.metadata

        import bubbletrack

        monkeypatch.delattr(bubbletrack, "__version__")
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")
        meta = build_export_metadata(
            self._state(), fps=1e6, um2px=3.2, rmax_fit_length=11,
        )
        assert meta["app_version"] == "9.9.9"

    def test_round_trips_through_mat_export(self, tmp_path):
        meta = build_export_metadata(
            self._state(), fps=1e6, um2px=3.2, rmax_fit_length=11,
        )
        out = str(tmp_path / "r.mat")
        export_r_data(out, np.array([1.0]), np.zeros((1, 2)), [None], metadata=meta)
        loaded = loadmat(out, simplify_cells=True)["BubbleTrackMeta"]
        assert loaded["detection_mode"] == "gradient"
        assert loaded["fps"] == pytest.approx(1e6)


class TestSafeLoadmat:
    def test_safe_loadmat_valid_file(self, tmp_path):
        """A valid .mat file with expected keys loads successfully."""
        from scipy.io import savemat as _savemat

        mat_path = str(tmp_path / "valid.mat")
        _savemat(mat_path, {"alpha": np.array([1, 2, 3]), "beta": np.array([4, 5])})

        expected = frozenset({"alpha", "beta"})
        data = safe_loadmat(mat_path, expected)
        assert "alpha" in data
        assert "beta" in data
        np.testing.assert_array_equal(data["alpha"].ravel(), [1, 2, 3])

    def test_safe_loadmat_invalid_file(self, tmp_path):
        """Garbage bytes should raise ValueError."""
        bad_path = str(tmp_path / "garbage.mat")
        with open(bad_path, "wb") as fh:
            fh.write(os.urandom(128))

        with pytest.raises(ValueError, match="Invalid MAT file"):
            safe_loadmat(bad_path, frozenset({"x"}))

    def test_safe_loadmat_missing_keys(self, tmp_path):
        """A .mat file missing expected keys should raise ValueError."""
        from scipy.io import savemat as _savemat

        mat_path = str(tmp_path / "partial.mat")
        _savemat(mat_path, {"alpha": np.array([1])})

        with pytest.raises(ValueError, match="missing keys"):
            safe_loadmat(mat_path, frozenset({"alpha", "beta", "gamma"}))
