"""Tests for data export functions."""

import os

import numpy as np
import pytest
from scipy.io import loadmat

from bubbletrack.model.export import export_r_data, export_rof_t_data, safe_loadmat


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
