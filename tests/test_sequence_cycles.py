"""Collapse / cycle detection on radius time series (``model.sequence.detect_cycles``)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bubbletrack.model.sequence import SequenceConfig, detect_cycles

NAN = float("nan")


def rayleigh(n: int, cycles: list[tuple[int, int, float]], floor: float = 1.5) -> np.ndarray:
    """Radius series from (start, period, Rmax) cycles: R = Rmax (1 - u^2)^0.4, u in [-1, 1]."""
    t = np.arange(n, dtype=float)
    r = np.full(n, floor)
    for start, period, rmax in cycles:
        u = (t - start - period / 2) / (period / 2)
        m = np.abs(u) < 1
        r[m] = np.maximum(rmax * (1 - u[m] ** 2) ** 0.4, floor)
    return r


def test_single_growth_without_collapse_is_one_cycle():
    r = np.linspace(10, 80, 30)
    assert detect_cycles(r) == ((), ((0, 29),))


def test_two_cycles_split_at_the_collapse():
    r = rayleigh(60, [(0, 30, 80.0), (30, 20, 40.0)])[1:]          # starts growing at frame 0
    collapses, cycles = detect_cycles(r)
    assert collapses == (29,)
    assert cycles == ((0, 29), (30, len(r) - 1))


def test_a_sharp_single_frame_collapse_is_found():
    """Rayleigh collapses are often one frame deep (24 -> 1.8 -> 14): no smoothing may erase them."""
    r = np.array([10, 30, 40, 45, 44, 38, 24, 1.8, 14.3, 18, 20, 19, 16, 12], float)
    collapses, cycles = detect_cycles(r)
    assert collapses == (7,)
    assert cycles[0] == (0, 7)


def test_a_one_frame_dip_mid_cycle_is_not_a_collapse():
    """A wrong detection (inner ring) mid-cycle: flat neighbours, one low frame."""
    r = np.array([60, 70, 76, 77, 77, 45, 77, 76, 74, 70], float)
    assert detect_cycles(r)[0] == ()


def test_small_late_cycles_count_relative_to_their_own_peaks():
    """lic 10 like: Rmax 116, then rebounds 51, 34, 20 -- every collapse is found."""
    r = np.concatenate([rayleigh(66, [(0, 66, 116.0)], 17)[1:],
                        rayleigh(28, [(0, 28, 51.0)], 15)[1:],
                        rayleigh(18, [(0, 18, 34.0)], 11)[1:],
                        rayleigh(14, [(0, 14, 20.0)], 13)[1:]])
    collapses, cycles = detect_cycles(r)
    assert len(collapses) == 3
    assert len(cycles) == 4


def test_noise_on_a_residual_bubble_is_not_a_collapse():
    rng = np.random.default_rng(0)
    r = 10 + rng.normal(0, 0.4, 120)
    assert detect_cycles(r)[0] == ()


def test_interior_gaps_are_interpolated_and_the_ends_are_outside():
    r = rayleigh(50, [(5, 30, 60.0), (35, 14, 25.0)])
    r[:5] = NAN                      # before nucleation
    r[47:] = -1.0                    # gone
    r[[12, 13, 40]] = NAN            # lost frames inside the cycles
    collapses, cycles = detect_cycles(r)
    assert cycles[0][0] == 5 and cycles[-1][1] == 46
    assert collapses[0] == 35


def test_the_last_collapse_before_the_bubble_vanishes_is_found():
    r = np.concatenate([rayleigh(30, [(0, 30, 60.0)], 2.0)[1:30], [NAN] * 5])
    collapses, cycles = detect_cycles(r)
    assert collapses == (28,)
    assert cycles == ((0, 28),)


def test_a_decreasing_radius_at_the_recording_end_is_not_a_collapse():
    r = rayleigh(30, [(0, 40, 60.0)])[1:30]
    assert detect_cycles(r)[0] == ()


def test_close_minima_keep_the_deeper_one():
    r = np.array([50, 60, 40, 20, 5, 22, 8, 30, 45, 50, 48], float)
    collapses, _ = detect_cycles(r, config=SequenceConfig(collapse_min_gap=3))
    assert collapses == (4,)


def test_degenerate_inputs():
    assert detect_cycles(np.array([])) == ((), ())
    assert detect_cycles(np.full(5, NAN)) == ((), ())
    assert detect_cycles(np.array([NAN, 3.0, 4.0, NAN])) == ((), ((1, 2),))


def test_input_is_not_modified():
    r = rayleigh(40, [(0, 20, 50.0), (20, 12, 20.0)])
    r[5] = NAN
    before = r.copy()
    detect_cycles(r)
    np.testing.assert_array_equal(r, before)


# --------------------------------------------------------------------------- #
#  Real R(t) series (local data only)
# --------------------------------------------------------------------------- #
def _data_root() -> Path | None:
    env = os.environ.get("BUBBLETRACK_TEST_DATA")
    candidates = [Path(env)] if env else []
    candidates += [p / "test_data" for p in Path(__file__).resolve().parents]
    return next((c for c in candidates if (c / "lic").is_dir()), None)


def _mat_radius(path: Path, key: str) -> np.ndarray:
    sio = pytest.importorskip("scipy.io")
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    r = sio.loadmat(path)[key].ravel().astype(float)
    return np.where(np.isfinite(r) & (r > 0), r, NAN)


ROOT = _data_root()
real = pytest.mark.skipif(ROOT is None, reason="local LIC test data not present")


@real
def test_lic10_collapses_from_the_2026_app_result():
    """The early collapses (frames 66, 93, 110, 1-based 67, 94, 111) must all be found."""
    r = _mat_radius(ROOT / "lic" / "refs" / "10_gelatin_Rt_132046.mat", "Radius")
    collapses, _ = detect_cycles(r)
    assert {66, 93, 110} <= set(collapses)
    assert all(c > 60 for c in collapses)


@real
def test_lic04_collapses_from_the_2024_matlab_result():
    folder = next(iter(sorted((ROOT / "lic").glob("04_*"))), None)
    if folder is None:
        pytest.skip("lic 04 not present")
    collapses, _ = detect_cycles(_mat_radius(folder / "R_data.mat", "CircleR"))
    assert collapses[:3] == (16, 32, 45)
