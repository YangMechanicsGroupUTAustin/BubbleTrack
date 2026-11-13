"""v3.1 export metadata (radius definition, Edge settings) and RofT confidence."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.io import loadmat

from bubbletrack.model.export import (
    build_export_metadata,
    export_csv,
    export_rof_t_data,
    radius_definition,
)
from bubbletrack.model.state import AppState

EDGE_STEEP = "area-equivalent sqrt(A/pi) of the 50% iso-contour"  # "Bubble may be non-round"
EDGE_CIRCLE = ("least-squares circle (Taubin, one outlier pass) through the bubble outline "
               "(50 % steep edge)")  # the default (engine edge-1.3.0)


def _meta(**state_fields) -> dict:
    return build_export_metadata(
        AppState(**state_fields), fps=1e6, um2px=3.2, rmax_fit_length=11,
    )


class TestRadiusDefinition:
    @pytest.mark.parametrize("mode, expected", [
        ("intensity", "circle fit (Taubin)"),
        ("gradient", "circle fit (Taubin)"),
        ("area", "area-equivalent (dark regions)"),
        ("edge", EDGE_CIRCLE),
    ])
    def test_per_detection_mode(self, mode, expected):
        assert radius_definition(mode) == expected
        assert _meta(detection_mode=mode)["radius_definition"] == expected

    def test_edge_non_round_is_the_area_of_the_outline(self):
        assert radius_definition("edge", non_round=True) == EDGE_STEEP
        meta = _meta(detection_mode="edge", bubble_non_round=True)
        assert meta["radius_definition"] == EDGE_STEEP and meta["bubble_non_round"] is True
        assert _meta(detection_mode="edge")["bubble_non_round"] is False

    def test_non_round_only_matters_in_edge_mode(self):
        assert radius_definition("intensity", non_round=True) == "circle fit (Taubin)"
        assert radius_definition("area", non_round=True) == "area-equivalent (dark regions)"

    def test_edge_outer_extent(self):
        for non_round in (False, True):
            text = radius_definition("edge", "outer", non_round)
            assert "outer extent" in text
            assert _meta(detection_mode="edge", edge_definition="outer", bubble_non_round=non_round)[
                "radius_definition"] == text

    def test_edge_definition_only_matters_in_edge_mode(self):
        assert radius_definition("intensity", "outer") == "circle fit (Taubin)"


class TestEdgeSettings:
    def test_recorded(self):
        meta = _meta(detection_mode="edge", edge_definition="outer",
                     edge_polarity="bright", reference_frame=4,
                     edge_seed=(12.5, 30.5))
        assert meta["edge_definition"] == "outer"
        assert meta["edge_polarity"] == "bright"
        assert meta["reference_frame"] == 5  # 1-based, like the Frame column
        assert meta["reference_frame_origin"] == "set"
        assert meta["edge_seed_used"] is True
        assert meta["edge_seed"] == [12.5, 30.5]

    def test_auto_reference_and_no_seed(self):
        meta = _meta(detection_mode="edge")
        assert meta["reference_frame"] is None
        assert meta["reference_frame_origin"] == "auto"
        assert meta["edge_seed_used"] is False
        assert meta["edge_seed"] is None

    def test_resolved_auto_reference_is_recorded(self):
        meta = build_export_metadata(
            AppState(detection_mode="edge"), fps=1e6, um2px=3.2,
            rmax_fit_length=11, reference_index=2,
        )
        assert meta["reference_frame"] == 3
        assert meta["reference_frame_origin"] == "auto"

    def test_seed_is_not_used_outside_edge_mode(self):
        meta = _meta(detection_mode="intensity", edge_seed=(1.0, 2.0))
        assert meta["edge_seed_used"] is False

    def test_all_values_survive_every_container(self, tmp_path):
        meta = _meta(detection_mode="edge", reference_frame=0, edge_seed=(3.5, 4.5),
                     bubble_non_round=True)
        mat = str(tmp_path / "rt.mat")
        t = np.arange(1, 31, dtype=float)
        ok, _ = export_rof_t_data(mat, 100.0 - 0.05 * (t - 15) ** 2, 3.2, 1e6, 11,
                                  metadata=meta)
        assert ok
        stored = loadmat(mat, simplify_cells=True)["BubbleTrackMeta"]
        assert stored["radius_definition"] == EDGE_STEEP and stored["bubble_non_round"]
        assert stored["reference_frame"] == 1
        assert stored["edge_seed_used"]

        csv_path = tmp_path / "r.csv"
        export_csv(str(csv_path), np.array([1.0]), np.zeros((1, 2)), metadata=meta)
        sidecar = json.loads((tmp_path / "r.meta.json").read_text(encoding="utf-8"))
        assert sidecar["radius_definition"] == EDGE_STEEP and sidecar["bubble_non_round"] is True
        assert sidecar["edge_seed_used"] is True


class TestRofTConfidence:
    R = 100.0 - 0.05 * (np.arange(1, 31, dtype=float) - 15.3) ** 2

    def test_vector_matches_t_and_r(self, tmp_path):
        radius = self.R.copy()
        radius[3] = -1.0  # a failed frame
        confidence = np.linspace(0.1, 0.9, 30)
        out = str(tmp_path / "rt.mat")
        ok, _ = export_rof_t_data(out, radius, 3.2, 1e6, 11, confidence=confidence)
        assert ok
        data = loadmat(out)
        t, conf = data["t"].ravel(), data["Confidence"].ravel()
        assert data["Confidence"].shape == data["R"].shape == (31, 1)
        # The inserted Rmax sample (t = 0) is not a measured frame.
        assert np.isnan(conf[t == 0.0]).all()
        measured = conf[t != 0.0]
        assert np.isnan(measured[3])  # failed frame: no confidence either
        np.testing.assert_allclose(np.delete(measured, 3), np.delete(confidence, 3))

    def test_absent_by_default(self, tmp_path):
        out = str(tmp_path / "rt.mat")
        ok, _ = export_rof_t_data(out, self.R, 3.2, 1e6, 11)
        assert ok
        assert "Confidence" not in loadmat(out)

    def test_physics_unchanged(self, tmp_path):
        plain, scored = str(tmp_path / "a.mat"), str(tmp_path / "b.mat")
        export_rof_t_data(plain, self.R, 3.2, 1e6, 11)
        export_rof_t_data(scored, self.R, 3.2, 1e6, 11, confidence=np.ones(30))
        a, b = loadmat(plain), loadmat(scored)
        for key in ("t", "R", "RmaxAll", "RmaxTimeLoc"):
            np.testing.assert_array_equal(a[key], b[key])

    def test_length_mismatch_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="confidence"):
            export_rof_t_data(str(tmp_path / "rt.mat"), self.R, 3.2, 1e6, 11,
                              confidence=np.ones(29))

    def test_discrete_peak_keeps_its_frame_confidence(self, tmp_path):
        radius = np.array([100.0, 90.0, 80.0, 70.0, 60.0])  # peak at frame 1: no fit
        confidence = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
        out = str(tmp_path / "rt.mat")
        ok, _ = export_rof_t_data(out, radius, 1.0, 1.0, 5, confidence=confidence)
        assert ok
        data = loadmat(out)
        np.testing.assert_allclose(data["t"].ravel(), [0, 1, 2, 3, 4])
        np.testing.assert_allclose(data["Confidence"].ravel(), confidence)
