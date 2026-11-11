"""Public API contract of ``bubbletrack.model.edge`` (other code is written against it)."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

import bubbletrack.model.edge as edge
from bubbletrack.model.edge import ENGINE_VERSION, EdgeParams, EdgeResult, choose_reference_frame, detect_edge

EDGE_DIR = Path(edge.__file__).parent

RESULT_FIELDS = ["radius", "rc", "cc", "radius_sigma", "edge_xy", "confidence", "flags", "r_area",
                 "outer_radius", "shoulder_width", "arc_coverage", "inlier_fraction", "mask", "provenance"]


def test_exports_exactly_the_contract():
    assert sorted(edge.__all__) == sorted(
        ["EdgeParams", "EdgeResult", "detect_edge", "choose_reference_frame", "ENGINE_VERSION"])
    assert isinstance(ENGINE_VERSION, str) and ENGINE_VERSION


def test_package_has_no_qt_imports():
    for path in EDGE_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "PyQt" not in text and "QtCore" not in text, path.name


def test_package_modules_are_small():
    for path in EDGE_DIR.glob("*.py"):
        assert len(path.read_text(encoding="utf-8").splitlines()) < 400, path.name


def test_edge_params_defaults():
    p = EdgeParams()
    assert (p.edge_definition, p.polarity, p.edge_choice, p.level) == ("steep", "dark", "outer", 0.5)
    assert p.seed is None and p.prior is None and p.confidence_gate == 0.3


def test_edge_params_is_frozen_and_hashable():
    p = EdgeParams(seed=[10, 20.5])
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.level = 0.4  # type: ignore[misc]
    assert p.seed == (10.0, 20.5) and isinstance(p.seed, tuple)
    assert hash(p) == hash(EdgeParams(seed=(10.0, 20.5)))


@pytest.mark.parametrize("kwargs", [
    {"edge_definition": "middle"}, {"polarity": "grey"}, {"edge_choice": "both"},
    {"level": 0.0}, {"level": 1.0}, {"level": "0.5"}, {"confidence_gate": 1.5},
    {"seed": (1.0,)}, {"seed": (1.0, float("nan"))}, {"seed": "ab"},
    {"prior": (1.0, 2.0)}, {"prior": (1.0, 2.0, -3.0)}, {"prior": (1.0, 2.0, float("inf"))},
])
def test_edge_params_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EdgeParams(**kwargs)


def test_edge_result_fields_follow_the_contract():
    names = [f.name for f in dataclasses.fields(EdgeResult)]
    assert names[:len(RESULT_FIELDS)] == RESULT_FIELDS
    assert EdgeResult.__dataclass_params__.frozen


def _result(radius: float, confidence: float, gate: float = 0.3) -> EdgeResult:
    return EdgeResult(radius=radius, rc=1.0, cc=1.0, radius_sigma=0.1, edge_xy=np.empty((0, 2)),
                      confidence=confidence, flags=frozenset(), r_area=float("nan"),
                      outer_radius=float("nan"), shoulder_width=float("nan"), arc_coverage=1.0,
                      inlier_fraction=1.0, mask=np.zeros((2, 2), bool), provenance={}, confidence_gate=gate)


def test_accepted_needs_a_finite_radius_and_the_gate():
    assert _result(10.0, 0.3).accepted
    assert not _result(10.0, 0.29).accepted
    assert not _result(float("nan"), 0.9).accepted
    assert _result(10.0, 0.5, gate=0.5).accepted and not _result(10.0, 0.49, gate=0.5).accepted


def test_detect_edge_signature():
    sig = inspect.signature(detect_edge)
    assert list(sig.parameters)[:3] == ["raw", "gridx", "gridy"]
    for name in ("reference", "params"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["reference"].default is None
    assert sig.parameters["params"].default == EdgeParams()


def test_choose_reference_frame_signature():
    sig = inspect.signature(choose_reference_frame)
    assert list(sig.parameters) == ["read_raw", "n_frames", "search"]
    assert sig.parameters["search"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["search"].default == 10


def test_params_gate_reaches_the_result():
    img = np.full((60, 80), 1000, np.uint16)
    res = detect_edge(img, (1, 60), (1, 80), params=EdgeParams(confidence_gate=0.7))
    assert res.confidence_gate == 0.7
    assert res.provenance["engine"] == ENGINE_VERSION
