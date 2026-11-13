"""The CLI must run the Edge engine for --mode edge, not fall back silently."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import pytest

from bubbletrack import cli
from bubbletrack.model import edge_pipeline


@dataclass(frozen=True)
class FakeParams:
    edge_definition: str = "steep"
    polarity: str = "dark"
    seed: tuple | None = None
    prior: tuple | None = None
    non_round: bool = False


@dataclass
class FakeResult:
    radius: float = 33.0
    rc: float = 125.0
    cc: float = 200.0
    confidence: float = 0.9
    flags: frozenset = frozenset()
    edge_xy: np.ndarray = None
    mask: np.ndarray = None

    @property
    def accepted(self):
        return True


@pytest.fixture
def engine(monkeypatch):
    calls = []

    def fake(raw, gridx, gridy, *, reference=None, params=None):
        calls.append({"reference": reference, "params": params})
        return FakeResult(edge_xy=np.array([[90.0, 200.0], [125.0, 233.0], [158.0, 200.0]]))

    monkeypatch.setattr(edge_pipeline, "_engine", lambda: fake)
    monkeypatch.setattr(edge_pipeline, "_params_class", lambda: FakeParams)
    monkeypatch.setattr(edge_pipeline, "_choose_reference", lambda read, n: 0)
    return calls


def test_cli_edge_mode_uses_engine(make_sequence, tmp_path, engine):
    folder = make_sequence("A", 4)
    out = tmp_path / "out.csv"
    cli.main(["--folder", str(folder), "--mode", "edge", "--edge-definition", "outer",
              "--output", str(out)])
    with out.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 4
    assert float(rows[1][rows[0].index("Radius_px")]) == pytest.approx(33.0)
    assert all(c["params"].edge_definition == "outer" for c in engine)
    assert all(c["reference"] is not None for c in engine)


def test_cli_explicit_reference_is_one_based(make_sequence, tmp_path, engine, monkeypatch):
    chosen = []
    monkeypatch.setattr(edge_pipeline, "resolve_reference_index",
                        lambda read, n, explicit: chosen.append(explicit) or explicit)
    folder = make_sequence("A", 4)
    cli.main(["--folder", str(folder), "--mode", "edge", "--reference", "2",
              "--output", str(tmp_path / "o.csv")])
    assert chosen == [1]


def test_cli_rejects_reference_out_of_range(make_sequence, tmp_path, engine):
    folder = make_sequence("A", 3)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--folder", str(folder), "--mode", "edge", "--reference", "9",
                  "--output", str(tmp_path / "o.csv")])
    assert exc.value.code == 2
