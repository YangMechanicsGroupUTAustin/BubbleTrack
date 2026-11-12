"""Batch config dialog and results browser, with hand-built sources and results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from bubbletrack.model.batch_experiments import BatchSource
from bubbletrack.model.batch_result import (
    BatchResultStore,
    FolderResult,
    format_score,
    mean_score,
    restore_into_state,
)
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState
from bubbletrack.ui.batch_config_dialog import BatchConfigDialog, source_label
from bubbletrack.ui.batch_results_dialog import BatchResultsBrowserDialog, error_text


def _result(name="a", *, radius=(10.0, 20.0, -1.0), conf=None, scores=None, **kw):
    radius = np.asarray(radius, dtype=float)
    return FolderResult(
        folder_path=f"/data/{name}", folder_name=name, success=kw.pop("success", True),
        message=kw.pop("message", "ok"), n_frames=len(radius),
        n_fitted=int((radius > 0).sum()), radius=radius,
        quality_scores=np.asarray(scores if scores is not None else [0.9] * len(radius)),
        confidence=None if conf is None else np.asarray(conf, dtype=float), **kw)


def _store(*results, error=""):
    return BatchResultStore(results=results, timestamp="t", gridx=(1, 250),
                            gridy=(1, 400), sensitivity=0.5, error=error)


# -- scores -----------------------------------------------------------------------------

def test_edge_results_are_scored_by_confidence():
    r = _result(conf=[0.8, 0.9, 0.1], scores=[0.0, 0.0, 0.0])
    assert mean_score(r) == pytest.approx(0.85)
    assert format_score(r) == "Avg conf 0.85"


def test_area_mode_results_are_not_scored():
    r = _result(scores=[np.nan, np.nan, 0.0])
    assert np.isnan(mean_score(r))
    assert format_score(r) == "Avg Q n/a"


def test_legacy_quality_score():
    assert format_score(_result(scores=[0.5, 0.7, 0.0])) == "Avg Q 60%"


def test_nothing_fitted_has_no_score():
    assert np.isnan(mean_score(_result(radius=[-1.0, -1.0])))


# -- restoring into the main view ---------------------------------------------------------

def test_restore_into_state_copies_edge_meta():
    state = AppState.create_empty().with_results_initialized(3)
    metrics = {"r_circle": np.array([1.0, 2.0, np.nan])}
    r = _result(conf=[0.8, 0.9, 0.1], flags=[frozenset({"x"}), frozenset(), frozenset()],
                metrics=metrics, circle_fit_par=np.ones((3, 2)),
                circle_xy=[np.zeros((3, 2))] * 3, region_count=np.zeros(3, dtype=int))
    assert restore_into_state(state, r) == 3
    np.testing.assert_allclose(state.radius, [10.0, 20.0, -1.0])
    np.testing.assert_allclose(state.fit_confidence, [0.8, 0.9, 0.1])
    assert state.fit_flags[0] == frozenset({"x"})
    np.testing.assert_allclose(state.fit_metrics["r_circle"][:2], [1.0, 2.0])
    assert list(state.fit_source) == [FitSource.AUTO, FitSource.AUTO, FitSource.NONE]


def test_restore_into_shorter_state_truncates():
    state = AppState.create_empty().with_results_initialized(2)
    assert restore_into_state(state, _result()) == 2


# -- browser ----------------------------------------------------------------------------

def test_browser_shows_failures_and_blocks_loading_them(qapp):
    failed = FolderResult.failure("/data/bad.avi", "bad.avi", "Cannot open: corrupt",
                                  source_kind="video")
    store = _store(_result("good"), failed)
    dlg = BatchResultsBrowserDialog(store)
    try:
        assert "1 source(s) failed" in dlg._error_label.text()
        assert "bad.avi: Cannot open: corrupt" in dlg._error_label.text()
        assert "Failed: Cannot open" in dlg._folder_list.item(1).text()
        assert "video" in dlg._folder_list.item(1).text()
        dlg._folder_list.setCurrentRow(1)
        assert not dlg._load_btn.isEnabled()
        dlg._folder_list.setCurrentRow(0)
        assert dlg._load_btn.isEnabled()
    finally:
        dlg.close()


def test_browser_edge_item_text(qapp):
    dlg = BatchResultsBrowserDialog(_store(_result("e", conf=[0.8, 0.9, 0.1])))
    try:
        text = dlg._folder_list.item(0).text()
        assert "Avg conf 0.85" in text and "nan" not in text
        assert dlg._error_label.isHidden()
    finally:
        dlg.close()


def test_error_text_truncates_long_summaries():
    failures = [FolderResult.failure(f"/d/{i}", f"s{i}", "boom") for i in range(12)]
    text = error_text(_store(*failures, error="disk full"))
    assert text.startswith("12 source(s) failed")
    assert "Batch stopped: disk full" in text
    assert "... and 5 more" in text


def test_error_text_empty_when_all_good():
    assert error_text(_store(_result())) == ""


# -- config dialog ------------------------------------------------------------------------

def _sources(tmp_path):
    return [BatchSource(tmp_path / "exp_a", "folder"),
            BatchSource(tmp_path / "s" / "shot.avi", "video"),
            BatchSource(tmp_path / "broken.avi", "video")]


def test_config_lists_type_and_frame_count(qapp, tmp_path):
    counts = {"exp_a": 120, "shot.avi": 256, "broken.avi": None}
    dlg = BatchConfigDialog(_sources(tmp_path), frame_counter=lambda s: counts[s.name])
    texts = [dlg._source_list.item(i).text() for i in range(3)]
    assert texts == [
        source_label(BatchSource(Path("exp_a")), 120),
        "shot.avi  —  video · 256 frames",
        "broken.avi  —  video · unreadable",
    ]
    assert "2 video(s)" in dlg._summary.text()
    # Unreadable sources start unticked
    cfg = dlg.get_config()
    assert [s.name for s in cfg.sources] == ["exp_a", "shot.avi"]
    assert cfg.folders == [tmp_path / "exp_a", tmp_path / "s" / "shot.avi"]


def test_config_start_needs_a_ticked_source(qapp, tmp_path):
    dlg = BatchConfigDialog(_sources(tmp_path)[:1], frame_counter=lambda s: 5)
    assert dlg._start_btn.isEnabled()
    dlg._source_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not dlg._start_btn.isEnabled()
    assert dlg.get_config().sources == []


def test_config_explains_edge_mode(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel

    dlg = BatchConfigDialog(_sources(tmp_path)[:1], detection_mode="edge",
                            frame_counter=lambda s: 5)
    labels = " ".join(w.text() for w in dlg.findChildren(QLabel))
    assert "reference frame is chosen automatically" in labels
