"""BatchFolderWorker: folders and videos, legacy and Edge modes, failures.

The worker is run synchronously (``run()`` on the test thread), so its
signals are delivered directly.  ``inline`` swaps the process pool for
threads except where the real worker processes are the point of the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import loadmat

from bubbletrack.controller import batch_folder_worker as bfw
from bubbletrack.controller.batch_folder_worker import BatchFolderWorker
from bubbletrack.model import edge_pipeline
from bubbletrack.model.batch_experiments import BatchSource
from bubbletrack.model.state import FIT_METRIC_KEYS
from conftest import write_sequence
from test_video_fitting import write_video

TS = "20260101_120000"
EDGE_RADII = [0.0, 0.0, 25.0, 35.0, 45.0, 55.0, 50.0, 40.0]  # 0 = no bubble


class Run:
    """Runs a worker synchronously and records every signal."""

    def __init__(self, sources, *, mode="intensity", inline=True, **kw):
        params = dict(
            sensitivity=0.58, gridx=(1, 250), gridy=(1, 400),
            removing_factor_slider=85, bubble_cross_edges=(False,) * 4,
            gaussian_sigma=3.0, detection_mode=mode, max_radius=400.0,
        )
        if mode == "edge":
            params["edge_params"] = edge_pipeline.make_params()
        params.update(kw)
        self.worker = BatchFolderWorker(sources, timestamp=TS, **params)
        self.worker.inline = inline
        self.results, self.done, self.errors, self.all_done = [], [], [], []
        self.worker.result_ready.connect(self.results.append)
        self.worker.folder_done.connect(lambda *a: self.done.append(a))
        self.worker.error.connect(self.errors.append)
        self.worker.all_done.connect(lambda *a: self.all_done.append(a))

    def __call__(self):
        self.worker.run()
        return self


def _folder(tmp_path, name, radii):
    return BatchSource(write_sequence(tmp_path / name, radii), "folder")


def _video(tmp_path, name, radii):
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return BatchSource(write_video(tmp_path / name / f"{name}.avi", radii), "video")


# -- legacy pipeline over folders and videos ------------------------------------------

def test_mixed_folders_and_videos(qapp, tmp_path):
    sources = [
        _folder(tmp_path, "exp_a", np.linspace(25, 60, 4)),
        _video(tmp_path, "shot_1", np.linspace(25, 70, 6)),
        _folder(tmp_path, "exp_b", np.linspace(30, 50, 3)),
    ]
    run = Run(sources)()

    assert [r.folder_name for r in run.results] == ["exp_a", "shot_1.avi", "exp_b"]
    assert [r.source_kind for r in run.results] == ["folder", "video", "folder"]
    assert all(r.success for r in run.results), [r.message for r in run.results]
    assert [r.n_frames for r in run.results] == [4, 6, 3]
    for r in run.results:
        assert (r.radius > 0).all(), r.message
        assert r.confidence is None
    video = run.results[1]
    assert video.radius[-1] > video.radius[0]
    assert run.all_done == [(3, 13)]
    assert run.errors == []

    # Outputs: inside each folder; next to the video, named after it
    assert (tmp_path / "exp_a" / f"R_data_{TS}.mat").is_file()
    video_out = tmp_path / "shot_1" / f"R_data_shot_1_{TS}.mat"
    assert video.outputs == (str(video_out),)
    saved = np.ravel(loadmat(str(video_out))["Radius"])
    np.testing.assert_allclose(saved, video.radius)


def test_physical_export_next_to_video(qapp, tmp_path):
    source = _video(tmp_path, "shot", np.linspace(25, 70, 6))
    run = Run([source], export_physical=True, fps=1e6, um2px=3.2, rmax_fit_length=3)()
    assert run.results[0].success, run.results[0].message
    assert (tmp_path / "shot" / f"RofT_data_shot_{TS}.mat").is_file()


# -- Edge engine ------------------------------------------------------------------------

def test_edge_batch_real_engine_in_worker_processes(qapp, tmp_path):
    other = [0.0, 0.0, 30.0, 40.0, 50.0, 45.0, 35.0, 28.0]
    truths = {"edge_a": EDGE_RADII, "edge_b": other, "edge_v.avi": EDGE_RADII}
    sources = [
        _folder(tmp_path, "edge_a", EDGE_RADII),
        _folder(tmp_path, "edge_b", other),
        _video(tmp_path, "edge_v", EDGE_RADII),
    ]
    run = Run(sources, mode="edge", inline=False)()
    assert run.errors == []
    assert all(r.success for r in run.results), [r.message for r in run.results]

    for r in run.results:
        truth = np.array(truths[r.folder_name])
        bubble = truth > 0
        assert r.confidence is not None and r.confidence.shape == (8,)
        assert (r.radius[~bubble] == -1).all(), r.message
        tol = 0.05 if r.source_kind == "video" else 0.02  # 8-bit MJPG vs 16-bit TIFF
        np.testing.assert_allclose(r.radius[bubble], truth[bubble], rtol=tol,
                                   err_msg=r.folder_name)
        assert (r.confidence[bubble] >= 0.3).all()
        # The browser colours Edge sources by confidence, stored as the score
        np.testing.assert_allclose(r.quality_scores[bubble], r.confidence[bubble])
        assert len(r.flags) == 8 and all(isinstance(f, frozenset) for f in r.flags)
        assert set(r.metrics) == set(FIT_METRIC_KEYS)
        assert (r.region_count == 0).all()
        assert "reference frame" in r.message


class _FakeFit:
    """What the engine returns, reduced to the fields fit_edge reads."""

    def __init__(self, radius):
        self.radius = radius
        self.rc, self.cc, self.confidence = 125.0, 200.0, 0.8 if radius > 0 else 0.1
        self.flags = frozenset({"fake_flag"}) if radius > 40 else frozenset()
        self.edge_xy = np.array([[85.0, 200.0], [125.0, 240.0], [165.0, 200.0]])
        self.mask = None
        self.r_circle = radius + 0.5
        self.area_px = np.pi * radius ** 2
        self.accepted = radius > 0


def test_edge_confidence_flags_and_metrics_are_stored(qapp, tmp_path, monkeypatch):
    radii = iter([0.0, 30.0, 45.0])  # frames run in order on a single thread
    monkeypatch.setattr(edge_pipeline, "_engine",
                        lambda: lambda raw, gx, gy, reference, params: _FakeFit(next(radii)))
    monkeypatch.setattr(edge_pipeline, "_choose_reference", lambda read, n: 0)
    run = Run([_folder(tmp_path, "a", [10.0, 30.0, 45.0])], mode="edge")
    run.worker._n_workers = 1
    run()
    r = run.results[0]
    np.testing.assert_allclose(r.radius, [-1.0, 30.0, 45.0])
    np.testing.assert_allclose(r.confidence, [0.1, 0.8, 0.8])
    assert r.flags == [frozenset(), frozenset(), frozenset({"fake_flag"})]
    np.testing.assert_allclose(r.metrics["r_circle"], [0.5, 30.5, 45.5])
    assert np.isnan(r.metrics["circularity"]).all()
    np.testing.assert_allclose(r.quality_scores, [0.0, 0.8, 0.8])
    assert "reference frame 1" in r.message


def test_edge_reference_is_resolved_per_source(qapp, tmp_path, monkeypatch):
    seen = []
    real = edge_pipeline.resolve_reference_index

    def spy(read_frame, n_frames, explicit):
        seen.append((n_frames, explicit))
        return real(read_frame, n_frames, explicit)

    monkeypatch.setattr(edge_pipeline, "resolve_reference_index", spy)
    sources = [_folder(tmp_path, "a", EDGE_RADII[:5]), _video(tmp_path, "v", EDGE_RADII)]
    Run(sources, mode="edge")()
    assert seen == [(5, None), (8, None)]


# -- failures ---------------------------------------------------------------------------

def test_bad_sources_are_reported_and_the_rest_still_runs(qapp, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    broken = tmp_path / "broken.avi"
    broken.write_bytes(b"not a video")
    good = _folder(tmp_path, "good", [30.0, 40.0])
    run = Run([BatchSource(empty, "folder"), BatchSource(broken, "video"), good])()

    by_name = {r.folder_name: r for r in run.results}
    assert not by_name["empty"].success and "No images" in by_name["empty"].message
    assert not by_name["broken.avi"].success
    assert "Cannot open" in by_name["broken.avi"].message
    assert by_name["good"].success
    assert [d[1] for d in run.done] == [False, False, True]
    assert run.all_done == [(1, 2)]


def test_failing_frames_are_counted_not_fatal(qapp, tmp_path, monkeypatch):
    real = bfw._process_single_frame

    def flaky(args):
        if args[0] == 1:
            raise RuntimeError("boom")
        return real(args)

    monkeypatch.setattr(bfw, "_process_single_frame", flaky)
    run = Run([_folder(tmp_path, "a", [30.0, 40.0, 50.0])])()
    result = run.results[0]
    assert result.success
    assert result.radius[1] == -1 and result.radius[0] > 0
    assert "1 frame(s) failed" in result.message


def test_export_failure_keeps_the_results(qapp, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(bfw, "export_r_data", refuse)
    run = Run([_folder(tmp_path, "a", [30.0, 40.0])])()
    result = run.results[0]
    assert not result.success and "R_data export failed" in result.message
    assert result.has_results and (result.radius > 0).all()


def test_run_never_dies_silently(qapp, tmp_path, monkeypatch):
    def crash(item):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(bfw, "as_source", crash)
    run = Run([tmp_path])()
    assert run.errors and "unexpected" in run.errors[0]
    assert run.all_done == [(0, 0)]


def test_crash_inside_a_source_is_a_failed_source(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(bfw, "compute_all_quality_scores",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    run = Run([_folder(tmp_path, "a", [30.0, 40.0]), _folder(tmp_path, "b", [30.0])])()
    assert [r.success for r in run.results] == [False, False]
    assert "bad" in run.results[0].message
    assert run.all_done == [(0, 0)]


def test_stop_before_start(qapp, tmp_path):
    run = Run([_folder(tmp_path, "a", [30.0, 40.0])])
    run.worker.request_stop()
    run()
    assert run.results == [] and run.all_done == [(0, 0)]


def test_plain_paths_are_accepted(qapp, tmp_path):
    folder = write_sequence(tmp_path / "a", [30.0, 40.0])
    run = Run([folder])()
    assert run.results[0].success and run.results[0].source_kind == "folder"


def test_edge_mode_without_params_uses_defaults(qapp, tmp_path):
    run = Run([_folder(tmp_path, "a", EDGE_RADII[:4])], mode="edge", edge_params=None)()
    assert run.results[0].success and run.results[0].confidence is not None


@pytest.mark.parametrize("mode", ["intensity", "area"])
def test_legacy_modes_score_quality(qapp, tmp_path, mode):
    run = Run([_folder(tmp_path, "a", [40.0, 50.0])], mode=mode)()
    scores = run.results[0].quality_scores
    if mode == "area":
        assert np.isnan(scores).all()
    else:
        assert (scores > 0).all()
