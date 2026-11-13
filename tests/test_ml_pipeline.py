"""ML assist on whole frames (scikit-learn): what is stored for a clean frame
(Auto guided by ML) and a hard one (ML's own outline), the stored plain-Auto
result reused, and the pool workers that retrain from the examples."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("sklearn")

import _ml_synth as S  # noqa: E402
from bubbletrack.controller.worker import BatchWorker  # noqa: E402
from bubbletrack.model import ml_pipeline  # noqa: E402
from bubbletrack.model.edge_pipeline import EdgeJob, fit_edge, make_params  # noqa: E402
from bubbletrack.model.ml_pipeline import MLJob, StoredAuto, measure_ml, train_job  # noqa: E402
from bubbletrack.model.provenance import FitSource  # noqa: E402
from bubbletrack.model.teach import MLExamples  # noqa: E402

GRID = ((1, S.H), (1, S.W))
TAUGHT = (3, 7)  # one clean and one hard frame, verified


@pytest.fixture(scope="module")
def data():
    frames = S.frames()
    examples = MLExamples()
    for f in TAUGHT:
        examples = examples.verify(f, S.circle(f))
    job = MLJob(examples.to_dict(), {f: frames[f] for f in TAUGHT}, *GRID, frames[0])
    edge = EdgeJob(params=make_params(), reference=frames[0], max_radius=float(S.W))
    return frames, job, edge, train_job(job)


def test_clean_frame_is_auto_guided_by_ml(data):
    frames, _, edge, model = data
    out = measure_ml(model, frames[2], *GRID, edge)
    assert out.source == FitSource.AUTO and out.ml_source == "auto_ml"
    assert out.fit.radius == pytest.approx(S.RADII[2], rel=0.05)
    assert "ml_guided" in out.fit.flags and len(out.fit.edge_xy) > 10


def test_hard_frame_gets_mls_own_outline(data):
    frames, _, edge, model = data
    assert fit_edge(frames[6], *GRID, edge).radius < 0          # Auto alone finds nothing
    out = measure_ml(model, frames[6], *GRID, edge)
    assert out.source == FitSource.ML and out.ml_source == "ml"
    assert out.fit.radius == pytest.approx(S.RADII[6], rel=0.1)
    assert (out.fit.rc, out.fit.cc) == pytest.approx(S.circle(6)[:2], abs=3.0)
    assert "ml_outline" in out.fit.flags and out.fit.confidence >= 0.3


def test_stored_auto_result_gives_the_same_answer(data):
    frames, _, edge, model = data
    plain = fit_edge(frames[2], *GRID, edge)
    stored = StoredAuto(plain.radius, plain.rc, plain.cc, plain.confidence, plain.flags,
                        plain.edge_xy, plain.metrics)
    fresh, reused = measure_ml(model, frames[2], *GRID, edge), measure_ml(model, frames[2], *GRID, edge, stored)
    assert (reused.ml_source, reused.fit.radius) == (fresh.ml_source, fresh.fit.radius)


def test_pool_workers_train_the_same_model(data):
    frames, job, edge, model = data
    ml_pipeline.init_worker(job)
    try:
        args = (6, frames[6], *GRID, edge, None)
        idx, radius, xy, _mask, rc, cc, _count, meta = ml_pipeline.process_frame(args)
        gui = measure_ml(model, frames[6], *GRID, edge)            # trained with 4 threads
        assert idx == 6 and radius == gui.fit.radius and (rc, cc) == (gui.fit.rc, gui.fit.cc)
        assert meta["source"] == FitSource.ML and meta["flags"] == gui.fit.flags
    finally:
        ml_pipeline.init_worker(MLJob({"version": 0}, {}, *GRID, None))   # leave a failed worker


def test_a_worker_that_cannot_train_reports_it_per_frame(data):
    frames, _, edge, _ = data
    ml_pipeline.init_worker(MLJob(MLExamples().to_dict(), {}, *GRID, None))
    with pytest.raises(RuntimeError, match="no examples"):
        ml_pipeline.process_frame((1, frames[1], *GRID, edge, None))


def _run(worker: BatchWorker, qapp, timeout_s=240.0):
    done, meta, finished = {}, {}, []
    worker.frame_done.connect(lambda idx, r, *rest: done.__setitem__(idx, r))
    worker.frame_meta.connect(lambda idx, m: meta.__setitem__(idx, m))
    worker.finished.connect(lambda: finished.append(True))
    worker.start()
    deadline = time.monotonic() + timeout_s
    while not finished and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    worker.wait(10_000)
    qapp.processEvents()
    return done, meta


@pytest.mark.parametrize("inline", [True, False], ids=["threads", "processes"])
def test_batch_worker_measures_a_frame_set_with_ml_assist(data, qapp, tmp_path, inline):
    frames, job, edge, _ = data
    folder = S.write_recording(tmp_path / "rec")
    images = sorted(str(p) for p in folder.glob("*.tiff"))
    plain = fit_edge(frames[2], *GRID, edge)
    stored = {2: StoredAuto(plain.radius, plain.rc, plain.cc, plain.confidence, plain.flags,
                            plain.edge_xy, plain.metrics)}
    worker = BatchWorker(images, 0, 9, 0.5, *GRID, 90, [False] * 4, detection_mode="edge",
                         edge_job=edge, ml_job=job, ml_stored=stored, frames=[2, 6])
    worker.inline = inline
    done, meta = _run(worker, qapp)
    assert sorted(done) == [2, 6] and sorted(meta) == [2, 6]
    assert meta[2]["source"] == FitSource.AUTO and meta[6]["source"] == FitSource.ML
    assert done[6] == pytest.approx(S.RADII[6], rel=0.1)
    assert np.isfinite(meta[6]["confidence"])
