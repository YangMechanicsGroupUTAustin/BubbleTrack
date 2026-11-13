"""Background training and previews of ML assist's teaching mode.

A request trains the model when the examples changed (< 0.5 s), then
measures the shown frame (with its P(bubble) map, for the tint) and the
suggested frames, emitting each as it is done.  One request runs at a time
on its own thread with a private frame reader; ``PreviewRunner`` keeps only
the newest wish while one runs and builds the next request from the current
state when the thread is free, so a burst of strokes trains once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from bubbletrack.model import edge_pipeline
from bubbletrack.model.edge_pipeline import EdgeJob
from bubbletrack.model.ml_pipeline import MLFrameFit, measure_ml

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PreviewRequest:
    version: int
    """The examples version: a model trained here belongs to it."""
    examples: object | None
    """``MLExamples`` to train on; None = measure with *model*."""
    model: object | None
    frames: tuple[int, ...]
    """The shown frame first (it gets the probability map), then the suggested ones."""
    images: tuple[str, ...]
    video_path: str | None
    gridx: tuple[int, int]
    gridy: tuple[int, int]
    job: EdgeJob
    stored: dict = field(default_factory=dict)
    """{frame: StoredAuto} plain-Auto results, as Apply would use them."""
    key: object = None
    """What the model is valid for (examples, ROI, reference), for the controller."""


@dataclass(frozen=True, eq=False)
class FramePreview:
    frame: int
    fit: MLFrameFit
    view: np.ndarray
    """The ROI as Auto sees it (frame / reference)."""
    probability: np.ndarray | None = None
    """P(bubble) on the ROI (the shown frame only)."""


class PreviewWorker(QThread):
    """Run one request: (train,) measure every frame, report as it goes."""

    trained = pyqtSignal(object, object, float)  # request, MLModel, training ms
    failed = pyqtSignal(object, str)  # request, message
    frame_ready = pyqtSignal(object, object)  # request, FramePreview

    def __init__(self, request: PreviewRequest) -> None:
        super().__init__()
        self.request = request
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        req, reader = self.request, None
        try:
            if req.video_path:
                from bubbletrack.model.frame_source import open_recording
                reader = open_recording(req.video_path)  # private: readers are not thread-safe
                read = reader.read_frame
            else:
                read = lambda i: edge_pipeline.read_raw(req.images[i])  # noqa: E731
            model = req.model
            if req.examples is not None:
                model = self._train(req, read)
            for k, frame in enumerate(req.frames):
                if self._stop:
                    return
                self.frame_ready.emit(req, self._measure(model, req, frame, read(frame), k == 0))
        except Exception as exc:  # noqa: BLE001 - shown in the panel, never fatal
            if not isinstance(exc, ValueError):
                logger.exception("ML assist preview failed")
            self.failed.emit(req, str(exc))
        finally:
            if reader is not None:
                reader.close()

    def _train(self, req: PreviewRequest, read: Callable[[int], np.ndarray]):
        from bubbletrack.model.teach import train_model

        t0 = time.perf_counter()
        model = train_model(req.examples, read, req.gridx, req.gridy, reference=req.job.reference)
        self.trained.emit(req, model, (time.perf_counter() - t0) * 1e3)
        return model

    @staticmethod
    def _measure(model, req: PreviewRequest, frame: int, raw: np.ndarray, shown: bool) -> FramePreview:
        prob = model.predict(raw).probability if shown else None
        fit = measure_ml(model, raw, req.gridx, req.gridy, req.job, req.stored.get(frame))
        view = edge_pipeline.view_edge(raw, req.gridx, req.gridy, req.job)
        return FramePreview(frame, fit, np.asarray(view, np.float32), prob)


class PreviewRunner(QObject):
    """One preview thread at a time; wishes made meanwhile are merged.

    *build(train)* makes the request from the current state when a thread is
    free (None: nothing to do).
    """

    def __init__(self, build: Callable[[bool], PreviewRequest | None], *,
                 on_trained: Callable, on_failed: Callable, on_frame: Callable,
                 worker_factory=PreviewWorker) -> None:
        super().__init__()
        self._build = build
        self._slots = (on_trained, on_failed, on_frame)
        self.worker_factory = worker_factory
        self._worker = None
        self._retired: list = []  # stopped threads still finishing (keep them alive)
        self._wanted = False
        self._want_train = False

    @property
    def busy(self) -> bool:
        return self._worker is not None

    def request(self, *, train: bool) -> None:
        self._wanted = True
        self._want_train = self._want_train or train
        self._kick()

    def cancel(self) -> None:
        """Forget every wish and stop the running thread (its results are dropped)."""
        self._wanted = self._want_train = False
        if self._worker is not None:
            self._worker.request_stop()
            self._retired.append(self._worker)
            self._worker = None

    def shutdown(self, timeout_ms: int = 10_000) -> None:
        self.cancel()
        for worker in self._retired:
            worker.wait(timeout_ms)
        self._retired.clear()

    def _kick(self) -> None:
        if self._worker is not None or not self._wanted:
            return
        train, self._wanted, self._want_train = self._want_train, False, False
        request = self._build(train)
        if request is None:
            return
        worker = self.worker_factory(request)
        on_trained, on_failed, on_frame = self._slots
        worker.trained.connect(lambda req, model, ms, w=worker: self._live(w) and on_trained(req, model, ms))
        worker.failed.connect(lambda req, msg, w=worker: self._live(w) and on_failed(req, msg))
        worker.frame_ready.connect(lambda req, prev, w=worker: self._live(w) and on_frame(req, prev))
        worker.finished.connect(lambda w=worker: self._on_finished(w))
        self._worker = worker
        worker.start()

    def _live(self, worker) -> bool:
        return worker is self._worker

    def _on_finished(self, worker) -> None:
        worker.wait(1000)  # 'finished' comes last: never drop a QThread still running
        if worker in self._retired:
            self._retired.remove(worker)
        if worker is self._worker:
            self._worker = None
            self._kick()
