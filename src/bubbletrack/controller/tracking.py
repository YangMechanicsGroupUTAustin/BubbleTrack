"""Sequence tracking after Fit All: background solve, store, review list.

The solve runs on a QThread with its own frame reader; it only reads a
snapshot of the results, which the GUI thread takes before starting it and
updates once it is done (the run lock keeps users from editing meanwhile).
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal

from bubbletrack.controller.edge_support import store_fit
from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.model.edge_pipeline import EdgeJob, read_raw
from bubbletrack.model.frame_source import open_recording
from bubbletrack.model.provenance import FitSource, is_user_confirmed
from bubbletrack.model.sequence import SOURCE_TRACKED, SequenceResult, solve_sequence
from bubbletrack.model.sequence_pipeline import (
    Redetector, TrackingStopped, anchors, observations,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingOutcome:
    """What a finished solve hands back to the GUI thread."""

    result: SequenceResult
    redetector: Redetector


class TrackingWorker(QThread):
    """Run the sequence solver over a snapshot of the current results."""

    solved = pyqtSignal(object)  # TrackingOutcome
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, state, job: EdgeJob) -> None:
        super().__init__()
        self._raw = observations(state)
        self._anchors = anchors(state)
        self._images = tuple(state.images)
        self._video_path = state.video_path if state.video_reader is not None else None
        self._gridx, self._gridy = state.gridx, state.gridy
        self._job = job
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        reader = None
        try:
            if self._video_path:
                reader = open_recording(self._video_path)  # private: not thread-safe
                read = reader.read_frame
            else:
                read = lambda i: read_raw(self._images[i])  # noqa: E731
            redetector = Redetector(read, self._gridx, self._gridy, self._job,
                                    stop_requested=lambda: self._stop)
            result = solve_sequence(self._raw, redetector, anchors=self._anchors)
        except TrackingStopped:
            self.stopped.emit()
            return
        except Exception as exc:
            logger.exception("Sequence tracking failed")
            self.failed.emit(str(exc))
            return
        finally:
            if reader is not None:
                reader.close()
        logger.info("Tracking: seed frame %d, %d re-detections, %d to review",
                    result.seed_frame, result.redetect_calls, len(result.review))
        self.solved.emit(TrackingOutcome(result, redetector))


def apply_tracking(state, outcome: TrackingOutcome,
                   allowed_frames: Collection[int] | None = None) -> list[int]:
    """Store the frames the solver re-measured; returns their indices.

    Other frames keep their single-frame results: confirmed (raw), anchors,
    and lost / no-bubble frames, which go to the review list instead.
    Manual (user-confirmed) and locked frames are never touched.  With
    *allowed_frames* only those frames may change (a re-track after an edit
    leaves frames that look good alone); None = every frame (Fit All).
    """
    changed = []
    for i, solution in enumerate(outcome.result.frames):
        if solution.source != SOURCE_TRACKED:
            continue
        if allowed_frames is not None and i not in allowed_frames:
            continue
        if state.fit_locked[i] or is_user_confirmed(state.fit_source[i]):
            continue
        fit = outcome.redetector.fit_for(i, solution)
        if fit is None:
            continue
        store_fit(state, i, fit, FitSource.TRACKED)
        changed.append(i)
    return changed


def review_items(result: SequenceResult) -> list[ReviewItem]:
    """The solver's review list, ranked by the review queue's own weights."""
    return [ReviewItem(frame, tuple(reasons), 0, result.frames[frame].confidence)
            for frame, reasons in result.review]
