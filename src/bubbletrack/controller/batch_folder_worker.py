"""Background worker for batch processing of many sources (folders and videos).

Every source gets the same parameters.  In Edge mode each source gets its
own automatically chosen reference frame.  One process pool serves the
whole run; frames are submitted a few at a time so memory stays flat even
for long videos decoded in this thread.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import (
    FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait,
)
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from bubbletrack.controller.worker import _process_single_frame, _process_single_frame_edge
from bubbletrack.model import edge_pipeline
from bubbletrack.model.batch_experiments import BatchSource, SourceFrames, as_source
from bubbletrack.model.batch_result import FolderResult
from bubbletrack.model.export import export_r_data, export_rof_t_data
from bubbletrack.model.quality import compute_all_quality_scores
from bubbletrack.model.state import FIT_METRIC_KEYS

logger = logging.getLogger(__name__)


class _FrameResults:
    """Per-frame arrays of one source, filled as frames complete."""

    def __init__(self, n: int, edge: bool) -> None:
        self.radius = np.full(n, -1.0)
        self.centres = np.full((n, 2), np.nan)
        self.edges: list = [None] * n
        self.region_count = np.zeros(n, dtype=int)
        self.confidence = np.full(n, np.nan) if edge else None
        self.flags = [frozenset()] * n if edge else None
        self.metrics = {k: np.full(n, np.nan) for k in FIT_METRIC_KEYS} if edge else None
        self.failed = 0

    def store(self, result: tuple) -> None:
        idx, r, edge_xy, _mask, rc, cc, count = result[:7]
        self.radius[idx] = r
        self.region_count[idx] = count
        if r > 0 and edge_xy is not None and edge_xy.shape[0] >= 3:
            self.centres[idx] = [rc, cc]
            self.edges[idx] = edge_xy
        if self.confidence is not None and len(result) > 7:
            meta = result[7]
            self.confidence[idx] = meta.get("confidence", np.nan)
            self.flags[idx] = frozenset(meta.get("flags", ()))
            reported = meta.get("metrics", {})
            for key, values in self.metrics.items():
                values[idx] = reported.get(key, np.nan)


class BatchFolderWorker(QThread):
    """Process several sources (image folders and videos) with identical parameters.

    Signals
    -------
    folder_started(int, int, str)
        (1-based index, total sources, source name)
    frame_progress(int, int)
        (done frames, total frames) within the current source
    folder_done(str, bool, str)
        (source name, success, message)
    result_ready(object)
        A ``FolderResult`` per source, failed sources included.
    error(str)
        The run itself failed; ``all_done`` still follows.
    all_done(int, int)
        (sources succeeded, total frames fitted); always emitted last.
    """

    folder_started = pyqtSignal(int, int, str)
    frame_progress = pyqtSignal(int, int)
    folder_done = pyqtSignal(str, bool, str)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    all_done = pyqtSignal(int, int)

    # Run frames on threads in this process instead of a process pool
    # (tests that inject fakes; debugging).
    inline = False

    def __init__(
        self,
        sources: list,
        sensitivity: float,
        gridx: tuple[int, int],
        gridy: tuple[int, int],
        removing_factor_slider: int,
        bubble_cross_edges: tuple[bool, ...],
        removing_obj_radius: int = 0,
        gaussian_sigma: float = 0.0,
        clahe_clip: float = 0.0,
        closing_radius: int = 0,
        opening_radius: int = 0,
        detection_mode: str = "intensity",
        gradient_boundary: float = 0.5,
        area_max_regions: int = 1,
        max_radius: float = float("inf"),
        export_physical: bool = False,
        fps: float = 1_000_000.0,
        um2px: float = 3.2,
        rmax_fit_length: int = 11,
        edge_params=None,
        timestamp: str | None = None,
    ) -> None:
        super().__init__()
        self._sources = list(sources)  # BatchSource, or plain paths
        self._gridx = gridx
        self._gridy = gridy
        self._mode = detection_mode
        self._max_radius = max_radius
        # Tuple order after (idx, source) MUST match worker._process_single_frame.
        self._legacy_args = (
            sensitivity, gridx, gridy, tuple(bubble_cross_edges),
            removing_factor_slider, removing_obj_radius, gaussian_sigma, clahe_clip,
            closing_radius, opening_radius, detection_mode, gradient_boundary,
            area_max_regions, max_radius,
        )
        self._edge_params = edge_params
        self._export_physical = export_physical
        self._fps = fps
        self._um2px = um2px
        self._rmax_fit_length = rmax_fit_length
        self._timestamp = timestamp  # one timestamp for every output of the run
        self._n_workers = min(4, os.cpu_count() or 1)
        self._executor = None
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    # -- run ---------------------------------------------------------------- #

    def run(self) -> None:
        ok = fitted = 0
        try:
            sources = [as_source(s) for s in self._sources]
            ts = self._timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, source in enumerate(sources):
                if self._stop:
                    break
                self.folder_started.emit(i + 1, len(sources), source.name)
                result = self._run_source(source, ts)
                if result is None:  # stopped part-way: nothing exported
                    break
                self.result_ready.emit(result)
                self.folder_done.emit(result.folder_name, result.success, result.message)
                if result.success:
                    ok += 1
                    fitted += result.n_fitted
        except Exception as exc:
            logger.exception("Batch run failed")
            self.error.emit(f"Batch failed: {exc}")
        finally:
            self._discard_pool(wait=not self._stop)
            self.all_done.emit(ok, fitted)

    def _run_source(self, source: BatchSource, ts: str) -> FolderResult | None:
        """Fit and export one source; failures become a failed result."""
        def fail(message: str) -> FolderResult:
            return FolderResult.failure(str(source.path), source.name, message,
                                        source_kind=source.kind)

        try:
            frames = SourceFrames(source)
        except Exception as exc:
            logger.warning("Cannot open %s: %s", source.path, exc)
            return fail(f"Cannot open: {exc}")
        try:
            if frames.n_frames == 0:
                return fail("No images found" if source.kind == "folder"
                            else "Video has no frames")
            job, note = self._edge_job(frames) if self._mode == "edge" else (None, "")
            results = self._fit_frames(frames, job)
            if results is None:
                return None
            return self._finish(source, results, ts, note)
        except BrokenProcessPool as exc:
            self._discard_pool(wait=False)  # the next source gets a fresh pool
            logger.error("Worker processes died on %s: %s", source.path, exc)
            return fail(f"Worker processes died: {exc}")
        except Exception as exc:
            logger.exception("Batch source %s failed", source.path)
            return fail(f"{type(exc).__name__}: {exc}")
        finally:
            frames.close()

    # -- fitting ------------------------------------------------------------ #

    def _edge_job(self, frames: SourceFrames) -> tuple[edge_pipeline.EdgeJob, str]:
        """The Edge job for one source, with its own auto-selected reference frame."""
        idx = edge_pipeline.resolve_reference_index(frames.read_raw, frames.n_frames, None)
        reference = frames.read_raw(idx) if idx is not None else None
        params = (self._edge_params if self._edge_params is not None
                  else edge_pipeline.make_params())
        job = edge_pipeline.EdgeJob(params=params, reference=reference,
                                    max_radius=self._max_radius)
        note = (f"reference frame {idx + 1}" if idx is not None
                else "no reference frame (surface fallback)")
        return job, note

    def _frame_args(self, i: int, source, job) -> tuple:
        if job is not None:
            return (i, source, self._gridx, self._gridy, job)
        return (i, source) + self._legacy_args

    def _pool(self):
        if self._executor is None:
            cls = ThreadPoolExecutor if self.inline else ProcessPoolExecutor
            self._executor = cls(max_workers=self._n_workers)
        return self._executor

    def _discard_pool(self, wait: bool) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    def _fit_frames(self, frames: SourceFrames, job) -> _FrameResults | None:
        """Fit every frame; None when stopped.  Bounded in-flight submission."""
        n = frames.n_frames
        out = _FrameResults(n, edge=job is not None)
        process = _process_single_frame_edge if job is not None else _process_single_frame
        pool = self._pool()
        indices = iter(range(n))
        pending: dict = {}
        done = 0

        def submit_next() -> bool:
            nonlocal done
            for i in indices:
                try:
                    source = frames.read(i)
                except Exception as exc:  # e.g. an undecodable video frame
                    logger.warning("Frame %d of %s unreadable: %s", i, frames.source.name, exc)
                    out.failed += 1
                    done += 1
                    self.frame_progress.emit(done, n)
                    continue
                pending[pool.submit(process, self._frame_args(i, source, job))] = i
                return True
            return False

        while len(pending) < 2 * self._n_workers and submit_next():
            pass
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                idx = pending.pop(fut)
                try:
                    out.store(fut.result())
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    logger.error("Frame %d failed in %s: %s", idx, frames.source.name, exc)
                    out.failed += 1
                done += 1
                self.frame_progress.emit(done, n)
            if self._stop:
                for fut in pending:
                    fut.cancel()
                return None
            while len(pending) < 2 * self._n_workers and submit_next():
                pass
        return out

    # -- results and export ------------------------------------------------- #

    def _scores(self, res: _FrameResults) -> np.ndarray:
        if res.confidence is not None:
            # The Edge engine scores its own fits (as the main chart does).
            return np.where(res.radius > 0, res.confidence, 0.0)
        scores = compute_all_quality_scores(
            res.radius, res.centres, res.edges, self._gridx, self._gridy,
        )
        if self._mode == "area":
            scores[res.radius > 0] = np.nan
        return scores

    def _finish(self, source: BatchSource, res: _FrameResults, ts: str,
                note: str) -> FolderResult:
        n = len(res.radius)
        n_fitted = int(np.sum(res.radius > 0))
        messages = [f"{n_fitted}/{n} fitted"]
        if res.failed:
            messages.append(f"{res.failed} frame(s) failed")
        if note:
            messages.append(note)
        scores = self._scores(res)
        outputs: list[str] = []
        success = True

        r_path = source.output_path("R_data", ts)
        try:
            export_r_data(str(r_path), res.radius, res.centres, res.edges, res.region_count)
            outputs.append(str(r_path))
        except Exception as exc:
            logger.error("R_data export to %s failed: %s", r_path, exc)
            messages.append(f"R_data export failed: {exc}")
            success = False

        if success and self._export_physical and n_fitted >= 3:
            messages.append(self._export_rof_t(source, res.radius, ts, outputs))

        return FolderResult(
            folder_path=str(source.path), folder_name=source.name, success=success,
            message=" | ".join(messages), n_frames=n, n_fitted=n_fitted,
            radius=res.radius, quality_scores=scores, circle_fit_par=res.centres,
            circle_xy=res.edges, region_count=res.region_count,
            source_kind=source.kind, confidence=res.confidence, flags=res.flags,
            metrics=res.metrics, outputs=tuple(outputs),
        )

    def _export_rof_t(self, source: BatchSource, radius: np.ndarray, ts: str,
                      outputs: list[str]) -> str:
        path = source.output_path("RofT_data", ts)
        try:
            ok, msg = export_rof_t_data(str(path), radius, self._um2px, self._fps,
                                        self._rmax_fit_length)
        except Exception as exc:
            logger.error("RofT export to %s failed: %s", path, exc)
            return f"RofT failed: {exc}"
        if not ok:
            return f"RofT skipped: {msg}"
        outputs.append(str(path))
        return "+ RofT_data"
