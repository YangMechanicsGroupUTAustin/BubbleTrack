"""Background worker thread for batch bubble detection + fitting.

Uses ``concurrent.futures.ProcessPoolExecutor`` to distribute frame
processing across multiple CPU cores.  The module-level function
``_process_single_frame`` is picklable and runs in worker processes.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import (
    FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait,
)

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


def _process_single_frame(args: tuple):
    """Process a single frame.  Must be module-level for multiprocessing pickle."""
    (
        idx, filepath, sensitivity, gridx, gridy,
        edges, rf_slider, removing_obj_radius,
        gaussian_sigma, clahe_clip,
        closing_radius, opening_radius, detection_mode, gradient_boundary,
        area_max_regions,
        max_radius,
    ) = args

    # Import inside the function — each worker process needs its own imports
    from bubbletrack.model.detection import detect_bubble_with_mode, measure_detection
    from bubbletrack.model.frame_source import normalize_source
    from bubbletrack.model.removing_factor import compute_removing_factor

    # ``filepath`` is an image path, or a raw frame already decoded from a video
    _, _, cur_img_roi, binary_roi = normalize_source(
        filepath, sensitivity, gridx, gridy,
        gaussian_sigma=gaussian_sigma, clahe_clip=clahe_clip,
    )
    rf = compute_removing_factor(rf_slider, gridx, gridy)
    processed, edge_xy = detect_bubble_with_mode(
        binary_roi, cur_img_roi, detection_mode, sensitivity,
        list(edges), rf, gridx, gridy, removing_obj_radius,
        opening_radius=opening_radius, closing_radius=closing_radius,
        gaussian_sigma=gaussian_sigma,
        gradient_boundary=gradient_boundary,
        area_max_regions=area_max_regions,
    )

    radius, rc, cc, count = measure_detection(
        processed, edge_xy, detection_mode, gridx, gridy,
    )
    if not np.isfinite(radius) or radius <= 0 or radius > max_radius:
        radius = -1.0
        edge_xy = np.empty((0, 2))
        count = 0

    return idx, radius, edge_xy, processed, rc, cc, count


def _process_single_frame_edge(args: tuple):
    """Edge-engine counterpart of ``_process_single_frame`` (module-level, picklable).

    Returns the same 7 fields plus the fit confidence.
    """
    idx, source, gridx, gridy, job = args
    from bubbletrack.model.edge_pipeline import fit_edge, read_raw

    fit = fit_edge(read_raw(source), gridx, gridy, job)
    return (idx, fit.radius, fit.edge_xy, fit.mask, fit.rc, fit.cc, 0, fit.confidence)


class BatchWorker(QThread):
    """Process a range of frames in the background using multiple processes.

    Signals
    -------
    progress(current, total)
    frame_done(frame_idx, radius, edge_xy, binary_roi)
    finished()
    error(str)
    """

    progress = pyqtSignal(int, int)
    frame_done = pyqtSignal(int, float, object, object, float, float, int)
    frame_error = pyqtSignal(int, str)  # (frame_idx, error_message)
    frame_meta = pyqtSignal(int, float)  # (frame_idx, confidence) — Edge engine only

    # Run frames in this process (threads) instead of a process pool.
    # Used by tests that inject fakes, and handy when debugging.
    inline = False
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        images: list[str],
        start: int,
        end: int,
        sensitivity: float,
        gridx: tuple[int, int],
        gridy: tuple[int, int],
        removing_factor_slider: int,
        bubble_cross_edges: list[bool],
        removing_obj_radius: int = 0,
        gaussian_sigma: float = 0.0,
        clahe_clip: float = 0.0,
        closing_radius: int = 0,
        opening_radius: int = 0,
        detection_mode: str = "intensity",
        gradient_boundary: float = 0.5,
        area_max_regions: int = 1,
        max_radius: float = float("inf"),
        video_path: str | None = None,
        edge_job=None,
    ):
        super().__init__()
        self._edge_job = edge_job  # EdgeJob when detection_mode == "edge"
        self._images = images
        self._video_path = video_path  # decode frames here instead of by path
        self._start = start        # 0-indexed
        self._end = end            # 0-indexed inclusive
        self._sensitivity = sensitivity
        self._gridx = gridx
        self._gridy = gridy
        self._rf_slider = removing_factor_slider
        self._edges = bubble_cross_edges
        self._obj_radius = removing_obj_radius
        self._gaussian_sigma = gaussian_sigma
        self._clahe_clip = clahe_clip
        self._closing_radius = closing_radius
        self._opening_radius = opening_radius
        self._detection_mode = detection_mode
        self._gradient_boundary = gradient_boundary
        self._area_max_regions = area_max_regions
        self._max_radius = max_radius
        self._stop = False

    def request_stop(self):
        self._stop = True

    def _frame_args(self, i: int, source) -> tuple:
        if self._edge_job is not None:
            return (i, source, self._gridx, self._gridy, self._edge_job)
        # Tuple order MUST match _process_single_frame.
        return (
            i, source, self._sensitivity, self._gridx, self._gridy,
            tuple(self._edges), self._rf_slider, self._obj_radius,
            self._gaussian_sigma, self._clahe_clip,
            self._closing_radius, self._opening_radius,
            self._detection_mode, self._gradient_boundary,
            self._area_max_regions, self._max_radius,
        )

    def _emit_failure(self, idx: int, exc: Exception) -> None:
        logger.error("Frame %d failed: %s", idx, exc)
        self.frame_error.emit(idx, str(exc))
        self.frame_done.emit(
            idx, -1.0, np.empty((0, 2)), None, float("nan"), float("nan"), 0,
        )

    def run(self):
        total = self._end - self._start + 1
        n_workers = min(4, os.cpu_count() or 1)
        logger.info(
            "Batch processing frames %d-%d with %d workers",
            self._start, self._end, n_workers,
        )

        reader = None
        if self._video_path:
            # A private reader: VideoCapture must not be shared across threads.
            from bubbletrack.model.image_io import VideoFrameReader
            try:
                reader = VideoFrameReader(self._video_path)
            except Exception as exc:
                self.error.emit(f"Cannot open video: {exc}")
                self.finished.emit()
                return

        # Bounded submission: only a few frames are in flight at once, so
        # memory stays flat for long sequences and decoded video frames.
        frame_iter = iter(range(self._start, self._end + 1))
        max_in_flight = 2 * n_workers
        done_count = 0
        try:
            executor = ThreadPoolExecutor if self.inline else ProcessPoolExecutor
            process = (_process_single_frame_edge if self._edge_job is not None
                       else _process_single_frame)
            with executor(max_workers=n_workers) as pool:
                pending: dict = {}

                def submit_next() -> bool:
                    nonlocal done_count
                    for i in frame_iter:
                        try:
                            source = reader.read_frame(i) if reader else self._images[i]
                        except Exception as exc:  # unreadable video frame
                            self._emit_failure(i, exc)
                            done_count += 1
                            self.progress.emit(done_count, total)
                            continue
                        pending[pool.submit(process, self._frame_args(i, source))] = i
                        return True
                    return False

                while len(pending) < max_in_flight and submit_next():
                    pass

                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        idx = pending.pop(fut)
                        try:
                            result = fut.result()
                            self.frame_done.emit(*result[:7])
                            if len(result) > 7:
                                self.frame_meta.emit(idx, float(result[7]))
                        except Exception as exc:
                            self._emit_failure(idx, exc)
                        done_count += 1
                        self.progress.emit(done_count, total)
                    if self._stop:
                        pool.shutdown(wait=False, cancel_futures=True)
                        logger.warning("Batch stopped by user")
                        break
                    while len(pending) < max_in_flight and submit_next():
                        pass
        except Exception as exc:  # e.g. the process pool could not start
            logger.exception("Batch worker failed")
            self.error.emit(f"Fitting failed: {exc}")
        finally:
            if reader is not None:
                reader.close()
            self.finished.emit()
