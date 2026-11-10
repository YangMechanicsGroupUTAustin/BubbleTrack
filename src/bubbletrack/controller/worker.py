"""Background worker thread for batch bubble detection + fitting.

Uses ``concurrent.futures.ProcessPoolExecutor`` to distribute frame
processing across multiple CPU cores.  The module-level function
``_process_single_frame`` is picklable and runs in worker processes.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    from bubbletrack.model.image_io import load_and_normalize
    from bubbletrack.model.removing_factor import compute_removing_factor

    _, _, cur_img_roi, binary_roi = load_and_normalize(
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
    ):
        super().__init__()
        self._images = images
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

    def run(self):
        total = self._end - self._start + 1
        n_workers = min(4, os.cpu_count() or 1)
        logger.info(
            "Batch processing frames %d-%d with %d workers",
            self._start, self._end, n_workers,
        )

        # Build picklable args for all frames (tuples, not lists for edges)
        all_args = []
        for i in range(self._start, self._end + 1):
            all_args.append((
                i,
                self._images[i],
                self._sensitivity,
                self._gridx,
                self._gridy,
                tuple(self._edges),
                self._rf_slider,
                self._obj_radius,
                self._gaussian_sigma,
                self._clahe_clip,
                self._closing_radius,
                self._opening_radius,
                self._detection_mode,
                self._gradient_boundary,
                self._area_max_regions,
                self._max_radius,
            ))

        done_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_single_frame, args): args[0]
                for args in all_args
            }

            for fut in as_completed(futures):
                if self._stop:
                    pool.shutdown(wait=False, cancel_futures=True)
                    logger.warning("Batch stopped by user")
                    break

                idx = futures[fut]
                try:
                    result = fut.result()
                    self.frame_done.emit(
                        *result,
                    )
                except Exception as exc:
                    logger.error("Frame %d failed: %s", idx, exc)
                    self.frame_error.emit(idx, str(exc))
                    self.frame_done.emit(
                        idx, -1.0, np.empty((0, 2)), None,
                        float("nan"), float("nan"), 0,
                    )

                done_count += 1
                self.progress.emit(done_count, total)

        self.finished.emit()
