"""Background worker for batch multi-folder processing."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from bubbletrack.controller.worker import _process_single_frame
from bubbletrack.model.export import export_r_data, export_rof_t_data
from bubbletrack.model.image_io import scan_folder

logger = logging.getLogger(__name__)


class BatchFolderWorker(QThread):
    """Process multiple experiment folders with identical parameters.

    Signals
    -------
    folder_started(int, int, str)
        (1-based index, total folders, folder name)
    frame_progress(int, int)
        (done frames, total frames) within the current folder
    folder_done(str, bool, str)
        (folder name, success, message)
    all_done(int, int)
        (folders succeeded, total frames fitted)
    """

    folder_started = pyqtSignal(int, int, str)
    frame_progress = pyqtSignal(int, int)
    folder_done = pyqtSignal(str, bool, str)
    all_done = pyqtSignal(int, int)
    # (folder_path, n_frames, n_fitted, radius_copy, quality_scores)
    folder_result_ready = pyqtSignal(str, int, int, object, object)

    def __init__(
        self,
        folders: list[Path],
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
        max_radius: float = float("inf"),
        export_physical: bool = False,
        fps: float = 1_000_000.0,
        um2px: float = 3.2,
        rmax_fit_length: int = 11,
    ) -> None:
        super().__init__()
        self._folders = folders
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
        self._max_radius = max_radius
        self._export_physical = export_physical
        self._fps = fps
        self._um2px = um2px
        self._rmax_fit_length = rmax_fit_length
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        folders = self._folders
        if not folders:
            self.all_done.emit(0, 0)
            return

        # Single timestamp for the entire batch run
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_workers = min(4, os.cpu_count() or 1)
        folders_ok = 0
        total_fitted = 0

        for fi, folder in enumerate(folders):
            if self._stop:
                break

            self.folder_started.emit(fi + 1, len(folders), folder.name)

            images = scan_folder(str(folder))
            if not images:
                self.folder_done.emit(folder.name, False, "No images found")
                continue

            n = len(images)
            radius = np.full(n, -1.0)
            circle_fit_par = np.full((n, 2), np.nan)
            circle_xy: list = [None] * n

            # Build args for ProcessPoolExecutor.  Tuple order MUST match
            # bubbletrack.controller.worker._process_single_frame.
            all_args = [
                (
                    i, images[i],
                    self._sensitivity, self._gridx, self._gridy,
                    tuple(self._edges), self._rf_slider, self._obj_radius,
                    self._gaussian_sigma, self._clahe_clip,
                    self._closing_radius, self._opening_radius,
                    self._detection_mode,
                    self._gradient_boundary,
                    self._max_radius,
                )
                for i in range(n)
            ]

            done_count = 0
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(_process_single_frame, args): args[0]
                        for args in all_args
                    }
                    for fut in as_completed(futures):
                        if self._stop:
                            pool.shutdown(wait=False, cancel_futures=True)
                            break
                        try:
                            idx, r, edge_xy, _ = fut.result()
                            radius[idx] = r
                            if r > 0 and edge_xy is not None and edge_xy.shape[0] >= 3:
                                from bubbletrack.model.circle_fit import circle_fit_taubin
                                rc, cc, _ = circle_fit_taubin(edge_xy)
                                circle_fit_par[idx] = [rc, cc]
                                circle_xy[idx] = edge_xy
                        except Exception as exc:
                            logger.error(
                                "Frame %d failed in %s: %s",
                                futures[fut], folder.name, exc,
                            )
                        done_count += 1
                        self.frame_progress.emit(done_count, n)
            except Exception as exc:
                self.folder_done.emit(folder.name, False, str(exc))
                continue

            if self._stop:
                break

            # Export results
            n_fitted = int(np.sum(radius > 0))

            # Compute quality scores while circle_xy is still available
            from bubbletrack.model.quality import compute_all_quality_scores
            q_scores = compute_all_quality_scores(
                radius, circle_fit_par, circle_xy,
                self._gridx, self._gridy,
            )
            self.folder_result_ready.emit(
                str(folder), n, n_fitted, radius.copy(), q_scores,
            )

            messages: list[str] = []

            # Always export R_data (pixel data)
            r_data_path = str(folder / f"R_data_{ts}.mat")
            try:
                export_r_data(r_data_path, radius, circle_fit_par, circle_xy)
                messages.append(f"{n_fitted}/{n} fitted")
            except Exception as exc:
                self.folder_done.emit(folder.name, False, f"R_data export failed: {exc}")
                continue

            # Optionally export RofT_data.mat (physical data)
            if self._export_physical and n_fitted >= 3:
                rof_t_path = str(folder / f"RofT_data_{ts}.mat")
                try:
                    ok, msg = export_rof_t_data(
                        rof_t_path, radius,
                        self._um2px, self._fps, self._rmax_fit_length,
                    )
                    if ok:
                        messages.append("+ RofT_data")
                    else:
                        messages.append(f"RofT skipped: {msg}")
                except Exception as exc:
                    messages.append(f"RofT failed: {exc}")

            self.folder_done.emit(folder.name, True, " | ".join(messages))
            folders_ok += 1
            total_fitted += n_fitted

        self.all_done.emit(folders_ok, total_fitted)
