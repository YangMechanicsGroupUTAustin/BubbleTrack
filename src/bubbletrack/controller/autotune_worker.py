"""Background worker for automatic parameter tuning."""

from __future__ import annotations

import threading

from PyQt6.QtCore import QThread, pyqtSignal

from bubbletrack.model.autotune import run_autotune


class AutoTuneWorker(QThread):
    """Runs auto-tune grid search in a background thread.

    Signals
    -------
    progress(int, int)      : (current_step, total_steps)
    result_ready(object)    : AutoTuneResult or None

    Note: Do NOT name the result signal ``finished`` — that shadows
    QThread.finished and breaks its internal thread-cleanup mechanism,
    causing 'QThread: Destroyed while thread is still running' crashes.
    """

    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(object)

    def __init__(
        self,
        image_path: str,
        gridx: tuple[int, int],
        gridy: tuple[int, int],
        bubble_cross_edges: tuple[bool, ...],
        *,
        gaussian_sigma: float = 0.0,
        clahe_clip: float = 0.0,
        opening_radius: int = 0,
        closing_radius: int = 0,
        max_radius: float = float("inf"),
        divisor: int | None = None,
    ) -> None:
        super().__init__()
        self._image_path = image_path
        self._gridx = gridx
        self._gridy = gridy
        self._edges = bubble_cross_edges
        self._gaussian_sigma = gaussian_sigma
        self._clahe_clip = clahe_clip
        self._opening_radius = opening_radius
        self._closing_radius = closing_radius
        self._max_radius = max_radius
        self._divisor = divisor
        self._cancel_event = threading.Event()

    def run(self) -> None:
        result = run_autotune(
            self._image_path,
            self._gridx,
            self._gridy,
            self._edges,
            gaussian_sigma=self._gaussian_sigma,
            clahe_clip=self._clahe_clip,
            opening_radius=self._opening_radius,
            closing_radius=self._closing_radius,
            max_radius=self._max_radius,
            divisor=self._divisor,
            progress_callback=self._on_progress,
            cancel_check=self._cancel_event.is_set,
        )
        self.result_ready.emit(result)

    def cancel(self) -> None:
        self._cancel_event.set()

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)
