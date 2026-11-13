"""Key-frame filmstrip: cached thumbnails and wiring (which frames: ``key_frames``).

Install with one call after the AppController (and its Review) exists::

    install_filmstrip(window, app_controller)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QPointF, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF

from bubbletrack.controller.display_mixin import current_source
from bubbletrack.controller.key_frames import (  # noqa: F401 (re-exported)
    KeyFrame, choose_key_frames, find_collapses, key_frame_tooltip, to_check_frames,
)
from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.model.conventions import roi_to_slice
from bubbletrack.model.image_io import ImageLoadError, imread_unicode, to_grayscale
from bubbletrack.ui.filmstrip import THUMB_HEIGHT, THUMB_MAX_WIDTH, Filmstrip, Thumbnail
from bubbletrack.ui.frame_scrubber import frame_file_label
from bubbletrack.ui.image_panel import is_ordered_contour

logger = logging.getLogger(__name__)

REFRESH_DEBOUNCE_MS = 250
THUMB_CACHE_SIZE = 64
CIRCLE_COLOUR = "#818cf8"
OUTLINE_COLOUR = "#ef4444"
ReviewRows = Callable[[], Sequence]  # the Review rows (ReviewSegment), in queue order


# --------------------------------------------------------------------- #
#  Thumbnails
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class _ThumbBase:
    gray: np.ndarray          # uint8 thumbnail of the ROI
    scale: float              # thumbnail px per image px
    origin: tuple[int, int]   # (row, col) of the ROI's top-left, 0-based


def _frame_gray_u8(state, idx: int) -> np.ndarray:
    """The raw frame, min-max scaled to uint8 like the image panel shows it."""
    source = current_source(state, idx)
    raw = source if isinstance(source, np.ndarray) else imread_unicode(source)
    if raw is None:
        raise ImageLoadError(f"cannot read frame {idx + 1}")
    gray = to_grayscale(raw).astype(np.float64)
    lo, hi = float(np.nanmin(gray)), float(np.nanmax(gray))
    scaled = (gray - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(gray)
    return np.nan_to_num(scaled).astype(np.uint8)


def make_thumb_base(gray: np.ndarray, gridx, gridy, height: int = THUMB_HEIGHT,
                    max_width: int = THUMB_MAX_WIDTH) -> _ThumbBase:
    """Crop the ROI and shrink it to *height* (or *max_width*) pixels."""
    rs, cs = roi_to_slice(gridx, gridy)
    crop = gray[rs, cs]
    if crop.size == 0:
        crop, rs, cs = gray, slice(0, gray.shape[0]), slice(0, gray.shape[1])
    h, w = crop.shape
    scale = min(height / h, max_width / w)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return _ThumbBase(np.ascontiguousarray(small), scale, (rs.start or 0, cs.start or 0))


def render_thumbnail(base: _ThumbBase, circle: tuple[float, float, float] | None,
                     outline: np.ndarray | None) -> QPixmap:
    """Draw the fitted circle (row, col, r) and boundary onto the thumbnail."""
    h, w = base.gray.shape
    image = QImage(base.gray.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_RGB32))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    r0, c0 = base.origin

    def to_xy(row: float, col: float) -> QPointF:
        return QPointF((col - c0) * base.scale, (row - r0) * base.scale)

    if circle is not None and all(np.isfinite(circle)):
        painter.setPen(QPen(QColor(CIRCLE_COLOUR), 1.5))
        rad = circle[2] * base.scale
        painter.drawEllipse(to_xy(circle[0], circle[1]), rad, rad)
    if outline is not None and len(outline):
        pts = [to_xy(r, c) for r, c in np.asarray(outline, dtype=float).reshape(-1, 2)]
        painter.setPen(QPen(QColor(OUTLINE_COLOUR), 1.2))
        if is_ordered_contour(outline):
            painter.drawPolygon(QPolygonF(pts))
        else:
            painter.drawPoints(QPolygonF(pts))
    painter.end()
    return pixmap


class ThumbnailCache:
    """LRU cache of ROI thumbnails (without outlines) keyed by frame + ROI."""

    def __init__(self, capacity: int = THUMB_CACHE_SIZE) -> None:
        self._items: OrderedDict = OrderedDict()
        self._capacity = capacity
        self.loads = 0  # frames read from disk / video (for diagnostics and tests)

    def get(self, state, idx: int) -> _ThumbBase:
        key = (state.images[idx], tuple(state.gridx), tuple(state.gridy))
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        base = make_thumb_base(_frame_gray_u8(state, idx), state.gridx, state.gridy)
        self.loads += 1
        self._items[key] = base
        if len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return base

    def clear(self) -> None:
        self._items.clear()


# --------------------------------------------------------------------- #
#  Controller
# --------------------------------------------------------------------- #

def _bytes(array) -> bytes:
    return b"" if array is None else np.asarray(array).tobytes()


def results_signature(state) -> int:
    """Cheap fingerprint of everything the strip shows; equal = nothing to redo."""
    return hash((
        len(state.images), state.images[0] if state.images else "",
        tuple(state.gridx), tuple(state.gridy), state.detection_mode,
        _bytes(state.radius), _bytes(state.circle_fit_par), _bytes(state.region_count),
        _bytes(state.fit_confidence), tuple(state.fit_flags or ()),
    ))


class FilmstripController(QObject):
    """Keeps the strip in step with the results and the Review queue;
    debounced and cached.  *review_rows* returns the Review rows (none: the
    strip shows no frame "to check").  A child of the strip: its Qt
    connections end with it."""

    def __init__(self, strip: Filmstrip, get_state: Callable, jump_to: Callable[[int], None],
                 bus, chart=None, review_rows: ReviewRows | None = None) -> None:
        super().__init__(strip)
        self._strip = strip
        self._get_state = get_state
        self._review_rows = review_rows
        self._cache = ThumbnailCache()
        self._running = False
        self._built_for: int | None = None  # signature of the shown strip
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFRESH_DEBOUNCE_MS)
        self._timer.timeout.connect(self.refresh)

        strip.frame_clicked.connect(jump_to)
        strip.expanded_changed.connect(lambda expanded: expanded and self.refresh())
        bus.subscribe("run_state_changed", self._on_run_state)
        bus.subscribe("source_loaded", self._on_source_loaded)
        bus.subscribe("frame_changed", self._on_frame_changed)
        if chart is not None and hasattr(chart, "data_changed"):
            chart.data_changed.connect(self.schedule_refresh)

    @property
    def cache(self) -> ThumbnailCache:
        return self._cache

    def schedule_refresh(self, *_args) -> None:
        """Refresh after a short pause (never while a fit is writing results)."""
        if not self._running:
            self._timer.start()

    def refresh(self, force: bool = False) -> None:
        """Rebuild the strip now (skipped when results and queue did not change)."""
        self._timer.stop()
        if not self._strip.is_expanded():
            return
        state = self._get_state()
        self._strip.set_current(state.image_no)
        if not state.images or state.radius is None:
            self._built_for = None
            self._strip.set_thumbnails([])
            return
        rows = list(self._review_rows()) if self._review_rows is not None else []
        items = {item.frame: item for row in rows for item in row.items}
        candidates = to_check_frames(rows)
        signature = hash((results_signature(state), tuple(candidates), frozenset(items)))
        if signature == self._built_for and not force:
            return
        self._built_for = signature
        keys = choose_key_frames(state.radius, candidates)
        thumbs = [t for t in (self._thumbnail(state, k, items.get(k.frame)) for k in keys)
                  if t is not None]
        self._strip.set_thumbnails(thumbs)
        self._strip.set_current(state.image_no)

    def _thumbnail(self, state, key: KeyFrame, item: ReviewItem | None) -> Thumbnail | None:
        idx = key.frame
        try:
            base = self._cache.get(state, idx)
        except Exception as exc:  # unreadable frame: leave it out, keep the rest
            logger.debug("Filmstrip: no thumbnail for frame %d: %s", idx + 1, exc)
            return None
        radius = float(state.radius[idx])
        circle = None
        if state.circle_fit_par is not None and not (
                state.region_count is not None and state.region_count[idx] > 0):
            rc, cc = state.circle_fit_par[idx]
            circle = (float(rc), float(cc), radius)
        outline = state.circle_xy[idx] if state.circle_xy is not None else None
        conf = float(state.fit_confidence[idx]) if state.fit_confidence is not None else np.nan
        tooltip = key_frame_tooltip(idx, frame_file_label(state.images[idx]), key.kinds,
                                    radius, conf, item)
        return Thumbnail(idx, render_thumbnail(base, circle, outline),
                         f"{key.kinds[0]} · {idx + 1}", tooltip)

    def _on_run_state(self, running: bool) -> None:
        self._running = bool(running)
        if not running:
            self.refresh()

    def _on_source_loaded(self, _n: int) -> None:
        self._cache.clear()
        self.refresh(force=True)

    def _on_frame_changed(self, idx: int) -> None:
        self._strip.set_current(idx)
        self.schedule_refresh()


def install_filmstrip(window, app_controller) -> FilmstripController:
    """Add the key-frame strip above the R(t) chart and wire it up.

    Clicking a thumbnail jumps to that frame.  Frames "to check" are the
    first frames of the Review rows (``app_controller.review``, if installed);
    the strip follows the queue as it shrinks.  The controller is kept alive
    by the strip widget and also returned.
    """
    strip = Filmstrip()
    window.set_filmstrip(strip)
    review = getattr(app_controller, "review", None)
    controller = FilmstripController(
        strip,
        app_controller._get_state,
        app_controller.file_ctrl.on_frame_changed,
        app_controller.bus,
        chart=window.radius_chart,
        review_rows=review.panel.segments if review is not None else None,
    )
    if review is not None:
        review.count_changed.connect(controller.schedule_refresh)  # skips, accepts, fixes
    strip.controller = controller  # a child of the strip: lives as long as it
    controller.refresh()
    return controller
