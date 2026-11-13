"""Does a frame look like the reference frame?  Review's "no bubble here" test.

Review recommends "No bubble" only for a frame that has no result, lies
outside the bubble's frames and looks like the bubble-free reference frame;
in doubt it recommends Fix (a wrong "no bubble" throws a radius away).

The test compares the raw frame with the reference, smoothed, divided and
normalised by the median ratio (so flicker of the whole frame does not
count).  A frame is background-like when no dark region of at least
``MIN_DARK_AREA`` pixels is ``DARK_DROP`` darker than the reference.  Unlike
the engine's view (flattened by a background surface) this keeps a wide,
soft bubble visible (blood), and a bright laser flash is not a dark bubble.
A bright bubble (polarity "bright") is mirrored first, as the engine does.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable

import cv2
import numpy as np

from bubbletrack.controller import edge_support
from bubbletrack.controller.display_mixin import current_source
from bubbletrack.model import edge_pipeline
from bubbletrack.model.conventions import clamp_roi, roi_to_slice
from bubbletrack.model.edge.normalize import to_gray_frame
from bubbletrack.model.suggestion import first_result, shown_suggestion

logger = logging.getLogger(__name__)

SMOOTH_SIGMA = 3.0  # px: averages speckle and sensor noise away
DARK_DROP = 0.2  # a bubble is at least this much darker than the reference
MIN_DARK_AREA = 30  # px: a smaller dark patch is debris or noise, not a bubble
MAX_BEFORE_SCAN = 200  # frames before the first result that are tested at most
_EPS = 1e-6


def looks_like_reference(frame: np.ndarray, reference: np.ndarray, *,
                         polarity: str = "dark") -> bool:
    """True if *frame* holds no bubble-sized region darker than *reference*.

    Both are 2-D grey ROIs of the same shape (any numeric dtype).
    """
    frame = np.asarray(frame, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if frame.shape != reference.shape or frame.ndim != 2:
        raise ValueError(f"frame {frame.shape} and reference {reference.shape} must be "
                         "2-D and of the same shape")
    ratio = (cv2.GaussianBlur(frame, (0, 0), SMOOTH_SIGMA)
             / np.maximum(cv2.GaussianBlur(reference, (0, 0), SMOOTH_SIGMA), _EPS))
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio /= max(float(np.median(ratio)), _EPS)
    if polarity == "bright":
        ratio = 2.0 - ratio
    return _largest_region(ratio < 1.0 - DARK_DROP) < MIN_DARK_AREA


def _largest_region(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def before_bubble(state, looks_empty: Callable[[int], bool | None],
                  limit: int = MAX_BEFORE_SCAN) -> frozenset[int]:
    """Frames before the bubble's first result that look like the reference.

    Only scored, unlocked frames without a result or a shown guess are
    tested (at most the first *limit* of them): the frames Review would
    otherwise list as "No result".
    """
    first = first_result(state)
    if first is None:
        return frozenset()
    n = min(first, limit)
    conf = state.fit_confidence
    locked = state.fit_locked
    candidates = [f for f in range(n)
                  if conf is not None and np.isfinite(conf[f])
                  and not (locked is not None and locked[f])
                  and shown_suggestion(state, f) is None]
    return frozenset(f for f in candidates if looks_empty(f) is True)


def source_key(state) -> Hashable:
    """What a background answer depends on: source, reference, ROI, polarity."""
    return (state.folder_path, state.video_path, len(state.images or ()),
            edge_support.reference_index(state), tuple(state.gridx), tuple(state.gridy),
            state.edge_polarity)


def roi_gray(state, idx: int) -> np.ndarray:
    """Frame *idx*'s ROI as grey values (raw, not display-normalised)."""
    return _roi(state, edge_pipeline.read_raw(current_source(state, idx)))


def reference_gray(state) -> np.ndarray | None:
    """The reference frame's ROI as grey values; None without a reference."""
    raw = edge_support.reference_raw(state)
    return None if raw is None else _roi(state, raw)


def _roi(state, raw: np.ndarray) -> np.ndarray:
    values = to_gray_frame(raw).values
    gx, gy = clamp_roi(tuple(state.gridx), tuple(state.gridy), *values.shape[:2])
    return values[roi_to_slice(gx, gy)]


class BackgroundCheck:
    """``looks_like_reference`` per frame, cached for one source.

    *key* identifies everything the answer depends on (source, reference,
    ROI, polarity); a new key starts over.
    """

    def __init__(self) -> None:
        self._key: Hashable | None = None
        self._answers: dict[int, bool | None] = {}
        self._reference: np.ndarray | None = None

    def for_state(self, state, frame: int) -> bool | None:
        """``looks_empty`` for *frame* of the loaded source (None: cannot tell)."""
        if not state.images:
            return None
        try:
            key = source_key(state)
        except Exception as exc:  # no readable source: cannot tell
            logger.warning("Background test unavailable: %s", exc)
            return None
        return self.looks_empty(key, frame, lambda i: roi_gray(state, i),
                                lambda: reference_gray(state), polarity=state.edge_polarity)

    def looks_empty(self, key: Hashable, frame: int, read: Callable[[int], np.ndarray],
                    reference: Callable[[], np.ndarray | None], *,
                    polarity: str = "dark") -> bool | None:
        """True/False for *frame*; None when it cannot tell (no reference,
        unreadable frame): then Review treats the frame as maybe a bubble."""
        if key != self._key:
            self._key, self._answers, self._reference = key, {}, None
        if frame not in self._answers:
            self._answers[frame] = self._answer(frame, read, reference, polarity)
        return self._answers[frame]

    def _answer(self, frame: int, read, reference, polarity: str) -> bool | None:
        try:
            if self._reference is None:
                self._reference = reference()
            if self._reference is None:
                return None
            return looks_like_reference(read(frame), self._reference, polarity=polarity)
        except Exception as exc:  # an unreadable frame must not break Review
            logger.warning("Background test of frame %d unavailable: %s", frame + 1, exc)
            return None
