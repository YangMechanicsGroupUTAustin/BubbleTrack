"""The recording's black level: the darkest level its frames reach (edge-1.7.0).

Stage A divides each frame by the bubble-free reference, which holds only when a pixel
value of 0 means no light.  Exported image sequences often carry a constant offset: the
HPV-X TIFF16 export adds 655 counts to every pixel of the raw ``.dat`` data.  Left in, the
offset compresses the bubble's contrast, most where the speckled reference is dark, and frames
the raw data shows fall below the confidence gate (Jin 2026-11-16 14:37:03, big-bubble frames
27/28 shown from the .dat, 14/28 from the TIFF export).

The darkest level of the recording is taken as black and subtracted from every frame and from
the reference (``EdgeParams.black_level``), as an image viewer's auto-levels do, so the result
no longer depends on such an offset.  A frame's level is a low percentile of its values below
full scale (saturated pixels say nothing about black); the recording's level is the lowest over
an evenly spaced sample of frames: the frames with the largest, darkest bubble set it.  Exact
zeros are left out when nothing else in the frame is near zero (a burned-in label on an offset
image); data whose black is truly 0 reaches it through the low values and keeps them.

Study: ``tasks/studies/offset_invariance_2026-11-24`` (LIC bench + 2026-11-16 set): the share
of frames with the same result after a +1 % offset rose from 89.7 to 97.2 %.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from bubbletrack.model.edge.config import DEFAULT_CONFIG, BlackConfig, EdgeConfig
from bubbletrack.model.edge.normalize import GrayFrame, to_gray_frame

logger = logging.getLogger(__name__)

__all__ = ["frame_dark_level", "gray_pair", "recording_black_level", "sample_frames", "with_black"]
_EPS = 1e-6


def with_black(frame: GrayFrame, black: float) -> GrayFrame:
    """*frame* with the recording's black level *black* subtracted (clipped at 0).

    The full scale moves with the values, so saturation is judged as before.  *frame* is
    returned unchanged for a black level of 0.
    """
    if not black > 0:
        return frame
    values = np.maximum(frame.values - np.float32(black), np.float32(0.0))
    return GrayFrame(values=values, full_scale=max(frame.full_scale - float(black), _EPS),
                     non_finite=frame.non_finite)


def gray_pair(raw: np.ndarray, reference: np.ndarray | None,
              black: float) -> tuple[GrayFrame, GrayFrame | None]:
    """A frame and its reference as stage A takes them: grey, the black level removed.

    Raises ValueError for a reference of another size (or what ``to_gray_frame`` rejects).
    """
    frame = with_black(to_gray_frame(raw), black)
    ref = with_black(to_gray_frame(reference), black) if reference is not None else None
    if ref is not None and ref.values.shape != frame.values.shape:
        raise ValueError(f"reference shape {ref.values.shape} differs from frame shape {frame.values.shape}")
    return frame, ref


def frame_dark_level(raw: np.ndarray, cfg: BlackConfig = DEFAULT_CONFIG.black) -> float | None:
    """The dark level of one frame (any dtype or layout ``to_gray_frame`` accepts).

    The ``cfg.percentile`` % point of the finite values below full scale; exact zeros are
    dropped first when no other value lies in ``(0, cfg.label_gap * full scale]``.  None when
    the frame has no such value (all saturated).
    """
    gray = to_gray_frame(raw)
    values = gray.values[~gray.non_finite]
    values = values[values < gray.full_scale]
    zero = values == 0
    if zero.any():
        rest = values[~zero]
        if rest.size and float(rest.min()) > cfg.label_gap * gray.full_scale:
            values = rest
    if values.size == 0:
        return None
    return float(np.percentile(values, cfg.percentile))


def sample_frames(n_frames: int, cap: int) -> list[int]:
    """At most *cap* frame indices, evenly spaced over ``range(n_frames)`` (first and last in)."""
    n = int(n_frames)
    if n <= 0:
        return []
    if n <= cap:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, int(cap))})


def recording_black_level(read_raw: Callable[[int], np.ndarray], n_frames: int, *,
                          config: EdgeConfig = DEFAULT_CONFIG) -> float:
    """The black level of a recording: the lowest :func:`frame_dark_level` of its sampled frames.

    Parameters
    ----------
    read_raw : callable
        ``read_raw(i)`` returns frame *i* (0-based) at its native bit depth.  A frame that
        raises or is malformed is skipped.
    n_frames : int
        Number of frames in the recording.

    Returns
    -------
    float
        The level in the frames' own units; 0.0 when no frame could be read.

    Raises
    ------
    ValueError
        If *n_frames* is negative.
    """
    if int(n_frames) < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    cfg = config.black
    levels = []
    for i in sample_frames(n_frames, cfg.sample_frames):
        try:
            level = frame_dark_level(read_raw(i), cfg)
        except Exception as exc:  # noqa: BLE001 - one unreadable frame must not stop the estimate
            logger.warning("Black level: frame %d unusable (%s)", i, exc)
            continue
        if level is not None:
            levels.append(level)
    return min(levels) if levels else 0.0
