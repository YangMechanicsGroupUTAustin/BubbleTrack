"""One entry point for normalising a frame from a file or from memory.

Image sequences are identified by file path; video frames are decoded by
the caller (a ``VideoFrameReader`` is not picklable and must not be shared
across threads) and passed in as raw arrays.  Both go through exactly the
same normalisation, so results do not depend on where a frame came from.

Recordings (videos and raw HPV-X2 ``.dat`` files) are opened through
:func:`open_recording`, so every caller supports both formats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bubbletrack.model.hpvx2_dat import DAT_EXTS, DatFrameReader
from bubbletrack.model.image_io import (
    VIDEO_EXTS, VideoFrameReader, load_and_normalize, normalize_frame,
)

FrameSource = str | np.ndarray
FrameReader = VideoFrameReader | DatFrameReader

RECORDING_EXTS = frozenset(VIDEO_EXTS | DAT_EXTS)
RECORDING_DIALOG_FILTER = (
    "Recordings (*.avi *.mp4 *.mov *.mkv *.dat);;"
    "HPV-X2 raw (*.dat);;All Files (*)"
)


def is_recording_file(path: str | Path) -> bool:
    """True for a file that holds a whole frame sequence (video or ``.dat``)."""
    return Path(path).suffix.lower() in RECORDING_EXTS


def open_recording(path: str | Path) -> FrameReader:
    """A lazy frame reader for the video or HPV-X2 ``.dat`` file at *path*.

    Readers are not thread-safe: open one per thread.
    """
    if Path(path).suffix.lower() in DAT_EXTS:
        return DatFrameReader(str(path))
    return VideoFrameReader(str(path))


def normalize_source(
    source: FrameSource,
    sensitivity: float,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    *,
    gaussian_sigma: float = 0.0,
    clahe_clip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`load_and_normalize`, but *source* may also be a raw frame."""
    if isinstance(source, np.ndarray):
        return normalize_frame(
            source, sensitivity, gridx, gridy,
            gaussian_sigma=gaussian_sigma, clahe_clip=clahe_clip,
        )
    return load_and_normalize(
        source, sensitivity, gridx, gridy,
        gaussian_sigma=gaussian_sigma, clahe_clip=clahe_clip,
    )
