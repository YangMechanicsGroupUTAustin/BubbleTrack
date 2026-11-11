"""One entry point for normalising a frame from a file or from memory.

Image sequences are identified by file path; video frames are decoded by
the caller (a ``VideoFrameReader`` is not picklable and must not be shared
across threads) and passed in as raw arrays.  Both go through exactly the
same normalisation, so results do not depend on where a frame came from.
"""

from __future__ import annotations

import numpy as np

from bubbletrack.model.image_io import load_and_normalize, normalize_frame

FrameSource = str | np.ndarray


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
