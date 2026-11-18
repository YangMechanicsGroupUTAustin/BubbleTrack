"""Which recording results belong to: a fingerprint taken when the recording is loaded.

The same path or the same file names can hold another shot (every HPV-X2 shot has
256 frames, every export is frame_001..), so a session stores the fingerprint of the
recording its results were measured on, and refuses them on any other.
"""

from __future__ import annotations

import hashlib
import logging
import os

import numpy as np

from bubbletrack.model.hpvx2_dat import is_dat_file
from bubbletrack.model.image_io import imread_unicode
from bubbletrack.model.long_paths import os_path

logger = logging.getLogger(__name__)

_FILE_END_BYTES = 1 << 16  # bytes hashed at each end of a file
_TIFF_EXTS = (".tif", ".tiff")
# The scheme leads the value, so a session from a build that hashed otherwise is
# recognised (and treated as having no fingerprint) instead of refused.
PIXELS = "px1:"  # decoded first and last frames (grayscale TIFF folders, HPV-X2 .dat)
FILE_BYTES = "b1:"  # size and both ends of the files (videos, JPEG / PNG / BMP folders)
SCHEMES = (PIXELS, FILE_BYTES)


def recording_fingerprint(images, video_path: str = "", reader=None) -> str | None:
    """A short hash of the recording's first and last frames; None if unreadable.

    Grayscale TIFF frames and an HPV-X2 .dat are hashed by their decoded pixels, so
    files re-saved with other metadata, or a .dat header rewritten with the same
    frames, still match.  Other files are hashed by their bytes: a JPEG or a compressed
    video decodes differently on other computers, and OpenCV decodes some 16-bit
    planar RGB TIFFs differently on every read.
    """
    try:
        if video_path:
            if is_dat_file(video_path):
                last = reader.total_frames - 1
                return PIXELS + _hash_pixels([reader.read_frame(0), reader.read_frame(last)])
            return FILE_BYTES + _hash_file_ends([video_path])
        files = list(dict.fromkeys((images[0], images[-1])))  # a one-file folder once
        if all(os.path.splitext(str(f))[1].lower() in _TIFF_EXTS for f in files):
            frames = [imread_unicode(f) for f in files]
            if any(frame is None for frame in frames):
                return None
            if all(frame.ndim == 2 for frame in frames):  # grayscale: a stable decode
                return PIXELS + _hash_pixels(frames)
        return FILE_BYTES + _hash_file_ends(files)
    except Exception as exc:  # noqa: BLE001 - never stops a recording from loading
        logger.warning("No fingerprint for %s: %s", video_path or (images[0] if images else ""), exc)
        return None


def _hash_pixels(frames) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        pixels = np.ascontiguousarray(frame)
        digest.update(f"{pixels.shape}{pixels.dtype}".encode())
        digest.update(pixels.tobytes())
    return digest.hexdigest()[:16]


def _hash_file_ends(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with open(os_path(path), "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            digest.update(str(size).encode())
            digest.update(fh.read(_FILE_END_BYTES))
            if size > 2 * _FILE_END_BYTES:
                fh.seek(size - _FILE_END_BYTES)
                digest.update(fh.read(_FILE_END_BYTES))
    return digest.hexdigest()[:16]
