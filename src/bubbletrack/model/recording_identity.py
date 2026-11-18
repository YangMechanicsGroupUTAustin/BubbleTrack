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

_VIDEO_BYTES = 1 << 16  # bytes hashed at each end of a compressed video


def recording_fingerprint(images, video_path: str = "", reader=None) -> str | None:
    """A short hash of the recording's first and last frames; None if unreadable.

    A folder or an HPV-X2 .dat is lossless: the decoded pixels are hashed, so files
    re-saved with other metadata, or a .dat header rewritten with the same frames,
    still match.  A compressed video is hashed by its size and the bytes of its start
    and end, since decoders differ between computers.
    """
    digest = hashlib.sha256()
    try:
        if video_path and not is_dat_file(video_path):
            _hash_file_ends(digest, video_path)
        else:
            if video_path:
                frames = [reader.read_frame(0), reader.read_frame(reader.total_frames - 1)]
            else:
                frames = [imread_unicode(images[0]), imread_unicode(images[-1])]
            for frame in frames:
                if frame is None:
                    return None
                pixels = np.ascontiguousarray(frame)
                digest.update(f"{pixels.shape}{pixels.dtype}".encode())
                digest.update(pixels.tobytes())
    except Exception as exc:  # noqa: BLE001 - never stops a recording from loading
        logger.warning("No fingerprint for %s: %s", video_path or (images[0] if images else ""), exc)
        return None
    return digest.hexdigest()[:16]


def _hash_file_ends(digest, path: str) -> None:
    with open(os_path(path), "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        digest.update(str(size).encode())
        digest.update(fh.read(_VIDEO_BYTES))
        if size > 2 * _VIDEO_BYTES:
            fh.seek(size - _VIDEO_BYTES)
            digest.update(fh.read(_VIDEO_BYTES))
