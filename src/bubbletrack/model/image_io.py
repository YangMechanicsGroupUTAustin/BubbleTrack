"""Image scanning, loading, normalisation and adaptive binarisation."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np

from bubbletrack.model.conventions import roi_to_slice

logger = logging.getLogger(__name__)

# cv::utils::logging::LOG_LEVEL_ERROR, for builds without cv2.utils.logging.
_CV_LOG_LEVEL_ERROR = 2


def quiet_opencv_logging(cv2_module=cv2) -> bool:
    """Lower OpenCV's native log level to ERROR.  Never raises.

    The lab camera writes TIFFs without a StripByteCounts tag, so libtiff
    prints a warning through OpenCV on every single frame read.  Errors are
    still shown.  An explicit ``OPENCV_LOG_LEVEL`` environment variable wins,
    so the warnings can be brought back for debugging.

    Parameters
    ----------
    cv2_module : module
        The OpenCV module (injectable for tests).

    Returns
    -------
    bool
        True if the level was changed.
    """
    if os.environ.get("OPENCV_LOG_LEVEL"):
        return False
    try:
        cv_logging = getattr(getattr(cv2_module, "utils", None), "logging", None)
        if cv_logging is not None and hasattr(cv_logging, "setLogLevel"):
            cv_logging.setLogLevel(cv_logging.LOG_LEVEL_ERROR)
            return True
        if hasattr(cv2_module, "setLogLevel"):
            cv2_module.setLogLevel(_CV_LOG_LEVEL_ERROR)
            return True
    except Exception as exc:  # never let log tuning break the import
        logger.debug("Could not lower the OpenCV log level: %s", exc)
    return False


quiet_opencv_logging()


class ImageLoadError(Exception):
    """Raised when an image file cannot be loaded or is corrupted."""


def imread_unicode(
    path: str | os.PathLike,
    flags: int = cv2.IMREAD_UNCHANGED,
) -> np.ndarray | None:
    """Read an image like ``cv2.imread``, but from any path, including non-ASCII ones.

    ``cv2.imread`` cannot open paths with characters outside the ANSI code
    page on Windows (CJK, ``µ``, ``Å``, emoji, ...) and silently returns None.
    Reading the bytes with numpy and decoding them in memory avoids the
    problem on every platform.

    Parameters
    ----------
    path : str or path-like
        Image file to read.
    flags : int
        ``cv2.IMREAD_*`` flags, as for ``cv2.imread``.  Defaults to
        ``IMREAD_UNCHANGED`` so the bit depth and channels are kept.

    Returns
    -------
    ndarray or None
        The decoded image, or None when the file is missing, empty, not a
        regular file, or cannot be decoded (the ``cv2.imread`` contract).
        For multi-page files only the first page is returned.
    """
    try:
        buf = np.fromfile(os.fspath(path), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        logger.debug("Cannot read file bytes %s: %s", path, exc)
        return None
    if buf.size == 0:
        return None
    try:
        return cv2.imdecode(buf, flags)
    except cv2.error as exc:
        logger.debug("Cannot decode %s: %s", path, exc)
        return None


# Supported extensions in priority order
_EXTENSIONS = (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp")

# Video file extensions
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def is_video_file(path: str) -> bool:
    """Return True if *path* has a recognised video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTS


def _natural_sort_key(s: str):
    """Sort key that handles embedded numbers naturally."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", s)
    ]


def scan_folder(folder: str) -> list[str]:
    """Return sorted image paths from *folder*.

    Scans for the first extension group that has files, in priority order
    (tiff > tif > png > jpg > bmp).  Files are natural-sorted by name.

    Validates that *folder* is an existing directory and skips symlinked
    files during iteration to avoid following potentially unsafe links.
    """
    p = Path(folder).resolve()
    if not p.is_dir():
        logger.warning("Not a directory: %s", folder)
        return []

    folder_str = str(p)
    for ext in _EXTENSIONS:
        files = []
        for f in p.iterdir():
            if f.is_symlink():
                logger.warning("Skipping symlink: %s", f)
                continue
            if f.name.lower().endswith(ext):
                files.append(str(f))
        if files:
            files.sort(key=lambda fp: _natural_sort_key(os.path.basename(fp)))
            return files
    return []


def detect_bit_depth(path: str) -> int:
    """Read the first image and return its max grey value (255 or 65535)."""
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 255
    if img.dtype == np.uint16:
        return 65535
    return 255


class VideoFrameReader:
    """Lazy frame reader for video files using OpenCV VideoCapture."""

    def __init__(self, path: str) -> None:
        self._path = path
        # Unlike cv2.imread, VideoCapture copes with non-ASCII paths on
        # Windows: the default FFmpeg backend opens UTF-8 paths (checked with
        # CJK, "µ", "Å" and emoji; see tests/test_unicode_paths.py).  Only the
        # Media Foundation backend (cv2.CAP_MSMF) fails on them, and it is
        # not selected for files by default.
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ImageLoadError(f"Cannot open video: {path}")
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def fps(self) -> float:
        return self._fps

    def read_frame(self, idx: int) -> np.ndarray:
        """Read a single greyscale frame by 0-based index.

        Raises ``ImageLoadError`` if the frame cannot be decoded.
        """
        if idx < 0 or idx >= self._total:
            raise ImageLoadError(
                f"Frame index {idx} out of range [0, {self._total}) "
                f"for {self._path}"
            )
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            raise ImageLoadError(f"Cannot read frame {idx} from {self._path}")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def close(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        self.close()


def normalize_frame(
    raw: np.ndarray,
    sensitivity: float,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    *,
    gaussian_sigma: float = 0.0,
    clahe_clip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise a raw greyscale frame, adaptive-threshold, and extract ROI.

    This is the core processing pipeline extracted from ``load_and_normalize``
    so that it can be reused with frames from any source (file, video, etc.).

    Parameters
    ----------
    raw : ndarray
        Raw greyscale image (uint8 or uint16, 2-D).
    sensitivity : float
        Adaptive threshold sensitivity in [0, 1].  Higher = more white.
    gridx : (row_start, row_end)
        1-indexed inclusive row bounds for the ROI.
    gridy : (col_start, col_end)
        1-indexed inclusive column bounds for the ROI.

    Returns
    -------
    cur_img : ndarray
        Full normalised greyscale image in [0, 1].
    cur_img_binary : ndarray
        Full binary image (bool).
    cur_img_roi : ndarray
        ROI slice of *cur_img*.
    cur_img_binary_roi : ndarray
        ROI slice of *cur_img_binary*.
    """
    # Convert to greyscale if needed
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    # Normalise to [0, 1]
    img_f = raw.astype(np.float64)
    img_min, img_max = img_f.min(), img_f.max()
    if img_max > img_min:
        cur_img = (img_f - img_min) / (img_max - img_min)
    else:
        cur_img = np.zeros_like(img_f)

    # Adaptive binarisation
    # MATLAB imbinarize 'adaptive' uses local mean with a sensitivity param.
    # sensitivity=0.5 is default (no offset).  Higher sensitivity -> more white.
    # block_size must be odd
    h, w = raw.shape[:2]
    block_size = int(np.ceil(min(h, w) / 16)) * 2 + 1
    block_size = max(block_size, 3)

    # For adaptive thresholding we need uint8 input
    if raw.dtype == np.uint16:
        raw_8 = (raw / 256).astype(np.uint8)
    else:
        raw_8 = raw.astype(np.uint8)

    # Pre-threshold filters
    if gaussian_sigma > 0:
        ksize = int(np.ceil(gaussian_sigma * 3)) * 2 + 1
        raw_8 = cv2.GaussianBlur(raw_8, (ksize, ksize), gaussian_sigma)

    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        raw_8 = clahe.apply(raw_8)

    # MATLAB: imbinarize(img, 'adaptive', 'Sensitivity', s)
    # OpenCV adaptiveThreshold: pixel > (mean - C) => 255
    # MATLAB offset maps: C = (sensitivity - 0.5) * range_per_block
    # Approximation: C ~ (sensitivity - 0.5) * 255
    c_val = (sensitivity - 0.5) * 255.0

    binary_8 = cv2.adaptiveThreshold(
        raw_8, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_val,  # OpenCV: pixel > (mean - C) => white; higher C => more white
    )
    cur_img_binary = binary_8 > 0

    # Extract ROI (1-indexed inclusive -> 0-indexed exclusive)
    rs, cs = roi_to_slice(gridx, gridy)
    cur_img_roi = cur_img[rs, cs]
    cur_img_binary_roi = cur_img_binary[rs, cs]

    return cur_img, cur_img_binary, cur_img_roi, cur_img_binary_roi


def load_and_normalize(
    image_path: str,
    sensitivity: float,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    *,
    gaussian_sigma: float = 0.0,
    clahe_clip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an image, normalise to [0,1], adaptive-threshold, and extract ROI.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    sensitivity : float
        Adaptive threshold sensitivity in [0, 1].  Higher = more white.
    gridx : (row_start, row_end)
        1-indexed inclusive row bounds for the ROI.
    gridy : (col_start, col_end)
        1-indexed inclusive column bounds for the ROI.

    Returns
    -------
    cur_img : ndarray
        Full normalised greyscale image in [0, 1].
    cur_img_binary : ndarray
        Full binary image (bool).
    cur_img_roi : ndarray
        ROI slice of *cur_img*.
    cur_img_binary_roi : ndarray
        ROI slice of *cur_img_binary*.
    """
    logger.debug("Loading image: %s", image_path)
    raw = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        logger.warning("Cannot read image: %s", image_path)
        raise ImageLoadError(
            f"Cannot read image (file may be missing or corrupted): {image_path}"
        )

    return normalize_frame(
        raw, sensitivity, gridx, gridy,
        gaussian_sigma=gaussian_sigma, clahe_clip=clahe_clip,
    )


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

def subtract_background(
    img: np.ndarray,
    method: str = "median",
    kernel_size: int = 51,
) -> np.ndarray:
    """Subtract estimated background from image.

    Parameters
    ----------
    img : ndarray
        uint8 or uint16 grayscale image.
    method : ``"median"`` or ``"gaussian"``
        Background estimation technique.
    kernel_size : int
        Blur kernel size (must be odd; if even it is bumped to the next odd).

    Returns
    -------
    Background-subtracted image (same dtype as input).

    Raises
    ------
    ValueError
        If *method* is not ``"median"`` or ``"gaussian"``.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    if method == "median":
        bg = cv2.medianBlur(img, kernel_size)
    elif method == "gaussian":
        bg = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    else:
        raise ValueError(f"Unknown background method: {method}")

    return cv2.subtract(img, bg)


def denoise_nlm(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Non-local means denoising (better than Gaussian for structured noise).

    Parameters
    ----------
    img : ndarray
        uint8 grayscale image.
    h : float
        Filter strength.  Higher removes more noise but also detail.

    Returns
    -------
    Denoised image (uint8).
    """
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
