"""Image scanning, loading, normalisation and adaptive binarisation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from bubbletrack.model.constants import UINT16_DIVISOR, UINT16_PACKED_DEPTHS
from bubbletrack.model.conventions import roi_to_slice
from bubbletrack.model.folder_scan import (  # noqa: F401  (re-exported)
    ScanResult,
    scan_folder_report,
    tiff_page_count,
)
from bubbletrack.model.long_paths import MAX_PATH, os_path, too_long

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


def read_failure(path: str | os.PathLike) -> str:
    """Why the image at *path* could not be read, in plain words (with the path).

    A path beyond Windows' MAX_PATH is named as the likely cause rather than a
    damaged file: plain Win32 calls cannot open it.
    """
    text = os.fspath(path)
    if too_long(text):
        return (f"Cannot read image: the path is too long ({len(os.path.abspath(text))} "
                f"characters; Windows allows {MAX_PATH}). Move the data to a shorter "
                f"folder: {text}")
    if not os.path.exists(os_path(text)):
        return f"Cannot read image: file not found: {text}"
    return f"Cannot read image (the file is damaged or not an image): {text}"


def imread_unicode(
    path: str | os.PathLike,
    flags: int = cv2.IMREAD_UNCHANGED,
) -> np.ndarray | None:
    """Read an image like ``cv2.imread``, but from any path, including non-ASCII ones.

    ``cv2.imread`` cannot open paths with characters outside the ANSI code
    page on Windows (CJK, ``µ``, ``Å``, emoji, ...) and silently returns None.
    Reading the bytes with numpy and decoding them in memory avoids the
    problem on every platform.  Paths beyond Windows' MAX_PATH are read
    through the extended-length prefix (see ``model.long_paths``).

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
        buf = np.fromfile(os_path(path), dtype=np.uint8)
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


# Video file extensions
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def is_video_file(path: str) -> bool:
    """Return True if *path* has a recognised video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTS


def scan_folder(folder: str) -> list[str]:
    """Return the natural-sorted frame paths of *folder*.

    Thin wrapper around :func:`~bubbletrack.model.folder_scan.scan_folder_report`
    for callers that only need the paths: one extension group is loaded
    (tif/tiff > png > jpg/jpeg > bmp, any letter case), and hidden files and
    symlinks are skipped.  Use ``scan_folder_report`` to also learn which
    files were ignored or are multi-page TIFFs (this wrapper reads no file
    headers, so existing callers pay nothing extra).
    """
    return list(scan_folder_report(folder, check_pages=False).images)


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
        self._cap = cv2.VideoCapture(os_path(path))  # FFmpeg takes the long-path prefix
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


_CV_COLOR_DTYPES = (np.uint8, np.uint16, np.float32)  # types cv2.cvtColor accepts
_BGR_TO_GRAY_WEIGHTS = np.array([0.114, 0.587, 0.299])  # same weights as cv2


def to_grayscale(raw: np.ndarray) -> np.ndarray:
    """Return the single-channel version of *raw*, keeping its dtype.

    Parameters
    ----------
    raw : ndarray
        2-D grey image (returned as is), or 3-D with 1 (grey), 2 (grey +
        alpha), 3 (BGR) or 4 (BGRA) channels, as ``cv2.imdecode`` returns
        them.  Alpha is ignored.

    Returns
    -------
    ndarray
        2-D image of the same dtype.

    Raises
    ------
    ImageLoadError
        For any other layout.
    """
    if raw.ndim == 2:
        return raw
    channels = raw.shape[2] if raw.ndim == 3 else 0
    if channels in (1, 2):
        return raw[:, :, 0]
    if channels in (3, 4):
        if raw.dtype in _CV_COLOR_DTYPES:
            code = cv2.COLOR_BGR2GRAY if channels == 3 else cv2.COLOR_BGRA2GRAY
            return cv2.cvtColor(raw, code)
        gray = raw[:, :, :3].astype(np.float64) @ _BGR_TO_GRAY_WEIGHTS
        return gray.astype(raw.dtype)
    raise ImageLoadError(f"Unsupported image layout {raw.shape} ({raw.dtype})")


def _replace_non_finite(values: np.ndarray) -> np.ndarray:
    """Return a copy of float *values* with NaN/-inf set to the finite minimum
    and +inf to the finite maximum (all zeros if nothing is finite).
    *values* itself is returned when it is already all finite."""
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    lo, hi = values[finite].min(), values[finite].max()
    return np.nan_to_num(values, nan=lo, posinf=hi, neginf=lo)


def _counts_divisor(peak: float) -> int:
    """Divisor that maps camera counts with this peak to 8 bits.

    10-bit (peak <= 1023) -> 4, 12-bit (<= 4095) -> 16, otherwise 16-bit -> 256
    (see ``UINT16_PACKED_DEPTHS``).
    """
    for max_value, divisor in UINT16_PACKED_DEPTHS:
        if peak <= max_value:
            return divisor
    return UINT16_DIVISOR


def to_threshold_uint8(gray: np.ndarray) -> np.ndarray:
    """Convert a 2-D grey frame to the uint8 image used for adaptive thresholding.

    The adaptive-threshold offset is in absolute 8-bit grey levels, so camera
    data keeps its absolute brightness: a fixed divisor, never a per-frame
    contrast stretch, which would change the effective threshold from frame
    to frame as the bubble changes size.

    * ``uint8``: unchanged.
    * ``uint16`` with frame max > 4095 (real 16-bit data): ``gray / 256``,
      the original conversion, unchanged bit for bit.
    * ``uint16`` with max <= 4095 (12-bit) or <= 1023 (10-bit data stored in
      uint16): divided by 16 or 4 instead, so the frame spans 256 grey
      levels rather than 16 or 4.  The choice is made per frame.
    * floats whose finite values all lie in [0, 1]: normalised intensities,
      multiplied by 255.
    * other types (floats, signed or 32-bit integers) whose values lie in
      [0, 65535]: camera counts (e.g. an ImageJ "32-bit" export of 16-bit
      data), converted exactly like uint16 data with the same peak.
    * anything else (negative values, or beyond 65535): min-max scaled to
      [0, 255].

    NaN and inf are replaced by the finite min/max first.  Results are
    truncated towards zero, like the integer paths always were.  The input
    is never modified.
    """
    if gray.dtype == np.uint8:
        return gray.astype(np.uint8)
    if gray.dtype == np.uint16:
        return (gray / _counts_divisor(int(gray.max()))).astype(np.uint8)

    is_float = np.issubdtype(gray.dtype, np.floating)
    values = _replace_non_finite(gray.astype(np.float64)) if is_float else gray.astype(np.float64)
    lo, hi = float(values.min()), float(values.max())
    if is_float and lo >= 0.0 and hi <= 1.0:
        scaled = values * 255.0
    elif lo >= 0.0 and hi <= np.iinfo(np.uint16).max:
        scaled = values / _counts_divisor(hi)
    elif hi > lo:
        scaled = (values - lo) / (hi - lo) * 255.0
    else:
        scaled = np.zeros_like(values)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


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
        Raw frame: 2-D grey, or 3-D grey / grey+alpha / BGR / BGRA, of any
        integer or float type (see :func:`to_grayscale` and
        :func:`to_threshold_uint8`).
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
    raw = to_grayscale(raw)
    if np.issubdtype(raw.dtype, np.floating):
        raw = _replace_non_finite(raw)

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
    raw_8 = to_threshold_uint8(raw)

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
        raise ImageLoadError(read_failure(image_path))

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
