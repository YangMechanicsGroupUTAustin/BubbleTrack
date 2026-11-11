"""Centralised constants for the BubbleTrack model layer.

Every magic number used in the detection, fitting, and export pipelines
lives here so that values are documented, discoverable, and easy to tune.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
#  Detection pipeline
# ------------------------------------------------------------------ #

MAX_AXIS_RATIO: float = 2.2
"""Reject blobs whose major axis exceeds this multiple of the image short side."""

MAX_ECCENTRICITY: float = 1.6
"""Reject blobs whose major/minor axis ratio exceeds this value."""

MIN_HOLE_AREA: int = 40
"""Minimum hole area (px) inside the bubble blob to keep (smaller holes are filled)."""

# ------------------------------------------------------------------ #
#  Bit depth (frame -> 8-bit image for adaptive thresholding)
# ------------------------------------------------------------------ #

UINT16_PACKED_DEPTHS: tuple[tuple[int, int], ...] = (
    (1023, 4),   # 10-bit data stored in uint16
    (4095, 16),  # 12-bit data stored in uint16
)
"""``(max grey value, divisor)`` for sub-16-bit data in uint16 frames, smallest first.

A uint16 frame whose maximum is at most the bound is divided by the divisor
to reach 8 bits; a frame brighter than every bound is real 16-bit data and
is divided by ``UINT16_DIVISOR``."""

UINT16_DIVISOR: int = 256
"""Divisor that maps 16-bit grey values to 8 bits (the original behaviour)."""

# ------------------------------------------------------------------ #
#  Removing factor (slider -> area mapping)
# ------------------------------------------------------------------ #

RF_MIN_AREA: int = 10
"""Minimum connected-component area at the low end of the slider."""

RF_MAX_AREA_FRACTION: float = 0.5
"""Fraction of ROI area used as upper bound for the removing-factor slider."""

# ------------------------------------------------------------------ #
#  Circle fitting (Taubin algorithm)
# ------------------------------------------------------------------ #

TAUBIN_EPSILON: float = 1e-12
"""Convergence tolerance for Newton's method in the Taubin fitter."""

TAUBIN_MAX_ITER: int = 20
"""Maximum Newton iterations for the Taubin fitter."""

TAUBIN_MIN_POINTS: int = 3
"""Minimum number of boundary points required for circle fitting."""

MIN_POINT_DISTANCE_PX: float = 5.0
"""Minimum Euclidean distance (px) between manual edge points.

Points closer than this threshold are rejected to prevent near-duplicate
entries that destabilize the Taubin circle fitter."""

# ------------------------------------------------------------------ #
#  Export defaults
# ------------------------------------------------------------------ #

DEFAULT_FPS: float = 1_000_000.0
"""Default frames-per-second for high-speed camera recordings."""

DEFAULT_UM2PX: float = 3.2
"""Default micrometres-per-pixel scale factor."""

DEFAULT_RMAX_FIT_LENGTH: int = 11
"""Default window length for quadratic Rmax fitting (must be odd)."""

RMAX_OUTLIER_TOLERANCE: float = 0.10
"""Warn when the Rmax frame exceeds its neighbours' median by more than this fraction.

Rmax is the global argmax, so one spurious detection can move t=0; the
export flags (but does not drop) such a peak."""

RMAX_OUTLIER_NEIGHBOURS: int = 2
"""Valid frames taken on each side of the Rmax frame for the outlier median."""

# ------------------------------------------------------------------ #
#  State defaults
# ------------------------------------------------------------------ #

DEFAULT_REMOVING_FACTOR: int = 90
"""Default removing-factor slider position (0-100)."""

# ------------------------------------------------------------------ #
#  UI debounce / throttle
# ------------------------------------------------------------------ #

DISPLAY_DEBOUNCE_MS: int = 50
"""Debounce interval (ms) for the frame-display timer."""

PREVIEW_DEBOUNCE_MS: int = 50
"""Debounce interval (ms) for the detection-preview timer."""

AUTO_DISPLAY_THROTTLE_MS: int = 150
"""Minimum interval (ms) between display refreshes during automatic fitting."""

# ------------------------------------------------------------------ #
#  Fit quality scoring
# ------------------------------------------------------------------ #

QUALITY_TARGET_EDGE_DENSITY: float = 0.6
"""Target edge points per pixel of circumference (~1 point every 1.7 px)."""

QUALITY_MAX_RMS_RATIO: float = 0.15
"""RMS residual / radius ratio above which the fit is considered poor."""

QUALITY_N_SECTORS: int = 8
"""Number of angular sectors for coverage scoring."""

QUALITY_MIN_RADIUS_FRACTION: float = 0.03
"""Minimum plausible radius as fraction of ROI short side."""

QUALITY_MAX_RADIUS_FRACTION: float = 0.85
"""Maximum plausible radius as fraction of ROI short side."""

QUALITY_GOOD_THRESHOLD: float = 0.65
"""Fit quality score >= this is considered reliable (green marker)."""

QUALITY_WARN_THRESHOLD: float = 0.35
"""Fit quality score >= this but < GOOD is marginal (amber). Below = unreliable (red)."""

FIT_ACCEPT_MIN_SECTOR_COVERAGE: float = 0.625
"""Minimum angular coverage for accepting a full in-ROI automatic fit.

0.625 means at least 5 of 8 angular sectors contain boundary points.  This
rejects half-circle fits whose radius is mostly extrapolated.
"""

FIT_ACCEPT_PARTIAL_MIN_SECTOR_COVERAGE: float = 0.375
"""Coverage threshold when the user explicitly marks a bubble crossing an ROI edge."""

FIT_ACCEPT_MIN_EDGE_DENSITY: float = 0.12
"""Minimum edge points per pixel of circumference for accepting an automatic fit."""

# ------------------------------------------------------------------ #
#  Auto-tune grid search
# ------------------------------------------------------------------ #

AUTOTUNE_THR_COARSE: tuple[float, ...] = (
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
)
"""Coarse threshold grid for auto-tune (9 points, step 0.10)."""

AUTOTUNE_RF_COARSE: tuple[int, ...] = (20, 35, 50, 65, 75, 85, 92, 98)
"""Coarse removing-factor grid for auto-tune (8 points)."""

AUTOTUNE_THR_FINE_STEP: float = 0.02
"""Fine-search threshold step around best coarse value."""

AUTOTUNE_RF_FINE_STEP: int = 3
"""Fine-search removing-factor step around best coarse value."""

AUTOTUNE_TOP_K: int = 5
"""Number of top coarse candidates to refine."""
