"""Frozen application state dataclass with immutable update helper.

AppState is a frozen dataclass: field reassignment raises FrozenInstanceError.
All state transitions go through ``update_state()`` which returns a new instance
via ``dataclasses.replace()``, making state changes explicit and trackable.

Pragmatic compromise: NumPy array *contents* remain mutable (element-level
assignment like ``state.radius[idx] = value`` still works) because copying
large arrays on every frame would be wasteful.  The key benefit is that all
*field-level* reassignments are funneled through ``update_state()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from bubbletrack.model.provenance import FitSource

# Per-frame shape metrics shared by the engine, export, review and charts.
FIT_METRIC_KEYS: tuple[str, ...] = (
    "r_circle",          # robust circle-fit radius (px)
    "area_px",           # area of the bubble region (px^2)
    "circularity",       # 4*pi*A / P^2
    "solidity",          # A / convex-hull area
    "fragment_count",    # other dark regions near the main body
    "fragment_area_px",  # their total area (px^2)
    "shoulder_width",    # outer extent minus steep edge (px)
)

from bubbletrack.model.constants import (
    DEFAULT_FPS,
    DEFAULT_REMOVING_FACTOR,
    DEFAULT_RMAX_FIT_LENGTH,
    DEFAULT_UM2PX,
)


@dataclass(frozen=True)
class AppState:
    """Immutable application state shared between controller and views.

    Use ``update_state(state, field=value)`` to create a modified copy.
    """

    # Image file list (tuple for true immutability)
    images: tuple[str, ...] = ()
    folder_path: str = ""
    image_no: int = 0  # 0-indexed (UI displays +1)

    # Threshold / binarisation
    img_thr: float = 0.5  # adaptive sensitivity [0, 1]

    # ROI bounds (1-indexed inclusive, matching MATLAB convention)
    gridx: tuple[int, int] = (1, 1)
    gridy: tuple[int, int] = (1, 1)

    # Removing factor slider 0-100
    removing_factor: int = DEFAULT_REMOVING_FACTOR

    # Bubble-crosses-edge flags [Top, Right, Down, Left] (tuple for immutability)
    bubble_cross_edges: tuple[bool, ...] = (False, False, False, False)

    # Morphological closing radius (0 = skip)
    removing_obj_radius: int = 0

    # Advanced filters (0 = disabled)
    gaussian_sigma: float = 0.0
    clahe_clip: float = 0.0
    closing_radius: int = 0
    opening_radius: int = 0
    detection_mode: str = "intensity"  # "intensity" | "gradient" | "area" | "edge"
    area_max_regions: int = 1
    gradient_boundary: float = 0.0  # gradient mode ring index: 0/0.5/1 -> 1st/2nd/3rd
    temporal_refine: bool = False  # use neighbouring frames to repair weak fits

    # Edge engine (detection_mode == "edge")
    edge_definition: str = "steep"  # "steep" (50% of the steep edge) | "outer" extent
    edge_polarity: str = "dark"  # "dark" bubble on bright background | "bright"
    reference_frame: int | None = None  # bubble-free frame index; None = auto-select
    edge_seed: tuple[float, float] | None = None  # (row, col) 1-based, picks the bubble
    # The radius definition ("Bubble may be non-round" in Advanced): False = the robust
    # circle through the wall (default); True = the area-equivalent radius of the outline.
    bubble_non_round: bool = False

    # ML assist (a helper of Auto, ``model.teach``), per recording: cleared when
    # another source is loaded, kept in sessions.  On: Fit All measures with it.
    ml_assist: bool = False
    ml_examples: object = None  # MLExamples | None: strokes (+ verified circles at the last sync)

    # Visual binary polarity swap (session-only, NOT persisted across launches).
    # When True, the binary panel displays ``~processed`` instead of
    # ``processed``.  This is a pure post-detection display toggle: detection,
    # circle fitting, and radius results are unaffected — only the rendered
    # binary image is bit-flipped.
    invert_mask: bool = False

    # Results arrays — allocated after folder is loaded.
    # Typed as object for frozen dataclass compatibility with NumPy.
    # Contents are mutable (element-level assignment), but field reassignment
    # is blocked by frozen=True.
    radius: object = None          # np.ndarray (N,) float, -1 = unprocessed
    circle_fit_par: object = None  # np.ndarray (N, 2) [row_c, col_c]
    circle_xy: object = None       # list of (M, 2) arrays | None
    region_count: object = None     # np.ndarray (N,) number of selected regions
    fit_source: object = None       # np.ndarray (N,) int8 FitSource codes
    fit_confidence: object = None   # np.ndarray (N,) float, NaN = not scored (Edge engine)
    fit_metrics: object = None      # dict[str, np.ndarray (N,)] keyed by FIT_METRIC_KEYS, NaN = n/a
    fit_flags: object = None        # list[frozenset[str]] per frame (engine/sequence flags)
    fit_locked: object = None       # np.ndarray (N,) bool, accepted in review: never overwritten
    fit_suggestion: object = None   # list[Suggestion | None]: Auto's rejected guess (display only)

    # Display parameters
    img_grayscale_max: int = 65535
    um2px: float = DEFAULT_UM2PX
    fps: float = DEFAULT_FPS
    rmax_fit_length: int = DEFAULT_RMAX_FIT_LENGTH

    # Export
    save_path: str = ""

    # Video mode (mutually exclusive with folder mode)
    video_path: str = ""
    video_reader: object = None  # VideoFrameReader | None

    # Runtime state
    realtime_play: bool = False
    cur_img: object = None          # np.ndarray | None
    cur_img_binary_roi: object = None  # np.ndarray | None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def total_frames(self) -> int:
        return len(self.images)

    @classmethod
    def create_empty(cls) -> AppState:
        """Create a fresh empty state with all defaults."""
        return cls()

    def with_results_initialized(self, n: int) -> AppState:
        """Return a new state with result arrays allocated for *n* frames."""
        return update_state(
            self,
            radius=np.full(n, -1.0),
            circle_fit_par=np.full((n, 2), np.nan),
            circle_xy=[None] * n,
            region_count=np.zeros(n, dtype=int),
            fit_source=np.zeros(n, dtype=np.int8),
            fit_confidence=np.full(n, np.nan),
            fit_metrics={key: np.full(n, np.nan) for key in FIT_METRIC_KEYS},
            fit_flags=[frozenset()] * n,
            fit_locked=np.zeros(n, dtype=bool),
            fit_suggestion=[None] * n,
        )


def clear_frame_result(state: AppState, idx: int) -> None:
    """Erase frame *idx*'s result in place: unfitted, unscored and unlocked.

    Uses the same empty values as :meth:`AppState.with_results_initialized`
    (radius -1, centre NaN, source NONE).  A lock protects a result, so an
    erased frame is unlocked and the next Fit All fits it again.  Auto's
    guess for the frame (``fit_suggestion``) goes too.
    """
    empty = (("radius", -1.0), ("circle_fit_par", np.nan), ("region_count", 0),
             ("fit_source", FitSource.NONE), ("fit_confidence", np.nan),
             ("fit_locked", False), ("circle_xy", None), ("fit_flags", frozenset()),
             ("fit_suggestion", None))
    for name, value in empty:
        values = getattr(state, name)
        if values is not None:
            values[idx] = value
    for values in (state.fit_metrics or {}).values():
        values[idx] = np.nan


def update_state(state: AppState, **kwargs) -> AppState:
    """Create a new AppState with the specified fields replaced.

    Uses ``dataclasses.replace()`` under the hood, so only valid field
    names are accepted.  Raises ``TypeError`` for unknown fields.
    """
    return replace(state, **kwargs)
