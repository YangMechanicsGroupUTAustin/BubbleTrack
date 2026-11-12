"""Settings of ML assist.  None of them is a user parameter: the defaults are the
measured best (``tools/studies/teach/RESULTS.md``); they exist for the study and tests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_CHANNELS = ("raw", "view")
_FORESTS = ("extra", "random")
_POLICIES = ("combined", "guarded", "prior", "contour")


def _positive(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive number, got {value!r}")


@dataclass(frozen=True)
class ExampleConfig:
    """Automatic examples from a verified circle, and how many pixels a frame gives."""

    inner: float = 0.8
    """Pixels within ``inner * R`` of a verified centre are bubble."""
    ring: tuple[float, float] = (1.25, 2.0)
    """Pixels between ``ring[0] * R`` and ``ring[1] * R`` are background ..."""
    far: float | None = 2.5
    """... and so is everything beyond ``far * R`` (None: only the ring)."""
    max_per_class: int = 3000
    """Training pixels per class and frame (strokes first, then automatic ones)."""
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.inner < 1.0):
            raise ValueError(f"inner must be in (0, 1), got {self.inner!r}")
        lo, hi = self.ring
        if not (1.0 <= lo < hi):
            raise ValueError(f"ring must satisfy 1 <= ring[0] < ring[1], got {self.ring!r}")
        if self.far is not None and self.far < hi:
            raise ValueError(f"far must be >= ring[1], got {self.far!r}")
        if int(self.max_per_class) < 1:
            raise ValueError("max_per_class must be >= 1")


@dataclass(frozen=True)
class FeatureConfig:
    """The fixed filter bank: Gaussian-smoothed intensity, gradient magnitude and
    Hessian eigenvalues at ``num_sigma`` scales from ``sigma_min`` to ``sigma_max``,
    of each channel ("raw" = the grey frame over its robust level; "view" = the
    engine's normalised view, frame / reference when a reference exists)."""

    channels: tuple[str, ...] = ("raw", "view")
    sigma_min: float = 1.0
    sigma_max: float = 16.0
    num_sigma: int = 5
    texture: bool = True
    half_resolution: bool = False
    """Compute features and predict on a 2x2-binned grid (sigmas halved), upsample P."""
    polarity: str = "dark"
    num_workers: int = 1

    def __post_init__(self) -> None:
        if not self.channels or any(c not in _CHANNELS for c in self.channels) \
                or len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channels must be distinct names from {_CHANNELS}, got {self.channels!r}")
        _positive("sigma_min", self.sigma_min)
        if not self.sigma_max >= self.sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")
        if int(self.num_sigma) < 1:
            raise ValueError("num_sigma must be >= 1")
        if self.polarity not in ("dark", "bright"):
            raise ValueError(f"polarity must be 'dark' or 'bright', got {self.polarity!r}")


@dataclass(frozen=True)
class ForestConfig:
    """The pixel classifier: a tree ensemble with balanced classes and a fixed seed."""

    kind: str = "extra"
    """"extra" (ExtraTrees: fast to train) or "random" (RandomForest)."""
    n_estimators: int = 60
    max_depth: int | None = 12
    min_samples_leaf: int = 2
    max_samples: int = 40_000
    """Total training pixels (each class capped at half, deterministic subsample)."""
    random_state: int = 0
    n_jobs: int = 4
    """Threads for training and prediction (the result does not depend on it)."""

    def __post_init__(self) -> None:
        if self.kind not in _FORESTS:
            raise ValueError(f"kind must be one of {_FORESTS}, got {self.kind!r}")
        if int(self.n_estimators) < 1 or int(self.max_samples) < 2 or int(self.n_jobs) < 1:
            raise ValueError("n_estimators, max_samples and n_jobs must be positive")


@dataclass(frozen=True)
class MeasureConfig:
    """From the probability map to the reported circle."""

    policy: str = "combined"
    """"combined": Auto measures the wall inside the ML region, else ML's own outline;
    "guarded": the same, but plain Auto's accepted result is kept before ML's outline;
    "prior": always Auto with the ML circle as its prior; "contour": always ML's outline."""
    level: float = 0.5
    smooth_sigma: float = 1.0
    min_radius_px: float = 3.0
    refine: bool = True
    """Move ML's outline to the half-contrast wall of the engine view near it."""
    refine_band: float = 0.15
    """Half-width of the window (fraction of R, at least 3 px) searched by the refinement."""
    refine_passes: int = 1
    """Each further pass recentres on the previous outline with half the window."""
    refine_sectors: int = 8
    """Angular sectors with their own wall crossing (1 = one shift for the whole outline)."""
    refine_min_contrast: float = 0.05
    agree_radius: tuple[float, float] = (0.15, 3.0)
    """Auto's wall belongs to the ML region when |dR| <= max(frac * R, px) ..."""
    agree_centre: tuple[float, float] = (0.25, 4.0)
    """... and the centres are within max(frac * R, px)."""
    confidence_gate: float = 0.3
    novelty_warn: float = 0.15
    """Flag ``ml_unfamiliar`` above this fraction of out-of-range features."""

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}, got {self.policy!r}")
        if not (0.0 < self.level < 1.0):
            raise ValueError(f"level must be in (0, 1), got {self.level!r}")
        if not (0.0 <= self.confidence_gate <= 1.0):
            raise ValueError("confidence_gate must be in [0, 1]")


@dataclass(frozen=True)
class MLConfig:
    examples: ExampleConfig = field(default_factory=ExampleConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
