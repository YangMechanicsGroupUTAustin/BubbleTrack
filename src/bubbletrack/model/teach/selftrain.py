"""Self-training: ML assist taught by Auto's own results, with no user input.

After Fit All (and tracking), the frames :func:`suggest_frames` would ask the user to
verify are taken as they are (Auto's accepted circles) and become the examples; the
trained model then re-measures only the frames that have no result.  Accepted
results are never touched, so recordings where Auto is good cannot get worse.

    plan = plan_self_training(radius, centres, confidence, protected=manual_or_locked)
    if plan.run:
        model = train_model(plan.examples(), read_raw, gridx, gridy, reference=ref)
        cfg = SelfTrainConfig().measure_config(model.config.measure)
        for k in plan.measure:
            res = measure_frame(model, read_raw(k), params=params, config=cfg)
            if res.accepted:
                store(k, res)            # an "ml" outline: always flagged for Review

Measured on the labelled recordings (lic 01-10, lic_hard 01-08, example_1; after Fit
All + tracking): where Auto is good 178 -> 179/182 correct (no frame lost, no extra
wrong one); hard recordings 54 -> 92/155 correct (accepted but wrong 12 -> 24,
labelled no-bubble frames shown 10 -> 11/116), against 116/155 (32 wrong, 22 no-bubble
frames shown) with two user-verified frames.

* Any accepted frame may teach: on the hard recordings nearly every result is below
  0.5, so requiring 0.5 (or 0.7, 0.8) leaves them untaught (hard 54 -> 54-63).
* ML's own outline needs confidence >= 0.5 (0.3: same 92 correct, 29 wrong, 15
  no-bubble frames shown).  Auto guided by ML practically never adds a frame here:
  the gain is ML's outline, which is always flagged for Review.
* The pass runs only when >= 5 % of the frames have no result (no effect on the
  counts: it skips recordings that need nothing) and >= 10 frames are accepted (on
  the 3 recordings with 5-7 accepted frames the pass gained 2 correct frames for 4
  no-bubble frames shown; one taught a pre-nucleation spot).
* Once the user verifies frames, train on those only: adding the self-trained frames
  gave fewer correct frames (113 vs 116/155) and more wrong ones (36 vs 32).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from bubbletrack.model.teach.config import MeasureConfig
from bubbletrack.model.teach.examples import MLExamples
from bubbletrack.model.teach.suggest import suggest_frames

Circle = tuple[float, float, float]


@dataclass(frozen=True)
class SelfTrainConfig:
    """When the automatic second pass runs and how it reports.  Defaults are the measured best."""

    n_frames: int = 2
    """Auto's results used as examples (``suggest_frames``: the biggest bubble and a small one)."""
    min_failed_fraction: float = 0.05
    """Run only when at least this fraction of the frames has no result."""
    min_accepted: int = 10
    """... and at least this many frames have an accepted result to learn from."""
    accept_gate: float = 0.3
    """An example frame needs a result with at least this confidence (Auto's own gate)."""
    outline_gate: float = 0.5
    """ML's own outline is reported only at this confidence or above."""

    def __post_init__(self) -> None:
        if int(self.n_frames) < 1 or int(self.min_accepted) < 1:
            raise ValueError("n_frames and min_accepted must be >= 1")
        for name in ("min_failed_fraction", "accept_gate", "outline_gate"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")

    def measure_config(self, base: MeasureConfig) -> MeasureConfig:
        """*base* with the outline gate of the automatic pass (Auto's own gate is unchanged)."""
        return replace(base, confidence_gate=float(self.outline_gate))


@dataclass(frozen=True)
class SelfTrainPlan:
    """What the automatic second pass would do on one recording."""

    train: Mapping[int, Circle] = field(default_factory=dict)
    """{frame: (rc, cc, R)}: Auto's accepted circles used as verified examples."""
    measure: tuple[int, ...] = ()
    """Frames to re-measure with ML assist (no result, not protected, not an example)."""
    reason: str = ""
    """Why the pass does not run ("" when it runs)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", MappingProxyType(dict(self.train)))

    __hash__ = None

    @property
    def run(self) -> bool:
        return not self.reason

    def examples(self) -> MLExamples:
        return MLExamples().with_verified(self.train)


def _accepted(radius: np.ndarray, centres: np.ndarray, conf: np.ndarray, gate: float) -> np.ndarray:
    has = np.isfinite(radius) & (radius > 0) & np.isfinite(centres).all(axis=1)
    return has & np.isfinite(conf) & (conf >= gate)


def plan_self_training(radius, centres, confidence, *, protected: Iterable[int] = (),
                       config: SelfTrainConfig = SelfTrainConfig()) -> SelfTrainPlan:
    """Plan the automatic second pass from the stored per-frame results.

    Parameters
    ----------
    radius : sequence of float, length N
        Stored radius per frame; <= 0 or NaN = no result (the app stores -1).
    centres : (N, 2) array
        Stored ``(rc, cc)`` per frame (1-based full-image), NaN without a result.
    confidence : sequence of float, length N
        Engine confidence per frame (NaN = none, e.g. a legacy mode: not used to teach).
    protected : iterable of int
        Frames the pass must never change (manual, drawn-circle and locked frames).
    config : SelfTrainConfig

    Returns
    -------
    SelfTrainPlan
        ``plan.run`` is False (with ``plan.reason``) when the recording does not need
        the pass or has too little to learn from.
    """
    r = np.asarray(radius, dtype=float)
    c = np.asarray(centres, dtype=float)
    q = np.asarray(confidence, dtype=float)
    if r.ndim != 1 or q.shape != r.shape or c.shape != (len(r), 2):
        raise ValueError(f"radius, centres and confidence must have shapes (N,), (N, 2), (N,); "
                         f"got {r.shape}, {c.shape}, {q.shape}")
    n = len(r)
    if n == 0:
        return SelfTrainPlan(reason="no frames")
    keep = {int(i) for i in protected}
    empty = [i for i in range(n) if not (np.isfinite(r[i]) and r[i] > 0) and i not in keep]
    if len(empty) < config.min_failed_fraction * n or not empty:
        return SelfTrainPlan(reason=f"{len(empty)} of {n} frames without a result "
                                    f"(< {100 * config.min_failed_fraction:.0f} %)")
    ok = _accepted(r, c, q, config.accept_gate)
    if int(ok.sum()) < config.min_accepted:
        return SelfTrainPlan(reason=f"{int(ok.sum())} accepted frames to learn from (< {config.min_accepted})")
    pick = suggest_frames(np.where(ok, r, np.nan), np.where(ok, q, 0.0), n=int(config.n_frames),
                          gate=config.accept_gate)
    train = {int(f): (float(c[f, 0]), float(c[f, 1]), float(r[f])) for f in pick if ok[f]}
    if not train:
        return SelfTrainPlan(reason="no accepted frame to learn from")
    return SelfTrainPlan(train=train, measure=tuple(i for i in empty if i not in train))
