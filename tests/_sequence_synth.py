"""Synthetic bubble SEQUENCES with exact ground truth, for the sequence-solver tests.

Ported from the hard-case research generator (``synth_hard.py``), without the
speckle model and with a post-collapse microbubble cloud added::

    I_t(x) = bg * T_t(x) + noise        (noise model as ``_edge_synth.add_noise``)

* T_t: a dark bubble disk (interior transmission ``1 - contrast``, bright inner
  rim), satellite bubbles that appear with the target (distractors, one of them
  larger than the target), and, from ``cloud_from`` on, a cloud of small dark
  fragments instead of the bubble.  Rendered at 2x super-sampling, PSF blur.
* R(t): Rayleigh-like cycles ``R = Rmax (1 - u^2)^0.4``, ``u`` in [-1, 1], with
  an optional rebound cycle; the floor is the nucleus radius ``rn``.
* c(t): constant drift, an acceleration after ``t_accel`` (jet-like) and an
  optional sudden jump at ``t_jump``.

Frame 0 is the bubble-free reference.  Ground truth is returned in the app's
**1-based** full-image coordinates; ``R`` is NaN on cloud frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from bubbletrack.model.sequence import Observation

SS = 2
SAT = 65504.0
QUANT = 32.0

SATELLITES = ((45.0, 300.0, 34.0), (200.0, 110.0, 22.0), (200.0, 300.0, 12.0))
"""(row, col, radius), 0-based: distractor bubbles, the first LARGER than the drift targets."""


@dataclass(frozen=True)
class SeqSpec:
    name: str = "base"
    h: int = 250
    w: int = 400
    n: int = 24                    # frames, frame 0 = bubble-free reference
    rmax: float = 30.0
    t_grow: int = 2                # first bubble frame
    period: int = 20               # frames of the first cycle
    rebound: float = 0.0           # second-cycle Rmax / Rmax (0 = none)
    rn: float = 1.5                # nucleus radius (px)
    c0: tuple = (125.0, 200.0)     # centre at t = 0 (row, col), 0-based
    drift: tuple = (0.0, 0.0)      # px / frame (row, col)
    accel: tuple = (0.0, 0.0)      # px / frame^2 after t_accel
    t_accel: int = 10 ** 6
    jump: tuple = (0.0, 0.0)       # sudden displacement at t_jump
    t_jump: int = 10 ** 6
    contrast: float = 0.9
    blur: float = 1.0
    noise: float = 1.0
    bg: float = 32500.0
    satellites: tuple = ()
    cloud_from: int | None = None  # first frame showing the fragment cloud instead of the bubble
    cloud_n: int = 14
    cloud_spread: float = 14.0     # fragments lie within this distance of the centre (px)
    seed: int = 0


def radius_series(s: SeqSpec) -> np.ndarray:
    t = np.arange(s.n, dtype=float)
    r = np.full(s.n, s.rn)
    u = (t - s.t_grow - s.period / 2) / (s.period / 2)
    m = np.abs(u) < 1
    r[m] = np.maximum(s.rmax * (1 - u[m] ** 2) ** 0.4, s.rn)
    if s.rebound > 0:
        p2 = max(int(round(s.period * s.rebound)), 4)
        t2 = s.t_grow + s.period
        u2 = (t - t2 - p2 / 2) / (p2 / 2)
        m2 = (np.abs(u2) < 1) & (t > t2)
        r[m2] = np.maximum(s.rebound * s.rmax * (1 - u2[m2] ** 2) ** 0.4, s.rn)
    r[0] = s.rn
    if s.cloud_from is not None:
        r[s.cloud_from:] = np.nan
    return r


def centre_series(s: SeqSpec) -> np.ndarray:
    t = np.arange(s.n, dtype=float)
    c = np.zeros((s.n, 2))
    for k in range(2):
        tb = np.maximum(t - s.t_grow, 0)
        ta = np.maximum(t - s.t_accel, 0)
        c[:, k] = s.c0[k] + s.drift[k] * tb + 0.5 * s.accel[k] * ta ** 2 + s.jump[k] * (t >= s.t_jump)
    c[0] = s.c0
    return c


def _cloud(rng: np.random.Generator, centre, s: SeqSpec) -> tuple:
    ang = rng.uniform(0, 2 * np.pi, s.cloud_n)
    dist = s.cloud_spread * np.sqrt(rng.uniform(0.05, 1.0, s.cloud_n))
    rad = rng.uniform(1.2, 2.6, s.cloud_n)
    return tuple((centre[0] + d * np.sin(a), centre[1] + d * np.cos(a), r) for a, d, r in zip(ang, dist, rad))


def render_seq(s: SeqSpec) -> tuple[np.ndarray, dict]:
    """Frames ``(n, h, w)`` uint16 and ground truth ``{"R", "rc", "cc"}`` (1-based)."""
    rng = np.random.default_rng(s.seed)
    radius = radius_series(s)
    centre = centre_series(s)
    yy = (np.arange(s.h * SS) + 0.5) / SS - 0.5
    xx = (np.arange(s.w * SS) + 0.5) / SS - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    trans = 1.0 - s.contrast
    frames = np.zeros((s.n, s.h, s.w), np.uint16)
    for t in range(s.n):
        T = np.ones(Y.shape)
        if t > 0 and np.isfinite(radius[t]):
            r = np.hypot(Y - centre[t, 0], X - centre[t, 1])
            interior = np.full(Y.shape, trans)
            if radius[t] > 8:
                interior = np.where(r > 0.88 * radius[t], trans + 0.11 * (1 - trans), interior)
            T = np.where(r < radius[t], interior, T)
        elif t > 0:
            for (fr, fc, frad) in _cloud(rng, centre[t], s):
                T = np.where(np.hypot(Y - fr, X - fc) < frad, trans, T)
        if t >= s.t_grow:
            for (sr, sc, srad) in s.satellites:
                T = np.where(np.hypot(Y - sr, X - sc) < srad, trans, T)
        T = T.reshape(s.h, SS, s.w, SS).mean(axis=(1, 3))
        if s.blur > 0:
            T = ndimage.gaussian_filter(T, s.blur, mode="nearest")
        frames[t] = add_noise(s.bg * T, s.noise, rng)
    return frames, {"R": radius, "rc": centre[:, 0] + 1.0, "cc": centre[:, 1] + 1.0}


def add_noise(clean: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    sig = scale * np.sqrt(150.0 ** 2 + 5.9 * np.clip(clean, 0, None))
    noisy = np.clip(clean + sig * rng.standard_normal(clean.shape), 0, SAT)
    return (np.round(noisy / QUANT) * QUANT).astype(np.uint16)


# --------------------------------------------------------------------------- #
#  Scenario families
# --------------------------------------------------------------------------- #
def drift_spec(speed: float, seed: int = 0) -> SeqSpec:
    """Constant drift; the 20-frame path is centred in the 400 x 250 frame."""
    n_b = 20
    start = 200.0 - speed * (n_b - 1) / 2
    return SeqSpec(name=f"drift{int(speed):02d}", n=n_b + 4, rmax=30.0, period=n_b, c0=(125.0, start),
                   drift=(0.0, float(speed)), satellites=SATELLITES, seed=seed)


def jet_spec(seed: int = 0) -> SeqSpec:
    """Still during growth, then accelerating during the collapse (to about 20 px / frame)."""
    return SeqSpec(name="jet_accel", n=26, rmax=40.0, period=22, c0=(125.0, 110.0), accel=(0.0, 2.0),
                   t_accel=12, satellites=SATELLITES, seed=seed)


def jump_spec(seed: int = 0) -> SeqSpec:
    """A sudden 35 px jump at frame 10 (e.g. a re-nucleated bubble)."""
    return SeqSpec(name="jump35", n=26, rmax=35.0, period=22, c0=(125.0, 180.0), jump=(10.0, 33.0),
                   t_jump=10, satellites=SATELLITES, seed=seed)


def satellite_spec(seed: int = 0) -> SeqSpec:
    """Static target next to a larger satellite (the target never exceeds R 30, the satellite is 34)."""
    return SeqSpec(name="satellite_larger", n=24, rmax=30.0, period=20, c0=(125.0, 170.0),
                   satellites=SATELLITES, seed=seed)


def cloud_spec(seed: int = 0) -> SeqSpec:
    """Collapse, rebound (0.45 Rmax), second collapse, then a fragment cloud to the end."""
    return SeqSpec(name="collapse_rebound_cloud", n=44, rmax=50.0, period=24, rebound=0.45, rn=1.5,
                   c0=(125.0, 200.0), cloud_from=38, seed=seed)


def catalogue(seed: int = 0) -> list[SeqSpec]:
    return ([drift_spec(v, seed) for v in (0, 2, 5, 10, 15, 20)]
            + [jet_spec(seed), jump_spec(seed), satellite_spec(seed), cloud_spec(seed)])


# --------------------------------------------------------------------------- #
#  Scripted observation worlds (no images) for the solver unit tests
# --------------------------------------------------------------------------- #
def rayleigh_radii(n: int, cycles, floor: float = 0.0) -> np.ndarray:
    """R(t) from ``(start, period, rmax)`` cycles; ``floor`` elsewhere (0 = no bubble)."""
    t = np.arange(n, dtype=float)
    r = np.full(n, floor)
    for start, period, rmax in cycles:
        u = (t - start - period / 2) / (period / 2)
        m = np.abs(u) < 1
        r[m] = np.maximum(rmax * (1 - u[m] ** 2) ** 0.4, floor)
    return r


def truth_track(radii, centre=(125.0, 200.0), velocity=(0.0, 0.0)) -> tuple:
    """Per-frame true circle ``(rc, cc, R)``, or None where R <= 0 / NaN."""
    out = []
    for t, r in enumerate(radii):
        ok = np.isfinite(r) and r > 0
        out.append((centre[0] + velocity[0] * t, centre[1] + velocity[1] * t, float(r)) if ok else None)
    return tuple(out)


class ScriptedDetector:
    """``redetect(idx, prior)`` over a scripted truth, recording every call.

    Returns the true circle when the prior's centre is within ``0.6 R + 8`` px of it,
    otherwise ``elsewhere[idx]`` (another object) or nothing.  ``overrides[idx]``
    replaces the answer for one frame (e.g. a failing frame)."""

    def __init__(self, truth, *, confidence: float = 0.9, elsewhere=None, overrides=None):
        self.truth = tuple(truth)
        self.confidence = confidence
        self.elsewhere = dict(elsewhere or {})
        self.overrides = dict(overrides or {})
        self.calls: list[tuple[int, tuple | None]] = []

    @staticmethod
    def nothing() -> Observation:
        return Observation(float("nan"), float("nan"), float("nan"), 0.0)

    def __call__(self, idx, prior):
        self.calls.append((idx, prior))
        if idx in self.overrides:
            return self.overrides[idx]
        t = self.truth[idx]
        if t is not None and prior is not None and np.hypot(prior[0] - t[0], prior[1] - t[1]) <= 0.6 * t[2] + 8:
            return Observation(t[2], t[0], t[1], self.confidence)
        return self.elsewhere.get(idx, self.nothing())

    @property
    def frames_called(self) -> list[int]:
        return sorted({i for i, _ in self.calls})


def observe(truth, *, confidence: float = 0.95, replace=None) -> tuple:
    """Raw single-frame results: the truth (or nothing), with ``replace[idx]`` swapped in."""
    nan = float("nan")
    out = []
    for i, t in enumerate(truth):
        if replace and i in replace:
            out.append(replace[i])
        elif t is None:
            out.append(Observation(nan, nan, nan, 0.0))
        else:
            out.append(Observation(t[2], t[0], t[1], confidence))
    return tuple(out)
