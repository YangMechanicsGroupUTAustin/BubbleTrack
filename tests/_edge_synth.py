"""Synthetic shadowgraph frames with exact ground truth, for the Edge engine tests.

Ported from the research prototype's generator (calibrated on example_1)::

    I(x, y) = B(x, y) * T(x, y) + noise,  clipped at 65504 and quantised to 32 DN

* B: background illumination (~32.5k DN), optional vignette / linear ramp /
  static features (smudge, side bands, lines, dust) and static speckle.
* T: transmission: 1 outside objects, ``trans`` inside the bubble, a brighter
  5-px inner rim, an optional faint ring at 0.8 R, an optional saturated centre
  spot, an optional outer dark halo, debris and dark blobs.  Rendered at 4x
  super-sampling (exact area coverage), then blurred by the optical PSF.
* noise sigma(I) = scale * sqrt(150^2 + 5.9 I)  (~460 DN at the background).

Ground truth: the target bubble's centre in **0-based** pixel-centre
coordinates (add 1 for the app's 1-based convention) and the area-equivalent
radius ``r_eq`` of its sharp boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

SS = 4                 # super-sampling factor
SAT = 65504.0
QUANT = 32.0


@dataclass(frozen=True)
class Bubble:
    rc: float
    cc: float
    r: float
    trans: float = 0.07          # interior transmission (1 - contrast)
    aspect: float = 1.0          # ellipse major/minor (area preserved)
    angle: float = 0.0
    wobble: tuple = ()           # ((k, rel_amp, phase), ...)
    rim: bool = True
    ring: bool = True
    spot: bool = True            # saturated bright centre spot
    spot_frac: float = 0.22
    halo_w: float = 0.0          # outer dark halo width (px)
    halo_trans: float = 1.0
    texture: float = 0.0         # interior mottling amplitude


@dataclass(frozen=True)
class Scene:
    h: int = 250
    w: int = 400
    bg: float = 32500.0
    bubbles: tuple = ()
    target: int = 0
    blur: float = 1.0
    noise: float = 1.0
    vignette: float = 0.0
    gradient: float = 0.0
    static: int = 0               # 0 none, 1 real-like, 2 strong
    speckle: float = 0.0          # static speckle dot darkness (0 = none)
    spk_scale: float = 1.0
    debris: tuple = ()            # ((rc, cc, r, trans), ...)
    blobs: tuple = ()             # ((rc, cc, a, b, ang, trans), ...)
    seed: int = 0


def boundary_radius(b: Bubble, theta: np.ndarray) -> np.ndarray:
    if b.aspect != 1.0:
        a_ax, b_ax = b.r * np.sqrt(b.aspect), b.r / np.sqrt(b.aspect)
        t = theta - b.angle
        r = a_ax * b_ax / np.sqrt((b_ax * np.cos(t)) ** 2 + (a_ax * np.sin(t)) ** 2)
    else:
        r = np.full_like(theta, b.r, dtype=float)
    for k, amp, ph in b.wobble:
        r = r * (1.0 + amp * np.cos(k * theta + ph))
    return r


def truth(b: Bubble) -> dict:
    th = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    rb = boundary_radius(b, th)
    r_eq = float(np.sqrt(0.5 * np.mean(rb ** 2) * 2 * np.pi / np.pi))
    return {"rc": b.rc, "cc": b.cc, "r_eq": r_eq}


def _static_features(scene: Scene) -> np.ndarray:
    s = np.ones((scene.h, scene.w))
    y0, x0 = np.mgrid[0:scene.h, 0:scene.w].astype(float)
    if scene.static:
        k = 1.0 if scene.static == 1 else 2.0
        s *= 1 - 0.12 * k * np.exp(-((y0 - 105) ** 2 + (x0 - 95) ** 2) / (2 * 9.0 ** 2))
        for xc in (22.0, 378.0):
            s *= 1 + 0.06 * np.exp(-((x0 - xc) ** 2) / (2 * 6.0 ** 2))
        for (r0, slope) in ((232.0, 0.05), (241.0, -0.03)):
            d = np.abs(y0 - (r0 + slope * (x0 - 200)))
            s *= 1 - 0.18 * k * np.exp(-d ** 2 / (2 * 0.8 ** 2))
        if scene.static == 2:
            s *= 1 - 0.30 * np.exp(-((y0 - 60) ** 2 + (x0 - 320) ** 2) / (2 * 7.0 ** 2)) ** 0.5
    if scene.speckle > 0:
        field = ndimage.gaussian_filter(np.random.default_rng(10_000 + scene.seed).standard_normal(s.shape),
                                        scene.spk_scale)
        field /= field.std()
        s *= 1.0 - scene.speckle / (1.0 + np.exp(-(field - 1.0) / 0.25))
    return s


def render_clean(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free frame and bubble-free background (both float)."""
    rng = np.random.default_rng(scene.seed + 777)
    hh, ww = scene.h * SS, scene.w * SS
    yy = (np.arange(hh) + 0.5) / SS - 0.5
    xx = (np.arange(ww) + 0.5) / SS - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    T = np.ones((hh, ww))
    for b in scene.bubbles:
        dy, dx = Y - b.rc, X - b.cc
        dist = np.hypot(dy, dx)
        rb = boundary_radius(b, np.arctan2(dy, dx))
        if b.halo_w > 0:
            T[(dist >= rb) & (dist < rb + b.halo_w)] *= b.halo_trans
        interior = np.full(T.shape, b.trans)
        if b.rim:
            interior = np.where(dist > rb - 5.0, b.trans + 0.11 * (1 - b.trans), interior)
        if b.ring:
            interior = np.where(np.abs(dist - 0.8 * rb) < 1.0, interior + 0.05 * (1 - b.trans), interior)
        if b.texture > 0:
            tex = ndimage.gaussian_filter(rng.standard_normal((scene.h, scene.w)), 3.0)
            tex = np.kron(tex / (tex.std() + 1e-12), np.ones((SS, SS)))
            interior = np.clip(interior + b.texture * tex, 0.01, 1.0)
        if b.spot:
            interior = np.where(dist < b.spot_frac * b.r, 2.2, interior)
        T = np.where(dist < rb, interior, T)
    for (rc, cc, r, tr) in scene.debris:
        T[np.hypot(Y - rc, X - cc) < r] = tr
    for (rc, cc, a, bb, ang, tr) in scene.blobs:
        dy, dx = Y - rc, X - cc
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = -dx * np.sin(ang) + dy * np.cos(ang)
        T[(u / a) ** 2 + (v / bb) ** 2 < 1] = tr
    T = T.reshape(scene.h, SS, scene.w, SS).mean(axis=(1, 3))
    if scene.blur > 0:
        T = ndimage.gaussian_filter(T, scene.blur, mode="nearest")
    y0, x0 = np.mgrid[0:scene.h, 0:scene.w].astype(float)
    rr2 = ((y0 - scene.h / 2) / (scene.h / 2)) ** 2 + ((x0 - scene.w / 2) / (scene.w / 2)) ** 2
    B = scene.bg * (1 - scene.vignette * rr2 / 2.0)
    B = B * (1 + scene.gradient * (x0 - scene.w / 2) / scene.w) * _static_features(scene)
    return B * T, B


def add_noise(clean: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    sig = scale * np.sqrt(150.0 ** 2 + 5.9 * np.clip(clean, 0, None))
    noisy = np.clip(clean + sig * rng.standard_normal(clean.shape), 0, SAT)
    return (np.round(noisy / QUANT) * QUANT).astype(np.uint16)


def render(scene: Scene) -> tuple[np.ndarray, dict]:
    """uint16 frame plus ``{"gt", "ref", "clean"}`` (ref = noisy bubble-free background)."""
    clean, background = render_clean(scene)
    rng = np.random.default_rng(scene.seed)
    img = add_noise(clean, scene.noise, rng)
    ref = add_noise(background, scene.noise, rng)
    gt = truth(scene.bubbles[scene.target]) if scene.bubbles else {}
    return img, {"gt": gt, "ref": ref, "clean": clean}


# --------------------------------------------------------------------------- #
#  Benchmark catalogue (one seed per scenario)
# --------------------------------------------------------------------------- #

AMBIGUOUS = frozenset({"halo_w6_t0.6", "halo_w12_t0.35", "target_smaller_than_distractor"})
"""Scenes where 'the bubble edge' is a matter of definition or of which object is meant."""


@dataclass(frozen=True)
class Case:
    group: str
    name: str
    scene: Scene
    seed_click: bool = False      # pass a click near the target (0-based coords + jitter)


def _jit(rng, s=8.0):
    return rng.uniform(-s, s)


def catalogue() -> list[Case]:
    """About 50 scenes covering the conditions of the prototype benchmark."""
    out: list[Case] = []

    def add(group, name, make, seed_click=False):
        rng = np.random.default_rng(1000 * len(out))
        out.append(Case(group, name, make(rng), seed_click))

    def base(rng, **kw):
        d = dict(rc=125 + _jit(rng), cc=200 + _jit(rng, 20), r=40 + _jit(rng, 4))
        d.update(kw)
        return Bubble(**d)

    add("base", "base", lambda g: Scene(bubbles=(base(g),)))
    for c in (0.05, 0.10, 0.20, 0.40, 0.80):
        add("contrast", f"contrast_{int(c * 100)}%", lambda g, c=c: Scene(bubbles=(base(g, trans=1 - c),)))
    for n in (0.5, 2.0, 4.0, 8.0):
        add("noise", f"noise_x{n:g}", lambda g, n=n: Scene(bubbles=(base(g),), noise=n))
    add("noise", "noise_x4_contrast20%", lambda g: Scene(bubbles=(base(g, trans=0.8),), noise=4.0))
    add("illum", "gradient_30%", lambda g: Scene(bubbles=(base(g),), gradient=0.3))
    add("illum", "vignette_40%", lambda g: Scene(bubbles=(base(g),), vignette=0.4))
    add("illum", "vig40+grad30+contrast20%",
        lambda g: Scene(bubbles=(base(g, trans=0.8),), vignette=0.4, gradient=0.3))
    for bl in (0.5, 1.5, 2.5):
        add("blur", f"blur_s{bl:g}", lambda g, bl=bl: Scene(bubbles=(base(g),), blur=bl))
    add("spot", "no_spot", lambda g: Scene(bubbles=(base(g, spot=False),)))
    add("spot", "big_saturated_spot", lambda g: Scene(bubbles=(base(g, spot_frac=0.45),)))
    for hw, ht in ((6, 0.6), (12, 0.35)):
        add("halo", f"halo_w{hw}_t{ht:g}",
            lambda g, hw=hw, ht=ht: Scene(bubbles=(base(g, halo_w=hw, halo_trans=ht),)))
    for f in (0.1, 0.3, 0.5, 0.7):
        add("clip", f"clip_top_{int(f * 100)}%",
            lambda g, f=f: Scene(bubbles=(Bubble(rc=60 * (1 - 2 * f), cc=200 + _jit(g, 20), r=60),)))
    add("clip", "clip_corner_30%", lambda g: Scene(bubbles=(Bubble(rc=24.0, cc=24.0, r=60),)))
    add("multi", "two_separate",
        lambda g: Scene(bubbles=(Bubble(rc=125, cc=140 + _jit(g), r=45), Bubble(rc=110, cc=300, r=28))))
    add("multi", "three_separate",
        lambda g: Scene(bubbles=(Bubble(rc=125, cc=200 + _jit(g), r=40), Bubble(rc=60, cc=320, r=25),
                                 Bubble(rc=200, cc=70, r=15))))
    add("multi", "two_touching",
        lambda g: Scene(bubbles=(Bubble(rc=125, cc=170, r=45), Bubble(rc=125, cc=170 + 45 + 28 - 3, r=28))))
    smaller = lambda g: Scene(bubbles=(Bubble(rc=125, cc=110, r=30),   # noqa: E731
                                       Bubble(rc=125, cc=280, r=50, spot=False)))
    add("multi", "target_smaller_than_distractor", smaller)
    add("multi", "target_smaller_with_seed", smaller, seed_click=True)
    add("multi", "two_separate_seed_on_small",
        lambda g: Scene(bubbles=(Bubble(rc=110, cc=300, r=28), Bubble(rc=125, cc=140, r=45))), seed_click=True)
    add("multi", "irregular_dark_distractor",
        lambda g: Scene(bubbles=(Bubble(rc=125, cc=140, r=35),), blobs=((150, 300, 70, 30, 0.4, 0.15),)))
    add("debris", "debris_near",
        lambda g: Scene(bubbles=(Bubble(rc=125, cc=200, r=40),),
                        debris=((125, 248, 5, 0.1), (80, 180, 4, 0.15), (165, 225, 3, 0.1))))
    add("debris", "debris_attached",
        lambda g: Scene(bubbles=(Bubble(rc=125, cc=200, r=40),), debris=((125, 244, 7, 0.08), (88, 185, 6, 0.1))))
    add("debris", "debris_field",
        lambda g: Scene(bubbles=(Bubble(rc=125, cc=200, r=40),),
                        debris=tuple((float(g.uniform(10, 240)), float(g.uniform(10, 390)),
                                      float(g.uniform(1.5, 4)), 0.2) for _ in range(40))))
    add("static", "static_reallike", lambda g: Scene(bubbles=(base(g),), static=1))
    add("static", "static_strong_contrast40%", lambda g: Scene(bubbles=(base(g, trans=0.6),), static=2))
    add("static", "static_speckle", lambda g: Scene(bubbles=(base(g),), speckle=0.45))
    add("static", "static_speckle_contrast40%", lambda g: Scene(bubbles=(base(g, trans=0.6),), speckle=0.3))
    for a in (1.15, 1.3):
        add("shape", f"ellipse_{a:g}",
            lambda g, a=a: Scene(bubbles=(base(g, aspect=a, angle=float(g.uniform(0, np.pi))),)))
    add("shape", "wobbly_k3-6_3%",
        lambda g: Scene(bubbles=(base(g, wobble=tuple((k, 0.03, float(g.uniform(0, 6.28))) for k in (3, 4, 5, 6))),)))
    add("shape", "rough_k8-20_textured",
        lambda g: Scene(bubbles=(base(g, trans=0.15, spot=False, ring=False, texture=0.06,
                                      wobble=tuple((k, 0.015, float(g.uniform(0, 6.28)))
                                                   for k in range(8, 21, 2))),)))
    for r in (6, 7, 10):
        add("size", f"tiny_R{r}", lambda g, r=r: Scene(bubbles=(Bubble(rc=125 + _jit(g), cc=200 + _jit(g), r=r,
                                                                       spot=r >= 7, ring=False),)))
    add("size", "huge_R100", lambda g: Scene(bubbles=(Bubble(rc=125, cc=200 + _jit(g), r=100),)))
    add("size", "huge_R112", lambda g: Scene(bubbles=(Bubble(rc=125, cc=200 + _jit(g, 5), r=112),)))
    for r in (30, 55, 80):
        add("realistic", f"real_like_R{r}_top_clipped",
            lambda g, r=r: Scene(bubbles=(Bubble(rc=30 + _jit(g, 2), cc=198 + _jit(g, 3), r=r),), static=1,
                                 vignette=0.1))
    add("realistic", "real_like_postcollapse_rough",
        lambda g: Scene(bubbles=(Bubble(rc=40, cc=198, r=32, trans=0.2, spot=False, ring=False, texture=0.06,
                                        wobble=tuple((k, 0.02, float(g.uniform(0, 6.28))) for k in range(5, 17, 2))),),
                        static=1))
    return out


def click_for(case: Case) -> tuple[float, float] | None:
    """A deterministic 1-based click within 0.3 R of the target centre (or None)."""
    if not case.seed_click:
        return None
    b = case.scene.bubbles[case.scene.target]
    rng = np.random.default_rng(len(case.name))
    return (b.rc + 1 + rng.uniform(-0.3, 0.3) * b.r, b.cc + 1 + rng.uniform(-0.3, 0.3) * b.r)
