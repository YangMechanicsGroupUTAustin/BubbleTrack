"""Deterministic detection cases that pin the current per-frame pipeline output.

The golden values in ``golden_detection.json`` were produced by the code at
commit d0fdbc7.  They guard the promise that the existing detection modes
(intensity / gradient / area) keep producing bit-identical results while new
features are added.

Regenerate ONLY when a change to existing results is intended and reviewed::

    python tests/_golden_cases.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

GOLDEN_PATH = Path(__file__).with_name("golden_detection.json")
REAL_FOLDER = Path(__file__).resolve().parents[1] / "example_1"

# Owner's everyday parameters (from their config.json), used for real frames.
OWNER_PARAMS = {
    "threshold": 0.58,
    "rf_slider": 85.0,
    "sigma": 3.0,
    "edges": (True, False, False, False),
}
REAL_FRAMES = (11, 23, 35, 53, 65, 89)


@dataclass(frozen=True)
class SceneSpec:
    """A synthetic shadowgraph-like frame: dark bubble on a bright background."""

    radius: float
    center: tuple[float, float]
    shape: tuple[int, int] = (250, 400)
    background: float = 32000.0
    interior: float = 0.06  # interior level as a fraction of the background
    edge_width: float = 1.0  # px, softness of the bubble wall
    spot_radius: float = 0.0  # saturated central spot (0 = none)
    halo_radius: float = 0.0  # outer radius of a dark halo (0 = none)
    halo_level: float = 0.5
    ramp: float = 0.0  # left-to-right illumination ramp amplitude (fraction)
    noise: float = 0.01  # Gaussian noise std as a fraction of the background
    debris: int = 0  # number of small dark particles
    bit_depth: int = 16
    seed: int = 0


@dataclass(frozen=True)
class PipelineSpec:
    """Parameters for one call of the Fit All per-frame worker."""

    mode: str = "intensity"
    threshold: float = 0.58
    rf_slider: float = 85.0
    sigma: float = 3.0
    clahe: float = 0.0
    closing: int = 0
    opening: int = 0
    edges: tuple[bool, bool, bool, bool] = (False, False, False, False)
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None  # None = full frame
    gradient_boundary: float = 0.0
    area_max_regions: int = 1


@dataclass(frozen=True)
class GoldenCase:
    name: str
    scene: SceneSpec
    pipeline: PipelineSpec = field(default_factory=PipelineSpec)


def _soft_disk(dist: np.ndarray, radius: float, width: float) -> np.ndarray:
    """1 inside the disk, 0 outside, with a tanh edge of the given width."""
    return 0.5 * (1.0 + np.tanh((radius - dist) / max(width, 1e-6)))


def render_scene(spec: SceneSpec) -> np.ndarray:
    """Render a deterministic frame as uint16 (or uint8 when bit_depth == 8)."""
    rng = np.random.default_rng(spec.seed)
    h, w = spec.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    bg = spec.background * (1.0 + spec.ramp * (xx / w - 0.5))
    img = bg.copy()

    dist = np.hypot(yy - spec.center[0], xx - spec.center[1])
    if spec.halo_radius > spec.radius:
        halo = _soft_disk(dist, spec.halo_radius, spec.edge_width)
        img = img * (1.0 - halo) + bg * spec.halo_level * halo
    bubble = _soft_disk(dist, spec.radius, spec.edge_width)
    img = img * (1.0 - bubble) + bg * spec.interior * bubble
    if spec.spot_radius > 0:
        spot = _soft_disk(dist, spec.spot_radius, 0.8)
        img = img * (1.0 - spot) + 65000.0 * spot

    for _ in range(spec.debris):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(2.0, 4.0)
        d = _soft_disk(np.hypot(yy - cy, xx - cx), r, 0.7)
        img = img * (1.0 - d) + bg * 0.2 * d

    img = img + rng.normal(0.0, spec.noise * spec.background, size=img.shape)
    img = np.clip(img, 0, 65535)
    if spec.bit_depth == 8:
        return np.clip(np.round(img / 257.0), 0, 255).astype(np.uint8)
    return np.round(img).astype(np.uint16)


def _base(**kw) -> SceneSpec:
    return SceneSpec(**{"radius": 60.0, "center": (125.0, 200.0), **kw})


CASES: tuple[GoldenCase, ...] = (
    GoldenCase("clean_owner", _base(seed=1)),
    GoldenCase("clean_default_params", _base(seed=2),
               PipelineSpec(threshold=0.5, rf_slider=90.0, sigma=0.0)),
    GoldenCase("small_bubble", _base(radius=18.0, seed=3)),
    GoldenCase("large_bubble", _base(radius=95.0, seed=4)),
    GoldenCase("bright_spot", _base(spot_radius=9.0, seed=5)),
    GoldenCase("clipped_top", _base(center=(30.0, 200.0), radius=70.0, seed=6),
               PipelineSpec(edges=(True, False, False, False))),
    GoldenCase("clipped_top_no_flag", _base(center=(30.0, 200.0), radius=70.0, seed=6)),
    GoldenCase("halo", _base(halo_radius=78.0, halo_level=0.45, seed=7)),
    GoldenCase("debris", _base(debris=25, seed=8)),
    GoldenCase("ramp_low_contrast", _base(interior=0.7, ramp=0.4, seed=9)),
    GoldenCase("eight_bit", _base(bit_depth=8, seed=10)),
    GoldenCase("roi_subset", _base(seed=11),
               PipelineSpec(roi=((40, 210), (90, 310)))),
    GoldenCase("morphology", _base(debris=10, seed=12),
               PipelineSpec(opening=2, closing=3)),
    GoldenCase("clahe", _base(ramp=0.3, seed=13), PipelineSpec(clahe=2.0)),
    GoldenCase("gradient_ring1", _base(halo_radius=78.0, halo_level=0.45, seed=14),
               PipelineSpec(mode="gradient", gradient_boundary=0.0)),
    GoldenCase("gradient_ring2", _base(halo_radius=78.0, halo_level=0.45, seed=14),
               PipelineSpec(mode="gradient", gradient_boundary=0.5)),
    GoldenCase("gradient_clean", _base(spot_radius=8.0, seed=15),
               PipelineSpec(mode="gradient")),
    GoldenCase("area_single", _base(spot_radius=8.0, seed=16),
               PipelineSpec(mode="area")),
    GoldenCase("area_multi", _base(debris=20, seed=17),
               PipelineSpec(mode="area", area_max_regions=3)),
    GoldenCase("area_clipped", _base(center=(30.0, 200.0), radius=70.0, seed=18),
               PipelineSpec(mode="area", edges=(True, False, False, False))),
)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def run_frame(path: Path, shape: tuple[int, int], pipe: PipelineSpec) -> dict:
    """Run the exact per-frame function used by Fit All and summarise its output."""
    from bubbletrack.controller.worker import _process_single_frame

    h, w = shape
    gridx, gridy = pipe.roi if pipe.roi is not None else ((1, h), (1, w))
    args = (
        0, str(path), pipe.threshold, gridx, gridy,
        list(pipe.edges), pipe.rf_slider, 0,
        pipe.sigma, pipe.clahe,
        pipe.closing, pipe.opening, pipe.mode, pipe.gradient_boundary,
        pipe.area_max_regions,
        float(max(h, w)),
    )
    _, radius, edge_xy, _, rc, cc, count = _process_single_frame(args)
    edge_xy = np.asarray(edge_xy, dtype=np.float64).reshape(-1, 2)
    return {
        "radius": _finite_or_none(radius),
        "rc": _finite_or_none(rc),
        "cc": _finite_or_none(cc),
        "count": int(count),
        "n_edge": int(edge_xy.shape[0]),
        "edge_sum": float(edge_xy.sum()),
    }


def run_case(case: GoldenCase, tmp_dir: Path) -> dict:
    frame = render_scene(case.scene)
    path = tmp_dir / f"{case.name}.tiff"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write {path}")
    return run_frame(path, frame.shape[:2], case.pipeline)


def real_frame_path(frame: int) -> Path:
    return REAL_FOLDER / f"Jin_17_01_56_{frame:03d}.tiff"


def run_real(frame: int, mode: str) -> dict:
    pipe = PipelineSpec(mode=mode, **OWNER_PARAMS)
    return run_frame(real_frame_path(frame), (250, 400), pipe)


def _generate() -> None:
    import tempfile

    golden: dict = {"commit": "d0fdbc7", "synthetic": {}, "example_1": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            golden["synthetic"][case.name] = run_case(case, Path(tmp))
    if REAL_FOLDER.is_dir():
        for frame in REAL_FRAMES:
            for mode in ("intensity", "gradient", "area"):
                golden["example_1"][f"{frame:03d}_{mode}"] = run_real(frame, mode)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} ({len(golden['synthetic'])} synthetic, "
          f"{len(golden['example_1'])} real)")


if __name__ == "__main__":
    _generate()
