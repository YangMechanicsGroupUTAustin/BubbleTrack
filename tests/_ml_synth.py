"""A small synthetic recording for ML assist: Auto measures the clean frames and
fails on the low-contrast speckled ones, where ML assist's own outline works.

Frame 0 has no bubble (the reference).  Circles are 1-based ``(rc, cc, R)``.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from _edge_synth import Bubble, Scene, render

H, W = 120, 160
CENTRE = (60.0, 80.0)  # 0-based
RADII = (0.0, 14.0, 18.0, 22.0, 26.0, 30.0, 26.0, 22.0, 18.0, 14.0)
HARD = frozenset({5, 6, 7, 8, 9})  # low contrast + speckle that the reference lacks
CLEAN_TRANS, HARD_TRANS, HARD_SPECKLE = 0.07, 0.6, 0.5


def circle(frame: int) -> tuple[float, float, float]:
    """The true circle of *frame*, 1-based (rc, cc, R)."""
    return CENTRE[0] + 1.0, CENTRE[1] + 1.0, RADII[frame]


def scene(frame: int) -> Scene:
    hard = frame in HARD
    bubbles = () if RADII[frame] <= 0 else (Bubble(
        rc=CENTRE[0], cc=CENTRE[1], r=RADII[frame], trans=HARD_TRANS if hard else CLEAN_TRANS,
        spot=False, ring=False),)
    return Scene(h=H, w=W, bubbles=bubbles, speckle=HARD_SPECKLE if hard else 0.0, spk_scale=2.0,
                 seed=100 + frame)


def frames() -> list:
    return [render(scene(i))[0] for i in range(len(RADII))]


def write_recording(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(frames()):
        if not cv2.imwrite(str(folder / f"ml_{i + 1:03d}.tiff"), img):
            raise OSError(f"could not write frame {i} to {folder}")
    return folder
