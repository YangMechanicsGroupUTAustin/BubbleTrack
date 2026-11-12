"""Shared pieces of the ML assist study: recordings, labels, scoring, simulated
examples (verified frames and thin strokes) and the choice of taught frames.

Data: ``test_data/lic`` and ``test_data/lic_hard`` (labels: ``test_data/lic/labels_edge.json``,
``tests/test_sequence_labels_hard.json``) and ``example_1`` (radius-only reference), found
as in ``tests/test_sequence_real.py`` (read only).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "tests", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_sequence_real as T  # noqa: E402

import bubbletrack  # noqa: E402
from bubbletrack.model.image_io import imread_unicode  # noqa: E402
from bubbletrack.model.teach import BACKGROUND, BUBBLE, MLExamples, stroke_pixels, suggest_frames  # noqa: E402

assert Path(bubbletrack.__file__).resolve().is_relative_to(REPO / "src"), bubbletrack.__file__
FULL = T.FULL
CRIT = {"radius_tol_px": 2.0, "radius_tol_frac": 0.05, "centre_tol_px": 3.0, "centre_tol_frac": 0.15}
GOOD_FOR_AUTO = ("lic/01", "lic/02", "lic/03", "lic/04", "lic/09", "lic/10", "example_1")
EX1_STRIDE = 3                      # example_1: every 3rd frame is scored (30 frames)


@dataclass(frozen=True)
class Recording:
    key: str                         # e.g. "lic/06", "lic_hard/05", "example_1"
    name: str
    folder: str
    seed: tuple[float, float]
    ref_index: int | None
    labels: dict = field(default_factory=dict)
    """{frame index: (R, rc, cc)} (1-based coords); example_1: (R, nan, nan)."""
    radius_only: bool = False

    def files(self) -> list[Path]:
        return T._frames(Path(self.folder))

    def reference(self, files=None):
        files = files or self.files()
        return imread_unicode(files[self.ref_index]) if self.ref_index is not None else None


def recordings() -> list[Recording]:
    out = []
    lic = json.loads((T.ROOT / "lic" / "labels_edge.json").read_text(encoding="utf-8"))
    for name, info in lic["sequences"].items():
        lab = {int(k) - 1: tuple(v["circle"]) for k, v in info["frames"].items()
               if v["label"] == "B" and v.get("circle") is not None}
        out.append(Recording(f"lic/{name[:2]}", name, str(T.ROOT / "lic" / name), tuple(info["seed"]),
                             lic["reference_frame_index"], lab))
    hard = json.loads((REPO / "tests" / "test_sequence_labels_hard.json").read_text(encoding="utf-8"))
    for name, info in hard["sequences"].items():
        lab = {int(k): tuple(c) for k, c in info["B"].items() if c is not None}
        out.append(Recording(f"lic_hard/{name[:2]}", name, str(T.ROOT / "lic_hard" / name), tuple(info["seed"]),
                             info["reference_index"], lab))
    ex1 = T.ROOT.parent / "example_1"
    if ex1.is_dir():
        lab = {i: (float(T.EX1_REFERENCE[i]), np.nan, np.nan) for i in range(0, len(T.EX1_REFERENCE), EX1_STRIDE)
               if i not in T.EX1_LOW}
        out.append(Recording("example_1", "example_1", str(ex1), (31.0, 199.0), None, lab, radius_only=True))
    return out


def agree(circle, label, radius_only: bool = False) -> bool:
    """The label criterion; *circle* = (R, rc, cc)."""
    if circle is None or not (np.isfinite(circle[0]) and circle[0] > 0):
        return False
    tol_r = max(CRIT["radius_tol_px"], CRIT["radius_tol_frac"] * label[0])
    if abs(circle[0] - label[0]) > tol_r:
        return False
    if radius_only:
        return True
    return float(np.hypot(circle[1] - label[1], circle[2] - label[2])) <= max(
        CRIT["centre_tol_px"], CRIT["centre_tol_frac"] * label[0])


# --------------------------------------------------------------------------- #
#  Which frames the user teaches
# --------------------------------------------------------------------------- #

def choose_frames(strategy: str, n: int, labels: dict, auto_conf: dict,
                  auto_radius: dict | None = None) -> tuple[int, ...]:
    """*n* taught frames among the labelled ones.

    largest: the n biggest bubbles; spread: the biggest, then sizes spread down to
    the 20th percentile; worst: lowest Auto confidence (what Review lists first);
    time: evenly spaced in time; suggest: the package's ``suggest_frames`` on Auto's
    own results (what the app can do without labels).
    """
    frames = sorted(labels)
    if n >= len(frames):
        return tuple(frames)
    if strategy == "suggest":
        radius = [(auto_radius or {}).get(i, np.nan) for i in frames]
        conf = [auto_conf.get(i, 0.0) for i in frames]
        return tuple(sorted(frames[k] for k in suggest_frames(radius, conf, n=n)))
    by_r = sorted(frames, key=lambda i: (labels[i][0], -i))
    if strategy == "largest":
        pick = by_r[::-1][:n]
    elif strategy == "spread":
        pos = np.round(np.linspace(len(by_r) - 1, 0.2 * (len(by_r) - 1), n)).astype(int)
        pick = [by_r[p] for p in pos]
    elif strategy == "worst":
        pick = sorted(frames, key=lambda i: (auto_conf.get(i, 0.0), i))[:n]
    elif strategy == "time":
        pos = np.round(np.linspace(0, len(frames) - 1, n + 2)[1:-1]).astype(int)
        pick = [frames[p] for p in pos]
    else:
        raise ValueError(strategy)
    out = []
    for f in pick:                                   # de-duplicate, keep order
        if f not in out:
            out.append(f)
    return tuple(sorted(out))


# --------------------------------------------------------------------------- #
#  Simulated examples
# --------------------------------------------------------------------------- #

def verified_examples(circles: dict) -> MLExamples:
    """The user verified these frames: {frame: (R, rc, cc)} -> verified circles (rc, cc, R)."""
    ex = MLExamples()
    for f, (r, rc, cc) in sorted(circles.items()):
        ex = ex.verify(f, (rc, cc, r))
    return ex


def _arc(rc, cc, radius, a0, span, n=10):
    t = np.linspace(a0, a0 + span, n)
    return list(zip(rc + radius * np.sin(t), cc + radius * np.cos(t)))


def stroke_examples(circles: dict, shape: tuple[int, int], brush: int = 2) -> MLExamples:
    """A few thin strokes per frame, as a user paints them (not dense regions).

    Bubble: two crossing lines through the centre (+-0.5 R and +-0.35 R).
    Background: three 50-degree arcs at 1.5 R and one long line far from the bubble.
    """
    ex = MLExamples()
    h, w = shape
    for f, (r, rc, cc) in sorted(circles.items()):
        rng = np.random.default_rng(1000 + f)
        th = rng.uniform(0, np.pi)
        for half, ang in ((0.5, th), (0.35, th + np.pi / 2)):
            d = half * r * np.array([np.sin(ang), np.cos(ang)])
            ex = ex.paint(f, BUBBLE, stroke_pixels([(rc - d[0], cc - d[1]), (rc + d[0], cc + d[1])], brush, shape))
        a0 = rng.uniform(0, 2 * np.pi)
        for k in range(3):
            ex = ex.paint(f, BACKGROUND, stroke_pixels(_arc(rc, cc, 1.5 * r, a0 + k * 2 * np.pi / 3, np.radians(50)),
                                                       brush, shape))
        far = _far_line(rc, cc, r, h, w)
        if far is not None:
            ex = ex.paint(f, BACKGROUND, stroke_pixels(far, brush, shape))
        if f not in ex.strokes or len(ex.strokes[f].background) < 50:
            for line in _corner_lines(rc, cc, r, h, w):      # a huge bubble: paint what is visible
                ex = ex.paint(f, BACKGROUND, stroke_pixels(line, brush, shape))
    return ex


def _corner_lines(rc, cc, r, h, w):
    """Short strokes from the frame corners towards the centre, stopping at 1.3 R."""
    out = []
    for cr, ccol in ((1.0, 1.0), (1.0, float(w)), (float(h), 1.0), (float(h), float(w))):
        d = float(np.hypot(cr - rc, ccol - cc))
        if d > 1.35 * r:
            t = min(1.0, (d - 1.3 * r) / d)
            out.append([(cr, ccol), (cr + t * (rc - cr), ccol + t * (cc - ccol))])
    return out


def _far_line(rc, cc, r, h, w):
    """A horizontal or vertical line at least 2.5 R from the centre, if the frame has room."""
    options = []
    for row in (rc - 2.5 * r, rc + 2.5 * r):
        if 5 <= row <= h - 5:
            options.append([(row, 0.2 * w), (row, 0.8 * w)])
    for col in (cc - 2.5 * r, cc + 2.5 * r):
        if 5 <= col <= w - 5:
            options.append([(0.2 * h, col), (0.8 * h, col)])
    return options[0] if options else None


def circle_dict(obj) -> dict | None:
    """JSON-able summary of an EdgeResult / MLRegion (None stays None)."""
    if obj is None:
        return None
    out = {"R": float(obj.radius), "rc": float(obj.rc), "cc": float(obj.cc), "conf": float(obj.confidence)}
    if hasattr(obj, "accepted"):
        out["acc"] = bool(obj.accepted)
    if hasattr(obj, "terms"):
        out.update(clipped=bool(obj.clipped), refined=bool(obj.refined),
                   terms={k: round(float(v), 4) for k, v in obj.terms.items()})
    return out
