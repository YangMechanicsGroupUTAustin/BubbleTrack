"""Real-data regression of the Edge engine on the local LIC test sequences.

The data (``test_data/lic``, ``test_data/lic_hard``, ``example_1``) is not in the repository.
It is found through the ``BUBBLETRACK_TEST_DATA`` environment variable (the ``test_data``
folder), else by looking for ``test_data/lic`` next to any parent of this file (works from
the main checkout and from a ``.claude/worktrees/*`` worktree).  Every test skips without it.

Mode: P2 + reference (frame 2, i.e. index 1) + seed click, single frame (no tracking).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from bubbletrack.model.edge import EdgeParams, choose_reference_frame, detect_edge
from bubbletrack.model.image_io import imread_unicode


def _data_root() -> Path | None:
    env = os.environ.get("BUBBLETRACK_TEST_DATA")
    candidates = [Path(env)] if env else []
    candidates += [p / "test_data" for p in Path(__file__).resolve().parents]
    return next((c for c in candidates if (c / "lic").is_dir()), None)


ROOT = _data_root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="local LIC test data not present")
FULL = ((1, 250), (1, 400))


def _seq_dir(group: str, prefix: str) -> Path:
    found = sorted((ROOT / group).glob(f"{prefix}_*"))
    if not found:
        pytest.skip(f"{group}/{prefix}_* not present")
    return found[0]


def _frames(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


def _labels() -> dict:
    path = ROOT / "lic" / "labels_edge.json"
    if not path.exists():
        pytest.skip("labels_edge.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


def _series(prefix: str, ref_r: np.ndarray, ref_c: np.ndarray, step: int = 2, **params):
    labels = _labels()
    folder = _seq_dir("lic", prefix)
    files = _frames(folder)
    ref = imread_unicode(files[labels["reference_frame_index"]])
    edge_params = EdgeParams(seed=tuple(labels["sequences"][folder.name]["seed"]), **params)
    d_r, d_c, n_shoulder = [], [], 0
    for i in range(0, len(ref_r), step):
        if not np.isfinite(ref_r[i]):
            continue
        res = detect_edge(imread_unicode(files[i]), *FULL, reference=ref, params=edge_params)
        if np.isfinite(res.radius):
            d_r.append(res.radius - ref_r[i])
            d_c.append(np.hypot(res.rc - ref_c[i, 0], res.cc - ref_c[i, 1]))
            n_shoulder += "shoulder" in res.flags
    return np.abs(np.array(d_r)), np.array(d_c), n_shoulder


def _r_data(prefix: str):
    sio = pytest.importorskip("scipy.io")
    m = sio.loadmat(_seq_dir("lic", prefix) / "R_data.mat")
    r = m["CircleR"].ravel().astype(float)
    return np.where(np.isfinite(r) & (r > 0), r, np.nan), m["CircleCenterSave"].astype(float)


def test_seq04_agrees_with_the_2024_matlab_result():
    """Measured: 127 frames, median |dR| 0.085 px, median centre distance 0.32 px."""
    d_r, d_c, _ = _series("04", *_r_data("04"))
    assert len(d_r) > 100
    assert np.median(d_r) < 0.3
    assert np.median(d_c) < 1.0          # both 1-based: the coordinate convention holds on real data


def test_seq02_outer_extent_matches_the_2024_outer_edge():
    """02 has a wide shoulder; the 2024 MATLAB result followed its outer extent (definition B).

    Measured (every 4th frame, 63 frames): steep radius 6.3 px inside the 2024 value, outer
    radius +0.9 px (median |dR| 0.9), median shoulder 7.3 px, flag ``shoulder`` on 63/63.
    """
    d_r, _, n_shoulder = _series("02", *_r_data("02"), step=4, edge_definition="outer")
    assert len(d_r) > 50
    assert np.median(d_r) < 1.5
    assert n_shoulder >= 0.9 * len(d_r)


def test_seq10_agrees_with_the_2026_app_result():
    """Measured: 128 frames, median |dR| 1.02 px (the app's edge sits ~1 px outside), centre 0.07 px."""
    sio = pytest.importorskip("scipy.io")
    path = ROOT / "lic" / "refs" / "10_gelatin_Rt_132046.mat"
    if not path.exists():
        pytest.skip("reference R(t) file not present")
    m = sio.loadmat(path)
    r = m["Radius"].ravel().astype(float)
    r = np.where(np.isfinite(r) & (r > 0), r, np.nan)
    d_r, d_c, _ = _series("10", r, m["CircleFitPar"].astype(float))
    assert len(d_r) > 100
    assert np.median(d_r) < 1.5
    assert np.median(d_c) < 1.0


def _agree(circle, ref, crit) -> bool:
    return (abs(circle[0] - ref[0]) <= max(crit["radius_tol_px"], crit["radius_tol_frac"] * ref[0])
            and np.hypot(circle[1] - ref[1], circle[2] - ref[2])
            <= max(crit["centre_tol_px"], crit["centre_tol_frac"] * ref[0]))


def test_labelled_bubble_frames_on_lic():
    """Visual labels of the real10 scorecard (32 sampled frames x 10 sequences, 235 bubble frames).

    Measured at the port (edge-1.0.0): 205/235 = 87.2 % correct (same frames as proto3); no-bubble
    frames rejected (confidence < 0.3) 75/85.  For comparison, prototype P2 without the
    saturated-speckle fix scored 214/235 (91.1 %): the fix stops masking isolated saturated
    speckle grains, which costs frames on the uint16 bright-speckle sequences 06 and 08 and
    helps the 8-bit dense-speckle hard sequences.
    """
    labels = _labels()
    crit = labels["criterion"]
    ok = n_b = 0
    for name, info in labels["sequences"].items():
        folder = ROOT / "lic" / name
        if not folder.is_dir():
            pytest.skip(f"{name} not present")
        files = _frames(folder)
        ref = imread_unicode(files[labels["reference_frame_index"]])
        params = EdgeParams(seed=tuple(info["seed"]))
        for frame_no, lab in info["frames"].items():
            if lab["label"] != "B":
                continue
            n_b += 1
            res = detect_edge(imread_unicode(files[int(frame_no) - 1]), *FULL, reference=ref, params=params)
            if lab["circle"] is not None and np.isfinite(res.radius):
                ok += _agree((res.radius, res.rc, res.cc), lab["circle"], crit)
    assert n_b == 235
    assert ok / n_b >= 0.85, f"{ok}/{n_b}"


def test_reference_frame_choice_on_real_recordings():
    """lic: frame 2 (index 1), except 01 whose flash comes at frame 10 (index 9) -> index 8.
    example_1 and lic_hard 01 start with a bubble -> None."""
    expected = {f"{k:02d}": 1 for k in range(2, 11)}
    expected["01"] = 8
    for prefix, want in expected.items():
        files = _frames(_seq_dir("lic", prefix))
        assert choose_reference_frame(lambda i: imread_unicode(files[i]), len(files)) == want, prefix
    for folder in (ROOT.parent / "example_1", *sorted((ROOT / "lic_hard").glob("01_*"))):
        if folder.is_dir():
            files = _frames(folder)
            assert choose_reference_frame(lambda i: imread_unicode(files[i]), len(files)) is None, folder.name
