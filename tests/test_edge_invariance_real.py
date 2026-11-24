"""Auto must not change with how a recording was exported (edge-1.7.0; owner, 2026-11-24).

A result that changes with the export is not automatic: 655 counts are nothing in 16 bit.  A
colleague who imported the HPV-X TIFF16 export of Jin 2026-11-16 14:37:03 (every pixel + 655, a
black info stamp) lost half the big-bubble frames that the .dat shows (edge-1.6.2: 14/28).

Each check runs Auto as the app does: the recording's black level (0 for a .dat, else estimated),
the automatic reference frame, app defaults (full-frame ROI, no seed), the confidence gate.  A
frame has the *same result* in two runs when both show nothing, or both show a circle with radii
within 2 px and centres within 3 px.  The frames that differ are tracked in
``tests/baselines/invariance_*.json`` (see ``_baselines``): any frame that moves fails until the
file is regenerated (``BUBBLETRACK_UPDATE_BASELINES=1``) and committed with the change.

Local real data only (``test_data``, git-excluded; found as in test_edge_real), else skipped:
``dat/Jin_14_37_03.dat`` with ``dat/Jin_14_37_03_tiff16_export/`` and the LIC sets.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

import _baselines
from bubbletrack.model import edge_pipeline

WORKERS = 2  # the owner's RAM budget (<= 5 GB for all background work)
OFFSET_FRACTION = 0.01  # +1 % of the dtype's full scale (655 counts in 16 bit)


def _data_root() -> Path | None:
    env = os.environ.get("BUBBLETRACK_TEST_DATA")
    candidates = [Path(env)] if env else []
    candidates += [p / "test_data" for p in Path(__file__).resolve().parents]
    return next((c for c in candidates if (c / "lic").is_dir() or (c / "dat").is_dir()), None)


ROOT = _data_root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="local real test data not present")


# -- the perturbations ------------------------------------------------------------------- #

def offset(frame: np.ndarray) -> np.ndarray:
    """*frame* plus 1 % of its dtype's full scale, clipped (an export's constant offset)."""
    top = np.iinfo(frame.dtype).max
    return np.minimum(frame.astype(np.int64) + round(OFFSET_FRACTION * top), top).astype(frame.dtype)


def to_8bit(frame: np.ndarray) -> np.ndarray:
    """A 16-bit frame as an 8-bit export writes it (top byte)."""
    return (frame >> 8).astype(np.uint8)


PERTURB = {"orig": None, "offset_1pct": offset, "8bit": to_8bit}


# -- Auto as the app runs it ------------------------------------------------------------------ #

def _tiffs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


def _reader(source: str, kind: str):
    """(read(i), n_frames, source name for the black level) of a .dat or an image folder."""
    if source.lower().endswith(".dat"):
        from bubbletrack.model.hpvx2_dat import DatFrameReader

        reader = DatFrameReader(source)
        return reader.read_frame, reader.total_frames, source
    files = _tiffs(Path(source))
    perturb = PERTURB[kind]

    def read(i: int) -> np.ndarray:
        raw = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED)
        return raw if perturb is None else perturb(raw)

    return read, len(files), ""


def run_auto(args) -> dict:
    """Auto on every *step*-th frame of *source* under perturbation *kind* (a worker task)."""
    source, kind, step = args
    read, n, name = _reader(source, kind)
    black = edge_pipeline.recording_black_level(read, n, name)
    ref = edge_pipeline.resolve_reference_index(read, n, None, black_level=black)
    h, w = read(0).shape[:2]
    job = edge_pipeline.EdgeJob(params=edge_pipeline.make_params(black_level=black),
                                reference=read(ref) if ref is not None else None,
                                max_radius=float(max(h, w)))
    frames = {}
    for i in range(0, n, step):
        fit = edge_pipeline.fit_edge(read(i), (1, h), (1, w), job)
        frames[i] = (fit.radius, fit.rc, fit.cc) if fit.radius > 0 else None
    return {"black": black, "reference": ref, "frames": frames}


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a[0] - b[0]) <= 2.0 and np.hypot(a[1] - b[1], a[2] - b[2]) <= 3.0


def differing(x: dict, y: dict) -> list[int]:
    """1-based frames whose results differ between runs *x* and *y*."""
    return [i + 1 for i in x["frames"] if not same(x["frames"][i], y["frames"][i])]


def _run_all(tasks: list) -> list[dict]:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(run_auto, tasks))


# -- the colleague's case ------------------------------------------------------------------- #

BIG = range(10, 66)  # 0-based frames 11-66 of 14:37:03: the bubble is larger than R = 100 px


def test_the_tiff16_export_of_14_37_03_gives_the_dat_result():
    """Measured (edge-1.7.0, every 2nd frame, 128): black level .dat 0 / TIFF 943 (655 + the darkest
    light); same result on 116/128 frames; big-bubble frames 11-66 shown .dat 27/28, TIFF 26/28
    (edge-1.6.2: 14/28).  The frames that differ are mostly radii 2-3 px apart: the TIFF's black
    level also removes the recording's darkest light (288 counts), which the .dat keeps."""
    dat = ROOT / "dat" / "Jin_14_37_03.dat"
    export = ROOT / "dat" / "Jin_14_37_03_tiff16_export"
    if not (dat.is_file() and export.is_dir()):
        pytest.skip("Jin_14_37_03 .dat or its TIFF16 export not present")
    a, b = _run_all([(str(dat), "orig", 2), (str(export), "orig", 2)])
    assert a["black"] == 0.0                                  # a .dat holds the sensor's own values
    assert b["black"] >= 655                                  # the export's offset (+ the darkest light)
    assert a["reference"] == b["reference"]
    diff = differing(a, b)
    big = [i for i in BIG if i in b["frames"]]
    shown = {k: sum(r["frames"][i] is not None for i in big) for k, r in (("dat", a), ("tiff", b))}
    print(f"\n14:37:03 black .dat {a['black']:g} / TIFF {b['black']:g}; same {len(a['frames']) - len(diff)}"
          f"/{len(a['frames'])}; big-bubble frames shown .dat {shown['dat']}/{len(big)}, "
          f"TIFF {shown['tiff']}/{len(big)}; differ {diff}")
    _baselines.check("invariance_14_37_03", {"Jin_14_37_03": {"dat_vs_tiff16": diff}})
    assert shown["tiff"] >= 26 and shown["dat"] >= 27
    assert len(a["frames"]) - len(diff) >= 116


# -- the LIC sets ------------------------------------------------------------------------------ #

def _lic_folders() -> list[Path]:
    folders = [p for group in ("lic", "lic_hard") if (ROOT / group).is_dir()
               for p in sorted((ROOT / group).iterdir()) if p.is_dir() and p.name[:2].isdigit()]
    if not folders:
        pytest.skip("LIC test sequences not present")
    return folders


def test_offset_and_8bit_copies_of_the_lic_sets_give_the_same_results():
    """Measured (edge-1.7.0, every 4th frame of 18 sequences): +1 % offset same on 1111/1152
    frames (96.4 %), 8 bit (12 sixteen-bit sequences) 751/768 (97.8 %).  Every reference frame
    is chosen the same."""
    tasks, keys = [], []
    for folder in _lic_folders():
        bit16 = cv2.imread(str(_tiffs(folder)[0]), cv2.IMREAD_UNCHANGED).dtype == np.uint16
        for kind in PERTURB if bit16 else ("orig", "offset_1pct"):
            tasks.append((str(folder), kind, 4))
            keys.append((f"{folder.parent.name}/{folder.name}", kind))
    runs = dict(zip(keys, _run_all(tasks)))
    found, totals = {}, {}
    for (seq, kind), run in runs.items():
        if kind == "orig":
            continue
        base = runs[(seq, "orig")]
        assert run["reference"] == base["reference"], (seq, kind)
        diff = differing(base, run)
        found.setdefault(seq, {})[kind] = diff
        n, k = totals.get(kind, (0, 0))
        totals[kind] = (n + len(base["frames"]), k + len(base["frames"]) - len(diff))
    print("\n" + "; ".join(f"{kind}: same {k}/{n} ({100 * k / n:.1f} %)" for kind, (n, k) in totals.items()))
    _baselines.check("invariance_lic", found)
    assert totals["offset_1pct"][1] >= 1111 and totals["8bit"][1] >= 751
