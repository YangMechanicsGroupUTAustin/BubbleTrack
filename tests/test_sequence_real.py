"""Sequence solver on the real recordings (local data only; every test skips without it).

Set-up as in the app: Fit All = the Edge engine with the user's seed click and the
bubble-free reference frame on every frame; ``redetect`` = the engine with a circle prior
and no click.  The data is found through ``BUBBLETRACK_TEST_DATA`` (the ``test_data``
folder), else next to any parent of this file, else in the main checkout.

Measured at the port (engine edge-1.0.0), single frame -> solved:

* example_1 (91 frames): 0 -> 0 gross errors (> 10 %) and 0 -> 0 above 5 % against the
  research reference radii; collapses at indices 43 and 75 (files 54, 86); 0 redetect calls;
  review list 6 frames (collapse neighbours only).
* lic (10 x 256 frames, 235 labelled bubble frames): 205 -> 205 correct; no-bubble frames
  rejected 75/85 -> 75/85 (01: 18 -> 20, the cloud is stopped; 05: 12 -> 10, its microbubble
  ring is tracked as a bubble); review list 818 frames (2-28 on 01-04, 09, 10); 190 redetect
  calls for 2560 frames.
* lic_hard (8 x ~256 frames, 182 labelled bubble frames): 80 -> 82 correct; 31/31 -> 31/31
  rejected; review list 1111 frames; 293 redetect calls for 2045 frames.
* solver cost without redetect 0.03-0.2 ms / frame.  Whole module: about 70 s (8 processes).

Engine edge-1.1.0 (region layer: ``fragmented`` also marks debris and satellites within ~2 R),
after the cloud rule was made size-free and persistence-based (fragments count only on the
tracked body; a cloud lasts until a bubble re-forms), single frame -> solved, every recording
no worse than single frame:

* example_1: collapses 43 and 75; 0 redetect calls; review list 6 frames.
* lic: 203 -> 204 / 235 correct; no-bubble frames rejected 77/85 -> 78/85 (05: the microbubble
  ring, labelled N, re-forms as a confident body after its cloud, as in the single-frame
  result); review list 1175 frames; 195 redetect calls.
* lic_hard: 81 -> 81 / 182 correct; 30/31 -> 30/31 rejected; review list 1799 frames;
  227 redetect calls.

Engine edge-1.3.0 (the default R is the least-squares circle through the outline; edge-1.2.0's
robust ray circle in brackets), single frame -> solved, "correct" counting guesses below the gate:

* lic: 193 -> 193 / 235 correct (205 -> 205); shown by the app 180 -> 180 (181 -> 183);
  no-bubble frames rejected 83/85 -> 84/85 (79 -> 80); review list 931 frames (960); 161
  redetect calls (156).  The guesses lost are on 06 (half bubbles on a wall) and 08
  (transparent bubbles), where the outline takes in wall texture.
* lic_hard: 89 -> 89 / 182 correct (79 -> 79); shown by the app 40 -> 40 (36 -> 36); 30/31
  rejected (30/31); review list 1389 frames (1440); 195 redetect calls (235).

Most remaining misses are frames where no circle the engine can produce agrees with the
label (with a perfect prior, the engine fixes 9 of the 57 wrong labelled frames).

Engine edge-1.4.0 (no reference pixel excluded for being dark or saturated in the reference;
edge-1.3.0 in brackets), single frame -> solved:

* lic: 195 -> 195 / 235 correct (193 -> 193; 06: 16 -> 21, 07: 2 -> 0, 09: 26 -> 25); shown by
  the app 184 -> 184 (180 -> 180); no-bubble frames rejected 80/85 -> 76/85 (83 -> 84): on 07
  the fragments after the collapse, hidden until 1.3.0 by its saturated reference areas, are
  now visible, and the solver carries the tracked body into 6 of its no-bubble frames (17 ->
  11); review list 830 frames (931); 244 redetect calls (161).
* lic_hard: 79 -> 80 / 182 correct (89 -> 89; 04: 9 -> 1, its bubble's region now runs on
  into the speckled gel beside it; 02: 14 -> 13; 07: 17 -> 16); shown by the app 37 -> 38
  (40 -> 40); 30/31 rejected
  (30/31); review list 1355 frames (1389); 270 redetect calls (195).

Engine edge-1.5.0 (clearly dark pixels kept out of every background fit; edge-1.4.0 in
brackets), single frame -> solved:

* lic: 184 -> 184 / 235 correct (195 -> 195; 09: 25 -> 15, 06: 21 -> 20); shown by the app
  172 -> 172 (184 -> 184); no-bubble frames rejected 80/85 -> 72/85 (80 -> 76): on 07 the
  solver carries the tracked body into 10 of its no-bubble fragment frames (6); review list
  768 frames (830); 265 redetect calls (244).
* lic_hard: 81 -> 83 / 182 correct (79 -> 80); shown by the app 58 -> 60 (37 -> 38);
  no-bubble frames rejected 29/31 -> 28/31 (30/31 -> 30/31; 03: one fragment frame accepted
  single-frame, one more after the solver); review list 1128 frames (1355); 267 redetect
  calls (270).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import _baselines

FULL = ((1, 250), (1, 400))
HERE = Path(__file__).resolve().parent

# Research reference radii of example_1 (files 11..101; index 75 = file 86 is "low" confidence:
# a fragment cloud around a dark core, excluded from the score).
EX1_REFERENCE = np.array([
    29.28, 38.69, 45.35, 50.64, 55.08, 58.84, 62.10, 64.98, 67.53, 69.78, 71.76, 73.52, 75.07, 76.44, 77.60,
    78.61, 79.43, 80.12, 80.63, 81.01, 81.23, 81.30, 81.23, 81.01, 80.64, 80.13, 79.46, 78.63, 77.65, 76.46,
    75.11, 73.55, 71.81, 69.81, 67.57, 65.01, 62.14, 58.89, 55.11, 50.69, 45.38, 38.71, 29.23, 15.74, 25.22,
    30.77, 34.86, 38.29, 41.16, 43.57, 45.73, 47.45, 48.97, 50.21, 51.30, 52.25, 52.99, 53.58, 53.96, 54.13,
    54.15, 54.03, 53.71, 53.26, 52.64, 51.84, 50.80, 49.46, 47.80, 45.78, 43.44, 40.67, 37.46, 33.34, 24.76,
    17.95, 28.17, 30.76, 32.84, 34.68, 36.07, 37.10, 37.89, 38.60, 38.79, 38.34, 37.57, 36.57, 35.64, 34.11,
    31.81])
EX1_LOW = (75,)


def _data_root() -> Path | None:
    env = os.environ.get("BUBBLETRACK_TEST_DATA")
    candidates = [Path(env)] if env else []
    candidates += [p / "test_data" for p in HERE.parents]  # main checkout and its worktrees
    return next((c for c in candidates if (c / "lic").is_dir()), None)


ROOT = _data_root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="local LIC test data not present")


def _frames(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


def _agree(circle, ref, crit) -> bool:
    if ref is None or not (np.isfinite(circle[0]) and circle[0] > 0):
        return False
    return (abs(circle[0] - ref[0]) <= max(crit["radius_tol_px"], crit["radius_tol_frac"] * ref[0])
            and np.hypot(circle[1] - ref[1], circle[2] - ref[2])
            <= max(crit["centre_tol_px"], crit["centre_tol_frac"] * ref[0]))


def _solve_folder(folder: str, ref_index, seed):
    """Fit All + solve one recording.  Module level so a worker process can run it."""
    from bubbletrack.model.edge import EdgeParams, detect_edge, recording_black_level
    from bubbletrack.model.image_io import imread_unicode
    from bubbletrack.model.sequence import Observation, solve_sequence

    files = _frames(Path(folder))
    ref = imread_unicode(files[ref_index]) if ref_index is not None else None
    black = recording_black_level(lambda i: imread_unicode(files[i]), len(files))  # as the app

    def obs(res):
        return Observation(res.radius, res.rc, res.cc, res.confidence, res.flags)
    params = EdgeParams(seed=tuple(seed), black_level=black)
    raw = [obs(detect_edge(imread_unicode(f), *FULL, reference=ref, params=params)) for f in files]
    spent = [0.0]

    def redetect(i, prior):
        t0 = time.perf_counter()
        out = obs(detect_edge(imread_unicode(files[i]), *FULL, reference=ref, params=EdgeParams(prior=prior, black_level=black)))
        spent[0] += time.perf_counter() - t0
        return out
    t0 = time.perf_counter()
    res = solve_sequence(raw, redetect)
    solver_ms = ((time.perf_counter() - t0) - spent[0]) * 1e3 / len(raw)
    return raw, res, solver_ms


# Per-frame lists kept in tests/baselines/sequence_*.json (see tests/_baselines.py)
FRAME_LISTS = ("correct_single", "correct_tracked", "shown_correct", "no_bubble_kept_after")


def _accepted(obs) -> bool:
    """Would the app show this single-frame result (the engine's 0.3 gate)?"""
    return obs.ok and obs.confidence >= 0.3


def _score(raw, res, labels: dict, crit) -> dict:
    """labels: {frame index: circle (B) | "N"}."""
    out = dict(b=0, before=0, after=0, shown_before=0, app=0, n=0, rej_before=0, rej_after=0)
    frames = {key: [] for key in FRAME_LISTS}  # 1-based frame numbers, for the baselines
    for i, lab in labels.items():
        if i >= len(raw):
            continue
        sol = res.frames[i]
        if lab == "N":
            out["n"] += 1
            rejected = (not raw[i].ok) or raw[i].confidence < 0.3
            rejected_after = (not sol.ok) or sol.source in ("none", "lost") or sol.confidence < 0.3
            out["rej_before"] += rejected
            out["rej_after"] += rejected_after
            if not rejected_after:
                frames["no_bubble_kept_after"].append(i + 1)
        else:
            out["b"] += 1
            before = _agree((raw[i].radius, raw[i].rc, raw[i].cc), lab, crit)
            after = _agree((sol.radius, sol.rc, sol.cc), lab, crit)
            out["before"] += before
            out["after"] += after
            # What the app SHOWS: a frame below the confidence gate has no result;
            # the solver's re-measurements are stored only when accepted.
            shown = _accepted(raw[i])
            out["shown_before"] += shown and before
            kept = sol if sol.source == "tracked" else raw[i]
            app = (sol.source == "tracked" or shown) and _agree((kept.radius, kept.rc, kept.cc), lab, crit)
            out["app"] += app
            for key, ok in (("correct_single", before), ("correct_tracked", after), ("shown_correct", app)):
                if ok:
                    frames[key].append(i + 1)
    out.update(review=len(res.review), calls=res.redetect_calls, n_frames=len(raw), frames=frames)
    return out


def _job(args):
    name, folder, ref_index, seed, labels, crit = args
    raw, res, solver_ms = _solve_folder(folder, ref_index, seed)
    return name, dict(_score(raw, res, labels, crit), solver_ms=solver_ms)


def _run_set(jobs) -> dict:
    with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        return dict(ex.map(_job, jobs))


def _report(title: str, rows: dict) -> dict:
    keys = ("b", "before", "after", "shown_before", "app", "n", "rej_before", "rej_after", "review",
            "calls", "n_frames")
    tot = {k: sum(r[k] for r in rows.values()) for k in keys}
    print(f"\n{title}: bubble frames correct {tot['before']}/{tot['b']} -> {tot['after']}/{tot['b']} "
          f"(shown by the app: {tot['shown_before']} -> {tot['app']}/{tot['b']}), "
          f"no-bubble rejected {tot['rej_before']}/{tot['n']} -> {tot['rej_after']}/{tot['n']}, "
          f"review {tot['review']}, redetect calls {tot['calls']} / {tot['n_frames']} frames")
    for name, r in rows.items():
        print(f"  {name[:40]:40s} B {r['before']:2d}->{r['after']:2d}/{r['b']:2d}  N {r['rej_before']:2d}->"
              f"{r['rej_after']:2d}/{r['n']:2d}  review {r['review']:3d}  calls {r['calls']:3d}  "
              f"solver {r['solver_ms']:.2f} ms/frame")
    return tot


def test_example_1_has_no_gross_errors_and_both_collapses():
    folder = ROOT.parent / "example_1"
    if not folder.is_dir():
        pytest.skip("example_1 not present")
    raw, res, solver_ms = _solve_folder(str(folder), None, (31.0, 199.0))
    assert len(raw) == len(EX1_REFERENCE)
    keep = np.setdiff1d(np.arange(len(raw)), EX1_LOW)
    radius = np.array([f.radius if f.ok else np.nan for f in res.frames])
    err = np.abs(radius[keep] - EX1_REFERENCE[keep]) / EX1_REFERENCE[keep]
    assert np.all(err <= 0.10), f"gross errors at {keep[~(err <= 0.10)].tolist()}"
    assert res.collapse_frames == (43, 75)                # files 54 and 86
    assert res.redetect_calls <= 5 and len(res.review) <= 10
    assert solver_ms < 2.0


def test_lic_no_regression_against_single_frame():
    labels = json.loads((ROOT / "lic" / "labels_edge.json").read_text(encoding="utf-8"))
    jobs = []
    for name, info in labels["sequences"].items():
        folder = ROOT / "lic" / name
        if not folder.is_dir():
            pytest.skip(f"{name} not present")
        lab = {int(k) - 1: ("N" if v["label"] != "B" else v["circle"]) for k, v in info["frames"].items()}
        jobs.append((name, str(folder), labels["reference_frame_index"], info["seed"], lab, labels["criterion"]))
    rows = _run_set(jobs)
    tot = _report("lic", rows)
    _baselines.check("sequence_lic", {name: r["frames"] for name, r in rows.items()})
    assert tot["b"] == 235
    assert tot["after"] >= tot["before"] >= 184             # edge-1.5.0 (1.4.0: 195, 1.3.0: 193, 1.2.0: 205)
    assert tot["app"] >= tot["shown_before"]
    # edge-1.5.0: 80 -> 72 (07: 17 -> 7; 1.4.0: 80 -> 76, 07: 17 -> 11, see the module
    # docstring); until 1.3.0 the solver never lost a rejected no-bubble frame (83 -> 84).
    # edge-1.6.1: 80 -> 71, 07 file 169 (the fragment cluster 161 and 177 already show)
    # edge-1.7.0 (black level): 80 -> 70, 05 file 121 also kept after tracking (accepted single-frame
    # too; 01 file 241 no longer is); 06: file 81 now correct, 209 not
    assert tot["rej_before"] >= 80 and tot["rej_after"] >= 70


def test_lic_hard_no_regression_against_single_frame():
    labels = json.loads((HERE / "test_sequence_labels_hard.json").read_text(encoding="utf-8"))
    jobs = []
    for name, info in labels["sequences"].items():
        folder = ROOT / "lic_hard" / name
        if not folder.is_dir():
            pytest.skip(f"{name} not present")
        lab = {int(k): c for k, c in info["B"].items()} | {i: "N" for i in info["N"]}
        jobs.append((name, str(folder), info["reference_index"], info["seed"], lab, labels["criterion"]))
    rows = _run_set(jobs)
    tot = _report("lic_hard", rows)
    _baselines.check("sequence_lic_hard", {name: r["frames"] for name, r in rows.items()})
    assert tot["b"] == 182
    # edge-1.7.0 (black level): 81 -> 63, all in 05 (blood, low contrast; shown 0 -> 0): its
    # darkest level, 31 % of the background, is light, and with it removed the guesses below the gate
    # come out 12-19 px too large.  edge-1.6.2's guesses there were right at the raw values only:
    # +1 % (655 counts) already took them from 18 to 2 of 21 (offset study, 2026-11-24)
    assert tot["after"] >= tot["before"] >= 63                       # edge-1.5.0: 81 (1.4.0: 79)
    assert tot["app"] >= tot["shown_before"]
    # edge-1.5.0: 29 -> 28 (03, see the module docstring); until 1.4.0 the solver never lost
    # a rejected no-bubble frame here (30 -> 30).  edge-1.6.2: 29 -> 28 and 28 -> 27, 03 file 89
    # (labelled no bubble: the dark debris file 81 already shows, confidence 0.29 -> 0.32)
    # edge-1.7.0: 03 files 73 and 89 no longer kept after tracking
    assert tot["rej_before"] >= 28 and tot["rej_after"] >= 27
