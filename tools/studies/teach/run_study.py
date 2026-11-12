"""ML assist study: per-frame records for every (recording, configuration).

Phases (run in order; each writes under ``--out``):

* ``auto``   plain Auto (seed click + reference) on every labelled frame -> ``auto/<rec>.json``;
* ``grid``   train on the chosen frames, measure every other labelled frame -> ``grid/<rec>__<cfg>.jsonl``;
* ``confirm`` the package's ``measure_frame`` with its final defaults on the key configurations
  -> ``confirm/<rec>__<cfg>.jsonl`` (the grid records were taken with the first refinement settings);
* ``cross``  novelty of a model on frames of its own and of the other recordings -> ``cross.json``;
* ``timing`` training / prediction times in one process, nothing else running -> ``timing.json``.

Then ``analyze.py`` turns the records into the tables of ``RESULTS.md``.

    python tools/studies/teach/run_study.py --phase all --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import common as C
from bubbletrack.model.edge import EdgeParams, detect_edge
from bubbletrack.model.edge.engine import engine_view
from bubbletrack.model.image_io import imread_unicode
from bubbletrack.model.teach import (
    ExampleConfig,
    FeatureConfig,
    ForestConfig,
    MLConfig,
    MLError,
    measure_frame,
    ml_region,
    select_result,
    train_model,
)

DEFAULT_OUT = Path(tempfile.gettempdir()) / "bubbletrack_teach_study"
FEATS = {"rawview": {}, "raw": {"channels": ("raw",)}, "view": {"channels": ("view",)},
         "rawview_half": {"half_resolution": True}}
REFINE_VARIANTS = {"reg_k1": {"refine_sectors": 1}, "reg_k4": {"refine_sectors": 4},
                   "reg_b10": {"refine_band": 0.10}, "reg_b25": {"refine_band": 0.25},
                   "reg_2p": {"refine_band": 0.2, "refine_passes": 2}}
MARGINS = {
    "m_in06": {"inner": 0.6}, "m_in09": {"inner": 0.9}, "m_r115": {"ring": (1.15, 2.0)},
    "m_r15": {"ring": (1.5, 2.5), "far": 3.0}, "m_nofar": {"far": None},
    "m_r3nofar": {"ring": (1.25, 3.0), "far": None},
}


def grid_configs() -> list[dict]:
    out = []

    def add(source, strategy, n, feats="rawview", forest="extra", margins=None):
        cid = f"{source}-{strategy}-{n}-{feats}-{forest}" + (f"-{margins}" if margins else "")
        out.append(dict(id=cid, source=source, strategy=strategy, n=n, feats=feats, forest=forest, margins=margins))

    for source in ("verified", "strokes"):
        for strategy in ("largest", "spread", "worst", "time"):
            for n in (1, 2, 3, 5):
                if not (strategy == "spread" and n == 1):          # = largest-1
                    add(source, strategy, n)
    for feats in ("raw", "view", "rawview_half"):
        add("verified", "spread", 2, feats)
        add("verified", "spread", 3, feats)
        add("strokes", "spread", 2, feats)
    add("verified", "spread", 2, forest="random")
    add("verified", "spread", 3, forest="random")
    add("strokes", "spread", 2, forest="random")
    for m in MARGINS:
        add("verified", "spread", 2, margins=m)
    return out


def _ml_config(spec: dict) -> MLConfig:
    return MLConfig(examples=ExampleConfig(**MARGINS.get(spec["margins"] or "", {})),
                    features=FeatureConfig(**FEATS[spec["feats"]]),
                    forest=ForestConfig(kind=spec["forest"], n_jobs=1))


# --------------------------------------------------------------------------- #
#  Phase auto
# --------------------------------------------------------------------------- #

def auto_job(rec: C.Recording) -> tuple[str, dict]:
    files = rec.files()
    ref = rec.reference(files)
    out = {}
    for f in sorted(rec.labels):
        res = detect_edge(imread_unicode(files[f]), *C.FULL, reference=ref, params=EdgeParams(seed=rec.seed))
        out[str(f)] = C.circle_dict(res)
    return rec.key, out


# --------------------------------------------------------------------------- #
#  Phase grid
# --------------------------------------------------------------------------- #

def training_circles(rec: C.Recording, frames, auto: dict) -> dict:
    """Verified circles of the taught frames (example_1: Auto's circle where it matches the radius)."""
    if not rec.radius_only:
        return {f: rec.labels[f] for f in frames}
    out = {}
    for f in frames:
        a = auto[str(f)]
        if a["acc"] and C.agree((a["R"], a["rc"], a["cc"]), rec.labels[f], radius_only=True):
            out[f] = (a["R"], a["rc"], a["cc"])
    return out


def _guided(raw, ref, seed, region):
    if region is None:
        return None
    return detect_edge(raw, *C.FULL, reference=ref,
                       params=EdgeParams(seed=seed, prior=(region.rc, region.cc, region.radius)))


def grid_job(args) -> tuple[str, str, int]:
    rec, spec, auto, out_dir = args
    path = Path(out_dir) / "grid" / f"{rec.key.replace('/', '_')}__{spec['id']}.jsonl"
    if path.exists():
        return rec.key, spec["id"], -1
    files = rec.files()
    ref = rec.reference(files)
    auto_conf = {int(k): (v["conf"] if v["acc"] else 0.0) for k, v in auto.items()}
    frames = C.choose_frames(spec["strategy"], spec["n"], rec.labels, auto_conf)
    circles = training_circles(rec, frames, auto)
    test = [f for f in sorted(rec.labels) if f not in frames]
    lines = []
    if circles and test:
        shape = imread_unicode(files[0]).shape[:2]
        ex = C.verified_examples(circles) if spec["source"] == "verified" else C.stroke_examples(circles, shape)
        cfg = _ml_config(spec)
        try:
            model = train_model(ex, lambda i: imread_unicode(files[i]), *C.FULL, reference=ref, config=cfg)
        except MLError as exc:            # the app refuses to train: every frame keeps plain Auto
            lines.append(dict(type="model", train_frames=list(frames), error=str(exc)))
            lines += [dict(type="frame", frame=f, reg_raw=None, reg_ref=None, g_ref=None, g_raw=None,
                           novelty=float("nan"), predict_ms=float("nan"), **{k: None for k in REFINE_VARIANTS})
                      for f in test]
            path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
            return rec.key, spec["id"], len(test)
        head = dict(type="model", train_frames=list(frames), train_ms=model.info.train_ms,
                    n_bubble=model.info.n_bubble, n_background=model.info.n_background)
        lines.append(head)
        m_raw, m_ref = replace(cfg.measure, refine=False), replace(cfg.measure, refine=True)
        for f in test:
            raw = imread_unicode(files[f])
            pred = model.predict(raw)
            view = engine_view(raw, *C.FULL, reference=ref)
            reg_raw = ml_region(pred.probability, None, seed=rec.seed, config=m_raw)
            reg_ref = ml_region(pred.probability, view, seed=rec.seed, config=m_ref)
            g_ref = _guided(raw, ref, rec.seed, reg_ref)
            same = reg_ref is not None and not reg_ref.refined
            g_raw = g_ref if same else _guided(raw, ref, rec.seed, reg_raw)
            variants = {name: C.circle_dict(ml_region(pred.probability, view, seed=rec.seed,
                                                      config=replace(m_ref, **kw)))
                        for name, kw in REFINE_VARIANTS.items()}
            lines.append(dict(type="frame", frame=f, reg_raw=C.circle_dict(reg_raw), reg_ref=C.circle_dict(reg_ref),
                              g_ref=C.circle_dict(g_ref), g_raw=C.circle_dict(g_raw), novelty=pred.novelty,
                              predict_ms=pred.ms, **variants))
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return rec.key, spec["id"], len(lines) - 1 if lines else 0


# --------------------------------------------------------------------------- #
#  Phase confirm: the package's own measure_frame with its final defaults
# --------------------------------------------------------------------------- #

CONFIRM = ("verified-spread-2", "verified-spread-3", "verified-time-1", "verified-worst-2", "verified-largest-1",
           "strokes-spread-2", "strokes-spread-3", "strokes-largest-1", "verified-suggest-2", "verified-suggest-3",
           "strokes-suggest-2")


def _result_dict(res) -> dict:
    return {"R": float(res.radius), "rc": float(res.rc), "cc": float(res.cc), "conf": float(res.confidence),
            "acc": bool(res.accepted), "source": res.source, "flags": sorted(res.flags)}


def confirm_job(args) -> tuple[str, str, int]:
    rec, cid, auto, out_dir = args
    path = Path(out_dir) / "confirm" / f"{rec.key.replace('/', '_')}__{cid}.jsonl"
    if path.exists():
        return rec.key, cid, -1
    source, strategy, n = cid.split("-")
    files = rec.files()
    ref = rec.reference(files)
    auto_conf = {int(k): (v["conf"] if v["acc"] else 0.0) for k, v in auto.items()}
    auto_r = {int(k): v["R"] for k, v in auto.items() if v["acc"]}
    frames = C.choose_frames(strategy, int(n), rec.labels, auto_conf, auto_r)
    circles = training_circles(rec, frames, auto)
    test = [f for f in sorted(rec.labels) if f not in frames]
    lines = []
    if circles and test:
        shape = imread_unicode(files[0]).shape[:2]
        ex = C.verified_examples(circles) if source == "verified" else C.stroke_examples(circles, shape)
        cfg = MLConfig(forest=ForestConfig(n_jobs=1))
        model = train_model(ex, lambda i: imread_unicode(files[i]), *C.FULL, reference=ref, config=cfg)
        lines.append(dict(type="model", train_frames=list(frames), train_ms=model.info.train_ms))
        params = EdgeParams(seed=rec.seed)
        for f in test:
            raw = imread_unicode(files[f])
            plain = detect_edge(raw, *C.FULL, reference=ref, params=params)
            t0 = time.perf_counter()
            res = measure_frame(model, raw, params=params, auto=plain)
            ms = (time.perf_counter() - t0) * 1e3
            combined = select_result(res.region, res.guided, auto=plain,
                                     config=replace(cfg.measure, policy="combined"))
            lines.append(dict(type="frame", frame=f, guarded=_result_dict(res), combined=_result_dict(combined),
                              novelty=res.novelty, ms=ms))
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return rec.key, cid, len(lines) - 1 if lines else 0


# --------------------------------------------------------------------------- #
#  Phase cross (novelty) and timing
# --------------------------------------------------------------------------- #

def cross_job(args) -> tuple[str, dict]:
    rec, others, auto = args
    files = rec.files()
    ref = rec.reference(files)
    auto_conf = {int(k): (v["conf"] if v["acc"] else 0.0) for k, v in auto.items()}
    frames = C.choose_frames("spread", 3, rec.labels, auto_conf)
    circles = training_circles(rec, frames, auto)
    if not circles:
        return rec.key, {}
    model = train_model(C.verified_examples(circles), lambda i: imread_unicode(files[i]), *C.FULL, reference=ref,
                        config=MLConfig(forest=ForestConfig(n_jobs=1)))
    out = {}
    for other in [rec] + others:
        o_files = other.files()
        pick = [f for f in sorted(other.labels) if other is not rec or f not in frames][::6][:4]
        vals = []
        for f in pick:
            raw = imread_unicode(o_files[f])
            if raw.shape[:2] == model.frame_shape:
                vals.append(model.predict(raw).novelty)
        out[other.key] = vals
    return rec.key, out


def _median_ms(fn, items) -> float:
    times = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def timing(recs: list[C.Recording]) -> dict:
    """Train / predict / measure times, one process, nothing else running."""
    out = {}
    for key in ("lic/06", "lic_hard/05", "lic/09"):
        rec = next(r for r in recs if r.key == key)
        files = rec.files()
        ref = rec.reference(files)
        frames_by_n = {n: C.choose_frames("spread", n, rec.labels, {}) for n in (1, 2, 3, 5)}
        test = [f for f in sorted(rec.labels) if f not in frames_by_n[5]][:6]
        raws = [imread_unicode(files[f]) for f in test]
        params = EdgeParams(seed=rec.seed)
        out[f"{key}|plain Auto|-|-|-"] = dict(
            train_ms=0.0, predict_ms=0.0, n_samples=0,
            measure_ms=_median_ms(lambda r: detect_edge(r, *C.FULL, reference=ref, params=params), raws))
        for source in ("verified", "strokes"):
            for n, frames in frames_by_n.items():
                circles = {f: rec.labels[f] for f in frames}
                shape = raws[0].shape[:2]
                ex = C.verified_examples(circles) if source == "verified" else C.stroke_examples(circles, shape)
                for half in (False, True):
                    for jobs in (1, 4):
                        cfg = MLConfig(features=FeatureConfig(half_resolution=half), forest=ForestConfig(n_jobs=jobs))
                        t0 = time.perf_counter()
                        model = train_model(ex, lambda i: imread_unicode(files[i]), *C.FULL, reference=ref,
                                            config=cfg)
                        train_ms = (time.perf_counter() - t0) * 1e3
                        pred_ms = float(np.median([model.predict(r).ms for r in raws]))
                        meas_ms = _median_ms(lambda r: measure_frame(model, r, params=params), raws)
                        out[f"{key}|{source}|{n}|half={half}|jobs={jobs}"] = dict(
                            train_ms=train_ms, predict_ms=pred_ms, measure_ms=meas_ms,
                            n_samples=model.info.n_bubble + model.info.n_background)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=("auto", "grid", "confirm", "cross", "timing", "all"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated recording keys")
    ap.add_argument("--configs", default="", help="comma-separated config ids (grid)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "auto").mkdir(parents=True, exist_ok=True)
    (out / "grid").mkdir(parents=True, exist_ok=True)
    (out / "confirm").mkdir(parents=True, exist_ok=True)
    recs = C.recordings()
    if args.only:
        keep = set(args.only.split(","))
        recs = [r for r in recs if r.key in keep]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    phases = ("auto", "grid", "confirm", "cross", "timing") if args.phase == "all" else (args.phase,)
    with ProcessPoolExecutor(args.workers) as ex:
        if "auto" in phases:
            todo = [r for r in recs if not (out / "auto" / f"{r.key.replace('/', '_')}.json").exists()]
            for key, res in ex.map(auto_job, todo):
                (out / "auto" / f"{key.replace('/', '_')}.json").write_text(json.dumps(res), encoding="utf-8")
                print("auto", key, flush=True)
        autos = {r.key: json.loads((out / "auto" / f"{r.key.replace('/', '_')}.json").read_text(encoding="utf-8"))
                 for r in recs}
        if "grid" in phases:
            specs = grid_configs()
            if args.configs:
                specs = [s for s in specs if s["id"] in set(args.configs.split(","))]
            jobs = [(r, s, autos[r.key], str(out)) for s in specs for r in recs]
            t0 = time.perf_counter()
            for k, (key, cid, n) in enumerate(ex.map(grid_job, jobs, chunksize=1)):
                if n >= 0:
                    print(f"[{k + 1}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {key} {cid}: {n} frames", flush=True)
        if "confirm" in phases:
            jobs = [(r, cid, autos[r.key], str(out)) for cid in CONFIRM for r in recs]
            for k, (key, cid, n) in enumerate(ex.map(confirm_job, jobs, chunksize=1)):
                if n >= 0:
                    print(f"confirm [{k + 1}/{len(jobs)}] {key} {cid}: {n} frames", flush=True)
        if "cross" in phases:
            cross = dict(ex.map(cross_job, [(r, [o for o in recs if o is not r], autos[r.key]) for r in recs]))
            (out / "cross.json").write_text(json.dumps(cross, indent=1), encoding="utf-8")
            print("cross done", flush=True)
    if "timing" in phases:
        (out / "timing.json").write_text(json.dumps(timing(recs), indent=1), encoding="utf-8")
        print("timing done", flush=True)


if __name__ == "__main__":
    main()
