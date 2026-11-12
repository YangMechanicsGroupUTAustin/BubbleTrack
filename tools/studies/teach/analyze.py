"""Tables of the ML assist study from the records of ``run_study.py`` (markdown on stdout).

    python tools/studies/teach/analyze.py [--out DIR] [--main CONFIG_ID]

"Correct" = ACCEPTED and within the label criterion (radius max(2 px, 5 %), centre
max(3 px, 15 % R); example_1 radius only); "w" = accepted but wrong.  Every
comparison is on the same test frames (the labelled frames that were not taught);
"good" = recordings where Auto is already good (lic 01-04, 09, 10, example_1).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C
import confirm_tables as CT
import policies as P
from run_study import DEFAULT_OUT, FEATS, MARGINS, grid_configs

MAIN = "verified-spread-2-rawview-extra"
SHOWN = ("auto", "i_prior", "ii_contour", "ii_refined", "iii_combined", "v_guarded", "iv_auto_first")


def load(out: Path):
    recs = {r.key: r for r in C.recordings()}
    autos = {k: json.loads((out / "auto" / f"{k.replace('/', '_')}.json").read_text(encoding="utf-8"))
             for k in recs if (out / "auto" / f"{k.replace('/', '_')}.json").exists()}
    grid, by_file = {}, {k.replace("/", "_"): k for k in recs}
    for path in sorted((out / "grid").glob("*.jsonl")):
        stem, cid = path.stem.split("__")
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        grid[(by_file[stem], cid)] = lines
    return recs, autos, grid


def frames_of(grid, key, cid) -> list[dict]:
    return [x for x in grid.get((key, cid), []) if x["type"] == "frame"]


def score(recs, autos, grid, key, cid, fns) -> dict:
    """{policy: dict(ok, wrong, n, branches)} for one recording and configuration."""
    frames = frames_of(grid, key, cid)
    if not frames:
        return {}
    rec, out = recs[key], {}
    for name, fn in fns.items():
        ok = wrong = 0
        branches = Counter()
        for x in frames:
            circle, acc, branch = fn(x, autos[key][str(x["frame"])])
            good = acc and C.agree(circle, rec.labels[x["frame"]], rec.radius_only)
            ok += good
            wrong += acc and not good
            branches[(branch, "ok" if good else ("wrong" if acc else "rejected"))] += 1
        out[name] = dict(ok=ok, wrong=wrong, n=len(frames), branches=branches)
    return out


def cell(s: dict | None) -> str:
    if not s or not s["n"]:
        return "-"
    return f"{s['ok']}/{s['n']}" + (f" ({s['wrong']}w)" if s["wrong"] else "")


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def totals(recs, autos, grid, cid, fns) -> dict:
    """{"good"|"hard": {policy: dict(ok, wrong, n)}}."""
    out = {g: defaultdict(lambda: dict(ok=0, wrong=0, n=0)) for g in ("good", "hard")}
    for key in recs:
        s = score(recs, autos, grid, key, cid, fns)
        group = "good" if key in C.GOOD_FOR_AUTO else "hard"
        for name, v in s.items():
            for k in ("ok", "wrong", "n"):
                out[group][name][k] += v[k]
    return out


# --------------------------------------------------------------------------- #
#  Sections
# --------------------------------------------------------------------------- #

def section_main(recs, autos, grid, cid, fns) -> str:
    rows, per = [], {"good": [], "hard": []}
    for key in recs:
        s = score(recs, autos, grid, key, cid, fns)
        if not s:
            continue
        rows.append([key] + [cell(s[n]) for n in SHOWN])
        per["good" if key in C.GOOD_FOR_AUTO else "hard"].append(s)
    tot = totals(recs, autos, grid, cid, fns)
    for g in ("good", "hard"):
        rows.append([f"**{g} total**"] + [f"**{cell(tot[g][n])}**" for n in SHOWN])
    return table(["recording"] + list(SHOWN), rows)


def _triple(branches: Counter, branch: str) -> str:
    return "/".join(str(branches[(branch, k)]) for k in ("ok", "wrong", "rejected"))


def section_branches(recs, autos, grid, cid, fns, policy: str) -> str:
    """Which branch gave each frame's result (ok / wrong / rejected) under *policy*."""
    names = ("auto_ml", "auto", "ml") if policy == "v_guarded" else ("auto_ml", "ml")
    rows = []
    for key in recs:
        s = score(recs, autos, grid, key, cid, fns)
        if not s:
            continue
        b = s[policy]["branches"]
        rows.append([key, s[policy]["n"]] + [_triple(b, n) for n in names] + [b[("none", "rejected")]])
    labels = {"auto_ml": "Auto (ML-guided)", "auto": "plain Auto", "ml": "ML outline"}
    return table(["recording", "test frames"] + [f"{labels[n]} ok/wrong/rejected" for n in names] + ["no result"],
                 rows)


def section_choice(recs, autos, grid, fns) -> str:
    rows = []
    for source in ("verified", "strokes"):
        for strategy in ("largest", "spread", "worst", "time"):
            for n in (1, 2, 3, 5):
                cid = f"{source}-{strategy}-{n}-rawview-extra"
                if strategy == "spread" and n == 1:
                    cid = f"{source}-largest-1-rawview-extra"
                tot = totals(recs, autos, grid, cid, fns)
                if not tot["hard"]["auto"]["n"]:
                    continue
                rows.append([source, strategy, n] + [cell(tot[g][p]) for g in ("good", "hard")
                                                     for p in ("auto", "iii_combined", "v_guarded")])
    return table(["examples", "frames", "n", "good: Auto", "good: combined", "good: guarded",
                  "hard: Auto", "hard: combined", "hard: guarded"], rows)


def section_variants(recs, autos, grid, fns, ids: list[tuple[str, str]]) -> str:
    rows = []
    for label, cid in ids:
        tot = totals(recs, autos, grid, cid, fns)
        if not tot["hard"]["auto"]["n"]:
            continue
        rows.append([label] + [cell(tot[g][p]) for g in ("good", "hard")
                               for p in ("iii_combined", "v_guarded", "ii_refined")])
    return table(["variant", "good: combined", "good: guarded", "good: ML outline",
                  "hard: combined", "hard: guarded", "hard: ML outline"], rows)


def section_refine(recs, autos, grid, fns, cids) -> str:
    names = [n for n in fns if n.startswith("ii_")]
    rows = []
    for g in ("good", "hard"):
        acc = {n: dict(ok=0, wrong=0, n=0) for n in names}
        for cid in cids:
            tot = totals(recs, autos, grid, cid, fns)
            for n in names:
                for k in acc[n]:
                    acc[n][k] += tot[g][n][k]
        rows.append([g] + [cell(acc[n]) for n in names])
    return table(["group"] + names, rows)


def section_gate(recs, autos, grid, cids, policy: str, gates=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) -> str:
    """ML-outline branch of *policy*: accepted correct / wrong at each gate."""
    rows = []
    for gate in gates:
        fns = {"iii": P.make_policies(gate=gate)[policy]}
        row = [gate]
        for g in ("good", "hard"):
            ok = wrong = 0
            for cid in cids:
                for key in recs:
                    if (key in C.GOOD_FOR_AUTO) != (g == "good"):
                        continue
                    s = score(recs, autos, grid, key, cid, fns)
                    if s:
                        ok += s["iii"]["branches"][("ml", "ok")]
                        wrong += s["iii"]["branches"][("ml", "wrong")]
            row.append(f"{ok} / {wrong}")
        rows.append(row)
    return table(["gate", "good: ML outline correct / wrong", "hard: ML outline correct / wrong"], rows)


def section_terms(recs, autos, grid, cids, guarded: bool = False) -> str:
    """How well each term separates correct from wrong ML outlines (ROC AUC), on frames
    where the combined (or guarded) policy falls back to the ML outline."""
    vals = defaultdict(lambda: ([], []))
    for cid in cids:
        for key, rec in recs.items():
            for x in frames_of(grid, key, cid):
                reg = x["reg_ref"]
                if reg is None or (guarded and autos[key][str(x["frame"])]["acc"]):
                    continue
                g = x["g_ref"]
                if g is not None and g["acc"] and P.wall_agrees(reg, g):
                    continue
                good = C.agree(P.circ(reg), rec.labels[x["frame"]], rec.radius_only)
                for name, v in list(reg["terms"].items()) + [("confidence", P.conf_product(reg["terms"]))]:
                    vals[name][0 if good else 1].append(v)
    rows = []
    for name, (good, bad) in sorted(vals.items()):
        if not good or not bad:
            continue
        g, b = np.asarray(good), np.asarray(bad)
        auc = float(np.mean(g[:, None] > b[None, :]) + 0.5 * np.mean(g[:, None] == b[None, :]))
        rows.append([name, f"{np.median(g):.3f}", f"{np.median(b):.3f}", f"{auc:.2f}", f"{len(g)} / {len(b)}"])
    return table(["term", "median (correct)", "median (wrong)", "AUC", "n correct / wrong"], rows)


def section_novelty(out: Path) -> str:
    path = out / "cross.json"
    if not path.exists():
        return "(no cross.json)"
    cross = json.loads(path.read_text(encoding="utf-8"))
    own, other = [], []
    for key, d in cross.items():
        for k, v in d.items():
            (own if k == key else other).extend(v)
    own, other = np.asarray(own), np.asarray(other)
    rows = [[f"{t:.2f}", f"{np.mean(own > t) * 100:.0f} %", f"{np.mean(other > t) * 100:.0f} %"]
            for t in (0.05, 0.1, 0.15, 0.2, 0.3)]
    head = (f"novelty, same recording: median {np.median(own):.3f} (95th pct {np.percentile(own, 95):.3f}); "
            f"other recordings: median {np.median(other):.3f} (5th pct {np.percentile(other, 5):.3f})\n\n")
    return head + table(["warn above", "same recording flagged", "other recording flagged"], rows)


def section_timing(out: Path) -> str:
    path = out / "timing.json"
    if not path.exists():
        return "(no timing.json)"
    t = json.loads(path.read_text(encoding="utf-8"))
    rows = [k.split("|") + [f"{v['train_ms']:.0f}", f"{v['predict_ms']:.0f}", f"{v.get('measure_ms', 0):.0f}",
                            v["n_samples"]] for k, v in t.items()]
    return table(["recording", "examples", "frames", "half res", "threads", "train ms", "predict ms / frame",
                  "measure_frame ms", "training pixels"], rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--main", default=MAIN)
    args = ap.parse_args()
    out = Path(args.out)
    recs, autos, grid = load(out)
    fns = P.make_policies()
    ids = {s["id"] for s in grid_configs()}
    confirm = CT.load_confirm(out, recs)
    if confirm:
        print(f"## Final defaults, per recording (verified-spread-2)\n\n"
              f"{CT.per_recording(confirm, recs, autos, 'verified-spread-2')}\n")
        cids = sorted({cid for _, cid in confirm})
        print(f"## Final defaults, by taught frames\n\n{CT.overview(confirm, recs, autos, cids)}\n")
        print(f"{CT.timing_from_confirm(confirm)}\n")
    print(f"## Per recording ({args.main})\n\n{section_main(recs, autos, grid, args.main, fns)}\n")
    for policy in ("iii_combined", "v_guarded"):
        branches = section_branches(recs, autos, grid, args.main, fns, policy)
        print(f"## Branches of {policy} ({args.main})\n\n{branches}\n")
    print(f"## Number and choice of taught frames\n\n{section_choice(recs, autos, grid, fns)}\n")
    feats = [(f"{src} spread-{n} {f}", f"{src}-spread-{n}-{f}-extra") for src, n in
             (("verified", 2), ("verified", 3), ("strokes", 2)) for f in FEATS]
    print(f"## Features\n\n{section_variants(recs, autos, grid, fns, feats)}\n")
    forests = [(f"{src} spread-{n} {k}", f"{src}-spread-{n}-rawview-{k}") for src, n in
               (("verified", 2), ("verified", 3), ("strokes", 2)) for k in ("extra", "random")]
    print(f"## Forest\n\n{section_variants(recs, autos, grid, fns, forests)}\n")
    margins = [("default (0.8 / 1.25-2.0 / 2.5)", MAIN)] + [
        (f"{m} {MARGINS[m]}", f"verified-spread-2-rawview-extra-{m}") for m in MARGINS]
    print(f"## Verified-frame margins\n\n{section_variants(recs, autos, grid, fns, margins)}\n")
    main_cids = sorted(c for c in ids if c.endswith("-rawview-extra") and (c.startswith("verified")
                                                                           or c.startswith("strokes")))
    print(f"## Refinement variants (all {len(main_cids)} main configurations)\n\n"
          f"{section_refine(recs, autos, grid, fns, main_cids)}\n")
    for policy in ("iii_combined", "v_guarded"):
        gate = section_gate(recs, autos, grid, main_cids, policy)
        print(f"## Gate of the ML outline, {policy} (all main configurations)\n\n{gate}\n")
    for guarded in (False, True):
        name = "v_guarded" if guarded else "iii_combined"
        print(f"## Confidence terms (ML-outline frames of {name}, all main configurations)\n\n"
              f"{section_terms(recs, autos, grid, main_cids, guarded)}\n")
    print(f"## Novelty\n\n{section_novelty(out)}\n")
    print(f"## Timing\n\n{section_timing(out)}\n")


if __name__ == "__main__":
    main()
