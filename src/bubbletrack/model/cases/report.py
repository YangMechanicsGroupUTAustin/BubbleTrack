"""The regression report: JSON (for the next run's ``--baseline``) and Markdown.

Frames are numbered as in the app (1-based).  A *regression* is a labelled
frame that was correct in the baseline report and is not correct now (wrong,
rejected, or its case failed to run); *fixed* is the opposite.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import bubbletrack
from bubbletrack.model.cases.scoring import (
    CORRECT, OUTCOMES, REJECTED, WRONG, CaseScore, FrameScore, Tolerance,
)

REPORT_FORMAT = "bubbletrack-hard-case-report"
REPORT_VERSION = 1
_COUNT_KEYS = ("labelled", *OUTCOMES, "unlabelled", "auto_results", "reproduced")


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _frame_entry(score: FrameScore) -> dict:
    now = score.replayed
    return {"frame": score.index + 1, "label": score.label.status, "by": score.label.by,
            "outcome": score.outcome, "label_radius": _round(score.label.radius),
            "radius": _round(now.radius), "radius_error": _round(score.radius_error),
            "center_error": _round(score.center_error), "tracked": now.tracked,
            "reason": now.reason}


def case_entry(score: CaseScore) -> dict:
    """One case of the report (plain JSON data)."""
    counts = {"labelled": score.labelled, **{o: score.count(o) for o in OUTCOMES},
              "unlabelled": score.unlabelled, "auto_results": score.auto_results,
              "reproduced": score.reproduced}
    entry = {"case_id": score.case_id, "bundle": score.bundle, "error": score.error or None,
             "counts": counts, "results": [_frame_entry(f) for f in score.frames]}
    case = score.case
    if case is not None:
        entry.update(source=case.name, kind=case.source.get("kind", ""),
                     mode=case.detection.mode, tracking=case.detection.tracking,
                     frames=[case.selection[0] + 1, case.selection[1] + 1],
                     reporter=case.reporter, note=case.note, exported=case.created,
                     exported_with=case.app_version)
    return entry


def _totals(cases: list[dict]) -> dict:
    totals = {key: sum(c["counts"][key] for c in cases) for key in _COUNT_KEYS}
    return {"cases": len(cases), "errors": sum(1 for c in cases if c.get("error")), **totals}


def _outcomes(cases: Iterable[dict]) -> dict[tuple[str, int], str]:
    return {(c["case_id"], r["frame"]): r["outcome"] for c in cases for r in c.get("results", [])}


def compare_with_baseline(cases: list[dict], baseline: dict) -> dict:
    """Regressions and fixes of *cases* relative to an earlier report."""
    before, now = _outcomes(baseline["cases"]), _outcomes(cases)
    current = {c["case_id"]: c for c in cases}
    regressions, fixed = [], []
    for (case_id, frame), old in sorted(before.items()):
        if case_id not in current:
            continue
        new = now.get((case_id, frame), "error" if current[case_id].get("error") else "unlabelled")
        change = {"case_id": case_id, "frame": frame, "before": old, "now": new}
        if old == CORRECT and new != CORRECT:
            regressions.append(change)
        elif old != CORRECT and new == CORRECT:
            fixed.append(change)
    old_ids = {c["case_id"] for c in baseline["cases"]}
    return {"regressions": regressions, "fixed": fixed,
            "missing_cases": sorted(old_ids - set(current)),
            "new_cases": sorted(set(current) - old_ids)}


def _engine_version() -> str:
    from bubbletrack.model.edge import ENGINE_VERSION
    return ENGINE_VERSION


def build_report(scores: Iterable[CaseScore], tol: Tolerance, *, baseline: dict | None = None,
                 baseline_name: str = "", now: datetime | None = None) -> dict:
    cases = [case_entry(s) for s in scores]
    report = {"format": REPORT_FORMAT, "format_version": REPORT_VERSION,
              "created": (now or datetime.now()).isoformat(timespec="seconds"),
              "app_version": getattr(bubbletrack, "__version__", ""),
              "engine_version": _engine_version(),
              "tolerance": {"radius_px": tol.radius_px, "radius_frac": tol.radius_frac,
                            "center_px": tol.center_px, "center_frac": tol.center_frac,
                            "rule": tol.describe()},
              "totals": _totals(cases), "cases": cases, "baseline": None}
    if baseline is not None:
        report["baseline"] = {"name": baseline_name, "created": baseline.get("created", ""),
                              **compare_with_baseline(cases, baseline)}
    return report


def _check_report(data) -> dict:
    ok = (isinstance(data, dict) and data.get("format") == REPORT_FORMAT
          and isinstance(data.get("cases"), list)
          and all(isinstance(c, dict) and isinstance(c.get("case_id"), str)
                  and isinstance(c.get("results", []), list)
                  and all(isinstance(r, dict) and isinstance(r.get("frame"), int)
                          and isinstance(r.get("outcome"), str) for r in c.get("results", []))
                  for c in data["cases"]))
    if not ok:
        raise ValueError("not a BubbleTrack hard-case report (eval-cases JSON)")
    return data


def load_report(path: str | os.PathLike) -> dict:
    """Read a report written by ``eval-cases``; ValueError if it is not one."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return _check_report(data)


def write_report(report: dict, out_dir: str | os.PathLike, stem: str) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` into *out_dir*; returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / f"{stem}.json", out / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# -- Markdown ------------------------------------------------------------------ #

def _cell(text) -> str:
    return "" if text is None else str(text).replace("|", "\\|").replace("\n", " ")


def _num(value) -> str:
    return "–" if value is None else f"{value:.2f}"


def _span(frames) -> str:
    if not frames:
        return ""
    return str(frames[0]) if frames[0] == frames[1] else f"{frames[0]}–{frames[1]}"


def _baseline_md(base: dict) -> list[str]:
    lines = [f"## Against baseline {_cell(base['name'])}", ""]
    regs, fixed = base["regressions"], base["fixed"]
    lines.append(f"**{len(regs)} regression(s)** (correct before, not now) · {len(fixed)} fixed")
    if regs or fixed:
        lines += ["", "| Case | Frame | Before | Now |", "|---|---:|---|---|"]
        for change, mark in [(c, "**") for c in regs] + [(c, "") for c in fixed]:
            lines.append(f"| {_cell(change['case_id'])} | {change['frame']} | {change['before']} "
                         f"| {mark}{change['now']}{mark} |")
    for key, text in (("missing_cases", "Not in this run"), ("new_cases", "New since the baseline")):
        if base[key]:
            lines += ["", f"{text}: " + ", ".join(_cell(c) for c in base[key])]
    return lines + [""]


def _summary_md(report: dict) -> list[str]:
    lines = ["## Summary", "",
             "| Case | Mode | Frames | Labelled | Correct | Wrong | Rejected | Reproduced |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for c in report["cases"]:
        n = c["counts"]
        if c.get("error"):
            lines.append(f"| {_cell(c['case_id'])} | {_cell(c.get('mode'))} | "
                         f"{_span(c.get('frames'))} | error | | | | |")
            continue
        lines.append(f"| {_cell(c['case_id'])} | {c['mode']} | {_span(c['frames'])} | "
                     f"{n['labelled']} | {n[CORRECT]} | {n[WRONG]} | {n[REJECTED]} | "
                     f"{n['reproduced']}/{n['auto_results']} |")
    t = report["totals"]
    lines.append(f"| **Total ({t['cases']} case(s))** | | | {t['labelled']} | {t[CORRECT]} | "
                 f"{t[WRONG]} | {t[REJECTED]} | {t['reproduced']}/{t['auto_results']} |")
    return lines + ["", "Wrong = accepted but not matching the label (the app shows a wrong "
                    "number); rejected = no result where the label has a bubble.  Reproduced = "
                    "the app's automatic results at export that this run gives again.", ""]


def _case_md(c: dict) -> list[str]:
    lines = [f"### {_cell(c['case_id'])}", ""]
    if c.get("error"):
        return lines + [f"Could not run: {_cell(c['error'])}", ""]
    who = f" · reported by {_cell(c['reporter'])}" if c.get("reporter") else ""
    lines.append(f"{_cell(c['source'])} ({c['kind']}), frames {_span(c['frames'])}, "
                 f"{c['mode']} mode{' with tracking' if c.get('tracking') else ''}{who} · "
                 f"exported {_cell(c['exported'])} with {_cell(c['exported_with'])}")
    if c.get("note"):
        lines += ["", f"> {_cell(c['note'])}"]
    bad = [r for r in c["results"] if r["outcome"] != CORRECT]
    lines.append("")
    if not c["results"]:
        return lines + ["No labelled frames.", ""]
    if not bad:
        return lines + [f"All {len(c['results'])} labelled frames correct.", ""]
    lines += ["| Frame | Label | Outcome | Label R | R now | R error | Centre error | Note |",
              "|---:|---|---|---:|---:|---:|---:|---|"]
    for r in bad:
        label = r["label"] + (f" ({r['by']})" if r.get("by") else "")
        note = "tracked" if r.get("tracked") else _cell(r.get("reason"))
        lines.append(f"| {r['frame']} | {label} | **{r['outcome']}** | {_num(r['label_radius'])} | "
                     f"{_num(r['radius'])} | {_num(r['radius_error'])} | "
                     f"{_num(r['center_error'])} | {note} |")
    return lines + [""]


def render_markdown(report: dict) -> str:
    t = report["totals"]
    lines = ["# Hard-case regression report", "",
             f"BubbleTrack {report['app_version']} · engine {report['engine_version']} · "
             f"{report['created']} · {t['cases']} case(s), {t['labelled']} labelled frame(s)", "",
             f"Correct: an accepted result with {report['tolerance']['rule']}; a \"no bubble\" "
             "label is correct when there is no result.", ""]
    if report.get("baseline"):
        lines += _baseline_md(report["baseline"])
    lines += _summary_md(report)
    lines += ["## Cases", ""]
    for c in report["cases"]:
        lines += _case_md(c)
    return "\n".join(lines).rstrip() + "\n"
