"""``bubbletrack-cli eval-cases``: run the hard-case regression library.

Re-runs the current engine on every hard-case bundle under a folder, scores
it against the users' fixes and writes a Markdown + JSON report.  With
``--baseline`` (an earlier report's JSON) frames that were correct then and
are not now are regressions, and the exit status is 1.

Exit status: 0 = done, no regressions; 1 = regressions against the baseline;
2 = invalid arguments, unreadable baseline, no bundles found or the report
could not be written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bubbletrack.logging_config import setup_logging
from bubbletrack.model.cases.bundle import find_bundles, open_bundle
from bubbletrack.model.cases.replay import replay_case
from bubbletrack.model.cases.report import build_report, load_report, write_report
from bubbletrack.model.cases.schema import CaseFormatError
from bubbletrack.model.cases.scoring import CORRECT, REJECTED, WRONG, CaseScore, Tolerance, score_case

logger = logging.getLogger(__name__)

REPORT_STEM = "hard_cases_report"
EXIT_OK, EXIT_REGRESSIONS, EXIT_USAGE = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bubbletrack-cli eval-cases",
        description="Re-run the current engine on hard-case bundles and score it "
                    "against the users' fixes (manual fits and accepted frames).",
    )
    parser.add_argument("folder", help="Folder holding hard-case bundles (.zip files or "
                                       "unpacked folders), searched recursively")
    parser.add_argument("--baseline", metavar="REPORT_JSON",
                        help="An earlier report (.json): frames correct there but not now "
                             "are regressions, and the exit status is then 1")
    parser.add_argument("--out", metavar="DIR", default=".",
                        help="Folder for the report (default: the current folder)")
    parser.add_argument("--name", default=REPORT_STEM,
                        help=f"Report file name without extension (default: {REPORT_STEM})")
    defaults = Tolerance()
    for flag, value, text in (
            ("--radius-tol-px", defaults.radius_px, "Radius tolerance floor in px"),
            ("--radius-tol-frac", defaults.radius_frac, "Radius tolerance as a fraction of R"),
            ("--center-tol-px", defaults.center_px, "Centre tolerance floor in px"),
            ("--center-tol-frac", defaults.center_frac, "Centre tolerance as a fraction of R")):
        parser.add_argument(flag, type=float, default=value, help=f"{text} (default: {value:g})")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Auto mode: skip tracking across frames even where the user had it on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser


def _tolerance(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Tolerance:
    values = (args.radius_tol_px, args.radius_tol_frac, args.center_tol_px, args.center_tol_frac)
    if any(not v >= 0 for v in values):
        parser.error("tolerances must be >= 0")
    return Tolerance(*values)


def run_case(path: Path, label: str, tol: Tolerance, *, tracking: bool | None = None) -> CaseScore:
    """Open, replay and score one bundle; a failure becomes the score's ``error``."""
    try:
        bundle = open_bundle(path)
    except CaseFormatError as exc:
        return CaseScore(label, None, error=str(exc))
    except Exception as exc:  # anything else unreadable: report it, keep going
        logger.debug("Opening %s failed", path, exc_info=True)
        return CaseScore(label, None, error=f"{type(exc).__name__}: {exc}")
    with bundle:
        try:
            replayed = replay_case(bundle.case, bundle.read_frame, tracking=tracking)
        except Exception as exc:  # one broken case must not stop the run
            logger.debug("Replay of %s failed", path, exc_info=True)
            return CaseScore(label, bundle.case, error=f"{type(exc).__name__}: {exc}")
        return score_case(label, bundle.case, replayed, tol)


def _line(score: CaseScore) -> str:
    if score.error:
        return f"  [error] {score.case_id}: {score.error}"
    return (f"  [ok]    {score.case_id}: {score.labelled} labelled, {score.count(CORRECT)} correct, "
            f"{score.count(WRONG)} wrong, {score.count(REJECTED)} rejected "
            f"(reproduced {score.reproduced}/{score.auto_results})")


def _run_all(folder: Path, tol: Tolerance, tracking: bool | None) -> list[CaseScore]:
    scores: list[CaseScore] = []
    seen: dict[str, str] = {}
    for path in find_bundles(folder):
        label = path.relative_to(folder).as_posix() if path != folder else path.name
        score = run_case(path, label, tol, tracking=tracking)
        if score.case is not None and score.case_id in seen:
            print(f"  [skip]  {label}: same case as {seen[score.case_id]}")
            continue
        seen[score.case_id] = label
        print(_line(score))
        scores.append(score)
    return scores


def _print_summary(report: dict) -> None:
    t = report["totals"]
    print(f"Total: {t['labelled']} labelled frame(s) in {t['cases']} case(s): "
          f"{t[CORRECT]} correct, {t[WRONG]} wrong, {t[REJECTED]} rejected"
          + (f"; {t['errors']} case(s) could not run" if t["errors"] else ""))
    base = report.get("baseline")
    if base:
        print(f"Against {base['name']}: {len(base['regressions'])} regression(s), "
              f"{len(base['fixed'])} fixed")
        for change in base["regressions"]:
            print(f"  REGRESSION {change['case_id']} frame {change['frame']}: "
                  f"{change['before']} -> {change['now']}")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the exit status (see the module docstring)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # notes and names may be any script
    parser = build_parser()
    args = parser.parse_args(argv)
    tol = _tolerance(parser, args)
    setup_logging(level=logging.DEBUG if args.verbose else logging.WARNING)
    folder = Path(args.folder)
    if not folder.is_dir():
        parser.error(f"not a folder: {folder}")
    baseline = None
    if args.baseline:
        try:
            baseline = load_report(args.baseline)
        except ValueError as exc:
            parser.error(f"--baseline: {exc}")

    print(f"Hard cases in {folder}:")
    scores = _run_all(folder, tol, False if args.no_tracking else None)
    if not scores:
        print(f"No hard-case bundles found in {folder}", file=sys.stderr)
        return EXIT_USAGE
    report = build_report(scores, tol, baseline=baseline,
                          baseline_name=Path(args.baseline).name if args.baseline else "")
    try:
        json_path, md_path = write_report(report, args.out, args.name)
    except OSError as exc:
        print(f"Cannot write the report to {args.out}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    _print_summary(report)
    print(f"Report: {md_path} and {json_path}")
    regressions = report["baseline"]["regressions"] if report["baseline"] else []
    return EXIT_REGRESSIONS if regressions else EXIT_OK
