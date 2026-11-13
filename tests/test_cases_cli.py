"""``bubbletrack-cli eval-cases``: bundles in, report out, regressions -> exit 1."""

from __future__ import annotations

import dataclasses
import json

import pytest

from _case_state import fit_all, folder_state, max_radius, set_manual
from bubbletrack import cases_cli, cli
from bubbletrack.model.cases import replay as replay_module
from bubbletrack.model.cases.build import CaseRequest, make_case, write_case
from bubbletrack.model.image_io import imread_unicode
from bubbletrack.model.provenance import FitSource
from conftest import write_sequence

LEGACY = {"img_thr": 0.58, "removing_factor": 85, "gaussian_sigma": 3.0}


@pytest.fixture
def cases(tmp_path):
    """Two bundles: Auto mode (3 labels) and Intensity mode (2 labels)."""
    folder = tmp_path / "cases"
    folder.mkdir()
    # Frames 1 and 2 have no bubble; frame 1 is the reference.
    edge = fit_all(folder_state(write_sequence(tmp_path / "edge", [0.0, 0.0, 20, 25, 30, 35]),
                                roi=((31, 220), (61, 340)), mode="edge"), reference=0)
    assert edge.radius[1] < 0 and edge.fit_source[1] == FitSource.NONE
    edge.fit_locked[1:3] = True  # accepted: frame 2 "no bubble", frame 3 its result
    set_manual(edge, 3, float(edge.radius[3]) + 0.4, *edge.circle_fit_par[3])
    _export(edge, folder / "edge_case.zip", 1, 3, reference=0)
    legacy = fit_all(folder_state(write_sequence(tmp_path / "legacy", [20, 25, 30, 35]), **LEGACY))
    legacy.fit_locked[1:3] = True
    _export(legacy, folder / "sub" / "legacy_case.zip", 1, 2)
    return folder


def _export(state, path, start, end, *, reference=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    case = make_case(state, CaseRequest(start=start, end=end, reference_index=reference,
                                        max_radius=max_radius(state)),
                     imread_unicode(state.images[0]).shape)
    write_case(path, case, lambda i: imread_unicode(state.images[i]))


def _run(folder, out, *extra) -> int:
    return cases_cli.main([str(folder), "--out", str(out), *extra])


def test_report_for_every_case(cases, tmp_path, capsys):
    out = tmp_path / "report"
    assert _run(cases, out) == 0
    report = json.loads((out / "hard_cases_report.json").read_text(encoding="utf-8"))
    by_id = {c["bundle"]: c for c in report["cases"]}
    assert set(by_id) == {"edge_case.zip", "sub/legacy_case.zip"}
    edge = by_id["edge_case.zip"]["counts"]
    assert (edge["labelled"], edge["correct"], edge["wrong"], edge["rejected"]) == (3, 3, 0, 0)
    legacy = by_id["sub/legacy_case.zip"]["counts"]
    assert (legacy["labelled"], legacy["correct"]) == (2, 2)
    assert legacy["reproduced"] == legacy["auto_results"] > 0
    assert (out / "hard_cases_report.md").read_text(encoding="utf-8").count("### ") == 2
    printed = capsys.readouterr().out
    assert "Total: 5 labelled frame(s) in 2 case(s): 5 correct, 0 wrong, 0 rejected" in printed


def test_regressions_against_a_baseline_exit_1(cases, tmp_path, monkeypatch, capsys):
    assert _run(cases, tmp_path / "before") == 0
    baseline = tmp_path / "before" / "hard_cases_report.json"
    real = replay_module.replay_case

    def worse(case, read_frame, **kw):  # an engine change that shrinks every bubble
        return {i: dataclasses.replace(r, radius=r.radius - 5.0) if r.accepted else r
                for i, r in real(case, read_frame, **kw).items()}

    monkeypatch.setattr(cases_cli, "replay_case", worse)
    capsys.readouterr()
    assert _run(cases, tmp_path / "after", "--baseline", str(baseline)) == 1
    printed = capsys.readouterr().out
    assert "4 regression(s), 0 fixed" in printed and "REGRESSION" in printed
    report = json.loads((tmp_path / "after" / "hard_cases_report.json").read_text("utf-8"))
    assert len(report["baseline"]["regressions"]) == 4  # the "no bubble" frame stays correct
    md = (tmp_path / "after" / "hard_cases_report.md").read_text(encoding="utf-8")
    assert "**4 regression(s)**" in md


def test_same_engine_against_its_own_baseline_exits_0(cases, tmp_path):
    assert _run(cases, tmp_path / "a") == 0
    assert _run(cases, tmp_path / "b", "--baseline", str(tmp_path / "a" / "hard_cases_report.json")) == 0


def test_broken_bundles_are_reported_not_fatal(cases, tmp_path, capsys):
    (cases / "broken.zip").write_bytes(b"not a zip")
    assert _run(cases, tmp_path / "r") == 0
    printed = capsys.readouterr().out
    assert "[error] broken.zip" in printed and "1 case(s) could not run" in printed


def test_duplicate_bundles_count_once(cases, tmp_path, capsys):
    import shutil

    shutil.copy(cases / "edge_case.zip", cases / "edge_copy.zip")
    assert _run(cases, tmp_path / "r") == 0
    assert "[skip]  edge_copy.zip: same case as edge_case.zip" in capsys.readouterr().out


def test_usage_errors_exit_2(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(empty, tmp_path / "r") == 2
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path / "missing", tmp_path / "r")
    assert exc.value.code == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(empty, tmp_path / "r", "--baseline", str(bad))
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        _run(empty, tmp_path / "r", "--radius-tol-px", "-1")
    assert exc.value.code == 2


def test_unwritable_report_folder_exits_2(cases, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert _run(cases, blocker) == 2


def test_subcommand_of_the_main_cli(cases, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["eval-cases", str(cases), "--out", str(tmp_path / "r")])
    assert exc.value.code == 0
    assert (tmp_path / "r" / "hard_cases_report.md").is_file()


def test_main_cli_still_needs_folder_and_output(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    assert "eval-cases" in cli._build_parser().format_help()
