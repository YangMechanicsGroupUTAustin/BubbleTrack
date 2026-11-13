"""Scoring replays against labels, and the report (pure, synthetic data)."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from bubbletrack.model.cases.replay import Replayed
from bubbletrack.model.cases.report import (
    build_report, compare_with_baseline, load_report, render_markdown, write_report,
)
from bubbletrack.model.cases.schema import (
    LABEL_BUBBLE, LABEL_NO_BUBBLE, ROLE_CONTEXT, ROLE_REFERENCE, ROLE_SELECTED, AppResult,
    CaseFrame, Detection, HardCase, Label,
)
from bubbletrack.model.cases.scoring import (
    CORRECT, REJECTED, WRONG, CaseScore, Tolerance, judge, score_case,
)

TOL = Tolerance()
BUBBLE = Label(LABEL_BUBBLE, 50.0, (100.0, 100.0), "manual")
NONE = Label(LABEL_NO_BUBBLE, by="accepted")


class TestJudge:
    @pytest.mark.parametrize("radius, center, outcome", [
        (50.0, (100.0, 100.0), CORRECT),
        (51.5, (100.0, 100.0), CORRECT),     # 3 % of 50 = 1.5 px
        (51.6, (100.0, 100.0), WRONG),
        (50.0, (104.9, 100.0), CORRECT),     # 10 % of 50 = 5 px
        (50.0, (103.0, 104.0), CORRECT),     # distance 5
        (50.0, (105.1, 100.0), WRONG),
        (None, None, REJECTED),
    ])
    def test_bubble_label(self, radius, center, outcome):
        assert judge(BUBBLE, Replayed(radius, center), TOL) == outcome

    def test_floors_apply_to_small_bubbles(self):
        small = Label(LABEL_BUBBLE, 10.0, (50.0, 50.0))
        assert judge(small, Replayed(10.9, (51.9, 50.0)), TOL) == CORRECT  # 1 px / 2 px floors
        assert judge(small, Replayed(11.1, (50.0, 50.0)), TOL) == WRONG
        assert judge(small, Replayed(10.0, (52.1, 50.0)), TOL) == WRONG

    def test_no_bubble_label(self):
        assert judge(NONE, Replayed(None), TOL) == CORRECT
        assert judge(NONE, Replayed(12.0, (5.0, 5.0)), TOL) == WRONG

    def test_label_without_centre_checks_the_radius_only(self):
        label = Label(LABEL_BUBBLE, 50.0, None)
        assert judge(label, Replayed(50.5, (1.0, 1.0)), TOL) == CORRECT

    def test_custom_tolerance(self):
        assert judge(BUBBLE, Replayed(51.0, (100.0, 100.0)), Tolerance(0.5, 0.0, 2, 0.1)) == WRONG


def _frame(index, role=ROLE_SELECTED, label=Label(), result=AppResult()):
    return CaseFrame(index, role, f"frames/frame_{index + 1:06d}.tif", result=result, label=label)


def make_case(case_id="case_a") -> HardCase:
    auto = AppResult(40.0, (90.0, 90.0), 0.9, source="auto")
    frames = (
        _frame(4, ROLE_CONTEXT, result=auto),
        _frame(5, label=BUBBLE, result=AppResult(50.0, (100.0, 100.0), source="manual")),
        _frame(6, label=NONE),
        _frame(7, label=Label(LABEL_BUBBLE, 30.0, (80.0, 80.0), "accepted"),
               result=AppResult(30.0, (80.0, 80.0), source="tracked", locked=True)),
        _frame(8, ROLE_CONTEXT, result=auto),
        _frame(20, ROLE_REFERENCE),
    )
    return HardCase(case_id=case_id, source=MappingProxyType({"kind": "folder", "name": "gel"}),
                    gridx=(1, 200), gridy=(1, 300),
                    detection=Detection("edge", MappingProxyType({}), tracking=True,
                                        reference_index=20),
                    selection=(5, 7), frames=frames, created="2026-11-12T10:00:00",
                    app_version="3.0.0", reporter="Ann", note="wall | merge\nsecond line")


REPLAYED = {4: Replayed(40.2, (90.0, 90.0)), 5: Replayed(55.0, (100.0, 100.0)),
            6: Replayed(None, reason="no bubble found"), 7: Replayed(None, reason="low confidence"),
            8: Replayed(None)}


class TestScoreCase:
    def test_counts(self):
        score = score_case("a.zip", make_case(), REPLAYED, TOL)
        assert [f.index for f in score.frames] == [5, 6, 7]  # labelled; never the reference
        assert [f.outcome for f in score.frames] == [WRONG, CORRECT, REJECTED]
        assert (score.labelled, score.count(CORRECT), score.count(WRONG), score.count(REJECTED),
                score.unlabelled) == (3, 1, 1, 1, 2)
        assert (score.auto_results, score.reproduced) == (3, 1)  # frames 4, 7, 8 were automatic
        assert score.frames[0].radius_error == pytest.approx(5.0)
        assert score.case_id == "case_a"

    def test_failed_case_is_named_by_its_bundle(self):
        assert CaseScore("broken.zip", None, error="bad").case_id == "broken.zip"


def _report(replayed=REPLAYED, baseline=None, case_id="case_a"):
    score = score_case(f"{case_id}.zip", make_case(case_id), replayed, TOL)
    return build_report([score], TOL, baseline=baseline, baseline_name="old.json")


class TestReport:
    def test_json_is_strict_and_complete(self):
        report = _report()
        json.dumps(report, allow_nan=False)
        (case,) = report["cases"]
        assert case["frames"] == [6, 8] and case["mode"] == "edge" and case["reporter"] == "Ann"
        assert [r["frame"] for r in case["results"]] == [6, 7, 8]
        assert report["totals"] == {"cases": 1, "errors": 0, "labelled": 3, "correct": 1,
                                    "wrong": 1, "rejected": 1, "unlabelled": 2,
                                    "auto_results": 3, "reproduced": 1}
        assert report["baseline"] is None

    def test_regressions_and_fixes_against_a_baseline(self):
        before = _report()
        after_replay = {**REPLAYED, 5: Replayed(50.2, (100.0, 100.0)), 6: Replayed(9.0, (1.0, 1.0))}
        base = _report(after_replay, baseline=before)["baseline"]
        assert base["regressions"] == [{"case_id": "case_a", "frame": 7, "before": CORRECT,
                                        "now": WRONG}]
        assert base["fixed"] == [{"case_id": "case_a", "frame": 6, "before": WRONG,
                                  "now": CORRECT}]

    def test_a_case_that_fails_now_regresses_its_correct_frames(self):
        before = _report()
        failed = build_report([CaseScore("case_a.zip", make_case(), error="boom")], TOL)
        base = compare_with_baseline(failed["cases"], before)
        assert base["regressions"] == [{"case_id": "case_a", "frame": 7, "before": CORRECT,
                                        "now": "error"}]

    def test_missing_and_new_cases(self):
        base = _report(case_id="case_b", baseline=_report())["baseline"]
        assert base["missing_cases"] == ["case_a"] and base["new_cases"] == ["case_b"]
        assert base["regressions"] == []

    def test_markdown(self):
        before = _report({**REPLAYED, 5: Replayed(50.0, (100.0, 100.0))})
        text = render_markdown(_report(baseline=before))
        assert "# Hard-case regression report" in text
        assert "**1 regression(s)**" in text and "| case_a | 6 | correct | **wrong** |" in text
        assert "| case_a | edge | 6–8 | 3 | 1 | 1 | 1 | 1/3 |" in text
        assert "> wall \\| merge second line" in text  # one line, pipes escaped
        assert "| 8 | bubble (accepted) | **rejected** |" in text

    def test_write_and_load(self, tmp_path):
        json_path, md_path = write_report(_report(), tmp_path / "out", "r")
        assert load_report(json_path)["totals"]["labelled"] == 3
        assert md_path.read_text(encoding="utf-8").startswith("# Hard-case")

    @pytest.mark.parametrize("content", ["{}", "not json", json.dumps({"format": "x", "cases": []}),
                                         json.dumps({"format": "bubbletrack-hard-case-report",
                                                     "cases": [{"case_id": 1}]})])
    def test_load_rejects_other_files(self, tmp_path, content):
        path = tmp_path / "r.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_report(path)
