"""Hard cases exported while ML assist is on: the switch and the examples are in
the parameters, the example frames are stored (not measured), and a replay
retrains ML assist from them, as Fit All would."""

from __future__ import annotations

from dataclasses import replace

import pytest

import _ml_synth as S
from _case_state import fit_all, folder_state, max_radius, set_manual
from bubbletrack.model.cases.build import CaseRequest, make_case, plan_frames, write_case
from bubbletrack.model.cases.bundle import open_bundle
from bubbletrack.model.cases.replay import replay_case, shift_examples
from bubbletrack.model.cases.schema import ROLE_EXAMPLE
from bubbletrack.model.image_io import imread_unicode
from bubbletrack.model.state import update_state
from bubbletrack.model.teach import BUBBLE, MLExamples

ROI = ((11, 110), (21, 150))
TAUGHT = (2, 7)


@pytest.fixture
def ml_state(tmp_path):
    state = fit_all(folder_state(S.write_recording(tmp_path / "rec"), roi=ROI, mode="edge",
                                 reference_frame=0), reference=0)
    for f in TAUGHT:
        rc, cc, r = S.circle(f)
        set_manual(state, f, r, rc, cc)
    strokes = MLExamples().paint(5, BUBBLE, [(61, 81), (61, 82)])
    return update_state(state, ml_assist=True, ml_examples=strokes)


def _case(state, start=8, end=9):
    return make_case(state, CaseRequest(start=start, end=end, reference_index=0,
                                        max_radius=max_radius(state)),
                     imread_unicode(state.images[0]).shape)


def test_example_frames_are_stored_but_not_measured():
    plan = dict(plan_frames(5, 6, 10, reference=0, context=1, examples=(2, 5, 7, 12)))
    assert plan == {0: "reference", 2: ROLE_EXAMPLE, 4: "context", 5: "selected", 6: "selected",
                    7: "context"}


def test_case_records_switch_examples_and_their_frames(ml_state):
    case = _case(ml_state)
    params = case.detection.params
    examples = MLExamples.from_dict(params["ml_examples"])
    assert params["ml_assist"] is True
    assert set(examples.verified) == set(TAUGHT) and 5 in examples.strokes
    assert case.frame(2).role == ROLE_EXAMPLE
    assert 2 not in [f.index for f in case.measured_frames()]


def test_plain_auto_cases_have_no_example_frames(ml_state):
    case = _case(update_state(ml_state, ml_assist=False))
    assert case.detection.params["ml_assist"] is False
    assert ROLE_EXAMPLE not in {f.role for f in case.frames}


def test_examples_move_into_the_crop():
    ex = MLExamples().paint(1, BUBBLE, [(5, 5), (30, 40)]).verify(2, (61.0, 81.0, 20.0))
    moved = shift_examples(ex, (10, 20), (100, 130))
    assert moved.strokes[1].bubble == frozenset({(20, 20)})        # (5, 5) is outside the crop
    assert moved.verified[2] == (51.0, 61.0, 20.0)


def test_replay_retrains_ml_assist_from_the_bundle(ml_state, tmp_path):
    pytest.importorskip("sklearn")
    path = tmp_path / "c.zip"
    write_case(path, _case(ml_state), lambda i: imread_unicode(ml_state.images[i]))
    with open_bundle(path) as bundle:
        case = bundle.case
        replayed = replay_case(case, bundle.read_frame)
        off = replace(case, detection=replace(case.detection,
                                              params={**case.detection.params, "ml_assist": False}))
        plain = replay_case(off, bundle.read_frame)
    assert 2 not in replayed and replayed[6].accepted                 # an example frame is not scored
    assert replayed[6].radius == pytest.approx(S.RADII[6], rel=0.1)
    assert replayed[6].center == pytest.approx(S.circle(6)[:2], abs=3.0)  # original-frame coordinates
    assert not plain[6].accepted                                       # plain Auto fails there
