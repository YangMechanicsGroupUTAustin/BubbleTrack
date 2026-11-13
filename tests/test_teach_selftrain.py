"""Self-training: after Fit All, ML assist is taught by Auto's own accepted results and
re-measures only the frames that have no result."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bubbletrack.model.teach import MeasureConfig, SelfTrainConfig, plan_self_training, suggest_frames

N = 100


def _results(n=N, failed=range(60, 100)):
    """A bubble that grows and collapses at (120, 200); *failed* frames have no result (-1)."""
    t = np.arange(n)
    radius = 10 + 60 * np.sin(np.pi * t / n)
    centres = np.column_stack([np.full(n, 120.0), np.full(n, 200.0)])
    conf = np.full(n, 0.4)
    for i in failed:
        radius[i] = -1.0
        centres[i] = np.nan
        conf[i] = 0.1
    return radius, centres, conf


def test_trains_on_the_frames_suggest_frames_picks_among_accepted_results():
    radius, centres, conf = _results()
    plan = plan_self_training(radius, centres, conf)
    assert plan.run, plan.reason
    expected = suggest_frames(np.where(radius > 0, radius, np.nan), conf, n=2)
    assert tuple(sorted(plan.train)) == expected
    for f, (rc, cc, r) in plan.train.items():
        assert (rc, cc, r) == (120.0, 200.0, radius[f])


def test_measures_only_frames_without_a_result_except_protected_and_training_frames():
    radius, centres, conf = _results()
    plan = plan_self_training(radius, centres, conf, protected=[61, 62])
    assert plan.measure == tuple(i for i in range(60, 100) if i not in (61, 62))
    assert not set(plan.measure) & set(plan.train)


def test_examples_are_the_training_circles_as_verified_frames():
    radius, centres, conf = _results()
    plan = plan_self_training(radius, centres, conf)
    ex = plan.examples()
    assert dict(ex.verified) == {f: c for f, c in plan.train.items()}
    assert not ex.strokes


def test_does_not_run_when_almost_every_frame_has_a_result():
    radius, centres, conf = _results(failed=range(97, 100))          # 3 % without a result
    plan = plan_self_training(radius, centres, conf)
    assert not plan.run and plan.train == {} and plan.measure == ()
    assert "without a result" in plan.reason


def test_does_not_run_with_too_few_accepted_results():
    radius, centres, conf = _results(failed=range(9, 100))           # 9 accepted frames
    plan = plan_self_training(radius, centres, conf)
    assert not plan.run and "accepted" in plan.reason
    assert plan_self_training(radius, centres, conf, config=SelfTrainConfig(min_accepted=5)).run


def test_results_below_the_gate_or_without_confidence_are_not_trusted():
    radius, centres, conf = _results()
    conf[:60] = np.nan                                               # a legacy mode: no confidence
    assert not plan_self_training(radius, centres, conf).run
    conf[:60] = 0.2                                                  # stored, but below the gate
    assert not plan_self_training(radius, centres, conf).run


def test_measure_config_raises_only_the_outline_gate():
    base = MeasureConfig()
    cfg = SelfTrainConfig().measure_config(base)
    assert cfg.confidence_gate == pytest.approx(0.5)
    assert cfg.policy == base.policy and cfg.refine == base.refine and cfg.agree_radius == base.agree_radius


def test_bad_input():
    radius, centres, conf = _results()
    with pytest.raises(ValueError):
        plan_self_training(radius[:-1], centres, conf)
    with pytest.raises(ValueError):
        plan_self_training(radius, centres[:, :1], conf)
    with pytest.raises(ValueError):
        SelfTrainConfig(n_frames=0)
    with pytest.raises(ValueError):
        SelfTrainConfig(min_failed_fraction=1.5)
    with pytest.raises(ValueError):
        SelfTrainConfig(outline_gate=math.nan)
