"""Which frames ML assist asks the user to verify first."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.model.teach import suggest_frames


def _sizes(n=60):
    t = np.arange(n)
    return 10 + 70 * np.sin(np.pi * t / n)          # grows to Rmax at the middle, then collapses


def test_biggest_bubble_and_a_small_one():
    radius = _sizes()
    conf = np.full(len(radius), 0.8)
    pick = suggest_frames(radius, conf, n=2)
    assert len(pick) == 2
    assert radius[list(pick)].max() == pytest.approx(radius.max())
    small = radius[list(pick)].min()
    assert small <= np.percentile(radius, 40) and small >= np.percentile(radius, 10)


def test_three_frames_are_spread_over_sizes_and_distinct():
    radius = _sizes()
    pick = suggest_frames(radius, np.full(len(radius), 0.8), n=3)
    assert len(set(pick)) == 3
    assert sorted(radius[list(pick)])[1] > sorted(radius[list(pick)])[0] + 5


def test_rejected_and_missing_results_are_not_trusted_for_sizes():
    radius = _sizes()
    conf = np.full(len(radius), 0.8)
    radius[30] = 500.0                              # a wild value ...
    conf[30] = 0.1                                  # ... that Auto itself rejected
    radius[5] = -1                                  # not processed (app convention)
    assert 30 not in suggest_frames(radius, conf, n=2)


def test_without_enough_accepted_frames_the_suggestion_is_spread_in_time():
    radius = np.full(90, np.nan)
    conf = np.zeros(90)
    assert suggest_frames(radius, conf, n=2) == (30, 60)
    assert suggest_frames(radius, conf, n=1) == (45,)


def test_frames_already_verified_are_skipped():
    radius = _sizes()
    conf = np.full(len(radius), 0.8)
    first = suggest_frames(radius, conf, n=2)
    again = suggest_frames(radius, conf, n=2, exclude=first)
    assert not set(first) & set(again)


def test_bad_input():
    with pytest.raises(ValueError):
        suggest_frames([1.0, 2.0], [0.5], n=2)
    with pytest.raises(ValueError):
        suggest_frames([1.0, 2.0], [0.5, 0.5], n=0)
