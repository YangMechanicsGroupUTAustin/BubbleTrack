"""Which frames ML assist asks the user to verify first."""

from __future__ import annotations

import numpy as np
import pytest

from bubbletrack.model.teach import suggest_frames
from bubbletrack.model.teach.suggest import SIZE_RATIO, FrameSize, differ, find_pair, pick_candidate


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


# --------------------------------------------------------------------------- #
#  Teaching: different situations and the candidate frames
# --------------------------------------------------------------------------- #

def test_two_frames_differ_by_bubble_size_or_by_time():
    n = 100                                         # a quarter of the recording: 25 frames
    assert differ(FrameSize(10, 40.0), FrameSize(12, 27.0), n)       # 30 % smaller or more
    assert not differ(FrameSize(10, 40.0), FrameSize(12, 30.0), n)   # 25 % smaller, close in time
    assert differ(FrameSize(10, 40.0), FrameSize(35, 40.0), n)       # same size, far apart
    assert not differ(FrameSize(10, 40.0), FrameSize(10, 20.0), n)  # the same frame never differs
    painted = FrameSize(12)                                          # size unknown: time only
    assert not differ(painted, FrameSize(10, 80.0), n) and differ(painted, FrameSize(60, 80.0), n)


def test_a_differing_pair_is_found_even_when_the_first_frame_is_like_all_others():
    frames = [FrameSize(10, 25.0), FrameSize(11, 30.0), FrameSize(12, 20.0)]
    assert {f.frame for f in find_pair(frames, 100)} == {11, 12}     # 20 / 30 = 0.67
    assert find_pair(frames[:2], 100) is None and find_pair([], 100) is None
    preferred = find_pair(frames + [FrameSize(90, 25.0)], 100, prefer=(12,))
    assert 12 in {f.frame for f in preferred}


def _outlines(n=60):
    radius = _sizes(n)
    return radius, np.full(n, 0.8), np.ones(n, bool)


def test_first_candidate_is_a_big_measured_bubble_then_one_that_differs():
    radius, conf, trusted = _outlines()
    first = pick_candidate(radius, conf, trusted)
    assert radius[first] >= np.percentile(radius, 85)
    second = pick_candidate(radius, conf, trusted, exclude={first},
                            keep_apart=[FrameSize(first, radius[first])])
    assert radius[second] <= SIZE_RATIO * radius[first]               # a smaller bubble


def test_measured_results_come_before_autos_guesses():
    radius, conf, trusted = _outlines()
    trusted[:] = False
    trusted[[5, 50]] = True                                           # two results, many guesses
    conf[~trusted] = 0.2
    assert pick_candidate(radius, conf, trusted) in (5, 50)


def test_a_rejected_frame_and_frames_like_it_are_not_offered_first():
    radius, conf, trusted = _outlines()
    first = pick_candidate(radius, conf, trusted)
    again = pick_candidate(radius, conf, trusted, exclude={first},
                           avoid=[FrameSize(first, radius[first])])
    assert again != first and differ(FrameSize(first, radius[first]), FrameSize(again, radius[again]), 60)


def test_no_candidate_when_nothing_with_an_outline_is_left():
    radius = np.full(100, np.nan)
    radius[[1, 2, 3]] = (20.0, -1.0, 30.0)
    conf, trusted = np.full(100, 0.8), np.ones(100, bool)
    assert pick_candidate(radius, conf, trusted, exclude={1, 3}) is None
    assert pick_candidate(radius, conf, trusted, exclude={3},
                          keep_apart=[FrameSize(3, 25.0)]) is None    # frame 1: too much alike
    assert pick_candidate(radius, conf, trusted, exclude={1}) == 3
