"""Synthetic accuracy benchmark of the Edge engine (CI-runnable, about 10 s).

53 scenes, one seed each (``_edge_synth.catalogue``): clean, contrast 5-80 %, noise x0.5-x8,
gradient / vignette, blur, no / large saturated spot, halo, clipping on one edge and in a
corner, several bubbles (with and without a seed click), debris, static features and static
speckle, ellipses up to 1.3, wobbly and rough outlines, tiny (R 6-10) and huge (R 100-112).
Error = |R - R_eq| / R_eq against the exact area-equivalent radius.  A failure is an error
above 5 % or no radius at all.  Three scenes are ambiguous by construction (two halos and a
smaller target beside a bigger distractor without a click) and are excluded from the score.

Measured at the port (engine edge-1.0.0, circle fit), on the 50 non-ambiguous scenes:
  * with the reference frame:  98.0 % within 2 %, 0 failures, median error 0.158 %
  * reference-free:            98.0 % within 2 %, 0 failures, median error 0.161 %
(the one miss is tiny_R6, 2.8 %, a 6-px bubble).  Both match the research prototype exactly.

Measured with the region layer (edge-1.1.0, area-equivalent radius):
  * with the reference frame:  98.0 % within 2 %, 0 failures, median error 0.159 %
  * reference-free:            98.0 % within 2 %, 0 failures, median error 0.162 %
(miss: tiny_R6, 2.8 %).  Ellipses improve (1.3: 0.6 % -> 0.2 %); two_touching stays at 0.07 %
because the smaller bubble is split off at the neck; debris_attached is 1.75 % (the attached
debris is part of the dark region).
"""

from __future__ import annotations

import numpy as np
import pytest

from _edge_synth import AMBIGUOUS, catalogue, click_for, render
from bubbletrack.model.edge import EdgeParams, detect_edge

WITHIN = 0.02
FAIL = 0.05


@pytest.fixture(scope="module")
def rendered():
    return [(case, *render(case.scene)) for case in catalogue()]


def _errors(rendered, use_reference: bool) -> dict[str, float]:
    errors = {}
    for case, img, meta in rendered:
        h, w = img.shape
        res = detect_edge(img, (1, h), (1, w), reference=meta["ref"] if use_reference else None,
                          params=EdgeParams(seed=click_for(case)))
        r_eq = meta["gt"]["r_eq"]
        errors[case.name] = abs(res.radius - r_eq) / r_eq if np.isfinite(res.radius) else np.inf
    return errors


def test_catalogue_size_and_ambiguous_set(rendered):
    names = [case.name for case, _, _ in rendered]
    assert 40 <= len(names) <= 60 and len(set(names)) == len(names)
    assert AMBIGUOUS <= set(names)


@pytest.mark.parametrize("use_reference", [True, False], ids=["reference", "reference_free"])
def test_accuracy_on_non_ambiguous_scenes(rendered, use_reference):
    errors = _errors(rendered, use_reference)
    scored = {k: v for k, v in errors.items() if k not in AMBIGUOUS}
    e = np.array(list(scored.values()))
    within = float(np.mean(e <= WITHIN))
    failures = sorted(k for k, v in scored.items() if v > FAIL)
    assert failures == []
    assert within >= 0.95, {k: round(100 * v, 2) for k, v in scored.items() if v > WITHIN}
    assert np.median(e) < 0.005
