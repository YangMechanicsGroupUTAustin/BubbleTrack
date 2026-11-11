"""The vectorised sector coverage must equal the original per-point loop."""

import numpy as np
import pytest

from bubbletrack.model.constants import QUALITY_N_SECTORS
from bubbletrack.model.quality import _sector_coverage


def _loop_reference(angles):
    sector_edges = np.linspace(-np.pi, np.pi, QUALITY_N_SECTORS + 1)
    hit = set()
    for a in angles:
        idx = int(np.searchsorted(sector_edges, a, side="right")) - 1
        hit.add(max(0, min(idx, QUALITY_N_SECTORS - 1)))
    return len(hit) / QUALITY_N_SECTORS


@pytest.mark.parametrize("seed", range(20))
def test_matches_loop_on_random_arcs(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 300))
    lo = rng.uniform(-np.pi, np.pi)
    angles = lo + rng.uniform(0, rng.uniform(0.1, 2 * np.pi), n)
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    assert _sector_coverage(angles) == _loop_reference(angles)


def test_boundary_angles():
    edges = np.linspace(-np.pi, np.pi, QUALITY_N_SECTORS + 1)
    angles = np.concatenate([edges, [np.nextafter(np.pi, 0), -np.pi]])
    assert _sector_coverage(angles) == _loop_reference(angles)
