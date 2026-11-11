"""Golden regression: existing detection modes must keep bit-identical output.

The expected values were generated from commit d0fdbc7 by ``_golden_cases.py``.
A failure here means a change altered the results of an existing mode
(intensity / gradient / area).  That must never happen silently: either fix
the change, or regenerate the golden file deliberately and say why.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from _golden_cases import (
    CASES,
    GOLDEN_PATH,
    REAL_FRAMES,
    real_frame_path,
    run_case,
    run_real,
)

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

# The golden values were generated on Windows, the platform the app is built
# and mostly used on, so there they must match bit for bit.  Other platforms
# link a different LAPACK (e.g. Apple Accelerate), so least-squares circle fits
# may differ in the last digits; allow a small relative tolerance there.
ON_REFERENCE_PLATFORM = sys.platform == "win32"
RTOL = 1e-9 if ON_REFERENCE_PLATFORM else 1e-6
ATOL = 1e-9 if ON_REFERENCE_PLATFORM else 1e-6


def _assert_matches(actual: dict, expected: dict) -> None:
    assert actual["count"] == expected["count"]
    assert actual["n_edge"] == expected["n_edge"]
    for key in ("radius", "rc", "cc", "edge_sum"):
        if expected[key] is None:
            assert actual[key] is None, f"{key}: expected NaN, got {actual[key]}"
        else:
            np.testing.assert_allclose(actual[key], expected[key], rtol=RTOL, atol=ATOL,
                                       err_msg=key)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_synthetic_case_unchanged(case, tmp_path):
    assert case.name in GOLDEN["synthetic"], "case missing from golden file"
    _assert_matches(run_case(case, tmp_path), GOLDEN["synthetic"][case.name])


def test_golden_file_covers_every_case():
    assert set(GOLDEN["synthetic"]) == {c.name for c in CASES}


@pytest.mark.parametrize("mode", ["intensity", "gradient", "area"])
@pytest.mark.parametrize("frame", REAL_FRAMES)
def test_real_frame_unchanged(frame, mode):
    key = f"{frame:03d}_{mode}"
    if key not in GOLDEN["example_1"] or not real_frame_path(frame).exists():
        pytest.skip("local example_1 data not present")
    _assert_matches(run_real(frame, mode), GOLDEN["example_1"][key])
