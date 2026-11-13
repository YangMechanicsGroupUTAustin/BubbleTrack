"""Shared wording of what users read: counts with proper plurals, the score's name.

The per-frame score is called "confidence" everywhere the user sees it and
shown on one scale, 0.00 (doubtful) to 1.00 (certain), see
:func:`confidence_text`.
"""

from __future__ import annotations

import math

CONFIDENCE = "confidence"  # the score's one name (chart legend, hover, header, Review)
CONFIDENCE_LOW, CONFIDENCE_HIGH = "low", "high"  # the two ends of the chart's colour ramp


def plural(n: int, word: str, many: str | None = None) -> str:
    """'1 frame', '2 frames' ('1 fix', '3 fixes' with *many*)."""
    return f"{n} {word if n == 1 else (many or word + 's')}"


def confidence_text(value: float) -> str:
    """'confidence 0.87' (two decimals, 0 to 1), or '' when the fit is not scored."""
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{CONFIDENCE} {float(value):.2f}"
