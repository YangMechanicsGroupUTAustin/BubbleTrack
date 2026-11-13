"""Shared wording of what users read: counts with proper plurals, the score's name.

Auto's per-frame score is called "confidence" everywhere the user sees it and
shown on one scale, 0.00 (doubtful) to 1.00 (certain).  The classic methods
have no confidence: their score (how well the edge points fit a circle) is the
"fit quality", on the same scale.  The chart (``chart_legend.CONFIDENCE`` /
``FIT_QUALITY``) uses the same words.
"""

from __future__ import annotations

CONFIDENCE = "confidence"  # Auto's score (chart legend, hover, header, Review)
FIT_QUALITY = "fit quality"  # the classic methods' score


def plural(n: int, word: str, many: str | None = None) -> str:
    """'1 frame', '2 frames' ('1 fix', '3 fixes' with *many*)."""
    return f"{n} {word if n == 1 else (many or word + 's')}"


def score_name(detection_mode: str) -> str:
    """'confidence' for Auto, 'fit quality' for the classic methods."""
    return CONFIDENCE if detection_mode == "edge" else FIT_QUALITY
