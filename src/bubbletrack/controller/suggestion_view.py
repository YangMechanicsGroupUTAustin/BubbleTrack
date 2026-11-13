"""How Auto's rejected guess (``model.suggestion``) looks: a yellow dashed outline.

Drawn on the image panel and on the Auto view while the frame has no result;
the panel's colour key names it, its tooltip and the header say what it is
and how to accept it.  No chart marker: a guess is not a result.
"""

from __future__ import annotations

import numpy as np

from bubbletrack.model.state import AppState
from bubbletrack.model.suggestion import Suggestion, shown_suggestion
from bubbletrack.ui.image_panel import is_ordered_contour
from bubbletrack.ui.overlay_legend import GUESS_COLOUR, GUESS_UNDER_COLOUR
from bubbletrack.ui.review_text import NOT_A_BUBBLE_HINT

# Yellow dashes over a slightly wider dark line: the dashes show on dark
# frames, the dark line (gaps and a thin rim) on the light Auto view.
GUESS_GAP_COLOUR = GUESS_UNDER_COLOUR
GUESS_WIDTH = 1.5
GUESS_UNDER_WIDTH = 2.5
GUESS_STATUS_COLOUR = "#a1a1aa"


def guess_text(suggestion: Suggestion) -> str:
    """'Auto's guess (confidence 0.24) — press A in Review to accept'."""
    conf = suggestion.confidence
    score = f" (confidence {conf:.2f})" if np.isfinite(conf) else ""
    return f"Auto's guess{score} — press A in Review to accept"


def draw_guess(state: AppState, panel, idx: int, offset: np.ndarray | None = None) -> bool:
    """Draw frame *idx*'s shown guess on *panel*; False if there is none.

    *offset* is subtracted from the full-image coordinates (the ROI corner,
    for the Auto view).  An outline that is not an ordered contour, or none
    at all, is drawn as the guessed circle instead.
    """
    suggestion = shown_suggestion(state, idx)
    if suggestion is None:
        return False
    shift = np.zeros(2) if offset is None else np.asarray(offset, dtype=float)
    xy = np.asarray(suggestion.edge_xy, dtype=float).reshape(-1, 2) - shift
    for colour, width, dashed in ((GUESS_GAP_COLOUR, GUESS_UNDER_WIDTH, False),
                                  (GUESS_COLOUR, GUESS_WIDTH, True)):
        if is_ordered_contour(xy):
            panel.draw_contour(xy, colour, width, dashed=dashed)
        else:
            panel.draw_circle(suggestion.rc - shift[0], suggestion.cc - shift[1],
                              suggestion.radius, colour, width, dashed=dashed)
    panel.set_overlay_tip(f"{guess_text(suggestion)}\n{NOT_A_BUBBLE_HINT}")
    return True


def announce_guess(state: AppState, window, idx: int) -> None:
    """Header status for a displayed frame that shows a guess."""
    suggestion = shown_suggestion(state, idx)
    if suggestion is not None:
        window.header.set_status(f"Frame {idx + 1}: {guess_text(suggestion)}",
                                 GUESS_STATUS_COLOUR)
