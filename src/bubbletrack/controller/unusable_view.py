"""The Auto view's "not usable" fill: reference pixels Auto cannot use, shown as what they are.

Pixels too dark (no light) or saturated in the reference frame carry no data: the engine
ignores them in every frame and gives them the background's value, so in the Auto view they
would look like plain background.  They are drawn as a translucent violet fill with a key
line, under the outline; when they cover more than ``NOTE_SHARE`` of the ROI the panel's
title says how much (an amber note) and its tooltip what to do.  The pixels depend only on
the reference frame and the ROI (``edge_support.unusable_pixels`` keeps them while frames
are scrubbed).
"""

from __future__ import annotations

import logging

import numpy as np

from bubbletrack.model.state import AppState
from bubbletrack.ui.overlay_legend import UNUSABLE_COLOUR, make_entry

logger = logging.getLogger(__name__)

FILL_ALPHA = 110
FILL_Z = -0.5  # above the frame (-1), under the outline and Auto's guess (0)
NOTE_SHARE = 0.10  # above this share of the ROI the title says how much is not usable
UNUSABLE_KEY = make_entry(UNUSABLE_COLOUR, "fill", None)  # "not usable (reference)"
NOTE = "· {percent} % not usable"
NOTE_TIP = ("Violet: {percent} % of the ROI is not usable. These pixels are too dark or saturated\n"
            "in the reference frame, so Auto cannot see the bubble there.\n"
            "Try another reference frame (Use this frame) or an ROI that leaves them out.")
_FILL_RGBA = np.array([*(int(UNUSABLE_COLOUR[i:i + 2], 16) for i in (1, 3, 5)), FILL_ALPHA], np.uint8)


def unusable_or_none(state: AppState) -> np.ndarray | None:
    """The ROI's pixels Auto cannot use with the reference frame; None without a reference
    or when the reference cannot be read (the Auto view is shown without the fill)."""
    from bubbletrack.controller import edge_support  # edge_support imports display_mixin

    try:
        return edge_support.unusable_pixels(state)
    except Exception as exc:  # noqa: BLE001 - a missing fill must not hide the view
        logger.warning("Unusable reference pixels unavailable: %s", exc)
        return None


def fill_rgba(mask: np.ndarray) -> np.ndarray:
    """(H, W, 4) uint8: *mask* in translucent violet, transparent elsewhere."""
    out = np.zeros((*mask.shape, 4), np.uint8)
    out[mask] = _FILL_RGBA
    return out


def draw_unusable(state: AppState, panel) -> float:
    """Fill the Auto view's unusable pixels (with their key line); their share of the ROI."""
    mask = unusable_or_none(state)
    if mask is None or not mask.any():
        return 0.0
    panel.draw_rgba_overlay(fill_rgba(mask), (UNUSABLE_KEY,), z=FILL_Z)
    return float(mask.mean())


def title_note(share: float) -> tuple[str, str]:
    """(amber note, what it means and what to do) for the Auto view's title; both empty
    up to ``NOTE_SHARE`` of the ROI (the fill and its key line say enough)."""
    if not share > NOTE_SHARE:
        return "", ""
    percent = round(100 * share)
    return NOTE.format(percent=percent), NOTE_TIP.format(percent=percent)
