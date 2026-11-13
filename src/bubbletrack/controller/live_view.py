"""Fit's live view: one frame drawn while the run goes on, as the result will look.

``show_live_frame`` draws the frame whose result the run just stored with the
same overlays and names as after the run (``draw_fit_overlay``: "fitted circle",
or "equal-area circle" with "Bubble may be non-round", and the measured outline),
and fills the right panel with what it shows after the run: the Auto view in
Auto mode, the method's binary mask otherwise.
"""

from __future__ import annotations

import logging

import numpy as np

from bubbletrack.controller.display_mixin import (
    MASK_TITLE, current_source, draw_fit_overlay, show_engine_view,
)
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.state import AppState

logger = logging.getLogger(__name__)


def show_live_frame(state: AppState, window, idx: int,
                    binary_roi: np.ndarray | None = None) -> None:
    """Draw frame *idx* (its result already in *state*) on both image panels.

    *binary_roi* is the worker's mask of the frame; only the classic methods
    show it (Auto mode shows the Auto view instead).  Never raises: a frame
    that cannot be read is skipped (the run goes on).
    """
    try:
        cur_img, _, _, _ = normalize_source(
            current_source(state, idx), state.img_thr, state.gridx, state.gridy,
            gaussian_sigma=state.gaussian_sigma, clahe_clip=state.clahe_clip,
        )
        panel = window.original_panel
        panel.set_image(cur_img)
        panel.draw_roi_rect(state.gridx, state.gridy)
        draw_fit_overlay(state, panel, idx)
        if state.detection_mode == "edge":
            show_engine_view(state, window, idx)
        elif binary_roi is not None:
            window.binary_panel.set_title(MASK_TITLE)
            window.binary_panel.set_image(~binary_roi if state.invert_mask else binary_roi)
    except Exception:
        logger.debug("Live view update failed for frame %d", idx + 1, exc_info=True)
