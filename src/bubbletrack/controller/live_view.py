"""Fit's live view: one frame drawn while the run goes on.

Interim version with the interface of fix/ux-visual's ``live_view`` (which
draws the frame exactly as the result is drawn after the run, with the Auto
view on the right).  When that branch is merged, its file replaces this one
as is: ``AutoController`` already calls ``show_live_frame(state, window, idx,
binary_roi)``.
"""

from __future__ import annotations

import logging

import numpy as np

from bubbletrack.controller.display_mixin import current_source
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.state import AppState

logger = logging.getLogger(__name__)


def show_live_frame(state: AppState, window, idx: int,
                    binary_roi: np.ndarray | None = None) -> None:
    """Draw frame *idx* (its result already in *state*) and the worker's mask.

    Never raises: a frame that cannot be read is skipped (the run goes on).
    """
    try:
        cur_img, _, _, _ = normalize_source(
            current_source(state, idx), state.img_thr, state.gridx, state.gridy,
            gaussian_sigma=state.gaussian_sigma, clahe_clip=state.clahe_clip,
        )
        panel = window.original_panel
        panel.set_image(cur_img)
        panel.draw_roi_rect(state.gridx, state.gridy)
        radius = float(state.radius[idx])
        edge_xy = state.circle_xy[idx]
        if radius > 0 and edge_xy is not None and len(edge_xy):
            rc, cc = state.circle_fit_par[idx]
            count = 0 if state.region_count is None else int(state.region_count[idx])
            if count == 0:
                panel.draw_circle(rc, cc, radius, "#6366f1")
            panel.draw_points(edge_xy, "#ef4444", 2.0)
    except Exception:
        logger.debug("Live view update failed for frame %d", idx + 1, exc_info=True)
    if binary_roi is not None:  # a visual swap only, as in display_frame
        window.binary_panel.set_image(~binary_roi if state.invert_mask else binary_roi)
