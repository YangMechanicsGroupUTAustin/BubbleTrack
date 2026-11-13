"""What the window shows of a frame the moment a running Fit measured it."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from bubbletrack.controller.display_mixin import current_source
from bubbletrack.model.conventions import frame_to_display
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.state import AppState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveFit:
    """One frame's result as the worker reports it."""

    radius: float
    edge_xy: np.ndarray | None
    binary_roi: np.ndarray | None
    rc: float
    cc: float
    count: int


def show_live_frame(state: AppState, window, idx: int, fit: LiveFit) -> None:
    """Frame *idx* with its circle and edge points, its mask and the R(t) chart so far.

    The chart is drawn without colours for speed; they come once the run ends.
    """
    window.status_bar.update_frame(frame_to_display(idx), state.total_frames)
    try:
        cur_img, _, _, _ = normalize_source(
            current_source(state, idx), state.img_thr, state.gridx, state.gridy,
            gaussian_sigma=state.gaussian_sigma, clahe_clip=state.clahe_clip,
        )
        window.original_panel.set_image(cur_img)
        window.original_panel.draw_roi_rect(state.gridx, state.gridy)
        if fit.radius > 0 and fit.edge_xy is not None and fit.edge_xy.shape[0] > 0:
            if fit.count == 0:
                window.original_panel.draw_circle(fit.rc, fit.cc, fit.radius, "#6366f1")
            window.original_panel.draw_points(fit.edge_xy, "#ef4444", 2.0)
    except Exception:
        logger.debug("Live view update failed for frame %d", idx, exc_info=True)
    # The worker's mask (no disk I/O); invert_mask is a visual swap, as elsewhere.
    if fit.binary_roi is not None:
        window.binary_panel.set_image(~fit.binary_roi if state.invert_mask else fit.binary_roi)
    if fit.radius > 0 and state.radius is not None:
        window.radius_chart.plot_all(np.arange(1, len(state.radius) + 1), state.radius)
