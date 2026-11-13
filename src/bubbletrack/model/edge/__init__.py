"""Edge engine: sub-pixel bubble edge measurement for HPV-X2 shadowgraphs.

Public API::

    from bubbletrack.model.edge import (
        EdgeParams, EdgeResult, detect_edge, choose_reference_frame, ENGINE_VERSION,
    )

    ref_idx = choose_reference_frame(read_raw, n_frames)          # 0-based index or None
    ref = read_raw(ref_idx) if ref_idx is not None else None
    res = detect_edge(raw, gridx, gridy, reference=ref, params=EdgeParams(seed=(row, col)))
    if res.accepted:
        use(res.radius, res.rc, res.cc)                            # 1-based full-image coords

``radius`` is the robust circle fit through the bubble wall (``r_circle``) by default; with
``EdgeParams(non_round=True)`` ("Bubble may be non-round") it is the area-equivalent radius
of the bubble region (any shape; completed with the circle fit where the ROI border hides it
and the visible wall agrees).  ``r_area`` is the region's area radius either way.

No Qt imports: this package belongs to the model layer.
"""

from bubbletrack.model.edge.engine import ENGINE_VERSION, EdgeParams, EdgeResult, detect_edge
from bubbletrack.model.edge.reference import choose_reference_frame

__all__ = ["ENGINE_VERSION", "EdgeParams", "EdgeResult", "choose_reference_frame", "detect_edge"]
