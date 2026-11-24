"""Edge engine: sub-pixel bubble edge measurement for HPV-X2 shadowgraphs.

Public API::

    from bubbletrack.model.edge import (
        EdgeParams, EdgeResult, detect_edge, choose_reference_frame, recording_black_level,
        ENGINE_VERSION,
    )

    black = recording_black_level(read_raw, n_frames)             # the recording's darkest level
    ref_idx = choose_reference_frame(read_raw, n_frames, black_level=black)  # 0-based or None
    ref = read_raw(ref_idx) if ref_idx is not None else None
    res = detect_edge(raw, gridx, gridy, reference=ref,
                      params=EdgeParams(seed=(row, col), black_level=black))
    if res.accepted:
        use(res.radius, res.rc, res.cc)                            # 1-based full-image coords

``radius`` is the least-squares circle (Taubin, one outlier pass) through the bubble outline
(``r_circle``) by default; with ``EdgeParams(non_round=True)`` ("Bubble may be non-round") it
is the area-equivalent radius of the bubble region (any shape; completed with the robust ray
circle where the ROI border hides it and the visible wall agrees).  ``r_area`` is the region's
area radius and ``r_circle`` the outline circle's either way.

The black level (edge-1.7.0) is removed from the frame and the reference before the ratio, so a
constant offset of the data (an export adding counts to every pixel) does not change the result;
a raw HPV-X2 ``.dat`` keeps black 0 (``edge_pipeline.recording_black_level``).

No Qt imports: this package belongs to the model layer.
"""

from bubbletrack.model.edge.black import recording_black_level
from bubbletrack.model.edge.engine import ENGINE_VERSION, EdgeParams, EdgeResult, detect_edge
from bubbletrack.model.edge.reference import choose_reference_frame

__all__ = ["ENGINE_VERSION", "EdgeParams", "EdgeResult", "choose_reference_frame", "detect_edge",
           "recording_black_level"]
