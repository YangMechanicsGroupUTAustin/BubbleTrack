"""Edge engine: sub-pixel bubble edge measurement for HPV-X2 shadowgraphs.

Public API::

    from bubbletrack.model.edge import EdgeParams, EdgeResult, detect_edge, ENGINE_VERSION

    res = detect_edge(raw, gridx, gridy, reference=ref, params=EdgeParams(seed=(row, col)))
    if res.accepted:
        use(res.radius, res.rc, res.cc)                            # 1-based full-image coords

No Qt imports: this package belongs to the model layer.
"""

from bubbletrack.model.edge.engine import ENGINE_VERSION, EdgeParams, EdgeResult, detect_edge

__all__ = ["ENGINE_VERSION", "EdgeParams", "EdgeResult", "detect_edge"]
