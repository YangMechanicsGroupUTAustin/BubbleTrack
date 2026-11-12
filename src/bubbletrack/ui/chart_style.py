"""Marker encoding for the R(t) chart: shape = provenance, colour = score.

Colour-blind safe by construction: the score ramp is a segment of viridis
(perceptually uniform, readable with every common colour-vision deficiency),
reversed so that *poor* fits are the brightest marks on the dark plot
background, and provenance is carried by marker shape, never by colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import pyqtgraph as pg

from bubbletrack.model.provenance import FitSource

# Viridis samples from t = 0.375 (blue-teal) to t = 1.0 (yellow).  Score 1
# (best) maps to the first stop, score 0 (worst) to the last.
RAMP_STOPS: tuple[str, ...] = (
    "#2c728e", "#21918c", "#27ad81", "#5ec962", "#addc30", "#fde725",
)
RAMP_BINS = 32
UNSCORED_COLOUR = "#a1a1aa"  # no score (area mode, or a live run)
SURFACE_COLOUR = "#0c0d12"   # plot background, used as the marker ring
FLAG_RING_COLOUR = "#fafafa"

SOURCE_SYMBOLS: dict[int, str] = {
    int(FitSource.NONE): "o",
    int(FitSource.AUTO): "o",
    int(FitSource.MANUAL): "d",
    int(FitSource.REFIT): "s",
    int(FitSource.NEIGHBOR): "t1",
    int(FitSource.TRACKED): "h",
}
LOCKED_SYMBOL = "star"
SYMBOL_SIZES = {"o": 9, "d": 11, "s": 9, "t1": 11, "h": 10, "star": 14}

# (glyph, label) pairs for the legend, in display order.
SHAPE_LEGEND: tuple[tuple[str, str], ...] = (
    ("●", "auto"), ("◆", "manual"), ("■", "refit"),
    ("▲", "neighbour"), ("⬢", "tracked"), ("★", "locked"),
)


def _hex_to_rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def ramp_colour(score: float) -> str:
    """Hex colour of a score in [0, 1] (1 = best); grey for NaN."""
    if score is None or not np.isfinite(score):
        return UNSCORED_COLOUR
    t = 1.0 - float(np.clip(score, 0.0, 1.0))  # 0 = best stop
    pos = t * (len(RAMP_STOPS) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(RAMP_STOPS) - 1)
    rgb = (1 - (pos - lo)) * _hex_to_rgb(RAMP_STOPS[lo]) + (pos - lo) * _hex_to_rgb(RAMP_STOPS[hi])
    return "#" + "".join(f"{int(round(c)):02x}" for c in rgb)


def ramp_bins(scores: np.ndarray) -> np.ndarray:
    """Quantise scores to brush-table indices; -1 marks an unscored point."""
    s = np.asarray(scores, dtype=float)
    bins = np.full(s.shape, -1, dtype=int)
    ok = np.isfinite(s)
    bins[ok] = np.rint(np.clip(s[ok], 0.0, 1.0) * (RAMP_BINS - 1)).astype(int)
    return bins


@lru_cache(maxsize=1)
def _brush_table() -> tuple:
    """Shared QBrush objects (one per ramp bin, plus the unscored grey last)."""
    brushes = [pg.mkBrush(ramp_colour(i / (RAMP_BINS - 1))) for i in range(RAMP_BINS)]
    return tuple(brushes + [pg.mkBrush(UNSCORED_COLOUR)])


def score_brushes(scores: np.ndarray | None, n: int) -> list:
    """One brush per point; all grey when *scores* is None."""
    table = _brush_table()
    if scores is None:
        return [table[-1]] * n
    return [table[b] for b in ramp_bins(scores)]


def marker_symbols(sources: np.ndarray | None, locked: np.ndarray | None,
                   n: int) -> np.ndarray:
    """pyqtgraph symbol names: shape by fit source, a star when locked."""
    if sources is None:
        symbols = np.full(n, "o", dtype=object)
    else:
        symbols = np.array(
            [SOURCE_SYMBOLS.get(int(s), "o") for s in np.asarray(sources)], dtype=object,
        )
    if locked is not None:
        symbols[np.asarray(locked, dtype=bool)] = LOCKED_SYMBOL
    return symbols


def marker_sizes(symbols: np.ndarray) -> np.ndarray:
    return np.array([SYMBOL_SIZES.get(s, 9) for s in symbols], dtype=float)


def flag_mask(flags: Sequence | np.ndarray | None, n: int) -> np.ndarray:
    """Boolean per frame: True when the frame carries any flag."""
    if flags is None:
        return np.zeros(n, dtype=bool)
    if isinstance(flags, np.ndarray) and flags.dtype == bool:
        return flags.copy()
    return np.array([bool(f) for f in flags], dtype=bool)


def source_name(source: int | None, locked: bool = False) -> str:
    if locked:
        return "locked"
    try:
        return FitSource(int(source)).name.lower() if source is not None else ""
    except ValueError:
        return ""


def ramp_gradient_css() -> str:
    """Stylesheet gradient matching the marker ramp (low score on the left)."""
    stops = ", ".join(
        f"stop:{i / (len(RAMP_STOPS) - 1):.2f} {c}"
        for i, c in enumerate(reversed(RAMP_STOPS))
    )
    return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops})"


def shape_legend_html() -> str:
    """Legend text for the marker shapes and the flag ring."""
    shapes = "&nbsp;&nbsp;".join(
        f'<span style="color:#e4e4e7;">{glyph}</span> {name}' for glyph, name in SHAPE_LEGEND
    )
    return (f'<span style="color:#a1a1aa; font-size:11px;">{shapes}&nbsp;&nbsp;'
            f'<span style="color:{FLAG_RING_COLOUR};">\u25EF</span> flagged</span>')
