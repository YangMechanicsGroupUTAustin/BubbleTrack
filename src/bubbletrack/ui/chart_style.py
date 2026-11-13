"""Look of the R(t) chart: marker shape = who decided, colour = score; the plot style.

Two shapes only: a dot for results the software computed (Fit All, refits,
tracking) and a diamond for results the user decided (manual fits and locked
frames).  Colour runs from red (poor) through yellow to green (good); frames
without a score are grey.  The exact source and any flags are in the hover
text, and flagged frames are listed in the Review queue rather than marked
on the chart (rings overlapped badly on dense series).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyqtgraph as pg

from bubbletrack.model.provenance import USER_CONFIRMED_SOURCES, FitSource

# Green -> yellow -> red, bright enough for the dark plot background.  Score 1
# (best) maps to the first stop, score 0 (worst) to the last.
RAMP_STOPS: tuple[str, ...] = (
    "#22c55e", "#a3e635", "#facc15", "#fb923c", "#ef4444",
)
RAMP_BINS = 32
UNSCORED_COLOUR = "#a1a1aa"  # no score (area mode, or a live run)
SURFACE_COLOUR = "#0c0d12"   # plot background, used as the marker outline
AXIS_LABEL_STYLE = {"color": "#a1a1aa", "font-size": "11px"}  # small: the plot keeps the height

# Computed results are dots; a result the user decided (a manual fit, or a
# drawn circle while it counts as confirmed) gets the diamond.
SOURCE_SYMBOLS: dict[int, str] = {int(s): "d" for s in USER_CONFIRMED_SOURCES}
LOCKED_SYMBOL = "d"  # accepted in review: decided by the user, like a manual fit
SYMBOL_SIZES = {"o": 9, "d": 12}

# (glyph, label) pairs for the legend, in display order.
SHAPE_LEGEND: tuple[tuple[str, str], ...] = (
    ("●", "computed"), ("◆", "manual / locked"),
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


def style_plot(plot: pg.PlotWidget) -> None:
    """Dark plot with compact axes, so the plot area gets the chart's height.

    pyqtgraph's own context menu and auto-range button are off: they could
    switch on log axes or transforms, and the view is managed by ``chart_view``.
    """
    plot.setBackground(SURFACE_COLOUR)
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.setMenuEnabled(False)
    plot.hideButtons()
    plot.plotItem.layout.setContentsMargins(2, 6, 10, 0)
    for axis_name, label in (("left", "Radius (px)"), ("bottom", "Frame")):
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(255, 255, 255, 15))
        axis.setTextPen(pg.mkPen("#71717a"))
        axis.setStyle(tickTextOffset=4)
        axis.label.document().setDocumentMargin(1)  # the label's text box, not 4 px of air
        plot.setLabel(axis_name, label, **AXIS_LABEL_STYLE)  # sizes the axis for it
