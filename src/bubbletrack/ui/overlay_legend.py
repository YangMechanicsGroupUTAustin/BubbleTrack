"""Overlay colours of the image panels and their key in the panel's top-left corner.

Each colour means one thing everywhere: red = the measured outline, indigo =
the circle with the result's radius, yellow dashes = Auto's guess (not a
result), green box = the ROI.  An image panel adds one key entry per kind of
overlay it draws (:func:`default_text` names the known colours) and shows the
key in a small, mouse-transparent box fixed to the viewport: it does not move
or scale with zoom and pan, and hides when nothing is drawn or when the panel
is too small for it to stay in a corner.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QAbstractScrollArea, QWidget

OUTLINE_COLOUR = "#ef4444"  # the measured outline (red)
CIRCLE_COLOUR = "#6366f1"  # the circle with the result's radius (indigo)
GUESS_COLOUR = "#fde047"  # Auto's guess: yellow dashes ...
GUESS_UNDER_COLOUR = "#18181b"  # ... over a thin dark line (visible on the light Auto view)
ROI_COLOUR = "#10b981"  # the ROI rectangle (green)
USER_CIRCLE_COLOUR = "#f59e0b"  # a drawn circle that was not stored (amber, dashed)
FOUND_COLOUR = "#a1a1aa"  # what Auto found instead of a drawn circle (grey)
PAINT_BUBBLE_COLOUR = "#22c55e"  # ML assist: painted bubble / P(bubble) tint
PAINT_BACKGROUND_COLOUR = "#ef4444"  # ML assist: painted background

KINDS = ("line", "dash", "dots", "box", "fill")

# Default key text of an overlay drawn in a known colour, by (colour, kind).
_DEFAULT_TEXTS: dict[tuple[str, str], str] = {
    (OUTLINE_COLOUR, "line"): "outline (measured)",
    (OUTLINE_COLOUR, "dots"): "edge points",
    # True in every mode; callers that know the method say "equal-area" / "fitted".
    (CIRCLE_COLOUR, "line"): "circle (radius R)",
    (GUESS_COLOUR, "dash"): "Auto's guess (not accepted)",
    (USER_CIRCLE_COLOUR, "dash"): "your circle (not stored)",
    (FOUND_COLOUR, "line"): "Auto found (not stored)",
    (FOUND_COLOUR, "dots"): "Auto found (not stored)",
    (ROI_COLOUR, "box"): "ROI",
}
# Key order: lines (results first, then the guess and the circle tool), then
# patches in the order they were added, the ROI box last.
_LINE_RANK = {c: k for k, c in enumerate((OUTLINE_COLOUR, CIRCLE_COLOUR, GUESS_COLOUR,
                                          USER_CIRCLE_COLOUR, FOUND_COLOUR))}
_KIND_GROUP = {"line": 0, "dash": 0, "dots": 0, "fill": 1, "box": 2}

# Look of the key.
FONT_PX = 11
PAD_X, PAD_Y, MARGIN = 7, 4, 6  # inner padding, distance from the viewport corner
SAMPLE_W, SAMPLE_GAP = 22, 6
MAX_WIDTH_SHARE, MAX_HEIGHT_SHARE = 0.6, 0.5  # larger than this: hidden, never covers the image
BACKGROUND = QColor(9, 9, 11, 190)
TEXT_COLOUR = QColor("#e4e4e7")
_RELAYOUT_EVENTS = (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show)


@dataclass(frozen=True)
class LegendEntry:
    """One key line: a sample of the overlay and what it is."""

    text: str
    colour: str
    kind: str = "line"  # one of KINDS
    under: str | None = None  # a wider dark line drawn below the sample (Auto's guess)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"legend kind must be one of {KINDS}, not {self.kind!r}")


def default_text(colour: str, kind: str) -> str | None:
    """Key text of an overlay drawn in a known colour, or None."""
    return _DEFAULT_TEXTS.get((QColor(colour).name(), kind))


def make_entry(colour: str, kind: str, legend: str | None) -> LegendEntry | None:
    """The key entry of one drawn overlay.

    *legend* None: the default text for a known colour (none for an unknown
    one, e.g. the dark line under Auto's guess); "": no entry; else the text.
    """
    text = default_text(colour, kind) if legend is None else legend
    if not text:
        return None
    name = QColor(colour).name()
    under = GUESS_UNDER_COLOUR if name == GUESS_COLOUR else None
    return LegendEntry(text, name, kind, under)


def _rank(entry: LegendEntry) -> tuple[int, int]:
    group = _KIND_GROUP.get(entry.kind, 0)
    return group, (_LINE_RANK.get(entry.colour, len(_LINE_RANK)) if group == 0 else 0)


class OverlayLegend(QWidget):
    """The key box over the top-left corner of *host*'s viewport.

    It is a child of the scroll area itself, not of its viewport, so
    scrolling (panning) the image never moves it.
    """

    def __init__(self, host: QAbstractScrollArea) -> None:
        super().__init__(host)
        self._entries: list[LegendEntry] = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._font = QFont(self.font())
        self._font.setPixelSize(FONT_PX)
        # The viewport's own events come after the scroll area laid it out
        # (the host's Resize arrives before), and a border change moves it.
        self._port = host.viewport()
        self._port.installEventFilter(self)
        self.hide()

    # -- content ---------------------------------------------------------- #

    def entries(self) -> tuple[LegendEntry, ...]:
        return tuple(self._entries)

    def texts(self) -> list[str]:
        return [e.text for e in self._entries]

    def add(self, entry: LegendEntry | None) -> None:
        """Add *entry* (once per text), keeping the key order."""
        if entry is None or entry.text in self.texts():
            return
        self._entries = sorted([*self._entries, entry], key=_rank)
        self._relayout()

    def clear(self) -> None:
        self._entries = []
        self.hide()

    # -- layout ----------------------------------------------------------- #

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt override)
        # Type first: no wrapper is touched while the panel is being destroyed.
        if event.type() in _RELAYOUT_EVENTS and obj is self._port:
            self._relayout()
        return False

    def _row_height(self) -> int:
        return QFontMetrics(self._font).height() + 2

    def content_size(self) -> tuple[int, int]:
        fm = QFontMetrics(self._font)
        text_w = max((fm.horizontalAdvance(e.text) for e in self._entries), default=0)
        width = 2 * PAD_X + SAMPLE_W + SAMPLE_GAP + text_w
        return width, 2 * PAD_Y + self._row_height() * len(self._entries)

    def _relayout(self) -> None:
        if not self._entries:
            self.hide()
            return
        width, height = self.content_size()
        port = self._port.geometry()
        if width > port.width() * MAX_WIDTH_SHARE or height > port.height() * MAX_HEIGHT_SHARE:
            self.hide()  # a key that would cover the image is worse than none
            return
        self.setGeometry(port.left() + MARGIN, port.top() + MARGIN, width, height)
        self.show()
        self.raise_()
        self.update()

    # -- painting --------------------------------------------------------- #

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BACKGROUND)
        painter.drawRoundedRect(QRectF(self.rect()), 4, 4)
        painter.setFont(self._font)
        row = self._row_height()
        for k, entry in enumerate(self._entries):
            top = PAD_Y + k * row
            _draw_sample(painter, entry, QRectF(PAD_X, top, SAMPLE_W, row))
            painter.setPen(TEXT_COLOUR)
            text_rect = QRectF(PAD_X + SAMPLE_W + SAMPLE_GAP, top, self.width(), row)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             entry.text)
        painter.end()


def _draw_sample(painter: QPainter, entry: LegendEntry, rect: QRectF) -> None:
    """A short picture of the overlay: a line, dashes, dots, a box or a patch."""
    colour = QColor(entry.colour)
    mid = rect.center().y()
    left, right = rect.left() + 1, rect.right() - 1
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if entry.kind in ("line", "dash"):
        if entry.under:
            painter.setPen(QPen(QColor(entry.under), 4.0))
            painter.drawLine(QPointF(left, mid), QPointF(right, mid))
        pen = QPen(colour, 2.0)
        if entry.kind == "dash":
            pen.setDashPattern([2.5, 1.5])
        painter.setPen(pen)
        painter.drawLine(QPointF(left, mid), QPointF(right, mid))
    elif entry.kind == "dots":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        for x in (left + 3, rect.center().x(), right - 3):
            painter.drawEllipse(QPointF(x, mid), 1.8, 1.8)
    else:  # "box" outline or "fill" patch
        box = QRectF(rect.center().x() - 6, mid - 4.5, 12, 9)
        if entry.kind == "fill":
            colour.setAlpha(170)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
        else:
            painter.setPen(QPen(colour, 1.5))
        painter.drawRect(box)
