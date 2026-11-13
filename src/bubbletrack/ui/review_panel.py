"""Review panel: the frames to check, and what to do with the one on screen.

::

    12 of 48 frames to check        [All kinds ▾]
    ●  Frames 120–180               [61]
       Auto lost the bubble
    ────────────────────────────────────────────
    Frame 125 · _135 ... the frame on screen, its advice and actions (ReviewCard)
    ✓ Frame 124 accepted.                   [Undo]
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from PyQt6.QtCore import QModelIndex, QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.controller.review_segments import (
    CATEGORIES, OTHER, ReviewSegment, queue_frames, segment_of,
)
from bubbletrack.ui.review_card import CardAdvice, DonePage, FeedbackLine, ReviewCard
from bubbletrack.ui.review_filter import KindFilter
from bubbletrack.ui.review_rows import ROW_ROLE, ReviewRowDelegate, row_data, tooltip_html
from bubbletrack.ui.review_text import (
    detail_text, frame_title, position_text, progress_text, segment_text,
)

EMPTY_BEFORE_FIT = "Frames to check show up here after a fit."
EMPTY_ALL_CLEAR = "Nothing to check: Auto is sure of every frame.\nNext: export R(t)."
EMPTY_ALL_DONE = "All done: every flagged frame is checked.\nNext: export R(t)."
EMPTY_FILTERED = "Nothing of the selected types."
_CATEGORY_ORDER = [c.key for c in CATEGORIES] + [OTHER.key]

# advisor(frame, item) -> what the card says about the frame on screen
Advisor = Callable[[int, ReviewItem | None], CardAdvice]


class ReviewPanel(QWidget):
    """The frames to check (rows) and the card for the frame on screen.

    The panel only shows and reports requests; ReviewController decides
    what they do.  A row is selected only while the shown frame lies in its
    span (its range is highlighted on the chart), so actions always hit the
    run the user is looking at.  Accept, Fix and No bubble act on the frame
    on screen, Accept all, Skip and No bubble on all on the selected row;
    afterwards Review moves on by itself.
    """

    ACTIONS = ("accept", "accept_all", "fix", "no_bubble", "no_bubble_all", "skip")
    ROW_ACTIONS = frozenset({"accept_all", "no_bubble_all", "skip"})  # need a selected row

    count_changed = pyqtSignal(int)  # frames waiting for review (every type)
    frame_activated = pyqtSignal(int)  # the user picked a row: show this frame
    action_requested = pyqtSignal(str)  # one of ACTIONS
    segment_changed = pyqtSignal(object)  # the selected ReviewSegment, or None
    export_case_requested = pyqtSignal(object)  # a row's ReviewSegment (right-click, More)
    export_requested = pyqtSignal()  # "Export R(t)..." once everything is checked
    undo_requested = pyqtSignal()  # the feedback line's Undo

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ReviewPanel")
        self.setMinimumWidth(260)
        self._all: list[ReviewSegment] = []
        self._visible: list[ReviewSegment] = []
        self._items: dict[int, ReviewItem] = {}  # every listed frame, of any type
        self._images: tuple[str, ...] = ()
        self._fitted = False
        self._hidden: frozenset[str] = frozenset()
        self._shown: int | None = None  # the frame on screen, as last reported
        self._selected: ReviewSegment | None = None  # as last announced
        self._last_row: int | None = None  # where J/K continue when nothing is selected
        self._count = -1
        self._ever: set[int] = set()  # listed since the last reset_progress()
        self._advisor: Advisor | None = None
        self._row_counter: Callable[[ReviewSegment], int] | None = None
        self._fix_label = "Fix with a circle"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(self._build_header())
        layout.addWidget(self._build_list(), 1)
        self.card = ReviewCard()
        self.card.action_requested.connect(self.action_requested)
        self.card.export_case_requested.connect(self._export_selected_case)
        layout.addWidget(self.card)
        self.feedback = FeedbackLine()
        self.feedback.undo_requested.connect(self.undo_requested)
        layout.addWidget(self.feedback)
        self._refresh_view()

    # -- construction ------------------------------------------------------ #

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 0, 0)
        self._summary = QLabel()
        self._summary.setObjectName("reviewSummary")
        self._summary.setStyleSheet("font-weight:600;")
        # Never let a long summary widen the dock (it is clipped instead).
        self._summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self._summary, 1)
        self.kind_filter = KindFilter()
        self.kind_filter.hidden_changed.connect(self.set_hidden_categories)
        row.addWidget(self.kind_filter)
        return row

    def _build_list(self) -> QStackedWidget:
        self._list = QListWidget()
        self._list.setObjectName("ReviewList")
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setItemDelegate(ReviewRowDelegate(self._list))
        self._list.setUniformItemSizes(True)
        self._list.setMouseTracking(True)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.clicked.connect(self._on_clicked)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self.done_page = DonePage()
        self.done_page.export_requested.connect(self.export_requested)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._list)
        self._stack.addWidget(self.done_page)
        return self._stack

    # -- segments ---------------------------------------------------------- #

    def set_segments(self, segments: Iterable[ReviewSegment], images: Sequence[str] = (),
                     *, fitted: bool = True) -> None:
        """Show *segments*; a selected row stays selected while it holds the
        shown frame.  *fitted* = there are results (picks the empty message)."""
        segments, images = list(segments), tuple(images)
        if (segments, images, fitted) == (self._all, self._images, self._fitted):
            return
        if images != self._images:  # another recording: start from the top
            self._last_row = None
            self._ever = set()
        self._all, self._images, self._fitted = segments, images, fitted
        self._items = {item.frame: item for s in segments for item in s.items}
        self._ever |= set(self._items)
        self._rebuild()

    def segments(self) -> list[ReviewSegment]:
        """The rows on show (after the type filter), in queue order."""
        return list(self._visible)

    def frames(self) -> list[int]:
        """Every frame on show, in queue order (row by row)."""
        return queue_frames(self._visible)

    def count(self) -> int:
        """Frames waiting for review, of every type."""
        return sum(s.count for s in self._all)

    def texts(self) -> list[str]:
        return [segment_text(s, self._images) for s in self._visible]

    def add_banner(self, widget: QWidget) -> None:
        """Show *widget* (e.g. a hint) between the summary line and the list."""
        self.layout().insertWidget(1, widget)

    # -- progress, card, feedback ------------------------------------------ #

    def reset_progress(self) -> None:
        """Count progress from the frames listed now (after a fit)."""
        self._ever = set(self._items)
        self._update_header()

    def progress(self) -> tuple[int, int]:
        """(frames still to check, frames listed since the last reset)."""
        return self.count(), len(self._ever)

    def set_advisor(self, advisor: Advisor) -> None:
        """*advisor(frame, item)*: what the card says about the frame on screen."""
        self._advisor = advisor
        self.refresh_card()

    def set_row_counter(self, counter: Callable[[ReviewSegment], int]) -> None:
        """*counter(row)*: how many of its frames Accept all would store."""
        self._row_counter = counter
        self.refresh_card()

    def set_fix_label(self, text: str) -> None:
        """'Fix with a circle' (Auto engine) or 'Fix with edge points'."""
        if text != self._fix_label:
            self._fix_label = text
            self.refresh_card()

    def shown_frame(self) -> int | None:
        """The frame on screen, row or not."""
        return self._shown

    def refresh_card(self) -> None:
        """Re-ask the advisor (e.g. after the frame's result changed)."""
        frame, segment = self._shown, self.current_segment()
        on = frame is not None and bool(self._images) and self._fitted  # after a fit
        self.card.setVisible(on)
        if not on:
            return
        frame = self.current_frame() if segment is not None else frame
        item = self._items.get(frame)  # may be of another type than the row
        advice = (self._advisor(frame, item) if self._advisor is not None
                  else CardAdvice("", "accept"))
        self.card.show_frame(
            frame_title(frame, self._images),
            position_text(segment) if segment is not None else "", advice,
            row_count=segment.count if segment is not None else 0,
            accept_count=self._accept_count(segment), fix_label=self._fix_label,
            why=detail_text(item, frame, self._images, None) if item is not None else "")

    def _accept_count(self, segment: ReviewSegment | None) -> int:
        if segment is None:
            return 0
        return self._row_counter(segment) if self._row_counter is not None else segment.count

    def show_feedback(self, text: str, *, undo: bool = True) -> None:
        self.feedback.show_text(text, undo=undo)

    def show_hint(self, text: str) -> None:
        self.feedback.show_hint(text)

    def clear_feedback(self) -> None:
        self.feedback.clear()

    # -- type filter ------------------------------------------------------- #

    def hidden_categories(self) -> frozenset[str]:
        return self._hidden

    def set_hidden_categories(self, keys: Iterable[str]) -> None:
        hidden = frozenset(keys)
        if hidden != self._hidden:
            self._hidden = hidden
            self._rebuild()

    def _present(self) -> list[tuple[str, int]]:
        """(category, frames) of every type in the queue, in menu order."""
        totals: dict[str, int] = {}
        for s in self._all:
            totals[s.category] = totals.get(s.category, 0) + s.count
        return [(k, totals[k]) for k in _CATEGORY_ORDER if k in totals]

    # -- selection --------------------------------------------------------- #

    def current_segment(self) -> ReviewSegment | None:
        row = self._list.currentRow()
        return self._visible[row] if 0 <= row < len(self._visible) else None

    def current_frame(self) -> int | None:
        """The shown frame, if a row is selected (it then lies in its span)."""
        segment = self.current_segment()
        if segment is None:
            return None
        return self._shown if self._shown is not None and segment.contains(self._shown) \
            else segment.start

    def select_frame(self, frame: int | None, *, keep_row: bool = True) -> bool:
        """The view shows *frame*: silently select the row holding it (or none).

        With *keep_row* the selected row is kept while its span holds the
        frame; without, the row that lists the frame wins (moving on to a
        frame of the queue).  True if a row is selected afterwards.
        """
        self._shown = frame
        prefer = self.current_segment() if keep_row else None
        segment = segment_of(self._visible, frame, prefer=prefer)
        self._set_row(self._visible.index(segment) if segment is not None else -1)
        return segment is not None

    def step(self, delta: int) -> None:
        """Select the next (+1) or previous (-1) row and show its first frame.

        With no row selected, continue from the row last selected (else the top).
        """
        if not self._visible:
            return
        row = self._list.currentRow()
        base = row if row >= 0 else self._last_row
        target = 0 if base is None else min(max(base + delta, 0), len(self._visible) - 1)
        if target != row:
            self._list.setCurrentRow(target)  # emits frame_activated

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._visible):
            self._shown = self._visible[row].start
        self._refresh_view()
        if 0 <= row < len(self._visible):
            self.frame_activated.emit(self._visible[row].start)

    def _on_clicked(self, index: QModelIndex) -> None:
        """Clicking the selected row again goes back to its first frame."""
        segment = self.current_segment()
        if segment is None or index.row() != self._list.currentRow() \
                or self._shown == segment.start:
            return
        self._shown = segment.start
        self._refresh_view()
        self.frame_activated.emit(segment.start)

    def context_menu_at(self, pos: QPoint) -> QMenu | None:
        """The right-click menu of the row at *pos* (list viewport coordinates).

        It acts on that row without selecting it (selecting would move to
        its first frame).
        """
        item = self._list.itemAt(pos)
        row = self._list.row(item) if item is not None else -1
        if not 0 <= row < len(self._visible):
            return None
        segment = self._visible[row]
        menu = QMenu(self._list)
        menu.setToolTipsVisible(True)
        action = menu.addAction("Export as hard case...")
        action.setToolTip("Save these frames, the settings and your fixes as one .zip "
                          "to send to the maintainer")
        action.triggered.connect(lambda: self.export_case_requested.emit(segment))
        return menu

    def _on_context_menu(self, pos: QPoint) -> None:
        menu = self.context_menu_at(pos)
        if menu is not None:
            menu.exec(self._list.viewport().mapToGlobal(pos))
            menu.deleteLater()

    def _export_selected_case(self) -> None:
        segment = self.current_segment()
        if segment is not None:
            self.export_case_requested.emit(segment)

    def _set_row(self, row: int) -> None:
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.blockSignals(False)
        self._refresh_view()

    # -- view -------------------------------------------------------------- #

    def _rebuild(self) -> None:
        """Refill the list; a selected row is followed to its successor."""
        old = self.current_segment()
        self._visible = [s for s in self._all if s.category not in self._hidden]
        self._list.blockSignals(True)
        self._list.clear()
        for segment in self._visible:
            item = QListWidgetItem(segment_text(segment, self._images))
            item.setData(ROW_ROLE, row_data(segment, self._images))
            item.setToolTip(tooltip_html(segment, self._images))
            self._list.addItem(item)
        target = self._successor(old)
        self._list.setCurrentRow(self._visible.index(target) if target is not None else -1)
        self._list.blockSignals(False)
        self._refresh_view()

    def _successor(self, old: ReviewSegment | None) -> ReviewSegment | None:
        """The row to keep after a rebuild: the same kind of problem still
        spanning the shown frame, else whichever row holds that frame."""
        if old is None or self._shown is None:
            return None
        same = next((s for s in self._visible
                     if s.category == old.category and s.contains(self._shown)), None)
        return segment_of(self._visible, self._shown, prefer=same)

    def _refresh_view(self) -> None:
        segment = self.current_segment()
        if segment is not None:
            self._last_row = self._list.currentRow()
        self._update_header()
        self.refresh_card()
        if segment != self._selected:
            self._selected = segment
            self.segment_changed.emit(segment)
        n = self.count()
        if n != self._count:
            self._count = n
            self.count_changed.emit(n)

    def _update_header(self) -> None:
        left, total = self.progress()
        self._summary.setText(progress_text(left, total) if self._fitted or self._all
                              else "Nothing to check yet")
        self.kind_filter.update_kinds(self._present(), self._hidden)
        if self._visible:
            self._stack.setCurrentWidget(self._list)
            return
        if self._all:
            self.done_page.set_message(EMPTY_FILTERED, offer_export=False)
        elif self._ever:
            self.done_page.set_message(EMPTY_ALL_DONE, offer_export=True)
        elif self._fitted:
            self.done_page.set_message(EMPTY_ALL_CLEAR, offer_export=True)
        else:
            self.done_page.set_message(EMPTY_BEFORE_FIT, offer_export=False)
        self._stack.setCurrentWidget(self.done_page)
