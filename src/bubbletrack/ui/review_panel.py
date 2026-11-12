"""Review panel: runs of suspicious frames (segments) and one-key actions on them.

::

    12 segments · 340 frames      [All types ▾]
    ●  Frames 120–180               [61]
       Lost track
    ●  Frame 44 · _054
       Sudden jump
    Frame 121 · _131 — lost track, low confidence 0.21
    [Accept (A)] [Skip (S)] ...
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PyQt6.QtCore import QModelIndex, QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMenu, QPushButton, QSizePolicy, QStackedWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from bubbletrack.controller.review_queue import ReviewItem
from bubbletrack.controller.review_segments import (
    CATEGORIES, OTHER, ReviewSegment, category_label, queue_frames, segment_of,
)
from bubbletrack.ui.review_rows import ROW_ROLE, ReviewRowDelegate, row_data, tooltip_html
from bubbletrack.ui.review_text import detail_text, segment_text, summary_text

EMPTY_BEFORE_FIT = "Doubtful frames show up here after a fit."
EMPTY_ALL_CLEAR = "All clear — nothing needs a second look."
EMPTY_FILTERED = "Nothing of the selected types."
_CATEGORY_ORDER = [c.key for c in CATEGORIES] + [OTHER.key]


class ReviewPanel(QWidget):
    """Segments to review with Accept / Refit / Manual / Delete / Skip.

    The panel only shows segments and reports requests; ReviewController
    decides what they do.  A segment is selected only while the shown frame
    lies in its span (its range is highlighted on the chart), so actions
    always hit the run the user is looking at.
    """

    ACTIONS = ("accept", "refit", "refit_neighbors", "manual", "delete", "skip")
    _BUTTONS = (  # (action, label, tooltip), laid out in pairs
        ("accept", "Accept (A)", "Keep the results of the whole row and lock them against Fit All"),
        ("skip", "Skip (S)", "Leave the row as it is and drop it from the queue"),
        ("refit", "Refit (R)", "Run detection again on every frame of the row"),
        ("refit_neighbors", "Neighbours (N)",
         "Refit the shown frame using neighbouring fits as a prior"),
        ("manual", "Manual (M)", "Click edge points on the shown frame in the Manual tab"),
        ("delete", "Delete (D)", "Remove the results of every frame of the row"),
    )

    count_changed = pyqtSignal(int)  # frames waiting for review (every type)
    frame_activated = pyqtSignal(int)  # the user picked a row: show this frame
    action_requested = pyqtSignal(str)  # one of ACTIONS
    segment_changed = pyqtSignal(object)  # the selected ReviewSegment, or None
    export_case_requested = pyqtSignal(object)  # a row's ReviewSegment (its right-click menu)

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(self._build_header())
        layout.addWidget(self._build_list(), 1)
        self._detail = QLabel()
        self._detail.setObjectName("reviewDetail")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color:#a1a1aa; font-size:11px; padding:0 2px;")
        layout.addWidget(self._detail)
        layout.addLayout(self._build_buttons())
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
        self._filter_btn = QToolButton()
        self._filter_btn.setObjectName("reviewFilter")
        self._filter_btn.setToolTip("Show only some types of problem")
        self._filter_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._filter_menu = QMenu(self._filter_btn)
        self._filter_menu.aboutToShow.connect(self._fill_filter_menu)
        self._filter_btn.setMenu(self._filter_menu)
        row.addWidget(self._filter_btn)
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
        self._empty = QLabel()
        self._empty.setObjectName("reviewEmpty")
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color:#71717a; padding:16px;")
        self._stack = QStackedWidget()
        self._stack.addWidget(self._list)
        self._stack.addWidget(self._empty)
        return self._stack

    def _build_buttons(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(4)
        self._action_buttons: dict[str, QPushButton] = {}
        for i, (action, label, tip) in enumerate(self._BUTTONS):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            if action != "accept":  # one primary action; the rest stay quiet
                btn.setObjectName("secondaryBtn")
            btn.clicked.connect(lambda _checked, a=action: self.action_requested.emit(a))
            grid.addWidget(btn, i // 2, i % 2)
            self._action_buttons[action] = btn
        self._prev_btn = QPushButton("▲ Prev (K)")
        self._next_btn = QPushButton("Next (J) ▼")
        for btn, delta in ((self._prev_btn, -1), (self._next_btn, 1)):
            btn.setObjectName("secondaryBtn")
            btn.setToolTip("Previous / next row (J/K); ←/→ step through a row's frames")
            btn.clicked.connect(lambda _checked, d=delta: self.step(d))
        nav_row = len(self._BUTTONS) // 2
        grid.addWidget(self._prev_btn, nav_row, 0)
        grid.addWidget(self._next_btn, nav_row, 1)
        return grid

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
        self._all, self._images, self._fitted = segments, images, fitted
        self._items = {item.frame: item for s in segments for item in s.items}
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

    def _fill_filter_menu(self) -> None:
        menu = self._filter_menu
        menu.clear()
        menu.addAction("Show all types", lambda: self.set_hidden_categories(()))
        menu.addSeparator()
        for key, frames in self._present():
            action = menu.addAction(f"{category_label(key)}  ({frames})")
            action.setCheckable(True)
            action.setChecked(key not in self._hidden)
            action.toggled.connect(lambda on, k=key: self.set_hidden_categories(
                self._hidden - {k} if on else self._hidden | {k}))

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
        self._update_controls(segment)
        self._update_header()
        frame = self.current_frame()
        shown = segment is not None and frame is not None
        item = self._items.get(frame) if shown else None  # may be of another type
        self._detail.setVisible(shown)
        self._detail.setText(detail_text(item, frame, self._images) if shown else "")
        self._detail.setToolTip(detail_text(item, frame, self._images, None) if shown else "")
        if segment != self._selected:
            self._selected = segment
            self.segment_changed.emit(segment)
        n = self.count()
        if n != self._count:
            self._count = n
            self.count_changed.emit(n)

    def _update_controls(self, segment: ReviewSegment | None) -> None:
        for btn in self._action_buttons.values():
            btn.setEnabled(segment is not None)
        self._prev_btn.setEnabled(bool(self._visible))
        self._next_btn.setEnabled(bool(self._visible))

    def _update_header(self) -> None:
        self._summary.setText(summary_text(len(self._visible), len(self.frames()))
                              or ("Every row hidden" if self._all else "Nothing to review"))
        present = self._present()
        shown = sum(1 for key, _ in present if key not in self._hidden)
        self._filter_btn.setVisible(len(present) > 1 or bool(self._hidden))
        self._filter_btn.setText("All types ▾" if shown == len(present)
                                 else f"{shown} of {len(present)} types ▾")
        if self._visible:
            self._stack.setCurrentWidget(self._list)
            return
        self._empty.setText(EMPTY_FILTERED if self._all
                            else EMPTY_ALL_CLEAR if self._fitted else EMPTY_BEFORE_FIT)
        self._stack.setCurrentWidget(self._empty)
