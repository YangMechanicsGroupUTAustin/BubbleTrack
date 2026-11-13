"""How one Review row looks: severity dot, title, problem, frame-count badge.

::

    ●  Frames 120–180            [61]
       Lost track

The colours follow the dark theme (style.qss); everything else about a row
(reasons, confidence) lives in its tooltip.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import NamedTuple

from PyQt6.QtCore import QModelIndex, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from bubbletrack.controller.review_segments import ReviewSegment
from bubbletrack.ui.review_text import SEP, file_tag, problem_text, segment_title, tooltip_lines

ROW_ROLE = Qt.ItemDataRole.UserRole + 1  # RowData of a list item

TIER_COLOURS = {"high": "#ef4444", "medium": "#f59e0b", "low": "#a1a1aa"}
TIER_NAMES = {"high": "likely wrong", "medium": "worth a check", "low": "for information"}
_TEXT, _DIM, _FAINT = "#e4e4e7", "#a1a1aa", "#71717a"
_SELECTED_FILL, _SELECTED_EDGE = QColor(99, 102, 241, 46), QColor(129, 140, 248, 110)
_HOVER_FILL, _BADGE_FILL = QColor(255, 255, 255, 10), QColor(255, 255, 255, 20)

ROW_HEIGHT = 44
_PAD_X, _DOT_D, _TEXT_X = 10, 8, 26  # left padding, dot diameter, text indent
_TITLE_PX, _LABEL_PX, _BADGE_PX = 12, 11, 11


class RowData(NamedTuple):
    title: str  # "Frames 120–180" or "Frame 44"
    tag: str  # "_054" for a single frame with a numbered file, else ""
    label: str  # the problem, e.g. "Lost track"
    badge: str  # frame count of a run, "" for one frame
    tier: str  # "high" | "medium" | "low"


def row_data(segment: ReviewSegment, images: Sequence[str]) -> RowData:
    if segment.is_single:  # the file number is drawn fainter than the frame
        return RowData(f"Frame {segment.start + 1}", file_tag(images, segment.start),
                       problem_text(segment), "", segment.tier)
    return RowData(segment_title(segment, images), "", problem_text(segment),
                   str(segment.count), segment.tier)


def tooltip_html(segment: ReviewSegment, images: Sequence[str]) -> str:
    """Rich text (Qt wraps it) with the first line in bold."""
    lines = [html.escape(line) for line in tooltip_lines(segment, images)]
    lines[0] = f"<b>{lines[0]}</b>"
    lines.insert(1, f"<span style='color:{TIER_COLOURS[segment.tier]}'>●</span> "
                    f"{TIER_NAMES[segment.tier]}")
    return "<br>".join(lines)


def _font(base: QFont, px: int, weight: QFont.Weight) -> QFont:
    font = QFont(base)
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


class ReviewRowDelegate(QStyledItemDelegate):
    """Paints rows from their ``RowData`` (no default item drawing)."""

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(option.rect.width(), ROW_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex) -> None:
        data = index.data(ROW_ROLE)
        if not isinstance(data, RowData):
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not option.state & QStyle.StateFlag.State_Enabled:
            painter.setOpacity(0.45)
        card = QRectF(option.rect).adjusted(1, 1, -1, -1)
        self._paint_card(painter, option, card)
        title_font = _font(option.font, _TITLE_PX, QFont.Weight.DemiBold)
        title_h = QFontMetrics(title_font).height()
        top = card.top() + (card.height() - title_h - QFontMetrics(
            _font(option.font, _LABEL_PX, QFont.Weight.Normal)).height()) / 2
        self._paint_dot(painter, card.left() + _PAD_X, top + title_h / 2, data.tier)
        right = card.right() - _PAD_X
        if data.badge:
            right = self._paint_badge(painter, option.font, card, data.badge) - 8
        text_left = card.left() + _TEXT_X
        self._paint_title(painter, title_font, QRectF(text_left, top, right - text_left,
                                                      title_h), data)
        label_font = _font(option.font, _LABEL_PX, QFont.Weight.Normal)
        painter.setFont(label_font)
        painter.setPen(QColor(_DIM))
        label_rect = QRectF(text_left, top + title_h, right - text_left,
                            QFontMetrics(label_font).height())
        painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         QFontMetrics(label_font).elidedText(
                             data.label, Qt.TextElideMode.ElideRight, int(label_rect.width())))
        painter.restore()

    @staticmethod
    def _paint_card(painter: QPainter, option: QStyleOptionViewItem, card: QRectF) -> None:
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(_SELECTED_EDGE)
            painter.setBrush(_SELECTED_FILL)
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_HOVER_FILL)
        else:
            return
        painter.drawRoundedRect(card, 6, 6)

    @staticmethod
    def _paint_dot(painter: QPainter, left: float, centre_y: float, tier: str) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(TIER_COLOURS.get(tier, _FAINT)))
        painter.drawEllipse(QRectF(left, centre_y - _DOT_D / 2, _DOT_D, _DOT_D))

    @staticmethod
    def _paint_title(painter: QPainter, font: QFont, rect: QRectF, data: RowData) -> None:
        metrics = QFontMetrics(font)
        painter.setFont(font)
        painter.setPen(QColor(_TEXT))
        align = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title = metrics.elidedText(data.title, Qt.TextElideMode.ElideRight, int(rect.width()))
        painter.drawText(rect, align, title)
        if not data.tag or title != data.title:
            return
        tag_font = QFont(font)
        tag_font.setWeight(QFont.Weight.Normal)
        painter.setFont(tag_font)
        painter.setPen(QColor(_FAINT))
        rest = rect.adjusted(metrics.horizontalAdvance(title), 0, 0, 0)
        painter.drawText(rest, align, QFontMetrics(tag_font).elidedText(
            SEP + data.tag, Qt.TextElideMode.ElideRight, int(rest.width())))

    @staticmethod
    def _paint_badge(painter: QPainter, base: QFont, card: QRectF, text: str) -> float:
        """Draw the count pill at the right edge; returns its left x."""
        font = _font(base, _BADGE_PX, QFont.Weight.DemiBold)
        width = QFontMetrics(font).horizontalAdvance(text) + 14
        height = 18.0
        pill = QRectF(card.right() - _PAD_X - width, card.center().y() - height / 2,
                      width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_BADGE_FILL)
        painter.drawRoundedRect(pill, height / 2, height / 2)
        painter.setFont(font)
        painter.setPen(QColor(_DIM))
        painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter), text)
        return pill.left()
