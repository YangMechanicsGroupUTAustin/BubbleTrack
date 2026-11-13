"""The colour key in the top-left corner of an image panel: one line per kind
of overlay drawn, fixed to the viewport, hidden when nothing is drawn."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from bubbletrack.controller.display_mixin import draw_fit_overlay
from bubbletrack.controller.suggestion_view import GUESS_COLOUR, GUESS_GAP_COLOUR
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import AppState, update_state
from bubbletrack.ui import overlay_legend as ol
from bubbletrack.ui.image_panel import ImagePanel


def _circle(n: int, r: float = 30.0, c=(60.0, 80.0)) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.sin(t), c[1] + r * np.cos(t)])


@pytest.fixture
def panel(qapp):
    p = ImagePanel("Test")
    p.resize(520, 380)
    p.show()
    qapp.processEvents()
    yield p
    p.close()


def _legend(panel: ImagePanel) -> ol.OverlayLegend:
    return panel._legend


def _shown(panel: ImagePanel) -> bool:
    return _legend(panel).isVisibleTo(panel)


# -- palette and entries ------------------------------------------------------------

class TestEntries:
    def test_known_colours_name_what_they_are(self):
        assert ol.default_text(ol.OUTLINE_COLOUR, "line") == "outline (measured)"
        # The plain circle is named for what is true in every mode.
        assert ol.default_text(ol.CIRCLE_COLOUR, "line") == "circle (radius R)"
        assert ol.default_text(ol.GUESS_COLOUR, "dash") == "Auto's guess (not accepted)"
        assert ol.default_text(ol.ROI_COLOUR, "box") == "ROI"
        assert ol.default_text("#123456", "line") is None
        assert ol.default_text(ol.OUTLINE_COLOUR.upper(), "line") == "outline (measured)"

    def test_the_guess_is_yellow_not_white(self):
        yellow = QColor(GUESS_COLOUR)
        assert yellow.red() > 200 and yellow.green() > 180 and yellow.blue() < 120
        assert QColor(GUESS_GAP_COLOUR).lightness() < 60  # the dark line under the dashes

    def test_your_circle_does_not_look_like_auto_s_guess(self):
        from bubbletrack.ui.circle_drag import PREVIEW_COLOUR

        mine, guess = QColor(ol.USER_CIRCLE_COLOUR), QColor(GUESS_COLOUR)
        apart = abs(mine.hue() - guess.hue())
        assert min(apart, 360 - apart) >= 120  # cyan against yellow, not amber (~10 degrees)
        for other in (ol.OUTLINE_COLOUR, ol.CIRCLE_COLOUR):  # the circles it may sit on
            hue = abs(mine.hue() - QColor(other).hue())
            assert min(hue, 360 - hue) >= 40, other
        assert QColor(PREVIEW_COLOUR) == mine  # the drag preview: the same colour
        assert ol.default_text(ol.USER_CIRCLE_COLOUR, "dash") == "your circle (not stored)"

    def test_make_entry_default_explicit_or_none(self):
        assert ol.make_entry(ol.OUTLINE_COLOUR, "line", None).text == "outline (measured)"
        assert ol.make_entry(ol.OUTLINE_COLOUR, "line", "") is None
        assert ol.make_entry(ol.CIRCLE_COLOUR, "line", "fitted circle").text == "fitted circle"
        assert ol.make_entry(GUESS_GAP_COLOUR, "line", None) is None  # the line under the guess
        guess = ol.make_entry(ol.GUESS_COLOUR, "dash", None)
        assert guess.under == ol.GUESS_UNDER_COLOUR and guess.kind == "dash"
        with pytest.raises(ValueError):
            ol.LegendEntry("x", "#ffffff", "wiggle")


# -- the key on a panel -------------------------------------------------------------

class TestLegendOnPanel:
    def test_hidden_until_something_is_drawn(self, panel):
        panel.set_image(np.zeros((150, 200)))
        assert not _shown(panel) and panel.legend_texts() == []

    def test_one_line_per_overlay_kind_in_a_fixed_order(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_roi_rect((1, 150), (1, 200))
        panel.draw_fit_outline(_circle(120))
        panel.draw_circle(60, 80, 30)
        panel.draw_circle(60, 80, 31)  # same kind again: no second line
        assert panel.legend_texts() == ["outline (measured)", "circle (radius R)", "ROI"]
        assert _shown(panel)

    def test_edge_points_and_explicit_text(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_points(_circle(10), ol.OUTLINE_COLOUR, 4.0)
        panel.draw_circle(60, 80, 30, legend="fitted circle")
        panel.draw_contour(_circle(50), "#ffffff", legend="")
        assert panel.legend_texts() == ["edge points", "fitted circle"]

    def test_image_keys_can_be_added(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.add_legend_entry(ol.LegendEntry("mask", "#22c55e", "fill"))
        assert panel.legend_texts() == ["mask"]

    def test_a_new_frame_clears_the_key(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_circle(60, 80, 30)
        panel.set_image(np.zeros((150, 200)))
        assert panel.legend_texts() == [] and not _shown(panel)
        panel.draw_circle(60, 80, 30)
        panel.clear_overlays()
        assert not _shown(panel)

    def test_sits_in_the_top_left_corner_and_stays_small(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_roi_rect((1, 150), (1, 200))
        panel.draw_fit_outline(_circle(120))
        panel.draw_circle(60, 80, 30)
        geo = _legend(panel).geometry()
        port = panel._view.viewport().geometry()
        assert geo.left() - port.left() <= 10 and geo.top() - port.top() <= 10
        assert (geo.width(), geo.height()) == _legend(panel).content_size()
        # Never more than a corner (text widths depend on the fonts installed).
        assert geo.width() <= ol.MAX_WIDTH_SHARE * port.width()
        assert geo.height() <= ol.MAX_HEIGHT_SHARE * port.height()
        assert geo.height() <= 3 * (_legend(panel)._row_height() + 2) + 2 * ol.PAD_Y

    def test_does_not_move_or_scale_with_zoom_and_pan(self, panel, qapp):
        panel.set_image(np.zeros((300, 400)))
        panel.draw_fit_outline(_circle(120))
        before = _legend(panel).geometry()
        panel.zoom_by(4.0)
        panel._view.horizontalScrollBar().setValue(200)
        panel._view.verticalScrollBar().setValue(150)
        qapp.processEvents()
        assert _legend(panel).geometry() == before

    def test_follows_a_resize_and_hides_when_the_panel_is_tiny(self, panel, qapp):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_fit_outline(_circle(120))
        panel.resize(900, 600)
        qapp.processEvents()
        port = panel._view.viewport().geometry()
        assert _legend(panel).geometry().left() - port.left() <= 10 and _shown(panel)
        panel.resize(200, 150)
        qapp.processEvents()
        assert not _shown(panel)  # would cover too much of the image
        panel.resize(520, 380)
        qapp.processEvents()
        assert _shown(panel)

    def test_clicks_go_through_to_the_image(self, panel):
        assert _legend(panel).testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def test_is_painted_dark_over_a_bright_image(self, panel, qapp):
        panel.set_image(np.ones((150, 200)))
        panel.draw_fit_outline(_circle(120))
        qapp.processEvents()
        geo = _legend(panel).geometry()
        shot = panel._view.grab().toImage()
        inside = shot.pixelColor(geo.right() - 4, geo.top() + 3)
        outside = shot.pixelColor(geo.right() + 20, geo.top() + 3)
        assert inside.lightness() < 120 < outside.lightness()


# -- what the fit overlay puts in the key -------------------------------------------

def _fitted(mode: str = "edge", source: FitSource = FitSource.AUTO, flags=frozenset()) -> AppState:
    state = update_state(AppState.create_empty(), detection_mode=mode).with_results_initialized(3)
    state.radius[1] = 30.0
    state.circle_fit_par[1] = [60.0, 80.0]
    state.circle_xy[1] = _circle(120)
    state.fit_source[1] = source
    state.fit_flags[1] = frozenset(flags)
    return state


@pytest.mark.parametrize("mode, source, flags, circle", [
    ("edge", FitSource.AUTO, (), "fitted circle"),  # Auto's default: the circle through the outline
    ("edge", FitSource.AUTO, ("area_radius",), "equal-area circle"),  # "Bubble may be non-round"
    ("edge", FitSource.MANUAL, (), "fitted circle"),  # a circle through clicked points
    ("intensity", FitSource.AUTO, (), "fitted circle"),  # classic modes fit a circle
])
def test_fit_overlay_names_its_circle(panel, mode, source, flags, circle):
    panel.set_image(np.zeros((150, 200)))
    draw_fit_overlay(_fitted(mode, source, flags), panel, 1)
    assert panel.legend_texts() == ["outline (measured)", circle]
