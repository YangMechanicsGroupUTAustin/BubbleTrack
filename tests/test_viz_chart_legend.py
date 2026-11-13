"""R(t) chart legend row: one name and one 0-1 scale for the colours; keys never clip."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QLabel

from bubbletrack.ui.chart_legend import KeyRow, keys_that_fit
from bubbletrack.ui.radius_chart import RadiusChart


class TestKeysThatFit:
    def test_everything_fits(self):
        assert keys_that_fit([50, 60, 70], [0, 0, 0], room=300, spacing=10) == [True] * 3

    def test_the_least_needed_keys_go_first(self):
        # drop order: the highest number goes first; 0 = always shown
        assert keys_that_fit([50, 60, 70], [0, 2, 1], room=150, spacing=10) == [True, False, True]
        assert keys_that_fit([50, 60, 70], [0, 2, 1], room=100, spacing=10) == [True, False, False]
        assert keys_that_fit([50, 60, 70], [0, 2, 1], room=10, spacing=10) == [True, False, False]


class TestKeyRow:
    def test_hides_keys_whole_and_names_them_in_its_tooltip(self, qapp):
        row = KeyRow()
        must, extra = QLabel("must"), QLabel("extra key")
        row.add_key(must, 0, "must")
        row.add_key(extra, 1, "extra key")
        row.show()
        both = must.sizeHint().width() + extra.sizeHint().width() + row.spacing()
        row.resize(both + 4, 20)
        qapp.processEvents()
        assert must.isVisible() and extra.isVisible() and row.toolTip() == ""
        row.resize(must.sizeHint().width() + 4, 20)
        qapp.processEvents()
        assert must.isVisible() and not extra.isVisible()
        assert "extra key" in row.toolTip()
        row.close()


@pytest.fixture
def chart(qapp):
    c = RadiusChart()
    c.resize(800, 300)
    c.show()
    qapp.processEvents()
    yield c
    c.close()


class TestScoreWording:
    def test_confidence_from_0_to_1_by_default(self, chart):
        assert chart._score_name.text() == "confidence 0"
        assert chart._score_high.text() == "1"
        assert "confidence" in chart._score_name.toolTip()

    def test_hover_uses_the_ramp_name_and_scale(self, chart):
        chart.plot_all(np.arange(1, 4), np.array([10.0, 20.0, 30.0]), np.array([0.2, 0.5, 0.9]))
        assert "confidence 0.50" in chart._hover_text(1)
        chart.set_score_name("fit quality")
        assert chart._score_name.text() == "fit quality 0"
        assert "fit quality 0.50" in chart._hover_text(1)
        assert "Q " not in chart._hover_text(1) and "%" not in chart._hover_text(1)


class TestLegendNeverClips:
    @pytest.mark.parametrize("width", [480, 640, 800, 1200, 2400])
    def test_visible_keys_are_whole(self, qapp, width):
        c = RadiusChart()
        c.resize(width, 300)
        c.show()
        qapp.processEvents()
        for key in c._legend_keys.keys():
            if key.isVisible():
                assert key.width() >= key.sizeHint().width(), (width, key.toolTip())
        c.close()

    def test_markers_go_first_the_colour_scale_last(self, qapp):
        c = RadiusChart()
        c.resize(2400, 300)
        c.show()
        qapp.processEvents()
        assert c._legend_keys.hidden_texts() == []
        order = ["● computed", "◆ manual / locked", "colour scale"]  # no collapse key
        for width in (900, 700, 500, 300):  # how much fits depends on the font
            c.resize(width, 300)
            qapp.processEvents()
            hidden = c._legend_keys.hidden_texts()
            assert hidden == order[:len(hidden)], width
            assert c._score_name.isVisible() == ("colour scale" not in hidden)
        assert all(text in c._legend_keys.toolTip() for text in hidden)
        c.close()
