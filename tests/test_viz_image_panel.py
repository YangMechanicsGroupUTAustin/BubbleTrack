"""Image panel: raw image + fitted outline, zoom kept across frames, view modes."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem

from bubbletrack.controller.display_mixin import draw_fit_overlay
from bubbletrack.model.state import update_state
from bubbletrack.ui.image_compare import CompareMode
from bubbletrack.ui.image_panel import ImagePanel, is_ordered_contour


def _circle(n: int, r: float = 30.0, c=(60.0, 80.0)) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.sin(t), c[1] + r * np.cos(t)])


def _raster(points: np.ndarray) -> np.ndarray:
    """The same pixels in row-major order, as np.nonzero would return them."""
    pts = np.round(points)
    return pts[np.lexsort((pts[:, 1], pts[:, 0]))]


class TestOrderedContour:
    def test_traced_outline_is_a_contour(self):
        assert is_ordered_contour(_circle(200))

    def test_sparse_ray_points_in_angle_order_are_a_contour(self):
        assert is_ordered_contour(_circle(24))

    def test_raster_ordered_edge_pixels_are_not(self):
        assert not is_ordered_contour(_raster(_circle(400)))

    def test_too_few_or_missing_points(self):
        assert not is_ordered_contour(None)
        assert not is_ordered_contour(_circle(5))
        assert not is_ordered_contour(np.zeros((20, 2)))  # degenerate: no hull


@pytest.fixture
def panel(qapp):
    p = ImagePanel("Test")
    p.resize(400, 300)
    p.show()
    qapp.processEvents()
    yield p
    p.close()


def _scale(panel: ImagePanel) -> float:
    return panel._view.transform().m11()


class TestZoomKeptAcrossFrames:
    def test_same_size_frame_keeps_zoom_and_pan(self, panel, qapp):
        panel.set_image(np.zeros((100, 200)))
        panel.zoom_in()
        panel.zoom_in()
        zoomed = _scale(panel)
        panel._view.horizontalScrollBar().setValue(40)
        pan = panel._view.horizontalScrollBar().value()

        panel.set_image(np.ones((100, 200)) * 0.5)  # next frame, same size
        qapp.processEvents()
        assert _scale(panel) == pytest.approx(zoomed)
        assert panel._view.horizontalScrollBar().value() == pan
        assert not panel.auto_fit

    def test_new_image_size_refits(self, panel):
        panel.set_image(np.zeros((100, 200)))
        panel.zoom_in()
        panel.set_image(np.zeros((300, 300)))
        assert panel.auto_fit
        rect = panel._view.mapFromScene(panel._scene.sceneRect()).boundingRect()
        assert rect.width() <= panel._view.viewport().width() + 2

    def test_reset_zoom_fits_and_follows_resizes(self, panel, qapp):
        panel.set_image(np.zeros((100, 200)))
        fitted = _scale(panel)
        panel.zoom_in()
        panel.reset_zoom()
        assert _scale(panel) == pytest.approx(fitted)
        assert panel.auto_fit
        panel.resize(800, 600)
        qapp.processEvents()
        assert _scale(panel) > fitted  # still fitting the bigger panel

    def test_manual_zoom_survives_resize(self, panel, qapp):
        panel.set_image(np.zeros((100, 200)))
        panel.zoom_by(3.0)
        zoomed = _scale(panel)
        panel.resize(700, 500)
        qapp.processEvents()
        assert _scale(panel) == pytest.approx(zoomed)

    def test_legacy_zoom_names_still_work(self, panel):
        panel.set_image(np.zeros((50, 50)))
        before = _scale(panel)
        panel._zoom_in()
        assert _scale(panel) > before
        panel._zoom_reset()
        assert _scale(panel) == pytest.approx(before)


class TestOverlayDrawing:
    def test_contour_is_one_closed_polyline(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_fit_outline(_circle(300))
        items = panel._overlay_items
        assert len(items) == 1 and isinstance(items[0], QGraphicsPathItem)
        path = items[0].path()
        assert path.elementCount() == 301  # moveTo + 299 lineTo + closing lineTo
        assert items[0].pen().isCosmetic()

    def test_unordered_points_are_one_item_of_dots(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_fit_outline(_raster(_circle(400)))
        assert len(panel._overlay_items) == 1
        assert panel._overlay_items[0].brush().style() != 0  # filled dots

    def test_new_frame_clears_overlays(self, panel):
        panel.set_image(np.zeros((150, 200)))
        panel.draw_circle(60, 80, 30)
        panel.set_image(np.zeros((150, 200)))
        assert panel._overlay_items == []


class TestFitOverlayOnRawImage:
    def _state(self, contour):
        from bubbletrack.model.state import AppState
        state = AppState.create_empty().with_results_initialized(3)
        state.radius[1] = 30.0
        state.circle_fit_par[1] = [60.0, 80.0]
        state.circle_xy[1] = contour
        return state

    def test_circle_and_contour_drawn(self, panel):
        panel.set_image(np.zeros((150, 200)))
        draw_fit_overlay(self._state(_circle(120)), panel, 1)
        kinds = [type(i) for i in panel._overlay_items]
        assert QGraphicsEllipseItem in kinds and QGraphicsPathItem in kinds

    def test_unfitted_frame_draws_nothing(self, panel):
        panel.set_image(np.zeros((150, 200)))
        draw_fit_overlay(self._state(_circle(120)), panel, 0)
        assert panel._overlay_items == []

    def test_area_result_has_no_circle(self, panel):
        state = self._state(_circle(120))
        state.region_count[1] = 2
        panel.set_image(np.zeros((150, 200)))
        draw_fit_overlay(state, panel, 1)
        assert [type(i) for i in panel._overlay_items] == [QGraphicsPathItem]


class TestTitle:
    LONG = "Auto view: frame ÷ reference, a title far too long for its panel"

    def test_a_long_title_ends_in_an_ellipsis_and_keeps_its_text(self, qapp):
        panel = ImagePanel(self.LONG)
        panel.resize(260, 200)
        panel.show()
        qapp.processEvents()
        label = panel._title_label
        assert label.text() == self.LONG
        assert label.width() < label.sizeHint().width()
        assert label.shown_text().endswith("…") and len(label.shown_text()) > 3
        panel.resize(1600, 200)
        qapp.processEvents()
        assert label.shown_text() == self.LONG
        panel.close()

    def test_the_title_does_not_widen_the_panel(self, qapp):
        long, short = ImagePanel(self.LONG), ImagePanel("Image…")
        assert long.minimumSizeHint().width() <= short.minimumSizeHint().width()
        assert long.sizeHint() == ImagePanel("Image").sizeHint()  # side by side: even split

    def test_title_tooltip(self, qapp):
        panel = ImagePanel()
        panel.set_title("Auto view", "frame ÷ reference")
        assert panel._title_label.toolTip() == "frame ÷ reference"
        panel.set_title("Binary Mask")
        assert panel._title_label.toolTip() == ""

    def test_auto_view_title_is_short_and_explained(self):
        from bubbletrack.controller.display_mixin import ENGINE_VIEW_TIP, ENGINE_VIEW_TITLE
        assert ENGINE_VIEW_TITLE == "Auto view"
        assert "reference" in ENGINE_VIEW_TIP


class TestViewModesInWindow:
    def test_image_and_mask_is_default(self, gui):
        window, _ = gui
        assert window.compare_mode is CompareMode.SIDE_BY_SIDE
        assert not window.binary_panel.isHidden()
        assert window.compare_toolbar._buttons[CompareMode.SIDE_BY_SIDE].isChecked()

    def test_image_only_hides_the_mask_and_marks_its_button(self, gui):
        window, _ = gui
        toolbar = window.compare_toolbar
        toolbar._buttons[CompareMode.IMAGE].click()
        assert window.compare_mode is CompareMode.IMAGE
        assert window.binary_panel.isHidden()
        checked = [m for m, b in toolbar._buttons.items() if b.isChecked()]
        assert checked == [CompareMode.IMAGE]
        toolbar._buttons[CompareMode.OVERLAY].click()
        assert window.binary_panel.isHidden()

    def test_fitted_frame_shows_raw_image_with_outline(self, gui, make_sequence):
        window, controller = gui
        controller.file_ctrl.on_folder_selected(str(make_sequence("seq", 4)))
        state = controller._get_state()
        state.radius[2] = 30.0
        state.circle_fit_par[2] = [125.0, 200.0]
        state.circle_xy[2] = _circle(90, 30.0, (125.0, 200.0))
        controller._state = update_state(state)
        controller.file_ctrl.on_frame_changed(2)
        kinds = {type(i).__name__ for i in window.original_panel._overlay_items}
        assert {"QGraphicsEllipseItem", "QGraphicsPathItem"} <= kinds
        assert window.radius_chart.current_frame() == 2
