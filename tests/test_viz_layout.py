"""Window layout: splitter, small screens, pinned Fit / Export, styling."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QPushButton, QSplitter, QWidget

import bubbletrack
from bubbletrack.ui import main_window as mw

STYLE = (Path(bubbletrack.__file__).parent / "resources" / "style.qss").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def temp_settings(tmp_path, monkeypatch):
    """Keep window-layout settings out of the user's registry / plist."""
    path = str(tmp_path / "layout.ini")
    monkeypatch.setattr(mw, "layout_settings",
                        lambda: QSettings(path, QSettings.Format.IniFormat))
    return path


@pytest.fixture
def styled(qapp):
    qapp.setStyleSheet(STYLE)
    yield qapp
    qapp.setStyleSheet("")


class TestSplitter:
    def test_images_and_chart_share_a_vertical_splitter(self, gui):
        window, _ = gui
        splitter = window.content_splitter
        assert isinstance(splitter, QSplitter) and splitter.count() == 2
        assert splitter.widget(0).isAncestorOf(window.original_panel)
        assert splitter.widget(0).isAncestorOf(window.frame_scrubber)
        assert splitter.widget(1).isAncestorOf(window.radius_chart)

    def test_sizes_are_remembered(self, gui, temp_settings, qapp):
        window, _ = gui
        window.resize(1400, 900)
        window.show()
        qapp.processEvents()
        window.content_splitter.setSizes([300, 400])
        window.save_layout()
        stored = window.content_splitter.sizes()

        window.content_splitter.setSizes([500, 200])
        assert window.restore_layout()
        assert window.content_splitter.sizes() == stored

    def test_bad_or_missing_settings_fall_back_to_defaults(self, gui, temp_settings):
        window, _ = gui
        assert not window.restore_layout()
        QSettings(temp_settings, QSettings.Format.IniFormat).setValue(
            mw.SPLITTER_SETTINGS_KEY, ["x", "y"])
        assert not window.restore_layout()

    def test_first_start_gives_the_chart_about_45_percent(self, gui, qapp):
        window, _ = gui
        window.resize(1440, 900)
        window.show()
        qapp.processEvents()
        images, chart_pane = window.content_splitter.sizes()
        strip = window.filmstrip.height() if window.filmstrip is not None else 0
        share = (chart_pane - strip) / (images + chart_pane - strip)
        assert 0.40 <= share <= 0.50, (images, chart_pane, strip)

    def test_saved_sizes_win_over_the_default(self, gui, temp_settings, qapp):
        QSettings(temp_settings, QSettings.Format.IniFormat).setValue(
            mw.SPLITTER_SETTINGS_KEY, [500, 200])
        window, _ = gui
        # Tall enough that the chart pane's minimum (chart 260 px + key-frame strip)
        # does not override the saved split, as it does on a 900 px screen.
        window.resize(1440, 1100)
        window.show()
        qapp.processEvents()
        images, chart_pane = window.content_splitter.sizes()
        assert images / (images + chart_pane) > 0.6  # the default gives images < 55 %

    def test_user_drag_saves_after_a_pause(self, gui, temp_settings, qapp):
        window, _ = gui
        window.content_splitter.splitterMoved.emit(100, 1)
        assert window._splitter_save_timer.isActive()
        window._splitter_save_timer.timeout.emit()
        assert QSettings(temp_settings, QSettings.Format.IniFormat).value(
            mw.SPLITTER_SETTINGS_KEY) is not None


class TestSmallScreens:
    def test_minimum_size_fits_1280x800(self, gui):
        window, _ = gui
        assert window.minimumWidth() <= 1280 and window.minimumHeight() <= 800
        hint = window.minimumSizeHint()
        assert hint.width() <= 1280 and hint.height() <= 800

    @pytest.mark.parametrize("size", [(1280, 800), (1440, 900), (1512, 982)])
    def test_window_takes_laptop_sizes(self, gui, qapp, size):
        window, _ = gui
        window.show()
        window.resize(*size)
        qapp.processEvents()
        assert (window.width(), window.height()) == size
        assert window.radius_chart.height() >= window.radius_chart.minimumHeight()
        assert window.original_panel.height() > 120

    def test_left_panel_scrolls_and_fit_and_export_stay_pinned(self, gui, qapp):
        window, _ = gui
        window.show()
        window.resize(1280, 800)
        qapp.processEvents()
        lp = window.left_panel
        fit, export = lp.fit_controls.fit_button, lp.export_button
        for button in (fit, export):
            assert not lp.scroll_area.widget().isAncestorOf(button)
            assert lp.pinned_footer.isAncestorOf(button)
        # Content height depends on the fonts available; force it to be tall.
        lp.scroll_area.widget().setMinimumHeight(2000)
        qapp.processEvents()
        scroll = lp.scroll_area.verticalScrollBar()
        assert scroll.maximum() > 0  # content is taller than the panel
        scroll.setValue(scroll.maximum())
        qapp.processEvents()
        assert fit.isVisible() and export.isVisible()


class TestSidePanel:
    def test_no_tabs_source_then_settings_then_footer(self, gui):
        window, _ = gui
        lp = window.left_panel
        content = lp.scroll_area.widget().layout()
        order = [content.itemAt(i).widget() for i in range(content.count())]
        assert order.index(lp.image_source) < order.index(lp.measure_panel)
        assert not hasattr(lp, "automatic_tab") and not hasattr(lp, "post_processing")

    def test_status_bar_shows_the_mode(self, gui):
        window, _ = gui
        lp = window.left_panel
        assert window.status_bar._mode_label.text() == "Mode: Measure"
        lp.tab_bar.select("Manual")
        assert window.status_bar._mode_label.text() == "Mode: Manual"
        lp.tab_bar.select("Pre-tune")
        assert window.status_bar._mode_label.text() == "Mode: Measure"

    def test_footer_follows_the_run_lock(self, gui):
        window, controller = gui
        lp = window.left_panel
        controller._apply_run_lock(True)
        assert not lp.export_button.isEnabled()
        controller._apply_run_lock(False)
        assert lp.export_button.isEnabled() and lp.fit_controls.fit_button.isEnabled()


def _centre_colour(widget) -> QColor:
    """Colour near the top centre of *widget* as drawn inside its window."""
    host = widget.window()
    host.show()
    QApplication.processEvents()
    image = host.grab().toImage()
    origin = widget.mapTo(host, widget.rect().topLeft())
    return image.pixelColor(origin.x() + widget.width() // 2, origin.y() + 4)


class TestStyling:
    def test_disabled_stop_button_is_not_solid_red(self, styled):
        host = QWidget()
        btn = QPushButton("Stop", host)
        btn.setObjectName("dangerBtn")
        btn.setGeometry(10, 10, 100, 34)
        host.resize(130, 60)
        enabled = _centre_colour(btn)
        btn.setEnabled(False)
        disabled = _centre_colour(btn)
        host.close()
        assert enabled.red() > 200 and enabled.green() < 100  # still red when active
        assert not (disabled.red() > 150 and disabled.green() < 100)

    def test_idle_app_shows_no_stop_button(self, gui):
        window, _ = gui
        stop = window.left_panel.fit_controls.stop_button
        assert not stop.isVisibleTo(window.left_panel) and not stop.isEnabled()

    def test_fit_is_the_primary_button(self, gui, styled):
        window, _ = gui
        fit = window.left_panel.fit_controls.fit_button
        assert fit.objectName() == "primaryBtn"
        colour = _centre_colour(fit)
        window.close()
        assert colour.blue() > 200  # indigo fill

    def test_active_view_mode_is_filled(self, gui, styled):
        window, controller = gui
        from bubbletrack.ui.image_compare import CompareMode
        controller.settings.set("show_all_view_modes", True)
        buttons = window.compare_toolbar._buttons
        for btn in buttons.values():
            btn.resize(110, 28)
        active = _centre_colour(buttons[CompareMode.SIDE_BY_SIDE])
        idle = _centre_colour(buttons[CompareMode.WIPE])
        assert active.blue() > 200  # indigo fill
        assert idle.blue() < 80
