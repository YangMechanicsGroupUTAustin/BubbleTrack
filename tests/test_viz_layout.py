"""Window layout: splitter, small screens, tab order, pinned buttons, styling."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QPushButton, QSplitter, QWidget

import bubbletrack
from bubbletrack.ui import main_window as mw
from bubbletrack.ui.tab_bar import TabBar

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

    def test_left_panel_scrolls_and_fit_buttons_stay_pinned(self, gui, qapp):
        window, _ = gui
        window.show()
        window.resize(1280, 800)
        qapp.processEvents()
        lp = window.left_panel
        bar = lp.pretune_tab.action_bar
        assert not lp.scroll_area.widget().isAncestorOf(bar)
        assert lp.pinned_footer.isAncestorOf(bar)
        # Content height depends on the fonts available; force it to be tall.
        lp.scroll_area.widget().setMinimumHeight(2000)
        qapp.processEvents()
        scroll = lp.scroll_area.verticalScrollBar()
        assert scroll.maximum() > 0  # content is taller than the panel
        scroll.setValue(scroll.maximum())
        qapp.processEvents()
        assert bar.isVisible() and lp.pretune_tab._fit_btn.isVisible()


class TestTabs:
    def test_workflow_order(self, gui):
        window, _ = gui
        assert TabBar.TAB_NAMES == ("Pre-tune", "Automatic", "Manual")
        stack = window.left_panel._stack
        lp = window.left_panel
        assert [stack.widget(i) for i in range(3)] == [
            lp.pretune_tab, lp.automatic_tab, lp.manual_tab]

    @pytest.mark.parametrize("name", TabBar.TAB_NAMES)
    def test_switching_updates_page_status_and_footer(self, gui, name):
        window, _ = gui
        lp = window.left_panel
        lp.tab_bar.select(name)
        page = {"Pre-tune": lp.pretune_tab, "Automatic": lp.automatic_tab,
                "Manual": lp.manual_tab}[name]
        assert lp._stack.currentWidget() is page
        assert window.status_bar._mode_label.text() == f"Mode: {name}"
        assert lp.pinned_footer.isHidden() == (name != "Pre-tune")
        assert lp.tab_bar.current_name() == name

    def test_short_tab_leaves_no_gap(self, gui):
        lp = gui[0].left_panel
        lp.tab_bar.select("Automatic")
        assert lp._stack.sizeHint().height() < lp.pretune_tab.sizeHint().height()
        lp.tab_bar.select("Pre-tune")
        assert lp._stack.sizeHint().height() >= lp.pretune_tab.sizeHint().height()

    def test_pinned_buttons_follow_the_run_lock(self, gui):
        window, controller = gui
        controller._apply_run_lock(True)
        assert not window.left_panel.pretune_tab.action_bar.isEnabled()
        controller._apply_run_lock(False)
        assert window.left_panel.pretune_tab.action_bar.isEnabled()

    def test_pinned_buttons_still_emit(self, gui):
        window, _ = gui
        tab = window.left_panel.pretune_tab
        hits = []
        tab.fit_clicked.connect(lambda: hits.append("fit"))
        tab.autotune_clicked.connect(lambda: hits.append("tune"))
        tab._fit_btn.click()
        tab._autotune_btn.click()
        assert hits == ["fit", "tune"]


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

    def test_idle_stop_button_in_the_app_is_not_red(self, gui, styled):
        window, _ = gui
        stop = window.left_panel.automatic_tab._stop_btn
        window.left_panel.tab_bar.select("Automatic")
        assert not stop.isEnabled()
        colour = _centre_colour(stop)
        window.close()
        assert not (colour.red() > 150 and colour.green() < 100)

    def test_active_view_mode_is_filled(self, gui, styled):
        window, _ = gui
        from bubbletrack.ui.image_compare import CompareMode
        buttons = window.compare_toolbar._buttons
        for btn in buttons.values():
            btn.resize(110, 28)
        active = _centre_colour(buttons[CompareMode.SIDE_BY_SIDE])
        idle = _centre_colour(buttons[CompareMode.WIPE])
        assert active.blue() > 200  # indigo fill
        assert idle.blue() < 80
