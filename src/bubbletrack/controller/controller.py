"""AppController — thin signal router that owns state and sub-controllers."""

from __future__ import annotations

import logging
import os
from functools import partial

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bubbletrack.controller.auto_controller import AutoController
from bubbletrack.controller.circle_controller import CircleController
from bubbletrack.controller.display_mixin import display_frame, refresh_chart
from bubbletrack.controller.export_controller import ExportController
from bubbletrack.controller.file_controller import FileController
from bubbletrack.controller.manual_controller import ManualController
from bubbletrack.controller.pretune_controller import PretuneController
from bubbletrack.controller.retrack_controller import RetrackController
from bubbletrack.controller.session_controller import SessionController
from bubbletrack.controller.settings_controller import SettingsController
from bubbletrack.event_bus import RESULTS_CHANGED, EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.constants import DISPLAY_DEBOUNCE_MS, PREVIEW_DEBOUNCE_MS
from bubbletrack.model.config import load_config, ui_setting
from bubbletrack.model.result_history import ResultHistory
from bubbletrack.model.state import AppState, clear_frame_result, update_state
from bubbletrack.ui.edge_options import STEEP_EDGE
from bubbletrack.ui.measure_panel import AUTO_METHOD
from bubbletrack.ui.shortcuts import setup_shortcuts
from bubbletrack.ui.wording import plural

logger = logging.getLogger(__name__)


def initial_state(saved: dict) -> AppState:
    """The persisted parameters; the app starts in Auto unless classic methods are on.

    Older versions saved "intensity" as the default method in every config,
    so a saved classic method only counts when Settings shows classic methods.
    (Sessions keep their own method: loading one restores it.)
    """
    params = {"detection_mode": AUTO_METHOD, **(saved or {})}
    if params["detection_mode"] != AUTO_METHOD and not ui_setting("show_classic_methods"):
        params["detection_mode"] = AUTO_METHOD
    return update_state(AppState(), **params)


class AppController:
    """Thin orchestrator: owns state, creates sub-controllers, wires Qt signals.

    All domain logic lives in the sub-controllers.  This class only:
    1. Holds the canonical ``AppState`` instance.
    2. Creates and configures sub-controllers.
    3. Connects Qt widget signals to the correct handler.
    """

    def __init__(self, window) -> None:
        self.w = window
        # Load persisted config and apply to initial state
        self._state = initial_state(load_config())
        self._max_radius: float = float("inf")

        # LRU image cache (200 MB limit)
        self._image_cache = ImageCache()

        # Event bus for decoupled inter-controller communication
        self.bus = EventBus()

        # Debounce timers for slider-driven updates
        self._display_timer = QTimer()
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(DISPLAY_DEBOUNCE_MS)
        self._display_timer.timeout.connect(
            lambda: display_frame(
                self._state, self.w, self._state.image_no, self._set_state,
                self._image_cache,
            )
        )

        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)

        # Create sub-controllers (shared state via get/set accessors)
        gs, ss = self._get_state, self._set_state

        self.file_ctrl = FileController(
            self.bus, gs, ss, self.w, self._set_max_radius,
            self._image_cache,
            is_busy=lambda: self.auto_ctrl.is_busy(),
        )
        self.pretune_ctrl = PretuneController(
            self.bus, gs, ss, self.w,
            self._display_timer, self._preview_timer, self._get_max_radius,
            self._image_cache,
        )
        # Manual fixes (points, circles) and automatic re-tracks share one
        # undo history: Ctrl+Z always undoes the newest of them.
        self.history = ResultHistory()
        self.manual_ctrl = ManualController(
            self.bus, gs, ss, self.w, self._get_max_radius,
            self._image_cache, history=self.history,
        )
        self.auto_ctrl = AutoController(
            self.bus, gs, ss, self.w, self._get_max_radius,
            self._image_cache,
        )
        self.retrack = RetrackController(
            self.bus, gs, ss, self.w, self._get_max_radius, self._image_cache,
            history=self.history, is_run_active=lambda: self.auto_ctrl.is_busy(),
            queued_frames=lambda: self.review.queued_frames(),
        )
        self.circle_ctrl = CircleController(
            self.bus, gs, ss, self.w, self._get_max_radius, self._image_cache,
            history=self.history, retrack=self.retrack,
            is_run_active=lambda: self.auto_ctrl.is_busy(),
        )
        self.export_ctrl = ExportController(self.bus, gs, ss, self.w)
        self.session_ctrl = SessionController(
            self.bus, gs, ss, self.w, self.file_ctrl,
            is_busy=lambda: self.auto_ctrl.is_busy(),
            locate_folder=lambda original: self._locate_moved_folder(original),
            after_restore=lambda: self._after_session_restore(),
        )

        # Wire preview timer to pretune handler
        self._preview_timer.timeout.connect(self.pretune_ctrl.preview_detection)

        # File > Settings: classic methods, image views (applied live)
        self.settings = SettingsController(self.w, self)

        # Lock everything that mutates results while a worker writes into them
        self.bus.subscribe("run_state_changed", self._apply_run_lock)
        self.w.set_close_guard(self._confirm_close)

        self._connect_signals()
        self._sync_ui_from_state()

        # Review queue dock (suspicious frames, one-key triage)
        from bubbletrack.controller.review_controller import install_review
        self.review = install_review(self.w, self)
        self.bus.subscribe("tracking_review", self.review.set_extra)
        self.auto_ctrl.review_count = lambda: self.review.count  # "· 15 to check in Review"

        # Key-frame filmstrip above the R(t) chart
        from bubbletrack.controller.filmstrip_controller import install_filmstrip
        self.filmstrip = install_filmstrip(self.w, self)

        # File > Export Hard Case... and the Review rows' right-click menu
        from bubbletrack.controller.hard_case_controller import install_hard_cases
        self.hard_cases = install_hard_cases(self.w, self)

        # ML assist: the Auto box's helper (teaching dock, Apply, Fit All)
        from bubbletrack.controller.ml_controller import install_ml_assist
        self.ml = install_ml_assist(self.w, self)

        # Keyboard shortcuts
        self._shortcuts = setup_shortcuts(self.w, {
            "frame_prev": lambda: self.file_ctrl.on_frame_changed(
                max(0, self._state.image_no - 1),
            ),
            "frame_next": lambda: self.file_ctrl.on_frame_changed(
                min(self._state.total_frames - 1, self._state.image_no + 1),
            ),
            "zoom_in": self.w.original_panel.zoom_in,
            "zoom_out": self.w.original_panel.zoom_out,
            "zoom_reset": self.w.original_panel.reset_zoom,
            "undo": self._on_undo,
            "play_pause": self.w.frame_scrubber.toggle_play,
        })
        self.circle_ctrl.sync_tool()

    def _on_undo(self) -> None:
        """Ctrl+Z: the last clicked edge point while selecting points, else the
        newest manual fix (points or circle), clear, accepted guess or
        automatic re-track.  While ML assist teaches, its last brush stroke
        comes first."""
        if self.ml.teach.undo_stroke():
            return
        if not self.manual_ctrl.undo_last_point():
            self.circle_ctrl.undo()

    # ------------------------------------------------------------------ #
    #  State accessors (shared with sub-controllers)
    # ------------------------------------------------------------------ #

    def _get_state(self) -> AppState:
        return self._state

    def _set_state(self, new: AppState) -> None:
        self._state = new

    def _get_max_radius(self) -> float:
        return self._max_radius

    def _set_max_radius(self, value: float) -> None:
        self._max_radius = value

    # ------------------------------------------------------------------ #
    #  Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        lp = self.w.left_panel

        # Image source
        lp.image_source.folder_selected.connect(self.file_ctrl.on_folder_selected)
        lp.image_source.video_selected.connect(self.file_ctrl.on_video_selected)

        # Frame scrubber
        self.w.frame_scrubber.value_changed.connect(self.file_ctrl.on_frame_changed)

        # Measuring settings (the Auto box; classic methods when shown)
        mp = lp.measure_panel
        mp.threshold_changed.connect(self.pretune_ctrl.on_threshold_changed)
        mp.invert_mask_changed.connect(self.pretune_ctrl.on_invert_mask_changed)
        mp.removing_factor_changed.connect(self.pretune_ctrl.on_removing_factor_changed)
        mp.edges_changed.connect(self.pretune_ctrl.on_edges_changed)
        mp.autotune_clicked.connect(self.pretune_ctrl.on_autotune)
        mp.filters_changed.connect(self.pretune_ctrl.on_filters_changed)

        # Clicks on the image: edge points of Review's Fix, or the seed pick
        self.w.original_panel.point_clicked.connect(self.manual_ctrl.on_point_clicked)
        self.w.original_panel.point_clicked.connect(self.pretune_ctrl.on_point_clicked)
        eb = mp.edge_box
        eb.use_current_reference_clicked.connect(self.pretune_ctrl.on_use_current_as_reference)
        eb.auto_reference_clicked.connect(self.pretune_ctrl.on_auto_reference)
        eb.pick_seed_clicked.connect(self.pretune_ctrl.on_pick_seed)
        eb.clear_seed_clicked.connect(self.pretune_ctrl.on_clear_seed)
        self.bus.subscribe("source_loaded", lambda _n: self.pretune_ctrl.refresh_edge_labels())

        # Fit (every frame) and Stop
        lp.fit_controls.fit_clicked.connect(self.auto_ctrl.on_auto_fit)
        lp.fit_controls.stop_clicked.connect(self.auto_ctrl.on_auto_stop)

        # Export dialog (side-panel button and File menu); Edit > Clear All Results
        lp.export_clicked.connect(self.export_ctrl.open_dialog)
        self.w.export_requested.connect(self.export_ctrl.open_dialog)
        self.w.clear_results_requested.connect(self._on_clear_results)

        # ROI: button click → enter selection mode, drag complete → apply ROI
        self.w.original_panel.select_roi_clicked.connect(self.pretune_ctrl.on_select_roi)
        self.w.original_panel.roi_selected.connect(self.pretune_ctrl.on_roi_selected)

        # R-t chart click-to-jump
        self.w.radius_chart.point_clicked.connect(self.file_ctrl.on_frame_changed)

        # R-t chart brushed range: refit / lock / unlock / no bubble (a point's
        # "Fix frame N..." is wired by the Review dock)
        chart = self.w.radius_chart
        chart.refit_range_requested.connect(self.auto_ctrl.fit_range)
        chart.lock_range_requested.connect(partial(self._on_lock_range, locked=True))
        chart.unlock_range_requested.connect(partial(self._on_lock_range, locked=False))
        chart.no_bubble_range_requested.connect(self._on_mark_no_bubble)

        # Compare mode change — refresh display immediately
        self.w.compare_mode_changed.connect(self._on_compare_mode_changed)

        # Session save/load (menubar File menu)
        self.w.save_session_requested.connect(self._on_save_session)
        self.w.load_session_requested.connect(self._on_load_session)

        # Last: these slots must run after the handlers connected above.
        self._connect_circle_tool()

    def _connect_circle_tool(self) -> None:
        """Review's "Drag a circle" tool and the automatic re-track."""
        lp, circle = self.w.left_panel, self.circle_ctrl
        self.w.original_panel.circle_drawn.connect(circle.on_circle_drawn)
        # After the measuring handlers stored the new detection mode / ROI:
        lp.measure_panel.filters_changed.connect(circle.sync_tool)
        self.w.original_panel.roi_selected.connect(lambda *_roi: circle.sync_tool())
        # After AutoController's Stop handler, so a re-track's status wins.
        lp.fit_controls.stop_clicked.connect(self.retrack.stop)

    def _sync_ui_from_state(self) -> None:
        """Reflect loaded/persisted state in the visible controls."""
        if self._state.edge_definition != STEEP_EDGE:
            # The "Wall at: outer extent" choice is gone: Auto uses the 50% edge.
            logger.info("Auto wall position %r replaced by the steep (50%%) edge",
                        self._state.edge_definition)
            self._state = update_state(self._state, edge_definition=STEEP_EDGE)
        pt = self.w.left_panel.measure_panel
        pt.set_threshold(self._state.img_thr)
        pt.set_removing_factor(self._state.removing_factor)
        pt.set_invert_mask(self._state.invert_mask)
        pt.set_edge_flags(self._state.bubble_cross_edges)
        pt.set_filter_params({
            "gaussian_sigma": self._state.gaussian_sigma,
            "clahe_clip": self._state.clahe_clip,
            "closing_radius": self._state.closing_radius,
            "opening_radius": self._state.opening_radius,
            "detection_mode": self._state.detection_mode,
            "area_max_regions": self._state.area_max_regions,
            "gradient_boundary": self._state.gradient_boundary,
            "temporal_refine": self._state.temporal_refine,
            "edge_definition": self._state.edge_definition,
            "edge_polarity": self._state.edge_polarity,
        })
        self.pretune_ctrl.refresh_edge_labels()
        self.settings.apply_classic()  # a classic method keeps its controls in view

    def _apply_run_lock(self, running: bool) -> None:
        """Disable every control that loads data or edits results during a run.

        Frame navigation stays available so the user can watch progress.
        """
        enabled = not running
        lp = self.w.left_panel
        for widget in (lp.image_source, lp.measure_panel, lp.export_button):
            widget.setEnabled(enabled)
        for action in (self.w.export_action, self.w.clear_results_action):
            action.setEnabled(enabled)
        self.w.save_session_action.setEnabled(enabled)
        self.w.load_session_action.setEnabled(enabled)
        self.w.radius_chart.set_editing_enabled(enabled)

    def _on_clear_results(self) -> None:
        """Edit > Clear All Results: remove every result once the user confirms."""
        radius = self._state.radius
        if radius is None or self.auto_ctrl.is_busy():
            return
        if not (radius > 0).any() and not (self._state.fit_locked is not None
                                           and self._state.fit_locked.any()):
            self.w.header.set_status("No results to clear", "#f59e0b")
            return
        if self._ask_yes_no(
            "Clear all results?",
            f"Remove the results of all {len(radius)} frames, including manual "
            "fixes and locked frames?\n\nThis cannot be undone.",
        ):
            self.auto_ctrl.on_auto_clear()

    # ------------------------------------------------------------------ #
    #  Compare mode refresh
    # ------------------------------------------------------------------ #

    def _on_compare_mode_changed(self, _mode_value: str) -> None:
        """Re-render the current frame when compare mode changes."""
        if self._state.images:
            display_frame(
                self._state, self.w, self._state.image_no,
                self._set_state, self._image_cache,
            )

    # ------------------------------------------------------------------ #
    #  Result editing (the chart's brushed range)
    # ------------------------------------------------------------------ #

    def _frames_in(self, start: int, end: int) -> range:
        """0-based frames start..end (inclusive), clipped to the loaded results."""
        n = 0 if self._state.radius is None else len(self._state.radius)
        return range(max(start, 0), min(end, n - 1) + 1)

    def _on_lock_range(self, start: int, end: int, *, locked: bool) -> None:
        """Lock the fitted frames of start..end (or unlock the whole range)."""
        flags = self._state.fit_locked
        if flags is None:
            return
        frames = [i for i in self._frames_in(start, end)
                  if not locked or self._state.radius[i] > 0]
        flags[frames] = locked
        self._refresh_chart()
        verb = "Locked" if locked else "Unlocked"
        self.w.header.set_status(f"{verb} {plural(len(frames), 'frame')}", "#10b981")
        self.bus.emit(RESULTS_CHANGED)

    def _on_mark_no_bubble(self, start: int, end: int) -> None:
        """Say "no bubble" for frames start..end: clear and lock them empty.

        The only way to say a frame is empty (nothing deletes a result):
        Fit All and tracking then leave the frames empty, and hard-case exports
        label them "no bubble".  Frames locked with a result are kept.
        """
        locked = self._state.fit_locked
        if locked is None:
            return
        frames = self._frames_in(start, end)
        targets = [i for i in frames if not (locked[i] and self._state.radius[i] > 0)]
        if not targets:
            self.w.header.set_status("Nothing to mark: every frame is locked with a result", "#f59e0b")
            return
        for i in targets:
            clear_frame_result(self._state, i)
            locked[i] = True
        self._refresh_chart()
        if self._state.image_no in targets:
            display_frame(self._state, self.w, self._state.image_no,
                          self._set_state, self._image_cache)
        kept = len(frames) - len(targets)
        text = f"Marked {plural(len(targets), 'frame')} as no bubble"
        self.w.header.set_status(text + (f" ({kept} locked kept)" if kept else ""), "#10b981")
        self.bus.emit(RESULTS_CHANGED)

    def _refresh_chart(self) -> None:
        """Refresh the R-t chart with quality-based coloring."""
        refresh_chart(self._state, self.w)

    # ------------------------------------------------------------------ #
    #  Session save/load
    # ------------------------------------------------------------------ #

    def save_session_file(self, path: str) -> bool:
        return self.session_ctrl.save(path)

    def load_session_file(self, path: str) -> bool:
        return self.session_ctrl.load(path)

    def _on_save_session(self) -> None:
        """Prompt the user for a path and save the current session."""
        self.session_ctrl.save_as()

    def _on_load_session(self) -> None:
        """Prompt the user for a .brt file and restore the session."""
        default_dir = self._state.folder_path or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self.w, "Load Session", default_dir,
            "BubbleTrack Session (*.brt);;All Files (*)",
        )
        if path:
            self.load_session_file(path)

    def _locate_moved_folder(self, original: str) -> str:
        """Ask the user where a session's image folder went; '' if cancelled."""
        QMessageBox.information(
            self.w, "Image folder not found",
            f"The session's image folder no longer exists:\n{original}\n\n"
            "Please select its new location.",
        )
        return QFileDialog.getExistingDirectory(
            self.w, "Locate the image folder", os.path.dirname(original),
        )

    def _after_session_restore(self) -> None:
        """Bring every view in line with a freshly restored state."""
        self._sync_ui_from_state()
        self.circle_ctrl.sync_tool()  # the restored detection mode decides the tool
        self.w.original_panel.set_roi_text(self._state.gridx, self._state.gridy)
        if self._state.images:
            self.file_ctrl.on_frame_changed(self._state.image_no)
        self._refresh_chart()
        self.ml.refresh_state_line()  # the restored ML assist examples and switch

    # ------------------------------------------------------------------ #
    #  Closing the window
    # ------------------------------------------------------------------ #

    def _ask_yes_no(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self.w, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_close(self) -> bool:
        """Return False to keep the window open.

        A running fit needs confirmation.  Results are always kept in
        ~/.bubbletrack/recovery/autosave.brt so closing never loses them.
        """
        if self.auto_ctrl.is_busy():
            if not self._ask_yes_no(
                "Fit in progress",
                "A fit is still running. Stop it and quit?",
            ):
                return False
            self.auto_ctrl.stop_and_wait()
        self.retrack.shutdown()
        self.ml.shutdown()
        self.session_ctrl.save_last_session()
        return True
