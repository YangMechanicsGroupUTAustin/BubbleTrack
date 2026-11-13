"""ML assist's teaching mode: paint examples, retrain in the background, preview.

While it is on, the left image panel paints strokes (green bubble, red
background, eraser; keys B / G / E, wheel = brush size, Ctrl+Z = undo a
stroke).  About half a second after the last change of the examples (a
stroke, or a frame verified anywhere in the app) the model is retrained on a
thread and the right panel shows ML's view of the shown frame: P(bubble) as
a green tint with the outline ML assist would store.  The suggested frames'
previews update too, and "Correct" keeps ML's result on one of them (locked:
a verified example from then on).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QKeySequence, QShortcut

from bubbletrack.controller import edge_support
from bubbletrack.controller.ml_preview import FramePreview, PreviewRequest, PreviewRunner
from bubbletrack.controller.ml_teach_view import card, describe, draw_strokes, show_preview
from bubbletrack.model.ml_assist import (
    default_brush, examples_for_run, protected_frames, readiness, stored_auto_map, suggestions,
    verified_circles,
)
from bubbletrack.model.result_history import record_edit, snapshot
from bubbletrack.model.teach import BACKGROUND, BUBBLE, MLExamples, stroke_pixels
from bubbletrack.ui.image_compare import CompareMode
from bubbletrack.ui.ml_images import BACKGROUND_COLOUR, BUBBLE_COLOUR
from bubbletrack.ui.ml_teach_panel import TOOL_KEYS, TOOLS
from bubbletrack.ui.shortcuts import text_input_has_focus

logger = logging.getLogger(__name__)

RETRAIN_DEBOUNCE_MS = 500  # quiet time after the last stroke before retraining
TOOL_COLOURS = {"bubble": BUBBLE_COLOUR, "background": BACKGROUND_COLOUR, "eraser": "#e4e4e7"}
TOOL_LABEL = {"bubble": BUBBLE, "background": BACKGROUND}
BRUSH_LIMITS = (1, 80)
_OK, _WARN = "#10b981", "#f59e0b"


class TeachingMode:
    """Strokes, undo, brush, background training and previews (see module doc)."""

    def __init__(self, ctrl, panel, dock) -> None:
        self._c = ctrl  # MLAssistController: state, window, app, set_examples
        self.panel = panel
        self.dock = dock
        self.active = False
        self._running = False
        self._tool = "bubble"
        self._radius: int | None = None  # brush radius (px); None: from the bubble size
        self._undo: list[MLExamples] = []
        self._version = 0  # bumped whenever the training examples change
        self._examples: MLExamples | None = None
        self._key: tuple | None = None  # what a model of the current examples is trained on
        self._model = None
        self._model_version = -1
        self._key_of_model: tuple | None = None
        self._previews: dict[int, FramePreview] = {}  # by the current model
        self._suggested: tuple[int, ...] = ()
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(RETRAIN_DEBOUNCE_MS)
        self._timer.timeout.connect(lambda: self.runner.request(train=True))
        self.runner = PreviewRunner(self._build_request, on_trained=self._on_trained,
                                    on_failed=self._on_failed, on_frame=self._on_frame)
        w = ctrl.w
        self._keys = []
        for tool in TOOLS:
            key = QShortcut(QKeySequence(TOOL_KEYS[tool]), w)
            key.activated.connect(lambda t=tool: self._on_key(t))
            key.setEnabled(False)
            self._keys.append(key)
        panel.tool_changed.connect(self.set_tool)
        panel.brush_size_changed.connect(self.set_radius)
        panel.undo_clicked.connect(self.undo_stroke)
        panel.clear_frame_clicked.connect(self.clear_frame)
        panel.frame_activated.connect(ctrl.go_to)
        panel.accept_requested.connect(self.keep_ml_result)
        w.original_panel.stroke_drawn.connect(self.on_stroke)
        w.original_panel.brush_scrolled.connect(self.on_wheel)
        w.original_panel.roi_selected.connect(lambda *_roi: self._settings_changed())
        box = w.left_panel.pretune_tab.edge_box
        box.use_current_reference_clicked.connect(self._settings_changed)
        box.auto_reference_clicked.connect(self._settings_changed)

    # -- entering and leaving ------------------------------------------------ #

    def enter(self) -> bool:
        state = self._c.state
        if not state.images or state.radius is None:
            self._c.status("Load images first", _WARN)
            return False
        self.active = True
        self._suggested = suggestions(state, reference=self._c.reference_index())
        self.panel.set_suggestions(self._suggested)
        if self._radius is None:
            self._radius = default_brush(state)
        self.panel.set_brush_size(self._radius)
        self.set_tool(self._tool)
        w = self._c.w
        if w.compare_mode is not CompareMode.SIDE_BY_SIDE:
            w.compare_toolbar.set_mode(CompareMode.SIDE_BY_SIDE)
        if self._on_display not in w.frame_display_hooks:
            w.frame_display_hooks.append(self._on_display)
        for key in self._keys:
            key.setEnabled(True)
        self.dock.show()
        self.dock.raise_()
        self.refresh_examples(force=True)
        if self._suggested and state.image_no not in self._suggested:
            self._c.go_to(self._suggested[0])
        else:
            self._c.redisplay()
        self.sync_mode()
        return True

    def leave(self) -> None:
        if not self.active:
            return
        self.active = False
        self._timer.stop()
        self.runner.cancel()
        self._previews.clear()
        w = self._c.w
        if self._on_display in w.frame_display_hooks:
            w.frame_display_hooks.remove(self._on_display)
        for key in self._keys:
            key.setEnabled(False)
        self.sync_mode()
        self.dock.hide()
        self._c.redisplay()

    def reset(self) -> None:
        """Another recording: forget strokes' undo, the model and the brush size."""
        self.leave()
        self._undo.clear()
        self._model, self._model_version, self._examples = None, -1, None
        self._key = self._key_of_model = None
        self._radius = None

    def has_model_for(self, examples: MLExamples) -> bool:
        """True when the preview model was trained on exactly *examples*."""
        return self._model is not None and self._key_of_model == self.model_key(examples)

    def model_key(self, examples: MLExamples) -> tuple:
        """A model is valid for these examples, this ROI and this reference frame."""
        state = self._c.state
        return examples, tuple(state.gridx), tuple(state.gridy), state.reference_frame

    def _settings_changed(self) -> None:
        """The ROI or the reference frame changed (after the pre-tune handlers)."""
        self.refresh_examples()
        self.sync_mode()

    # -- tools ------------------------------------------------------------------ #

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            return
        self._tool = tool
        self.panel.set_tool(tool)
        self._apply_brush()
        self.sync_mode()

    def tool(self) -> str:
        return self._tool

    def set_radius(self, radius: int) -> None:
        self._radius = int(np.clip(radius, *BRUSH_LIMITS))
        self.panel.set_brush_size(self._radius)
        self._apply_brush()

    @property
    def radius(self) -> int:
        return self._radius or default_brush(self._c.state)

    def on_wheel(self, step: int) -> None:
        if self.active:
            self.set_radius(self.radius + step * max(1, round(0.15 * self.radius)))

    def _apply_brush(self) -> None:
        self._c.w.original_panel.brush.set_brush(TOOL_COLOURS[self._tool], self.radius)

    def _on_key(self, tool: str) -> None:
        if self.active and not self._running and not text_input_has_focus():
            self.set_tool(tool)

    def sync_mode(self) -> None:
        """The image panel paints while teaching (other tools, e.g. the circle, win)."""
        panel = self._c.w.original_panel
        want = self.active and not self._running
        if want and panel.mode == "normal":
            panel.set_mode("paint")
        elif not want and panel.mode == "paint":
            panel.set_mode("normal")

    # -- strokes ------------------------------------------------------------------ #

    def _user_examples(self) -> MLExamples:
        ex = self._c.state.ml_examples
        return ex if isinstance(ex, MLExamples) else MLExamples()

    def on_stroke(self, points) -> None:
        """A finished stroke: scene (x, y) points on the shown frame."""
        state = self._c.state
        if not self.active or self._running or state.cur_img is None:
            return
        idx = state.image_no
        pts = [(y + 0.5, x + 0.5) for x, y in points]  # scene -> 1-based (row, col)
        px = stroke_pixels(pts, self.radius, state.cur_img.shape[:2])
        (r0, r1), (c0, c1) = state.gridx, state.gridy
        px = px[(px[:, 0] >= r0) & (px[:, 0] <= r1) & (px[:, 1] >= c0) & (px[:, 1] <= c1)]
        if not len(px):
            self._c.status("Paint inside the ROI: ML assist only looks there", _WARN)
            return
        old = self._user_examples()
        new = old.erase(idx, px) if self._tool == "eraser" else old.paint(idx, TOOL_LABEL[self._tool], px)
        self._commit(old, new)

    def clear_frame(self) -> None:
        idx = self._c.state.image_no
        old = self._user_examples()
        if self.active and not self._running and idx in old.strokes:
            self._commit(old, old.clear(idx))

    def undo_stroke(self) -> bool:
        """Ctrl+Z while teaching: the last stroke (False when there is none)."""
        if not self.active or self._running or not self._undo:
            return False
        self._c.set_examples(self._undo.pop())
        self._changed_strokes()
        self._c.status("Stroke undone", _OK)
        return True

    def _commit(self, old: MLExamples, new: MLExamples) -> None:
        if new == old:
            return
        self._undo.append(old)
        self._c.set_examples(new)
        self._changed_strokes()

    def _changed_strokes(self) -> None:
        self._c.redisplay()
        self.refresh_examples()

    # -- examples, training and previews ------------------------------------------ #

    def refresh_examples(self, *, force: bool = False) -> bool:
        """Re-read the examples (strokes + verified frames); retrain soon if they changed."""
        examples = examples_for_run(self._c.state)
        key = self.model_key(examples)
        changed = key != self._key
        if not (changed or force):
            return False
        self._examples, self._key = examples, key
        self._version += changed
        ready, text = readiness(examples)
        self.panel.set_quality(text, ready)
        self._update_cards()
        self._c.refresh_state_line()
        if not self.active or self._running:
            return changed
        if not ready:
            self.panel.set_preview("Paint on the bubble and on the background, or verify a frame.")
        elif self._model is not None and self._key_of_model == key:
            self._model_version = self._version  # the model still fits: measure only
            self.runner.request(train=False)
        else:
            self.panel.set_preview("Training…")
            self._timer.start()
        return changed

    def _build_request(self, train: bool) -> PreviewRequest | None:
        if not self.active or self._running:
            return None
        state = self._c.state
        if train and not readiness(self._examples)[0]:
            return None
        if not train and (self._model is None or self._key_of_model != self._key):
            return None
        frames = tuple(dict.fromkeys((state.image_no, *self._suggested)))
        if not train:
            frames = tuple(f for f in frames if f not in self._previews)
            if not frames:
                return None
        try:
            job = edge_support.build_job(state, self._c.max_radius())
        except Exception as exc:  # noqa: BLE001 - shown, never fatal
            self.panel.set_preview(f"Cannot preview: {exc}", warn=True)
            return None
        video = state.video_path if state.video_reader is not None else None
        return PreviewRequest(
            key=self._key if train else self._key_of_model,
            version=self._version if train else self._model_version,
            examples=self._examples if train else None, model=None if train else self._model,
            frames=frames, images=tuple(state.images), video_path=video, gridx=tuple(state.gridx),
            gridy=tuple(state.gridy), job=job, stored=stored_auto_map(state, frames))

    def _on_trained(self, req: PreviewRequest, model, ms: float) -> None:
        if req.version != self._version:
            return  # the examples changed meanwhile: a newer training follows
        self._model, self._model_version = model, req.version
        self._key_of_model = req.key
        self._previews.clear()
        info = model.info
        self.panel.set_quality_tip(f"Trained on {info.n_bubble + info.n_background:,} pixels of "
                                   f"{len(info.frames)} frame(s) in {ms / 1000:.1f} s")

    def _on_failed(self, req: PreviewRequest, message: str) -> None:
        if req.version in (self._version, self._model_version):
            self.panel.set_preview(f"ML assist cannot learn from these examples yet: {message}",
                                   warn=True)

    def _on_frame(self, req: PreviewRequest, preview: FramePreview) -> None:
        if req.version != self._model_version or not self.active:
            return
        self._previews[preview.frame] = preview
        if preview.frame in self._suggested:
            self._update_card(preview.frame)
        if preview.frame == self._c.state.image_no:
            self._show_preview(preview)

    def on_frame_changed(self, idx: int) -> None:
        if not self.active:
            return
        if idx in self._previews:
            self.panel.set_preview(*describe(self._previews[idx], protected_frames(self._c.state)))
        elif self._model is not None:
            self.panel.set_preview("Measuring…")
            self.runner.request(train=False)

    def on_results_changed(self) -> None:
        """Results changed elsewhere (a frame verified, a fit stored): re-sync."""
        if self.active and not self._running and not self.refresh_examples():
            self._update_cards()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.panel.set_running(self._running)
        if self._running:
            self._timer.stop()
            self.runner.cancel()
        elif self.active:
            self._previews.clear()  # results changed under them
            self.refresh_examples(force=True)
        self.sync_mode()

    # -- drawing --------------------------------------------------------------------- #

    def _on_display(self, state, idx: int) -> None:
        """Display hook: the frame's strokes, and ML's view on the right panel."""
        if not self.active:
            return
        draw_strokes(self._c.w, state, idx)
        preview = self._previews.get(idx)
        if preview is not None:
            self._show_preview(preview)

    def _show_preview(self, preview: FramePreview) -> None:
        state = self._c.state
        self.panel.set_preview(*describe(preview, protected_frames(state)))
        show_preview(self._c.w, state, preview)

    # -- suggested frames ------------------------------------------------------------- #

    def _update_cards(self) -> None:
        for frame in self._suggested:
            self._update_card(frame)

    def _update_card(self, frame: int) -> None:
        state = self._c.state
        preview = self._previews.get(frame)
        caption, pixmap = card(state, frame, preview)
        can_keep = (preview is not None and preview.fit.fit.radius > 0
                    and frame not in protected_frames(state) and not self._running)
        self.panel.set_card(frame, caption=caption, pixmap=pixmap,
                            verified=frame in verified_circles(state), can_accept=can_keep)

    def keep_ml_result(self, frame: int) -> None:
        """'Correct': store ML's result on *frame* and lock it (a verified example)."""
        preview = self._previews.get(frame)
        state = self._c.state
        if preview is None or preview.fit.fit.radius <= 0 or self._running or state.fit_locked is None:
            return
        before = snapshot(state, [frame])
        edge_support.store_fit(state, frame, preview.fit.fit, preview.fit.source)
        state.fit_locked[frame] = True
        self._c.history.push(record_edit(state, f"ML result kept on frame {frame + 1}", before))
        self._c.after_edit(frame)
        self._c.status(f"Frame {frame + 1}: ML's result kept and locked — an example from now on", _OK)
