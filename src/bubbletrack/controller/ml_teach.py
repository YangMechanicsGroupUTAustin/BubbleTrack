"""ML assist's teaching mode: choose two good example frames, then Train and apply.

The dock shows two frames at a time.  A candidate shows the outline that
would become the example (Auto's result, or Auto's guess) and asks "Is the
red outline on the bubble wall?":

* Looks right ✓ accepts the frame (locked, like Accept in Review; one
  Ctrl+Z): an example from now on, kept on its card;
* Looks wrong ✗ puts another frame in its place; a rejected frame is never
  offered again for this recording, and frames like it come last.

Frames confirmed anywhere else (Review, fixes) count and show as confirmed.
Once two confirmed frames differ (``ml_teach_steps``), Train and apply (the
controller) trains once and measures the frames that need review.  Nothing
trains here: no stroke and no confirmation starts a training.

"Paint examples by hand" (``ml_paint``) is folded away.  The image's mouse
has one owner at a time: entering ends Review's Fix (``end_fix``), a Fix
starting closes the teaching mode first (``yield_to_fix``), and the image
paints only while the painting section is open.
"""

from __future__ import annotations

import numpy as np

from bubbletrack.controller import ml_teach_steps as steps
from bubbletrack.controller.ml_paint import HandPainting
from bubbletrack.controller.ml_teach_view import card_picture, draw_strokes
from bubbletrack.model.ml_assist import protected_frames, stroke_count
from bubbletrack.model.teach import MLExamples
from bubbletrack.model.teach.suggest import FrameSize, differ, pick_candidate
from bubbletrack.ui.ml_teach_panel import CardView
from bubbletrack.ui.wording import plural

SLOTS = 2  # cards on show
WAIT_TEXT = "Wait for the running fit to finish."
YIELD_NOTE = "ML assist is closed (your examples are kept; Teach… opens it again)"
_OK, _WARN = "#10b981", "#f59e0b"


class TeachingMode:
    """The cards, Looks right / Looks wrong and the panel's words (see module doc)."""

    def __init__(self, ctrl, panel, dock) -> None:
        self._c = ctrl  # MLAssistController: state, window, accept_frame, scope, end_fix
        self.panel = panel
        self.dock = dock
        self.active = False
        self._running = False
        self._slots: list[int | None] = [None] * SLOTS
        self._rejected: set[int] = set()
        self._pictures: dict[tuple, object] = {}
        self.paint = HandPainting(ctrl, panel, wanted=self._painting, changed=self.refresh)
        panel.confirm_requested.connect(self.looks_right)
        panel.reject_requested.connect(self.looks_wrong)
        panel.frame_activated.connect(ctrl.go_to)
        panel.paint_toggled.connect(lambda _open: self.paint.sync())
        w = ctrl.w
        w.original_panel.roi_selected.connect(lambda *_roi: self._settings_changed())
        box = w.left_panel.measure_panel.edge_box
        box.use_current_reference_clicked.connect(self._settings_changed)
        box.auto_reference_clicked.connect(self._settings_changed)

    # -- what others read ------------------------------------------------------ #

    @property
    def slots(self) -> tuple[int | None, ...]:
        """The frames on the cards (None: an empty card)."""
        return tuple(self._slots)

    @property
    def rejected(self) -> frozenset[int]:
        return frozenset(self._rejected)

    def progress(self) -> steps.Progress:
        return steps.progress(self._c.state, prefer=[f for f in self._slots if f is not None])

    def undo_stroke(self) -> bool:
        """Ctrl+Z while painting by hand: the last stroke (False when there is none)."""
        return self.paint.undo_stroke()

    def _painting(self) -> bool:
        return self.active and self.panel.painting_open() and not self._running

    # -- entering and leaving ------------------------------------------------ #

    def enter(self) -> bool:
        state = self._c.state
        if not state.images or state.radius is None:
            self._c.status("Load images first", _WARN)
            return False
        ended = self._c.end_fix()  # one owner of the image at a time
        self.active = True
        self.paint.prepare()
        self.panel.set_painting(False)  # folded each time
        hooks = self._c.w.frame_display_hooks
        if self._on_display not in hooks:
            hooks.append(self._on_display)
        self.dock.show()
        self.dock.raise_()
        self.refresh()
        self._show_next_frame(stay=True)
        self.paint.sync()
        if ended:
            self._c.status(f"{ended}: ML assist uses the image now", _WARN)
        return True

    def leave(self) -> None:
        if not self.active:
            return
        self.active = False
        hooks = self._c.w.frame_display_hooks
        if self._on_display in hooks:
            hooks.remove(self._on_display)
        self.paint.sync()
        self.dock.hide()
        self._c.redisplay()

    def yield_to_fix(self) -> str:
        """``FixMode`` hook: a fix takes the image, so leave first; a note for the status."""
        if not self.active:
            return ""
        self.leave()
        return YIELD_NOTE

    def reset(self) -> None:
        """Another recording: forget the cards, the rejections and the painting's undo."""
        self.leave()
        self.paint.reset()
        self._slots = [None] * SLOTS
        self._rejected.clear()
        self._pictures.clear()

    def _settings_changed(self) -> None:
        """The ROI or the reference frame changed (after the pre-tune handlers)."""
        self.refresh()
        self.paint.sync()

    # -- the panel ------------------------------------------------------------ #

    def refresh(self) -> None:
        """Cards, step, next step and the Train button from the current results."""
        if not self.active:
            return
        self._fill_slots(self.progress())
        progress = self.progress()
        self.panel.set_cards([self._view(f, progress) for f in self._slots])
        self.panel.set_step(steps.step_text(progress))
        scope = len(self._c.scope())
        if self._running:
            self.panel.set_next(WAIT_TEXT)
        else:
            judge = sum(1 for f in self._slots if f is not None and f not in progress.frames)
            self.panel.set_next(*steps.next_text(progress, candidates=judge, scope=scope))
        self.panel.set_train(*steps.train_state(progress, scope=scope))
        self.panel.set_paint_summary(_paint_summary(self._c.state.ml_examples))
        self._c.refresh_state_line()

    def _fill_slots(self, progress: steps.Progress) -> None:
        """Keep the cards that still hold, show the examples, fill the rest."""
        radius, confidence, trusted = steps.outlines(self._c.state)
        n = len(radius)
        examples = {e.frame: e for e in progress.examples}

        def size(f: int) -> FrameSize:
            return examples[f] if f in examples else FrameSize(f, radius[f])

        want = self._examples_to_show(progress)
        slots = [f if f in want or (f not in examples and self._offerable(f, radius)) else None
                 for f in self._slots]
        for frame in (f for f in want if f not in slots):
            free = [k for k, f in enumerate(slots) if f is None] or \
                [k for k, f in enumerate(slots) if f not in want][-1:]
            slots[free[0]] = frame
        shown = [size(f) for f in slots if f in examples]
        for k, f in enumerate(slots):  # a candidate differs from what is shown before it
            before = shown + [size(g) for g in slots[:k] if g is not None and g not in examples]
            if f is not None and f not in examples and not all(differ(size(f), b, n) for b in before):
                slots[k] = None
        skip = (self._rejected | set(examples) | protected_frames(self._c.state)
                | {self._c.reference_index()})
        avoid = [FrameSize(f, radius[f]) for f in sorted(self._rejected)]
        for k in range(SLOTS):
            if slots[k] is None:
                on_show = [g for g in slots if g is not None]
                slots[k] = pick_candidate(radius, confidence, trusted, exclude=skip | set(on_show),
                                          keep_apart=[size(g) for g in on_show], avoid=avoid)
        self._slots = slots

    def _examples_to_show(self, progress: steps.Progress) -> list[int]:
        """Two different examples once there are; else one (the one on show first)."""
        if progress.pair is not None:
            return [e.frame for e in progress.pair]
        on_cards = [f for f in self._slots if f in progress.frames]
        return on_cards[:1] or [e.frame for e in progress.examples[:1]]

    def _offerable(self, frame: int | None, radius: np.ndarray) -> bool:
        """Still a candidate: an outline, not rejected, not the user's, not the reference."""
        return (frame is not None and 0 <= frame < len(radius) and bool(radius[frame] > 0)
                and frame not in self._rejected and frame != self._c.reference_index()
                and frame not in protected_frames(self._c.state))

    def _view(self, frame: int | None, progress: steps.Progress) -> CardView | None:
        if frame is None:
            return None
        state = self._c.state
        example = next((e for e in progress.examples if e.frame == frame), None)
        painted = example is not None and not example.sized  # an example by its strokes
        outline = None if painted else steps.outline(state, frame)
        ex = state.ml_examples
        strokes = ex.strokes.get(frame) if isinstance(ex, MLExamples) else None
        if outline is None:
            caption = f"Frame {frame + 1} · painted by hand"
        elif outline.guess:
            caption = f"Frame {frame + 1} · Auto's guess, R = {outline.radius:.0f} px"
        else:
            caption = f"Frame {frame + 1} · R = {outline.radius:.0f} px"
        return CardView(frame, caption, frame in progress.frames, self._picture(frame, outline, strokes))

    def _picture(self, frame: int, outline, strokes):
        key = (frame, None if outline is None else (outline.radius, outline.rc, outline.cc),
               None if strokes is None else (len(strokes.bubble), len(strokes.background)))
        if key not in self._pictures:
            self._pictures[key] = card_picture(self._c.state, frame, outline, strokes)
        return self._pictures[key]

    def _show_next_frame(self, *, stay: bool = False) -> None:
        """Show the frame to judge next; with *stay*, keep a shown frame that is on a card."""
        state = self._c.state
        done = self.progress().frames
        on_cards = [f for f in self._slots if f is not None]
        todo = [f for f in on_cards if f not in done] or on_cards
        if not todo or (stay and state.image_no in on_cards) or todo[0] == state.image_no:
            self._c.redisplay()
        else:
            self._c.go_to(todo[0])

    # -- Looks right / Looks wrong ------------------------------------------- #

    def looks_right(self, frame: int) -> None:
        """Accept *frame* (locked): an example from now on (Ctrl+Z undoes)."""
        if not self.active or self._running or frame not in self._slots:
            return
        if self._c.accept_frame(frame) is None:
            self.refresh()
            self._c.status(f"Frame {frame + 1} has no outline to confirm", _WARN)
            return
        self.refresh()
        self._show_next_frame()
        self._c.status(f"Frame {frame + 1} confirmed: an example for ML assist — Ctrl+Z undoes", _OK)

    def looks_wrong(self, frame: int) -> None:
        """Another frame instead of *frame* (never offered again); nothing is changed."""
        if not self.active or self._running or frame not in self._slots:
            return
        self._rejected.add(frame)
        k = self._slots.index(frame)
        self._slots[k] = None
        self.refresh()
        self._show_next_frame()
        new = self._slots[k]
        instead = f"frame {new + 1} instead" if new is not None else "no other frame is left to offer"
        self._c.status(f"Frame {frame + 1} will not be used as an example: {instead}", _OK)

    # -- runs and results ------------------------------------------------------ #

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.panel.set_running(self._running)
        if self._running:
            self.panel.set_next(WAIT_TEXT)
        else:
            self.refresh()
        self.paint.sync()

    def show_measuring(self, frames: int) -> None:
        """Train and apply runs: say so (the panel shows its progress and Stop)."""
        self.panel.set_next(f"Measuring {plural(frames, 'frame')} with ML assist… "
                            "Stop keeps the frames done so far.")

    def on_results_changed(self) -> None:
        """Results changed (a frame confirmed or fixed anywhere, a run): re-sync."""
        if self.active and not self._running:
            self.refresh()

    def _on_display(self, state, idx: int) -> None:
        """Display hook: the frame's painted strokes over the left image."""
        if self.active:
            draw_strokes(self._c.w, state, idx)


def _paint_summary(examples) -> str:
    if not isinstance(examples, MLExamples) or not examples.strokes:
        return "Nothing painted yet."
    return (f"Painted: {plural(stroke_count(examples), 'stroke')} "
            f"on {plural(len(examples.strokes), 'frame')}.")
