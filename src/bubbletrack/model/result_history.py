"""Undo history of per-frame results (circle fixes, automatic re-tracks).

An edit records each changed frame's result before and after.  Undo puts
the "before" back exactly, but only while every frame still holds what the
edit wrote: a later change (a delete, a refit, Fit All, another source) is
never overwritten by an old snapshot.  Qt-free.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from bubbletrack.model.undo import UndoStack

_NAN = float("nan")


@dataclass(frozen=True, eq=False)
class FrameResult:
    """One frame's entries of every result array (copies)."""

    radius: float
    centre: tuple[float, float]
    edge_xy: np.ndarray | None
    region_count: int
    source: int
    confidence: float
    metrics: tuple[tuple[str, float], ...]
    flags: frozenset
    locked: bool


def _item(values, idx: int, default):
    return default if values is None else values[idx]


def capture(state, idx: int) -> FrameResult:
    """Copy frame *idx*'s result out of the state arrays."""
    centre = state.circle_fit_par[idx] if state.circle_fit_par is not None else (_NAN, _NAN)
    xy = _item(state.circle_xy, idx, None)
    return FrameResult(
        radius=float(state.radius[idx]),
        centre=(float(centre[0]), float(centre[1])),
        edge_xy=None if xy is None else np.array(xy, dtype=float, copy=True),
        region_count=int(_item(state.region_count, idx, 0)),
        source=int(_item(state.fit_source, idx, 0)),
        confidence=float(_item(state.fit_confidence, idx, _NAN)),
        metrics=tuple((key, float(values[idx]))
                      for key, values in sorted((state.fit_metrics or {}).items())),
        flags=frozenset(_item(state.fit_flags, idx, frozenset())),
        locked=bool(_item(state.fit_locked, idx, False)),
    )


def _set(values, idx: int, value) -> None:
    if values is not None:
        values[idx] = value


def restore(state, idx: int, result: FrameResult) -> None:
    """Write *result* back into frame *idx* of the state arrays (in place)."""
    state.radius[idx] = result.radius
    _set(state.circle_fit_par, idx, list(result.centre))
    _set(state.circle_xy, idx, None if result.edge_xy is None else result.edge_xy.copy())
    _set(state.region_count, idx, result.region_count)
    _set(state.fit_source, idx, result.source)
    _set(state.fit_confidence, idx, result.confidence)
    _set(state.fit_flags, idx, result.flags)
    _set(state.fit_locked, idx, result.locked)
    metrics = dict(result.metrics)
    for key, values in (state.fit_metrics or {}).items():
        values[idx] = metrics.get(key, _NAN)


def _same_float(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def same_result(a: FrameResult, b: FrameResult) -> bool:
    """Bit-for-bit equal results (NaN equals NaN)."""
    if a.edge_xy is None or b.edge_xy is None:
        same_xy = a.edge_xy is None and b.edge_xy is None
    else:
        same_xy = np.array_equal(a.edge_xy, b.edge_xy, equal_nan=True)
    return (same_xy
            and all(_same_float(x, y) for x, y in ((a.radius, b.radius), (a.confidence, b.confidence),
                                                    (a.centre[0], b.centre[0]), (a.centre[1], b.centre[1])))
            and (a.region_count, a.source, a.flags, a.locked) == (b.region_count, b.source,
                                                                   b.flags, b.locked)
            and len(a.metrics) == len(b.metrics)
            and all(ka == kb and _same_float(va, vb)
                    for (ka, va), (kb, vb) in zip(a.metrics, b.metrics)))


def snapshot(state, frames: Iterable[int]) -> dict[int, FrameResult]:
    """Results of *frames* before an edit."""
    return {int(i): capture(state, int(i)) for i in sorted(set(frames))}


@dataclass(frozen=True)
class ResultEdit:
    """What one edit changed: frame -> result before / after."""

    label: str
    n_frames: int
    before: Mapping[int, FrameResult]
    after: Mapping[int, FrameResult]

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(sorted(self.before))


def record_edit(state, label: str, before: Mapping[int, FrameResult]) -> ResultEdit | None:
    """The edit made since *before* was taken, or None if nothing changed."""
    after = {i: capture(state, i) for i in before}
    changed = [i for i in before if not same_result(before[i], after[i])]
    if not changed:
        return None
    return ResultEdit(label, len(state.radius), {i: before[i] for i in changed},
                      {i: after[i] for i in changed})


@dataclass(frozen=True)
class UndoOutcome:
    edit: ResultEdit | None
    done: bool
    message: str


class ResultHistory:
    """Last-in-first-out undo of :class:`ResultEdit` (newest edits kept)."""

    def __init__(self, max_size: int = 50) -> None:
        self._stack = UndoStack(max_size)

    def push(self, edit: ResultEdit | None) -> None:
        if edit is not None:
            self._stack.push(edit)

    def can_undo(self) -> bool:
        return self._stack.can_undo()

    def clear(self) -> None:
        self._stack.clear()

    def undo(self, state) -> UndoOutcome:
        """Restore the newest edit; refused (and dropped) if its frames changed since."""
        edit = self._stack.undo()
        if edit is None:
            return UndoOutcome(None, False, "Nothing to undo")
        if state.radius is None or len(state.radius) != edit.n_frames:
            return UndoOutcome(edit, False, f"Cannot undo the {edit.label}: other data is loaded")
        moved = [i for i in edit.frames if not same_result(capture(state, i), edit.after[i])]
        if moved:
            return UndoOutcome(edit, False, f"Cannot undo the {edit.label}: "
                                            f"frame {moved[0] + 1} was changed since")
        for i in edit.frames:
            restore(state, i, edit.before[i])
        return UndoOutcome(edit, True, f"Undid the {edit.label}")
