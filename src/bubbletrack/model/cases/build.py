"""Assemble a hard case from the app state and write it (Qt-free).

The controller decides which frames (a chart selection, a Review row or a
typed range) and supplies the frames' pixels; this module picks the
neighbours and the reference frame, crops everything to the ROI, records
results and labels, and writes the bundle.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np

import bubbletrack
from bubbletrack.model.cases.bundle import frame_file_name, write_bundle
from bubbletrack.model.cases.labels import app_result, frame_label
from bubbletrack.model.cases.schema import (
    ROLE_CONTEXT, ROLE_REFERENCE, ROLE_SELECTED, CaseFrame, Detection, HardCase,
)
from bubbletrack.model.conventions import roi_to_slice
from bubbletrack.model.hpvx2_dat import DAT_EXTS
from bubbletrack.model.session import session_parameters

CONTEXT_FRAMES = 3  # neighbours stored before and after the reported frames
SIZE_WARN_BYTES = 50 * 1024 * 1024  # bundles above this are awkward to send
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


@dataclass(frozen=True)
class CaseRequest:
    """What the user asked for, plus what the controller knows about the run."""

    start: int  # first reported frame (0-based)
    end: int  # last reported frame (0-based, inclusive)
    note: str = ""
    reporter: str = ""
    reference_index: int | None = None  # Auto mode's reference frame (resolved)
    tracking: bool = False  # Auto mode: tracking after Fit All was on
    max_radius: float = float("inf")


def clip_range(start: int, end: int, n_frames: int) -> tuple[int, int]:
    """(first, last) of start..end in either order, clipped to the recording."""
    return max(0, min(start, end)), min(n_frames - 1, max(start, end))


def plan_frames(start: int, end: int, n_frames: int, *, reference: int | None = None,
                context: int = CONTEXT_FRAMES) -> tuple[tuple[int, str], ...]:
    """(index, role) of every frame to store, in frame order.

    The reported frames, *context* neighbours on each side (clipped to the
    recording) and the reference frame when it is not among them.
    """
    if n_frames <= 0:
        return ()
    start, end = clip_range(start, end, n_frames)
    roles = {i: ROLE_SELECTED if start <= i <= end else ROLE_CONTEXT
             for i in range(max(0, start - context), min(n_frames - 1, end + context) + 1)}
    if reference is not None and 0 <= reference < n_frames and reference not in roles:
        roles[reference] = ROLE_REFERENCE
    return tuple(sorted(roles.items()))


def crop_to_roi(raw: np.ndarray, gridx: tuple[int, int], gridy: tuple[int, int]) -> np.ndarray:
    """A copy of the ROI (1-based inclusive bounds) of *raw*, all channels kept."""
    rows, cols = roi_to_slice(gridx, gridy)
    return np.ascontiguousarray(raw[rows, cols]).copy()


def estimate_bytes(n_frames: int, gridx: tuple[int, int], gridy: tuple[int, int],
                   sample: np.ndarray) -> int:
    """Uncompressed size of *n_frames* ROI crops like *sample* (an upper bound)."""
    channels = 1 if sample.ndim == 2 else sample.shape[2]
    pixels = (gridx[1] - gridx[0] + 1) * (gridy[1] - gridy[0] + 1)
    return int(n_frames) * pixels * channels * sample.dtype.itemsize


def source_info(state, frame_shape: tuple[int, ...]) -> dict:
    """Kind and file name of the source: never a full path (bundles travel)."""
    if state.video_path:
        name = Path(state.video_path).name
        kind = "dat" if Path(name).suffix.lower() in DAT_EXTS else "video"
    else:
        name, kind = Path(state.folder_path).name, "folder"
    return {"kind": kind, "name": name, "n_frames": len(state.images),
            "frame_shape": [int(frame_shape[0]), int(frame_shape[1])]}


def _source_file(state, idx: int) -> str:
    return "" if state.video_path else os.path.basename(state.images[idx])


def _stem(name: str) -> str:
    return _UNSAFE_CHARS.sub("_", Path(name).stem).strip("._") or "case"


def default_file_name(state, start: int, end: int, now: datetime | None = None) -> str:
    """``<source>_hardcase_f121-180_<YYYYmmdd_HHMMSS>.zip`` (1-based frames)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    name = Path(state.video_path).name if state.video_path else Path(state.folder_path).name
    return f"{_stem(name)}_hardcase_f{start + 1}-{end + 1}_{stamp}.zip"


def _engine_version(mode: str) -> str:
    if mode != "edge":
        return ""
    from bubbletrack.model.edge import ENGINE_VERSION
    return ENGINE_VERSION


def _reference(index: int | None, n_frames: int) -> int | None:
    return int(index) if index is not None and 0 <= index < n_frames else None


def _bounds(values) -> tuple[int, int]:
    return int(values[0]), int(values[1])


def make_case(state, request: CaseRequest, frame_shape: tuple[int, ...], *,
              now: datetime | None = None) -> HardCase:
    """The case for *request* (without pixels: ``write_case`` adds them)."""
    now = now or datetime.now()
    n = len(state.images)
    start, end = clip_range(int(request.start), int(request.end), n)
    mode = state.detection_mode
    reference = _reference(request.reference_index, n) if mode == "edge" else None
    plan = plan_frames(start, end, n, reference=reference)
    frames = []
    for idx, role in plan:
        result = app_result(state, idx)
        frames.append(CaseFrame(index=idx, role=role, file=frame_file_name(idx),
                                source_file=_source_file(state, idx),
                                result=result, label=frame_label(result)))
    source = source_info(state, frame_shape)
    return HardCase(
        case_id=f"{_stem(source['name'])}_f{start + 1}-{end + 1}"
                f"_{now:%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:6]}",
        source=MappingProxyType(source),
        gridx=_bounds(state.gridx), gridy=_bounds(state.gridy),
        detection=Detection(
            mode=mode, params=MappingProxyType(session_parameters(state)),
            tracking=bool(request.tracking) and mode == "edge",
            reference_index=reference,
            reference_explicit=state.reference_frame is not None,
            max_radius=float(request.max_radius)),
        selection=(start, end),
        frames=tuple(frames),
        created=now.isoformat(timespec="seconds"),
        app_version=getattr(bubbletrack, "__version__", ""),
        engine_version=_engine_version(mode),
        reporter=request.reporter.strip(), note=request.note.strip(),
    )


def write_case(path: str | os.PathLike, case: HardCase,
               read_raw: Callable[[int], np.ndarray]) -> tuple[HardCase, int]:
    """Read, crop and store every frame of *case*; returns (case, zip bytes).

    *read_raw(idx)* returns frame *idx* as the app processes it (native
    bit depth: the image file's pixels, or the decoded video frame).
    """
    images = {f.index: crop_to_roi(read_raw(f.index), case.gridx, case.gridy)
              for f in case.frames}
    return write_bundle(path, case, images)
