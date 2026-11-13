"""What a hard case holds, and how it is stored in ``case.json``.

A hard case is a short stretch of one recording that the app got wrong,
sent in by a user: the frames (cropped to the ROI), the parameters the app
ran with, what the app showed on every frame and, where the user fixed a
frame, the right answer (its *label*).

Conventions (as in sessions): frame ``index`` is 0-based (the app shows
``index + 1``); radii, centres and ROI bounds are in pixels of the original,
uncropped frame, 1-based like everywhere in the app.  The cropped frame
pixel ``(i, j)`` (0-based) is original pixel ``(i + offset[0], j + offset[1])``.

No Qt here: bundles are read by the headless regression runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

FORMAT = "bubbletrack-hard-case"
FORMAT_VERSION = 1
CASE_JSON = "case.json"
FRAMES_DIR = "frames"

ROLE_SELECTED = "selected"    # the frames the user reported
ROLE_CONTEXT = "context"      # neighbours, for tracking and for the eye
ROLE_REFERENCE = "reference"  # Auto mode's bubble-free reference frame only
ROLES = (ROLE_SELECTED, ROLE_CONTEXT, ROLE_REFERENCE)

LABEL_BUBBLE = "bubble"          # the user's circle is the right answer
LABEL_NO_BUBBLE = "no_bubble"    # right answer: no result on this frame
LABEL_UNLABELLED = "unlabelled"  # nobody checked this frame
LABEL_STATUSES = (LABEL_BUBBLE, LABEL_NO_BUBBLE, LABEL_UNLABELLED)

BY_MANUAL = "manual"      # a manual fit (clicked edge points)
BY_ACCEPTED = "accepted"  # accepted / locked in review or on the chart


class CaseFormatError(ValueError):
    """The data is not a hard case this version can read."""


@dataclass(frozen=True)
class AppResult:
    """What the app showed on a frame when the case was exported."""

    radius: float | None = None  # None: no result (rejected, deleted, not fitted)
    center: tuple[float, float] | None = None  # (row, col)
    confidence: float | None = None
    flags: tuple[str, ...] = ()
    source: str = "none"  # FitSource name: auto, manual, refit, neighbor, tracked, none
    locked: bool = False


@dataclass(frozen=True)
class Label:
    """The verified answer for a frame, if there is one."""

    status: str = LABEL_UNLABELLED
    radius: float | None = None
    center: tuple[float, float] | None = None
    by: str = ""

    @property
    def is_labelled(self) -> bool:
        return self.status != LABEL_UNLABELLED


@dataclass(frozen=True)
class CaseFrame:
    index: int  # 0-based frame of the recording
    role: str
    file: str  # image path inside the bundle
    source_file: str = ""  # the original image's file name ('' for videos)
    result: AppResult = AppResult()
    label: Label = Label()


@dataclass(frozen=True)
class Detection:
    """How the app measured: mode, parameters and Auto-mode extras."""

    mode: str
    params: Mapping[str, Any]  # session_parameters() of the app state
    tracking: bool = False  # Auto mode: "Track across frames after Fit All"
    reference_index: int | None = None  # Auto mode: the reference frame used
    reference_explicit: bool = False  # set by the user (else auto-selected)
    max_radius: float = math.inf  # largest plausible radius (the frame's longer side)


@dataclass(frozen=True)
class HardCase:
    case_id: str
    source: Mapping[str, Any]  # kind, name (file name only), n_frames, frame_shape
    gridx: tuple[int, int]
    gridy: tuple[int, int]
    detection: Detection
    selection: tuple[int, int]  # reported frames, 0-based inclusive
    frames: tuple[CaseFrame, ...]
    created: str = ""
    app_version: str = ""
    engine_version: str = ""
    reporter: str = ""
    note: str = ""

    @property
    def offset(self) -> tuple[int, int]:
        """Original (row, col) of crop pixel (0, 0), minus 1 (add to crop coordinates)."""
        return self.gridx[0] - 1, self.gridy[0] - 1

    @property
    def name(self) -> str:
        return str(self.source.get("name", ""))

    def frame(self, index: int) -> CaseFrame | None:
        return next((f for f in self.frames if f.index == index), None)

    def measured_frames(self) -> tuple[CaseFrame, ...]:
        """Selected and context frames in frame order (not the reference-only one)."""
        return tuple(sorted((f for f in self.frames if f.role != ROLE_REFERENCE),
                            key=lambda f: f.index))


# -- to JSON ----------------------------------------------------------------- #

def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _pair(values) -> list[float] | None:
    return None if values is None else [float(values[0]), float(values[1])]


def _result_json(result: AppResult) -> dict:
    return {"radius": _finite_or_none(result.radius), "center": _pair(result.center),
            "confidence": _finite_or_none(result.confidence), "flags": list(result.flags),
            "source": result.source, "locked": bool(result.locked)}


def _label_json(label: Label) -> dict:
    out: dict[str, Any] = {"status": label.status}
    if label.status == LABEL_BUBBLE:
        out.update(radius=float(label.radius), center=_pair(label.center))
    if label.by:
        out["by"] = label.by
    return out


def _frame_json(frame: CaseFrame) -> dict:
    return {"index": frame.index, "number": frame.index + 1, "role": frame.role,
            "file": frame.file, "source_file": frame.source_file,
            "result": _result_json(frame.result), "label": _label_json(frame.label)}


def case_to_json(case: HardCase) -> dict:
    """The ``case.json`` content (strict JSON: no NaN or infinity)."""
    det = case.detection
    return {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "case_id": case.case_id, "created": case.created,
        "app_version": case.app_version, "engine_version": case.engine_version,
        "reporter": case.reporter, "note": case.note,
        "source": dict(case.source),
        "roi": {"gridx": list(case.gridx), "gridy": list(case.gridy),
                "offset": list(case.offset)},
        "detection": {"mode": det.mode, "params": dict(det.params), "tracking": det.tracking,
                      "reference_index": det.reference_index,
                      "reference_explicit": det.reference_explicit,
                      "max_radius": _finite_or_none(det.max_radius)},
        "selection": {"start": case.selection[0], "end": case.selection[1]},
        "frames": [_frame_json(f) for f in case.frames],
    }


# -- from JSON (validated: bundles come from other machines) ------------------ #

def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CaseFormatError(message)


def _int(value: Any, what: str, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             f"{what} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _number(value: Any, what: str) -> float | None:
    if value is None:
        return None
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), f"{what} must be a finite number, got {value!r}")
    return float(value)


def _center(value: Any, what: str) -> tuple[float, float] | None:
    if value is None:
        return None
    _require(isinstance(value, list) and len(value) == 2, f"{what} must be [row, col]")
    row, col = (_number(v, what) for v in value)
    return None if row is None or col is None else (row, col)


def _text(value: Any, what: str) -> str:
    _require(value is None or isinstance(value, str), f"{what} must be text")
    return value or ""


def _bounds(value: Any, what: str) -> tuple[int, int]:
    _require(isinstance(value, list) and len(value) == 2, f"{what} must be [first, last]")
    lo, hi = _int(value[0], what, 1), _int(value[1], what, 1)
    _require(lo <= hi, f"{what} must satisfy first <= last, got {value}")
    return lo, hi


def _result(data: Any) -> AppResult:
    _require(isinstance(data, dict), "frame result must be an object")
    radius = _number(data.get("radius"), "result radius")
    flags = data.get("flags") or []
    _require(isinstance(flags, list) and all(isinstance(f, str) for f in flags),
             "result flags must be a list of text")
    return AppResult(radius=radius if radius is not None and radius > 0 else None,
                     center=_center(data.get("center"), "result centre"),
                     confidence=_number(data.get("confidence"), "result confidence"),
                     flags=tuple(flags), source=_text(data.get("source"), "result source") or "none",
                     locked=bool(data.get("locked", False)))


def _label(data: Any) -> Label:
    _require(isinstance(data, dict), "frame label must be an object")
    status = data.get("status", LABEL_UNLABELLED)
    _require(status in LABEL_STATUSES, f"unknown label status {status!r}")
    by = _text(data.get("by"), "label by")
    if status != LABEL_BUBBLE:
        return Label(status=status, by=by)
    radius = _number(data.get("radius"), "label radius")
    _require(radius is not None and radius > 0, "a bubble label needs a radius > 0")
    return Label(status, radius, _center(data.get("center"), "label centre"), by)


def _frame(data: Any) -> CaseFrame:
    _require(isinstance(data, dict), "every frame must be an object")
    role = data.get("role")
    _require(role in ROLES, f"unknown frame role {role!r}")
    file = data.get("file")
    _require(isinstance(file, str) and bool(file), "every frame needs a file name")
    return CaseFrame(index=_int(data.get("index"), "frame index"), role=role, file=file,
                     source_file=_text(data.get("source_file"), "source file"),
                     result=_result(data.get("result", {})), label=_label(data.get("label", {})))


def _detection(data: Any) -> Detection:
    _require(isinstance(data, dict), "detection must be an object")
    from bubbletrack.model.config import DETECTION_MODES

    mode = data.get("mode")
    _require(mode in DETECTION_MODES, f"unknown detection mode {mode!r}")
    params = data.get("params", {})
    _require(isinstance(params, dict), "detection params must be an object")
    ref = data.get("reference_index")
    max_radius = _number(data.get("max_radius"), "max_radius")
    return Detection(mode=mode, params=MappingProxyType(dict(params)),
                     tracking=bool(data.get("tracking", False)),
                     reference_index=None if ref is None else _int(ref, "reference_index"),
                     reference_explicit=bool(data.get("reference_explicit", False)),
                     max_radius=math.inf if max_radius is None else max_radius)


def case_from_json(data: Any) -> HardCase:
    """Parse and validate ``case.json`` content.

    Raises
    ------
    CaseFormatError
        For anything that is not a hard case of a supported version.
    """
    _require(isinstance(data, dict), "case.json must hold an object")
    _require(data.get("format") == FORMAT, "not a BubbleTrack hard case")
    version = data.get("format_version")
    _require(isinstance(version, int) and 1 <= version <= FORMAT_VERSION,
             f"unsupported hard-case version {version!r} (this app reads <= {FORMAT_VERSION})")
    roi, selection = data.get("roi") or {}, data.get("selection") or {}
    _require(isinstance(roi, dict) and isinstance(selection, dict), "roi/selection must be objects")
    frames = data.get("frames")
    _require(isinstance(frames, list) and bool(frames), "a hard case needs frames")
    source = data.get("source") or {}
    _require(isinstance(source, dict), "source must be an object")
    start, end = _int(selection.get("start"), "selection start"), _int(selection.get("end"), "selection end")
    _require(start <= end, "selection start must not be after its end")
    parsed = tuple(_frame(f) for f in frames)
    _require(len({f.index for f in parsed}) == len(parsed), "a frame is listed twice")
    _require(len({f.file for f in parsed}) == len(parsed), "two frames share an image file")
    return HardCase(
        case_id=_text(data.get("case_id"), "case_id"),
        source=MappingProxyType(dict(source)),
        gridx=_bounds(roi.get("gridx"), "roi gridx"), gridy=_bounds(roi.get("gridy"), "roi gridy"),
        detection=_detection(data.get("detection")),
        selection=(start, end),
        frames=parsed,
        created=_text(data.get("created"), "created"),
        app_version=_text(data.get("app_version"), "app_version"),
        engine_version=_text(data.get("engine_version"), "engine_version"),
        reporter=_text(data.get("reporter"), "reporter"), note=_text(data.get("note"), "note"),
    )
