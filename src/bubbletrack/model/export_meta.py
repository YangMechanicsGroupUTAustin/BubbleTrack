"""Provenance metadata written alongside every export.

The same plain-data dict is stored as a MATLAB struct (``.mat``), a
``Metadata`` sheet (``.xlsx``) or a JSON sidecar (``.csv``).
"""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import bubbletrack
from bubbletrack.model.long_paths import os_path

if TYPE_CHECKING:
    from bubbletrack.model.state import AppState

META_MAT_VARIABLE = "BubbleTrackMeta"
"""Name of the struct variable holding metadata in exported ``.mat`` files."""

META_SHEET_TITLE = "Metadata"
"""Title of the key/value sheet holding metadata in exported ``.xlsx`` files."""

META_SIDECAR_SUFFIX = ".meta.json"
"""Suffix of the JSON file written next to an exported CSV (``run.meta.json``)."""

# What the exported ``Radius`` measures, per detection mode.  In Edge (Auto)
# mode it follows "Bubble may be non-round": by default the least-squares
# circle through the bubble outline at the 50 % wall (engine edge-1.3.0; the
# area-equivalent radius is exported as the ``area_px`` metric), with the
# option the area-equivalent radius of the outline (the circle as the
# ``r_circle`` metric).  Per frame, the flag ``area_radius`` marks a radius
# measured by area.
_CIRCLE_FIT = "circle fit (Taubin)"
_RADIUS_DEFINITIONS: dict[tuple[str, str | None, bool], str] = {
    ("area", None, False): "area-equivalent (dark regions)",
    ("edge", "steep", False): "least-squares circle (Taubin, one outlier pass) through the bubble outline "
                              "(50 % steep edge)",
    ("edge", "outer", False): "least-squares circle (Taubin, one outlier pass) through the outer extent "
                              "outline (Edge definition B)",
    ("edge", "steep", True): "area-equivalent sqrt(A/pi) of the 50% iso-contour",
    ("edge", "outer", True): "area-equivalent sqrt(A/pi) of the outer extent contour (Edge definition B)",
}


def radius_definition(detection_mode: str, edge_definition: str = "steep",
                      non_round: bool = False) -> str:
    """Describe what the exported radius measures for *detection_mode*.

    *edge_definition* ("steep" or "outer") and *non_round* ("Bubble may be
    non-round") only matter in Edge mode.  Manual fixes are always circle
    fits; the per-frame ``Source`` says which frames those are.
    """
    if detection_mode == "edge":
        key = ("edge", edge_definition if edge_definition == "outer" else "steep", bool(non_round))
    else:
        key = (detection_mode, None, False)
    return _RADIUS_DEFINITIONS.get(key, _CIRCLE_FIT)


def _plain_value(value: object) -> object:
    """Convert a metadata value to plain Python data.

    Results are str, int, float, bool, None, lists or dicts of those;
    NumPy scalars/arrays and tuples are unpacked, anything else is ``str()``.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def plain_metadata(metadata: dict) -> dict:
    """Return a plain-data copy of *metadata* (the input is not modified)."""
    return {str(k): _plain_value(v) for k, v in metadata.items()}


def _mat_value(value: object) -> object:
    """Make a plain metadata value storable by ``savemat`` (None -> "")."""
    if value is None:
        return ""
    if isinstance(value, list):
        return [_mat_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _mat_value(v) for k, v in value.items()}
    return value


def mat_metadata(metadata: dict) -> dict:
    """Return *metadata* as a dict ``savemat`` writes as a MATLAB struct."""
    return {k: _mat_value(v) for k, v in plain_metadata(metadata).items()}


def excel_value(value: object) -> object:
    """Render a plain metadata value for a single Excel cell."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def sidecar_path(filepath: str) -> Path:
    """Return the metadata sidecar path for a CSV file (``run.meta.json``)."""
    path = Path(filepath)
    return path.with_name(path.stem + META_SIDECAR_SUFFIX)


def write_json_sidecar(filepath: str, metadata: dict) -> None:
    """Write *metadata* as JSON next to *filepath*."""
    with open(os_path(sidecar_path(filepath)), "w", encoding="utf-8") as f:
        json.dump(plain_metadata(metadata), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _app_version() -> str:
    """Return the BubbleTrack version, falling back to package metadata."""
    version = getattr(bubbletrack, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib.metadata.version("bubbletrack")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_export_metadata(
    state: AppState,
    *,
    fps: float,
    um2px: float,
    rmax_fit_length: int,
    timestamp: datetime | None = None,
    reference_index: int | None = None,
) -> dict:
    """Assemble provenance metadata for an export.

    Parameters
    ----------
    state : AppState
        Current state; supplies the source path, ROI and detection settings.
    fps, um2px, rmax_fit_length
        Post-processing values as shown in the panel (they take precedence
        over the possibly stale copies in *state*).
    timestamp : datetime or None
        Export time; defaults to the current local time.
    reference_index : int or None
        0-based reference frame the Edge engine actually used (the
        auto-selected one when ``state.reference_frame`` is None).

    Returns
    -------
    dict
        Plain data keyed like the session file.  ``bubble_cross_edges`` is
        ordered ``[Top, Right, Down, Left]``; ``gridx``/``gridy`` are the
        1-based inclusive ROI row/column bounds.  ``radius_definition``
        says what ``Radius`` measures; ``reference_frame`` is 1-based (None
        when unknown) and ``reference_frame_origin`` is "set" or "auto".
    """
    stamp = timestamp or datetime.now().astimezone()
    reference = state.reference_frame if reference_index is None else reference_index
    return plain_metadata({
        "app_version": _app_version(),
        "exported_at": stamp.isoformat(timespec="seconds"),
        "folder_path": state.folder_path,
        "video_path": state.video_path,
        "detection_mode": state.detection_mode,
        "img_thr": state.img_thr,
        "removing_factor": state.removing_factor,
        "gaussian_sigma": state.gaussian_sigma,
        "clahe_clip": state.clahe_clip,
        "opening_radius": state.opening_radius,
        "closing_radius": state.closing_radius,
        "removing_obj_radius": state.removing_obj_radius,
        "area_max_regions": state.area_max_regions,
        "gradient_boundary": state.gradient_boundary,
        "temporal_refine": state.temporal_refine,
        "bubble_cross_edges": state.bubble_cross_edges,
        "gridx": state.gridx,
        "gridy": state.gridy,
        "fps": fps,
        "um2px": um2px,
        "rmax_fit_length": rmax_fit_length,
        "radius_definition": radius_definition(state.detection_mode, state.edge_definition,
                                               getattr(state, "bubble_non_round", False)),
        "edge_definition": state.edge_definition,
        "edge_polarity": state.edge_polarity,
        "bubble_non_round": bool(getattr(state, "bubble_non_round", False)),
        "reference_frame": None if reference is None else int(reference) + 1,
        "reference_frame_origin": "auto" if state.reference_frame is None else "set",
        "edge_seed_used": state.detection_mode == "edge" and state.edge_seed is not None,
        "edge_seed": state.edge_seed,
    })
