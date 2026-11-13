"""Session persistence — save/load full AppState snapshots as .brt (JSON) files.

A ``.brt`` file stores all parameters and result data needed to restore a
session, **excluding** raw image data (only image paths are saved).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bubbletrack.model.suggestion import from_json as suggestions_from_json
from bubbletrack.model.suggestion import to_json as suggestion_json

logger = logging.getLogger(__name__)

SESSION_VERSION = "3.1"  # 3.1 adds circle_xy + fit_source; 3.0 files still load
MIN_SUPPORTED_VERSION = (3, 0)


def _parse_version(value) -> tuple[int, ...]:
    """'3.10' -> (3, 10).  Compared as numbers, not strings ('10.0' > '3.0')."""
    try:
        return tuple(int(part) for part in str(value).split("."))
    except (TypeError, ValueError):
        return (0,)


def _to_json_array(values) -> list | None:
    """Array -> nested lists with non-finite values as null (strict JSON)."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    out = arr.astype(object)
    out[~np.isfinite(arr)] = None
    return out.tolist()


def _from_json_array(values) -> np.ndarray:
    """Nested lists with nulls -> float array with NaN."""
    def _fix(v):
        if isinstance(v, list):
            return [_fix(x) for x in v]
        return np.nan if v is None else v
    return np.array(_fix(values), dtype=np.float64)


def session_parameters(state) -> dict:
    """The analysis parameters of *state* as plain JSON data.

    Everything a session stores besides the source paths, the shown frame
    and the results: thresholds, ROI, filters, detection mode, Edge options
    and the physical scales.  Hard-case bundles store the same dict.
    """
    return {
        "img_thr": state.img_thr,
        "gridx": list(state.gridx),
        "gridy": list(state.gridy),
        "removing_factor": state.removing_factor,
        "bubble_cross_edges": list(state.bubble_cross_edges),
        "removing_obj_radius": state.removing_obj_radius,
        "gaussian_sigma": state.gaussian_sigma,
        "clahe_clip": state.clahe_clip,
        "closing_radius": state.closing_radius,
        "opening_radius": state.opening_radius,
        "detection_mode": state.detection_mode,
        "area_max_regions": state.area_max_regions,
        "gradient_boundary": state.gradient_boundary,
        "temporal_refine": state.temporal_refine,
        "edge_definition": state.edge_definition,
        "edge_polarity": state.edge_polarity,
        "reference_frame": state.reference_frame,
        "edge_seed": list(state.edge_seed) if state.edge_seed is not None else None,
        "um2px": state.um2px,
        "fps": state.fps,
        "rmax_fit_length": state.rmax_fit_length,
        "ml_assist": bool(getattr(state, "ml_assist", False)),
        "ml_examples": examples_to_json(getattr(state, "ml_examples", None)),
    }


def examples_to_json(examples) -> dict | None:
    """ML assist examples (``MLExamples``) as plain JSON data, None stays None."""
    return None if examples is None else examples.to_dict()


def examples_from_json(data):
    """``MLExamples`` from :func:`examples_to_json` data; None when absent or unreadable
    (a damaged entry must not stop a session from loading: it is logged and dropped)."""
    if data is None:
        return None
    from bubbletrack.model.teach.examples import MLExamples  # numpy only, but lazy

    try:
        return MLExamples.from_dict(data)
    except (ValueError, TypeError) as exc:
        logger.warning("ML assist examples could not be read (dropped): %s", exc)
        return None


def save_session(filepath: str, state) -> None:
    """Serialize the current AppState to a ``.brt`` JSON file.

    Parameters
    ----------
    filepath : str
        Destination file path (should end in ``.brt``).
    state : AppState
        The application state to persist.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    data = {
        "version": SESSION_VERSION,
        "folder_path": state.folder_path,
        "video_path": state.video_path,
        "image_no": state.image_no,
        **session_parameters(state),
        "radius": _to_json_array(state.radius),
        "circle_fit_par": _to_json_array(state.circle_fit_par),
        "region_count": (
            state.region_count.tolist() if state.region_count is not None else None
        ),
        "circle_xy": (
            [None if xy is None else _to_json_array(np.asarray(xy).reshape(-1, 2))
             for xy in state.circle_xy]
            if state.circle_xy is not None else None
        ),
        "fit_source": (
            state.fit_source.tolist() if state.fit_source is not None else None
        ),
        "fit_confidence": _to_json_array(state.fit_confidence),
        "fit_metrics": (
            {k: _to_json_array(v) for k, v in state.fit_metrics.items()}
            if state.fit_metrics is not None else None
        ),
        "fit_flags": (
            [sorted(flags) for flags in state.fit_flags]
            if state.fit_flags is not None else None
        ),
        # Auto's rejected guesses (optional key: older files load without it)
        "fit_suggestion": suggestion_json(getattr(state, "fit_suggestion", None)),
        "fit_locked": (
            [bool(v) for v in state.fit_locked] if state.fit_locked is not None else None
        ),
    }
    # allow_nan=False guarantees strict JSON (MATLAB's jsondecode rejects NaN)
    Path(filepath).write_text(json.dumps(data, indent=2, allow_nan=False),
                              encoding="utf-8")
    logger.info("Session saved to %s", filepath)


def load_session(filepath: str) -> dict:
    """Deserialize a ``.brt`` session file into a plain dict.

    The caller is responsible for applying the returned dict to create or
    update an ``AppState`` instance.

    Parameters
    ----------
    filepath : str
        Path to the ``.brt`` file.

    Returns
    -------
    dict
        Parsed session data with NumPy arrays reconstructed.

    Raises
    ------
    ValueError
        If the file version is unsupported or the JSON is invalid.
    FileNotFoundError
        If the file does not exist.
    """
    raw = Path(filepath).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid session file '{filepath}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Session file '{filepath}' is not a JSON object")

    version = data.get("version", "0.0")
    if _parse_version(version) < MIN_SUPPORTED_VERSION:
        min_text = ".".join(map(str, MIN_SUPPORTED_VERSION))
        raise ValueError(
            f"Session file version {version} is too old (need >= {min_text})"
        )

    # Reconstruct NumPy arrays from JSON lists (null -> NaN)
    if data.get("radius") is not None:
        data["radius"] = _from_json_array(data["radius"])
    if data.get("circle_fit_par") is not None:
        data["circle_fit_par"] = _from_json_array(data["circle_fit_par"])
    if data.get("region_count") is not None:
        data["region_count"] = np.array(data["region_count"], dtype=int)
    if data.get("circle_xy") is not None:
        data["circle_xy"] = [
            None if xy is None else _from_json_array(xy).reshape(-1, 2)
            for xy in data["circle_xy"]
        ]
    if data.get("fit_metrics") is not None:
        data["fit_metrics"] = {k: _from_json_array(v) for k, v in data["fit_metrics"].items()}
    if data.get("fit_flags") is not None:
        data["fit_flags"] = [frozenset(flags) for flags in data["fit_flags"]]
    if data.get("fit_suggestion") is not None:
        data["fit_suggestion"] = suggestions_from_json(data["fit_suggestion"])
    if data.get("fit_locked") is not None:
        data["fit_locked"] = np.array(data["fit_locked"], dtype=bool)
    if data.get("fit_confidence") is not None:
        data["fit_confidence"] = _from_json_array(data["fit_confidence"])
    if data.get("edge_seed") is not None:
        data["edge_seed"] = tuple(float(v) for v in data["edge_seed"])
    if data.get("fit_source") is not None:
        data["fit_source"] = np.array(data["fit_source"], dtype=np.int8)
    if "ml_examples" in data:
        data["ml_examples"] = examples_from_json(data["ml_examples"])
    if "ml_assist" in data:
        data["ml_assist"] = data["ml_assist"] is True and data.get("ml_examples") is not None

    # Reconstruct tuples from lists
    if "gridx" in data and isinstance(data["gridx"], list):
        data["gridx"] = tuple(data["gridx"])
    if "gridy" in data and isinstance(data["gridy"], list):
        data["gridy"] = tuple(data["gridy"])
    if "bubble_cross_edges" in data and isinstance(data["bubble_cross_edges"], list):
        data["bubble_cross_edges"] = tuple(data["bubble_cross_edges"])

    logger.info("Session loaded from %s (version %s)", filepath, version)
    return data
