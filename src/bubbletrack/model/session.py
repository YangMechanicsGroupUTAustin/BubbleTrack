"""Session persistence — save/load full AppState snapshots as .brt (JSON) files.

A ``.brt`` file stores all parameters and result data needed to restore a
session, **excluding** raw image data (only image paths are saved).
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata

import numpy as np

from bubbletrack.model.atomic_write import write_text_atomic
from bubbletrack.model.long_paths import os_path
from bubbletrack.model.provenance import AREA_RADIUS_FLAG, FitSource
from bubbletrack.model.suggestion import from_json as suggestions_from_json
from bubbletrack.model.suggestion import to_json as suggestion_json

logger = logging.getLogger(__name__)

# 3.1 adds circle_xy + fit_source, 3.2 bubble_non_round (the radius definition),
# 3.3 physical_set + image_names; 3.0 files still load
SESSION_VERSION = "3.3"
PHYSICAL_KEYS = frozenset({"fps", "scale"})
MIN_SUPPORTED_VERSION = (3, 0)
_ENGINE_SOURCES = frozenset({FitSource.AUTO, FitSource.REFIT, FitSource.NEIGHBOR, FitSource.TRACKED,
                             FitSource.GUIDED, FitSource.ML})


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
        "bubble_non_round": bool(getattr(state, "bubble_non_round", False)),
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


def image_names(state) -> list[str] | None:
    """The file names of a folder's frames (None for a video: its path names it)."""
    if state.video_path or not state.images:
        return None
    return [os.path.basename(p) for p in state.images]


def images_differ(saved: list | None, images) -> str | None:
    """Why results saved for the files *saved* do not belong to *images*, or None.

    Only same-length lists are compared (a count change is reported elsewhere).
    Names are compared in NFC: macOS and Windows spell accented names differently.
    """
    if not saved or not images or len(saved) != len(images):
        return None
    current = [os.path.basename(p) for p in images]
    norm = lambda name: unicodedata.normalize("NFC", str(name))  # noqa: E731
    differ = sum(norm(a) != norm(b) for a, b in zip(saved, current))
    if not differ:
        return None
    return (f"{differ} of {len(current)} image names differ from the measured ones; "
            "results were not restored")


def session_json(data: dict) -> str:
    """*data* as strict JSON with one top-level key per line and compact values.

    Indenting every number of the per-frame arrays (outlines: hundreds of points
    per frame) almost doubled the file: 258 MB instead of 142 MB at 10k frames,
    rewritten by every autosave into a folder that is often synced.  The keys stay
    readable; any JSON reader (MATLAB's jsondecode) loads it as before.
    ``allow_nan=False``: strict JSON, as jsondecode rejects NaN.
    """
    lines = [f"  {json.dumps(key)}: {json.dumps(value, separators=(',', ':'), allow_nan=False)}"
             for key, value in data.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


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
        "image_names": image_names(state),
        **session_parameters(state),
        "physical_set": sorted(state.physical_set),
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
    # Atomic: an interrupted save must never destroy the previous session file
    write_text_atomic(filepath, session_json(data))
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
    with open(os_path(filepath), encoding="utf-8-sig") as fh:  # a BOM (Notepad) is fine
        raw = fh.read()
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

    if "bubble_non_round" not in data:
        _legacy_radius_definition(data)
    if "physical_set" in data:
        values = data["physical_set"] if isinstance(data["physical_set"], list) else []
        data["physical_set"] = frozenset(v for v in values if v in PHYSICAL_KEYS)
    else:  # before 3.3 the values were not marked; such sessions always loaded as set
        data["physical_set"] = PHYSICAL_KEYS

    # Reconstruct tuples from lists
    if "gridx" in data and isinstance(data["gridx"], list):
        data["gridx"] = tuple(data["gridx"])
    if "gridy" in data and isinstance(data["gridy"], list):
        data["gridy"] = tuple(data["gridy"])
    if "bubble_cross_edges" in data and isinstance(data["bubble_cross_edges"], list):
        data["bubble_cross_edges"] = tuple(data["bubble_cross_edges"])

    logger.info("Session loaded from %s (version %s)", filepath, version)
    return data


def _legacy_radius_definition(data: dict) -> None:
    """A session saved before the radius definition was a setting (< 3.2): Auto measured
    the area-equivalent radius then.  It loads with "Bubble may be non-round" on and its
    engine results marked so (``area_radius``), as that mode would store them; frames whose
    radius was the circle fit (``circle_fallback``) and manual fits stay circles."""
    if data.get("detection_mode") != "edge":
        return
    data["bubble_non_round"] = True
    flags, radius = data.get("fit_flags"), data.get("radius")
    if flags is None or radius is None or len(flags) != len(radius):
        return
    sources = data.get("fit_source")
    if sources is None or len(sources) != len(radius):
        sources = [FitSource.AUTO] * len(radius)
    data["fit_flags"] = [
        frozenset(f) | {AREA_RADIUS_FLAG}
        if radius[i] > 0 and int(sources[i]) in _ENGINE_SOURCES and "circle_fallback" not in f
        else frozenset(f)
        for i, f in enumerate(flags)
    ]
