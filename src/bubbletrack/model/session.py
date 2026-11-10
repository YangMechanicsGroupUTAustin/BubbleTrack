"""Session persistence — save/load full AppState snapshots as .brt (JSON) files.

A ``.brt`` file stores all parameters and result data needed to restore a
session, **excluding** raw image data (only image paths are saved).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SESSION_VERSION = "3.0"


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
        "image_no": state.image_no,
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
        "um2px": state.um2px,
        "fps": state.fps,
        "rmax_fit_length": state.rmax_fit_length,
        "radius": state.radius.tolist() if state.radius is not None else None,
        "circle_fit_par": (
            state.circle_fit_par.tolist()
            if state.circle_fit_par is not None
            else None
        ),
        "region_count": (
            state.region_count.tolist() if state.region_count is not None else None
        ),
    }
    Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")
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
    if version < SESSION_VERSION:
        raise ValueError(
            f"Session file version {version} is too old (need >= {SESSION_VERSION})"
        )

    # Reconstruct NumPy arrays from JSON lists
    if data.get("radius") is not None:
        data["radius"] = np.array(data["radius"], dtype=np.float64)
    if data.get("circle_fit_par") is not None:
        data["circle_fit_par"] = np.array(data["circle_fit_par"], dtype=np.float64)
    if data.get("region_count") is not None:
        data["region_count"] = np.array(data["region_count"], dtype=int)

    # Reconstruct tuples from lists
    if "gridx" in data and isinstance(data["gridx"], list):
        data["gridx"] = tuple(data["gridx"])
    if "gridy" in data and isinstance(data["gridy"], list):
        data["gridy"] = tuple(data["gridy"])
    if "bubble_cross_edges" in data and isinstance(data["bubble_cross_edges"], list):
        data["bubble_cross_edges"] = tuple(data["bubble_cross_edges"])

    logger.info("Session loaded from %s (version %s)", filepath, version)
    return data
