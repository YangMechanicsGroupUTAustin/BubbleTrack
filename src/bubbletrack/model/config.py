"""Parameter persistence — save/load user-tuned parameters to ~/.bubbletrack/config.json.

On application exit the current values of PERSIST_KEYS are written to disk.
On startup the saved values are loaded back so the user picks up where they
left off without manually reconfiguring sliders every session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR: Path = Path.home() / ".bubbletrack"
CONFIG_FILE: Path = CONFIG_DIR / "config.json"

# Fields from AppState that should survive across sessions.
PERSIST_KEYS: frozenset[str] = frozenset({
    "img_thr",
    "removing_factor",
    "bubble_cross_edges",
    "gaussian_sigma",
    "clahe_clip",
    "closing_radius",
    "opening_radius",
    "detection_mode",
    "gradient_boundary",
    "temporal_refine",
    "um2px",
    "fps",
    "rmax_fit_length",
})


def save_config(state: object) -> None:
    """Persist the PERSIST_KEYS fields of *state* to CONFIG_FILE.

    Creates the config directory if it does not exist.  Silently logs
    and returns on any I/O or serialisation error so the application
    can always shut down cleanly.
    """
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        for key in sorted(PERSIST_KEYS):
            value = getattr(state, key, None)
            # Convert tuples to lists for JSON round-trip
            if isinstance(value, tuple):
                value = list(value)
            data[key] = value
        CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Config saved to %s", CONFIG_FILE)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save config: %s", exc)


def load_config() -> dict[str, Any]:
    """Load persisted parameters from CONFIG_FILE.

    Returns an empty dict when the file is missing, unreadable, or
    contains invalid JSON — the caller should fall back to defaults.
    """
    if not CONFIG_FILE.exists():
        return {}
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("Config file root is not a JSON object; ignoring")
            return {}
        # Only return keys we recognise to avoid injecting garbage into state.
        result: dict[str, Any] = {}
        for key, value in raw.items():
            if key in PERSIST_KEYS:
                # Restore lists back to tuples for tuple-typed fields
                if key == "bubble_cross_edges" and isinstance(value, list):
                    value = tuple(value)
                result[key] = value
        return result
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load config: %s", exc)
        return {}


# ------------------------------------------------------------------ #
#  Onboarding flag (meta-config, not a persisted AppState parameter)
# ------------------------------------------------------------------ #

def _load_raw_config() -> dict[str, Any]:
    """Load the full raw config dict (including meta-keys like onboarding_done)."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def is_onboarding_done() -> bool:
    """Return True if the user has dismissed the welcome dialog."""
    return bool(_load_raw_config().get("onboarding_done", False))


def set_onboarding_done() -> None:
    """Persist the onboarding_done flag so the welcome dialog is not shown again."""
    config = _load_raw_config()
    config["onboarding_done"] = True
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save onboarding flag: %s", exc)
