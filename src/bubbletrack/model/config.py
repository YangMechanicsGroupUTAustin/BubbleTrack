"""Parameter persistence — save/load user-tuned parameters to ~/.bubbletrack/config.json.

On application exit the current values of PERSIST_KEYS are written to disk.
On startup the saved values are loaded back so the user picks up where they
left off without manually reconfiguring sliders every session.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

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
    "edge_definition",
    "edge_polarity",
    "area_max_regions",
    "gradient_boundary",
    "temporal_refine",
    "um2px",
    "fps",
    "rmax_fit_length",
})

DETECTION_MODES: tuple[str, ...] = ("intensity", "gradient", "area", "edge")
# What users see for each mode (the ids above stay in sessions, configs and the CLI).
DETECTION_MODE_LABELS: dict[str, str] = {
    "intensity": "Intensity mask",
    "gradient": "Gradient rings",
    "area": "Dark region area",
    "edge": "Auto (reference frame)",
}
EDGE_DEFINITIONS: tuple[str, ...] = ("steep", "outer")
EDGE_POLARITIES: tuple[str, ...] = ("dark", "bright")
EXPORT_FORMATS: tuple[str, ...] = ("mat", "csv", "xlsx")


# ------------------------------------------------------------------ #
#  Validation of loaded values
# ------------------------------------------------------------------ #
#
# config.json is user-editable and outlives app versions, so every value is
# checked before it reaches AppState: a wrong type used to crash every launch
# (the UI widgets reject it), and out-of-range values were silently kept.
# Each validator returns the cleaned value, clamped to the range the UI
# allows (ui/pretune_tab.py and ui/post_processing.py), or raises ValueError,
# in which case the key is dropped and AppState's default applies.

def _as_number(value: Any) -> float:
    """Return *value* as a finite float; numeric strings are accepted."""
    if isinstance(value, bool):  # bool is an int subclass, but never a number here
        raise ValueError("a boolean is not a number")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        number = float(value.strip())  # raises ValueError if not numeric
    else:
        raise ValueError(f"expected a number, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError("not a finite number")
    return number


def _as_whole_number(value: Any) -> int:
    number = _as_number(value)
    if not number.is_integer():
        raise ValueError("not a whole number")
    return int(number)


def _clamp(number: float, lo: float, hi: float) -> float:
    return min(max(number, lo), hi)


def _float_in(lo: float, hi: float) -> Callable[[Any], float]:
    return lambda value: _clamp(_as_number(value), lo, hi)


def _positive_float_in(lo: float, hi: float) -> Callable[[Any], float]:
    """Physical scales: 0 or negative cannot be clamped to anything meaningful."""
    def check(value: Any) -> float:
        number = _as_number(value)
        if number <= 0:
            raise ValueError("must be positive")
        return _clamp(number, lo, hi)
    return check


def _int_in(lo: int, hi: int) -> Callable[[Any], int]:
    return lambda value: int(_clamp(_as_whole_number(value), lo, hi))


def _as_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError("expected true/false")


def _as_edges(value: Any) -> tuple[bool, bool, bool, bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("expected 4 flags [top, right, down, left]")
    top, right, down, left = (_as_flag(flag) for flag in value)
    return top, right, down, left


def _as_detection_mode(value: Any) -> str:
    mode = value.strip().lower() if isinstance(value, str) else None
    if mode not in DETECTION_MODES:
        raise ValueError(f"expected one of {', '.join(DETECTION_MODES)}")
    return mode


def _one_of(choices: tuple[str, ...]) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        text = value.strip().lower() if isinstance(value, str) else None
        if text not in choices:
            raise ValueError(f"expected one of {', '.join(choices)}")
        return text
    return check


def _as_gradient_boundary(value: Any) -> float:
    """Ring fraction in [0, 1]; values > 1 are a legacy ring stage 1..3.

    Mirrors how the Gradient Ring slider reads the value, so the stored
    number and the slider position agree.
    """
    number = _as_number(value)
    if number > 1.0:
        number = (number - 1.0) / 2.0
    return _clamp(number, 0.0, 1.0)


_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "img_thr": _float_in(0.0, 1.0),                   # Threshold slider 0-100 / 100
    "removing_factor": _int_in(0, 100),               # Remove slider
    "bubble_cross_edges": _as_edges,
    "gaussian_sigma": _float_in(0.0, 20.0),           # Gauss Blur, 0 = off
    "clahe_clip": _float_in(0.0, 40.0),               # CLAHE, 0 = off
    "closing_radius": _int_in(0, 30),                 # Morph Close, 0 = off
    "opening_radius": _int_in(0, 20),                 # Morph Open, 0 = off
    "detection_mode": _as_detection_mode,
    "edge_definition": _one_of(EDGE_DEFINITIONS),
    "edge_polarity": _one_of(EDGE_POLARITIES),
    "area_max_regions": _int_in(1, 20),               # Max regions spin box
    "gradient_boundary": _as_gradient_boundary,       # Gradient Ring 1-3
    "temporal_refine": _as_flag,
    "um2px": _positive_float_in(0.001, 9999.0),       # um/px spin box
    "fps": _positive_float_in(0.001, 99_999e6),       # FPS spin box (Hz..M)
    "rmax_fit_length": _int_in(3, 999),               # Rmax Fit spin box
}


def _unchanged(raw: Any, clean: Any) -> bool:
    """True if *clean* only changes the container/number type of *raw*."""
    return (tuple(raw) if isinstance(raw, list) else raw) == clean


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the valid, cleaned PERSIST_KEYS entries of a raw config dict.

    Unknown keys are dropped silently; invalid values are dropped and
    adjusted values are kept, both with a logged warning.  *raw* is not
    modified.
    """
    result: dict[str, Any] = {}
    for key, value in raw.items():
        validator = _VALIDATORS.get(key)
        if validator is None:
            continue
        try:
            clean = validator(value)
        except ValueError as exc:
            logger.warning(
                "config: ignoring %s=%r (%s); using the default", key, value, exc,
            )
            continue
        if not _unchanged(value, clean):
            logger.warning("config: %s=%r adjusted to %r", key, value, clean)
        result[key] = clean
    return result


def _write_config(data: dict[str, Any]) -> None:
    """Write *data* as CONFIG_FILE, creating the directory; raises OSError."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_config(state: object) -> None:
    """Persist the PERSIST_KEYS fields of *state* to CONFIG_FILE.

    The values are merged into the existing file, so other keys (the
    ``onboarding_done`` flag, keys of newer versions) are kept; rewriting
    only the persisted keys used to bring the welcome dialog back on every
    launch.  Creates the config directory if it does not exist.  Silently
    logs and returns on any I/O or serialisation error so the application
    can always shut down cleanly.
    """
    try:
        data: dict[str, Any] = dict(_load_raw_config())
        for key in sorted(PERSIST_KEYS):
            value = getattr(state, key, None)
            # Convert tuples to lists for JSON round-trip
            if isinstance(value, tuple):
                value = list(value)
            data[key] = value
        _write_config(data)
        logger.info("Config saved to %s", CONFIG_FILE)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save config: %s", exc)


def load_config() -> dict[str, Any]:
    """Load persisted parameters from CONFIG_FILE.

    Returns an empty dict when the file is missing, unreadable, or
    contains invalid JSON — the caller should fall back to defaults.
    Otherwise only recognised keys with valid values are returned (see
    :func:`validate_config`), so the result can always be applied to
    AppState safely.
    """
    if not CONFIG_FILE.exists():
        return {}
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:  # ValueError covers JSON and UTF-8 errors
        logger.warning("Failed to load config: %s", exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("Config file root is not a JSON object; ignoring")
        return {}
    return validate_config(raw)


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
    except (ValueError, OSError):  # ValueError covers JSON and UTF-8 errors
        return {}


def is_onboarding_done() -> bool:
    """Return True if the user has dismissed the welcome dialog."""
    return bool(_load_raw_config().get("onboarding_done", False))


def set_onboarding_done() -> None:
    """Persist the onboarding_done flag so the welcome dialog is not shown again."""
    config = _load_raw_config()
    config["onboarding_done"] = True
    try:
        _write_config(config)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save onboarding flag: %s", exc)


# ------------------------------------------------------------------ #
#  App bookkeeping (meta-config, not persisted AppState parameters)
# ------------------------------------------------------------------ #
#
# Validated like the parameters above, but kept apart: load_config()
# feeds AppState, which has no such fields.

_MAX_PATH_CHARS = 4096


def _as_folder(value: Any) -> str:
    """An absolute folder path; whether it still exists is the caller's concern."""
    text = value.strip() if isinstance(value, str) else ""
    if not text or "\x00" in text or len(text) > _MAX_PATH_CHARS or not os.path.isabs(text):
        raise ValueError("expected an absolute folder path")
    return text


_MAX_NAME_CHARS = 80


def _as_name(value: Any) -> str:
    """A person's name on one line ('' = none), e.g. who reports a hard case."""
    if not isinstance(value, str):
        raise ValueError("expected text")
    text = " ".join(value.split())
    if len(text) > _MAX_NAME_CHARS or any(ord(ch) < 32 for ch in text):
        raise ValueError(f"expected a name of at most {_MAX_NAME_CHARS} characters")
    return text


def _as_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected true or false")
    return value


def _as_timestamp(value: Any) -> float:
    """Seconds since the epoch (``time.time()``)."""
    number = _as_number(value)
    if number < 0:
        raise ValueError("must not be negative")
    return number


_META_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "last_export_dir": _as_folder,                  # export dialogs start here
    "last_export_format": _one_of(EXPORT_FORMATS),  # pre-selected Format combo
    "last_clean_exit": _as_timestamp,               # newer autosave => offer restore
    "hard_case_reporter": _as_name,                 # "Your name" in Export Hard Case
    "show_classic_methods": _as_bool,               # Settings: Intensity/Gradient/Area + their controls
    "show_all_view_modes": _as_bool,                # Settings: Image only / Overlay / Wipe views
}

# User settings (Settings dialog).  Off by default: the main window shows only
# the Auto method and the Image + Mask view.  Classic methods will be retired.
UI_SETTINGS: dict[str, bool] = {
    "show_classic_methods": False,
    "show_all_view_modes": False,
}


def ui_setting(name: str) -> bool:
    """The saved value of a Settings switch, else its default."""
    if name not in UI_SETTINGS:
        raise ValueError(f"unknown UI setting {name!r}")
    value = get_meta(name)
    return UI_SETTINGS[name] if value is None else bool(value)


def set_ui_setting(name: str, value: bool) -> bool:
    """Save a Settings switch; returns False if the config could not be written."""
    if name not in UI_SETTINGS:
        raise ValueError(f"unknown UI setting {name!r}")
    return set_meta(**{name: bool(value)})


def _meta_validator(key: str) -> Callable[[Any], Any]:
    validator = _META_VALIDATORS.get(key)
    if validator is None:
        raise ValueError(f"unknown config meta key {key!r}")
    return validator


def get_meta(key: str) -> Any | None:
    """Return the validated value of a meta key, or None if unset or invalid.

    Raises ValueError for a key that is not a known meta key.
    """
    validator = _meta_validator(key)
    raw = _load_raw_config()
    if key not in raw:
        return None
    try:
        return validator(raw[key])
    except ValueError as exc:
        logger.warning("config: ignoring %s=%r (%s)", key, raw[key], exc)
        return None


def set_meta(**values: Any) -> bool:
    """Validate and merge meta keys into CONFIG_FILE; other keys are kept.

    Returns False (and logs) if the file cannot be written.  Raises
    ValueError for an unknown key or an invalid value, before writing.
    """
    clean: dict[str, Any] = {}
    for key, value in values.items():
        try:
            clean[key] = _meta_validator(key)(value)
        except ValueError as exc:
            raise ValueError(f"{key}={value!r}: {exc}") from exc
    try:
        _write_config({**_load_raw_config(), **clean})
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save %s: %s", ", ".join(clean), exc)
        return False
    return True


def record_clean_exit(when: float | None = None) -> bool:
    """Remember that the app closed normally (default: now)."""
    return set_meta(last_clean_exit=time.time() if when is None else when)
