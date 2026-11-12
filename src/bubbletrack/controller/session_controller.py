"""SessionController — save and fully restore analysis sessions (.brt).

Restoring is done in three steps so nothing the user saved is lost:
1. load the image folder (this resets the frame-dependent views),
2. re-apply the saved parameters (ROI, thresholds, frame position, ...),
3. put the saved results back if the frame count still matches.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bubbletrack.controller.base import BaseController
from bubbletrack.event_bus import EventBus
from bubbletrack.model import config as config_module
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.session import load_session, save_session
from bubbletrack.model.state import AppState, update_state

logger = logging.getLogger(__name__)

RESULT_KEYS = ("radius", "circle_fit_par", "circle_xy", "region_count", "fit_source",
               "fit_confidence", "fit_metrics", "fit_flags", "fit_locked")
_NOT_PARAMS = {"version", "folder_path", "video_path", "images", *RESULT_KEYS}
_STATE_FIELDS = {f.name for f in dataclasses.fields(AppState)}
LAST_SESSION_NAME = "last_session.brt"

_OK, _WARN, _ERR = "#10b981", "#f59e0b", "#ef4444"


def split_session(data: dict) -> tuple[str, dict, dict]:
    """Split a loaded session dict into (folder, parameters, results)."""
    folder = data.get("folder_path") or ""
    results = {k: data[k] for k in RESULT_KEYS if data.get(k) is not None}
    params = {k: v for k, v in data.items()
              if k in _STATE_FIELDS and k not in _NOT_PARAMS}
    return folder, params, results


def sanitize_params(params: dict, n_frames: int,
                    full_roi: tuple[tuple[int, int], tuple[int, int]]) -> tuple[dict, list[str]]:
    """Drop parameters that no longer fit the loaded images; return warnings."""
    params = dict(params)
    warnings: list[str] = []
    (_, h), (_, w) = full_roi
    gx, gy = params.get("gridx"), params.get("gridy")
    if gx is not None or gy is not None:
        ok = (gx is not None and gy is not None
              and 1 <= gx[0] <= gx[1] <= h and 1 <= gy[0] <= gy[1] <= w)
        if not ok:
            params.pop("gridx", None)
            params.pop("gridy", None)
            warnings.append("saved ROI does not fit the images (ROI reset)")
    if "image_no" in params:
        params["image_no"] = int(min(max(params["image_no"], 0), max(n_frames - 1, 0)))
    return params, warnings


def restore_results(state: AppState, results: dict) -> str | None:
    """Copy saved result arrays into *state*; return a warning if impossible."""
    radius = results.get("radius")
    if radius is None or state.radius is None:
        return None
    n = len(state.radius)
    if len(radius) != n:
        return (f"frame count changed ({len(radius)} saved, {n} now); "
                "results were not restored")
    state.radius[:] = radius
    centers = results.get("circle_fit_par")
    if centers is not None and np.shape(centers) == (n, 2):
        state.circle_fit_par[:] = centers
    counts = results.get("region_count")
    if counts is not None and len(counts) == n and state.region_count is not None:
        state.region_count[:] = counts
    edges = results.get("circle_xy")
    if edges is not None and len(edges) == n:
        for i, xy in enumerate(edges):
            state.circle_xy[i] = xy
    metrics = results.get("fit_metrics")
    if metrics is not None and state.fit_metrics is not None:
        for key, values in metrics.items():
            if key in state.fit_metrics and len(values) == n:
                state.fit_metrics[key][:] = values
    flags = results.get("fit_flags")
    if flags is not None and len(flags) == n and state.fit_flags is not None:
        state.fit_flags[:] = list(flags)
    locked = results.get("fit_locked")
    if locked is not None and len(locked) == n and state.fit_locked is not None:
        state.fit_locked[:] = locked
    confidence = results.get("fit_confidence")
    if confidence is not None and len(confidence) == n and state.fit_confidence is not None:
        state.fit_confidence[:] = confidence
    sources = results.get("fit_source")
    if state.fit_source is not None:
        if sources is not None and len(sources) == n:
            state.fit_source[:] = sources
        else:  # sessions saved before provenance existed
            state.fit_source[:] = np.where(radius > 0, FitSource.AUTO, FitSource.NONE)
    return None


class SessionController(BaseController):
    """Save/load sessions and keep a last-session recovery file."""

    def __init__(
        self,
        bus: EventBus,
        get_state,
        set_state,
        window,
        file_ctrl,
        *,
        is_busy: Callable[[], bool],
        locate_folder: Callable[[str], str],
        after_restore: Callable[[], None],
    ) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._file_ctrl = file_ctrl
        self._is_busy = is_busy
        self._locate_folder = locate_folder
        self._after_restore = after_restore

    def save(self, path: str) -> bool:
        try:
            save_session(path, self.state)
        except Exception as exc:
            logger.error("Session save failed: %s", exc)
            self.w.header.set_status(f"Save failed: {exc}", _WARN)
            return False
        self.w.header.set_status(f"Session saved: {os.path.basename(path)}", _OK)
        return True

    def load(self, path: str) -> bool:
        if self._is_busy():
            self.w.header.set_status("Stop the running fit before loading a session", _WARN)
            return False
        try:
            data = load_session(path)
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            logger.error("Session load failed: %s", exc)
            self.w.header.set_status(f"Load failed: {exc}", _ERR)
            return False

        folder, params, results = split_session(data)
        video = data.get("video_path") or ""
        if video:
            return self._restore(path, params, results, video=video)
        if folder and not os.path.isdir(folder):
            located = self._locate_folder(folder)
            if not located:
                self.w.header.set_status(f"Image folder not found: {folder}", _ERR)
                return False
            folder = located
        return self._restore(path, params, results, folder=folder)

    def _restore(self, path: str, params: dict, results: dict, *,
                 folder: str = "", video: str = "") -> bool:
        if video:
            if not os.path.isfile(video):
                self.w.header.set_status(f"Video not found: {video}", _ERR)
                return False
            self._file_ctrl.on_video_selected(video)
            if self.state.video_path != video:
                return False  # opening failed; FileController reported why
        elif folder:
            self._file_ctrl.on_folder_selected(folder)
            if self.state.folder_path != folder:
                return False  # loading failed; FileController reported why

        full_roi = (self.state.gridx, self.state.gridy)
        params, warnings = sanitize_params(params, len(self.state.images), full_roi)
        try:
            self.state = update_state(self.state, **params)
        except TypeError as exc:
            logger.error("Session restore failed: %s", exc)
            self.w.header.set_status(f"Restore failed: {exc}", _ERR)
            return False

        warning = restore_results(self.state, results)
        if warning:
            warnings.append(warning)
        self._after_restore()

        name = os.path.basename(path)
        if warnings:
            self.w.header.set_status(f"Session loaded: {name} — " + "; ".join(warnings), _WARN)
        else:
            self.w.header.set_status(f"Session loaded: {name}", _OK)
        return True

    def save_last_session(self) -> str | None:
        """Write ~/.bubbletrack/recovery/last_session.brt if there are results."""
        radius = self.state.radius
        if radius is None or not np.any(np.asarray(radius) > 0):
            return None
        path = Path(config_module.CONFIG_DIR) / "recovery" / LAST_SESSION_NAME
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_session(str(path), self.state)
        except Exception as exc:
            logger.warning("Could not write %s: %s", path, exc)
            return None
        logger.info("Last session saved to %s", path)
        return str(path)
