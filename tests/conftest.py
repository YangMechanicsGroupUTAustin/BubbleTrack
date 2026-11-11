"""Shared fixtures for GUI-level tests.

GUI tests drive the real MainWindow + AppController offscreen.  The user
config is redirected to a temporary directory so tests never read or write
the real ``~/.bubbletrack/config.json``.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import cv2
import numpy as np
import pytest

from _golden_cases import SceneSpec, render_scene


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    from bubbletrack.model import config

    cfg_dir = tmp_path / "bubbletrack_home"
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", cfg_dir / "config.json")
    return cfg_dir


@pytest.fixture
def gui(qapp, isolated_config):
    """A real main window + controller with an isolated config."""
    from bubbletrack.controller.controller import AppController
    from bubbletrack.ui.main_window import MainWindow

    window = MainWindow()
    controller = AppController(window)
    yield window, controller
    worker = controller.auto_ctrl._worker
    if worker is not None and worker.isRunning():
        worker.request_stop()
        worker.wait(10_000)
    window.close()
    qapp.processEvents()


def write_sequence(folder: Path, radii, *, shape=(250, 400), center=(125.0, 200.0),
                   prefix: str = "frame") -> Path:
    """Write a synthetic bubble sequence (one TIFF per radius) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(radii):
        frame = render_scene(SceneSpec(radius=float(r), center=center, shape=shape, seed=i))
        if not cv2.imwrite(str(folder / f"{prefix}_{i + 1:03d}.tiff"), frame):
            raise OSError(f"could not write frame {i} to {folder}")
    return folder


@pytest.fixture
def make_sequence(tmp_path):
    def _make(name: str, n: int, *, r_min: float = 20.0, r_max: float = 70.0, **kw) -> Path:
        radii = np.linspace(r_min, r_max, n)
        return write_sequence(tmp_path / name, radii, **kw)

    return _make
