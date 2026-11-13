"""Upgraded installs start in Auto: old configs saved "intensity" as the default."""

from __future__ import annotations

from bubbletrack.controller.controller import initial_state
from bubbletrack.model import config


def test_fresh_start_is_auto(isolated_config):
    assert initial_state({}).detection_mode == "edge"


def test_old_classic_default_becomes_auto(isolated_config):
    assert initial_state({"detection_mode": "intensity", "img_thr": 0.6}).detection_mode == "edge"


def test_classic_kept_when_settings_show_classic_methods(isolated_config):
    config.set_ui_setting("show_classic_methods", True)
    assert initial_state({"detection_mode": "gradient"}).detection_mode == "gradient"
