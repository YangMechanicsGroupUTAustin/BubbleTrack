"""Settings switches: off by default, validated, saved in the user config."""

from __future__ import annotations

import pytest

from bubbletrack.model import config


def test_defaults_are_off(isolated_config):
    assert config.ui_setting("show_classic_methods") is False
    assert config.ui_setting("show_all_view_modes") is False


def test_saved_and_read_back(isolated_config):
    assert config.set_ui_setting("show_classic_methods", True)
    assert config.ui_setting("show_classic_methods") is True
    assert config.ui_setting("show_all_view_modes") is False


def test_unknown_names_and_bad_values_are_refused(isolated_config):
    with pytest.raises(ValueError):
        config.ui_setting("nope")
    with pytest.raises(ValueError):
        config.set_meta(show_all_view_modes="yes")
