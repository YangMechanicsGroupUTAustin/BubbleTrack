"""Tests for parameter persistence (model/config.py)."""

from __future__ import annotations

import json

import pytest

from bubbletrack.model.config import (
    PERSIST_KEYS,
    is_onboarding_done,
    load_config,
    save_config,
    set_onboarding_done,
)
from bubbletrack.model.state import AppState, update_state


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect CONFIG_DIR / CONFIG_FILE to a temp directory for every test."""
    config_dir = tmp_path / ".bubbletrack"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("bubbletrack.model.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("bubbletrack.model.config.CONFIG_FILE", config_file)


class TestSaveConfig:
    """Verify save_config writes the expected JSON."""

    def test_creates_directory_and_file(self, tmp_path):
        config_file = tmp_path / ".bubbletrack" / "config.json"
        state = AppState()
        save_config(state)
        assert config_file.exists()

    def test_saves_all_persist_keys(self, tmp_path):
        state = AppState()
        save_config(state)
        config_file = tmp_path / ".bubbletrack" / "config.json"
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert set(data.keys()) == PERSIST_KEYS

    def test_saves_modified_values(self, tmp_path):
        state = update_state(AppState(), img_thr=0.8, fps=500_000.0)
        save_config(state)
        config_file = tmp_path / ".bubbletrack" / "config.json"
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["img_thr"] == 0.8
        assert data["fps"] == 500_000.0

    def test_converts_tuples_to_lists(self, tmp_path):
        state = update_state(
            AppState(), bubble_cross_edges=(True, False, True, False),
        )
        save_config(state)
        config_file = tmp_path / ".bubbletrack" / "config.json"
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["bubble_cross_edges"] == [True, False, True, False]

    def test_survives_read_only_dir(self, tmp_path, monkeypatch):
        """save_config should not raise even when the directory is not writable."""
        bad_dir = tmp_path / "no_write"
        bad_dir.mkdir()
        bad_file = bad_dir / "config.json"
        monkeypatch.setattr("bubbletrack.model.config.CONFIG_DIR", bad_dir)
        monkeypatch.setattr("bubbletrack.model.config.CONFIG_FILE", bad_file)
        # Make the dir read-only (best-effort; Windows may not enforce)
        bad_dir.chmod(0o444)
        # Should not raise
        save_config(AppState())
        # Restore permissions for cleanup
        bad_dir.chmod(0o755)


class TestLoadConfig:
    """Verify load_config reads and filters the JSON correctly."""

    def test_returns_empty_dict_when_no_file(self):
        assert load_config() == {}

    def test_round_trip(self, tmp_path):
        state = update_state(
            AppState(),
            img_thr=0.7,
            removing_factor=50,
            bubble_cross_edges=(True, True, False, False),
            gaussian_sigma=1.5,
            clahe_clip=2.0,
            closing_radius=3,
            opening_radius=4,
            um2px=4.0,
            fps=250_000.0,
            rmax_fit_length=9,
        )
        save_config(state)
        loaded = load_config()
        assert loaded["img_thr"] == 0.7
        assert loaded["removing_factor"] == 50
        # Tuples should be restored from lists
        assert loaded["bubble_cross_edges"] == (True, True, False, False)
        assert isinstance(loaded["bubble_cross_edges"], tuple)
        assert loaded["gaussian_sigma"] == 1.5
        assert loaded["clahe_clip"] == 2.0
        assert loaded["closing_radius"] == 3
        assert loaded["opening_radius"] == 4
        assert loaded["um2px"] == 4.0
        assert loaded["fps"] == 250_000.0
        assert loaded["rmax_fit_length"] == 9

    def test_ignores_unknown_keys(self, tmp_path):
        config_file = tmp_path / ".bubbletrack" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            json.dumps({"img_thr": 0.6, "unknown_key": "garbage"}),
            encoding="utf-8",
        )
        loaded = load_config()
        assert "unknown_key" not in loaded
        assert loaded["img_thr"] == 0.6

    def test_returns_empty_on_invalid_json(self, tmp_path):
        config_file = tmp_path / ".bubbletrack" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("not valid json {{{", encoding="utf-8")
        assert load_config() == {}

    def test_returns_empty_on_non_object_json(self, tmp_path):
        config_file = tmp_path / ".bubbletrack" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert load_config() == {}


class TestLoadConfigApplyToState:
    """Verify that loaded config can be applied to create a new AppState."""

    def test_apply_loaded_config_to_state(self, tmp_path):
        original = update_state(AppState(), img_thr=0.3, fps=100_000.0)
        save_config(original)
        loaded = load_config()
        restored = update_state(AppState(), **loaded)
        assert restored.img_thr == 0.3
        assert restored.fps == 100_000.0
        # Fields not in PERSIST_KEYS should remain at defaults
        assert restored.image_no == 0
        assert restored.folder_path == ""


class TestOnboarding:
    """Verify onboarding flag helpers."""

    def test_default_is_not_done(self):
        assert is_onboarding_done() is False

    def test_set_and_read(self):
        set_onboarding_done()
        assert is_onboarding_done() is True

    def test_does_not_corrupt_persist_keys(self, tmp_path):
        """Setting onboarding_done should not interfere with persisted params."""
        state = update_state(AppState(), img_thr=0.9)
        save_config(state)
        set_onboarding_done()
        loaded = load_config()
        assert loaded["img_thr"] == 0.9
        assert is_onboarding_done() is True

    def test_survives_missing_config_dir(self):
        """set_onboarding_done should create the config dir if needed."""
        set_onboarding_done()
        assert is_onboarding_done() is True

    def test_onboarding_flag_survives_save_config(self):
        """Saving parameters on quit must not bring the welcome dialog back."""
        set_onboarding_done()
        save_config(update_state(AppState(), img_thr=0.7))
        assert is_onboarding_done() is True
        assert load_config()["img_thr"] == 0.7


class TestSaveKeepsOtherKeys:
    """save_config merges into the existing file instead of replacing it."""

    def test_unknown_keys_survive_save(self, tmp_path):
        config_file = tmp_path / ".bubbletrack" / "config.json"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(
            json.dumps({"future_option": [1, 2], "img_thr": 0.2}), encoding="utf-8",
        )
        save_config(update_state(AppState(), img_thr=0.7))
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["future_option"] == [1, 2]
        assert data["img_thr"] == 0.7

    def test_corrupt_existing_file_is_replaced(self, tmp_path):
        config_file = tmp_path / ".bubbletrack" / "config.json"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("{not json", encoding="utf-8")
        save_config(AppState())
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert set(data) == PERSIST_KEYS

    def test_non_utf8_file_neither_crashes_nor_sticks(self, tmp_path):
        config_file = tmp_path / ".bubbletrack" / "config.json"
        config_file.parent.mkdir(parents=True)
        config_file.write_bytes(b"\xff\xfe\x00garbage\x80")
        assert is_onboarding_done() is False
        assert load_config() == {}
        save_config(AppState())
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert set(data) == PERSIST_KEYS
