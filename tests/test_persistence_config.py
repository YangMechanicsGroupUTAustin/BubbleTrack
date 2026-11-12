"""App bookkeeping in config.json: last export folder/format and the clean-exit time.

These "meta" keys are validated like the persisted parameters, but they are
not AppState fields, so ``load_config`` must never return them.
"""

from __future__ import annotations

import json
import logging

import pytest

from bubbletrack.model import config
from bubbletrack.model.state import AppState, update_state


@pytest.fixture(autouse=True)
def cfg(isolated_config):
    return config.CONFIG_FILE


def _write(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestRoundTrip:
    def test_export_choice(self, tmp_path):
        folder = str(tmp_path / "exports")
        assert config.set_meta(last_export_dir=folder, last_export_format="csv")
        assert config.get_meta("last_export_dir") == folder
        assert config.get_meta("last_export_format") == "csv"

    def test_clean_exit(self):
        assert config.get_meta("last_clean_exit") is None
        config.record_clean_exit(1_700_000_000.5)
        assert config.get_meta("last_clean_exit") == 1_700_000_000.5

    def test_record_clean_exit_defaults_to_now(self, monkeypatch):
        monkeypatch.setattr(config.time, "time", lambda: 123.0)
        config.record_clean_exit()
        assert config.get_meta("last_clean_exit") == 123.0

    def test_missing_file_gives_none(self):
        assert config.get_meta("last_export_dir") is None


class TestCoexistence:
    def test_meta_keys_never_reach_app_state(self, cfg, tmp_path):
        config.set_meta(last_export_dir=str(tmp_path), last_export_format="xlsx")
        config.record_clean_exit(5.0)
        loaded = config.load_config()
        update_state(AppState(), **loaded)  # would raise TypeError on a meta key
        assert not set(loaded) & {"last_export_dir", "last_export_format",
                                  "last_clean_exit"}

    def test_save_config_keeps_meta_keys(self, cfg, tmp_path):
        config.set_meta(last_export_dir=str(tmp_path), last_export_format="mat")
        config.save_config(AppState(img_thr=0.3))
        data = _read(cfg)
        assert data["last_export_dir"] == str(tmp_path)
        assert data["img_thr"] == 0.3

    def test_set_meta_keeps_other_keys(self, cfg):
        _write(cfg, {"img_thr": 0.25, "onboarding_done": True})
        config.set_meta(last_export_format="csv")
        data = _read(cfg)
        assert data["img_thr"] == 0.25
        assert data["onboarding_done"] is True

    def test_corrupt_file_is_replaced(self, cfg):
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("{not json", encoding="utf-8")
        assert config.set_meta(last_export_format="csv")
        assert config.get_meta("last_export_format") == "csv"


class TestValidation:
    @pytest.mark.parametrize("value", ["CSV", " xlsx "])
    def test_format_is_normalised(self, cfg, value):
        _write(cfg, {"last_export_format": value})
        assert config.get_meta("last_export_format") == value.strip().lower()

    @pytest.mark.parametrize("value", ["pdf", "", 3, None, ["csv"]])
    def test_bad_format_is_ignored(self, cfg, value, caplog):
        _write(cfg, {"last_export_format": value})
        with caplog.at_level(logging.WARNING):
            assert config.get_meta("last_export_format") is None
        assert "last_export_format" in caplog.text

    @pytest.mark.parametrize("value", ["", "   ", "relative/dir", 42, None, "C:/a\x00b"])
    def test_bad_folder_is_ignored(self, cfg, value):
        _write(cfg, {"last_export_dir": value})
        assert config.get_meta("last_export_dir") is None

    def test_folder_is_stripped(self, cfg, tmp_path):
        _write(cfg, {"last_export_dir": f"  {tmp_path}  "})
        assert config.get_meta("last_export_dir") == str(tmp_path)

    @pytest.mark.parametrize("value", [-1.0, "abc", True, float("nan")])
    def test_bad_clean_exit_is_ignored(self, cfg, value):
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"last_clean_exit": value}), encoding="utf-8")
        assert config.get_meta("last_clean_exit") is None

    def test_numeric_string_clean_exit_is_coerced(self, cfg):
        _write(cfg, {"last_clean_exit": "12.5"})
        assert config.get_meta("last_clean_exit") == 12.5

    def test_invalid_values_are_not_written(self, cfg):
        with pytest.raises(ValueError, match="last_export_format"):
            config.set_meta(last_export_format="pdf")
        assert not cfg.exists()

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            config.set_meta(img_thr=0.5)
        with pytest.raises(ValueError, match="unknown"):
            config.get_meta("nope")

    def test_write_failure_returns_false(self, monkeypatch, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("", encoding="utf-8")  # a file where the dir should be
        monkeypatch.setattr(config, "CONFIG_DIR", blocker / "sub")
        monkeypatch.setattr(config, "CONFIG_FILE", blocker / "sub" / "config.json")
        assert config.set_meta(last_export_format="csv") is False
