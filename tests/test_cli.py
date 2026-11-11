"""Tests for the headless CLI (bubbletrack.cli)."""

from __future__ import annotations

import csv
import logging

import cv2
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack import cli
from bubbletrack.model import frame_pipeline
from bubbletrack.model.constants import DEFAULT_REMOVING_FACTOR
from bubbletrack.model.removing_factor import compute_removing_factor

SHAPE = (120, 160)
RADIUS = 40.0


@pytest.fixture(autouse=True)
def _restore_bubbletrack_logger():
    """cli.main() calls setup_logging(), which adds a handler on every call."""
    logger = logging.getLogger("bubbletrack")
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.fixture()
def frames(tmp_path):
    """A folder of three synthetic 16-bit frames with a 40 px bubble."""
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(3):
        img = render_scene(SceneSpec(radius=RADIUS, center=(60.0, 80.0), shape=SHAPE, seed=i))
        assert cv2.imwrite(str(folder / f"frame_{i:03d}.tiff"), img)
    return folder


def _run(folder, out, *extra: str) -> None:
    cli.main(["--folder", str(folder), "--output", str(out), *extra])


def _csv_radii(path) -> list[float]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    key = next(k for k in rows[0] if k.lower().startswith("radius")) if rows else None
    return [float(row[key]) for row in rows]


class TestModes:
    @pytest.mark.parametrize("mode", ["intensity", "gradient", "area"])
    def test_each_mode_fits_the_bubble(self, frames, tmp_path, capsys, mode):
        out = tmp_path / "out.csv"
        _run(frames, out, "--mode", mode, "--threshold", "0.58")
        assert "3/3 frames fitted" in capsys.readouterr().out
        radii = _csv_radii(out)
        assert len(radii) == 3
        assert all(abs(r - RADIUS) < 3.0 for r in radii), radii

    def test_default_mode_is_intensity(self, frames, tmp_path, monkeypatch):
        seen = _spy_params(monkeypatch)
        _run(frames, tmp_path / "out.csv")
        assert [p.mode for p in seen] == ["intensity"] * 3

    def test_unknown_mode_is_a_usage_error(self, frames, tmp_path):
        with pytest.raises(SystemExit) as exc:
            _run(frames, tmp_path / "out.csv", "--mode", "bogus")
        assert exc.value.code == 2


def _spy_params(monkeypatch) -> list:
    """Record the FrameParams the CLI passes per frame (frames still get fitted)."""
    seen = []
    real = frame_pipeline.fit_image_file

    def spy(path, params):
        seen.append(params)
        return real(path, params)

    monkeypatch.setattr(cli, "fit_image_file", spy)
    return seen


class TestRemovingFactor:
    """--rf is the GUI's Remove slider (0-100), mapped exactly like the GUI."""

    @pytest.mark.parametrize("rf", [None, "0", "50", "85.5", "100"])
    def test_rf_is_mapped_like_the_gui_slider(self, frames, tmp_path, monkeypatch, rf):
        seen = _spy_params(monkeypatch)
        extra = () if rf is None else ("--rf", rf)
        _run(frames, tmp_path / "out.csv", *extra)
        slider = DEFAULT_REMOVING_FACTOR if rf is None else float(rf)
        expected = compute_removing_factor(slider, (1, SHAPE[0]), (1, SHAPE[1]))
        assert {p.removing_factor for p in seen} == {expected}

    def test_rf_mapping_uses_the_roi(self, frames, tmp_path, monkeypatch):
        seen = _spy_params(monkeypatch)
        _run(frames, tmp_path / "out.csv", "--roi", "11", "110", "21", "140")
        expected = compute_removing_factor(DEFAULT_REMOVING_FACTOR, (11, 110), (21, 140))
        assert {p.removing_factor for p in seen} == {expected}

    def test_rf_area_is_a_raw_pixel_override(self, frames, tmp_path, monkeypatch):
        seen = _spy_params(monkeypatch)
        _run(frames, tmp_path / "out.csv", "--rf", "10", "--rf-area", "90")
        assert {p.removing_factor for p in seen} == {90}

    def test_help_explains_the_change(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])
        assert exc.value.code == 0
        text = " ".join(capsys.readouterr().out.split())
        assert "--rf-area" in text
        assert "slider" in text
        assert "raw pixel area" in text
        assert "Changed in 3.1" in text


class TestRadiusFilter:
    """Same sanity filter as the GUI: finite, > 0, <= max(h, w)."""

    @pytest.mark.parametrize(("fake_radius", "fitted"), [
        (40_000.0, 0), (float("nan"), 0), (max(SHAPE) + 0.5, 0), (float(max(SHAPE)), 3),
    ])
    def test_implausible_radii_are_not_counted(
        self, frames, tmp_path, monkeypatch, capsys, fake_radius, fitted,
    ):
        monkeypatch.setattr(
            frame_pipeline, "measure_detection",
            lambda *args, **kwargs: (fake_radius, 60.0, 80.0, 0),
        )
        out = tmp_path / "out.csv"
        try:
            _run(frames, out)
        except SystemExit:
            pass  # a run with no fitted frame may exit non-zero
        assert f"{fitted}/3 frames fitted" in capsys.readouterr().out
