"""Tests for the headless CLI (bubbletrack.cli)."""

from __future__ import annotations

import csv
import logging

import cv2
import pytest

from _golden_cases import SceneSpec, render_scene
from bubbletrack import cli
from bubbletrack.model import frame_pipeline

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
        seen = []
        real = frame_pipeline.fit_image_file

        def spy(path, params):
            seen.append(params.mode)
            return real(path, params)

        monkeypatch.setattr(cli, "fit_image_file", spy)
        _run(frames, tmp_path / "out.csv")
        assert seen == ["intensity"] * 3

    def test_unknown_mode_is_a_usage_error(self, frames, tmp_path):
        with pytest.raises(SystemExit) as exc:
            _run(frames, tmp_path / "out.csv", "--mode", "bogus")
        assert exc.value.code == 2


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
