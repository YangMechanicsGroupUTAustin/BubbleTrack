"""Batch source discovery: image folders AND video files are experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bubbletrack.model.batch_experiments import (
    BatchSource,
    SourceFrames,
    as_source,
    count_frames,
    find_batch_sources,
    find_experiment_folders,
)
from test_video_fitting import write_video


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _describe(sources):
    return [(s.path.name, s.kind) for s in sources]


class TestFindBatchSources:
    def test_mixed_folders_and_videos(self, tmp_path):
        _touch(tmp_path / "exp_a" / "f_001.tiff")
        _touch(tmp_path / "loose.avi")
        _touch(tmp_path / "shot_1" / "shot_1.avi")
        _touch(tmp_path / "notes" / "readme.txt")
        assert _describe(find_batch_sources(tmp_path)) == [
            ("loose.avi", "video"),
            ("exp_a", "folder"),
            ("shot_1.avi", "video"),
        ]

    def test_avi_archive_layout_one_subfolder_per_shot(self, tmp_path):
        # <date>/AVI8/<shot>/<shot>.avi: the user picks AVI8
        for shot in ("Jin_14_42_03", "Jin_14_36_05", "Jin_14_37_31"):
            _touch(tmp_path / shot / f"{shot}.avi")
        names = [s.path.name for s in find_batch_sources(tmp_path)]
        assert names == ["Jin_14_36_05.avi", "Jin_14_37_31.avi", "Jin_14_42_03.avi"]
        assert {s.kind for s in find_batch_sources(tmp_path)} == {"video"}

    def test_all_video_extensions_any_case(self, tmp_path):
        for ext in (".avi", ".MP4", ".mov", ".mkv"):
            _touch(tmp_path / f"v{ext}")
        assert len(find_batch_sources(tmp_path)) == 4

    def test_folder_with_images_and_a_video_lists_both(self, tmp_path):
        _touch(tmp_path / "shot" / "f_001.png")
        _touch(tmp_path / "shot" / "shot.avi")
        assert _describe(find_batch_sources(tmp_path)) == [
            ("shot", "folder"), ("shot.avi", "video"),
        ]

    def test_skips_hidden_entries_and_deeper_levels(self, tmp_path):
        _touch(tmp_path / ".hidden" / "f.png")
        _touch(tmp_path / ".cache.avi")
        _touch(tmp_path / "exp" / ".f.png")
        _touch(tmp_path / "exp" / "deeper" / "x.avi")
        assert find_batch_sources(tmp_path) == []

    def test_natural_order(self, tmp_path):
        for name in ("exp_10", "exp_2", "exp_1"):
            _touch(tmp_path / name / "f.tif")
        assert [s.path.name for s in find_batch_sources(tmp_path)] == [
            "exp_1", "exp_2", "exp_10",
        ]

    def test_missing_root(self):
        assert find_batch_sources("/this/does/not/exist") == []

    def test_find_experiment_folders_still_returns_folders_only(self, tmp_path):
        _touch(tmp_path / "exp" / "f.tif")
        _touch(tmp_path / "v.avi")
        assert [p.name for p in find_experiment_folders(str(tmp_path))] == ["exp"]


class TestBatchSource:
    def test_folder_outputs_go_inside_the_folder(self, tmp_path):
        src = BatchSource(tmp_path / "exp", "folder")
        assert src.name == "exp"
        assert src.output_path("R_data", "20260101_120000") == (
            tmp_path / "exp" / "R_data_20260101_120000.mat")

    def test_video_outputs_go_next_to_the_video_with_its_stem(self, tmp_path):
        src = BatchSource(tmp_path / "shot" / "Jin_1.avi", "video")
        assert src.name == "Jin_1.avi"
        assert src.output_path("RofT_data", "T") == tmp_path / "shot" / "RofT_data_Jin_1_T.mat"

    def test_as_source_accepts_paths(self, tmp_path):
        video = _touch(tmp_path / "a.avi")
        (tmp_path / "exp").mkdir()
        assert as_source(video).kind == "video"
        assert as_source(str(tmp_path / "exp")).kind == "folder"
        src = BatchSource(video, "video")
        assert as_source(src) is src

    def test_unknown_kind_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            BatchSource(tmp_path, "zip")


class TestFrames:
    def test_count_frames(self, make_sequence, tmp_path):
        folder = make_sequence("seq", 5)
        video = write_video(tmp_path / "v.avi", [30.0] * 7)
        assert count_frames(BatchSource(folder, "folder")) == 5
        assert count_frames(BatchSource(video, "video")) == 7

    def test_count_frames_unreadable_video_is_none(self, tmp_path):
        bogus = _touch(tmp_path / "broken.avi")
        assert count_frames(BatchSource(bogus, "video")) is None

    def test_folder_frames_are_paths(self, make_sequence):
        folder = make_sequence("seq", 3)
        with SourceFrames(BatchSource(folder, "folder")) as frames:
            assert frames.n_frames == 3
            assert isinstance(frames.read(1), str)
            assert frames.read_raw(1).dtype == np.uint16

    def test_video_frames_are_decoded_arrays(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [30.0, 40.0, 50.0])
        with SourceFrames(BatchSource(video, "video")) as frames:
            assert frames.n_frames == 3
            frame = frames.read(2)
            assert isinstance(frame, np.ndarray) and frame.shape == (250, 400)
