"""Find batch sources (image folders and video files) and read their frames.

A *source* is one experiment: a folder of images, or a single video file.
Much of the lab archive is video-only, typically one subfolder per shot
holding one ``.avi`` (``<date>/AVI8/<shot>/<shot>.avi``), so videos are
looked for both directly under the chosen root and in its immediate
subfolders.  Image folders are the immediate subfolders that hold images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bubbletrack.model.folder_scan import natural_sort_key
from bubbletrack.model.frame_source import FrameReader, is_recording_file, open_recording
from bubbletrack.model.image_io import scan_folder

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"}
SOURCE_KINDS = ("folder", "video")


@dataclass(frozen=True)
class BatchSource:
    """One experiment of a batch: an image folder or a video file."""

    path: Path
    kind: str = "folder"  # "folder" | "video"

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"Unknown batch source kind {self.kind!r}")

    @property
    def name(self) -> str:
        """Display name: the folder name or the video file name."""
        return self.path.name

    @property
    def output_dir(self) -> Path:
        """Results go inside an image folder, or next to a video."""
        return self.path if self.kind == "folder" else self.path.parent

    def output_path(self, prefix: str, timestamp: str) -> Path:
        """``<prefix>_<ts>.mat`` in a folder, ``<prefix>_<stem>_<ts>.mat`` for a video.

        Videos often share a folder, so their stem keeps the outputs apart.
        """
        if self.kind == "video":
            return self.output_dir / f"{prefix}_{self.path.stem}_{timestamp}.mat"
        return self.output_dir / f"{prefix}_{timestamp}.mat"


def as_source(item: BatchSource | str | Path) -> BatchSource:
    """A ``BatchSource`` for *item*; a plain path is a video if it is a video file."""
    if isinstance(item, BatchSource):
        return item
    path = Path(item)
    is_video = path.is_file() and is_recording_file(path.name)
    return BatchSource(path, "video" if is_video else "folder")


def _visible(path: Path) -> bool:
    return not path.name.startswith(".")


def _entries(folder: Path) -> list[Path]:
    """Visible entries of *folder* in natural order ([] when unreadable)."""
    try:
        entries = [p for p in folder.iterdir() if _visible(p)]
    except OSError as exc:
        logger.warning("Cannot list %s: %s", folder, exc)
        return []
    return sorted(entries, key=lambda p: natural_sort_key(p.name))


def _has_images(entries: list[Path]) -> bool:
    return any(p.suffix.lower() in _IMAGE_EXTS and p.is_file() for p in entries)


def _videos(entries: list[Path]) -> list[BatchSource]:
    return [BatchSource(p, "video") for p in entries
            if is_recording_file(p.name) and p.is_file()]


def find_batch_sources(root: str | Path) -> list[BatchSource]:
    """Image folders and videos under *root* (one level of subfolders).

    Returns videos directly in *root* first, then, per subfolder in natural
    order, the subfolder itself when it holds images followed by its videos.
    Hidden entries (names starting with ``"."``) are skipped.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        return []
    root_entries = _entries(root_path)
    sources = _videos(root_entries)
    for sub in root_entries:
        if not sub.is_dir():
            continue
        entries = _entries(sub)
        if _has_images(entries):
            sources.append(BatchSource(sub, "folder"))
        sources.extend(_videos(entries))
    return sources


def find_experiment_folders(root: str) -> list[Path]:
    """Image folders only (for callers that predate video sources)."""
    return [s.path for s in find_batch_sources(root) if s.kind == "folder"]


class SourceFrames:
    """The frames of one source, for one thread.

    ``read(i)`` returns the image path of a folder frame (cheap; worker
    processes read it) or the frame decoded from a video by a private
    ``VideoFrameReader`` (OpenCV captures must not be shared across threads).
    """

    def __init__(self, source: BatchSource) -> None:
        self.source = source
        self._reader: FrameReader | None = None
        self._images: list[str] = []
        if source.kind == "video":
            self._reader = open_recording(source.path)
            self.n_frames = self._reader.total_frames
        else:
            self._images = scan_folder(str(source.path))
            self.n_frames = len(self._images)

    def read(self, idx: int) -> str | np.ndarray:
        if self._reader is not None:
            return self._reader.read_frame(idx)
        return self._images[idx]

    def read_raw(self, idx: int) -> np.ndarray:
        """Frame *idx* at its native bit depth (what the Edge engine needs)."""
        from bubbletrack.model.edge_pipeline import read_raw
        return read_raw(self.read(idx))

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> SourceFrames:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def count_frames(source: BatchSource) -> int | None:
    """Number of frames in *source*, or None when it cannot be read."""
    try:
        with SourceFrames(source) as frames:
            return frames.n_frames
    except Exception as exc:
        logger.warning("Cannot read %s: %s", source.path, exc)
        return None


def batch_process_experiments(
    folders: list[Path],
    process_folder_fn: Callable[[Path], dict[str, Any]],
    callback: Callable[[int, int], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Process each folder as an independent experiment.

    Parameters
    ----------
    folders : list of Path
        Directories to process.
    process_folder_fn : callable(folder_path) -> result dict
        Must return ``{"success": True, "n_frames": int, "output": str}`` on
        success, or ``{"success": False, "error": str}`` on failure.
    callback : optional
        Progress callback invoked as ``callback(current, total)`` after each
        folder is processed.

    Returns
    -------
    dict mapping ``folder_name`` -> result dict
    """
    results: dict[str, dict[str, Any]] = {}
    for i, folder in enumerate(folders):
        try:
            result = process_folder_fn(folder)
            results[folder.name] = result
        except Exception as exc:
            logger.error("Failed to process %s: %s", folder.name, exc)
            results[folder.name] = {"success": False, "error": str(exc)}
        if callback:
            callback(i + 1, len(folders))
    return results
