"""Folder scanning: decide which files of a folder form the image sequence.

Only one image format is loaded per folder.  Extension variants of the same
format (``.tif`` / ``.tiff``, ``.jpg`` / ``.jpeg``, in any letter case) form
one group; when several groups are present the highest-priority one wins and
the others are reported, never silently dropped.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# (group name, extensions) in priority order.
IMAGE_EXTENSION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tiff", (".tif", ".tiff")),
    ("png", (".png",)),
    ("jpeg", (".jpg", ".jpeg")),
    ("bmp", (".bmp",)),
)

_GROUP_OF_EXTENSION: dict[str, str] = {
    ext: name for name, exts in IMAGE_EXTENSION_GROUPS for ext in exts
}


@dataclass(frozen=True)
class ScanResult:
    """What a folder scan found.

    Attributes
    ----------
    images : tuple of str
        Absolute paths of the frames, natural-sorted by file name.
    group : str or None
        Name of the extension group the frames belong to ("tiff", "png",
        "jpeg", "bmp"), or None if no frames were found.
    ignored_counts : dict
        Lower-case extension -> number of image files of a lower-priority
        group that were not loaded, e.g. ``{".png": 3}``.
    hidden_skipped : int
        Hidden files with an image extension that were skipped (macOS
        ``._<name>`` AppleDouble companions, other dot-files).
    """

    images: tuple[str, ...]
    group: str | None = None
    ignored_counts: dict[str, int] = field(default_factory=dict)
    hidden_skipped: int = 0

    @property
    def warnings(self) -> tuple[str, ...]:
        """Human-readable notes for the user, empty when nothing is unusual."""
        notes = []
        if self.ignored_counts:
            ignored = ", ".join(
                f"{count} {ext}" for ext, count in self.ignored_counts.items()
            )
            exts = "/".join(dict(IMAGE_EXTENSION_GROUPS)[self.group])
            notes.append(
                f"Ignored {ignored} file(s); only the {exts} frames are loaded."
            )
        if self.hidden_skipped:
            notes.append(
                f"Skipped {self.hidden_skipped} hidden file(s) "
                "(names starting with '.', e.g. macOS '._' files)."
            )
        return tuple(notes)


def natural_sort_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically ("f2" < "f10")."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _candidate_files(folder: Path) -> tuple[dict[str, list[str]], int]:
    """Group the image files of *folder* by extension group.

    Returns ``({group: [paths]}, hidden_skipped)``.  Symlinks, directories,
    non-image files and hidden files are left out.
    """
    groups: dict[str, list[str]] = {}
    hidden = 0
    with os.scandir(folder) as entries:
        for entry in entries:
            group = _GROUP_OF_EXTENSION.get(os.path.splitext(entry.name)[1].lower())
            if group is None:
                continue
            if entry.name.startswith("."):
                hidden += 1
                continue
            if entry.is_symlink():
                logger.warning("Skipping symlink: %s", entry.path)
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            groups.setdefault(group, []).append(entry.path)
    return groups, hidden


def _extension_counts(paths: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def scan_folder_report(folder: str) -> ScanResult:
    """Scan *folder* for an image sequence and report what was left out.

    The first extension group that has files, in priority order
    (tif/tiff > png > jpg/jpeg > bmp), provides the frames; files of the
    other groups are counted in ``ignored_counts`` and a warning is logged.
    Hidden files (names starting with ``.``) are skipped: macOS writes
    ``._<name>`` AppleDouble companions next to every frame on shared or
    exFAT drives, and they would otherwise sort first and shift every frame
    index.  Symlinked files are skipped to avoid following unsafe links.

    Parameters
    ----------
    folder : str
        Folder to scan.

    Returns
    -------
    ScanResult
        Empty (``images == ()``) when *folder* is not a directory or holds
        no images.
    """
    p = Path(folder).resolve()
    if not p.is_dir():
        logger.warning("Not a directory: %s", folder)
        return ScanResult(images=())

    groups, hidden = _candidate_files(p)
    chosen = next((name for name, _ in IMAGE_EXTENSION_GROUPS if name in groups), None)
    if hidden:
        logger.info("Skipped %d hidden file(s) in %s", hidden, p)
    if chosen is None:
        return ScanResult(images=(), hidden_skipped=hidden)

    images = tuple(sorted(
        groups[chosen], key=lambda fp: natural_sort_key(os.path.basename(fp)),
    ))
    ignored: dict[str, int] = {}
    for name, _ in IMAGE_EXTENSION_GROUPS:
        if name != chosen and name in groups:
            ignored.update(_extension_counts(groups[name]))
    if ignored:
        logger.warning(
            "Ignored image files in %s: %s (only %s frames are loaded)",
            p, ignored, chosen,
        )
    return ScanResult(
        images=images, group=chosen, ignored_counts=ignored, hidden_skipped=hidden,
    )
