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
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from bubbletrack.model.long_paths import os_path, plain

logger = logging.getLogger(__name__)

# Upper bound on IFDs followed, a guard against corrupt chains.
_TIFF_MAX_PAGES = 1_000_000

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
    multipage : dict
        Path -> page count for frames that are multi-page TIFFs.  Only the
        first page of such a file is read, so the UI should warn about them.
    """

    images: tuple[str, ...]
    group: str | None = None
    ignored_counts: dict[str, int] = field(default_factory=dict)
    hidden_skipped: int = 0
    multipage: dict[str, int] = field(default_factory=dict)

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
        for path, pages in self.multipage.items():
            notes.append(
                f"{os.path.basename(path)} has {pages} pages; "
                "only the first page is read."
            )
        return tuple(notes)


def tiff_page_count(path: str | os.PathLike) -> int:
    """Return the number of pages (top-level IFDs) of a TIFF file.

    Walks the IFD chain reading only the header and the IFD links (a few
    bytes per page, never pixel data), so it is cheap enough to run on every
    frame of a folder scan, and it opens files with Python's ``open`` so
    non-ASCII paths work.  ``cv2.imcount`` is header-only too but cannot open
    non-ASCII paths on Windows; ``cv2.imdecodemulti`` decodes every page.
    Classic and BigTIFF, little- and big-endian files are supported.

    Parameters
    ----------
    path : str or path-like
        File to inspect.

    Returns
    -------
    int
        Number of pages; 0 if the file is missing, unreadable or not a
        TIFF.  A truncated or looping IFD chain counts the pages read so far.
    """
    try:
        with open(os_path(path), "rb") as fh:
            return _count_ifds(fh)
    except OSError:
        return 0


def _count_ifds(fh: BinaryIO) -> int:
    header = fh.read(16)
    order = {b"II": "<", b"MM": ">"}.get(header[:2])
    if order is None or len(header) < 8:
        return 0
    magic = struct.unpack(order + "H", header[2:4])[0]
    if magic == 42:  # classic TIFF: 2-byte entry count, 12-byte entries, 4-byte links
        count_fmt, entry_size, link_fmt = "H", 12, "I"
        offset = struct.unpack(order + "I", header[4:8])[0]
    elif magic == 43 and len(header) == 16:  # BigTIFF: 8 / 20 / 8 bytes
        count_fmt, entry_size, link_fmt = "Q", 20, "Q"
        offset = struct.unpack(order + "Q", header[8:16])[0]
    else:
        return 0
    count_size = struct.calcsize(order + count_fmt)
    link_size = struct.calcsize(order + link_fmt)

    pages = 0
    seen: set[int] = set()
    while offset and offset not in seen and pages < _TIFF_MAX_PAGES:
        seen.add(offset)
        fh.seek(offset)
        raw = fh.read(count_size)
        if len(raw) < count_size:
            break
        pages += 1
        n_entries = struct.unpack(order + count_fmt, raw)[0]
        fh.seek(offset + count_size + n_entries * entry_size)
        raw = fh.read(link_size)
        if len(raw) < link_size:
            break
        offset = struct.unpack(order + link_fmt, raw)[0]
    return pages


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
    with os.scandir(os_path(folder)) as entries:  # paths beyond MAX_PATH too
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
            groups.setdefault(group, []).append(os.path.join(str(folder), entry.name))
    return groups, hidden


def _extension_counts(paths: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def _multipage_files(paths: tuple[str, ...]) -> dict[str, int]:
    counts = {path: tiff_page_count(path) for path in paths}
    return {path: pages for path, pages in counts.items() if pages > 1}


def scan_folder_report(folder: str, *, check_pages: bool = True) -> ScanResult:
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
    check_pages : bool
        If True (default) and the frames are TIFFs, read each file's header
        to flag multi-page files in ``multipage``.

    Returns
    -------
    ScanResult
        Empty (``images == ()``) when *folder* is not a directory or holds
        no images.
    """
    p = Path(plain(str(Path(folder).resolve())))
    if not os.path.isdir(os_path(p)):
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
    multipage = _multipage_files(images) if check_pages and chosen == "tiff" else {}
    if multipage:
        logger.warning(
            "%d multi-page TIFF(s) in %s; only the first page of each is read",
            len(multipage), p,
        )
    return ScanResult(
        images=images, group=chosen, ignored_counts=ignored,
        hidden_skipped=hidden, multipage=multipage,
    )
