"""Hard-case bundles on disk: one ``.zip``, or the folder it was unpacked to.

Layout::

    case.json                    the case (see ``schema``)
    frames/frame_000121.tif      one ROI crop per frame, named by frame number

Frames keep their native bit depth and are stored losslessly: a TIFF when
OpenCV writes the array back bit for bit (uint8 / uint16 / float32, grey or
colour), else a NumPy ``.npy`` file.  Users may unpack and re-zip a bundle,
so a zip whose files sit in one top-level folder reads the same.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from bubbletrack.model.cases.schema import (
    CASE_JSON, FRAMES_DIR, CaseFormatError, HardCase, case_from_json, case_to_json,
)

TIFF_EXT = ".tif"
NPY_EXT = ".npy"
MAX_MEMBER_BYTES = 1 << 30  # refuse absurd (zip-bomb sized) members


def frame_file_name(index: int, ext: str = TIFF_EXT) -> str:
    """``frames/frame_000121.tif`` for frame index 120 (the app's frame 121)."""
    return f"{FRAMES_DIR}/frame_{index + 1:06d}{ext}"


def encode_frame(image: np.ndarray) -> tuple[str, bytes]:
    """(extension, file bytes) storing *image* without loss."""
    try:
        ok, buf = cv2.imencode(TIFF_EXT, image)
    except cv2.error:
        ok, buf = False, None
    if ok:
        back = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if back is not None and back.dtype == image.dtype and back.shape == image.shape \
                and np.array_equal(back, image):
            return TIFF_EXT, buf.tobytes()
    out = io.BytesIO()
    np.save(out, np.ascontiguousarray(image), allow_pickle=False)
    return NPY_EXT, out.getvalue()


def decode_frame(name: str, data: bytes) -> np.ndarray:
    """The image stored in bundle file *name*; CaseFormatError if unreadable."""
    try:
        if name.lower().endswith(NPY_EXT):
            image = np.load(io.BytesIO(data), allow_pickle=False)
        else:
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (ValueError, OSError, cv2.error) as exc:
        raise CaseFormatError(f"cannot decode {name}: {exc}") from exc
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise CaseFormatError(f"cannot decode {name}")
    return image


def write_bundle(path: str | os.PathLike, case: HardCase,
                 images: Mapping[int, np.ndarray]) -> tuple[HardCase, int]:
    """Write *case* with its frame *images* (keyed by frame index) as one zip.

    Frame file names are assigned here (``frames/frame_<number>.<ext>``).
    The zip is written next to *path* first and moved into place, so a
    failed export never leaves a broken bundle behind.

    Returns
    -------
    (HardCase, int)
        The case as written (with its file names) and the zip size in bytes.

    Raises
    ------
    KeyError
        If an image is missing for a frame of *case*.
    OSError
        If the file cannot be written.
    """
    target = Path(path)
    encoded = {f.index: encode_frame(images[f.index]) for f in case.frames}
    frames = tuple(dataclasses.replace(f, file=frame_file_name(f.index, encoded[f.index][0]))
                   for f in case.frames)
    written = dataclasses.replace(case, frames=frames)
    text = json.dumps(case_to_json(written), indent=2, allow_nan=False)
    # A short temporary name: a long target name plus a suffix can pass MAX_PATH.
    fd, tmp_name = tempfile.mkstemp(prefix=".hardcase-", suffix=".part", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(CASE_JSON, text)
            for frame in frames:
                zf.writestr(frame.file, encoded[frame.index][1])
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return written, target.stat().st_size


class CaseBundle:
    """An opened bundle: its case and its frames, read on demand.

    Use as a context manager (or call ``close``) to release a zip file.
    """

    def __init__(self, path: Path, case: HardCase, read_bytes, close=lambda: None) -> None:
        self.path = path
        self.case = case
        self._read_bytes = read_bytes
        self._close = close

    def read_frame(self, index: int) -> np.ndarray:
        """The stored (ROI-cropped) image of frame *index*."""
        frame = self.case.frame(index)
        if frame is None:
            raise KeyError(f"frame {index + 1} is not in {self.path.name}")
        return decode_frame(frame.file, self._read_bytes(frame.file))

    def close(self) -> None:
        self._close()

    def __enter__(self) -> CaseBundle:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _parse_case(text: bytes, where: Path) -> HardCase:
    try:
        data = json.loads(text.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaseFormatError(f"{where.name}: invalid case.json ({exc})") from exc
    return case_from_json(data)


def _zip_prefix(names: list[str]) -> str:
    """'' or the one folder holding case.json (a re-zipped unpacked bundle)."""
    found = sorted((n for n in names if n == CASE_JSON or n.endswith("/" + CASE_JSON)), key=len)
    if not found:
        raise CaseFormatError("no case.json: not a BubbleTrack hard case")
    return found[0][: -len(CASE_JSON)]


def _open_zip(path: Path) -> CaseBundle:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise CaseFormatError(f"{path.name}: not a readable zip ({exc})") from exc
    try:
        prefix = _zip_prefix(zf.namelist())

        def read(name: str) -> bytes:
            try:
                info = zf.getinfo(prefix + name)
            except KeyError as exc:
                raise CaseFormatError(f"{path.name}: {name} is missing") from exc
            if info.file_size > MAX_MEMBER_BYTES:
                raise CaseFormatError(f"{path.name}: {name} is too large")
            return zf.read(info)

        return CaseBundle(path, _parse_case(read(CASE_JSON), path), read, zf.close)
    except Exception:
        zf.close()
        raise


def _open_folder(path: Path) -> CaseBundle:
    root = path.resolve()

    def read(name: str) -> bytes:
        file = (root / name).resolve()
        if root not in file.parents:
            raise CaseFormatError(f"{path.name}: {name} lies outside the bundle")
        try:
            return file.read_bytes()
        except OSError as exc:
            raise CaseFormatError(f"{path.name}: cannot read {name} ({exc})") from exc

    return CaseBundle(path, _parse_case(read(CASE_JSON), path), read)


def open_bundle(path: str | os.PathLike) -> CaseBundle:
    """Open a bundle (zip or unpacked folder).

    Raises
    ------
    CaseFormatError
        If *path* is not a readable hard case.
    """
    path = Path(path)
    if path.is_dir():
        return _open_folder(path)
    if not path.is_file():
        raise CaseFormatError(f"{path} does not exist")
    return _open_zip(path)


def find_bundles(folder: str | os.PathLike) -> list[Path]:
    """Every bundle under *folder*: ``.zip`` files and folders holding a case.json."""
    root = Path(folder)
    zips = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"]
    unpacked = [p.parent for p in root.rglob(CASE_JSON) if p.is_file()]
    return sorted(set(zips) | set(unpacked))
