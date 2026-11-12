"""Reader for raw Shimadzu HPV-X2 recordings (``.dat``).

The layout was reverse-engineered from files written by the HPV-X software
(version 1.0.3.3, recordings from 2024 to 2026).  Checked on three shots,
all 256 frames each: a TIFF16 export is bit-identical; an 8-bit RGB export
is exactly ``frame >> 8`` in every channel; another TIFF16 export is
``min(frame + 655, 65535)``, i.e. the export applied a brightness offset
that the ``.dat`` does not record.  The ``.dat`` holds the sensor values.

* Bytes 0-15: a Windows SYSTEMTIME (when the file was written); bytes 16-19:
  an int32 (always -5; meaning unknown).
* From byte 20: a chain of tagged entries, each
  ``tag:u32  type:u16  count:u32  offset:u32`` immediately followed by its
  payload at *offset*; the next entry starts right after the payload.
  Types: 0 = bytes (NUL-terminated ASCII), 1 = uint16, 2 = int32,
  3 = SYSTEMTIME (16 bytes).
* Tags used here: ``0x60036060`` width, ``0x60046060`` height,
  ``0x60056060`` frame count (int32 each); ``0x60066060`` per-frame time
  stamps (24-byte ASCII slots, ns: "890", "1890", ...); ``0x30073030`` the
  frame interval (ASCII, ns); ``0xA001A0A0`` the pixels.
* Pixels: uint16 little-endian, ``count = frames * height * width``, frames
  back to back, each stored bottom-up (the first row in the file is the
  bottom image row).  Values are 11-bit counts scaled by 32 (multiples of 32,
  at most 65504), exactly what the TIFF16 export holds.

The frame rate is 1e9 / the time-stamp spacing (else the interval field).
Every archive file inspected was recorded at 1 Mfps (1000 ns); other rates
could not be cross-checked against an independent source.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bubbletrack.model.image_io import ImageLoadError

DAT_EXTS = frozenset({".dat"})

TAG_WIDTH = 0x60036060
TAG_HEIGHT = 0x60046060
TAG_FRAMES = 0x60056060
TAG_TIMES = 0x60066060
TAG_INTERVAL = 0x30073030
TAG_PIXELS = 0xA001A0A0

_FIRST_ENTRY = 20  # after the SYSTEMTIME and the int32
_ENTRY = struct.Struct("<IHII")
_TYPE_SIZES = {0: 1, 1: 2, 2: 4, 3: 16}
_HEADER_READ = 1 << 20  # the directory sits in the first few KB
_MAX_ENTRIES = 4096


class DatFormatError(ImageLoadError):
    """The file is not an HPV-X2 recording this reader understands."""


def is_dat_file(path: str | os.PathLike) -> bool:
    """True if *path* has the HPV-X2 raw recording extension (``.dat``)."""
    return Path(path).suffix.lower() in DAT_EXTS


@dataclass(frozen=True)
class DatHeader:
    width: int
    height: int
    n_frames: int
    data_offset: int  # byte offset of the first frame
    interval_ns: float  # NaN when unknown

    @property
    def fps(self) -> float:
        """Recording frame rate, 0.0 when unknown (like VideoFrameReader)."""
        if math.isfinite(self.interval_ns) and self.interval_ns > 0:
            return 1e9 / self.interval_ns
        return 0.0

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 2


def _entries(buf: bytes) -> dict[int, tuple[int, int, bytes | int]]:
    """{tag: (type, count, payload)}; the pixel entry maps to its offset instead."""
    entries: dict[int, tuple[int, int, bytes | int]] = {}
    pos = _FIRST_ENTRY
    for _ in range(_MAX_ENTRIES):
        if pos + _ENTRY.size > len(buf):
            break
        tag, typ, count, offset = _ENTRY.unpack_from(buf, pos)
        if typ not in _TYPE_SIZES or offset != pos + _ENTRY.size:
            raise DatFormatError(f"not an HPV-X2 header (bad entry at byte {pos})")
        if tag == TAG_PIXELS:
            entries[tag] = (typ, count, offset)
            return entries
        size = _TYPE_SIZES[typ] * count
        if offset + size > len(buf):
            break
        entries[tag] = (typ, count, buf[offset:offset + size])
        pos = offset + size
    raise DatFormatError("no pixel data entry in the header")


def _int(entries: dict, tag: int, name: str) -> int:
    typ, count, payload = entries.get(tag, (None, 0, b""))
    if typ != 2 or count < 1:
        raise DatFormatError(f"header has no {name}")
    value = struct.unpack_from("<i", payload)[0]
    if value <= 0:
        raise DatFormatError(f"invalid {name}: {value}")
    return value


def _number(raw: bytes) -> float:
    text = raw.split(b"\0", 1)[0].decode("ascii", "replace").strip()
    try:
        return float(text)
    except ValueError:
        return math.nan


def _interval_ns(entries: dict, n_frames: int) -> float:
    """Spacing of the per-frame time stamps, else the interval field (ns)."""
    typ, _count, payload = entries.get(TAG_TIMES, (None, 0, b""))
    if typ == 0 and n_frames >= 2 and len(payload) >= n_frames:
        slot = len(payload) // n_frames
        times = np.array([_number(payload[i * slot:(i + 1) * slot]) for i in range(n_frames)])
        steps = np.diff(times)
        if np.isfinite(steps).all() and steps[0] > 0 and np.allclose(steps, steps[0]):
            return float(steps[0])
    typ, _count, payload = entries.get(TAG_INTERVAL, (None, 0, b""))
    value = _number(payload) if typ == 0 else math.nan
    return value if value > 0 else math.nan


def read_header(path: str | os.PathLike) -> DatHeader:
    """Parse and validate the header of an HPV-X2 ``.dat`` recording."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            buf = fh.read(_HEADER_READ)
    except OSError as exc:
        raise ImageLoadError(f"Cannot open {path}: {exc}") from exc
    if len(buf) < _FIRST_ENTRY + _ENTRY.size:
        raise DatFormatError(f"file too short for an HPV-X2 header: {path}")

    entries = _entries(buf)
    width = _int(entries, TAG_WIDTH, "width")
    height = _int(entries, TAG_HEIGHT, "height")
    n_frames = _int(entries, TAG_FRAMES, "frame count")
    typ, count, offset = entries[TAG_PIXELS]
    if typ != 1 or count != n_frames * height * width:
        raise DatFormatError(
            f"pixel entry holds {count} values, expected {n_frames}x{height}x{width}")
    if size < offset + 2 * count:
        raise DatFormatError(
            f"file truncated: {size} bytes, the frames need {offset + 2 * count}")
    return DatHeader(width, height, n_frames, offset, _interval_ns(entries, n_frames))


class DatFrameReader:
    """Lazy frame reader for HPV-X2 ``.dat`` files.

    Same interface as ``image_io.VideoFrameReader`` (``total_frames``,
    ``fps``, ``read_frame``, ``close``).  Frames are uint16 at the camera's
    native scaling, oriented like the camera software's TIFF export.  Not
    thread-safe (one open file): give each thread its own reader.
    """

    def __init__(self, path: str) -> None:
        self._path = str(path)
        self.header = read_header(self._path)
        try:
            self._file = open(self._path, "rb")
        except OSError as exc:
            raise ImageLoadError(f"Cannot open {self._path}: {exc}") from exc

    @property
    def total_frames(self) -> int:
        return self.header.n_frames

    @property
    def fps(self) -> float:
        return self.header.fps

    def read_frame(self, idx: int) -> np.ndarray:
        """Frame *idx* (0-based) as a (height, width) uint16 array."""
        header = self.header
        if self._file is None:
            raise ImageLoadError(f"Reader for {self._path} is closed")
        if not 0 <= idx < header.n_frames:
            raise ImageLoadError(
                f"Frame index {idx} out of range [0, {header.n_frames}) for {self._path}")
        self._file.seek(header.data_offset + idx * header.frame_bytes)
        buf = self._file.read(header.frame_bytes)
        if len(buf) != header.frame_bytes:
            raise ImageLoadError(f"Cannot read frame {idx} from {self._path}")
        frame = np.frombuffer(buf, dtype="<u2").reshape(header.height, header.width)
        return frame[::-1].astype(np.uint16)  # bottom-up on disk; copy is C-contiguous

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> DatFrameReader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_file", None) is not None:
            self.close()
