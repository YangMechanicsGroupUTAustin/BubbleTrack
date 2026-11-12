"""HPV-X2 raw ``.dat`` reader.

Synthetic files built with the reverse-engineered layout run everywhere; the
real recording (``test_data/dat``, git-excluded; found through
``BUBBLETRACK_TEST_DATA`` or next to any parent of this file, like
test_edge_real) is checked frame by frame against the camera software's
TIFF16 export.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubbletrack.model.hpvx2_dat import (
    DatFormatError,
    DatFrameReader,
    is_dat_file,
    read_header,
)
from bubbletrack.model.image_io import ImageLoadError


def _sample_dir() -> Path:
    env = os.environ.get("BUBBLETRACK_TEST_DATA")
    candidates = [Path(env)] if env else []
    candidates += [p / "test_data" for p in Path(__file__).resolve().parents]
    found = next((c for c in candidates if (c / "dat").is_dir()), candidates[-1])
    return found / "dat"


SAMPLE_DIR = _sample_dir()
SAMPLE = SAMPLE_DIR / "Jin_14_42_03.dat"
SAMPLE_TIFFS = SAMPLE_DIR / "Jin_14_42_03"

TAG_WIDTH, TAG_HEIGHT, TAG_FRAMES = 0x60036060, 0x60046060, 0x60056060
TAG_TIMES, TAG_INTERVAL, TAG_PIXELS = 0x60066060, 0x30073030, 0xA001A0A0


def _entry(tag: int, typ: int, count: int, payload: bytes, pos: int) -> bytes:
    return struct.pack("<IHII", tag, typ, count, pos + 14) + payload


def build_dat(frames: np.ndarray, *, interval_ns: int = 1000, times=True,
              truncate: int = 0, extra: bytes = b"") -> bytes:
    """A .dat with the camera's layout: tagged header, then bottom-up frames."""
    n, h, w = frames.shape
    out = bytearray(struct.pack("<8H", 2024, 2, 1, 12, 14, 42, 9, 512))
    out += struct.pack("<i", -5)
    entries = [(0x10011010, 0, 1024, bytes(1024)),
               (TAG_INTERVAL, 0, 24, str(interval_ns).encode().ljust(24, b"\0")),
               (TAG_WIDTH, 2, 1, struct.pack("<i", w)),
               (TAG_HEIGHT, 2, 1, struct.pack("<i", h)),
               (TAG_FRAMES, 2, 1, struct.pack("<i", n))]
    if times:
        slots = b"".join(str(890 + k * interval_ns).encode().ljust(24, b"\0") for k in range(n))
        entries.append((TAG_TIMES, 0, 24 * n, slots))
    for tag, typ, count, payload in entries:
        out += _entry(tag, typ, count, payload, len(out))
    out += struct.pack("<IHII", TAG_PIXELS, 1, n * h * w, len(out) + 14)
    out += frames[:, ::-1, :].astype("<u2").tobytes()  # stored bottom-up
    out += extra
    return bytes(out[:len(out) - truncate] if truncate else out)


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    return (rng.integers(0, 2048, size=(5, 6, 7)) * 32).astype(np.uint16)


@pytest.fixture
def dat(tmp_path, frames):
    path = tmp_path / "Jin_00_00_01.dat"
    path.write_bytes(build_dat(frames))
    return path


def test_is_dat_file():
    assert is_dat_file("a/Jin_14_42_03.dat") and is_dat_file("X.DAT")
    assert not is_dat_file("a.avi") and not is_dat_file("dat")


def test_header_fields(dat):
    header = read_header(dat)
    assert (header.n_frames, header.height, header.width) == (5, 6, 7)
    assert header.interval_ns == pytest.approx(1000.0)
    assert header.fps == pytest.approx(1e6)


def test_frames_round_trip_with_rows_flipped(dat, frames):
    reader = DatFrameReader(str(dat))
    try:
        assert reader.total_frames == 5 and reader.fps == pytest.approx(1e6)
        for i in range(5):
            frame = reader.read_frame(i)
            assert frame.dtype == np.uint16 and frame.flags.c_contiguous
            np.testing.assert_array_equal(frame, frames[i])
    finally:
        reader.close()


def test_fps_falls_back_to_the_interval_field(tmp_path, frames):
    path = tmp_path / "no_times.dat"
    path.write_bytes(build_dat(frames, interval_ns=200, times=False))
    assert DatFrameReader(str(path)).fps == pytest.approx(5e6)


def test_out_of_range_frame(dat):
    reader = DatFrameReader(str(dat))
    with pytest.raises(ImageLoadError):
        reader.read_frame(5)
    with pytest.raises(ImageLoadError):
        reader.read_frame(-1)


def test_closed_reader_refuses_to_read(dat):
    reader = DatFrameReader(str(dat))
    reader.close()
    reader.close()  # idempotent
    with pytest.raises(ImageLoadError):
        reader.read_frame(0)


def test_truncated_file_is_rejected(tmp_path, frames):
    path = tmp_path / "cut.dat"
    path.write_bytes(build_dat(frames, truncate=10))
    with pytest.raises(DatFormatError, match="truncated"):
        DatFrameReader(str(path))


@pytest.mark.parametrize("content", [b"", b"hello world" * 10, bytes(4096)])
def test_not_a_dat_file(tmp_path, content):
    path = tmp_path / "other.dat"
    path.write_bytes(content)
    with pytest.raises(DatFormatError):
        DatFrameReader(str(path))


def test_missing_file(tmp_path):
    with pytest.raises(ImageLoadError):
        DatFrameReader(str(tmp_path / "missing.dat"))


def test_unicode_path(tmp_path, frames):
    path = tmp_path / "µ_样本" / "shot.dat"
    path.parent.mkdir()
    path.write_bytes(build_dat(frames))
    np.testing.assert_array_equal(DatFrameReader(str(path)).read_frame(2), frames[2])


# -- the real recording ---------------------------------------------------------------

needs_sample = pytest.mark.skipif(
    not (SAMPLE.is_file() and SAMPLE_TIFFS.is_dir()), reason="local test_data/dat not present")


@needs_sample
def test_real_header():
    header = read_header(SAMPLE)
    assert (header.n_frames, header.height, header.width) == (256, 250, 400)
    assert header.fps == pytest.approx(1_000_000.0)


@needs_sample
def test_real_frames_are_bit_identical_to_the_tiff16_export():
    tiffs = sorted(SAMPLE_TIFFS.glob("*.tif*"))
    assert len(tiffs) == 256
    reader = DatFrameReader(str(SAMPLE))
    try:
        for i, tiff in enumerate(tiffs):
            expected = cv2.imread(str(tiff), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(reader.read_frame(i), expected, err_msg=tiff.name)
    finally:
        reader.close()
