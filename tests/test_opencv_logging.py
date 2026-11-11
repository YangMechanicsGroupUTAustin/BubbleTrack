"""OpenCV log level: the camera's TIFF warnings must not flood the console."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from bubbletrack.model.image_io import imread_unicode, quiet_opencv_logging

_LEVEL_ERROR = 2


def _tiff_without_strip_byte_counts(img: np.ndarray) -> bytes:
    """A valid single-strip uint16 TIFF that omits StripByteCounts (tag 279).

    libtiff reads it but warns, exactly like the lab camera's files.
    """
    h, w = img.shape
    data = img.astype("<u2").tobytes()
    tags = [  # (tag, type, count, value); type 3 = SHORT, 4 = LONG
        (256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 16), (259, 3, 1, 1),
        (262, 3, 1, 1), (273, 4, 1, 0), (277, 3, 1, 1), (278, 3, 1, h),
    ]
    data_offset = 8 + 2 + 12 * len(tags) + 4
    out = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(tags)))
    for tag, typ, count, value in tags:
        value = data_offset if tag == 273 else value
        if typ == 3:  # SHORT value, left-justified in the 4-byte field
            out += struct.pack("<HHIHH", tag, typ, count, value, 0)
        else:
            out += struct.pack("<HHII", tag, typ, count, value)
    out += struct.pack("<I", 0) + data
    return bytes(out)


def _run_python(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter; OpenCV's native stderr is piped.

    A subprocess is needed because OpenCV writes from its own C runtime,
    which pytest's fd capture cannot redirect on Windows.
    """
    env = {k: v for k, v in os.environ.items() if k != "OPENCV_LOG_LEVEL"}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
    )


def test_level_is_error_after_importing_image_io():
    out = _run_python(
        "import cv2; import bubbletrack.model.image_io; "
        "print(cv2.utils.logging.getLogLevel())"
    )
    assert out.stdout.strip() == str(_LEVEL_ERROR)


def test_missing_strip_byte_counts_warning_is_silenced(tmp_path):
    img = (np.arange(20 * 30, dtype=np.uint16).reshape(20, 30) * 50)
    path = tmp_path / "camera.tif"
    path.write_bytes(_tiff_without_strip_byte_counts(img))
    np.testing.assert_array_equal(imread_unicode(path), img)  # file is valid

    read = f"import numpy as np; cv2.imdecode(np.fromfile({str(path)!r}, np.uint8), -1)"
    plain = _run_python(f"import cv2; {read}")
    assert "StripByteCounts" in plain.stderr  # the warning is real

    quiet = _run_python(f"import cv2; import bubbletrack.model.image_io; {read}")
    assert "StripByteCounts" not in quiet.stderr


class TestNeverRaises:
    def test_falls_back_to_legacy_set_log_level(self, monkeypatch):
        monkeypatch.delenv("OPENCV_LOG_LEVEL", raising=False)
        calls = []
        fake = SimpleNamespace(setLogLevel=calls.append)
        assert quiet_opencv_logging(fake) is True
        assert calls == [_LEVEL_ERROR]

    def test_setter_that_raises_is_swallowed(self, monkeypatch):
        monkeypatch.delenv("OPENCV_LOG_LEVEL", raising=False)

        def boom(level):
            raise RuntimeError("no logging in this build")

        fake = SimpleNamespace(
            utils=SimpleNamespace(
                logging=SimpleNamespace(setLogLevel=boom, LOG_LEVEL_ERROR=_LEVEL_ERROR),
            ),
        )
        assert quiet_opencv_logging(fake) is False

    def test_no_logging_api_at_all(self, monkeypatch):
        monkeypatch.delenv("OPENCV_LOG_LEVEL", raising=False)
        assert quiet_opencv_logging(SimpleNamespace()) is False


def test_explicit_env_level_is_respected(monkeypatch):
    monkeypatch.setenv("OPENCV_LOG_LEVEL", "DEBUG")
    calls = []
    assert quiet_opencv_logging(SimpleNamespace(setLogLevel=calls.append)) is False
    assert calls == []
