"""Paths longer than Windows' MAX_PATH (260 characters).

Unless the registry switch ``LongPathsEnabled`` is on, Win32 file calls
(``open``, ``os.scandir``, OpenCV's video backends) fail on longer paths.
The extended-length prefix ``\\\\?\\`` lifts the limit; it must be put on an
absolute, normalised path.  The app keeps plain paths everywhere (state,
sessions, file labels) and converts them with :func:`os_path` only at the
OS call.
"""

from __future__ import annotations

import os
import sys

MAX_PATH = 260
# Directories fail sooner (CreateDirectory leaves room for an 8.3 file name).
_LIMIT = 248
_PREFIX = "\\\\?\\"
_UNC_PREFIX = _PREFIX + "UNC\\"
_WINDOWS = sys.platform == "win32"


def os_path(path: str | os.PathLike) -> str:
    """*path* in a form the OS opens even when it is longer than MAX_PATH.

    On Windows a path of 248 characters or more becomes an absolute,
    normalised ``\\\\?\\C:\\...`` (or ``\\\\?\\UNC\\server\\share\\...``) path;
    shorter paths, already prefixed ones and other platforms are unchanged.
    """
    text = os.fspath(path)
    if not _WINDOWS or text.startswith(_PREFIX):
        return text
    full = text if os.path.isabs(text) and len(text) < _LIMIT else os.path.abspath(text)
    if len(full) < _LIMIT:
        return text
    if full.startswith("\\\\"):
        return _UNC_PREFIX + full[2:]
    return _PREFIX + full


def too_long(path: str | os.PathLike) -> bool:
    """True when *path* is beyond MAX_PATH on Windows (plain Win32 calls fail)."""
    return _WINDOWS and len(os.path.abspath(os.fspath(path))) >= MAX_PATH


def plain(path: str) -> str:
    """*path* without the extended-length prefix (for messages and the app)."""
    if path.startswith(_UNC_PREFIX):
        return "\\\\" + path[len(_UNC_PREFIX):]
    if path.startswith(_PREFIX):
        return path[len(_PREFIX):]
    return path
