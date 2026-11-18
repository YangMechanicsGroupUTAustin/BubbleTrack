"""Write a file so that a failure part-way never leaves it truncated.

The text goes to a temporary file next to the target (unique per process, so
two running apps never share it) and replaces the target in one rename.  A
crash, a full disk or a sync client holding the file then leaves the previous
version intact.  Sync clients (OneDrive) briefly lock files they upload, so a
refused rename is retried a few times.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator

from bubbletrack.model.long_paths import os_path

_REPLACE_ATTEMPTS = 5
_REPLACE_WAIT_S = 0.2


def write_text_atomic(path: str | os.PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically; raises OSError when it cannot be written."""
    target = os_path(path)
    tmp = f"{target}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


@contextlib.contextmanager
def replacing(path: str | os.PathLike) -> Iterator[str]:
    """Write through a temporary file that replaces *path* only when the block succeeds.

    For writers that take a path (savemat, openpyxl, fpdf2): an interrupted export
    leaves the previous file intact.  The temporary name keeps the extension, which
    some writers check or append.
    """
    target = os_path(path)
    root, ext = os.path.splitext(target)
    tmp = f"{root}.{os.getpid()}.tmp{ext}"
    try:
        yield tmp
        _replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _replace(src: str, dst: str) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if attempt == _REPLACE_ATTEMPTS - 1:
                # Name the user's file, not the temporary one the OS error quotes
                raise PermissionError(
                    f"{os.path.basename(dst)} is open in another program "
                    "(close it there and try again)") from exc
            time.sleep(_REPLACE_WAIT_S)
