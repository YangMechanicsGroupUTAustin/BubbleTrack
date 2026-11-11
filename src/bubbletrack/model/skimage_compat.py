"""Version shims for scikit-image APIs whose signatures changed.

scikit-image 0.26 deprecated ``morphology.remove_small_objects(min_size=N)``
(removed in 2.0) in favour of ``max_size=M``.  The two differ by one pixel:

* old ``min_size=N`` removes objects with **fewer than** N pixels;
* new ``max_size=M`` removes objects with **at most** M pixels.

Detection results are pinned bit-for-bit, so every caller goes through
:func:`remove_small_objects_compat`, which keeps the old semantics on both
versions.
"""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
from skimage import morphology


def _accepts_max_size(func: Callable) -> bool:
    """Return True if *func* takes a ``max_size`` keyword argument."""
    try:
        return "max_size" in inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins / C functions without a signature
        return False


def remove_small_objects_compat(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components with fewer than *min_size* pixels.

    Parameters
    ----------
    mask : ndarray of bool
        Binary image.  It is not modified.
    min_size : int
        Smallest component size (pixels) that survives.  An object of exactly
        *min_size* pixels is kept, as with the pre-0.26 ``min_size`` argument.

    Returns
    -------
    ndarray of bool
        A new array with the small components removed.
    """
    func = morphology.remove_small_objects
    if _accepts_max_size(func):
        # max_size removes objects <= max_size, so N - 1 keeps objects of N px.
        # Clamp at 0: min_size <= 1 removes nothing under both semantics.
        return func(mask, max_size=max(int(min_size) - 1, 0))
    return func(mask, min_size=min_size)
