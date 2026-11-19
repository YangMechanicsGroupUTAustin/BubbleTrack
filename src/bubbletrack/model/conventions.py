"""Index convention helpers.

Convention summary:
- Frame indices: 0-based internally, 1-based in UI display
- ROI bounds (gridx, gridy): 1-based inclusive (MATLAB convention)
  - gridx = (row_min, row_max), gridy = (col_min, col_max)
  - To slice NumPy arrays: img[gridx[0]-1 : gridx[1], gridy[0]-1 : gridy[1]]
- Edge coordinates from detection: 1-based (row, col) in full image space
- Circle fit input/output: same coordinate space as edge coordinates
- Pixel k (1-based) has its centre at k.  A Qt scene puts it over [k - 1, k),
  centre k - 0.5: scene -> 1-based is +0.5 (``scene_to_image``), 1-based ->
  scene -0.5 (``image_to_scene``).  Clicks are converted where they are
  stored, overlays by the image panel's ``draw_*``.
"""

from __future__ import annotations


def clamp_roi(
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    img_height: int,
    img_width: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Clamp 1-based inclusive ROI to image dimensions.

    Ensures min >= 1, max <= dimension, and min <= max.
    """
    rx = (max(1, min(gridx[0], img_height)), max(1, min(gridx[1], img_height)))
    ry = (max(1, min(gridy[0], img_width)), max(1, min(gridy[1], img_width)))
    # Ensure min <= max
    rx = (min(rx[0], rx[1]), max(rx[0], rx[1]))
    ry = (min(ry[0], ry[1]), max(ry[0], ry[1]))
    return rx, ry


def roi_to_slice(
    gridx: tuple[int, int], gridy: tuple[int, int]
) -> tuple[slice, slice]:
    """Convert 1-based inclusive ROI bounds to NumPy slices.

    Parameters
    ----------
    gridx : (row_min, row_max)
        1-based inclusive row bounds.
    gridy : (col_min, col_max)
        1-based inclusive column bounds.

    Returns
    -------
    (row_slice, col_slice)
        Slices suitable for ``img[row_slice, col_slice]``.
    """
    return slice(gridx[0] - 1, gridx[1]), slice(gridy[0] - 1, gridy[1])


PIXEL_CENTRE = 0.5
"""Scene coordinate of a pixel's centre within the pixel (scene = 1-based - 0.5)."""


def scene_to_image(value: float) -> float:
    """A Qt scene coordinate as a 1-based image coordinate (pixel k's centre = k)."""
    return value + PIXEL_CENTRE


def image_to_scene(value: float) -> float:
    """A 1-based image coordinate as a Qt scene coordinate (pixel k spans [k - 1, k))."""
    return value - PIXEL_CENTRE


def frame_to_display(idx: int) -> int:
    """Convert 0-based frame index to 1-based display number."""
    return idx + 1


def display_to_frame(display_num: int) -> int:
    """Convert 1-based display number to 0-based frame index."""
    return display_num - 1
