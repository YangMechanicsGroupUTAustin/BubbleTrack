"""Bubble detection pipeline: expand, filter, label, crop, edge-detect."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from bubbletrack.model.constants import (
    MAX_AXIS_RATIO,
    MAX_ECCENTRICITY,
    MIN_HOLE_AREA,
)


def expand_binary_image(
    bw: np.ndarray,
    bubble_cross_edges: list[bool],
) -> np.ndarray:
    """Double the image in the first non-crossing direction, filled with True.

    Parameters
    ----------
    bw : 2-D bool array
        Binary ROI image.
    bubble_cross_edges : list of 4 bool
        ``[Top, Right, Down, Left]`` — True if the bubble crosses that edge.
    """
    h, w = bw.shape
    non_cross = [not e for e in bubble_cross_edges]
    # Find first non-crossing direction (1-indexed: 1=Top, 2=Right, 3=Down, 4=Left)
    direction = None
    for i, nc in enumerate(non_cross):
        if nc:
            direction = i + 1
            break
    if direction is None:
        direction = 1  # fallback: expand top

    if direction == 1:  # Top: original goes to bottom half
        expanded = np.ones((h * 2, w), dtype=bool)
        expanded[h:2 * h, :w] = bw
    elif direction == 2:  # Right: original goes to left half
        expanded = np.ones((h, w * 2), dtype=bool)
        expanded[:h, :w] = bw
    elif direction == 3:  # Down: original goes to top half
        expanded = np.ones((h * 2, w), dtype=bool)
        expanded[:h, :w] = bw
    elif direction == 4:  # Left: original goes to right half
        expanded = np.ones((h, w * 2), dtype=bool)
        expanded[:h, w:2 * w] = bw

    return expanded


def crop_expanded_image(
    expanded: np.ndarray,
    orig_rows: int,
    orig_cols: int,
    bubble_cross_edges: list[bool],
) -> np.ndarray:
    """Reverse :func:`expand_binary_image` — crop back to original size."""
    non_cross = [not e for e in bubble_cross_edges]
    direction = None
    for i, nc in enumerate(non_cross):
        if nc:
            direction = i + 1
            break
    if direction is None:
        direction = 1

    if direction == 1:
        return expanded[orig_rows:2 * orig_rows, :orig_cols]
    elif direction == 2:
        return expanded[:orig_rows, :orig_cols]
    elif direction == 3:
        return expanded[:orig_rows, :orig_cols]
    elif direction == 4:
        return expanded[:orig_rows, orig_cols:2 * orig_cols]


def detect_bubble(
    binary_roi: np.ndarray,
    bubble_cross_edges: list[bool],
    removing_factor: int,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    removing_obj_radius: int = 0,
    *,
    opening_radius: int = 0,
    closing_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect a bubble boundary in *binary_roi*.

    Parameters
    ----------
    binary_roi : 2-D bool array
        Binary image of the ROI (True = foreground/white).
    bubble_cross_edges : list of 4 bool
        ``[Top, Right, Down, Left]``.
    removing_factor : int
        Min connected-component area to keep.
    gridx, gridy : (start, end)
        1-indexed ROI bounds for coordinate conversion.
    removing_obj_radius : int
        Disk radius for morphological closing (0 = skip).

    Returns
    -------
    processed : 2-D bool array
        Processed binary mask (original ROI size) of the detected bubble.
    edge_xy : ndarray of shape (N, 2)
        ``[row, col]`` edge coordinates in **full-image** space (1-indexed).
    """
    bw = binary_roi.astype(bool)
    orig_h, orig_w = bw.shape

    # Expand for edge-crossing bubbles
    expanded = expand_binary_image(bw, bubble_cross_edges)

    # Remove small connected white areas
    expanded = morphology.remove_small_objects(
        expanded, min_size=removing_factor,
    )

    # Optional morphological opening (remove bead spurs on edges)
    if opening_radius > 0:
        selem = morphology.disk(opening_radius)
        expanded = morphology.opening(expanded, selem)

    # Optional morphological closing (fill small holes, smooth boundary)
    effective_closing = closing_radius if closing_radius > 0 else removing_obj_radius
    if effective_closing > 1:
        selem = morphology.disk(effective_closing)
        expanded = morphology.closing(expanded, selem)
        expanded = ndimage.binary_fill_holes(expanded)

    # Invert: bubble interior (dark in original) becomes foreground
    inverted = ~expanded
    labels = measure.label(inverted)
    props = measure.regionprops(labels)

    if not props:
        # No regions found — return empty
        processed = np.zeros((orig_h, orig_w), dtype=bool)
        return processed, np.empty((0, 2), dtype=np.float64)

    # Filter by shape: reject elongated / eccentric blobs
    short_side = min(inverted.shape)
    valid_props = []
    for p in props:
        if p.axis_major_length > MAX_AXIS_RATIO * short_side:
            continue
        if p.axis_minor_length > 0 and (
            p.axis_major_length > MAX_ECCENTRICITY * p.axis_minor_length
        ):
            continue
        valid_props.append(p)

    if not valid_props:
        # Fallback: use the largest region without filtering
        valid_props = props

    # Keep the largest blob
    largest = max(valid_props, key=lambda p: p.area)
    blob_mask = labels == largest.label

    # Remove small holes inside the blob
    holes = ~blob_mask
    holes_cleaned = morphology.remove_small_objects(holes, min_size=MIN_HOLE_AREA)
    blob_mask = ~holes_cleaned

    # Crop back to original ROI size
    processed = crop_expanded_image(blob_mask, orig_h, orig_w, bubble_cross_edges)
    processed = processed.astype(bool)

    # Edge detection via morphological gradient (dilation XOR original)
    selem_3x3 = np.ones((3, 3), dtype=bool)
    dilated = morphology.dilation(processed, selem_3x3)
    edge_mask = dilated ^ processed
    rows, cols = np.where(edge_mask)

    # Convert to full-image 1-indexed coordinates
    edge_xy = np.column_stack([
        rows + gridx[0],
        cols + gridy[0],
    ]).astype(np.float64)

    return processed, edge_xy


def detect_bubble_gradient(
    gray_roi: np.ndarray,
    sensitivity: float,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    *,
    boundary_bias: float = 0.5,
    prior_circle: tuple[float, float, float] | None = None,
    gaussian_sigma: float = 1.2,
    max_trials: int = 350,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect a bubble boundary from image gradients using RANSAC circle fit.

    This mode is intended for images where the bubble and its surrounding
    shadow are both dark, causing intensity-threshold masks to merge them.
    It thresholds the local gradient magnitude, then keeps the edge points
    that agree with the strongest circular boundary.

    ``sensitivity`` follows the UI convention: higher values keep more
    candidate edge pixels.
    """
    if gray_roi.ndim == 3:
        gray_roi = cv2.cvtColor(gray_roi, cv2.COLOR_BGR2GRAY)
    if gray_roi.size == 0:
        return (
            np.zeros(gray_roi.shape[:2], dtype=bool),
            np.empty((0, 2), dtype=np.float64),
        )

    img = np.asarray(gray_roi, dtype=np.float32)
    img_min = float(np.nanmin(img))
    img_max = float(np.nanmax(img))
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        return (
            np.zeros(img.shape, dtype=bool),
            np.empty((0, 2), dtype=np.float64),
        )

    img_8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if gaussian_sigma > 0:
        ksize = int(np.ceil(gaussian_sigma * 3)) * 2 + 1
        img_8 = cv2.GaussianBlur(img_8, (ksize, ksize), gaussian_sigma)

    radial = _radial_inner_gradient_edges(
        img,
        boundary_bias,
        gridx,
        gridy,
        gaussian_sigma=gaussian_sigma,
        prior_circle=prior_circle,
    )
    if radial is not None:
        return radial

    gx = cv2.Scharr(img_8, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_8, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)

    positive = grad[grad > 0]
    if positive.size < 3:
        return (
            np.zeros(img.shape, dtype=bool),
            np.empty((0, 2), dtype=np.float64),
        )

    # Higher sensitivity means a lower percentile cutoff and more edge pixels.
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    percentile = 97.0 - 45.0 * sens
    threshold = np.percentile(positive, percentile)
    edge_mask = grad >= threshold

    # Trim tiny speckles before RANSAC.  Canny-like gradients can be peppery.
    edge_mask = morphology.remove_small_objects(edge_mask, min_size=6)
    rows, cols = np.where(edge_mask)
    points = np.column_stack([rows, cols]).astype(np.float64)
    if points.shape[0] < 12:
        return (
            edge_mask.astype(bool),
            np.empty((0, 2), dtype=np.float64),
        )

    best = _ransac_circle(points, img.shape, max_trials=max_trials)
    if best is None:
        return (
            edge_mask.astype(bool),
            np.empty((0, 2), dtype=np.float64),
        )

    center_r, center_c, radius, inliers = best
    processed = np.zeros(img.shape, dtype=bool)
    inlier_pts = points[inliers]
    rr = np.clip(inlier_pts[:, 0].astype(int), 0, img.shape[0] - 1)
    cc = np.clip(inlier_pts[:, 1].astype(int), 0, img.shape[1] - 1)
    processed[rr, cc] = True

    # Add a thin fitted circle to the binary preview so partial inliers are
    # visually easy to judge while keeping edge points as the fit source.
    circle_overlay = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(
        circle_overlay,
        (int(round(center_c)), int(round(center_r))),
        int(round(radius)),
        1,
        1,
    )
    processed |= circle_overlay.astype(bool)

    edge_xy = np.column_stack([
        inlier_pts[:, 0] + gridx[0],
        inlier_pts[:, 1] + gridy[0],
    ]).astype(np.float64)
    return processed, edge_xy


def _radial_inner_gradient_edges(
    img: np.ndarray,
    boundary_bias: float,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    *,
    gaussian_sigma: float,
    prior_circle: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the inner bubble edge by searching radially from the dark core."""
    core = _estimate_dark_core(img)
    if core is None:
        return None
    center_r, center_c, core_radius = core
    if prior_circle is not None:
        prior_r, prior_c, prior_radius = prior_circle
        local_prior_r = prior_r - gridx[0]
        local_prior_c = prior_c - gridy[0]
        if (
            0 <= local_prior_r < img.shape[0]
            and 0 <= local_prior_c < img.shape[1]
            and prior_radius > 3
        ):
            center_r = float(local_prior_r)
            center_c = float(local_prior_c)
            core_radius = float(min(core_radius, prior_radius))
    h, w = img.shape
    short_side = min(h, w)
    if core_radius < 4 or core_radius > 0.5 * short_side:
        return None

    blur_sigma = max(1.0, gaussian_sigma)
    smooth = ndimage.gaussian_filter(img, blur_sigma)
    sens = float(np.clip(boundary_bias, 0.0, 1.0))

    roi_points = _collect_first_radial_peak_points(
        smooth,
        center_r,
        center_c,
        core_radius,
        sens,
    )
    if roi_points.shape[0] < 24:
        return None

    fit = _fit_circle_lstsq(roi_points)
    if fit is None:
        return None
    center_r, center_c, radius = fit
    if not (0.08 * short_side <= radius <= 0.72 * short_side):
        return None
    roi_points, fit = _keep_single_circle_points(
        roi_points,
        center_r,
        center_c,
        radius,
    )
    if roi_points.shape[0] < 24 or fit is None:
        return None
    center_r, center_c, radius = fit

    processed = np.zeros(img.shape, dtype=bool)
    rr = np.clip(roi_points[:, 0].astype(int), 0, h - 1)
    cc = np.clip(roi_points[:, 1].astype(int), 0, w - 1)
    processed[rr, cc] = True
    circle_overlay = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(
        circle_overlay,
        (int(round(center_c)), int(round(center_r))),
        int(round(radius)),
        1,
        1,
    )
    processed |= circle_overlay.astype(bool)

    edge_xy = np.column_stack([
        roi_points[:, 0] + gridx[0],
        roi_points[:, 1] + gridy[0],
    ]).astype(np.float64)
    return processed, edge_xy


def _keep_single_circle_points(
    points: np.ndarray,
    center_r: float,
    center_c: float,
    radius: float,
) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Keep only edge points that belong to the fitted circular boundary."""
    if points.shape[0] < 3 or radius <= 0:
        return points, (center_r, center_c, radius)

    dists = np.linalg.norm(points - np.array([center_r, center_c]), axis=1)
    residuals = np.abs(dists - radius)
    points = _keep_dominant_radius_band(points, dists)
    if points.shape[0] < 24:
        return points, (center_r, center_c, radius)
    fit = _fit_circle_lstsq(points)
    if fit is None:
        return points, None
    center_r, center_c, radius = fit
    dists = np.linalg.norm(points - np.array([center_r, center_c]), axis=1)
    residuals = np.abs(dists - radius)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    tolerance = max(2.0, min(6.0, np.median(residuals) + 2.5 * 1.4826 * mad))
    tolerance = max(tolerance, 0.035 * radius)
    keep = residuals <= tolerance
    if np.sum(keep) < 24:
        return points, (center_r, center_c, radius)

    filtered = points[keep]
    fit = _fit_circle_lstsq(filtered)
    if fit is None:
        return filtered, None

    # One more pass after refit removes a second nearby ring if both were
    # present in the initial point set.
    center_r, center_c, radius = fit
    dists = np.linalg.norm(filtered - np.array([center_r, center_c]), axis=1)
    residuals = np.abs(dists - radius)
    tolerance = max(2.0, min(5.0, np.percentile(residuals, 70)))
    tolerance = max(tolerance, 0.03 * radius)
    keep = residuals <= tolerance
    if np.sum(keep) >= 24:
        filtered = filtered[keep]
        fit = _fit_circle_lstsq(filtered) or fit
    return filtered, fit


def _keep_dominant_radius_band(points: np.ndarray, dists: np.ndarray) -> np.ndarray:
    """Keep the strongest single radial band when double rings are present."""
    if points.shape[0] < 36:
        return points
    bin_width = 2.0
    lo = float(np.min(dists))
    hi = float(np.max(dists))
    if hi - lo < 2 * bin_width:
        return points
    bins = np.arange(lo, hi + bin_width, bin_width)
    hist, edges = np.histogram(dists, bins=bins)
    if hist.size == 0 or np.max(hist) < 12:
        return points
    best = int(np.argmax(hist))
    band_lo = edges[max(0, best - 1)]
    band_hi = edges[min(len(edges) - 1, best + 2)]
    keep = (dists >= band_lo) & (dists <= band_hi)
    return points[keep] if np.sum(keep) >= 24 else points


def _collect_radial_boundary_points(
    smooth: np.ndarray,
    center_r: float,
    center_c: float,
    core_radius: float,
    boundary_level: float,
) -> np.ndarray:
    """Collect edge candidates in an annulus controlled by boundary level."""
    h, w = smooth.shape
    short_side = min(h, w)
    level = float(np.clip(boundary_level, 0.0, 1.0))
    inner_factor = 0.55 + 0.95 * level
    outer_factor = inner_factor + 0.85
    r_min = max(5.0, inner_factor * core_radius)
    r_max = min(0.72 * short_side, outer_factor * core_radius)
    if r_max <= r_min + 4:
        return np.empty((0, 2), dtype=np.float64)

    points: list[list[float]] = []
    angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    radii = np.arange(r_min, r_max, 0.5)
    for theta in angles:
        rr = center_r + np.sin(theta) * radii
        cc = center_c + np.cos(theta) * radii
        valid = (rr >= 1) & (rr < h - 2) & (cc >= 1) & (cc < w - 2)
        if np.sum(valid) < 5:
            continue
        vals = ndimage.map_coordinates(
            smooth,
            [rr[valid], cc[valid]],
            order=1,
            mode="nearest",
        )
        radial_derivative = np.gradient(vals)
        k = int(np.argmax(radial_derivative))
        min_rise = max(0.01, float(np.percentile(radial_derivative, 75)))
        if radial_derivative[k] > min_rise:
            points.append([float(rr[valid][k]), float(cc[valid][k])])
    return np.asarray(points, dtype=np.float64)


def _collect_first_radial_peak_points(
    smooth: np.ndarray,
    center_r: float,
    center_c: float,
    core_radius: float,
    boundary_bias: float,
) -> np.ndarray:
    """Collect the first outward dark-to-light gradient peak on each ray.

    The bubble images can contain two near-concentric gradient rings.  This
    function uses radial ordering rather than global edge strength: for each
    angle, it samples away from the dark core and picks the first significant
    positive derivative peak.  Moving ``boundary_bias`` right allows later
    peaks, useful when the desired boundary is intentionally the outer ring.
    """
    h, w = smooth.shape
    short_side = min(h, w)
    start_r = max(3.0, 0.25 * core_radius)
    stop_r = min(0.78 * short_side, 2.8 * core_radius)
    if stop_r <= start_r + 5:
        return np.empty((0, 2), dtype=np.float64)

    peak_order = 0
    if boundary_bias >= 0.82:
        peak_order = 2
    elif boundary_bias >= 0.48:
        peak_order = 1

    points: list[list[float]] = []
    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    radii = np.arange(start_r, stop_r, 0.5)
    for theta in angles:
        rr = center_r + np.sin(theta) * radii
        cc = center_c + np.cos(theta) * radii
        valid = (rr >= 1) & (rr < h - 2) & (cc >= 1) & (cc < w - 2)
        if np.sum(valid) < 8:
            continue

        vals = ndimage.map_coordinates(
            smooth,
            [rr[valid], cc[valid]],
            order=1,
            mode="nearest",
        )
        deriv = np.gradient(vals)
        if deriv.size < 5:
            continue

        floor = max(0.006, float(np.percentile(deriv, 70)))
        local_max = (
            (deriv[1:-1] > deriv[:-2])
            & (deriv[1:-1] >= deriv[2:])
            & (deriv[1:-1] > floor)
        )
        peak_idx = np.where(local_max)[0] + 1
        if peak_idx.size == 0:
            continue

        # Ignore tiny wiggles very close to the centre unless no other peak
        # exists.  The first real boundary should also have meaningful rise.
        strong = peak_idx[deriv[peak_idx] >= max(floor, 0.35 * np.max(deriv))]
        if strong.size > 0:
            peak_idx = strong

        chosen = peak_idx[min(peak_order, peak_idx.size - 1)]
        points.append([float(rr[valid][chosen]), float(cc[valid][chosen])])

    return np.asarray(points, dtype=np.float64)


def _score_gradient_circle(
    img: np.ndarray,
    points: np.ndarray,
    center_r: float,
    center_c: float,
    radius: float,
    boundary_level: float,
    boundary_bias: float,
    *,
    prior_circle: tuple[float, float, float] | None = None,
    gridx: tuple[int, int] | None = None,
    gridy: tuple[int, int] | None = None,
) -> float:
    """Score a candidate by circularity, contrast, coverage, and UI preference."""
    h, w = img.shape
    rr, cc = np.ogrid[:h, :w]
    dist = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)

    inner = dist <= max(1.0, radius * 0.82)
    outer = (dist >= radius * 1.05) & (dist <= radius * 1.28)
    if np.sum(inner) < 12 or np.sum(outer) < 12:
        return -np.inf

    inside_mean = float(np.mean(img[inner]))
    outside_mean = float(np.mean(img[outer]))
    inside_std = float(np.std(img[inner]))
    outside_std = float(np.std(img[outer]))
    pooled = max(0.02, 0.5 * (inside_std + outside_std))
    contrast = (outside_mean - inside_mean) / pooled
    contrast_score = 1.0 / (1.0 + np.exp(-contrast))

    dists = np.linalg.norm(points - np.array([center_r, center_c]), axis=1)
    residual = float(np.sqrt(np.mean((dists - radius) ** 2)))
    circularity_score = np.exp(-residual / max(2.0, 0.08 * radius))

    coverage = _sector_coverage(points, center_r, center_c)
    preference = np.exp(-abs(boundary_level - boundary_bias) / 0.35)
    prior_score = 0.5
    if prior_circle is not None and gridx is not None and gridy is not None:
        prior_r, prior_c, prior_radius = prior_circle
        full_r = center_r + gridx[0]
        full_c = center_c + gridy[0]
        center_delta = np.hypot(full_r - prior_r, full_c - prior_c)
        radius_delta = abs(radius - prior_radius)
        center_scale = max(4.0, 0.35 * prior_radius)
        radius_scale = max(4.0, 0.40 * prior_radius)
        prior_score = np.exp(
            -center_delta / center_scale - radius_delta / radius_scale
        )

    return (
        0.25 * contrast_score
        + 0.18 * circularity_score
        + 0.12 * coverage
        + 0.25 * preference
        + 0.20 * prior_score
    )


def _estimate_dark_core(img: np.ndarray) -> tuple[float, float, float] | None:
    """Estimate dark bubble core centre and equivalent radius."""
    cutoff = np.percentile(img, 20)
    mask = img <= cutoff
    min_size = max(30, int(img.size * 0.002))
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.closing(mask, morphology.disk(2))
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return None

    h, w = img.shape
    margin_r = max(2, int(0.03 * h))
    margin_c = max(2, int(0.03 * w))
    candidates = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        touches = (
            minr <= margin_r or minc <= margin_c
            or maxr >= h - margin_r or maxc >= w - margin_c
        )
        score = prop.area * (0.35 if touches else 1.0)
        candidates.append((score, prop))
    prop = max(candidates, key=lambda item: item[0])[1]
    center_r, center_c = prop.centroid
    core_radius = float(np.sqrt(prop.area / np.pi))
    return float(center_r), float(center_c), core_radius


def _fit_circle_lstsq(
    points: np.ndarray,
) -> tuple[float, float, float] | None:
    """Least-squares circle fit for ``[row, col]`` points."""
    if points.shape[0] < 3:
        return None
    rows = points[:, 0]
    cols = points[:, 1]
    a = np.column_stack([2 * cols, 2 * rows, np.ones_like(rows)])
    b = cols ** 2 + rows ** 2
    try:
        center_c, center_r, constant = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = constant + center_c ** 2 + center_r ** 2
    if radius_sq <= 0:
        return None
    return float(center_r), float(center_c), float(np.sqrt(radius_sq))


def detect_bubble_with_mode(
    binary_roi: np.ndarray,
    gray_roi: np.ndarray,
    mode: str,
    sensitivity: float,
    bubble_cross_edges: list[bool],
    removing_factor: int,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    removing_obj_radius: int = 0,
    *,
    opening_radius: int = 0,
    closing_radius: int = 0,
    gaussian_sigma: float = 0.0,
    gradient_boundary: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the selected detection mode."""
    if mode == "gradient":
        auto_roi = _gradient_roi_from_intensity(
            binary_roi,
            bubble_cross_edges,
            removing_factor,
            gridx,
            gridy,
            removing_obj_radius,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
        )
        if auto_roi is not None:
            r_slice, c_slice, prior_circle, prior_processed, prior_edge_xy = auto_roi
            sub_gridx = (gridx[0] + r_slice.start, gridx[0] + r_slice.stop - 1)
            sub_gridy = (gridy[0] + c_slice.start, gridy[0] + c_slice.stop - 1)
            sub_processed, edge_xy = detect_bubble_gradient(
                gray_roi[r_slice, c_slice],
                sensitivity,
                sub_gridx,
                sub_gridy,
                boundary_bias=gradient_boundary,
                prior_circle=prior_circle,
                gaussian_sigma=gaussian_sigma if gaussian_sigma > 0 else 1.2,
            )
            edge_xy = _constrain_edge_xy_to_prior(
                edge_xy,
                prior_circle,
                gradient_boundary,
            )
            if edge_xy.shape[0] >= 3:
                processed = np.zeros_like(binary_roi, dtype=bool)
                processed = _edge_xy_to_roi_mask(edge_xy, binary_roi.shape, gridx, gridy)
                return processed, edge_xy
            if prior_edge_xy.shape[0] >= 3:
                return prior_processed, prior_edge_xy
        return detect_bubble_gradient(
            gray_roi,
            sensitivity,
            gridx,
            gridy,
            boundary_bias=gradient_boundary,
            gaussian_sigma=gaussian_sigma if gaussian_sigma > 0 else 1.2,
        )
    return detect_bubble(
        binary_roi,
        bubble_cross_edges,
        removing_factor,
        gridx,
        gridy,
        removing_obj_radius,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
    )


def _constrain_edge_xy_to_prior(
    edge_xy: np.ndarray,
    prior_circle: tuple[float, float, float],
    boundary_bias: float,
) -> np.ndarray:
    """Hard-filter gradient edge points to stay near the intensity prior circle."""
    if edge_xy.shape[0] < 3:
        return edge_xy
    prior_r, prior_c, prior_radius = prior_circle
    if prior_radius <= 0:
        return edge_xy

    dists = np.linalg.norm(edge_xy - np.array([prior_r, prior_c]), axis=1)
    # The contrast prior is mainly a centre/location prior.  Its radius can
    # correspond to the outer halo, while the desired gradient boundary can be
    # the smaller first inner ring.
    min_radius = max(3.0, prior_radius * (0.35 + 0.25 * boundary_bias))
    max_radius = prior_radius * (1.20 + 0.40 * boundary_bias)
    keep = (
        (dists >= min_radius)
        & (dists <= max_radius)
    )
    if np.sum(keep) < 36:
        return np.empty((0, 2), dtype=np.float64)

    filtered = edge_xy[keep]
    if _sector_coverage(filtered, prior_r, prior_c) < 0.50:
        return np.empty((0, 2), dtype=np.float64)

    fit = _fit_circle_lstsq(filtered)
    if fit is None:
        return filtered
    center_r, center_c, radius = fit

    center_delta = np.hypot(center_r - prior_r, center_c - prior_c)
    if center_delta > max(4.0, 0.30 * prior_radius):
        return np.empty((0, 2), dtype=np.float64)
    if radius > max_radius:
        return np.empty((0, 2), dtype=np.float64)

    dists = np.linalg.norm(filtered - np.array([center_r, center_c]), axis=1)
    residuals = np.abs(dists - radius)
    tol = max(2.0, min(5.0, np.percentile(residuals, 70)))
    final = filtered[residuals <= tol]
    if final.shape[0] >= 36 and _sector_coverage(final, center_r, center_c) >= 0.50:
        return final
    if filtered.shape[0] >= 36 and _sector_coverage(filtered, center_r, center_c) >= 0.50:
        return filtered
    return np.empty((0, 2), dtype=np.float64)


def _edge_xy_to_roi_mask(
    edge_xy: np.ndarray,
    shape: tuple[int, int],
    gridx: tuple[int, int],
    gridy: tuple[int, int],
) -> np.ndarray:
    """Render full-image edge coordinates into an ROI-sized binary mask."""
    mask = np.zeros(shape, dtype=bool)
    rows = np.rint(edge_xy[:, 0] - gridx[0]).astype(int)
    cols = np.rint(edge_xy[:, 1] - gridy[0]).astype(int)
    valid = (
        (rows >= 0) & (rows < shape[0])
        & (cols >= 0) & (cols < shape[1])
    )
    mask[rows[valid], cols[valid]] = True
    return mask


def _gradient_roi_from_intensity(
    binary_roi: np.ndarray,
    bubble_cross_edges: list[bool],
    removing_factor: int,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    removing_obj_radius: int = 0,
    *,
    opening_radius: int = 0,
    closing_radius: int = 0,
) -> tuple[
    slice,
    slice,
    tuple[float, float, float],
    np.ndarray,
    np.ndarray,
] | None:
    """Use the intensity-mask result as an automatic search ROI for gradient."""
    try:
        processed, edge_xy = detect_bubble(
            binary_roi,
            bubble_cross_edges,
            removing_factor,
            gridx,
            gridy,
            removing_obj_radius,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
        )
    except Exception:
        return None

    rows, cols = np.where(processed)
    if rows.size < 20 and edge_xy.shape[0] >= 20:
        rows = (edge_xy[:, 0] - gridx[0]).astype(int)
        cols = (edge_xy[:, 1] - gridy[0]).astype(int)
    if rows.size < 20:
        return None

    h, w = binary_roi.shape
    prior_circle = None
    if edge_xy.shape[0] >= 3:
        prior_circle = _fit_circle_lstsq(edge_xy)
    if prior_circle is None:
        full_points = np.column_stack([
            rows + gridx[0],
            cols + gridy[0],
        ]).astype(np.float64)
        prior_circle = _fit_circle_lstsq(full_points)
    if prior_circle is None:
        return None

    r0, r1 = int(np.min(rows)), int(np.max(rows)) + 1
    c0, c1 = int(np.min(cols)), int(np.max(cols)) + 1
    box_h = r1 - r0
    box_w = c1 - c0
    if box_h < 8 or box_w < 8:
        return None

    pad = int(max(8, 0.35 * max(box_h, box_w)))
    r0 = max(0, r0 - pad)
    r1 = min(h, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(w, c1 + pad)

    # If the intensity result covers almost the whole manual ROI, it is not a
    # useful refinement. Fall back to the user ROI in that case.
    if (r1 - r0) * (c1 - c0) > 0.85 * h * w:
        return None
    return slice(r0, r1), slice(c0, c1), prior_circle, processed, edge_xy


def _ransac_circle(
    points: np.ndarray,
    shape: tuple[int, int],
    *,
    max_trials: int,
) -> tuple[float, float, float, np.ndarray] | None:
    """Return the best circle model and inlier mask for edge points."""
    h, w = shape
    short_side = min(h, w)
    min_radius = max(6.0, 0.08 * short_side)
    max_radius = 0.60 * short_side
    rng = np.random.default_rng(12345)
    n = points.shape[0]
    if n < 3:
        return None

    best_score = -np.inf
    best_model = None
    sample_n = min(n, 1800)
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        fit_points = points[idx]
    else:
        fit_points = points

    for _ in range(max_trials):
        tri_idx = rng.choice(fit_points.shape[0], size=3, replace=False)
        model = _circle_from_three_points(fit_points[tri_idx])
        if model is None:
            continue
        center_r, center_c, radius = model
        if not (min_radius <= radius <= max_radius):
            continue
        if center_r < -0.25 * h or center_r > 1.25 * h:
            continue
        if center_c < -0.25 * w or center_c > 1.25 * w:
            continue

        dists = np.linalg.norm(fit_points - np.array([center_r, center_c]), axis=1)
        tolerance = max(2.5, 0.035 * radius)
        inliers = np.abs(dists - radius) <= tolerance
        inlier_count = int(np.sum(inliers))
        if inlier_count < max(20, 0.08 * fit_points.shape[0]):
            continue

        coverage = _sector_coverage(fit_points[inliers], center_r, center_c)
        score = inlier_count * (0.5 + coverage)
        if score > best_score:
            best_score = score
            best_model = (center_r, center_c, radius)

    if best_model is None:
        return None

    center_r, center_c, radius = best_model
    all_dists = np.linalg.norm(points - np.array([center_r, center_c]), axis=1)
    tolerance = max(2.5, 0.04 * radius)
    inliers = np.abs(all_dists - radius) <= tolerance
    if np.sum(inliers) < 12:
        return None
    return center_r, center_c, radius, inliers


def _circle_from_three_points(
    pts: np.ndarray,
) -> tuple[float, float, float] | None:
    """Fit the unique circle through three ``[row, col]`` points."""
    p1, p2, p3 = pts
    a = np.array([
        [2 * (p2[1] - p1[1]), 2 * (p2[0] - p1[0])],
        [2 * (p3[1] - p1[1]), 2 * (p3[0] - p1[0])],
    ])
    b = np.array([
        p2[1] ** 2 + p2[0] ** 2 - p1[1] ** 2 - p1[0] ** 2,
        p3[1] ** 2 + p3[0] ** 2 - p1[1] ** 2 - p1[0] ** 2,
    ])
    det = np.linalg.det(a)
    if abs(det) < 1e-6:
        return None
    center_c, center_r = np.linalg.solve(a, b)
    radius = float(np.linalg.norm(p1 - np.array([center_r, center_c])))
    return float(center_r), float(center_c), radius


def _sector_coverage(points: np.ndarray, center_r: float, center_c: float) -> float:
    """Fraction of 12 angular sectors touched by candidate edge points."""
    if points.size == 0:
        return 0.0
    angles = np.arctan2(points[:, 0] - center_r, points[:, 1] - center_c)
    sectors = np.floor((angles + np.pi) / (2 * np.pi) * 12).astype(int)
    sectors = np.clip(sectors, 0, 11)
    return len(set(sectors.tolist())) / 12.0
