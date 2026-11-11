"""Automatic parameter tuning via coarse-to-fine grid search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.constants import (
    AUTOTUNE_RF_COARSE,
    AUTOTUNE_RF_FINE_STEP,
    AUTOTUNE_THR_COARSE,
    AUTOTUNE_THR_FINE_STEP,
    AUTOTUNE_TOP_K,
)
from bubbletrack.model.detection import detect_bubble
from bubbletrack.model.frame_source import normalize_source
from bubbletrack.model.quality import FitQuality, compute_fit_quality, score_failed_fit
from bubbletrack.model.removing_factor import compute_removing_factor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoTuneResult:
    """Result of automatic parameter tuning."""

    threshold: float          # Best img_thr [0, 1]
    removing_factor: int      # Best RF slider value [0, 100]
    quality: FitQuality       # Quality metrics at best params
    candidates_evaluated: int # Total combinations tested


def run_autotune(
    image_path: str | np.ndarray,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    bubble_cross_edges: tuple[bool, ...],
    *,
    gaussian_sigma: float = 0.0,
    clahe_clip: float = 0.0,
    opening_radius: int = 0,
    closing_radius: int = 0,
    max_radius: float = float("inf"),
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> AutoTuneResult | None:
    """Find optimal (threshold, removing_factor) via coarse-to-fine grid search.

    Parameters
    ----------
    image_path : Path to the image file to tune on.
    gridx, gridy : 1-indexed ROI bounds.
    bubble_cross_edges : Edge crossing flags.
    gaussian_sigma, clahe_clip, opening_radius, closing_radius :
        Fixed advanced filter values (not searched).
    max_radius : Upper bound for plausible radius.
    progress_callback : Called with (current_step, total_steps).
    cancel_check : Returns True if user cancelled.

    Returns
    -------
    AutoTuneResult or None if cancelled / all candidates fail.
    """
    roi_h = gridx[1] - gridx[0] + 1
    roi_w = gridy[1] - gridy[0] + 1
    roi_short_side = min(roi_h, roi_w)

    coarse_total = len(AUTOTUNE_THR_COARSE) * len(AUTOTUNE_RF_COARSE)
    # Fine search: 5x5 grid per candidate minus duplicates (upper bound)
    fine_total = AUTOTUNE_TOP_K * (25 - 1)
    total_steps = coarse_total + fine_total
    step = 0

    def _report() -> None:
        if progress_callback:
            progress_callback(step, total_steps)

    def _cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    # ---- Phase 1: Coarse search ----
    coarse_results: list[tuple[float, int, FitQuality]] = []
    # Cache binary ROI per threshold to avoid redundant load_and_normalize
    # when the fine search revisits the same threshold value.
    _binary_cache: dict[float, np.ndarray] = {}

    for thr in AUTOTUNE_THR_COARSE:
        if _cancelled():
            return None

        # One load_and_normalize per threshold value
        try:
            _, _, _, binary_roi = normalize_source(
                image_path, thr, gridx, gridy,
                gaussian_sigma=gaussian_sigma,
                clahe_clip=clahe_clip,
            )
            _binary_cache[round(thr, 4)] = binary_roi
        except Exception:
            logger.debug("autotune: load failed for thr=%.2f", thr, exc_info=True)
            step += len(AUTOTUNE_RF_COARSE)
            _report()
            continue

        for rf_slider in AUTOTUNE_RF_COARSE:
            if _cancelled():
                return None

            quality = _evaluate_params(
                binary_roi, bubble_cross_edges, rf_slider,
                gridx, gridy, opening_radius, closing_radius,
                max_radius, roi_short_side,
            )
            coarse_results.append((thr, rf_slider, quality))
            step += 1
            _report()

    if not coarse_results:
        return None

    # Sort by score descending, take top K
    coarse_results.sort(key=lambda x: x[2].score, reverse=True)
    top_candidates = coarse_results[:AUTOTUNE_TOP_K]

    # ---- Phase 2: Fine search around top candidates ----
    fine_results: list[tuple[float, int, FitQuality]] = []
    seen: set[tuple[float, int]] = set()
    # Mark all coarse combos as seen
    for thr, rf, _ in coarse_results:
        seen.add((thr, rf))

    for base_thr, base_rf, _ in top_candidates:
        fine_thrs = [
            round(base_thr + i * AUTOTUNE_THR_FINE_STEP, 2)
            for i in range(-2, 3)
        ]
        fine_rfs = [
            base_rf + i * AUTOTUNE_RF_FINE_STEP
            for i in range(-2, 3)
        ]

        for thr in fine_thrs:
            if thr < 0.05 or thr > 0.95:
                step += len(fine_rfs)
                _report()
                continue
            if _cancelled():
                return None

            # Cache binary_roi per threshold (shared across all RF values)
            thr_key = round(thr, 4)
            if thr_key in _binary_cache:
                binary_roi = _binary_cache[thr_key]
            else:
                try:
                    _, _, _, binary_roi = normalize_source(
                        image_path, thr, gridx, gridy,
                        gaussian_sigma=gaussian_sigma,
                        clahe_clip=clahe_clip,
                    )
                    _binary_cache[thr_key] = binary_roi
                except Exception:
                    logger.debug("autotune: load failed for thr=%.2f (fine)", thr, exc_info=True)
                    step += len(fine_rfs)
                    _report()
                    continue

            for rf_slider in fine_rfs:
                if _cancelled():
                    return None
                rf_slider = max(0, min(100, rf_slider))
                key = (thr, rf_slider)
                if key in seen:
                    step += 1
                    _report()
                    continue
                seen.add(key)

                quality = _evaluate_params(
                    binary_roi, bubble_cross_edges, rf_slider,
                    gridx, gridy, opening_radius, closing_radius,
                    max_radius, roi_short_side,
                )
                fine_results.append((thr, rf_slider, quality))
                step += 1
                _report()

    # Combine coarse + fine, pick best
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda x: x[2].score, reverse=True)

    best_thr, best_rf, best_quality = all_results[0]
    if best_quality.score <= 0:
        return None

    return AutoTuneResult(
        threshold=best_thr,
        removing_factor=best_rf,
        quality=best_quality,
        candidates_evaluated=len(seen),
    )


def _evaluate_params(
    binary_roi: np.ndarray,
    bubble_cross_edges: tuple[bool, ...],
    rf_slider: int,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    opening_radius: int,
    closing_radius: int,
    max_radius: float,
    roi_short_side: int,
) -> FitQuality:
    """Run detect + fit + score for one parameter combination."""
    try:
        rf = compute_removing_factor(rf_slider, gridx, gridy)
        _, edge_xy = detect_bubble(
            binary_roi, list(bubble_cross_edges), rf,
            gridx, gridy, 0,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
        )

        if edge_xy.shape[0] < 3:
            return score_failed_fit()

        rc, cc, radius = circle_fit_taubin(edge_xy)

        if not np.isfinite(radius) or radius <= 0 or radius > max_radius:
            return score_failed_fit()

        return compute_fit_quality(edge_xy, rc, cc, radius, roi_short_side)

    except Exception:
        logger.debug("_evaluate_params failed for rf=%d", rf_slider, exc_info=True)
        return score_failed_fit()
