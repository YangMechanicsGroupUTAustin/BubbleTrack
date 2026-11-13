"""Tunable constants of the Edge engine, grouped per pipeline stage.

Every number the engine uses lives here, so a value can be found, documented
and changed in one place.  The defaults reproduce the validated research
prototype (``proto3``: P2 plus the saturated-speckle fix).  All classes are
frozen: build a modified copy with :func:`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizeConfig:
    """Stage A: saturation mask, background surface, reference normalisation."""

    sat_peak_fraction: float = 0.75
    """Look for saturation only when the frame peak reaches this fraction of full scale."""
    sat_margin_fraction: float = 1.0 / 4096.0
    """Pixels within this fraction of full scale below the peak count as saturated."""
    sat_min_pixels: int = 3
    """Fewer saturated pixels than this in total means no saturation at all."""
    sat_min_blob_px: int = 20
    """Only saturated blobs of at least this many pixels are glare; smaller ones are speckle."""

    bg_downsample: int = 4
    """The background surface is fitted on an image shrunk by this factor."""
    bg_poly_degree: int = 3
    bg_iterations: int = 5
    bg_dark_start: float = 0.5
    """Pixels darker than this fraction of the median are excluded from the first fit."""
    bg_dark_k: float = 4.0
    bg_dark_floor: float = 0.015
    bg_bright_k: float = 6.0
    bg_bright_floor: float = 0.10
    bg_sigma: float = 20.0
    """Gaussian scale (full-resolution px) of the masked local correction."""
    bg_min_weight: float = 0.3
    """Minimum local weight of kept pixels for the local correction to apply."""

    ref_blur_sigma: float = 1.0
    ref_invalid_level: float = 0.12
    """Reference pixels darker than this fraction of its median carry no usable light."""
    ref_invalid_dilate_px: int = 5
    gain_min_pixels: int = 100
    """Minimum valid pixels to estimate the frame-to-reference gain (flicker)."""

    noise_bright_level: float = 0.7
    noise_min_pixels: int = 100


@dataclass(frozen=True)
class SegmentConfig:
    """Stages B and E: coarse mask, hole filling, candidate scoring and selection."""

    contrast_blur_sigma: float = 3.0
    contrast_min: float = 0.01
    seg_level: float = 0.5
    """Coarse mask threshold, as a fraction of the frame contrast."""
    smooth_sigmas: tuple[float, ...] = (0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    smooth_fallback: float = 6.0
    texture_target: float = 1.0 / 8.0
    """Smallest smoothing whose residual background texture is <= contrast * this."""
    texture_bg_level: float = 0.35
    texture_subsample: int = 3
    texture_min_pixels: int = 20

    min_area_px: int = 6
    max_candidates: int = 10
    props_factor: int = 3
    """Components examined = max_candidates * props_factor, largest first."""
    min_boundary_pts: int = 5
    robust_fit_min_pts: int = 60
    min_radius_px: float = 1.5
    max_radius_factor: float = 1.2
    """Candidate radius must be below this multiple of the longer ROI side."""
    rms_clip_frac: float = 0.3
    circularity_scale: float = 0.06
    band_fraction: float = 0.95
    """Components spanning this fraction of the ROI height or width are bands (walls)."""
    band_penalty: float = 0.1
    polarity_margin_px: float = 2.0
    polarity_min_inside: float = 0.6
    """At least this fraction of a candidate must lie inside its own circle."""
    annulus_factors: tuple[float, ...] = (1.08, 1.2, 1.32)
    annulus_offset_px: float = 3.0
    annulus_samples: int = 72
    annulus_min_samples: int = 12
    invalid_margin_px: int = 5
    prior_centre_frac: float = 0.5
    prior_centre_px: float = 5.0
    prior_radius_frac: float = 0.3
    prior_radius_px: float = 5.0
    rng_seed: int = 0


@dataclass(frozen=True)
class RayConfig:
    """Stage C: radial sub-pixel edges, and the outer extent (definition B)."""

    min_rays: int = 90
    max_rays: int = 720
    step_px: float = 0.5
    init_window_min: float = 6.0
    init_window_frac: float = 0.35
    init_window_pad: float = 3.0
    refine_window_min: float = 4.0
    refine_window_frac: float = 0.12
    refine_window_pad: float = 2.0
    refine_passes: int = 1
    min_samples: int = 12
    amplitude_half_px: float = 2.0
    plateau_px: float = 4.0
    min_step_frac: float = 0.25
    """A step must rise by at least this fraction of the frame contrast."""
    noise_step_k: float = 4.0 / 3.0
    cluster_gap_px: float = 2.0
    edge_blur_min: float = 0.7
    edge_blur_frac: float = 0.6

    outer_rays: int = 360
    outer_step_px: float = 0.25
    outer_blur_sigma: float = 1.0
    outer_level: float = 0.9
    """Definition B: first radius where the profile recovers this fraction of the step."""
    outer_inner_min: float = 12.0
    outer_inner_frac: float = 0.45
    outer_reach_min: float = 25.0
    outer_reach_frac: float = 0.6
    outer_dark_min: float = 6.0
    outer_dark_frac: float = 0.25
    outer_dark_gap: float = 2.0
    outer_bg_min: float = 15.0
    outer_bg_frac: float = 0.4
    outer_dark_pct: float = 20.0
    outer_bg_pct: float = 75.0
    outer_min_dark_samples: int = 4
    outer_min_bg_samples: int = 8
    outer_min_contrast: float = 0.1
    outer_min_points: int = 12


@dataclass(frozen=True)
class FitConfig:
    """Stage D: RANSAC / Taubin initialisation, Tukey inlier selection, geometric LSQ (the robust
    fit that finds the bubble); and the default definition's circle through its outline."""

    ransac_iterations: int = 150
    ransac_tol_px: float = 1.5
    ransac_rmax_factor: float = 3.0
    ransac_rmax_pad: float = 10.0
    robust_min_pts: int = 8
    irls_iterations: int = 12
    lsq_iterations: int = 8
    tukey_c: float = 4.685
    min_scale_px: float = 0.25
    converge_tol: float = 1e-4
    lsq_min_inliers: int = 5
    centre_jump_px: float = 5.0
    centre_jump_frac: float = 0.5
    radius_ratio_max: float = 2.0
    inlier_weight: float = 0.1
    arc_bins: int = 36
    rng_seed: int = 12345
    outline_min_points: int = 20
    """The default radius definition's circle (Taubin through the genuine outline points, one
    trimming pass: ``fit.outline_circle``) needs this many points; with fewer, or when that fit
    fails, the robust ray fit is reported (owner decision 2026-11-13)."""
    outline_trim_k: float = 3.0
    """Its trimming pass drops points whose radial residual is more than this many robust sigmas
    (1.4826 MAD) from the median residual, then refits once."""


@dataclass(frozen=True)
class RegionConfig:
    """Stage G: the bubble as a region (iso-contour), border completion and fragments."""

    level_window_deg: float = 15.0
    """The outline follows the per-ray 50 % levels smoothed by a running median over this
    polar angle (0 = one global median level for the whole region)."""
    neck_ratio: float = 0.5
    """Two lobes are separate bodies when the inscribed radius at their neck is below this
    fraction of the smaller lobe's.  Two bubbles touching with a 3-px overlap measure 0.36,
    a peanut-shaped bubble (two disks overlapping by 16 px) 0.59."""
    min_lobe_px: float = 3.0
    """A lobe needs an inscribed radius of at least this many px to be split off."""
    crop_margin_px: int = 2
    invalid_touch_px: int = 3
    """Contour vertices within this window of excluded pixels close the region, they are no edge."""
    fragment_reach: float = 2.0
    """Fragments are other dark regions within this many main-body radii of it."""
    fragment_min_px: float = 12.0
    fragment_min_frac: float = 0.003
    """A fragment's area must exceed max(fragment_min_px, fragment_min_frac * main area)."""
    disk_nodes: int = 2048
    """Quadrature nodes for the part of the fitted disk outside the ROI."""
    arc_samples: int = 720
    """Samples of the fitted circle for the hidden arc (perimeter and hull)."""
    gap_frac: float = 0.03
    gap_max_px: float = 4.0
    """Gaps in the bubble wall narrower than ~2 x min(gap_frac * R, gap_max_px) are bridged
    when looking for its enclosed interior (a jet slot 16 px wide in R = 60 is not)."""
    rim_inset_px: float = 1.0
    """The ray-edge line that closes a broken wall runs this far inside the edge points."""
    step_px: float = 2.0
    """Half-distance along the contour normal of the step-amplitude test."""


@dataclass(frozen=True)
class QualityConfig:
    """Stage F: confidence terms and flags."""

    min_edge_points: int = 5
    multi_edge_flag: float = 0.3
    ambiguous_flag: float = 0.5
    low_snr_ratio: float = 6.0
    non_circular_rel: float = 0.02
    non_circular_solidity: float = 0.95
    """Area definition: ``non_circular`` describes an outline whose circle and area radii differ by
    more than ``non_circular_rel`` or whose solidity is below this (information only)."""
    circle_check_rel: float = 0.05
    """Circle definition, bubble inside the ROI: ``non_circular`` (worth a check) when the circle
    radius is more than this fraction off the outline's area-equivalent radius (owner, 2026-11-13)."""
    clipped_check_rel: float = 0.06
    clipped_check_px: float = 1.0
    """A bubble cut by the ROI border: its radius and its visible area completed with an ellipse
    (``shape_check``) must agree within max(rel * R, px), else ``clipped_not_round`` (circle) or
    ``clipped_unmeasurable`` (area).  Calibrated on real round bubbles cut by the border (example_1,
    LIC 01, 05, 06, 10, lic_hard 07): at most 0.04; synthetic round ones (R 10-100, cut up to 65 %)
    at most 0.05; three lobes of 8 % cut at their phase 0 up to 50 %: 0.049; 1.5:1 ellipses: >= 0.12."""
    clipped_check_hidden: float = 0.02
    """A clipped region with at most this share of its area hidden (an outline that only touches
    a glare blob or the ROI border) is checked by its area, like one inside the ROI."""
    roundness_free: float = 0.05
    roundness_span: float = 0.10
    roundness_floor: float = 0.6
    """Circle definition: the ``roundness`` confidence term falls linearly from 1 at a circle check
    of ``roundness_free`` (the 5 % tolerance) to ``roundness_floor`` at ``free + span``: a misfit
    lowers the confidence, but a clean non-round bubble keeps a result and goes to Review by its
    flag.  Calibrated on the labelled LIC / lic_hard frames (417 bubble, 116 no-bubble frames):
    accepted and correct 217 (219 without the term), wrong 18 (25), no-bubble rejected 109 (105)."""
    shoulder_flag_px: float = 3.5
    precision_scale: float = 0.02
    """Precision term = exp(-(sigma_R / R) / scale), sigma_R from the noise along the outline.
    Calibrated on LIC: 78/85 no-bubble frames below 0.3 (75 before), 182 labelled bubble frames
    accepted and correct (178 before); synthetic no-bubble frames stay below 0.13."""
    snr_floor: float = 3.0
    snr_span: float = 7.0
    multi_scale: float = 0.3
    completion_free: float = 0.5
    """Hidden (extrapolated) fractions up to this cost no confidence; beyond, linearly to 0 at 1.
    The area definition counts the hidden area, the circle definition the hidden circumference."""
    min_roi_px: int = 8


@dataclass(frozen=True)
class ReferenceConfig:
    """Automatic choice of the bubble-free reference frame."""

    downsample: int = 4
    blur_sigma: float = 1.0
    change_ratio: float = 1.25
    """A pixel changed when its gain-normalised ratio to the previous frame leaves [1/r, r]."""
    event_change_frac: float = 0.02
    """A frame is an event (flash or bubble) when this fraction of its pixels changed."""
    flash_brightness_ratio: float = 1.10
    """A frame this much brighter than the previous one is a flash."""
    brightness_tolerance: float = 0.10
    min_bubble_radius_px: float = 3.0


@dataclass(frozen=True)
class EdgeConfig:
    """All tunable constants of the Edge engine."""

    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    rays: RayConfig = field(default_factory=RayConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    region: RegionConfig = field(default_factory=RegionConfig)


DEFAULT_CONFIG = EdgeConfig()
