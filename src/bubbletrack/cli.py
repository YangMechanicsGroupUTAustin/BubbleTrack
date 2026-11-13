"""Command-line interface for headless batch bubble radius fitting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from bubbletrack.logging_config import setup_logging
from bubbletrack.model import edge_pipeline
from bubbletrack.model.config import DETECTION_MODES, EDGE_DEFINITIONS
from bubbletrack.model.constants import DEFAULT_REMOVING_FACTOR
from bubbletrack.model.export import export_csv, export_excel, export_r_data
from bubbletrack.model.frame_pipeline import FrameParams, fit_image_file
from bubbletrack.model.image_io import imread_unicode, scan_folder
from bubbletrack.model.removing_factor import compute_removing_factor

logger = logging.getLogger(__name__)

EVAL_CASES = "eval-cases"  # subcommand: the hard-case regression runner (cases_cli)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="bubbletrack-cli",
        description="BubbleTrack -- headless batch bubble radius fitting",
        epilog=f"Also: bubbletrack-cli {EVAL_CASES} FOLDER runs the hard-case regression "
               f"library (see bubbletrack-cli {EVAL_CASES} --help).",
    )
    parser.add_argument(
        "--folder", required=True,
        help="Image folder path",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Adaptive threshold sensitivity [0-1] (default: 0.5)",
    )
    parser.add_argument(
        "--rf", type=float, default=DEFAULT_REMOVING_FACTOR,
        help=(
            "Removing factor as the GUI's Remove slider value [0-100] "
            f"(default: {DEFAULT_REMOVING_FACTOR}); it is mapped to a minimum "
            "blob area over the ROI exactly as in the GUI. Changed in 3.1: "
            "--rf used to be a raw pixel area, so old scripts that pass e.g. "
            "--rf 90 now remove far larger blobs; use --rf-area for a raw "
            "pixel area."
        ),
    )
    parser.add_argument(
        "--rf-area", type=int, default=None, metavar="PIXELS",
        help="Minimum blob area as a raw pixel area; overrides --rf",
    )
    parser.add_argument(
        "--mode", choices=DETECTION_MODES, default="intensity",
        help="Detection mode, as in the GUI (default: intensity)",
    )
    parser.add_argument(
        "--edge-definition", choices=EDGE_DEFINITIONS, default="steep",
        help="Auto (reference frame) mode, --mode edge: wall at the 50%% crossing of the steep edge (steep, "
             "default) or at the outer extent (outer)",
    )
    parser.add_argument(
        "--non-round", action="store_true",
        help="Auto (reference frame) mode, --mode edge: the bubble may be non-round, so R is the "
             "area-equivalent radius of the outline (default: the circle through the wall)",
    )
    parser.add_argument(
        "--reference", type=int, default=None, metavar="FRAME",
        help="Auto (reference frame) mode, --mode edge: 1-based bubble-free reference frame "
             "(default: chosen automatically)",
    )
    parser.add_argument(
        "--roi", nargs=4, type=int,
        metavar=("R0", "R1", "C0", "C1"),
        help="ROI bounds: row_start row_end col_start col_end (1-indexed inclusive)",
    )
    parser.add_argument(
        "--edges", nargs=4, type=int, choices=(0, 1), default=[0, 0, 0, 0],
        metavar=("T", "R", "D", "L"),
        help="Bubble crosses edge flags: top right down left (0 or 1)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file path (.mat / .csv / .xlsx)",
    )
    parser.add_argument(
        "--fps", type=float, default=1e6,
        help="Frames per second for time column (default: 1e6)",
    )
    parser.add_argument(
        "--scale", type=float, default=3.2,
        help="Micrometres per pixel (default: 3.2)",
    )
    parser.add_argument(
        "--format", choices=["mat", "csv", "xlsx"], default=None,
        help="Output format (auto-detected from extension if omitted)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject bad arguments before any frame is read (exits with status 2)."""
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"--threshold must be in [0, 1], got {args.threshold}")
    if not 0.0 <= args.rf <= 100.0:
        parser.error(f"--rf must be a slider value in [0, 100], got {args.rf}")
    if args.rf_area is not None and args.rf_area < 0:
        parser.error(f"--rf-area must be >= 0, got {args.rf_area}")
    if not args.fps > 0:
        parser.error(f"--fps must be > 0, got {args.fps}")
    if not args.scale > 0:
        parser.error(f"--scale must be > 0, got {args.scale}")
    out_dir = Path(args.output).resolve().parent
    if not out_dir.is_dir():
        parser.error(f"--output directory does not exist: {out_dir}")


def _validate_roi(
    parser: argparse.ArgumentParser,
    roi: list[int] | None,
    shape: tuple[int, int],
) -> None:
    """Check a 1-based inclusive ROI against the image size, like the GUI."""
    if roi is None:
        return
    r0, r1, c0, c1 = roi
    h, w = shape
    if not (1 <= r0 <= r1 <= h and 1 <= c0 <= c1 <= w):
        parser.error(
            f"--roi {r0} {r1} {c0} {c1} must satisfy 1 <= R0 <= R1 <= {h} and "
            f"1 <= C0 <= C1 <= {w} (1-based inclusive, image is {h}x{w})"
        )


def _detect_format(output_path: str, explicit_format: str | None) -> str:
    """Return the export format string, auto-detecting from extension if needed."""
    if explicit_format is not None:
        return explicit_format
    ext = Path(output_path).suffix.lower()
    return {".csv": "csv", ".xlsx": "xlsx"}.get(ext, "mat")


def _image_shape(images: list[str]) -> tuple[int, int] | None:
    """Return (height, width) of the first frame, or None if it is unreadable."""
    sample = imread_unicode(images[0])
    return None if sample is None else sample.shape[:2]


def _resolve_roi(
    args: argparse.Namespace,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (gridx, gridy) ROI bounds from args or the full image."""
    if args.roi:
        return (args.roi[0], args.roi[1]), (args.roi[2], args.roi[3])
    h, w = shape
    return (1, h), (1, w)


def _removing_area(
    args: argparse.Namespace,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
) -> int:
    """Minimum blob area in pixels: --rf-area, else --rf mapped like the GUI."""
    if args.rf_area is not None:
        return args.rf_area
    return compute_removing_factor(args.rf, gridx, gridy)


def _process_frames(
    images: list[str],
    params: FrameParams,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Fit every frame with the GUI's Fit All pipeline.

    Returns (radius, centers, edge_pts, region_count); unfitted frames have
    radius -1.
    """
    n = len(images)
    radius = np.full(n, -1.0)
    centers = np.full((n, 2), np.nan)
    edge_pts: list[np.ndarray | None] = [None] * n
    region_count = np.zeros(n, dtype=int)

    for i, img_path in enumerate(images):
        try:
            fit = fit_image_file(img_path, params)
            if fit.fitted:
                radius[i] = fit.radius
                centers[i] = [fit.center_row, fit.center_col]
                edge_pts[i] = fit.edge_xy
                region_count[i] = fit.region_count
        except Exception as exc:
            if verbose:
                print(f"Frame {i + 1}: {exc}", file=sys.stderr)

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"Processed {i + 1}/{n} frames")

    return radius, centers, edge_pts, region_count


def _write_output(
    fmt: str,
    output_path: str,
    radius: np.ndarray,
    centers: np.ndarray,
    edge_pts: list,
    fps: float,
    scale: float,
    region_count: np.ndarray | None = None,
) -> None:
    """Write results to the chosen format (region counts only for area mode)."""
    if fmt == "csv":
        export_csv(output_path, radius, centers, fps=fps, um2px=scale,
                   region_count=region_count)
    elif fmt == "xlsx":
        export_excel(output_path, radius, centers, fps=fps, um2px=scale,
                     region_count=region_count)
    else:
        export_r_data(output_path, radius, centers, edge_pts, region_count)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI.

    Exit status: 0 on success, 1 if no frame could be read or fitted, 2 for
    invalid arguments.  ``bubbletrack-cli eval-cases ...`` runs the hard-case
    regression library instead (``bubbletrack.cases_cli``, own exit codes).
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == EVAL_CASES:
        from bubbletrack.cases_cli import main as eval_cases
        sys.exit(eval_cases(argv[1:]))
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    images = scan_folder(args.folder)
    if not images:
        print(f"No images found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    shape = _image_shape(images)
    if shape is None:
        print(f"Cannot read the first image: {images[0]}", file=sys.stderr)
        sys.exit(1)
    _validate_roi(parser, args.roi, shape)
    if args.reference is not None and not 1 <= args.reference <= len(images):
        parser.error(f"--reference must be in [1, {len(images)}], got {args.reference}")
    gridx, gridy = _resolve_roi(args, shape)

    if args.mode == "edge":
        result = _process_frames_edge(images, args, gridx, gridy, float(max(shape)))
        _finish(args, images, *result)
        return
    params = FrameParams(
        sensitivity=args.threshold,
        gridx=gridx,
        gridy=gridy,
        removing_factor=_removing_area(args, gridx, gridy),
        edges=tuple(bool(e) for e in args.edges),
        mode=args.mode,
        max_radius=float(max(shape)),  # the GUI's sanity bound
    )

    radius, centers, edge_pts, region_count = _process_frames(
        images, params, args.verbose,
    )
    _finish(args, images, radius, centers, edge_pts, region_count)


def _process_frames_edge(images: list[str], args: argparse.Namespace,
                         gridx: tuple[int, int], gridy: tuple[int, int],
                         max_radius: float) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Fit every frame with the Edge engine, using the GUI's acceptance rules."""
    explicit = args.reference - 1 if args.reference is not None else None
    ref_idx = edge_pipeline.resolve_reference_index(
        lambda i: edge_pipeline.read_raw(images[i]), len(images), explicit,
    )
    reference = edge_pipeline.read_raw(images[ref_idx]) if ref_idx is not None else None
    print("Reference frame: "
          + (f"{ref_idx + 1}" if ref_idx is not None else "none (surface fallback)"))
    job = edge_pipeline.EdgeJob(
        params=edge_pipeline.make_params(edge_definition=args.edge_definition,
                                         non_round=args.non_round),
        reference=reference, max_radius=max_radius,
    )
    n = len(images)
    radius = np.full(n, -1.0)
    centers = np.full((n, 2), np.nan)
    edge_pts: list[np.ndarray | None] = [None] * n
    for i, img_path in enumerate(images):
        try:
            fit = edge_pipeline.fit_edge(edge_pipeline.read_raw(img_path), gridx, gridy, job)
            if fit.radius > 0:
                radius[i] = fit.radius
                centers[i] = [fit.rc, fit.cc]
                edge_pts[i] = fit.edge_xy
            elif args.verbose:
                print(f"Frame {i + 1}: {fit.rejected_reason}", file=sys.stderr)
        except Exception as exc:
            if args.verbose:
                print(f"Frame {i + 1}: {exc}", file=sys.stderr)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"Processed {i + 1}/{n} frames")
    return radius, centers, edge_pts, np.zeros(n, dtype=int)


def _finish(args: argparse.Namespace, images: list[str], radius: np.ndarray,
            centers: np.ndarray, edge_pts: list, region_count: np.ndarray) -> None:
    n_valid = int((radius > 0).sum())
    if n_valid == 0:
        # Do not overwrite a previous result with an all-invalid file.
        print(f"Done. 0/{len(images)} frames fitted.")
        print("No frame was fitted; output not written.", file=sys.stderr)
        sys.exit(1)

    fmt = _detect_format(args.output, args.format)
    _write_output(
        fmt, args.output, radius, centers, edge_pts, args.fps, args.scale,
        region_count if args.mode == "area" else None,
    )
    print(f"Done. {n_valid}/{len(images)} frames fitted. Output: {args.output}")


if __name__ == "__main__":
    main()
