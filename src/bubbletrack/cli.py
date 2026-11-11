"""Command-line interface for headless batch bubble radius fitting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from bubbletrack.logging_config import setup_logging
from bubbletrack.model.config import DETECTION_MODES
from bubbletrack.model.export import export_csv, export_excel, export_r_data
from bubbletrack.model.frame_pipeline import FrameParams, fit_image_file
from bubbletrack.model.image_io import imread_unicode, scan_folder

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argument parser and return parsed arguments."""
    parser = argparse.ArgumentParser(
        prog="bubbletrack-cli",
        description="BubbleTrack -- headless batch bubble radius fitting",
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
        "--rf", type=int, default=90,
        help="Removing factor [0-100] (default: 90)",
    )
    parser.add_argument(
        "--mode", choices=DETECTION_MODES, default="intensity",
        help="Detection mode, as in the GUI (default: intensity)",
    )
    parser.add_argument(
        "--roi", nargs=4, type=int,
        metavar=("R0", "R1", "C0", "C1"),
        help="ROI bounds: row_start row_end col_start col_end (1-indexed inclusive)",
    )
    parser.add_argument(
        "--edges", nargs=4, type=int, default=[0, 0, 0, 0],
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
    return parser.parse_args(argv)


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
    """Entry point for the CLI."""
    args = _parse_args(argv)

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
    gridx, gridy = _resolve_roi(args, shape)
    params = FrameParams(
        sensitivity=args.threshold,
        gridx=gridx,
        gridy=gridy,
        removing_factor=args.rf,
        edges=tuple(bool(e) for e in args.edges),
        mode=args.mode,
        max_radius=float(max(shape)),  # the GUI's sanity bound
    )

    radius, centers, edge_pts, region_count = _process_frames(
        images, params, args.verbose,
    )

    fmt = _detect_format(args.output, args.format)
    _write_output(
        fmt, args.output, radius, centers, edge_pts, args.fps, args.scale,
        region_count if args.mode == "area" else None,
    )

    n_valid = int((radius > 0).sum())
    print(f"Done. {n_valid}/{len(images)} frames fitted. Output: {args.output}")


if __name__ == "__main__":
    main()
