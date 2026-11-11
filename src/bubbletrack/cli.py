"""Command-line interface for headless batch bubble radius fitting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from bubbletrack.logging_config import setup_logging
from bubbletrack.model.circle_fit import circle_fit_taubin
from bubbletrack.model.detection import detect_bubble
from bubbletrack.model.export import export_csv, export_excel, export_r_data
from bubbletrack.model.image_io import (
    imread_unicode,
    load_and_normalize,
    scan_folder,
)

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


def _resolve_roi(
    args: argparse.Namespace,
    images: list[str],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (gridx, gridy) ROI bounds from args or full-image fallback."""
    if args.roi:
        return (args.roi[0], args.roi[1]), (args.roi[2], args.roi[3])

    # Use full image dimensions from the first frame
    sample = imread_unicode(images[0])
    if sample is not None:
        h, w = sample.shape[:2]
        return (1, h), (1, w)
    return (1, 9999), (1, 9999)


def _process_frames(
    images: list[str],
    threshold: float,
    gridx: tuple[int, int],
    gridy: tuple[int, int],
    edges: tuple[bool, bool, bool, bool],
    rf: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Run detection + circle fitting on all frames.

    Returns (radius, centers, edge_pts).
    """
    n = len(images)
    radius = np.full(n, -1.0)
    centers = np.full((n, 2), np.nan)
    edge_pts: list[np.ndarray | None] = [None] * n

    for i, img_path in enumerate(images):
        try:
            _, _, _, binary_roi = load_and_normalize(
                img_path, threshold, gridx, gridy,
            )
            _, edge_xy = detect_bubble(
                binary_roi, list(edges), rf, gridx, gridy,
            )
            if edge_xy is not None and len(edge_xy) >= 3:
                rc, cc, r = circle_fit_taubin(edge_xy)
                radius[i] = r
                centers[i] = [rc, cc]
                edge_pts[i] = edge_xy
        except Exception as exc:
            if verbose:
                print(f"Frame {i + 1}: {exc}", file=sys.stderr)

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"Processed {i + 1}/{n} frames")

    return radius, centers, edge_pts


def _write_output(
    fmt: str,
    output_path: str,
    radius: np.ndarray,
    centers: np.ndarray,
    edge_pts: list,
    fps: float,
    scale: float,
) -> None:
    """Write results to the chosen format."""
    if fmt == "csv":
        export_csv(output_path, radius, centers, fps=fps, um2px=scale)
    elif fmt == "xlsx":
        export_excel(output_path, radius, centers, fps=fps, um2px=scale)
    else:
        export_r_data(output_path, radius, centers, edge_pts)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = _parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    images = scan_folder(args.folder)
    if not images:
        print(f"No images found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    gridx, gridy = _resolve_roi(args, images)
    edges = tuple(bool(e) for e in args.edges)

    radius, centers, edge_pts = _process_frames(
        images, args.threshold, gridx, gridy, edges, args.rf, args.verbose,
    )

    fmt = _detect_format(args.output, args.format)
    _write_output(fmt, args.output, radius, centers, edge_pts, args.fps, args.scale)

    n_valid = int((radius > 0).sum())
    print(f"Done. {n_valid}/{len(images)} frames fitted. Output: {args.output}")


if __name__ == "__main__":
    main()
