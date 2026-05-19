"""Plot 3D intensity / contrast / gradient surfaces for one bubble image.

Examples
--------
Run on the default example image and save plots:

    python scripts/debug_contrast_surface.py

Focus on a 1-based ROI like the app uses:

    python scripts/debug_contrast_surface.py --roi 40 155 90 230

Show an interactive window as well as saving files:

    python scripts/debug_contrast_surface.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D projection
from scipy import ndimage


DEFAULT_IMAGE = Path("Example") / "Jin_15_04_56_049.tiff"
DEFAULT_OUTDIR = Path("debug_outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize bubble image intensity/contrast as 3D surfaces.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help=f"Image path. Default: {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("R0", "R1", "C0", "C1"),
        help="1-based inclusive ROI bounds, matching BubbleTrack.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Plot every Nth pixel for readability. Default: 2.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Output folder. Default: {DEFAULT_OUTDIR}",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib windows after saving PNGs.",
    )
    return parser.parse_args()


def load_gray(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    img = raw.astype(np.float64)
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return img


def apply_roi(img: np.ndarray, roi: list[int] | None) -> tuple[np.ndarray, str]:
    if roi is None:
        return img, "full"
    r0, r1, c0, c1 = roi
    # Convert BubbleTrack's 1-based inclusive ROI to Python slices.
    r0 = max(1, r0)
    c0 = max(1, c0)
    r1 = min(img.shape[0], r1)
    c1 = min(img.shape[1], c1)
    if r1 < r0 or c1 < c0:
        raise ValueError(f"Invalid ROI: {roi}")
    return img[r0 - 1:r1, c0 - 1:c1], f"r{r0}-{r1}_c{c0}-{c1}"


def local_contrast(img: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Return a simple local contrast map.

    Positive values mean brighter than local background; negative values mean
    darker than local background.  For plotting we use the absolute magnitude.
    """
    background = ndimage.gaussian_filter(img, sigma=sigma)
    contrast = img - background
    return contrast


def gradient_magnitude(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    smooth = ndimage.gaussian_filter(img, sigma=sigma)
    gy, gx = np.gradient(smooth)
    return np.hypot(gx, gy)


def robust_scale(z: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] using robust percentiles for clearer plots."""
    lo, hi = np.percentile(z, [1, 99])
    if hi <= lo:
        return np.zeros_like(z)
    return np.clip((z - lo) / (hi - lo), 0, 1)


def make_surface(
    z: np.ndarray,
    title: str,
    z_label: str,
    outpath: Path,
    *,
    downsample: int,
    cmap: str,
) -> None:
    ds = max(1, int(downsample))
    z_plot = z[::ds, ::ds]
    yy, xx = np.mgrid[0:z.shape[0]:ds, 0:z.shape[1]:ds]

    fig = plt.figure(figsize=(10, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx,
        yy,
        z_plot,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.set_title(title)
    ax.set_xlabel("x / column")
    ax.set_ylabel("y / row")
    ax.set_zlabel(z_label)
    ax.view_init(elev=45, azim=-135)
    fig.tight_layout()
    fig.savefig(outpath)


def make_topdown(z: np.ndarray, title: str, outpath: Path, *, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    im = ax.imshow(z, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x / column")
    ax.set_ylabel("y / row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = load_gray(image_path)
    roi_img, roi_tag = apply_roi(img, args.roi)

    contrast_signed = local_contrast(roi_img)
    contrast_abs = np.abs(contrast_signed)
    grad = gradient_magnitude(roi_img)

    stem = f"{image_path.stem}_{roi_tag}"
    products = [
        ("intensity", roi_img, "Normalized intensity", "viridis"),
        ("darkness", 1.0 - roi_img, "Darkness (1 - intensity)", "magma"),
        (
            "local_contrast_abs",
            robust_scale(contrast_abs),
            "Abs local contrast, robust-scaled",
            "plasma",
        ),
        (
            "local_contrast_signed",
            robust_scale(contrast_signed),
            "Signed local contrast, robust-scaled",
            "coolwarm",
        ),
        ("gradient", robust_scale(grad), "Gradient magnitude, robust-scaled", "inferno"),
    ]

    print(f"Image: {image_path}")
    print(f"ROI shape: {roi_img.shape}")
    print(f"Output folder: {outdir.resolve()}")

    for name, data, label, cmap in products:
        surface_path = outdir / f"{stem}_{name}_3d.png"
        topdown_path = outdir / f"{stem}_{name}_topdown.png"
        make_surface(
            data,
            f"{name} surface ({roi_tag})",
            label,
            surface_path,
            downsample=args.downsample,
            cmap=cmap,
        )
        make_topdown(data, f"{name} top-down ({roi_tag})", topdown_path, cmap=cmap)
        print(f"Saved {surface_path}")
        print(f"Saved {topdown_path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
