"""Tests for image I/O utilities."""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

from bubbletrack.model.image_io import (
    ImageLoadError,
    detect_bit_depth,
    load_and_normalize,
    scan_folder,
)


@pytest.fixture()
def img_folder(tmp_path):
    """Create a temp folder with a few synthetic 8-bit PNG images."""
    for i in range(3):
        img = np.full((100, 120), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), img)
    return str(tmp_path)


@pytest.fixture()
def img_16bit(tmp_path):
    """Create a single 16-bit TIFF."""
    img = np.full((64, 64), 30000, dtype=np.uint16)
    path = str(tmp_path / "img16.tiff")
    cv2.imwrite(path, img)
    return path


class TestScanFolder:
    def test_finds_png(self, img_folder):
        paths = scan_folder(img_folder)
        assert len(paths) == 3
        assert all(p.endswith(".png") for p in paths)

    def test_natural_sort(self, tmp_path):
        for name in ["img_2.png", "img_10.png", "img_1.png"]:
            cv2.imwrite(str(tmp_path / name), np.zeros((10, 10), np.uint8))
        paths = scan_folder(str(tmp_path))
        basenames = [os.path.basename(p) for p in paths]
        assert basenames == ["img_1.png", "img_2.png", "img_10.png"]

    def test_empty_folder(self, tmp_path):
        assert scan_folder(str(tmp_path)) == []

    def test_tiff_priority(self, tmp_path):
        cv2.imwrite(str(tmp_path / "a.png"), np.zeros((10, 10), np.uint8))
        cv2.imwrite(str(tmp_path / "b.tiff"), np.zeros((10, 10), np.uint8))
        paths = scan_folder(str(tmp_path))
        assert all(p.endswith(".tiff") for p in paths)

    def test_skips_appledouble_and_dot_files(self, tmp_path):
        """macOS '._*' companions and dot-files must not become (first) frames."""
        for name in ("frame_001.tif", "frame_002.tif", "frame_003.tif"):
            cv2.imwrite(str(tmp_path / name), np.zeros((10, 10), np.uint8))
        (tmp_path / "._frame_001.tif").write_bytes(b"\x00\x05\x16\x07AppleDouble")
        cv2.imwrite(str(tmp_path / ".hidden.tif"), np.zeros((10, 10), np.uint8))
        paths = scan_folder(str(tmp_path))
        assert [os.path.basename(p) for p in paths] == [
            "frame_001.tif", "frame_002.tif", "frame_003.tif",
        ]

    def test_folder_of_only_hidden_files_is_empty(self, tmp_path):
        (tmp_path / "._a.tif").write_bytes(b"\x00")
        assert scan_folder(str(tmp_path)) == []

    def test_scan_nonexistent_dir_returns_empty(self):
        """Passing a path that doesn't exist returns an empty list."""
        result = scan_folder("/this/path/does/not/exist/at/all")
        assert result == []

    def test_scan_file_not_dir_returns_empty(self, tmp_path):
        """Passing a file path instead of a directory returns an empty list."""
        f = tmp_path / "some_file.txt"
        f.write_text("hello")
        result = scan_folder(str(f))
        assert result == []

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlink creation requires elevated privileges on Windows",
    )
    def test_scan_skips_symlinks(self, tmp_path):
        """Symlinked image files are skipped; only real files are returned."""
        real_img = tmp_path / "real.png"
        cv2.imwrite(str(real_img), np.zeros((10, 10), np.uint8))
        link_img = tmp_path / "link.png"
        link_img.symlink_to(real_img)

        paths = scan_folder(str(tmp_path))
        basenames = [os.path.basename(p) for p in paths]
        assert "real.png" in basenames
        assert "link.png" not in basenames


class TestDetectBitDepth:
    def test_8bit(self, img_folder):
        paths = scan_folder(img_folder)
        assert detect_bit_depth(paths[0]) == 255

    def test_16bit(self, img_16bit):
        assert detect_bit_depth(img_16bit) == 65535


class TestLoadAndNormalize:
    def test_output_shapes(self, img_folder):
        paths = scan_folder(img_folder)
        full_img, full_bin, roi_img, roi_bin = load_and_normalize(
            paths[0], 0.5, (1, 50), (1, 60)
        )
        assert full_img.shape == (100, 120)
        assert full_bin.shape == (100, 120)
        assert roi_img.shape == (50, 60)
        assert roi_bin.shape == (50, 60)

    def test_normalised_range(self, img_folder):
        paths = scan_folder(img_folder)
        full_img, _, _, _ = load_and_normalize(paths[0], 0.5, (1, 100), (1, 120))
        # uniform image normalises to all-zero
        assert full_img.min() >= 0.0
        assert full_img.max() <= 1.0

    def test_binary_is_bool(self, img_folder):
        paths = scan_folder(img_folder)
        _, full_bin, _, roi_bin = load_and_normalize(paths[0], 0.5, (1, 100), (1, 120))
        assert full_bin.dtype == bool
        assert roi_bin.dtype == bool

    def test_sensitivity_affects_binary(self, tmp_path):
        # Gradient image so threshold matters
        grad = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
        path = str(tmp_path / "grad.png")
        cv2.imwrite(path, grad)
        _, bin_low, _, _ = load_and_normalize(path, 0.3, (1, 100), (1, 256))
        _, bin_high, _, _ = load_and_normalize(path, 0.7, (1, 100), (1, 256))
        # Higher sensitivity => more white
        assert bin_high.sum() >= bin_low.sum()

    def test_missing_file_raises(self):
        with pytest.raises(ImageLoadError):
            load_and_normalize("/nonexistent.png", 0.5, (1, 10), (1, 10))

    def test_nonexistent_file_raises(self):
        """Passing a path that does not exist raises ImageLoadError."""
        with pytest.raises(ImageLoadError, match="missing or corrupted"):
            load_and_normalize("/does/not/exist/image.tif", 0.5, (1, 10), (1, 10))

    def test_corrupted_file_raises(self, tmp_path):
        """A file with random bytes (not a valid image) raises ImageLoadError."""
        corrupted = tmp_path / "corrupted.tif"
        corrupted.write_bytes(os.urandom(256))
        with pytest.raises(ImageLoadError, match="missing or corrupted"):
            load_and_normalize(str(corrupted), 0.5, (1, 10), (1, 10))

    def test_full_roi(self, img_folder):
        paths = scan_folder(img_folder)
        full, _, roi, _ = load_and_normalize(paths[0], 0.5, (1, 100), (1, 120))
        np.testing.assert_array_equal(full, roi)

    def test_gaussian_blur_smooths(self, tmp_path):
        """Gaussian blur should reduce high-frequency noise."""
        rng = np.random.default_rng(42)
        noisy = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        path = str(tmp_path / "noisy.png")
        cv2.imwrite(path, noisy)
        _, bin_no_blur, _, _ = load_and_normalize(
            path, 0.5, (1, 100), (1, 100),
        )
        _, bin_blur, _, _ = load_and_normalize(
            path, 0.5, (1, 100), (1, 100), gaussian_sigma=5.0,
        )
        # Blurred binary should have fewer transitions (smoother)
        edges_no = np.sum(np.abs(np.diff(bin_no_blur.astype(int), axis=1)))
        edges_blur = np.sum(np.abs(np.diff(bin_blur.astype(int), axis=1)))
        assert edges_blur < edges_no

    def test_clahe_does_not_change_shape(self, img_folder):
        """CLAHE should preserve image dimensions."""
        paths = scan_folder(img_folder)
        full_no, _, _, _ = load_and_normalize(paths[0], 0.5, (1, 100), (1, 120))
        full_cl, _, _, _ = load_and_normalize(
            paths[0], 0.5, (1, 100), (1, 120), clahe_clip=2.0,
        )
        assert full_no.shape == full_cl.shape

    def test_filters_default_to_noop(self, img_folder):
        """With sigma=0 and clip=0, output should be identical to no-filter."""
        paths = scan_folder(img_folder)
        r1 = load_and_normalize(paths[0], 0.5, (1, 100), (1, 120))
        r2 = load_and_normalize(
            paths[0], 0.5, (1, 100), (1, 120),
            gaussian_sigma=0.0, clahe_clip=0.0,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])
