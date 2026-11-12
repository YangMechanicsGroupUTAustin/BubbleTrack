"""FileController — folder/video loading, frame navigation, tab switching."""

from __future__ import annotations

import logging
import os

import numpy as np

from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import display_frame
from bubbletrack.event_bus import EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.conventions import frame_to_display
from bubbletrack.model.frame_source import open_recording
from bubbletrack.model.image_io import (
    detect_bit_depth,
    load_and_normalize,
    normalize_frame,
    scan_folder,
)
from bubbletrack.model.state import update_state

logger = logging.getLogger(__name__)


class FileController(BaseController):
    """Handles folder selection, frame navigation, and tab switching."""

    def __init__(self, bus: EventBus, get_state, set_state, window,
                 set_max_radius, cache: ImageCache | None = None,
                 is_busy=lambda: False) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._set_max_radius = set_max_radius
        self._cache = cache
        self._is_busy = is_busy

    def _refuse_while_busy(self) -> bool:
        """A running fit writes into the current arrays: never swap them out."""
        if self._is_busy():
            self.w.header.set_status(
                "Stop the running fit before loading other data", "#f59e0b",
            )
            return True
        return False

    # -- public handlers -------------------------------------------------- #

    def on_folder_selected(self, folder: str) -> None:
        if self._refuse_while_busy():
            return
        # Close any previous video reader
        if self.state.video_reader is not None:
            self.state.video_reader.close()
            self._update(video_reader=None, video_path="")

        images = scan_folder(folder)
        if not images:
            self.w.header.set_status("No images found", "#ef4444")
            return

        # Read the first frame BEFORE touching state, so an unreadable folder
        # leaves the current session intact instead of half-loaded.
        try:
            first_img, _, _, _ = load_and_normalize(
                images[0], 0.5, (1, 99999), (1, 99999),
            )
            bit_depth = detect_bit_depth(images[0])
        except Exception as exc:
            logger.error("Cannot load %s: %s", images[0], exc)
            self.w.header.set_status(
                f"Cannot read {os.path.basename(images[0])}: {exc}", "#ef4444",
            )
            return

        logger.info("Loaded %d images from %s", len(images), folder)

        self._update(
            folder_path=folder,
            images=tuple(images),
            image_no=0,
            img_grayscale_max=bit_depth,
        )
        self.state = self.state.with_results_initialized(len(images))

        # Set default ROI to full image
        h, w = first_img.shape
        self._update(gridx=(1, h), gridy=(1, w))
        self._set_max_radius(float(max(h, w)))

        self.w.left_panel.image_source.set_folder_loaded(folder, len(images))
        self._reset_views(len(images), os.path.splitext(images[0])[1])
        self.w.header.set_status("Ready", "#10b981")

        display_frame(self.state, self.w, 0, self._set_state, self._cache)

    def on_video_selected(self, video_path: str) -> None:
        """Open a video or HPV-X2 .dat recording and set up a virtual frame list."""
        if self._refuse_while_busy():
            return
        # Close any previous video reader
        if self.state.video_reader is not None:
            self.state.video_reader.close()

        try:
            reader = open_recording(video_path)
        except Exception as exc:
            self.w.header.set_status(f"Cannot open video: {exc}", "#ef4444")
            return

        n = reader.total_frames
        if n == 0:
            self.w.header.set_status("Video has no frames", "#ef4444")
            reader.close()
            return

        logger.info("Opened video %s: %d frames, %.1f fps", video_path, n, reader.fps)

        # Generate virtual image paths ("video:<idx>") for the frame list.
        # These are never used as file paths — they serve as identifiers in
        # state.images so the rest of the pipeline (scrubber, chart, etc.)
        # still sees a frame list of the correct length.
        virtual_images = tuple(f"video:{i}" for i in range(n))

        self._update(
            video_path=video_path,
            video_reader=reader,
            folder_path=os.path.dirname(video_path),
            images=virtual_images,
            image_no=0,
            img_grayscale_max=255,  # corrected from the first frame below
        )
        self.state = self.state.with_results_initialized(n)

        # Read first frame to determine dimensions and set default ROI
        first_raw = reader.read_frame(0)
        if np.issubdtype(first_raw.dtype, np.integer):
            self._update(img_grayscale_max=int(np.iinfo(first_raw.dtype).max))
        first_img, _, _, _ = normalize_frame(
            first_raw, 0.5, (1, 99999), (1, 99999),
        )
        h, w = first_img.shape
        self._update(gridx=(1, h), gridy=(1, w))
        self._set_max_radius(float(max(h, w)))

        # If the video has a valid FPS, store it in state and show it
        lp = self.w.left_panel
        if reader.fps > 0:
            self._update(fps=reader.fps)
            lp.post_processing.set_physical_params(
                self.state.fps, self.state.um2px, self.state.rmax_fit_length,
            )

        lp.image_source.set_video_loaded(video_path, n, reader.fps)
        self._reset_views(n, os.path.splitext(video_path)[1])
        self.w.header.set_status("Ready (video)", "#10b981")

        display_frame(self.state, self.w, 0, self._set_state, self._cache)

    def on_frame_changed(self, idx: int) -> None:
        if not self._valid_frame(idx):
            return
        self._show_frame(idx)

    def on_tab_frame_selected(self, idx: int) -> None:
        """Frame selected from a tab's frame spinbox."""
        if not self._valid_frame(idx):
            return
        self._show_frame(idx)

    # -- helpers ------------------------------------------------------------ #

    def _valid_frame(self, idx: int) -> bool:
        return bool(self.state.images) and 0 <= idx < len(self.state.images)

    def _show_frame(self, idx: int) -> None:
        """Make *idx* the current frame and keep every frame control in sync.

        Callers include the scrubber itself, the tab spinboxes, the chart and
        the arrow-key shortcuts, so all of them are updated here.
        """
        self._update(image_no=idx)
        scrubber = self.w.frame_scrubber
        scrubber.blockSignals(True)
        scrubber.set_value(idx)
        scrubber.blockSignals(False)
        self.w.left_panel.pretune_tab.set_frame_value(idx)
        self.w.left_panel.manual_tab.set_frame_value(idx)
        self.bus.emit("frame_changed", idx)
        display_frame(self.state, self.w, idx, self._set_state, self._cache)

    def _reset_views(self, n: int, file_ext: str) -> None:
        """Reset every frame-dependent view for a newly loaded source of *n* frames.

        Signals are blocked while ranges shrink: otherwise clamping the
        scrubber emits a frame change to the old last frame and the shown
        frame, state.image_no and the scrubber disagree.
        """
        lp = self.w.left_panel
        scrubber = self.w.frame_scrubber
        scrubber.blockSignals(True)
        scrubber.set_range(n)
        scrubber.set_value(0)
        scrubber.blockSignals(False)
        lp.pretune_tab.set_frame_range(n)
        lp.manual_tab.set_frame_range(n)
        lp.pretune_tab.set_frame_value(0)
        lp.manual_tab.set_frame_value(0)
        lp.automatic_tab.set_range(n)
        self.w.original_panel.set_roi_text(self.state.gridx, self.state.gridy)
        self.w.radius_chart.clear()
        self.w.radius_chart.set_total_frames(n)
        self.w.status_bar.update_frame(frame_to_display(0), n)
        self.w.status_bar.update_format(file_ext)
        self.bus.emit("source_loaded", n)

    def on_tab_changed(self, idx: int) -> None:
        """Tab bar switched — exit interactive modes, update status."""
        self.w.original_panel.set_mode("normal")
        self.w.left_panel.manual_tab.reset()
        tab_names = ("Pre-tune", "Manual", "Automatic")
        self.w.status_bar.update_mode(tab_names[idx] if idx < len(tab_names) else "")
        self.bus.emit("tab_changed", idx)
