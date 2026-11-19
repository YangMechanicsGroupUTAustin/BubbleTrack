"""FileController — folder/video loading and frame navigation."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import numpy as np

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.display_mixin import display_frame
from bubbletrack.event_bus import FRAME_CHANGED, SOURCE_LOADED, EventBus
from bubbletrack.model.cache import ImageCache
from bubbletrack.model.conventions import frame_to_display
from bubbletrack.model.folder_scan import scan_folder_report, tiff_page_count
from bubbletrack.model.frame_source import open_recording
from bubbletrack.model.hpvx2_dat import DAT_EXTS
from bubbletrack.model.image_io import (
    detect_bit_depth,
    imread_unicode,
    load_and_normalize,
    normalize_frame,
    sample_counts_divisor,
)
from bubbletrack.model.recording_identity import recording_fingerprint
from bubbletrack.ui.frame_scrubber import frame_file_labels

logger = logging.getLogger(__name__)

NOT_LOADED = "Nothing loaded: the current recording is still open"


def recording_kind(path: str) -> str:
    """'HPV-X2 recording' for a raw .dat file, else 'video'."""
    return "HPV-X2 recording" if os.path.splitext(path)[1].lower() in DAT_EXTS else "video"


def scan_notes(report) -> list[str]:
    """What the user should know about a scanned folder: files left out, and a
    multi-page TIFF (a stack) of which only the first page is read.  Only the first
    frame's pages are counted: opening every file would download a whole folder
    kept online-only by a sync client."""
    notes = list(report.warnings)
    if report.group == "tiff" and report.images:
        first = report.images[0]
        pages = tiff_page_count(first)
        if pages > 1:
            notes.append(f"{os.path.basename(first)} has {pages} pages; "
                         "only the first page is read.")
    return notes


class FileController(BaseController):
    """Handles folder selection and frame navigation."""

    def __init__(self, bus: EventBus, get_state, set_state, window,
                 set_max_radius, cache: ImageCache | None = None,
                 is_busy=lambda: False) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._set_max_radius = set_max_radius
        self._cache = cache
        self._is_busy = is_busy
        # Before another recording replaces this one: False keeps it (the
        # session controller asks about unsaved review decisions).
        self.confirm_leave: Callable[[], bool] = lambda: True

    def _refuse_while_busy(self) -> bool:
        """A running fit writes into the current arrays: never swap them out."""
        if self._is_busy():
            self.w.header.set_status(
                "Stop the running fit before loading other data", "#f59e0b",
            )
            return True
        return False

    # -- public handlers -------------------------------------------------- #

    def on_folder_selected(self, folder: str, *, ask: bool = True) -> None:
        """Load an image folder; with *ask*, unsaved review decisions are asked about."""
        if self._refuse_while_busy():
            return
        report = scan_folder_report(folder, check_pages=False)
        images = list(report.images)
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
            divisor = sample_counts_divisor(
                lambda i: imread_unicode(images[i]), len(images))
        except Exception as exc:
            logger.error("Cannot load %s: %s", images[0], exc)
            self.w.header.set_status(
                f"Cannot read {os.path.basename(images[0])}: {exc}", "#ef4444",
            )
            return
        if ask and not self.confirm_leave():
            self.w.header.set_status(NOT_LOADED, "#a1a1aa")
            return
        if self.state.video_reader is not None:  # the folder replaces the video
            self.state.video_reader.close()
            self._update(video_reader=None, video_path="")

        logger.info("Loaded %d images from %s", len(images), folder)

        self._update(
            folder_path=folder,
            images=tuple(images),
            source_fingerprint=recording_fingerprint(images),
            image_no=0,
            img_grayscale_max=bit_depth,
            counts_divisor=divisor,
            reference_frame=None,  # the previous recording's choices do not apply here
            edge_seed=None,
        )
        self.state = self.state.with_results_initialized(len(images))

        # Set default ROI to full image
        h, w = first_img.shape
        self._update(gridx=(1, h), gridy=(1, w))
        self._set_max_radius(float(max(h, w)))

        self.w.left_panel.image_source.set_folder_loaded(folder, len(images))
        self._reset_views(len(images), os.path.splitext(images[0])[1])
        notes = scan_notes(report)
        if notes:
            self.w.header.set_status("Ready — " + " ".join(notes), "#f59e0b")
        else:
            self.w.header.set_status("Ready", "#10b981")

        display_frame(self.state, self.w, 0, self._set_state, self._cache)

    def on_video_selected(self, video_path: str, *, ask: bool = True) -> None:
        """Open a video or HPV-X2 .dat recording and set up a virtual frame list.

        With *ask*, unsaved review decisions of the current recording are asked about.
        """
        if self._refuse_while_busy():
            return
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
        # Read the first frame BEFORE touching state (as for folders): a video
        # that cannot be decoded leaves the current recording intact.
        try:
            first_raw = reader.read_frame(0)
            first_img, _, _, _ = normalize_frame(first_raw, 0.5, (1, 99999), (1, 99999))
            divisor = sample_counts_divisor(
                lambda i: first_raw if i == 0 else reader.read_frame(i), n)
        except Exception as exc:
            logger.error("Cannot read the first frame of %s: %s", video_path, exc)
            reader.close()
            self.w.header.set_status(f"Cannot open video: {exc}", "#ef4444")
            return
        if ask and not self.confirm_leave():
            reader.close()
            self.w.header.set_status(NOT_LOADED, "#a1a1aa")
            return
        if self.state.video_reader is not None:  # the new recording replaces it
            self.state.video_reader.close()

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
            source_fingerprint=recording_fingerprint((), video_path, reader),
            image_no=0,
            img_grayscale_max=255,  # corrected from the first frame below
            counts_divisor=divisor,
            reference_frame=None,  # the previous recording's choices do not apply here
            edge_seed=None,
        )
        self.state = self.state.with_results_initialized(n)

        # The first frame (read above) gives the dimensions and the default ROI
        if np.issubdtype(first_raw.dtype, np.integer):
            self._update(img_grayscale_max=int(np.iinfo(first_raw.dtype).max))
        h, w = first_img.shape
        self._update(gridx=(1, h), gridy=(1, w))
        self._set_max_radius(float(max(h, w)))

        # A valid video frame rate becomes the export's frame rate.
        if reader.fps > 0:
            self._update(fps=reader.fps)

        self.w.left_panel.image_source.set_video_loaded(video_path, n, reader.fps)
        self._reset_views(n, os.path.splitext(video_path)[1])
        self.w.header.set_status(f"Ready ({recording_kind(video_path)})", "#10b981")

        display_frame(self.state, self.w, 0, self._set_state, self._cache)

    def on_frame_changed(self, idx: int) -> None:
        if not self._valid_frame(idx):
            return
        self._show_frame(idx)

    # -- helpers ------------------------------------------------------------ #

    def _valid_frame(self, idx: int) -> bool:
        return bool(self.state.images) and 0 <= idx < len(self.state.images)

    def _show_frame(self, idx: int) -> None:
        """Make *idx* the current frame and keep every frame control in sync.

        Callers include the scrubber itself, the chart, Review and the
        arrow-key shortcuts, so all of them are updated here.
        """
        self._update(image_no=idx)
        scrubber = self.w.frame_scrubber
        scrubber.blockSignals(True)
        scrubber.set_value(idx)
        scrubber.blockSignals(False)
        self.bus.emit(FRAME_CHANGED, idx)
        display_frame(self.state, self.w, idx, self._set_state, self._cache)

    def _reset_views(self, n: int, file_ext: str) -> None:
        """Reset every frame-dependent view for a newly loaded source of *n* frames.

        Signals are blocked while ranges shrink: otherwise clamping the
        scrubber emits a frame change to the old last frame and the shown
        frame, state.image_no and the scrubber disagree.
        """
        # Cached frames are keyed by path (video frames by index only) and a folder can
        # be re-exported in place: nothing cached belongs to the new source.
        if self._cache is not None:
            self._cache.invalidate()
        edge_support.forget_references()
        scrubber = self.w.frame_scrubber
        # "Frame 12/91 · _023": the file number next to every frame index
        labels = frame_file_labels(self.state.images)
        scrubber.set_frame_labels(labels)
        self.w.status_bar.set_frame_labels(labels)
        self.w.radius_chart.set_frame_labels(labels)
        scrubber.blockSignals(True)
        scrubber.set_range(n)
        scrubber.set_value(0)
        scrubber.blockSignals(False)
        self.w.original_panel.set_roi_text(self.state.gridx, self.state.gridy)
        self.w.radius_chart.clear()
        self.w.radius_chart.set_total_frames(n)
        self.w.status_bar.update_frame(frame_to_display(0), n)
        self.w.status_bar.update_format(file_ext)
        self.bus.emit(SOURCE_LOADED, n)
