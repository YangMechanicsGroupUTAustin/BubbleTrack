"""ExportController — pixel-data, physical-data, and PDF report export.

Every export carries the per-frame fit results and provenance metadata.
The folder and format of the last successful export are remembered in
config.json: dialogs start in that folder and the Format combo starts on
that format at the next launch.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.model import config
from bubbletrack.model.constants import DEFAULT_RMAX_FIT_LENGTH
from bubbletrack.model.export import (
    build_export_metadata,
    export_csv,
    export_excel,
    export_r_data,
    export_rof_t_data,
    fit_result_arrays,
)
from bubbletrack.model.report import generate_report

logger = logging.getLogger(__name__)

# File dialog filter strings keyed by format
_FORMAT_FILTERS: dict[str, str] = {
    "mat": "MAT Files (*.mat);;All Files (*)",
    "csv": "CSV Files (*.csv);;All Files (*)",
    "xlsx": "Excel Files (*.xlsx);;All Files (*)",
}

_FORMAT_EXT: dict[str, str] = {
    "mat": ".mat",
    "csv": ".csv",
    "xlsx": ".xlsx",
}

# Item texts of the PostProcessing Format combo, keyed by format.
_FORMAT_LABELS: dict[str, str] = {"mat": "MAT", "csv": "CSV", "xlsx": "Excel"}

# Characters that are not allowed in a Windows file name (or are control chars).
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Amber, as used elsewhere in the UI for "marginal" states.
_WARNING_COLOUR = "#f59e0b"


def _show_export_success(pp, path: str, warning: str = "") -> None:
    """Report a finished export: green when clean, amber with a warning.

    ``PostProcessingPanel.set_status`` only knows success (green) and
    error (red), so the warning colour is applied to its status label.
    """
    if not warning:
        pp.set_status(f"Exported: {os.path.basename(path)}", True)
        return
    logger.warning("Export %s: %s", path, warning)
    pp.set_status(f"Exported with warning: {warning}", True)
    label = getattr(pp, "_status", None)
    if label is not None:
        label.setStyleSheet(f"color:{_WARNING_COLOUR}; font-size:11px;")


def source_stem(state) -> str:
    """File-name stem of the loaded data: the video stem or the folder name."""
    if state.video_path:
        stem = Path(state.video_path).stem
    elif state.folder_path:
        stem = Path(state.folder_path).name
    else:
        return ""
    return _UNSAFE_NAME_CHARS.sub("_", stem).strip(" .")


def default_filename(state, prefix: str, ext: str, now: datetime | None = None) -> str:
    """``<source>_<prefix>_<YYYYmmdd_HHMMSS><ext>``, without the source if none."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = source_stem(state)
    return f"{stem}_{prefix}_{stamp}{ext}" if stem else f"{prefix}_{stamp}{ext}"


def select_format(panel, fmt: str) -> bool:
    """Pre-select *fmt* ("mat" / "csv" / "xlsx") in the panel's Format combo.

    Uses ``panel.set_format`` when the panel has one; PostProcessing does
    not (yet), so its combo is driven by item text.  Returns False when the
    format cannot be selected.
    """
    setter = getattr(panel, "set_format", None)
    if callable(setter):
        setter(fmt)
        return True
    combo = getattr(panel, "_format_combo", None)
    label = _FORMAT_LABELS.get(fmt)
    index = combo.findText(label) if combo is not None and label else -1
    if index < 0:
        return False
    combo.setCurrentIndex(index)
    return True


def _aligned_fit_results(state) -> dict[str, object]:
    """The state's per-frame fit results as export kwargs, one entry per frame.

    A field whose length no longer matches ``radius`` is left out (with a
    warning) so that a stale ancillary array can never block an export.
    """
    n = len(state.radius)
    kwargs: dict[str, object] = {}
    dropped: list[str] = []
    for name, values in fit_result_arrays(state).items():
        if values is None:
            continue
        if name == "fit_metrics":
            metrics = {k: v for k, v in values.items() if len(v) == n}
            dropped += [f"fit_metrics[{k!r}]" for k in values if k not in metrics]
            if metrics:
                kwargs[name] = metrics
        elif len(values) == n:
            kwargs[name] = values
        else:
            dropped.append(name)
    if dropped:
        logger.warning("Export: %s do(es) not match %d frames; left out",
                       ", ".join(dropped), n)
    return kwargs


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class ExportController(BaseController):
    """Handles exporting radius data to .mat / CSV / Excel files."""

    def __init__(self, bus, get_state, set_state, window) -> None:
        super().__init__(bus, get_state, set_state, window)
        self.restore_export_format()

    def restore_export_format(self) -> None:
        """Pre-select the format of the last export in the Format combo."""
        fmt = config.get_meta("last_export_format")
        if fmt is not None:
            select_format(self.w.left_panel.post_processing, fmt)

    # -- public handlers -------------------------------------------------- #

    def on_export_r_data(self) -> None:
        pp = self.w.left_panel.post_processing
        if self.state.radius is None:
            pp.set_status("No data to export", False)
            return

        fmt = pp.get_format()
        path = self._ask_save_path("Export Pixel Data", "R_data", fmt)
        if not path:
            return  # user cancelled

        try:
            warning = self._do_export_r_data(path, fmt, metadata=self._export_metadata(pp))
        except Exception as exc:
            pp.set_status(f"Error: {exc}", False)
            return
        logger.info("Export: %s", path)
        _show_export_success(pp, path, warning)
        self._remember_export(path, fmt)

    def on_export_rof_t(self) -> None:
        pp = self.w.left_panel.post_processing
        if self.state.radius is None:
            pp.set_status("No data to export", False)
            return

        fmt = pp.get_format()
        path = self._ask_save_path("Export Physical Data", "RofT_data", fmt)
        if not path:
            return  # user cancelled

        try:
            metadata = self._export_metadata(pp)
            if fmt in ("csv", "xlsx"):
                ok, msg = True, self._do_export_r_data(
                    path, fmt,
                    fps=pp.get_fps(),
                    um2px=pp.get_scale(),
                    rmax_fit_length=pp.get_fit_length(),
                    metadata=metadata,
                )
            else:
                ok, msg = export_rof_t_data(
                    path,
                    self.state.radius,
                    pp.get_scale(),
                    pp.get_fps(),
                    pp.get_fit_length(),
                    metadata=metadata,
                    confidence=_aligned_fit_results(self.state).get("fit_confidence"),
                )
        except Exception as exc:
            pp.set_status(f"Error: {exc}", False)
            return
        if not ok:
            pp.set_status(msg, False)
            return
        logger.info("Export: %s", path)
        _show_export_success(pp, path, msg)
        self.w.status_bar.update_scale(pp.get_scale())
        self._remember_export(path, fmt)

    def on_generate_report(self) -> None:
        pp = self.w.left_panel.post_processing
        if self.state.radius is None:
            pp.set_status("No data for report", False)
            return

        path, _ = QFileDialog.getSaveFileName(
            self.w, "Save PDF Report", self._default_path("report", ".pdf"),
            "PDF Files (*.pdf);;All Files (*)",
        )
        if not path:
            return  # user cancelled

        chart_path = self._save_chart_image()
        try:
            generate_report(
                path, self.state, chart_image_path=chart_path,
                fps=pp.get_fps(), um2px=pp.get_scale(),
            )
        except Exception as exc:
            pp.set_status(f"Report error: {exc}", False)
            return
        finally:
            if chart_path:
                _remove_quietly(chart_path)
        logger.info("Report: %s", path)
        pp.set_status(f"Report saved: {os.path.basename(path)}", True)
        self._remember_export(path)

    # -- private helpers -------------------------------------------------- #

    def _ask_save_path(self, title: str, prefix: str, fmt: str) -> str:
        """Ask where to save an export; '' if cancelled."""
        path, _ = QFileDialog.getSaveFileName(
            self.w, title, self._default_path(prefix, _FORMAT_EXT.get(fmt, ".mat")),
            _FORMAT_FILTERS.get(fmt, _FORMAT_FILTERS["mat"]),
        )
        return path

    def _save_chart_image(self) -> str | None:
        """Render the R-t chart to a temporary PNG; None if that is not possible."""
        try:
            from pyqtgraph.exporters import ImageExporter
        except ImportError:
            return None
        fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            # The report shows the data, not the cursor/selection/hover text.
            with self.w.radius_chart.export_view() as plot_item:
                exporter = ImageExporter(plot_item)
                exporter.parameters()["width"] = 1200
                exporter.export(chart_path)
        except Exception as exc:
            logger.warning("Could not export chart image: %s", exc)
            _remove_quietly(chart_path)
            return None
        return chart_path

    def _do_export_r_data(
        self,
        path: str,
        fmt: str,
        *,
        fps: float | None = None,
        um2px: float | None = None,
        rmax_fit_length: int = DEFAULT_RMAX_FIT_LENGTH,
        metadata: dict | None = None,
    ) -> str:
        """Dispatch pixel-data export to the correct format handler.

        Returns the exporter's warning message ("" when there is none).
        """
        fit_results = _aligned_fit_results(self.state)
        if fmt in ("csv", "xlsx"):
            writer = export_csv if fmt == "csv" else export_excel
            return writer(
                path,
                self.state.radius,
                self.state.circle_fit_par,
                fps=fps,
                um2px=um2px,
                region_count=self.state.region_count,
                rmax_fit_length=rmax_fit_length,
                metadata=metadata,
                **fit_results,
            )
        export_r_data(
            path,
            self.state.radius,
            self.state.circle_fit_par,
            self.state.circle_xy,
            self.state.region_count,
            metadata=metadata,
            **fit_results,
        )
        return ""

    def _export_metadata(self, pp) -> dict:
        """Provenance for an export: current state plus the panel values."""
        return build_export_metadata(
            self.state,
            fps=pp.get_fps(),
            um2px=pp.get_scale(),
            rmax_fit_length=pp.get_fit_length(),
            reference_index=self._edge_reference_index(),
        )

    def _edge_reference_index(self) -> int | None:
        """The reference frame the Edge engine uses (auto-selected or set)."""
        state = self.state
        if state.detection_mode != "edge" or not state.images:
            return None
        try:
            return edge_support.reference_index(state)
        except Exception as exc:  # reading frames can fail; never block an export
            logger.warning("Could not resolve the Edge reference frame: %s", exc)
            return None

    def _default_path(self, prefix: str, ext: str) -> str:
        return os.path.join(self._default_export_dir(),
                            default_filename(self.state, prefix, ext))

    def _default_export_dir(self) -> str:
        """The last export folder if it still exists, else the source folder, else home."""
        remembered = config.get_meta("last_export_dir")
        if remembered and os.path.isdir(remembered):
            return remembered
        if self.state.folder_path:
            return self.state.folder_path
        if self.state.video_path:
            return os.path.dirname(self.state.video_path)
        return os.path.expanduser("~")

    @staticmethod
    def _remember_export(path: str, fmt: str | None = None) -> None:
        values = {"last_export_dir": os.path.dirname(os.path.abspath(path))}
        if fmt is not None:
            values["last_export_format"] = fmt
        try:
            config.set_meta(**values)
        except ValueError as exc:
            logger.warning("Export choice not remembered: %s", exc)
