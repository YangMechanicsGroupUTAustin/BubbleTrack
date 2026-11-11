"""ExportController — pixel-data, physical-data, and PDF report export."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime

from PyQt6.QtWidgets import QFileDialog

from bubbletrack.controller.base import BaseController
from bubbletrack.event_bus import EventBus
from bubbletrack.model.constants import DEFAULT_RMAX_FIT_LENGTH
from bubbletrack.model.export import (
    build_export_metadata,
    export_csv,
    export_excel,
    export_r_data,
    export_rof_t_data,
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


class ExportController(BaseController):
    """Handles exporting radius data to .mat / CSV / Excel files."""

    # -- public handlers -------------------------------------------------- #

    def on_export_r_data(self) -> None:
        pp = self.w.left_panel.post_processing
        if self.state.radius is None:
            pp.set_status("No data to export", False)
            return

        fmt = pp.get_format()
        ext = _FORMAT_EXT.get(fmt, ".mat")
        default = os.path.join(
            self._default_export_dir(),
            self._default_filename("R_data", ext=ext),
        )
        path, _ = QFileDialog.getSaveFileName(
            self.w, "Export Pixel Data", default,
            _FORMAT_FILTERS.get(fmt, _FORMAT_FILTERS["mat"]),
        )
        if not path:
            return  # user cancelled

        try:
            self._do_export_r_data(path, fmt, metadata=self._export_metadata(pp))
            logger.info("Export: %s", path)
            pp.set_status(f"Exported: {os.path.basename(path)}", True)
        except Exception as exc:
            pp.set_status(f"Error: {exc}", False)

    def on_export_rof_t(self) -> None:
        pp = self.w.left_panel.post_processing
        if self.state.radius is None:
            pp.set_status("No data to export", False)
            return

        fmt = pp.get_format()
        ext = _FORMAT_EXT.get(fmt, ".mat")
        default = os.path.join(
            self._default_export_dir(),
            self._default_filename("RofT_data", ext=ext),
        )
        path, _ = QFileDialog.getSaveFileName(
            self.w, "Export Physical Data", default,
            _FORMAT_FILTERS.get(fmt, _FORMAT_FILTERS["mat"]),
        )
        if not path:
            return  # user cancelled

        try:
            metadata = self._export_metadata(pp)
            if fmt in ("csv", "xlsx"):
                warning = self._do_export_r_data(
                    path, fmt,
                    fps=pp.get_fps(),
                    um2px=pp.get_scale(),
                    rmax_fit_length=pp.get_fit_length(),
                    metadata=metadata,
                )
                logger.info("Export: %s", path)
                _show_export_success(pp, path, warning)
                self.w.status_bar.update_scale(pp.get_scale())
            else:
                ok, msg = export_rof_t_data(
                    path,
                    self.state.radius,
                    pp.get_scale(),
                    pp.get_fps(),
                    pp.get_fit_length(),
                    metadata=metadata,
                )
                if ok:
                    logger.info("Export: %s", path)
                    _show_export_success(pp, path, msg)
                    self.w.status_bar.update_scale(pp.get_scale())
                else:
                    pp.set_status(msg, False)
        except Exception as exc:
            pp.set_status(f"Error: {exc}", False)

    def on_generate_report(self) -> None:
        pp = self.w.left_panel.post_processing
        if self.state.radius is None:
            pp.set_status("No data for report", False)
            return

        default = os.path.join(
            self._default_export_dir(),
            self._default_filename("report", ext=".pdf"),
        )
        path, _ = QFileDialog.getSaveFileName(
            self.w, "Save PDF Report", default,
            "PDF Files (*.pdf);;All Files (*)",
        )
        if not path:
            return  # user cancelled

        # Save chart image to a temporary file for embedding in the PDF
        chart_path: str | None = None
        try:
            chart_widget = self.w.radius_chart
            # Use pyqtgraph's export API to save the chart as a PNG
            exporter_cls = None
            try:
                from pyqtgraph.exporters import ImageExporter
                exporter_cls = ImageExporter
            except ImportError:
                pass

            if exporter_cls is not None:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False,
                )
                tmp.close()
                chart_path = tmp.name
                exporter = exporter_cls(chart_widget._plot.plotItem)
                exporter.parameters()["width"] = 1200
                exporter.export(chart_path)
        except Exception as exc:
            logger.warning("Could not export chart image: %s", exc)
            chart_path = None

        try:
            generate_report(
                path, self.state, chart_image_path=chart_path,
                fps=pp.get_fps(), um2px=pp.get_scale(),
            )
            logger.info("Report: %s", path)
            pp.set_status(f"Report saved: {os.path.basename(path)}", True)
        except Exception as exc:
            pp.set_status(f"Report error: {exc}", False)
        finally:
            # Clean up temporary chart image
            if chart_path:
                try:
                    os.unlink(chart_path)
                except OSError:
                    pass

    # -- private helpers -------------------------------------------------- #

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
        if fmt == "csv":
            return export_csv(
                path,
                self.state.radius,
                self.state.circle_fit_par,
                fps=fps,
                um2px=um2px,
                region_count=self.state.region_count,
                rmax_fit_length=rmax_fit_length,
                metadata=metadata,
            )
        if fmt == "xlsx":
            return export_excel(
                path,
                self.state.radius,
                self.state.circle_fit_par,
                fps=fps,
                um2px=um2px,
                region_count=self.state.region_count,
                rmax_fit_length=rmax_fit_length,
                metadata=metadata,
            )
        export_r_data(
            path,
            self.state.radius,
            self.state.circle_fit_par,
            self.state.circle_xy,
            self.state.region_count,
            metadata=metadata,
        )
        return ""

    def _export_metadata(self, pp) -> dict:
        """Provenance for an export: current state plus the panel values."""
        return build_export_metadata(
            self.state,
            fps=pp.get_fps(),
            um2px=pp.get_scale(),
            rmax_fit_length=pp.get_fit_length(),
        )

    def _default_export_dir(self) -> str:
        """Return the image folder as default export directory, or home."""
        return self.state.folder_path or os.path.expanduser("~")

    @staticmethod
    def _default_filename(prefix: str, ext: str = ".mat") -> str:
        """Generate a timestamped filename like ``R_data_20260319_143012.mat``."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}{ext}"
