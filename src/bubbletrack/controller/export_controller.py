"""ExportController — the Export dialog: pixel data, physical data, PDF report.

Every export carries the per-frame fit results and provenance metadata.
The folder and format of the last successful export are remembered in
config.json: the next dialog proposes a file in that folder, in that format.
The dialog's frame rate, scale and Rmax fit length go straight into the
state, so sessions, the config and reports record the values exported with.

Frame rate and scale count as set for a recording only once the user typed or
confirmed them in the dialog, or a session file restored them: the values in
the config belong to whatever recording came before.  Until then the dialog
marks them and physical export waits; pixel exports record them as unknown,
and the status bar says "Scale: not set".
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from bubbletrack.controller import edge_support
from bubbletrack.controller.base import BaseController
from bubbletrack.controller.session_controller import SESSION_RESTORED
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
from bubbletrack.ui.export_dialog import (
    FORMAT_EXT, PHYSICAL_FIELDS, ExportDialog, report_path_for,
)

logger = logging.getLogger(__name__)

# File-name prefix of each kind of export (``<source>_<prefix>_<stamp>.<ext>``).
PREFIXES: dict[str, str] = {"pixel": "R_data", "physical": "RofT_data"}

# Characters that are not allowed in a Windows file name (or are control chars).
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_OK_COLOUR = "#10b981"
_WARNING_COLOUR = "#f59e0b"  # amber, as used elsewhere for "marginal" states

Outcome = tuple[str, str]  # (status text, colour) of a finished export


def _show_export_success(panel, path: str, warning: str = "") -> Outcome:
    """Report a finished export: green when clean, amber with a warning."""
    if not warning:
        text = f"Exported: {os.path.basename(path)}"
        panel.set_status(text, True)
        return text, _OK_COLOUR
    logger.warning("Export %s: %s", path, warning)
    text = f"Exported with warning: {warning}"
    panel.set_status(text, True)
    label = getattr(panel, "_status", None)
    if label is not None:
        label.setStyleSheet(f"color:{_WARNING_COLOUR}; font-size:11px;")
    return text, _WARNING_COLOUR


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
    """Opens the Export dialog and writes .mat / CSV / Excel files and reports.

    The ``export_*`` methods take a *panel* with the dialog's getters
    (``get_format``, ``get_fps``, ``get_scale``, ``get_fit_length``) and
    ``set_status``; they return the outcome, or None after reporting why not.
    """

    def __init__(self, bus, get_state, set_state, window) -> None:
        super().__init__(bus, get_state, set_state, window)
        self._units = "physical"  # the last export's choice, offered next time
        self._set_fields: frozenset[str] = frozenset()  # 'fps' / 'scale' set for this recording
        bus.subscribe("source_loaded", lambda _n: self._set_physical(frozenset()))
        bus.subscribe(SESSION_RESTORED, lambda: self._set_physical(PHYSICAL_FIELDS))

    # -- frame rate and scale of this recording ------------------------------ #

    def physical_set(self) -> bool:
        """Frame rate and scale were set (or restored) for the loaded recording."""
        return self._set_fields == PHYSICAL_FIELDS

    def _set_physical(self, fields: frozenset[str]) -> None:
        self._set_fields = frozenset(fields) & PHYSICAL_FIELDS
        status_bar = getattr(self.w, "status_bar", None)
        if status_bar is not None:
            status_bar.update_scale(self.state.um2px if "scale" in self._set_fields else None)

    # -- the dialog ---------------------------------------------------------- #

    def make_dialog(self) -> ExportDialog:
        state = self.state
        dialog = ExportDialog(
            fps=state.fps, um2px=state.um2px, fit_length=state.rmax_fit_length,
            fmt=config.get_meta("last_export_format") or "mat", units=self._units,
            default_path=self.default_path, unset=PHYSICAL_FIELDS - self._set_fields,
            parent=self.w,
        )
        dialog.physical_params_changed.connect(lambda: self.on_physical_params_changed(dialog))
        dialog.export_clicked.connect(lambda: self.export_from(dialog))
        return dialog

    def open_dialog(self) -> None:
        self.make_dialog().exec()

    def on_physical_params_changed(self, panel) -> None:
        self._update(fps=panel.get_fps(), um2px=panel.get_scale(),
                     rmax_fit_length=panel.get_fit_length())
        self._set_physical(PHYSICAL_FIELDS - panel.unset_fields())

    def export_from(self, dialog: ExportDialog) -> bool:
        """Export what the dialog says; close it on success, else show why not."""
        if self.state.radius is None:
            dialog.set_status("No data to export", False)
            return False
        path = dialog.get_path()
        if not path:
            dialog.set_status("Choose where to save the file", False)
            return False
        units = dialog.get_units()
        if units == "physical" and not dialog.physical_ready():
            dialog.set_status("Check the frame rate and scale of this recording first "
                              "(or choose Pixels).", False)
            return False
        report = report_path_for(path) if dialog.report_enabled() else ""
        if not dialog.confirm_overwrite([path, report]):
            return False
        self._units = units
        export = self.export_pixel if units == "pixel" else self.export_physical
        outcome = export(dialog, path)
        if outcome is not None and report:
            text, colour = outcome
            if self.export_report(dialog, report) is None:
                return False
            outcome = (f"{text}; report {os.path.basename(report)}", colour)
        if outcome is None:
            return False
        header = getattr(self.w, "header", None)
        if header is not None:
            header.set_status(*outcome)
        dialog.accept()
        return True

    # -- the exports (no dialogs) ------------------------------------------------ #

    def _ready(self, panel, path: str, what: str = "data to export") -> bool:
        if self.state.radius is None:
            panel.set_status(f"No {what}", False)
            return False
        if not path:
            panel.set_status("Choose where to save the file", False)
            return False
        return True

    def export_pixel(self, panel, path: str) -> Outcome | None:
        """R per frame in pixels (with centres, edges and the fit results)."""
        if not self._ready(panel, path):
            return None
        fmt = panel.get_format()
        try:
            warning = self._do_export_r_data(
                path, fmt, metadata=self._export_metadata(panel, physical=False))
        except Exception as exc:
            panel.set_status(f"Error: {exc}", False)
            return None
        logger.info("Export: %s", path)
        outcome = _show_export_success(panel, path, warning)
        self._remember_export(path, fmt)
        return outcome

    def export_physical(self, panel, path: str) -> Outcome | None:
        """R(t) in physical units, normalised by the fitted Rmax."""
        if not self._ready(panel, path):
            return None
        fmt = panel.get_format()
        try:
            metadata = self._export_metadata(panel, physical=True)
            if fmt in ("csv", "xlsx"):
                ok, msg = True, self._do_export_r_data(
                    path, fmt,
                    fps=panel.get_fps(),
                    um2px=panel.get_scale(),
                    rmax_fit_length=panel.get_fit_length(),
                    metadata=metadata,
                )
            else:
                ok, msg = export_rof_t_data(
                    path,
                    self.state.radius,
                    panel.get_scale(),
                    panel.get_fps(),
                    panel.get_fit_length(),
                    metadata=metadata,
                    confidence=_aligned_fit_results(self.state).get("fit_confidence"),
                )
        except Exception as exc:
            panel.set_status(f"Error: {exc}", False)
            return None
        if not ok:
            panel.set_status(msg, False)
            return None
        logger.info("Export: %s", path)
        outcome = _show_export_success(panel, path, msg)
        self._remember_export(path, fmt)
        return outcome

    def export_report(self, panel, path: str) -> Outcome | None:
        """The PDF report: R(t) chart and fit summary."""
        if not self._ready(panel, path, "data for report"):
            return None
        chart_path = self._save_chart_image()
        try:
            generate_report(
                path, self.state, chart_image_path=chart_path,
                fps=panel.get_fps(), um2px=panel.get_scale(),
                physical_set=self.physical_set(),
            )
        except Exception as exc:
            panel.set_status(f"Report error: {exc}", False)
            return None
        finally:
            if chart_path:
                _remove_quietly(chart_path)
        logger.info("Report: %s", path)
        text = f"Report saved: {os.path.basename(path)}"
        panel.set_status(text, True)
        self._remember_export(path)
        return text, _OK_COLOUR

    # -- private helpers -------------------------------------------------- #

    def _save_chart_image(self) -> str | None:
        """Render the R-t chart to a temporary PNG; None if that is not possible."""
        chart = getattr(self.w, "radius_chart", None)
        if chart is None:
            return None
        try:
            from pyqtgraph.exporters import ImageExporter
        except ImportError:
            return None
        fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            # The report shows the data, not the cursor/selection/hover text.
            with chart.export_view() as plot_item:
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

    def _export_metadata(self, panel, *, physical: bool) -> dict:
        """Provenance for an export: current state plus the dialog values.

        A pixel export records frame rate and scale only when they were set for
        this recording (None: unknown).
        """
        known = physical or self.physical_set()
        return build_export_metadata(
            self.state,
            fps=panel.get_fps() if known else None,
            um2px=panel.get_scale() if known else None,
            rmax_fit_length=panel.get_fit_length(),
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

    def default_path(self, units: str, fmt: str) -> str:
        """Where the dialog proposes to save: last folder, source name, timestamp."""
        name = default_filename(self.state, PREFIXES.get(units, PREFIXES["pixel"]),
                                FORMAT_EXT.get(fmt, ".mat"))
        return os.path.join(self._default_export_dir(), name)

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
