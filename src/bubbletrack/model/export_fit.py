"""Per-frame fit results in the pixel exports.

Where each radius came from (``FitSource``), how far it can be trusted
(confidence, flags), whether it was accepted in review (lock) and the shape
metrics are exported next to the radius:

- CSV / Excel: columns appended after the existing ones,
- MAT: the variables ``FitSource`` (+ ``FitSourceNames``), ``FitConfidence``,
  ``FitLocked``, ``FitFlags`` (cellstr) and the struct ``FitMetrics``.

Every field is optional; a field that is not given adds nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import FIT_METRIC_KEYS

FIT_SOURCE_NAMES: tuple[str, ...] = tuple(source.name for source in FitSource)
"""Source names indexed by their code (``FIT_SOURCE_NAMES[FitSource.AUTO]``)."""

FLAG_SEPARATOR = "|"

_FMT_TEXT = "s"
_FMT_INT = "d"
_FMT_SCORE = ".4f"
_FMT_COUNT = ".0f"  # counts are float arrays (NaN = n/a); ".0f" renders both

# Metric key -> (column header, CSV format).  Unknown keys use the key itself.
METRIC_COLUMNS: dict[str, tuple[str, str]] = {
    "r_circle": ("R_Circle_px", _FMT_SCORE),
    "area_px": ("Area_px2", _FMT_SCORE),
    "circularity": ("Circularity", _FMT_SCORE),
    "solidity": ("Solidity", _FMT_SCORE),
    "fragment_count": ("Fragment_Count", _FMT_COUNT),
    "fragment_area_px": ("Fragment_Area_px2", _FMT_SCORE),
    "shoulder_width": ("Shoulder_Width_px", _FMT_SCORE),
}

_FIELDS = ("fit_source", "fit_confidence", "fit_locked", "fit_flags", "fit_metrics")


def fit_result_arrays(state) -> dict[str, object]:
    """Return the per-frame fit results of *state* as exporter keyword arguments.

    Usage: ``export_csv(path, r, c, **fit_result_arrays(state))``.
    """
    return {name: getattr(state, name, None) for name in _FIELDS}


def source_name(code: int) -> str:
    """Name of a FitSource code; unknown codes are rendered as the number."""
    return FIT_SOURCE_NAMES[code] if 0 <= code < len(FIT_SOURCE_NAMES) else str(code)


def _checked(name: str, values, n: int, dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).ravel()
    if len(array) != n:
        raise ValueError(f"{name} has {len(array)} entries, expected {n} (one per frame)")
    return array


def _metric_column(key: str) -> tuple[str, str]:
    """(header, CSV format) of a metric column."""
    return METRIC_COLUMNS.get(key, (key, _FMT_SCORE))


def _metric_order(metrics: Mapping[str, object]) -> list[str]:
    known = [key for key in FIT_METRIC_KEYS if key in metrics]
    return known + sorted(key for key in metrics if key not in FIT_METRIC_KEYS)


def _cellstr(strings: Sequence[str]) -> np.ndarray:
    """A 1 x N object array, which ``savemat`` writes as a MATLAB cellstr."""
    cells = np.empty((1, len(strings)), dtype=object)
    cells[0, :] = list(strings)
    return cells


@dataclass(frozen=True)
class FitColumns:
    """Validated per-frame fit results; ``None`` fields are not exported."""

    source: np.ndarray | None = None
    confidence: np.ndarray | None = None
    locked: np.ndarray | None = None
    flags: tuple[str, ...] | None = None  # pipe-joined, sorted; "" = no flag
    metrics: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        n_frames: int,
        *,
        fit_source=None,
        fit_confidence=None,
        fit_locked=None,
        fit_flags=None,
        fit_metrics: Mapping[str, object] | None = None,
    ) -> FitColumns:
        """Validate the exporter arguments; raises ValueError on a length mismatch."""
        n = n_frames
        flags = None
        if fit_flags is not None:
            if len(fit_flags) != n:
                raise ValueError(
                    f"fit_flags has {len(fit_flags)} entries, expected {n} (one per frame)")
            flags = tuple(FLAG_SEPARATOR.join(sorted(f or ())) for f in fit_flags)
        metrics = {
            key: _checked(f"fit_metrics[{key!r}]", fit_metrics[key], n, np.float64)
            for key in _metric_order(fit_metrics or {})
        }
        return cls(
            source=None if fit_source is None else _checked("fit_source", fit_source, n, np.int8),
            confidence=(None if fit_confidence is None
                        else _checked("fit_confidence", fit_confidence, n, np.float64)),
            locked=None if fit_locked is None else _checked("fit_locked", fit_locked, n, bool),
            flags=flags,
            metrics=metrics,
        )

    # -- CSV / Excel ---------------------------------------------------------

    def header(self) -> list[str]:
        header: list[str] = []
        if self.source is not None:
            header.append("Source")
        if self.confidence is not None:
            header.append("Confidence")
        if self.locked is not None:
            header.append("Locked")
        if self.flags is not None:
            header.append("Flags")
        header.extend(_metric_column(key)[0] for key in self.metrics)
        return header

    def cells(self, i: int) -> list[tuple[object, str]]:
        """Cells of frame *i* (0-based), in :meth:`header` order."""
        cells: list[tuple[object, str]] = []
        if self.source is not None:
            cells.append((source_name(int(self.source[i])), _FMT_TEXT))
        if self.confidence is not None:
            cells.append((float(self.confidence[i]), _FMT_SCORE))
        if self.locked is not None:
            cells.append((int(self.locked[i]), _FMT_INT))
        if self.flags is not None:
            cells.append((self.flags[i], _FMT_TEXT))
        for key, values in self.metrics.items():
            cells.append((float(values[i]), _metric_column(key)[1]))
        return cells

    # -- MAT -------------------------------------------------------------------

    def mat_variables(self) -> dict[str, object]:
        """Variables for ``savemat``; per-frame vectors are 1 x N like ``Radius``."""
        data: dict[str, object] = {}
        if self.source is not None:
            data["FitSource"] = self.source.reshape(1, -1)
            data["FitSourceNames"] = _cellstr(FIT_SOURCE_NAMES)
        if self.confidence is not None:
            data["FitConfidence"] = self.confidence.reshape(1, -1)
        if self.locked is not None:
            data["FitLocked"] = self.locked.reshape(1, -1)
        if self.flags is not None:
            data["FitFlags"] = _cellstr(self.flags)
        if self.metrics:
            data["FitMetrics"] = {k: v.reshape(1, -1) for k, v in self.metrics.items()}
        return data
