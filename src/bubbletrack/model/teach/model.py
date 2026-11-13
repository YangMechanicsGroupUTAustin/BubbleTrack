"""The ML assist pixel classifier: train from :class:`MLExamples`, predict P(bubble).

A tree ensemble (scikit-learn ExtraTrees by default) on the fixed feature bank,
with balanced classes, capped samples and a fixed seed: the same examples on
the same frames always give the same model, so a session stores only the
examples (JSON) and retrains on load.  Prediction is split into row chunks
predicted on threads with single-threaded forests: fast and still
bit-for-bit deterministic for any thread count.

scikit-learn is imported lazily; without it :func:`train_model` raises
:class:`MLUnavailableError`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from bubbletrack.model.edge.normalize import to_gray_frame
from bubbletrack.model.teach.config import ForestConfig, MLConfig
from bubbletrack.model.teach.examples import MLExamples
from bubbletrack.model.teach.features import FeatureStack, compute_features, roi_bounds
from bubbletrack.model.teach.sampling import training_pixels

_RANGE_PIXELS = 4000          # pixels per training frame kept for the novelty quantiles
_NOVELTY_LEVELS = np.linspace(0.02, 0.98, 25)
_NOVELTY_STRIDE = 7           # every 7th pixel of a predicted frame enters its quantiles


class MLError(ValueError):
    """The examples cannot train a model (e.g. one class is missing)."""


class MLUnavailableError(RuntimeError):
    """scikit-learn is not installed."""


@dataclass(frozen=True)
class TrainInfo:
    frames: tuple[int, ...]
    n_bubble: int
    n_background: int
    n_features: int
    train_ms: float


@dataclass(frozen=True, eq=False)
class Prediction:
    probability: np.ndarray
    """P(bubble) on the full-resolution ROI, float32 in [0, 1]."""
    novelty: float
    """How unlike the training frames this frame is: per feature, the mean distance
    between its quantiles and the training quantiles over the training spread;
    the median over features (0 = same distribution, ~1 = shifted by the spread)."""
    ms: float


def _forest(cfg: ForestConfig, n_jobs: int):
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    except ImportError as exc:                            # pragma: no cover - depends on the env
        raise MLUnavailableError("ML assist needs scikit-learn (pip install scikit-learn)") from exc
    cls = ExtraTreesClassifier if cfg.kind == "extra" else RandomForestClassifier
    return cls(n_estimators=int(cfg.n_estimators), max_depth=cfg.max_depth,
               min_samples_leaf=int(cfg.min_samples_leaf), class_weight="balanced",
               random_state=int(cfg.random_state), n_jobs=n_jobs)


@dataclass(frozen=True, eq=False)
class MLModel:
    """A trained classifier bound to its ROI and reference frame."""

    forest: object
    config: MLConfig
    gridx: tuple[int, int]
    gridy: tuple[int, int]
    frame_shape: tuple[int, int]
    reference: np.ndarray | None
    quantiles: np.ndarray
    """(Q, F) training quantiles of every feature (novelty check)."""
    info: TrainInfo

    def features(self, raw: np.ndarray) -> FeatureStack:
        shape = to_gray_frame(raw).values.shape
        if shape != self.frame_shape:
            raise ValueError(f"frame shape {shape} differs from the training frames {self.frame_shape}")
        return compute_features(raw, self.gridx, self.gridy, reference=self.reference,
                                config=self.config.features)

    def predict(self, raw: np.ndarray) -> Prediction:
        """P(bubble) of every ROI pixel of *raw*, and how unlike the training frames it is."""
        t0 = time.perf_counter()
        stack = self.features(raw)
        h, w, n = stack.values.shape
        flat = stack.values.reshape(-1, n)
        p = _predict_chunks(self.forest, flat, int(self.config.forest.n_jobs)).reshape(h, w)
        if stack.step != 1:
            p = cv2.resize(p, stack.roi_shape[::-1], interpolation=cv2.INTER_LINEAR)
        out = np.clip(p, 0.0, 1.0).astype(np.float32)
        novelty = _novelty(flat[::_NOVELTY_STRIDE], self.quantiles)
        return Prediction(probability=out, novelty=novelty, ms=(time.perf_counter() - t0) * 1e3)

    def probability(self, raw: np.ndarray) -> np.ndarray:
        return self.predict(raw).probability


def _novelty(x: np.ndarray, train_q: np.ndarray) -> float:
    q = np.quantile(x, _NOVELTY_LEVELS, axis=0)
    spread = np.maximum(train_q[-1] - train_q[0], 1e-6)
    return float(np.median(np.abs(q - train_q).mean(axis=0) / spread))


def _predict_chunks(forest, x: np.ndarray, n_jobs: int) -> np.ndarray:
    """P(class 1) of the rows of *x*; chunks on threads, each forest call single-threaded."""
    col = list(forest.classes_).index(1)
    if n_jobs <= 1 or len(x) < 4096:
        return forest.predict_proba(x)[:, col].astype(np.float32)
    chunks = np.array_split(x, n_jobs)
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(lambda c: forest.predict_proba(c)[:, col], chunks))
    return np.concatenate(parts).astype(np.float32)


def _frame_samples(examples: MLExamples, frame: int, raw: np.ndarray, shape: tuple[int, int], gx, gy,
                   reference, cfg: MLConfig,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bubble features, background features, range sample) of one example frame."""
    stack = compute_features(raw, gx, gy, reference=reference, config=cfg.features)
    grid = stack.values
    origin = np.array([gx[0] - 1, gy[0] - 1])
    out = []
    for px in training_pixels(examples, frame, shape, cfg.examples):
        local = px - origin
        keep = ((local >= 0) & (local < np.array(stack.roi_shape))).all(axis=1)
        cell = np.minimum(local[keep] // stack.step, np.array(grid.shape[:2]) - 1)
        out.append(grid[cell[:, 0], cell[:, 1]])
    flat = grid.reshape(-1, grid.shape[-1])
    sample = flat[rng.choice(len(flat), min(_RANGE_PIXELS, len(flat)), replace=False)]
    return out[0], out[1], sample


def _cap(rng: np.random.Generator, x: np.ndarray, n: int) -> np.ndarray:
    return x if len(x) <= n else x[np.sort(rng.choice(len(x), n, replace=False))]


def train_model(examples: MLExamples, read_frame: Callable[[int], np.ndarray], gridx, gridy, *,
                reference: np.ndarray | None = None, config: MLConfig = MLConfig()) -> MLModel:
    """Train the pixel classifier on every frame that has examples.

    Parameters
    ----------
    examples : MLExamples
    read_frame : callable
        ``read_frame(i)`` returns the raw frame *i* (0-based), any dtype / layout.
    gridx, gridy : (min, max)
        1-based inclusive ROI bounds (the model measures inside this ROI only).
    reference : ndarray, optional
        The bubble-free reference frame (the "view" channel uses it).
    config : MLConfig

    Raises
    ------
    MLError
        When the examples have no bubble or no background pixel inside the ROI.
    MLUnavailableError
        When scikit-learn is not installed.
    """
    t0 = time.perf_counter()
    if examples.is_empty:
        raise MLError("no examples: verify a frame or paint bubble and background")
    rng = np.random.default_rng(config.forest.random_state)
    pos, neg, samples, frame_shape, gx, gy = [], [], [], None, None, None
    for frame in examples.frames:
        raw = read_frame(frame)
        shape = to_gray_frame(raw).values.shape
        if frame_shape is None:
            frame_shape = shape
            gx, gy = roi_bounds(gridx, gridy, shape)
        elif shape != frame_shape:
            raise MLError(f"frame {frame} has shape {shape}, expected {frame_shape}")
        b, g, s = _frame_samples(examples, frame, raw, shape, gx, gy, reference, config, rng)
        pos.append(b)
        neg.append(g)
        samples.append(s)
    x_pos, x_neg = np.concatenate(pos), np.concatenate(neg)
    if not len(x_pos) or not len(x_neg):
        raise MLError("the examples need both bubble and background pixels inside the ROI")
    half = int(config.forest.max_samples) // 2
    x_pos, x_neg = _cap(rng, x_pos, half), _cap(rng, x_neg, half)
    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([np.ones(len(x_pos), np.int8), np.zeros(len(x_neg), np.int8)])
    forest = _forest(config.forest, int(config.forest.n_jobs))
    forest.fit(x, y)
    forest.set_params(n_jobs=1)                 # prediction parallelises over chunks instead
    quantiles = np.quantile(np.concatenate(samples), _NOVELTY_LEVELS, axis=0).astype(np.float32)
    info = TrainInfo(frames=examples.frames, n_bubble=len(x_pos), n_background=len(x_neg),
                     n_features=x.shape[1], train_ms=(time.perf_counter() - t0) * 1e3)
    ref = None if reference is None else np.array(reference, copy=True)
    return MLModel(forest=forest, config=config, gridx=gx, gridy=gy, frame_shape=frame_shape,
                   reference=ref, quantiles=quantiles, info=info)
