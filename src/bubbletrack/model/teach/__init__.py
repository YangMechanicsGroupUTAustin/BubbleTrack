"""ML assist ("Teach"): a pixel classifier that helps Auto find the bubble.

The user teaches with a few examples, a trained classifier gives P(bubble) per
pixel, and the region it proposes guides Auto's wall measurement::

    from bubbletrack.model.teach import MLExamples, measure_frame, train_model

    ex = MLExamples().verify(i, (rc, cc, R))          # a verified frame (manual / locked)
    ex = ex.paint(j, BUBBLE, stroke_pixels(points, 3, shape))   # optional strokes
    model = train_model(ex, read_raw, gridx, gridy, reference=ref)
    res = measure_frame(model, read_raw(k), params=EdgeParams(seed=seed))
    if res.accepted:
        use(res.radius, res.rc, res.cc, res.source)   # "auto_ml" or "ml" (review the latter)

Examples are plain JSON data (``MLExamples.to_dict`` / ``from_dict``): sessions
store them and retrain on load (training is deterministic).

scikit-learn is needed for training (imported lazily, :class:`MLUnavailableError`
without it); everything else is numpy / scipy / scikit-image.  No Qt imports.
"""

from bubbletrack.model.teach.config import ExampleConfig, FeatureConfig, ForestConfig, MeasureConfig, MLConfig
from bubbletrack.model.teach.examples import BACKGROUND, BUBBLE, LABELS, FrameStrokes, MLExamples, stroke_pixels
from bubbletrack.model.teach.features import FeatureStack, compute_features, feature_names
from bubbletrack.model.teach.measure import (
    FLAG_GUIDED,
    FLAG_OUTLINE,
    FLAG_UNFAMILIAR,
    SOURCE_AUTO_ML,
    SOURCE_ML,
    SOURCE_NONE,
    MLResult,
    measure_frame,
    select_result,
    wall_agrees,
)
from bubbletrack.model.teach.model import MLError, MLModel, MLUnavailableError, Prediction, TrainInfo, train_model
from bubbletrack.model.teach.region import MLRegion, ml_region
from bubbletrack.model.teach.sampling import circle_masks, training_pixels

__all__ = [
    "BACKGROUND", "BUBBLE", "FLAG_GUIDED", "FLAG_OUTLINE", "FLAG_UNFAMILIAR", "LABELS", "SOURCE_AUTO_ML",
    "SOURCE_ML", "SOURCE_NONE", "ExampleConfig", "FeatureConfig", "FeatureStack", "ForestConfig", "FrameStrokes",
    "MLConfig", "MLError", "MLExamples", "MLModel", "MLRegion", "MLResult", "MLUnavailableError", "MeasureConfig",
    "Prediction", "TrainInfo", "circle_masks", "compute_features", "feature_names", "measure_frame",
    "ml_region", "select_result", "stroke_pixels", "train_model", "training_pixels", "wall_agrees",
]
