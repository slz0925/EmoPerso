"""Four binary Macro-F1 scores and their arithmetic mean, in [0, 1]."""

import numpy as np

from .labels import DIMENSIONS


def _numeric_array(values, name):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real numbers")
    return array


def _binary_matrix(values, name):
    array = _numeric_array(values, name)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 4:
        raise ValueError(f"{name} must have nonempty shape (N, 4)")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary labels 0/1")
    return array.astype(np.int64, copy=False)


def predictions_to_bits(
    predictions, *, prediction_type="labels", threshold=0.5, class_order=(0, 1)
):
    """Convert explicit labels/probabilities/logits to an (N, 4) int64 array.

    Scores shaped (N, 4) refer to class 1 (E/S/F/J). Scores shaped
    (N, 4, 2) use the supplied last-axis class_order, either (0,1) or (1,0).
    Paired probabilities must sum to one. All score ties at the threshold
    select class 1, including probability 0.5 and logit 0 at the default.
    """
    if prediction_type not in ("labels", "probabilities", "logits"):
        raise ValueError("prediction_type must be 'labels', 'probabilities', or 'logits'")
    threshold_array = _numeric_array(threshold, "threshold")
    if threshold_array.ndim != 0 or not 0 < float(threshold_array) < 1:
        raise ValueError("threshold must be a scalar strictly between 0 and 1")
    try:
        order = tuple(class_order)
    except TypeError as exc:
        raise ValueError("class_order must be (0, 1) or (1, 0)") from exc
    if order not in ((0, 1), (1, 0)):
        raise ValueError("class_order must be (0, 1) or (1, 0)")
    if prediction_type == "labels":
        if order != (0, 1):
            raise ValueError("class_order applies only to paired scores shaped (N, 4, 2)")
        return _binary_matrix(predictions, "predictions").copy()

    scores = _numeric_array(predictions, "predictions")
    if (
        scores.ndim not in (2, 3) or scores.shape[0] == 0 or scores.shape[1] != 4
        or (scores.ndim == 3 and scores.shape[2] != 2)
    ):
        raise ValueError("scores must have nonempty shape (N, 4) or (N, 4, 2)")
    if scores.ndim == 2 and order != (0, 1):
        raise ValueError("class_order applies only to paired scores shaped (N, 4, 2)")
    scores = scores.astype(np.float64, copy=False)
    if prediction_type == "probabilities":
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("probabilities must lie in [0, 1]")
        if scores.ndim == 3:
            if not np.allclose(scores.sum(axis=-1), 1.0, rtol=0, atol=1e-6):
                raise ValueError("paired class probabilities must sum to one")
            scores = scores[..., order.index(1)]
        return (scores >= float(threshold_array)).astype(np.int64)

    if scores.ndim == 3:
        # Softmax class-1 odds equal exp(logit_1 - logit_0); avoid softmax overflow.
        with np.errstate(over="ignore"):
            scores = scores[..., order.index(1)] - scores[..., order.index(0)]
    cutoff = np.log(float(threshold_array)) - np.log1p(-float(threshold_array))
    return (scores >= cutoff).astype(np.int64)


def evaluate_mbti(
    y_true, predictions, *, prediction_type="labels", threshold=0.5, class_order=(0, 1)
):
    """Return {I/E, S/N, T/F, P/J, average} Macro-F1 scores.

    Both binary classes are always included, with zero F1 for an undefined
    class. Thus a perfect all-zero column has Macro-F1 0.5, not 1.0.
    No threshold fitting, model selection, or input mutation is performed.
    """
    truth = _binary_matrix(y_true, "y_true")
    predicted = predictions_to_bits(
        predictions, prediction_type=prediction_type, threshold=threshold,
        class_order=class_order,
    )
    if predicted.shape != truth.shape:
        raise ValueError(f"prediction shape {predicted.shape} does not match y_true {truth.shape}")
    result = {}
    for column, dimension in enumerate(DIMENSIONS):
        confusion = np.bincount(
            2 * truth[:, column] + predicted[:, column], minlength=4
        ).reshape(2, 2)
        diagonal = np.diag(confusion).astype(np.float64)
        denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
        f1 = np.divide(2 * diagonal, denominator, out=np.zeros(2), where=denominator != 0)
        result[dimension] = float(f1.mean())
    result["average"] = float(np.mean([result[key] for key in DIMENSIONS]))
    return result
