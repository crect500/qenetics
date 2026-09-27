from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

METRICS_HEADERS: str = (
    "accuracy,tpr,fpr,f1,true_positives,false_positives,"
    "false_negatives,true_negatives"
)


@dataclass
class ConfusionMatrix:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


@dataclass
class Metrics:
    confusion_matrix: ConfusionMatrix
    accuracy: float
    true_positive_rate: float
    false_positive_rate: float
    f1: float


def calculate_accuracy(confusion_matrix: ConfusionMatrix) -> float:
    denominator: float = (
        confusion_matrix.true_positives
        + confusion_matrix.false_positives
        + confusion_matrix.false_negatives
        + confusion_matrix.true_negatives
    )
    if denominator == 0.0:
        return denominator
    return (
        confusion_matrix.true_positives + confusion_matrix.true_negatives
    ) / denominator


def calculate_tpr(confusion_matrix: ConfusionMatrix) -> float:
    denominator: float = (
        confusion_matrix.true_positives + confusion_matrix.false_negatives
    )
    if denominator == 0.0:
        return denominator
    return confusion_matrix.true_positives / denominator


def calculate_fpr(confusion_matrix: ConfusionMatrix) -> float:
    denominator: float = (
        confusion_matrix.true_negatives + confusion_matrix.false_positives
    )
    if denominator == 0.0:
        return denominator
    return confusion_matrix.false_positives / denominator


def calculate_f1_score(confusion_matrix: ConfusionMatrix) -> float:
    denominator: float = (
        2 * confusion_matrix.true_positives
        + confusion_matrix.false_positives
        + confusion_matrix.false_negatives
    )
    if denominator == 0.0:
        return denominator
    return 2 * confusion_matrix.true_positives / denominator


def generate_confusion_matrix(
    predictions: NDArray[int], truth: NDArray[int]
) -> ConfusionMatrix:
    not_predictions: NDArray[bool] = np.logical_not(predictions)
    not_truth: NDArray[bool] = np.logical_not(truth)
    return ConfusionMatrix(
        true_positives=int(np.sum(np.logical_and(predictions, truth))),
        false_positives=int(np.sum(np.logical_and(predictions, not_truth))),
        false_negatives=int(np.sum(np.logical_and(not_predictions, truth))),
        true_negatives=int(np.sum(np.logical_and(not_predictions, not_truth))),
    )


def generate_metrics(predictions: NDArray[int], truth: NDArray[int]) -> Metrics:
    logger.debug(f"Predictions: {predictions}")
    logger.debug(f"truth: {truth}")
    confusion_matrix = generate_confusion_matrix(predictions, truth)
    return Metrics(
        confusion_matrix=confusion_matrix,
        accuracy=calculate_accuracy(confusion_matrix),
        true_positive_rate=calculate_tpr(confusion_matrix),
        false_positive_rate=calculate_fpr(confusion_matrix),
        f1=calculate_f1_score(confusion_matrix),
    )


@dataclass
class CellMetrics:
    """
    Classification performance on the observed sites of one cell or pool.

    Attributes
    ----------
    observed_sites: The quantity of sites with an observed label.
    methylated_sites: The quantity of observed sites labeled methylated.
    auc: The area under the ROC curve, or NaN if only one class is observed.
    accuracy: The fraction of observed sites classified correctly.
    true_positive_rate: The fraction of methylated sites predicted methylated.
    false_positive_rate: The fraction of unmethylated sites predicted
        methylated.
    f1: The F1 score.
    confusion_matrix: The confusion matrix of the observed sites.
    roc_false_positive_rates: The false positive rates of the ROC curve.
    roc_true_positive_rates: The true positive rates of the ROC curve.
    """

    observed_sites: int
    methylated_sites: int
    auc: float
    accuracy: float
    true_positive_rate: float
    false_positive_rate: float
    f1: float
    confusion_matrix: ConfusionMatrix
    roc_false_positive_rates: list[float]
    roc_true_positive_rates: list[float]


def _thin_curve(
    x_values: NDArray[float], y_values: NDArray[float], maximum_points: int
) -> tuple[list[float], list[float]]:
    """
    Reduce a curve to at most `maximum_points` points, keeping both ends.

    Args
    ----
    x_values: The x coordinates.
    y_values: The y coordinates.
    maximum_points: The maximum quantity of points to keep.

    Returns
    -------
    The kept x and y coordinates.
    """
    if len(x_values) > maximum_points:
        indices = np.unique(
            np.linspace(0, len(x_values) - 1, maximum_points).round()
        ).astype(int)
        x_values, y_values = x_values[indices], y_values[indices]

    return x_values.tolist(), y_values.tolist()


def cell_metrics(
    predictions: NDArray[float],
    labels: NDArray[float],
    threshold: float = 0.5,
    *,
    maximum_roc_points: int = 201,
) -> CellMetrics:
    """
    Measure how well predictions match the observed labels of one cell.

    Labels that are NaN are unobserved and left out. Following Angermueller
    et al. (2017), predictions greater than the threshold count as methylated.

    Args
    ----
    predictions: The predicted methylation probability of each site.
    labels: The binary methylation label of each site, NaN if unobserved.
    threshold: The probability above which a prediction counts as methylated.
    maximum_roc_points: The maximum quantity of ROC curve points to keep.

    Returns
    -------
    The classification performance on the observed sites.
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    is_observed: NDArray[np.bool_] = ~np.isnan(labels)
    predictions, labels = predictions[is_observed], labels[is_observed]
    truth: NDArray[np.bool_] = labels == 1.0
    confusion_matrix: ConfusionMatrix = generate_confusion_matrix(
        predictions > threshold, truth
    )

    if 0 < truth.sum() < len(truth):
        auc: float = float(roc_auc_score(truth, predictions))
        false_positive_rates, true_positive_rates, _ = roc_curve(
            truth, predictions
        )
        roc_x, roc_y = _thin_curve(
            false_positive_rates, true_positive_rates, maximum_roc_points
        )
    else:
        auc = float("nan")
        roc_x, roc_y = [], []

    return CellMetrics(
        observed_sites=len(labels),
        methylated_sites=int(truth.sum()),
        auc=auc,
        accuracy=calculate_accuracy(confusion_matrix),
        true_positive_rate=calculate_tpr(confusion_matrix),
        false_positive_rate=calculate_fpr(confusion_matrix),
        f1=calculate_f1_score(confusion_matrix),
        confusion_matrix=confusion_matrix,
        roc_false_positive_rates=roc_x,
        roc_true_positive_rates=roc_y,
    )


def experiment_metrics(
    predictions: NDArray[float],
    labels: NDArray[float],
    threshold: float = 0.5,
) -> tuple[list[CellMetrics], CellMetrics]:
    """
    Measure performance for each cell and for all cells pooled.

    Args
    ----
    predictions: The predictions, with one column per cell.
    labels: The labels, with one column per cell and NaN if unobserved.
    threshold: The probability above which a prediction counts as methylated.

    Returns
    -------
    The performance of each cell, in column order, and of all cells pooled.
    """
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    if labels.ndim == 1:
        labels = labels[:, np.newaxis]

    return (
        [
            cell_metrics(predictions[:, cell], labels[:, cell], threshold)
            for cell in range(labels.shape[1])
        ],
        cell_metrics(predictions, labels, threshold),
    )


def metrics_to_csv_row(metrics: Metrics) -> str:
    row: str = str(metrics.accuracy)
    row += ","
    row += str(metrics.true_positive_rate)
    row += ","
    row += str(metrics.false_positive_rate)
    row += ","
    row += str(metrics.f1)
    row += ","
    row += str(metrics.confusion_matrix.true_positives)
    row += ","
    row += str(metrics.confusion_matrix.false_positives)
    row += ","
    row += str(metrics.confusion_matrix.false_negatives)
    row += ","
    row += str(metrics.confusion_matrix.true_negatives)

    return row
