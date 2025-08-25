from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from numpy.typing import NDArray

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
    return confusion_matrix.true_negatives / denominator


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
        false_positives=int(np.sum(np.logical_and(not_predictions, truth))),
        false_negatives=int(np.sum(np.logical_and(predictions, not_truth))),
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
