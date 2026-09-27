import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from qenetics.tools import metrics


def test_generate_confusion_matrix() -> None:
    predictions = np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool)
    truth = np.array([1, 1, 0, 1, 0, 0, 0], dtype=bool)
    assert metrics.generate_confusion_matrix(
        predictions, truth
    ) == metrics.ConfusionMatrix(
        true_positives=2, false_positives=1, false_negatives=1, true_negatives=3
    )


def test_generate_metrics() -> None:
    result = metrics.generate_metrics(
        np.array([1, 1, 1, 0, 0, 0, 0]), np.array([1, 1, 0, 1, 0, 0, 0])
    )
    assert result.accuracy == pytest.approx(5 / 7)
    assert result.true_positive_rate == pytest.approx(2 / 3)
    assert result.false_positive_rate == pytest.approx(1 / 4)
    assert result.f1 == pytest.approx(4 / 6)


def test_cell_metrics() -> None:
    predictions = np.array([0.9, 0.6, 0.4, 0.2, 0.5, 0.7, 0.1])
    labels = np.array([1.0, 1.0, np.nan, 0.0, 0.0, np.nan, 1.0])
    result = metrics.cell_metrics(predictions, labels)

    is_observed = ~np.isnan(labels)
    assert result.observed_sites == 5
    assert result.methylated_sites == 3
    assert result.auc == pytest.approx(
        roc_auc_score(labels[is_observed], predictions[is_observed])
    )
    # A prediction of exactly 0.5 counts as unmethylated.
    assert result.confusion_matrix == metrics.ConfusionMatrix(
        true_positives=2, false_positives=0, false_negatives=1, true_negatives=2
    )
    assert result.accuracy == pytest.approx(4 / 5)
    assert result.roc_false_positive_rates[0] == 0.0
    assert result.roc_true_positive_rates[-1] == 1.0


@pytest.mark.parametrize("labels", [[1.0, 1.0, np.nan], [np.nan, np.nan]])
def test_cell_metrics_undefined_auc(labels: list[float]) -> None:
    result = metrics.cell_metrics(np.full(len(labels), 0.7), np.array(labels))
    assert math.isnan(result.auc)
    assert result.roc_false_positive_rates == []
    assert result.observed_sites == int(np.sum(~np.isnan(labels)))


def test_cell_metrics_thins_roc_curve() -> None:
    rng = np.random.default_rng(0)
    result = metrics.cell_metrics(
        rng.random(5000), rng.integers(0, 2, 5000), maximum_roc_points=20
    )
    assert len(result.roc_false_positive_rates) <= 20
    assert result.roc_false_positive_rates[0] == 0.0
    assert result.roc_false_positive_rates[-1] == 1.0


def test_experiment_metrics() -> None:
    predictions = np.array([[0.9, 0.2], [0.1, 0.8], [0.7, 0.4], [0.3, 0.6]])
    labels = np.array([[1.0, 0.0], [0.0, np.nan], [1.0, 0.0], [0.0, 1.0]])
    cell_results, pooled_result = metrics.experiment_metrics(
        predictions, labels
    )

    assert len(cell_results) == 2
    assert cell_results[0].observed_sites == 4
    assert cell_results[1].observed_sites == 3
    assert pooled_result.observed_sites == 7
    assert pooled_result == metrics.cell_metrics(predictions, labels)

    single_results, _ = metrics.experiment_metrics(
        predictions[:, 0], labels[:, 0]
    )
    assert single_results == [cell_results[0]]
