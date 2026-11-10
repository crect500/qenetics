import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)

from qenetics.tools import bootstrap


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = (rng.random((300, 3)) < 0.4).astype(float)
    # Rounded scores, so that some are tied.
    scores = np.round(0.3 * labels + 0.7 * rng.random((300, 3)), 2)
    labels[rng.random((300, 3)) < 0.2] = np.nan
    labels[:, 2] = np.where(np.isnan(labels[:, 2]), np.nan, 1.0)
    return scores, labels


def test_weighted_aucs(data: tuple[np.ndarray, np.ndarray]) -> None:
    scores, labels = data
    observed = ~np.isnan(labels[:, 0])
    s, y = scores[observed, 0], labels[observed, 0]
    weights = np.random.default_rng(1).poisson(1.0, size=(4, len(s)))
    aucs = bootstrap.weighted_aucs(s, y, np.vstack([np.ones(len(s)), weights]))
    assert aucs[0] == pytest.approx(roc_auc_score(y, s))
    for row, auc in zip(weights, aucs[1:], strict=True):
        assert auc == pytest.approx(
            roc_auc_score(np.repeat(y, row), np.repeat(s, row))
        )


def test_scored_cells_and_run_replicates(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    scores, labels = data
    assert bootstrap.scored_cells(labels) == [0, 1]
    weights = bootstrap.site_weights(10, len(labels), 3)
    assert weights.shape == (10, 300) and weights.dtype == np.uint8
    assert (weights == bootstrap.site_weights(10, len(labels), 3)).all()
    values = bootstrap.run_replicates(scores, labels, weights, [0, 1])
    assert values.shape == (11,)
    expected = np.mean(
        [
            roc_auc_score(
                labels[~np.isnan(labels[:, c]), c],
                scores[~np.isnan(labels[:, c]), c],
            )
            for c in (0, 1)
        ]
    )
    assert values[0] == pytest.approx(expected)


def test_model_replicates() -> None:
    rng = np.random.default_rng(0)
    replicate_quantity = 20000
    # Two seeds and no test-set noise: the seed noise has the unbiased
    # variance s^2 / n.
    runs = np.array(
        [[0.70] * (replicate_quantity + 1), [0.72] * (replicate_quantity + 1)]
    )
    replicates = bootstrap.model_replicates(runs, rng, 0.0)
    assert replicates.mean() == pytest.approx(0.71, abs=1e-3)
    assert replicates.var() == pytest.approx(
        np.var([0.70, 0.72], ddof=1) / 2, rel=0.05
    )
    single = bootstrap.model_replicates(runs[:1], rng, 1e-4)
    assert single.std() == pytest.approx(1e-2, rel=0.05)


def test_compare() -> None:
    rng = np.random.default_rng(0)
    runs = 0.7 + rng.normal(0, 0.002, size=(3, 1001))
    same = bootstrap.compare(runs, runs, np.random.default_rng(1))
    assert same.difference == 0.0 and same.p_value > 0.5
    better = bootstrap.compare(runs + 0.02, runs, np.random.default_rng(1))
    assert better.difference == pytest.approx(0.02)
    assert better.low > 0 and better.p_value < 0.01


def test_benjamini_hochberg() -> None:
    q = bootstrap.benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert q == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
    assert len(bootstrap.benjamini_hochberg([])) == 0


def test_pooled_seed_variance() -> None:
    assert bootstrap.pooled_seed_variance(
        [np.array([1.0, 3.0]), np.array([2.0]), np.array([0.0, 1.0, 2.0])]
    ) == pytest.approx((2.0 * 1 + 1.0 * 2) / 3)
    assert np.isnan(bootstrap.pooled_seed_variance([np.array([1.0])]))


def test_cell_metrics(data: tuple[np.ndarray, np.ndarray]) -> None:
    scores, labels = data
    precision, correlation = bootstrap.cell_metrics(
        scores, labels, threshold=0.4
    )
    expected = [
        (
            average_precision_score(labels[m, c] == 1, scores[m, c]),
            matthews_corrcoef(labels[m, c] == 1, scores[m, c] > 0.4),
        )
        for c in (0, 1)
        for m in [~np.isnan(labels[:, c])]
    ]
    assert precision == pytest.approx(np.mean([e[0] for e in expected]))
    assert correlation == pytest.approx(np.mean([e[1] for e in expected]))
    assert all(np.isnan(bootstrap.cell_metrics(scores[:, 2:], labels[:, 2:])))
