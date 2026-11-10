"""
The paired bootstrap of model comparisons in the quantum models report
(Section 5, "Statistics", revised 2026-10-10), and per-cell secondary metrics.

A comparison of models A and B is

    D = mean over A's seeds of its mean per-cell test AUC - the same for B.

Each of R replicates combines, for each model:

- test-set noise: Poisson(1) weights of the test sites, the same weights for
  both models, so that both are scored on the same resampled test set, averaged
  over the model's seeds;
- seed noise: the model's seeds resampled with replacement, with the deviation
  of their mean from the mean over all seeds scaled by sqrt(n / (n - 1)), so
  that its variance is the unbiased s^2 / n. A model with a single seed gets a
  normal deviation with a pooled seed variance instead.

The p value is two-sided: twice the share of replicates on the far side of zero
from D. Tied scores count one half in every AUC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, matthews_corrcoef

# Replicates per weighted-AUC evaluation, which bounds its memory.
REPLICATE_CHUNK: int = 250


def site_weights(
    replicate_quantity: int, site_quantity: int, seed: int
) -> NDArray[np.uint8]:
    """
    Poisson(1) weights of the test sites, one row per replicate.

    Args
    ----
    replicate_quantity: The replicates R.
    site_quantity: The test sites.
    seed: The seed; every model scored on the same test set uses the same.

    Returns
    -------
    The (R, sites) weights.
    """
    return (
        np.random.default_rng(seed)
        .poisson(1.0, size=(replicate_quantity, site_quantity))
        .astype(np.uint8)
    )


def weighted_aucs(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    weights: NDArray[np.integer],
) -> NDArray[np.float64]:
    """
    The AUC of the scores for each row of site weights, ties counting 1/2.

    Args
    ----
    scores: The (n,) scores of the observed sites.
    labels: Their (n,) binary labels.
    weights: (k, n) weights of the sites.

    Returns
    -------
    The (k,) AUCs.
    """
    order: NDArray[np.int64] = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    positives: NDArray[np.uint8] = (labels[order] == 1).astype(np.uint8)
    ordered_weights = weights[:, order]
    starts: NDArray[np.int64] = np.flatnonzero(
        np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    )
    positive = np.add.reduceat(
        ordered_weights * positives, starts, axis=1, dtype=np.int64
    )
    negative = np.add.reduceat(
        ordered_weights * (1 - positives), starts, axis=1, dtype=np.int64
    )
    below = np.cumsum(negative, axis=1) - negative
    return (positive * (below + 0.5 * negative)).sum(1) / (
        positive.sum(1) * negative.sum(1)
    )


def scored_cells(labels: NDArray[np.floating]) -> list[int]:
    """The cells with both classes among their observed labels."""
    observed = ~np.isnan(labels)
    return [
        cell
        for cell in range(labels.shape[1])
        if 0 < np.nansum(labels[:, cell]) < observed[:, cell].sum()
    ]


def run_replicates(
    predictions: NDArray[np.floating],
    labels: NDArray[np.floating],
    weights: NDArray[np.integer],
    cells: list[int],
) -> NDArray[np.float64]:
    """
    A run's mean per-cell AUC, unweighted and for each replicate.

    Args
    ----
    predictions: The (sites, cells) test predictions.
    labels: The (sites, cells) labels, NaN if unobserved.
    weights: The (R, sites) site weights.
    cells: The cells to average over, the same for every model compared.

    Returns
    -------
    (R + 1,) values: the unweighted mean per-cell AUC, then each replicate's.
    """
    per_cell: list[NDArray[np.float64]] = []
    for cell in cells:
        observed = ~np.isnan(labels[:, cell])
        scores = predictions[observed, cell].astype(np.float32)
        cell_labels = labels[observed, cell]
        values = [
            weighted_aucs(
                scores, cell_labels, np.ones((1, len(scores)), np.uint8)
            )
        ]
        for start in range(0, len(weights), REPLICATE_CHUNK):
            values.append(
                weighted_aucs(
                    scores,
                    cell_labels,
                    weights[start : start + REPLICATE_CHUNK][:, observed],
                )
            )
        per_cell.append(np.concatenate(values))
    return np.mean(per_cell, axis=0)


def model_replicates(
    runs: NDArray[np.float64],
    rng: np.random.Generator,
    single_seed_variance: float,
) -> NDArray[np.float64]:
    """
    Replicates of a model's mean over seeds, with test-set and seed noise.

    Args
    ----
    runs: (seeds, R + 1) values of `run_replicates`, one row per seed.
    rng: The random generator.
    single_seed_variance: The seed variance of a model with one seed.

    Returns
    -------
    The (R,) replicates.
    """
    seed_quantity: int = len(runs)
    mean: float = float(runs[:, 0].mean())
    replicate_quantity: int = runs.shape[1] - 1
    test_noise = runs[:, 1:].mean(0) - mean
    if seed_quantity >= 2:
        indices = rng.integers(
            0, seed_quantity, size=(replicate_quantity, seed_quantity)
        )
        seed_noise = np.sqrt(seed_quantity / (seed_quantity - 1)) * (
            runs[indices, 0].mean(1) - mean
        )
    else:
        seed_noise = rng.normal(
            0.0, np.sqrt(single_seed_variance), size=replicate_quantity
        )
    return mean + test_noise + seed_noise


def pooled_seed_variance(seed_values: list[NDArray[np.floating]]) -> float:
    """
    The pooled within-model variance (ddof 1) of models with several seeds.

    Args
    ----
    seed_values: Each model's per-seed values.

    Returns
    -------
    The pooled variance, or NaN if no model has two seeds.
    """
    total: float = 0.0
    degrees: int = 0
    for values in seed_values:
        if len(values) >= 2:
            total += float(np.var(values, ddof=1)) * (len(values) - 1)
            degrees += len(values) - 1
    return total / degrees if degrees else float("nan")


@dataclass
class Comparison:
    """
    The bootstrap of a difference A - B.

    Attributes
    ----------
    difference: D, from the unweighted AUCs.
    low, high: The 95% percentile interval.
    p_value: The two-sided p value.
    """

    difference: float
    low: float
    high: float
    p_value: float


def compare(
    runs_a: NDArray[np.float64],
    runs_b: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    single_seed_variance_a: float = 0.0,
    single_seed_variance_b: float = 0.0,
) -> Comparison:
    """
    Bootstrap the difference of two models scored on the same test set.

    Args
    ----
    runs_a, runs_b: (seeds, R + 1) values of `run_replicates` of each model,
        computed with the same site weights.
    rng: The random generator.
    single_seed_variance_a, single_seed_variance_b: The seed variances used
        if a model has one seed.

    Returns
    -------
    The comparison.
    """
    difference: float = float(runs_a[:, 0].mean() - runs_b[:, 0].mean())
    replicates = model_replicates(
        runs_a, rng, single_seed_variance_a
    ) - model_replicates(runs_b, rng, single_seed_variance_b)
    low, high = np.percentile(replicates, [2.5, 97.5])
    far_side = (replicates <= 0) if difference > 0 else (replicates >= 0)
    p_value: float = min(
        1.0, 2 * (1 + int(far_side.sum())) / (len(replicates) + 1)
    )
    return Comparison(difference, float(low), float(high), p_value)


def benjamini_hochberg(p_values: list[float]) -> NDArray[np.float64]:
    """The Benjamini–Hochberg q values of p values."""
    p = np.asarray(p_values, float)
    if not len(p):
        return p
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    return np.minimum(q, 1.0)


def cell_metrics(
    predictions: NDArray[np.floating],
    labels: NDArray[np.floating],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    The mean per-cell average precision (AUPRC) and Matthews correlation.

    Args
    ----
    predictions: The (sites, cells) predictions.
    labels: The (sites, cells) labels, NaN if unobserved.
    threshold: The prediction above which a site is called methylated, for
        the Matthews correlation.

    Returns
    -------
    The means over cells with both classes, NaN if there is none.
    """
    precisions: list[float] = []
    correlations: list[float] = []
    for cell in scored_cells(labels):
        observed = ~np.isnan(labels[:, cell])
        truth = labels[observed, cell] == 1
        scores = predictions[observed, cell]
        precisions.append(float(average_precision_score(truth, scores)))
        correlations.append(float(matthews_corrcoef(truth, scores > threshold)))
    if not precisions:
        return float("nan"), float("nan")
    return float(np.mean(precisions)), float(np.mean(correlations))
