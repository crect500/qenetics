"""
Readouts of simulated measurement probabilities, and finite-shot estimates.

Circuits are simulated exactly. `sample_probabilities` replaces exact
probabilities by the frequencies of a finite number of measurements, drawn
exactly as a multinomial, so that a trained model can be evaluated as if its
circuits were measured `shots` times each.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

PROBABILITY_READOUT: str = "probability"
EXPECTATION_READOUT: str = "expectation"
MARGINAL_READOUT: str = "marginal"
READOUTS: tuple[str, ...] = (
    PROBABILITY_READOUT,
    EXPECTATION_READOUT,
    MARGINAL_READOUT,
)


def check_wires(wires: Sequence[int], wire_quantity: int) -> None:
    """
    Raises
    ------
    ValueError if a wire is not in a register of wire_quantity wires.
    """
    invalid: list[int] = [
        wire for wire in wires if not 0 <= wire < wire_quantity
    ]
    if invalid:
        raise ValueError(
            f"Wires {invalid} are not in a register of {wire_quantity} wires"
        )


def wire_bits(wire_quantity: int, wires: Sequence[int]) -> NDArray[np.float32]:
    """
    Find the value of wires in every basis state.

    Args
    ----
    wire_quantity: The number of wires of the register, wire 0 being the
        most significant bit.
    wires: The wires to read.

    Returns
    -------
    A (2^n, len(wires)) array, 1 where a wire is |1>. Probabilities times it
    give P(wire = 1) of each wire.

    Raises
    ------
    ValueError if a wire is not in the register.
    """
    check_wires(wires, wire_quantity)
    states: NDArray[np.int64] = np.arange(2**wire_quantity)[:, np.newaxis]
    shifts: NDArray[np.int64] = wire_quantity - 1 - np.asarray(wires)
    return ((states >> shifts[np.newaxis, :]) & 1).astype(np.float32)


def expectation_matrix(wire_quantity: int) -> NDArray[np.float32]:
    """Probabilities times this (2^n, n) matrix give <Z> of every wire."""
    return 1 - 2 * wire_bits(wire_quantity, range(wire_quantity))


def marginal_matrix(
    wire_quantity: int, marginal_wire_quantity: int
) -> NDArray[np.float32]:
    """
    Probabilities times this (2^n, 2^r) matrix give the marginal
    probabilities of the first r wires.

    Raises
    ------
    ValueError if r is not between 1 and n.
    """
    if not 1 <= marginal_wire_quantity <= wire_quantity:
        raise ValueError(
            f"Cannot take the marginal of {marginal_wire_quantity} wires of "
            f"a register of {wire_quantity} wires"
        )

    states: NDArray[np.int64] = np.arange(2**wire_quantity)
    leading: NDArray[np.int64] = states >> (
        wire_quantity - marginal_wire_quantity
    )
    return (
        leading[:, np.newaxis] == np.arange(2**marginal_wire_quantity)
    ).astype(np.float32)


def _check_shots(shots: int) -> None:
    if shots < 1:
        raise ValueError(f"Shots must be at least 1, not {shots}")


def sample_probabilities(
    probabilities: torch.Tensor,
    shots: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Estimate probabilities from a finite number of measurements.

    The counts of `shots` measurements are drawn exactly as a multinomial,
    down a binary tree over the outcomes: each node's count is split between
    its halves by a binomial draw, so d outcomes take log2(d) vectorized
    draws. Rows that sum to less than 1, such as amplitude-encoded windows
    holding unknown nucleotides, are sampled from their normalized
    distribution and rescaled by their sum.

    Args
    ----
    probabilities: The probabilities, (..., d) with d a power of two.
    shots: The number of measurements of each row.
    generator: The random number generator.

    Returns
    -------
    The measured frequencies, of the probabilities' shape and dtype.

    Raises
    ------
    ValueError if shots is less than 1 or d is not a power of two.
    """
    _check_shots(shots)
    outcome_quantity: int = probabilities.shape[-1]
    if outcome_quantity < 1 or outcome_quantity & (outcome_quantity - 1):
        raise ValueError(
            "The number of outcomes must be a power of two, not "
            f"{outcome_quantity}"
        )

    totals: torch.Tensor = probabilities.sum(-1, keepdim=True)
    normalized: torch.Tensor = (
        (probabilities / totals.clamp_min(1e-12)).clamp_min(0).float()
    )
    # masses[k] holds the summed probabilities of 2^k neighbouring outcomes.
    masses: list[torch.Tensor] = [normalized]
    while masses[-1].shape[-1] > 1:
        masses.append(masses[-1].reshape(*masses[-1].shape[:-1], -1, 2).sum(-1))

    counts: torch.Tensor = torch.full(
        (*normalized.shape[:-1], 1), float(shots), device=normalized.device
    )
    for mass in reversed(masses[:-1]):
        pairs: torch.Tensor = mass.reshape(*mass.shape[:-1], -1, 2)
        left_fractions: torch.Tensor = (
            pairs[..., 0] / pairs.sum(-1).clamp_min(1e-12)
        ).clamp(0, 1)
        left_counts: torch.Tensor = torch.binomial(
            counts, left_fractions, generator=generator
        )
        counts = torch.stack([left_counts, counts - left_counts], -1).reshape(
            *counts.shape[:-1], -1
        )
    return (counts / shots * totals).to(probabilities.dtype)


def sample_bernoulli_frequencies(
    probabilities: torch.Tensor,
    shots: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Estimate probabilities of single outcomes, such as P(wire = 1), from
    `shots` measurements each.

    Raises
    ------
    ValueError if shots is less than 1.
    """
    _check_shots(shots)
    counts: torch.Tensor = torch.binomial(
        torch.full_like(probabilities, float(shots), dtype=torch.float),
        probabilities.float().clamp(0, 1),
        generator=generator,
    )
    return (counts / shots).to(probabilities.dtype)
