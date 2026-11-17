"""
Interaction-order diagnostics of a function of one-hot sequences.

A function f of a sequence has interaction order k if it is a sum of terms
that each depend on at most k positions. Two measurements test it:

- Three-way epistasis. For a sequence s, three positions and a substitute
  base at each,

      e3 = sum over subsets B of the positions of (-1)^(3 - |B|) f(s_B),

  where s_B has the substitutions at B applied. It is zero for every triple
  if and only if f has order at most 2. The pairwise e2 of each pair of the
  triple, from the same eight values, gives its scale.
- Mean dimension. With independent positions,

      sum over j of E[Var over the base at j of f] / Var(f)

  is the mean, weighted by variance, of the number of positions in the terms
  of f's ANOVA decomposition. It is at most k for a function of order k.

Functions map (n, length, 4) one-hot sequences to (n,) or (n, outputs)
values. Evaluate logits rather than probabilities: a sigmoid alone creates
every order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

SequenceFunction = Callable[[Tensor], Tensor]
WINDOW_STRATUM: str = "window"
NEAR_STRATUM: str = "near"
ANYWHERE_STRATUM: str = "anywhere"
# The span in bp that holds a triple of each stratum; None for any span.
STRATUM_SPANS: dict[str, int | None] = {
    WINDOW_STRATUM: 16,
    NEAR_STRATUM: 100,
    ANYWHERE_STRATUM: None,
}
# The subsets of a triple's positions, as substituted (1) or not (0), and the
# sign of each in e3.
_CORNERS: tuple[tuple[int, int, int], ...] = tuple(
    (first, second, third)
    for first in (0, 1)
    for second in (0, 1)
    for third in (0, 1)
)


def _evaluate(
    function: SequenceFunction, sequences: Tensor, batch_size: int
) -> Tensor:
    """Apply the function in batches, without gradients."""
    with torch.no_grad():
        return torch.cat(
            [
                function(sequences[start : start + batch_size])
                for start in range(0, len(sequences), batch_size)
            ]
        )


def draw_triples(
    sequences: Tensor,
    triple_quantity: int,
    stratum: str,
    generator: torch.Generator,
    excluded_positions: Iterable[int] = (),
) -> tuple[Tensor, Tensor]:
    """
    Draw triples of positions and a substitute base at each.

    Args
    ----
    sequences: (n, length, 4) one-hot sequences.
    triple_quantity: The triples per sequence.
    stratum: `WINDOW_STRATUM` for three positions within 16 bp,
        `NEAR_STRATUM` within 100 bp, or `ANYWHERE_STRATUM`.
    generator: The random generator.
    excluded_positions: Positions never substituted, such as the central CpG.

    Returns
    -------
    The (n, triples, 3) positions, increasing within a triple, and the
    (n, triples, 3) substitute bases, each drawn uniformly from the three
    bases other than the sequence's own.

    Raises
    ------
    ValueError if the stratum is unknown or too few positions are allowed.
    """
    if stratum not in STRATUM_SPANS:
        raise ValueError(
            f"Unknown stratum {stratum}; expected one of "
            f"{', '.join(STRATUM_SPANS)}"
        )
    sequence_quantity, length, _ = sequences.shape
    span: int = min(STRATUM_SPANS[stratum] or length, length)
    allowed = torch.ones(length, dtype=torch.bool)
    allowed[list(excluded_positions)] = False
    # Starts whose span holds at least three allowed positions.
    counts = allowed.float().unfold(0, span, 1).sum(1)
    starts = torch.nonzero(counts >= 3).flatten()
    if not len(starts):
        raise ValueError(
            f"No span of {span} positions holds three positions to substitute"
        )
    shape = (sequence_quantity, triple_quantity)
    start = starts[
        torch.randint(len(starts), shape, generator=generator)
    ].unsqueeze(-1)
    # Three distinct allowed offsets of each span: the largest random keys.
    keys = torch.rand((*shape, span), generator=generator)
    keys[~allowed[start + torch.arange(span)]] = -1.0
    positions = (start + keys.topk(3, dim=-1).indices).sort(-1).values
    bases = sequences.argmax(-1)
    own = bases[torch.arange(sequence_quantity)[:, None, None], positions]
    substitutes = (
        own + torch.randint(1, 4, positions.shape, generator=generator)
    ) % 4
    return positions, substitutes


@dataclass
class Epistasis:
    """
    The epistasis of triples.

    Attributes
    ----------
    three_way: e3 of each triple, (n, triples) or (n, triples, outputs).
    pairwise: e2 of the three pairs of each triple, with the third position
        unsubstituted: (n, triples, 3) or (n, triples, 3, outputs), for the
        pairs of positions (0, 1), (0, 2) and (1, 2).
    """

    three_way: Tensor
    pairwise: Tensor


def triple_epistasis(
    function: SequenceFunction,
    sequences: Tensor,
    positions: Tensor,
    substitutes: Tensor,
    batch_size: int = 1024,
) -> Epistasis:
    """
    Measure the three-way and pairwise epistasis of triples of substitutions.

    Args
    ----
    function: The function of (n, length, 4) one-hot sequences.
    sequences: (n, length, 4) one-hot sequences.
    positions: (n, triples, 3) positions, see `draw_triples`.
    substitutes: (n, triples, 3) substitute bases.
    batch_size: The sequences per call of the function.

    Returns
    -------
    The epistasis.
    """
    sequence_quantity, triple_quantity, _ = positions.shape
    triple_index = torch.arange(triple_quantity)
    replacement = torch.nn.functional.one_hot(substitutes, 4).to(
        sequences.dtype
    )
    per_sequence: list[Tensor] = []
    for index, sequence in enumerate(sequences):
        # (triples, 8, length, 4): each triple's eight variants.
        variants = sequence.expand(
            triple_quantity, len(_CORNERS), -1, -1
        ).clone()
        for corner_index, substituted in enumerate(_CORNERS):
            for member, is_substituted in enumerate(substituted):
                if is_substituted:
                    variants[
                        triple_index, corner_index, positions[index, :, member]
                    ] = replacement[index, :, member]
        per_sequence.append(
            _evaluate(function, variants.flatten(0, 1), batch_size)
        )
    values = torch.stack(per_sequence)
    values = values.view(
        sequence_quantity, triple_quantity, len(_CORNERS), *values.shape[2:]
    )

    def corner(*substituted: int) -> Tensor:
        return values[:, :, _CORNERS.index(substituted)]

    three_way = sum(
        (-1) ** (3 - sum(substituted)) * corner(*substituted)
        for substituted in _CORNERS
    )
    pairwise = torch.stack(
        [
            corner(*both) - corner(*first) - corner(*second) + corner(0, 0, 0)
            for both, first, second in (
                ((1, 1, 0), (1, 0, 0), (0, 1, 0)),
                ((1, 0, 1), (1, 0, 0), (0, 0, 1)),
                ((0, 1, 1), (0, 1, 0), (0, 0, 1)),
            )
        ],
        dim=2,
    )
    return Epistasis(three_way, pairwise)


def draw_independent_sequences(
    frequencies: Tensor, sequence_quantity: int, generator: torch.Generator
) -> Tensor:
    """
    Draw one-hot sequences with independent positions.

    Args
    ----
    frequencies: (length, 4) base frequencies of each position; a position
        with one base of frequency 1 is fixed.
    sequence_quantity: The sequences to draw.
    generator: The random generator.

    Returns
    -------
    (n, length, 4) one-hot sequences, of the frequencies' type.
    """
    bases = torch.multinomial(
        frequencies.double(),
        sequence_quantity,
        replacement=True,
        generator=generator,
    ).T
    return torch.nn.functional.one_hot(bases, 4).to(frequencies.dtype)


@dataclass
class MeanDimension:
    """
    A mean dimension estimate.

    Attributes
    ----------
    value: The mean dimension, one per output.
    standard_error: The standard error of its numerator over sequences,
        divided by the variance; the variance's own error is not included.
    sequence_quantity: The sequences of the numerator.
    """

    value: Tensor
    standard_error: Tensor
    sequence_quantity: int


def mean_dimension(
    function: SequenceFunction,
    frequencies: Tensor,
    sequence_quantity: int,
    generator: torch.Generator,
    variance_sequence_quantity: int = 20000,
    batch_size: int = 1024,
) -> MeanDimension:
    """
    Estimate the mean dimension of a function under independent positions.

    For each drawn sequence and position, the function is evaluated with each
    of the four bases at that position, and the variance over them is
    weighted by the position's frequencies. Positions fixed by the
    frequencies (one base of frequency 1) contribute nothing and are skipped.

    Args
    ----
    function: The function of (n, length, 4) one-hot sequences.
    frequencies: (length, 4) base frequencies of each position, rows summing
        to 1.
    sequence_quantity: The sequences of the numerator; each costs four
        evaluations per free position.
    generator: The random generator.
    variance_sequence_quantity: Further sequences, one evaluation each, for
        the variance of the function.
    batch_size: The sequences per call of the function.

    Returns
    -------
    The estimate.
    """
    length: int = len(frequencies)
    free = torch.nonzero(frequencies.max(1).values < 1).flatten()
    sequences = draw_independent_sequences(
        frequencies, sequence_quantity, generator
    )
    identity = torch.eye(4, dtype=frequencies.dtype)
    totals: list[Tensor] = []
    for sequence in sequences:
        # (free positions, 4, length, 4): every base at every free position.
        variants = sequence.expand(len(free), 4, length, 4).clone()
        variants[torch.arange(len(free)), :, free] = identity
        values = _evaluate(function, variants.flatten(0, 1), batch_size)
        values = values.view(len(free), 4, -1)
        weights = frequencies[free].to(values.dtype)[:, :, None]
        mean = (weights * values).sum(1, keepdim=True)
        totals.append((weights * (values - mean) ** 2).sum(1).sum(0))
    numerator = torch.stack(totals)
    variance = _evaluate(
        function,
        draw_independent_sequences(
            frequencies, variance_sequence_quantity, generator
        ),
        batch_size,
    )
    variance = variance.view(variance_sequence_quantity, -1).var(0)
    return MeanDimension(
        numerator.mean(0) / variance,
        numerator.std(0) / sequence_quantity**0.5 / variance,
        sequence_quantity,
    )
