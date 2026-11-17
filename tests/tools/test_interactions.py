import pytest
import torch

from qenetics.tools import interactions

LENGTH: int = 24


def _sequences(quantity: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    bases = torch.randint(0, 4, (quantity, LENGTH), generator=generator)
    return torch.nn.functional.one_hot(bases, 4).double()


def _quadratic(seed: int = 1):
    """A function of interaction order 2: linear terms and every pair."""
    generator = torch.Generator().manual_seed(seed)
    linear = torch.randn(LENGTH * 4, generator=generator, dtype=torch.double)
    pairs = torch.randn(
        LENGTH * 4, LENGTH * 4, generator=generator, dtype=torch.double
    )
    return lambda x: (
        x.flatten(1) @ linear + ((x.flatten(1) @ pairs) * x.flatten(1)).sum(1)
    )


def _cubic(x: torch.Tensor) -> torch.Tensor:
    """A pure three-way term of positions 2, 5 and 11 under uniform bases."""
    return (x[:, 2, 0] - 0.25) * (x[:, 5, 1] - 0.25) * (x[:, 11, 3] - 0.25)


@pytest.mark.parametrize("stratum", list(interactions.STRATUM_SPANS))
def test_draw_triples(stratum: str) -> None:
    sequences = _sequences(5)
    generator = torch.Generator().manual_seed(3)
    positions, substitutes = interactions.draw_triples(
        sequences, 50, stratum, generator, excluded_positions=(12, 13)
    )
    assert positions.shape == substitutes.shape == (5, 50, 3)
    assert not torch.isin(positions, torch.tensor([12, 13])).any()
    assert (positions[..., 0] < positions[..., 1]).all()
    assert (positions[..., 1] < positions[..., 2]).all()
    span = interactions.STRATUM_SPANS[stratum] or LENGTH
    assert (positions[..., 2] - positions[..., 0] < span).all()
    own = sequences.argmax(-1)[torch.arange(5)[:, None, None], positions]
    assert (substitutes != own).all()
    assert ((substitutes >= 0) & (substitutes < 4)).all()


def test_draw_triples_errors() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="Unknown stratum"):
        interactions.draw_triples(_sequences(1), 1, "far", generator)
    with pytest.raises(ValueError, match="three positions"):
        interactions.draw_triples(
            _sequences(1), 1, "window", generator, range(2, LENGTH)
        )


def test_triple_epistasis() -> None:
    sequences = _sequences(6)
    generator = torch.Generator().manual_seed(4)
    positions, substitutes = interactions.draw_triples(
        sequences, 40, interactions.ANYWHERE_STRATUM, generator
    )
    quadratic = interactions.triple_epistasis(
        _quadratic(), sequences, positions, substitutes, batch_size=64
    )
    assert quadratic.three_way.shape == (6, 40)
    assert quadratic.pairwise.shape == (6, 40, 3)
    assert quadratic.three_way.abs().max() < 1e-9
    assert quadratic.pairwise.abs().median() > 0.1

    # Order 3: substituting all of positions 2, 5 and 11 changes each factor.
    positions = torch.tensor([[[2, 5, 11]]]).expand(6, 1, 3)
    own = sequences.argmax(-1)[:, [2, 5, 11]]
    targets = torch.tensor([0, 1, 3])
    substitutes = torch.where(own == targets, (own + 1) % 4, targets)[:, None]
    cubic = interactions.triple_epistasis(
        _cubic, sequences, positions, substitutes
    )
    # Each factor changes by +-1, so e3 is the product of the changes.
    assert torch.allclose(cubic.three_way.abs(), torch.ones(6, 1).double())

    # Several outputs keep their own axis.
    both = interactions.triple_epistasis(
        lambda x: torch.stack([_cubic(x), 2 * _cubic(x)], 1),
        sequences,
        positions,
        substitutes,
    )
    assert both.three_way.shape == (6, 1, 2)
    assert both.pairwise.shape == (6, 1, 3, 2)
    assert torch.allclose(both.three_way[..., 1], 2 * cubic.three_way)


def test_mean_dimension() -> None:
    uniform = torch.full((LENGTH, 4), 0.25, dtype=torch.double)
    generator = torch.Generator().manual_seed(5)
    weights = torch.randn(LENGTH * 4, generator=generator, dtype=torch.double)

    additive = interactions.mean_dimension(
        lambda x: x.flatten(1) @ weights, uniform, 20, generator
    )
    assert additive.value.shape == additive.standard_error.shape == (1,)
    assert additive.value.item() == pytest.approx(1.0, abs=0.05)
    assert additive.sequence_quantity == 20

    quadratic = interactions.mean_dimension(
        _quadratic(), uniform, 60, generator
    )
    assert 1.0 < quadratic.value.item() < 2.1

    cubic = interactions.mean_dimension(_cubic, uniform, 400, generator)
    assert cubic.value.item() == pytest.approx(3.0, abs=0.4)

    # A fixed position is never varied: with positions 2, 5 and 11 fixed the
    # cubic term is constant, and a function of them has no variance.
    fixed = uniform.clone()
    fixed[[2, 5, 11]] = torch.tensor([1.0, 0, 0, 0], dtype=torch.double)
    sequences = interactions.draw_independent_sequences(fixed, 30, generator)
    assert (sequences[:, [2, 5, 11], 0] == 1).all()
    skewed = uniform.clone()
    skewed[2] = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.double)
    # Under non-uniform frequencies the weighted variances still give 1 for
    # an additive function.
    additive = interactions.mean_dimension(
        lambda x: x.flatten(1) @ weights, skewed, 20, generator
    )
    assert additive.value.item() == pytest.approx(1.0, abs=0.05)


def test_mean_dimension_known_value() -> None:
    # Independent positions with unequal base frequencies, and a fixed
    # central CpG (positions 11 and 12 here).
    generator = torch.Generator().manual_seed(6)
    frequencies = torch.rand(LENGTH, 4, generator=generator, dtype=torch.double)
    frequencies = frequencies / frequencies.sum(1, keepdim=True)
    frequencies[11] = torch.tensor([0.0, 0, 0, 1], dtype=torch.double)
    frequencies[12] = torch.tensor([0.0, 0, 1, 0], dtype=torch.double)
    values = torch.randn(6, 4, generator=generator, dtype=torch.double)
    a, b, c = 3, 9, 20
    # Each g centred under its position's frequencies.
    centred = [
        values[k] - (values[k] * frequencies[p]).sum()
        for k, p in enumerate((a, a, b, a, b, c))
    ]

    def g(k: int, x: torch.Tensor, p: int) -> torch.Tensor:
        return x[:, p] @ centred[k]

    # A term of the fixed C: constant under the frequencies, so it changes
    # neither the variance nor the true value.
    fixed = torch.randn(4, generator=generator, dtype=torch.double)

    def f(x: torch.Tensor) -> torch.Tensor:
        return (
            x[:, 11] @ fixed
            + g(0, x, a)
            + g(1, x, a) * g(2, x, b)
            + g(3, x, a) * g(4, x, b) * g(5, x, c)
        )

    def variance(k: int, p: int) -> torch.Tensor:
        return (frequencies[p] * centred[k] ** 2).sum()

    v1 = variance(0, a)
    v2 = variance(1, a) * variance(2, b)
    v3 = variance(3, a) * variance(4, b) * variance(5, c)
    true = ((v1 + 2 * v2 + 3 * v3) / (v1 + v2 + v3)).item()
    estimate = interactions.mean_dimension(
        f, frequencies, 400, generator, variance_sequence_quantity=50000
    )
    assert estimate.value.item() == pytest.approx(true, abs=0.08)

    # Weighting each position's variance uniformly instead, as the plan
    # warned, counts bases that never occur at the fixed CpG and overstates
    # the order.
    sequences = interactions.draw_independent_sequences(
        frequencies, 400, generator
    )
    identity = torch.eye(4, dtype=torch.double)
    totals = []
    for sequence in sequences:
        variants = sequence.expand(LENGTH, 4, LENGTH, 4).clone()
        variants[torch.arange(LENGTH), :, torch.arange(LENGTH)] = identity
        outputs = f(variants.flatten(0, 1)).view(LENGTH, 4)
        totals.append(outputs.var(1, unbiased=False).sum())
    total_variance = f(
        interactions.draw_independent_sequences(frequencies, 50000, generator)
    ).var()
    uniform = (torch.stack(totals).mean() / total_variance).item()
    assert uniform > true + 0.2
