import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from qenetics.qcpg import model_factory

SCRIPT: Path = Path("scripts/cluster/interaction_diagnostics.py")
LENGTH: int = 65


@pytest.fixture(scope="module")
def script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "interaction_diagnostics", SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules["interaction_diagnostics"] = module
    specification.loader.exec_module(module)
    return module


def _sequences(quantity: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    bases = torch.randint(0, 4, (quantity, LENGTH), generator=generator)
    sequences = torch.nn.functional.one_hot(bases, 4).double()
    # The central CpG: channels are A, T, G, C.
    sequences[:, LENGTH // 2] = torch.tensor([0.0, 0, 0, 1])
    sequences[:, LENGTH // 2 + 1] = torch.tensor([0.0, 0, 1, 0])
    return sequences


def _model(family: str, **options: object) -> torch.nn.Module:
    torch.manual_seed(1)
    model = model_factory.build_model(family, LENGTH, 3, options=options)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.5)
    return model.double().eval()


def _measure(script: ModuleType, model, output: str, **options: object):
    sequences = _sequences(40)
    return script.measure(
        script.model_function(model, output, window=options.pop("window", 0)),
        output,
        sequences[:6],
        script.base_frequencies(sequences),
        torch.Generator().manual_seed(2),
        triple_quantity=20,
        dimension_sequence_quantity=options.pop("dimension", 8),
        batch_size=256,
        **options,
    )


def test_base_frequencies(script: ModuleType) -> None:
    frequencies = script.base_frequencies(_sequences(200))
    assert frequencies.shape == (LENGTH, 4)
    assert torch.allclose(frequencies.sum(1), torch.ones(LENGTH).double())
    assert frequencies[LENGTH // 2].tolist() == [0, 0, 0, 1]
    assert frequencies[LENGTH // 2 + 1].tolist() == [0, 0, 1, 0]
    assert script.centre_positions(1001) == (500, 501)
    assert script.token_window(1001) == (129, 502)


def test_order_2_models(script: ModuleType) -> None:
    # A factorization machine and the complete pairwise model have order 2
    # in their logits: no three-way epistasis, and a mean dimension of at
    # most 2.
    for family, options in (
        ("factorization_machine", {"rank": 3}),
        ("pairwise_model", {}),
    ):
        rows = _measure(script, _model(family, **options), script.LOGIT)
        assert [(r["component"], r["stratum"]) for r in rows] == [
            (component, stratum)
            for stratum in ("window", "near", "anywhere")
            for component in ("first_cell", "mean_over_cells")
        ]
        for row in rows:
            assert row["epistasis_3_maximum"] < 1e-9
            assert row["epistasis_2_median"] > 1e-4
            assert 1.0 <= row["mean_dimension"] < 2.2
            assert row["dimension_sequences"] == 8
            assert (row["sequences"], row["triples"]) == (6, 20)
        # The sigmoid alone creates higher orders.
        rows = _measure(
            script, _model(family, **options), script.PROBABILITY, dimension=0
        )
        assert all(row["epistasis_3_median"] > 1e-9 for row in rows)
        assert all(row["dimension_sequences"] == 0 for row in rows)


def test_filter_bank_channels(script: ModuleType) -> None:
    model = _model(
        "filter_bank_network", window_lengths=[16], quantum_layer_quantity=2
    )
    # One window's probability is a quadratic form of the window's one-hot
    # amplitudes: order 2, and so is its sum over windows.
    window = _measure(
        script, model, script.CHANNEL_WINDOW, window=8, span=(8, 16)
    )
    assert [(r["component"], r["stratum"]) for r in window] == [
        ("channel", "window")
    ]
    assert window[0]["epistasis_3_maximum"] < 1e-12
    assert window[0]["epistasis_2_median"] > 1e-6
    assert window[0]["dimension_sequences"] == 0
    total = _measure(script, model, script.CHANNEL_SUM)
    assert all(row["epistasis_3_maximum"] < 1e-10 for row in total)
    assert all(row["mean_dimension"] < 2.2 for row in total)
    # The affine map, ReLU and tail lift the order of the whole model.
    logit = _measure(script, model, script.LOGIT, dimension=0)
    assert max(row["epistasis_3_maximum"] for row in logit) > 1e-6
    with pytest.raises(ValueError, match="window's span"):
        _measure(script, model, script.CHANNEL_WINDOW)


def test_model_function_errors(script: ModuleType) -> None:
    model = _model("factorization_machine", rank=2)
    # The logits are the model's own, not an inverted sigmoid, and the
    # sigmoid is restored afterwards.
    sequences = _sequences(4)
    logits = script.model_function(model, script.LOGIT)(sequences)
    probabilities = script.model_function(model, script.PROBABILITY)(sequences)
    assert torch.allclose(torch.sigmoid(logits[:, 0]), probabilities[:, 0])
    assert logits.abs().max() > 1
    pure = _model("window_average", window_length=4, quantum_layer_quantity=1)
    with pytest.raises(ValueError, match="does not end with one sigmoid"):
        script.model_function(pure, script.LOGIT)(sequences)
    assert torch.sigmoid(torch.zeros(1)).item() == 0.5
    with pytest.raises(ValueError, match="Unknown output"):
        script.model_function(model, "angle")
    with pytest.raises(ValueError, match="Only bases and k-mer"):
        script.model_function(model, script.LOGIT, input_name="cpg_bpe64")


def test_contracted_global_circuit(script: ModuleType) -> None:
    # A short sequence: 4 address qubits and 2 data qubits.
    length = 13
    torch.manual_seed(3)
    model = model_factory.build_model(
        "global_circuit",
        length,
        3,
        options={"ansatz": "basic_ry", "quantum_layer_quantity": 2},
    )
    with torch.no_grad():
        model.circuit_weights.normal_(0.0, 1.0)
    model = model.double().eval()
    assert script.is_contractible(model)
    matrix = script.circuit_matrix(model)
    assert matrix.shape == (64, 64)
    # The circuit is unitary.
    assert torch.allclose(
        matrix @ matrix.conj().T, torch.eye(64, dtype=matrix.dtype), atol=1e-10
    )
    generator = torch.Generator().manual_seed(4)
    bases = torch.randint(0, 4, (5, length), generator=generator)
    sequences = torch.nn.functional.one_hot(bases, 4).double()
    channels, _ = model.encoder(sequences)
    with torch.no_grad():
        simulated = script.model_logits(model, channels)
        contracted = script.model_outputs(
            model, channels, logits=True, matrix=matrix
        )
    assert torch.allclose(contracted, simulated, atol=1e-8)
    # The contracted logits are what the diagnostics measure.
    function = script.model_function(model, script.LOGIT)
    values = function(sequences)
    assert torch.allclose(values[:, 0], simulated[:, 0], atol=1e-8)
    assert not script.is_contractible(_model("pairwise_model"))
