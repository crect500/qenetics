from math import pi

import pennylane as qp
import pytest
import torch

from qenetics.circuits.ansatz import LayeredAnsatz
from qenetics.qcpg import sequence_model
from qenetics.qcpg.pure_models import (
    FeatureAmplitudeClassifier,
    FeatureReuploadingClassifier,
    QuantumAutomaton,
    WindowAverageClassifier,
)


def _one_hot(batch_size: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    symbols = torch.randint(0, 4, (batch_size, length), generator=generator)
    return torch.nn.functional.one_hot(symbols, 4).double()


def _randomize(model: torch.nn.Module) -> torch.nn.Module:
    torch.manual_seed(1)
    with torch.no_grad():
        model.circuit_weights.normal_(0, 1)
    return model


def test_window_average_matches_window_circuits() -> None:
    model = WindowAverageClassifier(
        10,
        3,
        window_length=2,
        quantum_layer_quantity=2,
        readout_wires=[0, 1, 2],
    ).double()
    with torch.no_grad():
        model.bank.circuit_weights.normal_(0, 1)
    inputs = _one_hot(2, 10)
    weights = model.bank.circuit_weights[0].detach()

    @qp.qnode(qp.device("default.qubit", wires=3), interface="torch")
    def circuit(amplitudes: torch.Tensor) -> list:
        qp.StatePrep(amplitudes, wires=range(3), normalize=True)
        LayeredAnsatz(weights, wires=range(3))
        return [qp.expval(qp.PauliZ(wire)) for wire in range(3)]

    windows = inputs.unfold(1, 2, 1).transpose(2, 3).reshape(2, 9, 8)
    expectations = torch.stack(circuit(windows.reshape(-1, 8)), -1).reshape(
        2, 9, 3
    )
    expected = ((1 - expectations) / 2).mean(1)
    assert torch.allclose(model(inputs), expected, atol=1e-10)


def test_window_average_circuits_and_centre() -> None:
    model = WindowAverageClassifier(
        20, 2, window_length=4, circuit_quantity=3, centre_length=8
    ).double()
    inputs = _one_hot(3, 20)
    # Only the central 8 positions, i.e. 5 windows, are read.
    shifted = inputs.clone()
    shifted[:, :6] = torch.roll(shifted[:, :6], 1, 0)
    assert torch.allclose(model(inputs), model(shifted))
    probabilities = model(inputs)
    assert probabilities.shape == (3, 2)
    assert torch.equal(probabilities[:, 0], probabilities[:, 1])


def test_window_average_errors() -> None:
    with pytest.raises(ValueError, match="centre length 4"):
        WindowAverageClassifier(20, 2, window_length=8, centre_length=4)
    with pytest.raises(ValueError, match="do not fit"):
        WindowAverageClassifier(6, 2, window_length=8)


def test_window_average_shots() -> None:
    model = WindowAverageClassifier(12, 1, window_length=4).eval()
    model.shots = 5
    assert model.bank.shots == 5
    assert model(_one_hot(2, 12).float()).shape == (2, 1)


def test_feature_amplitude_matches_qnode() -> None:
    model = _randomize(
        FeatureAmplitudeClassifier(
            40, 2, quantum_layer_quantity=3, readout_wires=[0, 5]
        ).double()
    ).eval()
    inputs = _one_hot(4, 40)
    channels, _ = model.encoder(inputs)
    statistics = sequence_model.sequence_statistics(channels)
    amplitudes = torch.nn.functional.pad(
        torch.cat([torch.ones(4, 1, dtype=torch.float64), statistics], 1),
        (0, 64 - 45),
    )
    weights = model.circuit_weights.detach()

    @qp.qnode(qp.device("default.qubit", wires=6), interface="torch")
    def circuit(state: torch.Tensor) -> list:
        qp.AmplitudeEmbedding(state, wires=range(6), normalize=True)
        qp.StronglyEntanglingLayers(weights, wires=range(6))
        return [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(5))]

    # Untrained batch normalization in evaluation leaves the statistics.
    expected = (1 - torch.stack(circuit(amplitudes), -1)) / 2
    assert torch.allclose(model(inputs), expected, atol=1e-10)


def test_feature_reuploading_matches_qnode() -> None:
    model = _randomize(
        FeatureReuploadingClassifier(40, 3, wire_quantity=8).double()
    ).eval()
    assert model.upload_quantity == 6
    inputs = _one_hot(2, 40)
    channels, _ = model.encoder(inputs)
    statistics = torch.nn.functional.pad(
        sequence_model.sequence_statistics(channels), (0, 4)
    )
    angles = (pi / 2) * torch.tanh(statistics).reshape(2, 6, 8)
    weights = model.circuit_weights.detach()

    @qp.qnode(qp.device("default.qubit", wires=8), interface="torch")
    def circuit(angles: torch.Tensor) -> qp.measurements.ExpectationMP:
        for upload in range(6):
            for wire in range(8):
                qp.RY(angles[:, upload, wire], wires=wire)
            qp.StronglyEntanglingLayers(weights[upload], wires=range(8))
        return qp.expval(qp.PauliZ(0))

    expected = ((1 - circuit(angles)) / 2).unsqueeze(1).expand(-1, 3)
    assert torch.allclose(model(inputs), expected, atol=1e-10)


@pytest.mark.parametrize(
    "model_class", [FeatureAmplitudeClassifier, FeatureReuploadingClassifier]
)
def test_feature_models_need_bases(model_class: type) -> None:
    with pytest.raises(ValueError, match="needs one-hot nucleotides"):
        model_class(20, 2, encoding="token", vocabulary_size=8)


def test_feature_model_on_lightning() -> None:
    model = FeatureAmplitudeClassifier(
        30, 2, device_name="lightning.qubit", diff_method="adjoint"
    )
    probabilities = model(_one_hot(3, 30).float())
    probabilities.sum().backward()
    assert model.circuit_weights.grad is not None
    assert probabilities.dtype == torch.float32


def test_automaton_matches_sequential_circuit() -> None:
    model = _randomize(
        QuantumAutomaton(
            11, 2, width=5, wire_quantity=3, readout_wires="per_output"
        ).double()
    )
    inputs = _one_hot(2, 11)
    inputs[1, 4] = 0.0  # an unknown nucleotide applies the identity
    unitaries = [
        qp.matrix(
            qp.StronglyEntanglingLayers(
                model.circuit_weights[symbol].detach(), wires=range(3)
            )
        )
        for symbol in range(4)
    ]

    @qp.qnode(qp.device("default.qubit", wires=3), interface="torch")
    def circuit(sample: torch.Tensor) -> list:
        for position in range(3, 8):
            if sample[position].sum() > 0:
                qp.QubitUnitary(
                    unitaries[int(sample[position].argmax())], wires=range(3)
                )
        return [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(1))]

    expected = [(1 - torch.stack(circuit(sample))) / 2 for sample in inputs]
    assert torch.allclose(model(inputs), torch.stack(expected), atol=1e-10)


def test_automaton_errors() -> None:
    with pytest.raises(ValueError, match="width 30 exceeds"):
        QuantumAutomaton(20, 2, width=30)
    with pytest.raises(ValueError, match="more than"):
        QuantumAutomaton(20, 2, width=10, wire_quantity=9)
    model = QuantumAutomaton(
        6, 1, width=4, encoding="token", token_lookup=torch.ones(2, 2)
    )
    with pytest.raises(ValueError, match="one symbol per position"):
        model(torch.zeros(6))


def test_automaton_on_tokens() -> None:
    model = QuantumAutomaton(8, 2, width=6, encoding="token", vocabulary_size=5)
    assert model.circuit_weights.shape[0] == 5
    probabilities = model(torch.randint(0, 5, (3, 8)))
    assert probabilities.shape == (3, 2)
    assert ((probabilities > 0) & (probabilities < 1)).all()


@pytest.mark.parametrize(
    "model",
    [
        WindowAverageClassifier(30, 2, window_length=4),
        FeatureAmplitudeClassifier(30, 2, quantum_layer_quantity=2),
        FeatureReuploadingClassifier(30, 2, wire_quantity=4),
        QuantumAutomaton(30, 2, width=10),
    ],
)
def test_pure_models_train(model: torch.nn.Module) -> None:
    inputs = _one_hot(4, 30).float()
    probabilities = model(inputs)
    assert probabilities.shape == (4, 2)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    probabilities.sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    # No trained classical parameters.
    assert [name for name, _ in model.named_parameters()] in (
        ["circuit_weights"],
        ["bank.circuit_weights"],
    )
    # Single sequences; batch normalization needs evaluation mode for them.
    assert model.eval()(inputs[0]).shape == (2,)


@pytest.mark.parametrize(
    "model",
    [
        FeatureAmplitudeClassifier(30, 2),
        FeatureReuploadingClassifier(30, 2, wire_quantity=4),
        QuantumAutomaton(30, 2, width=10),
    ],
)
def test_pure_model_shots(model: torch.nn.Module) -> None:
    model.eval()
    model.shots = 4
    probabilities = model(_one_hot(3, 30).float())
    clamped = probabilities.clamp(0.01, 0.99)
    # Frequencies of 4 shots, kept off 0 and 1.
    assert torch.allclose(
        clamped[(clamped > 0.01) & (clamped < 0.99)] * 4,
        (clamped[(clamped > 0.01) & (clamped < 0.99)] * 4).round(),
        atol=1e-4,
    )
