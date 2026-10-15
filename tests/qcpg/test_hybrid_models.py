from math import pi

import pennylane as qp
import pytest
import torch

from qenetics.qcpg import qcpg_models
from qenetics.qcpg.hybrid_models import (
    CNNQuantumHead,
    FilterBankNetwork,
    GlobalCircuitModel,
)


def _one_hot(batch_size: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    symbols = torch.randint(0, 4, (batch_size, length), generator=generator)
    return torch.nn.functional.one_hot(symbols, 4).float()


def _gradients_reach_every_parameter(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize(
    ("options", "channel_quantity"),
    [
        ({}, 64),
        ({"window_lengths": [4, 8], "tail": "pooling"}, 16 + 32),
        (
            {
                "circuit_quantity": 2,
                "readout": "expectation",
                "activation": "none",
            },
            12,
        ),
        ({"tail": "pooling", "pooling": "mean", "channel_mix": True}, 64),
        ({"tail": "deepcpg", "first_pool": "average"}, 64),
        ({"activation": "none", "tail": "quantum_filter_bank"}, 64),
    ],
)
def test_filter_bank_network(options: dict, channel_quantity: int) -> None:
    model = FilterBankNetwork(101, 3, **options)
    assert model.front_end.output_channel_quantity == channel_quantity
    probabilities = model(_one_hot(4, 101))
    assert probabilities.shape == (4, 3)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    probabilities.sum().backward()
    assert _gradients_reach_every_parameter(model)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"tail": "invalid"}, "Unknown tail"),
        ({"activation": "invalid"}, "Unknown activation"),
        ({"tail": "pooling", "first_pool": "max"}, "apply to the deepcpg tail"),
        ({"channel_mix": True}, "apply to the pooling tail"),
        ({"tail_window_length": 2}, "apply to the quantum_filter_bank tail"),
        ({"window_lengths": [64]}, "no positions|do not fit"),
    ],
)
def test_filter_bank_network_errors(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FilterBankNetwork(60, 2, **options)


def test_filter_bank_network_tokens_match_one_hot() -> None:
    torch.manual_seed(0)
    one_hot = FilterBankNetwork(
        40, 2, window_lengths=[2], tail="pooling", channel_quantity=8
    ).eval()
    tokens = FilterBankNetwork(
        40,
        2,
        window_lengths=[2],
        tail="pooling",
        encoding="token",
        vocabulary_size=8,
    ).eval()
    tokens.load_state_dict(one_hot.state_dict())
    ids = torch.randint(0, 8, (3, 40))
    assert torch.allclose(
        tokens(ids),
        one_hot(torch.nn.functional.one_hot(ids, 8).float()),
    )


def test_global_circuit_matches_repository_qnn() -> None:
    torch.manual_seed(1)
    sequence_length, output_quantity = 6, 3
    qnn = qcpg_models.QNN(
        sequence_length,
        2,
        output_quantity,
        entangling="basic",
        measurement="probability",
        device_name="default.qubit",
        diff_method="backprop",
    )
    model = GlobalCircuitModel(
        sequence_length,
        output_quantity,
        quantum_layer_quantity=2,
        feature_scale="repo",
        initialization="uniform",
    )
    with torch.no_grad():
        model.circuit_weights.copy_(qnn.qnn.weights)
        model.head[0].weight.copy_(qnn.linear.weight)
        model.head[0].bias.copy_(qnn.linear.bias)
    inputs = _one_hot(4, sequence_length)
    assert torch.allclose(model(inputs), qnn(inputs), atol=1e-5)


def test_global_circuit_pure_head_matches_qnode() -> None:
    model = GlobalCircuitModel(
        5,
        2,
        ansatz="strong",
        quantum_layer_quantity=2,
        head="pure",
        readout_wires=[0, 4],
    ).double()
    torch.manual_seed(2)
    with torch.no_grad():
        model.circuit_weights.normal_(0, 1)
    # 5 positions in 3 address qubits, the last 3 addresses holding |00>.
    assert model.wire_quantity == 5
    inputs = _one_hot(3, 5).double()
    padding = torch.zeros(3, 3, 4, dtype=torch.float64)
    padding[..., 0] = 1.0
    amplitudes = torch.cat([inputs, padding], 1).reshape(3, 32) / 8**0.5
    weights = model.circuit_weights.detach()

    @qp.qnode(qp.device("default.qubit", wires=5), interface="torch")
    def circuit(state: torch.Tensor) -> list:
        qp.AmplitudeEmbedding(state, wires=range(5))
        qp.StronglyEntanglingLayers(weights, wires=range(5))
        return [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(4))]

    expected = (1 - torch.stack(circuit(amplitudes), -1)) / 2
    assert torch.allclose(model(inputs), expected, atol=1e-10)


def test_global_circuit_block_and_shifts() -> None:
    model = GlobalCircuitModel(
        16,
        2,
        ansatz="strong",
        quantum_layer_quantity=2,
        block_qubit_quantity=4,
        shifts=[0, 3],
        hidden_layer_sizes=[5],
    )
    assert model.circuit_wires == [2, 3, 4, 5]
    assert model.head[0].in_features == 2 * 2**6
    probabilities = model(_one_hot(3, 16))
    probabilities.sum().backward()
    assert probabilities.shape == (3, 2)
    assert _gradients_reach_every_parameter(model)


def test_global_circuit_block_acts_on_every_block() -> None:
    """A circuit on the data and lowest address wires acts on each block."""
    model = GlobalCircuitModel(
        8, 1, ansatz="strong", block_qubit_quantity=3, head="pure"
    ).double()
    torch.manual_seed(3)
    with torch.no_grad():
        model.circuit_weights.normal_(0, 1)
    inputs = _one_hot(1, 8).double()
    # Swapping the two halves of the sequence swaps its two blocks of 2^1
    # addresses, which the pure head's wire 0 (the top address) tells apart,
    # but P(wire 0 = 1) is the probability mass of the second half.
    swapped = torch.cat([inputs[:, 4:], inputs[:, :4]], 1)
    assert torch.allclose(
        model(inputs) + model(swapped), torch.ones(1, 1, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"head": "invalid"}, "Unknown head"),
        ({"feature_scale": "invalid"}, "Unknown feature scale"),
        ({"block_qubit_quantity": 9}, "between 1 and the register's 7"),
        ({"shifts": [40]}, "Shifts must be between 0 and 19"),
        ({"head": "pure", "shifts": [0, 2]}, "unshifted"),
        (
            {"device_name": "lightning.qubit", "diff_method": "adjoint"},
            "cannot differentiate probability",
        ),
    ],
)
def test_global_circuit_errors(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GlobalCircuitModel(20, 2, **options)


def test_global_circuit_tokens() -> None:
    model = GlobalCircuitModel(6, 2, encoding="token", vocabulary_size=8)
    # 6 positions in 3 address qubits and 8 tokens in 3 data qubits.
    assert model.wire_quantity == 6
    assert model(torch.randint(0, 8, (2, 6))).shape == (2, 2)


def test_global_circuit_unknown_nucleotides_are_normalized() -> None:
    model = GlobalCircuitModel(4, 1, head="pure")
    inputs = _one_hot(1, 4)
    inputs[0, 1] = 0.0
    probabilities = model(inputs)
    assert torch.isfinite(probabilities).all()


def test_cnn_quantum_head_matches_qnode() -> None:
    torch.manual_seed(4)
    model = (
        CNNQuantumHead(
            60,
            3,
            wire_quantity=4,
            quantum_layer_quantity=2,
            upload_quantity=2,
            hidden_layer_sizes=[6],
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        model.circuit_weights.normal_(0, 1)
    inputs = _one_hot(2, 60).double()
    channels, _ = model.encoder(inputs)
    angles = pi * torch.tanh(model.angle_layers(model.body(channels))).reshape(
        2, 2, 4
    )
    weights = model.circuit_weights.detach()

    @qp.qnode(qp.device("default.qubit", wires=4), interface="torch")
    def circuit(angles: torch.Tensor) -> list:
        for upload in range(2):
            for wire in range(4):
                qp.RY(angles[:, upload, wire], wires=wire)
            qp.StronglyEntanglingLayers(weights[upload], wires=range(4))
        return [qp.expval(qp.PauliZ(wire)) for wire in range(3)]

    expected = (1 - torch.stack(circuit(angles.detach()), -1)) / 2
    assert torch.allclose(model(inputs), expected, atol=1e-10)


@pytest.mark.parametrize("separate_angles", [True, False])
def test_cnn_quantum_head_trains(separate_angles: bool) -> None:
    model = CNNQuantumHead(
        60,
        10,
        wire_quantity=4,
        separate_angles=separate_angles,
        ansatz="real",
        quantum_layer_quantity=3,
    )
    probabilities = model(_one_hot(3, 60))
    assert probabilities.shape == (3, 10)
    probabilities.sum().backward()
    assert _gradients_reach_every_parameter(model)
    # Outputs 0 and 4 read the same wire, 0 mod 4.
    assert torch.allclose(probabilities[:, 0], probabilities[:, 4])


def test_cnn_quantum_head_errors() -> None:
    with pytest.raises(ValueError, match="no positions"):
        CNNQuantumHead(20, 2)
    with pytest.raises(ValueError, match="upload quantity"):
        CNNQuantumHead(60, 2, upload_quantity=0)


@pytest.mark.parametrize(
    "model",
    [
        GlobalCircuitModel(10, 2),
        GlobalCircuitModel(10, 2, head="pure"),
        CNNQuantumHead(60, 2, wire_quantity=3),
    ],
)
def test_hybrid_model_shots(model: torch.nn.Module) -> None:
    length: int = model.sequence_length
    inputs = _one_hot(3, length)
    model.eval()
    exact = model(inputs)
    model.shots = 2
    sampled = model(inputs)
    assert sampled.shape == exact.shape
    assert not torch.equal(sampled, exact)
