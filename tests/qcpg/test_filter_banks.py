from math import ceil, log2

import pennylane as qp
import pytest
import torch

from qenetics.circuits import ansatz
from qenetics.circuits.ansatz import LayeredAnsatz
from qenetics.qcpg import heads
from qenetics.qcpg.filter_banks import (
    WINDOW_NORMALIZATION,
    MultiScaleFilterBank,
    QuantumFilterBank,
)


def _random_one_hot(
    batch_size: int, channel_quantity: int, length: int
) -> torch.Tensor:
    symbols = torch.randint(0, channel_quantity, (batch_size, length))
    return (
        torch.nn.functional.one_hot(symbols, channel_quantity)
        .transpose(1, 2)
        .double()
    )


def _window_probabilities(
    bank: QuantumFilterBank, channels: torch.Tensor, circuit: int = 0
) -> torch.Tensor:
    """
    Run every window through a QNode: its amplitudes (channel-minor, padded
    with the bank's constant), the circuit, and all probabilities.
    """
    wire_quantity: int = bank.wire_quantity
    device = qp.device("default.qubit", wires=wire_quantity)
    weights = bank.circuit_weights[circuit].detach()

    @qp.qnode(device, interface="torch")
    def circuit_probabilities(amplitudes: torch.Tensor) -> torch.Tensor:
        qp.StatePrep(amplitudes, wires=range(wire_quantity), normalize=True)
        LayeredAnsatz.compute_decomposition(
            weights, range(wire_quantity), bank.ansatz
        )
        return qp.probs(wires=range(wire_quantity))

    windows = channels.unfold(2, bank.window_length, 1)  # (B, A, P, w)
    amplitudes = windows.permute(0, 2, 3, 1).reshape(
        channels.shape[0], windows.shape[2], -1
    )
    padding = torch.full(
        (*amplitudes.shape[:2], 2**wire_quantity - amplitudes.shape[2]),
        bank.constant,
        dtype=amplitudes.dtype,
    )
    amplitudes = torch.cat([amplitudes, padding], 2)
    return circuit_probabilities(
        amplitudes.reshape(-1, 2**wire_quantity)
    ).reshape(*amplitudes.shape[:2], -1)


@pytest.mark.parametrize(
    "ansatz_name", [ansatz.STRONG_ANSATZ_STR, ansatz.REAL_ANSATZ_STR]
)
@pytest.mark.parametrize(
    ("channel_quantity", "window_length"), [(4, 1), (4, 3), (4, 4), (5, 2)]
)
def test_filter_bank_matches_window_circuits(
    ansatz_name: str, channel_quantity: int, window_length: int
) -> None:
    torch.manual_seed(0)
    bank = QuantumFilterBank(
        channel_quantity,
        window_length,
        quantum_layer_quantity=2,
        ansatz=ansatz_name,
        initialization_scale=1.0,
    ).double()
    channels = _random_one_hot(2, channel_quantity, 9)
    features = bank(channels)
    outcome_quantity: int = 2**bank.wire_quantity
    assert bank.wire_quantity == ceil(log2(channel_quantity * window_length))
    assert features.shape == (2, outcome_quantity, 9 - window_length + 1)
    expected = _window_probabilities(bank, channels).transpose(1, 2)
    assert torch.allclose(features / outcome_quantity, expected, atol=1e-10)


def test_filter_bank_unknown_nucleotides_are_unnormalized() -> None:
    bank = QuantumFilterBank(4, 4).double()
    channels = _random_one_hot(1, 4, 8)
    channels[:, :, 2] = 0.0
    features = bank(channels) / 2**bank.wire_quantity
    totals = features.sum(1)[0]
    # Windows holding position 2 have 3 of 4 nucleotides.
    assert torch.allclose(
        totals, torch.tensor([0.75, 0.75, 0.75, 1, 1]).double()
    )


def test_filter_bank_circuits_and_readouts() -> None:
    torch.manual_seed(1)
    channels = _random_one_hot(2, 4, 12)
    probabilities_bank = QuantumFilterBank(
        4, 4, circuit_quantity=3, initialization_scale=1.0
    ).double()
    probabilities = probabilities_bank(channels).reshape(2, 3, 16, -1) / 16
    for circuit in range(3):
        expected = _window_probabilities(probabilities_bank, channels, circuit)
        assert torch.allclose(
            probabilities[:, circuit], expected.transpose(1, 2), atol=1e-10
        )

    expectation_bank = QuantumFilterBank(
        4, 4, circuit_quantity=3, readout="expectation"
    ).double()
    expectation_bank.load_state_dict(probabilities_bank.state_dict())
    expectations = expectation_bank(channels).reshape(2, 3, 4, -1)
    bits = torch.tensor(
        [
            [(state >> (3 - wire)) & 1 for wire in range(4)]
            for state in range(16)
        ]
    ).double()
    assert torch.allclose(
        expectations,
        torch.einsum("bcjp,jk->bckp", probabilities, 1 - 2 * bits),
        atol=1e-10,
    )

    marginal_bank = QuantumFilterBank(
        4,
        4,
        circuit_quantity=3,
        readout="marginal",
        marginal_wire_quantity=2,
    ).double()
    marginal_bank.load_state_dict(probabilities_bank.state_dict())
    marginals = marginal_bank(channels).reshape(2, 3, 4, -1) / 4
    assert torch.allclose(
        marginals, probabilities.reshape(2, 3, 4, 4, -1).sum(3), atol=1e-10
    )
    assert marginal_bank.output_channel_quantity == 12


def test_constant_amplitude_bank() -> None:
    torch.manual_seed(2)
    bank = QuantumFilterBank(
        4, 3, constant_amplitude=1.0, initialization_scale=1.0
    ).double()
    # 12 window amplitudes and 4 constants in 4 qubits, the block's norm
    # that of a one-hot window.
    assert bank.wire_quantity == 4
    assert bank.constant**2 * 4 == pytest.approx(3)
    channels = _random_one_hot(2, 4, 7)
    expected = _window_probabilities(bank, channels).transpose(1, 2)
    assert torch.allclose(bank(channels) / 16, expected, atol=1e-10)


def test_window_normalization() -> None:
    torch.manual_seed(3)
    bank = QuantumFilterBank(
        3, 2, normalization=WINDOW_NORMALIZATION, initialization_scale=1.0
    ).double()
    channels = torch.rand(2, 3, 6, dtype=torch.float64)
    expected = _window_probabilities(bank, channels).transpose(1, 2)
    assert torch.allclose(bank(channels) / 8, expected, atol=1e-10)


def test_token_embedding() -> None:
    torch.manual_seed(4)
    vocabulary_size: int = 16
    bank = QuantumFilterBank(
        vocabulary_size, 1, initialization_scale=1.0
    ).double()
    tokens = torch.randint(0, vocabulary_size, (3, 5))
    channels = (
        torch.nn.functional.one_hot(tokens, vocabulary_size)
        .transpose(1, 2)
        .double()
    )
    unitary = ansatz.ansatz_unitary(bank.circuit_weights[0], bank.ansatz)
    expected = (unitary.abs() ** 2)[:, tokens].permute(
        1, 0, 2
    ) * vocabulary_size
    assert torch.allclose(bank(channels), expected, atol=1e-10)


def test_frozen_bank() -> None:
    bank = QuantumFilterBank(4, 4, trainable_circuit=False)
    features = bank(_random_one_hot(2, 4, 8).float())
    assert not bank.circuit_weights.requires_grad
    assert not features.requires_grad


def test_chunked_bank_matches_unchunked() -> None:
    torch.manual_seed(5)
    bank = QuantumFilterBank(4, 32, quantum_layer_quantity=2)
    channels = _random_one_hot(70, 4, 40).float()
    chunked = bank(channels)
    whole = torch.cat([bank(channels[:35]), bank(channels[35:])])
    assert torch.allclose(chunked, whole, atol=1e-6)
    chunked.sum().backward()
    assert bank.circuit_weights.grad is not None


def test_bank_shots_apply_in_evaluation() -> None:
    bank = QuantumFilterBank(4, 4).eval()
    channels = _random_one_hot(1, 4, 8).float()
    exact = bank(channels)
    bank.shots = 10
    sampled = bank(channels)
    # Frequencies of 10 shots, times 16 outcomes.
    assert torch.allclose(
        sampled * 10 / 16, (sampled * 10 / 16).round(), atol=1e-4
    )
    bank.train()
    assert torch.allclose(bank(channels), exact)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"ansatz": "invalid"}, "Unknown ansatz"),
        ({"readout": "invalid"}, "Unknown readout"),
        ({"normalization": "invalid"}, "Unknown normalization"),
        ({"constant_amplitude": -1.0}, "must not be negative"),
        (
            {"constant_amplitude": 1.0, "normalization": WINDOW_NORMALIZATION},
            "fixed normalization",
        ),
        ({"circuit_quantity": 0}, "circuit quantity"),
        ({"quantum_layer_quantity": 0}, "quantum layer quantity"),
        ({"readout": "marginal", "marginal_wire_quantity": 7}, "marginal of 7"),
    ],
)
def test_filter_bank_errors(arguments: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        QuantumFilterBank(4, 16, **arguments)


def test_filter_bank_size_limit() -> None:
    with pytest.raises(ValueError, match="13 qubits"):
        QuantumFilterBank(4096, 2)


def test_filter_bank_input_errors() -> None:
    bank = QuantumFilterBank(4, 8)
    with pytest.raises(ValueError, match=r"shape \(batch, 4, length\)"):
        bank(torch.zeros(1, 5, 10))
    with pytest.raises(ValueError, match="do not fit"):
        bank(torch.zeros(1, 4, 7))


def test_multi_scale_bank_aligns_centres() -> None:
    torch.manual_seed(6)
    banks = MultiScaleFilterBank(4, [2, 4], quantum_layer_quantity=1)
    channels = _random_one_hot(2, 4, 10).float()
    features = banks(channels)
    assert banks.output_channel_quantity == 8 + 16
    assert banks.position_quantity(10) == 7
    assert features.shape == (2, 24, 7)
    # The 2 bp bank's 9 offsets are cropped by one at each end.
    assert torch.allclose(features[:, :8], banks.banks[0](channels)[:, :, 1:8])
    assert torch.allclose(features[:, 8:], banks.banks[1](channels))
    with pytest.raises(ValueError, match="At least one window length"):
        MultiScaleFilterBank(4, [])


def test_multilayer_perceptron() -> None:
    layers = heads.multilayer_perceptron(10, [8, 4], 2, dropout=0.2)
    assert [type(layer).__name__ for layer in layers] == [
        "Linear",
        "ReLU",
        "Dropout",
        "Linear",
        "ReLU",
        "Dropout",
        "Linear",
    ]
    assert len(heads.multilayer_perceptron(10, [], 2)) == 1
    with pytest.raises(ValueError, match="Dropout"):
        heads.multilayer_perceptron(10, [], 2, dropout=1.0)
    with pytest.raises(ValueError, match="layer size"):
        heads.multilayer_perceptron(10, [0], 2)


@pytest.mark.parametrize("first_pool", [heads.MAX_POOL, heads.AVERAGE_POOL])
def test_deepcpg_tail(first_pool: str) -> None:
    tail = heads.DeepCpGTail(64, 986, 6, first_pool=first_pool)
    assert tail(torch.rand(2, 64, 986)).shape == (2, 6)
    # DeepCpG's tail on a 16 bp front end of 1001 bp: 4,048,006 parameters
    # with 6 outputs.
    assert sum(parameter.numel() for parameter in tail.parameters()) == 4048006


def test_deepcpg_tail_errors() -> None:
    with pytest.raises(ValueError, match="no positions"):
        heads.DeepCpGTail(8, 15, 2)
    with pytest.raises(ValueError, match="Unknown first pool"):
        heads.DeepCpGTail(8, 100, 2, first_pool="invalid")


def test_deepcpg_body() -> None:
    body, flattened = heads.deepcpg_body(4, 1001)
    assert flattened == 256 * 122
    assert body(torch.rand(2, 4, 1001)).shape == (2, flattened)
    with pytest.raises(ValueError, match="no positions"):
        heads.deepcpg_body(4, 20)


@pytest.mark.parametrize(
    ("pooling", "channel_mix", "parameter_quantity"),
    [
        ("max_mean", False, 2 * 5 * 3 + 3),
        ("mean", False, 5 * 3 + 3),
        ("max_mean", True, 30 + 33),
    ],
)
def test_pooling_head(
    pooling: str, channel_mix: bool, parameter_quantity: int
) -> None:
    head = heads.PoolingHead(5, 3, pooling=pooling, channel_mix=channel_mix)
    assert head(torch.rand(2, 5, 9)).shape == (2, 3)
    assert (
        sum(parameter.numel() for parameter in head.parameters())
        == parameter_quantity
    )
    with pytest.raises(ValueError, match="Unknown pooling"):
        heads.PoolingHead(5, 3, pooling="invalid")


def test_quantum_filter_bank_tail() -> None:
    tail = heads.QuantumFilterBankTail(64, 986, 6)
    assert tail.bank.wire_quantity == 8
    assert tail(torch.rand(2, 64, 986) * 64).shape == (2, 6)
    with pytest.raises(ValueError, match="do not fit|no positions"):
        heads.QuantumFilterBankTail(8, 10, 2)


def test_channel_affine_relu() -> None:
    activation = heads.ChannelAffineReLU(3)
    with torch.no_grad():
        activation.affine.weight.copy_(
            torch.tensor([1.0, -1.0, 2.0]).reshape(3, 1, 1)
        )
        activation.affine.bias.zero_()
    channels = torch.ones(1, 3, 2)
    assert activation(channels)[0, :, 0].tolist() == [1.0, 0.0, 2.0]
