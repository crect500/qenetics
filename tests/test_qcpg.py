from math import ceil, log2
from unittest import mock

from numpy.typing import NDArray
import pennylane as qml
from pennylane import numpy as pnp
import pytest
from torch import nn, optim, tensor

from qenetics.qcpg import qcpg, qcpg_models
from qenetics.tools import cpg_sampler

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


@pytest.mark.parametrize(("nucleotide_character"), ["A", "T", "C", "G", "B"])
def test_convert_nucleotide_to_enum(nucleotide_character: str) -> None:
    if nucleotide_character == "A":
        assert (
            qcpg.convert_nucleotide_to_enum(nucleotide_character)
            == qcpg.Nucleodtide.A
        )
    elif nucleotide_character == "T":
        assert (
            qcpg.convert_nucleotide_to_enum(nucleotide_character)
            == qcpg.Nucleodtide.T
        )
    elif nucleotide_character == "C":
        assert (
            qcpg.convert_nucleotide_to_enum(nucleotide_character)
            == qcpg.Nucleodtide.C
        )
    elif nucleotide_character == "G":
        assert (
            qcpg.convert_nucleotide_to_enum(nucleotide_character)
            == qcpg.Nucleodtide.G
        )
    else:
        with pytest.raises(
            ValueError, match="not recognized as valid nucleotide"
        ):
            _ = qcpg.convert_nucleotide_to_enum(nucleotide_character)


@pytest.mark.parametrize(
    ("nucleotide_value", "index"),
    [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (0, 2), (0, 2), (0, 8)],
)
def test_ampltiude_encode_nucleotide(nucleotide_value: int, index: int) -> None:
    if index == 0:
        address_register_size: int = 1
    else:
        address_register_size = ceil(log2(index)) + 1
    nucleotide = qcpg.Nucleodtide(nucleotide_value)
    circuit_width: int = address_register_size + int(
        log2(UNIQUE_NUCLEOTIDE_QUANTITY)
    )
    address_range: int = 2**address_register_size
    passive_nucleotides: list[int] = list(range(UNIQUE_NUCLEOTIDE_QUANTITY))
    passive_nucleotides.remove(nucleotide_value)
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def run_circuit() -> qml.measurements.ProbabilityMP:
        for qubit_index in range(address_register_size):
            qml.Hadamard(qubit_index)
        qcpg.amplitude_encode_nucleotide(
            nucleotide, index, address_register_size
        )
        return qml.probs(wires=list(range(circuit_width)))

    target_index: int = index * 4 + nucleotide.value
    results: qml.measurements.ProbabilityMP = run_circuit()
    assert results.sum() == pytest.approx(1.0)
    assert results[target_index] == pytest.approx(1 / address_range)


@pytest.mark.parametrize(
    ("sequence_values"), [[0], [1], [0, 0], [0, 1], [0, 1, 2]]
)
def test_amplitude_encode_all_nucleotides(sequence_values: list[int]) -> None:
    sequence: list[qcpg.Nucleodtide] = [
        qcpg.Nucleodtide(value) for value in sequence_values
    ]
    address_register_size: int = qcpg.calculate_address_register_size(
        len(sequence_values)
    )
    circuit_width: int = address_register_size + int(
        log2(UNIQUE_NUCLEOTIDE_QUANTITY)
    )

    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def run_circuit() -> qml.measurements.ProbabilityMP:
        for qubit_index in range(address_register_size):
            qml.Hadamard(qubit_index)
        qcpg.amplitude_encode_all_nucleotides(sequence)
        return qml.probs(wires=list(range(circuit_width)))

    results: qml.measurements.ProbabilityMP = run_circuit()
    assert results.sum() == pytest.approx(1.0)
    for index, nucleotide in enumerate(sequence_values):
        target_index: int = index * 4 + nucleotide
        assert results[target_index] == pytest.approx(
            1 / 2**address_register_size
        )


def test_train_qcpg_circuit(test_sequences: list[str]) -> None:
    iterations: int = 5
    targets: NDArray[int] = pnp.ones(len(test_sequences), dtype=int)
    samples: list[list[qcpg.Nucleodtide]] = [
        qcpg.string_to_nucleotides(sequence) for sequence in test_sequences
    ]
    address_register_size: int = qcpg.calculate_address_register_size(
        len(samples)
    )
    layer_quantity: int = 2
    params_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=layer_quantity,
        n_wires=address_register_size + UNIQUE_NUCLEOTIDE_QUANTITY,
    )
    parameters: NDArray = pnp.random.default_rng().random(size=params_shape)
    trained_parameters, loss_history = (
        qcpg.train_strongly_entangled_qcpg_circuit(
            parameters, samples, targets, iterations
        )
    )
    assert parameters.shape == trained_parameters.shape
    assert len(loss_history) == iterations


@pytest.mark.parametrize(
    ("sequence", "layer_quantity", "output_quantity"),
    [
        ("A", 1, 1),
        ("AT", 1, 1),
        ("ATC", 1, 1),
        ("C", 2, 1),
        ("G", 1, 2),
        ("ATCGATCG", 2, 2),
    ],
)
def test_train_one_epoch(
    sequence: str, layer_quantity: int, output_quantity: int
) -> None:
    batch_size: int = 2
    model = qcpg_models.QNN(len(sequence), layer_quantity, output_quantity)
    with (
        mock.patch("qenetics.tools.cpg_sampler.H5CpGDataset"),
        mock.patch(
            "qenetics.tools.cpg_sampler.H5CpGDataset.__getitem__"
        ) as mock_get,
    ):
        mock_get.return_value = [
            tensor(cpg_sampler.nucleotide_string_to_numpy(sequence))
        ] * batch_size
        loss: float = qcpg._train_one_epoch(
            model,
            1,
            cpg_sampler.H5CpGDataset([]),
            optim.SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            report_every=1,
        )
