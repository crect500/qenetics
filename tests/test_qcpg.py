from pathlib import Path
from math import ceil, log2
from tempfile import TemporaryDirectory
from unittest import mock

import pennylane as qml
import pytest
from torch import nn, optim, tensor

from qenetics.qcpg import qcpg, qcpg_models
from qenetics.tools import cpg_sampler

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


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
        _ = qcpg._train_one_epoch(
            model,
            1,
            cpg_sampler.H5CpGDataset([]),
            optim.SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            report_every=1,
        )


def test_train_qnn_circuit(test_dataset_directory: Path) -> None:
    with TemporaryDirectory() as temp_dir:
        qcpg.train_qnn_circuit(
            training_parameters=qcpg.TrainingParameters(
                data_directory=test_dataset_directory,
                output_filepath=Path(temp_dir) / "output.dat",
                training_chromosomes=["1", "2"],
                validation_chromosomes=["1", "2"],
                batch_size=1,
                epochs=2,
            )
        )
