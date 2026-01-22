import numpy as np
import pennylane as qml
import pytest
import torch
from torch import tensor

from qenetics.qcpg import qcpg_models
from qenetics.tools import data, dna


@pytest.mark.parametrize(
    ("input", "qubit_quantity"), [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]
)
def test_calculate_address_register_size(
    input: int, qubit_quantity: int
) -> None:
    assert qcpg_models.calculate_address_register_size(input) == qubit_quantity


@pytest.mark.parametrize(
    ("nucleotide_value", "index"),
    [
        ("A", 0),
        ("C", 0),
        ("T", 0),
        ("G", 0),
        ("A", 1),
        ("C", 1),
        ("A", 2),
        ("A", 2),
        ("A", 8),
    ],
)
def test_basis_encode_nucleotide(nucleotide_value: str, index: int) -> None:
    address_register_size = qcpg_models.calculate_address_register_size(
        index + 1
    )
    nucleotide = np.expand_dims(
        data.nucleotide_character_to_numpy(nucleotide_value), 0
    )
    circuit_width: int = address_register_size + data.UNIQUE_NUCLEOTIDE_QUANTITY
    address_range: int = 2**address_register_size
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def run_circuit() -> qml.measurements.ProbabilityMP:
        for qubit_index in range(address_register_size):
            qml.Hadamard(qubit_index)
        qcpg_models.basis_encode_nucleotide(
            nucleotide, index, address_register_size
        )
        return qml.probs(wires=list(range(circuit_width)))

    target_index_list: list[int] = [0] * 4
    target_index_list[
        dna.convert_nucleotide_to_enum(nucleotide_value).value
    ] = 1
    target_index_string: str = bin(index).split("b")[1] + "".join(
        str(i) for i in target_index_list
    )
    target_index: int = int(target_index_string, 2)
    results: qml.measurements.ProbabilityMP = run_circuit()
    assert results.sum() == pytest.approx(1.0)
    assert results[target_index] == pytest.approx(1 / address_range)


@pytest.mark.parametrize("sequence", ["A", "C", "AA", "AC", "ACTG"])
def test_basis_encode_all_nucleotides(sequence: str) -> None:
    sequence_tensor = tensor(data.nucleotide_string_to_numpy(sequence))
    address_register_size: int = qcpg_models.calculate_address_register_size(
        sequence_tensor.shape[0]
    )
    circuit_width: int = address_register_size + data.UNIQUE_NUCLEOTIDE_QUANTITY

    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def run_circuit() -> qml.measurements.ProbabilityMP:
        for qubit_index in range(address_register_size):
            qml.Hadamard(qubit_index)
        qcpg_models.single_encode_all_nucleotides(sequence_tensor)
        return qml.probs(wires=list(range(circuit_width)))

    results: qml.measurements.ProbabilityMP = run_circuit()
    assert results.sum() == pytest.approx(1.0)
    for index, nucleotide in enumerate(sequence):
        target_index_list: list[int] = [0] * 4
        target_index_list[dna.convert_nucleotide_to_enum(nucleotide).value] = 1
        target_index_string: str = bin(index).split("b")[1] + "".join(
            str(i) for i in target_index_list
        )
        target_index: int = int(target_index_string, 2)
        assert results[target_index] == pytest.approx(
            1 / 2**address_register_size
        )


@pytest.mark.parametrize(
    ("nucleotide_characters", "index"),
    [
        (["A"], 0),
        (["T"], 0),
        (["C"], 0),
        (["G"], 0),
        (["A"], 1),
        (["T"], 1),
        (["C"], 2),
        (["A"], 2),
        (["A"], 6),
        (["C"], 6),
        (["A", "T"], 2),
        (["C", "G"], 2),
    ],
)
def test_amplitiude_encode_nucleotide(
    nucleotide_characters: list[str], index: int
) -> None:
    nucleotide = np.array(
        [
            data.nucleotide_character_to_numpy(nucleotide_character)
            for nucleotide_character in nucleotide_characters
        ]
    )
    address_register_size = qcpg_models.calculate_address_register_size(
        index + 1
    )
    circuit_width: int = (
        address_register_size + qcpg_models.AMPLITUDE_QUBIT_QUANTITY
    )
    address_range: int = 2**address_register_size
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def run_circuit() -> qml.measurements.ProbabilityMP:
        for qubit_index in range(address_register_size):
            qml.Hadamard(qubit_index)
        qcpg_models.amplitude_encode_nucleotide(
            nucleotide, index, address_register_size
        )
        return qml.probs(wires=list(range(circuit_width)))

    expanded_circuit = qml.transforms.broadcast_expand(run_circuit)
    results: qml.measurements.ProbabilityMP = expanded_circuit()
    for result_index, nucleotide_character in enumerate(nucleotide_characters):
        target_index: int = (
            index * 4
            + dna.convert_nucleotide_to_enum(nucleotide_character).value
        )
        assert results[result_index].sum() == pytest.approx(1.0)
        assert results[result_index][target_index] == pytest.approx(
            1 / address_range
        )


@pytest.mark.parametrize("sequences", [["A"], ["C"], ["AA"], ["AC"], ["ACTG"]])
def test_amplitude_encode_all_nucleotides(sequences: list[str]) -> None:
    sequence_tensors = tensor(
        [data.nucleotide_string_to_numpy(sequence) for sequence in sequences]
    )
    address_register_size: int = qcpg_models.calculate_address_register_size(
        sequence_tensors.shape[1]
    )
    circuit_width: int = (
        address_register_size + qcpg_models.AMPLITUDE_QUBIT_QUANTITY
    )
    address_range: int = 2**address_register_size
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def run_circuit() -> qml.measurements.ProbabilityMP:
        for qubit_index in range(address_register_size):
            qml.Hadamard(qubit_index)
        qcpg_models.amplitude_encode_all_nucleotides(sequence_tensors)
        return qml.probs(wires=list(range(circuit_width)))

    expanded_circuit = qml.transforms.broadcast_expand(run_circuit)
    results: qml.measurements.ProbabilityMP = expanded_circuit()
    for result_index, sequence in enumerate(sequences):
        for nucleotide_index, nucleotide in enumerate(sequence):
            target_index: int = (
                nucleotide_index * 4
                + dna.convert_nucleotide_to_enum(nucleotide).value
            )
            assert results[result_index].sum() == pytest.approx(1.0)
            assert results[result_index][target_index] == pytest.approx(
                1 / address_range
            )


@pytest.mark.parametrize(
    ("layer_quantity", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3)],
)
def test_basis_basic_entangling_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    wire_quantity = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + data.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    quantum_layer: qml.qnn.torch.TorchLayer = (
        qcpg_models.single_basic_entangling_torch(
            sequence_length, layer_quantity
        )
    )
    assert quantum_layer.weights.shape == (layer_quantity, wire_quantity)


@pytest.mark.parametrize(
    ("layer_quantity", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3)],
)
def test_basis_strongly_entangled_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    rotations_quantity: int = 3
    wire_quantity = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + data.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    quantum_layer: qml.qnn.torch.TorchLayer = (
        qcpg_models.basis_strongly_entangled_torch(
            sequence_length, layer_quantity
        )
    )
    assert quantum_layer.weights.shape == (
        layer_quantity,
        rotations_quantity,
        wire_quantity,
    )


@pytest.mark.parametrize(
    ("layer_quantity", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3)],
)
def test_amplitude_basic_entangling_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    wire_quantity = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + qcpg_models.AMPLITUDE_QUBIT_QUANTITY
    )
    quantum_layer: qml.qnn.torch.TorchLayer = (
        qcpg_models.amplitude_basic_entangling_torch(
            sequence_length, layer_quantity
        )
    )
    assert quantum_layer.weights.shape == (layer_quantity, wire_quantity)


@pytest.mark.parametrize(
    ("layer_quantity", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3)],
)
def test_basis_strongly_entangled_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    wire_quantity = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + qcpg_models.AMPLITUDE_QUBIT_QUANTITY
    )
    quantum_layer: qml.qnn.torch.TorchLayer = (
        qcpg_models.amplitude_strongly_entangling_torch(
            sequence_length, layer_quantity
        )
    )
    assert quantum_layer.weights.shape == (
        layer_quantity,
        qcpg_models.UNIQUE_ROTATIONS_QUANTITY,
        wire_quantity,
    )


def test_QNN() -> None:
    sequence_length: int = 8
    output_quantity: int = 4
    model = qcpg_models.QNN(
        sequence_length=sequence_length,
        quantum_layer_quantity=2,
        output_quantity=output_quantity,
        entangler="basic",
    )
    model.train(True)
    single_input: torch.Tensor = torch.tensor(
        [data.nucleotide_string_to_numpy("ATCGATCG")], requires_grad=False
    )
    single_input[single_input < 0.5] = 0
    single_input[single_input > 0.0] = 1
    single_input = torch.tensor(single_input, dtype=torch.float)
    output: torch.Tensor = model(single_input)
    assert output.shape == (1, 4)

    sequences: list[str] = ["ATCGATCG", "AATTCCGG", "GGCCTTAA", "GCTAGCTA"]
    multiple_input: torch.Tensor = torch.tensor(
        [data.nucleotide_string_to_numpy(sequence) for sequence in sequences],
        requires_grad=False,
    )
    multiple_input[multiple_input < 0.5] = 0
    multiple_input[multiple_input > 0.0] = 1
    multiple_input = torch.tensor(multiple_input, dtype=torch.float)
    output: torch.Tensor = model(multiple_input)
    assert output.shape == (4, 4)
