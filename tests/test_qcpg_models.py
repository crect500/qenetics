import pennylane as qml
import pytest
from torch import tensor

from qenetics.qcpg import qcpg_models
from qenetics.tools import cpg_sampler


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
def test_single_encode_nucleotide(nucleotide_value: str, index: int) -> None:
    address_register_size = qcpg_models.calculate_address_register_size(
        index + 1
    )
    nucleotide = cpg_sampler.nucleotide_character_to_numpy(nucleotide_value)
    circuit_width: int = (
        address_register_size + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    address_range: int = 2**address_register_size
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def run_circuit() -> qml.measurements.ProbabilityMP:
        for qubit_index in range(address_register_size):
            qml.Hadamard(qubit_index)
        qcpg_models.single_encode_nucleotide(
            nucleotide, index, address_register_size
        )
        return qml.probs(wires=list(range(circuit_width)))

    target_index_list: list[int] = [0] * 4
    target_index_list[
        cpg_sampler.convert_nucleotide_to_enum(nucleotide_value).value
    ] = 1
    target_index_string: str = bin(index).split("b")[1] + "".join(
        str(i) for i in target_index_list
    )
    target_index: int = int(target_index_string, 2)
    results: qml.measurements.ProbabilityMP = run_circuit()
    assert results.sum() == pytest.approx(1.0)
    assert results[target_index] == pytest.approx(1 / address_range)


@pytest.mark.parametrize("sequence", ["A", "C", "AA", "AC", "ACTG"])
def test_singe_encode_all_nucleotides(sequence: str) -> None:
    sequence_tensor = tensor(cpg_sampler.nucleotide_string_to_numpy(sequence))
    address_register_size: int = qcpg_models.calculate_address_register_size(
        sequence_tensor.shape[0]
    )
    circuit_width: int = (
        address_register_size + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY
    )

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
        target_index_list[
            cpg_sampler.convert_nucleotide_to_enum(nucleotide).value
        ] = 1
        target_index_string: str = bin(index).split("b")[1] + "".join(
            str(i) for i in target_index_list
        )
        target_index: int = int(target_index_string, 2)
        assert results[target_index] == pytest.approx(
            1 / 2**address_register_size
        )


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


# @pytest.mark.parametrize(
#     ("sequence_values"), [[0], [1], [0, 0], [0, 1], [0, 1, 2]]
# )
# def test_amplitude_encode_all_nucleotides(sequence_values: list[int]) -> None:
#     sequence: list[qcpg.Nucleodtide] = [
#         qcpg.Nucleodtide(value) for value in sequence_values
#     ]
#     address_register_size: int = qcpg.calculate_address_register_size(
#         len(sequence_values)
#     )
#     circuit_width: int = address_register_size + int(
#         log2(UNIQUE_NUCLEOTIDE_QUANTITY)
#     )
#
#     device = qml.device("default.qubit", wires=circuit_width)
#
#     @qml.qnode(device)
#     def run_circuit() -> qml.measurements.ProbabilityMP:
#         for qubit_index in range(address_register_size):
#             qml.Hadamard(qubit_index)
#         qcpg.amplitude_encode_all_nucleotides(sequence)
#         return qml.probs(wires=list(range(circuit_width)))
#
#     results: qml.measurements.ProbabilityMP = run_circuit()
#     assert results.sum() == pytest.approx(1.0)
#     for index, nucleotide in enumerate(sequence_values):
#         target_index: int = index * 4 + nucleotide
#         assert results[target_index] == pytest.approx(
#             1 / 2**address_register_size
#         )


@pytest.mark.parametrize(
    ("layer_quantity", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3)],
)
def test_single_basic_entangling_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    wire_quantity = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY
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
def test_strongly_entangled_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    rotations_quantity: int = 3
    wire_quantity = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    quantum_layer: qml.qnn.torch.TorchLayer = (
        qcpg_models.strongly_entangled_torch(sequence_length, layer_quantity)
    )
    assert quantum_layer.weights.shape == (
        layer_quantity,
        rotations_quantity,
        wire_quantity,
    )
