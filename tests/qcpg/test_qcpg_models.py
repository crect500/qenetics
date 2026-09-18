import numpy as np
import pennylane as qp
import pytest
import torch
from torch import tensor

from qenetics.qcpg import qcpg_models
from qenetics.tools import converters, dna


@pytest.mark.parametrize(
    ("input", "qubit_quantity"), [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]
)
def test_calculate_address_register_size(
    input: int, qubit_quantity: int
) -> None:
    assert qcpg_models.calculate_address_register_size(input) == qubit_quantity


@pytest.mark.parametrize(
    ("device_name", "distribute"), [("default.qubit", False)]
)
def test_device_setup(device_name: str, distribute: bool) -> None:
    wire_quantity: int = 2
    device: qp.devices.Device = qcpg_models._device_setup(
        device_name, wire_quantity, distribute=distribute
    )
    if device_name == "default.qubit":
        assert isinstance(device, qp.devices.DefaultQubit)
    elif device_name == "lightning.gpu":
        assert isinstance(device, qp.devices.LightningGPU)


@pytest.mark.parametrize("sequence", ["A", "C", "AA", "AC", "ACTG"])
def test_encode_all_nucleotides(sequence: list[str]) -> None:
    sequence_tensors = tensor(converters.nucleotide_string_to_numpy(sequence))
    sequence_length: int = sequence_tensors.shape[0]
    address_register_size: int = qcpg_models.calculate_address_register_size(
        sequence_length
    )
    circuit_width: int = (
        address_register_size + qcpg_models.AMPLITUDE_QUBIT_QUANTITY
    )
    address_range: int = 2**address_register_size
    device = qp.device("default.qubit", wires=circuit_width)

    @qp.qnode(device)
    def run_circuit() -> qp.measurements.ProbabilityMP:
        qcpg_models._encode_all_nucleotides(
            sequence_tensors, sequence_length, address_register_size
        )
        return qp.probs(wires=list(range(circuit_width)))

    expanded_circuit = qp.transforms.broadcast_expand(run_circuit)
    results: qp.measurements.ProbabilityMP = expanded_circuit()
    for nucleotide_index, nucleotide in enumerate(sequence):
        target_index: int = (
            nucleotide_index * 4
            + dna.convert_nucleotide_to_enum(nucleotide).value
        )
        assert results.sum() == pytest.approx(1.0)
        assert results[target_index] == pytest.approx(1 / address_range)


@pytest.mark.parametrize(
    ("layer_quantity", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3)],
)
def test_basic_entangling_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    test_input = torch.tensor(
        np.array(
            [[converters.nucleotide_character_to_numpy("A")] * sequence_length]
            * 2,
            dtype=float,
        )
    )
    wire_quantity: int = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + qcpg_models.AMPLITUDE_QUBIT_QUANTITY
    )
    quantum_layer: qp.qnn.torch.TorchLayer = qcpg_models._torch_qnn_layer(
        sequence_length, layer_quantity, entangling="basic"
    )
    assert quantum_layer.weights.shape == (layer_quantity, wire_quantity)
    assert len(quantum_layer(test_input[0])) == 2**wire_quantity

    quantum_layer: qp.qnn.torch.TorchLayer = qcpg_models._torch_qnn_layer(
        sequence_length,
        layer_quantity,
        entangling="basic",
        measurement="expectation",
    )
    assert len(quantum_layer(test_input[0])) == wire_quantity


@pytest.mark.parametrize(
    ("layer_quantity", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3)],
)
def test_strongly_entangled_torch(
    layer_quantity: int, sequence_length: int
) -> None:
    test_input = torch.tensor(
        np.array(
            [[converters.nucleotide_character_to_numpy("A")] * sequence_length]
            * 2,
            dtype=float,
        )
    )
    wire_quantity: int = (
        qcpg_models.calculate_address_register_size(sequence_length)
        + qcpg_models.AMPLITUDE_QUBIT_QUANTITY
    )
    quantum_layer: qp.qnn.torch.TorchLayer = qcpg_models._torch_qnn_layer(
        sequence_length, layer_quantity, entangling="strong"
    )
    assert quantum_layer.weights.shape == (
        layer_quantity,
        wire_quantity,
        qcpg_models.UNIQUE_ROTATIONS_QUANTITY,
    )
    assert len(quantum_layer(test_input[0])) == 2**wire_quantity

    quantum_layer: qp.qnn.torch.TorchLayer = qcpg_models._torch_qnn_layer(
        sequence_length,
        layer_quantity,
        entangling="basic",
        measurement="expectation",
    )
    assert len(quantum_layer(test_input[0])) == wire_quantity


@pytest.mark.parametrize(
    ("entangling", "measurement", "encoding", "fcl_quantity"),
    [
        ("basic", "probability", "onehot", 1),
        ("basic", "expectation", "onehot", 1),
        ("strong", "probability", "onehot", 1),
        ("strong", "expectation", "onehot", 1),
        ("basic", "expectation", "token", 1),
        ("strong", "probability", "token", 1),
        ("invalid", "probability", "onehot", 1),
        ("strong", "invalid", "onehot", 1),
    ],
)
def test_QNN(
    entangling: str,
    measurement: str,
    encoding: bool,
    fcl_quantity: int | None,
) -> None:
    sequence_length: int = 8
    output_quantity: int = 3
    if encoding == "token":
        embedding_qubit_quantity: int = 1
        vocabulary_size: int = 4
    else:
        embedding_qubit_quantity = None
        vocabulary_size = None

    if entangling == "invalid":
        with pytest.raises(
            ValueError, match="Unknown entangling layer type invalid"
        ):
            _ = qcpg_models.QNN(
                sequence_length=sequence_length,
                quantum_layer_quantity=2,
                output_quantity=output_quantity,
                encoding=encoding,
                embedding_qubit_quantity=embedding_qubit_quantity,
                vocabulary_size=vocabulary_size,
                fcl_quantity=fcl_quantity,
                entangling=entangling,
                measurement=measurement,
            )
    elif measurement == "invalid":
        with pytest.raises(
            ValueError,
            match="Unknown measurement type invalid",
        ):
            _ = qcpg_models.QNN(
                sequence_length=sequence_length,
                quantum_layer_quantity=2,
                output_quantity=output_quantity,
                encoding=encoding,
                embedding_qubit_quantity=embedding_qubit_quantity,
                vocabulary_size=vocabulary_size,
                fcl_quantity=fcl_quantity,
                entangling=entangling,
                measurement=measurement,
            )
    else:
        model = qcpg_models.QNN(
            sequence_length=sequence_length,
            quantum_layer_quantity=2,
            output_quantity=output_quantity,
            encoding=encoding,
            embedding_qubit_quantity=embedding_qubit_quantity,
            vocabulary_size=vocabulary_size,
            fcl_quantity=fcl_quantity,
            entangling=entangling,
            measurement=measurement,
        )
        model.train(True)
        if encoding == "onehot":
            single_input: torch.Tensor = torch.tensor(
                converters.nucleotide_string_to_numpy("ATCGATCG"),
                requires_grad=False,
                dtype=torch.float,
            )
        else:
            single_input = torch.tensor(
                [0, 1, 2, 3, 3, 2, 1, 0], requires_grad=False, dtype=torch.int
            )

        output: torch.Tensor = model(single_input)
        assert output.shape == torch.Size([output_quantity])

        if encoding == "onehot":
            sequences: list[str] = ["ATCGATCG", "AATTCCGG", "GGCCTTAA"]
            multiple_input: torch.Tensor = torch.tensor(
                np.array(
                    [
                        converters.nucleotide_string_to_numpy(sequence)
                        for sequence in sequences
                    ],
                    dtype=float,
                ),
                requires_grad=False,
                dtype=torch.float,
            )
        else:
            multiple_input = torch.tensor(
                [
                    [0, 1, 2, 3, 3, 2, 1, 0],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    [3, 3, 2, 2, 1, 1, 0, 0],
                ],
                requires_grad=False,
                dtype=torch.int,
            )
        output: torch.Tensor = model(multiple_input)
        assert output.shape == torch.Size(
            [multiple_input.shape[0], output_quantity]
        )


@pytest.mark.parametrize(
    ("input_indices", "hidden_indices", "layer_quantity", "entangling"),
    [
        ([0], [1], 1, "basic"),
        ([0], [1], 1, "strong"),
        ([0], [1], 2, "basic"),
        ([0], [1], 2, "strong"),
        ([0], [1, 2], 2, "basic"),
        ([0, 1], [2], 2, "strong"),
        ([0, 1], [2, 3], 2, "basic"),
    ],
)
def test_qrnn_block(
    input_indices: list[int],
    hidden_indices: list[int],
    layer_quantity: int,
    entangling: str,
) -> None:
    input_qubit_quantity: int = len(input_indices)
    hidden_qubit_quantity: int = len(hidden_indices)
    wire_quantity: int = input_qubit_quantity + hidden_qubit_quantity
    device: qp.devices.Device = qp.device("default.qubit", wires=wire_quantity)

    if entangling == "basic":
        weights: torch.Tensor = torch.zeros(
            qp.BasicEntanglerLayers.shape(layer_quantity, hidden_qubit_quantity)
        )
    else:
        weights = torch.zeros(
            qp.StronglyEntanglingLayers.shape(
                layer_quantity, hidden_qubit_quantity
            )
        )

    @qp.qnode(device)
    def _test_qrnn() -> qp.measurements.ProbabilityMP:
        qcpg_models._qrnn_block(
            weights, input_indices, hidden_indices, entangling=entangling
        )
        return [
            qp.expval(qp.PauliZ(index))
            for index in range(input_qubit_quantity, wire_quantity)
        ]

    assert len(_test_qrnn()) == hidden_qubit_quantity


def test_torch_rqnn_layer() -> None:
    embedding_qubit_quantity: int = 2
    hidden_qubit_quantity: int = 2
    embeddings: torch.Tensor = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0] for _ in range(3)]
    )
    rqnn_layer: qp.qnn.TorchLayer = qcpg_models._torch_rqnn_layer(
        embedding_qubit_quantity=embedding_qubit_quantity,
        hidden_qubit_quantity=hidden_qubit_quantity,
        hidden_layer_quantity=2,
        measurement="probability",
    )
    assert len(rqnn_layer(embeddings)) == 2**hidden_qubit_quantity

    rqnn_layer: qp.qnn.TorchLayer = qcpg_models._torch_rqnn_layer(
        embedding_qubit_quantity=embedding_qubit_quantity,
        hidden_qubit_quantity=hidden_qubit_quantity,
        hidden_layer_quantity=2,
    )
    assert rqnn_layer(torch.stack([embeddings for _ in range(2)])).shape == (
        2,
        hidden_qubit_quantity,
    )


def test_RQNN() -> None:
    hidden_qubit_quantity: int = 2
    vocabulary_size: int = 3
    embedding_size: int = 4
    model = qcpg_models.RQNN(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        hidden_qubit_quantity=hidden_qubit_quantity,
        output_quantity=1,
        hidden_layer_quantity=2,
    )
    assert len(model(torch.tensor([0, 1, 2], dtype=torch.int64))) == 1

    model = qcpg_models.RQNN(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        hidden_qubit_quantity=hidden_qubit_quantity,
        output_quantity=2,
        hidden_layer_quantity=1,
    )
    assert model(
        torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64)
    ).shape == (2, 2)
