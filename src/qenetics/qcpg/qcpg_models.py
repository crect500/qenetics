from __future__ import annotations

from math import ceil, log2

import pennylane as qml
from torch import nn, Tensor
from torch.nn import functional

from qenetics.tools import cpg_sampler


class QNN(nn.Module):
    def __init__(
        self: QNN,
        sequence_length: int,
        quantum_layer_quantity: int,
        output_quantity: int,
        entangler: str = "basic",
    ) -> None:
        super(QNN, self).__init__()

        if entangler == "basic":
            self.qnn = single_basic_entangling_torch(
                sequence_length, quantum_layer_quantity
            )
        elif entangler == "strong":
            self.qnn = strongly_entangled_torch(
                sequence_length, quantum_layer_quantity
            )
        else:
            raise ValueError(
                f"Entangler specified not recognized: {entangler}."
            )

        wire_quantity: int = (
            calculate_address_register_size(sequence_length)
            + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY
        )
        self.linear = nn.Linear(2**wire_quantity, output_quantity)

    def forward(self: QNN, x: Tensor) -> Tensor:
        # Must change has_batch_dim in pennylane\qnn\torch.py:forward to True for multi-dimensional inputs
        x = self.qnn(x)
        x = self.linear(x)
        output: Tensor = functional.sigmoid(x)

        return output


def calculate_address_register_size(encode_quantity: int) -> int:
    """
    Determine the address register size for the quantum circuit.

    Args
    ----
    encode_quantity: The number of values that the quantum circuit will encode.

    Returns
    -------
    The minimum quantum address register qubits required to encode all values.

    Raises
    ------
    ValueError if values_to_encode is less than 1.
    """
    if encode_quantity < 1:
        raise ValueError(f"Value {encode_quantity} must be greater than 0.")

    if encode_quantity == 1:
        return 1

    return int(ceil(log2(encode_quantity)))


def single_encode_nucleotide(
    nucleotide: Tensor, index: int, address_register_size: int
) -> None:
    """
    One-hot encode nucleotide value into a quantum circuit at the appropriate address.

    Args
    ----
    nucleotide: A one-hot encoded nucleotide representation.
    index: The location of the nucleotide in the sequence.
    address_register_size: The size of the quantum address register.
    """
    controls: list[int] = [int(value) for value in bin(index).split("b")[1]]
    if len(controls) != address_register_size:
        controls = [0] * (address_register_size - len(controls)) + controls
    qml.ctrl(
        qml.BasisEmbedding,
        list(range(address_register_size)),
        control_values=controls,
    )(
        nucleotide,
        list(
            range(
                address_register_size,
                address_register_size + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY,
            )
        ),
    )


def single_encode_all_nucleotides(
    nucleotides: Tensor,
) -> None:
    """
    One-hot encode nucleotide values into a quantum circuit at appropriate addresses.

    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    address_register_size: int = calculate_address_register_size(
        len(nucleotides)
    )
    for index, nucleotide in enumerate(nucleotides):
        single_encode_nucleotide(nucleotide, index, address_register_size)


def amplitude_encode_nucleotide(
    nucleotide: Tensor, index: int, address_register_size: int
) -> None:
    """
    Amplitude-encode nucleotide value into a quantum circuit at the appropriate index.

    Args
    ----
    nucleotide: An enumerated value corresponding to a type of nucleotide.
    index: The location of the nucleotide in the sequence.
    address_register_size: The size of the quantum address register.
    """
    controls: list[int] = [int(value) for value in bin(index).split("b")[1]]
    if len(controls) != address_register_size:
        controls = [0] * (address_register_size - len(controls)) + controls
    qml.ctrl(
        qml.AmplitudeEmbedding,
        list(range(address_register_size)),
        control_values=controls,
    )(
        nucleotide,
        list(
            range(
                address_register_size,
                address_register_size + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY,
            )
        ),
    )


def amplitude_encode_all_nucleotides(
    nucleotides: Tensor,
) -> None:
    """
    Amplitude encode nucleotide values into a quantum circuit at appropriate addresses.

    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    address_register_size: int = calculate_address_register_size(
        len(nucleotides)
    )
    for index, nucleotide in enumerate(nucleotides):
        amplitude_encode_nucleotide(nucleotide, index, address_register_size)


def strongly_entangled_jax(parameters: Tensor) -> None:
    layer_quantity: int = parameters.shape[0]
    wire_quantity: int = parameters.shape[1]
    rotations_quantity: int = parameters.shape[2]
    for layer_index in range(layer_quantity):
        qml.StronglyEntanglingLayers(
            parameters[layer_index, :, :].reshape(
                (1, wire_quantity, rotations_quantity)
            ),
            range(wire_quantity),
        )


def single_basic_entangling_torch(
    sequence_length: int, quantum_layer_quantity: int
) -> qml.qnn.TorchLayer:
    wire_quantity: int = (
        calculate_address_register_size(sequence_length)
        + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    device = qml.device("default.qubit", wires=wire_quantity)

    @qml.qnode(device)
    def _qnode(inputs: Tensor, weights: Tensor):
        single_encode_all_nucleotides(inputs)
        qml.BasicEntanglerLayers(weights, wires=list(range(wire_quantity)))

        return qml.probs(wires=list(range(wire_quantity)))

    return qml.qnn.TorchLayer(
        _qnode, {"weights": (quantum_layer_quantity, wire_quantity)}
    )


def strongly_entangled_torch(
    sequence_length: int, quantum_layer_quantity: int
) -> qml.qnn.TorchLayer:
    wire_quantity: int = (
        calculate_address_register_size(sequence_length)
        + cpg_sampler.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    device = qml.device("default.qubit", wires=wire_quantity)

    @qml.qnode(device)
    def _qnode(inputs: Tensor, weights: Tensor):
        single_encode_all_nucleotides(inputs)
        qml.StronglyEntanglingLayers(
            weights,
            wires=list(range(wire_quantity)),
        )

        return qml.probs(wires=list(range(wire_quantity)))

    return qml.qnn.TorchLayer(
        _qnode,
        {
            "weights": qml.StronglyEntanglingLayers.shape(
                n_layers=quantum_layer_quantity, n_wires=wire_quantity
            )
        },
    )
