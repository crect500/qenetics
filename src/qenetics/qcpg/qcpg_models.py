from __future__ import annotations

from math import ceil, log2

import pennylane as qml
from torch import nn, Tensor
from torch.nn import functional

from qenetics.tools import data

AMPLITUDE_QUBIT_QUANTITY: int = 2
UNIQUE_ROTATIONS_QUANTITY: int = 3
SEQUENCE_LENGTH: int = -1


class QNN(nn.Module):
    def __init__(
        self: QNN,
        sequence_length: int,
        quantum_layer_quantity: int,
        output_quantity: int,
        entangler: str = "strong",
        encoding: str = "amplitude",
        device_name: str = "lightning.qubit",
        distribute: bool = True,
    ) -> None:
        super(QNN, self).__init__()
        global SEQUENCE_LENGTH
        SEQUENCE_LENGTH = sequence_length

        self.set_quantum_layer(
            encoding,
            entangler,
            device_name,
            sequence_length,
            quantum_layer_quantity,
            distribute,
        )

        wire_quantity: int = (
            calculate_address_register_size(sequence_length)
            + AMPLITUDE_QUBIT_QUANTITY
        )
        self.linear = nn.Linear(2**wire_quantity, output_quantity)

    def forward(self: QNN, x: Tensor) -> Tensor:
        # Must change has_batch_dim in pennylane\qnn\torch.py:forward to True for multi-dimensional inputs
        x = self.qnn(x)
        x = self.linear(x)
        output: Tensor = functional.sigmoid(x)

        return output

    def set_quantum_layer(
        self: "QNN",
        encoding: str,
        entangler: str,
        device_name: str,
        sequence_length: int,
        quantum_layer_quantity: int,
        distribute: bool,
    ) -> None:
        if entangler == "basic":
            if encoding == "amplitude":
                self.qnn = amplitude_basic_entangling_torch(
                    sequence_length,
                    quantum_layer_quantity,
                    device_name,
                    distribute,
                )
            else:
                raise ValueError(
                    f"Encoding method {encoding} not recognized or supported."
                )
        elif entangler == "strong":
            if encoding == "amplitude":
                self.qnn = amplitude_strongly_entangling_torch(
                    sequence_length,
                    quantum_layer_quantity,
                    device_name,
                    distribute,
                )
            else:
                raise ValueError(
                    f"Encoding method {encoding} not recognized or supported."
                )
        else:
            raise ValueError(
                f"Entangler {entangler} not recognized or supported."
            )


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


def basis_encode_nucleotide(
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
                address_register_size + data.UNIQUE_NUCLEOTIDE_QUANTITY,
            )
        ),
    )


def basis_encode_all_nucleotides(
    nucleotides: Tensor,
) -> None:
    """
    One-hot encode nucleotide values into a quantum circuit at appropriate addresses.

    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    sequence_length: int = nucleotides.shape[1]
    address_register_size: int = calculate_address_register_size(
        sequence_length
    )
    for index in range(sequence_length):
        basis_encode_nucleotide(
            nucleotides[:, index, :], index, address_register_size
        )


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
                address_register_size + AMPLITUDE_QUBIT_QUANTITY,
            )
        ),
    )


def amplitude_encode_all_nucleotides(
    nucleotides: Tensor,
) -> None:
    """
    One-hot encode nucleotide values into a quantum circuit at appropriate addresses.

    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    sequence_length: int = nucleotides.shape[1]
    address_register_size: int = calculate_address_register_size(
        sequence_length
    )
    for index in range(sequence_length):
        amplitude_encode_nucleotide(
            nucleotides[:, index, :], index, address_register_size
        )


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


def _device_setup(
    device_name: str, wire_quantity: int, distribute: bool = False
) -> qml.devices.Device:
    if device_name == "default.qubit":
        return qml.device(device_name, wires=wire_quantity)

    if device_name == "lightning.gpu":
        if distribute:
            return qml.device(device_name, wires=wire_quantity, batch_obs=True)
        else:
            return qml.device(device_name, wires=wire_quantity, batch_obs=False)


def basis_basic_entangling_torch(
    sequence_length: int,
    quantum_layer_quantity: int,
    device_name: str = "default.qubit",
    distribute: bool = False,
) -> qml.qnn.TorchLayer:
    wire_quantity: int = (
        calculate_address_register_size(sequence_length)
        + data.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    device = _device_setup(device_name, wire_quantity, distribute)

    @qml.qnode(device)
    def _qnode(inputs: Tensor, weights: Tensor):
        basis_encode_all_nucleotides(inputs)
        qml.BasicEntanglerLayers(weights, wires=list(range(wire_quantity)))

        return qml.probs(wires=list(range(wire_quantity)))

    return qml.qnn.TorchLayer(
        _qnode, {"weights": (quantum_layer_quantity, wire_quantity)}
    )


def basis_strongly_entangled_torch(
    sequence_length: int,
    quantum_layer_quantity: int,
    device_name: str = "default.qubit",
    distribute: bool = False,
) -> qml.qnn.TorchLayer:
    wire_quantity: int = (
        calculate_address_register_size(sequence_length)
        + data.UNIQUE_NUCLEOTIDE_QUANTITY
    )
    device = _device_setup(device_name, wire_quantity, distribute)

    @qml.qnode(device)
    def _qnode(inputs: Tensor, weights: Tensor):
        basis_encode_all_nucleotides(inputs)
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


def amplitude_basic_entangling_torch(
    sequence_length: int,
    quantum_layer_quantity: int,
    device_name: str = "default.qubit",
    distribute: bool = False,
) -> qml.qnn.TorchLayer:
    wire_quantity: int = (
        calculate_address_register_size(sequence_length)
        + AMPLITUDE_QUBIT_QUANTITY
    )
    device = _device_setup(device_name, wire_quantity, distribute)

    @qml.qnode(device)
    def _qnode(inputs: Tensor, weights: Tensor):
        amplitude_encode_all_nucleotides(inputs)
        qml.BasicEntanglerLayers(weights, wires=list(range(wire_quantity)))

        return qml.probs(wires=list(range(wire_quantity)))

    return qml.qnn.TorchLayer(
        qml.transforms.broadcast_expand(_qnode),
        {"weights": (quantum_layer_quantity, wire_quantity)},
    )


def amplitude_strongly_entangling_torch(
    sequence_length: int,
    quantum_layer_quantity: int,
    device_name: str = "default.qubit",
    distribute: bool = False,
) -> qml.qnn.TorchLayer:
    wire_quantity: int = (
        calculate_address_register_size(sequence_length)
        + AMPLITUDE_QUBIT_QUANTITY
    )
    device = _device_setup(device_name, wire_quantity, distribute)

    @qml.qnode(device)
    def _qnode(inputs: Tensor, weights: Tensor):
        amplitude_encode_all_nucleotides(inputs)
        qml.StronglyEntanglingLayers(weights, wires=list(range(wire_quantity)))

        return qml.probs(wires=list(range(wire_quantity)))

    return qml.qnn.TorchLayer(
        qml.transforms.broadcast_expand(_qnode),
        {
            "weights": (
                quantum_layer_quantity,
                wire_quantity,
                UNIQUE_ROTATIONS_QUANTITY,
            )
        },
    )
