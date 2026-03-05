from __future__ import annotations

from collections.abc import Callable
from math import ceil, log2

import pennylane as qml
import torch.nn
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
        measurement: str = "probability",
        device_name: str = "default.qubit",
        distribute: bool = True,
    ) -> None:
        super(QNN, self).__init__()
        global SEQUENCE_LENGTH
        SEQUENCE_LENGTH = sequence_length

        self.qnn = define_torch_qnode(
            sequence_length,
            quantum_layer_quantity,
            device_name=device_name,
            distribute=distribute,
            encoding=encoding,
            entangling=entangler,
            measurement=measurement,
        )

        wire_quantity: int = (
            calculate_address_register_size(sequence_length)
            + AMPLITUDE_QUBIT_QUANTITY
        )
        self.linear = _define_fcl(
            wire_quantity, output_quantity, measurement=measurement
        )

    def forward(self: QNN, x: Tensor) -> Tensor:
        # Must change has_batch_dim in pennylane\qnn\torch.py:forward to True for multi-dimensional inputs
        x = self.qnn(x)
        x = self.linear(x)
        output: Tensor = functional.sigmoid(x)

        return output


def _define_fcl(
    wire_quantity: int,
    output_quantity: int,
    *,
    measurement: str = "probability",
) -> torch.nn.Linear:
    if measurement == "probability":
        return nn.Linear(2**wire_quantity, output_quantity)

    if measurement == "expectation":
        return nn.Linear(wire_quantity, output_quantity)

    raise ValueError(f"Unknown measurement type {measurement}")


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


def _encode_all_nucleotides(nucleotides, encoding: str = "amplitude") -> None:
    if encoding == "amplitude":
        amplitude_encode_all_nucleotides(nucleotides)
    elif encoding == "basis":
        basis_encode_all_nucleotides(nucleotides)
    else:
        raise ValueError(f"Unknown encoding type {encoding}")


def _determine_wire_quantity(
    sequence_length: int, encoding: str = "amplitude"
) -> int:
    if encoding == "amplitude":
        return (
            calculate_address_register_size(sequence_length)
            + AMPLITUDE_QUBIT_QUANTITY
        )

    if encoding == "basis":
        return (
            calculate_address_register_size(sequence_length)
            + data.UNIQUE_NUCLEOTIDE_QUANTITY
        )

    raise ValueError(f"Unknown encoding method {encoding}")


def _apply_entangling_layer(
    weights: Tensor, wire_quantity: int, *, entangling: str = "basic"
) -> None:
    if entangling == "basic":
        qml.BasicEntanglerLayers(weights, wires=list(range(wire_quantity)))
    elif entangling == "strong":
        qml.StronglyEntanglingLayers(weights, wires=list(range(wire_quantity)))
    else:
        raise ValueError(f"Unknown entangling layer type {entangling}")


def _measure(
    wire_quantity: int, *, measurement: str = "probability"
) -> qml.measurements.ProbabilityMP | list[qml.measurements.ExpectationMP]:
    if measurement == "probability":
        return qml.probs(wires=list(range(wire_quantity)))

    if measurement == "expectation":
        return [
            qml.expval(qml.PauliZ(qubit_index))
            for qubit_index in range(wire_quantity)
        ]

    raise ValueError(f"Unknown measurement type {measurement}")


def _convert_qnode_to_torch_layer(
    qnode: Callable[Tensor, Tensor],
    quantum_layer_quantity: int,
    wire_quantity: int,
    *,
    entangling: str = "basic",
) -> qml.qnn.TorchLayer:
    if entangling == "basic":
        weights: dict[str, tuple[int, int, int]] = {
            "weights": qml.BasicEntanglerLayers.shape(
                n_layers=quantum_layer_quantity, n_wires=wire_quantity
            )
        }
    elif entangling == "strong":
        weights = {
            "weights": qml.StronglyEntanglingLayers.shape(
                n_layers=quantum_layer_quantity, n_wires=wire_quantity
            )
        }
    else:
        raise ValueError(f"Unknown entangling layer type {entangling}")

    return qml.qnn.TorchLayer(qml.transforms.broadcast_expand(qnode), weights)


def define_torch_qnode(
    sequence_length: int,
    quantum_layer_quantity: int,
    *,
    device_name: str = "default.qubit",
    distribute: bool = False,
    encoding: str = "amplitude",
    entangling: str = "basic",
    measurement: str = "probability",
) -> qml.qnn.TorchLayer:
    wire_quantity: int = _determine_wire_quantity(sequence_length, encoding)
    device: qml.devices.Device = _device_setup(
        device_name, wire_quantity, distribute
    )

    @qml.qnode(device)
    def _qnode(inputs: Tensor, weights: Tensor):
        _encode_all_nucleotides(inputs, encoding)
        for _ in range(quantum_layer_quantity):
            _apply_entangling_layer(
                weights, wire_quantity, entangling=entangling
            )

        return _measure(wire_quantity, measurement=measurement)

    return _convert_qnode_to_torch_layer(
        _qnode, quantum_layer_quantity, wire_quantity, entangling=entangling
    )
