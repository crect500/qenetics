# Must change the following in pennylane/qnn/torch.py
# - # reshape to the correct number of dimensions
# - if has_batch_dim:
# -     results = torch.reshape(results, (batch_dims, *results.shape[1:]))
#
# def _combine_dimensions(_res):
#   if len(x.shape) > 1:
# -     _res = [torch.reshape(r, (x.shape[0], -1)) for r in _res]
# - return torch.hstack(_res).type(x.dtype)
# + return torch.stack(_res).type(x.dtype)

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil, log2, sqrt

import pennylane as qp
import torch.nn
from torch import Tensor, nn
from torch.nn import functional

AMPLITUDE_QUBIT_QUANTITY: int = 2
UNIQUE_ROTATIONS_QUANTITY: int = 3
SEQUENCE_LENGTH: int = -1


class QNN(nn.Module):
    """
    Amplitude-encoded QNN with an FCL.

    """

    def __init__(
        self: QNN,
        sequence_length: int,
        quantum_layer_quantity: int,
        output_quantity: int,
        *,
        encoding: str = "onehot",
        embedding_qubit_quantity: int | None = None,
        vocabulary_size: int | None = None,
        fcl_quantity: int | None = 1,
        entangling: str = "basic",
        measurement: str = "expectation",
        device_name: str = "default.qubit",
        diff_method: str = "adjoint",
        distribute: bool = False,
    ) -> None:
        super().__init__()
        global SEQUENCE_LENGTH
        SEQUENCE_LENGTH = sequence_length
        if encoding == "token":
            self.embedding = nn.Embedding(
                vocabulary_size, 2**embedding_qubit_quantity
            )
            data_qubit_quantity: int = embedding_qubit_quantity
        elif encoding == "onehot":
            self.embedding = None
            data_qubit_quantity = AMPLITUDE_QUBIT_QUANTITY

        wire_quantity: int = (
            calculate_address_register_size(sequence_length)
            + data_qubit_quantity
        )
        self.measurement = measurement
        self.wire_quantity = wire_quantity

        self.qnn = _torch_qnn_layer(
            sequence_length,
            quantum_layer_quantity,
            encoding=encoding,
            embedding_qubit_quantity=embedding_qubit_quantity,
            device_name=device_name,
            distribute=distribute,
            entangling=entangling,
            measurement=measurement,
            diff_method=diff_method,
        )

        if fcl_quantity is None:
            self.linear = None
            if output_quantity != 1:
                raise NotImplementedError(
                    "Purely quantum network with more than one output is not supported"
                )
        elif fcl_quantity == 1:
            self.linear = _define_fcl(
                wire_quantity, output_quantity, measurement=measurement
            )
        else:
            raise NotImplementedError(
                f"fcl quantity of {fcl_quantity} not supported"
            )

    def forward(self: QNN, x: Tensor) -> Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
            x = functional.normalize(x, dim=-1)

        x = self.qnn(x)
        if self.measurement == "probability":
            x = x * 2**self.wire_quantity

        if self.linear is None:
            return x
        elif isinstance(self.linear, nn.Linear):
            x = self.linear(x)
            return functional.sigmoid(x)
        else:
            raise NotImplementedError("Multiple linear layers not supported")


class RQNN(nn.Module):
    def __init__(
        self: QNN,
        vocabulary_size: int,
        embedding_size: int,
        hidden_qubit_quantity: int,
        output_quantity: int,
        *,
        hidden_layer_quantity: int,
        entangling: str = "strong",
        measurement: str = "expectation",
        device_name: str = "default.qubit",
        diff_method: str = "best",
        shots: int = 1024,
    ) -> None:
        super().__init__()
        embedding_qubit_quantity: int = ceil(log2(embedding_size))
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        self.qnn = _torch_rqnn_layer(
            embedding_qubit_quantity,
            hidden_qubit_quantity,
            hidden_layer_quantity=hidden_layer_quantity,
            device_name=device_name,
            entangling=entangling,
            measurement=measurement,
            diff_method=diff_method,
            shots=shots,
        )

        self.linear = _define_fcl(
            hidden_qubit_quantity, output_quantity, measurement=measurement
        )

    def forward(self: QNN, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = functional.normalize(x, dim=-1)
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

    return ceil(log2(encode_quantity))


def _sequence_amplitudes(
    nucleotides: Tensor,
    sequence_length: int,
    address_register_size: int,
    *,
    embedding_qubit_quantity: int = AMPLITUDE_QUBIT_QUANTITY,
) -> Tensor:
    """
    Flatten a nucleotide sequence into the amplitude vector it addresses.

    Args
    ----
    nucleotides: A sequence of per-position amplitude vectors, batched or not.
    sequence_length: The number of positions in the sequence.
    address_register_size: The size of the quantum address register.
    embedding_qubit_quantity: The size of the quantum data register.

    Returns
    -------
    The amplitude vector of the full circuit register.
    """
    address_range: int = 2**address_register_size
    block_size: int = 2**embedding_qubit_quantity
    is_batched: bool = len(nucleotides.shape) == 3
    if not is_batched:
        nucleotides = nucleotides.unsqueeze(0)

    # Addresses past the end of the sequence are never loaded, so their data
    # register stays in the all-zero basis state.
    padding = torch.zeros(
        nucleotides.shape[0],
        address_range - sequence_length,
        block_size,
        dtype=nucleotides.dtype,
        device=nucleotides.device,
    )
    padding[..., 0] = 1.0
    amplitudes: Tensor = torch.cat(
        [nucleotides[:, :sequence_length, :], padding], dim=1
    ).reshape(-1, address_range * block_size) / sqrt(address_range)

    return amplitudes if is_batched else amplitudes[0]


def _encode_all_nucleotides(
    nucleotides: Tensor,
    sequence_length: int,
    address_register_size: int,
    *,
    embedding_qubit_quantity: int = AMPLITUDE_QUBIT_QUANTITY,
) -> None:
    """
    Encode nucleotide values into a quantum circuit at appropriate addresses.


    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    qp.AmplitudeEmbedding(
        _sequence_amplitudes(
            nucleotides,
            sequence_length,
            address_register_size,
            embedding_qubit_quantity=embedding_qubit_quantity,
        ),
        wires=list(range(address_register_size + embedding_qubit_quantity)),
    )


def _device_setup(
    device_name: str, wire_quantity: int, *, distribute: bool = False
) -> qp.devices.Device:

    if device_name in ["default.qubit", "lightning.qubit"]:
        return qp.device(device_name, wires=wire_quantity)

    if device_name == "lightning.gpu":
        if distribute:
            return qp.device(device_name, wires=wire_quantity, batch_obs=True)
        else:
            return qp.device(device_name, wires=wire_quantity, batch_obs=False)

    raise ValueError(f"Device name {device_name} not recognized")


def _apply_entangling_layer(
    weights: Tensor,
    wire_quantity: int,
    *,
    entangling: str = "basic",
) -> None:
    if entangling == "basic":
        qp.BasicEntanglerLayers(weights, wires=list(range(wire_quantity)))
    elif entangling == "strong":
        qp.StronglyEntanglingLayers(weights, wires=list(range(wire_quantity)))
    else:
        raise ValueError(f"Unknown entangling layer type {entangling}")


def _measure(
    wire_quantity: int,
    *,
    measurement: str = "probability",
) -> qp.measurements.ProbabilityMP | list[qp.measurements.ExpectationMP]:
    # Reads out the address register alongside the data register. Measuring the
    # data register alone traces the address register out, which averages the
    # readout over sequence positions and leaves the model blind to nucleotide
    # order.
    if measurement == "probability":
        return qp.probs(wires=list(range(wire_quantity)))

    if measurement == "expectation":
        return [
            qp.expval(qp.PauliZ(qubit_index))
            for qubit_index in range(wire_quantity)
        ]

    raise ValueError(f"Unknown measurement type {measurement}")


def _convert_qnode_to_torch_layer(
    qnode: Callable[Tensor, Tensor],
    quantum_layer_quantity: int,
    wire_quantity: int,
    *,
    entangling: str = "basic",
) -> qp.qnn.TorchLayer:
    if entangling == "basic":
        weights: dict[str, tuple[int, int, int]] = {
            "weights": qp.BasicEntanglerLayers.shape(
                n_layers=quantum_layer_quantity,
                n_wires=wire_quantity,
            )
        }
    elif entangling == "strong":
        weights = {
            "weights": qp.StronglyEntanglingLayers.shape(
                n_layers=quantum_layer_quantity,
                n_wires=wire_quantity,
            )
        }
    else:
        raise ValueError(f"Unknown entangling layer type {entangling}")

    return qp.qnn.TorchLayer(qnode, weights)


def _torch_qnn_layer(
    sequence_length: int,
    quantum_layer_quantity: int,
    *,
    encoding: str = "onehot",
    embedding_qubit_quantity: int | None = None,
    device_name: str = "default.qubit",
    distribute: bool = False,
    entangling: str = "basic",
    measurement: str = "probability",
    diff_method="adjoint",
) -> qp.qnn.TorchLayer:
    if encoding == "onehot":
        wire_quantity: int = (
            calculate_address_register_size(sequence_length)
            + AMPLITUDE_QUBIT_QUANTITY
        )
        embedding_qubit_quantity = AMPLITUDE_QUBIT_QUANTITY
    elif encoding == "token":
        if embedding_qubit_quantity is None or embedding_qubit_quantity < 1:
            raise ValueError(
                "Must provide embedding_qubit_quantity if token encoding is desired"
            )

        wire_quantity: int = (
            calculate_address_register_size(sequence_length)
            + embedding_qubit_quantity
        )
    else:
        raise ValueError(f"Encoding method {encoding} not recognized")

    device: qp.devices.Device = _device_setup(
        device_name, wire_quantity, distribute=distribute
    )

    address_register_size: int = calculate_address_register_size(
        sequence_length
    )

    @qp.qnode(device, interface="torch", diff_method=diff_method)
    def _qnode(inputs: Tensor, weights: Tensor):
        _encode_all_nucleotides(
            inputs,
            sequence_length,
            address_register_size,
            embedding_qubit_quantity=embedding_qubit_quantity,
        )
        _apply_entangling_layer(
            weights,
            wire_quantity,
            entangling=entangling,
        )

        return _measure(wire_quantity, measurement=measurement)

    return _convert_qnode_to_torch_layer(
        _qnode,
        quantum_layer_quantity,
        wire_quantity,
        entangling=entangling,
    )


def _qrnn_block(
    weights: Tensor,
    input_wire_indices: Sequence[int],
    hidden_wire_indices: Sequence[int],
    *,
    entangling: str = "basic",
) -> None:
    for input_index in input_wire_indices:
        for hidden_index in hidden_wire_indices:
            qp.CNOT(wires=[input_index, hidden_index])

    if entangling == "basic":
        qp.BasicEntanglerLayers(weights, wires=hidden_wire_indices)
    elif entangling == "strong":
        qp.StronglyEntanglingLayers(weights, wires=hidden_wire_indices)
    else:
        raise ValueError(f"Unknown entangling layer type {entangling}")


def _torch_rqnn_layer(
    embedding_qubit_quantity: int,
    hidden_qubit_quantity: int,
    *,
    hidden_layer_quantity: int,
    device_name: str = "default.qubit",
    entangling: str = "basic",
    measurement: str = "expectation",
    diff_method: str = "best",
    shots: int = 1024,
) -> qp.qnn.TorchLayer:
    if diff_method == "adjoint":
        raise ValueError("RQNN circuit does not support adjoint diff method.")

    wire_quantity: int = embedding_qubit_quantity + hidden_qubit_quantity
    device: qp.devices.Device = _device_setup(device_name, wire_quantity)
    input_wire_indices: list[int] = list(range(embedding_qubit_quantity))
    hidden_wire_indices: list[int] = list(
        range(embedding_qubit_quantity, wire_quantity)
    )

    @qp.set_shots(shots)
    @qp.qnode(
        device,
        interface="torch",
        mcm_method="one-shot",
        diff_method=diff_method,
    )
    def _rqnn_circuit(inputs: Tensor, weights: Tensor) -> None:
        if len(inputs.shape) == 2:
            for embedding in inputs:
                qp.AmplitudeEmbedding(
                    embedding,
                    list(range(embedding_qubit_quantity)),
                    normalize=True,
                )

                _qrnn_block(
                    weights,
                    input_wire_indices=input_wire_indices,
                    hidden_wire_indices=hidden_wire_indices,
                    entangling=entangling,
                )

                for index in input_wire_indices:
                    qp.measure(index, reset=True)

        elif len(inputs.shape) == 3:
            for index in range(inputs.shape[1]):
                qp.AmplitudeEmbedding(
                    inputs[:, index, :],
                    list(range(embedding_qubit_quantity)),
                    normalize=True,
                )

                _qrnn_block(
                    weights,
                    input_wire_indices=input_wire_indices,
                    hidden_wire_indices=hidden_wire_indices,
                    entangling=entangling,
                )

                for index in input_wire_indices:
                    qp.measure(index, reset=True)
        else:
            raise ValueError(f"Invalid inputs shape {inputs.shape}")

        if measurement == "probability":
            return qp.probs(
                wires=list(range(embedding_qubit_quantity, wire_quantity))
            )

        if measurement == "expectation":
            return [
                qp.expval(qp.PauliZ(qubit_index))
                for qubit_index in range(
                    embedding_qubit_quantity, wire_quantity
                )
            ]

    return _convert_qnode_to_torch_layer(
        _rqnn_circuit,
        hidden_layer_quantity,
        wire_quantity=hidden_qubit_quantity,
        entangling=entangling,
    )
