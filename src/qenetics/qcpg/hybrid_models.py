"""
Hybrid quantum-classical methylation models: trained circuits combined with
trained classical layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, log2, pi, sqrt
from typing import Any

import pennylane as qp
import torch
from torch import Tensor, nn

from qenetics.circuits import ansatz as circuit_ansatz
from qenetics.circuits import device_utils, measurement
from qenetics.circuits.ansatz import LayeredAnsatz
from qenetics.qcpg import heads
from qenetics.qcpg.filter_banks import MultiScaleFilterBank
from qenetics.qcpg.qcpg_models import (
    calculate_address_register_size,
    sequence_amplitudes,
)
from qenetics.qcpg.sequence_model import (
    PER_OUTPUT_READOUT,
    SHARED_READOUT,
    SequenceModel,
    build_qnode,
    check_probability_differentiation,
    measured_wires,
    require_choice,
    require_positive,
    resolve_readout_wires,
    wire_probabilities,
)

AFFINE_RELU_ACTIVATION: str = "affine_relu"
NO_ACTIVATION: str = "none"
ACTIVATIONS: tuple[str, ...] = (AFFINE_RELU_ACTIVATION, NO_ACTIVATION)
DEEPCPG_TAIL: str = "deepcpg"
POOLING_TAIL: str = "pooling"
QUANTUM_TAIL: str = "quantum_filter_bank"
TAILS: tuple[str, ...] = (DEEPCPG_TAIL, POOLING_TAIL, QUANTUM_TAIL)
PURE_HEAD: str = "pure"
DENSE_HEAD: str = "dense"
HEADS: tuple[str, ...] = (PURE_HEAD, DENSE_HEAD)
REPO_FEATURE_SCALE: str = "repo"
NORMALIZED_FEATURE_SCALE: str = "normalized"
FEATURE_SCALES: tuple[str, ...] = (REPO_FEATURE_SCALE, NORMALIZED_FEATURE_SCALE)


def _require_tail_options(
    tail: str, options: dict[str, object], tail_of_options: str
) -> None:
    """
    Raises
    ------
    ValueError if options of one tail are given for another.
    """
    given: list[str] = [
        name for name, value in options.items() if value is not None
    ]
    if given and tail != tail_of_options:
        raise ValueError(
            f"Options {', '.join(given)} apply to the {tail_of_options} tail, "
            f"not the {tail} tail"
        )


class FilterBankNetwork(SequenceModel):
    """
    Quantum filter banks as the first layer, an optional per-channel affine
    map and ReLU, and a classical or quantum tail.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        window_lengths: Sequence[int] = (16,),
        quantum_layer_quantity: int = 4,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        circuit_quantity: int = 1,
        readout: str = measurement.PROBABILITY_READOUT,
        marginal_wire_quantity: int = 1,
        constant_amplitude: float = 0.0,
        trainable_circuit: bool = True,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
        activation: str = AFFINE_RELU_ACTIVATION,
        tail: str = DEEPCPG_TAIL,
        hidden_layer_sizes: Sequence[int] | None = None,
        dropout: float | None = None,
        first_pool: str | None = None,
        pooling: str | None = None,
        channel_mix: bool | None = None,
        tail_window_length: int | None = None,
        tail_quantum_layer_quantity: int | None = None,
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        sequence_length: The positions of every input.
        output_quantity: The number of outputs (cells).
        window_lengths: One filter bank per window length, see
            `MultiScaleFilterBank`.
        quantum_layer_quantity, ansatz, circuit_quantity, readout,
        marginal_wire_quantity, constant_amplitude, trainable_circuit,
        initialization, initialization_scale: The options of every bank, see
            `QuantumFilterBank`.
        activation: `AFFINE_RELU_ACTIVATION` or `NO_ACTIVATION` after the
            banks.
        tail: `DEEPCPG_TAIL`, `POOLING_TAIL` or `QUANTUM_TAIL`.
        hidden_layer_sizes: The tail's dense layers. Defaults to DeepCpG's
            128 units, or none for the pooling tail.
        dropout: The tail's dropout. Defaults to 0.2, or 0 for the pooling
            tail.
        first_pool: The DeepCpG tail's first pool, see `heads.DeepCpGTail`.
        pooling, channel_mix: The pooling tail's options, see
            `heads.PoolingHead`.
        tail_window_length, tail_quantum_layer_quantity: The quantum tail's
            options, see `heads.QuantumFilterBankTail`.
        encoding_options: The input encoding, see `SequenceModel`.

        Raises
        ------
        ValueError if an option is invalid, or given for a tail that does
        not use it.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        require_choice("activation", activation, ACTIVATIONS)
        require_choice("tail", tail, TAILS)
        _require_tail_options(tail, {"first_pool": first_pool}, DEEPCPG_TAIL)
        _require_tail_options(
            tail, {"pooling": pooling, "channel_mix": channel_mix}, POOLING_TAIL
        )
        _require_tail_options(
            tail,
            {
                "tail_window_length": tail_window_length,
                "tail_quantum_layer_quantity": tail_quantum_layer_quantity,
            },
            QUANTUM_TAIL,
        )
        self.front_end = MultiScaleFilterBank(
            self.channel_quantity,
            window_lengths,
            quantum_layer_quantity=quantum_layer_quantity,
            ansatz=ansatz,
            circuit_quantity=circuit_quantity,
            readout=readout,
            marginal_wire_quantity=marginal_wire_quantity,
            constant_amplitude=constant_amplitude,
            trainable_circuit=trainable_circuit,
            initialization=initialization,
            initialization_scale=initialization_scale,
        )
        channel_quantity: int = self.front_end.output_channel_quantity
        position_quantity: int = self.front_end.position_quantity(
            sequence_length
        )
        self.activation: nn.Module = (
            heads.ChannelAffineReLU(channel_quantity)
            if activation == AFFINE_RELU_ACTIVATION
            else nn.Identity()
        )
        if tail == DEEPCPG_TAIL:
            self.tail: nn.Module = heads.DeepCpGTail(
                channel_quantity,
                position_quantity,
                output_quantity,
                hidden_layer_sizes=(
                    heads.DEEPCPG_HIDDEN_LAYER_SIZES
                    if hidden_layer_sizes is None
                    else hidden_layer_sizes
                ),
                dropout=heads.DEEPCPG_DROPOUT if dropout is None else dropout,
                first_pool=first_pool or heads.MAX_POOL,
            )
        elif tail == POOLING_TAIL:
            self.tail = heads.PoolingHead(
                channel_quantity,
                output_quantity,
                pooling=pooling or heads.MAX_MEAN_POOLING,
                channel_mix=bool(channel_mix),
                hidden_layer_sizes=hidden_layer_sizes or (),
                dropout=dropout or 0.0,
            )
        else:
            self.tail = heads.QuantumFilterBankTail(
                channel_quantity,
                position_quantity,
                output_quantity,
                window_length=tail_window_length or 3,
                quantum_layer_quantity=tail_quantum_layer_quantity or 4,
                ansatz=ansatz,
                hidden_layer_sizes=(
                    heads.DEEPCPG_HIDDEN_LAYER_SIZES
                    if hidden_layer_sizes is None
                    else hidden_layer_sizes
                ),
                dropout=heads.DEEPCPG_DROPOUT if dropout is None else dropout,
                initialization=initialization,
                initialization_scale=initialization_scale,
            )

    def _probabilities(self, channels: Tensor) -> Tensor:
        return torch.sigmoid(
            self.tail(self.activation(self.front_end(channels)))
        )


class GlobalCircuitModel(SequenceModel):
    """
    The whole sequence in one circuit, as `qcpg_models.QNN`.

    Channel c of position p is the amplitude of |p>|c> on an address register
    and a data register, and addresses past the sequence hold the data
    register's |0...0>. A trained circuit acts on every wire or, with
    `block_qubit_quantity`, on the lowest wires only, i.e. identically on
    every aligned block of positions. The pure head predicts P(wire = 1); the
    dense head feeds all probabilities, scaled, to dense layers.

    The encoded state is normalized, as every device needs. Unknown
    nucleotides, encoded as zeros, therefore give slightly different states
    than the report's simulator, which left them unnormalized; sequences
    without them give the same.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        quantum_layer_quantity: int = 5,
        ansatz: str = circuit_ansatz.BASIC_ANSATZ_STR,
        head: str = DENSE_HEAD,
        hidden_layer_sizes: Sequence[int] = (),
        dropout: float = 0.2,
        feature_scale: str = NORMALIZED_FEATURE_SCALE,
        readout_wires: str | Sequence[int] = SHARED_READOUT,
        block_qubit_quantity: int | None = None,
        shifts: Sequence[int] = (0,),
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
        device_name: str = device_utils.DEFAULT_DEVICE,
        diff_method: str = "best",
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        quantum_layer_quantity: The circuit's layers.
        ansatz: The ansatz, see `ansatz.ANSATZES_STRINGS`.
        head: `PURE_HEAD` or `DENSE_HEAD`.
        hidden_layer_sizes: The dense head's hidden layers; none for a
            linear layer.
        dropout: The dropout after each hidden layer.
        feature_scale: The dense head's input is the probabilities times 2^n
            for `REPO_FEATURE_SCALE`, as `qcpg_models.QNN`, or times
            2^n / sqrt(2^address qubits) for `NORMALIZED_FEATURE_SCALE`.
        readout_wires: The pure head's wires, see
            `sequence_model.resolve_readout_wires`. Wire 0 is the most
            significant address qubit; the data qubits are the last wires.
        block_qubit_quantity: If given, the circuit acts on this many of the
            lowest wires only.
        shifts: The dense head reads the sequence from each offset, padded at
            the end, and concatenates the probabilities.
        initialization: The initialization of the angles.
        initialization_scale: The standard deviation of normal angles.
        device_name: The PennyLane device.
        diff_method: The PennyLane differentiation method.

        Raises
        ------
        ValueError if an option is invalid.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        require_choice("head", head, HEADS)
        require_choice("feature scale", feature_scale, FEATURE_SCALES)
        self.address_quantity: int = calculate_address_register_size(
            sequence_length
        )
        self.data_quantity: int = max(1, ceil(log2(self.channel_quantity)))
        self.wire_quantity: int = self.address_quantity + self.data_quantity
        circuit_wire_quantity: int = block_qubit_quantity or self.wire_quantity
        if not 1 <= circuit_wire_quantity <= self.wire_quantity:
            raise ValueError(
                f"The circuit's {circuit_wire_quantity} qubits must be between "
                f"1 and the register's {self.wire_quantity}"
            )
        self.shifts: list[int] = list(shifts)
        if not self.shifts or any(
            not 0 <= shift < sequence_length for shift in self.shifts
        ):
            raise ValueError(
                f"Shifts must be between 0 and {sequence_length - 1}, not "
                f"{self.shifts}"
            )
        if head == PURE_HEAD and self.shifts != [0]:
            raise ValueError("The pure head reads the unshifted sequence only")

        self.ansatz = ansatz
        self.head_kind = head
        self.circuit_wires: list[int] = list(
            range(
                self.wire_quantity - circuit_wire_quantity, self.wire_quantity
            )
        )
        self.circuit_weights = nn.Parameter(
            circuit_ansatz.initial_weights(
                circuit_ansatz.weight_shape(
                    ansatz, quantum_layer_quantity, circuit_wire_quantity
                ),
                initialization,
                initialization_scale,
            )
        )
        if head == PURE_HEAD:
            self.measured_wires, self.readout_indices = measured_wires(
                resolve_readout_wires(
                    readout_wires, output_quantity, self.wire_quantity
                )
            )
            self.head = None
        else:
            check_probability_differentiation(device_name, diff_method)
            self.feature_scale: float = 2**self.wire_quantity / (
                1.0
                if feature_scale == REPO_FEATURE_SCALE
                else sqrt(2**self.address_quantity)
            )
            self.head = heads.multilayer_perceptron(
                2**self.wire_quantity * len(self.shifts),
                hidden_layer_sizes,
                output_quantity,
                dropout=dropout,
            )
        self.qnode = build_qnode(
            self._circuit, self.wire_quantity, device_name, diff_method
        )
        self.shots: int = 0

    def _circuit(
        self, amplitudes: Tensor, weights: Tensor
    ) -> qp.measurements.MeasurementProcess | list:
        qp.AmplitudeEmbedding(
            amplitudes, wires=range(self.wire_quantity), normalize=True
        )
        LayeredAnsatz(weights, wires=self.circuit_wires, ansatz=self.ansatz)
        if self.head is None:
            return [qp.expval(qp.PauliZ(wire)) for wire in self.measured_wires]
        return qp.probs(wires=range(self.wire_quantity))

    def _amplitudes(self, channels: Tensor, shift: int) -> Tensor:
        """The register's amplitudes for the sequence read from `shift`."""
        positions: Tensor = channels[:, :, shift:].transpose(1, 2)
        positions = nn.functional.pad(
            positions, (0, 2**self.data_quantity - self.channel_quantity)
        )
        return sequence_amplitudes(
            positions,
            positions.shape[1],
            self.address_quantity,
            embedding_qubit_quantity=self.data_quantity,
        )

    def _probabilities(self, channels: Tensor) -> Tensor:
        if self.head is None:
            expectations: Tensor = torch.stack(
                list(
                    self.qnode(
                        self._amplitudes(channels, 0), self.circuit_weights
                    )
                ),
                -1,
            )[:, self.readout_indices]
            return wire_probabilities(
                expectations.to(channels.dtype),
                self.shots,
                is_training=self.training,
            )

        features: list[Tensor] = []
        for shift in self.shifts:
            probabilities: Tensor = self.qnode(
                self._amplitudes(channels, shift), self.circuit_weights
            ).to(channels.dtype)
            if self.shots and not self.training:
                probabilities = measurement.sample_probabilities(
                    probabilities, self.shots
                )
            features.append(probabilities * self.feature_scale)
        return torch.sigmoid(self.head(torch.cat(features, 1)))


class CNNQuantumHead(SequenceModel):
    """
    DeepCpG's convolutional layers and dense layers give angles, which are
    uploaded as RY rotations before each of several trained circuit blocks.
    The prediction for output c is P(wire = 1) of its readout wire.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        wire_quantity: int = 8,
        quantum_layer_quantity: int = 2,
        upload_quantity: int = 3,
        separate_angles: bool = True,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        hidden_layer_sizes: Sequence[int] = heads.DEEPCPG_HIDDEN_LAYER_SIZES,
        dropout: float = heads.DEEPCPG_DROPOUT,
        readout_wires: str | Sequence[int] = PER_OUTPUT_READOUT,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
        device_name: str = device_utils.DEFAULT_DEVICE,
        diff_method: str = "best",
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        wire_quantity: The circuit's qubits.
        quantum_layer_quantity: The layers of each circuit block.
        upload_quantity: The number of angle uploads and circuit blocks.
        separate_angles: Upload different angles, pi tanh(x), each time, or
            the same angles, (pi / 2) tanh(x), every time.
        ansatz: The ansatz of the blocks, see `ansatz.ANSATZES_STRINGS`.
        hidden_layer_sizes: The dense layers between the convolutions and
            the angles.
        dropout: The dropout after each dense layer.
        readout_wires: See `sequence_model.resolve_readout_wires`.
        initialization: The initialization of the angles.
        initialization_scale: The standard deviation of normal angles.
        device_name: The PennyLane device.
        diff_method: The PennyLane differentiation method.

        Raises
        ------
        ValueError if an option is invalid or the sequences are too short.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        require_positive("wire quantity", wire_quantity)
        require_positive("upload quantity", upload_quantity)
        self.wire_quantity = wire_quantity
        self.upload_quantity = upload_quantity
        self.separate_angles = separate_angles
        self.ansatz = ansatz
        self.body, flattened_quantity = heads.deepcpg_body(
            self.channel_quantity, sequence_length
        )
        self.angle_layers = heads.multilayer_perceptron(
            flattened_quantity,
            hidden_layer_sizes,
            wire_quantity * (upload_quantity if separate_angles else 1),
            dropout=dropout,
        )
        self.circuit_weights = nn.Parameter(
            circuit_ansatz.initial_weights(
                (
                    upload_quantity,
                    *circuit_ansatz.weight_shape(
                        ansatz, quantum_layer_quantity, wire_quantity
                    ),
                ),
                initialization,
                initialization_scale,
            )
        )
        self.measured_wires, self.readout_indices = measured_wires(
            resolve_readout_wires(readout_wires, output_quantity, wire_quantity)
        )
        self.qnode = build_qnode(
            self._circuit, wire_quantity, device_name, diff_method
        )
        self.shots: int = 0

    def _circuit(
        self, angles: Tensor, weights: Tensor
    ) -> list[qp.measurements.ExpectationMP]:
        for upload in range(self.upload_quantity):
            for wire in range(self.wire_quantity):
                qp.RY(angles[:, upload, wire], wires=wire)
            LayeredAnsatz(
                weights[upload],
                wires=range(self.wire_quantity),
                ansatz=self.ansatz,
            )
        return [qp.expval(qp.PauliZ(wire)) for wire in self.measured_wires]

    def _probabilities(self, channels: Tensor) -> Tensor:
        activations: Tensor = torch.tanh(self.angle_layers(self.body(channels)))
        if self.separate_angles:
            angles: Tensor = pi * activations.reshape(
                -1, self.upload_quantity, self.wire_quantity
            )
        else:
            angles = (pi / 2) * activations.unsqueeze(1).expand(
                -1, self.upload_quantity, -1
            )
        expectations: Tensor = torch.stack(
            list(self.qnode(angles, self.circuit_weights)), -1
        )[:, self.readout_indices]
        return wire_probabilities(
            expectations.to(channels.dtype),
            self.shots,
            is_training=self.training,
        )
