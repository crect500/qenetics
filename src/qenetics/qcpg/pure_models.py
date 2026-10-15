"""
Pure quantum methylation models: no trained classical parameters.

Each prediction is a measured probability. Fixed, untrained preprocessing is
allowed: one-hot encoding, sequence statistics standardized without trained
parameters, and averaging over windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, pi
from typing import Any

import pennylane as qp
import torch
from torch import Tensor, nn
from torch.nn import functional

from qenetics.circuits import ansatz as circuit_ansatz
from qenetics.circuits import device_utils, measurement
from qenetics.circuits.ansatz import LayeredAnsatz
from qenetics.qcpg.filter_banks import QuantumFilterBank
from qenetics.qcpg.sequence_model import (
    SHARED_READOUT,
    STATISTIC_QUANTITY,
    SequenceModel,
    build_qnode,
    clamp_probabilities,
    measured_wires,
    require_positive,
    resolve_readout_wires,
    sequence_statistics,
    wire_probabilities,
)


class WindowAverageClassifier(SequenceModel):
    """
    A window circuit measured at every offset: the prediction is P(wire = 1)
    averaged over offsets and circuits, i.e. the probability of measuring 1
    for a window and circuit chosen uniformly at random.
    """

    readout_matrix: Tensor

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        window_length: int = 16,
        quantum_layer_quantity: int = 4,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        circuit_quantity: int = 1,
        readout_wires: str | Sequence[int] = SHARED_READOUT,
        centre_length: int | None = None,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        sequence_length: The positions of every input.
        output_quantity: The number of outputs (cells).
        window_length: The positions of each window.
        quantum_layer_quantity: The layers of each circuit.
        ansatz: The ansatz, see `ansatz.ANSATZES_STRINGS`.
        circuit_quantity: The number of circuits averaged over.
        readout_wires: See `sequence_model.resolve_readout_wires`.
        centre_length: If given, only windows within this many central
            positions are used.
        initialization: The initialization of the angles.
        initialization_scale: The standard deviation of normal angles.
        encoding_options: The input encoding, see `SequenceModel`.

        Raises
        ------
        ValueError if an option is invalid.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        if centre_length is not None and not (
            window_length <= centre_length <= sequence_length
        ):
            raise ValueError(
                f"The centre length {centre_length} must be between the window "
                f"length {window_length} and the sequence length "
                f"{sequence_length}"
            )
        self.centre_length = centre_length
        self.bank = QuantumFilterBank(
            self.channel_quantity,
            window_length,
            quantum_layer_quantity=quantum_layer_quantity,
            ansatz=ansatz,
            circuit_quantity=circuit_quantity,
            initialization=initialization,
            initialization_scale=initialization_scale,
        )
        self.bank.position_quantity(centre_length or sequence_length)
        wires: list[int] = resolve_readout_wires(
            readout_wires, output_quantity, self.bank.wire_quantity
        )
        self.register_buffer(
            "readout_matrix",
            torch.tensor(measurement.wire_bits(self.bank.wire_quantity, wires)),
            persistent=False,
        )

    def _probabilities(self, channels: Tensor) -> Tensor:
        if self.centre_length is not None:
            start: int = self.sequence_length // 2 - self.centre_length // 2
            channels = channels[:, :, start : start + self.centre_length]
        # (batch, circuits * outcomes, offsets), each window summing to 1.
        probabilities: Tensor = self.bank(channels) / 2**self.bank.wire_quantity
        batch_size: int = len(probabilities)
        mean_probabilities: Tensor = probabilities.reshape(
            batch_size, self.bank.circuit_quantity, -1, probabilities.shape[2]
        ).mean((1, 3))
        return clamp_probabilities(
            mean_probabilities
            @ self.readout_matrix.to(mean_probabilities.dtype)
        )

    @property
    def shots(self) -> int:
        return self.bank.shots

    @shots.setter
    def shots(self, shots: int) -> None:
        self.bank.shots = shots


class FeatureAmplitudeClassifier(SequenceModel):
    """
    The 44 sequence statistics, standardized without trained parameters, and a
    constant 1, amplitude-encoded into 6 qubits; a trained circuit; P(wire = 1).
    """

    WIRE_QUANTITY: int = 6

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        quantum_layer_quantity: int = 12,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        readout_wires: str | Sequence[int] = SHARED_READOUT,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
        device_name: str = device_utils.DEFAULT_DEVICE,
        diff_method: str = "best",
        **encoding_options: Any,
    ) -> None:
        """
        Raises
        ------
        ValueError if an option is invalid or the inputs are not nucleotides.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        self._require_bases()
        self.ansatz = ansatz
        self.normalization = nn.BatchNorm1d(STATISTIC_QUANTITY, affine=False)
        self.circuit_weights = nn.Parameter(
            circuit_ansatz.initial_weights(
                circuit_ansatz.weight_shape(
                    ansatz, quantum_layer_quantity, self.WIRE_QUANTITY
                ),
                initialization,
                initialization_scale,
            )
        )
        self.measured_wires, self.readout_indices = measured_wires(
            resolve_readout_wires(
                readout_wires, output_quantity, self.WIRE_QUANTITY
            )
        )
        self.qnode = build_qnode(
            self._circuit, self.WIRE_QUANTITY, device_name, diff_method
        )
        self.shots: int = 0

    def _circuit(
        self, amplitudes: Tensor, weights: Tensor
    ) -> list[qp.measurements.ExpectationMP]:
        wires: range = range(self.WIRE_QUANTITY)
        qp.AmplitudeEmbedding(amplitudes, wires=wires, normalize=True)
        LayeredAnsatz(weights, wires=wires, ansatz=self.ansatz)
        return [qp.expval(qp.PauliZ(wire)) for wire in self.measured_wires]

    def _probabilities(self, channels: Tensor) -> Tensor:
        statistics: Tensor = self.normalization(sequence_statistics(channels))
        amplitudes: Tensor = functional.pad(
            torch.cat([torch.ones_like(statistics[:, :1]), statistics], 1),
            (0, 2**self.WIRE_QUANTITY - STATISTIC_QUANTITY - 1),
        )
        expectations: Tensor = torch.stack(
            list(self.qnode(amplitudes, self.circuit_weights)), -1
        )[:, self.readout_indices]
        return wire_probabilities(
            expectations.to(statistics.dtype),
            self.shots,
            is_training=self.training,
        )


class FeatureReuploadingClassifier(SequenceModel):
    """
    The 44 sequence statistics, standardized without trained parameters, as
    RY angles (pi / 2) tanh(x) re-uploaded between trained circuit blocks;
    P(wire = 1).
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        wire_quantity: int = 8,
        quantum_layer_quantity: int = 2,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        readout_wires: str | Sequence[int] = SHARED_READOUT,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
        device_name: str = device_utils.DEFAULT_DEVICE,
        diff_method: str = "best",
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        wire_quantity: The qubits; the 44 angles are uploaded in
            ceil(44 / wire_quantity) rounds.
        quantum_layer_quantity: The layers of each block.

        Raises
        ------
        ValueError if an option is invalid or the inputs are not nucleotides.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        self._require_bases()
        require_positive("wire quantity", wire_quantity)
        self.ansatz = ansatz
        self.wire_quantity = wire_quantity
        self.upload_quantity: int = ceil(STATISTIC_QUANTITY / wire_quantity)
        self.normalization = nn.BatchNorm1d(STATISTIC_QUANTITY, affine=False)
        self.circuit_weights = nn.Parameter(
            circuit_ansatz.initial_weights(
                (
                    self.upload_quantity,
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
        statistics: Tensor = self.normalization(sequence_statistics(channels))
        statistics = functional.pad(
            statistics,
            (0, self.upload_quantity * self.wire_quantity - STATISTIC_QUANTITY),
        )
        angles: Tensor = (pi / 2) * torch.tanh(statistics).reshape(
            -1, self.upload_quantity, self.wire_quantity
        )
        expectations: Tensor = torch.stack(
            list(self.qnode(angles, self.circuit_weights)), -1
        )[:, self.readout_indices]
        return wire_probabilities(
            expectations.to(statistics.dtype),
            self.shots,
            is_training=self.training,
        )


class QuantumAutomaton(SequenceModel):
    """
    A register starts in |0...0>, and for each position of the central `width`
    positions, left to right, the trained block of that position's symbol is
    applied; unknown or padding positions apply the identity. The prediction
    is P(wire = 1) at the end.
    """

    readout_matrix: Tensor

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        width: int = 201,
        wire_quantity: int = 6,
        quantum_layer_quantity: int = 2,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        readout_wires: str | Sequence[int] = SHARED_READOUT,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        width: The central positions read.
        wire_quantity: The qubits of the register.
        quantum_layer_quantity: The layers of each symbol's block.

        Raises
        ------
        ValueError if an option is invalid.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        require_positive("width", width)
        require_positive("wire quantity", wire_quantity)
        if width > sequence_length:
            raise ValueError(
                f"The width {width} exceeds the sequence length "
                f"{sequence_length}"
            )
        if wire_quantity > circuit_ansatz.UNITARY_WIRE_LIMIT:
            raise ValueError(
                f"Automata of more than {circuit_ansatz.UNITARY_WIRE_LIMIT} "
                "wires are not supported"
            )
        self.width = width
        self.wire_quantity = wire_quantity
        self.ansatz = ansatz
        self.circuit_weights = nn.Parameter(
            circuit_ansatz.initial_weights(
                (
                    self.channel_quantity,
                    *circuit_ansatz.weight_shape(
                        ansatz, quantum_layer_quantity, wire_quantity
                    ),
                ),
                initialization,
                initialization_scale,
            )
        )
        wires: list[int] = resolve_readout_wires(
            readout_wires, output_quantity, wire_quantity
        )
        self.register_buffer(
            "readout_matrix",
            torch.tensor(measurement.wire_bits(wire_quantity, wires)),
            persistent=False,
        )
        self.shots: int = 0

    def _probabilities(self, channels: Tensor) -> Tensor:
        """
        Raises
        ------
        ValueError if a position holds more than one symbol.
        """
        start: int = self.sequence_length // 2 - self.width // 2
        window: Tensor = channels[:, :, start : start + self.width]
        activity: Tensor = window.sum(1)
        if (activity > 1).any():
            raise ValueError(
                "The automaton reads one symbol per position, but inputs hold "
                "several"
            )
        # Positions without a symbol select the identity, appended last.
        symbols: Tensor = torch.where(
            activity > 0,
            window.argmax(1),
            torch.full_like(activity, self.channel_quantity, dtype=torch.long),
        )
        unitaries: Tensor = circuit_ansatz.ansatz_unitary(
            self.circuit_weights, self.ansatz
        )
        outcome_quantity: int = 2**self.wire_quantity
        if not torch.is_complex(unitaries):
            unitaries = unitaries + 0j
        unitaries = torch.cat(
            [
                unitaries,
                torch.eye(
                    outcome_quantity,
                    dtype=unitaries.dtype,
                    device=unitaries.device,
                ).unsqueeze(0),
            ]
        )
        states: Tensor = torch.zeros(
            len(channels),
            outcome_quantity,
            1,
            dtype=unitaries.dtype,
            device=channels.device,
        )
        states[:, 0] = 1
        for position in range(self.width):
            states = torch.bmm(unitaries[symbols[:, position]], states)
        probabilities: Tensor = states[..., 0].abs() ** 2
        if self.shots and not self.training:
            probabilities = measurement.sample_probabilities(
                probabilities, self.shots
            )
        return clamp_probabilities(
            probabilities.to(self.readout_matrix.dtype) @ self.readout_matrix
        )
