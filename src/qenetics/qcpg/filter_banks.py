"""
Quantum filter banks: a trained circuit measured on every window of a sequence.

Each window of w positions of an input of A channels is amplitude-encoded into
m = ceil(log2(A w)) qubits, at index A * position + channel. A trained circuit
U is applied, and the window's features are its measurement probabilities
|U x|^2, or <Z> of each wire, or the marginals of its first wires. As the
encoding is linear, the amplitudes U x of all windows are a convolution of the
input with the rows of U. U is therefore built once per batch with
`ansatz.ansatz_unitary` and applied by `conv1d`, which is exact and as fast as
a classical convolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, log2, sqrt
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from qenetics.circuits import ansatz as circuit_ansatz
from qenetics.circuits import measurement
from qenetics.qcpg.sequence_model import require_choice, require_positive

FIXED_NORMALIZATION: str = "fixed"
WINDOW_NORMALIZATION: str = "window"
NORMALIZATIONS: tuple[str, ...] = (FIXED_NORMALIZATION, WINDOW_NORMALIZATION)
# Exact unitaries are built for at most this many qubits.
MAXIMUM_WIRE_QUANTITY: int = 12
# Banks of more channels evaluate this many sequences at a time, recomputing
# their amplitudes in the backward pass instead of storing them.
CHUNK_CHANNEL_QUANTITY: int = 128
CHUNK_SEQUENCE_QUANTITY: int = 64


class QuantumFilterBank(nn.Module):
    """
    One or more independent circuits applied to every window of a sequence.

    Input (batch, channels, length); output (batch, `output_channel_quantity`,
    length - window_length + 1).
    """

    readout_matrix: Tensor | None

    def __init__(
        self,
        channel_quantity: int,
        window_length: int,
        *,
        quantum_layer_quantity: int = 4,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        circuit_quantity: int = 1,
        readout: str = measurement.PROBABILITY_READOUT,
        marginal_wire_quantity: int = 1,
        normalization: str = FIXED_NORMALIZATION,
        constant_amplitude: float = 0.0,
        trainable_circuit: bool = True,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
    ) -> None:
        """
        Args
        ----
        channel_quantity: The input channels A, e.g. 4 nucleotides or the
            vocabulary of one-hot tokens.
        window_length: The positions w of each window.
        quantum_layer_quantity: The layers of each circuit.
        ansatz: The ansatz of each circuit, see `ansatz.ANSATZES_STRINGS`.
        circuit_quantity: The number of independent circuits.
        readout: `PROBABILITY_READOUT` for all 2^m probabilities (times 2^m),
            `EXPECTATION_READOUT` for <Z> of each wire, or `MARGINAL_READOUT`
            for the probabilities of the first `marginal_wire_quantity` wires
            (times their 2^r outcomes).
        marginal_wire_quantity: The wires of the marginal readout.
        normalization: `FIXED_NORMALIZATION` divides each window by
            sqrt(w), the norm of a one-hot window, so windows with unknown
            nucleotides are encoded with less than unit norm, as in the
            report. `WINDOW_NORMALIZATION` divides each window by its own
            norm, for inputs that are not one-hot.
        constant_amplitude: If positive, the spare amplitudes of each window
            hold a constant, so each probability gains a signed linear term:
            |u.x + c|^2. The value is the block's norm relative to a one-hot
            window's.
        trainable_circuit: Train the circuits' angles, or keep them fixed.
        initialization: `ansatz.NORMAL_INITIALIZATION_STR` or
            `ansatz.UNIFORM_INITIALIZATION_STR`.
        initialization_scale: The standard deviation of normal angles.

        Raises
        ------
        ValueError if an option is unknown or out of range, or the windows
        need more than `MAXIMUM_WIRE_QUANTITY` qubits.
        """
        super().__init__()
        require_positive("channel quantity", channel_quantity)
        require_positive("window length", window_length)
        require_positive("quantum layer quantity", quantum_layer_quantity)
        require_positive("circuit quantity", circuit_quantity)
        require_choice("readout", readout, measurement.READOUTS)
        require_choice("normalization", normalization, NORMALIZATIONS)
        if constant_amplitude < 0:
            raise ValueError(
                "The constant amplitude must not be negative, not "
                f"{constant_amplitude}"
            )
        if constant_amplitude and normalization != FIXED_NORMALIZATION:
            raise ValueError(
                "A constant amplitude block needs fixed normalization"
            )

        amplitude_quantity: int = channel_quantity * window_length
        wire_quantity: int = max(
            1, ceil(log2(amplitude_quantity + (constant_amplitude > 0)))
        )
        if wire_quantity > MAXIMUM_WIRE_QUANTITY:
            raise ValueError(
                f"Windows of {amplitude_quantity} amplitudes need "
                f"{wire_quantity} qubits, more than the "
                f"{MAXIMUM_WIRE_QUANTITY} supported; use shorter windows or a "
                "smaller vocabulary"
            )

        self.channel_quantity = channel_quantity
        self.window_length = window_length
        self.ansatz = ansatz
        self.circuit_quantity = circuit_quantity
        self.readout = readout
        self.normalization = normalization
        self.wire_quantity = wire_quantity
        constant_quantity: int = 2**wire_quantity - amplitude_quantity
        self.constant: float = (
            constant_amplitude * sqrt(window_length / constant_quantity)
            if constant_amplitude
            else 0.0
        )
        self.squared_norm: float = (
            window_length + self.constant**2 * constant_quantity
        )
        self.circuit_weights = nn.Parameter(
            circuit_ansatz.initial_weights(
                (
                    circuit_quantity,
                    *circuit_ansatz.weight_shape(
                        ansatz, quantum_layer_quantity, wire_quantity
                    ),
                ),
                initialization,
                initialization_scale,
            ),
            requires_grad=trainable_circuit,
        )

        outcome_quantity: int = 2**wire_quantity
        if readout == measurement.PROBABILITY_READOUT:
            readout_matrix: Tensor | None = None
            channels_per_circuit: int = outcome_quantity
        elif readout == measurement.EXPECTATION_READOUT:
            readout_matrix = torch.tensor(
                measurement.expectation_matrix(wire_quantity)
            )
            channels_per_circuit = wire_quantity
        else:
            readout_matrix = torch.tensor(
                measurement.marginal_matrix(
                    wire_quantity, marginal_wire_quantity
                )
                * 2**marginal_wire_quantity
            )
            channels_per_circuit = 2**marginal_wire_quantity
        self.register_buffer("readout_matrix", readout_matrix, persistent=False)
        self.output_channel_quantity: int = (
            circuit_quantity * channels_per_circuit
        )
        # Measurements per window in evaluation; 0 for exact probabilities.
        self.shots: int = 0

    def position_quantity(self, sequence_length: int) -> int:
        """
        Raises
        ------
        ValueError if the sequence is shorter than a window.
        """
        if sequence_length < self.window_length:
            raise ValueError(
                f"Windows of {self.window_length} positions do not fit in a "
                f"sequence of {sequence_length}"
            )
        return sequence_length - self.window_length + 1

    def _filters(self) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
        """
        Real and imaginary convolution filters and biases from the circuits'
        unitaries. The imaginary parts are None for real circuits.
        """
        unitaries: Tensor = circuit_ansatz.ansatz_unitary(
            self.circuit_weights, self.ansatz
        )
        outcome_quantity: int = 2**self.wire_quantity
        amplitude_quantity: int = self.channel_quantity * self.window_length
        scale: float = (
            1 / sqrt(self.squared_norm)
            if self.normalization == FIXED_NORMALIZATION
            else 1.0
        )
        filters: Tensor = (
            unitaries[..., :amplitude_quantity]
            .reshape(
                self.circuit_quantity * outcome_quantity,
                self.window_length,
                self.channel_quantity,
            )
            # Amplitude A * position + channel holds input (channel, position).
            .transpose(1, 2)
            * scale
        )
        biases: Tensor = (
            self.constant
            * scale
            * unitaries[..., amplitude_quantity:].sum(-1).reshape(-1)
        )
        if not torch.is_complex(unitaries):
            return filters.contiguous(), None, biases, None

        return (
            filters.real.contiguous(),
            filters.imag.contiguous(),
            biases.real,
            biases.imag,
        )

    def _features(
        self,
        channels: Tensor,
        real_filters: Tensor,
        imaginary_filters: Tensor | None,
        real_biases: Tensor,
        imaginary_biases: Tensor | None,
    ) -> Tensor:
        real_parts: Tensor = functional.conv1d(
            channels, real_filters, real_biases if self.constant else None
        )
        probabilities: Tensor = real_parts * real_parts
        if imaginary_filters is not None:
            imaginary_parts: Tensor = functional.conv1d(
                channels,
                imaginary_filters,
                imaginary_biases if self.constant else None,
            )
            probabilities = probabilities + imaginary_parts * imaginary_parts
        if self.normalization == WINDOW_NORMALIZATION:
            squared_norms: Tensor = functional.conv1d(
                channels * channels,
                torch.ones_like(real_filters[:1]),
            )
            probabilities = probabilities / squared_norms.clamp_min(1e-12)

        batch_size, _, position_quantity = probabilities.shape
        outcome_quantity: int = 2**self.wire_quantity
        probabilities = probabilities.reshape(
            batch_size, self.circuit_quantity, outcome_quantity, -1
        )
        if self.shots and not self.training:
            probabilities = measurement.sample_probabilities(
                probabilities.transpose(2, 3), self.shots
            ).transpose(2, 3)
        if self.readout_matrix is None:
            features: Tensor = probabilities * outcome_quantity
        else:
            features = torch.einsum(
                "bcjp,jk->bckp",
                probabilities,
                self.readout_matrix.to(probabilities.dtype),
            )
        return features.reshape(batch_size, -1, position_quantity)

    def forward(self, channels: Tensor) -> Tensor:
        """
        Raises
        ------
        ValueError if the input does not have this bank's channels or is
        shorter than a window.
        """
        if channels.ndim != 3 or channels.shape[1] != self.channel_quantity:
            raise ValueError(
                f"Filter bank inputs must have shape (batch, "
                f"{self.channel_quantity}, length), not "
                f"{tuple(channels.shape)}"
            )
        self.position_quantity(channels.shape[2])
        filters = self._filters()
        if (
            self.circuit_quantity * 2**self.wire_quantity
            <= CHUNK_CHANNEL_QUANTITY
            or len(channels) <= CHUNK_SEQUENCE_QUANTITY
        ):
            return self._features(channels, *filters)

        chunks: list[Tensor] = []
        for start in range(0, len(channels), CHUNK_SEQUENCE_QUANTITY):
            chunk: Tensor = channels[start : start + CHUNK_SEQUENCE_QUANTITY]
            if torch.is_grad_enabled():
                chunks.append(
                    checkpoint(
                        self._features, chunk, *filters, use_reentrant=False
                    )
                )
            else:
                chunks.append(self._features(chunk, *filters))
        return torch.cat(chunks)


class MultiScaleFilterBank(nn.Module):
    """
    Quantum filter banks on windows of several lengths, whose outputs are
    cropped to the positions they share, aligned on their centres.
    """

    def __init__(
        self,
        channel_quantity: int,
        window_lengths: Sequence[int],
        **bank_options: Any,
    ) -> None:
        """
        Args
        ----
        channel_quantity: The input channels.
        window_lengths: The window length of each bank.
        bank_options: The options of every `QuantumFilterBank`.

        Raises
        ------
        ValueError if no window lengths are given or a bank is invalid.
        """
        super().__init__()
        if not window_lengths:
            raise ValueError("At least one window length is needed")

        banks: list[QuantumFilterBank] = [
            QuantumFilterBank(channel_quantity, window_length, **bank_options)
            for window_length in window_lengths
        ]
        self.banks = nn.ModuleList(banks)
        self.output_channel_quantity: int = sum(
            bank.output_channel_quantity for bank in banks
        )

    def position_quantity(self, sequence_length: int) -> int:
        return min(
            cast(QuantumFilterBank, bank).position_quantity(sequence_length)
            for bank in self.banks
        )

    def forward(self, channels: Tensor) -> Tensor:
        outputs: list[Tensor] = [bank(channels) for bank in self.banks]
        position_quantity: int = min(output.shape[2] for output in outputs)
        return torch.cat(
            [
                output[
                    :,
                    :,
                    (output.shape[2] - position_quantity) // 2 : (
                        output.shape[2] - position_quantity
                    )
                    // 2
                    + position_quantity,
                ]
                for output in outputs
            ],
            1,
        )
