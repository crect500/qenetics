"""
Shared parts of the quantum and hybrid methylation models.

Every model takes one-hot inputs, (batch, length, channels) floats, or token
inputs, (batch, length) ids, which a lookup table maps to channels. Inside, a
model sees (batch, channels, length) and predicts one methylation probability
per output, as `qcpg_models.QNN` does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import cast

import pennylane as qp
import torch
from torch import Tensor, nn

from qenetics.circuits import device_utils
from qenetics.circuits.measurement import (
    check_wires,
    sample_bernoulli_frequencies,
)
from qenetics.tools import converters
from qenetics.tools.dna import Nucleotide

ONEHOT_ENCODING: str = "onehot"
TOKEN_ENCODING: str = "token"
ENCODINGS: tuple[str, ...] = (ONEHOT_ENCODING, TOKEN_ENCODING)
SHARED_READOUT: str = "shared"
PER_OUTPUT_READOUT: str = "per_output"
# Probabilities are kept this far from 0 and 1, as the logits of the report.
PROBABILITY_EPSILON: float = 1e-6
# The 44 sequence statistics: dinucleotide frequencies within each of
# `DINUCLEOTIDE_RADII` of the site, then CpG and GC densities within each of
# `DENSITY_RADII`.
DINUCLEOTIDE_RADII: tuple[int, ...] = (50, 500)
DENSITY_RADII: tuple[int, ...] = (10, 25, 50, 100, 250, 500)
STATISTIC_QUANTITY: int = 16 * len(DINUCLEOTIDE_RADII) + 2 * len(DENSITY_RADII)


def require_positive(name: str, value: int) -> None:
    """
    Raises
    ------
    ValueError if the value is less than 1.
    """
    if value < 1:
        raise ValueError(f"The {name} must be at least 1, not {value}")


def require_choice(name: str, value: str, choices: Sequence[str]) -> None:
    """
    Raises
    ------
    ValueError if the value is not one of the choices.
    """
    if value not in choices:
        raise ValueError(
            f"Unknown {name} {value}; expected one of {', '.join(choices)}"
        )


def one_hot_lookup(
    vocabulary_size: int, padding_token: int | None = None
) -> Tensor:
    """
    Map token ids to one-hot channels.

    Args
    ----
    vocabulary_size: The number of token ids.
    padding_token: An id that encodes as all zeros, such as [PAD].

    Returns
    -------
    The (vocabulary_size, vocabulary_size) lookup table.

    Raises
    ------
    ValueError if the vocabulary is empty or the padding token is not in it.
    """
    require_positive("vocabulary size", vocabulary_size)
    lookup: Tensor = torch.eye(vocabulary_size)
    if padding_token is not None:
        if not 0 <= padding_token < vocabulary_size:
            raise ValueError(
                f"Padding token {padding_token} is not in a vocabulary of "
                f"{vocabulary_size} tokens"
            )
        lookup[padding_token] = 0.0
    return lookup


class InputEncoder(nn.Module):
    """Encode one-hot or token inputs as (batch, channels, length) floats."""

    lookup: Tensor | None

    def __init__(
        self,
        sequence_length: int,
        encoding: str = ONEHOT_ENCODING,
        *,
        channel_quantity: int = converters.UNIQUE_NUCLEOTIDE_QUANTITY,
        vocabulary_size: int | None = None,
        padding_token: int | None = None,
        token_lookup: Tensor | None = None,
    ) -> None:
        """
        Args
        ----
        sequence_length: The number of positions of every input.
        encoding: `ONEHOT_ENCODING` or `TOKEN_ENCODING`.
        channel_quantity: The channels of one-hot inputs.
        vocabulary_size: The number of token ids, if no lookup is given.
        padding_token: A token id that encodes as all zeros.
        token_lookup: The channels of each token id, (vocabulary, channels).
            Defaults to `one_hot_lookup`.

        Raises
        ------
        ValueError if the encoding is unknown or its settings are invalid.
        """
        super().__init__()
        require_positive("sequence length", sequence_length)
        require_choice("encoding", encoding, ENCODINGS)
        self.sequence_length = sequence_length
        self.encoding = encoding
        if encoding == ONEHOT_ENCODING:
            require_positive("channel quantity", channel_quantity)
            self.channel_quantity = channel_quantity
            self.lookup = None
            return

        if token_lookup is None:
            if vocabulary_size is None:
                raise ValueError(
                    "Token encoding needs a vocabulary size or a token lookup"
                )
            token_lookup = one_hot_lookup(vocabulary_size, padding_token)
        token_lookup = torch.as_tensor(token_lookup, dtype=torch.float)
        if token_lookup.ndim != 2 or 0 in token_lookup.shape:
            raise ValueError(
                "A token lookup must be a non-empty (vocabulary, channels) "
                f"table, not of shape {tuple(token_lookup.shape)}"
            )
        if vocabulary_size is not None and vocabulary_size != len(token_lookup):
            raise ValueError(
                f"A vocabulary of {vocabulary_size} tokens does not match a "
                f"lookup of {len(token_lookup)} tokens"
            )
        self.channel_quantity = token_lookup.shape[1]
        self.register_buffer("lookup", token_lookup, persistent=False)

    def forward(self, inputs: Tensor) -> tuple[Tensor, bool]:
        """
        Returns
        -------
        The channels, (batch, channels, length), and whether the inputs were
        batched.

        Raises
        ------
        ValueError if the inputs have the wrong shape, or tokens are not in
        the vocabulary.
        """
        if self.encoding == ONEHOT_ENCODING:
            if inputs.ndim not in (2, 3) or inputs.shape[-2:] != (
                self.sequence_length,
                self.channel_quantity,
            ):
                raise ValueError(
                    "One-hot inputs must have shape (batch, "
                    f"{self.sequence_length}, {self.channel_quantity}) or "
                    f"({self.sequence_length}, {self.channel_quantity}), not "
                    f"{tuple(inputs.shape)}"
                )
            is_batched: bool = inputs.ndim == 3
            channels: Tensor = inputs if is_batched else inputs.unsqueeze(0)
            if not channels.is_floating_point():
                channels = channels.float()
            return channels.transpose(1, 2), is_batched

        if (
            inputs.ndim not in (1, 2)
            or inputs.shape[-1] != self.sequence_length
        ):
            raise ValueError(
                f"Token inputs must have shape (batch, {self.sequence_length}) "
                f"or ({self.sequence_length},), not {tuple(inputs.shape)}"
            )
        is_batched = inputs.ndim == 2
        lookup: Tensor = cast(Tensor, self.lookup)
        tokens: Tensor = (inputs if is_batched else inputs.unsqueeze(0)).long()
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= len(lookup)):
            raise ValueError(
                f"Token ids must be between 0 and {len(lookup) - 1}, but "
                f"range from {tokens.min().item()} to {tokens.max().item()}"
            )
        return lookup[tokens].transpose(1, 2), is_batched


class SequenceModel(nn.Module, ABC):
    """
    Base of the methylation models.

    Subclasses implement `_probabilities`, from channels to one methylation
    probability per output. Parameters of circuits are named
    `circuit_weights`, which `model_factory.count_parameters` relies on.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        encoding: str = ONEHOT_ENCODING,
        channel_quantity: int = converters.UNIQUE_NUCLEOTIDE_QUANTITY,
        vocabulary_size: int | None = None,
        padding_token: int | None = None,
        token_lookup: Tensor | None = None,
    ) -> None:
        """
        Raises
        ------
        ValueError if a quantity is less than 1 or the encoding is invalid.
        """
        super().__init__()
        require_positive("output quantity", output_quantity)
        self.encoder = InputEncoder(
            sequence_length,
            encoding,
            channel_quantity=channel_quantity,
            vocabulary_size=vocabulary_size,
            padding_token=padding_token,
            token_lookup=token_lookup,
        )
        self.sequence_length = sequence_length
        self.output_quantity = output_quantity

    @property
    def channel_quantity(self) -> int:
        return self.encoder.channel_quantity

    def forward(self, inputs: Tensor) -> Tensor:
        channels, is_batched = self.encoder(inputs)
        probabilities: Tensor = self._probabilities(channels)
        return probabilities if is_batched else probabilities[0]

    @abstractmethod
    def _probabilities(self, channels: Tensor) -> Tensor:
        """(batch, channels, length) -> (batch, outputs) probabilities."""

    def _require_bases(self) -> None:
        """
        Raises
        ------
        ValueError if the inputs are not four nucleotide channels.
        """
        if self.channel_quantity != converters.UNIQUE_NUCLEOTIDE_QUANTITY:
            raise ValueError(
                f"{type(self).__name__} needs one-hot nucleotides (4 "
                f"channels), not {self.channel_quantity} channels"
            )


def clamp_probabilities(probabilities: Tensor) -> Tensor:
    return probabilities.clamp(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)


def resolve_readout_wires(
    readout_wires: str | Sequence[int],
    output_quantity: int,
    wire_quantity: int,
) -> list[int]:
    """
    Choose the wire whose P(wire = 1) predicts each output.

    Args
    ----
    readout_wires: `SHARED_READOUT` for wire 0 for every output,
        `PER_OUTPUT_READOUT` for wire c mod n for output c, or a list of one
        wire for every output or one per output.
    output_quantity: The number of outputs.
    wire_quantity: The number of wires.

    Returns
    -------
    The wire of each output.

    Raises
    ------
    ValueError if the readout is unknown or a wire is not in the register.
    """
    if isinstance(readout_wires, str):
        require_choice(
            "readout", readout_wires, (SHARED_READOUT, PER_OUTPUT_READOUT)
        )
        if readout_wires == SHARED_READOUT:
            return [0] * output_quantity
        return [output % wire_quantity for output in range(output_quantity)]

    wires: list[int] = list(readout_wires)
    if len(wires) not in (1, output_quantity):
        raise ValueError(
            f"Give one readout wire or one per output ({output_quantity}), "
            f"not {len(wires)}"
        )
    check_wires(wires, wire_quantity)
    return wires * output_quantity if len(wires) == 1 else wires


def measured_wires(wires: Sequence[int]) -> tuple[list[int], list[int]]:
    """
    Find the distinct wires to measure for a list of readout wires.

    Returns
    -------
    The distinct wires, in order, and the index among them of each wire.
    """
    distinct: list[int] = sorted(set(wires))
    return distinct, [distinct.index(wire) for wire in wires]


def wire_probabilities(
    expectations: Tensor, shots: int = 0, *, is_training: bool = False
) -> Tensor:
    """
    Convert <Z> of wires to P(wire = 1), estimated from `shots` measurements
    outside training if shots is positive.
    """
    probabilities: Tensor = (1 - expectations) / 2
    if shots and not is_training:
        probabilities = sample_bernoulli_frequencies(probabilities, shots)
    return clamp_probabilities(probabilities)


def check_probability_differentiation(
    device_name: str, diff_method: str
) -> None:
    """
    Raises
    ------
    ValueError if a lightning device is asked to differentiate measured
    probabilities by the adjoint method, which it does not support.
    """
    if device_name.startswith("lightning") and diff_method == "adjoint":
        raise ValueError(
            f"{device_name} cannot differentiate probability readouts with "
            "the adjoint method; use diff_method 'best' or "
            "'parameter-shift', or default.qubit"
        )


def build_qnode(
    circuit: Callable,
    wire_quantity: int,
    device_name: str = device_utils.DEFAULT_DEVICE,
    diff_method: str = "best",
) -> qp.QNode:
    """
    Raises
    ------
    ValueError if the device name is not supported.
    """
    return qp.QNode(
        circuit,
        device_utils.device_setup(device_name, wire_quantity),
        interface="torch",
        diff_method=diff_method,
    )


def sequence_statistics(channels: Tensor) -> Tensor:
    """
    Compute the 44 sequence statistics of one-hot nucleotide windows.

    These are the 16 dinucleotide frequencies within 50 and 500 bp of the
    window's centre, and the CpG and GC densities within 10, 25, 50, 100, 250
    and 500 bp.

    Args
    ----
    channels: One-hot nucleotides, (batch, 4, length), in the order A, T, G,
        C, with the site at position length // 2.

    Returns
    -------
    The statistics, (batch, `STATISTIC_QUANTITY`).

    Raises
    ------
    ValueError if the channels are not four nucleotides of at least two
    positions.
    """
    if channels.ndim != 3 or channels.shape[1] != 4 or channels.shape[2] < 2:
        raise ValueError(
            "Sequence statistics need one-hot nucleotides of shape (batch, 4, "
            f"length >= 2), not {tuple(channels.shape)}"
        )

    nucleotides: Tensor = channels.transpose(1, 2)
    batch_size, length, _ = nucleotides.shape
    centre: int = length // 2
    dinucleotides: Tensor = torch.einsum(
        "bli,blj->blij", nucleotides[:, :-1], nucleotides[:, 1:]
    ).reshape(batch_size, length - 1, 16)
    statistics: list[Tensor] = [
        dinucleotides[:, max(centre - radius, 0) : centre + radius].mean(1)
        for radius in DINUCLEOTIDE_RADII
    ]
    gc: Tensor = (
        nucleotides[:, :, Nucleotide.G.value]
        + nucleotides[:, :, Nucleotide.C.value]
    )
    cpg: Tensor = dinucleotides[
        :, :, 4 * Nucleotide.C.value + Nucleotide.G.value
    ]
    for radius in DENSITY_RADII:
        window = slice(max(centre - radius, 0), centre + radius)
        statistics.append(cpg[:, window].mean(1, keepdim=True))
        statistics.append(gc[:, window].mean(1, keepdim=True))
    return torch.cat(statistics, 1)
