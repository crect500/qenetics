"""
Classical layers around the circuits: DeepCpG's convolutional body and tail,
pooling heads, dense layers, and the quantum second layer of Series I's FQ3.

DeepCpG's DNA model (CnnL2h128) is conv 128 x 11 + ReLU, max pool 4,
conv 256 x 3 + ReLU, max pool 2, a dense layer of 128 units with dropout 0.2,
and the output. Its "tail" is everything after the first convolution.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from qenetics.circuits import ansatz as circuit_ansatz
from qenetics.qcpg.filter_banks import WINDOW_NORMALIZATION, QuantumFilterBank
from qenetics.qcpg.sequence_model import require_choice, require_positive

DEEPCPG_FIRST_CHANNELS: int = 128
DEEPCPG_FIRST_WIDTH: int = 11
DEEPCPG_FIRST_POOL: int = 4
DEEPCPG_SECOND_CHANNELS: int = 256
DEEPCPG_SECOND_WIDTH: int = 3
DEEPCPG_SECOND_POOL: int = 2
DEEPCPG_HIDDEN_LAYER_SIZES: tuple[int, ...] = (128,)
DEEPCPG_DROPOUT: float = 0.2
MAX_POOL: str = "max"
AVERAGE_POOL: str = "average"
FIRST_POOLS: tuple[str, ...] = (MAX_POOL, AVERAGE_POOL)
MAX_MEAN_POOLING: str = "max_mean"
MEAN_POOLING: str = "mean"
POOLINGS: tuple[str, ...] = (MAX_MEAN_POOLING, MEAN_POOLING)


def _check_dropout(dropout: float) -> None:
    if not 0 <= dropout < 1:
        raise ValueError(f"Dropout must be in [0, 1), not {dropout}")


def multilayer_perceptron(
    input_size: int,
    hidden_layer_sizes: Sequence[int],
    output_size: int,
    *,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Dense layers with ReLU and dropout between them, and a linear output.

    Raises
    ------
    ValueError if a size is less than 1 or dropout is not in [0, 1).
    """
    _check_dropout(dropout)
    layers: list[nn.Module] = []
    size: int = input_size
    for hidden_size in (*hidden_layer_sizes, output_size):
        require_positive("layer size", hidden_size)
    for hidden_size in hidden_layer_sizes:
        layers.extend([nn.Linear(size, hidden_size), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        size = hidden_size
    layers.append(nn.Linear(size, output_size))
    return nn.Sequential(*layers)


class ChannelAffineReLU(nn.Module):
    """A trained scale and offset per channel, then ReLU."""

    def __init__(self, channel_quantity: int) -> None:
        super().__init__()
        self.affine = nn.Conv1d(
            channel_quantity, channel_quantity, 1, groups=channel_quantity
        )

    def forward(self, channels: Tensor) -> Tensor:
        return torch.relu(self.affine(channels))


def _flattened_quantity(
    channel_quantity: int, position_quantity: int, error_context: str
) -> int:
    if position_quantity < 1:
        raise ValueError(
            f"{error_context} leaves no positions; use longer sequences"
        )
    return channel_quantity * position_quantity


def deepcpg_body(
    channel_quantity: int, sequence_length: int
) -> tuple[nn.Sequential, int]:
    """
    DeepCpG's convolutional layers, from the input to the flattened features.

    Returns
    -------
    The layers and the number of flattened features.

    Raises
    ------
    ValueError if the sequences are too short for the layers.
    """
    positions: int = (
        (sequence_length - DEEPCPG_FIRST_WIDTH + 1) // DEEPCPG_FIRST_POOL
        - DEEPCPG_SECOND_WIDTH
        + 1
    ) // DEEPCPG_SECOND_POOL
    flattened: int = _flattened_quantity(
        DEEPCPG_SECOND_CHANNELS,
        positions,
        f"DeepCpG's convolutions on {sequence_length} positions",
    )
    return (
        nn.Sequential(
            nn.Conv1d(
                channel_quantity, DEEPCPG_FIRST_CHANNELS, DEEPCPG_FIRST_WIDTH
            ),
            nn.ReLU(),
            nn.MaxPool1d(DEEPCPG_FIRST_POOL),
            nn.Conv1d(
                DEEPCPG_FIRST_CHANNELS,
                DEEPCPG_SECOND_CHANNELS,
                DEEPCPG_SECOND_WIDTH,
            ),
            nn.ReLU(),
            nn.MaxPool1d(DEEPCPG_SECOND_POOL),
            nn.Flatten(),
        ),
        flattened,
    )


class DeepCpGTail(nn.Module):
    """DeepCpG's layers after its first convolution, giving logits."""

    def __init__(
        self,
        channel_quantity: int,
        position_quantity: int,
        output_quantity: int,
        *,
        hidden_layer_sizes: Sequence[int] = DEEPCPG_HIDDEN_LAYER_SIZES,
        dropout: float = DEEPCPG_DROPOUT,
        first_pool: str = MAX_POOL,
    ) -> None:
        """
        Args
        ----
        channel_quantity: The channels of the front end.
        position_quantity: The positions of the front end.
        output_quantity: The number of outputs.
        hidden_layer_sizes: The dense layers' sizes.
        dropout: The dropout after each dense layer.
        first_pool: `MAX_POOL`, as DeepCpG, or `AVERAGE_POOL`.

        Raises
        ------
        ValueError if an option is invalid or there are too few positions.
        """
        super().__init__()
        require_choice("first pool", first_pool, FIRST_POOLS)
        positions: int = (
            position_quantity // DEEPCPG_FIRST_POOL - DEEPCPG_SECOND_WIDTH + 1
        ) // DEEPCPG_SECOND_POOL
        flattened: int = _flattened_quantity(
            DEEPCPG_SECOND_CHANNELS,
            positions,
            f"DeepCpG's tail on {position_quantity} positions",
        )
        pool: nn.Module = (
            nn.MaxPool1d(DEEPCPG_FIRST_POOL)
            if first_pool == MAX_POOL
            else nn.AvgPool1d(DEEPCPG_FIRST_POOL)
        )
        self.body = nn.Sequential(
            pool,
            nn.Conv1d(
                channel_quantity, DEEPCPG_SECOND_CHANNELS, DEEPCPG_SECOND_WIDTH
            ),
            nn.ReLU(),
            nn.MaxPool1d(DEEPCPG_SECOND_POOL),
            nn.Flatten(),
        )
        self.head = multilayer_perceptron(
            flattened, hidden_layer_sizes, output_quantity, dropout=dropout
        )

    def forward(self, channels: Tensor) -> Tensor:
        return self.head(self.body(channels))


class PoolingHead(nn.Module):
    """
    Every channel pooled over positions, then dense layers, giving logits:
    the report's "global pooling + linear output".
    """

    def __init__(
        self,
        channel_quantity: int,
        output_quantity: int,
        *,
        pooling: str = MAX_MEAN_POOLING,
        channel_mix: bool = False,
        hidden_layer_sizes: Sequence[int] = (),
        dropout: float = 0.0,
    ) -> None:
        """
        Args
        ----
        channel_quantity: The channels of the front end.
        output_quantity: The number of outputs.
        pooling: `MAX_MEAN_POOLING` for the maximum and mean of each
            channel, or `MEAN_POOLING` for the mean only.
        channel_mix: Mix the channels with a trained 1 x 1 convolution
            before pooling.
        hidden_layer_sizes: The dense layers' sizes.
        dropout: The dropout after each dense layer.

        Raises
        ------
        ValueError if an option is invalid.
        """
        super().__init__()
        require_choice("pooling", pooling, POOLINGS)
        self.pooling = pooling
        self.mix: nn.Module = (
            nn.Conv1d(channel_quantity, channel_quantity, 1)
            if channel_mix
            else nn.Identity()
        )
        pooled_quantity: int = channel_quantity * (
            2 if pooling == MAX_MEAN_POOLING else 1
        )
        self.head = multilayer_perceptron(
            pooled_quantity,
            hidden_layer_sizes,
            output_quantity,
            dropout=dropout,
        )

    def forward(self, channels: Tensor) -> Tensor:
        channels = self.mix(channels)
        if self.pooling == MAX_MEAN_POOLING:
            pooled: Tensor = torch.cat([channels.amax(2), channels.mean(2)], 1)
        else:
            pooled = channels.mean(2)
        return self.head(pooled)


class QuantumFilterBankTail(nn.Module):
    """
    Max pool 4, a second quantum filter bank on windows of every channel
    at `window_length` pooled positions, each normalized by its own norm,
    max pool 2, dense layers, giving logits.
    """

    def __init__(
        self,
        channel_quantity: int,
        position_quantity: int,
        output_quantity: int,
        *,
        window_length: int = 3,
        quantum_layer_quantity: int = 4,
        ansatz: str = circuit_ansatz.STRONG_ANSATZ_STR,
        hidden_layer_sizes: Sequence[int] = DEEPCPG_HIDDEN_LAYER_SIZES,
        dropout: float = DEEPCPG_DROPOUT,
        initialization: str = circuit_ansatz.NORMAL_INITIALIZATION_STR,
        initialization_scale: float = 0.05,
    ) -> None:
        """
        Raises
        ------
        ValueError if an option is invalid or there are too few positions.
        """
        super().__init__()
        self.pool = nn.MaxPool1d(DEEPCPG_FIRST_POOL)
        self.bank = QuantumFilterBank(
            channel_quantity,
            window_length,
            quantum_layer_quantity=quantum_layer_quantity,
            ansatz=ansatz,
            normalization=WINDOW_NORMALIZATION,
            initialization=initialization,
            initialization_scale=initialization_scale,
        )
        positions: int = (
            self.bank.position_quantity(position_quantity // DEEPCPG_FIRST_POOL)
            // DEEPCPG_SECOND_POOL
        )
        flattened: int = _flattened_quantity(
            self.bank.output_channel_quantity,
            positions,
            f"The quantum tail on {position_quantity} positions",
        )
        self.head = nn.Sequential(
            nn.MaxPool1d(DEEPCPG_SECOND_POOL),
            nn.Flatten(),
            multilayer_perceptron(
                flattened, hidden_layer_sizes, output_quantity, dropout=dropout
            ),
        )

    def forward(self, channels: Tensor) -> Tensor:
        return self.head(self.bank(self.pool(channels)))
