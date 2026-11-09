"""
Classical references for the quantum and hybrid methylation models: the
classical models of the quantum models report's matched comparisons, and the
ceilings of its circuit families. They are built like the circuits' models
from `SequenceModel`, so that they train, evaluate and record the same way.

`ClassicalNetwork` is a classical first layer (the front end), an optional
per-channel affine map and ReLU, and DeepCpG's tail or the pooling head, as
`FilterBankNetwork` is with a circuit. Its front ends:

- `CONVOLUTION_FRONT`: a convolution and ReLU; DeepCpG's CNN with 128 filters
  of 11 positions, the report's C_* models otherwise.
- `IDENTITY_FRONT`: the raw window as channels, the filter bank with every
  angle at 0 (ID_*).
- `LOW_RANK_FRONT`: a rank-r convolution, r filters mixed into 64 channels
  (LR_*).
- `GIVENS_FRONT`: |Q x|^2 of an orthogonal Q made of Givens rotations (G_*).
- `DENSE_FRONT`: |W x|^2 of a trained unitary, rotation, free complex or free
  real matrix W of the window (U64, O64, FC64, FR64 and their 64 bp versions).
- `MULTI_SCALE_FRONT`: convolutions of several widths, with ReLU or as
  quadrature-pair energies (MSC_*, Cq_*).
- `TWO_LAYER_FRONT`: DeepCpG's two convolutional layers, narrower (CNN2).

Whole models: `FeatureClassifier` on the 44 sequence statistics (FL_lin,
FM_4), `OneHotClassifier` on the flattened sequence (OH_*),
`TokenFrequencyClassifier` on the token frequencies (TF64_LR) and
`FactorizationMachine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, sqrt
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from qenetics.qcpg import heads
from qenetics.qcpg.hybrid_models import (
    ACTIVATIONS,
    AFFINE_RELU_ACTIVATION,
    DEEPCPG_TAIL,
    POOLING_TAIL,
)
from qenetics.qcpg.sequence_model import (
    STATISTIC_QUANTITY,
    SequenceModel,
    require_choice,
    require_positive,
    sequence_statistics,
)

CONVOLUTION_FRONT: str = "convolution"
IDENTITY_FRONT: str = "identity"
LOW_RANK_FRONT: str = "low_rank"
GIVENS_FRONT: str = "givens"
DENSE_FRONT: str = "dense"
MULTI_SCALE_FRONT: str = "multi_scale"
TWO_LAYER_FRONT: str = "two_layer"
FRONTS: tuple[str, ...] = (
    CONVOLUTION_FRONT,
    IDENTITY_FRONT,
    LOW_RANK_FRONT,
    GIVENS_FRONT,
    DENSE_FRONT,
    MULTI_SCALE_FRONT,
    TWO_LAYER_FRONT,
)
CLASSICAL_TAILS: tuple[str, ...] = (DEEPCPG_TAIL, POOLING_TAIL)
UNITARY_MATRIX: str = "unitary"
ORTHOGONAL_MATRIX: str = "orthogonal"
COMPLEX_MATRIX: str = "complex"
REAL_MATRIX: str = "real"
MATRICES: tuple[str, ...] = (
    UNITARY_MATRIX,
    ORTHOGONAL_MATRIX,
    COMPLEX_MATRIX,
    REAL_MATRIX,
)
RELU_MODE: str = "relu"
QUADRATURE_MODE: str = "quadrature"
MULTI_SCALE_MODES: tuple[str, ...] = (RELU_MODE, QUADRATURE_MODE)
LOW_RANK_CHANNEL_QUANTITY: int = 64
GIVENS_PAIRING_SEED: int = 1234


def _window_filters(
    matrix: Tensor, window_length: int, channels: int
) -> Tensor:
    """
    The convolution filters that apply a d x d matrix to every window, whose
    amplitude A * position + channel holds input (channel, position).
    """
    return matrix.reshape(len(matrix), window_length, channels).transpose(1, 2)


def _require_window(window_length: int, sequence_length: int) -> int:
    """
    Raises
    ------
    ValueError if the sequence is shorter than a window.
    """
    if sequence_length < window_length:
        raise ValueError(
            f"Windows of {window_length} positions do not fit in a sequence "
            f"of {sequence_length}"
        )
    return sequence_length - window_length + 1


class ConvolutionFront(nn.Module):
    """A convolution of `filter_quantity` filters of `window_length`, and ReLU."""

    def __init__(
        self, channel_quantity: int, window_length: int, filter_quantity: int
    ) -> None:
        super().__init__()
        require_positive("window length", window_length)
        require_positive("filter quantity", filter_quantity)
        self.window_length = window_length
        self.convolution = nn.Conv1d(
            channel_quantity, filter_quantity, window_length
        )
        self.output_channel_quantity: int = filter_quantity

    def position_quantity(self, sequence_length: int) -> int:
        return _require_window(self.window_length, sequence_length)

    def forward(self, channels: Tensor) -> Tensor:
        return torch.relu(self.convolution(channels))


class IdentityFront(nn.Module):
    """
    The raw window of A channels and w positions as A w channels, at index
    A * position + channel, times A: the filter bank with every angle at 0,
    with features of the same mean scale.
    """

    identity: Tensor

    def __init__(self, channel_quantity: int, window_length: int) -> None:
        super().__init__()
        require_positive("window length", window_length)
        dimension: int = channel_quantity * window_length
        self.channel_quantity = channel_quantity
        self.window_length = window_length
        self.register_buffer(
            "identity",
            _window_filters(
                torch.eye(dimension), window_length, channel_quantity
            ).contiguous(),
            persistent=False,
        )
        self.output_channel_quantity: int = dimension

    def position_quantity(self, sequence_length: int) -> int:
        return _require_window(self.window_length, sequence_length)

    def forward(self, channels: Tensor) -> Tensor:
        return (
            functional.conv1d(channels, self.identity.to(channels.dtype))
            * self.channel_quantity
        )


class LowRankFront(nn.Module):
    """r filters of `window_length`, mixed into 64 channels, and ReLU."""

    def __init__(
        self, channel_quantity: int, window_length: int, rank: int
    ) -> None:
        super().__init__()
        require_positive("window length", window_length)
        require_positive("rank", rank)
        self.window_length = window_length
        self.layers = nn.Sequential(
            nn.Conv1d(channel_quantity, rank, window_length, bias=False),
            nn.Conv1d(rank, LOW_RANK_CHANNEL_QUANTITY, 1),
            nn.ReLU(),
        )
        self.output_channel_quantity: int = LOW_RANK_CHANNEL_QUANTITY

    def position_quantity(self, sequence_length: int) -> int:
        return _require_window(self.window_length, sequence_length)

    def forward(self, channels: Tensor) -> Tensor:
        return self.layers(channels)


class _MatrixFront(nn.Module):
    """|W x|^2 times d of each window x, scaled by 1/sqrt(w), for a matrix W."""

    def __init__(self, channel_quantity: int, window_length: int) -> None:
        super().__init__()
        require_positive("window length", window_length)
        self.channel_quantity = channel_quantity
        self.window_length = window_length
        self.dimension: int = channel_quantity * window_length
        self.output_channel_quantity: int = self.dimension

    def position_quantity(self, sequence_length: int) -> int:
        return _require_window(self.window_length, sequence_length)

    def transform(self) -> Tensor:
        raise NotImplementedError

    def forward(self, channels: Tensor) -> Tensor:
        filters: Tensor = _window_filters(
            self.transform(), self.window_length, self.channel_quantity
        ) / sqrt(self.window_length)
        real_parts: Tensor = functional.conv1d(
            channels, filters.real.contiguous()
        )
        probabilities: Tensor = real_parts * real_parts
        if filters.is_complex():
            imaginary_parts: Tensor = functional.conv1d(
                channels, filters.imag.contiguous()
            )
            probabilities = probabilities + imaginary_parts * imaginary_parts
        return probabilities * self.dimension


class GivensFront(_MatrixFront):
    """
    An orthogonal Q of layers of d/2 disjoint Givens rotations on random
    pairings, `angle_quantity` of whose angles are trained (the others stay
    0). Its parameters are whole layers, as in the report.
    """

    pairings: Tensor
    mask: Tensor

    def __init__(
        self,
        channel_quantity: int,
        window_length: int,
        angle_quantity: int,
        *,
        initialization_scale: float = 0.05,
    ) -> None:
        super().__init__(channel_quantity, window_length)
        require_positive("angle quantity", angle_quantity)
        half: int = self.dimension // 2
        layer_quantity: int = ceil(angle_quantity / half)
        generator = torch.Generator().manual_seed(GIVENS_PAIRING_SEED)
        self.register_buffer(
            "pairings",
            torch.stack(
                [
                    torch.randperm(self.dimension, generator=generator)
                    for _ in range(layer_quantity)
                ]
            ),
        )
        self.angles = nn.Parameter(
            torch.randn(layer_quantity, half) * initialization_scale
        )
        mask = torch.zeros(layer_quantity * half)
        mask[:angle_quantity] = 1
        self.register_buffer("mask", mask.reshape(layer_quantity, half))

    def transform(self) -> Tensor:
        matrix: Tensor = torch.eye(self.dimension, device=self.angles.device)
        angles: Tensor = self.angles * self.mask
        for layer in range(len(angles)):
            first = self.pairings[layer, 0::2]
            second = self.pairings[layer, 1::2]
            cosines = torch.cos(angles[layer])[:, None]
            sines = torch.sin(angles[layer])[:, None]
            rows_first, rows_second = matrix[first], matrix[second]
            matrix = matrix.index_copy(
                0, first, cosines * rows_first - sines * rows_second
            ).index_copy(0, second, sines * rows_first + cosines * rows_second)
        return matrix


class DenseFilterBank(_MatrixFront):
    """
    A trained d x d matrix W of the window:

    - `UNITARY_MATRIX`: W = exp(i H), H Hermitian from d^2 parameters, which
      reaches every unitary: the ceiling of the circuits.
    - `ORTHOGONAL_MATRIX`: W = exp(S - S^T) from d (d - 1) / 2 parameters,
      every rotation: the ceiling of real circuits.
    - `COMPLEX_MATRIX`: W free complex, 2 d^2 parameters: energies of free
      pairs of real filters (quadrature pairs).
    - `REAL_MATRIX`: W free real, d^2 parameters: squared real filters.

    W starts at the identity plus noise of size `initialization_scale`.
    """

    upper_indices: Tensor

    def __init__(
        self,
        channel_quantity: int,
        window_length: int,
        *,
        matrix: str = UNITARY_MATRIX,
        initialization_scale: float = 0.05,
    ) -> None:
        super().__init__(channel_quantity, window_length)
        require_choice("matrix", matrix, MATRICES)
        self.matrix = matrix
        dimension: int = self.dimension
        if matrix == ORTHOGONAL_MATRIX:
            self.register_buffer(
                "upper_indices",
                torch.triu_indices(dimension, dimension, 1),
                persistent=False,
            )
            self.generator = nn.Parameter(
                torch.randn(dimension * (dimension - 1) // 2)
                * initialization_scale
            )
        else:
            self.generator = nn.Parameter(
                torch.randn(dimension, dimension) * initialization_scale
            )
        if matrix == COMPLEX_MATRIX:
            self.imaginary_generator = nn.Parameter(
                torch.randn(dimension, dimension) * initialization_scale
            )

    def transform(self) -> Tensor:
        """The trained d x d matrix W."""
        if self.matrix == UNITARY_MATRIX:
            hermitian: Tensor = torch.complex(
                (self.generator + self.generator.T) / 2,
                (self.generator - self.generator.T) / 2,
            )
            return torch.linalg.matrix_exp(1j * hermitian)
        if self.matrix == ORTHOGONAL_MATRIX:
            skew: Tensor = self.generator.new_zeros(
                self.dimension, self.dimension
            )
            skew = skew.index_put(
                (self.upper_indices[0], self.upper_indices[1]), self.generator
            )
            return torch.linalg.matrix_exp(skew - skew.T)
        identity: Tensor = torch.eye(
            self.dimension, device=self.generator.device
        )
        if self.matrix == COMPLEX_MATRIX:
            return torch.complex(
                identity + self.generator, self.imaginary_generator
            )
        return identity + self.generator


class MultiScaleFront(nn.Module):
    """
    For each (w, C) of `scales`, C channels from filters of w positions:
    with `RELU_MODE` a convolution with bias and ReLU, with
    `QUADRATURE_MODE` the energies (a.x)^2 + (b.x)^2 of C pairs of filters
    without bias. Channels of different widths are aligned on their centres.
    """

    def __init__(
        self,
        channel_quantity: int,
        scales: Sequence[Sequence[int]],
        mode: str,
    ) -> None:
        super().__init__()
        require_choice("mode", mode, MULTI_SCALE_MODES)
        if not scales:
            raise ValueError("A multi-scale front needs scales")
        for window_length, filter_quantity in scales:
            require_positive("window length", window_length)
            require_positive("filter quantity", filter_quantity)
        self.mode = mode
        filters_per_channel: int = 2 if mode == QUADRATURE_MODE else 1
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                channel_quantity,
                filters_per_channel * filter_quantity,
                window_length,
                bias=mode == RELU_MODE,
            )
            for window_length, filter_quantity in scales
        )
        self.filter_quantities: list[int] = [quantity for _, quantity in scales]
        self.longest_window: int = max(length for length, _ in scales)
        self.output_channel_quantity: int = sum(self.filter_quantities)

    def position_quantity(self, sequence_length: int) -> int:
        return _require_window(self.longest_window, sequence_length)

    def forward(self, channels: Tensor) -> Tensor:
        outputs: list[Tensor] = []
        for convolution, quantity in zip(
            self.convolutions, self.filter_quantities, strict=True
        ):
            output: Tensor = convolution(channels)
            if self.mode == QUADRATURE_MODE:
                output = (
                    (output * output)
                    .reshape(output.shape[0], quantity, 2, -1)
                    .sum(2)
                )
            else:
                output = torch.relu(output)
            outputs.append(output)
        positions: int = min(output.shape[2] for output in outputs)
        return torch.cat(
            [
                output[
                    :,
                    :,
                    (output.shape[2] - positions) // 2 : (
                        output.shape[2] - positions
                    )
                    // 2
                    + positions,
                ]
                for output in outputs
            ],
            1,
        )


class TwoLayerConvolutionFront(nn.Module):
    """
    DeepCpG's two convolutional layers with `filter_quantities` filters:
    conv 11 + ReLU, max pool 4, conv 3 + ReLU.
    """

    def __init__(
        self, channel_quantity: int, filter_quantities: Sequence[int]
    ) -> None:
        super().__init__()
        if len(filter_quantities) != 2:
            raise ValueError(
                "A two-layer front needs two filter quantities, not "
                f"{list(filter_quantities)}"
            )
        first, second = filter_quantities
        require_positive("filter quantity", first)
        require_positive("filter quantity", second)
        self.layers = nn.Sequential(
            nn.Conv1d(channel_quantity, first, heads.DEEPCPG_FIRST_WIDTH),
            nn.ReLU(),
            nn.MaxPool1d(heads.DEEPCPG_FIRST_POOL),
            nn.Conv1d(first, second, heads.DEEPCPG_SECOND_WIDTH),
            nn.ReLU(),
        )
        self.output_channel_quantity: int = second

    def position_quantity(self, sequence_length: int) -> int:
        positions: int = (
            (sequence_length - heads.DEEPCPG_FIRST_WIDTH + 1)
            // heads.DEEPCPG_FIRST_POOL
            - heads.DEEPCPG_SECOND_WIDTH
            + 1
        )
        if positions < 1:
            raise ValueError(
                f"DeepCpG's convolutions on {sequence_length} positions leave "
                "no positions"
            )
        return positions

    def forward(self, channels: Tensor) -> Tensor:
        return self.layers(channels)


class ClassicalNetwork(SequenceModel):
    """
    A classical front end, an optional per-channel affine map and ReLU, and
    DeepCpG's tail or the pooling head.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        front: str = CONVOLUTION_FRONT,
        window_length: int = 16,
        filter_quantity: int = heads.DEEPCPG_FIRST_CHANNELS,
        rank: int = 1,
        angle_quantity: int = 72,
        matrix: str = UNITARY_MATRIX,
        scales: Sequence[Sequence[int]] = ((4, 4), (8, 4), (16, 4), (32, 2)),
        mode: str = RELU_MODE,
        filter_quantities: Sequence[int] = (64, 96),
        initialization_scale: float = 0.05,
        activation: str = AFFINE_RELU_ACTIVATION,
        tail: str = DEEPCPG_TAIL,
        hidden_layer_sizes: Sequence[int] | None = None,
        dropout: float | None = None,
        first_pool: str | None = None,
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        sequence_length: The positions of every input.
        output_quantity: The number of outputs (cells).
        front: One of `FRONTS`.
        window_length: The window of the convolution, identity, low-rank,
            Givens and dense fronts.
        filter_quantity: The convolution front's filters.
        rank: The low-rank front's rank.
        angle_quantity: The Givens front's trained angles.
        matrix: The dense front's matrix, one of `MATRICES`.
        scales, mode: The multi-scale front's (window, filters) and mode.
        filter_quantities: The two-layer front's filters.
        initialization_scale: The noise of the Givens and dense fronts'
            initial parameters.
        activation: `AFFINE_RELU_ACTIVATION` or `NO_ACTIVATION` after the
            front.
        tail: `DEEPCPG_TAIL` or `POOLING_TAIL`.
        hidden_layer_sizes: The tail's dense layers. Defaults to DeepCpG's
            128 units, or none for the pooling tail.
        dropout: The tail's dropout. Defaults to 0.2, or 0 for the pooling
            tail.
        first_pool: The DeepCpG tail's first pool, see `heads.DeepCpGTail`.
        encoding_options: The input encoding, see `SequenceModel`.

        Raises
        ------
        ValueError if an option is invalid or the sequences are too short.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        require_choice("front", front, FRONTS)
        require_choice("activation", activation, ACTIVATIONS)
        require_choice("tail", tail, CLASSICAL_TAILS)
        channels: int = self.channel_quantity
        if front == CONVOLUTION_FRONT:
            self.front_end: nn.Module = ConvolutionFront(
                channels, window_length, filter_quantity
            )
        elif front == IDENTITY_FRONT:
            self.front_end = IdentityFront(channels, window_length)
        elif front == LOW_RANK_FRONT:
            self.front_end = LowRankFront(channels, window_length, rank)
        elif front == GIVENS_FRONT:
            self.front_end = GivensFront(
                channels,
                window_length,
                angle_quantity,
                initialization_scale=initialization_scale,
            )
        elif front == DENSE_FRONT:
            self.front_end = DenseFilterBank(
                channels,
                window_length,
                matrix=matrix,
                initialization_scale=initialization_scale,
            )
        elif front == MULTI_SCALE_FRONT:
            self.front_end = MultiScaleFront(channels, scales, mode)
        else:
            self.front_end = TwoLayerConvolutionFront(
                channels, filter_quantities
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
        else:
            if first_pool is not None:
                raise ValueError("The pooling tail has no first pool")
            self.tail = heads.PoolingHead(
                channel_quantity,
                output_quantity,
                hidden_layer_sizes=hidden_layer_sizes or (),
                dropout=dropout or 0.0,
            )

    def _probabilities(self, channels: Tensor) -> Tensor:
        return torch.sigmoid(
            self.tail(self.activation(self.front_end(channels)))
        )


class FeatureClassifier(SequenceModel):
    """
    The 44 sequence statistics of the feature-amplitude circuit, standardized
    without trained parameters, into a linear layer or an MLP.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        hidden_layer_sizes: Sequence[int] = (),
        **encoding_options: Any,
    ) -> None:
        """
        Raises
        ------
        ValueError if the inputs are not nucleotides or a size is invalid.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        self._require_bases()
        self.normalization = nn.BatchNorm1d(STATISTIC_QUANTITY, affine=False)
        self.head = heads.multilayer_perceptron(
            STATISTIC_QUANTITY, hidden_layer_sizes, output_quantity
        )

    def _probabilities(self, channels: Tensor) -> Tensor:
        return torch.sigmoid(
            self.head(self.normalization(sequence_statistics(channels)))
        )


class OneHotClassifier(SequenceModel):
    """
    The flattened encoded sequence into a linear layer, i.e. position-specific
    logistic regression, or an MLP with dropout 0.2.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        hidden_layer_sizes: Sequence[int] = (),
        **encoding_options: Any,
    ) -> None:
        super().__init__(sequence_length, output_quantity, **encoding_options)
        self.head = heads.multilayer_perceptron(
            self.channel_quantity * sequence_length,
            hidden_layer_sizes,
            output_quantity,
            dropout=heads.DEEPCPG_DROPOUT if hidden_layer_sizes else 0.0,
        )

    def _probabilities(self, channels: Tensor) -> Tensor:
        return torch.sigmoid(self.head(channels.flatten(1)))


class TokenFrequencyClassifier(SequenceModel):
    """
    Logistic regression on the frequencies of the input's channels (the mean
    encoding over positions), with one output shared by all cells, as the pure
    token-window circuits have.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        **encoding_options: Any,
    ) -> None:
        super().__init__(sequence_length, output_quantity, **encoding_options)
        self.head = nn.Linear(self.channel_quantity, 1)

    def _probabilities(self, channels: Tensor) -> Tensor:
        return torch.sigmoid(
            self.head(channels.mean(2)).expand(-1, self.output_quantity)
        )


class FactorizationMachine(SequenceModel):
    """
    A factorization machine of the flattened sequence encoding x, with one
    output per cell:

        logit_c = b_c + w_c . x + h_c . f(x),
        f(x) = ((V^T x)^2 - (V^2)^T x^2) / 2,

    where f(x) holds, for each of r factors, the sum over pairs i < j of
    V_ik V_jk x_i x_j. The factors V are shared by the cells, as a global
    circuit is, and each cell weighs them with its own h_c. With h_c all ones
    it is the usual factorization machine of rank r.
    """

    def __init__(
        self,
        sequence_length: int,
        output_quantity: int,
        *,
        rank: int = 4,
        initialization_scale: float = 0.01,
        **encoding_options: Any,
    ) -> None:
        """
        Args
        ----
        sequence_length: The positions of every input.
        output_quantity: The number of outputs (cells).
        rank: The number of factors r.
        initialization_scale: The standard deviation of the initial factors.
        encoding_options: The input encoding, see `SequenceModel`.

        Raises
        ------
        ValueError if the rank is less than 1.
        """
        super().__init__(sequence_length, output_quantity, **encoding_options)
        require_positive("rank", rank)
        feature_quantity: int = self.channel_quantity * sequence_length
        self.linear = nn.Linear(feature_quantity, output_quantity)
        self.factors = nn.Parameter(
            torch.randn(feature_quantity, rank) * initialization_scale
        )
        self.interaction_weights = nn.Linear(rank, output_quantity, bias=False)

    def _probabilities(self, channels: Tensor) -> Tensor:
        features: Tensor = channels.flatten(1)
        interactions: Tensor = (
            (features @ self.factors) ** 2
            - (features * features) @ (self.factors * self.factors)
        ) / 2
        return torch.sigmoid(
            self.linear(features) + self.interaction_weights(interactions)
        )
