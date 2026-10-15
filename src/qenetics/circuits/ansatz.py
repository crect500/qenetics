"""
Layered ansatzes as PennyLane operations with fast exact simulation.

Every circuit of the methylation models is a stack of layers, each a
single-qubit rotation on every wire followed by CNOTs:

- `STRONG_ANSATZ_STR`: Rot(phi, theta, omega) and CNOTs of range
  (layer mod (n - 1)) + 1, as `qp.StronglyEntanglingLayers`.
- `REAL_ANSATZ_STR`: RY and the same CNOTs, so real amplitudes stay real.
- `BASIC_ANSATZ_STR`: RX and a ring of CNOTs, as `qp.BasicEntanglerLayers`.
- `BASIC_RY_ANSATZ_STR`: RY and the ring.
- `BASIC_RZY_ANSATZ_STR`: RY(a) then RZ(b) and the ring.

Wire 0 is the most significant bit, as in PennyLane.

`LayeredAnsatz` decomposes into these gates, so every device can run it. Gate
by gate, though, a 320-layer circuit takes seconds per training step, so
`ansatz_unitary` builds the unitary at once: each layer's rotations as one
Kronecker product, its CNOTs as a row permutation, and the layers multiplied
as a balanced tree. On default.qubit, the operation is applied by a kernel
that multiplies the state by this unitary or, above `UNITARY_WIRE_LIMIT`
wires, applies each layer as two half-register Kronecker products. The kernel
is registered with PennyLane's `apply_operation`, which PennyLane documents as
extensible but internal.
"""

from __future__ import annotations

import functools
from math import pi

import numpy as np
import pennylane as qp
import torch
from numpy.typing import NDArray
from pennylane.devices.qubit.apply_operation import apply_operation
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike
from torch.utils.checkpoint import checkpoint

from qenetics.circuits.register_axes import (
    merge_register_axes,
    split_register_axes,
)

STRONG_ANSATZ_STR: str = "strong"
REAL_ANSATZ_STR: str = "real"
BASIC_ANSATZ_STR: str = "basic"
BASIC_RY_ANSATZ_STR: str = "basic_ry"
BASIC_RZY_ANSATZ_STR: str = "basic_rzy"
ANSATZES_STRINGS: tuple[str, ...] = (
    STRONG_ANSATZ_STR,
    REAL_ANSATZ_STR,
    BASIC_ANSATZ_STR,
    BASIC_RY_ANSATZ_STR,
    BASIC_RZY_ANSATZ_STR,
)
NORMAL_INITIALIZATION_STR: str = "normal"
UNIFORM_INITIALIZATION_STR: str = "uniform"
INITIALIZATIONS_STRINGS: tuple[str, ...] = (
    NORMAL_INITIALIZATION_STR,
    UNIFORM_INITIALIZATION_STR,
)
# Registers of up to this many wires are multiplied by their whole unitary.
UNITARY_WIRE_LIMIT: int = 8
# Layers applied between checkpoints when larger registers are trained.
CHECKPOINT_LAYER_QUANTITY: int = 24

_ANGLE_QUANTITIES: dict[str, int] = {
    STRONG_ANSATZ_STR: 3,
    REAL_ANSATZ_STR: 1,
    BASIC_ANSATZ_STR: 1,
    BASIC_RY_ANSATZ_STR: 1,
    BASIC_RZY_ANSATZ_STR: 2,
}


def check_ansatz(ansatz: str) -> None:
    """
    Check whether the ansatz key string is known.

    Args
    ----
    ansatz: The ansatz as a key string.

    Raises
    ------
    ValueError if the ansatz is unknown.
    """
    if ansatz not in ANSATZES_STRINGS:
        raise ValueError(
            f"Unknown ansatz {ansatz}; expected one of {', '.join(ANSATZES_STRINGS)}"
        )


def is_real_ansatz(ansatz: str) -> bool:
    """
    Whether an ansatz keeps real amplitudes real.

    Args
    ----
    ansatz: The ansatz as a string key.

    Returns
    -------
    Whether the ansatz keeps real amplitudes real.
    """
    check_ansatz(ansatz)
    return ansatz in (REAL_ANSATZ_STR, BASIC_RY_ANSATZ_STR)


def weight_shape(
    ansatz: str, layer_quantity: int, wire_quantity: int
) -> tuple[int, ...]:
    """
    Find the shape of an ansatz's weights.

    Args
    ----
    ansatz: The ansatz name.
    layer_quantity: The number of layers.
    wire_quantity: The number of wires.

    Returns
    -------
    (layers, wires) for one angle per wire, else (layers, wires, angles).

    Raises
    ------
    ValueError if the ansatz is unknown or a quantity is less than 1.
    """
    check_ansatz(ansatz)
    if layer_quantity < 1 or wire_quantity < 1:
        raise ValueError(
            "An ansatz needs at least one layer and one wire, not "
            f"{layer_quantity} layers and {wire_quantity} wires"
        )

    angle_quantity: int = _ANGLE_QUANTITIES[ansatz]
    if angle_quantity == 1:
        return (layer_quantity, wire_quantity)

    return (layer_quantity, wire_quantity, angle_quantity)


def initial_weights(
    shape: tuple[int, ...],
    initialization: str = NORMAL_INITIALIZATION_STR,
    scale: float = 0.05,
) -> torch.Tensor:
    """
    Draw initial circuit angles.

    Args
    ----
    shape: The shape of the weights.
    initialization: `NORMAL_INITIALIZATION_STR` for angles of standard deviation
        `scale` around 0, or `UNIFORM_INITIALIZATION_STR` for angles uniform in
        [0, 2 pi), the default of `qp.qnn.TorchLayer`.
    scale: The standard deviation of normal initialization.

    Returns
    -------
    The angles.

    Raises
    ------
    ValueError if the initialization is unknown.
    """
    if initialization == NORMAL_INITIALIZATION_STR:
        return torch.randn(shape) * scale

    if initialization == UNIFORM_INITIALIZATION_STR:
        return torch.rand(shape) * 2 * pi

    raise ValueError(
        f"Unknown initialization {initialization}; expected one of "
        f"{', '.join(INITIALIZATIONS_STRINGS)}"
    )


def layer_and_wire_quantities(
    weights: TensorLike, ansatz: str
) -> tuple[int, int]:
    """
    Read the number of layers and wires from an ansatz's weights.

    Args
    ----
    weights: The weights for the model.

    Returns
    -------
    (layer_quantity, wire_quantity)

    Raises
    ------
    ValueError if the ansatz is unknown or the weights have the wrong shape.
    """
    check_ansatz(ansatz)
    shape: tuple[int, ...] = tuple(qp.math.shape(weights))
    angle_quantity: int = _ANGLE_QUANTITIES[ansatz]
    angle_axes: int = 0 if angle_quantity == 1 else 1
    if len(shape) < 2 + angle_axes or (
        angle_axes and shape[-1] != angle_quantity
    ):
        expected: str = "(..., layers, wires" + (
            f", {angle_quantity})" if angle_axes else ")"
        )
        raise ValueError(
            f"Weights of the {ansatz} ansatz must have shape {expected}, "
            f"not {shape}"
        )

    return shape[-2 - angle_axes], shape[-1 - angle_axes]


def _stack_matrices(
    top_left: TensorLike,
    top_right: TensorLike,
    bottom_left: TensorLike,
    bottom_right: TensorLike,
) -> TensorLike:
    """
    Build 2 x 2 matrices their four elements.

    Returns
    -------
    The 2 x 2 matrices.

    """
    return qp.math.stack(
        [
            qp.math.stack([top_left, top_right], axis=-1),
            qp.math.stack([bottom_left, bottom_right], axis=-1),
        ],
        axis=-2,
    )


def _rot_matrices(
    phi: TensorLike, theta: TensorLike, omega: TensorLike
) -> TensorLike:
    """
    Builds rotation matrices for all sets of (phi, theta, omega).

    Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi), for all
    (phi, theta, omega).

    Args
    ----
    phi: The angles of the right RZ rotations.
    theta: The angles of the middle RY rotations.
    omega: The angles of the left RZ rotations.

    Returns
    -------
    Rotation matrices for all angle sets.

    """
    cosines: TensorLike = qp.math.cos(theta / 2)
    sines: TensorLike = qp.math.sin(theta / 2)
    sum_phases: TensorLike = qp.math.exp(-0.5j * (phi + omega))
    difference_phases: TensorLike = qp.math.exp(0.5j * (phi - omega))
    return _stack_matrices(
        sum_phases * cosines,
        -difference_phases * sines,
        qp.math.conj(difference_phases) * sines,
        qp.math.conj(sum_phases) * cosines,
    )


def _rotation_matrices(weights: TensorLike, ansatz: str) -> TensorLike:
    """
    The 2 x 2 single-qubit gate unitary of every layer and wire, (..., L, n, 2, 2).

    Args
    ----
    weights: The weights for the ansatz.
    ansatz: The ansatz key string.


    Returns
    -------
    All rotation matrices.

    """
    if ansatz == STRONG_ANSATZ_STR:
        return _rot_matrices(weights[..., 0], weights[..., 1], weights[..., 2])

    if ansatz == BASIC_RZY_ANSATZ_STR:
        return _rot_matrices(
            qp.math.zeros_like(weights[..., 0]),
            weights[..., 0],
            weights[..., 1],
        )

    cosines: TensorLike = qp.math.cos(weights / 2)
    sines: TensorLike = qp.math.sin(weights / 2)
    if ansatz == BASIC_ANSATZ_STR:
        return _stack_matrices(
            cosines + 0j, -1j * sines, -1j * sines, cosines + 0j
        )

    return _stack_matrices(cosines, -sines, sines, cosines)


def _wire_kronecker(gates: TensorLike) -> TensorLike:
    """
    Compute the Kronecker product over all 2 x 2 unitaries.

    The Kronecker product over wires, (..., n, 2, 2) -> (..., 2^n, 2^n), with
    wire 0 as the most significant bit.

    Args
    ----
    gates: An arbitrary number of 2 x 2 unitaries.

    Returns
    -------
    The resultant Kronecker product.

    """
    product: TensorLike = gates[..., 0, :, :]
    for wire in range(1, qp.math.shape(gates)[-3]):
        size: int = 2 * qp.math.shape(product)[-1]
        product = qp.math.reshape(
            qp.math.einsum(
                "...ij,...kl->...ikjl", product, gates[..., wire, :, :]
            ),
            (*qp.math.shape(product)[:-2], size, size),
        )
    return product


def entangler_pairs(
    ansatz: str, wire_quantity: int, layer_index: int
) -> tuple[tuple[int, int], ...]:
    """
    The (control, target) wires of a layer's CNOTs, in order.

    Matches `pennylane.StronglyEntanglingLayers` for the strong and real ansatzes and
    `pennylane.BasicEntanglerLayers` for the basic ones.

    Args
    ----
    ansatz: The ansatz key string.
    wire_quantity: The number of wires in the circuit.
    layer_index: The index of the layer to make controls for.

    Returns
    -------
    (control, target) indices for each CNOT in a layer.

    """
    if wire_quantity == 1:
        return ()

    if ansatz in (STRONG_ANSATZ_STR, REAL_ANSATZ_STR):
        cnot_range: int = layer_index % (wire_quantity - 1) + 1
        return tuple(
            (wire, (wire + cnot_range) % wire_quantity)
            for wire in range(wire_quantity)
        )

    if wire_quantity == 2:
        return ((0, 1),)

    return tuple(
        (wire, (wire + 1) % wire_quantity) for wire in range(wire_quantity)
    )


@functools.lru_cache
def _entangler_permutation(
    wire_quantity: int, pairs: tuple[tuple[int, int], ...]
) -> NDArray[np.int64]:
    """
    Indices p with (P psi)[i] = psi[p[i]] for a sequence of CNOTs.

    Args
    ----
    wire_quantity: The number of wires in the circuit.
    pairs: (control, target) pairs

    Returns
    -------
    The indices of the CNOT ladder.

    """
    basis: NDArray[np.int64] = np.arange(2**wire_quantity)
    indices: NDArray[np.int64] = basis.copy()
    for control, target in pairs:
        control_bit: int = 1 << (wire_quantity - 1 - control)
        target_bit: int = 1 << (wire_quantity - 1 - target)
        indices = indices[
            np.where(basis & control_bit, basis ^ target_bit, basis)
        ]
    return indices


def _layer_permutation(
    ansatz: str, wire_quantity: int, layer_index: int
) -> NDArray[np.int64]:
    """
    Return the CNOT ladder for the layer.

    Args
    ----
    ansatz: The ansatz key string.
    wire_quantity: The number of wires in the circuit.
    layer_index: The index of the layer to create the CNOT ladder for.

    Returns
    -------
    The CNOT ladder as an array.

    """
    return _entangler_permutation(
        wire_quantity, entangler_pairs(ansatz, wire_quantity, layer_index)
    )


def _tree_product(matrices: TensorLike) -> TensorLike:
    """
    Multiplies a set of matrices.

    [M_0, ..., M_{L-1}] along axis -3 -> M_{L-1} ... M_0.

    Args
    ----
    matrices: The matrices to be multiplied. The first index in the
        container is the rightmost matrix.

    Returns
    -------
    The resultant matrix.

    """
    while (quantity := qp.math.shape(matrices)[-3]) > 1:
        paired: int = quantity - quantity % 2
        products: TensorLike = (
            matrices[..., 1:paired:2, :, :] @ matrices[..., 0:paired:2, :, :]
        )
        if quantity % 2:
            products = qp.math.concatenate(
                [products, matrices[..., -1:, :, :]], axis=-3
            )
        matrices = products
    return matrices[..., 0, :, :]


def ansatz_unitary(weights: TensorLike, ansatz: str) -> TensorLike:
    """
    Build the unitary of an ansatz.

    Args
    ----
    weights: The angles, of `weight_shape`, optionally with leading axes of
        independent circuits.
    ansatz: The ansatz name.

    Returns
    -------
    The unitaries U, output = U psi, with the weights' leading axes. Real for
    real ansatzes, complex otherwise. Differentiable in the weights' interface.

    Raises
    ------
    ValueError if the ansatz is unknown or the weights have the wrong shape.
    """
    layer_quantity, wire_quantity = layer_and_wire_quantities(weights, ansatz)
    layers: TensorLike = _wire_kronecker(_rotation_matrices(weights, ansatz))
    if wire_quantity > 1:
        permutations: NDArray[np.int64] = np.stack(
            [
                _layer_permutation(ansatz, wire_quantity, layer)
                for layer in range(layer_quantity)
            ]
        )
        layers = layers[
            ..., np.arange(layer_quantity)[:, np.newaxis], permutations, :
        ]
    return _tree_product(layers)


def _apply_layer_range(
    states: TensorLike,
    weights: TensorLike,
    ansatz: str,
    start: int,
    stop: int,
) -> TensorLike:
    """
    Apply layers [start, stop) to states (B, 2^n) one layer at a time.

    A layer's rotations are K_high (x) K_low on the two halves of the
    register, which act on the state viewed as a 2^h x 2^(n-h) matrix S as
    K_high S K_low^T.

    Args
    ----
    states: The states to apply the layers to.
    weights: The weights of the rotation matrices.
    ansatz: The ansatz key string.
    start: The index of the first layer to apply.
    stop: The index of the last layer to apply.

    Returns
    -------
    The resultant states.

    """
    wire_quantity: int = layer_and_wire_quantities(weights, ansatz)[1]
    high_quantity: int = wire_quantity // 2
    state_quantity: int = qp.math.shape(states)[0]
    gates: TensorLike = qp.math.cast_like(
        _rotation_matrices(weights[start:stop], ansatz), states
    )
    for offset in range(stop - start):
        high: TensorLike = _wire_kronecker(gates[offset, :high_quantity])
        low: TensorLike = _wire_kronecker(gates[offset, high_quantity:])
        matrices: TensorLike = qp.math.reshape(
            states,
            (
                state_quantity,
                2**high_quantity,
                2 ** (wire_quantity - high_quantity),
            ),
        )
        states = qp.math.reshape(
            high @ matrices @ qp.math.transpose(low),
            (state_quantity, 2**wire_quantity),
        )
        states = states[
            ..., _layer_permutation(ansatz, wire_quantity, start + offset)
        ]
    return states


def apply_ansatz_by_layer(
    states: TensorLike, weights: TensorLike, ansatz: str
) -> TensorLike:
    """
    Apply an ansatz to states (B, 2^n) one layer at a time.

    When torch gradients are recorded, every `CHECKPOINT_LAYER_QUANTITY`
    layers are checkpointed, so that the backward pass stores the states of
    one block of layers rather than of the whole circuit.

    Args
    ----
    states: The states to apply the ansatz to.
    weights: The weights of the rotation matrices.
    ansatz: The ansatz key string.

    Raises
    ------
    ValueError if the ansatz is unknown, the weights have the wrong shape or
    leading axes, or the states do not match the weights' wires.
    """
    layer_quantity, wire_quantity = layer_and_wire_quantities(weights, ansatz)
    if len(qp.math.shape(weights)) != len(
        weight_shape(ansatz, layer_quantity, wire_quantity)
    ):
        raise ValueError("Weights applied by layer must not have batch axes")
    if (
        len(qp.math.shape(states)) != 2
        or qp.math.shape(states)[1] != 2**wire_quantity
    ):
        raise ValueError(
            f"States must have shape (B, {2**wire_quantity}), not "
            f"{tuple(qp.math.shape(states))}"
        )

    is_checkpointed: bool = (
        qp.math.get_interface(states, weights) == "torch"
        and torch.is_grad_enabled()
        and (qp.math.requires_grad(states) or qp.math.requires_grad(weights))
        and layer_quantity > CHECKPOINT_LAYER_QUANTITY
    )
    block: int = (
        CHECKPOINT_LAYER_QUANTITY if is_checkpointed else layer_quantity
    )
    for start in range(0, layer_quantity, block):
        stop: int = min(layer_quantity, start + block)
        if is_checkpointed:
            states = checkpoint(
                _apply_layer_range,
                states,
                weights,
                ansatz,
                start,
                stop,
                use_reentrant=False,
            )
        else:
            states = _apply_layer_range(states, weights, ansatz, start, stop)
    return states


class LayeredAnsatz(Operation):
    """
    A layered ansatz on a register, see the module documentation.

    Its decomposition is the matching PennyLane template or gates, which
    every device can run. On default.qubit it is applied by a fast kernel.
    """

    # Trainable instances are decomposed by gradient transforms, whose gates
    # carry their own gradient recipes.
    grad_method = None

    def __init__(
        self,
        weights: TensorLike,
        wires: WiresLike,
        ansatz: str = STRONG_ANSATZ_STR,
        id: str | None = None,
    ) -> None:
        wires = Wires(wires)
        wire_quantity: int = layer_and_wire_quantities(weights, ansatz)[1]
        if len(qp.math.shape(weights)) != len(
            weight_shape(ansatz, 1, wire_quantity)
        ):
            raise ValueError(
                "LayeredAnsatz weights must not have batch axes, but have "
                f"shape {tuple(qp.math.shape(weights))}"
            )
        if wire_quantity != len(wires):
            raise ValueError(
                f"Weights for {wire_quantity} wires cannot act on "
                f"{len(wires)} wires"
            )
        self.hyperparameters["ansatz"] = ansatz
        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self) -> int:
        return 1

    @staticmethod
    def compute_matrix(weights: TensorLike, ansatz: str) -> TensorLike:
        unitary: TensorLike = ansatz_unitary(weights, ansatz)
        # Real unitaries are cast so that they multiply complex states.
        return unitary + 0j if is_real_ansatz(ansatz) else unitary

    @staticmethod
    def compute_decomposition(
        weights: TensorLike, wires: WiresLike, ansatz: str
    ) -> list[Operation]:
        wires = Wires(wires)
        if ansatz == STRONG_ANSATZ_STR:
            return [qp.StronglyEntanglingLayers(weights, wires=wires)]

        if ansatz == BASIC_ANSATZ_STR:
            return [qp.BasicEntanglerLayers(weights, wires=wires)]

        if ansatz == BASIC_RY_ANSATZ_STR:
            return [
                qp.BasicEntanglerLayers(weights, wires=wires, rotation=qp.RY)
            ]

        operations: list[Operation] = []
        for layer in range(qp.math.shape(weights)[0]):
            for wire_index, wire in enumerate(wires):
                if ansatz == REAL_ANSATZ_STR:
                    operations.append(qp.RY(weights[layer, wire_index], wire))
                else:
                    operations.append(
                        qp.RY(weights[layer, wire_index, 0], wire)
                    )
                    operations.append(
                        qp.RZ(weights[layer, wire_index, 1], wire)
                    )
            operations.extend(
                qp.CNOT([wires[control], wires[target]])
                for control, target in entangler_pairs(
                    ansatz, len(wires), layer
                )
            )
        return operations


@apply_operation.register
def _apply_layered_ansatz(
    op: LayeredAnsatz,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger: object = None,
    **_: object,
) -> TensorLike:
    """
    Apply an ansatz to a state.

    Args
    ----
    op: The ansatz operation to apply.
    state: The state on which to apply the ansatz.
    is_state_batched: Whether or not multiple states are provided.
    debugger: The debugger, if provided.

    Returns
    -------
    The resultant states.

    """
    merged, permutation, other_shape = merge_register_axes(
        state, op.wires, is_state_batched
    )
    weights: TensorLike = op.data[0]
    ansatz: str = op.hyperparameters["ansatz"]
    wire_quantity: int = len(op.wires)
    if wire_quantity <= UNITARY_WIRE_LIMIT:
        unitary: TensorLike = qp.math.cast_like(
            ansatz_unitary(weights, ansatz), merged
        )
        merged = merged @ qp.math.transpose(unitary)
    else:
        rows: TensorLike = qp.math.reshape(merged, (-1, 2**wire_quantity))
        merged = qp.math.reshape(
            apply_ansatz_by_layer(rows, weights, ansatz),
            qp.math.shape(merged),
        )
    return split_register_axes(merged, permutation, other_shape, wire_quantity)
