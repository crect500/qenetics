"""
Reshape simulator states so that custom kernels can act on one register.

default.qubit holds a state with one axis per wire, after an optional batch
axis. A kernel that acts on a register moves the register's axes to the end,
merges them into one axis of its basis states, and splits them back after.
"""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.typing import TensorLike
from pennylane.wires import Wires


def merge_register_axes(
    state: TensorLike, wires: Wires, is_state_batched: bool
) -> tuple[TensorLike, list[int], tuple[int, ...]]:
    """
    Move the axes of a register to the end of a state and merge them.

    Args
    ----
    state: The state, with one axis per wire after any batch axis.
    wires: The indices of the register's wires.
    is_state_batched: Whether the state has a leading batch axis.

    Returns
    -------
    The state with the register's basis states along its last axis, the
    permutation applied to the state's axes, and the shape of the other axes.
    """
    register_axes: list[int] = [wire + is_state_batched for wire in wires]
    other_axes: list[int] = [
        axis for axis in range(qp.math.ndim(state)) if axis not in register_axes
    ]
    permutation: list[int] = other_axes + register_axes
    moved: TensorLike = qp.math.transpose(state, permutation)
    other_shape: tuple[int, ...] = tuple(
        qp.math.shape(moved)[: len(other_axes)]
    )
    return (
        qp.math.reshape(moved, (*other_shape, 2 ** len(wires))),
        permutation,
        other_shape,
    )


def split_register_axes(
    merged: TensorLike,
    permutation: list[int],
    other_shape: tuple[int, ...],
    wire_quantity: int,
) -> TensorLike:
    """Invert `merge_register_axes`."""
    split: TensorLike = qp.math.reshape(
        merged, (*other_shape, *(2,) * wire_quantity)
    )
    return qp.math.transpose(split, np.argsort(permutation).tolist())
