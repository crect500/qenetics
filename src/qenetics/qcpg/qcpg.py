from enum import IntEnum
import logging
from math import ceil, log2
from typing import Iterable

import jax
from jax import numpy as jnp
from numpy.typing import NDArray
import optax
import pennylane as qml
from pennylane import numpy as pnp

from qenetics.qcpg import qcpg_models

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class Nucleodtide(IntEnum):
    """
    Maps nucleotide abbreviations to enum.

    A - 0
    T - 1
    C - 2
    G - 3
    """

    A = (0,)
    T = (1,)
    C = (2,)
    G = 3


def convert_nucleotide_to_enum(nucleotide: str) -> Nucleodtide:
    """
    Convert nucleotide abbreviation to corresponding enum.

    A - 0, T - 1, C - 2, G - 3

    Args
    ----
    nucleotide: The ASCII representation of the nucleotide abbreviation.

    Returns
    -------
    The corresponding nucleotide enum.

    """
    if nucleotide == "A":
        return Nucleodtide.A
    if nucleotide == "T":
        return Nucleodtide.T
    if nucleotide == "C":
        return Nucleodtide.C
    if nucleotide == "G":
        return Nucleodtide.G

    raise ValueError(f"Chracter {nucleotide} not recognized as valid nucleotide")


def string_to_nucleotides(nucleotide_string: str) -> list[Nucleodtide]:
    """
    Convert a string to nucleotide enumerations.

    Args
    ----
    nucleotide_string: A string of characters in the set A, T, C, G.

    Returns
    -------
    A list of nucleotide enumerations.
    """
    return [convert_nucleotide_to_enum(character) for character in nucleotide_string]


def calculate_address_register_size(values_to_encode: int) -> int:
    """
    Determine the address register size for the quantum circuit.

    Args
    ----
    values_to_encode: The number of values that the quantum circuit will encode.

    Returns
    -------
    The minimum quantum address register qubits required to encode all values.

    Raises
    ------
    ValueError if values_to_encode is less than 1.
    """
    if values_to_encode < 1:
        raise ValueError(f"Value {values_to_encode} must be greater than 0.")

    if values_to_encode == 1:
        return 1

    return int(ceil(log2(values_to_encode)))


def single_encode_nucleotide(
    nucleotide: Nucleodtide, index: int, address_register_size: int
) -> None:
    """
    One-hot encode nucleotide value into a quantum circuit at the appropriate address.

    Args
    ----
    nucleotide: An enumerated value corresponding to a type of nucleotide.
    index: The location of the nucleotide in the sequence.
    address_register_size: The size of the quantum address register.
    """
    controls: list[int] = [int(value) for value in bin(index).split("b")[1]]
    if len(controls) != address_register_size:
        controls = [0] * (address_register_size - len(controls)) + controls
    qml.MultiControlledX(
        wires=list(range(address_register_size))
        + [address_register_size + nucleotide.value],
        control_values=controls,
    )


def single_encode_all_nucleotides(nucleotides: Iterable[Nucleodtide]) -> None:
    """
    One-hot encode nucleotide values into a quantum circuit at appropriate addresses.

    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    address_register_size: int = calculate_address_register_size(len(nucleotides))
    for index, nucleotide in enumerate(nucleotides):
        single_encode_nucleotide(nucleotide, index, address_register_size)


def amplitude_encode_nucleotide(
    nucleotide: Nucleodtide, index: int, address_register_size: int
) -> None:
    """
    Amplitude-encode nucleotide value into a quantum circuit at the appropriate index.

    Args
    ----
    nucleotide:
    nucleotide: An enumerated value corresponding to a type of nucleotide.
    index: The location of the nucleotide in the sequence.
    address_register_size: The size of the quantum address register.
    """
    controls: list[int] = [int(value) for value in bin(index).split("b")[1]]
    if len(controls) != address_register_size:
        controls = [0] * (address_register_size - len(controls)) + controls
    if nucleotide == Nucleodtide.T or nucleotide == Nucleodtide.G:
        qml.MultiControlledX(
            wires=list(range(address_register_size)) + [address_register_size + 1],
            control_values=controls,
        )
    if nucleotide == Nucleodtide.C or nucleotide == Nucleodtide.G:
        qml.MultiControlledX(
            wires=list(range(address_register_size)) + [address_register_size],
            control_values=controls,
        )


def amplitude_encode_all_nucleotides(nucleotides: Iterable[Nucleodtide]) -> None:
    """
    Amplitude encode nucleotide values into a quantum circuit at appropriate addresses.

    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    address_register_size: int = calculate_address_register_size(len(nucleotides))
    for index, nucleotide in enumerate(nucleotides):
        amplitude_encode_nucleotide(nucleotide, index, address_register_size)


def _strongly_entangled_run_calculate_mean_squared_error(
    parameters: NDArray[float],
    nucleotides: list[Nucleodtide],
    target: NDArray[int],
) -> NDArray[float]:
    data_register_size: int = 4
    address_register_size: int = calculate_address_register_size(len(nucleotides))
    circuit_width: int = address_register_size + data_register_size
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def _run_circuit(
        sequence: list[Nucleodtide],
        circuit_parameters: NDArray[float],
    ):
        single_encode_all_nucleotides(sequence)
        qcpg_models.strongly_entangled_jax(circuit_parameters)

        return qml.expval(
            qml.PauliZ(wires=circuit_width - 2) @ qml.PauliZ(wires=circuit_width - 1)
        )

    return (_run_circuit(nucleotides, parameters) - pnp.array(target)) ** 2


def _strongly_entangled_run_calculate_loss(
    parameters: NDArray[float],
    nucleotides: list[Nucleodtide],
    targets: NDArray[int],
):
    predictions = jnp.array(
        [
            _strongly_entangled_run_calculate_mean_squared_error(
                parameters, sequence, target
            )
            for (sequence, target) in zip(nucleotides, targets, strict=True)
        ]
    )
    loss: jax.Array = jnp.mean(predictions)
    return loss


def _strongly_entangled_run_update_parameters(
    iteration: int,
    parameters: NDArray[float],
    nucleotides: NDArray[int],
    targets: NDArray[int],
    optimizer,
    optimizer_state: optax.GradientTransformationExtraArgs,
):
    logging.debug(f"Training loop {iteration}")
    logging.debug("Executing circuit...")
    loss_value, grads = jax.value_and_grad(_strongly_entangled_run_calculate_loss)(
        parameters, nucleotides, targets
    )
    logging.debug("Updating parameters...")
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    parameters = optax.apply_updates(parameters, updates)

    return (parameters, optimizer_state, loss_value)


def train_strongly_entangled_qcpg_circuit(
    parameters: NDArray[float],
    nucleotides: NDArray[int],
    targets: NDArray[int],
    max_steps: int = 50,
):
    optimizer = optax.adam(learning_rate=0.05)
    loss_history: list[float] = list()
    opt_state = optimizer.init(parameters)
    for iteration in range(max_steps):
        parameters, opt_state, loss_value = _strongly_entangled_run_update_parameters(
            iteration, parameters, nucleotides, targets, optimizer, opt_state
        )
        loss_history.append(loss_value)

    return parameters, loss_history
