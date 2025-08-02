from enum import IntEnum
import logging
from math import ceil, log2
from pathlib import Path
from typing import Iterable

import jax
from jax import numpy as jnp
import numpy as np
from numpy.typing import NDArray
import optax
import pennylane as qml

from qenetics.qcpg import qcpg_models
from qenetics.tools import metrics

logger = logging.getLogger(__name__)


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

    raise ValueError(
        f"Chracter {nucleotide} not recognized as valid nucleotide"
    )


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
    return [
        convert_nucleotide_to_enum(character) for character in nucleotide_string
    ]


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
    address_register_size: int = calculate_address_register_size(
        len(nucleotides)
    )
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
            wires=list(range(address_register_size))
            + [address_register_size + 1],
            control_values=controls,
        )
    if nucleotide == Nucleodtide.C or nucleotide == Nucleodtide.G:
        qml.MultiControlledX(
            wires=list(range(address_register_size)) + [address_register_size],
            control_values=controls,
        )


def amplitude_encode_all_nucleotides(
    nucleotides: Iterable[Nucleodtide],
) -> None:
    """
    Amplitude encode nucleotide values into a quantum circuit at appropriate addresses.

    Args
    ----
    nucleotides: A sequence of enum-mapped nucleotide values.
    """
    address_register_size: int = calculate_address_register_size(
        len(nucleotides)
    )
    for index, nucleotide in enumerate(nucleotides):
        amplitude_encode_nucleotide(nucleotide, index, address_register_size)


def _strongly_entangled_run_circuit(
    parameters: NDArray[float],
    nucleotides: list[Nucleodtide],
    target: int,
) -> NDArray[float]:
    logging.debug(
        f"Training sequence {str(nucleotides)} with methhylation {target}"
    )
    data_register_size: int = 4
    address_register_size: int = calculate_address_register_size(
        len(nucleotides)
    )
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
            qml.PauliZ(wires=circuit_width - 2)
            @ qml.PauliZ(wires=circuit_width - 1)
        )

    return _run_circuit(nucleotides, parameters)


def _calculate_squared_error(
    result: NDArray[float], target: NDArray[float]
) -> NDArray[float]:
    return (result - target) ** 2


def _strongly_entangled_run_calculate_loss(
    parameters: NDArray[float],
    nucleotides: list[list[Nucleodtide]],
    targets: NDArray[int],
):
    logger.debug(f"{len(nucleotides)} sequences and {len(targets)} targets")
    predictions = jnp.array(
        [
            _calculate_squared_error(
                _strongly_entangled_run_circuit(parameters, sequence, target),
                target,
            )
            for (sequence, target) in zip(nucleotides, targets, strict=True)
        ]
    )
    loss: jax.Array = jnp.mean(predictions)
    return loss


def _strongly_entangled_run_update_parameters(
    iteration: int,
    parameters: NDArray[float],
    nucleotides: list[list[Nucleodtide]],
    targets: NDArray[int],
    optimizer,
    optimizer_state: optax.GradientTransformationExtraArgs,
):
    logging.info(f"Training loop {iteration}")
    logging.debug("Executing circuit.")
    loss_value, grads = jax.value_and_grad(
        _strongly_entangled_run_calculate_loss
    )(parameters, nucleotides, targets)
    logging.debug("Updating parameters.")
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    parameters = optax.apply_updates(parameters, updates)

    return (parameters, optimizer_state, loss_value)


def _evaluate_test_performance(
    parameters: NDArray[float],
    test_sequences: list[list[Nucleodtide]],
    test_methylations: NDArray[int],
    threshold: float = 0.5,
) -> metrics.Metrics:
    scaled_threshold: float = threshold * 2 - 1.0
    normalized_truth: NDArray[int] = np.array(
        [0 if truth == -1 else 1 for truth in test_methylations], dtype=int
    )
    predictions = np.array(
        [
            0
            if _strongly_entangled_run_circuit(
                parameters, sequence, methylation
            )
            < scaled_threshold
            else 1
            for (sequence, methylation) in zip(
                test_sequences, test_methylations, strict=True
            )
        ],
        dtype=int,
    )
    return metrics.generate_metrics(predictions, normalized_truth)


def train_strongly_entangled_qcpg_circuit(
    parameters: NDArray[float],
    training_nucleotides: list[list[Nucleodtide]],
    training_targets: NDArray[int],
    test_nucleotides: list[list[Nucleodtide]],
    test_targets: NDArray[int],
    output_file: Path,
    max_steps: int = 50,
):
    logger.info(f"Saving results to {output_file}")
    with output_file.open("w") as fd:
        fd.write(f"iteration,loss,{metrics.METRICS_HEADERS}\n")
        optimizer = optax.adam(learning_rate=0.05)
        loss_history: list[float] = list()
        opt_state = optimizer.init(parameters)
        for iteration in range(max_steps):
            parameters, opt_state, loss_value = (
                _strongly_entangled_run_update_parameters(
                    iteration,
                    parameters,
                    training_nucleotides,
                    training_targets,
                    optimizer,
                    opt_state,
                )
            )
            result_metrics: metrics.Metrics = _evaluate_test_performance(
                parameters, test_nucleotides, test_targets
            )
            fd.write(
                f"{iteration},"
                f"{str(loss_value)},"
                f"{(metrics.metrics_to_csv_row(result_metrics))}\n"
            )

    return parameters, loss_history
