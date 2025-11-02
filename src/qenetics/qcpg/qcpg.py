from dataclasses import dataclass, field
import logging
from math import isnan
from pathlib import Path

import jax
import torch
from jax import numpy as jnp
import numpy as np
from numpy.typing import NDArray
import optax
import pennylane as qml
from sklearn.metrics import auc, roc_curve
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from qenetics.qcpg import qcpg_models
from qenetics.tools import cpg_sampler, metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingParameters:
    data_directory: Path
    output_filepath: Path
    training_chromosomes: list[str] = field(
        default_factory=lambda: ["1", "3", "5", "7", "9", "11"]
    )
    validation_chromosomes: list[str] = field(
        default_factory=lambda: ["2", "4", "6", "8", "10", "12"]
    )
    entangler: str = "basic"
    layer_quantity: int = 1
    epochs: int = 100
    learning_rate: float = 0.0001
    l1_regularizer: float = 0.0
    l2_regularizer: float = 0.0
    batch_size: int = 1
    report_every: int = 1000
    model_filepath: Path | None = None


def _strongly_entangled_run_circuit(
    parameters: NDArray[float],
    sequence: list[cpg_sampler.Nucleotide],
) -> float:
    """
    Run a qcpg circuit with a StronglyEntangled ansatz as the underlying model.


    Args
    ----
    parameters: The parameters for the ansatz.
    sequence: The nucleotide sequence to encode in the circuit.

    Returns
    -------

    """
    logging.debug(f"Training on sequence {str(sequence)}")
    data_register_size: int = 4
    address_register_size: int = qcpg_models.calculate_address_register_size(
        len(sequence)
    )
    circuit_width: int = address_register_size + data_register_size
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def _run_circuit(
        sequence: list[cpg_sampler.Nucleotide],
        circuit_parameters: NDArray[float],
    ):
        qcpg_models.single_encode_all_nucleotides(sequence)
        qcpg_models.strongly_entangled_jax(circuit_parameters)

        return qml.expval(
            qml.PauliZ(wires=circuit_width - 2)
            @ qml.PauliZ(wires=circuit_width - 1)
        )

    return _run_circuit(sequence, parameters)


def _strongly_entangled_run_calculate_loss(
    parameters: NDArray[float],
    sequences: list[list[cpg_sampler.Nucleotide]],
    methylations: NDArray[int],
) -> jax.Array:
    """
    Run circuit and calculate loss.

    Args
    ----
    parameters: The current parameters of the ansatz.
    sequences: The sequence samples.
    methylations: The associated methylation truth for each sample.

    Returns
    -------
    The mean of the losses for the batch.
    """
    logger.debug(
        f"{len(sequences)} sequences and {len(methylations)} methylations"
    )
    predictions = jnp.array(
        [
            (
                _strongly_entangled_run_circuit(parameters, sequence)
                - methylation
            )
            ** 2
            for (sequence, methylation) in zip(
                sequences, methylations, strict=True
            )
        ]
    )
    loss: jax.Array = jnp.mean(predictions)
    return loss


def _strongly_entangled_run_update_parameters(
    parameters: NDArray[float],
    sequences: list[list[cpg_sampler.Nucleotide]],
    methylations: NDArray[int],
    optimizer: optax.GradientTransformationExtraArgs,
    optimizer_state: optax.GradientTransformationExtraArgs,
) -> tuple[jax.Array, optax.GradientTransformationExtraArgs, jax.Array]:
    """
    Update the parameters for the qcpg circuit.

    parameters: The circuit parameters.
    sequences: The list of sequences to train the circuit on.
    methylations: The methylation truth associated with each sequence.
    optimizer: The chosen optimizer.
    optimizer_state: The current state of the optimizer.

    Returns
    -------
    The parameters, current optimizer state, and loss value of the iteration.
    """
    logging.debug("Executing circuit.")
    loss_value, grads = jax.value_and_grad(
        _strongly_entangled_run_calculate_loss
    )(parameters, sequences, methylations)
    logging.debug("Updating parameters.")
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    parameters = optax.apply_updates(parameters, updates)

    return parameters, optimizer_state, loss_value


def _evaluate_test_performance(
    parameters: NDArray[float],
    test_sequences: list[list[cpg_sampler.Nucleotide]],
    test_methylations: NDArray[int],
    threshold: float = 0.5,
) -> metrics.Metrics:
    """
    Evaluate the current performance metrics at the current iteration.

    parameters: The circuit parameters.
    test_sequences: The list of sequences in the test set.
    test_methylations: The methylation truth associated with each sequence.
    threshold: The threshold at which to consider a circuit result positive.

    Returns
    -------
    The performance metrics.
    """
    scaled_threshold: float = threshold * 2 - 1.0
    normalized_truth: NDArray[int] = np.array(
        [0 if truth == -1 else 1 for truth in test_methylations], dtype=int
    )
    predictions = np.array(
        [
            0
            if _strongly_entangled_run_circuit(parameters, sequence)
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
    training_sequences: list[list[cpg_sampler.Nucleotide]],
    training_methylations: NDArray[int],
    test_sequences: list[list[cpg_sampler.Nucleotide]],
    test_methylations: NDArray[int],
    output_file: Path,
    max_steps: int = 50,
) -> tuple[NDArray[float], list[float], list[metrics.Metrics]]:
    """
    Train a qcpg circuit given training and test sets.

    parameters: The initial circuit parameters.
    training_sequences: The list of sequences to train on.
    training_methylations: The methylation truth associated with the training sequences.
    test_sequences: The list of sequences to use as a test set.
    test_methylations: The methylation truth associated with the test sequences.
    output_file: The filepath in which to store the training results.
    max_steps: The maximum training iterations.

    Returns
    -------
    The final parameters, the loss history, and the performance metrics history.
    """
    logger.info(f"Saving results to {output_file}")
    with output_file.open("w") as fd:
        fd.write(f"iteration,loss,{metrics.METRICS_HEADERS}\n")
        optimizer = optax.adam(learning_rate=0.05)
        loss_history: list[float] = list()
        metrics_history: list[metrics.Metrics] = []
        opt_state = optimizer.init(parameters)
        for iteration in range(max_steps):
            logging.info(f"Training loop {iteration}")
            parameters, opt_state, loss_value = (
                _strongly_entangled_run_update_parameters(
                    parameters,
                    training_sequences,
                    training_methylations,
                    optimizer,
                    opt_state,
                )
            )
            result_metrics: metrics.Metrics = _evaluate_test_performance(
                parameters, test_sequences, test_methylations
            )
            fd.write(
                f"{iteration},"
                f"{str(loss_value)},"
                f"{(metrics.metrics_to_csv_row(result_metrics))}\n"
            )
            loss_history.append(float(loss_value))
            metrics_history.append(result_metrics)

    return parameters, loss_history, metrics_history


def _non_nan_indices(truth: Tensor) -> list[int]:
    indices: list[int] = []
    for index, truth_value in enumerate(truth):
        if not truth_value.isnan():
            indices.append(index)

    return indices


def _train_one_epoch(
    model: nn.Module,
    epoch: int,
    training_loader: DataLoader,
    optimizer: optim.Optimizer,
    training_parameters: TrainingParameters,
) -> float:
    accumulated_loss: float = 0.0
    return_loss: float = 0.0
    model.train(True)

    for batch_index, batch_data in enumerate(training_loader):
        inputs, labels = batch_data
        optimizer.zero_grad()
        outputs: Tensor = model(inputs[0])
        non_nan_indices = _non_nan_indices(labels[0])
        loss: Tensor = nn.functional.binary_cross_entropy(
            outputs[non_nan_indices], labels[0][non_nan_indices]
        )
        if training_parameters.l1_regularizer != 0.0:
            loss += training_parameters.l1_regularizer * sum(
                parameter_vector.abs().sum()
                for parameter_vector in model.parameters()
            )

        if training_parameters.l2_regularizer != 0.0:
            loss += training_parameters.l2_regularizer * sum(
                parameter_vector.pow(2).sum()
                for parameter_vector in model.parameters()
            )

        loss.backward()
        optimizer.step()
        accumulated_loss += loss.item()
        if (
            batch_index % training_parameters.report_every == 0
            and batch_index != 0
        ):
            return_loss = accumulated_loss / training_parameters.report_every
            logger.info(
                f"Epoch {epoch} - Batch {batch_index} loss: {return_loss}"
            )
            accumulated_loss = 0.0

    return return_loss


def _evaluate_validation_set(
    model: nn.Module,
    validation_loader: DataLoader,
    training_parameters: TrainingParameters,
) -> tuple[float, float]:
    accumulated_loss: float = 0.0
    accumulated_auc: float = 0.0
    model.eval()
    invalid_auc_count = 0
    with torch.no_grad():
        for batch_index, validation_data in enumerate(validation_loader):
            inputs, labels = validation_data
            outputs: Tensor = model(inputs[0])

            non_nan_indices = _non_nan_indices(labels[0])
            true_positive_rate, false_positive_rate, _ = roc_curve(
                labels[0][non_nan_indices], outputs[non_nan_indices]
            )
            batch_auc: float = auc(false_positive_rate, true_positive_rate)
            if isnan(batch_auc):
                invalid_auc_count += 0
            else:
                accumulated_auc += auc(false_positive_rate, true_positive_rate)

            loss: Tensor = nn.functional.binary_cross_entropy(
                outputs[non_nan_indices], labels[0][non_nan_indices]
            )
            if training_parameters.l1_regularizer != 0.0:
                loss += training_parameters.l1_regularizer * sum(
                    parameter_vector.abs().sum()
                    for parameter_vector in model.parameters()
                )

            if training_parameters.l2_regularizer != 0.0:
                loss += training_parameters.l2_regularizer * sum(
                    parameter_vector.pow(2).sum()
                    for parameter_vector in model.parameters()
                )

            accumulated_loss += loss

    return accumulated_loss / (
        batch_index + 1 - invalid_auc_count
    ), accumulated_auc / (batch_index + 1)


def _train_all_epochs(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: optim.Optimizer,
    output_filepath: Path,
    training_parameters: TrainingParameters,
) -> None:
    for epoch in range(training_parameters.epochs):
        logger.info(f"Training epoch {epoch}")
        average_training_loss: float = _train_one_epoch(
            model, epoch, training_loader, optimizer, training_parameters
        )
        average_validation_loss, average_auc = _evaluate_validation_set(
            model, validation_loader, training_parameters
        )
        logger.info(
            f"Epoch {epoch} Training loss: {average_training_loss}, Validation loss: "
            f"{average_validation_loss}, Validation AUC: {average_auc}"
        )
        torch.save(model.state_dict(), output_filepath)


def train_qnn_circuit(training_parameters: TrainingParameters) -> None:
    training_loader = DataLoader(
        cpg_sampler.H5CpGDataset(
            [
                training_parameters.data_directory / f"chr{chromosome}.h5"
                for chromosome in training_parameters.training_chromosomes
            ]
        ),
        batch_size=training_parameters.batch_size,
    )

    logger.info(
        f"Loaded {len(training_loader.dataset)} samples from training files:"
    )
    for chromosome in training_parameters.training_chromosomes:
        logger.info(training_parameters.data_directory / f"chr{chromosome}.h5")

    validation_loader = DataLoader(
        cpg_sampler.H5CpGDataset(
            [
                training_parameters.data_directory / f"chr{chromosome}.h5"
                for chromosome in training_parameters.validation_chromosomes
            ]
        ),
        batch_size=training_parameters.batch_size,
    )
    logger.info(
        f"Loading {len(validation_loader.dataset)} samples from validation files:"
    )
    for chromosome in training_parameters.validation_chromosomes:
        logger.info(training_parameters.data_directory / f"chr{chromosome}.h5")

    logger.info("Using the following training parameters:")
    logger.info(f"\tEntangler: {training_parameters.entangler}")
    logger.info(f"\tLayer quantity: {training_parameters.layer_quantity}")
    logger.info(f"\tLearning rate: {training_parameters.learning_rate}")
    logger.info(f"\tEpochs: {training_parameters.epochs}")

    sequence_length: int = training_loader.dataset.data.shape[1]
    output_shape: int = training_loader.dataset.experiment_quantity
    model = qcpg_models.QNN(
        sequence_length,
        training_parameters.layer_quantity,
        output_shape,
        entangler=training_parameters.entangler,
    )
    if training_parameters.model_filepath is not None:
        model.load_state_dict(torch.load(training_parameters.model_filepath))

    optimizer = optim.SGD(
        model.parameters(), lr=training_parameters.learning_rate
    )
    _train_all_epochs(
        model=model,
        training_loader=training_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        output_filepath=training_parameters.output_filepath,
        training_parameters=training_parameters,
    )
