import logging
import os
import socket
from dataclasses import dataclass, field, fields
from pathlib import Path

import jax
import numpy as np
import optax
import pennylane as qml
import torch
import torch.multiprocessing as mp
from jax import numpy as jnp
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from qenetics.qcpg import qcpg_models, records
from qenetics.tools import data, dna, metrics, report

logger = logging.getLogger(__name__)

# The cell name recorded for labels stored as a single unnamed dataset.
UNNAMED_EXPERIMENT: str = "unnamed"


@dataclass
class TrainingParameters:
    data_directory: Path
    output_directory: Path
    # The chromosome splits of Angermueller et al. (2017). Validation
    # chromosomes default to every chromosome file in the data directory not
    # used for training or testing.
    training_chromosomes: list[str] = field(
        default_factory=lambda: ["1", "3", "5", "7", "9", "11"]
    )
    validation_chromosomes: list[str] | None = None
    test_chromosomes: list[str] = field(
        default_factory=lambda: ["2", "4", "6", "8", "10", "12"]
    )
    # Records of each run are stored under
    # `records_directory / experiment_name`, defaulting to
    # `output_directory / "records"` and the data directory's name.
    experiment_name: str | None = None
    records_directory: Path | None = None
    update_report: bool = True
    encoding: str = data.ONEHOT_ENCODING_STR
    entangler: str = "basic"
    measurement: str = "probability"
    diff_method: str = "best"
    layer_quantity: int = 1
    epochs: int = 100
    learning_rate: float = 0.0001
    l1_regularizer: float = 0.0
    l2_regularizer: float = 0.0
    batch_size: int = 128
    report_every: int = 1
    device_name: str = "lightning.qubit"
    tokenizer: AutoTokenizer | None = None
    embedding_qubit_quantity: int | None = None
    vocabulary_size: int | None = None
    fcl_quantity: int | None = 1
    shots: int = -1
    distributed: bool = False
    gpu_quantity: int = 0
    model_filepath: Path | None = None
    log_directory: Path | None = None
    log_level: int = logging.INFO


def _get_free_port() -> int:
    """
    Finds and returns a free port on the server.

    Returns
    -------
    The free port number.

    """
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(("", 0))
    return free_socket.getsockname()[1]


def _ddp_setup(rank: int, world_size: int, master_port: int) -> None:
    """
    Set up the Torch Cuda multiprocessing environment.

    Args
    ----
    rank: The rank of the device.
    world_size: The number of devices in the processing group.

    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _strongly_entangled_run_circuit(
    parameters: NDArray[float],
    sequence: list[dna.Nucleotide],
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
    logger.debug(f"Training on sequence {sequence!s}")
    data_register_size: int = 4
    address_register_size: int = qcpg_models.calculate_address_register_size(
        len(sequence)
    )
    circuit_width: int = address_register_size + data_register_size
    device = qml.device("default.qubit", wires=circuit_width)

    @qml.qnode(device)
    def _run_circuit(
        sequence: list[dna.Nucleotide],
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
    sequences: list[list[dna.Nucleotide]],
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
    sequences: list[list[dna.Nucleotide]],
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
    logger.debug("Executing circuit.")
    loss_value, grads = jax.value_and_grad(
        _strongly_entangled_run_calculate_loss
    )(parameters, sequences, methylations)
    logger.debug("Updating parameters.")
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    parameters = optax.apply_updates(parameters, updates)

    return parameters, optimizer_state, loss_value


def _evaluate_test_performance(
    parameters: NDArray[float],
    test_sequences: list[list[dna.Nucleotide]],
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
    training_sequences: list[list[dna.Nucleotide]],
    training_methylations: NDArray[int],
    test_sequences: list[list[dna.Nucleotide]],
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
        loss_history: list[float] = []
        metrics_history: list[metrics.Metrics] = []
        opt_state = optimizer.init(parameters)
        for iteration in range(max_steps):
            logger.info(f"Training loop {iteration}")
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
                f"{loss_value!s},"
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


def _observed_loss(outputs: Tensor, labels: Tensor) -> Tensor | None:
    """
    Compute the binary cross-entropy over the observed labels only.

    Labels of sites not covered in an experiment, stored as NaN, are left
    out of the loss.

    Args
    ----
    outputs: The predicted methylation probabilities.
    labels: The methylation labels, with NaN for unobserved labels.

    Returns
    -------
    The mean binary cross-entropy of the observed labels, or None if no
    labels are observed.
    """
    if len(labels.shape) == 1 and len(outputs.shape) > 1:
        outputs = outputs.squeeze(1)

    is_observed: Tensor = ~labels.isnan()
    if not is_observed.any():
        return None

    return nn.functional.binary_cross_entropy(
        outputs[is_observed], labels[is_observed]
    )


@dataclass
class _Datasets:
    """
    The datasets of each chromosome split.

    Attributes
    ----------
    training: The training dataset.
    validation: The validation dataset.
    test: The test dataset, or None if no test chromosomes are given.
    chromosomes: The chromosomes of each split, indexed by split name.
    """

    training: data.QuantumTorchDataset
    validation: data.QuantumTorchDataset
    test: data.QuantumTorchDataset | None
    chromosomes: dict[str, list[str]]


def _chromosome_sort_key(chromosome: str) -> tuple[int, int | str]:
    """
    Order chromosomes numerically, followed by named chromosomes such as X.

    Args
    ----
    chromosome: The chromosome name.

    Returns
    -------
    The sort key.
    """
    return (0, int(chromosome)) if chromosome.isdigit() else (1, chromosome)


def _resolve_chromosomes(
    training_parameters: TrainingParameters,
) -> dict[str, list[str]]:
    """
    Determine the chromosomes of each split and verify their files exist.

    Args
    ----
    training_parameters: The training parameters.

    Returns
    -------
    The chromosomes of each split, indexed by split name.

    Raises
    ------
    FileNotFoundError if a chromosome's file is missing.
    ValueError if no validation chromosomes remain.
    """
    data_directory: Path = training_parameters.data_directory
    training: list[str] = list(training_parameters.training_chromosomes)
    test: list[str] = list(training_parameters.test_chromosomes)
    if training_parameters.validation_chromosomes is None:
        validation: list[str] = sorted(
            (
                data.chromosome_name(filepath)
                for filepath in data_directory.glob("chr*.h5")
                if data.chromosome_name(filepath) not in {*training, *test}
            ),
            key=_chromosome_sort_key,
        )
    else:
        validation = list(training_parameters.validation_chromosomes)

    if not training:
        raise ValueError("No training chromosomes specified.")
    if not validation:
        raise ValueError(
            f"No validation chromosomes: none specified, and no chromosome "
            f"files in {data_directory} remain after the training and test "
            "chromosomes."
        )

    missing_filepaths: list[str] = [
        str(data_directory / f"chr{chromosome}.h5")
        for chromosome in dict.fromkeys([*training, *validation, *test])
        if not (data_directory / f"chr{chromosome}.h5").exists()
    ]
    if missing_filepaths:
        raise FileNotFoundError(
            f"Chromosome files not found: {', '.join(missing_filepaths)}"
        )

    return {
        records.TRAINING_SPLIT: training,
        records.VALIDATION_SPLIT: validation,
        records.TEST_SPLIT: test,
    }


def _load_datasets(training_parameters: TrainingParameters) -> _Datasets:
    """
    Load the training, validation and test datasets.

    Args
    ----
    training_parameters: The training parameters.

    Returns
    -------
    The datasets of each split.

    Raises
    ------
    ValueError if the splits hold labels of different experiments.
    """
    chromosomes: dict[str, list[str]] = _resolve_chromosomes(
        training_parameters
    )
    datasets: dict[str, data.QuantumTorchDataset | None] = {}
    for split, split_chromosomes in chromosomes.items():
        if not split_chromosomes:
            datasets[split] = None
            continue

        dataset = data.QuantumTorchDataset(
            [
                training_parameters.data_directory / f"chr{chromosome}.h5"
                for chromosome in split_chromosomes
            ],
            encoding=training_parameters.encoding,
            tokenizer=training_parameters.tokenizer,
        )
        logger.info(
            "Loaded %d samples from %s chromosomes %s",
            len(dataset),
            split,
            ", ".join(split_chromosomes),
        )
        if split != records.TRAINING_SPLIT:
            data.check_experiment_names(
                datasets[records.TRAINING_SPLIT].experiment_names,
                dataset.experiment_names,
                "the training chromosomes",
                f"the {split} chromosomes",
            )
        datasets[split] = dataset

    return _Datasets(
        training=datasets[records.TRAINING_SPLIT],
        validation=datasets[records.VALIDATION_SPLIT],
        test=datasets[records.TEST_SPLIT],
        chromosomes=chromosomes,
    )


def _prepare_training(
    training_parameters: TrainingParameters,
    rank: int | None = None,
    datasets: _Datasets | None = None,
) -> tuple[
    DataLoader,
    DataLoader,
    qcpg_models.QNN | DistributedDataParallel,
    optim.Optimizer,
]:
    if datasets is None:
        datasets = _load_datasets(training_parameters)

    training_dataset: data.QuantumTorchDataset = datasets.training
    validation_dataset: data.QuantumTorchDataset = datasets.validation
    logger.debug(
        "Training set shape %s", str(tuple(training_dataset.sequences.shape))
    )
    if training_parameters.encoding in ["token", "bpe"]:
        encoding: str = "token"
    else:
        encoding = training_parameters.encoding

    logger.debug(
        "Validation set shape %s",
        str(tuple(validation_dataset.sequences.shape)),
    )
    sequence_length: int = training_dataset.sequence_length
    output_shape: int = training_dataset.experiment_quantity
    model: qcpg_models.QNN | DistributedDataParallel = qcpg_models.QNN(
        sequence_length,
        training_parameters.layer_quantity,
        output_shape,
        entangling=training_parameters.entangler,
        encoding=encoding,
        embedding_qubit_quantity=training_parameters.embedding_qubit_quantity,
        vocabulary_size=training_parameters.vocabulary_size,
        fcl_quantity=training_parameters.fcl_quantity,
        measurement=training_parameters.measurement,
        device_name=training_parameters.device_name,
        distribute=training_parameters.distributed,
        diff_method=training_parameters.diff_method,
    )
    if rank is not None:
        model = model.to(rank)
        pin_memory: bool = True
        if training_parameters.gpu_quantity > 1:
            training_sampler: DistributedSampler | None = DistributedSampler(
                training_dataset
            )
            validation_sampler: DistributedSampler | None = DistributedSampler(
                validation_dataset
            )
            model = DistributedDataParallel(model, device_ids=[rank])
        else:
            training_sampler = None
            validation_sampler = None
    else:
        pin_memory = False
        training_sampler = None
        validation_sampler = None

    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=training_parameters.batch_size,
        shuffle=training_sampler is None,
        sampler=training_sampler,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=training_parameters.batch_size,
        shuffle=False,
        sampler=validation_sampler,
        pin_memory=pin_memory,
    )
    return (
        training_loader,
        validation_loader,
        model,
        optim.Adam(model.parameters(), lr=training_parameters.learning_rate),
    )


def _train_one_epoch(
    model: nn.Module,
    epoch: int,
    training_loader: DataLoader,
    optimizer: optim.Optimizer,
    training_parameters: TrainingParameters,
    *,
    rank: int | None = None,
) -> float:
    accumulated_loss: float = 0.0
    accumulated_batches: int = 0
    return_loss: float = 0.0
    model.train(True)
    if training_parameters.gpu_quantity > 1:
        training_loader.sampler.set_epoch(epoch)

    for batch_index, batch_data in enumerate(training_loader):
        inputs, labels = batch_data
        if rank is not None:
            inputs = inputs.to(rank)
            labels = labels.to(rank)
        optimizer.zero_grad()
        outputs: Tensor = model(inputs)
        loss: Tensor | None = _observed_loss(outputs, labels)
        if loss is None:
            continue

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
        accumulated_batches += 1
        if (
            batch_index % training_parameters.report_every == 0
            and batch_index != 0
        ):
            return_loss = accumulated_loss / accumulated_batches
            logger.info(
                "Epoch %d - Batch %d loss: %.4f",
                epoch,
                batch_index,
                return_loss,
            )
            accumulated_loss = 0.0
            accumulated_batches = 0

    return return_loss


@dataclass
class _Evaluation:
    """
    The outputs of evaluating a model on a dataset.

    Attributes
    ----------
    loss: The mean loss over batches with observed labels, NaN if none.
    auc: The AUC of all observed labels pooled, NaN if undefined.
    predictions: The predictions, one row per sample in dataset order and one
        column per experiment.
    labels: The labels, in the same layout, NaN if unobserved.
    """

    loss: float
    auc: float
    predictions: NDArray[np.float32]
    labels: NDArray[np.float32]


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    training_parameters: TrainingParameters,
    *,
    rank: int | None = None,
) -> _Evaluation:
    """
    Evaluate a model on every sample of a loader.

    Args
    ----
    model: The model.
    loader: The samples to evaluate, in the order to report them.
    training_parameters: The training parameters.
    rank: The device to evaluate on, if a GPU.

    Returns
    -------
    The loss, pooled AUC, predictions and labels.
    """
    accumulated_loss: float = 0.0
    loss_batches: int = 0
    model.eval()
    all_outputs: list[Tensor] = []
    all_labels: list[Tensor] = []
    with torch.no_grad():
        for inputs, labels in loader:
            if rank is not None:
                inputs = inputs.to(rank)
                labels = labels.to(rank)
            outputs: Tensor = model(inputs)

            all_outputs.append(outputs.detach().cpu().reshape(len(outputs), -1))
            all_labels.append(labels.detach().cpu().reshape(len(labels), -1))

            loss: Tensor | None = _observed_loss(outputs, labels)
            if loss is None:
                continue

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

            accumulated_loss += loss.item()
            loss_batches += 1

    predictions: NDArray[np.float32] = torch.cat(all_outputs).numpy()
    labels_array: NDArray[np.float32] = torch.cat(all_labels).numpy()
    if loss_batches == 0:
        logger.warning("Evaluation set has no observed labels.")
        return _Evaluation(
            float("nan"), float("nan"), predictions, labels_array
        )

    auc: float = metrics.cell_metrics(predictions, labels_array).auc
    if np.isnan(auc):
        logger.warning("Evaluation set is single-class; AUC is undefined.")

    return _Evaluation(
        accumulated_loss / loss_batches, auc, predictions, labels_array
    )


def _evaluate_validation_set(
    model: nn.Module,
    validation_loader: DataLoader,
    training_parameters: TrainingParameters,
    *,
    rank: int | None = None,
) -> tuple[float, float]:
    evaluation: _Evaluation = _evaluate(
        model, validation_loader, training_parameters, rank=rank
    )
    return evaluation.loss, evaluation.auc


def _train_all_epochs(
    model: nn.Module | DistributedDataParallel,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: optim.Optimizer,
    output_directory: Path,
    training_parameters: TrainingParameters,
    rank: int | None = None,
    run_directory: Path | None = None,
) -> None:
    for epoch in range(training_parameters.epochs):
        logger.info(f"Training epoch {epoch}")
        average_training_loss: float = _train_one_epoch(
            model,
            epoch,
            training_loader,
            optimizer,
            training_parameters,
            rank=rank,
        )
        evaluation: _Evaluation = _evaluate(
            model, validation_loader, training_parameters, rank=rank
        )
        logger.info(
            f"Epoch {epoch} Training loss: {average_training_loss}, Validation loss: "
            f"{evaluation.loss}, Validation AUC: {evaluation.auc}"
        )
        if run_directory is not None:
            cell_results, _ = metrics.experiment_metrics(
                evaluation.predictions, evaluation.labels
            )
            records.append_epoch(
                run_directory,
                epoch,
                average_training_loss,
                evaluation.loss,
                evaluation.auc,
                [cell_result.auc for cell_result in cell_results],
            )
        if training_parameters.gpu_quantity > 1 and rank == 0:
            torch.save(model.module.state_dict(), output_directory / "model.pt")
        else:
            torch.save(model.state_dict(), output_directory / "model.pt")


def _experiment_name(training_parameters: TrainingParameters) -> str:
    """
    Name the genomic experiment a run trains on.

    Args
    ----
    training_parameters: The training parameters.

    Returns
    -------
    The given experiment name, or the data directory's name.
    """
    if training_parameters.experiment_name:
        return training_parameters.experiment_name

    return training_parameters.data_directory.resolve().name


def _records_directory(training_parameters: TrainingParameters) -> Path:
    """
    Find the root folder of the run records.

    Args
    ----
    training_parameters: The training parameters.

    Returns
    -------
    The given records directory, or `output_directory / "records"`.
    """
    if training_parameters.records_directory is not None:
        return training_parameters.records_directory

    return training_parameters.output_directory / "records"


def _parameters_to_record(
    training_parameters: TrainingParameters,
) -> dict[str, object]:
    """
    Collect the training parameters to store in a run record.

    Args
    ----
    training_parameters: The training parameters.

    Returns
    -------
    The parameters, with the tokenizer replaced by its name.
    """
    parameters: dict[str, object] = {}
    for parameter in fields(training_parameters):
        value: object = getattr(training_parameters, parameter.name)
        if parameter.name == "tokenizer" and value is not None:
            value = getattr(value, "name_or_path", type(value).__name__)
        parameters[parameter.name] = value

    return parameters


def _update_report(
    training_parameters: TrainingParameters, run_directory: Path
) -> None:
    """
    Add a run to the HTML report, without letting a failure stop training.

    Args
    ----
    training_parameters: The training parameters.
    run_directory: The run's record folder.
    """
    if not training_parameters.update_report:
        return

    try:
        report.add_run(_records_directory(training_parameters), run_directory)
    except Exception:
        logger.exception("Could not add run %s to the report", run_directory)


def _start_run_record(
    training_parameters: TrainingParameters, datasets: _Datasets
) -> Path:
    """
    Create a run's record and list it in the report as running.

    Args
    ----
    training_parameters: The training parameters.
    datasets: The datasets of each split.

    Returns
    -------
    The run's record folder.
    """
    run_directory: Path = records.create_run_record(
        _records_directory(training_parameters),
        _experiment_name(training_parameters),
        _parameters_to_record(training_parameters),
        datasets.chromosomes,
        datasets.training.experiment_names or [UNNAMED_EXPERIMENT],
    )
    logger.info("Recording run in %s", run_directory)
    _update_report(training_parameters, run_directory)
    return run_directory


def _record_final_evaluation(
    model: nn.Module | DistributedDataParallel,
    datasets: _Datasets,
    training_parameters: TrainingParameters,
    run_directory: Path,
    rank: int | None = None,
) -> None:
    """
    Evaluate the final model on the training and test splits and record it.

    Every sample is evaluated in dataset order, so each output row matches the
    dataset's chromosome and position of that sample.

    Args
    ----
    model: The trained model.
    datasets: The datasets of each split.
    training_parameters: The training parameters.
    run_directory: The run's record folder.
    rank: The device to evaluate on, if a GPU.
    """
    # Evaluate the underlying model so a distributed model does not wait on
    # the other processes.
    if isinstance(model, DistributedDataParallel):
        model = model.module

    for split, dataset in [
        (records.TRAINING_SPLIT, datasets.training),
        (records.TEST_SPLIT, datasets.test),
    ]:
        if dataset is None:
            continue

        evaluation: _Evaluation = _evaluate(
            model,
            DataLoader(
                dataset,
                batch_size=training_parameters.batch_size,
                shuffle=False,
            ),
            training_parameters,
            rank=rank,
        )
        records.write_split_outputs(
            run_directory,
            split,
            records.SplitOutputs(
                chromosome_names=dataset.chromosome_names,
                chromosome_indices=dataset.chromosome_indices,
                positions=dataset.positions,
                labels=evaluation.labels,
                predictions=evaluation.predictions,
            ),
        )
        cell_results, pooled_result = metrics.experiment_metrics(
            evaluation.predictions, evaluation.labels
        )
        records.write_metrics(
            run_directory, split, evaluation.loss, cell_results, pooled_result
        )
        logger.info(
            "Final %s loss: %s, AUC: %s",
            split,
            evaluation.loss,
            pooled_result.auc,
        )

    torch.save(model.state_dict(), run_directory / records.MODEL_FILENAME)


def _train_and_record(
    model: nn.Module | DistributedDataParallel,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: optim.Optimizer,
    training_parameters: TrainingParameters,
    datasets: _Datasets,
    *,
    rank: int | None = None,
    record: bool = True,
) -> None:
    """
    Train a model and keep a record of the run.

    Args
    ----
    model: The model.
    training_loader: The training samples.
    validation_loader: The validation samples.
    optimizer: The optimizer.
    training_parameters: The training parameters.
    datasets: The datasets of each split.
    rank: The device to train on, if a GPU.
    record: Keep a record of the run. Only one process of a distributed run
        records it.
    """
    run_directory: Path | None = (
        _start_run_record(training_parameters, datasets) if record else None
    )
    status: str = records.STATUS_FAILED
    try:
        _train_all_epochs(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            output_directory=training_parameters.output_directory,
            training_parameters=training_parameters,
            rank=rank,
            run_directory=run_directory,
        )
        if run_directory is not None:
            _record_final_evaluation(
                model, datasets, training_parameters, run_directory, rank=rank
            )
        status = records.STATUS_COMPLETED
    finally:
        if run_directory is not None:
            records.finish_run(run_directory, status)
            _update_report(training_parameters, run_directory)


def _single_distributed_gpu_train_qcpg_circuit(
    rank: int,
    world_size: int,
    master_port: int,
    training_parameters: TrainingParameters,
) -> None:
    _ddp_setup(rank, world_size, master_port)
    logging.basicConfig(
        filename=training_parameters.log_directory / "qcpg_train.log",
        level=training_parameters.log_level,
        format=f"[Rank {rank}] " + "%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Setting up training on GPU %d", rank)
    datasets: _Datasets = _load_datasets(training_parameters)
    training_loader, validation_loader, model, optimizer = _prepare_training(
        training_parameters, rank=rank, datasets=datasets
    )
    logger.info("Finished training setup on GPU %d", rank)
    _train_and_record(
        model,
        training_loader,
        validation_loader,
        optimizer,
        training_parameters,
        datasets,
        rank=rank,
        record=rank == 0,
    )
    destroy_process_group()


def _multi_gpu_train_qcpq_circuit(
    training_parameters: TrainingParameters,
) -> None:
    if torch.cuda.device_count() < training_parameters.gpu_quantity:
        raise RuntimeError(
            "%d GPUs requested. Only found %d",
            training_parameters.gpu_quantity,
            torch.cuda.device_count(),
        )

    world_size: int = training_parameters.gpu_quantity
    logger.info("Found %d GPUs", world_size)
    if world_size > 1:
        mp.spawn(
            _single_distributed_gpu_train_qcpg_circuit,
            args=(world_size, _get_free_port(), training_parameters),
            nprocs=world_size,
        )
    else:
        raise RuntimeError("This function requires more than 1 GPU to run.")


def _set_up_logging(training_parameters: TrainingParameters) -> None:
    logging.basicConfig(
        filename=training_parameters.log_directory / "qcpg_train.log",
        level=training_parameters.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Using the following training parameters:")
    logger.info("\tEntangler: %s", training_parameters.entangler)
    logger.info("\tEncoding: %s", training_parameters.encoding)
    logger.info("\tMeasurement: %s", training_parameters.measurement)
    logger.info("\tLayer quantity: %d", training_parameters.layer_quantity)
    logger.info("\tLearning rate: %f", training_parameters.learning_rate)
    logger.info("\tEpochs: %d", training_parameters.epochs)
    logger.info("\tL1 regularization: %f", training_parameters.l1_regularizer)
    logger.info("\tL2 regularization: %f", training_parameters.l2_regularizer)


def train_qnn_circuit(training_parameters: TrainingParameters) -> None:
    _set_up_logging(training_parameters)

    if training_parameters.gpu_quantity in [0, 1]:
        if training_parameters.gpu_quantity == 1:
            rank: int | None = 0
        else:
            rank = None
        datasets: _Datasets = _load_datasets(training_parameters)
        training_loader, validation_loader, model, optimizer = (
            _prepare_training(training_parameters, rank=rank, datasets=datasets)
        )
        _train_and_record(
            model,
            training_loader,
            validation_loader,
            optimizer,
            training_parameters,
            datasets,
            rank=rank,
        )
    elif training_parameters.gpu_quantity > 1:
        _multi_gpu_train_qcpq_circuit(training_parameters)
    else:
        raise ValueError("Negative GPU quantity specified.")


def train_rqnn_circuit(training_parameters: TrainingParameters) -> None:
    _set_up_logging(training_parameters)
