import logging
from collections.abc import Sequence

from torch import Tensor, nn, optim, save, tensor

from qenetics.qcpg.qcpg import (
    TrainingParameters,
    _non_nan_indices,
    _prepare_training,
)

logger = logging.getLogger(__name__)


def eval_gradient_variance(
    model: nn.Module,
    training_loader,
    optimizer: optim.Optimizer,
    training_parameters: TrainingParameters,
    *,
    stop_after: int | None = None,
) -> Sequence:
    model.train(True)
    gradients: list[list[Tensor]] = []
    for batch_index, batch_data in enumerate(training_loader):
        if stop_after is not None and batch_index == stop_after:
            break

        inputs, labels = batch_data

        optimizer.zero_grad()
        outputs: Tensor = model(inputs)
        outputs: Tensor = model(inputs)
        if len(labels.shape) > 1:
            non_nan_indices = _non_nan_indices(labels)
            loss: Tensor = nn.functional.binary_cross_entropy(
                outputs[non_nan_indices], labels[non_nan_indices]
            )
        else:
            loss: Tensor = nn.functional.binary_cross_entropy(
                outputs.squeeze(1), labels
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

        if (
            batch_index % training_parameters.report_every == 0
            and batch_index != 0
        ):
            logger.info("Training batch %d", batch_index)

        loss.backward()
        batch_gradients: list[Tensor] = []
        for parameter in model.parameters():
            if parameter.grad is not None:
                batch_gradients.append(parameter.grad.detach().data)

        gradients.append(batch_gradients)

    return gradients


def eval_grads(
    training_parameters: TrainingParameters, stop_after: int | None = None
) -> None:
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
    logger.info("\tL1 regularization: %f", training_parameters.l1_regularizer)
    logger.info("\tL2 regularization: %f", training_parameters.l2_regularizer)
    logger.info("\tStoping after %d batches", stop_after)

    training_loader, _, model, optimizer = _prepare_training(
        training_parameters
    )
    gradients = eval_gradient_variance(
        model=model,
        training_loader=training_loader,
        optimizer=optimizer,
        training_parameters=training_parameters,
        stop_after=stop_after,
    )

    save(tensor(gradients, training_parameters.output_filepath / "grads.pt"))
