from dataclasses import dataclass
import logging
from pathlib import Path

from keras import optimizers
from keras.losses import BinaryCrossentropy
from keras.models import Model
from numpy.typing import NDArray

from qenetics.deepcpg import models
from qenetics.deepcpg.metrics import CLA_METRICS

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    layer_decays: list[float]
    dropout_rate: float = 0.0
    batch_size: int = 128
    learning_rate: float = 0.0001
    fine_tune: bool = False


def _get_sequence_length(input_data: NDArray[int]) -> int:
    return input_data.shape[1]


def _build_model(
    input_data: NDArray[int],
    config: TrainingConfig,
    model_filepath: Path | None = None,
) -> Model:
    if model_filepath is not None:
        logger.info("Loading existing DNA model ...")
    else:
        logger.info("Building DNA model ...")
        dna_model_builder = models.CnnL2h128(
            l1_decay=config.layer_decays[0],
            l2_decay=config.layer_decays[1],
            dropout=config.dropout_rate,
        )
        sequence_length = _get_sequence_length(input_data)
        dna_inputs = dna_model_builder.inputs(sequence_length)
        dna_model = dna_model_builder(dna_inputs)

    output = models.add_output_layers(dna_model.outputs)
    model = Model(input=dna_model.inputs, output=output, name=dna_model.name)
    return model


def train_model(
    training_sequences: NDArray[int],
    training_methylations: NDArray[int],
    validation_sequences: NDArray[int],
    validation_methylation: NDArray[int],
    layer_decays: list[float],
    output_filepath: Path,
    model_filepath: Path | None = None,
) -> None:
    config = TrainingConfig(layer_decays)
    model: Model = _build_model(training_sequences, config, model_filepath)
    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer, loss=BinaryCrossentropy(), metrics=CLA_METRICS
    )
    model.fit(
        training_sequences,
        training_methylations,
        batch_size=config.batch_size,
        validation_data=(validation_sequences, validation_methylation),
    )
    model.save(output_filepath)
