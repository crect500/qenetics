from dataclasses import dataclass
import logging
from pathlib import Path

from keras import optimizers
from keras.losses import BinaryCrossentropy
from keras.models import Model
import numpy as np
from numpy.typing import NDArray

from qenetics.deepcpg import models
from qenetics.deepcpg.metrics import CLA_METRICS
from qenetics.tools.cpg_sampler import (
    load_methylation_samples,
    sequence_to_numpy,
)

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
    input_file: Path, config: TrainingConfig, model_filepath: Path | None = None
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
        sequence_length = _get_sequence_length(input_file)
        dna_inputs = dna_model_builder.inputs(sequence_length)
        dna_model = dna_model_builder(dna_inputs)

    output = models.add_output_layers(dna_model.outputs)
    model = Model(input=dna_model.inputs, output=output, name=dna_model.name)
    return model


def train_model(
    train_filepath: Path,
    validation_filepath: Path,
    config: TrainingConfig,
    output_filepath: Path,
    model_filepath: Path | None = None,
) -> None:
    train_sequences, train_methylation = load_methylation_samples(
        train_filepath
    )
    train_input: NDArray[int] = np.array(
        [sequence_to_numpy(sequence) for sequence in train_sequences], dtype=int
    )
    validation_sequences, validation_methylation = load_methylation_samples(
        validation_filepath
    )
    validation_input: NDArray[int] = np.array(
        [sequence_to_numpy(sequence) for sequence in validation_sequences],
        dtype=int,
    )
    model: Model = _build_model(train_filepath, config, model_filepath)
    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer, loss=BinaryCrossentropy(), metrics=CLA_METRICS
    )
    model.fit(
        train_input,
        np.array(train_methylation),
        batch_size=config.batch_size,
        validation_data=(validation_input, np.array(validation_methylation)),
    )
    model.save(output_filepath)
