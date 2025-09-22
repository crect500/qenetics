from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import json
from pathlib import Path

from qenetics.deepcpg import train
from qenetics.tools import cpg_sampler


@dataclass
class RunConfig:
    training_filepath: Path
    validation_filepath: Path
    output_filepath: Path


def _parse_script_args() -> Namespace:
    parser = ArgumentParser("train_dna")
    parser.add_argument(
        "-c", "--config", dest="config_filepath", type=Path, required=True
    )

    return parser.parse_args()


def _parse_config(
    config_filepath: Path,
) -> tuple[RunConfig, train.TrainingConfig]:
    with open(config_filepath) as fd:
        config: dict[str, str | float | int | list[float]] = json.load(fd)

    return RunConfig(
        training_filepath=Path(config["training_filepath"]),
        validation_filepath=Path(config["validation_filepath"]),
        output_filepath=Path(config["output_filepath"]),
    ), train.TrainingConfig(
        layer_decays=config["layer_decays"],
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
    )


def train_deepcpg(
    run_config: RunConfig, training_config: train.TrainingConfig
) -> None:
    training_sequences, training_methylation = cpg_sampler.samples_to_numpy(
        run_config.training_filepath
    )
    validation_sequences, validation_methylation = cpg_sampler.samples_to_numpy(
        run_config.validation_filepath
    )
    train.train_model(
        training_sequences=training_sequences,
        training_methylations=training_methylation,
        validation_sequences=validation_sequences,
        validation_methylation=validation_methylation,
        config=training_config,
        output_filepath=run_config.output_filepath,
    )


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    run_config, training_config = _parse_config(args.config_filepath)
    train_deepcpg(run_config, training_config)
