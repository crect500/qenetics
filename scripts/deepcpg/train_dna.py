from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import json
from pathlib import Path

from qenetics.deepcpg import train
from qenetics.deepcpg import deepcpg_utils


@dataclass
class RunConfig:
    training_filepath: Path
    validation_filepath: Path
    layer_decays: list[float]
    output_filepath: Path


def _parse_script_args() -> Namespace:
    parser = ArgumentParser("train_dna")
    parser.add_argument(
        "-c", "--config", dest="config_filepath", type=Path, required=True
    )

    return parser.parse_args()


def _parse_config(config_filepath: Path) -> RunConfig:
    with open(config_filepath) as fd:
        config: dict[str, str | list[float]] = json.load(fd)

    return RunConfig(
        training_filepath=Path(config["training_filepath"]),
        validation_filepath=Path(config["validation_filepath"]),
        layer_decays=config["layer_decays"],
        output_filepath=Path(config["output_filepath"]),
    )


def train_deepcpg(config: RunConfig) -> None:
    training_sequences, training_methylation = deepcpg_utils.load_samples(
        config.training_filepath
    )
    validation_sequences, validation_methylation = deepcpg_utils.load_samples(
        config.validation_filepath
    )
    train.train_model(
        training_sequences=training_sequences,
        training_methylations=training_methylation,
        validation_sequences=validation_sequences,
        validation_methylation=validation_methylation,
        layer_decays=config.layer_decays,
        output_filepath=config.output_filepath,
    )


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    config: RunConfig = _parse_config(args.config_filepath)
    train_deepcpg(config)
