import logging
from collections.abc import Sequence
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from qenetics.tools.data import (
    UNIQUE_NUCLEOTIDE_QUANTITY,
    nucleotide_array_to_numpy,
)

logger = logging.getLogger(__name__)


def read_quantity_examples_per_chromosome(
    deepcpg_directory: Path,
) -> dict[str, int]:
    quantity_per_chromosome: dict[str, int] = {}
    for filepath in deepcpg_directory.iterdir():
        split: list[str] = filepath.stem.split("_")
        chromosome: str = split[0]
        max_example_quantity = int(split[1].split("-")[1])
        if chromosome in quantity_per_chromosome:
            quantity_per_chromosome[chromosome] = max(
                quantity_per_chromosome[chromosome], max_example_quantity
            )
        else:
            quantity_per_chromosome[chromosome] = max_example_quantity

    return quantity_per_chromosome


def _determine_sequence_length(deepcpg_directory: Path) -> int:
    with h5py.File(next(iter(deepcpg_directory.iterdir()))) as dataset:
        return dataset["inputs"]["dna"].shape[1]


def extract_deepcpg_experiment_to_qcpg(
    deepcpg_directory: Path,
    qcpg_directory: Path,
    experiment_name: str,
    threshold: float = -1.0,
) -> None:
    if not deepcpg_directory.is_dir():
        raise ValueError("Filepath must be a directory: %s", deepcpg_directory)

    if not qcpg_directory.is_dir():
        raise ValueError("Filepath must be a directory: %s", qcpg_directory)

    if threshold == -1.0:
        polars_truth_dtype = pl.Float32
        h5_truth_dtype = "f4"
    else:
        polars_truth_dtype = pl.Int8
        h5_truth_dtype = "i1"

    sequence_length: int = _determine_sequence_length(deepcpg_directory)
    logger.info("Found sequences of length %d", sequence_length)
    schema: dict[str, pl.Array | pl.Float64] = {
        "methylation_sequences": pl.Array(
            pl.Int8, (sequence_length, UNIQUE_NUCLEOTIDE_QUANTITY)
        ),
        "methylation_ratios": pl.Float64,
    }
    chromosomes: set[str] = {
        filepath.stem.split("_")[0][1:]
        for filepath in deepcpg_directory.iterdir()
    }
    logger.info("Found chromosomes %s", str(chromosomes))
    current_data = pl.DataFrame(schema=schema)
    for chromosome in chromosomes:
        current_data = current_data.clear()
        deepcpg_filepaths: list[Path] = [
            Path(filepath)
            for filepath in glob(str(deepcpg_directory / f"c{chromosome}_*.h5"))
        ]
        for filepath in deepcpg_filepaths:
            logger.debug("Processing file %s", str(filepath))
            with h5py.File(filepath) as deepcpg_dataset:
                if experiment_name not in deepcpg_dataset["outputs"]:
                    raise RuntimeError(
                        "Experiment %s not found in file %s",
                        experiment_name,
                        filepath,
                    )

                methylation_sequences = pl.Series(
                    [
                        nucleotide_array_to_numpy(sequence)
                        for sequence in deepcpg_dataset["inputs"]["dna"]
                    ],
                    dtype=pl.Array(
                        pl.Int64, (sequence_length, UNIQUE_NUCLEOTIDE_QUANTITY)
                    ),
                )
                methylation_ratios = pl.Series(
                    np.array(
                        deepcpg_dataset["outputs"][experiment_name], dtype=float
                    ),
                )
                if len(methylation_ratios) != len(methylation_sequences):
                    raise RuntimeError(
                        "Found %d sequences but %d ratios in file %s",
                        len(methylation_sequences),
                        len(methylation_ratios),
                        str(filepath),
                    )
                current_data = pl.concat(
                    [
                        current_data,
                        pl.DataFrame(
                            [methylation_sequences, methylation_ratios],
                            schema=schema,
                        ).filter(pl.col("methylation_ratios") != -1.0),
                    ]
                )

        logger.info("Samples found after filtering: %d", len(current_data))

        if polars_truth_dtype == pl.Int8:
            current_data = current_data.with_columns(
                pl.when(pl.col("methylation_ratios") >= threshold)
                .then(1)
                .otherwise(0)
                .alias("rounded_methylation_ratios")
            )
            current_data = current_data.drop("methylation_ratios")
            current_data.rename(
                {"rounded_methylation_ratios": "methylation_ratios"}
            )

        with h5py.File(qcpg_directory / f"chr{chromosome}.h5", "w") as qcpg_fd:
            qcpg_fd.create_dataset(
                "methylation_sequences",
                shape=(
                    len(current_data["methylation_sequences"]),
                    sequence_length,
                    UNIQUE_NUCLEOTIDE_QUANTITY,
                ),
                dtype="i1",
                data=current_data["methylation_sequences"],
            )
            qcpg_fd.create_dataset(
                "methylation_ratios",
                shape=len(current_data["methylation_ratios"]),
                dtype=h5_truth_dtype,
                data=current_data["methylation_ratios"],
            )


def _one_hot_to_integer(
    one_hot_encoding: Sequence, *, include_zero: bool = False
) -> np.ndarray:
    for index, value in enumerate(one_hot_encoding):
        if value == 1:
            if include_zero:
                return index + 1
            else:
                return index

    if not include_zero:
        raise ValueError(
            f"Improper one-hot encoding for array {one_hot_encoding}"
        )

    return 0


def _one_hot_sequence_to_integers(
    one_hot_sequences: Sequence, *, include_zero: bool = False
) -> np.ndarray:
    return np.array(
        [
            _one_hot_to_integer(encoding, include_zero=include_zero)
            for encoding in one_hot_sequences
        ],
        dtype=np.int8,
    )


def h5_one_hot_to_integer(
    input_filepath: Path, output_directory: Path, *, include_zero: bool = False
) -> None:
    methylation_sequences_str: str = "methylation_sequences"
    methylation_ratios_str: str = "methylation_ratios"
    output_filepath: Path = output_directory / input_filepath.name
    with (
        h5py.File(input_filepath) as input_dataset,
        h5py.File(output_filepath, "w") as output_fd,
    ):
        if isinstance(input_dataset[methylation_ratios_str], h5py.Group):
            ratios_group: h5py.Group = output_fd.create_group(
                methylation_ratios_str
            )
            for experiment_name in input_dataset[methylation_ratios_str]:
                ratios_group.create_dataset(
                    experiment_name,
                    data=input_dataset[methylation_ratios_str][experiment_name],
                )
        else:
            output_fd.create_dataset(
                methylation_ratios_str,
                data=input_dataset[methylation_ratios_str],
            )

        samples_quantity: int = input_dataset[methylation_sequences_str].shape[
            0
        ]
        sequence_length: int = input_dataset[methylation_sequences_str].shape[1]
        output_sequences: h5py.Dataset = output_fd.create_dataset(
            methylation_sequences_str,
            shape=(samples_quantity, sequence_length),
            dtype="i8",
        )
        for index, sample in enumerate(
            input_dataset[methylation_sequences_str]
        ):
            output_sequences[index] = _one_hot_sequence_to_integers(
                sample, include_zero=include_zero
            )
