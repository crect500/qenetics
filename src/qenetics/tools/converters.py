import logging
from collections.abc import Sequence
from csv import DictReader
from glob import glob
from math import sqrt
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from numpy.typing import NDArray

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4

logger = logging.getLogger(__name__)


def nucleotide_character_to_numpy(
    nucleotide: str,
    encoding: str = "amplitude",
) -> NDArray[int]:
    """
    Convert a nucleotide designator to a one-hot array.

    A = [1, 0, 0, 0], T = [0, 1, 0, 0], G = [0, 0, 1, 0], C = [0, 0, 0, 1].

    Args
    ----
    nucleotide: The ASCII nucleotide designator.

    Returns
    -------
    The one-hot encoded array.
    """
    if nucleotide == "A":
        return np.array([1, 0, 0, 0], dtype=int)
    if nucleotide == "T":
        return np.array([0, 1, 0, 0], dtype=int)
    if nucleotide == "G":
        return np.array([0, 0, 1, 0], dtype=int)
    if nucleotide == "C":
        return np.array([0, 0, 0, 1], dtype=int)
    if nucleotide == "N":
        if encoding == "amplitude":
            equal_superposition: float = 1 / sqrt(2)
            return np.array([equal_superposition] * 4, dtype=float)
        else:
            return np.array([0, 0, 0, 0], dtype=int)

    raise ValueError(f"{nucleotide} is not a valid nucleotide designator")


def nucleotide_string_to_numpy(
    sequence: str, encoding: str = "amplitude"
) -> NDArray[int] | None:
    """
    Convert a list of ASCII nucleotide designators to one-hot arrays.

    Args
    ----
    sequence: A list of ASCII nucleotide designators.

    Returns
    -------
    A matrix of one-hot encoded values.
    """
    return np.array(
        [
            nucleotide_character_to_numpy(nucleotide, encoding)
            for nucleotide in sequence
        ],
        dtype=int,
    )


def nucleotide_integer_to_numpy(nucleotide: int) -> NDArray[int]:
    """
    Convert a nucleotide integer representation to a one-hot array.

    Args
    ----
    nucleotide: The nucleotide integer, with A = 0, T = 1, G = 2, C = 3 and
        -1 for an unknown nucleotide.

    Returns
    -------
    The one-hot encoded array.
    """
    if nucleotide == 0:
        return np.array([1, 0, 0, 0], dtype=int)
    if nucleotide == 1:
        return np.array([0, 1, 0, 0], dtype=int)
    if nucleotide == 2:
        return np.array([0, 0, 1, 0], dtype=int)
    if nucleotide == 3:
        return np.array([0, 0, 0, 1], dtype=int)
    if nucleotide == -1:
        return np.array([0, 0, 0, 0], dtype=int)

    raise ValueError(f"{nucleotide} is not a valid nucleotide designator")


def nucleotide_array_to_numpy(sequence: Sequence[int]) -> NDArray[int] | None:
    """
    Convert a list of nucleotide integer representations to one-hot arrays.

    Args
    ----
    sequence: A list of nucleotide integer representations.

    Returns
    -------
    A matrix of one-hot encoded values.
    """
    return np.array(
        [nucleotide_integer_to_numpy(nucleotide) for nucleotide in sequence],
        dtype=int,
    )


def samples_to_numpy(
    methylation_filepath: Path, threshold: float = 0.5
) -> tuple[NDArray[int], NDArray[int]]:
    """
    Create input and truth samples for sequences of nucleotides and their methylations.

    Sequences holding unknown nucleotides ('N') are skipped.

    Args
    ----
    methylation_filepath: The filepath of a file containing methylation_profiles.
    threshold: The threshold at which to consider a site methylated.

    Returns
    -------
    A matrix of one-hot input sample encodings and the truth values.
    """
    logger.debug(
        f"Loading methylation data from {methylation_filepath} with threshold "
        f"{threshold}."
    )
    with open(methylation_filepath) as fd:
        csv_reader = DictReader(fd)
        read_data: list[tuple[NDArray[int], NDArray[int]]] = [
            (
                nucleotide_string_to_numpy(line["sequence"]),
                np.array(
                    0 if float(line["ratio_methylated"]) < threshold else 1,
                    dtype=int,
                ),
            )
            for line in csv_reader
            if "N" not in line["sequence"]
        ]
        return np.array([row[0] for row in read_data], dtype=int), np.array(
            [row[1] for row in read_data], dtype=int
        )


def _integer_to_nucleotide_char(value: int, allow_N: bool = False) -> str:
    """
    Convert an integer representation to its corresponding nucleotide character.

    Args
    ----
    value: The value to convert.
    allow_N: Allow the character 'N' to represent an unknown nucleotide.

    Returns
    -------
    The nucleotide character.

    Raises
    ------
    ValueError if an invalid integer is provided.
    """
    if value == 0:
        return "A"
    if value == 1:
        return "T"
    if value == 2:
        return "G"
    if value == 3:
        return "C"

    if value == -1:
        if allow_N:
            return "N"
        else:
            raise ValueError(
                f"Invalid value {value}. To allow nucleotide value 'N' to be valid, set `allow_N` to True"
            )

    raise ValueError(f"Invalid value {value}")


def integer_array_to_nucleotide_str(
    values: Sequence[int], allow_N: bool = False
) -> str:
    """
    Convert an integer array to its corresponding nucleotide sequence string.

    Args
    ----
    value: The array to convert.
    allow_N: Allow the character 'N' to represent an unknown nucleotide.

    Returns
    -------
    The nucleotide string.
    """
    return "".join(
        [
            _integer_to_nucleotide_char(value, allow_N=allow_N)
            for value in values
        ]
    )


def _one_hot_to_nucleotide(
    one_hot_array: Sequence[int | float], allow_N: bool = False
) -> str:
    """
    Convert an one-hot representation to its corresponding nucleotide character.

    Args
    ----
    value: The one-hot array to convert.
    allow_N: Allow the character 'N' to represent an unknown nucleotide.

    Returns
    -------
    The nucleotide character.

    Raises
    ------
    ValueError if an invalid one-hot encoding is provided.
    """
    if all(one_hot_array == np.array([1, 0, 0, 0])):
        return "A"
    if all(one_hot_array == np.array([0, 1, 0, 0])):
        return "T"
    if all(one_hot_array == np.array([0, 0, 1, 0])):
        return "G"
    if all(one_hot_array == np.array([0, 0, 0, 1])):
        return "C"

    if all(np.array(one_hot_array) == np.array([0, 0, 0, 0])):
        if allow_N:
            return "N"
        else:
            raise ValueError(
                f"Invalid one-hot encoding {one_hot_array}. To allow nucleotide value 'N' to be valid, set `allow_N` to True"
            )

    raise ValueError(f"Invalid one-hot encoding {one_hot_array}")


def one_hot_sequence_to_nucleotide_str(
    one_hot_sequence: Sequence[Sequence[int | float]], allow_N: bool = False
) -> str:
    """
    Convert a one-hot encoded sequence to its corresponding nucleotide sequence string.

    Args
    ----
    value: The one-hot encoded sequence to convert.
    allow_N: Allow the character 'N' to represent an unknown nucleotide.

    Returns
    -------
    The nucleotide string.
    """
    return "".join(
        [
            _one_hot_to_nucleotide(value, allow_N=allow_N)
            for value in one_hot_sequence
        ]
    )


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
    one_hot_encoding: Sequence, *, allow_N: bool = False
) -> np.ndarray:
    for index, value in enumerate(one_hot_encoding):
        if value == 1:
            if allow_N:
                return index + 1
            else:
                return index

    if not allow_N:
        raise ValueError(
            f"Improper one-hot encoding for array {one_hot_encoding}"
        )

    return 0


def one_hot_sequence_to_integers(
    one_hot_sequences: Sequence, *, allow_N: bool = False
) -> np.ndarray:
    return np.array(
        [
            _one_hot_to_integer(encoding, allow_N=allow_N)
            for encoding in one_hot_sequences
        ],
        dtype=np.int8,
    )


def h5_one_hot_to_integer(
    input_filepath: Path, output_directory: Path, *, allow_N: bool = False
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
            output_sequences[index] = one_hot_sequence_to_integers(
                sample, allow_N=allow_N
            )
