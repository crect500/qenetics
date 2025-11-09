from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from io import TextIOBase
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from qenetics.tools import dna, methylation

logger = logging.getLogger(__name__)

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


@dataclass
class ChromosomeIndices:
    start: int
    end: int


class H5CpGDataset(Dataset):
    def __init__(
        self: H5CpGDataset,
        filepaths: list[Path],
        threshold: float = 0.5,
    ) -> None:
        self.file_list = filepaths
        with h5py.File(filepaths[0]) as fd:
            self.sequence_length = fd["methylation_sequences"].shape[1]
            self.experiment_names = fd["methylation_ratios"].keys()
            self.experiment_quantity = len(self.experiment_names)

        self.chromosome_indices = {}
        self._allocate_tensors(filepaths)
        self._fill_tensors(filepaths, threshold)

    def __len__(self: H5CpGDataset) -> int:
        return len(self.data)

    def __getitem__(
        self: H5CpGDataset, idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]

    def _allocate_tensors(self: H5CpGDataset, filepaths: list[Path]) -> None:
        sample_quantity: int = 0
        for filepath in filepaths:
            with h5py.File(filepath) as fd:
                new_sample_quantity: int = (
                    sample_quantity + fd["methylation_sequences"].shape[0]
                )
                self.chromosome_indices[filepath.stem[3:]] = ChromosomeIndices(
                    start=sample_quantity, end=new_sample_quantity
                )
                sample_quantity = new_sample_quantity

        self.data = torch.empty(
            sample_quantity,
            self.sequence_length,
            UNIQUE_NUCLEOTIDE_QUANTITY,
            dtype=torch.float,
        )
        self.labels = torch.empty(
            sample_quantity, self.experiment_quantity, dtype=torch.float
        )

    def _fill_tensors(
        self: H5CpGDataset, filepaths: list[Path], threshold: float = 0.5
    ) -> None:
        for filepath in filepaths:
            chromosome: str = filepath.stem[3:]
            with h5py.File(filepath) as fd:
                self.data[
                    self.chromosome_indices[
                        chromosome
                    ].start : self.chromosome_indices[chromosome].end,
                    :,
                    :,
                ] = torch.tensor(fd["methylation_sequences"])
                for label_index, experiment_name in enumerate(
                    fd["methylation_ratios"].keys()
                ):
                    self.labels[
                        self.chromosome_indices[
                            chromosome
                        ].start : self.chromosome_indices[chromosome].end,
                        label_index,
                    ] = torch.tensor(
                        [
                            0 if methylation_ratio < threshold else 1
                            for methylation_ratio in fd["methylation_ratios"][
                                experiment_name
                            ]
                        ],
                        dtype=torch.float,
                    )


def nucleotide_character_to_numpy(nucleotide: str) -> NDArray[int]:
    """
    Convert a nucleotide designator to a one-hot array.

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
    if nucleotide == "C":
        return np.array([0, 0, 1, 0], dtype=int)
    if nucleotide == "G":
        return np.array([0, 0, 0, 1], dtype=int)

    raise ValueError(f"{nucleotide} is not a valid nucleotide designator")


def nucleotide_string_to_numpy(sequence: str) -> NDArray[int] | None:
    """
    Convert a list of ASCII nucleotide designators to one-hot arrays.

    Args
    ----
    sequence: A list of ASCII nucleotide designators.

    Returns
    -------
    A matrix of one-hot encoded values.
    """
    if "N" in sequence:
        return None
    else:
        return np.array(
            [
                nucleotide_character_to_numpy(nucleotide)
                for nucleotide in sequence
            ],
            dtype=int,
        )


def samples_to_numpy(
    methylation_filepath: Path, threshold: float = 0.5
) -> tuple[NDArray[int], NDArray[int]]:
    """
    Create input and truth samples for sequences of nucleotides and their methylations.

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
        ]
        return np.array(
            [row[0] for row in read_data if row[0] is not None], dtype=int
        ), np.array(
            [row[1] for row in read_data if row[0] is not None], dtype=int
        )


def _increment_or_create_entry(dictionary: dict[Any, ...], key: Any) -> None:
    entry: Any | None = dictionary[key]
    if entry is None:
        dictionary[key] = 1
    else:
        dictionary[key] += 1


def _validate_sequence(sequence: str) -> bool:
    middle_index: int = int(len(sequence) / 2) - 1
    if sequence[middle_index : middle_index + 2] != "CG" or "N" in sequence:
        return False

    return True


def _retrieve_chromosome_sequences(
    profiles_by_position: dict[int, dict[str, float]],
    chromosome: str,
    fasta_file_descriptor: TextIOBase,
    fasta_metadata,
    fasta_line_length: int,
    sequence_length: int,
    experiment_names: list[str],
) -> tuple[NDArray[bool], NDArray[float]]:
    unique_nucleotide_quantity: int = 4
    sequences = np.ndarray(
        (0, sequence_length, unique_nucleotide_quantity), dtype=int
    )
    experiment_mapping: dict[str, int] = {
        experiment_name: index
        for index, experiment_name in enumerate(experiment_names)
    }
    methylation_ratios = np.ndarray((0, len(experiment_names)))
    for position, profile_by_experiment in profiles_by_position.items():
        sequence: str | None = dna.find_methylation_sequence(
            chromosome=chromosome,
            position=position,
            genome_metadata=fasta_metadata,
            file_descriptor=fasta_file_descriptor,
            sequence_length=sequence_length,
            line_length=fasta_line_length,
        )
        if sequence is None or not _validate_sequence(sequence):
            continue
        else:
            sequences = np.append(
                sequences,
                nucleotide_string_to_numpy(sequence).reshape(
                    1, sequence_length, unique_nucleotide_quantity
                ),
                axis=0,
            )

        ratios_by_experiment: np.ndarray = np.array(
            [np.nan] * len(experiment_names)
        ).reshape(1, len(experiment_names))
        for experiment_name, methylation_ratio in profile_by_experiment.items():
            ratios_by_experiment[0][experiment_mapping[experiment_name]] = (
                methylation_ratio
            )

        methylation_ratios = np.append(
            methylation_ratios, ratios_by_experiment, axis=0
        )

    return sequences, methylation_ratios


def _create_h5_dataset(
    h5_filepath: Path,
    sequences: NDArray[bool],
    methylation_ratios: NDArray[float],
    experiment_names: list[str],
) -> None:
    with h5py.File(h5_filepath, "w") as fd:
        (
            fd.create_dataset(
                "methylation_sequences", data=sequences, dtype="i1", chunks=True
            ),
        )
        ratios_dataset: h5py.Group = fd.create_group("methylation_ratios")
        for experiment_index, experiment_name in enumerate(experiment_names):
            ratios_dataset.create_dataset(
                experiment_name,
                shape=(sequences.shape[0],),
                chunks=True,
                dtype="f4",
            )
            for sample_index, experiment_ratios in enumerate(
                methylation_ratios
            ):
                ratios_dataset[experiment_name][sample_index] = (
                    experiment_ratios[experiment_index]
                )


def _create_h5_files(
    profiles_by_chromosome: dict[str, dict[int, dict[str, float]]],
    experiment_names: list[str],
    dataset_directory: Path,
    fasta_filepath: Path,
    sequence_length: int,
) -> None:
    fasta_line_length: int = dna.determine_line_length(fasta_filepath)
    with open(fasta_filepath) as fasta_fd:
        fasta_metadata: dict[str, dna.SequenceInfo] = (
            dna.extract_fasta_metadata(fasta_filepath)
        )
        for chromosome, profiles_by_position in profiles_by_chromosome.items():
            h5_filepath: Path = dataset_directory / ("chr" + chromosome + ".h5")
            sequences, methylation_ratios = _retrieve_chromosome_sequences(
                profiles_by_position=profiles_by_position,
                chromosome=chromosome,
                fasta_file_descriptor=fasta_fd,
                fasta_metadata=fasta_metadata,
                fasta_line_length=fasta_line_length,
                sequence_length=sequence_length,
                experiment_names=experiment_names,
            )
            _create_h5_dataset(
                h5_filepath=h5_filepath,
                sequences=sequences,
                methylation_ratios=methylation_ratios,
                experiment_names=experiment_names,
            )


def create_h5_dataset_from_methylation_profiles(
    methylation_directory: Path,
    fasta_filepath: Path,
    dataset_directory: Path,
    sequence_length: int,
    minimum_samples: int = 1,
) -> None:
    experiment_names: list[str] = [
        filename.stem.split(".")[0]
        for filename in methylation_directory.iterdir()
    ]
    profiles_by_chromosome: dict[str, dict[int, dict[str, float]]] = (
        methylation.record_methylation_profiles(
            methylation_directory, minimum_samples
        )
    )
    _create_h5_files(
        profiles_by_chromosome=profiles_by_chromosome,
        experiment_names=experiment_names,
        dataset_directory=dataset_directory,
        fasta_filepath=fasta_filepath,
        sequence_length=sequence_length,
    )
