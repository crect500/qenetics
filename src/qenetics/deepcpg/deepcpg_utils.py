from __future__ import annotations

import io
from collections.abc import Generator
from dataclasses import dataclass
import logging

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from qenetics.tools import cpg_sampler

logger = logging.getLogger(__name__)


@dataclass
class MethylationInfo:
    """
    Stores information from a methylation experiment data file.

    Attributes
    ----------
    chromosome: The chromosome of the sequence referenced.
    position: The position of the cytosine of the CpG site.
    methylation_ratio: The ratio of sites found to be methylated.
    experiment_count: The number of experiments performed.
    """

    chromosome: str
    position: int
    methylation_ratio: float
    experiment_count: int


def _process_methylation_line(
    line: str, minimum_samples: int = 1
) -> MethylationInfo | None:
    """
    Process one line from a methylation data file.

    Args
    ----
    line: The line read from the methylation file.

    Returns
    -------
    The methylation information, if a numbered chromosome. None otherwise.
    """
    line_split: list[str] = line.rstrip().split()

    count_methylated = int(line_split[4])
    count_unmethylated = int(line_split[5])
    total_experiments: int = count_methylated + count_unmethylated

    if total_experiments < minimum_samples:
        return None

    return MethylationInfo(
        chromosome=line_split[0],
        position=int(line_split[1]),
        methylation_ratio=count_methylated / total_experiments,
        experiment_count=total_experiments,
    )


def retrieve_methylation_data(
    methylation_filepath: Path, minimum_samples: int = 1
) -> Generator[MethylationInfo, None, None]:
    """
    Create a Generator for MethylationInfo objects from the methylation file.

    Args
    ----
    methylation_filepath: The file storing methylation profiles

    Returns
    -------
    Generator for MethylationInfo objects from the methylation file.
    """
    with open(methylation_filepath) as fd:
        for line in fd.readlines():
            methylation_information: MethylationInfo | None = (
                _process_methylation_line(line, minimum_samples)
            )
            if not methylation_information:
                continue

            yield methylation_information


def _write_deepcpg_methylation(
    file_descriptor: io.TextIOBase,
    methylation_info: MethylationInfo,
    threshold: float = 0.5,
) -> None:
    """
    Write methylation details to a file in a format that deepcpg expects.

    Args
    ----
    file_descriptor: An open ASCII file descriptor.
    methylation_info: The methylation details.
    threshold: Optional threshold at which to consider a CpG site methylated.
    """
    file_descriptor.write(
        f"{methylation_info.chromosome}"
        f"\t{methylation_info.position}"
        f"\t{1 if methylation_info.methylation_ratio >= threshold else 0}\n"
    )


def write_all_deepcpg_methylations(
    methylation_filepath: Path,
    output_file: Path,
    minimum_samples: int = 1,
    threshold: float = 0.5,
) -> None:
    """
    Write all methylation info from file into another file in deepcpg format.

    methylation_filepath: The file storing methylation profiles.
    output_file: The filepath to write out the methylation profiles.
    threshold: Optional threshold at which to consider a CpG site methylated.
    """
    with open(output_file, "w") as fd:
        for methylation_profile in retrieve_methylation_data(
            methylation_filepath, minimum_samples, threshold
        ):
            _write_deepcpg_methylation(fd, methylation_profile, threshold)


def _write_sequence_row(
    file_descriptor: io.TextIOBase,
    sequence: str,
    methylation_profile: MethylationInfo,
) -> None:
    file_descriptor.write(sequence)
    file_descriptor.write(",")
    file_descriptor.write(str(methylation_profile.methylation_ratio))
    file_descriptor.write("\n")


def create_sequence_dataset(
    methylation_filepath: Path,
    fasta_file: Path,
    sequence_length: int,
    chromosomes: list[str],
    output_file: Path,
    minimum_samples: int = 1,
) -> None:
    metadata: dict[str, cpg_sampler.SequenceInfo] = (
        cpg_sampler.extract_fasta_metadata(fasta_file)
    )
    line_length: int = cpg_sampler.determine_line_length(fasta_file)
    with open(output_file, "w") as output_fd, open(fasta_file) as fasta_fd:
        output_fd.write("sequence,ratio_methylated\n")
        for methylation_profile in retrieve_methylation_data(
            methylation_filepath, minimum_samples
        ):
            if methylation_profile.chromosome in chromosomes:
                sequence: str | None = cpg_sampler.find_methylation_sequence(
                    methylation_profile.chromosome,
                    methylation_profile.position,
                    metadata,
                    fasta_fd,
                    sequence_length,
                    line_length,
                )
                if sequence:
                    _write_sequence_row(
                        output_fd, sequence, methylation_profile
                    )


def load_samples(
    data_filepath: Path, threshold: float = 0.5
) -> tuple[NDArray[int], NDArray[int]]:
    sequences, methylation = cpg_sampler.load_methylation_samples(
        data_filepath, threshold=threshold
    )
    input_data = np.array(
        [cpg_sampler.sequence_to_numpy(sequence) for sequence in sequences]
    )
    return input_data, np.array(methylation)
