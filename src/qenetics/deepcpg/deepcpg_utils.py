from __future__ import annotations

import io
import logging
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from qenetics.tools import data, dna

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
        for line in fd:
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
            methylation_filepath, minimum_samples
        ):
            _write_deepcpg_methylation(fd, methylation_profile, threshold)


def _write_sequence_row(
    file_descriptor: io.TextIOBase, sequence: str, methylation_ratio: float
) -> None:
    file_descriptor.write(sequence)
    file_descriptor.write(",")
    file_descriptor.write(str(methylation_ratio))
    file_descriptor.write("\n")


def create_sequence_dataset(
    methylation_filepath: Path,
    fasta_file: Path,
    sequence_length: int,
    chromosomes: list[str],
    output_file: Path,
    minimum_samples: int = 1,
) -> None:
    """
    Write the sequence window and methylation ratio of each CpG site to a CSV.

    Sites are found as in `data.create_h5_dataset_from_methylation_profiles`:
    calls on both strands of a CpG site are combined, windows are centered on
    and include the site, and sites whose windows exceed the chromosome or
    hold unknown nucleotides are skipped.

    Args
    ----
    methylation_filepath: The methylation profile file of one experiment, with
        1-based positions.
    fasta_file: The Ensembl FASTA file of the reference genome.
    sequence_length: The length of the sequence windows.
    chromosomes: The chromosomes whose sites to write.
    output_file: The CSV file to write.
    minimum_samples: The minimum reads of a site, summed over both strands.
    """
    metadata: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        fasta_file
    )
    line_length: int = dna.determine_line_length(fasta_file)
    counts_by_chromosome: dict[str, NDArray[np.int64]] = (
        data.read_methylation_counts([methylation_filepath])
    )
    with open(output_file, "w") as output_fd:
        output_fd.write("sequence,ratio_methylated\n")
        for chromosome in chromosomes:
            if (
                chromosome not in counts_by_chromosome
                or chromosome not in metadata
            ):
                continue

            counts: NDArray[np.int64] = counts_by_chromosome[chromosome]
            reference: NDArray[np.uint8] = data.read_reference(
                fasta_file, metadata[chromosome], line_length
            )
            cpg_indices: NDArray[np.int64] = data.locate_cpg_sites(
                reference, counts[:, 0]
            )
            is_cpg: NDArray[np.bool_] = cpg_indices >= 0
            sites, ratios = data.aggregate_cpg_counts(
                cpg_indices[is_cpg],
                counts[is_cpg],
                experiment_quantity=1,
                minimum_reads=minimum_samples,
                binarize=False,
            )
            in_bounds: NDArray[np.bool_] = data.windows_in_bounds(
                sites, sequence_length, len(reference)
            )
            sites, ratios = sites[in_bounds], ratios[in_bounds, 0]
            are_known: NDArray[np.bool_] = data.windows_are_known(
                reference, sites, sequence_length
            )
            sites, ratios = sites[are_known], ratios[are_known]

            for start, ratio in zip(
                data.window_starts(sites, sequence_length), ratios, strict=True
            ):
                window: NDArray[np.uint8] = reference[
                    start : start + sequence_length
                ]
                _write_sequence_row(output_fd, window.tobytes().decode(), ratio)


def create_dataset_from_directory(
    directory: Path,
    fasta_filepath: Path,
    sequence_length: int,
    output_directory: Path,
    minimum_samples: int,
) -> None:
    identifier_length: int = 11
    training_chromosomes: list[str] = ["1", "3", "5", "7", "9", "11"]
    validation_chromosomes: list[str] = ["2", "4", "6", "8", "10", "12"]
    for methylation_file in directory.iterdir():
        output_file_common_name: str = methylation_file.name.split(".")[0][
            identifier_length:
        ]
        create_sequence_dataset(
            methylation_filepath=methylation_file,
            fasta_file=fasta_filepath,
            sequence_length=sequence_length,
            chromosomes=training_chromosomes,
            output_file=output_directory
            / f"training_{output_file_common_name}.csv",
            minimum_samples=minimum_samples,
        )
        create_sequence_dataset(
            methylation_filepath=methylation_file,
            fasta_file=fasta_filepath,
            sequence_length=sequence_length,
            chromosomes=validation_chromosomes,
            output_file=output_directory
            / f"validation_{output_file_common_name}.csv",
            minimum_samples=minimum_samples,
        )
