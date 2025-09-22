from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from io import StringIO, TextIOBase
import logging
from pathlib import Path
from typing import Generator

import numpy as np
from numpy.typing import NDArray

from qenetics.qcpg.qcpg import string_to_nucleotides, Nucleodtide

logger = logging.getLogger(__name__)


@dataclass
class TenGenomicsSequenceInfo:
    name: str
    is_chromosome: bool
    length: int


@dataclass
class SequenceInfo:
    """
    Stores information about one sequence in a FASTA file.

    Attributes
    ----------
    length: The number of nucleotides in the sequence.
    is_chromosome: Flags whether the sequence is part of a chromosome or not.
    file_position: The start position of the sequence in the FASTA file.
    """

    length: int
    is_chromosome: bool
    file_position: int = 0


@dataclass
class MethylationInfo:
    """
    Stores information about methylation experiments.

    Attributes
    ----------
    count_methylated: The number of experiments that found the CpG site to be
     methylated.
    count_non_methylated: The number of experiments that found the CpG site to not be
     methylated.
    ratio_methylated: The fraction of the total experiments that were methylated.
    """

    count_methylated: int
    count_non_methylated: int
    ratio_methylated: float = -1.0


@dataclass
class MethylationSequence:
    """
    A struct that stores both the nucleotide sequence and methylation data.

    Attributes
    ----------
    sequence: The sequence of nucleotides (A, C, T, or G).
    methylation_profile: The experimental results of sampling a CpG site for
     methylation.
    """

    sequence: str
    methylation_profile: MethylationInfo


def _write_sequence(
    sequence: str, name: str, is_chromosome: bool, output_filepath: Path
) -> None:
    """
    Append the annotation and sequence to a file.

    Args
    ----
    sequence: The sequence of nucleotides.
    name: The name of the sequence to be written
    is_chromosome: Designates whether the sequence belongs to a nucleotide or not.
    output_filepath:  The filepath to write the data to.
    """
    with open(output_filepath, "a") as fd:
        fd.write(">")
        fd.write(name)
        fd.write(" ")
        if is_chromosome:
            fd.write("dna:chromosome chromosome:GRCm38:")
        else:
            fd.write("dna_sm:scaffold scaffold:GRCm38:")
        fd.write(name)
        fd.write(":1:")
        fd.write(str(len(sequence) - sequence.count("\n")))
        fd.write(":1 REF\n")
        fd.write(sequence)


def _read_tengenomics_annotation(line: str) -> tuple[str, bool]:
    """
    Retrieve the name of the sequence and whether it is a chromosome or not.

    Args
    ----
    line: The read line.

    Returns
    -------
    The name and whether the sequence belongs to a chromosome.
    """
    annotation: list[str] = line.split()
    return annotation[1], "chr" in annotation[0]


def write_ensembl_from_tengenomics(
    tengenomics_filepath: Path, output_filepath: Path
) -> None:
    """
    Write the contents of a 10xGenomics FASTA file into an ensembl-formatted file.

    Args
    ----
    tengenomics_filepath: The existing 10xGenomics filepath.
    output_filepath: The intended filepath of the Ensembl-formatted file.
    """
    block_read_size: int = 2**20  # 1MB read size
    sequence: str = ""
    with open(tengenomics_filepath) as fd:
        buffer = StringIO(fd.read(block_read_size))
        line: str = buffer.readline()
        while len(line) > 0:
            if line[0] == ">":
                if line[-1] != "\n":
                    buffer = StringIO(fd.read(block_read_size))
                    line += buffer.readline()
                if len(sequence) > 0:
                    _write_sequence(
                        sequence, name, is_chromosome, output_filepath
                    )
                name, is_chromosome = _read_tengenomics_annotation(line)
                sequence = ""
            else:
                sequence += line
                if line[-1] != "\n":
                    buffer = StringIO(fd.read(block_read_size))
                    sequence += buffer.readline()

            line = buffer.readline()
            if len(line) == 0:
                buffer = StringIO(fd.read(block_read_size))
                line = buffer.readline()

        if len(sequence) > 0:
            _write_sequence(sequence, name, is_chromosome, output_filepath)


def extract_line_annotation(line: str) -> tuple[str, SequenceInfo]:
    """
    Extract sequence data info from FASTA data comment line.

    Args
    ----
    line: FASTA data comment line.

    Returns
    -------
    An object containing the chromosome number and length if valid, None if not.
    """
    info: str = line.split()[2]
    details: list[str] = info.split(":")

    return (
        details[2],
        SequenceInfo(
            length=int(details[4]), is_chromosome="chromosome" in details
        ),
    )


def find_next_comment(file_descriptor: TextIOBase, offset: int) -> bool:
    """
    Move reader offset to the next FAFSA comment and verify.

    Args
    ----
    file_descriptor: An open file read-only ASCII file descriptor.
    offset: The offset which to move the file pointer to.

    Returns
    -------
    True if found, False otherwise.
    """
    file_descriptor.seek(offset)
    if file_descriptor.read(1) == ">":
        return True

    return False


def determine_line_length(fasta_file: Path) -> int:
    """
    Find the standard line length of the nucleotide data in the file.

    Args
    ----
    fasta_file: The reference genome file, in FASTA format.

    Returns
    -------
    The length of the first line of the first nucleotide sequence.
    """
    with fasta_file.open() as fd:
        if not find_next_comment(fd, 0):
            raise IOError(
                "Unable to determine sequence line length of fasta file."
            )

        fd.readline()
        return len(fd.readline()) - 1


def extract_fasta_metadata(fasta_file: Path) -> dict[str, SequenceInfo]:
    """
    Extract all metadata from FAFSA comment lines.

    Args
    ----
    fasta_file: The filepath of a valid FAFSA file.

    Returns
    -------
    The FASTA metadata indexed by chromosome.
    """
    annotations: dict[str, SequenceInfo] = {}

    line_length: int = determine_line_length(fasta_file)

    read_position: int = 0
    with fasta_file.open() as fd:
        while find_next_comment(fd, read_position):
            chromosome, sequence_info = extract_line_annotation(fd.readline())
            sequence_info.file_position = fd.tell()
            if sequence_info.is_chromosome:
                annotations[chromosome] = sequence_info
            newline_quantity = int(sequence_info.length / line_length) + 1
            read_position = (
                sequence_info.file_position
                + sequence_info.length
                + newline_quantity
            )  # Skip newlines

    return annotations


def _load_methlation_file_data(
    methylation_file: Path,
    methylation_data: dict[str, dict[int, MethylationInfo]],
) -> None:
    """
    Load methylation counts and associated reference genome positions from TSV file.

    Args
    ----
    methylation_file: A TSV file containing methylation data.
    methylation_data: An instantiated dictionary of methylation counts index by
    chromosome and reference genome position.
    """
    with methylation_file.open() as fd:
        entry: str = fd.readline()
        while entry:
            tab_split: list[str] = entry.split()
            chromosome: str = tab_split[0]
            position = int(tab_split[1])
            count_methylated = int(tab_split[4])
            count_non_methylated = int(tab_split[5])
            if methylation_data.get(chromosome):
                if methylation_data[chromosome].get(position):
                    methylation_data[chromosome][
                        position
                    ].count_methylated += count_methylated
                    methylation_data[chromosome][
                        position
                    ].count_non_methylated += count_non_methylated
                else:
                    methylation_data[chromosome][position] = MethylationInfo(
                        count_methylated=count_methylated,
                        count_non_methylated=count_non_methylated,
                    )
            else:
                methylation_data[chromosome] = {
                    position: MethylationInfo(
                        count_methylated=count_methylated,
                        count_non_methylated=count_non_methylated,
                    )
                }
            entry = fd.readline()


def combine_methylation_results(
    methylation_directory: list[Path],
) -> dict[str, dict[int, MethylationInfo]]:
    """
    Aggregate the methylation counts from multiple files.

    Args
    ----
    methylation_directory: A directory containing TSV methylation profiles.

    Returns
    -------
    Methylation counts indexed by chromosome and reference genome position.
    """
    methylation_data: dict[str, dict[int, MethylationInfo]] = dict()
    for methylation_file in methylation_directory:
        _load_methlation_file_data(methylation_file, methylation_data)

    return methylation_data


def filter_and_calculate_methylation(
    methylation_profiles: dict[str, dict[int, MethylationInfo]],
    minimum_count: int = 1,
) -> dict[str, dict[int, MethylationInfo]]:
    """
    Remove methylation profiles with fewer than minimum counts and calculate ratio.

    Args
    ----
    methylation_profiles: Location and count of methylation.
    minimum_count: Profiles below minimum count are filtered out.

    Returns
    -------
    Remaining methylation profiles with methylation ratio filled in.
    """
    filtered_data: dict[str, dict[int, MethylationInfo]] = dict()
    for chromosome, chromosome_profiles in methylation_profiles.items():
        for position, methylation_profile in chromosome_profiles.items():
            total_counted: int = (
                methylation_profile.count_methylated
                + methylation_profile.count_non_methylated
            )
            if total_counted >= minimum_count:
                methylation_profile.ratio_methylated = (
                    methylation_profile.count_methylated / total_counted
                )
                if filtered_data.get(chromosome):
                    filtered_data[chromosome][position] = methylation_profile
                else:
                    filtered_data[chromosome] = {position: methylation_profile}

    return filtered_data


def _read_sequence(
    file_descriptor: TextIOBase, sequence_length: int, line_length: int
) -> str | None:
    """
    Read a sequence of nucleotides from a FASTA file.

    Args
    ----
    file_descriptor: The file descriptor, already open for reading ASCII data.
    sequence_length: The length of the sequence to extract.
    line_length: The length of a line of nucleotide data in the FASTA file.

    Returns
    -------
    A sequence of nucleotides, if valid. Otherwise, returns None.
    """
    cg_length: int = 2
    included_cg_length: int = sequence_length + cg_length
    sequence: str = file_descriptor.readline().rstrip()
    half_length = int(sequence_length / 2)
    if len(sequence) > included_cg_length:
        return (
            sequence[:half_length]
            + sequence[half_length + 2 : included_cg_length]
        )

    newline_quantity = int((sequence_length - len(sequence)) / line_length)
    sequence += file_descriptor.read(
        included_cg_length - len(sequence) + newline_quantity
    ).replace("\n", "")
    if len(sequence) > sequence_length:
        return (
            sequence[:half_length]
            + sequence[half_length + cg_length : included_cg_length]
        )

    return None


def find_methylation_sequence(
    chromosome: str,
    position: int,
    genome_metadata: dict[str, SequenceInfo],
    file_descriptor: TextIOBase,
    sequence_length: int,
    line_length: int,
) -> str | None:
    """
    Extract nucleotide sequence from file given methylation site information.

    Args
    ----
    chromosome: The chromosome character designator.
    position: The start position of the CpG site within the chromosome.
    genome_metadata: The reference genome file metadata.
    file_descriptor: An open ASCII file descriptor for reference genome.
    sequence_length: The length of sequence to retrieve.
    line_length: The length of a line of nucleotide data in the FASTA file.

    Returns
    -------
    A sequence of nucleotides, if valid. Otherwise, returns None.
    """
    if not genome_metadata.get(chromosome):
        return None

    chromosome_metadata: SequenceInfo = genome_metadata[chromosome]
    half_sequence_length = int(sequence_length / 2)
    if (
        position - half_sequence_length < 0
        or position + half_sequence_length + 2 >= chromosome_metadata.length
    ):
        return None

    file_descriptor.seek(
        chromosome_metadata.file_position + position - half_sequence_length
    )
    return _read_sequence(file_descriptor, sequence_length, line_length)


def retrieve_all_cpg_sequences(
    genome_metadata: dict[str, SequenceInfo],
    fasta_file: Path,
    methylation_profiles: dict[str, dict[int, MethylationInfo]],
    sequence_length: int,
) -> Generator[MethylationSequence, None, None]:
    """
    Retrieve all sequences surrounding provided methylation sites.

    Args
    ----
    genome_metadata: The reference genome file metadata.
    fasta_file: The filepath to the reference genome data.
    methylation_profiles: The methylation profile data.
    sequence_length: The length of sequence to extract.

    Yields
    ------
    Sequence information for each CpG site.
    """
    line_length: int = determine_line_length(fasta_file)
    with fasta_file.open() as fd:
        for chromosome, chromosome_data in methylation_profiles.items():
            for position, methylation_profile in chromosome_data.items():
                sequence: str = find_methylation_sequence(
                    chromosome,
                    position,
                    genome_metadata,
                    fd,
                    sequence_length,
                    line_length,
                )
                if sequence:
                    yield MethylationSequence(
                        sequence=sequence,
                        methylation_profile=methylation_profile,
                    )


def load_and_save_all_cpg_sequences(
    fasta_file: Path,
    methylation_directory: list[Path],
    sequence_length: int,
    output_directory: Path,
    minimum_count: int = 1,
) -> None:
    """
    Extract all methylation sequences and save to a file.

    Retrieve all sequences surrounding CpG sites with at least minimum_count methylation
    samples.

    Args
    ----
    fasta_file: The reference genome FASTA file.
    methylation_directory: A list of empirical methylation profiles.
    sequence_length: The length of the surrounding sequence to extract.
    output_directory: The directory in which to save the sequences.
    minimum_count: Filter out CpG sites below minimum_count.
    """
    metadata: dict[str, SequenceInfo] = extract_fasta_metadata(fasta_file)
    methylation_profiles: dict[str, dict[int, MethylationInfo]] = (
        filter_and_calculate_methylation(
            combine_methylation_results(methylation_directory), minimum_count
        )
    )
    sequences: Generator[MethylationSequence, None, None] = (
        retrieve_all_cpg_sequences(
            metadata, fasta_file, methylation_profiles, sequence_length
        )
    )
    output_file: Path = (
        output_directory
        / f"cpg_sequences_s{sequence_length}_m{minimum_count}.csv"
    )
    with output_file.open("w") as fd:
        fd.write("sequence,ratio_methylated\n")
        for sequence_info in sequences:
            fd.write(sequence_info.sequence)
            fd.write(",")
            fd.write(str(sequence_info.methylation_profile.ratio_methylated))
            fd.write("\n")


def load_methylation_samples(
    methylation_file: Path, threshold: float = 0.5, negative_state: int = 0
) -> tuple[list[list[Nucleodtide]], list[int]]:
    """
    Load all sequences and their associated methylation ratios from a file.

    Args
    ----
    methylation_file: A file storing sequences and methylation ratios.
    threshold: The threshold for which to consider a CpG site methylated.
    negative_state: The integer to set the negative truth value to.

    Returns
    -------
    The sequences and their associated methylation ratios.
    """
    logger.debug(
        f"Loading methylation data from {methylation_file} with threshold "
        f"{threshold}."
    )
    sequences: list[list[Nucleodtide]] = []
    methylation_ratios: list[int] = []
    with methylation_file.open() as fd:
        csv_file = DictReader(fd)
        for row in csv_file:
            sequence: str = row["sequence"]
            if "N" in sequence:
                continue
            sequences.append(string_to_nucleotides(sequence))
            methylation_ratios.append(
                1
                if float(row["ratio_methylated"]) > threshold
                else negative_state
            )

    logging.debug(f"Loaded {len(sequences)} methylation samples.")

    return sequences, methylation_ratios


def nucleotide_to_numpy(nucleotide: Nucleodtide) -> NDArray[int]:
    """
    Convert a Nucleotide object to a one-hot encoded array.

    Args
    ----
    nucleotide:

    Returns
    -------
    The one-hot encoded array.
    """
    one_hot_array: NDArray[int] = np.array([0] * 4)
    one_hot_array[nucleotide.value] = 1
    return one_hot_array


def sequence_to_numpy(sequence: list[Nucleodtide]) -> NDArray[int]:
    """
    Convert a sequence of nucleotides to an array of one-hot encoded arrays.

    Args
    ----
    sequence: The nucleotide sequence.

    Returns
    -------
    The two-dimensional, one-hot encoded array.
    """
    return np.array(
        [nucleotide_to_numpy(nucleotide) for nucleotide in sequence]
    )
