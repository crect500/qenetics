from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from enum import IntEnum
from io import StringIO, TextIOBase
import logging
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


class MethylationFormat(IntEnum):
    COV = 1
    CPG = 2
    DEEPCPG = 3


@dataclass
class TenGenomicsSequenceInfo:
    name: str
    is_chromosome: bool
    length: int


class Nucleotide(IntEnum):
    """
    Maps nucleotide abbreviations to enum.

    A - 0
    T - 1
    C - 2
    G - 3
    """

    A = 0
    T = 1
    C = 2
    G = 3


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


def convert_nucleotide_to_enum(nucleotide: str) -> Nucleotide:
    """
    Convert nucleotide abbreviation to corresponding enum.

    A - 0, T - 1, C - 2, G - 3

    Args
    ----
    nucleotide: The ASCII representation of the nucleotide abbreviation.

    Returns
    -------
    The corresponding nucleotide enum.

    Raises
    ------
    ValueError if valid character not provided.

    """
    if nucleotide == "A":
        return Nucleotide.A
    if nucleotide == "T":
        return Nucleotide.T
    if nucleotide == "C":
        return Nucleotide.C
    if nucleotide == "G":
        return Nucleotide.G

    raise ValueError(
        f"Character {nucleotide} not recognized as valid nucleotide"
    )


def string_to_nucleotides(nucleotide_string: str) -> list[Nucleotide]:
    """
    Convert a string to nucleotide enumerations.

    Args
    ----
    nucleotide_string: A string of characters in the set A, T, C, G.

    Returns
    -------
    A list of nucleotide enumerations.
    """
    return [
        convert_nucleotide_to_enum(character) for character in nucleotide_string
    ]


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
    Stores information from a methylation experiment data file.

    Attributes
    ----------
    chromosome: The chromosome of the sequence referenced.
    position: The position of the cytosine of the CpG site.
    methylation_ratio: The ratio of sites found to be methylated.
    experiment_count: The number of experiments performed.
    count_methylated: The number of samples found to be methylated.
    count_unmethylated: The number of samples found not to be methylated.
    c_context: The nucleotide at the position.
    strand:
    trinucleotide_context: The three next nucleotides at the position.
    """

    chromosome: str
    position: int
    methylation_ratio: float
    experiment_count: int
    count_methylated: int
    count_unmethylated: int
    c_context: str = ""
    strand: str = ""
    trinucleotide_context: str = ""


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
    name: str = ""
    is_chromosome: bool = False
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
            newline_quantity = int(sequence_info.length / line_length)
            if sequence_info.length % line_length != 0.0:
                newline_quantity += 1
            read_position = (
                sequence_info.file_position
                + sequence_info.length
                + newline_quantity
            )  # Skip newlines

    return annotations


def _process_cov_methylation_line(
    line: str, minimum_samples: int = 1
) -> MethylationInfo | None:
    """
    Process one line in a CpG methylation coverage file.

    Args
    ----
    line: One line from the CpG methylation coverage file.
    minimum_samples: The minimum samples for which to consider a sample.

    Returns
    -------
    The CpG methylation experiment results.
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
        count_methylated=count_methylated,
        count_unmethylated=count_unmethylated,
    )


def _process_cpg_methylation_line(
    line: str, minimum_samples: int = 1
) -> MethylationInfo | None:
    """
    Process one line in a CpG methylation cpg file.

    Args
    ----
    line: One line from the CpG methylation cpgcl file.
    minimum_samples: The minimum samples for which to consider a sample.

    Returns
    -------
    The CpG methylation experiment results.
    """
    line_split: list[str] = line.rstrip().split()
    total_experiments = int(line_split[4])
    if total_experiments < minimum_samples:
        return None

    return MethylationInfo(
        chromosome=line_split[0][3:],
        position=int(line_split[1]),
        methylation_ratio=float(line_split[7]),
        experiment_count=total_experiments,
        count_methylated=int(line_split[5]),
        count_unmethylated=int(line_split[6]),
        c_context=line[2],
        strand=line[3],
        trinucleotide_context=line[8],
    )


def _process_deepcpg_methylation_line(
    line: str, minimum_count: int = 1
) -> MethylationInfo | None:
    line_split: list[str] = line.rstrip().split()
    experiment_count: int = int(line_split[3])
    if experiment_count < experiment_count:
        return None

    return MethylationInfo(
        chromosome=line_split[0][3:],
        position=int(line_split[1]),
        methylation_ratio=float(line_split[2]),
        experiment_count=experiment_count,
        count_methylated=int(line_split[4]),
        count_unmethylated=int(line_split[5]),
    )


def _process_methylation_line(
    line: str,
    data_format: MethylationFormat = MethylationFormat.COV,
    minimum_samples: int = 1,
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
    if data_format == MethylationFormat.COV:
        return _process_cov_methylation_line(line, minimum_samples)

    if data_format == MethylationFormat.CPG:
        return _process_cpg_methylation_line(line, minimum_samples)

    if data_format == MethylationFormat.DEEPCPG:
        return _process_deepcpg_methylation_line(line, minimum_samples)


def _determine_format(methylation_filepath: Path) -> MethylationFormat:
    """
    Determine the format of the methylation file based on the filename.

    Args
    ----
    methylation_filepath: The filepath of the methylation profiles.

    Returns
    -------
    The format of the methylation file.
    """
    if "cov" in methylation_filepath.name:
        return MethylationFormat.COV
    if "cpg" in methylation_filepath.name:
        return MethylationFormat.CPG
    if "tsv" in methylation_filepath.name:
        return MethylationFormat.DEEPCPG
    else:
        raise ValueError(
            f"File format for {methylation_filepath} not recognized."
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
    data_format: MethylationFormat = _determine_format(methylation_filepath)
    with open(methylation_filepath) as fd:
        for line in fd.readlines():
            methylation_information: MethylationInfo | None = (
                _process_methylation_line(line, data_format, minimum_samples)
            )
            if not methylation_information:
                continue

            yield methylation_information


def _cov_methylation_line(methylation_profile: MethylationInfo) -> str:
    return (
        f"{methylation_profile.chromosome}\t"
        f"{methylation_profile.position}\t"
        f"{methylation_profile.position}\t"
        f"{methylation_profile.methylation_ratio * 100.0}\t"
        f"{methylation_profile.count_methylated}\t"
        f"{methylation_profile.count_unmethylated}\n"
    )


def _cpg_methylation_line(methylation_profile: MethylationInfo) -> str:
    if methylation_profile.c_context != "":
        c_context: str = methylation_profile.c_context
    else:
        c_context = "N"

    if methylation_profile.strand != "":
        strand = methylation_profile.strand
    else:
        strand: str = "N"

    if methylation_profile.trinucleotide_context != "":
        trinucleotide_context = methylation_profile.trinucleotide_context
    else:
        trinucleotide_context: str = "NNN"

    return (
        f"chr{methylation_profile.chromosome}\t"
        f"{methylation_profile.position}\t"
        f"{c_context}\t"
        f"{strand}\t"
        f"{methylation_profile.experiment_count}\t"
        f"{methylation_profile.count_methylated}\t"
        f"{methylation_profile.count_unmethylated}\t"
        f"{methylation_profile.methylation_ratio}\t"
        f"{trinucleotide_context}\t"
        "CpG\n"
    )


def _deepcpg_methylation_line(methylation_profile: MethylationInfo) -> str:
    return (
        f"{methylation_profile.chromosome}\t"
        f"{methylation_profile.position}\t"
        f"{methylation_profile.methylation_ratio}\t"
        f"{methylation_profile.experiment_count}\t"
        f"{methylation_profile.count_methylated}\t"
        f"{methylation_profile.count_unmethylated}\n"
    )


def _methylation_line(
    methylation_profile: MethylationInfo, methylation_format: MethylationFormat
) -> str:
    if methylation_format == MethylationFormat.COV:
        return _cov_methylation_line(methylation_profile)

    if methylation_format == MethylationFormat.CPG:
        return _cpg_methylation_line(methylation_profile)

    if methylation_format == MethylationFormat.DEEPCPG:
        return _deepcpg_methylation_line(methylation_profile)


def convert_methylation_profiles(
    input_filepath: Path,
    output_directory: Path,
    output_format: MethylationFormat,
    minimum_samples: int = 1,
) -> None:
    input_format: MethylationFormat = _determine_format(input_filepath)
    if output_format == MethylationFormat.COV:
        suffix: str = ".cov.txt"
    elif output_format == MethylationFormat.CPG:
        suffix = "cpg.txt"
    elif output_format == MethylationFormat.DEEPCPG:
        suffix = ".tsv"

    if (
        input_format == MethylationFormat.COV
        or input_format == MethylationFormat.CPG
    ):
        output_filepath: Path = output_directory / (
            input_filepath.stem.split(".")[0] + suffix
        )
    else:
        output_filepath = output_directory / (input_filepath.stem + suffix)

    with (
        open(input_filepath) as input_fd,
        open(output_filepath, "w") as output_fd,
    ):
        for line in input_fd.readlines():
            read_methylation: MethylationInfo | None = (
                _process_methylation_line(line, input_format, minimum_samples)
            )
            if read_methylation is not None:
                output_fd.write(
                    _methylation_line(read_methylation, output_format)
                )


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

    newline_quantity = int((included_cg_length - len(sequence)) / line_length)
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

    cg_length: int = 2
    chromosome_metadata: SequenceInfo = genome_metadata[chromosome]
    half_sequence_length = int(sequence_length / 2)
    if (
        position - half_sequence_length < 0
        or position + half_sequence_length + cg_length
        >= chromosome_metadata.length
    ):
        return None

    file_descriptor.seek(
        chromosome_metadata.file_position + position - half_sequence_length
    )
    return _read_sequence(file_descriptor, sequence_length, line_length)


def load_methylation_samples(
    methylation_file: Path, threshold: float = 0.5, negative_state: int = 0
) -> tuple[list[list[Nucleotide]], list[int]]:
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
    sequences: list[list[Nucleotide]] = []
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


def nucleotide_to_numpy(nucleotide: Nucleotide) -> NDArray[int]:
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


def sequence_to_numpy(sequence: list[Nucleotide]) -> NDArray[int]:
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


def read_methylation_files(
    methylation_directory: Path,
    minimum_samples: int = 1,
) -> list[MethylationInfo]:
    counts_by_chromosome: dict[str, int] = {}
    methylation_profiles: list[MethylationInfo] = []
    for methylation_filepath in methylation_directory.iterdir():
        for methylation_profile in retrieve_methylation_data(
            methylation_filepath
        ):
            if methylation_profile.experiment_count >= minimum_samples:
                _increment_or_create_entry(
                    counts_by_chromosome, methylation_profile.chromosome
                )
                methylation_profiles.append(methylation_profile)

    return methylation_profiles


def _record_methylation_profiles(
    methylation_directory: Path, minimum_samples: int = 1
) -> dict[str, dict[int, dict[str, float]]]:
    profiles_by_chromosome: dict[str, dict[int, dict[str, float]]] = {}
    for filepath in methylation_directory.iterdir():
        experiment_name: str = filepath.stem.split(".")[0]
        for methylation_profile in retrieve_methylation_data(
            filepath, minimum_samples
        ):
            profiles_by_position: dict[int, dict[str, float]] | None = (
                profiles_by_chromosome.get(methylation_profile.chromosome)
            )
            if profiles_by_position is None:
                profiles_by_chromosome[methylation_profile.chromosome] = {
                    methylation_profile.position: {
                        experiment_name: methylation_profile.methylation_ratio
                    }
                }
            else:
                profiles_by_experiment: dict[str, float] | None = (
                    profiles_by_position.get(methylation_profile.position)
                )
                if profiles_by_experiment is None:
                    profiles_by_position[methylation_profile.position] = {
                        experiment_name: methylation_profile.methylation_ratio
                    }
                else:
                    profiles_by_experiment[experiment_name] = (
                        methylation_profile.methylation_ratio
                    )

    return profiles_by_chromosome


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
        sequence: str | None = find_methylation_sequence(
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
) -> list[Path]:
    fasta_line_length: int = determine_line_length(fasta_filepath)
    with open(fasta_filepath) as fasta_fd:
        fasta_metadata: dict[str, SequenceInfo] = extract_fasta_metadata(
            fasta_filepath
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
        _record_methylation_profiles(methylation_directory, minimum_samples)
    )
    _create_h5_files(
        profiles_by_chromosome=profiles_by_chromosome,
        experiment_names=experiment_names,
        dataset_directory=dataset_directory,
        fasta_filepath=fasta_filepath,
        sequence_length=sequence_length,
    )
