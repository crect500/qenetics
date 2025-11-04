from dataclasses import dataclass
from enum import IntEnum
from io import StringIO, TextIOBase
from pathlib import Path


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


def extract_fasta_metadata(
    fasta_file: Path, crlf: bool = False
) -> dict[str, SequenceInfo]:
    """
    Extract all metadata from FAFSA comment lines.

    Args
    ----
    fasta_file: The filepath of a valid FAFSA file.
    crlf: Set to true if ASCII file is CRLF.

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
            if crlf:
                newline_quantity *= 2
            if sequence_info.length % line_length != 0:
                newline_quantity += 1
            read_position = (
                sequence_info.file_position
                + sequence_info.length
                + newline_quantity
            )  # Skip newlines

    return annotations
