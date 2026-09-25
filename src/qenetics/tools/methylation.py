import gzip
import logging
from collections.abc import Generator
from dataclasses import dataclass
from enum import IntEnum
from io import TextIOBase
from pathlib import Path

logger = logging.getLogger(__name__)


class MethylationFormat(IntEnum):
    COV = 1
    CPG = 2
    DEEPCPG = 3


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


def normalize_chromosome(chromosome: str) -> str:
    """
    Convert a chromosome name to the Ensembl naming convention.

    UCSC-style names such as 'chr1' and 'chrM' become '1' and 'MT', matching
    the sequence names of Ensembl FASTA files.

    Args
    ----
    chromosome: The chromosome name.

    Returns
    -------
    The Ensembl chromosome name.
    """
    if chromosome.lower().startswith("chr"):
        chromosome = chromosome[3:]

    return "MT" if chromosome == "M" else chromosome


def _is_data_line(line_split: list[str]) -> bool:
    """
    Check whether a split line holds data rather than a header or blank line.

    Args
    ----
    line_split: The whitespace-split line.

    Returns
    -------
    True if the line has a numeric position column. False otherwise.
    """
    return len(line_split) > 1 and line_split[1].isdigit()


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
    if not _is_data_line(line_split):
        return None

    count_methylated = int(line_split[4])
    count_unmethylated = int(line_split[5])
    total_experiments: int = count_methylated + count_unmethylated
    if total_experiments == 0:
        raise ValueError(f"Experiment count is zero for line: {line!r}")

    if total_experiments < minimum_samples:
        return None

    return MethylationInfo(
        chromosome=normalize_chromosome(line_split[0]),
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
    if not _is_data_line(line_split):
        return None

    total_experiments = int(line_split[4])
    if total_experiments < minimum_samples:
        return None

    return MethylationInfo(
        chromosome=normalize_chromosome(line_split[0]),
        position=int(line_split[1]),
        methylation_ratio=float(line_split[7]),
        experiment_count=total_experiments,
        count_methylated=int(line_split[5]),
        count_unmethylated=int(line_split[6]),
        c_context=line_split[2],
        strand=line_split[3],
        trinucleotide_context=line_split[8],
    )


def _process_deepcpg_methylation_line(
    line: str, minimum_count: int = 1
) -> MethylationInfo | None:
    line_split: list[str] = line.rstrip().split()
    if not _is_data_line(line_split):
        return None

    experiment_count: int = int(line_split[3])
    if experiment_count == 0:
        raise ValueError(f"Experiment count is zero for line: {line!r}")

    if experiment_count < minimum_count:
        return None

    return MethylationInfo(
        chromosome=normalize_chromosome(line_split[0]),
        position=int(line_split[1]),
        methylation_ratio=float(line_split[2]),
        experiment_count=experiment_count,
        count_methylated=int(line_split[4]),
        count_unmethylated=int(line_split[5]),
    )


def process_methylation_line(
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


def open_methylation_file(methylation_filepath: Path) -> TextIOBase:
    """
    Open a methylation file for reading text, decompressing it if gzipped.

    Args
    ----
    methylation_filepath: The filepath of the methylation profiles. Files
        ending in '.gz' are read as gzip-compressed.

    Returns
    -------
    A text file descriptor open for reading.
    """
    if methylation_filepath.suffix == ".gz":
        return gzip.open(methylation_filepath, "rt")

    return methylation_filepath.open()


def determine_format(methylation_filepath: Path) -> MethylationFormat:
    """
    Determine the format of the methylation file based on the filename.

    Args
    ----
    methylation_filepath: The filepath of the methylation profiles.

    Returns
    -------
    The format of the methylation file.
    """
    filename: str = methylation_filepath.name.lower()
    if "cov" in filename:
        return MethylationFormat.COV
    if "cpg" in filename:
        return MethylationFormat.CPG
    if "tsv" in filename:
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
    data_format: MethylationFormat = determine_format(methylation_filepath)
    lines_read: int = 0
    with open_methylation_file(methylation_filepath) as fd:
        for line in fd:
            methylation_information: MethylationInfo | None = (
                process_methylation_line(line, data_format, minimum_samples)
            )
            if not methylation_information:
                continue

            lines_read += 1
            yield methylation_information

    if lines_read == 0:
        logger.warning(
            "No methylation profiles read from %s", methylation_filepath
        )


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
    input_format: MethylationFormat = determine_format(input_filepath)
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
        open_methylation_file(input_filepath) as input_fd,
        open(output_filepath, "w") as output_fd,
    ):
        for line in input_fd:
            read_methylation: MethylationInfo | None = process_methylation_line(
                line, input_format, minimum_samples
            )
            if read_methylation is not None:
                output_fd.write(
                    _methylation_line(read_methylation, output_format)
                )


def record_methylation_profiles(
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
