from __future__ import annotations

import io
from collections.abc import Generator
from dataclasses import dataclass
import logging
from pathlib import Path

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

    chromosome: int
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
    line_split: list[str] = line.rstrip().split("\t")
    if not line_split[0].isnumeric():
        logger.debug(f"Chromosome {line_split[0]} is not numeric.")
        return None

    count_methylated = int(line_split[4])
    count_unmethylated = int(line_split[5])
    total_experiments: int = count_methylated + count_unmethylated

    if total_experiments < minimum_samples:
        return None

    return MethylationInfo(
        chromosome=int(line_split[0]),
        position=int(line_split[1]),
        methylation_ratio=count_methylated / total_experiments,
        experiment_count=total_experiments,
    )


def retrieve_methylation_data(
    methylation_filepath: Path, minimum_samples: int = 1, threshold: float = 0.5
) -> Generator[MethylationInfo, None, None]:
    """
    Create a Generator for MethylationInfo objects from the methylation file.

    Args
    ----
    methylation_filepath: The file storing methylation profiles
    threshold: Optional threshold at which to consider a CpG site methylated.

    Returns
    -------
    Generator for MethylationInfo objects from the methylation file.
    """
    with open(methylation_filepath) as fd:
        for line in fd.readlines():
            methylation_information: MethylationInfo | None = (
                _process_methylation_line(line),
                minimum_samples,
            )
            if (
                methylation_information
                and methylation_information.methylation_ratio >= threshold
            ):
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
