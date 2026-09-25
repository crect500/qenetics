import csv
from dataclasses import asdict, dataclass, fields
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from numpy.typing import NDArray

from qenetics.tools import data, dna

INVALID_BY_CHROMOSOME: str = "invalid_by_chromosome"
INVALID_BY_NON_CPG: str = "invalid_by_non_cpg"
INVALID_BY_MINIMUM: str = "invalid_by_minimum"
INVALID_BY_BOUNDARIES: str = "invalid_by_boundaries"
INVALID_BY_MISSING: str = "invalid_by_missing"
VALID_METHYLATED: str = "valid_methylated"
VALID_UNMETHYLATED: str = "valid_unmethylated"
INVALID_CATEGORIES: tuple[str, ...] = (
    INVALID_BY_CHROMOSOME,
    INVALID_BY_NON_CPG,
    INVALID_BY_MINIMUM,
    INVALID_BY_BOUNDARIES,
    INVALID_BY_MISSING,
)


@dataclass
class ExperimentStatistics:
    """
    Counts of the methylation calls and CpG sites of an experiment.

    Mirrors the filtering of `data.create_h5_dataset_from_methylation_profiles`.
    `total_reads`, `invalid_by_chromosome` and `invalid_by_non_cpg` count
    methylation calls. The remaining attributes count CpG sites, after the
    calls on both strands of each site are combined.

    Attributes
    ----------
    total_reads: The quantity of methylation calls.
    invalid_by_chromosome: Calls on chromosomes missing from the reference.
    invalid_by_non_cpg: Calls not at a CpG site of the reference.
    invalid_by_minimum: Sites with fewer reads than the minimum.
    invalid_by_sequence_length: Sites whose windows exceed the chromosome.
    invalid_by_missing_nucleotide: Sites whose windows hold unknown
        nucleotides.
    valid_methylated: Sites with more methylated than unmethylated reads.
    valid_unmethylated: Remaining valid sites.
    """

    total_reads: int = 0
    invalid_by_chromosome: int = 0
    invalid_by_non_cpg: int = 0
    invalid_by_minimum: int = 0
    invalid_by_sequence_length: int = 0
    invalid_by_missing_nucleotide: int = 0
    valid_methylated: int = 0
    valid_unmethylated: int = 0


_STATISTIC_BY_CATEGORY: dict[str, str] = {
    INVALID_BY_CHROMOSOME: "invalid_by_chromosome",
    INVALID_BY_NON_CPG: "invalid_by_non_cpg",
    INVALID_BY_MINIMUM: "invalid_by_minimum",
    INVALID_BY_BOUNDARIES: "invalid_by_sequence_length",
    INVALID_BY_MISSING: "invalid_by_missing_nucleotide",
    VALID_METHYLATED: "valid_methylated",
    VALID_UNMETHYLATED: "valid_unmethylated",
}


def _classify_chromosome_sites(
    counts: NDArray[np.int64],
    reference: NDArray[np.uint8],
    sequence_length: int,
    minimum_samples: int = 1,
) -> dict[str, list[int]]:
    """
    Sort the methylation calls of one chromosome and experiment by outcome.

    Args
    ----
    counts: The read counts of each call, as returned by
        `data.read_methylation_counts`.
    reference: The ASCII nucleotides of the chromosome.
    sequence_length: The length of the sequence windows.
    minimum_samples: The minimum reads of a site, summed over both strands.

    Returns
    -------
    The 1-based positions in each non-empty category. Positions of the
    `INVALID_BY_NON_CPG` category are those of the calls. Positions of the
    other categories are those of the C of each CpG site.
    """
    cpg_indices: NDArray[np.int64] = data.locate_cpg_sites(
        reference, counts[:, 0]
    )
    is_cpg: NDArray[np.bool_] = cpg_indices >= 0
    positions_by_category: dict[str, NDArray[np.int64]] = {
        INVALID_BY_NON_CPG: counts[~is_cpg, 0]
    }

    covered_sites, _ = data.aggregate_cpg_counts(
        cpg_indices[is_cpg], counts[is_cpg], experiment_quantity=1
    )
    sites, labels = data.aggregate_cpg_counts(
        cpg_indices[is_cpg],
        counts[is_cpg],
        experiment_quantity=1,
        minimum_reads=minimum_samples,
    )
    positions_by_category[INVALID_BY_MINIMUM] = (
        np.setdiff1d(covered_sites, sites) + 1
    )

    in_bounds: NDArray[np.bool_] = data.windows_in_bounds(
        sites, sequence_length, len(reference)
    )
    positions_by_category[INVALID_BY_BOUNDARIES] = sites[~in_bounds] + 1
    sites, labels = sites[in_bounds], labels[in_bounds, 0]

    are_known: NDArray[np.bool_] = data.windows_are_known(
        reference, sites, sequence_length
    )
    positions_by_category[INVALID_BY_MISSING] = sites[~are_known] + 1
    sites, labels = sites[are_known], labels[are_known]

    positions_by_category[VALID_METHYLATED] = sites[labels == 1.0] + 1
    positions_by_category[VALID_UNMETHYLATED] = sites[labels == 0.0] + 1

    return {
        category: positions.tolist()
        for category, positions in positions_by_category.items()
        if len(positions) > 0
    }


def _classify_sites(
    methylation_filepath: Path,
    sequence_length: int,
    fasta_filepath: Path,
    fasta_metadata: dict[str, dna.SequenceInfo],
    fasta_line_length: int,
    minimum_samples: int = 1,
) -> tuple[int, dict[str, dict[str, list[int]]]]:
    """
    Sort the methylation calls of an experiment by outcome.

    Args
    ----
    methylation_filepath: The methylation profile file of the experiment.
    sequence_length: The length of the sequence windows.
    fasta_filepath: The FASTA file of the reference genome.
    fasta_metadata: The metadata of the reference genome.
    fasta_line_length: The length of a line of nucleotide data in the FASTA
        file.
    minimum_samples: The minimum reads of a site, summed over both strands.

    Returns
    -------
    The quantity of methylation calls, and the 1-based positions in each
    non-empty category, indexed by chromosome.
    """
    counts_by_chromosome: dict[str, NDArray[np.int64]] = (
        data.read_methylation_counts([methylation_filepath])
    )
    call_quantity: int = sum(
        len(counts) for counts in counts_by_chromosome.values()
    )
    positions_by_chromosome: dict[str, dict[str, list[int]]] = {}
    for chromosome, counts in counts_by_chromosome.items():
        if chromosome not in fasta_metadata:
            positions_by_chromosome[chromosome] = {
                INVALID_BY_CHROMOSOME: counts[:, 0].tolist()
            }
            continue

        positions_by_chromosome[chromosome] = _classify_chromosome_sites(
            counts,
            data.read_reference(
                fasta_filepath, fasta_metadata[chromosome], fasta_line_length
            ),
            sequence_length,
            minimum_samples,
        )

    return call_quantity, positions_by_chromosome


def get_profile_counts(
    methylation_filepath: Path,
    sequence_length: int,
    fasta_filepath: Path,
    fasta_metadata: dict[str, dna.SequenceInfo],
    fasta_line_length: int,
    minimum_samples: int = 1,
) -> ExperimentStatistics:
    call_quantity, positions_by_chromosome = _classify_sites(
        methylation_filepath,
        sequence_length,
        fasta_filepath,
        fasta_metadata,
        fasta_line_length,
        minimum_samples,
    )
    experiment_statistics = ExperimentStatistics(total_reads=call_quantity)
    for positions_by_category in positions_by_chromosome.values():
        for category, positions in positions_by_category.items():
            statistic: str = _STATISTIC_BY_CATEGORY[category]
            setattr(
                experiment_statistics,
                statistic,
                getattr(experiment_statistics, statistic) + len(positions),
            )

    return experiment_statistics


def get_dataset_stats(
    methylation_filepaths: list[Path],
    sequence_length: int,
    fasta_filepath: Path,
    minimum_samples: int = 1,
) -> dict[str, ExperimentStatistics]:
    fasta_line_length: int = dna.determine_line_length(fasta_filepath)
    fasta_metadata: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        fasta_filepath
    )
    stats_by_experiment: dict[str, ExperimentStatistics] = {}
    for methylation_filepath in methylation_filepaths:
        experiment_name: str = methylation_filepath.stem.split(".")[0]
        stats_by_experiment[experiment_name] = get_profile_counts(
            methylation_filepath,
            sequence_length,
            fasta_filepath,
            fasta_metadata,
            fasta_line_length,
            minimum_samples=minimum_samples,
        )

    return stats_by_experiment


def write_experiment_statistics(
    experiment_statistics: dict[str, ExperimentStatistics],
    output_filepath: Path,
) -> None:
    statistic_names: list[str] = [
        field.name for field in fields(ExperimentStatistics)
    ]
    totals = ExperimentStatistics()
    with output_filepath.open("w") as fd:
        csv_writer = csv.DictWriter(
            fd, fieldnames=["experiment_name", *statistic_names]
        )
        csv_writer.writeheader()
        for experiment_name, statistics in experiment_statistics.items():
            csv_writer.writerow(
                {"experiment_name": experiment_name, **asdict(statistics)}
            )
            for statistic in statistic_names:
                setattr(
                    totals,
                    statistic,
                    getattr(totals, statistic) + getattr(statistics, statistic),
                )

        csv_writer.writerow({"experiment_name": "total", **asdict(totals)})


def _find_invalid_sites(
    methylation_filepath: Path,
    sequence_length: int,
    fasta_filepath: Path,
    fasta_metadata: dict[str, dna.SequenceInfo],
    fasta_line_length: int,
    minimum_samples: int = 1,
) -> dict[str, dict[str, list[int]]]:
    _, positions_by_chromosome = _classify_sites(
        methylation_filepath,
        sequence_length,
        fasta_filepath,
        fasta_metadata,
        fasta_line_length,
        minimum_samples,
    )
    invalid_sites: dict[str, dict[str, list[int]]] = {}
    for chromosome, positions_by_category in positions_by_chromosome.items():
        invalid_positions: dict[str, list[int]] = {
            category: positions
            for category, positions in positions_by_category.items()
            if category in INVALID_CATEGORIES
        }
        if invalid_positions:
            invalid_sites[chromosome] = invalid_positions

    return invalid_sites


def record_invalid_sites(
    methylation_filepaths: list[Path],
    fasta_filepath: Path,
    sequence_length: int,
    minimum_samples: int = 1,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    fasta_line_length: int = dna.determine_line_length(fasta_filepath)
    fasta_metadata: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        fasta_filepath
    )
    sites_by_experiment: dict[str, dict[str, dict[str, list[int]]]] = {}
    for methylation_filepath in methylation_filepaths:
        sites_by_experiment[methylation_filepath.name.split(".")[0]] = (
            _find_invalid_sites(
                methylation_filepath,
                sequence_length,
                fasta_filepath,
                fasta_metadata,
                fasta_line_length,
                minimum_samples,
            )
        )

    return sites_by_experiment


def compare_sites(
    invalid_sites: dict[str, dict[str, dict[str, list[int]]]],
    deepcpg_data_directory: Path,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    missing_sites = {}
    for experiment, sites_by_chromosome in invalid_sites.items():
        missing_sites[experiment] = {}
        for chromosome, sites_by_reason in sites_by_chromosome.items():
            missing_sites[experiment][chromosome] = ExperimentStatistics()
            chromosome_filepaths: list[Path] = [
                Path(filepath)
                for filepath in glob(
                    str(deepcpg_data_directory) + f"c{chromosome}_*.h5"
                )
            ]
            deepcpg_positions = pl.Series(dtype=pl.Int32)
            with h5py.File(chromosome_filepaths[0]) as dataset:
                output_names = dataset["outputs"]["cpg"].keys()

            deepcpg_outputs = pl.DataFrame(
                schema={group: pl.Int8 for group in output_names}
            )

            for filepath in chromosome_filepaths:
                with h5py.File(filepath) as dataset:
                    deepcpg_positions.extend(
                        pl.Series(dataset["pos"], dtype=pl.Int32)
                    )
                    deepcpg_outputs.extend(
                        pl.DataFrame(
                            {
                                group: pl.Series(
                                    dataset["outputs"]["cpg"], dtype=pl.Int8
                                )
                            }
                            for group in output_names
                        )
                    )

            matching_experiment_split: list[str] = experiment.split("_")
            matching_experiment_name: str = (
                matching_experiment_split[1]
                + "_"
                + matching_experiment_split[2]
            )
            for reason, new_positions in sites_by_reason.items():
                for position in new_positions:
                    index: int = deepcpg_positions.index_of(position)
                    if deepcpg_positions[matching_experiment_name][index] != -1:
                        statistics: ExperimentStatistics = missing_sites[
                            experiment
                        ][chromosome]
                        statistic: str = _STATISTIC_BY_CATEGORY[reason]
                        setattr(
                            statistics,
                            statistic,
                            getattr(statistics, statistic) + 1,
                        )

    return missing_sites
