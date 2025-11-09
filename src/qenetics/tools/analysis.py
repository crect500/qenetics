from dataclasses import dataclass
from pathlib import Path
import csv

from qenetics.tools import dna, methylation


@dataclass
class ExperimentStatistics:
    total_reads: int = 0
    invalid_by_minimum: int = 0
    invalid_by_sequence_length: int = 0
    invalid_by_missing_nucleotide: int = 0
    valid_methylated: int = 0
    valid_unmethylated: int = 0


def get_profile_counts(
    methylation_filepath: Path,
    sequence_length: int,
    fasta_filepath: Path,
    fasta_metadata: dict[str, dna.SequenceInfo],
    fasta_line_length: int,
    minimum_samples: int = 1,
) -> ExperimentStatistics:
    methylation_threshold: float = 0.5
    experiment_statistics = ExperimentStatistics()
    methylation_format: methylation.MethylationFormat = (
        methylation.determine_format(methylation_filepath)
    )
    with (
        methylation_filepath.open() as methylation_fd,
        fasta_filepath.open() as fasta_fd,
    ):
        for methylation_line in methylation_fd.readlines():
            experiment_statistics.total_reads += 1
            methylation_profile: methylation.MethylationInfo | None = (
                methylation.process_methylation_line(
                    methylation_line, methylation_format, minimum_samples
                )
            )
            if methylation_profile is None:
                experiment_statistics.invalid_by_minimum += 1
                continue

            sequence: str | None = dna.find_methylation_sequence(
                chromosome=methylation_profile.chromosome,
                position=methylation_profile.position,
                genome_metadata=fasta_metadata,
                file_descriptor=fasta_fd,
                sequence_length=sequence_length,
                line_length=fasta_line_length,
            )
            if sequence is None:
                experiment_statistics.invalid_by_sequence_length += 1
                continue

            if "N" in sequence:
                experiment_statistics.invalid_by_missing_nucleotide += 1
                continue

            if methylation_profile.methylation_ratio >= methylation_threshold:
                experiment_statistics.valid_methylated += 1
            else:
                experiment_statistics.valid_unmethylated += 1

    return experiment_statistics


def get_dataset_stats(
    methylation_filepaths: list[Path],
    sequence_length: int,
    fasta_filepath: Path,
    minimum_samples: int = 1,
    crlf: bool = False,
) -> dict[str, ExperimentStatistics]:
    fasta_line_length: int = dna.determine_line_length(fasta_filepath)
    fasta_metadata: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        fasta_filepath, crlf=crlf
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
    totals = ExperimentStatistics()
    with output_filepath.open("w") as fd:
        csv_writer = csv.DictWriter(
            fd,
            fieldnames=[
                "experiment_name",
                "total_reads",
                "invalid_by_minimum",
                "invalid_by_sequence_length",
                "invalid_by_missing_nucleotide",
                "valid_methylated",
                "valid_unmethylated",
            ],
        )
        csv_writer.writeheader()
        for experiment_name, statistics in experiment_statistics.items():
            csv_writer.writerow(
                {
                    "experiment_name": experiment_name,
                    "total_reads": statistics.total_reads,
                    "invalid_by_minimum": statistics.invalid_by_minimum,
                    "invalid_by_sequence_length": statistics.invalid_by_sequence_length,
                    "invalid_by_missing_nucleotide": statistics.invalid_by_missing_nucleotide,
                    "valid_methylated": statistics.valid_methylated,
                    "valid_unmethylated": statistics.valid_unmethylated,
                }
            )
            totals.total_reads += statistics.total_reads
            totals.invalid_by_minimum += statistics.invalid_by_minimum
            totals.invalid_by_sequence_length += (
                statistics.invalid_by_sequence_length
            )
            totals.invalid_by_missing_nucleotide += (
                statistics.invalid_by_missing_nucleotide
            )
            totals.valid_methylated += statistics.valid_methylated
            totals.valid_unmethylated += statistics.valid_unmethylated

        csv_writer.writerow(
            {
                "experiment_name": "total",
                "total_reads": totals.total_reads,
                "invalid_by_minimum": totals.invalid_by_minimum,
                "invalid_by_sequence_length": totals.invalid_by_sequence_length,
                "invalid_by_missing_nucleotide": totals.invalid_by_missing_nucleotide,
                "valid_methylated": totals.valid_methylated,
                "valid_unmethylated": totals.valid_unmethylated,
            }
        )
