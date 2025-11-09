from pathlib import Path

from qenetics.tools import dna, analysis


def test_get_profile_counts(
    test_methylation_file: Path, test_fasta_file: Path
) -> None:
    fasta_metadata: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        test_fasta_file, crlf=True
    )
    fasta_line_length: int = dna.determine_line_length(test_fasta_file)
    experiment_statistics: analysis.ExperimentStatistics = (
        analysis.get_profile_counts(
            test_methylation_file,
            4,
            test_fasta_file,
            fasta_metadata,
            fasta_line_length,
        )
    )
    assert experiment_statistics.total_reads == 6
    assert experiment_statistics.invalid_by_minimum == 0
    assert experiment_statistics.invalid_by_sequence_length == 2
    assert experiment_statistics.invalid_by_missing_nucleotide == 0
    assert experiment_statistics.valid_methylated == 3
    assert experiment_statistics.valid_unmethylated == 1


def test_get_dataset_stats(
    test_fasta_file: Path, test_methylation_file: Path
) -> None:
    filelist: list[Path] = [test_methylation_file, test_methylation_file]
    stats: dict[str, analysis.ExperimentStatistics] = (
        analysis.get_dataset_stats(filelist, 2, test_fasta_file, crlf=True)
    )
    assert list(stats.keys()) == ["test_methylation_profile"]
