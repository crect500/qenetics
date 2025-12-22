from pathlib import Path

from qenetics.tools import analysis, dna, methylation


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


def test_add_to_invalid_dict() -> None:
    invalid_sites: dict[str, dict[str, list[int]]] = {}
    methylation_profile = methylation.MethylationInfo(
        "1",
        1,
        0.5,
        count_methylated=1,
        count_unmethylated=1,
        experiment_count=2,
    )
    analysis._add_to_invalid_dict(invalid_sites, methylation_profile, "key1")
    assert invalid_sites["1"]["key1"] == [1]

    methylation_profile.chromosome = "2"
    analysis._add_to_invalid_dict(invalid_sites, methylation_profile, "key1")
    assert list(invalid_sites.keys()) == ["1", "2"]

    methylation_profile.position = 2
    analysis._add_to_invalid_dict(invalid_sites, methylation_profile, "key1")
    assert invalid_sites["2"]["key1"] == [1, 2]


def test_find_invalid_sites(
    test_fasta_file: Path, test_methylation_file: Path
) -> None:
    fasta_line_length: int = dna.determine_line_length(test_fasta_file)
    fasta_metadata: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        test_fasta_file, crlf=True
    )
    invalid_sites: dict[str, dict[str, list[int]]] = (
        analysis._find_invalid_sites(
            test_methylation_file,
            8,
            test_fasta_file,
            fasta_metadata,
            fasta_line_length,
            2,
        )
    )

    assert list(invalid_sites.keys()) == ["1", "2", "X"]
    assert invalid_sites["1"]["invalid_by_minimum"] == [1, 3]
    assert invalid_sites["1"]["invalid_by_boundaries"] == [2]
    assert invalid_sites["2"]["invalid_by_missing"] == [5]
