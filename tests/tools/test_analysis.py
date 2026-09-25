import csv
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import pytest

from qenetics.tools import analysis, data, dna

#            0         1         2         3
#            0123456789012345678901234567890
SEQUENCE: str = "CGAATTCGAATTNCGAATTCGAATTCGAACG"
SEQUENCE_LENGTH: int = 5
MINIMUM_SAMPLES: int = 2

# 1-based positions and read counts of each methylation call.
METHYLATION_LINES: list[str] = [
    "#chr\tstart\tend\tpercent\tmethylated\tunmethylated",
    "1\t1\t1\t100\t2\t0",  # CpG at 0: window before the chromosome start
    "1\t7\t7\t50\t1\t1",  # CpG at 6: forward strand
    "1\t8\t8\t100\t1\t0",  # CpG at 6: reverse strand, 2 of 3 reads methylated
    "1\t14\t14\t100\t2\t0",  # CpG at 13: window holds N
    "1\t20\t20\t50\t1\t1",  # CpG at 19: tie, so unmethylated
    "1\t26\t26\t100\t1\t0",  # CpG at 25: fewer reads than the minimum
    "1\t31\t31\t0\t0\t3",  # CpG at 29, reverse strand: window after the end
    "1\t3\t3\t100\t1\t0",  # not a CpG
    "Y\t10\t10\t100\t1\t0",  # chromosome missing from the reference
]


def _write_inputs(temp_path: Path) -> tuple[Path, Path]:
    fasta_filepath = temp_path / "genome.fa"
    with fasta_filepath.open("w", newline="\n") as fd:
        fd.write(
            f">1 dna:chromosome chromosome:GRCm38:1:1:{len(SEQUENCE)}:1 REF\n"
        )
        for start in range(0, len(SEQUENCE), 8):
            fd.write(SEQUENCE[start : start + 8] + "\n")

    methylation_directory = temp_path / "methylation"
    methylation_directory.mkdir()
    methylation_filepath = methylation_directory / "cell.cov.txt.gz"
    with gzip.open(methylation_filepath, "wt") as fd:
        fd.write("\n".join(METHYLATION_LINES) + "\n")

    return fasta_filepath, methylation_filepath


def _classify_arguments(
    fasta_filepath: Path, methylation_filepath: Path
) -> tuple:
    return (
        methylation_filepath,
        SEQUENCE_LENGTH,
        fasta_filepath,
        dna.extract_fasta_metadata(fasta_filepath),
        dna.determine_line_length(fasta_filepath),
        MINIMUM_SAMPLES,
    )


def test_get_profile_counts() -> None:
    with TemporaryDirectory() as temp_dir:
        fasta_filepath, methylation_filepath = _write_inputs(Path(temp_dir))
        experiment_statistics: analysis.ExperimentStatistics = (
            analysis.get_profile_counts(
                *_classify_arguments(fasta_filepath, methylation_filepath)
            )
        )

    assert experiment_statistics == analysis.ExperimentStatistics(
        total_reads=9,
        invalid_by_chromosome=1,
        invalid_by_non_cpg=1,
        invalid_by_minimum=1,
        invalid_by_sequence_length=2,
        invalid_by_missing_nucleotide=1,
        valid_methylated=1,
        valid_unmethylated=1,
    )


def test_find_invalid_sites() -> None:
    with TemporaryDirectory() as temp_dir:
        fasta_filepath, methylation_filepath = _write_inputs(Path(temp_dir))
        invalid_sites: dict[str, dict[str, list[int]]] = (
            analysis._find_invalid_sites(
                *_classify_arguments(fasta_filepath, methylation_filepath)
            )
        )

    assert invalid_sites == {
        "1": {
            analysis.INVALID_BY_NON_CPG: [3],
            analysis.INVALID_BY_MINIMUM: [26],
            analysis.INVALID_BY_BOUNDARIES: [1, 30],
            analysis.INVALID_BY_MISSING: [14],
        },
        "Y": {analysis.INVALID_BY_CHROMOSOME: [10]},
    }


def test_classify_sites_matches_h5_dataset() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fasta_filepath, methylation_filepath = _write_inputs(temp_path)
        _, positions_by_chromosome = analysis._classify_sites(
            *_classify_arguments(fasta_filepath, methylation_filepath)
        )
        data.create_h5_dataset_from_methylation_profiles(
            methylation_filepath.parent,
            fasta_filepath,
            temp_path,
            SEQUENCE_LENGTH,
            MINIMUM_SAMPLES,
        )
        with h5py.File(temp_path / "chr1.h5") as fd:
            h5_positions: list[int] = fd[data.POSITIONS_KEY][()].tolist()
            h5_labels: list[float] = fd[data.METHYLATION_RATIOS_KEY]["cell"][
                ()
            ].tolist()

    positions_by_category = positions_by_chromosome["1"]
    assert positions_by_category[analysis.VALID_METHYLATED] == [7]
    assert positions_by_category[analysis.VALID_UNMETHYLATED] == [20]
    assert h5_positions == [7, 20]
    assert h5_labels == [1.0, 0.0]


def test_get_dataset_stats() -> None:
    with TemporaryDirectory() as temp_dir:
        fasta_filepath, methylation_filepath = _write_inputs(Path(temp_dir))
        stats: dict[str, analysis.ExperimentStatistics] = (
            analysis.get_dataset_stats(
                [methylation_filepath], SEQUENCE_LENGTH, fasta_filepath, 2
            )
        )

    assert list(stats.keys()) == ["cell"]
    assert stats["cell"].valid_methylated == 1


def test_write_experiment_statistics() -> None:
    statistics: dict[str, analysis.ExperimentStatistics] = {
        "cellA": analysis.ExperimentStatistics(
            total_reads=3, invalid_by_non_cpg=1, valid_methylated=2
        ),
        "cellB": analysis.ExperimentStatistics(
            total_reads=2, invalid_by_chromosome=1, valid_unmethylated=1
        ),
    }
    with TemporaryDirectory() as temp_dir:
        output_filepath = Path(temp_dir) / "statistics.csv"
        analysis.write_experiment_statistics(statistics, output_filepath)
        with output_filepath.open() as fd:
            rows: list[dict[str, str]] = list(csv.DictReader(fd))

    assert [row["experiment_name"] for row in rows] == [
        "cellA",
        "cellB",
        "total",
    ]
    assert rows[2] == {
        "experiment_name": "total",
        "total_reads": "5",
        "invalid_by_chromosome": "1",
        "invalid_by_non_cpg": "1",
        "invalid_by_minimum": "0",
        "invalid_by_sequence_length": "0",
        "invalid_by_missing_nucleotide": "0",
        "valid_methylated": "2",
        "valid_unmethylated": "1",
    }


@pytest.mark.parametrize(
    "category", [*analysis.INVALID_CATEGORIES, analysis.VALID_METHYLATED]
)
def test_statistic_by_category(category: str) -> None:
    statistic_names: set[str] = {
        field for field in analysis.ExperimentStatistics.__dataclass_fields__
    }
    assert analysis._STATISTIC_BY_CATEGORY[category] in statistic_names
