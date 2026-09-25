from csv import DictReader
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from qenetics.deepcpg import deepcpg_utils


@pytest.mark.parametrize(
    ("line", "chromosome", "position", "ratio", "count"),
    [
        ("1\t2\t2\t100\t5\t0", "1", 2, 1.0, 5),
        ("X\t3\t3\t33.33333\t1\t2", "X", 3, 0.3333333, 3),
    ],
)
def test_process_methylation_line(
    line: str, chromosome: str, position: int, ratio: float, count: int
) -> None:
    minimum_count: int = 4
    methylation_profile: deepcpg_utils.MethylationInfo | None = (
        deepcpg_utils._process_methylation_line(line, minimum_count)
    )
    if count >= minimum_count:
        assert methylation_profile.chromosome == chromosome
        assert methylation_profile.position == position
        assert methylation_profile.methylation_ratio == pytest.approx(ratio)
        assert methylation_profile.experiment_count == count
    else:
        assert methylation_profile is None


@pytest.mark.parametrize(
    ("minimum_samples", "profiles_quantity"), [(1, 6), (2, 3), (3, 1)]
)
def test_retrieve_methylation_data(
    minimum_samples: int, profiles_quantity: int, test_methylation_file: Path
) -> None:
    methylation_profiles: list[deepcpg_utils.MethylationInfo] = list(
        deepcpg_utils.retrieve_methylation_data(
            test_methylation_file, minimum_samples
        )
    )
    assert len(methylation_profiles) == profiles_quantity


def test_write_sequence_row() -> None:
    sequence: str = "abc123"
    with TemporaryDirectory() as temp_dir:
        write_filepath = Path(temp_dir) / "output.csv"
        with open(write_filepath, "w") as fd:
            deepcpg_utils._write_sequence_row(fd, sequence, 0.5)

        with open(write_filepath) as fd:
            line: str = fd.readline()
            assert line == f"{sequence},0.5\n"


def test_write_all_deepcpg_methylations() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        methylation_filepath = temp_path / "cell.cov.txt"
        methylation_filepath.write_text(
            "1\t5\t5\t100\t2\t0\n1\t9\t9\t50\t1\t1\n1\t13\t13\t0\t0\t1\n"
        )
        output_filepath = temp_path / "cell.tsv"
        deepcpg_utils.write_all_deepcpg_methylations(
            methylation_filepath,
            output_filepath,
            minimum_samples=2,
            threshold=0.75,
        )

        assert output_filepath.read_text() == "1\t5\t1\n1\t9\t0\n"


#                   0         1         2
#                   01234567890123456789012345678
CHROMOSOME_1: str = "AAAACGAAAATTCGTTTTNNCGAAAACGA"
CHROMOSOME_2: str = "GGCGGG"
SEQUENCE_LENGTH: int = 5

# 1-based positions and read counts of each methylation call.
METHYLATION_LINES: list[str] = [
    "chr1\t5\t5\t50\t1\t1",  # CpG at 4, forward strand
    "chr1\t6\t6\t100\t1\t0",  # CpG at 4, reverse strand: 2 of 3 methylated
    "chr1\t13\t13\t0\t0\t1",  # CpG at 12, 1 read
    "chr1\t21\t21\t100\t1\t0",  # CpG at 20: window holds N
    "chr1\t28\t28\t75\t3\t1",  # CpG at 26, reverse strand
    "chr1\t2\t2\t100\t1\t0",  # not a CpG
    "chr2\t3\t3\t100\t1\t0",  # CpG at 2
]


def _write_genome(fasta_filepath: Path) -> None:
    with fasta_filepath.open("w", newline="\n") as fd:
        for name, sequence in [("1", CHROMOSOME_1), ("2", CHROMOSOME_2)]:
            fd.write(
                f">{name} dna:chromosome chromosome:GRCm38:{name}:1:"
                f"{len(sequence)}:1 REF\n"
            )
            for start in range(0, len(sequence), 8):
                fd.write(sequence[start : start + 8] + "\n")


def _read_rows(csv_filepath: Path) -> list[tuple[str, float]]:
    with open(csv_filepath) as fd:
        reader = DictReader(fd)
        assert reader.fieldnames == ["sequence", "ratio_methylated"]
        return [
            (row["sequence"], float(row["ratio_methylated"])) for row in reader
        ]


@pytest.mark.parametrize(
    ("chromosomes", "minimum_samples", "expected_rows"),
    [
        (["1"], 1, [("AACGA", 2 / 3), ("TTCGT", 0.0), ("AACGA", 0.75)]),
        (["1"], 2, [("AACGA", 2 / 3), ("AACGA", 0.75)]),
        (["2"], 1, [("GGCGG", 1.0)]),
        (
            ["1", "2"],
            1,
            [
                ("AACGA", 2 / 3),
                ("TTCGT", 0.0),
                ("AACGA", 0.75),
                ("GGCGG", 1.0),
            ],
        ),
        (["X"], 1, []),
    ],
)
def test_create_sequence_dataset(
    chromosomes: list[str],
    minimum_samples: int,
    expected_rows: list[tuple[str, float]],
) -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fasta_filepath = temp_path / "genome.fa"
        _write_genome(fasta_filepath)
        methylation_filepath = temp_path / "cell.cov.txt"
        methylation_filepath.write_text("\n".join(METHYLATION_LINES) + "\n")
        output_filepath = temp_path / "output.csv"
        deepcpg_utils.create_sequence_dataset(
            methylation_filepath=methylation_filepath,
            fasta_file=fasta_filepath,
            sequence_length=SEQUENCE_LENGTH,
            chromosomes=chromosomes,
            minimum_samples=minimum_samples,
            output_file=output_filepath,
        )
        rows = _read_rows(output_filepath)

    assert [sequence for sequence, _ in rows] == [
        sequence for sequence, _ in expected_rows
    ]
    assert [ratio for _, ratio in rows] == pytest.approx(
        [ratio for _, ratio in expected_rows]
    )


def test_create_dataset_from_directory() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fasta_filepath = temp_path / "genome.fa"
        _write_genome(fasta_filepath)
        methylation_directory = temp_path / "methylation"
        methylation_directory.mkdir()
        (methylation_directory / "GSM0000001_cellA.cov.txt").write_text(
            "\n".join(METHYLATION_LINES) + "\n"
        )
        output_directory = temp_path / "output"
        output_directory.mkdir()
        deepcpg_utils.create_dataset_from_directory(
            methylation_directory,
            fasta_filepath,
            SEQUENCE_LENGTH,
            output_directory,
            minimum_samples=2,
        )
        training_rows = _read_rows(output_directory / "training_cellA.csv")
        validation_rows = _read_rows(output_directory / "validation_cellA.csv")

    # Chromosome 1 is a training chromosome and 2 a validation chromosome.
    # The only site of chromosome 2 has 1 read, below the minimum of 2.
    assert [sequence for sequence, _ in training_rows] == ["AACGA", "AACGA"]
    assert validation_rows == []
