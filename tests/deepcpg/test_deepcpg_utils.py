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
    ("minimum_samples", "profiles_quantity"), [(1, 5), (2, 3), (3, 1)]
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
    methylation_info = deepcpg_utils.MethylationInfo(
        chromosome="1", position=0, methylation_ratio=0.5, experiment_count=2
    )
    with TemporaryDirectory() as temp_dir:
        write_filepath = Path(temp_dir) / "output.csv"
        with open(write_filepath, "w") as fd:
            deepcpg_utils._write_sequence_row(fd, sequence, methylation_info)

        with open(write_filepath) as fd:
            line: str = fd.readline()
            assert line == f"{sequence},{methylation_info.methylation_ratio}\n"


@pytest.mark.parametrize(
    ("chromosomes", "minimum_samples", "samples_written"),
    [(["1"], 1, 3), (["1"], 2, 2), (["2"], 1, 1), (["1", "X"], 1, 3)],
)
def test_create_sequence_dataset(
    chromosomes: list[str],
    minimum_samples: int,
    samples_written: int,
    test_methylation_file: Path,
    test_fasta_file: Path,
) -> None:
    sequence_length: int = 4
    with TemporaryDirectory() as temp_dir:
        output_filepath = Path(temp_dir) / "output.csv"
        deepcpg_utils.create_sequence_dataset(
            methylation_filepath=test_methylation_file,
            fasta_file=test_fasta_file,
            sequence_length=sequence_length,
            chromosomes=chromosomes,
            minimum_samples=minimum_samples,
            output_file=output_filepath,
        )
        with open(output_filepath) as fd:
            reader = DictReader(fd)
            assert "sequence" in reader.fieldnames
            assert "ratio_methylated" in reader.fieldnames
            assert len(list(reader)) == samples_written
