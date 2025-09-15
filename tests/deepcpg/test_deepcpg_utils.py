from csv import DictReader
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from qenetics.deepcpg import deepcpg_utils


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
