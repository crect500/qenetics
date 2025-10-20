from pathlib import Path

import pytest

from qenetics.tools import cpg_sampler


@pytest.fixture
def test_fasta_metadata() -> dict[str, cpg_sampler.SequenceInfo]:
    return {
        "1": cpg_sampler.SequenceInfo(
            length=44, is_chromosome=True, file_position=50
        ),
        "2": cpg_sampler.SequenceInfo(
            length=88, is_chromosome=True, file_position=145
        ),
    }


@pytest.fixture
def test_fasta_file() -> Path:
    return Path("tests/test_files/test_sequence.fa")


@pytest.fixture
def test_methylation_file() -> Path:
    return Path("tests/test_files/test_methylation_profile.cov.txt")


@pytest.fixture
def test_sequences() -> list[str]:
    return [
        "ACTGACTGACTGACTG",
        "GTCAGTCAGTCAGTCA",
        "AAAATTTTCCCCGGGG",
        "GGGGCCCCTTTTAAAA",
    ]


@pytest.fixture
def test_input_file() -> Path:
    return Path("tests/test_files/test_input_data.csv")


@pytest.fixture
def test_dataset_directory() -> Path:
    return Path("tests/test_files/test_dataset")
