from pathlib import Path

import pytest

from qenetics.tools import dna, data


@pytest.fixture
def test_fasta_metadata() -> dict[str, dna.SequenceInfo]:
    return {
        "1": dna.SequenceInfo(length=44, is_chromosome=True, file_position=50),
        "2": dna.SequenceInfo(length=88, is_chromosome=True, file_position=145),
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
def test_deepcpg_dataset_directory() -> Path:
    return Path("tests/test_files/test_deepcpg_dataset")


@pytest.fixture
def test_qcpg_dataset_directory() -> Path:
    return Path("tests/test_files/test_qcpg_dataset")


@pytest.fixture
def test_single_amplitude_dataset_directory() -> Path:
    return Path("tests/test_files/test_single_amplitude_dataset")


@pytest.fixture
def test_h5_loader() -> data.H5CpGDataset:
    return data.H5CpGDataset(
        [
            Path("tests/test_files/test_qcpg_dataset/chr1.h5"),
            Path("tests/test_files/test_qcpg_dataset/chr2.h5"),
        ]
    )
