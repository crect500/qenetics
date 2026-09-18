from pathlib import Path

import pytest
from transformers import AutoTokenizer

from qenetics.tools import data, dna


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
def test_inputs_h5_file() -> Path:
    return Path("tests/test_files/test_dataset/c1_032768-065536.h5")


@pytest.fixture
def test_methylation_h5_file() -> Path:
    return Path("tests/test_files/test_dataset/chr1.h5")


@pytest.fixture
def test_h5_loader() -> data.QuantumTorchDataset:
    return data.QuantumTorchDataset(
        [
            Path("tests/test_files/test_qcpg_dataset/chr1.h5"),
            Path("tests/test_files/test_qcpg_dataset/chr2.h5"),
        ],
        encoding=data.ONEHOT_ENCODING_STR,
        allow_N=True,
    )


@pytest.fixture(scope="package")
def grover_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("PoetschLab/GROVER")
