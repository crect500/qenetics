from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import h5py
import numpy as np
from numpy.typing import NDArray
import pytest
from torch.utils.data import DataLoader

from qenetics.tools import data


def test_H5CpGDataset(test_qcpg_dataset_directory: Path) -> None:
    test_files: list[Path] = [
        test_qcpg_dataset_directory / f"chr{i}.h5" for i in ["1", "2"]
    ]
    dataset = data.H5CpGDataset(test_files)
    assert dataset.data.shape == (16, 10, 4)
    assert dataset.data.sum() == 128.0
    assert dataset.labels.shape == (16, 4)
    assert dataset.labels.sum() == 24
    assert len(dataset) == 16
    cpg_data, labels = dataset[0]
    assert (cpg_data == data.nucleotide_string_to_numpy("NATCGNATCG")).all()
    assert list(labels) == [0.0, 0.0, 1.0, 1.0]


def test_h5_cpg_data_loader(test_qcpg_dataset_directory: Path) -> None:
    test_files: list[Path] = [
        test_qcpg_dataset_directory / f"chr{i}.h5" for i in ["1", "2"]
    ]
    data_loader = DataLoader(data.H5CpGDataset(test_files), batch_size=1)
    for test_samples in data_loader:
        test_sequences, test_labels = test_samples
        assert test_sequences.shape == (1, 10, 4)
        assert test_labels.shape == (1, 4)

    data_loader = DataLoader(data.H5CpGDataset(test_files), batch_size=2)
    for test_samples in data_loader:
        test_sequences, test_labels = test_samples
        assert test_sequences.shape == (2, 10, 4)
        assert test_labels.shape == (2, 4)


def test_retrieve_chromosome_sequences() -> None:
    experiment_names: list[str] = ["test1", "test2", "test3"]
    unique_nucleotides_quantity: int = 4
    profiles_by_position: dict[int, dict[str, float]] = {
        1: {"test1": 0.0, "test2": 0.25},
        2: {"test2": 0.5},
        3: {"test1": 0.75},
        4: {"test3": 1.0},
    }
    sequence_length: int = 8
    valid_sequence: str = "ACTCGCTG"
    invalid_sequence: str = "ACTTTCTG"
    nan_sequence: str = "NNNNNNNN"
    with mock.patch(
        "qenetics.tools.dna.find_methylation_sequence"
    ) as mock_find:
        mock_find.side_effect = [
            valid_sequence,
            invalid_sequence,
            nan_sequence,
            valid_sequence,
        ]
        sequences, methylation_ratios = data._retrieve_chromosome_sequences(
            profiles_by_position=profiles_by_position,
            chromosome="1",
            fasta_file_descriptor=None,
            fasta_metadata={},
            fasta_line_length=20,
            sequence_length=sequence_length,
            experiment_names=experiment_names,
        )

    assert sequences.shape == (2, sequence_length, unique_nucleotides_quantity)
    assert methylation_ratios[0][0] == 0.0
    assert methylation_ratios[1][2] == 1.0


@pytest.mark.parametrize(
    ("nucleotide", "expected_int"),
    [("A", 0), ("T", 1), ("C", 2), ("G", 3), ("N", -1), ("x", -2)],
)
def test_nucleotide_character_to_numpy(
    nucleotide: str, expected_int: int
) -> None:
    if expected_int > -1:
        expected_array: NDArray[int] = np.array([0] * 4, dtype=int)
        expected_array[expected_int] = 1
        assert (
            data.nucleotide_character_to_numpy(nucleotide) == expected_array
        ).all()
    elif expected_int == -1:
        assert (
            data.nucleotide_character_to_numpy(nucleotide)
            == np.array([0] * 4, dtype=int)
        ).all()
    else:
        with pytest.raises(
            ValueError,
            match=f"{nucleotide} is not a valid nucleotide designator",
        ):
            _ = data.nucleotide_character_to_numpy(nucleotide)


@pytest.mark.parametrize(
    ("sequence", "expected_array"),
    [
        ("A", [0]),
        ("AT", [0, 1]),
        ("ATC", [0, 1, 2]),
        ("ATCG", [0, 1, 2, 3]),
        ("NATCGN", [-1, 0, 1, 2, 3, -1]),
    ],
)
def test_nucleotide_string_to_numpy(
    sequence: str, expected_array: list[int]
) -> None:
    one_hot_matrix: NDArray[int] = data.nucleotide_string_to_numpy(sequence)
    if len(expected_array) == 0:
        assert one_hot_matrix is None
    else:
        working_matrix: list[list[int]] = []
        for nucleotide in expected_array:
            working_array: list[int] = [0] * 4
            if nucleotide != -1:
                working_array[nucleotide] = 1
            working_matrix.append(working_array)
        assert (one_hot_matrix == np.array(working_matrix, dtype=int)).all()


@pytest.mark.parametrize(
    ("nucleotide_integer"),
    [0, 1, 2, 3, -1, -2],
)
def test_nucleotide_integer_to_numpy(nucleotide_integer: int) -> None:
    if nucleotide_integer > -1:
        expected_array: NDArray[int] = np.array([0] * 4, dtype=int)
        expected_array[nucleotide_integer] = 1
        assert (
            data.nucleotide_integer_to_numpy(nucleotide_integer)
            == expected_array
        ).all()
    elif nucleotide_integer == -1:
        assert (
            data.nucleotide_integer_to_numpy(nucleotide_integer)
            == np.array([0] * 4, dtype=int)
        ).all()
    else:
        with pytest.raises(
            ValueError,
            match=f"{nucleotide_integer} is not a valid nucleotide designator",
        ):
            _ = data.nucleotide_integer_to_numpy(nucleotide_integer)


@pytest.mark.parametrize(
    ("sequence"),
    [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [-1, 0, 1, 2, 3, -1],
    ],
)
def test_nucleotide_array_to_numpy(sequence: list[int]) -> None:
    one_hot_matrix: NDArray[int] = data.nucleotide_array_to_numpy(sequence)
    working_matrix: list[list[int]] = []
    for nucleotide in sequence:
        working_array: list[int] = [0] * 4
        if nucleotide != -1:
            working_array[nucleotide] = 1
        working_matrix.append(working_array)
    assert (one_hot_matrix == np.array(working_matrix, dtype=int)).all()


@pytest.mark.parametrize(
    ("threshold", "quantity_methylated"), [(0.0, 3), (0.5, 2), (1.0, 1)]
)
def test_samples_to_numpy(
    threshold: float, quantity_methylated: int, test_input_file: Path
) -> None:
    valid_samples: int = 3
    sequence_length: int = 12
    unique_nucleotides_quantity: int = 4
    samples, methylations = data.samples_to_numpy(test_input_file, threshold)
    assert np.sum(methylations) == quantity_methylated
    assert samples.shape == (
        valid_samples,
        sequence_length,
        unique_nucleotides_quantity,
    )


@pytest.mark.parametrize(
    ("sequence", "expected_result"),
    [
        ("ACGT", True),
        ("ATCGTA", True),
        ("AGCT", False),
        ("CGAT", False),
        ("ATCG", False),
    ],
)
def test_validate_sequence(sequence: str, expected_result: bool) -> None:
    assert data._validate_sequence(sequence) == expected_result


def test_create_h5_dataset() -> None:
    experiment_names: list[str] = ["test1", "test2", "test3"]
    sequences: np.ndarray = np.array(
        [[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]], dtype=int
    )
    methylation_ratios: np.ndarray = np.array(
        [[0.0, np.nan, np.nan], [np.nan, 0.50, 1.0]]
    )
    with TemporaryDirectory() as temp_dir:
        temp_h5_filepath = Path(temp_dir) / "chr1.h5"
        data._create_h5_dataset(
            temp_h5_filepath, sequences, methylation_ratios, experiment_names
        )
        with h5py.File(temp_h5_filepath) as fd:
            assert fd["methylation_sequences"].shape == (2, 2, 4)
            assert fd["methylation_ratios"]["test1"][0] == 0.0
            assert np.isnan(fd["methylation_ratios"]["test1"][1])
            assert np.isnan(fd["methylation_ratios"]["test2"][0])
            assert fd["methylation_ratios"]["test2"][1] == 0.5
            assert np.isnan(fd["methylation_ratios"]["test3"][0])
            assert fd["methylation_ratios"]["test3"][1] == 1.0
