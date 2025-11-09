from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import h5py
import numpy as np
from numpy.typing import NDArray
import pytest

from qenetics.tools import data


def test_H5CpGDataset(test_dataset_directory: Path) -> None:
    test_files: list[Path] = [
        test_dataset_directory / f"chr{i}.h5" for i in ["1", "2"]
    ]
    dataset = data.H5CpGDataset(test_files)
    assert dataset.data.shape == (8, 8, 4)
    assert dataset.data.sum() == 64.0
    assert dataset.labels.shape == (8, 2)
    assert dataset.labels.sum() == 8
    assert len(dataset) == 8
    cpg_data, labels = dataset[0]
    assert (cpg_data == data.nucleotide_string_to_numpy("ATCGATCG")).all()
    assert list(labels) == [0, 0]


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
    [("A", 0), ("T", 1), ("C", 2), ("G", 3), ("N", -1)],
)
def test_nucleotide_character_to_numpy(
    nucleotide: str, expected_int: int
) -> None:
    if expected_int >= 0:
        expected_array: NDArray[int] = np.array([0] * 4, dtype=int)
        expected_array[expected_int] = 1
        assert (
            data.nucleotide_character_to_numpy(nucleotide) == expected_array
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
        ("NATCGN", []),
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
