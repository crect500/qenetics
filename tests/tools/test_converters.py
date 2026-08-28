from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from h5py import File
from numpy.typing import NDArray

from qenetics.tools import converters


def test_determine_sequence_length(
    test_deepcpg_dataset_directory: Path,
) -> None:
    assert (
        converters._determine_sequence_length(test_deepcpg_dataset_directory)
        == 10
    )


def test_extract_deepcpg_experiment_to_qcpg(
    test_deepcpg_dataset_directory: Path,
    test_single_experiment_dataset_directory: Path,
) -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        converters.extract_deepcpg_experiment_to_qcpg(
            test_deepcpg_dataset_directory, temp_path, "experiment0"
        )
        assert len(list(temp_path.iterdir())) == len(
            list(test_deepcpg_dataset_directory.iterdir())
        )

        with File(
            test_single_experiment_dataset_directory / "chr1.h5"
        ) as expected_dataset:
            expected_dataset_sequences: NDArray = np.array(
                expected_dataset["methylation_sequences"]
            )
            expected_dataset_ratios: NDArray = np.array(
                expected_dataset["methylation_ratios"]
            )

        with File(temp_path / "chr1.h5") as actual_dataset:
            actual_dataset_sequenecs: NDArray = np.array(
                actual_dataset["methylation_sequences"]
            )
            actual_dataset_ratios: NDArray = np.array(
                actual_dataset["methylation_ratios"]
            )

        assert (expected_dataset_sequences == actual_dataset_sequenecs).all()
        assert (expected_dataset_ratios == actual_dataset_ratios).all()


@pytest.mark.parametrize(
    ("encoding_array", "include_zero", "expected_integer"),
    [
        ([0], False, None),
        ([0], True, 0),
        ([1], False, 0),
        ([1], True, 1),
        ([1, 0], False, 0),
        ([0, 1], False, 1),
        ([0, 0, 0, 0], False, None),
        ([0, 0, 0, 0], True, 0),
        ([1, 0, 0, 0], False, 0),
        ([0, 0, 0, 1], False, 3),
    ],
)
def test_one_hot_to_integer(
    encoding_array: list[int], include_zero: bool, expected_integer: int
) -> None:
    if expected_integer is None and not include_zero:
        with pytest.raises(
            ValueError, match=r"Improper one-hot encoding for array"
        ):
            converters._one_hot_to_integer(
                encoding_array, include_zero=include_zero
            )
    else:
        assert (
            converters._one_hot_to_integer(
                encoding_array, include_zero=include_zero
            )
            == expected_integer
        )


@pytest.mark.parametrize(
    ("one_hot_sequence", "expected_result"),
    [
        ([[1]], [0]),
        ([[1], [1]], [0, 0]),
        ([[1, 0], [0, 1]], [0, 1]),
        ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], [0, 1, 3]),
    ],
)
def test_one_hot_sequence_to_integers(
    one_hot_sequence: list[list[int]], expected_result: list[int]
) -> None:
    assert (
        converters._one_hot_sequence_to_integers(one_hot_sequence)
        == np.array(expected_result)
    ).all()


def test_h5_one_hot_to_integer() -> None:
    one_hot_filepath = Path("tests/test_files/test_qcpg_dataset/chr1.h5")
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        converters.h5_one_hot_to_integer(
            one_hot_filepath, temp_path, include_zero=True
        )
        with File(temp_path / one_hot_filepath.name) as dataset:
            assert set(dataset.keys()) == {
                "methylation_ratios",
                "methylation_sequences",
            }
            assert len(dataset["methylation_ratios"].keys())
