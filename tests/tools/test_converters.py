from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from h5py import File
from numpy.typing import NDArray

from qenetics.tools import converters


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
            converters.nucleotide_character_to_numpy(nucleotide)
            == expected_array
        ).all()
    elif expected_int == -1:
        assert converters.nucleotide_character_to_numpy(
            nucleotide
        ) == pytest.approx(np.array([1 / sqrt(2)] * 4, dtype=float))
        assert converters.nucleotide_character_to_numpy(
            nucleotide, encoding="basis"
        ) == pytest.approx(np.array([0] * 4, dtype=int))
    else:
        with pytest.raises(
            ValueError,
            match=f"{nucleotide} is not a valid nucleotide designator",
        ):
            _ = converters.nucleotide_character_to_numpy(nucleotide)


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
    one_hot_matrix: NDArray[int] = converters.nucleotide_string_to_numpy(
        sequence
    )
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
            converters.nucleotide_integer_to_numpy(nucleotide_integer)
            == expected_array
        ).all()
    elif nucleotide_integer == -1:
        assert (
            converters.nucleotide_integer_to_numpy(nucleotide_integer)
            == np.array([0] * 4, dtype=int)
        ).all()
    else:
        with pytest.raises(
            ValueError,
            match=f"{nucleotide_integer} is not a valid nucleotide designator",
        ):
            _ = converters.nucleotide_integer_to_numpy(nucleotide_integer)


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
    one_hot_matrix: NDArray[int] = converters.nucleotide_array_to_numpy(
        sequence
    )
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
    samples, methylations = converters.samples_to_numpy(
        test_input_file, threshold
    )
    assert np.sum(methylations) == quantity_methylated
    assert samples.shape == (
        valid_samples,
        sequence_length,
        unique_nucleotides_quantity,
    )


@pytest.mark.parametrize(
    ("nucleotide", "allow_n", "expected_result"),
    [
        (0, False, "A"),
        (1, False, "T"),
        (2, False, "C"),
        (3, False, "G"),
        (-1, False, ""),
        (-1, True, "N"),
        (-3, False, ""),
    ],
)
def test_integer_to_nucleotide_char(
    nucleotide: int, allow_n: bool, expected_result: str
) -> None:
    if expected_result:
        assert (
            converters._integer_to_nucleotide_char(nucleotide, allow_n)
            == expected_result
        )
    else:
        if nucleotide == -1:
            match_str: str = (
                f"Invalid value {nucleotide}. To allow nucleotide value"
            )
        else:
            match_str = f"Invalid value {nucleotide}"
        with pytest.raises(ValueError, match=match_str):
            _ = converters._integer_to_nucleotide_char(nucleotide, allow_n)


@pytest.mark.parametrize(
    ("values", "expected_result"),
    [([], ""), ([0], "A"), ([0, 1, 2, 3], "ATCG")],
)
def test_integer_array_to_nucleotide_str(
    values: list[int], expected_result: str
) -> None:
    assert converters.integer_array_to_nucleotide_str(values) == expected_result


@pytest.mark.parametrize(
    ("one_hot", "allow_n", "expected_result"),
    [
        ([1, 0, 0, 0], False, "A"),
        ([0, 1, 0, 0], False, "T"),
        ([0, 0, 1, 0], False, "C"),
        ([0, 0, 0, 1], False, "G"),
        ([0, 0, 0, 0], False, ""),
        ([0, 0, 0, 0], True, "N"),
        ([1, 0, 0, 1], False, ""),
    ],
)
def test_one_hot_to_nucleotide(
    one_hot: int, allow_n: bool, expected_result: str
) -> None:
    if expected_result:
        assert (
            converters._one_hot_to_nucleotide(one_hot, allow_n)
            == expected_result
        )
        assert (
            converters._one_hot_to_nucleotide(np.array(one_hot), allow_n)
            == expected_result
        )

    else:
        if one_hot == [0, 0, 0, 0]:
            match_str: str = r" To allow nucleotide value"
        else:
            match_str = r"Invalid one-hot encoding"
        with pytest.raises(ValueError, match=match_str):
            _ = converters._one_hot_to_nucleotide(one_hot, allow_n)


@pytest.mark.parametrize(
    ("one_hot_sequence", "expected_result"),
    [
        ([], ""),
        ([[1, 0, 0, 0]], "A"),
        ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "ATCG"),
    ],
)
def test_one_hot_sequence_to_nucleotide_str(
    one_hot_sequence: list[list[int]], expected_result: str
) -> None:
    assert (
        converters.one_hot_sequence_to_nucleotide_str(one_hot_sequence)
        == expected_result
    )
    assert (
        converters.one_hot_sequence_to_nucleotide_str(
            np.array(one_hot_sequence)
        )
        == expected_result
    )


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
def testone_hot_sequence_to_integers(
    one_hot_sequence: list[list[int]], expected_result: list[int]
) -> None:
    assert (
        converters.one_hot_sequence_to_integers(one_hot_sequence)
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
