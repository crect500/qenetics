from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import h5py
import numpy as np
import pytest
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qenetics.tools import data


def test_QuantumTorchDataset(
    test_qcpg_dataset_directory: Path, test_methylation_h5_file: Path
) -> None:
    tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
    test_files: list[Path] = [
        test_qcpg_dataset_directory / f"chr{i}.h5" for i in ["1", "2"]
    ]
    dataset = data.QuantumTorchDataset(test_files, allow_N=True)
    assert dataset.data.shape == (16, 10)
    assert dataset.labels.shape == (16, 4)
    assert dataset.labels.sum() == 24

    dataset = data.QuantumTorchDataset(
        test_files, encoding=data.ONEHOT_ENCODING_STR, allow_N=True
    )
    assert dataset.data.shape == (16, 10, 4)
    assert dataset.data.sum() == 128.0
    assert dataset.labels.shape == (16, 4)
    assert dataset.labels.sum() == 24

    dataset = data.QuantumTorchDataset(
        test_files, encoding=data.BPE_ENCODING_STR, tokenizer=tokenizer
    )
    assert dataset.data.shape == (16, 10)

    dataset = data.QuantumTorchDataset(
        [test_methylation_h5_file, test_methylation_h5_file]
    )
    assert dataset.data.shape == (8, 8)

    dataset = data.QuantumTorchDataset(
        [test_methylation_h5_file, test_methylation_h5_file],
        encoding=data.ONEHOT_ENCODING_STR,
    )
    assert dataset.data.shape == (8, 8, 4)
    assert dataset.data.sum() == 64.0

    dataset = data.QuantumTorchDataset(
        [test_methylation_h5_file, test_methylation_h5_file],
        encoding=data.BPE_ENCODING_STR,
        tokenizer=tokenizer,
    )
    assert dataset.data.shape == (8, 8)


def test_h5_cpg_data_loader(test_qcpg_dataset_directory: Path) -> None:
    test_files: list[Path] = [
        test_qcpg_dataset_directory / f"chr{i}.h5" for i in ["1", "2"]
    ]
    data_loader = DataLoader(data.QuantumTorchDataset(test_files), batch_size=1)
    for test_samples in data_loader:
        test_sequences, test_labels = test_samples
        assert test_sequences.shape == (1, 10, 4)
        assert test_labels.shape == (1, 4)

    data_loader = DataLoader(data.QuantumTorchDataset(test_files), batch_size=2)
    for test_samples in data_loader:
        test_sequences, test_labels = test_samples
        assert test_sequences.shape == (2, 10, 4)
        assert test_labels.shape == (2, 4)


@pytest.mark.parametrize(
    ("filestrings", "expected_result"),
    [
        ([], "empty"),
        (["unsupported.f6"], "unsupported"),
        (["h5_file.h5", "csv_file.csv"], "inhomogeneous"),
        (["file.h5"], "h5"),
        (["file1.h5", "file2.h5"], "h5"),
    ],
)
def test_check_file_types(filestrings: list[str], expected_result: str) -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        filepaths: list[Path] = [
            temp_path / filestring for filestring in filestrings
        ]

    if expected_result == "empty":
        with pytest.raises(ValueError, match="No filepaths provided"):
            _ = data._check_file_types(filepaths)
    elif expected_result == "unsupported":
        with pytest.raises(
            NotImplementedError,
            match=r"Unsupported file format or mixed file formats in files",
        ):
            _ = data._check_file_types(filepaths)
    elif expected_result == "inhomogeneous":
        with pytest.raises(
            ValueError, match=r"Inhomogeneous file type found for filepath"
        ):
            _ = data._check_file_types(filepaths)
    else:
        assert data._check_file_types(filepaths) == expected_result


def test_find_h5_sample_key(
    test_inputs_h5_file: Path, test_methylation_h5_file: Path
) -> None:
    with h5py.File(test_inputs_h5_file) as dataset:
        assert data._find_h5_samples_key(dataset) == data.INPUTS_STR
    with h5py.File(test_methylation_h5_file) as dataset:
        assert data._find_h5_samples_key(dataset) == data.METHYLATION_STR
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "invalid.h5"
        with h5py.File(temp_path, "w") as fd:
            fd.create_dataset("invalid_name", shape=(10, 16), dtype="i4")

        with (
            h5py.File(temp_path) as dataset,
            pytest.raises(
                RuntimeError, match="Sequence samples dataset not found"
            ),
        ):
            _ = data._find_h5_samples_key(dataset)


def test_determine_h5_dimensions_and_type(
    test_inputs_h5_file: Path, test_methylation_h5_file: Path
) -> None:
    assert data._determine_h5_dimensions_and_type(
        test_inputs_h5_file, data.INPUTS_STR
    ) == (2, np.int64)
    assert data._determine_h5_dimensions_and_type(
        test_methylation_h5_file, data.METHYLATION_STR
    ) == (3, np.int8)


@pytest.mark.parametrize(
    ("dtype", "expected_result"),
    [
        (np.int8, True),
        (np.int16, True),
        (np.int32, True),
        (np.int64, True),
        (np.float32, False),
    ],
)
def test_check_np_int(
    dtype: np.typing.DTypeLike, expected_result: bool
) -> None:
    assert data._check_np_int(dtype) == expected_result


def test_determine_h5_sample_encoding(
    test_inputs_h5_file: Path, test_methylation_h5_file: Path
) -> None:
    assert (
        data._determine_h5_sample_encoding(
            [test_inputs_h5_file, test_inputs_h5_file], data.INPUTS_STR
        )
        == data.TOKEN_ENCODING_STR
    )
    assert (
        data._determine_h5_sample_encoding(
            [test_methylation_h5_file, test_methylation_h5_file],
            data.METHYLATION_STR,
        )
        == data.ONEHOT_ENCODING_STR
    )


def test_determine_sample_encoding(test_inputs_h5_file: Path) -> None:
    with mock.patch(
        "qenetics.tools.data._determine_h5_sample_encoding"
    ) as mock_h5:
        mock_h5.return_value = True
        _ = data._determine_sample_encoding(
            [test_inputs_h5_file, test_inputs_h5_file],
            data.H5_STR,
            data.INPUTS_STR,
        )
        mock_h5.assert_called_once()

    with pytest.raises(
        NotImplementedError, match=r"File parsing not implemented for file type"
    ):
        _ = data._determine_sample_encoding([], "invalid_format", "")


def test_determine_dataset_sample_quantity(
    test_inputs_h5_file: Path, test_methylation_h5_file: Path
) -> None:
    with h5py.File(test_inputs_h5_file) as dataset:
        assert (
            data._determine_dataset_sample_quantity(dataset, data.INPUTS_STR)
            == 10
        )

    with h5py.File(test_methylation_h5_file) as dataset:
        assert (
            data._determine_dataset_sample_quantity(
                dataset, data.METHYLATION_STR
            )
            == 4
        )


def test_determine_sample_quantity(test_inputs_h5_file: Path) -> None:
    assert (
        data._determine_sample_quantity(
            [test_inputs_h5_file, test_inputs_h5_file],
            data.H5_STR,
            data.INPUTS_STR,
        )
        == 20
    )


def test_convert_token_dataset_to_onehot(test_inputs_h5_file: Path) -> None:
    with h5py.File(test_inputs_h5_file) as dataset:
        samples: Tensor = data._convert_token_dataset_to_onehot(
            dataset, data.INPUTS_STR
        )

    assert samples.shape == (10, 16, 4)

    with (
        h5py.File(test_inputs_h5_file) as dataset,
        pytest.raises(
            ValueError,
            match="Conversion to one-hot encoding is not supported for H5 structure invalid",
        ),
    ):
        _ = data._convert_token_dataset_to_onehot(dataset, "invalid")


def test_convert_onehot_dataset_to_token(
    test_methylation_h5_file: Path,
) -> None:
    with h5py.File(test_methylation_h5_file) as dataset:
        samples: Tensor = data._convert_onehot_dataset_to_token(
            dataset, data.METHYLATION_STR
        )

    assert samples.shape == (4, 8)

    with (
        h5py.File(test_methylation_h5_file) as dataset,
        pytest.raises(
            ValueError,
            match="Conversion to token encoding is not supported for H5 structure invalid",
        ),
    ):
        _ = data._convert_onehot_dataset_to_token(dataset, "invalid")


def test_convert_token_dataset_to_bpe(test_inputs_h5_file: Path) -> None:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        "PoetschLab/GROVER"
    )
    with h5py.File(test_inputs_h5_file) as dataset:
        samples: Tensor = data._convert_token_dataset_to_bpe(
            dataset, tokenizer, data.INPUTS_STR
        )

    assert samples.shape == (10, 16)

    with (
        h5py.File(test_inputs_h5_file) as dataset,
        pytest.raises(
            ValueError,
            match="BPE token conversion is not supported for H5 structure invalid",
        ),
    ):
        _ = data._convert_token_dataset_to_bpe(dataset, tokenizer, "invalid")


def test_convert_onehot_dataset_to_bpe(test_methylation_h5_file: Path) -> None:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        "PoetschLab/GROVER"
    )
    with h5py.File(test_methylation_h5_file) as dataset:
        samples: Tensor = data._convert_onehot_dataset_to_bpe(
            dataset, tokenizer, data.METHYLATION_STR
        )

    assert samples.shape == (4, 8)

    with (
        h5py.File(test_methylation_h5_file) as dataset,
        pytest.raises(
            ValueError,
            match="BPE token conversion is not supported for H5 structure invalid",
        ),
    ):
        _ = data._convert_onehot_dataset_to_bpe(dataset, tokenizer, "invalid")


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


@pytest.mark.parametrize(
    ("original_length", "new_length", "expected_start", "expected_end"),
    [
        (1, 1, None, 1),
        (2, 1, 1, None),
        (4, 2, 1, 3),
        (10, 4, 3, 7),
        (5, 3, 1, 4),
        (9, 5, 2, 7),
        (1001, 33, 484, 517),
    ],
)
def test_find_bounds(
    original_length: int,
    new_length: int,
    expected_start: int,
    expected_end: int,
) -> None:
    if expected_start is None:
        with pytest.raises(ValueError, match=r"is the same or greater"):
            _ = data._find_bounds(original_length, new_length)
    elif expected_end is None:
        with pytest.raises(
            ValueError, match=r"divisibility by 2 must be the same as"
        ):
            _ = data._find_bounds(original_length, new_length)
    else:
        start, end = data._find_bounds(original_length, new_length)
        assert (start, end) == (expected_start, expected_end)


def test_reduce_sample_size(test_single_amplitude_dataset_directory) -> None:
    test_filepath: Path = test_single_amplitude_dataset_directory / "chr1.h5"
    new_size: int = 4
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data._reduce_sample_size(test_filepath, temp_path, new_size)
        with (
            h5py.File(test_filepath) as original_dataset,
            h5py.File(
                temp_path / (test_filepath.stem + "_" + str(new_size) + ".h5")
            ) as new_dataset,
        ):
            original_sequences: h5py.Dataset = original_dataset[
                data.METHYLATION_SEQUENCES_KEY
            ]
            original_ratios: h5py.Dataset = original_dataset[
                data.METHYLATION_RATIOS_KEY
            ]
            new_sequences: h5py.Dataset = new_dataset[
                data.METHYLATION_SEQUENCES_KEY
            ]
            new_ratios: h5py.Dataset = new_dataset[data.METHYLATION_RATIOS_KEY]
            assert new_sequences.shape == (
                original_sequences.shape[0],
                new_size,
                original_sequences.shape[2],
            )
            assert original_ratios.shape == new_ratios.shape
