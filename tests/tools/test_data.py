from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import h5py
import numpy as np
import pytest
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qenetics.tools import converters, data


def test_QuantumTorchDataset(
    test_qcpg_dataset_directory: Path,
    test_methylation_h5_file: Path,
    grover_tokenizer: AutoTokenizer,
) -> None:
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
        test_files,
        encoding=data.BPE_ENCODING_STR,
        tokenizer=grover_tokenizer,
        allow_N=True,
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
        tokenizer=grover_tokenizer,
    )
    assert dataset.data.shape == (8, 8)


def test_h5_cpg_data_loader(test_qcpg_dataset_directory: Path) -> None:
    test_files: list[Path] = [
        test_qcpg_dataset_directory / f"chr{i}.h5" for i in ["1", "2"]
    ]
    dataset = data.QuantumTorchDataset(
        test_files, encoding=data.ONEHOT_ENCODING_STR
    )
    data_loader = DataLoader(dataset, batch_size=1)
    for test_samples in data_loader:
        test_sequences, test_labels = test_samples
        assert test_sequences.shape == (1, 10, 4)
        assert test_labels.shape == (1, 4)

    data_loader = DataLoader(dataset, batch_size=2)
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


def test_convert_token_dataset_to_bpe(
    test_inputs_h5_file: Path, grover_tokenizer: AutoTokenizer
) -> None:
    with h5py.File(test_inputs_h5_file) as dataset:
        samples: Tensor = data._convert_token_dataset_to_bpe(
            dataset, grover_tokenizer, data.INPUTS_STR
        )

    assert samples.shape == (10, 16)

    with (
        h5py.File(test_inputs_h5_file) as dataset,
        pytest.raises(
            ValueError,
            match="BPE token conversion is not supported for H5 structure invalid",
        ),
    ):
        _ = data._convert_token_dataset_to_bpe(
            dataset, grover_tokenizer, "invalid"
        )


def test_convert_onehot_dataset_to_bpe(
    test_methylation_h5_file: Path, grover_tokenizer: AutoTokenizer
) -> None:
    with h5py.File(test_methylation_h5_file) as dataset:
        samples: Tensor = data._convert_onehot_dataset_to_bpe(
            dataset, grover_tokenizer, data.METHYLATION_STR
        )

    assert samples.shape == (4, 8)

    with (
        h5py.File(test_methylation_h5_file) as dataset,
        pytest.raises(
            ValueError,
            match="BPE token conversion is not supported for H5 structure invalid",
        ),
    ):
        _ = data._convert_onehot_dataset_to_bpe(
            dataset, grover_tokenizer, "invalid"
        )


def _reference(sequence: str) -> np.ndarray:
    return np.frombuffer(sequence.encode(), dtype=np.uint8)


def _write_fasta(
    fasta_filepath: Path,
    sequences: dict[str, str],
    line_length: int = 60,
    newline: str = "\n",
) -> None:
    with fasta_filepath.open("w", newline="") as fd:
        for name, sequence in sequences.items():
            fd.write(
                f">{name} dna:chromosome chromosome:GRCm38:{name}:1:"
                f"{len(sequence)}:1 REF{newline}"
            )
            for start in range(0, len(sequence), line_length):
                fd.write(sequence[start : start + line_length] + newline)


def _counts(rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def test_find_methylation_filepaths(caplog: pytest.LogCaptureFixture) -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for filename in [
            "cellB.cov.txt",
            "cellA.CpG.txt.gz",
            "RSC27_4.cov.txt",
        ]:
            (temp_path / filename).touch()

        with caplog.at_level("WARNING", logger="qenetics.tools.data"):
            filepaths_by_experiment = data._find_methylation_filepaths(
                temp_path, ["RSC27_4", "Ca26"]
            )

        assert filepaths_by_experiment == {
            "cellA": temp_path / "cellA.CpG.txt.gz",
            "cellB": temp_path / "cellB.cov.txt",
        }
        assert "Excluded experiment Ca26 not found" in caplog.text

        (temp_path / "cellA.cov.txt").touch()
        with pytest.raises(ValueError, match="share the experiment name"):
            _ = data._find_methylation_filepaths(temp_path)


def test_read_methylation_counts() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cov_filepath = temp_path / "cellA.cov.txt"
        cov_filepath.write_text("chr1\t5\t5\t50\t1\t1\n2\t9\t9\t0\t0\t3\n")
        cpg_filepath = temp_path / "cellB.CpG.txt"
        cpg_filepath.write_text(
            "#Chr\tPos\tRef\tChain\tTotal\tMet\tUnMet\tMetRate\tRef_context"
            "\tType\n"
            "chr1\t6\tG\t-\t4\t4\t0\t1.0\tCGA\tCpG\n"
        )

        counts_by_chromosome = data.read_methylation_counts(
            [cov_filepath, cpg_filepath]
        )

    assert set(counts_by_chromosome) == {"1", "2"}
    assert counts_by_chromosome["1"].tolist() == [[5, 1, 1, 0], [6, 4, 0, 1]]
    assert counts_by_chromosome["2"].tolist() == [[9, 0, 3, 0]]


@pytest.mark.parametrize(
    ("position", "expected_index"),
    [
        (3, 2),  # C of the first CpG
        (4, 2),  # G of the first CpG, reverse strand call
        (1, -1),  # A
        (5, -1),  # T
        (7, 6),  # C of the CpG at the end of the reference
        (8, 6),  # G of the CpG at the end of the reference
        (0, -1),  # before the reference
        (9, -1),  # after the reference
    ],
)
def test_locate_cpg_sites(position: int, expected_index: int) -> None:
    reference = _reference("AACGTACG")
    assert data.locate_cpg_sites(
        reference, np.array([position], dtype=np.int64)
    ).tolist() == [expected_index]


def test_aggregate_cpg_counts() -> None:
    cpg_indices = np.array([10, 10, 10, 20, 30, 30], dtype=np.int64)
    counts = _counts(
        [
            (11, 1, 1, 0),  # forward strand of site 10, experiment 0
            (12, 1, 0, 0),  # reverse strand of site 10, experiment 0
            (11, 1, 1, 1),  # tie in experiment 1
            (21, 0, 2, 0),
            (31, 1, 0, 0),  # below minimum reads
            (31, 3, 1, 1),
        ]
    )

    sites, labels = data.aggregate_cpg_counts(
        cpg_indices, counts, experiment_quantity=2, minimum_reads=2
    )
    assert sites.tolist() == [10, 20, 30]
    np.testing.assert_array_equal(
        labels, np.array([[1.0, 0.0], [0.0, np.nan], [np.nan, 1.0]])
    )

    sites, labels = data.aggregate_cpg_counts(
        cpg_indices, counts, experiment_quantity=2, binarize=False
    )
    assert sites.tolist() == [10, 20, 30]
    np.testing.assert_allclose(
        labels,
        np.array([[2 / 3, 0.5], [0.0, np.nan], [1.0, 0.75]]),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("sequence_length", "expected_start"),
    [(1001, 500), (5, 8), (4, 9), (2, 10)],
)
def test_window_starts(sequence_length: int, expected_start: int) -> None:
    sites = np.array([10], dtype=np.int64)
    if sequence_length == 1001:
        sites = np.array([1000], dtype=np.int64)
    assert data.window_starts(sites, sequence_length).tolist() == [
        expected_start
    ]


def test_append_to_dataset() -> None:
    with (
        TemporaryDirectory() as temp_dir,
        h5py.File(Path(temp_dir) / "test.h5", "w") as fd,
    ):
        dataset = data._create_resizable_dataset(fd, "values", (2,), "i8")
        data._append_to_dataset(dataset, np.array([[1, 2]]))
        data._append_to_dataset(dataset, np.empty((0, 2)))
        data._append_to_dataset(dataset, np.array([[3, 4], [5, 6]]))
        assert dataset[()].tolist() == [[1, 2], [3, 4], [5, 6]]


@pytest.mark.parametrize("allow_N", [False, True])
def test_write_chromosome_h5(allow_N: bool) -> None:
    #                      0123456789012345
    reference = _reference("ATCGATNCGTACGTCG")
    sites = np.array([2, 7, 11, 14], dtype=np.int64)
    labels = np.array(
        [[1.0, np.nan], [0.0, 1.0], [np.nan, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    with TemporaryDirectory() as temp_dir:
        h5_filepath = Path(temp_dir) / "chr1.h5"
        samples_written = data._write_chromosome_h5(
            h5_filepath,
            reference,
            sites,
            labels,
            ["cellA", "cellB"],
            5,
            allow_N=allow_N,
        )
        with h5py.File(h5_filepath) as fd:
            sequences = [
                converters.one_hot_sequence_to_nucleotide_str(
                    sequence, allow_N=True
                )
                for sequence in fd[data.METHYLATION_SEQUENCES_KEY]
            ]
            positions = fd[data.POSITIONS_KEY][()].tolist()
            ratios_a = fd[data.METHYLATION_RATIOS_KEY]["cellA"][()]
            ratios_b = fd[data.METHYLATION_RATIOS_KEY]["cellB"][()]

    # Site 14 is dropped because its window exceeds the reference.
    if allow_N:
        assert samples_written == 3
        assert sequences == ["ATCGA", "TNCGT", "TACGT"]
        assert positions == [3, 8, 12]
        np.testing.assert_array_equal(ratios_a, [1.0, 0.0, np.nan])
        np.testing.assert_array_equal(ratios_b, [np.nan, 1.0, 0.0])
    else:
        assert samples_written == 2
        assert sequences == ["ATCGA", "TACGT"]
        assert positions == [3, 12]
        np.testing.assert_array_equal(ratios_a, [1.0, np.nan])
        np.testing.assert_array_equal(ratios_b, [np.nan, 0.0])


@pytest.mark.parametrize(
    ("newline", "sequence_length"),
    [("\n", 11), ("\r\n", 11), ("\n", 10), ("\n", 1001)],
)
def test_create_h5_dataset_from_methylation_profiles(
    newline: str, sequence_length: int, caplog: pytest.LogCaptureFixture
) -> None:
    rng = np.random.default_rng(0)
    chromosome_sequences: dict[str, str] = {
        name: "".join(rng.choice(list("ACGT"), size=length))
        for name, length in [("1", 5003), ("2", 2999)]
    }
    # Lowercase soft-masked nucleotides must be treated as uppercase.
    chromosome_sequences["2"] = (
        chromosome_sequences["2"][:1000].lower()
        + chromosome_sequences["2"][1000:]
    )
    upper_sequences = {
        name: sequence.upper()
        for name, sequence in chromosome_sequences.items()
    }
    cpg_positions: dict[str, list[int]] = {
        name: [
            index + 1
            for index in range(len(sequence) - 1)
            if sequence[index : index + 2] == "CG"
        ]
        for name, sequence in upper_sequences.items()
    }
    non_cpg_position: int = next(
        position
        for position in range(2, len(upper_sequences["1"]))
        if "CG"
        not in (
            upper_sequences["1"][position - 1 : position + 1],
            upper_sequences["1"][position - 2 : position],
        )
    )
    cell_b_positions: list[int] = cpg_positions["1"][::2]

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fasta_filepath = temp_path / "genome.fa"
        _write_fasta(fasta_filepath, chromosome_sequences, newline=newline)
        methylation_directory = temp_path / "methylation"
        methylation_directory.mkdir()
        output_directory = temp_path / "output"
        output_directory.mkdir()

        # cellA: Bismark coverage, forward strand, 1 methylated read.
        # cellB: Hou et al. CpG report, reverse strand at the G, 1 methylated
        # and 2 unmethylated reads.
        # Ca26: excluded.
        with (methylation_directory / "cellA.cov.txt").open("w") as fd:
            for chromosome, positions in cpg_positions.items():
                for position in positions:
                    fd.write(
                        f"chr{chromosome}\t{position}\t{position}\t100\t1\t0\n"
                    )
            fd.write(  # not a CpG in the reference
                f"chr1\t{non_cpg_position}\t{non_cpg_position}\t100\t1\t0\n"
            )
            fd.write("chrY\t10\t10\t100\t1\t0\n")  # not in the reference
        with (methylation_directory / "cellB.CpG.txt").open("w") as fd:
            fd.write("#Chr\tPos\tRef\tChain\tTotal\tMet\tUnMet\tMetRate\n")
            for position in cell_b_positions:
                fd.write(
                    f"chr1\t{position + 1}\tG\t-\t3\t1\t2\t0.33\tCGA\tCpG\n"
                )
        (methylation_directory / "Ca26.cov.txt").write_text(
            "1\t5\t5\t100\t1\t0\n"
        )

        with caplog.at_level("WARNING", logger="qenetics.tools.data"):
            data.create_h5_dataset_from_methylation_profiles(
                methylation_directory,
                fasta_filepath,
                output_directory,
                sequence_length,
                excluded_experiments=["Ca26"],
            )

        chromosome_1_calls: int = (
            len(cpg_positions["1"]) + 1 + len(cell_b_positions)
        )
        assert (
            f"1 of {chromosome_1_calls} methylation calls on chromosome 1 are "
            "not at a CpG site"
        ) in caplog.text
        assert "chromosome 2 are not at a CpG site" not in caplog.text
        assert "chromosome Y, which is not" in caplog.text
        assert sorted(path.name for path in output_directory.iterdir()) == [
            "chr1.h5",
            "chr2.h5",
        ]

        offset: int = (sequence_length - 1) // 2
        for chromosome, sequence in upper_sequences.items():
            expected_positions = [
                position
                for position in cpg_positions[chromosome]
                if position - 1 - offset >= 0
                and position - 1 - offset + sequence_length <= len(sequence)
            ]
            with h5py.File(output_directory / f"chr{chromosome}.h5") as fd:
                positions = fd[data.POSITIONS_KEY][()].tolist()
                windows = [
                    converters.one_hot_sequence_to_nucleotide_str(window)
                    for window in fd[data.METHYLATION_SEQUENCES_KEY]
                ]
                ratios = fd[data.METHYLATION_RATIOS_KEY]
                assert list(ratios) == ["cellA", "cellB"]
                ratios_a = ratios["cellA"][()]
                ratios_b = ratios["cellB"][()]

            assert positions == expected_positions
            for position, window in zip(positions, windows, strict=True):
                start = position - 1 - offset
                assert window == sequence[start : start + sequence_length]
                assert window[offset : offset + 2] == "CG"

            assert (ratios_a == 1.0).all()
            for position, ratio in zip(positions, ratios_b, strict=True):
                if chromosome == "1" and position in cell_b_positions:
                    assert ratio == 0.0
                else:
                    assert np.isnan(ratio)

        dataset = data.QuantumTorchDataset(
            [output_directory / "chr1.h5", output_directory / "chr2.h5"],
            encoding=data.ONEHOT_ENCODING_STR,
        )
        assert dataset.data.shape[1:] == (sequence_length, 4)
        assert dataset.labels.shape[1] == 2


def test_create_h5_dataset_from_methylation_profiles_minimum_reads() -> None:
    sequence: str = "ATATACGATATATACGATATA"
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fasta_filepath = temp_path / "genome.fa"
        _write_fasta(fasta_filepath, {"1": sequence}, line_length=8)
        methylation_directory = temp_path / "methylation"
        methylation_directory.mkdir()
        # Site at 6 has 2 + 2 reads across both strands, site at 15 has 3.
        (methylation_directory / "cell.cov.txt").write_text(
            "1\t6\t6\t100\t2\t0\n1\t7\t7\t0\t0\t2\n1\t15\t15\t100\t3\t0\n"
        )

        data.create_h5_dataset_from_methylation_profiles(
            methylation_directory,
            fasta_filepath,
            temp_path,
            5,
            minimum_samples=4,
            binarize=False,
        )
        with h5py.File(temp_path / "chr1.h5") as fd:
            assert fd[data.POSITIONS_KEY][()].tolist() == [6]
            assert fd[data.METHYLATION_RATIOS_KEY]["cell"][()].tolist() == [0.5]
            assert (
                converters.one_hot_sequence_to_nucleotide_str(
                    fd[data.METHYLATION_SEQUENCES_KEY][0]
                )
                == "TACGA"
            )


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
            # Training loads files by name, e.g. chr1.h5, so the name is kept.
            h5py.File(temp_path / test_filepath.name) as new_dataset,
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
