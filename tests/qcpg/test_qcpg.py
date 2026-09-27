from math import nan
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import h5py
import numpy as np
import pytest
import torch
from torch import optim, tensor
from transformers import AutoTokenizer

from qenetics.qcpg import qcpg, qcpg_models, records
from qenetics.tools import converters, data

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


def test_get_free_port() -> None:
    qcpg._get_free_port()


@pytest.mark.parametrize(
    ("encoding"), [data.ONEHOT_ENCODING_STR, data.TOKEN_ENCODING_STR]
)
def test_prepare_training(
    encoding: str,
    test_single_amplitude_dataset_directory: Path,
    grover_tokenizer: AutoTokenizer,
) -> None:
    if encoding == data.TOKEN_ENCODING_STR:
        embedding_qubit_quantity: int | None = 1
        vocabulary_size: int | None = 4
    else:
        embedding_qubit_quantity = None
        vocabulary_size = None

    with TemporaryDirectory() as temp_dir:
        training_parameters = qcpg.TrainingParameters(
            data_directory=test_single_amplitude_dataset_directory,
            output_directory=Path(temp_dir),
            training_chromosomes=["1", "2"],
            validation_chromosomes=["1", "2"],
            test_chromosomes=["X"],
            encoding=encoding,
            embedding_qubit_quantity=embedding_qubit_quantity,
            vocabulary_size=vocabulary_size,
            batch_size=2,
            epochs=2,
        )
    _, _, _, _ = qcpg._prepare_training(training_parameters)


@pytest.mark.parametrize(
    ("truth", "expected_indices"),
    [([0.0], [0]), ([0.0, nan], [0]), ([nan, 1.0], [1])],
)
def test_remove_nans(truth: list[float], expected_indices: list[float]) -> None:
    truth_tensor = tensor(truth)
    indices = qcpg._non_nan_indices(truth_tensor)
    assert indices == expected_indices


@pytest.mark.parametrize(
    ("sequences", "layer_quantity", "output_quantity"),
    [
        (["A"], 1, 1),
        (["AT"], 1, 1),
        (["ATC"], 1, 1),
        (["C"], 2, 1),
        (["G"], 1, 2),
        (["ATCGATCG"], 2, 2),
        (["A", "C"], 1, 1),
        (["ATCGATCG", "GCTAGCTA"], 2, 2),
    ],
)
def test_train_one_epoch(
    sequences: list[str],
    layer_quantity: int,
    output_quantity: int,
    test_h5_loader: data.QuantumTorchDataset,
) -> None:
    model = qcpg_models.QNN(len(sequences[0]), layer_quantity, output_quantity)
    training_parameters = qcpg.TrainingParameters(
        data_directory=Path("."),
        output_directory=Path("."),
    )
    with (
        mock.patch(
            "qenetics.tools.data.QuantumTorchDataset.__getitem__"
        ) as mock_get,
    ):
        if output_quantity > 1:
            labels: torch.Tensor = tensor(
                [[0.0] * output_quantity for _ in sequences],
                dtype=torch.float,
            )
        else:
            labels = tensor(
                [0.0 * output_quantity for _ in sequences],
                dtype=torch.float,
            )
        mock_get.side_effect = [
            (
                tensor(
                    np.array(
                        [
                            converters.nucleotide_string_to_numpy(sequence)
                            for sequence in sequences
                        ],
                        dtype=float,
                    ),
                    dtype=torch.float,
                ),
                labels,
            )
        ]
        _ = qcpg._train_one_epoch(
            model,
            1,
            test_h5_loader,
            optim.SGD(model.parameters(), lr=0.01),
            training_parameters,
        )


def test_train_one_epoch_with_nans(
    test_h5_loader: data.QuantumTorchDataset,
) -> None:
    sequence_length: int = 2
    layer_quantity: int = 1
    output_quantity: int = 2
    model = qcpg_models.QNN(sequence_length, layer_quantity, output_quantity)
    training_parameters = qcpg.TrainingParameters(
        data_directory=Path("."),
        output_directory=Path("."),
        l1_regularizer=0.1,
        l2_regularizer=0.1,
    )
    with (
        mock.patch(
            "qenetics.tools.data.QuantumTorchDataset.__getitem__"
        ) as mock_get,
    ):
        mock_get.side_effect = [
            (
                tensor([[[0, 0, 0, 1], [0, 0, 1, 0]]], dtype=torch.float),
                tensor([[1.0, nan]], dtype=torch.float),
            )
        ]
        _ = qcpg._train_one_epoch(
            model,
            1,
            test_h5_loader,
            optim.SGD(model.parameters(), lr=0.01),
            training_parameters,
        )

    assert all(parameter.isfinite().all() for parameter in model.parameters())


def test_prepare_training_mismatched_experiments() -> None:
    reference = np.frombuffer(b"ATCGATCGATCGAT", dtype=np.uint8)
    sites = np.array([2, 6, 10], dtype=np.int64)
    labels = np.zeros((3, 2), dtype=np.float32)
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for chromosome, names in [
            ("1", ["cellA", "cellB"]),
            ("2", ["cellA", "cellC"]),
        ]:
            data._write_chromosome_h5(
                temp_path / f"chr{chromosome}.h5",
                reference,
                sites,
                labels,
                names,
                4,
            )

        training_parameters = qcpg.TrainingParameters(
            data_directory=temp_path,
            output_directory=temp_path,
            training_chromosomes=["1"],
            validation_chromosomes=["2"],
            test_chromosomes=[],
        )
        with pytest.raises(
            ValueError,
            match="validation chromosomes do not match those of the training",
        ):
            _ = qcpg._prepare_training(training_parameters)


def test_observed_loss() -> None:
    outputs = tensor([[0.8, 0.3], [0.6, 0.1]], dtype=torch.float)
    labels = tensor([[1.0, nan], [nan, 0.0]], dtype=torch.float)
    assert qcpg._observed_loss(outputs, labels).item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy(
            tensor([0.8, 0.1]), tensor([1.0, 0.0])
        ).item()
    )

    single_outputs = tensor([[0.8], [0.3]], dtype=torch.float)
    single_labels = tensor([nan, 0.0], dtype=torch.float)
    assert qcpg._observed_loss(
        single_outputs, single_labels
    ).item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy(
            tensor([0.3]), tensor([0.0])
        ).item()
    )

    assert qcpg._observed_loss(outputs, torch.full((2, 2), nan)) is None


def test_evaluate_validation_set_with_nans(
    test_h5_loader: data.QuantumTorchDataset,
) -> None:
    model = qcpg_models.QNN(2, 1, 2)
    training_parameters = qcpg.TrainingParameters(
        data_directory=Path("."),
        output_directory=Path("."),
    )
    with mock.patch(
        "qenetics.tools.data.QuantumTorchDataset.__getitem__"
    ) as mock_get:
        mock_get.side_effect = [
            (
                tensor([[[0, 0, 1, 0], [1, 0, 0, 0]]], dtype=torch.float),
                tensor([[0.0, nan]], dtype=torch.float),
            ),
            (
                tensor([[[0, 0, 0, 1], [0, 0, 1, 0]]], dtype=torch.float),
                tensor([[nan, nan]], dtype=torch.float),
            ),
            (
                tensor([[[0, 1, 0, 0], [0, 0, 1, 0]]], dtype=torch.float),
                tensor([[nan, 1.0]], dtype=torch.float),
            ),
        ]
        loss, auc = qcpg._evaluate_validation_set(
            model, test_h5_loader, training_parameters
        )

    assert np.isfinite(float(loss))
    assert np.isfinite(auc)


def test_evaluate_validation_set(
    test_h5_loader: data.QuantumTorchDataset,
) -> None:
    sequence_length: int = 2
    layer_quantity: int = 1
    output_quantity: int = 2
    model = qcpg_models.QNN(sequence_length, layer_quantity, output_quantity)
    training_parameters = qcpg.TrainingParameters(
        data_directory=Path("."),
        output_directory=Path("."),
    )
    with (
        mock.patch(
            "qenetics.tools.data.QuantumTorchDataset.__getitem__"
        ) as mock_get,
    ):
        mock_get.side_effect = [
            (
                tensor([[[0, 0, 1, 0], [1, 0, 0, 0]]], dtype=torch.float),
                tensor([[0.0, 1.0]], dtype=torch.float),
            ),
            (
                tensor([[[0, 0, 0, 1], [0, 0, 1, 0]]], dtype=torch.float),
                tensor([[1.0, 1.0]], dtype=torch.float),
            ),
        ]
        qcpg._evaluate_validation_set(
            model, test_h5_loader, training_parameters
        )


def test_train_qnn_circuit(
    test_single_amplitude_dataset_directory: Path,
) -> None:
    with TemporaryDirectory() as temp_dir:
        training_parameters = qcpg.TrainingParameters(
            data_directory=test_single_amplitude_dataset_directory,
            output_directory=Path(temp_dir),
            log_directory=Path(temp_dir),
            training_chromosomes=["1", "2"],
            validation_chromosomes=["1", "2"],
            test_chromosomes=["X"],
            batch_size=2,
            epochs=2,
        )
        qcpg.train_qnn_circuit(training_parameters)
        training_parameters.entangler = "strong"
        qcpg.train_qnn_circuit(training_parameters)


CELL_NAMES: list[str] = ["cellA", "cellB", "cellC"]


def _write_multi_cell_dataset(
    data_directory: Path, chromosomes: list[str], sample_quantity: int = 8
) -> None:
    rng = np.random.default_rng(0)
    reference = np.frombuffer(
        ("ATCG" * sample_quantity + "AT").encode(), dtype=np.uint8
    )
    sites = np.arange(2, 4 * sample_quantity, 4, dtype=np.int64)
    for chromosome in chromosomes:
        labels = rng.integers(0, 2, size=(sample_quantity, 3)).astype(
            np.float32
        )
        labels[rng.random(labels.shape) < 0.3] = np.nan
        # Every cell observes both classes.
        labels[:2] = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
        data._write_chromosome_h5(
            data_directory / f"chr{chromosome}.h5",
            reference,
            sites,
            labels,
            CELL_NAMES,
            4,
        )


def test_resolve_chromosomes() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for chromosome in ["1", "2", "3", "10", "X"]:
            (temp_path / f"chr{chromosome}.h5").touch()

        training_parameters = qcpg.TrainingParameters(
            data_directory=temp_path,
            output_directory=temp_path,
            training_chromosomes=["1"],
            test_chromosomes=["2"],
        )
        # Validation defaults to the remaining chromosomes, in numeric order.
        assert qcpg._resolve_chromosomes(training_parameters) == {
            records.TRAINING_SPLIT: ["1"],
            records.VALIDATION_SPLIT: ["3", "10", "X"],
            records.TEST_SPLIT: ["2"],
        }

        training_parameters.test_chromosomes = ["4"]
        with pytest.raises(FileNotFoundError, match=r"chr4\.h5"):
            _ = qcpg._resolve_chromosomes(training_parameters)

        training_parameters.test_chromosomes = ["2", "3", "10", "X"]
        with pytest.raises(ValueError, match="No validation chromosomes"):
            _ = qcpg._resolve_chromosomes(training_parameters)


def test_train_qnn_circuit_records() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_directory = temp_path / "serum"
        data_directory.mkdir()
        _write_multi_cell_dataset(data_directory, ["1", "2", "3"])
        training_parameters = qcpg.TrainingParameters(
            data_directory=data_directory,
            output_directory=temp_path,
            log_directory=temp_path,
            training_chromosomes=["1"],
            test_chromosomes=["2"],
            batch_size=4,
            epochs=2,
        )
        qcpg.train_qnn_circuit(training_parameters)

        run_directories = records.find_run_directories(temp_path / "records")
        assert len(run_directories) == 1
        run_directory = run_directories[0]
        # The genomic experiment defaults to the data directory's name.
        assert run_directory.parent.name == "serum"
        run = records.read_run(run_directory)
        assert run["status"] == records.STATUS_COMPLETED
        assert run["cell_names"] == CELL_NAMES
        assert run["chromosomes"] == {
            records.TRAINING_SPLIT: ["1"],
            records.VALIDATION_SPLIT: ["3"],
            records.TEST_SPLIT: ["2"],
        }
        assert len(run["epochs"]) == 2
        assert len(run["epochs"][0]["cell_validation_aucs"]) == 3
        assert set(run["metrics"]) == {
            records.TRAINING_SPLIT,
            records.TEST_SPLIT,
        }
        assert len(run["metrics"][records.TEST_SPLIT]["cells"]) == 3

        test_outputs = records.read_split_outputs(
            run_directory, records.TEST_SPLIT
        )
        with h5py.File(data_directory / "chr2.h5") as fd:
            expected_positions = fd[data.POSITIONS_KEY][()]
            expected_labels = np.stack(
                [
                    fd[data.METHYLATION_RATIOS_KEY][name][()]
                    for name in CELL_NAMES
                ],
                axis=1,
            )
        assert test_outputs.chromosome_names == ["2"]
        np.testing.assert_array_equal(
            test_outputs.positions, expected_positions
        )
        np.testing.assert_array_equal(test_outputs.labels, expected_labels)
        assert test_outputs.predictions.shape == (8, 3)
        assert (run_directory / records.MODEL_FILENAME).exists()

        # The run was added to the report.
        assert (temp_path / "records" / "index.html").exists()
        assert (run_directory / "cells" / "cellA" / "index.html").exists()


def test_train_qnn_circuit_records_failure() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _write_multi_cell_dataset(temp_path, ["1", "2", "3"])
        training_parameters = qcpg.TrainingParameters(
            data_directory=temp_path,
            output_directory=temp_path,
            log_directory=temp_path,
            training_chromosomes=["1"],
            test_chromosomes=["2"],
            experiment_name="failing",
            epochs=1,
        )
        with (
            mock.patch(
                "qenetics.qcpg.qcpg._train_all_epochs",
                side_effect=RuntimeError("training failed"),
            ),
            pytest.raises(RuntimeError, match="training failed"),
        ):
            qcpg.train_qnn_circuit(training_parameters)

        (run_directory,) = records.find_run_directories(temp_path / "records")
        assert run_directory.parent.name == "failing"
        assert (
            records.read_run(run_directory)["status"] == records.STATUS_FAILED
        )
        assert "status-failed" in (
            run_directory.parent / "index.html"
        ).read_text(encoding="utf-8")
