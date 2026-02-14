from math import nan
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
import torch
from torch import optim, tensor

from qenetics.qcpg import qcpg, qcpg_models
from qenetics.tools import data

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


def test_get_free_port() -> None:
    qcpg._get_free_port()


def test_prepare_training(
    test_single_amplitude_dataset_directory: Path,
) -> None:
    with TemporaryDirectory() as temp_dir:
        training_parameters = qcpg.TrainingParameters(
            data_directory=test_single_amplitude_dataset_directory,
            output_filepath=Path(temp_dir) / "output.dat",
            training_chromosomes=["1", "2"],
            validation_chromosomes=["1", "2"],
            batch_size=2,
            epochs=2,
        )
    training_loader, validation_loader, model, optimizer = (
        qcpg._prepare_training(training_parameters)
    )


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
    test_h5_loader: data.H5CpGDataset,
) -> None:
    model = qcpg_models.QNN(len(sequences[0]), layer_quantity, output_quantity)
    training_parameters = qcpg.TrainingParameters(
        data_directory=Path("."),
        output_filepath=Path("."),
    )
    with (
        mock.patch("qenetics.tools.data.H5CpGDataset.__getitem__") as mock_get,
    ):
        mock_get.side_effect = [
            (
                tensor(
                    [
                        data.nucleotide_string_to_numpy(sequence)
                        for sequence in sequences
                    ],
                    dtype=torch.float,
                ),
                tensor(
                    [[0.0] * output_quantity for _ in sequences],
                    dtype=torch.float,
                ),
            )
        ]
        _ = qcpg._train_one_epoch(
            model,
            1,
            test_h5_loader,
            optim.SGD(model.parameters(), lr=0.01),
            training_parameters,
        )


def test_train_one_epoch_with_nans(test_h5_loader: data.H5CpGDataset) -> None:
    sequence_length: int = 2
    layer_quantity: int = 1
    output_quantity: int = 2
    model = qcpg_models.QNN(sequence_length, layer_quantity, output_quantity)
    training_parameters = qcpg.TrainingParameters(
        data_directory=Path("."),
        output_filepath=Path("."),
        l1_regularizer=0.1,
        l2_regularizer=0.1,
    )
    with (
        mock.patch("qenetics.tools.data.H5CpGDataset.__getitem__") as mock_get,
    ):
        mock_get.side_effect = [
            (
                [tensor([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float)],
                [tensor([1.0, nan], dtype=torch.float)],
            )
        ]
        _ = qcpg._train_one_epoch(
            model,
            1,
            test_h5_loader,
            optim.SGD(model.parameters(), lr=0.01),
            training_parameters,
        )


def test_evaluate_validation_set(
    test_h5_loader: data.H5CpGDataset,
) -> None:
    sequence_length: int = 2
    layer_quantity: int = 1
    output_quantity: int = 2
    model = qcpg_models.QNN(sequence_length, layer_quantity, output_quantity)
    training_parameters = qcpg.TrainingParameters(
        data_directory=Path("."),
        output_filepath=Path("."),
    )
    with (
        mock.patch("qenetics.tools.data.H5CpGDataset.__getitem__") as mock_get,
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
            output_filepath=Path(temp_dir) / "output.dat",
            training_chromosomes=["1", "2"],
            validation_chromosomes=["1", "2"],
            batch_size=2,
            epochs=2,
        )
        qcpg.train_qnn_circuit(training_parameters)
        training_parameters.entangler = "strong"
        qcpg.train_qnn_circuit(training_parameters)
