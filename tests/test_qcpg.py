from pathlib import Path
from math import ceil, log2
from tempfile import TemporaryDirectory
from unittest import mock

import pennylane as qml
import pytest
from torch import nn, optim, tensor

from qenetics.qcpg import qcpg, qcpg_models
from qenetics.tools import cpg_sampler

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


@pytest.mark.parametrize(
    ("sequence", "layer_quantity", "output_quantity"),
    [
        ("A", 1, 1),
        ("AT", 1, 1),
        ("ATC", 1, 1),
        ("C", 2, 1),
        ("G", 1, 2),
        ("ATCGATCG", 2, 2),
    ],
)
def test_train_one_epoch(
    sequence: str, layer_quantity: int, output_quantity: int
) -> None:
    batch_size: int = 2
    model = qcpg_models.QNN(len(sequence), layer_quantity, output_quantity)
    with (
        mock.patch("qenetics.tools.cpg_sampler.H5CpGDataset"),
        mock.patch(
            "qenetics.tools.cpg_sampler.H5CpGDataset.__getitem__"
        ) as mock_get,
    ):
        mock_get.return_value = [
            tensor(cpg_sampler.nucleotide_string_to_numpy(sequence))
        ] * batch_size
        _ = qcpg._train_one_epoch(
            model,
            1,
            cpg_sampler.H5CpGDataset([]),
            optim.SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            report_every=1,
        )


def test_train_qnn_circuit(test_dataset_directory: Path) -> None:
    with TemporaryDirectory() as temp_dir:
        training_parameters = qcpg.TrainingParameters(
            data_directory=test_dataset_directory,
            output_filepath=Path(temp_dir) / "output.dat",
            training_chromosomes=["1", "2"],
            validation_chromosomes=["1", "2"],
            batch_size=1,
            epochs=2,
        )
        qcpg.train_qnn_circuit(training_parameters)
        training_parameters.entangler = "strong"
        qcpg.train_qnn_circuit(training_parameters)
