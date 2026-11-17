from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from qenetics.qcpg import qcpg, records
from qenetics.tools import data, tokenization

WINDOW_LENGTH: int = 41


def _write_dataset(directory: Path, chromosomes: list[str]) -> None:
    rng = np.random.default_rng(0)
    for chromosome in chromosomes:
        reference = np.frombuffer(
            "".join(rng.choice(list("ACGT"), 1500)).encode(), dtype=np.uint8
        ).copy()
        text: str = reference.tobytes().decode()
        sites = np.array(
            [site for site in range(30, 1460) if text[site : site + 2] == "CG"],
            dtype=np.int64,
        )[:40]
        labels = rng.integers(0, 2, (len(sites), 2)).astype(np.float32)
        labels[:2] = [[0.0, 1.0], [1.0, 0.0]]
        data._write_chromosome_h5(
            directory / f"chr{chromosome}.h5",
            reference,
            sites,
            labels,
            ["cellA", "cellB"],
            WINDOW_LENGTH,
        )


def _parameters(directory: Path, **overrides) -> qcpg.TrainingParameters:
    settings: dict = {
        "data_directory": directory / "data",
        "output_directory": directory / "output",
        "training_chromosomes": ["1"],
        "validation_chromosomes": ["2"],
        "test_chromosomes": ["3"],
        "model": "window_average",
        "model_options": {"window_length": 4, "quantum_layer_quantity": 1},
        "device_name": "default.qubit",
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 0.01,
        "update_report": False,
    }
    settings.update(overrides)
    return qcpg.TrainingParameters(**settings)


@pytest.fixture
def dataset_directory():
    with TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        (directory / "data").mkdir()
        _write_dataset(directory / "data", ["1", "2", "3"])
        yield directory


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "invalid"}, "Unknown model invalid"),
        ({"model": "qnn", "model_options": {"a": 1}}, "not the qnn model"),
        ({"model": "qnn", "evaluation_shots": [10]}, "not the qnn model"),
        ({"learning_rate_decay": 0.0}, r"decay must be in \(0, 1\]"),
        ({"quantum_learning_rate": -1.0}, "must be positive"),
        ({"selection_metric": "invalid"}, "Unknown selection metric"),
        ({"patience": 2}, "needs a selection metric"),
        (
            {"patience": 0, "selection_metric": "pooled_auc"},
            "Patience must be at least 1",
        ),
        ({"minimum_improvement": -0.1}, "must not be negative"),
        ({"training_sample_quantity": 0}, "at least 1"),
        ({"loader_worker_quantity": -1}, "must not be negative"),
        ({"evaluation_shots": [0]}, "at least 1"),
    ],
)
def test_check_training_parameters(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        qcpg.check_training_parameters(_parameters(Path("."), **overrides))


def _only_run(directory: Path) -> Path:
    run_directories = records.find_run_directories(
        directory / "output" / "records"
    )
    assert len(run_directories) == 1
    return run_directories[0]


def test_train_model_family(dataset_directory: Path) -> None:
    training_parameters = _parameters(
        dataset_directory,
        selection_metric=qcpg.MEAN_CELL_AUC,
        patience=3,
        learning_rate_decay=0.9,
        evaluation_shots=[20],
        seed=0,
    )
    qcpg.train_qnn_circuit(training_parameters)
    run_directory: Path = _only_run(dataset_directory)
    run = records.read_run(run_directory)
    assert run["status"] == records.STATUS_COMPLETED
    assert run["parameters"]["model"] == "window_average"
    metrics = records.read_json(run_directory / records.METRICS_FILENAME)
    assert set(metrics) == {
        records.TRAINING_SPLIT,
        records.TEST_SPLIT,
        f"{records.TEST_SPLIT}_shots_20",
    }
    assert (dataset_directory / "output" / "model.pt").exists()
    assert (dataset_directory / "output" / "qcpg_train.log").exists()


def test_early_stopping_and_best_epoch(dataset_directory: Path) -> None:
    training_parameters = _parameters(
        dataset_directory,
        epochs=6,
        selection_metric=qcpg.POOLED_AUC,
        patience=1,
        minimum_improvement=1.0,
    )
    qcpg.train_qnn_circuit(training_parameters)
    run_directory: Path = _only_run(dataset_directory)
    epochs = records.read_json(run_directory / records.EPOCHS_FILENAME)
    # No gain can exceed 1.0, so training stops after the second epoch.
    assert len(epochs) == 2
    best = max(epochs, key=lambda epoch: epoch["validation_auc"])
    model_state = torch.load(dataset_directory / "output" / "model.pt")
    final_state = torch.load(run_directory / records.MODEL_FILENAME)
    assert all(
        torch.equal(model_state[name], final_state[name])
        for name in model_state
    )
    assert best["epoch"] in (0, 1)


def test_seeded_runs_match(dataset_directory: Path) -> None:
    states: list[dict[str, torch.Tensor]] = []
    for name in ("first", "second"):
        training_parameters = _parameters(
            dataset_directory,
            output_directory=dataset_directory / name,
            seed=7,
        )
        qcpg.train_qnn_circuit(training_parameters)
        states.append(torch.load(dataset_directory / name / "model.pt"))
    assert all(
        torch.equal(states[0][name], states[1][name]) for name in states[0]
    )


def test_learning_rates(dataset_directory: Path) -> None:
    training_parameters = _parameters(
        dataset_directory,
        model="filter_bank_network",
        model_options={"window_lengths": [2], "tail": "pooling"},
        learning_rate_decay=0.5,
        quantum_learning_rate=0.1,
    )
    datasets = qcpg._load_datasets(training_parameters)
    training_loader, validation_loader, model, optimizer = (
        qcpg._prepare_training(training_parameters, datasets=datasets)
    )
    assert [group["lr"] for group in optimizer.param_groups] == [0.01, 0.1]
    assert optimizer.param_groups[1]["params"] == [
        model.front_end.banks[0].circuit_weights
    ]
    training_parameters.output_directory.mkdir()
    qcpg._train_all_epochs(
        model,
        training_loader,
        validation_loader,
        optimizer,
        training_parameters.output_directory,
        training_parameters,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [
        pytest.approx(0.0025),
        pytest.approx(0.025),
    ]


def test_training_subset(dataset_directory: Path) -> None:
    dataset = data.QuantumTorchDataset(
        [dataset_directory / "data" / "chr1.h5"],
        encoding=data.ONEHOT_ENCODING_STR,
    )
    subset = qcpg._training_subset(dataset, 10)
    assert len(subset) == 10
    assert subset.indices == sorted(subset.indices)
    assert qcpg._training_subset(dataset, 10).indices == subset.indices
    assert qcpg._training_subset(dataset, None) is dataset
    assert qcpg._training_subset(dataset, 1000) is dataset


def test_non_finite_loss(
    dataset_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _nan_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return outputs.sum() * float("nan")

    monkeypatch.setattr(qcpg, "_observed_loss", _nan_loss)
    with pytest.raises(FloatingPointError, match="epoch 0, batch 0"):
        qcpg.train_qnn_circuit(_parameters(dataset_directory))


def test_train_on_token_files(dataset_directory: Path) -> None:
    data.create_token_h5_dataset(
        dataset_directory / "data",
        dataset_directory / "kmer",
        tokenization.CPG_KMER,
        kmer_length=2,
        flank_token_quantity=6,
    )
    training_parameters = _parameters(
        dataset_directory,
        data_directory=dataset_directory / "kmer",
        encoding=data.TOKEN_ENCODING_STR,
        model="filter_bank_network",
        model_options={"window_lengths": [1], "tail": "pooling"},
    )
    qcpg.train_qnn_circuit(training_parameters)
    run = records.read_run(_only_run(dataset_directory))
    assert run["status"] == records.STATUS_COMPLETED

    model = qcpg.build_model(
        training_parameters,
        13,
        2,
        data.read_h5_layout(dataset_directory / "kmer" / "chr1.h5")[2],
    )
    # 16 k-mers, with no channel for "no token".
    assert model.channel_quantity == 16
    training_parameters.padding_token = 0
    with pytest.raises(ValueError, match="define their own padding"):
        qcpg.build_model(
            training_parameters,
            13,
            2,
            data.read_h5_layout(dataset_directory / "kmer" / "chr1.h5")[2],
        )


def test_build_model_nucleotide_tokens(dataset_directory: Path) -> None:
    training_parameters = _parameters(
        dataset_directory, encoding=data.TOKEN_ENCODING_STR
    )
    model = qcpg.build_model(training_parameters, WINDOW_LENGTH, 2, {})
    assert model.channel_quantity == 4


def test_feature_model_with_one_site_left_over(
    dataset_directory: Path,
) -> None:
    # 40 training sites in batches of 13 leave one site, on which BatchNorm
    # cannot train; that batch is dropped.
    parameters = _parameters(
        dataset_directory,
        model="feature_classifier",
        model_options={},
        batch_size=13,
        epochs=1,
    )
    qcpg.train_qnn_circuit(parameters)
    run = records.read_run(_only_run(dataset_directory))
    assert run["status"] == records.STATUS_COMPLETED


def test_epoch_loss_is_mean_of_batches() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 2))
    inputs = torch.rand(10, 2, 4)
    labels = (torch.rand(10, 2) < 0.5).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels), batch_size=4
    )

    def probabilities(batch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(model(batch))

    wrapped = torch.nn.Module()
    wrapped.forward = probabilities  # type: ignore[method-assign]
    wrapped.add_module("model", model)
    with torch.no_grad():
        expected = np.mean(
            [
                torch.nn.functional.binary_cross_entropy(
                    probabilities(batch), batch_labels
                ).item()
                for batch, batch_labels in loader
            ]
        )
    # A learning rate of 0 keeps the weights, so every batch's loss is known.
    loss = qcpg._train_one_epoch(
        wrapped,
        0,
        loader,
        torch.optim.SGD(model.parameters(), lr=0.0),
        qcpg.TrainingParameters(Path("."), Path(".")),
    )
    assert loss == pytest.approx(expected, rel=1e-6)


def test_shared_chromosomes_warn(
    dataset_directory: Path, caplog: pytest.LogCaptureFixture
) -> None:
    qcpg._resolve_chromosomes(_parameters(dataset_directory))
    assert "more than one of the training" not in caplog.text
    qcpg._resolve_chromosomes(
        _parameters(dataset_directory, training_chromosomes=["1", "2", "3"])
    )
    assert "Chromosomes 2, 3 are in more than one" in caplog.text
