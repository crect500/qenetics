from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from qenetics.qcpg import records
from qenetics.tools import metrics

START_TIME = datetime(2026, 9, 27, 13, 30, 5, tzinfo=UTC)


def _create_run(records_directory: Path, **kwargs) -> Path:
    return records.create_run_record(
        records_directory,
        kwargs.get("experiment_name", "serum mESC"),
        {"data_directory": Path("data"), "epochs": 2},
        {"training": ["1"], "validation": ["3"], "test": ["2"]},
        ["cellA", "cellB"],
        now=kwargs.get("now", START_TIME),
    )


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [
        ("cellA", "cellA"),
        ("serum mESC", "serum_mESC"),
        ("Ca26/HCC", "Ca26_HCC"),
        ("RSC27_4.1", "RSC27_4.1"),
        ("..", "_"),
        ("", "_"),
    ],
)
def test_safe_name(name: str, expected_name: str) -> None:
    assert records.safe_name(name) == expected_name


def test_create_run_record() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        run_directory = _create_run(records_directory)
        same_time_run_directory = _create_run(records_directory)

        assert run_directory == (
            records_directory / "serum_mESC" / "20260927-133005"
        )
        # A second run started in the same second gets a distinct id.
        assert same_time_run_directory.name == "20260927-133005-1"

        run = records.read_run(run_directory)
        assert run["run_id"] == "20260927-133005"
        assert run["experiment_name"] == "serum mESC"
        assert run["status"] == records.STATUS_RUNNING
        assert run["finished"] is None
        assert run["parameters"] == {"data_directory": "data", "epochs": 2}
        assert run["cell_names"] == ["cellA", "cellB"]
        assert run["epochs"] == []
        assert run["metrics"] == {}
        assert records.find_run_directories(records_directory) == [
            run_directory,
            same_time_run_directory,
        ]


def test_append_epoch_and_finish_run() -> None:
    with TemporaryDirectory() as temp_dir:
        run_directory = _create_run(Path(temp_dir))
        records.append_epoch(run_directory, 0, 0.7, 0.8, 0.55, [0.5, np.nan])
        records.append_epoch(run_directory, 1, 0.6, 0.75, 0.6, [0.6, 0.7])
        records.finish_run(
            run_directory,
            records.STATUS_COMPLETED,
            now=datetime(2026, 9, 27, 14, 0, 0, tzinfo=UTC),
        )
        run = records.read_run(run_directory)

    assert [epoch["epoch"] for epoch in run["epochs"]] == [0, 1]
    assert run["epochs"][1]["training_loss"] == 0.6
    assert np.isnan(run["epochs"][0]["cell_validation_aucs"][1])
    assert run["status"] == records.STATUS_COMPLETED
    assert run["finished"] == "2026-09-27T14:00:00+00:00"


def test_split_outputs_round_trip() -> None:
    outputs = records.SplitOutputs(
        chromosome_names=["2", "X"],
        chromosome_indices=np.array([0, 0, 1], dtype=np.int32),
        positions=np.array([10, 30, 5], dtype=np.int64),
        labels=np.array(
            [[1.0, np.nan], [0.0, 1.0], [np.nan, 0.0]], dtype=np.float32
        ),
        predictions=np.array(
            [[0.9, 0.2], [0.1, 0.8], [0.4, 0.3]], dtype=np.float32
        ),
    )
    with TemporaryDirectory() as temp_dir:
        run_directory = _create_run(Path(temp_dir))
        assert records.read_split_outputs(run_directory, "test") is None

        records.write_split_outputs(run_directory, "test", outputs)
        replaced = records.SplitOutputs(
            **{**vars(outputs), "positions": outputs.positions + 1}
        )
        records.write_split_outputs(run_directory, "test", replaced)
        records.write_split_outputs(run_directory, "training", outputs)

        test_outputs = records.read_split_outputs(run_directory, "test")
        training_outputs = records.read_split_outputs(run_directory, "training")

    assert test_outputs.chromosome_names == ["2", "X"]
    np.testing.assert_array_equal(test_outputs.positions, [11, 31, 6])
    np.testing.assert_array_equal(test_outputs.labels, outputs.labels)
    np.testing.assert_array_equal(
        training_outputs.predictions, outputs.predictions
    )
    np.testing.assert_array_equal(
        training_outputs.chromosome_indices, outputs.chromosome_indices
    )


def test_write_metrics() -> None:
    predictions = np.array([[0.9, 0.2], [0.1, 0.8], [0.7, 0.4]])
    labels = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, np.nan]])
    cell_results, pooled_result = metrics.experiment_metrics(
        predictions, labels
    )
    with TemporaryDirectory() as temp_dir:
        run_directory = _create_run(Path(temp_dir))
        records.write_metrics(
            run_directory, "training", 0.5, cell_results, pooled_result
        )
        records.write_metrics(
            run_directory, "test", 0.6, cell_results, pooled_result
        )
        stored = records.read_run(run_directory)["metrics"]

    assert set(stored) == {"training", "test"}
    assert stored["test"]["loss"] == 0.6
    assert stored["test"]["pooled"]["observed_sites"] == 5
    assert [cell["observed_sites"] for cell in stored["test"]["cells"]] == [
        3,
        2,
    ]
    assert stored["test"]["cells"][0]["confusion_matrix"]["true_positives"] == 2
