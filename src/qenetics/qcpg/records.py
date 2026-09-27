"""
Per-run records of training and test outputs.

Each training run gets its own folder under
`<records_directory>/<experiment_name>/<run_id>/` holding:

- `run.json`: parameters, chromosome splits, cell names, times and status.
- `epochs.json`: per-epoch losses and pooled and per-cell validation AUC.
- `metrics.json`: final per-cell and pooled metrics of each split.
- `outputs.h5`: per split, each site's chromosome, position, labels and
  predictions, with one column per cell.
- `model.pt`: the final model weights.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from qenetics.tools.metrics import CellMetrics

RUN_FILENAME: str = "run.json"
EPOCHS_FILENAME: str = "epochs.json"
METRICS_FILENAME: str = "metrics.json"
OUTPUTS_FILENAME: str = "outputs.h5"
MODEL_FILENAME: str = "model.pt"

STATUS_RUNNING: str = "running"
STATUS_COMPLETED: str = "completed"
STATUS_FAILED: str = "failed"

TRAINING_SPLIT: str = "training"
VALIDATION_SPLIT: str = "validation"
TEST_SPLIT: str = "test"

RUN_ID_FORMAT: str = "%Y%m%d-%H%M%S"


@dataclass
class SplitOutputs:
    """
    The per-site outputs of one split.

    Attributes
    ----------
    chromosome_names: The chromosome names that `chromosome_indices` refer to.
    chromosome_indices: The index of each site's chromosome.
    positions: The 1-based position of each site's C, -1 if unknown.
    labels: The labels, with one column per cell and NaN if unobserved.
    predictions: The predictions, with one column per cell.
    """

    chromosome_names: list[str]
    chromosome_indices: NDArray[np.int32]
    positions: NDArray[np.int64]
    labels: NDArray[np.float32]
    predictions: NDArray[np.float32]


def safe_name(name: str) -> str:
    """
    Make a name safe to use as a folder or file name.

    Args
    ----
    name: The name.

    Returns
    -------
    The name with characters other than letters, digits, '.', '_' and '-'
    replaced by '_'.
    """
    sanitized: str = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(".")
    return sanitized or "_"


def _to_jsonable(value: Any) -> Any:
    """
    Convert a value to one that JSON can store.

    Args
    ----
    value: The value.

    Returns
    -------
    The value, with paths as strings, sequences as lists, and any other
    unsupported value as its string representation.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(filepath: Path, content: Any) -> None:
    """
    Write JSON atomically, so readers never see a partially written file.

    Args
    ----
    filepath: The file to write.
    content: The content to write.
    """
    temporary_filepath: Path = filepath.with_name(
        f".{filepath.name}.{os.getpid()}.tmp"
    )
    temporary_filepath.write_text(json.dumps(_to_jsonable(content), indent=2))
    os.replace(temporary_filepath, filepath)


def read_json(filepath: Path) -> Any:
    """
    Read a JSON file.

    Args
    ----
    filepath: The file to read.

    Returns
    -------
    The file's content.
    """
    return json.loads(filepath.read_text())


def create_run_record(
    records_directory: Path,
    experiment_name: str,
    parameters: dict[str, Any],
    chromosomes: dict[str, list[str]],
    cell_names: list[str],
    *,
    now: datetime | None = None,
) -> Path:
    """
    Create the folder and initial files of a new training run's record.

    Args
    ----
    records_directory: The root folder of all records.
    experiment_name: The genomic experiment (dataset) the run trains on.
    parameters: The training parameters.
    chromosomes: The chromosomes of each split.
    cell_names: The name of each cell, in label column order.
    now: The start time of the run. Defaults to the current time.

    Returns
    -------
    The run's folder.
    """
    started: datetime = now or datetime.now().astimezone()
    experiment_directory: Path = records_directory / safe_name(experiment_name)
    experiment_directory.mkdir(parents=True, exist_ok=True)

    base_run_id: str = started.strftime(RUN_ID_FORMAT)
    run_id: str = base_run_id
    suffix: int = 1
    while True:
        run_directory: Path = experiment_directory / run_id
        try:
            run_directory.mkdir()
            break
        except FileExistsError:
            run_id = f"{base_run_id}-{suffix}"
            suffix += 1

    write_json(
        run_directory / RUN_FILENAME,
        {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "status": STATUS_RUNNING,
            "started": started.isoformat(timespec="seconds"),
            "finished": None,
            "parameters": parameters,
            "chromosomes": chromosomes,
            "cell_names": cell_names,
        },
    )
    write_json(run_directory / EPOCHS_FILENAME, [])
    return run_directory


def append_epoch(
    run_directory: Path,
    epoch: int,
    training_loss: float,
    validation_loss: float,
    validation_auc: float,
    cell_validation_aucs: list[float],
) -> None:
    """
    Record the results of one training epoch.

    Args
    ----
    run_directory: The run's folder.
    epoch: The epoch index.
    training_loss: The training loss.
    validation_loss: The validation loss.
    validation_auc: The validation AUC of all cells pooled.
    cell_validation_aucs: The validation AUC of each cell.
    """
    epochs: list[dict[str, Any]] = read_json(run_directory / EPOCHS_FILENAME)
    epochs.append(
        {
            "epoch": epoch,
            "training_loss": float(training_loss),
            "validation_loss": float(validation_loss),
            "validation_auc": float(validation_auc),
            "cell_validation_aucs": [
                float(auc) for auc in cell_validation_aucs
            ],
        }
    )
    write_json(run_directory / EPOCHS_FILENAME, epochs)


def write_split_outputs(
    run_directory: Path, split: str, outputs: SplitOutputs
) -> None:
    """
    Store the per-site outputs of one split, replacing any stored before.

    Args
    ----
    run_directory: The run's folder.
    split: The split, e.g. `TRAINING_SPLIT` or `TEST_SPLIT`.
    outputs: The per-site outputs.
    """
    with h5py.File(run_directory / OUTPUTS_FILENAME, "a") as fd:
        if split in fd:
            del fd[split]
        group: h5py.Group = fd.create_group(split)
        group.create_dataset(
            "chromosome_names",
            data=np.array(outputs.chromosome_names, dtype=h5py.string_dtype()),
        )
        for name, values, dtype in [
            ("chromosome_indices", outputs.chromosome_indices, "i4"),
            ("positions", outputs.positions, "i8"),
            ("labels", outputs.labels, "f4"),
            ("predictions", outputs.predictions, "f4"),
        ]:
            group.create_dataset(
                name, data=values, dtype=dtype, compression="gzip"
            )


def read_split_outputs(run_directory: Path, split: str) -> SplitOutputs | None:
    """
    Read the per-site outputs of one split.

    Args
    ----
    run_directory: The run's folder.
    split: The split.

    Returns
    -------
    The per-site outputs, or None if the split was not recorded.
    """
    outputs_filepath: Path = run_directory / OUTPUTS_FILENAME
    if not outputs_filepath.exists():
        return None

    with h5py.File(outputs_filepath) as fd:
        if split not in fd:
            return None
        group: h5py.Group = fd[split]
        return SplitOutputs(
            chromosome_names=[
                name.decode() if isinstance(name, bytes) else str(name)
                for name in group["chromosome_names"][()]
            ],
            chromosome_indices=group["chromosome_indices"][()],
            positions=group["positions"][()],
            labels=group["labels"][()],
            predictions=group["predictions"][()],
        )


def write_metrics(
    run_directory: Path,
    split: str,
    loss: float,
    cell_metrics: list[CellMetrics],
    pooled_metrics: CellMetrics,
) -> None:
    """
    Store the final metrics of one split.

    Args
    ----
    run_directory: The run's folder.
    split: The split.
    loss: The split's loss.
    cell_metrics: The metrics of each cell, in label column order.
    pooled_metrics: The metrics of all cells pooled.
    """
    metrics_filepath: Path = run_directory / METRICS_FILENAME
    all_metrics: dict[str, Any] = (
        read_json(metrics_filepath) if metrics_filepath.exists() else {}
    )
    all_metrics[split] = {
        "loss": float(loss),
        "pooled": asdict(pooled_metrics),
        "cells": [asdict(metrics) for metrics in cell_metrics],
    }
    write_json(metrics_filepath, all_metrics)


def finish_run(
    run_directory: Path, status: str, *, now: datetime | None = None
) -> None:
    """
    Mark a run as finished.

    Args
    ----
    run_directory: The run's folder.
    status: `STATUS_COMPLETED` or `STATUS_FAILED`.
    now: The finish time. Defaults to the current time.
    """
    run: dict[str, Any] = read_json(run_directory / RUN_FILENAME)
    run["status"] = status
    run["finished"] = (now or datetime.now().astimezone()).isoformat(
        timespec="seconds"
    )
    write_json(run_directory / RUN_FILENAME, run)


def read_run(run_directory: Path) -> dict[str, Any]:
    """
    Read a run's summary files.

    Args
    ----
    run_directory: The run's folder.

    Returns
    -------
    The run's `run.json` content, with its epochs under "epochs" and its final
    metrics under "metrics" (empty if not yet recorded).
    """
    run: dict[str, Any] = read_json(run_directory / RUN_FILENAME)
    epochs_filepath: Path = run_directory / EPOCHS_FILENAME
    metrics_filepath: Path = run_directory / METRICS_FILENAME
    run["epochs"] = (
        read_json(epochs_filepath) if epochs_filepath.exists() else []
    )
    run["metrics"] = (
        read_json(metrics_filepath) if metrics_filepath.exists() else {}
    )
    return run


def find_run_directories(records_directory: Path) -> list[Path]:
    """
    Find the folders of every recorded run.

    Args
    ----
    records_directory: The root folder of all records.

    Returns
    -------
    The run folders, sorted by experiment and run id.
    """
    return sorted(
        filepath.parent
        for filepath in records_directory.glob(f"*/*/{RUN_FILENAME}")
    )
