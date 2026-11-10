import importlib.util
import json
import math
import shlex
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from qenetics.qcpg import model_factory, records
from qenetics.tools import data

SCRIPT: Path = Path("scripts/cluster/plan_runs.py")
PRESETS_FILEPATH: Path = Path("config/quantum_models.json")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    specification = importlib.util.spec_from_file_location("plan_runs", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def presets() -> dict[str, model_factory.Preset]:
    return model_factory.load_presets(PRESETS_FILEPATH)


def _write_chromosomes(directory: Path, chromosomes: list[str]) -> None:
    """Write small chromosome files of 41 bp windows, chr<n> with n + 1 sites."""
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for chromosome in chromosomes:
        reference = np.frombuffer(
            "".join(rng.choice(list("ACGT"), 600)).encode(), dtype=np.uint8
        ).copy()
        sites = np.arange(30, 560, 40, dtype=np.int64)
        reference[sites] = ord("C")
        reference[sites + 1] = ord("G")
        quantity: int = int(chromosome) + 1 if chromosome.isdigit() else 1
        data._write_chromosome_h5(
            directory / f"chr{chromosome}.h5",
            reference,
            sites[:quantity],
            np.zeros((quantity, 1), dtype=np.float32),
            ["cell"],
            41,
        )


@pytest.fixture
def data_root():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _write_chromosomes(root / "2i", ["1", "2", "3", "5", "13"])
        _write_chromosomes(
            root / "hepg2", ["1", "2", "3", "4", "5", "6", "13", "X"]
        )
        yield root


@pytest.mark.parametrize(
    "text, hours",
    [
        ("2-12:00:00", 60.0),
        ("1-02:30", 26.5),
        ("3-6", 78.0),
        ("12:30:00", 12.5),
        ("30:00", 0.5),
        ("90", 1.5),
        ("infinite", math.inf),
        ("UNLIMITED", math.inf),
    ],
)
def test_slurm_hours(script: ModuleType, text: str, hours: float) -> None:
    assert script.slurm_hours(text) == hours


@pytest.mark.parametrize("text", ["INVALID", "1:2:3:4", "a-01:00"])
def test_slurm_hours_errors(script: ModuleType, text: str) -> None:
    with pytest.raises(ValueError, match="not a SLURM time"):
        script.slurm_hours(text)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("cpg_bpe64", ("cpg_bpe", 64)),
        ("cpg_kmer3", ("cpg_kmer", 3)),
        ("painted64", ("painted", 64)),
        ("painted_bases64", ("painted_bases", 64)),
    ],
)
def test_parse_input(
    script: ModuleType, name: str, expected: tuple[str, int]
) -> None:
    assert script.parse_input(name) == expected


def test_parse_input_error(script: ModuleType) -> None:
    with pytest.raises(ValueError, match="bases is not"):
        script.parse_input("bases")


def test_choose_presets(
    script: ModuleType, presets: dict[str, model_factory.Preset]
) -> None:
    chosen, skipped = script.choose_presets(
        presets, ["*"], ["deepcpg_tail/Q_L320", "whole_model/H1b_cnnq_ry"]
    )
    assert skipped["deepcpg_tail/Q_L320"] == "excluded"
    assert skipped["whole_model/H1b_cnnq_ry"] == "excluded"
    assert skipped["series_1/QA1"] == "the same model as deepcpg_tail/Q_L4"
    assert skipped["series_1/LD5k_Q1"] == "the same model as series_1/Q1"
    assert (
        skipped["classical_tail/CNN_deepcpg"]
        == "the same model as classical_tail/CNN"
    )
    assert "classical_tail/CNN" in chosen and "ceiling/U64" in chosen
    assert "classical_whole/FacM_r4" in chosen
    assert len(chosen) + len(skipped) == len(presets)
    assert "series_1/Q1" in chosen
    assert chosen == [name for name in presets if name in chosen]

    chosen, skipped = script.choose_presets(
        presets, ["series_1/QA1"], [], keep_duplicates=True
    )
    assert chosen == ["series_1/QA1"] and not skipped
    with pytest.raises(ValueError, match="No presets match nothing/"):
        script.choose_presets(presets, ["nothing/*"], [])


def test_run_patience(
    script: ModuleType, presets: dict[str, model_factory.Preset]
) -> None:
    # Patience 2, and none.
    assert script.run_patience(presets["deepcpg_tail/Q_L4"], 3) == 3
    assert script.run_patience(presets["series_1/Q1"], 3) == 3
    assert script.run_patience(presets["deepcpg_tail/Q_L4"], 1) == 2


def test_splits(script: ModuleType) -> None:
    human: list[str] = [str(chromosome) for chromosome in range(1, 23)]
    mouse: list[str] = [str(chromosome) for chromosome in range(1, 20)]
    assert script.split_a("hcc", human) == [
        (
            "",
            {
                "training": ["1", "3", "5", "7", "9"],
                "validation": ["13"],
                "test": ["2", "4"],
            },
        )
    ]
    assert script.split_a("ser", mouse)[0][1]["test"] == ["2"]
    assert script.full_training_sets("hepg2", human) == [
        (
            "",
            {
                "training": [c for c in human if c not in {"2", "4", "13"}],
                "validation": ["13"],
                "test": ["2", "4"],
            },
        )
    ]
    assert script.full_training_sets("2i", mouse) == script.split_a("2i", mouse)
    assert script.widened_training_sets("hepg2", human) == []
    assert script.widened_training_sets("2i", mouse) == [
        (
            "",
            {
                "training": [c for c in mouse if c not in {"2", "13"}],
                "validation": ["13"],
                "test": ["2"],
            },
        )
    ]
    folds = script.chromosome_folds("mesc", mouse)
    assert [label for label, _ in folds] == ["fold0", "fold1", "fold2", "fold3"]
    pool: list[str] = [c for c in mouse if c != "13"]
    assert folds[0][1] == {
        "training": [c for c in pool if c not in {"1", "5", "9", "14", "18"}],
        "validation": ["13"],
        "test": ["1", "5", "9", "14", "18"],
    }
    assert script.split_c("hepg2", human) == [
        (
            "",
            {
                "training": ["1", "3", "5", "7", "9", "11"],
                "validation": [str(c) for c in range(13, 23)],
                "test": ["2", "4", "6", "8", "10", "12"],
            },
        )
    ]
    assert script.split_c("2i", mouse)[0][1]["validation"] == [
        str(c) for c in range(13, 20)
    ]
    assert script.sweep_training_sets(
        "hcc", human
    ) == script.full_training_sets("hcc", human)
    assert script.sweep_training_sets("hepg2", human) == []
    assert (
        sorted(
            (c for _, chromosomes in folds for c in chromosomes["test"]),
            key=int,
        )
        == pool
    )


def test_load_dataset(script: ModuleType, data_root: Path) -> None:
    dataset = script.load_dataset(data_root / "hepg2")
    assert dataset.key == "hepg2"
    assert dataset.autosomes == ["1", "2", "3", "4", "5", "6", "13"]
    assert dataset.sites(["1", "3"]) == 2 + 4
    with pytest.raises(FileNotFoundError, match="chr9.h5"):
        script.check_chromosome_files(
            dataset, script.split_a("hcc", dataset.autosomes)[0][1]
        )
    with pytest.raises(ValueError, match="No training chromosome files"):
        script.check_chromosome_files(
            dataset, {"training": [], "validation": ["13"], "test": ["2"]}
        )
    (data_root / "other").mkdir()
    with pytest.raises(ValueError, match="no splits of other"):
        script.load_dataset(data_root / "other")


def test_plan_entries(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_root: Path,
) -> None:
    dataset = script.load_dataset(data_root / "2i")
    output = data_root / "out"
    entries: list[dict[str, Any]] = script.plan_entries(
        {"full": [presets["whole_model/WP_L4"], presets["tokens/P64_QE_L4"]]},
        [dataset],
        [0, 1],
        output,
        {"whole_model/WP_L4": 100.0, "tokens/P64_QE_L4": 1000.0},
        max_epochs=25,
        minimum_patience=3,
        presets_filepath=Path("presets.json"),
    )
    assert [entry["name"] for entry in entries] == [
        "tokens/2i/cpg_bpe64",
        "tokens/2i/painted64",
        "full/2i/tokens/P64_QE_L4/seed0",
        "full/2i/tokens/P64_QE_L4/seed1",
        "full/2i/whole_model/WP_L4/seed0",
        "full/2i/whole_model/WP_L4/seed1",
    ]
    bpe, painted, token_run, _, run, _ = entries
    bpe_directory = output / "tokens" / "2i" / "cpg_bpe64"
    assert bpe["arguments"][-5:] == [
        "64",
        "--training-chromosomes",
        "1",
        "3",
        "5",
    ]
    assert bpe["resources"] == "cpu"
    assert painted["arguments"][-2:] == [
        "--tokenizer",
        str(bpe_directory / "tokenizer.json"),
    ]
    assert painted["requires"] == [str(bpe_directory / "COMPLETED")]
    assert token_run["requires"] == [painted["marker"]]
    arguments: list[str] = token_run["arguments"]
    assert arguments[arguments.index("--data-directory") + 1] == str(
        output / "tokens" / "2i" / "painted64"
    )
    assert arguments[arguments.index("--experiment-name") + 1] == "2i"
    assert arguments[arguments.index("--records-directory") + 1] == str(
        output / "full" / "records"
    )
    assert run["requires"] == []
    assert run["marker"] == str(
        output / "full" / "2i" / "whole_model__WP_L4" / "seed0" / "COMPLETED"
    )
    assert run["arguments"][-17:] == [
        "--training-chromosomes",
        "1",
        "3",
        "5",
        "--validation-chromosomes",
        "13",
        "--test-chromosomes",
        "2",
        "--presets-file",
        "presets.json",
        "--all-training-samples",
        "--epochs",
        "25",
        "--patience",
        "3",
        "--selection-metric",
        "mean_cell_auc",
    ]
    # 100 microseconds x 12 sites x 26 epochs' worth of sites.
    assert run["hours"] == pytest.approx(100e-6 * 12 * 26 / 3600)


def test_plan_experiments(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_root: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    output = data_root / "out"
    for seed, auc in ((0, 0.7), (1, 0.5001)):
        record = records.create_run_record(
            output / "E4" / "records",
            "2i",
            {
                "output_directory": str(
                    output
                    / "E4"
                    / "2i"
                    / "deepcpg_tail__R_w32_L384_avg"
                    / f"seed{seed}"
                )
            },
            {},
            ["cell"],
        )
        records.write_json(
            record / records.METRICS_FILENAME,
            {"test": {"cells": [{"auc": auc}, {"auc": math.nan}]}},
        )
        records.finish_run(record, records.STATUS_COMPLETED)

    chosen, skipped = script.experiment_presets(
        presets, ["E4", "E12", "E13", "Q1"], ["*"], []
    )
    assert not skipped
    references = [
        "classical_tail/CNN",
        "classical_tail/C_w4_170",
        "classical_tail/C_w8_90",
        "ceiling/U64",
    ]
    assert {
        name: [preset.name for preset in experiment_presets]
        for name, experiment_presets in chosen.items()
    } == {
        "E4": ["deepcpg_tail/R_w32_L384_avg", "deepcpg_tail/R_w64_L384_avg"],
        "E12": ["deepcpg_tail/Q_L320", *references],
        "E13": ["deepcpg_tail/Q_L320"],
        "Q1": ["whole_model/G1_lin_ry", "whole_model/G3_mlp_ry"],
    }
    entries: list[dict[str, Any]] = script.plan_entries(
        chosen,
        [
            script.load_dataset(data_root / "2i"),
            script.load_dataset(data_root / "hepg2"),
        ],
        None,
        output,
        {
            preset.name: 1000.0
            for experiment_presets in chosen.values()
            for preset in experiment_presets
        },
        max_epochs=25,
        minimum_patience=3,
        presets_filepath=Path("presets.json"),
    )
    by_name: dict[str, dict[str, Any]] = {
        entry["name"]: entry for entry in entries
    }
    assert len(by_name) == len(entries)
    # E4: 2 presets x 2 datasets x 3 seeds, and the rerun of the collapsed
    # seed; E12: 5 presets x 4 folds x 2 datasets; E13: 2i only, 2 seeds; Q1:
    # 2 x 2 x 3.
    assert Counter(name.split("/")[0] for name in by_name) == {
        "E4": 13,
        "E12": 40,
        "E13": 2,
        "Q1": 12,
    }
    assert capsys.readouterr().out.strip() == ""
    assert "E12/hepg2/ceiling/U64/fold3/seed0" in by_name
    assert "E4/2i/deepcpg_tail/R_w32_L384_avg/seed11" in by_name
    assert "E4/2i/deepcpg_tail/R_w32_L384_avg/seed10" not in by_name

    wide_bank: list[str] = by_name[
        "E4/hepg2/deepcpg_tail/R_w32_L384_avg/seed0"
    ]["arguments"]
    assert "--all-training-samples" not in wide_bank
    assert wide_bank[wide_bank.index("--training-chromosomes") :][:4] == [
        "--training-chromosomes",
        "1",
        "3",
        "5",
    ]
    assert wide_bank[wide_bank.index("--patience") + 1] == "3"
    fold = by_name["E12/hepg2/deepcpg_tail/Q_L320/fold0/seed0"]
    assert fold["marker"] == str(
        output
        / "E12"
        / "hepg2"
        / "deepcpg_tail__Q_L320"
        / "fold0"
        / "seed0"
        / "COMPLETED"
    )
    arguments: list[str] = fold["arguments"]
    assert arguments[arguments.index("--training-chromosomes") :][:13] == [
        "--training-chromosomes",
        "2",
        "3",
        "4",
        "6",
        "--validation-chromosomes",
        "13",
        "--test-chromosomes",
        "1",
        "5",
        "--presets-file",
        "presets.json",
        "--all-training-samples",
    ]
    widened: list[str] = by_name["E13/2i/deepcpg_tail/Q_L320/seed1"][
        "arguments"
    ]
    assert "--all-training-samples" in widened
    global_circuit: list[str] = by_name["Q1/2i/whole_model/G3_mlp_ry/seed2"][
        "arguments"
    ]
    assert global_circuit[-6:] == [
        "--epochs",
        "25",
        "--patience",
        "3",
        "--selection-metric",
        "mean_cell_auc",
    ]


def _entry(name: str, hours: float, **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "script": "train",
        "arguments": [],
        "marker": f"/does/not/exist/{name}",
        "requires": [],
        "resources": "any",
        "hours": hours,
        **fields,
    }


def test_choose_entry(script: ModuleType) -> None:
    with TemporaryDirectory() as temp_dir:
        token_marker = Path(temp_dir) / "tokens"
        entries = [
            _entry("tokens", 0.0, resources="cpu", marker=str(token_marker)),
            _entry("long", 30.0),
            _entry("on tokens", 5.0, requires=[str(token_marker)]),
            _entry("short", 1.0),
        ]
        state = script.QueueState(claimed=set(), finished={})
        assert script.choose_entry(entries, state, "gpu") == (1, [])
        assert script.choose_entry(entries, state, "cpu") == (0, [])

        state = script.QueueState(claimed={0}, finished={1: 0})
        assert script.choose_entry(entries, state, "gpu") == (3, [2])
        assert script.choose_entry(entries, state, "cpu", hours_limit=0.5) == (
            None,
            [],
        )
        assert script.choose_entry(entries, state, "cpu") == (3, [])

        state = script.QueueState(claimed={0, 1, 3}, finished={})
        assert script.choose_entry(entries, state, "cpu") == (None, [2])
        token_marker.touch()
        assert script.choose_entry(entries, state, "cpu") == (2, [])

        # GPU workers write inputs when no run is left for them.
        state = script.QueueState(claimed={1, 2, 3}, finished={})
        token_marker.unlink()
        assert script.choose_entry(entries, state, "gpu") == (0, [])


def test_inputs_pending(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TemporaryDirectory() as temp_dir:
        queue_directory = Path(temp_dir)
        marker = str(queue_directory / "tokens")
        entries = [
            _entry("tokens", 0.0, resources="cpu", marker=marker),
            _entry("on tokens", 5.0, requires=[marker]),
        ]
        unclaimed = script.QueueState(claimed=set(), finished={})
        assert script.inputs_pending(queue_directory, entries, unclaimed, [1])
        failed = script.QueueState(claimed={0}, finished={0: 1})
        assert not script.inputs_pending(queue_directory, entries, failed, [1])

        (queue_directory / "claims").mkdir()
        assert script.claim(queue_directory, 0, {"job": "7"})
        assert not script.claim(queue_directory, 0, {"job": "8"})
        running = script.QueueState(claimed={0}, finished={})
        monkeypatch.setattr(script, "_active_job_ids", lambda: {"7"})
        assert script.inputs_pending(queue_directory, entries, running, [1])
        monkeypatch.setattr(script, "_active_job_ids", lambda: {"8"})
        assert not script.inputs_pending(queue_directory, entries, running, [1])


def test_work_and_status(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    commands: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, stdout, stderr, env, check):
        commands.append((command, env))
        returncode: int = 1 if "fails" in command else 0
        if returncode:
            stdout.write("Traceback (most recent call last):\n")
            stdout.write("RuntimeError: the reason it failed\n\n")
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(script, "_remaining_hours", lambda: math.inf)
    monkeypatch.setattr(script, "_active_job_ids", lambda: set())
    with TemporaryDirectory() as temp_dir:
        output = Path(temp_dir)
        token_marker = output / "tokens" / "COMPLETED"
        entries = [
            _entry(
                "tokens",
                0.0,
                script="tokenize",
                resources="cpu",
                marker=str(token_marker),
            ),
            _entry(
                "on tokens",
                3.0,
                requires=[str(token_marker)],
                marker=str(output / "a" / "COMPLETED"),
            ),
            _entry(
                "fails",
                2.0,
                arguments=["fails"],
                marker=str(output / "b" / "COMPLETED"),
            ),
            _entry("too long", 100.0, marker=str(output / "c" / "COMPLETED")),
        ]
        queue_directory = script.write_queue(
            output, entries, {"cpu_nodes": 1, "cpu_max_hours": 48.0}
        )
        script.main(
            ["work", str(queue_directory), "--resources", "cpu", "--cpus", "10"]
        )
        assert [command[1] for command, _ in commands] == [
            str(script.SCRIPTS["tokenize"]),
            str(script.SCRIPTS["train"]),
            str(script.SCRIPTS["train"]),
        ]
        tokenize_command, tokenize_environment = commands[0]
        assert "--threads" not in tokenize_command
        assert tokenize_environment["OMP_NUM_THREADS"] == "10"
        # CPU workers take the shortest runs first.
        train_command, train_environment = commands[1]
        assert train_command[-4:] == ["--threads", "8", "--loader-workers", "2"]
        assert train_environment["CUDA_VISIBLE_DEVICES"] == ""
        assert "fails" in train_command
        assert token_marker.exists()
        assert (output / "a" / "COMPLETED").exists()
        assert not (output / "b" / "COMPLETED").exists()
        assert script.read_state(queue_directory).finished == {
            0: 0,
            1: 0,
            2: 1,
        }

        script.main(
            [
                "work",
                str(queue_directory),
                "--resources",
                "gpu",
                "--gpu",
                "2",
                "--cpus",
                "7",
            ]
        )
        gpu_command, gpu_environment = commands[3]
        assert gpu_command[-6:] == [
            "--threads",
            "4",
            "--loader-workers",
            "3",
            "--gpus",
            "1",
        ]
        assert gpu_environment["CUDA_VISIBLE_DEVICES"] == "2"
        assert len(commands) == 4

        (queue_directory / "claims" / "9").mkdir()
        entries.append(_entry("interrupted", 1.0))
        with (queue_directory / "entries.jsonl").open("a") as fd:
            for entry in [_entry(f"pending {i}", 1.0) for i in range(4, 9)]:
                fd.write(json.dumps(entry) + "\n")
            fd.write(json.dumps(entries[-1]) + "\n")
        capsys.readouterr()
        script.main(["status", str(output)])
        output_lines: list[str] = capsys.readouterr().out.splitlines()
        assert output_lines[0] == f"Queue {queue_directory}: 10 entries"
        assert output_lines[1:5] == [
            "  completed: 3",
            "  failed: 1",
            "  interrupted: 1",
            "  pending: 5",
        ]
        assert output_lines[5].startswith("failed: fails, log ")
        assert output_lines[6] == "  RuntimeError: the reason it failed"
        assert output_lines[7].startswith("interrupted: interrupted, log ")


def test_submit_dry_run(
    script: ModuleType,
    data_root: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    output = data_root / "out"
    run_directory = output / "full" / "2i" / "whole_model__WP_L4" / "seed0"
    run_directory.mkdir(parents=True)
    (run_directory / "COMPLETED").touch()
    script.main(
        [
            "submit",
            "--data-directories",
            str(data_root / "2i"),
            str(data_root / "hepg2"),
            "--output-directory",
            str(output),
            "--experiments",
            "full",
            "E4",
            "--presets",
            "whole_model/WP_L4",
            "tokens/T64_QE_L4",
            "series_1/QA1",
            "deepcpg_tail/R_w32_L384_avg",
            "--seeds",
            "0",
            "1",
            "--cpu-time",
            "2-00:00:00",
            "--sbatch-args=--account=test",
            "--dry-run",
        ]
    )
    lines: list[str] = capsys.readouterr().out.splitlines()
    assert (
        lines[0] == "Skipping series_1/QA1: the same model as deepcpg_tail/Q_L4"
    )
    # full: 3 presets x 2 datasets x 2 seeds; E4: 1 x 2 x 2; 2 BPE inputs.
    assert lines[1] == "1 of 18 entries are already completed"
    assert lines[2].startswith("Wrote 15 runs (")
    assert " and 2 token inputs to " in lines[2]
    queue_directory = Path(lines[2].split(" to ")[-1])
    entries = script.read_entries(queue_directory)
    assert len(entries) == 17
    assert Counter(entry["name"].split("/")[0] for entry in entries) == {
        "tokens": 2,
        "full": 11,
        "E4": 4,
    }
    settings = json.loads((queue_directory / "queue.json").read_text())
    del settings["created"]
    # 15 runs on 4 GPUs per node; 2 token inputs and 15 runs on 4 workers.
    assert settings == {
        "gpu_nodes": 4,
        "cpu_nodes": 5,
        "cpu_max_hours": 48.0,
        "jobs": {},
    }
    gpu_command: list[str] = shlex.split(lines[4])
    cpu_command: list[str] = shlex.split(lines[5])
    assert gpu_command[:6] == [
        "sbatch",
        "--parsable",
        "--partition=gpu",
        "--array=0-3%4",
        "--time=7-00:00:00",
        f"--output={output / 'logs'}/%x_%A_%a.out",
    ]
    assert gpu_command[-3:] == [
        "--account=test",
        str(script.TEMPLATES["gpu"]),
        str(queue_directory),
    ]
    assert cpu_command[2:5] == [
        "--partition=main",
        "--array=0-4%5",
        "--time=2-00:00:00",
    ]
    assert cpu_command[-3:] == [
        str(script.TEMPLATES["cpu"]),
        str(queue_directory),
        "4",
    ]


def _record_run(
    records_directory: Path,
    output_directory: Path,
    started: datetime,
    *,
    status: str,
    cell_validation_aucs: list[list[float]] = (),
    test_aucs: list[float] = (),
    selection_metric: str = "mean_cell_auc",
) -> Path:
    """Record a run as training does, finishing 90 minutes after it started."""
    run_directory = records.create_run_record(
        records_directory,
        output_directory.parts[-4],
        {
            "output_directory": str(output_directory),
            "selection_metric": selection_metric,
        },
        {},
        ["a", "b"],
        now=started,
    )
    for epoch, aucs in enumerate(cell_validation_aucs):
        records.append_epoch(run_directory, epoch, 0.6, 0.6, 0.1 * epoch, aucs)
    if test_aucs:
        records.write_json(
            run_directory / records.METRICS_FILENAME,
            {
                "test": {
                    "pooled": {"auc": 0.77, "observed_sites": 1000},
                    "cells": [{"auc": auc} for auc in test_aucs],
                }
            },
        )
    if status != records.STATUS_RUNNING:
        records.finish_run(
            run_directory, status, now=started + timedelta(minutes=90)
        )
    return run_directory


def test_summarize(script: ModuleType, capsys: pytest.CaptureFixture) -> None:
    with TemporaryDirectory() as temp_dir:
        output = Path(temp_dir).resolve()
        full_records = output / "full" / "records"
        run = output / "full" / "2i" / "whole_model__WP_L4"
        start = datetime(2026, 11, 1, 12, tzinfo=UTC)
        # A failed attempt and its successful rerun.
        _record_run(full_records, run / "seed0", start, status="failed")
        completed = _record_run(
            full_records,
            run / "seed0",
            start + timedelta(days=1),
            status="completed",
            cell_validation_aucs=[[0.6, 0.7], [0.7, 0.8], [0.65, math.nan]],
            test_aucs=[0.8, 0.7],
        )
        # A completed run, then a stale record of the same run.
        _record_run(
            full_records,
            run / "seed1",
            start,
            status="completed",
            cell_validation_aucs=[[0.6, 0.6], [0.5, 0.5]],
            test_aucs=[0.6, math.nan],
            selection_metric="pooled_auc",
        )
        _record_run(
            full_records,
            run / "seed1",
            start + timedelta(days=2),
            status="running",
        )
        # A collapsed fold, and a run of another output directory.
        _record_run(
            output / "E12" / "records",
            output
            / "E12"
            / "hepg2"
            / "deepcpg_tail__Q_L320"
            / "fold2"
            / "seed0",
            start,
            status="completed",
            cell_validation_aucs=[[0.5, 0.5]],
            test_aucs=[0.5001, 0.4999],
        )
        _record_run(
            full_records,
            Path(temp_dir).parent / "elsewhere" / "2i" / "x" / "seed0",
            start,
            status="completed",
        )

        script.main(["summarize", str(output)])
        assert capsys.readouterr().out.strip() == (
            f"Wrote 3 runs to {output / 'summary.tsv'} (completed 3; "
            "1 collapsed)"
        )
        lines: list[list[str]] = [
            line.split("\t")
            for line in (output / "summary.tsv").read_text().splitlines()
        ]
        assert lines[0] == list(script.SUMMARY_COLUMNS)
        rows: list[dict[str, str]] = [
            dict(zip(lines[0], line)) for line in lines[1:]
        ]
        assert [
            (
                row["experiment"],
                row["dataset"],
                row["preset"],
                row["fold"],
                row["seed"],
            )
            for row in rows
        ] == [
            ("E12", "hepg2", "deepcpg_tail/Q_L320", "fold2", "0"),
            ("full", "2i", "whole_model/WP_L4", "", "0"),
            ("full", "2i", "whole_model/WP_L4", "", "1"),
        ]
        collapsed, rerun, pooled = rows
        assert collapsed["collapsed"] == "True"
        assert rerun == {
            "experiment": "full",
            "dataset": "2i",
            "preset": "whole_model/WP_L4",
            "fold": "",
            "seed": "0",
            "status": "completed",
            "epochs_trained": "3",
            "best_epoch": "1",
            "best_validation_score": "0.750000",
            "test_mean_cell_auc": "0.750000",
            "test_pooled_auc": "0.770000",
            "test_observed_labels": "1000",
            "collapsed": "False",
            "hours": "1.500000",
            "record": str(completed),
        }
        # Pooled selection keeps epoch 1 (pooled AUC 0.1); NaN cells are skipped.
        assert (pooled["status"], pooled["best_epoch"]) == ("completed", "1")
        assert pooled["best_validation_score"] == "0.100000"
        assert pooled["test_mean_cell_auc"] == "0.600000"


def test_status_collapsed_runs(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(script, "_active_job_ids", lambda: set())
    with TemporaryDirectory() as temp_dir:
        output = Path(temp_dir).resolve()
        records_directory = output / "full" / "records"
        preset_directory = (
            output / "full" / "hepg2" / "deepcpg_tail__Q_w64_L128"
        )
        start = datetime(2026, 11, 1, 12, tzinfo=UTC)
        # Seed 0 collapsed and is not rerun yet; seed 2 collapsed, and so did
        # its rerun, seed 12.
        entries: list[dict[str, Any]] = []
        for seed in (0, 2, 12):
            run_directory = preset_directory / f"seed{seed}"
            _record_run(
                records_directory,
                run_directory,
                start + timedelta(hours=seed),
                status="completed",
                test_aucs=[0.5001, 0.4999],
            )
            run_directory.mkdir(parents=True)
            (run_directory / "COMPLETED").touch()
            entries.append(
                _entry(
                    f"full/hepg2/deepcpg_tail/Q_w64_L128/seed{seed}",
                    1.0,
                    arguments=["--records-directory", str(records_directory)],
                    marker=str(run_directory / "COMPLETED"),
                )
            )
        script.write_queue(
            output, entries, {"cpu_nodes": 0, "cpu_max_hours": 1}
        )

        script.main(["status", str(output)])
        assert capsys.readouterr().out.splitlines()[1:] == [
            "  completed: 3",
            "collapsed: full/hepg2/deepcpg_tail/Q_w64_L128/seed0",
            "collapsed, rerun as seed12: full/hepg2/deepcpg_tail/Q_w64_L128/seed2",
            (
                "collapsed after its rerun, kept: "
                "full/hepg2/deepcpg_tail/Q_w64_L128/seed12"
            ),
            (
                "Submit again to rerun failed and interrupted entries, and to "
                "rerun collapsed runs with seed + 10."
            ),
        ]

        # Once seed 0's rerun completes, nothing is left to submit.
        (preset_directory / "seed10").mkdir()
        (preset_directory / "seed10" / "COMPLETED").touch()
        script.main(["status", str(output)])
        lines: list[str] = capsys.readouterr().out.splitlines()
        assert lines[2] == (
            "collapsed, rerun as seed10: full/hepg2/deepcpg_tail/Q_w64_L128/seed0"
        )
        assert not any(line.startswith("Submit again") for line in lines)


def test_learning_rate_sweep(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_root: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    output = data_root / "out"
    dataset = script.load_dataset(data_root / "2i")
    sweeps = list(script.SWEEP_EXPERIMENTS)
    assert sweeps == ["LR1e-2", "LR3e-3", "LR1e-3", "LR3e-4"]
    chosen, _ = script.experiment_presets(
        presets, [*sweeps, "full"], ["classical_whole/FacM_r2"], []
    )
    costs = {"classical_whole/FacM_r2": 1.0}
    options = {
        "max_epochs": 25,
        "minimum_patience": 3,
        "presets_filepath": Path("presets.json"),
    }
    entries = script.plan_entries(
        chosen, [dataset], None, output, costs, **options
    )
    # One run per rate on 2i's full training set (the sweep's other dataset,
    # HCC, is not given); the full runs wait for the sweep.
    assert sorted(entry["name"] for entry in entries) == sorted(
        f"{sweep}/2i/classical_whole/FacM_r2/seed0" for sweep in sweeps
    )
    assert sorted(
        entry["arguments"][entry["arguments"].index("--learning-rate") + 1]
        for entry in entries
    ) == ["0.0003", "0.001", "0.003", "0.01"]
    assert all("--all-training-samples" in e["arguments"] for e in entries)
    assert "Waiting for the learning-rate sweep: classical_whole/FacM_r2" in (
        capsys.readouterr().out
    )

    # Every rate on both sweep datasets: 3e-3 has the best mean validation.
    start = datetime(2026, 11, 1, tzinfo=UTC)
    scores = {"LR1e-2": 0.70, "LR3e-3": 0.74, "LR1e-3": 0.73, "LR3e-4": 0.71}
    for hour, (sweep, score) in enumerate(scores.items()):
        for name in ("hcc", "2i"):
            _record_run(
                output / sweep / "records",
                output / sweep / name / "classical_whole__FacM_r2" / "seed0",
                start + timedelta(hours=hour),
                status="completed",
                cell_validation_aucs=[[score, score]],
                test_aucs=[0.7],
            )
            if sweep == "LR3e-4":
                break
    assert script.swept_learning_rates(output) == {}
    _record_run(
        output / "LR3e-4" / "records",
        output / "LR3e-4" / "2i" / "classical_whole__FacM_r2" / "seed0",
        start,
        status="completed",
        cell_validation_aucs=[[0.71, 0.71]],
        test_aucs=[0.7],
    )
    assert script.swept_learning_rates(output) == {
        "classical_whole/FacM_r2": 0.003
    }
    entries = script.plan_entries(
        {"full": chosen["full"]}, [dataset], None, output, costs, **options
    )
    assert [entry["name"] for entry in entries] == [
        f"full/2i/classical_whole/FacM_r2/seed{seed}" for seed in (0, 1, 2)
    ]
    arguments = entries[0]["arguments"]
    assert arguments[arguments.index("--learning-rate") + 1] == "0.003"


# Sizes of the matched comparisons' tests: front end, total, kind, input.
SIZES: dict[str, tuple[int, int, str, str]] = {
    "classical_tail/CNN": (5760, 4102918, "tail", "bases"),
    "classical_tail/C_2": (100, 4000000, "tail", "bases"),
    "classical_tail/C_3": (240, 4050000, "tail", "bases"),
    "classical_pooling/C_4": (200, 3000, "pooling", "bases"),
    "deepcpg_tail/Q_L4": (200, 4048206, "tail", "bases"),
    "deepcpg_tail/R_L12": (200, 4048206, "tail", "bases"),
    "deepcpg_tail/Q_L1": (146, 4048152, "tail", "bases"),
}


def _fake_sizes(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Measure models with `SIZES`, recording the presets measured."""
    measured: list[str] = []

    def preset_size(preset: model_factory.Preset, record: Path) -> Any:
        measured.append(preset.name)
        return script.ModelSize(*SIZES[preset.name])

    monkeypatch.setattr(script, "preset_size", preset_size)
    return measured


def test_matched_pairs(script: ModuleType) -> None:
    sizes = {name: script.ModelSize(*size) for name, size in SIZES.items()}
    sizes["whole_model/WP_L4"] = script.ModelSize(3500, 3500, "whole", "bases")
    sizes["whole_model/G1_lin"] = script.ModelSize(80, 80, "whole", "bases")
    sizes["ceiling/U64"] = script.ModelSize(150, 4000000, "tail", "bases")
    # (mean validation score, mean test AUC); C_3 has the better validation
    # score and the worse test AUC.
    means = {
        "classical_tail/CNN": (0.80, 0.77),
        "classical_tail/C_2": (0.60, 0.73),
        "classical_tail/C_3": (0.70, 0.72),
        "classical_pooling/C_4": (0.90, 0.80),
        "ceiling/U64": (0.95, 0.90),
        "deepcpg_tail/Q_L4": (0.70, 0.74),
        "deepcpg_tail/Q_L1": (0.65, 0.735),
        "whole_model/WP_L4": (0.60, 0.70),
        "whole_model/G1_lin": (0.60, 0.70),
    }
    pairs = {
        (pair.table, pair.quantum): pair
        for pair in script.matched_pairs(means, sizes)
    }
    assert {key: pair.classical for key, pair in pairs.items()} == {
        # Q_L4's front-end candidates are C_2 and C_3 (at most 250); C_3's
        # validation score is the better. The pooling head's C_4 and the
        # ceiling are not candidates.
        ("front-end", "deepcpg_tail/Q_L4"): "classical_tail/C_3",
        # Only C_2 has at most 1.25 x 146 front-end parameters.
        ("front-end", "deepcpg_tail/Q_L1"): "classical_tail/C_2",
        # Whole models meet every classical model on bases by total size,
        # and are not in the total table; none is as small as G1_lin.
        ("front-end", "whole_model/WP_L4"): "classical_pooling/C_4",
        ("total", "deepcpg_tail/Q_L4"): "classical_tail/CNN",
        ("total", "deepcpg_tail/Q_L1"): "classical_tail/CNN",
    }
    q_l4 = pairs[("front-end", "deepcpg_tail/Q_L4")]
    assert (q_l4.quantum_size, q_l4.classical_size, q_l4.scored) == (
        200,
        240,
        True,
    )
    # Q_L1 leads C_2, which is less than 0.8 times its size: not scored.
    assert not pairs[("front-end", "deepcpg_tail/Q_L1")].scored
    whole = pairs[("front-end", "whole_model/WP_L4")]
    assert (whole.quantum_size, whole.classical_size, whole.scored) == (
        3500,
        3000,
        True,
    )
    assert pairs[("total", "deepcpg_tail/Q_L4")].classical_size == 4102918


def test_close_gaps(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_sizes(script, monkeypatch)
    output = data_root / "out"
    start = datetime(2026, 11, 1, tzinfo=UTC)
    for experiment, preset, auc in (
        ("full", "classical_tail__CNN", 0.770),
        ("full", "classical_tail__C_3", 0.700),
        ("full", "deepcpg_tail__Q_L4", 0.766),
        ("full", "deepcpg_tail__Q_L1", 0.750),
        ("E12", "deepcpg_tail__Q_L320", 0.700),
        ("E12", "ceiling__U64", 0.702),
        ("E12", "classical_tail__CNN", 0.690),
    ):
        for seed in range(3):
            fold: tuple[str, ...] = ("fold0",) if experiment == "E12" else ()
            _record_run(
                output / experiment / "records",
                output.joinpath(experiment, "2i", preset, *fold, f"seed{seed}"),
                start + timedelta(minutes=seed),
                status="completed",
                cell_validation_aucs=[[auc, auc]],
                test_aucs=[auc + 0.001 * seed, auc - 0.001 * seed],
            )
    # Q_L4 is within 0.005 of its total-size comparator, the CNN, and ahead
    # of its front-end one, C_3. Q_L1 has no front-end comparator (C_3 has
    # more than 1.25 x 146 parameters) and trails the CNN by 0.02. In E12,
    # Q_L320 is within 0.005 of the best reference, the ceiling U64.
    assert script.close_gap_targets(output, 0.005, PRESETS_FILEPATH) == {
        "full": {
            ("2i", "", "classical_tail/CNN"),
            ("2i", "", "classical_tail/C_3"),
            ("2i", "", "deepcpg_tail/Q_L4"),
        },
        "E12": {
            ("2i", "fold0", "deepcpg_tail/Q_L320"),
            ("2i", "fold0", "ceiling/U64"),
        },
    }
    targets = script.close_gap_targets(output, 0.001, PRESETS_FILEPATH)
    assert targets == {
        "full": {
            ("2i", "", "classical_tail/C_3"),
            ("2i", "", "deepcpg_tail/Q_L4"),
        }
    }

    chosen = {
        "full": [
            presets[name]
            for name in (
                "classical_tail/CNN",
                "deepcpg_tail/Q_L4",
                "deepcpg_tail/Q_L1",
            )
        ]
    }
    entries = script.plan_entries(
        chosen,
        [script.load_dataset(data_root / "2i")],
        None,
        output,
        {preset.name: 1.0 for preset in chosen["full"]},
        max_epochs=25,
        minimum_patience=3,
        presets_filepath=PRESETS_FILEPATH,
        close_gap_tolerance=0.001,
    )
    # The 3 seeds of each model are recorded but not marked completed, so
    # they are planned again; Q_L4 also gets seeds 3 and 4, and the CNN, 0.004
    # ahead of it, does not (C_3 is not among the chosen presets).
    assert Counter(entry["name"].rsplit("/", 1)[0] for entry in entries) == {
        "full/2i/classical_tail/CNN": 3,
        "full/2i/deepcpg_tail/Q_L4": 5,
        "full/2i/deepcpg_tail/Q_L1": 3,
    }


def _record_outputs(
    output: Path,
    experiment: str,
    preset: str,
    seed: int,
    labels: np.ndarray,
    predictions: np.ndarray,
    validation: float = 0.7,
) -> None:
    """Record a completed run with its test outputs, as training does."""
    run_directory = (
        output / experiment / "2i" / preset.replace("/", "__") / f"seed{seed}"
    )
    record = records.create_run_record(
        output / experiment / "records",
        "2i",
        {
            "output_directory": str(run_directory),
            "selection_metric": "mean_cell_auc",
        },
        {},
        ["a", "b"],
        now=datetime(2026, 11, 1, tzinfo=UTC)
        + timedelta(seconds=len(preset) + seed),
    )
    records.append_epoch(record, 0, 0.6, 0.6, 0.6, [validation, validation])
    records.write_split_outputs(
        record,
        "test",
        records.SplitOutputs(
            chromosome_names=["2"],
            chromosome_indices=np.zeros(len(labels), np.int32),
            positions=np.arange(len(labels), dtype=np.int64) * 10,
            labels=labels,
            predictions=predictions,
        ),
    )
    cells = [
        {
            "auc": float(
                roc_auc_score(
                    labels[~np.isnan(labels[:, c]), c],
                    predictions[~np.isnan(labels[:, c]), c],
                )
            )
        }
        for c in range(labels.shape[1])
    ]
    records.write_json(
        record / records.METRICS_FILENAME, {"test": {"cells": cells}}
    )
    records.finish_run(record, records.STATUS_COMPLETED)


def _read_tsv(filepath: Path) -> list[dict[str, str]]:
    lines = filepath.read_text().splitlines()
    return [
        dict(zip(lines[0].split("\t"), line.split("\t"), strict=True))
        for line in lines[1:]
    ]


def test_compare(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    measured = _fake_sizes(script, monkeypatch)
    rng = np.random.default_rng(0)
    labels = (rng.random((400, 2)) < 0.5).astype(np.float32)
    labels[rng.random((400, 2)) < 0.1] = np.nan

    def predictions(signal: float) -> np.ndarray:
        return (signal * np.nan_to_num(labels) + rng.random((400, 2))).astype(
            np.float32
        )

    with TemporaryDirectory() as temp_dir:
        output = Path(temp_dir).resolve()
        for seed in (0, 1):
            for preset, signal, validation in (
                ("classical_tail/CNN", 1.0, 0.8),
                ("deepcpg_tail/Q_L4", 0.5, 0.7),
            ):
                _record_outputs(
                    output,
                    "full",
                    preset,
                    seed,
                    labels,
                    predictions(signal),
                    validation,
                )
        for preset, signal, validation in (
            ("deepcpg_tail/Q_L1", 0.4, 0.7),
            ("deepcpg_tail/R_L12", 0.45, 0.7),
            # C_3 has the better validation score, C_2 the better test AUC.
            ("classical_tail/C_2", 0.3, 0.6),
            ("classical_tail/C_3", 0.2, 0.7),
        ):
            _record_outputs(
                output,
                "full",
                preset,
                0,
                labels,
                predictions(signal),
                validation,
            )
        for preset in ("deepcpg_tail/Q_L128", "classical_tail/CNN"):
            _record_outputs(output, "splitC", preset, 0, labels, predictions(1))

        script.main(
            ["compare", str(output), "--replicates", "200", "--sites", "300"]
        )
        assert "Wrote 8 comparisons" in capsys.readouterr().out
        # Only the full sets' models are measured.
        assert sorted(measured) == sorted(
            name for name in SIZES if name != "classical_pooling/C_4"
        )
        folder = output / "comparisons"
        rows = _read_tsv(folder / "comparisons.tsv")
        assert {(r["family"], r["model_a"], r["model_b"]) for r in rows} == {
            (
                "full matched front-end",
                "deepcpg_tail/Q_L1",
                "classical_tail/C_2",
            ),
            (
                "full matched front-end",
                "deepcpg_tail/Q_L4",
                "classical_tail/C_3",
            ),
            (
                "full matched front-end",
                "deepcpg_tail/R_L12",
                "classical_tail/C_3",
            ),
            ("full matched total", "deepcpg_tail/Q_L1", "classical_tail/CNN"),
            ("full matched total", "deepcpg_tail/Q_L4", "classical_tail/CNN"),
            ("full matched total", "deepcpg_tail/R_L12", "classical_tail/CNN"),
            (
                "full complex vs real",
                "deepcpg_tail/Q_L4",
                "deepcpg_tail/R_L12",
            ),
            (
                "splitC vs classical_tail/CNN",
                "deepcpg_tail/Q_L128",
                "classical_tail/CNN",
            ),
        }
        assert [
            (r["model_a"], r["model_b"])
            for r in _read_tsv(folder / "matched_front_end.tsv")
        ] == [
            ("deepcpg_tail/Q_L1", "classical_tail/C_2"),
            ("deepcpg_tail/Q_L4", "classical_tail/C_3"),
            ("deepcpg_tail/R_L12", "classical_tail/C_3"),
        ]
        assert len(_read_tsv(folder / "matched_total.tsv")) == 3
        q_l4 = next(
            r
            for r in rows
            if r["family"] == "full matched total"
            and r["model_a"] == "deepcpg_tail/Q_L4"
        )
        assert (q_l4["seeds_a"], q_l4["seeds_b"], q_l4["sites"]) == (
            "2",
            "2",
            "300",
        )
        assert (q_l4["size_a"], q_l4["size_b"], q_l4["scored"]) == (
            "4048206",
            "4102918",
            "True",
        )
        assert (q_l4["validation_a"], q_l4["validation_b"]) == (
            "0.700000",
            "0.800000",
        )
        assert (
            float(q_l4["low"])
            < float(q_l4["difference"])
            < float(q_l4["high"])
            < 0
        )
        assert float(q_l4["p_value"]) < 0.05
        assert float(q_l4["q_value"]) >= float(q_l4["p_value"])
        # The unsampled difference uses every site, from the run records.
        assert float(q_l4["difference_all_sites"]) == pytest.approx(
            float(q_l4["test_a"]) - float(q_l4["test_b"]), abs=2e-6
        )
        assert float(q_l4["difference_all_sites"]) < 0
        # Q_L1 leads C_2, which is less than 0.8 times its size.
        q_l1 = next(
            r
            for r in rows
            if r["family"] == "full matched front-end"
            and r["model_a"] == "deepcpg_tail/Q_L1"
        )
        assert q_l1["scored"] == "False"
        twins = next(r for r in rows if r["family"] == "full complex vs real")
        assert (twins["size_a"], twins["scored"]) == ("", "")

        metric_rows = _read_tsv(folder / "metrics.tsv")
        assert len(metric_rows) == 10
        # The CNN's scores separate the classes, so its AUPRC is 1.
        assert all(
            0 < float(r["test_mean_cell_auprc"]) <= 1 for r in metric_rows
        )
        cnn = next(
            r
            for r in metric_rows
            if r["preset"] == "classical_tail/CNN" and r["experiment"] == "full"
        )
        q_l1_metrics = next(
            r for r in metric_rows if r["preset"] == "deepcpg_tail/Q_L1"
        )
        assert float(cnn["test_mean_cell_mcc"]) > float(
            q_l1_metrics["test_mean_cell_mcc"]
        )

        # A second run reuses the cached bootstrap values.
        lines = (folder / "comparisons.tsv").read_text().splitlines()
        cache = list((folder / "cache").glob("*.npy"))
        assert len(cache) == 10
        script.main(
            ["compare", str(output), "--replicates", "200", "--sites", "300"]
        )
        assert (folder / "comparisons.tsv").read_text().splitlines() == lines


def test_complex_real_pairs(
    script: ModuleType, presets: dict[str, model_factory.Preset]
) -> None:
    pairs = script.load_pairs(script.PAIRS_FILEPATH)
    assert len(pairs) == len(set(pairs)) == 37
    for complex_name, real_name in pairs:
        complex_preset, real_preset = presets[complex_name], presets[real_name]
        assert complex_preset.model == real_preset.model
        assert complex_preset.input == real_preset.input
        assert real_preset.options.get("ansatz") in ("real", "basic_ry")
        assert complex_preset.options.get("ansatz") not in ("real", "basic_ry")


def test_preset_size(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_root: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    record = records.create_run_record(
        data_root / "records",
        "2i",
        {
            "data_directory": str(data_root / "2i"),
            "training_chromosomes": ["1"],
        },
        {},
        ["cell"],
    )
    for name, kind in (
        ("pooling_head/Q_L4", "pooling"),
        ("classical_pooling/C_4", "pooling"),
        ("classical_whole/OH_lin", "whole"),
        ("whole_model/WP_L4", "whole"),
    ):
        preset = presets[name]
        model = model_factory.build_model(
            preset.model, 41, 1, options=preset.options
        )
        assert script.preset_size(preset, record) == script.ModelSize(
            *model_factory.model_sizes(model), kind, "bases"
        )
    # 41 bp sequences are shorter than ID_w64's windows.
    sizes = script.measure_sizes(
        [
            {"preset": name, "dataset": "2i", "record": str(record)}
            for name in ("pooling_head/Q_L4", "classical_tail/ID_w64", "x/y")
        ],
        presets,
    )
    assert list(sizes) == [("pooling_head/Q_L4", "2i")]
    output = capsys.readouterr().out
    assert "Not measuring classical_tail/ID_w64 on 2i: Windows of 64" in output
    assert "Not measuring x/y: not in the presets file" in output


@pytest.mark.parametrize(
    "nodes", [["--cpu-nodes", "13"], ["--gpu-nodes", "5", "--cpu-nodes", "0"]]
)
def test_submit_node_limit(script: ModuleType, nodes: list[str]) -> None:
    with pytest.raises(SystemExit):
        script._parse_script_args(
            [
                "submit",
                "--data-directories",
                "d",
                "--output-directory",
                "o",
                *nodes,
            ]
        )


def test_submit_cpu_nodes_fill_limit(
    script: ModuleType, data_root: Path, capsys: pytest.CaptureFixture
) -> None:
    # Without GPU nodes, CPU nodes may take all 16: 19 presets x 4 seeds,
    # all short, on 4 workers each would need 19.
    script.main(
        [
            "submit",
            "--data-directories",
            str(data_root / "2i"),
            "--output-directory",
            str(data_root / "out"),
            "--experiments",
            "full",
            "--presets",
            "deepcpg_tail/Q_L*",
            "pooling_head/*",
            "--seeds",
            "0",
            "1",
            "2",
            "3",
            "--gpu-nodes",
            "0",
            "--dry-run",
        ]
    )
    output: str = capsys.readouterr().out
    assert "--array=0-15%16" in output
    assert "--partition=gpu" not in output
