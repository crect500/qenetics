import importlib.util
import json
import math
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from qenetics.qcpg import model_factory, records
from qenetics.tools import data

SCRIPT: Path = Path("scripts/cluster/plan_runs.py")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    specification = importlib.util.spec_from_file_location("plan_runs", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def presets() -> dict[str, model_factory.Preset]:
    return model_factory.load_presets(Path("config/quantum_models.json"))


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
        presets, ["*"], script.REPORTED_PRESETS
    )
    assert skipped["deepcpg_tail/Q_L320"] == "excluded"
    assert skipped["whole_model/H1b_cnnq_ry"] == "excluded"
    assert skipped["series_1/QA1"] == "the same model as deepcpg_tail/Q_L4"
    assert skipped["series_1/LD5k_Q1"] == "the same model as series_1/Q1"
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
        presets, ["E4", "E12", "E13", "Q1"], ["*"], script.REPORTED_PRESETS
    )
    assert not skipped
    assert {
        name: [preset.name for preset in experiment_presets]
        for name, experiment_presets in chosen.items()
    } == {
        "E4": ["deepcpg_tail/R_w32_L384_avg", "deepcpg_tail/R_w64_L384_avg"],
        "E12": ["deepcpg_tail/Q_L320"],
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
    # seed; E12: 4 folds x 2 datasets; E13: 2i only, 2 seeds; Q1: 2 x 2 x 3.
    assert Counter(name.split("/")[0] for name in by_name) == {
        "E4": 13,
        "E12": 8,
        "E13": 2,
        "Q1": 12,
    }
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
        assert output_lines[6].startswith("interrupted: interrupted, log ")


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
