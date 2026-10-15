import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import pytest

SCRIPT: Path = Path("scripts/cluster/slurm_runs.py")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    specification = importlib.util.spec_from_file_location("slurm_runs", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_select_presets(script: ModuleType) -> None:
    names: list[str] = ["a/Q_L1", "a/Q_L4", "a/R_L3", "b/Q_L4"]
    assert script.select_presets(["a/Q_*", "b/Q_L4"], names) == [
        "a/Q_L1",
        "a/Q_L4",
        "b/Q_L4",
    ]
    with pytest.raises(ValueError, match="No presets match c/"):
        script.select_presets(["c/*"], names)


def test_build_runs(script: ModuleType) -> None:
    runs = script.build_runs(
        ["deepcpg_tail/Q_L4"],
        [Path("/data/hcc"), Path("/data/mesc")],
        [0, 1],
        Path("/scratch"),
        ["--epochs", "3"],
        gpu=True,
    )
    assert [run["name"] for run in runs] == [
        "hcc/deepcpg_tail/Q_L4/seed0",
        "hcc/deepcpg_tail/Q_L4/seed1",
        "mesc/deepcpg_tail/Q_L4/seed0",
        "mesc/deepcpg_tail/Q_L4/seed1",
    ]
    arguments: list[str] = runs[1]["arguments"]
    assert arguments[arguments.index("--output-directory") + 1] == str(
        Path("/scratch/hcc/deepcpg_tail__Q_L4/seed1")
    )
    assert arguments[arguments.index("--seed") + 1] == "1"
    assert "--no-report" in arguments
    assert arguments[-4:] == ["--gpus", "1", "--epochs", "3"]


def test_submit_dry_run(
    script: ModuleType, capsys: pytest.CaptureFixture
) -> None:
    with TemporaryDirectory() as temp_dir:
        data_directory = Path(temp_dir) / "hcc"
        data_directory.mkdir()
        script.main(
            [
                "submit",
                "--presets",
                "whole_model/G1_*",
                "--data-directories",
                str(data_directory),
                "--seeds",
                "0",
                "1",
                "--output-directory",
                str(Path(temp_dir) / "runs"),
                "--max-concurrent",
                "4",
                "--sbatch-args=--partition=main --time=1:00:00",
                "--dry-run",
                "--",
                "--test-chromosomes",
                "2",
            ]
        )
        output: list[str] = capsys.readouterr().out.splitlines()
        runs_filepath = Path(output[0].split(" to ")[1])
        runs = [
            json.loads(line) for line in runs_filepath.read_text().splitlines()
        ]
        # G1_lin, G1_lin_ry, G1_lin_ry6 and G1_lin_ryz, two seeds each.
        assert len(runs) == 8
        assert runs[0]["arguments"][-2:] == ["--test-chromosomes", "2"]
        assert "--array=0-7%4" in output[1]
        assert "--partition=main --time=1:00:00" in output[1]
        assert "train_cpu.sbatch" in output[1]


def test_submit_errors(script: ModuleType) -> None:
    with pytest.raises(ValueError, match="Data directories not found"):
        script.main(
            [
                "submit",
                "--presets",
                "whole_model/G1_lin",
                "--data-directories",
                "/does/not/exist",
                "--output-directory",
                ".",
                "--dry-run",
            ]
        )


def test_run(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        script.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )
    with TemporaryDirectory() as temp_dir:
        runs_filepath = Path(temp_dir) / "runs.jsonl"
        runs_filepath.write_text(
            json.dumps({"name": "a", "arguments": ["--preset", "x y"]})
            + "\n"
            + json.dumps({"name": "b", "arguments": ["--seed", "1"]})
            + "\n"
        )
        script.main(["run", str(runs_filepath), "1"])
        assert calls == [
            [sys.executable, str(script.TRAIN_SCRIPT), "--seed", "1"]
        ]
        with pytest.raises(IndexError, match="Run 2 does not exist"):
            script.main(["run", str(runs_filepath), "2"])
