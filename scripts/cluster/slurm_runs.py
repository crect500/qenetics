"""
Run presets of train_quantum_model.py as SLURM job arrays.

`submit` writes a run list, one training run per preset, dataset and seed,
each with its own output directory, and submits one array task per run with
train_cpu.sbatch or train_gpu.sbatch. Array tasks call `run`, which starts
their run's training. Runs skip the HTML report; build it afterwards with
scripts/build_training_report.py on the records directory.

    python scripts/cluster/slurm_runs.py submit \\
        --presets 'deepcpg_tail/Q_L*' whole_model/H1b_cnnq \\
        --data-directories /data/hcc --seeds 0 1 2 \\
        --output-directory /scratch/qenetics --resources gpu \\
        --sbatch-args='--partition=gpu --time=24:00:00' \\
        -- --training-chromosomes 1 3 5 7 9 --validation-chromosomes 13 \\
        --test-chromosomes 2 4
"""

import json
import shlex
import shutil
import subprocess
import sys
from argparse import REMAINDER, ArgumentParser, Namespace
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

REPOSITORY: Path = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT: Path = REPOSITORY / "scripts" / "train_quantum_model.py"
PRESETS_FILEPATH: Path = REPOSITORY / "config" / "quantum_models.json"
TEMPLATES: dict[str, Path] = {
    "cpu": Path(__file__).resolve().parent / "train_cpu.sbatch",
    "gpu": Path(__file__).resolve().parent / "train_gpu.sbatch",
}


def _parse_script_args(arguments: list[str] | None = None) -> Namespace:
    parser = ArgumentParser("slurm_runs", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser(
        "submit", help="Write a run list and submit it as a job array."
    )
    submit.add_argument(
        "--presets",
        nargs="+",
        required=True,
        help="Preset names or shell-style patterns, e.g. 'deepcpg_tail/*'.",
    )
    submit.add_argument(
        "--data-directories",
        type=Path,
        nargs="+",
        required=True,
        help="Dataset directories, each named after its dataset.",
    )
    submit.add_argument(
        "--seeds", type=int, nargs="+", default=[0], help="The run seeds."
    )
    submit.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="The root of run outputs, records, run lists and logs.",
    )
    submit.add_argument(
        "--resources",
        choices=list(TEMPLATES),
        default="cpu",
        help="The batch template: CPU-only nodes or one GPU per run.",
    )
    submit.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="The most array tasks to run at once; 0 for no limit.",
    )
    submit.add_argument(
        "--sbatch-args",
        default="",
        help="Further sbatch options, e.g. '--partition=gpu --time=24:00:00'.",
    )
    submit.add_argument("--presets-file", type=Path, default=PRESETS_FILEPATH)
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the run list and print the sbatch command only.",
    )
    submit.add_argument(
        "training_arguments",
        nargs=REMAINDER,
        help="Arguments for every run, after '--'.",
    )
    run = commands.add_parser("run", help="Start one run of a run list.")
    run.add_argument("runs_filepath", type=Path)
    run.add_argument("index", type=int)
    return parser.parse_args(arguments)


def select_presets(patterns: list[str], names: list[str]) -> list[str]:
    """
    Find the presets matching each pattern, in the presets file's order.

    Args
    ----
    patterns: The patterns to search for.
    names: The known names.

    Returns
    -------
    Any names matching the pattern.

    Raises
    ------
    ValueError if a pattern matches no preset.
    """
    unmatched: list[str] = [
        pattern
        for pattern in patterns
        if not any(fnmatchcase(name, pattern) for name in names)
    ]
    if unmatched:
        raise ValueError(f"No presets match {', '.join(unmatched)}")

    return [
        name
        for name in names
        if any(fnmatchcase(name, pattern) for pattern in patterns)
    ]


def build_runs(
    presets: list[str],
    data_directories: list[Path],
    seeds: list[int],
    output_directory: Path,
    training_arguments: list[str],
    *,
    gpu: bool = False,
) -> list[dict[str, Any]]:
    """
    Return one training run descriptor per preset, dataset and seed.

    Args
    ----
    presets: The known model keywords.
    data_directories: The provided data directory filepaths.
    seeds: The random number seed to use for generators.
    output_directory: The filepath at which to save the training output.
    training_arguments: The training parameters for each run.

    Returns
    -------
    Each run's name and its train_quantum_model.py arguments.
    """
    runs: list[dict[str, Any]] = []
    for data_directory in data_directories:
        dataset: str = data_directory.name
        for preset in presets:
            for seed in seeds:
                run_directory: Path = (
                    output_directory
                    / dataset
                    / preset.replace("/", "__")
                    / f"seed{seed}"
                )
                runs.append(
                    {
                        "name": f"{dataset}/{preset}/seed{seed}",
                        "arguments": [
                            "--preset",
                            preset,
                            "--data-directory",
                            str(data_directory),
                            "--output-directory",
                            str(run_directory),
                            "--records-directory",
                            str(output_directory / "records"),
                            "--experiment-name",
                            dataset,
                            "--seed",
                            str(seed),
                            "--no-report",
                            *(["--gpus", "1"] if gpu else []),
                            *training_arguments,
                        ],
                    }
                )
    return runs


def _submit(args: Namespace) -> None:
    """
    Submits a SLURM batch job given the CLAs.

    Args
    ----
    args: The Command Line Arguments (CLAs).

    Raises
    ------
    ValueError if a data directory is missing, no presets match, or sbatch
    is not available.
    """
    missing: list[str] = [
        str(directory)
        for directory in args.data_directories
        if not directory.is_dir()
    ]
    if missing:
        raise ValueError(f"Data directories not found: {', '.join(missing)}")
    if not args.dry_run and shutil.which("sbatch") is None:
        raise ValueError("sbatch is not available; use --dry-run to inspect")

    with args.presets_file.open() as fd:
        preset_names: list[str] = list(json.load(fd)["presets"])
    training_arguments: list[str] = args.training_arguments
    if training_arguments[:1] == ["--"]:
        training_arguments = training_arguments[1:]
    runs: list[dict[str, Any]] = build_runs(
        select_presets(args.presets, preset_names),
        [directory.resolve() for directory in args.data_directories],
        args.seeds,
        args.output_directory.resolve(),
        training_arguments,
        gpu=args.resources == "gpu",
    )

    runs_directory: Path = args.output_directory / "runs"
    log_directory: Path = args.output_directory / "logs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    runs_filepath: Path = (
        runs_directory / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}.jsonl"
    ).resolve()
    with runs_filepath.open("w") as fd:
        for run in runs:
            fd.write(json.dumps(run) + "\n")

    limit: str = f"%{args.max_concurrent}" if args.max_concurrent else ""
    command: list[str] = [
        "sbatch",
        f"--array=0-{len(runs) - 1}{limit}",
        f"--output={log_directory.resolve()}/%x_%A_%a.out",
        f"--export=ALL,QENETICS_REPOSITORY={REPOSITORY}",
        *shlex.split(args.sbatch_args),
        str(TEMPLATES[args.resources]),
        str(runs_filepath),
    ]
    print(f"Wrote {len(runs)} runs to {runs_filepath}")
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


def _run(runs_filepath: Path, index: int) -> None:
    """
    Run one job on the local computer.

    Args
    ----
    runs_filepath: The filepath of a JSON file with details for one or more runs.
    index: The index in the JSON file of the run to choose.

    Raises
    ------
    IndexError if the run list has no run of that index.
    subprocess.CalledProcessError if training fails.
    """
    with runs_filepath.open() as fd:
        runs: list[dict[str, Any]] = [
            json.loads(line) for line in fd if line.strip()
        ]
    if not 0 <= index < len(runs):
        raise IndexError(
            f"Run {index} does not exist; {runs_filepath} holds {len(runs)}"
        )

    print(f"Run {index}: {runs[index]['name']}", flush=True)
    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *runs[index]["arguments"]],
        check=True,
    )


def main(arguments: list[str] | None = None) -> None:
    args: Namespace = _parse_script_args(arguments)
    if args.command == "submit":
        _submit(args)
    else:
        _run(args.runs_filepath, args.index)


if __name__ == "__main__":
    main()
