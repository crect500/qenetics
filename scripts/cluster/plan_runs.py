"""
Run the quantum training plan on a SLURM cluster.

The plan's experiments with the repository's presets, under its shared
protocol (early stopping on mean per-cell validation AUC, capped at 25
epochs; collapsed runs rerun with seed + 10):

- full: every preset on its full training sets, 3 seeds: the circuit models,
  the classical models of the report's matched comparisons
  (`classical_*/`), the ceilings (`ceiling/`) and the factorization
  machines.
- E4: the angle-matched real wide banks with the average pool, split A.
- E12: 4-fold chromosome cross-validation of Q_L320 and the classical
  references (DeepCpG's CNN, C_w4_170, C_w8_90 and U64) on every site, 1
  seed.
- E13: Q_L320 on the 2i and serum sets widened to every other autosome, 2
  seeds.
- Q1: the global circuits G1_lin_ry and G3_mlp_ry, split A.
- LR1e-2, LR3e-3, LR1e-3, LR3e-4: the factorization machines' learning-rate
  sweep on the full training sets of HCC and 2i. Their full runs wait for it
  and use the rate with the best mean validation score.
- splitC: the best quantum models and DeepCpG's CNN, with this protocol's and
  with DeepCpG's settings, on DeepCpG's own split.

`submit` plans these runs, and the token inputs the token presets read, as a
queue in the output directory. It then submits whole-node jobs, at most
--gpu-nodes on the GPU partition and --cpu-nodes on the CPU partition at a
time, and at most 16 nodes in all, the cluster's per-user limit. Their workers, one per GPU or --cpu-workers-per-node per CPU node, take
entries from the queue until none is left:

- GPU workers take the longest runs first. CPU workers write the token inputs,
  then take the shortest runs, if estimated to last at most --cpu-max-hours
  and to fit in their job's remaining time.
- Runs on token inputs wait until their inputs are written.
- Each finished entry leaves COMPLETED in its output directory, and `submit`
  skips those, so submitting again resumes after failures and time limits,
  and adds the reruns of runs that collapsed.

With --close-gaps, `submit` also adds seeds 3 and 4 to every quantum model
within 0.005 of a classical model it is compared with, and to that model.

`summarize` writes one row per recorded run, with its best epoch and test
AUCs, to summary.tsv in the output directory. `compare` runs the paired
bootstrap (`qenetics.tools.bootstrap`) of the matched comparisons on the
full training sets: every quantum model against the classical model of its
kind with the best mean validation score among those with at most 1.25 times
its trained parameters, counted on the first layer (the front-end table) and
on the whole model (the total table). It also compares the complex circuits
with their real twins (config/complex_real_pairs.json) and, in the other
experiments, every quantum model with each classical model beside it, and
writes every run's per-cell AUPRC and Matthews correlation, into the
comparisons folder.

    python scripts/cluster/plan_runs.py submit \\
        --data-directories /data/2i /data/hcc /data/hepg2 /data/ser \\
        /data/mESC --output-directory /scratch/qcpg
    python scripts/cluster/plan_runs.py status /scratch/qcpg
    python scripts/cluster/plan_runs.py summarize /scratch/qcpg
    python scripts/cluster/plan_runs.py compare /scratch/qcpg --processes 40
    python scripts/build_training_report.py -r /scratch/qcpg/full/records
"""

import getpass
import hashlib
import inspect
import json
import math
import os
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
import zlib
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from qenetics.qcpg import heads, model_factory, qcpg, records
from qenetics.tools import bootstrap, data, tokenization

REPOSITORY: Path = Path(__file__).resolve().parents[2]
CLUSTER_DIRECTORY: Path = Path(__file__).resolve().parent
TRAIN: str = "train"
TOKENIZE: str = "tokenize"
SCRIPTS: dict[str, Path] = {
    TRAIN: REPOSITORY / "scripts" / "train_quantum_model.py",
    TOKENIZE: REPOSITORY / "scripts" / "tokenize_cpg_windows.py",
}
PRESETS_FILEPATH: Path = REPOSITORY / "config" / "quantum_models.json"
COSTS_FILEPATH: Path = CLUSTER_DIRECTORY / "preset_costs.json"
GPU: str = "gpu"
CPU: str = "cpu"
TEMPLATES: dict[str, Path] = {
    GPU: CLUSTER_DIRECTORY / "pool_gpu.sbatch",
    CPU: CLUSTER_DIRECTORY / "pool_cpu.sbatch",
}
# The job names set by the templates, which find a campaign's jobs.
JOB_NAMES: dict[str, str] = {GPU: "qcpg-pool-gpu", CPU: "qcpg-pool-cpu"}
GPUS_PER_NODE: int = 4
# The cluster's per-user limits: at most this many nodes at once, CPU and GPU
# nodes together (its CPU quota allows 16 CPU nodes, or 12 beside 4 GPU
# nodes), and at most this many GPU nodes.
MAX_NODES: int = 16
MAX_GPU_NODES: int = 4
PAIRS_FILEPATH: Path = REPOSITORY / "config" / "complex_real_pairs.json"
# The sections of the classical models, and of the ceilings: the dense
# matrices the circuits' first layers can reach, which are references but not
# comparators of the matched comparisons.
CLASSICAL_PREFIX: str = "classical_"
CEILING_SECTION: str = "ceiling/"
CLASSICAL_REFERENCES: tuple[str, ...] = (
    "classical_tail/CNN",
    "classical_tail/C_w4_170",
    "classical_tail/C_w8_90",
    "ceiling/U64",
)
FACTORIZATION_MACHINES: tuple[str, ...] = (
    "classical_whole/FacM_r2",
    "classical_whole/FacM_r4",
    "classical_whole/FacM_r8",
)
# The plan tunes every new model family with these learning rates on the full
# training sets of these datasets, then trains it with the rate of the best
# mean validation score.
SWEEP_LEARNING_RATES: tuple[float, ...] = (1e-2, 3e-3, 1e-3, 3e-4)
SWEEP_DATASETS: frozenset[str] = frozenset({"hcc", "2i"})
SWEPT_PRESETS: tuple[str, ...] = FACTORIZATION_MACHINES
# --close-gaps adds these seeds to comparisons within this AUC.
CLOSE_GAP_SEEDS: tuple[int, ...] = (3, 4)
CLOSE_GAP_TOLERANCE: float = 0.005
COMPLETED_FILENAME: str = "COMPLETED"
ENTRIES_FILENAME: str = "entries.jsonl"
QUEUE_FILENAME: str = "queue.json"
CLAIM_FILENAME: str = "claim.json"
# Entries any worker may run, and entries for CPU workers only.
ANY_RESOURCES: str = "any"
CPU_RESOURCES: str = "cpu"
# CPU workers take a run only if this multiple of its estimate fits in their
# job's remaining time, as the estimates were measured on another computer.
CPU_TIME_SAFETY: float = 1.5
# The share of a worker's CPUs that load training data.
LOADER_SHARES: dict[str, float] = {GPU: 0.5, CPU: 0.2}
POLL_SECONDS: int = 120
SECONDS_PER_HOUR: int = 3600
# A run collapsed if its mean per-cell test AUC is this close to 0.5. It is
# rerun once, with its seed plus the offset.
COLLAPSE_TOLERANCE: float = 0.0002
REPLACEMENT_SEED_OFFSET: int = 10

# The chromosomes of each split, indexed by split name.
Chromosomes = dict[str, list[str]]
# The chromosome splits of a dataset, each with a label naming it among them
# (empty if there is one), given the dataset's lowercase name and autosomes.
SplitFunction = Callable[[str, list[str]], list[tuple[str, Chromosomes]]]

VALIDATION_CHROMOSOME: str = "13"
# Split A's training chromosomes, by lowercase dataset name.
SPLIT_A_TRAINING: dict[str, list[str]] = {
    "hepg2": ["1", "3", "5"],
    "hcc": ["1", "3", "5", "7", "9"],
    "mesc": ["1", "3", "5", "7", "9", "11"],
    "2i": ["1", "3", "5"],
    "ser": ["1", "3", "5"],
}
# The scBS datasets, which split A tests on chromosome 2 only.
SCBS_DATASETS: set[str] = {"2i", "ser"}
FOLD_QUANTITY: int = 4
# DeepCpG's own holdout, which validates on the other autosomes.
SPLIT_C_TRAINING: list[str] = ["1", "3", "5", "7", "9", "11"]
SPLIT_C_TEST: list[str] = ["2", "4", "6", "8", "10", "12"]


def _split(
    training: list[str], validation: list[str], test: list[str]
) -> Chromosomes:
    return {
        records.TRAINING_SPLIT: training,
        records.VALIDATION_SPLIT: validation,
        records.TEST_SPLIT: test,
    }


def _test_chromosomes(dataset: str) -> list[str]:
    """Split A's test chromosomes of a dataset."""
    return ["2"] if dataset in SCBS_DATASETS else ["2", "4"]


def split_a(
    dataset: str, autosomes: list[str]
) -> list[tuple[str, Chromosomes]]:
    """The report's main split, split A."""
    return [
        (
            "",
            _split(
                SPLIT_A_TRAINING[dataset],
                [VALIDATION_CHROMOSOME],
                _test_chromosomes(dataset),
            ),
        )
    ]


def split_c(
    dataset: str, autosomes: list[str]
) -> list[tuple[str, Chromosomes]]:
    """
    DeepCpG's holdout: training on chromosomes 1, 3, 5, 7, 9 and 11, testing
    on 2, 4, 6, 8, 10 and 12, and validating on the other autosomes.
    """
    held_out: set[str] = {*SPLIT_C_TRAINING, *SPLIT_C_TEST}
    return [
        (
            "",
            _split(
                SPLIT_C_TRAINING,
                [c for c in autosomes if c not in held_out],
                SPLIT_C_TEST,
            ),
        )
    ]


def full_training_sets(
    dataset: str, autosomes: list[str]
) -> list[tuple[str, Chromosomes]]:
    """
    The report's full training sets: split A, training on every other
    autosome for HepG2, HCC and mESC, and on chromosomes 1, 3 and 5 for 2i
    and serum.
    """
    test: list[str] = _test_chromosomes(dataset)
    training: list[str] = (
        SPLIT_A_TRAINING[dataset]
        if dataset in SCBS_DATASETS
        else [
            chromosome
            for chromosome in autosomes
            if chromosome not in {VALIDATION_CHROMOSOME, *test}
        ]
    )
    return [("", _split(training, [VALIDATION_CHROMOSOME], test))]


def sweep_training_sets(
    dataset: str, autosomes: list[str]
) -> list[tuple[str, Chromosomes]]:
    """The full training sets of the learning-rate sweep's datasets only."""
    return (
        full_training_sets(dataset, autosomes)
        if dataset in SWEEP_DATASETS
        else []
    )


def widened_training_sets(
    dataset: str, autosomes: list[str]
) -> list[tuple[str, Chromosomes]]:
    """Split A of 2i and serum, training on every other autosome (E13)."""
    if dataset not in SCBS_DATASETS:
        return []

    test: list[str] = _test_chromosomes(dataset)
    training: list[str] = [
        chromosome
        for chromosome in autosomes
        if chromosome not in {VALIDATION_CHROMOSOME, *test}
    ]
    return [("", _split(training, [VALIDATION_CHROMOSOME], test))]


def chromosome_folds(
    dataset: str, autosomes: list[str]
) -> list[tuple[str, Chromosomes]]:
    """
    Chromosome cross-validation (E12): the autosomes but chromosome 13,
    dealt in order into `FOLD_QUANTITY` folds, each fold tested on after
    training on the others. Chromosome 13 validates every fold.
    """
    pool: list[str] = [
        chromosome
        for chromosome in autosomes
        if chromosome != VALIDATION_CHROMOSOME
    ]
    splits: list[tuple[str, Chromosomes]] = []
    for fold_index in range(FOLD_QUANTITY):
        fold: list[str] = pool[fold_index::FOLD_QUANTITY]
        splits.append(
            (
                f"fold{fold_index}",
                _split(
                    [
                        chromosome
                        for chromosome in pool
                        if chromosome not in fold
                    ],
                    [VALIDATION_CHROMOSOME],
                    fold,
                ),
            )
        )
    return splits


@dataclass(frozen=True)
class Experiment:
    """
    A group of runs of the training plan.

    Attributes
    ----------
    description: What its runs are.
    presets: The presets it trains, as names or shell-style patterns.
    seeds: Its run seeds.
    splits: The chromosome splits of each dataset.
    all_sites: Train on every training site, rather than the preset's
        subset.
    deduplicated: Leave out the presets of --exclude and those that build
        the same model on the same input as an earlier preset.
    learning_rate: The learning rate of every run, instead of the preset's,
        for a learning-rate sweep.
    """

    description: str
    presets: tuple[str, ...]
    seeds: tuple[int, ...]
    splits: SplitFunction
    all_sites: bool
    deduplicated: bool = False
    learning_rate: float | None = None


def _sweep_name(learning_rate: float) -> str:
    return f"LR{learning_rate:.0e}".replace("e-0", "e-")


# The experiments of the training plan that the presets cover. Its QFT,
# attention and Gray-code models (Q2-Q4) are not presets.
EXPERIMENTS: dict[str, Experiment] = {
    "full": Experiment(
        "every preset on the full training sets",
        ("*",),
        (0, 1, 2),
        full_training_sets,
        all_sites=True,
        deduplicated=True,
    ),
    "E4": Experiment(
        "the angle-matched real wide banks with the average pool, split A",
        ("deepcpg_tail/R_w32_L384_avg", "deepcpg_tail/R_w64_L384_avg"),
        (0, 1, 2),
        split_a,
        all_sites=False,
    ),
    "E12": Experiment(
        "4-fold chromosome cross-validation of the best quantum model and the "
        "classical references",
        ("deepcpg_tail/Q_L320", *CLASSICAL_REFERENCES),
        (0,),
        chromosome_folds,
        all_sites=True,
    ),
    "E13": Experiment(
        "the best quantum model on 2i and serum, every other autosome",
        ("deepcpg_tail/Q_L320",),
        (0, 1),
        widened_training_sets,
        all_sites=True,
    ),
    "Q1": Experiment(
        "the real global circuits, split A",
        ("whole_model/G1_lin_ry", "whole_model/G3_mlp_ry"),
        (0, 1, 2),
        split_a,
        all_sites=False,
    ),
    **{
        _sweep_name(rate): Experiment(
            f"the factorization machines' learning-rate sweep at {rate:g}",
            SWEPT_PRESETS,
            (0,),
            sweep_training_sets,
            all_sites=True,
            learning_rate=rate,
        )
        for rate in SWEEP_LEARNING_RATES
    },
    "splitC": Experiment(
        "the best quantum models and DeepCpG's CNN on DeepCpG's split",
        (
            "deepcpg_tail/Q_L128",
            "deepcpg_tail/Q_L320",
            "whole_model/H1b_cnnq",
            "classical_tail/CNN",
            "classical_tail/CNN_deepcpg",
        ),
        (0, 1, 2),
        split_c,
        all_sites=True,
    ),
}
SWEEP_EXPERIMENTS: tuple[str, ...] = tuple(
    _sweep_name(rate) for rate in SWEEP_LEARNING_RATES
)


@dataclass
class Dataset:
    """
    A dataset's chromosome files.

    Attributes
    ----------
    name: The dataset's name, that of its directory.
    directory: The directory of its chr<name>.h5 files.
    autosomes: The numbered chromosomes it has files of, in order.
    site_quantities: The sites of each chromosome counted so far.
    """

    name: str
    directory: Path
    autosomes: list[str]
    site_quantities: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """The dataset's lowercase name, as the split tables use it."""
        return self.name.lower()

    def sites(self, chromosomes: list[str]) -> int:
        """Count the sites of chromosomes."""
        for chromosome in chromosomes:
            if chromosome not in self.site_quantities:
                self.site_quantities[chromosome] = data.read_h5_sample_quantity(
                    self.directory / f"chr{chromosome}.h5"
                )
        return sum(
            self.site_quantities[chromosome] for chromosome in chromosomes
        )


@dataclass
class QueueState:
    """
    The progress of a queue's entries.

    Attributes
    ----------
    claimed: The entries a worker has taken.
    finished: The return code of each finished entry, by entry.
    """

    claimed: set[int]
    finished: dict[int, int]


def _parse_script_args(arguments: list[str] | None = None) -> Namespace:
    parser = ArgumentParser("plan_runs", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser(
        "submit",
        help="Queue the plan's runs that have not completed and submit node "
        "jobs that train them.",
    )
    submit.add_argument(
        "--data-directories",
        type=Path,
        nargs="+",
        required=True,
        help="Dataset directories, each named after its dataset: "
        f"{', '.join(SPLIT_A_TRAINING)} (any case).",
    )
    submit.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="The root of run outputs, records, token inputs, queues and logs.",
    )
    submit.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
        help="The experiments to run: "
        + "; ".join(
            f"{name}, {experiment.description}"
            for name, experiment in EXPERIMENTS.items()
        )
        + ".",
    )
    submit.add_argument(
        "--presets",
        nargs="+",
        default=["*"],
        help="Run only the experiments' presets matching these names or "
        "shell-style patterns.",
    )
    submit.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Presets or patterns the full experiment leaves out.",
    )
    submit.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Also run, in the full experiment, presets that build the same "
        "model on the same input as an earlier preset, such as series_1/QA1 "
        "(deepcpg_tail/Q_L4).",
    )
    submit.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="The run seeds of every experiment, instead of its own.",
    )
    submit.add_argument(
        "--max-epochs",
        type=int,
        default=25,
        help="The epoch limit of every run.",
    )
    submit.add_argument(
        "--minimum-patience",
        type=int,
        default=3,
        help="Stop after this many epochs without validation gain, or the "
        "preset's patience if more.",
    )
    submit.add_argument(
        "--close-gaps",
        action="store_true",
        help=f"Add seeds {', '.join(map(str, CLOSE_GAP_SEEDS))} to every "
        "quantum model within --close-gap-tolerance of its matched classical "
        "comparators on the full training sets, or of the best classical "
        "model of its experiment, cell type and fold elsewhere, and to those "
        "models.",
    )
    submit.add_argument(
        "--close-gap-tolerance",
        type=float,
        default=CLOSE_GAP_TOLERANCE,
        help="The AUC gap below which --close-gaps adds seeds.",
    )
    submit.add_argument("--gpu-partition", default="gpu")
    submit.add_argument("--cpu-partition", default="main")
    submit.add_argument(
        "--gpu-nodes",
        type=int,
        default=MAX_GPU_NODES,
        help=f"The most GPU nodes ({GPUS_PER_NODE} GPUs each) to use at once, "
        f"at most {MAX_GPU_NODES}.",
    )
    submit.add_argument(
        "--cpu-nodes",
        type=int,
        help="The most CPU nodes to use at once. Defaults to the rest of the "
        f"{MAX_NODES}-node limit beside the GPU nodes used: {MAX_NODES} without "
        f"GPU nodes, {MAX_NODES - MAX_GPU_NODES} beside {MAX_GPU_NODES}.",
    )
    submit.add_argument(
        "--cpu-workers-per-node",
        type=int,
        default=4,
        help="The runs at once on each CPU node, which share its CPUs.",
    )
    submit.add_argument(
        "--gpu-time",
        default="7-00:00:00",
        help="The time limit of GPU node jobs, or the partition's if shorter.",
    )
    submit.add_argument(
        "--cpu-time",
        default="7-00:00:00",
        help="The time limit of CPU node jobs, or the partition's if shorter.",
    )
    submit.add_argument(
        "--cpu-max-hours",
        type=float,
        default=48.0,
        help="CPU workers take runs estimated to last at most this long.",
    )
    submit.add_argument(
        "--sbatch-args",
        default="",
        help="Further sbatch options of every node job, e.g. '--account=x'.",
    )
    submit.add_argument("--presets-file", type=Path, default=PRESETS_FILEPATH)
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the queue and print the sbatch commands only.",
    )
    status = commands.add_parser(
        "status", help="Summarize the progress of a queue."
    )
    status.add_argument(
        "output_directory", type=Path, help="The output directory of submit."
    )
    status.add_argument(
        "--queue",
        help="The queue, a folder of <output directory>/queues. Defaults to "
        "the latest.",
    )
    summarize = commands.add_parser(
        "summarize",
        help="Write one row per recorded run, with its test AUCs, as "
        "tab-separated values.",
    )
    summarize.add_argument(
        "output_directory", type=Path, help="The output directory of submit."
    )
    summarize.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"The file to write. Defaults to {SUMMARY_FILENAME} in the "
        "output directory.",
    )
    compare = commands.add_parser(
        "compare",
        help="Bootstrap the matched comparisons of the full training sets, "
        "the complex and real twins, and the other experiments' quantum and "
        "classical models, and compute every run's AUPRC and MCC.",
    )
    compare.add_argument(
        "output_directory", type=Path, help="The output directory of submit."
    )
    compare.add_argument(
        "--replicates",
        type=int,
        default=1000,
        help="The bootstrap replicates.",
    )
    compare.add_argument(
        "--sites",
        type=int,
        default=200_000,
        help="The most test sites the bootstrap resamples: a fixed random "
        "sample of each test set, the same for every model on it.",
    )
    compare.add_argument(
        "--processes", type=int, default=1, help="The processes to use."
    )
    compare.add_argument(
        "--pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("A", "B"),
        help="Also compare preset A with preset B wherever both ran on the "
        "same test set. Repeat for more.",
    )
    compare.add_argument(
        "--presets-file",
        type=Path,
        default=PRESETS_FILEPATH,
        help="The presets file, whose models the matched comparisons measure.",
    )
    compare.add_argument(
        "--pairs-file",
        type=Path,
        default=PAIRS_FILEPATH,
        help="The complex circuit models and their real twins to compare.",
    )
    work = commands.add_parser(
        "work",
        help="Run entries of a queue until none is left; started by the node "
        "jobs.",
    )
    work.add_argument("queue_directory", type=Path)
    work.add_argument("--resources", choices=[GPU, CPU], required=True)
    work.add_argument(
        "--cpus", type=int, required=True, help="The worker's CPUs."
    )
    work.add_argument(
        "--gpu", help="The worker's GPU, as an entry of CUDA_VISIBLE_DEVICES."
    )
    args: Namespace = parser.parse_args(arguments)
    if args.command == "submit" and (
        min(args.gpu_nodes, args.cpu_nodes or 0) < 0
        or args.cpu_workers_per_node < 1
    ):
        parser.error(
            "node quantities must not be negative, and CPU nodes need workers"
        )
    if args.command == "submit" and (
        args.gpu_nodes > MAX_GPU_NODES
        or args.gpu_nodes + (args.cpu_nodes or 0) > MAX_NODES
    ):
        parser.error(
            f"the cluster allows at most {MAX_NODES} nodes at once, at most "
            f"{MAX_GPU_NODES} of them GPU nodes"
        )
    if args.command == "work" and (args.resources == GPU) != (
        args.gpu is not None
    ):
        parser.error("GPU workers, and only they, need --gpu")
    return args


def slurm_hours(text: str) -> float:
    """
    Convert a SLURM time to hours.

    Args
    ----
    text: A time as sbatch, sinfo or squeue write it, e.g. '2-12:00:00',
        '30:00' (minutes and seconds) or 'infinite'.

    Returns
    -------
    The hours, infinite for no limit.

    Raises
    ------
    ValueError if the text is not a SLURM time.
    """
    text = text.strip()
    if text.lower() in {"infinite", "unlimited"}:
        return math.inf

    days, separator, clock = text.partition("-")
    if not separator:
        days, clock = "0", text
    try:
        day_quantity: int = int(days)
        parts: list[int] = [int(part) for part in clock.split(":")]
    except ValueError as error:
        raise ValueError(f"{text} is not a SLURM time") from error

    if separator and len(parts) <= 3:
        hours, minutes, seconds = parts + [0] * (3 - len(parts))
    elif len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0
    elif len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"{text} is not a SLURM time")
    return day_quantity * 24 + hours + minutes / 60 + seconds / SECONDS_PER_HOUR


def _slurm_output(command: list[str]) -> str | None:
    """
    Run a SLURM query.

    Args
    ----
    command: The query, e.g. an squeue command.

    Returns
    -------
    Its output, or None if SLURM is not available or the query fails.
    """
    if shutil.which(command[0]) is None:
        return None

    result = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def _active_job_ids() -> set[str] | None:
    """The ids of the user's queued and running jobs, if SLURM can tell."""
    output: str | None = _slurm_output(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%A"]
    )
    return None if output is None else set(output.split())


def _remaining_hours() -> float:
    """The hours left to this SLURM job, infinite outside one."""
    job: str | None = os.environ.get("SLURM_JOB_ID")
    output: str | None = (
        None
        if job is None
        else _slurm_output(["squeue", "-h", "-j", job, "-o", "%L"])
    )
    try:
        return slurm_hours(output) if output else math.inf
    except ValueError:
        return math.inf


def _time_limit(requested: str, partition: str) -> str:
    """
    Choose the time limit of a partition's jobs.

    Args
    ----
    requested: The requested time limit.
    partition: The partition.

    Returns
    -------
    The requested time limit, or the partition's if that is shorter.
    """
    output: str | None = _slurm_output(
        ["sinfo", "-h", "-p", partition, "-o", "%l"]
    )
    limit: str = output.split()[0] if output and output.split() else ""
    try:
        is_shorter: bool = bool(limit) and slurm_hours(limit) < slurm_hours(
            requested
        )
    except ValueError:
        is_shorter = False
    if is_shorter:
        print(f"Partition {partition} allows jobs of at most {limit}")
        return limit
    return requested


def model_identity(preset: model_factory.Preset) -> str:
    """
    Identify the model a preset builds.

    Args
    ----
    preset: The preset.

    Returns
    -------
    Its family, input and options, with the family's defaults for options
    the preset does not set.
    """
    parameters = inspect.signature(
        model_factory.MODEL_FAMILIES[preset.model].__init__
    ).parameters
    options: dict[str, Any] = {
        name: parameters[name].default
        for name in model_factory.model_option_names(preset.model)
    }
    options.update(preset.options)
    return json.dumps(
        [preset.model, preset.input, options], sort_keys=True, default=str
    )


def choose_presets(
    presets: dict[str, model_factory.Preset],
    patterns: Iterable[str],
    exclusions: Iterable[str],
    *,
    keep_duplicates: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """
    Choose the presets to run, in the presets file's order.

    Args
    ----
    presets: The presets, indexed by name.
    patterns: Preset names or shell-style patterns to run.
    exclusions: Preset names or patterns to leave out.
    keep_duplicates: Keep presets that build the same model on the same
        input as an earlier preset.

    Returns
    -------
    The chosen names, and each other matching name with why it is left out.

    Raises
    ------
    ValueError if a pattern matches no preset.
    """
    patterns, exclusions = list(patterns), list(exclusions)
    unmatched: list[str] = [
        pattern
        for pattern in patterns
        if not any(fnmatchcase(name, pattern) for name in presets)
    ]
    if unmatched:
        raise ValueError(f"No presets match {', '.join(unmatched)}")

    chosen: list[str] = []
    skipped: dict[str, str] = {}
    first_names: dict[str, str] = {}
    for name, preset in presets.items():
        identity: str = model_identity(preset)
        first_name: str = first_names.setdefault(identity, name)
        if not any(fnmatchcase(name, pattern) for pattern in patterns):
            continue
        if any(fnmatchcase(name, pattern) for pattern in exclusions):
            skipped[name] = "excluded"
        elif first_name != name and not keep_duplicates:
            skipped[name] = f"the same model as {first_name}"
        else:
            chosen.append(name)
    return chosen, skipped


def run_patience(preset: model_factory.Preset, minimum: int) -> int:
    """The patience of a preset's runs."""
    return max(preset.training.get("patience") or 0, minimum)


def load_costs(filepath: Path, names: Iterable[str]) -> dict[str, float]:
    """
    Read the costs of presets' training steps.

    Args
    ----
    filepath: The costs file.
    names: The presets that need a cost.

    Returns
    -------
    The microseconds per training site of each preset, the median of the
    file's for presets it lacks.
    """
    with filepath.open() as fd:
        costs: dict[str, float] = json.load(fd)["microseconds_per_site"]
    names = list(dict.fromkeys(names))
    missing: list[str] = [name for name in names if name not in costs]
    if missing:
        print(
            f"No costs in {filepath} for {', '.join(missing)}; using the "
            "median cost"
        )
    median: float = statistics.median(costs.values())
    return {name: costs.get(name, median) for name in names}


def load_dataset(data_directory: Path) -> Dataset:
    """
    Find a dataset's autosome files.

    Args
    ----
    data_directory: The directory of a dataset's chr<name>.h5 files, named
        after the dataset.

    Returns
    -------
    The dataset.

    Raises
    ------
    ValueError if the plan has no splits of the dataset.
    """
    if data_directory.name.lower() not in SPLIT_A_TRAINING:
        raise ValueError(
            f"The plan has no splits of {data_directory.name}; its datasets "
            f"are {', '.join(SPLIT_A_TRAINING)}"
        )

    return Dataset(
        name=data_directory.name,
        directory=data_directory,
        autosomes=sorted(
            (
                chromosome
                for chromosome in map(
                    data.chromosome_name, data_directory.glob("chr*.h5")
                )
                if chromosome.isdigit()
            ),
            key=int,
        ),
    )


def check_chromosome_files(dataset: Dataset, chromosomes: Chromosomes) -> None:
    """
    Check that a split's chromosome files exist.

    Raises
    ------
    ValueError if the split has no training chromosomes.
    FileNotFoundError if a chromosome's file is missing.
    """
    if not chromosomes[records.TRAINING_SPLIT]:
        raise ValueError(f"No training chromosome files in {dataset.directory}")

    missing: list[str] = [
        str(dataset.directory / f"chr{chromosome}.h5")
        for split_chromosomes in chromosomes.values()
        for chromosome in split_chromosomes
        if not (dataset.directory / f"chr{chromosome}.h5").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Chromosome files not found: {', '.join(missing)}"
        )


def parse_input(name: str) -> tuple[str, int]:
    """
    Find the tokenization of a token input.

    Args
    ----
    name: The input's name, e.g. 'cpg_kmer3' or 'painted_bases64'.

    Returns
    -------
    The tokenization, and its k-mer length or vocabulary size.

    Raises
    ------
    ValueError if the input is not a tokenization's.
    """
    for method in sorted(tokenization.TOKENIZATIONS, key=len, reverse=True):
        size: str = name.removeprefix(method)
        if size != name and size.isdigit():
            return method, int(size)

    raise ValueError(f"Input {name} is not that of a tokenization")


def _token_directory(output_directory: Path, dataset: str, name: str) -> Path:
    return output_directory / "tokens" / dataset / name


def tokenization_entries(
    dataset: Dataset, input_names: set[str], output_directory: Path
) -> list[dict[str, Any]]:
    """
    Plan the entries that write a dataset's token inputs.

    BPE vocabularies are learned on the training chromosomes of the full
    training sets, and painted inputs reuse the one of their size.

    Args
    ----
    dataset: The dataset.
    input_names: The token inputs runs read.
    output_directory: The root of the outputs.

    Returns
    -------
    The entries, BPE inputs first.
    """
    needed: set[str] = set(input_names)
    for name in input_names:
        method, size = parse_input(name)
        if method in (tokenization.PAINTED, tokenization.PAINTED_BASES):
            needed.add(tokenization.input_name(tokenization.CPG_BPE, size))

    entries: list[dict[str, Any]] = []
    for name in sorted(needed):
        method, size = parse_input(name)
        directory: Path = _token_directory(output_directory, dataset.name, name)
        arguments: list[str] = [
            "--input-directory",
            str(dataset.directory),
            "--output-directory",
            str(directory),
            "--method",
            method,
        ]
        requires: list[str] = []
        if method == tokenization.CPG_KMER:
            arguments += ["--kmer-length", str(size)]
        elif method == tokenization.CPG_BPE:
            _, chromosomes = full_training_sets(dataset.key, dataset.autosomes)[
                0
            ]
            arguments += [
                "--vocabulary-size",
                str(size),
                "--training-chromosomes",
                *chromosomes[records.TRAINING_SPLIT],
            ]
        else:
            bpe_directory: Path = _token_directory(
                output_directory,
                dataset.name,
                tokenization.input_name(tokenization.CPG_BPE, size),
            )
            arguments += [
                "--vocabulary-size",
                str(size),
                "--tokenizer",
                str(bpe_directory / "tokenizer.json"),
            ]
            requires.append(str(bpe_directory / COMPLETED_FILENAME))
        entries.append(
            {
                "name": f"tokens/{dataset.name}/{name}",
                "script": TOKENIZE,
                "arguments": arguments,
                "marker": str(directory / COMPLETED_FILENAME),
                "requires": requires,
                "resources": CPU_RESOURCES,
                "hours": 0.0,
            }
        )
    return entries


def collapsed_runs(records_directories: Iterable[Path]) -> set[str]:
    """
    Find the completed runs whose output collapsed to a constant.

    Only the latest completed record of each output directory counts.

    Args
    ----
    records_directories: The records directories to search.

    Returns
    -------
    The output directories of runs whose mean per-cell test AUC is within
    `COLLAPSE_TOLERANCE` of 0.5.
    """
    test_aucs: dict[str, float] = {}
    for records_directory in records_directories:
        if not records_directory.is_dir():
            continue
        for run_directory in records.find_run_directories(records_directory):
            run: dict[str, Any] = records.read_run(run_directory)
            auc: float = _test_mean_cell_auc(run)
            if run["status"] == records.STATUS_COMPLETED and math.isfinite(auc):
                test_aucs[str(run["parameters"]["output_directory"])] = auc
    return {
        directory for directory, auc in test_aucs.items() if _is_collapsed(auc)
    }


def mean_cell_auc(aucs: Iterable[float]) -> float:
    """The mean of cells' AUCs over the defined ones; NaN if none is."""
    defined: list[float] = [
        auc
        for auc in aucs
        if isinstance(auc, int | float) and math.isfinite(auc)
    ]
    return statistics.fmean(defined) if defined else math.nan


def _test_mean_cell_auc(run: dict[str, Any]) -> float:
    """The mean per-cell test AUC of a run read by `records.read_run`."""
    return mean_cell_auc(
        cell["auc"]
        for cell in run["metrics"].get(records.TEST_SPLIT, {}).get("cells", [])
    )


def _is_collapsed(test_mean_cell_auc: float) -> bool:
    return abs(test_mean_cell_auc - 0.5) <= COLLAPSE_TOLERANCE


def _training_entry(
    name: str,
    preset: model_factory.Preset,
    data_directory: Path,
    run_directory: Path,
    records_directory: Path,
    dataset: Dataset,
    chromosomes: Chromosomes,
    seed: int,
    training_arguments: list[str],
    hours: float,
) -> dict[str, Any]:
    is_tokens: bool = preset.input != model_factory.BASES_STR
    return {
        "name": name,
        "script": TRAIN,
        "arguments": [
            "--preset",
            preset.name,
            "--data-directory",
            str(data_directory),
            "--output-directory",
            str(run_directory),
            "--records-directory",
            str(records_directory),
            "--experiment-name",
            dataset.name,
            "--seed",
            str(seed),
            "--no-report",
            "--training-chromosomes",
            *chromosomes[records.TRAINING_SPLIT],
            "--validation-chromosomes",
            *chromosomes[records.VALIDATION_SPLIT],
            "--test-chromosomes",
            *chromosomes[records.TEST_SPLIT],
            *training_arguments,
        ],
        "marker": str(run_directory / COMPLETED_FILENAME),
        "requires": (
            [str(data_directory / COMPLETED_FILENAME)] if is_tokens else []
        ),
        "resources": ANY_RESOURCES,
        "hours": hours,
    }


def is_swept(name: str) -> bool:
    """Whether a preset takes its learning rate from the sweep."""
    return any(fnmatchcase(name, pattern) for pattern in SWEPT_PRESETS)


def swept_learning_rates(output_directory: Path) -> dict[str, float]:
    """
    Choose the learning rate of each swept preset.

    Args
    ----
    output_directory: The output directory of `submit`.

    Returns
    -------
    For each preset with a completed sweep run at every rate on every sweep
    dataset, the rate with the best mean validation score over the datasets.
    """
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in summarize_runs(output_directory):
        if (
            row["experiment"] in SWEEP_EXPERIMENTS
            and row["status"] == records.STATUS_COMPLETED
        ):
            score: float = row["best_validation_score"]
            scores[row["preset"]][row["experiment"]][row["dataset"].lower()] = (
                score if math.isfinite(score) else -math.inf
            )
    rates: dict[str, float] = {}
    for preset, by_sweep in scores.items():
        means: dict[str, float] = {
            name: statistics.fmean(by_dataset[d] for d in SWEEP_DATASETS)
            for name, by_dataset in by_sweep.items()
            if SWEEP_DATASETS <= set(by_dataset)
        }
        if len(means) == len(SWEEP_EXPERIMENTS):
            rate: float | None = EXPERIMENTS[
                max(means, key=means.get)
            ].learning_rate
            if rate is not None:
                rates[preset] = rate
    return rates


def is_reference(name: str) -> bool:
    """Whether a preset is a classical model or a ceiling, without a circuit."""
    return name.startswith((CLASSICAL_PREFIX, CEILING_SECTION))


@dataclass(frozen=True)
class ModelSize:
    """
    A model's trained parameters, as the matched comparisons count them.

    Attributes
    ----------
    front_end: Those of its first layer, or of the whole model if it has
        none (`model_factory.model_sizes`).
    total: Those of the whole model.
    kind: `TAIL_KIND`, `POOLING_KIND` or `WHOLE_KIND`: a first layer with
        DeepCpG's tail (or another tail), one with the pooling head, or a
        whole model.
    input: Its input, `model_factory.BASES_STR` or a token input.
    """

    front_end: int
    total: int
    kind: str
    input: str


TAIL_KIND: str = "tail"
POOLING_KIND: str = "pooling"
WHOLE_KIND: str = "whole"
# A matched comparator has at most this multiple of the quantum model's
# trained parameters; a quantum lead counts only if a classical model of at
# least the smaller multiple, a similar size, was among the candidates.
MATCHED_SIZE_RATIO: float = 1.25
SIMILAR_SIZE_RATIO: float = 0.8
FRONT_END_TABLE: str = "front-end"
TOTAL_TABLE: str = "total"
# The experiments whose models are compared in matched pairs; the others
# compare every quantum model with each classical model beside it.
MATCHED_EXPERIMENTS: frozenset[str] = frozenset({"full"})


def preset_size(preset: model_factory.Preset, record: Path) -> ModelSize:
    """
    Measure a preset's model as one of its runs built it.

    Args
    ----
    preset: The preset.
    record: The record of one of its runs, whose data it builds the model
        for: the number of cells, and the token lookup of token inputs.

    Returns
    -------
    The model's size.

    Raises
    ------
    OSError or KeyError if the run's data cannot be read; ValueError if the
    model cannot be built.
    """
    parameters: dict[str, Any] = records.read_run(record)["parameters"]
    sequence_length, cell_quantity, token_attributes = data.read_h5_layout(
        Path(parameters["data_directory"])
        / f"chr{parameters['training_chromosomes'][0]}.h5"
    )
    encoding_options: dict[str, Any] = {}
    if token_attributes:
        encoding_options = {
            "encoding": data.TOKEN_ENCODING_STR,
            "token_lookup": tokenization.token_lookup(
                str(token_attributes[data.TOKENIZATION_ATTRIBUTE]),
                int(token_attributes[data.VOCABULARY_SIZE_ATTRIBUTE]),
            ),
        }
    model = model_factory.build_model(
        preset.model,
        sequence_length,
        cell_quantity,
        options=preset.options,
        **encoding_options,
    )
    front_end, total = model_factory.model_sizes(model)
    kind: str = (
        WHOLE_KIND
        if not any(
            hasattr(model, module) for module in model_factory.FRONT_END_MODULES
        )
        else POOLING_KIND
        if isinstance(getattr(model, "tail", None), heads.PoolingHead)
        else TAIL_KIND
    )
    return ModelSize(front_end, total, kind, preset.input)


def measure_sizes(
    rows: list[dict[str, Any]], presets: dict[str, model_factory.Preset]
) -> dict[tuple[str, str], ModelSize]:
    """
    Measure the model of every preset run on each dataset.

    Args
    ----
    rows: Rows of `summarize_runs`.
    presets: The presets, indexed by name.

    Returns
    -------
    The size of each (preset, dataset) model that could be built; the others
    are printed.
    """
    records_of: dict[tuple[str, str], str] = {}
    for row in rows:
        records_of.setdefault((row["preset"], row["dataset"]), row["record"])
    sizes: dict[tuple[str, str], ModelSize] = {}
    for (name, dataset), record in sorted(records_of.items()):
        if name not in presets:
            print(f"Not measuring {name}: not in the presets file")
            continue
        try:
            sizes[(name, dataset)] = preset_size(presets[name], Path(record))
        except (OSError, KeyError, IndexError, ValueError) as error:
            print(f"Not measuring {name} on {dataset}: {error}")
    return sizes


@dataclass(frozen=True)
class MatchedPair:
    """
    A quantum model and its matched classical comparator.

    Attributes
    ----------
    table: `FRONT_END_TABLE` or `TOTAL_TABLE`.
    quantum, classical: The presets.
    quantum_size, classical_size: Their trained parameters as the table
        counts them.
    scored: Whether a classical candidate of a similar size existed, or the
        quantum model is not ahead; otherwise its lead is not scored.
    """

    table: str
    quantum: str
    classical: str
    quantum_size: int
    classical_size: int
    scored: bool


def matched_pairs(
    means: dict[str, tuple[float, float]], sizes: dict[str, ModelSize]
) -> list[MatchedPair]:
    """
    Match the quantum models of one experiment, cell type and fold with
    classical models.

    Each quantum model's candidates are the classical models (not the
    ceilings) on its input and of its kind, and every classical model on its
    input for whole models. The front-end table counts the trained parameters
    of first layers, and of whole models for whole quantum models; the total
    table counts those of whole models, for the quantum first layers. Among
    the candidates with at most `MATCHED_SIZE_RATIO` times the quantum
    model's parameters, smaller ones included, its comparator has the best
    mean validation score.

    Args
    ----
    means: The mean validation score and test AUC of each preset.
    sizes: The size of each preset's model.

    Returns
    -------
    The pairs, by table and quantum model.
    """
    pairs: list[MatchedPair] = []
    for table in (FRONT_END_TABLE, TOTAL_TABLE):
        for quantum in sorted(means):
            size: ModelSize | None = sizes.get(quantum)
            if is_reference(quantum) or size is None:
                continue
            if table == TOTAL_TABLE and size.kind == WHOLE_KIND:
                continue

            def count(name: str, *, by_total: bool) -> int:
                return sizes[name].total if by_total else sizes[name].front_end

            by_total: bool = table == TOTAL_TABLE or size.kind == WHOLE_KIND
            quantity: int = count(quantum, by_total=by_total)
            candidates: list[str] = [
                name
                for name in means
                if name.startswith(CLASSICAL_PREFIX)
                and name in sizes
                and sizes[name].input == size.input
                and (size.kind == WHOLE_KIND or sizes[name].kind == size.kind)
                and count(name, by_total=by_total)
                <= MATCHED_SIZE_RATIO * quantity
            ]
            if not candidates:
                continue
            best: str = max(
                candidates,
                key=lambda name: (
                    means[name][0] if math.isfinite(means[name][0]) else -1.0,
                    name,
                ),
            )
            pairs.append(
                MatchedPair(
                    table,
                    quantum,
                    best,
                    quantity,
                    count(best, by_total=by_total),
                    any(
                        count(name, by_total=by_total)
                        >= SIMILAR_SIZE_RATIO * quantity
                        for name in candidates
                    )
                    or means[quantum][1] <= means[best][1],
                )
            )
    return pairs


def _usable(row: dict[str, Any]) -> bool:
    """Whether a run of `summarize_runs` completed, did not collapse and is
    not part of the learning-rate sweep."""
    return (
        row["status"] == records.STATUS_COMPLETED
        and not row["collapsed"]
        and math.isfinite(row["test_mean_cell_auc"])
        and row["experiment"] not in SWEEP_EXPERIMENTS
    )


def _means(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """The mean validation score (over defined scores) and test AUC of runs."""
    validation: list[float] = [
        row["best_validation_score"]
        for row in rows
        if math.isfinite(row["best_validation_score"])
    ]
    return (
        statistics.fmean(validation) if validation else math.nan,
        statistics.fmean(row["test_mean_cell_auc"] for row in rows),
    )


def comparison_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]]:
    """The usable runs of each experiment, cell type and fold, by preset."""
    groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        if _usable(row):
            key = (row["experiment"], row["dataset"], row["fold"])
            groups[key][row["preset"]].append(row)
    return groups


def close_gap_targets(
    output_directory: Path, tolerance: float, presets_filepath: Path
) -> dict[str, set[tuple[str, str, str]]]:
    """
    Find the comparisons too close to call with the seeds run so far.

    A quantum model whose mean test AUC is at least a classical model's minus
    `tolerance` is a target, and so is that classical model: its matched
    comparators (`matched_pairs`) in `MATCHED_EXPERIMENTS`, and the best
    classical model or ceiling of its experiment, cell type and fold
    otherwise.

    Args
    ----
    output_directory: The output directory of `submit`.
    tolerance: The AUC gap.
    presets_filepath: The presets file, whose models the matched comparisons
        measure.

    Returns
    -------
    The targets of each experiment, as (dataset, fold, preset).
    """
    rows: list[dict[str, Any]] = summarize_runs(output_directory)
    groups = comparison_groups(rows)
    sizes: dict[tuple[str, str], ModelSize] = {}
    if any(key[0] in MATCHED_EXPERIMENTS for key in groups):
        sizes = measure_sizes(
            [
                row
                for row in rows
                if _usable(row) and row["experiment"] in MATCHED_EXPERIMENTS
            ],
            model_factory.load_presets(presets_filepath),
        )
    targets: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for (experiment, dataset, fold), by_preset in groups.items():
        means: dict[str, tuple[float, float]] = {
            preset: _means(preset_rows)
            for preset, preset_rows in by_preset.items()
        }
        pairs: list[tuple[str, str]]
        if experiment in MATCHED_EXPERIMENTS:
            pairs = [
                (pair.quantum, pair.classical)
                for pair in matched_pairs(
                    means,
                    {
                        preset: sizes[(preset, dataset)]
                        for preset in means
                        if (preset, dataset) in sizes
                    },
                )
            ]
        else:
            references: list[str] = [p for p in means if is_reference(p)]
            if not references:
                continue
            best: str = max(references, key=lambda p: means[p][1])
            pairs = [(p, best) for p in means if not is_reference(p)]
        for quantum, classical in pairs:
            if means[quantum][1] >= means[classical][1] - tolerance:
                targets[experiment] |= {
                    (dataset, fold, quantum),
                    (dataset, fold, classical),
                }
    return targets


def training_entries(
    experiment_name: str,
    dataset: Dataset,
    presets: list[model_factory.Preset],
    seeds: Iterable[int],
    output_directory: Path,
    costs: dict[str, float],
    collapsed: set[str],
    *,
    max_epochs: int,
    minimum_patience: int,
    presets_filepath: Path,
    learning_rates: dict[str, float] | None = None,
    extra_seeds: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Plan an experiment's runs on a dataset.

    Every run keeps the epoch of the best mean per-cell validation AUC,
    with at most `max_epochs` epochs and at least `minimum_patience`. A run
    that collapsed is run again with its seed plus `REPLACEMENT_SEED_OFFSET`.
    Its estimated hours on a CPU worker assume it trains for every epoch and
    evaluates about one epoch's sites.

    Args
    ----
    experiment_name: The experiment, one of `EXPERIMENTS`.
    dataset: The dataset.
    presets: The presets.
    seeds: The run seeds.
    output_directory: The root of the outputs.
    costs: The microseconds per training site of each preset.
    collapsed: The output directories of collapsed runs.
    max_epochs: The epoch limit.
    minimum_patience: See `run_patience`.
    presets_filepath: The presets file.
    learning_rates: The learning rate of presets that do not use their own,
        unless the experiment sets one.
    extra_seeds: (dataset, fold, preset) that also run `CLOSE_GAP_SEEDS`.

    Returns
    -------
    The entries, one per split, preset and seed, and per rerun.

    Raises
    ------
    See `check_chromosome_files`.
    """
    experiment: Experiment = EXPERIMENTS[experiment_name]
    experiment_directory: Path = output_directory / experiment_name
    entries: list[dict[str, Any]] = []
    for label, chromosomes in experiment.splits(dataset.key, dataset.autosomes):
        check_chromosome_files(dataset, chromosomes)
        training_sites: int = dataset.sites(chromosomes[records.TRAINING_SPLIT])
        for preset in presets:
            learning_rate: float | None = (
                experiment.learning_rate
                if experiment.learning_rate is not None
                else (learning_rates or {}).get(preset.name)
            )
            preset_seeds: list[int] = list(
                dict.fromkeys(
                    [
                        *seeds,
                        *(
                            CLOSE_GAP_SEEDS
                            if (dataset.name, label, preset.name)
                            in (extra_seeds or set())
                            else ()
                        ),
                    ]
                )
            )
            data_directory: Path = (
                dataset.directory
                if preset.input == model_factory.BASES_STR
                else _token_directory(
                    output_directory, dataset.name, preset.input
                )
            )
            sites: int = training_sites
            if not experiment.all_sites:
                sites = min(
                    sites,
                    preset.training.get("training_sample_quantity") or sites,
                )
            training_arguments: list[str] = [
                "--presets-file",
                str(presets_filepath),
                *(["--all-training-samples"] if experiment.all_sites else []),
                "--epochs",
                str(max_epochs),
                "--patience",
                str(run_patience(preset, minimum_patience)),
                "--selection-metric",
                qcpg.MEAN_CELL_AUC,
                *(
                    ["--learning-rate", f"{learning_rate:g}"]
                    if learning_rate is not None
                    else []
                ),
            ]
            preset_directory: Path = (
                experiment_directory
                / dataset.name
                / preset.name.replace("/", "__")
                / label
            )
            for seed in preset_seeds:
                run_seeds: list[int] = [seed]
                if str(preset_directory / f"seed{seed}") in collapsed:
                    run_seeds.append(seed + REPLACEMENT_SEED_OFFSET)
                for run_seed in run_seeds:
                    entries.append(
                        _training_entry(
                            "/".join(
                                part
                                for part in (
                                    experiment_name,
                                    dataset.name,
                                    preset.name,
                                    label,
                                    f"seed{run_seed}",
                                )
                                if part
                            ),
                            preset,
                            data_directory,
                            preset_directory / f"seed{run_seed}",
                            experiment_directory / "records",
                            dataset,
                            chromosomes,
                            run_seed,
                            training_arguments,
                            costs[preset.name]
                            * 1e-6
                            * sites
                            * (max_epochs + 1)
                            / SECONDS_PER_HOUR,
                        )
                    )
    return entries


def plan_entries(
    experiment_presets: dict[str, list[model_factory.Preset]],
    datasets: list[Dataset],
    seeds: list[int] | None,
    output_directory: Path,
    costs: dict[str, float],
    *,
    max_epochs: int,
    minimum_patience: int,
    presets_filepath: Path,
    close_gap_tolerance: float | None = None,
) -> list[dict[str, Any]]:
    """
    Plan every entry of a queue: the token inputs, then the training runs,
    longest first.

    Swept presets outside the sweep wait until it completes, and then use
    its best rate (`swept_learning_rates`).

    Args
    ----
    experiment_presets: The presets of each experiment to run.
    datasets: The datasets.
    seeds: The run seeds of every experiment, or None for each one's own.
    output_directory: The root of the outputs.
    costs, max_epochs, minimum_patience, presets_filepath: See
        `training_entries`.
    close_gap_tolerance: Add `CLOSE_GAP_SEEDS` to the targets of
        `close_gap_targets` with this tolerance, or None for none.

    Returns
    -------
    The entries.
    """
    collapsed: set[str] = collapsed_runs(
        output_directory / name / "records" for name in EXPERIMENTS
    )
    learning_rates: dict[str, float] = swept_learning_rates(output_directory)
    waiting: set[str] = set()
    ready: dict[str, list[model_factory.Preset]] = {}
    for name, presets in experiment_presets.items():
        ready[name] = [
            preset
            for preset in presets
            if name in SWEEP_EXPERIMENTS
            or not is_swept(preset.name)
            or preset.name in learning_rates
        ]
        waiting |= {p.name for p in presets} - {p.name for p in ready[name]}
    experiment_presets = ready
    if waiting:
        print(
            f"Waiting for the learning-rate sweep: {', '.join(sorted(waiting))}"
        )
    targets: dict[str, set[tuple[str, str, str]]] = (
        close_gap_targets(
            output_directory, close_gap_tolerance, presets_filepath
        )
        if close_gap_tolerance is not None
        else {}
    )
    input_names: set[str] = {
        preset.input
        for presets in experiment_presets.values()
        for preset in presets
        if preset.input != model_factory.BASES_STR
    }
    tokens: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for dataset in datasets:
        if input_names:
            tokens += tokenization_entries(
                dataset, input_names, output_directory
            )
        for name, presets in experiment_presets.items():
            runs += training_entries(
                name,
                dataset,
                presets,
                EXPERIMENTS[name].seeds if seeds is None else seeds,
                output_directory,
                costs,
                collapsed,
                max_epochs=max_epochs,
                minimum_patience=minimum_patience,
                presets_filepath=presets_filepath,
                learning_rates=learning_rates,
                extra_seeds=targets.get(name),
            )
    return tokens + sorted(runs, key=lambda entry: -entry["hours"])


def read_entries(queue_directory: Path) -> list[dict[str, Any]]:
    """Read the entries of a queue."""
    with (queue_directory / ENTRIES_FILENAME).open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def write_queue(
    output_directory: Path,
    entries: list[dict[str, Any]],
    settings: dict[str, Any],
) -> Path:
    """
    Write a new queue.

    Args
    ----
    output_directory: The root of the outputs.
    entries: The queue's entries.
    settings: The settings its workers read.

    Returns
    -------
    The queue's folder.
    """
    queue_directory: Path = (
        output_directory
        / "queues"
        / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    )
    queue_directory.mkdir(parents=True)
    for folder in ("claims", "finished", "logs"):
        (queue_directory / folder).mkdir()
    with (queue_directory / ENTRIES_FILENAME).open("w") as fd:
        for entry in entries:
            fd.write(json.dumps(entry) + "\n")
    records.write_json(queue_directory / QUEUE_FILENAME, settings)
    return queue_directory


def read_state(queue_directory: Path) -> QueueState:
    """Read which entries of a queue are claimed and finished."""
    return QueueState(
        claimed={
            int(directory.name)
            for directory in (queue_directory / "claims").iterdir()
        },
        finished={
            int(filepath.stem): records.read_json(filepath)["returncode"]
            for filepath in (queue_directory / "finished").glob("*.json")
        },
    )


def _claim_job(queue_directory: Path, index: int) -> str | None:
    """The job that claimed an entry, if recorded."""
    filepath: Path = queue_directory / "claims" / str(index) / CLAIM_FILENAME
    return records.read_json(filepath).get("job") if filepath.exists() else None


def choose_entry(
    entries: list[dict[str, Any]],
    state: QueueState,
    resources: str,
    *,
    hours_limit: float = math.inf,
) -> tuple[int | None, list[int]]:
    """
    Choose the entry a worker runs next.

    GPU workers take the longest runs first, and entries for CPU workers
    only when no run is left, so that no input waits for a CPU node. CPU
    workers take the entries for CPU workers first, then the shortest runs
    estimated to last at most `hours_limit`.

    Args
    ----
    entries: The queue's entries, longest runs first.
    state: The queue's progress.
    resources: The worker's resources, `GPU` or `CPU`.
    hours_limit: The longest run a CPU worker takes.

    Returns
    -------
    The index of the entry to run, if any, and the entries the worker could
    run once their inputs are written.
    """
    cpu_entries: list[int] = [
        index
        for index, entry in enumerate(entries)
        if entry["resources"] == CPU_RESOURCES
    ]
    if resources == GPU:
        order: list[int] = [
            index
            for index, entry in enumerate(entries)
            if entry["resources"] == ANY_RESOURCES
        ] + cpu_entries
    else:
        order = cpu_entries + [
            index
            for index, entry in reversed(list(enumerate(entries)))
            if entry["resources"] == ANY_RESOURCES
            and entry["hours"] <= hours_limit
        ]

    waiting: list[int] = []
    for index in order:
        entry: dict[str, Any] = entries[index]
        if (
            index in state.claimed
            or index in state.finished
            or Path(entry["marker"]).exists()
        ):
            continue
        if all(Path(requirement).exists() for requirement in entry["requires"]):
            return index, waiting
        waiting.append(index)
    return None, waiting


def inputs_pending(
    queue_directory: Path,
    entries: list[dict[str, Any]],
    state: QueueState,
    waiting: list[int],
) -> bool:
    """
    Whether inputs that waiting entries need may still be written.

    Args
    ----
    queue_directory: The queue's folder.
    entries: The queue's entries.
    state: The queue's progress.
    waiting: The entries waiting for inputs.

    Returns
    -------
    True if an entry writing a missing input has not run yet, or runs in a
    job SLURM still lists.
    """
    producers: dict[str, int] = {
        entry["marker"]: index for index, entry in enumerate(entries)
    }
    active_jobs: set[str] | None = None
    for index in waiting:
        for requirement in entries[index]["requires"]:
            producer: int | None = producers.get(requirement)
            if (
                Path(requirement).exists()
                or producer is None
                or producer in state.finished
            ):
                continue
            if producer not in state.claimed:
                return True
            if active_jobs is None:
                active_jobs = _active_job_ids()
            if (
                active_jobs is None
                or _claim_job(queue_directory, producer) in active_jobs
            ):
                return True
    return False


def claim(queue_directory: Path, index: int, details: dict[str, Any]) -> bool:
    """
    Claim an entry, unless another worker did.

    Args
    ----
    queue_directory: The queue's folder.
    index: The entry.
    details: What to record of the claiming worker.

    Returns
    -------
    Whether this worker claimed the entry.
    """
    directory: Path = queue_directory / "claims" / str(index)
    try:
        directory.mkdir()
    except FileExistsError:
        return False

    records.write_json(directory / CLAIM_FILENAME, details)
    return True


def run_entry(
    queue_directory: Path,
    index: int,
    entry: dict[str, Any],
    resources: str,
    cpus: int,
    gpu: str | None,
) -> int:
    """
    Run an entry, logging its output in the queue's logs folder.

    Training runs share the worker's CPUs between computing and loading
    data, and train on its GPU if it has one. A successful entry leaves its
    marker.

    Args
    ----
    queue_directory: The queue's folder.
    index: The entry's index.
    entry: The entry.
    resources: The worker's resources, `GPU` or `CPU`.
    cpus: The worker's CPUs.
    gpu: The worker's GPU, if a GPU worker.

    Returns
    -------
    The entry's return code.
    """
    threads: int = cpus
    command: list[str] = [
        sys.executable,
        str(SCRIPTS[entry["script"]]),
        *entry["arguments"],
    ]
    if entry["script"] == TRAIN:
        loader_workers: int = int(cpus * LOADER_SHARES[resources])
        threads = max(1, cpus - loader_workers)
        command += [
            "--threads",
            str(threads),
            "--loader-workers",
            str(loader_workers),
            *(["--gpus", "1"] if gpu is not None else []),
        ]
    environment: dict[str, str] = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "RAYON_NUM_THREADS": str(threads),
        "CUDA_VISIBLE_DEVICES": gpu or "",
    }
    started: float = time.time()
    with (queue_directory / "logs" / f"{index}.log").open("w") as fd:
        returncode: int = subprocess.run(
            command,
            stdout=fd,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        ).returncode
    finished: str = datetime.now().astimezone().isoformat(timespec="seconds")
    if returncode == 0:
        marker: Path = Path(entry["marker"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        records.write_json(
            marker,
            {
                "queue": str(queue_directory),
                "entry": index,
                "finished": finished,
            },
        )
    records.write_json(
        queue_directory / "finished" / f"{index}.json",
        {
            "returncode": returncode,
            "finished": finished,
            "hours": round((time.time() - started) / SECONDS_PER_HOUR, 3),
        },
    )
    return returncode


def _work(args: Namespace) -> None:
    """
    Run entries of a queue until none is left that this worker may take.

    Args
    ----
    args: The command line arguments of `work`.
    """
    queue_directory: Path = args.queue_directory.resolve()
    settings: dict[str, Any] = records.read_json(
        queue_directory / QUEUE_FILENAME
    )
    entries: list[dict[str, Any]] = read_entries(queue_directory)
    label: str = f"{socket.gethostname()} {args.resources}" + (
        f" {args.gpu}" if args.gpu is not None else ""
    )
    started: float = time.time()
    remaining_hours: float = _remaining_hours()
    while True:
        hours_limit: float = min(
            settings["cpu_max_hours"],
            (remaining_hours - (time.time() - started) / SECONDS_PER_HOUR)
            / CPU_TIME_SAFETY,
        )
        state: QueueState = read_state(queue_directory)
        index, waiting = choose_entry(
            entries, state, args.resources, hours_limit=hours_limit
        )
        if index is None:
            if waiting and inputs_pending(
                queue_directory, entries, state, waiting
            ):
                time.sleep(POLL_SECONDS)
                continue

            print(f"{label}: no entries left to take", flush=True)
            return

        if not claim(
            queue_directory,
            index,
            {
                "job": os.environ.get("SLURM_JOB_ID"),
                "worker": label,
                "started": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            },
        ):
            continue

        print(
            f"{label}: started {entries[index]['name']} ({index})", flush=True
        )
        returncode: int = run_entry(
            queue_directory,
            index,
            entries[index],
            args.resources,
            args.cpus,
            args.gpu,
        )
        print(
            f"{label}: {entries[index]['name']} ({index}) ended with code "
            f"{returncode}",
            flush=True,
        )


def sbatch_command(
    resources: str,
    partition: str,
    node_quantity: int,
    time_limit: str,
    queue_directory: Path,
    log_directory: Path,
    sbatch_arguments: str,
    cpu_workers_per_node: int,
) -> list[str]:
    """
    Build the sbatch command of a queue's node jobs on one partition.

    Args
    ----
    resources: `GPU` or `CPU`.
    partition: The partition.
    node_quantity: The node jobs, which all may run at once.
    time_limit: Their time limit.
    queue_directory: The queue's folder.
    log_directory: The folder of the jobs' output.
    sbatch_arguments: Further sbatch options.
    cpu_workers_per_node: The workers of each CPU node.

    Returns
    -------
    The command.
    """
    return [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--array=0-{node_quantity - 1}%{node_quantity}",
        f"--time={time_limit}",
        f"--output={log_directory}/%x_%A_%a.out",
        f"--export=ALL,QENETICS_REPOSITORY={REPOSITORY}",
        *shlex.split(sbatch_arguments),
        str(TEMPLATES[resources]),
        str(queue_directory),
        *([str(cpu_workers_per_node)] if resources == CPU else []),
    ]


def experiment_presets(
    presets: dict[str, model_factory.Preset],
    experiments: list[str],
    patterns: list[str],
    exclusions: list[str],
    *,
    keep_duplicates: bool = False,
) -> tuple[dict[str, list[model_factory.Preset]], dict[str, str]]:
    """
    Choose the presets of each experiment.

    Args
    ----
    presets: The presets, indexed by name.
    experiments: The experiments, of `EXPERIMENTS`.
    patterns: Run only presets matching these names or patterns.
    exclusions, keep_duplicates: See `choose_presets`; they apply to
        deduplicated experiments.

    Returns
    -------
    The presets of each experiment, and the presets a deduplicated
    experiment leaves out, with why.

    Raises
    ------
    ValueError if a pattern matches no preset.
    """
    choose_presets(presets, patterns, [], keep_duplicates=True)
    chosen: dict[str, list[model_factory.Preset]] = {}
    skipped: dict[str, str] = {}
    for name in experiments:
        experiment: Experiment = EXPERIMENTS[name]
        names, left_out = choose_presets(
            presets,
            experiment.presets,
            exclusions if experiment.deduplicated else [],
            keep_duplicates=keep_duplicates or not experiment.deduplicated,
        )
        chosen[name] = [
            presets[preset]
            for preset in names
            if any(fnmatchcase(preset, pattern) for pattern in patterns)
        ]
        skipped.update(
            (preset, reason)
            for preset, reason in left_out.items()
            if any(fnmatchcase(preset, pattern) for pattern in patterns)
        )
    return chosen, skipped


def _submit(args: Namespace) -> None:
    """
    Queue the plan's runs that have not completed and submit the node jobs.

    Args
    ----
    args: The command line arguments of `submit`.

    Raises
    ------
    ValueError if a data directory is missing, sbatch is not available, or
    node jobs of an earlier queue are still queued or running.
    """
    missing: list[str] = [
        str(directory)
        for directory in args.data_directories
        if not directory.is_dir()
    ]
    if missing:
        raise ValueError(f"Data directories not found: {', '.join(missing)}")
    if not args.dry_run:
        if shutil.which("sbatch") is None:
            raise ValueError(
                "sbatch is not available; use --dry-run to inspect"
            )
        running: str | None = _slurm_output(
            [
                "squeue",
                "-h",
                "-u",
                getpass.getuser(),
                "-n",
                ",".join(JOB_NAMES.values()),
                "-o",
                "%i",
            ]
        )
        if running and running.split():
            raise ValueError(
                "Node jobs of an earlier queue are still queued or running: "
                f"{' '.join(running.split())}. Wait for them, or cancel them "
                f"with scancel -n {','.join(JOB_NAMES.values())}"
            )

    presets: dict[str, model_factory.Preset] = model_factory.load_presets(
        args.presets_file
    )
    chosen, skipped = experiment_presets(
        presets,
        args.experiments,
        args.presets,
        args.exclude,
        keep_duplicates=args.keep_duplicates,
    )
    for name, reason in skipped.items():
        print(f"Skipping {name}: {reason}")
    output_directory: Path = args.output_directory.resolve()
    entries: list[dict[str, Any]] = plan_entries(
        chosen,
        [
            load_dataset(directory.resolve())
            for directory in args.data_directories
        ],
        args.seeds,
        output_directory,
        load_costs(
            COSTS_FILEPATH,
            (preset.name for names in chosen.values() for preset in names),
        ),
        max_epochs=args.max_epochs,
        minimum_patience=args.minimum_patience,
        presets_filepath=args.presets_file.resolve(),
        close_gap_tolerance=args.close_gap_tolerance
        if args.close_gaps
        else None,
    )
    pending: list[dict[str, Any]] = [
        entry for entry in entries if not Path(entry["marker"]).exists()
    ]
    print(
        f"{len(entries) - len(pending)} of {len(entries)} entries are already "
        "completed"
    )
    if not pending:
        return

    time_limits: dict[str, str] = {
        GPU: _time_limit(args.gpu_time, args.gpu_partition),
        CPU: _time_limit(args.cpu_time, args.cpu_partition),
    }
    cpu_hours_limit: float = min(
        args.cpu_max_hours, slurm_hours(time_limits[CPU]) / CPU_TIME_SAFETY
    )
    runs: list[dict[str, Any]] = [
        entry for entry in pending if entry["script"] == TRAIN
    ]
    token_quantity: int = len(pending) - len(runs)
    cpu_run_quantity: int = sum(
        entry["hours"] <= cpu_hours_limit for entry in runs
    )
    gpu_node_quantity: int = min(
        args.gpu_nodes, math.ceil(len(runs) / GPUS_PER_NODE)
    )
    node_quantities: dict[str, int] = {
        GPU: gpu_node_quantity,
        CPU: min(
            MAX_NODES - gpu_node_quantity
            if args.cpu_nodes is None
            else args.cpu_nodes,
            math.ceil(
                (token_quantity + cpu_run_quantity) / args.cpu_workers_per_node
            ),
        ),
    }
    if node_quantities[GPU] == 0 and cpu_run_quantity < len(runs):
        print(
            f"Without GPU nodes, {len(runs) - cpu_run_quantity} runs are too "
            "long to run"
        )

    queue_directory: Path = write_queue(
        output_directory,
        pending,
        {
            "created": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "gpu_nodes": node_quantities[GPU],
            "cpu_nodes": node_quantities[CPU],
            "cpu_max_hours": args.cpu_max_hours,
            "jobs": {},
        },
    )
    experiments: Counter[str] = Counter(
        entry["name"].split("/")[0] for entry in runs
    )
    print(
        f"Wrote {len(runs)} runs "
        f"({', '.join(f'{name} {count}' for name, count in experiments.items())})"
        f" and {token_quantity} token inputs to {queue_directory}"
    )
    print(
        f"Estimated {sum(entry['hours'] for entry in runs):.0f} CPU worker "
        "hours at most, if every run trains to its epoch limit; "
        f"{cpu_run_quantity} runs are short enough for CPU workers (at most "
        f"{cpu_hours_limit:.0f} hours)"
    )

    log_directory: Path = output_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, str] = {}
    for resources, partition in (
        (GPU, args.gpu_partition),
        (CPU, args.cpu_partition),
    ):
        if node_quantities[resources] == 0:
            continue

        command: list[str] = sbatch_command(
            resources,
            partition,
            node_quantities[resources],
            time_limits[resources],
            queue_directory,
            log_directory,
            args.sbatch_args,
            args.cpu_workers_per_node,
        )
        print(shlex.join(command))
        if not args.dry_run:
            output: str = subprocess.run(
                command, check=True, capture_output=True, text=True
            ).stdout
            jobs[resources] = output.strip().split(";")[0]
            print(f"Submitted {resources} node jobs {jobs[resources]}")
    if jobs:
        settings: dict[str, Any] = records.read_json(
            queue_directory / QUEUE_FILENAME
        )
        settings["jobs"] = jobs
        records.write_json(queue_directory / QUEUE_FILENAME, settings)


def _last_line(filepath: Path) -> str:
    """The last non-empty line of a log, such as the error a run ended with."""
    if not filepath.exists():
        return "(no log)"
    with filepath.open(errors="replace") as fd:
        lines: list[str] = [line.rstrip() for line in fd if line.strip()]
    return lines[-1] if lines else "(empty log)"


def _status(output_directory: Path, queue: str | None) -> None:
    """
    Summarize the progress of a queue.

    Args
    ----
    output_directory: The output directory of `submit`.
    queue: The queue's folder name, or None for the latest.

    Raises
    ------
    FileNotFoundError if the output directory holds no queue.
    """
    queues_directory: Path = output_directory / "queues"
    names: list[str] = (
        sorted(path.name for path in queues_directory.iterdir())
        if queues_directory.is_dir()
        else []
    )
    if not names:
        raise FileNotFoundError(f"No queues in {queues_directory}")

    queue_directory: Path = queues_directory / (queue or names[-1])
    entries: list[dict[str, Any]] = read_entries(queue_directory)
    state: QueueState = read_state(queue_directory)
    active_jobs: set[str] | None = _active_job_ids()
    categories: dict[int, str] = {}
    for index, entry in enumerate(entries):
        if Path(entry["marker"]).exists():
            categories[index] = "completed"
        elif index in state.finished:
            categories[index] = "failed"
        elif index in state.claimed:
            categories[index] = (
                "claimed"
                if active_jobs is None
                else "running"
                if _claim_job(queue_directory, index) in active_jobs
                else "interrupted"
            )
        elif all(
            Path(requirement).exists() for requirement in entry["requires"]
        ):
            categories[index] = "pending"
        else:
            categories[index] = "waiting for inputs"

    counts: Counter[str] = Counter(categories.values())
    print(f"Queue {queue_directory}: {len(entries)} entries")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    for index, category in categories.items():
        if category in ("failed", "interrupted"):
            log_filepath: Path = queue_directory / "logs" / f"{index}.log"
            print(f"{category}: {entries[index]['name']}, log {log_filepath}")
            if category == "failed":
                print(f"  {_last_line(log_filepath)}")
    records_directories: set[Path] = {
        Path(arguments[arguments.index("--records-directory") + 1])
        for arguments in (entry["arguments"] for entry in entries)
        if "--records-directory" in arguments
    }
    collapsed: set[str] = collapsed_runs(records_directories)
    rerun_quantity: int = 0
    for entry in entries:
        run_directory: Path = Path(entry["marker"]).parent
        if str(run_directory) not in collapsed:
            continue
        seed: int = int(run_directory.name.removeprefix("seed"))
        replaced: Path = run_directory.with_name(
            f"seed{seed - REPLACEMENT_SEED_OFFSET}"
        )
        replacement: Path = run_directory.with_name(
            f"seed{seed + REPLACEMENT_SEED_OFFSET}"
        )
        if str(replaced) in collapsed:
            print(f"collapsed after its rerun, kept: {entry['name']}")
        elif (replacement / COMPLETED_FILENAME).exists():
            print(f"collapsed, rerun as {replacement.name}: {entry['name']}")
        else:
            rerun_quantity += 1
            print(f"collapsed: {entry['name']}")
    if counts["failed"] or counts["interrupted"] or rerun_quantity:
        print(
            "Submit again to rerun failed and interrupted entries, and to "
            f"rerun collapsed runs with seed + {REPLACEMENT_SEED_OFFSET}."
        )


def _run_identity(output_directory: Path, root: Path) -> dict[str, str] | None:
    """
    Name a run from its output directory, as `training_entries` lays it out.

    Args
    ----
    output_directory: The run's output directory.
    root: The output directory of `submit`.

    Returns
    -------
    The run's experiment, dataset, preset, fold (empty if none) and seed, or
    None if the directory is not a run of `submit`.
    """
    try:
        parts: tuple[str, ...] = output_directory.relative_to(root).parts
    except ValueError:
        return None
    if (
        len(parts) not in (4, 5)
        or parts[0] not in EXPERIMENTS
        or not parts[-1].removeprefix("seed").isdigit()
    ):
        return None

    return {
        "experiment": parts[0],
        "dataset": parts[1],
        "preset": parts[2].replace("__", "/"),
        "fold": parts[3] if len(parts) == 5 else "",
        "seed": parts[-1].removeprefix("seed"),
    }


def _best_epoch(
    epochs: list[dict[str, Any]], selection_metric: str | None
) -> tuple[int | None, float]:
    """
    Find the epoch a run kept, as `qcpg` selects it.

    Args
    ----
    epochs: The run's epochs, as `records.append_epoch` writes them.
    selection_metric: The run's selection metric.

    Returns
    -------
    The first epoch of the best validation score and that score, or None and
    NaN if no epoch had a defined score.
    """
    best_epoch: int | None = None
    best_score: float = -math.inf
    for epoch in epochs:
        score: float = (
            epoch["validation_auc"]
            if selection_metric == qcpg.POOLED_AUC
            else mean_cell_auc(epoch["cell_validation_aucs"])
        )
        if score > best_score:
            best_epoch, best_score = epoch["epoch"], score
    return best_epoch, best_score if best_epoch is not None else math.nan


def _run_hours(run: dict[str, Any]) -> float:
    """A run's wall time from its record; NaN if it has not finished."""
    if not run.get("started") or not run.get("finished"):
        return math.nan

    duration = datetime.fromisoformat(run["finished"]) - datetime.fromisoformat(
        run["started"]
    )
    return duration.total_seconds() / SECONDS_PER_HOUR


SUMMARY_FILENAME: str = "summary.tsv"
SUMMARY_COLUMNS: tuple[str, ...] = (
    "experiment",
    "dataset",
    "preset",
    "fold",
    "seed",
    "status",
    "epochs_trained",
    "best_epoch",
    "best_validation_score",
    "test_mean_cell_auc",
    "test_pooled_auc",
    "test_observed_labels",
    "collapsed",
    "hours",
    "record",
)


def summarize_runs(output_directory: Path) -> list[dict[str, Any]]:
    """
    Summarize the recorded runs of every experiment, one row per run.

    A run's latest completed record counts, or its latest record if none
    completed, so failed attempts that were rerun do not appear.

    Args
    ----
    output_directory: The output directory of `submit`.

    Returns
    -------
    The rows, with the values of `SUMMARY_COLUMNS`, ordered by experiment,
    dataset, preset, fold and seed.
    """
    rows: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        records_directory: Path = output_directory / experiment / "records"
        if not records_directory.is_dir():
            continue
        for run_directory in records.find_run_directories(records_directory):
            run: dict[str, Any] = records.read_run(run_directory)
            parameters: dict[str, Any] = run["parameters"]
            run_output: str = str(parameters.get("output_directory", ""))
            identity: dict[str, str] | None = _run_identity(
                Path(run_output), output_directory
            )
            previous: dict[str, Any] | None = rows.get(run_output)
            if identity is None or (
                previous is not None
                and previous["status"] == records.STATUS_COMPLETED
                and run["status"] != records.STATUS_COMPLETED
            ):
                continue

            best_epoch, best_score = _best_epoch(
                run["epochs"], parameters.get("selection_metric")
            )
            test: dict[str, Any] = run["metrics"].get(records.TEST_SPLIT, {})
            pooled: dict[str, Any] = test.get("pooled", {})
            test_auc: float = _test_mean_cell_auc(run)
            rows[run_output] = {
                **identity,
                "status": run["status"],
                "epochs_trained": len(run["epochs"]),
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "test_mean_cell_auc": test_auc,
                "test_pooled_auc": pooled.get("auc", math.nan),
                "test_observed_labels": pooled.get("observed_sites"),
                "collapsed": math.isfinite(test_auc)
                and _is_collapsed(test_auc),
                "hours": _run_hours(run),
                "record": str(run_directory),
            }
    return sorted(
        rows.values(),
        key=lambda row: (
            row["experiment"],
            row["dataset"],
            row["preset"],
            row["fold"],
            int(row["seed"]),
        ),
    )


def _summary_value(value: Any) -> str:
    """Write a summary value: floats to 6 decimals, missing values empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _summarize(output_directory: Path, summary_filepath: Path | None) -> None:
    """
    Write the summary of every recorded run as tab-separated values.

    Args
    ----
    output_directory: The output directory of `submit`.
    summary_filepath: The file to write, by default `SUMMARY_FILENAME` in
        the output directory.
    """
    output_directory = output_directory.resolve()
    summary_filepath = summary_filepath or output_directory / SUMMARY_FILENAME
    rows: list[dict[str, Any]] = summarize_runs(output_directory)
    with summary_filepath.open("w", newline="") as fd:
        fd.write("\t".join(SUMMARY_COLUMNS) + "\n")
        for row in rows:
            fd.write(
                "\t".join(
                    _summary_value(row[column]) for column in SUMMARY_COLUMNS
                )
                + "\n"
            )
    statuses: Counter[str] = Counter(row["status"] for row in rows)
    print(
        f"Wrote {len(rows)} runs to {summary_filepath} ("
        + ", ".join(
            f"{status} {count}" for status, count in sorted(statuses.items())
        )
        + f"; {sum(row['collapsed'] for row in rows)} collapsed)"
    )


COMPARISONS_FOLDER: str = "comparisons"
COMPARISONS_FILENAME: str = "comparisons.tsv"
METRICS_FILENAME: str = "metrics.tsv"
MATCHED_FILENAMES: dict[str, str] = {
    FRONT_END_TABLE: "matched_front_end.tsv",
    TOTAL_TABLE: "matched_total.tsv",
}
COMPARISON_COLUMNS: tuple[str, ...] = (
    "family",
    "setting",
    "dataset",
    "fold",
    "model_a",
    "model_b",
    "seeds_a",
    "seeds_b",
    "size_a",
    "size_b",
    "scored",
    "validation_a",
    "validation_b",
    "test_a",
    "test_b",
    "sites",
    "difference",
    "low",
    "high",
    "p_value",
    "q_value",
    "difference_all_sites",
)
METRIC_COLUMNS: tuple[str, ...] = (
    "experiment",
    "dataset",
    "preset",
    "fold",
    "seed",
    "test_mean_cell_auc",
    "test_mean_cell_auprc",
    "test_mean_cell_mcc",
    "record",
)
_WEIGHTS: dict[tuple[int, int, int], Any] = {}


def _test_sites_fingerprint(outputs: records.SplitOutputs) -> str:
    """Identify a test set by its sites, in order."""
    return hashlib.sha1(
        np.ascontiguousarray(outputs.chromosome_indices, np.int64).tobytes()
        + np.ascontiguousarray(outputs.positions, np.int64).tobytes()
    ).hexdigest()


def _site_sample(site_quantity: int, sample_quantity: int, seed: int) -> Any:
    """A fixed random sample of a test set's sites, in order."""
    if site_quantity <= sample_quantity:
        return np.arange(site_quantity)
    return np.sort(
        np.random.default_rng(seed).choice(
            site_quantity, sample_quantity, replace=False
        )
    )


def run_statistics(job: dict[str, Any]) -> dict[str, Any]:
    """
    Compute one run's statistics; called by `compare` in worker processes.

    Args
    ----
    job: The run's record, its test set's fingerprint and seed, the sample
        size, the replicates and the cache file of its bootstrap values.

    Returns
    -------
    The record, and its bootstrap values (`bootstrap.run_replicates`) on its
    test set's sample and its mean per-cell AUPRC and MCC on every test site,
    or an error.
    """
    outputs: records.SplitOutputs | None = records.read_split_outputs(
        Path(job["record"]), records.TEST_SPLIT
    )
    if outputs is None:
        return {"record": job["record"], "error": "no test outputs"}
    if _test_sites_fingerprint(outputs) != job["fingerprint"]:
        return {"record": job["record"], "error": "different test sites"}

    precision, correlation = bootstrap.cell_metrics(
        outputs.predictions, outputs.labels
    )
    cache: Path = Path(job["cache"])
    if cache.exists():
        values = np.load(cache)
    else:
        sample = _site_sample(len(outputs.labels), job["sites"], job["seed"])
        labels = outputs.labels[sample]
        key = (job["seed"], len(sample), job["replicates"])
        if key not in _WEIGHTS:
            _WEIGHTS.clear()
            _WEIGHTS[key] = bootstrap.site_weights(
                job["replicates"], len(sample), job["seed"]
            )
        values = bootstrap.run_replicates(
            outputs.predictions[sample],
            labels,
            _WEIGHTS[key],
            bootstrap.scored_cells(labels),
        )
        np.save(cache, values)
    return {
        "record": job["record"],
        "values": values,
        "auprc": precision,
        "mcc": correlation,
        "sites": min(len(outputs.labels), job["sites"]),
    }


def _write_tsv(
    filepath: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    with filepath.open("w", newline="") as fd:
        fd.write("\t".join(columns) + "\n")
        for row in rows:
            fd.write(
                "\t".join(_summary_value(row[column]) for column in columns)
                + "\n"
            )


def load_pairs(filepath: Path) -> list[tuple[str, str]]:
    """Read the complex circuit models and their real twins, as (complex,
    real) presets."""
    with filepath.open() as fd:
        return [
            (pair["complex"], pair["real"]) for pair in json.load(fd)["pairs"]
        ]


def _compare(
    output_directory: Path,
    replicate_quantity: int,
    site_quantity: int,
    processes: int,
    pairs: list[tuple[str, str]],
    presets_filepath: Path = PRESETS_FILEPATH,
    pairs_filepath: Path = PAIRS_FILEPATH,
) -> None:
    """
    Bootstrap the comparisons of every usable run's model, and write every
    run's secondary metrics.

    Runs are compared within an experiment, cell type and fold: in
    `MATCHED_EXPERIMENTS` every quantum model with its matched comparators
    (`matched_pairs`), elsewhere every quantum model with every classical
    model and ceiling; and everywhere the complex and real twins of
    `pairs_filepath`, and `pairs`. Each test set is resampled on a fixed
    sample of at most `site_quantity` sites with the same weights for every
    model. Single-seed models borrow the pooled seed variance of their
    experiment and cell type, else their cell type's, else everyone's. q
    values are Benjamini–Hochberg within each family: one matched table, one
    experiment's comparisons with one classical model, its complex and real
    twins, or its other pairs.

    Args
    ----
    output_directory: The output directory of `submit`.
    replicate_quantity: The bootstrap replicates.
    site_quantity: The most sites of each test set's sample.
    processes: The worker processes.
    pairs: Further (A, B) preset pairs.
    presets_filepath: The presets file, whose models the matched
        comparisons measure.
    pairs_filepath: The complex and real twins, see `load_pairs`.
    """
    output_directory = output_directory.resolve()
    folder: Path = output_directory / COMPARISONS_FOLDER
    cache_folder: Path = folder / "cache"
    cache_folder.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = summarize_runs(output_directory)
    usable: list[dict[str, Any]] = [row for row in rows if _usable(row)]
    groups = comparison_groups(rows)

    jobs: list[dict[str, Any]] = []
    for by_preset in groups.values():
        group_rows: list[dict[str, Any]] = [
            row for preset_rows in by_preset.values() for row in preset_rows
        ]
        reference = records.read_split_outputs(
            Path(group_rows[0]["record"]), records.TEST_SPLIT
        )
        if reference is None:
            continue
        fingerprint: str = _test_sites_fingerprint(reference)
        for row in group_rows:
            record_hash: str = hashlib.sha1(row["record"].encode()).hexdigest()
            jobs.append(
                {
                    "record": row["record"],
                    "fingerprint": fingerprint,
                    "seed": zlib.crc32(fingerprint.encode()),
                    "sites": site_quantity,
                    "replicates": replicate_quantity,
                    "cache": str(
                        cache_folder
                        / f"{record_hash}_R{replicate_quantity}_S{site_quantity}.npy"
                    ),
                }
            )
    print(f"Computing the statistics of {len(jobs)} runs", flush=True)
    jobs.sort(key=lambda job: job["fingerprint"])
    if processes > 1:
        with Pool(processes) as pool:
            results = list(pool.imap(run_statistics, jobs, chunksize=4))
    else:
        results = [run_statistics(job) for job in jobs]
    by_record: dict[str, dict[str, Any]] = {}
    for result in results:
        if "error" in result:
            print(f"Skipping {result['record']}: {result['error']}")
        else:
            by_record[result["record"]] = result

    metric_rows: list[dict[str, Any]] = [
        {
            **{
                column: row[column]
                for column in METRIC_COLUMNS
                if column in row
            },
            "test_mean_cell_auprc": by_record[row["record"]]["auprc"],
            "test_mean_cell_mcc": by_record[row["record"]]["mcc"],
        }
        for row in usable
        if row["record"] in by_record
    ]
    _write_tsv(folder / METRICS_FILENAME, METRIC_COLUMNS, metric_rows)

    values: dict[tuple[str, str, str], dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )
    means: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = {}
    sample_sizes: dict[tuple[str, str, str], int] = {}
    for key, by_preset in groups.items():
        for preset, preset_rows in by_preset.items():
            computed: list[dict[str, Any]] = [
                row for row in preset_rows if row["record"] in by_record
            ]
            for row in computed:
                values[key][preset].append(by_record[row["record"]]["values"])
                sample_sizes[key] = by_record[row["record"]]["sites"]
            if computed:
                means.setdefault(key, {})[preset] = _means(computed)
    variances: dict[Any, float] = {}
    for level in ("setting", "dataset", "all"):
        pools: dict[Any, list[Any]] = defaultdict(list)
        for (setting, dataset, _), by_preset in values.items():
            pool_key = {
                "setting": (setting, dataset),
                "dataset": dataset,
                "all": "all",
            }[level]
            pools[pool_key] += [
                np.array([run[0] for run in runs])
                for runs in by_preset.values()
            ]
        for pool_key, seed_values in pools.items():
            variance: float = bootstrap.pooled_seed_variance(seed_values)
            if math.isfinite(variance):
                variances[pool_key] = variance

    def seed_variance(setting: str, dataset: str) -> float:
        for pool_key in ((setting, dataset), dataset, "all"):
            if pool_key in variances:
                return variances[pool_key]
        return 0.0

    sizes: dict[tuple[str, str], ModelSize] = {}
    if any(key[0] in MATCHED_EXPERIMENTS for key in values):
        matched_rows: list[dict[str, Any]] = [
            row
            for row in usable
            if row["experiment"] in MATCHED_EXPERIMENTS
            and row["record"] in by_record
        ]
        print(f"Measuring the models of {len(matched_rows)} runs", flush=True)
        sizes = measure_sizes(
            matched_rows, model_factory.load_presets(presets_filepath)
        )
    twins: list[tuple[str, str]] = load_pairs(pairs_filepath)
    comparisons: list[dict[str, Any]] = []
    for (setting, dataset, fold), by_preset in sorted(values.items()):
        group_key = (setting, dataset, fold)
        group_means: dict[str, tuple[float, float]] = means[group_key]
        candidates: list[tuple[str, str, str, MatchedPair | None]]
        if setting in MATCHED_EXPERIMENTS:
            candidates = [
                (
                    f"{setting} matched {pair.table}",
                    pair.quantum,
                    pair.classical,
                    pair,
                )
                for pair in matched_pairs(
                    group_means,
                    {
                        preset: sizes[(preset, dataset)]
                        for preset in group_means
                        if (preset, dataset) in sizes
                    },
                )
            ]
        else:
            references = sorted(p for p in by_preset if is_reference(p))
            quantum = sorted(p for p in by_preset if not is_reference(p))
            candidates = [
                (f"{setting} vs {r}", q, r, None)
                for r in references
                for q in quantum
            ]
        candidates += [
            (f"{setting} {family}", a, b, None)
            for family, family_pairs in (
                ("complex vs real", twins),
                ("pairs", pairs),
            )
            for a, b in family_pairs
            if a in by_preset and b in by_preset
        ]
        variance: float = seed_variance(setting, dataset)
        for family, a, b, matched in candidates:
            comparison = bootstrap.compare(
                np.stack(by_preset[a]),
                np.stack(by_preset[b]),
                np.random.default_rng(
                    zlib.crc32(f"{family}|{dataset}|{fold}|{a}|{b}".encode())
                ),
                single_seed_variance_a=variance,
                single_seed_variance_b=variance,
            )
            comparisons.append(
                {
                    "family": family,
                    "setting": setting,
                    "dataset": dataset,
                    "fold": fold,
                    "model_a": a,
                    "model_b": b,
                    "seeds_a": len(by_preset[a]),
                    "seeds_b": len(by_preset[b]),
                    "size_a": matched.quantum_size if matched else None,
                    "size_b": matched.classical_size if matched else None,
                    "scored": matched.scored if matched else None,
                    "validation_a": group_means[a][0],
                    "validation_b": group_means[b][0],
                    "test_a": group_means[a][1],
                    "test_b": group_means[b][1],
                    "sites": sample_sizes[group_key],
                    "difference": comparison.difference,
                    "low": comparison.low,
                    "high": comparison.high,
                    "p_value": comparison.p_value,
                    "difference_all_sites": group_means[a][1]
                    - group_means[b][1],
                }
            )
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comparison in comparisons:
        families[comparison["family"]].append(comparison)
    for family_comparisons in families.values():
        q_values = bootstrap.benjamini_hochberg(
            [comparison["p_value"] for comparison in family_comparisons]
        )
        for comparison, q_value in zip(
            family_comparisons, q_values, strict=True
        ):
            comparison["q_value"] = float(q_value)
    _write_tsv(folder / COMPARISONS_FILENAME, COMPARISON_COLUMNS, comparisons)
    for table, filename in MATCHED_FILENAMES.items():
        _write_tsv(
            folder / filename,
            COMPARISON_COLUMNS,
            [
                comparison
                for comparison in comparisons
                if comparison["family"].endswith(f" matched {table}")
            ],
        )
    print(
        f"Wrote {len(comparisons)} comparisons to {folder / COMPARISONS_FILENAME} "
        f"and the metrics of {len(metric_rows)} runs to {folder / METRICS_FILENAME}"
    )


def main(arguments: list[str] | None = None) -> None:
    args: Namespace = _parse_script_args(arguments)
    if args.command == "submit":
        _submit(args)
    elif args.command == "status":
        _status(args.output_directory, args.queue)
    elif args.command == "summarize":
        _summarize(args.output_directory, args.output)
    elif args.command == "compare":
        _compare(
            args.output_directory,
            args.replicates,
            args.sites,
            args.processes,
            [tuple(pair) for pair in args.pair],
            args.presets_file,
            args.pairs_file,
        )
    else:
        _work(args)


if __name__ == "__main__":
    main()
