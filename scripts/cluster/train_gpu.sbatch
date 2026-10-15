#!/bin/bash
# One run of a run list from `slurm_runs.py submit`, on one GPU.
# Adjust the resources to your cluster, or override them with --sbatch-args.
#SBATCH --job-name=qenetics-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

runs_filepath="$1"
repository="${QENETICS_REPOSITORY:?Submit with slurm_runs.py, which sets QENETICS_REPOSITORY}"
# The Python environment with qenetics installed; load CUDA modules here if
# your cluster needs them.
source "${QENETICS_ENV:-$repository/.venv}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

python "$repository/scripts/cluster/slurm_runs.py" run \
    "$runs_filepath" "$SLURM_ARRAY_TASK_ID"
