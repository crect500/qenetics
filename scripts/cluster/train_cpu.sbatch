#!/bin/bash
# One run of a run list from `slurm_runs.py submit`, on CPUs.
# Adjust the resources to your cluster, or override them with --sbatch-args.
#SBATCH --job-name=qenetics-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

runs_filepath="$1"
repository="${QENETICS_REPOSITORY:?Submit with slurm_runs.py, which sets QENETICS_REPOSITORY}"
# The Python environment with qenetics installed; load modules here if needed.
source "${QENETICS_ENV:-$repository/.venv}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

python "$repository/scripts/cluster/slurm_runs.py" run \
    "$runs_filepath" "$SLURM_ARRAY_TASK_ID"
