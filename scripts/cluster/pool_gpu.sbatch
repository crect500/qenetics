#!/bin/bash
# One whole GPU node of a `plan_runs.py submit` queue: a worker per
# GPU takes entries of the queue until none is left.
# Adjust the resources to your cluster, or override them with --sbatch-args.
#SBATCH --job-name=qcpg-pool-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0

set -euo pipefail

queue_directory="$1"
repository="${QENETICS_REPOSITORY:?Submit with plan_runs.py, which sets QENETICS_REPOSITORY}"
# The Python environment with qenetics installed; load CUDA modules here if
# your cluster needs them.
source "${QENETICS_ENV:-$repository/.venv}/bin/activate"

IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:?No GPUs were allocated}"
cpus_per_worker=$(( SLURM_CPUS_PER_TASK / ${#gpus[@]} ))
for gpu in "${gpus[@]}"; do
    python "$repository/scripts/cluster/plan_runs.py" work \
        "$queue_directory" --resources gpu --gpu "$gpu" \
        --cpus "$cpus_per_worker" &
done
wait
