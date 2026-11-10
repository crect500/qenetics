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

# The cluster's Python: load its container module, then run Python in the
# environment with qenetics installed (QENETICS_ENV). The module scripts are
# not written for `set -u`, so they run before it.
enable_lmod
module load container_env python3/2025
environment="${QENETICS_ENV:-$HOME/envs/pennylane-lightning}"

set -euo pipefail

queue_directory="$1"
repository="${QENETICS_REPOSITORY:?Submit with plan_runs.py, which sets QENETICS_REPOSITORY}"
# Return large tensors to the system when freed: otherwise the allocator can
# strand freed memory during long evaluations until the node runs out.
export MALLOC_MMAP_THRESHOLD_=1048576

IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:?No GPUs were allocated}"
cpus_per_worker=$(( SLURM_CPUS_PER_TASK / ${#gpus[@]} ))
for gpu in "${gpus[@]}"; do
    crun.python3 -p "$environment" python "$repository/scripts/cluster/plan_runs.py" work \
        "$queue_directory" --resources gpu --gpu "$gpu" \
        --cpus "$cpus_per_worker" &
done
wait
