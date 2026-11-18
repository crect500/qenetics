#!/bin/bash
# One GPU of a `plan_runs.py submit` queue on the high-memory GPU partition
# (an A100): one worker takes entries of the queue until none is left, as
# a worker of pool_gpu.sbatch does. Each high-gpu-mem node has 64 cores,
# 512 GB and 2 A100s, so one A100 gets half of the node.
# Adjust the resources to your cluster, or override them with --sbatch-args.
#SBATCH --job-name=qcpg-pool-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G

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

gpu="${CUDA_VISIBLE_DEVICES:?No GPU was allocated}"
crun.python3 -p "$environment" python "$repository/scripts/cluster/plan_runs.py" work \
    "$queue_directory" --resources gpu --gpu "${gpu%%,*}" --cpus "$SLURM_CPUS_PER_TASK"
