#!/bin/bash
# One whole CPU node of a `plan_runs.py submit` queue: workers that
# share its CPUs take entries of the queue until none is left. The second
# argument is the number of workers.
# Adjust the resources to your cluster, or override them with --sbatch-args.
#SBATCH --job-name=qcpg-pool-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=0

set -euo pipefail

queue_directory="$1"
worker_quantity="${2:-4}"
repository="${QENETICS_REPOSITORY:?Submit with plan_runs.py, which sets QENETICS_REPOSITORY}"
# The Python environment with qenetics installed; load modules here if needed.
source "${QENETICS_ENV:-$repository/.venv}/bin/activate"

cpus_per_worker=$(( SLURM_CPUS_PER_TASK / worker_quantity ))
for (( worker = 0; worker < worker_quantity; worker++ )); do
    python "$repository/scripts/cluster/plan_runs.py" work \
        "$queue_directory" --resources cpu --cpus "$cpus_per_worker" &
done
wait
