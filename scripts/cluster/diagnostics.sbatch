#!/bin/bash
# The interaction-order diagnostics of trained models of a
# `plan_runs.py submit` output directory, on one GPU. Further arguments go to
# interaction_diagnostics.py. Run from the repository:
#     sbatch scripts/cluster/diagnostics.sbatch /scratch/qcpg -o diagnostics.tsv
# Adjust the resources to your cluster.
#SBATCH --job-name=qcpg-diagnostics
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# The cluster's Python: load its container module, then run Python in the
# environment with qenetics installed (QENETICS_ENV). The module scripts are
# not written for `set -u`, so they run before it.
enable_lmod
module load container_env python3/2025
environment="${QENETICS_ENV:-$HOME/envs/pennylane-lightning}"

set -euo pipefail

repository="${QENETICS_REPOSITORY:-$SLURM_SUBMIT_DIR}"
crun.python3 -p "$environment" python \
    "$repository/scripts/cluster/interaction_diagnostics.py" "$@" --device cuda
