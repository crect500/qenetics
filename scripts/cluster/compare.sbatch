#!/bin/bash
# The bootstrap comparisons of a `plan_runs.py submit` output directory, in
# the variants of the gap plan: single sites (with the ceilings), the same
# without the 50-epoch and refit runs in the pool, the setting-wide token
# pool with the folds' own seed variance, and genomic blocks of 1 Mb and of
# 100 kb. Run from the repository:
#     sbatch scripts/cluster/compare.sbatch /scratch/qcpg
# Adjust the resources to your cluster.
#SBATCH --job-name=qcpg-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-00:00:00

# The cluster's Python: load its container module, then run Python in the
# environment with qenetics installed (QENETICS_ENV). The module scripts are
# not written for `set -u`, so they run before it.
enable_lmod
module load container_env python3/2025
environment="${QENETICS_ENV:-$HOME/envs/pennylane-lightning}"

set -euo pipefail

output_directory="$1"
repository="${QENETICS_REPOSITORY:-$SLURM_SUBMIT_DIR}"
export MALLOC_MMAP_THRESHOLD_=1048576

# Each ceiling against DeepCpG's CNN, and the deepest bank against the
# ceilings of its window.
pairs=()
for ceiling in ceiling/U64 ceiling/O64 ceiling/U256 ceiling/O256 \
    classical_tail/FC64 classical_tail/FR64 classical_tail/FC256 \
    classical_tail/FR256; do
    pairs+=(--pair "$ceiling" classical_tail/CNN)
done
pairs+=(--pair deepcpg_tail/Q_L320 ceiling/U64 --pair deepcpg_tail/Q_L320 ceiling/O64)

compare() {
    crun.python3 -p "$environment" python "$repository/scripts/cluster/plan_runs.py" \
        compare "$output_directory" --processes "$SLURM_CPUS_PER_TASK" \
        --match-ceilings "${pairs[@]}" "$@"
}

compare
compare --name comparisons_original_pool --no-substitutes
compare --name comparisons_setting_pool --token-pool setting \
    --single-seed-variance same-models
compare --name comparisons_block_1mb --block-size 1000000
compare --name comparisons_block_100kb --block-size 100000
