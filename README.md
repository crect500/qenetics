# qenetics

Quantum and hybrid quantum-classical models that predict single-cell CpG
methylation from DNA sequence, implemented with PennyLane and PyTorch.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # .venv/Scripts/activate on Windows
pip install -e ".[dev]"            # add ",gpu" for lightning.gpu
```

## Data

Models read one H5 file per chromosome, `chr<name>.h5`, holding a sequence
window per CpG site with the C at position (length - 1) // 2, and the
methylation label of each cell. Create them from methylation profiles with
`scripts/create_h5_samples.py`, or from DeepCpG data files with
`scripts/create_h5_samples_from_deepcpg.py`.

The token models take CpG-centred tokens: each flank is read away from the
CpG, the left one reverse-complemented, and tokenized. Write token files with
`scripts/tokenize_cpg_windows.py`:

```bash
# BPE learned on the training chromosomes; tokenizer.json is saved with it.
python scripts/tokenize_cpg_windows.py -i hcc -o hcc_bpe64 --method cpg_bpe \
    --vocabulary-size 64 --training-chromosomes 1 3 5 7 9
python scripts/tokenize_cpg_windows.py -i hcc -o hcc_kmer3 --method cpg_kmer \
    --kmer-length 3
# Base-aligned BPE tokens, alone or with the bases.
python scripts/tokenize_cpg_windows.py -i hcc -o hcc_painted_bases64 \
    --method painted_bases --vocabulary-size 64 \
    --tokenizer hcc_bpe64/tokenizer.json
```

## Models

Circuits are PennyLane operations (`qenetics.circuits.ansatz.LayeredAnsatz`)
that decompose into `StronglyEntanglingLayers`, `BasicEntanglerLayers` or
RY/RZ and CNOT gates. They are simulated exactly. Circuits applied to every
window of a sequence act linearly on the encoded window, so their unitary is
built once per batch and applied as a convolution. Circuits whose input
changes per sample run as QNodes on the device given by `--device-name`.

| family | models of the report |
|---|---|
| `filter_bank_network` | quantum filter banks with the DeepCpG tail or a pooling head (Q\_\*, R\_\*, MK\*, MS4\*), token embeddings and token windows (T\*, K\*, P\*), Series I Q1, QA\*, QB\*, QS\*, K1, FQ3 |
| `window_average` | pure window-averaged circuits (WP\_\*, WT64\*) |
| `feature_amplitude` | pure classifier on 44 amplitude-encoded sequence statistics (FA\_\*) |
| `feature_reuploading` | pure classifier on re-uploaded sequence statistics (Series I P5) |
| `automaton` | pure quantum automata (AU\_\*, Series I A1, A2) |
| `global_circuit` | the whole sequence in one 12-qubit circuit, pure or with a dense head (GP\_\*, G\*, Series I L1, L2) |
| `cnn_quantum_head` | DeepCpG's CNN with a re-uploading output circuit (H1\*) |
| `classical_network` | classical: a classical first layer before DeepCpG's tail or the pooling head: convolutions (CNN, C\_\*, CNN2), the raw window (ID\_\*), low-rank (LR\_\*), Givens (G\_\*), multi-scale and quadrature-pair (MSC\_\*, Cq\_\*) and dense-matrix (FC\*, FR\*) layers, and the ceilings U64, O64, U256, O256 |
| `feature_classifier` | classical: the 44 sequence statistics into a linear layer or an MLP (FL\_lin, FM\_4) |
| `one_hot_classifier` | classical: the flattened sequence into a linear layer or an MLP (OH\_\*) |
| `token_frequency_classifier` | classical: logistic regression on token frequencies (TF64\_LR) |
| `factorization_machine` | classical: linear terms and rank-r pairwise interactions of the one-hot sequence (FacM\_r\*) |

`config/quantum_models.json` holds every pure and hybrid model of the report,
and the classical models of its matched comparisons (`classical_tail/`,
`classical_pooling/`, `classical_whole/`, `classical_tokens/`) and the
ceilings of its filter banks (`ceiling/`), as a preset,
`<section>/<report name>`, with its options and the training settings of its
main results table. List them with
`python scripts/train_quantum_model.py --list-presets`.

## Training

```bash
python scripts/train_quantum_model.py --preset deepcpg_tail/Q_L4 \
    -d hcc -o runs/hcc/Q_L4 --training-chromosomes 1 3 5 7 9 \
    --validation-chromosomes 13 --test-chromosomes 2 4 --seed 0
```

Arguments override the preset: `--quantum-layers`, `--hidden-layer-sizes`
(the dense layers), `--ansatz`, any option as `--option key=value`, and every
training setting. A model family can be configured from scratch with
`--model`. `--describe` prints the model's size for the data without training.

- *Devices:* default.qubit is the fastest for every model and runs on a GPU
  with `--gpus 1`. lightning.qubit and lightning.gpu differentiate expectation
  readouts by the adjoint method, and probability readouts (the dense head of
  `global_circuit`) only by parameter shifts, which is slow.
- *Finite measurements:* `--shots 100 1000` also evaluates the test set with
  every measured probability estimated from that many shots.

Each run is recorded in the records directory and added to its HTML report.

## Cluster (SLURM)

`scripts/cluster/slurm_runs.py submit` runs presets, datasets and seeds as a
job array, one run per task with its own output directory, on CPU nodes or
one GPU per task:

```bash
python scripts/cluster/slurm_runs.py submit --presets 'deepcpg_tail/*' \
    --data-directories /data/hcc --seeds 0 1 2 --output-directory /scratch/q \
    --resources gpu --max-concurrent 8 --sbatch-args='--partition=gpu' \
    -- --training-chromosomes 1 3 5 7 9 --validation-chromosomes 13 \
    --test-chromosomes 2 4
python scripts/build_training_report.py -r /scratch/q/records
```

Adjust the resources and environment of `train_cpu.sbatch` and
`train_gpu.sbatch` to the cluster. They load the cluster's container module
(`enable_lmod`, `module load container_env python3/2025`) and run Python with
`crun.python3 -p` in the environment `QENETICS_ENV`, by default
`~/envs/pennylane-lightning`, which must have qenetics installed. `--max-concurrent` caps
the array tasks at once, by default at the cluster's per-user limit of 16
nodes (80 eight-CPU tasks, or 16 one-GPU tasks). `--dry-run` writes the run
list and prints the sbatch command only.

`scripts/cluster/plan_runs.py` runs the experiments of the training plan with
the presets:

- `full`: every preset, quantum, classical and ceiling, on the full training
  sets;
- `E4`: angle-matched real wide banks; `Q1`: the real global circuits, both
  on split A with 200k sites;
- `E12`: chromosome cross-validation of Q_L320 and the classical references
  (DeepCpG's CNN, C_w4_170, C_w8_90, U64); `E13`: Q_L320 on widened 2i and
  serum training sets;
- `LR1e-2` to `LR3e-4`: the factorization machines' learning-rate sweep on the
  full training sets of HCC and 2i; their full runs wait for it and use its
  best rate;
- `splitC`: the best quantum models and DeepCpG's CNN on DeepCpG's split.

Runs keep the epoch of the best mean per-cell validation AUC, train for at
most 25 epochs, and a run that collapses is rerun with its seed + 10.
`submit` queues the runs that have not completed and submits whole-node jobs,
at most `--gpu-nodes` (4) and `--cpu-nodes` at a time, 16 nodes in all: the
cluster's per-user limit allows 12 CPU nodes beside 4 GPU nodes, or 16 CPU
nodes without them, and `--cpu-nodes` defaults to the rest (`pool_gpu.sbatch`,
`pool_cpu.sbatch`), whose workers take runs from the queue; it also writes
the token inputs the token presets read. Submitting again resumes the plan,
and `--close-gaps` adds seeds 3 and 4 to every quantum model within 0.005 of
a classical model it is compared with, and to that model.

`status` shows a queue's progress and why runs failed. `summarize` writes one
row per recorded run, with its best epoch and test AUCs, to `summary.tsv`.
`compare` bootstraps (`qenetics.tools.bootstrap`, paired over test sites and
seeds, with Benjamini–Hochberg q values) the matched comparisons of the full
training sets: every quantum model against the classical model of its kind
with the best mean validation score among those with at most 1.25 times its
trained parameters, counted on the first layer (`matched_front_end.tsv`) and
on the whole model (`matched_total.tsv`). It also compares the complex
circuits with their real twins (`config/complex_real_pairs.json`) and, in the
other experiments, every quantum model with each classical model beside it,
and writes every run's AUPRC and Matthews correlation, into `comparisons/`.

```bash
python scripts/cluster/plan_runs.py submit --data-directories /data/2i \
    /data/hcc /data/hepg2 /data/ser /data/mESC --output-directory /scratch/q
python scripts/cluster/plan_runs.py status /scratch/q
python scripts/cluster/plan_runs.py summarize /scratch/q
python scripts/cluster/plan_runs.py compare /scratch/q --processes 40
python scripts/build_training_report.py -r /scratch/q/full/records
```

## Tests

```bash
pytest tests
```
