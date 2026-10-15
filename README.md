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

`config/quantum_models.json` holds every pure and hybrid model of the report
as a preset, `<section>/<report name>`, with its options and the training
settings of its main results table. List them with
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
`train_gpu.sbatch` to the cluster; `QENETICS_ENV` names the Python
environment, by default `.venv` in the repository. `--dry-run` writes the run
list and prints the sbatch command only.

`scripts/cluster/plan_runs.py` runs the quantum experiments of the training
plan: every preset without a full-dataset run on the full training sets
(`full`), the wide-bank control (`E4`), chromosome cross-validation (`E12`)
and widened 2i and serum training sets (`E13`) of Q_L320, and the global
circuits (`Q1`). Runs keep the epoch of the best mean per-cell validation
AUC, train for at most 25 epochs, and a run that collapses is rerun with its
seed + 10. `submit` queues the runs that have not completed and submits
whole-node jobs, at most `--gpu-nodes` and `--cpu-nodes` at a time
(`pool_gpu.sbatch`, `pool_cpu.sbatch`), whose workers take runs from the
queue; it also writes the token inputs the token presets read. Submitting
again resumes the plan.

```bash
python scripts/cluster/plan_runs.py submit --data-directories /data/2i \
    /data/hcc /data/hepg2 /data/ser /data/mESC --output-directory /scratch/q
python scripts/cluster/plan_runs.py status /scratch/q
python scripts/build_training_report.py -r /scratch/q/full/records
```

## Tests

```bash
pytest tests
```
