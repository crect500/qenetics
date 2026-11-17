"""
Measure the interaction order of trained models of a `plan_runs.py submit`
output directory (`qenetics.tools.interactions`).

For each model and dataset, the run of seed 0 on the full training sets is
loaded from its record with its saved weights, and one of its outputs is
measured on real test sequences (three-way and pairwise epistasis of triples
of substitutions, in three strata) and on sequences with independent
positions (mean dimension):

- logit: the logit of its prediction, of the first cell and averaged over
  the cells, for models that end with a sigmoid;
- probability: its prediction, for the pure models, which output a measured
  probability;
- channel_window: one channel of its first layer at one window, before the
  affine map and ReLU, with triples inside that window;
- channel_sum: that channel summed over the windows.

The central CpG is never substituted. Models run in double precision where
they can. The global circuits with a dense head are contracted first: their
circuit, applied to the whole sequence at once, is a fixed matrix after
training, so each output is a matrix product instead of a circuit
simulation. Each loaded model is checked against the predictions its run
recorded for the first test sites, in the records' precision (float32), and
the double-precision model against the float32 one.

    python scripts/cluster/interaction_diagnostics.py /scratch/qcpg \\
        --datasets hepg2 mESC --device cuda -o diagnostics.tsv
"""

import math
import sys
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from qenetics.qcpg import model_factory, records
from qenetics.qcpg.sequence_model import SequenceModel
from qenetics.tools import data, interactions, tokenization

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_runs

LOGIT: str = "logit"
PROBABILITY: str = "probability"
CHANNEL_WINDOW: str = "channel_window"
CHANNEL_SUM: str = "channel_sum"
OUTPUTS: tuple[str, ...] = (LOGIT, PROBABILITY, CHANNEL_WINDOW, CHANNEL_SUM)
# The best window-average circuit of a dataset by validation score.
BEST_WINDOW_AVERAGE: str = "whole_model/WP_*"
# The models and outputs the gap plan (E17) examines, with the order-2 model
# of --order-2 first.
DEFAULT_MODELS: tuple[tuple[str, str], ...] = (
    ("whole_model/G1_lin_ry", LOGIT),
    (BEST_WINDOW_AVERAGE, PROBABILITY),
    ("deepcpg_tail/Q_L320", CHANNEL_WINDOW),
    ("deepcpg_tail/Q_L320", CHANNEL_SUM),
    ("deepcpg_tail/Q_L320", LOGIT),
    ("classical_tail/CNN", LOGIT),
    ("whole_model/FA_L24", PROBABILITY),
    ("tokens/K4_QE_L4", CHANNEL_WINDOW),
    ("tokens/K4_QE_L4", LOGIT),
    ("whole_model/H1b_cnnq_ry", PROBABILITY),
)
COLUMNS: tuple[str, ...] = (
    "dataset",
    "preset",
    "output",
    "component",
    "stratum",
    "sequences",
    "triples",
    "epistasis_3_median",
    "epistasis_3_maximum",
    "epistasis_2_median",
    "mean_dimension",
    "mean_dimension_standard_error",
    "dimension_sequences",
    "quantity",
    "precision",
    "contracted",
    "check_float32",
    "check_passed",
    "float64_minus_float32",
    "seconds",
    "record",
)
SequenceFunction = Callable[[Tensor], Tensor]
# A loaded model reproduces its run if its float32 predictions for the first
# test sites differ from the recorded ones by less than this. A loading
# error shows as differences near 1e-2.
CHECK_TOLERANCE: float = 1e-4


def centre_positions(sequence_length: int) -> tuple[int, int]:
    """The positions of the central C and G of a window."""
    return sequence_length // 2, sequence_length // 2 + 1


def token_window(sequence_length: int) -> tuple[int, int]:
    """
    The first k-mer token right of the CpG.

    Args
    ----
    sequence_length: The bp of the windows the tokens were made from.

    Returns
    -------
    The token's index among a token model's positions, and the position of
    its first base: the base after the central G.
    """
    return (
        tokenization.FLANK_TOKEN_QUANTITY + 1,
        centre_positions(sequence_length)[1] + 1,
    )


def model_logits(model: SequenceModel, channels: Tensor) -> Tensor:
    """
    The logits of a model that ends with a sigmoid.

    The model's final sigmoid is skipped rather than inverted: inverting it
    loses the precision of large logits, which shows as spurious epistasis.

    Args
    ----
    model: The model.
    channels: Its encoded inputs, (batch, channels, length).

    Returns
    -------
    The (batch, outputs) logits.

    Raises
    ------
    ValueError if the model does not apply exactly one sigmoid.
    """
    calls: list[Tensor] = []

    def identity(values: Tensor) -> Tensor:
        calls.append(values)
        return values

    originals = (torch.sigmoid, torch.nn.functional.sigmoid)
    torch.sigmoid = torch.nn.functional.sigmoid = identity
    try:
        logits: Tensor = model._probabilities(channels)
    finally:
        torch.sigmoid, torch.nn.functional.sigmoid = originals
    if len(calls) != 1 or calls[0] is not logits:
        raise ValueError(
            f"{type(model).__name__} does not end with one sigmoid; examine "
            f"its {PROBABILITY} instead"
        )
    return logits


def circuit_matrix(model: Any) -> Tensor:
    """
    The matrix of a global circuit model's trained circuit on its register.

    Returns
    -------
    M, (2^n, 2^n) with row k the state the circuit makes of basis state k:
    the state of normalized amplitudes a is a @ M.
    """
    import pennylane as qp

    from qenetics.circuits.ansatz import LayeredAnsatz

    dimension: int = 2**model.wire_quantity
    device = qp.device("default.qubit", wires=model.wire_quantity)
    weights: Tensor = model.circuit_weights.detach().to(torch.float64).cpu()

    @qp.qnode(device, interface="torch")
    def state(amplitudes: Tensor) -> Any:
        qp.AmplitudeEmbedding(amplitudes, wires=range(model.wire_quantity))
        LayeredAnsatz(weights, wires=model.circuit_wires, ansatz=model.ansatz)
        return qp.state()

    identity = torch.eye(dimension, dtype=torch.complex128)
    return torch.cat(
        [
            state(identity[start : start + 256])
            for start in range(0, dimension, 256)
        ]
    )


def is_contractible(model: SequenceModel) -> bool:
    """Whether a model is a global circuit with a dense head, without shots."""
    from qenetics.qcpg.hybrid_models import GlobalCircuitModel

    return (
        isinstance(model, GlobalCircuitModel)
        and model.head is not None
        and not model.shots
    )


def model_outputs(
    model: SequenceModel,
    channels: Tensor,
    *,
    logits: bool,
    matrix: Tensor | None = None,
) -> Tensor:
    """
    A model's (batch, cells) predictions or logits.

    Args
    ----
    model: The model, in evaluation mode.
    channels: Its encoded inputs.
    logits: Return logits (`model_logits`) rather than probabilities.
    matrix: For a contracted global circuit, its `circuit_matrix`.
    """
    if matrix is None:
        return (
            model_logits(model, channels)
            if logits
            else model._probabilities(channels)
        )
    features: list[Tensor] = []
    for shift in model.shifts:
        amplitudes = model._amplitudes(channels, shift).to(matrix.dtype)
        amplitudes = amplitudes / amplitudes.abs().pow(2).sum(1).sqrt()[:, None]
        probabilities = (amplitudes @ matrix).abs().pow(2)
        features.append(probabilities.to(channels.dtype) * model.feature_scale)
    values: Tensor = model.head(torch.cat(features, 1))
    return values if logits else torch.sigmoid(values)


def model_function(
    model: SequenceModel,
    output: str,
    *,
    input_name: str = model_factory.BASES_STR,
    channel: int = 0,
    window: int = 0,
) -> SequenceFunction:
    """
    One output of a model as a function of one-hot base sequences.

    Args
    ----
    model: The model, in evaluation mode.
    output: One of `OUTPUTS`.
    input_name: The model's input: bases, or CpG-anchored k-mers, which are
        tokenized from the bases at every call.
    channel: The first-layer channel of the channel outputs.
    window: The window (position of the first layer's output) of
        `CHANNEL_WINDOW`.

    Returns
    -------
    A function of (n, length, 4) sequences giving (n, 2) values, the first
    cell's and the mean over cells, for `LOGIT` and `PROBABILITY`, and (n, 1)
    values for the channel outputs.

    Raises
    ------
    ValueError if the output or the input is not supported.
    """
    if output not in OUTPUTS:
        raise ValueError(f"Unknown output {output}; expected one of {OUTPUTS}")
    kmer_length: int = 0
    if input_name != model_factory.BASES_STR:
        if not input_name.startswith(tokenization.CPG_KMER):
            raise ValueError(
                f"Only bases and k-mer inputs can be substituted, not {input_name}"
            )
        kmer_length = int(input_name.removeprefix(tokenization.CPG_KMER))
    parameter: Tensor = next(model.parameters())
    matrix: Tensor | None = None
    if is_contractible(model) and output in (LOGIT, PROBABILITY):
        matrix = circuit_matrix(model).to(parameter.device)

    def function(sequences: Tensor) -> Tensor:
        if kmer_length:
            codes = sequences.argmax(-1).cpu().numpy().astype(np.int8)
            inputs: Tensor = torch.from_numpy(
                tokenization.cpg_kmer_tokens(codes, kmer_length).astype(
                    np.int64
                )
            ).to(parameter.device)
        else:
            inputs = sequences.to(parameter)
        channels, _ = model.encoder(inputs)
        if output in (CHANNEL_WINDOW, CHANNEL_SUM):
            first_layer: Tensor = model.front_end(channels)[:, channel]
            values = (
                first_layer[:, window : window + 1]
                if output == CHANNEL_WINDOW
                else first_layer.sum(-1, keepdim=True)
            )
            return values.double().cpu()
        values = model_outputs(
            model, channels, logits=output == LOGIT, matrix=matrix
        ).double()
        return torch.stack([values[:, 0], values.mean(1)], 1).cpu()

    return function


def base_frequencies(sequences: Tensor) -> Tensor:
    """
    The base frequencies of each position, with the central CpG fixed.

    Args
    ----
    sequences: (n, length, 4) one-hot sequences, channels A, T, G, C.

    Returns
    -------
    The (length, 4) frequencies; the centre rows hold C and G only.
    """
    frequencies: Tensor = sequences.double().mean(0)
    frequencies = frequencies / frequencies.sum(1, keepdim=True)
    cytosine, guanine = centre_positions(len(frequencies))
    frequencies[cytosine] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    frequencies[guanine] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    return frequencies


def measure(
    function: SequenceFunction,
    output: str,
    sequences: Tensor,
    frequencies: Tensor,
    generator: torch.Generator,
    *,
    triple_quantity: int,
    dimension_sequence_quantity: int,
    span: tuple[int, int] | None = None,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """
    Measure one output's epistasis and mean dimension.

    Args
    ----
    function: The output, see `model_function`.
    output: Its kind, one of `OUTPUTS`.
    sequences: (n, length, 4) real one-hot sequences for the epistasis.
    frequencies: (length, 4) base frequencies for the mean dimension.
    generator: The random generator.
    triple_quantity: The triples per sequence and stratum.
    dimension_sequence_quantity: The sequences of the mean dimension, which
        is not measured for `CHANNEL_WINDOW` or if this is 0.
    span: For `CHANNEL_WINDOW`, the first position and length of the window
        the channel reads; its triples lie inside it.
    batch_size: The sequences per call of the function.

    Returns
    -------
    One row per component of the output and stratum, see `COLUMNS`.
    """
    sequences = sequences.double()
    excluded = centre_positions(sequences.shape[1])
    strata: dict[str, tuple[Tensor, Tensor]] = {}
    if output == CHANNEL_WINDOW:
        if span is None:
            raise ValueError("A window's channel needs the window's span")
        start, length = span
        positions, substitutes = interactions.draw_triples(
            sequences[:, start : start + length],
            triple_quantity,
            interactions.ANYWHERE_STRATUM,
            generator,
            [p - start for p in excluded if start <= p < start + length],
        )
        strata[interactions.WINDOW_STRATUM] = (positions + start, substitutes)
    else:
        for stratum in interactions.STRATUM_SPANS:
            strata[stratum] = interactions.draw_triples(
                sequences, triple_quantity, stratum, generator, excluded
            )
    dimension: interactions.MeanDimension | None = None
    if output != CHANNEL_WINDOW and dimension_sequence_quantity:
        dimension = interactions.mean_dimension(
            function,
            frequencies.double(),
            dimension_sequence_quantity,
            generator,
            batch_size=batch_size,
        )
    components: list[str] = (
        ["first_cell", "mean_over_cells"]
        if output in (LOGIT, PROBABILITY)
        else ["channel"]
    )
    rows: list[dict[str, Any]] = []
    for stratum, (positions, substitutes) in strata.items():
        epistasis = interactions.triple_epistasis(
            function, sequences, positions, substitutes, batch_size
        )
        for index, component in enumerate(components):
            three_way = epistasis.three_way[..., index].abs()
            rows.append(
                {
                    "output": output,
                    "component": component,
                    "stratum": stratum,
                    "sequences": len(sequences),
                    "triples": triple_quantity,
                    "epistasis_3_median": float(three_way.median()),
                    "epistasis_3_maximum": float(three_way.max()),
                    "epistasis_2_median": float(
                        epistasis.pairwise[..., index].abs().median()
                    ),
                    "mean_dimension": float(dimension.value[index])
                    if dimension
                    else math.nan,
                    "mean_dimension_standard_error": float(
                        dimension.standard_error[index]
                    )
                    if dimension
                    else math.nan,
                    "dimension_sequences": dimension.sequence_quantity
                    if dimension
                    else 0,
                }
            )
    return rows


def load_model(
    preset: model_factory.Preset, record: Path, device: torch.device
) -> SequenceModel:
    """
    Build a run's model as `plan_runs.preset_size` does, with its saved
    weights, in evaluation mode.

    Raises
    ------
    OSError or KeyError if the run's data or weights cannot be read.
    """
    parameters: dict[str, Any] = records.read_run(record)["parameters"]
    sequence_length, cell_quantity, token_attributes = data.read_h5_layout(
        Path(parameters["data_directory"])
        / f"chr{parameters['training_chromosomes'][0]}.h5"
    )
    encoding_options: dict[str, Any] = {}
    if token_attributes:
        encoding_options = {
            "encoding": data.TOKEN_ENCODING_STR,
            "token_lookup": tokenization.token_lookup(
                str(token_attributes[data.TOKENIZATION_ATTRIBUTE]),
                int(token_attributes[data.VOCABULARY_SIZE_ATTRIBUTE]),
            ),
        }
    model = model_factory.build_model(
        preset.model,
        sequence_length,
        cell_quantity,
        options=preset.options,
        **encoding_options,
    )
    model.load_state_dict(
        torch.load(record / records.MODEL_FILENAME, map_location="cpu")
    )
    return model.to(device).eval()


def predictions(
    model: SequenceModel,
    preset: model_factory.Preset,
    sequences: Tensor,
    *,
    contracted: bool,
) -> Tensor:
    """A model's (n, cells) predictions for one-hot sequences, in float64."""
    parameter: Tensor = next(model.parameters())
    matrix = circuit_matrix(model).to(parameter.device) if contracted else None
    with torch.no_grad():
        channels, _ = model.encoder(_model_inputs(sequences, preset, parameter))
        return (
            model_outputs(model, channels, logits=False, matrix=matrix)
            .double()
            .cpu()
        )


def recorded_predictions(record: Path, quantity: int) -> Tensor | None:
    """The first predictions a run recorded for its test sites."""
    outputs = records.read_split_outputs(record, records.TEST_SPLIT)
    if outputs is None:
        return None
    return torch.from_numpy(
        np.asarray(outputs.predictions[:quantity], np.float64)
    )


def _model_inputs(
    sequences: Tensor, preset: model_factory.Preset, parameter: Tensor
) -> Tensor:
    if preset.input == model_factory.BASES_STR:
        return sequences.to(parameter)
    kmer_length = int(preset.input.removeprefix(tokenization.CPG_KMER))
    codes = sequences.argmax(-1).cpu().numpy().astype(np.int8)
    return torch.from_numpy(
        tokenization.cpg_kmer_tokens(codes, kmer_length).astype(np.int64)
    ).to(parameter.device)


def _resolve_preset(
    pattern: str, dataset: str, rows: list[dict[str, Any]], experiment: str
) -> str | None:
    """The preset of a pattern with the best mean validation score."""
    scores: dict[str, list[float]] = {}
    for row in rows:
        if (
            row["experiment"] == experiment
            and row["dataset"] == dataset
            and plan_runs.fnmatchcase(row["preset"], pattern)
            and plan_runs._usable(row)
        ):
            scores.setdefault(row["preset"], []).append(
                row["best_validation_score"]
            )
    if not scores:
        return None
    return max(scores, key=lambda name: float(np.mean(scores[name])))


def _run_record(
    preset: str, dataset: str, rows: list[dict[str, Any]], experiment: str
) -> Path | None:
    """The record of a preset's usable run with the lowest seed."""
    usable = [
        row
        for row in rows
        if row["experiment"] == experiment
        and row["dataset"] == dataset
        and row["preset"] == preset
        and plan_runs._usable(row)
    ]
    if not usable:
        return None
    return Path(min(usable, key=lambda row: int(row["seed"]))["record"])


def _test_sequences(record: Path, quantity: int) -> Tensor:
    """The first sites of a run's first test chromosome, as one-hot bases."""
    parameters: dict[str, Any] = records.read_run(record)["parameters"]
    dataset = data.QuantumTorchDataset(
        [
            Path(parameters["data_directory"])
            / f"chr{parameters['test_chromosomes'][0]}.h5"
        ],
        encoding=data.ONEHOT_ENCODING_STR,
    )
    return dataset[list(range(min(quantity, len(dataset))))][0].double()


def _parse_script_args(arguments: list[str] | None) -> Namespace:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "output_directory", type=Path, help="The output directory of submit."
    )
    parser.add_argument("--datasets", nargs="+", default=["hepg2", "mESC"])
    parser.add_argument(
        "--order-2",
        default="classical_whole/FacM_r8",
        help="The order-2 model whose logit sets the numerical noise floor.",
    )
    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("PRESET", "OUTPUT"),
        help=f"A model and one of its outputs ({', '.join(OUTPUTS)}), instead "
        "of the default list. Repeat for more.",
    )
    parser.add_argument("--experiment", default="full")
    parser.add_argument("--sequences", type=int, default=200)
    parser.add_argument("--triples", type=int, default=200)
    parser.add_argument(
        "--dimension-sequences",
        type=int,
        default=500,
        help="The sequences of the mean dimension, each evaluated with every "
        "base at every position.",
    )
    parser.add_argument(
        "--slow-dimension-sequences",
        type=int,
        default=50,
        help="The same for models that run circuits as QNodes.",
    )
    parser.add_argument(
        "--frequency-sequences",
        type=int,
        default=20000,
        help="The test sequences that give the base frequencies.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--presets-file", type=Path, default=None)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("interaction_diagnostics.tsv")
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args: Namespace = _parse_script_args(arguments)
    presets = model_factory.load_presets(
        args.presets_file or plan_runs.PRESETS_FILEPATH
    )
    rows = plan_runs.summarize_runs(args.output_directory.resolve())
    models: list[tuple[str, str]] = (
        [tuple(pair) for pair in args.model]
        if args.model
        else [(args.order_2, LOGIT), *DEFAULT_MODELS]
    )
    device = torch.device(args.device)
    results: list[dict[str, Any]] = []
    for dataset in args.datasets:
        base_record = _run_record(
            "classical_tail/CNN", dataset, rows, args.experiment
        )
        if base_record is None:
            print(f"No run of classical_tail/CNN on {dataset}; skipping it")
            continue
        test_sequences = _test_sequences(base_record, args.frequency_sequences)
        frequencies = base_frequencies(test_sequences)
        for pattern, output in models:
            name = _resolve_preset(pattern, dataset, rows, args.experiment)
            record = name and _run_record(name, dataset, rows, args.experiment)
            if not name or not record:
                print(f"No usable run of {pattern} on {dataset}; skipping it")
                continue
            preset = presets[name]
            started = time.perf_counter()
            sample = test_sequences[:256]
            recorded = recorded_predictions(record, len(sample))
            model = load_model(preset, record, device)
            contracted = is_contractible(model)
            single = predictions(model, preset, sample, contracted=False)
            check = (
                float((single - recorded).abs().max())
                if recorded is not None
                else math.nan
            )
            precision = "float64"
            try:
                model.double()
                double = predictions(
                    model, preset, sample, contracted=contracted
                )
            except (RuntimeError, TypeError) as error:
                print(f"{name} does not run in double precision: {error}")
                model = load_model(preset, record, device)
                precision = "float32"
                double = predictions(
                    model, preset, sample, contracted=contracted
                )
            difference = float((double - single).abs().max())
            if not check < CHECK_TOLERANCE:
                print(
                    f"{name} on {dataset} does not reproduce its recorded "
                    f"predictions (largest difference {check:.2g}); its rows "
                    "should not be used",
                    flush=True,
                )
            span: tuple[int, int] | None = None
            window = 0
            if output == CHANNEL_WINDOW:
                if preset.input == model_factory.BASES_STR:
                    window_length = int(
                        preset.options.get("window_lengths", [16])[0]
                    )
                    # A window left of the CpG, clear of the central bases.
                    window = test_sequences.shape[1] // 2 - 2 * window_length
                    span = (window, window_length)
                else:
                    kmer_length = int(
                        preset.input.removeprefix(tokenization.CPG_KMER)
                    )
                    window, first_base = token_window(test_sequences.shape[1])
                    span = (first_base, kmer_length)
            generator = torch.Generator().manual_seed(args.seed)
            measured = measure(
                model_function(
                    model, output, input_name=preset.input, window=window
                ),
                output,
                test_sequences[: args.sequences],
                frequencies,
                generator,
                triple_quantity=args.triples,
                dimension_sequence_quantity=args.slow_dimension_sequences
                if model_factory.uses_device(preset.model) and not contracted
                else args.dimension_sequences,
                span=span,
                batch_size=args.batch_size,
            )
            seconds = time.perf_counter() - started
            for row in measured:
                results.append(
                    {
                        **row,
                        "dataset": dataset,
                        "preset": name,
                        "quantity": "logit"
                        if output == LOGIT
                        else "probability"
                        if output == PROBABILITY
                        else "first-layer channel",
                        "precision": precision,
                        "contracted": contracted
                        and output in (LOGIT, PROBABILITY),
                        "check_float32": check,
                        "check_passed": check < CHECK_TOLERANCE,
                        "float64_minus_float32": difference,
                        "seconds": seconds,
                        "record": str(record),
                    }
                )
            print(
                f"{dataset} {name} {output}: "
                + "; ".join(
                    f"{row['component']} {row['stratum']} "
                    f"e3 {row['epistasis_3_median']:.3g} "
                    f"(max {row['epistasis_3_maximum']:.3g}), "
                    f"e2 {row['epistasis_2_median']:.3g}"
                    for row in measured
                )
                + f"; mean dimension {measured[0]['mean_dimension']:.3f}"
                f"; float32 check {check:.2g}, float64 - float32 "
                f"{difference:.2g}; {seconds:.0f} s",
                flush=True,
            )
            plan_runs._write_tsv(args.output, COLUMNS, results)
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
