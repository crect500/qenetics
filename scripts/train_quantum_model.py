"""
Train a quantum or hybrid methylation model of the quantum models report.

Choose a preset of config/quantum_models.json (see --list-presets), whose
model, options and training settings any argument here overrides, or a model
family with its options. Runs are recorded as by train_qcpg_model.py.
"""

import json
import logging
import os
from argparse import ArgumentParser, Namespace
from dataclasses import fields
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from qenetics.circuits import ansatz, device_utils
from qenetics.qcpg import model_factory, qcpg
from qenetics.tools import data

PRESETS_FILEPATH: Path = (
    Path(__file__).resolve().parents[1] / "config" / "quantum_models.json"
)
# Arguments that set training parameters, and the parameters they set.
_TRAINING_ARGUMENTS: dict[str, str] = {
    "epochs": "epochs",
    "batch_size": "batch_size",
    "learning_rate": "learning_rate",
    "learning_rate_decay": "learning_rate_decay",
    "quantum_learning_rate": "quantum_learning_rate",
    "selection_metric": "selection_metric",
    "patience": "patience",
    "minimum_improvement": "minimum_improvement",
    "training_samples": "training_sample_quantity",
}
# Arguments that set model options, and the options they set.
_OPTION_ARGUMENTS: dict[str, str] = {
    "quantum_layers": "quantum_layer_quantity",
    "hidden_layer_sizes": "hidden_layer_sizes",
    "ansatz": "ansatz",
}

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """
    Converts a string to an integer, if positive.

    Args
    ----
    value: the string to convert

    Returns
    -------
    The integer.

    Raises
    ------
    ValueError if `value` is not positive.

    """
    number = int(value)
    if number < 1:
        raise ValueError(f"{value} is not positive")
    return number


def _parse_script_args(arguments: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        "train_quantum_model",
        description="Train a quantum or hybrid model that predicts the "
        "methylation of CpG sites from their sequence.",
    )
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--preset",
        help="A model of the quantum models report, e.g. deepcpg_tail/Q_L4.",
    )
    model.add_argument(
        "--model",
        choices=list(model_factory.MODEL_FAMILIES),
        help="A model family, configured by --option and the model flags.",
    )
    parser.add_argument(
        "--presets-file",
        type=Path,
        default=PRESETS_FILEPATH,
        help="The presets file.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List the presets and exit.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Build the model for the data, print its size and exit.",
    )
    parser.add_argument(
        "--option",
        dest="options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="A model option, with a JSON value or a string, e.g. "
        "window_lengths=[4,8] or tail=pooling. Repeat for more.",
    )
    parser.add_argument(
        "--quantum-layers",
        type=_positive_int,
        help="The number of layers of each circuit (block).",
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        type=_positive_int,
        nargs="*",
        help="The sizes of the dense (MLP) layers; none for a linear layer.",
    )
    parser.add_argument(
        "--ansatz",
        choices=ansatz.ANSATZES_STRINGS,
        help="The circuits' ansatz.",
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        type=Path,
        help="The directory of chr<name>.h5 methylation files.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        help="The directory for the trained model and logs.",
    )
    parser.add_argument(
        "--training-chromosomes",
        nargs="+",
        default=["1", "3", "5", "7", "9", "11"],
        help="The chromosomes to train on.",
    )
    parser.add_argument(
        "--validation-chromosomes",
        nargs="+",
        default=None,
        help="The chromosomes to select epochs on. Defaults to every "
        "chromosome file not used for training or testing.",
    )
    parser.add_argument(
        "--test-chromosomes",
        nargs="*",
        default=["2", "4", "6", "8", "10", "12"],
        help="The chromosomes to test on; none to skip testing.",
    )
    parser.add_argument(
        "--experiment-name",
        help="The dataset name of the run's record. Defaults to the data "
        "directory's name.",
    )
    parser.add_argument(
        "--records-directory",
        type=Path,
        help="The directory of run records. Defaults to 'records' in the "
        "output directory.",
    )
    parser.add_argument(
        "--no-report",
        dest="update_report",
        action="store_false",
        help="Record the run without adding it to the HTML report.",
    )
    parser.add_argument(
        "--encoding",
        choices=[
            data.ONEHOT_ENCODING_STR,
            data.TOKEN_ENCODING_STR,
            data.BPE_ENCODING_STR,
        ],
        help="How to read sequences. Defaults to token for tokenized files, "
        "else onehot.",
    )
    parser.add_argument(
        "--tokenizer",
        help="The HuggingFace tokenizer of the bpe encoding.",
    )
    parser.add_argument(
        "--vocabulary-size",
        type=_positive_int,
        help="The token ids of token data without tokenization attributes.",
    )
    parser.add_argument(
        "--padding-token",
        type=int,
        help="A token id that encodes as zeros.",
    )
    parser.add_argument(
        "--device-name",
        choices=device_utils.DEVICES,
        default=device_utils.DEFAULT_DEVICE,
        help="The PennyLane device of circuits run per sample.",
    )
    parser.add_argument(
        "--diff-method",
        default="best",
        help="The PennyLane differentiation method of those circuits.",
    )
    parser.add_argument(
        "--gpus",
        dest="gpu_quantity",
        type=int,
        default=0,
        help="The number of GPUs; 0 trains on CPUs.",
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        help="CPU threads. Defaults to SLURM_CPUS_PER_TASK if set.",
    )
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--learning-rate-decay",
        type=float,
        help="The factor applied to the learning rate after every epoch.",
    )
    parser.add_argument(
        "--quantum-learning-rate",
        type=float,
        help="The learning rate of circuit angles.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=qcpg.SELECTION_METRICS,
        help="Keep the epoch with the best validation score by this metric.",
    )
    parser.add_argument(
        "--patience",
        type=_positive_int,
        help="Stop after this many epochs without validation gain.",
    )
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        help="The validation gain that resets patience.",
    )
    sample_quantity = parser.add_mutually_exclusive_group()
    sample_quantity.add_argument(
        "--training-samples",
        type=_positive_int,
        help="Train on a fixed random subset of this many sites.",
    )
    sample_quantity.add_argument(
        "--all-training-samples",
        action="store_true",
        help="Train on every site, overriding a preset's subset.",
    )
    parser.add_argument("--seed", type=int, help="The random seed.")
    parser.add_argument(
        "--shots",
        type=_positive_int,
        nargs="+",
        default=[],
        help="Also evaluate the test set with probabilities estimated from "
        "each number of measurements.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="Processes that load data.",
    )
    parser.add_argument(
        "--log-directory",
        type=Path,
        help="The log directory. Defaults to the output directory.",
    )
    parser.add_argument(
        "--log-level", choices=["debug", "info"], default="info"
    )
    args: Namespace = parser.parse_args(arguments)
    if not args.list_presets:
        missing: list[str] = [
            flag
            for flag, value in (
                ("--preset or --model", args.preset or args.model),
                ("--data-directory", args.data_directory),
                ("--output-directory", args.output_directory or args.describe),
            )
            if not value
        ]
        if missing:
            parser.error(
                f"the following arguments are required: {', '.join(missing)}"
            )
    return args


def _parse_options(assignments: list[str]) -> dict[str, Any]:
    """
    Parse KEY=VALUE model options, reading values as JSON if they are.

    Args
    ----
    assignments: The options passed to the parser.

    Returns
    -------
    The key-value pairs for each option.

    Raises
    ------
    ValueError if an assignment has no key.
    """
    options: dict[str, Any] = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator or not key:
            raise ValueError(f"Option '{assignment}' is not KEY=VALUE")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _find_preset(
    presets: dict[str, model_factory.Preset], name: str
) -> model_factory.Preset:
    """
    Returns the preset model matching the key.

    Args
    ----
    presets: The established preset models.
    name: The preset name provided by the user.

    Returns
    -------
    The selected model class.

    Raises
    ------
    ValueError if the preset does not exist.
    """
    if name in presets:
        return presets[name]

    suggestions: list[str] = get_close_matches(name, presets, n=3)
    raise ValueError(
        f"Unknown preset {name}"
        + (f"; did you mean {', '.join(suggestions)}?" if suggestions else "")
    )


def _first_training_filepath(args: Namespace) -> Path:
    """
    Returns the filepath corresponding to the first training chromosome specified by the user.

    Args
    ----
    args: The parsed ArgumentParser arguments.

    Returns
    ------
    The filepath corresponding to the first training chromosome specified by the user.

    Raises
    ------
    FileNotFoundError if the first training chromosome's file is missing.
    """
    filepath: Path = (
        args.data_directory / f"chr{args.training_chromosomes[0]}.h5"
    )
    if not filepath.exists():
        raise FileNotFoundError(
            f"Training chromosome file {filepath} not found"
        )
    return filepath


def build_training_parameters(
    args: Namespace, presets: dict[str, model_factory.Preset]
) -> qcpg.TrainingParameters:
    """
    Combine a preset, the model options and the training arguments.

    Arguments given override the preset, and the preset overrides the
    defaults of `qcpg.TrainingParameters`.

    Args
    ----
    args: The parsed ArgumentParser arguments.
    presets: The established preset models.

    Returns
    -------
    The training parameters.

    Raises
    ------
    ValueError if the preset is unknown, an option is given twice, or the
    preset's input does not match the data.
    """
    _, _, token_attributes = data.read_h5_layout(_first_training_filepath(args))
    data_input: str = str(
        token_attributes.get(data.INPUT_NAME_STR, model_factory.BASES_STR)
    )
    training: dict[str, Any] = {}
    options: dict[str, Any] = {}
    if args.preset:
        preset: model_factory.Preset = _find_preset(presets, args.preset)
        if preset.input != data_input:
            raise ValueError(
                f"Preset {preset.name} takes {preset.input} inputs, but "
                f"{args.data_directory} holds {data_input} inputs; see "
                "scripts/tokenize_cpg_windows.py"
            )
        model: str = preset.model
        training.update(preset.training)
        options.update(preset.options)
    else:
        model = args.model

    command_options: dict[str, Any] = _parse_options(args.options)
    for argument, option in _OPTION_ARGUMENTS.items():
        value = getattr(args, argument)
        if value is None:
            continue
        if option in command_options:
            raise ValueError(f"Option {option} is given twice")
        command_options[option] = value
    options.update(command_options)

    unknown: list[str] = sorted(
        set(training)
        - {parameter.name for parameter in fields(qcpg.TrainingParameters)}
    )
    if unknown:
        raise ValueError(
            f"Unknown training settings {', '.join(unknown)} in the presets"
        )
    for argument, parameter in _TRAINING_ARGUMENTS.items():
        if getattr(args, argument) is not None:
            training[parameter] = getattr(args, argument)
    if args.all_training_samples:
        training["training_sample_quantity"] = None

    encoding: str = args.encoding or (
        data.TOKEN_ENCODING_STR
        if token_attributes
        else data.ONEHOT_ENCODING_STR
    )
    tokenizer: Any = None
    vocabulary_size: int | None = args.vocabulary_size
    padding_token: int | None = args.padding_token
    if encoding == data.BPE_ENCODING_STR:
        if args.tokenizer is None:
            raise ValueError("The bpe encoding needs a --tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        vocabulary_size = vocabulary_size or tokenizer.vocab_size
        if padding_token is None:
            padding_token = tokenizer.pad_token_id

    return qcpg.TrainingParameters(
        data_directory=args.data_directory,
        output_directory=args.output_directory or args.data_directory,
        training_chromosomes=args.training_chromosomes,
        validation_chromosomes=args.validation_chromosomes,
        test_chromosomes=args.test_chromosomes,
        experiment_name=args.experiment_name,
        records_directory=args.records_directory,
        update_report=args.update_report,
        encoding=encoding,
        tokenizer=tokenizer,
        vocabulary_size=vocabulary_size,
        padding_token=padding_token,
        device_name=args.device_name,
        diff_method=args.diff_method,
        gpu_quantity=args.gpu_quantity,
        distributed=args.gpu_quantity > 1,
        model=model,
        model_options=options,
        seed=args.seed,
        evaluation_shots=args.shots,
        loader_worker_quantity=args.loader_workers,
        log_directory=args.log_directory,
        log_level=logging.DEBUG if args.log_level == "debug" else logging.INFO,
        **training,
    )


def describe(training_parameters: qcpg.TrainingParameters) -> str:
    """
    Describe the model the training parameters build for their data.

    Args
    ----
    training_parameters: The training parameters.

    Returns
    -------
    The model's family, options and numbers of trained parameters.
    """
    sequence_length, experiment_quantity, token_attributes = (
        data.read_h5_layout(
            training_parameters.data_directory
            / f"chr{training_parameters.training_chromosomes[0]}.h5"
        )
    )
    model = qcpg.build_model(
        training_parameters,
        sequence_length,
        experiment_quantity,
        token_attributes,
    )
    quantum_quantity, classical_quantity = model_factory.count_parameters(model)
    return (
        f"{training_parameters.model} {json.dumps(training_parameters.model_options)}\n"
        f"input length {sequence_length}, {model.channel_quantity} channels, "
        f"{experiment_quantity} outputs\n"
        f"trained circuit angles: {quantum_quantity}\n"
        f"other trained parameters: {classical_quantity}"
    )


def _set_threads(thread_quantity: int | None) -> None:
    """
    Sets the number of threads that Pytorch can use.

    Args
    ----
    thread_quantity: The number of threads.

    """
    thread_quantity = thread_quantity or int(
        os.environ.get("SLURM_CPUS_PER_TASK", "0")
    )
    if thread_quantity:
        torch.set_num_threads(thread_quantity)


def main(arguments: list[str] | None = None) -> None:
    args: Namespace = _parse_script_args(arguments)
    presets: dict[str, model_factory.Preset] = (
        model_factory.load_presets(args.presets_file)
        if args.list_presets or args.preset
        else {}
    )
    if args.list_presets:
        for name, preset in presets.items():
            print(f"{name}\t{preset.input}\t{preset.description}")
        return

    _set_threads(args.threads)
    training_parameters: qcpg.TrainingParameters = build_training_parameters(
        args, presets
    )
    if args.describe:
        print(describe(training_parameters))
        return

    qcpg.train_qnn_circuit(training_parameters)


if __name__ == "__main__":
    main()
