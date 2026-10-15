"""
Build the quantum and hybrid methylation models by name, and read presets.

A preset names one model of the quantum models report: its family, options,
training settings and the input it was trained on. Presets are stored in
`config/quantum_models.json`.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor, nn

from qenetics.circuits import device_utils
from qenetics.qcpg.hybrid_models import (
    CNNQuantumHead,
    FilterBankNetwork,
    GlobalCircuitModel,
)
from qenetics.qcpg.pure_models import (
    FeatureAmplitudeClassifier,
    FeatureReuploadingClassifier,
    QuantumAutomaton,
    WindowAverageClassifier,
)
from qenetics.qcpg.sequence_model import (
    ONEHOT_ENCODING,
    SequenceModel,
    require_choice,
)
from qenetics.tools import converters

MODEL_FAMILIES: dict[str, type[SequenceModel]] = {
    "filter_bank_network": FilterBankNetwork,
    "window_average": WindowAverageClassifier,
    "feature_amplitude": FeatureAmplitudeClassifier,
    "feature_reuploading": FeatureReuploadingClassifier,
    "automaton": QuantumAutomaton,
    "global_circuit": GlobalCircuitModel,
    "cnn_quantum_head": CNNQuantumHead,
}
# Parameters every model is given by `build_model` rather than options.
_BUILD_PARAMETERS: frozenset[str] = frozenset(
    {"self", "sequence_length", "output_quantity", "device_name", "diff_method"}
)
CIRCUIT_WEIGHTS_NAME: str = "circuit_weights"
BASES_STR: str = "bases"


def model_option_names(family: str) -> list[str]:
    """
    List the options of a model family.

    Raises
    ------
    ValueError if the family is unknown.
    """
    require_choice("model family", family, tuple(MODEL_FAMILIES))
    return [
        name
        for name, parameter in inspect.signature(
            MODEL_FAMILIES[family].__init__
        ).parameters.items()
        if name not in _BUILD_PARAMETERS
        and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def uses_device(family: str) -> bool:
    """Whether a family runs circuits as QNodes on a PennyLane device."""
    require_choice("model family", family, tuple(MODEL_FAMILIES))
    return (
        "device_name"
        in inspect.signature(MODEL_FAMILIES[family].__init__).parameters
    )


def build_model(
    family: str,
    sequence_length: int,
    output_quantity: int,
    *,
    encoding: str = ONEHOT_ENCODING,
    channel_quantity: int = converters.UNIQUE_NUCLEOTIDE_QUANTITY,
    vocabulary_size: int | None = None,
    padding_token: int | None = None,
    token_lookup: Tensor | None = None,
    device_name: str = device_utils.DEFAULT_DEVICE,
    diff_method: str = "best",
    options: dict[str, Any] | None = None,
) -> SequenceModel:
    """
    Build a model.

    Args
    ----
    family: One of `MODEL_FAMILIES`.
    sequence_length: The positions of every input.
    output_quantity: The number of outputs (cells).
    encoding, channel_quantity, vocabulary_size, padding_token,
    token_lookup: The input encoding, see `sequence_model.InputEncoder`.
    device_name: The PennyLane device of families that run QNodes.
    diff_method: The PennyLane differentiation method of those families.
    options: The family's options, see `model_option_names`.

    Returns
    -------
    The model.

    Raises
    ------
    ValueError if the family or an option is unknown, or an option is
    invalid.
    """
    options = dict(options or {})
    unknown: list[str] = sorted(set(options) - set(model_option_names(family)))
    if unknown:
        raise ValueError(
            f"Unknown options {', '.join(unknown)} for the {family} model; "
            f"its options are {', '.join(model_option_names(family))}"
        )

    if uses_device(family):
        options.update(device_name=device_name, diff_method=diff_method)
    return MODEL_FAMILIES[family](
        sequence_length,
        output_quantity,
        encoding=encoding,
        channel_quantity=channel_quantity,
        vocabulary_size=vocabulary_size,
        padding_token=padding_token,
        token_lookup=token_lookup,
        **options,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count a model's trained parameters.

    Returns
    -------
    The number of trained circuit angles and of other trained parameters.
    """
    quantum: int = 0
    classical: int = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.split(".")[-1] == CIRCUIT_WEIGHTS_NAME:
            quantum += parameter.numel()
        else:
            classical += parameter.numel()
    return quantum, classical


def set_evaluation_shots(model: nn.Module, shots: int) -> None:
    """
    Estimate every measured probability of a model from `shots`
    measurements when it is evaluated; 0 restores exact probabilities.

    Raises
    ------
    ValueError if shots is negative.
    """
    if shots < 0:
        raise ValueError(f"Shots must not be negative, not {shots}")

    for module in model.modules():
        if hasattr(module, "shots"):
            setattr(module, "shots", shots)  # noqa: B010


@dataclass
class Preset:
    """
    A model of the quantum models report.

    Attributes
    ----------
    name: "<section>/<report name>".
    model: The model family.
    options: The family's options.
    training: The training parameters it was trained with.
    input: The input it was trained on: `BASES_STR` or the name of a
        tokenization, see `tokenization.input_name`.
    description: What the model is.
    """

    name: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    input: str = BASES_STR
    description: str = ""


def load_presets(filepath: Path) -> dict[str, Preset]:
    """
    Read the presets of a presets file.

    The file holds "training_settings", named sets of training parameters,
    and "presets", each naming a model family, its options, one of the
    training settings and its input.

    Args
    ----
    filepath: The presets file.

    Returns
    -------
    The presets, indexed by name.

    Raises
    ------
    ValueError if the file is malformed, or a preset names an unknown family,
    option or training setting.
    """
    with filepath.open() as fd:
        content: dict[str, Any] = json.load(fd)

    try:
        training_settings: dict[str, dict[str, Any]] = content[
            "training_settings"
        ]
        entries: dict[str, dict[str, Any]] = content["presets"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Presets file {filepath} must hold 'training_settings' and "
            "'presets'"
        ) from error

    presets: dict[str, Preset] = {}
    for name, entry in entries.items():
        settings_name: str = entry.get("training_settings", "")
        if settings_name not in training_settings:
            raise ValueError(
                f"Preset {name} names unknown training settings "
                f"'{settings_name}'"
            )
        unknown_keys: set[str] = set(entry) - {
            "model",
            "options",
            "training_settings",
            "input",
            "description",
        }
        if unknown_keys or "model" not in entry:
            raise ValueError(
                f"Preset {name} must name a model and may only hold options, "
                "training_settings, input and description"
            )
        options: dict[str, Any] = entry.get("options", {})
        unknown_options: list[str] = sorted(
            set(options) - set(model_option_names(entry["model"]))
        )
        if unknown_options:
            raise ValueError(
                f"Preset {name} has unknown options "
                f"{', '.join(unknown_options)}"
            )
        presets[name] = Preset(
            name=name,
            model=entry["model"],
            options=options,
            training=dict(training_settings[settings_name]),
            input=entry.get("input", BASES_STR),
            description=entry.get("description", ""),
        )
    return presets
