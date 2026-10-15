import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from qenetics.qcpg import model_factory
from qenetics.tools import tokenization

PRESETS_FILEPATH: Path = Path("config/quantum_models.json")
# The report's sequence length and token lookup of each input.
INPUTS: dict[str, tuple[int, tuple[str, int] | None]] = {
    "bases": (1001, None),
    "cpg_bpe64": (257, (tokenization.CPG_BPE, 64)),
    "cpg_bpe256": (257, (tokenization.CPG_BPE, 256)),
    "cpg_kmer3": (257, (tokenization.CPG_KMER, 65)),
    "cpg_kmer4": (257, (tokenization.CPG_KMER, 257)),
    "painted64": (1001, (tokenization.PAINTED, 64)),
    "painted_bases64": (1001, (tokenization.PAINTED_BASES, 320)),
}
# Trained circuit angles and total trained parameters of the report's tables,
# HepG2 (6 cells).
REPORT_COUNTS: dict[str, tuple[int, int]] = {
    "deepcpg_tail/Q_L4": (72, 72 + 4048134),
    "deepcpg_tail/Q_L320": (5760, 5760 + 4048134),
    "deepcpg_tail/MKm8_L4": (576, 576 + 4048134),
    "pooling_head/MS4_L4": (264, 264 + 3366),
    "pooling_head/MK16w4_L4": (768, 768 + 3590),
    "whole_model/FA_L24": (432, 432),
    "whole_model/AU_w201_L2": (144, 144),
    "whole_model/G1_lin": (60, 60 + 24582),
    "whole_model/G3_mlp": (60, 60 + 525190),
    "whole_model/G3s_L22": (792, 792 + 525190),
    "whole_model/H1b_cnnq": (144, 144 + 4105240),
    "tokens/T64_QE_L4": (72, 1066318),
    "tokens/T64_Qw8_L16": (432, 1378870),
    "tokens/T256_QE_L4": (96, 1214182),
    "tokens/P64b_QE_L4": (84, 4163034),
    "tokens/H1b_T64": (144, 1142440),
    "series_1/QA2": (144, 144 + 4097414),
    "series_1/FQ3": (168, 168 + 3998598),
    "series_1/L2": (108, 108 + 1049478),
    "series_1/K1": (0, 4048006),
}


@pytest.fixture(scope="module")
def presets() -> dict[str, model_factory.Preset]:
    return model_factory.load_presets(PRESETS_FILEPATH)


def _build_preset(
    preset: model_factory.Preset, output_quantity: int = 6
) -> model_factory.SequenceModel:
    sequence_length, lookup = INPUTS[preset.input]
    encoding_options: dict = {}
    if lookup is not None:
        encoding_options = {
            "encoding": "token",
            "token_lookup": tokenization.token_lookup(*lookup),
        }
    return model_factory.build_model(
        preset.model,
        sequence_length,
        output_quantity,
        options=preset.options,
        **encoding_options,
    )


def test_presets_cover_report_sections(
    presets: dict[str, model_factory.Preset],
) -> None:
    sections: set[str] = {name.split("/")[0] for name in presets}
    assert sections == {
        "deepcpg_tail",
        "pooling_head",
        "whole_model",
        "tokens",
        "series_1",
    }
    assert {preset.input for preset in presets.values()} <= set(INPUTS)
    assert all(preset.description for preset in presets.values())


@pytest.mark.parametrize("name", list(REPORT_COUNTS))
def test_preset_parameter_counts(
    presets: dict[str, model_factory.Preset], name: str
) -> None:
    quantum, classical = model_factory.count_parameters(
        _build_preset(presets[name])
    )
    assert (quantum, quantum + classical) == REPORT_COUNTS[name]


def test_every_preset_builds(presets: dict[str, model_factory.Preset]) -> None:
    for preset in presets.values():
        model = _build_preset(preset, output_quantity=2)
        quantum, _ = model_factory.count_parameters(model)
        assert quantum > 0 or not preset.options.get("trainable_circuit", True)


@pytest.mark.parametrize(
    "name",
    [
        "pooling_head/MS4_L4",
        "whole_model/WP_L4",
        "whole_model/FA_L12",
        "whole_model/GP_basic_L5",
        "tokens/WT64_L4",
    ],
)
def test_presets_run(
    presets: dict[str, model_factory.Preset], name: str
) -> None:
    preset = presets[name]
    model = _build_preset(preset, output_quantity=2)
    sequence_length, lookup = INPUTS[preset.input]
    if lookup is None:
        inputs = torch.nn.functional.one_hot(
            torch.randint(0, 4, (2, sequence_length)), 4
        ).float()
    else:
        inputs = torch.randint(0, lookup[1], (2, sequence_length))
    assert model(inputs).shape == (2, 2)


def test_model_option_names() -> None:
    options = model_factory.model_option_names("window_average")
    assert "window_length" in options and "sequence_length" not in options
    assert "device_name" not in model_factory.model_option_names(
        "global_circuit"
    )
    assert model_factory.uses_device("global_circuit")
    assert not model_factory.uses_device("filter_bank_network")
    with pytest.raises(ValueError, match="Unknown model family"):
        model_factory.model_option_names("invalid")


def test_build_model_errors() -> None:
    with pytest.raises(ValueError, match="Unknown options windows"):
        model_factory.build_model(
            "filter_bank_network", 100, 2, options={"windows": [4]}
        )
    with pytest.raises(ValueError, match="Unknown model family"):
        model_factory.build_model("invalid", 100, 2)


def test_build_model_device() -> None:
    model = model_factory.build_model(
        "feature_amplitude",
        40,
        2,
        device_name="lightning.qubit",
        diff_method="adjoint",
    )
    assert model.qnode.device.name == "lightning.qubit"


def test_set_evaluation_shots() -> None:
    model = model_factory.build_model(
        "filter_bank_network",
        40,
        2,
        options={"window_lengths": [4, 8], "tail": "pooling"},
    )
    model_factory.set_evaluation_shots(model, 100)
    assert [bank.shots for bank in model.front_end.banks] == [100, 100]
    model_factory.set_evaluation_shots(model, 0)
    assert [bank.shots for bank in model.front_end.banks] == [0, 0]
    with pytest.raises(ValueError, match="must not be negative"):
        model_factory.set_evaluation_shots(model, -1)


def _write_presets(directory: Path, content: dict) -> Path:
    filepath = directory / "presets.json"
    filepath.write_text(json.dumps(content))
    return filepath


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({"presets": {}}, "must hold 'training_settings' and 'presets'"),
        (
            {
                "training_settings": {},
                "presets": {"a": {"model": "automaton"}},
            },
            "unknown training settings",
        ),
        (
            {
                "training_settings": {"s": {}},
                "presets": {"a": {"training_settings": "s"}},
            },
            "must name a model",
        ),
        (
            {
                "training_settings": {"s": {}},
                "presets": {
                    "a": {
                        "model": "automaton",
                        "training_settings": "s",
                        "options": {"windows": 3},
                    }
                },
            },
            "unknown options windows",
        ),
    ],
)
def test_load_presets_errors(content: dict, message: str) -> None:
    with (
        TemporaryDirectory() as directory,
        pytest.raises(ValueError, match=message),
    ):
        model_factory.load_presets(_write_presets(Path(directory), content))


def test_load_presets() -> None:
    with TemporaryDirectory() as directory:
        presets = model_factory.load_presets(
            _write_presets(
                Path(directory),
                {
                    "training_settings": {"s": {"epochs": 3}},
                    "presets": {
                        "a/b": {
                            "model": "automaton",
                            "training_settings": "s",
                            "options": {"width": 11},
                        }
                    },
                },
            )
        )
    assert presets["a/b"] == model_factory.Preset(
        name="a/b",
        model="automaton",
        options={"width": 11},
        training={"epochs": 3},
    )
