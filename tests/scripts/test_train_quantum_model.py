import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
import pytest

from qenetics.qcpg import model_factory, qcpg
from qenetics.tools import data, tokenization

SCRIPT: Path = Path("scripts/train_quantum_model.py")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "train_quantum_model", SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def presets() -> dict[str, model_factory.Preset]:
    return model_factory.load_presets(Path("config/quantum_models.json"))


@pytest.fixture
def data_directory():
    with TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        rng = np.random.default_rng(0)
        reference = np.frombuffer(
            "".join(rng.choice(list("ACGT"), 600)).encode(), dtype=np.uint8
        ).copy()
        text: str = reference.tobytes().decode()
        sites = np.array(
            [site for site in range(30, 560) if text[site : site + 2] == "CG"],
            dtype=np.int64,
        )
        data._write_chromosome_h5(
            directory / "chr1.h5",
            reference,
            sites,
            np.zeros((len(sites), 1), dtype=np.float32),
            ["cell"],
            41,
        )
        yield directory


def _parameters(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
    *arguments: str,
) -> qcpg.TrainingParameters:
    args = script._parse_script_args(
        [
            "--data-directory",
            str(data_directory),
            "--output-directory",
            str(data_directory / "output"),
            "--training-chromosomes",
            "1",
            *arguments,
        ]
    )
    return script.build_training_parameters(args, presets)


def test_preset_settings(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
) -> None:
    parameters = _parameters(
        script, presets, data_directory, "--preset", "deepcpg_tail/Q_L4"
    )
    preset = presets["deepcpg_tail/Q_L4"]
    assert parameters.model == "filter_bank_network"
    assert parameters.model_options == preset.options
    assert parameters.epochs == preset.training["epochs"] == 10
    assert parameters.patience == 2
    assert parameters.batch_size == 256
    assert parameters.learning_rate_decay == 0.95
    assert parameters.selection_metric == qcpg.MEAN_CELL_AUC
    assert parameters.training_sample_quantity == 200000
    assert parameters.encoding == data.ONEHOT_ENCODING_STR
    assert parameters.device_name == "default.qubit"


def test_arguments_override_preset(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
) -> None:
    parameters = _parameters(
        script,
        presets,
        data_directory,
        "--preset",
        "deepcpg_tail/Q_L4",
        "--quantum-layers",
        "8",
        "--hidden-layer-sizes",
        "32",
        "16",
        "--option",
        "tail=pooling",
        "--option",
        "window_lengths=[4, 8]",
        "--epochs",
        "3",
        "--all-training-samples",
        "--shots",
        "10",
        "100",
        "--seed",
        "2",
        "--device-name",
        "lightning.qubit",
    )
    assert parameters.model_options == {
        "window_lengths": [4, 8],
        "quantum_layer_quantity": 8,
        "hidden_layer_sizes": [32, 16],
        "tail": "pooling",
    }
    assert parameters.epochs == 3
    assert parameters.patience == 2
    assert parameters.training_sample_quantity is None
    assert parameters.evaluation_shots == [10, 100]
    assert parameters.seed == 2
    assert parameters.device_name == "lightning.qubit"


def test_model_family(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
) -> None:
    parameters = _parameters(
        script,
        presets,
        data_directory,
        "--model",
        "automaton",
        "--option",
        "width=11",
        "--ansatz",
        "real",
    )
    assert parameters.model == "automaton"
    assert parameters.model_options == {"width": 11, "ansatz": "real"}
    assert parameters.selection_metric is None


def test_describe(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
) -> None:
    parameters = _parameters(
        script,
        presets,
        data_directory,
        "--preset",
        "pooling_head/MS4_L4",
        "--option",
        "window_lengths=[4, 8]",
    )
    description: str = script.describe(parameters)
    assert "trained circuit angles: 108" in description
    assert "input length 41, 4 channels, 1 outputs" in description


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--preset", "deepcpg_tail/Q_L5"], "did you mean deepcpg_tail/Q_L4"),
        (["--preset", "tokens/T64_QE_L4"], "takes cpg_bpe64 inputs"),
        (
            [
                "--model",
                "automaton",
                "--option",
                "ansatz=real",
                "--ansatz",
                "real",
            ],
            "given twice",
        ),
        (["--model", "automaton", "--option", "width"], "is not KEY=VALUE"),
        (["--model", "automaton", "--encoding", "bpe"], "needs a --tokenizer"),
    ],
)
def test_argument_errors(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
    arguments: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parameters(script, presets, data_directory, *arguments)


def test_missing_training_file(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
) -> None:
    args = script._parse_script_args(
        [
            "--model",
            "automaton",
            "-d",
            str(data_directory),
            "-o",
            str(data_directory),
            "--training-chromosomes",
            "9",
        ]
    )
    with pytest.raises(FileNotFoundError, match="chr9.h5"):
        script.build_training_parameters(args, presets)


def test_required_arguments(script: ModuleType) -> None:
    with pytest.raises(SystemExit):
        script._parse_script_args(["--model", "automaton"])
    with pytest.raises(SystemExit):
        script._parse_script_args(
            ["--preset", "a", "--model", "automaton", "-d", ".", "-o", "."]
        )
    assert script._parse_script_args(["--list-presets"]).list_presets


def test_token_data_selects_token_encoding(
    script: ModuleType,
    presets: dict[str, model_factory.Preset],
    data_directory: Path,
) -> None:
    data.create_token_h5_dataset(
        data_directory,
        data_directory / "kmer",
        tokenization.CPG_KMER,
        kmer_length=3,
        flank_token_quantity=12,
    )
    parameters = _parameters(
        script, presets, data_directory / "kmer", "--preset", "tokens/K3_QE_L4"
    )
    assert parameters.encoding == data.TOKEN_ENCODING_STR
    assert "64 channels" in script.describe(parameters)


def test_list_presets(
    script: ModuleType, capsys: pytest.CaptureFixture
) -> None:
    script.main(["--list-presets"])
    lines: list[str] = capsys.readouterr().out.splitlines()
    names: list[str] = [line.split("\t")[0] for line in lines]
    with Path("config/quantum_models.json").open() as fd:
        assert names == list(json.load(fd)["presets"])
