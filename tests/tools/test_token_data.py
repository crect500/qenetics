from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest
import torch

from qenetics.tools import data, tokenization

CELL_NAMES: list[str] = ["cellA", "cellB"]
WINDOW_LENGTH: int = 41


def _write_bases(directory: Path, chromosomes: list[str]) -> None:
    rng = np.random.default_rng(0)
    for chromosome in chromosomes:
        reference = np.frombuffer(
            "".join(rng.choice(list("ACGT"), 3000)).encode(), dtype=np.uint8
        ).copy()
        text: str = reference.tobytes().decode()
        sites = np.array(
            [site for site in range(30, 2960) if text[site : site + 2] == "CG"],
            dtype=np.int64,
        )
        labels = rng.integers(0, 2, (len(sites), 2)).astype(np.float32)
        labels[rng.random(labels.shape) < 0.2] = np.nan
        data._write_chromosome_h5(
            directory / f"chr{chromosome}.h5",
            reference,
            sites,
            labels,
            CELL_NAMES,
            WINDOW_LENGTH,
        )


@pytest.mark.parametrize(
    ("method", "options", "length", "vocabulary_size", "name"),
    [
        (
            tokenization.CPG_BPE,
            {"vocabulary_size": 16},
            2 * 8 + 1,
            16,
            "cpg_bpe16",
        ),
        (
            tokenization.CPG_KMER,
            {"kmer_length": 2},
            2 * 8 + 1,
            17,
            "cpg_kmer2",
        ),
        (
            tokenization.PAINTED,
            {"vocabulary_size": 16},
            WINDOW_LENGTH,
            16,
            "painted16",
        ),
        (
            tokenization.PAINTED_BASES,
            {"vocabulary_size": 16},
            WINDOW_LENGTH,
            80,
            "painted_bases16",
        ),
    ],
)
def test_create_token_h5_dataset(
    method: str, options: dict, length: int, vocabulary_size: int, name: str
) -> None:
    with TemporaryDirectory() as temp_dir:
        bases_directory = Path(temp_dir) / "bases"
        bases_directory.mkdir()
        _write_bases(bases_directory, ["1", "2"])
        output_directory = Path(temp_dir) / name
        filepaths = data.create_token_h5_dataset(
            bases_directory,
            output_directory,
            method,
            training_chromosomes=["1"],
            flank_token_quantity=8,
            **options,
        )
        assert [filepath.name for filepath in filepaths] == [
            "chr1.h5",
            "chr2.h5",
        ]
        if method == tokenization.CPG_BPE:
            assert (output_directory / "tokenizer.json").exists()

        tokens = data.QuantumTorchDataset(
            filepaths, encoding=data.TOKEN_ENCODING_STR
        )
        bases = data.QuantumTorchDataset(
            sorted(bases_directory.glob("chr*.h5")),
            encoding=data.ONEHOT_ENCODING_STR,
        )
        assert tokens.token_attributes == {
            data.TOKENIZATION_ATTRIBUTE: method,
            data.INPUT_NAME_STR: name,
            data.VOCABULARY_SIZE_ATTRIBUTE: vocabulary_size,
        }
        assert tokens.sequences.shape == (len(bases), length)
        assert tokens.sequences.dtype == torch.int16
        assert 0 <= tokens.sequences.min() and (
            tokens.sequences.max() < vocabulary_size
        )
        assert tokens.experiment_names == CELL_NAMES
        assert torch.equal(
            tokens.labels.nan_to_num(-9), bases.labels.nan_to_num(-9)
        )
        assert np.array_equal(tokens.positions, bases.positions)
        assert data.read_h5_layout(filepaths[0]) == (
            length,
            2,
            tokens.token_attributes,
        )


def test_create_token_h5_dataset_reuses_tokenizer() -> None:
    with TemporaryDirectory() as temp_dir:
        bases_directory = Path(temp_dir) / "bases"
        bases_directory.mkdir()
        _write_bases(bases_directory, ["1"])
        data.create_token_h5_dataset(
            bases_directory,
            Path(temp_dir) / "bpe",
            tokenization.CPG_BPE,
            vocabulary_size=16,
            training_chromosomes=["1"],
            flank_token_quantity=8,
        )
        tokenizer_filepath = Path(temp_dir) / "bpe" / "tokenizer.json"
        painted = data.create_token_h5_dataset(
            bases_directory,
            Path(temp_dir) / "painted",
            tokenization.PAINTED,
            vocabulary_size=16,
            tokenizer_filepath=tokenizer_filepath,
            flank_token_quantity=8,
        )
        with h5py.File(painted[0]) as fd:
            centre: int = (WINDOW_LENGTH - 1) // 2
            sequences = fd[data.METHYLATION_SEQUENCES_KEY][()]
            assert (
                sequences[:, centre : centre + 2] == tokenization.CPG_ID
            ).all()
        assert not (Path(temp_dir) / "painted" / "tokenizer.json").exists()


@pytest.mark.parametrize(
    ("method", "options", "error", "message"),
    [
        ("invalid", {}, ValueError, "Unknown tokenization invalid"),
        (tokenization.CPG_KMER, {}, ValueError, "needs a k-mer length"),
        (tokenization.CPG_BPE, {}, ValueError, "needs a vocabulary size"),
        (
            tokenization.CPG_BPE,
            {"vocabulary_size": 16},
            ValueError,
            "needs training chromosomes",
        ),
        (
            tokenization.CPG_BPE,
            {"vocabulary_size": 16, "training_chromosomes": ["7"]},
            ValueError,
            "chr7.h5",
        ),
    ],
)
def test_create_token_h5_dataset_errors(
    method: str, options: dict, error: type, message: str
) -> None:
    with TemporaryDirectory() as temp_dir:
        _write_bases(Path(temp_dir), ["1"])
        with pytest.raises(error, match=message):
            data.create_token_h5_dataset(
                Path(temp_dir), Path(temp_dir) / "out", method, **options
            )


def test_create_token_h5_dataset_without_files() -> None:
    with (
        TemporaryDirectory() as temp_dir,
        pytest.raises(FileNotFoundError, match="No chr"),
    ):
        data.create_token_h5_dataset(
            Path(temp_dir), Path(temp_dir) / "out", tokenization.CPG_KMER
        )


def _write_tokens(
    filepath: Path, tokens: np.ndarray, attributes: dict[str, str | int]
) -> None:
    with h5py.File(filepath, "w") as fd:
        fd.attrs.update(attributes)
        datasets = data._ChromosomeDatasets.create(
            fd,
            tokens.shape[1],
            None,
            sample_shape=(tokens.shape[1],),
            dtype=tokens.dtype.str,
        )
        datasets.append(
            tokens,
            np.arange(len(tokens), dtype=np.int64),
            np.zeros((len(tokens), 1), dtype=np.float32),
        )


def test_dataset_keeps_wide_tokens() -> None:
    tokens = np.array([[1, 300, 255], [-2, 0, 32000]], dtype=np.int16)
    with TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "chr1.h5"
        _write_tokens(filepath, tokens, {})
        dataset = data.QuantumTorchDataset(
            [filepath], encoding=data.TOKEN_ENCODING_STR
        )
        assert dataset.sequences.dtype == torch.int16
        assert dataset.sequences.tolist() == tokens.tolist()
        assert dataset.experiment_names is None
        samples, _ = dataset[1]
        assert samples.tolist() == [-2.0, 0.0, 32000.0]


def test_dataset_token_attributes_must_match() -> None:
    tokens = np.ones((2, 5), dtype=np.int16)
    with TemporaryDirectory() as temp_dir:
        first = Path(temp_dir) / "chr1.h5"
        second = Path(temp_dir) / "chr2.h5"
        _write_tokens(first, tokens, {data.INPUT_NAME_STR: "cpg_bpe64"})
        _write_tokens(second, tokens, {data.INPUT_NAME_STR: "cpg_bpe256"})
        with pytest.raises(ValueError, match="Token attributes of"):
            data.QuantumTorchDataset(
                [first, second], encoding=data.TOKEN_ENCODING_STR
            )


def test_sequence_codes_wide_tokens() -> None:
    samples = np.array([[300, -5]])
    codes = data._sequence_codes(
        samples,
        data.TOKEN_ENCODING_STR,
        data.TOKEN_ENCODING_STR,
        dtype=np.int16,
    )
    assert codes.dtype == np.int16 and codes.tolist() == [[300, -5]]
    with pytest.raises(ValueError, match="16-bit"):
        data._sequence_codes(
            np.array([[40000]]),
            data.TOKEN_ENCODING_STR,
            data.TOKEN_ENCODING_STR,
            dtype=np.int16,
        )
