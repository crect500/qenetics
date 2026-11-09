import itertools
import json
from pathlib import Path

import pytest
import torch

from qenetics.qcpg import classical_models, heads, model_factory
from qenetics.qcpg.sequence_model import STATISTIC_QUANTITY
from qenetics.tools import tokenization

PRESETS_FILEPATH: Path = Path("config/quantum_models.json")
# Front-end and total trained parameters of Series II's classical models and
# ceilings, HepG2 (6 cells), by repository preset.
SERIES2_SIZES: Path = Path("tests/test_files/series2_classical_sizes.json")
INPUTS: dict[str, tuple[int, tuple[str, int] | None]] = {
    "bases": (1001, None),
    "cpg_bpe64": (257, (tokenization.CPG_BPE, 64)),
    "cpg_bpe256": (257, (tokenization.CPG_BPE, 256)),
    "cpg_kmer3": (257, (tokenization.CPG_KMER, 65)),
    "cpg_kmer4": (257, (tokenization.CPG_KMER, 257)),
    "painted64": (1001, (tokenization.PAINTED, 64)),
    "painted_bases64": (1001, (tokenization.PAINTED_BASES, 320)),
    "painted256": (1001, (tokenization.PAINTED, 256)),
    "painted_bases256": (1001, (tokenization.PAINTED_BASES, 1280)),
}


def _one_hot(batch: int, length: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.nn.functional.one_hot(
        torch.randint(0, 4, (batch, length)), 4
    ).float()


@pytest.mark.parametrize(
    "name, sizes", list(json.loads(SERIES2_SIZES.read_text()).items())
)
def test_series2_sizes(name: str, sizes: list[int]) -> None:
    preset = model_factory.load_presets(PRESETS_FILEPATH)[name]
    sequence_length, lookup = INPUTS[preset.input]
    encoding: dict = (
        {}
        if lookup is None
        else {
            "encoding": "token",
            "token_lookup": tokenization.token_lookup(*lookup),
        }
    )
    model = model_factory.build_model(
        preset.model, sequence_length, 6, options=preset.options, **encoding
    )
    assert model_factory.model_sizes(model) == tuple(sizes)


@pytest.mark.parametrize(
    "options, channels",
    [
        ({"front": "convolution", "window_length": 4, "filter_quantity": 5}, 5),
        ({"front": "identity", "window_length": 3}, 12),
        ({"front": "low_rank", "rank": 2, "window_length": 4}, 64),
        ({"front": "givens", "angle_quantity": 10, "window_length": 4}, 16),
        ({"front": "dense", "matrix": "complex", "window_length": 2}, 8),
        (
            {
                "front": "multi_scale",
                "scales": [[2, 3], [5, 2]],
                "mode": "quadrature",
            },
            5,
        ),
        ({"front": "two_layer", "filter_quantities": [6, 7]}, 7),
    ],
)
@pytest.mark.parametrize("tail", ["deepcpg", "pooling"])
def test_classical_network_fronts(
    options: dict, channels: int, tail: str
) -> None:
    model = classical_models.ClassicalNetwork(101, 3, tail=tail, **options)
    assert model.front_end.output_channel_quantity == channels
    probabilities = model(_one_hot(2, 101))
    assert probabilities.shape == (2, 3)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    probabilities.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.front_end.parameters()
        if parameter.requires_grad
    )
    expected_tail = (
        heads.DeepCpGTail if tail == "deepcpg" else heads.PoolingHead
    )
    assert isinstance(model.tail, expected_tail)


@pytest.mark.parametrize("matrix", ["unitary", "orthogonal", "complex", "real"])
def test_dense_filter_bank(matrix: str) -> None:
    bank = classical_models.DenseFilterBank(
        4, 2, matrix=matrix, initialization_scale=0.3
    )
    transform = bank.transform()
    assert transform.is_complex() == (matrix in ("unitary", "complex"))
    identity = torch.eye(8, dtype=transform.dtype)
    is_isometry = torch.allclose(
        transform @ transform.conj().T, identity, atol=1e-5
    )
    assert is_isometry == (matrix in ("unitary", "orthogonal"))
    assert (
        sum(p.numel() for p in bank.parameters())
        == {
            "unitary": 64,
            "orthogonal": 28,
            "complex": 128,
            "real": 64,
        }[matrix]
    )
    # A one-hot window has norm 1 after scaling, so the d probabilities of an
    # isometry, times d, sum to d at every position.
    features = bank(_one_hot(3, 10).transpose(1, 2))
    assert features.shape == (3, 8, 9)
    if is_isometry:
        assert torch.allclose(
            features.sum(1), torch.full((3, 9), 8.0), atol=1e-4
        )


def test_identity_and_givens_fronts() -> None:
    sequence = _one_hot(1, 3).transpose(1, 2)
    identity = classical_models.IdentityFront(4, 2)
    givens = classical_models.GivensFront(4, 2, 4)
    with torch.no_grad():
        givens.angles.zero_()
    # Both give each window's amplitudes at index A * position + channel; the
    # identity times A, the Givens front with every angle 0 as |x|^2 d / w.
    for position in range(2):
        window = sequence[0, :, position : position + 2].T.reshape(-1)
        assert torch.allclose(identity(sequence)[0, :, position], window * 4)
        assert torch.allclose(givens(sequence)[0, :, position], window * 8 / 2)
    # One layer of d / 2 = 4 angles, all trained.
    assert givens.angles.shape == (1, 4) and givens.mask.sum() == 4
    rotation = classical_models.GivensFront(4, 2, 6, initialization_scale=1.0)
    assert rotation.angles.shape == (2, 4) and rotation.mask.sum() == 6
    q = rotation.transform()
    assert torch.allclose(q @ q.T, torch.eye(8), atol=1e-5)


def test_multi_scale_front_alignment() -> None:
    front = classical_models.MultiScaleFront(4, [[2, 1], [6, 1]], "relu")
    sequence = _one_hot(1, 20).transpose(1, 2)
    features = front(sequence)
    assert features.shape == (1, 2, 15)
    # The 2-position channel is cropped to the 6-position channel's centres.
    narrow = torch.relu(front.convolutions[0](sequence))
    assert torch.allclose(features[:, 0], narrow[0, 0, 2:17])


def test_whole_classifiers() -> None:
    sequence = _one_hot(4, 30)
    features = classical_models.FeatureClassifier(30, 2, hidden_layer_sizes=[4])
    assert features.normalization.num_features == STATISTIC_QUANTITY
    assert features(sequence).shape == (4, 2)
    one_hot = classical_models.OneHotClassifier(30, 2)
    assert one_hot.head[0].in_features == 120
    assert one_hot(sequence).shape == (4, 2)
    frequencies = classical_models.TokenFrequencyClassifier(30, 3)
    probabilities = frequencies(sequence)
    assert probabilities.shape == (4, 3)
    assert torch.allclose(probabilities, probabilities[:, :1].expand(-1, 3))
    with pytest.raises(ValueError, match="nucleotides"):
        classical_models.FeatureClassifier(
            30,
            2,
            encoding="token",
            token_lookup=tokenization.token_lookup(tokenization.CPG_BPE, 64),
        )


def test_factorization_machine() -> None:
    model = classical_models.FactorizationMachine(
        3, 2, rank=2, initialization_scale=0.5
    )
    sequence = _one_hot(4, 3)
    features = sequence.transpose(1, 2).flatten(1)
    factors = model.factors.detach()
    pairwise = torch.stack(
        [
            sum(
                factors[i] * factors[j] * x[i] * x[j]
                for i, j in itertools.combinations(range(12), 2)
            )
            for x in features
        ]
    )
    logits = model.linear(features) + model.interaction_weights(pairwise)
    assert torch.allclose(model(sequence), torch.sigmoid(logits), atol=1e-6)
    with pytest.raises(ValueError, match="rank"):
        classical_models.FactorizationMachine(3, 2, rank=0)


def test_classical_network_errors() -> None:
    with pytest.raises(ValueError, match="Unknown front"):
        classical_models.ClassicalNetwork(101, 2, front="attention")
    with pytest.raises(ValueError, match="Unknown matrix free"):
        classical_models.ClassicalNetwork(101, 2, front="dense", matrix="free")
    with pytest.raises(ValueError, match="no first pool"):
        classical_models.ClassicalNetwork(
            101, 2, tail="pooling", first_pool="average"
        )


def test_model_sizes() -> None:
    model = classical_models.ClassicalNetwork(
        101, 2, front="identity", window_length=2
    )
    front, total = model_factory.model_sizes(model)
    assert front == 16  # the affine map of 8 channels
    assert total == sum(p.numel() for p in model.parameters())
    whole = classical_models.OneHotClassifier(30, 2)
    assert model_factory.model_sizes(whole) == (242, 242)
