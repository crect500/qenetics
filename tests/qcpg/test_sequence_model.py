import pytest
import torch

from qenetics.qcpg import sequence_model
from qenetics.qcpg.sequence_model import InputEncoder
from qenetics.tools import converters


def _one_hot(sequence: str) -> torch.Tensor:
    return torch.tensor(
        converters.nucleotide_string_to_numpy(sequence), dtype=torch.float
    )


def test_one_hot_lookup() -> None:
    lookup = sequence_model.one_hot_lookup(4, padding_token=0)
    assert lookup.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    with pytest.raises(ValueError, match="Padding token 4"):
        sequence_model.one_hot_lookup(4, padding_token=4)
    with pytest.raises(ValueError, match="vocabulary size must be at least 1"):
        sequence_model.one_hot_lookup(0)


def test_one_hot_encoder() -> None:
    encoder = InputEncoder(4)
    channels, is_batched = encoder(_one_hot("ACGT"))
    assert not is_batched
    assert channels.shape == (1, 4, 4)
    assert torch.equal(channels[0].T, _one_hot("ACGT"))
    batched, is_batched = encoder(torch.stack([_one_hot("ACGT")] * 3))
    assert is_batched and batched.shape == (3, 4, 4)


def test_token_encoder_matches_one_hot() -> None:
    tokens = torch.tensor([[0.0, 3.0, 2.0, 1.0]])
    channels, _ = InputEncoder(
        4, sequence_model.TOKEN_ENCODING, vocabulary_size=4
    )(tokens)
    expected, _ = InputEncoder(4)(_one_hot("ACGT").unsqueeze(0))
    assert torch.equal(channels, expected)


def test_token_encoder_lookup() -> None:
    lookup = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    encoder = InputEncoder(
        2, sequence_model.TOKEN_ENCODING, token_lookup=lookup
    )
    assert encoder.channel_quantity == 2
    channels, _ = encoder(torch.tensor([2, 0]))
    assert channels[0].T.tolist() == [[1.0, 1.0], [0.0, 0.0]]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"encoding": "invalid"}, "Unknown encoding invalid"),
        ({"encoding": "token"}, "vocabulary size or a token lookup"),
        (
            {
                "encoding": "token",
                "vocabulary_size": 3,
                "token_lookup": torch.eye(4),
            },
            "does not match a lookup of 4",
        ),
        (
            {"encoding": "token", "token_lookup": torch.zeros(3)},
            "non-empty",
        ),
        ({"channel_quantity": 0}, "channel quantity"),
    ],
)
def test_encoder_errors(arguments: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InputEncoder(4, **arguments)


@pytest.mark.parametrize(
    ("encoder", "inputs", "message"),
    [
        (InputEncoder(4), torch.zeros(2, 5, 4), r"shape \(batch, 4, 4\)"),
        (InputEncoder(4), torch.zeros(2, 4, 3), r"shape \(batch, 4, 4\)"),
        (
            InputEncoder(4, "token", vocabulary_size=4),
            torch.zeros(2, 3),
            r"shape \(batch, 4\)",
        ),
        (
            InputEncoder(4, "token", vocabulary_size=4),
            torch.tensor([0, 1, 2, 4]),
            "between 0 and 3",
        ),
    ],
)
def test_encoder_input_errors(
    encoder: InputEncoder, inputs: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        encoder(inputs)


def test_resolve_readout_wires() -> None:
    assert sequence_model.resolve_readout_wires("shared", 3, 6) == [0, 0, 0]
    assert sequence_model.resolve_readout_wires("per_output", 8, 6) == [
        0,
        1,
        2,
        3,
        4,
        5,
        0,
        1,
    ]
    assert sequence_model.resolve_readout_wires([4], 2, 6) == [4, 4]
    assert sequence_model.resolve_readout_wires([1, 3], 2, 6) == [1, 3]
    with pytest.raises(ValueError, match="one per output"):
        sequence_model.resolve_readout_wires([1, 2], 3, 6)
    with pytest.raises(ValueError, match=r"Wires \[6\]"):
        sequence_model.resolve_readout_wires([6], 3, 6)
    with pytest.raises(ValueError, match="Unknown readout"):
        sequence_model.resolve_readout_wires("invalid", 3, 6)


def test_measured_wires() -> None:
    assert sequence_model.measured_wires([3, 0, 3, 1]) == (
        [0, 1, 3],
        [2, 0, 2, 1],
    )


def test_check_probability_differentiation() -> None:
    sequence_model.check_probability_differentiation("default.qubit", "adjoint")
    sequence_model.check_probability_differentiation("lightning.qubit", "best")
    with pytest.raises(ValueError, match="adjoint"):
        sequence_model.check_probability_differentiation(
            "lightning.qubit", "adjoint"
        )


def test_sequence_statistics() -> None:
    sequence: str = "ACGTCGGA"
    channels, _ = InputEncoder(len(sequence))(_one_hot(sequence))
    statistics = sequence_model.sequence_statistics(channels)
    assert statistics.shape == (1, sequence_model.STATISTIC_QUANTITY)
    # The window holds every dinucleotide within the radii: AC CG GT TC CG GG
    # GA, so CG (C = 3, G = 2) has frequency 2 / 7.
    cg_index: int = 4 * 3 + 2
    assert statistics[0, cg_index].item() == pytest.approx(2 / 7)
    assert statistics[0, :16].sum().item() == pytest.approx(1.0)
    # The last pair is the CpG and GC density within 500 bp.
    assert statistics[0, -1].item() == pytest.approx(5 / 8)
    assert statistics[0, -2].item() == pytest.approx(2 / 7)


def test_sequence_statistics_errors() -> None:
    with pytest.raises(ValueError, match="one-hot nucleotides"):
        sequence_model.sequence_statistics(torch.zeros(1, 5, 10))
