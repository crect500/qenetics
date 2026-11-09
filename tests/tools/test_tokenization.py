import numpy as np
import pytest
import torch

from qenetics.tools import tokenization

# Nucleotide codes: A = 0, T = 1, G = 2, C = 3, -1 unknown.
_CODES: dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3, "N": -1}


def _codes(*sequences: str) -> np.ndarray:
    return np.array(
        [[_CODES[letter] for letter in sequence] for sequence in sequences],
        dtype=np.int8,
    )


def _letters(codes: np.ndarray) -> str:
    return "".join("ATGCN"[code] for code in codes)


def _cpg_windows(
    window_quantity: int, length: int, seed: int = 0
) -> np.ndarray:
    codes = (
        np.random.default_rng(seed)
        .integers(0, 4, (window_quantity, length))
        .astype(np.int8)
    )
    centre: int = (length - 1) // 2
    codes[:, centre] = _CODES["C"]
    codes[:, centre + 1] = _CODES["G"]
    return codes


def test_cpg_flanks() -> None:
    right, left = tokenization.cpg_flanks(_codes("ANTCGAC"))
    # The right flank follows the G; the left flank is the reverse
    # complement of ANT, read from the base next to the C.
    assert _letters(right[0]) == "AC"
    assert _letters(left[0]) == "ANT"


def test_check_cpg_centres() -> None:
    tokenization.check_cpg_centres(_codes("AACGT", "TTCGA"))
    with pytest.raises(ValueError, match=r"1 windows do not hold a CpG"):
        tokenization.check_cpg_centres(_codes("AACGT", "TTGCA"))
    with pytest.raises(ValueError, match="length >= 3"):
        tokenization.check_cpg_centres(_codes("CG"))


@pytest.fixture(scope="module")
def tokenizer() -> tokenization.Tokenizer:
    return tokenization.train_cpg_bpe_tokenizer(_cpg_windows(200, 61), 16)


def test_train_cpg_bpe_tokenizer(tokenizer: tokenization.Tokenizer) -> None:
    assert tokenizer.token_to_id(tokenization.PAD_TOKEN) == tokenization.PAD_ID
    assert tokenizer.token_to_id(tokenization.CPG_TOKEN) == tokenization.CPG_ID
    assert tokenizer.get_vocab_size() <= 16
    tokenization.check_tokenizer(tokenizer, 16)
    with pytest.raises(ValueError, match="more than the vocabulary size 8"):
        tokenization.check_tokenizer(tokenizer, 8)


@pytest.mark.parametrize("vocabulary_size", [6, 40000])
def test_train_cpg_bpe_tokenizer_size(vocabulary_size: int) -> None:
    with pytest.raises(ValueError, match="vocabulary size must be between"):
        tokenization.train_cpg_bpe_tokenizer(
            _cpg_windows(4, 11), vocabulary_size
        )


def test_cpg_bpe_tokens(tokenizer: tokenization.Tokenizer) -> None:
    codes = _cpg_windows(3, 61, seed=1)
    flank_token_quantity: int = 40
    tokens = tokenization.cpg_bpe_tokens(
        codes, tokenizer, flank_token_quantity=flank_token_quantity
    )
    assert tokens.shape == (3, 81) and tokens.dtype == np.int16
    assert (tokens[:, flank_token_quantity] == tokenization.CPG_ID).all()
    vocabulary: dict[int, str] = {
        token_id: token for token, token_id in tokenizer.get_vocab().items()
    }
    right, left = tokenization.cpg_flanks(codes)
    for window in range(3):
        right_ids = [
            token
            for token in tokens[window, flank_token_quantity + 1 :]
            if token != tokenization.PAD_ID
        ]
        # Left tokens are stored far to near.
        left_ids = [
            token
            for token in tokens[window, :flank_token_quantity][::-1]
            if token != tokenization.PAD_ID
        ]
        assert "".join(vocabulary[token] for token in right_ids) == _letters(
            right[window]
        )
        assert "".join(vocabulary[token] for token in left_ids) == _letters(
            left[window]
        )
        # Padding is far from the CpG.
        assert (
            tokens[window, : flank_token_quantity - len(left_ids)] == 0
        ).all()


def test_cpg_bpe_tokens_truncate(tokenizer: tokenization.Tokenizer) -> None:
    tokens = tokenization.cpg_bpe_tokens(
        _cpg_windows(2, 61), tokenizer, flank_token_quantity=2
    )
    assert tokens.shape == (2, 5)
    assert (tokens != tokenization.PAD_ID).all()


def test_cpg_kmer_tokens() -> None:
    codes = _codes("TTGAT" + "CG" + "CATNA")
    tokens = tokenization.cpg_kmer_tokens(codes, 2, flank_token_quantity=3)
    # Right flank CATNA: CA, then TN, which holds an unknown nucleotide. Left
    # flank TTGAT reversed and complemented: ATCAA, i.e. AT, CA nearest first.
    index = {
        name: 1 + 4 * _CODES[name[0]] + _CODES[name[1]] for name in ("AT", "CA")
    }
    assert tokens.tolist() == [
        [0, index["CA"], index["AT"], 0, index["CA"], 0, 0]
    ]
    assert tokenization.kmer_vocabulary_size(3) == 65
    with pytest.raises(ValueError, match="k-mer length"):
        tokenization.kmer_vocabulary_size(0)
    with pytest.raises(ValueError, match="k-mer length"):
        tokenization.kmer_vocabulary_size(8)


def test_painted_tokens() -> None:
    # Ids 5 and 6 span 2 and 3 bases; left tokens are stored far to near.
    lengths = np.zeros(8, dtype=np.int64)
    lengths[5], lengths[6] = 2, 3
    tokens = np.array([[0, 6, 5, 1, 5, 6, 0]], dtype=np.int16)
    painted = tokenization.painted_tokens(tokens, lengths, 13)
    # Centre C at 6: left bases 5..0 near to far get 5, 5, 6, 6, 6, then
    # [PAD]; right bases 8.. get 5, 5, 6, 6, 6.
    assert painted.tolist() == [[0, 6, 6, 6, 5, 5, 1, 1, 5, 5, 6, 6, 6]]


def test_pack_painted_bases_and_lookup() -> None:
    painted = np.array([[0, 2, 1]], dtype=np.int16)
    codes = _codes("NCG")
    packed = tokenization.pack_painted_bases(painted, codes)
    lookup = tokenization.token_lookup(tokenization.PAINTED_BASES, 5 * 4)
    channels = lookup[torch.from_numpy(packed.astype(np.int64))]
    # Unknown base and [PAD]: zeros. C with token 2, G with token 1.
    assert channels[0, 0].sum() == 0
    assert channels[0, 1].tolist() == [0, 0, 0, 1, 0, 0, 1, 0]
    assert channels[0, 2].tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError, match="same shape"):
        tokenization.pack_painted_bases(painted, _codes("NC"))


def test_token_lookup() -> None:
    bpe = tokenization.token_lookup(tokenization.CPG_BPE, 64)
    assert bpe.shape == (64, 64) and bpe[0].sum() == 0 and bpe[5, 5] == 1
    kmer = tokenization.token_lookup(tokenization.CPG_KMER, 65)
    assert kmer.shape == (65, 64) and kmer[0].sum() == 0 and kmer[1, 0] == 1
    assert tokenization.token_lookup(tokenization.PAINTED_BASES, 320).shape == (
        320,
        68,
    )
    with pytest.raises(ValueError, match="multiple of 5"):
        tokenization.token_lookup(tokenization.PAINTED_BASES, 64)
    with pytest.raises(ValueError, match="Unknown tokenization"):
        tokenization.token_lookup("invalid", 64)


def test_input_name() -> None:
    assert tokenization.input_name(tokenization.CPG_BPE, 64) == "cpg_bpe64"
    assert tokenization.input_name(tokenization.CPG_KMER, 3) == "cpg_kmer3"
