"""
CpG-centred tokenization of sequence windows.

Every window holds its CpG at the centre, with the C at position
(L - 1) // 2. The right flank, from the base after the G, is read 5' to 3'.
The left flank is reverse-complemented, i.e. read 5' to 3' on the opposite
strand, also starting next to the CpG. Both flanks are thus sequence read
away from a CpG, so one vocabulary tokenizes both, and token j of either flank
is the j-th token away from the site. A tokenized window holds the left
flank's tokens far to near, a [CPG] token, and the right flank's tokens near
to far, each flank padded with [PAD] to `flank_token_quantity` tokens.

- `CPG_BPE`: byte-pair encoding learned on training-chromosome flanks. A
  vocabulary of 64 or 256 tokens makes a token a basis state of 6 or 8 qubits.
- `CPG_KMER`: non-overlapping k-mers, the fixed-length control for BPE. Ids
  are 1 + the k-mer's base-4 index, and 0 for no token or a k-mer holding an
  unknown nucleotide.
- `PAINTED`: every base carries the id of the BPE token covering it, [CPG]
  at the CpG and [PAD] beyond the tokenized flanks.
- `PAINTED_BASES`: as painted, packed with the base itself into one id,
  `PAINTED_BASE_STATES` * token + base code + 1.

Nucleotide codes are those of `data._sequence_codes`: A = 0, T = 1, G = 2,
C = 3 and -1 for an unknown nucleotide.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray
from tokenizers import Tokenizer, models, trainers

from qenetics.tools.dna import Nucleotide

logger = logging.getLogger(__name__)

CPG_BPE: str = "cpg_bpe"
CPG_KMER: str = "cpg_kmer"
PAINTED: str = "painted"
PAINTED_BASES: str = "painted_bases"
TOKENIZATIONS: tuple[str, ...] = (CPG_BPE, CPG_KMER, PAINTED, PAINTED_BASES)
PAD_TOKEN: str = "[PAD]"
CPG_TOKEN: str = "[CPG]"
PAD_ID: int = 0
CPG_ID: int = 1
BPE_ALPHABET: str = "ACGTN"
FLANK_TOKEN_QUANTITY: int = 128
VOCABULARY_SAMPLE_QUANTITY: int = 100_000
VOCABULARY_SAMPLE_SEED: int = 7
# An unknown nucleotide and the four nucleotides of a painted-bases id.
PAINTED_BASE_STATES: int = 5
# Token ids are stored as 16-bit integers.
MAXIMUM_TOKEN_ID: int = np.iinfo(np.int16).max
ENCODE_BATCH_SIZE: int = 10_000

# The letter of each nucleotide code, with the last for unknown nucleotides.
_LETTERS: NDArray[np.str_] = np.array(["A", "T", "G", "C", "N"])
# The complement of each nucleotide code, with unknown nucleotides unchanged.
_COMPLEMENTS: NDArray[np.int8] = np.array([1, 0, 3, 2, -1], dtype=np.int8)


def _check_tokenization(tokenization: str) -> None:
    if tokenization not in TOKENIZATIONS:
        raise ValueError(
            f"Unknown tokenization {tokenization}; expected one of "
            f"{', '.join(TOKENIZATIONS)}"
        )


def _check_codes(codes: NDArray[np.int8]) -> None:
    if codes.ndim != 2 or codes.shape[1] < 3:
        raise ValueError(
            "Nucleotide codes must have shape (windows, length >= 3), not "
            f"{codes.shape}"
        )


def check_cpg_centres(codes: NDArray[np.int8]) -> None:
    """
    Verify that every window holds a CpG at its centre.

    Raises
    ------
    ValueError if a window's centre is not a C followed by a G.
    """
    _check_codes(codes)
    centre: int = (codes.shape[1] - 1) // 2
    is_cpg: NDArray[np.bool_] = (codes[:, centre] == Nucleotide.C) & (
        codes[:, centre + 1] == Nucleotide.G
    )
    if not is_cpg.all():
        windows: NDArray[np.int64] = np.flatnonzero(~is_cpg)
        raise ValueError(
            f"{len(windows)} windows do not hold a CpG at positions {centre} "
            f"and {centre + 1}, e.g. windows {windows[:5].tolist()}"
        )


def cpg_flanks(
    codes: NDArray[np.int8],
) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
    """
    Split windows into their flanks, each read away from the CpG.

    Args
    ----
    codes: Nucleotide codes, (windows, length), with the CpG's C at
        (length - 1) // 2.

    Returns
    -------
    The right flanks, read from the base after the G, and the
    reverse-complemented left flanks, read from the base before the C.
    """
    _check_codes(codes)
    centre: int = (codes.shape[1] - 1) // 2
    right: NDArray[np.int8] = codes[:, centre + 2 :]
    left: NDArray[np.int8] = _COMPLEMENTS[codes[:, :centre][:, ::-1]]
    return right, left


def _flank_strings(flanks: NDArray[np.int8]) -> list[str]:
    return ["".join(letters) for letters in _LETTERS[flanks]]


def train_cpg_bpe_tokenizer(
    codes: NDArray[np.int8],
    vocabulary_size: int,
    *,
    sample_quantity: int = VOCABULARY_SAMPLE_QUANTITY,
    seed: int = VOCABULARY_SAMPLE_SEED,
) -> Tokenizer:
    """
    Learn a byte-pair encoding of the flanks of CpG-centred windows.

    Args
    ----
    codes: Nucleotide codes of training windows, (windows, length).
    vocabulary_size: The vocabulary's size, including [PAD] (id 0), [CPG]
        (id 1) and the letters of `BPE_ALPHABET`.
    sample_quantity: The windows to learn from, a random subset if there are
        more.
    seed: The seed of the subset.

    Returns
    -------
    The tokenizer.

    Raises
    ------
    ValueError if the vocabulary cannot hold the special tokens and the
    alphabet, or its ids do not fit in 16-bit integers.
    """
    minimum_size: int = 2 + len(BPE_ALPHABET)
    if not minimum_size <= vocabulary_size <= MAXIMUM_TOKEN_ID + 1:
        raise ValueError(
            f"The vocabulary size must be between {minimum_size} and "
            f"{MAXIMUM_TOKEN_ID + 1}, not {vocabulary_size}"
        )
    check_cpg_centres(codes)

    rows: NDArray[np.int64] = np.sort(
        np.random.default_rng(seed).choice(
            len(codes), min(sample_quantity, len(codes)), replace=False
        )
    )
    right, left = cpg_flanks(codes[rows])
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.train_from_iterator(
        _flank_strings(right) + _flank_strings(left),
        trainer=trainers.BpeTrainer(
            vocab_size=vocabulary_size,
            special_tokens=[PAD_TOKEN, CPG_TOKEN],
            initial_alphabet=list(BPE_ALPHABET),
            show_progress=False,
        ),
    )
    logger.info(
        "Learned %d BPE tokens from %d windows",
        tokenizer.get_vocab_size(),
        len(rows),
    )
    return tokenizer


def check_tokenizer(tokenizer: Tokenizer, vocabulary_size: int) -> None:
    """
    Raises
    ------
    ValueError if the tokenizer's special tokens are not [PAD] = 0 and
    [CPG] = 1, or its vocabulary exceeds the vocabulary size.
    """
    if (
        tokenizer.token_to_id(PAD_TOKEN) != PAD_ID
        or tokenizer.token_to_id(CPG_TOKEN) != CPG_ID
    ):
        raise ValueError(
            f"The tokenizer must map {PAD_TOKEN} to {PAD_ID} and {CPG_TOKEN} "
            f"to {CPG_ID}"
        )
    if tokenizer.get_vocab_size() > vocabulary_size:
        raise ValueError(
            f"The tokenizer has {tokenizer.get_vocab_size()} tokens, more "
            f"than the vocabulary size {vocabulary_size}"
        )


def token_lengths(
    tokenizer: Tokenizer, vocabulary_size: int
) -> NDArray[np.int64]:
    """The nucleotides each token id spans, 0 for special tokens."""
    lengths: NDArray[np.int64] = np.zeros(vocabulary_size, dtype=np.int64)
    for token, token_id in tokenizer.get_vocab().items():
        if token not in (PAD_TOKEN, CPG_TOKEN):
            lengths[token_id] = len(token)
    return lengths


def _empty_tokens(
    window_quantity: int, flank_token_quantity: int
) -> NDArray[np.int16]:
    if flank_token_quantity < 1:
        raise ValueError(
            "The flank token quantity must be at least 1, not "
            f"{flank_token_quantity}"
        )
    tokens: NDArray[np.int16] = np.full(
        (window_quantity, 2 * flank_token_quantity + 1), PAD_ID, dtype=np.int16
    )
    return tokens


def cpg_bpe_tokens(
    codes: NDArray[np.int8],
    tokenizer: Tokenizer,
    *,
    flank_token_quantity: int = FLANK_TOKEN_QUANTITY,
) -> NDArray[np.int16]:
    """
    Tokenize windows by a CpG-centred byte-pair encoding.

    Args
    ----
    codes: Nucleotide codes, (windows, length).
    tokenizer: A tokenizer from `train_cpg_bpe_tokenizer`.
    flank_token_quantity: The tokens K kept per flank.

    Returns
    -------
    The token ids, (windows, 2K + 1): the left flank far to near, [CPG], and
    the right flank near to far, padded with [PAD].

    Raises
    ------
    ValueError if a window is not CpG-centred or K is less than 1.
    """
    check_cpg_centres(codes)
    tokens: NDArray[np.int16] = _empty_tokens(len(codes), flank_token_quantity)
    tokens[:, flank_token_quantity] = CPG_ID
    for start in range(0, len(codes), ENCODE_BATCH_SIZE):
        right, left = cpg_flanks(codes[start : start + ENCODE_BATCH_SIZE])
        right_encodings = tokenizer.encode_batch(_flank_strings(right))
        left_encodings = tokenizer.encode_batch(_flank_strings(left))
        for offset, (left_encoding, right_encoding) in enumerate(
            zip(left_encodings, right_encodings, strict=True)
        ):
            left_ids: list[int] = left_encoding.ids[:flank_token_quantity]
            right_ids: list[int] = right_encoding.ids[:flank_token_quantity]
            row: int = start + offset
            tokens[
                row, flank_token_quantity - len(left_ids) : flank_token_quantity
            ] = left_ids[::-1]
            tokens[
                row,
                flank_token_quantity + 1 : flank_token_quantity
                + 1
                + len(right_ids),
            ] = right_ids
    return tokens


def kmer_vocabulary_size(kmer_length: int) -> int:
    """
    The number of k-mer token ids: every k-mer and 0 for no token.

    Raises
    ------
    ValueError if k is less than 1 or the ids do not fit in 16-bit integers.
    """
    if kmer_length < 1 or 4**kmer_length > MAXIMUM_TOKEN_ID:
        raise ValueError(
            "The k-mer length must be at least 1 and give fewer than "
            f"{MAXIMUM_TOKEN_ID} k-mers, not {kmer_length}"
        )
    return 4**kmer_length + 1


def _kmer_ids(
    flanks: NDArray[np.int8], kmer_length: int, flank_token_quantity: int
) -> NDArray[np.int16]:
    quantity: int = min(flank_token_quantity, flanks.shape[1] // kmer_length)
    kmers: NDArray[np.int64] = (
        flanks[:, : quantity * kmer_length]
        .reshape(len(flanks), quantity, kmer_length)
        .astype(np.int64)
    )
    indices: NDArray[np.int64] = (
        kmers * 4 ** np.arange(kmer_length - 1, -1, -1)
    ).sum(2)
    return np.where((kmers >= 0).all(2), indices + 1, PAD_ID).astype(np.int16)


def cpg_kmer_tokens(
    codes: NDArray[np.int8],
    kmer_length: int,
    *,
    flank_token_quantity: int = FLANK_TOKEN_QUANTITY,
) -> NDArray[np.int16]:
    """
    Tokenize windows by CpG-anchored non-overlapping k-mers.

    Returns
    -------
    The token ids, (windows, 2K + 1), laid out as `cpg_bpe_tokens`, with 0
    at the centre and wherever there is no k-mer.

    Raises
    ------
    ValueError if a window is not CpG-centred, or k or K is invalid.
    """
    kmer_vocabulary_size(kmer_length)
    check_cpg_centres(codes)
    tokens: NDArray[np.int16] = _empty_tokens(len(codes), flank_token_quantity)
    right, left = cpg_flanks(codes)
    right_ids: NDArray[np.int16] = _kmer_ids(
        right, kmer_length, flank_token_quantity
    )
    left_ids: NDArray[np.int16] = _kmer_ids(
        left, kmer_length, flank_token_quantity
    )
    tokens[
        :,
        flank_token_quantity + 1 : flank_token_quantity
        + 1
        + right_ids.shape[1],
    ] = right_ids
    tokens[
        :, flank_token_quantity - left_ids.shape[1] : flank_token_quantity
    ] = left_ids[:, ::-1]
    return tokens


def painted_tokens(
    tokens: NDArray[np.int16],
    lengths: NDArray[np.int64],
    sequence_length: int,
) -> NDArray[np.int16]:
    """
    Give every base the id of the CpG-centred token covering it.

    Args
    ----
    tokens: Tokens from `cpg_bpe_tokens`, (windows, 2K + 1).
    lengths: The nucleotides each token id spans, from `token_lengths`.
    sequence_length: The length of the tokenized windows.

    Returns
    -------
    The painted ids, (windows, length): [CPG] at the CpG and [PAD] beyond the
    tokenized flanks.
    """
    flank_token_quantity: int = tokens.shape[1] // 2
    centre: int = (sequence_length - 1) // 2
    right_length: int = sequence_length - centre - 2
    painted: NDArray[np.int16] = np.full(
        (len(tokens), sequence_length), PAD_ID, dtype=np.int16
    )
    painted[:, centre : centre + 2] = CPG_ID
    for row, window_tokens in enumerate(tokens):
        right: NDArray[np.int16] = window_tokens[flank_token_quantity + 1 :]
        right_bases: NDArray[np.int16] = np.repeat(right, lengths[right])[
            :right_length
        ]
        painted[row, centre + 2 : centre + 2 + len(right_bases)] = right_bases
        # The left flank's tokens, near to far.
        left: NDArray[np.int16] = window_tokens[:flank_token_quantity][::-1]
        left_bases: NDArray[np.int16] = np.repeat(left, lengths[left])[:centre]
        painted[row, centre - len(left_bases) : centre] = left_bases[::-1]
    return painted


def pack_painted_bases(
    painted: NDArray[np.int16], codes: NDArray[np.int8]
) -> NDArray[np.int16]:
    """
    Pack each base and its painted token into one id.

    Raises
    ------
    ValueError if the shapes differ or the ids do not fit in 16-bit integers.
    """
    if painted.shape != codes.shape:
        raise ValueError(
            f"Painted tokens {painted.shape} and nucleotide codes "
            f"{codes.shape} must have the same shape"
        )
    if (
        painted.size
        and painted.max() * PAINTED_BASE_STATES + 4 > MAXIMUM_TOKEN_ID
    ):
        raise ValueError("Painted token ids are too large to pack with bases")
    return (
        painted.astype(np.int16) * PAINTED_BASE_STATES
        + codes.astype(np.int16)
        + 1
    )


def input_name(tokenization: str, size: int) -> str:
    """
    Name the input a tokenization gives, e.g. 'cpg_bpe64' or 'cpg_kmer3'.

    Args
    ----
    tokenization: One of `TOKENIZATIONS`.
    size: The k-mer length for `CPG_KMER`, else the BPE vocabulary size.
    """
    _check_tokenization(tokenization)
    return f"{tokenization}{size}"


def token_lookup(tokenization: str, vocabulary_size: int) -> torch.Tensor:
    """
    Map the token ids of a tokenization to model input channels.

    Args
    ----
    tokenization: One of `TOKENIZATIONS`.
    vocabulary_size: The number of token ids the files hold.

    Returns
    -------
    The (ids, channels) lookup table: one-hot tokens with [PAD] as zeros for
    BPE and painted tokens, the 4^k k-mers with id 0 as zeros, or the one-hot
    base followed by the one-hot painted token for painted bases.

    Raises
    ------
    ValueError if the tokenization is unknown or the vocabulary size does
    not suit it.
    """
    _check_tokenization(tokenization)
    if tokenization in (CPG_BPE, PAINTED):
        lookup: torch.Tensor = torch.eye(vocabulary_size)
        lookup[PAD_ID] = 0.0
        return lookup

    if tokenization == CPG_KMER:
        return torch.cat(
            [
                torch.zeros(1, vocabulary_size - 1),
                torch.eye(vocabulary_size - 1),
            ]
        )

    if vocabulary_size % PAINTED_BASE_STATES:
        raise ValueError(
            "Painted bases need a multiple of "
            f"{PAINTED_BASE_STATES} ids, not {vocabulary_size}"
        )
    token_quantity: int = vocabulary_size // PAINTED_BASE_STATES
    bases: torch.Tensor = torch.cat([torch.zeros(1, 4), torch.eye(4)])
    tokens: torch.Tensor = torch.eye(token_quantity)
    tokens[PAD_ID] = 0.0
    ids: torch.Tensor = torch.arange(vocabulary_size)
    return torch.cat(
        [
            bases[ids % PAINTED_BASE_STATES],
            tokens[ids // PAINTED_BASE_STATES],
        ],
        1,
    )
