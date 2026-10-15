"""
Tokenize the CpG-centred windows of a dataset, as in Section 6 of the quantum
models report: CpG-centred BPE, CpG-anchored k-mers, and base-aligned
("painted") BPE tokens with or without the bases.

BPE learns its vocabulary on the training chromosomes and saves it as
tokenizer.json in the output directory. Painted tokens reuse that tokenizer:

    python scripts/tokenize_cpg_windows.py -i hcc -o hcc_bpe64 \\
        --method cpg_bpe --vocabulary-size 64 --training-chromosomes 1 3 5 7 9
    python scripts/tokenize_cpg_windows.py -i hcc -o hcc_painted64 \\
        --method painted --vocabulary-size 64 \\
        --tokenizer hcc_bpe64/tokenizer.json
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools import data, tokenization


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        "tokenize_cpg_windows",
        description="Write CpG-centred token files of a dataset's chromosome "
        "files.",
    )
    parser.add_argument(
        "-i",
        "--input-directory",
        type=Path,
        required=True,
        help="The directory of chr<name>.h5 files of nucleotides.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        required=True,
        help="The directory for the token files.",
    )
    parser.add_argument(
        "--method",
        choices=tokenization.TOKENIZATIONS,
        required=True,
        help="The tokenization.",
    )
    parser.add_argument(
        "--vocabulary-size",
        type=int,
        help="The BPE vocabulary size (64 or 256 in the report), for every "
        "method but cpg_kmer.",
    )
    parser.add_argument(
        "--kmer-length",
        type=int,
        help="The k-mer length (3 or 4 in the report), for cpg_kmer.",
    )
    parser.add_argument(
        "--training-chromosomes",
        nargs="+",
        help="The chromosomes to learn a BPE vocabulary from.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="A tokenizer.json to reuse instead of learning one.",
    )
    parser.add_argument(
        "--flank-tokens",
        type=int,
        default=tokenization.FLANK_TOKEN_QUANTITY,
        help="The tokens kept on each side of the CpG.",
    )
    parser.add_argument(
        "--log-level", choices=["debug", "info"], default="info"
    )
    args: Namespace = parser.parse_args()
    if args.method == tokenization.CPG_KMER and args.kmer_length is None:
        parser.error("cpg_kmer needs --kmer-length")
    if args.method != tokenization.CPG_KMER and args.vocabulary_size is None:
        parser.error(f"{args.method} needs --vocabulary-size")
    if (
        args.method != tokenization.CPG_KMER
        and args.tokenizer is None
        and not args.training_chromosomes
    ):
        parser.error(
            f"{args.method} needs --training-chromosomes or --tokenizer"
        )
    return args


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    logging.basicConfig(
        level=logging.DEBUG if args.log_level == "debug" else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    data.create_token_h5_dataset(
        args.input_directory,
        args.output_directory,
        args.method,
        vocabulary_size=args.vocabulary_size,
        kmer_length=args.kmer_length,
        training_chromosomes=args.training_chromosomes,
        tokenizer_filepath=args.tokenizer,
        flank_token_quantity=args.flank_tokens,
    )
