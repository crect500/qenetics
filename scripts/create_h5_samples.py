from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

from qenetics.tools.data import (
    create_h5_dataset_from_methylation_profiles,
)


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        prog="create_h5_sampes",
        description="Store sequences and their methylations as H5 files",
    )
    parser.add_argument(
        "-i",
        "--methylation_directory",
        dest="methylation_directory",
        type=Path,
        required=True,
        help="Filepath of the CpG methylation profiles.",
    )
    parser.add_argument(
        "-f",
        "--fasta-filepath",
        dest="fasta_filepath",
        type=Path,
        required=True,
        help="Filepath of the associated reference genome FASTA file.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        dest="output_directory",
        type=Path,
        required=True,
        help="The filepath of the directory in which to write the H5 files.",
    )
    parser.add_argument(
        "-s",
        "--sequence-length",
        dest="sequence_length",
        type=int,
        required=True,
        help="The length of the window of nucleotides centered on and "
        "including each CpG site, e.g. 1001.",
    )
    parser.add_argument(
        "-m",
        "--minimum_samples",
        dest="minimum_samples",
        type=int,
        required=False,
        default=1,
        help="The minimum reads of a CpG site in an experiment, summed over "
        "both strands, for the site to be labeled, e.g. 4 for scRRBS-seq.",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        dest="excluded_experiments",
        nargs="+",
        default=[],
        help="The names of experiments to leave out, matching the part of "
        "each methylation filename before the first '.', e.g. the cells "
        "RSC27_4, RSC27_7 and Ca26.",
    )
    parser.add_argument(
        "--binarize",
        dest="binarize",
        action=BooleanOptionalAction,
        default=True,
        help="Label sites methylated if they have more methylated than "
        "unmethylated reads, and unmethylated otherwise. Use --no-binarize "
        "to label sites with methylation ratios instead.",
    )
    parser.add_argument(
        "--allow-N",
        dest="allow_N",
        action="store_true",
        help="Keep windows holding nucleotides other than A, T, C and G, "
        "encoded as all zeros. By default, such windows are dropped.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    create_h5_dataset_from_methylation_profiles(
        args.methylation_directory,
        args.fasta_filepath,
        args.output_directory,
        args.sequence_length,
        args.minimum_samples,
        excluded_experiments=args.excluded_experiments,
        binarize=args.binarize,
        allow_N=args.allow_N,
    )
