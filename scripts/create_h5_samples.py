from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools.cpg_sampler import (
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
        help="The window of nucleotides around the CpG site to retrieve.",
    )
    parser.add_argument(
        "-m",
        "--minimum_samples",
        dest="minimum_samples",
        type=int,
        required=False,
        default=1,
        help="The minimum experiments for a site to be considered.",
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
    )
