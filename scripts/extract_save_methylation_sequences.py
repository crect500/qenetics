from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools import cpg_sampler


def _parse_script_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        "extract_save_methylation_sequences",
        usage="python3 "
        "extract_save_methylation_sequence.py "
        "-f [fasfa-file] "
        "-m [methylation-directory] "
        "-l [sequence-length] "
        "-o [output-directory] "
        "-c [minimum-count]",
        description="Extract all methylation sequences "
        "with more than [minimum_count] "
        "samples from reference genome.",
    )
    parser.add_argument(
        "-f",
        "--fasfa-file",
        dest="fasfa_file",
        required=True,
        type=Path,
        help="The filepath of a fasfa reference genome file.",
    )
    parser.add_argument(
        "-m",
        "--methylation-directory",
        dest="methylation_directory",
        required=True,
        type=Path,
        help="The filepath of a directory containing only methlyation profile data "
        "files.",
    )
    parser.add_argument(
        "-l",
        "--sequence-length",
        dest="sequence_length",
        required=True,
        type=int,
        nargs="+",
        help="One or more nucleotide sequence lengths",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        required=True,
        type=Path,
        help="The filepath of the desired output directory.",
    )
    parser.add_argument(
        "-c",
        "--minimum-count",
        dest="minimum_count",
        required=False,
        type=int,
        default=[1],
        nargs="+",
        help="The minimum number of methylation samples for a position to be "
        "considered.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    for sequence_length in args.sequence_length:
        for minimum_count in args.minimum_count:
            cpg_sampler.load_and_save_all_cpg_sequences(
                fasfa_file=args.fasfa_file,
                methylation_directory=list(args.methylation_directory.iterdir()),
                sequence_length=sequence_length,
                output_directory=args.output_directory,
                minimum_count=minimum_count,
            )
