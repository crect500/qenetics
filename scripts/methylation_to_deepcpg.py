from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools import deepcpg_utils


def _parse_script_args() -> Namespace:
    parser = ArgumentParser("Converts a methylation file into deepcpg format")
    parser.add_argument(
        "-m",
        "--methylation",
        dest="methylation_file",
        type=Path,
        required=True,
        help="The filepath of the methylation data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="The directory in which to store deepcpg format methylation data.",
    )
    parser.add_argument(
        "-c",
        "--count",
        dest="minimum_samples",
        type=int,
        required=False,
        default=1,
        help="An optional setting for the minimum numbers of samples to be considered.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        type=float,
        required=False,
        default=0.5,
        help="An optional threshold at which to consider a site methylated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_script_args()
    output_file: Path = args.output_directory / (
        args.methylation_file.stem + f"deepcpg_m{args.minimum_samples}.tsv"
    )
    deepcpg_utils.write_all_deepcpg_methylations(
        args.methylation_file, output_file, args.threshold
    )
