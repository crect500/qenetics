from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools.converters import extract_deepcpg_experiment_to_qcpg


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        "Extracts the methylation profiles of a single experiment "
        "from a deepcpg H5 dataset"
    )
    parser.add_argument(
        "-d",
        "--deepcpg_directory",
        dest="deepcpg_directory",
        type=Path,
        required=True,
        help="The directory containing deepcpg H5 file input",
    )
    parser.add_argument(
        "-q",
        "--qcpg-directory",
        dest="qcpg_directory",
        type=Path,
        required=True,
        help="The directory at which to store the extracted data in "
        "a format that the qcpg module can process",
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        dest="experiment_name",
        type=str,
        required=True,
        help="The experiment name to filter by",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        type=float,
        required=False,
        default=-1.0,
        help="The threshold at which to consider a sample methylated.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_script_args()
    extract_deepcpg_experiment_to_qcpg(
        args.deepcpg_directory,
        args.qcpg_directory,
        args.experiment_name,
        args.threshold,
    )
