from argparse import ArgumentParser, Namespace
import logging
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
    parser.add_argument(
        "-l",
        "--log-file",
        dest="log_filepath",
        type=Path,
        required=False,
        help="The filepath at which to write log output.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        required=False,
        default="info",
        help="The level of the logging output.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_script_args()
    if args.log_filepath is not None:
        if args.log_level == "debug":
            logging.basicConfig(
                filename=str(args.log_filepath.absolute()), level=logging.DEBUG
            )
        else:
            logging.basicConfig(
                filename=str(args.log_filepath.absolute()), level=logging.INFO
            )

    extract_deepcpg_experiment_to_qcpg(
        args.deepcpg_directory,
        args.qcpg_directory,
        args.experiment_name,
        args.threshold,
    )
