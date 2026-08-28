from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools import converters


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        "Convert a one-hot encoded h5 dataset to integer encoding"
    )
    parser.add_argument(
        "-d",
        "--input_directory",
        dest="input_directory",
        type=Path,
        required=True,
        help="The directory containing one-hot encoded data.",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        type=Path,
        required=True,
        help="The directory to write integer-encoded data.",
    )
    parser.add_argument(
        "-z",
        "--include-zero",
        dest="include_zero",
        type=bool,
        required=False,
        default=False,
        help="Whether or not to treat an array of zeros ('N') as a valid encoding",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_script_args()
    for filepath in args.input_directory.iterdir():
        converters.h5_one_hot_to_integer(
            filepath, args.output_directory, include_zero=args.include_zero
        )
