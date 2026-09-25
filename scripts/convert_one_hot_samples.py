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
        "--allow-N",
        "--include-zero",
        dest="allow_N",
        action="store_true",
        help="Treat an array of zeros ('N') as a valid encoding, encoded as 0 "
        "with A, T, G and C encoded as 1 to 4.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_script_args()
    for filepath in args.input_directory.iterdir():
        converters.h5_one_hot_to_integer(
            filepath, args.output_directory, allow_N=args.allow_N
        )
