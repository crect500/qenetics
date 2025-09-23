from argparse import ArgumentParser, Namespace
from csv import DictReader, DictWriter
from pathlib import Path


def _parse_script_args() -> Namespace:
    parser = ArgumentParser("aggregate_methylations")
    parser.add_argument(
        "-d",
        "--directory",
        dest="input_directory",
        type=Path,
        required=True,
        help="data directory",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        dest="output_file",
        type=Path,
        required=True,
        help="Output filename",
    )
    parser.add_argument(
        "-t",
        dest="training",
        action="store_true",
        help="Training if set. Validation otherwise",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    if args.training:
        keyword: str = "training"
    else:
        keyword = "validation"
    with open(args.output_file, "w") as out_fd:
        csv_writer = DictWriter(
            out_fd, fieldnames=["sequence", "ratio_methylated"]
        )
        csv_writer.writeheader()
        for methylation_file in args.input_directory.iterdir():
            if keyword in methylation_file.name:
                with open(methylation_file) as input_fd:
                    csv_reader = DictReader(input_fd)
                    for row in csv_reader:
                        if "N" in row["sequence"]:
                            continue
                        else:
                            csv_writer.writerow(
                                {
                                    "sequence": row["sequence"],
                                    "ratio_methylated": row["ratio_methylated"],
                                }
                            )
