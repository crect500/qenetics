from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools import report


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        prog="build_training_report",
        description="Add training run records to the navigable HTML report. "
        "Runs are added incrementally: only runs whose pages are missing or "
        "older than their records are rendered.",
    )
    parser.add_argument(
        "-r",
        "--records-directory",
        dest="records_directory",
        type=Path,
        required=True,
        help="The directory holding the training run records.",
    )
    parser.add_argument(
        "--run",
        dest="run_directory",
        type=Path,
        required=False,
        default=None,
        help="Add or refresh only this run's record directory.",
    )
    parser.add_argument(
        "--rebuild",
        dest="rebuild",
        action="store_true",
        help="Render every run's pages, even if they are up to date.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    rendered: list[Path] = report.build_report(
        args.records_directory,
        run_directory=args.run_directory,
        rebuild=args.rebuild,
    )
    print(f"Rendered {len(rendered)} run(s).")
    print(f"Report: {args.records_directory / report.INDEX_FILENAME}")
