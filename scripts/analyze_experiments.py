from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools import analysis


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        "analyze_experiments",
        description="Parse sequences and their methylation data."
        " Write results to a csv file",
    )
    parser.add_argument(
        "-d",
        "--experiments-directory",
        dest="experiments_directory",
        type=Path,
        required=True,
        help="The filepath of the directory that contains methylation "
        "profiles.",
    )
    parser.add_argument(
        "-f",
        "--fasta-file",
        dest="fasta_filepath",
        type=Path,
        required=True,
        help="The filepath to the FASTA reference genome associated "
        "with the methylation profiles.",
    )
    parser.add_argument(
        "-l",
        "--sequence-length",
        dest="sequence_length",
        type=int,
        required=True,
        help="The length of sequence surrounding the CpG site to extract.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_filepath",
        type=Path,
        required=True,
        help="The filepath of the CSV output.",
    )
    parser.add_argument(
        "-m",
        "--minimum-samples",
        dest="minimum_samples",
        type=int,
        required=True,
        default=1,
        help="The minimum number of samples to consider a methylation "
        "profile CpG site.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    stats_by_experiment: dict[str, analysis.ExperimentStatistics] = (
        analysis.get_dataset_stats(
            list(args.experiments_directory.iterdir()),
            args.sequence_length,
            args.fasta_filepath,
            minimum_samples=args.minimum_samples,
        )
    )
    analysis.write_experiment_statistics(
        stats_by_experiment, args.output_filepath
    )
