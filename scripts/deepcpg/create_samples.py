from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.deepcpg.deepcpg_utils import create_sequence_dataset


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        prog="create_samples",
        description="Creates sequence samples from a FASTA "
        "reference genome file and methylation "
        "profiles.",
    )
    parser.add_argument(
        "-f",
        "--fasta_file",
        dest="fasta_filepath",
        type=Path,
        required=True,
        help="The filepath of a FASTA format reference genome file.",
    )
    parser.add_argument(
        "-m",
        "--methylation-file",
        dest="methylation_filepath",
        type=Path,
        required=True,
        help="The filepath of a single cell's methylation profile.",
    )
    parser.add_argument(
        "-c",
        "--chromosomes",
        dest="chromosomes",
        type=str,
        required=True,
        nargs="+",
        help="A list of chromosome numbers to filter.",
    )
    parser.add_argument(
        "-s",
        "--sequence-length",
        dest="sequence_length",
        type=int,
        required=True,
        help="The length of the sequence to extract for each CpG site.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_filepath",
        type=Path,
        required=True,
        help="The filepath to write the results to.",
    )
    parser.add_argument(
        "--minimum-samples",
        dest="minimum_samples",
        type=int,
        required=False,
        default=1,
        help="The minimum samples for a methylation profile to be "
        "considered viable.",
    )

    return parser.parse_args()


def create_samples(
    fasta_filepath: Path,
    methylation_filepath: Path,
    chromosomes: list[str],
    sequence_length: int,
    output_filepath: Path,
    minimum_samples: int = 1,
) -> None:
    create_sequence_dataset(
        methylation_filepath=methylation_filepath,
        fasta_file=fasta_filepath,
        sequence_length=sequence_length,
        chromosomes=chromosomes,
        minimum_samples=minimum_samples,
        output_file=output_filepath,
    )


if __name__ == "__main__":
    args = _parse_script_args()
    create_samples(
        fasta_filepath=args.fasta_filepath,
        methylation_filepath=args.methylation_filepath,
        chromosomes=args.chromosomes,
        sequence_length=args.sequence_length,
        output_filepath=args.output_filepath,
        minimum_samples=args.minimum_samples,
    )
