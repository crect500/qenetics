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
    chromosomes: list[int],
    sequence_length: int,
    output_filepath: Path,
    minimum_samples: int = 1,
) -> None:
    create_sequence_dataset(
        methylation_filepath,
        fasta_filepath,
        sequence_length,
        chromosomes,
        minimum_samples,
        output_filepath,
    )


if __name__ == "__main__":
    args = _parse_script_args()
    create_samples(
        args.fasta_filepath,
        args.methylation_filepath,
        args.chromosomes,
        args.sequence_length,
        args.output_filepath,
        args.minimum_samples,
    )
