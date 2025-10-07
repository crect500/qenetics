from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.tools.cpg_sampler import (
    convert_methylation_profiles,
    MethylationFormat,
)


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        prog="convert_methylation",
        description="Convert cpg methylation profiles between formats.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_filepath",
        type=Path,
        required=True,
        help="The filepath of a file or directory containing methylation data.",
    )
    parser.add_argument(
        "-o--output_directory",
        dest="output_directory",
        type=Path,
        required=True,
        help="The directory in which to write converted files.",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        type=str,
        required=True,
        help="The desired format of the output files. Options are cov, cpg, deepcpg",
    )
    parser.add_argument(
        "-m",
        "--minimum-samples",
        dest="minimum_samples",
        type=int,
        required=False,
        default=1,
        help="The minimum number of experiments for a site to be considered.",
    )

    return parser.parse_args()


def _parse_format(format_string: str) -> MethylationFormat:
    if format_string.lower() == "cov":
        return MethylationFormat.COV

    if format_string.lower() == "cpg":
        return MethylationFormat.CPG

    if format_string.lower() == "deepcpg":
        return MethylationFormat.DEEPCPG

    raise ValueError(f"Format {format_string} not recognized.")


def transfer_methylation_profiles(
    input_filepath: Path,
    output_filepath: Path,
    output_format_string: MethylationFormat,
    minimum_samples: int = 1,
) -> None:
    output_format: MethylationFormat = _parse_format(output_format_string)
    if input_filepath.is_dir():
        for methylation_file in input_filepath.iterdir():
            convert_methylation_profiles(
                methylation_file,
                output_filepath,
                output_format,
                minimum_samples,
            )
    else:
        convert_methylation_profiles(
            input_filepath, output_filepath, output_format, minimum_samples
        )


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    transfer_methylation_profiles(
        args.input_filepath,
        args.output_directory,
        args.output_format,
        args.minimum_samples,
    )
