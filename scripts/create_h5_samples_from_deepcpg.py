import logging
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

from qenetics.tools.data import create_h5_dataset_from_deepcpg_files


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        prog="create_h5_samples_from_deepcpg",
        description="Convert DeepCpG data files, e.g. "
        "'c1_3801088-3833856.h5', to one H5 file of sequences and their "
        "methylations per chromosome.",
    )
    parser.add_argument(
        "-i",
        "--deepcpg-directory",
        dest="deepcpg_directory",
        type=Path,
        required=True,
        help="The directory of DeepCpG data files.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        dest="output_directory",
        type=Path,
        required=True,
        help="The filepath of the directory in which to write the H5 files.",
    )
    parser.add_argument(
        "-s",
        "--sequence-length",
        dest="sequence_length",
        type=int,
        default=None,
        help="The length of the window of nucleotides centered on and "
        "including each CpG site, cut from the DeepCpG windows. Defaults to "
        "the length of the DeepCpG windows.",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        dest="excluded_experiments",
        nargs="+",
        default=[],
        help="The names of experiments to leave out, matching the dataset "
        "names in the 'outputs' group of the DeepCpG data files.",
    )
    parser.add_argument(
        "--binarize",
        dest="binarize",
        action=BooleanOptionalAction,
        default=True,
        help="Label sites methylated if their methylation ratio, averaged "
        "over both strands, exceeds 0.5, and unmethylated otherwise. Use "
        "--no-binarize to label sites with methylation ratios instead.",
    )
    parser.add_argument(
        "--allow-N",
        dest="allow_N",
        action="store_true",
        help="Keep windows holding nucleotides other than A, T, C and G, "
        "encoded as all zeros. By default, such windows are dropped.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args: Namespace = _parse_script_args()
    create_h5_dataset_from_deepcpg_files(
        args.deepcpg_directory,
        args.output_directory,
        args.sequence_length,
        excluded_experiments=args.excluded_experiments,
        binarize=args.binarize,
        allow_N=args.allow_N,
    )
