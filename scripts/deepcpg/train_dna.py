from argparse import ArgumentParser, Namespace
from pathlib import Path

from qenetics.deepcpg import train


def _parse_script_args() -> Namespace:
    parser = ArgumentParser("train_dna")
    parser.add_argument(
        "-c", "--config", dest="config_filepath", type=Path, required=True
    )

    return parser.parse_args()


def train_deepcpg() -> None:
    pass


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    train_deepcpg()
