from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split

from qenetics.qcpg import qcpg
from qenetics.tools.cpg_sampler import load_methylation_samples

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        "extract_save_methylation_sequences ",
        usage="python3 extract_save_methylation_sequence.py ",
        description="Train the parameters of a quantum circuit that predicts whether "
        "CpG sites are methylated given an input nucleotide sequence.",
    )
    parser.add_argument(
        "-m",
        "--methylation-file",
        dest="methylation_file",
        required=True,
        type=Path,
        help="The filepath to the methylation sequence data.",
    )
    parser.add_argument(
        "--max-iterations",
        dest="max_iterations",
        required=False,
        type=int,
        default=100,
        help="The number of iterations for which to batch train.",
    )
    parser.add_argument(
        "-l",
        "--layer_quantity",
        dest="layer_quantity",
        required=False,
        type=int,
        default=1,
        help="The number of parametrized layers in the ansatz.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        dest="output_directory",
        required=True,
        type=Path,
        help="The path at which to save the trained parameters.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="methylation_threshold",
        required=False,
        type=float,
        default=0.5,
        help="The threshold at which to consider a site positively methylated.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        required=False,
        type=int,
        default=1,
        help="The seed for the sample generator.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    sequences, methylation = load_methylation_samples(args.methylation_file)
    training_sequences, training_methylation, test_sequences, test_methylation = train_test_split(sequences, methylation, random_state=args.seed)
    address_register_size: int = qcpg.calculate_address_register_size(
        len(training_sequences[0])
    )
    params_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=args.layer_quantity,
        n_wires=address_register_size + UNIQUE_NUCLEOTIDE_QUANTITY,
    )
    parameters: NDArray = pnp.random.default_rng().random(size=params_shape)
    trained_parameters, loss_history = (
        qcpg.train_strongly_entangled_qcpg_circuit(
            parameters, training_sequences, training_methylation, args.max_iterations
        )
    )
    np.save(args.output_directory / "model.npy", trained_parameters)
    loss_file: Path = args.output_directory / "loss_history.txt"
    with loss_file.open("w") as fd:
        for loss in loss_history:
            fd.write(str(loss))
            fd.write("\n")
