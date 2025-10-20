from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

from qenetics.qcpg import qcpg

UNIQUE_NUCLEOTIDE_QUANTITY: int = 4


logger = logging.getLogger(__name__)


def _parse_script_args() -> Namespace:
    parser = ArgumentParser(
        "train_qcpg_model",
        usage="python3 train_qcpg_model.py ",
        description="Train the parameters of a quantum circuit that predicts whether "
        "CpG sites are methylated given an input nucleotide sequence.",
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        dest="data_directory",
        required=True,
        type=Path,
        help="The filepath to H5 methylation files storing methylation data.",
    )
    parser.add_argument(
        "--training-chromosomes",
        dest="training_chromosomes",
        required=False,
        type=str,
        nargs="+",
        default=["1", "3", "5", "7", "9", "11"],
        help="Specify the chromosomes to use as training data.",
    )
    parser.add_argument(
        "--validation-chromosomes",
        dest="validation_chromosomes",
        required=False,
        type=str,
        nargs="+",
        default=["2", "4", "6", "8", "10", "12"],
        help="Specify the chromosomes to use as validation data.",
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
        "--output-filepath",
        dest="output_filepath",
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
        "-r",
        "--learning-rate",
        dest="learning_rate",
        required=False,
        type=float,
        default=0.0001,
        help="The learning rate for the optimizer.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        required=False,
        type=int,
        default=1,
        help="The seed for the sample generator.",
    )
    parser.add_argument(
        "--log-directory",
        dest="log_directory",
        required=False,
        type=Path,
        default=None,
        help="The filepath to the desired log directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    if args.log_directory:
        logging.basicConfig(
            filename=args.log_directory / "qcpg_train.log", level=logging.INFO
        )
    qcpg.train_qnn_circuit(
        qcpg.TrainingParameters(
            data_directory=args.data_directory,
            output_filepath=args.output_filepath,
            training_chromosomes=args.training_chromosomes,
            validation_chromosomes=args.validation_chromosomes,
            layer_quantity=args.layer_quantity,
            epochs=args.max_iterations,
            learning_rate=args.learning_rate,
        )
    )
