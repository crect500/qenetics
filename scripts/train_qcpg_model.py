import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from transformers import AutoTokenizer

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
        "--model-filepath",
        dest="model_filepath",
        required=False,
        type=Path,
        default=None,
        help="Load an existing model instead of creating a new model.",
    )
    parser.add_argument(
        "-e",
        "--entangler",
        dest="entangler",
        required=False,
        type=str,
        default="basic",
        help="'basic' for BasicEntanglerLayers, 'strong' for StronglyEntanglingLayers.",
    )
    parser.add_argument(
        "--encoding",
        dest="encoding",
        required=False,
        type=str,
        default="onehot",
        help="'onehot', 'token', or 'bpe'",
    )
    parser.add_argument(
        "--embedding-qubits",
        dest="embedding_qubit_quantity",
        required=False,
        type=int,
        help="The number of qubits to use for the embedding layer",
    )
    parser.add_argument(
        "--tokenizer",
        dest="tokenizer",
        required=False,
        type=str,
        default="PoetschLab/GROVER",
        help="The huggingface pretrained tokenizer name to use",
    )
    parser.add_argument(
        "--measurement",
        dest="measurement",
        required=False,
        type=str,
        default="probability",
        help="'probability' for state probabilities, 'expectation' for observables.",
    )
    parser.add_argument(
        "--diff-method",
        dest="diff_method",
        required=False,
        type=str,
        default="adjoint",
        help="The differentiation method for the Pennylane QNodes.",
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
        "-r",
        "--learning-rate",
        dest="learning_rate",
        required=False,
        type=float,
        default=0.0001,
        help="The learning rate for the optimizer.",
    )
    parser.add_argument(
        "-l1",
        dest="l1_regularization",
        required=False,
        type=float,
        default=0.0,
        help="The L1 (LASSO) regularization lambda value.",
    )
    parser.add_argument(
        "-l2",
        dest="l2_regularization",
        required=False,
        type=float,
        default=0.0,
        help="The L2 (Ridge) regularization lambda value.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        required=False,
        default=128,
        type=int,
        help="The training batch size.",
    )
    parser.add_argument(
        "--log-directory",
        dest="log_directory",
        required=False,
        type=Path,
        default=None,
        help="The filepath to the desired log directory",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        required=False,
        default="info",
        help="The log level for the current script.",
    )
    parser.add_argument(
        "--gpus",
        dest="gpu_quantity",
        type=int,
        required=False,
        default=0,
        help="The number of GPUs to use. If 0, CPUs will be used.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = _parse_script_args()
    if args.log_level == "debug":
        log_level: int = logging.DEBUG
    else:
        log_level = logging.INFO

    if args.gpu_quantity > 0:
        device_name: str = "lightning.gpu"
        if args.gpu_quantity > 1:
            distribute: bool = True
        else:
            distribute = False
    else:
        device_name = "lightning.qubit"
        distribute = False

    if args.tokenizer is not None:
        tokenizer: AutoTokenizer | None = AutoTokenizer.from_pretrained(
            args.tokenizer
        )
        vocabulary_size: int | None = tokenizer.vocab_size
    else:
        tokenizer = None
        vocabulary_size = None

    qcpg.train_qnn_circuit(
        qcpg.TrainingParameters(
            data_directory=args.data_directory,
            output_directory=args.output_directory,
            training_chromosomes=args.training_chromosomes,
            validation_chromosomes=args.validation_chromosomes,
            entangler=args.entangler,
            encoding=args.encoding,
            embedding_qubit_quantity=args.embedding_qubit_quantity,
            tokenizer=tokenizer,
            vocabulary_size=vocabulary_size,
            measurement=args.measurement,
            diff_method=args.diff_method,
            layer_quantity=args.layer_quantity,
            epochs=args.max_iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l1_regularizer=args.l1_regularization,
            l2_regularizer=args.l2_regularization,
            model_filepath=args.model_filepath,
            log_directory=args.log_directory,
            log_level=log_level,
            gpu_quantity=args.gpu_quantity,
            device_name=device_name,
            distributed=distribute,
        )
    )
