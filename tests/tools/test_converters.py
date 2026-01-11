from pathlib import Path
from tempfile import TemporaryDirectory

from h5py import File
import numpy as np
from numpy.typing import NDArray

from qenetics.tools import converters


def test_determine_sequence_length(
    test_deepcpg_dataset_directory: Path,
) -> None:
    assert (
        converters._determine_sequence_length(test_deepcpg_dataset_directory)
        == 10
    )


def test_extract_deepcpg_experiment_to_qcpg(
    test_deepcpg_dataset_directory: Path,
    test_single_experiment_dataset_directory: Path,
) -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        converters.extract_deepcpg_experiment_to_qcpg(
            test_deepcpg_dataset_directory, temp_path, "experiment0"
        )
        assert len(list(temp_path.iterdir())) == len(
            list(test_deepcpg_dataset_directory.iterdir())
        )

        with File(
            test_single_experiment_dataset_directory / "chr1.h5"
        ) as expected_dataset:
            expected_dataset_sequences: NDArray = np.array(
                expected_dataset["methylation_sequences"]
            )
            expected_dataset_ratios: NDArray = np.array(
                expected_dataset["methylation_ratios"]
            )

        with File(temp_path / "chr1.h5") as actual_dataset:
            actual_dataset_sequenecs: NDArray = np.array(
                actual_dataset["methylation_sequences"]
            )
            actual_dataset_ratios: NDArray = np.array(
                actual_dataset["methylation_ratios"]
            )

        assert (expected_dataset_sequences == actual_dataset_sequenecs).all()
        assert (expected_dataset_ratios == actual_dataset_ratios).all()
