from __future__ import annotations

import logging
from collections.abc import Sequence
from io import TextIOBase
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from qenetics.tools import converters, dna, methylation

logger = logging.getLogger(__name__)

METHYLATION_SEQUENCES_KEY: str = "methylation_sequences"
METHYLATION_RATIOS_KEY: str = "methylation_ratios"
INPUTS_STR: str = "inputs"
DNA_STR: str = "dna"
METHYLATION_STR: str = "methylation_sequences"
TOKEN_ENCODING_STR: str = "token"
ONEHOT_ENCODING_STR: str = "one-hot"
BPE_ENCODING_STR: str = "bpe"
H5_STR: str = "h5"


class QuantumTorchDataset(Dataset):
    """
    Facilitates processing a dataset of nucleotide sequences.
    """

    def __init__(
        self: QuantumTorchDataset,
        filepaths: list[Path],
        threshold: float = 0.5,
        encoding: str = TOKEN_ENCODING_STR,
        *,
        tokenizer: TokenizersBackend | None = None,
        allow_N: bool = False,
    ) -> None:
        """
        Load a dataset for use in torch training pipelines.

        Args
        ----
        filepaths: The files to load. Currently only 'h5' file formats are supported.
        threshold: The threshold for the binary predictions.
        encoding: The encoding method. Currently supported encodings are 'token', 'one-hot', and 'BPE.
        tokenizer: The BPE tokenizer.
        allow_N: Whether to allow N in the original dataset encoding.
        """
        file_format: str = _check_file_types(filepaths)
        self.file_list = filepaths
        if encoding == BPE_ENCODING_STR and tokenizer is None:
            raise RuntimeError(
                "Must specify tokenizer if BPE encoding is desired."
            )

        input_encoding, samples_key = self._retrieve_data_parameters(
            filepaths, file_format
        )

        self._allocate_tensors(filepaths, file_format, samples_key, encoding)
        self._fill_tensors(
            filepaths,
            file_format,
            input_encoding,
            samples_key,
            encoding,
            threshold,
            tokenizer=tokenizer,
            allow_N=allow_N,
        )

    def __len__(self: QuantumTorchDataset) -> int:
        return len(self.data)

    def __getitem__(
        self: QuantumTorchDataset, idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]

    def _retrieve_data_parameters(
        self: QuantumTorchDataset, filepaths: Sequence[Path], file_format: str
    ) -> tuple[str, str]:
        """
        Retrieve the sequence length, experiment names, and experiment quantity from the files.

        Args
        ----
        filepaths: The files to retrieve the parameters from.
        file_format: The type of the files.

        Returns
        -------
        The encoding method of the input and the H5 samples key, if applicable.

        Raises
        ------
        RuntimeError if H5 file structure is not recognized.
        NotImplementedError if file_format processing is not supported.
        """
        if file_format == H5_STR:
            with h5py.File(filepaths[0]) as dataset:
                self.sequence_length = dataset[METHYLATION_SEQUENCES_KEY].shape[
                    1
                ]
                if isinstance(dataset[METHYLATION_RATIOS_KEY], h5py.Group):
                    self.experiment_names = dataset[
                        METHYLATION_RATIOS_KEY
                    ].keys()
                    self.experiment_quantity = len(self.experiment_names)
                else:
                    self.experiment_quantity = 1

                try:
                    samples_key: str = _find_h5_samples_key(dataset)
                except RuntimeError as e:
                    raise RuntimeError(f"{e} in file {filepaths[0]}")

            input_encoding: str = _determine_sample_encoding(
                filepaths, file_format, samples_key
            )
            return input_encoding, samples_key
        else:
            raise NotImplementedError(
                f"Processing of {file_format} files is not implemented"
            )

    def _allocate_tensors(
        self: QuantumTorchDataset,
        filepaths: list[Path],
        file_format: str,
        samples_key: str,
        encoding: str = TOKEN_ENCODING_STR,
    ) -> None:
        """
        Allocate empty tensors to hold the sample data.

        Args
        ----
        filepaths: The files to load into tensors.
        file_format: The type of the files.
        encoding: The encoding method. Currently supported encodings are 'token', 'one-hot', and 'BPE'.

        Raises
        ------
        ValueError if encoding method is not supported.
        """
        sample_quantity = _determine_sample_quantity(
            filepaths, file_format, samples_key
        )

        if encoding in [TOKEN_ENCODING_STR, BPE_ENCODING_STR]:
            self.data = torch.empty(
                sample_quantity,
                self.sequence_length,
                dtype=torch.float,
                requires_grad=False,
            )
        elif encoding == ONEHOT_ENCODING_STR:
            self.data = torch.empty(
                sample_quantity,
                self.sequence_length,
                converters.UNIQUE_NUCLEOTIDE_QUANTITY,
                dtype=torch.float,
                requires_grad=False,
            )

        else:
            raise ValueError(f"Encoding method {encoding} not recognized")

        if self.experiment_quantity > 1:
            self.labels = torch.empty(
                sample_quantity,
                self.experiment_quantity,
                dtype=torch.float,
                requires_grad=False,
            )
        else:
            self.labels = torch.empty(
                sample_quantity, dtype=torch.float, requires_grad=False
            )
        logger.debug(
            "Initialized dataset for %d samples of length %d with %s encoding",
            sample_quantity,
            self.sequence_length,
            encoding,
        )

    def _fill_tensors(
        self: QuantumTorchDataset,
        filepaths: list[Path],
        file_format: str,
        input_encoding: str,
        samples_key: str,
        encoding: str,
        threshold: float,
        *,
        tokenizer: TokenizersBackend | None,
        allow_N: bool = False,
    ) -> None:
        """
        Load the dataset into the pre-allocated tensors.

        Args
        ----
        filepaths: The files to load into tensors.
        file_format: The type of the files.
        input_encoding: The encoding of the files to be loaded into tensors.
        samples_key: The key that represents a certain H5 file structure.
        encoding: The encoding method. Currently supported encodings are 'token', 'one-hot', and 'BPE'.
        threshold: The threshold for the binary predictions.
        tokenizer: The tokenizer used for BPE encodings.
        allow_N: Whether to allow zero in the original dataset encoding.
        """
        current_index: int = 0
        for filepath in filepaths:
            logger.debug("Loading data from %s", str(filepath))

            if file_format == H5_STR:
                with h5py.File(filepath) as dataset:
                    file_sample_quantity: int = (
                        _determine_dataset_sample_quantity(dataset, samples_key)
                    )
                    if input_encoding == TOKEN_ENCODING_STR:
                        if encoding == TOKEN_ENCODING_STR:
                            if samples_key == INPUTS_STR:
                                self.data[
                                    current_index : current_index
                                    + file_sample_quantity,
                                    :,
                                ] = torch.tensor(
                                    dataset[INPUTS_STR][DNA_STR],
                                    dtype=torch.float,
                                    requires_grad=False,
                                )
                            elif samples_key == METHYLATION_STR:
                                self.data[
                                    current_index : current_index
                                    + file_sample_quantity,
                                    :,
                                ] = torch.tensor(
                                    dataset[METHYLATION_STR],
                                    dtype=torch.float,
                                    requires_grad=False,
                                )
                        elif encoding == ONEHOT_ENCODING_STR:
                            self.data[
                                current_index : current_index
                                + file_sample_quantity,
                                :,
                            ] = torch.tensor(
                                _convert_token_dataset_to_onehot(
                                    dataset, samples_key
                                )
                            )
                        elif encoding == "bpe":
                            if tokenizer is None:
                                raise ValueError(
                                    "Must provide a tokenizer to to convert to BPE"
                                )

                            self.data[
                                current_index : current_index
                                + file_sample_quantity,
                            ] = _convert_token_dataset_to_bpe(
                                dataset, tokenizer, samples_key
                            )
                    elif input_encoding == ONEHOT_ENCODING_STR:
                        if encoding == ONEHOT_ENCODING_STR:
                            if samples_key == INPUTS_STR:
                                self.data[
                                    current_index : current_index
                                    + file_sample_quantity,
                                    :,
                                    :,
                                ] = torch.tensor(
                                    dataset[INPUTS_STR][DNA_STR],
                                    dtype=torch.float,
                                    requires_grad=False,
                                )
                            elif samples_key == METHYLATION_STR:
                                self.data[
                                    current_index : current_index
                                    + file_sample_quantity,
                                    :,
                                    :,
                                ] = torch.tensor(
                                    np.array(
                                        dataset[METHYLATION_SEQUENCES_KEY],
                                        dtype=float,
                                    ),
                                    dtype=torch.float,
                                    requires_grad=False,
                                )
                        elif encoding == TOKEN_ENCODING_STR:
                            self.data[
                                current_index : current_index
                                + file_sample_quantity,
                                :,
                            ] = _convert_onehot_dataset_to_token(
                                dataset, samples_key, allow_N=allow_N
                            )
                        elif encoding == "bpe":
                            if tokenizer is None:
                                raise ValueError(
                                    "Must provide a tokenizer to to convert to BPE"
                                )

                            self.data[
                                current_index : current_index
                                + file_sample_quantity,
                            ] = _convert_onehot_dataset_to_bpe(
                                dataset, tokenizer, samples_key, allow_N=allow_N
                            )

                    if self.experiment_quantity > 1:
                        for label_index, experiment_name in enumerate(
                            dataset[METHYLATION_RATIOS_KEY].keys()
                        ):
                            self.labels[
                                current_index : current_index
                                + file_sample_quantity,
                                label_index,
                            ] = torch.tensor(
                                dataset[METHYLATION_RATIOS_KEY][
                                    experiment_name
                                ],
                                dtype=torch.float,
                                requires_grad=False,
                            )
                    else:
                        self.labels[
                            current_index : current_index + file_sample_quantity
                        ] = torch.tensor(
                            dataset[METHYLATION_RATIOS_KEY],
                            dtype=torch.float,
                            requires_grad=False,
                        )
            else:
                raise NotImplementedError(
                    f"Reading from file format {file_format} is not supported."
                )

            current_index += file_sample_quantity

        if 0.0 <= threshold <= 1.0:
            self.labels[self.labels < threshold] = 0.0
            self.labels[self.labels > 0.0] = 1.0
        elif threshold != -1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, inclusive, if desired. If not, threshold must be -1.0, not {threshold}"
            )


def _check_file_types(filepaths: Sequence[Path]) -> str:
    """
    Determine the file type of the filepaths provided.

    This function currently supports only 'h5' file formats.

    Args
    ----
    filepaths: A sequence of filepaths.

    Returns
    -------
    The file extension.

    Raises
    ------
    ValueError if not all filepaths have the same extension.
    NotImplemented error if the filepath has an unsupported extension.
    """
    if not filepaths:
        raise ValueError("No filepaths provided")

    if all(filepath.suffix == ".h5" for filepath in filepaths):
        return H5_STR

    first_filepath_suffix: str = filepaths[0].suffix
    for filepath in filepaths:
        if filepath.suffix != first_filepath_suffix:
            raise ValueError(
                f"Inhomogeneous file type found for filepath {filepath}"
            )

    raise NotImplementedError(
        f"Unsupported file format or mixed file formats in files {filepaths}"
    )


def _find_h5_samples_key(dataset: h5py.Dataset) -> str:
    """
    Finds the key that corresponds to a certain H5 file structure.

    Args
    ----
    filepath: The file to process.

    Returns
    -------
    The key that represents a certain H5 file structure.

    Raises
    ------
    RuntimeError if H5 file structure is not recognized.
    """
    if INPUTS_STR in dataset:
        return INPUTS_STR

    if METHYLATION_STR in dataset:
        return METHYLATION_STR

    raise RuntimeError("Sequence samples dataset not found")


def _determine_h5_dimensions_and_type(filepath: Path, samples_key: str) -> int:
    """
    Determine the number of dimensions in the samples stored in the file provided.

    Args
    ----
    filepath: The file to process.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The number of dimensions of the samples in the file.
    """
    with h5py.File(filepath) as dataset:
        if samples_key == INPUTS_STR:
            return (
                len(dataset[INPUTS_STR][DNA_STR].shape),
                dataset[INPUTS_STR][DNA_STR].dtype,
            )

        if samples_key == METHYLATION_STR:
            return (
                len(dataset[METHYLATION_STR].shape),
                dataset[METHYLATION_STR].dtype,
            )


def _check_np_int(dtype: np.typing.DTypeLike) -> bool:
    """
    Check whether the dtype is any of the numpy integer dtypes.

    Args
    ----
    dtype: The data type.

    Returns
    -------
    True if a numpy integer dtype. False otherwise.
    """
    return (
        dtype == np.int8
        or dtype == np.int16
        or dtype == np.int32
        or dtype == np.int64
    )


def _determine_h5_sample_encoding(
    filepaths: Sequence[Path], samples_key: str
) -> str:
    """
    Determine the encoding of the h5 files being read.

    Args
    ----
    filepaths: The files to process.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The encoding.

    Raises
    ------
    RuntimeError if encoding method cannot be determined.
    """
    all_dimensions_types: list[tuple[int, np.typing.DTypeLike]] = [
        _determine_h5_dimensions_and_type(filepath, samples_key)
        for filepath in filepaths
    ]
    if all(
        dimensions_types[0] == 2 for dimensions_types in all_dimensions_types
    ) and all(
        _check_np_int(dimensions_types[1])
        for dimensions_types in all_dimensions_types
    ):
        return TOKEN_ENCODING_STR

    if all(
        dimensions_types[0] == 3 for dimensions_types in all_dimensions_types
    ):
        return ONEHOT_ENCODING_STR

    raise RuntimeError(
        f"Samples of type {all_dimensions_types[0][1]} and {all_dimensions_types[0][0]} dimensions are not supported."
    )


def _determine_sample_encoding(
    filepaths: Sequence[Path], file_format: str, samples_key: str
) -> str:
    """
    Determine the encoding of the files being read.

    Args
    ----
    filepaths: The files to process.
    file_format: The type of files provided.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The encoding.
    """
    if file_format == H5_STR:
        return _determine_h5_sample_encoding(filepaths, samples_key)

    raise NotImplementedError(
        f"File parsing not implemented for file type {file_format}"
    )


def _determine_dataset_sample_quantity(
    dataset: h5py.Dataset, samples_key: str
) -> int:
    """
    Determine the quantity of samples in the dataset.

    Args
    ----
    dataset: The dataset to process.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The quantity of samples in the dataset provided.

    Raises
    ------
    RuntimeError if H5 file structure is not recognized.

    """
    if samples_key == INPUTS_STR:
        return dataset[INPUTS_STR][DNA_STR].shape[0]

    if samples_key == METHYLATION_STR:
        return dataset[METHYLATION_STR].shape[0]

    raise ValueError(f"Samples key {samples_key} not recognized.")


def _determine_sample_quantity(
    filepaths: Sequence[Path], file_format: str, samples_key: str
):
    """
    Determine the total sample quantity in the provided filepaths.

    Args
    ----
    filepaths: The files to count samples from.
    file_format: The file type of the files.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The total sample quantity in all provided files.

    Raises
    ------
    ValueError if parsing for the file_format provided is not supported.
    """
    sample_quantity: int = 0
    if file_format == H5_STR:
        for filepath in filepaths:
            with h5py.File(filepath) as dataset:
                file_sample_quantity: int = _determine_dataset_sample_quantity(
                    dataset, samples_key
                )
                logger.debug(
                    "Found %d samples in file %s",
                    file_sample_quantity,
                    str(filepath),
                )
                sample_quantity += file_sample_quantity

        return sample_quantity

    raise NotImplementedError(f"File format {file_format} not supported")


def _convert_token_dataset_to_onehot(
    dataset: h5py.Dataset, samples_key: str
) -> torch.Tensor:
    """
    Convert a token-encoded H5 dataset to a one-hot encoded tensor.

    Args
    ----
    dataset: The token-encoded H5 dataset to convert.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The one-hot encoded tensor.

    Raises
    ------
    ValueError if samples_key does not correlate to a known H5 file structure.
    """
    if samples_key == INPUTS_STR:
        return torch.tensor(
            np.array(
                [
                    converters.nucleotide_array_to_numpy(sequence)
                    for sequence in dataset[INPUTS_STR][DNA_STR]
                ],
                dtype=int,
            ),
            dtype=torch.int,
            requires_grad=False,
        )

    if samples_key == METHYLATION_STR:
        return torch.tensor(
            [
                converters.nucleotide_array_to_numpy(sequence)
                for sequence in dataset[METHYLATION_STR]
            ],
            dtype=torch.int,
            requires_grad=False,
        )

    raise ValueError(
        f"Conversion to one-hot encoding is not supported for H5 structure {samples_key}"
    )


def _convert_onehot_dataset_to_token(
    dataset: h5py.Dataset, samples_key: str, *, allow_N: bool = False
) -> torch.Tensor:
    """
    Convert a one-hot encoded H5 dataset to a token-encoded tensor.

    Args
    ----
    dataset: The one-hot encoded H5 dataset to convert.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The token-encoded tensor.

    Raises
    ------
    ValueError if samples_key does not correlate to a known H5 file structure.
    """
    if samples_key == INPUTS_STR:
        return torch.tensor(
            np.array(
                [
                    converters.one_hot_sequence_to_integers(
                        sequence, allow_N=allow_N
                    )
                    for sequence in dataset[INPUTS_STR][DNA_STR]
                ],
                dtype=float,
            ),
            dtype=torch.float,
            requires_grad=False,
        )

    if samples_key == METHYLATION_STR:
        return torch.tensor(
            np.array(
                [
                    converters.one_hot_sequence_to_integers(
                        sequence, allow_N=allow_N
                    )
                    for sequence in dataset[METHYLATION_STR]
                ],
                dtype=float,
            ),
            dtype=torch.float,
            requires_grad=False,
        )

    raise ValueError(
        f"Conversion to token encoding is not supported for H5 structure {samples_key}"
    )


def _convert_token_dataset_to_bpe(
    dataset: h5py.Dataset, tokenizer: TokenizersBackend, samples_key: str
) -> torch.Tensor:
    """
    Convert a simple token H5 dataset to a Byte-Pair Encoding (BPE) tensor.

    Args
    ----
    dataset: The simple token encoded H5 dataset to convert.
    tokenizer: The tokenzier to convert nucleotide sequences to tokens.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The BPE tensor.

    Raises
    ------
    ValueError if samples_key does not correlate to a known H5 file structure.
    """
    if samples_key == INPUTS_STR:
        sample_quantity, sequence_length = dataset[INPUTS_STR][DNA_STR].shape
        encodings: torch.Tensor = torch.empty(
            (sample_quantity, sequence_length),
            dtype=torch.int,
            requires_grad=False,
        )
        for index, sequence in enumerate(dataset[INPUTS_STR][DNA_STR]):
            encoding = torch.tensor(
                tokenizer(converters.integer_array_to_nucleotide_str(sequence))[
                    "input_ids"
                ],
                dtype=torch.int,
                requires_grad=False,
            )
            half_padding: float = (sequence_length - len(encoding)) / 2
            if half_padding.is_integer():
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding))
                )
            else:
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding) + 1)
                )
        return encodings

    if samples_key == METHYLATION_STR:
        sample_quantity, sequence_length = dataset[METHYLATION_STR].shape
        encodings: torch.Tensor = torch.empty(
            (sample_quantity, sequence_length),
            dtype=torch.int,
            requires_grad=False,
        )
        for index, sequence in enumerate(dataset[METHYLATION_STR]):
            encoding = torch.tensor(
                tokenizer(converters.integer_array_to_nucleotide_str(sequence))[
                    "input_ids"
                ],
                dtype=torch.int,
                requires_grad=False,
            )
            half_padding: float = (sequence_length - len(encoding)) / 2
            if half_padding.is_integer():
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding))
                )
            else:
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding) + 1)
                )
        return encodings

    raise ValueError(
        f"BPE token conversion is not supported for H5 structure {samples_key}"
    )


def _convert_onehot_dataset_to_bpe(
    dataset: h5py.Dataset,
    tokenizer: TokenizersBackend,
    samples_key: str,
    *,
    allow_N: bool = False,
) -> torch.Tensor:
    """
    Convert a one-hot encoded H5 dataset to a Byte Pair Encoding (BPE) tensor.

    Args
    ----
    dataset: The one-hot encoded H5 dataset to convert.
    samples_key: The key that represents a certain H5 file structure.

    Returns
    -------
    The BPE tensor.

    Raises
    ------
    ValueError if samples_key does not correlate to a known H5 file structure.
    """
    if samples_key == INPUTS_STR:
        sample_quantity, sequence_length, _ = dataset[INPUTS_STR][DNA_STR].shape
        encodings: torch.Tensor = torch.empty(
            (sample_quantity, sequence_length),
            dtype=torch.int,
            requires_grad=False,
        )
        for index, sequence in enumerate(dataset[INPUTS_STR][DNA_STR]):
            encoding = torch.tensor(
                tokenizer(
                    converters.one_hot_sequence_to_nucleotide_str(
                        sequence, allow_N=allow_N
                    )
                )["input_ids"],
                dtype=torch.int,
                requires_grad=False,
            )
            half_padding: float = (sequence_length - len(encoding)) / 2
            if half_padding.is_integer():
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding))
                )
            else:
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding) + 1)
                )
        return encodings

    if samples_key == METHYLATION_STR:
        sample_quantity, sequence_length, _ = dataset[METHYLATION_STR].shape
        encodings: torch.Tensor = torch.empty(
            (sample_quantity, sequence_length),
            dtype=torch.int,
            requires_grad=False,
        )
        for index, sequence in enumerate(dataset[METHYLATION_STR]):
            encoding = torch.tensor(
                tokenizer(
                    converters.one_hot_sequence_to_nucleotide_str(
                        sequence, allow_N=allow_N
                    )
                )["input_ids"],
                dtype=torch.int,
                requires_grad=False,
            )
            half_padding: float = (sequence_length - len(encoding)) / 2
            if half_padding.is_integer():
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding))
                )
            else:
                encodings[index] = torch.nn.functional.pad(
                    encoding, (int(half_padding), int(half_padding) + 1)
                )
        return encodings

    raise ValueError(
        f"BPE token conversion is not supported for H5 structure {samples_key}"
    )


def _increment_or_create_entry(dictionary: dict[Any, ...], key: Any) -> None:
    entry: Any | None = dictionary[key]
    if entry is None:
        dictionary[key] = 1
    else:
        dictionary[key] += 1


def _validate_sequence(sequence: str) -> bool:
    middle_index: int = int(len(sequence) / 2) - 1
    return not (
        sequence[middle_index : middle_index + 2] != "CG" or "N" in sequence
    )


def _retrieve_chromosome_sequences(
    profiles_by_position: dict[int, dict[str, float]],
    chromosome: str,
    fasta_file_descriptor: TextIOBase,
    fasta_metadata,
    fasta_line_length: int,
    sequence_length: int,
    experiment_names: list[str],
) -> tuple[NDArray[bool], NDArray[float]]:
    sequences = np.ndarray(
        (0, sequence_length, converters.UNIQUE_NUCLEOTIDE_QUANTITY), dtype=int
    )
    experiment_mapping: dict[str, int] = {
        experiment_name: index
        for index, experiment_name in enumerate(experiment_names)
    }
    methylation_ratios = np.ndarray((0, len(experiment_names)))
    for position, profile_by_experiment in profiles_by_position.items():
        sequence: str | None = dna.find_methylation_sequence(
            chromosome=chromosome,
            position=position,
            genome_metadata=fasta_metadata,
            file_descriptor=fasta_file_descriptor,
            sequence_length=sequence_length,
            line_length=fasta_line_length,
        )
        if sequence is None or not _validate_sequence(sequence):
            continue
        else:
            sequences = np.append(
                sequences,
                converters.nucleotide_string_to_numpy(sequence).reshape(
                    1, sequence_length, converters.UNIQUE_NUCLEOTIDE_QUANTITY
                ),
                axis=0,
            )

        ratios_by_experiment: np.ndarray = np.array(
            [np.nan] * len(experiment_names)
        ).reshape(1, len(experiment_names))
        for experiment_name, methylation_ratio in profile_by_experiment.items():
            ratios_by_experiment[0][experiment_mapping[experiment_name]] = (
                methylation_ratio
            )

        methylation_ratios = np.append(
            methylation_ratios, ratios_by_experiment, axis=0
        )

    return sequences, methylation_ratios


def _create_h5_dataset(
    h5_filepath: Path,
    sequences: NDArray[bool],
    methylation_ratios: NDArray[float],
    experiment_names: list[str],
) -> None:
    with h5py.File(h5_filepath, "w") as fd:
        (
            fd.create_dataset(
                METHYLATION_SEQUENCES_KEY,
                data=sequences,
                dtype="i1",
                chunks=True,
            ),
        )
        ratios_dataset: h5py.Group = fd.create_group(METHYLATION_RATIOS_KEY)
        for experiment_index, experiment_name in enumerate(experiment_names):
            ratios_dataset.create_dataset(
                experiment_name,
                shape=(sequences.shape[0],),
                chunks=True,
                dtype="f4",
            )
            for sample_index, experiment_ratios in enumerate(
                methylation_ratios
            ):
                ratios_dataset[experiment_name][sample_index] = (
                    experiment_ratios[experiment_index]
                )


def _create_h5_files(
    profiles_by_chromosome: dict[str, dict[int, dict[str, float]]],
    experiment_names: list[str],
    dataset_directory: Path,
    fasta_filepath: Path,
    sequence_length: int,
) -> None:
    fasta_line_length: int = dna.determine_line_length(fasta_filepath)
    with open(fasta_filepath) as fasta_fd:
        fasta_metadata: dict[str, dna.SequenceInfo] = (
            dna.extract_fasta_metadata(fasta_filepath)
        )
        for chromosome, profiles_by_position in profiles_by_chromosome.items():
            h5_filepath: Path = dataset_directory / ("chr" + chromosome + ".h5")
            sequences, methylation_ratios = _retrieve_chromosome_sequences(
                profiles_by_position=profiles_by_position,
                chromosome=chromosome,
                fasta_file_descriptor=fasta_fd,
                fasta_metadata=fasta_metadata,
                fasta_line_length=fasta_line_length,
                sequence_length=sequence_length,
                experiment_names=experiment_names,
            )
            _create_h5_dataset(
                h5_filepath=h5_filepath,
                sequences=sequences,
                methylation_ratios=methylation_ratios,
                experiment_names=experiment_names,
            )


def create_h5_dataset_from_methylation_profiles(
    methylation_directory: Path,
    fasta_filepath: Path,
    dataset_directory: Path,
    sequence_length: int,
    minimum_samples: int = 1,
) -> None:
    experiment_names: list[str] = [
        filename.stem.split(".")[0]
        for filename in methylation_directory.iterdir()
    ]
    profiles_by_chromosome: dict[str, dict[int, dict[str, float]]] = (
        methylation.record_methylation_profiles(
            methylation_directory, minimum_samples
        )
    )
    _create_h5_files(
        profiles_by_chromosome=profiles_by_chromosome,
        experiment_names=experiment_names,
        dataset_directory=dataset_directory,
        fasta_filepath=fasta_filepath,
        sequence_length=sequence_length,
    )


def _find_bounds(original_length: int, new_length: int) -> tuple[int, int]:
    if new_length >= original_length:
        raise ValueError(
            f"Requested length {new_length} is the same or greater than "
            f"the original length {original_length}"
        )

    is_original_even: bool = original_length % 2 == 0
    is_new_even: bool = new_length % 2 == 0
    if is_original_even != is_new_even:
        raise ValueError(
            f"Original length of {original_length} divisibility by 2 must "
            f"be the same as new length of {new_length} divisibility"
        )

    half_length: int = new_length // 2
    middle: int = original_length // 2
    start: int = middle - half_length
    if is_original_even:
        end: int = middle + half_length
    else:
        end = middle + half_length + 1

    return start, end


def _reduce_sample_size(
    data_filepath: Path, output_directory: Path, new_length: int
) -> None:
    with h5py.File(data_filepath) as data:
        original_sequence_length: int = data[METHYLATION_SEQUENCES_KEY].shape[1]
        start, end = _find_bounds(original_sequence_length, new_length)
        output_filepath: Path = output_directory / data_filepath.name

        with h5py.File(output_filepath, "w") as fd:
            fd.create_dataset(
                METHYLATION_SEQUENCES_KEY,
                data=data[METHYLATION_SEQUENCES_KEY][:, start:end, :],
            )
            fd.create_dataset(
                METHYLATION_RATIOS_KEY, data=data[METHYLATION_RATIOS_KEY]
            )


def reduce_dataset_sequence_length(
    original_directory: Path, output_directory: Path, new_length: int
) -> None:
    for filepath in original_directory.iterdir():
        _reduce_sample_size(filepath, output_directory, new_length)
