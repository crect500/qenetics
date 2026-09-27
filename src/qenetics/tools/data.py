from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
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
ONEHOT_ENCODING_STR: str = "onehot"
BPE_ENCODING_STR: str = "bpe"
H5_STR: str = "h5"
POSITIONS_KEY: str = "positions"
SEQUENCE_BATCH_SIZE: int = 4096
H5_CHUNK_SAMPLES: int = 64
# DeepCpG's label for an unobserved methylation state, read as NaN.
MISSING_LABEL: float = -1.0
# Keys of the data files written by DeepCpG, which store the sequence window
# of each site under `INPUTS_STR`/`DNA_STR`.
DEEPCPG_CHROMOSOME_KEY: str = "chromo"
DEEPCPG_POSITION_KEY: str = "pos"
DEEPCPG_OUTPUTS_KEY: str = "outputs"
# DeepCpG's integer code of each nucleotide, as in deepcpg.data.dna.CHAR_TO_INT.
DEEPCPG_NUCLEOTIDE_CODES: dict[str, int] = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 4,
}

# One-hot encodings indexed by ASCII code. Unknown nucleotides encode as zeros.
_ONE_HOT_NUCLEOTIDES: NDArray[np.int8] = np.zeros(
    (256, converters.UNIQUE_NUCLEOTIDE_QUANTITY), dtype=np.int8
)
_KNOWN_NUCLEOTIDES: NDArray[np.bool_] = np.zeros(256, dtype=bool)
# The same, indexed by DeepCpG nucleotide code viewed as unsigned.
_DEEPCPG_ONE_HOT_NUCLEOTIDES: NDArray[np.int8] = np.zeros_like(
    _ONE_HOT_NUCLEOTIDES
)
_DEEPCPG_KNOWN_NUCLEOTIDES: NDArray[np.bool_] = np.zeros_like(
    _KNOWN_NUCLEOTIDES
)
for _nucleotide in "ATCG":
    _one_hot: NDArray[np.int8] = converters.nucleotide_character_to_numpy(
        _nucleotide
    )
    _ONE_HOT_NUCLEOTIDES[ord(_nucleotide)] = _one_hot
    _KNOWN_NUCLEOTIDES[ord(_nucleotide)] = True
    _DEEPCPG_ONE_HOT_NUCLEOTIDES[DEEPCPG_NUCLEOTIDE_CODES[_nucleotide]] = (
        _one_hot
    )
    _DEEPCPG_KNOWN_NUCLEOTIDES[DEEPCPG_NUCLEOTIDE_CODES[_nucleotide]] = True


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
        encoding: The encoding method. Currently supported encodings are 'token', 'onehot', and 'BPE.
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
                self.experiment_names: list[str] | None = (
                    _read_h5_experiment_names(dataset)
                )
                if self.experiment_names is None:
                    self.experiment_quantity = 1
                else:
                    self.experiment_quantity = len(self.experiment_names)

                try:
                    samples_key: str = _find_h5_samples_key(dataset)
                except RuntimeError as e:
                    raise RuntimeError(f"{e} in file {filepaths[0]}")

            for filepath in filepaths[1:]:
                with h5py.File(filepath) as dataset:
                    check_experiment_names(
                        self.experiment_names,
                        _read_h5_experiment_names(dataset),
                        str(filepaths[0]),
                        str(filepath),
                    )

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
        encoding: The encoding method. Currently supported encodings are 'token', 'onehot', and 'BPE'.

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

        # The 1-based position of each sample's CpG site, -1 if not stored,
        # and the index into `chromosome_names` of each sample's chromosome.
        self.positions: NDArray[np.int64] = np.full(
            sample_quantity, -1, dtype=np.int64
        )
        self.chromosome_names: list[str] = [
            chromosome_name(filepath) for filepath in filepaths
        ]
        self.chromosome_indices: NDArray[np.int32] = np.empty(
            sample_quantity, dtype=np.int32
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
        encoding: The encoding method. Currently supported encodings are 'token', 'onehot', and 'BPE'.
        threshold: The threshold for the binary predictions.
        tokenizer: The tokenizer used for BPE encodings.
        allow_N: Whether to allow zero in the original dataset encoding.
        """
        current_index: int = 0
        for file_index, filepath in enumerate(filepaths):
            logger.debug("Loading data from %s", str(filepath))

            if file_format == H5_STR:
                with h5py.File(filepath) as dataset:
                    file_sample_quantity: int = (
                        _determine_dataset_sample_quantity(dataset, samples_key)
                    )
                    file_samples = slice(
                        current_index, current_index + file_sample_quantity
                    )
                    self.chromosome_indices[file_samples] = file_index
                    if POSITIONS_KEY in dataset:
                        self.positions[file_samples] = dataset[POSITIONS_KEY][
                            ()
                        ]
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

                    file_labels: torch.Tensor = torch.tensor(
                        _read_h5_labels(dataset),
                        dtype=torch.float,
                        requires_grad=False,
                    )
                    if self.experiment_quantity > 1:
                        self.labels[
                            current_index : current_index + file_sample_quantity
                        ] = file_labels
                    else:
                        self.labels[
                            current_index : current_index + file_sample_quantity
                        ] = file_labels[:, 0]
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


def chromosome_name(filepath: Path) -> str:
    """
    Find the chromosome a dataset file holds from its name.

    Args
    ----
    filepath: The dataset file, e.g. 'chr1.h5'.

    Returns
    -------
    The chromosome name, e.g. '1', or the file's stem if it does not follow
    the 'chr<name>' pattern.
    """
    return filepath.stem.removeprefix("chr")


def _read_h5_experiment_names(dataset: h5py.File) -> list[str] | None:
    """
    Read the names of the experiments whose labels an H5 file holds.

    Args
    ----
    dataset: The open H5 file.

    Returns
    -------
    The experiment names in the order of the label columns, or None if the
    labels are stored as a single unnamed dataset.
    """
    ratios: h5py.Group | h5py.Dataset = dataset[METHYLATION_RATIOS_KEY]
    if isinstance(ratios, h5py.Group):
        return list(ratios)

    return None


def check_experiment_names(
    expected_names: list[str] | None,
    names: list[str] | None,
    expected_source: str,
    source: str,
) -> None:
    """
    Verify that two sources hold labels of the same experiments in order.

    Args
    ----
    expected_names: The experiment names of the reference source, or None
        for labels stored as a single unnamed dataset.
    names: The experiment names of the source to check, or None for labels
        stored as a single unnamed dataset.
    expected_source: A description of the reference source.
    source: A description of the source to check.

    Raises
    ------
    ValueError if the experiments or their order differ.
    """
    if names == expected_names:
        return

    if names is None or expected_names is None:
        difference: str = "one stores labels as a single unnamed dataset"
    elif set(names) == set(expected_names):
        difference = "the experiments are in a different order"
    else:
        missing: list[str] = [
            name for name in expected_names if name not in names
        ]
        extra: list[str] = [
            name for name in names if name not in expected_names
        ]
        difference = f"missing {missing}, unexpected {extra}"

    raise ValueError(
        f"Experiments of {source} do not match those of {expected_source}: "
        f"{difference}. Expected {expected_names}, found {names}."
    )


def _read_h5_labels(dataset: h5py.File) -> NDArray[np.float32]:
    """
    Read the methylation labels of every experiment in an H5 file.

    Labels are stored either as a group holding one dataset per experiment,
    even if there is only one experiment, or as a single dataset. Unobserved
    labels marked with `MISSING_LABEL` are converted to NaN.

    Args
    ----
    dataset: The open H5 file.

    Returns
    -------
    The labels, with one column per experiment in the order of the group's
    keys, and NaN for unobserved labels.
    """
    ratios: h5py.Group | h5py.Dataset = dataset[METHYLATION_RATIOS_KEY]
    if isinstance(ratios, h5py.Group):
        labels: NDArray[np.float32] = np.stack(
            [ratios[experiment_name][()] for experiment_name in ratios], axis=1
        ).astype(np.float32)
    else:
        labels = np.asarray(ratios[()], dtype=np.float32).reshape(-1, 1)

    labels[labels == MISSING_LABEL] = np.nan
    return labels


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


def _find_methylation_filepaths(
    methylation_directory: Path, excluded_experiments: Collection[str] = ()
) -> dict[str, Path]:
    """
    Find the methylation profile file of each experiment in a directory.

    Args
    ----
    methylation_directory: The directory of methylation profile files, one
        file per experiment (cell).
    excluded_experiments: The names of experiments to leave out, e.g. cells
        RSC27_4, RSC27_7 and Ca26.

    Returns
    -------
    The filepath of each experiment, indexed by experiment name and sorted by
    name.

    Raises
    ------
    ValueError if two files share an experiment name.
    """
    filepaths_by_experiment: dict[str, Path] = {}
    for filepath in sorted(methylation_directory.iterdir()):
        experiment_name: str = filepath.name.split(".")[0]
        if experiment_name in filepaths_by_experiment:
            raise ValueError(
                f"Files {filepaths_by_experiment[experiment_name]} and "
                f"{filepath} share the experiment name {experiment_name}"
            )
        filepaths_by_experiment[experiment_name] = filepath

    for experiment_name in excluded_experiments:
        if filepaths_by_experiment.pop(experiment_name, None) is None:
            logger.warning(
                "Excluded experiment %s not found in %s",
                experiment_name,
                methylation_directory,
            )

    return filepaths_by_experiment


def read_methylation_counts(
    methylation_filepaths: Sequence[Path],
) -> dict[str, NDArray[np.int64]]:
    """
    Read the methylation read counts of every experiment.

    Args
    ----
    methylation_filepaths: The methylation profile file of each experiment.

    Returns
    -------
    An array for each chromosome in which each row holds the 1-based position,
    methylated read count, unmethylated read count and experiment index of one
    methylation call.
    """
    counts_by_chromosome: dict[str, list[NDArray[np.int64]]] = {}
    for experiment_index, filepath in enumerate(methylation_filepaths):
        logger.info("Reading methylation profiles from %s", filepath)
        rows_by_chromosome: dict[str, list[tuple[int, int, int]]] = {}
        for methylation_profile in methylation.retrieve_methylation_data(
            filepath
        ):
            rows_by_chromosome.setdefault(
                methylation_profile.chromosome, []
            ).append(
                (
                    methylation_profile.position,
                    methylation_profile.count_methylated,
                    methylation_profile.count_unmethylated,
                )
            )

        for chromosome, rows in rows_by_chromosome.items():
            counts: NDArray[np.int64] = np.empty((len(rows), 4), dtype=np.int64)
            counts[:, :3] = rows
            counts[:, 3] = experiment_index
            counts_by_chromosome.setdefault(chromosome, []).append(counts)

    return {
        chromosome: np.concatenate(counts)
        for chromosome, counts in counts_by_chromosome.items()
    }


def locate_cpg_sites(
    reference: NDArray[np.uint8], positions: NDArray[np.int64]
) -> NDArray[np.int64]:
    """
    Find the CpG site each methylation call belongs to.

    Calls on the forward strand are positioned at the C of the CpG, while calls
    on the reverse strand are positioned at the G. Both are assigned to the
    CpG's forward-strand C.

    Args
    ----
    reference: The ASCII nucleotides of the chromosome.
    positions: The 1-based positions of the methylation calls.

    Returns
    -------
    The 0-based index of the C of each call's CpG in the reference, or -1 if
    the call is not at a CpG in the reference.
    """
    cytosine, guanine = ord("C"), ord("G")
    indices: NDArray[np.int64] = positions.astype(np.int64) - 1
    last_index: int = len(reference) - 1
    in_bounds: NDArray[np.bool_] = (indices >= 0) & (indices <= last_index)
    nucleotides = reference[np.clip(indices, 0, last_index)]
    next_nucleotides = reference[np.clip(indices + 1, 0, last_index)]
    previous_nucleotides = reference[np.clip(indices - 1, 0, last_index)]

    is_cytosine: NDArray[np.bool_] = (
        in_bounds
        & (indices < last_index)
        & (nucleotides == cytosine)
        & (next_nucleotides == guanine)
    )
    is_guanine: NDArray[np.bool_] = (
        in_bounds
        & (indices > 0)
        & (nucleotides == guanine)
        & (previous_nucleotides == cytosine)
    )

    return np.where(is_cytosine, indices, np.where(is_guanine, indices - 1, -1))


def aggregate_cpg_counts(
    cpg_indices: NDArray[np.int64],
    counts: NDArray[np.int64],
    experiment_quantity: int,
    minimum_reads: int = 1,
    binarize: bool = True,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Combine the read counts of each CpG site and experiment into labels.

    Read counts of both strands of a CpG site are summed before filtering and
    labeling. Binary labels mark a site as methylated if it has more
    methylated than unmethylated reads, and unmethylated otherwise.

    Args
    ----
    cpg_indices: The 0-based index of the C of each call's CpG site.
    counts: The read counts of each call, as returned by
        `read_methylation_counts`.
    experiment_quantity: The quantity of experiments.
    minimum_reads: The minimum reads of a site in an experiment to be labeled.
    binarize: Label sites with binary methylation states if True. Otherwise,
        label sites with methylation ratios.

    Returns
    -------
    The sorted 0-based indices of the CpG sites and a label for each site and
    experiment. Sites not covered in an experiment are labeled NaN.
    """
    keys: NDArray[np.int64] = cpg_indices * experiment_quantity + counts[:, 3]
    unique_keys, key_indices = np.unique(keys, return_inverse=True)
    counts_methylated = np.bincount(
        key_indices, weights=counts[:, 1], minlength=len(unique_keys)
    )
    counts_unmethylated = np.bincount(
        key_indices, weights=counts[:, 2], minlength=len(unique_keys)
    )
    counts_total = counts_methylated + counts_unmethylated

    is_covered: NDArray[np.bool_] = counts_total >= minimum_reads
    unique_keys = unique_keys[is_covered]
    counts_methylated = counts_methylated[is_covered]
    counts_unmethylated = counts_unmethylated[is_covered]
    counts_total = counts_total[is_covered]

    if binarize:
        values = (counts_methylated > counts_unmethylated).astype(np.float32)
    else:
        values = (counts_methylated / counts_total).astype(np.float32)

    site_indices, experiment_indices = np.divmod(
        unique_keys, experiment_quantity
    )
    sites, label_rows = np.unique(site_indices, return_inverse=True)
    labels: NDArray[np.float32] = np.full(
        (len(sites), experiment_quantity), np.nan, dtype=np.float32
    )
    labels[label_rows, experiment_indices] = values

    return sites, labels


def window_starts(
    sites: NDArray[np.int64], sequence_length: int
) -> NDArray[np.int64]:
    """
    Find the start of the sequence window centered on each CpG site.

    For odd sequence lengths, the C of the CpG is the center nucleotide.
    For even sequence lengths, the CpG dinucleotide is the center pair.

    Args
    ----
    sites: The 0-based indices of the C of each CpG site.
    sequence_length: The length of the windows.

    Returns
    -------
    The 0-based index of the first nucleotide of each window.
    """
    return sites - (sequence_length - 1) // 2


def windows_in_bounds(
    sites: NDArray[np.int64], sequence_length: int, reference_length: int
) -> NDArray[np.bool_]:
    """
    Check whether the sequence window of each CpG site fits in the reference.

    Args
    ----
    sites: The 0-based indices of the C of each CpG site.
    sequence_length: The length of the windows.
    reference_length: The length of the chromosome.

    Returns
    -------
    True for each site whose window lies within the chromosome.
    """
    starts: NDArray[np.int64] = window_starts(sites, sequence_length)
    return (starts >= 0) & (starts + sequence_length <= reference_length)


def windows_are_known(
    reference: NDArray[np.uint8],
    sites: NDArray[np.int64],
    sequence_length: int,
) -> NDArray[np.bool_]:
    """
    Check whether the sequence window of each CpG site holds only A, T, C, G.

    Args
    ----
    reference: The ASCII nucleotides of the chromosome.
    sites: The 0-based indices of the C of each CpG site, whose windows must
        lie within the chromosome.
    sequence_length: The length of the windows.

    Returns
    -------
    True for each site whose window holds no unknown nucleotides.
    """
    starts: NDArray[np.int64] = window_starts(sites, sequence_length)
    window_offsets: NDArray[np.int64] = np.arange(sequence_length)
    are_known: NDArray[np.bool_] = np.empty(len(sites), dtype=bool)
    for batch_start in range(0, len(sites), SEQUENCE_BATCH_SIZE):
        batch = slice(batch_start, batch_start + SEQUENCE_BATCH_SIZE)
        windows: NDArray[np.uint8] = reference[
            starts[batch, np.newaxis] + window_offsets
        ]
        are_known[batch] = _KNOWN_NUCLEOTIDES[windows].all(axis=1)

    return are_known


def read_reference(
    fasta_filepath: Path, sequence_info: dna.SequenceInfo, line_length: int
) -> NDArray[np.uint8]:
    """
    Read a chromosome of the reference genome as an array of ASCII codes.

    Args
    ----
    fasta_filepath: The FASTA file of the reference genome.
    sequence_info: The metadata of the chromosome.
    line_length: The length of a line of nucleotide data in the FASTA file.

    Returns
    -------
    The upper-cased ASCII nucleotides of the chromosome.
    """
    return np.frombuffer(
        dna.read_chromosome(fasta_filepath, sequence_info, line_length),
        dtype=np.uint8,
    )


def _create_resizable_dataset(
    group: h5py.Group, name: str, sample_shape: tuple[int, ...], dtype: str
) -> h5py.Dataset:
    """
    Create an empty H5 dataset that grows along its first axis.

    Args
    ----
    group: The file or group in which to create the dataset.
    name: The name of the dataset.
    sample_shape: The shape of one sample.
    dtype: The data type of the dataset.

    Returns
    -------
    The created dataset.
    """
    return group.create_dataset(
        name,
        shape=(0, *sample_shape),
        maxshape=(None, *sample_shape),
        chunks=(H5_CHUNK_SAMPLES, *sample_shape),
        dtype=dtype,
        compression="gzip",
    )


def _append_to_dataset(dataset: h5py.Dataset, samples: NDArray) -> None:
    """
    Append samples to a resizable H5 dataset.

    Args
    ----
    dataset: The resizable dataset.
    samples: The samples to append.
    """
    if len(samples) == 0:
        return

    previous_length: int = dataset.shape[0]
    dataset.resize(previous_length + len(samples), axis=0)
    dataset[previous_length:] = samples


@dataclass
class _ChromosomeDatasets:
    """
    The datasets of a chromosome's H5 file, which grow as samples are
    appended.
    """

    sequences: h5py.Dataset
    positions: h5py.Dataset
    ratios: list[h5py.Dataset]

    @classmethod
    def create(
        cls,
        fd: h5py.File,
        sequence_length: int,
        experiment_names: Sequence[str],
    ) -> _ChromosomeDatasets:
        """
        Create the empty datasets of a chromosome's H5 file.

        Args
        ----
        fd: The H5 file, open for writing.
        sequence_length: The length of the sequence windows.
        experiment_names: The name of each experiment.

        Returns
        -------
        The datasets of the one-hot encoded sequence windows, the 1-based
        position of each site's C, and the labels of each experiment.
        """
        ratios_group: h5py.Group = fd.create_group(METHYLATION_RATIOS_KEY)
        return cls(
            sequences=_create_resizable_dataset(
                fd,
                METHYLATION_SEQUENCES_KEY,
                (sequence_length, converters.UNIQUE_NUCLEOTIDE_QUANTITY),
                "i1",
            ),
            positions=_create_resizable_dataset(fd, POSITIONS_KEY, (), "i8"),
            ratios=[
                _create_resizable_dataset(
                    ratios_group, experiment_name, (), "f4"
                )
                for experiment_name in experiment_names
            ],
        )

    def append(
        self,
        sequences: NDArray[np.int8],
        positions: NDArray[np.int64],
        labels: NDArray[np.float32],
    ) -> None:
        """
        Append samples to the datasets.

        Args
        ----
        sequences: The one-hot encoded sequence window of each sample.
        positions: The 1-based position of each sample's C.
        labels: The label of each sample in each experiment.
        """
        _append_to_dataset(self.sequences, sequences)
        _append_to_dataset(self.positions, positions)
        for experiment_index, ratios_dataset in enumerate(self.ratios):
            _append_to_dataset(ratios_dataset, labels[:, experiment_index])


def _write_chromosome_h5(
    h5_filepath: Path,
    reference: NDArray[np.uint8],
    sites: NDArray[np.int64],
    labels: NDArray[np.float32],
    experiment_names: Sequence[str],
    sequence_length: int,
    *,
    allow_N: bool = False,
) -> int:
    """
    Write the one-hot encoded sequence windows and labels of a chromosome.

    Args
    ----
    h5_filepath: The H5 file to write.
    reference: The ASCII nucleotides of the chromosome.
    sites: The sorted 0-based indices of the C of each CpG site.
    labels: The label of each site in each experiment.
    experiment_names: The name of each experiment.
    sequence_length: The length of the sequence windows.
    allow_N: Keep windows holding nucleotides other than A, T, C and G, which
        are encoded as all zeros, if True. Otherwise, drop them.

    Returns
    -------
    The quantity of samples written.
    """
    in_bounds: NDArray[np.bool_] = windows_in_bounds(
        sites, sequence_length, len(reference)
    )
    if not in_bounds.all():
        logger.info(
            "Dropping %d sites whose windows exceed the bounds of %s",
            np.count_nonzero(~in_bounds),
            h5_filepath.stem,
        )
    sites, labels = sites[in_bounds], labels[in_bounds]

    if not allow_N:
        are_known: NDArray[np.bool_] = windows_are_known(
            reference, sites, sequence_length
        )
        if not are_known.all():
            logger.info(
                "Dropping %d sites whose windows hold unknown nucleotides in "
                "%s",
                np.count_nonzero(~are_known),
                h5_filepath.stem,
            )
        sites, labels = sites[are_known], labels[are_known]

    starts: NDArray[np.int64] = window_starts(sites, sequence_length)
    window_offsets: NDArray[np.int64] = np.arange(sequence_length)
    with h5py.File(h5_filepath, "w") as fd:
        datasets = _ChromosomeDatasets.create(
            fd, sequence_length, experiment_names
        )
        for batch_start in range(0, len(sites), SEQUENCE_BATCH_SIZE):
            batch = slice(batch_start, batch_start + SEQUENCE_BATCH_SIZE)
            windows: NDArray[np.uint8] = reference[
                starts[batch, np.newaxis] + window_offsets
            ]
            datasets.append(
                _ONE_HOT_NUCLEOTIDES[windows], sites[batch] + 1, labels[batch]
            )

    return len(sites)


def create_h5_dataset_from_methylation_profiles(
    methylation_directory: Path,
    fasta_filepath: Path,
    dataset_directory: Path,
    sequence_length: int,
    minimum_samples: int = 1,
    *,
    excluded_experiments: Collection[str] = (),
    binarize: bool = True,
    allow_N: bool = False,
) -> None:
    """
    Create one H5 file of labeled CpG sequence windows per chromosome.

    Each window is centered on a CpG site and includes it. Calls on either
    strand of a CpG site are combined, and sites are labeled methylated if
    they have more methylated than unmethylated reads. Each file holds the
    one-hot encoded windows, the 1-based position of each site's C, and the
    labels of each experiment, with NaN for sites not covered in an experiment.

    The reference genome must be the assembly that the methylation calls are
    positioned in, e.g. GRCm38 for scBS-seq and GRCh38 or GRCm38 for lifted
    scRRBS-seq calls.

    Args
    ----
    methylation_directory: The directory of methylation profile files, one
        file per experiment (cell), with 1-based positions.
    fasta_filepath: The Ensembl FASTA file of the reference genome.
    dataset_directory: The directory in which to write the H5 files.
    sequence_length: The length of the sequence windows, e.g. 1001.
    minimum_samples: The minimum reads of a site in an experiment, summed over
        both strands, to be labeled, e.g. 4 for scRRBS-seq.
    excluded_experiments: The names of experiments to leave out.
    binarize: Label sites with binary methylation states if True. Otherwise,
        label sites with methylation ratios.
    allow_N: Keep windows holding nucleotides other than A, T, C and G if
        True. Otherwise, drop them.
    """
    filepaths_by_experiment: dict[str, Path] = _find_methylation_filepaths(
        methylation_directory, excluded_experiments
    )
    experiment_names: list[str] = list(filepaths_by_experiment)
    counts_by_chromosome: dict[str, NDArray[np.int64]] = (
        read_methylation_counts(list(filepaths_by_experiment.values()))
    )

    fasta_line_length: int = dna.determine_line_length(fasta_filepath)
    fasta_metadata: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        fasta_filepath
    )
    for chromosome in sorted(counts_by_chromosome):
        counts: NDArray[np.int64] = counts_by_chromosome[chromosome]
        if chromosome not in fasta_metadata:
            logger.warning(
                "Skipping %d methylation calls on chromosome %s, which is not "
                "in %s",
                len(counts),
                chromosome,
                fasta_filepath,
            )
            continue

        reference: NDArray[np.uint8] = read_reference(
            fasta_filepath, fasta_metadata[chromosome], fasta_line_length
        )
        cpg_indices: NDArray[np.int64] = locate_cpg_sites(
            reference, counts[:, 0]
        )
        is_cpg: NDArray[np.bool_] = cpg_indices >= 0
        if not is_cpg.all():
            logger.warning(
                "%d of %d methylation calls on chromosome %s are not at a CpG "
                "site of %s. Check that the calls are 1-based and positioned "
                "in the same assembly as the reference genome.",
                np.count_nonzero(~is_cpg),
                len(counts),
                chromosome,
                fasta_filepath,
            )

        sites, labels = aggregate_cpg_counts(
            cpg_indices[is_cpg],
            counts[is_cpg],
            len(experiment_names),
            minimum_reads=minimum_samples,
            binarize=binarize,
        )
        if len(sites) == 0:
            logger.warning("No CpG sites found on chromosome %s", chromosome)
            continue

        samples_written: int = _write_chromosome_h5(
            dataset_directory / f"chr{chromosome}.h5",
            reference,
            sites,
            labels,
            experiment_names,
            sequence_length,
            allow_N=allow_N,
        )
        logger.info(
            "Wrote %d samples for chromosome %s", samples_written, chromosome
        )


@dataclass
class _DeepCpGRows:
    """Rows of DeepCpG data files, sorted by position."""

    # The 1-based position of each row's methylation call.
    positions: NDArray[np.int64]
    # The nucleotide codes of the sequence window centered on each position.
    windows: NDArray[np.int8]
    # The label of each row in each experiment, NaN if unobserved.
    labels: NDArray[np.float32]

    def select(self, rows: NDArray[np.bool_]) -> _DeepCpGRows:
        return _DeepCpGRows(
            self.positions[rows], self.windows[rows], self.labels[rows]
        )

    def concatenate(self, other: _DeepCpGRows) -> _DeepCpGRows:
        return _DeepCpGRows(
            np.concatenate([self.positions, other.positions]),
            np.concatenate([self.windows, other.windows]),
            np.concatenate([self.labels, other.labels]),
        )


def _find_deepcpg_files(
    deepcpg_directory: Path,
) -> tuple[dict[str, list[Path]], list[str], int]:
    """
    Find the DeepCpG data files of each chromosome in a directory.

    Args
    ----
    deepcpg_directory: The directory of DeepCpG data files.

    Returns
    -------
    The files of each chromosome in position order, the names of the
    experiments, and the length of the sequence windows.

    Raises
    ------
    FileNotFoundError if the directory holds no H5 files.
    ValueError if a file is not a DeepCpG data file, or the files differ in
    experiments or window length.
    """
    first_positions: dict[str, list[tuple[int, Path]]] = {}
    experiment_names: list[str] = []
    window_length: int = 0
    first_filepath: Path | None = None
    for filepath in sorted(deepcpg_directory.glob("*.h5")):
        with h5py.File(filepath, "r") as fd:
            missing_keys: list[str] = [
                key
                for key in [
                    DEEPCPG_CHROMOSOME_KEY,
                    DEEPCPG_POSITION_KEY,
                    f"{INPUTS_STR}/{DNA_STR}",
                    DEEPCPG_OUTPUTS_KEY,
                ]
                if key not in fd
            ]
            if missing_keys:
                raise ValueError(
                    f"{filepath} is not a DeepCpG data file, it lacks "
                    f"{missing_keys}"
                )

            outputs: h5py.Group = fd[DEEPCPG_OUTPUTS_KEY]
            if not all(
                isinstance(outputs[name], h5py.Dataset) for name in outputs
            ):
                raise ValueError(
                    f"{filepath} must hold one label dataset per experiment "
                    f"in '{DEEPCPG_OUTPUTS_KEY}'"
                )

            file_window_length: int = fd[INPUTS_STR][DNA_STR].shape[1]
            if first_filepath is None:
                first_filepath = filepath
                experiment_names = list(outputs)
                window_length = file_window_length
            else:
                check_experiment_names(
                    experiment_names,
                    list(outputs),
                    str(first_filepath),
                    str(filepath),
                )
                if file_window_length != window_length:
                    raise ValueError(
                        f"Windows of {filepath} have length "
                        f"{file_window_length}, but those of {first_filepath} "
                        f"have length {window_length}"
                    )

            positions: h5py.Dataset = fd[DEEPCPG_POSITION_KEY]
            if len(positions) == 0:
                logger.warning("No sites in %s", filepath)
                continue

            chromosome: str = methylation.normalize_chromosome(
                fd[DEEPCPG_CHROMOSOME_KEY].asstr()[0]
            )
            first_positions.setdefault(chromosome, []).append(
                (int(positions[0]), filepath)
            )

    if first_filepath is None:
        raise FileNotFoundError(f"No H5 files found in {deepcpg_directory}")

    return (
        {
            chromosome: [filepath for _, filepath in sorted(files)]
            for chromosome, files in first_positions.items()
        },
        experiment_names,
        window_length,
    )


def _read_deepcpg_rows(
    filepath: Path, experiment_names: Sequence[str]
) -> _DeepCpGRows:
    """
    Read the rows of a DeepCpG data file.

    Args
    ----
    filepath: The DeepCpG data file.
    experiment_names: The names of the experiments whose labels to read.

    Returns
    -------
    The rows of the file.

    Raises
    ------
    ValueError if the file holds sites of more than one chromosome.
    """
    with h5py.File(filepath, "r") as fd:
        if len(np.unique(fd[DEEPCPG_CHROMOSOME_KEY][()])) > 1:
            raise ValueError(
                f"{filepath} holds sites of more than one chromosome"
            )

        outputs: h5py.Group = fd[DEEPCPG_OUTPUTS_KEY]
        labels: NDArray[np.float32] = np.stack(
            [outputs[name][()] for name in experiment_names], axis=1
        ).astype(np.float32)
        labels[labels == MISSING_LABEL] = np.nan
        return _DeepCpGRows(
            positions=fd[DEEPCPG_POSITION_KEY][()].astype(np.int64),
            windows=fd[INPUTS_STR][DNA_STR][()],
            labels=labels,
        )


def _deepcpg_sites(rows: _DeepCpGRows) -> NDArray[np.int64]:
    """
    Find the CpG site of each row of DeepCpG data files.

    DeepCpG centers each row's window on the row's position, which is either
    the C or, for calls on the reverse strand, the G of a CpG site.

    Args
    ----
    rows: The rows.

    Returns
    -------
    The 1-based position of the C of each row's CpG site, or -1 if the row's
    position is not in a CpG site.
    """
    center: int = rows.windows.shape[1] // 2
    c_code: int = DEEPCPG_NUCLEOTIDE_CODES["C"]
    g_code: int = DEEPCPG_NUCLEOTIDE_CODES["G"]
    is_c: NDArray[np.bool_] = (rows.windows[:, center] == c_code) & (
        rows.windows[:, center + 1] == g_code
    )
    is_g: NDArray[np.bool_] = (rows.windows[:, center] == g_code) & (
        rows.windows[:, center - 1] == c_code
    )
    return np.where(
        is_c, rows.positions, np.where(is_g, rows.positions - 1, -1)
    )


def _merge_deepcpg_labels(
    labels: NDArray[np.float32],
    first_rows: NDArray[np.int64],
    *,
    binarize: bool = True,
) -> tuple[NDArray[np.float32], int]:
    """
    Combine the labels of the rows of each CpG site of DeepCpG data files.

    A site has a row for each strand with a call. DeepCpG data files hold
    methylation ratios rather than read counts, so the ratios of a site's
    strands are averaged in each experiment. Binary labels mark a site as
    methylated if the averaged ratio exceeds 0.5, i.e. it has more methylated
    than unmethylated reads when both strands have equal coverage.

    Args
    ----
    labels: The label of each row in each experiment, NaN if unobserved,
        with the rows of each site consecutive.
    first_rows: The index of the first row of each site.
    binarize: Label sites with binary methylation states if True. Otherwise,
        label sites with methylation ratios.

    Returns
    -------
    The label of each site in each experiment, NaN if unobserved, and the
    quantity of labels whose strands disagree on the methylation state.
    """
    is_observed: NDArray[np.bool_] = ~np.isnan(labels)
    observed_quantities: NDArray[np.int64] = np.add.reduceat(
        is_observed.astype(np.int64), first_rows, axis=0
    )
    methylated_quantities: NDArray[np.int64] = np.add.reduceat(
        (labels > 0.5).astype(np.int64), first_rows, axis=0
    )
    ratio_sums: NDArray[np.float64] = np.add.reduceat(
        np.where(is_observed, labels, 0.0), first_rows, axis=0
    )
    discordant_quantity: int = np.count_nonzero(
        (methylated_quantities > 0)
        & (methylated_quantities < observed_quantities)
    )

    with np.errstate(invalid="ignore"):
        ratios: NDArray[np.float64] = ratio_sums / observed_quantities

    if binarize:
        ratios = np.where(np.isnan(ratios), np.nan, ratios > 0.5)

    return ratios.astype(np.float32), discordant_quantity


def _convert_deepcpg_chromosome(
    filepaths: Sequence[Path],
    h5_filepath: Path,
    experiment_names: Sequence[str],
    sequence_length: int,
    *,
    binarize: bool = True,
    allow_N: bool = False,
) -> int:
    """
    Write the DeepCpG data files of a chromosome as one H5 file of samples.

    The rows of each CpG site, one per strand with a call, are combined into
    one sample centered on the site's C, as by `window_starts`. Its window is
    cut from the window of its row on the C, or on the G if the C has no
    call. The files are read one at a time.

    Args
    ----
    filepaths: The chromosome's DeepCpG data files in position order.
    h5_filepath: The H5 file to write.
    experiment_names: The names of the experiments to write labels of.
    sequence_length: The length of the sequence windows, at most the length
        of the DeepCpG windows.
    binarize: Label sites with binary methylation states if True. Otherwise,
        label sites with methylation ratios.
    allow_N: Keep windows holding nucleotides other than A, T, C and G, which
        are encoded as all zeros, if True. Otherwise, drop them.

    Returns
    -------
    The quantity of samples written.

    Raises
    ------
    ValueError if the positions of the files are unsorted or overlap.
    """
    samples_written: int = 0
    non_cpg_rows: int = 0
    unobserved_sites: int = 0
    cut_off_sites: int = 0
    unknown_sites: int = 0
    discordant_labels: int = 0
    pending: _DeepCpGRows | None = None
    with h5py.File(h5_filepath, "w") as fd:
        datasets = _ChromosomeDatasets.create(
            fd, sequence_length, experiment_names
        )
        for file_index, filepath in enumerate(filepaths):
            rows: _DeepCpGRows = _read_deepcpg_rows(filepath, experiment_names)
            if pending is not None:
                rows = pending.concatenate(rows)
            if np.any(np.diff(rows.positions) <= 0):
                raise ValueError(
                    f"Positions of {filepath} are unsorted or overlap those "
                    "of the preceding file"
                )

            sites: NDArray[np.int64] = _deepcpg_sites(rows)
            is_cpg: NDArray[np.bool_] = sites >= 0
            non_cpg_rows += np.count_nonzero(~is_cpg)
            rows, sites = rows.select(is_cpg), sites[is_cpg]

            # The rows of the last site may continue in the next file.
            pending = None
            if file_index < len(filepaths) - 1 and len(sites) > 0:
                is_pending: NDArray[np.bool_] = sites == sites[-1]
                pending = rows.select(is_pending)
                rows, sites = rows.select(~is_pending), sites[~is_pending]
            if len(sites) == 0:
                continue

            unique_sites, first_rows = np.unique(sites, return_index=True)
            labels, discordant_quantity = _merge_deepcpg_labels(
                rows.labels, first_rows, binarize=binarize
            )
            discordant_labels += discordant_quantity

            window_length: int = rows.windows.shape[1]
            offsets: NDArray[np.int64] = (
                window_length // 2
                - (rows.positions[first_rows] - unique_sites)
                - (sequence_length - 1) // 2
            )
            is_observed: NDArray[np.bool_] = ~np.isnan(labels).all(axis=1)
            fits: NDArray[np.bool_] = (offsets >= 0) & (
                offsets + sequence_length <= window_length
            )
            unobserved_sites += np.count_nonzero(~is_observed)
            cut_off_sites += np.count_nonzero(is_observed & ~fits)
            keep: NDArray[np.bool_] = is_observed & fits

            windows: NDArray[np.uint8] = rows.windows[
                first_rows[keep, np.newaxis],
                offsets[keep, np.newaxis] + np.arange(sequence_length),
            ].view(np.uint8)
            unique_sites, labels = unique_sites[keep], labels[keep]
            if not allow_N:
                are_known: NDArray[np.bool_] = _DEEPCPG_KNOWN_NUCLEOTIDES[
                    windows
                ].all(axis=1)
                unknown_sites += np.count_nonzero(~are_known)
                windows = windows[are_known]
                unique_sites, labels = (
                    unique_sites[are_known],
                    labels[are_known],
                )

            datasets.append(
                _DEEPCPG_ONE_HOT_NUCLEOTIDES[windows], unique_sites, labels
            )
            samples_written += len(unique_sites)

    chromosome: str = h5_filepath.stem
    if non_cpg_rows > 0:
        logger.warning(
            "Dropping %d rows of %s whose positions are not in a CpG site",
            non_cpg_rows,
            chromosome,
        )
    if unobserved_sites > 0:
        logger.info(
            "Dropping %d sites of %s unobserved in every experiment",
            unobserved_sites,
            chromosome,
        )
    if cut_off_sites > 0:
        logger.info(
            "Dropping %d sites of %s with only a reverse strand call, whose "
            "window exceeds the DeepCpG window",
            cut_off_sites,
            chromosome,
        )
    if unknown_sites > 0:
        logger.info(
            "Dropping %d sites whose windows hold unknown nucleotides in %s",
            unknown_sites,
            chromosome,
        )
    if discordant_labels > 0:
        logger.info(
            "%d labels of %s average strands that disagree on the "
            "methylation state",
            discordant_labels,
            chromosome,
        )

    return samples_written


def create_h5_dataset_from_deepcpg_files(
    deepcpg_directory: Path,
    dataset_directory: Path,
    sequence_length: int | None = None,
    *,
    excluded_experiments: Collection[str] = (),
    binarize: bool = True,
    allow_N: bool = False,
) -> None:
    """
    Create one H5 file of labeled CpG sequence windows per chromosome from
    DeepCpG data files.

    DeepCpG data files, named e.g. 'c1_3801088-3833856.h5', each hold
    consecutive sites of one chromosome: their chromosome ('chromo'), 1-based
    position ('pos'), the nucleotide codes of the sequence window centered on
    them ('inputs/dna'), and one dataset of methylation ratios per experiment
    in 'outputs', with -1 for unobserved sites. The files written match those
    of `create_h5_dataset_from_methylation_profiles`: calls on either strand
    of a CpG site are combined into one sample centered on its C.

    Args
    ----
    deepcpg_directory: The directory of DeepCpG data files.
    dataset_directory: The directory in which to write the H5 files.
    sequence_length: The length of the sequence windows, at most the length
        of the DeepCpG windows. Defaults to the length of the DeepCpG windows.
    excluded_experiments: The names of experiments to leave out.
    binarize: Label sites with binary methylation states if True. Otherwise,
        label sites with methylation ratios.
    allow_N: Keep windows holding nucleotides other than A, T, C and G if
        True. Otherwise, drop them.

    Raises
    ------
    ValueError if the sequence length exceeds the length of the DeepCpG
    windows or every experiment is excluded.
    """
    filepaths_by_chromosome, experiment_names, window_length = (
        _find_deepcpg_files(deepcpg_directory)
    )
    for experiment_name in excluded_experiments:
        if experiment_name not in experiment_names:
            logger.warning(
                "Excluded experiment %s not found in %s",
                experiment_name,
                deepcpg_directory,
            )
    experiment_names = [
        experiment_name
        for experiment_name in experiment_names
        if experiment_name not in excluded_experiments
    ]
    if not experiment_names:
        raise ValueError(f"Every experiment of {deepcpg_directory} is excluded")

    if sequence_length is None:
        sequence_length = window_length
    elif not 0 < sequence_length <= window_length:
        raise ValueError(
            f"Sequence length must be between 1 and {window_length}, the "
            f"length of the windows in {deepcpg_directory}, not "
            f"{sequence_length}"
        )

    dataset_directory.mkdir(parents=True, exist_ok=True)
    for chromosome in sorted(filepaths_by_chromosome):
        h5_filepath: Path = dataset_directory / f"chr{chromosome}.h5"
        samples_written: int = _convert_deepcpg_chromosome(
            filepaths_by_chromosome[chromosome],
            h5_filepath,
            experiment_names,
            sequence_length,
            binarize=binarize,
            allow_N=allow_N,
        )
        if samples_written == 0:
            h5_filepath.unlink()
            logger.warning("No CpG sites found on chromosome %s", chromosome)
        else:
            logger.info(
                "Wrote %d samples for chromosome %s",
                samples_written,
                chromosome,
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
