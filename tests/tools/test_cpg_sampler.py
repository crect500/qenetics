from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from numpy.typing import NDArray
import pytest

from qenetics.tools import cpg_sampler


@pytest.mark.parametrize(
    ("sequence", "name", "is_chromosome"),
    [
        ("abc\ncde\n", "1", True),
        ("abc\ncd\n", "MT", True),
        ("abc\ncde\nf", "abcd.1", False),
    ],
)
def test_write_sequence(sequence: str, name: str, is_chromosome: bool) -> None:
    with TemporaryDirectory() as temp_dir:
        output_filepath = Path(temp_dir) / "sequence.fa"
        cpg_sampler._write_sequence(
            sequence, name, is_chromosome, output_filepath
        )
        with open(output_filepath) as fd:
            annotation: str = fd.readline()
            if is_chromosome:
                assert (
                    annotation
                    == f">{name} dna:chromosome chromosome:GRCm38:{name}:1:{len(sequence) - 2}:1 REF\n"
                )
            else:
                assert (
                    annotation
                    == f">{name} dna_sm:scaffold scaffold:GRCm38:{name}:1:{len(sequence) - 2}:1 REF\n"
                )


@pytest.mark.parametrize(
    ("annotation", "name", "is_chromosome"),
    [
        ("chr1 1", "1", True),
        ("chrM MT", "MT", True),
        ("abcd.1 abcd.1", "abcd.1", False),
    ],
)
def test_read_tengenomics_annotation(
    annotation: str, name: str, is_chromosome: bool
) -> None:
    result_name, result_is_chromosome = (
        cpg_sampler._read_tengenomics_annotation(annotation)
    )
    assert result_name == name
    assert result_is_chromosome == is_chromosome


def test_write_ensembl_from_tengenomics() -> None:
    with TemporaryDirectory() as temp_dir:
        ensembl_file = Path(temp_dir) / "sequence.fa"
        cpg_sampler.write_ensembl_from_tengenomics(
            Path("tests/test_files/test_tengenomics_data.fa"), ensembl_file
        )

        assert (
            ensembl_file.read_text()
            == Path("tests/test_files/test_sequence.fa").read_text()
        )


@pytest.mark.parametrize(
    ("test_line", "chromosome", "length"),
    [
        (">1 dna:chromosome chromosome:abc:1:1:2:1 REF", "1", 2),
        (">2 dna:chromosome chromosome:abc:2:1:50:1 REF", "2", 50),
        (">A dna:chromosome chromosome:abc:X:1:150:1 REF", "X", 150),
        (">A dna:nomonome nomosome:abc:A:1:20:1 REF", "A", 20),
    ],
)
def test_extract_line_annotations(
    test_line: str, chromosome: int, length: int
) -> None:
    chromosome_number, info = cpg_sampler.extract_line_annotation(test_line)
    assert chromosome_number == chromosome
    assert info.length == length


def test_find_next_comment() -> None:
    with Path("tests/test_files/test_sequence.fa").open() as fd:
        assert cpg_sampler.find_next_comment(fd, 0)
        assert fd.tell() == 1

        assert cpg_sampler.find_next_comment(fd, 95)
        assert fd.tell() == 96

        assert cpg_sampler.find_next_comment(fd, 235)
        assert fd.tell() == 236


def test_extract_fasta_metadata() -> None:
    annotations: dict[str, cpg_sampler.SequenceInfo] = (
        cpg_sampler.extract_fasta_metadata(
            Path("tests/test_files/test_sequence.fa")
        )
    )
    assert len(annotations) == 2

    assert annotations["1"].length == 43
    assert annotations["1"].file_position == 50

    assert annotations["2"].length == 86
    assert annotations["2"].file_position == 145


@pytest.mark.parametrize(
    ("line", "chromosome", "position", "ratio", "count"),
    [
        ("1\t2\t2\t100\t5\t0", "1", 2, 1.0, 5),
        ("X\t3\t3\t33.33333\t1\t2", "X", 3, 0.3333333, 3),
    ],
)
def test_process_cov_methylation_line(
    line: str, chromosome: str, position: int, ratio: float, count: int
) -> None:
    minimum_count: int = 4
    methylation_profile: cpg_sampler.MethylationInfo | None = (
        cpg_sampler._process_cov_methylation_line(line, minimum_count)
    )
    if count >= minimum_count:
        assert methylation_profile.chromosome == chromosome
        assert methylation_profile.position == position
        assert methylation_profile.methylation_ratio == pytest.approx(ratio)
        assert methylation_profile.experiment_count == count
    else:
        assert methylation_profile is None


@pytest.mark.parametrize(
    ("line", "chromosome", "position", "ratio", "count"),
    [
        ("chr1\t2\tC\t-\t5\t5\t0\t1.0\tCGC\tCpG", "1", 2, 1.0, 5),
        ("chrX\t3\tg\t+\t3\t1\t0\t0.33333\tCGC\tCpG", "X", 3, 0.3333333, 3),
    ],
)
def test_process_cpg_methylation_line(
    line: str, chromosome: str, position: int, ratio: float, count: int
) -> None:
    minimum_count: int = 4
    methylation_profile: cpg_sampler.MethylationInfo | None = (
        cpg_sampler._process_cpg_methylation_line(line, minimum_count)
    )
    if count >= minimum_count:
        assert methylation_profile.chromosome == chromosome
        assert methylation_profile.position == position
        assert methylation_profile.methylation_ratio == pytest.approx(ratio)
        assert methylation_profile.experiment_count == count
    else:
        assert methylation_profile is None


@pytest.mark.parametrize(
    ("line", "chromosome", "position", "ratio", "count"),
    [
        ("chr1\t2\t1.0\t5\t5\t0", "1", 2, 1.0, 5),
        ("chrX\t3\t0.33333\t3\t1\t2", "X", 3, 0.3333333, 3),
    ],
)
def test_process_deepcpg_methylation_line(
    line: str, chromosome: str, position: int, ratio: float, count: int
) -> None:
    minimum_count: int = 4
    methylation_profile: cpg_sampler.MethylationInfo | None = (
        cpg_sampler._process_deepcpg_methylation_line(line, minimum_count)
    )
    if count >= minimum_count:
        assert methylation_profile.chromosome == chromosome
        assert methylation_profile.position == position
        assert methylation_profile.methylation_ratio == pytest.approx(
            ratio, rel=1e-4
        )
        assert methylation_profile.experiment_count == count


def test_process_methylation_line() -> None:
    with (
        mock.patch(
            "qenetics.tools.cpg_sampler._process_cov_methylation_line"
        ) as mock_cov,
        mock.patch(
            "qenetics.tools.cpg_sampler._process_cpg_methylation_line"
        ) as mock_cpg,
    ):
        _ = cpg_sampler._process_methylation_line(
            "", cpg_sampler.MethylationFormat.COV
        )
        mock_cov.assert_called_once()
        mock_cpg.assert_not_called()

        _ = cpg_sampler._process_methylation_line(
            "", cpg_sampler.MethylationFormat.CPG
        )
        mock_cov.assert_called_once()
        mock_cpg.assert_called_once()


def test_retrieve_methylation_data(test_methylation_file: Path) -> None:
    methylation_profiles: list[cpg_sampler.MethylationInfo] = list(
        cpg_sampler.retrieve_methylation_data(test_methylation_file)
    )
    assert len(methylation_profiles) == 5

    with (
        TemporaryDirectory() as temp_dir,
        mock.patch(
            "qenetics.tools.cpg_sampler._process_cpg_methylation_line"
        ) as mock_cpg,
    ):
        test_file = Path(temp_dir) / "test_file.cpg.txt"
        test_file.write_text("line")
        _ = list(cpg_sampler.retrieve_methylation_data(test_file))
        mock_cpg.assert_called_once()


def test_cov_methylation_line() -> None:
    assert (
        cpg_sampler._cov_methylation_line(
            cpg_sampler.MethylationInfo(
                chromosome="1",
                position=2,
                methylation_ratio=0.2,
                experiment_count=5,
                count_methylated=1,
                count_unmethylated=4,
            )
        )
        == "1\t2\t2\t20.0\t1\t4\n"
    )


def test_cpg_methylation_line() -> None:
    assert (
        cpg_sampler._cpg_methylation_line(
            cpg_sampler.MethylationInfo(
                chromosome="1",
                position=2,
                methylation_ratio=0.2,
                experiment_count=5,
                count_methylated=1,
                count_unmethylated=4,
                c_context="g",
                strand="+",
                trinucleotide_context="CGT",
            )
        )
        == "chr1\t2\tg\t+\t5\t1\t4\t0.2\tCGT\tCpG\n"
    )

    assert (
        cpg_sampler._cpg_methylation_line(
            cpg_sampler.MethylationInfo(
                chromosome="1",
                position=2,
                methylation_ratio=0.2,
                experiment_count=5,
                count_methylated=1,
                count_unmethylated=4,
            )
        )
        == "chr1\t2\tN\tN\t5\t1\t4\t0.2\tNNN\tCpG\n"
    )


def test_deepcpg_methylation_line() -> None:
    assert (
        cpg_sampler._deepcpg_methylation_line(
            cpg_sampler.MethylationInfo(
                chromosome="1",
                position=2,
                methylation_ratio=0.2,
                experiment_count=5,
                count_methylated=1,
                count_unmethylated=4,
            )
        )
        == "chr1\t2\t0.2\t5\t1\t4\n"
    )


def test_methylation_line() -> None:
    with (
        mock.patch(
            "qenetics.tools.cpg_sampler._cov_methylation_line"
        ) as mock_cov,
        mock.patch(
            "qenetics.tools.cpg_sampler._cpg_methylation_line"
        ) as mock_cpg,
        mock.patch(
            "qenetics.tools.cpg_sampler._deepcpg_methylation_line"
        ) as mock_deepcpg,
    ):
        methylation_profile = cpg_sampler.MethylationInfo(
            chromosome="1",
            position=1,
            methylation_ratio=0.5,
            experiment_count=2,
            count_methylated=1,
            count_unmethylated=1,
        )
        cpg_sampler._methylation_line(
            methylation_profile, cpg_sampler.MethylationFormat.COV
        )
        mock_cov.assert_called_once()

        cpg_sampler._methylation_line(
            methylation_profile, cpg_sampler.MethylationFormat.CPG
        )
        mock_cpg.assert_called_once()

        cpg_sampler._methylation_line(
            methylation_profile, cpg_sampler.MethylationFormat.DEEPCPG
        )
        mock_deepcpg.assert_called_once()


@pytest.mark.parametrize(
    ("whole_sequence", "sequence_length", "line_length", "correct_sequence"),
    [
        ("A", 4, 2, None),
        ("ACTGACTG", 4, 80, "ACAC"),
        ("ACTG\nACTGACTG", 4, 80, "ACAC"),
        ("ACTGACTG\nACTGACTG\nACTG", 16, 8, "ACTGACTGTGACTGAC"),
    ],
)
def test_read_sequence(
    whole_sequence: str,
    sequence_length: int,
    line_length: int,
    correct_sequence: str | None,
) -> None:
    with TemporaryDirectory() as temp_dir:
        test_file: Path = Path(temp_dir) / "test_file.fa"
        with test_file.open("w") as fd:
            fd.write(whole_sequence)

        with test_file.open() as fd:
            assert (
                cpg_sampler._read_sequence(fd, sequence_length, line_length)
                == correct_sequence
            )


def test_find_methylation_sequence(
    test_fasta_metadata: dict[str, cpg_sampler.SequenceInfo],
) -> None:
    with Path("tests/test_files/test_sequence.fa").open() as fd:
        assert (
            cpg_sampler.find_methylation_sequence(
                "2", 16, test_fasta_metadata, fd, 4, 45
            )
            == "ATCG"
        )
        assert (
            cpg_sampler.find_methylation_sequence(
                "2", 7, test_fasta_metadata, fd, 8, 45
            )
            == "ACTGAATG"
        )
        assert (
            cpg_sampler.find_methylation_sequence(
                "2", 42, test_fasta_metadata, fd, 8, 45
            )
            == "GGGGAATT"
        )
        assert not cpg_sampler.find_methylation_sequence(
            "1", 1, test_fasta_metadata, fd, 4, 45
        )
        assert not cpg_sampler.find_methylation_sequence(
            "1", 43, test_fasta_metadata, fd, 4, 45
        )


@pytest.mark.parametrize(
    ("nucleotide", "expected_int"),
    [("A", 0), ("T", 1), ("C", 2), ("G", 3), ("N", -1)],
)
def test_nucleotide_character_to_numpy(
    nucleotide: str, expected_int: int
) -> None:
    if expected_int >= 0:
        expected_array: NDArray[int] = np.array([0] * 4, dtype=int)
        expected_array[expected_int] = 1
        assert (
            cpg_sampler.nucleotide_character_to_numpy(nucleotide)
            == expected_array
        ).all()
    else:
        with pytest.raises(
            ValueError,
            match=f"{nucleotide} is not a valid nucleotide designator",
        ):
            _ = cpg_sampler.nucleotide_character_to_numpy(nucleotide)


@pytest.mark.parametrize(
    ("sequence", "expected_array"),
    [
        ("A", [0]),
        ("AT", [0, 1]),
        ("ATC", [0, 1, 2]),
        ("ATCG", [0, 1, 2, 3]),
        ("NATCGN", []),
    ],
)
def test_nucleotide_string_to_numpy(
    sequence: str, expected_array: list[int]
) -> None:
    one_hot_matrix: NDArray[int] = cpg_sampler.nucleotide_string_to_numpy(
        sequence
    )
    if len(expected_array) == 0:
        assert one_hot_matrix is None
    else:
        working_matrix: list[list[int]] = []
        for nucleotide in expected_array:
            working_array: list[int] = [0] * 4
            working_array[nucleotide] = 1
            working_matrix.append(working_array)
        assert (one_hot_matrix == np.array(working_matrix, dtype=int)).all()


@pytest.mark.parametrize(
    ("threshold", "quantity_methylated"), [(0.0, 3), (0.5, 2), (1.0, 1)]
)
def test_samples_to_numpy(
    threshold: float, quantity_methylated: int, test_input_file: Path
) -> None:
    valid_samples: int = 3
    sequence_length: int = 12
    unique_nucleotides_quantity: int = 4
    samples, methylations = cpg_sampler.samples_to_numpy(
        test_input_file, threshold
    )
    print(samples)
    assert np.sum(methylations) == quantity_methylated
    assert samples.shape == (
        valid_samples,
        sequence_length,
        unique_nucleotides_quantity,
    )
