from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from qenetics.tools import dna


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
        dna._write_sequence(sequence, name, is_chromosome, output_filepath)
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
    result_name, result_is_chromosome = dna._read_tengenomics_annotation(
        annotation
    )
    assert result_name == name
    assert result_is_chromosome == is_chromosome


def test_write_ensembl_from_tengenomics() -> None:
    with TemporaryDirectory() as temp_dir:
        ensembl_file = Path(temp_dir) / "sequence.fa"
        dna.write_ensembl_from_tengenomics(
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
    chromosome_number, info = dna.extract_line_annotation(test_line)
    assert chromosome_number == chromosome
    assert info.length == length


def test_find_next_comment() -> None:
    with Path("tests/test_files/test_sequence.fa").open() as fd:
        assert dna.find_next_comment(fd, 0)
        assert fd.tell() == 1

        assert dna.find_next_comment(fd, 95)
        assert fd.tell() == 96

        assert dna.find_next_comment(fd, 235)
        assert fd.tell() == 236


def test_extract_fasta_metadata() -> None:
    annotations: dict[str, dna.SequenceInfo] = dna.extract_fasta_metadata(
        Path("tests/test_files/test_sequence.fa"), crlf=True
    )
    assert len(annotations) == 3

    assert annotations["1"].length == 43
    assert annotations["1"].file_position == 50

    assert annotations["2"].length == 86
    assert annotations["2"].file_position == 145

    assert annotations["3"].length == 50
    assert annotations["3"].file_position == 285


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
                dna._read_sequence(fd, sequence_length, line_length)
                == correct_sequence
            )


def test_find_methylation_sequence(
    test_fasta_metadata: dict[str, dna.SequenceInfo],
) -> None:
    with Path("tests/test_files/test_sequence.fa").open() as fd:
        assert (
            dna.find_methylation_sequence(
                "2", 16, test_fasta_metadata, fd, 4, 45
            )
            == "ATCG"
        )
        assert (
            dna.find_methylation_sequence(
                "2", 7, test_fasta_metadata, fd, 8, 45
            )
            == "ACTGAATG"
        )
        assert (
            dna.find_methylation_sequence(
                "2", 42, test_fasta_metadata, fd, 8, 45
            )
            == "GGGGAATT"
        )
        assert not dna.find_methylation_sequence(
            "1", 1, test_fasta_metadata, fd, 4, 45
        )
        assert not dna.find_methylation_sequence(
            "1", 43, test_fasta_metadata, fd, 4, 45
        )
