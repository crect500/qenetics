from pathlib import Path
import pytest
from tempfile import TemporaryDirectory

from qenetics.tools import cpg_sampler


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

        assert cpg_sampler.find_next_comment(fd, 92)
        assert fd.tell() == 93

        assert cpg_sampler.find_next_comment(fd, 229)
        assert fd.tell() == 230


def test_extract_fasfa_metadata() -> None:
    annotations: dict[str, cpg_sampler.SequenceInfo] = (
        cpg_sampler.extract_fasfa_metadata(
            Path("tests/test_files/test_sequence.fa")
        )
    )
    assert len(annotations) == 2

    assert annotations["1"].length == 44
    assert annotations["1"].file_position == 47

    assert annotations["2"].length == 88
    assert annotations["2"].file_position == 139


def test_load_methlation_file_data() -> None:
    methylation_data: dict[str, dict[int, cpg_sampler.MethylationInfo]] = dict()
    cpg_sampler._load_methlation_file_data(
        Path("tests/test_files/test_methylation_profile.tsv"), methylation_data
    )
    assert len(methylation_data) == 3
    assert len(methylation_data["1"]) == 3
    assert methylation_data["1"][2].count_methylated == 2
    assert methylation_data["1"][2].count_non_methylated == 1
    assert methylation_data["2"][5].count_methylated == 1
    assert methylation_data["2"][5].count_non_methylated == 2
    assert methylation_data["X"][6].count_methylated == 0
    assert methylation_data["X"][6].count_non_methylated == 1


def test_filter_and_calculate_methylation(
    test_methylation_profiles: dict[
        str, dict[int, cpg_sampler.MethylationInfo]
    ],
) -> None:
    filtered_profiles: dict[str, dict[int, cpg_sampler.MethylationInfo]] = (
        cpg_sampler.filter_and_calculate_methylation(test_methylation_profiles)
    )
    assert len(filtered_profiles) == 3
    assert len(filtered_profiles["1"]) == 3
    assert len(filtered_profiles["2"]) == 1
    assert len(filtered_profiles["X"]) == 1
    assert filtered_profiles["1"][2].ratio_methylated == pytest.approx(
        0.67, abs=0.01
    )

    filtered_profiles = cpg_sampler.filter_and_calculate_methylation(
        test_methylation_profiles, 2
    )
    assert len(filtered_profiles) == 2
    assert len(filtered_profiles["1"]) == 2
    assert len(filtered_profiles["2"]) == 1


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
    test_fasfa_metadata: dict[str, cpg_sampler.SequenceInfo],
) -> None:
    with Path("tests/test_files/test_sequence.fa").open() as fd:
        assert (
            cpg_sampler.find_methylation_sequence(
                "2", 16, test_fasfa_metadata, fd, 4, 45
            )
            == "ATCG"
        )
        assert (
            cpg_sampler.find_methylation_sequence(
                "2", 7, test_fasfa_metadata, fd, 8, 45
            )
            == "ACTGAATG"
        )
        assert (
            cpg_sampler.find_methylation_sequence(
                "2", 42, test_fasfa_metadata, fd, 8, 45
            )
            == "GGGGAATT"
        )
        assert not cpg_sampler.find_methylation_sequence(
            "1", 1, test_fasfa_metadata, fd, 4, 45
        )
        assert not cpg_sampler.find_methylation_sequence(
            "1", 43, test_fasfa_metadata, fd, 4, 45
        )


def test_retrieve_all_cpg_sequences(
    test_fasfa_file: Path,
    test_fasfa_metadata: dict[str, cpg_sampler.SequenceInfo],
    test_methylation_profiles: dict[
        str, dict[int, cpg_sampler.MethylationInfo]
    ],
) -> None:
    results: list[cpg_sampler.MethylationSequence] = list(
        cpg_sampler.retrieve_all_cpg_sequences(
            test_fasfa_metadata, test_fasfa_file, test_methylation_profiles, 8
        )
    )
    assert len(results) == 2

    assert results[0].sequence == "ACTGTGAC"
    assert results[0].methylation_profile.count_methylated == 1
    assert results[0].methylation_profile.count_non_methylated == 1

    assert results[1].sequence == "NNACAAAA"
    assert results[1].methylation_profile.count_methylated == 1
    assert results[1].methylation_profile.count_non_methylated == 2
