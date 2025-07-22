from pathlib import Path

import pytest

from qenetics.tools import cpg_sampler


@pytest.fixture
def test_fasfa_metadata() -> dict[str, cpg_sampler.SequenceInfo]:
    return {
        "1": cpg_sampler.SequenceInfo(
            length=44, is_chromosome=True, file_position=47
        ),
        "2": cpg_sampler.SequenceInfo(
            length=88, is_chromosome=True, file_position=139
        ),
    }


@pytest.fixture
def test_fasfa_file() -> Path:
    return Path("tests/test_files/test_sequence.fa")


@pytest.fixture
def test_methylation_profiles() -> dict[
    str, dict[int, cpg_sampler.MethylationInfo]
]:
    return {
        "1": {
            2: cpg_sampler.MethylationInfo(
                count_methylated=2, count_non_methylated=1
            ),
            3: cpg_sampler.MethylationInfo(
                count_methylated=1, count_non_methylated=0
            ),
            4: cpg_sampler.MethylationInfo(
                count_methylated=1, count_non_methylated=1
            ),
        },
        "2": {
            5: cpg_sampler.MethylationInfo(
                count_methylated=1, count_non_methylated=2
            )
        },
        "X": {
            6: cpg_sampler.MethylationInfo(
                count_methylated=0, count_non_methylated=1
            )
        },
    }


@pytest.fixture
def test_methylation_file() -> Path:
    return Path("tests/test_files/test_methylation_profile.tsv")


@pytest.fixture
def test_sequences() -> list[str]:
    return [
        "ACTGACTGACTGACTG",
        "GTCAGTCAGTCAGTCA",
        "AAAATTTTCCCCGGGG",
        "GGGGCCCCTTTTAAAA",
    ]
