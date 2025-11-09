from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from qenetics.tools import methylation


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
    methylation_profile: methylation.MethylationInfo | None = (
        methylation._process_cov_methylation_line(line, minimum_count)
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
    methylation_profile: methylation.MethylationInfo | None = (
        methylation._process_cpg_methylation_line(line, minimum_count)
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
    methylation_profile: methylation.MethylationInfo | None = (
        methylation._process_deepcpg_methylation_line(line, minimum_count)
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
            "qenetics.tools.methylation._process_cov_methylation_line"
        ) as mock_cov,
        mock.patch(
            "qenetics.tools.methylation._process_cpg_methylation_line"
        ) as mock_cpg,
    ):
        _ = methylation.process_methylation_line(
            "", methylation.MethylationFormat.COV
        )
        mock_cov.assert_called_once()
        mock_cpg.assert_not_called()

        _ = methylation.process_methylation_line(
            "", methylation.MethylationFormat.CPG
        )
        mock_cov.assert_called_once()
        mock_cpg.assert_called_once()


def test_retrieve_methylation_data(test_methylation_file: Path) -> None:
    methylation_profiles: list[methylation.MethylationInfo] = list(
        methylation.retrieve_methylation_data(test_methylation_file)
    )
    assert len(methylation_profiles) == 5

    with (
        TemporaryDirectory() as temp_dir,
        mock.patch(
            "qenetics.tools.methylation._process_cpg_methylation_line"
        ) as mock_cpg,
    ):
        test_file = Path(temp_dir) / "test_file.cpg.txt"
        test_file.write_text("line")
        _ = list(methylation.retrieve_methylation_data(test_file))
        mock_cpg.assert_called_once()


def test_cov_methylation_line() -> None:
    assert (
        methylation._cov_methylation_line(
            methylation.MethylationInfo(
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
        methylation._cpg_methylation_line(
            methylation.MethylationInfo(
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
        methylation._cpg_methylation_line(
            methylation.MethylationInfo(
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
        methylation._deepcpg_methylation_line(
            methylation.MethylationInfo(
                chromosome="1",
                position=2,
                methylation_ratio=0.2,
                experiment_count=5,
                count_methylated=1,
                count_unmethylated=4,
            )
        )
        == "1\t2\t0.2\t5\t1\t4\n"
    )


def test_methylation_line() -> None:
    with (
        mock.patch(
            "qenetics.tools.methylation._cov_methylation_line"
        ) as mock_cov,
        mock.patch(
            "qenetics.tools.methylation._cpg_methylation_line"
        ) as mock_cpg,
        mock.patch(
            "qenetics.tools.methylation._deepcpg_methylation_line"
        ) as mock_deepcpg,
    ):
        methylation_profile = methylation.MethylationInfo(
            chromosome="1",
            position=1,
            methylation_ratio=0.5,
            experiment_count=2,
            count_methylated=1,
            count_unmethylated=1,
        )
        methylation._methylation_line(
            methylation_profile, methylation.MethylationFormat.COV
        )
        mock_cov.assert_called_once()

        methylation._methylation_line(
            methylation_profile, methylation.MethylationFormat.CPG
        )
        mock_cpg.assert_called_once()

        methylation._methylation_line(
            methylation_profile, methylation.MethylationFormat.DEEPCPG
        )
        mock_deepcpg.assert_called_once()


def test_record_methylation_profiles(test_methylation_file: Path) -> None:
    with TemporaryDirectory() as temp_file:
        temp_path = Path(temp_file)
        shutil.copy(test_methylation_file, temp_path)
        profiles_by_chromosome: dict[str, dict[int, dict[str, float]]] = (
            methylation.record_methylation_profiles(temp_path)
        )

    assert len(profiles_by_chromosome) == 3
    assert len(profiles_by_chromosome["1"]) == 3
