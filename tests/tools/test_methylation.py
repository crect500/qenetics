import gzip
import shutil
from pathlib import Path
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
    assert len(methylation_profiles) == 6

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
    assert len(profiles_by_chromosome["1"]) == 4


@pytest.mark.parametrize(
    ("chromosome", "expected_result"),
    [
        ("1", "1"),
        ("chr1", "1"),
        ("Chr12", "12"),
        ("chrX", "X"),
        ("chrM", "MT"),
        ("M", "MT"),
        ("MT", "MT"),
    ],
)
def test_normalize_chromosome(chromosome: str, expected_result: str) -> None:
    assert methylation.normalize_chromosome(chromosome) == expected_result


@pytest.mark.parametrize(
    ("line", "expected_result"),
    [
        ("1\t2\t2\t100\t5\t0", True),
        ("#Chr\tPos\tRef\tChain\tTotal", False),
        ("<Chr>\t<Pos>\t<Ref>", False),
        ("", False),
        ("1", False),
    ],
)
def test_is_data_line(line: str, expected_result: bool) -> None:
    assert methylation._is_data_line(line.split()) == expected_result


@pytest.mark.parametrize(
    "process_line",
    [
        methylation._process_cov_methylation_line,
        methylation._process_cpg_methylation_line,
        methylation._process_deepcpg_methylation_line,
    ],
)
def test_process_methylation_line_skips_header(process_line) -> None:
    assert process_line("#Chr\tPos\tRef\tChain\tTotal\tMet") is None


def test_process_cov_methylation_line_normalizes_chromosome() -> None:
    methylation_profile: methylation.MethylationInfo | None = (
        methylation._process_cov_methylation_line("chr1\t2\t2\t50\t1\t1")
    )
    assert methylation_profile.chromosome == "1"
    assert methylation_profile.methylation_ratio == pytest.approx(0.5)


def test_process_cov_methylation_line_zero_count() -> None:
    with pytest.raises(ValueError, match="Experiment count is zero"):
        _ = methylation._process_cov_methylation_line("1\t2\t2\t0\t0\t0")


def test_process_cpg_methylation_line_columns() -> None:
    methylation_profile: methylation.MethylationInfo | None = (
        methylation._process_cpg_methylation_line(
            "chr1\t2\tC\t-\t5\t4\t1\t0.8\tCGC\tCpG"
        )
    )
    assert methylation_profile.chromosome == "1"
    assert methylation_profile.c_context == "C"
    assert methylation_profile.strand == "-"
    assert methylation_profile.trinucleotide_context == "CGC"
    assert methylation_profile.count_methylated == 4
    assert methylation_profile.count_unmethylated == 1


@pytest.mark.parametrize(
    ("line", "chromosome"),
    [("chr1\t2\t1.0\t5\t5\t0", "1"), ("12\t2\t1.0\t5\t5\t0", "12")],
)
def test_process_deepcpg_methylation_line_normalizes_chromosome(
    line: str, chromosome: str
) -> None:
    methylation_profile: methylation.MethylationInfo | None = (
        methylation._process_deepcpg_methylation_line(line)
    )
    assert methylation_profile.chromosome == chromosome


def test_process_deepcpg_methylation_line_zero_count() -> None:
    with pytest.raises(ValueError, match="Experiment count is zero"):
        _ = methylation._process_deepcpg_methylation_line(
            "chr1\t2\t0.0\t0\t0\t0"
        )


def test_retrieve_methylation_data_warns_when_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "empty.cov.txt"
        test_file.write_text("#Chr\tStart\tEnd\tPercent\tMet\tUnMet\n")
        with caplog.at_level("WARNING", logger="qenetics.tools.methylation"):
            assert list(methylation.retrieve_methylation_data(test_file)) == []

    assert "No methylation profiles read from" in caplog.text


def test_retrieve_methylation_data_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "sample.cov.txt"
        test_file.write_text("1\t2\t2\t100\t1\t0\n")
        with caplog.at_level("WARNING", logger="qenetics.tools.methylation"):
            assert (
                len(list(methylation.retrieve_methylation_data(test_file))) == 1
            )

    assert "No methylation profiles read from" not in caplog.text


@pytest.mark.parametrize("filename", ["sample.cov.txt", "sample.cov.txt.gz"])
def test_open_methylation_file(filename: str) -> None:
    contents: str = "1\t2\t2\t100\t1\t0\n"
    with TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / filename
        if test_file.suffix == ".gz":
            with gzip.open(test_file, "wt") as fd:
                fd.write(contents)
        else:
            test_file.write_text(contents)

        with methylation.open_methylation_file(test_file) as fd:
            assert fd.read() == contents


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [
        ("sample.cov.txt", methylation.MethylationFormat.COV),
        ("sample.COV.txt", methylation.MethylationFormat.COV),
        ("sample.CpG.cov.gz", methylation.MethylationFormat.COV),
        ("sample.cpg.txt", methylation.MethylationFormat.CPG),
        ("sample_RRBS.single.CpG.txt.gz", methylation.MethylationFormat.CPG),
        ("sample.CPG.txt", methylation.MethylationFormat.CPG),
        ("sample.tsv", methylation.MethylationFormat.DEEPCPG),
        ("sample.TSV.gz", methylation.MethylationFormat.DEEPCPG),
    ],
)
def test_determine_format(
    filename: str, expected_format: methylation.MethylationFormat
) -> None:
    assert methylation.determine_format(Path(filename)) == expected_format


def test_determine_format_unrecognized() -> None:
    with pytest.raises(ValueError, match="not recognized"):
        _ = methylation.determine_format(Path("sample.txt"))


def test_retrieve_methylation_data_gzip(test_methylation_file: Path) -> None:
    with TemporaryDirectory() as temp_dir:
        gzip_file = Path(temp_dir) / (test_methylation_file.name + ".gz")
        with gzip.open(gzip_file, "wb") as fd:
            fd.write(test_methylation_file.read_bytes())

        assert list(methylation.retrieve_methylation_data(gzip_file)) == list(
            methylation.retrieve_methylation_data(test_methylation_file)
        )


def test_convert_methylation_profiles_gzip() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "sample.cov.txt.gz"
        with gzip.open(input_file, "wt") as fd:
            fd.write("chr1\t2\t2\t50\t1\t1\n")

        methylation.convert_methylation_profiles(
            input_file, temp_path, methylation.MethylationFormat.DEEPCPG
        )

        assert (temp_path / "sample.tsv").read_text() == "1\t2\t0.5\t2\t1\t1\n"
