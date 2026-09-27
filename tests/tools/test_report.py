import html
import json
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

import numpy as np
import pytest

from qenetics.qcpg import records
from qenetics.tools import metrics, report

CELL_NAMES: list[str] = ["cellA", "cell <B>&C"]


def _create_run(
    records_directory: Path,
    now: datetime = datetime(2026, 9, 27, 13, 0, 0, tzinfo=UTC),
    experiment_name: str = "serum",
) -> Path:
    return records.create_run_record(
        records_directory,
        experiment_name,
        {"encoding": "onehot", "layer_quantity": 1, "epochs": 2},
        {"training": ["1"], "validation": ["3"], "test": ["2", "X"]},
        CELL_NAMES,
        now=now,
    )


def _complete_run(
    run_directory: Path, status: str = records.STATUS_COMPLETED
) -> None:
    """Record epochs and outputs where cellA sees both classes and the
    second cell only methylated sites, so its AUC is undefined."""
    rng = np.random.default_rng(0)
    for epoch in range(2):
        records.append_epoch(
            run_directory, epoch, 0.7 - 0.1 * epoch, 0.8, 0.6, [0.6, np.nan]
        )
    for split, chromosome_names in [("training", ["1"]), ("test", ["2", "X"])]:
        site_quantity = 12
        labels = rng.integers(0, 2, (site_quantity, 2)).astype(np.float32)
        labels[rng.random(labels.shape) < 0.3] = np.nan
        labels[:2, 0] = [0.0, 1.0]
        labels[:, 1] = np.where(np.isnan(labels[:, 1]), np.nan, 1.0)
        labels[0, 1] = 1.0
        predictions = rng.random((site_quantity, 2)).astype(np.float32)
        records.write_split_outputs(
            run_directory,
            split,
            records.SplitOutputs(
                chromosome_names=chromosome_names,
                chromosome_indices=(
                    np.arange(site_quantity) % len(chromosome_names)
                ).astype(np.int32),
                positions=(site_quantity - np.arange(site_quantity)) * 10,
                labels=labels,
                predictions=predictions,
            ),
        )
        cell_results, pooled_result = metrics.experiment_metrics(
            predictions, labels
        )
        records.write_metrics(
            run_directory, split, 0.5, cell_results, pooled_result
        )
    records.finish_run(run_directory, status)


def _broken_links(records_directory: Path) -> list[tuple[Path, str]]:
    broken: list[tuple[Path, str]] = []
    for page in records_directory.rglob("*.html"):
        for href in re.findall(r'href="([^"]+)"', page.read_text("utf-8")):
            target = page.parent / unquote(html.unescape(href))
            if not target.exists():
                broken.append((page, href))
    return broken


def _page_times(run_directory: Path) -> dict[Path, int]:
    return {
        page: page.stat().st_mtime_ns for page in run_directory.rglob("*.html")
    }


def test_add_run() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        run_directory = _create_run(records_directory)
        _complete_run(run_directory)
        report.add_run(records_directory, run_directory)

        cell_directories = report._cell_directories(CELL_NAMES)
        assert cell_directories == ["cellA", "cell__B__C"]
        cell_a = run_directory / "cells" / "cellA"
        for page in [
            records_directory / "index.html",
            records_directory / "serum" / "index.html",
            run_directory / "index.html",
            cell_a / "index.html",
            cell_a / "training" / "chr1.html",
            cell_a / "test" / "chr2.html",
            cell_a / "test" / "chrX.html",
            cell_a / "test" / "largest-errors.html",
            run_directory / "cells" / "cell__B__C" / "index.html",
        ]:
            assert page.exists(), page
        assert _broken_links(records_directory) == []

        run_page = (run_directory / "index.html").read_text("utf-8")
        assert "cell &lt;B&gt;&amp;C" in run_page
        assert "cell <B>" not in run_page
        # The second cell's AUC is undefined and shown as a dash.
        assert report.MISSING in run_page

        # The site table lists cellA's observed sites on chromosome 2 in
        # position order.
        outputs = records.read_split_outputs(run_directory, "test")
        is_expected = (outputs.chromosome_indices == 0) & ~np.isnan(
            outputs.labels[:, 0]
        )
        site_page = (cell_a / "test" / "chr2.html").read_text("utf-8")
        site_data = json.loads(
            re.search(
                r'<script type="application/json" id="site-data">(.*?)</script>',
                site_page,
            ).group(1)
        )
        assert [row[0] for row in site_data["rows"]] == sorted(
            outputs.positions[is_expected].tolist()
        )


def test_add_run_is_incremental() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        first_run = _create_run(records_directory)
        _complete_run(first_run)
        report.add_run(records_directory, first_run)
        first_run_pages = _page_times(first_run)

        second_run = _create_run(
            records_directory, now=datetime(2026, 9, 28, 9, 0, 0, tzinfo=UTC)
        )
        _complete_run(second_run)
        report.add_run(records_directory, second_run)

        # Adding the second run leaves the first run's pages untouched.
        assert _page_times(first_run) == first_run_pages
        experiment_index = (
            records_directory / "serum" / "index.html"
        ).read_text("utf-8")
        # Newest run first.
        assert experiment_index.index(second_run.name) < experiment_index.index(
            first_run.name
        )
        assert '<td class="number" data-value="2">2</td>' in (
            records_directory / "index.html"
        ).read_text("utf-8")
        assert _broken_links(records_directory) == []


def test_add_running_run() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        run_directory = _create_run(records_directory)
        report.add_run(records_directory, run_directory)

        assert "still training" in (run_directory / "index.html").read_text(
            "utf-8"
        )
        assert not (run_directory / "cells").exists()
        assert "status-running" in (
            records_directory / "serum" / "index.html"
        ).read_text("utf-8")

        _complete_run(run_directory)
        report.add_run(records_directory, run_directory)
        assert "still training" not in (run_directory / "index.html").read_text(
            "utf-8"
        )
        assert (run_directory / "cells" / "cellA" / "index.html").exists()


def test_add_failed_run() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        run_directory = _create_run(records_directory)
        _complete_run(run_directory, records.STATUS_FAILED)
        report.add_run(records_directory, run_directory)

        run_page = (run_directory / "index.html").read_text("utf-8")
        assert "status-failed" in run_page
        assert "results may be incomplete" in run_page


def test_concurrent_add_run() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        run_directories = [
            _create_run(
                records_directory,
                now=datetime(2026, 9, 27, 13, 0, 0, tzinfo=UTC),
            ),
            _create_run(
                records_directory,
                now=datetime(2026, 9, 27, 14, 0, 0, tzinfo=UTC),
                experiment_name="2i",
            ),
        ]
        for run_directory in run_directories:
            _complete_run(run_directory)

        threads = [
            threading.Thread(
                target=report.add_run, args=(records_directory, run_directory)
            )
            for run_directory in run_directories
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        top_index = (records_directory / "index.html").read_text("utf-8")
        assert "serum" in top_index
        assert "2i" in top_index
        assert not (records_directory / report.LOCK_FILENAME).exists()


def test_build_report() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        first_run = _create_run(records_directory)
        second_run = _create_run(
            records_directory, now=datetime(2026, 9, 28, 9, 0, 0, tzinfo=UTC)
        )
        for run_directory in [first_run, second_run]:
            _complete_run(run_directory)

        assert report.build_report(records_directory) == [first_run, second_run]
        assert report.build_report(records_directory) == []

        # A record updated after its pages were rendered is stale.
        future: float = time.time() + 100
        os.utime(first_run / records.METRICS_FILENAME, (future, future))
        assert report.is_run_stale(first_run)
        assert report.build_report(records_directory) == [first_run]

        assert report.build_report(records_directory, rebuild=True) == [
            first_run,
            second_run,
        ]
        assert report.build_report(
            records_directory, run_directory=second_run
        ) == [second_run]


def test_report_lock() -> None:
    with TemporaryDirectory() as temp_dir:
        records_directory = Path(temp_dir)
        lock_filepath = records_directory / report.LOCK_FILENAME

        # A lock left by a crashed process is removed once it is old.
        lock_filepath.touch()
        past: float = time.time() - 2 * report.LOCK_TIMEOUT_SECONDS
        os.utime(lock_filepath, (past, past))
        report.refresh_indexes(records_directory)
        assert not lock_filepath.exists()

        # A lock held by another process makes others wait, then give up.
        lock_filepath.touch()
        future: float = time.time() + 100
        os.utime(lock_filepath, (future, future))
        with (
            pytest.raises(TimeoutError),
            report._report_lock(records_directory, timeout=0.2),
        ):
            pass


def test_line_chart() -> None:
    chart = report._line_chart(
        "Loss",
        [0, 1, 2, 3, 4],
        [("training", [0.5, 0.4, np.nan, 0.3, 0.2], "#000")],
        x_label="epoch",
    )
    # The missing value splits the line in two.
    assert chart.count("<polyline") == 2

    assert "No data" in report._line_chart(
        "Loss", [0, 1], [("training", [np.nan, None], "#000")], x_label="epoch"
    )
