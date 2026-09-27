"""
A navigable HTML report of training run records.

The report mirrors the records tree written by `qenetics.qcpg.records`:

- `<records>/index.html` lists the genomic experiments.
- `<records>/<experiment>/index.html` lists each experiment's runs.
- `<records>/<experiment>/<run>/index.html` shows a run and indexes its cells.
- `<run>/cells/<cell>/index.html` shows one cell's training and test results.
- `<run>/cells/<cell>/<split>/chr<name>.html` lists the cell's sites on one
  chromosome, and `<split>/largest-errors.html` its worst-predicted sites.

Runs are added incrementally: adding a run renders only that run's pages and
refreshes the two index pages from each run's small summary files, so earlier
runs' pages are never rebuilt.
"""

from __future__ import annotations

import html
import json
import math
import os
import shutil
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from qenetics.qcpg import records

INDEX_FILENAME: str = "index.html"
CELLS_DIRECTORY: str = "cells"
LARGEST_ERRORS_FILENAME: str = "largest-errors.html"
LOCK_FILENAME: str = ".report.lock"
LARGEST_ERRORS_QUANTITY: int = 500
SITES_PER_PAGE: int = 100
LOCK_TIMEOUT_SECONDS: float = 60.0

SPLIT_COLORS: dict[str, str] = {
    records.TRAINING_SPLIT: "#1f77b4",
    records.VALIDATION_SPLIT: "#ff7f0e",
    records.TEST_SPLIT: "#2ca02c",
}
REPORTED_SPLITS: tuple[str, ...] = (records.TRAINING_SPLIT, records.TEST_SPLIT)
MISSING: str = "—"

_STYLE: str = """
body { font-family: system-ui, sans-serif; margin: 1.5rem auto; max-width: 72rem;
       padding: 0 1rem; color: #222; }
nav { font-size: 0.9rem; margin-bottom: 1rem; }
nav a { color: #0b5cad; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.15rem; margin-top: 1.75rem; }
table { border-collapse: collapse; margin: 0.5rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; }
th { background: #f2f2f2; }
table.sortable th { cursor: pointer; }
table.sortable th[data-order="ascending"]::after { content: " \\25B2"; }
table.sortable th[data-order="descending"]::after { content: " \\25BC"; }
td.number { text-align: right; font-variant-numeric: tabular-nums; }
.status { padding: 0.1rem 0.4rem; border-radius: 0.3rem; font-size: 0.85rem; }
.status-completed { background: #d8f0d8; }
.status-running { background: #fff0c2; }
.status-failed { background: #f6d2d2; }
.charts { display: flex; flex-wrap: wrap; gap: 1.5rem; }
.muted { color: #666; }
.pager { margin: 0.5rem 0; }
.pager button { margin-right: 0.5rem; }
"""

_SORT_SCRIPT: str = """
document.querySelectorAll("table.sortable").forEach((table) => {
  table.querySelectorAll("th").forEach((header, column) => {
    header.addEventListener("click", () => {
      const ascending = header.dataset.order !== "ascending";
      table.querySelectorAll("th").forEach((other) => delete other.dataset.order);
      header.dataset.order = ascending ? "ascending" : "descending";
      const body = table.tBodies[0];
      const value = (row) => {
        const cell = row.cells[column];
        const raw = cell.dataset.value ?? cell.textContent;
        const number = parseFloat(raw);
        return raw === "" ? null : (isNaN(number) ? raw : number);
      };
      const rows = Array.from(body.rows);
      rows.sort((first, second) => {
        const a = value(first), b = value(second);
        if (a === null) return b === null ? 0 : 1;
        if (b === null) return -1;
        return (a < b ? -1 : a > b ? 1 : 0) * (ascending ? 1 : -1);
      });
      rows.forEach((row) => body.appendChild(row));
    });
  });
});
"""

_SITE_TABLE_SCRIPT: str = """
const siteData = JSON.parse(document.getElementById("site-data").textContent);
const pageSize = SITES_PER_PAGE;
let page = 0;
let sortColumn = null;
let ascending = true;
const table = document.getElementById("sites");
const pageLabel = document.getElementById("page-label");
function format(value, column) {
  if (value === null) return "\\u2014";
  return siteData.decimals[column] === null ? String(value)
    : Number(value).toFixed(siteData.decimals[column]);
}
function render() {
  const pages = Math.max(1, Math.ceil(siteData.rows.length / pageSize));
  page = Math.min(Math.max(page, 0), pages - 1);
  const body = table.tBodies[0];
  body.replaceChildren();
  for (const row of siteData.rows.slice(page * pageSize, (page + 1) * pageSize)) {
    const tableRow = body.insertRow();
    row.forEach((value, column) => {
      const cell = tableRow.insertCell();
      cell.textContent = format(value, column);
      if (typeof value === "number") cell.className = "number";
    });
  }
  pageLabel.textContent = `Page ${page + 1} of ${pages} (${siteData.rows.length} sites)`;
}
table.querySelectorAll("th").forEach((header, column) => {
  header.addEventListener("click", () => {
    ascending = sortColumn === column ? !ascending : true;
    sortColumn = column;
    table.querySelectorAll("th").forEach((other) => delete other.dataset.order);
    header.dataset.order = ascending ? "ascending" : "descending";
    siteData.rows.sort((first, second) => {
      const a = first[column], b = second[column];
      return (a < b ? -1 : a > b ? 1 : 0) * (ascending ? 1 : -1);
    });
    page = 0;
    render();
  });
});
document.getElementById("previous").addEventListener("click", () => { page -= 1; render(); });
document.getElementById("next").addEventListener("click", () => { page += 1; render(); });
render();
""".replace("SITES_PER_PAGE", str(SITES_PER_PAGE))


# --------------------------------------------------------------------------
# Writing and locking


def _write_text(filepath: Path, content: str) -> None:
    """
    Write a text file atomically, so readers never see a partial page.

    Args
    ----
    filepath: The file to write.
    content: The text to write.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temporary_filepath: Path = filepath.with_name(
        f".{filepath.name}.{os.getpid()}.tmp"
    )
    temporary_filepath.write_text(content, encoding="utf-8")
    os.replace(temporary_filepath, filepath)


@contextmanager
def _report_lock(
    records_directory: Path, timeout: float = LOCK_TIMEOUT_SECONDS
) -> Iterator[None]:
    """
    Hold the lock that serializes index refreshes across processes.

    A lock left behind by a crashed process is removed once it is older than
    the timeout.

    Args
    ----
    records_directory: The root folder of all records.
    timeout: The seconds to wait for the lock.

    Raises
    ------
    TimeoutError if the lock cannot be taken in time.
    """
    records_directory.mkdir(parents=True, exist_ok=True)
    lock_filepath: Path = records_directory / LOCK_FILENAME
    deadline: float = time.monotonic() + timeout
    while True:
        try:
            lock_descriptor: int = os.open(
                lock_filepath, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            break
        except FileExistsError:
            try:
                if time.time() - lock_filepath.stat().st_mtime > timeout:
                    lock_filepath.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for the report lock {lock_filepath}"
                ) from None
            time.sleep(0.05)

    try:
        yield
    finally:
        os.close(lock_descriptor)
        lock_filepath.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# HTML building blocks


def _escape(value: object) -> str:
    return html.escape(str(value))


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _format(value: object, decimals: int = 3) -> str:
    """
    Format a value for display, showing missing values as a dash.

    Args
    ----
    value: The value.
    decimals: The decimal places of floats.

    Returns
    -------
    The escaped display text.
    """
    if _is_missing(value):
        return MISSING
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return _escape(value)


def _cell(value: object, decimals: int = 3, *, link: str | None = None) -> str:
    """
    Render a table cell with a sortable value.

    Args
    ----
    value: The value.
    decimals: The decimal places of floats.
    link: The relative link to wrap the value in, if any.

    Returns
    -------
    The `<td>` element.
    """
    sort_value: str = "" if _is_missing(value) else _escape(value)
    is_number: bool = isinstance(value, int | float) and not isinstance(
        value, bool
    )
    content: str = _format(value, decimals)
    if link is not None:
        content = f'<a href="{_escape(link)}">{content}</a>'
    css_class: str = ' class="number"' if is_number else ""
    return f'<td{css_class} data-value="{sort_value}">{content}</td>'


def _table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], sortable: bool = True
) -> str:
    """
    Render a table from rendered cells.

    Args
    ----
    headers: The column headers.
    rows: The rows of rendered `<td>` cells.
    sortable: Allow sorting by clicking a header.

    Returns
    -------
    The `<table>` element.
    """
    css_class: str = ' class="sortable"' if sortable else ""
    header_html: str = "".join(
        f"<th>{_escape(header)}</th>" for header in headers
    )
    body_html: str = "\n".join(f"<tr>{''.join(row)}</tr>" for row in rows)
    return (
        f"<table{css_class}><thead><tr>{header_html}</tr></thead>"
        f"<tbody>\n{body_html}\n</tbody></table>"
    )


def _key_value_table(items: Sequence[tuple[str, object]]) -> str:
    rows: list[list[str]] = [
        [f"<th>{_escape(key)}</th>", _cell(value)] for key, value in items
    ]
    return (
        "<table><tbody>"
        + "".join(f"<tr>{''.join(row)}</tr>" for row in rows)
        + "</tbody></table>"
    )


def _status_badge(status: str) -> str:
    return f'<span class="status status-{_escape(status)}">{_escape(status)}</span>'


def _page(
    title: str,
    breadcrumbs: Sequence[tuple[str, str | None]],
    body: str,
    *,
    script: str = _SORT_SCRIPT,
) -> str:
    """
    Render a complete HTML page.

    Args
    ----
    title: The page title.
    breadcrumbs: The navigation trail of (label, relative link) pairs, the
        last usually without a link.
    body: The page's body content.
    script: The inline script to run.

    Returns
    -------
    The page.
    """
    trail: str = " › ".join(
        _escape(label)
        if link is None
        else f'<a href="{_escape(link)}">{_escape(label)}</a>'
        for label, link in breadcrumbs
    )
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{_escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><nav>{trail}</nav>\n{body}\n<script>{script}</script>"
        "</body></html>\n"
    )


# --------------------------------------------------------------------------
# Charts


def _line_chart(
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float | None], str]],
    *,
    x_label: str,
    y_range: tuple[float, float] | None = None,
    diagonal: bool = False,
    width: int = 480,
    height: int = 260,
) -> str:
    """
    Render an SVG line chart.

    Args
    ----
    title: The chart title.
    x_values: The shared x coordinates.
    series: The (name, y values, color) of each line. Missing y values break
        the line.
    x_label: The x axis label.
    y_range: The fixed y axis range, or None to fit the data.
    diagonal: Draw the y = x reference line, e.g. for ROC curves.
    width: The chart width in pixels.
    height: The chart height in pixels.

    Returns
    -------
    The chart's `<figure>` element, or a note if there is nothing to plot.
    """
    finite_values: list[float] = [
        float(value)
        for _, values, _ in series
        for value in values
        if not _is_missing(value)
    ]
    if not finite_values or len(x_values) == 0:
        return (
            f"<figure><figcaption>{_escape(title)}</figcaption>"
            f'<p class="muted">No data.</p></figure>'
        )

    left, right, top, bottom = 48, 12, 24, 36
    plot_width: int = width - left - right
    plot_height: int = height - top - bottom
    x_minimum, x_maximum = float(min(x_values)), float(max(x_values))
    if x_maximum == x_minimum:
        x_maximum = x_minimum + 1.0
    y_minimum, y_maximum = y_range or (min(finite_values), max(finite_values))
    if y_maximum == y_minimum:
        y_minimum, y_maximum = y_minimum - 0.5, y_maximum + 0.5

    def x_pixel(x: float) -> float:
        return left + (x - x_minimum) / (x_maximum - x_minimum) * plot_width

    def y_pixel(y: float) -> float:
        return top + (y_maximum - y) / (y_maximum - y_minimum) * plot_height

    elements: list[str] = [
        (
            f'<rect x="{left}" y="{top}" width="{plot_width}" '
            f'height="{plot_height}" fill="none" stroke="#999"/>'
        )
    ]
    for tick in np.linspace(y_minimum, y_maximum, 5):
        y: float = y_pixel(tick)
        elements.append(
            f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_width}" '
            f'y2="{y:.1f}" stroke="#eee"/>'
            f'<text x="{left - 6}" y="{y + 4:.1f}" font-size="10" '
            f'text-anchor="end">{tick:.3g}</text>'
        )
    for tick in np.linspace(x_minimum, x_maximum, min(6, len(x_values))):
        x: float = x_pixel(tick)
        elements.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 14}" font-size="10" '
            f'text-anchor="middle">{tick:.3g}</text>'
        )
    elements.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 4}" '
        f'font-size="11" text-anchor="middle">{_escape(x_label)}</text>'
    )
    if diagonal:
        elements.append(
            f'<line x1="{x_pixel(0):.1f}" y1="{y_pixel(0):.1f}" '
            f'x2="{x_pixel(1):.1f}" y2="{y_pixel(1):.1f}" stroke="#bbb" '
            'stroke-dasharray="4 3"/>'
        )

    for series_index, (name, y_values, color) in enumerate(series):
        segment: list[str] = []
        segments: list[list[str]] = []
        for x_value, y_value in zip(x_values, y_values, strict=False):
            if _is_missing(y_value):
                if segment:
                    segments.append(segment)
                segment = []
                continue
            segment.append(
                f"{x_pixel(float(x_value)):.1f},{y_pixel(float(y_value)):.1f}"
            )
        if segment:
            segments.append(segment)
        for points in segments:
            if len(points) == 1:
                x_text, y_text = points[0].split(",")
                elements.append(
                    f'<circle cx="{x_text}" cy="{y_text}" r="3" '
                    f'fill="{color}"/>'
                )
            else:
                elements.append(
                    f'<polyline points="{" ".join(points)}" fill="none" '
                    f'stroke="{color}" stroke-width="2"/>'
                )
        legend_y: int = top + 12 + 14 * series_index
        elements.append(
            f'<rect x="{left + 8}" y="{legend_y - 8}" width="10" height="10" '
            f'fill="{color}"/><text x="{left + 22}" y="{legend_y + 1}" '
            f'font-size="11">{_escape(name)}</text>'
        )

    return (
        f"<figure><figcaption>{_escape(title)}</figcaption>"
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="{_escape(title)}">{"".join(elements)}</svg></figure>'
    )


# --------------------------------------------------------------------------
# Run summaries


def _cell_directories(cell_names: Sequence[str]) -> list[str]:
    """
    Choose a unique folder name for each cell.

    Args
    ----
    cell_names: The cell names.

    Returns
    -------
    The folder name of each cell, in order.
    """
    directories: list[str] = []
    used: set[str] = set()
    for cell_name in cell_names:
        directory: str = records.safe_name(cell_name)
        candidate: str = directory
        suffix: int = 1
        while candidate.lower() in used:
            candidate = f"{directory}-{suffix}"
            suffix += 1
        used.add(candidate.lower())
        directories.append(candidate)

    return directories


def _split_metrics(run: dict[str, Any], split: str) -> dict[str, Any]:
    return run["metrics"].get(split, {})


def _mean_cell_auc(run: dict[str, Any], split: str) -> float:
    aucs: list[float] = [
        cell["auc"]
        for cell in _split_metrics(run, split).get("cells", [])
        if not _is_missing(cell["auc"])
    ]
    return float(np.mean(aucs)) if aucs else float("nan")


def _pooled_auc(run: dict[str, Any], split: str) -> float:
    return _split_metrics(run, split).get("pooled", {}).get("auc", float("nan"))


def _read_summary(run_directory: Path) -> dict[str, Any] | None:
    """
    Read a run's summary files, skipping runs whose record is unreadable.

    Args
    ----
    run_directory: The run's folder.

    Returns
    -------
    The run summary, or None if it cannot be read.
    """
    try:
        return records.read_run(run_directory)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# Index pages


def refresh_indexes(records_directory: Path) -> None:
    """
    Rewrite the top index and every experiment index from the run summaries.

    Only each run's `run.json`, `epochs.json` and `metrics.json` are read.

    Args
    ----
    records_directory: The root folder of all records.
    """
    with _report_lock(records_directory):
        runs_by_experiment: dict[Path, list[tuple[Path, dict[str, Any]]]] = {}
        for run_directory in records.find_run_directories(records_directory):
            summary: dict[str, Any] | None = _read_summary(run_directory)
            if summary is not None:
                runs_by_experiment.setdefault(run_directory.parent, []).append(
                    (run_directory, summary)
                )

        experiment_rows: list[list[str]] = []
        for experiment_directory, runs in sorted(runs_by_experiment.items()):
            runs.sort(key=lambda run: run[0].name, reverse=True)
            experiment_name: str = runs[0][1].get(
                "experiment_name", experiment_directory.name
            )
            _write_experiment_index(experiment_directory, experiment_name, runs)
            latest_directory, latest_run = runs[0]
            experiment_rows.append(
                [
                    _cell(
                        experiment_name,
                        link=f"{quote(experiment_directory.name)}/{INDEX_FILENAME}",
                    ),
                    _cell(len(runs)),
                    _cell(
                        sum(
                            run["status"] == records.STATUS_COMPLETED
                            for _, run in runs
                        )
                    ),
                    _cell(
                        latest_directory.name,
                        link=(
                            f"{quote(experiment_directory.name)}/"
                            f"{quote(latest_directory.name)}/{INDEX_FILENAME}"
                        ),
                    ),
                    f"<td>{_status_badge(latest_run['status'])}</td>",
                    _cell(_pooled_auc(latest_run, records.TEST_SPLIT)),
                ]
            )

        body: str = "<h1>Training records</h1>"
        if experiment_rows:
            body += _table(
                [
                    "Genomic experiment",
                    "Runs",
                    "Completed",
                    "Latest run",
                    "Latest status",
                    "Latest test AUC",
                ],
                experiment_rows,
            )
        else:
            body += '<p class="muted">No training runs recorded yet.</p>'
        _write_text(
            records_directory / INDEX_FILENAME,
            _page("Training records", [("All experiments", None)], body),
        )


def _write_experiment_index(
    experiment_directory: Path,
    experiment_name: str,
    runs: Sequence[tuple[Path, dict[str, Any]]],
) -> None:
    """
    Write the index of one experiment's runs, newest first.

    Args
    ----
    experiment_directory: The experiment's folder.
    experiment_name: The experiment's name.
    runs: The (folder, summary) of each run, newest first.
    """
    rows: list[list[str]] = []
    for run_directory, run in runs:
        parameters: dict[str, Any] = run.get("parameters", {})
        rows.append(
            [
                _cell(
                    run_directory.name,
                    link=f"{quote(run_directory.name)}/{INDEX_FILENAME}",
                ),
                _cell(run.get("started")),
                (
                    f'<td data-value="{_escape(run["status"])}">'
                    f"{_status_badge(run['status'])}</td>"
                ),
                _cell(parameters.get("encoding")),
                _cell(parameters.get("layer_quantity")),
                _cell(parameters.get("epochs")),
                _cell(parameters.get("learning_rate"), 6),
                _cell(len(run.get("cell_names", []))),
                _cell(_pooled_auc(run, records.TEST_SPLIT)),
                _cell(_mean_cell_auc(run, records.TEST_SPLIT)),
            ]
        )

    body: str = f"<h1>{_escape(experiment_name)}</h1>" + _table(
        [
            "Run",
            "Started",
            "Status",
            "Encoding",
            "Layers",
            "Epochs",
            "Learning rate",
            "Cells",
            "Test AUC (pooled)",
            "Test AUC (mean of cells)",
        ],
        rows,
    )
    _write_text(
        experiment_directory / INDEX_FILENAME,
        _page(
            experiment_name,
            [
                ("All experiments", f"../{INDEX_FILENAME}"),
                (experiment_name, None),
            ],
            body,
        ),
    )


# --------------------------------------------------------------------------
# Run pages


def _run_breadcrumbs(
    run: dict[str, Any], depth: int
) -> list[tuple[str, str | None]]:
    """
    Build the navigation trail from a page `depth` folders below the run.

    Args
    ----
    run: The run summary.
    depth: How many folders below the run folder the page is.

    Returns
    -------
    The trail up to and including the run.
    """
    up: str = "../" * depth
    return [
        ("All experiments", f"{up}../../{INDEX_FILENAME}"),
        (run["experiment_name"], f"{up}../{INDEX_FILENAME}"),
        (run["run_id"], f"{up}{INDEX_FILENAME}" if depth else None),
    ]


def _summary_section(run: dict[str, Any]) -> str:
    """
    Render a run's status, chromosome splits and parameters.

    Args
    ----
    run: The run summary.

    Returns
    -------
    The HTML section.
    """
    chromosomes: dict[str, list[str]] = run.get("chromosomes", {})
    parameters: dict[str, Any] = run.get("parameters", {})
    return (
        f"<h1>Run {_escape(run['run_id'])} {_status_badge(run['status'])}</h1>"
        + _key_value_table(
            [
                ("Genomic experiment", run["experiment_name"]),
                ("Started", run.get("started")),
                ("Finished", run.get("finished")),
                ("Cells", len(run.get("cell_names", []))),
            ]
            + [
                (
                    f"{split.capitalize()} chromosomes",
                    ", ".join(split_chromosomes),
                )
                for split, split_chromosomes in chromosomes.items()
            ]
        )
        + "<details><summary>Training parameters</summary>"
        + _key_value_table(sorted(parameters.items()))
        + "</details>"
    )


def _epoch_charts(run: dict[str, Any], cell_index: int | None = None) -> str:
    """
    Render the per-epoch loss and validation AUC charts.

    Args
    ----
    run: The run summary.
    cell_index: The cell whose validation AUC to plot, or None for the run's
        losses and pooled validation AUC.

    Returns
    -------
    The charts.
    """
    epochs: list[dict[str, Any]] = run.get("epochs", [])
    x_values: list[int] = [epoch["epoch"] for epoch in epochs]
    if cell_index is not None:
        return _line_chart(
            "Validation AUC per epoch",
            x_values,
            [
                (
                    "validation",
                    [
                        epoch["cell_validation_aucs"][cell_index]
                        for epoch in epochs
                    ],
                    SPLIT_COLORS[records.VALIDATION_SPLIT],
                )
            ],
            x_label="epoch",
            y_range=(0.0, 1.0),
        )

    return (
        '<div class="charts">'
        + _line_chart(
            "Loss per epoch",
            x_values,
            [
                (
                    "training",
                    [epoch["training_loss"] for epoch in epochs],
                    SPLIT_COLORS[records.TRAINING_SPLIT],
                ),
                (
                    "validation",
                    [epoch["validation_loss"] for epoch in epochs],
                    SPLIT_COLORS[records.VALIDATION_SPLIT],
                ),
            ],
            x_label="epoch",
        )
        + _line_chart(
            "Validation AUC per epoch (pooled)",
            x_values,
            [
                (
                    "validation",
                    [epoch["validation_auc"] for epoch in epochs],
                    SPLIT_COLORS[records.VALIDATION_SPLIT],
                )
            ],
            x_label="epoch",
            y_range=(0.0, 1.0),
        )
        + "</div>"
    )


def _final_metrics_table(run: dict[str, Any]) -> str:
    """
    Render the final pooled metrics of the training and test splits.

    Args
    ----
    run: The run summary.

    Returns
    -------
    The table.
    """
    rows: list[list[str]] = []
    for split in REPORTED_SPLITS:
        split_metrics: dict[str, Any] = _split_metrics(run, split)
        if not split_metrics:
            continue
        pooled: dict[str, Any] = split_metrics["pooled"]
        rows.append(
            [
                _cell(split),
                _cell(split_metrics["loss"]),
                _cell(pooled["observed_sites"]),
                _cell(pooled["auc"]),
                _cell(_mean_cell_auc(run, split)),
                _cell(pooled["accuracy"]),
                _cell(pooled["f1"]),
            ]
        )

    if not rows:
        return '<p class="muted">No final results recorded.</p>'

    return _table(
        [
            "Split",
            "Loss",
            "Observed sites",
            "AUC (pooled)",
            "AUC (mean of cells)",
            "Accuracy",
            "F1",
        ],
        rows,
        sortable=False,
    )


def _render_run_index(
    run_directory: Path, run: dict[str, Any], cell_directories: Sequence[str]
) -> None:
    """
    Write a run's page, including its cell index once the run has finished.

    Args
    ----
    run_directory: The run's folder.
    run: The run summary.
    cell_directories: The folder name of each cell.
    """
    body: str = _summary_section(run)
    if run["status"] == records.STATUS_RUNNING:
        body += (
            '<p class="muted">This run is still training. Its results appear '
            "here when it finishes.</p>"
        )
    else:
        body += "<h2>Training</h2>" + _epoch_charts(run)
        body += "<h2>Final results</h2>" + _final_metrics_table(run)
        if run["status"] == records.STATUS_FAILED:
            body += (
                '<p class="muted">This run failed; results may be incomplete.'
                "</p>"
            )

        cell_rows: list[list[str]] = []
        epochs: list[dict[str, Any]] = run.get("epochs", [])
        for cell_index, (cell_name, cell_directory) in enumerate(
            zip(run["cell_names"], cell_directories, strict=True)
        ):
            row: list[str] = [
                _cell(
                    cell_name,
                    link=(
                        f"{CELLS_DIRECTORY}/{quote(cell_directory)}/"
                        f"{INDEX_FILENAME}"
                    ),
                )
            ]
            for split in REPORTED_SPLITS:
                cells: list[dict[str, Any]] = _split_metrics(run, split).get(
                    "cells", []
                )
                row.append(
                    _cell(
                        cells[cell_index]["observed_sites"] if cells else None
                    )
                )
            for split in REPORTED_SPLITS:
                cells = _split_metrics(run, split).get("cells", [])
                row.append(_cell(cells[cell_index]["auc"] if cells else None))
            test_cells: list[dict[str, Any]] = _split_metrics(
                run, records.TEST_SPLIT
            ).get("cells", [])
            row += [
                _cell(
                    test_cells[cell_index]["accuracy"] if test_cells else None
                ),
                _cell(test_cells[cell_index]["f1"] if test_cells else None),
                _cell(
                    epochs[-1]["cell_validation_aucs"][cell_index]
                    if epochs
                    else None
                ),
            ]
            cell_rows.append(row)

        body += "<h2>Cells</h2>" + _table(
            [
                "Cell",
                "Training sites",
                "Test sites",
                "Training AUC",
                "Test AUC",
                "Test accuracy",
                "Test F1",
                "Last validation AUC",
            ],
            cell_rows,
        )

    _write_text(
        run_directory / INDEX_FILENAME,
        _page(
            f"{run['experiment_name']} – {run['run_id']}",
            _run_breadcrumbs(run, 0),
            body,
        ),
    )


def _confusion_matrix_table(confusion_matrix: dict[str, int]) -> str:
    return (
        "<table><thead><tr><th></th><th>Predicted methylated</th>"
        "<th>Predicted unmethylated</th></tr></thead><tbody>"
        f"<tr><th>Methylated</th>{_cell(confusion_matrix['true_positives'])}"
        f"{_cell(confusion_matrix['false_negatives'])}</tr>"
        f"<tr><th>Unmethylated</th>{_cell(confusion_matrix['false_positives'])}"
        f"{_cell(confusion_matrix['true_negatives'])}</tr></tbody></table>"
    )


def _render_cell_page(
    run_directory: Path,
    run: dict[str, Any],
    cell_index: int,
    cell_directory: str,
    site_links: dict[str, list[tuple[str, str, int]]],
) -> None:
    """
    Write one cell's page.

    Args
    ----
    run_directory: The run's folder.
    run: The run summary.
    cell_index: The cell's label column.
    cell_directory: The cell's folder name.
    site_links: For each split, the (label, relative link, site count) of
        each of the cell's site pages.
    """
    cell_name: str = run["cell_names"][cell_index]
    metric_rows: list[list[str]] = []
    roc_series: list[tuple[str, list[float], list[float], str]] = []
    confusion_matrices: str = ""
    for split in REPORTED_SPLITS:
        cells: list[dict[str, Any]] = _split_metrics(run, split).get(
            "cells", []
        )
        if not cells:
            continue
        cell: dict[str, Any] = cells[cell_index]
        metric_rows.append(
            [
                _cell(split),
                _cell(cell["observed_sites"]),
                _cell(cell["methylated_sites"]),
                _cell(cell["auc"]),
                _cell(cell["accuracy"]),
                _cell(cell["true_positive_rate"]),
                _cell(cell["false_positive_rate"]),
                _cell(cell["f1"]),
            ]
        )
        roc_series.append(
            (
                split,
                cell["roc_false_positive_rates"],
                cell["roc_true_positive_rates"],
                SPLIT_COLORS[split],
            )
        )
        confusion_matrices += (
            f"<figure><figcaption>{_escape(split.capitalize())}</figcaption>"
            f"{_confusion_matrix_table(cell['confusion_matrix'])}</figure>"
        )

    body: str = f"<h1>{_escape(cell_name)}</h1>"
    if metric_rows:
        body += "<h2>Results</h2>" + _table(
            [
                "Split",
                "Observed sites",
                "Methylated sites",
                "AUC",
                "Accuracy",
                "True positive rate",
                "False positive rate",
                "F1",
            ],
            metric_rows,
            sortable=False,
        )
        body += f'<div class="charts">{confusion_matrices}</div>'
        roc_charts: str = "".join(
            _line_chart(
                f"{split.capitalize()} ROC curve",
                false_positive_rates,
                [("ROC", true_positive_rates, color)],
                x_label="false positive rate",
                y_range=(0.0, 1.0),
                diagonal=True,
                width=320,
                height=300,
            )
            for split, false_positive_rates, true_positive_rates, color in (
                roc_series
            )
        )
        body += (
            '<div class="charts">'
            + roc_charts
            + _epoch_charts(run, cell_index)
            + "</div>"
        )
    else:
        body += '<p class="muted">No final results recorded.</p>'

    for split in REPORTED_SPLITS:
        links: list[tuple[str, str, int]] = site_links.get(split, [])
        if not links:
            continue
        items: str = "".join(
            f'<li><a href="{_escape(link)}">{_escape(label)}</a> '
            f'<span class="muted">({count} sites)</span></li>'
            for label, link, count in links
        )
        body += f"<h2>{_escape(split.capitalize())} sites</h2><ul>{items}</ul>"

    _write_text(
        run_directory / CELLS_DIRECTORY / cell_directory / INDEX_FILENAME,
        _page(
            f"{cell_name} – {run['run_id']}",
            [*_run_breadcrumbs(run, 2), (cell_name, None)],
            body,
        ),
    )


def _site_page(
    title: str,
    breadcrumbs: Sequence[tuple[str, str | None]],
    headers: Sequence[str],
    decimals: Sequence[int | None],
    rows: list[list[Any]],
) -> str:
    """
    Render a page with a paged, sortable table of sites.

    Args
    ----
    title: The page title.
    breadcrumbs: The navigation trail.
    headers: The column headers.
    decimals: The decimal places of each column, or None for exact values.
    rows: The table rows.

    Returns
    -------
    The page.
    """
    site_data: str = json.dumps(
        {"decimals": list(decimals), "rows": rows}, separators=(",", ":")
    ).replace("</", "<\\/")
    header_html: str = "".join(
        f"<th>{_escape(header)}</th>" for header in headers
    )
    body: str = (
        f"<h1>{_escape(title)}</h1>"
        '<div class="pager"><button id="previous">Previous</button>'
        '<button id="next">Next</button><span id="page-label"></span></div>'
        f'<table id="sites" class="sortable"><thead><tr>{header_html}</tr>'
        "</thead><tbody></tbody></table>"
        f'<script type="application/json" id="site-data">{site_data}</script>'
    )
    return _page(title, breadcrumbs, body, script=_SITE_TABLE_SCRIPT)


def _render_site_pages(
    run_directory: Path,
    run: dict[str, Any],
    cell_directories: Sequence[str],
) -> dict[int, dict[str, list[tuple[str, str, int]]]]:
    """
    Write every cell's site pages for the training and test splits.

    Args
    ----
    run_directory: The run's folder.
    run: The run summary.
    cell_directories: The folder name of each cell.

    Returns
    -------
    For each cell, the (label, link from the cell page, site count) of its
    site pages in each split.
    """
    site_links: dict[int, dict[str, list[tuple[str, str, int]]]] = {
        cell_index: {} for cell_index in range(len(cell_directories))
    }
    for split in REPORTED_SPLITS:
        outputs: records.SplitOutputs | None = records.read_split_outputs(
            run_directory, split
        )
        if outputs is None:
            continue

        labels: NDArray[np.float32] = outputs.labels.reshape(
            len(outputs.labels), -1
        )
        predictions: NDArray[np.float32] = outputs.predictions.reshape(
            len(outputs.predictions), -1
        )
        for cell_index, cell_directory in enumerate(cell_directories):
            cell_name: str = run["cell_names"][cell_index]
            split_directory: Path = (
                run_directory / CELLS_DIRECTORY / cell_directory / split
            )
            breadcrumbs: list[tuple[str, str | None]] = [
                *_run_breadcrumbs(run, 3),
                (cell_name, f"../{INDEX_FILENAME}"),
            ]
            is_observed: NDArray[np.bool_] = ~np.isnan(labels[:, cell_index])
            cell_labels: NDArray[np.float32] = labels[is_observed, cell_index]
            cell_predictions: NDArray[np.float32] = predictions[
                is_observed, cell_index
            ]
            cell_positions: NDArray[np.int64] = outputs.positions[is_observed]
            cell_chromosomes: NDArray[np.int32] = outputs.chromosome_indices[
                is_observed
            ]
            errors: NDArray[np.float32] = np.abs(cell_predictions - cell_labels)
            links: list[tuple[str, str, int]] = []

            if len(errors):
                worst: NDArray[np.int64] = np.argsort(-errors, kind="stable")[
                    :LARGEST_ERRORS_QUANTITY
                ]
                _write_text(
                    split_directory / LARGEST_ERRORS_FILENAME,
                    _site_page(
                        f"{cell_name}: largest {split} errors",
                        [*breadcrumbs, (f"{split} largest errors", None)],
                        [
                            "Chromosome",
                            "Position",
                            "Label",
                            "Prediction",
                            "Error",
                        ],
                        [None, None, None, 4, 4],
                        [
                            [
                                outputs.chromosome_names[cell_chromosomes[row]],
                                int(cell_positions[row]),
                                int(cell_labels[row]),
                                round(float(cell_predictions[row]), 4),
                                round(float(errors[row]), 4),
                            ]
                            for row in worst
                        ],
                    ),
                )
                links.append(
                    (
                        f"Largest {len(worst)} errors",
                        f"{split}/{LARGEST_ERRORS_FILENAME}",
                        len(worst),
                    )
                )

            for chromosome_index, chromosome in enumerate(
                outputs.chromosome_names
            ):
                on_chromosome: NDArray[np.bool_] = (
                    cell_chromosomes == chromosome_index
                )
                if not on_chromosome.any():
                    continue
                order: NDArray[np.int64] = np.argsort(
                    cell_positions[on_chromosome], kind="stable"
                )
                filename: str = f"chr{records.safe_name(chromosome)}.html"
                chromosome_rows: list[list[Any]] = [
                    [
                        int(position),
                        int(label),
                        round(float(prediction), 4),
                        round(float(error), 4),
                    ]
                    for position, label, prediction, error in zip(
                        cell_positions[on_chromosome][order],
                        cell_labels[on_chromosome][order],
                        cell_predictions[on_chromosome][order],
                        errors[on_chromosome][order],
                        strict=True,
                    )
                ]
                _write_text(
                    split_directory / filename,
                    _site_page(
                        f"{cell_name}: {split} sites on chromosome {chromosome}",
                        [
                            *breadcrumbs,
                            (f"{split} chromosome {chromosome}", None),
                        ],
                        ["Position", "Label", "Prediction", "Error"],
                        [None, None, 4, 4],
                        chromosome_rows,
                    ),
                )
                links.append(
                    (
                        f"Chromosome {chromosome}",
                        f"{split}/{quote(filename)}",
                        len(chromosome_rows),
                    )
                )

            site_links[cell_index][split] = links

    return site_links


def render_run(run_directory: Path) -> None:
    """
    Write all of a run's report pages inside its own folder.

    A running run gets only its summary page; a finished run also gets its
    cell and site pages. Pages of other runs are never touched.

    Args
    ----
    run_directory: The run's folder.
    """
    run: dict[str, Any] = records.read_run(run_directory)
    cell_directories: list[str] = _cell_directories(run["cell_names"])
    if run["status"] != records.STATUS_RUNNING:
        shutil.rmtree(run_directory / CELLS_DIRECTORY, ignore_errors=True)
        site_links = _render_site_pages(run_directory, run, cell_directories)
        for cell_index, cell_directory in enumerate(cell_directories):
            _render_cell_page(
                run_directory,
                run,
                cell_index,
                cell_directory,
                site_links[cell_index],
            )

    # Written last, so its time marks when the run's pages were complete.
    _render_run_index(run_directory, run, cell_directories)


def add_run(records_directory: Path, run_directory: Path) -> None:
    """
    Add or refresh one run in the report.

    Renders the run's pages and refreshes the index pages; other runs' pages
    are left as they are.

    Args
    ----
    records_directory: The root folder of all records.
    run_directory: The run's folder.
    """
    render_run(run_directory)
    refresh_indexes(records_directory)


def is_run_stale(run_directory: Path) -> bool:
    """
    Check whether a run's report pages are missing or older than its record.

    Args
    ----
    run_directory: The run's folder.

    Returns
    -------
    True if the run's pages need rendering.
    """
    page: Path = run_directory / INDEX_FILENAME
    if not page.exists():
        return True

    record_times: list[float] = [
        filepath.stat().st_mtime
        for filename in [
            records.RUN_FILENAME,
            records.EPOCHS_FILENAME,
            records.METRICS_FILENAME,
            records.OUTPUTS_FILENAME,
        ]
        if (filepath := run_directory / filename).exists()
    ]
    return page.stat().st_mtime < max(record_times, default=0.0)


def build_report(
    records_directory: Path,
    *,
    run_directory: Path | None = None,
    rebuild: bool = False,
) -> list[Path]:
    """
    Bring the report up to date with the records.

    Args
    ----
    records_directory: The root folder of all records.
    run_directory: A single run to add or refresh. Otherwise, every run whose
        pages are missing or stale is rendered.
    rebuild: Render every run's pages, even if up to date.

    Returns
    -------
    The runs whose pages were rendered.
    """
    if run_directory is not None:
        run_directories: list[Path] = [run_directory]
    else:
        run_directories = [
            directory
            for directory in records.find_run_directories(records_directory)
            if rebuild or is_run_stale(directory)
        ]

    for directory in run_directories:
        render_run(directory)

    refresh_indexes(records_directory)
    return run_directories
