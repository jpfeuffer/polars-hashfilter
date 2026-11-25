"""Correctness checks for the investigation's disk-backed paths."""

import json
import os
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from benchmarks.disk_backed_uniqueness import sqlite_check


@pytest.mark.parametrize("method", ["group", "count", "sort", "sqlite"])
@pytest.mark.parametrize("rows,distinct", [(0, 1), (60, 25), (60, 60)])
@pytest.mark.skipif(sys.platform == "win32", reason="RSS probe uses Unix resource")
@pytest.mark.skipif(
    pl.__version__ != "2.1.0", reason="Disk probe is specific to Polars 2.1.0"
)
def test_disk_probe_methods(tmp_path, method, rows, distinct):
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "benchmarks/disk_backed_uniqueness.py"),
        "--rows",
        str(rows),
        "--distinct",
        str(distinct),
        "--prefix",
        "10",
        "--batch-rows",
        "20",
        "--budget",
        "1",
        "--methods",
        method,
        "--output",
        str(tmp_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "POLARS_MAX_THREADS": "1"},
    )
    record = json.loads(result.stdout)
    assert record["duplicates"] == rows - min(rows, distinct)
    assert record["distinct"] == min(rows, distinct)
    assert (tmp_path / f"{method}.stderr").is_file()


def test_sqlite_full_string_equality(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    values = ["", "a", "a\x00b", "a\x00c", "\u00e9", "e\u0301", None, "A"]
    for i, batch in enumerate([values, values[:4]]):
        pl.DataFrame({"key": batch}).write_parquet(source / f"{i}.parquet")
    duplicates, distinct = sqlite_check(source, tmp_path / "keys.sqlite", 1)
    assert distinct == 7
    assert duplicates == 4
    assert (tmp_path / "keys.sqlite").stat().st_size > 0
