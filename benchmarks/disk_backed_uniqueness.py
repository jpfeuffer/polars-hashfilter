"""Probe exact disk-backed checks in fresh processes on synthetic Parquet."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time


METHODS = ("group", "count", "sort", "sqlite")


def directory_bytes(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except FileNotFoundError:
            pass  # Spill files can disappear between enumeration and stat.
    return total


def generate(
    path: Path, rows: int, distinct: int, prefix: int, batch_rows: int
) -> None:
    import polars as pl

    path.mkdir()
    for start in range(0, rows, batch_rows):
        pl.DataFrame(
            {"i": pl.arange(start, min(start + batch_rows, rows), eager=True)}
        ).select(
            (pl.lit("x" * prefix) + (pl.col("i") % distinct).cast(pl.String)).alias(
                "key"
            )
        ).write_parquet(path / f"{start:012}.parquet", row_group_size=20_000)
    pl.DataFrame({"key": pl.Series([None, None], dtype=pl.String)}).write_parquet(
        path / "nulls.parquet"
    )


def sqlite_check(path: Path, database: Path, cache_mb: int) -> tuple[int, int]:
    import polars as pl

    rows = 0
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(f"PRAGMA cache_size = {-cache_mb * 1024}")
        connection.execute("PRAGMA mmap_size = 0")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute(
            "CREATE TABLE seen (key TEXT COLLATE BINARY PRIMARY KEY) WITHOUT ROWID"
        )
        for file in sorted(path.glob("*.parquet")):
            # Fixture file sizes are bounded by --batch-rows, not total input size.
            series = pl.read_parquet(file, columns=["key"])["key"]
            rows += len(series) - series.null_count()
            connection.executemany(
                "INSERT OR IGNORE INTO seen(key) VALUES (?)",
                ((value,) for value in series if value is not None),
            )
            connection.commit()
        distinct = connection.total_changes
    return rows - distinct, distinct


def worker(args: argparse.Namespace) -> None:
    import resource

    import polars as pl

    if pl.__version__ != "2.1.0":
        raise RuntimeError("This probe is pinned to Polars 2.1.0")
    stop = threading.Event()
    disk_peak = 0

    def observe() -> None:
        nonlocal disk_peak
        while not stop.is_set():
            disk_peak = max(disk_peak, directory_bytes(args.spill))
            stop.wait(0.005)

    monitor = threading.Thread(target=observe)
    monitor.start()
    started = time.perf_counter()
    try:
        lf = pl.scan_parquet(args.path / "*.parquet").drop_nulls("key")
        if args.worker == "sqlite":
            duplicates, distinct = sqlite_check(
                args.path, args.spill / "keys.sqlite", args.budget
            )
        elif args.worker == "group":
            result = (
                lf.group_by("key")
                .len()
                .select(
                    (pl.col("len").cast(pl.UInt64) - 1).sum().alias("duplicates"),
                    pl.len().alias("distinct"),
                )
                .collect(engine="streaming")
            )
            duplicates, distinct = result.row(0)
        elif args.worker == "count":
            result = lf.select(
                pl.len().cast(pl.UInt64).alias("rows"),
                pl.col("key").n_unique().alias("distinct"),
            ).collect(engine="streaming")
            rows, distinct = result.row(0)
            duplicates = rows - distinct
        else:
            duplicates = (
                lf.sort("key")
                .select((pl.col("key") == pl.col("key").shift(1)).sum())
                .collect(engine="streaming")
                .item()
            )
            distinct = args.rows - duplicates
    finally:
        stop.set()
        monitor.join()
    disk_peak = max(disk_peak, directory_bytes(args.spill))
    expected_distinct = min(args.rows, args.distinct)
    assert distinct == expected_distinct, (distinct, expected_distinct)
    assert duplicates == args.rows - expected_distinct
    scale = 1024**2 if sys.platform == "darwin" else 1024
    print(
        json.dumps(
            {
                "method": args.worker,
                "threads": pl.thread_pool_size(),
                "budget_mb": args.budget,
                "rows": args.rows,
                "prefix": args.prefix,
                "batch_rows": args.batch_rows,
                "duplicates": duplicates,
                "distinct": distinct,
                "seconds": time.perf_counter() - started,
                "peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale,
                "disk_peak_mib": disk_peak / 1024**2,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--distinct", type=int)
    parser.add_argument("--prefix", type=int, default=4000)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=20_000)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--worker", choices=METHODS)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--spill", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.distinct is None:
        args.distinct = args.rows
    if args.rows < 0 or args.distinct <= 0 or args.budget <= 0 or args.batch_rows <= 0:
        parser.error("rows must be nonnegative; distinct, budget and batches positive")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hashfilter-disk-probe-") as directory:
        root = Path(directory)
        source = root / "input"
        generate(source, args.rows, args.distinct, args.prefix, args.batch_rows)
        for method in args.methods:
            spill = root / method
            spill.mkdir()
            command = [
                sys.executable,
                __file__,
                "--worker",
                method,
                "--rows",
                str(args.rows),
                "--distinct",
                str(args.distinct),
                "--prefix",
                str(args.prefix),
                "--batch-rows",
                str(args.batch_rows),
                "--budget",
                str(args.budget),
                "--path",
                str(source),
                "--spill",
                str(spill),
                "--output",
                str(args.output),
            ]
            result = subprocess.run(
                command,
                env={
                    **os.environ,
                    "POLARS_MAX_THREADS": str(args.threads),
                    "POLARS_OOC_MEMORY_BUDGET_MB": str(args.budget),
                    "POLARS_OOC_SPILL_DIR": str(spill),
                    "POLARS_OOC_LOG_METRICS": "1",
                    "POLARS_VERBOSE": "1",
                },
                capture_output=True,
                text=True,
            )
            (args.output / f"{method}.stderr").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"{method} exited {result.returncode}; see {args.output}"
                )
            print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
