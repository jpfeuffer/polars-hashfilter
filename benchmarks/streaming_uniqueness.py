"""Fresh-process benchmarks; only synthetic local Parquet files are used."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


METHODS = ("eager", "count", "first_distinct", "plugin")


def generate(path: Path, prefix_length: int, rows: int) -> None:
    import polars as pl

    pl.DataFrame({"i": pl.arange(0, rows, eager=True)}).select(
        (pl.lit("x" * prefix_length) + pl.col("i").cast(pl.String)).alias("key")
    ).write_parquet(path, row_group_size=20_000)


def peak_mib() -> float:
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def worker(path: Path, method: str, rows: int) -> None:
    import polars as pl
    from polars_hashfilter import StringHashSet, update_bool

    assert pl.__version__ == "2.1.0"
    baseline = peak_mib()
    start = time.perf_counter()
    lf = pl.scan_parquet(path)
    if method == "eager":
        duplicate = lf.collect()["key"].is_duplicated().any()
    elif method == "count":
        duplicate = (
            lf.select(pl.col("key").n_unique() != pl.len())
            .collect(engine="streaming")
            .item()
        )
    elif method == "first_distinct":
        duplicate = (
            lf.select((~pl.col("key").is_first_distinct()).any())
            .collect(engine="streaming")
            .item()
        )
    else:
        seen = StringHashSet()
        duplicate = (
            lf.select((~update_bool("key", seen)).any())
            .collect(engine="streaming")
            .item()
        )
        assert len(seen) == rows
    elapsed = time.perf_counter() - start
    assert duplicate is False
    print(
        json.dumps(
            {
                "method": method,
                "threads": pl.thread_pool_size(),
                "seconds": elapsed,
                "baseline_mib": baseline,
                "peak_mib": peak_mib(),
                "duplicate": duplicate,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[100, 1000, 4000])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 12])
    parser.add_argument("--worker", choices=METHODS)
    parser.add_argument("--generate", type=int)
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    if args.generate is not None:
        generate(args.path, args.generate, args.rows)
        return
    if args.worker:
        worker(args.path, args.worker, args.rows)
        return
    with tempfile.TemporaryDirectory(prefix="hashfilter-benchmark-") as directory:
        for prefix in args.prefixes:
            path = Path(directory) / f"strings-{prefix}.parquet"
            command = [
                sys.executable,
                __file__,
                "--path",
                str(path),
                "--rows",
                str(args.rows),
            ]
            subprocess.run(
                [*command, "--generate", str(prefix)],
                env={**os.environ, "POLARS_MAX_THREADS": "1"},
                check=True,
            )
            for threads in args.threads:
                for method in METHODS:
                    result = subprocess.run(
                        [*command, "--worker", method],
                        env={**os.environ, "POLARS_MAX_THREADS": str(threads)},
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    record = json.loads(result.stdout)
                    record.update(prefix_length=prefix, rows=args.rows)
                    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
