"""Regression tests for plan-owned state and repeated execution."""

import gc
import sys
import threading
import weakref

import polars as pl
import pytest

from polars.plugins import register_plugin_function

from polars_hashfilter import (
    PLUGIN_PATH,
    StringHashSet,
    is_in_hashset,
    not_in_and_update,
    not_in_and_update_rowwise,
    not_in_hashset,
    update,
    update_bool,
    update_chain,
)


def test_plan_retains_owner():
    seen = StringHashSet()
    before = sys.getrefcount(seen)
    query = pl.LazyFrame({"id": ["a", "a", "b"]}).select(
        update_bool("id", seen).alias("new")
    )
    gc.collect()
    assert sys.getrefcount(seen) > before
    assert query.collect(engine="streaming")["new"].sum() == 2
    del query
    gc.collect()
    assert sys.getrefcount(seen) == before


@pytest.mark.parametrize(
    "function",
    [
        is_in_hashset,
        not_in_hashset,
        not_in_and_update,
        not_in_and_update_rowwise,
        update,
        update_chain,
        update_bool,
    ],
)
def test_temporary_owner_survives_plan_clones_and_is_released(function):
    seen = StringHashSet()
    reference = weakref.ref(seen)
    query = pl.LazyFrame({"id": ["a", "a", "b"]}).select(function("id", seen))
    clone = query.clone()
    del query, seen
    gc.collect()
    assert reference() is not None
    assert clone.collect(engine="streaming").height == 3
    del clone
    gc.collect()
    assert reference() is None


def test_expired_handle_is_a_compute_error():
    seen = StringHashSet()
    expression = register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="update_bool",
        args="id",
        kwargs={"set_id": seen._id()},
        is_elementwise=True,
        is_deterministic=False,
    )
    del seen
    gc.collect()
    with pytest.raises(pl.exceptions.ComputeError, match="handle expired"):
        pl.LazyFrame({"id": ["a"]}).select(expression).collect()


def test_persistent_state_and_fresh_execution():
    seen = StringHashSet()
    lf = pl.LazyFrame({"id": ["a", "a", "b", None]})
    query = lf.select(update_bool("id", seen).sum())
    assert query.collect(engine="streaming").item() == 2
    assert query.collect(engine="streaming").item() == 0
    seen.clear()
    assert query.collect(engine="streaming").item() == 2
    assert (
        lf.select(update_bool("id", StringHashSet()).sum())
        .collect(engine="streaming")
        .item()
        == 2
    )


@pytest.mark.parametrize("values", [[], [None, None], ["a", None, "a", "b", None]])
def test_empty_and_nulls(values):
    seen = StringHashSet()
    lf = pl.DataFrame({"id": pl.Series(values, dtype=pl.String)}).lazy()
    result = lf.select((~update_bool("id", seen)).sum().alias("duplicates")).collect(
        engine="streaming"
    )
    expected = len([v for v in values if v is not None]) - len(
        {v for v in values if v is not None}
    )
    assert result.item() == expected
    assert len(seen) == len({v for v in values if v is not None})


@pytest.mark.parametrize("values", [[], [None, None]])
def test_inferred_null_dtype(values):
    seen = StringHashSet()
    result = (
        pl.LazyFrame({"id": values})
        .select(update_bool("id", seen))
        .collect(engine="streaming")
    )
    assert result.height == len(values)
    assert result["id"].null_count() == len(values)
    assert len(seen) == 0


def test_nonstring_dtype_is_rejected():
    with pytest.raises(pl.exceptions.ComputeError, match="expected `String`"):
        pl.LazyFrame({"id": [1, 2]}).select(update_bool("id", StringHashSet())).collect(
            engine="streaming"
        )


def test_mutations_are_not_common_subexpressions():
    seen = StringHashSet()
    expression = update_bool("id", seen)
    result = (
        pl.LazyFrame({"id": ["a", "a", "b"]})
        .select(expression.alias("left"), expression.alias("right"))
        .collect(engine="streaming")
    )
    assert result["left"].sum() + result["right"].sum() == 2


def test_parquet_cross_batch_and_shared_sink(tmp_path):
    values = [f"key-{i % 25_000}" for i in range(60_000)] + [None, None]
    source = tmp_path / "source.parquet"
    destination = tmp_path / "sink.parquet"
    pl.DataFrame({"id": values}).write_parquet(source, row_group_size=20_000)
    rows = 0
    calls = 0
    lock = threading.Lock()

    def observe(batch):
        nonlocal rows, calls
        with lock:
            rows += batch.height
            calls += 1
        return batch

    lf = pl.scan_parquet(source).map_batches(observe, streamable=True)
    seen = StringHashSet()
    validation = lf.select((~update_bool("id", seen)).sum().alias("duplicates"))
    sink = lf.sink_parquet(destination, lazy=True, row_group_size=20_000)
    result, _ = pl.collect_all([validation, sink], engine="streaming")
    assert result.item() == 35_000
    assert len(seen) == 25_000
    assert rows == len(values), (rows, calls)
    assert calls > 1
    assert pl.read_parquet(destination)["id"].to_list() == values
