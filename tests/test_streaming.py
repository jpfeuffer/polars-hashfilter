"""Tests for streaming behavior and batch vs rowwise semantics."""

import polars as pl
from polars_hashfilter import (
    StringHashSet,
    not_in_and_update,
    not_in_and_update_rowwise,
    update,
    update_chain,
    update_bool,
)


class TestBatchVsRowwise:
    """Tests demonstrating differences between batch and rowwise semantics."""

    def test_duplicates_within_batch_behavior(self):
        """Show how duplicates are handled differently in batch vs rowwise."""
        # Test data with duplicates: "a" appears twice
        df = pl.DataFrame({"id": ["a", "b", "a", "c"]})

        # Batch semantics: all rows evaluated against initial set state
        seen_batch = StringHashSet()
        result_batch = df.select(
            not_in_and_update(pl.col("id"), seen_batch).alias("is_new")
        )
        # All three unique values should return True on first batch
        # Both "a" instances return True because evaluation happens before insertion
        assert result_batch["is_new"].to_list() == [True, True, True, True]
        assert set(seen_batch.to_list()) == {"a", "b", "c"}

        # Rowwise semantics: rows evaluated sequentially
        seen_rowwise = StringHashSet()
        result_rowwise = df.select(
            not_in_and_update_rowwise(pl.col("id"), seen_rowwise).alias("is_new")
        )
        # First "a" returns True, second "a" returns False (already inserted)
        assert result_rowwise["is_new"].to_list() == [True, True, False, True]
        assert set(seen_rowwise.to_list()) == {"a", "b", "c"}

    def test_batch_keeps_all_duplicates(self):
        """Batch semantics keeps all instances of duplicates in first batch."""
        df = pl.DataFrame({"id": ["x", "x", "x", "y"]})

        seen = StringHashSet()
        result = df.filter(pl.col("id").hashfilter.not_in_and_update(seen))

        # All three "x" instances are kept (batch evaluation)
        assert result["id"].to_list() == ["x", "x", "x", "y"]
        assert len(seen) == 2  # Only unique values

    def test_rowwise_keeps_only_first_duplicate(self):
        """Rowwise semantics keeps only first occurrence of duplicates."""
        df = pl.DataFrame({"id": ["x", "x", "x", "y"]})

        seen = StringHashSet()
        result = df.filter(pl.col("id").hashfilter.not_in_and_update_rowwise(seen))

        # Only first "x" is kept (row-by-row evaluation)
        assert result["id"].to_list() == ["x", "y"]
        assert len(seen) == 2  # Only unique values


class TestStreamingExecution:
    """Tests for streaming mode compatibility."""

    def test_batch_works_in_streaming(self):
        """Verify batch semantics work in streaming mode."""
        lf = pl.LazyFrame({"id": ["a", "b", "a", "c", "b"]})

        seen = StringHashSet()
        result = lf.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect(
            engine="streaming"
        )

        # In streaming mode, batches are processed independently
        # So behavior depends on batch boundaries
        assert "a" in result["id"].to_list()
        assert "b" in result["id"].to_list()
        assert "c" in result["id"].to_list()

    def test_rowwise_works_in_streaming(self):
        """Verify rowwise semantics work in streaming mode."""
        lf = pl.LazyFrame({"id": ["a", "b", "a", "c", "b"]})

        seen = StringHashSet()
        result = lf.filter(
            pl.col("id").hashfilter.not_in_and_update_rowwise(seen)
        ).collect(engine="streaming")

        # Rowwise should deduplicate within each batch
        assert set(result["id"].to_list()).issubset({"a", "b", "c"})
        assert len(seen) == 3

    def test_multiple_lazyframes_batch(self):
        """Test batch semantics across multiple LazyFrames (documented use case)."""
        lazy_frames = [
            pl.LazyFrame({"id": ["a", "b", "a"]}),
            pl.LazyFrame({"id": ["b", "c", "c"]}),
            pl.LazyFrame({"id": ["c", "d"]}),
        ]

        seen = StringHashSet()
        results = []

        for lf in lazy_frames:
            df = lf.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect(
                engine="in-memory"
            )
            results.append(df["id"].to_list())

        # First frame: all instances kept (including duplicate "a")
        assert results[0] == ["a", "b", "a"]
        # Second frame: only "c" instances (b already seen)
        assert results[1] == ["c", "c"]
        # Third frame: only "d" (c already seen)
        assert results[2] == ["d"]

        assert set(seen.to_list()) == {"a", "b", "c", "d"}

    def test_multiple_lazyframes_rowwise(self):
        """Test rowwise semantics across multiple LazyFrames."""
        lazy_frames = [
            pl.LazyFrame({"id": ["a", "b", "a"]}),
            pl.LazyFrame({"id": ["b", "c", "c"]}),
            pl.LazyFrame({"id": ["c", "d"]}),
        ]

        seen = StringHashSet()
        results = []

        for lf in lazy_frames:
            df = lf.filter(
                pl.col("id").hashfilter.not_in_and_update_rowwise(seen)
            ).collect()
            results.append(df["id"].to_list())

        # First frame: only first "a" kept (rowwise dedup)
        assert results[0] == ["a", "b"]
        # Second frame: only first "c" (b already seen, rowwise dedup)
        assert results[1] == ["c"]
        # Third frame: only "d" (c already seen)
        assert results[2] == ["d"]

        assert set(seen.to_list()) == {"a", "b", "c", "d"}

    def test_is_elementwise_allows_streaming(self):
        """Verify expressions are marked as elementwise for streaming support."""
        # This test verifies streaming actually works by processing larger data
        lf = pl.LazyFrame({"id": [f"val_{i % 100}" for i in range(10000)]})

        seen = StringHashSet()

        # Should not raise, proves streaming is supported
        result = lf.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect(
            engine="streaming"
        )

        assert len(seen) == 100  # 100 unique values
        assert len(result) > 0

    def test_demo_batch_vs_rowwise(self):
        """Visual demo showing difference between batch and rowwise semantics."""
        # Demo data with duplicates
        df = pl.DataFrame({"id": ["a", "b", "a", "c", "b"]})

        # Batch semantics (default)
        seen_batch = StringHashSet()
        result_batch = df.filter(pl.col("id").hashfilter.not_in_and_update(seen_batch))
        # Keeps all instances of duplicates in first batch
        assert result_batch["id"].to_list() == ["a", "b", "a", "c", "b"]
        assert set(seen_batch.to_list()) == {"a", "b", "c"}

        # Rowwise semantics
        seen_rowwise = StringHashSet()
        result_rowwise = df.filter(
            pl.col("id").hashfilter.not_in_and_update_rowwise(seen_rowwise)
        )
        # Keeps only first occurrence of each value
        assert result_rowwise["id"].to_list() == ["a", "b", "c"]
        assert set(seen_rowwise.to_list()) == {"a", "b", "c"}


class TestNullHandling:
    """Test that both variants handle nulls consistently."""

    def test_batch_with_nulls(self):
        """Nulls should be ignored in batch mode."""
        df = pl.DataFrame({"id": ["a", None, "b", None, "a"]})
        seen = StringHashSet()

        result = df.select(not_in_and_update(pl.col("id"), seen).alias("is_new"))

        # Nulls return None, both "a" instances return True
        assert result["is_new"].to_list() == [True, None, True, None, True]

    def test_rowwise_with_nulls(self):
        """Nulls should be ignored in rowwise mode."""
        df = pl.DataFrame({"id": ["a", None, "b", None, "a"]})
        seen = StringHashSet()

        result = df.select(
            not_in_and_update_rowwise(pl.col("id"), seen).alias("is_new")
        )

        # Nulls return None, first "a" True, second "a" False
        assert result["is_new"].to_list() == [True, None, True, None, False]


class TestStreamingChunkBoundaryBehavior:
    """
    Tests demonstrating streaming chunk boundary behavior.

    IMPORTANT: This class documents that `not_in_and_update` batch semantics
    only apply WITHIN a single expression call (i.e., within a single chunk).
    In streaming mode, Polars processes data in chunks, and each chunk is a
    separate expression call.

    This means:
    - For data that fits in a single chunk: batch semantics work as expected
    - For data split across chunks: behavior at chunk boundaries is like rowwise

    This is NOT a bug - it's inherent to how streaming works. The documentation
    and these tests make this behavior explicit.
    """

    def test_batch_semantics_single_chunk_collect(self):
        """Verify batch semantics work correctly with non-streaming collect.

        With regular collect(), all data is processed as a single batch,
        so duplicates within the data all return True.
        """
        df = pl.DataFrame({"id": ["a", "b", "a", "c", "b", "a"]})
        seen = StringHashSet()

        result = df.select(not_in_and_update(pl.col("id"), seen).alias("is_new"))

        # ALL instances return True because lookup happens before insert
        assert result["is_new"].to_list() == [True, True, True, True, True, True]
        assert set(seen.to_list()) == {"a", "b", "c"}

    def test_empty_set_no_filtering_single_batch(self):
        """CRITICAL: Empty set must not filter ANY values in first batch.

        This is the key guarantee of batch semantics: when starting with an
        empty set, all values in the first expression call should return True.
        """
        df = pl.DataFrame({"id": ["a", "a", "a", "b", "b", "c"]})
        seen = StringHashSet()

        result = df.filter(pl.col("id").hashfilter.not_in_and_update(seen))

        # ALL rows kept - nothing filtered when starting with empty set
        assert len(result) == 6
        assert result["id"].to_list() == ["a", "a", "a", "b", "b", "c"]

    def test_streaming_may_split_chunks(self):
        """Demonstrate that streaming processes data in chunks.

        This test shows that with larger data, streaming mode may process
        data in multiple chunks. The exact behavior depends on Polars'
        internal chunking strategy which may vary.

        NOTE: We cannot guarantee exact chunk boundaries, but we can verify
        that the function works correctly regardless of how data is chunked.
        """
        # Create data with many duplicates
        n_rows = 10000
        lf = pl.LazyFrame({"id": [f"val_{i % 50}" for i in range(n_rows)]})

        seen = StringHashSet()
        result = lf.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect(
            engine="streaming"
        )

        # Should have all 50 unique values
        assert len(seen) == 50

        # In streaming: result count depends on chunk boundaries
        # - If all data in one chunk: would keep all n_rows
        # - If chunked: keeps first occurrence per chunk boundary
        # We just verify it works and returns valid data
        assert len(result) >= 50  # At minimum, one of each unique value
        assert len(result) <= n_rows  # At most, all rows

    def test_streaming_vs_collect_difference_documented(self):
        """Document potential difference between streaming and regular collect.

        This test demonstrates that streaming mode MAY produce different
        results than regular collect for `not_in_and_update` when data is
        split across chunks. This is expected and documented behavior.
        """
        # Small data - likely single chunk in streaming
        df = pl.DataFrame({"id": ["a", "b", "a"]})

        seen_regular = StringHashSet()
        result_regular = df.filter(
            pl.col("id").hashfilter.not_in_and_update(seen_regular)
        )

        seen_streaming = StringHashSet()
        result_streaming = (
            df.lazy()
            .filter(pl.col("id").hashfilter.not_in_and_update(seen_streaming))
            .collect(engine="streaming")
        )

        # For small data like this, both should produce same result
        # (fits in single chunk)
        assert result_regular["id"].to_list() == ["a", "b", "a"]

        # Streaming result may be same (single chunk) or different (multiple chunks)
        # We document this rather than assert equality
        print(f"Regular collect: {result_regular['id'].to_list()}")
        print(f"Streaming collect: {result_streaming['id'].to_list()}")

        # Both should have same unique values in the set
        assert set(seen_regular.to_list()) == set(seen_streaming.to_list())

    def test_rowwise_consistent_across_modes(self):
        """Rowwise semantics are consistent regardless of execution mode.

        Unlike batch semantics, rowwise semantics produce predictable results
        regardless of whether streaming is used or how data is chunked.
        """
        df = pl.DataFrame({"id": ["a", "b", "a", "c", "b"]})

        seen_regular = StringHashSet()
        result_regular = df.filter(
            pl.col("id").hashfilter.not_in_and_update_rowwise(seen_regular)
        )

        seen_streaming = StringHashSet()
        result_streaming = (
            df.lazy()
            .filter(pl.col("id").hashfilter.not_in_and_update_rowwise(seen_streaming))
            .collect(engine="streaming")
        )

        # Both produce first-occurrence-only semantics
        assert result_regular["id"].to_list() == ["a", "b", "c"]
        # Streaming should be consistent (order may vary based on parallel execution)
        assert set(result_streaming["id"].to_list()) == {"a", "b", "c"}


class TestUpdateFunctions:
    """Tests for the update, update_chain, and update_bool functions."""

    def test_update_basic(self):
        """Test update function adds values to set."""
        df = pl.DataFrame({"id": ["a", "b", "c"]})
        seen = StringHashSet()

        # update returns null series
        result = df.select(update(pl.col("id"), seen))
        assert result.shape == (3, 1)
        assert result["id"].null_count() == 3

        # Values should be in the set
        assert set(seen.to_list()) == {"a", "b", "c"}

    def test_update_with_nulls(self):
        """Test update ignores null values."""
        df = pl.DataFrame({"id": ["a", None, "b", None]})
        seen = StringHashSet()

        df.select(update(pl.col("id"), seen))

        # Only non-null values in set
        assert set(seen.to_list()) == {"a", "b"}

    def test_update_chain_returns_original(self):
        """Test update_chain returns original series."""
        df = pl.DataFrame({"id": ["a", "b", "c"]})
        seen = StringHashSet()

        result = df.select(update_chain(pl.col("id"), seen).alias("processed"))

        # Returns original values
        assert result["processed"].to_list() == ["a", "b", "c"]
        # And updates the set
        assert set(seen.to_list()) == {"a", "b", "c"}

    def test_update_chain_allows_chaining(self):
        """Test update_chain can be chained with other operations."""
        df = pl.DataFrame({"id": ["hello", "world"]})
        seen = StringHashSet()

        result = df.select(
            pl.col("id").hashfilter.update_chain(seen).str.to_uppercase().alias("upper")
        )

        assert result["upper"].to_list() == ["HELLO", "WORLD"]
        assert set(seen.to_list()) == {"hello", "world"}

    def test_update_bool_returns_insertion_status(self):
        """Test update_bool returns True for new values, False for existing."""
        df = pl.DataFrame({"id": ["a", "b", "a", "c", "b"]})
        seen = StringHashSet()

        result = df.select(update_bool(pl.col("id"), seen).alias("was_new"))

        # Row-by-row: first occurrence is new, subsequent are not
        assert result["was_new"].to_list() == [True, True, False, True, False]
        assert set(seen.to_list()) == {"a", "b", "c"}

    def test_update_bool_with_nulls(self):
        """Test update_bool returns null for null input."""
        df = pl.DataFrame({"id": ["a", None, "a", None]})
        seen = StringHashSet()

        result = df.select(update_bool(pl.col("id"), seen).alias("was_new"))

        assert result["was_new"].to_list() == [True, None, False, None]
        assert set(seen.to_list()) == {"a"}

    def test_update_bool_is_same_as_rowwise(self):
        """update_bool is functionally equivalent to not_in_and_update_rowwise."""
        df = pl.DataFrame({"id": ["a", "b", "a", "c", "b", "d"]})

        seen_bool = StringHashSet()
        result_bool = df.select(update_bool(pl.col("id"), seen_bool).alias("is_new"))

        seen_rowwise = StringHashSet()
        result_rowwise = df.select(
            not_in_and_update_rowwise(pl.col("id"), seen_rowwise).alias("is_new")
        )

        assert result_bool["is_new"].to_list() == result_rowwise["is_new"].to_list()
        assert set(seen_bool.to_list()) == set(seen_rowwise.to_list())

    def test_not_in_then_update_pattern(self):
        """Test the recommended streaming pattern: not_in + update.

        This pattern gives explicit control over lookup and update timing,
        making behavior predictable regardless of execution mode.
        """
        df = pl.DataFrame({"id": ["a", "b", "a", "c"]})
        seen = StringHashSet()

        # Step 1: Filter using read-only lookup
        filtered = df.filter(pl.col("id").hashfilter.not_in(seen))
        assert filtered["id"].to_list() == ["a", "b", "a", "c"]  # All pass (empty set)

        # Step 2: Explicitly update the set with filtered values
        filtered.select(pl.col("id").hashfilter.update(seen))
        assert set(seen.to_list()) == {"a", "b", "c"}

        # Step 3: Next batch would filter correctly
        df2 = pl.DataFrame({"id": ["a", "d", "b"]})
        filtered2 = df2.filter(pl.col("id").hashfilter.not_in(seen))
        assert filtered2["id"].to_list() == ["d"]

    def test_namespace_update_methods(self):
        """Test all update methods via namespace."""
        df = pl.DataFrame({"id": ["x", "y", "x"]})

        seen1 = StringHashSet()
        df.select(pl.col("id").hashfilter.update(seen1))
        assert set(seen1.to_list()) == {"x", "y"}

        seen2 = StringHashSet()
        result2 = df.select(pl.col("id").hashfilter.update_chain(seen2))
        assert result2["id"].to_list() == ["x", "y", "x"]
        assert set(seen2.to_list()) == {"x", "y"}

        seen3 = StringHashSet()
        result3 = df.select(pl.col("id").hashfilter.update_bool(seen3))
        assert result3["id"].to_list() == [True, True, False]
        assert set(seen3.to_list()) == {"x", "y"}
