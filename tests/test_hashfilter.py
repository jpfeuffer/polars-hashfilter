"""Tests for polars_hashfilter plugin."""

import polars as pl
import pytest

from polars_hashfilter import StringHashSet, is_in_hashset, not_in_and_update, not_in_hashset


class TestStringHashSet:
    """Tests for the StringHashSet class."""

    def test_new_empty(self):
        """Test creating an empty set."""
        s = StringHashSet()
        assert len(s) == 0
        assert s.is_empty()

    def test_with_capacity(self):
        """Test creating a set with capacity."""
        s = StringHashSet.with_capacity(100)
        assert len(s) == 0
        assert s.is_empty()

    def test_from_iter(self):
        """Test creating a set from an iterable."""
        s = StringHashSet.from_iter(["a", "b", "c"])
        assert len(s) == 3
        assert s.contains("a")
        assert s.contains("b")
        assert s.contains("c")
        assert not s.contains("d")

    def test_insert(self):
        """Test inserting values."""
        s = StringHashSet()
        assert s.insert("a") is True  # New value
        assert s.insert("a") is False  # Already exists
        assert len(s) == 1

    def test_remove(self):
        """Test removing values."""
        s = StringHashSet.from_iter(["a", "b"])
        assert s.remove("a") is True
        assert s.remove("a") is False  # Already removed
        assert len(s) == 1

    def test_clear(self):
        """Test clearing the set."""
        s = StringHashSet.from_iter(["a", "b", "c"])
        s.clear()
        assert len(s) == 0
        assert s.is_empty()

    def test_to_list(self):
        """Test converting to list."""
        s = StringHashSet.from_iter(["a", "b", "c"])
        lst = s.to_list()
        assert set(lst) == {"a", "b", "c"}

    def test_extend(self):
        """Test extending the set."""
        s = StringHashSet.from_iter(["a"])
        s.extend(["b", "c"])
        assert len(s) == 3

    def test_repr(self):
        """Test string representation."""
        s = StringHashSet.from_iter(["a", "b"])
        assert repr(s) == "StringHashSet(len=2)"

    def test_ptr_is_stable(self):
        """Test that the pointer is stable across operations."""
        s = StringHashSet()
        ptr1 = s._ptr()
        s.insert("a")
        s.insert("b")
        ptr2 = s._ptr()
        assert ptr1 == ptr2, "Pointer should be stable (no reallocation of Arc)"


class TestIsInHashSet:
    """Tests for the is_in_hashset expression."""

    def test_basic_filter(self):
        """Test basic is_in filtering."""
        seen = StringHashSet.from_iter(["a", "c"])
        df = pl.DataFrame({"id": ["a", "b", "c", "d"]})

        result = df.select(is_in_hashset(pl.col("id"), seen).alias("in_set"))
        assert result["in_set"].to_list() == [True, False, True, False]

    def test_filter_with_namespace(self):
        """Test is_in using the expression namespace."""
        seen = StringHashSet.from_iter(["a", "c"])
        df = pl.DataFrame({"id": ["a", "b", "c", "d"]})

        result = df.filter(pl.col("id").hashfilter.is_in(seen))
        assert result["id"].to_list() == ["a", "c"]

    def test_with_nulls(self):
        """Test is_in with null values."""
        seen = StringHashSet.from_iter(["a", "b"])
        df = pl.DataFrame({"id": ["a", None, "b", "c"]})

        result = df.select(is_in_hashset(pl.col("id"), seen).alias("in_set"))
        assert result["in_set"].to_list() == [True, None, True, False]


class TestNotInHashSet:
    """Tests for the not_in_hashset expression."""

    def test_basic_filter(self):
        """Test basic not_in filtering."""
        seen = StringHashSet.from_iter(["a", "c"])
        df = pl.DataFrame({"id": ["a", "b", "c", "d"]})

        result = df.select(not_in_hashset(pl.col("id"), seen).alias("not_in_set"))
        assert result["not_in_set"].to_list() == [False, True, False, True]

    def test_filter_with_namespace(self):
        """Test not_in using the expression namespace."""
        seen = StringHashSet.from_iter(["a", "c"])
        df = pl.DataFrame({"id": ["a", "b", "c", "d"]})

        result = df.filter(pl.col("id").hashfilter.not_in(seen))
        assert result["id"].to_list() == ["b", "d"]


class TestNotInAndUpdate:
    """Tests for the not_in_and_update expression."""

    def test_basic_antijoin(self):
        """Test basic not_in_and_update."""
        seen = StringHashSet()
        df = pl.DataFrame({"id": ["a", "b", "a", "c"]})

        # First call: checks against initially empty set, then adds all unique values
        # Note: within a single expression, all values are checked first, then added
        # So duplicates within the same DataFrame are all marked as "new"
        result = df.select(not_in_and_update(pl.col("id"), seen).alias("is_new"))
        assert result["is_new"].to_list() == [True, True, True, True]

        # Set should now contain a, b, c (deduped)
        assert len(seen) == 3
        assert seen.contains("a")
        assert seen.contains("b")
        assert seen.contains("c")

    def test_across_dataframes(self):
        """Test deduplication across multiple DataFrames."""
        seen = StringHashSet()

        df1 = pl.DataFrame({"id": ["a", "b"]})
        df2 = pl.DataFrame({"id": ["b", "c"]})
        df3 = pl.DataFrame({"id": ["c", "d"]})

        # Process df1
        result1 = df1.filter(pl.col("id").hashfilter.not_in_and_update(seen))
        assert result1["id"].to_list() == ["a", "b"]
        assert len(seen) == 2

        # Process df2 - b is already seen
        result2 = df2.filter(pl.col("id").hashfilter.not_in_and_update(seen))
        assert result2["id"].to_list() == ["c"]
        assert len(seen) == 3

        # Process df3 - c is already seen
        result3 = df3.filter(pl.col("id").hashfilter.not_in_and_update(seen))
        assert result3["id"].to_list() == ["d"]
        assert len(seen) == 4

    def test_with_lazyframes(self):
        """Test deduplication across multiple LazyFrames."""
        seen = StringHashSet()

        lf1 = pl.LazyFrame({"id": ["x", "y"]})
        lf2 = pl.LazyFrame({"id": ["y", "z"]})

        result1 = lf1.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect()
        assert result1["id"].to_list() == ["x", "y"]

        result2 = lf2.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect()
        assert result2["id"].to_list() == ["z"]

    def test_with_nulls(self):
        """Test not_in_and_update with null values (nulls are not added to set)."""
        seen = StringHashSet()
        df = pl.DataFrame({"id": ["a", None, "b"]})

        result = df.select(not_in_and_update(pl.col("id"), seen).alias("is_new"))
        # a: new, None: null result, b: new
        assert result["is_new"].to_list() == [True, None, True]

        # Only non-null values added
        assert len(seen) == 2
        assert seen.contains("a")
        assert seen.contains("b")


class TestZeroCopy:
    """Tests to verify zero-copy behavior."""

    def test_pointer_stability_across_expressions(self):
        """Test that the set pointer remains stable when used in expressions."""
        seen = StringHashSet.from_iter(["a", "b"])
        ptr_before = seen._ptr()

        df = pl.DataFrame({"id": ["a", "c", "d"]})
        df.filter(pl.col("id").hashfilter.not_in_and_update(seen))

        ptr_after = seen._ptr()
        assert ptr_before == ptr_after, "Arc pointer should not change"

    def test_clone_shares_data(self):
        """Test that Python variable assignment shares the same underlying data."""
        s1 = StringHashSet.from_iter(["a"])
        s2 = s1  # Python assignment

        # Both should point to the same underlying data
        assert s1._ptr() == s2._ptr()

        # Modifications through one affect the other
        s1.insert("b")
        assert s2.contains("b")
