"""Polars plugin exposing PlHashSet for efficient filtering with persistent sets.

This plugin provides:
- `StringHashSet`: A Python-accessible wrapper around Polars' PlHashSet<String>
- Expression namespace `hashfilter` with filtering methods

Example usage for deduplication across LazyFrames:
    ```python
    import polars as pl
    from polars_hashfilter import StringHashSet

    # Create a persistent set to track seen values
    seen = StringHashSet()

    for lf in lazy_frames:
        # Keep only rows we haven't seen before, and remember them
        df = lf.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect()
        process(df)
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_hashfilter._internal import StringHashSet
from polars_hashfilter._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

__all__ = [
    "StringHashSet",
    "__version__",
    "is_in_hashset",
    "not_in_hashset",
    "not_in_and_update",
    "not_in_and_update_rowwise",
    "update",
    "update_chain",
    "update_bool",
]

PLUGIN_PATH = Path(__file__).parent


def is_in_hashset(expr: IntoExpr, hashset: StringHashSet) -> pl.Expr:
    """Check if values exist in the StringHashSet.

    Args:
        expr: The expression to filter (must be string type)
        hashset: The StringHashSet to check against

    Returns:
        Boolean expression where True indicates the value exists in the set
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="is_in_hashset",
        args=expr,
        kwargs={"set_ptr": hashset._ptr()},
        is_elementwise=True,
    )


def not_in_hashset(expr: IntoExpr, hashset: StringHashSet) -> pl.Expr:
    """Check if values do NOT exist in the StringHashSet.

    Args:
        expr: The expression to filter (must be string type)
        hashset: The StringHashSet to check against

    Returns:
        Boolean expression where True indicates the value does NOT exist in the set
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="not_in_hashset",
        args=expr,
        kwargs={"set_ptr": hashset._ptr()},
        is_elementwise=True,
    )


def not_in_and_update(expr: IntoExpr, hashset: StringHashSet) -> pl.Expr:
    """Check if values do NOT exist in the set, then add them to the set.

    This implements an "anti-join with update" pattern - perfect for
    deduplication across multiple LazyFrames.

    **Batch Semantics**: All rows are evaluated against the initial set state
    BEFORE any values are inserted. For duplicates within a batch, all instances
    return True.

    ⚠️ **Streaming Mode Warning**:
    In streaming mode (`collect(engine="streaming")`), Polars processes data in
    chunks. Each chunk triggers a separate expression call, meaning:
    - Batch semantics only apply WITHIN each chunk
    - Across chunk boundaries, values from previous chunks are already in the set
    - For large data, this effectively behaves like `not_in_and_update_rowwise`
      at chunk boundaries

    **Recommended patterns for streaming:**
    - Use `not_in_and_update_rowwise` for consistent row-by-row deduplication
    - OR use `not_in_hashset` for lookup + separate `update` call for explicit control

    Args:
        expr: The expression to filter (must be string type)
        hashset: The StringHashSet to check against and update

    Returns:
        Boolean expression where True indicates the value was NOT in the set
        (and has now been added to it)
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="not_in_and_update",
        args=expr,
        kwargs={"set_ptr": hashset._ptr()},
        is_elementwise=True,
    )


def not_in_and_update_rowwise(expr: IntoExpr, hashset: StringHashSet) -> pl.Expr:
    """Check if values do NOT exist in the set, then add them (row-by-row evaluation).

    IMPORTANT: Uses row-by-row semantics where duplicates within the same batch
    are handled sequentially. The first occurrence returns True, subsequent
    duplicates return False.

    Example:
        Input: ["a", "b", "a"] with empty set
        - not_in_and_update: [True, True, True] (batch: all evaluated first)
        - not_in_and_update_rowwise: [True, True, False] (second "a" sees first)

    Use this when:
    - You have no duplicates within batches, OR
    - You explicitly want to keep only first occurrence of duplicates

    Args:
        expr: The expression to filter (must be string type)
        hashset: The StringHashSet to check against and update

    Returns:
        Boolean expression where True indicates the value was NOT in the set
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="not_in_and_update_rowwise",
        args=expr,
        kwargs={"set_ptr": hashset._ptr()},
        is_elementwise=True,
    )


def update(expr: IntoExpr, hashset: StringHashSet) -> pl.Expr:
    """Add all (non-null) string values from a Series to the StringHashSet.

    This is a write-only operation - no filtering, just bulk insert.
    Returns a null series (the expression produces no meaningful output).

    Use this for explicit control over when values are added to the set,
    especially in streaming scenarios where you want to separate lookup from update.

    Example:
        ```python
        seen = StringHashSet()
        # First: filter using read-only lookup
        filtered = df.filter(pl.col("id").hashfilter.not_in(seen))
        # Then: explicitly add the filtered values
        filtered.select(pl.col("id").hashfilter.update(seen))
        ```

    Args:
        expr: The expression containing values to add (must be string type)
        hashset: The StringHashSet to update

    Returns:
        Null series (side-effect only operation)
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="update",
        args=expr,
        kwargs={"set_ptr": hashset._ptr()},
        is_elementwise=True,
    )


def update_chain(expr: IntoExpr, hashset: StringHashSet) -> pl.Expr:
    """Add all (non-null) string values from a Series to the StringHashSet.

    Returns the original input Series unchanged, allowing expression chaining.

    Example:
        ```python
        # Chain update with other operations
        df.select(pl.col("id").hashfilter.update_chain(seen).alias("processed_id"))
        ```

    Args:
        expr: The expression containing values to add (must be string type)
        hashset: The StringHashSet to update

    Returns:
        The original input Series unchanged
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="update_chain",
        args=expr,
        kwargs={"set_ptr": hashset._ptr()},
        is_elementwise=True,
    )


def update_bool(expr: IntoExpr, hashset: StringHashSet) -> pl.Expr:
    """Add all (non-null) string values from a Series to the StringHashSet.

    Returns a boolean mask where True indicates the value was newly inserted
    (was not already in the set), and False indicates it was already present.
    Null values return null.

    Uses row-by-row evaluation, so duplicates within the same batch:
    - First occurrence: returns True (newly inserted)
    - Subsequent occurrences: return False (already present from earlier row)

    Example:
        ```python
        seen = StringHashSet()
        df.select(pl.col("id").hashfilter.update_bool(seen).alias("was_new"))
        ```

    Args:
        expr: The expression containing values to add (must be string type)
        hashset: The StringHashSet to update

    Returns:
        Boolean expression where True indicates the value was newly inserted
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="update_bool",
        args=expr,
        kwargs={"set_ptr": hashset._ptr()},
        is_elementwise=True,
    )


@pl.api.register_expr_namespace("hashfilter")
class HashFilterNameSpace:
    """Expression namespace for hashset filtering operations.

    Usage:
        ```python
        seen = StringHashSet()
        df.filter(pl.col("id").hashfilter.is_in(seen))
        df.filter(pl.col("id").hashfilter.not_in(seen))
        df.filter(pl.col("id").hashfilter.not_in_and_update(seen))
        ```
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def is_in(self, hashset: StringHashSet) -> pl.Expr:
        """Check if values exist in the StringHashSet.

        Args:
            hashset: The StringHashSet to check against

        Returns:
            Boolean expression where True indicates the value exists in the set
        """
        return is_in_hashset(self._expr, hashset)

    def not_in(self, hashset: StringHashSet) -> pl.Expr:
        """Check if values do NOT exist in the StringHashSet.

        Args:
            hashset: The StringHashSet to check against

        Returns:
            Boolean expression where True indicates the value does NOT exist in the set
        """
        return not_in_hashset(self._expr, hashset)

    def not_in_and_update(self, hashset: StringHashSet) -> pl.Expr:
        """Check if values do NOT exist in the set, then add them.

        This implements an "anti-join with update" pattern - perfect for
        deduplication across multiple LazyFrames.

        **Batch Semantics**: Within each expression call, all lookups happen
        before any inserts.

        ⚠️ **Streaming Warning**: In streaming mode, each chunk is a separate
        expression call. Use `not_in_and_update_rowwise` for consistent behavior,
        or `not_in` + `update` for explicit control.

        Args:
            hashset: The StringHashSet to check against and update

        Returns:
            Boolean expression where True indicates the value was NOT in the set
            (and has now been added to it)
        """
        return not_in_and_update(self._expr, hashset)

    def not_in_and_update_rowwise(self, hashset: StringHashSet) -> pl.Expr:
        """Check if values do NOT exist in the set, then add them (row-by-row).

        Row-by-row evaluation: duplicates within same batch are handled sequentially.
        First occurrence returns True, subsequent duplicates return False.

        Args:
            hashset: The StringHashSet to check against and update

        Returns:
            Boolean expression where True indicates the value was NOT in the set
        """
        return not_in_and_update_rowwise(self._expr, hashset)

    def update(self, hashset: StringHashSet) -> pl.Expr:
        """Add all values to the StringHashSet.

        Write-only operation with no return value (null series).
        Use for explicit control over when values are added to the set.

        Args:
            hashset: The StringHashSet to update

        Returns:
            Null series (side-effect only operation)
        """
        return update(self._expr, hashset)

    def update_chain(self, hashset: StringHashSet) -> pl.Expr:
        """Add all values to the StringHashSet, returning the original Series.

        Allows chaining with other operations.

        Args:
            hashset: The StringHashSet to update

        Returns:
            The original input Series unchanged
        """
        return update_chain(self._expr, hashset)

    def update_bool(self, hashset: StringHashSet) -> pl.Expr:
        """Add all values to the StringHashSet, returning insertion status.

        Row-by-row evaluation: returns True if value was newly inserted,
        False if already present.

        Args:
            hashset: The StringHashSet to update

        Returns:
            Boolean expression where True indicates newly inserted
        """
        return update_bool(self._expr, hashset)
