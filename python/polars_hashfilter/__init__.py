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

__all__ = ["StringHashSet", "__version__", "is_in_hashset", "not_in_hashset", "not_in_and_update"]

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

        Args:
            hashset: The StringHashSet to check against and update

        Returns:
            Boolean expression where True indicates the value was NOT in the set
            (and has now been added to it)
        """
        return not_in_and_update(self._expr, hashset)
