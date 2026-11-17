//! Polars expression plugins for filtering with StringHashSet.

use crate::hashset::{insert_borrowed, lookup};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Kwargs for is_in_hashset expression.
#[derive(Deserialize)]
struct IsInHashSetKwargs {
    set_id: u64,
}

/// Check if values in the input column exist in the StringHashSet.
///
/// Returns a boolean mask where `true` indicates the value exists in the set.
/// This is a read-only operation on the set.
#[polars_expr(output_type=Boolean)]
fn is_in_hashset(inputs: &[Series], kwargs: IsInHashSetKwargs) -> PolarsResult<Series> {
    let ca = string_values(&inputs[0])?;

    // The expression retains the Python owner; upgrade its checked weak handle.
    let set = lookup(kwargs.set_id)?;

    let guard = set.read().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire read lock: {}", e).into())
    })?;

    let out: BooleanChunked = ca
        .iter()
        .map(|opt_v| opt_v.map(|v| guard.contains(v)))
        .collect();

    Ok(out.into_series())
}

/// Kwargs for filter_and_update expression.
#[derive(Deserialize)]
struct FilterAndUpdateKwargs {
    set_id: u64,
}

/// Filter rows NOT in the set, then add those values to the set.
///
/// This implements an "anti-join with update" pattern:
/// - Returns a boolean mask where `true` indicates the value is NOT in the set
/// - Adds all non-matching (non-null) values to the set for future calls
///
/// # Batch Semantics
///
/// Within a single expression call, ALL lookups are performed BEFORE any inserts.
/// This means duplicates within the same batch all return `true`.
///
/// # ⚠️ Streaming Mode Warning
///
/// In streaming mode (`collect(engine="streaming")`), Polars processes data in chunks.
/// Each chunk triggers a separate expression call, meaning:
/// - Batch semantics only apply WITHIN each chunk
/// - Across chunk boundaries, values from previous chunks are already in the set
/// - For large data, this effectively behaves like `not_in_and_update_rowwise`
///   at chunk boundaries
///
/// **Recommended patterns for streaming:**
/// - Use `not_in_and_update_rowwise` for consistent row-by-row deduplication
/// - OR use `not_in_hashset` for lookup + separate `update` call for explicit control
///
/// # Perfect Use Case: Multiple LazyFrames
///
/// ```python
/// seen = StringHashSet()
/// for lf in lazy_frames:
///     # Batch semantics apply per expression call, not necessarily per collect.
///     df = lf.filter(col("id").hashfilter.not_in_and_update(seen)).collect()
/// ```
#[polars_expr(output_type=Boolean)]
fn not_in_and_update(inputs: &[Series], kwargs: FilterAndUpdateKwargs) -> PolarsResult<Series> {
    let ca = string_values(&inputs[0])?;

    let set = lookup(kwargs.set_id)?;

    // Hold one lock across both passes so parallel batches cannot interleave.
    let mut guard = set.write().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire write lock: {}", e).into())
    })?;

    // First pass: determine which values are NOT in the set
    let results: Vec<Option<bool>> = {
        ca.iter()
            .map(|opt_v| opt_v.map(|v| !guard.contains(v)))
            .collect()
    };

    // Second pass: add new values to the set
    {
        for (opt_v, opt_result) in ca.iter().zip(results.iter()) {
            if let (Some(v), Some(true)) = (opt_v, opt_result) {
                insert_borrowed(&mut guard, v);
            }
        }
    }

    let out: BooleanChunked = results.into_iter().collect();
    Ok(out.into_series())
}

/// Filter rows NOT in the set, then add those values - row-by-row evaluation.
///
/// **IMPORTANT**: This uses single-pass row-by-row evaluation, which means duplicates
/// within the same batch will be handled differently than `not_in_and_update`:
///
/// Example with `["a", "b", "a"]`:
/// - `not_in_and_update`: Returns `[true, true, true]` (batch semantics: all evaluated
///   against initial set state, then all inserted)
/// - `not_in_and_update_rowwise`: Returns `[true, true, false]` (row-by-row: second "a"
///   sees first "a" already inserted)
///
/// **Use this when**:
/// - You have no duplicates within batches, OR
/// - You explicitly want row-by-row semantics (e.g., keeping only first occurrence)
///
/// **Advantages**:
/// - One pass (vs separate lookup and insertion passes)
/// - Better streaming performance
/// - Lower memory (no intermediate Vec)
#[polars_expr(output_type=Boolean)]
fn not_in_and_update_rowwise(
    inputs: &[Series],
    kwargs: FilterAndUpdateKwargs,
) -> PolarsResult<Series> {
    let ca = string_values(&inputs[0])?;
    let set = lookup(kwargs.set_id)?;

    // Single pass: check and insert atomically with write lock
    let mut guard = set.write().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire write lock: {}", e).into())
    })?;

    let out: BooleanChunked = ca
        .iter()
        .map(|opt_v| {
            opt_v.map(|v| {
                // insert returns true if newly inserted (was not present before)
                insert_borrowed(&mut guard, v)
            })
        })
        .collect();

    Ok(out.into_series())
}

/// Check if values in the input column do NOT exist in the StringHashSet.
///
/// Returns a boolean mask where `true` indicates the value does NOT exist in the set.
/// This is a read-only operation on the set.
#[polars_expr(output_type=Boolean)]
fn not_in_hashset(inputs: &[Series], kwargs: IsInHashSetKwargs) -> PolarsResult<Series> {
    let ca = string_values(&inputs[0])?;

    let set = lookup(kwargs.set_id)?;

    let guard = set.read().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire read lock: {}", e).into())
    })?;

    let out: BooleanChunked = ca
        .iter()
        .map(|opt_v| opt_v.map(|v| !guard.contains(v)))
        .collect();

    Ok(out.into_series())
}

/// Kwargs for update expression.
#[derive(Deserialize)]
struct UpdateKwargs {
    set_id: u64,
}

/// Add all (non-null) string values from a Series to the StringHashSet.
///
/// This is a write-only operation - no filtering, just bulk insert.
/// Returns a null literal (the expression produces no meaningful output).
///
/// Use this for explicit control over when values are added to the set,
/// especially in streaming scenarios where you want to separate lookup from update.
///
/// ```python
/// seen = StringHashSet()
/// # First: filter using read-only lookup
/// filtered = df.filter(pl.col("id").hashfilter.not_in(seen))
/// # Then: explicitly add the filtered values
/// filtered.select(pl.col("id").hashfilter.update(seen))
/// ```
#[polars_expr(output_type=Null)]
fn update(inputs: &[Series], kwargs: UpdateKwargs) -> PolarsResult<Series> {
    let ca = string_values(&inputs[0])?;
    let set = lookup(kwargs.set_id)?;

    let mut guard = set.write().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire write lock: {}", e).into())
    })?;

    // Bulk insert all non-null values
    for value in ca.iter().flatten() {
        insert_borrowed(&mut guard, value);
    }

    // Return a null series of the same length
    Ok(Series::new_null(PlSmallStr::EMPTY, ca.len()))
}

/// Add all (non-null) string values from a Series to the StringHashSet.
///
/// Returns the original input Series unchanged, allowing expression chaining.
///
/// ```python
/// # Chain update with other operations
/// df.select(pl.col("id").hashfilter.update_chain(seen).alias("processed_id"))
/// ```
#[polars_expr(output_type_func=same_output_type)]
fn update_chain(inputs: &[Series], kwargs: UpdateKwargs) -> PolarsResult<Series> {
    let ca = string_values(&inputs[0])?;
    let set = lookup(kwargs.set_id)?;

    let mut guard = set.write().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire write lock: {}", e).into())
    })?;

    // Bulk insert all non-null values
    for value in ca.iter().flatten() {
        insert_borrowed(&mut guard, value);
    }

    // Return the original series
    Ok(inputs[0].clone())
}

/// Add all (non-null) string values from a Series to the StringHashSet.
///
/// Returns a boolean mask where `true` indicates the value was newly inserted
/// (was not already in the set), and `false` indicates it was already present.
/// Null values return `null`.
///
/// This uses row-by-row evaluation, so duplicates within the same batch:
/// - First occurrence: returns `true` (newly inserted)
/// - Subsequent occurrences: return `false` (already present from earlier row)
///
/// ```python
/// seen = StringHashSet()
/// df.select(pl.col("id").hashfilter.update_bool(seen).alias("was_new"))
/// ```
#[polars_expr(output_type=Boolean)]
fn update_bool(inputs: &[Series], kwargs: UpdateKwargs) -> PolarsResult<Series> {
    let ca = string_values(&inputs[0])?;
    let set = lookup(kwargs.set_id)?;

    let mut guard = set.write().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire write lock: {}", e).into())
    })?;

    // Row-by-row: insert returns true if newly inserted
    let out: BooleanChunked = ca
        .iter()
        .map(|opt_v| opt_v.map(|v| insert_borrowed(&mut guard, v)))
        .collect();

    Ok(out.into_series())
}

/// Helper function for expressions that return the same type as input.
fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(input_fields[0].clone())
}

fn string_values(input: &Series) -> PolarsResult<StringChunked> {
    if input.dtype() == &DataType::Null {
        Ok(StringChunked::full_null(input.name().clone(), input.len()))
    } else {
        input.str().cloned()
    }
}
