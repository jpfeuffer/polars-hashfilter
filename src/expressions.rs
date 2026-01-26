//! Polars expression plugins for filtering with StringHashSet.

use crate::StringHashSet;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Kwargs for is_in_hashset expression.
/// The set_ptr is the memory address of the Arc<RwLock<PlHashSet>>.
#[derive(Deserialize)]
struct IsInHashSetKwargs {
    set_ptr: usize,
}

/// Check if values in the input column exist in the StringHashSet.
///
/// Returns a boolean mask where `true` indicates the value exists in the set.
/// This is a read-only operation on the set.
#[polars_expr(output_type=Boolean)]
fn is_in_hashset(inputs: &[Series], kwargs: IsInHashSetKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    // SAFETY: We reconstruct the Arc from the pointer passed from Python.
    // This is safe because:
    // 1. Python holds a reference to StringHashSet, keeping the Arc alive
    // 2. We increment the Arc count to prevent double-free
    // 3. The RwLock ensures thread-safe access
    let set = unsafe { reconstruct_arc(kwargs.set_ptr) };

    let guard = set.inner.read().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire read lock: {}", e).into())
    })?;

    let out: BooleanChunked = ca
        .into_iter()
        .map(|opt_v| opt_v.map(|v| guard.contains(v)))
        .collect();

    Ok(out.into_series())
}

/// Kwargs for filter_and_update expression.
#[derive(Deserialize)]
struct FilterAndUpdateKwargs {
    set_ptr: usize,
}

/// Filter rows NOT in the set, then add those values to the set.
///
/// This implements an "anti-join with update" pattern:
/// - Returns a boolean mask where `true` indicates the value is NOT in the set
/// - Adds all non-matching (non-null) values to the set for future calls
///
/// This is perfect for deduplication across multiple LazyFrames:
/// ```python
/// seen = StringHashSet()
/// for lf in lazy_frames:
///     # Keep only rows we haven't seen before, and remember them
///     df = lf.filter(col("id").hashfilter.not_in_and_update(seen)).collect()
/// ```
#[polars_expr(output_type=Boolean)]
fn not_in_and_update(inputs: &[Series], kwargs: FilterAndUpdateKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    let set = unsafe { reconstruct_arc(kwargs.set_ptr) };

    // First pass: determine which values are NOT in the set (read lock)
    let results: Vec<Option<bool>> = {
        let guard = set.inner.read().map_err(|e| {
            PolarsError::ComputeError(format!("Failed to acquire read lock: {}", e).into())
        })?;

        ca.into_iter()
            .map(|opt_v| opt_v.map(|v| !guard.contains(v)))
            .collect()
    };

    // Second pass: add new values to the set (write lock)
    {
        let mut guard = set.inner.write().map_err(|e| {
            PolarsError::ComputeError(format!("Failed to acquire write lock: {}", e).into())
        })?;

        for (opt_v, opt_result) in ca.into_iter().zip(results.iter()) {
            if let (Some(v), Some(true)) = (opt_v, opt_result) {
                guard.insert(v.to_string());
            }
        }
    }

    let out: BooleanChunked = results.into_iter().collect();
    Ok(out.into_series())
}

/// Check if values in the input column do NOT exist in the StringHashSet.
///
/// Returns a boolean mask where `true` indicates the value does NOT exist in the set.
/// This is a read-only operation on the set.
#[polars_expr(output_type=Boolean)]
fn not_in_hashset(inputs: &[Series], kwargs: IsInHashSetKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    let set = unsafe { reconstruct_arc(kwargs.set_ptr) };

    let guard = set.inner.read().map_err(|e| {
        PolarsError::ComputeError(format!("Failed to acquire read lock: {}", e).into())
    })?;

    let out: BooleanChunked = ca
        .into_iter()
        .map(|opt_v| opt_v.map(|v| !guard.contains(v)))
        .collect();

    Ok(out.into_series())
}

/// Reconstruct an Arc<StringHashSet> from a raw pointer.
///
/// # Safety
/// The caller must ensure:
/// - The pointer came from a valid StringHashSet._ptr() call
/// - The original StringHashSet is still alive (Python holds a reference)
unsafe fn reconstruct_arc(ptr: usize) -> StringHashSet {
    use std::sync::{Arc, RwLock};

    let arc_ptr = ptr as *const RwLock<polars::prelude::PlHashSet<String>>;
    // Increment the reference count so we don't double-free
    Arc::increment_strong_count(arc_ptr);
    let inner = Arc::from_raw(arc_ptr);

    StringHashSet { inner }
}
