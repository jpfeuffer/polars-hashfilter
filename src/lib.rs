//! Polars plugin exposing PlHashSet for efficient filtering with persistent sets.
//!
//! This crate provides:
//! - `StringHashSet`: A Python-accessible wrapper around PlHashSet<String>
//! - `is_in_hashset`: Filter rows where column values exist in the set
//! - `filter_and_update`: Filter rows and add non-matching values to the set (antijoin pattern)

mod expressions;
mod hashset;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

pub use hashset::StringHashSet;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<StringHashSet>()?;
    Ok(())
}
