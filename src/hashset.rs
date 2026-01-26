//! Python-accessible StringHashSet wrapper using Arc<RwLock> for zero-copy sharing.

use polars::prelude::{InitHashMaps, PlHashSet};
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};

/// A thread-safe wrapper around PlHashSet<String> that can be shared between Python and Rust
/// without copying the underlying data.
///
/// The set is stored behind Arc<RwLock> allowing:
/// - Multiple readers OR one writer at a time
/// - Zero-copy when passed to Polars expressions
/// - Thread-safe concurrent access from multiple LazyFrames
#[pyclass]
#[derive(Clone)]
pub struct StringHashSet {
    pub(crate) inner: Arc<RwLock<PlHashSet<String>>>,
}

#[pymethods]
impl StringHashSet {
    /// Create a new empty StringHashSet.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(PlHashSet::new())),
        }
    }

    /// Create a new StringHashSet with pre-allocated capacity.
    #[staticmethod]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(PlHashSet::with_capacity(capacity))),
        }
    }

    /// Create a StringHashSet from an iterable of strings.
    /// This is the only place where we copy data from Python to Rust.
    #[staticmethod]
    pub fn from_values(values: Vec<String>) -> Self {
        let set: PlHashSet<String> = values.into_iter().collect();
        Self {
            inner: Arc::new(RwLock::new(set)),
        }
    }

    /// Insert a value into the set. Returns true if the value was newly inserted.
    pub fn insert(&self, value: String) -> PyResult<bool> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(guard.insert(value))
    }

    /// Check if a value exists in the set.
    pub fn contains(&self, value: &str) -> PyResult<bool> {
        let guard = self
            .inner
            .read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(guard.contains(value))
    }

    /// Remove a value from the set. Returns true if the value was present.
    pub fn remove(&self, value: &str) -> PyResult<bool> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(guard.remove(value))
    }

    /// Get the number of elements in the set.
    pub fn __len__(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(guard.len())
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> PyResult<bool> {
        let guard = self
            .inner
            .read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(guard.is_empty())
    }

    /// Clear all elements from the set.
    pub fn clear(&self) -> PyResult<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        guard.clear();
        Ok(())
    }

    /// Return all values as a Python list.
    /// This is one of the few places where we copy data from Rust to Python.
    pub fn to_list(&self) -> PyResult<Vec<String>> {
        let guard = self
            .inner
            .read()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(guard.iter().cloned().collect())
    }

    /// String representation showing length.
    pub fn __repr__(&self) -> PyResult<String> {
        let len = self.__len__()?;
        Ok(format!("StringHashSet(len={})", len))
    }

    /// Extend the set with values from an iterable.
    /// Copies values from Python to Rust.
    pub fn extend(&self, values: Vec<String>) -> PyResult<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        guard.extend(values);
        Ok(())
    }

    /// Get the memory address of the underlying set (for debugging zero-copy behavior).
    pub fn _ptr(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
}

impl Default for StringHashSet {
    fn default() -> Self {
        Self::new()
    }
}
