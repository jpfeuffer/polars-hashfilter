//! Python-accessible StringHashSet wrapper using Arc<RwLock> for zero-copy sharing.

use polars::prelude::{InitHashMaps, PlHashSet};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

type SharedSet = RwLock<PlHashSet<String>>;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static SETS: OnceLock<Mutex<BTreeMap<u64, Weak<SharedSet>>>> = OnceLock::new();

fn registry() -> &'static Mutex<BTreeMap<u64, Weak<SharedSet>>> {
    SETS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(crate) fn lookup(id: u64) -> polars::prelude::PolarsResult<Arc<SharedSet>> {
    use polars::prelude::PolarsError;
    let sets = registry()
        .lock()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    sets.get(&id).and_then(Weak::upgrade).ok_or_else(|| {
        PolarsError::ComputeError(
            "StringHashSet handle expired or belongs to another process".into(),
        )
    })
}

pub(crate) fn insert_borrowed(set: &mut PlHashSet<String>, value: &str) -> bool {
    let mut inserted = false;
    set.get_or_insert_with(value, |value| {
        inserted = true;
        value.to_owned()
    });
    inserted
}

/// A thread-safe wrapper around PlHashSet<String> that can be shared between Python and Rust
/// without copying the underlying data.
///
/// The set is stored behind Arc<RwLock> allowing:
/// - Multiple readers OR one writer at a time
/// - Zero-copy when passed to Polars expressions
/// - Thread-safe concurrent access from multiple LazyFrames
#[pyclass(skip_from_py_object, weakref)]
pub struct StringHashSet {
    pub(crate) inner: Arc<RwLock<PlHashSet<String>>>,
    id: u64,
}

impl StringHashSet {
    fn from_set(set: PlHashSet<String>) -> Self {
        let inner = Arc::new(RwLock::new(set));
        let id = NEXT_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("StringHashSet handle space exhausted");
        registry()
            .lock()
            .expect("StringHashSet registry poisoned")
            .insert(id, Arc::downgrade(&inner));
        Self { inner, id }
    }
}

impl Drop for StringHashSet {
    fn drop(&mut self) {
        registry()
            .lock()
            .expect("StringHashSet registry poisoned")
            .remove(&self.id);
    }
}

#[pymethods]
impl StringHashSet {
    /// Create a new empty StringHashSet.
    #[new]
    pub fn new() -> Self {
        Self::from_set(PlHashSet::new())
    }

    /// Create a new StringHashSet with pre-allocated capacity.
    #[staticmethod]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_set(PlHashSet::with_capacity(capacity))
    }

    /// Create a StringHashSet from an iterable of strings.
    /// This is the only place where we copy data from Python to Rust.
    #[staticmethod]
    pub fn from_values(values: Vec<String>) -> Self {
        let set: PlHashSet<String> = values.into_iter().collect();
        Self::from_set(set)
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

    pub fn _id(&self) -> u64 {
        self.id
    }
}

impl Default for StringHashSet {
    fn default() -> Self {
        Self::new()
    }
}
