"""Type stubs for the Rust extension module."""

from typing import List

__version__: str

class StringHashSet:
    """A thread-safe wrapper around PlHashSet<String> for zero-copy sharing with Polars.

    The set is stored behind Arc<RwLock> allowing:
    - Multiple readers OR one writer at a time
    - Zero-copy when passed to Polars expressions
    - Thread-safe concurrent access from multiple LazyFrames
    """

    def __new__(cls) -> StringHashSet:
        """Create a new empty StringHashSet."""
        ...

    @staticmethod
    def with_capacity(capacity: int) -> StringHashSet:
        """Create a new StringHashSet with pre-allocated capacity."""
        ...

    @staticmethod
    def from_values(values: List[str]) -> StringHashSet:
        """Create a StringHashSet from a list of strings.

        This is one of the few places where data is copied from Python to Rust.
        """
        ...

    def insert(self, value: str) -> bool:
        """Insert a value into the set. Returns True if the value was newly inserted."""
        ...

    def contains(self, value: str) -> bool:
        """Check if a value exists in the set."""
        ...

    def remove(self, value: str) -> bool:
        """Remove a value from the set. Returns True if the value was present."""
        ...

    def __len__(self) -> int:
        """Get the number of elements in the set."""
        ...

    def is_empty(self) -> bool:
        """Check if the set is empty."""
        ...

    def clear(self) -> None:
        """Clear all elements from the set."""
        ...

    def to_list(self) -> List[str]:
        """Return all values as a Python list.

        This is one of the few places where data is copied from Rust to Python.
        """
        ...

    def __repr__(self) -> str:
        """String representation showing length."""
        ...

    def extend(self, values: List[str]) -> None:
        """Extend the set with values from a list.

        Copies values from Python to Rust.
        """
        ...

    def _ptr(self) -> int:
        """Get the memory address of the underlying set (for internal use)."""
        ...
