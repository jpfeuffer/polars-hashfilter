# polars-hashfilter

A Polars plugin exposing `PlHashSet` for efficient filtering with persistent sets across LazyFrames.

## Features

- **Zero-copy `StringHashSet`**: A Python-accessible wrapper around Polars' `PlHashSet<String>` that can be shared between Python and Rust without copying the underlying data
- **Persistent filtering**: Use the same hashset across multiple LazyFrames for deduplication
- **Streaming compatible**: All expressions work in streaming mode with `engine="streaming"`
- **Seven filtering/update expressions**:
  - `is_in`: Check if values exist in the set (read-only)
  - `not_in`: Check if values do NOT exist in the set (read-only)
  - `not_in_and_update`: Check if values are NOT in the set, then add them (batch semantics)
  - `not_in_and_update_rowwise`: Same as above but with row-by-row semantics
  - `update`: Bulk insert all values into the set (returns null)
  - `update_chain`: Bulk insert all values, return original Series (for chaining)
  - `update_bool`: Insert values row-by-row, return bool indicating if newly inserted

## Installation

```bash
# From source with uv
uv pip install .

# Development mode
just dev
```

## Usage

### Basic Example

```python
import polars as pl
from polars_hashfilter import StringHashSet

# Create a persistent set
seen = StringHashSet.from_values(["alice", "bob"])

# Filter using the set
df = pl.DataFrame({"user": ["alice", "charlie", "bob", "dave"]})

# Using standalone functions
from polars_hashfilter import is_in_hashset, not_in_hashset

df.filter(is_in_hashset(pl.col("user"), seen))
# shape: (2, 1)
# ┌───────┐
# │ user  │
# │ ---   │
# │ str   │
# ╞═══════╡
# │ alice │
# │ bob   │
# └───────┘

# Using expression namespace
df.filter(pl.col("user").hashfilter.not_in(seen))
# shape: (2, 1)
# ┌─────────┐
# │ user    │
# │ ---     │
# │ str     │
# ╞═════════╡
# │ charlie │
# │ dave    │
# └─────────┘
```

### Deduplication Across Multiple LazyFrames (Anti-Join Pattern)

This is the primary use case - efficiently deduplicate records across many large LazyFrames:

```python
import polars as pl
from polars_hashfilter import StringHashSet

# Create a persistent set to track seen IDs
seen = StringHashSet()

# Process multiple LazyFrames, keeping only new records
lazy_frames = [
    pl.LazyFrame({"id": ["a", "b"], "value": [1, 2]}),
    pl.LazyFrame({"id": ["b", "c"], "value": [3, 4]}),
    pl.LazyFrame({"id": ["c", "d"], "value": [5, 6]}),
]

for lf in lazy_frames:
    # Keep only rows we haven't seen before, and remember them
    df = lf.filter(pl.col("id").hashfilter.not_in_and_update(seen)).collect()
    print(df)
    # First:  id=["a", "b"], value=[1, 2]
    # Second: id=["c"],      value=[4]      (b already seen)
    # Third:  id=["d"],      value=[6]      (c already seen)

# The set now contains all unique IDs
print(seen.to_list())  # ["a", "b", "c", "d"]
```

### Batch vs Rowwise Semantics

**IMPORTANT**: When processing data with duplicates, you must choose the right semantics:

#### Batch Semantics (`not_in_and_update` - default)
All rows are evaluated against the **initial** set state, then new values are inserted. Duplicates within the same batch all return `True`.

```python
df = pl.DataFrame({"id": ["a", "b", "a", "c", "b"]})

seen = StringHashSet()
result = df.filter(pl.col("id").hashfilter.not_in_and_update(seen))

print(result["id"].to_list())
# Output: ["a", "b", "a", "c", "b"]  <- All instances kept!
print(seen.to_list())
# Output: ["a", "b", "c"]  <- Only unique values stored
```

**Use batch semantics when**: You want to keep all instances of new values in the first occurrence batch, or when processing separate LazyFrames.

#### Rowwise Semantics (`not_in_and_update_rowwise`)
Rows are evaluated **sequentially**. First occurrence returns `True`, subsequent duplicates return `False`.

```python
df = pl.DataFrame({"id": ["a", "b", "a", "c", "b"]})

seen = StringHashSet()
result = df.filter(pl.col("id").hashfilter.not_in_and_update_rowwise(seen))

print(result["id"].to_list())
# Output: ["a", "b", "c"]  <- Only first occurrence kept!
print(seen.to_list())
# Output: ["a", "b", "c"]  <- Only unique values stored
```

**Use rowwise semantics when**: You explicitly want to deduplicate within the same batch/DataFrame, keeping only the first occurrence.

#### ⚠️ Streaming Mode Warning

In streaming mode (`collect(engine="streaming")`), Polars processes data in **chunks**. Each chunk triggers a separate expression call, which means:

- **Batch semantics only apply WITHIN each chunk**
- Across chunk boundaries, values from previous chunks are already in the set
- For large data split across chunks, `not_in_and_update` effectively behaves like `not_in_and_update_rowwise` at chunk boundaries

**Recommended patterns for streaming:**
- Use `not_in_and_update_rowwise` for consistent row-by-row deduplication
- OR use `not_in` (read-only lookup) + `update` (explicit insert) for full control:

```python
seen = StringHashSet()

# Pattern: Separate lookup from update
filtered = lf.filter(pl.col("id").hashfilter.not_in(seen))
filtered.select(pl.col("id").hashfilter.update(seen))  # Explicit update
```

### Update Functions

For explicit control over when values are added to the set:

```python
# update: Bulk insert, returns null (side-effect only)
df.select(pl.col("id").hashfilter.update(seen))

# update_chain: Bulk insert, returns original Series (for chaining)
df.select(
    pl.col("id")
    .hashfilter.update_chain(seen)
    .str.to_uppercase()  # Can chain other operations
)

# update_bool: Row-by-row insert, returns bool (True if newly inserted)
result = df.select(pl.col("id").hashfilter.update_bool(seen))
# First occurrence: True, duplicates: False
```

### StringHashSet API

```python
from polars_hashfilter import StringHashSet

# Creation
s = StringHashSet()                      # Empty set
s = StringHashSet.with_capacity(1000)    # Pre-allocated
s = StringHashSet.from_values(["a", "b"])  # From iterable

# Operations
s.insert("value")      # Insert, returns True if new
s.contains("value")    # Check membership
s.remove("value")      # Remove, returns True if existed
s.extend(["x", "y"])   # Bulk insert
s.clear()              # Remove all elements

# Inspection
len(s)                 # Number of elements
s.is_empty()           # Check if empty
s.to_list()            # Export as Python list (copies data)

# Debug
s._ptr()               # Memory address (for verifying zero-copy)
```

## Zero-Copy Guarantee

The `StringHashSet` is stored behind `Arc<RwLock>`, meaning:

1. **No copies when passing to expressions**: The set's memory address remains stable
2. **Thread-safe**: Multiple readers OR one writer at a time
3. **Copies only when necessary**:
   - `StringHashSet.from_values()` - copying Python strings to Rust
   - `StringHashSet.extend()` - copying Python strings to Rust
   - `StringHashSet.to_list()` - copying Rust strings to Python

You can verify zero-copy behavior:

```python
seen = StringHashSet()
ptr1 = seen._ptr()

# Use in many expressions...
df.filter(pl.col("id").hashfilter.not_in_and_update(seen))

ptr2 = seen._ptr()
assert ptr1 == ptr2  # Same memory address
```

## Development

```bash
# Setup
just venv

# Build (debug)
just dev

# Build (release)
just release

# Test
just test

# Format
just fmt

# Lint
just lint
```

## License

MIT
