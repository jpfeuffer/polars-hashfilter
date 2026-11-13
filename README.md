# polars-hashfilter

A Polars plugin exposing `PlHashSet` for efficient filtering with persistent sets across LazyFrames.

## Features

- **Shared `StringHashSet`**: A Python-accessible wrapper around Polars' `PlHashSet<String>` that shares the set between Python and Rust. Each distinct input string is copied into owned Rust storage; duplicates do not allocate another string.
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

Requires Python 3.11+ and Python Polars 2.1.0. Rust bindings are built against
Polars 0.55.2 / pyo3-polars 0.28. Rebuild the extension after changing Rust code.

```bash
# From source with uv
uv sync --group dev

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

**Use rowwise semantics when**: You want one retained occurrence per distinct value.
Within each call rows are sequential, but parallel streaming batches can acquire
the set lock out of source order. The selected occurrence is not guaranteed to be
the first source row. Duplicate counts and uniqueness booleans are order-independent.

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
filtered = filtered.collect(engine="streaming")
filtered.select(pl.col("id").hashfilter.update(seen))  # Explicit update
```

This two-step pattern materializes the filtered rows before updating the set;
merely constructing a lazy expression does not execute its side effects.

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

### Exact streaming uniqueness and query-scoped state

```python
from pathlib import Path
import polars as pl
from polars_hashfilter import StringHashSet, update_bool

def build_validation(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Create fresh state for each independent execution.
    seen = StringHashSet()
    return lf.select((~update_bool("id", seen)).any().alias("has_duplicates"))

lf = pl.scan_parquet("input.parquet")
check = build_validation(lf)
sink = lf.sink_parquet(Path("output.parquet"), lazy=True)
validation, _ = pl.collect_all([check, sink], engine="streaming")
has_duplicates = validation.item()
```

Nulls are explicitly **ignored**: they produce null masks, are never inserted,
and `any()` ignores them. Empty and all-null inputs have no duplicates under this
policy. Built-in `n_unique` and `is_first_distinct` include null as a distinct
value; drop nulls when comparing their behavior to this plugin.

Expressions own their Python set through a pure identity `map_batches` callback.
The callback runs once per evaluated batch, returns the input unchanged, and
adds no per-row Python calls. Native kernels use checked process-local weak
registry handles, not dereferenced raw addresses. Dropping the last owner/plan
releases the set and unregisters its handle. `_ptr()` is diagnostic only.
Plans containing this state are process-local, not a serialization/distributed
execution API.

State is persistent by design. Re-collecting `check` above reuses the already
populated set and therefore marks previously inserted values as duplicates.
Call `build_validation(lf)` again for each independent execution. `seen.clear()`
is an explicit alternative only when no query using that set is running.
Do not reset state inside a per-batch callback.
Only evaluated expressions have side effects; schema inspection, unused
projections, or rows pruned by slicing are not a full-source validation.

All plugin operations, including lookups against mutable external state, are
registered with `is_deterministic=False` to prevent common-subexpression/subplan
reuse of stateful results. Do not expect two branches using the same mutating
expression/set to receive independent checks or an identical cached row mask.
Keep validation on its own branch and share the **unmodified source** with the
sink. In the regression test, an instrumented source processes 60,002 rows
exactly once across validation and a lazy Parquet sink in `collect_all`.
This is verified on the pinned version, not a general optimizer guarantee.

The sink runs even if validation returns `True`; `collect_all` is not a
transaction or a conditional publication mechanism. Use a staged output path
and publish it only after successful validation in production code.

### Measured streaming benchmark

Run `.venv/bin/python benchmarks/streaming_uniqueness.py` after building.
The runner generates synthetic local Parquet files in a temporary directory
(`row_group_size=20000`) in separate generation processes and runs every
method/thread/prefix combination in a fresh process. Each file contains
200,000 all-distinct strings: `"x" * prefix_length + str(index)`.

The table gives absolute peak RSS in MiB / elapsed query seconds from one run on
macOS arm64, CPython 3.14.10, Polars 2.1.0, release extension. The roughly 60 MiB
import/interpreter baseline is included. Timings exclude imports and fixture
generation; files may be in the OS cache. These are single-run observations,
not stable throughput guarantees.

| Prefix | Threads | Eager `is_duplicated().any()` | Streaming `n_unique()!=len()` | Streaming `(~is_first_distinct()).any()` | Plugin `(~update_bool()).any()` |
|---|---:|---:|---:|---:|---:|
| 100 | 1 | 108 / 0.022 | 137 / 0.018 | 108 / 0.015 | 132 / 0.027 |
| 100 | 12 | 119 / 0.016 | 166 / 0.011 | 140 / 0.013 | 145 / 0.023 |
| 1000 | 1 | 284 / 0.081 | 531 / 0.091 | 318 / 0.049 | 366 / 0.108 |
| 1000 | 12 | 294 / 0.068 | 680 / 0.045 | 484 / 0.044 | 495 / 0.106 |
| 4000 | 1 | 866 / 0.235 | 1977 / 0.343 | 991 / 0.209 | 1163 / 0.317 |
| 4000 | 12 | 876 / 0.200 | 2390 / 0.157 | 1320 / 0.161 | 1346 / 0.295 |

A preceding fresh-process matrix gave 1750 / 0.344 for the 4000-character,
one-thread count case and 1063 / 0.331 for the plugin. In-flight buffering and
allocation timing can change peak RSS even with the same thread count.

The plugin is exact across batches and uses only distinct string-key state
(plus hash-table overhead), not a general group-by. That still requires
O(total distinct key bytes) memory, with no spill-to-disk. The write lock
serializes insertions; owned strings and in-flight scan buffers dominate the
long-string all-distinct case. It substantially lowers peak RSS relative to the
streaming count approach here, but **does not improve on `is_first_distinct`**
and is slower. There is no performance basis from this workload to replace
the built-in first-distinct check with this plugin.

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
   - Inserting a previously unseen input string into owned Rust storage

This shares the set, not the input string buffers. Expression ownership also
requires the per-batch Python identity callback described above.

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
