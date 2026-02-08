# polars-matmul

High-performance similarity search for Polars.

[![PyPI](https://img.shields.io/pypi/v/polars-matmul.svg)](https://pypi.org/project/polars-matmul/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why?

Computing similarity between embedding vectors in Polars is slow:

| Approach | Time (1000×10000, 256d) | Memory |
|----------|-------------------------|--------|
| Polars cross-join + list ops | > 10 min | OOM (>32GB) |
| NumPy matmul + argpartition | ~75ms | ~800MB |
| **polars-matmul** | **~45ms** | **~160MB** |

`polars-matmul` provides efficient similarity search by:
- Using **faer**, a pure Rust high-performance linear algebra library
- Avoiding cross-join memory explosion
- Operating directly on contiguous arrays via zero-copy extraction
- Compiling on all platforms without complex dependencies

## Installation

```bash
pip install polars-matmul
```

Pre-built wheels available for macOS, Linux, and Windows (x86_64 and arm64).

## Quick Start

```python
import polars as pl
import polars_matmul  # registers the .pmm namespace

# Sample data
queries = pl.DataFrame({
    "id": [0, 1, 2],
    "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
})

corpus = pl.DataFrame({
    "embedding": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]],
    "label": ["a", "b", "c"],
})

# Find top-2 similar items per query
result = queries.with_columns(
    pl.col("embedding").pmm.topk(corpus["embedding"], k=2).alias("matches")
)
```

**Output:**
```
┌─────┬─────────────────┬───────────────────────────────┐
│ id  │ embedding       │ matches                       │
│ i64 │ list[f64]       │ list[struct[2]]               │
╞═════╪═════════════════╪═══════════════════════════════╡
│ 0   │ [1.0, 0.0, 0.0] │ [{0, 0.994}, {1, 0.110}]      │
│ 1   │ [0.0, 1.0, 0.0] │ [{1, 0.994}, {0, 0.110}]      │
│ 2   │ [0.0, 0.0, 1.0] │ [{2, 0.994}, {1, 0.110}]      │
└─────┴─────────────────┴───────────────────────────────┘
```

## API Reference

### `topk(corpus, k, metric="cosine")`

Find top-k similar items for each embedding.

```python
pl.col("embedding").pmm.topk(corpus["embedding"], k=10)
```

**Parameters:**
- `corpus`: Series of embeddings
- `k`: Number of results per query
- `metric`: `"cosine"` (default), `"dot"`, or `"euclidean"`

**Returns:** `List[Struct{index: u32, score: f64}]`

### `matmul(corpus, flatten=False)`

Compute all pairwise dot products.

```python
# Default: Array per row
pl.col("embedding").pmm.matmul(corpus["embedding"])

# Flatten: 1D array (for NumPy interop)
pl.col("embedding").pmm.matmul(corpus["embedding"], flatten=True)
```

**Parameters:**
- `corpus`: Series of embeddings
- `flatten`: If True, returns flat 1D series instead of per-row arrays

**Returns:** `Array[f64, N]` or `Array[f32, N]` (where N = len(corpus))

---

## Common Patterns

### Explode and Join

Flatten results and join with corpus metadata:

```python
flat_results = (
    queries
    .with_columns(pl.col("embedding").pmm.topk(corpus["embedding"], k=2).alias("match"))
    .explode("match")
    .unnest("match")
    .join(corpus.with_row_index("index"), on="index")
)
```

---

### Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `"cosine"` | Cosine similarity (default) | Text embeddings |
| `"dot"` | Raw dot product | Pre-normalized vectors |
| `"euclidean"` | L2 distance (lower = more similar) | Clustering |

---

## Performance Tips

### 1. Use Array Type (2.4x faster)

Fixed-size `Array` enables zero-copy extraction:

```python
# Convert List to Array for best performance
df = df.with_columns(
    pl.col("embedding").cast(pl.Array(pl.Float32, 256))
)
```

| Input Type | vs NumPy |
|------------|----------|
| `pl.Array` | 2.1x slower |
| `pl.List` | 5.0x slower |

### 2. Use Float32 (2x memory savings)

```python
df = df.with_columns(
    pl.col("embedding").cast(pl.Array(pl.Float32, dim))
)
```

### 3. Float16 Not Recommended

Float16 requires conversion to Float32 for computation (no CPU support), so there's no performance benefit. Use Float16 for *storage* only.

---

## Performance Benchmarks

Benchmarked on 1000 queries × 10000 corpus, 256 dimensions:

| Operation | NumPy | polars-matmul | Ratio |
|-----------|-------|---------------|-------|
| **Top-K Search (k=10)** | 73ms | **45ms** | **0.64x** ✅ |
| Raw Matmul (f32, Array) | 5ms | 11ms | 2.1x |
| Raw Matmul (f64, Array) | 17ms | 22ms | 1.3x |

> **Why Top-K is faster:** polars-matmul fuses normalization, multiplication, and selection into a single Rust pass, avoiding Python data materialization.

Run benchmarks yourself:
```bash
python examples/benchmark_topk.py    # End-to-end search
python examples/benchmark_matmul.py  # Raw matmul
```

---

## Development

```bash
git clone https://github.com/NivekNey/polars-matmul
cd polars-matmul

python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy pyarrow

maturin develop --release
pytest tests/
```

## License

MIT
