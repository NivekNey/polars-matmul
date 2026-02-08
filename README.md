# polars-matmul

High-performance similarity joins for Polars.

[![PyPI](https://img.shields.io/pypi/v/polars-matmul.svg)](https://pypi.org/project/polars-matmul/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why?

Computing similarity between embedding vectors in Polars is slow:

| Approach | Time (1000×10000, 256d) | Memory |
|----------|-----------------------|--------|
| Polars cross-join + list ops | > 10 min | OOM (>32GB) |
| NumPy matmul + argpartition | ~75ms | ~800MB |
| **polars-matmul** | **~40ms** | **~160MB** |

This plugin provides efficient matrix multiplication by:
- Using `faer`, a pure Rust high-performance linear algebra library
- Avoiding cross-join memory explosion
- Operating directly on contiguous arrays
- Compiles on all platforms (Linux, macOS, Windows) without complex external dependencies

## Installation

```bash
pip install polars-matmul
```

Pre-built wheels are available for:
- macOS (x86_64, arm64)
- Linux (x86_64, aarch64)
- Windows (x86_64)

## Usage

### Similarity Join

Find the top-k most similar items from a corpus for each query:

```python
import polars as pl
import polars_matmul as pmm

# Query embeddings
queries = pl.DataFrame({
    "query_id": [0, 1, 2],
    "embedding": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
})

# Corpus to search
corpus = pl.DataFrame({
    "corpus_id": [0, 1, 2, 3, 4],
    "embedding": [
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ],
    "label": ["a", "b", "c", "ab", "bc"],
})

# Find top-2 most similar corpus items for each query
result = pmm.similarity_join(
    left=queries,
    right=corpus,
    left_on="embedding",
    right_on="embedding",
    k=2,
    metric="cosine",  # or "dot", "euclidean"
)

print(result)
# ┌──────────┬───────────┬───────┬──────────┐
# │ query_id │ corpus_id │ label │ _score   │
# │ ---      │ ---       │ ---   │ ---      │
# │ i64      │ i64       │ str   │ f64      │
# ╞══════════╪═══════════╪═══════╪══════════╡
# │ 0        │ 0         │ a     │ 0.994... │
# │ 0        │ 3         │ ab    │ 0.707... │
# │ 1        │ 1         │ b     │ 0.994... │
# │ 1        │ 3         │ ab    │ 0.707... │
# │ 2        │ 2         │ c     │ 0.994... │
# │ 2        │ 4         │ bc    │ 0.707... │
# └──────────┴───────────┴───────┴──────────┘
```

### With LazyFrame

Works seamlessly with LazyFrame:

```python
result = pmm.similarity_join(
    left=queries.lazy(),
    right=corpus.lazy(),
    left_on="embedding",
    right_on="embedding",
    k=10,
)
# Returns LazyFrame
result.collect()
```

### With Filtering

Pre-filter your corpus before the similarity search:

```python
result = pmm.similarity_join(
    left=queries,
    right=corpus.filter(pl.col("label").is_in(["a", "b"])),
    left_on="embedding",
    right_on="embedding",
    k=5,
)
```

### Full Similarity Matrix

For computing all pairwise similarities (without top-k selection):

```python
# Returns List[f64] for each row - all dot products
similarities = pmm.matmul(queries["embedding"], corpus["embedding"])
# similarities[i] contains dot products of query i with all corpus vectors
```

### Float32 Support

For 2x memory efficiency, use Float32 embeddings. The library automatically detects the dtype and uses the appropriate routines:

```python
# Cast embeddings to f32 for memory efficiency
queries_f32 = queries.with_columns(
    pl.col("embedding").cast(pl.List(pl.Float32))
)
corpus_f32 = corpus.with_columns(
    pl.col("embedding").cast(pl.List(pl.Float32))
)

# Works the same way - automatically uses f32
result = pmm.similarity_join(
    left=queries_f32,
    right=corpus_f32,
    left_on="embedding",
    right_on="embedding",
    k=10,
)

# matmul also supports f32 - returns List[f32]
similarities_f32 = pmm.matmul(queries_f32["embedding"], corpus_f32["embedding"])
```

### Batch Processing

For very large corpuses that don't fit in memory, use the `batch_size` parameter to process in chunks:

```python
# Process 10,000 corpus items at a time
result = pmm.similarity_join(
    left=queries,
    right=large_corpus,  # millions of rows
    left_on="embedding",
    right_on="embedding",
    k=10,
    batch_size=10000,  # Reduces peak memory usage
)
```

The results are automatically merged across batches to give you the global top-k.


## Metrics

| Metric | Description | Best for |
|--------|-------------|----------|
| `"cosine"` | Cosine similarity (default) | Normalized embeddings, NLP |
| `"dot"` | Raw dot product | Pre-normalized vectors |
| `"euclidean"` | Euclidean (L2) distance | Spatial data, clustering |

## Performance

`polars-matmul` is designed to be significantly faster and more memory-efficient than pure Polars implementations. For end-to-end similarity joins, it even outperforms NumPy by performing the Top-K selection in Rust, avoiding expensive data materialization in Python.

| Operation (1000×10000, 256d, k=10) | NumPy | **polars-matmul** | Ratio |
|-----------------------------------|-------|-------------------|-------|
| **Similarity Join** (End-to-End)  | ~72ms | **~40ms**         | **0.55x** |
| Raw Matmul (Series micro-benchmark)| ~5ms  | ~21ms             | 4.20x |

> **Analysis**: While NumPy is faster at raw micro-benchmarks of binary matrix multiplication (due to data conversion overhead), `polars-matmul` wins on the end-to-end task because it fuses normalization, multiplication, and top-k selection into a single optimized Rust pass.

### Performance Tip: Use Arrays
For best performance, use the `Array[f64, dim]` or `Array[f32, dim]` type instead of `List`. The fixed-width Array type allows for zero-copy buffer extraction:

```python
# Convert List to Array for ~3x faster loading
df = df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, 256)))
```

## Benchmarking

You can run the systematic benchmarks yourself:

```bash
# Benchmark raw matrix multiplication (micro-benchmark)
python examples/benchmark_matmul.py

# Benchmark end-to-end similarity join (primary use case)
python examples/benchmark_topk.py
```

## Development

```bash
# Clone and setup
git clone https://github.com/NivekNey/polars-matmul
cd polars-matmul

# Create venv and install dev dependencies
python -m venv .venv
source .venv/bin/activate
pip install maturin

# Build and install in development mode
maturin develop --release

# Run tests
pip install pytest numpy pyarrow
pytest tests/
```

## Roadmap

Planned features (contributions welcome!):

- [x] **Float32 support** - Native f32 operations for 2x memory efficiency
- [x] **Batch processing** - Chunked computation for large datasets that don't fit in memory
- [x] **Cross-platform** - Pure Rust implementation (via faer) supports Linux, macOS, and Windows without BLAS complexity.
- [ ] **Polars Expression API** - More native `pl.col("embedding").pmm.topk(...)` syntax

## License

MIT

