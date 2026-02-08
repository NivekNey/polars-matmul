# polars-matmul

BLAS-accelerated similarity joins for Polars.

[![PyPI](https://img.shields.io/pypi/v/polars-matmul.svg)](https://pypi.org/project/polars-matmul/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why?

Computing similarity between embedding vectors in Polars is slow:

| Approach | Time (100×2000, 100d) | Memory |
|----------|----------------------|--------|
| Polars cross-join + list ops | ~1800ms | 7.8GB |
| NumPy matmul | ~2ms | 160MB |
| **polars-matmul** | ~1.3ms | 160MB |

This plugin provides NumPy-level performance by:
- Using BLAS-accelerated matrix multiplication (Accelerate on macOS, OpenBLAS on Linux)
- Avoiding cross-join memory explosion
- Operating directly on contiguous arrays

## Installation

```bash
pip install polars-matmul
```

Pre-built wheels are available for:
- macOS (x86_64, arm64)
- Linux (x86_64, aarch64)

> **Note**: Windows is not currently supported due to BLAS linking complexity. Contributions welcome!

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

For 2x memory efficiency, use Float32 embeddings. The library automatically detects the dtype and uses the appropriate BLAS routines (sgemm for f32, dgemm for f64):

```python
# Cast embeddings to f32 for memory efficiency
queries_f32 = queries.with_columns(
    pl.col("embedding").cast(pl.List(pl.Float32))
)
corpus_f32 = corpus.with_columns(
    pl.col("embedding").cast(pl.List(pl.Float32))
)

# Works the same way - automatically uses f32 BLAS
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

Benchmarks on Apple M1 (using Accelerate framework):

| Query × Corpus × Dim | NumPy | polars-matmul | Ratio |
|---------------------|-------|---------------|-------|
| 100 × 2,000 × 100   | 2.38ms | 1.49ms | **0.63x** (faster) |
| 100 × 2,000 × 1,000 | 6.00ms | 7.36ms | 1.23x |
| 1,000 × 10,000 × 100| 81.16ms | 34.13ms | **0.42x** (2.4x faster) |

> **Tip**: For best performance, use `Array[f64, dim]` type instead of `List[f64]`:
>
> ```python
> # Convert List to Array for 3x faster extraction
> df = df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float64, 128)))
> ```
>
> | Input Type | Time | Overhead |
> |------------|------|----------|
> | Array[f64, dim] | 0.21ms | **1.09x** |
> | List[f64] | 0.68ms | 3.62x |
>
> The Array type allows zero-copy extraction since values are stored contiguously.

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
- [ ] **Polars Expression API** - More native `pl.col("embedding").pmm.topk(...)` syntax
- [ ] **Windows support** - Pre-built wheels for Windows (blocked by BLAS linking complexity)

## License

MIT

