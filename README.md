# polars-matmul

High-performance similarity search for Polars.

[![PyPI](https://img.shields.io/pypi/v/polars-matmul.svg)](https://pypi.org/project/polars-matmul/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why?

Computing similarity between embedding vectors in Polars is slow:

| Approach | Time (1000×10000, 256d) | Memory |
|----------|-----------------------|--------|
| Polars cross-join + list ops | > 10 min | OOM (>32GB) |
| NumPy matmul + argpartition | ~75ms | ~800MB |
| **polars-matmul** | **~40ms** | **~160MB** |

This plugin provides efficient similarity search by:
- Using `faer`, a pure Rust high-performance linear algebra library
- Avoiding cross-join memory explosion
- Operating directly on contiguous arrays
- Compiling on all platforms (Linux, macOS, Windows) without complex dependencies

## Installation

```bash
pip install polars-matmul
```

Pre-built wheels are available for:
- macOS (x86_64, arm64)
- Linux (x86_64, aarch64)
- Windows (x86_64)

## Usage

Import the package to register the `.pmm` namespace on Polars expressions:

```python
import polars as pl
import polars_matmul  # registers the .pmm namespace
```

### Top-K Similarity Search

Find the top-k most similar items from a corpus for each query:

```python
queries = pl.DataFrame({
    "query_id": [0, 1, 2],
    "embedding": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
})

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

# Find top-2 similar items per query
result = queries.with_columns(
    pl.col("embedding").pmm.topk(corpus["embedding"], k=2).alias("matches")
)

print(result)
# ┌──────────┬─────────────────┬───────────────────────────────┐
# │ query_id │ embedding       │ matches                       │
# │ ---      │ ---             │ ---                           │
# │ i64      │ list[f64]       │ list[struct[2]]               │
# ╞══════════╪═════════════════╪═══════════════════════════════╡
# │ 0        │ [1.0, 0.0, 0.0] │ [{0, 0.994...}, {3, 0.707...}]│
# │ 1        │ [0.0, 1.0, 0.0] │ [{1, 0.994...}, {3, 0.707...}]│
# │ 2        │ [0.0, 0.0, 1.0] │ [{2, 0.994...}, {4, 0.707...}]│
# └──────────┴─────────────────┴───────────────────────────────┘
```

### Explode and Join Pattern

Flatten the results and join with corpus metadata:

```python
# Explode to get flat results, then join with corpus
flat_results = (
    queries
    .with_columns(
        pl.col("embedding").pmm.topk(corpus["embedding"], k=2).alias("match")
    )
    .explode("match")
    .unnest("match")
    .join(corpus.with_row_index("index"), on="index")
)

print(flat_results)
# ┌──────────┬───────┬───────────┬───────┐
# │ query_id │ score │ corpus_id │ label │
# │ ---      │ ---   │ ---       │ ---   │
# │ i64      │ f64   │ i64       │ str   │
# ╞══════════╪═══════╪═══════════╪═══════╡
# │ 0        │ 0.99  │ 0         │ a     │
# │ 0        │ 0.71  │ 3         │ ab    │
# │ 1        │ 0.99  │ 1         │ b     │
# │ ...      │ ...   │ ...       │ ...   │
# └──────────┴───────┴───────────┴───────┘
```

### Full Similarity Matrix

Compute all pairwise dot products (no top-k filtering):

```python
result = queries.with_columns(
    pl.col("embedding").pmm.matmul(corpus["embedding"]).alias("scores")
)
# Each row has a list of len(corpus) scores
```

### Metrics

The `metric` parameter controls the similarity function:

```python
# Cosine similarity (default) - best for normalized embeddings
pl.col("embedding").pmm.topk(corpus_emb, k=5, metric="cosine")

# Dot product - for pre-normalized vectors
pl.col("embedding").pmm.topk(corpus_emb, k=5, metric="dot")

# Euclidean distance - lower is more similar
pl.col("embedding").pmm.topk(corpus_emb, k=5, metric="euclidean")
```

| Metric | Description | Best for |
|--------|-------------|----------|
| `"cosine"` | Cosine similarity (default) | Normalized embeddings, NLP |
| `"dot"` | Raw dot product | Pre-normalized vectors |
| `"euclidean"` | Euclidean (L2) distance | Spatial data, clustering |

### Float32 Support

For 2x memory efficiency, use Float32 embeddings:

```python
# Cast to f32 for memory efficiency
queries_f32 = queries.with_columns(
    pl.col("embedding").cast(pl.List(pl.Float32))
)
corpus_f32 = corpus.with_columns(
    pl.col("embedding").cast(pl.List(pl.Float32))
)

# Works the same way - automatically uses f32 operations
result = queries_f32.with_columns(
    pl.col("embedding").pmm.topk(corpus_f32["embedding"], k=10).alias("matches")
)
```

### A Note on Float16

Float16 (`pl.Float16`) is **not supported** by polars-matmul. Here's why:

1. **No native CPU support** - Most CPUs lack f16 arithmetic, requiring conversion to f32 for all operations
2. **No memory benefit** - Since we must convert to f32 before computation, peak memory usage is identical to f32
3. **Polars marks Float16 as unstable** - The Polars team recommends casting to Float32 before computation

**Recommendation:** If your embeddings are stored as Float16 (e.g., in Parquet), cast them to Float32 when loading:

```python
# Load Float16 embeddings and convert to Float32 for computation
df = pl.read_parquet("embeddings.parquet").with_columns(
    pl.col("embedding").cast(pl.List(pl.Float32))
)
```

Float16 is excellent for *storage* (2x smaller files), but Float32 is optimal for *computation*.

### Performance Tip: Use Arrays

For best performance, use the `Array[f64, dim]` or `Array[f32, dim]` type instead of `List`. The fixed-width Array type allows for zero-copy buffer extraction:

```python
# Convert List to Array for ~3x faster loading
df = df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, 256)))
```

## API Reference

### `pl.col(...).pmm.topk(corpus, k, metric="cosine")`

Find top-k similar items from corpus for each embedding.

**Parameters:**
- `corpus`: `pl.Series` - Corpus embeddings (List or Array type)
- `k`: `int` - Number of top matches per query
- `metric`: `str` - Similarity metric ("cosine", "dot", "euclidean")

**Returns:** Expression evaluating to `List[Struct{index: u32, score: f64}]`

### `pl.col(...).pmm.matmul(corpus)`

Compute all pairwise dot products.

**Parameters:**
- `corpus`: `pl.Series` - Corpus embeddings (List or Array type)

**Returns:** Expression evaluating to `Array[f64, N]` or `Array[f32, N]` where N = len(corpus)

## Performance

`polars-matmul` is designed to be significantly faster and more memory-efficient than pure Polars implementations. For end-to-end similarity search, it outperforms NumPy by performing the Top-K selection in Rust, avoiding expensive data materialization in Python.

| Operation (1000×10000, 256d, k=10) | NumPy | **polars-matmul** | Ratio |
|-----------------------------------|-------|-------------------|-------|
| **Top-K Similarity** (End-to-End)  | ~73ms | **~45ms**         | **0.64x** |
| Raw Matmul f32 (micro-benchmark)  | ~5ms  | ~11ms             | 2.1x |
| Raw Matmul f64 (micro-benchmark)  | ~17ms | ~22ms             | 1.3x |

> **Analysis**: The raw matmul overhead comes from Polars Series construction. For end-to-end top-k search, `polars-matmul` is **1.6x faster** than NumPy by fusing normalization, multiplication, and selection into a single optimized Rust pass.

## Benchmarking

Run the benchmarks yourself:

```bash
# Benchmark raw matrix multiplication (micro-benchmark)
python examples/benchmark_matmul.py

# Benchmark end-to-end similarity search (primary use case)
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
- [x] **Cross-platform** - Pure Rust implementation (via faer) supports Linux, macOS, and Windows
- [x] **Polars Expression API** - Native `pl.col("embedding").pmm.topk(...)` syntax
- [x] **Float16 note** - Documented why f16 is not supported (no CPU support, no memory benefit)
- [x] **Zero-copy input extraction** - For Array types, data is read directly without copying
- [x] **Direct faer views** - Use `faer::mat::from_raw_parts` directly on Polars' memory
- [x] **Efficient output construction** - Use ListPrimitiveChunkedBuilder for minimal allocations
- [ ] **Zero-copy output** - Write faer output directly into Arrow buffer (further optimization)

## License

MIT
