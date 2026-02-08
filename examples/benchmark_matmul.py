#!/usr/bin/env python3
"""
Benchmark: Raw Matrix Multiplication (polars-matmul vs NumPy)

Measures the core matmul operation: Query @ Corpus.T
"""

import time
import numpy as np
import polars as pl
import polars_matmul  # noqa: F401 - registers the .pmm namespace


def polars_matmul(left: pl.Series, right: pl.Series) -> pl.Series:
    """polars-matmul matrix multiplication using expression API."""
    df = pl.DataFrame({"embedding": left})
    result = df.select(
        pl.col("embedding").pmm.matmul(right).alias("scores")
    )
    return result["scores"]


def benchmark_numpy(query: np.ndarray, corpus: np.ndarray, n_runs: int = 10) -> float:
    """Benchmark NumPy matmul, return median time in ms."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = np.dot(query, corpus.T)
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


def benchmark_pmm(left: pl.Series, right: pl.Series, n_runs: int = 10) -> float:
    """Benchmark polars-matmul, return median time in ms."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = polars_matmul(left, right)
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


def run_single(n_queries: int, n_corpus: int, dim: int, dtype, input_type: str = "array") -> dict:
    """Run benchmark for a specific configuration.
    
    Args:
        input_type: "array" for pl.Array (fast path) or "list" for pl.List
    """
    np.random.seed(42)
    query_np = np.random.randn(n_queries, dim).astype(dtype)
    corpus_np = np.random.randn(n_corpus, dim).astype(dtype)

    pl_dtype = pl.Float32 if dtype == np.float32 else pl.Float64
    
    if input_type == "array":
        left = pl.Series("l", query_np.tolist()).cast(pl.Array(pl_dtype, dim))
        right = pl.Series("r", corpus_np.tolist()).cast(pl.Array(pl_dtype, dim))
    else:
        left = pl.Series("l", query_np.tolist()).cast(pl.List(pl_dtype))
        right = pl.Series("r", corpus_np.tolist()).cast(pl.List(pl_dtype))

    # Warmup
    for _ in range(3):
        np.dot(query_np, corpus_np.T)
        polars_matmul(left, right)

    numpy_time = benchmark_numpy(query_np, corpus_np)
    pmm_time = benchmark_pmm(left, right)
    ratio = pmm_time / numpy_time

    return {
        "queries": n_queries,
        "corpus": n_corpus,
        "dim": dim,
        "dtype": "f32" if dtype == np.float32 else "f64",
        "input_type": input_type,
        "numpy_ms": numpy_time,
        "pmm_ms": pmm_time,
        "ratio": ratio,
    }


def print_results(results: list[dict], title: str):
    """Print results in a tabular format."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = f"{'Queries':>8} {'Corpus':>8} {'Dim':>6} {'Dtype':>6} {'NumPy (ms)':>12} {'PMM (ms)':>12} {'Ratio':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        row = (
            f"{r['queries']:>8} {r['corpus']:>8} {r['dim']:>6} {r['dtype']:>6} "
            f"{r['numpy_ms']:>12.2f} {r['pmm_ms']:>12.2f} {r['ratio']:>8.2f}x"
        )
        print(row)


def main():
    print("=" * 80)
    print("  polars-matmul Benchmark: Raw Matrix Multiplication")
    print("=" * 80)

    # Base configuration
    base_queries, base_corpus, base_dim = 1000, 10000, 256
    base_dtype = np.float32

    all_results = []

    # Vary queries
    results = []
    for q in [base_queries // 2, base_queries, base_queries * 2]:
        results.append(run_single(q, base_corpus, base_dim, base_dtype))
    print_results(results, "Varying Queries (corpus=10000, dim=256, dtype=f32)")
    all_results.extend(results)

    # Vary corpus
    results = []
    for c in [base_corpus // 2, base_corpus, base_corpus * 2]:
        results.append(run_single(base_queries, c, base_dim, base_dtype))
    print_results(results, "Varying Corpus (queries=1000, dim=256, dtype=f32)")
    all_results.extend(results)

    # Vary dim
    results = []
    for d in [base_dim // 2, base_dim, base_dim * 2]:
        results.append(run_single(base_queries, base_corpus, d, base_dtype))
    print_results(results, "Varying Dimension (queries=1000, corpus=10000, dtype=f32)")
    all_results.extend(results)

    # Vary dtype
    results = []
    for dtype in [np.float32, np.float64]:
        results.append(run_single(base_queries, base_corpus, base_dim, dtype))
    print_results(results, "Varying Dtype (queries=1000, corpus=10000, dim=256)")
    all_results.extend(results)

    # Vary input type (List vs Array)
    results = []
    for input_type in ["array", "list"]:
        results.append(run_single(base_queries, base_corpus, base_dim, base_dtype, input_type))
    print_results_with_type(results, "Varying Input Type (queries=1000, corpus=10000, dim=256, dtype=f32)")
    all_results.extend(results)

    print(f"\n{'=' * 80}")
    print("  Summary")
    print(f"{'=' * 80}")
    faster_count = sum(1 for r in all_results if r["ratio"] < 1.0)
    total_count = len(all_results)
    print(f"  PMM faster in {faster_count}/{total_count} configurations")
    avg_ratio = np.mean([r["ratio"] for r in all_results])
    print(f"  Average ratio: {avg_ratio:.2f}x")


def print_results_with_type(results: list[dict], title: str):
    """Print results including input type column."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    header = f"{'Queries':>8} {'Corpus':>8} {'Dim':>6} {'Dtype':>6} {'Type':>6} {'NumPy (ms)':>12} {'PMM (ms)':>12} {'Ratio':>8}"
    print(header)
    print("-" * 90)
    for r in results:
        row = (
            f"{r['queries']:>8} {r['corpus']:>8} {r['dim']:>6} {r['dtype']:>6} {r['input_type']:>6} "
            f"{r['numpy_ms']:>12.2f} {r['pmm_ms']:>12.2f} {r['ratio']:>8.2f}x"
        )
        print(row)


if __name__ == "__main__":
    main()
