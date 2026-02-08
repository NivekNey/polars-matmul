#!/usr/bin/env python3
"""
Benchmark polars-matmul vs NumPy for matrix multiplication.

This script measures the core operation: query @ corpus.T
"""

import time
import numpy as np
import polars as pl
import polars_matmul as pmm

def benchmark_numpy(query: np.ndarray, corpus: np.ndarray, n_iterations: int = 10) -> float:
    """Benchmark NumPy matmul, return median time in ms."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = np.dot(query, corpus.T)
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


def benchmark_pmm(left: pl.Series, right: pl.Series, n_iterations: int = 10) -> float:
    """Benchmark polars-matmul, return median time in ms."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = pmm.matmul(left, right)
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


def run_benchmark(n_queries: int, n_corpus: int, dim: int, dtype=np.float64):
    """Run benchmark for a specific size."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_queries} queries × {n_corpus} corpus × {dim} dim ({dtype.__name__})")
    print(f"{'='*60}")
    
    # Generate data
    np.random.seed(42)
    query_np = np.random.randn(n_queries, dim).astype(dtype)
    corpus_np = np.random.randn(n_corpus, dim).astype(dtype)
    
    # Create Polars Array type for optimal performance
    pl_dtype = pl.Float32 if dtype == np.float32 else pl.Float64
    left = pl.Series("l", query_np.tolist()).cast(pl.Array(pl_dtype, dim))
    right = pl.Series("r", corpus_np.tolist()).cast(pl.Array(pl_dtype, dim))
    
    # Warmup (important for accurate measurement)
    for _ in range(3):
        _ = np.dot(query_np, corpus_np.T)
        _ = pmm.matmul(left, right)
    
    # Benchmark
    numpy_time = benchmark_numpy(query_np, corpus_np, n_iterations=20)
    pmm_time = benchmark_pmm(left, right, n_iterations=20)
    
    ratio = pmm_time / numpy_time
    
    print(f"  NumPy:         {numpy_time:8.2f} ms")
    print(f"  polars-matmul: {pmm_time:8.2f} ms")
    print(f"  Ratio:         {ratio:8.2f}x {'(slower)' if ratio > 1 else '(faster)'}")
    
    return {
        "size": f"{n_queries}×{n_corpus}×{dim}",
        "dtype": dtype.__name__,
        "numpy_ms": numpy_time,
        "pmm_ms": pmm_time,
        "ratio": ratio,
    }


def main():
    print("=" * 60)
    print("polars-matmul Benchmark")
    print("=" * 60)
    
    results = []
    
    # Various sizes to test
    configs = [
        # (queries, corpus, dim)
        (100, 1000, 128),    # Small
        (100, 10000, 128),   # Medium corpus
        (1000, 10000, 128),  # Larger
        (100, 1000, 768),    # High-dim (BERT embeddings)
        (100, 10000, 768),   # High-dim + large corpus
    ]
    
    for n_queries, n_corpus, dim in configs:
        for dtype in [np.float64, np.float32]:
            result = run_benchmark(n_queries, n_corpus, dim, dtype)
            results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Size':<20} {'Dtype':<10} {'NumPy':<10} {'PMM':<10} {'Ratio':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['size']:<20} {r['dtype']:<10} {r['numpy_ms']:<10.2f} {r['pmm_ms']:<10.2f} {r['ratio']:<10.2f}")


if __name__ == "__main__":
    main()
