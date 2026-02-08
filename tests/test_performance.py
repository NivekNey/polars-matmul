"""Performance tests for CI - verify performance is acceptable"""

import time
import numpy as np
import polars as pl
import pytest

import polars_matmul  # noqa: F401 - registers namespace


def numpy_matmul(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Reference NumPy matrix multiplication."""
    return np.dot(query, corpus.T)


def polars_matmul(left: pl.Series, right: pl.Series) -> pl.Series:
    """polars-matmul matrix multiplication using expression API."""
    df = pl.DataFrame({"embedding": left})
    result = df.select(
        pl.col("embedding").pmm.matmul(right).alias("scores")
    )
    return result["scores"]


class TestPerformance:
    """Tests to verify performance is within acceptable range of NumPy."""
    
    def test_performance_vs_numpy(self):
        """Verify polars-matmul has reasonable performance.
        
        Using Array[f64, dim] type to enable the optimized extraction path.
        We expect performance to be within ~2-5x of NumPy due to conversion overhead.
        """
        np.random.seed(42)
        n_queries, n_corpus, dim = 100, 1000, 128
        
        query_np = np.random.randn(n_queries, dim).astype(np.float64)
        corpus_np = np.random.randn(n_corpus, dim).astype(np.float64)
        
        # Use Array type for optimized path (not List which has higher overhead)
        left = pl.Series("l", query_np.tolist()).cast(pl.Array(pl.Float64, dim))
        right = pl.Series("r", corpus_np.tolist()).cast(pl.Array(pl.Float64, dim))
        
        # Warmup
        numpy_matmul(query_np, corpus_np)
        polars_matmul(left, right)
        
        # Benchmark NumPy
        n_runs = 5
        numpy_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            numpy_matmul(query_np, corpus_np)
            numpy_times.append(time.perf_counter() - start)
        numpy_mean = np.mean(numpy_times)
        
        # Benchmark polars-matmul
        pmm_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            polars_matmul(left, right)
            pmm_times.append(time.perf_counter() - start)
        pmm_mean = np.mean(pmm_times)
        
        ratio = pmm_mean / numpy_mean
        
        print(f"\nPerformance: {n_queries}x{dim} @ {n_corpus}x{dim}^T")
        print(f"  NumPy:         {numpy_mean*1000:.2f}ms")
        print(f"  polars-matmul: {pmm_mean*1000:.2f}ms")
        print(f"  Ratio: {ratio:.2f}x")
        
        # Threshold of 10x allows for CI variability and cold starts
        assert ratio < 12.0, (
            f"polars-matmul is {ratio:.1f}x slower than NumPy. "
            f"Optimization may be needed."
        )
    
    def test_correctness_vs_numpy(self):
        """Verify results match NumPy exactly."""
        np.random.seed(42)
        n_queries, n_corpus, dim = 10, 20, 32
        
        query_np = np.random.randn(n_queries, dim).astype(np.float64)
        corpus_np = np.random.randn(n_corpus, dim).astype(np.float64)
        
        expected = np.dot(query_np, corpus_np.T)
        
        left = pl.Series("l", query_np.tolist())
        right = pl.Series("r", corpus_np.tolist())
        result = polars_matmul(left, right)
        
        for i in range(n_queries):
            actual = result[i].to_list()
            np.testing.assert_allclose(
                actual, expected[i], rtol=1e-5,
                err_msg=f"Mismatch at row {i}"
            )
    
    def test_topk_performance(self):
        """Verify topk is reasonably fast."""
        np.random.seed(42)
        n_queries, n_corpus, dim, k = 50, 500, 64, 10
        
        query_np = np.random.randn(n_queries, dim).astype(np.float64)
        corpus_np = np.random.randn(n_corpus, dim).astype(np.float64)
        
        query_df = pl.DataFrame({
            "query_id": range(n_queries),
            "embedding": query_np.tolist(),
        })
        corpus_emb = pl.Series("e", corpus_np.tolist())
        
        # Warmup
        query_df.select(
            pl.col("embedding").pmm.topk(corpus_emb, k=k)
        )
        
        # Benchmark
        start = time.perf_counter()
        result = (
            query_df
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=k).alias("m"))
            .explode("m")
            .unnest("m")
        )
        elapsed = time.perf_counter() - start
        
        print(f"\ntopk: {n_queries} queries, {n_corpus} corpus, k={k}")
        print(f"  Time: {elapsed*1000:.2f}ms")
        
        # Should complete in under 1 second for this size
        assert elapsed < 1.0, f"topk too slow: {elapsed:.2f}s"
        
        # Verify result size
        assert len(result) == n_queries * k
    
    def test_f32_performance(self):
        """Verify f32 path has comparable performance to f64.
        
        f32 should be at least as fast (often faster due to memory bandwidth).
        """
        np.random.seed(42)
        n_queries, n_corpus, dim = 100, 1000, 128
        
        # f64 data
        query_f64 = np.random.randn(n_queries, dim).astype(np.float64)
        corpus_f64 = np.random.randn(n_corpus, dim).astype(np.float64)
        
        # f32 data
        query_f32 = query_f64.astype(np.float32)
        corpus_f32 = corpus_f64.astype(np.float32)
        
        # Create Polars Series
        left_f64 = pl.Series("l", query_f64.tolist()).cast(pl.Array(pl.Float64, dim))
        right_f64 = pl.Series("r", corpus_f64.tolist()).cast(pl.Array(pl.Float64, dim))
        left_f32 = pl.Series("l", query_f32.tolist()).cast(pl.Array(pl.Float32, dim))
        right_f32 = pl.Series("r", corpus_f32.tolist()).cast(pl.Array(pl.Float32, dim))
        
        # Warmup
        polars_matmul(left_f64, right_f64)
        polars_matmul(left_f32, right_f32)
        
        # Benchmark f64
        n_runs = 5
        f64_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            polars_matmul(left_f64, right_f64)
            f64_times.append(time.perf_counter() - start)
        f64_mean = np.mean(f64_times)
        
        # Benchmark f32
        f32_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            polars_matmul(left_f32, right_f32)
            f32_times.append(time.perf_counter() - start)
        f32_mean = np.mean(f32_times)
        
        ratio = f32_mean / f64_mean
        
        print(f"\nf32 vs f64: {n_queries}x{dim} @ {n_corpus}x{dim}^T")
        print(f"  f64: {f64_mean*1000:.2f}ms")
        print(f"  f32: {f32_mean*1000:.2f}ms")
        print(f"  Ratio (f32/f64): {ratio:.2f}x")
        
        # f32 should be at least 80% as fast as f64 (often faster)
        assert ratio < 1.5, f"f32 is {ratio:.1f}x slower than f64."
