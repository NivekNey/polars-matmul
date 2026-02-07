"""Performance tests for CI - verify BLAS acceleration is working"""

import time
import numpy as np
import polars as pl
import pytest

import polars_matmul as pmm


def numpy_matmul(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Reference NumPy matrix multiplication."""
    return query @ corpus.T


def polars_matmul(left: pl.Series, right: pl.Series):
    """polars-matmul matrix multiplication."""
    return pmm.matmul(left, right)


class TestBLASPerformance:
    """Tests to verify BLAS acceleration is working correctly."""
    
    def test_blas_performance_vs_numpy(self):
        """Verify polars-matmul has BLAS acceleration working.
        
        If BLAS is not linked correctly, this will be 100x+ slower.
        We expect performance to be comparable to NumPy (within 5x)
        now that we measure raw execution without conversion overhead.
        """
        np.random.seed(42)
        n_queries, n_corpus, dim = 100, 1000, 128
        
        query_np = np.random.randn(n_queries, dim).astype(np.float64)
        corpus_np = np.random.randn(n_corpus, dim).astype(np.float64)
        
        left = pl.Series("l", query_np.tolist())
        right = pl.Series("r", corpus_np.tolist())
        
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
        
        # BLAS should provide acceleration
        # Without conversion overhead, we should be much closer to NumPy
        assert ratio < 1.5, (
            f"polars-matmul is {ratio:.1f}x slower than NumPy. "
            f"BLAS may not be linked correctly."
        )
    
    def test_correctness_vs_numpy(self):
        """Verify results match NumPy exactly."""
        np.random.seed(42)
        n_queries, n_corpus, dim = 10, 20, 32
        
        query_np = np.random.randn(n_queries, dim).astype(np.float64)
        corpus_np = np.random.randn(n_corpus, dim).astype(np.float64)
        
        expected = query_np @ corpus_np.T
        
        left = pl.Series("l", query_np.tolist())
        right = pl.Series("r", corpus_np.tolist())
        result = pmm.matmul(left, right)
        
        for i in range(n_queries):
            actual = result[i].to_list()
            np.testing.assert_allclose(
                actual, expected[i], rtol=1e-5,
                err_msg=f"Mismatch at row {i}"
            )
    
    def test_similarity_join_performance(self):
        """Verify similarity_join is reasonably fast."""
        np.random.seed(42)
        n_queries, n_corpus, dim, k = 50, 500, 64, 10
        
        query_np = np.random.randn(n_queries, dim).astype(np.float64)
        corpus_np = np.random.randn(n_corpus, dim).astype(np.float64)
        
        query_df = pl.DataFrame({
            "query_id": range(n_queries),
            "embedding": query_np.tolist(),
        })
        corpus_df = pl.DataFrame({
            "corpus_id": range(n_corpus),
            "embedding": corpus_np.tolist(),
        })
        
        # Warmup
        pmm.similarity_join(
            left=query_df, right=corpus_df,
            left_on="embedding", right_on="embedding",
            k=k, metric="cosine"
        )
        
        # Benchmark
        start = time.perf_counter()
        result = pmm.similarity_join(
            left=query_df, right=corpus_df,
            left_on="embedding", right_on="embedding",
            k=k, metric="cosine"
        )
        elapsed = time.perf_counter() - start
        
        print(f"\nsimilarity_join: {n_queries} queries, {n_corpus} corpus, k={k}")
        print(f"  Time: {elapsed*1000:.2f}ms")
        
        # Should complete in under 1 second for this size
        assert elapsed < 1.0, f"similarity_join too slow: {elapsed:.2f}s"
        
        # Verify result size
        assert len(result) == n_queries * k
