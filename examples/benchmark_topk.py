#!/usr/bin/env python3
"""
Benchmark: Similarity Search with Top-K (polars-matmul vs NumPy)

Measures the end-to-end similarity search: normalize, matmul, top-k selection.
"""

import time
import numpy as np
import polars as pl
import polars_matmul  # noqa: F401 - registers the .pmm namespace


def numpy_topk_cosine(
    query: np.ndarray, corpus: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference implementation for top-k cosine similarity."""
    # Normalize
    query_norm = query / np.sqrt(np.sum(query**2, axis=1, keepdims=True))
    corpus_norm = corpus / np.sqrt(np.sum(corpus**2, axis=1, keepdims=True))

    # Matmul
    similarities = np.dot(query_norm, corpus_norm.T)

    # Top-k selection (positive index is faster than negating the matrix)
    partitioned = np.argpartition(similarities, -k, axis=1)[:, -k:]
    rows = np.arange(len(query))[:, None]
    top_k_similarities = similarities[rows, partitioned]
    sorted_within_k = np.argsort(-top_k_similarities, axis=1)

    indices = partitioned[rows, sorted_within_k]
    scores = top_k_similarities[rows, sorted_within_k]
    return indices, scores


def benchmark_numpy(
    query: np.ndarray, corpus: np.ndarray, k: int, n_runs: int = 5
) -> float:
    """Benchmark NumPy top-k. Returns median time in ms."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        numpy_topk_cosine(query, corpus, k)
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


def benchmark_pmm(
    query_df: pl.DataFrame, corpus_emb: pl.Series, k: int, n_runs: int = 5
) -> float:
    """Benchmark polars-matmul topk. Returns median time in ms."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = (
            query_df
            .with_columns(
                pl.col("embedding").pmm.topk(corpus_emb, k=k, metric="cosine").alias("m")
            )
            .explode("m")
            .unnest("m")
        )
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


def run_single(n_queries: int, n_corpus: int, dim: int, k: int, dtype) -> dict:
    """Run benchmark for a specific configuration."""
    np.random.seed(42)
    query_np = np.random.randn(n_queries, dim).astype(dtype)
    corpus_np = np.random.randn(n_corpus, dim).astype(dtype)

    query_df = pl.DataFrame(
        {"query_id": range(n_queries), "embedding": query_np.tolist()}
    )
    corpus_emb = pl.Series("embedding", corpus_np.tolist())

    # Cast to appropriate dtype
    pl_dtype = pl.Float32 if dtype == np.float32 else pl.Float64
    query_df = query_df.with_columns(pl.col("embedding").cast(pl.List(pl_dtype)))
    corpus_emb = corpus_emb.cast(pl.List(pl_dtype))

    # Warmup
    for _ in range(2):
        numpy_topk_cosine(query_np, corpus_np, k)
        _ = query_df.with_columns(
            pl.col("embedding").pmm.topk(corpus_emb, k=k, metric="cosine").alias("m")
        )

    numpy_time = benchmark_numpy(query_np, corpus_np, k)
    pmm_time = benchmark_pmm(query_df, corpus_emb, k)
    ratio = pmm_time / numpy_time

    return {
        "queries": n_queries,
        "corpus": n_corpus,
        "dim": dim,
        "k": k,
        "dtype": "f32" if dtype == np.float32 else "f64",
        "numpy_ms": numpy_time,
        "pmm_ms": pmm_time,
        "ratio": ratio,
    }


def print_results(results: list[dict], title: str):
    """Print results in a tabular format."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    header = f"{'Queries':>8} {'Corpus':>8} {'Dim':>6} {'k':>4} {'Dtype':>6} {'NumPy (ms)':>12} {'PMM (ms)':>12} {'Ratio':>8}"
    print(header)
    print("-" * 90)
    for r in results:
        row = (
            f"{r['queries']:>8} {r['corpus']:>8} {r['dim']:>6} {r['k']:>4} {r['dtype']:>6} "
            f"{r['numpy_ms']:>12.2f} {r['pmm_ms']:>12.2f} {r['ratio']:>8.2f}x"
        )
        print(row)


def verify_correctness(
    query_np: np.ndarray, corpus_np: np.ndarray, pmm_result: pl.DataFrame, k: int
) -> bool:
    """Verify PMM results match NumPy."""
    np_indices, np_scores = numpy_topk_cosine(query_np, corpus_np, k)
    n_queries = len(query_np)

    for i in range(n_queries):
        pmm_query = pmm_result.filter(pl.col("query_id") == i).sort(
            "score", descending=True
        )
        pmm_scores = sorted(pmm_query["score"].to_list(), reverse=True)
        np_scores_sorted = sorted(np_scores[i].tolist(), reverse=True)

        if not np.allclose(pmm_scores, np_scores_sorted, rtol=1e-4):
            return False
    return True


def main():
    print("=" * 90)
    print("  polars-matmul Benchmark: Top-K Similarity Search")
    print("=" * 90)

    # Base configuration
    base_queries, base_corpus, base_dim, base_k = 1000, 10000, 256, 10
    base_dtype = np.float32

    all_results = []

    # Vary queries
    results = []
    for q in [base_queries // 2, base_queries, base_queries * 2]:
        results.append(run_single(q, base_corpus, base_dim, base_k, base_dtype))
    print_results(results, "Varying Queries (corpus=10000, dim=256, k=10, dtype=f32)")
    all_results.extend(results)

    # Vary corpus
    results = []
    for c in [base_corpus // 2, base_corpus, base_corpus * 2]:
        results.append(run_single(base_queries, c, base_dim, base_k, base_dtype))
    print_results(results, "Varying Corpus (queries=1000, dim=256, k=10, dtype=f32)")
    all_results.extend(results)

    # Vary dim
    results = []
    for d in [base_dim // 2, base_dim, base_dim * 2]:
        results.append(run_single(base_queries, base_corpus, d, base_k, base_dtype))
    print_results(results, "Varying Dimension (queries=1000, corpus=10000, k=10, dtype=f32)")
    all_results.extend(results)

    # Vary k
    results = []
    for k in [base_k // 2, base_k, base_k * 2]:
        results.append(run_single(base_queries, base_corpus, base_dim, k, base_dtype))
    print_results(results, "Varying K (queries=1000, corpus=10000, dim=256, dtype=f32)")
    all_results.extend(results)

    # Vary dtype
    results = []
    for dtype in [np.float32, np.float64]:
        results.append(run_single(base_queries, base_corpus, base_dim, base_k, dtype))
    print_results(results, "Varying Dtype (queries=1000, corpus=10000, dim=256, k=10)")
    all_results.extend(results)

    # Correctness verification
    print(f"\n{'=' * 90}")
    print("  Correctness Verification")
    print(f"{'=' * 90}")
    np.random.seed(42)
    query_np = np.random.randn(100, 64).astype(np.float64)
    corpus_np = np.random.randn(500, 64).astype(np.float64)
    query_df = pl.DataFrame({"query_id": range(100), "embedding": query_np.tolist()})
    corpus_emb = pl.Series("embedding", corpus_np.tolist())
    pmm_result = (
        query_df
        .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=10, metric="cosine").alias("m"))
        .explode("m")
        .unnest("m")
    )
    is_correct = verify_correctness(query_np, corpus_np, pmm_result, k=10)
    print(f"  Result: {'✅ PASSED' if is_correct else '❌ FAILED'}")

    # Summary
    print(f"\n{'=' * 90}")
    print("  Summary")
    print(f"{'=' * 90}")
    faster_count = sum(1 for r in all_results if r["ratio"] < 1.0)
    total_count = len(all_results)
    print(f"  PMM faster in {faster_count}/{total_count} configurations")
    avg_ratio = np.mean([r["ratio"] for r in all_results])
    print(f"  Average ratio: {avg_ratio:.2f}x")


if __name__ == "__main__":
    main()
