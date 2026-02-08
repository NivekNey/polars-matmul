"""Tests for polars-matmul Expression API"""

import pytest
import polars as pl
import numpy as np

import polars_matmul  # noqa: F401 - registers the .pmm namespace


class TestTopk:
    """Tests for the .pmm.topk() expression method"""
    
    def test_basic_cosine(self):
        """Test basic cosine similarity top-k"""
        queries = pl.DataFrame({
            "query_id": [0, 1],
            "embedding": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        })
        
        corpus = pl.DataFrame({
            "corpus_id": [0, 1, 2],
            "embedding": [
                [1.0, 0.0, 0.0],  # Exact match for query 0
                [0.0, 1.0, 0.0],  # Exact match for query 1
                [0.0, 0.0, 1.0],  # Orthogonal to both
            ],
            "label": ["a", "b", "c"],
        })
        
        result = queries.with_columns(
            pl.col("embedding").pmm.topk(corpus["embedding"], k=2).alias("matches")
        )
        
        # Should have 2 rows (one per query)
        assert len(result) == 2
        
        # Check that matches column is List[Struct]
        assert result["matches"].dtype == pl.List(pl.Struct({"index": pl.UInt32, "score": pl.Float64}))
        
        # Query 0's top match should be corpus 0 (cosine = 1.0)
        query0_matches = result.filter(pl.col("query_id") == 0)["matches"][0]
        top_match = query0_matches[0]
        assert top_match["index"] == 0
        assert abs(top_match["score"] - 1.0) < 1e-6
        
        # Query 1's top match should be corpus 1 (cosine = 1.0)
        query1_matches = result.filter(pl.col("query_id") == 1)["matches"][0]
        top_match1 = query1_matches[0]
        assert top_match1["index"] == 1
        assert abs(top_match1["score"] - 1.0) < 1e-6
    
    def test_explode_unnest_pattern(self):
        """Test the Polars-native explode/unnest pattern"""
        queries = pl.DataFrame({
            "query_id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        
        # Use explode and unnest to flatten
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=2).alias("matches"))
            .explode("matches")
            .unnest("matches")
        )
        
        # Should have 4 rows (2 queries × 2 top-k)
        assert len(result) == 4
        assert "index" in result.columns
        assert "score" in result.columns
    
    def test_dot_product(self):
        """Test dot product metric"""
        queries = pl.DataFrame({
            "embedding": [[2.0, 0.0]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [3.0, 0.0]])
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=2, metric="dot").alias("m"))
            .explode("m")
            .unnest("m")
        )
        
        # Top match should be [3, 0] with dot product 6.0
        top = result.sort("score", descending=True).row(0)
        assert top[1] == 1  # index
        assert abs(top[2] - 6.0) < 1e-6  # score
    
    def test_euclidean(self):
        """Test euclidean distance metric"""
        queries = pl.DataFrame({
            "embedding": [[0.0, 0.0]],
        })
        
        corpus_emb = pl.Series("e", [[3.0, 4.0], [1.0, 0.0]])  # distances: 5, 1
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=2, metric="euclidean").alias("m"))
            .explode("m")
            .unnest("m")
        )
        
        # For euclidean, lower is better - so [1, 0] should be first
        top = result.sort("score").row(0)
        assert top[1] == 1  # index (corpus item [1,0])
        assert abs(top[2] - 1.0) < 1e-6
    
    def test_k_larger_than_corpus(self):
        """Test when k > corpus size"""
        queries = pl.DataFrame({
            "embedding": [[1.0, 0.0]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=10).alias("m"))
            .explode("m")
            .unnest("m")
        )
        
        # Should return all 2 corpus items
        assert len(result) == 2
    
    def test_join_with_corpus_metadata(self):
        """Test joining back with corpus metadata"""
        queries = pl.DataFrame({
            "query_id": [0],
            "embedding": [[1.0, 0.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "corpus_id": [0, 1, 2],
            "embedding": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "label": ["a", "b", "c"],
        })
        
        # Pattern: topk -> explode -> unnest -> join
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus["embedding"], k=2).alias("m"))
            .explode("m")
            .unnest("m")
            .join(corpus.with_row_index("index"), on="index")
        )
        
        assert "label" in result.columns
        assert "corpus_id" in result.columns
        assert "score" in result.columns


class TestMatmul:
    """Tests for .pmm.matmul() expression method"""
    
    def test_basic(self):
        """Test basic matrix multiplication"""
        df = pl.DataFrame({
            "embedding": [[1.0, 2.0], [3.0, 4.0]]
        })
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = df.select(
            pl.col("embedding").pmm.matmul(corpus_emb).alias("scores")
        )
        
        assert result["scores"].len() == 2
        # [1, 2] @ [[1, 0], [0, 1]]^T = [1, 2]
        assert result["scores"][0].to_list() == pytest.approx([1.0, 2.0])
        # [3, 4] @ [[1, 0], [0, 1]]^T = [3, 4]
        assert result["scores"][1].to_list() == pytest.approx([3.0, 4.0])
    
    def test_against_numpy(self):
        """Verify matmul matches NumPy results"""
        np.random.seed(42)
        left_np = np.random.randn(10, 32)
        right_np = np.random.randn(20, 32)
        
        df = pl.DataFrame({"embedding": left_np.tolist()})
        corpus_emb = pl.Series("e", right_np.tolist())
        
        result = df.select(
            pl.col("embedding").pmm.matmul(corpus_emb).alias("scores")
        )
        expected = np.dot(left_np, right_np.T)
        
        for i in range(10):
            actual = result["scores"][i].to_list()
            np.testing.assert_allclose(actual, expected[i], rtol=1e-5)


class TestNumpyEquivalence:
    """Tests verifying equivalence with NumPy implementations"""
    
    def test_cosine_similarity_matches_numpy(self):
        """Test cosine similarity matches NumPy implementation"""
        np.random.seed(42)
        query_np = np.random.randn(5, 16)
        corpus_np = np.random.randn(20, 16)
        
        # Expected cosine similarities
        query_norm = query_np / np.linalg.norm(query_np, axis=1, keepdims=True)
        corpus_norm = corpus_np / np.linalg.norm(corpus_np, axis=1, keepdims=True)
        expected = np.dot(query_norm, corpus_norm.T)
        
        query_df = pl.DataFrame({"embedding": query_np.tolist()})
        corpus_emb = pl.Series("e", corpus_np.tolist())
        
        # Get all matches (k=20)
        result = (
            query_df
            .with_row_index("qid")
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=20).alias("m"))
            .explode("m")
            .unnest("m")
        )
        
        # Check that scores match expected values (sorted)
        for i in range(5):
            query_results = result.filter(pl.col("qid") == i).sort("index")
            actual_scores = query_results["score"].to_list()
            expected_scores = expected[i].tolist()
            np.testing.assert_allclose(
                sorted(actual_scores, reverse=True),
                sorted(expected_scores, reverse=True),
                rtol=1e-5
            )


class TestErrorHandling:
    """Tests for proper error handling"""
    
    def test_invalid_metric(self):
        """Test that invalid metric raises a clear error"""
        df = pl.DataFrame({"embedding": [[1.0, 0.0]]})
        corpus_emb = pl.Series("e", [[1.0, 0.0]])
        
        with pytest.raises(Exception, match="Unknown metric"):
            df.select(
                pl.col("embedding").pmm.topk(corpus_emb, k=1, metric="invalid_metric")
            )
    
    def test_corpus_expression_raises_error(self):
        """Test that passing an expression for corpus raises TypeError"""
        df = pl.DataFrame({"embedding": [[1.0, 0.0]]})
        
        with pytest.raises(TypeError, match="corpus must be a Polars Series"):
            df.select(
                pl.col("embedding").pmm.topk(pl.col("embedding"), k=1)  # expr, not series
            )
    
    def test_empty_query(self):
        """Test that empty query DataFrame returns empty result (not an error)"""
        df = pl.DataFrame({
            "embedding": [],
        }).cast({"embedding": pl.List(pl.Float64)})
        
        corpus_emb = pl.Series("e", [[1.0, 0.0]])
        
        # Empty queries should return empty result, not raise an error
        result = df.select(
            pl.col("embedding").pmm.topk(corpus_emb, k=1)
        )
        assert len(result) == 0
    
    def test_empty_corpus(self):
        """Test error with empty corpus Series"""
        df = pl.DataFrame({"embedding": [[1.0, 0.0]]})
        corpus_emb = pl.Series("e", [], dtype=pl.List(pl.Float64))
        
        with pytest.raises(Exception, match="Empty"):
            df.select(
                pl.col("embedding").pmm.topk(corpus_emb, k=1)
            )
    
    def test_matmul_dimension_mismatch(self):
        """Test matmul with mismatched dimensions"""
        df = pl.DataFrame({"embedding": [[1.0, 2.0]]})  # 2D
        corpus_emb = pl.Series("e", [[1.0, 2.0, 3.0]])  # 3D
        
        with pytest.raises(Exception, match="Dimension mismatch"):
            df.select(
                pl.col("embedding").pmm.matmul(corpus_emb)
            )
    
    def test_topk_dimension_mismatch(self):
        """Test topk with mismatched dimensions"""
        df = pl.DataFrame({"embedding": [[1.0, 2.0]]})  # 2D
        corpus_emb = pl.Series("e", [[1.0, 2.0, 3.0]])  # 3D
        
        with pytest.raises(Exception, match="Dimension mismatch"):
            df.select(
                pl.col("embedding").pmm.topk(corpus_emb, k=1)
            )


class TestFloat32Support:
    """Tests for Float32 support - 2x memory efficiency"""
    
    def test_matmul_f32(self):
        """Test matmul with f32 input returns f32 output"""
        df = pl.DataFrame({
            "embedding": [[1.0, 2.0], [3.0, 4.0]]
        }).with_columns(pl.col("embedding").cast(pl.List(pl.Float32)))
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]]).cast(pl.List(pl.Float32))
        
        result = df.select(
            pl.col("embedding").pmm.matmul(corpus_emb).alias("scores")
        )
        
        # Result should be Array[f32, n_corpus]
        assert result["scores"].dtype == pl.Array(pl.Float32, 2)
        
        # Check values
        expected = np.dot(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]).T)
        for i in range(2):
            np.testing.assert_allclose(result["scores"][i].to_list(), expected[i], rtol=1e-5)
    
    def test_matmul_f64(self):
        """Test matmul with f64 input returns f64 output"""
        df = pl.DataFrame({"embedding": [[1.0, 2.0], [3.0, 4.0]]})
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = df.select(
            pl.col("embedding").pmm.matmul(corpus_emb).alias("scores")
        )
        
        # Result should be Array[f64, n_corpus]
        assert result["scores"].dtype == pl.Array(pl.Float64, 2)
    
    def test_topk_f32(self):
        """Test topk with f32 embeddings"""
        np.random.seed(42)
        dim = 32
        
        df = pl.DataFrame({
            "query_id": [0, 1],
            "embedding": [
                [float(x) for x in np.random.randn(dim)],
                [float(x) for x in np.random.randn(dim)],
            ],
        }).with_columns(
            pl.col("embedding").cast(pl.List(pl.Float32))
        )
        
        corpus_emb = pl.Series("e", [
            [float(x) for x in np.random.randn(dim)] for _ in range(10)
        ]).cast(pl.List(pl.Float32))
        
        result = (
            df
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=2).alias("m"))
            .explode("m")
            .unnest("m")
        )
        
        # Should have 4 results
        assert len(result) == 4
        
        # Scores should be in valid range for cosine
        scores = result["score"].to_list()
        assert all(-1.01 <= s <= 1.01 for s in scores)
    
    def test_mixed_f32_f64_uses_f64(self):
        """Test that mixed f32/f64 inputs fall back to f64"""
        df = pl.DataFrame({
            "embedding": [[1.0, 2.0]]
        }).with_columns(pl.col("embedding").cast(pl.List(pl.Float32)))
        
        corpus_emb = pl.Series("e", [[1.0, 0.0]])  # f64
        
        result = df.select(
            pl.col("embedding").pmm.matmul(corpus_emb).alias("scores")
        )
        
        # Mixed types should use f64 path
        assert result["scores"].dtype == pl.Array(pl.Float64, 1)
    
    def test_f32_array_type(self):
        """Test f32 with fixed-size Array type for optimal performance"""
        dim = 8
        df = pl.DataFrame({
            "embedding": [[1.0] * dim, [2.0] * dim]
        }).with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, dim)))
        
        corpus_emb = pl.Series("e", [[1.0] * dim, [0.5] * dim]).cast(pl.Array(pl.Float32, dim))
        
        result = df.select(
            pl.col("embedding").pmm.matmul(corpus_emb).alias("scores")
        )
        
        # Should work with Array type - output is Array[f32, n_corpus]
        assert result["scores"].dtype == pl.Array(pl.Float32, 2)
        assert len(result) == 2


class TestLazyFrameEdgeCases:
    """Tests for LazyFrame usage and edge cases to ensure optimization doesn't break"""
    
    def test_lazy_basic_topk(self):
        """Test basic lazy evaluation with topk"""
        queries = pl.LazyFrame({
            "query_id": [0, 1, 2],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=2).alias("matches"))
            .collect()
        )
        
        assert len(result) == 3
        assert "matches" in result.columns
    
    def test_lazy_with_filter_before(self):
        """Test filter before pmm operation"""
        queries = pl.LazyFrame({
            "query_id": [0, 1, 2, 3],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
            "active": [True, False, True, True],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .filter(pl.col("active"))
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("matches"))
            .collect()
        )
        
        # Only active queries should be processed
        assert len(result) == 3
        assert 1 not in result["query_id"].to_list()
    
    def test_lazy_with_filter_after(self):
        """Test filter after pmm operation"""
        queries = pl.LazyFrame({
            "query_id": [0, 1, 2],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=2).alias("matches"))
            .filter(pl.col("query_id") > 0)
            .collect()
        )
        
        # Only query_id > 0 should remain
        assert len(result) == 2
        assert 0 not in result["query_id"].to_list()
    
    def test_lazy_with_select(self):
        """Test select with pmm to only keep specific columns"""
        queries = pl.LazyFrame({
            "query_id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "metadata": ["a", "b"],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .select([
                pl.col("query_id"),
                pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("top_match"),
            ])
            .collect()
        )
        
        assert result.columns == ["query_id", "top_match"]
        assert "metadata" not in result.columns
    
    def test_lazy_multiple_pmm_operations(self):
        """Test multiple pmm operations in same query"""
        queries = pl.LazyFrame({
            "query_id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        })
        
        corpus1 = pl.Series("c1", [[1.0, 0.0], [0.0, 1.0]])
        corpus2 = pl.Series("c2", [[0.5, 0.5], [1.0, 1.0]])
        
        result = (
            queries
            .with_columns([
                pl.col("embedding").pmm.topk(corpus1, k=1).alias("matches_corpus1"),
                pl.col("embedding").pmm.topk(corpus2, k=1).alias("matches_corpus2"),
            ])
            .collect()
        )
        
        assert "matches_corpus1" in result.columns
        assert "matches_corpus2" in result.columns
        assert len(result) == 2
    
    def test_lazy_explode_unnest_chain(self):
        """Test lazy explode and unnest chain"""
        queries = pl.LazyFrame({
            "query_id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=2).alias("matches"))
            .explode("matches")
            .unnest("matches")
            .collect()
        )
        
        assert len(result) == 4  # 2 queries × 2 top-k
        assert "index" in result.columns
        assert "score" in result.columns
    
    def test_lazy_with_join_after(self):
        """Test joining corpus metadata after pmm"""
        queries = pl.LazyFrame({
            "query_id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        })
        
        corpus = pl.DataFrame({
            "corpus_id": [0, 1, 2],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            "label": ["cat", "dog", "bird"],
        })
        
        corpus_emb = corpus["embedding"]
        
        # Get corpus metadata as lazy for join
        corpus_meta = corpus.select(["label"]).with_row_index("index").lazy()
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("matches"))
            .explode("matches")
            .unnest("matches")
            .join(corpus_meta, on="index", how="left")
            .collect()
        )
        
        assert "label" in result.columns
        assert len(result) == 2
    
    def test_lazy_with_group_by_after(self):
        """Test group_by aggregation after pmm"""
        queries = pl.LazyFrame({
            "category": ["A", "A", "B"],
            "embedding": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("matches"))
            .explode("matches")
            .unnest("matches")
            .group_by("category")
            .agg([
                pl.col("score").mean().alias("avg_score"),
                pl.col("index").n_unique().alias("unique_matches"),
            ])
            .collect()
        )
        
        assert len(result) == 2  # A and B categories
        assert "avg_score" in result.columns
    
    def test_lazy_matmul_basic(self):
        """Test lazy matmul operation"""
        queries = pl.LazyFrame({
            "embedding": [[1.0, 2.0], [3.0, 4.0]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.matmul(corpus_emb).alias("scores"))
            .collect()
        )
        
        assert "scores" in result.columns
        # First row: [1,2] @ [[1,0], [0,1]]^T = [1, 2]
        scores_0 = result["scores"][0].to_list()
        assert abs(scores_0[0] - 1.0) < 1e-6
        assert abs(scores_0[1] - 2.0) < 1e-6
    
    def test_lazy_with_streaming(self):
        """Test lazy evaluation with streaming (if supported)"""
        # Create a larger dataset to test streaming behavior
        n_queries = 100
        dim = 32
        
        np.random.seed(42)
        embeddings = [np.random.randn(dim).tolist() for _ in range(n_queries)]
        
        queries = pl.LazyFrame({
            "query_id": list(range(n_queries)),
            "embedding": embeddings,
        })
        
        corpus_emb = pl.Series("e", [np.random.randn(dim).tolist() for _ in range(50)])
        
        # This should work with or without streaming
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=5).alias("matches"))
            .collect()
        )
        
        assert len(result) == n_queries
    
    def test_lazy_empty_after_filter(self):
        """Test handling when filter results in empty dataframe"""
        queries = pl.LazyFrame({
            "query_id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "active": [False, False],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0]])
        
        result = (
            queries
            .filter(pl.col("active"))  # Filters out everything
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("matches"))
            .collect()
        )
        
        # Should return empty dataframe, not error
        assert len(result) == 0
        assert "matches" in result.columns
    
    def test_lazy_with_limit(self):
        """Test lazy with limit/head before pmm"""
        queries = pl.LazyFrame({
            "query_id": list(range(100)),
            "embedding": [[float(i), 0.0] for i in range(100)],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .head(5)
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("matches"))
            .collect()
        )
        
        assert len(result) == 5
    
    def test_lazy_with_sort_before(self):
        """Test lazy with sort before pmm"""
        queries = pl.LazyFrame({
            "query_id": [2, 0, 1],
            "embedding": [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
        })
        
        corpus_emb = pl.Series("e", [[1.0, 0.0], [0.0, 1.0]])
        
        result = (
            queries
            .sort("query_id")
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("matches"))
            .collect()
        )
        
        assert result["query_id"].to_list() == [0, 1, 2]
        assert len(result) == 3
    
    def test_lazy_array_type_optimization(self):
        """Test that Array type works correctly in lazy context"""
        dim = 8
        queries = pl.LazyFrame({
            "embedding": [[1.0] * dim, [2.0] * dim, [0.5] * dim],
        }).with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, dim)))
        
        corpus_emb = pl.Series("e", [[1.0] * dim, [0.0] * dim]).cast(pl.Array(pl.Float32, dim))
        
        result = (
            queries
            .with_columns(pl.col("embedding").pmm.topk(corpus_emb, k=1).alias("matches"))
            .collect()
        )
        
        assert len(result) == 3

