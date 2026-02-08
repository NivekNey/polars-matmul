"""Tests for polars-matmul"""

import pytest
import polars as pl
import numpy as np

import polars_matmul as pmm


class TestSimilarityJoin:
    """Tests for the similarity_join function"""
    
    def test_basic_cosine(self):
        """Test basic cosine similarity join"""
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
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=2,
            metric="cosine",
        )
        
        # Should have 4 rows (2 queries × 2 top-k)
        assert len(result) == 4
        
        # Check columns
        assert "query_id" in result.columns
        assert "corpus_id" in result.columns
        assert "label" in result.columns
        assert "_score" in result.columns
        
        # Embedding columns should be excluded
        assert "embedding" not in result.columns
        
        # Check that query 0's top match is corpus 0 (cosine = 1.0)
        query0_results = result.filter(pl.col("query_id") == 0)
        top_match = query0_results.sort("_score", descending=True).row(0)
        assert top_match[1] == 0  # corpus_id
        assert abs(top_match[3] - 1.0) < 1e-6  # score
    
    def test_dot_product(self):
        """Test dot product metric"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[2.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "id": [0, 1],
            "embedding": [[1.0, 0.0], [3.0, 0.0]],
        })
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=2,
            metric="dot",
        )
        
        # Top match should be [3, 0] with dot product 6.0
        top = result.sort("_score", descending=True).row(0)
        assert top[1] == 1  # corpus id
        assert abs(top[2] - 6.0) < 1e-6  # score (2*3 = 6)
    
    def test_euclidean(self):
        """Test euclidean distance metric"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[0.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "id": [0, 1],
            "embedding": [[3.0, 4.0], [1.0, 0.0]],  # distances: 5, 1
        })
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=2,
            metric="euclidean",
        )
        
        # For euclidean, lower is better - so [1, 0] should be first
        top = result.sort("_score").row(0)
        assert top[1] == 1  # corpus id with distance 1
        assert abs(top[2] - 1.0) < 1e-6
    
    def test_lazyframe(self):
        """Test that LazyFrame input returns LazyFrame output"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        }).lazy()
        
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        }).lazy()
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=1,
        )
        
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert len(collected) == 1
    
    def test_k_larger_than_corpus(self):
        """Test when k > corpus size"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        })
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=10,  # Much larger than corpus
        )
        
        # Should return all 2 corpus items
        assert len(result) == 2
    
    def test_suffix_handling(self):
        """Test column name suffix for conflicts"""
        df = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
            "value": [100],
        })
        
        result = pmm.similarity_join(
            left=df,
            right=df,
            left_on="embedding",
            right_on="embedding",
            k=1,
            suffix="_corpus",
        )
        
        # Should have both 'value' and 'value_corpus'
        assert "value" in result.columns
        assert "value_corpus" in result.columns


class TestMatmul:
    """Tests for matmul function"""
    
    def test_basic(self):
        """Test basic matrix multiplication"""
        left = pl.Series("l", [[1.0, 2.0], [3.0, 4.0]])
        right = pl.Series("r", [[1.0, 0.0], [0.0, 1.0]])
        
        result = pmm.matmul(left, right)
        
        assert result.len() == 2
        # [1, 2] @ [[1, 0], [0, 1]]^T = [1*1 + 2*0, 1*0 + 2*1] = [1, 2]
        assert result[0].to_list() == pytest.approx([1.0, 2.0])
        # [3, 4] @ [[1, 0], [0, 1]]^T = [3*1 + 4*0, 3*0 + 4*1] = [3, 4]
        assert result[1].to_list() == pytest.approx([3.0, 4.0])
    
    def test_against_numpy(self):
        """Verify matmul matches NumPy results"""
        np.random.seed(42)
        left_np = np.random.randn(10, 32)
        right_np = np.random.randn(20, 32)
        
        left = pl.Series("l", left_np.tolist())
        right = pl.Series("r", right_np.tolist())
        
        result = pmm.matmul(left, right)
        expected = np.dot(left_np, right_np.T)
        
        for i in range(10):
            actual = result[i].to_list()
            np.testing.assert_allclose(actual, expected[i], rtol=1e-5)


class TestNumpyEquivalence:
    """Tests verifying equivalence with NumPy implementations"""
    
    def test_cosine_similarity_via_join(self):
        """Test cosine similarity matches NumPy implementation"""
        np.random.seed(42)
        query = np.random.randn(5, 16)
        corpus = np.random.randn(20, 16)
        
        # Normalize for cosine similarity
        query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
        corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        expected = np.dot(query_norm, corpus_norm.T)
        
        query_df = pl.DataFrame({
            "id": range(5),
            "embedding": query.tolist(),
        })
        corpus_df = pl.DataFrame({
            "id": range(20),
            "embedding": corpus.tolist(),
        })
        
        # Get all matches (k=20)
        result = pmm.similarity_join(
            left=query_df,
            right=corpus_df,
            left_on="embedding",
            right_on="embedding",
            k=20,
            metric="cosine",
        )
        
        # Check that for each query, the scores match the expected values
        for i in range(5):
            query_results = result.filter(pl.col("id") == i).sort("id_right")
            actual_scores = query_results["_score"].to_list()
            expected_scores = sorted(expected[i].tolist(), reverse=True)
            actual_sorted = sorted(actual_scores, reverse=True)
            np.testing.assert_allclose(actual_sorted, expected_scores, rtol=1e-5)


class TestErrorHandling:
    """Tests for proper error handling and reporting"""
    
    def test_invalid_metric(self):
        """Test that invalid metric raises a clear error"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        with pytest.raises(RuntimeError, match="Unknown metric.*invalid_metric"):
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="embedding",
                right_on="embedding",
                k=1,
                metric="invalid_metric",
            )
    
    def test_missing_left_column(self):
        """Test error when left_on column doesn't exist"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        with pytest.raises(Exception):  # Column not found
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="nonexistent",
                right_on="embedding",
                k=1,
            )
    
    def test_missing_right_column(self):
        """Test error when right_on column doesn't exist"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        with pytest.raises(Exception):  # Column not found
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="embedding",
                right_on="nonexistent",
                k=1,
            )
    
    def test_empty_query_dataframe(self):
        """Test error with empty query DataFrame"""
        queries = pl.DataFrame({
            "id": [],
            "embedding": [],
        }).cast({"embedding": pl.List(pl.Float64)})
        
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        with pytest.raises(RuntimeError, match="Empty"):
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="embedding",
                right_on="embedding",
                k=1,
            )
    
    def test_empty_corpus_dataframe(self):
        """Test error with empty corpus DataFrame"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "id": [],
            "embedding": [],
        }).cast({"embedding": pl.List(pl.Float64)})
        
        with pytest.raises(RuntimeError, match="Empty"):
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="embedding",
                right_on="embedding",
                k=1,
            )
    
    def test_k_zero(self):
        """Test that k=0 raises ValueError"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        # k=0 is validated in Python wrapper as invalid
        with pytest.raises(ValueError, match="k must be positive"):
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="embedding",
                right_on="embedding",
                k=0,
            )
    
    def test_matmul_empty_series(self):
        """Test matmul with empty series"""
        left = pl.Series("l", [], dtype=pl.List(pl.Float64))
        right = pl.Series("r", [[1.0, 0.0]])
        
        with pytest.raises(RuntimeError, match="Empty"):
            pmm.matmul(left, right)
    
    def test_matmul_dimension_mismatch(self):
        """Test matmul with mismatched dimensions raises clear error"""
        left = pl.Series("l", [[1.0, 2.0]])  # 2D vectors
        right = pl.Series("r", [[1.0, 2.0, 3.0]])  # 3D vectors
        
        with pytest.raises(RuntimeError, match="Dimension mismatch"):
            pmm.matmul(left, right)
    
    def test_similarity_join_dimension_mismatch(self):
        """Test similarity join with mismatched embedding dimensions"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 2.0]],  # 2D vectors
        })
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 2.0, 3.0]],  # 3D vectors
        })
        
        with pytest.raises(RuntimeError, match="Dimension mismatch"):
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="embedding",
                right_on="embedding",
                k=1,
            )
    
    def test_scalar_embedding_column(self):
        """Test error when embedding column is scalar, not list"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [1.0],  # Scalar, not list
        })
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        # Scalar columns get treated as 1D vectors, causing dimension mismatch
        with pytest.raises(RuntimeError, match="Dimension mismatch"):
            pmm.similarity_join(
                left=queries,
                right=corpus,
                left_on="embedding",
                right_on="embedding",
                k=1,
            )


class TestFloat32Support:
    """Tests for Float32 support - 2x memory efficiency"""
    
    def test_matmul_f32(self):
        """Test matmul with f32 input returns f32 output"""
        left = pl.Series("l", [[1.0, 2.0], [3.0, 4.0]]).cast(pl.List(pl.Float32))
        right = pl.Series("r", [[1.0, 0.0], [0.0, 1.0]]).cast(pl.List(pl.Float32))
        
        result = pmm.matmul(left, right)
        
        # Result should be List[f32]
        assert result.dtype == pl.List(pl.Float32)
        
        # Check values
        expected = np.dot(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]).T)
        for i in range(2):
            np.testing.assert_allclose(result[i].to_list(), expected[i], rtol=1e-5)
    
    def test_matmul_f64(self):
        """Test matmul with f64 input returns f64 output"""
        left = pl.Series("l", [[1.0, 2.0], [3.0, 4.0]])  # Default is f64
        right = pl.Series("r", [[1.0, 0.0], [0.0, 1.0]])
        
        result = pmm.matmul(left, right)
        
        # Result should be List[f64]
        assert result.dtype == pl.List(pl.Float64)
    
    def test_similarity_join_f32(self):
        """Test similarity join with f32 embeddings"""
        np.random.seed(42)
        dim = 32
        
        queries = pl.DataFrame({
            "query_id": [0, 1],
            "embedding": [
                [float(x) for x in np.random.randn(dim)],
                [float(x) for x in np.random.randn(dim)],
            ],
        }).with_columns(
            pl.col("embedding").cast(pl.List(pl.Float32))
        )
        
        corpus = pl.DataFrame({
            "corpus_id": [0, 1, 2],
            "embedding": [
                [float(x) for x in np.random.randn(dim)],
                [float(x) for x in np.random.randn(dim)],
                [float(x) for x in np.random.randn(dim)],
            ],
        }).with_columns(
            pl.col("embedding").cast(pl.List(pl.Float32))
        )
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=2,
            metric="cosine",
        )
        
        # Should work and return results
        assert len(result) == 4  # 2 queries × 2 top-k
        assert "_score" in result.columns
        
        # Scores should be in valid range for cosine
        scores = result["_score"].to_list()
        assert all(-1.01 <= s <= 1.01 for s in scores)
    
    def test_mixed_f32_f64_uses_f64(self):
        """Test that mixed f32/f64 inputs fall back to f64"""
        left = pl.Series("l", [[1.0, 2.0]]).cast(pl.List(pl.Float32))
        right = pl.Series("r", [[1.0, 0.0]])  # f64
        
        result = pmm.matmul(left, right)
        
        # Mixed types should use f64 path
        assert result.dtype == pl.List(pl.Float64)
    
    def test_f32_array_type(self):
        """Test f32 with fixed-size Array type for optimal performance"""
        dim = 8
        left = pl.Series("l", [[1.0] * dim, [2.0] * dim]).cast(pl.Array(pl.Float32, dim))
        right = pl.Series("r", [[1.0] * dim, [0.5] * dim]).cast(pl.Array(pl.Float32, dim))
        
        result = pmm.matmul(left, right)
        
        # Should work with Array type
        assert result.dtype == pl.List(pl.Float32)
        assert len(result) == 2


class TestBatchProcessing:
    """Tests for batch processing of large corpuses"""
    
    def test_batch_same_results(self):
        """Verify batch processing produces same results as non-batched"""
        np.random.seed(42)
        
        queries = pl.DataFrame({
            "query_id": list(range(5)),
            "embedding": np.random.randn(5, 32).tolist()
        })
        corpus = pl.DataFrame({
            "corpus_id": list(range(100)),
            "embedding": np.random.randn(100, 32).tolist()
        })
        
        # Without batch
        result1 = pmm.similarity_join(
            queries, corpus, "embedding", "embedding", k=3
        ).sort(["query_id", "_score"], descending=[False, True])
        
        # With batch
        result2 = pmm.similarity_join(
            queries, corpus, "embedding", "embedding", k=3, batch_size=30
        ).sort(["query_id", "_score"], descending=[False, True])
        
        # Should have same number of results
        assert len(result1) == len(result2)
        
        # Top scores should match
        for qid in range(5):
            s1 = result1.filter(pl.col("query_id") == qid)["_score"].to_list()
            s2 = result2.filter(pl.col("query_id") == qid)["_score"].to_list()
            np.testing.assert_allclose(s1, s2, rtol=1e-5)
    
    def test_batch_smaller_than_corpus(self):
        """Test when batch_size is smaller than k"""
        queries = pl.DataFrame({
            "query_id": [0],
            "embedding": [[1.0, 0.0, 0.0]]
        })
        corpus = pl.DataFrame({
            "corpus_id": list(range(20)),
            "embedding": [[float(i), 0.0, 0.0] for i in range(20)]
        })
        
        # batch_size < k should still work
        result = pmm.similarity_join(
            queries, corpus, "embedding", "embedding", k=5, batch_size=3
        )
        
        assert len(result) == 5
    
    def test_batch_disabled_when_none(self):
        """Test that batch_size=None uses regular path"""
        queries = pl.DataFrame({
            "query_id": [0],
            "embedding": [[1.0, 0.0]]
        })
        corpus = pl.DataFrame({
            "corpus_id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]]
        })
        
        result = pmm.similarity_join(
            queries, corpus, "embedding", "embedding", k=2, batch_size=None
        )
        
        assert len(result) == 2
    
    def test_batch_euclidean(self):
        """Test batch processing with euclidean metric (lower is better)"""
        np.random.seed(42)
        
        queries = pl.DataFrame({
            "query_id": list(range(3)),
            "embedding": np.random.randn(3, 16).tolist()
        })
        corpus = pl.DataFrame({
            "corpus_id": list(range(50)),
            "embedding": np.random.randn(50, 16).tolist()
        })
        
        result1 = pmm.similarity_join(
            queries, corpus, "embedding", "embedding", k=5, metric="euclidean"
        ).sort(["query_id", "_score"])
        
        result2 = pmm.similarity_join(
            queries, corpus, "embedding", "embedding", k=5, metric="euclidean", batch_size=15
        ).sort(["query_id", "_score"])
        
        # Scores should match
        np.testing.assert_allclose(
            result1["_score"].to_list(),
            result2["_score"].to_list(),
            rtol=1e-5
        )
