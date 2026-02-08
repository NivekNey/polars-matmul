"""
polars-matmul: High-performance similarity search for Polars

This package provides fast similarity search operations on embedding columns
using pure Rust matrix multiplication (faer).

Usage:
    >>> import polars as pl
    >>> import polars_matmul  # registers the .pmm namespace
    >>> 
    >>> queries = pl.DataFrame({
    ...     "query_id": [0, 1],
    ...     "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ... })
    >>> corpus = pl.DataFrame({
    ...     "corpus_id": [0, 1, 2],
    ...     "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ... })
    >>> 
    >>> # Find top-k similar items
    >>> result = queries.with_columns(
    ...     pl.col("embedding").pmm.topk(corpus["embedding"], k=2)
    ... )
"""

from polars_matmul._polars_matmul import (
    _matmul,
    _topk,
)
import polars as pl
from typing import Literal

__version__ = "0.1.0"
__all__ = ["PmmNamespace", "topk_search"]

Metric = Literal["cosine", "dot", "euclidean"]


@pl.api.register_expr_namespace("pmm")
class PmmNamespace:
    """
    Polars Expression API for similarity search operations.
    
    This namespace is automatically registered when you import polars_matmul.
    
    Example:
        >>> import polars as pl
        >>> import polars_matmul  # registers .pmm namespace
        >>> 
        >>> # Find top-k matches
        >>> df.with_columns(
        ...     pl.col("embedding").pmm.topk(corpus["embedding"], k=5)
        ... )
        >>> 
        >>> # Compute all pairwise scores
        >>> df.with_columns(
        ...     pl.col("embedding").pmm.matmul(corpus["embedding"])
        ... )
    """
    def __init__(self, expr: pl.Expr):
        self._expr = expr
    
    def topk(
        self,
        corpus: pl.Series,
        k: int,
        metric: Metric = "cosine",
    ) -> pl.Expr:
        """
        Find top-k similar items in corpus for each embedding.
        
        Returns a List[Struct] column where each struct has:
        - `index` (u32): Index in the corpus
        - `score` (f64): Similarity score
        
        Args:
            corpus: Series of corpus embeddings (List or Array type)
            k: Number of top matches to return per query
            metric: Similarity metric:
                - "cosine": Cosine similarity (default)
                - "dot": Raw dot product  
                - "euclidean": Euclidean distance (lower = more similar)
            
        Returns:
            Expression evaluating to List[Struct{index, score}]
            
        Example:
            >>> # Basic usage
            >>> df.with_columns(
            ...     matches=pl.col("embedding").pmm.topk(corpus["embedding"], k=5)
            ... )
            >>> 
            >>> # Explode to get flat results
            >>> df.select(
            ...     pl.col("query_id"),
            ...     pl.col("embedding").pmm.topk(corpus["embedding"], k=5).alias("match")
            ... ).explode("match").unnest("match")
            >>> 
            >>> # Join with corpus metadata
            >>> matches = df.with_columns(
            ...     pl.col("embedding").pmm.topk(corpus["embedding"], k=5).alias("match")
            ... ).explode("match").unnest("match")
            >>> 
            >>> result = matches.join(
            ...     corpus.with_row_index("index"),
            ...     on="index"
            ... )
        """
        if isinstance(corpus, pl.Expr):
            raise TypeError(
                "corpus must be a Polars Series, not an Expression. "
                "Use corpus['column_name'] or corpus.get_column('column_name')."
            )
             
        return self._expr.map_batches(
            lambda s: _topk(s, corpus, k, metric),
            is_elementwise=True,
            return_dtype=pl.List(pl.Struct({"index": pl.UInt32, "score": pl.Float64})),
        )

    def matmul(
        self,
        corpus: pl.Series,
        flatten: bool = False,
    ) -> pl.Expr:
        """
        Compute full matrix multiplication (all pairwise dot products).
        
        For each embedding in the expression, computes dot products with
        all embeddings in corpus. Useful when you need all scores, not just top-k.
        
        Args:
            corpus: Series of corpus embeddings (List or Array type)
            flatten: If True, returns scores as a 1D flat array instead of 
                per-row arrays. The flat array has shape (n_queries * n_corpus,)
                in row-major order. Useful for numpy interop.
            
        Returns:
            If flatten=False (default):
                Expression evaluating to Array[f64, N] or Array[f32, N] where N = len(corpus)
            If flatten=True:
                Expression evaluating to a single-row Series with all scores flattened
            
            (dtype is f32 if both inputs are f32, otherwise f64)
            
        Example:
            >>> # Get all pairwise scores (default - nested arrays)
            >>> df.with_columns(
            ...     scores=pl.col("embedding").pmm.matmul(corpus["embedding"])
            ... )
            >>> # Result: each row has an array of len(corpus) scores
            >>> # Access individual scores: df["scores"].arr.get(0)
            >>>
            >>> # Flatten mode for numpy interop
            >>> flat_scores = df.select(
            ...     pl.col("embedding").pmm.matmul(corpus["embedding"], flatten=True)
            ... )["embedding"].to_numpy()  # Shape: (n_queries * n_corpus,)
        """
        if isinstance(corpus, pl.Expr):
            raise TypeError(
                "corpus must be a Polars Series, not an Expression. "
                "Use corpus['column_name'] or corpus.get_column('column_name')."
            )

        # Determine inner dtype
        n_corpus = len(corpus)
        try:
            inner = corpus.dtype.inner
            is_f32 = inner == pl.Float32
        except:
            is_f32 = False

        if flatten:
            # Flatten mode: return all scores as a 1D array
            inner_dtype = pl.Float32 if is_f32 else pl.Float64
            
            def _matmul_flatten(s: pl.Series) -> pl.Series:
                result = _matmul(s, corpus)
                # Flatten the Array column to a single 1D series
                flat = result.explode()
                return flat
            
            return self._expr.map_batches(
                _matmul_flatten,
                is_elementwise=False,  # Output length differs from input
                return_dtype=inner_dtype,
            )
        else:
            # Default: return Array per row
            dtype = pl.Array(pl.Float32 if is_f32 else pl.Float64, n_corpus)
            
            return self._expr.map_batches(
                lambda s: _matmul(s, corpus),
                is_elementwise=True,
                return_dtype=dtype,
            )


def topk_search(
    queries: pl.DataFrame,
    corpus: pl.DataFrame,
    query_col: str = "embedding",
    corpus_col: str = "embedding",
    k: int = 10,
    metric: Metric = "cosine",
) -> pl.DataFrame:
    """
    Find top-k similar corpus items for each query (convenience function).
    
    This is a high-level function that handles the full similarity search workflow:
    topk -> explode -> unnest -> join with corpus.
    
    Args:
        queries: DataFrame containing query embeddings
        corpus: DataFrame containing corpus embeddings  
        query_col: Column name for query embeddings (default: "embedding")
        corpus_col: Column name for corpus embeddings (default: "embedding")
        k: Number of top matches per query
        metric: Similarity metric ("cosine", "dot", "euclidean")
        
    Returns:
        DataFrame with all query columns + "score" + all corpus columns
        (one row per query-corpus match)
        
    Example:
        >>> from polars_matmul import topk_search
        >>> 
        >>> results = topk_search(queries, corpus, k=5)
        >>> # Returns flat DataFrame ready for analysis:
        >>> # query_id | score | corpus_id | label | ...
    """
    # Get the corpus embedding series
    corpus_emb = corpus[corpus_col]
    
    # Run topk, explode, unnest
    result = (
        queries
        .with_columns(
            pl.col(query_col).pmm.topk(corpus_emb, k=k, metric=metric).alias("_pmm_match")
        )
        .explode("_pmm_match")
        .unnest("_pmm_match")
    )
    
    # Prepare corpus for join (add index, drop embedding col to avoid collision)
    corpus_for_join = corpus.with_row_index("index")
    if corpus_col in corpus_for_join.columns:
        corpus_for_join = corpus_for_join.drop(corpus_col)
    
    # Join and clean up
    result = result.join(corpus_for_join, on="index").drop("index")
    
    return result
