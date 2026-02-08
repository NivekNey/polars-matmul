//! Core matrix multiplication and similarity join implementation

use ndarray::Array2;
use polars::prelude::*;
use crate::metrics::{Metric, compute_similarity_matrix, compute_similarity_matrix_f32, matmul_f64, matmul_f32};
use crate::topk::{select_topk_with_scores, select_topk_with_scores_f32};

/// Check if a series contains f32 data (either List[f32] or Array[f32, dim])
fn is_f32_series(series: &Series) -> bool {
    match series.dtype() {
        DataType::List(inner) => matches!(inner.as_ref(), DataType::Float32),
        DataType::Array(inner, _) => matches!(inner.as_ref(), DataType::Float32),
        _ => false,
    }
}

/// Convert a Polars Series of List/Array to an ndarray matrix (f64)
/// 
/// Optimized for fixed-size arrays (Array[f64, dim]) where we can extract
/// the contiguous buffer directly, avoiding O(n*d) element-by-element iteration.
pub fn series_to_matrix(series: &Series) -> PolarsResult<Array2<f64>> {
    let n_rows = series.len();
    if n_rows == 0 {
        return Err(PolarsError::ComputeError("Empty series".into()));
    }
    
    // Try fast path for fixed-size arrays first
    if let Ok(arr_chunked) = series.array() {
        return array_chunked_to_matrix_f64(arr_chunked);
    }
    
    // Fall back to List type handling
    let series = series.cast(&DataType::List(Box::new(DataType::Float64)))?;
    let ca = series.list()?;
    list_chunked_to_matrix_f64(ca)
}

/// Convert a Polars Series of List/Array to an ndarray matrix (f32)
pub fn series_to_matrix_f32(series: &Series) -> PolarsResult<Array2<f32>> {
    let n_rows = series.len();
    if n_rows == 0 {
        return Err(PolarsError::ComputeError("Empty series".into()));
    }
    
    // Try fast path for fixed-size arrays first
    if let Ok(arr_chunked) = series.array() {
        return array_chunked_to_matrix_f32(arr_chunked);
    }
    
    // Fall back to List type handling
    let series = series.cast(&DataType::List(Box::new(DataType::Float32)))?;
    let ca = series.list()?;
    list_chunked_to_matrix_f32(ca)
}

/// Fast path: Convert ArrayChunked to ndarray matrix (f64)
fn array_chunked_to_matrix_f64(arr: &ArrayChunked) -> PolarsResult<Array2<f64>> {
    let n_rows = arr.len();
    if n_rows == 0 {
        return Err(PolarsError::ComputeError("Empty series".into()));
    }
    
    let dim = arr.width();
    if dim == 0 {
        return Err(PolarsError::ComputeError("Zero-dimensional vectors".into()));
    }
    
    let inner = arr.get_inner();
    let inner_f64 = inner.cast(&DataType::Float64)?;
    let ca_f64 = inner_f64.f64()?;
    
    if let Ok(slice) = ca_f64.cont_slice() {
        let matrix = Array2::from_shape_vec((n_rows, dim), slice.to_vec())
            .map_err(|e| PolarsError::ComputeError(format!("Shape error: {}", e).into()))?;
        return Ok(matrix);
    }
    
    let mut matrix = Array2::zeros((n_rows, dim));
    for (idx, val) in ca_f64.iter().enumerate() {
        let row = idx / dim;
        let col = idx % dim;
        matrix[[row, col]] = val.unwrap_or(0.0);
    }
    
    Ok(matrix)
}

/// Fast path: Convert ArrayChunked to ndarray matrix (f32)
fn array_chunked_to_matrix_f32(arr: &ArrayChunked) -> PolarsResult<Array2<f32>> {
    let n_rows = arr.len();
    if n_rows == 0 {
        return Err(PolarsError::ComputeError("Empty series".into()));
    }
    
    let dim = arr.width();
    if dim == 0 {
        return Err(PolarsError::ComputeError("Zero-dimensional vectors".into()));
    }
    
    let inner = arr.get_inner();
    let inner_f32 = inner.cast(&DataType::Float32)?;
    let ca_f32 = inner_f32.f32()?;
    
    if let Ok(slice) = ca_f32.cont_slice() {
        let matrix = Array2::from_shape_vec((n_rows, dim), slice.to_vec())
            .map_err(|e| PolarsError::ComputeError(format!("Shape error: {}", e).into()))?;
        return Ok(matrix);
    }
    
    let mut matrix = Array2::zeros((n_rows, dim));
    for (idx, val) in ca_f32.iter().enumerate() {
        let row = idx / dim;
        let col = idx % dim;
        matrix[[row, col]] = val.unwrap_or(0.0);
    }
    
    Ok(matrix)
}

/// Slow path: Convert ListChunked to ndarray matrix (f64)
fn list_chunked_to_matrix_f64(ca: &ListChunked) -> PolarsResult<Array2<f64>> {
    let n_rows = ca.len();
    if n_rows == 0 {
        return Err(PolarsError::ComputeError("Empty series".into()));
    }
    
    let dim = ca.get_as_series(0)
        .ok_or_else(|| PolarsError::ComputeError("First element is null".into()))?
        .len();
    
    if dim == 0 {
        return Err(PolarsError::ComputeError("Zero-dimensional vectors".into()));
    }
    
    let mut matrix = Array2::zeros((n_rows, dim));
    
    for i in 0..n_rows {
        if let Some(inner) = ca.get_as_series(i) {
            let values = inner.f64()?;
            for (j, val) in values.iter().enumerate() {
                matrix[[i, j]] = val.unwrap_or(0.0);
            }
        }
    }
    
    Ok(matrix)
}

/// Slow path: Convert ListChunked to ndarray matrix (f32)
fn list_chunked_to_matrix_f32(ca: &ListChunked) -> PolarsResult<Array2<f32>> {
    let n_rows = ca.len();
    if n_rows == 0 {
        return Err(PolarsError::ComputeError("Empty series".into()));
    }
    
    let dim = ca.get_as_series(0)
        .ok_or_else(|| PolarsError::ComputeError("First element is null".into()))?
        .len();
    
    if dim == 0 {
        return Err(PolarsError::ComputeError("Zero-dimensional vectors".into()));
    }
    
    let mut matrix = Array2::zeros((n_rows, dim));
    
    for i in 0..n_rows {
        if let Some(inner) = ca.get_as_series(i) {
            let values = inner.f32()?;
            for (j, val) in values.iter().enumerate() {
                matrix[[i, j]] = val.unwrap_or(0.0);
            }
        }
    }
    
    Ok(matrix)
}

/// Compute full matrix multiplication between two Series
/// 
/// Automatically uses f32 BLAS operations when input is f32, providing
/// 2x memory efficiency and potentially faster computation.
pub fn matmul_impl(left: &Series, right: &Series) -> PolarsResult<Series> {
    // Use f32 path if both inputs are f32
    let use_f32 = is_f32_series(left) && is_f32_series(right);
    
    if use_f32 {
        matmul_impl_f32(left, right)
    } else {
        matmul_impl_f64(left, right)
    }
}

fn matmul_impl_f64(left: &Series, right: &Series) -> PolarsResult<Series> {
    let left_matrix = series_to_matrix(left)?;
    let right_matrix = series_to_matrix(right)?;
    
    if left_matrix.ncols() != right_matrix.ncols() {
        return Err(PolarsError::ComputeError(
            format!(
                "Dimension mismatch: left has {} dimensional vectors, right has {} dimensional vectors",
                left_matrix.ncols(),
                right_matrix.ncols()
            ).into()
        ));
    }
    
    let result = matmul_f64(&left_matrix, &right_matrix);
    
    let lists: Vec<Series> = result
        .outer_iter()
        .enumerate()
        .map(|(i, row)| {
            let values: Vec<f64> = row.to_vec();
            Series::new(format!("{}", i).into(), values)
        })
        .collect();
    
    Series::new("matmul".into(), lists).cast(&DataType::List(Box::new(DataType::Float64)))
}

fn matmul_impl_f32(left: &Series, right: &Series) -> PolarsResult<Series> {
    let left_matrix = series_to_matrix_f32(left)?;
    let right_matrix = series_to_matrix_f32(right)?;
    
    if left_matrix.ncols() != right_matrix.ncols() {
        return Err(PolarsError::ComputeError(
            format!(
                "Dimension mismatch: left has {} dimensional vectors, right has {} dimensional vectors",
                left_matrix.ncols(),
                right_matrix.ncols()
            ).into()
        ));
    }
    
    let result = matmul_f32(&left_matrix, &right_matrix);
    
    // Return f32 output for memory efficiency
    let lists: Vec<Series> = result
        .outer_iter()
        .enumerate()
        .map(|(i, row)| {
            let values: Vec<f32> = row.to_vec();
            Series::new(format!("{}", i).into(), values)
        })
        .collect();
    
    Series::new("matmul".into(), lists).cast(&DataType::List(Box::new(DataType::Float32)))
}

/// Main similarity join implementation
/// 
/// Uses f32 operations internally when input is f32 for memory efficiency,
/// but always returns f64 scores for precision.
pub fn similarity_join_impl(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &str,
    right_on: &str,
    k: usize,
    metric_str: &str,
    suffix: &str,
) -> PolarsResult<DataFrame> {
    let metric = Metric::from_str(metric_str)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    
    let left_embeddings = left.column(left_on)?;
    let right_embeddings = right.column(right_on)?;
    
    // Use f32 path if both inputs are f32
    let use_f32 = is_f32_series(left_embeddings.as_materialized_series()) 
        && is_f32_series(right_embeddings.as_materialized_series());
    
    // Compute topk - separate paths for f32 and f64
    // Both return (indices, scores as f64 Vec for uniform handling)
    let (topk_indices, topk_scores): (ndarray::Array2<usize>, Vec<f64>) = if use_f32 {
        let query_matrix = series_to_matrix_f32(left_embeddings.as_materialized_series())?;
        let corpus_matrix = series_to_matrix_f32(right_embeddings.as_materialized_series())?;
        
        if query_matrix.ncols() != corpus_matrix.ncols() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Dimension mismatch: left has {} dimensional vectors, right has {} dimensional vectors",
                    query_matrix.ncols(),
                    corpus_matrix.ncols()
                ).into()
            ));
        }
        
        let k = k.min(corpus_matrix.nrows());
        let similarity = compute_similarity_matrix_f32(&query_matrix, &corpus_matrix, metric);
        let (indices, scores_f32) = select_topk_with_scores_f32(&similarity, k, metric.higher_is_better());
        // Convert f32 scores to f64 Vec for uniform output
        let scores_f64: Vec<f64> = scores_f32.iter().map(|&x| x as f64).collect();
        (indices, scores_f64)
    } else {
        let query_matrix = series_to_matrix(left_embeddings.as_materialized_series())?;
        let corpus_matrix = series_to_matrix(right_embeddings.as_materialized_series())?;
        
        if query_matrix.ncols() != corpus_matrix.ncols() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Dimension mismatch: left has {} dimensional vectors, right has {} dimensional vectors",
                    query_matrix.ncols(),
                    corpus_matrix.ncols()
                ).into()
            ));
        }
        
        let k = k.min(corpus_matrix.nrows());
        let similarity = compute_similarity_matrix(&query_matrix, &corpus_matrix, metric);
        let (indices, scores_f64) = select_topk_with_scores(&similarity, k, metric.higher_is_better());
        let scores_vec: Vec<f64> = scores_f64.iter().copied().collect();
        (indices, scores_vec)
    };
    
    // Build result DataFrame
    let left_col_names: Vec<PlSmallStr> = left.get_columns()
        .iter()
        .map(|c| c.name().clone())
        .collect();
    
    let k = topk_indices.ncols();
    
    let mut all_cols: Vec<Column> = Vec::new();
    for col in left.get_columns() {
        let col_name = col.name();
        if col_name.as_str() == left_on {
            continue;
        }
        let expanded = repeat_each(col.as_materialized_series(), k)?;
        all_cols.push(expanded.into_column());
    }
    
    let flat_indices: Vec<u32> = topk_indices.iter().map(|&x| x as u32).collect();
    let indices_series = Series::new("idx".into(), flat_indices);
    let idx_ca = indices_series.idx()?;
    
    for col in right.get_columns() {
        let col_name = col.name().clone();
        if col_name.as_str() == right_on {
            continue;
        }
        
        let gathered = unsafe { col.as_materialized_series().take_unchecked(idx_ca) };
        
        let new_name = if left_col_names.contains(&col_name) {
            PlSmallStr::from(format!("{}{}", col_name, suffix))
        } else {
            col_name
        };
        
        all_cols.push(gathered.with_name(new_name).into_column());
    }
    
    // Scores already collected as f64 Vec
    let score_series = Series::new("_score".into(), topk_scores);
    all_cols.push(score_series.into_column());
    
    DataFrame::new(all_cols)
}

/// Repeat each element of a Series k times
fn repeat_each(series: &Series, k: usize) -> PolarsResult<Series> {
    let n = series.len();
    let indices: Vec<u32> = (0..n as u32)
        .flat_map(|i| std::iter::repeat_n(i, k))
        .collect();
    
    let idx_series = Series::new("idx".into(), indices);
    let idx_ca = idx_series.idx()?;
    
    Ok(unsafe { series.take_unchecked(idx_ca) })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_series_to_matrix_f64() {
        let s = Series::new("test".into(), vec![
            Series::new("".into(), vec![1.0f64, 2.0, 3.0]),
            Series::new("".into(), vec![4.0f64, 5.0, 6.0]),
        ]);
        
        let matrix = series_to_matrix(&s).unwrap();
        
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 3);
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((matrix[[1, 2]] - 6.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_series_to_matrix_f32() {
        let s = Series::new("test".into(), vec![
            Series::new("".into(), vec![1.0f32, 2.0, 3.0]),
            Series::new("".into(), vec![4.0f32, 5.0, 6.0]),
        ]);
        
        let matrix = series_to_matrix_f32(&s).unwrap();
        
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 3);
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((matrix[[1, 2]] - 6.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_repeat_each() {
        let s = Series::new("test".into(), vec![1i32, 2, 3]);
        let repeated = repeat_each(&s, 2).unwrap();
        
        assert_eq!(repeated.len(), 6);
        let values: Vec<i32> = repeated.i32().unwrap().into_no_null_iter().collect();
        assert_eq!(values, vec![1, 1, 2, 2, 3, 3]);
    }
}
