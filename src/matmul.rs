//! Core matrix multiplication and similarity search implementation

use ndarray::Array2;
use polars::prelude::*;
use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;
use crate::metrics::{
    Metric, compute_similarity_matrix, compute_similarity_matrix_f32, 
    matmul_f64, matmul_f32, matmul_slice_f64, matmul_slice_f32
};
use crate::topk::{select_topk_with_scores, select_topk_with_scores_f32};

/// Check if a series contains f32 data (either List[f32] or Array[f32, dim])
fn is_f32_series(series: &Series) -> bool {
    match series.dtype() {
        DataType::List(inner) => matches!(inner.as_ref(), DataType::Float32),
        DataType::Array(inner, _) => matches!(inner.as_ref(), DataType::Float32),
        _ => false,
    }
}

/// Zero-copy extraction result - owns the data to keep it alive
pub struct ContiguousData<T> {
    /// The raw slice data - valid as long as _owner is alive
    pub ptr: *const T,
    pub len: usize,
    pub n_rows: usize,
    pub dim: usize,
    /// Keeps the underlying data alive
    _owner: Series,
}

impl<T> ContiguousData<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Try to extract contiguous f64 data from a Polars Array Series
fn try_extract_contiguous_f64(series: &Series) -> Option<ContiguousData<f64>> {
    let arr = series.array().ok()?;
    let n_rows = arr.len();
    if n_rows == 0 {
        return None;
    }
    let dim = arr.width();
    if dim == 0 {
        return None;
    }
    
    let inner = arr.get_inner();
    // Check if already f64
    let ca_f64 = inner.f64().ok()?;
    let slice = ca_f64.cont_slice().ok()?;
    
    if slice.len() == n_rows * dim {
        Some(ContiguousData {
            ptr: slice.as_ptr(),
            len: slice.len(),
            n_rows,
            dim,
            _owner: series.clone(),
        })
    } else {
        None
    }
}

/// Try to extract contiguous f32 data from a Polars Array Series
fn try_extract_contiguous_f32(series: &Series) -> Option<ContiguousData<f32>> {
    let arr = series.array().ok()?;
    let n_rows = arr.len();
    if n_rows == 0 {
        return None;
    }
    let dim = arr.width();
    if dim == 0 {
        return None;
    }
    
    let inner = arr.get_inner();
    let ca_f32 = inner.f32().ok()?;
    let slice = ca_f32.cont_slice().ok()?;
    
    if slice.len() == n_rows * dim {
        Some(ContiguousData {
            ptr: slice.as_ptr(),
            len: slice.len(),
            n_rows,
            dim,
            _owner: series.clone(),
        })
    } else {
        None
    }
}

/// Efficiently convert a flat Vec<f64> to a List[f64] Series
/// Uses ListPrimitiveChunkedBuilder for minimal allocations
fn vec_to_list_series_f64(data: Vec<f64>, n_rows: usize, row_len: usize) -> PolarsResult<Series> {
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        PlSmallStr::from_static("matmul"),
        n_rows,
        row_len,
        DataType::Float64,
    );
    
    for chunk in data.chunks(row_len) {
        builder.append_slice(chunk);
    }
    
    Ok(builder.finish().into_series())
}

/// Efficiently convert a flat Vec<f32> to a List[f32] Series
fn vec_to_list_series_f32(data: Vec<f32>, n_rows: usize, row_len: usize) -> PolarsResult<Series> {
    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        PlSmallStr::from_static("matmul"),
        n_rows,
        row_len,
        DataType::Float32,
    );
    
    for chunk in data.chunks(row_len) {
        builder.append_slice(chunk);
    }
    
    Ok(builder.finish().into_series())
}

/// Convert flat Vec<f64> to Array[f64, width] Series (more efficient for fixed-size output)
/// Uses ChunkedArray::from_vec for zero-copy, then reshapes
fn vec_to_array_series_f64(data: Vec<f64>, n_rows: usize, width: usize) -> PolarsResult<Series> {
    use polars::prelude::ReshapeDimension;
    
    // Create a flat Float64Chunked from the Vec (zero-copy - takes ownership)
    let flat = Float64Chunked::from_vec("values".into(), data);
    
    // Reshape into Array type - this creates a FixedSizeList view over the same data
    let reshaped = flat.into_series().reshape_array(&[
        ReshapeDimension::new(n_rows as i64),
        ReshapeDimension::new(width as i64),
    ])?;
    
    Ok(reshaped.with_name("matmul".into()))
}

/// Convert flat Vec<f32> to Array[f32, width] Series
fn vec_to_array_series_f32(data: Vec<f32>, n_rows: usize, width: usize) -> PolarsResult<Series> {
    use polars::prelude::ReshapeDimension;
    
    let flat = Float32Chunked::from_vec("values".into(), data);
    let reshaped = flat.into_series().reshape_array(&[
        ReshapeDimension::new(n_rows as i64),
        ReshapeDimension::new(width as i64),
    ])?;
    Ok(reshaped.with_name("matmul".into()))
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
/// Automatically uses f32 operations when input is f32, providing
/// 2x memory efficiency and potentially faster computation.
/// 
/// Uses zero-copy path when data is contiguous (Array type), 
/// falls back to copying for List type.
pub fn matmul_impl(left: &Series, right: &Series) -> PolarsResult<Series> {
    // Handle empty left series - return empty result with correct type
    if left.len() == 0 {
        let inner_dtype = if is_f32_series(left) && is_f32_series(right) {
            DataType::Float32
        } else {
            DataType::Float64
        };
        let empty_series = Series::new_empty("matmul".into(), &DataType::List(Box::new(inner_dtype)));
        return Ok(empty_series);
    }
    
    // Use f32 path if both inputs are f32
    let use_f32 = is_f32_series(left) && is_f32_series(right);
    
    if use_f32 {
        matmul_impl_f32(left, right)
    } else {
        matmul_impl_f64(left, right)
    }
}

fn matmul_impl_f64(left: &Series, right: &Series) -> PolarsResult<Series> {
    // Try zero-copy path first
    if let (Some(left_data), Some(right_data)) = 
        (try_extract_contiguous_f64(left), try_extract_contiguous_f64(right)) 
    {
        if left_data.dim != right_data.dim {
            return Err(PolarsError::ComputeError(
                format!(
                    "Dimension mismatch: left has {} dimensional vectors, right has {} dimensional vectors",
                    left_data.dim, right_data.dim
                ).into()
            ));
        }
        
        // Zero-copy matmul!
        let m = left_data.n_rows;
        let n = right_data.n_rows;
        let result_vec = matmul_slice_f64(
            left_data.as_slice(), 
            right_data.as_slice(), 
            m, 
            n, 
            left_data.dim
        );
        // Use Array output (efficient - uses from_vec + reshape)
        return vec_to_array_series_f64(result_vec, m, n);
    }
    
    // Fallback: copy to ndarray
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
    
    // Use Array output for consistency
    let m = result.nrows();
    let n = result.ncols();
    let flat: Vec<f64> = result.iter().copied().collect();
    vec_to_array_series_f64(flat, m, n)
}

fn matmul_impl_f32(left: &Series, right: &Series) -> PolarsResult<Series> {
    // Try zero-copy path first
    if let (Some(left_data), Some(right_data)) = 
        (try_extract_contiguous_f32(left), try_extract_contiguous_f32(right)) 
    {
        if left_data.dim != right_data.dim {
            return Err(PolarsError::ComputeError(
                format!(
                    "Dimension mismatch: left has {} dimensional vectors, right has {} dimensional vectors",
                    left_data.dim, right_data.dim
                ).into()
            ));
        }
        
        // Zero-copy matmul!
        let m = left_data.n_rows;
        let n = right_data.n_rows;
        let result_vec = matmul_slice_f32(
            left_data.as_slice(), 
            right_data.as_slice(), 
            m, 
            n, 
            left_data.dim
        );
        // Use Array output (efficient - uses from_vec + reshape)
        return vec_to_array_series_f32(result_vec, m, n);
    }
    
    // Fallback: copy to ndarray
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
    
    // Use Array output for consistency
    let m = result.nrows();
    let n = result.ncols();
    let flat: Vec<f32> = result.iter().copied().collect();
    vec_to_array_series_f32(flat, m, n)
}

/// Helper to compute top-k indices and scores
fn compute_topk_indices_scores(
    queries: &Series,
    corpus: &Series,
    k: usize,
    metric: Metric,
) -> PolarsResult<(ndarray::Array2<usize>, Vec<f64>)> {
    // Use f32 path if both inputs are f32
    let use_f32 = is_f32_series(queries) && is_f32_series(corpus);
    
    if use_f32 {
        let query_matrix = series_to_matrix_f32(queries)?;
        let corpus_matrix = series_to_matrix_f32(corpus)?;
        
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
        Ok((indices, scores_f64))
    } else {
        let query_matrix = series_to_matrix(queries)?;
        let corpus_matrix = series_to_matrix(corpus)?;
        
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
        Ok((indices, scores_vec))
    }
}

/// Top-k implementation for Expression API
/// Returns a Series of List[Struct { index: u32, score: f64 }]
pub fn topk_impl(
    queries: &Series,
    corpus: &Series,
    k: usize,
    metric_str: &str,
) -> PolarsResult<Series> {
    // Handle empty queries - return empty series with correct type
    if queries.len() == 0 {
        let struct_dtype = DataType::Struct(vec![
            Field::new("index".into(), DataType::UInt32),
            Field::new("score".into(), DataType::Float64),
        ]);
        let empty_series = Series::new_empty("topk".into(), &DataType::List(Box::new(struct_dtype)));
        return Ok(empty_series);
    }
    
    let metric = Metric::from_str(metric_str)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
        
    let (indices, scores) = compute_topk_indices_scores(queries, corpus, k, metric)?;
    
    let n_queries = indices.nrows();
    let k_actual = indices.ncols();
    
    let mut list_rows_series: Vec<Series> = Vec::with_capacity(n_queries);
    
    for i in 0..n_queries {
        let row_indices_vals = indices.row(i);
        // Extract scores for this row from the flattened scores vec
        let row_start = i * k_actual;
        let row_end = row_start + k_actual;
        let row_scores_vals = &scores[row_start..row_end];
        
        let idx_vec: Vec<u32> = row_indices_vals.iter().map(|&x| x as u32).collect();
        let score_vec: Vec<f64> = row_scores_vals.into();
        
        let s_idx = Series::new("index".into(), idx_vec);
        let s_score = Series::new("score".into(), score_vec);
        
        let struct_df = DataFrame::new(vec![s_idx.into_column(), s_score.into_column()])?;
        let struct_series = struct_df.into_struct("match".into()).into_series();
        
        list_rows_series.push(struct_series);
    }
    
    Ok(Series::new("topk".into(), list_rows_series))
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
}
