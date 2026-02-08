//! Similarity metrics implementation

use ndarray::{Array1, Array2, Axis};
use faer::mat::{from_raw_parts, from_raw_parts_mut};
use faer::linalg::matmul::matmul;
use faer::Parallelism;

/// Supported similarity/distance metrics
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    /// Cosine similarity: dot(a, b) / (||a|| * ||b||)
    Cosine,
    /// Raw dot product
    Dot,
    /// Euclidean distance (L2)
    Euclidean,
}

impl Metric {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Metric::Cosine),
            "dot" => Ok(Metric::Dot),
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            _ => Err(format!("Unknown metric: '{}'. Supported: cosine, dot, euclidean", s)),
        }
    }
    
    /// Returns true if higher scores are better (similarity), false if lower is better (distance)
    pub fn higher_is_better(&self) -> bool {
        match self {
            Metric::Cosine | Metric::Dot => true,
            Metric::Euclidean => false,
        }
    }
}

/// Compute matrix multiplication C = A * B^T using faer (0.19)
/// Note: This computes A * B^T because that's what we need for similarity search (Query * Corpus^T)
pub fn matmul_f64(query: &Array2<f64>, corpus: &Array2<f64>) -> Array2<f64> {
    let m = query.nrows();
    let k = query.ncols();
    let n = corpus.nrows();
    
    // Ensure dimensions match for Query * Corpus^T
    // Query: M x K
    // Corpus: N x K
    assert_eq!(corpus.ncols(), k, "Matrix dimensions mismatch for matmul");
    
    let mut result = Array2::<f64>::zeros((m, n));
    
    let q_strides = query.strides();
    let rsa = q_strides[0];
    let csa = q_strides[1];
    
    let c_strides = corpus.strides();
    let rsb = c_strides[0];
    let csb = c_strides[1];
    
    let r_strides = result.strides();
    let rsc = r_strides[0];
    let csc = r_strides[1];
    
    unsafe {
        // Create views with arbitrary strides
        // from_raw_parts(ptr, nrows, ncols, row_stride, col_stride)
        let lhs = from_raw_parts::<f64>(
            query.as_ptr(),
            m, k,
            rsa, csa,
        );
        
        let rhs = from_raw_parts::<f64>(
            corpus.as_ptr(),
            n, k,
            rsb, csb,
        );
        
        let dest = from_raw_parts_mut::<f64>(
            result.as_mut_ptr(),
            m, n,
            rsc, csc,
        );
        
        // Compute dest = lhs * rhs^T
        matmul(
            dest,
            lhs,
            rhs.transpose(),
            None,
            1.0,
            Parallelism::Rayon(0), // Use Rayon(0) for auto parallelism
        );
    }
    
    result
}

/// Zero-copy matmul: C = A * B^T using faer directly on raw slices
/// This avoids the ndarray allocation entirely.
/// 
/// # Arguments
/// * `query_slice` - Contiguous slice of query data, row-major (m * k elements)
/// * `corpus_slice` - Contiguous slice of corpus data, row-major (n * k elements)
/// * `m` - Number of query rows
/// * `n` - Number of corpus rows
/// * `k` - Vector dimension
/// 
/// # Returns
/// Vec<f64> of size m*n in row-major order
pub fn matmul_slice_f64(
    query_slice: &[f64],
    corpus_slice: &[f64],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f64> {
    assert_eq!(query_slice.len(), m * k, "Query slice size mismatch");
    assert_eq!(corpus_slice.len(), n * k, "Corpus slice size mismatch");
    
    let mut result = vec![0.0f64; m * n];
    
    unsafe {
        // Query: m x k, row-major (row_stride = k, col_stride = 1)
        let lhs = from_raw_parts::<f64>(
            query_slice.as_ptr(),
            m, k,
            k as isize, 1,
        );
        
        // Corpus: n x k, row-major (row_stride = k, col_stride = 1)
        let rhs = from_raw_parts::<f64>(
            corpus_slice.as_ptr(),
            n, k,
            k as isize, 1,
        );
        
        // Result: m x n, row-major (row_stride = n, col_stride = 1)
        let dest = from_raw_parts_mut::<f64>(
            result.as_mut_ptr(),
            m, n,
            n as isize, 1,
        );
        
        // Compute dest = lhs * rhs^T
        matmul(
            dest,
            lhs,
            rhs.transpose(),
            None,
            1.0,
            Parallelism::Rayon(0),
        );
    }
    
    result
}

/// Zero-copy matmul for f32
pub fn matmul_slice_f32(
    query_slice: &[f32],
    corpus_slice: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(query_slice.len(), m * k, "Query slice size mismatch");
    assert_eq!(corpus_slice.len(), n * k, "Corpus slice size mismatch");
    
    let mut result = vec![0.0f32; m * n];
    
    unsafe {
        let lhs = from_raw_parts::<f32>(
            query_slice.as_ptr(),
            m, k,
            k as isize, 1,
        );
        
        let rhs = from_raw_parts::<f32>(
            corpus_slice.as_ptr(),
            n, k,
            k as isize, 1,
        );
        
        let dest = from_raw_parts_mut::<f32>(
            result.as_mut_ptr(),
            m, n,
            n as isize, 1,
        );
        
        matmul(
            dest,
            lhs,
            rhs.transpose(),
            None,
            1.0,
            Parallelism::Rayon(0),
        );
    }
    
    result
}

pub fn matmul_f32(query: &Array2<f32>, corpus: &Array2<f32>) -> Array2<f32> {
    let m = query.nrows();
    let k = query.ncols();
    let n = corpus.nrows();
    
    assert_eq!(corpus.ncols(), k, "Matrix dimensions mismatch for matmul");
    
    let mut result = Array2::<f32>::zeros((m, n));
    
    let q_strides = query.strides();
    let rsa = q_strides[0];
    let csa = q_strides[1];
    
    let c_strides = corpus.strides();
    let rsb = c_strides[0];
    let csb = c_strides[1];
    
    let r_strides = result.strides();
    let rsc = r_strides[0];
    let csc = r_strides[1];
    
    unsafe {
        let lhs = from_raw_parts::<f32>(
            query.as_ptr(),
            m, k,
            rsa, csa,
        );
        
        let rhs = from_raw_parts::<f32>(
            corpus.as_ptr(),
            n, k,
            rsb, csb,
        );
        
        let dest = from_raw_parts_mut::<f32>(
            result.as_mut_ptr(),
            m, n,
            rsc, csc,
        );
        
        matmul(
            dest,
            lhs,
            rhs.transpose(),
            None,
            1.0,
            Parallelism::Rayon(0),
        );
    }
    
    result
}

/// Compute similarity/distance matrix between query and corpus matrices (f64)
pub fn compute_similarity_matrix(
    query: &Array2<f64>,
    corpus: &Array2<f64>,
    metric: Metric,
) -> Array2<f64> {
    match metric {
        Metric::Dot => {
            matmul_f64(query, corpus)
        }
        Metric::Cosine => {
            // Normalize vectors then dot product
            let query_norms = compute_norms_f64(query);
            let corpus_norms = compute_norms_f64(corpus);
            
            let mut result = matmul_f64(query, corpus);
            
            // Divide by norms
            for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
                let q_norm = query_norms[i];
                if q_norm > 1e-10 {
                    for (j, val) in row.iter_mut().enumerate() {
                        let c_norm = corpus_norms[j];
                        if c_norm > 1e-10 {
                            *val /= q_norm * c_norm;
                        } else {
                            *val = 0.0;
                        }
                    }
                } else {
                    row.fill(0.0);
                }
            }
            result
        }
        Metric::Euclidean => {
            let query_sq_norms = compute_squared_norms_f64(query);
            let corpus_sq_norms = compute_squared_norms_f64(corpus);
            
            let dot_products = matmul_f64(query, corpus);
            
            let n_queries = query.nrows();
            let n_corpus = corpus.nrows();
            let mut result = Array2::zeros((n_queries, n_corpus));
            
            for i in 0..n_queries {
                for j in 0..n_corpus {
                    let sq_dist = query_sq_norms[i] + corpus_sq_norms[j] - 2.0 * dot_products[[i, j]];
                    result[[i, j]] = sq_dist.max(0.0).sqrt();
                }
            }
            result
        }
    }
}

/// Compute similarity/distance matrix between query and corpus matrices (f32)
pub fn compute_similarity_matrix_f32(
    query: &Array2<f32>,
    corpus: &Array2<f32>,
    metric: Metric,
) -> Array2<f32> {
    match metric {
        Metric::Dot => {
            matmul_f32(query, corpus)
        }
        Metric::Cosine => {
            let query_norms = compute_norms_f32(query);
            let corpus_norms = compute_norms_f32(corpus);
            
            let mut result = matmul_f32(query, corpus);
            
            for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
                let q_norm = query_norms[i];
                if q_norm > 1e-6 {
                    for (j, val) in row.iter_mut().enumerate() {
                        let c_norm = corpus_norms[j];
                        if c_norm > 1e-6 {
                            *val /= q_norm * c_norm;
                        } else {
                            *val = 0.0;
                        }
                    }
                } else {
                    row.fill(0.0);
                }
            }
            result
        }
        Metric::Euclidean => {
            let query_sq_norms = compute_squared_norms_f32(query);
            let corpus_sq_norms = compute_squared_norms_f32(corpus);
            
            let dot_products = matmul_f32(query, corpus);
            
            let n_queries = query.nrows();
            let n_corpus = corpus.nrows();
            let mut result = Array2::zeros((n_queries, n_corpus));
            
            for i in 0..n_queries {
                for j in 0..n_corpus {
                    let sq_dist = query_sq_norms[i] + corpus_sq_norms[j] - 2.0 * dot_products[[i, j]];
                    result[[i, j]] = sq_dist.max(0.0).sqrt();
                }
            }
            result
        }
    }
}

/// Compute L2 norms for each row (f64)
fn compute_norms_f64(matrix: &Array2<f64>) -> Array1<f64> {
    matrix.map_axis(Axis(1), |row| {
        row.dot(&row).sqrt()
    })
}

/// Compute squared L2 norms for each row (f64)
fn compute_squared_norms_f64(matrix: &Array2<f64>) -> Array1<f64> {
    matrix.map_axis(Axis(1), |row| {
        row.dot(&row)
    })
}

/// Compute L2 norms for each row (f32)
fn compute_norms_f32(matrix: &Array2<f32>) -> Array1<f32> {
    matrix.map_axis(Axis(1), |row| {
        row.dot(&row).sqrt()
    })
}

/// Compute squared L2 norms for each row (f32)
fn compute_squared_norms_f32(matrix: &Array2<f32>) -> Array1<f32> {
    matrix.map_axis(Axis(1), |row| {
        row.dot(&row)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_dot_product_f64() {
        let query = array![[1.0f64, 0.0], [0.0, 1.0]];
        let corpus = array![[1.0f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
        
        let result = compute_similarity_matrix(&query, &corpus, Metric::Dot);
        
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_dot_product_f32() {
        let query = array![[1.0f32, 0.0], [0.0, 1.0]];
        let corpus = array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]];
        
        let result = compute_similarity_matrix_f32(&query, &corpus, Metric::Dot);
        
        assert!((result[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result[[0, 1]] - 0.0).abs() < 1e-5);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let query = array![[1.0f64, 0.0], [0.0, 1.0]];
        let corpus = array![[2.0f64, 0.0], [0.0, 3.0]];
        
        let result = compute_similarity_matrix(&query, &corpus, Metric::Cosine);
        
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 0.0).abs() < 1e-10);
    }
}
