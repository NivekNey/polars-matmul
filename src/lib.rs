//! polars-matmul: BLAS-accelerated similarity joins for Polars
//!
//! This crate provides fast similarity search operations using BLAS-accelerated
//! matrix multiplication.

// Force BLAS static linking - these extern crate declarations prevent
// the compiler from optimizing away the unused BLAS dependencies
#[cfg(target_os = "linux")]
extern crate openblas_src;

#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(target_os = "windows")]
extern crate openblas_src;

mod matmul;
mod metrics;
mod topk;

use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
use polars::prelude::*;

/// Perform similarity join between two DataFrames
#[pyfunction]
#[pyo3(signature = (left, right, left_on, right_on, k, metric, suffix))]
fn _similarity_join_eager(
    py: Python<'_>,
    left: PyDataFrame,
    right: PyDataFrame,
    left_on: &str,
    right_on: &str,
    k: usize,
    metric: &str,
    suffix: &str,
) -> PyResult<PyDataFrame> {
    py.detach(|| {
        let left_df: DataFrame = left.into();
        let right_df: DataFrame = right.into();
        
        let result = matmul::similarity_join_impl(
            &left_df,
            &right_df,
            left_on,
            right_on,
            k,
            metric,
            suffix,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyDataFrame(result))
    })
}

/// Compute full matrix multiplication
#[pyfunction]
#[pyo3(signature = (left, right))]
fn _matmul(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<PySeries> {
    let left_series: Series = left.extract::<PySeries>()?.0;
    let right_series: Series = right.extract::<PySeries>()?.0;
    
    py.detach(|| {
        matmul::matmul_impl(&left_series, &right_series)
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    .map(PySeries)
}

#[pymodule]
fn _polars_matmul(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_similarity_join_eager, m)?)?;
    m.add_function(wrap_pyfunction!(_matmul, m)?)?;
    Ok(())
}
