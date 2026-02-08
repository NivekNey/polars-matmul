//! polars-matmul: High-performance similarity search for Polars
//!
//! This crate provides fast similarity search operations using pure Rust matrix
//! multiplication (faer).

mod matmul;
mod metrics;
mod topk;

use pyo3::prelude::*;
use pyo3_polars::PySeries;
use polars::prelude::*;

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

/// Compute top-k matches for each row in left against all rows in right
#[pyfunction]
#[pyo3(signature = (left, right, k, metric))]
fn _topk(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    k: usize,
    metric: &str,
) -> PyResult<PySeries> {
    let left_series: Series = left.extract::<PySeries>()?.0;
    let right_series: Series = right.extract::<PySeries>()?.0;
    
    py.detach(|| {
        matmul::topk_impl(
            &left_series,
            &right_series,
            k,
            metric,
        )
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    .map(PySeries)
}

#[pymodule]
fn _polars_matmul(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(_topk, m)?)?;
    Ok(())
}
