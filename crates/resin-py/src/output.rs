//! NumPy SoA transport: acquire array borrows once, retain them through commit.

use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use resin_core::logprobs::{LogprobView, TopLogprobView, validate_indptr};

use crate::core_error;

/// Borrowed NumPy SoA output, valid until the synchronous step returns.
///
/// All arrays must have their exact declared dtype, one dimension, and contiguous
/// storage. Construction retains the arrays without copying. Do not mutate or
/// recycle their storage until step() returns. Token rows use token_indptr;
/// score rows use logprob_indptr (defaults to token_indptr). An empty score row
/// opts out; every other score row must cover the entire raw output token row.
/// top_indptr partitions top_token_ids/top_logprobs by scored token. All three
/// top_* arrays are optional together.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub struct ForwardOutput {
    #[pyo3(get)]
    prompt_logprobs: Option<Py<PyArray1<f32>>>,
    #[pyo3(get)]
    prompt_logprob_indptr: Option<Py<PyArray1<u32>>>,
    #[pyo3(get)]
    token_ids: Py<PyArray1<u32>>,
    #[pyo3(get)]
    token_indptr: Py<PyArray1<u32>>,
    #[pyo3(get)]
    logprobs: Option<Py<PyArray1<f32>>>,
    #[pyo3(get)]
    logprob_indptr: Option<Py<PyArray1<u32>>>,
    #[pyo3(get)]
    top_indptr: Option<Py<PyArray1<u32>>>,
    #[pyo3(get)]
    top_token_ids: Option<Py<PyArray1<u32>>>,
    #[pyo3(get)]
    top_logprobs: Option<Py<PyArray1<f32>>>,
}

#[pymethods]
impl ForwardOutput {
    #[new]
    #[pyo3(signature = (token_ids, token_indptr, *, logprobs = None, logprob_indptr = None, top_indptr = None, top_token_ids = None, top_logprobs = None, prompt_logprobs = None, prompt_logprob_indptr = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        token_ids: Py<PyArray1<u32>>,
        token_indptr: Py<PyArray1<u32>>,
        logprobs: Option<Py<PyArray1<f32>>>,
        logprob_indptr: Option<Py<PyArray1<u32>>>,
        top_indptr: Option<Py<PyArray1<u32>>>,
        top_token_ids: Option<Py<PyArray1<u32>>>,
        top_logprobs: Option<Py<PyArray1<f32>>>,
        prompt_logprobs: Option<Py<PyArray1<f32>>>,
        prompt_logprob_indptr: Option<Py<PyArray1<u32>>>,
    ) -> PyResult<Self> {
        if prompt_logprobs.is_some() != prompt_logprob_indptr.is_some() {
            return Err(PyValueError::new_err(
                "prompt scores require both prompt_logprobs and prompt_logprob_indptr",
            ));
        }
        let top_count = usize::from(top_indptr.is_some())
            + usize::from(top_token_ids.is_some())
            + usize::from(top_logprobs.is_some());
        if top_count != 0 && top_count != 3 {
            return Err(PyValueError::new_err(
                "top logprobs require all three top_* arrays",
            ));
        }
        if logprobs.is_none() && (logprob_indptr.is_some() || top_count != 0) {
            return Err(PyValueError::new_err(
                "score offsets and alternatives require logprobs",
            ));
        }
        let logprob_indptr = logprobs
            .as_ref()
            .map(|_| logprob_indptr.unwrap_or_else(|| token_indptr.clone_ref(py)));
        Ok(Self {
            prompt_logprobs,
            prompt_logprob_indptr,
            token_ids,
            token_indptr,
            logprobs,
            logprob_indptr,
            top_indptr,
            top_token_ids,
            top_logprobs,
        })
    }
}

fn borrow<'py, T: Element>(
    array: &Py<PyArray1<T>>,
    py: Python<'py>,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    let guard = array
        .bind(py)
        .try_readonly()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    guard.as_slice().map_err(|_| {
        PyValueError::new_err(
            "ForwardOutput arrays must be contiguous; no implicit copy is performed",
        )
    })?;
    Ok(guard)
}

struct BorrowedTop<'py> {
    indptr: PyReadonlyArray1<'py, u32>,
    token_ids: PyReadonlyArray1<'py, u32>,
    values: PyReadonlyArray1<'py, f32>,
}

pub struct BorrowedLogprobs<'py> {
    indptr: PyReadonlyArray1<'py, u32>,
    values: PyReadonlyArray1<'py, f32>,
    top: Option<BorrowedTop<'py>>,
}

pub struct BorrowedOutput<'py> {
    pub tokens: Vec<Vec<u32>>,
    pub logprobs: Option<BorrowedLogprobs<'py>>,
    pub prompt_logprobs: Option<BorrowedLogprobs<'py>>,
}

impl BorrowedLogprobs<'_> {
    pub fn view(&self) -> LogprobView<'_> {
        // Contiguity is checked when each guard is acquired. Guards retain the
        // arrays (and their base owners); the synchronous step bounds the borrow.
        LogprobView {
            indptr: self.indptr.as_slice().unwrap(),
            values: self.values.as_slice().unwrap(),
            top: self.top.as_ref().map(|top| TopLogprobView {
                indptr: top.indptr.as_slice().unwrap(),
                token_ids: top.token_ids.as_slice().unwrap(),
                values: top.values.as_slice().unwrap(),
            }),
        }
    }
}

impl ForwardOutput {
    pub fn borrow<'py>(
        &self,
        py: Python<'py>,
        expected_rows: usize,
    ) -> PyResult<BorrowedOutput<'py>> {
        let tokens = borrow(&self.token_ids, py)?;
        let offsets = borrow(&self.token_indptr, py)?;
        let tokens = tokens.as_slice().unwrap();
        let offsets = offsets.as_slice().unwrap();
        validate_indptr(offsets, expected_rows, tokens.len()).map_err(core_error)?;
        // Token IDs join the core's existing owned output contract. Score and
        // alternative matrices stay in NumPy storage throughout validation/commit.
        let rows = offsets
            .windows(2)
            .map(|pair| tokens[pair[0] as usize..pair[1] as usize].to_vec())
            .collect();
        let logprobs = self
            .logprobs
            .as_ref()
            .map(|values| {
                let top = self
                    .top_indptr
                    .as_ref()
                    .map(|indptr| {
                        Ok::<_, PyErr>(BorrowedTop {
                            indptr: borrow(indptr, py)?,
                            token_ids: borrow(self.top_token_ids.as_ref().unwrap(), py)?,
                            values: borrow(self.top_logprobs.as_ref().unwrap(), py)?,
                        })
                    })
                    .transpose()?;
                Ok::<_, PyErr>(BorrowedLogprobs {
                    indptr: borrow(self.logprob_indptr.as_ref().unwrap(), py)?,
                    values: borrow(values, py)?,
                    top,
                })
            })
            .transpose()?;
        let prompt_logprobs = self
            .prompt_logprobs
            .as_ref()
            .map(|values| {
                Ok::<_, PyErr>(BorrowedLogprobs {
                    indptr: borrow(self.prompt_logprob_indptr.as_ref().unwrap(), py)?,
                    values: borrow(values, py)?,
                    top: None,
                })
            })
            .transpose()?;
        Ok(BorrowedOutput {
            tokens: rows,
            logprobs,
            prompt_logprobs,
        })
    }
}
