//! Python adapter for the shared Rust execution layout.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyTuple};
use resin_core::{Parallelism as CoreParallelism, ParallelismError};

fn layout_error(error: ParallelismError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn strict_index(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(PyValueError::new_err("expected a nonnegative integer"));
    }
    value
        .extract()
        .map_err(|_| PyValueError::new_err("expected a nonnegative integer"))
}

/// Attention and expert meshes over the same GPU ranks.
///
/// Attention uses [ATTN_DP, CP, REPLICATION, ATTN_TP]; unassigned attention
/// ranks replicate the same request. Experts use [EP, EXPERT_TP], where
/// EXPERT_TP is world_size // ep. CP shards prefill and replicates decode.
#[pyclass(frozen, eq, hash, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct Parallelism {
    inner: CoreParallelism,
}

#[pymethods]
impl Parallelism {
    #[new]
    #[pyo3(signature = (
        *,
        world_size: "int" = 1,
        attn_dp: "int" = 1,
        attn_tp: "int" = 1,
        cp: "int" = 1,
        ep: "int" = 1
    ))]
    fn new(
        #[pyo3(from_py_with = strict_index)] world_size: usize,
        #[pyo3(from_py_with = strict_index)] attn_dp: usize,
        #[pyo3(from_py_with = strict_index)] attn_tp: usize,
        #[pyo3(from_py_with = strict_index)] cp: usize,
        #[pyo3(from_py_with = strict_index)] ep: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreParallelism::new(world_size, attn_dp, attn_tp, cp, ep)
                .map_err(layout_error)?,
        })
    }

    #[getter]
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }

    #[getter]
    fn attn_dp(&self) -> usize {
        self.inner.attn_dp()
    }

    #[getter]
    fn attn_tp(&self) -> usize {
        self.inner.attn_tp()
    }

    #[getter]
    fn cp(&self) -> usize {
        self.inner.cp()
    }

    #[getter]
    fn ep(&self) -> usize {
        self.inner.ep()
    }

    #[getter]
    fn expert_tp(&self) -> usize {
        self.inner.expert_tp()
    }

    #[getter]
    fn attn_replication(&self) -> usize {
        self.inner.attn_replication()
    }

    #[pyo3(signature = (rank: "int"))]
    fn coordinates(
        &self,
        #[pyo3(from_py_with = strict_index)] rank: usize,
    ) -> PyResult<(usize, usize, usize, usize)> {
        let [dp, cp, replication, tp] = self.inner.coordinates(rank).map_err(layout_error)?;
        Ok((dp, cp, replication, tp))
    }

    #[pyo3(signature = (attn_dp: "int", cp: "int", replication: "int", attn_tp: "int"))]
    fn rank(
        &self,
        #[pyo3(from_py_with = strict_index)] attn_dp: usize,
        #[pyo3(from_py_with = strict_index)] cp: usize,
        #[pyo3(from_py_with = strict_index)] replication: usize,
        #[pyo3(from_py_with = strict_index)] attn_tp: usize,
    ) -> PyResult<usize> {
        self.inner
            .rank(attn_dp, cp, replication, attn_tp)
            .map_err(layout_error)
    }

    #[pyo3(signature = () -> "dict[str, tuple[tuple[int, ...], ...]]")]
    fn partitions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (name, groups) in self.inner.partitions() {
            let groups = groups
                .into_iter()
                .map(|ranks| PyTuple::new(py, ranks))
                .collect::<PyResult<Vec<_>>>()?;
            result.set_item(name, PyTuple::new(py, groups)?)?;
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "Parallelism(world_size={}, attn_dp={}, attn_tp={}, cp={}, ep={})",
            self.world_size(),
            self.attn_dp(),
            self.attn_tp(),
            self.cp(),
            self.ep()
        )
    }

    #[pyo3(signature = () -> "tuple[tuple[()], dict[str, int]]")]
    fn __getnewargs_ex__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let kwargs = PyDict::new(py);
        for (name, value) in [
            ("world_size", self.world_size()),
            ("attn_dp", self.attn_dp()),
            ("attn_tp", self.attn_tp()),
            ("cp", self.cp()),
            ("ep", self.ep()),
        ] {
            kwargs.set_item(name, value)?;
        }
        Ok((PyTuple::empty(py), kwargs))
    }
}
