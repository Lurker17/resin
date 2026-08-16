//! Python adapters for the pure Rust attention-KV descriptors in `resin-core`.

use std::num::NonZeroU32;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use resin_core::resources::kv::{
    KvGroupLayout, KvGroupPlan, KvLayout, KvLayoutError, KvPlanError, KvRetention,
};

fn layout_error(error: KvLayoutError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn plan_error(error: KvPlanError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Cache-block lifetime policy.
///
/// ``HISTORY`` retains every logical block. ``WINDOW`` retains only the
/// rolling suffix selected by ``CacheGroupLayout.window_tokens``.
#[pyclass(
    frozen,
    eq,
    hash,
    rename_all = "SCREAMING_SNAKE_CASE",
    from_py_object,
    module = "resin._resin"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CacheRetention {
    History,
    Window,
}

fn group_repr(group: KvGroupLayout) -> String {
    match group.retention() {
        KvRetention::FullContext => format!(
            "CacheGroupLayout(block_tokens={}, blocks_per_parent={}, retention=CacheRetention.HISTORY)",
            group.block_tokens(),
            group.blocks_per_parent()
        ),
        KvRetention::SlidingWindow { tokens } => format!(
            "CacheGroupLayout(block_tokens={}, blocks_per_parent={}, retention=CacheRetention.WINDOW, window_tokens={tokens})",
            group.block_tokens(),
            group.blocks_per_parent()
        ),
    }
}

/// Geometry and retention policy for one logical cache group.
///
/// Groups with different tensor layouts share the parent pool. A group packs
/// ``blocks_per_parent`` logical blocks into each equally-sized parent.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheGroupLayout {
    inner: KvGroupLayout,
}

impl CacheGroupLayout {
    pub(crate) fn from_parts(
        block_tokens: u32,
        blocks_per_parent: u32,
        retention: CacheRetention,
        window_tokens: Option<u32>,
    ) -> PyResult<Self> {
        let retention = match retention {
            CacheRetention::History => {
                if window_tokens.is_some() {
                    return Err(PyValueError::new_err(
                        "history retention does not accept window_tokens",
                    ));
                }
                KvRetention::FullContext
            }
            CacheRetention::Window => {
                let tokens = window_tokens.ok_or_else(|| {
                    PyValueError::new_err("window retention requires window_tokens")
                })?;
                let tokens = NonZeroU32::new(tokens)
                    .ok_or_else(|| PyValueError::new_err("window_tokens must be positive"))?;
                KvRetention::SlidingWindow { tokens }
            }
        };
        Ok(Self {
            inner: KvGroupLayout::new(block_tokens, blocks_per_parent, retention)
                .map_err(layout_error)?,
        })
    }

    fn from_inner(inner: KvGroupLayout) -> Self {
        Self { inner }
    }

    #[cfg(test)]
    pub(crate) const fn inner(&self) -> KvGroupLayout {
        self.inner
    }

    fn repr_string(&self) -> String {
        group_repr(self.inner)
    }
}

#[pymethods]
impl CacheGroupLayout {
    #[new]
    #[pyo3(signature = (
        block_tokens,
        *,
        blocks_per_parent = 1,
        retention = CacheRetention::History,
        window_tokens = None
    ))]
    fn new(
        block_tokens: u32,
        blocks_per_parent: u32,
        retention: CacheRetention,
        window_tokens: Option<u32>,
    ) -> PyResult<Self> {
        Self::from_parts(block_tokens, blocks_per_parent, retention, window_tokens)
    }

    #[getter]
    /// Number of token positions represented by one logical block.
    fn block_tokens(&self) -> u32 {
        self.inner.block_tokens().get()
    }

    #[getter]
    /// Number of this group's logical blocks packed into one parent allocation.
    fn blocks_per_parent(&self) -> u32 {
        self.inner.blocks_per_parent().get()
    }

    #[getter]
    /// Lifetime policy for logical blocks in this group.
    fn retention(&self) -> CacheRetention {
        match self.inner.retention() {
            KvRetention::FullContext => CacheRetention::History,
            KvRetention::SlidingWindow { .. } => CacheRetention::Window,
        }
    }

    #[getter]
    /// Retained suffix length for ``WINDOW`` groups, otherwise ``None``.
    fn window_tokens(&self) -> Option<u32> {
        self.inner.retention().window_tokens().map(NonZeroU32::get)
    }

    #[getter]
    /// Maximum resident logical blocks per sequence for a window group.
    fn window_resident_block_count(&self) -> Option<u32> {
        self.inner.window_resident_blocks().map(NonZeroU32::get)
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

/// Immutable grouped-cache geometry owned by one Engine.
///
/// ``parent_capacity`` is the number of fungible physical parents shared by
/// every group. Group order is stable and identifies each group in BatchPlan.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheLayout {
    inner: KvLayout,
}

impl CacheLayout {
    pub(crate) fn from_groups(
        parent_capacity: u32,
        groups: Vec<CacheGroupLayout>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KvLayout::new(
                parent_capacity,
                groups.into_iter().map(|group| group.inner).collect(),
            )
            .map_err(layout_error)?,
        })
    }

    pub(crate) const fn inner_native(&self) -> &KvLayout {
        &self.inner
    }

    #[cfg(test)]
    pub(crate) fn groups_native(&self) -> &[KvGroupLayout] {
        self.inner.groups()
    }

    fn repr_string(&self) -> String {
        let groups = self
            .inner
            .groups()
            .iter()
            .copied()
            .map(group_repr)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "CacheLayout(parent_capacity={}, groups=[{groups}])",
            self.inner.parent_capacity()
        )
    }
}

#[pymethods]
impl CacheLayout {
    #[new]
    fn new(
        py: Python<'_>,
        parent_capacity: u32,
        groups: Vec<Py<CacheGroupLayout>>,
    ) -> PyResult<Self> {
        let groups = groups.into_iter().map(|group| *group.borrow(py)).collect();
        Self::from_groups(parent_capacity, groups)
    }

    #[getter]
    /// Number of fungible physical parent allocations in the pool.
    fn parent_capacity(&self) -> u32 {
        self.inner.parent_capacity().get()
    }

    #[getter]
    /// Number of logical cache groups.
    fn group_count(&self) -> usize {
        self.inner.groups().len()
    }

    #[getter]
    /// Ordered immutable cache-group descriptors.
    fn groups(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let groups = self
            .inner
            .groups()
            .iter()
            .copied()
            .map(CacheGroupLayout::from_inner)
            .map(|group| Py::new(py, group))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, groups)?.unbind())
    }

    fn group_kernel_page_capacity(&self, group_index: u32) -> PyResult<u32> {
        self.inner.kernel_page_capacity(group_index).ok_or_else(|| {
            PyValueError::new_err("group_index is outside the cache layout group range")
        })
    }

    fn group_kernel_page_count(&self, group_index: u32) -> PyResult<u64> {
        self.inner.kernel_page_count(group_index).ok_or_else(|| {
            PyValueError::new_err("group_index is outside the cache layout group range")
        })
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

/// Physical page placement for one cache group in a BatchPlan.
///
/// ``block_indptr`` and ``block_ids`` form a CSR block table with one row per
/// sequence. ``first_logical_blocks`` gives the logical number represented by
/// each row's first physical block. The two ``append_*`` arrays contain one
/// write target per flattened batch token.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheGroupPlan {
    inner: KvGroupPlan,
}

impl CacheGroupPlan {
    pub(crate) fn from_inner(inner: KvGroupPlan) -> Self {
        Self { inner }
    }

    pub(crate) fn from_parts(
        group_index: u32,
        block_indptr: Vec<u32>,
        block_ids: Vec<u32>,
        first_logical_blocks: Vec<u32>,
        append_block_ids: Vec<u32>,
        append_block_offsets: Vec<u32>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KvGroupPlan::new(
                group_index,
                block_indptr,
                block_ids,
                first_logical_blocks,
                append_block_ids,
                append_block_offsets,
            )
            .map_err(plan_error)?,
        })
    }

    pub(crate) fn validate_local(&self) -> PyResult<()> {
        self.inner.validate().map_err(plan_error)
    }

    pub(crate) const fn inner_native(&self) -> &KvGroupPlan {
        &self.inner
    }

    fn repr_string(&self) -> String {
        format!(
            "CacheGroupPlan(group_index={}, sequences={}, blocks={}, appends={})",
            self.inner.group_index,
            self.inner.first_logical_blocks.len(),
            self.inner.block_ids.len(),
            self.inner.append_block_ids.len()
        )
    }
}

#[pymethods]
impl CacheGroupPlan {
    #[new]
    fn new(
        group_index: u32,
        block_indptr: Vec<u32>,
        block_ids: Vec<u32>,
        first_logical_blocks: Vec<u32>,
        append_block_ids: Vec<u32>,
        append_block_offsets: Vec<u32>,
    ) -> PyResult<Self> {
        Self::from_parts(
            group_index,
            block_indptr,
            block_ids,
            first_logical_blocks,
            append_block_ids,
            append_block_offsets,
        )
    }

    #[getter]
    /// Zero-based index into the Engine's CacheLayout groups.
    fn group_index(&self) -> u32 {
        self.inner.group_index
    }

    #[getter]
    /// CSR row offsets into ``block_ids``; length is sequence count plus one.
    fn block_indptr(&self) -> Vec<u32> {
        self.inner.block_indptr.clone()
    }

    #[getter]
    /// Flattened physical block IDs, grouped chronologically by sequence.
    fn block_ids(&self) -> Vec<u32> {
        self.inner.block_ids.clone()
    }

    #[getter]
    /// Logical block number corresponding to each sequence row's first block.
    fn first_logical_blocks(&self) -> Vec<u32> {
        self.inner.first_logical_blocks.clone()
    }

    #[getter]
    /// Physical destination block for every flattened batch token.
    fn append_block_ids(&self) -> Vec<u32> {
        self.inner.append_block_ids.clone()
    }

    #[getter]
    /// In-block destination offset for every flattened batch token.
    fn append_block_offsets(&self) -> Vec<u32> {
        self.inner.append_block_offsets.clone()
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(block_tokens: u32, packing: u32) -> CacheGroupLayout {
        CacheGroupLayout::from_parts(block_tokens, packing, CacheRetention::History, None).unwrap()
    }

    #[test]
    fn python_layout_adapter_preserves_frozen_groups() {
        Python::initialize();
        Python::attach(|py| {
            let group = Py::new(py, history(16, 2)).unwrap();
            let layout = CacheLayout::new(py, 4, vec![group]).unwrap();
            assert_eq!(layout.groups_native(), &[history(16, 2).inner()]);
            assert_eq!(layout.groups(py).unwrap().bind(py).len(), 1);
            assert_eq!(layout.group_kernel_page_capacity(0).unwrap(), 8);
        });
    }

    #[test]
    fn representations_keep_the_public_python_vocabulary() {
        let history = history(128, 1);
        let window =
            CacheGroupLayout::from_parts(128, 1, CacheRetention::Window, Some(128)).unwrap();
        let layout = CacheLayout::from_groups(8, vec![history, window]).unwrap();
        assert_eq!(
            layout.repr_string(),
            "CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention=CacheRetention.HISTORY), CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention=CacheRetention.WINDOW, window_tokens=128)])"
        );
    }
}
