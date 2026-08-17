//! Python adapters for the pure Rust cache descriptors in `resin-core`.

use std::num::NonZeroU32;
use std::sync::Arc;

use numpy::{PyArray1, PyArrayLike1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use resin_core::resources::kv::{
    KvGroupFamily, KvGroupLayout, KvGroupPlan, KvLayout, KvLayoutError, KvPlanError, KvRetention,
};

fn layout_error(error: KvLayoutError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn plan_error(error: KvPlanError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Cache-block lifetime policy.
///
/// For history groups, ``HISTORY`` retains every logical block and ``WINDOW``
/// retains the suffix selected by ``CacheGroupLayout.window_tokens``. Recurrent
/// state groups use ``HISTORY`` and retain their two-page checkpoint ring.
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

/// Whether a group stores token history or recurrent checkpoints.
#[pyclass(
    frozen,
    eq,
    hash,
    rename_all = "SCREAMING_SNAKE_CASE",
    from_py_object,
    module = "resin._resin"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CacheGroupFamily {
    History,
    State,
}

fn group_repr(group: KvGroupLayout) -> String {
    let family = match group.family() {
        KvGroupFamily::History => "",
        KvGroupFamily::State => ", family=CacheGroupFamily.STATE",
    };
    match group.retention() {
        KvRetention::FullContext => format!(
            "CacheGroupLayout(block_tokens={}, blocks_per_parent={}, retention=CacheRetention.HISTORY{family})",
            group.block_tokens(),
            group.blocks_per_parent()
        ),
        KvRetention::SlidingWindow { tokens } => format!(
            "CacheGroupLayout(block_tokens={}, blocks_per_parent={}, retention=CacheRetention.WINDOW, window_tokens={tokens}{family})",
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
        family: CacheGroupFamily,
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
        let family = match family {
            CacheGroupFamily::History => KvGroupFamily::History,
            CacheGroupFamily::State => KvGroupFamily::State,
        };
        Ok(Self {
            inner: KvGroupLayout::new_with_family(
                block_tokens,
                blocks_per_parent,
                retention,
                family,
            )
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
        window_tokens = None,
        family = CacheGroupFamily::History
    ))]
    fn new(
        block_tokens: u32,
        blocks_per_parent: u32,
        retention: CacheRetention,
        window_tokens: Option<u32>,
        family: CacheGroupFamily,
    ) -> PyResult<Self> {
        Self::from_parts(
            block_tokens,
            blocks_per_parent,
            retention,
            window_tokens,
            family,
        )
    }

    /// Number of token positions represented by one logical block.
    #[getter]
    fn block_tokens(&self) -> u32 {
        self.inner.block_tokens().get()
    }

    /// Number of this group's logical blocks packed into one parent allocation.
    #[getter]
    fn blocks_per_parent(&self) -> u32 {
        self.inner.blocks_per_parent().get()
    }

    /// Lifetime policy for logical blocks in this group.
    #[getter]
    fn retention(&self) -> CacheRetention {
        match self.inner.retention() {
            KvRetention::FullContext => CacheRetention::History,
            KvRetention::SlidingWindow { .. } => CacheRetention::Window,
        }
    }

    /// Whether this group stores token history or recurrent checkpoints.
    #[getter]
    fn family(&self) -> CacheGroupFamily {
        match self.inner.family() {
            KvGroupFamily::History => CacheGroupFamily::History,
            KvGroupFamily::State => CacheGroupFamily::State,
        }
    }

    /// Retained suffix length for ``WINDOW`` groups, otherwise ``None``.
    #[getter]
    fn window_tokens(&self) -> Option<u32> {
        self.inner.retention().window_tokens().map(NonZeroU32::get)
    }

    /// Maximum resident logical blocks per sequence for a window group.
    #[getter]
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

    /// Number of fungible physical parent allocations in the pool.
    #[getter]
    fn parent_capacity(&self) -> u32 {
        self.inner.parent_capacity().get()
    }

    /// Number of logical cache groups.
    #[getter]
    fn group_count(&self) -> usize {
        self.inner.groups().len()
    }

    /// Ordered immutable cache-group descriptors.
    #[getter]
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

/// Plan for one type of cache in a BatchPlan.
///
/// A model can have one or more types of cache, such as full attention,
/// sliding-window attention, or recurrent state, each described by one
/// ``CacheGroupLayout``. Cache memory is organized into fungible slabs. Each
/// slab is assigned to one cache group and holds ``blocks_per_parent`` blocks.
///
/// A block is indexed by ``(slab_id, slotnum)``. The parallel
/// ``block_parent_ids`` and ``block_child_slots`` arrays contain these values:
/// a parent ID is a one-based slab ID, and a child slot is a zero-based slot
/// number. A cached token location is ``(block, block_offset)``, equivalently
/// ``(slab_id, slotnum, block_offset)``.
///
/// The plan covers every sequence in the batch. Adjacent ``block_indptr``
/// values select each sequence's chronological block-table row.
/// ``first_logical_blocks[i]`` is sequence ``i``'s block-table start index, so
/// row entry ``j`` represents block ``first_logical_blocks[i] + j``; block
/// ``b`` begins at token position ``b * CacheGroupLayout.block_tokens``. Full-
/// attention tables start at zero; sliding-window and recurrent tables advance
/// as old blocks leave their resident rings.
///
/// For flattened batch token ``k``, the ``append_*`` arrays supply exactly that
/// cached-token location: ``(append_parent_ids[k], append_child_slots[k],
/// append_block_offsets[k])``.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheGroupPlan {
    inner: Arc<KvGroupPlan>,
}

impl CacheGroupPlan {
    pub(crate) fn from_inner(inner: KvGroupPlan) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub(crate) fn from_parts(
        group_index: u32,
        block_indptr: Vec<u32>,
        block_parent_ids: Vec<u32>,
        block_child_slots: Vec<u32>,
        first_logical_blocks: Vec<u32>,
        append_parent_ids: Vec<u32>,
        append_child_slots: Vec<u32>,
        append_block_offsets: Vec<u32>,
        blocks_per_parent: u32,
    ) -> PyResult<Self> {
        let blocks_per_parent = NonZeroU32::new(blocks_per_parent)
            .ok_or_else(|| PyValueError::new_err("blocks_per_parent must be positive"))?;
        Ok(Self {
            inner: Arc::new(
                KvGroupPlan::new(
                    group_index,
                    block_indptr,
                    block_parent_ids,
                    block_child_slots,
                    first_logical_blocks,
                    append_parent_ids,
                    append_child_slots,
                    append_block_offsets,
                    blocks_per_parent,
                )
                .map_err(plan_error)?,
            ),
        })
    }

    pub(crate) fn validate_local(&self) -> PyResult<()> {
        self.inner.validate().map_err(plan_error)
    }

    pub(crate) fn inner_native(&self) -> &KvGroupPlan {
        self.inner.as_ref()
    }

    fn repr_string(&self) -> String {
        format!(
            "CacheGroupPlan(group_index={}, blocks_per_parent={}, sequences={}, blocks={}, appends={})",
            self.inner.group_index,
            self.inner.blocks_per_parent,
            self.inner.first_logical_blocks.len(),
            self.inner.block_parent_ids.len(),
            self.inner.append_parent_ids.len()
        )
    }
}

#[pymethods]
impl CacheGroupPlan {
    #[new]
    #[pyo3(signature = (
        group_index,
        block_indptr,
        block_parent_ids,
        block_child_slots,
        first_logical_blocks,
        append_parent_ids,
        append_child_slots,
        append_block_offsets,
        *,
        blocks_per_parent = 1
    ))]
    fn new(
        group_index: u32,
        block_indptr: PyArrayLike1<'_, u32>,
        block_parent_ids: PyArrayLike1<'_, u32>,
        block_child_slots: PyArrayLike1<'_, u32>,
        first_logical_blocks: PyArrayLike1<'_, u32>,
        append_parent_ids: PyArrayLike1<'_, u32>,
        append_child_slots: PyArrayLike1<'_, u32>,
        append_block_offsets: PyArrayLike1<'_, u32>,
        blocks_per_parent: u32,
    ) -> PyResult<Self> {
        Self::from_parts(
            group_index,
            crate::arrays::owned_vec(block_indptr),
            crate::arrays::owned_vec(block_parent_ids),
            crate::arrays::owned_vec(block_child_slots),
            crate::arrays::owned_vec(first_logical_blocks),
            crate::arrays::owned_vec(append_parent_ids),
            crate::arrays::owned_vec(append_child_slots),
            crate::arrays::owned_vec(append_block_offsets),
            blocks_per_parent,
        )
    }

    /// Zero-based index of this plan's matching CacheGroupLayout. BatchPlan
    /// orders group plans by this index and rejects gaps or reordering.
    #[getter]
    fn group_index(&self) -> u32 {
        self.inner.group_index
    }

    /// Number of this group's physical cache blocks packed into one parent.
    /// Valid child slots are ``0 <= slot < blocks_per_parent``.
    #[getter]
    fn blocks_per_parent(&self) -> u32 {
        self.inner.blocks_per_parent
    }

    /// CSR row pointer into the block-address arrays. Its length is one more
    /// than the batch sequence count.
    #[getter]
    fn block_indptr<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(this.clone().into_any(), &this.get().inner.block_indptr)
    }

    /// One-based physical parent ID for each flattened block-table entry.
    /// Pair with ``block_child_slots``; parent zero is reserved as null.
    #[getter]
    fn block_parent_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(this.clone().into_any(), &this.get().inner.block_parent_ids)
    }

    /// Child-block slot for each flattened block-table entry. Pair with
    /// ``block_parent_ids``.
    #[getter]
    fn block_child_slots<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(this.clone().into_any(), &this.get().inner.block_child_slots)
    }

    /// Absolute logical-block index at which each sequence's table row starts.
    /// This is zero for full-attention history and advances for sliding-window
    /// and recurrent rings. It has one entry per sequence and is unrelated to
    /// the physical slab ID.
    #[getter]
    fn first_logical_blocks<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(
            this.clone().into_any(),
            &this.get().inner.first_logical_blocks,
        )
    }

    /// One-based physical parent ID of each token's write target. Entry ``k``
    /// corresponds to ``BatchPlan.token_ids[k]``.
    #[getter]
    fn append_parent_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(this.clone().into_any(), &this.get().inner.append_parent_ids)
    }

    /// Child-block slot of each token's write target. Entry ``k`` corresponds
    /// to ``BatchPlan.token_ids[k]``.
    #[getter]
    fn append_child_slots<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(
            this.clone().into_any(),
            &this.get().inner.append_child_slots,
        )
    }

    /// Token offset within each token's destination block. Entry ``k``
    /// corresponds to ``BatchPlan.token_ids[k]`` and is less than the matching
    /// ``CacheGroupLayout.block_tokens``.
    #[getter]
    fn append_block_offsets<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(
            this.clone().into_any(),
            &this.get().inner.append_block_offsets,
        )
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(block_tokens: u32, packing: u32) -> CacheGroupLayout {
        CacheGroupLayout::from_parts(
            block_tokens,
            packing,
            CacheRetention::History,
            None,
            CacheGroupFamily::History,
        )
        .unwrap()
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
        let window = CacheGroupLayout::from_parts(
            128,
            1,
            CacheRetention::Window,
            Some(128),
            CacheGroupFamily::History,
        )
        .unwrap();
        let layout = CacheLayout::from_groups(8, vec![history, window]).unwrap();
        assert_eq!(
            layout.repr_string(),
            "CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention=CacheRetention.HISTORY), CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention=CacheRetention.WINDOW, window_tokens=128)])"
        );
    }
}
