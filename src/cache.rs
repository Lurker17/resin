//! Public layout and frozen-plan descriptors for Resin's grouped cache pool.
//!
//! A layout describes logical cache policy and the way equal-sized physical
//! parents are subdivided.  It deliberately does not describe tensor fields,
//! dtypes, layers, or kernels: the runner owns those physical details and
//! binds them to the native pool once.

#![allow(dead_code)] // Integrated into the extension module by the Engine cutover.

use std::num::NonZeroU32;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

const MAX_KERNEL_VALUE: u32 = i32::MAX as u32;

/// Native retention policy for one ordered cache group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CacheRetention {
    /// Keep the complete logical block history.
    History,
    /// Keep the chronological suffix needed by a token window.
    Window { tokens: NonZeroU32 },
}

impl CacheRetention {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::History => "history",
            Self::Window { .. } => "window",
        }
    }

    pub(crate) const fn window_tokens(self) -> Option<NonZeroU32> {
        match self {
            Self::Window { tokens } => Some(tokens),
            Self::History => None,
        }
    }
}

/// One ordered logical cache group.
///
/// Its position in [`CacheLayout::groups_native`] is its complete native
/// identity.  Names and process-global registries are intentionally absent.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheGroupLayout {
    block_tokens: NonZeroU32,
    blocks_per_parent: NonZeroU32,
    retention: CacheRetention,
}

impl CacheGroupLayout {
    pub(crate) fn from_parts(
        block_tokens: u32,
        blocks_per_parent: u32,
        retention: &str,
        window_tokens: Option<u32>,
    ) -> PyResult<Self> {
        let block_tokens = NonZeroU32::new(block_tokens)
            .ok_or_else(|| PyValueError::new_err("block_tokens must be positive"))?;
        if block_tokens.get() > MAX_KERNEL_VALUE {
            return Err(PyValueError::new_err(
                "block_tokens exceeds the signed int32 kernel metadata range",
            ));
        }
        let blocks_per_parent = NonZeroU32::new(blocks_per_parent)
            .ok_or_else(|| PyValueError::new_err("blocks_per_parent must be positive"))?;

        let retention = match retention {
            "history" => {
                if window_tokens.is_some() {
                    return Err(PyValueError::new_err(
                        "history retention does not accept window_tokens",
                    ));
                }
                CacheRetention::History
            }
            "window" => {
                let tokens = NonZeroU32::new(window_tokens.ok_or_else(|| {
                    PyValueError::new_err("window retention requires window_tokens")
                })?)
                .ok_or_else(|| PyValueError::new_err("window_tokens must be positive"))?;
                if tokens.get() > MAX_KERNEL_VALUE {
                    return Err(PyValueError::new_err(
                        "window_tokens exceeds the signed int32 kernel metadata range",
                    ));
                }
                CacheRetention::Window { tokens }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "retention must be 'history' or 'window'",
                ));
            }
        };

        Ok(Self {
            block_tokens,
            blocks_per_parent,
            retention,
        })
    }

    pub(crate) const fn block_tokens_native(&self) -> NonZeroU32 {
        self.block_tokens
    }

    pub(crate) const fn blocks_per_parent_native(&self) -> NonZeroU32 {
        self.blocks_per_parent
    }

    pub(crate) const fn retention_native(&self) -> CacheRetention {
        self.retention
    }

    /// Maximum request-owned suffix width once a window is saturated.
    pub(crate) const fn window_resident_blocks(&self) -> Option<NonZeroU32> {
        match self.retention {
            CacheRetention::Window { tokens } => {
                // One block contains the current write position and the rest
                // cover the preceding token window.  The expression is at
                // most `tokens`, hence nonzero and within u32 by construction.
                let blocks = 1 + (tokens.get() - 1).div_ceil(self.block_tokens.get());
                NonZeroU32::new(blocks)
            }
            CacheRetention::History => None,
        }
    }

    fn repr_string(&self) -> String {
        match self.retention {
            CacheRetention::History => format!(
                "CacheGroupLayout(block_tokens={}, blocks_per_parent={}, retention='history')",
                self.block_tokens, self.blocks_per_parent
            ),
            CacheRetention::Window { tokens } => format!(
                "CacheGroupLayout(block_tokens={}, blocks_per_parent={}, retention='window', window_tokens={tokens})",
                self.block_tokens, self.blocks_per_parent
            ),
        }
    }
}

#[pymethods]
impl CacheGroupLayout {
    #[new]
    #[pyo3(signature = (
        block_tokens,
        *,
        blocks_per_parent = 1,
        retention = "history",
        window_tokens = None
    ))]
    fn new(
        block_tokens: u32,
        blocks_per_parent: u32,
        retention: &str,
        window_tokens: Option<u32>,
    ) -> PyResult<Self> {
        Self::from_parts(block_tokens, blocks_per_parent, retention, window_tokens)
    }

    #[getter]
    fn block_tokens(&self) -> u32 {
        self.block_tokens.get()
    }

    #[getter]
    fn blocks_per_parent(&self) -> u32 {
        self.blocks_per_parent.get()
    }

    #[getter]
    fn retention(&self) -> &'static str {
        self.retention.as_str()
    }

    #[getter]
    fn window_tokens(&self) -> Option<u32> {
        self.retention.window_tokens().map(NonZeroU32::get)
    }

    #[getter]
    fn window_resident_block_count(&self) -> Option<u32> {
        self.window_resident_blocks().map(NonZeroU32::get)
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

/// Frozen runner-owned declaration of one physical cache pool.
///
/// Kernel page IDs are positive. Page zero is a runner-provided null row used
/// for safe table padding, so a group with `N * K` usable IDs needs
/// `1 + N * K` physical slab rows.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheLayout {
    parent_capacity: NonZeroU32,
    groups: Box<[CacheGroupLayout]>,
}

impl CacheLayout {
    pub(crate) fn from_groups(
        parent_capacity: u32,
        groups: Vec<CacheGroupLayout>,
    ) -> PyResult<Self> {
        let parent_capacity = NonZeroU32::new(parent_capacity)
            .ok_or_else(|| PyValueError::new_err("parent_capacity must be positive"))?;
        if groups.is_empty() {
            return Err(PyValueError::new_err(
                "groups must contain at least one cache group",
            ));
        }
        if groups.len() > u32::MAX as usize {
            return Err(PyValueError::new_err(
                "group count exceeds the u32 group-index range",
            ));
        }

        for (group_index, group) in groups.iter().enumerate() {
            let kernel_page_capacity =
                u64::from(parent_capacity.get()) * u64::from(group.blocks_per_parent.get());
            if kernel_page_capacity > u64::from(MAX_KERNEL_VALUE) {
                return Err(PyValueError::new_err(format!(
                    "cache group {group_index} page capacity exceeds the signed int32 kernel page-id range"
                )));
            }
            let kernel_page_count = kernel_page_capacity.checked_add(1).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "cache group {group_index} physical slab row count overflows u64"
                ))
            })?;
            usize::try_from(kernel_page_count).map_err(|_| {
                PyValueError::new_err(format!(
                    "cache group {group_index} physical slab row count exceeds usize"
                ))
            })?;
            let maximum_resident_blocks = group.window_resident_blocks();
            if maximum_resident_blocks
                .is_some_and(|blocks| u64::from(blocks.get()) > kernel_page_capacity)
            {
                return Err(PyValueError::new_err(format!(
                    "cache group {group_index} retention requires more blocks than its physical page capacity"
                )));
            }
        }

        Ok(Self {
            parent_capacity,
            groups: groups.into_boxed_slice(),
        })
    }

    pub(crate) const fn parent_capacity_native(&self) -> NonZeroU32 {
        self.parent_capacity
    }

    pub(crate) fn groups_native(&self) -> &[CacheGroupLayout] {
        &self.groups
    }

    pub(crate) fn group(&self, index: u32) -> Option<&CacheGroupLayout> {
        self.groups.get(index as usize)
    }

    pub(crate) fn packings(&self) -> Vec<NonZeroU32> {
        self.groups
            .iter()
            .map(CacheGroupLayout::blocks_per_parent_native)
            .collect()
    }

    pub(crate) fn kernel_page_capacity(&self, group_index: u32) -> Option<u32> {
        let packing = self.group(group_index)?.blocks_per_parent_native().get();
        Some(
            self.parent_capacity
                .get()
                .checked_mul(packing)
                .expect("layout validation bounds group kernel page capacity"),
        )
    }

    /// Physical rows required by a group view, including the null page zero.
    pub(crate) fn kernel_page_count(&self, group_index: u32) -> Option<u64> {
        self.kernel_page_capacity(group_index)
            .map(|capacity| u64::from(capacity) + 1)
    }

    fn repr_string(&self) -> String {
        let groups = self
            .groups
            .iter()
            .map(CacheGroupLayout::repr_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "CacheLayout(parent_capacity={}, groups=[{groups}])",
            self.parent_capacity
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
        let groups = groups
            .into_iter()
            .map(|group| group.borrow(py).clone())
            .collect();
        Self::from_groups(parent_capacity, groups)
    }

    #[getter]
    fn parent_capacity(&self) -> u32 {
        self.parent_capacity.get()
    }

    #[getter]
    fn group_count(&self) -> usize {
        self.groups.len()
    }

    #[getter]
    fn groups(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let groups = self
            .groups
            .iter()
            .cloned()
            .map(|group| Py::new(py, group))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, groups)?.unbind())
    }

    /// Number of positive, allocatable kernel page IDs. Page zero is omitted.
    fn group_kernel_page_capacity(&self, group_index: u32) -> PyResult<u32> {
        self.kernel_page_capacity(group_index).ok_or_else(|| {
            PyValueError::new_err("group_index is outside the cache layout group range")
        })
    }

    /// Number of physical slab rows, including the null page at row zero.
    fn group_kernel_page_count(&self, group_index: u32) -> PyResult<u64> {
        self.kernel_page_count(group_index).ok_or_else(|| {
            PyValueError::new_err("group_index is outside the cache layout group range")
        })
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

/// Generic frozen block-table metadata for one ordered cache group.
///
/// Rows are CSR-encoded in batch sequence order.  A row is a chronological,
/// contiguous logical-block map covering every block needed during this
/// execution; `first_logical_blocks[row]` identifies its first logical block.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheGroupPlan {
    group_index: u32,
    block_indptr: Vec<u32>,
    block_ids: Vec<u32>,
    first_logical_blocks: Vec<u32>,
    append_block_ids: Vec<u32>,
    append_block_offsets: Vec<u32>,
}

impl CacheGroupPlan {
    pub(crate) fn from_parts(
        group_index: u32,
        block_indptr: Vec<u32>,
        block_ids: Vec<u32>,
        first_logical_blocks: Vec<u32>,
        append_block_ids: Vec<u32>,
        append_block_offsets: Vec<u32>,
    ) -> PyResult<Self> {
        let plan = Self {
            group_index,
            block_indptr,
            block_ids,
            first_logical_blocks,
            append_block_ids,
            append_block_offsets,
        };
        plan.validate_local()?;
        Ok(plan)
    }

    /// Validate invariants which do not require the enclosing layout or batch.
    pub(crate) fn validate_local(&self) -> PyResult<()> {
        if self.block_ids.len() > u32::MAX as usize {
            return Err(PyValueError::new_err(
                "block_ids length exceeds the u32 CSR range",
            ));
        }
        if self.block_indptr.len() != self.first_logical_blocks.len() + 1 {
            return Err(PyValueError::new_err(
                "block_indptr must contain one more item than first_logical_blocks",
            ));
        }
        if self.block_indptr.first() != Some(&0) {
            return Err(PyValueError::new_err("block_indptr must start at zero"));
        }
        if self.block_indptr.last().copied() != Some(self.block_ids.len() as u32) {
            return Err(PyValueError::new_err(
                "block_indptr must end at the block_ids length",
            ));
        }
        if self.block_indptr.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(PyValueError::new_err(
                "block_indptr must be monotonically nondecreasing",
            ));
        }
        if self.block_ids.contains(&0) {
            return Err(PyValueError::new_err(
                "block_ids must contain only positive kernel page IDs",
            ));
        }
        if self.append_block_ids.len() != self.append_block_offsets.len() {
            return Err(PyValueError::new_err(
                "append_block_ids and append_block_offsets must have equal length",
            ));
        }
        if self.append_block_ids.contains(&0) {
            return Err(PyValueError::new_err(
                "append_block_ids must contain only positive kernel page IDs",
            ));
        }
        Ok(())
    }

    pub(crate) const fn group_index_native(&self) -> u32 {
        self.group_index
    }

    pub(crate) fn block_indptr_native(&self) -> &[u32] {
        &self.block_indptr
    }

    pub(crate) fn block_ids_native(&self) -> &[u32] {
        &self.block_ids
    }

    pub(crate) fn first_logical_blocks_native(&self) -> &[u32] {
        &self.first_logical_blocks
    }

    pub(crate) fn append_block_ids_native(&self) -> &[u32] {
        &self.append_block_ids
    }

    pub(crate) fn append_block_offsets_native(&self) -> &[u32] {
        &self.append_block_offsets
    }

    fn repr_string(&self) -> String {
        format!(
            "CacheGroupPlan(group_index={}, sequences={}, blocks={}, appends={})",
            self.group_index,
            self.first_logical_blocks.len(),
            self.block_ids.len(),
            self.append_block_ids.len()
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
    fn group_index(&self) -> u32 {
        self.group_index
    }

    #[getter]
    fn block_indptr(&self) -> Vec<u32> {
        self.block_indptr.clone()
    }

    #[getter]
    fn block_ids(&self) -> Vec<u32> {
        self.block_ids.clone()
    }

    #[getter]
    fn first_logical_blocks(&self) -> Vec<u32> {
        self.first_logical_blocks.clone()
    }

    #[getter]
    fn append_block_ids(&self) -> Vec<u32> {
        self.append_block_ids.clone()
    }

    #[getter]
    fn append_block_offsets(&self) -> Vec<u32> {
        self.append_block_offsets.clone()
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(block_tokens: u32, packing: u32) -> CacheGroupLayout {
        CacheGroupLayout::from_parts(block_tokens, packing, "history", None).unwrap()
    }

    #[test]
    fn history_layout_freezes_order_and_kernel_geometry() {
        let layout = CacheLayout::from_groups(8, vec![history(128, 1), history(32, 4)]).unwrap();

        assert_eq!(layout.parent_capacity_native().get(), 8);
        assert_eq!(layout.groups_native().len(), 2);
        assert_eq!(layout.group(0).unwrap().block_tokens_native().get(), 128);
        assert_eq!(layout.group(1).unwrap().block_tokens_native().get(), 32);
        assert!(layout.group(2).is_none());
        assert_eq!(
            layout.packings(),
            [NonZeroU32::new(1).unwrap(), NonZeroU32::new(4).unwrap()]
        );
        assert_eq!(layout.kernel_page_capacity(0), Some(8));
        assert_eq!(layout.kernel_page_capacity(1), Some(32));
        assert_eq!(layout.kernel_page_count(0), Some(9));
        assert_eq!(layout.kernel_page_count(1), Some(33));
    }

    #[test]
    fn retention_descriptors_are_structural_and_unambiguous() {
        let window = CacheGroupLayout::from_parts(128, 1, "window", Some(128)).unwrap();
        assert_eq!(
            window.retention_native(),
            CacheRetention::Window {
                tokens: NonZeroU32::new(128).unwrap()
            }
        );
        assert_eq!(window.window_resident_blocks().unwrap().get(), 2);

        let generalized = CacheGroupLayout::from_parts(64, 2, "window", Some(193)).unwrap();
        assert_eq!(generalized.window_resident_blocks().unwrap().get(), 4);
    }

    #[test]
    fn group_validation_rejects_ambiguous_or_out_of_range_policy() {
        assert!(CacheGroupLayout::from_parts(0, 1, "history", None).is_err());
        assert!(CacheGroupLayout::from_parts(MAX_KERNEL_VALUE + 1, 1, "history", None).is_err());
        assert!(CacheGroupLayout::from_parts(1, 0, "history", None).is_err());
        assert!(CacheGroupLayout::from_parts(1, 1, "other", None).is_err());
        assert!(CacheGroupLayout::from_parts(1, 1, "history", Some(1)).is_err());
        assert!(CacheGroupLayout::from_parts(1, 1, "window", None).is_err());
        assert!(CacheGroupLayout::from_parts(1, 1, "window", Some(0)).is_err());
        assert!(CacheGroupLayout::from_parts(1, 1, "state", None).is_err());
    }

    #[test]
    fn layout_validation_bounds_physical_geometry() {
        assert!(CacheLayout::from_groups(0, vec![history(4, 1)]).is_err());
        assert!(CacheLayout::from_groups(1, Vec::new()).is_err());
        assert!(CacheLayout::from_groups(MAX_KERNEL_VALUE, vec![history(1, 2)]).is_err());

        let oversized_window = CacheGroupLayout::from_parts(1, 1, "window", Some(3)).unwrap();
        assert!(CacheLayout::from_groups(2, vec![oversized_window]).is_err());
    }

    #[test]
    fn descriptors_have_complete_stable_representations() {
        let history = history(128, 1);
        let window = CacheGroupLayout::from_parts(128, 1, "window", Some(128)).unwrap();
        assert_eq!(
            history.repr_string(),
            "CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention='history')"
        );
        assert_eq!(
            window.repr_string(),
            "CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention='window', window_tokens=128)"
        );
        let layout = CacheLayout::from_groups(8, vec![history, window]).unwrap();
        assert_eq!(
            layout.repr_string(),
            "CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention='history'), CacheGroupLayout(block_tokens=128, blocks_per_parent=1, retention='window', window_tokens=128)])"
        );
    }

    #[test]
    fn python_layout_constructor_copies_frozen_group_values() {
        Python::initialize();
        Python::attach(|py| {
            let group = Py::new(py, history(16, 2)).unwrap();
            let layout = CacheLayout::new(py, 4, vec![group]).unwrap();
            assert_eq!(layout.groups_native(), &[history(16, 2)]);
            let groups = layout.groups(py).unwrap();
            assert_eq!(groups.bind(py).len(), 1);
            assert_eq!(layout.group_kernel_page_capacity(0).unwrap(), 8);
            assert_eq!(layout.group_kernel_page_count(0).unwrap(), 9);
            assert!(layout.group_kernel_page_count(1).is_err());
        });
    }

    #[test]
    fn maximum_positive_page_id_still_has_a_distinct_null_slab_row() {
        let layout = CacheLayout::from_groups(MAX_KERNEL_VALUE, vec![history(1, 1)]).unwrap();
        assert_eq!(layout.kernel_page_capacity(0), Some(MAX_KERNEL_VALUE));
        assert_eq!(
            layout.kernel_page_count(0),
            Some(u64::from(MAX_KERNEL_VALUE) + 1)
        );
    }

    fn valid_plan() -> CacheGroupPlan {
        CacheGroupPlan::from_parts(
            1,
            vec![0, 2, 3],
            vec![7, 8, 11],
            vec![0, 5],
            vec![8, 11],
            vec![3, 0],
        )
        .unwrap()
    }

    #[test]
    fn group_plan_is_generic_frozen_csr_and_append_metadata() {
        let plan = valid_plan();
        assert_eq!(plan.group_index_native(), 1);
        assert_eq!(plan.block_indptr_native(), [0, 2, 3]);
        assert_eq!(plan.block_ids_native(), [7, 8, 11]);
        assert_eq!(plan.first_logical_blocks_native(), [0, 5]);
        assert_eq!(plan.append_block_ids_native(), [8, 11]);
        assert_eq!(plan.append_block_offsets_native(), [3, 0]);
        assert_eq!(
            plan.repr_string(),
            "CacheGroupPlan(group_index=1, sequences=2, blocks=3, appends=2)"
        );
        plan.validate_local().unwrap();
    }

    #[test]
    fn group_plan_local_validation_rejects_malformed_shapes_and_null_ids() {
        let make = |indptr, ids, first, append_ids, append_offsets| {
            CacheGroupPlan::from_parts(0, indptr, ids, first, append_ids, append_offsets)
        };
        assert!(make(vec![], vec![], vec![], vec![], vec![]).is_err());
        assert!(make(vec![0], vec![], vec![0], vec![], vec![]).is_err());
        assert!(make(vec![1, 1], vec![2], vec![0], vec![], vec![]).is_err());
        assert!(make(vec![0, 2], vec![2], vec![0], vec![], vec![]).is_err());
        assert!(make(vec![0, 2, 1], vec![2], vec![0, 1], vec![], vec![]).is_err());
        assert!(make(vec![0, 1], vec![0], vec![0], vec![], vec![]).is_err());
        assert!(make(vec![0, 1], vec![2], vec![0], vec![2], vec![]).is_err());
        assert!(make(vec![0, 1], vec![2], vec![0], vec![0], vec![0]).is_err());
    }
}
