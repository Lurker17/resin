use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use resin_core::resources::slots::{
    AccessClaim, AccessLease, PreparedAccess, SlotHandle, SlotPool as CoreSlotPool, SlotPoolError,
    SlotPoolErrorKind, mint_mapping_arena_id,
};

pub(crate) use resin_core::resources::slots::AccessMode as StateAccessMode;
pub(crate) type StateAccessClaim = AccessClaim;
pub(crate) type PreparedStateAccessLease = PreparedAccess;
pub(crate) type StateAccessLease = AccessLease;

fn state_pool_error(error: SlotPoolError, pending_message: Option<&'static str>) -> PyErr {
    let message = if error.kind() == SlotPoolErrorKind::PendingRetirement {
        pending_message.map_or_else(|| error.to_string(), str::to_owned)
    } else {
        error.to_string()
    };
    match error.kind() {
        SlotPoolErrorKind::Memory => PyMemoryError::new_err(message),
        SlotPoolErrorKind::Runtime => PyRuntimeError::new_err(message),
        SlotPoolErrorKind::Value | SlotPoolErrorKind::PendingRetirement => {
            PyValueError::new_err(message)
        }
    }
}

pub(crate) fn mint_state_mapping_arena_id() -> PyResult<u64> {
    mint_mapping_arena_id().map_err(|error| state_pool_error(error, None))
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct StateHandle {
    inner: SlotHandle,
}

impl StateHandle {
    fn from_inner(inner: SlotHandle) -> Self {
        Self { inner }
    }

    pub(crate) const fn inner(&self) -> SlotHandle {
        self.inner
    }

    pub(crate) const fn slot_native(&self) -> u32 {
        self.inner.slot()
    }

    pub(crate) const fn generation_native(&self) -> u64 {
        self.inner.generation()
    }
}

#[pymethods]
impl StateHandle {
    #[getter]
    fn slot(&self) -> u32 {
        self.slot_native()
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.generation_native()
    }

    fn __repr__(&self) -> String {
        format!(
            "StateHandle(slot={}, generation={})",
            self.slot_native(),
            self.generation_native()
        )
    }
}

#[pyclass(module = "resin._resin")]
pub(crate) struct StatePool {
    inner: CoreSlotPool,
}

#[pymethods]
impl StatePool {
    #[new]
    fn new(capacity: usize) -> PyResult<Self> {
        Ok(Self {
            inner: CoreSlotPool::new(capacity).map_err(|error| state_pool_error(error, None))?,
        })
    }

    #[pyo3(signature = (count = 1))]
    fn allocate(&mut self, py: Python<'_>, count: usize) -> PyResult<Vec<Py<StateHandle>>> {
        let core_handles = self
            .inner
            .allocate(count)
            .map_err(|error| state_pool_error(error, None))?;
        let mut handles = Vec::new();
        if handles.try_reserve_exact(core_handles.len()).is_err() {
            self.inner
                .release_external(core_handles.iter().copied())
                .expect("newly allocated handles have one external reference");
            return Err(PyMemoryError::new_err(
                "could not allocate Python state-handle metadata",
            ));
        }
        for handle in core_handles.iter().copied() {
            match Py::new(py, StateHandle::from_inner(handle)) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    self.inner
                        .release_external(core_handles.iter().copied())
                        .expect("newly allocated handles have one external reference");
                    return Err(error);
                }
            }
        }
        Ok(handles)
    }

    fn retain(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.inner
            .retain_external(handles.iter().map(|handle| handle.borrow(py).inner()))
            .map_err(|error| {
                state_pool_error(error, Some("cannot retain state pending retirement"))
            })
    }

    fn release(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.inner
            .release_external(handles.iter().map(|handle| handle.borrow(py).inner()))
            .map_err(|error| state_pool_error(error, None))
    }

    fn pin(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.inner
            .pin_external(handles.iter().map(|handle| handle.borrow(py).inner()))
            .map_err(|error| state_pool_error(error, Some("cannot pin state pending retirement")))
    }

    fn unpin(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.inner
            .unpin_external(handles.iter().map(|handle| handle.borrow(py).inner()))
            .map_err(|error| state_pool_error(error, None))
    }

    fn contains(&self, handle: PyRef<'_, StateHandle>) -> bool {
        self.inner.contains(handle.inner())
    }

    fn counts(&self, handle: PyRef<'_, StateHandle>) -> PyResult<(u32, u32)> {
        self.inner
            .counts(handle.inner())
            .map_err(|error| state_pool_error(error, None))
    }

    fn access_counts(&self, handle: PyRef<'_, StateHandle>) -> PyResult<(u32, bool)> {
        self.inner
            .access_counts(handle.inner())
            .map_err(|error| state_pool_error(error, None))
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[getter]
    fn pool_id(&self) -> u64 {
        self.inner.id()
    }

    #[getter]
    fn physical_mapping(&self) -> Option<(String, u64)> {
        self.inner
            .physical_mapping()
            .map(|(protocol, arena_id)| (protocol.to_owned(), arena_id))
    }

    #[getter]
    fn free_count(&self) -> usize {
        self.inner.free_count()
    }

    #[getter]
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    #[getter]
    fn retiring_count(&self) -> usize {
        self.inner.retiring_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "StatePool(capacity={}, active={}, retiring={})",
            self.inner.capacity(),
            self.inner.active_count(),
            self.inner.retiring_count()
        )
    }
}

impl StatePool {
    pub(crate) fn id(&self) -> u64 {
        self.inner.id()
    }

    pub(crate) fn slot_capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub(crate) fn physical_mapping_descriptor(&self) -> Option<(&str, u64)> {
        self.inner.physical_mapping()
    }

    pub(crate) fn bind_physical_mapping(&mut self, protocol: &str, arena_id: u64) -> PyResult<()> {
        self.inner
            .bind_physical_mapping(protocol, arena_id)
            .map_err(|error| state_pool_error(error, None))
    }

    pub(crate) fn retain_handles(&mut self, handles: &[SlotHandle]) -> PyResult<()> {
        self.inner
            .retain_engine(handles.iter().copied())
            .map_err(|error| {
                state_pool_error(error, Some("cannot retain state pending retirement"))
            })
    }

    pub(crate) fn release_handles(&mut self, handles: &[SlotHandle]) -> PyResult<()> {
        self.inner
            .release_engine(handles.iter().copied())
            .map_err(|error| state_pool_error(error, None))
    }

    pub(crate) fn pin_handles(&mut self, handles: &[SlotHandle]) -> PyResult<()> {
        self.inner
            .pin_engine(handles.iter().copied())
            .map_err(|error| state_pool_error(error, Some("cannot pin state pending retirement")))
    }

    pub(crate) fn unpin_handles(&mut self, handles: &[SlotHandle]) -> PyResult<()> {
        self.inner
            .unpin_engine(handles.iter().copied())
            .map_err(|error| state_pool_error(error, None))
    }

    pub(crate) fn prepare_engine_access(
        &self,
        claims: &[StateAccessClaim],
    ) -> PyResult<PreparedStateAccessLease> {
        self.inner.prepare_access(claims).map_err(|error| {
            state_pool_error(error, Some("cannot access state pending retirement"))
        })
    }

    pub(crate) fn acquire_engine_access(
        &mut self,
        prepared: PreparedStateAccessLease,
    ) -> PyResult<StateAccessLease> {
        self.inner.acquire_access(prepared).map_err(|error| {
            state_pool_error(error, Some("cannot access state pending retirement"))
        })
    }

    pub(crate) fn release_engine_access(&mut self, lease: StateAccessLease) {
        self.inner.release_access(lease);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_handle_adapter_preserves_generation_identity() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let first = pool.allocate(py, 1).unwrap().pop().unwrap();
            pool.release(py, vec![first.clone_ref(py)]).unwrap();
            let second = pool.allocate(py, 1).unwrap().pop().unwrap();
            assert_eq!(
                first.borrow(py).slot_native(),
                second.borrow(py).slot_native()
            );
            assert_ne!(
                first.borrow(py).generation_native(),
                second.borrow(py).generation_native()
            );
            assert!(!pool.contains(first.borrow(py)));
        });
    }
}
