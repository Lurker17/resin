use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_STATE_MAPPING_ARENA_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn mint_state_mapping_arena_id() -> PyResult<u64> {
    NEXT_STATE_MAPPING_ARENA_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| PyRuntimeError::new_err("state mapping arena id space is exhausted"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StatePhysicalMapping {
    protocol: String,
    arena_id: u64,
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct StateHandle {
    pub(crate) pool_id: u64,
    pub(crate) slot: u32,
    pub(crate) generation: u64,
}

#[pymethods]
impl StateHandle {
    #[getter]
    fn slot(&self) -> u32 {
        self.slot
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.generation
    }

    fn __repr__(&self) -> String {
        format!(
            "StateHandle(slot={}, generation={})",
            self.slot, self.generation
        )
    }
}

#[derive(Clone, Debug, Default)]
struct Slot {
    generation: u64,
    external_references: u32,
    engine_references: u32,
    external_completion_pins: u32,
    engine_completion_pins: u32,
    engine_read_leases: u32,
    engine_write_lease: bool,
    allocated: bool,
}

impl Slot {
    fn total_references(&self) -> u64 {
        u64::from(self.external_references) + u64::from(self.engine_references)
    }

    fn total_completion_pins(&self) -> u64 {
        u64::from(self.external_completion_pins) + u64::from(self.engine_completion_pins)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StateAccessMode {
    Read,
    Write,
}

impl StateAccessMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StateAccessClaim {
    row_index: usize,
    handle: StateHandle,
    mode: StateAccessMode,
}

impl StateAccessClaim {
    pub(crate) fn new(row_index: usize, handle: StateHandle, mode: StateAccessMode) -> Self {
        Self {
            row_index,
            handle,
            mode,
        }
    }
}

#[derive(Debug)]
struct StateAccessEntry {
    handle: StateHandle,
    mode: StateAccessMode,
}

#[derive(Debug)]
#[must_use = "a prepared state access must be acquired or discarded"]
pub(crate) struct PreparedStateAccessLease {
    pool_id: u64,
    entries: Vec<StateAccessEntry>,
}

#[derive(Debug)]
#[must_use = "an acquired state access must be released or quarantined"]
pub(crate) struct StateAccessLease {
    pool_id: u64,
    entries: Vec<StateAccessEntry>,
}

#[pyclass(module = "resin._resin")]
pub(crate) struct StatePool {
    id: u64,
    slots: Vec<Slot>,
    free: Vec<u32>,
    // A slot generation may outlive any one Engine, so its physical meaning
    // cannot be rebound after the first recurrent-state runner claims it.
    physical_mapping: Option<StatePhysicalMapping>,
}

#[derive(Clone, Copy)]
enum CountOperation {
    RetainExternal,
    ReleaseExternal,
    RetainEngine,
    ReleaseEngine,
    PinExternal,
    UnpinExternal,
    PinEngine,
    UnpinEngine,
}

#[pymethods]
impl StatePool {
    #[new]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be positive"));
        }
        if capacity > u32::MAX as usize {
            return Err(PyValueError::new_err(
                "capacity exceeds the slot index range",
            ));
        }
        let id = NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            return Err(PyRuntimeError::new_err("state pool id space is exhausted"));
        }
        Ok(Self {
            id,
            slots: vec![Slot::default(); capacity],
            free: (0..capacity as u32).rev().collect(),
            physical_mapping: None,
        })
    }

    #[pyo3(signature = (count = 1))]
    fn allocate(&mut self, py: Python<'_>, count: usize) -> PyResult<Vec<Py<StateHandle>>> {
        if count == 0 {
            return Err(PyValueError::new_err("count must be positive"));
        }
        if count > self.free.len() {
            return Err(PyMemoryError::new_err(format!(
                "state pool has {} free slots but {count} were requested",
                self.free.len()
            )));
        }

        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            let slot_index = self.free.pop().expect("capacity checked");
            let slot = &mut self.slots[slot_index as usize];
            slot.generation = slot
                .generation
                .checked_add(1)
                .ok_or_else(|| PyRuntimeError::new_err("slot generation is exhausted"))?;
            slot.external_references = 1;
            slot.engine_references = 0;
            slot.external_completion_pins = 0;
            slot.engine_completion_pins = 0;
            slot.engine_read_leases = 0;
            slot.engine_write_lease = false;
            slot.allocated = true;
            handles.push(Py::new(
                py,
                StateHandle {
                    pool_id: self.id,
                    slot: slot_index,
                    generation: slot.generation,
                },
            )?);
        }
        Ok(handles)
    }

    fn retain(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.apply_counts(py, handles, CountOperation::RetainExternal)
    }

    fn release(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.apply_counts(py, handles, CountOperation::ReleaseExternal)
    }

    fn pin(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.apply_counts(py, handles, CountOperation::PinExternal)
    }

    fn unpin(&mut self, py: Python<'_>, handles: Vec<Py<StateHandle>>) -> PyResult<()> {
        self.apply_counts(py, handles, CountOperation::UnpinExternal)
    }

    fn contains(&self, handle: PyRef<'_, StateHandle>) -> bool {
        self.slot_for(&handle).is_ok()
    }

    fn counts(&self, handle: PyRef<'_, StateHandle>) -> PyResult<(u32, u32)> {
        let slot = self.slot_for(&handle)?;
        Ok((
            u32::try_from(slot.total_references()).expect("reference total is bounded"),
            u32::try_from(slot.total_completion_pins()).expect("completion pin total is bounded"),
        ))
    }

    fn access_counts(&self, handle: PyRef<'_, StateHandle>) -> PyResult<(u32, bool)> {
        let slot = self.slot_for(&handle)?;
        Ok((slot.engine_read_leases, slot.engine_write_lease))
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[getter]
    fn pool_id(&self) -> u64 {
        self.id
    }

    #[getter]
    fn physical_mapping(&self) -> Option<(String, u64)> {
        self.physical_mapping
            .as_ref()
            .map(|mapping| (mapping.protocol.clone(), mapping.arena_id))
    }

    #[getter]
    fn free_count(&self) -> usize {
        self.free.len()
    }

    #[getter]
    fn active_count(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    #[getter]
    fn retiring_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.allocated && slot.total_references() == 0)
            .count()
    }

    fn __repr__(&self) -> String {
        format!(
            "StatePool(capacity={}, active={}, retiring={})",
            self.capacity(),
            self.active_count(),
            self.retiring_count()
        )
    }
}

impl StatePool {
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn slot_capacity(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn physical_mapping_descriptor(&self) -> Option<(&str, u64)> {
        self.physical_mapping
            .as_ref()
            .map(|mapping| (mapping.protocol.as_str(), mapping.arena_id))
    }

    pub(crate) fn bind_physical_mapping(&mut self, protocol: &str, arena_id: u64) -> PyResult<()> {
        if protocol.is_empty() {
            return Err(PyValueError::new_err(
                "state mapping protocol must be non-empty",
            ));
        }
        if arena_id == 0 {
            return Err(PyValueError::new_err(
                "state mapping arena id must be positive",
            ));
        }
        let candidate = StatePhysicalMapping {
            protocol: protocol.to_owned(),
            arena_id,
        };
        match &self.physical_mapping {
            None => {
                self.physical_mapping = Some(candidate);
                Ok(())
            }
            Some(current) if current == &candidate => Ok(()),
            Some(current) => Err(PyValueError::new_err(format!(
                "state pool is already bound to physical mapping ({:?}, {}), cannot bind ({protocol:?}, {arena_id})",
                current.protocol, current.arena_id
            ))),
        }
    }

    fn slot_for(&self, handle: &StateHandle) -> PyResult<&Slot> {
        if handle.pool_id != self.id {
            return Err(PyValueError::new_err(
                "state handle belongs to a different pool",
            ));
        }
        let Some(slot) = self.slots.get(handle.slot as usize) else {
            return Err(PyValueError::new_err("state handle slot is out of range"));
        };
        if !slot.allocated || slot.generation != handle.generation {
            return Err(PyValueError::new_err("state handle is stale"));
        }
        Ok(slot)
    }

    fn handle_counts(
        &self,
        py: Python<'_>,
        handles: Vec<Py<StateHandle>>,
    ) -> PyResult<BTreeMap<StateHandle, u32>> {
        let mut counts = BTreeMap::new();
        for handle in handles {
            let handle = handle.borrow(py).clone();
            self.slot_for(&handle)?;
            let count = counts.entry(handle).or_insert(0_u32);
            *count = count
                .checked_add(1)
                .ok_or_else(|| PyRuntimeError::new_err("handle count overflow"))?;
        }
        Ok(counts)
    }

    fn value_counts(&self, handles: &[StateHandle]) -> PyResult<BTreeMap<StateHandle, u32>> {
        let mut counts = BTreeMap::new();
        for handle in handles {
            self.slot_for(handle)?;
            let count = counts.entry(handle.clone()).or_insert(0_u32);
            *count = count
                .checked_add(1)
                .ok_or_else(|| PyRuntimeError::new_err("handle count overflow"))?;
        }
        Ok(counts)
    }

    fn apply_counts(
        &mut self,
        py: Python<'_>,
        handles: Vec<Py<StateHandle>>,
        operation: CountOperation,
    ) -> PyResult<()> {
        let counts = self.handle_counts(py, handles)?;
        self.apply_count_map(counts, operation)
    }

    fn apply_values(&mut self, handles: &[StateHandle], operation: CountOperation) -> PyResult<()> {
        let counts = self.value_counts(handles)?;
        self.apply_count_map(counts, operation)
    }

    fn apply_count_map(
        &mut self,
        counts: BTreeMap<StateHandle, u32>,
        operation: CountOperation,
    ) -> PyResult<()> {
        // Validate the complete operation first. A bad handle cannot leave a
        // partially updated ownership graph.
        for (handle, count) in &counts {
            let slot = self.slot_for(handle)?;
            match operation {
                CountOperation::RetainExternal => {
                    if slot.total_references() == 0 {
                        return Err(PyValueError::new_err(
                            "cannot retain state pending retirement",
                        ));
                    }
                    if slot.total_references() + u64::from(*count) > u64::from(u32::MAX) {
                        return Err(PyRuntimeError::new_err("reference count overflow"));
                    }
                    slot.external_references
                        .checked_add(*count)
                        .ok_or_else(|| PyRuntimeError::new_err("reference count overflow"))?;
                }
                CountOperation::ReleaseExternal => {
                    if *count > slot.external_references {
                        return Err(PyValueError::new_err(
                            "release count exceeds caller-owned state references",
                        ));
                    }
                }
                CountOperation::RetainEngine => {
                    if slot.total_references() == 0 {
                        return Err(PyValueError::new_err(
                            "cannot retain state pending retirement",
                        ));
                    }
                    if slot.total_references() + u64::from(*count) > u64::from(u32::MAX) {
                        return Err(PyRuntimeError::new_err("reference count overflow"));
                    }
                    slot.engine_references
                        .checked_add(*count)
                        .ok_or_else(|| PyRuntimeError::new_err("reference count overflow"))?;
                }
                CountOperation::ReleaseEngine => {
                    if *count > slot.engine_references {
                        return Err(PyValueError::new_err(
                            "release count exceeds engine-owned state references",
                        ));
                    }
                }
                CountOperation::PinExternal => {
                    if slot.total_references() == 0 {
                        return Err(PyValueError::new_err("cannot pin state pending retirement"));
                    }
                    if slot.total_completion_pins() + u64::from(*count) > u64::from(u32::MAX) {
                        return Err(PyRuntimeError::new_err("completion pin overflow"));
                    }
                    slot.external_completion_pins
                        .checked_add(*count)
                        .ok_or_else(|| PyRuntimeError::new_err("completion pin overflow"))?;
                }
                CountOperation::UnpinExternal => {
                    if *count > slot.external_completion_pins {
                        return Err(PyValueError::new_err(
                            "unpin count exceeds caller-owned completion pins",
                        ));
                    }
                }
                CountOperation::PinEngine => {
                    if slot.total_references() == 0 {
                        return Err(PyValueError::new_err("cannot pin state pending retirement"));
                    }
                    if slot.total_completion_pins() + u64::from(*count) > u64::from(u32::MAX) {
                        return Err(PyRuntimeError::new_err("completion pin overflow"));
                    }
                    slot.engine_completion_pins
                        .checked_add(*count)
                        .ok_or_else(|| PyRuntimeError::new_err("completion pin overflow"))?;
                }
                CountOperation::UnpinEngine => {
                    if *count > slot.engine_completion_pins {
                        return Err(PyValueError::new_err(
                            "unpin count exceeds engine-owned completion pins",
                        ));
                    }
                }
            }
        }

        for (handle, count) in counts {
            let slot = &mut self.slots[handle.slot as usize];
            match operation {
                CountOperation::RetainExternal => slot.external_references += count,
                CountOperation::ReleaseExternal => slot.external_references -= count,
                CountOperation::RetainEngine => slot.engine_references += count,
                CountOperation::ReleaseEngine => slot.engine_references -= count,
                CountOperation::PinExternal => slot.external_completion_pins += count,
                CountOperation::UnpinExternal => slot.external_completion_pins -= count,
                CountOperation::PinEngine => slot.engine_completion_pins += count,
                CountOperation::UnpinEngine => slot.engine_completion_pins -= count,
            }
            if slot.total_references() == 0
                && slot.total_completion_pins() == 0
                && slot.engine_read_leases == 0
                && !slot.engine_write_lease
            {
                slot.allocated = false;
                self.free.push(handle.slot);
            }
        }
        Ok(())
    }

    pub(crate) fn retain_handles(&mut self, handles: &[StateHandle]) -> PyResult<()> {
        self.apply_values(handles, CountOperation::RetainEngine)
    }

    pub(crate) fn release_handles(&mut self, handles: &[StateHandle]) -> PyResult<()> {
        self.apply_values(handles, CountOperation::ReleaseEngine)
    }

    pub(crate) fn pin_handles(&mut self, handles: &[StateHandle]) -> PyResult<()> {
        self.apply_values(handles, CountOperation::PinEngine)
    }

    pub(crate) fn unpin_handles(&mut self, handles: &[StateHandle]) -> PyResult<()> {
        self.apply_values(handles, CountOperation::UnpinEngine)
    }

    pub(crate) fn prepare_engine_access(
        &self,
        claims: &[StateAccessClaim],
    ) -> PyResult<PreparedStateAccessLease> {
        let mut sorted = Vec::new();
        sorted
            .try_reserve_exact(claims.len())
            .map_err(|_| PyMemoryError::new_err("could not prepare state access claims"))?;
        sorted.extend(claims.iter().cloned());
        sorted.sort_unstable_by(|left, right| {
            left.handle
                .cmp(&right.handle)
                .then(left.row_index.cmp(&right.row_index))
        });

        let mut entries = Vec::new();
        entries
            .try_reserve_exact(sorted.len())
            .map_err(|_| PyMemoryError::new_err("could not prepare state access lease"))?;
        let mut start = 0;
        while start < sorted.len() {
            let mut end = start + 1;
            while end < sorted.len() && sorted[end].handle == sorted[start].handle {
                end += 1;
            }
            let claims = &sorted[start..end];
            let first_row = claims[0].row_index;
            let multiple_rows = claims.iter().any(|claim| claim.row_index != first_row);
            let has_write = claims
                .iter()
                .any(|claim| claim.mode == StateAccessMode::Write);
            if multiple_rows && has_write {
                return Err(PyValueError::new_err(
                    "state write access conflicts with another batch row",
                ));
            }
            let mode = if has_write {
                StateAccessMode::Write
            } else {
                StateAccessMode::Read
            };
            let slot = self.slot_for(&claims[0].handle)?;
            Self::validate_access_available(slot, mode)?;
            entries.push(StateAccessEntry {
                handle: claims[0].handle.clone(),
                mode,
            });
            start = end;
        }
        Ok(PreparedStateAccessLease {
            pool_id: self.id,
            entries,
        })
    }

    pub(crate) fn acquire_engine_access(
        &mut self,
        prepared: PreparedStateAccessLease,
    ) -> PyResult<StateAccessLease> {
        if prepared.pool_id != self.id {
            return Err(PyValueError::new_err(
                "prepared state access belongs to a different pool",
            ));
        }
        for entry in &prepared.entries {
            let slot = self.slot_for(&entry.handle)?;
            Self::validate_access_available(slot, entry.mode)?;
        }
        for entry in &prepared.entries {
            let slot = &mut self.slots[entry.handle.slot as usize];
            match entry.mode {
                StateAccessMode::Read => slot.engine_read_leases += 1,
                StateAccessMode::Write => slot.engine_write_lease = true,
            }
        }
        Ok(StateAccessLease {
            pool_id: prepared.pool_id,
            entries: prepared.entries,
        })
    }

    pub(crate) fn release_engine_access(&mut self, lease: StateAccessLease) {
        assert_eq!(lease.pool_id, self.id, "state access lease keeps its pool");
        for entry in &lease.entries {
            let slot = self
                .slots
                .get(entry.handle.slot as usize)
                .expect("state access lease keeps an in-range slot");
            assert!(slot.allocated, "state access lease keeps its slot live");
            assert_eq!(
                slot.generation, entry.handle.generation,
                "state access lease keeps its generation"
            );
            match entry.mode {
                StateAccessMode::Read => assert!(
                    slot.engine_read_leases != 0,
                    "read access lease remains active"
                ),
                StateAccessMode::Write => {
                    assert!(slot.engine_write_lease, "write access lease remains active")
                }
            }
        }
        for entry in lease.entries {
            let slot = &mut self.slots[entry.handle.slot as usize];
            match entry.mode {
                StateAccessMode::Read => slot.engine_read_leases -= 1,
                StateAccessMode::Write => slot.engine_write_lease = false,
            }
            if slot.total_references() == 0
                && slot.total_completion_pins() == 0
                && slot.engine_read_leases == 0
                && !slot.engine_write_lease
            {
                slot.allocated = false;
                self.free.push(entry.handle.slot);
            }
        }
    }

    fn validate_access_available(slot: &Slot, mode: StateAccessMode) -> PyResult<()> {
        if slot.total_references() == 0 {
            return Err(PyValueError::new_err(
                "cannot access state pending retirement",
            ));
        }
        match mode {
            StateAccessMode::Read => {
                if slot.engine_write_lease {
                    return Err(PyValueError::new_err(
                        "state read access conflicts with an active writer",
                    ));
                }
                slot.engine_read_leases
                    .checked_add(1)
                    .ok_or_else(|| PyRuntimeError::new_err("state read lease overflow"))?;
            }
            StateAccessMode::Write => {
                if slot.engine_write_lease || slot.engine_read_leases != 0 {
                    return Err(PyValueError::new_err(
                        "state write access conflicts with an active lease",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_mapping_claim_is_permanent_and_exact_rebind_is_idempotent() {
        let mut pool = StatePool::new(2).unwrap();
        assert_eq!(pool.physical_mapping(), None);
        pool.bind_physical_mapping("synthetic.ssm_state.v1", 41)
            .unwrap();
        pool.bind_physical_mapping("synthetic.ssm_state.v1", 41)
            .unwrap();
        assert_eq!(
            pool.physical_mapping(),
            Some(("synthetic.ssm_state.v1".to_owned(), 41))
        );

        assert!(
            pool.bind_physical_mapping("synthetic.ssm_state.v2", 41)
                .is_err()
        );
        assert!(
            pool.bind_physical_mapping("synthetic.ssm_state.v1", 42)
                .is_err()
        );
        assert_eq!(
            pool.physical_mapping(),
            Some(("synthetic.ssm_state.v1".to_owned(), 41))
        );
    }

    #[test]
    fn state_mapping_arena_ids_are_positive_and_never_reused() {
        let first = mint_state_mapping_arena_id().unwrap();
        let second = mint_state_mapping_arena_id().unwrap();
        let third = mint_state_mapping_arena_id().unwrap();
        assert_ne!(first, 0);
        assert_eq!(second, first + 1);
        assert_eq!(third, second + 1);
    }

    #[test]
    fn completion_pin_defers_reuse_and_generation_rejects_stale_handle() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let first = pool.allocate(py, 1).unwrap().pop().unwrap();
            pool.pin(py, vec![first.clone_ref(py)]).unwrap();
            pool.release(py, vec![first.clone_ref(py)]).unwrap();
            assert_eq!(pool.free_count(), 0);
            assert_eq!(pool.retiring_count(), 1);
            pool.unpin(py, vec![first.clone_ref(py)]).unwrap();
            assert_eq!(pool.free_count(), 1);

            let second = pool.allocate(py, 1).unwrap().pop().unwrap();
            assert_eq!(first.borrow(py).slot, second.borrow(py).slot);
            assert_ne!(first.borrow(py).generation, second.borrow(py).generation);
            assert!(pool.slot_for(&first.borrow(py)).is_err());
        });
    }

    #[test]
    fn failed_bulk_release_is_atomic() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let handle = pool.allocate(py, 1).unwrap().pop().unwrap();
            assert!(
                pool.release(py, vec![handle.clone_ref(py), handle.clone_ref(py)])
                    .is_err()
            );
            assert_eq!(pool.counts(handle.borrow(py)).unwrap(), (1, 0));
        });
    }

    #[test]
    fn caller_operations_cannot_consume_engine_ownership() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let handle = pool.allocate(py, 1).unwrap().pop().unwrap();
            let value = handle.borrow(py).clone();
            pool.retain_handles(std::slice::from_ref(&value)).unwrap();
            pool.release(py, vec![handle.clone_ref(py)]).unwrap();
            pool.pin_handles(std::slice::from_ref(&value)).unwrap();

            assert!(pool.release(py, vec![handle.clone_ref(py)]).is_err());
            assert!(pool.unpin(py, vec![handle.clone_ref(py)]).is_err());
            assert_eq!(pool.counts(handle.borrow(py)).unwrap(), (1, 1));

            pool.release_handles(std::slice::from_ref(&value)).unwrap();
            pool.unpin_handles(std::slice::from_ref(&value)).unwrap();
            assert_eq!(pool.free_count(), 1);
        });
    }

    #[test]
    fn shared_reads_deduplicate_per_batch_and_compose_across_leases() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let handle = pool.allocate(py, 1).unwrap().pop().unwrap();
            let value = handle.borrow(py).clone();
            let claims = vec![
                StateAccessClaim::new(0, value.clone(), StateAccessMode::Read),
                StateAccessClaim::new(0, value.clone(), StateAccessMode::Read),
                StateAccessClaim::new(1, value.clone(), StateAccessMode::Read),
            ];
            let first = pool
                .acquire_engine_access(pool.prepare_engine_access(&claims).unwrap())
                .unwrap();
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (1, false));

            let second = pool
                .acquire_engine_access(
                    pool.prepare_engine_access(&[StateAccessClaim::new(
                        2,
                        value,
                        StateAccessMode::Read,
                    )])
                    .unwrap(),
                )
                .unwrap();
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (2, false));
            pool.release_engine_access(first);
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (1, false));
            pool.release_engine_access(second);
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (0, false));
        });
    }

    #[test]
    fn writes_are_exclusive_and_same_row_write_dominates_duplicates() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let handle = pool.allocate(py, 1).unwrap().pop().unwrap();
            let value = handle.borrow(py).clone();

            let conflicting = vec![
                StateAccessClaim::new(0, value.clone(), StateAccessMode::Read),
                StateAccessClaim::new(1, value.clone(), StateAccessMode::Write),
            ];
            assert!(pool.prepare_engine_access(&conflicting).is_err());
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (0, false));

            let same_row = vec![
                StateAccessClaim::new(0, value.clone(), StateAccessMode::Read),
                StateAccessClaim::new(0, value.clone(), StateAccessMode::Write),
            ];
            let lease = pool
                .acquire_engine_access(pool.prepare_engine_access(&same_row).unwrap())
                .unwrap();
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (0, true));
            assert!(
                pool.prepare_engine_access(&[StateAccessClaim::new(
                    1,
                    value,
                    StateAccessMode::Read,
                )])
                .is_err()
            );
            pool.release_engine_access(lease);
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (0, false));
        });
    }

    #[test]
    fn prepared_write_revalidates_exclusion_at_acquisition() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let handle = pool.allocate(py, 1).unwrap().pop().unwrap();
            let value = handle.borrow(py).clone();
            let claims = [StateAccessClaim::new(0, value, StateAccessMode::Write)];
            let first = pool.prepare_engine_access(&claims).unwrap();
            let second = pool.prepare_engine_access(&claims).unwrap();
            let lease = pool.acquire_engine_access(first).unwrap();
            assert!(pool.acquire_engine_access(second).is_err());
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (0, true));
            pool.release_engine_access(lease);
            assert_eq!(pool.access_counts(handle.borrow(py)).unwrap(), (0, false));
        });
    }

    #[test]
    fn access_lease_defers_reuse_and_stale_preparation_cannot_touch_reuse() {
        Python::initialize();
        Python::attach(|py| {
            let mut pool = StatePool::new(1).unwrap();
            let first = pool.allocate(py, 1).unwrap().pop().unwrap();
            let first_value = first.borrow(py).clone();
            let claims = [StateAccessClaim::new(
                0,
                first_value.clone(),
                StateAccessMode::Read,
            )];
            let stale_prepared = pool.prepare_engine_access(&claims).unwrap();
            let lease = pool
                .acquire_engine_access(pool.prepare_engine_access(&claims).unwrap())
                .unwrap();
            pool.release(py, vec![first.clone_ref(py)]).unwrap();
            assert_eq!(pool.free_count(), 0);
            assert_eq!(pool.retiring_count(), 1);
            pool.release_engine_access(lease);
            assert_eq!(pool.free_count(), 1);

            let second = pool.allocate(py, 1).unwrap().pop().unwrap();
            assert_eq!(first.borrow(py).slot, second.borrow(py).slot);
            assert_ne!(first.borrow(py).generation, second.borrow(py).generation);
            assert!(pool.acquire_engine_access(stale_prepared).is_err());
            assert_eq!(pool.access_counts(second.borrow(py)).unwrap(), (0, false));

            let mut foreign = StatePool::new(1).unwrap();
            let prepared = pool
                .prepare_engine_access(&[StateAccessClaim::new(
                    0,
                    second.borrow(py).clone(),
                    StateAccessMode::Read,
                )])
                .unwrap();
            assert!(foreign.acquire_engine_access(prepared).is_err());
            assert_eq!(foreign.free_count(), 1);
        });
    }
}
