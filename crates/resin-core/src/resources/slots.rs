use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_MAPPING_ARENA_ID: AtomicU64 = AtomicU64::new(1);

pub fn mint_mapping_arena_id() -> Result<u64, SlotPoolError> {
    NEXT_MAPPING_ARENA_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| SlotPoolError::runtime("state mapping arena id space is exhausted"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PhysicalMapping {
    protocol: String,
    arena_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SlotHandle {
    pool_id: u64,
    slot: u32,
    generation: u64,
}

impl SlotHandle {
    pub const fn pool_id(self) -> u64 {
        self.pool_id
    }

    pub const fn slot(self) -> u32 {
        self.slot
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessMode {
    Read,
    Write,
}

impl AccessMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AccessClaim {
    row_index: usize,
    handle: SlotHandle,
    mode: AccessMode,
}

impl AccessClaim {
    pub const fn new(row_index: usize, handle: SlotHandle, mode: AccessMode) -> Self {
        Self {
            row_index,
            handle,
            mode,
        }
    }
}

#[derive(Debug)]
struct AccessEntry {
    handle: SlotHandle,
    mode: AccessMode,
}

#[derive(Debug)]
#[must_use = "prepared slot access must be acquired or discarded"]
pub struct PreparedAccess {
    pool_id: u64,
    entries: Vec<AccessEntry>,
}

#[derive(Debug)]
#[must_use = "slot access must be released or retained as quarantine"]
pub struct AccessLease {
    pool_id: u64,
    entries: Vec<AccessEntry>,
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

    fn reclaimable(&self) -> bool {
        self.total_references() == 0
            && self.total_completion_pins() == 0
            && self.engine_read_leases == 0
            && !self.engine_write_lease
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotPoolErrorKind {
    Value,
    Memory,
    Runtime,
    PendingRetirement,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotPoolError {
    kind: SlotPoolErrorKind,
    message: Cow<'static, str>,
}

impl SlotPoolError {
    fn value(message: &'static str) -> Self {
        Self::new(SlotPoolErrorKind::Value, message)
    }

    fn memory(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(SlotPoolErrorKind::Memory, message)
    }

    fn runtime(message: &'static str) -> Self {
        Self::new(SlotPoolErrorKind::Runtime, message)
    }

    fn pending() -> Self {
        Self::new(
            SlotPoolErrorKind::PendingRetirement,
            "state is pending retirement",
        )
    }

    fn new(kind: SlotPoolErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub const fn kind(&self) -> SlotPoolErrorKind {
        self.kind
    }
}

impl fmt::Display for SlotPoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SlotPoolError {}

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

#[derive(Debug)]
pub struct SlotPool {
    id: u64,
    slots: Vec<Slot>,
    free: Vec<u32>,
    physical_mapping: Option<PhysicalMapping>,
}

impl SlotPool {
    pub fn new(capacity: usize) -> Result<Self, SlotPoolError> {
        if capacity == 0 {
            return Err(SlotPoolError::value("capacity must be positive"));
        }
        if capacity > u32::MAX as usize {
            return Err(SlotPoolError::value(
                "capacity exceeds the slot index range",
            ));
        }
        let id = NEXT_POOL_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| SlotPoolError::runtime("state pool id space is exhausted"))?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| SlotPoolError::memory("could not allocate state-pool metadata"))?;
        slots.resize_with(capacity, Slot::default);
        let mut free = Vec::new();
        free.try_reserve_exact(capacity)
            .map_err(|_| SlotPoolError::memory("could not allocate state-pool metadata"))?;
        free.extend((0..capacity as u32).rev());
        Ok(Self {
            id,
            slots,
            free,
            physical_mapping: None,
        })
    }

    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    pub fn active_count(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn retiring_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.allocated && slot.total_references() == 0)
            .count()
    }

    pub fn allocate(&mut self, count: usize) -> Result<Vec<SlotHandle>, SlotPoolError> {
        self.allocate_owned(count, true)
    }

    /// Allocate slots whose initial lifetime belongs to the inference engine.
    /// Unlike [`Self::allocate`], callers do not need to manufacture a Python
    /// or other external owner before admitting a request.
    pub fn allocate_engine(&mut self, count: usize) -> Result<Vec<SlotHandle>, SlotPoolError> {
        self.allocate_owned(count, false)
    }

    fn allocate_owned(
        &mut self,
        count: usize,
        external_owner: bool,
    ) -> Result<Vec<SlotHandle>, SlotPoolError> {
        if count == 0 {
            return Err(SlotPoolError::value("count must be positive"));
        }
        if count > self.free.len() {
            return Err(SlotPoolError::memory(format!(
                "state pool has {} free slots but {count} were requested",
                self.free.len()
            )));
        }
        for index in self.free[self.free.len() - count..].iter().copied() {
            self.slots[index as usize]
                .generation
                .checked_add(1)
                .ok_or_else(|| SlotPoolError::runtime("slot generation is exhausted"))?;
        }
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(count)
            .map_err(|_| SlotPoolError::memory("could not allocate state-pool metadata"))?;
        for _ in 0..count {
            let slot_index = self.free.pop().expect("capacity preflighted");
            let slot = &mut self.slots[slot_index as usize];
            slot.generation += 1;
            slot.external_references = u32::from(external_owner);
            slot.engine_references = u32::from(!external_owner);
            slot.external_completion_pins = 0;
            slot.engine_completion_pins = 0;
            slot.engine_read_leases = 0;
            slot.engine_write_lease = false;
            slot.allocated = true;
            handles.push(SlotHandle {
                pool_id: self.id,
                slot: slot_index,
                generation: slot.generation,
            });
        }
        Ok(handles)
    }

    pub fn contains(&self, handle: SlotHandle) -> bool {
        self.slot_for(handle).is_ok()
    }

    pub fn counts(&self, handle: SlotHandle) -> Result<(u32, u32), SlotPoolError> {
        let slot = self.slot_for(handle)?;
        Ok((
            u32::try_from(slot.total_references()).expect("reference total is bounded"),
            u32::try_from(slot.total_completion_pins()).expect("pin total is bounded"),
        ))
    }

    pub fn access_counts(&self, handle: SlotHandle) -> Result<(u32, bool), SlotPoolError> {
        let slot = self.slot_for(handle)?;
        Ok((slot.engine_read_leases, slot.engine_write_lease))
    }

    pub fn physical_mapping(&self) -> Option<(&str, u64)> {
        self.physical_mapping
            .as_ref()
            .map(|mapping| (mapping.protocol.as_str(), mapping.arena_id))
    }

    pub fn bind_physical_mapping(
        &mut self,
        protocol: &str,
        arena_id: u64,
    ) -> Result<(), SlotPoolError> {
        if protocol.is_empty() {
            return Err(SlotPoolError::value(
                "state mapping protocol must be non-empty",
            ));
        }
        if arena_id == 0 {
            return Err(SlotPoolError::value(
                "state mapping arena id must be positive",
            ));
        }
        let candidate = PhysicalMapping {
            protocol: protocol.to_owned(),
            arena_id,
        };
        match &self.physical_mapping {
            None => {
                self.physical_mapping = Some(candidate);
                Ok(())
            }
            Some(current) if current == &candidate => Ok(()),
            Some(current) => Err(SlotPoolError::new(
                SlotPoolErrorKind::Value,
                format!(
                    "state pool is already bound to physical mapping ({:?}, {}), cannot bind ({:?}, {})",
                    current.protocol, current.arena_id, candidate.protocol, candidate.arena_id
                ),
            )),
        }
    }

    pub fn retain_external(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::RetainExternal)
    }

    pub fn release_external(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::ReleaseExternal)
    }

    pub fn pin_external(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::PinExternal)
    }

    pub fn unpin_external(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::UnpinExternal)
    }

    pub fn retain_engine(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::RetainEngine)
    }

    pub fn release_engine(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::ReleaseEngine)
    }

    pub fn pin_engine(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::PinEngine)
    }

    pub fn unpin_engine(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<(), SlotPoolError> {
        self.apply(handles, CountOperation::UnpinEngine)
    }

    pub fn prepare_access(&self, claims: &[AccessClaim]) -> Result<PreparedAccess, SlotPoolError> {
        let mut sorted = Vec::new();
        sorted
            .try_reserve_exact(claims.len())
            .map_err(|_| SlotPoolError::memory("could not allocate state-pool metadata"))?;
        sorted.extend_from_slice(claims);
        sorted.sort_unstable_by(|left, right| {
            left.handle
                .cmp(&right.handle)
                .then(left.row_index.cmp(&right.row_index))
        });

        let mut entries = Vec::new();
        entries
            .try_reserve_exact(sorted.len())
            .map_err(|_| SlotPoolError::memory("could not allocate state-pool metadata"))?;
        let mut start = 0;
        while start < sorted.len() {
            let mut end = start + 1;
            while end < sorted.len() && sorted[end].handle == sorted[start].handle {
                end += 1;
            }
            let claims = &sorted[start..end];
            let first_row = claims[0].row_index;
            let multiple_rows = claims.iter().any(|claim| claim.row_index != first_row);
            let has_write = claims.iter().any(|claim| claim.mode == AccessMode::Write);
            if multiple_rows && has_write {
                return Err(SlotPoolError::value(
                    "state write access conflicts with another batch row",
                ));
            }
            let mode = if has_write {
                AccessMode::Write
            } else {
                AccessMode::Read
            };
            Self::validate_access(self.slot_for(claims[0].handle)?, mode)?;
            entries.push(AccessEntry {
                handle: claims[0].handle,
                mode,
            });
            start = end;
        }
        Ok(PreparedAccess {
            pool_id: self.id,
            entries,
        })
    }

    pub fn acquire_access(
        &mut self,
        prepared: PreparedAccess,
    ) -> Result<AccessLease, SlotPoolError> {
        if prepared.pool_id != self.id {
            return Err(SlotPoolError::value(
                "prepared state access belongs to a different pool",
            ));
        }
        for entry in &prepared.entries {
            Self::validate_access(self.slot_for(entry.handle)?, entry.mode)?;
        }
        for entry in &prepared.entries {
            let slot = &mut self.slots[entry.handle.slot as usize];
            match entry.mode {
                AccessMode::Read => slot.engine_read_leases += 1,
                AccessMode::Write => slot.engine_write_lease = true,
            }
        }
        Ok(AccessLease {
            pool_id: prepared.pool_id,
            entries: prepared.entries,
        })
    }

    pub fn release_access(&mut self, lease: AccessLease) {
        assert_eq!(lease.pool_id, self.id, "slot access lease keeps its pool");
        for entry in &lease.entries {
            let slot = self
                .slots
                .get(entry.handle.slot as usize)
                .expect("slot access lease keeps an in-range slot");
            assert!(slot.allocated, "slot access lease keeps its slot live");
            assert_eq!(
                slot.generation, entry.handle.generation,
                "slot access lease keeps its generation"
            );
            match entry.mode {
                AccessMode::Read => assert_ne!(slot.engine_read_leases, 0),
                AccessMode::Write => assert!(slot.engine_write_lease),
            }
        }
        for entry in lease.entries {
            let slot = &mut self.slots[entry.handle.slot as usize];
            match entry.mode {
                AccessMode::Read => slot.engine_read_leases -= 1,
                AccessMode::Write => slot.engine_write_lease = false,
            }
            self.reclaim_if_unused(entry.handle.slot);
        }
    }

    fn slot_for(&self, handle: SlotHandle) -> Result<&Slot, SlotPoolError> {
        if handle.pool_id != self.id {
            return Err(SlotPoolError::value(
                "state handle belongs to a different pool",
            ));
        }
        let slot = self
            .slots
            .get(handle.slot as usize)
            .ok_or_else(|| SlotPoolError::value("state handle slot is out of range"))?;
        if !slot.allocated || slot.generation != handle.generation {
            return Err(SlotPoolError::value("state handle is stale"));
        }
        Ok(slot)
    }

    fn counts_for(
        &self,
        handles: impl IntoIterator<Item = SlotHandle>,
    ) -> Result<BTreeMap<SlotHandle, u32>, SlotPoolError> {
        let mut counts = BTreeMap::new();
        for handle in handles {
            self.slot_for(handle)?;
            let count = counts.entry(handle).or_insert(0_u32);
            *count = count
                .checked_add(1)
                .ok_or_else(|| SlotPoolError::runtime("handle count overflow"))?;
        }
        Ok(counts)
    }

    fn apply(
        &mut self,
        handles: impl IntoIterator<Item = SlotHandle>,
        operation: CountOperation,
    ) -> Result<(), SlotPoolError> {
        let counts = self.counts_for(handles)?;
        for (handle, count) in &counts {
            let slot = self.slot_for(*handle)?;
            match operation {
                CountOperation::RetainExternal | CountOperation::RetainEngine => {
                    if slot.total_references() == 0 {
                        return Err(SlotPoolError::pending());
                    }
                    if slot.total_references() + u64::from(*count) > u64::from(u32::MAX) {
                        return Err(SlotPoolError::runtime("reference count overflow"));
                    }
                }
                CountOperation::ReleaseExternal => {
                    if *count > slot.external_references {
                        return Err(SlotPoolError::value(
                            "release count exceeds caller-owned state references",
                        ));
                    }
                }
                CountOperation::ReleaseEngine => {
                    if *count > slot.engine_references {
                        return Err(SlotPoolError::value(
                            "release count exceeds engine-owned state references",
                        ));
                    }
                }
                CountOperation::PinExternal | CountOperation::PinEngine => {
                    if slot.total_references() == 0 {
                        return Err(SlotPoolError::pending());
                    }
                    if slot.total_completion_pins() + u64::from(*count) > u64::from(u32::MAX) {
                        return Err(SlotPoolError::runtime("completion pin overflow"));
                    }
                }
                CountOperation::UnpinExternal => {
                    if *count > slot.external_completion_pins {
                        return Err(SlotPoolError::value(
                            "unpin count exceeds caller-owned completion pins",
                        ));
                    }
                }
                CountOperation::UnpinEngine => {
                    if *count > slot.engine_completion_pins {
                        return Err(SlotPoolError::value(
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
            self.reclaim_if_unused(handle.slot);
        }
        Ok(())
    }

    fn reclaim_if_unused(&mut self, slot_index: u32) {
        let slot = &mut self.slots[slot_index as usize];
        if slot.allocated && slot.reclaimable() {
            slot.allocated = false;
            self.free.push(slot_index);
        }
    }

    fn validate_access(slot: &Slot, mode: AccessMode) -> Result<(), SlotPoolError> {
        if slot.total_references() == 0 {
            return Err(SlotPoolError::pending());
        }
        match mode {
            AccessMode::Read => {
                if slot.engine_write_lease {
                    return Err(SlotPoolError::value(
                        "state read access conflicts with an active writer",
                    ));
                }
                slot.engine_read_leases
                    .checked_add(1)
                    .ok_or_else(|| SlotPoolError::runtime("state read lease overflow"))?;
            }
            AccessMode::Write => {
                if slot.engine_write_lease || slot.engine_read_leases != 0 {
                    return Err(SlotPoolError::value(
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
    fn generations_and_retirement_are_explicit() {
        let mut pool = SlotPool::new(1).unwrap();
        let first = pool.allocate(1).unwrap()[0];
        pool.retain_engine([first]).unwrap();
        pool.release_external([first]).unwrap();
        pool.pin_engine([first]).unwrap();
        pool.release_engine([first]).unwrap();
        assert_eq!(pool.retiring_count(), 1);
        pool.unpin_engine([first]).unwrap();
        let second = pool.allocate(1).unwrap()[0];
        assert_eq!(first.slot(), second.slot());
        assert!(second.generation() > first.generation());
        assert_eq!(
            pool.counts(first).unwrap_err().to_string(),
            "state handle is stale"
        );
    }

    #[test]
    fn batch_access_is_preflighted_and_exclusive() {
        let mut pool = SlotPool::new(1).unwrap();
        let handle = pool.allocate(1).unwrap()[0];
        let prepared = pool
            .prepare_access(&[AccessClaim::new(0, handle, AccessMode::Write)])
            .unwrap();
        let lease = pool.acquire_access(prepared).unwrap();
        assert_eq!(
            pool.prepare_access(&[AccessClaim::new(0, handle, AccessMode::Read)])
                .unwrap_err()
                .to_string(),
            "state read access conflicts with an active writer"
        );
        pool.release_access(lease);
        assert!(
            pool.prepare_access(&[AccessClaim::new(0, handle, AccessMode::Read)])
                .is_ok()
        );
    }

    #[test]
    fn mapping_is_permanent_and_exact() {
        let mut pool = SlotPool::new(1).unwrap();
        pool.bind_physical_mapping("state.v1", 7).unwrap();
        pool.bind_physical_mapping("state.v1", 7).unwrap();
        assert!(pool.bind_physical_mapping("state.v2", 7).is_err());
    }
}
