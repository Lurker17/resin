//! Fungible physical placement for heterogeneous paged-cache groups.
//!
//! The pool owns equal-sized physical parents.  An empty parent may serve any
//! cache group; while occupied, it is bound to one group and subdivided into
//! that group's configured number of child blocks.  Token retention, request
//! page-table policy, prefix identity, and device byte layout deliberately live
//! outside this module.

use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Ordered cache-group identity.  The runner declaration and native layout use
/// the same index; there is no process-global group registry.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GroupId(u32);

impl GroupId {
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    pub(crate) const fn index(self) -> u32 {
        self.0
    }
}

/// Stable identity of one live child block. This is a copyable capability, not
/// ownership; request ownership stays in `PendingRequestPages`, `RequestPages`,
/// or an active writer lease.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct PageHandle {
    pool_id: u64,
    group: GroupId,
    parent: u32,
    slot: u32,
    generation: u64,
}

/// Pool-local identity of the one sequence allowed to mutate private pages.
///
/// The Engine may use its already-unique request id. Keeping the type distinct
/// prevents a page handle or group index from being substituted accidentally.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct SequenceId(u64);

impl SequenceId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// Pool-minted, non-reusing identity of one request reservation. Logical
/// sequence ids are caller metadata and are not trusted as write authority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct WriterId(u64);

impl PageHandle {
    /// One-based physical parent id.  Parent zero is reserved for the device
    /// null page and is never allocated.
    #[cfg(test)]
    pub(crate) const fn parent(self) -> u32 {
        self.parent
    }

    #[cfg(test)]
    pub(crate) const fn slot(self) -> u32 {
        self.slot
    }

    #[cfg(test)]
    pub(crate) const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GroupDemand {
    group: GroupId,
    blocks: usize,
}

impl GroupDemand {
    pub(crate) const fn new(group: GroupId, blocks: usize) -> Self {
        Self { group, blocks }
    }
}

/// One ordered physical table for one input demand. Duplicate group demands
/// retain distinct tables even though placement shares partially filled
/// parents.
#[derive(Debug)]
pub(crate) struct GroupTable<R> {
    group: GroupId,
    pages: Vec<R>,
}

impl<R> GroupTable<R> {
    pub(crate) const fn group(&self) -> GroupId {
        self.group
    }

    pub(crate) fn pages(&self) -> &[R] {
        &self.pages
    }
}

/// Move-only ownership of one request page. A raw handle may be copied for
/// metadata, but only this token accounts for the request reference.
#[derive(Debug)]
pub(crate) struct PageRef {
    handle: PageHandle,
}

impl PageRef {
    pub(crate) const fn handle(&self) -> PageHandle {
        self.handle
    }
}

/// Provisional ownership returned by one all-or-nothing request reservation.
#[derive(Debug)]
#[must_use = "pending request pages must be promoted or aborted"]
pub(crate) struct PendingRequestPages {
    pool_id: u64,
    sequence: SequenceId,
    writer: WriterId,
    groups: Vec<GroupTable<PageRef>>,
}

impl PendingRequestPages {
    #[cfg(test)]
    pub(crate) const fn sequence(&self) -> SequenceId {
        self.sequence
    }

    #[cfg(test)]
    pub(crate) fn groups(&self) -> &[GroupTable<PageRef>] {
        &self.groups
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.groups.iter().map(|group| group.pages.len()).sum()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Move-only ownership of all physical pages belonging to one live request.
#[derive(Debug)]
#[must_use = "request pages must be retired through their issuing pool"]
pub(crate) struct RequestPages {
    pool_id: u64,
    sequence: SequenceId,
    writer: WriterId,
    groups: Vec<GroupTable<PageRef>>,
}

impl RequestPages {
    pub(crate) const fn sequence(&self) -> SequenceId {
        self.sequence
    }

    pub(crate) fn groups(&self) -> &[GroupTable<PageRef>] {
        &self.groups
    }

    /// Construct an exact-writer claim. Passing a page from another request is
    /// harmless: pool validation rejects the substituted writer atomically.
    pub(crate) const fn write_claim(&self, page: &PageRef) -> WriterAccessClaim {
        WriterAccessClaim {
            pool_id: self.pool_id,
            writer: self.writer,
            handle: page.handle,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WriterAccessClaim {
    pool_id: u64,
    writer: WriterId,
    handle: PageHandle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedWriterClaim {
    claim: WriterAccessClaim,
    expected_write_epoch: u64,
}

#[derive(Debug)]
#[must_use = "prepared writer access must be acquired or discarded"]
pub(crate) struct PreparedWriterAccess {
    pool_id: u64,
    claims: Vec<PreparedWriterClaim>,
}

#[derive(Debug)]
#[must_use = "writer access must be released or retained as quarantine"]
pub(crate) struct WriterAccessLease {
    pool_id: u64,
    claims: Vec<PreparedWriterClaim>,
}

impl WriterAccessLease {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.claims.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CachePoolError {
    InvalidParentCapacity,
    EmptyGroupLayout,
    TooManyGroups,
    KernelPageIdOverflow {
        group: GroupId,
    },
    PoolIdExhausted,
    WriterIdExhausted,
    MetadataAllocation,
    UnknownGroup {
        group: GroupId,
    },
    DemandOverflow,
    InsufficientCapacity {
        group: GroupId,
        requested: usize,
        available: usize,
    },
    ParentGenerationExhausted {
        parent: u32,
    },
    ForeignPool,
    StaleHandle,
    ParentGroupMismatch {
        handle_group: GroupId,
        bound_group: GroupId,
    },
    DuplicateHandle,
    ForeignAccess,
    WrongWriter,
    PageNotWritable,
    DuplicateWriteClaim,
    AccessConflict,
    WriteEpochExhausted,
    WriteEpochChanged {
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for CachePoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParentCapacity => {
                formatter.write_str("cache pool parent capacity must be positive")
            }
            Self::EmptyGroupLayout => {
                formatter.write_str("cache pool requires at least one cache group")
            }
            Self::TooManyGroups => formatter.write_str("cache pool has too many groups"),
            Self::KernelPageIdOverflow { group } => write!(
                formatter,
                "cache group {} exceeds the signed int32 kernel page-id range",
                group.index()
            ),
            Self::PoolIdExhausted => formatter.write_str("cache pool id space is exhausted"),
            Self::WriterIdExhausted => {
                formatter.write_str("cache pool writer id space is exhausted")
            }
            Self::MetadataAllocation => {
                formatter.write_str("could not allocate cache-pool metadata")
            }
            Self::UnknownGroup { group } => {
                write!(formatter, "cache group {} is not declared", group.index())
            }
            Self::DemandOverflow => formatter.write_str("cache block demand overflows usize"),
            Self::InsufficientCapacity {
                group,
                requested,
                available,
            } => write!(
                formatter,
                "cache group {} requested {requested} blocks but only {available} fit",
                group.index()
            ),
            Self::ParentGenerationExhausted { parent } => {
                write!(formatter, "cache parent {parent} generation is exhausted")
            }
            Self::ForeignPool => {
                formatter.write_str("cache allocation belongs to a different pool")
            }
            Self::StaleHandle => formatter.write_str("cache page handle is stale"),
            Self::ParentGroupMismatch {
                handle_group,
                bound_group,
            } => write!(
                formatter,
                "cache page names group {}, but its parent is bound to group {}",
                handle_group.index(),
                bound_group.index()
            ),
            Self::DuplicateHandle => {
                formatter.write_str("cache allocation contains a duplicate page handle")
            }
            Self::ForeignAccess => formatter.write_str("writer access belongs to another pool"),
            Self::WrongWriter => {
                formatter.write_str("cache page belongs to a different request writer")
            }
            Self::PageNotWritable => {
                formatter.write_str("cache page has no exclusive mutable request owner")
            }
            Self::DuplicateWriteClaim => {
                formatter.write_str("one cache page has duplicate exclusive write claims")
            }
            Self::AccessConflict => {
                formatter.write_str("cache page already has an incompatible access lease")
            }
            Self::WriteEpochExhausted => {
                formatter.write_str("cache page write completion epoch is exhausted")
            }
            Self::WriteEpochChanged { expected, actual } => write!(
                formatter,
                "cache page write epoch changed from {expected} to {actual} after access preparation"
            ),
        }
    }
}

impl std::error::Error for CachePoolError {}

/// A failed retirement returns the exact move-only request token unchanged.
#[derive(Debug)]
pub(crate) struct RetireRequestFailure {
    pub(crate) error: CachePoolError,
    pub(crate) request: RequestPages,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PageMode {
    Reserved { writer: WriterId },
    Mutable { writer: WriterId },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum LeaseState {
    #[default]
    Idle,
    Writer,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PageSlot {
    generation: u64,
    mode: Option<PageMode>,
    request_refs: u32,
    lease: LeaseState,
    write_epoch: u64,
}

impl PageSlot {
    fn is_live(self) -> bool {
        matches!(self.mode, Some(PageMode::Reserved { .. }))
            || self.request_refs != 0
            || self.lease != LeaseState::Idle
    }

    fn reserved_request(generation: u64, writer: WriterId) -> Self {
        Self {
            generation,
            mode: Some(PageMode::Reserved { writer }),
            ..Self::default()
        }
    }

    fn clear_lifetime(&mut self) {
        debug_assert!(!self.is_live());
        self.mode = None;
        self.request_refs = 0;
        self.lease = LeaseState::Idle;
        self.write_epoch = 0;
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Parent {
    binding: Option<GroupId>,
    slots: Vec<PageSlot>,
    occupied_slots: u32,
    /// Last generation issued anywhere in this parent.  It survives slot
    /// reuse, slot-vector clearing, and cross-group rebinding.
    next_generation: u64,
    /// Position in the pool's compact list of partially filled parents.
    /// Full and empty parents are absent.
    partial_index: Option<usize>,
}

#[derive(Debug)]
struct PlannedSlot {
    slot: u32,
    generation: u64,
    writer: WriterId,
}

#[derive(Debug)]
enum PlannedStorage {
    /// Only touched holes are recorded for an already-bound parent.
    Existing { updates: Vec<PlannedSlot> },
    /// A newly bound parent's complete slot vector is prepared off-pool and
    /// moved into place during the infallible commit.
    New { slots: Vec<PageSlot> },
}

#[derive(Debug)]
struct PlannedParent {
    parent: u32,
    group: GroupId,
    next_generation: u64,
    occupied_after: u32,
    next_slot: usize,
    storage: PlannedStorage,
}

impl PlannedParent {
    fn existing(parent_id: u32, parent: &Parent, group: GroupId) -> Self {
        debug_assert_eq!(parent.binding, Some(group));
        debug_assert!(parent.partial_index.is_some());
        Self {
            parent: parent_id,
            group,
            next_generation: parent.next_generation,
            occupied_after: parent.occupied_slots,
            next_slot: 0,
            storage: PlannedStorage::Existing {
                updates: Vec::new(),
            },
        }
    }

    fn new(
        parent_id: u32,
        parent: &Parent,
        group: GroupId,
        packing: u32,
    ) -> Result<Self, CachePoolError> {
        debug_assert!(parent.binding.is_none());
        debug_assert_eq!(parent.occupied_slots, 0);
        debug_assert!(parent.slots.is_empty());
        debug_assert!(parent.partial_index.is_none());

        let mut slots = Vec::new();
        slots
            .try_reserve_exact(packing as usize)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        slots.resize(packing as usize, PageSlot::default());
        Ok(Self {
            parent: parent_id,
            group,
            next_generation: parent.next_generation,
            occupied_after: 0,
            next_slot: 0,
            storage: PlannedStorage::New { slots },
        })
    }

    fn is_full(&self, actual: &Parent) -> bool {
        self.occupied_after as usize == self.slot_count(actual)
    }

    fn slot_count(&self, actual: &Parent) -> usize {
        match &self.storage {
            PlannedStorage::Existing { .. } => actual.slots.len(),
            PlannedStorage::New { slots } => slots.len(),
        }
    }

    fn allocate_up_to(
        &mut self,
        actual: &Parent,
        pool_id: u64,
        writer: WriterId,
        requested: usize,
        pages: &mut Vec<PageRef>,
    ) -> Result<usize, CachePoolError> {
        debug_assert_eq!(self.group, actual.binding.unwrap_or(self.group));
        let available = self
            .slot_count(actual)
            .saturating_sub(self.occupied_after as usize);
        let target = requested.min(available);
        if target == 0 {
            return Ok(0);
        }

        if let PlannedStorage::Existing { updates } = &mut self.storage {
            updates
                .try_reserve(target)
                .map_err(|_| CachePoolError::MetadataAllocation)?;
        }

        let mut allocated = 0;
        while allocated != target {
            let slot_index = self.next_slot;
            let is_live = match &self.storage {
                PlannedStorage::Existing { .. } => actual.slots[slot_index].is_live(),
                PlannedStorage::New { slots } => slots[slot_index].is_live(),
            };
            self.next_slot += 1;
            if is_live {
                continue;
            }

            let generation = self.next_generation.checked_add(1).ok_or(
                CachePoolError::ParentGenerationExhausted {
                    parent: self.parent,
                },
            )?;
            self.next_generation = generation;
            let reserved = PageSlot::reserved_request(generation, writer);
            match &mut self.storage {
                PlannedStorage::Existing { updates } => updates.push(PlannedSlot {
                    slot: slot_index as u32,
                    generation,
                    writer,
                }),
                PlannedStorage::New { slots } => slots[slot_index] = reserved,
            }
            self.occupied_after += 1;
            allocated += 1;
            pages.push(PageRef {
                handle: PageHandle {
                    pool_id,
                    group: self.group,
                    parent: self.parent,
                    slot: slot_index as u32,
                    generation,
                },
            });
        }
        Ok(allocated)
    }
}

#[derive(Debug)]
struct GroupPlanState {
    /// Next entry in the pool's compact partial-parent list to inspect.  Each
    /// demanded group filters that list at most once; with D demanded groups
    /// and P partial parents, this part of admission is O(D * P).
    next_partial_index: usize,
    /// The one touched parent which still has room.  Packing always fills a
    /// parent before advancing, so one cursor is sufficient per group.
    current: Option<usize>,
}

#[derive(Debug)]
struct PlacementPlan {
    parents: Vec<PlannedParent>,
    group_states: Vec<Option<GroupPlanState>>,
    new_parent_ids: Vec<u32>,
    block_count: usize,
}

impl PlacementPlan {
    fn new(
        group_count: usize,
        total_blocks: usize,
        parent_capacity: usize,
    ) -> Result<Self, CachePoolError> {
        let touched_bound = total_blocks.min(parent_capacity);
        let mut parents = Vec::new();
        parents
            .try_reserve(touched_bound)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        let mut group_states = Vec::new();
        group_states
            .try_reserve_exact(group_count)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        group_states.resize_with(group_count, || None);
        let mut new_parent_ids = Vec::new();
        new_parent_ids
            .try_reserve(touched_bound)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        Ok(Self {
            parents,
            group_states,
            new_parent_ids,
            block_count: 0,
        })
    }
}

/// One flat pool of fungible, equal-sized physical parents.
#[derive(Debug)]
pub(crate) struct CachePool {
    id: u64,
    /// Last writer identity committed to a successful reservation. It is
    /// monotonic for this pool and never rewound by abort or retirement.
    last_writer_id: u64,
    group_packings: Box<[NonZeroU32]>,
    parents: Vec<Parent>,
    free_parents: VecDeque<u32>,
    /// Compact unordered membership. Removal uses `swap_remove`, so placement
    /// promises dense packing but deliberately no stable partial-parent
    /// priority. Capacity is fixed up front for allocation-free teardown.
    partial_parents: Vec<u32>,
    occupied_blocks: usize,
}

impl CachePool {
    pub(crate) fn new(
        parent_capacity: u32,
        group_packings: &[NonZeroU32],
    ) -> Result<Self, CachePoolError> {
        if parent_capacity == 0 {
            return Err(CachePoolError::InvalidParentCapacity);
        }
        if group_packings.is_empty() {
            return Err(CachePoolError::EmptyGroupLayout);
        }
        if group_packings.len() > u32::MAX as usize {
            return Err(CachePoolError::TooManyGroups);
        }
        for (index, packing) in group_packings.iter().enumerate() {
            let max_page_id = u64::from(parent_capacity) * u64::from(packing.get());
            if max_page_id > i32::MAX as u64 {
                return Err(CachePoolError::KernelPageIdOverflow {
                    group: GroupId::new(index as u32),
                });
            }
        }
        let id = NEXT_POOL_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| CachePoolError::PoolIdExhausted)?;

        let parent_capacity =
            usize::try_from(parent_capacity).map_err(|_| CachePoolError::MetadataAllocation)?;
        let mut packings = Vec::new();
        packings
            .try_reserve_exact(group_packings.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        packings.extend_from_slice(group_packings);

        let mut parents = Vec::new();
        parents
            .try_reserve_exact(parent_capacity)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        parents.resize_with(parent_capacity, Parent::default);

        let mut free_parents = VecDeque::new();
        free_parents
            .try_reserve_exact(parent_capacity)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        free_parents.extend(1..=parent_capacity as u32);

        let mut partial_parents = Vec::new();
        partial_parents
            .try_reserve_exact(parent_capacity)
            .map_err(|_| CachePoolError::MetadataAllocation)?;

        Ok(Self {
            id,
            last_writer_id: 0,
            group_packings: packings.into_boxed_slice(),
            parents,
            free_parents,
            partial_parents,
            occupied_blocks: 0,
        })
    }

    pub(crate) const fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn blocks_per_parent(&self, group: GroupId) -> Result<NonZeroU32, CachePoolError> {
        self.group_packings
            .get(group.index() as usize)
            .copied()
            .ok_or(CachePoolError::UnknownGroup { group })
    }

    pub(crate) fn free_parent_count(&self) -> usize {
        self.free_parents.len()
    }

    pub(crate) const fn occupied_block_count(&self) -> usize {
        self.occupied_blocks
    }

    #[cfg(test)]
    pub(crate) fn bound_group(&self, parent: u32) -> Option<GroupId> {
        self.parents
            .get(parent.checked_sub(1)? as usize)
            .and_then(|parent| parent.binding)
    }

    /// Reserve every ordered group demand for one request atomically. Logical
    /// state is unchanged on every error; output tables preserve input order
    /// and multiplicity.
    pub(crate) fn reserve_request(
        &mut self,
        sequence: SequenceId,
        demands: &[GroupDemand],
    ) -> Result<PendingRequestPages, CachePoolError> {
        let writer = WriterId(
            self.last_writer_id
                .checked_add(1)
                .ok_or(CachePoolError::WriterIdExhausted)?,
        );
        let mut total_blocks = 0usize;
        for demand in demands {
            self.blocks_per_parent(demand.group)?;
            total_blocks = total_blocks
                .checked_add(demand.blocks)
                .ok_or(CachePoolError::DemandOverflow)?;
        }
        self.occupied_blocks
            .checked_add(total_blocks)
            .ok_or(CachePoolError::DemandOverflow)?;

        let mut groups = Vec::new();
        groups
            .try_reserve_exact(demands.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        for demand in demands {
            let mut pages = Vec::new();
            pages
                .try_reserve_exact(demand.blocks)
                .map_err(|_| CachePoolError::MetadataAllocation)?;
            groups.push(GroupTable {
                group: demand.group,
                pages,
            });
        }

        let mut plan =
            PlacementPlan::new(self.group_packings.len(), total_blocks, self.parents.len())?;
        for (demand, group) in demands.iter().zip(&mut groups) {
            self.plan_partition(&mut plan, writer, *demand, &mut group.pages)?;
        }
        debug_assert_eq!(
            groups.iter().map(|group| group.pages.len()).sum::<usize>(),
            total_blocks
        );
        debug_assert_eq!(plan.block_count, total_blocks);

        // Every fallible operation completed in prepared metadata.
        // Publishing touches only planned parents and performs no allocation.
        self.commit_plan(plan);
        self.last_writer_id = writer.0;
        Ok(PendingRequestPages {
            pool_id: self.id,
            sequence,
            writer,
            groups,
        })
    }

    /// Make a fully reserved request executable. Private token provenance makes
    /// this transition infallible.
    pub(crate) fn promote_request(&mut self, pending: PendingRequestPages) -> RequestPages {
        self.assert_pending_request(&pending);
        for page in pending.groups.iter().flat_map(|group| &group.pages) {
            let slot = self.slot_mut(page.handle);
            debug_assert_eq!(
                slot.mode,
                Some(PageMode::Reserved {
                    writer: pending.writer
                })
            );
            debug_assert_eq!(slot.request_refs, 0);
            slot.mode = Some(PageMode::Mutable {
                writer: pending.writer,
            });
            slot.request_refs = 1;
        }
        RequestPages {
            pool_id: pending.pool_id,
            sequence: pending.sequence,
            writer: pending.writer,
            groups: pending.groups,
        }
    }

    /// Discard a fresh request reservation. Like promotion, this is infallible
    /// after successful reservation.
    pub(crate) fn abort_request(&mut self, pending: PendingRequestPages) {
        self.assert_pending_request(&pending);
        for group in pending.groups.iter().rev() {
            for page in group.pages.iter().rev() {
                let slot = self.slot_mut(page.handle);
                debug_assert_eq!(slot.request_refs, 0);
                slot.mode = None;
                self.retire_slot_if_unused(page.handle);
            }
        }
    }

    /// Retire one move-only request owner. Validation completes before the first
    /// mutation; failure returns the exact token unchanged. Active leases keep
    /// their pages physically live until runner completion or quarantine.
    pub(crate) fn retire_request(
        &mut self,
        request: RequestPages,
    ) -> Result<(), RetireRequestFailure> {
        if let Err(error) = self.validate_request(&request) {
            return Err(RetireRequestFailure { error, request });
        }
        for group in request.groups.iter().rev() {
            for page in group.pages.iter().rev() {
                let slot = self.slot_mut(page.handle);
                debug_assert_ne!(slot.request_refs, 0);
                slot.request_refs -= 1;
                self.retire_slot_if_unused(page.handle);
            }
        }
        Ok(())
    }

    /// Authenticate and canonicalize an exact set of mutable request pages
    /// before runner acceptance. No lease state changes here.
    pub(crate) fn prepare_writer_access(
        &self,
        mut claims: Vec<WriterAccessClaim>,
    ) -> Result<PreparedWriterAccess, CachePoolError> {
        for claim in &claims {
            self.validate_writer_claim(*claim)?;
        }
        claims.sort_unstable_by_key(|claim| claim.handle);
        if claims
            .windows(2)
            .any(|pair| pair[0].handle == pair[1].handle)
        {
            return Err(CachePoolError::DuplicateWriteClaim);
        }

        let mut prepared_claims = Vec::new();
        prepared_claims
            .try_reserve_exact(claims.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        for claim in &claims {
            let slot = self.validate_writer_available(*claim)?;
            prepared_claims.push(PreparedWriterClaim {
                claim: *claim,
                expected_write_epoch: slot.write_epoch,
            });
        }
        Ok(PreparedWriterAccess {
            pool_id: self.id,
            claims: prepared_claims,
        })
    }

    /// Acquire every prepared writer lease atomically. Revalidation catches an
    /// intervening retirement or competing lease without partially acquiring.
    pub(crate) fn acquire_writer_access(
        &mut self,
        prepared: PreparedWriterAccess,
    ) -> Result<WriterAccessLease, CachePoolError> {
        if prepared.pool_id != self.id {
            return Err(CachePoolError::ForeignAccess);
        }
        for prepared_claim in &prepared.claims {
            let claim = prepared_claim.claim;
            self.validate_writer_claim(claim)?;
            let slot = self.validate_writer_available(claim)?;
            if slot.write_epoch != prepared_claim.expected_write_epoch {
                return Err(CachePoolError::WriteEpochChanged {
                    expected: prepared_claim.expected_write_epoch,
                    actual: slot.write_epoch,
                });
            }
        }
        for prepared_claim in &prepared.claims {
            self.slot_mut(prepared_claim.claim.handle).lease = LeaseState::Writer;
        }
        Ok(WriterAccessLease {
            pool_id: prepared.pool_id,
            claims: prepared.claims,
        })
    }

    /// Release a known-live writer lease. This path allocates nothing. A caller
    /// which cannot prove runner completion retains the lease as quarantine.
    pub(crate) fn release_writer_access(&mut self, lease: WriterAccessLease) {
        assert_eq!(
            lease.pool_id, self.id,
            "writer access can only return to its issuing cache pool"
        );
        for prepared_claim in &lease.claims {
            let claim = prepared_claim.claim;
            let slot = self
                .validate_live_handle(claim.handle)
                .expect("writer lease keeps its page live");
            assert_eq!(slot.lease, LeaseState::Writer);
            assert_eq!(slot.write_epoch, prepared_claim.expected_write_epoch);
            assert_ne!(slot.write_epoch, u64::MAX);
        }
        for prepared_claim in lease.claims {
            let claim = prepared_claim.claim;
            let slot = self.slot_mut(claim.handle);
            slot.lease = LeaseState::Idle;
            slot.write_epoch += 1;
            self.retire_slot_if_unused(claim.handle);
        }
    }

    /// Translate a pool handle into the dense page id consumed by one group's
    /// device view.  Page zero remains the null page.
    pub(crate) fn kernel_page_id(&self, handle: PageHandle) -> Result<u32, CachePoolError> {
        self.validate_live_handle(handle)?;
        let packing = self.blocks_per_parent(handle.group)?.get();
        let page_id =
            1u64 + u64::from(handle.parent - 1) * u64::from(packing) + u64::from(handle.slot);
        u32::try_from(page_id).map_err(|_| CachePoolError::KernelPageIdOverflow {
            group: handle.group,
        })
    }

    fn plan_partition(
        &self,
        plan: &mut PlacementPlan,
        writer: WriterId,
        demand: GroupDemand,
        pages: &mut Vec<PageRef>,
    ) -> Result<(), CachePoolError> {
        let packing = self.blocks_per_parent(demand.group)?.get();
        let PlacementPlan {
            parents,
            group_states,
            new_parent_ids,
            block_count,
        } = plan;
        let state = group_states[demand.group.index() as usize].get_or_insert(GroupPlanState {
            next_partial_index: 0,
            current: None,
        });
        let mut remaining = demand.blocks;

        while remaining != 0 {
            if let Some(plan_index) = state.current {
                let parent_id = parents[plan_index].parent;
                let actual = &self.parents[(parent_id - 1) as usize];
                let allocated = parents[plan_index]
                    .allocate_up_to(actual, self.id, writer, remaining, pages)?;
                remaining -= allocated;
                *block_count += allocated;
                if parents[plan_index].is_full(actual) {
                    state.current = None;
                }
                continue;
            }

            let next_existing = loop {
                let Some(&parent_id) = self.partial_parents.get(state.next_partial_index) else {
                    break None;
                };
                state.next_partial_index += 1;
                let actual = &self.parents[(parent_id - 1) as usize];
                if actual.binding == Some(demand.group) {
                    break Some(parent_id);
                }
            };
            if let Some(parent_id) = next_existing {
                let actual = &self.parents[(parent_id - 1) as usize];
                debug_assert!(actual.partial_index.is_some());
                let plan_index = parents.len();
                parents.push(PlannedParent::existing(parent_id, actual, demand.group));
                state.current = Some(plan_index);
                continue;
            }

            let Some(parent_id) = self.free_parents.get(new_parent_ids.len()).copied() else {
                return Err(CachePoolError::InsufficientCapacity {
                    group: demand.group,
                    requested: demand.blocks,
                    available: pages.len(),
                });
            };
            let actual = &self.parents[(parent_id - 1) as usize];
            let planned = PlannedParent::new(parent_id, actual, demand.group, packing)?;
            let plan_index = parents.len();
            parents.push(planned);
            new_parent_ids.push(parent_id);
            state.current = Some(plan_index);
        }
        Ok(())
    }

    fn commit_plan(&mut self, plan: PlacementPlan) {
        let PlacementPlan {
            parents,
            group_states: _,
            new_parent_ids,
            block_count,
        } = plan;
        let mut expected_new_parents = new_parent_ids.into_iter();

        for planned in parents {
            let PlannedParent {
                parent: parent_id,
                group,
                next_generation,
                occupied_after,
                next_slot: _,
                storage,
            } = planned;
            let parent_index = (parent_id - 1) as usize;
            match storage {
                PlannedStorage::Existing { updates } => {
                    let became_full = {
                        let parent = &mut self.parents[parent_index];
                        debug_assert_eq!(parent.binding, Some(group));
                        debug_assert!(parent.partial_index.is_some());
                        for update in updates {
                            let slot = &mut parent.slots[update.slot as usize];
                            debug_assert!(!slot.is_live());
                            *slot = PageSlot::reserved_request(update.generation, update.writer);
                        }
                        parent.occupied_slots = occupied_after;
                        parent.next_generation = next_generation;
                        occupied_after as usize == parent.slots.len()
                    };
                    if became_full {
                        self.remove_partial(parent_id);
                    }
                }
                PlannedStorage::New { slots } => {
                    debug_assert_eq!(expected_new_parents.next(), Some(parent_id));
                    assert_eq!(
                        self.free_parents.pop_front(),
                        Some(parent_id),
                        "cache-pool free-parent order changed during reservation"
                    );
                    let is_partial = {
                        let parent = &mut self.parents[parent_index];
                        debug_assert!(parent.binding.is_none());
                        debug_assert_eq!(parent.occupied_slots, 0);
                        debug_assert!(parent.slots.is_empty());
                        debug_assert!(parent.partial_index.is_none());
                        parent.binding = Some(group);
                        parent.slots = slots;
                        parent.occupied_slots = occupied_after;
                        parent.next_generation = next_generation;
                        occupied_after as usize != parent.slots.len()
                    };
                    if is_partial {
                        self.insert_partial(parent_id);
                    }
                }
            }
        }
        debug_assert_eq!(expected_new_parents.next(), None);
        self.occupied_blocks += block_count;
    }

    /// Preflight a request retirement without changing ownership or placement.
    pub(crate) fn validate_request(&self, request: &RequestPages) -> Result<(), CachePoolError> {
        if request.pool_id != self.id {
            return Err(CachePoolError::ForeignPool);
        }
        if Self::has_duplicate_request_handle(request) {
            return Err(CachePoolError::DuplicateHandle);
        }
        for group in &request.groups {
            for page in &group.pages {
                if group.group != page.handle.group {
                    return Err(CachePoolError::StaleHandle);
                }
                let slot = self.validate_live_handle(page.handle)?;
                match slot.mode {
                    Some(PageMode::Mutable { writer }) => {
                        if writer != request.writer {
                            return Err(CachePoolError::WrongWriter);
                        }
                        if slot.request_refs != 1 {
                            return Err(CachePoolError::StaleHandle);
                        }
                    }
                    _ => return Err(CachePoolError::StaleHandle),
                }
            }
        }
        Ok(())
    }

    fn has_duplicate_request_handle(request: &RequestPages) -> bool {
        for (group_index, group) in request.groups.iter().enumerate() {
            for (page_index, page) in group.pages.iter().enumerate() {
                if group.pages[page_index + 1..]
                    .iter()
                    .any(|later| later.handle == page.handle)
                {
                    return true;
                }
                if request.groups[group_index + 1..]
                    .iter()
                    .any(|later| later.pages.iter().any(|entry| entry.handle == page.handle))
                {
                    return true;
                }
            }
        }
        false
    }

    fn validate_writer_claim(&self, claim: WriterAccessClaim) -> Result<(), CachePoolError> {
        if claim.pool_id != self.id {
            return Err(CachePoolError::ForeignAccess);
        }
        let slot = self.validate_live_handle(claim.handle)?;
        let Some(PageMode::Mutable { writer }) = slot.mode else {
            return Err(CachePoolError::PageNotWritable);
        };
        if writer != claim.writer {
            return Err(CachePoolError::WrongWriter);
        }
        if slot.request_refs != 1 {
            return Err(CachePoolError::PageNotWritable);
        }
        Ok(())
    }

    fn validate_writer_available(
        &self,
        claim: WriterAccessClaim,
    ) -> Result<&PageSlot, CachePoolError> {
        let slot = self.validate_live_handle(claim.handle)?;
        if slot.write_epoch == u64::MAX {
            return Err(CachePoolError::WriteEpochExhausted);
        }
        if slot.lease != LeaseState::Idle {
            return Err(CachePoolError::AccessConflict);
        }
        Ok(slot)
    }

    fn validate_live_handle(&self, handle: PageHandle) -> Result<&PageSlot, CachePoolError> {
        if handle.pool_id != self.id {
            return Err(CachePoolError::ForeignPool);
        }
        let parent = self
            .parents
            .get(
                handle
                    .parent
                    .checked_sub(1)
                    .ok_or(CachePoolError::StaleHandle)? as usize,
            )
            .ok_or(CachePoolError::StaleHandle)?;
        let bound_group = parent.binding.ok_or(CachePoolError::StaleHandle)?;
        if bound_group != handle.group {
            return Err(CachePoolError::ParentGroupMismatch {
                handle_group: handle.group,
                bound_group,
            });
        }
        let slot = parent
            .slots
            .get(handle.slot as usize)
            .ok_or(CachePoolError::StaleHandle)?;
        if !slot.is_live() || slot.generation != handle.generation {
            return Err(CachePoolError::StaleHandle);
        }
        Ok(slot)
    }

    fn assert_pending_request(&self, pending: &PendingRequestPages) {
        assert_eq!(
            pending.pool_id, self.id,
            "pending request pages belong to a different cache pool"
        );
        for group in &pending.groups {
            for page in &group.pages {
                assert_eq!(group.group, page.handle.group);
                let slot = self
                    .validate_live_handle(page.handle)
                    .expect("fresh pending request page remains live");
                assert_eq!(
                    slot.mode,
                    Some(PageMode::Reserved {
                        writer: pending.writer
                    })
                );
                assert_eq!(slot.request_refs, 0);
                assert_eq!(slot.lease, LeaseState::Idle);
                assert_eq!(slot.write_epoch, 0);
            }
        }
    }

    fn slot_mut(&mut self, handle: PageHandle) -> &mut PageSlot {
        &mut self.parents[(handle.parent - 1) as usize].slots[handle.slot as usize]
    }

    fn insert_partial(&mut self, parent_id: u32) {
        let parent_index = (parent_id - 1) as usize;
        {
            let parent = &self.parents[parent_index];
            debug_assert!(parent.binding.is_some());
            debug_assert_ne!(parent.occupied_slots, 0);
            debug_assert!((parent.occupied_slots as usize) < parent.slots.len());
            debug_assert!(parent.partial_index.is_none());
        }
        debug_assert!(self.partial_parents.len() < self.parents.len());
        let partial_index = self.partial_parents.len();
        self.partial_parents.push(parent_id);
        self.parents[parent_index].partial_index = Some(partial_index);
    }

    fn remove_partial(&mut self, parent_id: u32) {
        let parent_index = (parent_id - 1) as usize;
        let partial_index = self.parents[parent_index]
            .partial_index
            .expect("only a listed partial parent may be removed");
        let removed = self.partial_parents.swap_remove(partial_index);
        debug_assert_eq!(removed, parent_id);
        self.parents[parent_index].partial_index = None;
        if let Some(&moved_parent_id) = self.partial_parents.get(partial_index) {
            self.parents[(moved_parent_id - 1) as usize].partial_index = Some(partial_index);
        }
    }

    fn retire_slot_if_unused(&mut self, handle: PageHandle) {
        let parent_index = (handle.parent - 1) as usize;
        let (became_empty, became_partial, was_partial) = {
            let parent = &mut self.parents[parent_index];
            let slot = &mut parent.slots[handle.slot as usize];
            if slot.is_live() {
                return;
            }
            slot.clear_lifetime();
            let was_full = parent.occupied_slots as usize == parent.slots.len();
            parent.occupied_slots -= 1;
            (
                parent.occupied_slots == 0,
                was_full && parent.occupied_slots != 0,
                parent.partial_index.is_some(),
            )
        };
        self.occupied_blocks -= 1;

        if became_empty {
            if was_partial {
                self.remove_partial(handle.parent);
            }
            let parent = &mut self.parents[parent_index];
            parent.binding = None;
            parent.slots.clear();
            self.free_parents.push_back(handle.parent);
        } else if became_partial {
            debug_assert!(!was_partial);
            self.insert_partial(handle.parent);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }

    fn sequence(value: u64) -> SequenceId {
        SequenceId::new(value)
    }

    fn request_handles(request: &RequestPages) -> Vec<PageHandle> {
        request
            .groups()
            .iter()
            .flat_map(|group| group.pages().iter().map(PageRef::handle))
            .collect()
    }

    fn writer_claims(request: &RequestPages) -> Vec<WriterAccessClaim> {
        request
            .groups()
            .iter()
            .flat_map(|group| group.pages().iter().map(|page| request.write_claim(page)))
            .collect()
    }

    #[test]
    fn empty_parent_is_fungible_across_groups_and_generation_survives_rebinding() {
        let mut pool = CachePool::new(1, &[nz(2), nz(1)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first_writer = first.writer;
        let first_handle = first.groups()[0].pages()[0].handle();
        assert_eq!(pool.bound_group(1), Some(GroupId::new(0)));
        pool.abort_request(first);
        assert_eq!(pool.bound_group(1), None);

        let second = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(1), 1)])
            .unwrap();
        assert!(second.writer > first_writer);
        let second_handle = second.groups()[0].pages()[0].handle();
        assert_eq!(second_handle.parent(), first_handle.parent());
        assert!(second_handle.generation() > first_handle.generation());
        assert_eq!(pool.bound_group(1), Some(GroupId::new(1)));
        pool.abort_request(second);
    }

    #[test]
    fn group_packing_maps_children_to_dense_kernel_page_ids() {
        let mut pool = CachePool::new(2, &[nz(3)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 4)])
            .unwrap();
        let page_ids = pending.groups()[0]
            .pages()
            .iter()
            .map(|page| pool.kernel_page_id(page.handle()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(page_ids, vec![1, 2, 3, 4]);
        assert_eq!(pool.free_parent_count(), 0);
        assert_eq!(pool.occupied_block_count(), 4);
        pool.abort_request(pending);
    }

    #[test]
    fn multi_group_reservation_is_transactional_on_capacity_failure() {
        let mut pool = CachePool::new(2, &[nz(2), nz(1)]).unwrap();
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();
        let before_partial = pool.partial_parents.clone();
        let before_occupied = pool.occupied_blocks;

        let error = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(0), 3),
                    GroupDemand::new(GroupId::new(1), 1),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error,
            CachePoolError::InsufficientCapacity {
                group: GroupId::new(1),
                requested: 1,
                available: 0,
            }
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert_eq!(pool.partial_parents, before_partial);
        assert_eq!(pool.occupied_blocks, before_occupied);
    }

    #[test]
    fn duplicate_demands_share_placement_but_preserve_ordered_group_tables() {
        let mut pool = CachePool::new(2, &[nz(2), nz(1)]).unwrap();
        let pending = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(0), 1),
                    GroupDemand::new(GroupId::new(1), 1),
                    GroupDemand::new(GroupId::new(0), 1),
                ],
            )
            .unwrap();
        assert_eq!(pending.groups().len(), 3);
        assert_eq!(pending.groups()[0].group(), GroupId::new(0));
        assert_eq!(pending.groups()[1].group(), GroupId::new(1));
        assert_eq!(pending.groups()[2].group(), GroupId::new(0));
        let first = pending.groups()[0].pages()[0].handle();
        let third = pending.groups()[2].pages()[0].handle();
        assert_eq!(first.parent(), third.parent());
        assert_ne!(first.slot(), third.slot());
        assert_eq!(pool.free_parent_count(), 0);
        pool.abort_request(pending);
    }

    #[test]
    fn promotion_installs_exact_request_refs_and_retirement_reclaims_parent() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let pending = pool
            .reserve_request(sequence(7), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        assert_eq!(pending.sequence(), sequence(7));
        let request = pool.promote_request(pending);
        assert_eq!(request.sequence(), sequence(7));
        let writer = request.writer;
        for handle in request_handles(&request) {
            let slot = pool.validate_live_handle(handle).unwrap();
            assert_eq!(slot.mode, Some(PageMode::Mutable { writer }));
            assert_eq!(slot.request_refs, 1);
        }
        assert_eq!(pool.occupied_block_count(), 2);
        pool.retire_request(request).unwrap();
        assert_eq!(pool.occupied_block_count(), 0);
        assert_eq!(pool.free_parent_count(), 1);
    }

    #[test]
    fn terminal_return_paths_use_pre_reserved_parent_metadata() {
        let mut pool = CachePool::new(4, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 3)])
            .unwrap();
        let request = pool.promote_request(pending);

        // Admission does not replace or shrink either terminal-path buffer.
        let empty = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 0)])
            .unwrap();
        pool.abort_request(empty);
        assert!(pool.free_parents.capacity() >= pool.parents.len());
        assert!(pool.partial_parents.capacity() >= pool.parents.len());
        pool.retire_request(request).unwrap();
        assert_eq!(pool.free_parent_count(), 4);
    }

    #[test]
    fn partial_parent_swap_removal_updates_the_moved_parent_index() {
        let mut pool = CachePool::new(3, &[nz(2), nz(2), nz(2)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let middle = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(1), 1)])
            .unwrap();
        let middle = pool.promote_request(middle);
        let last = pool
            .reserve_request(sequence(3), &[GroupDemand::new(GroupId::new(2), 1)])
            .unwrap();
        let last = pool.promote_request(last);
        assert_eq!(pool.partial_parents, vec![1, 2, 3]);

        pool.retire_request(middle).unwrap();
        assert_eq!(pool.partial_parents, vec![1, 3]);
        assert_eq!(pool.parents[2].partial_index, Some(1));

        let fill_last = pool
            .reserve_request(sequence(4), &[GroupDemand::new(GroupId::new(2), 1)])
            .unwrap();
        assert_eq!(fill_last.groups()[0].pages()[0].handle().parent(), 3);
        assert_eq!(pool.partial_parents, vec![1]);
        assert_eq!(pool.parents[2].partial_index, None);
        pool.abort_request(fill_last);
        pool.retire_request(last).unwrap();
        pool.retire_request(first).unwrap();
        assert!(pool.partial_parents.is_empty());
    }

    #[test]
    fn generation_exhaustion_before_commit_leaves_the_pool_unchanged() {
        let mut pool = CachePool::new(2, &[nz(1), nz(1)]).unwrap();
        pool.parents[1].next_generation = u64::MAX;
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();

        let error = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(0), 1),
                    GroupDemand::new(GroupId::new(1), 1),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error,
            CachePoolError::ParentGenerationExhausted { parent: 2 }
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert!(pool.partial_parents.is_empty());
        assert_eq!(pool.occupied_block_count(), 0);
    }

    #[test]
    fn writer_id_exhaustion_leaves_the_pool_unchanged() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        pool.last_writer_id = u64::MAX;
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();

        assert_eq!(
            pool.reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
                .unwrap_err(),
            CachePoolError::WriterIdExhausted
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert!(pool.partial_parents.is_empty());
        assert_eq!(pool.occupied_block_count(), 0);
        assert_eq!(pool.last_writer_id, u64::MAX);
    }

    #[test]
    fn late_failure_rolls_back_planned_holes_in_an_existing_parent() {
        let mut pool = CachePool::new(2, &[nz(3), nz(1)]).unwrap();
        let seed = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let seed = pool.promote_request(seed);
        pool.parents[1].next_generation = u64::MAX;
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();
        let before_partial = pool.partial_parents.clone();

        let error = pool
            .reserve_request(
                sequence(2),
                &[
                    GroupDemand::new(GroupId::new(0), 1),
                    GroupDemand::new(GroupId::new(1), 1),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error,
            CachePoolError::ParentGenerationExhausted { parent: 2 }
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert_eq!(pool.partial_parents, before_partial);
        assert_eq!(pool.occupied_block_count(), 1);
        pool.retire_request(seed).unwrap();
    }

    #[test]
    fn same_slot_reuse_bumps_generation_while_its_parent_stays_bound() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let stale = first.groups()[0].pages()[0].handle();
        let sibling = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let sibling = pool.promote_request(sibling);
        pool.retire_request(first).unwrap();

        let replacement = pool
            .reserve_request(sequence(3), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let fresh = replacement.groups()[0].pages()[0].handle();
        assert_eq!(fresh.parent(), stale.parent());
        assert_eq!(fresh.slot(), stale.slot());
        assert!(fresh.generation() > stale.generation());
        assert_eq!(
            pool.kernel_page_id(stale).unwrap_err(),
            CachePoolError::StaleHandle
        );
        pool.abort_request(replacement);
        pool.retire_request(sibling).unwrap();
    }

    #[test]
    fn interleaved_pending_allocations_share_a_parent_in_either_finish_order() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let second = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        assert_eq!(
            first.groups()[0].pages()[0].handle().parent(),
            second.groups()[0].pages()[0].handle().parent()
        );
        pool.abort_request(first);
        let second = pool.promote_request(second);
        pool.retire_request(second).unwrap();
        assert_eq!(pool.free_parent_count(), 1);

        let first = pool
            .reserve_request(sequence(3), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let second = pool
            .reserve_request(sequence(4), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        pool.abort_request(second);
        pool.retire_request(first).unwrap();
        assert_eq!(pool.free_parent_count(), 1);
        assert!(pool.partial_parents.is_empty());
    }

    #[test]
    fn forged_duplicate_retirement_is_rejected_without_consuming_the_token() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let mut request = pool.promote_request(pending);
        let handle = request.groups[0].pages[0].handle;
        request.groups[0].pages.push(PageRef { handle });

        let mut failure = pool.retire_request(request).unwrap_err();
        assert_eq!(failure.error, CachePoolError::DuplicateHandle);
        assert_eq!(
            failure.request.groups[0]
                .pages
                .iter()
                .map(PageRef::handle)
                .collect::<Vec<_>>(),
            vec![handle, handle]
        );
        failure.request.groups[0].pages.pop();
        pool.retire_request(failure.request).unwrap();
    }

    #[test]
    fn failed_retirement_returns_the_unchanged_request_token() {
        let mut owner_pool = CachePool::new(1, &[nz(1)]).unwrap();
        let mut foreign_pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = owner_pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let request = owner_pool.promote_request(pending);
        let expected = request_handles(&request);
        let failure = foreign_pool.retire_request(request).unwrap_err();
        assert_eq!(failure.error, CachePoolError::ForeignPool);
        assert_eq!(request_handles(&failure.request), expected);
        owner_pool.retire_request(failure.request).unwrap();
    }

    #[test]
    fn stale_foreign_and_wrong_group_handles_are_rejected() {
        let mut pool = CachePool::new(1, &[nz(1), nz(1)]).unwrap();
        let foreign = CachePool::new(1, &[nz(1), nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let live = pending.groups()[0].pages()[0].handle();

        let mut foreign_handle = live;
        foreign_handle.pool_id = foreign.id();
        assert_eq!(
            foreign.kernel_page_id(live).unwrap_err(),
            CachePoolError::ForeignPool
        );
        assert_eq!(
            pool.kernel_page_id(foreign_handle).unwrap_err(),
            CachePoolError::ForeignPool
        );

        let mut wrong_group = live;
        wrong_group.group = GroupId::new(1);
        assert_eq!(
            pool.kernel_page_id(wrong_group).unwrap_err(),
            CachePoolError::ParentGroupMismatch {
                handle_group: GroupId::new(1),
                bound_group: GroupId::new(0),
            }
        );
        pool.abort_request(pending);

        let replacement = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        assert_eq!(
            pool.kernel_page_id(live).unwrap_err(),
            CachePoolError::StaleHandle
        );
        pool.abort_request(replacement);
    }

    #[test]
    fn zero_block_demands_still_preserve_output_order() {
        let mut pool = CachePool::new(1, &[nz(1), nz(2)]).unwrap();
        let pending = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(1), 0),
                    GroupDemand::new(GroupId::new(0), 0),
                ],
            )
            .unwrap();
        assert_eq!(pending.groups().len(), 2);
        assert_eq!(pending.groups()[0].group(), GroupId::new(1));
        assert_eq!(pending.groups()[1].group(), GroupId::new(0));
        assert!(pending.is_empty());
        pool.abort_request(pending);
        assert_eq!(pool.free_parent_count(), 1);
    }

    #[test]
    fn pending_request_pages_are_not_writer_accessible_before_promotion() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(9), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let page = &pending.groups()[0].pages()[0];
        let claim = WriterAccessClaim {
            pool_id: pool.id(),
            writer: pending.writer,
            handle: page.handle(),
        };
        assert_eq!(
            pool.prepare_writer_access(vec![claim]).unwrap_err(),
            CachePoolError::PageNotWritable
        );
        pool.abort_request(pending);
    }

    #[test]
    fn duplicate_sequence_id_cannot_substitute_request_writer() {
        let mut pool = CachePool::new(2, &[nz(1), nz(1)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let second = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(1), 1)])
            .unwrap();
        let second = pool.promote_request(second);
        assert_eq!(first.sequence(), second.sequence());
        assert_ne!(first.writer, second.writer);
        let claims = vec![
            first.write_claim(&first.groups()[0].pages()[0]),
            first.write_claim(&second.groups()[0].pages()[0]),
        ];
        assert_eq!(
            pool.prepare_writer_access(claims).unwrap_err(),
            CachePoolError::WrongWriter
        );
        for handle in request_handles(&first)
            .into_iter()
            .chain(request_handles(&second))
        {
            assert_eq!(
                pool.validate_live_handle(handle).unwrap().lease,
                LeaseState::Idle
            );
        }
        pool.retire_request(first).unwrap();
        pool.retire_request(second).unwrap();
    }

    #[test]
    fn duplicate_and_competing_writer_leases_are_rejected_atomically() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let request = pool.promote_request(pending);
        let first_claim = request.write_claim(&request.groups()[0].pages()[0]);
        assert_eq!(
            pool.prepare_writer_access(vec![first_claim, first_claim])
                .unwrap_err(),
            CachePoolError::DuplicateWriteClaim
        );

        let first = pool
            .acquire_writer_access(pool.prepare_writer_access(vec![first_claim]).unwrap())
            .unwrap();
        let all = writer_claims(&request);
        assert_eq!(
            pool.prepare_writer_access(all).unwrap_err(),
            CachePoolError::AccessConflict
        );
        let second_handle = request.groups()[0].pages()[1].handle();
        assert_eq!(
            pool.validate_live_handle(second_handle).unwrap().lease,
            LeaseState::Idle
        );
        pool.release_writer_access(first);
        pool.retire_request(request).unwrap();
    }

    #[test]
    fn prepared_multi_page_acquisition_revalidates_without_partial_leasing() {
        let mut pool = CachePool::new(2, &[nz(1)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let second = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let second = pool.promote_request(second);
        let first_handle = first.groups()[0].pages()[0].handle();
        let prepared = pool
            .prepare_writer_access(vec![
                first.write_claim(&first.groups()[0].pages()[0]),
                second.write_claim(&second.groups()[0].pages()[0]),
            ])
            .unwrap();
        pool.retire_request(second).unwrap();
        assert_eq!(
            pool.acquire_writer_access(prepared).unwrap_err(),
            CachePoolError::StaleHandle
        );
        assert_eq!(
            pool.validate_live_handle(first_handle).unwrap().lease,
            LeaseState::Idle
        );
        pool.retire_request(first).unwrap();
    }

    #[test]
    fn writer_release_advances_epoch_and_epoch_exhaustion_is_non_mutating() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let request = pool.promote_request(pending);
        let claims = writer_claims(&request);
        let lease = pool
            .acquire_writer_access(pool.prepare_writer_access(claims).unwrap())
            .unwrap();
        assert_eq!(lease.len(), 2);
        pool.release_writer_access(lease);
        let handles = request_handles(&request);
        assert!(
            handles
                .iter()
                .all(|handle| pool.validate_live_handle(*handle).unwrap().write_epoch == 1)
        );

        pool.slot_mut(handles[1]).write_epoch = u64::MAX;
        assert_eq!(
            pool.prepare_writer_access(writer_claims(&request))
                .unwrap_err(),
            CachePoolError::WriteEpochExhausted
        );
        assert!(handles
            .iter()
            .all(|handle| pool.validate_live_handle(*handle).unwrap().lease == LeaseState::Idle));
        pool.slot_mut(handles[1]).write_epoch = 1;
        pool.retire_request(request).unwrap();
    }

    #[test]
    fn prepared_writer_access_rejects_an_intervening_completed_write() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let request = pool.promote_request(pending);
        let claim = request.write_claim(&request.groups()[0].pages()[0]);
        let stale = pool.prepare_writer_access(vec![claim]).unwrap();
        let newer = pool.prepare_writer_access(vec![claim]).unwrap();

        let newer = pool.acquire_writer_access(newer).unwrap();
        pool.release_writer_access(newer);
        assert_eq!(
            pool.acquire_writer_access(stale).unwrap_err(),
            CachePoolError::WriteEpochChanged {
                expected: 0,
                actual: 1,
            }
        );
        assert_eq!(
            pool.validate_live_handle(claim.handle).unwrap().lease,
            LeaseState::Idle
        );
        pool.retire_request(request).unwrap();
    }

    #[test]
    fn active_writer_lease_is_the_quarantine_lifetime_after_request_retirement() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let request = pool.promote_request(pending);
        let handle = request.groups()[0].pages()[0].handle();
        let lease = pool
            .acquire_writer_access(pool.prepare_writer_access(writer_claims(&request)).unwrap())
            .unwrap();
        pool.retire_request(request).unwrap();
        let slot = pool.validate_live_handle(handle).unwrap();
        assert_eq!(slot.request_refs, 0);
        assert_eq!(slot.lease, LeaseState::Writer);
        assert_eq!(pool.occupied_block_count(), 1);
        assert_eq!(
            pool.reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)],)
                .unwrap_err(),
            CachePoolError::InsufficientCapacity {
                group: GroupId::new(0),
                requested: 1,
                available: 0,
            }
        );

        pool.release_writer_access(lease);
        assert_eq!(pool.occupied_block_count(), 0);
        let replacement = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let fresh = replacement.groups()[0].pages()[0].handle();
        assert!(fresh.generation() > handle.generation());
        pool.abort_request(replacement);
    }
}
