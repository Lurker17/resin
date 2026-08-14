use std::cmp::Reverse;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

mod cache;
mod planning;
mod runner;
mod state_pool;

use cache::{CacheGroupLayout, CacheGroupPlan, CacheLayout};
use planning::{build_plan_for_engine, schedule_ready_prefix};
use resin_core::resources::paged::{CachePoolError, RequestPages, WriterAccessLease};
use resin_core::resources::paged_kv::{PagedKvError, PagedKvResources};
use resin_core::resources::slots::SlotHandle as CoreStateHandle;
use resin_core::resources::{RequestResourceRef, ResourceSet, WithResources};
use resin_core::{Phase, RequestId, Schedulable, Scheduler, SequenceStep};
use runner::{
    RunnerExecution, bind_runner_resources, resolve_runner_contract,
    resolve_runner_max_batch_tokens, resolve_runner_prefill_chunk_size, resolve_runner_resources,
    runner_error_is_post_accept, runner_supports_continuation_prefill,
};
use state_pool::{
    PreparedStateAccessLease, StateAccessClaim, StateAccessLease, StateAccessMode, StateHandle,
    StatePool,
};

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);

fn mint_engine_id() -> PyResult<u64> {
    NEXT_ENGINE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| PyRuntimeError::new_err("engine id space is exhausted"))
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct SchedulerConfig {
    max_batch_tokens: Option<NonZeroU32>,
    prefill_chunk_size: NonZeroU32,
    schedule_conservativeness: f64,
}

const DEFAULT_PREFILL_CHUNK_SIZE: NonZeroU32 = NonZeroU32::new(8192).unwrap();
const INITIAL_NEW_TOKEN_RATIO: f64 = 0.7;
const MIN_NEW_TOKEN_RATIO_FACTOR: f64 = 0.14;
const NEW_TOKEN_RATIO_DECAY_STEPS: f64 = 600.0;
const RETRACTION_DECODE_HEADROOM: usize = 20;

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: None,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
            schedule_conservativeness: 1.0,
        }
    }
}

#[pymethods]
impl SchedulerConfig {
    #[new]
    #[pyo3(signature = (
        *,
        max_batch_tokens = None,
        prefill_chunk_size = 8192,
        schedule_conservativeness = 1.0
    ))]
    fn new(
        max_batch_tokens: Option<u32>,
        prefill_chunk_size: u32,
        schedule_conservativeness: f64,
    ) -> PyResult<Self> {
        if !schedule_conservativeness.is_finite() || schedule_conservativeness < 0.0 {
            return Err(PyValueError::new_err(
                "schedule_conservativeness must be finite and non-negative",
            ));
        }
        Ok(Self {
            max_batch_tokens: max_batch_tokens
                .map(|value| {
                    NonZeroU32::new(value)
                        .ok_or_else(|| PyValueError::new_err("max_batch_tokens must be positive"))
                })
                .transpose()?,
            prefill_chunk_size: NonZeroU32::new(prefill_chunk_size)
                .ok_or_else(|| PyValueError::new_err("prefill_chunk_size must be positive"))?,
            schedule_conservativeness,
        })
    }

    #[getter]
    fn max_batch_tokens(&self) -> Option<u32> {
        self.max_batch_tokens.map(NonZeroU32::get)
    }

    #[getter]
    fn prefill_chunk_size(&self) -> u32 {
        self.prefill_chunk_size.get()
    }

    #[getter]
    fn schedule_conservativeness(&self) -> f64 {
        self.schedule_conservativeness
    }

    fn __repr__(&self) -> String {
        format!(
            "SchedulerConfig(max_batch_tokens={}, prefill_chunk_size={}, schedule_conservativeness={:?})",
            self.max_batch_tokens()
                .map_or_else(|| "None".to_owned(), |value| value.to_string()),
            self.prefill_chunk_size(),
            self.schedule_conservativeness,
        )
    }
}

#[pyclass(
    frozen,
    eq,
    hash,
    rename_all = "SCREAMING_SNAKE_CASE",
    from_py_object,
    module = "resin._resin"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum BatchPhase {
    Prefill,
    Decode,
}

impl BatchPhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "PREFILL",
            Self::Decode => "DECODE",
        }
    }
}

impl SchedulerConfig {
    fn with_runner_limits(
        self,
        runner_prefill_limit: Option<NonZeroU32>,
        runner_batch_limit: Option<NonZeroU32>,
    ) -> Self {
        let prefill_chunk_size = runner_prefill_limit.map_or(self.prefill_chunk_size, |limit| {
            self.prefill_chunk_size.min(limit)
        });
        let max_batch_tokens = match (self.max_batch_tokens, runner_batch_limit) {
            (Some(configured), Some(limit)) => Some(configured.min(limit)),
            (configured, limit) => configured.or(limit),
        };
        Self {
            max_batch_tokens,
            prefill_chunk_size,
            ..self
        }
    }

    fn initial_new_token_ratio(self) -> f64 {
        (INITIAL_NEW_TOKEN_RATIO * self.schedule_conservativeness).min(1.0)
    }
}

/// Frozen runner-declared recurrent-state ABI.
///
/// `access_modes` is the exact, ordered handle pattern for every submitted
/// sequence. The protocol is deliberately opaque to the native control plane:
/// runners own its layout and interpretation while Rust owns admission,
/// scheduling, and lifetime enforcement.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct StateMappingConfig {
    protocol: String,
    access_modes: Vec<StateAccessMode>,
}

#[pymethods]
impl StateMappingConfig {
    #[new]
    fn new(protocol: String, access_modes: Vec<String>) -> PyResult<Self> {
        if protocol.trim().is_empty() {
            return Err(PyValueError::new_err("protocol must be nonempty"));
        }
        if access_modes.is_empty() {
            return Err(PyValueError::new_err(
                "access_modes must contain at least one mode",
            ));
        }
        let handles_per_sequence = access_modes.len();
        let access_modes = parse_state_access_modes(Some(access_modes), handles_per_sequence)?;
        Ok(Self {
            protocol,
            access_modes,
        })
    }

    #[getter]
    fn protocol(&self) -> &str {
        &self.protocol
    }

    #[getter]
    fn access_modes(&self) -> Vec<&'static str> {
        self.access_modes.iter().map(|mode| mode.as_str()).collect()
    }

    #[getter]
    fn handles_per_sequence(&self) -> usize {
        self.access_modes.len()
    }

    fn __repr__(&self) -> String {
        let access_modes = self
            .access_modes
            .iter()
            .map(|mode| format!("'{}'", mode.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "StateMappingConfig(protocol={:?}, access_modes=[{}])",
            self.protocol, access_modes
        )
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct BatchPlan {
    batch_id: u64,
    batch_key: Option<u64>,
    engine_id: Option<u64>,
    phase: BatchPhase,
    token_ids: Vec<u32>,
    positions: Vec<u32>,
    sequence_ids: Vec<u64>,
    sequence_indptr: Vec<u32>,
    context_lengths: Vec<u32>,
    // Rust event/commit mask: runner results remain dense at one token per row,
    // while rows omitted here advance prefill state without sampling a token.
    sample_sequence_indices: Vec<u32>,
    state_slots: Vec<u32>,
    state_generations: Vec<u64>,
    state_access_modes: Vec<StateAccessMode>,
    state_indptr: Vec<u32>,
    state_pool_id: Option<u64>,
    cache_pool_id: Option<u64>,
    cache_groups: Vec<CacheGroupPlan>,
}

fn validate_sequence_indices(name: &str, indices: &[u32], sequence_count: usize) -> PyResult<()> {
    if indices
        .iter()
        .any(|sequence_index| *sequence_index as usize >= sequence_count)
    {
        return Err(PyValueError::new_err(format!(
            "{name} contains an index outside the batch sequence range"
        )));
    }
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(PyValueError::new_err(format!(
            "{name} must be strictly sorted and unique"
        )));
    }
    Ok(())
}

fn parse_state_access_modes(
    modes: Option<Vec<String>>,
    handle_count: usize,
) -> PyResult<Vec<StateAccessMode>> {
    let Some(modes) = modes else {
        return Ok(vec![StateAccessMode::Write; handle_count]);
    };
    if modes.len() != handle_count {
        return Err(PyValueError::new_err(
            "state_access_modes must contain one item per state handle",
        ));
    }
    modes
        .into_iter()
        .map(|mode| match mode.as_str() {
            "read" => Ok(StateAccessMode::Read),
            "write" => Ok(StateAccessMode::Write),
            _ => Err(PyValueError::new_err(
                "state_access_modes entries must be 'read' or 'write'",
            )),
        })
        .collect()
}

impl BatchPlan {
    fn checked(plan: Self) -> PyResult<Self> {
        let Self {
            engine_id,
            token_ids,
            positions,
            sequence_ids,
            sequence_indptr,
            context_lengths,
            sample_sequence_indices,
            state_slots,
            state_generations,
            state_access_modes,
            state_indptr,
            state_pool_id,
            ..
        } = &plan;
        if engine_id == &Some(0) {
            return Err(PyValueError::new_err(
                "engine_id must be positive when supplied",
            ));
        }
        if token_ids.is_empty() {
            return Err(PyValueError::new_err(
                "a batch must contain at least one token",
            ));
        }
        if token_ids.len() > u32::MAX as usize {
            return Err(PyValueError::new_err(
                "batch token count exceeds the u32 metadata range",
            ));
        }
        if token_ids.len() != positions.len() {
            return Err(PyValueError::new_err(
                "token_ids and positions must have equal length",
            ));
        }
        if sequence_ids.is_empty() {
            return Err(PyValueError::new_err(
                "a batch must contain at least one sequence",
            ));
        }
        if sequence_indptr.len() != sequence_ids.len() + 1 {
            return Err(PyValueError::new_err(
                "sequence_indptr must contain one more item than sequence_ids",
            ));
        }
        if context_lengths.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "context_lengths must contain one item per sequence",
            ));
        }
        validate_sequence_indices(
            "sample_sequence_indices",
            sample_sequence_indices,
            sequence_ids.len(),
        )?;
        if state_slots.len() != state_generations.len() {
            return Err(PyValueError::new_err(
                "state_slots and state_generations must have equal length",
            ));
        }
        if state_slots.len() != state_access_modes.len() {
            return Err(PyValueError::new_err(
                "state_slots and state_access_modes must have equal length",
            ));
        }
        if state_indptr.len() != sequence_ids.len() + 1 {
            return Err(PyValueError::new_err(
                "state_indptr must contain one more item than sequence_ids",
            ));
        }
        if state_indptr.first() != Some(&0) {
            return Err(PyValueError::new_err("state_indptr must start at zero"));
        }
        if state_indptr.last().copied() != Some(state_slots.len() as u32) {
            return Err(PyValueError::new_err(
                "state_indptr must end at the state handle count",
            ));
        }
        if state_indptr.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(PyValueError::new_err(
                "state_indptr must be monotonically nondecreasing",
            ));
        }
        match (state_slots.is_empty(), state_pool_id) {
            (true, None) => {}
            (true, Some(_)) => {
                return Err(PyValueError::new_err(
                    "state_pool_id must be None when state metadata is empty",
                ));
            }
            (false, Some(pool_id)) if *pool_id != 0 => {}
            (false, Some(_)) => {
                return Err(PyValueError::new_err(
                    "state_pool_id must be positive when state metadata is present",
                ));
            }
            (false, None) => {
                return Err(PyValueError::new_err(
                    "state_pool_id is required when state metadata is present",
                ));
            }
        }
        if sequence_indptr.first() != Some(&0) {
            return Err(PyValueError::new_err("sequence_indptr must start at zero"));
        }
        if sequence_indptr.last().copied() != Some(token_ids.len() as u32) {
            return Err(PyValueError::new_err(
                "sequence_indptr must end at the token count",
            ));
        }
        if sequence_indptr.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(
                "each sequence must contain at least one token",
            ));
        }
        let mut sorted_ids = sequence_ids.clone();
        sorted_ids.sort_unstable();
        if sorted_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(PyValueError::new_err(
                "sequence ids must be unique within a batch",
            ));
        }

        plan.validate_cache()?;

        Ok(plan)
    }

    fn validate_cache(&self) -> PyResult<()> {
        match (self.cache_pool_id, self.cache_groups.is_empty()) {
            (None, true) => return Ok(()),
            (Some(pool_id), false) => {
                if pool_id == 0 {
                    return Err(PyValueError::new_err(
                        "cache_pool_id must be positive when cache group metadata is present",
                    ));
                }
            }
            (Some(_), true) => {
                return Err(PyValueError::new_err(
                    "cache_groups must be nonempty when cache_pool_id is present",
                ));
            }
            (None, false) => {
                return Err(PyValueError::new_err(
                    "cache_pool_id is required when cache group metadata is present",
                ));
            }
        }

        for (group_index, group) in self.cache_groups.iter().enumerate() {
            group.validate_local()?;
            let group = group.inner_native();
            if group.group_index != group_index as u32 {
                return Err(PyValueError::new_err(
                    "cache_groups must be ordered by contiguous group_index",
                ));
            }
            if group.first_logical_blocks.len() != self.sequence_ids.len() {
                return Err(PyValueError::new_err(
                    "each cache group must contain one block-table row per sequence",
                ));
            }
            if group.append_block_ids.len() != self.token_ids.len() {
                return Err(PyValueError::new_err(
                    "each cache group must contain one append target per batch token",
                ));
            }
            for sequence_index in 0..self.sequence_ids.len() {
                let block_start = group.block_indptr[sequence_index] as usize;
                let block_end = group.block_indptr[sequence_index + 1] as usize;
                let row_blocks = &group.block_ids[block_start..block_end];
                let token_start = self.sequence_indptr[sequence_index] as usize;
                let token_end = self.sequence_indptr[sequence_index + 1] as usize;
                if group.append_block_ids[token_start..token_end]
                    .iter()
                    .any(|page_id| !row_blocks.contains(page_id))
                {
                    return Err(PyValueError::new_err(
                        "cache append targets must belong to their sequence block-table row",
                    ));
                }
            }
        }
        Ok(())
    }
}
#[pymethods]
impl BatchPlan {
    #[new]
    #[pyo3(signature = (
        token_ids,
        positions,
        sequence_ids,
        sequence_indptr,
        context_lengths,
        *,
        phase = BatchPhase::Prefill,
        batch_id = 0,
        batch_key = None,
        engine_id = None,
        state_pool_id = None,
        state_slots = None,
        state_generations = None,
        state_indptr = None,
        state_access_modes = None,
        cache_pool_id = None,
        cache_groups = None,
        sample_sequence_indices = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        token_ids: Vec<u32>,
        positions: Vec<u32>,
        sequence_ids: Vec<u64>,
        sequence_indptr: Vec<u32>,
        context_lengths: Vec<u32>,
        phase: BatchPhase,
        batch_id: u64,
        batch_key: Option<u64>,
        engine_id: Option<u64>,
        state_pool_id: Option<u64>,
        state_slots: Option<Vec<u32>>,
        state_generations: Option<Vec<u64>>,
        state_indptr: Option<Vec<u32>>,
        state_access_modes: Option<Vec<String>>,
        cache_pool_id: Option<u64>,
        cache_groups: Option<Vec<Py<CacheGroupPlan>>>,
        sample_sequence_indices: Option<Vec<u32>>,
    ) -> PyResult<Self> {
        let sequence_count = sequence_ids.len();
        let (state_slots, state_generations, state_indptr) = match (
            state_slots,
            state_generations,
            state_indptr,
        ) {
            (None, None, None) => (
                Vec::new(),
                Vec::new(),
                vec![0; sequence_count.saturating_add(1)],
            ),
            (Some(slots), Some(generations), Some(indptr)) => (slots, generations, indptr),
            _ => {
                return Err(PyValueError::new_err(
                    "state_slots, state_generations, and state_indptr must be supplied together",
                ));
            }
        };
        let state_access_modes = parse_state_access_modes(state_access_modes, state_slots.len())?;
        let cache_groups = cache_groups
            .unwrap_or_default()
            .into_iter()
            .map(|group| group.borrow(py).clone())
            .collect();
        let sample_sequence_indices = sample_sequence_indices
            .unwrap_or_else(|| (0..sequence_count as u32).collect::<Vec<_>>());
        Self::checked(Self {
            batch_id,
            batch_key,
            engine_id,
            phase,
            token_ids,
            positions,
            sequence_ids,
            sequence_indptr,
            context_lengths,
            sample_sequence_indices,
            state_slots,
            state_generations,
            state_access_modes,
            state_indptr,
            state_pool_id,
            cache_pool_id,
            cache_groups,
        })
    }

    #[getter]
    fn batch_id(&self) -> u64 {
        self.batch_id
    }

    #[getter]
    fn batch_key(&self) -> Option<u64> {
        self.batch_key
    }

    #[getter]
    fn engine_id(&self) -> Option<u64> {
        self.engine_id
    }

    #[getter]
    fn phase(&self) -> BatchPhase {
        self.phase
    }

    #[getter]
    fn token_ids(&self) -> Vec<u32> {
        self.token_ids.clone()
    }

    #[getter]
    fn positions(&self) -> Vec<u32> {
        self.positions.clone()
    }

    #[getter]
    fn sequence_ids(&self) -> Vec<u64> {
        self.sequence_ids.clone()
    }

    #[getter]
    fn sequence_indptr(&self) -> Vec<u32> {
        self.sequence_indptr.clone()
    }

    #[getter]
    fn context_lengths(&self) -> Vec<u32> {
        self.context_lengths.clone()
    }

    #[getter]
    fn sample_sequence_indices(&self) -> Vec<u32> {
        self.sample_sequence_indices.clone()
    }

    #[getter]
    fn state_slots(&self) -> Vec<u32> {
        self.state_slots.clone()
    }

    #[getter]
    fn state_generations(&self) -> Vec<u64> {
        self.state_generations.clone()
    }

    #[getter]
    fn state_access_modes(&self) -> Vec<&'static str> {
        self.state_access_modes
            .iter()
            .map(|mode| mode.as_str())
            .collect()
    }

    #[getter]
    fn state_indptr(&self) -> Vec<u32> {
        self.state_indptr.clone()
    }

    #[getter]
    fn state_pool_id(&self) -> Option<u64> {
        self.state_pool_id
    }

    #[getter]
    fn cache_pool_id(&self) -> Option<u64> {
        self.cache_pool_id
    }

    #[getter]
    fn cache_groups(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let groups = self
            .cache_groups
            .iter()
            .cloned()
            .map(|group| Py::new(py, group))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, groups)?.unbind())
    }

    #[getter]
    fn num_sequences(&self) -> usize {
        self.sequence_ids.len()
    }

    fn token_range(&self, sequence_index: usize) -> PyResult<(usize, usize)> {
        if sequence_index >= self.sequence_ids.len() {
            return Err(PyValueError::new_err("sequence index is out of range"));
        }
        Ok((
            self.sequence_indptr[sequence_index] as usize,
            self.sequence_indptr[sequence_index + 1] as usize,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchPlan(batch_id={}, phase=BatchPhase.{}, sequences={}, tokens={}, states={}, cache_groups={})",
            self.batch_id,
            self.phase.as_str(),
            self.sequence_ids.len(),
            self.token_ids.len(),
            self.state_slots.len(),
            self.cache_groups.len()
        )
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct TokenEvent {
    request_id: u64,
    token_id: u32,
    index: usize,
    finished: bool,
}

#[derive(Debug, Default)]
struct RequestLeaseState {
    done: bool,
    cancelled: bool,
    result: Option<Vec<u32>>,
}

/// A caller-owned lifetime handle for one submitted request.
///
/// The Engine keeps the shared state only while the request is active. Once
/// the request retires, its generated tokens are moved into this lease. If
/// the caller drops the lease without taking the result, the token buffer is
/// released when the last internal reference disappears.
#[pyclass(module = "resin._resin")]
struct RequestLease {
    engine_id: u64,
    request_id: u64,
    state: Arc<Mutex<RequestLeaseState>>,
}

#[pymethods]
impl RequestLease {
    #[getter]
    fn id(&self) -> u64 {
        self.request_id
    }

    #[getter]
    fn request_id(&self) -> u64 {
        self.request_id
    }

    #[getter]
    fn done(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .done
    }

    #[getter]
    fn cancelled(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancelled
    }

    /// Consume the completed token buffer exactly once.
    fn take_result(&self) -> PyResult<Vec<u32>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.done {
            return Err(PyRuntimeError::new_err("request has not completed yet"));
        }
        state
            .result
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("request result has already been consumed"))
    }

    fn __int__(&self) -> u64 {
        self.request_id
    }

    fn __index__(&self) -> u64 {
        self.request_id
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyRef<'_, RequestLease>>() {
            return Ok(self.request_id == other.request_id);
        }
        Ok(other.extract::<u64>().is_ok_and(|id| id == self.request_id))
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestLease(id={}, done={}, cancelled={})",
            self.request_id,
            self.done(),
            self.cancelled()
        )
    }
}

fn complete_request_lease(
    state: &Arc<Mutex<RequestLeaseState>>,
    result: Vec<u32>,
    cancelled: bool,
) {
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.done = true;
    state.cancelled = cancelled;
    state.result = Some(result);
}

#[pymethods]
impl TokenEvent {
    #[getter]
    fn request_id(&self) -> u64 {
        self.request_id
    }

    #[getter]
    fn token_id(&self) -> u32 {
        self.token_id
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn finished(&self) -> bool {
        self.finished
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenEvent(request_id={}, token_id={}, index={}, finished={})",
            self.request_id, self.token_id, self.index, self.finished
        )
    }
}

#[derive(Debug)]
struct RequestData {
    id: u64,
    lease: Arc<Mutex<RequestLeaseState>>,
    priority: i32,
    batch_key: Option<u64>,
    turn_ordinal: u128,
    running: bool,
    prompt: Vec<u32>,
    resident_tokens: u32,
    generated: Vec<u32>,
    max_new_tokens: usize,
    cancel_pending: bool,
}

#[derive(Debug)]
struct RecurrentRequestState {
    handles: Vec<CoreStateHandle>,
    access_modes: Vec<StateAccessMode>,
}

type RequestResources = ResourceSet<Option<RequestPages>, RecurrentRequestState>;
type Request = WithResources<RequestData, RequestResources>;

impl RequestData {
    fn prompt_length(&self) -> u32 {
        u32::try_from(self.prompt.len()).expect("submit bounds prompt length to u32")
    }

    fn prefill_length(&self) -> PyResult<u32> {
        let generated_inputs =
            u32::try_from(self.generated.len().saturating_sub(1)).map_err(|_| {
                PyValueError::new_err("generated sequence length exceeds the u32 metadata range")
            })?;
        self.prompt_length()
            .checked_add(generated_inputs)
            .ok_or_else(|| {
                PyValueError::new_err("sequence position exceeds the u32 metadata range")
            })
    }

    fn logical_context(&self) -> PyResult<u32> {
        let prefill_length = self.prefill_length()?;
        if self.resident_tokens > prefill_length
            || (self.resident_tokens == prefill_length && self.generated.is_empty())
        {
            return Err(PyRuntimeError::new_err(format!(
                "request {} has invalid resident progress {} for input length {prefill_length}",
                self.id, self.resident_tokens
            )));
        }
        Ok(self.resident_tokens)
    }

    fn sample_at_prefill_end(&self) -> bool {
        self.generated.is_empty()
    }
}

impl Schedulable for RequestData {
    fn priority(&self) -> i32 {
        self.priority
    }

    fn batch_key(&self) -> Option<u64> {
        self.batch_key
    }

    fn phase(&self) -> Phase {
        let input_length = self.prompt.len() + self.generated.len().saturating_sub(1);
        if (self.resident_tokens as usize) < input_length {
            Phase::Prefill
        } else {
            Phase::Decode
        }
    }

    fn is_running(&self) -> bool {
        self.running
    }

    fn set_running(&mut self, running: bool) {
        self.running = running;
    }

    fn turn_ordinal(&self) -> u128 {
        self.turn_ordinal
    }

    fn set_turn_ordinal(&mut self, turn: u128) {
        self.turn_ordinal = turn;
    }
}

fn state_compatible_prefix_len(requests: &[&Request]) -> usize {
    let mut batch_claims = std::collections::BTreeMap::new();
    for (row_index, request) in requests.iter().enumerate() {
        debug_assert_eq!(
            request.resources.recurrent.handles.len(),
            request.resources.recurrent.access_modes.len()
        );
        let mut row_claims = std::collections::BTreeMap::new();
        for (handle, mode) in request
            .resources
            .recurrent
            .handles
            .iter()
            .cloned()
            .zip(request.resources.recurrent.access_modes.iter().copied())
        {
            row_claims
                .entry(handle)
                .and_modify(|effective| {
                    if mode == StateAccessMode::Write {
                        *effective = StateAccessMode::Write;
                    }
                })
                .or_insert(mode);
        }
        if row_claims.iter().any(|(handle, mode)| {
            batch_claims.get(handle).is_some_and(|existing| {
                *existing == StateAccessMode::Write || *mode == StateAccessMode::Write
            })
        }) {
            return row_index;
        }
        batch_claims.extend(row_claims);
    }
    requests.len()
}

enum InFlightExecution {
    SyncComplete(Vec<u32>),
    FailedAfterAcceptance,
    AsyncHandle(Py<PyAny>),
}

enum RunnerCallOutcome {
    Accepted(InFlightExecution),
    Rejected(PyErr),
    FailedAfterAcceptance { stage: &'static str, error: PyErr },
}

type ScheduledRow = SequenceStep;

struct InFlightRow {
    request: Request,
    scheduled: ScheduledRow,
}

struct PreparedProgress {
    apply: bool,
    resident_tokens: u32,
    sampled_token: Option<u32>,
}

struct InFlightData {
    batch_id: u64,
    rows: Vec<InFlightRow>,
    execution: InFlightExecution,
}

struct RecurrentBatchState {
    handles: Vec<CoreStateHandle>,
    access: Option<StateAccessLease>,
}

type BatchResources = ResourceSet<Option<WriterAccessLease>, RecurrentBatchState>;
type InFlightBatch = WithResources<InFlightData, BatchResources>;

#[pyclass(module = "resin._resin")]
struct Engine {
    // Accepted work must be destroyed before the bound runner callables which
    // keep its execution-plane owner alive. This also makes fail-stop teardown
    // safe for runner completion handles whose destructor waits for work.
    in_flight: Option<InFlightBatch>,
    runner: Py<PyAny>,
    runner_execution: RunnerExecution,
    runner_release: Option<Py<PyAny>>,
    runner_prefill_chunk_size: Option<NonZeroU32>,
    runner_max_batch_tokens: Option<NonZeroU32>,
    scheduler: Scheduler<Request>,
    scheduler_config: SchedulerConfig,
    new_token_ratio: f64,
    minimum_new_token_ratio: f64,
    new_token_ratio_decay: f64,
    context_length: NonZeroU32,
    eos_token_id: Option<u32>,
    engine_id: u64,
    next_request_id: u64,
    next_batch_id: u64,
    state_pool: Option<Py<StatePool>>,
    recurrent_state_mapping: Option<StateMappingConfig>,
    cache_layout: Option<CacheLayout>,
    kv_resources: Option<PagedKvResources>,
    poisoned_error: Option<String>,
    closed: bool,
}

const DROP_DRAIN_TIMEOUT: Duration = Duration::from_millis(250);
const CLOSE_POLL_INTERVAL: Duration = Duration::from_millis(1);

fn cache_pool_error(error: CachePoolError) -> PyErr {
    match error {
        CachePoolError::MetadataAllocation | CachePoolError::InsufficientCapacity { .. } => {
            PyMemoryError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

fn paged_kv_error(error: PagedKvError) -> PyErr {
    match error {
        PagedKvError::Cache(error) => cache_pool_error(error),
        PagedKvError::Invalid(_) => PyRuntimeError::new_err(error.to_string()),
    }
}

fn core_request_id(request_id: u64) -> RequestId {
    RequestId::new(NonZeroU64::new(request_id).expect("Engine request ids are positive"))
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        runner: "resin.protocols.Runner",
        *,
        max_running_requests = 32,
        context_length = 32768,
        scheduler_config = None,
        eos_token_id = None,
        state_pool = None
    ))]
    fn new(
        py: Python<'_>,
        runner: Py<PyAny>,
        max_running_requests: usize,
        context_length: u32,
        scheduler_config: Option<Py<SchedulerConfig>>,
        eos_token_id: Option<u32>,
        state_pool: Option<Py<StatePool>>,
    ) -> PyResult<Self> {
        if max_running_requests == 0 {
            return Err(PyValueError::new_err(
                "max_running_requests must be positive",
            ));
        }
        let context_length = NonZeroU32::new(context_length)
            .ok_or_else(|| PyValueError::new_err("context_length must be positive"))?;
        let scheduler_config = scheduler_config
            .as_ref()
            .map(|config| *config.borrow(py))
            .unwrap_or_default();
        let new_token_ratio = scheduler_config.initial_new_token_ratio();
        let minimum_new_token_ratio = new_token_ratio * MIN_NEW_TOKEN_RATIO_FACTOR;
        let new_token_ratio_decay =
            (new_token_ratio - minimum_new_token_ratio) / NEW_TOKEN_RATIO_DECAY_STEPS;
        let (runner_execution, runner_release) = resolve_runner_contract(py, runner.bind(py))?;
        let runner_prefill_chunk_size = resolve_runner_prefill_chunk_size(py, runner.bind(py))?;
        let runner_max_batch_tokens = resolve_runner_max_batch_tokens(py, runner.bind(py))?;

        let runner_resources = resolve_runner_resources(py, runner.bind(py), state_pool.as_ref())?;
        let kv_resources = runner_resources
            .cache_layout
            .as_ref()
            .map(|layout| {
                let layout = layout.borrow(py);
                PagedKvResources::new(layout.inner_native().clone()).map_err(cache_pool_error)
            })
            .transpose()?;
        // Complete every fallible native allocation before the runner's one
        // resource-shape-specific binding callback publishes physical identity.
        let engine_id = mint_engine_id()?;
        let (cache_layout, recurrent_state_mapping) = bind_runner_resources(
            py,
            &runner_resources,
            kv_resources.as_ref().map(|resources| resources.pool().id()),
            state_pool.as_ref(),
        )?;

        Ok(Self {
            in_flight: None,
            runner: runner.clone_ref(py),
            runner_execution,
            runner_release,
            runner_prefill_chunk_size,
            runner_max_batch_tokens,
            scheduler: Scheduler::new(
                NonZeroUsize::new(max_running_requests).expect("validated positive batch limit"),
            ),
            scheduler_config,
            new_token_ratio,
            minimum_new_token_ratio,
            new_token_ratio_decay,
            context_length,
            eos_token_id,
            engine_id,
            next_request_id: 1,
            next_batch_id: 1,
            state_pool,
            recurrent_state_mapping,
            cache_layout,
            kv_resources,
            poisoned_error: None,
            closed: false,
        })
    }

    #[pyo3(signature = (
        token_ids,
        max_new_tokens = None,
        *,
        state_handles = None,
        state_access_modes = None,
        priority = 0,
        batch_key = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &mut self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        max_new_tokens: Option<usize>,
        state_handles: Option<Vec<Py<StateHandle>>>,
        state_access_modes: Option<Vec<String>>,
        priority: i32,
        batch_key: Option<u64>,
    ) -> PyResult<Py<RequestLease>> {
        self.raise_if_open()?;
        if token_ids.is_empty() {
            return Err(PyValueError::new_err("token_ids must be non-empty"));
        }
        if max_new_tokens == Some(0) {
            return Err(PyValueError::new_err("max_new_tokens must be positive"));
        }
        let max_new_tokens = max_new_tokens.unwrap_or_else(|| {
            (self.context_length.get() as usize).saturating_sub(token_ids.len()) + 1
        });
        let max_input_tokens = token_ids
            .len()
            .checked_add(max_new_tokens - 1)
            .filter(|length| *length <= u32::MAX as usize)
            .ok_or_else(|| {
                PyValueError::new_err("maximum request input length exceeds the u32 metadata range")
            })? as u32;
        if max_input_tokens > self.context_length.get() {
            return Err(PyValueError::new_err(format!(
                "request context length {max_input_tokens} exceeds Engine context_length {}",
                self.context_length.get()
            )));
        }

        let request_id = self.next_request_id;
        let next_request_id = request_id
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("request id space is exhausted"))?;
        let state_handles = state_handles
            .unwrap_or_default()
            .into_iter()
            .map(|handle| handle.borrow(py).inner())
            .collect::<Vec<_>>();
        let state_access_modes = parse_state_access_modes(state_access_modes, state_handles.len())?;
        if let Some(mapping) = &self.recurrent_state_mapping
            && (state_handles.len() != mapping.access_modes.len()
                || state_access_modes != mapping.access_modes)
        {
            let handle_count = if mapping.access_modes.len() == 1 {
                "one".to_owned()
            } else {
                mapping.access_modes.len().to_string()
            };
            let handle_noun = if mapping.access_modes.len() == 1 {
                "handle"
            } else {
                "handles"
            };
            let expected_modes = mapping
                .access_modes
                .iter()
                .map(|mode| format!("'{}'", mode.as_str()))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(PyValueError::new_err(format!(
                "recurrent-state mapped requests require exactly {handle_count} state {handle_noun} with access modes [{expected_modes}]",
            )));
        }

        // Create the caller-owned lease before any native admission side
        // effect. A Python allocation failure therefore leaves submission
        // fully transactional.
        let lease_state = Arc::new(Mutex::new(RequestLeaseState::default()));
        let lease = Py::new(
            py,
            RequestLease {
                engine_id: self.engine_id,
                request_id,
                state: Arc::clone(&lease_state),
            },
        )?;

        let mut pages = self
            .kv_resources
            .as_mut()
            .map(|resources| {
                resources
                    .attach(core_request_id(request_id), 0)
                    .map_err(cache_pool_error)
            })
            .transpose()?;

        if !state_handles.is_empty() {
            let Some(state_pool) = &self.state_pool else {
                if let Some(pages) = pages.take() {
                    self.kv_resources
                        .as_mut()
                        .expect("attached KV pages keep their resources")
                        .retire(pages)
                        .expect("unleased newly attached KV pages retire");
                }
                return Err(PyValueError::new_err(
                    "state_handles require an Engine state_pool",
                ));
            };
            // The Engine takes its own state reference. The caller may keep
            // another reference for sharing or release it to transfer ownership.
            if let Err(error) = state_pool
                .bind(py)
                .borrow_mut()
                .retain_handles(&state_handles)
            {
                if let Some(pages) = pages.take() {
                    self.kv_resources
                        .as_mut()
                        .expect("attached KV pages keep their resources")
                        .retire(pages)
                        .expect("unleased newly attached KV pages retire");
                }
                return Err(error);
            }
        }
        self.next_request_id = next_request_id;
        self.scheduler.enqueue(Request {
            inner: RequestData {
                id: request_id,
                lease: lease_state,
                priority,
                batch_key,
                turn_ordinal: 0,
                running: false,
                prompt: token_ids,
                resident_tokens: 0,
                generated: Vec::new(),
                max_new_tokens,
                cancel_pending: false,
            },
            resources: ResourceSet {
                attention: pages,
                recurrent: RecurrentRequestState {
                    handles: state_handles,
                    access_modes: state_access_modes,
                },
            },
        });
        Ok(lease)
    }

    fn dispatch(&mut self, py: Python<'_>) -> PyResult<Option<u64>> {
        self.raise_if_open()?;
        if self.in_flight.is_some() || self.scheduler.is_empty() {
            return Ok(None);
        }

        let (scheduled_rows, selection) = {
            let (mut selected_requests, mut selection) = self.scheduler.selected();
            let admitted = self.admission_prefix_len(&selected_requests)?;
            if admitted == 0 {
                (selected_requests, selection) = self.scheduler.selected_running();
            } else {
                selected_requests.truncate(admitted);
                selection.truncate(admitted);
            }
            if selected_requests.is_empty() {
                return Ok(None);
            }
            let compatible = state_compatible_prefix_len(&selected_requests);
            debug_assert_ne!(compatible, 0);
            selected_requests.truncate(compatible);
            selection.truncate(compatible);
            let scheduled_rows = schedule_ready_prefix(
                &selected_requests,
                self.scheduler_config.with_runner_limits(
                    self.runner_prefill_chunk_size,
                    self.runner_max_batch_tokens,
                ),
            )?;
            if scheduled_rows.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "scheduler produced no token-bearing rows for a nonempty cohort",
                ));
            }
            if scheduled_rows
                .iter()
                .any(|row| row.phase == Phase::Prefill && !row.sample)
                && !runner_supports_continuation_prefill(py, self.runner.bind(py))?
            {
                return Err(PyValueError::new_err(
                    "prefill_chunk_size creates continuation rows but the runner does not declare supports_continuation_prefill is True",
                ));
            }
            selection.truncate(scheduled_rows.len());
            (scheduled_rows, selection)
        };
        let selected = scheduled_rows.len();
        let batch_id = self.next_batch_id;
        self.next_batch_id = self
            .next_batch_id
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("batch id space is exhausted"))?;
        let mut scheduled = self.scheduler.take_selected(selection);

        if let Some(resources) = self.kv_resources.as_mut() {
            let mut growth_error = None;
            for (request, row) in scheduled.iter_mut().zip(&scheduled_rows) {
                let context = row
                    .token_end()
                    .ok_or_else(|| PyValueError::new_err("scheduled token range overflows u32"))?;
                let pages = request
                    .resources
                    .attention
                    .as_mut()
                    .expect("managed requests keep their KV resource state");
                if let Err(error) = resources.ensure_context(pages, context) {
                    growth_error = Some(error);
                    break;
                }
            }
            if let Some(error) = growth_error {
                self.scheduler.restore_rejected(scheduled);
                if matches!(error, CachePoolError::InsufficientCapacity { .. })
                    && self.retract_one(py)?
                {
                    return self.dispatch(py);
                }
                return Err(cache_pool_error(error));
            }
        }

        let mut acquired_kv = match self.kv_resources.as_mut() {
            Some(resources) => {
                let requests = scheduled
                    .iter()
                    .map(|request| RequestResourceRef {
                        request_id: core_request_id(request.id),
                        state: request
                            .resources
                            .attention
                            .as_ref()
                            .expect("managed requests keep their KV resource state"),
                    })
                    .collect::<Vec<_>>();
                match resources.acquire(&requests, &scheduled_rows) {
                    Ok(acquired) => Some(acquired),
                    Err(error) => {
                        self.scheduler.restore_rejected(scheduled);
                        return Err(paged_kv_error(error));
                    }
                }
            }
            None => {
                if scheduled
                    .iter()
                    .any(|request| request.resources.attention.is_some())
                {
                    self.scheduler.restore_rejected(scheduled);
                    return Err(PyRuntimeError::new_err(
                        "unmanaged Engine request unexpectedly owns KV resource state",
                    ));
                }
                None
            }
        };

        let plan = match build_plan_for_engine(
            batch_id,
            Some(self.engine_id),
            &scheduled,
            &scheduled_rows,
            acquired_kv.as_ref().map(|acquired| &acquired.view),
        ) {
            Ok(plan) => plan,
            Err(error) => {
                self.release_kv_access(acquired_kv.take().map(|access| access.lease));
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let plan = match Py::new(py, plan) {
            Ok(plan) => plan,
            Err(error) => {
                self.release_kv_access(acquired_kv.take().map(|access| access.lease));
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let recurrent = match self.acquire_recurrent_resources(py, &scheduled) {
            Ok(resources) => resources,
            Err(error) => {
                self.release_kv_access(acquired_kv.take().map(|access| access.lease));
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let mut batch_resources = ResourceSet {
            attention: acquired_kv.map(|access| access.lease),
            recurrent,
        };

        // The lease spans runner acceptance and every pending poll. A
        // post-accept error therefore quarantines the exact writer capability.
        let outcome = match &self.runner_execution {
            RunnerExecution::Sync { forward } => {
                match forward.bind(py).call1((plan.clone_ref(py),)) {
                    Err(error) if runner_error_is_post_accept(py, &error) => {
                        RunnerCallOutcome::FailedAfterAcceptance {
                            stage: "runner forward failed after accepting work",
                            error,
                        }
                    }
                    Err(error) => RunnerCallOutcome::Rejected(error),
                    Ok(output) => match output.extract::<Vec<u32>>() {
                        Err(error) => RunnerCallOutcome::FailedAfterAcceptance {
                            stage: "runner forward protocol failure after accepting work",
                            error,
                        },
                        Ok(output) if output.len() != selected => {
                            RunnerCallOutcome::FailedAfterAcceptance {
                                stage: "runner forward protocol failure after accepting work",
                                error: PyRuntimeError::new_err(format!(
                                    "runner returned {} token ids for {selected} sequences",
                                    output.len()
                                )),
                            }
                        }
                        Ok(output) => {
                            RunnerCallOutcome::Accepted(InFlightExecution::SyncComplete(output))
                        }
                    },
                }
            }
            RunnerExecution::Async { dispatch, .. } => {
                match dispatch.bind(py).call1((plan.clone_ref(py),)) {
                    Ok(handle) => {
                        RunnerCallOutcome::Accepted(InFlightExecution::AsyncHandle(handle.unbind()))
                    }
                    Err(error) if runner_error_is_post_accept(py, &error) => {
                        RunnerCallOutcome::FailedAfterAcceptance {
                            stage: "runner dispatch failed after accepting work",
                            error,
                        }
                    }
                    Err(error) => RunnerCallOutcome::Rejected(error),
                }
            }
        };
        let execution = match outcome {
            RunnerCallOutcome::Accepted(execution) => execution,
            RunnerCallOutcome::FailedAfterAcceptance { stage, error } => {
                let batch = InFlightBatch {
                    inner: InFlightData {
                        batch_id,
                        rows: scheduled
                            .into_iter()
                            .zip(scheduled_rows)
                            .map(|(request, scheduled)| InFlightRow { request, scheduled })
                            .collect(),
                        execution: InFlightExecution::FailedAfterAcceptance,
                    },
                    resources: batch_resources,
                };
                return self.quarantine(batch, stage, error);
            }
            RunnerCallOutcome::Rejected(error) => {
                self.rollback_batch_resources(py, &mut batch_resources);
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };

        self.in_flight = Some(InFlightBatch {
            inner: InFlightData {
                batch_id,
                rows: scheduled
                    .into_iter()
                    .zip(scheduled_rows)
                    .map(|(request, scheduled)| InFlightRow { request, scheduled })
                    .collect(),
                execution,
            },
            resources: batch_resources,
        });
        Ok(Some(batch_id))
    }

    fn poll(&mut self, py: Python<'_>) -> PyResult<Vec<Py<TokenEvent>>> {
        self.raise_if_poisoned()?;
        let Some(mut batch) = self.in_flight.take() else {
            return Ok(Vec::new());
        };

        let output_token_ids = match &batch.execution {
            InFlightExecution::SyncComplete(output) => output.clone(),
            InFlightExecution::FailedAfterAcceptance => {
                let error = PyRuntimeError::new_err(
                    "a failed accepted batch exists without a poisoned Engine",
                );
                return self.quarantine(batch, "native acceptance protocol failure", error);
            }
            InFlightExecution::AsyncHandle(handle) => {
                let RunnerExecution::Async { poll, .. } = &self.runner_execution else {
                    let error = PyRuntimeError::new_err(
                        "an asynchronous batch exists on a synchronous runner",
                    );
                    return self.quarantine(batch, "native async protocol failure", error);
                };
                let result = match poll.bind(py).call1((handle.clone_ref(py),)) {
                    Ok(result) => result,
                    Err(error) => return self.quarantine(batch, "runner poll failed", error),
                };
                if result.is_none() {
                    self.in_flight = Some(batch);
                    return Ok(Vec::new());
                }
                if !result.is_instance_of::<PyList>() {
                    let error = PyTypeError::new_err(
                        "runner.poll must return None or a Python list of token IDs",
                    );
                    return self.quarantine(batch, "runner poll protocol failure", error);
                }
                match result.extract::<Vec<u32>>() {
                    Ok(output) => output,
                    Err(error) => {
                        return self.quarantine(batch, "runner poll protocol failure", error);
                    }
                }
            }
        };
        if output_token_ids.len() != batch.rows.len() {
            let error = PyRuntimeError::new_err(format!(
                "runner returned {} token ids for {} sequences",
                output_token_ids.len(),
                batch.rows.len()
            ));
            return self.quarantine(batch, "runner poll protocol failure", error);
        }
        let decoded_batch = batch
            .rows
            .first()
            .is_some_and(|row| row.scheduled.phase == Phase::Decode);

        let mut retired = Vec::with_capacity(batch.rows.len());
        let mut retired_ids = Vec::new();
        let mut events = Vec::with_capacity(batch.rows.len());
        for (row, token_id) in batch.rows.iter().zip(output_token_ids.iter().copied()) {
            let request = &row.request;
            if request.cancel_pending {
                retired.push(true);
                retired_ids.push(request.id);
                continue;
            }
            if !row.scheduled.sample {
                retired.push(false);
                continue;
            }
            let finished = request.generated.len() + 1 >= request.max_new_tokens
                || self.eos_token_id == Some(token_id);
            let event = match Py::new(
                py,
                TokenEvent {
                    request_id: request.id,
                    token_id,
                    index: request.generated.len(),
                    finished,
                },
            ) {
                Ok(event) => event,
                Err(error) => {
                    return self.quarantine(batch, "token event construction failed", error);
                }
            };
            events.push(event);
            retired.push(finished);
            if finished {
                retired_ids.push(request.id);
            }
        }

        let prepared_progress =
            match self.prepare_scheduled_progress(&mut batch.rows, &output_token_ids) {
                Ok(prepared) => prepared,
                Err(error) => {
                    return self.quarantine(batch, "native progress commit failed", error);
                }
            };

        // Validate every physical retirement and reserve all destination
        // storage before asking the runner or any allocator to release state.
        let finished_state = batch
            .rows
            .iter()
            .zip(&retired)
            .filter(|(_, retire)| **retire)
            .flat_map(|(row, _)| row.request.resources.recurrent.handles.iter().cloned())
            .collect::<Vec<_>>();
        for (row, retire) in batch.rows.iter().zip(&retired) {
            if !*retire {
                continue;
            }
            match (
                self.kv_resources.as_ref(),
                row.request.resources.attention.as_ref(),
            ) {
                (Some(resources), Some(pages)) => {
                    if let Err(error) = resources.pool().validate_request(pages) {
                        return self.quarantine(
                            batch,
                            "native cache retirement failed",
                            cache_pool_error(error),
                        );
                    }
                }
                (None, None) => {}
                _ => {
                    let error = PyRuntimeError::new_err(
                        "retiring request and Engine cache ownership disagree",
                    );
                    return self.quarantine(batch, "native cache retirement failed", error);
                }
            }
        }
        let mut survivors = Vec::new();
        if survivors
            .try_reserve_exact(batch.rows.len().saturating_sub(retired_ids.len()))
            .is_err()
        {
            let error = PyMemoryError::new_err("could not reserve surviving request storage");
            return self.quarantine(batch, "native survivor storage failed", error);
        }

        if !retired_ids.is_empty()
            && let Some(release) = &self.runner_release
            && let Err(error) = release.bind(py).call1((retired_ids.clone(),))
        {
            return self.quarantine(batch, "runner release failed", error);
        }
        if let Err(error) = self.unpin_state(py, &batch.resources.recurrent.handles) {
            return self.quarantine(batch, "native completion unpin failed", error);
        }

        // Runner cleanup and generic-state unpinning prove physical execution
        // is terminal. Only now may the single cache writer lease return.
        self.release_kv_access(batch.resources.attention.take());

        if let Err(error) = self.release_state(py, &finished_state) {
            return self.quarantine(batch, "native state retirement failed", error);
        }
        let state_access = batch.resources.recurrent.access.take();
        self.release_state_access(py, state_access);

        for (row_index, retire) in retired.iter().copied().enumerate() {
            if !retire {
                continue;
            }
            let Some(pages) = batch.rows[row_index].request.resources.attention.take() else {
                continue;
            };
            let resources = self
                .kv_resources
                .as_mut()
                .expect("request KV pages keep their resources");
            if let Err(failure) = resources.retire(pages) {
                batch.rows[row_index].request.resources.attention = Some(failure.request);
                return self.quarantine(
                    batch,
                    "native cache retirement failed after preflight",
                    cache_pool_error(failure.error),
                );
            }
        }

        // Logical progress becomes visible only after every fallible physical
        // cleanup step has completed.
        Self::apply_scheduled_progress(&mut batch.rows, prepared_progress);
        for (row, retire) in batch.rows.drain(..).zip(retired) {
            let request = row.request;
            if retire {
                let RequestData {
                    lease,
                    generated,
                    cancel_pending,
                    ..
                } = request.inner;
                complete_request_lease(&lease, generated, cancel_pending);
            } else {
                survivors.push(request);
            }
        }
        self.scheduler.requeue_survivors(survivors);
        if decoded_batch {
            self.decay_new_token_ratio();
        }
        Ok(events)
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Vec<Py<TokenEvent>>> {
        self.raise_if_open()?;
        self.dispatch(py)?;
        self.poll(py)
    }

    fn cancel(&mut self, py: Python<'_>, request: PyRef<'_, RequestLease>) -> PyResult<bool> {
        self.raise_if_open()?;
        if request.engine_id != self.engine_id {
            return Err(PyValueError::new_err(
                "RequestLease belongs to a different Engine",
            ));
        }
        let request_id = request.request_id;
        if let Some(index) = self.scheduler.position(|request| request.id == request_id) {
            let mut request = self.scheduler.remove(index);
            match (
                self.kv_resources.as_ref(),
                request.resources.attention.as_ref(),
            ) {
                (Some(resources), Some(pages)) => {
                    if let Err(error) = resources.pool().validate_request(pages) {
                        self.scheduler.restore(index, request);
                        let error = cache_pool_error(error);
                        self.poison("native cache cancellation retirement failed", &error);
                        return Err(error);
                    }
                }
                (None, None) => {}
                _ => {
                    self.scheduler.restore(index, request);
                    let error = PyRuntimeError::new_err(
                        "ready request and Engine cache ownership disagree",
                    );
                    self.poison("native cache cancellation retirement failed", &error);
                    return Err(error);
                }
            }
            if let Some(release) = &self.runner_release
                && let Err(error) = release.bind(py).call1((vec![request_id],))
            {
                self.scheduler.restore(index, request);
                self.poison("runner release failed during ready cancellation", &error);
                return Err(error);
            }
            if let Err(error) = self.release_state(py, &request.resources.recurrent.handles) {
                self.scheduler.restore(index, request);
                self.poison("native state cancellation retirement failed", &error);
                return Err(error);
            }
            if let Some(pages) = request.resources.attention.take()
                && let Err(failure) = self
                    .kv_resources
                    .as_mut()
                    .expect("request KV pages keep their resources")
                    .retire(pages)
            {
                request.resources.attention = Some(failure.request);
                self.scheduler.restore(index, request);
                let error = cache_pool_error(failure.error);
                self.poison(
                    "native cache cancellation retirement failed after preflight",
                    &error,
                );
                return Err(error);
            }
            let generated = std::mem::take(&mut request.inner.generated);
            complete_request_lease(&request.lease, generated, true);
            return Ok(true);
        }

        if let Some(batch) = &mut self.in_flight
            && let Some(row) = batch
                .rows
                .iter_mut()
                .find(|row| row.request.id == request_id)
        {
            row.request.cancel_pending = true;
            return Ok(true);
        }
        Ok(false)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }
        self.raise_if_poisoned()?;
        self.close_healthy(py, None)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.raise_if_open()?;
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }

    fn has_work(&self) -> bool {
        !self.scheduler.is_empty() || self.in_flight.is_some()
    }

    #[getter]
    fn active_request_count(&self) -> usize {
        self.scheduler.len() + self.in_flight.as_ref().map_or(0, |batch| batch.rows.len())
    }

    #[getter]
    fn ready_request_count(&self) -> usize {
        self.scheduler.len()
    }

    #[getter]
    fn in_flight_request_count(&self) -> usize {
        self.in_flight.as_ref().map_or(0, |batch| batch.rows.len())
    }

    #[getter]
    fn in_flight_batch_count(&self) -> usize {
        usize::from(self.in_flight.is_some())
    }

    #[getter]
    fn in_flight_batch_id(&self) -> Option<u64> {
        self.in_flight.as_ref().map(|batch| batch.batch_id)
    }

    #[getter]
    fn max_running_requests(&self) -> usize {
        self.scheduler.max_running_requests()
    }

    #[getter]
    fn asynchronous(&self) -> bool {
        self.runner_execution.is_async()
    }

    #[getter]
    fn poisoned(&self) -> bool {
        self.poisoned_error.is_some()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    #[getter]
    fn poison_error(&self) -> Option<&str> {
        self.poisoned_error.as_deref()
    }

    #[getter]
    fn scheduler_config(&self, py: Python<'_>) -> PyResult<Py<SchedulerConfig>> {
        Py::new(py, self.scheduler_config)
    }

    #[getter]
    fn new_token_ratio(&self) -> f64 {
        self.new_token_ratio
    }

    #[getter]
    fn context_length(&self) -> u32 {
        self.context_length.get()
    }

    #[getter]
    fn cache_layout(&self, py: Python<'_>) -> PyResult<Option<Py<CacheLayout>>> {
        self.cache_layout
            .as_ref()
            .cloned()
            .map(|layout| Py::new(py, layout))
            .transpose()
    }

    #[getter]
    fn cache_pool_id(&self) -> Option<u64> {
        self.kv_resources
            .as_ref()
            .map(|resources| resources.pool().id())
    }

    #[getter]
    fn cache_parent_capacity(&self) -> Option<u32> {
        self.cache_layout
            .as_ref()
            .map(|layout| layout.inner_native().parent_capacity().get())
    }

    #[getter]
    fn cache_free_parent_count(&self) -> Option<usize> {
        self.kv_resources
            .as_ref()
            .map(|resources| resources.pool().free_parent_count())
    }

    #[getter]
    fn cache_occupied_block_count(&self) -> Option<usize> {
        self.kv_resources
            .as_ref()
            .map(|resources| resources.pool().occupied_block_count())
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(ready_requests={}, in_flight_requests={}, max_running_requests={}, context_length={}, closed={}, poisoned={}, cache_blocks={})",
            self.ready_request_count(),
            self.in_flight_request_count(),
            self.max_running_requests(),
            self.context_length(),
            self.closed,
            self.poisoned(),
            self.kv_resources
                .as_ref()
                .map_or(0, |resources| resources.pool().occupied_block_count())
        )
    }
}

impl Engine {
    fn estimated_context(&self, request: &Request) -> u32 {
        let future_inputs = request.max_new_tokens.saturating_sub(1);
        let estimated_future = ((future_inputs as f64) * self.new_token_ratio).ceil() as usize;
        let estimate = request
            .prompt
            .len()
            .saturating_add(estimated_future)
            .max(
                request
                    .prefill_length()
                    .expect("active request length is valid") as usize,
            )
            .min(self.context_length.get() as usize);
        estimate as u32
    }

    fn admission_prefix_len(&self, selected: &[&Request]) -> PyResult<usize> {
        let Some(resources) = &self.kv_resources else {
            return Ok(selected.len());
        };
        if selected.iter().all(|request| request.is_running()) {
            return Ok(selected.len());
        }

        let mut contexts = self
            .scheduler
            .iter()
            .filter(|request| request.is_running())
            .map(|request| self.estimated_context(request))
            .collect::<Vec<_>>();
        let running_count = contexts.len();
        let selected_running = selected
            .iter()
            .take_while(|request| request.is_running())
            .count();
        let mut admitted = selected_running;
        for request in &selected[selected_running..] {
            contexts.push(self.estimated_context(request));
            let parents = resources
                .estimated_parent_count(&contexts)
                .map_err(cache_pool_error)?;
            if parents > u64::from(resources.parent_capacity())
                && (running_count != 0 || admitted != 0)
            {
                contexts.pop();
                break;
            }
            admitted += 1;
        }
        Ok(admitted.max(usize::from(running_count == 0 && !selected.is_empty())))
    }

    fn decay_new_token_ratio(&mut self) {
        self.new_token_ratio =
            (self.new_token_ratio - self.new_token_ratio_decay).max(self.minimum_new_token_ratio);
    }

    fn update_new_token_ratio_after_retraction(&mut self) {
        let running = self.scheduler.iter().filter(|request| request.is_running());
        let (count, generated, limits) = running.fold(
            (0usize, 0usize, 0usize),
            |(count, generated, limits), request| {
                (
                    count + 1,
                    generated.saturating_add(request.generated.len()),
                    limits.saturating_add(request.max_new_tokens),
                )
            },
        );
        self.new_token_ratio = if count == 0 {
            self.scheduler_config.initial_new_token_ratio()
        } else {
            ((generated.saturating_add(RETRACTION_DECODE_HEADROOM * count)) as f64
                / limits.saturating_add(1) as f64)
                .min(1.0)
        };
    }

    fn retract_one(&mut self, py: Python<'_>) -> PyResult<bool> {
        let running_count = self
            .scheduler
            .iter()
            .filter(|request| request.is_running())
            .count();
        if running_count <= 1 {
            return Ok(false);
        }
        let Some(request_id) = self
            .scheduler
            .iter()
            .filter(|request| {
                request.is_running()
                    && request.resources.attention.is_some()
                    && request.resources.recurrent.handles.is_empty()
            })
            .min_by_key(|request| (request.generated.len(), Reverse(request.prompt.len())))
            .map(|request| request.id)
        else {
            return Ok(false);
        };
        let index = self
            .scheduler
            .position(|request| request.id == request_id)
            .expect("selected retraction victim remains ready");
        let mut request = self.scheduler.remove(index);

        if let Some(release) = &self.runner_release
            && let Err(error) = release.bind(py).call1((vec![request.id],))
        {
            self.scheduler.restore(index, request);
            self.poison("runner release failed during request retraction", &error);
            return Err(error);
        }
        let pages = request
            .resources
            .attention
            .as_mut()
            .expect("retraction victim owns managed attention state");
        if let Err(error) = self
            .kv_resources
            .as_mut()
            .expect("managed attention state keeps its allocator")
            .reset(pages)
        {
            self.scheduler.restore(index, request);
            let error = cache_pool_error(error);
            self.poison(
                "native cache reset failed during request retraction",
                &error,
            );
            return Err(error);
        }
        request.resident_tokens = 0;
        self.scheduler.requeue_retracted(request);
        self.update_new_token_ratio_after_retraction();
        Ok(true)
    }

    fn raise_if_open(&self) -> PyResult<()> {
        if self.closed {
            return Err(PyRuntimeError::new_err("Engine is closed"));
        }
        self.raise_if_poisoned()
    }

    fn raise_if_poisoned(&self) -> PyResult<()> {
        if let Some(error) = &self.poisoned_error {
            return Err(PyRuntimeError::new_err(format!(
                "Engine is poisoned and its resources are quarantined: {error}"
            )));
        }
        Ok(())
    }

    fn poison(&mut self, stage: &str, error: &PyErr) {
        if self.poisoned_error.is_none() {
            self.poisoned_error = Some(format!("{stage}: {error}"));
        }
    }

    fn poison_message(&mut self, message: impl Into<String>) {
        if self.poisoned_error.is_none() {
            self.poisoned_error = Some(message.into());
        }
    }

    fn quarantine<T>(&mut self, batch: InFlightBatch, stage: &str, error: PyErr) -> PyResult<T> {
        self.in_flight = Some(batch);
        self.poison(stage, &error);
        Err(error)
    }

    fn close_healthy(&mut self, py: Python<'_>, deadline: Option<Instant>) -> PyResult<()> {
        if let Some(batch) = &mut self.in_flight {
            for row in &mut batch.rows {
                row.request.cancel_pending = true;
            }
        }

        while self.in_flight.is_some() {
            self.poll(py)?;
            if self.in_flight.is_none() {
                break;
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                let message = format!(
                    "Engine destructor could not drain accepted work within {} ms; resources remain quarantined",
                    DROP_DRAIN_TIMEOUT.as_millis()
                );
                self.poison_message(message.clone());
                return Err(PyRuntimeError::new_err(message));
            }
            if deadline.is_none() {
                py.check_signals()?;
            }
            py.detach(|| std::thread::sleep(CLOSE_POLL_INTERVAL));
        }

        self.retire_ready(py)?;
        self.closed = true;
        Ok(())
    }

    fn retire_ready(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.scheduler.is_empty() {
            return Ok(());
        }
        let request_ids = self
            .scheduler
            .iter()
            .map(|request| request.id)
            .collect::<Vec<_>>();
        let retirement_validation = self.scheduler.iter().try_for_each(|request| {
            match (
                self.kv_resources.as_ref(),
                request.resources.attention.as_ref(),
            ) {
                (Some(resources), Some(pages)) => resources
                    .pool()
                    .validate_request(pages)
                    .map_err(cache_pool_error),
                (None, None) => Ok(()),
                _ => Err(PyRuntimeError::new_err(
                    "ready request and Engine cache ownership disagree",
                )),
            }
        });
        if let Err(error) = retirement_validation {
            self.poison("native cache retirement failed during Engine close", &error);
            return Err(error);
        }
        let state = self
            .scheduler
            .iter()
            .flat_map(|request| request.resources.recurrent.handles.iter().cloned())
            .collect::<Vec<_>>();
        let mut retired = Vec::new();
        retired
            .try_reserve_exact(self.scheduler.len())
            .map_err(|_| PyMemoryError::new_err("could not reserve ready request retirement"))?;

        if let Some(release) = &self.runner_release
            && let Err(error) = release.bind(py).call1((request_ids,))
        {
            self.poison("runner release failed during Engine close", &error);
            return Err(error);
        }
        if let Err(error) = self.release_state(py, &state) {
            self.poison("native state retirement failed during Engine close", &error);
            return Err(error);
        }

        retired.extend(self.scheduler.drain());
        for mut request in retired {
            if let Some(pages) = request.resources.attention.take()
                && let Err(failure) = self
                    .kv_resources
                    .as_mut()
                    .expect("request KV pages keep their resources")
                    .retire(pages)
            {
                let error = cache_pool_error(failure.error);
                self.poison(
                    "native cache retirement failed after close preflight",
                    &error,
                );
                return Err(error);
            }
            let generated = std::mem::take(&mut request.inner.generated);
            complete_request_lease(&request.lease, generated, true);
        }
        Ok(())
    }

    fn prepare_scheduled_progress(
        &self,
        rows: &mut [InFlightRow],
        output_token_ids: &[u32],
    ) -> PyResult<Vec<PreparedProgress>> {
        if rows.len() != output_token_ids.len() {
            return Err(PyRuntimeError::new_err(
                "dense runner output does not cover every scheduled row",
            ));
        }
        let mut prepared = Vec::with_capacity(rows.len());
        for (row, token_id) in rows.iter_mut().zip(output_token_ids.iter().copied()) {
            let request = &mut row.request;
            let scheduled = row.scheduled;
            let context = request.logical_context()?;
            if scheduled.phase != request.phase() || scheduled.token_start() != context {
                return Err(PyRuntimeError::new_err(format!(
                    "request {} progress changed while its batch was in flight",
                    request.id
                )));
            }
            let token_end = scheduled
                .token_end()
                .ok_or_else(|| PyValueError::new_err("scheduled token range overflows u32"))?;
            let expected_sample = match scheduled.phase {
                Phase::Prefill => {
                    request.sample_at_prefill_end() && token_end == request.prefill_length()?
                }
                Phase::Decode => {
                    if scheduled.token_count().get() != 1 {
                        return Err(PyRuntimeError::new_err(
                            "scheduled decode rows must contain exactly one token",
                        ));
                    }
                    true
                }
            };
            if scheduled.sample != expected_sample {
                return Err(PyRuntimeError::new_err(
                    "frozen sampling metadata does not match scheduled progress",
                ));
            }
            if scheduled.phase == Phase::Prefill && token_end > request.prefill_length()? {
                return Err(PyRuntimeError::new_err(
                    "scheduled prefill commit exceeds the request replay input",
                ));
            }
            if request.cancel_pending {
                prepared.push(PreparedProgress {
                    apply: false,
                    resident_tokens: request.resident_tokens,
                    sampled_token: None,
                });
                continue;
            }
            if scheduled.sample {
                request.generated.try_reserve(1).map_err(|_| {
                    PyMemoryError::new_err("could not reserve generated-token storage")
                })?;
            }
            prepared.push(PreparedProgress {
                apply: true,
                resident_tokens: token_end,
                sampled_token: scheduled.sample.then_some(token_id),
            });
        }
        Ok(prepared)
    }

    fn apply_scheduled_progress(rows: &mut [InFlightRow], prepared: Vec<PreparedProgress>) {
        assert_eq!(rows.len(), prepared.len());
        for (row, progress) in rows.iter_mut().zip(prepared) {
            if !progress.apply {
                continue;
            }
            row.request.resident_tokens = progress.resident_tokens;
            if let Some(token_id) = progress.sampled_token {
                row.request.generated.push(token_id);
            }
        }
    }

    fn acquire_recurrent_resources(
        &self,
        py: Python<'_>,
        requests: &[Request],
    ) -> PyResult<RecurrentBatchState> {
        let handles = requests
            .iter()
            .flat_map(|request| request.resources.recurrent.handles.iter().cloned())
            .collect::<Vec<_>>();
        let prepared = self.prepare_state_access(py, requests)?;
        self.pin_state(py, &handles)?;
        let access = prepared
            .map(|prepared| {
                self.state_pool
                    .as_ref()
                    .expect("prepared state access keeps its pool")
                    .bind(py)
                    .borrow_mut()
                    .acquire_engine_access(prepared)
            })
            .transpose();
        match access {
            Ok(access) => Ok(RecurrentBatchState { handles, access }),
            Err(error) => {
                self.unpin_after_error(py, &handles);
                Err(error)
            }
        }
    }

    fn release_kv_access(&mut self, access: Option<WriterAccessLease>) {
        if let Some(access) = access {
            self.kv_resources
                .as_mut()
                .expect("writer access keeps its KV resources")
                .release(access);
        }
    }

    fn rollback_batch_resources(&mut self, py: Python<'_>, resources: &mut BatchResources) {
        self.release_kv_access(resources.attention.take());
        self.release_state_access(py, resources.recurrent.access.take());
        self.unpin_after_error(py, &resources.recurrent.handles);
    }

    fn prepare_state_access(
        &self,
        py: Python<'_>,
        requests: &[Request],
    ) -> PyResult<Option<PreparedStateAccessLease>> {
        let claim_count = requests
            .iter()
            .try_fold(0usize, |count, request| {
                count.checked_add(request.resources.recurrent.handles.len())
            })
            .ok_or_else(|| PyRuntimeError::new_err("state access claim count overflow"))?;
        if claim_count == 0 {
            return Ok(None);
        }
        let mut claims = Vec::new();
        claims
            .try_reserve_exact(claim_count)
            .map_err(|_| PyMemoryError::new_err("could not prepare state access metadata"))?;
        for (row_index, request) in requests.iter().enumerate() {
            debug_assert_eq!(
                request.resources.recurrent.handles.len(),
                request.resources.recurrent.access_modes.len()
            );
            claims.extend(
                request
                    .resources
                    .recurrent
                    .handles
                    .iter()
                    .cloned()
                    .zip(request.resources.recurrent.access_modes.iter().copied())
                    .map(|(handle, mode)| StateAccessClaim::new(row_index, handle, mode)),
            );
        }
        self.state_pool
            .as_ref()
            .expect("requests with state require a pool")
            .bind(py)
            .borrow()
            .prepare_engine_access(&claims)
            .map(Some)
    }

    fn release_state_access(&self, py: Python<'_>, access: Option<StateAccessLease>) {
        let Some(access) = access else {
            return;
        };
        self.state_pool
            .as_ref()
            .expect("state access keeps its pool")
            .bind(py)
            .borrow_mut()
            .release_engine_access(access);
    }

    fn pin_state(&self, py: Python<'_>, handles: &[CoreStateHandle]) -> PyResult<()> {
        if handles.is_empty() {
            return Ok(());
        }
        self.state_pool
            .as_ref()
            .expect("requests with state require a pool")
            .bind(py)
            .borrow_mut()
            .pin_handles(handles)
    }

    fn unpin_state(&self, py: Python<'_>, handles: &[CoreStateHandle]) -> PyResult<()> {
        if handles.is_empty() {
            return Ok(());
        }
        self.state_pool
            .as_ref()
            .expect("requests with state require a pool")
            .bind(py)
            .borrow_mut()
            .unpin_handles(handles)
    }

    fn release_state(&self, py: Python<'_>, handles: &[CoreStateHandle]) -> PyResult<()> {
        if handles.is_empty() {
            return Ok(());
        }
        self.state_pool
            .as_ref()
            .expect("requests with state require a pool")
            .bind(py)
            .borrow_mut()
            .release_handles(handles)
    }

    fn unpin_after_error(&self, py: Python<'_>, handles: &[CoreStateHandle]) {
        // The original runner error is more actionable. A failure here would
        // indicate a native ownership invariant bug and is covered by tests.
        let _ = self.unpin_state(py, handles);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.closed || self.poisoned_error.is_some() {
            return;
        }

        let deadline = Instant::now() + DROP_DRAIN_TIMEOUT;
        match Python::try_attach(|py| self.close_healthy(py, Some(deadline))) {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                self.poison("Engine destructor cleanup failed", &error);
            }
            None => self.poison_message(
                "Engine destructor could not attach to Python; resources remain quarantined",
            ),
        }
    }
}

#[pyfunction]
fn mint_state_mapping_arena_id() -> PyResult<u64> {
    state_pool::mint_state_mapping_arena_id()
}

#[pymodule(name = "_resin")]
mod _resin {
    #[pymodule_export]
    use super::{
        BatchPhase, BatchPlan, CacheGroupLayout, CacheGroupPlan, CacheLayout, Engine, RequestLease,
        SchedulerConfig, StateHandle, StateMappingConfig, StatePool, TokenEvent,
        mint_state_mapping_arena_id,
    };

    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = env!("CARGO_PKG_VERSION");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planning::{build_plan, schedule_ready_prefix};

    fn request(id: u64, prompt: &[u32], generated: &[u32], pages: Option<RequestPages>) -> Request {
        Request {
            inner: RequestData {
                id,
                lease: Arc::new(Mutex::new(RequestLeaseState::default())),
                priority: 0,
                batch_key: None,
                turn_ordinal: 0,
                running: false,
                prompt: prompt.to_vec(),
                resident_tokens: if generated.is_empty() {
                    0
                } else {
                    (prompt.len() + generated.len() - 1) as u32
                },
                generated: generated.to_vec(),
                max_new_tokens: 8,
                cancel_pending: false,
            },
            resources: ResourceSet {
                attention: pages,
                recurrent: RecurrentRequestState {
                    handles: Vec::new(),
                    access_modes: Vec::new(),
                },
            },
        }
    }

    fn history_layout() -> CacheLayout {
        CacheLayout::from_groups(
            8,
            vec![
                CacheGroupLayout::from_parts(4, 1, "history", None).unwrap(),
                CacheGroupLayout::from_parts(2, 2, "history", None).unwrap(),
            ],
        )
        .unwrap()
    }

    fn window_layout() -> CacheLayout {
        CacheLayout::from_groups(
            4,
            vec![CacheGroupLayout::from_parts(4, 1, "window", Some(4)).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn heterogeneous_phase_schedule_is_rejected() {
        let prefill = request(1, &[10, 11, 12], &[], None);
        let decode = request(2, &[20, 21], &[99], None);
        let scheduled = [&prefill, &decode];
        let error = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("scheduler selected requests from different phases")
        );
    }

    #[test]
    fn history_plan_is_sparse_and_writer_claims_match_active_pages() {
        let layout = history_layout();
        let mut resources = PagedKvResources::new(layout.inner_native().clone()).unwrap();
        let pages = resources.attach(core_request_id(41), 10).unwrap();
        let mut managed = request(41, &[1, 2, 3, 4, 5, 6], &[], Some(pages));
        managed.resident_tokens = 4;
        let scheduled = [&managed];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap();
        let resource_refs = [RequestResourceRef {
            request_id: core_request_id(41),
            state: managed.resources.attention.as_ref().unwrap(),
        }];
        let acquired = resources.acquire(&resource_refs, &rows).unwrap();
        let plan = build_plan(9, &scheduled, &rows, Some(&acquired.view)).unwrap();

        assert_eq!(plan.cache_pool_id, Some(resources.pool().id()));
        assert_eq!(plan.token_ids, vec![5, 6]);
        assert_eq!(plan.positions, vec![4, 5]);
        assert_eq!(plan.cache_groups.len(), 2);
        let full = &plan.cache_groups[0];
        assert_eq!(full.inner_native().block_indptr, [0, 2]);
        assert_eq!(full.inner_native().block_ids.len(), 2);
        assert_eq!(full.inner_native().first_logical_blocks, [0]);
        assert_eq!(full.inner_native().append_block_offsets, [0, 1]);
        let fine = &plan.cache_groups[1];
        assert_eq!(fine.inner_native().block_indptr, [0, 3]);
        assert_eq!(fine.inner_native().block_ids.len(), 3);
        assert_eq!(fine.inner_native().append_block_offsets, [0, 1]);

        assert_eq!(acquired.lease.len(), 5);
        resources.release(acquired.lease);

        let pages = managed.resources.attention.take().unwrap();
        resources.pool().validate_request(&pages).unwrap();
        resources.retire(pages).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);
        assert_eq!(resources.pool().free_parent_count(), 8);
    }

    #[test]
    fn window_plan_is_chronological_and_reuses_only_an_evicted_block() {
        let layout = window_layout();
        let mut resources = PagedKvResources::new(layout.inner_native().clone()).unwrap();
        let pages = resources.attach(core_request_id(51), 12).unwrap();
        let physical = pages.groups()[0]
            .pages()
            .iter()
            .map(|page| resources.pool().kernel_page_id(page.handle()).unwrap())
            .collect::<Vec<_>>();
        let mut managed = request(
            51,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[],
            Some(pages),
        );
        managed.resident_tokens = 8;
        let scheduled = [&managed];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap();
        assert_eq!(rows[0].token_start(), 8);
        assert_eq!(rows[0].token_count().get(), 4);

        let resource_refs = [RequestResourceRef {
            request_id: core_request_id(51),
            state: managed.resources.attention.as_ref().unwrap(),
        }];
        let acquired = resources.acquire(&resource_refs, &rows).unwrap();
        let plan = build_plan(10, &scheduled, &rows, Some(&acquired.view)).unwrap();
        let window = &plan.cache_groups[0];
        assert_eq!(window.inner_native().first_logical_blocks, [1]);
        assert_eq!(window.inner_native().block_indptr, [0, 2]);
        assert_eq!(window.inner_native().block_ids, [physical[1], physical[0]]);
        assert_eq!(
            window.inner_native().append_block_ids,
            [physical[0], physical[0], physical[0], physical[0]]
        );
        assert_eq!(window.inner_native().append_block_offsets, [0, 1, 2, 3]);

        assert_eq!(acquired.lease.len(), 2);
        resources.release(acquired.lease);

        resources
            .retire(managed.resources.attention.take().unwrap())
            .unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);
    }

    #[test]
    fn window_prefill_can_cross_logical_block_boundaries() {
        let mut first = request(1, &[1, 2, 3, 4, 5, 6], &[], None);
        first.resident_tokens = 3;
        let second = request(2, &[7, 8, 9, 10, 11, 12], &[], None);
        let scheduled = [&first, &second];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap();
        assert_eq!(rows[0].token_count().get(), 3);
        assert_eq!(rows[1].token_count().get(), 6);
    }

    #[test]
    fn batch_plan_rejects_noncanonical_cache_group_order() {
        let unmanaged = request(1, &[1, 2], &[], None);
        let scheduled = [&unmanaged];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap();
        let mut plan = build_plan(1, &scheduled, &rows, None).unwrap();
        plan.cache_pool_id = Some(3);
        plan.cache_groups = vec![
            CacheGroupPlan::from_parts(1, vec![0, 1], vec![7], vec![0], vec![7, 7], vec![0, 1])
                .unwrap(),
        ];
        assert!(BatchPlan::checked(plan).is_err());
    }

    #[test]
    fn token_budget_and_prefill_chunk_size_still_compose() {
        let first = request(1, &[1, 2, 3, 4], &[], None);
        let second = request(2, &[5, 6, 7, 8], &[], None);
        let scheduled = [&first, &second];
        let config = SchedulerConfig {
            max_batch_tokens: NonZeroU32::new(3),
            prefill_chunk_size: NonZeroU32::new(2).expect("nonzero test chunk"),
            ..SchedulerConfig::default()
        };
        let rows = schedule_ready_prefix(&scheduled, config).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].token_count().get(), 2);
        assert_eq!(rows[1].token_count().get(), 1);
        assert!(!rows[0].sample);
        assert!(!rows[1].sample);
    }
}
