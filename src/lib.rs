use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

mod cache;
mod cache_pool;
mod planning;
mod runner;
mod scheduler;
mod state_pool;

use cache::{CacheGroupLayout, CacheGroupPlan, CacheLayout, CacheRetention};
use cache_pool::{
    CachePool, CachePoolError, GroupDemand, GroupId, RequestPages, SequenceId, WriterAccessLease,
};
#[cfg(test)]
use planning::{build_plan, schedule_ready_prefix};
use planning::{
    build_plan_for_engine, schedule_ready_prefix_for_layout, scheduled_writer_access_claims,
};
use runner::{
    RunnerExecution, bind_runner_resources, resolve_runner_contract,
    resolve_runner_prefill_chunk_size, resolve_runner_resources, runner_error_is_post_accept,
    runner_supports_continuation_prefill,
};
use scheduler::{Phase, QueueMode, Schedulable, Scheduler, SchedulerPolicy};
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SchedulerConfig {
    policy: SchedulerPolicy,
    max_batch_tokens: Option<NonZeroU32>,
    prefill_chunk_size: NonZeroU32,
}

const DEFAULT_PREFILL_CHUNK_SIZE: NonZeroU32 = NonZeroU32::new(8192).unwrap();

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulerPolicy::default(),
            max_batch_tokens: None,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
        }
    }
}

#[pymethods]
impl SchedulerConfig {
    #[new]
    #[pyo3(signature = (
        *,
        queue_mode = "round_robin",
        allow_mixed = true,
        max_batch_tokens = None,
        prefill_chunk_size = 8192
    ))]
    fn new(
        queue_mode: &str,
        allow_mixed: bool,
        max_batch_tokens: Option<u32>,
        prefill_chunk_size: u32,
    ) -> PyResult<Self> {
        let queue_mode = match queue_mode {
            "round_robin" => QueueMode::RoundRobin,
            "fifo" => QueueMode::Fifo,
            _ => {
                return Err(PyValueError::new_err(
                    "queue_mode must be 'round_robin' or 'fifo'",
                ));
            }
        };
        Ok(Self {
            policy: SchedulerPolicy {
                queue_mode,
                allow_mixed,
            },
            max_batch_tokens: max_batch_tokens
                .map(|value| {
                    NonZeroU32::new(value)
                        .ok_or_else(|| PyValueError::new_err("max_batch_tokens must be positive"))
                })
                .transpose()?,
            prefill_chunk_size: NonZeroU32::new(prefill_chunk_size)
                .ok_or_else(|| PyValueError::new_err("prefill_chunk_size must be positive"))?,
        })
    }

    #[getter]
    fn queue_mode(&self) -> &'static str {
        self.policy.queue_mode.as_str()
    }

    #[getter]
    fn allow_mixed(&self) -> bool {
        self.policy.allow_mixed
    }

    #[getter]
    fn max_batch_tokens(&self) -> Option<u32> {
        self.max_batch_tokens.map(NonZeroU32::get)
    }

    #[getter]
    fn prefill_chunk_size(&self) -> u32 {
        self.prefill_chunk_size.get()
    }

    fn __repr__(&self) -> String {
        format!(
            "SchedulerConfig(queue_mode='{}', allow_mixed={}, max_batch_tokens={}, prefill_chunk_size={})",
            self.queue_mode(),
            if self.allow_mixed() { "True" } else { "False" },
            self.max_batch_tokens()
                .map_or_else(|| "None".to_owned(), |value| value.to_string()),
            self.prefill_chunk_size(),
        )
    }
}

impl SchedulerConfig {
    fn with_runner_prefill_limit(self, runner_limit: Option<NonZeroU32>) -> Self {
        let Some(runner_limit) = runner_limit else {
            return self;
        };
        Self {
            prefill_chunk_size: self.prefill_chunk_size.min(runner_limit),
            ..self
        }
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
    mode: String,
    token_ids: Vec<u32>,
    positions: Vec<u32>,
    sequence_ids: Vec<u64>,
    sequence_indptr: Vec<u32>,
    context_lengths: Vec<u32>,
    prefill_sequence_indices: Vec<u32>,
    decode_sequence_indices: Vec<u32>,
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

fn phase_mode(prefill_sequence_indices: &[u32], decode_sequence_indices: &[u32]) -> &'static str {
    match (
        prefill_sequence_indices.is_empty(),
        decode_sequence_indices.is_empty(),
    ) {
        (false, true) => "prefill",
        (true, false) => "decode",
        (false, false) => "mixed",
        (true, true) => unreachable!("a valid phase partition contains at least one row"),
    }
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
            mode,
            token_ids,
            positions,
            sequence_ids,
            sequence_indptr,
            context_lengths,
            prefill_sequence_indices,
            decode_sequence_indices,
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
        if !matches!(mode.as_str(), "prefill" | "decode" | "mixed") {
            return Err(PyValueError::new_err(
                "mode must be 'prefill', 'decode', or 'mixed'",
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
            "prefill_sequence_indices",
            prefill_sequence_indices,
            sequence_ids.len(),
        )?;
        validate_sequence_indices(
            "decode_sequence_indices",
            decode_sequence_indices,
            sequence_ids.len(),
        )?;
        validate_sequence_indices(
            "sample_sequence_indices",
            sample_sequence_indices,
            sequence_ids.len(),
        )?;
        let mut phase_by_sequence = vec![0_u8; sequence_ids.len()];
        for sequence_index in prefill_sequence_indices {
            phase_by_sequence[*sequence_index as usize] = 1;
        }
        for sequence_index in decode_sequence_indices {
            let phase = &mut phase_by_sequence[*sequence_index as usize];
            if *phase != 0 {
                return Err(PyValueError::new_err(
                    "prefill_sequence_indices and decode_sequence_indices must be disjoint",
                ));
            }
            *phase = 2;
        }
        if phase_by_sequence.contains(&0) {
            return Err(PyValueError::new_err(
                "prefill_sequence_indices and decode_sequence_indices must exactly partition the batch sequences",
            ));
        }
        let expected_mode = phase_mode(prefill_sequence_indices, decode_sequence_indices);
        if mode != expected_mode {
            return Err(PyValueError::new_err(format!(
                "mode must be '{expected_mode}' for the explicit phase partition"
            )));
        }
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
            if group.group_index_native() != group_index as u32 {
                return Err(PyValueError::new_err(
                    "cache_groups must be ordered by contiguous group_index",
                ));
            }
            if group.first_logical_blocks_native().len() != self.sequence_ids.len() {
                return Err(PyValueError::new_err(
                    "each cache group must contain one block-table row per sequence",
                ));
            }
            if group.append_block_ids_native().len() != self.token_ids.len() {
                return Err(PyValueError::new_err(
                    "each cache group must contain one append target per batch token",
                ));
            }
            for sequence_index in 0..self.sequence_ids.len() {
                let block_start = group.block_indptr_native()[sequence_index] as usize;
                let block_end = group.block_indptr_native()[sequence_index + 1] as usize;
                let row_blocks = &group.block_ids_native()[block_start..block_end];
                let token_start = self.sequence_indptr[sequence_index] as usize;
                let token_end = self.sequence_indptr[sequence_index + 1] as usize;
                if group.append_block_ids_native()[token_start..token_end]
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
        mode = "prefill",
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
        prefill_sequence_indices,
        decode_sequence_indices,
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
        mode: &str,
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
        prefill_sequence_indices: Vec<u32>,
        decode_sequence_indices: Vec<u32>,
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
            mode: mode.to_owned(),
            token_ids,
            positions,
            sequence_ids,
            sequence_indptr,
            context_lengths,
            prefill_sequence_indices,
            decode_sequence_indices,
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
    fn mode(&self) -> &str {
        &self.mode
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
    fn prefill_sequence_indices(&self) -> Vec<u32> {
        self.prefill_sequence_indices.clone()
    }

    #[getter]
    fn decode_sequence_indices(&self) -> Vec<u32> {
        self.decode_sequence_indices.clone()
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
            "BatchPlan(batch_id={}, mode='{}', sequences={}, tokens={}, states={}, cache_groups={})",
            self.batch_id,
            self.mode,
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
struct Request {
    id: u64,
    lease: Arc<Mutex<RequestLeaseState>>,
    priority: i32,
    batch_key: Option<u64>,
    turn_ordinal: u128,
    prompt: Vec<u32>,
    prefill_cursor: u32,
    generated: Vec<u32>,
    max_new_tokens: usize,
    state_handles: Vec<StateHandle>,
    state_access_modes: Vec<StateAccessMode>,
    pages: Option<RequestPages>,
    cancel_pending: bool,
}

impl Request {
    fn prompt_length(&self) -> u32 {
        u32::try_from(self.prompt.len()).expect("submit bounds prompt length to u32")
    }

    fn logical_context(&self) -> PyResult<u32> {
        let prompt_length = self.prompt_length();
        if self.generated.is_empty() {
            if self.prefill_cursor >= prompt_length {
                return Err(PyRuntimeError::new_err(format!(
                    "request {} has an invalid prefill cursor {} for prompt length {prompt_length}",
                    self.id, self.prefill_cursor
                )));
            }
            Ok(self.prefill_cursor)
        } else {
            if self.prefill_cursor != prompt_length {
                return Err(PyRuntimeError::new_err(format!(
                    "request {} entered decode before its prompt was fully committed",
                    self.id
                )));
            }
            let generated_inputs = u32::try_from(self.generated.len() - 1).map_err(|_| {
                PyValueError::new_err("generated sequence length exceeds the u32 metadata range")
            })?;
            prompt_length.checked_add(generated_inputs).ok_or_else(|| {
                PyValueError::new_err("sequence position exceeds the u32 metadata range")
            })
        }
    }
}

impl Schedulable for Request {
    fn priority(&self) -> i32 {
        self.priority
    }

    fn batch_key(&self) -> Option<u64> {
        self.batch_key
    }

    fn phase(&self) -> Phase {
        if self.generated.is_empty() {
            Phase::Prefill
        } else {
            Phase::Decode
        }
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
            request.state_handles.len(),
            request.state_access_modes.len()
        );
        let mut row_claims = std::collections::BTreeMap::new();
        for (handle, mode) in request
            .state_handles
            .iter()
            .cloned()
            .zip(request.state_access_modes.iter().copied())
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScheduledRow {
    phase: Phase,
    token_start: u32,
    token_count: NonZeroU32,
    sample: bool,
}

impl ScheduledRow {
    fn token_end(self) -> PyResult<u32> {
        self.token_start
            .checked_add(self.token_count.get())
            .ok_or_else(|| PyValueError::new_err("scheduled token range overflows u32"))
    }
}

struct InFlightRow {
    request: Request,
    scheduled: ScheduledRow,
}

struct PreparedProgress {
    apply: bool,
    prefill_cursor: u32,
    sampled_token: Option<u32>,
}

struct InFlightBatch {
    batch_id: u64,
    rows: Vec<InFlightRow>,
    scheduled_state: Vec<StateHandle>,
    state_access: Option<StateAccessLease>,
    writer_access: Option<WriterAccessLease>,
    execution: InFlightExecution,
}

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
    scheduler: Scheduler<Request>,
    scheduler_config: SchedulerConfig,
    context_length: NonZeroU32,
    eos_token_id: Option<u32>,
    engine_id: u64,
    next_request_id: u64,
    next_batch_id: u64,
    state_pool: Option<Py<StatePool>>,
    recurrent_state_mapping: Option<StateMappingConfig>,
    cache_layout: Option<CacheLayout>,
    cache_pool: Option<CachePool>,
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
        let (runner_execution, runner_release) = resolve_runner_contract(py, runner.bind(py))?;
        let runner_prefill_chunk_size = resolve_runner_prefill_chunk_size(py, runner.bind(py))?;

        let runner_resources = resolve_runner_resources(py, runner.bind(py), state_pool.as_ref())?;
        let cache_pool = runner_resources
            .cache_layout
            .as_ref()
            .map(|layout| {
                let layout = layout.borrow(py);
                CachePool::new(layout.parent_capacity_native().get(), &layout.packings())
                    .map_err(cache_pool_error)
            })
            .transpose()?;
        // Complete every fallible native allocation before the runner's one
        // resource-shape-specific binding callback publishes physical identity.
        let engine_id = mint_engine_id()?;
        let (cache_layout, recurrent_state_mapping) = bind_runner_resources(
            py,
            &runner_resources,
            cache_pool.as_ref().map(CachePool::id),
            state_pool.as_ref(),
        )?;

        Ok(Self {
            in_flight: None,
            runner: runner.clone_ref(py),
            runner_execution,
            runner_release,
            runner_prefill_chunk_size,
            scheduler: Scheduler::new(
                NonZeroUsize::new(max_running_requests).expect("validated positive batch limit"),
                scheduler_config.policy,
            ),
            scheduler_config,
            context_length,
            eos_token_id,
            engine_id,
            next_request_id: 1,
            next_batch_id: 1,
            state_pool,
            recurrent_state_mapping,
            cache_layout,
            cache_pool,
            poisoned_error: None,
            closed: false,
        })
    }

    #[pyo3(signature = (
        token_ids,
        max_new_tokens,
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
        max_new_tokens: usize,
        state_handles: Option<Vec<Py<StateHandle>>>,
        state_access_modes: Option<Vec<String>>,
        priority: i32,
        batch_key: Option<u64>,
    ) -> PyResult<Py<RequestLease>> {
        self.raise_if_open()?;
        if token_ids.is_empty() {
            return Err(PyValueError::new_err("token_ids must be non-empty"));
        }
        if max_new_tokens == 0 {
            return Err(PyValueError::new_err("max_new_tokens must be positive"));
        }
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
            .map(|handle| handle.borrow(py).clone())
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

        let demands = self.cache_layout.as_ref().map(|layout| {
            layout
                .groups_native()
                .iter()
                .enumerate()
                .map(|(group_index, group)| {
                    let blocks = match group.retention_native() {
                        CacheRetention::History => {
                            max_input_tokens.div_ceil(group.block_tokens_native().get())
                        }
                        CacheRetention::Window { .. } => group
                            .window_resident_blocks()
                            .expect("window retention has a resident width")
                            .get(),
                    };
                    GroupDemand::new(GroupId::new(group_index as u32), blocks as usize)
                })
                .collect::<Vec<_>>()
        });
        let mut pending_pages = match (demands.as_deref(), self.cache_pool.as_mut()) {
            (Some(demands), Some(pool)) => Some(
                pool.reserve_request(SequenceId::new(request_id), demands)
                    .map_err(cache_pool_error)?,
            ),
            (None, None) => None,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "Engine cache layout and cache pool disagree",
                ));
            }
        };

        if !state_handles.is_empty() {
            let Some(state_pool) = &self.state_pool else {
                if let Some(pending) = pending_pages.take() {
                    self.cache_pool
                        .as_mut()
                        .expect("pending cache pages keep their pool")
                        .abort_request(pending);
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
                if let Some(pending) = pending_pages.take() {
                    self.cache_pool
                        .as_mut()
                        .expect("pending cache pages keep their pool")
                        .abort_request(pending);
                }
                return Err(error);
            }
        }

        // Every fallible admission step is complete. Promotion is an
        // allocation-free publication of all declared History groups.
        let pages = pending_pages.map(|pending| {
            self.cache_pool
                .as_mut()
                .expect("pending cache pages keep their pool")
                .promote_request(pending)
        });
        self.next_request_id = next_request_id;
        self.scheduler.enqueue(Request {
            id: request_id,
            lease: lease_state,
            priority,
            batch_key,
            turn_ordinal: 0,
            prompt: token_ids,
            prefill_cursor: 0,
            generated: Vec::new(),
            max_new_tokens,
            state_handles,
            state_access_modes,
            pages,
            cancel_pending: false,
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
            let compatible = state_compatible_prefix_len(&selected_requests);
            debug_assert_ne!(compatible, 0);
            selected_requests.truncate(compatible);
            selection.truncate(compatible);
            let scheduled_rows = schedule_ready_prefix_for_layout(
                &selected_requests,
                self.scheduler_config
                    .with_runner_prefill_limit(self.runner_prefill_chunk_size),
                self.cache_layout.as_ref(),
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
        let scheduled = self.scheduler.take_selected(selection);

        let plan = match build_plan_for_engine(
            batch_id,
            Some(self.engine_id),
            &scheduled,
            &scheduled_rows,
            self.cache_layout.as_ref(),
            self.cache_pool.as_ref(),
        ) {
            Ok(plan) => plan,
            Err(error) => {
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let plan = match Py::new(py, plan) {
            Ok(plan) => plan,
            Err(error) => {
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let scheduled_state = scheduled
            .iter()
            .flat_map(|request| request.state_handles.iter().cloned())
            .collect::<Vec<_>>();
        let prepared_state_access = match self.prepare_state_access(py, &scheduled) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let writer_claims = match scheduled_writer_access_claims(
            &scheduled,
            &scheduled_rows,
            self.cache_layout.as_ref(),
            self.cache_pool.as_ref(),
        ) {
            Ok(claims) => claims,
            Err(error) => {
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let prepared_writer_access = match (self.cache_pool.as_ref(), writer_claims.is_empty()) {
            (Some(pool), _) => match pool.prepare_writer_access(writer_claims) {
                Ok(prepared) => Some(prepared),
                Err(error) => {
                    self.scheduler.restore_rejected(scheduled);
                    return Err(cache_pool_error(error));
                }
            },
            (None, true) => None,
            (None, false) => {
                self.scheduler.restore_rejected(scheduled);
                return Err(PyRuntimeError::new_err(
                    "cache writer claims exist without a cache pool",
                ));
            }
        };

        if let Err(error) = self.pin_state(py, &scheduled_state) {
            self.scheduler.restore_rejected(scheduled);
            return Err(error);
        }
        let state_access = match self.acquire_state_access(py, prepared_state_access) {
            Ok(access) => access,
            Err(error) => {
                self.unpin_after_error(py, &scheduled_state);
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let writer_access = match prepared_writer_access {
            Some(prepared) => match self
                .cache_pool
                .as_mut()
                .expect("prepared cache access keeps its pool")
                .acquire_writer_access(prepared)
            {
                Ok(access) => Some(access),
                Err(error) => {
                    self.release_state_access(py, state_access);
                    self.unpin_after_error(py, &scheduled_state);
                    self.scheduler.restore_rejected(scheduled);
                    return Err(cache_pool_error(error));
                }
            },
            None => None,
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
                    batch_id,
                    rows: scheduled
                        .into_iter()
                        .zip(scheduled_rows)
                        .map(|(request, scheduled)| InFlightRow { request, scheduled })
                        .collect(),
                    scheduled_state,
                    state_access,
                    writer_access,
                    execution: InFlightExecution::FailedAfterAcceptance,
                };
                return self.quarantine(batch, stage, error);
            }
            RunnerCallOutcome::Rejected(error) => {
                if let Some(access) = writer_access {
                    self.cache_pool
                        .as_mut()
                        .expect("writer access keeps its cache pool")
                        .release_writer_access(access);
                }
                self.release_state_access(py, state_access);
                self.unpin_after_error(py, &scheduled_state);
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };

        self.in_flight = Some(InFlightBatch {
            batch_id,
            rows: scheduled
                .into_iter()
                .zip(scheduled_rows)
                .map(|(request, scheduled)| InFlightRow { request, scheduled })
                .collect(),
            scheduled_state,
            state_access,
            writer_access,
            execution,
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
            .flat_map(|(row, _)| row.request.state_handles.iter().cloned())
            .collect::<Vec<_>>();
        for (row, retire) in batch.rows.iter().zip(&retired) {
            if !*retire {
                continue;
            }
            match (self.cache_pool.as_ref(), row.request.pages.as_ref()) {
                (Some(pool), Some(pages)) => {
                    if let Err(error) = pool.validate_request(pages) {
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
        if let Err(error) = self.unpin_state(py, &batch.scheduled_state) {
            return self.quarantine(batch, "native completion unpin failed", error);
        }

        // Runner cleanup and generic-state unpinning prove physical execution
        // is terminal. Only now may the single cache writer lease return.
        if let Some(access) = batch.writer_access.take() {
            self.cache_pool
                .as_mut()
                .expect("writer access keeps its cache pool")
                .release_writer_access(access);
        }

        if let Err(error) = self.release_state(py, &finished_state) {
            return self.quarantine(batch, "native state retirement failed", error);
        }
        let state_access = batch.state_access.take();
        self.release_state_access(py, state_access);

        for (row_index, retire) in retired.iter().copied().enumerate() {
            if !retire {
                continue;
            }
            let Some(pages) = batch.rows[row_index].request.pages.take() else {
                continue;
            };
            let pool = self
                .cache_pool
                .as_mut()
                .expect("request cache pages keep their pool");
            if let Err(failure) = pool.retire_request(pages) {
                batch.rows[row_index].request.pages = Some(failure.request);
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
                complete_request_lease(&request.lease, request.generated, request.cancel_pending);
            } else {
                survivors.push(request);
            }
        }
        self.scheduler.requeue_survivors(survivors);
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
            match (self.cache_pool.as_ref(), request.pages.as_ref()) {
                (Some(pool), Some(pages)) => {
                    if let Err(error) = pool.validate_request(pages) {
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
            if let Err(error) = self.release_state(py, &request.state_handles) {
                self.scheduler.restore(index, request);
                self.poison("native state cancellation retirement failed", &error);
                return Err(error);
            }
            if let Some(pages) = request.pages.take()
                && let Err(failure) = self
                    .cache_pool
                    .as_mut()
                    .expect("request cache pages keep their pool")
                    .retire_request(pages)
            {
                request.pages = Some(failure.request);
                self.scheduler.restore(index, request);
                let error = cache_pool_error(failure.error);
                self.poison(
                    "native cache cancellation retirement failed after preflight",
                    &error,
                );
                return Err(error);
            }
            complete_request_lease(&request.lease, request.generated, true);
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
        debug_assert_eq!(self.scheduler_config.policy, self.scheduler.policy());
        Py::new(py, self.scheduler_config)
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
        self.cache_pool.as_ref().map(CachePool::id)
    }

    #[getter]
    fn cache_parent_capacity(&self) -> Option<u32> {
        self.cache_layout
            .as_ref()
            .map(|layout| layout.parent_capacity_native().get())
    }

    #[getter]
    fn cache_free_parent_count(&self) -> Option<usize> {
        self.cache_pool.as_ref().map(CachePool::free_parent_count)
    }

    #[getter]
    fn cache_occupied_block_count(&self) -> Option<usize> {
        self.cache_pool
            .as_ref()
            .map(CachePool::occupied_block_count)
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
            self.cache_pool
                .as_ref()
                .map_or(0, CachePool::occupied_block_count)
        )
    }
}

impl Engine {
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
            match (self.cache_pool.as_ref(), request.pages.as_ref()) {
                (Some(pool), Some(pages)) => pool.validate_request(pages).map_err(cache_pool_error),
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
            .flat_map(|request| request.state_handles.iter().cloned())
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
            if let Some(pages) = request.pages.take()
                && let Err(failure) = self
                    .cache_pool
                    .as_mut()
                    .expect("request cache pages keep their pool")
                    .retire_request(pages)
            {
                let error = cache_pool_error(failure.error);
                self.poison(
                    "native cache retirement failed after close preflight",
                    &error,
                );
                return Err(error);
            }
            complete_request_lease(&request.lease, request.generated, true);
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
            if scheduled.phase != request.phase() || scheduled.token_start != context {
                return Err(PyRuntimeError::new_err(format!(
                    "request {} progress changed while its batch was in flight",
                    request.id
                )));
            }
            let token_end = scheduled.token_end()?;
            let expected_sample = match scheduled.phase {
                Phase::Prefill => token_end == request.prompt_length(),
                Phase::Decode => {
                    if scheduled.token_count.get() != 1 {
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
            if scheduled.phase == Phase::Prefill && token_end > request.prompt_length() {
                return Err(PyRuntimeError::new_err(
                    "scheduled prefill commit exceeds the request prompt",
                ));
            }
            if request.cancel_pending {
                prepared.push(PreparedProgress {
                    apply: false,
                    prefill_cursor: request.prefill_cursor,
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
                prefill_cursor: match scheduled.phase {
                    Phase::Prefill => token_end,
                    Phase::Decode => request.prompt_length(),
                },
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
            row.request.prefill_cursor = progress.prefill_cursor;
            if let Some(token_id) = progress.sampled_token {
                row.request.generated.push(token_id);
            }
        }
    }

    fn prepare_state_access(
        &self,
        py: Python<'_>,
        requests: &[Request],
    ) -> PyResult<Option<PreparedStateAccessLease>> {
        let claim_count = requests
            .iter()
            .try_fold(0usize, |count, request| {
                count.checked_add(request.state_handles.len())
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
                request.state_handles.len(),
                request.state_access_modes.len()
            );
            claims.extend(
                request
                    .state_handles
                    .iter()
                    .cloned()
                    .zip(request.state_access_modes.iter().copied())
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

    fn acquire_state_access(
        &self,
        py: Python<'_>,
        prepared: Option<PreparedStateAccessLease>,
    ) -> PyResult<Option<StateAccessLease>> {
        let Some(prepared) = prepared else {
            return Ok(None);
        };
        self.state_pool
            .as_ref()
            .expect("prepared state access keeps its pool")
            .bind(py)
            .borrow_mut()
            .acquire_engine_access(prepared)
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

    fn pin_state(&self, py: Python<'_>, handles: &[StateHandle]) -> PyResult<()> {
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

    fn unpin_state(&self, py: Python<'_>, handles: &[StateHandle]) -> PyResult<()> {
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

    fn release_state(&self, py: Python<'_>, handles: &[StateHandle]) -> PyResult<()> {
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

    fn unpin_after_error(&self, py: Python<'_>, handles: &[StateHandle]) {
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
        BatchPlan, CacheGroupLayout, CacheGroupPlan, CacheLayout, Engine, RequestLease,
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

    fn request(id: u64, prompt: &[u32], generated: &[u32], pages: Option<RequestPages>) -> Request {
        Request {
            id,
            lease: Arc::new(Mutex::new(RequestLeaseState::default())),
            priority: 0,
            batch_key: None,
            turn_ordinal: 0,
            prompt: prompt.to_vec(),
            prefill_cursor: if generated.is_empty() {
                0
            } else {
                prompt.len() as u32
            },
            generated: generated.to_vec(),
            max_new_tokens: 8,
            state_handles: Vec::new(),
            state_access_modes: Vec::new(),
            pages,
            cancel_pending: false,
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
    fn mixed_unmanaged_plan_keeps_base_batch_contract() {
        let prefill = request(1, &[10, 11, 12], &[], None);
        let decode = request(2, &[20, 21], &[99], None);
        let scheduled = [&prefill, &decode];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap();
        let plan = build_plan(7, &scheduled, &rows, None, None).unwrap();

        assert_eq!(plan.mode, "mixed");
        assert_eq!(plan.token_ids, vec![10, 11, 12, 99]);
        assert_eq!(plan.positions, vec![0, 1, 2, 2]);
        assert_eq!(plan.sequence_ids, vec![1, 2]);
        assert_eq!(plan.sequence_indptr, vec![0, 3, 4]);
        assert_eq!(plan.prefill_sequence_indices, vec![0]);
        assert_eq!(plan.decode_sequence_indices, vec![1]);
        assert_eq!(plan.sample_sequence_indices, vec![0, 1]);
        assert_eq!(plan.cache_pool_id, None);
        assert!(plan.cache_groups.is_empty());
    }

    #[test]
    fn history_plan_is_sparse_and_writer_claims_match_active_pages() {
        let layout = history_layout();
        let mut pool =
            CachePool::new(layout.parent_capacity_native().get(), &layout.packings()).unwrap();
        let pending = pool
            .reserve_request(
                SequenceId::new(41),
                &[
                    GroupDemand::new(GroupId::new(0), 3),
                    GroupDemand::new(GroupId::new(1), 5),
                ],
            )
            .unwrap();
        let pages = pool.promote_request(pending);
        let mut managed = request(41, &[1, 2, 3, 4, 5, 6], &[], Some(pages));
        managed.prefill_cursor = 4;
        let scheduled = [&managed];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap();
        let plan = build_plan(9, &scheduled, &rows, Some(&layout), Some(&pool)).unwrap();

        assert_eq!(plan.cache_pool_id, Some(pool.id()));
        assert_eq!(plan.token_ids, vec![5, 6]);
        assert_eq!(plan.positions, vec![4, 5]);
        assert_eq!(plan.cache_groups.len(), 2);
        let full = &plan.cache_groups[0];
        assert_eq!(full.block_indptr_native(), &[0, 2]);
        assert_eq!(full.block_ids_native().len(), 2);
        assert_eq!(full.first_logical_blocks_native(), &[0]);
        assert_eq!(full.append_block_offsets_native(), &[0, 1]);
        let fine = &plan.cache_groups[1];
        assert_eq!(fine.block_indptr_native(), &[0, 3]);
        assert_eq!(fine.block_ids_native().len(), 3);
        assert_eq!(fine.append_block_offsets_native(), &[0, 1]);

        let claims =
            scheduled_writer_access_claims(&scheduled, &rows, Some(&layout), Some(&pool)).unwrap();
        assert_eq!(claims.len(), 5);
        let prepared = pool.prepare_writer_access(claims).unwrap();
        let lease = pool.acquire_writer_access(prepared).unwrap();
        assert_eq!(lease.len(), 5);
        pool.release_writer_access(lease);

        let pages = managed.pages.take().unwrap();
        pool.validate_request(&pages).unwrap();
        pool.retire_request(pages).unwrap();
        assert_eq!(pool.occupied_block_count(), 0);
        assert_eq!(pool.free_parent_count(), 8);
    }

    #[test]
    fn window_plan_is_chronological_and_reuses_only_an_evicted_block() {
        let layout = window_layout();
        let mut pool =
            CachePool::new(layout.parent_capacity_native().get(), &layout.packings()).unwrap();
        let pending = pool
            .reserve_request(SequenceId::new(51), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let pages = pool.promote_request(pending);
        let physical = pages.groups()[0]
            .pages()
            .iter()
            .map(|page| pool.kernel_page_id(page.handle()).unwrap())
            .collect::<Vec<_>>();
        let mut managed = request(
            51,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[],
            Some(pages),
        );
        managed.prefill_cursor = 8;
        let scheduled = [&managed];
        let rows =
            schedule_ready_prefix_for_layout(&scheduled, SchedulerConfig::default(), Some(&layout))
                .unwrap();
        assert_eq!(rows[0].token_start, 8);
        assert_eq!(rows[0].token_count.get(), 4);

        let plan = build_plan(10, &scheduled, &rows, Some(&layout), Some(&pool)).unwrap();
        let window = &plan.cache_groups[0];
        assert_eq!(window.first_logical_blocks_native(), &[1]);
        assert_eq!(window.block_indptr_native(), &[0, 2]);
        assert_eq!(window.block_ids_native(), &[physical[1], physical[0]]);
        assert_eq!(
            window.append_block_ids_native(),
            &[physical[0], physical[0], physical[0], physical[0]]
        );
        assert_eq!(window.append_block_offsets_native(), &[0, 1, 2, 3]);

        let claims =
            scheduled_writer_access_claims(&scheduled, &rows, Some(&layout), Some(&pool)).unwrap();
        assert_eq!(claims.len(), 2);
        let prepared = pool.prepare_writer_access(claims).unwrap();
        let lease = pool.acquire_writer_access(prepared).unwrap();
        pool.release_writer_access(lease);

        pool.retire_request(managed.pages.take().unwrap()).unwrap();
        assert_eq!(pool.occupied_block_count(), 0);
    }

    #[test]
    fn window_prefill_never_crosses_a_logical_block_boundary() {
        let layout = window_layout();
        let mut first = request(1, &[1, 2, 3, 4, 5, 6], &[], None);
        first.prefill_cursor = 3;
        let second = request(2, &[7, 8, 9, 10, 11, 12], &[], None);
        let scheduled = [&first, &second];
        let rows =
            schedule_ready_prefix_for_layout(&scheduled, SchedulerConfig::default(), Some(&layout))
                .unwrap();
        assert_eq!(rows[0].token_count.get(), 1);
        assert_eq!(rows[1].token_count.get(), 4);
    }

    #[test]
    fn batch_plan_rejects_noncanonical_cache_group_order() {
        let unmanaged = request(1, &[1, 2], &[], None);
        let scheduled = [&unmanaged];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default()).unwrap();
        let mut plan = build_plan(1, &scheduled, &rows, None, None).unwrap();
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
        assert_eq!(rows[0].token_count.get(), 2);
        assert_eq!(rows[1].token_count.get(), 1);
        assert!(!rows[0].sample);
        assert!(!rows[1].sample);
    }
}
