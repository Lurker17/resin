use std::cmp::Reverse;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use numpy::{PyArray1, PyArrayLike1};
use pyo3::exceptions::{PyAttributeError, PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

mod arrays;
mod cache;
mod http;
mod planning;
mod runner;

use cache::{CacheGroupFamily, CacheGroupLayout, CacheGroupPlan, CacheLayout, CacheRetention};
use planning::{build_plan_for_engine, schedule_ready_prefix};
use resin_core::resources::RequestResourceRef;
use resin_core::resources::paged::{CachePoolError, RequestPages, WriterAccessLease};
use resin_core::resources::paged_kv::{PagedKvError, PagedKvResources};
use resin_core::{Phase, RequestId, Schedulable, Scheduler, SequenceStep};
use runner::{
    ResourceBinding, RunnerConfig, bind_runner_resources, resolve_runner_config,
    resolve_runner_contract, runner_error_is_post_accept,
};

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);

fn default_sampling_seed(engine_id: u64, request_id: u64) -> i64 {
    // SplitMix64 gives stable, well-distributed row seeds without adding a
    // control-plane RNG or making batch composition part of randomness.
    let mut value = engine_id ^ request_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) & i64::MAX as u64) as i64
}

fn mint_engine_id() -> PyResult<u64> {
    NEXT_ENGINE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| PyRuntimeError::new_err("engine id space is exhausted"))
}

/// Native admission and token-budget policy for an Engine.
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

/// Per-request output-token sampling policy.
///
/// A zero temperature selects the maximum-logit token. Positive temperatures
/// enable stochastic sampling after optional top-k and top-p filtering.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct SamplingParams {
    temperature: f64,
    top_p: f64,
    top_k: Option<u32>,
    seed: Option<i64>,
    extra: Option<Arc<Py<PyAny>>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            seed: None,
            extra: None,
        }
    }
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (*, temperature = 1.0, top_p = 1.0, top_k = None, seed = None, extra = None))]
    fn new(
        temperature: f64,
        top_p: f64,
        top_k: Option<u32>,
        seed: Option<i64>,
        extra: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(PyValueError::new_err(
                "temperature must be finite and non-negative",
            ));
        }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
            return Err(PyValueError::new_err(
                "top_p must be finite and in the interval (0, 1]",
            ));
        }
        if top_k == Some(0) {
            return Err(PyValueError::new_err(
                "top_k must be positive when supplied",
            ));
        }
        if seed.is_some_and(|seed| seed < 0) {
            return Err(PyValueError::new_err(
                "seed must be non-negative when supplied",
            ));
        }
        Ok(Self {
            temperature,
            top_p,
            top_k,
            seed,
            extra: extra.map(Arc::new),
        })
    }

    #[getter]
    fn temperature(&self) -> f64 {
        self.temperature
    }

    #[getter]
    fn top_p(&self) -> f64 {
        self.top_p
    }

    #[getter]
    fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    #[getter]
    fn seed(&self) -> Option<i64> {
        self.seed
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.extra
            .as_ref()
            .map(|extra| extra.as_ref().clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingParams(temperature={:?}, top_p={:?}, top_k={:?}, seed={:?}, extra={})",
            self.temperature,
            self.top_p,
            self.top_k,
            self.seed,
            if self.extra.is_some() {
                "<object>"
            } else {
                "None"
            }
        )
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

/// Immutable execution plan for one prefill or decode step.
///
/// This runs one forward pass on a batch of sequences. `token_ids` contains
/// the newly scheduled input tokens, packed by sequence into one flat array;
/// `token_indptr` is the CSR row pointer, so sequence `i` owns
/// `token_ids[token_indptr[i]:token_indptr[i + 1]]`.
///
/// `context_lengths[i]` is the number of tokens processed before this step,
/// so that sequence's new tokens occupy consecutive absolute positions
/// beginning there. `cache_groups` contains one placement plan per declared
/// `CacheLayout` group, and every group plan covers every sequence.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct BatchPlan {
    sequence_ids: Vec<u64>,
    context_lengths: Vec<u32>,
    token_ids: Vec<u32>,
    token_indptr: Vec<u32>,
    phase: BatchPhase,
    batch_id: u64,
    batch_key: Option<u64>,
    cache_groups: Vec<CacheGroupPlan>,
    sampling_mask: Vec<bool>,
    sampling_params: Vec<SamplingParams>,
    media: Vec<Option<Arc<Py<PyAny>>>>,
    positions: OnceLock<Vec<u32>>,
}

impl BatchPlan {
    fn checked(plan: Self) -> PyResult<Self> {
        let Self {
            token_ids,
            sequence_ids,
            token_indptr,
            context_lengths,
            sampling_params,
            sampling_mask,
            media,
            ..
        } = &plan;
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
        if sequence_ids.is_empty() {
            return Err(PyValueError::new_err(
                "a batch must contain at least one sequence",
            ));
        }
        if token_indptr.len() != sequence_ids.len() + 1 {
            return Err(PyValueError::new_err(
                "token_indptr must contain one more item than sequence_ids",
            ));
        }
        if context_lengths.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "context_lengths must contain one item per sequence",
            ));
        }
        if sampling_params.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "sampling_params must contain one item per sequence",
            ));
        }
        if sampling_params.iter().any(|params| params.seed.is_none()) {
            return Err(PyValueError::new_err(
                "sampling_params must contain an effective seed for every sequence",
            ));
        }
        if sampling_mask.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "sampling_mask must contain one item per sequence",
            ));
        }
        if media.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "media must contain one item per sequence",
            ));
        }
        if token_indptr.first() != Some(&0) {
            return Err(PyValueError::new_err("token_indptr must start at zero"));
        }
        if token_indptr.last().copied() != Some(token_ids.len() as u32) {
            return Err(PyValueError::new_err(
                "token_indptr must end at the token count",
            ));
        }
        if token_indptr.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(
                "each sequence must contain at least one token",
            ));
        }
        if context_lengths
            .iter()
            .zip(token_indptr.windows(2))
            .any(|(context, range)| context.checked_add(range[1] - range[0]).is_none())
        {
            return Err(PyValueError::new_err(
                "scheduled positions exceed the u32 metadata range",
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
        if self.cache_groups.is_empty() {
            return Ok(());
        }

        for (group_index, group) in self.cache_groups.iter().enumerate() {
            group.validate_local()?;
            let group = group.inner_native();
            if group.group_index != group_index as u32 {
                return Err(PyValueError::new_err(
                    "cache_groups must be ordered by contiguous group_index",
                ));
            }
            if group.block_table_start_indices.len() != self.sequence_ids.len() {
                return Err(PyValueError::new_err(
                    "each cache group must contain one block-table segment per sequence",
                ));
            }
            if group.append_slab_ids.len() != self.token_ids.len()
                || group.append_slotnums.len() != self.token_ids.len()
            {
                return Err(PyValueError::new_err(
                    "each cache group must contain one append target per batch token",
                ));
            }
            for sequence_index in 0..self.sequence_ids.len() {
                let block_start = group.block_indptr[sequence_index] as usize;
                let block_end = group.block_indptr[sequence_index + 1] as usize;
                let row_slabs = &group.block_slab_ids[block_start..block_end];
                let row_slotnums = &group.block_slotnums[block_start..block_end];
                let token_start = self.token_indptr[sequence_index] as usize;
                let token_end = self.token_indptr[sequence_index + 1] as usize;
                if group.append_slab_ids[token_start..token_end]
                    .iter()
                    .zip(&group.append_slotnums[token_start..token_end])
                    .any(|(append_slab, append_slotnum)| {
                        !row_slabs
                            .iter()
                            .zip(row_slotnums)
                            .any(|(slab, slotnum)| slab == append_slab && slotnum == append_slotnum)
                    })
                {
                    return Err(PyValueError::new_err(
                        "cache append targets must belong to their sequence's block table",
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
        sequence_ids,
        context_lengths,
        token_ids,
        token_indptr,
        *,
        phase = BatchPhase::Prefill,
        batch_id = 0,
        batch_key = None,
        cache_groups = None,
        sampling_mask = None,
        sampling_params = None,
        media = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        sequence_ids: PyArrayLike1<'_, u64>,
        context_lengths: PyArrayLike1<'_, u32>,
        token_ids: PyArrayLike1<'_, u32>,
        token_indptr: PyArrayLike1<'_, u32>,
        phase: BatchPhase,
        batch_id: u64,
        batch_key: Option<u64>,
        cache_groups: Option<Vec<Py<CacheGroupPlan>>>,
        sampling_mask: Option<PyArrayLike1<'_, bool>>,
        sampling_params: Option<Vec<Py<SamplingParams>>>,
        media: Option<Vec<Option<Py<PyAny>>>>,
    ) -> PyResult<Self> {
        let sequence_ids = arrays::owned_vec(sequence_ids);
        let context_lengths = arrays::owned_vec(context_lengths);
        let token_ids = arrays::owned_vec(token_ids);
        let token_indptr = arrays::owned_vec(token_indptr);
        let sequence_count = sequence_ids.len();
        let cache_groups = cache_groups
            .unwrap_or_default()
            .into_iter()
            .map(|group| group.borrow(py).clone())
            .collect();
        let sampling_mask = sampling_mask
            .map(arrays::owned_vec)
            .unwrap_or_else(|| vec![true; sequence_count]);
        let mut sampling_params = sampling_params
            .map(|params| {
                params
                    .into_iter()
                    .map(|params| params.borrow(py).clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![SamplingParams::default(); sequence_count]);
        for (params, sequence_id) in sampling_params.iter_mut().zip(&sequence_ids) {
            params
                .seed
                .get_or_insert_with(|| default_sampling_seed(0, *sequence_id));
        }
        let media = media
            .unwrap_or_else(|| (0..sequence_count).map(|_| None).collect())
            .into_iter()
            .map(|payload| payload.map(Arc::new))
            .collect();
        Self::checked(Self {
            sequence_ids,
            context_lengths,
            token_ids,
            token_indptr,
            phase,
            batch_id,
            batch_key,
            cache_groups,
            sampling_mask,
            sampling_params,
            media,
            positions: OnceLock::new(),
        })
    }

    /// Stable request identifier for each sequence.
    #[getter]
    fn sequence_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u64>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().sequence_ids)
    }

    /// Number of tokens already processed for each sequence.
    #[getter]
    fn context_lengths<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().context_lengths)
    }

    /// Flattened scheduled token IDs, grouped by sequence.
    #[getter]
    fn token_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().token_ids)
    }

    /// Offsets into `token_ids` for each sequence.
    #[getter]
    fn token_indptr<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().token_indptr)
    }

    /// Absolute position of every flattened scheduled token.
    #[getter]
    fn positions<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        let plan = this.get();
        let positions = plan.positions.get_or_init(|| {
            let mut positions = Vec::with_capacity(plan.token_ids.len());
            for (context, range) in plan
                .context_lengths
                .iter()
                .zip(plan.token_indptr.windows(2))
            {
                positions.extend(*context..*context + range[1] - range[0]);
            }
            positions
        });
        arrays::readonly_view(this.clone().into_any(), positions)
    }

    /// Whether this is a prefill or decode batch. Mixed-chunk is not supported.
    #[getter]
    fn phase(&self) -> BatchPhase {
        self.phase
    }

    /// Monotonic identifier used to match an async dispatch with its result.
    #[getter]
    fn batch_id(&self) -> u64 {
        self.batch_id
    }

    /// Optional key that prevents incompatible requests from sharing a batch.
    #[getter]
    fn batch_key(&self) -> Option<u64> {
        self.batch_key
    }

    /// Ordered physical placement plans corresponding to CacheLayout groups.
    /// There is one plan per layout group, not one plan per sequence; each plan
    /// contains one block-table row for every sequence in this batch.
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

    /// Which sequences should generate a sampled token. Used for chunked
    /// prefill, when some sequences only advance the cache.
    #[getter]
    fn sampling_mask<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<bool>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().sampling_mask)
    }

    /// Effective sampling parameters, one per sequence.
    #[getter]
    fn sampling_params(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let params = self
            .sampling_params
            .iter()
            .cloned()
            .map(|params| Py::new(py, params))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, params)?.unbind())
    }

    /// Opaque runner inputs, one per sequence. The scheduler never inspects them.
    #[getter]
    fn media(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let payloads = self
            .media
            .iter()
            .map(|payload| {
                payload
                    .as_ref()
                    .map_or_else(|| py.None(), |payload| payload.clone_ref(py))
            })
            .collect::<Vec<_>>();
        Ok(PyTuple::new(py, payloads)?.unbind())
    }

    /// Number of sequences (batch size).
    #[getter]
    fn num_sequences(&self) -> usize {
        self.sequence_ids.len()
    }

    /// Number of flattened scheduled tokens.
    #[getter]
    fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Return the half-open range in `token_ids` for one sequence.
    fn token_range(&self, sequence_index: usize) -> PyResult<(usize, usize)> {
        if sequence_index >= self.sequence_ids.len() {
            return Err(PyValueError::new_err("sequence index is out of range"));
        }
        Ok((
            self.token_indptr[sequence_index] as usize,
            self.token_indptr[sequence_index + 1] as usize,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchPlan(batch_id={}, phase=BatchPhase.{}, sequences={}, tokens={}, cache_groups={})",
            self.batch_id,
            self.phase.as_str(),
            self.sequence_ids.len(),
            self.token_ids.len(),
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
    finish_reason: Option<CompletionReason>,
}

#[derive(Clone, Copy, Debug)]
enum CompletionReason {
    Stop,
    Length,
}

impl CompletionReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Debug, Default)]
struct RequestLeaseState {
    done: bool,
    cancelled: bool,
    cached_tokens: u32,
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

    /// Number of prompt tokens restored from the prefix cache at admission.
    #[getter]
    fn cached_tokens(&self) -> u32 {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_tokens
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

    #[getter]
    fn finish_reason(&self) -> Option<&'static str> {
        self.finish_reason.map(CompletionReason::as_str)
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenEvent(request_id={}, token_id={}, index={}, finished={}, finish_reason={:?})",
            self.request_id,
            self.token_id,
            self.index,
            self.finished,
            self.finish_reason.map(CompletionReason::as_str)
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
    running: bool,
    prompt: Vec<u32>,
    cache_prompt: Option<Vec<u32>>,
    atomic_prefill_spans: Vec<(u32, u32)>,
    resident_tokens: u32,
    generated: Vec<u32>,
    max_new_tokens: usize,
    stop_token_ids: Vec<u32>,
    sampling_params: SamplingParams,
    media: Option<Arc<Py<PyAny>>>,
    cancel_pending: bool,
    cache: Option<RequestPages>,
}

impl Request {
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

impl Schedulable for Request {
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

type ScheduledRow = SequenceStep;

struct BatchRow {
    request: Request,
    scheduled: ScheduledRow,
}

struct PreparedProgress {
    apply: bool,
    resident_tokens: u32,
    sampled_tokens: Vec<u32>,
}

fn extract_runner_token_rows(
    output: &Bound<'_, PyAny>,
    expected_rows: usize,
) -> PyResult<Vec<Vec<u32>>> {
    let flat_error = match output.extract::<Vec<u32>>() {
        Ok(token_ids) => {
            if token_ids.len() != expected_rows {
                return Err(PyRuntimeError::new_err(format!(
                    "runner returned {} token ids for {expected_rows} sequences",
                    token_ids.len()
                )));
            }
            return Ok(token_ids
                .into_iter()
                .map(|token_id| vec![token_id])
                .collect());
        }
        Err(error) => error,
    };

    let token_rows = output.extract::<Vec<Vec<u32>>>().map_err(|_| flat_error)?;
    if token_rows.len() != expected_rows {
        return Err(PyRuntimeError::new_err(format!(
            "runner returned {} token rows for {expected_rows} sequences",
            token_rows.len()
        )));
    }
    if let Some(row_index) = token_rows.iter().position(Vec::is_empty) {
        return Err(PyRuntimeError::new_err(format!(
            "runner returned an empty token row for sequence {row_index}"
        )));
    }
    Ok(token_rows)
}

struct Batch {
    rows: Vec<BatchRow>,
    resources: BatchResources,
}

type BatchResources = Option<WriterAccessLease>;

#[pyclass(module = "resin._resin")]
struct Engine {
    // A post-accept failure keeps the exact requests and writer capability
    // unreachable until the poisoned Engine itself is destroyed.
    quarantined: Option<Batch>,
    runner_forward: Py<PyAny>,
    runner_release: Option<Py<PyAny>>,
    runner_config: RunnerConfig,
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
    kv_resources: Option<PagedKvResources>,
    poisoned_error: Option<String>,
    closed: bool,
}

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
        runner,
        *,
        max_running_requests = 32,
        context_length = 32768,
        scheduler_config = None,
        eos_token_id = None,
        prefix_cache = false
    ))]
    fn new(
        py: Python<'_>,
        runner: Py<PyAny>,
        max_running_requests: usize,
        context_length: u32,
        scheduler_config: Option<Py<SchedulerConfig>>,
        eos_token_id: Option<u32>,
        prefix_cache: bool,
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
        let (runner_forward, runner_release) = resolve_runner_contract(py, runner.bind(py))?;
        let runner_config = resolve_runner_config(py, runner.bind(py))?;
        let kv_resources = runner_config
            .config
            .cache_layout
            .as_ref()
            .map(|layout| {
                let layout = layout.borrow(py);
                if prefix_cache {
                    PagedKvResources::with_prefix_cache(layout.inner_native().clone())
                } else {
                    PagedKvResources::new(layout.inner_native().clone())
                }
                .map_err(cache_pool_error)
            })
            .transpose()?;
        // Complete every fallible native allocation before the runner's one
        // resource-shape-specific binding callback publishes physical identity.
        let engine_id = mint_engine_id()?;
        let runner_config = bind_runner_resources(
            py,
            runner_config,
            kv_resources.as_ref().map(|resources| resources.pool().id()),
        )?;

        Ok(Self {
            quarantined: None,
            runner_forward,
            runner_release,
            runner_config,
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
            kv_resources,
            poisoned_error: None,
            closed: false,
        })
    }

    #[pyo3(signature = (
        token_ids = None,
        max_new_tokens = None,
        *,
        priority = 0,
        batch_key = None,
        sampling_params = None,
        stop_token_ids = None,
        media = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &mut self,
        py: Python<'_>,
        token_ids: Option<Vec<u32>>,
        max_new_tokens: Option<usize>,
        priority: i32,
        batch_key: Option<u64>,
        sampling_params: Option<Py<SamplingParams>>,
        stop_token_ids: Option<Vec<u32>>,
        media: Option<Py<PyAny>>,
    ) -> PyResult<Py<RequestLease>> {
        self.raise_if_open()?;
        let token_ids = match token_ids {
            Some(token_ids) => token_ids,
            None => media
                .as_ref()
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "token_ids may be omitted only when media supplies token_ids",
                    )
                })?
                .getattr(py, "token_ids")?
                .extract(py)?,
        };
        if token_ids.is_empty() {
            return Err(PyValueError::new_err("token_ids must be non-empty"));
        }
        let cache_prompt: Option<Vec<u32>> = match media.as_ref() {
            Some(media) => match media.getattr(py, "cache_key") {
                Ok(identity) => Some(identity.extract::<Vec<u32>>(py)?),
                Err(error) if error.is_instance_of::<PyAttributeError>(py) => {
                    match media.getattr(py, "cache_token_ids") {
                        Ok(identity) => Some(identity.extract::<Vec<u32>>(py)?),
                        Err(error) if error.is_instance_of::<PyAttributeError>(py) => None,
                        Err(error) => return Err(error),
                    }
                }
                Err(error) => return Err(error),
            },
            None => None,
        };
        if cache_prompt
            .as_ref()
            .is_some_and(|identity| identity.len() != token_ids.len())
        {
            return Err(PyValueError::new_err(
                "media cache_key must match token_ids length",
            ));
        }
        let atomic_prefill_spans = match media.as_ref() {
            Some(media) => match media.getattr(py, "token_spans") {
                Ok(spans) => spans
                    .extract::<Vec<(u32, u32)>>(py)?
                    .into_iter()
                    .map(|(position, length)| (position, position.saturating_add(length)))
                    .collect(),
                Err(error) if error.is_instance_of::<PyAttributeError>(py) => {
                    match media.getattr(py, "prefill_atomic_spans") {
                        Ok(spans) => spans.extract::<Vec<(u32, u32)>>(py)?,
                        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Vec::new(),
                        Err(error) => return Err(error),
                    }
                }
                Err(error) => return Err(error),
            },
            None => Vec::new(),
        };
        let mut previous_end = 0;
        let effective_limits = self.scheduler_config.with_runner_limits(
            self.runner_config.prefill_chunk_size,
            self.runner_config.max_batch_tokens,
        );
        let atomic_limit = effective_limits
            .max_batch_tokens
            .map_or(effective_limits.prefill_chunk_size, |budget| {
                budget.min(effective_limits.prefill_chunk_size)
            })
            .get();
        for &(start, end) in &atomic_prefill_spans {
            if start >= end
                || end as usize > token_ids.len()
                || start < previous_end
                || end - start > atomic_limit
            {
                return Err(PyValueError::new_err(
                    "media token_spans must be sorted, disjoint, within the prompt, and fit one prefill row",
                ));
            }
            previous_end = end;
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
        let mut sampling_params = sampling_params
            .map(|params| params.borrow(py).clone())
            .unwrap_or_default();
        sampling_params
            .seed
            .get_or_insert_with(|| default_sampling_seed(self.engine_id, request_id));
        let mut stop_token_ids =
            stop_token_ids.unwrap_or_else(|| self.eos_token_id.into_iter().collect::<Vec<_>>());
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
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
        // Media may provide a same-length, digest-derived identity stream. A
        // media object without one stays out of the token-only prefix cache.
        let cache_identity = cache_prompt
            .as_deref()
            .or_else(|| media.is_none().then_some(token_ids.as_slice()));
        let resident_tokens = match (&mut self.kv_resources, &mut pages, cache_identity) {
            (Some(resources), Some(pages), Some(identity)) => resources
                .restore_prefix(pages, identity)
                .map_err(cache_pool_error)?,
            _ => 0,
        };
        lease_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_tokens = resident_tokens;
        self.next_request_id = next_request_id;
        self.scheduler.enqueue(Request {
            id: request_id,
            lease: lease_state,
            priority,
            batch_key,
            turn_ordinal: 0,
            running: false,
            prompt: token_ids,
            cache_prompt,
            atomic_prefill_spans,
            resident_tokens,
            generated: Vec::new(),
            max_new_tokens,
            stop_token_ids,
            sampling_params,
            media: media.map(Arc::new),
            cancel_pending: false,
            cache: pages,
        });
        Ok(lease)
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Vec<Py<TokenEvent>>> {
        self.raise_if_open()?;
        if self.scheduler.is_empty() {
            return Ok(Vec::new());
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
                return Ok(Vec::new());
            }
            let scheduled_rows = schedule_ready_prefix(
                &selected_requests,
                self.scheduler_config.with_runner_limits(
                    self.runner_config.prefill_chunk_size,
                    self.runner_config.max_batch_tokens,
                ),
                self.prefix_alignment().and_then(NonZeroU32::new),
            )?;
            if scheduled_rows.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "scheduler produced no token-bearing rows for a nonempty cohort",
                ));
            }
            if scheduled_rows
                .iter()
                .any(|row| row.phase == Phase::Prefill && !row.sample)
                && !self.runner_config.continuation_prefill
            {
                return Err(PyValueError::new_err(
                    "prefill_chunk_size creates continuation rows but runner_config.continuation_prefill is false",
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

        let growth_error = if let Some(resources) = self.kv_resources.as_mut() {
            let mut growth_error = None;
            for (request, row) in scheduled.iter_mut().zip(&scheduled_rows) {
                let scheduled_end = row
                    .token_end()
                    .ok_or_else(|| PyValueError::new_err("scheduled token range overflows u32"))?;
                let context = if row.phase == Phase::Decode {
                    scheduled_end
                        .checked_add(self.runner_config.cache_write_lookahead_tokens)
                        .ok_or_else(|| {
                            PyValueError::new_err("decode cache write lookahead overflows u32")
                        })?
                } else {
                    scheduled_end
                };
                let pages = request
                    .cache
                    .as_mut()
                    .expect("managed requests keep their cache state");
                if let Err(error) = resources.ensure_context(pages, row.token_start(), context) {
                    growth_error = Some(error);
                    break;
                }
            }
            growth_error
        } else {
            None
        };
        if let Some(error) = growth_error {
            self.scheduler.restore_rejected(scheduled);
            if matches!(error, CachePoolError::InsufficientCapacity { .. })
                && self.retract_one(py)?
            {
                return self.step(py);
            }
            return Err(cache_pool_error(error));
        }

        let mut acquired_kv = match self.kv_resources.as_mut() {
            Some(resources) => {
                let requests = scheduled
                    .iter()
                    .map(|request| RequestResourceRef {
                        request_id: core_request_id(request.id),
                        state: request
                            .cache
                            .as_ref()
                            .expect("managed requests keep their cache state"),
                    })
                    .collect::<Vec<_>>();
                match resources.acquire(
                    &requests,
                    &scheduled_rows,
                    self.runner_config.cache_write_lookahead_tokens,
                ) {
                    Ok(acquired) => Some(acquired),
                    Err(error) => {
                        self.scheduler.restore_rejected(scheduled);
                        return Err(paged_kv_error(error));
                    }
                }
            }
            None => {
                if scheduled.iter().any(|request| request.cache.is_some()) {
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
        let mut batch = Batch {
            rows: scheduled
                .into_iter()
                .zip(scheduled_rows)
                .map(|(request, scheduled)| BatchRow { request, scheduled })
                .collect(),
            resources: acquired_kv.map(|access| access.lease),
        };

        // The lease spans runner execution and native progress commit. A
        // post-accept error therefore quarantines the exact writer capability.
        let output = match self.runner_forward.bind(py).call1((plan.clone_ref(py),)) {
            Err(error) if runner_error_is_post_accept(py, &error) => {
                return self.quarantine(batch, "runner forward failed after accepting work", error);
            }
            Err(error) => {
                self.rollback_batch_resources(py, &mut batch.resources);
                self.scheduler
                    .restore_rejected(batch.rows.into_iter().map(|row| row.request).collect());
                return Err(error);
            }
            Ok(output) => output,
        };
        let output_token_ids = match extract_runner_token_rows(&output, selected) {
            Ok(output) => output,
            Err(error) => {
                return self.quarantine(
                    batch,
                    "runner forward protocol failure after accepting work",
                    error,
                );
            }
        };
        // Validate the complete execution result before terminal output limits
        // can hide writes beyond the leased capacity or extra prefill tokens.
        if let Err(error) = self.validate_runner_output(&batch.rows, &output_token_ids) {
            return self.quarantine(batch, "runner output exceeds its execution contract", error);
        }
        let decoded_batch = batch
            .rows
            .first()
            .is_some_and(|row| row.scheduled.phase == Phase::Decode);

        let mut retired = Vec::with_capacity(batch.rows.len());
        let mut retired_ids = Vec::new();
        let event_capacity = output_token_ids.iter().map(Vec::len).sum();
        let mut events = Vec::with_capacity(event_capacity);
        let mut committed_token_ids = Vec::with_capacity(batch.rows.len());
        for (row, token_ids) in batch.rows.iter().zip(&output_token_ids) {
            let request = &row.request;
            if request.cancel_pending {
                retired.push(true);
                retired_ids.push(request.id);
                committed_token_ids.push(token_ids.clone());
                continue;
            }
            if !row.scheduled.sample {
                retired.push(false);
                committed_token_ids.push(token_ids.clone());
                continue;
            }
            let remaining = request
                .max_new_tokens
                .saturating_sub(request.generated.len());
            let mut committed = Vec::with_capacity(token_ids.len().min(remaining));
            let mut finished = false;
            for &token_id in token_ids.iter().take(remaining) {
                let finish_reason = if request.stop_token_ids.binary_search(&token_id).is_ok() {
                    Some(CompletionReason::Stop)
                } else if request.generated.len() + committed.len() + 1 >= request.max_new_tokens {
                    Some(CompletionReason::Length)
                } else {
                    None
                };
                finished = finish_reason.is_some();
                let event = match Py::new(
                    py,
                    TokenEvent {
                        request_id: request.id,
                        token_id,
                        index: request.generated.len() + committed.len(),
                        finished,
                        finish_reason,
                    },
                ) {
                    Ok(event) => event,
                    Err(error) => {
                        return self.quarantine(batch, "token event construction failed", error);
                    }
                };
                events.push(event);
                committed.push(token_id);
                if finished {
                    break;
                }
            }
            retired.push(finished);
            if finished {
                retired_ids.push(request.id);
            }
            committed_token_ids.push(committed);
        }

        let prepared_progress =
            match self.prepare_scheduled_progress(&mut batch.rows, &committed_token_ids) {
                Ok(prepared) => prepared,
                Err(error) => {
                    return self.quarantine(batch, "native progress commit failed", error);
                }
            };

        // Validate every physical retirement and reserve all destination
        // storage before asking the runner or any allocator to release state.
        for (row, retire) in batch.rows.iter().zip(&retired) {
            if !*retire {
                continue;
            }
            match (self.kv_resources.as_ref(), row.request.cache.as_ref()) {
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
        // Runner cleanup proves physical execution is terminal. Only now may
        // the cache writer lease return.
        self.release_kv_access(batch.resources.take());

        if let Err(error) = self.publish_batch_prefixes(py, &mut batch, &prepared_progress) {
            return self.quarantine(batch, "native prefix publication failed", error);
        }

        for (row_index, retire) in retired.iter().copied().enumerate() {
            if !retire {
                continue;
            }
            let Some(pages) = batch.rows[row_index].request.cache.take() else {
                continue;
            };
            let resources = self
                .kv_resources
                .as_mut()
                .expect("request KV pages keep their resources");
            if let Err(failure) = resources.retire(pages) {
                batch.rows[row_index].request.cache = Some(failure.request);
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
                let Request {
                    lease,
                    generated,
                    cancel_pending,
                    ..
                } = request;
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
            match (self.kv_resources.as_ref(), request.cache.as_ref()) {
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
            if let Some(pages) = request.cache.take()
                && let Err(failure) = self
                    .kv_resources
                    .as_mut()
                    .expect("request KV pages keep their resources")
                    .retire(pages)
            {
                request.cache = Some(failure.request);
                self.scheduler.restore(index, request);
                let error = cache_pool_error(failure.error);
                self.poison(
                    "native cache cancellation retirement failed after preflight",
                    &error,
                );
                return Err(error);
            }
            let generated = std::mem::take(&mut request.generated);
            complete_request_lease(&request.lease, generated, true);
            return Ok(true);
        }

        Ok(false)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }
        self.raise_if_poisoned()?;
        self.close_healthy(py)
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
        !self.scheduler.is_empty()
    }

    #[getter]
    fn active_request_count(&self) -> usize {
        self.scheduler.len()
    }

    #[getter]
    fn ready_request_count(&self) -> usize {
        self.scheduler.len()
    }

    #[getter]
    fn max_running_requests(&self) -> usize {
        self.scheduler.max_running_requests()
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
    fn cache_layout(&self, py: Python<'_>) -> Option<Py<CacheLayout>> {
        self.runner_config
            .cache_layout
            .as_ref()
            .map(|layout| layout.clone_ref(py))
    }

    #[getter]
    fn cache_pool_id(&self) -> Option<u64> {
        self.kv_resources
            .as_ref()
            .map(|resources| resources.pool().id())
    }

    #[getter]
    fn cache_parent_capacity(&self) -> Option<u32> {
        self.runner_config
            .cache_layout
            .as_ref()
            .map(|layout| layout.get().inner_native().parent_capacity().get())
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

    /// `(entries, hits, misses, evictions)` when prefix caching is enabled.
    #[getter]
    fn prefix_cache_stats(&self) -> Option<(usize, u64, u64, u64)> {
        self.kv_resources
            .as_ref()
            .and_then(PagedKvResources::prefix_cache_metrics)
    }

    /// Drop all reusable cache prefixes.
    fn clear_prefix_cache(&mut self) -> PyResult<()> {
        if let Some(resources) = self.kv_resources.as_mut() {
            resources.clear_prefix_cache().map_err(cache_pool_error)?;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(ready_requests={}, max_running_requests={}, context_length={}, closed={}, poisoned={}, cache_blocks={})",
            self.ready_request_count(),
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
    fn prefix_alignment(&self) -> Option<u32> {
        self.kv_resources
            .as_ref()
            .and_then(PagedKvResources::prefix_alignment)
    }

    fn publish_batch_prefixes(
        &mut self,
        _py: Python<'_>,
        batch: &mut Batch,
        progress: &[PreparedProgress],
    ) -> PyResult<()> {
        let Some(alignment) = self.prefix_alignment() else {
            return Ok(());
        };
        for (row, progress) in batch.rows.iter_mut().zip(progress) {
            if !progress.apply
                || (row.request.media.is_some() && row.request.cache_prompt.is_none())
                || progress.resident_tokens == 0
                || progress.resident_tokens % alignment != 0
            {
                continue;
            }
            let resident = progress.resident_tokens as usize;
            let mut tokens = Vec::new();
            tokens
                .try_reserve_exact(
                    row.request
                        .prompt
                        .len()
                        .saturating_add(row.request.generated.len())
                        .saturating_add(progress.sampled_tokens.len()),
                )
                .map_err(|_| PyMemoryError::new_err("could not publish prefix token identity"))?;
            tokens.extend_from_slice(
                row.request
                    .cache_prompt
                    .as_deref()
                    .unwrap_or(&row.request.prompt),
            );
            tokens.extend_from_slice(&row.request.generated);
            tokens.extend_from_slice(&progress.sampled_tokens);
            if resident > tokens.len() {
                return Err(PyRuntimeError::new_err(
                    "resident KV progress exceeds the request token history",
                ));
            }
            tokens.truncate(resident);
            let resources = self
                .kv_resources
                .as_mut()
                .expect("prefix alignment requires managed cache resources");
            let pages = row
                .request
                .cache
                .as_mut()
                .expect("prefix-enabled requests own managed cache pages");
            resources
                .publish_prefix(pages, &tokens, progress.resident_tokens)
                .map_err(cache_pool_error)?;
        }
        Ok(())
    }

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
            .filter(|request| request.is_running() && request.cache.is_some())
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
            .cache
            .as_mut()
            .expect("retraction victim owns managed cache state");
        if let Err(error) = self
            .kv_resources
            .as_mut()
            .expect("managed cache state keeps its allocator")
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

    fn quarantine<T>(&mut self, batch: Batch, stage: &str, error: PyErr) -> PyResult<T> {
        self.quarantined = Some(batch);
        self.poison(stage, &error);
        Err(error)
    }

    fn close_healthy(&mut self, py: Python<'_>) -> PyResult<()> {
        self.retire_ready(py)?;
        if let Some(resources) = self.kv_resources.as_mut() {
            resources.clear_prefix_cache().map_err(cache_pool_error)?;
        }
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
            match (self.kv_resources.as_ref(), request.cache.as_ref()) {
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
        retired.extend(self.scheduler.drain());
        for mut request in retired {
            if let Some(pages) = request.cache.take()
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
            let generated = std::mem::take(&mut request.generated);
            complete_request_lease(&request.lease, generated, true);
        }
        Ok(())
    }

    fn prepare_scheduled_progress(
        &self,
        rows: &mut [BatchRow],
        output_token_ids: &[Vec<u32>],
    ) -> PyResult<Vec<PreparedProgress>> {
        if rows.len() != output_token_ids.len() {
            return Err(PyRuntimeError::new_err(
                "dense runner output does not cover every scheduled row",
            ));
        }
        let mut prepared = Vec::with_capacity(rows.len());
        for (row, token_ids) in rows.iter_mut().zip(output_token_ids) {
            let request = &mut row.request;
            let scheduled = row.scheduled;
            let context = request.logical_context()?;
            if scheduled.phase != request.phase() || scheduled.token_start() != context {
                return Err(PyRuntimeError::new_err(format!(
                    "request {} progress changed during its runner step",
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
            if token_ids.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "sampled runner output rows must contain at least one token",
                ));
            }
            if request.cancel_pending {
                prepared.push(PreparedProgress {
                    apply: false,
                    resident_tokens: request.resident_tokens,
                    sampled_tokens: Vec::new(),
                });
                continue;
            }
            if scheduled.sample {
                request
                    .generated
                    .try_reserve(token_ids.len())
                    .map_err(|_| {
                        PyMemoryError::new_err("could not reserve generated-token storage")
                    })?;
            }
            let resident_tokens = if scheduled.phase == Phase::Decode {
                let additional = u32::try_from(token_ids.len() - 1)
                    .map_err(|_| PyValueError::new_err("speculative token row is too long"))?;
                token_end.checked_add(additional).ok_or_else(|| {
                    PyValueError::new_err("request token position space is exhausted")
                })?
            } else {
                token_end
            };
            prepared.push(PreparedProgress {
                apply: true,
                resident_tokens,
                sampled_tokens: if scheduled.sample {
                    token_ids.clone()
                } else {
                    Default::default()
                },
            });
        }
        Ok(prepared)
    }

    fn apply_scheduled_progress(rows: &mut [BatchRow], prepared: Vec<PreparedProgress>) {
        assert_eq!(rows.len(), prepared.len());
        for (row, progress) in rows.iter_mut().zip(prepared) {
            if !progress.apply {
                continue;
            }
            row.request.resident_tokens = progress.resident_tokens;
            row.request.generated.extend(progress.sampled_tokens);
        }
    }

    fn validate_runner_output(&self, rows: &[BatchRow], token_rows: &[Vec<u32>]) -> PyResult<()> {
        for (row, tokens) in rows.iter().zip(token_rows) {
            if row.scheduled.phase == Phase::Prefill && tokens.len() != 1 {
                return Err(PyRuntimeError::new_err(
                    "speculative token rows are only supported during decode",
                ));
            }
            let maximum = u64::from(self.runner_config.cache_write_lookahead_tokens) + 1;
            if row.request.cache.is_some() && tokens.len() as u64 > maximum {
                return Err(PyRuntimeError::new_err(format!(
                    "runner returned {} decode tokens but leased cache space for at most {maximum}",
                    tokens.len(),
                )));
            }
        }
        Ok(())
    }

    fn release_kv_access(&mut self, access: Option<WriterAccessLease>) {
        if let Some(access) = access {
            self.kv_resources
                .as_mut()
                .expect("writer access keeps its KV resources")
                .release(access);
        }
    }

    fn rollback_batch_resources(&mut self, _py: Python<'_>, resources: &mut BatchResources) {
        self.release_kv_access(resources.take());
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.closed || self.poisoned_error.is_some() {
            return;
        }

        match Python::try_attach(|py| self.close_healthy(py)) {
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

#[pymodule(name = "_resin")]
mod _resin {
    #[pymodule_export]
    use super::{
        BatchPhase, BatchPlan, CacheGroupFamily, CacheGroupLayout, CacheGroupPlan, CacheLayout,
        CacheRetention, Engine, RequestLease, ResourceBinding, RunnerConfig, SamplingParams,
        SchedulerConfig, TokenEvent,
    };

    #[pymodule_export]
    use crate::http::NativeHttpServer;

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
            id,
            lease: Arc::new(Mutex::new(RequestLeaseState::default())),
            priority: 0,
            batch_key: None,
            turn_ordinal: 0,
            running: false,
            prompt: prompt.to_vec(),
            cache_prompt: None,
            atomic_prefill_spans: Vec::new(),
            resident_tokens: if generated.is_empty() {
                0
            } else {
                (prompt.len() + generated.len() - 1) as u32
            },
            generated: generated.to_vec(),
            max_new_tokens: 8,
            stop_token_ids: Vec::new(),
            sampling_params: SamplingParams {
                seed: Some(default_sampling_seed(0, id)),
                ..SamplingParams::default()
            },
            media: None,
            cancel_pending: false,
            cache: pages,
        }
    }

    fn history_layout() -> CacheLayout {
        CacheLayout::from_groups(
            8,
            vec![
                CacheGroupLayout::from_parts(
                    4,
                    1,
                    CacheRetention::History,
                    None,
                    CacheGroupFamily::History,
                )
                .unwrap(),
                CacheGroupLayout::from_parts(
                    2,
                    2,
                    CacheRetention::History,
                    None,
                    CacheGroupFamily::History,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn window_layout() -> CacheLayout {
        CacheLayout::from_groups(
            4,
            vec![
                CacheGroupLayout::from_parts(
                    4,
                    1,
                    CacheRetention::Window,
                    Some(4),
                    CacheGroupFamily::History,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn heterogeneous_phase_schedule_is_rejected() {
        let prefill = request(1, &[10, 11, 12], &[], None);
        let decode = request(2, &[20, 21], &[99], None);
        let scheduled = [&prefill, &decode];
        let error =
            schedule_ready_prefix(&scheduled, SchedulerConfig::default(), None).unwrap_err();

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
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default(), None).unwrap();
        let resource_refs = [RequestResourceRef {
            request_id: core_request_id(41),
            state: managed.cache.as_ref().unwrap(),
        }];
        let acquired = resources.acquire(&resource_refs, &rows, 0).unwrap();
        let plan = build_plan(9, &scheduled, &rows, Some(&acquired.view)).unwrap();

        assert_eq!(plan.token_ids, vec![5, 6]);
        assert_eq!(plan.context_lengths, vec![4]);
        assert_eq!(plan.cache_groups.len(), 2);
        let full = &plan.cache_groups[0];
        assert_eq!(full.inner_native().block_indptr, [0, 2]);
        assert_eq!(full.inner_native().block_slab_ids.len(), 2);
        assert_eq!(full.inner_native().block_table_start_indices, [0]);
        assert_eq!(full.inner_native().append_block_offsets, [0, 1]);
        let fine = &plan.cache_groups[1];
        assert_eq!(fine.inner_native().block_indptr, [0, 3]);
        assert_eq!(fine.inner_native().block_slab_ids.len(), 3);
        assert_eq!(fine.inner_native().append_block_offsets, [0, 1]);

        assert_eq!(acquired.lease.len(), 5);
        resources.release(acquired.lease);

        let pages = managed.cache.take().unwrap();
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
            .map(|page| resources.pool().physical_address(page.handle()).unwrap())
            .collect::<Vec<_>>();
        let mut managed = request(
            51,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[],
            Some(pages),
        );
        managed.resident_tokens = 8;
        let scheduled = [&managed];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default(), None).unwrap();
        assert_eq!(rows[0].token_start(), 8);
        assert_eq!(rows[0].token_count().get(), 4);

        let resource_refs = [RequestResourceRef {
            request_id: core_request_id(51),
            state: managed.cache.as_ref().unwrap(),
        }];
        let acquired = resources.acquire(&resource_refs, &rows, 0).unwrap();
        let plan = build_plan(10, &scheduled, &rows, Some(&acquired.view)).unwrap();
        let window = &plan.cache_groups[0];
        assert_eq!(window.inner_native().block_table_start_indices, [1]);
        assert_eq!(window.inner_native().block_indptr, [0, 2]);
        assert_eq!(
            window.inner_native().block_slab_ids,
            [physical[1].0, physical[0].0]
        );
        assert_eq!(
            window.inner_native().block_slotnums,
            [physical[1].1, physical[0].1]
        );
        assert_eq!(
            window.inner_native().append_slab_ids,
            [physical[0].0, physical[0].0, physical[0].0, physical[0].0]
        );
        assert_eq!(
            window.inner_native().append_slotnums,
            [physical[0].1, physical[0].1, physical[0].1, physical[0].1]
        );
        assert_eq!(window.inner_native().append_block_offsets, [0, 1, 2, 3]);

        assert_eq!(acquired.lease.len(), 2);
        resources.release(acquired.lease);

        resources.retire(managed.cache.take().unwrap()).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);
    }

    #[test]
    fn window_prefill_can_cross_logical_block_boundaries() {
        let mut first = request(1, &[1, 2, 3, 4, 5, 6], &[], None);
        first.resident_tokens = 3;
        let second = request(2, &[7, 8, 9, 10, 11, 12], &[], None);
        let scheduled = [&first, &second];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default(), None).unwrap();
        assert_eq!(rows[0].token_count().get(), 3);
        assert_eq!(rows[1].token_count().get(), 6);
    }

    #[test]
    fn batch_plan_rejects_noncanonical_cache_group_order() {
        let unmanaged = request(1, &[1, 2], &[], None);
        let scheduled = [&unmanaged];
        let rows = schedule_ready_prefix(&scheduled, SchedulerConfig::default(), None).unwrap();
        let mut plan = build_plan(1, &scheduled, &rows, None).unwrap();
        plan.cache_groups = vec![
            CacheGroupPlan::from_parts(
                1,
                vec![0, 1],
                vec![7],
                vec![0],
                vec![0],
                vec![7, 7],
                vec![0, 0],
                vec![0, 1],
                1,
            )
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
        let rows = schedule_ready_prefix(&scheduled, config, None).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].token_count().get(), 2);
        assert_eq!(rows[1].token_count().get(), 1);
        assert!(!rows[0].sample);
        assert!(!rows[1].sample);
    }
}
