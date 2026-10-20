use std::num::NonZeroU32;
use std::sync::{Arc, OnceLock};

use numpy::{PyArray1, PyArrayLike1};
use pyo3::exceptions::{PyAttributeError, PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

mod arrays;
mod cache;
mod http;
mod output;
mod planning;
mod runner;

use cache::{CacheGroupFamily, CacheGroupLayout, CacheGroupPlan, CacheLayout, CacheRetention};
use output::{BorrowedLogprobs, ForwardOutput};
use resin_core::engine::{self as core, CompletionReason};
use runner::{
    ResourceBinding, RunnerConfig, bind_runner_resources, resolve_runner_config,
    resolve_runner_contract, runner_error_is_post_accept,
};

fn default_sampling_seed(engine_id: u64, request_id: u64) -> i64 {
    // SplitMix64 gives stable, well-distributed row seeds without adding a
    // control-plane RNG or making batch composition part of randomness.
    let mut value = engine_id ^ request_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) & i64::MAX as u64) as i64
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

    const fn core(self) -> resin_core::Phase {
        match self {
            Self::Prefill => resin_core::Phase::Prefill,
            Self::Decode => resin_core::Phase::Decode,
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
    logprobs: bool,
    top_logprobs: usize,
    extra: Option<Arc<Py<PyAny>>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            seed: None,
            logprobs: false,
            top_logprobs: 0,
            extra: None,
        }
    }
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (*, temperature = 1.0, top_p = 1.0, top_k = None, seed = None, logprobs = false, top_logprobs = 0, extra = None))]
    fn new(
        temperature: f64,
        top_p: f64,
        top_k: Option<u32>,
        seed: Option<i64>,
        logprobs: bool,
        top_logprobs: usize,
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
        if top_logprobs > 1000 {
            return Err(PyValueError::new_err(
                "top_logprobs must be between 0 and 1000",
            ));
        }
        Ok(Self {
            temperature,
            top_p,
            top_k,
            seed,
            logprobs: logprobs || top_logprobs > 0,
            top_logprobs,
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
    fn logprobs(&self) -> bool {
        self.logprobs
    }

    #[getter]
    fn top_logprobs(&self) -> usize {
        self.top_logprobs
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.extra
            .as_ref()
            .map(|extra| extra.as_ref().clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingParams(temperature={:?}, top_p={:?}, top_k={:?}, seed={:?}, logprobs={}, top_logprobs={}, extra={})",
            self.temperature,
            self.top_p,
            self.top_k,
            self.seed,
            self.logprobs,
            self.top_logprobs,
            if self.extra.is_some() {
                "<object>"
            } else {
                "None"
            }
        )
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

    /// Monotonic identifier for tracing and correlating runner steps.
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

    /// Check natively so token-only graph replay does not box per-row policies.
    #[getter]
    fn needs_logprobs(&self) -> bool {
        self.sampling_mask
            .iter()
            .zip(&self.sampling_params)
            .any(|(sample, params)| *sample && params.logprobs)
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
    logprobs: Option<core::TokenLogprobs>,
}

#[pymethods]
impl TokenEvent {
    #[getter]
    fn logprobs(&self) -> Option<TokenLogprobs> {
        self.logprobs.clone().map(|inner| TokenLogprobs { inner })
    }
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

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
struct TokenLogprobs {
    inner: core::TokenLogprobs,
}

#[pymethods]
impl TokenLogprobs {
    #[new]
    #[pyo3(signature = (logprob, top_logprobs = Vec::new()))]
    fn new(logprob: f32, top_logprobs: Vec<(u32, f32)>) -> Self {
        Self {
            inner: core::TokenLogprobs {
                logprob,
                top_logprobs,
            },
        }
    }
    #[getter]
    fn logprob(&self) -> f32 {
        self.inner.logprob
    }
    #[getter]
    fn top_logprobs(&self) -> Vec<(u32, f32)> {
        self.inner.top_logprobs.clone()
    }
}

#[pyclass(module = "resin._resin")]
struct RequestLease {
    inner: core::RequestHandle,
}
#[pymethods]
impl RequestLease {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }
    #[getter]
    fn request_id(&self) -> u64 {
        self.inner.id()
    }
    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.cancelled()
    }
    /// Number of prompt tokens restored from the prefix cache at admission.
    #[getter]
    fn cached_tokens(&self) -> u32 {
        self.inner.cached_tokens()
    }
    /// Consume the completed token buffer exactly once.
    fn take_result(&self) -> PyResult<Vec<u32>> {
        self.inner.take_result().map_err(core_error)
    }
    fn __int__(&self) -> u64 {
        self.inner.id()
    }
    fn __index__(&self) -> u64 {
        self.inner.id()
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyRef<'_, RequestLease>>() {
            return Ok(self.inner.id() == other.inner.id()
                && self.inner.engine_id() == other.inner.engine_id());
        }
        Ok(other.extract::<u64>().is_ok_and(|id| id == self.inner.id()))
    }
    fn __repr__(&self) -> String {
        format!(
            "RequestLease(id={}, done={}, cancelled={})",
            self.id(),
            self.done(),
            self.cancelled()
        )
    }
}

#[derive(Clone, Debug)]
struct RequestPayload {
    sampling_params: SamplingParams,
    media: Option<Arc<Py<PyAny>>>,
}

fn core_error(error: core::Error) -> PyErr {
    match error.kind {
        core::ErrorKind::InvalidInput => PyValueError::new_err(error.to_string()),
        core::ErrorKind::Memory => PyMemoryError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Only language conversion and callback invocation belong in this adapter.
struct PythonRunner<'a, 'py> {
    py: Python<'py>,
    forward: &'a Py<PyAny>,
    release: Option<&'a Py<PyAny>>,
    error: Option<PyErr>,
    logprobs: Option<BorrowedLogprobs<'py>>,
}
impl PythonRunner<'_, '_> {
    fn failed(&mut self, error: PyErr, accepted: bool) -> core::Error {
        let message = error.to_string();
        self.error = Some(error);
        if accepted {
            core::Error::execution(message)
        } else {
            core::Error::rejected(message)
        }
    }
    fn result<T>(&mut self, result: core::Result<T>) -> PyResult<T> {
        result.map_err(|error| self.error.take().unwrap_or_else(|| core_error(error)))
    }
}
impl<'py> core::Runner<RequestPayload> for PythonRunner<'_, 'py> {
    fn forward(
        &mut self,
        plan: core::ExecutionPlan<RequestPayload>,
    ) -> core::Result<Vec<Vec<u32>>> {
        let plan = Py::new(self.py, planning::from_core(plan))
            .map_err(|error| self.failed(error, false))?;
        let output = self
            .forward
            .bind(self.py)
            .call1((&plan,))
            .map_err(|error| {
                let accepted = runner_error_is_post_accept(self.py, &error);
                self.failed(error, accepted)
            })?;
        let plan = plan.borrow(self.py);
        let (tokens, logprobs) =
            extract_runner_output(&output, &plan).map_err(|error| self.failed(error, true))?;
        self.logprobs = logprobs;
        Ok(tokens)
    }
    fn logprobs(&self) -> Option<core::LogprobView<'_>> {
        self.logprobs.as_ref().map(BorrowedLogprobs::view)
    }
    fn release(&mut self, ids: &[u64]) -> core::Result<()> {
        if let Some(release) = self.release {
            release
                .bind(self.py)
                .call1((ids.to_vec(),))
                .map_err(|error| self.failed(error, true))?;
        }
        Ok(())
    }
}

#[pyclass(module = "resin._resin")]
struct Engine {
    core: core::EngineCore<RequestPayload>,
    runner_forward: Py<PyAny>,
    runner_release: Option<Py<PyAny>>,
    runner_config: RunnerConfig,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (runner, *, max_running_requests = 32, context_length = 32768, scheduler_config = None, eos_token_id = None, prefix_cache = false))]
    fn new(
        py: Python<'_>,
        runner: Py<PyAny>,
        max_running_requests: usize,
        context_length: u32,
        scheduler_config: Option<Py<SchedulerConfig>>,
        eos_token_id: Option<u32>,
        prefix_cache: bool,
    ) -> PyResult<Self> {
        let (runner_forward, runner_release) = resolve_runner_contract(py, runner.bind(py))?;
        let resolved = resolve_runner_config(py, runner.bind(py))?;
        let config = &resolved.config;
        let scheduler = scheduler_config
            .as_ref()
            .map(|config| *config.borrow(py))
            .unwrap_or_default();
        let core = core::EngineCore::new(
            core::RunnerConfig {
                max_batch_tokens: config.max_batch_tokens,
                prefill_chunk_size: config.prefill_chunk_size,
                continuation_prefill: config.continuation_prefill,
                cache_write_lookahead_tokens: config.cache_write_lookahead_tokens,
                cache_layout: config
                    .cache_layout
                    .as_ref()
                    .map(|layout| layout.get().inner_native().clone()),
            },
            max_running_requests,
            context_length,
            core::SchedulerConfig {
                max_batch_tokens: scheduler.max_batch_tokens,
                prefill_chunk_size: scheduler.prefill_chunk_size,
                schedule_conservativeness: scheduler.schedule_conservativeness,
            },
            eos_token_id,
            prefix_cache,
        )
        .map_err(core_error)?;
        let runner_config = bind_runner_resources(py, resolved, core.cache_pool_id())?;
        Ok(Self {
            core,
            runner_forward,
            runner_release,
            runner_config,
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
        self.core.raise_if_open().map_err(core_error)?;
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
        let mut sampling_params = sampling_params
            .map(|params| params.borrow(py).clone())
            .unwrap_or_default();
        sampling_params.seed.get_or_insert_with(|| {
            default_sampling_seed(self.core.id(), self.core.next_request_id())
        });
        let cacheable = media.is_none() || cache_prompt.is_some();
        let input = core::Submission {
            token_ids,
            max_new_tokens,
            priority,
            batch_key,
            stop_token_ids,
            cache_key: cache_prompt,
            cacheable,
            atomic_prefill_spans,
            payload: RequestPayload {
                sampling_params,
                media: media.map(Arc::new),
            },
        };
        let mut handle_error = None;
        self.core
            .submit_with(input, |inner| {
                Py::new(py, RequestLease { inner }).map_err(|error| {
                    handle_error = Some(error);
                    core::Error::memory("could not allocate request handle")
                })
            })
            .map_err(|error| handle_error.unwrap_or_else(|| core_error(error)))
    }

    #[pyo3(signature = (*, phase = None))]
    fn step(&mut self, py: Python<'_>, phase: Option<BatchPhase>) -> PyResult<Vec<Py<TokenEvent>>> {
        let mut runner = PythonRunner {
            py,
            forward: &self.runner_forward,
            release: self.runner_release.as_ref(),
            error: None,
            logprobs: None,
        };
        let mut event_error = None;
        let result = self
            .core
            .step_phase_with(&mut runner, phase.map(BatchPhase::core), |event| {
                Py::new(
                    py,
                    TokenEvent {
                        request_id: event.request_id,
                        token_id: event.token_id,
                        index: event.index,
                        finished: event.finished,
                        finish_reason: event.finish_reason,
                        logprobs: event.logprobs,
                    },
                )
                .map_err(|error| {
                    event_error = Some(error);
                    core::Error::memory("could not allocate token event")
                })
            });
        result.map_err(|error| {
            runner
                .error
                .or(event_error)
                .unwrap_or_else(|| core_error(error))
        })
    }

    fn next_phase(&self) -> PyResult<Option<BatchPhase>> {
        self.core
            .next_phase()
            .map(|phase| {
                phase.map(|phase| match phase {
                    resin_core::Phase::Prefill => BatchPhase::Prefill,
                    resin_core::Phase::Decode => BatchPhase::Decode,
                })
            })
            .map_err(core_error)
    }

    /// Contain a failure reported by a peer without releasing local resources.
    fn poison(&mut self, reason: &str) {
        self.core.poison_message(reason);
    }

    fn cancel(&mut self, py: Python<'_>, request: PyRef<'_, RequestLease>) -> PyResult<bool> {
        let mut runner = PythonRunner {
            py,
            forward: &self.runner_forward,
            release: self.runner_release.as_ref(),
            error: None,
            logprobs: None,
        };
        let result = self.core.cancel(&mut runner, &request.inner);
        runner.result(result)
    }
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut runner = PythonRunner {
            py,
            forward: &self.runner_forward,
            release: self.runner_release.as_ref(),
            error: None,
            logprobs: None,
        };
        let result = self.core.close(&mut runner);
        runner.result(result)
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.core.raise_if_open().map_err(core_error)?;
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
        self.core.has_work()
    }

    #[getter]
    fn active_request_count(&self) -> usize {
        self.core.ready_request_count()
    }

    #[getter]
    fn ready_request_count(&self) -> usize {
        self.core.ready_request_count()
    }

    #[getter]
    fn max_running_requests(&self) -> usize {
        self.core.max_running_requests()
    }

    #[getter]
    fn poisoned(&self) -> bool {
        self.core.poisoned()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.core.closed()
    }

    #[getter]
    fn poison_error(&self) -> Option<&str> {
        self.core.poison_error()
    }

    #[getter]
    fn scheduler_config(&self, py: Python<'_>) -> PyResult<Py<SchedulerConfig>> {
        {
            let config = self.core.scheduler_config();
            Py::new(
                py,
                SchedulerConfig {
                    max_batch_tokens: config.max_batch_tokens,
                    prefill_chunk_size: config.prefill_chunk_size,
                    schedule_conservativeness: config.schedule_conservativeness,
                },
            )
        }
    }

    #[getter]
    fn new_token_ratio(&self) -> f64 {
        self.core.new_token_ratio()
    }

    #[getter]
    fn context_length(&self) -> u32 {
        self.core.context_length()
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
        self.core.cache_pool_id()
    }

    #[getter]
    fn cache_parent_capacity(&self) -> Option<u32> {
        self.core.cache_parent_capacity()
    }

    #[getter]
    fn cache_free_parent_count(&self) -> Option<usize> {
        self.core.cache_free_parent_count()
    }

    #[getter]
    fn cache_occupied_block_count(&self) -> Option<usize> {
        self.core.cache_occupied_block_count()
    }

    /// `(entries, hits, misses, evictions)` when prefix caching is enabled.
    #[getter]
    fn prefix_cache_stats(&self) -> Option<(usize, u64, u64, u64)> {
        self.core.prefix_cache_stats()
    }

    /// Drop all reusable cache prefixes.
    fn clear_prefix_cache(&mut self) -> PyResult<()> {
        self.core.clear_prefix_cache().map_err(core_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(ready_requests={}, max_running_requests={}, context_length={}, closed={}, poisoned={}, cache_blocks={})",
            self.ready_request_count(),
            self.max_running_requests(),
            self.context_length(),
            self.core.closed(),
            self.poisoned(),
            self.core.cache_occupied_block_count().unwrap_or(0)
        )
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.core.closed() || self.core.poisoned() {
            return;
        }

        match Python::try_attach(|py| self.close(py)) {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                self.core
                    .poison_message(format!("Engine destructor cleanup failed: {error}"));
            }
            None => self.core.poison_message(
                "Engine destructor could not attach to Python; resources remain quarantined",
            ),
        }
    }
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

fn extract_runner_output<'py>(
    output: &Bound<'py, PyAny>,
    plan: &BatchPlan,
) -> PyResult<(Vec<Vec<u32>>, Option<BorrowedLogprobs<'py>>)> {
    let (tokens, logprobs) = if let Ok(output) = output.cast::<ForwardOutput>() {
        output
            .borrow()
            .borrow(output.py(), plan.sequence_ids.len())?
    } else {
        (
            extract_runner_token_rows(output, plan.sequence_ids.len())?,
            None,
        )
    };
    let view = logprobs.as_ref().map(BorrowedLogprobs::view);
    for (row, params) in plan.sampling_params.iter().enumerate() {
        if !params.logprobs || !plan.sampling_mask[row] {
            continue;
        }
        let range = view.and_then(|scores| {
            Some(*scores.indptr.get(row)? as usize..*scores.indptr.get(row + 1)? as usize)
        });
        if !range.as_ref().is_some_and(|range| {
            !range.is_empty()
                && range.len() == tokens[row].len()
                && view.unwrap().values.get(range.clone()).is_some()
        }) {
            return Err(PyRuntimeError::new_err(
                "runner did not return requested token logprobs",
            ));
        }
        if params.top_logprobs > 0 {
            let top = view.and_then(|scores| scores.top);
            for index in range.unwrap() {
                let count = top.and_then(|top| {
                    let start = *top.indptr.get(index)?;
                    let end = *top.indptr.get(index + 1)?;
                    end.checked_sub(start).map(|count| count as usize)
                });
                if !count.is_some_and(|count| count > 0 && count <= params.top_logprobs) {
                    return Err(PyRuntimeError::new_err(
                        "runner did not return requested top logprobs",
                    ));
                }
            }
        }
    }
    Ok((tokens, logprobs))
}

#[pymodule(name = "_resin")]
mod _resin {
    #[pymodule_export]
    use super::{
        BatchPhase, BatchPlan, CacheGroupFamily, CacheGroupLayout, CacheGroupPlan, CacheLayout,
        CacheRetention, Engine, ForwardOutput, RequestLease, ResourceBinding, RunnerConfig,
        SamplingParams, SchedulerConfig, TokenEvent, TokenLogprobs,
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
    #[test]
    fn batch_plan_rejects_noncanonical_cache_group_order() {
        let group = CacheGroupPlan::from_parts(
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
        .unwrap();
        let plan = BatchPlan {
            sequence_ids: vec![1],
            context_lengths: vec![0],
            token_ids: vec![1, 2],
            token_indptr: vec![0, 2],
            phase: BatchPhase::Prefill,
            batch_id: 1,
            batch_key: None,
            cache_groups: vec![group],
            sampling_mask: vec![true],
            sampling_params: vec![SamplingParams {
                seed: Some(1),
                ..SamplingParams::default()
            }],
            media: vec![None],
            positions: OnceLock::new(),
        };
        assert!(BatchPlan::checked(plan).is_err());
    }
}
