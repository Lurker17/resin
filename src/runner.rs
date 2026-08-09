use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyString, PyTuple};

use crate::StateMappingConfig;
use crate::cache::CacheLayout;
use crate::state_pool::StatePool;

pub(super) enum RunnerExecution {
    Sync {
        forward: Py<PyAny>,
    },
    Async {
        dispatch: Py<PyAny>,
        poll: Py<PyAny>,
    },
}

impl RunnerExecution {
    pub(super) fn is_async(&self) -> bool {
        matches!(self, Self::Async { .. })
    }
}

enum RunnerResourceBinder {
    None,
    Cache(Py<PyAny>),
    RecurrentState(Py<PyAny>),
    CacheAndRecurrentState(Py<PyAny>),
}

/// Frozen runner-owned resource declarations plus the one callback selected
/// for their exact shape. A runner which owns both resources is deliberately
/// represented by one callback, never two independently fallible callbacks.
pub(super) struct RunnerResources {
    pub(super) cache_layout: Option<Py<CacheLayout>>,
    recurrent_state_mapping: Option<StateMappingConfig>,
    required_recurrent_arena_id: Option<u64>,
    binder: RunnerResourceBinder,
}

fn optional_runner_callable(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    match runner.getattr(name) {
        Ok(value) => {
            if !value.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "runner.{name} must be callable"
                )));
            }
            Ok(Some(value.unbind()))
        }
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn runner_error_is_post_accept(py: Python<'_>, error: &PyErr) -> bool {
    match error.value(py).getattr("_resin_post_accept") {
        Ok(marker) => marker.is(PyBool::new(py, true)),
        Err(marker_error) if marker_error.is_instance_of::<PyAttributeError>(py) => false,
        // Failure to inspect an explicitly raised exception cannot justify
        // returning native resources to allocators which may still be in use.
        Err(_) => true,
    }
}

pub(super) fn resolve_runner_contract(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
) -> PyResult<(RunnerExecution, Option<Py<PyAny>>)> {
    let dispatch = optional_runner_callable(py, runner, "dispatch")?;
    let poll = optional_runner_callable(py, runner, "poll")?;
    let execution = match (dispatch, poll) {
        (Some(dispatch), Some(poll)) => RunnerExecution::Async { dispatch, poll },
        (None, None) => {
            let forward = optional_runner_callable(py, runner, "forward")?.ok_or_else(|| {
                PyTypeError::new_err(
                    "runner must define callable forward or callable dispatch and poll",
                )
            })?;
            RunnerExecution::Sync { forward }
        }
        _ => {
            return Err(PyTypeError::new_err(
                "runner.dispatch and runner.poll must be supplied together",
            ));
        }
    };
    let release = optional_runner_callable(py, runner, "release")?;
    Ok((execution, release))
}

pub(super) fn runner_supports_continuation_prefill(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    match runner.getattr("supports_continuation_prefill") {
        Ok(value) => Ok(value.is(PyBool::new(py, true))),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Ok(false),
        Err(error) => Err(error),
    }
}

pub(super) fn resolve_runner_prefill_chunk_size(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
) -> PyResult<Option<std::num::NonZeroU32>> {
    let declared = match runner.getattr("prefill_chunk_size") {
        Ok(value) => value,
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => return Ok(None),
        Err(error) => return Err(error),
    };
    if !declared.is_exact_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "runner.prefill_chunk_size must be exactly a positive int",
        ));
    }
    let value = declared.extract::<u32>().map_err(|_| {
        PyValueError::new_err("runner.prefill_chunk_size must fit in unsigned 32-bit metadata")
    })?;
    std::num::NonZeroU32::new(value)
        .ok_or_else(|| PyValueError::new_err("runner.prefill_chunk_size must be positive"))
        .map(Some)
}

fn resolve_runner_cache_layout(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<CacheLayout>>> {
    let declared = match runner.getattr("cache_layout") {
        Ok(value) => value,
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => return Ok(None),
        Err(error) => return Err(error),
    };
    if declared.is_none() {
        return Ok(None);
    }
    let layout = declared
        .cast::<CacheLayout>()
        .map_err(|_| PyTypeError::new_err("runner.cache_layout must be exactly a CacheLayout"))?
        .clone()
        .unbind();
    Ok(Some(layout))
}

fn bind_runner_cache_pool(
    py: Python<'_>,
    binder: &Py<PyAny>,
    layout: &Py<CacheLayout>,
    pool_id: u64,
) -> PyResult<()> {
    let acknowledgement = binder.bind(py).call1((pool_id, layout.clone_ref(py)))?;
    if !acknowledgement.is_exact_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "runner cache-pool acknowledgement must be exactly an int",
        ));
    }
    let acknowledged_pool_id = acknowledgement.extract::<u64>()?;
    if acknowledged_pool_id != pool_id {
        return Err(PyValueError::new_err(format!(
            "runner acknowledged cache pool {acknowledged_pool_id}, expected {pool_id}"
        )));
    }
    Ok(())
}

fn resolve_runner_recurrent_state_mapping(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    state_pool: Option<&Py<StatePool>>,
) -> PyResult<Option<(StateMappingConfig, Option<u64>)>> {
    match runner.getattr("supports_recurrent_state_mapping") {
        Ok(value) if value.is(PyBool::new(py, true)) => {}
        Ok(_) => return Ok(None),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => return Ok(None),
        Err(error) => return Err(error),
    }
    let declared_mapping = runner.getattr("recurrent_state_mapping").map_err(|error| {
        if error.is_instance_of::<PyAttributeError>(py) {
            PyTypeError::new_err(
                "runner.supports_recurrent_state_mapping is True but runner.recurrent_state_mapping is missing",
            )
        } else {
            error
        }
    })?;
    let declared_mapping = declared_mapping
        .cast::<StateMappingConfig>()
        .map_err(|_| {
            PyTypeError::new_err(
                "runner.recurrent_state_mapping must be exactly a StateMappingConfig",
            )
        })?
        .borrow()
        .clone();
    let state_pool = state_pool.ok_or_else(|| {
        PyValueError::new_err(
            "runner recurrent-state mapping support requires an Engine state_pool",
        )
    })?;
    let bound_mapping = {
        let pool = state_pool.bind(py).borrow();
        pool.physical_mapping_descriptor()
            .map(|(protocol, arena_id)| (protocol.to_owned(), arena_id))
    };
    if let Some((bound_protocol, _)) = bound_mapping.as_ref()
        && bound_protocol.as_str() != declared_mapping.protocol
    {
        return Err(PyValueError::new_err(format!(
            "state pool is already bound to recurrent-state protocol {bound_protocol:?}, cannot bind descriptor protocol {:?}",
            declared_mapping.protocol,
        )));
    }
    Ok(Some((
        declared_mapping,
        bound_mapping.map(|(_, arena_id)| arena_id),
    )))
}

fn recurrent_state_pool_descriptor(
    py: Python<'_>,
    state_pool: Option<&Py<StatePool>>,
) -> PyResult<(u64, usize)> {
    let state_pool = state_pool.ok_or_else(|| {
        PyValueError::new_err(
            "runner recurrent-state mapping support requires an Engine state_pool",
        )
    })?;
    let pool = state_pool.bind(py).borrow();
    Ok((pool.id(), pool.slot_capacity()))
}

fn bind_runner_recurrent_state_mapping(
    py: Python<'_>,
    binder: &Py<PyAny>,
    config: &StateMappingConfig,
    state_pool: Option<&Py<StatePool>>,
) -> PyResult<()> {
    let (pool_id, capacity) = recurrent_state_pool_descriptor(py, state_pool)?;
    let protocol = config.protocol.as_str();
    let acknowledgement_object = binder.bind(py).call1((pool_id, capacity, protocol))?;
    let acknowledgement_tuple = acknowledgement_object.cast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(
            "runner recurrent-state mapping acknowledgement must be a four-item tuple",
        )
    })?;
    if acknowledgement_tuple.len() != 4
        || !acknowledgement_tuple
            .get_item(0)?
            .is_exact_instance_of::<PyInt>()
        || !acknowledgement_tuple
            .get_item(1)?
            .is_exact_instance_of::<PyInt>()
        || !acknowledgement_tuple
            .get_item(2)?
            .is_exact_instance_of::<PyString>()
        || !acknowledgement_tuple
            .get_item(3)?
            .is_exact_instance_of::<PyInt>()
    {
        return Err(PyTypeError::new_err(
            "runner recurrent-state mapping acknowledgement must be exactly (int, int, str, int)",
        ));
    }
    let acknowledgement = acknowledgement_tuple.extract::<(u64, usize, String, u64)>()?;
    if acknowledgement.0 != pool_id
        || acknowledgement.1 != capacity
        || acknowledgement.2 != protocol
        || acknowledgement.3 == 0
    {
        return Err(PyValueError::new_err(format!(
            "runner acknowledged recurrent-state mapping {acknowledgement:?}, expected pool_id={pool_id}, capacity={capacity}, protocol={protocol:?}, and a positive arena_id"
        )));
    }
    state_pool
        .expect("descriptor required a state pool")
        .bind(py)
        .borrow_mut()
        .bind_physical_mapping(protocol, acknowledgement.3)?;
    Ok(())
}

pub(super) fn resolve_runner_resources(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    state_pool: Option<&Py<StatePool>>,
) -> PyResult<RunnerResources> {
    // Resolve and validate both declarations before inspecting any callback.
    // This guarantees malformed descriptors cannot trigger runner-side state.
    let cache_layout = resolve_runner_cache_layout(py, runner)?;
    let resolved_recurrent_state_mapping =
        resolve_runner_recurrent_state_mapping(py, runner, state_pool)?;
    let binder = match (&cache_layout, &resolved_recurrent_state_mapping) {
        (None, None) => RunnerResourceBinder::None,
        (Some(_), None) => RunnerResourceBinder::Cache(
            optional_runner_callable(py, runner, "bind_cache_pool")?.ok_or_else(|| {
                PyTypeError::new_err(
                    "runner.cache_layout requires a callable runner.bind_cache_pool",
                )
            })?,
        ),
        (None, Some(_)) => RunnerResourceBinder::RecurrentState(
            optional_runner_callable(py, runner, "bind_recurrent_state_mapping")?.ok_or_else(
                || {
                    PyTypeError::new_err(
                        "runner.supports_recurrent_state_mapping is True but runner.bind_recurrent_state_mapping is missing",
                    )
                },
            )?,
        ),
        (Some(_), Some(_)) => RunnerResourceBinder::CacheAndRecurrentState(
            optional_runner_callable(py, runner, "bind_cache_and_recurrent_state")?.ok_or_else(
                || {
                    PyTypeError::new_err(
                        "runner with cache and recurrent-state declarations requires a callable runner.bind_cache_and_recurrent_state",
                    )
                },
            )?,
        ),
    };
    let (recurrent_state_mapping, required_recurrent_arena_id) =
        match resolved_recurrent_state_mapping {
            Some((mapping, required_arena_id)) => (Some(mapping), required_arena_id),
            None => (None, None),
        };
    Ok(RunnerResources {
        cache_layout,
        recurrent_state_mapping,
        required_recurrent_arena_id,
        binder,
    })
}

pub(super) fn bind_runner_resources(
    py: Python<'_>,
    resources: &RunnerResources,
    cache_pool_id: Option<u64>,
    state_pool: Option<&Py<StatePool>>,
) -> PyResult<(Option<CacheLayout>, Option<StateMappingConfig>)> {
    let cache_layout = resources
        .cache_layout
        .as_ref()
        .map(|layout| layout.borrow(py).clone());
    match &resources.binder {
        RunnerResourceBinder::None => {
            if cache_pool_id.is_some() {
                return Err(PyRuntimeError::new_err(
                    "native cache pool exists without a runner cache layout",
                ));
            }
        }
        RunnerResourceBinder::Cache(binder) => {
            let layout = resources
                .cache_layout
                .as_ref()
                .expect("cache binder requires a layout");
            let pool_id = cache_pool_id.ok_or_else(|| {
                PyRuntimeError::new_err("runner cache layout exists without a native cache pool")
            })?;
            bind_runner_cache_pool(py, binder, layout, pool_id)?;
        }
        RunnerResourceBinder::RecurrentState(binder) => {
            let config = resources
                .recurrent_state_mapping
                .as_ref()
                .expect("recurrent-state binder requires a descriptor");
            bind_runner_recurrent_state_mapping(py, binder, config, state_pool)?;
        }
        RunnerResourceBinder::CacheAndRecurrentState(binder) => {
            let layout = resources
                .cache_layout
                .as_ref()
                .expect("combined binder requires a cache layout");
            let cache_pool_id = cache_pool_id.ok_or_else(|| {
                PyRuntimeError::new_err("runner cache layout exists without a native cache pool")
            })?;
            let config = resources
                .recurrent_state_mapping
                .as_ref()
                .expect("combined binder requires a recurrent-state descriptor");
            let (state_pool_id, capacity) = recurrent_state_pool_descriptor(py, state_pool)?;
            let protocol = config.protocol.as_str();
            let acknowledgement_object = binder.bind(py).call1((
                cache_pool_id,
                layout.clone_ref(py),
                state_pool_id,
                capacity,
                protocol,
                resources.required_recurrent_arena_id,
            ))?;
            let acknowledgement_tuple = acknowledgement_object.cast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err(
                    "runner combined resource acknowledgement must be a five-item tuple",
                )
            })?;
            if acknowledgement_tuple.len() != 5
                || !acknowledgement_tuple
                    .get_item(0)?
                    .is_exact_instance_of::<PyInt>()
                || !acknowledgement_tuple
                    .get_item(1)?
                    .is_exact_instance_of::<PyInt>()
                || !acknowledgement_tuple
                    .get_item(2)?
                    .is_exact_instance_of::<PyInt>()
                || !acknowledgement_tuple
                    .get_item(3)?
                    .is_exact_instance_of::<PyString>()
                || !acknowledgement_tuple
                    .get_item(4)?
                    .is_exact_instance_of::<PyInt>()
            {
                return Err(PyTypeError::new_err(
                    "runner combined resource acknowledgement must be exactly (int, int, int, str, int)",
                ));
            }
            let acknowledgement =
                acknowledgement_tuple.extract::<(u64, u64, usize, String, u64)>()?;
            if acknowledgement.0 != cache_pool_id
                || acknowledgement.1 != state_pool_id
                || acknowledgement.2 != capacity
                || acknowledgement.3 != protocol
                || acknowledgement.4 == 0
                || resources
                    .required_recurrent_arena_id
                    .is_some_and(|required| acknowledgement.4 != required)
            {
                return Err(PyValueError::new_err(format!(
                    "runner acknowledged combined resources {acknowledgement:?}, expected cache_pool_id={cache_pool_id}, state_pool_id={state_pool_id}, capacity={capacity}, protocol={protocol:?}, and arena_id={:?} (or any positive arena id when unbound)",
                    resources.required_recurrent_arena_id,
                )));
            }
            // The runner has atomically accepted both resources and the one
            // acknowledgement is now fully validated. Only now may native
            // state generations acquire a permanent physical interpretation.
            state_pool
                .expect("descriptor required a state pool")
                .bind(py)
                .borrow_mut()
                .bind_physical_mapping(protocol, acknowledgement.4)?;
        }
    }
    Ok((cache_layout, resources.recurrent_state_mapping.clone()))
}
