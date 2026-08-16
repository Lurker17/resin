use std::{
    collections::HashMap,
    net::IpAddr,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pythonize::{depythonize, pythonize};
use resin_http::{FinishReason, GenerationEvent, GenerationJob, RouteConfig};
use serde::Deserialize;
use tokio::sync::mpsc;

use crate::{CompletionReason, Engine, RequestLease, SamplingParams};

#[derive(Debug, Deserialize)]
struct RenderedPrompt {
    token_ids: Vec<u32>,
    #[serde(default)]
    stop_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ParsedEvent {
    ReasoningDelta {
        delta: String,
    },
    TextDelta {
        delta: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: usize,
        delta: String,
    },
}

impl ParsedEvent {
    fn into_http(self) -> GenerationEvent {
        match self {
            Self::ReasoningDelta { delta } => GenerationEvent::ReasoningDelta(delta),
            Self::TextDelta { delta } => GenerationEvent::TextDelta(delta),
            Self::ToolCallStart { index, id, name } => {
                GenerationEvent::ToolCallStart { index, id, name }
            }
            Self::ToolCallArgumentsDelta { index, delta } => {
                GenerationEvent::ToolCallArgumentsDelta { index, delta }
            }
        }
    }
}

struct ActiveRequest {
    lease: Py<RequestLease>,
    parser: Py<PyAny>,
    events: mpsc::UnboundedSender<GenerationEvent>,
    started: Instant,
    first_token_at: Option<Instant>,
    input_tokens: usize,
    output_tokens: usize,
}

#[pyclass(name = "_HttpServer", module = "resin._resin")]
pub(crate) struct NativeHttpServer {
    engine: Py<Engine>,
    codec: Py<PyAny>,
    model: String,
    routes: RouteConfig,
    running: AtomicBool,
}

#[pymethods]
impl NativeHttpServer {
    #[new]
    #[pyo3(signature = (
        engine,
        codec,
        *,
        model,
        enable_openai = true,
        enable_anthropic = true,
        enable_demo = true
    ))]
    fn new(
        engine: Py<Engine>,
        codec: Py<PyAny>,
        model: String,
        enable_openai: bool,
        enable_anthropic: bool,
        enable_demo: bool,
    ) -> PyResult<Self> {
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("model must be nonempty"));
        }
        if enable_demo && !enable_openai {
            return Err(PyValueError::new_err(
                "enable_demo=True requires enable_openai=True",
            ));
        }
        Ok(Self {
            engine,
            codec,
            model,
            routes: RouteConfig {
                enable_openai,
                enable_anthropic,
                enable_demo,
            },
            running: AtomicBool::new(false),
        })
    }

    #[pyo3(signature = (*, host = "127.0.0.1", port = 8000))]
    fn serve(&self, py: Python<'_>, host: &str, port: u16) -> PyResult<()> {
        let host = host.parse::<IpAddr>().map_err(|_| {
            PyValueError::new_err("host must be an IPv4 or IPv6 address such as '0.0.0.0'")
        })?;
        if self.running.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err("HttpServer is already running"));
        }
        let engine = self.engine.clone_ref(py);
        let codec = self.codec.clone_ref(py);
        let model = self.model.clone();
        let routes = self.routes;
        let result = py.detach(move || run_server(engine, codec, model, routes, host, port));
        self.running.store(false, Ordering::Release);
        result.map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn model(&self) -> &str {
        &self.model
    }

    #[getter]
    fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn __repr__(&self) -> String {
        format!(
            "_HttpServer(model={:?}, enable_openai={}, enable_anthropic={}, enable_demo={}, running={})",
            self.model,
            self.routes.enable_openai,
            self.routes.enable_anthropic,
            self.routes.enable_demo,
            self.running.load(Ordering::Acquire),
        )
    }
}

fn run_server(
    engine: Py<Engine>,
    codec: Py<PyAny>,
    model: String,
    routes: RouteConfig,
    host: IpAddr,
    port: u16,
) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("failed to create the HTTP runtime: {error}"))?;
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind((host, port)))
        .map_err(|error| format!("failed to bind {host}:{port}: {error}"))?;
    let (jobs, receiver) = mpsc::unbounded_channel();
    let actor = thread::Builder::new()
        .name("resin-http-engine".to_owned())
        .spawn(move || engine_actor(engine, codec, receiver))
        .map_err(|error| format!("failed to start the inference thread: {error}"))?;
    let server_result = runtime.block_on(resin_http::serve(listener, model, jobs, routes));
    let actor_result = actor
        .join()
        .map_err(|_| "the inference thread panicked".to_owned())?;
    server_result
        .map_err(|error| format!("HTTP server failed: {error}"))
        .and(actor_result)
}

fn engine_actor(
    engine: Py<Engine>,
    codec: Py<PyAny>,
    mut jobs: mpsc::UnboundedReceiver<GenerationJob>,
) -> Result<(), String> {
    let mut active = HashMap::<u64, ActiveRequest>::new();
    let mut disconnected = false;
    loop {
        if active.is_empty() && !disconnected && !engine_has_work(&engine)? {
            match jobs.blocking_recv() {
                Some(job) => admit(&engine, &codec, job, &mut active),
                None => disconnected = true,
            }
        }
        while !disconnected {
            match jobs.try_recv() {
                Ok(job) => admit(&engine, &codec, job, &mut active),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            Python::attach(|py| {
                let mut engine = engine.bind(py).borrow_mut();
                for request in active.values() {
                    let _ = engine.cancel(py, request.lease.bind(py).borrow());
                }
                engine.close(py)
            })
            .map_err(|error| format_python_error("failed to stop the inference engine", error))?;
            return Ok(());
        }
        cancel_disconnected(&engine, &mut active)?;
        let token_events = Python::attach(|py| {
            let mut engine = engine.bind(py).borrow_mut();
            engine.step(py).map(|events| {
                events
                    .into_iter()
                    .map(|event| {
                        let event = event.bind(py).borrow();
                        (event.request_id, event.token_id, event.finish_reason)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .map_err(|error| format_python_error("engine step failed", error))?;

        if token_events.is_empty() {
            thread::sleep(Duration::from_micros(50));
            continue;
        }
        let mut grouped = HashMap::<u64, Vec<(u32, Option<CompletionReason>)>>::new();
        for (request_id, token_id, reason) in token_events {
            grouped
                .entry(request_id)
                .or_default()
                .push((token_id, reason));
        }
        for (request_id, tokens) in grouped {
            let Some(request) = active.get_mut(&request_id) else {
                continue;
            };
            request.first_token_at.get_or_insert_with(Instant::now);
            request.output_tokens += tokens.len();
            let token_ids = tokens.iter().map(|(token, _)| *token).collect::<Vec<_>>();
            match parser_events(&request.parser, &token_ids) {
                Ok(events) => {
                    for event in events {
                        let _ = request.events.send(event.into_http());
                    }
                }
                Err(error) => {
                    let _ = request.events.send(GenerationEvent::Error {
                        message: format_python_error("response parser failed", error),
                    });
                    Python::attach(|py| {
                        engine
                            .bind(py)
                            .borrow_mut()
                            .cancel(py, request.lease.bind(py).borrow())
                    })
                    .map_err(|error| format_python_error("request cancellation failed", error))?;
                    active.remove(&request_id);
                    continue;
                }
            }
            if let Some(reason) = tokens.iter().find_map(|(_, reason)| *reason) {
                let reason = match reason {
                    CompletionReason::Stop => FinishReason::Stop,
                    CompletionReason::Length => FinishReason::Length,
                };
                let _ = request.events.send(GenerationEvent::Finished {
                    reason,
                    input_tokens: request.input_tokens,
                    output_tokens: request.output_tokens,
                    elapsed_ms: request.started.elapsed().as_secs_f64() * 1000.0,
                    generation_elapsed_ms: request
                        .first_token_at
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    time_to_first_token_ms: request.first_token_at.map_or(0.0, |first| {
                        first.duration_since(request.started).as_secs_f64() * 1000.0
                    }),
                });
                active.remove(&request_id);
            }
        }
    }
}

fn admit(
    engine: &Py<Engine>,
    codec: &Py<PyAny>,
    job: GenerationJob,
    active: &mut HashMap<u64, ActiveRequest>,
) {
    if job.events.is_closed() {
        return;
    }
    let result = Python::attach(|py| {
        let request = pythonize(py, &job.request)?;
        let rendered = codec.bind(py).call_method1("render", (&request,))?;
        let media = rendered
            .cast::<PyDict>()?
            .get_item("media")?
            .filter(|value| !value.is_none())
            .map(Bound::unbind);
        let rendered: RenderedPrompt = depythonize(&rendered)?;
        if rendered.token_ids.is_empty() {
            return Err(PyValueError::new_err(
                "codec.render returned an empty token_ids sequence",
            ));
        }
        let parser = codec
            .bind(py)
            .call_method1("create_parser", (&request, rendered.token_ids.clone()))?
            .unbind();
        let extra = pythonize(py, &job.request.extra)?.unbind();
        let sampling_params = Py::new(
            py,
            SamplingParams::new(
                job.request.temperature,
                job.request.top_p,
                job.request.top_k,
                job.request.seed,
                Some(extra),
            )?,
        )?;
        let input_tokens = rendered.token_ids.len();
        let lease = engine.bind(py).borrow_mut().submit(
            py,
            Some(rendered.token_ids),
            Some(job.request.max_output_tokens),
            0,
            None,
            Some(sampling_params),
            Some(rendered.stop_token_ids),
            media,
        )?;
        let request_id = lease.bind(py).borrow().request_id;
        Ok::<_, PyErr>((request_id, lease, parser, input_tokens))
    });
    match result {
        Ok((request_id, lease, parser, input_tokens)) => {
            let _ = job.events.send(GenerationEvent::Started { input_tokens });
            active.insert(
                request_id,
                ActiveRequest {
                    lease,
                    parser,
                    events: job.events,
                    started: Instant::now(),
                    first_token_at: None,
                    input_tokens,
                    output_tokens: 0,
                },
            );
        }
        Err(error) => {
            let _ = job.events.send(GenerationEvent::Error {
                message: format_python_error("request admission failed", error),
            });
        }
    }
}

fn parser_events(parser: &Py<PyAny>, token_ids: &[u32]) -> PyResult<Vec<ParsedEvent>> {
    Python::attach(|py| {
        let events = parser.bind(py).call_method1("feed", (token_ids,))?;
        depythonize(&events).map_err(Into::into)
    })
}

fn cancel_disconnected(
    engine: &Py<Engine>,
    active: &mut HashMap<u64, ActiveRequest>,
) -> Result<(), String> {
    let disconnected = active
        .iter()
        .filter_map(|(request_id, request)| request.events.is_closed().then_some(*request_id))
        .collect::<Vec<_>>();
    for request_id in disconnected {
        let request = active
            .remove(&request_id)
            .expect("request id came from the active map");
        Python::attach(|py| {
            engine
                .bind(py)
                .borrow_mut()
                .cancel(py, request.lease.bind(py).borrow())
        })
        .map_err(|error| format_python_error("request cancellation failed", error))?;
    }
    Ok(())
}

fn engine_has_work(engine: &Py<Engine>) -> Result<bool, String> {
    Ok(Python::attach(|py| engine.bind(py).borrow().has_work()))
}

fn format_python_error(context: &str, error: PyErr) -> String {
    format!("{context}: {error}")
}
