//! Python protocol adapters; model-format state lives in resin-recipe.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use pyo3::{
    exceptions::{PyAttributeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pythonize::pythonize;
use resin_recipe::{
    Decoder, Error, Event, FinishReason, InklingParser, InklingTokens, ResponseParser, Tokenizer,
    TokenizerDecoder,
};

#[derive(Clone)]
enum TokenizerSource {
    Native(Arc<Tokenizer>),
    Python(Arc<Py<PyAny>>),
}

/// A tokenizer prepared once for a model's response parsers.
///
/// Fast Hugging Face tokenizers are snapshotted into Rust. Custom Python
/// tokenizers remain supported through their decode method.
#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct RecipeTokenizer {
    source: TokenizerSource,
}

#[pymethods]
impl RecipeTokenizer {
    #[new]
    fn new(tokenizer: &Bound<'_, PyAny>) -> PyResult<Self> {
        let backend =
            optional_attr(tokenizer, "backend_tokenizer")?.unwrap_or_else(|| tokenizer.clone());
        let serialize = optional_attr(&backend, "to_str")?.filter(|method| method.is_callable());
        let source = if let Some(serialize) = serialize {
            let json: String = serialize.call0()?.extract()?;
            let native = Tokenizer::from_bytes(json.as_bytes())
                .map_err(|error| PyValueError::new_err(format!("invalid tokenizer: {error}")))?;
            TokenizerSource::Native(Arc::new(native))
        } else {
            let decode = tokenizer.getattr("decode")?;
            if !decode.is_callable() {
                return Err(PyValueError::new_err("tokenizer.decode must be callable"));
            }
            TokenizerSource::Python(Arc::new(decode.unbind()))
        };
        Ok(Self { source })
    }

    #[getter]
    fn is_native(&self) -> bool {
        matches!(self.source, TokenizerSource::Native(_))
    }
}

fn optional_attr<'py>(
    object: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match object.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_instance_of::<PyAttributeError>(object.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

impl RecipeTokenizer {
    fn decoder(&self) -> Box<dyn Decoder> {
        match &self.source {
            TokenizerSource::Native(tokenizer) => {
                Box::new(TokenizerDecoder::new(tokenizer.clone()))
            }
            TokenizerSource::Python(decode) => Box::new(PythonDecoder {
                decode: decode.clone(),
                ids: Vec::new(),
                emitted: String::new(),
            }),
        }
    }
}

/// Compatibility adapter for custom tokenizers without a serialized backend.
/// These retain cumulative decoding; production fast tokenizers use Rust's
/// streaming decoder instead.
struct PythonDecoder {
    decode: Arc<Py<PyAny>>,
    ids: Vec<u32>,
    emitted: String,
}

impl PythonDecoder {
    fn decoded(&self) -> Result<String, Error> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("skip_special_tokens", false)?;
            self.decode
                .bind(py)
                .call((&self.ids,), Some(&kwargs))?
                .extract()
        })
        .map_err(|error: PyErr| Error::new(error.to_string()))
    }

    fn delta(&mut self, text: &str) -> Result<String, Error> {
        let delta = text
            .strip_prefix(&self.emitted)
            .ok_or_else(|| Error::new("decoded output is not prefix-stable"))?
            .to_owned();
        self.emitted = text.to_owned();
        Ok(delta)
    }
}

impl Decoder for PythonDecoder {
    fn push(&mut self, ids: &[u32]) -> Result<String, Error> {
        self.ids.extend_from_slice(ids);
        let text = self.decoded()?;
        self.delta(text.trim_end_matches('\u{fffd}'))
    }

    fn finish(&mut self) -> Result<String, Error> {
        if self.ids.is_empty() {
            return Ok(String::new());
        }
        let text = self.decoded()?;
        let delta = self.delta(&text)?;
        self.reset();
        Ok(delta)
    }

    fn reset(&mut self) {
        self.ids.clear();
        self.emitted.clear();
    }
}

pub(crate) type SharedParser = Arc<Mutex<Box<dyn ResponseParser>>>;

/// Parse Inkling response blocks into the events accepted by HttpServer.
#[pyclass(name = "InklingResponseParser", frozen, module = "resin._resin")]
pub(crate) struct NativeInklingParser {
    pub(crate) inner: SharedParser,
}

#[pymethods]
impl NativeInklingParser {
    #[new]
    #[pyo3(signature = (tokenizer, token_ids, *, prompt_ids = Vec::new()))]
    fn new(
        tokenizer: &RecipeTokenizer,
        token_ids: HashMap<String, u32>,
        prompt_ids: Vec<u32>,
    ) -> PyResult<Self> {
        let get = |name: &str| {
            token_ids
                .get(name)
                .copied()
                .ok_or_else(|| PyValueError::new_err(format!("missing Inkling token {name}")))
        };
        let tokens = InklingTokens {
            message_model: get("<|message_model|>")?,
            content_text: get("<|content_text|>")?,
            content_thinking: get("<|content_thinking|>")?,
            content_invoke_tool_json: get("<|content_invoke_tool_json|>")?,
            content_invoke_tool_text: get("<|content_invoke_tool_text|>")?,
            content_tool_error: get("<|content_tool_error|>")?,
            content_model_end_sampling: get("<|content_model_end_sampling|>")?,
            end_message: get("<|end_message|>")?,
        };
        let parser =
            InklingParser::new(tokens, tokenizer.decoder(), &prompt_ids).map_err(recipe_error)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Box::new(parser))),
        })
    }

    fn feed(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Py<PyAny>> {
        let events = py.detach(|| feed(&self.inner, &token_ids))?;
        Ok(pythonize(py, &events)?.unbind())
    }

    fn finish(&self, py: Python<'_>, reason: &str) -> PyResult<Py<PyAny>> {
        let reason = match reason {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            _ => {
                return Err(PyValueError::new_err(
                    "finish reason must be 'stop' or 'length'",
                ));
            }
        };
        let events = py.detach(|| finish(&self.inner, reason))?;
        Ok(pythonize(py, &events)?.unbind())
    }
}

pub(crate) fn feed(parser: &SharedParser, ids: &[u32]) -> PyResult<Vec<Event>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .feed(ids)
        .map_err(recipe_error)
}

pub(crate) fn finish(parser: &SharedParser, reason: FinishReason) -> PyResult<Vec<Event>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .finish(reason)
        .map_err(recipe_error)
}

fn recipe_error(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}
