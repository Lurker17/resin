//! Python protocol adapters; model-format state lives in resin-recipe.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use pyo3::{
    exceptions::{PyAttributeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict},
};
use pythonize::{depythonize, pythonize};
use resin_recipe::{
    Decoder, Encoder, Error, Event, FinishReason, InklingParser, InklingTokens, ModelFormat,
    Prompt, PromptSpan, ResponseParser, Tokenizer, TokenizerDecoder, TokenizerEncoder,
};
use serde_json::{Map, Value};

#[derive(Clone)]
enum TokenizerSource {
    Native(Arc<Tokenizer>, Arc<TokenizerEncoder>),
    Python {
        object: Arc<Py<PyAny>>,
        decode: Arc<Py<PyAny>>,
    },
}

/// A tokenizer prepared once for a model's response parsers.
///
/// Fast Hugging Face tokenizers are snapshotted into Rust. Custom Python
/// tokenizers remain supported through their decode method.
#[pyclass(frozen, skip_from_py_object, module = "resin._resin")]
#[derive(Clone)]
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
            {
                let tokenizer = Arc::new(native);
                let encoder =
                    Arc::new(TokenizerEncoder::new(tokenizer.clone()).map_err(recipe_error)?);
                TokenizerSource::Native(tokenizer, encoder)
            }
        } else {
            let decode = tokenizer.getattr("decode")?;
            if !decode.is_callable() {
                return Err(PyValueError::new_err("tokenizer.decode must be callable"));
            }
            TokenizerSource::Python {
                object: Arc::new(tokenizer.clone().unbind()),
                decode: Arc::new(decode.unbind()),
            }
        };
        Ok(Self { source })
    }

    #[getter]
    fn is_native(&self) -> bool {
        matches!(self.source, TokenizerSource::Native(..))
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
            TokenizerSource::Native(tokenizer, _) => {
                Box::new(TokenizerDecoder::new(tokenizer.clone()))
            }
            TokenizerSource::Python { decode, .. } => Box::new(PythonDecoder {
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

    fn feed_attributed(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Py<PyAny>> {
        let events = py.detach(|| feed_attributed(&self.inner, &token_ids))?;
        Ok(pythonize(py, &events)?.unbind())
    }

    fn finish_attributed(&self, py: Python<'_>, reason: &str) -> PyResult<Py<PyAny>> {
        let reason = parse_finish_reason(reason)?;
        let events = py.detach(|| finish_attributed(&self.inner, reason))?;
        Ok(pythonize(py, &events)?.unbind())
    }

    fn finish(&self, py: Python<'_>, reason: &str) -> PyResult<Py<PyAny>> {
        let reason = parse_finish_reason(reason)?;
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

pub(crate) fn feed_attributed(
    parser: &SharedParser,
    ids: &[u32],
) -> PyResult<Vec<resin_recipe::AttributedEvent>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .feed_attributed(ids)
        .map_err(recipe_error)
}

pub(crate) fn finish_attributed(
    parser: &SharedParser,
    reason: FinishReason,
) -> PyResult<Vec<resin_recipe::AttributedEvent>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .finish_attributed(reason)
        .map_err(recipe_error)
}

fn parse_finish_reason(reason: &str) -> PyResult<FinishReason> {
    match reason {
        "stop" => Ok(FinishReason::Stop),
        "length" => Ok(FinishReason::Length),
        _ => Err(PyValueError::new_err(
            "finish reason must be 'stop' or 'length'",
        )),
    }
}

fn recipe_error(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

impl RecipeTokenizer {
    fn encoder(&self) -> Arc<dyn Encoder> {
        match &self.source {
            TokenizerSource::Native(_, encoder) => encoder.clone(),
            TokenizerSource::Python { object, .. } => Arc::new(PythonEncoder(object.clone())),
        }
    }
}
struct PythonEncoder(Arc<Py<PyAny>>);
impl PythonEncoder {
    fn raw(tokenizer: &Bound<'_, PyAny>, text: &str) -> PyResult<Vec<u32>> {
        let kwargs = PyDict::new(tokenizer.py());
        kwargs.set_item("add_special_tokens", false)?;
        tokenizer
            .call_method("encode", (text,), Some(&kwargs))?
            .extract()
    }
}
impl Encoder for PythonEncoder {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error> {
        Python::attach(|py| {
            let tokenizer = self.0.bind(py);
            if let Some(ordinary) = optional_attr(tokenizer, "encode_ordinary")? {
                ordinary.call1((text,))?.extract()
            } else {
                Self::raw(tokenizer, text)
            }
        })
        .map_err(|error: PyErr| Error::new(error.to_string()))
    }
    fn special_id(&self, spelling: &str) -> Result<u32, Error> {
        Python::attach(|py| {
            let tokenizer = self.0.bind(py);
            let value = if let Some(method) = optional_attr(tokenizer, "convert_tokens_to_ids")? {
                method.call1((spelling,))?
            } else {
                tokenizer.call_method1("token_to_id", (spelling,))?
            };
            if value.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err("control token IDs must be integers"));
            }
            let id: u32 = value.extract()?;
            let inverse = match optional_attr(tokenizer, "convert_ids_to_tokens")? {
                Some(inverse) => Some(inverse),
                None => optional_attr(tokenizer, "id_to_token")?,
            };
            if let Some(inverse) = inverse {
                let round_trip: String = inverse.call1((id,))?.extract()?;
                if round_trip != spelling {
                    return Err(PyValueError::new_err("control token did not round-trip"));
                }
            }
            Ok(id)
        })
        .map_err(|error: PyErr| {
            Error::new(format!("missing model control token {spelling:?}: {error}"))
        })
    }
}

/// Common native handle recognized directly by the HTTP actor.
#[pyclass(name = "RecipeResponseParser", frozen, module = "resin._resin")]
pub(crate) struct NativeResponseParser {
    pub(crate) inner: SharedParser,
}
#[pymethods]
impl NativeResponseParser {
    fn feed(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Py<PyAny>> {
        let events = py.detach(|| feed(&self.inner, &token_ids))?;
        Ok(pythonize(py, &events)?.unbind())
    }

    fn feed_attributed(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Py<PyAny>> {
        let events = py.detach(|| feed_attributed(&self.inner, &token_ids))?;
        Ok(pythonize(py, &events)?.unbind())
    }

    fn finish_attributed(&self, py: Python<'_>, reason: &str) -> PyResult<Py<PyAny>> {
        let reason = parse_finish_reason(reason)?;
        let events = py.detach(|| finish_attributed(&self.inner, reason))?;
        Ok(pythonize(py, &events)?.unbind())
    }
    fn finish(&self, py: Python<'_>, reason: &str) -> PyResult<Py<PyAny>> {
        let reason = parse_finish_reason(reason)?;
        let events = py.detach(|| finish(&self.inner, reason))?;
        Ok(pythonize(py, &events)?.unbind())
    }
}

/// Model format and prepared tokenizer, shared by all request parsers.
#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct ModelRecipe {
    tokenizer: RecipeTokenizer,
    encoder: Arc<dyn Encoder>,
    model: ModelFormat,
    options: Value,
}
impl ModelRecipe {
    fn request(&self, request: &Bound<'_, PyAny>) -> PyResult<Value> {
        let mut request: Value = depythonize(request)?;
        let map = request
            .as_object_mut()
            .ok_or_else(|| PyValueError::new_err("request must be a dictionary"))?;
        let mut extra = self.options.as_object().cloned().unwrap_or_default();
        match map.remove("extra") {
            Some(Value::Object(overrides)) => extra.extend(overrides),
            None | Some(Value::Null) => (),
            Some(_) => return Err(PyValueError::new_err("request.extra must be a dictionary")),
        }
        map.insert("extra".into(), Value::Object(extra));
        Ok(request)
    }
}
#[pymethods]
impl ModelRecipe {
    #[new]
    #[pyo3(signature = (tokenizer, *, model, options=None))]
    fn new(
        tokenizer: &RecipeTokenizer,
        model: &str,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let options: Value = options
            .map(depythonize)
            .transpose()?
            .unwrap_or(Value::Object(Map::new()));
        if !options.is_object() {
            return Err(PyValueError::new_err("options must be a dictionary"));
        }
        Ok(Self {
            tokenizer: tokenizer.clone(),
            encoder: tokenizer.encoder(),
            model: ModelFormat::parse(model).map_err(recipe_error)?,
            options,
        })
    }
    fn render_text(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<String> {
        let request = self.request(request)?;
        py.detach(|| {
            self.model
                .render(&request)
                .map(|prompt| prompt.rendered_text())
                .map_err(recipe_error)
        })
    }
    #[pyo3(signature = (request, *, media_replacements=None))]
    fn render_ids(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        media_replacements: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<Vec<u32>> {
        let request = self.request(request)?;
        py.detach(|| {
            self.model
                .render(&request)
                .and_then(|prompt| {
                    let prompt = if let Some(replacements) = &media_replacements {
                        self.expand_media(prompt, replacements)?
                    } else {
                        prompt
                    };
                    prompt.encode(self.encoder.as_ref())
                })
                .map_err(recipe_error)
        })
    }
    fn get_stop_token_ids(&self) -> PyResult<Vec<u32>> {
        self.model
            .stop_token_ids(self.encoder.as_ref())
            .map_err(recipe_error)
    }
    fn create_parser(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        prompt_ids: Vec<u32>,
    ) -> PyResult<NativeResponseParser> {
        let request = self.request(request)?;
        let parser = py.detach(|| {
            self.model
                .create_parser(
                    self.encoder.as_ref(),
                    self.tokenizer.decoder(),
                    &request,
                    &prompt_ids,
                )
                .map_err(recipe_error)
        })?;
        Ok(NativeResponseParser {
            inner: Arc::new(Mutex::new(parser)),
        })
    }
}

impl ModelRecipe {
    fn expand_media(
        &self,
        prompt: Prompt,
        replacements: &HashMap<String, Vec<String>>,
    ) -> Result<Prompt, Error> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        let mut expanded = Prompt::new();
        for span in prompt.spans {
            match span {
                PromptSpan::Text(text) => {
                    expanded.text(text);
                }
                PromptSpan::Special(token) => {
                    if let Some(values) = replacements.get(&token) {
                        let index = counts
                            .entry(replacements.get_key_value(&token).unwrap().0.as_str())
                            .or_default();
                        let value = values.get(*index).ok_or_else(|| {
                            Error::new(format!("missing media expansion for {token}"))
                        })?;
                        *index += 1;
                        match &self.tokenizer.source {
                            TokenizerSource::Native(_, encoder) => {
                                encoder.append_format(&mut expanded, value)
                            }
                            TokenizerSource::Python { .. } => {
                                expanded.text(value);
                            }
                        }
                    } else {
                        expanded.special(token);
                    }
                }
            }
        }
        for (marker, values) in replacements {
            if counts.get(marker.as_str()).copied().unwrap_or(0) != values.len() {
                return Err(Error::new(format!(
                    "media expansion count does not match {marker}"
                )));
            }
        }
        Ok(expanded)
    }
}
