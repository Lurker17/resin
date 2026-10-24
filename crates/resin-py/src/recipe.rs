//! Python protocol adapters; model-format state lives in resin-recipe.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::recipe_events::{self, ParserEvent};
use crate::recipe_types::{MediaSources, Message, RenderOptions, Role, Tool};
use pyo3::{
    exceptions::{PyAttributeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict},
};
use pythonize::{depythonize, pythonize};
use resin_recipe::{
    AttributedEvent, AttributedText, Decoder, Encoder, Error, FinishReason, InklingParser,
    InklingTokens, ModelFormat, Prompt, PromptSpan, ResponseParser, Tokenizer, TokenizerDecoder,
    TokenizerEncoder,
};
use serde_json::{Map, Value, json};

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
    fn decoder(&self) -> Result<Box<dyn Decoder>, Error> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => {
                Ok(Box::new(TokenizerDecoder::new(tokenizer.clone())?))
            }
            TokenizerSource::Python { decode, .. } => Ok(Box::new(PythonDecoder {
                decode: decode.clone(),
                ids: Vec::new(),
                emitted: String::new(),
            })),
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
    fn push(&mut self, ids: &[u32], _token_start: u32) -> Result<AttributedText, Error> {
        self.ids.extend_from_slice(ids);
        let text = self.decoded()?;
        self.delta(text.trim_end_matches('\u{fffd}'))
            .map(AttributedText::plain)
    }

    fn finish(&mut self) -> Result<AttributedText, Error> {
        if self.ids.is_empty() {
            return Ok(AttributedText::default());
        }
        let text = self.decoded()?;
        let delta = self.delta(&text)?;
        self.reset();
        Ok(AttributedText::plain(delta))
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
        let parser = InklingParser::new(
            tokens,
            tokenizer.decoder().map_err(recipe_error)?,
            &prompt_ids,
        )
        .map_err(recipe_error)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Box::new(parser))),
        })
    }

    fn feed(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Vec<Py<ParserEvent>>> {
        let events = py.detach(|| feed(&self.inner, &token_ids))?;
        recipe_events::to_python(py, events)
    }

    fn finish(&self, py: Python<'_>, reason: &str) -> PyResult<Vec<Py<ParserEvent>>> {
        let reason = parse_finish_reason(reason)?;
        let events = py.detach(|| finish(&self.inner, reason))?;
        recipe_events::to_python(py, events)
    }
}

pub(crate) fn feed(parser: &SharedParser, ids: &[u32]) -> PyResult<Vec<AttributedEvent>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .feed(ids)
        .map_err(recipe_error)
}

pub(crate) fn finish(
    parser: &SharedParser,
    reason: FinishReason,
) -> PyResult<Vec<AttributedEvent>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .finish(reason)
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
    fn feed(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Vec<Py<ParserEvent>>> {
        let events = py.detach(|| feed(&self.inner, &token_ids))?;
        recipe_events::to_python(py, events)
    }

    fn finish(&self, py: Python<'_>, reason: &str) -> PyResult<Vec<Py<ParserEvent>>> {
        let reason = parse_finish_reason(reason)?;
        let events = py.detach(|| finish(&self.inner, reason))?;
        recipe_events::to_python(py, events)
    }
}

/// Rendered IDs and optional prepared media. Native parser state contains no pixels.
#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct RenderedPrompt {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) stop_token_ids: Vec<u32>,
    pub(crate) media: Option<Py<PyAny>>,
    seed: Option<ParserSeed>,
}
struct ParserSeed {
    recipe_id: u64,
    request: Value,
}
#[pymethods]
impl RenderedPrompt {
    #[new]
    #[pyo3(signature=(token_ids, *, stop_token_ids=Vec::new(), media=None))]
    fn new(token_ids: Vec<u32>, stop_token_ids: Vec<u32>, media: Option<Py<PyAny>>) -> Self {
        Self {
            token_ids,
            stop_token_ids,
            media,
            seed: None,
        }
    }
    #[getter]
    fn token_ids(&self) -> Vec<u32> {
        self.token_ids.clone()
    }
    #[getter]
    fn stop_token_ids(&self) -> Vec<u32> {
        self.stop_token_ids.clone()
    }
    #[getter]
    fn media(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.media.as_ref().map(|m| m.clone_ref(py))
    }
}

/// Shared native implementation inherited by the concrete model recipes.
#[pyclass(name = "_NativeRecipe", frozen, subclass, module = "resin._resin")]
pub(crate) struct NativeRecipe {
    tokenizer: RecipeTokenizer,
    original_tokenizer: Arc<Py<PyAny>>,
    encoder: Arc<dyn Encoder>,
    model: ModelFormat,
    name: &'static str,
    options: Map<String, Value>,
    id: u64,
    stop_token_ids: Option<Vec<u32>>,
    media_owner: Arc<Py<PyAny>>,
    media_config: Value,
    media_preparer: Mutex<Option<Py<PyAny>>>,
}
impl NativeRecipe {
    fn new(
        py: Python<'_>,
        owner: &Bound<'_, PyAny>,
        name: &'static str,
        options: Option<&RenderOptions>,
        stop_token_ids: Option<Vec<u32>>,
        media_config: Value,
    ) -> PyResult<Self> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let original_tokenizer =
            optional_attr(owner, "tokenizer")?.unwrap_or_else(|| owner.clone());
        let tokenizer = RecipeTokenizer::new(&original_tokenizer)?;
        let mut defaults = options.map(|o| o.value.clone()).unwrap_or_default();
        if name == "gpt_oss" && !defaults.contains_key("conversation_start_date") {
            let date: String = py
                .import("datetime")?
                .getattr("date")?
                .call_method0("today")?
                .call_method0("isoformat")?
                .extract()?;
            defaults.insert("conversation_start_date".into(), Value::String(date));
        }
        let value = Self {
            encoder: tokenizer.encoder(),
            tokenizer,
            original_tokenizer: Arc::new(original_tokenizer.unbind()),
            model: ModelFormat::parse(if name == "qwen3_8" { "qwen3_5" } else { name })
                .map_err(recipe_error)?,
            name,
            options: defaults,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            stop_token_ids,
            media_owner: Arc::new(owner.clone().unbind()),
            media_config,
            media_preparer: Mutex::new(None),
        };
        value.resolve_options(None)?;
        Ok(value)
    }
    pub(crate) fn message_from_wire(&self, py: Python<'_>, value: &Value) -> PyResult<Message> {
        let mut message = Message::from_wire(py, value)?;
        message.lower_wire_fields(self.name);
        Ok(message)
    }
    fn resolve_options(&self, options: Option<&RenderOptions>) -> PyResult<Map<String, Value>> {
        let mut resolved = self.options.clone();
        if let Some(options) = options {
            resolved.extend(options.value.clone());
        }
        for key in resolved.keys() {
            let supported = key == "add_generation_prompt"
                || match self.name {
                    "gpt_oss" => matches!(
                        key.as_str(),
                        "reasoning_effort"
                            | "use_system_prompt"
                            | "model_identity"
                            | "knowledge_cutoff"
                            | "conversation_start_date"
                            | "auto_drop_analysis"
                    ),
                    "qwen3_5" | "qwen3_8" => matches!(
                        key.as_str(),
                        "reasoning_effort"
                            | "enable_thinking"
                            | "preserve_thinking"
                            | "add_vision_id"
                    ),
                    "deepseek_v4" | "deepseek_v41" => matches!(
                        key.as_str(),
                        "reasoning_effort"
                            | "thinking_mode"
                            | "enable_thinking"
                            | "response_format"
                    ),
                    "gemma4" => key == "enable_thinking",
                    "inkling" => key == "reasoning_effort",
                    "lfm2" => key == "preserve_thinking",
                    _ => false,
                };
            if !supported {
                return Err(PyValueError::new_err(format!(
                    "{} does not support rendering option {key}",
                    self.name
                )));
            }
        }
        if self.name.starts_with("qwen3_") {
            resolved.insert("recipe_variant".into(), json!(self.name));
        }
        if self.name.starts_with("deepseek_")
            && let Some(enabled) = resolved.remove("enable_thinking")
        {
            let mode = if enabled == true { "thinking" } else { "chat" };
            if resolved.get("thinking_mode").is_some_and(|v| v != mode) {
                return Err(PyValueError::new_err(
                    "conflicting enable_thinking and thinking_mode",
                ));
            }
            resolved.insert("thinking_mode".into(), json!(mode));
        }
        Ok(resolved)
    }
    fn request(
        &self,
        py: Python<'_>,
        messages: &[Py<Message>],
        tools: &[Py<Tool>],
        options: Option<&RenderOptions>,
    ) -> PyResult<(Value, MediaSources)> {
        let mut values = Vec::new();
        let mut sources = Vec::new();
        let mut definitions = tools
            .iter()
            .map(|t| t.borrow(py).declaration())
            .collect::<Vec<_>>();
        for (index, message) in messages.iter().enumerate() {
            let message = message.borrow(py);
            message.validate_for(self.name)?;
            if self.name == "deepseek_v41" && message.has_media() && message.role != Role::USER {
                return Err(PyValueError::new_err(
                    "DeepSeek V4.1 images belong in user messages",
                ));
            }
            let mut rendered = message.render_values(py, &mut sources)?;
            if !message.tools.is_empty() {
                if !matches!(message.role, Role::SYSTEM | Role::DEVELOPER) {
                    return Err(PyValueError::new_err(
                        "tool declarations belong to instruction messages",
                    ));
                }
                if !self.name.starts_with("deepseek_") {
                    if index != 0 {
                        return Err(PyValueError::new_err(format!(
                            "{} supports tool declarations only in the initial instruction message",
                            self.name
                        )));
                    }
                    if !definitions.is_empty() {
                        return Err(PyValueError::new_err(
                            "pass tools either on the initial message or to render, not both",
                        ));
                    }
                    definitions = message.tools.iter().map(Tool::declaration).collect();
                    for item in &mut rendered {
                        item.as_object_mut().unwrap().remove("tools");
                    }
                }
            }
            values.extend(rendered);
        }
        Ok((
            json!({"messages":values,"tools":definitions,"extra":self.resolve_options(options)?}),
            sources,
        ))
    }
    fn preparer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut stored = self
            .media_preparer
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(preparer) = stored.as_ref() {
            return Ok(preparer.clone_ref(py));
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            "eos_token_ids",
            self.stop_token_ids.as_deref().unwrap_or(&[]),
        )?;
        kwargs.set_item(
            "vision_config",
            pythonize(py, &self.media_config["vision_config"])?,
        )?;
        kwargs.set_item(
            "image_token_id",
            pythonize(py, &self.media_config["image_token_id"])?,
        )?;
        let preparer = py
            .import("resin._recipe_media")?
            .getattr("make_preparer")?
            .call((self.name, self.media_owner.bind(py)), Some(&kwargs))?
            .unbind();
        *stored = Some(preparer.clone_ref(py));
        Ok(preparer)
    }
    fn render_request(
        &self,
        py: Python<'_>,
        request: Value,
        sources: MediaSources,
        prepare: bool,
    ) -> PyResult<RenderedPrompt> {
        let mut preparation = None;
        let mut replacements = None;
        if prepare && !sources.is_empty() {
            let preparer = self.preparer(py)?;
            let result = preparer.bind(py).call_method1("prepare", (sources,))?;
            let (markers, state): (HashMap<String, Vec<String>>, Py<PyAny>) = result.extract()?;
            replacements = Some(markers);
            preparation = Some((preparer, state));
        }
        let token_ids = py.detach(|| {
            let prompt = self.model.render(&request).map_err(recipe_error)?;
            let prompt = if let Some(replacements) = &replacements {
                self.expand_media(prompt, replacements)
                    .map_err(recipe_error)?
            } else {
                prompt
            };
            prompt.encode(self.encoder.as_ref()).map_err(recipe_error)
        })?;
        let media = if let Some((preparer, state)) = preparation {
            let media = preparer
                .bind(py)
                .call_method1("finalize", (state, &token_ids))?;
            (!media.is_none()).then(|| media.unbind())
        } else {
            None
        };
        // Parser creation needs controls, schemas and IDs, never images or conversation history.
        let seed = ParserSeed {
            recipe_id: self.id,
            request: json!({"tools":request["tools"],"extra":request["extra"]}),
        };
        Ok(RenderedPrompt {
            token_ids,
            stop_token_ids: self.get_stop_token_ids()?,
            media,
            seed: Some(seed),
        })
    }
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
#[pymethods]
impl NativeRecipe {
    #[getter]
    fn tokenizer(&self, py: Python<'_>) -> Py<PyAny> {
        self.original_tokenizer.clone_ref(py)
    }
    #[pyo3(signature=(messages, *, tools=Vec::new(), options=None))]
    pub(crate) fn render(
        &self,
        py: Python<'_>,
        messages: Vec<Py<Message>>,
        tools: Vec<Py<Tool>>,
        options: Option<&RenderOptions>,
    ) -> PyResult<RenderedPrompt> {
        let (request, sources) = self.request(py, &messages, &tools, options)?;
        self.render_request(py, request, sources, true)
    }
    #[pyo3(signature=(messages, *, tools=Vec::new(), options=None))]
    fn render_text(
        &self,
        py: Python<'_>,
        messages: Vec<Py<Message>>,
        tools: Vec<Py<Tool>>,
        options: Option<&RenderOptions>,
    ) -> PyResult<String> {
        let (request, _) = self.request(py, &messages, &tools, options)?;
        py.detach(|| {
            self.model
                .render(&request)
                .map(|prompt| prompt.rendered_text())
                .map_err(recipe_error)
        })
    }
    #[pyo3(signature=(messages, *, tools=Vec::new(), options=None))]
    fn render_ids(
        &self,
        py: Python<'_>,
        messages: Vec<Py<Message>>,
        tools: Vec<Py<Tool>>,
        options: Option<&RenderOptions>,
    ) -> PyResult<Vec<u32>> {
        let (request, sources) = self.request(py, &messages, &tools, options)?;
        let prepare = sources.iter().any(|(_, source)| !source.is_none(py));
        Ok(self
            .render_request(py, request, sources, prepare)?
            .token_ids)
    }
    pub(crate) fn get_stop_token_ids(&self) -> PyResult<Vec<u32>> {
        if let Some(ids) = &self.stop_token_ids {
            return Ok(ids.clone());
        }
        self.model
            .stop_token_ids(self.encoder.as_ref())
            .map_err(recipe_error)
    }
    pub(crate) fn create_parser(
        &self,
        py: Python<'_>,
        prompt: &RenderedPrompt,
    ) -> PyResult<NativeResponseParser> {
        let seed = prompt
            .seed
            .as_ref()
            .filter(|seed| seed.recipe_id == self.id)
            .ok_or_else(|| {
                PyValueError::new_err("prompt must have been rendered by this recipe instance")
            })?;
        let parser = py.detach(|| {
            self.model
                .create_parser(
                    self.encoder.as_ref(),
                    self.tokenizer.decoder().map_err(recipe_error)?,
                    &seed.request,
                    &prompt.token_ids,
                )
                .map_err(recipe_error)
        })?;
        Ok(NativeResponseParser {
            inner: Arc::new(Mutex::new(parser)),
        })
    }
    #[pyo3(signature=(token_ids, *, prompt))]
    fn parse_response(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        prompt: &RenderedPrompt,
    ) -> PyResult<Py<PyAny>> {
        let parser = self.create_parser(py, prompt)?;
        let events = py.detach(|| {
            let mut events = feed(&parser.inner, &token_ids)?;
            events.extend(finish(&parser.inner, FinishReason::Stop)?);
            Ok::<_, PyErr>(events)
        })?;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::<Value>::new();
        let mut indices = HashMap::new();
        for event in events {
            let value = serde_json::to_value(event)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            match value["type"].as_str() {
                Some("text_delta") => content.push_str(value["delta"].as_str().unwrap_or("")),
                Some("reasoning_delta") => {
                    reasoning.push_str(value["delta"].as_str().unwrap_or(""))
                }
                Some("tool_call_start") => {
                    indices.insert(value["index"].as_u64().unwrap_or(0), calls.len());
                    calls.push(json!({"id":value["id"],"name":value["name"],"arguments":""}));
                }
                Some("tool_call_arguments_delta") => {
                    if let Some(index) = indices.get(&value["index"].as_u64().unwrap_or(0)) {
                        let previous = calls[*index]["arguments"].as_str().unwrap_or("");
                        calls[*index]["arguments"] = json!(format!(
                            "{previous}{}",
                            value["delta"].as_str().unwrap_or("")
                        ));
                    }
                }
                _ => (),
            }
        }
        Ok(pythonize(
            py,
            &json!({"content":content,"reasoning_content":reasoning,"tool_calls":calls}),
        )?
        .unbind())
    }
}

macro_rules! recipe_class {
    ($class:ident,$model:literal) => {
        #[pyclass(extends=NativeRecipe, frozen, module="resin._resin")]
        pub(crate) struct $class;
        #[pymethods]
        impl $class {
            #[new]
            #[pyo3(signature=(tokenizer, *, options=None))]
            fn new(
                py: Python<'_>,
                tokenizer: &Bound<'_, PyAny>,
                options: Option<&RenderOptions>,
            ) -> PyResult<PyClassInitializer<Self>> {
                Ok(PyClassInitializer::from(NativeRecipe::new(
                    py,
                    tokenizer,
                    $model,
                    options,
                    None,
                    Value::Null,
                )?)
                .add_subclass(Self))
            }
        }
    };
}
recipe_class!(GptOssRecipe, "gpt_oss");
recipe_class!(DeepSeekV4Recipe, "deepseek_v4");
recipe_class!(Qwen35Recipe, "qwen3_5");
recipe_class!(Qwen38Recipe, "qwen3_8");
recipe_class!(InklingRecipe, "inkling");
recipe_class!(Lfm2Recipe, "lfm2");
#[pyclass(extends=NativeRecipe, frozen, module="resin._resin")]
pub(crate) struct Gemma4Recipe;
#[pymethods]
impl Gemma4Recipe {
    #[new]
    #[pyo3(signature=(tokenizer, *, options=None, eos_token_ids=Vec::new()))]
    fn new(
        py: Python<'_>,
        tokenizer: &Bound<'_, PyAny>,
        options: Option<&RenderOptions>,
        eos_token_ids: Vec<u32>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(NativeRecipe::new(
            py,
            tokenizer,
            "gemma4",
            options,
            (!eos_token_ids.is_empty()).then_some(eos_token_ids),
            Value::Null,
        )?)
        .add_subclass(Self))
    }
}
#[pyclass(extends=NativeRecipe, frozen, module="resin._resin")]
pub(crate) struct DeepSeekV41Recipe;
#[pymethods]
impl DeepSeekV41Recipe {
    #[new]
    #[pyo3(signature=(tokenizer, *, options=None, vision_config=None, image_token_id=None))]
    fn new(
        py: Python<'_>,
        tokenizer: &Bound<'_, PyAny>,
        options: Option<&RenderOptions>,
        vision_config: Option<&Bound<'_, PyAny>>,
        image_token_id: Option<u32>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let vision_config = vision_config
            .map(depythonize::<Value>)
            .transpose()?
            .unwrap_or(Value::Null);
        Ok(PyClassInitializer::from(NativeRecipe::new(
            py,
            tokenizer,
            "deepseek_v41",
            options,
            None,
            json!({"vision_config":vision_config,"image_token_id":image_token_id}),
        )?)
        .add_subclass(Self))
    }
}
