//! Python protocol adapters; model-format state lives in resin-recipe.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::media_preprocess::{self, Kind as MediaKind, Processor as MediaProcessor};
use crate::recipe_events::{self, ParserEvent};
use crate::recipe_media;
use crate::recipe_types::{
    BlockKind, MediaSources, Message, MessagePolicy, RenderOptions, Role, Tool, ToolScope,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyAttributeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict},
};
use pythonize::{depythonize, pythonize};
use resin_recipe::{
    AttributedEvent, AttributedText, Decoder, Encoder, Error, FinishReason, InklingParser,
    InklingTokens, ModelFormat, ResponseParser, Tokenizer, TokenizerDecoder, TokenizerEncoder,
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

/// A tokenizer shared by rendering, response parsing, and token metadata.
///
/// Fast Hugging Face tokenizers are snapshotted into Rust. Custom Python
/// tokenizers remain supported through their decode method.
#[pyclass(frozen, skip_from_py_object, module = "resin._resin")]
#[derive(Clone)]
pub(crate) struct RecipeTokenizer {
    source: TokenizerSource,
    name_or_path: Option<String>,
}

#[pymethods]
impl RecipeTokenizer {
    #[new]
    pub(crate) fn new(tokenizer: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(prepared) = tokenizer.cast::<Self>() {
            return Ok(prepared.borrow().clone());
        }
        let name_or_path = optional_attr(tokenizer, "name_or_path")?
            .and_then(|value| value.extract::<String>().ok());
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
        Ok(Self {
            source,
            name_or_path,
        })
    }

    #[getter]
    fn is_native(&self) -> bool {
        matches!(self.source, TokenizerSource::Native(..))
    }

    #[getter]
    fn name_or_path(&self) -> Option<&str> {
        self.name_or_path.as_deref()
    }

    #[pyo3(signature = (text, *, add_special_tokens = false))]
    fn encode(&self, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => tokenizer
                .encode(text, add_special_tokens)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| PyValueError::new_err(error.to_string())),
            TokenizerSource::Python { object, .. } => Python::attach(|py| {
                let kwargs = PyDict::new(py);
                kwargs.set_item("add_special_tokens", add_special_tokens)?;
                object
                    .bind(py)
                    .call_method("encode", (text,), Some(&kwargs))?
                    .extract()
            }),
        }
    }

    #[pyo3(signature = (token_ids, *, skip_special_tokens = false))]
    fn decode(&self, token_ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => tokenizer
                .decode(&token_ids, skip_special_tokens)
                .map_err(|error| PyValueError::new_err(error.to_string())),
            TokenizerSource::Python { decode, .. } => Python::attach(|py| {
                let kwargs = PyDict::new(py);
                kwargs.set_item("skip_special_tokens", skip_special_tokens)?;
                decode.bind(py).call((token_ids,), Some(&kwargs))?.extract()
            }),
        }
    }

    fn token_to_id(&self, token: &str) -> PyResult<Option<u32>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => Ok(tokenizer.token_to_id(token)),
            TokenizerSource::Python { object, .. } => Python::attach(|py| {
                let tokenizer = object.bind(py);
                let method = match optional_attr(tokenizer, "convert_tokens_to_ids")? {
                    Some(method) => method,
                    None => tokenizer.getattr("token_to_id")?,
                };
                method.call1((token,))?.extract()
            }),
        }
    }

    fn id_to_token(&self, token_id: u32) -> PyResult<Option<String>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => Ok(tokenizer.id_to_token(token_id)),
            TokenizerSource::Python { object, .. } => Python::attach(|py| {
                let tokenizer = object.bind(py);
                let method = match optional_attr(tokenizer, "convert_ids_to_tokens")? {
                    Some(method) => method,
                    None => tokenizer.getattr("id_to_token")?,
                };
                method.call1((token_id,))?.extract()
            }),
        }
    }

    /// Serialize the prepared native snapshot, for external tokenizer tools.
    fn to_str(&self) -> PyResult<String> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => tokenizer
                .to_string(false)
                .map_err(|error| PyValueError::new_err(error.to_string())),
            TokenizerSource::Python { .. } => Err(PyValueError::new_err(
                "custom Python tokenizer has no serialized backend",
            )),
        }
    }
}

pub(crate) fn optional_attr<'py>(
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
    pub(crate) fn native(&self) -> Option<Arc<Tokenizer>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => Some(tokenizer.clone()),
            TokenizerSource::Python { .. } => None,
        }
    }

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
#[pyclass(name = "_NativeResponseParser", frozen, module = "resin._resin")]
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
    #[pyo3(signature=(token_ids, *, media=None))]
    fn new(token_ids: Vec<u32>, media: Option<Py<PyAny>>) -> Self {
        Self {
            token_ids,
            media,
            seed: None,
        }
    }
    #[getter]
    fn token_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(this.clone().into_any(), &this.get().token_ids)
    }
    #[getter]
    fn media(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.media.as_ref().map(|m| m.clone_ref(py))
    }
}

type OptionInitializer = fn(Python<'_>, &mut Map<String, Value>) -> PyResult<()>;
type OptionNormalizer = fn(&mut Map<String, Value>) -> PyResult<()>;

/// A concrete recipe selects its grammar, input contract, and optional media factory.
/// Shared rendering does not infer behavior from a checkpoint or model name.
struct RecipeDefinition {
    name: &'static str,
    format: ModelFormat,
    messages: MessagePolicy,
    options: &'static [&'static str],
    initialize_options: Option<OptionInitializer>,
    normalize_options: Option<OptionNormalizer>,
    media_kind: Option<MediaKind>,
}
impl RecipeDefinition {
    const fn new(
        name: &'static str,
        format: ModelFormat,
        options: &'static [&'static str],
    ) -> Self {
        Self {
            name,
            format,
            messages: MessagePolicy::STANDARD,
            options,
            initialize_options: None,
            normalize_options: None,
            media_kind: None,
        }
    }
}

/// Shared native implementation inherited by the concrete model recipes.
#[pyclass(name = "_NativeRecipe", frozen, subclass, module = "resin._resin")]
pub(crate) struct NativeRecipe {
    tokenizer: RecipeTokenizer,
    encoder: Arc<dyn Encoder>,
    definition: &'static RecipeDefinition,
    options: Map<String, Value>,
    id: u64,
    stop_token_ids: OnceLock<Vec<u32>>,
    media_processor: Option<MediaProcessor>,
}
impl NativeRecipe {
    fn new(
        py: Python<'_>,
        owner: &Bound<'_, PyAny>,
        definition: &'static RecipeDefinition,
        options: Option<&RenderOptions>,
        stop_token_ids: Option<Vec<u32>>,
        media_config: Option<Value>,
    ) -> PyResult<Self> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        if media_config.is_some() && definition.media_kind.is_none() {
            return Err(PyValueError::new_err(
                "this recipe does not accept media_config",
            ));
        }
        let original_tokenizer =
            optional_attr(owner, "tokenizer")?.unwrap_or_else(|| owner.clone());
        let tokenizer = RecipeTokenizer::new(&original_tokenizer)?;
        let mut defaults = options.map(|o| o.value.clone()).unwrap_or_default();
        if let Some(initialize) = definition.initialize_options {
            initialize(py, &mut defaults)?;
        }
        let value = Self {
            encoder: tokenizer.encoder(),
            tokenizer,
            definition,
            options: defaults,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            stop_token_ids: stop_token_ids.map(OnceLock::from).unwrap_or_default(),
            media_processor: definition
                .media_kind
                .map(|kind| MediaProcessor::new(kind, owner, media_config))
                .transpose()?,
        };
        value.resolve_options(None)?;
        Ok(value)
    }
    pub(crate) fn resolved_stop_token_ids(&self) -> PyResult<&[u32]> {
        if let Some(ids) = self.stop_token_ids.get() {
            return Ok(ids);
        }
        let ids = self
            .definition
            .format
            .stop_token_ids(self.encoder.as_ref())
            .map_err(recipe_error)?;
        // Resolve lazily so template inspection does not require a complete
        // tokenizer. Once published, the backing vector never changes.
        let _ = self.stop_token_ids.set(ids);
        Ok(self.stop_token_ids.get().unwrap())
    }
    pub(crate) fn message_from_wire(&self, py: Python<'_>, value: &Value) -> PyResult<Message> {
        let mut message = Message::from_wire(py, value)?;
        message.lower_wire_fields(&self.definition.messages);
        Ok(message)
    }
    fn resolve_options(&self, options: Option<&RenderOptions>) -> PyResult<Map<String, Value>> {
        let mut resolved = self.options.clone();
        if let Some(options) = options {
            resolved.extend(options.value.clone());
        }
        for key in resolved.keys() {
            if key != "add_generation_prompt" && !self.definition.options.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "{} does not support rendering option {key}",
                    self.definition.name
                )));
            }
        }
        if let Some(normalize) = self.definition.normalize_options {
            normalize(&mut resolved)?;
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
            message.validate_for(&self.definition.messages)?;
            let mut rendered = message.render_values(py, &mut sources)?;
            if !message.tools.is_empty() {
                if !matches!(message.role, Role::SYSTEM | Role::DEVELOPER) {
                    return Err(PyValueError::new_err(
                        "tool declarations belong to instruction messages",
                    ));
                }
                if matches!(
                    self.definition.messages.tools,
                    ToolScope::InitialInstruction
                ) {
                    if index != 0 {
                        return Err(PyValueError::new_err(format!(
                            "{} supports tool declarations only in the initial instruction message",
                            self.definition.name
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
    fn render_request(
        &self,
        py: Python<'_>,
        request: Value,
        sources: MediaSources,
    ) -> PyResult<RenderedPrompt> {
        let sources = media_preprocess::sources(py, sources)?;
        let (token_ids, preparation, placements) = py.detach(|| -> PyResult<_> {
            let preparation = if sources.is_empty() {
                None
            } else {
                let processor = self
                    .media_processor
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("this recipe does not support media"))?;
                Some(
                    processor
                        .process(sources, self.encoder.as_ref())
                        .map_err(recipe_error)?,
                )
            };
            let prompt = self
                .definition
                .format
                .render(&request)
                .map_err(recipe_error)?;
            let expansions = preparation
                .as_ref()
                .map(|p| p.expansions.as_slice())
                .unwrap_or(&[]);
            let (ids, placements) = recipe_media::encode(prompt, expansions, self.encoder.as_ref())
                .map_err(recipe_error)?;
            Ok((ids, preparation, placements))
        })?;
        let media = preparation
            .map(|p| p.bind(py, &token_ids, &placements))
            .transpose()?;
        // Parser creation needs controls, schemas and IDs, never images or conversation history.
        let seed = ParserSeed {
            recipe_id: self.id,
            request: json!({"tools":request["tools"],"extra":request["extra"]}),
        };
        Ok(RenderedPrompt {
            token_ids,
            media,
            seed: Some(seed),
        })
    }
}
#[pymethods]
impl NativeRecipe {
    #[getter]
    fn tokenizer(&self) -> RecipeTokenizer {
        self.tokenizer.clone()
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
        self.render_request(py, request, sources)
    }
    #[pyo3(signature=(messages, *, tools=Vec::new(), options=None))]
    /// Inspect the template before tokenization and media expansion.
    fn debug_render_text(
        &self,
        py: Python<'_>,
        messages: Vec<Py<Message>>,
        tools: Vec<Py<Tool>>,
        options: Option<&RenderOptions>,
    ) -> PyResult<String> {
        let (request, _) = self.request(py, &messages, &tools, options)?;
        py.detach(|| {
            self.definition
                .format
                .render(&request)
                .map(|prompt| prompt.rendered_text())
                .map_err(recipe_error)
        })
    }
    /// Static model stop tokens, returned as an owned list.
    #[getter]
    fn stop_token_ids(&self) -> PyResult<Vec<u32>> {
        Ok(self.resolved_stop_token_ids()?.to_vec())
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
            self.definition
                .format
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
    ($class:ident, $definition:expr) => {
        #[pyclass(extends=NativeRecipe, frozen, module="resin._resin")]
        pub(crate) struct $class;
        impl $class {
            const DEFINITION: RecipeDefinition = $definition;
        }
        #[pymethods]
        impl $class {
            #[new]
            #[pyo3(signature=(tokenizer, *, options=None, stop_token_ids=None, media_config=None))]
            fn new(
                py: Python<'_>,
                tokenizer: &Bound<'_, PyAny>,
                options: Option<&RenderOptions>,
                stop_token_ids: Option<Vec<u32>>,
                media_config: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<PyClassInitializer<Self>> {
                Ok(PyClassInitializer::from(NativeRecipe::new(
                    py,
                    tokenizer,
                    &Self::DEFINITION,
                    options,
                    stop_token_ids,
                    media_config.map(depythonize::<Value>).transpose()?,
                )?)
                .add_subclass(Self))
            }
        }
    };
}
fn conversation_date(py: Python<'_>, options: &mut Map<String, Value>) -> PyResult<()> {
    if !options.contains_key("conversation_start_date") {
        let date: String = py
            .import("datetime")?
            .getattr("date")?
            .call_method0("today")?
            .call_method0("isoformat")?
            .extract()?;
        options.insert("conversation_start_date".into(), Value::String(date));
    }
    Ok(())
}
fn deepseek_thinking(options: &mut Map<String, Value>) -> PyResult<()> {
    if let Some(enabled) = options.remove("enable_thinking") {
        let mode = if enabled == true { "thinking" } else { "chat" };
        if options.get("thinking_mode").is_some_and(|v| v != mode) {
            return Err(PyValueError::new_err(
                "conflicting enable_thinking and thinking_mode",
            ));
        }
        options.insert("thinking_mode".into(), json!(mode));
    }
    Ok(())
}
const DEEPSEEK: RecipeDefinition = RecipeDefinition {
    messages: MessagePolicy {
        roles: &[
            Role::SYSTEM,
            Role::DEVELOPER,
            Role::USER,
            Role::ASSISTANT,
            Role::TOOL,
            Role::LATEST_REMINDER,
        ],
        controls: &["task", "wo_eos", "response_format"],
        tools: ToolScope::AnyInstruction,
        ..MessagePolicy::STANDARD
    },
    normalize_options: Some(deepseek_thinking),
    ..RecipeDefinition::new(
        "deepseek_v4",
        ModelFormat::DeepSeek,
        &[
            "reasoning_effort",
            "thinking_mode",
            "enable_thinking",
            "response_format",
        ],
    )
};
const QWEN: RecipeDefinition = RecipeDefinition {
    media_kind: Some(MediaKind::Qwen),
    ..RecipeDefinition::new(
        "qwen3_5",
        ModelFormat::Qwen,
        &[
            "reasoning_effort",
            "enable_thinking",
            "preserve_thinking",
            "add_vision_id",
        ],
    )
};
recipe_class!(
    GptOssRecipe,
    RecipeDefinition {
        initialize_options: Some(conversation_date),
        ..RecipeDefinition::new(
            "gpt_oss",
            ModelFormat::GptOss,
            &[
                "reasoning_effort",
                "use_system_prompt",
                "model_identity",
                "knowledge_cutoff",
                "conversation_start_date",
                "auto_drop_analysis"
            ]
        )
    }
);
recipe_class!(DeepSeekV4Recipe, DEEPSEEK);
recipe_class!(
    Qwen35Recipe,
    RecipeDefinition {
        normalize_options: Some(|options| {
            options.insert("recipe_variant".into(), json!("qwen3_5"));
            Ok(())
        }),
        ..QWEN
    }
);
recipe_class!(
    Qwen38Recipe,
    RecipeDefinition {
        name: "qwen3_8",
        normalize_options: Some(|options| {
            options.insert("recipe_variant".into(), json!("qwen3_8"));
            Ok(())
        }),
        ..QWEN
    }
);
recipe_class!(
    InklingRecipe,
    RecipeDefinition {
        media_kind: Some(MediaKind::Inkling),
        ..RecipeDefinition::new("inkling", ModelFormat::Inkling, &["reasoning_effort"])
    }
);
recipe_class!(
    Lfm2Recipe,
    RecipeDefinition::new("lfm2", ModelFormat::Lfm2, &["preserve_thinking"])
);
#[pyclass(extends=NativeRecipe, frozen, module="resin._resin")]
pub(crate) struct Gemma4Recipe;
impl Gemma4Recipe {
    const DEFINITION: RecipeDefinition = RecipeDefinition {
        messages: MessagePolicy {
            assistant_order: &[
                BlockKind::Thinking,
                BlockKind::Call,
                BlockKind::Result,
                BlockKind::Text,
            ],
            reorder_wire_fields: true,
            ..MessagePolicy::STANDARD
        },
        media_kind: Some(MediaKind::Gemma),
        ..RecipeDefinition::new("gemma4", ModelFormat::Gemma, &["enable_thinking"])
    };
}
#[pymethods]
impl Gemma4Recipe {
    #[new]
    #[pyo3(signature=(tokenizer, *, options=None, stop_token_ids=None, media_config=None))]
    fn new(
        py: Python<'_>,
        tokenizer: &Bound<'_, PyAny>,
        options: Option<&RenderOptions>,
        stop_token_ids: Option<Vec<u32>>,
        media_config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(NativeRecipe::new(
            py,
            tokenizer,
            &Self::DEFINITION,
            options,
            stop_token_ids,
            media_config.map(depythonize::<Value>).transpose()?,
        )?)
        .add_subclass(Self))
    }
}
#[pyclass(extends=NativeRecipe, frozen, module="resin._resin")]
pub(crate) struct DeepSeekV41Recipe;
impl DeepSeekV41Recipe {
    const DEFINITION: RecipeDefinition = RecipeDefinition {
        name: "deepseek_v41",
        format: ModelFormat::DeepSeekV41,
        messages: MessagePolicy {
            media_roles: Some(&[Role::USER]),
            ..DEEPSEEK.messages
        },
        media_kind: Some(MediaKind::DeepSeek),
        ..DEEPSEEK
    };
}
#[pymethods]
impl DeepSeekV41Recipe {
    #[new]
    #[pyo3(signature=(tokenizer, *, options=None, stop_token_ids=None, vision_config=None, image_token_id=None))]
    fn new(
        py: Python<'_>,
        tokenizer: &Bound<'_, PyAny>,
        options: Option<&RenderOptions>,
        stop_token_ids: Option<Vec<u32>>,
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
            &Self::DEFINITION,
            options,
            stop_token_ids,
            Some(json!({"vision_config":vision_config,"image_token_id":image_token_id})),
        )?)
        .add_subclass(Self))
    }
}
