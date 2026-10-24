//! Native, immutable conversation inputs. JSON is confined to tool/schema data.

use std::sync::Arc;

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyTuple},
};
use pythonize::{depythonize, pythonize};
use serde_json::{Map, Value, json};

/// Conversation role. This enum is extensible: new roles may be added.
/// Code matching roles should include a fallback for unsupported roles.
/// Some roles are model-specific; recipes validate supported roles.
#[non_exhaustive]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[pyclass(frozen, eq, eq_int, from_py_object, module = "resin._resin")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Role {
    SYSTEM,
    DEVELOPER,
    USER,
    ASSISTANT,
    TOOL,
    LATEST_REMINDER,
}
impl Role {
    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::SYSTEM => "system",
            Self::DEVELOPER => "developer",
            Self::USER => "user",
            Self::ASSISTANT => "assistant",
            Self::TOOL => "tool",
            Self::LATEST_REMINDER => "latest_reminder",
        }
    }
    fn from_wire(value: &Value) -> PyResult<Self> {
        match value.as_str() {
            Some("system") => Ok(Self::SYSTEM),
            Some("developer") => Ok(Self::DEVELOPER),
            Some("user") => Ok(Self::USER),
            Some("assistant" | "model") => Ok(Self::ASSISTANT),
            Some("tool") => Ok(Self::TOOL),
            Some("latest_reminder") => Ok(Self::LATEST_REMINDER),
            _ => Err(PyValueError::new_err("unsupported message role")),
        }
    }
}

macro_rules! text_block {
    ($name:ident) => {
        #[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
        #[derive(Clone)]
        pub(crate) struct $name {
            pub(crate) text: String,
        }
        #[pymethods]
        impl $name {
            #[new]
            fn new(text: String) -> Self {
                Self { text }
            }
        }
    };
}
text_block!(Text);
text_block!(Thinking);

macro_rules! media_block {
    ($name:ident) => {
        #[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
        #[derive(Clone)]
        pub(crate) struct $name {
            pub(crate) source: Arc<Py<PyAny>>,
        }
        #[pymethods]
        impl $name {
            #[new]
            fn new(source: Py<PyAny>) -> Self {
                Self {
                    source: Arc::new(source),
                }
            }
            #[getter]
            fn source(&self, py: Python<'_>) -> Py<PyAny> {
                self.source.clone_ref(py)
            }
        }
    };
}
media_block!(Video);
media_block!(Audio);
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct Image {
    pub(crate) source: Arc<Py<PyAny>>,
}
#[pymethods]
impl Image {
    #[new]
    fn new(source: Py<PyAny>) -> Self {
        Self {
            source: Arc::new(source),
        }
    }
    #[staticmethod]
    #[pyo3(signature=(data, *, media_type))]
    fn from_bytes(py: Python<'_>, data: &[u8], media_type: &str) -> PyResult<Self> {
        let source = PyTuple::new(
            py,
            [
                PyBytes::new(py, data).into_any(),
                media_type.into_pyobject(py)?.into_any(),
            ],
        )?;
        Ok(Self::new(source.into_any().unbind()))
    }
    #[staticmethod]
    fn from_rgb(array: Py<PyAny>) -> Self {
        Self::new(array)
    }
    #[getter]
    fn source(&self, py: Python<'_>) -> Py<PyAny> {
        self.source.clone_ref(py)
    }
}

fn json_value(value: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    Ok(value.map(depythonize).transpose()?.unwrap_or(Value::Null))
}
fn json_get(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(pythonize(py, value)?.unbind())
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct Tool {
    pub(crate) value: Value,
}
#[pymethods]
impl Tool {
    #[new]
    #[pyo3(signature=(name, *, description=None, parameters=None, response=None, namespace=None))]
    fn new(
        name: String,
        description: Option<String>,
        parameters: Option<&Bound<'_, PyAny>>,
        response: Option<&Bound<'_, PyAny>>,
        namespace: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut value = json!({"name":name});
        if let Some(description) = description {
            value["description"] = Value::String(description);
        }
        for (key, data) in [
            ("parameters", parameters),
            ("response", response),
            ("namespace", namespace),
        ] {
            if let Some(data) = data {
                value[key] = depythonize(data)?;
            }
        }
        Self::from_wire(&value)
    }
    #[getter]
    fn name(&self) -> &str {
        self.value["name"].as_str().unwrap()
    }
    #[getter]
    fn description(&self) -> Option<&str> {
        self.value["description"].as_str()
    }
    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(py, &self.value["parameters"])
    }
    #[getter]
    fn response(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(py, &self.value["response"])
    }
    #[getter]
    fn namespace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(py, &self.value["namespace"])
    }
}
impl Tool {
    pub(crate) fn declaration(&self) -> Value {
        json!({"type":"function","function":self.value})
    }
    pub(crate) fn from_wire(value: &Value) -> PyResult<Self> {
        let mut function = value.get("function").unwrap_or(value).clone();
        if function["type"] == "function" {
            function.as_object_mut().unwrap().remove("type");
        }
        if function["name"].as_str().is_none_or(str::is_empty) {
            return Err(PyValueError::new_err("tool name must be a nonempty string"));
        }
        if let Some(namespace) = value.get("namespace").filter(|v| !v.is_null()) {
            function["namespace"] = namespace.clone();
        }
        Ok(Self { value: function })
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct ToolCall {
    pub(crate) value: Value,
}
#[pymethods]
impl ToolCall {
    #[new]
    #[pyo3(signature=(name, arguments, *, id=None, namespace=None))]
    fn new(
        name: String,
        arguments: &Bound<'_, PyAny>,
        id: Option<String>,
        namespace: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if name.is_empty() {
            return Err(PyValueError::new_err("tool call name must not be empty"));
        }
        let mut function = json!({"name":name,"arguments":depythonize::<Value>(arguments)?});
        if let Some(namespace) = namespace {
            function["namespace"] = depythonize(namespace)?;
        }
        Ok(Self {
            value: json!({"id":id,"type":"function","function":function}),
        })
    }
    #[getter]
    fn name(&self) -> &str {
        self.value["function"]["name"].as_str().unwrap_or("")
    }
    #[getter]
    fn id(&self) -> Option<&str> {
        self.value["id"].as_str()
    }
    #[getter]
    fn arguments(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(py, &self.value["function"]["arguments"])
    }
    #[getter]
    fn namespace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(py, &self.value["function"]["namespace"])
    }
}

impl ToolCall {
    fn from_wire(value: &Value) -> PyResult<Self> {
        let mut function = value.get("function").unwrap_or(value).clone();
        if function["name"].as_str().is_none_or(str::is_empty) {
            return Err(PyValueError::new_err(
                "tool call name must be a nonempty string",
            ));
        }
        if let Some(namespace) = value.get("namespace").filter(|v| !v.is_null()) {
            function["namespace"] = namespace.clone();
        }
        let id = value
            .get("id")
            .or_else(|| function.get("id"))
            .cloned()
            .unwrap_or(Value::Null);
        Ok(Self {
            value: json!({"id":id,"type":"function","function":function}),
        })
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct ToolResult {
    pub(crate) value: Value,
}
#[pymethods]
impl ToolResult {
    #[new]
    #[pyo3(signature=(content, *, call_id=None, name=None))]
    fn new(
        py: Python<'_>,
        content: &Bound<'_, PyAny>,
        call_id: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let content = match depythonize::<Value>(content) {
            Ok(value) => value,
            Err(_) => {
                let blocks = extract_blocks(content)?;
                let mut sources = Vec::new();
                let values = blocks
                    .iter()
                    .map(|block| block.to_value(py, &mut sources))
                    .collect::<PyResult<Vec<_>>>()?;
                if !sources.is_empty() {
                    return Err(PyValueError::new_err(
                        "media in tool results is not supported",
                    ));
                }
                Value::Array(values)
            }
        };
        Ok(Self {
            value: json!({"type":"tool_result","tool_use_id":call_id,"name":name,"content":content}),
        })
    }
    #[getter]
    fn call_id(&self) -> Option<&str> {
        self.value["tool_use_id"].as_str()
    }
    #[getter]
    fn name(&self) -> Option<&str> {
        self.value["name"].as_str()
    }
    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(py, &self.value["content"])
    }
}

#[derive(Clone)]
pub(crate) enum Block {
    Text(String),
    Thinking(String),
    Media(&'static str, Arc<Py<PyAny>>),
    Call(ToolCall),
    Result(ToolResult),
}
pub(crate) type MediaSources = Vec<(&'static str, Py<PyAny>)>;
pub(crate) fn media_blocks(py: Python<'_>, sources: MediaSources) -> PyResult<Vec<Py<PyAny>>> {
    sources
        .into_iter()
        .map(|(kind, source)| Block::Media(kind, Arc::new(source)).to_python(py))
        .collect()
}
impl Block {
    fn from_wire(py: Python<'_>, part: &Value) -> PyResult<Self> {
        if let Some(text) = part.as_str() {
            return Ok(Self::Text(text.into()));
        }
        match part["type"].as_str() {
            Some("text" | "input_text" | "output_text") => Ok(Self::Text(
                part["text"]
                    .as_str()
                    .ok_or_else(|| PyValueError::new_err("text block needs text"))?
                    .into(),
            )),
            Some("thinking") => Ok(Self::Thinking(
                part["thinking"].as_str().unwrap_or("").into(),
            )),
            Some(
                "image" | "image_url" | "input_image" | "video" | "audio" | "input_audio"
                | "audio_url",
            ) => {
                let kind = match part["type"].as_str().unwrap() {
                    "video" => "video",
                    "audio" | "input_audio" | "audio_url" => "audio",
                    _ => "image",
                };
                let source = part
                    .get(kind)
                    .or_else(|| part.get(format!("{kind}_url")))
                    .unwrap_or(&Value::Null);
                let source = source.get("url").unwrap_or(source);
                Ok(Self::Media(kind, Arc::new(pythonize(py, source)?.unbind())))
            }
            Some("tool_result") => Ok(Self::Result(ToolResult {
                value: part.clone(),
            })),
            _ => Err(PyValueError::new_err("unsupported content block")),
        }
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(v) = value.cast::<Text>() {
            return Ok(Self::Text(v.borrow().text.clone()));
        }
        if let Ok(v) = value.cast::<Thinking>() {
            return Ok(Self::Thinking(v.borrow().text.clone()));
        }
        if let Ok(v) = value.cast::<Image>() {
            return Ok(Self::Media("image", v.borrow().source.clone()));
        }
        if let Ok(v) = value.cast::<Video>() {
            return Ok(Self::Media("video", v.borrow().source.clone()));
        }
        if let Ok(v) = value.cast::<Audio>() {
            return Ok(Self::Media("audio", v.borrow().source.clone()));
        }
        if let Ok(v) = value.cast::<ToolCall>() {
            return Ok(Self::Call(v.borrow().clone()));
        }
        if let Ok(v) = value.cast::<ToolResult>() {
            return Ok(Self::Result(v.borrow().clone()));
        }
        Err(PyValueError::new_err(
            "message content must contain native content blocks",
        ))
    }
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Self::Text(text) => Py::new(py, Text { text: text.clone() })?.into_any(),
            Self::Thinking(text) => Py::new(py, Thinking { text: text.clone() })?.into_any(),
            Self::Media("image", source) => Py::new(
                py,
                Image {
                    source: source.clone(),
                },
            )?
            .into_any(),
            Self::Media("video", source) => Py::new(
                py,
                Video {
                    source: source.clone(),
                },
            )?
            .into_any(),
            Self::Media(_, source) => Py::new(
                py,
                Audio {
                    source: source.clone(),
                },
            )?
            .into_any(),
            Self::Call(call) => Py::new(py, call.clone())?.into_any(),
            Self::Result(result) => Py::new(py, result.clone())?.into_any(),
        })
    }
    fn to_value(&self, py: Python<'_>, sources: &mut MediaSources) -> PyResult<Value> {
        Ok(match self {
            Self::Text(text) => json!({"type":"text","text":text}),
            Self::Thinking(text) => json!({"type":"thinking","thinking":text}),
            Self::Media(kind, source) => {
                sources.push((*kind, source.clone_ref(py)));
                json!({"type":kind})
            }
            Self::Call(call) => call.value.clone(),
            Self::Result(result) => result.value.clone(),
        })
    }
}
fn extract_blocks(content: &Bound<'_, PyAny>) -> PyResult<Vec<Block>> {
    content
        .try_iter()?
        .map(|item| Block::from_py(&item?))
        .collect()
}

/// Structural requirements supplied by a concrete recipe, not inferred from model names.
#[derive(Clone, Copy)]
pub(crate) struct MessagePolicy {
    pub(crate) roles: &'static [Role],
    pub(crate) assistant_order: &'static [BlockKind],
    pub(crate) reorder_wire_fields: bool,
    pub(crate) controls: &'static [&'static str],
    pub(crate) tools: ToolScope,
    pub(crate) media_roles: Option<&'static [Role]>,
}
impl MessagePolicy {
    pub(crate) const STANDARD: Self = Self {
        roles: &[
            Role::SYSTEM,
            Role::DEVELOPER,
            Role::USER,
            Role::ASSISTANT,
            Role::TOOL,
        ],
        assistant_order: &[BlockKind::Thinking, BlockKind::Text, BlockKind::Call],
        reorder_wire_fields: false,
        controls: &[],
        tools: ToolScope::InitialInstruction,
        media_roles: None,
    };
}
#[derive(Clone, Copy)]
pub(crate) enum ToolScope {
    InitialInstruction,
    AnyInstruction,
}
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum BlockKind {
    Thinking,
    Text,
    Call,
    Result,
}
impl Block {
    fn kind(&self) -> BlockKind {
        match self {
            Self::Thinking(_) => BlockKind::Thinking,
            Self::Call(_) => BlockKind::Call,
            Self::Result(_) => BlockKind::Result,
            _ => BlockKind::Text,
        }
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct Message {
    pub(crate) role: Role,
    blocks: Vec<Block>,
    pub(crate) tools: Vec<Tool>,
    metadata: Map<String, Value>,
}
#[pymethods]
impl Message {
    #[new]
    #[pyo3(signature=(role, content, *, tools=Vec::new(), task=None, continuation=false, response_format=None))]
    fn new(
        py: Python<'_>,
        role: Role,
        content: &Bound<'_, PyAny>,
        tools: Vec<Py<Tool>>,
        task: Option<String>,
        continuation: bool,
        response_format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut metadata = Map::new();
        if let Some(task) = task {
            metadata.insert("task".into(), Value::String(task));
        }
        if continuation {
            metadata.insert("wo_eos".into(), Value::Bool(true));
        }
        if let Some(format) = response_format {
            metadata.insert("response_format".into(), depythonize(format)?);
        }
        Ok(Self {
            role,
            blocks: extract_blocks(content)?,
            tools: tools.iter().map(|t| t.borrow(py).clone()).collect(),
            metadata,
        })
    }
    #[getter]
    fn role(&self) -> Role {
        self.role
    }
    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.blocks
                .iter()
                .map(|block| block.to_python(py))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .unbind())
    }
    #[getter]
    fn tools(&self, py: Python<'_>) -> PyResult<Vec<Py<Tool>>> {
        self.tools
            .iter()
            .cloned()
            .map(|tool| Py::new(py, tool))
            .collect()
    }
    #[getter]
    fn task(&self) -> Option<&str> {
        self.metadata.get("task").and_then(Value::as_str)
    }
    #[getter]
    fn continuation(&self) -> bool {
        self.metadata.get("wo_eos") == Some(&Value::Bool(true))
    }
    #[getter]
    fn response_format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(
            py,
            self.metadata.get("response_format").unwrap_or(&Value::Null),
        )
    }
}
impl Message {
    pub(crate) fn from_wire(py: Python<'_>, value: &Value) -> PyResult<Self> {
        let role = Role::from_wire(&value["role"])?;
        let mut blocks = Vec::new();
        let content = value
            .get("content_blocks")
            .or_else(|| value.get("content"))
            .unwrap_or(&Value::Null);
        if role == Role::TOOL {
            blocks.push(Block::Result(ToolResult {
                value: json!({"type":"tool_result", "content":content,
                    "tool_use_id":value["tool_call_id"], "name":value["name"]}),
            }));
        } else {
            if let Some(reasoning) = value
                .get("reasoning_content")
                .or_else(|| value.get("reasoning"))
                .or_else(|| value.get("thinking"))
                .and_then(Value::as_str)
            {
                blocks.push(Block::Thinking(reasoning.into()));
            }
            match content {
                Value::Null => (),
                Value::String(text) => blocks.push(Block::Text(text.clone())),
                Value::Array(parts) => {
                    for part in parts {
                        blocks.push(Block::from_wire(py, part)?);
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "message content must be text or content blocks",
                    ));
                }
            }
        }
        for call in value["tool_calls"].as_array().into_iter().flatten() {
            blocks.push(Block::Call(ToolCall::from_wire(call)?));
        }
        for result in value["tool_responses"].as_array().into_iter().flatten() {
            blocks.push(Block::Result(ToolResult {
                value: json!({"type":"tool_result", "name":result["name"], "content":result["response"]}),
            }));
        }
        let tools = value["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(Tool::from_wire)
            .collect::<PyResult<Vec<_>>>()?;
        let metadata = ["task", "wo_eos", "response_format"]
            .into_iter()
            .filter_map(|key| value.get(key).map(|v| (key.into(), v.clone())))
            .collect();
        Ok(Self {
            role,
            blocks,
            tools,
            metadata,
        })
    }
    pub(crate) fn lower_wire_fields(&mut self, policy: &MessagePolicy) {
        // JSON named fields have no block order. Lower them into the concrete
        // template's declared order before exposing the same typed Message API.
        if self.role == Role::ASSISTANT && policy.reorder_wire_fields {
            self.blocks.sort_by_key(|block| {
                policy
                    .assistant_order
                    .iter()
                    .position(|kind| *kind == block.kind())
            });
        }
    }
    pub(crate) fn validate_for(&self, policy: &MessagePolicy) -> PyResult<()> {
        if !policy.roles.contains(&self.role) {
            return Err(PyValueError::new_err(format!(
                "this recipe does not support the {} role",
                self.role.spelling()
            )));
        }
        if self
            .metadata
            .iter()
            .any(|(key, value)| !value.is_null() && !policy.controls.contains(&key.as_str()))
        {
            return Err(PyValueError::new_err(
                "this recipe does not support these per-message controls",
            ));
        }
        if self.has_media()
            && policy
                .media_roles
                .is_some_and(|roles| !roles.contains(&self.role))
        {
            return Err(PyValueError::new_err(
                "this recipe does not support media in this message role",
            ));
        }
        if self.role != Role::ASSISTANT {
            return Ok(());
        }
        let mut previous = 0;
        for block in &self.blocks {
            let stage = policy.assistant_order.iter().position(|kind| *kind == block.kind())
                .ok_or_else(|| PyValueError::new_err(
                    "this recipe does not support this assistant block; use a tool message for tool results",
                ))?;
            if stage < previous {
                return Err(PyValueError::new_err(
                    "unsupported assistant block order for this recipe",
                ));
            }
            previous = stage;
        }
        Ok(())
    }
    pub(crate) fn render_values(
        &self,
        py: Python<'_>,
        sources: &mut MediaSources,
    ) -> PyResult<Vec<Value>> {
        let mut message = Value::Object(self.metadata.clone());
        message["role"] = Value::String(self.role.spelling().into());
        if !self.tools.is_empty() {
            message["tools"] = Value::Array(self.tools.iter().map(Tool::declaration).collect());
        }
        let mut content = Vec::new();
        let mut reasoning = None::<String>;
        let mut calls = Vec::new();
        let mut results = Vec::new();
        let mut emitted = Vec::new();
        for block in &self.blocks {
            match block {
                Block::Thinking(text) if self.role == Role::ASSISTANT => {
                    reasoning.get_or_insert_default().push_str(text)
                }
                Block::Call(call) => {
                    if self.role != Role::ASSISTANT {
                        return Err(PyValueError::new_err(
                            "tool calls belong to assistant messages",
                        ));
                    }
                    calls.push(call.value.clone());
                }
                Block::Result(result) if self.role == Role::ASSISTANT => results
                    .push(json!({"name":result.value["name"],"response":result.value["content"]})),
                Block::Result(result) => {
                    if !content.is_empty() {
                        let mut segment = message.clone();
                        segment["content"] = content_value(std::mem::take(&mut content));
                        emitted.push(segment);
                    }
                    emitted.push(json!({"role":"tool","tool_call_id":result.value["tool_use_id"],"name":result.value["name"],"content":result.value["content"]}));
                }
                _ => content.push(block.to_value(py, sources)?),
            }
        }
        if let Some(reasoning) = reasoning {
            message["reasoning_content"] = Value::String(reasoning);
        }
        if !calls.is_empty() {
            message["tool_calls"] = Value::Array(calls);
        }
        if !results.is_empty() {
            message["tool_responses"] = Value::Array(results);
        }
        if !content.is_empty() || emitted.is_empty() {
            message["content"] = content_value(content);
            emitted.push(message);
        }
        Ok(emitted)
    }
    pub(crate) fn has_media(&self) -> bool {
        self.blocks.iter().any(|b| matches!(b, Block::Media(..)))
    }
}
fn content_value(mut content: Vec<Value>) -> Value {
    if content.len() == 1 && content[0]["type"] == "text" {
        content.remove(0)["text"].take()
    } else {
        Value::Array(content)
    }
}

/// Explicit rendering controls; absent fields inherit recipe defaults.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Default)]
pub(crate) struct RenderOptions {
    pub(crate) value: Map<String, Value>,
}
#[pymethods]
impl RenderOptions {
    #[new]
    #[pyo3(signature=(*, reasoning_effort=None, enable_thinking=None, preserve_thinking=None, add_generation_prompt=None, add_vision_id=None, use_system_prompt=None, model_identity=None, knowledge_cutoff=None, conversation_start_date=None, auto_drop_analysis=None, thinking_mode=None, response_format=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        reasoning_effort: Option<&Bound<'_, PyAny>>,
        enable_thinking: Option<bool>,
        preserve_thinking: Option<bool>,
        add_generation_prompt: Option<bool>,
        add_vision_id: Option<bool>,
        use_system_prompt: Option<bool>,
        model_identity: Option<String>,
        knowledge_cutoff: Option<String>,
        conversation_start_date: Option<String>,
        auto_drop_analysis: Option<bool>,
        thinking_mode: Option<String>,
        response_format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut value = Map::new();
        for (key, v) in [
            ("enable_thinking", enable_thinking),
            ("preserve_thinking", preserve_thinking),
            ("add_generation_prompt", add_generation_prompt),
            ("add_vision_id", add_vision_id),
            ("use_system_prompt", use_system_prompt),
            ("auto_drop_analysis", auto_drop_analysis),
        ] {
            if let Some(v) = v {
                value.insert(key.into(), Value::Bool(v));
            }
        }
        for (key, v) in [
            ("model_identity", model_identity),
            ("knowledge_cutoff", knowledge_cutoff),
            ("conversation_start_date", conversation_start_date),
            ("thinking_mode", thinking_mode),
        ] {
            if let Some(v) = v {
                value.insert(key.into(), Value::String(v));
            }
        }
        for (key, v) in [
            ("reasoning_effort", reasoning_effort),
            ("response_format", response_format),
        ] {
            let v = json_value(v)?;
            if !v.is_null() {
                value.insert(key.into(), v);
            }
        }
        Self::validate(value)
    }
    #[getter]
    fn enable_thinking(&self) -> Option<bool> {
        self.value.get("enable_thinking").and_then(Value::as_bool)
    }
    #[getter]
    fn preserve_thinking(&self) -> Option<bool> {
        self.value.get("preserve_thinking").and_then(Value::as_bool)
    }
    #[getter]
    fn add_generation_prompt(&self) -> Option<bool> {
        self.value
            .get("add_generation_prompt")
            .and_then(Value::as_bool)
    }
    #[getter]
    fn add_vision_id(&self) -> Option<bool> {
        self.value.get("add_vision_id").and_then(Value::as_bool)
    }
    #[getter]
    fn use_system_prompt(&self) -> Option<bool> {
        self.value.get("use_system_prompt").and_then(Value::as_bool)
    }
    #[getter]
    fn auto_drop_analysis(&self) -> Option<bool> {
        self.value
            .get("auto_drop_analysis")
            .and_then(Value::as_bool)
    }
    #[getter]
    fn model_identity(&self) -> Option<&str> {
        self.value.get("model_identity").and_then(Value::as_str)
    }
    #[getter]
    fn knowledge_cutoff(&self) -> Option<&str> {
        self.value.get("knowledge_cutoff").and_then(Value::as_str)
    }
    #[getter]
    fn conversation_start_date(&self) -> Option<&str> {
        self.value
            .get("conversation_start_date")
            .and_then(Value::as_str)
    }
    #[getter]
    fn thinking_mode(&self) -> Option<&str> {
        self.value.get("thinking_mode").and_then(Value::as_str)
    }
    #[getter]
    fn reasoning_effort(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(
            py,
            self.value.get("reasoning_effort").unwrap_or(&Value::Null),
        )
    }
    #[getter]
    fn response_format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_get(
            py,
            self.value.get("response_format").unwrap_or(&Value::Null),
        )
    }
    fn __repr__(&self) -> String {
        format!("RenderOptions({})", Value::Object(self.value.clone()))
    }
}
impl RenderOptions {
    pub(crate) const FIELDS: &'static [&'static str] = &[
        "reasoning_effort",
        "enable_thinking",
        "preserve_thinking",
        "add_generation_prompt",
        "add_vision_id",
        "use_system_prompt",
        "model_identity",
        "knowledge_cutoff",
        "conversation_start_date",
        "auto_drop_analysis",
        "thinking_mode",
        "response_format",
    ];
    pub(crate) fn from_wire(request: &Value) -> PyResult<Self> {
        let mut value = Map::new();
        for key in Self::FIELDS {
            if let Some(v) = request
                .get(*key)
                .filter(|v| !v.is_null())
                .or_else(|| request["extra"].get(*key).filter(|v| !v.is_null()))
            {
                value.insert((*key).into(), v.clone());
            }
        }
        if !value.contains_key("thinking_mode") && request["extra"]["thinking"]["type"] == "enabled"
        {
            value.insert("thinking_mode".into(), json!("thinking"));
        }
        Self::validate(value)
    }
    fn validate(value: Map<String, Value>) -> PyResult<Self> {
        for (key, v) in &value {
            match key.as_str() {
                "reasoning_effort" if !v.is_string() && !v.is_number() => {
                    return Err(PyValueError::new_err(
                        "reasoning_effort must be a name or number",
                    ));
                }
                "enable_thinking"
                | "preserve_thinking"
                | "add_generation_prompt"
                | "add_vision_id"
                | "use_system_prompt"
                | "auto_drop_analysis"
                    if !v.is_boolean() =>
                {
                    return Err(PyValueError::new_err(format!("{key} must be a bool")));
                }
                "model_identity" | "knowledge_cutoff" | "conversation_start_date"
                    if !v.is_string() =>
                {
                    return Err(PyValueError::new_err(format!("{key} must be a string")));
                }
                "thinking_mode" if !matches!(v.as_str(), Some("thinking" | "chat")) => {
                    return Err(PyValueError::new_err(
                        "thinking_mode must be thinking or chat",
                    ));
                }
                _ => (),
            }
        }
        Ok(Self { value })
    }
}
