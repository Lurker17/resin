//! Small input conventions shared by the model renderers.

use crate::Error;
use serde_json::{Map, Value};
use std::borrow::Cow;

pub(crate) fn option<'a>(request: &'a Value, key: &str) -> Option<&'a Value> {
    request
        .get(key)
        .filter(|value| !value.is_null())
        .or_else(|| {
            request
                .get("extra")?
                .get(key)
                .filter(|value| !value.is_null())
        })
}

pub(crate) fn messages<'a>(request: &'a Value, model: &str) -> Result<&'a [Value], Error> {
    request["messages"]
        .as_array()
        .filter(|messages| !messages.is_empty())
        .map(Vec::as_slice)
        .ok_or_else(|| Error::new(format!("{model} requires nonempty messages")))
}

pub(crate) fn list(value: &Value) -> &[Value] {
    value.as_array().map(Vec::as_slice).unwrap_or(&[])
}

pub(crate) fn text(value: &Value) -> &str {
    value.as_str().unwrap_or("")
}

pub(crate) fn function(value: &Value) -> &Value {
    value.get("function").unwrap_or(value)
}

pub(crate) fn value_or(value: &Value, fallback: Value) -> Value {
    if value.is_null() {
        fallback
    } else {
        value.clone()
    }
}

pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64().unwrap_or(0.0) != 0.0,
    }
}

/// Decode JSON-string arguments without cloning an already structured object.
pub(crate) fn object_arguments(value: &Value) -> Result<Cow<'_, Map<String, Value>>, Error> {
    match value {
        Value::Object(value) => Ok(Cow::Borrowed(value)),
        Value::String(text) => serde_json::from_str(text)
            .map(Cow::Owned)
            .map_err(|_| Error::new("tool call arguments must be a JSON object")),
        _ => Err(Error::new("tool call arguments must be a JSON object")),
    }
}

/// Common text forms; each model decides how to handle other content blocks.
pub(crate) fn text_content(
    value: &Value,
    other: impl Fn(&Value) -> Result<String, Error>,
) -> Result<String, Error> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| {
                if block["type"] == "text" {
                    block["text"]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| Error::new("text content must be a string"))
                } else {
                    other(block)
                }
            })
            .collect(),
        _ => Err(Error::new("message content must be text or content blocks")),
    }
}

/// Python/Jinja whitespace includes the ASCII information separators.
pub(crate) fn whitespace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}
