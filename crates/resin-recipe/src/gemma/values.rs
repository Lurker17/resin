//! Gemma's schema/value notation, whose strings use explicit quote tokens.

use crate::render::{list, text, truthy};
use crate::{Error, Prompt};
use serde_json::{Map, Value};

pub(super) fn sorted(map: &Map<String, Value>) -> Vec<(&str, &Value)> {
    let mut items: Vec<_> = map
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    // Jinja's dictsort is case-insensitive and stable for equivalent keys.
    items.sort_by_key(|(key, _)| key.to_lowercase());
    items
}

fn quoted(prompt: &mut Prompt, value: &str) {
    prompt.special("<|\"|>").text(value).special("<|\"|>");
}

pub(super) fn argument(prompt: &mut Prompt, value: &Value, escape_keys: bool) {
    match value {
        Value::String(value) => quoted(prompt, value),
        Value::Bool(value) => {
            prompt.text(if *value { "true" } else { "false" });
        }
        Value::Null => {
            prompt.text("None");
        }
        Value::Number(value) => {
            prompt.text(crate::json::spaced(&Value::Number(value.clone())));
        }
        Value::Array(values) => {
            prompt.text("[");
            prompt.separated(values, ",", |prompt, value| {
                argument(prompt, value, escape_keys);
            });
            prompt.text("]");
        }
        Value::Object(values) => {
            prompt.text("{");
            prompt.separated(sorted(values), ",", |prompt, (key, value)| {
                if escape_keys {
                    quoted(prompt, key);
                } else {
                    prompt.text(key);
                }
                prompt.text(":");
                argument(prompt, value, escape_keys);
            });
            prompt.text("}");
        }
    }
}

fn upper(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.to_uppercase(),
        Some(Value::Array(values)) => format!(
            "[{}]",
            values
                .iter()
                .map(|value| format!("'{}'", text(value).to_uppercase()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => String::new(),
    }
}

fn required(prompt: &mut Prompt, value: Option<&Value>) {
    prompt.text("required:[");
    prompt.separated(list(value.unwrap_or(&Value::Null)), ",", |prompt, item| {
        quoted(prompt, text(item))
    });
    prompt.text("]");
}

fn separator(prompt: &mut Prompt, has_fields: &mut bool) {
    if *has_fields {
        prompt.text(",");
    }
    *has_fields = true;
}

fn parameters(prompt: &mut Prompt, properties: &Map<String, Value>, filter_standard: bool) {
    let mut first = true;
    for (key, value) in sorted(properties) {
        if filter_standard
            && ["description", "type", "properties", "required", "nullable"].contains(&key)
        {
            continue;
        }
        if !first {
            prompt.text(",");
        }
        first = false;
        prompt.text(key).text(":{");
        let mut has_fields = false;
        if let Some(description) = value.get("description").filter(|value| truthy(value)) {
            prompt.text("description:");
            quoted(prompt, text(description));
            has_fields = true;
        }
        let kind = upper(value.get("type"));
        if kind == "STRING" {
            if let Some(enumeration) = value.get("enum").filter(|value| truthy(value)) {
                separator(prompt, &mut has_fields);
                prompt.text("enum:");
                argument(prompt, enumeration, true);
            }
        } else if kind == "ARRAY"
            && let Some(items) = value
                .get("items")
                .and_then(Value::as_object)
                .filter(|items| !items.is_empty())
        {
            separator(prompt, &mut has_fields);
            prompt.text("items:{");
            let mut first = true;
            for (key, value) in sorted(items)
                .into_iter()
                .filter(|(_, value)| !value.is_null())
            {
                if !first {
                    prompt.text(",");
                }
                first = false;
                match key {
                    "properties" => {
                        prompt.text("properties:{");
                        if let Some(properties) = value.as_object() {
                            parameters(prompt, properties, false);
                        }
                        prompt.text("}");
                    }
                    "required" => required(prompt, Some(value)),
                    "type" => {
                        prompt.text("type:");
                        if let Some(values) = value.as_array() {
                            argument(
                                prompt,
                                &Value::Array(
                                    values
                                        .iter()
                                        .map(|value| Value::String(upper(Some(value))))
                                        .collect(),
                                ),
                                true,
                            );
                        } else {
                            quoted(prompt, &upper(Some(value)));
                        }
                    }
                    _ => {
                        prompt.text(key).text(":");
                        argument(prompt, value, true);
                    }
                }
            }
            prompt.text("}");
        }
        if value.get("nullable").is_some_and(truthy) {
            separator(prompt, &mut has_fields);
            prompt.text("nullable:true");
        }
        if kind == "OBJECT" {
            if let Some(properties) = value.get("properties").and_then(Value::as_object) {
                separator(prompt, &mut has_fields);
                prompt.text("properties:{");
                parameters(prompt, properties, false);
                prompt.text("}");
            } else if let Some(properties) = value.as_object() {
                separator(prompt, &mut has_fields);
                prompt.text("properties:{");
                parameters(prompt, properties, true);
                prompt.text("}");
            }
            if value.get("required").is_some_and(truthy) {
                separator(prompt, &mut has_fields);
                required(prompt, value.get("required"));
            }
        }
        separator(prompt, &mut has_fields);
        prompt.text("type:");
        quoted(prompt, &kind);
        prompt.text("}");
    }
}

pub(super) fn declaration(prompt: &mut Prompt, function: &Value) {
    prompt.text(format!(
        "declaration:{}{{description:",
        text(&function["name"])
    ));
    quoted(prompt, text(&function["description"]));
    if let Some(params) = function.get("parameters").filter(|value| truthy(value)) {
        prompt.text(",parameters:{");
        if let Some(properties) = params
            .get("properties")
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty())
        {
            prompt.text("properties:{");
            parameters(prompt, properties, false);
            prompt.text("},");
        }
        if params.get("required").is_some_and(truthy) {
            required(prompt, params.get("required"));
            prompt.text(",");
        }
        if params.get("type").is_some_and(truthy) {
            prompt.text("type:");
            quoted(prompt, &upper(params.get("type")));
            prompt.text("}");
        }
    }
    if let Some(response) = function.get("response") {
        prompt.text(",response:{");
        if response.get("description").is_some_and(truthy) {
            prompt.text("description:");
            quoted(prompt, text(&response["description"]));
            prompt.text(",");
        }
        if upper(response.get("type")) == "OBJECT" {
            prompt.text("type:");
            quoted(prompt, &upper(response.get("type")));
            prompt.text("}");
        }
    }
    prompt.text("}");
}

/// Quotes are grammar only when they arrived as actual control-token IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Character {
    Text(char),
    Quote,
}

/// Normalize Gemma's explicit quote tokens, unquoted keys and `None` to JSON;
/// let serde validate the nested value grammar instead of maintaining a second
/// recursive JSON parser. Plain text resembling the quote token stays literal.
pub(super) fn parse_call(input: &[Character]) -> Result<(String, Value), Error> {
    let invalid = || Error::new("invalid Gemma tool call");
    let body = input
        .iter()
        .position(|unit| *unit == Character::Text('{'))
        .ok_or_else(invalid)?;
    let header: Option<String> = input[..body]
        .iter()
        .map(|unit| match unit {
            Character::Text(character) => Some(*character),
            Character::Quote => None,
        })
        .collect();
    let header = header.ok_or_else(invalid)?;
    let (call, name) = header.split_once(':').ok_or_else(invalid)?;
    let name = name.trim();
    if call.trim() != "call" || name.is_empty() || name.contains([':', ',', '[', ']']) {
        return Err(invalid());
    }
    let mut json = String::new();
    let mut cursor = body;
    while let Some(unit) = input.get(cursor) {
        match unit {
            Character::Quote => {
                cursor += 1;
                let start = cursor;
                while matches!(input.get(cursor), Some(Character::Text(_))) {
                    cursor += 1;
                }
                if input.get(cursor) != Some(&Character::Quote) {
                    return Err(invalid());
                }
                let text: String = input[start..cursor]
                    .iter()
                    .filter_map(|unit| match unit {
                        Character::Text(character) => Some(*character),
                        Character::Quote => None,
                    })
                    .collect();
                json.push_str(&serde_json::to_string(&text).unwrap());
                cursor += 1;
            }
            Character::Text(character) if "{}[],:".contains(*character) => {
                json.push(*character);
                cursor += 1;
            }
            _ => {
                let start = cursor;
                while matches!(input.get(cursor), Some(Character::Text(character)) if !"{}[],:".contains(*character))
                {
                    cursor += 1;
                }
                let atom: String = input[start..cursor]
                    .iter()
                    .filter_map(|unit| match unit {
                        Character::Text(character) => Some(*character),
                        Character::Quote => None,
                    })
                    .collect();
                let atom = atom.trim();
                if input.get(cursor) == Some(&Character::Text(':')) {
                    if atom.is_empty() {
                        return Err(invalid());
                    }
                    json.push_str(&serde_json::to_string(atom).unwrap());
                } else {
                    json.push_str(if atom == "None" { "null" } else { atom });
                }
            }
        }
    }
    let arguments: Value = serde_json::from_str(&json).map_err(|_| invalid())?;
    if !arguments.is_object() {
        return Err(invalid());
    }
    Ok((name.to_owned(), arguments))
}
