//! Gemma's schema/value notation, whose strings use explicit quote tokens.

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

fn text(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("")
}
fn truthy(value: &Value) -> bool {
    super::truthy(value)
}
fn list(value: Option<&Value>) -> &[Value] {
    super::array(value)
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
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    prompt.text(",");
                }
                argument(prompt, value, escape_keys);
            }
            prompt.text("]");
        }
        Value::Object(values) => {
            prompt.text("{");
            for (index, (key, value)) in sorted(values).into_iter().enumerate() {
                if index > 0 {
                    prompt.text(",");
                }
                if escape_keys {
                    quoted(prompt, key);
                } else {
                    prompt.text(key);
                }
                prompt.text(":");
                argument(prompt, value, escape_keys);
            }
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
                .map(|value| format!("'{}'", text(Some(value)).to_uppercase()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => String::new(),
    }
}

fn required(prompt: &mut Prompt, value: Option<&Value>) {
    prompt.text("required:[");
    for (index, item) in list(value).iter().enumerate() {
        if index > 0 {
            prompt.text(",");
        }
        quoted(prompt, text(Some(item)));
    }
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
            quoted(prompt, text(Some(description)));
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
        text(function.get("name"))
    ));
    quoted(prompt, text(function.get("description")));
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
            quoted(prompt, text(response.get("description")));
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

struct Reader<'a> {
    input: &'a [Character],
    position: usize,
}

impl Reader<'_> {
    fn whitespace(&mut self) {
        while matches!(self.input.get(self.position), Some(Character::Text(character)) if character.is_whitespace())
        {
            self.position += 1;
        }
    }

    fn take(&mut self, expected: char) -> bool {
        self.whitespace();
        if self.input.get(self.position) == Some(&Character::Text(expected)) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn require(&mut self, expected: char) -> Result<(), Error> {
        if self.take(expected) {
            Ok(())
        } else {
            Err(Error::new(format!("Gemma tool call expected {expected:?}")))
        }
    }

    fn quoted(&mut self) -> Result<String, Error> {
        self.whitespace();
        if self.input.get(self.position) != Some(&Character::Quote) {
            return Err(Error::new(
                "Gemma tool string must start with its quote token",
            ));
        }
        self.position += 1;
        let mut value = String::new();
        while let Some(character) = self.input.get(self.position) {
            self.position += 1;
            match character {
                Character::Text(character) => value.push(*character),
                Character::Quote => return Ok(value),
            }
        }
        Err(Error::new("unterminated Gemma tool string"))
    }

    fn atom(&mut self) -> Result<String, Error> {
        self.whitespace();
        let start = self.position;
        while let Some(Character::Text(character)) = self.input.get(self.position) {
            if [':', ',', '{', '}', '[', ']'].contains(character) {
                break;
            }
            self.position += 1;
        }
        if self.position == start {
            return Err(Error::new("missing Gemma tool value"));
        }
        Ok(self.input[start..self.position]
            .iter()
            .filter_map(|character| match character {
                Character::Text(character) => Some(*character),
                Character::Quote => None,
            })
            .collect::<String>()
            .trim()
            .to_owned())
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.require('{')?;
        let mut values = Map::new();
        if self.take('}') {
            return Ok(Value::Object(values));
        }
        loop {
            self.whitespace();
            let key = if self.input.get(self.position) == Some(&Character::Quote) {
                self.quoted()?
            } else {
                self.atom()?
            };
            self.require(':')?;
            values.insert(key, self.value()?);
            if self.take('}') {
                break;
            }
            self.require(',')?;
        }
        Ok(Value::Object(values))
    }

    fn value(&mut self) -> Result<Value, Error> {
        self.whitespace();
        match self.input.get(self.position) {
            Some(Character::Quote) => Ok(Value::String(self.quoted()?)),
            Some(Character::Text('{')) => self.object(),
            Some(Character::Text('[')) => {
                self.position += 1;
                let mut values = Vec::new();
                if self.take(']') {
                    return Ok(Value::Array(values));
                }
                loop {
                    values.push(self.value()?);
                    if self.take(']') {
                        break;
                    }
                    self.require(',')?;
                }
                Ok(Value::Array(values))
            }
            _ => {
                let atom = self.atom()?;
                if atom == "None" {
                    return Ok(Value::Null);
                }
                serde_json::from_str(&atom)
                    .map_err(|_| Error::new(format!("invalid Gemma tool value {atom:?}")))
            }
        }
    }
}

pub(super) fn parse_call(input: &[Character]) -> Result<(String, Value), Error> {
    let mut reader = Reader { input, position: 0 };
    let call = reader.atom()?;
    if call != "call" {
        return Err(Error::new("Gemma tool call must start with call:"));
    }
    reader.require(':')?;
    let name = reader.atom()?;
    if name.is_empty() {
        return Err(Error::new("Gemma tool call requires a name"));
    }
    let arguments = reader.object()?;
    reader.whitespace();
    if reader.position != input.len() {
        return Err(Error::new("trailing Gemma tool call data"));
    }
    Ok((name, arguments))
}
