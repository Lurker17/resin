//! LFM2's tool grammar is a list of calls with literal keyword arguments.
//! Parsing never evaluates model-produced code or resolves a Python name.

use serde_json::{Map, Value};
use unicode_general_category::{GeneralCategory, get_general_category};

use crate::Error;

pub(super) fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::String(text) => {
            let quote = if text.contains('\'') && !text.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut result = String::from(quote);
            for character in text.chars() {
                match character {
                    '\0' => result.push_str("\\x00"),
                    '\\' => result.push_str("\\\\"),
                    '\n' => result.push_str("\\n"),
                    '\r' => result.push_str("\\r"),
                    '\t' => result.push_str("\\t"),
                    character if character == quote => {
                        result.push('\\');
                        result.push(character);
                    }
                    character => {
                        if !python_printable(character) {
                            let code = character as u32;
                            result.push_str(&if code <= 0xff {
                                format!("\\x{code:02x}")
                            } else if code <= 0xffff {
                                format!("\\u{code:04x}")
                            } else {
                                format!("\\U{code:08x}")
                            });
                        } else {
                            result.push(character);
                        }
                    }
                }
            }
            result.push(quote);
            result
        }
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    repr(&Value::String(key.clone())),
                    repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => crate::json::spaced(value),
    }
}

// Match the Python 3.13/Hugging Face oracle's Unicode 15.1 repertoire rather
// than the compiler's changing Unicode tables. Unicode 15.1 added only these
// 627 printable characters to the dependency's Unicode 15.0 category data.
fn python_printable(character: char) -> bool {
    if character == ' ' || matches!(character as u32, 0x2ffc..=0x2fff | 0x31ef | 0x2ebf0..=0x2ee5d)
    {
        return true;
    }
    !matches!(
        get_general_category(character),
        GeneralCategory::Control
            | GeneralCategory::Format
            | GeneralCategory::PrivateUse
            | GeneralCategory::Surrogate
            | GeneralCategory::Unassigned
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
            | GeneralCategory::SpaceSeparator
    )
}

pub(super) fn parse_calls(text: &str) -> Result<Vec<(String, Value)>, Error> {
    let mut reader = Reader { text };
    reader.expect('[')?;
    let mut calls = Vec::new();
    while !reader.consume(']') {
        let name = reader.before('(')?.trim().to_owned();
        if name.is_empty()
            || !name
                .chars()
                .all(|character| character.is_alphanumeric() || "_.-".contains(character))
        {
            return Err(invalid());
        }
        reader.expect('(')?;
        let mut arguments = Map::new();
        while !reader.consume(')') {
            let key = reader.before('=')?.trim().to_owned();
            if key.is_empty()
                || !key
                    .chars()
                    .all(|character| character.is_alphanumeric() || character == '_')
            {
                return Err(invalid());
            }
            reader.expect('=')?;
            let value = reader.value(0)?;
            if arguments.insert(key, value).is_some() {
                return Err(Error::new("LFM2 emitted duplicate tool arguments"));
            }
            if !reader.consume(',') {
                reader.expect(')')?;
                break;
            }
        }
        calls.push((name, Value::Object(arguments)));
        if !reader.consume(',') {
            reader.expect(']')?;
            break;
        }
    }
    if !reader.text.trim().is_empty() {
        return Err(invalid());
    }
    Ok(calls)
}

fn invalid() -> Error {
    Error::new("LFM2 emitted an invalid literal tool-call list")
}

struct Reader<'a> {
    text: &'a str,
}

impl Reader<'_> {
    fn consume(&mut self, character: char) -> bool {
        self.text = self.text.trim_start();
        if let Some(text) = self.text.strip_prefix(character) {
            self.text = text;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, character: char) -> Result<(), Error> {
        if self.consume(character) {
            Ok(())
        } else {
            Err(invalid())
        }
    }
    fn before(&mut self, character: char) -> Result<&str, Error> {
        let position = self.text.find(character).ok_or_else(invalid)?;
        let (prefix, rest) = self.text.split_at(position);
        self.text = rest;
        Ok(prefix)
    }
    fn value(&mut self, depth: usize) -> Result<Value, Error> {
        if depth > 64 {
            return Err(Error::new("LFM2 tool arguments are nested too deeply"));
        }
        self.text = self.text.trim_start();
        match self.text.chars().next().ok_or_else(invalid)? {
            '\'' | '"' => self.string().map(Value::String),
            '[' | '(' => {
                let close = if self.consume('[') {
                    ']'
                } else {
                    self.expect('(')?;
                    ')'
                };
                let mut values = Vec::new();
                while !self.consume(close) {
                    values.push(self.value(depth + 1)?);
                    if !self.consume(',') {
                        self.expect(close)?;
                        break;
                    }
                }
                Ok(Value::Array(values))
            }
            '{' => {
                self.expect('{')?;
                let mut values = Map::new();
                while !self.consume('}') {
                    let key = self.string()?;
                    self.expect(':')?;
                    if values.insert(key, self.value(depth + 1)?).is_some() {
                        return Err(invalid());
                    }
                    if !self.consume(',') {
                        self.expect('}')?;
                        break;
                    }
                }
                Ok(Value::Object(values))
            }
            _ => {
                let end = self
                    .text
                    .find(|character: char| character.is_whitespace() || ",)]}".contains(character))
                    .unwrap_or(self.text.len());
                let value = &self.text[..end];
                self.text = &self.text[end..];
                match value {
                    "True" | "true" => Ok(Value::Bool(true)),
                    "False" | "false" => Ok(Value::Bool(false)),
                    "None" | "null" => Ok(Value::Null),
                    _ => serde_json::from_str(value).map_err(|_| invalid()),
                }
            }
        }
    }
    fn character(&mut self) -> Result<char, Error> {
        let character = self.text.chars().next().ok_or_else(invalid)?;
        self.text = &self.text[character.len_utf8()..];
        Ok(character)
    }
    fn string(&mut self) -> Result<String, Error> {
        self.text = self.text.trim_start();
        let quote = self.character()?;
        if quote != '\'' && quote != '"' {
            return Err(invalid());
        }
        let mut value = String::new();
        loop {
            let character = self.character()?;
            if character == quote {
                return Ok(value);
            }
            if character != '\\' {
                value.push(character);
                continue;
            }
            let escaped = self.character()?;
            value.push(match escaped {
                '\\' | '\'' | '"' => escaped,
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                'b' => '\u{8}',
                'f' => '\u{c}',
                'a' => '\u{7}',
                'v' => '\u{b}',
                'x' | 'u' | 'U' => {
                    let length = match escaped {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let digits = self.text.get(..length).ok_or_else(invalid)?;
                    let mut code = u32::from_str_radix(digits, 16).map_err(|_| invalid())?;
                    self.text = &self.text[length..];
                    if escaped == 'u' && (0xd800..=0xdbff).contains(&code) {
                        let rest = self.text.strip_prefix("\\u").ok_or_else(invalid)?;
                        let low = u32::from_str_radix(rest.get(..4).ok_or_else(invalid)?, 16)
                            .map_err(|_| invalid())?;
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return Err(invalid());
                        }
                        self.text = &rest[4..];
                        code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                    }
                    char::from_u32(code).ok_or_else(invalid)?
                }
                _ => return Err(invalid()),
            });
        }
    }
}
