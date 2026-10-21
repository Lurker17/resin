use crate::parser::{Grammar, Parser, emit_tool, tokens};
use crate::{Decoder, Error, Event, FinishReason};
use serde_json::{Map, Value};

tokens! { QwenTokens {
    im_end => ["<|im_end|>"], endoftext => ["<|endoftext|>"],
    think => ["<think>"], think_end => ["</think>"],
    tool_call => ["<tool_call>"], tool_call_end => ["</tool_call>"],
}}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Text,
    Reasoning,
    Tool,
    BetweenTools,
}

/// Prefix-stable streaming equivalent of Python's `strip()`: hold trailing
/// whitespace until a later non-whitespace character proves it is internal.
#[derive(Default)]
struct Trim {
    started: bool,
    pending: String,
}
impl Trim {
    fn push(&mut self, text: &str) -> String {
        let text = if self.started {
            text
        } else {
            text.trim_start_matches(super::whitespace)
        };
        let stable = text.trim_end_matches(super::whitespace);
        if stable.is_empty() {
            self.pending.push_str(text);
            return String::new();
        }
        let mut result = std::mem::take(&mut self.pending);
        result.push_str(stable);
        self.pending.push_str(&text[stable.len()..]);
        self.started = true;
        result
    }
}

/// Qwen's token-delimited reasoning and schema-aware XML tool calls.
pub type QwenParser = Parser<QwenGrammar>;
pub struct QwenGrammar {
    tokens: QwenTokens,
    state: State,
    trim: Trim,
    tool_text: String,
    tools: Vec<Value>,
    tool_index: usize,
    ended: bool,
}

impl QwenParser {
    pub fn new(
        tokens: QwenTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
        tools: &Value,
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let reasoning = prompt_ids
            .iter()
            .rev()
            .find_map(|&id| {
                if id == tokens.think {
                    Some(true)
                } else if id == tokens.think_end {
                    Some(false)
                } else {
                    None
                }
            })
            .unwrap_or(false);
        Ok(Self::with_grammar(
            QwenGrammar {
                tokens,
                state: if reasoning {
                    State::Reasoning
                } else {
                    State::Text
                },
                trim: Trim::default(),
                tool_text: String::new(),
                tools: tools.as_array().cloned().unwrap_or_default(),
                tool_index: 0,
                ended: false,
            },
            decoder,
        ))
    }
}

impl QwenGrammar {
    fn emit_tool(&mut self, events: &mut Vec<Event>) {
        let Some((_, function)) = self.tool_text.split_once("<function=") else {
            return;
        };
        let Some((name, _)) = function.split_once('>') else {
            return;
        };
        if name.is_empty() {
            return;
        }
        let properties = self
            .tools
            .iter()
            .map(|tool| tool.get("function").unwrap_or(tool))
            .find(|tool| tool.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|tool| tool.get("parameters"))
            .and_then(|parameters| parameters.get("properties"));
        let mut arguments = Map::new();
        let mut tail = function;
        while let Some((_, parameter)) = tail.split_once("<parameter=") {
            let Some((key, value)) = parameter.split_once('>') else {
                return;
            };
            let Some((value, rest)) = value.split_once("</parameter>") else {
                return;
            };
            let value = value.trim_matches(super::whitespace);
            let schema = properties.and_then(|properties| properties.get(key));
            let declared = schema.and_then(|schema| schema.get("type"));
            let string = declared.and_then(Value::as_str) == Some("string")
                || declared
                    .and_then(Value::as_array)
                    .is_some_and(|types| types.len() == 1 && types[0] == "string");
            let value = if string {
                Value::String(value.to_owned())
            } else {
                serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
            };
            arguments.insert(key.to_owned(), value);
            tail = rest;
        }
        emit_tool(
            &mut self.tool_index,
            name,
            Value::Object(arguments).to_string(),
            events,
        );
    }
}

impl Grammar for QwenGrammar {
    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.ended
    }
    fn text(&mut self, text: String, events: &mut Vec<Event>) -> Result<(), Error> {
        if self.state == State::Tool {
            self.tool_text.push_str(&text);
            return Ok(());
        }
        if self.state == State::BetweenTools {
            return Ok(());
        }
        let delta = self.trim.push(&text);
        if !delta.is_empty() {
            events.push(if self.state == State::Reasoning {
                Event::ReasoningDelta { delta }
            } else {
                Event::TextDelta { delta }
            });
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut Vec<Event>) -> Result<(), Error> {
        if id == self.tokens.im_end || id == self.tokens.endoftext {
            self.ended = true;
        } else if id == self.tokens.tool_call {
            self.state = State::Tool;
            self.tool_text.clear();
        } else if id == self.tokens.tool_call_end {
            if self.state == State::Tool {
                self.emit_tool(events);
            }
            self.tool_text.clear();
            self.state = State::BetweenTools;
        } else if id == self.tokens.think {
            self.state = State::Reasoning;
        } else if id == self.tokens.think_end {
            self.state = State::Text;
        }
        self.trim = Trim::default();
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut Vec<Event>) -> Result<(), Error> {
        self.tool_text.clear();
        Ok(())
    }
}
