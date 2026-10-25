use crate::parser::{EventSink, Grammar, Parser, emit_tool, tokens};
use crate::{AttributedText, Decoder, Error, FinishReason};
use serde_json::{Map, Value};

tokens! { GlmTokens {
    endoftext => ["<|endoftext|>"], user => ["<|user|>"], observation => ["<|observation|>"],
    think => ["<think>"], think_end => ["</think>"],
    tool_call => ["<tool_call>"], tool_call_end => ["</tool_call>"],
    arg_key => ["<arg_key>"], arg_key_end => ["</arg_key>"],
    arg_value => ["<arg_value>"], arg_value_end => ["</arg_value>"],
}}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Text,
    Reasoning,
    Name,
    Key,
    AfterKey,
    Value,
    BetweenArgs,
    Invalid,
}

/// GLM reasoning and complete XML calls, driven by actual control-token IDs.
pub type GlmParser = Parser<GlmGrammar>;
pub struct GlmGrammar {
    tokens: GlmTokens,
    state: State,
    name: String,
    key: String,
    value: String,
    arguments: Map<String, Value>,
    tools: Vec<Value>,
    tool_index: usize,
    ended: bool,
}
impl GlmParser {
    pub fn new(
        tokens: GlmTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
        tools: &Value,
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let state = if prompt_ids.last() == Some(&tokens.think) {
            State::Reasoning
        } else {
            State::Text
        };
        Ok(Self::with_grammar(
            GlmGrammar {
                tokens,
                state,
                name: String::new(),
                key: String::new(),
                value: String::new(),
                arguments: Map::new(),
                tools: tools.as_array().cloned().unwrap_or_default(),
                tool_index: 0,
                ended: false,
            },
            decoder,
        ))
    }
}
impl GlmGrammar {
    fn in_tool(&self) -> bool {
        !matches!(self.state, State::Text | State::Reasoning)
    }
    fn reset_tool(&mut self) {
        self.name.clear();
        self.key.clear();
        self.value.clear();
        self.arguments.clear();
    }
    fn commit_argument(&mut self) {
        let key = self.key.trim_matches(super::whitespace);
        if key.is_empty() || self.arguments.contains_key(key) {
            self.state = State::Invalid;
            return;
        }
        let schema = self
            .tools
            .iter()
            .map(super::function)
            .find(|tool| tool["name"].as_str() == Some(self.name.trim_matches(super::whitespace)))
            .and_then(|tool| tool.get("parameters"))
            .and_then(|parameters| parameters.get("properties"))
            .and_then(|properties| properties.get(key));
        let declared = schema.and_then(|schema| schema.get("type"));
        let string = declared.and_then(Value::as_str) == Some("string")
            || declared
                .and_then(Value::as_array)
                .is_some_and(|types| types.len() == 1 && types[0] == "string");
        let value = if string {
            Value::String(self.value.clone())
        } else {
            serde_json::from_str(&self.value).unwrap_or_else(|_| Value::String(self.value.clone()))
        };
        self.arguments.insert(key.into(), value);
        self.key.clear();
        self.value.clear();
        self.state = State::BetweenArgs;
    }
}
impl Grammar for GlmGrammar {
    fn channel(&self) -> Option<&'static str> {
        match self.state {
            State::Text => Some("content"),
            State::Reasoning => Some("reasoning"),
            _ => None,
        }
    }
    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.ended
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if text.is_empty() {
            return Ok(());
        }
        match self.state {
            State::Text => events.text(text, false),
            State::Reasoning => events.text(text, true),
            State::Name => self.name.push_str(&text),
            State::Key => self.key.push_str(&text),
            State::Value => self.value.push_str(&text),
            State::AfterKey | State::BetweenArgs if !text.chars().all(super::whitespace) => {
                self.state = State::Invalid
            }
            _ => {}
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        let t = self.tokens;
        if [t.endoftext, t.user, t.observation].contains(&id) {
            self.ended = true;
        } else if id == t.tool_call {
            self.reset_tool();
            self.state = State::Name;
        } else if id == t.tool_call_end {
            if matches!(self.state, State::Name | State::BetweenArgs) {
                let name = self.name.trim_matches(super::whitespace);
                if !name.is_empty() {
                    emit_tool(
                        &mut self.tool_index,
                        name,
                        Value::Object(std::mem::take(&mut self.arguments)).to_string(),
                        events,
                    );
                }
            }
            self.reset_tool();
            self.state = State::Text;
        } else if id == t.think || id == t.think_end {
            self.state = if self.in_tool() {
                State::Invalid
            } else if id == t.think {
                State::Reasoning
            } else {
                State::Text
            };
        } else if id == t.arg_key && matches!(self.state, State::Name | State::BetweenArgs) {
            self.state = State::Key;
        } else if id == t.arg_key_end && self.state == State::Key {
            self.state = State::AfterKey;
        } else if id == t.arg_value && self.state == State::AfterKey {
            self.state = State::Value;
        } else if id == t.arg_value_end && self.state == State::Value {
            self.commit_argument();
        } else if self.in_tool() {
            self.state = State::Invalid;
        }
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut EventSink) -> Result<(), Error> {
        self.reset_tool();
        Ok(())
    }
}
