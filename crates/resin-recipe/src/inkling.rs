use serde_json::Value;

use crate::{Decoder, Error, Event, FinishReason, ResponseParser};

/// Token IDs that delimit Inkling's typed response blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InklingTokens {
    pub message_model: u32,
    pub content_text: u32,
    pub content_thinking: u32,
    pub content_invoke_tool_json: u32,
    pub content_invoke_tool_text: u32,
    pub content_tool_error: u32,
    pub content_model_end_sampling: u32,
    pub end_message: u32,
}

impl InklingTokens {
    /// Reject ambiguous token assignments before parsing a response.
    pub fn validate(&self) -> Result<(), Error> {
        let ids = self.ids();
        for (index, id) in ids.iter().enumerate() {
            if ids[..index].contains(id) {
                return Err(Error::new(
                    "Inkling response markers must have distinct token IDs",
                ));
            }
        }
        Ok(())
    }

    fn ids(&self) -> [u32; 8] {
        [
            self.message_model,
            self.content_text,
            self.content_thinking,
            self.content_invoke_tool_json,
            self.content_invoke_tool_text,
            self.content_tool_error,
            self.content_model_end_sampling,
            self.end_message,
        ]
    }

    fn state(&self, id: u32) -> Option<State> {
        if id == self.message_model {
            Some(State::Header)
        } else if id == self.content_thinking {
            Some(State::Reasoning)
        } else if id == self.content_invoke_tool_json {
            Some(State::Tool)
        } else if self.ids().contains(&id) {
            Some(State::Content)
        } else {
            None
        }
    }

    fn initial_state(&self, prompt_ids: &[u32]) -> State {
        for &id in prompt_ids.iter().rev() {
            if id == self.content_thinking {
                return State::Reasoning;
            }
            if id == self.content_text {
                return State::Content;
            }
            if id == self.message_model || self.ids().contains(&id) {
                break;
            }
        }
        State::Header
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Header,
    Content,
    Reasoning,
    Tool,
}

/// Incrementally parse Inkling's reasoning, text, and JSON tool-call blocks.
///
/// Only explicit control-token IDs change state. Ordinary decoded text cannot
/// introduce a control marker, even if it spells the same string.
pub struct InklingParser {
    tokens: InklingTokens,
    decoder: Box<dyn Decoder>,
    state: State,
    tool_text: String,
    tool_has_tokens: bool,
    tool_index: usize,
    finished: bool,
}

impl InklingParser {
    pub fn new(
        tokens: InklingTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        Ok(Self {
            tokens,
            decoder,
            state: tokens.initial_state(prompt_ids),
            tool_text: String::new(),
            tool_has_tokens: false,
            tool_index: 0,
            finished: false,
        })
    }

    fn push_text(&mut self, text: String, events: &mut Vec<Event>) {
        if text.is_empty() {
            return;
        }
        match self.state {
            State::Header => {}
            State::Content => events.push(Event::TextDelta { delta: text }),
            State::Reasoning => events.push(Event::ReasoningDelta { delta: text }),
            State::Tool => self.tool_text.push_str(&text),
        }
    }

    fn push_tokens(&mut self, ids: &[u32], events: &mut Vec<Event>) -> Result<(), Error> {
        if ids.is_empty() || self.state == State::Header {
            return Ok(());
        }
        if self.state == State::Tool {
            self.tool_has_tokens = true;
        }
        let text = self.decoder.push(ids)?;
        self.push_text(text, events);
        Ok(())
    }

    fn close_block(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        let text = self.decoder.finish()?;
        self.push_text(text, events);
        if self.state == State::Tool && self.tool_has_tokens {
            self.emit_tool(events)?;
        }
        self.decoder.reset();
        self.tool_text.clear();
        self.tool_has_tokens = false;
        Ok(())
    }

    fn emit_tool(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        let payload: Value = serde_json::from_str(&self.tool_text)
            .map_err(|_| Error::new("Inkling emitted an invalid JSON tool call"))?;
        let name = payload
            .as_object()
            .and_then(|payload| payload.get("name"))
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new("Inkling tool call must contain a string name"))?;
        let empty = Value::Object(Default::default());
        let arguments = payload
            .get("args")
            .or_else(|| payload.get("arguments"))
            .unwrap_or(&empty);
        if !arguments.is_object() {
            return Err(Error::new("Inkling tool call arguments must be an object"));
        }
        let index = self.tool_index;
        events.push(Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: name.to_owned(),
        });
        events.push(Event::ToolCallArgumentsDelta {
            index,
            delta: arguments.to_string(),
        });
        self.tool_index += 1;
        Ok(())
    }
}

impl ResponseParser for InklingParser {
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed an Inkling parser after finish"));
        }
        let mut events = Vec::new();
        let mut start = 0;
        for (index, &id) in ids.iter().enumerate() {
            if let Some(state) = self.tokens.state(id) {
                self.push_tokens(&ids[start..index], &mut events)?;
                self.close_block(&mut events)?;
                self.state = state;
                start = index + 1;
            }
        }
        self.push_tokens(&ids[start..], &mut events)?;
        Ok(events)
    }

    fn finish(&mut self, _reason: FinishReason) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        let text = self.decoder.finish()?;
        // An unterminated tool block may be truncated JSON. Never invent a tool
        // call from it; text and reasoning still flush their decoded suffixes.
        self.push_text(text, &mut events);
        self.tool_text.clear();
        self.tool_has_tokens = false;
        self.finished = true;
        Ok(events)
    }
}
