use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

#[derive(Clone, Debug, Serialize)]
pub struct ConversationRequest {
    pub id: String,
    pub model: String,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub max_output_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u32>,
    pub seed: Option<i64>,
    pub stop: Vec<String>,
    pub reasoning_effort: Option<String>,
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

impl FinishReason {
    pub const fn openai(self, has_tool_calls: bool) -> &'static str {
        if has_tool_calls {
            "tool_calls"
        } else {
            match self {
                Self::Stop => "stop",
                Self::Length => "length",
                Self::Cancelled => "stop",
            }
        }
    }

    pub const fn anthropic(self, has_tool_calls: bool) -> &'static str {
        if has_tool_calls {
            "tool_use"
        } else {
            match self {
                Self::Stop => "end_turn",
                Self::Length => "max_tokens",
                Self::Cancelled => "end_turn",
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum GenerationEvent {
    Started {
        input_tokens: usize,
    },
    ReasoningDelta(String),
    TextDelta(String),
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: usize,
        delta: String,
    },
    Finished {
        reason: FinishReason,
        input_tokens: usize,
        cached_tokens: usize,
        output_tokens: usize,
        elapsed_ms: f64,
        generation_elapsed_ms: f64,
        time_to_first_token_ms: f64,
    },
    Error {
        message: String,
    },
}

#[derive(Debug)]
pub struct GenerationJob {
    pub request: ConversationRequest,
    pub events: mpsc::Sender<GenerationEvent>,
}

#[derive(Debug, Default)]
pub(crate) struct ResponseAccumulator {
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub output_tokens: usize,
    pub reasoning: String,
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub reason: Option<FinishReason>,
    pub elapsed_ms: f64,
    pub generation_elapsed_ms: f64,
    pub time_to_first_token_ms: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ResponseAccumulator {
    pub fn apply(&mut self, event: &GenerationEvent) {
        match event {
            GenerationEvent::Started { input_tokens } => self.input_tokens = *input_tokens,
            GenerationEvent::ReasoningDelta(delta) => self.reasoning.push_str(delta),
            GenerationEvent::TextDelta(delta) => self.text.push_str(delta),
            GenerationEvent::ToolCallStart { index, id, name } => {
                if self.tool_calls.len() <= *index {
                    self.tool_calls.resize_with(*index + 1, ToolCall::default);
                }
                self.tool_calls[*index].id.clone_from(id);
                self.tool_calls[*index].name.clone_from(name);
            }
            GenerationEvent::ToolCallArgumentsDelta { index, delta } => {
                if self.tool_calls.len() <= *index {
                    self.tool_calls.resize_with(*index + 1, ToolCall::default);
                }
                self.tool_calls[*index].arguments.push_str(delta);
            }
            GenerationEvent::Finished {
                reason,
                input_tokens,
                cached_tokens,
                output_tokens,
                elapsed_ms,
                generation_elapsed_ms,
                time_to_first_token_ms,
            } => {
                self.reason = Some(*reason);
                self.input_tokens = *input_tokens;
                self.cached_tokens = *cached_tokens;
                self.output_tokens = *output_tokens;
                self.elapsed_ms = *elapsed_ms;
                self.generation_elapsed_ms = *generation_elapsed_ms;
                self.time_to_first_token_ms = *time_to_first_token_ms;
            }
            GenerationEvent::Error { .. } => {}
        }
    }
}

pub(crate) fn json_error(message: impl Into<String>, error_type: &str) -> Value {
    serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
        }
    })
}
