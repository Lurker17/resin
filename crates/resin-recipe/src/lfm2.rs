//! The text, reasoning, and Python-style tool-call format of LFM2.5.

mod literals;

use serde_json::Value;

use crate::{Decoder, Error, Event, FinishReason, Prompt, ResponseParser};

const BOS: &str = "<|startoftext|>";
const START: &str = "<|im_start|>";
const END: &str = "<|im_end|>";
const THINK: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL: &str = "<|tool_call_start|>";
const TOOL_END: &str = "<|tool_call_end|>";

/// Render the LFM2.5 checkpoint's chat template without running Python/Jinja.
pub fn render(request: &Value) -> Result<Prompt, Error> {
    let messages = request["messages"]
        .as_array()
        .ok_or_else(|| Error::new("messages must be an array"))?;
    if messages.is_empty() {
        return Err(Error::new("conversation must contain at least one message"));
    }
    let mut prompt = Prompt::new();
    prompt.special(BOS);
    let (system, messages) = if messages[0]["role"] == "system" {
        (content(&messages[0]["content"])?, &messages[1..])
    } else {
        (String::new(), messages.as_slice())
    };
    let tools = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty());
    if !system.is_empty() || tools.is_some() {
        prompt.special(START);
        prompt.text("system\n");
        prompt.text(&system);
        if let Some(tools) = tools {
            if !system.is_empty() {
                prompt.text("\n");
            }
            prompt.text("List of tools: [");
            for (index, tool) in tools.iter().enumerate() {
                if index > 0 {
                    prompt.text(", ");
                }
                prompt.text(
                    tool.as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| crate::json::spaced(tool)),
                );
            }
            prompt.text("]");
        }
        prompt.special(END);
        prompt.text("\n");
    }
    let last_user = messages
        .iter()
        .rposition(|message| message["role"] == "user");
    let preserve =
        request["extra"]["preserve_thinking"] == true || request["preserve_thinking"] == true;
    for (index, message) in messages.iter().enumerate() {
        let role = message["role"]
            .as_str()
            .ok_or_else(|| Error::new("message role must be a string"))?;
        if !["system", "developer", "user", "assistant", "tool"].contains(&role) {
            return Err(Error::new(format!("unsupported LFM2 message role: {role}")));
        }
        prompt.special(START);
        prompt.text(role);
        prompt.text("\n");
        if role == "assistant" {
            let keeps_thinking = preserve || last_user.is_none_or(|last| index > last);
            let thinking = ["thinking", "reasoning", "reasoning_content"]
                .into_iter()
                .filter_map(|key| message[key].as_str())
                .find(|text| !text.is_empty());
            if let Some(thinking) = thinking.filter(|_| keeps_thinking) {
                prompt.special(THINK);
                prompt.text(thinking);
                prompt.special(THINK_END);
            }
            let mut text = content(&message["content"])?;
            if !keeps_thinking && let Some((_, answer)) = text.rsplit_once(THINK_END) {
                text = answer.trim().to_owned();
            }
            let calls = message.get("tool_calls").filter(|value| !value.is_null());
            const CONTINUE: &str = "CONTINUE_FINAL_MESSAGE_TAG ";
            let continued = calls.is_some() && text.ends_with(CONTINUE);
            prompt.text(if continued {
                &text[..text.len() - CONTINUE.len()]
            } else {
                &text
            });
            if let Some(calls) = calls {
                render_calls(
                    &mut prompt,
                    calls
                        .as_array()
                        .ok_or_else(|| Error::new("tool_calls must be an array"))?,
                )?;
            }
            if continued {
                prompt.text(CONTINUE);
            }
        } else {
            prompt.text(content(&message["content"])?);
        }
        prompt.special(END);
        prompt.text("\n");
    }
    if request["add_generation_prompt"] != false {
        prompt.special(START);
        prompt.text("assistant\n");
    }
    Ok(prompt)
}

fn content(value: &Value) -> Result<String, Error> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => text.push_str(
                        block["text"]
                            .as_str()
                            .ok_or_else(|| Error::new("text content must be a string"))?,
                    ),
                    Some("image") => {
                        return Err(Error::new(
                            "the stock LFM2 runner supports text content only",
                        ));
                    }
                    _ => text.push_str(&crate::json::spaced(block)),
                }
            }
            Ok(text)
        }
        _ => Err(Error::new("message content must be text or content blocks")),
    }
}

fn render_calls(prompt: &mut Prompt, calls: &[Value]) -> Result<(), Error> {
    prompt.special(TOOL);
    prompt.text("[");
    for (index, call) in calls.iter().enumerate() {
        if index > 0 {
            prompt.text(", ");
        }
        let function = call.get("function").unwrap_or(call);
        let name = function["name"]
            .as_str()
            .ok_or_else(|| Error::new("tool call name must be a string"))?;
        prompt.text(name);
        prompt.text("(");
        let arguments = &function["arguments"];
        let decoded;
        let arguments = if let Some(text) = arguments.as_str() {
            decoded = serde_json::from_str(text)
                .map_err(|_| Error::new("tool call arguments must be a JSON object"))?;
            &decoded
        } else {
            arguments
        };
        let arguments = arguments
            .as_object()
            .ok_or_else(|| Error::new("tool call arguments must be a JSON object"))?;
        for (index, (name, value)) in arguments.iter().enumerate() {
            if index > 0 {
                prompt.text(", ");
            }
            prompt.text(name);
            prompt.text("=");
            prompt.text(match value {
                Value::String(text) => format!("'{text}'"),
                Value::Object(_) => crate::json::spaced(value),
                _ => literals::repr(value),
            });
        }
        prompt.text(")");
    }
    prompt.text("]");
    prompt.special(TOOL_END);
    Ok(())
}

/// Control IDs used by LFM2.5; several are added tokens marked `special=false`
/// by the upstream tokenizer, but still delimit this model's response format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lfm2Tokens {
    pub message_start: u32,
    pub message_end: u32,
    pub thinking_start: u32,
    pub thinking_end: u32,
    pub tool_call_start: u32,
    pub tool_call_end: u32,
}

impl Lfm2Tokens {
    fn ids(&self) -> [u32; 6] {
        [
            self.message_start,
            self.message_end,
            self.thinking_start,
            self.thinking_end,
            self.tool_call_start,
            self.tool_call_end,
        ]
    }
    pub fn validate(&self) -> Result<(), Error> {
        let ids = self.ids();
        for (index, id) in ids.iter().enumerate() {
            if ids[..index].contains(id) {
                return Err(Error::new(
                    "LFM2 response markers must have distinct token IDs",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Stage {
    Content,
    Reasoning,
    Header,
    Tool,
    Done,
}

/// Parse LFM2.5 output incrementally, buffering only an unfinished tool block.
pub struct Lfm2Parser {
    tokens: Lfm2Tokens,
    decoder: Box<dyn Decoder>,
    stage: Stage,
    tool: String,
    tool_index: usize,
    finished: bool,
}

impl Lfm2Parser {
    pub fn new(
        tokens: Lfm2Tokens,
        decoder: Box<dyn Decoder>,
        _prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        Ok(Self {
            tokens,
            decoder,
            stage: Stage::Content,
            tool: String::new(),
            tool_index: 0,
            finished: false,
        })
    }

    fn text(&mut self, text: String, events: &mut Vec<Event>) {
        if text.is_empty() {
            return;
        }
        match self.stage {
            Stage::Content => events.push(Event::TextDelta { delta: text }),
            Stage::Reasoning => events.push(Event::ReasoningDelta { delta: text }),
            Stage::Tool => self.tool.push_str(&text),
            Stage::Header => {
                if let Some((_, content)) = text.split_once('\n') {
                    self.stage = Stage::Content;
                    self.text(content.to_owned(), events);
                }
            }
            Stage::Done => {}
        }
    }

    fn close(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        let text = self.decoder.finish()?;
        self.text(text, events);
        self.decoder.reset();
        Ok(())
    }

    fn control(&mut self, id: u32, events: &mut Vec<Event>) -> Result<(), Error> {
        self.close(events)?;
        if id == self.tokens.tool_call_end {
            if self.stage != Stage::Tool {
                return Err(Error::new(
                    "LFM2 emitted an unexpected tool-call end marker",
                ));
            }
            for (name, arguments) in literals::parse_calls(&self.tool)? {
                let index = self.tool_index;
                events.push(Event::ToolCallStart {
                    index,
                    id: format!("call_{index}"),
                    name,
                });
                events.push(Event::ToolCallArgumentsDelta {
                    index,
                    delta: arguments.to_string(),
                });
                self.tool_index += 1;
            }
            self.tool.clear();
            self.stage = Stage::Content;
        } else if id == self.tokens.tool_call_start {
            self.tool.clear();
            self.stage = Stage::Tool;
        } else if id == self.tokens.thinking_start {
            self.stage = Stage::Reasoning;
        } else if id == self.tokens.thinking_end {
            self.stage = Stage::Content;
        } else if id == self.tokens.message_start {
            self.stage = Stage::Header;
        } else {
            self.tool.clear();
            self.stage = Stage::Done;
        }
        Ok(())
    }
}

impl ResponseParser for Lfm2Parser {
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed an LFM2 parser after finish"));
        }
        let mut events = Vec::new();
        let mut start = 0;
        for (index, &id) in ids.iter().enumerate() {
            if self.tokens.ids().contains(&id) {
                let text = self.decoder.push(&ids[start..index])?;
                self.text(text, &mut events);
                self.control(id, &mut events)?;
                start = index + 1;
            }
        }
        let text = self.decoder.push(&ids[start..])?;
        self.text(text, &mut events);
        Ok(events)
    }
    fn finish(&mut self, _reason: FinishReason) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        self.close(&mut events)?;
        self.tool.clear();
        self.finished = true;
        Ok(events)
    }
}
