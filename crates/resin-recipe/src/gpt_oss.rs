//! GPT-OSS's Harmony conversation format.

mod schema;

use serde_json::Value;

use crate::{Decoder, Error, Event, FinishReason, Prompt, ResponseParser};

fn option<'a>(request: &'a Value, name: &str) -> Option<&'a Value> {
    request
        .get(name)
        .filter(|value| !value.is_null())
        .or_else(|| request.get("extra").and_then(|extra| extra.get(name)))
}

fn content(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                Some("text") => part.get("text").and_then(Value::as_str),
                Some("thinking") => part.get("thinking").and_then(Value::as_str),
                _ => None,
            })
            .collect(),
        _ => String::new(),
    }
}

fn message(
    prompt: &mut Prompt,
    author: &str,
    channel: Option<&str>,
    recipient: Option<&str>,
    body: &str,
) {
    prompt.special("<|start|>");
    prompt.text(author);
    if let Some(recipient) = recipient {
        prompt.text(format!(" to={recipient}"));
    }
    if let Some(channel) = channel {
        prompt.special("<|channel|>");
        prompt.text(channel);
    }
    prompt.special("<|message|>");
    prompt.text(body);
    prompt.special(if author == "assistant" && recipient.is_some() {
        "<|call|>"
    } else {
        "<|end|>"
    });
}

fn tools_section(tools: &[Value]) -> String {
    let mut lines = vec!["# Tools\n\n## functions\n\nnamespace functions {\n".to_owned()];
    for tool in tools {
        let tool = tool.get("function").unwrap_or(tool);
        for line in tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .lines()
        {
            lines.push(format!("// {line}"));
        }
        let empty = serde_json::json!({});
        let parameters = tool
            .get("parameters")
            .filter(|value| !value.is_null())
            .unwrap_or(&empty);
        lines.push(format!(
            "type {} = (_: {}) => any;\n",
            tool.get("name").and_then(Value::as_str).unwrap_or(""),
            schema::schema_type(parameters, "")
        ));
    }
    lines.push("} // namespace functions".into());
    lines.join("\n")
}

/// Render the stock GPT-OSS Harmony scaffold without a Python renderer.
///
/// Ordinary text and explicit control markers remain distinct until encoding.
/// Options are accepted at the request root or in `extra`; a codec supplies its
/// construction-time date through `conversation_start_date`.
pub fn render(request: &Value) -> Result<Prompt, Error> {
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .ok_or_else(|| Error::new("GPT-OSS requires nonempty messages"))?;
    let tools = request
        .get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let effort = option(request, "reasoning_effort")
        .and_then(Value::as_str)
        .unwrap_or("medium")
        .to_ascii_lowercase();
    if !["low", "medium", "high"].contains(&effort.as_str()) {
        return Err(Error::new(
            "GPT-OSS reasoning_effort must be low, medium, or high",
        ));
    }
    let mut prompt = Prompt::new();
    if option(request, "use_system_prompt")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        let identity = option(request, "model_identity")
            .and_then(Value::as_str)
            .unwrap_or("You are ChatGPT, a large language model trained by OpenAI.");
        let cutoff = option(request, "knowledge_cutoff")
            .and_then(Value::as_str)
            .unwrap_or("2024-06");
        let mut system = format!("{identity}\nKnowledge cutoff: {cutoff}");
        if let Some(date) = option(request, "conversation_start_date").and_then(Value::as_str) {
            system.push_str(&format!("\nCurrent date: {date}"));
        }
        system.push_str(&format!("\n\nReasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message."));
        if !tools.is_empty() {
            system
                .push_str("\nCalls to these tools must go to the commentary channel: 'functions'.");
        }
        message(&mut prompt, "system", None, None, &system);
    }
    let first_system = messages
        .iter()
        .position(|message| message.get("role").and_then(Value::as_str) == Some("system"));
    if first_system.is_some() || !tools.is_empty() {
        let mut sections = Vec::new();
        if let Some(index) = first_system {
            let instructions = content(messages[index].get("content"));
            if !instructions.is_empty() {
                sections.push(format!("# Instructions\n\n{instructions}"));
            }
        }
        if !tools.is_empty() {
            sections.push(tools_section(tools));
        }
        message(&mut prompt, "developer", None, None, &sections.join("\n\n"));
    }
    let auto_drop_analysis = option(request, "auto_drop_analysis")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let add_generation_prompt = option(request, "add_generation_prompt")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    for (index, item) in messages.iter().enumerate() {
        if first_system == Some(index) {
            continue;
        }
        let body = content(item.get("content"));
        match item.get("role").and_then(Value::as_str).unwrap_or("") {
            "user" => message(&mut prompt, "user", None, None, &body),
            "tool" => {
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let author = if name.starts_with("functions.") {
                    name.to_owned()
                } else {
                    format!("functions.{name}")
                };
                message(
                    &mut prompt,
                    &author,
                    Some("commentary"),
                    Some("assistant"),
                    &body,
                );
            }
            "assistant" => {
                let calls = item
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let keep_analysis = !auto_drop_analysis
                    || (!calls.is_empty()
                        && !messages[index + 1..].iter().any(|later| {
                            later.get("role").and_then(Value::as_str) == Some("assistant")
                                && later
                                    .get("tool_calls")
                                    .and_then(Value::as_array)
                                    .is_none_or(Vec::is_empty)
                                && !content(later.get("content")).is_empty()
                        }));
                let reasoning = item
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .filter(|_| keep_analysis);
                if let Some(reasoning) = reasoning {
                    message(&mut prompt, "assistant", Some("analysis"), None, reasoning);
                }
                // Assistant thinking parts live in reasoning_content, not
                // its public text list, matching Harmony history rendering.
                let text = match item.get("content") {
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect(),
                    _ => body,
                };
                if !text.is_empty() || (reasoning.is_none() && calls.is_empty()) {
                    message(&mut prompt, "assistant", Some("final"), None, &text);
                }
                for call in calls {
                    let function = call.get("function").unwrap_or(call);
                    let name = function.get("name").and_then(Value::as_str).unwrap_or("");
                    let recipient = if name.starts_with("functions.") {
                        name.to_owned()
                    } else {
                        format!("functions.{name}")
                    };
                    let arguments = function
                        .get("arguments")
                        .map(|arguments| {
                            arguments
                                .as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| crate::json::spaced(arguments))
                        })
                        .unwrap_or_else(|| "{}".into());
                    message(
                        &mut prompt,
                        "assistant",
                        Some("commentary"),
                        Some(&recipient),
                        &arguments,
                    );
                }
            }
            _ => message(
                &mut prompt,
                "developer",
                None,
                None,
                &format!("# Instructions\n\n{body}"),
            ),
        }
    }
    if add_generation_prompt {
        prompt.special("<|start|>");
        prompt.text("assistant");
        prompt.special("<|channel|>");
        prompt.text("analysis");
        prompt.special("<|message|>");
    } else if messages
        .last()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("assistant")
        && messages
            .last()
            .and_then(|message| message.get("tool_calls"))
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
        && let Some(crate::PromptSpan::Special(last)) = prompt.spans.last_mut()
        && last == "<|end|>"
    {
        *last = "<|return|>".into();
    }
    Ok(prompt)
}

/// The Harmony vocabulary used by GPT-OSS (canonical names and HF aliases).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GptOssTokens {
    pub start: u32,
    pub end: u32,
    pub turn_end: u32,
    pub tool_end: u32,
    pub channel: u32,
    pub message: u32,
    pub constrain: u32,
}

impl GptOssTokens {
    pub fn validate(&self) -> Result<(), Error> {
        let ids = self.ids();
        for (index, id) in ids.iter().enumerate() {
            if ids[..index].contains(id) {
                return Err(Error::new("Harmony markers must have distinct token IDs"));
            }
        }
        Ok(())
    }

    fn ids(self) -> [u32; 7] {
        [
            self.start,
            self.end,
            self.turn_end,
            self.tool_end,
            self.channel,
            self.message,
            self.constrain,
        ]
    }
}

#[derive(Debug)]
enum State {
    Header,
    Text,
    Reasoning,
    Tool(String),
    Ignore,
}

/// Incrementally decode Harmony headers and channel bodies.
///
/// A tool call commits when its message terminator arrives. Incomplete JSON
/// at a generation limit is discarded; text and reasoning always flush.
pub struct GptOssParser {
    tokens: GptOssTokens,
    decoder: Box<dyn Decoder>,
    state: State,
    header: String,
    channel_offset: Option<usize>,
    tool_text: String,
    tool_index: usize,
    ended: bool,
    finished: bool,
}

impl GptOssParser {
    pub fn new(
        tokens: GptOssTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let mut parser = Self {
            tokens,
            decoder,
            state: State::Text,
            header: String::new(),
            channel_offset: None,
            tool_text: String::new(),
            tool_index: 0,
            ended: false,
            finished: false,
        };
        // The generation prompt opens a message, usually on analysis. Parse
        // only its header, never preceding conversation text or prompt body.
        if let Some(start) = prompt_ids.iter().rposition(|&id| id == tokens.start) {
            parser.state = State::Header;
            let tail = &prompt_ids[start + 1..];
            let end = tail.iter().position(|&id| id == tokens.message);
            parser.feed(&tail[..end.map_or(tail.len(), |end| end + 1)])?;
        }
        Ok(parser)
    }

    fn text(&mut self, text: String, events: &mut Vec<Event>) {
        if text.is_empty() {
            return;
        }
        match self.state {
            State::Header => self.header.push_str(&text),
            State::Text => events.push(Event::TextDelta { delta: text }),
            State::Reasoning => events.push(Event::ReasoningDelta { delta: text }),
            State::Tool(_) => self.tool_text.push_str(&text),
            State::Ignore => {}
        }
    }

    fn flush(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        let text = self.decoder.finish()?;
        self.text(text, events);
        self.decoder.reset();
        Ok(())
    }

    fn open_body(&mut self) {
        let role = self.header.split_whitespace().next().unwrap_or("");
        let channel = self
            .channel_offset
            .and_then(|offset| self.header[offset..].split_whitespace().next());
        let recipient = self
            .header
            .split_whitespace()
            .find_map(|word| word.strip_prefix("to="));
        self.state = if role != "assistant" {
            State::Ignore
        } else if let Some(name) = recipient.filter(|name| *name != "all") {
            State::Tool(name.strip_prefix("functions.").unwrap_or(name).to_owned())
        } else if channel == Some("analysis") {
            State::Reasoning
        } else {
            State::Text
        };
    }

    fn close_body(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        if let State::Tool(name) = &self.state {
            // Match the reference renderer's closed, valid JSON contract.
            // Invalid generated calls remain uncommitted, just like a
            // truncated call whose terminator never arrived.
            if !name.is_empty()
                && serde_json::from_str::<serde_json::Value>(&self.tool_text).is_ok()
            {
                let index = self.tool_index;
                events.push(Event::ToolCallStart {
                    index,
                    id: format!("call_{index}"),
                    name: name.clone(),
                });
                events.push(Event::ToolCallArgumentsDelta {
                    index,
                    // Harmony already emits JSON. Preserve its numeric
                    // lexemes and whitespace instead of reparsing then
                    // serializing the entire payload just to send it on.
                    delta: self.tool_text.clone(),
                });
                self.tool_index += 1;
            }
        }
        self.tool_text.clear();
        self.state = State::Ignore;
        Ok(())
    }
}

impl ResponseParser for GptOssParser {
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed a Harmony parser after finish"));
        }
        if self.ended {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        let mut start = 0;
        for (index, &id) in ids.iter().enumerate() {
            if !self.tokens.ids().contains(&id) {
                continue;
            }
            let text = self.decoder.push(&ids[start..index])?;
            self.text(text, &mut events);
            self.flush(&mut events)?;
            if id == self.tokens.start {
                // A new header may follow a nonterminal analysis message.
                self.tool_text.clear();
                self.header.clear();
                self.channel_offset = None;
                self.state = State::Header;
            } else if id == self.tokens.channel && matches!(self.state, State::Header) {
                self.header.push(' ');
                self.channel_offset = Some(self.header.len());
            } else if id == self.tokens.constrain && matches!(self.state, State::Header) {
                self.header.push(' ');
            } else if id == self.tokens.message && matches!(self.state, State::Header) {
                self.open_body();
            } else if [self.tokens.end, self.tokens.turn_end, self.tokens.tool_end].contains(&id) {
                self.close_body(&mut events)?;
                if id == self.tokens.turn_end {
                    self.ended = true;
                    return Ok(events);
                }
            }
            start = index + 1;
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
        self.flush(&mut events)?;
        self.tool_text.clear();
        self.finished = true;
        Ok(events)
    }
}
