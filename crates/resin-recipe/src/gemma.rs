//! Gemma 4's turn, thought-channel, and structured tool-call format.

mod values;

use serde_json::Value;

use crate::{Decoder, Error, Event, FinishReason, Prompt, ResponseParser};

fn option<'a>(request: &'a Value, name: &str) -> Option<&'a Value> {
    request
        .get(name)
        .filter(|value| !value.is_null())
        .or_else(|| {
            request
                .get("extra")
                .and_then(|extra| extra.get(name))
                .filter(|value| !value.is_null())
        })
}

fn array(value: Option<&Value>) -> &[Value] {
    value
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn string(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("")
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64().unwrap_or(0.0) != 0.0,
    }
}

fn strip_thinking(text: &str) -> String {
    text.split("<channel|>")
        .map(|part| {
            part.split_once("<|channel>")
                .map_or(part, |(before, _)| before)
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn content(prompt: &mut Prompt, value: Option<&Value>, assistant: bool) -> Option<&'static str> {
    let mut last_media = None;
    let text = |prompt: &mut Prompt, value: &str| {
        prompt.text(if assistant {
            strip_thinking(value)
        } else {
            value.trim().to_owned()
        });
    };
    match value {
        Some(Value::String(value)) => text(prompt, value),
        Some(Value::Array(parts)) => {
            for part in parts {
                match string(part.get("type")) {
                    "text" | "input_text" => text(prompt, string(part.get("text"))),
                    "image" | "image_url" | "input_image" => {
                        prompt.special("<|image|>");
                        last_media = Some("image");
                    }
                    "video" => {
                        prompt.special("<|video|>");
                        last_media = Some("video");
                    }
                    "audio" => {
                        prompt.special("<|audio|>");
                        last_media = Some("audio");
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    last_media
}

fn tool_response(prompt: &mut Prompt, name: &str, response: &Value) {
    prompt
        .special("<|tool_response>")
        .text(format!("response:{name}{{"));
    if let Some(response) = response.as_object() {
        for (index, (key, value)) in values::sorted(response).into_iter().enumerate() {
            if index > 0 {
                prompt.text(",");
            }
            prompt.text(key).text(":");
            values::argument(prompt, value, false);
        }
    } else {
        prompt.text("value:");
        values::argument(prompt, response, false);
    }
    prompt.text("}").special("<tool_response|>");
}

/// Render the pinned Gemma 4 template, preserving explicit control-token spans.
pub fn render(request: &Value) -> Result<Prompt, Error> {
    let messages = array(request.get("messages"));
    if messages.is_empty() {
        return Err(Error::new("Gemma 4 requires nonempty messages"));
    }
    let tools = array(request.get("tools"));
    let enable_thinking = option(request, "enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut prompt = Prompt::new();
    prompt.special("<bos>");
    let system = matches!(string(messages[0].get("role")), "system" | "developer");
    let mut previous_kind = None;
    if enable_thinking || !tools.is_empty() || system {
        prompt.special("<|turn>").text("system\n");
        if enable_thinking {
            prompt.special("<|think|>").text("\n");
            previous_kind = Some("think");
        }
        if system {
            match messages[0].get("content") {
                Some(Value::String(text)) => {
                    prompt.text(text.trim());
                }
                Some(Value::Array(parts)) => {
                    for part in parts {
                        prompt.text(string(part.get("text")).trim()).text(" ");
                    }
                }
                _ => {}
            }
        }
        for tool in tools {
            prompt.special("<|tool>");
            values::declaration(&mut prompt, tool.get("function").unwrap_or(tool));
            prompt.special("<tool|>");
            previous_kind = Some("tool");
        }
        prompt.special("<turn|>").text("\n");
    }
    let messages = if system { &messages[1..] } else { messages };
    let last_user = messages
        .iter()
        .rposition(|message| string(message.get("role")) == "user");
    let mut previous_role = "";
    for (index, message) in messages.iter().enumerate() {
        let role = string(message.get("role"));
        if role == "tool" {
            continue;
        }
        if !matches!(
            role,
            "user" | "assistant" | "system" | "developer" | "model"
        ) {
            return Err(Error::new(format!("unsupported Gemma 4 role {role:?}")));
        }
        previous_kind = None;
        let assistant = role == "assistant" || role == "model";
        if !(assistant && previous_role == "assistant") {
            prompt
                .special("<|turn>")
                .text(if assistant { "model" } else { role })
                .text("\n");
        }
        previous_role = role;
        let calls = array(message.get("tool_calls"));
        let reasoning = message
            .get("reasoning")
            .filter(|value| truthy(value))
            .or_else(|| message.get("reasoning_content"));
        if !calls.is_empty()
            && last_user.is_none_or(|last| index > last)
            && let Some(reasoning) = reasoning
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
        {
            prompt
                .special("<|channel>")
                .text("thought\n")
                .text(reasoning)
                .text("\n")
                .special("<channel|>");
        }
        for call in calls {
            let function = call.get("function").unwrap_or(call);
            prompt
                .special("<|tool_call>")
                .text(format!("call:{}{{", string(function.get("name"))));
            match function.get("arguments") {
                Some(Value::Object(arguments)) => {
                    for (index, (name, value)) in values::sorted(arguments).into_iter().enumerate()
                    {
                        if index > 0 {
                            prompt.text(",");
                        }
                        prompt.text(name).text(":");
                        values::argument(&mut prompt, value, false);
                    }
                }
                Some(Value::String(arguments)) => {
                    prompt.text(arguments);
                }
                _ => {}
            }
            prompt.text("}").special("<tool_call|>");
            previous_kind = Some("tool_call");
        }
        let mut has_response = false;
        let responses = array(message.get("tool_responses"));
        if !responses.is_empty() {
            for response in responses {
                tool_response(
                    &mut prompt,
                    response
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown"),
                    response.get("response").unwrap_or(&Value::Null),
                );
                has_response = true;
                previous_kind = Some("tool_response");
            }
        } else if !calls.is_empty() {
            for follow in messages[index + 1..]
                .iter()
                .take_while(|follow| string(follow.get("role")) == "tool")
            {
                let name = calls
                    .iter()
                    .rfind(|call| call.get("id") == follow.get("tool_call_id"))
                    .map(|call| string(call.get("function").unwrap_or(call).get("name")))
                    .unwrap_or_else(|| {
                        follow
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown")
                    });
                let body = follow.get("content").unwrap_or(&Value::Null);
                if let Some(parts) = body.as_array() {
                    let text = parts
                        .iter()
                        .filter(|part| string(part.get("type")) == "text")
                        .map(|part| string(part.get("text")))
                        .collect::<String>();
                    tool_response(&mut prompt, name, &Value::String(text));
                } else {
                    tool_response(&mut prompt, name, body);
                }
                has_response = true;
                previous_kind = Some("tool_response");
            }
        }
        let mut body = Prompt::new();
        if let Some(kind) = content(&mut body, message.get("content"), assistant) {
            previous_kind = Some(kind);
        }
        let has_content = !body.rendered_text().trim().is_empty();
        for span in body.spans {
            match span {
                crate::PromptSpan::Text(text) => {
                    prompt.text(text);
                }
                crate::PromptSpan::Special(token) => {
                    prompt.special(token);
                }
            }
        }
        if previous_kind == Some("tool_call") && !has_response {
            prompt.special("<|tool_response>");
        } else if !(has_response && !has_content) {
            prompt.special("<turn|>").text("\n");
        }
    }
    if option(request, "add_generation_prompt")
        .and_then(Value::as_bool)
        .unwrap_or(true)
        && !matches!(previous_kind, Some("tool_response" | "tool_call"))
    {
        prompt.special("<|turn>").text("model\n");
        if !enable_thinking {
            prompt
                .special("<|channel>")
                .text("thought\n")
                .special("<channel|>");
        }
    }
    Ok(prompt)
}

/// Explicit Gemma 4 grammar markers. Ordinary marker spellings remain text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4Tokens {
    pub turn_start: u32,
    pub turn_end: u32,
    pub channel_start: u32,
    pub channel_end: u32,
    pub tool_call_start: u32,
    pub tool_call_end: u32,
    pub tool_response_start: u32,
    pub tool_response_end: u32,
    pub escape: u32,
    pub eos: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Content,
    ChannelHeader,
    Reasoning,
    TurnHeader,
    Tool,
    ToolResponse,
    Ended,
}

/// Incrementally parse thought channels and Gemma's structured tool arguments.
pub struct Gemma4Parser {
    tokens: Gemma4Tokens,
    decoder: Box<dyn Decoder>,
    state: State,
    header: String,
    tool: Vec<values::Character>,
    tool_index: usize,
    finished: bool,
}

impl Gemma4Parser {
    pub fn new(
        tokens: Gemma4Tokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        let markers = [
            tokens.turn_start,
            tokens.turn_end,
            tokens.channel_start,
            tokens.channel_end,
            tokens.tool_call_start,
            tokens.tool_call_end,
            tokens.tool_response_start,
            tokens.tool_response_end,
            tokens.escape,
            tokens.eos,
        ];
        for (index, marker) in markers.iter().enumerate() {
            if markers[..index].contains(marker) {
                return Err(Error::new("Gemma 4 markers must be distinct"));
            }
        }
        let mut state = State::Content;
        for &id in prompt_ids {
            state = if id == tokens.channel_start {
                State::ChannelHeader
            } else if id == tokens.channel_end {
                State::Content
            } else if id == tokens.tool_call_start {
                State::Tool
            } else if id == tokens.tool_response_start {
                State::ToolResponse
            } else if id == tokens.tool_response_end
                || id == tokens.turn_start
                || id == tokens.turn_end
            {
                State::Content
            } else {
                state
            };
        }
        // The generation prompt may leave the thought header open. Its decoded
        // header is already in the prompt, so generated text starts reasoning.
        if state == State::ChannelHeader {
            state = State::Reasoning;
        }
        Ok(Self {
            tokens,
            decoder,
            state,
            header: String::new(),
            tool: Vec::new(),
            tool_index: 0,
            finished: false,
        })
    }

    fn text(&mut self, text: String, events: &mut Vec<Event>) {
        if text.is_empty() {
            return;
        }
        match self.state {
            State::Content => events.push(Event::TextDelta { delta: text }),
            State::Reasoning => events.push(Event::ReasoningDelta { delta: text }),
            State::Tool => self.tool.extend(text.chars().map(values::Character::Text)),
            State::ChannelHeader | State::TurnHeader => {
                self.header.push_str(&text);
                if let Some((_, body)) = self.header.split_once('\n') {
                    let body = body.to_owned();
                    self.header.clear();
                    self.state = if self.state == State::ChannelHeader {
                        State::Reasoning
                    } else {
                        State::Content
                    };
                    self.text(body, events);
                }
            }
            State::ToolResponse | State::Ended => {}
        }
    }

    fn close(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        let text = self.decoder.finish()?;
        self.text(text, events);
        self.decoder.reset();
        Ok(())
    }

    fn emit_tool(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        let (name, arguments) = values::parse_call(&self.tool)?;
        let index = self.tool_index;
        self.tool_index += 1;
        events.push(Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name,
        });
        events.push(Event::ToolCallArgumentsDelta {
            index,
            delta: arguments.to_string(),
        });
        self.tool.clear();
        Ok(())
    }
}

impl ResponseParser for Gemma4Parser {
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed a finished Gemma 4 parser"));
        }
        if self.state == State::Ended {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        let mut start = 0;
        for (index, &id) in ids.iter().enumerate() {
            let t = self.tokens;
            let control = [
                t.turn_start,
                t.turn_end,
                t.channel_start,
                t.channel_end,
                t.tool_call_start,
                t.tool_call_end,
                t.tool_response_start,
                t.tool_response_end,
                t.escape,
                t.eos,
            ]
            .contains(&id);
            if !control {
                continue;
            }
            if start < index {
                let text = self.decoder.push(&ids[start..index])?;
                self.text(text, &mut events);
            }
            self.close(&mut events)?;
            if id == t.escape {
                if self.state == State::Tool {
                    self.tool.push(values::Character::Quote);
                }
            } else if id == t.tool_call_end {
                if self.state == State::Tool {
                    self.emit_tool(&mut events)?;
                }
                self.state = State::Content;
            } else if id == t.tool_call_start {
                self.tool.clear();
                self.state = State::Tool;
            } else if id == t.channel_start {
                self.header.clear();
                self.state = State::ChannelHeader;
            } else if id == t.channel_end {
                self.header.clear();
                self.state = State::Content;
            } else if id == t.turn_start {
                self.header.clear();
                self.state = State::TurnHeader;
            } else if id == t.tool_response_start {
                self.state = State::ToolResponse;
            } else if id == t.tool_response_end {
                self.state = State::Content;
            } else {
                self.state = State::Ended;
            }
            start = index + 1;
            if self.state == State::Ended {
                return Ok(events);
            }
        }
        if start < ids.len() {
            let text = self.decoder.push(&ids[start..])?;
            self.text(text, &mut events);
        }
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
