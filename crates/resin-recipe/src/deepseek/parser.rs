use std::collections::HashSet;

use crate::{Decoder, Error, Event, FinishReason, ResponseParser};

use super::DSML;

/// The actual special-token IDs, rather than their spellings in ordinary text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4Tokens {
    pub bos: u32,
    pub eos: u32,
    pub thinking_start: u32,
    pub thinking_end: u32,
    pub dsml: u32,
}

impl DeepSeekV4Tokens {
    pub fn validate(&self) -> Result<(), Error> {
        let ids = [
            self.bos,
            self.eos,
            self.thinking_start,
            self.thinking_end,
            self.dsml,
        ];
        for (index, id) in ids.iter().enumerate() {
            if ids[..index].contains(id) {
                return Err(Error::new(
                    "DeepSeek response markers must have distinct token IDs",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    Reasoning,
    Content,
    Calls,
    Name,
    Arguments,
    ParameterName,
    ParameterAttribute,
    ParameterValue { string: bool },
    Done,
}

/// A marker whose DSML segment must originate in a real control token.
struct Marker {
    text: String,
    special: Vec<bool>,
}

impl Marker {
    fn plain(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            special: vec![false; text.len()],
        }
    }

    fn dsml(before: &str, after: &str) -> Self {
        let text = format!("{before}{DSML}{after}");
        let mut special = vec![false; text.len()];
        special[before.len()..before.len() + DSML.len()].fill(true);
        Self { text, special }
    }
}

/// The pending suffix is bounded by a marker's length, except for a tool name
/// or parameter key that must be complete before it can be JSON escaped.
#[derive(Default)]
struct Buffer {
    text: String,
    special: Vec<bool>,
}

impl Buffer {
    fn push(&mut self, text: &str, special: bool) {
        self.text.push_str(text);
        self.special
            .extend(std::iter::repeat_n(special, text.len()));
    }

    fn take(&mut self, length: usize) -> String {
        self.special.drain(..length);
        self.text.drain(..length).collect()
    }

    fn clear(&mut self) {
        self.text.clear();
        self.special.clear();
    }

    fn matches(&self, at: usize, marker: &Marker, length: usize) -> bool {
        self.text.as_bytes().get(at..at + length) == Some(&marker.text.as_bytes()[..length])
            && self.special.get(at..at + length) == Some(&marker.special[..length])
    }

    /// Return stable text before a marker, consuming a complete marker if found.
    fn until(&mut self, marker: &Marker) -> (String, bool) {
        let mut offset = 0;
        while let Some(at) = self.text[offset..].find(&marker.text) {
            let at = offset + at;
            if self.matches(at, marker, marker.text.len()) {
                let text = self.take(at);
                self.take(marker.text.len());
                return (text, true);
            }
            offset = at + self.text[at..].chars().next().unwrap().len_utf8();
        }
        let held = (1..=self.text.len().min(marker.text.len() - 1))
            .rev()
            .find(|&length| {
                let start = self.text.len() - length;
                self.text.is_char_boundary(start) && self.matches(start, marker, length)
            })
            .unwrap_or(0);
        (self.take(self.text.len() - held), false)
    }

    fn skip_whitespace(&mut self) {
        let length = self
            .text
            .bytes()
            .zip(&self.special)
            .take_while(|(byte, special)| !**special && byte.is_ascii_whitespace())
            .count();
        self.take(length);
    }

    /// Consume one of the expected markers, or wait for more bytes.
    fn expect(&mut self, markers: &[Marker]) -> Result<Option<usize>, Error> {
        for (index, marker) in markers.iter().enumerate() {
            if self.text.len() >= marker.text.len() && self.matches(0, marker, marker.text.len()) {
                self.take(marker.text.len());
                return Ok(Some(index));
            }
        }
        if markers.iter().any(|marker| {
            self.text.len() < marker.text.len() && self.matches(0, marker, self.text.len())
        }) {
            return Ok(None);
        }
        Err(Error::new("DeepSeek emitted malformed DSML tool markup"))
    }
}

/// Incremental parser for the pinned DeepSeek V4 output grammar.
///
/// Token decoding and marker scanning consume only newly received data. Tool
/// argument strings are escaped as they arrive; plain marker spellings cannot
/// become control tokens. Truncation preserves already streamed partial calls.
pub struct DeepSeekV4Parser {
    tokens: DeepSeekV4Tokens,
    decoder: Box<dyn Decoder>,
    stage: Stage,
    buffer: Buffer,
    name: String,
    parameter_name: String,
    parameter_names: HashSet<String>,
    tool_index: usize,
    first_parameter: bool,
    ended: bool,
    finished: bool,
}

impl DeepSeekV4Parser {
    pub fn new(
        tokens: DeepSeekV4Tokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
        thinking: bool,
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let thinking = prompt_ids
            .iter()
            .rev()
            .find_map(|&id| {
                if id == tokens.thinking_start {
                    Some(true)
                } else if id == tokens.thinking_end {
                    Some(false)
                } else {
                    None
                }
            })
            .unwrap_or(thinking);
        Ok(Self {
            tokens,
            decoder,
            stage: if thinking {
                Stage::Reasoning
            } else {
                Stage::Content
            },
            buffer: Buffer::default(),
            name: String::new(),
            parameter_name: String::new(),
            parameter_names: HashSet::new(),
            tool_index: 0,
            first_parameter: true,
            ended: false,
            finished: false,
        })
    }

    fn argument(&self, text: String, events: &mut Vec<Event>) {
        if !text.is_empty() {
            events.push(Event::ToolCallArgumentsDelta {
                index: self.tool_index,
                delta: text,
            });
        }
    }

    fn text(&mut self, text: &str, special: bool, events: &mut Vec<Event>) -> Result<(), Error> {
        self.buffer.push(text, special);
        loop {
            match self.stage {
                Stage::Reasoning => {
                    let text = self.buffer.take(self.buffer.text.len());
                    if !text.is_empty() {
                        events.push(Event::ReasoningDelta { delta: text });
                    }
                    break;
                }
                Stage::Content => {
                    let (text, found) = self.buffer.until(&Marker::dsml("\n\n<", "tool_calls>"));
                    if !text.is_empty() {
                        events.push(Event::TextDelta { delta: text });
                    }
                    if !found {
                        break;
                    }
                    self.stage = Stage::Calls;
                }
                Stage::Calls => {
                    self.buffer.skip_whitespace();
                    match self.buffer.expect(&[
                        Marker::dsml("<", "invoke name=\""),
                        Marker::dsml("</", "tool_calls>"),
                    ])? {
                        None => break,
                        Some(0) => {
                            self.stage = Stage::Name;
                        }
                        Some(_) => {
                            self.stage = Stage::Done;
                        }
                    }
                }
                Stage::Name => {
                    let (text, found) = self.buffer.until(&Marker::plain("\">"));
                    self.name.push_str(&text);
                    if !found {
                        break;
                    }
                    events.push(Event::ToolCallStart {
                        index: self.tool_index,
                        id: format!("call_{}", self.tool_index),
                        name: std::mem::take(&mut self.name),
                    });
                    self.argument("{".into(), events);
                    self.parameter_names.clear();
                    self.first_parameter = true;
                    self.stage = Stage::Arguments;
                }
                Stage::Arguments => {
                    self.buffer.skip_whitespace();
                    match self.buffer.expect(&[
                        Marker::dsml("<", "parameter name=\""),
                        Marker::dsml("</", "invoke>"),
                    ])? {
                        None => break,
                        Some(0) => {
                            self.stage = Stage::ParameterName;
                        }
                        Some(_) => {
                            self.argument("}".into(), events);
                            self.tool_index += 1;
                            self.stage = Stage::Calls;
                        }
                    }
                }
                Stage::ParameterName => {
                    let (text, found) = self.buffer.until(&Marker::plain("\" string=\""));
                    self.parameter_name.push_str(&text);
                    if !found {
                        break;
                    }
                    if !self.parameter_names.insert(self.parameter_name.clone()) {
                        return Err(Error::new("DeepSeek emitted a duplicate tool parameter"));
                    }
                    let separator = if self.first_parameter { "" } else { ", " };
                    self.argument(
                        format!(
                            "{separator}{}: ",
                            serde_json::to_string(&self.parameter_name).unwrap()
                        ),
                        events,
                    );
                    self.parameter_name.clear();
                    self.first_parameter = false;
                    self.stage = Stage::ParameterAttribute;
                }
                Stage::ParameterAttribute => {
                    match self
                        .buffer
                        .expect(&[Marker::plain("true\">"), Marker::plain("false\">")])?
                    {
                        None => break,
                        Some(index) => {
                            let string = index == 0;
                            if string {
                                self.argument("\"".into(), events);
                            }
                            self.stage = Stage::ParameterValue { string };
                        }
                    }
                }
                Stage::ParameterValue { string } => {
                    let (text, found) = self.buffer.until(&Marker::dsml("</", "parameter>"));
                    let text = if string {
                        let quoted = serde_json::to_string(&text).unwrap();
                        quoted[1..quoted.len() - 1].to_owned()
                    } else {
                        text
                    };
                    self.argument(text, events);
                    if !found {
                        break;
                    }
                    if string {
                        self.argument("\"".into(), events);
                    }
                    self.stage = Stage::Arguments;
                }
                Stage::Done => {
                    if !self.buffer.text.is_empty() {
                        return Err(Error::new("DeepSeek emitted content after its tool calls"));
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    fn flush_decoder(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        let text = self.decoder.finish()?;
        self.text(&text, false, events)?;
        self.decoder.reset();
        Ok(())
    }

    fn control(&mut self, id: u32, events: &mut Vec<Event>) -> Result<(), Error> {
        self.flush_decoder(events)?;
        if id == self.tokens.eos {
            match self.stage {
                Stage::Content => {
                    let delta = self.buffer.take(self.buffer.text.len());
                    if !delta.is_empty() {
                        events.push(Event::TextDelta { delta });
                    }
                }
                Stage::Done => {}
                _ => {
                    return Err(Error::new(
                        "DeepSeek ended an incomplete reasoning or tool-call block",
                    ));
                }
            }
            self.ended = true;
        } else if id == self.tokens.thinking_end {
            if self.stage != Stage::Reasoning {
                return Err(Error::new(
                    "DeepSeek emitted an unexpected thinking-end marker",
                ));
            }
            self.stage = Stage::Content;
        } else if id == self.tokens.dsml {
            self.text(DSML, true, events)?;
        } else {
            return Err(Error::new(
                "DeepSeek emitted an unexpected response control token",
            ));
        }
        Ok(())
    }
}

impl ResponseParser for DeepSeekV4Parser {
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed a DeepSeek parser after finish"));
        }
        let mut events = Vec::new();
        let mut start = 0;
        for (index, &id) in ids.iter().enumerate() {
            if self.ended {
                return Err(Error::new("DeepSeek emitted tokens after EOS"));
            }
            if [
                self.tokens.bos,
                self.tokens.eos,
                self.tokens.thinking_start,
                self.tokens.thinking_end,
                self.tokens.dsml,
            ]
            .contains(&id)
            {
                let text = self.decoder.push(&ids[start..index])?;
                self.text(&text, false, &mut events)?;
                self.control(id, &mut events)?;
                start = index + 1;
            }
        }
        if !self.ended {
            let text = self.decoder.push(&ids[start..])?;
            self.text(&text, false, &mut events)?;
        }
        Ok(events)
    }

    fn finish(&mut self, _reason: FinishReason) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        if !self.ended {
            self.flush_decoder(&mut events)?;
            let delta = self.buffer.take(self.buffer.text.len());
            match self.stage {
                Stage::Reasoning if !delta.is_empty() => {
                    events.push(Event::ReasoningDelta { delta })
                }
                Stage::Content if !delta.is_empty() => events.push(Event::TextDelta { delta }),
                // An unfinished markup suffix is not part of the answer.
                _ => {}
            }
        }
        self.buffer.clear();
        self.finished = true;
        Ok(events)
    }
}
