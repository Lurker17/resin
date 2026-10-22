use super::{BOS, DSML, EOS, THINKING_END, THINKING_START, Version};
use crate::parser::{EventSink, Grammar, Parser, tokens};
use crate::{AttributedText, Decoder, Error, Event, FinishReason};
use std::collections::HashSet;

tokens! { DeepSeekV4Tokens {
    bos => [BOS], eos => [EOS], thinking_start => [THINKING_START],
    thinking_end => [THINKING_END], dsml => [DSML],
}}

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
    text: AttributedText,
    special: Vec<bool>,
}

impl Buffer {
    fn push(&mut self, text: AttributedText, special: bool) {
        self.special
            .extend(std::iter::repeat_n(special, text.len()));
        self.text.push(text);
    }

    fn take(&mut self, length: usize) -> AttributedText {
        self.special.drain(..length);
        self.text.take(length)
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
    fn until(&mut self, marker: &Marker) -> (AttributedText, bool) {
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

/// Incremental DSML parsing. Only actual control IDs introduce DSML markers.
pub type DeepSeekV4Parser = Parser<DeepSeekV4Grammar>;
pub struct DeepSeekV4Grammar {
    version: Version,
    tokens: DeepSeekV4Tokens,
    stage: Stage,
    buffer: Buffer,
    name: String,
    parameter_name: String,
    parameter_names: HashSet<String>,
    tool_index: usize,
    first_parameter: bool,
    ended: bool,
}

impl DeepSeekV4Parser {
    pub fn new(
        tokens: DeepSeekV4Tokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
        thinking: bool,
    ) -> Result<Self, Error> {
        Self::with_version(tokens, decoder, prompt_ids, thinking, Version::V4)
    }

    pub fn with_version(
        tokens: DeepSeekV4Tokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
        thinking: bool,
        version: Version,
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
        Ok(Self::with_grammar(
            DeepSeekV4Grammar {
                version,
                tokens,
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
            },
            decoder,
        ))
    }
}

impl DeepSeekV4Grammar {
    fn argument(&self, text: String, events: &mut EventSink) {
        if !text.is_empty() {
            events.push(Event::ToolCallArgumentsDelta {
                index: self.tool_index,
                delta: text,
            });
        }
    }

    fn push(
        &mut self,
        text: AttributedText,
        special: bool,
        events: &mut EventSink,
    ) -> Result<(), Error> {
        let (calls, invoke, parameter) = self.version.tags();
        self.buffer.push(text, special);
        loop {
            match self.stage {
                Stage::Reasoning => {
                    let text = self.buffer.take(self.buffer.text.len());
                    if !text.is_empty() {
                        events.text(text, true);
                    }
                    break;
                }
                Stage::Content => {
                    let (text, found) = self
                        .buffer
                        .until(&Marker::dsml("\n\n<", &format!("{calls}>")));
                    if !text.is_empty() {
                        events.text(text, false);
                    }
                    if !found {
                        break;
                    }
                    self.stage = Stage::Calls;
                }
                Stage::Calls => {
                    self.buffer.skip_whitespace();
                    match self.buffer.expect(&[
                        Marker::dsml("<", &format!("{invoke} name=\"")),
                        Marker::dsml("</", &format!("{calls}>")),
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
                        Marker::dsml("<", &format!("{parameter} name=\"")),
                        Marker::dsml("</", &format!("{invoke}>")),
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
                    let (text, found) = self
                        .buffer
                        .until(&Marker::dsml("</", &format!("{parameter}>")));
                    let text = if string {
                        let quoted = serde_json::to_string(&text.text).unwrap();
                        quoted[1..quoted.len() - 1].to_owned()
                    } else {
                        text.text
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
}

impl Grammar for DeepSeekV4Grammar {
    const STRICT_END: bool = true;
    fn channel(&self) -> Option<&'static str> {
        (!matches!(self.stage, Stage::Content | Stage::Reasoning)).then_some("tool")
    }
    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.ended
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        self.push(text, false, events)
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        if id == self.tokens.eos {
            match self.stage {
                Stage::Content => {
                    let delta = self.buffer.take(self.buffer.text.len());
                    if !delta.is_empty() {
                        events.text(delta, false);
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
            self.push(AttributedText::plain(DSML), true, events)?;
        } else {
            return Err(Error::new(
                "DeepSeek emitted an unexpected response control token",
            ));
        }
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, events: &mut EventSink) -> Result<(), Error> {
        if !self.ended {
            let delta = self.buffer.take(self.buffer.text.len());
            match self.stage {
                Stage::Reasoning if !delta.is_empty() => events.text(delta, true),
                Stage::Content if !delta.is_empty() => events.text(delta, false),
                _ => {}
            }
        }
        self.buffer.clear();
        Ok(())
    }
}
