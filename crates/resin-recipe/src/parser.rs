//! Shared token-stream mechanics; model grammars only interpret text and controls.

use crate::{Decoder, Encoder, Error, Event, FinishReason, ResponseParser};

pub trait Grammar: Send {
    const STRICT_END: bool = false;
    fn channel(&self) -> Option<&'static str> {
        None
    }
    fn is_control(&self, id: u32) -> bool;
    fn text(&mut self, text: String, events: &mut Vec<Event>) -> Result<(), Error>;
    fn control(&mut self, id: u32, events: &mut Vec<Event>) -> Result<(), Error>;
    fn ended(&self) -> bool {
        false
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut Vec<Event>) -> Result<(), Error> {
        Ok(())
    }
}

/// Batch ordinary token runs, flushing the decoder before each actual control.
/// Ordinary text spelling a control token never enters the grammar's control path.
pub struct Parser<G> {
    pub(crate) grammar: G,
    decoder: Box<dyn Decoder>,
    finished: bool,
}

impl<G: Grammar> Parser<G> {
    pub(crate) fn with_grammar(grammar: G, decoder: Box<dyn Decoder>) -> Self {
        Self {
            grammar,
            decoder,
            finished: false,
        }
    }
    fn text(&mut self, ids: &[u32], events: &mut Vec<Event>) -> Result<(), Error> {
        if !ids.is_empty() {
            self.grammar.text(self.decoder.push(ids)?, events)?;
        }
        Ok(())
    }
    fn flush(&mut self, events: &mut Vec<Event>) -> Result<(), Error> {
        self.grammar.text(self.decoder.finish()?, events)?;
        self.decoder.reset();
        Ok(())
    }
}

impl<G: Grammar> ResponseParser for Parser<G> {
    fn token_channel(&self, id: u32) -> Option<&'static str> {
        if self.finished || self.grammar.ended() {
            None
        } else if self.grammar.is_control(id) {
            Some("control")
        } else {
            self.grammar.channel()
        }
    }

    fn feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed a response parser after finish"));
        }
        let mut events = Vec::new();
        let mut start = 0;
        for (index, &id) in ids.iter().enumerate() {
            if self.grammar.ended() {
                if G::STRICT_END {
                    return Err(Error::new("response contains tokens after EOS"));
                }
                return Ok(events);
            }
            if self.grammar.is_control(id) {
                self.text(&ids[start..index], &mut events)?;
                self.flush(&mut events)?;
                self.grammar.control(id, &mut events)?;
                start = index + 1;
            }
        }
        if !self.grammar.ended() {
            self.text(&ids[start..], &mut events)?;
        }
        Ok(events)
    }
    fn finish(&mut self, reason: FinishReason) -> Result<Vec<Event>, Error> {
        let mut events = Vec::new();
        if !self.finished {
            if !self.grammar.ended() {
                self.flush(&mut events)?;
            }
            self.grammar.finish(reason, &mut events)?;
            self.finished = true;
        }
        Ok(events)
    }
}

pub(crate) fn emit_tool(
    index: &mut usize,
    name: impl Into<String>,
    arguments: impl Into<String>,
    events: &mut Vec<Event>,
) {
    events.push(Event::ToolCallStart {
        index: *index,
        id: format!("call_{index}"),
        name: name.into(),
    });
    events.push(Event::ToolCallArgumentsDelta {
        index: *index,
        delta: arguments.into(),
    });
    *index += 1;
}

pub(crate) fn resolve(encoder: &dyn Encoder, spellings: &[&str]) -> Result<u32, Error> {
    spellings
        .iter()
        .find_map(|spelling| encoder.special_id(spelling).ok())
        .ok_or_else(|| {
            Error::new(format!(
                "missing model control token (aliases: {})",
                spellings.join(", ")
            ))
        })
}

macro_rules! tokens {
    ($name:ident { $($field:ident => [$($spelling:expr),+]),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $name { $(pub $field: u32),+ }
        impl $name {
            fn ids(&self) -> [u32; [$(stringify!($field)),+].len()] { [$(self.$field),+] }
            pub fn resolve(encoder: &dyn crate::Encoder) -> Result<Self, crate::Error> {
                Ok(Self { $($field: crate::parser::resolve(encoder, &[$($spelling),+])?),+ })
            }
            pub fn validate(&self) -> Result<(), crate::Error> {
                let ids = self.ids();
                if ids.iter().enumerate().any(|(index, id)| ids[..index].contains(id)) {
                    return Err(crate::Error::new(concat!(stringify!($name), " markers must have distinct token IDs")));
                }
                Ok(())
            }
        }
    };
}
pub(crate) use tokens;
