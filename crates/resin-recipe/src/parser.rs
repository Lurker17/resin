//! Shared token-stream mechanics; model grammars only interpret text and controls.

use crate::{
    AttributedEvent, AttributedText, Decoder, Encoder, Error, Event, FinishReason, ResponseParser,
};

#[derive(Default)]
pub struct EventSink(Vec<AttributedEvent>);
impl EventSink {
    pub fn push(&mut self, event: Event) {
        self.0.push(event.into());
    }
    pub fn text(&mut self, text: AttributedText, reasoning: bool) {
        self.0.push(text.event(reasoning));
    }
}

pub trait Grammar: Send {
    const STRICT_END: bool = false;
    fn channel(&self) -> Option<&'static str> {
        None
    }
    fn is_control(&self, id: u32) -> bool;
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error>;
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error>;
    fn ended(&self) -> bool {
        false
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut EventSink) -> Result<(), Error> {
        Ok(())
    }
}

/// Batch ordinary token runs, flushing the decoder before each actual control.
/// Ordinary text spelling a control token never enters the grammar's control path.
pub struct Parser<G> {
    pub(crate) grammar: G,
    decoder: Box<dyn Decoder>,
    finished: bool,
    pub(crate) next_token_index: usize,
}

impl<G: Grammar> Parser<G> {
    pub(crate) fn with_grammar(grammar: G, decoder: Box<dyn Decoder>) -> Self {
        Self {
            grammar,
            decoder,
            finished: false,
            next_token_index: 0,
        }
    }
    fn text(
        &mut self,
        ids: &[u32],
        token_start: usize,
        traced: bool,
        events: &mut EventSink,
    ) -> Result<(), Error> {
        if !ids.is_empty() {
            let text = if traced {
                self.decoder.push_attributed(ids, token_start)?
            } else {
                AttributedText::plain(self.decoder.push(ids)?)
            };
            self.grammar.text(text, events)?;
        }
        Ok(())
    }
    fn flush(&mut self, traced: bool, events: &mut EventSink) -> Result<(), Error> {
        let text = if traced {
            self.decoder.finish_attributed()?
        } else {
            AttributedText::plain(self.decoder.finish()?)
        };
        self.grammar.text(text, events)?;
        self.decoder.reset();
        Ok(())
    }
    fn feed_events(&mut self, ids: &[u32], traced: bool) -> Result<Vec<AttributedEvent>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed a response parser after finish"));
        }
        let base = self.next_token_index;
        self.next_token_index += ids.len();
        let mut events = EventSink::default();
        let mut start = 0;
        for (index, &id) in ids.iter().enumerate() {
            if self.grammar.ended() {
                if G::STRICT_END {
                    return Err(Error::new("response contains tokens after EOS"));
                }
                return Ok(events.0);
            }
            if self.grammar.is_control(id) {
                self.text(&ids[start..index], base + start, traced, &mut events)?;
                self.flush(traced, &mut events)?;
                self.grammar.control(id, &mut events)?;
                start = index + 1;
            }
        }
        if !self.grammar.ended() {
            self.text(&ids[start..], base + start, traced, &mut events)?;
        }
        Ok(events.0)
    }
    fn finish_events(
        &mut self,
        reason: FinishReason,
        traced: bool,
    ) -> Result<Vec<AttributedEvent>, Error> {
        let mut events = EventSink::default();
        if !self.finished {
            if !self.grammar.ended() {
                self.flush(traced, &mut events)?;
            }
            self.grammar.finish(reason, &mut events)?;
            self.finished = true;
        }
        Ok(events.0)
    }
}

impl<G: Grammar> ResponseParser for Parser<G> {
    fn supports_attribution(&mut self) -> bool {
        self.decoder.supports_attribution()
    }
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
        self.feed_events(ids, false)
            .map(|events| events.into_iter().map(|item| item.event).collect())
    }
    fn finish(&mut self, reason: FinishReason) -> Result<Vec<Event>, Error> {
        self.finish_events(reason, false)
            .map(|events| events.into_iter().map(|item| item.event).collect())
    }
    fn feed_attributed(&mut self, ids: &[u32]) -> Result<Vec<AttributedEvent>, Error> {
        self.feed_events(ids, true)
    }
    fn finish_attributed(&mut self, reason: FinishReason) -> Result<Vec<AttributedEvent>, Error> {
        self.finish_events(reason, true)
    }
}

pub(crate) fn emit_tool(
    index: &mut usize,
    name: impl Into<String>,
    arguments: impl Into<String>,
    events: &mut EventSink,
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
