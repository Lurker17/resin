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
    pub(crate) next_token_index: u32,
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
    fn text(&mut self, ids: &[u32], token_start: u32, events: &mut EventSink) -> Result<(), Error> {
        if !ids.is_empty() {
            let text = self.decoder.push(ids, token_start)?;
            self.grammar.text(text, events)?;
        }
        Ok(())
    }
    fn flush(&mut self, events: &mut EventSink) -> Result<(), Error> {
        let text = self.decoder.finish()?;
        self.grammar.text(text, events)?;
        self.decoder.reset();
        Ok(())
    }
    fn feed_events(&mut self, ids: &[u32]) -> Result<Vec<AttributedEvent>, Error> {
        if self.finished {
            return Err(Error::new("cannot feed a response parser after finish"));
        }
        let base = self.next_token_index;
        self.next_token_index = u32::try_from(ids.len())
            .ok()
            .and_then(|length| base.checked_add(length))
            .ok_or_else(|| Error::new("generated token indices exceed the u32 range"))?;
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
                self.text(&ids[start..index], base + start as u32, &mut events)?;
                self.flush(&mut events)?;
                self.grammar.control(id, &mut events)?;
                start = index + 1;
            }
        }
        if !self.grammar.ended() {
            self.text(&ids[start..], base + start as u32, &mut events)?;
        }
        Ok(events.0)
    }
    fn finish_events(&mut self, reason: FinishReason) -> Result<Vec<AttributedEvent>, Error> {
        let mut events = EventSink::default();
        if !self.finished {
            if !self.grammar.ended() {
                self.flush(&mut events)?;
            }
            self.grammar.finish(reason, &mut events)?;
            self.finished = true;
        }
        Ok(events.0)
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
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<AttributedEvent>, Error> {
        self.feed_events(ids)
    }
    fn finish(&mut self, reason: FinishReason) -> Result<Vec<AttributedEvent>, Error> {
        self.finish_events(reason)
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture;
    impl Decoder for Fixture {
        fn push(&mut self, _: &[u32], _: u32) -> Result<AttributedText, Error> {
            panic!("overflow must be rejected before decoder state changes")
        }
        fn finish(&mut self) -> Result<AttributedText, Error> {
            Ok(AttributedText::default())
        }
        fn reset(&mut self) {}
    }
    impl Grammar for Fixture {
        fn is_control(&self, _: u32) -> bool {
            false
        }
        fn text(&mut self, _: AttributedText, _: &mut EventSink) -> Result<(), Error> {
            Ok(())
        }
        fn control(&mut self, _: u32, _: &mut EventSink) -> Result<(), Error> {
            Ok(())
        }
    }

    #[test]
    fn overflowing_source_indices_fail_before_decoder_mutation() {
        let mut parser = Parser::with_grammar(Fixture, Box::new(Fixture));
        parser.next_token_index = u32::MAX - 1;
        assert!(parser.feed(&[1, 2]).is_err());
        assert_eq!(parser.next_token_index, u32::MAX - 1);
        assert!(parser.feed(&[]).unwrap().is_empty());
    }
}
