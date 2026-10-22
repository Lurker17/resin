//! Byte provenance retained while grammars slice, trim, and buffer decoded text.
use crate::Event;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, Range};

/// One source-token fragment contributing bytes to a semantic text delta.
/// All ranges are end-exclusive bytes. A Unicode scalar split across tokens
/// gives each contributing token the scalar's whole event interval; intervals
/// may therefore overlap. Source ranges always refer to `TokenBytes::get`.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TokenSpan {
    /// Zero-based index in generated IDs, including headers and controls.
    pub token_index: usize,
    pub token_start: usize,
    pub token_end: usize,
    /// Full original token byte extent, before grammar or stop trimming.
    pub token_len: usize,
    pub event_start: usize,
    pub event_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AttributedEvent {
    #[serde(flatten)]
    pub event: Event,
    #[serde(default, rename = "token_spans", skip_serializing_if = "Vec::is_empty")]
    pub spans: Vec<TokenSpan>,
}
impl From<Event> for AttributedEvent {
    fn from(event: Event) -> Self {
        Self {
            event,
            spans: Vec::new(),
        }
    }
}

/// Decoded text with exact source ranges; generated syntax has no source spans.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AttributedText {
    pub text: String,
    pub spans: Vec<TokenSpan>,
}
impl Deref for AttributedText {
    type Target = str;
    fn deref(&self) -> &str {
        &self.text
    }
}
impl AttributedText {
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            spans: Vec::new(),
        }
    }
    pub fn slice(&self, range: Range<usize>) -> Self {
        let spans = self
            .spans
            .iter()
            .filter_map(|span| {
                let start = span.event_start.max(range.start);
                let end = span.event_end.min(range.end);
                if start >= end {
                    return None;
                }
                let mut clipped = span.clone();
                if span.token_end - span.token_start == span.event_end - span.event_start {
                    clipped.token_start += start - span.event_start;
                    clipped.token_end = clipped.token_start + end - start;
                } else if start != span.event_start || end != span.event_end {
                    // A replacement character may represent one invalid source
                    // byte. It cannot be subdivided into invented byte ownership.
                    return None;
                }
                clipped.event_start = start - range.start;
                clipped.event_end = end - range.start;
                Some(clipped)
            })
            .collect();
        Self {
            text: self.text[range].to_owned(),
            spans,
        }
    }
    pub fn take(&mut self, length: usize) -> Self {
        let head = self.slice(0..length);
        *self = self.slice(length..self.text.len());
        head
    }
    pub fn push(&mut self, text: Self) {
        let offset = self.text.len();
        self.text.push_str(&text.text);
        for mut span in text.spans {
            span.event_start += offset;
            span.event_end += offset;
            if let Some(previous) = self.spans.last_mut()
                && previous.token_index == span.token_index
                && previous.token_len == span.token_len
                && previous.token_end == span.token_start
                && previous.event_end == span.event_start
                && previous.token_end - previous.token_start
                    == previous.event_end - previous.event_start
                && span.token_end - span.token_start == span.event_end - span.event_start
            {
                previous.token_end = span.token_end;
                previous.event_end = span.event_end;
            } else {
                self.spans.push(span);
            }
        }
    }
    pub fn clear(&mut self) {
        self.text.clear();
        self.spans.clear();
    }
    pub fn event(self, reasoning: bool) -> AttributedEvent {
        AttributedEvent {
            event: if reasoning {
                Event::ReasoningDelta { delta: self.text }
            } else {
                Event::TextDelta { delta: self.text }
            },
            spans: self.spans,
        }
    }
}
