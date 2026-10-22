//! Byte provenance retained while grammars slice, trim, and buffer decoded text.
use crate::Event;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, Range};

/// One token's source bytes mapped to a semantic delta's UTF-8 bytes.
/// Ranges are end-exclusive. Split Unicode scalars give each contributing token
/// the scalar's whole text interval, so text ranges may overlap.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(from = "(u32, u32, u32, u32, u32)", into = "(u32, u32, u32, u32, u32)")]
pub struct TokenSpan {
    /// Zero-based generated token index, including headers and controls.
    pub token: u32,
    pub source: Range<u32>,
    pub text: Range<u32>,
}
impl From<(u32, u32, u32, u32, u32)> for TokenSpan {
    fn from(
        (token, source_start, source_end, text_start, text_end): (u32, u32, u32, u32, u32),
    ) -> Self {
        Self {
            token,
            source: source_start..source_end,
            text: text_start..text_end,
        }
    }
}
impl From<TokenSpan> for (u32, u32, u32, u32, u32) {
    fn from(span: TokenSpan) -> Self {
        (
            span.token,
            span.source.start,
            span.source.end,
            span.text.start,
            span.text.end,
        )
    }
}
impl TokenSpan {
    pub(crate) fn clipped(&self, range: Range<u32>) -> Option<Self> {
        let start = self.text.start.max(range.start);
        let end = self.text.end.min(range.end);
        if start >= end {
            return None;
        }
        let mut source = self.source.clone();
        if source.end - source.start == self.text.end - self.text.start {
            source.start += start - self.text.start;
            source.end = source.start + end - start;
        } else if start != self.text.start || end != self.text.end {
            // Split scalars and replacement characters have indivisible ownership.
            return None;
        }
        Some(Self {
            token: self.token,
            source,
            text: start - range.start..end - range.start,
        })
    }
    fn joins(&self, next: &Self) -> bool {
        self.token == next.token
            && self.source.end == next.source.start
            && self.text.end == next.text.start
            && self.source.end - self.source.start == self.text.end - self.text.start
            && next.source.end - next.source.start == next.text.end - next.text.start
    }
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
        let bytes = u32::try_from(range.start).expect("text range exceeds u32")
            ..u32::try_from(range.end).expect("text range exceeds u32");
        Self {
            text: self.text[range].to_owned(),
            spans: self
                .spans
                .iter()
                .filter_map(|span| span.clipped(bytes.clone()))
                .collect(),
        }
    }
    pub fn take(&mut self, length: usize) -> Self {
        if length == self.text.len() {
            return std::mem::take(self);
        }
        if length == 0 {
            return Self::default();
        }
        let split = u32::try_from(length).expect("text range exceeds u32");
        let end = u32::try_from(self.text.len()).expect("text range exceeds u32");
        let tail = self.text.split_off(length);
        let head = std::mem::replace(&mut self.text, tail);
        let mut tail_spans = Vec::new();
        self.spans.retain_mut(|span| {
            if let Some(tail) = span.clipped(split..end) {
                tail_spans.push(tail);
            }
            if let Some(head) = span.clipped(0..split) {
                *span = head;
                true
            } else {
                false
            }
        });
        Self {
            text: head,
            spans: std::mem::replace(&mut self.spans, tail_spans),
        }
    }
    pub fn push(&mut self, text: Self) {
        if self.text.is_empty() {
            *self = text;
            return;
        }
        let offset = u32::try_from(self.text.len()).expect("text range exceeds u32");
        self.text.push_str(&text.text);
        for mut span in text.spans {
            span.text.start = span
                .text
                .start
                .checked_add(offset)
                .expect("text range exceeds u32");
            span.text.end = span
                .text
                .end
                .checked_add(offset)
                .expect("text range exceeds u32");
            self.push_span(span);
        }
    }
    pub(crate) fn push_span(&mut self, span: TokenSpan) {
        if let Some(previous) = self.spans.last_mut()
            && previous.joins(&span)
        {
            previous.source.end = span.source.end;
            previous.text.end = span.text.end;
        } else {
            self.spans.push(span);
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
