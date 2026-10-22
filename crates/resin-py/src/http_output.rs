//! Join visible text with complete source-token scores after parser/stop buffering.
use std::collections::BTreeMap;

use resin_http::{GeneratedTokenLogprob, GenerationEvent};
use resin_recipe::{AttributedEvent, AttributedText, Event};

#[derive(Default)]
pub(crate) struct OutputScores {
    tokens: BTreeMap<usize, GeneratedTokenLogprob>,
    pending: AttributedText,
    retired_before: usize,
}

impl OutputScores {
    pub fn insert(&mut self, token: GeneratedTokenLogprob) {
        if token.token_index >= self.retired_before {
            self.tokens.insert(token.token_index, token);
        }
    }

    pub fn push(&mut self, attributed: AttributedEvent) -> Vec<GenerationEvent> {
        let AttributedEvent { event, spans } = attributed;
        if let Event::TextDelta { delta } = event {
            if delta.is_empty() {
                return Vec::new();
            }
            if spans.is_empty() {
                let mut output = self.finish_text();
                output.push(GenerationEvent::TextDelta {
                    delta,
                    logprobs: Vec::new(),
                });
                return output;
            }
            let text = AttributedText { text: delta, spans };
            if self.pending.is_empty() {
                self.pending = text;
            } else {
                self.pending.push(text);
            }
            let mut boundary = self.pending.len();
            if let Some(last) = self.pending.spans.last() {
                let fragments = self
                    .pending
                    .spans
                    .iter()
                    .filter(|span| span.token_index == last.token_index);
                let mut covered = 0;
                let mut start = boundary;
                let mut contiguous = true;
                for span in fragments {
                    start = start.min(span.event_start);
                    contiguous &= span.token_start == covered;
                    covered = span.token_end;
                }
                // A missing leading prefix cannot become a whole visible token.
                // A trailing prefix can still be completed by the next event.
                if contiguous && covered < last.token_len {
                    boundary = start;
                    let mut ranges = BTreeMap::<usize, (usize, usize)>::new();
                    for span in &self.pending.spans {
                        let range = ranges
                            .entry(span.token_index)
                            .or_insert((span.event_start, span.event_end));
                        range.0 = range.0.min(span.event_start);
                        range.1 = range.1.max(span.event_end);
                    }
                    // Byte tokens can overlap one Unicode character. Preserve
                    // every source token connected to that held character too.
                    loop {
                        let previous = boundary;
                        while !self.pending.is_char_boundary(boundary) {
                            boundary -= 1;
                        }
                        for &(start, end) in ranges.values() {
                            if start < boundary && end > boundary {
                                boundary = start;
                            }
                        }
                        if boundary == previous {
                            break;
                        }
                    }
                }
            }
            return self.emit(boundary).into_iter().collect();
        }
        let mut output = self.finish_text();
        if let Some(last) = spans.iter().map(|span| span.token_index).max() {
            self.retire(last + 1);
        }
        output.push(match event {
            Event::ReasoningDelta { delta } => GenerationEvent::ReasoningDelta(delta),
            Event::ToolCallStart { index, id, name } => {
                GenerationEvent::ToolCallStart { index, id, name }
            }
            Event::ToolCallArgumentsDelta { index, delta } => {
                GenerationEvent::ToolCallArgumentsDelta { index, delta }
            }
            Event::TextDelta { .. } => unreachable!(),
        });
        output
    }

    pub fn finish(&mut self) -> Vec<GenerationEvent> {
        let output = self.finish_text();
        self.tokens.clear();
        output
    }

    fn finish_text(&mut self) -> Vec<GenerationEvent> {
        self.emit(self.pending.len()).into_iter().collect()
    }

    fn emit(&mut self, boundary: usize) -> Option<GenerationEvent> {
        if boundary == 0 {
            return None;
        }
        let text = if boundary == self.pending.len() {
            std::mem::take(&mut self.pending)
        } else {
            self.pending.take(boundary)
        };
        let mut coverage = BTreeMap::<usize, (usize, usize, bool)>::new();
        for span in &text.spans {
            let entry = coverage
                .entry(span.token_index)
                .or_insert((0, span.token_len, true));
            entry.2 &= entry.0 == span.token_start
                && entry.1 == span.token_len
                && span.token_end <= span.token_len;
            entry.0 = span.token_end;
        }
        let mut logprobs = Vec::new();
        for (index, (covered, length, contiguous)) in &coverage {
            if *contiguous
                && covered == length
                && let Some(mut token) = self.tokens.remove(index)
            {
                token.channel = Some("content".to_owned());
                logprobs.push(token);
            }
        }
        if let Some(last) = coverage.keys().next_back() {
            self.retire(last + 1);
        }
        Some(GenerationEvent::TextDelta {
            delta: text.text,
            logprobs,
        })
    }

    fn retire(&mut self, before: usize) {
        self.retired_before = self.retired_before.max(before);
        self.tokens = self.tokens.split_off(&self.retired_before);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_http::TokenLogprob;
    use resin_recipe::TokenSpan;

    fn token(index: usize, bytes: &[u8]) -> GeneratedTokenLogprob {
        GeneratedTokenLogprob {
            token: TokenLogprob {
                token_id: index as u32 + 100,
                token: String::from_utf8_lossy(bytes).into_owned(),
                bytes: Some(bytes.to_vec()),
                logprob: -0.5,
            },
            token_index: index,
            channel: None,
            top_logprobs: vec![TokenLogprob {
                token_id: 200,
                token: "alternative".to_owned(),
                bytes: Some(b"alternative".to_vec()),
                logprob: -1.0,
            }],
        }
    }

    fn span(
        index: usize,
        source: std::ops::Range<usize>,
        length: usize,
        event: std::ops::Range<usize>,
    ) -> TokenSpan {
        TokenSpan {
            token_index: index,
            token_start: source.start,
            token_end: source.end,
            token_len: length,
            event_start: event.start,
            event_end: event.end,
        }
    }

    fn text(delta: &str, spans: Vec<TokenSpan>) -> AttributedEvent {
        AttributedEvent {
            event: Event::TextDelta {
                delta: delta.to_owned(),
            },
            spans,
        }
    }

    fn unpack(events: Vec<GenerationEvent>) -> Vec<(String, Vec<GeneratedTokenLogprob>)> {
        events
            .into_iter()
            .map(|event| match event {
                GenerationEvent::TextDelta { delta, logprobs } => (delta, logprobs),
                _ => panic!("expected visible text"),
            })
            .collect()
    }

    #[test]
    fn withheld_suffix_later_completes_one_source_token() {
        let mut output = OutputScores::default();
        let original = token(0, b"hello ");
        output.insert(original.clone());
        assert!(
            output
                .push(text("hello", vec![span(0, 0..5, 6, 0..5)]))
                .is_empty()
        );
        let events = unpack(output.push(text(" ", vec![span(0, 5..6, 6, 0..1)])));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "hello ");
        assert_eq!(events[0].1.len(), 1);
        let mut expected = original;
        expected.channel = Some("content".to_owned());
        assert_eq!(
            serde_json::to_value(&events[0].1[0]).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        assert!(output.finish().is_empty());
    }

    #[test]
    fn incomplete_source_at_finish_or_after_trim_never_gets_a_partial_score() {
        let mut output = OutputScores::default();
        output.insert(token(0, b"hello "));
        assert!(
            output
                .push(text("hello", vec![span(0, 0..5, 6, 0..5)]))
                .is_empty()
        );
        let events = unpack(output.finish());
        assert_eq!(events[0].0, "hello");
        assert!(events[0].1.is_empty());

        output.insert(token(1, b" hello"));
        let events = unpack(output.push(text("hello", vec![span(1, 1..6, 6, 0..5)])));
        assert_eq!(events[0].0, "hello");
        assert!(events[0].1.is_empty());
    }

    #[test]
    fn emits_whole_tokens_before_a_held_partial_tail_and_on_source_advance() {
        let mut output = OutputScores::default();
        output.insert(token(0, b"one"));
        output.insert(token(1, b" two "));
        let events = unpack(output.push(text(
            "one two",
            vec![span(0, 0..3, 3, 0..3), span(1, 0..4, 5, 3..7)],
        )));
        assert_eq!(events[0].0, "one");
        assert_eq!(events[0].1[0].token_index, 0);
        output.insert(token(2, b"!"));
        let events = unpack(output.push(text("!", vec![span(2, 0..1, 1, 0..1)])));
        assert_eq!(events[0].0, " two!");
        assert_eq!(
            events[0]
                .1
                .iter()
                .map(|t| t.token_index)
                .collect::<Vec<_>>(),
            [2]
        );
    }

    #[test]
    fn utf8_byte_tokens_emit_together_with_original_bytes() {
        let mut output = OutputScores::default();
        output.insert(token(0, &[0xc3]));
        output.insert(token(1, &[0xa9, b' ']));
        // The character is valid UTF-8, but its second source token still has
        // a withheld space. Neither score can accompany a partial character.
        assert!(
            output
                .push(text(
                    "é",
                    vec![span(0, 0..1, 1, 0..1), span(1, 0..1, 2, 1..2)]
                ))
                .is_empty()
        );
        let events = unpack(output.push(text(" ", vec![span(1, 1..2, 2, 0..1)])));
        assert_eq!(events[0].0, "é ");
        assert_eq!(
            events[0]
                .1
                .iter()
                .map(|t| t.token_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            events[0].1[0].token.bytes.as_deref(),
            Some([0xc3].as_slice())
        );
        assert_eq!(
            events[0].1[1].token.bytes.as_deref(),
            Some([0xa9, b' '].as_slice())
        );
    }

    #[test]
    fn held_utf8_character_also_holds_earlier_bytes_of_its_source_token() {
        let mut output = OutputScores::default();
        output.insert(token(0, &[b'a', 0xc3]));
        output.insert(token(1, &[0xa9, b' ']));
        assert!(
            output
                .push(text(
                    "aé",
                    vec![span(0, 0..2, 2, 0..3), span(1, 0..1, 2, 1..3)]
                ))
                .is_empty()
        );
        let events = unpack(output.push(text(" ", vec![span(1, 1..2, 2, 0..1)])));
        assert_eq!(events[0].0, "aé ");
        assert_eq!(
            events[0]
                .1
                .iter()
                .map(|t| t.token_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn lane_changes_flush_partial_text_and_unknown_spans_never_guess() {
        let mut output = OutputScores::default();
        output.insert(token(0, b"text "));
        assert!(
            output
                .push(text("text", vec![span(0, 0..4, 5, 0..4)]))
                .is_empty()
        );
        let events = output.push(
            Event::ReasoningDelta {
                delta: "reason".to_owned(),
            }
            .into(),
        );
        assert!(
            matches!(&events[0], GenerationEvent::TextDelta { delta, logprobs } if delta == "text" && logprobs.is_empty())
        );
        assert!(matches!(&events[1], GenerationEvent::ReasoningDelta(delta) if delta == "reason"));
        output.insert(token(1, b"unknown"));
        let events = unpack(output.push(text("unknown", vec![])));
        assert_eq!(events[0].0, "unknown");
        assert!(events[0].1.is_empty());
    }

    #[test]
    fn scores_are_once_only_and_require_gapless_source_coverage() {
        let mut output = OutputScores::default();
        let mut first = token(0, b"x");
        // Provenance still supplies the byte extent when vocabulary bytes are unknown.
        first.token.bytes = None;
        output.insert(first.clone());
        let events = unpack(output.push(text("x", vec![span(0, 0..1, 1, 0..1)])));
        assert_eq!(events[0].1.len(), 1);
        assert!(events[0].1[0].token.bytes.is_none());
        output.insert(first);
        let events = unpack(output.push(text("x", vec![span(0, 0..1, 1, 0..1)])));
        assert!(events[0].1.is_empty());

        output.insert(token(1, b"abcd"));
        let events = unpack(output.push(text(
            "acd",
            vec![span(1, 0..1, 4, 0..1), span(1, 2..4, 4, 1..3)],
        )));
        assert_eq!(events[0].0, "acd");
        assert!(events[0].1.is_empty());
    }
}
