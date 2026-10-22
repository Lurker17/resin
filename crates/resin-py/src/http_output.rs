//! Join visible text with complete source-token scores after parser/stop buffering.
use std::collections::VecDeque;

use resin_http::{GeneratedTokenLogprob, GenerationEvent};
use resin_recipe::{AttributedEvent, AttributedText, Event};

#[derive(Default)]
pub(crate) struct OutputScores {
    // Scores arrive in generated-token order; retain capacity across emissions.
    tokens: VecDeque<GeneratedTokenLogprob>,
    pending: AttributedText,
    retired_before: usize,
}

impl OutputScores {
    pub fn insert(&mut self, token: GeneratedTokenLogprob) {
        if token.token_index >= self.retired_before && token.token.bytes.is_some() {
            self.tokens.push_back(token);
        }
    }

    fn position(&self, token: u32) -> Option<usize> {
        self.tokens
            .binary_search_by_key(&(token as usize), |score| score.token_index)
            .ok()
    }

    pub fn source_len(&self, token: u32) -> Option<usize> {
        let score = &self.tokens[self.position(token)?];
        score.token.bytes.as_ref().map(Vec::len)
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
            self.pending.push(AttributedText { text: delta, spans });
            let mut boundary = self.pending.len();
            if let Some(last) = self.pending.spans.last() {
                let mut covered = 0;
                let mut start = boundary;
                let mut contiguous = true;
                for span in self
                    .pending
                    .spans
                    .iter()
                    .filter(|span| span.token == last.token)
                {
                    start = start.min(span.text.start as usize);
                    contiguous &= span.source.start == covered;
                    covered = span.source.end;
                }
                // Only a missing trailing suffix can still complete this token.
                if contiguous
                    && self
                        .source_len(last.token)
                        .is_some_and(|len| (covered as usize) < len)
                {
                    boundary = start;
                    // Split UTF-8 scalars connect tokens. Keep the entire source
                    // token connected to a held character, without a range map.
                    loop {
                        let previous = boundary;
                        while !self.pending.is_char_boundary(boundary) {
                            boundary -= 1;
                        }
                        for group in self.pending.spans.chunk_by(|a, b| a.token == b.token) {
                            let start = group.iter().map(|s| s.text.start as usize).min().unwrap();
                            let end = group.iter().map(|s| s.text.end as usize).max().unwrap();
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
        if let Some(last) = spans.last() {
            self.retire(last.token as usize + 1);
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
        let text = self.pending.take(boundary);
        let mut logprobs = Vec::new();
        for group in text.spans.chunk_by(|a, b| a.token == b.token) {
            let token = group[0].token;
            let mut covered = 0;
            let contiguous = group.iter().all(|span| {
                let adjacent = covered == span.source.start;
                covered = span.source.end;
                adjacent
            });
            if contiguous
                && self.source_len(token) == Some(covered as usize)
                && let Some(position) = self.position(token)
            {
                let mut score = self.tokens.remove(position).unwrap();
                score.channel = Some("content".to_owned());
                logprobs.push(score);
            }
        }
        if let Some(last) = text.spans.last() {
            self.retire(last.token as usize + 1);
        }
        Some(GenerationEvent::TextDelta {
            delta: text.text,
            logprobs,
        })
    }

    fn retire(&mut self, before: usize) {
        self.retired_before = self.retired_before.max(before);
        while self
            .tokens
            .front()
            .is_some_and(|score| score.token_index < self.retired_before)
        {
            self.tokens.pop_front();
        }
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
        event: std::ops::Range<usize>,
    ) -> TokenSpan {
        TokenSpan {
            token: index as u32,
            source: source.start as u32..source.end as u32,
            text: event.start as u32..event.end as u32,
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
                .push(text("hello", vec![span(0, 0..5, 0..5)]))
                .is_empty()
        );
        let events = unpack(output.push(text(" ", vec![span(0, 5..6, 0..1)])));
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
                .push(text("hello", vec![span(0, 0..5, 0..5)]))
                .is_empty()
        );
        let events = unpack(output.finish());
        assert_eq!(events[0].0, "hello");
        assert!(events[0].1.is_empty());

        output.insert(token(1, b" hello"));
        let events = unpack(output.push(text("hello", vec![span(1, 1..6, 0..5)])));
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
            vec![span(0, 0..3, 0..3), span(1, 0..4, 3..7)],
        )));
        assert_eq!(events[0].0, "one");
        assert_eq!(events[0].1[0].token_index, 0);
        output.insert(token(2, b"!"));
        let events = unpack(output.push(text("!", vec![span(2, 0..1, 0..1)])));
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
                .push(text("é", vec![span(0, 0..1, 0..1), span(1, 0..1, 1..2)]))
                .is_empty()
        );
        let events = unpack(output.push(text(" ", vec![span(1, 1..2, 0..1)])));
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
                .push(text("aé", vec![span(0, 0..2, 0..3), span(1, 0..1, 1..3)]))
                .is_empty()
        );
        let events = unpack(output.push(text(" ", vec![span(1, 1..2, 0..1)])));
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
                .push(text("text", vec![span(0, 0..4, 0..4)]))
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
        let first = token(0, b"x");
        output.insert(first.clone());
        let capacity = output.tokens.capacity();
        let events = unpack(output.push(text("x", vec![span(0, 0..1, 0..1)])));
        assert_eq!(events[0].1.len(), 1);
        assert_eq!(output.tokens.capacity(), capacity);
        output.insert(first);
        let events = unpack(output.push(text("x", vec![span(0, 0..1, 0..1)])));
        assert!(events[0].1.is_empty());

        output.insert(token(1, b"abcd"));
        let events =
            unpack(output.push(text("acd", vec![span(1, 0..1, 0..1), span(1, 2..4, 1..3)])));
        assert_eq!(events[0].0, "acd");
        assert!(events[0].1.is_empty());

        // Without original vocabulary bytes, completeness cannot be established.
        let mut unknown = token(2, b"x");
        unknown.token.bytes = None;
        output.insert(unknown);
        let events = unpack(output.push(text("x", vec![span(2, 0..1, 0..1)])));
        assert!(events[0].1.is_empty());
    }
}
