//! Stop strings operate on decoded output, independently of tokenizer boundaries.

use pyo3::{PyResult, exceptions::PyValueError};
use resin_recipe::{AttributedEvent, AttributedText, Event};

pub(crate) struct StopFilter {
    stops: Vec<String>,
    pending: Option<AttributedEvent>,
    matched: Option<String>,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> PyResult<Self> {
        if stops.iter().any(String::is_empty) {
            return Err(PyValueError::new_err("stop sequences must not be empty"));
        }
        Ok(Self {
            stops,
            pending: None,
            matched: None,
        })
    }

    pub fn stopped(&self) -> bool {
        self.matched.is_some()
    }

    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    pub fn push(&mut self, mut event: AttributedEvent) -> Vec<AttributedEvent> {
        if self.stopped() {
            return Vec::new();
        }
        if self.stops.is_empty() {
            return vec![event];
        }
        let mut output = Vec::new();
        if let Some(pending) = self.pending.take() {
            if same_lane(&pending.event, &event.event) {
                let mut text = attributed_text(&pending).unwrap();
                text.push(attributed_text(&event).unwrap());
                event = with_text(pending, text);
            } else {
                output.push(pending);
            }
        }
        let Some(mut text) = attributed_text(&event) else {
            output.push(event);
            return output;
        };
        if let Some((position, stop)) = self
            .stops
            .iter()
            .filter_map(|stop| text.find(stop).map(|position| (position, stop)))
            .min_by_key(|(position, _)| *position)
        {
            text = text.slice(0..position);
            self.matched = Some(stop.clone());
        } else {
            // Retain exactly the suffix that could become a stop on the next
            // delta. Character boundaries keep split Unicode markers valid.
            let held = self
                .stops
                .iter()
                .flat_map(|stop| stop.char_indices().skip(1).map(|(end, _)| &stop[..end]))
                .filter(|prefix| text.ends_with(prefix))
                .map(str::len)
                .max()
                .unwrap_or(0);
            if held != 0 {
                let head = text.take(text.len() - held);
                self.pending = Some(with_text(event.clone(), text));
                text = head;
            }
        }
        if !text.is_empty() {
            output.push(with_text(event, text));
        }
        output
    }

    pub fn finish(&mut self) -> Vec<AttributedEvent> {
        self.pending.take().into_iter().collect()
    }
}

fn delta(event: &Event) -> Option<&str> {
    match event {
        Event::TextDelta { delta }
        | Event::ReasoningDelta { delta }
        | Event::ToolCallArgumentsDelta { delta, .. } => Some(delta),
        Event::ToolCallStart { .. } => None,
    }
}

fn attributed_text(event: &AttributedEvent) -> Option<AttributedText> {
    Some(AttributedText {
        text: delta(&event.event)?.to_owned(),
        spans: event.spans.clone(),
    })
}

fn with_text(mut event: AttributedEvent, text: AttributedText) -> AttributedEvent {
    match &mut event.event {
        Event::TextDelta { delta }
        | Event::ReasoningDelta { delta }
        | Event::ToolCallArgumentsDelta { delta, .. } => *delta = text.text,
        Event::ToolCallStart { .. } => unreachable!("text event"),
    }
    event.spans = text.spans;
    event
}

fn same_lane(left: &Event, right: &Event) -> bool {
    match (left, right) {
        (Event::TextDelta { .. }, Event::TextDelta { .. })
        | (Event::ReasoningDelta { .. }, Event::ReasoningDelta { .. }) => true,
        (
            Event::ToolCallArgumentsDelta { index: a, .. },
            Event::ToolCallArgumentsDelta { index: b, .. },
        ) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(value: &str) -> AttributedEvent {
        Event::TextDelta {
            delta: value.into(),
        }
        .into()
    }
    fn filter(stops: &[&str]) -> StopFilter {
        StopFilter::new(stops.iter().map(|s| s.to_string()).collect()).unwrap()
    }

    #[test]
    fn markers_are_hidden_across_every_possible_chunk_boundary() {
        let input = "hello🦊ENDtail";
        for split in input.char_indices().map(|(index, _)| index) {
            let mut filter = filter(&["🦊END"]);
            let output = [
                filter.push(text(&input[..split])),
                filter.push(text(&input[split..])),
                filter.finish(),
            ]
            .concat();
            let actual: String = output
                .into_iter()
                .map(|e| match e.event {
                    Event::TextDelta { delta } => delta,
                    _ => unreachable!(),
                })
                .collect();
            assert_eq!(actual, "hello");
            assert!(filter.stopped());
            assert!(filter.push(text("ignored")).is_empty());
        }
    }

    #[test]
    fn overlapping_prefixes_do_not_hide_unmatched_text_at_length_limit() {
        let mut filter = filter(&["aab", "abc"]);
        assert_eq!(filter.push(text("za")), vec![text("z")]);
        assert_eq!(filter.push(text("aa")), vec![text("a")]);
        assert_eq!(filter.finish(), vec![text("aa")]);
        assert!(!filter.stopped());
    }

    #[test]
    fn first_match_wins_and_channels_do_not_join() {
        let mut filter = filter(&["STOP", "END"]);
        assert_eq!(filter.push(text("aENDbSTOP")), vec![text("a")]);
        assert!(filter.stopped());
        let mut filter = self::filter(&["STOP"]);
        assert!(
            filter
                .push(Event::ReasoningDelta { delta: "ST".into() }.into())
                .is_empty()
        );
        assert_eq!(
            filter.push(text("OP")),
            vec![
                Event::ReasoningDelta { delta: "ST".into() }.into(),
                text("OP")
            ]
        );
        assert!(!filter.stopped());
    }

    #[test]
    fn invalid_empty_marker_is_rejected() {
        assert!(StopFilter::new(vec![String::new()]).is_err());
    }

    #[test]
    fn held_suffix_preserves_original_token_bytes_until_resolved() {
        use resin_recipe::TokenSpan;

        let mut filter = filter(&["SX"]);
        let mut first = text("abS");
        first.spans.push(TokenSpan {
            token_index: 0,
            token_start: 0,
            token_end: 3,
            token_len: 3,
            event_start: 0,
            event_end: 3,
        });
        let head = filter.push(first);
        assert_eq!(delta(&head[0].event), Some("ab"));
        assert_eq!(head[0].spans[0].token_end, 2);
        let mut next = text("Y");
        next.spans.push(TokenSpan {
            token_index: 1,
            token_start: 0,
            token_end: 1,
            token_len: 1,
            event_start: 0,
            event_end: 1,
        });
        let released = filter.push(next);
        assert_eq!(delta(&released[0].event), Some("SY"));
        assert_eq!(released[0].spans[0].token_index, 0);
        assert_eq!(released[0].spans[0].token_start, 2);
        assert_eq!(released[0].spans[0].token_end, 3);
        assert_eq!(released[0].spans[1].token_index, 1);
        assert_eq!(released[0].spans[1].event_start, 1);
        assert!(!filter.stopped());
    }
}
