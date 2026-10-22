//! Stop strings operate on decoded output, independently of tokenizer boundaries.

use pyo3::{PyResult, exceptions::PyValueError};
use resin_recipe::Event;

pub(crate) struct StopFilter {
    stops: Vec<String>,
    pending: Option<Event>,
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

    pub fn push(&mut self, mut event: Event) -> Vec<Event> {
        if self.stopped() {
            return Vec::new();
        }
        if self.stops.is_empty() {
            return vec![event];
        }
        let mut output = Vec::new();
        if let Some(mut pending) = self.pending.take() {
            if same_lane(&pending, &event) {
                delta(&mut pending)
                    .unwrap()
                    .push_str(delta(&mut event).unwrap());
                event = pending;
            } else {
                output.push(pending);
            }
        }
        let Some(text) = delta(&mut event) else {
            output.push(event);
            return output;
        };
        if let Some((position, stop)) = self
            .stops
            .iter()
            .filter_map(|stop| text.find(stop).map(|position| (position, stop)))
            .min_by_key(|(position, _)| *position)
        {
            text.truncate(position);
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
                let suffix = text.split_off(text.len() - held);
                let mut pending = event.clone();
                *delta(&mut pending).unwrap() = suffix;
                self.pending = Some(pending);
            }
        }
        if !delta(&mut event).unwrap().is_empty() {
            output.push(event);
        }
        output
    }

    pub fn finish(&mut self) -> Vec<Event> {
        self.pending.take().into_iter().collect()
    }
}

fn delta(event: &mut Event) -> Option<&mut String> {
    match event {
        Event::TextDelta { delta }
        | Event::ReasoningDelta { delta }
        | Event::ToolCallArgumentsDelta { delta, .. } => Some(delta),
        Event::ToolCallStart { .. } => None,
    }
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

    fn text(value: &str) -> Event {
        Event::TextDelta {
            delta: value.into(),
        }
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
                .map(|e| match e {
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
                .push(Event::ReasoningDelta { delta: "ST".into() })
                .is_empty()
        );
        assert_eq!(
            filter.push(text("OP")),
            vec![Event::ReasoningDelta { delta: "ST".into() }, text("OP")]
        );
        assert!(!filter.stopped());
    }

    #[test]
    fn invalid_empty_marker_is_rejected() {
        assert!(StopFilter::new(vec![String::new()]).is_err());
    }
}
