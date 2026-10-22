use resin_recipe::{Error, Event, FinishReason, ResponseParser};

/// Grammar fixtures compare semantic events; provenance has its own integration tests.
pub trait SemanticParser: ResponseParser {
    fn semantic_feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        self.feed(ids)
            .map(|events| events.into_iter().map(|event| event.event).collect())
    }
    fn semantic_finish(&mut self, reason: FinishReason) -> Result<Vec<Event>, Error> {
        self.finish(reason)
            .map(|events| events.into_iter().map(|event| event.event).collect())
    }
}
impl<T: ResponseParser> SemanticParser for T {}
