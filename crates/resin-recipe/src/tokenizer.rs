mod bytes;
mod trace;
pub use bytes::TokenBytes;

use std::sync::Arc;

pub use tokenizers::Tokenizer;
use tokenizers::tokenizer::step_decode_stream;

use crate::{AttributedText, Decoder, Error};

/// Hugging Face's streaming decoder with owned state and a shared tokenizer.
///
/// Retains the context needed by the tokenizer's decoder and incomplete UTF-8,
/// rather than decoding the accumulated response on every update.
pub struct TokenizerDecoder {
    tokenizer: Arc<Tokenizer>,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    trace: Option<trace::Trace>,
    untraced: bool,
}

impl TokenizerDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            trace: None,
            untraced: false,
        }
    }
    fn push_text(&mut self, ids: &[u32]) -> Result<String, Error> {
        if ids.is_empty() {
            return Ok(String::new());
        }
        step_decode_stream(
            &self.tokenizer,
            ids.to_vec(),
            false,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map(Option::unwrap_or_default)
        .map_err(|error| Error::new(error.to_string()))
    }

    fn finish_text(&mut self) -> Result<String, Error> {
        let text = self
            .tokenizer
            .decode(&self.ids, false)
            .map_err(|error| Error::new(error.to_string()))?;
        let suffix = text
            .strip_prefix(&self.prefix)
            .ok_or_else(|| Error::new("decoded output is not prefix-stable"))?
            .to_owned();
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
        Ok(suffix)
    }
}

impl Decoder for TokenizerDecoder {
    fn supports_attribution(&mut self) -> bool {
        self.trace
            .get_or_insert_with(|| trace::Trace::new(&self.tokenizer))
            .supported()
    }
    fn push(&mut self, ids: &[u32]) -> Result<String, Error> {
        if !ids.is_empty() {
            self.untraced = true;
            if let Some(trace) = &mut self.trace {
                trace.mark_unknown();
            }
        }
        self.push_text(ids)
    }
    fn finish(&mut self) -> Result<String, Error> {
        let text = self.finish_text()?;
        self.reset();
        Ok(text)
    }
    fn push_attributed(
        &mut self,
        ids: &[u32],
        token_start: usize,
    ) -> Result<AttributedText, Error> {
        let trace = self
            .trace
            .get_or_insert_with(|| trace::Trace::new(&self.tokenizer));
        if self.untraced {
            trace.mark_unknown();
        }
        trace.push(&self.tokenizer, ids, token_start);
        let text = self.push_text(ids)?;
        Ok(self.trace.as_mut().unwrap().output(text))
    }

    fn finish_attributed(&mut self) -> Result<AttributedText, Error> {
        let text = self.finish_text()?;
        let result = match &mut self.trace {
            Some(trace) => trace.output(text),
            None => AttributedText::plain(text),
        };
        self.reset();
        Ok(result)
    }

    fn reset(&mut self) {
        self.untraced = false;
        if let Some(trace) = &mut self.trace {
            trace.clear();
        }
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
    }
}
