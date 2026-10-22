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
    trace: trace::Trace,
}

impl TokenizerDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Result<Self, Error> {
        let trace = trace::Trace::new(&tokenizer)?;
        Ok(Self {
            tokenizer,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            trace,
        })
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
    fn push(&mut self, ids: &[u32], token_start: u32) -> Result<AttributedText, Error> {
        self.trace.push(&self.tokenizer, ids, token_start)?;
        let text = self.push_text(ids)?;
        self.trace.output(text)
    }

    fn finish(&mut self) -> Result<AttributedText, Error> {
        let text = self.finish_text()?;
        let result = self.trace.output(text);
        self.reset();
        result
    }

    fn reset(&mut self) {
        self.trace.clear();
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
    }
}
