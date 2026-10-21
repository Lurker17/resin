use std::sync::Arc;

pub use tokenizers::Tokenizer;
use tokenizers::tokenizer::step_decode_stream;

use crate::{Decoder, Error};

/// Hugging Face's streaming decoder with owned state and a shared tokenizer.
///
/// Retains the context needed by the tokenizer's decoder and incomplete UTF-8,
/// rather than decoding the accumulated response on every update.
pub struct TokenizerDecoder {
    tokenizer: Arc<Tokenizer>,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl TokenizerDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }
}

impl Decoder for TokenizerDecoder {
    fn push(&mut self, ids: &[u32]) -> Result<String, Error> {
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

    fn finish(&mut self) -> Result<String, Error> {
        let text = self
            .tokenizer
            .decode(&self.ids, false)
            .map_err(|error| Error::new(error.to_string()))?;
        let suffix = text
            .strip_prefix(&self.prefix)
            .ok_or_else(|| Error::new("decoded output is not prefix-stable"))?
            .to_owned();
        self.reset();
        Ok(suffix)
    }

    fn reset(&mut self) {
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
    }
}
