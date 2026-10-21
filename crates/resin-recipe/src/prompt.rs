use aho_corasick::{AhoCorasick, MatchKind};
use std::sync::Arc;

use crate::{Error, Tokenizer};

/// Ordinary text and explicitly inserted model control tokens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptSpan {
    Text(String),
    Special(String),
}

/// A prompt before tokenization. Text coalesces across formatting operations so
/// BPE sees the same neighboring bytes as the reference encoder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Prompt {
    pub spans: Vec<PromptSpan>,
}

impl Prompt {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(&mut self, text: impl AsRef<str>) -> &mut Self {
        let text = text.as_ref();
        if !text.is_empty() {
            if let Some(PromptSpan::Text(previous)) = self.spans.last_mut() {
                previous.push_str(text);
            } else {
                self.spans.push(PromptSpan::Text(text.to_owned()));
            }
        }
        self
    }

    pub fn special(&mut self, token: impl Into<String>) -> &mut Self {
        self.spans.push(PromptSpan::Special(token.into()));
        self
    }

    pub fn rendered_text(&self) -> String {
        self.spans
            .iter()
            .map(|span| match span {
                PromptSpan::Text(text) | PromptSpan::Special(text) => text.as_str(),
            })
            .collect()
    }

    pub fn encode(&self, encoder: &dyn Encoder) -> Result<Vec<u32>, Error> {
        let mut ids = Vec::new();
        for span in &self.spans {
            match span {
                PromptSpan::Text(text) => ids.extend(encoder.encode_text(text)?),
                PromptSpan::Special(token) => ids.push(encoder.special_id(token)?),
            }
        }
        Ok(ids)
    }
}

/// Tokenization is injected independently of a model's serialization grammar.
pub trait Encoder: Send + Sync {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error>;
    fn special_id(&self, spelling: &str) -> Result<u32, Error>;
}

/// A shared HF tokenizer plus a variant that treats special spellings literally.
pub struct TokenizerEncoder {
    tokenizer: Arc<Tokenizer>,
    ordinary: Tokenizer,
    format_tokens: AhoCorasick,
}

impl TokenizerEncoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Result<Self, Error> {
        let format_tokens = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokenizer.get_added_tokens_decoder().into_values()
                .filter(|token| token.special || token.content.starts_with('<') || token.content.starts_with('｜'))
                .map(|token| token.content))
            .map_err(|error| Error::new(error.to_string()))?;
        let mut ordinary = (*tokenizer).clone();
        // Some checkpoints mark format delimiters (e.g. LFM's <think>) as
        // ordinary added tokens. They still require the same provenance rule.
        let controls: Vec<_> = ordinary
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.content.starts_with('<') || token.content.starts_with('｜'))
            .map(|token| token.special(true))
            .collect();
        ordinary
            .add_special_tokens(controls)
            .map_err(|error| Error::new(error.to_string()))?;
        ordinary.set_encode_special_tokens(true);
        Ok(Self {
            tokenizer,
            ordinary,
            format_tokens,
        })
    }

    /// Append trusted processor output while coalescing its text boundaries.
    pub fn append_format(&self, prompt: &mut Prompt, text: &str) {
        let mut previous = 0;
        for found in self.format_tokens.find_iter(text) {
            prompt
                .text(&text[previous..found.start()])
                .special(&text[found.start()..found.end()]);
            previous = found.end();
        }
        prompt.text(&text[previous..]);
    }
}

impl Encoder for TokenizerEncoder {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.ordinary
            .encode_fast(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| Error::new(error.to_string()))
    }

    fn special_id(&self, spelling: &str) -> Result<u32, Error> {
        self.tokenizer
            .token_to_id(spelling)
            .or_else(|| {
                let alias = match spelling {
                    "<|start|>" => "<|im_start|>",
                    "<|end|>" => "<|im_end|>",
                    "<|return|>" => "<|fim_suffix|>",
                    "<|call|>" => "<|ghissue|>",
                    "<|channel|>" => "<|meta_sep|>",
                    "<|message|>" => "<|im_sep|>",
                    "<|constrain|>" => "<|meta_start|>",
                    _ => return None,
                };
                self.tokenizer.token_to_id(alias)
            })
            .ok_or_else(|| Error::new(format!("missing model control token {spelling:?}")))
    }
}
