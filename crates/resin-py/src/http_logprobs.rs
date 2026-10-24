//! Token representations prepared once, independently of semantic parsing.
use std::sync::Arc;

use crate::recipe::{RecipeTokenizer, optional_attr};
use pyo3::{exceptions::PyTypeError, prelude::*};
use resin_http::{GeneratedTokenLogprob, TokenLogprob};
use resin_recipe::{TokenBytes, Tokenizer};

pub(crate) struct Vocabulary {
    tokenizer: Option<Arc<Tokenizer>>,
    bytes: Option<TokenBytes>,
}

impl Vocabulary {
    pub fn from_recipe(py: Python<'_>, recipe: &Py<PyAny>) -> PyResult<Self> {
        let tokenizer = optional_attr(recipe.bind(py), "tokenizer")?
            .filter(|tokenizer| !tokenizer.is_none())
            .map(|tokenizer| {
                tokenizer
                    .cast::<RecipeTokenizer>()
                    .map(|tokenizer| tokenizer.borrow().native())
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "recipe.tokenizer must be resin.recipe.Tokenizer or None",
                        )
                    })
            })
            .transpose()?
            .flatten();
        let bytes = tokenizer
            .as_ref()
            .map(|tokenizer| TokenBytes::new(tokenizer));
        Ok(Self { tokenizer, bytes })
    }

    pub fn token(&self, token_id: u32, logprob: f32) -> TokenLogprob {
        let Some(tokenizer) = &self.tokenizer else {
            return TokenLogprob {
                token_id,
                token: String::new(),
                bytes: None,
                logprob,
            };
        };
        let spelling = tokenizer.id_to_token(token_id).unwrap_or_default();
        let bytes = self
            .bytes
            .as_ref()
            .and_then(|bytes| bytes.get(token_id, &spelling));
        let token = bytes
            .as_ref()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_else(|| tokenizer.decode(&[token_id], false).unwrap_or(spelling));
        TokenLogprob {
            token_id,
            token,
            bytes,
            logprob,
        }
    }

    pub fn generated(
        &self,
        token_id: u32,
        token_index: usize,
        channel: Option<&str>,
        scores: &resin_core::engine::TokenLogprobs,
    ) -> GeneratedTokenLogprob {
        GeneratedTokenLogprob {
            token: self.token(token_id, scores.logprob),
            token_index,
            channel: channel.map(str::to_owned),
            top_logprobs: scores
                .top_logprobs
                .iter()
                .map(|&(id, score)| self.token(id, score))
                .collect(),
        }
    }
}
