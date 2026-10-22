//! Token representations prepared once, independently of semantic parsing.
use pyo3::{exceptions::PyAttributeError, prelude::*};
use resin_http::{GeneratedTokenLogprob, TokenLogprob};
use resin_recipe::Tokenizer;
use std::collections::HashSet;

pub(crate) struct Vocabulary {
    tokenizer: Option<Tokenizer>,
    byte_level: bool,
    byte_fallback: bool,
    metaspace: bool,
    special_ids: HashSet<u32>,
}

impl Vocabulary {
    pub fn from_codec(py: Python<'_>, codec: &Py<PyAny>) -> PyResult<Self> {
        let tokenizer = match codec.bind(py).getattr("tokenizer") {
            Ok(tokenizer) => {
                let backend = tokenizer.getattr("backend_tokenizer").unwrap_or(tokenizer);
                match backend.getattr("to_str") {
                    Ok(serialize) => {
                        let json: String = serialize.call0()?.extract()?;
                        Some(Tokenizer::from_bytes(json.as_bytes()).map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(error.to_string())
                        })?)
                    }
                    Err(error) if error.is_instance_of::<PyAttributeError>(py) => None,
                    Err(error) => return Err(error),
                }
            }
            Err(error) if error.is_instance_of::<PyAttributeError>(py) => None,
            Err(error) => return Err(error),
        };
        let decoder = tokenizer
            .as_ref()
            .and_then(|tokenizer| tokenizer.get_decoder())
            .and_then(|decoder| serde_json::to_value(decoder).ok());
        let decoder = decoder.unwrap_or_default();
        let special_ids = tokenizer
            .as_ref()
            .map(|tokenizer| {
                tokenizer
                    .get_added_tokens_decoder()
                    .into_iter()
                    .filter_map(|(id, token)| token.special.then_some(id))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            tokenizer,
            special_ids,
            byte_level: has_decoder(&decoder, "ByteLevel"),
            byte_fallback: has_decoder(&decoder, "ByteFallback"),
            metaspace: has_decoder(&decoder, "Metaspace") || replaces_metaspace(&decoder),
        })
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
        let special = self.special_ids.contains(&token_id);
        let bytes = if special {
            Some(spelling.as_bytes().to_vec())
        } else if self.byte_level {
            // ByteLevel falls back to literal UTF-8 for an added token that
            // contains characters outside its byte alphabet.
            Some(
                spelling
                    .chars()
                    .map(byte_level_byte)
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_else(|| spelling.as_bytes().to_vec()),
            )
        } else if self.byte_fallback
            && spelling.len() == 6
            && spelling.starts_with("<0x")
            && spelling.ends_with('>')
        {
            u8::from_str_radix(&spelling[3..spelling.len() - 1], 16)
                .ok()
                .map(|byte| vec![byte])
        } else if self.byte_fallback {
            Some(if self.metaspace {
                spelling.replace('▁', " ").into_bytes()
            } else {
                spelling.as_bytes().to_vec()
            })
        } else {
            None
        };
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

fn has_decoder(value: &serde_json::Value, kind: &str) -> bool {
    value["type"] == kind
        || value["decoders"]
            .as_array()
            .is_some_and(|children| children.iter().any(|child| has_decoder(child, kind)))
}

fn replaces_metaspace(value: &serde_json::Value) -> bool {
    (value["type"] == "Replace" && value["pattern"]["String"] == "▁" && value["content"] == " ")
        || value["decoders"]
            .as_array()
            .is_some_and(|children| children.iter().any(replaces_metaspace))
}

fn byte_level_byte(character: char) -> Option<u8> {
    let code = character as u32;
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return Some(code as u8);
    }
    let mut encoded = 256;
    for byte in 0..=255_u8 {
        if !(33..=126).contains(&byte)
            && !(161..=172).contains(&byte)
            && !(174..=255).contains(&byte)
        {
            if code == encoded {
                return Some(byte);
            }
            encoded += 1;
        }
    }
    None
}
