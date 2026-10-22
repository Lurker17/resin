//! Shared model-token byte representation for parser provenance and wire logprobs.
use crate::Tokenizer;
use serde_json::Value;
use std::collections::HashSet;

/// Prepared byte interpretation, independent of streaming decoder state.
/// Unknown decoder conventions return `None`, never guessed UTF-8 bytes.
pub struct TokenBytes {
    pub(super) decoder: Value,
    pub(super) special: HashSet<u32>,
    byte_level: bool,
    byte_fallback: bool,
    metaspace: bool,
}
impl TokenBytes {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let decoder = tokenizer
            .get_decoder()
            .and_then(|decoder| serde_json::to_value(decoder).ok())
            .unwrap_or_default();
        Self {
            byte_level: has_decoder(&decoder, "ByteLevel"),
            byte_fallback: has_decoder(&decoder, "ByteFallback"),
            metaspace: has_decoder(&decoder, "Metaspace") || replaces_metaspace(&decoder),
            decoder,
            special: tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .filter_map(|(id, token)| token.special.then_some(id))
                .collect(),
        }
    }
    pub fn get(&self, token_id: u32, spelling: &str) -> Option<Vec<u8>> {
        if self.special.contains(&token_id) {
            Some(spelling.as_bytes().to_vec())
        } else if self.byte_level {
            // Added tokens outside the byte alphabet decode as literal UTF-8.
            Some(
                spelling
                    .chars()
                    .map(byte_level_byte)
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_else(|| spelling.as_bytes().to_vec()),
            )
        } else if self.byte_fallback {
            if let Some(byte) = fallback_byte(spelling) {
                Some(vec![byte])
            } else {
                Some(if self.metaspace {
                    spelling.replace('▁', " ").into_bytes()
                } else {
                    spelling.as_bytes().to_vec()
                })
            }
        } else {
            None
        }
    }
}
pub(super) fn fallback_byte(text: &str) -> Option<u8> {
    (text.len() == 6 && text.starts_with("<0x") && text.ends_with('>'))
        .then(|| u8::from_str_radix(&text[3..5], 16).ok())
        .flatten()
}
fn has_decoder(value: &Value, kind: &str) -> bool {
    value["type"] == kind
        || value["decoders"]
            .as_array()
            .is_some_and(|children| children.iter().any(|child| has_decoder(child, kind)))
}
fn replaces_metaspace(value: &Value) -> bool {
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
