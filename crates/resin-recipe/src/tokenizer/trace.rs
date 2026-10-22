//! Exact byte ownership for the stock ByteLevel and ByteFallback decoder chains.
//! Native tokenizers still decides when text is stable; unsupported chains are untraced.
use super::bytes::{TokenBytes, fallback_byte};
use crate::{AttributedText, TokenSpan, Tokenizer};
use serde_json::Value;
use std::collections::VecDeque;

#[derive(Clone, Copy)]
enum Kind {
    ByteLevel,
    ByteFallback,
}
struct Piece {
    index: usize,
    bytes: Vec<u8>,
    start: usize,
    fallback: bool,
}

pub(super) struct Trace {
    kind: Option<Kind>,
    bytes: TokenBytes,
    unknown: bool,
    pending: VecDeque<Piece>,
}
impl Trace {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let bytes = TokenBytes::new(tokenizer);
        let kind = kind(&bytes.decoder);
        Self {
            kind,
            bytes,
            unknown: false,
            pending: VecDeque::new(),
        }
    }
    pub fn supported(&self) -> bool {
        self.kind.is_some()
    }
    pub fn clear(&mut self) {
        self.pending.clear();
        self.unknown = false;
    }
    pub fn mark_unknown(&mut self) {
        self.pending.clear();
        self.unknown = true;
    }
    pub fn push(&mut self, tokenizer: &Tokenizer, ids: &[u32], token_start: usize) {
        let Some(kind) = self.kind.filter(|_| !self.unknown) else {
            return;
        };
        for (offset, &id) in ids.iter().enumerate() {
            let spelling = tokenizer.id_to_token(id).unwrap_or_default();
            let fallback = !self.bytes.special.contains(&id)
                && matches!(kind, Kind::ByteFallback)
                && fallback_byte(&spelling).is_some();
            let bytes = self.bytes.get(id, &spelling).unwrap();
            self.pending.push_back(Piece {
                index: token_start + offset,
                bytes,
                start: 0,
                fallback,
            });
        }
    }
    pub fn output(&mut self, text: String) -> AttributedText {
        if text.is_empty() {
            return AttributedText::default();
        }
        let Some(kind) = self.kind.filter(|_| !self.unknown) else {
            return AttributedText::plain(text);
        };
        let pieces: Vec<_> = self.pending.iter().collect();
        let mut decoded = AttributedText::default();
        let mut offset = 0;
        while offset < pieces.len() {
            let end = match kind {
                Kind::ByteLevel => pieces.len(),
                Kind::ByteFallback if pieces[offset].fallback => {
                    offset
                        + pieces[offset..]
                            .iter()
                            .take_while(|piece| piece.fallback)
                            .count()
                }
                Kind::ByteFallback => offset + 1,
            };
            decode(
                &pieces[offset..end],
                matches!(kind, Kind::ByteFallback),
                &mut decoded,
            );
            offset = end;
        }
        // This is an equality assertion against the decoder's exact output prefix,
        // never a substring search or an inferred association by displayed text.
        if !decoded.text.starts_with(&text) {
            self.mark_unknown();
            return AttributedText::plain(text);
        }
        let result = decoded.slice(0..text.len());
        let consumed: usize = result
            .spans
            .iter()
            .map(|span| span.token_end - span.token_start)
            .sum();
        let mut left = consumed;
        while let Some(piece) = self.pending.front_mut() {
            let length = piece.bytes.len() - piece.start;
            if length > left {
                piece.start += left;
                break;
            }
            left -= length;
            self.pending.pop_front();
            if left == 0 {
                break;
            }
        }
        result
    }
}

fn kind(value: &Value) -> Option<Kind> {
    let children = if value["type"] == "Sequence" {
        value["decoders"].as_array()?.iter().collect::<Vec<_>>()
    } else {
        vec![value]
    };
    match children.as_slice() {
        [decoder] if decoder["type"] == "ByteLevel" => Some(Kind::ByteLevel),
        [decoder] if decoder["type"] == "ByteFallback" => Some(Kind::ByteFallback),
        [decoder, fuse] if decoder["type"] == "ByteFallback" && fuse["type"] == "Fuse" => {
            Some(Kind::ByteFallback)
        }
        [replace, decoder, fuse]
            if replace["type"] == "Replace"
                && replace["pattern"]["String"] == "▁"
                && replace["content"] == " "
                && decoder["type"] == "ByteFallback"
                && fuse["type"] == "Fuse" =>
        {
            Some(Kind::ByteFallback)
        }
        _ => None,
    }
}
fn decode(pieces: &[&Piece], fallback: bool, result: &mut AttributedText) {
    let bytes: Vec<_> = pieces
        .iter()
        .flat_map(|piece| piece.bytes[piece.start..].iter().copied())
        .collect();
    let invalid_fallback = fallback && std::str::from_utf8(&bytes).is_err();
    let mut source = 0;
    let mut piece_index = 0;
    let mut piece_offset = 0;
    while source < bytes.len() {
        let (valid_len, invalid_len) = if invalid_fallback {
            (0, 1)
        } else {
            match std::str::from_utf8(&bytes[source..]) {
                Ok(_) => (bytes.len() - source, 0),
                Err(error) => (
                    error.valid_up_to(),
                    error
                        .error_len()
                        .unwrap_or(bytes.len() - source - error.valid_up_to()),
                ),
            }
        };
        let valid = std::str::from_utf8(&bytes[source..source + valid_len]).unwrap();
        for (character, source_len) in valid
            .chars()
            .map(|character| (character, character.len_utf8()))
            .chain((invalid_len > 0).then_some(('�', invalid_len)))
        {
            let event_start = result.text.len();
            result.text.push(character);
            let end = source + source_len;
            while source < end {
                let piece = pieces[piece_index];
                let length = piece.bytes.len() - piece.start;
                if source >= piece_offset + length {
                    piece_offset += length;
                    piece_index += 1;
                    continue;
                }
                let fragment_end = end.min(piece_offset + length);
                let span = TokenSpan {
                    token_index: piece.index,
                    token_start: piece.start + source - piece_offset,
                    token_end: piece.start + fragment_end - piece_offset,
                    token_len: piece.bytes.len(),
                    event_start,
                    event_end: result.text.len(),
                };
                if let Some(previous) = result.spans.last_mut()
                    && previous.token_index == span.token_index
                    && previous.token_end == span.token_start
                    && previous.event_end == span.event_start
                    && previous.token_end - previous.token_start
                        == previous.event_end - previous.event_start
                    && span.token_end - span.token_start == span.event_end - span.event_start
                {
                    previous.token_end = span.token_end;
                    previous.event_end = span.event_end;
                } else {
                    result.spans.push(span);
                }
                source = fragment_end;
            }
        }
    }
}
