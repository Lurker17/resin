//! Typed media expansions and token placement, independent of model preprocessing.

use std::{collections::HashMap, sync::Arc};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};
use resin_recipe::{Encoder, Error, Prompt, PromptSpan};

/// Explicit media IDs replacing one trusted template placeholder.
///
/// Prefix and suffix are ordinary text and share BPE boundaries with nearby text.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct MediaTokens {
    #[pyo3(get)]
    placeholder: String,
    token_ids: Arc<[u32]>,
    #[pyo3(get)]
    prefix: String,
    #[pyo3(get)]
    suffix: String,
}
impl MediaTokens {
    pub(crate) fn from_parts(
        placeholder: &str,
        token_ids: Vec<u32>,
        prefix: String,
        suffix: String,
    ) -> Self {
        Self {
            placeholder: placeholder.into(),
            token_ids: token_ids.into(),
            prefix,
            suffix,
        }
    }
}
#[pymethods]
impl MediaTokens {
    #[new]
    #[pyo3(signature=(placeholder, token_ids, *, prefix=String::new(), suffix=String::new()))]
    fn new(
        placeholder: String,
        token_ids: Vec<u32>,
        prefix: String,
        suffix: String,
    ) -> PyResult<Self> {
        if placeholder.is_empty() || token_ids.is_empty() {
            return Err(PyValueError::new_err(
                "media expansions need a placeholder and token IDs",
            ));
        }
        Ok(Self {
            placeholder,
            token_ids: token_ids.into(),
            prefix,
            suffix,
        })
    }
    #[getter]
    fn token_ids(&self) -> Vec<u32> {
        self.token_ids.to_vec()
    }
}

/// Model-processed media expansions and an opaque payload retained until binding.
#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct ProcessedMedia {
    pub(crate) spans: Vec<MediaTokens>,
    #[pyo3(get)]
    payload: Py<PyAny>,
}
#[pymethods]
impl ProcessedMedia {
    #[new]
    fn new(py: Python<'_>, spans: Vec<Py<MediaTokens>>, payload: Py<PyAny>) -> Self {
        Self {
            spans: spans.iter().map(|s| s.borrow(py).clone()).collect(),
            payload,
        }
    }
    #[getter]
    fn spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let spans = self
            .spans
            .iter()
            .cloned()
            .map(|s| Py::new(py, s))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, spans)?.unbind())
    }
}

enum EncodingSpan {
    Text(String),
    Special(String),
    Media(usize),
}
fn text(spans: &mut Vec<EncodingSpan>, value: &str) {
    if value.is_empty() {
        return;
    }
    if let Some(EncodingSpan::Text(previous)) = spans.last_mut() {
        previous.push_str(value);
    } else {
        spans.push(EncodingSpan::Text(value.into()));
    }
}

type EncodedMedia = (Vec<u32>, Vec<(usize, usize)>);

/// Encode trusted media slots and return their absolute (position, length) in input order.
/// Literal marker spellings in ordinary message text never become media slots.
pub(crate) fn encode(
    prompt: Prompt,
    media: &[MediaTokens],
    encoder: &dyn Encoder,
) -> Result<EncodedMedia, Error> {
    if media.is_empty() {
        return Ok((prompt.encode(encoder)?, Vec::new()));
    }
    let mut replacements: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, span) in media.iter().enumerate() {
        replacements
            .entry(&span.placeholder)
            .or_default()
            .push(index);
    }
    let mut used: HashMap<&str, usize> = HashMap::new();
    let mut spans = Vec::new();
    for span in prompt.spans {
        match span {
            PromptSpan::Text(value) => text(&mut spans, &value),
            PromptSpan::Special(token) => {
                if let Some((marker, indices)) = replacements.get_key_value(token.as_str()) {
                    let count = used.entry(marker).or_default();
                    let index = *indices.get(*count).ok_or_else(|| {
                        Error::new(format!("missing media expansion for {token}"))
                    })?;
                    *count += 1;
                    let replacement = &media[index];
                    text(&mut spans, &replacement.prefix);
                    spans.push(EncodingSpan::Media(index));
                    text(&mut spans, &replacement.suffix);
                } else {
                    spans.push(EncodingSpan::Special(token));
                }
            }
        }
    }
    for (marker, indices) in replacements {
        if used.get(marker).copied().unwrap_or(0) != indices.len() {
            return Err(Error::new(format!(
                "media expansion count does not match {marker}"
            )));
        }
    }
    let mut ids = Vec::new();
    let mut placements = vec![(0, 0); media.len()];
    for span in spans {
        match span {
            EncodingSpan::Text(value) => ids.extend(encoder.encode_text(&value)?),
            EncodingSpan::Special(token) => ids.push(encoder.special_id(&token)?),
            EncodingSpan::Media(index) => {
                let tokens = &media[index].token_ids;
                placements[index] = (ids.len(), tokens.len());
                ids.extend_from_slice(tokens);
            }
        }
    }
    Ok((ids, placements))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TextEncoder;
    impl Encoder for TextEncoder {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(match text {
                "left 00:00 " => vec![7],
                "tail right<image>" => vec![8],
                _ => return Err(Error::new(format!("unexpected text boundary: {text:?}"))),
            })
        }
        fn special_id(&self, _: &str) -> Result<u32, Error> {
            Ok(9)
        }
    }
    #[test]
    fn media_tokens_preserve_text_boundaries_and_literal_markers() {
        let mut prompt = Prompt::new();
        prompt
            .text("left ")
            .special("<image>")
            .text(" right<image>");
        let media = [MediaTokens::new(
            "<image>".into(),
            vec![100, 101],
            "00:00 ".into(),
            "tail".into(),
        )
        .unwrap()];
        let (ids, positions) = encode(prompt, &media, &TextEncoder).unwrap();
        assert_eq!(ids, [7, 100, 101, 8]);
        assert_eq!(positions, [(1, 2)]);
    }
    #[test]
    fn media_positions_follow_input_order_and_counts_must_match() {
        let mut prompt = Prompt::new();
        prompt
            .special("<video>")
            .special("<image>")
            .special("<image>");
        let media = [
            MediaTokens::new("<image>".into(), vec![10], String::new(), String::new()).unwrap(),
            MediaTokens::new("<video>".into(), vec![20, 21], String::new(), String::new()).unwrap(),
            MediaTokens::new("<image>".into(), vec![30], String::new(), String::new()).unwrap(),
        ];
        let (ids, positions) = encode(prompt.clone(), &media, &TextEncoder).unwrap();
        assert_eq!(ids, [20, 21, 10, 30]);
        assert_eq!(positions, [(2, 1), (0, 2), (3, 1)]);
        assert!(encode(prompt, &media[..2], &TextEncoder).is_err());
        assert!(encode(Prompt::new(), &media, &TextEncoder).is_err());
    }
}
