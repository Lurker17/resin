use resin_recipe::{
    AttributedEvent, Decoder, Event, FinishReason, ModelFormat, ResponseParser, Tokenizer,
    TokenizerDecoder, TokenizerEncoder,
};
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, sync::Arc};

const CONTROLS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|fim_suffix|>",
    "<|ghissue|>",
    "<|meta_sep|>",
    "<|im_sep|>",
    "<|meta_start|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
    "<|content_model_end_sampling|>",
    "<|end_message|>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
    "<eos>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
];
fn byte_character(byte: u8) -> char {
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
        return char::from(byte);
    }
    let offset = (0..byte)
        .filter(|b| !(33..=126).contains(b) && !(161..=172).contains(b) && !(174..=255).contains(b))
        .count();
    char::from_u32(256 + offset as u32).unwrap()
}
fn tokenizer(fallback: bool, extra: &[&str]) -> Arc<Tokenizer> {
    let mut vocab = Map::new();
    for byte in 0..=255_u8 {
        vocab.insert(
            if fallback {
                format!("<0x{byte:02X}>")
            } else {
                byte_character(byte).to_string()
            },
            json!(byte),
        );
    }
    for (index, text) in CONTROLS.iter().chain(extra).enumerate() {
        vocab.insert((*text).to_owned(), json!(256 + index));
    }
    let decoder = if fallback {
        json!({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})
    } else {
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
    };
    Arc::new(Tokenizer::from_bytes(serde_json::to_vec(&json!({"version":"1.0","added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":decoder,"model":{"type":"BPE","vocab":vocab,"merges":[]}})).unwrap()).unwrap())
}
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}
fn id(tokenizer: &Tokenizer, text: &str) -> u32 {
    tokenizer.token_to_id(text).unwrap()
}
fn parser(
    model: ModelFormat,
    tokenizer: Arc<Tokenizer>,
    prompt: &[u32],
) -> Box<dyn ResponseParser> {
    model
        .create_parser(
            &TokenizerEncoder::new(tokenizer.clone()).unwrap(),
            Box::new(TokenizerDecoder::new(tokenizer)),
            &json!({}),
            prompt,
        )
        .unwrap()
}

type Ownership = BTreeMap<usize, Vec<(usize, usize, usize, usize, usize)>>;
fn normalize(events: Vec<AttributedEvent>) -> (String, String, Ownership) {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut owners = BTreeMap::<_, Vec<_>>::new();
    for event in events {
        let target = match event.event {
            Event::TextDelta { delta } => {
                content.push_str(&delta);
                (&content, delta.len(), 0)
            }
            Event::ReasoningDelta { delta } => {
                reasoning.push_str(&delta);
                (&reasoning, delta.len(), 1)
            }
            _ => {
                assert!(event.spans.is_empty());
                continue;
            }
        };
        let base = target.0.len() - target.1;
        for span in event.spans {
            assert!(span.event_end <= target.1);
            assert!(span.token_end <= span.token_len);
            owners.entry(span.token_index).or_default().push((
                span.token_start,
                span.token_end,
                base + span.event_start,
                base + span.event_end,
                target.2,
            ));
        }
    }
    (content, reasoning, owners)
}

#[test]
fn all_model_grammars_preserve_unicode_and_source_indices_across_batching() {
    for model in [
        ModelFormat::GptOss,
        ModelFormat::Inkling,
        ModelFormat::DeepSeek,
        ModelFormat::Qwen,
        ModelFormat::Gemma,
        ModelFormat::Lfm2,
    ] {
        for fallback in [false, true] {
            let tokenizer = tokenizer(fallback, &[]);
            let (prompt, mut generated) = match model {
                ModelFormat::GptOss => {
                    let mut prompt = vec![id(&tokenizer, "<|im_start|>")];
                    prompt.extend(bytes("assistant"));
                    prompt.push(id(&tokenizer, "<|meta_sep|>"));
                    prompt.extend(bytes("analysis"));
                    prompt.push(id(&tokenizer, "<|im_sep|>"));
                    (prompt, bytes("hi 🦀!"))
                }
                ModelFormat::Inkling => (
                    vec![id(&tokenizer, "<|content_thinking|>")],
                    bytes("hi 🦀!"),
                ),
                ModelFormat::DeepSeek | ModelFormat::Qwen => {
                    (vec![id(&tokenizer, "<think>")], bytes("hi 🦀!"))
                }
                ModelFormat::Lfm2 => {
                    let mut generated = vec![id(&tokenizer, "<think>")];
                    generated.extend(bytes("hi 🦀!"));
                    (vec![], generated)
                }
                ModelFormat::Gemma => (vec![id(&tokenizer, "<|channel>")], bytes("hi 🦀!")),
            };
            match model {
                ModelFormat::GptOss => {
                    generated.push(id(&tokenizer, "<|im_end|>"));
                    generated.push(id(&tokenizer, "<|im_start|>"));
                    generated.extend(bytes("assistant"));
                    generated.push(id(&tokenizer, "<|im_sep|>"));
                }
                ModelFormat::Inkling => generated.push(id(&tokenizer, "<|content_text|>")),
                ModelFormat::DeepSeek | ModelFormat::Qwen | ModelFormat::Lfm2 => {
                    generated.push(id(&tokenizer, "</think>"))
                }
                ModelFormat::Gemma => generated.push(id(&tokenizer, "<channel|>")),
            }
            generated.extend(bytes("ok 🧠!"));
            let mut expected = None;
            for chunk_size in [1, 2, 3, 7, generated.len()] {
                let mut decoder = parser(model, tokenizer.clone(), &prompt);
                assert!(decoder.supports_attribution());
                let mut events = Vec::new();
                for chunk in generated.chunks(chunk_size) {
                    assert!(decoder.supports_attribution());
                    events.extend(decoder.feed_attributed(chunk).unwrap());
                }
                events.extend(decoder.finish_attributed(FinishReason::Length).unwrap());
                let actual = normalize(events);
                assert_eq!(
                    (&actual.0, &actual.1),
                    (&"ok 🧠!".to_owned(), &"hi 🦀!".to_owned())
                );
                assert_eq!(actual.2.len(), "hi 🦀!ok 🧠!".len());
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected);
                } else {
                    expected = Some(actual);
                }
            }
        }
    }
}

#[test]
fn qwen_trim_retains_delayed_internal_whitespace_and_marks_partial_tokens() {
    let tokenizer = tokenizer(false, &["ĠhelloĠ", "worldĠ"]);
    let mut parser = parser(ModelFormat::Qwen, tokenizer.clone(), &[]);
    let first = id(&tokenizer, "ĠhelloĠ");
    let second = id(&tokenizer, "worldĠ");
    let mut events = parser.feed_attributed(&[first]).unwrap();
    events.extend(parser.feed_attributed(&[second]).unwrap());
    events.extend(parser.finish_attributed(FinishReason::Length).unwrap());
    let (text, _, owners) = normalize(events);
    assert_eq!(text, "hello world");
    assert_eq!(owners[&0], vec![(1, 6, 0, 5, 0), (6, 7, 5, 6, 0)]);
    assert_eq!(owners[&1], vec![(0, 5, 6, 11, 0)]);
}

#[test]
fn gemma_header_body_inside_one_token_and_deepseek_marker_prefix_are_attributed() {
    let tokenizer = tokenizer(false, &["analysisĊbody", "aĊĊ<", "x"]);
    let mut gemma = parser(ModelFormat::Gemma, tokenizer.clone(), &[]);
    let events = gemma
        .feed_attributed(&[
            id(&tokenizer, "<|channel>"),
            id(&tokenizer, "analysisĊbody"),
        ])
        .unwrap();
    let (_, reasoning, owners) = normalize(events);
    assert_eq!(reasoning, "body");
    assert_eq!(owners[&1], vec![(9, 13, 0, 4, 1)]);
    let mut deepseek = parser(ModelFormat::DeepSeek, tokenizer.clone(), &[]);
    let mut events = deepseek.feed_attributed(&[id(&tokenizer, "aĊĊ<")]).unwrap();
    events.extend(deepseek.feed_attributed(&[id(&tokenizer, "x")]).unwrap());
    events.extend(deepseek.finish_attributed(FinishReason::Length).unwrap());
    let (text, _, owners) = normalize(events);
    assert_eq!(text, "a\n\n<x");
    assert_eq!(owners[&0], vec![(0, 1, 0, 1, 0), (1, 4, 1, 4, 0)]);
}

#[test]
fn incomplete_utf8_flush_owns_replacements_and_custom_decoders_are_explicitly_unknown() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[]);
        let mut decoder = TokenizerDecoder::new(tokenizer);
        assert!(
            decoder
                .push_attributed(&[0xF0, 0x9F], 7)
                .unwrap()
                .is_empty()
        );
        let tail = decoder.finish_attributed().unwrap();
        assert_eq!(tail.text, if fallback { "��" } else { "�" });
        assert_eq!(
            tail.spans.iter().map(|s| s.token_index).collect::<Vec<_>>(),
            vec![7, 8]
        );
        assert!(
            tail.spans
                .iter()
                .all(|s| s.token_start == 0 && s.token_end == 1 && s.token_len == 1)
        );
    }
    struct Plain;
    impl Decoder for Plain {
        fn push(&mut self, _: &[u32]) -> Result<String, resin_recipe::Error> {
            Ok("hello".into())
        }
        fn finish(&mut self) -> Result<String, resin_recipe::Error> {
            Ok(String::new())
        }
        fn reset(&mut self) {}
    }
    let mut plain = Plain;
    assert!(!plain.supports_attribution());
    assert!(plain.push_attributed(&[1], 0).unwrap().spans.is_empty());
}

#[test]
fn unscored_feed_never_calls_attributed_decoder() {
    struct Unscored;
    impl Decoder for Unscored {
        fn push(&mut self, _: &[u32]) -> Result<String, resin_recipe::Error> {
            Ok("x".into())
        }
        fn finish(&mut self) -> Result<String, resin_recipe::Error> {
            Ok(String::new())
        }
        fn reset(&mut self) {}
        fn push_attributed(
            &mut self,
            _: &[u32],
            _: usize,
        ) -> Result<resin_recipe::AttributedText, resin_recipe::Error> {
            panic!("unscored path traced")
        }
    }
    let tokenizer = tokenizer(false, &[]);
    let mut parser = ModelFormat::Qwen
        .create_parser(
            &TokenizerEncoder::new(tokenizer).unwrap(),
            Box::new(Unscored),
            &Value::Null,
            &[],
        )
        .unwrap();
    assert!(!parser.supports_attribution());
    assert_eq!(
        parser.feed(&[1]).unwrap(),
        vec![Event::TextDelta { delta: "x".into() }]
    );
    parser.finish(FinishReason::Length).unwrap();
}

#[test]
fn transformed_tools_have_no_fabricated_provenance_and_serde_keeps_flat_events() {
    let tokenizer = tokenizer(false, &[]);
    for (model, open, body, close) in [
        (
            ModelFormat::Inkling,
            "<|content_invoke_tool_json|>",
            r#"{"name":"f","args":{"a":1}}"#,
            "<|content_text|>",
        ),
        (
            ModelFormat::Qwen,
            "<tool_call>",
            "<function=f><parameter=a>1</parameter></function>",
            "</tool_call>",
        ),
        (
            ModelFormat::Gemma,
            "<|tool_call>",
            "call:f{a:1}",
            "<tool_call|>",
        ),
        (
            ModelFormat::Lfm2,
            "<|tool_call_start|>",
            "[f(a=1)]",
            "<|tool_call_end|>",
        ),
    ] {
        let mut parser = parser(model, tokenizer.clone(), &[]);
        let mut ids = vec![id(&tokenizer, open)];
        ids.extend(bytes(body));
        ids.push(id(&tokenizer, close));
        let events = parser.feed_attributed(&ids).unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.event, Event::ToolCallStart { .. }))
        );
        assert!(events.iter().all(|event| event.spans.is_empty()));
    }
    let mut parser = parser(ModelFormat::Qwen, tokenizer, &[]);
    let event = parser.feed_attributed(&bytes("x")).unwrap().pop().unwrap();
    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["type"], "text_delta");
    assert_eq!(value["delta"], "x");
    assert_eq!(value["token_spans"][0]["token_index"], 0);
    assert_eq!(
        serde_json::from_value::<AttributedEvent>(value).unwrap(),
        event
    );
    let plain: AttributedEvent = Event::TextDelta { delta: "x".into() }.into();
    assert_eq!(
        serde_json::to_value(plain).unwrap(),
        json!({"type":"text_delta","delta":"x"})
    );
}

#[test]
fn source_bytes_share_decoder_conventions_and_unknown_decoders_stay_unknown() {
    use resin_recipe::TokenBytes;
    let tokenizer = tokenizer(true, &["▁hello", "<0x0000>", "<0xGG>"]);
    let bytes = TokenBytes::new(&tokenizer);
    for (spelling, expected) in [
        ("▁hello", b" hello".as_slice()),
        ("<0x0000>", b"<0x0000>"),
        ("<0xGG>", b"<0xGG>"),
    ] {
        assert_eq!(
            bytes.get(id(&tokenizer, spelling), spelling).unwrap(),
            expected
        );
    }
    let tokenizer = Tokenizer::from_bytes(br#"{"version":"1.0","added_tokens":[],"decoder":{"type":"Fuse"},"model":{"type":"BPE","vocab":{"hello":0},"merges":[]}}"#).unwrap();
    let mut decoder = TokenizerDecoder::new(Arc::new(tokenizer));
    assert!(!decoder.supports_attribution());
    let text = decoder.push_attributed(&[0], 0).unwrap();
    assert_eq!(text.text, "hello");
    assert!(text.spans.is_empty());
}

#[test]
fn mixed_mode_utf8_is_unknown_until_the_next_control_boundary() {
    let tokenizer = tokenizer(false, &[]);
    let mut parser = parser(ModelFormat::Qwen, tokenizer.clone(), &[]);
    assert!(parser.feed(&[0xF0, 0x9F]).unwrap().is_empty());
    let mixed = parser.feed_attributed(&[0xA6, 0x80]).unwrap();
    assert_eq!(
        mixed[0].event,
        Event::TextDelta {
            delta: "🦀".into()
        }
    );
    assert!(mixed[0].spans.is_empty());
    let later = parser.feed_attributed(&bytes("old segment")).unwrap();
    assert!(later.iter().all(|event| event.spans.is_empty()));
    let mut ids = vec![id(&tokenizer, "<think>")];
    ids.extend(bytes("new"));
    let recovered = parser.feed_attributed(&ids).unwrap();
    assert!(recovered.iter().any(|event| !event.spans.is_empty()));
    assert!(recovered[0].spans.iter().all(|span| span.token_index >= 16));

    let mut parser = crate::parser(ModelFormat::Qwen, tokenizer.clone(), &[]);
    assert!(parser.feed_attributed(&[0xF0, 0x9F]).unwrap().is_empty());
    assert_eq!(
        parser.feed(&[0xA6, 0x80]).unwrap(),
        vec![Event::TextDelta {
            delta: "🦀".into()
        }]
    );
    assert!(
        parser
            .feed_attributed(&bytes("🦀"))
            .unwrap()
            .iter()
            .all(|event| event.spans.is_empty())
    );
    parser
        .feed_attributed(&[id(&tokenizer, "<think>")])
        .unwrap();
    assert!(
        !parser.feed_attributed(&bytes("x")).unwrap()[0]
            .spans
            .is_empty()
    );
}

#[test]
fn qwen_and_deepseek_tool_channel_hints_exclude_deferred_content() {
    let tokenizer = tokenizer(false, &[]);
    let mut qwen = parser(ModelFormat::Qwen, tokenizer.clone(), &[]);
    assert_eq!(qwen.token_channel(u32::from(b'x')), None);
    let open = id(&tokenizer, "<tool_call>");
    assert_eq!(qwen.token_channel(open), Some("control"));
    qwen.feed_attributed(&[open]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b'x')), Some("tool"));
    qwen.feed_attributed(&bytes("<function=f></function>"))
        .unwrap();
    qwen.feed_attributed(&[id(&tokenizer, "</tool_call>")])
        .unwrap();
    assert_eq!(qwen.token_channel(u32::from(b' ')), Some("tool"));
    qwen.feed_attributed(&[id(&tokenizer, "<think>")]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b'x')), None);

    let mut deepseek = parser(ModelFormat::DeepSeek, tokenizer.clone(), &[]);
    deepseek.feed_attributed(&bytes("\n\n<")).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'x')), None);
    deepseek
        .feed_attributed(&[id(&tokenizer, "｜DSML｜")])
        .unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'x')), None);
    deepseek.feed_attributed(&bytes("tool_calls>")).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'<')), Some("tool"));
}
