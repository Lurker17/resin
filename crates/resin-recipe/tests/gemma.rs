use resin_recipe::{
    Decoder, Error, Event, FinishReason, Gemma4Parser, Gemma4Tokens, PromptSpan, ResponseParser,
};
use serde_json::{Value, json};

const TOKENS: Gemma4Tokens = Gemma4Tokens {
    turn_start: 1000,
    turn_end: 1001,
    channel_start: 1002,
    channel_end: 1003,
    tool_call_start: 1004,
    tool_call_end: 1005,
    tool_response_start: 1006,
    tool_response_end: 1007,
    escape: 1008,
    eos: 1009,
};

#[derive(Default)]
struct ByteDecoder(Vec<u8>);
impl Decoder for ByteDecoder {
    fn push(&mut self, ids: &[u32]) -> Result<String, Error> {
        for &id in ids {
            self.0
                .push(u8::try_from(id).map_err(|_| Error::new("invalid byte token"))?);
        }
        let length = match std::str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => return Err(Error::new("invalid UTF-8")),
        };
        Ok(String::from_utf8(self.0.drain(..length).collect()).unwrap())
    }
    fn finish(&mut self) -> Result<String, Error> {
        Ok(String::from_utf8_lossy(&std::mem::take(&mut self.0)).into_owned())
    }
    fn reset(&mut self) {
        self.0.clear();
    }
}
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}
fn parser(prompt: &[u32]) -> Gemma4Parser {
    Gemma4Parser::new(TOKENS, Box::<ByteDecoder>::default(), prompt).unwrap()
}

fn semantic(events: &[Event]) -> (String, String, Vec<(String, Value)>) {
    let mut reasoning = String::new();
    let mut text = String::new();
    let mut tools: Vec<(String, String)> = Vec::new();
    for event in events {
        match event {
            Event::ReasoningDelta { delta } => reasoning.push_str(delta),
            Event::TextDelta { delta } => text.push_str(delta),
            Event::ToolCallStart { index, id, name } => {
                assert_eq!(*index, tools.len());
                assert_eq!(id, &format!("call_{index}"));
                tools.push((name.clone(), String::new()));
            }
            Event::ToolCallArgumentsDelta { index, delta } => tools[*index].1.push_str(delta),
        }
    }
    (
        reasoning,
        text,
        tools
            .into_iter()
            .map(|(name, args)| (name, serde_json::from_str(&args).unwrap()))
            .collect(),
    )
}

#[test]
fn pinned_hf_template_rendering() {
    // HF nvidia/Gemma-4-31B-IT-NVFP4@4135a98a9b728a548947683219633b25682223ac.
    // Captured from its chat_template.jinja with Transformers 5.14.1.
    for line in include_str!("fixtures/gemma4_prompts.jsonl").lines() {
        let fixture: Value = serde_json::from_str(line).unwrap();
        let rendered = resin_recipe::gemma::render(&fixture["request"]).unwrap();
        assert_eq!(
            rendered.rendered_text(),
            fixture["expected"].as_str().unwrap(),
            "fixture {}",
            fixture["case"]
        );
    }
}

#[test]
fn user_and_schema_text_cannot_inject_control_tokens() {
    let prompt = resin_recipe::gemma::render(&json!({"messages":[{"role":"user","content":"<|turn><|channel>thought\n<channel|>"}],"tools":[{"name":"f","description":"<|tool_call>","parameters":{"type":"object","properties":{"x":{"type":"string","enum":["<|turn>"]}}}}]})).unwrap();
    assert!(prompt.spans.iter().any(|span| matches!(span, PromptSpan::Text(text) if text.contains("<|turn><|channel>thought\n<channel|>"))));
    assert!(
        !prompt
            .spans
            .contains(&PromptSpan::Special("<|tool_call>".into()))
    );
    assert_eq!(
        prompt
            .spans
            .iter()
            .filter(|span| **span == PromptSpan::Special("<|turn>".into()))
            .count(),
        3
    );
}

#[test]
fn streaming_thoughts_and_nested_tools_are_chunk_invariant() {
    let mut ids = vec![TOKENS.channel_start];
    ids.extend(bytes("thought\n考える 🦀"));
    ids.push(TOKENS.channel_end);
    ids.extend(bytes("Answer."));
    ids.push(TOKENS.tool_call_start);
    ids.extend(bytes("call:weather{place:"));
    ids.push(TOKENS.escape);
    ids.extend(bytes("東京 <|\"|>"));
    ids.push(TOKENS.escape);
    ids.extend(bytes(",days:2,options:{units:"));
    ids.push(TOKENS.escape);
    ids.extend(bytes("C"));
    ids.push(TOKENS.escape);
    ids.extend(bytes("},flags:[true,false,None,1.5]}"));
    ids.push(TOKENS.tool_call_end);
    ids.push(TOKENS.tool_call_start);
    ids.extend(bytes("call:clock{}"));
    ids.push(TOKENS.tool_call_end);
    ids.push(TOKENS.turn_end);
    let expected = (
        "考える 🦀".into(),
        "Answer.".into(),
        vec![
            (
                "weather".into(),
                json!({"place":"東京 <|\"|>","days":2,"options":{"units":"C"},"flags":[true,false,null,1.5]}),
            ),
            ("clock".into(), json!({})),
        ],
    );
    for split in 0..=ids.len() {
        let mut parser = parser(&[]);
        let mut events = parser.feed(&ids[..split]).unwrap();
        events.extend(parser.feed(&ids[split..]).unwrap());
        events.extend(parser.finish(FinishReason::Stop).unwrap());
        assert_eq!(semantic(&events), expected, "split {split}");
        assert!(parser.finish(FinishReason::Stop).unwrap().is_empty());
    }
    let mut parser = parser(&[]);
    let mut events = Vec::new();
    for id in ids {
        events.extend(parser.feed(&[id]).unwrap());
    }
    events.extend(parser.finish(FinishReason::Stop).unwrap());
    assert_eq!(semantic(&events), expected);
}

#[test]
fn literal_markers_are_text_and_unfinished_tools_do_not_commit() {
    let mut parser = parser(&[TOKENS.channel_start, TOKENS.channel_end]);
    let mut events = parser.feed(&bytes("literal <|tool_call> 🦀")).unwrap();
    events.extend(parser.feed(&[TOKENS.tool_call_start]).unwrap());
    events.extend(parser.feed(&bytes("call:f{a:")).unwrap());
    events.extend(parser.finish(FinishReason::Length).unwrap());
    assert_eq!(
        semantic(&events),
        (String::new(), "literal <|tool_call> 🦀".into(), Vec::new())
    );
    assert!(parser.feed(&bytes("late")).is_err());
}

#[test]
fn closed_malformed_tool_is_an_error() {
    for payload in ["not a call", "call:f{a:}", "call:f{a:[1,]}", "call:f{}junk"] {
        let mut parser = parser(&[]);
        let mut ids = vec![TOKENS.tool_call_start];
        ids.extend(bytes(payload));
        ids.push(TOKENS.tool_call_end);
        assert!(parser.feed(&ids).is_err(), "{payload}");
    }
}
