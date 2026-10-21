//! A model format selects its matching renderer, parser, and stop markers.
use crate::{
    Decoder, DeepSeekV4Parser, DeepSeekV4Tokens, Encoder, Error, Gemma4Parser, Gemma4Tokens,
    GptOssParser, GptOssTokens, InklingParser, InklingTokens, Lfm2Parser, Lfm2Tokens, Prompt,
    QwenParser, QwenTokens, ResponseParser,
};
use serde_json::Value;

#[derive(Clone, Copy)]
pub enum ModelFormat {
    Inkling,
    GptOss,
    DeepSeek,
    Qwen,
    Gemma,
    Lfm2,
}
impl ModelFormat {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "inkling" => Ok(Self::Inkling),
            "gpt_oss" => Ok(Self::GptOss),
            "deepseek_v4" => Ok(Self::DeepSeek),
            "qwen3_5" => Ok(Self::Qwen),
            "gemma4" => Ok(Self::Gemma),
            "lfm2" => Ok(Self::Lfm2),
            _ => Err(Error::new(format!("unknown recipe model: {name}"))),
        }
    }
    pub fn render(self, request: &Value) -> Result<Prompt, Error> {
        match self {
            Self::Inkling => crate::inkling::render(request),
            Self::GptOss => crate::gpt_oss::render(request),
            Self::DeepSeek => crate::deepseek::render(request),
            Self::Qwen => crate::qwen::render(request),
            Self::Gemma => crate::gemma::render(request),
            Self::Lfm2 => crate::lfm2::render(request),
        }
    }
}

impl ModelFormat {
    pub fn create_parser(
        self,
        encoder: &dyn Encoder,
        decoder: Box<dyn Decoder>,
        request: &Value,
        prompt_ids: &[u32],
    ) -> Result<Box<dyn ResponseParser>, Error> {
        let id = |spelling: &str| encoder.special_id(spelling);
        let alias = |names: &[&str]| aliases(encoder, names);
        Ok(match self {
            Self::Inkling => Box::new(InklingParser::new(
                InklingTokens {
                    message_model: id("<|message_model|>")?,
                    content_text: id("<|content_text|>")?,
                    content_thinking: id("<|content_thinking|>")?,
                    content_invoke_tool_json: id("<|content_invoke_tool_json|>")?,
                    content_invoke_tool_text: id("<|content_invoke_tool_text|>")?,
                    content_tool_error: id("<|content_tool_error|>")?,
                    content_model_end_sampling: id("<|content_model_end_sampling|>")?,
                    end_message: id("<|end_message|>")?,
                },
                decoder,
                prompt_ids,
            )?),
            Self::GptOss => Box::new(GptOssParser::new(
                GptOssTokens {
                    start: alias(&["<|im_start|>", "<|start|>"])?,
                    end: alias(&["<|im_end|>", "<|end|>"])?,
                    turn_end: alias(&["<|fim_suffix|>", "<|return|>"])?,
                    tool_end: alias(&["<|ghissue|>", "<|call|>"])?,
                    channel: alias(&["<|meta_sep|>", "<|channel|>"])?,
                    message: alias(&["<|im_sep|>", "<|message|>"])?,
                    constrain: alias(&["<|meta_start|>", "<|constrain|>"])?,
                },
                decoder,
                prompt_ids,
            )?),
            Self::Qwen => Box::new(QwenParser::new(
                QwenTokens {
                    im_end: id("<|im_end|>")?,
                    endoftext: id("<|endoftext|>")?,
                    think: id("<think>")?,
                    think_end: id("</think>")?,
                    tool_call: id("<tool_call>")?,
                    tool_call_end: id("</tool_call>")?,
                },
                decoder,
                prompt_ids,
                &request["tools"],
            )?),
            Self::Gemma => Box::new(Gemma4Parser::new(
                Gemma4Tokens {
                    turn_start: id("<|turn>")?,
                    turn_end: id("<turn|>")?,
                    channel_start: id("<|channel>")?,
                    channel_end: id("<channel|>")?,
                    tool_call_start: id("<|tool_call>")?,
                    tool_call_end: id("<tool_call|>")?,
                    tool_response_start: id("<|tool_response>")?,
                    tool_response_end: id("<tool_response|>")?,
                    escape: id("<|\"|>")?,
                    eos: id("<eos>")?,
                },
                decoder,
                prompt_ids,
            )?),
            Self::Lfm2 => Box::new(Lfm2Parser::new(
                Lfm2Tokens {
                    message_start: id("<|im_start|>")?,
                    message_end: id("<|im_end|>")?,
                    thinking_start: id("<think>")?,
                    thinking_end: id("</think>")?,
                    tool_call_start: id("<|tool_call_start|>")?,
                    tool_call_end: id("<|tool_call_end|>")?,
                },
                decoder,
                prompt_ids,
            )?),
            Self::DeepSeek => Box::new(DeepSeekV4Parser::new(
                DeepSeekV4Tokens {
                    bos: id(crate::deepseek::BOS)?,
                    eos: id(crate::deepseek::EOS)?,
                    thinking_start: id("<think>")?,
                    thinking_end: id("</think>")?,
                    dsml: id("｜DSML｜")?,
                },
                decoder,
                prompt_ids,
                crate::deepseek::thinking_settings(request)?.0,
            )?),
        })
    }
    pub fn stop_token_ids(self, encoder: &dyn Encoder) -> Result<Vec<u32>, Error> {
        let id = |spelling: &str| encoder.special_id(spelling);
        let alias = |names: &[&str]| aliases(encoder, names);
        match self {
            Self::Inkling => Ok(vec![id("<|content_model_end_sampling|>")?]),
            Self::GptOss => Ok(vec![
                alias(&["<|fim_suffix|>", "<|return|>"])?,
                alias(&["<|ghissue|>", "<|call|>"])?,
            ]),
            Self::DeepSeek => Ok(vec![id(crate::deepseek::EOS)?]),
            Self::Lfm2 => Ok(vec![id("<|im_end|>")?]),
            Self::Qwen => Ok(vec![id("<|im_end|>")?, id("<|endoftext|>")?]),
            Self::Gemma => Ok(vec![id("<turn|>")?, id("<eos>")?]),
        }
    }
}

fn aliases(encoder: &dyn Encoder, names: &[&str]) -> Result<u32, Error> {
    names
        .iter()
        .find_map(|name| encoder.special_id(name).ok())
        .ok_or_else(|| {
            Error::new(format!(
                "missing model control token (aliases: {})",
                names.join(", ")
            ))
        })
}
