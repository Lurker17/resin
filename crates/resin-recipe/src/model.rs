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
        Ok(match self {
            Self::Inkling => Box::new(InklingParser::new(
                InklingTokens::resolve(encoder)?,
                decoder,
                prompt_ids,
            )?),
            Self::GptOss => Box::new(GptOssParser::new(
                GptOssTokens::resolve(encoder)?,
                decoder,
                prompt_ids,
            )?),
            Self::Qwen => Box::new(QwenParser::new(
                QwenTokens::resolve(encoder)?,
                decoder,
                prompt_ids,
                &request["tools"],
            )?),
            Self::Gemma => Box::new(Gemma4Parser::new(
                Gemma4Tokens::resolve(encoder)?,
                decoder,
                prompt_ids,
            )?),
            Self::Lfm2 => Box::new(Lfm2Parser::new(
                Lfm2Tokens::resolve(encoder)?,
                decoder,
                prompt_ids,
            )?),
            Self::DeepSeek => Box::new(DeepSeekV4Parser::new(
                DeepSeekV4Tokens::resolve(encoder)?,
                decoder,
                prompt_ids,
                crate::deepseek::thinking_settings(request)?.0,
            )?),
        })
    }
    pub fn stop_token_ids(self, encoder: &dyn Encoder) -> Result<Vec<u32>, Error> {
        let id = |spelling: &str| encoder.special_id(spelling);
        let alias = |names: &[&str]| crate::parser::resolve(encoder, names);
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
