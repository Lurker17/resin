//! The conversation format shipped with DeepSeek V4 Flash-0731 and Pro-0813.
//!
//! Those two pinned checkpoints have identical encoders. Their history format
//! differs from the later V4/V4.1 encoders in `deepseek-recipe`; keep the version
//! boundary explicit instead of silently changing prompts with an oracle update.

mod parser;

use serde_json::Value;

use crate::json::spaced as json_spaced;

use crate::{Error, Prompt};

pub use parser::{DeepSeekV4Parser, DeepSeekV4Tokens};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START: &str = "<think>";
pub const THINKING_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const REMINDER: &str = "<｜latest_reminder｜>";

const HIGH_EFFORT: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const MAX_EFFORT: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

/// Resolve Resin's API effort aliases without changing the checkpoint default.
pub fn thinking_settings(request: &Value) -> Result<(bool, &'static str), Error> {
    let extra = &request["extra"];
    let configured_effort = request
        .get("reasoning_effort")
        .filter(|value| !value.is_null())
        .or_else(|| {
            extra
                .get("reasoning_effort")
                .filter(|value| !value.is_null())
        });
    let effort = match configured_effort {
        None => "low",
        Some(Value::String(value)) => match value.as_str() {
            "minimal" | "low" => "low",
            "medium" | "high" => "high",
            "xhigh" | "max" => "max",
            _ => {
                return Err(Error::new(format!(
                    "unsupported reasoning_effort: {value:?}"
                )));
            }
        },
        Some(_) => return Err(Error::new("reasoning_effort must be a string")),
    };
    let thinking = match extra.get("thinking_mode").filter(|value| !value.is_null()) {
        Some(Value::String(value)) if value == "chat" => false,
        Some(Value::String(value)) if value == "thinking" => true,
        Some(_) => return Err(Error::new("thinking_mode must be 'chat' or 'thinking'")),
        None => configured_effort.is_some() || extra["thinking"]["type"] == "enabled",
    };
    Ok((thinking, effort))
}

/// Render the pinned checkpoint's text format, with explicit control spans.
pub fn render(request: &Value) -> Result<Prompt, Error> {
    let (thinking, effort) = thinking_settings(request)?;
    let mut messages = request["messages"]
        .as_array()
        .ok_or_else(|| Error::new("messages must be an array"))?
        .clone();
    if messages.is_empty() {
        return Err(Error::new("conversation must contain at least one message"));
    }
    let tools = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty());
    let response_format = request["extra"]
        .get("response_format")
        .filter(|value| !value.is_null());
    if tools.is_some() || response_format.is_some() {
        if messages[0]["role"] != "system" {
            messages.insert(0, serde_json::json!({"role": "system", "content": ""}));
        }
        if let Some(tools) = tools {
            messages[0]["tools"] = Value::Array(tools.clone());
        }
        if let Some(format) = response_format {
            messages[0]["response_format"] = format.clone();
        }
    }
    let messages = merge_messages(messages)?;
    let drop_thinking = !messages.iter().any(|message| {
        message["tools"]
            .as_array()
            .is_some_and(|tools| !tools.is_empty())
    });
    let last_user = messages
        .iter()
        .rposition(|message| matches!(message["role"].as_str(), Some("user" | "developer")));
    let messages: Vec<_> = messages
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut message)| {
            if thinking && drop_thinking && last_user.is_some_and(|last| index < last) {
                match message["role"].as_str() {
                    Some("developer") => return None,
                    Some("assistant") => {
                        message.as_object_mut().unwrap().remove("reasoning_content");
                    }
                    _ => {}
                }
            }
            Some(message)
        })
        .collect();
    let last_user = messages
        .iter()
        .rposition(|message| matches!(message["role"].as_str(), Some("user" | "developer")));
    let mut prompt = Prompt::new();
    prompt.special(BOS);
    if thinking {
        prompt.text(match effort {
            "high" => HIGH_EFFORT,
            "max" => MAX_EFFORT,
            _ => "",
        });
    }
    for (index, message) in messages.iter().enumerate() {
        let role = message["role"]
            .as_str()
            .ok_or_else(|| Error::new("message role must be a string"))?;
        match role {
            "system" | "developer" => {
                if role == "developer" {
                    prompt.special(USER);
                }
                prompt.text(content(message)?);
                if let Some(tools) = message["tools"]
                    .as_array()
                    .filter(|tools| !tools.is_empty())
                {
                    prompt.text("\n\n");
                    render_tools(&mut prompt, tools)?;
                }
                if let Some(format) = message
                    .get("response_format")
                    .filter(|value| !value.is_null())
                {
                    prompt.text("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                    prompt.text(json_spaced(format));
                }
            }
            "user" => {
                prompt.special(USER);
                if let Some(blocks) = message["content_blocks"].as_array() {
                    for (index, block) in blocks.iter().enumerate() {
                        if index > 0 {
                            prompt.text("\n\n");
                        }
                        match block["type"].as_str() {
                            Some("text") => {
                                prompt.text(block["text"].as_str().unwrap_or(""));
                            }
                            Some("tool_result") => {
                                prompt.text("<tool_result>");
                                prompt.text(text_value(&block["content"])?);
                                prompt.text("</tool_result>");
                            }
                            _ => return Err(Error::new("unsupported DeepSeek user content block")),
                        }
                    }
                } else {
                    prompt.text(content(message)?);
                }
            }
            "latest_reminder" => {
                prompt.special(REMINDER);
                prompt.text(content(message)?);
            }
            "assistant" => {
                let after_last_user = last_user.is_none_or(|last| index > last);
                let previous_has_task = index > 0
                    && messages[index - 1]
                        .get("task")
                        .is_some_and(|value| !value.is_null());
                if thinking && !previous_has_task && (!drop_thinking || after_last_user) {
                    prompt.text(message["reasoning_content"].as_str().unwrap_or(""));
                    prompt.special(THINKING_END);
                }
                prompt.text(content(message)?);
                if let Some(calls) = message["tool_calls"]
                    .as_array()
                    .filter(|calls| !calls.is_empty())
                {
                    render_tool_calls(&mut prompt, calls)?;
                }
                if message["wo_eos"] != true {
                    prompt.special(EOS);
                }
            }
            _ => {
                return Err(Error::new(format!(
                    "unsupported DeepSeek message role: {role}"
                )));
            }
        }
        let follows_assistant = messages.get(index + 1).is_none_or(|next| {
            matches!(next["role"].as_str(), Some("assistant" | "latest_reminder"))
        });
        if follows_assistant
            && (index + 1 < messages.len() || request["add_generation_prompt"] != false)
        {
            if let Some(task) = message.get("task").filter(|value| !value.is_null()) {
                let task = task
                    .as_str()
                    .ok_or_else(|| Error::new("DeepSeek task must be a string"))?;
                if ![
                    "action",
                    "query",
                    "authority",
                    "domain",
                    "title",
                    "read_url",
                ]
                .contains(&task)
                {
                    return Err(Error::new("unsupported DeepSeek task"));
                }
                if task == "action" {
                    prompt.special(ASSISTANT);
                    prompt.special(if thinking {
                        THINKING_START
                    } else {
                        THINKING_END
                    });
                }
                prompt.special(format!("<｜{task}｜>"));
            } else if role == "user" || role == "developer" {
                prompt.special(ASSISTANT);
                prompt.special(
                    if thinking && (!drop_thinking || last_user.is_none_or(|last| index >= last)) {
                        THINKING_START
                    } else {
                        THINKING_END
                    },
                );
            }
        }
    }
    Ok(prompt)
}

fn content(message: &Value) -> Result<String, Error> {
    text_value(&message["content"])
}

fn text_value(value: &Value) -> Result<String, Error> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                if block["type"] != "text" {
                    return Err(Error::new("DeepSeek V4 recipe supports text content only"));
                }
                text.push_str(
                    block["text"]
                        .as_str()
                        .ok_or_else(|| Error::new("text content must be a string"))?,
                );
            }
            Ok(text)
        }
        _ => Err(Error::new("message content must be text")),
    }
}

fn merge_messages(messages: Vec<Value>) -> Result<Vec<Value>, Error> {
    let mut merged: Vec<Value> = Vec::new();
    for mut message in messages {
        let role = message["role"].as_str();
        if matches!(role, Some("user" | "tool")) {
            let block = if role == Some("tool") {
                serde_json::json!({"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content(&message)?})
            } else {
                serde_json::json!({"type": "text", "text": content(&message)?})
            };
            if let Some(previous) = merged.last_mut().filter(|previous| {
                previous["role"] == "user"
                    && previous["content_blocks"].is_array()
                    && (role == Some("tool") || previous["task"].is_null())
            }) {
                previous["content_blocks"]
                    .as_array_mut()
                    .unwrap()
                    .push(block);
            } else {
                message["role"] = Value::String("user".into());
                message["content_blocks"] = Value::Array(vec![block]);
                merged.push(message);
            }
        } else {
            merged.push(message);
        }
    }
    let mut tool_order = Vec::<String>::new();
    for message in &mut merged {
        if message["role"] == "assistant" {
            if let Some(calls) = message["tool_calls"]
                .as_array()
                .filter(|calls| !calls.is_empty())
            {
                tool_order = calls
                    .iter()
                    .map(|call| {
                        call["id"]
                            .as_str()
                            .or_else(|| call["function"]["id"].as_str())
                            .unwrap_or("")
                            .to_owned()
                    })
                    .collect();
            }
        } else if let Some(blocks) = message["content_blocks"].as_array_mut() {
            let mut results: Vec<_> = blocks
                .iter()
                .filter(|block| block["type"] == "tool_result")
                .cloned()
                .collect();
            results.sort_by_key(|block| {
                tool_order
                    .iter()
                    .position(|id| Some(id.as_str()) == block["tool_use_id"].as_str())
                    .unwrap_or(0)
            });
            let mut results = results.into_iter();
            for block in blocks {
                if block["type"] == "tool_result" {
                    *block = results.next().unwrap();
                }
            }
        }
    }
    Ok(merged)
}

fn render_tools(prompt: &mut Prompt, tools: &[Value]) -> Result<(), Error> {
    // Only the template's delimiters are control tokens; schemas remain text.
    let mut template = include_str!("deepseek/tools.txt");
    while let Some((at, marker)) = [DSML, THINKING_START, THINKING_END]
        .into_iter()
        .filter_map(|marker| template.find(marker).map(|at| (at, marker)))
        .min_by_key(|(at, _)| *at)
    {
        prompt.text(&template[..at]);
        prompt.special(marker);
        template = &template[at + marker.len()..];
    }
    prompt.text(template);
    for tool in tools {
        let tool = tool.get("function").unwrap_or(tool);
        if !tool.is_object() {
            return Err(Error::new("tool schema must be an object"));
        }
        prompt.text(json_spaced(tool));
        prompt.text("\n");
    }
    prompt.text("\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
    Ok(())
}

fn tag(prompt: &mut Prompt, prefix: &str, suffix: &str) {
    prompt.text(prefix);
    prompt.special(DSML);
    prompt.text(suffix);
}

fn render_tool_calls(prompt: &mut Prompt, calls: &[Value]) -> Result<(), Error> {
    tag(prompt, "\n\n<", "tool_calls>\n");
    for (index, call) in calls.iter().enumerate() {
        if index > 0 {
            prompt.text("\n");
        }
        let function = call.get("function").unwrap_or(call);
        let name = function["name"]
            .as_str()
            .ok_or_else(|| Error::new("tool call name must be a string"))?;
        tag(prompt, "<", "invoke name=\"");
        prompt.text(name);
        prompt.text("\">\n");
        let arguments = function["arguments"]
            .as_str()
            .ok_or_else(|| Error::new("tool call arguments must be a JSON string"))?;
        let arguments: serde_json::Map<String, Value> = serde_json::from_str(arguments)
            .unwrap_or_else(|_| {
                serde_json::Map::from_iter([(
                    "arguments".to_owned(),
                    Value::String(arguments.to_owned()),
                )])
            });
        for (index, (key, value)) in arguments.iter().enumerate() {
            if index > 0 {
                prompt.text("\n");
            }
            tag(prompt, "<", "parameter name=\"");
            prompt.text(key);
            prompt.text(if value.is_string() {
                "\" string=\"true\">"
            } else {
                "\" string=\"false\">"
            });
            prompt.text(
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| json_spaced(value)),
            );
            tag(prompt, "</", "parameter>");
        }
        tag(prompt, "\n</", "invoke>");
    }
    tag(prompt, "\n</", "tool_calls>");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_defaults_and_explicit_mode_match_stock_codec() {
        assert_eq!(
            thinking_settings(&serde_json::json!({})).unwrap(),
            (false, "low")
        );
        assert_eq!(
            thinking_settings(&serde_json::json!({"reasoning_effort": "medium"})).unwrap(),
            (true, "high")
        );
        assert_eq!(thinking_settings(&serde_json::json!({"reasoning_effort": "xhigh", "extra": {"thinking_mode": "chat"}})).unwrap(), (false, "max"));
        assert_eq!(
            thinking_settings(&serde_json::json!({"extra": {"thinking": {"type": "enabled"}}}))
                .unwrap(),
            (true, "low")
        );
        assert!(thinking_settings(&serde_json::json!({"reasoning_effort": "none"})).is_err());
    }
}
