//! Inkling's pinned template; user text never becomes a control token.
use crate::render::{function, object_arguments, option, value_or};
use crate::{Error, Prompt};
use serde_json::{Value, json};

fn compact(value: &Value) -> String {
    let mut value = value.clone();
    value.sort_all_objects();
    crate::json::compact(&value)
}
fn block(prompt: &mut Prompt, role: &str, kind: &str, text: &str) {
    prompt
        .special(role)
        .special(kind)
        .text(text)
        .special("<|end_message|>");
}
fn effort(request: &Value) -> Result<String, Error> {
    let value = option(request, "reasoning_effort");
    let num = match value {
        None => 0.9,
        Some(Value::String(value)) => match value.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "max" | "xhigh" => 0.99,
            _ => return Err(Error::new("unknown Inkling reasoning_effort")),
        },
        Some(value) => value
            .as_f64()
            .ok_or_else(|| Error::new("reasoning_effort must be a number or effort name"))?,
    };
    if !(0.0..=0.99).contains(&num) {
        return Err(Error::new("reasoning_effort must be in [0.0, 0.99]"));
    }
    Ok(if num == 0.0 {
        "0".to_owned()
    } else {
        num.to_string()
    })
}
fn emit_effort(prompt: &mut Prompt, effort: &str) {
    block(
        prompt,
        "<|message_system|>",
        "<|content_text|>",
        &format!("Thinking effort level: {effort}"),
    );
}
pub fn render(request: &Value) -> Result<Prompt, Error> {
    let messages = request["messages"]
        .as_array()
        .ok_or_else(|| Error::new("messages must be an array"))?;
    let mut prompt = Prompt::new();
    if let Some(tools) = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty())
    {
        let specs: Vec<Value> = tools
            .iter()
            .map(|tool| {
                let function = function(tool);
                json!({
                    "description": value_or(&function["description"], json!("")),
                    "name": function["name"],
                    "parameters": value_or(&function["parameters"], json!({})),
                    "type": value_or(&tool["type"], json!("function")),
                })
            })
            .collect();
        prompt
            .special("<|message_system|>")
            .text("tool_declare")
            .special("<|content_xml|>")
            .text(compact(&Value::Array(specs)))
            .special("<|end_message|>");
    }
    let effort = effort(request)?;
    let mut effort_emitted = false;
    for message in messages {
        let role = message["role"].as_str().unwrap_or("");
        let token = match role {
            "system" => "<|message_system|>",
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "tool" => "<|message_tool|>",
            _ => return Err(Error::new(format!("unknown Inkling message role: {role}"))),
        };
        if !effort_emitted && role != "system" {
            emit_effort(&mut prompt, &effort);
            effort_emitted = true;
        }
        if role == "tool" {
            let name = message["name"]
                .as_str()
                .filter(|v| !v.is_empty())
                .or_else(|| {
                    let id = message["tool_call_id"].as_str()?;
                    messages
                        .iter()
                        .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten())
                        .filter(|call| call["id"].as_str() == Some(id))
                        .filter_map(|call| call["function"]["name"].as_str())
                        .next_back()
                })
                .unwrap_or("");
            prompt
                .special(token)
                .text(name)
                .special("<|content_text|>")
                .text(message["content"].as_str().unwrap_or(""))
                .special("<|end_message|>");
            continue;
        }
        if role == "assistant"
            && let Some(reasoning) = message["reasoning_content"]
                .as_str()
                .filter(|v| !v.is_empty())
        {
            block(&mut prompt, token, "<|content_thinking|>", reasoning);
        }
        match &message["content"] {
            Value::String(text) => block(&mut prompt, token, "<|content_text|>", text),
            Value::Array(parts) => {
                for part in parts {
                    if let Some(text) = part.as_str() {
                        block(&mut prompt, token, "<|content_text|>", text);
                        continue;
                    }
                    match part["type"].as_str().unwrap_or("text") {
                        "text" | "input_text" => block(
                            &mut prompt,
                            token,
                            "<|content_text|>",
                            part["text"].as_str().unwrap_or(""),
                        ),
                        "image" | "input_image" | "image_url" => {
                            prompt
                                .special(token)
                                .special("<|content_image|>")
                                .special("<|unused_200054|>")
                                .special("<|end_message|>");
                        }
                        "audio" | "input_audio" | "audio_url" => {
                            prompt
                                .special(token)
                                .special("<|content_audio_input|>")
                                .special("<|unused_200053|>")
                                .special("<|audio_end|>")
                                .special("<|end_message|>");
                        }
                        other => {
                            return Err(Error::new(format!(
                                "unsupported Inkling content part: {other}"
                            )));
                        }
                    }
                }
            }
            _ => (),
        }
        if role == "assistant" {
            for call in message["tool_calls"].as_array().into_iter().flatten() {
                let function = &call["function"];
                let name = function["name"]
                    .as_str()
                    .ok_or_else(|| Error::new("tool call function name must be a string"))?;
                let args = value_or(&function["arguments"], json!({}));
                let args = object_arguments(&args)?;
                prompt
                    .special(token)
                    .text(name)
                    .special("<|content_invoke_tool_json|>")
                    .text(format!(
                        "{{\"name\":{},\"args\":{}}}",
                        compact(&Value::String(name.into())),
                        compact(&Value::Object(args.into_owned()))
                    ))
                    .special("<|end_message|>");
            }
            prompt.special("<|content_model_end_sampling|>");
        }
    }
    if !effort_emitted {
        emit_effort(&mut prompt, &effort);
    }
    if request["add_generation_prompt"].as_bool().unwrap_or(true) {
        prompt.special("<|message_model|>");
    }
    Ok(prompt)
}
