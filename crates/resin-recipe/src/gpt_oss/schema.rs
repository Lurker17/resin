// Adapted from OpenAI Harmony's JSON-schema tool formatter (Apache-2.0).
// https://github.com/openai/harmony/blob/abd677f7ac962629c808197caa1feb9e3e95d2b0/src/encoding.rs
// Whitespace, enum defaults and oneOf comments are part of the model format.

use serde_json::Value;

fn string<'a>(schema: &'a Value, key: &str) -> Option<&'a str> {
    schema.get(key).and_then(Value::as_str)
}

fn default_comment(schema: &Value, quote_enum: bool) -> Option<String> {
    // Harmony quotes enum defaults in root unions, but not object properties.
    let value = schema.get("default")?;
    let has_enum = schema["enum"]
        .as_array()
        .is_some_and(|values| !values.is_empty());
    let value = match value.as_str() {
        Some(text) if !has_enum => format!("\"{text}\""),
        Some(text) if !quote_enum => text.to_owned(),
        _ => value.to_string(),
    };
    Some(format!("default: {value}"))
}

fn trailing(out: &mut String, description: Option<&str>, schema: &Value, quote_enum: bool) {
    let mut comments = Vec::new();
    if let Some(description) = description {
        comments.push(description.to_owned());
    }
    if let Some(default) = default_comment(schema, quote_enum) {
        comments.push(default);
    }
    if !comments.is_empty() {
        out.push_str(&format!(" // {}", comments.join(" ")));
    }
}

fn nullable(schema: &Value, indent: &str) -> String {
    let mut text = schema_type(schema, indent);
    if schema["nullable"].as_bool() == Some(true) && !text.contains("null") {
        text.push_str(" | null");
    }
    text
}

pub(super) fn schema_type(schema: &Value, indent: &str) -> String {
    if let Some(variants) = schema["oneOf"].as_array() {
        let mut out = String::new();
        for variant in variants {
            out.push_str(&format!(
                "\n{indent} | {}",
                nullable(variant, &format!("{indent}   "))
            ));
            trailing(&mut out, string(variant, "description"), variant, true);
        }
        return out;
    }
    if let Some(types) = schema["type"].as_array() {
        let types: Vec<_> = types
            .iter()
            .filter_map(Value::as_str)
            .map(|kind| if kind == "integer" { "number" } else { kind })
            .collect();
        if !types.is_empty() {
            return types.join(" | ");
        }
    }
    match string(schema, "type") {
        Some("object") => object(schema, indent),
        Some("string") => {
            let variants: Vec<_> = schema["enum"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(|value| format!("\"{value}\""))
                .collect();
            if variants.is_empty() {
                "string".into()
            } else {
                variants.join(" | ")
            }
        }
        Some("number" | "integer") => "number".into(),
        Some("boolean") => "boolean".into(),
        Some("array") => schema.get("items").map_or_else(
            || "Array<any>".into(),
            |items| format!("{}[]", schema_type(items, indent)),
        ),
        _ => "any".into(),
    }
}

fn object(schema: &Value, indent: &str) -> String {
    let mut out = String::new();
    if let Some(description) = string(schema, "description") {
        out.push_str(&format!("{indent}// {description}\n"));
    }
    out.push_str("{\n");
    let required: Vec<_> = schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    for (name, property) in schema["properties"].as_object().into_iter().flatten() {
        if let Some(title) = string(property, "title") {
            out.push_str(&format!("{indent}// {title}\n{indent}//\n"));
        }
        let description = string(property, "description");
        if property.get("oneOf").is_none()
            && let Some(description) = description
        {
            out.push_str(&format!("{indent}// {description}\n"));
        }
        if let Some(examples) = property["examples"]
            .as_array()
            .filter(|examples| !examples.is_empty())
        {
            out.push_str(&format!("{indent}// Examples:\n"));
            for example in examples.iter().filter_map(Value::as_str) {
                out.push_str(&format!("{indent}// - \"{example}\"\n"));
            }
        }
        let optional = if required.contains(&name.as_str()) {
            ""
        } else {
            "?"
        };
        if let Some(variants) = property["oneOf"].as_array() {
            let own_description = description.filter(|description| {
                Some(*description)
                    != variants
                        .first()
                        .and_then(|variant| string(variant, "description"))
            });
            if let Some(description) = own_description {
                out.push_str(&format!("{indent}// {description}\n"));
            }
            if let Some(default) = default_comment(property, false) {
                out.push_str(&format!("{indent}// {default}\n"));
            }
            out.push_str(&format!("{indent}{name}{optional}:\n"));
            for (index, variant) in variants.iter().enumerate() {
                out.push_str(&format!(
                    "{indent} | {}",
                    nullable(variant, &format!("{indent}   "))
                ));
                let description = string(variant, "description").filter(|text| {
                    Some(*text) != description && !(index == 0 && own_description.is_some())
                });
                trailing(&mut out, description, variant, false);
                out.push('\n');
            }
            out.push_str(&format!("{indent},\n"));
        } else {
            out.push_str(&format!(
                "{indent}{name}{optional}: {},",
                nullable(property, &format!("{indent}    "))
            ));
            if property.get("oneOf").is_none() {
                trailing(&mut out, None, property, false);
            }
            out.push('\n');
        }
    }
    out.push_str(&format!("{indent}}}"));
    out
}
