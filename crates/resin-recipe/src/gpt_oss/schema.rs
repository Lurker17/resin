// Adapted from OpenAI Harmony's JSON-schema tool formatter (Apache-2.0).
// https://github.com/openai/harmony/blob/abd677f7ac962629c808197caa1feb9e3e95d2b0/src/encoding.rs
// Preserve its exact whitespace, enum, default, and oneOf behavior: this text
// becomes model input, so semantic TypeScript equivalence is insufficient.

pub(super) fn schema_type(schema: &serde_json::Value, indent: &str) -> String {
    // Helper to check if this schema is an enum
    fn is_enum(schema: &serde_json::Value) -> bool {
        schema
            .get("enum")
            .and_then(|e| e.as_array())
            .is_some_and(|arr| !arr.is_empty())
    }

    // Handle oneOf at the top level
    if let Some(one_of) = schema.get("oneOf")
        && let Some(arr) = one_of.as_array()
    {
        let mut out = String::new();
        let mut first = true;
        for variant in arr {
            if !first {
                out.push('\n');
                out.push_str(&format!("{indent} | "));
            } else {
                out.push_str(&format!("\n{indent} | "));
                first = false;
            }
            let type_str = schema_type(variant, &format!("{indent}   "));
            let mut type_str = type_str;
            if variant
                .get("nullable")
                .and_then(|n| n.as_bool())
                .unwrap_or(false)
                && !type_str.contains("null")
            {
                type_str = format!("{type_str} | null");
            }
            out.push_str(&type_str);
            // Add trailing comments (description, default)
            let mut trailing_comments = Vec::new();
            if let Some(desc) = variant.get("description")
                && let Some(desc_str) = desc.as_str()
            {
                trailing_comments.push(desc_str.to_string());
            }
            if let Some(default) = variant.get("default") {
                if default.is_string() && !is_enum(variant) {
                    trailing_comments.push(format!("default: \"{}\"", default.as_str().unwrap()));
                } else {
                    trailing_comments.push(format!("default: {default}"));
                }
            }
            if !trailing_comments.is_empty() {
                out.push_str(&format!(" // {}", trailing_comments.join(" ")));
            }
        }
        return out;
    }
    // Handle type as array (e.g., ["number", "string"])
    if let Some(types) = schema.get("type").and_then(|v| v.as_array()) {
        let mut type_strings = Vec::new();
        for ty in types {
            if let Some(ty_str) = ty.as_str() {
                let mapped = match ty_str {
                    "integer" => "number",
                    other => other,
                };
                type_strings.push(mapped.to_string());
            }
        }
        if !type_strings.is_empty() {
            return type_strings.join(" | ");
        }
    }
    // Handle type
    if let Some(ty) = schema.get("type").and_then(|v| v.as_str()) {
        match ty {
            "object" => {
                let mut out = String::new();
                // Render object-level description as comment
                if let Some(desc) = schema.get("description")
                    && let Some(desc_str) = desc.as_str()
                {
                    out.push_str(&format!("{indent}// {desc_str}\n"));
                }
                out.push_str("{\n");

                if let Some(props) = schema.get("properties")
                    && let Some(props_map) = props.as_object()
                {
                    // Determine required fields
                    let mut required = std::collections::HashSet::new();
                    if let Some(req) = schema.get("required")
                        && let Some(req_arr) = req.as_array()
                    {
                        for r in req_arr {
                            if let Some(s) = r.as_str() {
                                required.insert(s);
                            }
                        }
                    }
                    for (key, val) in props_map {
                        // Render title, description, and examples as comments
                        if let Some(title) = val.get("title")
                            && let Some(title_str) = title.as_str()
                        {
                            out.push_str(&format!("{indent}// {title_str}\n{indent}//\n"));
                        }
                        // Only render description here if not a oneOf property
                        if val.get("oneOf").is_none()
                            && let Some(desc) = val.get("description")
                            && let Some(desc_str) = desc.as_str()
                        {
                            out.push_str(&format!("{indent}// {desc_str}\n"));
                        }
                        if let Some(examples) = val.get("examples")
                            && let Some(arr) = examples.as_array()
                            && !arr.is_empty()
                        {
                            out.push_str(&format!("{indent}// Examples:\n"));
                            for ex in arr {
                                if let Some(ex_str) = ex.as_str() {
                                    out.push_str(&format!("{indent}// - \"{ex_str}\"\n"));
                                }
                            }
                        }
                        // Handle oneOf at the property level
                        if let Some(one_of) = val.get("oneOf")
                            && let Some(arr) = one_of.as_array()
                        {
                            // Deduplicate property-level description if it matches the first variant's description
                            let mut property_desc: Option<&str> = None;
                            if let Some(desc) = val.get("description")
                                && let Some(desc_str) = desc.as_str()
                            {
                                property_desc = Some(desc_str);
                            }
                            let mut skip_property_desc = false;
                            if let Some(desc_str) = property_desc
                                && let Some(first_variant) = arr.first()
                                && let Some(variant_desc) = first_variant.get("description")
                                && let Some(variant_desc_str) = variant_desc.as_str()
                                && desc_str == variant_desc_str
                            {
                                skip_property_desc = true;
                            }
                            // Add property-level comments above the property name if not skipped
                            let mut rendered_property_desc_above = false;
                            if !skip_property_desc && let Some(desc_str) = property_desc {
                                out.push_str(&format!("{indent}// {desc_str}\n"));
                                rendered_property_desc_above = true;
                            }
                            if let Some(default) = val.get("default") {
                                if default.is_string() && !is_enum(val) {
                                    out.push_str(&format!(
                                        "{}// default: \"{}\"\n",
                                        indent,
                                        default.as_str().unwrap()
                                    ));
                                } else if default.is_string() {
                                    out.push_str(&format!(
                                        "{}// default: {}\n",
                                        indent,
                                        default.as_str().unwrap()
                                    ));
                                } else {
                                    out.push_str(&format!("{indent}// default: {default}\n"));
                                }
                            }
                            // Add property name and optional marker
                            out.push_str(&format!(
                                "{}{}{}:\n",
                                indent,
                                key,
                                if required.contains(key.as_str()) {
                                    ""
                                } else {
                                    "?"
                                }
                            ));
                            // Render each variant
                            for (i, variant) in arr.iter().enumerate() {
                                out.push_str(&format!("{indent} | "));
                                let type_str = schema_type(variant, &format!("{indent}   "));
                                // Handle nullable in variant
                                let mut type_str = type_str;
                                if variant
                                    .get("nullable")
                                    .and_then(|n| n.as_bool())
                                    .unwrap_or(false)
                                    && !type_str.contains("null")
                                {
                                    type_str = format!("{type_str} | null");
                                }
                                out.push_str(&type_str);
                                // Add variant-level comments after the type
                                let mut trailing_comments = Vec::new();
                                if i == 0 && rendered_property_desc_above {
                                    // Do not add any description for the first variant if property-level description was rendered above
                                } else if let Some(desc) = variant.get("description")
                                    && let Some(desc_str) = desc.as_str()
                                {
                                    // Only render if not equal to property-level description
                                    if Some(desc_str) != property_desc {
                                        trailing_comments.push(desc_str.to_string());
                                    }
                                }
                                if let Some(default) = variant.get("default") {
                                    if default.is_string() && !is_enum(variant) {
                                        trailing_comments.push(format!(
                                            "default: \"{}\"",
                                            default.as_str().unwrap()
                                        ));
                                    } else if default.is_string() {
                                        trailing_comments.push(format!(
                                            "default: {}",
                                            default.as_str().unwrap()
                                        ));
                                    } else {
                                        trailing_comments.push(format!("default: {default}"));
                                    }
                                }
                                if !trailing_comments.is_empty() {
                                    out.push_str(&format!(" // {}", trailing_comments.join(" ")));
                                }
                                out.push('\n');
                            }
                            out.push_str(&format!("{indent},\n"));
                            continue;
                        }
                        // Normal property rendering
                        out.push_str(&format!(
                            "{}{}{}: ",
                            indent,
                            key,
                            if required.contains(key.as_str()) {
                                ""
                            } else {
                                "?"
                            }
                        ));
                        // Handle nullable
                        let mut type_str = schema_type(val, &format!("{indent}    "));
                        if val
                            .get("nullable")
                            .and_then(|n| n.as_bool())
                            .unwrap_or(false)
                            && !type_str.contains("null")
                        {
                            type_str = format!("{type_str} | null");
                        }
                        out.push_str(&type_str);
                        out.push(',');
                        // Add default as comment if present (and not already handled)
                        if val.get("oneOf").is_none()
                            && let Some(default) = val.get("default")
                        {
                            if default.is_string() && !is_enum(val) {
                                out.push_str(&format!(
                                    " // default: \"{}\"",
                                    default.as_str().unwrap()
                                ));
                            } else if default.is_string() {
                                out.push_str(&format!(
                                    " // default: {}",
                                    default.as_str().unwrap()
                                ));
                            } else {
                                out.push_str(&format!(" // default: {default}"));
                            }
                        }
                        out.push('\n');
                    }
                }
                out.push_str(&format!("{indent}}}"));
                out
            }
            "string" => {
                if let Some(enum_vals) = schema.get("enum")
                    && let Some(arr) = enum_vals.as_array()
                {
                    let enums: Vec<String> = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| format!("\"{s}\"")))
                        .collect();
                    if !enums.is_empty() {
                        return enums.join(" | ");
                    }
                }
                "string".to_string()
            }
            "number" => "number".to_string(),
            "integer" => "number".to_string(),
            "boolean" => "boolean".to_string(),
            "array" => {
                if let Some(items) = schema.get("items") {
                    format!("{}[]", schema_type(items, indent))
                } else {
                    "Array<any>".to_string()
                }
            }
            _ => "any".to_string(),
        }
    } else if let Some(one_of) = schema.get("oneOf") {
        // Defensive: already handled above, but just in case
        if let Some(arr) = one_of.as_array() {
            let mut out = String::new();
            let mut first = true;
            for variant in arr {
                if !first {
                    out.push_str("\n | ");
                } else {
                    first = false;
                }
                out.push_str(&schema_type(variant, indent));
            }
            return out;
        }
        "any".to_string()
    } else {
        "any".to_string()
    }
}
