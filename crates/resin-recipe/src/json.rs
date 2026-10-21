//! Python-compatible JSON separators and float spelling for model prompts.

use serde_json::{Number, Value};

pub(crate) fn spaced(value: &Value) -> String {
    format(value, ", ", ": ")
}
pub(crate) fn compact(value: &Value) -> String {
    format(value, ",", ":")
}

fn format(value: &Value, comma: &str, colon: &str) -> String {
    fn write(out: &mut String, value: &Value, comma: &str, colon: &str) {
        match value {
            Value::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        out.push_str(comma);
                    }
                    write(out, value, comma, colon);
                }
                out.push(']');
            }
            Value::Object(values) => {
                out.push('{');
                for (index, (key, value)) in values.iter().enumerate() {
                    if index != 0 {
                        out.push_str(comma);
                    }
                    out.push_str(&serde_json::to_string(key).unwrap());
                    out.push_str(colon);
                    write(out, value, comma, colon);
                }
                out.push('}');
            }
            Value::Number(number) if number.is_f64() => out.push_str(&float(number)),
            _ => out.push_str(&value.to_string()),
        }
    }
    let mut out = String::new();
    write(&mut out, value, comma, colon);
    out
}

fn float(number: &Number) -> String {
    // Keep serde's correctly rounded shortest digits; only Python's notation
    // thresholds and signed, zero-padded exponent differ. Rust Debug rounds
    // some shortest-decimal ties differently, so cannot replace this step.
    let compact = number.to_string();
    let (sign, unsigned) = compact
        .strip_prefix('-')
        .map_or(("", compact.as_str()), |text| ("-", text));
    let (mantissa, power) = unsigned
        .split_once('e')
        .map_or((unsigned, 0), |(mantissa, power)| {
            (mantissa, power.parse::<i32>().unwrap())
        });
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits = mantissa.replace('.', "");
    let zeros = digits.len() - digits.trim_start_matches('0').len();
    if zeros == digits.len() {
        return format!("{sign}0.0");
    }
    let digits = digits[zeros..].trim_end_matches('0');
    let exponent = decimal + power - zeros as i32 - 1;
    let body = if !(-4..16).contains(&exponent) {
        let tail = if digits.len() == 1 {
            String::new()
        } else {
            format!(".{}", &digits[1..])
        };
        format!("{}{tail}e{exponent:+03}", &digits[..1])
    } else if exponent < 0 {
        format!("0.{}{digits}", "0".repeat((-exponent - 1) as usize))
    } else {
        let whole = exponent as usize + 1;
        if whole < digits.len() {
            format!("{}.{}", &digits[..whole], &digits[whole..])
        } else {
            format!("{digits}{}.0", "0".repeat(whole - digits.len()))
        }
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_json_float_thresholds_and_separators() {
        let value = serde_json::json!({"floats": [1e-7, 1e-5, 0.0001, 1e15, 1e16, -0.0, 1.25e-7], "unicode": "東京"});
        assert_eq!(
            spaced(&value),
            "{\"floats\": [1e-07, 1e-05, 0.0001, 1000000000000000.0, 1e+16, -0.0, 1.25e-07], \"unicode\": \"東京\"}"
        );
    }
    #[test]
    fn float_midpoints_keep_python_shortest_spelling() {
        for text in [
            "5.960464477539063e-08",
            "1441495712046950.2",
            "226180940831505.12",
        ] {
            let value: Value = serde_json::from_str(text).unwrap();
            assert_eq!(spaced(&value), text);
        }
    }
}
