//! Python-compatible JSON spacing and float formatting for model prompts.

use serde::Serialize;
use serde_json::{
    Value,
    ser::{CompactFormatter, Formatter},
};

struct PythonFormatter {
    spaced: bool,
}

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        writer.write_all(if first {
            b""
        } else if self.spaced {
            b", "
        } else {
            b","
        })
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        writer.write_all(if first {
            b""
        } else if self.spaced {
            b", "
        } else {
            b","
        })
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(if self.spaced { b": " } else { b":" })
    }

    fn write_f64<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        value: f64,
    ) -> std::io::Result<()> {
        let mut compact = Vec::new();
        CompactFormatter.write_f64(&mut compact, value)?;
        let compact = String::from_utf8(compact).expect("float formatter emits ASCII");
        let (sign, unsigned) = compact
            .strip_prefix('-')
            .map_or(("", compact.as_str()), |text| ("-", text));
        let (mantissa, power) = unsigned
            .split_once('e')
            .map_or((unsigned, 0), |(mantissa, power)| {
                (mantissa, power.parse::<i32>().unwrap())
            });
        let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
        let digits: String = mantissa
            .chars()
            .filter(|&character| character != '.')
            .collect();
        let zeros = digits.bytes().take_while(|&byte| byte == b'0').count();
        if zeros == digits.len() {
            return writer.write_all(if sign.is_empty() { b"0.0" } else { b"-0.0" });
        }
        let digits = digits[zeros..].trim_end_matches('0');
        let exponent = decimal + power - zeros as i32 - 1;
        writer.write_all(sign.as_bytes())?;
        if !(-4..16).contains(&exponent) {
            writer.write_all(&digits.as_bytes()[..1])?;
            if digits.len() > 1 {
                writer.write_all(b".")?;
                writer.write_all(&digits.as_bytes()[1..])?;
            }
            write!(
                writer,
                "e{}{exponent:02}",
                if exponent < 0 { "-" } else { "+" },
                exponent = exponent.abs()
            )
        } else if exponent < 0 {
            writer.write_all(b"0.")?;
            for _ in 0..-exponent - 1 {
                writer.write_all(b"0")?;
            }
            writer.write_all(digits.as_bytes())
        } else {
            let whole = exponent as usize + 1;
            if whole < digits.len() {
                writer.write_all(&digits.as_bytes()[..whole])?;
                writer.write_all(b".")?;
                writer.write_all(&digits.as_bytes()[whole..])
            } else {
                writer.write_all(digits.as_bytes())?;
                for _ in digits.len()..whole {
                    writer.write_all(b"0")?;
                }
                writer.write_all(b".0")
            }
        }
    }
}

pub(crate) fn spaced(value: &Value) -> String {
    format(value, true)
}
pub(crate) fn compact(value: &Value) -> String {
    format(value, false)
}
fn format(value: &Value, spaced: bool) -> String {
    let mut bytes = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            PythonFormatter { spaced },
        ))
        .expect("JSON values serialize to memory");
    String::from_utf8(bytes).expect("JSON serialization produces UTF-8")
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
    fn float_at_binary_midpoint_keeps_python_shortest_spelling() {
        let value: Value = serde_json::from_str("5.960464477539063e-08").unwrap();
        assert_eq!(spaced(&value), "5.960464477539063e-08");
    }
}
