use std::io::BufRead;

use arrow_schema::ArrowError;
use serde_json::{Map, Number, Value};

use crate::reader::tape::{Tape, TapeDecoder, TapeElement};

/// JSON file reader that produces a serde_json::Value iterator from a Read trait
///
/// # Example
///
/// ```
/// use std::fs::File;
/// use std::io::BufReader;
/// use arrow_json::reader::ValueIter;
///
/// let mut reader =
///     BufReader::new(File::open("test/data/mixed_arrays.json").unwrap());
/// let mut value_reader = ValueIter::new(&mut reader, None);
/// for value in value_reader {
///     println!("JSON value: {}", value.unwrap());
/// }
/// ```
#[derive(Debug)]
pub struct ValueIter<R: BufRead> {
    reader: R,
    max_read_records: Option<usize>,
    record_count: usize,
    decoder: TapeDecoder,
}

impl<R: BufRead> ValueIter<R> {
    /// Creates a new `ValueIter`
    pub fn new(reader: R, max_read_records: Option<usize>) -> Self {
        Self {
            reader,
            max_read_records,
            record_count: 0,
            decoder: TapeDecoder::new(1, 0),
        }
    }

    /// Returns the number of records this iterator has consumed
    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// Decodes a single JSON object, returning `Ok(None)` if all rows have been read
    fn read_value(&mut self) -> Result<Option<Value>, ArrowError> {
        loop {
            let byte_cnt = self.decoder.decode(self.reader.fill_buf()?)?;
            if byte_cnt == 0 {
                break;
            }

            self.reader.consume(byte_cnt);
        }

        debug_assert!(self.decoder.num_buffered_rows() <= 1);
        debug_assert!(!self.decoder.has_partial_row());

        let tape = self.decoder.finish()?;
        if tape.num_rows() == 0 {
            return Ok(None);
        }

        let (value, _) = tape.to_value(1)?;
        self.decoder.clear();

        Ok(Some(value))
    }
}

impl<R: BufRead> Iterator for ValueIter<R> {
    type Item = Result<Value, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(max) = self.max_read_records {
            if self.record_count >= max {
                return None;
            }
        }

        match self.read_value() {
            Ok(Some(record)) => {
                self.record_count += 1;
                Some(Ok(record))
            }
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    }
}

impl<'a> Tape<'a> {
    fn to_value(&self, idx: u32) -> Result<(Value, u32), ArrowError> {
        Ok(match self.get(idx) {
            TapeElement::StartObject(end) => {
                let mut cur_idx = idx + 1;
                let mut fields = Map::new();
                while cur_idx < end {
                    let TapeElement::String(key) = self.get(cur_idx) else {
                        unreachable!();
                    };
                    let key = self.get_string(key).into();
                    let (value, next_idx) = self.to_value(cur_idx + 1)?;
                    fields.insert(key, value);
                    debug_assert!(next_idx > cur_idx);
                    cur_idx = next_idx;
                }
                (Value::Object(fields), end + 1)
            }
            TapeElement::EndObject(_) => unreachable!(),
            TapeElement::StartList(end) => {
                let mut cur_idx = idx + 1;
                let mut elements = Vec::new();
                while cur_idx < end {
                    let (value, next_idx) = self.to_value(cur_idx)?;
                    elements.push(value);
                    debug_assert!(next_idx > cur_idx);
                    cur_idx = next_idx;
                }
                (Value::Array(elements), end + 1)
            }
            TapeElement::EndList(_) => unreachable!(),
            TapeElement::String(s) => (Value::String(self.get_string(s).into()), idx + 1),
            TapeElement::Number(s) => {
                use lexical_core::parse;

                let s = self.get_string(s);

                let b = s.as_bytes();
                let value = parse::<u64>(b)
                    .map(Number::from)
                    .or_else(|_| parse::<i64>(b).map(Number::from))
                    .or_else(|_| parse::<f64>(b).map(Number::from_f64).map(Option::unwrap))
                    .map_err(|_| ArrowError::JsonError(format!("failed to parse {s}")))?;

                (value.into(), idx + 1)
            }
            TapeElement::I64(high) => {
                let TapeElement::I32(low) = self.get(idx + 1) else {
                    unreachable!();
                };
                let value = ((high as i64) << 32) | (low as u32) as i64;
                (Number::from(value).into(), idx + 2)
            }
            TapeElement::I32(value) => (Number::from(value).into(), idx + 1),
            TapeElement::F64(high) => {
                let TapeElement::F32(low) = self.get(idx + 1) else {
                    unreachable!();
                };
                let value = f64::from_bits(((high as u64) << 32) | low as u64);
                (Number::from_f64(value).into(), idx + 2)
            }
            TapeElement::F32(value) => (Number::from(value).into(), idx + 1),
            TapeElement::True => (Value::Bool(true), idx + 1),
            TapeElement::False => (Value::Bool(false), idx + 1),
            TapeElement::Null => (Value::Null, idx + 1),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_incomplete_record() {
        let src = r#"{"foo":"bar","other":42}\n{"foo":"bar","#;
        let mut iter = ValueIter::new(src.as_bytes(), None);

        let row = json!({ "foo": "bar", "other": 42 });

        assert_eq!(iter.next().unwrap().unwrap(), row);
        assert!(iter.next().unwrap().is_err());
    }
}
