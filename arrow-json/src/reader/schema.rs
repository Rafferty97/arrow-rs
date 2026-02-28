// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{BufRead, Seek};
use std::usize;

use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use bumpalo::Bump;

use super::tape::{Tape, TapeDecoder, TapeElement};

type Result<T> = std::result::Result<T, ArrowError>;

/// Infer the fields of a JSON file by reading the first n records of the file, with
/// `max_read_records` controlling the maximum number of records to read.
///
/// If `max_read_records` is not set, the whole file is read to infer its field types.
///
/// Returns inferred schema and number of records read.
///
/// Contrary to [`infer_json_schema`], this function will seek back to the start of the `reader`.
/// That way, the `reader` can be used immediately afterwards to create a [`Reader`].
///
/// # Examples
/// ```
/// use std::fs::File;
/// use std::io::BufReader;
/// use arrow_json::reader::infer_json_schema_from_seekable;
///
/// let file = File::open("test/data/arrays.json").unwrap();
/// // file's cursor's offset at 0
/// let mut reader = BufReader::new(file);
/// let inferred_schema = infer_json_schema_from_seekable(&mut reader, None).unwrap();
/// // file's cursor's offset automatically set at 0
/// ```
///
/// [`Reader`]: super::Reader
pub fn infer_json_schema_from_seekable<R: BufRead + Seek>(
    mut reader: R,
    max_read_records: Option<usize>,
) -> Result<(Schema, usize)> {
    let schema = infer_json_schema(&mut reader, max_read_records);
    // return the reader seek back to the start
    reader.rewind()?;

    schema
}

/// Infer the fields of a JSON file by reading the first n records of the buffer, with
/// `max_read_records` controlling the maximum number of records to read.
///
/// If `max_read_records` is not set, the whole file is read to infer its field types.
///
/// Returns inferred schema and number of records read.
///
/// This function will not seek back to the start of the `reader`. The user has to manage the
/// original file's cursor. This function is useful when the `reader`'s cursor is not available
/// (does not implement [`Seek`]), such is the case for compressed streams decoders.
///
///
/// Note that JSON is not able to represent all Arrow data types exactly. So the inferred schema
/// might be different from the schema of the original data that was encoded as JSON. For example,
/// JSON does not have different integer types, so all integers are inferred as `Int64`. Another
/// example is binary data, which is encoded as a [Base16] string in JSON and therefore inferred
/// as String type by this function.
///
/// [Base16]: https://en.wikipedia.org/wiki/Base16#Base16
///
/// # Examples
/// ```
/// use std::fs::File;
/// use std::io::{BufReader, SeekFrom, Seek};
/// use flate2::read::GzDecoder;
/// use arrow_json::reader::infer_json_schema;
///
/// let mut file = File::open("test/data/arrays.json.gz").unwrap();
///
/// // file's cursor's offset at 0
/// let mut reader = BufReader::new(GzDecoder::new(&file));
/// let inferred_schema = infer_json_schema(&mut reader, None).unwrap();
/// // cursor's offset at end of file
///
/// // seek back to start so that the original file is usable again
/// file.seek(SeekFrom::Start(0)).unwrap();
/// ```
pub fn infer_json_schema<R: BufRead>(
    mut reader: R,
    max_read_records: Option<usize>,
) -> Result<(Schema, usize)> {
    let arena = Bump::new();
    let mut decoder = SchemaDecoder::new(max_read_records, &arena);

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }
        let read = buf.len();

        let decoded = decoder.decode(buf)?;
        reader.consume(read);
        if decoded != read {
            break;
        }
    }

    decoder.finish()
}

/// Infer the fields of a JSON file by reading all items from the JSON Value Iterator.
///
/// The following type coercion logic is implemented:
/// * `Int64` and `Float64` are converted to `Float64`
/// * Incompatible scalars are coerced to `Utf8` (String)
///
/// Note that the above coercion logic is different from what Spark has, where it would default to
/// String type in case of List and Scalar values appeared in the same field.
///
/// The reason we diverge here is because we don't have utilities to deal with JSON data once it's
/// interpreted as Strings. We should match Spark's behavior once we added more JSON parsing
/// kernels in the future.
pub fn infer_json_schema_from_iterator<I, V>(value_iter: I) -> Result<Schema>
where
    I: Iterator<Item = Result<V>>,
    V: Borrow<serde_json::Value>,
{
    let arena = &Bump::new();

    value_iter
        .into_iter()
        .try_fold(EMPTY_OBJECT_TY, |ty, record| {
            infer_json_type(record?.borrow(), ty, arena)
        })?
        .into_schema()
}

struct SchemaDecoder<'a> {
    decoder: TapeDecoder,
    max_read_records: Option<usize>,
    record_count: usize,
    schema: &'a InferredTy<'a>,
    arena: &'a Bump,
}

impl<'a> SchemaDecoder<'a> {
    pub fn new(max_read_records: Option<usize>, arena: &'a Bump) -> Self {
        Self {
            decoder: TapeDecoder::new(1024, 8),
            max_read_records,
            record_count: 0,
            schema: NEVER_TY,
            arena,
        }
    }

    pub fn decode(&mut self, buf: &[u8]) -> Result<usize> {
        let read = self.decoder.decode(buf)?;
        if read != buf.len() {
            self.infer_batch()?;
        }
        Ok(read)
    }

    pub fn finish(mut self) -> Result<(Schema, usize)> {
        self.infer_batch()?;
        Ok((self.schema.into_schema()?, self.record_count))
    }

    fn infer_batch(&mut self) -> Result<()> {
        let tape = self.decoder.finish()?;

        let remaining_records = self
            .max_read_records
            .map_or(usize::MAX, |max| max - self.record_count);

        for value in iter_rows(&tape).take(remaining_records) {
            self.schema = infer_json_type(value, self.schema, self.arena)?;
            self.record_count += 1;
        }

        self.decoder.clear();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum InferredTy<'a> {
    Never,
    Scalar(ScalarTy),
    Array(&'a InferredTy<'a>),
    Object(InferredFields<'a>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScalarTy {
    Bool,
    Int64,
    Float64,
    String,
    // NOTE: Null isn't needed because it's absorbed into Never
}

type InferredFields<'a> = &'a [(&'a str, &'a InferredTy<'a>)];

static NEVER_TY: &InferredTy<'static> = &InferredTy::Never;
static BOOL_TY: &InferredTy<'static> = &InferredTy::Scalar(ScalarTy::Bool);
static INT64_TY: &InferredTy<'static> = &InferredTy::Scalar(ScalarTy::Int64);
static FLOAT64_TY: &InferredTy<'static> = &InferredTy::Scalar(ScalarTy::Float64);
static STRING_TY: &InferredTy<'static> = &InferredTy::Scalar(ScalarTy::String);
static ARRAY_OF_NEVER_TY: &InferredTy<'static> = &InferredTy::Array(NEVER_TY);
static EMPTY_OBJECT_TY: &InferredTy<'static> = &InferredTy::Object(&[]);

/// Abstraction over a JSON value that is only concerned with the
/// type of the value rather than the value itself
trait JsonValue<'a>: Debug + Copy {
    fn get(&self) -> JsonType;

    fn elements(&self) -> impl Iterator<Item = Self>;

    fn fields(&self) -> impl Iterator<Item = (&'a str, Self)>;
}

enum JsonType {
    Null,
    Bool,
    Int64,
    Float64,
    String,
    Array,
    Object,
}

#[derive(Copy, Clone, Debug)]
struct TapeValue<'a> {
    tape: &'a Tape<'a>,
    idx: u32,
}

fn iter_rows<'a>(tape: &'a Tape<'a>) -> impl Iterator<Item = TapeValue<'a>> {
    tape.iter_rows().map(move |idx| TapeValue { tape, idx })
}

impl<'a> JsonValue<'a> for TapeValue<'a> {
    fn get(&self) -> JsonType {
        match self.tape.get(self.idx) {
            TapeElement::Null => JsonType::Null,
            TapeElement::False => JsonType::Bool,
            TapeElement::True => JsonType::Bool,
            TapeElement::I64(_) | TapeElement::I32(_) => JsonType::Int64,
            TapeElement::F64(_) | TapeElement::F32(_) => JsonType::Float64,
            TapeElement::Number(s) => {
                if self.tape.get_string(s).parse::<i64>().is_ok() {
                    JsonType::Int64
                } else {
                    JsonType::Float64
                }
            }
            TapeElement::String(_) => JsonType::String,
            TapeElement::StartList(_) => JsonType::Array,
            TapeElement::EndList(_) => unreachable!(),
            TapeElement::StartObject(_) => JsonType::Object,
            TapeElement::EndObject(_) => unreachable!(),
        }
    }

    fn elements(&self) -> impl Iterator<Item = Self> {
        self.tape
            .iter_elements(self.idx)
            .map(move |idx| Self { idx, ..*self })
    }

    fn fields(&self) -> impl Iterator<Item = (&'a str, Self)> {
        self.tape
            .iter_fields(self.idx)
            .map(move |(key, idx)| (key, Self { idx, ..*self }))
    }
}

impl<'a> JsonValue<'a> for &'a serde_json::Value {
    fn get(&self) -> JsonType {
        use serde_json::Value;

        match self {
            Value::Null => JsonType::Null,
            Value::Bool(_) => JsonType::Bool,
            Value::Number(n) => {
                if n.is_i64() {
                    JsonType::Int64
                } else {
                    JsonType::Float64
                }
            }
            Value::String(_) => JsonType::String,
            Value::Array(_) => JsonType::Array,
            Value::Object(_) => JsonType::Object,
        }
    }

    fn elements(&self) -> impl Iterator<Item = Self> {
        use serde_json::Value;

        match self {
            Value::Array(elements) => elements.iter(),
            _ => panic!("Expected an array"),
        }
    }

    fn fields(&self) -> impl Iterator<Item = (&'a str, Self)> {
        use serde_json::Value;

        match self {
            Value::Object(fields) => fields.iter().map(|(key, value)| (key.as_str(), value)),
            _ => panic!("Expected an object"),
        }
    }
}

fn infer_json_type<'a, 't>(
    value: impl JsonValue<'a>,
    expected: &'t InferredTy<'t>,
    arena: &'t Bump,
) -> Result<&'t InferredTy<'t>> {
    let make_err = |got| {
        let expected = match expected {
            InferredTy::Never => unreachable!(),
            InferredTy::Scalar(_) => "a scalar value",
            InferredTy::Array(_) => "an array",
            InferredTy::Object(_) => "an object",
        };
        let msg = format!("Expected {expected}, found {got}");
        ArrowError::JsonError(msg)
    };

    let infer_scalar = |scalar: ScalarTy, got: &'static str| {
        Ok(match expected {
            InferredTy::Never => match scalar {
                ScalarTy::Bool => BOOL_TY,
                ScalarTy::Int64 => INT64_TY,
                ScalarTy::Float64 => FLOAT64_TY,
                ScalarTy::String => STRING_TY,
            },
            InferredTy::Scalar(expect) => match (expect, &scalar) {
                (ScalarTy::Bool, ScalarTy::Bool) => BOOL_TY,
                (ScalarTy::Int64, ScalarTy::Int64) => INT64_TY,
                // Mixed numbers coerce to f64
                (ScalarTy::Int64 | ScalarTy::Float64, ScalarTy::Int64 | ScalarTy::Float64) => {
                    FLOAT64_TY
                }
                // Any other combination coerces to string
                _ => STRING_TY,
            },
            _ => Err(make_err(got))?,
        })
    };

    match value.get() {
        JsonType::Null => Ok(expected),
        JsonType::Bool => infer_scalar(ScalarTy::Bool, "a boolean"),
        JsonType::Int64 => infer_scalar(ScalarTy::Int64, "a number"),
        JsonType::Float64 => infer_scalar(ScalarTy::Float64, "a number"),
        JsonType::String => infer_scalar(ScalarTy::String, "a string"),
        JsonType::Array => {
            println!("zzz: {expected:?} ...");
            let (expected, expected_elem) = match *expected {
                InferredTy::Never => (ARRAY_OF_NEVER_TY, NEVER_TY),
                InferredTy::Array(inner) => (expected, inner),
                _ => Err(make_err("an array"))?,
            };
            println!("zzz: {expected:?} {expected_elem:?}");
            println!("has {:?}", value.elements().collect::<Vec<_>>());

            let elem = value
                .elements()
                .try_fold(expected_elem, |expected, value| {
                    let result = infer_json_type(value, expected, arena);
                    println!("infer array: {expected:?} + {value:?} = {result:?}");
                    result
                })?;

            Ok(if std::ptr::eq(elem, expected_elem) {
                expected
            } else {
                arena.alloc(InferredTy::Array(elem))
            })
        }
        JsonType::Object => {
            let (expected, expected_fields) = match *expected {
                InferredTy::Never => (EMPTY_OBJECT_TY, &[] as &[_]),
                InferredTy::Object(fields) => (expected, fields),
                _ => Err(make_err("an object"))?,
            };

            let mut num_fields = expected_fields.len();
            let mut substs = HashMap::<usize, (&'t str, &'t InferredTy<'t>)>::new();

            for (key, value) in value.fields() {
                let existing_field = expected_fields
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, (existing_key, _))| *existing_key == key);

                match existing_field {
                    Some((field_idx, (key, expect_ty))) => {
                        let ty = infer_json_type(value, expect_ty, arena)?;
                        if !std::ptr::eq(ty, expect_ty) {
                            substs.insert(field_idx, (key, ty));
                        }
                    }
                    None => {
                        let field_idx = num_fields;
                        num_fields += 1;
                        let key = arena.alloc_str(key);
                        let ty = infer_json_type(value, NEVER_TY, arena)?;
                        substs.insert(field_idx, (key, ty));
                    }
                };
            }

            Ok(if substs.is_empty() {
                expected
            } else {
                let fields =
                    arena.alloc_slice_fill_with(num_fields, |idx| match substs.get(&idx) {
                        Some(subst) => *subst,
                        None => expected_fields[idx],
                    });
                arena.alloc(InferredTy::Object(fields))
            })
        }
    }
}

impl<'a> InferredTy<'a> {
    fn into_datatype(self) -> DataType {
        match self {
            Self::Never => DataType::Null,
            Self::Scalar(s) => s.into_datatype(),
            Self::Array(elem) => DataType::List(elem.into_list_field().into()),
            Self::Object(fields) => {
                DataType::Struct(fields.iter().map(|(key, ty)| ty.into_field(*key)).collect())
            }
        }
    }

    fn into_field(self, name: impl Into<String>) -> Field {
        Field::new(name, self.into_datatype(), true)
    }

    fn into_list_field(self) -> Field {
        Field::new_list_field(self.into_datatype(), true)
    }

    fn into_schema(self) -> Result<Schema> {
        let Self::Object(fields) = self else {
            Err(ArrowError::JsonError(format!(
                "Expected JSON object, found {self:?}",
            )))?
        };

        let fields = fields
            .iter()
            .map(|(key, ty)| ty.into_field(*key))
            .collect::<Fields>();

        Ok(Schema::new(fields))
    }
}

impl ScalarTy {
    fn into_datatype(self) -> DataType {
        match self {
            Self::Bool => DataType::Boolean,
            Self::Int64 => DataType::Int64,
            Self::Float64 => DataType::Float64,
            Self::String => DataType::Utf8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::fs::File;
    use std::io::{BufReader, Cursor};
    use std::sync::Arc;

    #[test]
    fn test_json_infer_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", list_type_of(DataType::Float64), true),
            Field::new("c", list_type_of(DataType::Boolean), true),
            Field::new("d", DataType::Utf8, true),
        ]);

        let mut reader = BufReader::new(File::open("test/data/arrays.json").unwrap());
        let (inferred_schema, n_rows) = infer_json_schema_from_seekable(&mut reader, None).unwrap();

        assert_eq!(inferred_schema, schema);
        assert_eq!(n_rows, 3);

        let file = File::open("test/data/arrays.json.gz").unwrap();
        let mut reader = BufReader::new(GzDecoder::new(&file));
        let (inferred_schema, n_rows) = infer_json_schema(&mut reader, None).unwrap();

        assert_eq!(inferred_schema, schema);
        assert_eq!(n_rows, 3);
    }

    #[test]
    fn test_row_limit() {
        let mut reader = BufReader::new(File::open("test/data/basic.json").unwrap());

        let (_, n_rows) = infer_json_schema_from_seekable(&mut reader, None).unwrap();
        assert_eq!(n_rows, 12);

        let (_, n_rows) = infer_json_schema_from_seekable(&mut reader, Some(5)).unwrap();
        assert_eq!(n_rows, 5);
    }

    #[test]
    fn test_json_infer_schema_nested_structs() {
        let schema = Schema::new(vec![
            Field::new(
                "c1",
                DataType::Struct(Fields::from(vec![
                    Field::new("a", DataType::Boolean, true),
                    Field::new(
                        "b",
                        DataType::Struct(vec![Field::new("c", DataType::Utf8, true)].into()),
                        true,
                    ),
                ])),
                true,
            ),
            Field::new("c2", DataType::Int64, true),
            Field::new("c3", DataType::Utf8, true),
        ]);

        let inferred_schema = infer_json_schema_from_iterator(
            vec![
                Ok(serde_json::json!({"c1": {"a": true, "b": {"c": "text"}}, "c2": 1})),
                Ok(serde_json::json!({"c1": {"a": false, "b": null}, "c2": 0})),
                Ok(serde_json::json!({"c1": {"a": true, "b": {"c": "text"}}, "c3": "ok"})),
            ]
            .into_iter(),
        )
        .unwrap();

        assert_eq!(inferred_schema, schema);
    }

    #[test]
    fn test_json_infer_schema_struct_in_list() {
        let schema = Schema::new(vec![
            Field::new(
                "c1",
                list_type_of(DataType::Struct(Fields::from(vec![
                    Field::new("a", DataType::Utf8, true),
                    Field::new("b", DataType::Int64, true),
                    Field::new("c", DataType::Boolean, true),
                ]))),
                true,
            ),
            Field::new("c2", DataType::Float64, true),
            Field::new(
                "c3",
                // empty json array's inner types are inferred as null
                list_type_of(DataType::Null),
                true,
            ),
        ]);

        let inferred_schema = infer_json_schema_from_iterator(
            vec![
                Ok(serde_json::json!({
                    "c1": [{"a": "foo", "b": 100}], "c2": 1, "c3": [],
                })),
                Ok(serde_json::json!({
                    "c1": [{"a": "bar", "b": 2}, {"a": "foo", "c": true}], "c2": 0, "c3": [],
                })),
                Ok(serde_json::json!({"c1": [], "c2": 0.5, "c3": []})),
            ]
            .into_iter(),
        )
        .unwrap();

        assert_eq!(inferred_schema, schema);
    }

    #[test]
    fn test_json_infer_schema_nested_list() {
        let schema = Schema::new(vec![
            Field::new("c1", list_type_of(list_type_of(DataType::Utf8)), true),
            Field::new("c2", DataType::Float64, true),
        ]);

        let inferred_schema = infer_json_schema_from_iterator(
            vec![
                Ok(serde_json::json!({
                    "c1": [],
                    "c2": 12,
                })),
                Ok(serde_json::json!({
                    "c1": [["a", "b"], ["c"]],
                })),
                Ok(serde_json::json!({
                    "c1": [["foo"]],
                    "c2": 0.11,
                })),
            ]
            .into_iter(),
        )
        .unwrap();

        assert_eq!(inferred_schema, schema);
    }

    #[test]
    fn test_infer_json_schema_bigger_than_i64_max() {
        let bigger_than_i64_max = (i64::MAX as i128) + 1;
        let smaller_than_i64_min = (i64::MIN as i128) - 1;
        let json = format!(
            "{{ \"bigger_than_i64_max\": {bigger_than_i64_max}, \"smaller_than_i64_min\": {smaller_than_i64_min} }}",
        );
        let mut buf_reader = BufReader::new(json.as_bytes());
        let (inferred_schema, _) = infer_json_schema(&mut buf_reader, Some(1)).unwrap();
        let fields = inferred_schema.fields();

        let (_, big_field) = fields.find("bigger_than_i64_max").unwrap();
        assert_eq!(big_field.data_type(), &DataType::Float64);
        let (_, small_field) = fields.find("smaller_than_i64_min").unwrap();
        assert_eq!(small_field.data_type(), &DataType::Float64);
    }

    #[test]
    fn test_invalid_json_infer_schema() {
        let re = infer_json_schema_from_seekable(Cursor::new(b"}"), None);
        assert_eq!(
            re.err().unwrap().to_string(),
            "Json error: Encountered unexpected '}' whilst parsing value",
        );
    }

    #[test]
    fn test_null_field_inferred_as_null() {
        let data = r#"
            {"in":1,    "ni":null, "ns":null, "sn":"4",  "n":null, "an":[],   "na": null, "nas":null}
            {"in":null, "ni":2,    "ns":"3",  "sn":null, "n":null, "an":null, "na": [],   "nas":["8"]}
            {"in":1,    "ni":null, "ns":null, "sn":"4",  "n":null, "an":[],   "na": null, "nas":[]}
        "#;
        let (inferred_schema, _) =
            infer_json_schema_from_seekable(Cursor::new(data), None).expect("infer");
        let schema = Schema::new(vec![
            Field::new("in", DataType::Int64, true),
            Field::new("ni", DataType::Int64, true),
            Field::new("ns", DataType::Utf8, true),
            Field::new("sn", DataType::Utf8, true),
            Field::new("n", DataType::Null, true),
            Field::new("an", list_type_of(DataType::Null), true),
            Field::new("na", list_type_of(DataType::Null), true),
            Field::new("nas", list_type_of(DataType::Utf8), true),
        ]);
        assert_eq!(inferred_schema, schema);
    }

    #[test]
    fn test_infer_from_null_then_object() {
        let data = r#"
            {"obj":null}
            {"obj":{"foo":1}}
        "#;
        let (inferred_schema, _) =
            infer_json_schema_from_seekable(Cursor::new(data), None).expect("infer");
        let schema = Schema::new(vec![Field::new(
            "obj",
            DataType::Struct(
                [Field::new("foo", DataType::Int64, true)]
                    .into_iter()
                    .collect(),
            ),
            true,
        )]);
        assert_eq!(inferred_schema, schema);
    }

    /// Shorthand for building list data type of `ty`
    fn list_type_of(ty: DataType) -> DataType {
        DataType::List(Arc::new(Field::new_list_field(ty, true)))
    }
}
