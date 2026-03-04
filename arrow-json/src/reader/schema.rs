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
use std::io::{BufRead, Seek};
use std::sync::Arc;

use arrow_schema::{ArrowError, Schema};
use serde_json::Value;

use crate::reader::tape::TapeDecoderOptions;

use self::infer::{InferTy, infer_json_type};
use self::json_type::{JsonType, JsonValue, TapeValue};
use super::tape::TapeDecoder;

mod infer;
mod json_type;

/// Configures how to infer the [`Schema`] of a JSON file.
#[derive(Clone, Default)]
pub struct InferJsonSchemaOptions {
    /// The maximum number of records to read to infer the schema.
    ///
    /// If set to `None`, the whole file will be read. Defaults to `None`.
    pub max_read_records: Option<usize>,
    /// The estimated number of fields per row, if known.
    ///
    /// This is purely an optimization hint, with no effect on the output schema.
    pub num_fields: Option<usize>,
    /// Whether to flatten top-level arrays such that their elements
    /// are treated as individual rows.
    ///
    /// See [`ReaderBuilder::with_flatten`] for more detail. Defaults to `false`.
    ///
    /// [`ReaderBuilder::with_flatten`]: super::ReaderBuilder::with_flatten
    pub flatten_top_level_arrays: bool,
    /// Whether to treat each row as the value for a single field in
    /// the resulting schema. This allows each row of the file to contain
    /// any JSON value rather than requiring an object.
    ///
    /// * If set to `Some(name)`, the resulting schema wil contain a single field
    ///   called `name` whose type is the type of each row in the JSON file.
    /// * If set to `None`, then each row of the JSON file is expected to be an object,
    ///   whose keys will become the fields in the resulting schema.
    ///
    /// See [`ReaderBuilder::new_with_field`] for more detail. Defaults to `None`.
    ///
    /// [`ReaderBuilder::new_with_field`]: super::ReaderBuilder::new_with_field
    pub single_field: Option<String>,
}

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
/// let file = File::open("test/data/mixed_arrays.json").unwrap();
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
) -> Result<(Schema, usize), ArrowError> {
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
/// Returns the inferred schema and number of records read.
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
/// let mut file = File::open("test/data/mixed_arrays.json.gz").unwrap();
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
    reader: R,
    max_read_records: Option<usize>,
) -> Result<(Schema, usize), ArrowError> {
    infer_json_schema_with_options(
        reader,
        InferJsonSchemaOptions {
            max_read_records,
            ..Default::default()
        },
    )
}

/// Infer the fields of a JSON file by reading the first n records of the buffer.
///
/// Returns the inferred schema and number of records read.
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
/// use arrow_json::reader::{infer_json_schema_with_options, InferJsonSchemaOptions};
///
/// let mut file = std::fs::File::open("test/data/arrays.json").unwrap();
/// let mut reader = std::io::BufReader::new(file);
///
/// let options = InferJsonSchemaOptions {
///     max_read_records: Some(512),
///     single_field: Some("data".into()),
///     ..Default::default()
/// };
/// let inferred_schema = infer_json_schema_with_options(&mut reader, options).unwrap();
/// ```
pub fn infer_json_schema_with_options<R: BufRead>(
    mut reader: R,
    options: InferJsonSchemaOptions,
) -> Result<(Schema, usize), ArrowError> {
    let mut decoder = SchemaDecoder::new(options);

    loop {
        let buf = reader.fill_buf()?;
        let read = buf.len();

        if read == 0 {
            break;
        }

        let decoded = decoder.decode(buf)?;
        reader.consume(decoded);

        if decoded == 0 || decoder.has_read_max_records() {
            break;
        }
    }

    decoder.finish()
}

/// Infer the fields of a JSON file by reading all items from the JSON Value Iterator.
///
/// The following type coercion logic is implemented:
/// * `Int64` and `Float64` are converted to `Float64`
/// * Lists and scalars are coerced to a list of a compatible scalar
/// * All other cases are coerced to `Utf8` (String)
///
/// Note that the above coercion logic is different from what Spark has, where it would default to
/// String type in case of List and Scalar values appeared in the same field.
///
/// The reason we diverge here is because we don't have utilities to deal with JSON data once it's
/// interpreted as Strings. We should match Spark's behavior once we added more JSON parsing
/// kernels in the future.
pub fn infer_json_schema_from_iterator<I, V>(value_iter: I) -> Result<Schema, ArrowError>
where
    I: Iterator<Item = Result<V, ArrowError>>,
    V: Borrow<Value>,
{
    value_iter
        .into_iter()
        .try_fold(InferTy::any(), |ty, record| {
            infer_json_type(record?.borrow(), ty)
        })?
        .into_schema()
}

struct SchemaDecoder {
    decoder: TapeDecoder,
    single_field: Option<String>,
    max_read_records: usize,
    record_count: usize,
    schema: InferTy,
}

impl SchemaDecoder {
    pub fn new(options: InferJsonSchemaOptions) -> Self {
        let max_read_records = options.max_read_records.unwrap_or(usize::MAX);

        let decoder = TapeDecoder::new(TapeDecoderOptions {
            // Use a sensible batch size, capped to `max_read_records`
            batch_size: max_read_records.min(1024),
            num_fields: 8,
            flatten_top_level_arrays: options.flatten_top_level_arrays,
        });

        let schema = match &options.single_field {
            Some(_) => InferTy::Any,
            None => InferTy::empty_object(),
        };

        Self {
            decoder,
            single_field: options.single_field,
            max_read_records,
            record_count: 0,
            schema,
        }
    }

    pub fn decode(&mut self, buf: &[u8]) -> Result<usize, ArrowError> {
        let read = self.decoder.decode(buf)?;
        if read != buf.len() {
            self.infer_batch()?;
        }
        Ok(read)
    }

    pub fn has_read_max_records(&self) -> bool {
        self.record_count >= self.max_read_records
    }

    pub fn finish(mut self) -> Result<(Schema, usize), ArrowError> {
        self.infer_batch()?;
        match self.single_field {
            Some(name) => {
                let field = Arc::new(self.schema.to_field(name));
                Ok((Schema::new([field]), self.record_count))
            }
            None => Ok((self.schema.into_schema()?, self.record_count)),
        }
    }

    fn infer_batch(&mut self) -> Result<(), ArrowError> {
        let tape = self.decoder.finish()?;

        let records = tape
            .iter_rows()
            .map(|idx| TapeValue::new(&tape, idx))
            .take(self.max_read_records - self.record_count);

        for record in records {
            if record.get() == JsonType::Null {
                Err(ArrowError::JsonError(format!("top-level null")))?
            }
            self.schema = infer_json_type(record, self.schema.clone())?;
            self.record_count += 1;
        }

        self.decoder.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;
    use std::io::{BufReader, Cursor};
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Fields};
    use flate2::read::GzDecoder;

    #[test]
    fn test_json_infer_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", list_type_of(DataType::Float64), true),
            Field::new("c", list_type_of(DataType::Boolean), true),
            Field::new("d", list_type_of(DataType::Utf8), true),
        ]);

        let mut reader = BufReader::new(File::open("test/data/mixed_arrays.json").unwrap());
        let (inferred_schema, n_rows) = infer_json_schema_from_seekable(&mut reader, None).unwrap();

        assert_eq!(inferred_schema, schema);
        assert_eq!(n_rows, 4);

        let file = File::open("test/data/mixed_arrays.json.gz").unwrap();
        let mut reader = BufReader::new(GzDecoder::new(&file));
        let (inferred_schema, n_rows) = infer_json_schema(&mut reader, None).unwrap();

        assert_eq!(inferred_schema, schema);
        assert_eq!(n_rows, 4);
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
        assert!(re.err().unwrap().to_string().contains("Json error:"));
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
            DataType::Struct(std::iter::once(Field::new("foo", DataType::Int64, true)).collect()),
            true,
        )]);
        assert_eq!(inferred_schema, schema);
    }

    #[test]
    fn test_read_all_rows() {
        let mut data = "{}\n".repeat(2048);
        data.push_str("{\"late\":true}\n");
        let (schema, _) = infer_json_schema(Cursor::new(data), None).unwrap();
        assert_eq!(
            schema.field(0),
            &Field::new("late", DataType::Boolean, true)
        );
    }

    #[test]
    fn test_empty_input() {
        let values = std::iter::empty::<Result<Value, ArrowError>>();
        let schema = infer_json_schema_from_iterator(values).unwrap();
        assert_eq!(schema, Schema::empty());
    }

    #[test]
    fn test_ignore_trailing_invalid() {
        let data = b"{\"a\":1}\nthis is not JSON\n";
        let (schema, record_count) = infer_json_schema(Cursor::new(data), Some(1)).unwrap();
        assert_eq!(record_count, 1);
        assert_eq!(schema.field(0), &Field::new("a", DataType::Int64, true));
    }

    #[test]
    fn test_infer_empty_array() {
        let (inferred_schema, _) = infer_json_schema_with_options(
            b"[]" as &[u8],
            InferJsonSchemaOptions {
                flatten_top_level_arrays: true,
                ..Default::default()
            },
        )
        .unwrap();
        let schema = Schema::new([]);
        assert_eq!(inferred_schema, schema);
    }

    /// Shorthand for building list data type of `ty`
    fn list_type_of(ty: DataType) -> DataType {
        DataType::List(Arc::new(Field::new_list_field(ty, true)))
    }
}
