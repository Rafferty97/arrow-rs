// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.kind() (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.kind()
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashMap;

use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use bumpalo::Bump;

/// Abstraction of a JSON value that is only concerned with the
/// type of the value rather than the value itself
pub trait JsonValue<'a> {
    fn get(&self) -> JsonType;

    fn elements(&self) -> impl Iterator<Item = Self>;

    fn fields(&self) -> impl Iterator<Item = (&'a str, Self)>;
}

pub enum JsonType {
    Null,
    Bool,
    Int64,
    Float64,
    String,
    Array,
    Object,
}

#[derive(Clone, Copy, Debug)]
pub struct InferredType<'t>(&'t TyKind<'t>);

#[derive(Clone, Copy, Debug)]
enum TyKind<'t> {
    Never,
    Scalar(ScalarTy),
    Array(InferredType<'t>),
    Object(&'t [(&'t str, InferredType<'t>)]),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScalarTy {
    Bool,
    Int64,
    Float64,
    String,
    // NOTE: Null isn't needed because it's absorbed into Never
}

static NEVER_TY: InferredType<'static> = InferredType(&TyKind::Never);
static BOOL_TY: InferredType<'static> = InferredType(&TyKind::Scalar(ScalarTy::Bool));
static INT64_TY: InferredType<'static> = InferredType(&TyKind::Scalar(ScalarTy::Int64));
static FLOAT64_TY: InferredType<'static> = InferredType(&TyKind::Scalar(ScalarTy::Float64));
static STRING_TY: InferredType<'static> = InferredType(&TyKind::Scalar(ScalarTy::String));
static ARRAY_OF_NEVER_TY: InferredType<'static> = InferredType(&TyKind::Array(NEVER_TY));
static EMPTY_OBJECT_TY: InferredType<'static> = InferredType(&TyKind::Object(&[]));

pub fn never_type() -> InferredType<'static> {
    NEVER_TY
}

pub fn empty_object_type() -> InferredType<'static> {
    EMPTY_OBJECT_TY
}

pub fn infer_json_type<'a, 't>(
    value: impl JsonValue<'a>,
    expected: InferredType<'t>,
    arena: &'t Bump,
) -> Result<InferredType<'t>, ArrowError> {
    let make_err = |got| {
        let expected = match expected.kind() {
            TyKind::Never => unreachable!(),
            TyKind::Scalar(_) => "a scalar value",
            TyKind::Array(_) => "an array",
            TyKind::Object(_) => "an object",
        };
        let msg = format!("Expected {expected}, found {got}");
        ArrowError::JsonError(msg)
    };

    let infer_scalar = |scalar: ScalarTy, got: &'static str| {
        Ok(match expected.kind() {
            TyKind::Never => match scalar {
                ScalarTy::Bool => BOOL_TY,
                ScalarTy::Int64 => INT64_TY,
                ScalarTy::Float64 => FLOAT64_TY,
                ScalarTy::String => STRING_TY,
            },
            TyKind::Scalar(expect) => match (expect, &scalar) {
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
            let (expected, expected_elem) = match *expected.kind() {
                TyKind::Never => (ARRAY_OF_NEVER_TY, NEVER_TY),
                TyKind::Array(inner) => (expected, inner),
                _ => Err(make_err("an array"))?,
            };

            let elem = value
                .elements()
                .try_fold(expected_elem, |expected, value| {
                    let result = infer_json_type(value, expected, arena);
                    result
                })?;

            Ok(if elem.ptr_eq(expected_elem) {
                expected
            } else {
                InferredType::new_array(elem, arena)
            })
        }
        JsonType::Object => {
            let (expected, expected_fields) = match *expected.kind() {
                TyKind::Never => (EMPTY_OBJECT_TY, &[] as &[_]),
                TyKind::Object(fields) => (expected, fields),
                _ => Err(make_err("an object"))?,
            };

            let mut num_fields = expected_fields.len();
            let mut substs = HashMap::<usize, (&'t str, InferredType<'t>)>::new();

            for (key, value) in value.fields() {
                let existing_field = expected_fields
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, (existing_key, _))| *existing_key == key);

                match existing_field {
                    Some((field_idx, (key, expect_ty))) => {
                        let ty = infer_json_type(value, expect_ty, arena)?;
                        if !ty.ptr_eq(expect_ty) {
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

            if substs.is_empty() {
                return Ok(expected);
            }

            let fields = (0..num_fields).map(|idx| match substs.get(&idx) {
                Some(subst) => *subst,
                None => expected_fields[idx],
            });

            Ok(InferredType::new_object(fields, arena))
        }
    }
}

impl<'t> InferredType<'t> {
    fn new_array(inner: InferredType<'t>, arena: &'t Bump) -> Self {
        Self(arena.alloc(TyKind::Array(inner)))
    }

    fn new_object<F>(fields: F, arena: &'t Bump) -> Self
    where
        F: IntoIterator<Item = (&'t str, InferredType<'t>)>,
        F::IntoIter: ExactSizeIterator,
    {
        let fields = arena.alloc_slice_fill_iter(fields);
        Self(arena.alloc(TyKind::Object(fields)))
    }

    fn kind(self) -> &'t TyKind<'t> {
        self.0
    }

    fn ptr_eq(self, other: Self) -> bool {
        std::ptr::eq(self.kind(), other.kind())
    }

    pub fn into_datatype(self) -> DataType {
        match self.kind() {
            TyKind::Never => DataType::Null,
            TyKind::Scalar(s) => s.into_datatype(),
            TyKind::Array(elem) => DataType::List(elem.into_list_field().into()),
            TyKind::Object(fields) => {
                DataType::Struct(fields.iter().map(|(key, ty)| ty.into_field(*key)).collect())
            }
        }
    }

    pub fn into_field(self, name: impl Into<String>) -> Field {
        Field::new(name, self.into_datatype(), true)
    }

    pub fn into_list_field(self) -> Field {
        Field::new_list_field(self.into_datatype(), true)
    }

    pub fn into_schema(self) -> Result<Schema, ArrowError> {
        let TyKind::Object(fields) = self.kind() else {
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
