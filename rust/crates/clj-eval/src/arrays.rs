// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Java arrays (S15): the 35 array natives (`amap`/`areduce` are
//! `core.clj` macros). Constructors route through `Numbers` rules
//! (narrowing fills, exact-wrapper inits), access through the
//! `RT.aset`/`RT.aget` overload shape (direct first step, raw
//! `Array.get`/`set` tails), and `vector-of` through the per-type
//! core coercions. Oracle: `docs/zero-oos-s15-arrays/probes/`.

use std::sync::{Arc, LazyLock};

use clj_value::{
    Array, ArrayDesc, ArrayStore, BaseType, Elem8, MAX_ARRAY_BYTES, PrimitiveVector, ThrowClass,
    Value,
};
use num_traits::ToPrimitive;

use crate::core::{
    Num, bigint_int_value, cce_err, double_long, int_cast_double, int_cast_value, int_exact_long,
    native_byte, native_char, native_double, native_float, native_long, native_short,
    npe_char_value, npe_double_value, npe_long_value, nth_index, out_of_range, tower_intcast,
};
use crate::error::{NativeError, jvm_class_of};
use crate::eval::Evaluator;
use crate::lazy::{as_subvector, collect_all, is_seqable};
use crate::ns::{NativeFnPtr, check_arity, signal_to_native};

/// The S15 natives installed into `clojure.core` by [`boot`](crate::ns::boot).
pub(crate) const ARRAY_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("aclone", native_aclone),
    ("aget", native_aget),
    ("alength", native_alength),
    ("aset", native_aset),
    ("aset-boolean", native_aset_boolean),
    ("aset-byte", native_aset_byte),
    ("aset-char", native_aset_char),
    ("aset-double", native_aset_double),
    ("aset-float", native_aset_float),
    ("aset-int", native_aset_int),
    ("aset-long", native_aset_long),
    ("aset-short", native_aset_short),
    ("boolean-array", native_boolean_array),
    ("booleans", native_booleans),
    ("byte-array", native_byte_array),
    ("bytes", native_bytes),
    ("bytes?", native_bytes_q),
    ("char-array", native_char_array),
    ("chars", native_chars),
    ("double-array", native_double_array),
    ("doubles", native_doubles),
    ("float-array", native_float_array),
    ("floats", native_floats),
    ("int-array", native_int_array),
    ("into-array", native_into_array),
    ("ints", native_ints),
    ("long-array", native_long_array),
    ("longs", native_longs),
    ("make-array", native_make_array),
    ("object-array", native_object_array),
    ("short-array", native_short_array),
    ("shorts", native_shorts),
    ("to-array", native_to_array),
    ("to-array-2d", native_to_array_2d),
    ("vector-of", native_vector_of),
];

/// `IAE | More than one matching method found: {name}` (nil
/// through the `RT` overloads — oracle `a-nil`, `b-nil`).
fn more_than_one(name: &str) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("More than one matching method found: {name}"),
    )
}

/// `IAE | No matching method {name} found taking {n} args`
/// (non-arrays, and direct-`aset` value misses — oracles
/// `a-nonarray`, `b-narrow`, `g-clone-nonarray`).
fn no_match(name: &str, n: usize) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("No matching method {name} found taking {n} args"),
    )
}

/// `AIOOBE | Index {i} out of bounds for length {n}` (the VM
/// bounds text — oracles `a-oob`, `b-oob`).
fn aioobe_index(i: i32, n: usize) -> NativeError {
    NativeError::new(
        ThrowClass::ArrayIndexOutOfBoundsException,
        format!("Index {i} out of bounds for length {n}"),
    )
}

/// The empty-message `AIOOBE` (`Array.setX` bounds, `nth` — oracles
/// `c-int-oob`, `a-nth-oob`).
fn aioobe_empty() -> NativeError {
    NativeError::new(ThrowClass::ArrayIndexOutOfBoundsException, String::new())
}

/// `NegativeArraySizeException | {n}` (the size renders raw —
/// oracles `d-neg-size`, `e-neg-dim`).
fn neg_size(n: i32) -> NativeError {
    NativeError::new(ThrowClass::NegativeArraySizeException, n.to_string())
}

/// The empty-message `OutOfMemoryError` (the 256 MiB allocation
/// guard — port behavior, spec §Design).
fn oom() -> NativeError {
    NativeError::new(ThrowClass::OutOfMemoryError, String::new())
}

/// The empty-message `NPE` (`Array.set` on nil, `make-array` nil
/// class — oracles `b-aset-nilinner`, `e-nil-class`).
fn npe_empty() -> NativeError {
    NativeError::new(ThrowClass::NullPointerException, String::new())
}

/// The multi-`aget` nil-inner `NPE` (the 2-arity body's `(class
/// array)` call, inlined — oracle `a-nil-inner`, verbatim).
fn npe_inner_class() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Class.getComponentType()\" because the return value of \"clojure.core$class.invokeStatic(Object)\" is null".to_owned(),
    )
}

/// The `to-array-2d` nil `NPE` (the `^Collection` hint passes nil,
/// `.size` fails — oracle `g-to2d-nil`, verbatim).
fn npe_coll_size() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.util.Collection.size()\" because \"coll\" is null".to_owned(),
    )
}

/// `IAE | Argument is not an array` (raw tails — oracles
/// `a-deep-nonarray`, `e-setint-vec`).
fn not_an_array() -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        "Argument is not an array".to_owned(),
    )
}

/// `IAE | Argument is not an array of primitive type`
/// (`aset-*` on refs/nesting — oracles `c-int-objarr`,
/// `b-setint-2d`).
fn not_primitive() -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        "Argument is not an array of primitive type".to_owned(),
    )
}

/// `IAE | argument type mismatch` (primitive `Array.set`/`setX`
/// misses — oracles `c-int-wrongarr`, `e-tail-int-long`).
fn arg_mismatch() -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        "argument type mismatch".to_owned(),
    )
}

/// `IAE | array element type mismatch` (reference `Array.set`
/// misses, `into-array` checks — oracles `f-into-mixed`,
/// `c-long-double` is the primitive sibling).
fn elem_mismatch() -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        "array element type mismatch".to_owned(),
    )
}

/// The empty-message `IAE` (nil into a primitive via `Array.set` —
/// oracles `c-long-nil`, `c-double-nil`).
fn iae_empty() -> NativeError {
    NativeError::new(ThrowClass::IllegalArgumentException, String::new())
}

/// `ArrayStoreException | {jvm class of val}` (direct stores into
/// refs — oracles `b-store-obj`, `b-store-2d`).
fn array_store(v: &Value) -> NativeError {
    NativeError::new(ThrowClass::ArrayStoreException, jvm_class_of(v))
}

/// `RuntimeException | Unable to convert: class X to Object[]`
/// (`to-array` of junk — oracles `d-num`, `g-to2d-num`).
fn bad_convert(v: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::RuntimeException,
        format!("Unable to convert: class {} to Object[]", jvm_class_of(v)),
    )
}

/// `IAE | Unrecognized type {str}` (`vector-of` of anything but
/// the eight keywords — oracle `f-vec-badtype`).
fn unrecognized(eval: &mut Evaluator, t: &Value) -> Result<NativeError, NativeError> {
    let s = crate::core::str_of_one(eval, t)?;
    Ok(NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("Unrecognized type {s}"),
    ))
}

/// `RT.EMPTY_ARRAY`: the one shared empty `Object[]` (`to-array`
/// of nil answers this same handle — oracle `e-to-nil-ident`).
/// Length-zero arrays admit no successful `aset`, so sharing is
/// unobservable beyond `identical?`.
static EMPTY_OBJ_ARRAY: LazyLock<Array> = LazyLock::new(|| {
    Array::new(
        ArrayDesc::flat(BaseType::Object),
        ArrayStore::Objs(Vec::new()),
    )
});

/// The allocation guard: `slots` slots of `base` whose payload
/// (or a failed `try_reserve`) throws `OutOfMemoryError` instead
/// of aborting (port behavior, spec §Design).
fn guard_alloc(base: BaseType, slots: usize) -> Result<(), NativeError> {
    let bytes = (slots as u64).saturating_mul(base.slot_bytes());
    if bytes > MAX_ARRAY_BYTES {
        return Err(oom());
    }
    Ok(())
}

/// Realize a seqable to items (`RT.seq` + full walk): arrays
/// snapshot (boxing reads), nil walks empty, anything else
/// non-seqable keeps the port's `ISeq` failure (oracles
/// `b-int-true`, `d-int-boolinit`).
fn seq_items(eval: &mut Evaluator, v: &Value) -> Result<Vec<Value>, NativeError> {
    match v {
        Value::Array(a) => Ok(a.snapshot()),
        _ => collect_all(eval, v).map_err(signal_to_native),
    }
}

/// Whether `to-array` walks the value: the seqables plus the two
/// vector-likes `is_seqable` omits (subvectors, eductions — both
/// `Iterable`, like the JVM).
fn is_toarray_walkable(v: &Value) -> bool {
    is_seqable(v) || as_subvector(v).is_some() || crate::xform::Eduction::parts(v).is_some()
}

/// Whether `to-array-2d` accepts the value: the `^Collection`
/// hint — vectors/lists/sets/entries/PVs and every seq shape,
/// but never nil (the hint passes it to the `.size` NPE), maps,
/// strings, arrays, records, or scalars (oracles `t2d-map`,
/// `t2d-str`, `g-to2d-nil`).
fn is_collection(v: &Value) -> bool {
    match v {
        Value::Vector(_)
        | Value::List(_)
        | Value::Set(_)
        | Value::MapEntry { .. }
        | Value::PrimitiveVector(_) => true,
        // Maps, strings, arrays, records, and scalars fail the
        // hint even when seqable (a record is a map, not a
        // `Collection`).
        Value::Nil
        | Value::Map(_)
        | Value::Str(_)
        | Value::Array(_)
        | Value::HostType(_)
        | Value::MultiFn(_) => false,
        Value::Extension(_) => is_seqable(v) || as_subvector(v).is_some(),
        _ => false,
    }
}

/// The `Numbers` fill `NPE`: `Cannot invoke
/// "java.lang.{T}.{m}()" because the return value of
/// "clojure.lang.ISeq.first()" is null` (unboxing a nil seq
/// element — oracles `d-int-of-nil`, `b-double-nil`,
/// `b-bool-nil`, `e-char-nil`).
fn npe_fill_method(recv: &str, method: &str) -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        format!(
            "Cannot invoke \"java.lang.{recv}.{method}()\" because the return value of \"clojure.lang.ISeq.first()\" is null"
        ),
    )
}

/// Low 64 bits, sign-extended (`longValue` narrowing — the
/// `bigint_int_value` shape one width up).
fn bigint_long_value(n: &num_bigint::BigInt) -> i64 {
    let mut le = n.to_signed_bytes_le();
    le.resize(
        8,
        if n.sign() == num_bigint::Sign::Minus {
            0xFF
        } else {
            0
        },
    );
    i64::from_le_bytes([le[0], le[1], le[2], le[3], le[4], le[5], le[6], le[7]])
}

/// An `intValue` fill: low-32 narrowing, doubles saturating
/// (`NaN` → 0 — Rust's saturating cast is Java's), tower through
/// truncation; chars and anything else non-numeric fail to
/// `Number`, nil trips the `intValue` fill `NPE` (oracles
/// `b-byte-trunc`, `b-int-wrap`, `d-int-of-nil`, `b-int-bool`).
fn num_int_narrow(v: &Value) -> Result<i32, NativeError> {
    match v {
        Value::Int(i) => Ok(*i as i32),
        Value::Double(d) => Ok(*d as i32),
        Value::BigInt(n) | Value::BigInteger(n) => Ok(bigint_int_value(n)),
        Value::Ratio(r) => Ok(bigint_int_value(&(r.numer() / r.denom()))),
        Value::BigDec(d) => Ok(bigint_int_value(&d.to_bigint_trunc())),
        Value::Nil => Err(npe_fill_method("Number", "intValue")),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

/// A `longValue` fill (oracles `b-long-mixed`, `b-long-huge`,
/// `b-long-str`).
fn num_long_narrow(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Double(d) => Ok(*d as i64),
        Value::BigInt(n) | Value::BigInteger(n) => Ok(bigint_long_value(n)),
        Value::Ratio(r) => Ok(bigint_long_value(&(r.numer() / r.denom()))),
        Value::BigDec(d) => Ok(bigint_long_value(&d.to_bigint_trunc())),
        Value::Nil => Err(npe_fill_method("Number", "longValue")),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

/// A `shortValue`/`byteValue` fill: low-16/8 of the `intValue`
/// narrowing (oracles `b-short-wrap`, `b-byte-trunc`,
/// `b-byte-neg`).
fn num_int_narrow_then<T>(
    v: &Value,
    method: &str,
    f: impl FnOnce(i32) -> T,
) -> Result<T, NativeError> {
    match v {
        Value::Nil => Err(npe_fill_method("Number", method)),
        _ => num_int_narrow(v).map(f),
    }
}

/// A `floatValue`/`doubleValue` fill: the exact `f64` (tower via
/// `to_f64`, `NaN` past range — the port keeps `float` exact per
/// D3, so both widths share this; oracles `b-float-seq`,
/// `b-double-seq`, `b-double-nil`, `b-float-inf`).
fn num_double_narrow(v: &Value, method: &str) -> Result<f64, NativeError> {
    match v {
        Value::Int(i) => Ok(*i as f64),
        Value::Double(d) => Ok(*d),
        Value::BigInt(n) | Value::BigInteger(n) => Ok(n.to_f64().unwrap_or(f64::NAN)),
        Value::Ratio(r) => Ok(r.to_f64().unwrap_or(f64::NAN)),
        Value::BigDec(d) => Ok(d.to_f64()),
        Value::Nil => Err(npe_fill_method("Number", method)),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

/// A `char[]` fill: the strict `(Character)` unbox (oracles
/// `d-char-of-int`, `d-char-of-str`, `e-char-nil`, `b-char-str`).
fn char_fill(v: &Value) -> Result<char, NativeError> {
    match v {
        Value::Char(c) => Ok(*c),
        Value::Nil => Err(npe_fill_method("Character", "charValue")),
        _ => Err(cce_err(v, "java.lang.Character")),
    }
}

/// A `boolean[]` fill: the strict `(Boolean)` unbox (oracles
/// `b-bool-seq`, `b-bool-nil`).
fn bool_fill(v: &Value) -> Result<bool, NativeError> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Nil => Err(npe_fill_method("Boolean", "booleanValue")),
        _ => Err(cce_err(v, "java.lang.Boolean")),
    }
}

/// A range failure rendering the ORIGINAL value (the `*Cast`
/// range checks name `x`, not the narrowed `long` — e.g.
/// `300.5`, or a bare char). The wildcard renders anything
/// else's `Debug` (production callers only pass numbers/chars
/// past the `long` stop, so it is covered directly).
fn oor_val(kind: &str, v: &Value) -> NativeError {
    let x = match v {
        Value::Int(i) => i.to_string(),
        Value::Double(d) => clj_value::java_double_string(*d),
        Value::BigInt(n) | Value::BigInteger(n) => n.to_string(),
        Value::Ratio(r) => format!("{r}"),
        Value::BigDec(d) => d.to_bigdec_string(),
        Value::Char(c) => c.to_string(),
        _ => format!("{v:?}"),
    };
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("Value out of range for {kind}: {x}"),
    )
}

/// `RT.intCast` over a value (`into-array` `Integer/TYPE` and
/// multi-arity walk indices — NOT core-`int`: doubles and
/// decimals take the `long` stop, chars convert, strings fail
/// to `Character`; oracles `f-into-typed`,
/// `f-into-typemismatch`, `a-into-huge`, `m4-*`, `m3-idx1dbl`).
fn intcast_value(v: &Value) -> Result<i32, NativeError> {
    match v {
        Value::Int(i) => int_exact_long(*i),
        Value::Double(d) => int_cast_double(*d),
        Value::Char(c) => Ok(*c as i32),
        Value::BigInt(n) | Value::BigInteger(n) => tower_intcast(n),
        Value::Ratio(r) => tower_intcast(&(r.numer() / r.denom())),
        Value::BigDec(d) => int_exact_long(double_long(d.to_f64())?),
        Value::Nil => Err(npe_char_value()),
        _ => Err(cce_err(v, "java.lang.Character")),
    }
}

/// A `longValue` with the `long` stop (`byteCast`/`shortCast`/
/// `charCast` route everything through `longCast` first).
fn longstop_value(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Double(d) => double_long(*d),
        Value::Char(c) => Ok(*c as i64),
        Value::BigInt(n) | Value::BigInteger(n) => n
            .to_i64()
            .ok_or_else(|| out_of_range("long", Num::BigInt(n.as_ref().clone()))),
        Value::Ratio(r) => {
            let t = r.numer() / r.denom();
            t.to_i64()
                .ok_or_else(|| out_of_range("long", Num::BigInt(t.clone())))
        }
        Value::BigDec(d) => double_long(d.to_f64()),
        Value::Nil => Err(npe_double_value()),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

/// `RT.byteCast`/`shortCast` over a value (`into-array`
/// `Byte/TYPE`/`Short/TYPE` — the `long` stop, then the width
/// range rendering the original; oracles `a-byte-*`,
/// `a-short-*`).
fn narrow_cast_value(v: &Value, kind: &'static str, lo: i64, hi: i64) -> Result<i64, NativeError> {
    let l = longstop_value(v)?;
    if l < lo || l > hi {
        return Err(oor_val(kind, v));
    }
    Ok(l)
}

/// `RT.charCast` over a value (`into-array` `Character/TYPE` —
/// the `long` stop, then the `u16` range; lone surrogates fail
/// like the port's `char` posture; oracles `a-char-*`).
fn charcast_value(v: &Value) -> Result<char, NativeError> {
    if let Value::Char(c) = v {
        return Ok(*c);
    }
    if matches!(v, Value::Nil) {
        return Err(npe_long_value());
    }
    let l = longstop_value(v)?;
    if !(0..=0xFFFF).contains(&l) {
        return Err(oor_val("char", v));
    }
    char::from_u32(l as u32).ok_or_else(|| oor_val("char", v))
}

/// Core-`int` of a value (direct idx, dims, `aset-int` — the
/// answer kind is the native's own). Calls the inner cast
/// directly: `native_int` is the 1-arity check (always satisfied
/// here) plus `int_cast_value` wrapped in `Int`, so errors are
/// identical and no variant unwrap remains.
fn core_int(_eval: &mut Evaluator, v: &Value) -> Result<i64, NativeError> {
    int_cast_value(v)
}

/// Core-`long` of a value (`aset-long`). The native answers
/// `Int` by construction; re-narrowing through `longstop_value`
/// is total on that shape (identity, infallible), so the
/// single-variant match and its untestable fallback are gone.
fn core_long(eval: &mut Evaluator, v: &Value) -> Result<i64, NativeError> {
    longstop_value(&native_long(eval, std::slice::from_ref(v))?)
}

/// Core-`double` of a value (`aset-double`). The native answers
/// `Double` by construction; re-narrowing is total on that
/// shape (identity, infallible).
fn core_double(eval: &mut Evaluator, v: &Value) -> Result<f64, NativeError> {
    num_double_narrow(
        &native_double(eval, std::slice::from_ref(v))?,
        "doubleValue",
    )
}

/// Core-`float` of a value (`aset-float`, `into-array` float).
/// The native answers `Double` by construction; re-narrowing is
/// total on that shape (identity, infallible).
fn core_float(eval: &mut Evaluator, v: &Value) -> Result<f64, NativeError> {
    num_double_narrow(&native_float(eval, std::slice::from_ref(v))?, "floatValue")
}

/// Core-`short` of a value (`aset-short`, D2 direct stores). The
/// native answers an in-range `Int` by construction;
/// re-narrowing through `int_cast_value` is total on that shape
/// (identity, infallible).
fn core_short(eval: &mut Evaluator, v: &Value) -> Result<i64, NativeError> {
    int_cast_value(&native_short(eval, std::slice::from_ref(v))?)
}

/// Core-`byte` of a value (`aset-byte`, D2 direct stores). The
/// native answers an in-range `Int` by construction;
/// re-narrowing through `int_cast_value` is total on that shape
/// (identity, infallible).
fn core_byte(eval: &mut Evaluator, v: &Value) -> Result<i64, NativeError> {
    int_cast_value(&native_byte(eval, std::slice::from_ref(v))?)
}

/// Core-`char` of a value (`aset-char`). The native answers
/// `Char` by construction; re-narrowing through `char_fill` is
/// total on that shape (identity, infallible).
fn core_char(eval: &mut Evaluator, v: &Value) -> Result<char, NativeError> {
    char_fill(&native_char(eval, std::slice::from_ref(v))?)
}

/// Whether the value is a `Number` (the ctor size/init dispatch —
/// chars are not numbers).
fn is_number(v: &Value) -> bool {
    matches!(
        v,
        Value::Int(_)
            | Value::Double(_)
            | Value::BigInt(_)
            | Value::BigInteger(_)
            | Value::Ratio(_)
            | Value::BigDec(_)
    )
}

/// A zeroed allocation (the allocation guard runs first; a
/// failed reserve OOMs instead of aborting). Plain function, not
/// a macro, so the reserve-failure closure shares its line with
/// the always-executed call.
fn zero_vec<T: Clone>(base: BaseType, size: usize, zero: T) -> Result<Vec<T>, NativeError> {
    guard_alloc(base, size)?;
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(size).map_err(|_| oom())?;
    v.resize(size, zero);
    Ok(v)
}

/// A zeroed store (the allocation guard runs first; a failed
/// reserve OOMs instead of aborting). Split by width for the
/// complexity gate (nine fallible arms trip cyclomatic 19).
fn zero_store(base: BaseType, size: usize) -> Result<ArrayStore, NativeError> {
    Ok(match base {
        BaseType::Boolean => ArrayStore::Bools(zero_vec(base, size, false)?),
        BaseType::Byte => ArrayStore::Bytes(zero_vec(base, size, 0_i8)?),
        BaseType::Char => ArrayStore::Chars(zero_vec(base, size, '\0')?),
        BaseType::Short => ArrayStore::Shorts(zero_vec(base, size, 0_i16)?),
        BaseType::Int => ArrayStore::Ints(zero_vec(base, size, 0_i32)?),
        wide => zero_store_wide(wide, size)?,
    })
}

/// Wide/object zeroed stores (see `zero_store`).
fn zero_store_wide(base: BaseType, size: usize) -> Result<ArrayStore, NativeError> {
    Ok(match base {
        BaseType::Long => ArrayStore::Longs(zero_vec(base, size, 0_i64)?),
        BaseType::Float => ArrayStore::Floats(zero_vec(base, size, 0.0_f64)?),
        BaseType::Double => ArrayStore::Doubles(zero_vec(base, size, 0.0_f64)?),
        BaseType::Object => ArrayStore::Objs(zero_vec(base, size, Value::Nil)?),
        _ => unreachable!("narrow types stay in zero_store"),
    })
}

/// Fill a zeroed store from items, truncated/padded (`Numbers`
/// 2-arity seq-fill — errors partway discard the array, like the
/// JVM).
fn fill_store(base: BaseType, size: usize, items: &[Value]) -> Result<ArrayStore, NativeError> {
    macro_rules! fill {
        ($variant:ident, $narrow:expr) => {{
            let mut v = match zero_store(base, size)? {
                ArrayStore::$variant(v) => v,
                _ => unreachable!("zeroed variant"),
            };
            for (slot, item) in v.iter_mut().zip(items.iter()) {
                *slot = $narrow(item)?;
            }
            Ok(ArrayStore::$variant(v))
        }};
    }
    match base {
        BaseType::Boolean => fill!(Bools, bool_fill),
        BaseType::Byte => fill!(Bytes, |v: &Value| num_int_narrow_then(
            v,
            "byteValue",
            |i| i as i8
        )),
        BaseType::Char => fill!(Chars, char_fill),
        BaseType::Short => fill!(Shorts, |v: &Value| num_int_narrow_then(
            v,
            "shortValue",
            |i| i as i16
        )),
        BaseType::Int => fill!(Ints, num_int_narrow),
        BaseType::Long => fill!(Longs, num_long_narrow),
        BaseType::Float => fill!(Floats, |v: &Value| num_double_narrow(v, "floatValue")),
        BaseType::Double => fill!(Doubles, |v: &Value| num_double_narrow(v, "doubleValue")),
        BaseType::Object => {
            let mut v: Vec<Value> = zero_vec(base, size, Value::Nil)?;
            for (slot, item) in v.iter_mut().zip(items.iter()) {
                *slot = item.clone();
            }
            Ok(ArrayStore::Objs(v))
        }
    }
}

/// The 1-arity `Numbers` constructor: `Number` sizes narrow via
/// `intValue` (negative → `NegativeArraySizeException`), anything
/// else seq-fills (nil → empty; oracles `a-*`, `b-*`, `d-*`).
fn ctor_seq(
    eval: &mut Evaluator,
    base: BaseType,
    size_or_seq: &Value,
) -> Result<Value, NativeError> {
    if is_number(size_or_seq) {
        // `num_int_narrow` cannot fail a `Number`.
        let n = num_int_narrow(size_or_seq)?;
        if n < 0 {
            return Err(neg_size(n));
        }
        let store = zero_store(base, n as usize)?;
        return Ok(Value::Array(Array::new(ArrayDesc::flat(base), store)));
    }
    // `Object` fills through the same store (a full-length
    // fill is exactly `Objs(items)`, behind the same guard).
    let items = seq_items(eval, size_or_seq)?;
    let store = fill_store(base, items.len(), &items)?;
    Ok(Value::Array(Array::new(ArrayDesc::flat(base), store)))
}

/// Whether a 2-arity `Numbers` constructor init scalar-fills:
/// `int`/`long`/`float`/`double` fill `Number` inits,
/// `boolean`/`char` fill exact wrappers, `byte`/`short` always
/// seq-fill (the port erased the wrappers; oracles `c-*`,
/// `b-*-longseq`, `d-*-init`).
fn ctor_is_scalar(base: BaseType, init: &Value) -> bool {
    match base {
        BaseType::Int | BaseType::Long | BaseType::Float | BaseType::Double => is_number(init),
        BaseType::Boolean => matches!(init, Value::Bool(_)),
        BaseType::Char => matches!(init, Value::Char(_)),
        BaseType::Byte | BaseType::Short | BaseType::Object => false,
    }
}

/// Scalar-fill an `int[]` (the `intValue` narrowing).
fn ctor_scalar_ints(base: BaseType, size: usize, init: &Value) -> Result<ArrayStore, NativeError> {
    let f = num_int_narrow(init)?;
    let mut v: Vec<i32> = zero_vec(base, size, 0)?;
    v.fill(f);
    Ok(ArrayStore::Ints(v))
}

/// Scalar-fill a `long[]` (the `longValue` narrowing).
fn ctor_scalar_longs(base: BaseType, size: usize, init: &Value) -> Result<ArrayStore, NativeError> {
    let f = num_long_narrow(init)?;
    let mut v: Vec<i64> = zero_vec(base, size, 0)?;
    v.fill(f);
    Ok(ArrayStore::Longs(v))
}

/// Scalar-fill a `float[]`/`double[]` (the exact `f64` — the port
/// keeps `float` exact per D3, so both widths share this).
fn ctor_scalar_floats(
    base: BaseType,
    size: usize,
    init: &Value,
) -> Result<ArrayStore, NativeError> {
    let method = if base == BaseType::Float {
        "floatValue"
    } else {
        "doubleValue"
    };
    let f = num_double_narrow(init, method)?;
    let mut v: Vec<f64> = zero_vec(base, size, 0.0)?;
    v.fill(f);
    if base == BaseType::Float {
        Ok(ArrayStore::Floats(v))
    } else {
        Ok(ArrayStore::Doubles(v))
    }
}

/// Scalar-fill a `boolean[]` (the strict `(Boolean)` unbox).
fn ctor_scalar_bools(base: BaseType, size: usize, init: &Value) -> Result<ArrayStore, NativeError> {
    let f = bool_fill(init)?;
    let mut v: Vec<bool> = zero_vec(base, size, false)?;
    v.fill(f);
    Ok(ArrayStore::Bools(v))
}

/// Scalar-fill a `char[]` (the strict `(Character)` unbox).
fn ctor_scalar_chars(base: BaseType, size: usize, init: &Value) -> Result<ArrayStore, NativeError> {
    let f = char_fill(init)?;
    let mut v: Vec<char> = zero_vec(base, size, '\0')?;
    v.fill(f);
    Ok(ArrayStore::Chars(v))
}

/// Scalar-fill one store past the [`ctor_is_scalar`] gate (one arm
/// per fill family).
fn ctor_scalar_store(base: BaseType, size: usize, init: &Value) -> Result<ArrayStore, NativeError> {
    match base {
        BaseType::Int => ctor_scalar_ints(base, size, init),
        BaseType::Long => ctor_scalar_longs(base, size, init),
        BaseType::Float | BaseType::Double => ctor_scalar_floats(base, size, init),
        BaseType::Boolean => ctor_scalar_bools(base, size, init),
        // The scalar gate admits only `Char` past the earlier
        // arms (`Byte`/`Short`/`Object` never scalar-fill,
        // asserted in debug builds).
        _ => {
            debug_assert_eq!(base, BaseType::Char);
            ctor_scalar_chars(base, size, init)
        }
    }
}

/// The 2-arity `Numbers` constructor: sizes through
/// `checkCast`+`intCast` (`nth_index`), then scalar-or-seq init
/// per base (see [`ctor_is_scalar`]).
fn ctor_init(
    eval: &mut Evaluator,
    base: BaseType,
    size: i32,
    init: &Value,
) -> Result<Value, NativeError> {
    if size < 0 {
        return Err(neg_size(size));
    }
    let size = size as usize;
    if !ctor_is_scalar(base, init) {
        let items = seq_items(eval, init)?;
        let store = fill_store(base, size, &items)?;
        return Ok(Value::Array(Array::new(ArrayDesc::flat(base), store)));
    }
    let store = ctor_scalar_store(base, size, init)?;
    Ok(Value::Array(Array::new(ArrayDesc::flat(base), store)))
}

/// The shared primitive constructor (arities 1–2).
fn ctor_array(
    eval: &mut Evaluator,
    display: &str,
    base: BaseType,
    args: &[Value],
) -> Result<Value, NativeError> {
    check_arity(display, args, &[1, 2])?;
    if args.len() == 1 {
        return ctor_seq(eval, base, &args[0]);
    }
    let size = nth_index(&args[0])?;
    ctor_init(eval, base, size, &args[1])
}

fn native_boolean_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/boolean-array", BaseType::Boolean, args)
}

fn native_byte_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/byte-array", BaseType::Byte, args)
}

fn native_char_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/char-array", BaseType::Char, args)
}

fn native_short_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/short-array", BaseType::Short, args)
}

fn native_int_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/int-array", BaseType::Int, args)
}

fn native_long_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/long-array", BaseType::Long, args)
}

fn native_float_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/float-array", BaseType::Float, args)
}

fn native_double_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ctor_array(eval, "clojure.core/double-array", BaseType::Double, args)
}

/// `object-array` is 1-arity only (`RT.object_array` — oracles
/// `c-object-init`, `b-obj-size`, `b-obj-nil`, `oa-*`).
fn native_object_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/object-array", args, &[1])?;
    ctor_seq(eval, BaseType::Object, &args[0])
}

/// Read slot `i` (callers bounds-check): the boxing reads —
/// narrow numbers collapse to `Int`/`Double`, like the port's
/// `byte`/`short`/`int`/`float` natives.
pub(crate) fn array_read(a: &Array, i: usize) -> Value {
    match &*a.lock() {
        ArrayStore::Bools(v) => Value::Bool(v[i]),
        ArrayStore::Bytes(v) => Value::Int(i64::from(v[i])),
        ArrayStore::Chars(v) => Value::Char(v[i]),
        ArrayStore::Shorts(v) => Value::Int(i64::from(v[i])),
        ArrayStore::Ints(v) => Value::Int(i64::from(v[i])),
        ArrayStore::Longs(v) => Value::Int(v[i]),
        ArrayStore::Floats(v) | ArrayStore::Doubles(v) => Value::Double(v[i]),
        ArrayStore::Objs(v) => v[i].clone(),
        ArrayStore::Arrays(v) => v[i].clone().map(Value::Array).unwrap_or(Value::Nil),
    }
}

/// An `intValue` array key (`RT.getFrom`/`RT.contains` narrow
/// every `Number`, leniently — `None` for nil/non-`Number`;
/// oracles `b-get`, `f-get-*`, `e-get-*`, `b-contains`,
/// `f-contains-*`, `e-contains-nil`).
pub(crate) fn array_intvalue_key(k: &Value) -> Option<i32> {
    match k {
        Value::Int(i) => Some(*i as i32),
        Value::Double(d) => Some(*d as i32),
        Value::BigInt(n) | Value::BigInteger(n) => Some(bigint_int_value(n)),
        Value::Ratio(r) => Some(bigint_int_value(&(r.numer() / r.denom()))),
        Value::BigDec(d) => Some(bigint_int_value(&d.to_bigint_trunc())),
        _ => None,
    }
}

/// Bounds-check `i` against `n` (the VM `Index…` text — array
/// lengths always fit `i32` under the 256 MiB guard).
fn check_bounds(i: i32, n: usize) -> Result<usize, NativeError> {
    if i < 0 || i as usize >= n {
        return Err(aioobe_index(i, n));
    }
    Ok(i as usize)
}

/// Direct 2-arity `aget` (the inline `RT.aget`: core-`int` idx —
/// oracles `a-idx-type`, `a-idx-nil`, `a-idx-str`,
/// `a-aget-huge`, `a-aget-hugedbl`).
fn aget_direct(eval: &mut Evaluator, array: &Value, idx: &Value) -> Result<Value, NativeError> {
    // The inline `(int idx)` evaluates before overload matching,
    // so idx failures beat array-match failures.
    // Core-`int` range-checks, so the `i64` always fits `i32`.
    let i = core_int(eval, idx)? as i32;
    aget_step(array, i)
}

/// A multi-arity walk step (steps 1..n-1 pair `intcast_value`
/// with the run-time reflector match — oracles `m4-*`, `ms-*`).
fn aget_walk(array: &Value, idx: &Value) -> Result<Value, NativeError> {
    aget_step(array, intcast_value(idx)?)
}

/// The shared direct-step match: nil → the stable
/// more-than-one text (D4), non-array → the 2-arg no-match,
/// else the indexed read (oracle `m4-negidx`, `b-oob` shape).
fn aget_step(array: &Value, i: i32) -> Result<Value, NativeError> {
    let a = match array {
        Value::Nil => return Err(more_than_one("aget")),
        Value::Array(a) => a,
        _ => return Err(no_match("aget", 2)),
    };
    let n = a.len();
    Ok(array_read(a, check_bounds(i, n)?))
}

/// A raw tail `aget` step (`Array.get`: `nth_index` idx, nil →
/// the `getComponentType` NPE, OOB the empty `AIOOBE` — oracles
/// `a-nil-inner`, `a-deep-nonarray`, `c-aget-tail-idx`,
/// `a-tail-hugedbl`, `tail-oob-get` in `mi-3d/mid-steps.out`).
fn aget_raw(array: &Value, idx: &Value) -> Result<Value, NativeError> {
    let a = match array {
        Value::Nil => return Err(npe_inner_class()),
        Value::Array(a) => a,
        _ => return Err(not_an_array()),
    };
    let i = nth_index(idx)?;
    let n = a.len();
    if i < 0 || i as usize >= n {
        return Err(aioobe_empty());
    }
    Ok(array_read(a, i as usize))
}

/// Walk all-but-last indices via `aget` walk steps (the
/// `core.clj` bodies recurse through `apply`, so middles pair
/// `intcast_value` with the run-time match — oracles `m4-*`,
/// `ms-*`).
fn walk_direct(array: Value, idxs: &[Value]) -> Result<Value, NativeError> {
    let mut cur = array;
    for idx in idxs {
        cur = aget_walk(&cur, idx)?;
    }
    Ok(cur)
}

fn native_aget(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() < 2 {
        return Err(NativeError::arity("clojure.core/aget", args.len()));
    }
    if args.len() == 2 {
        return aget_direct(eval, &args[0], &args[1]);
    }
    let target = walk_direct(args[0].clone(), &args[1..args.len() - 1])?;
    aget_raw(&target, &args[args.len() - 1])
}

/// A direct-`aset` store past the shape match and coercion:
/// the reflector converts args before invoking, so a failing
/// coercion beats bounds (oracle `ord-int-huge-oob`).
enum DirectStore {
    Bool(bool),
    Byte(i8),
    Char(char),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f64),
    Double(f64),
    Ref,
}

/// Shape-match the val through the direct `RT.aset` primitive
/// overloads (else the 3-arg no-match — oracle `ix-order`),
/// coercing (failing coercions beat bounds — oracle
/// `ord-int-huge-oob`). Caller routes refs and nesting to
/// [`DirectStore::Ref`] first (see [`aset_direct_shape`).
fn aset_direct_prim(
    eval: &mut Evaluator,
    base: BaseType,
    v: &Value,
) -> Result<DirectStore, NativeError> {
    match (base, v) {
        (BaseType::Boolean, Value::Bool(b)) => Ok(DirectStore::Bool(*b)),
        (BaseType::Char, Value::Char(c)) => Ok(DirectStore::Char(*c)),
        (BaseType::Float, Value::Double(d)) => Ok(DirectStore::Float(*d)),
        (BaseType::Double, Value::Double(d)) => Ok(DirectStore::Double(*d)),
        (BaseType::Int, Value::Int(l)) => Ok(DirectStore::Int(int_exact_long(*l)?)),
        (BaseType::Long, Value::Int(l)) => Ok(DirectStore::Long(*l)),
        // D2: the JVM only matches the erased `Short`/`Byte`
        // wrappers; the port stores in-range `Int`s (else the
        // width range error — without this `(aset s 0 (short
        // x))` could never work). The range check runs with the
        // other coercions, before bounds — uniform with `Int`
        // (unobservable on the JVM, which never matches here).
        (BaseType::Short, Value::Int(l)) => {
            Ok(DirectStore::Short(core_short(eval, &Value::Int(*l))? as i16))
        }
        (BaseType::Byte, Value::Int(l)) => {
            Ok(DirectStore::Byte(core_byte(eval, &Value::Int(*l))? as i8))
        }
        _ => Err(no_match("aset", 3)),
    }
}

/// Shape-match the val through the direct `RT.aset` overloads:
/// refs and nesting match anything (the store check runs after
/// bounds), primitives narrow through [`aset_direct_prim`].
fn aset_direct_shape(
    eval: &mut Evaluator,
    base: BaseType,
    dims: u32,
    v: &Value,
) -> Result<DirectStore, NativeError> {
    if dims >= 2 || base == BaseType::Object {
        return Ok(DirectStore::Ref);
    }
    aset_direct_prim(eval, base, v)
}

/// Store a coerced direct val past bounds (caller bounds-checks).
fn aset_direct_store(
    a: &Array,
    i: usize,
    store: DirectStore,
    v: &Value,
) -> Result<(), NativeError> {
    let mut cell = a.lock();
    match (store, &mut *cell) {
        (DirectStore::Bool(b), ArrayStore::Bools(v)) => v[i] = b,
        (DirectStore::Char(c), ArrayStore::Chars(v)) => v[i] = c,
        // D3: the exact `f64` (the JVM literal rule narrows to
        // `Inf` past `f32` — the port keeps floats exact).
        (DirectStore::Float(d), ArrayStore::Floats(v)) => v[i] = d,
        (DirectStore::Double(d), ArrayStore::Doubles(v)) => v[i] = d,
        (DirectStore::Int(l), ArrayStore::Ints(v)) => v[i] = l,
        (DirectStore::Long(l), ArrayStore::Longs(v)) => v[i] = l,
        (DirectStore::Short(l), ArrayStore::Shorts(v)) => v[i] = l,
        (DirectStore::Byte(l), ArrayStore::Bytes(v)) => v[i] = l,
        (DirectStore::Ref, _) => {
            drop(cell);
            return store_ref(a, i, v);
        }
        _ => unreachable!("direct store shape"),
    }
    Ok(())
}

/// Store through the direct `RT.aset` overloads: shape-match the
/// val (see [`aset_direct_shape`]), bounds-check (the `Index`
/// text — oracle `b-oob`), store (see [`aset_direct_store`]).
fn aset_direct(eval: &mut Evaluator, a: &Array, i: i32, v: &Value) -> Result<(), NativeError> {
    let desc = a.desc();
    let store = aset_direct_shape(eval, desc.base, desc.dims, v)?;
    let n = a.len();
    let i = check_bounds(i, n)?;
    aset_direct_store(a, i, store, v)
}

/// Store into refs: dims-1 checks the `into-array` narrowing
/// (nil fits), dims>1 checks inner-descriptor assignability
/// (covariant through `Object`, nil fits) — misses are the VM
/// `ArrayStoreException` (oracles `b-obj`, `b-store-obj`,
/// `b-store-2d`, `e-obj-nil`, `e-obj-any`).
fn store_ref(a: &Array, i: usize, v: &Value) -> Result<(), NativeError> {
    let desc = a.desc().clone();
    let mut cell = a.lock();
    match &mut *cell {
        ArrayStore::Objs(slots) => {
            if let Some(component) = &desc.component {
                let want: &str = component;
                if !matches!(v, Value::Nil) && jvm_class_of(v).as_str() != want {
                    return Err(array_store(v));
                }
            }
            slots[i] = v.clone();
            Ok(())
        }
        ArrayStore::Arrays(slots) => match v {
            Value::Nil => {
                slots[i] = None;
                Ok(())
            }
            Value::Array(inner) => {
                let want = desc.inner();
                let got = inner.desc();
                let fits = got.dims == want.dims
                    && (if want.base == BaseType::Object {
                        got.base == BaseType::Object
                    } else {
                        got.base == want.base
                    });
                if !fits {
                    return Err(array_store(v));
                }
                slots[i] = Some(inner.clone());
                Ok(())
            }
            _ => Err(array_store(v)),
        },
        _ => unreachable!("ref store shape"),
    }
}

/// A raw tail `Array.set` into `int[]`: `set` widens
/// `Character` to `int`, but never narrows `Long` (the port's
/// `Int`) down to it.
fn aset_raw_int(a: &Array, i: usize, v: &Value) -> Result<(), NativeError> {
    let x = match v {
        Value::Char(c) => Some(*c as i32),
        _ => None,
    };
    let mut cell = a.lock();
    match (&mut *cell, x) {
        (ArrayStore::Ints(slots), Some(x)) => {
            slots[i] = x;
            Ok(())
        }
        _ => Err(arg_mismatch()),
    }
}

/// A raw tail `Array.set` into `long[]` (`Int` stores, `Char`
/// widens).
fn aset_raw_long(a: &Array, i: usize, v: &Value) -> Result<(), NativeError> {
    let x = match v {
        Value::Int(l) => Some(*l),
        Value::Char(c) => Some(*c as i64),
        _ => None,
    };
    let mut cell = a.lock();
    match (&mut *cell, x) {
        (ArrayStore::Longs(slots), Some(x)) => {
            slots[i] = x;
            Ok(())
        }
        _ => Err(arg_mismatch()),
    }
}

/// A raw tail `Array.set` into `char[]` (exact `Character` unbox).
fn aset_raw_char(a: &Array, i: usize, v: &Value) -> Result<(), NativeError> {
    let x = match v {
        Value::Char(c) => Some(*c),
        _ => None,
    };
    let mut cell = a.lock();
    match (&mut *cell, x) {
        (ArrayStore::Chars(slots), Some(x)) => {
            slots[i] = x;
            Ok(())
        }
        _ => Err(arg_mismatch()),
    }
}

/// A raw tail `Array.set` into `float[]`: `long`/`char` widen to
/// `float`, `double` never narrows.
fn aset_raw_float(a: &Array, i: usize, v: &Value) -> Result<(), NativeError> {
    let x = match v {
        Value::Int(l) => Some(*l as f64),
        Value::Char(c) => Some(*c as i64 as f64),
        _ => None,
    };
    let mut cell = a.lock();
    match (&mut *cell, x) {
        (ArrayStore::Floats(slots), Some(x)) => {
            slots[i] = x;
            Ok(())
        }
        _ => Err(arg_mismatch()),
    }
}

/// A raw tail `Array.set` into `double[]` (`double`/`long`/`char`
/// widen in).
fn aset_raw_double(a: &Array, i: usize, v: &Value) -> Result<(), NativeError> {
    let x = match v {
        Value::Double(d) => Some(*d),
        Value::Int(l) => Some(*l as f64),
        Value::Char(c) => Some(*c as i64 as f64),
        _ => None,
    };
    let mut cell = a.lock();
    match (&mut *cell, x) {
        (ArrayStore::Doubles(slots), Some(x)) => {
            slots[i] = x;
            Ok(())
        }
        _ => Err(arg_mismatch()),
    }
}

/// A raw tail `Array.set` into `boolean[]` (exact `Boolean` unbox).
fn aset_raw_bool(a: &Array, i: usize, v: &Value) -> Result<(), NativeError> {
    let x = match v {
        Value::Bool(b) => Some(*b),
        _ => None,
    };
    let mut cell = a.lock();
    match (&mut *cell, x) {
        (ArrayStore::Bools(slots), Some(x)) => {
            slots[i] = x;
            Ok(())
        }
        _ => Err(arg_mismatch()),
    }
}

/// A raw tail `Array.set` into a flat (dims-1) array: one arm per
/// base — primitives unbox exactly (see the `aset_raw_*` leaves),
/// flat `Object` tails store through the ref rule (same
/// destination as nesting — a live arm instead of an untestable
/// fallback).
fn aset_raw_flat(a: &Array, i: usize, desc: &ArrayDesc, v: &Value) -> Result<(), NativeError> {
    match desc.base {
        BaseType::Int => aset_raw_int(a, i, v),
        BaseType::Long => aset_raw_long(a, i, v),
        // No port value unboxes to `short`/`byte` (the exact
        // wrappers are erased, `Long` never narrows).
        BaseType::Short | BaseType::Byte => Err(arg_mismatch()),
        BaseType::Char => aset_raw_char(a, i, v),
        BaseType::Float => aset_raw_float(a, i, v),
        BaseType::Double => aset_raw_double(a, i, v),
        BaseType::Boolean => aset_raw_bool(a, i, v),
        BaseType::Object => store_raw_ref(a, i, desc, v),
    }
}

/// A raw tail `aset` step (`Array.set`: bounds first, then the
/// exact-unbox + JLS-widen rules — the port reads `Int` as JVM
/// `Long`, so `int[]`←`Int` misses while `long[]`←`Int` stores;
/// oracles `b-2d`, `c-aset-*`, `e-tail-*`).
fn aset_raw(a: &Array, i: i32, v: &Value) -> Result<(), NativeError> {
    // `Array.set` order: nil-into-primitive beats bounds
    // (oracle `tail-oob-nilprim`), bounds (the empty `AIOOBE`
    // — oracles `tail-oob-set`, `tail-oob-mismatch`,
    // `tail-ref-oob-mismatch`) beat unbox and ref misses.
    let desc = a.desc().clone();
    let is_ref = desc.dims > 1 || desc.base == BaseType::Object;
    if !is_ref && matches!(v, Value::Nil) {
        return Err(iae_empty());
    }
    let n = a.len();
    if i < 0 || i as usize >= n {
        return Err(aioobe_empty());
    }
    let i = i as usize;
    // Nesting stores through the ref rule; flat tails dispatch
    // per base below.
    if desc.dims > 1 {
        return store_raw_ref(a, i, &desc, v);
    }
    aset_raw_flat(a, i, &desc, v)
}

/// A raw tail nil store into refs (nil fits every ref cell).
fn store_raw_ref_nil(a: &Array, i: usize) -> Result<(), NativeError> {
    let mut cell = a.lock();
    match &mut *cell {
        ArrayStore::Objs(slots) => slots[i] = Value::Nil,
        ArrayStore::Arrays(slots) => slots[i] = None,
        _ => unreachable!("ref tail shape"),
    }
    Ok(())
}

/// A raw tail store into flat `Object` slots: untyped fits, typed
/// checks the component class, misses are `array element type
/// mismatch`.
fn store_raw_ref_obj(
    slots: &mut [Value],
    desc: &ArrayDesc,
    i: usize,
    v: &Value,
) -> Result<(), NativeError> {
    match &desc.component {
        None => {
            slots[i] = v.clone();
            Ok(())
        }
        Some(component) => {
            let want: &str = component;
            if jvm_class_of(v).as_str() == want {
                slots[i] = v.clone();
                Ok(())
            } else {
                Err(elem_mismatch())
            }
        }
    }
}

/// A raw tail store into nested slots: an assignable inner fits
/// (covariant through `Object`), misses are `array element type
/// mismatch`.
fn store_raw_ref_nested(
    slots: &mut [Option<Array>],
    desc: &ArrayDesc,
    i: usize,
    inner: &Array,
) -> Result<(), NativeError> {
    let want = desc.inner();
    let got = inner.desc();
    let fits = got.dims == want.dims
        && (if want.base == BaseType::Object {
            got.base == BaseType::Object
        } else {
            got.base == want.base
        });
    if !fits {
        return Err(elem_mismatch());
    }
    slots[i] = Some(inner.clone());
    Ok(())
}

/// A raw tail store into refs: nil fits, assignable fits, misses
/// are `array element type mismatch` (never `ArrayStore` —
/// oracle `tail-ref-mismatch` pins the `Array.set` text). The
/// wildcard is live only for descriptor/store shape mismatches
/// (direct unit coverage).
fn store_raw_ref(a: &Array, i: usize, desc: &ArrayDesc, v: &Value) -> Result<(), NativeError> {
    if matches!(v, Value::Nil) {
        return store_raw_ref_nil(a, i);
    }
    let mut cell = a.lock();
    match (&mut *cell, v) {
        (ArrayStore::Objs(slots), _) => store_raw_ref_obj(slots, desc, i, v),
        (ArrayStore::Arrays(slots), Value::Array(inner)) => {
            store_raw_ref_nested(slots, desc, i, inner)
        }
        _ => Err(elem_mismatch()),
    }
}

/// Direct 3-arity `aset` (the inline `RT.aset`: idx via
/// core-`int` — oracle `b-idx-type`). Answers the original val.
fn aset_direct_full(
    eval: &mut Evaluator,
    array: &Value,
    idx: &Value,
    val: &Value,
) -> Result<Value, NativeError> {
    // Idx coerces before matching (inline arg order).
    let i = core_int(eval, idx)? as i32;
    let a = match array {
        Value::Nil => return Err(more_than_one("aset")),
        Value::Array(a) => a,
        _ => return Err(no_match("aset", 3)),
    };
    aset_direct(eval, a, i, val)?;
    Ok(val.clone())
}

/// A raw tail `aset` step over a stepped value (nil → the empty
/// `NPE` — oracle `b-aset-nilinner`).
fn aset_raw_step(target: &Value, idx: &Value, val: &Value) -> Result<Value, NativeError> {
    // The call-site unbox runs before `Array.set` sees the array.
    let i = nth_index(idx)?;
    let a = match target {
        Value::Nil => return Err(npe_empty()),
        Value::Array(a) => a,
        _ => return Err(not_an_array()),
    };
    aset_raw(a, i, val)?;
    Ok(val.clone())
}

fn native_aset(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() < 3 {
        return Err(NativeError::arity("clojure.core/aset", args.len()));
    }
    if args.len() == 3 {
        return aset_direct_full(eval, &args[0], &args[1], &args[2]);
    }
    let target = walk_direct(args[0].clone(), &args[1..args.len() - 2])?;
    aset_raw_step(&target, &args[args.len() - 2], &args[args.len() - 1])
}

/// The eight `Array.setX` methods (`def-aset`).
#[derive(Clone, Copy)]
enum SetX {
    Boolean,
    Byte,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
}

/// A `setX` val past its core coercion.
enum SetXVal {
    Bool(bool),
    Int(i64),
    Char(char),
    Double(f64),
}

/// Coerce the val through the matching wide core fn (see
/// [`setx_coerce`]).
fn setx_coerce_wide(eval: &mut Evaluator, x: SetX, v: &Value) -> Result<SetXVal, NativeError> {
    match x {
        SetX::Int => Ok(SetXVal::Int(core_int(eval, v)?)),
        SetX::Long => Ok(SetXVal::Int(core_long(eval, v)?)),
        SetX::Float => Ok(SetXVal::Double(core_float(eval, v)?)),
        SetX::Double => Ok(SetXVal::Double(core_double(eval, v)?)),
        _ => unreachable!("narrow setX stays in setx_coerce"),
    }
}

/// Coerce the val through the matching core fn (`boolean` is
/// truthiness and never fails — oracle `c-bool`). Split by width
/// for the complexity gate (eight fallible arms trip cyclomatic
/// 16).
fn setx_coerce(eval: &mut Evaluator, x: SetX, v: &Value) -> Result<SetXVal, NativeError> {
    match x {
        SetX::Boolean => Ok(SetXVal::Bool(!matches!(v, Value::Nil | Value::Bool(false)))),
        SetX::Byte => Ok(SetXVal::Int(core_byte(eval, v)?)),
        SetX::Char => Ok(SetXVal::Char(core_char(eval, v)?)),
        SetX::Short => Ok(SetXVal::Int(core_short(eval, v)?)),
        wide => setx_coerce_wide(eval, wide, v),
    }
}

/// Store a coerced `setX` val through the JLS widen table
/// (`setInt`→int/long/float/double, `setLong`→long/float/double,
/// `setShort`→short↑, `setByte`→byte↑, `setChar`→char↑,
/// `setFloat`→float/double, `setDouble`→double,
/// `setBoolean`→boolean; misses are `argument type mismatch` —
/// oracles `c-int`, `c-long`, `c-int-wrongarr`, `d-set*`).
fn setx_store(a: &Array, x: SetX, i: usize, co: &SetXVal) -> Result<(), NativeError> {
    let mut cell = a.lock();
    macro_rules! widens {
        ($variant:ident, $ok:expr) => {
            match (&mut *cell, $ok) {
                (ArrayStore::$variant(slots), Some(x)) => {
                    slots[i] = x;
                    Ok(())
                }
                _ => Err(arg_mismatch()),
            }
        };
    }
    match (x, a.desc().base) {
        (SetX::Boolean, BaseType::Boolean) => widens!(
            Bools,
            match co {
                SetXVal::Bool(b) => Some(*b),
                _ => None,
            }
        ),
        (SetX::Byte, BaseType::Byte) => widens!(
            Bytes,
            match co {
                SetXVal::Int(l) => Some(*l as i8),
                _ => None,
            }
        ),
        (SetX::Byte, BaseType::Short) => widens!(
            Shorts,
            match co {
                SetXVal::Int(l) => Some(*l as i16),
                _ => None,
            }
        ),
        (SetX::Byte | SetX::Short | SetX::Char | SetX::Int, BaseType::Int) => {
            widens!(
                Ints,
                match co {
                    SetXVal::Int(l) => Some(*l as i32),
                    SetXVal::Char(c) => Some(*c as i32),
                    _ => None,
                }
            )
        }
        (SetX::Byte | SetX::Short | SetX::Char | SetX::Int | SetX::Long, BaseType::Long) => {
            widens!(
                Longs,
                match co {
                    SetXVal::Int(l) => Some(*l),
                    SetXVal::Char(c) => Some(*c as i64),
                    _ => None,
                }
            )
        }
        (SetX::Char, BaseType::Char) => widens!(
            Chars,
            match co {
                SetXVal::Char(c) => Some(*c),
                _ => None,
            }
        ),
        (SetX::Short, BaseType::Short) => widens!(
            Shorts,
            match co {
                SetXVal::Int(l) => Some(*l as i16),
                _ => None,
            }
        ),
        (
            SetX::Byte | SetX::Short | SetX::Char | SetX::Int | SetX::Long | SetX::Float,
            BaseType::Float,
        ) => widens!(
            Floats,
            match co {
                SetXVal::Int(l) => Some(*l as f64),
                SetXVal::Char(c) => Some(*c as i64 as f64),
                SetXVal::Double(d) => Some(*d),
                _ => None,
            }
        ),
        (
            SetX::Byte
            | SetX::Short
            | SetX::Char
            | SetX::Int
            | SetX::Long
            | SetX::Float
            | SetX::Double,
            BaseType::Double,
        ) => widens!(
            Doubles,
            match co {
                SetXVal::Int(l) => Some(*l as f64),
                SetXVal::Char(c) => Some(*c as i64 as f64),
                SetXVal::Double(d) => Some(*d),
                _ => None,
            }
        ),
        _ => Err(arg_mismatch()),
    }
}

/// The `def-aset` 3-arity body: `nth_index` idx, coerce val
/// (inline arg order — oracle `setx-both-bad`), array checks
/// (nil → the empty NPE), the primitive check (before bounds —
/// oracle `ix-setint-order`), bounds (the empty `AIOOBE`), the
/// widen table, store. Answers the ORIGINAL val (oracles `c-*`,
/// `b-set*`, `e-set*`).
fn setx_full(
    eval: &mut Evaluator,
    x: SetX,
    array: &Value,
    idx: &Value,
    val: &Value,
) -> Result<Value, NativeError> {
    let i = nth_index(idx)?;
    let co = setx_coerce(eval, x, val)?;
    let a = match array {
        Value::Nil => return Err(npe_empty()),
        Value::Array(a) => a,
        _ => return Err(not_an_array()),
    };
    let desc = a.desc();
    if desc.base == BaseType::Object || desc.dims > 1 {
        return Err(not_primitive());
    }
    let n = a.len();
    if i < 0 || i as usize >= n {
        return Err(aioobe_empty());
    }
    setx_store(a, x, i as usize, &co)?;
    Ok(val.clone())
}

/// The shared `def-aset` native (3+ arity, recursing like `aset`
/// — oracle `c-int-2d`). The walk runs before the final step's
/// val coercion (`(apply name (aget …) …)` evaluates the step
/// first).
fn setx_array(
    eval: &mut Evaluator,
    display: &str,
    x: SetX,
    args: &[Value],
) -> Result<Value, NativeError> {
    if args.len() < 3 {
        return Err(NativeError::arity(display, args.len()));
    }
    if args.len() == 3 {
        return setx_full(eval, x, &args[0], &args[1], &args[2]);
    }
    let target = walk_direct(args[0].clone(), &args[1..args.len() - 2])?;
    setx_full(
        eval,
        x,
        &target,
        &args[args.len() - 2],
        &args[args.len() - 1],
    )
}

fn native_aset_boolean(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-boolean", SetX::Boolean, args)
}

fn native_aset_byte(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-byte", SetX::Byte, args)
}

fn native_aset_char(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-char", SetX::Char, args)
}

fn native_aset_short(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-short", SetX::Short, args)
}

fn native_aset_int(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-int", SetX::Int, args)
}

fn native_aset_long(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-long", SetX::Long, args)
}

fn native_aset_float(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-float", SetX::Float, args)
}

fn native_aset_double(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    setx_array(eval, "clojure.core/aset-double", SetX::Double, args)
}

/// The shared checkcast (`Numbers` casts): nil → nil, the exact
/// dims-1 descriptor → the same handle, else the `CCE` (arrays
/// count as `java.base`/`bootstrap` — oracles `d-*`,
/// `h-ints-2d`, `h-ints-clone-same`).
fn cast_array(display: &str, base: BaseType, args: &[Value]) -> Result<Value, NativeError> {
    check_arity(display, args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::Array(a) if a.desc().dims == 1 && a.desc().base == base => {
            Ok(Value::Array(a.clone()))
        }
        v => Err(cce_err(v, &ArrayDesc::flat(base).descriptor())),
    }
}

fn native_booleans(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/booleans", BaseType::Boolean, args)
}

fn native_bytes(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/bytes", BaseType::Byte, args)
}

fn native_chars(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/chars", BaseType::Char, args)
}

fn native_shorts(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/shorts", BaseType::Short, args)
}

fn native_ints(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/ints", BaseType::Int, args)
}

fn native_longs(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/longs", BaseType::Long, args)
}

fn native_floats(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/floats", BaseType::Float, args)
}

fn native_doubles(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    cast_array("clojure.core/doubles", BaseType::Double, args)
}

/// `bytes?`: true iff a dims-1 `byte[]`, false for nil and
/// everything else (never throws — oracle `e-bytes-*`).
fn native_bytes_q(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/bytes?", args, &[1])?;
    match &args[0] {
        Value::Array(a) => Ok(Value::Bool(
            a.desc().dims == 1 && a.desc().base == BaseType::Byte,
        )),
        _ => Ok(Value::Bool(false)),
    }
}

fn native_alength(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/alength", args, &[1])?;
    match &args[0] {
        Value::Nil => Err(more_than_one("alength")),
        Value::Array(a) => Ok(Value::Int(a.len() as i64)),
        _ => Err(no_match("alength", 1)),
    }
}

/// `aclone` is a shallow copy: fresh identity, same descriptor,
/// cloned slots (inners shared — oracles `g-clone-eq`,
/// `g-clone-mut`, `g-clone-2d-shallow`, `g-clone-obj`).
fn native_aclone(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/aclone", args, &[1])?;
    match &args[0] {
        Value::Nil => Err(more_than_one("aclone")),
        Value::Array(a) => Ok(Value::Array(a.shallow_clone())),
        _ => Err(no_match("aclone", 1)),
    }
}

/// A class designator: the nine plain keywords (the port has no
/// classes yet — spec §8 extension, S16 adds class values).
/// Anything else non-nil fails the `^Class` hint `CCE`
/// (oracles `e-kw-class`, `e-num-class` name the JVM shape);
/// nil passes the hint and fails at use.
fn class_designator(t: &Value) -> Result<Option<BaseType>, NativeError> {
    match t {
        Value::Nil => Ok(None),
        Value::Keyword(k) if k.namespace().is_none() => match k.name() {
            "boolean" => Ok(Some(BaseType::Boolean)),
            "byte" => Ok(Some(BaseType::Byte)),
            "char" => Ok(Some(BaseType::Char)),
            "short" => Ok(Some(BaseType::Short)),
            "int" => Ok(Some(BaseType::Int)),
            "long" => Ok(Some(BaseType::Long)),
            "float" => Ok(Some(BaseType::Float)),
            "double" => Ok(Some(BaseType::Double)),
            "object" => Ok(Some(BaseType::Object)),
            _ => Err(cce_err(t, "java.lang.Class")),
        },
        _ => Err(cce_err(t, "java.lang.Class")),
    }
}

/// Allocate one rectangular level (`dims[0]` slots, inners
/// pre-filled — dims are non-negative here).
fn alloc_rect(base: BaseType, dims: &[i32]) -> Result<ArrayStore, NativeError> {
    let n = dims[0] as usize;
    if dims.len() == 1 {
        return zero_store(base, n);
    }
    let mut v: Vec<Option<Array>> = Vec::new();
    v.try_reserve_exact(n).map_err(|_| oom())?;
    for _ in 0..n {
        let inner = alloc_rect(base, &dims[1..])?;
        let desc = ArrayDesc {
            base,
            dims: (dims.len() as u32) - 1,
            component: None,
        };
        v.push(Some(Array::new(desc, inner)));
    }
    Ok(ArrayStore::Arrays(v))
}

fn native_make_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() < 2 {
        return Err(NativeError::arity("clojure.core/make-array", args.len()));
    }
    // The `^Class` hint checks first, then the body coerces dims
    // through core-`int` (oracles `e-bad-dim`, `e-extra-bad-dim`,
    // `f-make-dlen`, `f-make-nillen`, `f-make-chardim`,
    // `a-make-huge`).
    let base = class_designator(&args[0])?;
    let mut dims = Vec::with_capacity(args.len() - 1);
    for d in &args[1..] {
        dims.push(core_int(eval, d)? as i32);
    }
    let Some(base) = base else {
        return Err(npe_empty());
    };
    for &d in &dims {
        if d < 0 {
            return Err(neg_size(d));
        }
    }
    // The OOM guard runs over the full product (checked — a
    // product overflow is out of memory too).
    let mut slots: u64 = 1;
    for &d in &dims {
        slots = slots.checked_mul(d as u64).ok_or_else(oom)?;
    }
    let bytes = slots.checked_mul(base.slot_bytes()).ok_or_else(oom)?;
    if bytes > MAX_ARRAY_BYTES {
        return Err(oom());
    }
    let store = alloc_rect(base, &dims)?;
    let desc = ArrayDesc {
        base,
        dims: dims.len() as u32,
        component: None,
    };
    Ok(Value::Array(Array::new(desc, store)))
}

/// The 1-arity `into-array`: infer the component from the first
/// element's class (nil/empty → `Object`), then check every
/// element — nil fits, misses are `array element type mismatch`
/// (oracles `f-into-plain`, `f-into-mixed`, `f-into-empty`,
/// `a-into-nil-elem`, `a-into-empty-cls`).
fn into_inferred(items: &[Value]) -> Result<Value, NativeError> {
    let component: Option<String> = match items.first() {
        None | Some(Value::Nil) => None,
        Some(first) => Some(jvm_class_of(first)),
    };
    if let Some(want) = &component {
        for v in items {
            if !matches!(v, Value::Nil) && jvm_class_of(v).as_str() != want.as_str() {
                return Err(elem_mismatch());
            }
        }
    }
    let desc = ArrayDesc {
        base: BaseType::Object,
        dims: 1,
        component: component.map(Arc::from),
    };
    Ok(Value::Array(Array::new(
        desc,
        ArrayStore::Objs(items.to_vec()),
    )))
}

/// The raw `Array.set` element rule for `into-array`
/// `Long/TYPE`/`Double/TYPE`/`Boolean/TYPE` (exact-unbox + widen;
/// nil → the empty `IAE`; oracles `a-long-*`, `a-double-*`,
/// `a-bool-*`, `c-long-nil`, `c-double-nil`, `c-long-double`).
fn into_raw_long(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(l) => Ok(*l),
        Value::Char(c) => Ok(*c as i64),
        Value::Nil => Err(iae_empty()),
        _ => Err(arg_mismatch()),
    }
}

/// The raw rule for `into-array` `Double/TYPE`.
fn into_raw_double(v: &Value) -> Result<f64, NativeError> {
    match v {
        Value::Double(d) => Ok(*d),
        Value::Int(l) => Ok(*l as f64),
        Value::Char(c) => Ok(*c as i64 as f64),
        Value::Nil => Err(iae_empty()),
        _ => Err(arg_mismatch()),
    }
}

/// The raw rule for `into-array` `Boolean/TYPE` (strict — no
/// truthiness here).
fn into_raw_bool(v: &Value) -> Result<bool, NativeError> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Nil => Err(iae_empty()),
        _ => Err(arg_mismatch()),
    }
}

/// The 2-arity `into-array`: `int`/`byte`/`short`/`float`/`char`
/// run the `RT.*Cast` per element, `long`/`double`/`boolean`
/// the raw `Array.set` rules, `:object` fits everything
/// (oracles `f-into-typed`, `f-into-typemismatch`, `f-into-strs`,
/// `a-*`, `c-*`).
fn into_typed(eval: &mut Evaluator, base: BaseType, items: &[Value]) -> Result<Value, NativeError> {
    macro_rules! casted {
        ($variant:ident, $t:ty, $cast:expr) => {{
            let mut v: Vec<$t> = Vec::with_capacity(items.len());
            for it in items {
                v.push($cast(it)?);
            }
            ArrayStore::$variant(v)
        }};
    }
    let store = match base {
        BaseType::Int => casted!(Ints, i32, intcast_value),
        BaseType::Byte => casted!(Bytes, i8, |v: &Value| narrow_cast_value(
            v, "byte", -128, 127
        )
        .map(|l| l as i8)),
        BaseType::Short => casted!(Shorts, i16, |v: &Value| narrow_cast_value(
            v, "short", -32768, 32767
        )
        .map(|l| l as i16)),
        BaseType::Char => casted!(Chars, char, charcast_value),
        BaseType::Float => casted!(Floats, f64, |v: &Value| core_float(eval, v)),
        BaseType::Long => casted!(Longs, i64, into_raw_long),
        BaseType::Double => casted!(Doubles, f64, into_raw_double),
        BaseType::Boolean => casted!(Bools, bool, into_raw_bool),
        BaseType::Object => ArrayStore::Objs(items.to_vec()),
    };
    Ok(Value::Array(Array::new(ArrayDesc::flat(base), store)))
}

fn native_into_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/into-array", args, &[1, 2])?;
    if args.len() == 1 {
        let items = seq_items(eval, &args[0])?;
        return into_inferred(&items);
    }
    // The `^Class` hint checks before `(seq aseq)`; nil fails at
    // `newInstance` (oracle `f-into-niltype`).
    let base = class_designator(&args[0])?;
    let items = seq_items(eval, &args[1])?;
    let Some(base) = base else {
        return Err(npe_empty());
    };
    into_typed(eval, base, &items)
}

/// `RT.toArray` over one value: nil → the shared empty, `Object`
/// arrays (any dims — `instanceof`) → the same handle, other
/// arrays → a boxed copy, walkables → their elements, anything
/// else → `Unable to convert` (oracles `f-to-*`, `d-ident`,
/// `d-prim-cls`, `d-num`, `d-kw`, `d-range`, `d-lazy`).
fn to_array_value(eval: &mut Evaluator, v: &Value) -> Result<Array, NativeError> {
    match v {
        Value::Nil => Ok(EMPTY_OBJ_ARRAY.clone()),
        Value::Array(a) if a.desc().base == BaseType::Object => Ok(a.clone()),
        Value::Array(a) => Ok(Array::new(
            ArrayDesc::flat(BaseType::Object),
            ArrayStore::Objs(a.snapshot()),
        )),
        _ if is_toarray_walkable(v) => {
            let items = collect_all(eval, v).map_err(signal_to_native)?;
            Ok(Array::new(
                ArrayDesc::flat(BaseType::Object),
                ArrayStore::Objs(items),
            ))
        }
        _ => Err(bad_convert(v)),
    }
}

fn native_to_array(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/to-array", args, &[1])?;
    Ok(Value::Array(to_array_value(eval, &args[0])?))
}

fn native_to_array_2d(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/to-array-2d", args, &[1])?;
    let coll = &args[0];
    if matches!(coll, Value::Nil) {
        return Err(npe_coll_size());
    }
    if !is_collection(coll) {
        return Err(cce_err(coll, "java.util.Collection"));
    }
    let items = collect_all(eval, coll).map_err(signal_to_native)?;
    let mut rows: Vec<Option<Array>> = Vec::with_capacity(items.len());
    for it in &items {
        rows.push(Some(to_array_value(eval, it)?));
    }
    let desc = ArrayDesc {
        base: BaseType::Object,
        dims: 2,
        component: None,
    };
    Ok(Value::Array(Array::new(desc, ArrayStore::Arrays(rows))))
}

/// Coerce one element through a wide `vector-of` type's core fn
/// (see [`vector_coerce`]).
fn vector_coerce_wide(eval: &mut Evaluator, elem: Elem8, v: &Value) -> Result<Value, NativeError> {
    match elem {
        Elem8::Int => Ok(Value::Int(core_int(eval, v)?)),
        Elem8::Long => Ok(Value::Int(core_long(eval, v)?)),
        Elem8::Float => Ok(Value::Double(core_float(eval, v)?)),
        Elem8::Double => Ok(Value::Double(core_double(eval, v)?)),
        _ => unreachable!("narrow vector-of stays in vector_coerce"),
    }
}

/// Coerce one element through the `vector-of` type's core fn
/// (the `ArrayManager` rule — `boolean` is truthiness; shared
/// with `conj`/`assoc`). Split by width for the complexity gate
/// (eight fallible arms trip cyclomatic 16).
pub(crate) fn vector_coerce(
    eval: &mut Evaluator,
    elem: Elem8,
    v: &Value,
) -> Result<Value, NativeError> {
    match elem {
        Elem8::Boolean => Ok(Value::Bool(!matches!(v, Value::Nil | Value::Bool(false)))),
        Elem8::Byte => Ok(Value::Int(core_byte(eval, v)?)),
        Elem8::Char => Ok(Value::Char(core_char(eval, v)?)),
        Elem8::Short => Ok(Value::Int(core_short(eval, v)?)),
        wide => vector_coerce_wide(eval, wide, v),
    }
}

/// Resolve a wide `vector-of` type keyword (anything else fails
/// `Unrecognized type` — oracle `f-vec-badtype`).
fn vector_of_keyword_wide(
    eval: &mut Evaluator,
    t: &Value,
    name: &str,
) -> Result<Elem8, NativeError> {
    match name {
        "int" => Ok(Elem8::Int),
        "long" => Ok(Elem8::Long),
        "float" => Ok(Elem8::Float),
        "double" => Ok(Elem8::Double),
        _ => Err(unrecognized(eval, t)?),
    }
}

/// Resolve a `vector-of` type keyword (the eight plain keywords;
/// anything else fails `Unrecognized type` — oracle
/// `f-vec-badtype`).
fn vector_of_keyword(eval: &mut Evaluator, t: &Value, name: &str) -> Result<Elem8, NativeError> {
    match name {
        "boolean" => Ok(Elem8::Boolean),
        "byte" => Ok(Elem8::Byte),
        "char" => Ok(Elem8::Char),
        "short" => Ok(Elem8::Short),
        _ => vector_of_keyword_wide(eval, t, name),
    }
}

/// Resolve the `vector-of` type designator (a plain keyword —
/// anything else fails `Unrecognized type`).
fn vector_of_elem(eval: &mut Evaluator, t: &Value) -> Result<Elem8, NativeError> {
    match t {
        Value::Keyword(k) if k.namespace().is_none() => vector_of_keyword(eval, t, k.name()),
        _ => Err(unrecognized(eval, t)?),
    }
}

fn native_vector_of(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/vector-of", 0));
    }
    let elem = vector_of_elem(eval, &args[0])?;
    let mut items = Vec::with_capacity(args.len() - 1);
    for v in &args[1..] {
        items.push(vector_coerce(eval, elem, v)?);
    }
    Ok(Value::PrimitiveVector(PrimitiveVector::new(
        elem, items, None,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A descriptor/store shape mismatch panics (production
    /// constructors never build one — the panic is the contract).
    #[test]
    #[should_panic(expected = "direct store shape")]
    fn direct_store_shape_mismatch_panics() {
        let mut eval = crate::boot();
        let a = Array::new(ArrayDesc::flat(BaseType::Int), ArrayStore::Longs(vec![0]));
        let _ = aset_direct(&mut eval, &a, 0, &Value::Int(5));
    }

    /// `zero_store_wide` only takes wide types (`zero_store` keeps
    /// the narrow arms) — the catch-all panics by contract.
    #[test]
    #[should_panic(expected = "narrow types stay in zero_store")]
    fn zero_store_wide_rejects_narrow() {
        let _ = zero_store_wide(BaseType::Int, 1);
    }

    /// `setx_coerce_wide` only takes wide methods (`setx_coerce`
    /// keeps the narrow arms) — the catch-all panics by contract.
    #[test]
    #[should_panic(expected = "narrow setX stays in setx_coerce")]
    fn setx_coerce_wide_rejects_narrow() {
        let mut eval = crate::boot();
        let _ = setx_coerce_wide(&mut eval, SetX::Boolean, &Value::Bool(true));
    }

    /// `vector_coerce_wide` only takes wide element types
    /// (`vector_coerce` keeps the narrow arms) — the catch-all
    /// panics by contract.
    #[test]
    #[should_panic(expected = "narrow vector-of stays in vector_coerce")]
    fn vector_coerce_wide_rejects_narrow() {
        let mut eval = crate::boot();
        let _ = vector_coerce_wide(&mut eval, Elem8::Boolean, &Value::Bool(true));
    }

    /// See [`direct_store_shape_mismatch_panics`].
    #[test]
    #[should_panic(expected = "ref store shape")]
    fn ref_store_shape_mismatch_panics() {
        let a = Array::new(ArrayDesc::flat(BaseType::Object), ArrayStore::Ints(vec![0]));
        let _ = store_ref(&a, 0, &Value::Int(1));
    }

    /// See [`direct_store_shape_mismatch_panics`].
    #[test]
    #[should_panic(expected = "ref tail shape")]
    fn raw_ref_nil_shape_mismatch_panics() {
        let a = Array::new(ArrayDesc::flat(BaseType::Object), ArrayStore::Ints(vec![0]));
        let desc = a.desc().clone();
        let _ = store_raw_ref(&a, 0, &desc, &Value::Nil);
    }

    /// Anything else renders its `Debug` in a range failure
    /// (production callers only pass numbers/chars past the
    /// `long` stop, so the wildcard is covered directly).
    #[test]
    fn oor_val_renders_other_values_debug() {
        let err = oor_val("int", &Value::Str("x".into()));
        assert_eq!(err.class, ThrowClass::IllegalArgumentException);
        assert!(err.message.starts_with("Value out of range for int: "));
    }

    /// The tail `fits` check rejects a primitive cell with the
    /// `Array.set` text (production arrays never mix a ref
    /// descriptor with a primitive store).
    #[test]
    fn raw_ref_fits_rejects_primitive_cell() {
        let a = Array::new(ArrayDesc::flat(BaseType::Object), ArrayStore::Ints(vec![0]));
        let desc = a.desc().clone();
        let err = store_raw_ref(&a, 0, &desc, &Value::Int(1)).expect_err("primitive cell misses");
        assert_eq!(err.class, ThrowClass::IllegalArgumentException);
        assert_eq!(err.message, "array element type mismatch");
    }

    /// Each `setX` widens only its coerced shape: a foreign
    /// coercion misses with `argument type mismatch`.
    /// Unreachable through the natives (whose coercions always
    /// match their `setX`), so covered directly.
    #[test]
    fn setx_store_mismatched_coercion_misses() {
        let arrays = [
            Array::new(
                ArrayDesc::flat(BaseType::Boolean),
                ArrayStore::Bools(vec![false]),
            ),
            Array::new(ArrayDesc::flat(BaseType::Byte), ArrayStore::Bytes(vec![0])),
            Array::new(
                ArrayDesc::flat(BaseType::Short),
                ArrayStore::Shorts(vec![0]),
            ),
            Array::new(ArrayDesc::flat(BaseType::Long), ArrayStore::Longs(vec![0])),
            Array::new(
                ArrayDesc::flat(BaseType::Char),
                ArrayStore::Chars(vec!['\0']),
            ),
            Array::new(
                ArrayDesc::flat(BaseType::Float),
                ArrayStore::Floats(vec![0.0]),
            ),
            Array::new(
                ArrayDesc::flat(BaseType::Double),
                ArrayStore::Doubles(vec![0.0]),
            ),
            Array::new(ArrayDesc::flat(BaseType::Int), ArrayStore::Ints(vec![0])),
        ];
        let cases: &[(SetX, usize, SetXVal)] = &[
            (SetX::Boolean, 0, SetXVal::Int(0)),
            (SetX::Byte, 1, SetXVal::Bool(true)),
            (SetX::Byte, 2, SetXVal::Bool(true)),
            (SetX::Int, 3, SetXVal::Bool(true)),
            (SetX::Char, 4, SetXVal::Int(0)),
            (SetX::Short, 2, SetXVal::Bool(true)),
            (SetX::Int, 5, SetXVal::Bool(true)),
            (SetX::Int, 6, SetXVal::Bool(true)),
            (SetX::Int, 7, SetXVal::Bool(true)),
        ];
        for (x, slot, co) in cases {
            let err = setx_store(&arrays[*slot], *x, 0, co).expect_err("foreign coercion misses");
            assert_eq!(err.class, ThrowClass::IllegalArgumentException);
            assert_eq!(err.message, "argument type mismatch");
        }
    }
}
