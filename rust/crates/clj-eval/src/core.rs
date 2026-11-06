// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S5D scalar natives: arithmetic (C1), comparison (C2), predicates,
//! the `str` family, `format` (US-locale subset), the `ex-info`
//! family, `trampoline`, and `reduced`. Oracle: probe39; rules:
//! `docs/spec/t1174/s5d-contract.md`.

use std::any::Any;
use std::collections::HashSet;
use std::hash::Hasher;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use num_traits::{Signed, ToPrimitive, Zero};

use smallvec::SmallVec;

use clj_reader::{ReadCond, Reader, ReaderOpts};
use clj_value::{
    ExtensionValue, Instant, Keyword, List, Map, PrintOpts, Set, Symbol, TagError, ThrowClass,
    Throwable, Uuid, Value, Vector, bigdec_result, new_map, new_set, pr_to_string, seq_vec,
    validate_uuid,
};

use crate::error::{NativeError, jvm_class_of};
use crate::eval::Evaluator;
use crate::ext::{FastArith, NativeFn};
use crate::lazy::{
    as_pending, as_realized, as_subvector, collect_all, equiv_vals, is_counted_seq, is_seq_value,
    realize_deep,
};
use crate::ns::{
    NativeFnPtr, check_arity, core_var, core_var_handle, deref_core, intern_routed, npe,
    signal_to_native, throw_class_for, var_value,
};
use crate::reader::{LiveReadHooks, map_reader_error, seed_read_gates, sync_reader_ns};

/// `Cannot invoke "Object.getClass()" because "x" is null`: the
/// numbers-`nil` failure (every arithmetic/comparison op probes it).
pub(crate) fn npe_getclass() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"Object.getClass()\" because \"x\" is null".to_owned(),
    )
}

/// The JVM cast text (S6 moved the shared home to `clj-value` for
/// the reader's spanful `TableCce`; this re-export keeps every
/// `crate::core::cce_message` caller working).
pub(crate) use clj_value::cce_message;

/// A cast failure to `to_jvm` (NPEs never reach here — nils fail
/// loudly at each call site instead).
pub(crate) fn cce_err(v: &Value, to_jvm: &str) -> NativeError {
    NativeError::new(
        ThrowClass::ClassCastException,
        cce_message(&jvm_class_of(v), to_jvm),
    )
}

/// A `take`/`drop` count: `Int` as-is, `Double` rounded up (the
/// JVM `pos?`/`dec` loop takes `ceil(n)`), `NaN` takes nothing,
/// infinities take everything-or-nothing, `nil` is the numbers
/// NPE, anything else fails to `Number`.
pub(crate) fn take_count(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Double(d) => Ok(take_double(*d)),
        // Tower counts like `double` counts (the JVM `pos?`/`dec`
        // loop behaves as `ceil`: oracle `index-tower`,
        // `index-big` with its `1.0E20` rendering).
        Value::BigInt(n) | Value::BigInteger(n) => Ok(take_double(n.to_f64().unwrap_or(f64::NAN))),
        Value::Ratio(r) => Ok(take_double(r.to_f64().unwrap_or(f64::NAN))),
        Value::BigDec(d) => Ok(take_double(d.to_f64())),
        Value::Nil => Err(npe_getclass()),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

/// A `take`/`drop` count from a `double`: `NaN` takes nothing,
/// infinities saturate, the rest rounds up.
fn take_double(d: f64) -> i64 {
    if d.is_nan() {
        0
    } else if d >= i64::MAX as f64 {
        i64::MAX
    } else if d <= i64::MIN as f64 {
        i64::MIN
    } else {
        d.ceil() as i64
    }
}

/// A `repeat` count: `intValue` truncation (`NaN` is zero,
/// infinities saturate; `nil` trips the `Repeat.create`
/// `doubleValue` NPE, anything else fails to `Number`).
pub(crate) fn repeat_count(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Double(d) => repeat_double(*d),
        // Tower truncates like `intValue`, through the same
        // range check (the drop-path `double` rendering stays).
        Value::BigInt(n) | Value::BigInteger(n) => repeat_double(n.to_f64().unwrap_or(f64::NAN)),
        Value::Ratio(r) => repeat_double(r.to_f64().unwrap_or(f64::NAN)),
        Value::BigDec(d) => repeat_double(d.to_f64()),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.lang.Number.doubleValue()\" because \"x\" is null".to_owned(),
        )),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

/// A `repeat` count from a `double`: `intValue` truncation
/// (`NaN` is zero, past-`i64` fails `out of range for long`).
fn repeat_double(d: f64) -> Result<i64, NativeError> {
    if d.is_nan() {
        Ok(0)
    } else if d > i64::MAX as f64 || d < i64::MIN as f64 {
        Err(out_of_range("long", Num::Double(d)))
    } else {
        Ok(d as i64)
    }
}

/// `Math.toIntExact` over a `long` (out of `i32` range is
/// `ArithmeticException "integer overflow"`).
pub(crate) fn int_exact_long(l: i64) -> Result<i32, NativeError> {
    i32::try_from(l).map_err(|_| {
        NativeError::new(
            ThrowClass::ArithmeticException,
            "integer overflow".to_owned(),
        )
    })
}

/// `RT.longCast` + `intCast` over a `double` (out of `i64` range
/// is `IllegalArgumentException "Value out of range for long: X"`
/// with Java double rendering, then the `int` range check).
pub(crate) fn int_cast_double(d: f64) -> Result<i32, NativeError> {
    if d < i64::MIN as f64 || d > i64::MAX as f64 {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!(
                "Value out of range for long: {}",
                clj_value::java_double_string(d)
            ),
        ));
    }
    int_exact_long(d as i64)
}

/// The `drop`/`nthrest` `IDrop` count coercion: `Int` narrows
/// exactly, `Double` ceils through the long range check. Callers
/// `pos?`-check first, so nil/non-numbers are unreachable — but
/// validated identically for totality.
pub(crate) fn idrop_coerce(v: &Value) -> Result<i32, NativeError> {
    // `drop`/`nthrest` validate `n` first, so only numbers arrive;
    // `as_num` keeps the identical failures total (arms covered).
    // Tower counts like `double` counts (the JVM `pos?`/`dec`
    // loop behaves as `ceil`, and huge counts render `1.0E20`:
    // oracle `index-tower`, `index-big`).
    match as_num(v)? {
        Num::Int(i) => int_exact_long(i),
        Num::Double(d) => int_cast_double(d.ceil()),
        Num::BigInt(n) => int_cast_double(n.to_f64().unwrap_or(f64::NAN).ceil()),
        Num::Ratio(r) => int_cast_double(r.to_f64().unwrap_or(f64::NAN).ceil()),
        Num::BigDec(d) => int_cast_double(d.to_f64().ceil()),
    }
}

/// An `nth`/`subvec`/`subs` index through the `RT.intCast`
/// shape: `nil` trips the `charValue` NPE quirk, numbers narrow
/// checked (long range then `int` range), anything else fails to
/// `Number`. Staged for `coll.rs` (S5D); shared with the `clj-rt`
/// string fast path (same function, zero drift).
pub fn nth_index(v: &Value) -> Result<i32, NativeError> {
    if matches!(v, Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.lang.Character.charValue()\" because \"x\" is null".to_owned(),
        ));
    }
    match v {
        Value::Int(i) => int_exact_long(*i),
        Value::Double(d) => int_cast_double(*d),
        // `nth` truncates toward zero first, then takes the `long`
        // range check (exact digits) and `toIntExact` (oracle
        // `nth-trunc2`, `nth-big`, `nth-intrange`); decimals
        // narrow through `double` first (like `int`).
        Value::BigInt(n) | Value::BigInteger(n) => tower_intcast(n),
        Value::Ratio(r) => tower_intcast(&trunc_ratio(r)),
        Value::BigDec(d) => bigdec_intcast(d),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

/// An `RT.intCast` index from a truncated tower integer:
/// past-`i64` fails `Value out of range for long` with the exact
/// digits, past-`i32` fails `integer overflow` (oracles
/// `nth-big`, `nth-intrange`, `conv-hugeint` — shared by `int`,
/// `nth`, and `subvec`).
pub(crate) fn tower_intcast(n: &num_bigint::BigInt) -> Result<i32, NativeError> {
    let l = n
        .to_i64()
        .ok_or_else(|| out_of_range("long", Num::BigInt(n.clone())))?;
    int_exact_long(l)
}

/// An `RT.intCast` index from a decimal: through the `long`
/// range check first (double rendering), then `toIntExact`
/// (oracle `conv-hugedecint`).
fn bigdec_intcast(d: &clj_value::BigDec) -> Result<i32, NativeError> {
    int_exact_long(double_long(d.to_f64())?)
}

/// A `long` from a `double`: `NaN` is zero, past-`i64` fails
/// `Value out of range for long` (double rendering), else
/// truncation toward zero.
pub(crate) fn double_long(d: f64) -> Result<i64, NativeError> {
    if d.is_nan() {
        Ok(0)
    } else if d >= 9223372036854775808.0 || d < -9223372036854775808.0 {
        Err(out_of_range("long", Num::Double(d)))
    } else {
        Ok(d.trunc() as i64)
    }
}

/// A `long` from a tower integer: past-`i64` fails `Value out of
/// range for long` with the exact digits (`short`/`byte` narrow
/// through here, so `(short 2^63N)` names `long`; oracle
/// `tower-r2e`).
/// Narrow a `long` into a smaller-width coercion target.
fn narrow_range(l: i64, lo: i64, hi: i64, kind: &'static str) -> Result<i64, NativeError> {
    if l < lo || l > hi {
        Err(out_of_range(kind, Num::Int(l)))
    } else {
        Ok(l)
    }
}

/// Narrow any number into a smaller-width coercion target:
/// doubles range-check directly (double rendering), unbounded
/// integers go through the `long` stop first, decimals through
/// `double` like `int` (oracles `tower-r2e`, `tower-r2f`).
fn narrow_num(n: &Num, lo: i64, hi: i64, kind: &'static str) -> Result<i64, NativeError> {
    match n {
        Num::Double(d) => {
            if d.is_nan() || *d > hi as f64 || *d < lo as f64 {
                return Err(out_of_range(kind, Num::Double(*d)));
            }
            Ok(d.trunc() as i64)
        }
        Num::Int(i) => narrow_range(*i, lo, hi, kind),
        Num::BigInt(n) => narrow_range(bigint_long(n)?, lo, hi, kind),
        Num::Ratio(r) => narrow_range(bigint_long(&trunc_ratio(r))?, lo, hi, kind),
        Num::BigDec(d) => narrow_range(double_long(d.to_f64())?, lo, hi, kind),
    }
}

/// Java `intValue`: low 32 bits, sign-extended (`BigDecimal`
/// `(char 1E300M)` needs the NUL, not a range error; oracle
/// `tower-r2f`).
pub(crate) fn bigint_int_value(n: &num_bigint::BigInt) -> i32 {
    let mut le = n.to_signed_bytes_le();
    le.resize(
        4,
        if n.sign() == num_bigint::Sign::Minus {
            0xFF
        } else {
            0
        },
    );
    i32::from_le_bytes([le[0], le[1], le[2], le[3]])
}

/// A `char` from an unbounded integer: the full value
/// range-checks (never the `long` stop — oracle `tower-r2g`),
/// then UTF-32 narrows (surrogates stay the documented
/// divergence).
fn char_from_bigint(n: &num_bigint::BigInt) -> Result<char, NativeError> {
    match n
        .to_u32()
        .filter(|&u| u <= 1114111)
        .and_then(char::from_u32)
    {
        Some(c) => Ok(c),
        None => Err(out_of_range("char", Num::BigInt(n.clone()))),
    }
}

/// A `char` from a decimal: `BigDecimal.intValue` first, then the
/// `char` range check.
fn char_from_bigdec(d: &clj_value::BigDec) -> Result<char, NativeError> {
    let i = bigint_int_value(&d.to_bigint_trunc());
    if !(0..=1114111).contains(&i) {
        return Err(out_of_range("char", Num::Int(i64::from(i))));
    }
    match char::from_u32(i as u32) {
        Some(c) => Ok(c),
        None => Err(out_of_range("char", Num::Int(i64::from(i)))),
    }
}

/// A `char` from any number: unbounded integers range-check
/// whole (never the `long` stop), ratios truncate first,
/// decimals go through `BigDecimal.intValue` (oracles
/// `tower-r2f`, `tower-r2g`).
fn char_of_num(n: &Num) -> Result<char, NativeError> {
    match n {
        Num::Int(i) => match char::from_u32(u32::try_from(*i).unwrap_or(u32::MAX)) {
            // Surrogates are unrepresentable (documented
            // divergence — the JVM accepts them, unprintably).
            Some(c) => Ok(c),
            None => Err(out_of_range("char", Num::Int(*i))),
        },
        Num::Double(d) => {
            if d.is_nan() || *d > 1114111.0 || *d < 0.0 {
                return Err(out_of_range("char", Num::Double(*d)));
            }
            match char::from_u32(d.trunc() as u32) {
                Some(c) => Ok(c),
                None => Err(out_of_range("char", Num::Double(*d))),
            }
        }
        Num::BigInt(n) => char_from_bigint(n),
        Num::Ratio(r) => char_from_bigint(&trunc_ratio(r)),
        Num::BigDec(d) => char_from_bigdec(d),
    }
}

/// Whether the value is callable (`fn?` plus the `IFn` collections,
/// keywords, symbols, and vars; `Unbound` passes through to the
/// caller's unbound error; S12 struct accessors are `IFn` but not
/// `fn?`, like the JVM's `PersistentStructMap$1` — and struct
/// instances are `IFn` too, like the JVM's `PersistentStructMap`,
/// while records stay non-`IFn` on both sides).
pub(crate) fn is_ifn(v: &Value) -> bool {
    match v {
        Value::Extension(e) => {
            e.as_any().downcast_ref::<crate::ext::Closure>().is_some()
                || e.as_any().downcast_ref::<crate::ext::NativeFn>().is_some()
                || e.as_any()
                    .downcast_ref::<crate::collmisc::AccessorValue>()
                    .is_some()
                || as_subvector(v).is_some()
        }
        Value::Keyword(_)
        | Value::Symbol(_)
        | Value::Vector(_)
        | Value::MapEntry { .. }
        | Value::Map(_)
        | Value::Set(_)
        | Value::Var(_)
        | Value::Promise(_)
        | Value::Ref(_)
        | Value::MultiFn(_)
        | Value::PrimitiveVector(_)
        | Value::Unbound(_) => true,
        // Struct instances only (records and bases miss — S12
        // oracle `sm-ifn`; records stay pinned non-`IFn` by
        // `types.rs rec_*`).
        _ => crate::types::is_host_struct(v),
    }
}

/// Whether the value is a `fn?` (closures and natives only).
pub(crate) fn is_fn_val(v: &Value) -> bool {
    match v {
        Value::Extension(e) => {
            e.as_any().downcast_ref::<crate::ext::Closure>().is_some()
                || e.as_any().downcast_ref::<crate::ext::NativeFn>().is_some()
        }
        _ => false,
    }
}

/// Check a higher-order callee: nil is the nil-invoke `NPE`,
/// non-`IFn`s are the `IFn` cast failure.
pub(crate) fn check_ifn(eval: &mut Evaluator, f: &Value) -> Result<(), crate::error::Signal> {
    if matches!(f, Value::Nil) {
        return Err(eval.throw(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke nil as a function".to_owned(),
        )));
    }
    if !is_ifn(f) {
        return Err(eval.throw(NativeError::new(
            ThrowClass::ClassCastException,
            cce_message(&jvm_class_of(f), "clojure.lang.IFn"),
        )));
    }
    Ok(())
}

/// A numeric operand (nil is the numbers NPE, anything else
/// non-numeric the `Number` cast failure). Tower ranks for
/// promotion: `Int` < `BigInt` < `Ratio` < `BigDec` < `Double`
/// (oracle-pinned: `(+ 1/2 5N)` is a `Ratio`, `(+ 5N 5M)` a
/// `BigDec`, `(+ 5M 0.5)` a `Double`).
#[derive(Clone)]
pub(crate) enum Num {
    Int(i64),
    BigInt(num_bigint::BigInt),
    Ratio(num_rational::BigRational),
    BigDec(clj_value::BigDec),
    Double(f64),
}

pub(crate) fn as_num(v: &Value) -> Result<Num, NativeError> {
    match v {
        Value::Int(i) => Ok(Num::Int(*i)),
        // `BigInteger` folds into `Num::BigInt`: ops absorb it
        // (results print `N` — S2 oracles `r8-add`..`r8-typeop`).
        Value::BigInt(n) | Value::BigInteger(n) => Ok(Num::BigInt(n.as_ref().clone())),
        Value::Ratio(r) => Ok(Num::Ratio(r.as_ref().clone())),
        Value::BigDec(d) => Ok(Num::BigDec(d.as_ref().clone())),
        Value::Double(d) => Ok(Num::Double(*d)),
        Value::Nil => Err(npe_getclass()),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

fn num_value(n: Num) -> Value {
    match n {
        Num::Int(i) => Value::Int(i),
        Num::BigInt(n) => Value::BigInt(Arc::new(n)),
        Num::Ratio(r) => normalize_rational(r),
        // Arithmetic results canonicalize through `valueOf`
        // (scale-0 `0..=10` share — fence 4a-big, oracle P5).
        Num::BigDec(d) => bigdec_result(d),
        Num::Double(d) => Value::Double(d),
    }
}

/// A rational with denominator 1 is a `BigInt` (the reader and
/// `/` rely on this as well as the ops below).
fn normalize_rational(r: num_rational::BigRational) -> Value {
    if r.denom() == &num_bigint::BigInt::from(1) {
        Value::BigInt(Arc::new(r.numer().clone()))
    } else {
        Value::Ratio(Arc::new(r))
    }
}

/// Promotion rank (`Double` > `BigDec` > `Ratio` > `BigInt` > `Int`).
fn num_rank(n: &Num) -> u8 {
    match n {
        Num::Int(_) => 0,
        Num::BigInt(_) => 1,
        Num::Ratio(_) => 2,
        Num::BigDec(_) => 3,
        Num::Double(_) => 4,
    }
}

/// The checked `long overflow` (C1: always checked, never wraps).
fn overflow() -> NativeError {
    NativeError::new(ThrowClass::ArithmeticException, "long overflow".to_owned())
}

/// The integer `Divide by zero` (`/`, `mod`, and the
/// `quot`/`rem`/`mod` double-zero paths; plain `/` on doubles goes
/// to `Inf`/`NaN` instead).
fn divide_by_zero() -> NativeError {
    NativeError::new(ThrowClass::ArithmeticException, "Divide by zero".to_owned())
}

/// The integer `/ by zero` (`quot`/`rem` int paths only — the
/// JVM's dedicated integer-division text, fixprobe `:quot-i0`).
fn slash_by_zero() -> NativeError {
    NativeError::new(ThrowClass::ArithmeticException, "/ by zero".to_owned())
}

/// The `Infinite or NaN` (`quot`/`rem`/`mod` double paths with a
/// non-finite operand or quotient, fixprobe `:quot-nan-div`).
fn infinite_or_nan() -> NativeError {
    NativeError::new(
        ThrowClass::NumberFormatException,
        "Infinite or NaN".to_owned(),
    )
}

/// Promoted `+`: `Int`/`Int` keeps the checked errors;
/// otherwise the top rank present computes (`Double` > `BigDec` >
/// `Ratio` > `BigInt`, oracle `promo1`). A bound
/// `*math-context*` rounds `BigDec`-involved pairs (S2).
fn add(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => x.checked_add(*y).map(Num::Int).ok_or_else(overflow),
        _ if either_double(&a, &b) => Ok(Num::Double(to_f64(a) + to_f64(b))),
        _ if num_rank(&a).max(num_rank(&b)) == 3 => {
            let (x, y, mc) = dec_pair(eval, &a, &b)?;
            Ok(Num::BigDec(match mc {
                None => x.add(&y),
                Some(mc) => x.add_mc(&y, &mc).map_err(arithmetic)?,
            }))
        }
        _ => Ok(num_ratio(num_rational(&a) + num_rational(&b))),
    }
}

/// Promoted `-` (same lattice as [`add`]).
fn sub(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => x.checked_sub(*y).map(Num::Int).ok_or_else(overflow),
        _ if either_double(&a, &b) => Ok(Num::Double(to_f64(a) - to_f64(b))),
        _ if num_rank(&a).max(num_rank(&b)) == 3 => {
            let (x, y, mc) = dec_pair(eval, &a, &b)?;
            Ok(Num::BigDec(match mc {
                None => x.sub(&y),
                Some(mc) => x.sub_mc(&y, &mc).map_err(arithmetic)?,
            }))
        }
        _ => Ok(num_ratio(num_rational(&a) - num_rational(&b))),
    }
}

/// Promoted `*` (same lattice as [`add`]).
fn mul(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => x.checked_mul(*y).map(Num::Int).ok_or_else(overflow),
        _ if either_double(&a, &b) => Ok(Num::Double(to_f64(a) * to_f64(b))),
        _ if num_rank(&a).max(num_rank(&b)) == 3 => {
            let (x, y, mc) = dec_pair(eval, &a, &b)?;
            Ok(Num::BigDec(match mc {
                None => x.mul(&y),
                Some(mc) => x.mul_mc(&y, &mc).map_err(arithmetic)?,
            }))
        }
        _ => Ok(num_ratio(num_rational(&a) * num_rational(&b))),
    }
}

/// Promoted `/`: `Int`/`Int` is exact when divisible (a reduced
/// ratio otherwise — `(/ 1 2)` is `1/2`, never a `Double`); any
/// `MIN` side routes through `BigIntOps.divide`, so integral
/// quotients (`MIN/-1`, `MIN`/`MIN`, `MIN`/2, `0`/`MIN`) come
/// back `BigInt`. Tower division is exact (`(/ 5N 2)` → `5/2`, a
/// non-terminating decimal fails like `(bigdec 1/3)`), or rounded
/// under a bound `*math-context*` (S2).
/// `Int`/`Int` division (split from [`div`] for the complexity
/// budget): exact when divisible, else a reduced ratio; any `MIN`
/// side takes the exact path, where integral quotients fold to
/// `BigInt` (`LongOps.divide` → `BigIntOps.divide`: `(/ MIN
/// MIN)` is `1N`, `(/ 0 MIN)` is `0N`).
fn div_int(x: i64, y: i64, a: &Num, b: &Num) -> Result<Num, NativeError> {
    if y == 0 {
        return Err(divide_by_zero());
    }
    if x == i64::MIN || y == i64::MIN {
        return Ok(num_ratio(num_rational(a) / num_rational(b)));
    }
    // `checked_div` fails only on `MIN/-1` (a `MIN` side,
    // returned above) and `/0` (guarded above): `expect` keeps
    // the invariant while staying on the covered path (a `None`
    // arm would be an uncoverable line under the 100% floor).
    let q = x
        .checked_div(y)
        .expect("non-MIN operands divide without overflow");
    if x % y == 0 {
        Ok(Num::Int(q))
    } else {
        Ok(num_ratio(num_rational(a) / num_rational(b)))
    }
}

fn div(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => div_int(*x, *y, &a, &b),
        _ if either_double(&a, &b) => Ok(Num::Double(to_f64(a) / to_f64(b))),
        _ if num_rank(&a).max(num_rank(&b)) == 3 => {
            let (x, y, mc) = dec_pair(eval, &a, &b)?;
            Ok(Num::BigDec(match mc {
                None => x.div_exact(&y).map_err(arithmetic)?,
                Some(mc) => x.div_mc(&y, &mc).map_err(arithmetic)?,
            }))
        }
        _ => {
            let (x, y) = (num_rational(&a), num_rational(&b));
            if y.numer().is_zero() {
                return Err(divide_by_zero());
            }
            Ok(num_ratio(x / y))
        }
    }
}

/// A left-associative variadic fold (`+`/`-`/`*`/`/`; every caller
/// rejects empty args first).
fn fold(
    eval: &mut Evaluator,
    args: &[Value],
    op: fn(&mut Evaluator, Num, Num) -> Result<Num, NativeError>,
) -> Result<Value, NativeError> {
    if args.len() == 1 {
        // Single args pass through unchecked (probe39 shapes).
        return Ok(args[0].clone());
    }
    let mut acc = as_num(&args[0])?;
    for arg in &args[1..] {
        acc = op(eval, acc, as_num(arg)?)?;
    }
    Ok(num_value(acc))
}

fn native_add(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Int(0));
    }
    fold(eval, args, add)
}

fn native_sub(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/-", 0));
    }
    if args.len() == 1 {
        return match as_num(&args[0])? {
            Num::Int(x) => x.checked_neg().map(Value::Int).ok_or_else(overflow),
            Num::BigInt(x) => Ok(Value::BigInt(Arc::new(-x))),
            Num::Ratio(x) => Ok(num_value(num_ratio(-x))),
            Num::BigDec(x) => Ok(bigdec_result(match math_context_of(eval)? {
                None => x.neg(),
                Some(mc) => x.neg_mc(&mc).map_err(arithmetic)?,
            })),
            Num::Double(x) => Ok(Value::Double(-x)),
        };
    }
    fold(eval, args, sub)
}

fn native_mul(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Int(1));
    }
    fold(eval, args, mul)
}

fn native_div(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core//", 0));
    }
    if args.len() == 1 {
        return div(eval, Num::Int(1), as_num(&args[0])?).map(num_value);
    }
    fold(eval, args, div)
}

fn native_quot(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/quot", args, &[2])?;
    let (a, b) = (as_num(&args[0])?, as_num(&args[1])?);
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => {
            if *y == 0 {
                return Err(slash_by_zero());
            }
            // `LongOps.quotient` is raw Java `/`: `MIN/-1` is `MIN`
            // (JLS 15.17.2), never an overflow error.
            Ok(Value::Int(x.wrapping_div(*y)))
        }
        _ if either_double(&a, &b) => quot_double(to_f64(a), to_f64(b)),
        // Tower truncating division (oracle `quotrem`/`quotmix`/
        // `quotdec`): any tower zero divisor is `Divide by zero`
        // (unlike the `Int` `/ by zero` wedge, which stays).
        _ => quot_tower(eval, &a, &b),
    }
}

/// `quot` over a `Double` pair: zero first (even over `NaN`,
/// `:quot-nan-0`), then finiteness (either side, `:quot-nan-div`),
/// then the quotient's own finiteness (`:quot-ovf`); `-0.0`
/// normalizes to `+0.0` (`:quot-negfrac`, `:quot-2-neginf`).
fn quot_double(x: f64, y: f64) -> Result<Value, NativeError> {
    if y == 0.0 {
        return Err(divide_by_zero());
    }
    if !x.is_finite() || y.is_nan() {
        return Err(infinite_or_nan());
    }
    let q = (x / y).trunc();
    if !q.is_finite() {
        return Err(infinite_or_nan());
    }
    Ok(Value::Double(if q == 0.0 { 0.0 } else { q }))
}

/// `quot` over tower kinds: exact truncating division — or the
/// preferred-scale integral quotient rounded to the bound
/// `*math-context*` (S2 oracles `r2-quot`/`r7-quotmix`/
/// `r8-quotscale`).
fn quot_tower(eval: &mut Evaluator, a: &Num, b: &Num) -> Result<Value, NativeError> {
    if num_rational(b).numer().is_zero() {
        return Err(divide_by_zero());
    }
    if num_rank(a).max(num_rank(b)) == 3 {
        let (x, y, mc) = dec_pair(eval, a, b)?;
        Ok(bigdec_result(match mc {
            None => x.quot_trunc(&y),
            Some(mc) => x.quot_mc(&y, &mc).map_err(arithmetic)?,
        }))
    } else {
        let (x, y) = (num_rational(a), num_rational(b));
        Ok(num_value(num_ratio(trunc_div(&x, &y))))
    }
}

/// Truncating rational division toward zero (the `quot` rule for
/// `BigInt`/`Ratio` operands).
fn trunc_div(
    x: &num_rational::BigRational,
    y: &num_rational::BigRational,
) -> num_rational::BigRational {
    let q = x.numer() * y.denom() / (x.denom() * y.numer());
    num_rational::BigRational::from_integer(q)
}

fn native_rem(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/rem", args, &[2])?;
    let (a, b) = (as_num(&args[0])?, as_num(&args[1])?);
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => {
            if *y == 0 {
                return Err(slash_by_zero());
            }
            // `LongOps.remainder` is raw Java `%`: `MIN % -1` is `0`
            // (JLS 15.17.3), never an overflow error.
            Ok(Value::Int(x.wrapping_rem(*y)))
        }
        _ if either_double(&a, &b) => rem_double(to_f64(a), to_f64(b)),
        // Tower remainder (oracle `dec-each`): exact, in the
        // dividend's kind (`(rem 5M 2)` → `1M`, `(rem 1/2 1)` →
        // `1/2`).
        _ => rem_tower(eval, &a, &b),
    }
}

/// `rem` over a `Double` pair: zero first (`:rem-nan-0`), then
/// finiteness (`:rem-nan-num`); an infinite divisor is `NaN`, NOT
/// `x` (`:rem-num-inf`, `:rem-num-neginf`).
fn rem_double(x: f64, y: f64) -> Result<Value, NativeError> {
    if y == 0.0 {
        return Err(divide_by_zero());
    }
    if !x.is_finite() || y.is_nan() {
        return Err(infinite_or_nan());
    }
    if !y.is_finite() {
        return Ok(Value::Double(f64::NAN));
    }
    Ok(Value::Double(x % y))
}

/// `rem` over tower kinds: exact, in the dividend's kind. The
/// ratio conversion still consults a bound `*math-context*`
/// (S2 oracles `r7-remread`/`r7-remmix`); the remainder itself
/// never rounds (S2 oracles `wp-rem`/`r3-rem`).
fn rem_tower(eval: &mut Evaluator, a: &Num, b: &Num) -> Result<Value, NativeError> {
    if num_rational(b).numer().is_zero() {
        return Err(divide_by_zero());
    }
    if num_rank(a).max(num_rank(b)) == 3 {
        let (x, y, _) = dec_pair(eval, a, b)?;
        Ok(bigdec_result(x.rem(&y)))
    } else {
        let (x, y) = (num_rational(a), num_rational(b));
        let q = trunc_div(&x, &y);
        Ok(num_value(num_ratio(x - q * y)))
    }
}

fn native_mod(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/mod", args, &[2])?;
    let (a, b) = (as_num(&args[0])?, as_num(&args[1])?);
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => {
            if *y == 0 {
                return Err(divide_by_zero());
            }
            // Java `%` (`MIN % -1` is `0`, so `(mod MIN -1)` is `0`);
            // the sign adjust below cannot overflow (`|r| < |y|` with
            // differing signs keeps `|r + y| < |y|` — still checked).
            let r = x.wrapping_rem(*y);
            Ok(Value::Int(if r == 0 || (r > 0) == (*y > 0) {
                r
            } else {
                r.checked_add(*y).ok_or_else(overflow)?
            }))
        }
        _ if either_double(&a, &b) => mod_double(to_f64(a), to_f64(b)),
        // Tower modulus (oracle `dec-each`): the remainder
        // adjusted to the divisor's sign, in the dividend's kind.
        _ => mod_tower(eval, &a, &b),
    }
}

/// `mod` over a `Double` pair: zero first (`:mod-nan-0`), then
/// finiteness (`:mod-nan-num`); an infinite divisor falls out of
/// the formula as `NaN` (`:mod-num-inf`).
fn mod_double(x: f64, y: f64) -> Result<Value, NativeError> {
    if y == 0.0 {
        return Err(divide_by_zero());
    }
    if !x.is_finite() || y.is_nan() {
        return Err(infinite_or_nan());
    }
    Ok(Value::Double(((x % y) + y) % y))
}

/// `mod` over tower kinds: the remainder adjusted to the
/// divisor's sign, in the dividend's kind (conversion
/// consults a bound context like `rem`, the result never
/// rounds — S2 oracles `wp-mod`/`r7-remmix`).
fn mod_tower(eval: &mut Evaluator, a: &Num, b: &Num) -> Result<Value, NativeError> {
    if num_rational(b).numer().is_zero() {
        return Err(divide_by_zero());
    }
    if num_rank(a).max(num_rank(b)) == 3 {
        let (x, y, _) = dec_pair(eval, a, b)?;
        Ok(bigdec_result(bigdec_mod(&x, &y)))
    } else {
        mod_rational(&num_rational(a), &num_rational(b))
    }
}

/// `mod` over two exact rationals (sign-adjusted remainder,
/// folded).
fn mod_rational(
    x: &num_rational::BigRational,
    y: &num_rational::BigRational,
) -> Result<Value, NativeError> {
    let q = trunc_div(x, y);
    let y_sign = y.numer().sign();
    let r = x - q * y.clone();
    Ok(num_value(num_ratio(
        if r.numer().is_zero() || r.numer().sign() == y_sign {
            r
        } else {
            r + y.clone()
        },
    )))
}

/// `BigDec` modulus: the remainder adjusted to the divisor's
/// sign (mirrors the `Int` formula above).
fn bigdec_mod(x: &clj_value::BigDec, y: &clj_value::BigDec) -> clj_value::BigDec {
    let r = x.rem(y);
    if r.is_zero() || r.sign() == y.sign() {
        r
    } else {
        r.add(y)
    }
}

/// A number as `f64` (mixed/double paths; tower narrows
/// approximately, infinities included).
fn to_f64(n: Num) -> f64 {
    match n {
        Num::Int(i) => i as f64,
        Num::BigInt(n) => n.to_f64().unwrap_or(f64::NAN),
        Num::Ratio(r) => r.to_f64().unwrap_or(f64::NAN),
        Num::BigDec(d) => d.to_f64(),
        Num::Double(d) => d,
    }
}

/// An exact rational for a number. `Double` never reaches here
/// (doubles always take the `f64` path); the `expect` documents
/// that invariant coverably.
fn num_rational(n: &Num) -> num_rational::BigRational {
    match n {
        Num::Int(i) => num_rational::BigRational::from_integer(num_bigint::BigInt::from(*i)),
        Num::BigInt(n) => num_rational::BigRational::from_integer(n.clone()),
        Num::Ratio(r) => r.clone(),
        Num::BigDec(d) => d.to_rational(),
        Num::Double(_) => {
            unreachable!("doubles take the f64 path, never the exact one")
        }
    }
}

/// Exact ordering without needless expansion: `BigDec` against
/// `BigDec`/`Int`/`BigInt` decides by sign and magnitude first
/// (astronomical scales never expand — `compareTo` parity);
/// everything else keeps the rational path.
fn compare_exact(a: &Num, b: &Num) -> std::cmp::Ordering {
    match (a, b) {
        (Num::BigDec(d), Num::BigDec(e)) => d.compare(e),
        (Num::BigDec(d), Num::Int(i)) => d.compare(&clj_value::BigDec::from_bigint((*i).into())),
        (Num::Int(i), Num::BigDec(e)) => clj_value::BigDec::from_bigint((*i).into()).compare(e),
        (Num::BigDec(d), Num::BigInt(n)) => d.compare(&clj_value::BigDec::from_bigint(n.clone())),
        (Num::BigInt(n), Num::BigDec(d)) => clj_value::BigDec::from_bigint(n.clone()).compare(d),
        _ => num_rational(a).cmp(&num_rational(b)),
    }
}

/// An exact `BigDec` for a non-`Double` number (a `Ratio` with no
/// terminating expansion fails like `(bigdec 1/3)`).
fn num_bigdec(n: &Num) -> Result<clj_value::BigDec, NativeError> {
    match n {
        Num::Int(i) => Ok(clj_value::BigDec::from_bigint(num_bigint::BigInt::from(*i))),
        Num::BigInt(n) => Ok(clj_value::BigDec::from_bigint(n.clone())),
        Num::BigDec(d) => Ok(d.clone()),
        Num::Ratio(r) => clj_value::BigDec::from_rational_exact(r).map_err(|m| {
            NativeError::new(clj_value::ThrowClass::ArithmeticException, m.to_owned())
        }),
        Num::Double(_) => {
            unreachable!("rank dispatch keeps doubles out of BigDec ops")
        }
    }
}

/// An exact `BigDec` under a context (S2): ratios divide with
/// guard digits (oracle `wp-mixratio`); everything else is exact.
fn num_bigdec_mc(n: &Num, mc: &clj_value::MathContext) -> Result<clj_value::BigDec, NativeError> {
    match n {
        Num::Ratio(r) => clj_value::BigDec::from_ratio_mc(r, mc).map_err(arithmetic),
        _ => num_bigdec(n),
    }
}

/// Both sides as decimals plus the bound context (S2): the
/// context resolves only when a `BigDec`-involved op needs it
/// (oracle `r5-garbage-int`); ratios convert exactly without a
/// context, via guarded division with one (oracles
/// `wp-nomixratio`/`wp-mixratio`).
fn dec_pair(
    eval: &mut Evaluator,
    a: &Num,
    b: &Num,
) -> Result<
    (
        clj_value::BigDec,
        clj_value::BigDec,
        Option<clj_value::MathContext>,
    ),
    NativeError,
> {
    match math_context_of(eval)? {
        None => Ok((num_bigdec(a)?, num_bigdec(b)?, None)),
        Some(mc) => Ok((num_bigdec_mc(a, &mc)?, num_bigdec_mc(b, &mc)?, Some(mc))),
    }
}

/// A bound `*math-context*` (S2): precision + rounding. Stored
/// as an extension value (no new `Value` kind); `=`/`hash` are
/// structural like `MathContext.equals` (oracles
/// `wp-eq2`/`wp-eq3`/`r2-eqrm*`).
#[derive(Clone, Debug)]
pub(crate) struct MathContextVal {
    pub id: u64,
    pub ctx: clj_value::MathContext,
}

impl ExtensionValue for MathContextVal {
    fn class_name(&self) -> String {
        "java.math.MathContext".to_owned()
    }

    fn print(&self) -> String {
        format!(
            "#object[java.math.MathContext 0x{:x} \"{}\"]",
            self.id,
            self.ctx.to_context_string()
        )
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<MathContextVal>()
            .is_some_and(|o| o.ctx == self.ctx)
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.ctx.precision());
        state.write(self.ctx.rounding().name().as_bytes());
    }

    fn print_str(&self) -> Option<String> {
        Some(self.ctx.to_context_string())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The visible `*math-context*` (S2): nil (or an uninstalled
/// var) is no context; anything bound must BE a context —
/// garbage fails the `MathContext` cast exactly where a
/// `BigDec` op consults it (oracles `r5-garbage*`).
fn math_context_of(eval: &Evaluator) -> Result<Option<clj_value::MathContext>, NativeError> {
    match deref_core(eval, "*math-context*") {
        Value::Nil => Ok(None),
        Value::Extension(e) => match e.as_any().downcast_ref::<MathContextVal>() {
            Some(m) => Ok(Some(m.ctx)),
            None => Err(cce_err(
                &Value::Extension(e.clone()),
                "java.math.MathContext",
            )),
        },
        v => Err(cce_err(&v, "java.math.MathContext")),
    }
}

/// A rational as a `Num` (denominator 1 folds to `BigInt`).
fn num_ratio(r: num_rational::BigRational) -> Num {
    if r.denom() == &num_bigint::BigInt::from(1) {
        Num::BigInt(r.numer().clone())
    } else {
        Num::Ratio(r)
    }
}

/// An `Ordering` as the `f64` pair the chained comparisons take
/// (`<`/`>`/`<=`/`>=` all read correctly off `(ord, 0.0)`).
fn ord_num(ord: std::cmp::Ordering) -> f64 {
    match ord {
        std::cmp::Ordering::Less => -1.0,
        std::cmp::Ordering::Equal => 0.0,
        std::cmp::Ordering::Greater => 1.0,
    }
}

/// The sign of a number (tower via the exact rational, whose
/// denominator is always positive; `Double` keeps its own arms
/// at the call sites).
fn num_sign(n: &Num) -> num_bigint::Sign {
    num_rational(n).numer().sign()
}

/// Whether either side is a `Double` (the approximate path).
fn either_double(a: &Num, b: &Num) -> bool {
    matches!(a, Num::Double(_)) || matches!(b, Num::Double(_))
}

/// The rational `1` (tower steps).
fn rat_unit() -> num_rational::BigRational {
    num_rational::BigRational::from_integer(num_bigint::BigInt::from(1))
}

/// One step (tower kinds keep their kind: `(inc 5M)` → `6M`,
/// `(inc 1/2)` → `3/2`; `Int` still overflows — the wedge). A
/// bound `*math-context*` rounds the `BigDec` step (S2 oracle
/// `wp-inc`).
fn inc_one(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) => x.checked_add(1).map(Value::Int).ok_or_else(overflow),
        Num::BigInt(x) => Ok(Value::BigInt(Arc::new(x + 1))),
        Num::Ratio(x) => Ok(num_value(num_ratio(x + rat_unit()))),
        Num::BigDec(x) => {
            let one = clj_value::BigDec::from_bigint(num_bigint::BigInt::from(1));
            Ok(bigdec_result(match math_context_of(eval)? {
                None => x.add(&one),
                Some(mc) => x.add_mc(&one, &mc).map_err(arithmetic)?,
            }))
        }
        Num::Double(x) => Ok(Value::Double(x + 1.0)),
    }
}

/// One step down (mirror of [`inc_one`], S2 oracle `wp-dec`).
fn dec_one(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) => x.checked_sub(1).map(Value::Int).ok_or_else(overflow),
        Num::BigInt(x) => Ok(Value::BigInt(Arc::new(x - 1))),
        Num::Ratio(x) => Ok(num_value(num_ratio(x - rat_unit()))),
        Num::BigDec(x) => {
            let one = clj_value::BigDec::from_bigint(num_bigint::BigInt::from(1));
            Ok(bigdec_result(match math_context_of(eval)? {
                None => x.sub(&one),
                Some(mc) => x.sub_mc(&one, &mc).map_err(arithmetic)?,
            }))
        }
        Num::Double(x) => Ok(Value::Double(x - 1.0)),
    }
}

fn native_inc(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/inc", args, &[1])?;
    inc_one(eval, as_num(&args[0])?)
}

fn native_dec(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/dec", args, &[1])?;
    dec_one(eval, as_num(&args[0])?)
}

/// `inc'`: auto-promoting increment — `MAX` grows to
/// `BigInt` (the S11 in-slice promotion fix; oracles `ip-*`),
/// everything else keeps the `inc` rules.
fn native_inc_prime(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/inc'", args, &[1])?;
    inc_prime_one(eval, as_num(&args[0])?)
}

/// `max`/`min`: ties keep the LATTER (`(max 1 1.0)` → `1.0`),
/// and `NaN` poisons from either side (`(max 1 ##NaN)` and `(max
/// ##NaN 1)` are both `##NaN`); single args pass through unchecked
/// (fixprobe `:max-tie`, `:max-nan1/2`).
fn minmax(display: &str, args: &[Value], want_max: bool) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity(display, 0));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    let mut acc = as_num(&args[0])?;
    // The winning INPUT answers (upstream returns the object,
    // not a recomputation — identity preserved).
    let mut winner = 0;
    for (i, arg) in args[1..].iter().enumerate() {
        let n = as_num(arg)?;
        // The incumbent survives only on a strict win; anything
        // else (a tie, or a latter-side `NaN` since every `NaN`
        // comparison is false) takes the latter — except a `NaN`
        // incumbent, which always survives (`:max-nan1`). Tower
        // pairs without a `Double` compare exactly.
        let keep = match (&acc, &n) {
            (Num::Int(a), Num::Int(b)) => {
                if want_max {
                    a > b
                } else {
                    a < b
                }
            }
            _ if either_double(&acc, &n) => {
                let (a, b) = (to_f64(acc.clone()), to_f64(n.clone()));
                if a.is_nan() {
                    true
                } else if want_max {
                    a > b
                } else {
                    a < b
                }
            }
            _ => {
                let ord = compare_exact(&acc, &n);
                if want_max {
                    ord == std::cmp::Ordering::Greater
                } else {
                    ord == std::cmp::Ordering::Less
                }
            }
        };
        if !keep {
            acc = n;
            winner = i + 1;
        }
    }
    Ok(args[winner].clone())
}

fn native_max(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    minmax("clojure.core/max", args, true)
}

fn native_min(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    minmax("clojure.core/min", args, false)
}

/// `=` as `Util.equiv` (sequential pairs walk pairwise with
/// early exit, so finite-vs-infinite answers `false` instead of
/// hanging; maps/sets recurse; the rest is `==`).
fn eq_bool(eval: &mut Evaluator, args: &[Value]) -> Result<bool, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/=", 0));
    }
    let mut first = &args[0];
    for arg in &args[1..] {
        if !equiv_vals(eval, first, arg).map_err(signal_to_native)? {
            return Ok(false);
        }
        first = arg;
    }
    Ok(true)
}

fn native_eq(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(eq_bool(eval, args)?))
}

fn native_not_eq(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(!eq_bool(eval, args)?))
}

/// Cross-category numeric equivalence (`(== 1 1.0)` is true;
/// `NaN` never is, not even to itself).
fn native_num_eq(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/==", 0));
    }
    if args.len() == 1 {
        // Single args pass unchecked (`(== :a)` → `true`,
        // fixprobe `:eqeq-1kw`).
        return Ok(Value::Bool(true));
    }
    let mut prev = as_num(&args[0])?;
    for arg in &args[1..] {
        let n = as_num(arg)?;
        // `Int`/`Int` is exact; `Double` pairs compare as `f64`
        // (`NaN` never is); tower pairs compare exactly.
        let same = match (&prev, &n) {
            (Num::Int(a), Num::Int(b)) => a == b,
            _ if either_double(&prev, &n) => to_f64(prev.clone()) == to_f64(n.clone()),
            _ => num_rational(&prev) == num_rational(&n),
        };
        if !same {
            return Ok(Value::Bool(false));
        }
        prev = n;
    }
    Ok(Value::Bool(true))
}

/// A chained numeric comparison (`<`/`>`/`<=`/`>=`, IEEE for
/// doubles, so `NaN` orders false everywhere).
fn chain_cmp(
    display: &str,
    args: &[Value],
    op: fn(f64, f64) -> bool,
    int_op: fn(i64, i64) -> bool,
) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity(display, 0));
    }
    if args.len() == 1 {
        // Single args pass unchecked (`(< "s")` → `true`,
        // fixprobe `:lt-1str`).
        return Ok(Value::Bool(true));
    }
    let mut prev = as_num(&args[0])?;
    for arg in &args[1..] {
        let n = as_num(arg)?;
        // `Int`/`Int` uses the integer predicate; `Double` pairs
        // use the `f64` one (`NaN` orders false everywhere);
        // tower pairs compare exactly, read off `(ord, 0.0)`.
        let holds = match (&prev, &n) {
            (Num::Int(a), Num::Int(b)) => int_op(*a, *b),
            _ if either_double(&prev, &n) => op(to_f64(prev.clone()), to_f64(n.clone())),
            _ => op(ord_num(compare_exact(&prev, &n)), 0.0),
        };
        if !holds {
            return Ok(Value::Bool(false));
        }
        prev = n;
    }
    Ok(Value::Bool(true))
}

fn native_lt(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    chain_cmp("clojure.core/<", args, |a, b| a < b, |a, b| a < b)
}

fn native_gt(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    chain_cmp("clojure.core/>", args, |a, b| a > b, |a, b| a > b)
}

fn native_lte(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    chain_cmp("clojure.core/<=", args, |a, b| a <= b, |a, b| a <= b)
}

fn native_gte(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    chain_cmp("clojure.core/>=", args, |a, b| a >= b, |a, b| a >= b)
}

/// The sign of a tower number as a `zero?`/`pos?`/`neg?`
/// answer (`Double` keeps its historic arms above).
fn tower_sign(n: &Num, want: num_bigint::Sign) -> bool {
    num_sign(n) == want
}

fn native_zero(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/zero?", args, &[1])?;
    Ok(Value::Bool(match as_num(&args[0])? {
        Num::Int(x) => x == 0,
        Num::Double(x) => x == 0.0,
        n => tower_sign(&n, num_bigint::Sign::NoSign),
    }))
}

fn native_pos(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/pos?", args, &[1])?;
    Ok(Value::Bool(match as_num(&args[0])? {
        Num::Int(x) => x > 0,
        Num::Double(x) => x > 0.0,
        n => tower_sign(&n, num_bigint::Sign::Plus),
    }))
}

fn native_neg(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/neg?", args, &[1])?;
    Ok(Value::Bool(match as_num(&args[0])? {
        Num::Int(x) => x < 0,
        Num::Double(x) => x < 0.0,
        n => tower_sign(&n, num_bigint::Sign::Minus),
    }))
}

/// JVM `compare` with exact cast failures (C2): nils sort first,
/// numbers cross-compare (the `<`/`>` pair, so `NaN` ties
/// everything), vectors compare lexicographically (recursively),
/// and anything else is `x.compareTo(y)` — the checkcast of `y`
/// to `x`'s class, except numbers (whose target is `Number`) and
/// non-`Comparable`s (whose own class fails against
/// `Comparable`). All probed.
pub(crate) fn jvm_compare(x: &Value, y: &Value) -> Result<std::cmp::Ordering, NativeError> {
    if let Some(ord) = compare_nil(x, y) {
        return Ok(ord);
    }
    if let Some(ord) = compare_numbers(x, y)? {
        return Ok(ord);
    }
    if let Some(ord) = compare_same_type(x, y)? {
        return Ok(ord);
    }
    Err(compare_cast_err(x, y))
}

/// Nils sort first (`nil` is less than everything, equal to itself).
fn compare_nil(x: &Value, y: &Value) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    match (x, y) {
        (Value::Nil, Value::Nil) => Some(Ordering::Equal),
        (Value::Nil, _) => Some(Ordering::Less),
        (_, Value::Nil) => Some(Ordering::Greater),
        _ => None,
    }
}

/// Numeric cross-compare (`None` when `x` is not numeric, so the
/// caller falls through; a numeric `x` against a non-numeric `y`
/// fails `y` to `Number`).
fn compare_numbers(x: &Value, y: &Value) -> Result<Option<std::cmp::Ordering>, NativeError> {
    use std::cmp::Ordering;
    if !is_number(x) {
        return Ok(None);
    }
    if !is_number(y) {
        return Err(cce_err(y, "java.lang.Number"));
    }
    let (a, b) = (as_num(x)?, as_num(y)?);
    // `Double` pairs compare as `f64` (the `<`/`>` pair, never
    // `total_cmp`: `NaN` ties); every other numeric pairing
    // compares exactly (oracle `cmpd`, `sort-mixed2`).
    if either_double(&a, &b) {
        let (a, b) = (to_f64(a), to_f64(b));
        if a < b {
            return Ok(Some(Ordering::Less));
        }
        if a > b {
            return Ok(Some(Ordering::Greater));
        }
        return Ok(Some(Ordering::Equal));
    }
    Ok(Some(compare_exact(&a, &b)))
}

/// Same-type scalars and vectors (`None` when the types differ).
fn compare_same_type(x: &Value, y: &Value) -> Result<Option<std::cmp::Ordering>, NativeError> {
    match (x, y) {
        (Value::Str(a), Value::Str(b)) => Ok(Some(compare_utf16(a, b))),
        (Value::Char(a), Value::Char(b)) => Ok(Some(a.cmp(b))),
        (Value::Bool(a), Value::Bool(b)) => Ok(Some(a.cmp(b))),
        (Value::Keyword(a), Value::Keyword(b)) => Ok(Some(compare_named(a, b))),
        (Value::Symbol(a), Value::Symbol(b)) => Ok(Some(compare_named(a, b))),
        (Value::Vector(a), Value::Vector(b)) => Ok(Some(compare_vectors(a, b)?)),
        // S9 scalars order within their kind (the port takes the
        // sign; the JVM returns raw `compareTo` ints).
        (Value::Uuid(_), Value::Uuid(_))
        | (Value::Instant(_), Value::Instant(_))
        | (Value::Uri(_), Value::Uri(_)) => Ok(Some(compare_scalar_pair(x, y))),
        // Entries order exactly like their 2-vectors (Oracle FX2
        // `compare`, `sort-entries`, `maxkey`).
        (Value::MapEntry { .. }, Value::Vector(_))
        | (Value::Vector(_), Value::MapEntry { .. })
        | (Value::MapEntry { .. }, Value::MapEntry { .. }) => compare_entry_pair(x, y),
        _ => Ok(None),
    }
}

/// S9 scalar order within one kind (split from
/// `compare_same_type` for the complexity budget).
fn compare_scalar_pair(x: &Value, y: &Value) -> std::cmp::Ordering {
    match (x, y) {
        (Value::Uuid(a), Value::Uuid(b)) => a.cmp_uuid(b),
        (Value::Instant(a), Value::Instant(b)) => a.cmp_instant(b),
        (Value::Uri(a), Value::Uri(b)) => a.compare_uri(b).cmp(&0),
        _ => unreachable!("scalar gate"),
    }
}

/// Entry pairs order as their 2-vectors (split from
/// `compare_same_type` for the complexity budget).
fn compare_entry_pair(x: &Value, y: &Value) -> Result<Option<std::cmp::Ordering>, NativeError> {
    match (x, y) {
        (Value::MapEntry { key, val, .. }, Value::Vector(b)) => {
            let tmp = clj_value::Vector::from_vec(vec![(**key).clone(), (**val).clone()]);
            Ok(Some(compare_vectors(&tmp, b)?))
        }
        (Value::Vector(a), Value::MapEntry { key, val, .. }) => {
            let tmp = clj_value::Vector::from_vec(vec![(**key).clone(), (**val).clone()]);
            Ok(Some(compare_vectors(a, &tmp)?))
        }
        (
            Value::MapEntry {
                key: k1, val: v1, ..
            },
            Value::MapEntry {
                key: k2, val: v2, ..
            },
        ) => {
            let a = clj_value::Vector::from_vec(vec![(**k1).clone(), (**v1).clone()]);
            let b = clj_value::Vector::from_vec(vec![(**k2).clone(), (**v2).clone()]);
            Ok(Some(compare_vectors(&a, &b)?))
        }
        _ => unreachable!("entry gate"),
    }
}

/// Lexicographic vector order (recursive element `compare`).
fn compare_vectors(
    a: &clj_value::Vector,
    b: &clj_value::Vector,
) -> Result<std::cmp::Ordering, NativeError> {
    use std::cmp::Ordering;
    let mut ai = a.iter();
    let mut bi = b.iter();
    loop {
        match (ai.next(), bi.next()) {
            (None, None) => return Ok(Ordering::Equal),
            (None, _) => return Ok(Ordering::Less),
            (_, None) => return Ok(Ordering::Greater),
            (Some(u), Some(v)) => match jvm_compare(&u, &v)? {
                Ordering::Equal => {}
                ord => return Ok(ord),
            },
        }
    }
}

/// `x.compareTo(y)`: checkcast `y` to `x`'s class (vectors name
/// the interface); non-`Comparable` `x` fails itself.
fn compare_cast_err(x: &Value, y: &Value) -> NativeError {
    let target = match x {
        Value::Str(_) => "java.lang.String",
        Value::Char(_) => "java.lang.Character",
        Value::Bool(_) => "java.lang.Boolean",
        Value::Keyword(_) => "clojure.lang.Keyword",
        Value::Symbol(_) => "clojure.lang.Symbol",
        // S9 scalars are `Comparable` (oracles `u-cmp-str`,
        // `i-cmp-str`, `r-cmp-str` pin the texts).
        Value::Uuid(_) => "java.util.UUID",
        Value::Instant(_) => "java.util.Date",
        Value::Uri(_) => "java.net.URI",
        Value::Vector(_) | Value::MapEntry { .. } => "clojure.lang.IPersistentVector",
        _ => return cce_err(x, "java.lang.Comparable"),
    };
    cce_err(y, target)
}

/// UTF-16 code-unit order (the JVM's `String.compareTo`).
fn compare_utf16(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// Keyword/symbol order (namespace, then name — like the JVM's
/// `compareTo`, which orders the composite).
fn compare_named<A, B>(a: &A, b: &B) -> std::cmp::Ordering
where
    A: NamedParts,
    B: NamedParts,
{
    (a.ns(), a.nm()).cmp(&(b.ns(), b.nm()))
}

/// Namespace/name parts for ordering.
trait NamedParts {
    fn ns(&self) -> Option<&str>;
    fn nm(&self) -> &str;
}

impl NamedParts for Keyword {
    fn ns(&self) -> Option<&str> {
        self.namespace()
    }
    fn nm(&self) -> &str {
        self.name()
    }
}

impl NamedParts for Symbol {
    fn ns(&self) -> Option<&str> {
        self.namespace()
    }
    fn nm(&self) -> &str {
        self.name()
    }
}

fn native_compare(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/compare", args, &[2])?;
    Ok(Value::Int(match jvm_compare(&args[0], &args[1])? {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }))
}

/// A 1-arity predicate native.
macro_rules! pred1 {
    ($name:ident, $display:literal, $test:expr) => {
        fn $name(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
            check_arity($display, args, &[1])?;
            let test: fn(&Value) -> bool = $test;
            Ok(Value::Bool(test(&args[0])))
        }
    };
}

fn is_nil(v: &Value) -> bool {
    matches!(v, Value::Nil)
}
fn is_some_val(v: &Value) -> bool {
    !is_nil(v)
}
fn is_truthy_val(v: &Value) -> bool {
    v.is_truthy()
}
fn is_falsy_val(v: &Value) -> bool {
    !v.is_truthy()
}
fn is_true(v: &Value) -> bool {
    matches!(v, Value::Bool(true))
}
fn is_false(v: &Value) -> bool {
    matches!(v, Value::Bool(false))
}
fn is_boolean(v: &Value) -> bool {
    matches!(v, Value::Bool(_))
}
fn is_integer(v: &Value) -> bool {
    // `BigInteger` is an integer (S2 oracle `r8-integerp`).
    matches!(v, Value::Int(_) | Value::BigInt(_) | Value::BigInteger(_))
}
/// `int?` stays `Int`-only (oracle: `(int? 5N)` is false — the
/// pinned `(int? 1)` wedge for `Long` stays too).
fn is_int(v: &Value) -> bool {
    matches!(v, Value::Int(_))
}
fn is_double(v: &Value) -> bool {
    matches!(v, Value::Double(_))
}
fn is_number(v: &Value) -> bool {
    matches!(
        v,
        Value::Int(_)
            | Value::Double(_)
            | Value::BigInt(_)
            | Value::BigInteger(_)
            | Value::BigDec(_)
            | Value::Ratio(_)
    )
}
fn is_ratio(v: &Value) -> bool {
    matches!(v, Value::Ratio(_))
}
fn is_decimal(v: &Value) -> bool {
    matches!(v, Value::BigDec(_))
}
/// Upstream `(or (integer? n) (ratio? n) (decimal? n))` —
/// decimals INCLUDED (S2 oracles `pred-rat-*`).
fn is_rational(v: &Value) -> bool {
    matches!(
        v,
        Value::Int(_)
            | Value::BigInt(_)
            | Value::BigInteger(_)
            | Value::Ratio(_)
            | Value::BigDec(_)
    )
}
fn is_string(v: &Value) -> bool {
    matches!(v, Value::Str(_))
}
fn is_char(v: &Value) -> bool {
    matches!(v, Value::Char(_))
}
fn is_keyword(v: &Value) -> bool {
    matches!(v, Value::Keyword(_))
}
fn is_symbol(v: &Value) -> bool {
    matches!(v, Value::Symbol(_))
}
fn is_ident(v: &Value) -> bool {
    is_keyword(v) || is_symbol(v)
}
fn ident_ns(v: &Value) -> Option<Option<&str>> {
    match v {
        Value::Keyword(k) => Some(k.namespace()),
        Value::Symbol(s) => Some(s.namespace()),
        _ => None,
    }
}
fn is_qualified_ident(v: &Value) -> bool {
    ident_ns(v).is_some_and(|ns| ns.is_some())
}
fn is_simple_ident(v: &Value) -> bool {
    ident_ns(v).is_some_and(|ns| ns.is_none())
}
fn is_qualified_keyword(v: &Value) -> bool {
    matches!(v, Value::Keyword(k) if k.namespace().is_some())
}
fn is_simple_keyword(v: &Value) -> bool {
    matches!(v, Value::Keyword(k) if k.namespace().is_none())
}
fn is_qualified_symbol(v: &Value) -> bool {
    matches!(v, Value::Symbol(s) if s.namespace().is_some())
}
fn is_simple_symbol(v: &Value) -> bool {
    matches!(v, Value::Symbol(s) if s.namespace().is_none())
}
fn is_coll(v: &Value) -> bool {
    matches!(
        v,
        Value::List(_)
            | Value::Vector(_)
            | Value::MapEntry { .. }
            | Value::Map(_)
            | Value::Set(_)
            | Value::PrimitiveVector(_)
    ) || as_subvector(v).is_some()
        || is_seq_value(v)
        || crate::types::is_host_record(v)
        || crate::types::is_host_struct(v)
}
fn is_sequential(v: &Value) -> bool {
    matches!(
        v,
        Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || as_subvector(v).is_some()
        || is_seq_value(v)
}
fn is_associative(v: &Value) -> bool {
    matches!(
        v,
        Value::Map(_) | Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || as_subvector(v).is_some()
        || crate::types::is_host_record(v)
        || crate::types::is_host_struct(v)
}
pub(crate) fn is_counted(v: &Value) -> bool {
    matches!(
        v,
        Value::List(_)
            | Value::Vector(_)
            | Value::MapEntry { .. }
            | Value::Map(_)
            | Value::Set(_)
            | Value::PrimitiveVector(_)
    ) || crate::coll::is_transient(v)
        || as_subvector(v).is_some()
        || is_counted_seq(v)
        || crate::types::is_host_record(v)
        || crate::types::is_host_struct(v)
        || crate::collmisc::as_chunk_buffer(v).is_some()
}
fn is_list(v: &Value) -> bool {
    matches!(v, Value::List(_))
}
fn is_vector(v: &Value) -> bool {
    matches!(
        v,
        Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || as_subvector(v).is_some()
}
fn is_map(v: &Value) -> bool {
    matches!(v, Value::Map(_)) || crate::types::is_host_record(v) || crate::types::is_host_struct(v)
}
fn is_set(v: &Value) -> bool {
    matches!(v, Value::Set(_))
}
// Sorted maps and sets answer true (FT6); everything else answers
// false, matching the oracle for all constructible inputs.
fn is_sorted(v: &Value) -> bool {
    match v {
        Value::Map(m) => m.is_sorted(),
        Value::Set(s) => s.is_sorted(),
        _ => false,
    }
}
// `IPersistentVector` (vectors, map entries, subvectors) implements
// `clojure.lang.Reversible`, as do sorted maps and sets (FT6).
// Lists, hash maps, hash sets, queues, seqs (including `rseq`),
// and transients do not (`docs/spec/t1192/probes/fx2b-reversible`).
fn is_reversible(v: &Value) -> bool {
    matches!(
        v,
        Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || as_subvector(v).is_some()
        || is_sorted(v)
}
fn is_var(v: &Value) -> bool {
    matches!(v, Value::Var(_))
}
fn is_seqable(v: &Value) -> bool {
    matches!(
        v,
        Value::Nil
            | Value::Str(_)
            | Value::List(_)
            | Value::Vector(_)
            | Value::MapEntry { .. }
            | Value::Map(_)
            | Value::Set(_)
            | Value::Array(_)
            | Value::PrimitiveVector(_)
    ) || as_subvector(v).is_some()
        || is_seq_value(v)
        || crate::types::is_host_record(v)
        || crate::types::is_host_struct(v)
}
// `record?` (S-typechain stage 1): true for record instances —
// including records with extension entries — and nothing else
// (deftype instances, descriptors, and plain maps all answer
// false).
fn is_record(v: &Value) -> bool {
    crate::types::is_host_record(v)
}
// `any?` (S1): constantly true (probe `p-any-*`).
fn is_any(_v: &Value) -> bool {
    true
}
// `delay?` (S1): delays only — futures, promises, and atoms
// answer false (probes `p-delay-*`).
fn is_delay(v: &Value) -> bool {
    matches!(v, Value::Delay(_))
}
// `map-entry?` (S1): map entries only, never 2-vectors
// (probes `p-mapseq-*`).
fn is_map_entry(v: &Value) -> bool {
    matches!(v, Value::MapEntry { .. })
}
// `indexed?` (S1): vectors, map entries, and subvectors;
// strings, seqs, maps, sets, and arrays answer false
// (probes `p-indexed-*`).
fn is_indexed(v: &Value) -> bool {
    matches!(
        v,
        Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || as_subvector(v).is_some()
}
// `nat-int?` / `neg-int?` / `pos-int?` (S1): composed over the
// port's `int?` (`Int`-only — BigInts answer false exactly like
// the oracle, probes `p-nat-*` / `p-neg-*` / `p-pos-*`).
fn is_nat_int(v: &Value) -> bool {
    matches!(v, Value::Int(n) if *n >= 0)
}
fn is_neg_int(v: &Value) -> bool {
    matches!(v, Value::Int(n) if *n < 0)
}
fn is_pos_int(v: &Value) -> bool {
    matches!(v, Value::Int(n) if *n > 0)
}
// `special-symbol?` (S1): the oracle's 23-member set (probes
// `p-special-*`); qualified symbols never qualify.
fn is_special_symbol(v: &Value) -> bool {
    match v {
        Value::Symbol(s) if s.namespace().is_none() => matches!(
            s.name(),
            "def"
                | "if"
                | "do"
                | "quote"
                | "var"
                | "fn*"
                | "loop*"
                | "recur"
                | "throw"
                | "try"
                | "catch"
                | "finally"
                | "monitor-enter"
                | "monitor-exit"
                | "."
                | "new"
                | "set!"
                | "&"
                | "let*"
                | "case*"
                | "deftype*"
                | "reify*"
                | "letfn*"
        ),
        _ => false,
    }
}

fn is_tagged_literal(v: &Value) -> bool {
    matches!(v, Value::TaggedLiteral { .. })
}
fn is_reader_conditional(v: &Value) -> bool {
    matches!(v, Value::ReaderConditional { .. })
}

pred1!(native_nil, "clojure.core/nil?", is_nil);
pred1!(native_not, "clojure.core/not", is_falsy_val);
pred1!(native_boolean, "clojure.core/boolean", is_truthy_val);
pred1!(native_somep, "clojure.core/some?", is_some_val);
pred1!(native_truep, "clojure.core/true?", is_true);
pred1!(native_falsep, "clojure.core/false?", is_false);
pred1!(native_booleanp, "clojure.core/boolean?", is_boolean);
pred1!(native_integerp, "clojure.core/integer?", is_integer);
pred1!(native_intp, "clojure.core/int?", is_int);
pred1!(native_floatp, "clojure.core/float?", is_double);
pred1!(native_doublep, "clojure.core/double?", is_double);
pred1!(native_numberp, "clojure.core/number?", is_number);
pred1!(native_ratiop, "clojure.core/ratio?", is_ratio);
pred1!(native_decimalp, "clojure.core/decimal?", is_decimal);
pred1!(native_rationalp, "clojure.core/rational?", is_rational);
pred1!(native_stringp, "clojure.core/string?", is_string);
pred1!(native_charp, "clojure.core/char?", is_char);
pred1!(native_keywordp, "clojure.core/keyword?", is_keyword);
pred1!(native_symbolp, "clojure.core/symbol?", is_symbol);
pred1!(native_identp, "clojure.core/ident?", is_ident);
pred1!(
    native_qualified_identp,
    "clojure.core/qualified-ident?",
    is_qualified_ident
);
pred1!(
    native_simple_identp,
    "clojure.core/simple-ident?",
    is_simple_ident
);
pred1!(
    native_qualified_keywordp,
    "clojure.core/qualified-keyword?",
    is_qualified_keyword
);
pred1!(
    native_simple_keywordp,
    "clojure.core/simple-keyword?",
    is_simple_keyword
);
pred1!(
    native_qualified_symbolp,
    "clojure.core/qualified-symbol?",
    is_qualified_symbol
);
pred1!(
    native_simple_symbolp,
    "clojure.core/simple-symbol?",
    is_simple_symbol
);
pred1!(native_collp, "clojure.core/coll?", is_coll);
pred1!(
    native_sequentialp,
    "clojure.core/sequential?",
    is_sequential
);
pred1!(
    native_associativep,
    "clojure.core/associative?",
    is_associative
);
pred1!(native_countedp, "clojure.core/counted?", is_counted);
pred1!(native_listp, "clojure.core/list?", is_list);
pred1!(native_vectorp, "clojure.core/vector?", is_vector);
pred1!(native_mapp, "clojure.core/map?", is_map);
pred1!(native_setp, "clojure.core/set?", is_set);
pred1!(native_sortedp, "clojure.core/sorted?", is_sorted);
pred1!(
    native_reversiblep,
    "clojure.core/reversible?",
    is_reversible
);
pred1!(native_fnp, "clojure.core/fn?", is_fn_val);
pred1!(native_ifnp, "clojure.core/ifn?", is_ifn);
pred1!(native_varp, "clojure.core/var?", is_var);
pred1!(native_seqp, "clojure.core/seq?", is_seq_value);
pred1!(native_seqablep, "clojure.core/seqable?", is_seqable);
pred1!(native_recordp, "clojure.core/record?", is_record);
pred1!(native_anyp, "clojure.core/any?", is_any);
pred1!(native_delayp, "clojure.core/delay?", is_delay);
pred1!(native_indexedp, "clojure.core/indexed?", is_indexed);
pred1!(native_map_entryp, "clojure.core/map-entry?", is_map_entry);
pred1!(native_nat_intp, "clojure.core/nat-int?", is_nat_int);
pred1!(native_neg_intp, "clojure.core/neg-int?", is_neg_int);
pred1!(native_pos_intp, "clojure.core/pos-int?", is_pos_int);
pred1!(
    native_special_symbolp,
    "clojure.core/special-symbol?",
    is_special_symbol
);

/// `NaN?` / `infinite?` over the tower (S1): a `Double` tests
/// its payload, the other numbers answer false, nil is the
/// `doubleValue` NPE, anything else CCEs to `Number` (probes
/// `p-nan-*` / `p-inf-*`).
fn double_predicate(v: &Value, test: fn(f64) -> bool) -> Result<bool, NativeError> {
    match v {
        Value::Double(d) => Ok(test(*d)),
        Value::Int(_)
        | Value::BigInt(_)
        | Value::BigInteger(_)
        | Value::Ratio(_)
        | Value::BigDec(_) => Ok(false),
        Value::Nil => Err(npe_double_value()),
        _ => Err(cce_err(v, "java.lang.Number")),
    }
}

fn native_nanp(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/NaN?", args, &[1])?;
    Ok(Value::Bool(double_predicate(&args[0], f64::is_nan)?))
}

fn native_infinitep(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/infinite?", args, &[1])?;
    Ok(Value::Bool(double_predicate(&args[0], f64::is_infinite)?))
}
pred1!(
    native_tagged_literalp,
    "clojure.core/tagged-literal?",
    is_tagged_literal
);
pred1!(
    native_reader_conditionalp,
    "clojure.core/reader-conditional?",
    is_reader_conditional
);

/// `tagged-literal`: 2-arity `[tag form]` (probe `b-ctor`); nil tags
/// coerce to `nil` (probe `b-niltag-vs-nilsym`), other non-symbols
/// fail the `Symbol` cast (probes `b-ctor-strtag`, `b-ctor-kwtag`).
fn native_tagged_literal(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/tagged-literal", args, &[2])?;
    let tag = match &args[0] {
        Value::Symbol(sym) => sym.clone(),
        Value::Nil => Symbol::new(None, "nil"),
        other => return Err(cce_err(other, "clojure.lang.Symbol")),
    };
    Ok(clj_value::tagged_literal(tag, args[1].clone()))
}

/// `reader-conditional`: 2-arity `[form splicing?]` (probe `c-ctor`);
/// the form is unguarded (probes `c-badform*`), splicing takes
/// `Boolean` only (nil is the verbatim NPE, probe
/// `c-badsplice-nil`).
fn native_reader_conditional(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/reader-conditional", args, &[2])?;
    let Value::Bool(splicing) = &args[1] else {
        if matches!(args[1], Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"java.lang.Boolean.booleanValue()\" because \"splicing_QMARK_\" is null"
                    .to_owned(),
            ));
        }
        return Err(cce_err(&args[1], "java.lang.Boolean"));
    };
    Ok(clj_value::reader_conditional(args[0].clone(), *splicing))
}

/// A validator failure as a native error (the validator reports JVM
/// simple class names).
fn tag_native(err: TagError) -> NativeError {
    NativeError::new(throw_class_for(err.class), err.message)
}

/// The `inst` default reader (shared by the `default-data-readers`
/// table and the eval walk): nil is the verbatim NPE (probe
/// `d-inst-nil`), strings validate (probes `d-inst-*`), anything
/// else fails the `CharSequence` cast (probes `d-inst-nonstr`,
/// `d-inst-vec`). Valid payloads answer native `Instant` (#103:
/// `FromStr` reuses the validator, so the error surface is
/// unchanged, then constructs infallibly).
fn default_inst_reader(form: &Value) -> Result<Value, NativeError> {
    if matches!(form, Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.lang.CharSequence.length()\" because \"this.text\" is null"
                .to_owned(),
        ));
    }
    let Value::Str(s) = form else {
        return Err(cce_err(form, "java.lang.CharSequence"));
    };
    Ok(Value::Instant(Instant::from_str(s).map_err(tag_native)?))
}

/// The `uuid` default reader (shared like `inst`): non-strings
/// (nil included) fail as `IllegalArgumentException` (probes
/// `d-uuid-nonstr`, `d-uuid-nil`), strings validate (probes
/// `d-uuid-*`), then construct native `Uuid` (#103). Validation
/// stays first: it keeps the S6 verdict (including the `+`-led
/// group gap — `e-uuid-plus` parses upstream — inherited
/// unchanged per the S9 spec), and the residual `FromStr`
/// failure (S6-valid, `fromString`-invalid `i64` overflow)
/// maps to the NFE `Error at index` text (probes
/// `e-uuid-overflow*`).
fn default_uuid_reader(form: &Value) -> Result<Value, NativeError> {
    let Value::Str(s) = form else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "#uuid data reader expected string".to_owned(),
        ));
    };
    validate_uuid(s).map_err(tag_native)?;
    Ok(Value::Uuid(Uuid::from_str(s).map_err(tag_native)?))
}

/// One default-table entry: a 1-arity native fn value (the arity
/// error names the JVM var — probes `d-direct-arity0`,
/// `d-direct-uuid-arity0`).
fn default_reader_fn(
    eval: &mut Evaluator,
    display: &'static str,
    reader: fn(&Value) -> Result<Value, NativeError>,
) -> Value {
    Value::Extension(Arc::new(NativeFn {
        id: eval.next_id(),
        display: Arc::from(display),
        func: Arc::new(move |_, args| {
            check_arity(display, args, &[1])?;
            reader(&args[0])
        }),
        owned: None,
        meta: None,
        fast: FastArith::None,
    }))
}

/// The `default-data-readers` root (S6): `{inst, uuid}` over the
/// 1-arity default natives (probes `a-defaults*`, `a-ddr-*`).
/// Stashes the pristine pair on the evaluator (#94r): the
/// default-layer snapshot compares the live root against these by
/// reference (stashed at mint time, so no extraction branches).
pub(crate) fn default_data_readers_map(eval: &mut Evaluator) -> Value {
    let inst = default_reader_fn(
        eval,
        "clojure.instant/read-instant-date",
        default_inst_reader,
    );
    let uuid = default_reader_fn(
        eval,
        "clojure.uuid/default-uuid-reader",
        default_uuid_reader,
    );
    eval.canonical_default_readers = Some((inst.clone(), uuid.clone()));
    Value::Map(
        new_map()
            .insert(Value::Symbol(Symbol::new(None, "inst")), inst)
            .insert(Value::Symbol(Symbol::new(None, "uuid")), uuid),
    )
}

/// The table's symbol keys for the read-side consult, plus the
/// borked-table class when the value is neither nil nor a
/// lookupable (nil, vectors, entries, and records consult without
/// failing — probes `e-nonmap-nil`, `e-vectable`).
fn table_snapshot(table: &Value) -> (HashSet<Symbol>, Option<String>) {
    if let Some(inst) = crate::types::record_inst(table) {
        return (symbol_keys(&inst.content), None);
    }
    match table {
        Value::Nil | Value::Vector(_) | Value::MapEntry { .. } => (HashSet::new(), None),
        Value::Map(m) => (symbol_keys(m), None),
        _ => (HashSet::new(), Some(jvm_class_of(table))),
    }
}

/// A map's symbol keys (the read-side membership set ignores
/// non-symbol keys, which never match a tag — probe `e-strkey`).
fn symbol_keys(m: &Map) -> HashSet<Symbol> {
    m.iter()
        .filter_map(|(k, _)| match k {
            Value::Symbol(sym) => Some(sym.clone()),
            _ => None,
        })
        .collect()
}

/// Snapshot the live tag consult into `opts` (S6): defer mode on,
/// the table keys (or the borked class), the default-layer keys
/// (or its borked class, #94), and the default-fn flag. Read
/// paths re-snapshot per form (user code runs between forms, so
/// the consult stays fresh like the JVM's per-tag deref). The
/// autoload scan runs first, so freshly mounted mapping files are
/// already in the table the snapshot reads.
pub(crate) fn snapshot_data_readers(
    eval: &mut Evaluator,
    opts: &mut clj_reader::ReaderOpts,
) -> Result<(), NativeError> {
    ensure_data_readers_autoloaded(eval)?;
    let (keys, borked) = table_snapshot(&deref_core(eval, "*data-readers*"));
    opts.data_readers = true;
    opts.data_reader_keys = keys;
    opts.data_readers_borked = borked;
    snapshot_default_layer(eval, opts);
    opts.default_reader_fn = !matches!(deref_core(eval, "*default-data-reader-fn*"), Value::Nil);
    Ok(())
}

/// Snapshot the default layer (`default-data-readers` root) into
/// `opts` (#94): the symbol keys plus the borked-table class,
/// like the table above. Shared by `snapshot_data_readers` and
/// the mapping-file read (which skips the autoload scan but
/// resolves tags against the same layers).
fn snapshot_default_layer(eval: &Evaluator, opts: &mut clj_reader::ReaderOpts) {
    let root = deref_core(eval, "default-data-readers");
    let (keys, borked) = table_snapshot(&root);
    opts.default_reader_keys = keys;
    opts.default_readers_borked = borked;
    opts.default_reader_canonical = canonical_default_keys(eval, &root);
}

/// Canonical tags whose root value is the pristine boot native
/// (#94r): only these inline-validate string payloads read-side.
/// Reference-compared (`identical?` semantics): re-installing the
/// original value object stays canonical; any other object —
/// including a same-display native — is a replacement and defers
/// for eval-side invocation (oracle `h-replace-inst-bad`).
fn canonical_default_keys(eval: &Evaluator, root: &Value) -> HashSet<Symbol> {
    let (Some((inst0, uuid0)), Value::Map(m)) = (eval.canonical_default_readers.as_ref(), root)
    else {
        return HashSet::new();
    };
    [
        (Symbol::new(None, "inst"), inst0),
        (Symbol::new(None, "uuid"), uuid0),
    ]
    .into_iter()
    .filter(|(sym, pristine)| {
        m.get(&Value::Symbol(sym.clone()))
            .is_some_and(|v| crate::classemu::identical_values(v, pristine))
    })
    .map(|(sym, _)| sym)
    .collect()
}

/// `data_readers.clj` autoload (zero-fence reader): upstream
/// `load-data-readers` scans the classpath roots once at startup;
/// the port has no classpath, so every read entry re-scans when
/// the `*load-path*` value moved since the last scan and merges
/// each new mapping file into the `*data-readers*` root (file
/// order replays `data-reader-urls`: all `.clj` first, then all
/// `.cljc`). Files merge once each; scan errors (bad maps, bad
/// keys, conflicts, unreadable files) fail the read raw, like the
/// entry checks. A useless `*load-path*` value skips the scan —
/// reads never consulted it before, so they must not start
/// failing on it now.
pub(crate) fn ensure_data_readers_autoloaded(eval: &mut Evaluator) -> Result<(), NativeError> {
    let Some(dirs) = autoload_dirs(eval) else {
        return Ok(());
    };
    if eval.data_reader_scan_path.as_ref() == Some(&dirs) {
        return Ok(());
    }
    // Bare evaluators install no vars: nothing to merge into (the
    // mark still advances, so the miss costs one compare per read).
    let Some(id) = core_var(eval, "*data-readers*") else {
        eval.data_reader_scan_path = Some(dirs);
        return Ok(());
    };
    // Optimistic mark BEFORE reading: mapping reads resolve tags,
    // so user fns may read again mid-scan — nested scans fast-path
    // out on this mark instead of recursing. Failures revert it,
    // so the next read retries the same files.
    let prev = eval.data_reader_scan_path.clone();
    eval.data_reader_scan_path = Some(dirs.clone());
    let root = eval.registry.root_cloned(id).unwrap_or(Value::Nil);
    let mut acc = root;
    let mut entries: Vec<(Symbol, Value)> = Vec::new();
    let mut fresh: Vec<String> = Vec::new();
    if let Err(e) = autoload_scan(eval, &dirs, &mut acc, &mut entries, &mut fresh) {
        eval.data_reader_scan_path = prev;
        return Err(e);
    }
    if !fresh.is_empty() {
        // Merge failures revert the mark like scan failures: the
        // files stay unrecorded, so the next read retries them
        // instead of silently skipping them forever.
        if let Err(e) = merge_autoload_entries(eval, id, &entries, fresh) {
            eval.data_reader_scan_path = prev;
            return Err(e);
        }
    }
    Ok(())
}

/// Merge scanned entries onto the LIVE `*data-readers*` root, one
/// `assoc` per entry (a nested scan may have merged sibling files
/// meanwhile — wholesale replace would drop them). Nothing installs
/// until every assoc succeeds, so a failure leaves the root and the
/// recorded-file set untouched for the retry.
fn merge_autoload_entries(
    eval: &mut Evaluator,
    id: u64,
    entries: &[(Symbol, Value)],
    fresh: Vec<String>,
) -> Result<(), NativeError> {
    let assoc_fn = deref_core(eval, "assoc");
    let mut live = eval.registry.root_cloned(id).unwrap_or(Value::Nil);
    for (tag, var) in entries {
        let pair = [live, Value::Symbol(tag.clone()), var.clone()];
        live = eval.call(&assoc_fn, &pair).map_err(signal_to_native)?;
    }
    eval.registry.with_state_mut(id, |st| {
        st.root = Some(live);
    });
    eval.loaded_data_reader_files.extend(fresh);
    Ok(())
}

/// The scannable `*load-path*` dirs, leniently: an unlistable value
/// skips the scan (`None`), and non-string entries are dropped
/// (they can never name a file).
fn autoload_dirs(eval: &Evaluator) -> Option<Vec<String>> {
    seq_vec(&deref_core(eval, "*load-path*")).map(|items| {
        items
            .iter()
            .filter_map(|item| match item {
                Value::Str(s) => Some(s.to_string()),
                _ => None,
            })
            .collect()
    })
}

/// Scan every dir for both mapping names, accumulating new files'
/// entries (conflict-checked against `acc`, the scan-start root
/// plus the earlier files — upstream `load-data-reader-file`
/// checks each file against the same accumulator).
fn autoload_scan(
    eval: &mut Evaluator,
    dirs: &[String],
    acc: &mut Value,
    entries: &mut Vec<(Symbol, Value)>,
    fresh: &mut Vec<String>,
) -> Result<(), NativeError> {
    for name in ["data_readers.clj", "data_readers.cljc"] {
        for dir in dirs {
            // `/` separators on every platform (like `*file*`
            // during loads — and the `:url` ex-data below).
            let path = format!("{dir}/{name}").replace('\\', "/");
            if eval.loaded_data_reader_files.contains(&path) {
                continue;
            }
            // Missing files skip; anything else unreadable fails
            // (upstream `openStream` fails the same way — even a
            // directory wearing the mapping name, which is why the
            // unit test below covers this arm with one).
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(crate::ns::fs_err(&path, &e)),
            };
            load_data_reader_file(eval, &path, &text, name.ends_with("cljc"), acc, entries)?;
            fresh.push(path);
        }
    }
    Ok(())
}

/// Upstream `load-data-reader-file`: read one form (`{:eof nil}`,
/// plus `:read-cond :allow` for `.cljc`), reject non-maps, then
/// intern each value and conflict-check each tag.
fn load_data_reader_file(
    eval: &mut Evaluator,
    path: &str,
    text: &str,
    is_cljc: bool,
    acc: &mut Value,
    entries: &mut Vec<(Symbol, Value)>,
) -> Result<(), NativeError> {
    let mappings = read_data_reader_map(eval, path, text, is_cljc)?;
    let Value::Map(m) = &mappings else {
        return Err(data_reader_ex_info(
            "Not a valid data-reader map",
            new_map().insert(url_key(), Value::Str(Arc::from(path))),
        ));
    };
    let assoc_fn = deref_core(eval, "assoc");
    for (k, v) in m.iter() {
        let Value::Symbol(tag) = k else {
            return Err(data_reader_ex_info(
                "Invalid form in data-reader file",
                new_map()
                    .insert(url_key(), Value::Str(Arc::from(path)))
                    .insert(form_key(), k.clone()),
            ));
        };
        let var = data_reader_var(eval, v)?;
        if let Some(prev) = acc_lookup(acc, tag) {
            if prev != var {
                return Err(data_reader_ex_info(
                    "Conflicting data-reader mapping",
                    new_map()
                        .insert(url_key(), Value::Str(Arc::from(path)))
                        .insert(conflict_key(), k.clone())
                        .insert(mappings_key(), acc.clone()),
                ));
            }
        }
        // `assoc` onto the check basis (borked roots fail exactly
        // like the upstream `(assoc mappings k v-var)` would).
        let pair = [acc.clone(), Value::Symbol(tag.clone()), var.clone()];
        *acc = eval.call(&assoc_fn, &pair).map_err(signal_to_native)?;
        entries.push((tag.clone(), var));
    }
    Ok(())
}

/// Read the mapping file's single form: a full `read` (gates, live
/// hooks, tag resolution) with `*file*` bound to the path, EOF
/// answering nil. Syntax failures wrap in `ReaderException` (the
/// read is line-numbered, like upstream's).
fn read_data_reader_map(
    eval: &mut Evaluator,
    path: &str,
    text: &str,
    is_cljc: bool,
) -> Result<Value, NativeError> {
    let seed = seed_read_gates(eval)?;
    let cur = eval.current_ns_name();
    let mut opts = ReaderOpts {
        current_ns: cur,
        read_cond: if is_cljc {
            ReadCond::Allow
        } else {
            ReadCond::Deny
        },
        ..ReaderOpts::default()
    };
    sync_reader_ns(eval, &mut opts);
    // Defer-mode keys WITHOUT the autoload scan (the scan is the
    // caller — re-entering it here would recurse): the table the
    // file read resolves against is the pre-scan root, exactly
    // like upstream (whose nested reads deref the pre-merge root).
    let (keys, borked) = table_snapshot(&deref_core(eval, "*data-readers*"));
    opts.data_readers = true;
    opts.data_reader_keys = keys;
    opts.data_readers_borked = borked;
    snapshot_default_layer(eval, &mut opts);
    opts.default_reader_fn = !matches!(deref_core(eval, "*default-data-reader-fn*"), Value::Nil);
    opts.read_eval = seed.read_eval;
    opts.suppress_read = seed.suppress_read;
    opts.resolver_bound = seed.resolver_bound;
    // Upstream binds `*file*` to the mapping URL for the read.
    let depth = core_var(eval, "*file*")
        .map(|id| eval.push_binding(core_var_handle(id, "*file*"), Value::Str(Arc::from(path))));
    let mut reader = Reader::new(text, &opts);
    let outcome = {
        let mut hooks = LiveReadHooks::new(eval);
        reader.read_form(false, &mut hooks)
    };
    let fired = reader.take_fired();
    // The `*file*` binding covers tag resolution too (upstream
    // invokes readers during `read`, inside the same binding);
    // the closure lets the pop run on every exit path.
    let out = (|| -> Result<Value, NativeError> {
        drain_fired_tags(eval, fired, opts.suppress_read)?;
        match outcome {
            Ok(Some(v)) => resolve_data_readers(eval, v, opts.suppress_read),
            Ok(None) => Ok(Value::Nil),
            Err(e) => Err(map_reader_error(eval, &e, false)),
        }
    })();
    if let Some(depth) = depth {
        eval.pop_bindings(depth);
    }
    out
}

/// Look one tag up in the conflict basis (`contains?` semantics:
/// only maps hit — nil and anything else miss without throwing).
fn acc_lookup(acc: &Value, tag: &Symbol) -> Option<Value> {
    match acc {
        Value::Map(m) => m.get(&Value::Symbol(tag.clone())).cloned(),
        _ => None,
    }
}

/// Upstream `data-reader-var`: `(intern (create-ns (symbol
/// (namespace v))) (symbol (name v)))`. The `namespace` call runs
/// first, so garbage fails exactly like it (nil NPEs, others CCE
/// to `Named`); unqualified names then fail `(symbol nil)` with
/// IAE. Keywords ride the same `Named` path upstream, so they do
/// here too. Interning creates the namespace and the (possibly
/// unbound) var without loading anything.
fn data_reader_var(eval: &mut Evaluator, v: &Value) -> Result<Value, NativeError> {
    let name = match v {
        Value::Symbol(s) => s.name(),
        Value::Keyword(k) => k.name(),
        // Fails the `namespace` call below; this arm only yields a
        // binding the error path discards.
        _ => "",
    };
    let ns_val = native_namespace(eval, std::slice::from_ref(v))?;
    let Value::Str(ns) = &ns_val else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "no conversion to symbol".to_owned(),
        ));
    };
    let ns: &str = ns;
    eval.registry.ensure_ns(ns);
    let target = Symbol::new(Some(ns), name);
    let (id, _) = intern_routed(eval, ns, None, &target)?;
    Ok(var_value(id, &target))
}

/// An autoload ex-info: `ExceptionInfo` with its data map (the
/// message texts and keys replay upstream verbatim).
fn data_reader_ex_info(message: &str, data: Map) -> NativeError {
    let mut err = NativeError::new(ThrowClass::ExceptionInfo, message.to_owned());
    err.data = Some(Value::Map(data));
    err
}

/// The `:url` ex-data key (unqualified, like upstream).
fn url_key() -> Value {
    Value::Keyword(Keyword::new(None, "url"))
}

/// The `:form` ex-data key (unqualified, like upstream).
fn form_key() -> Value {
    Value::Keyword(Keyword::new(None, "form"))
}

/// The `:conflict` ex-data key (unqualified, like upstream).
fn conflict_key() -> Value {
    Value::Keyword(Keyword::new(None, "conflict"))
}

/// The `:mappings` ex-data key (unqualified, like upstream).
fn mappings_key() -> Value {
    Value::Keyword(Keyword::new(None, "mappings"))
}

/// Whether any tag still needs resolution (the walk's fast path:
/// tag-free forms return untouched, so S6 is a no-op for them).
/// Tags nested in preserved conditionals need nothing (the walk
/// skips those subtrees — probe `g-preserve-usertag`).
fn needs_resolution(v: &Value) -> bool {
    match v {
        Value::TaggedLiteral { .. } => true,
        Value::Symbol(s) => s.meta().is_some_and(map_needs),
        Value::List(l) => l.iter().any(needs_resolution) || l.meta().is_some_and(map_needs),
        Value::Vector(v) => {
            v.iter().any(|v| needs_resolution(&v)) || v.meta().is_some_and(map_needs)
        }
        Value::Map(m) => map_needs(m),
        Value::Set(s) => s.iter().any(needs_resolution) || s.meta().is_some_and(map_needs),
        _ => false,
    }
}

/// Whether a map holds (or is meta-wrapped with) a tag.
fn map_needs(m: &Map) -> bool {
    m.iter()
        .any(|(k, v)| needs_resolution(k) || needs_resolution(v))
        || m.meta().is_some_and(map_needs)
}

/// Resolve deferred tags in one read form (S6): bottom-up in read
/// order, skipping `ReaderConditional` subtrees and never
/// re-descending into fn results (probe `e-no-redescent`).
/// Metadata resolves too (tags inside meta maps invoke — probe
/// `e-meta-inner`). Suppressed reads skip entirely (S7): every tag
/// preserved, so nothing deferred (probes `e-tag-sup`, `p-sup-tag`,
/// `m-load-sup`). The `suppressed` ENTRY seed decides — threaded,
/// never re-deref'd: read-time user code (`#=`, data-reader fns)
/// may mutate the root mid-read, skewing a live lookup.
pub(crate) fn resolve_data_readers(
    eval: &mut Evaluator,
    form: Value,
    suppressed: bool,
) -> Result<Value, NativeError> {
    if suppressed {
        return Ok(form);
    }
    if !needs_resolution(&form) {
        return Ok(form);
    }
    resolve_inner(eval, form)
}

/// One resolved metadata map (`None` stays `None`).
fn resolve_opt_meta(eval: &mut Evaluator, meta: Option<Map>) -> Result<Option<Map>, NativeError> {
    match meta {
        Some(m) => Ok(Some(resolve_map(eval, m)?)),
        None => Ok(None),
    }
}

/// One resolved map: keys and values in insertion order (the JVM
/// builds from resolved entries too), plus the meta.
fn resolve_map(eval: &mut Evaluator, m: Map) -> Result<Map, NativeError> {
    let mut out = new_map();
    for (k, v) in m.iter() {
        let key = resolve_inner(eval, k.clone())?;
        let val = resolve_inner(eval, v.clone())?;
        out = out.insert(key, val);
    }
    Ok(out.with_meta(resolve_opt_meta(eval, m.meta().cloned())?))
}

/// The resolving walk (the fast path already confirmed tags).
fn resolve_inner(eval: &mut Evaluator, form: Value) -> Result<Value, NativeError> {
    match form {
        Value::TaggedLiteral { tag, form, .. } => {
            let inner = resolve_inner(eval, *form)?;
            resolve_tag(eval, &tag, inner)
        }
        Value::Symbol(s) => resolve_symbol(eval, s),
        Value::List(l) => resolve_list(eval, l),
        Value::Vector(v) => resolve_vector(eval, v),
        Value::Map(m) => Ok(Value::Map(resolve_map(eval, m)?)),
        Value::Set(s) => resolve_set(eval, s),
        // Preserved conditionals stay data (probe
        // `g-preserve-usertag`); scalars, entries, and host values
        // never hold unresolved tags (the reader yields none).
        other => Ok(other),
    }
}

/// One resolved symbol (meta only).
fn resolve_symbol(eval: &mut Evaluator, s: Symbol) -> Result<Value, NativeError> {
    let meta = resolve_opt_meta(eval, s.meta().cloned())?;
    Ok(Value::Symbol(s.with_meta(meta)))
}

/// One resolved list (items in order, plus the meta).
fn resolve_list(eval: &mut Evaluator, l: List) -> Result<Value, NativeError> {
    let mut items = Vec::with_capacity(l.len());
    for v in l.iter() {
        items.push(resolve_inner(eval, v.clone())?);
    }
    let meta = resolve_opt_meta(eval, l.meta().cloned())?;
    Ok(Value::List(List::from_vec(items).with_meta(meta)))
}

/// One resolved vector (items in order, plus the meta).
fn resolve_vector(eval: &mut Evaluator, v: Vector) -> Result<Value, NativeError> {
    let mut items = Vec::with_capacity(v.len());
    for item in v.iter() {
        items.push(resolve_inner(eval, item.clone())?);
    }
    let meta = resolve_opt_meta(eval, v.meta().cloned())?;
    Ok(Value::Vector(Vector::from_vec(items).with_meta(meta)))
}

/// One resolved set (members in insertion order, plus the meta).
fn resolve_set(eval: &mut Evaluator, s: Set) -> Result<Value, NativeError> {
    let mut out = new_set();
    for v in s.iter() {
        out = out.insert(resolve_inner(eval, v.clone())?);
    }
    let meta = resolve_opt_meta(eval, s.meta().cloned())?;
    Ok(Value::Set(out.with_meta(meta)))
}

/// The table lookup: `RT.get` semantics (nil entries miss — probe
/// `e-nilentry); anything but nil and the lookupables fails the
/// `ILookup` cast (probe `e-nonmap`).
fn table_lookup(table: &Value, tag: &Symbol) -> Result<Option<Value>, NativeError> {
    if let Some(inst) = crate::types::record_inst(table) {
        return Ok(table_entry(&inst.content, tag));
    }
    match table {
        Value::Nil | Value::Vector(_) | Value::MapEntry { .. } => Ok(None),
        Value::Map(m) => Ok(table_entry(m, tag)),
        _ => Err(cce_err(table, "clojure.lang.ILookup")),
    }
}

/// One table entry (a nil entry misses like a missing one).
fn table_entry(m: &Map, tag: &Symbol) -> Option<Value> {
    m.get(&Value::Symbol(tag.clone()))
        .cloned()
        .filter(|v| !matches!(v, Value::Nil))
}

/// Resolve one tag: the table (fresh deref per tag — user code runs
/// between tags), then the `default-data-readers` root (#94), then
/// the default fn with `[tag form]`, else the `RuntimeException`
/// (probes `e-*`, `f-*`, `h-*`). The throw stays reachable when
/// the table mutates mid-walk (the reader consults the pre-read
/// snapshot).
fn resolve_tag(eval: &mut Evaluator, tag: &Symbol, form: Value) -> Result<Value, NativeError> {
    if let Some(f) = table_lookup(&deref_core(eval, "*data-readers*"), tag)? {
        return eval.call(&f, &[form]).map_err(signal_to_native);
    }
    if let Some(f) = table_lookup(&deref_core(eval, "default-data-readers"), tag)? {
        return eval.call(&f, &[form]).map_err(signal_to_native);
    }
    let default_fn = deref_core(eval, "*default-data-reader-fn*");
    if !matches!(default_fn, Value::Nil) {
        let tagv = Value::Symbol(tag.clone());
        return eval
            .call(&default_fn, &[tagv, form])
            .map_err(signal_to_native);
    }
    Err(NativeError::new(
        ThrowClass::RuntimeException,
        format!("No reader function for tag {tag}"),
    ))
}

/// Drain fired-discards in read order (S6): each entry's form
/// resolves first (nested tags fire through it), then the outer tag
/// invokes for effect (the result drops — `#_` discards). The
/// `suppressed` entry seed threads into the nested resolve (M2).
pub(crate) fn drain_fired_tags(
    eval: &mut Evaluator,
    fired: Vec<clj_reader::FiredTag>,
    suppressed: bool,
) -> Result<(), NativeError> {
    for entry in fired {
        let form = resolve_data_readers(eval, entry.form, suppressed)?;
        resolve_tag(eval, &entry.tag, form)?;
    }
    Ok(())
}

/// Printer options from the bound `*print-*` vars (`readably`
/// overrides the binding when given — `print` binds false, `pr`
/// honors).
/// A `*print-length*`/`*print-level*` binding as a printer limit
/// (fixprobe9): nil, negative, fractional, and NaN are unlimited
/// (`None`) — except a non-positive level, which prints `#`
/// (`Some(0)`, via dec-first); integral doubles act as that int;
/// anything else CCEs to `Number`.
fn print_limit(v: &Value, is_level: bool) -> Result<Option<usize>, NativeError> {
    match v {
        Value::Nil => Ok(None),
        Value::Int(n) => {
            if *n < 0 {
                Ok(if is_level { Some(0) } else { None })
            } else {
                Ok(Some(*n as usize))
            }
        }
        Value::Double(d) => {
            if d.is_nan() || d.fract() != 0.0 {
                Ok(None)
            } else if *d < 0.0 {
                Ok(if is_level { Some(0) } else { None })
            } else {
                Ok(Some(*d as usize))
            }
        }
        v => Err(cce_err(v, "java.lang.Number")),
    }
}

/// Printer options from the bound `*print-*` vars (`readably`
/// overrides the binding when given — `print` passes false, `pr`
/// honors). Public for S7 drivers: the REPL prints results exactly
/// like `prn` (pass `None`).
pub fn print_opts(
    eval: &Evaluator,
    readably: Option<bool>,
) -> Result<clj_value::PrintOpts, NativeError> {
    Ok(clj_value::PrintOpts {
        length: print_limit(&deref_core(eval, "*print-length*"), false)?,
        level: print_limit(&deref_core(eval, "*print-level*"), true)?,
        namespace_maps: deref_core(eval, "*print-namespace-maps*").is_truthy(),
        readably: readably.unwrap_or_else(|| deref_core(eval, "*print-readably*").is_truthy()),
        meta: deref_core(eval, "*print-meta*").is_truthy(),
        dup: deref_core(eval, "*print-dup*").is_truthy(),
    })
}

/// Whether the value is a JVM `LazySeq` (whose `toString` never
/// prints contents): a pending seq reporting the `LazySeq` class
/// (forced or not — memoizing never changes the class), or a
/// realized seq explicitly kinded `LazySeq`. Shared with
/// `print-simple`'s object half (single source).
pub(crate) fn is_lazy_seq_str(v: &Value) -> bool {
    if as_pending(v).is_some() {
        return jvm_class_of(v) == "clojure.lang.LazySeq";
    }
    if let Some(s) = as_realized(v) {
        return s.kind() == clj_value::SeqKind::Lazy;
    }
    false
}

/// `str` of one value (also `%s`): nil vanishes; strings/chars are
/// raw; a `LazySeq` renders `clojure.lang.LazySeq@<hex>` with the
/// Java `List` content hash over the fully forced items (the hash
/// walk realizes, so infinite lazies hang, like the JVM — callers
/// take-bound first); every other seq object and strict collection
/// prints readably without limits (like `RT.printString`); anything
/// else prints plainly.
pub(crate) fn str_of_one(eval: &mut Evaluator, v: &Value) -> Result<String, NativeError> {
    match v {
        Value::Nil => Ok(String::new()),
        Value::Str(s) => Ok(s.to_string()),
        Value::Char(c) => Ok(c.to_string()),
        _ if is_lazy_seq_str(v) => lazy_seq_str(eval, v),
        _ => str_of_realized(eval, v),
    }
}

/// A `LazySeq` renders `clojure.lang.LazySeq@<hex>` with the Java
/// `List` content hash over the fully forced items.
fn lazy_seq_str(eval: &mut Evaluator, v: &Value) -> Result<String, NativeError> {
    let items = collect_all(eval, v.clone()).map_err(signal_to_native)?;
    let mut h: i32 = 1;
    for item in &items {
        let r = realize_deep(eval, item).map_err(signal_to_native)?;
        h = h.wrapping_mul(31).wrapping_add(clj_value::hash_code(&r));
    }
    Ok(format!("clojure.lang.LazySeq@{:x}", h as u32))
}

/// `str` of a realized non-`LazySeq` value: seq objects and strict
/// collections print readably without limits, atoms/delays render
/// their `toString`, anything else prints plainly.
fn str_of_realized(eval: &mut Evaluator, v: &Value) -> Result<String, NativeError> {
    let r = realize_deep(eval, v).map_err(signal_to_native)?;
    Ok(match &r {
        // Port-defined types run a zero-arg `toString` method when
        // defined, else the `Name@hex` fallback (oracle `reify-str`,
        // `j-str-rec`); descriptors render their name.
        Value::HostType(h) => crate::types::host_str(eval, h, r.clone())?,
        // Multimethods render the `toString` fallback (no methods
        // run for them, oracle `d-prefer`).
        Value::MultiFn(m) => clj_value::multifn_str(&m.name),
        // Atoms, delays, futures, promises, agents, volatiles,
        // refs, and executors render their `toString` (`Class@hex`),
        // never the `#object[…]` print form.
        Value::Atom(_)
        | Value::Delay(_)
        | Value::Future(_)
        | Value::Promise(_)
        | Value::Agent(_)
        | Value::Volatile(_)
        // Arrays render `toString` too (`{desc}@{id}` — S15 oracle
        // `d-str-int`); refs likewise (S19); executors likewise
        // (fence 3c).
        | Value::Array(_)
        | Value::Ref(_)
        | Value::Executor(_) => clj_value::java_debug(&r),
        // Classes render `Class.toString` (`class `/`interface `
        // plus the name — shared with the print engine).
        Value::Class(c) => clj_value::class_java_debug(c),
        Value::List(_) | Value::Vector(_) | Value::Map(_) | Value::Set(_) => {
            clj_value::pr_to_string(&r, &clj_value::PrintOpts::pr())
        }
        // S9 scalars answer `toString` (bare UUID/URI,
        // `Date.toString` instants) — `print` stays tagged while
        // `str` goes bare.
        Value::Uuid(_) | Value::Instant(_) | Value::Uri(_) => clj_value::java_debug(&r),
        // Extensions with a `toString` or `str` display (namespaces,
        // `MathContext` — S2 oracle `wp-str`) render it; the rest
        // print readably.
        Value::Extension(e) => match e.java_str().or_else(|| e.print_str()) {
            Some(s) => s,
            None => clj_value::pr_to_string(&r, &clj_value::PrintOpts::pr()),
        },
        _ => pr_to_string(&r, &PrintOpts::print()),
    })
}

/// `str`: concatenates the per-arg rendering (no spaces).
/// `str` concatenated (shared by `str` and the `throw-*!` natives).
fn str_concat(eval: &mut Evaluator, args: &[Value]) -> Result<String, NativeError> {
    let mut out = String::new();
    for arg in args {
        str_push(&mut out, eval, arg)?;
    }
    Ok(out)
}

/// Append `v`'s `str` piece ([`str_of_one`]; chars, strings and nil
/// append in place, with no piece string).
fn str_push(out: &mut String, eval: &mut Evaluator, v: &Value) -> Result<(), NativeError> {
    match v {
        Value::Nil => {}
        Value::Str(s) => out.push_str(s),
        Value::Char(c) => out.push(*c),
        // A long's `toString` is its decimal digits (the printer's
        // rendering, without the print machinery).
        Value::Int(n) => {
            use std::fmt::Write;
            write!(out, "{n}").expect("writing to a String cannot fail");
        }
        _ => out.push_str(&str_of_one(eval, v)?),
    }
    Ok(())
}

/// `(apply str a … coll)` without spreading `coll` into an argument
/// array, when `coll` is already realized (a vector, list, string or
/// nil — no realization effects to order before the pieces): the
/// pieces append into one buffer, so a 600K-char vector costs its
/// text, not 64 bytes per char. `None` leaves other colls to the
/// spread (a lazy seq realizes first, like the JVM's
/// `spreadToArray`).
pub(crate) fn apply_str_direct(
    eval: &mut Evaluator,
    args: &[Value],
) -> Result<Option<Value>, NativeError> {
    let (coll, lead) = args.split_last().expect("apply passes its spread");
    let items: Box<dyn Iterator<Item = Value> + '_> = match coll {
        Value::Nil => Box::new(std::iter::empty()),
        Value::Vector(x) => Box::new(x.iter()),
        Value::List(l) => Box::new(l.iter().cloned()),
        Value::Str(s) => Box::new(s.chars().map(Value::Char)),
        _ => return Ok(None),
    };
    let mut out = String::new();
    for arg in lead {
        str_push(&mut out, eval, arg)?;
    }
    for item in items {
        str_push(&mut out, eval, &item)?;
    }
    Ok(Some(Value::Str(Arc::from(out.as_str()))))
}

pub(crate) fn native_str(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_str_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_str_owned(
    eval: &mut Evaluator,
    args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    let s = str_concat(eval, &args)?;
    Ok(Value::Str(Arc::from(s.as_str())))
}

/// `subs` (char-indexed; the JVM's UTF-16 units agree on the BMP —
/// unpinned astral divergence is documented). Borrows the input
/// and slices by byte window (#227): no whole-string `to_string`
/// copy and no `chars` vec — only the answer allocates.
fn native_subs(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(NativeError::arity("clojure.core/subs", args.len()));
    }
    let s: &str = match &args[0] {
        Value::Str(s) => s,
        Value::Nil => {
            let method = if args.len() == 2 {
                "String.substring(int)"
            } else {
                "String.substring(int, int)"
            };
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                format!("Cannot invoke \"{method}\" because \"s\" is null"),
            ));
        }
        v => return Err(cce_err(v, "java.lang.String")),
    };
    // ASCII short-circuits (chars are bytes — one SIMD scan, then a
    // byte slice); anything else walks `char_indices` windows.
    if s.is_ascii() {
        return subs_ascii(s, args);
    }
    let len = s.chars().count() as i64;
    // The `nth`/`subvec` index path: doubles truncate, nil trips the
    // `charValue` NPE, anything else fails to `Number`.
    let begin = i64::from(nth_index(&args[1])?);
    let end = if args.len() == 3 {
        i64::from(nth_index(&args[2])?)
    } else {
        len
    };
    if begin < 0 || end < begin || end > len {
        return Err(subs_oob(begin, end, len));
    }
    let start_b = char_byte_at(s, begin, len);
    let end_b = char_byte_at(s, end, len);
    Ok(Value::Str(Arc::from(&s[start_b..end_b])))
}

/// The byte offset of char `idx` (`len`-capped callers only: `idx`
/// is in bounds, and `idx == len` is the end of the string).
fn char_byte_at(s: &str, idx: i64, len: i64) -> usize {
    if idx == len {
        return s.len();
    }
    s.char_indices()
        .nth(idx as usize)
        .map(|(b, _)| b)
        .expect("subs index is in bounds")
}

/// The `subs` out-of-bounds error (split for the ASCII/non-ASCII arms).
fn subs_oob(begin: i64, end: i64, len: i64) -> NativeError {
    NativeError::new(
        ThrowClass::StringIndexOutOfBoundsException,
        format!("Range [{begin}, {end}) out of bounds for length {len}"),
    )
}

/// The `subs` ASCII arm (chars are bytes, so `len` is `s.len()`
/// and the window slices directly — same indexes, errors, and
/// answers as the scalar walk).
fn subs_ascii(s: &str, args: &[Value]) -> Result<Value, NativeError> {
    let len = s.len() as i64;
    let begin = i64::from(nth_index(&args[1])?);
    let end = if args.len() == 3 {
        i64::from(nth_index(&args[2])?)
    } else {
        len
    };
    if begin < 0 || end < begin || end > len {
        return Err(subs_oob(begin, end, len));
    }
    Ok(Value::Str(Arc::from(&s[begin as usize..end as usize])))
}

/// `pr-str`/`print-str`: space-joined (nothing for zero args).
/// Like the JVM's `with-out-str` + `pr`/`print`: `*out*` binds to
/// a fresh string writer (so user methods see it) and every arg
/// renders through the `pr-on` driver.
fn pr_print_str(
    eval: &mut Evaluator,
    args: &[Value],
    readably: Option<bool>,
) -> Result<Value, NativeError> {
    // Eager validation first (limits CCE before anything realizes,
    // as before).
    print_opts(eval, readably)?;
    let mut vals = Vec::with_capacity(args.len());
    for arg in args {
        vals.push(realize_deep(eval, arg).map_err(signal_to_native)?);
    }
    let sw = crate::io::string_writer(eval);
    let depth = eval.bindings_depth();
    crate::printsys::bind_core_var(eval, "*out*", sw.clone());
    crate::printsys::maybe_bind_print_readably(eval, readably);
    let result = pr_str_args(eval, &vals);
    eval.pop_bindings(depth);
    result?;
    let contents = crate::io::writer_contents(&sw).unwrap_or_default();
    Ok(Value::Str(Arc::from(contents.as_str())))
}

/// One `pr-on` per arg into the bound string writer (separators
/// between, `*out*` re-read per arg like the `pr` path).
fn pr_str_args(eval: &mut Evaluator, vals: &[Value]) -> Result<(), NativeError> {
    use crate::ns::deref_core;
    for (i, v) in vals.iter().enumerate() {
        let out = deref_core(eval, "*out*");
        crate::io::check_writer(&out)?;
        if i > 0 {
            crate::io::write_to_writer(&out, " ")?;
        }
        crate::printsys::pr_on(eval, v, &out).map_err(signal_to_native)?;
    }
    Ok(())
}

fn native_pr_str(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // `pr` honors the `*print-readably*` binding (root true).
    pr_print_str(eval, args, None)
}

fn native_print_str(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // `print` binds `*print-readably*` false (probed).
    pr_print_str(eval, args, Some(false))
}

/// Split `ns/name` on the first `/` (empty sides stay empty, like
/// the JVM's `split`).
fn split_ns_name(s: &str) -> (Option<&str>, &str) {
    match s.find('/') {
        Some(i) => (Some(&s[..i]), &s[i + 1..]),
        None => (None, s),
    }
}

fn native_keyword(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() > 2 {
        return Err(NativeError::arity("clojure.core/keyword", args.len()));
    }
    if args.len() == 2 {
        let ns = match &args[0] {
            Value::Str(s) => s.to_string(),
            v => return Err(cce_err(v, "java.lang.String")),
        };
        let name = match &args[1] {
            Value::Str(s) => s.to_string(),
            v => return Err(cce_err(v, "java.lang.String")),
        };
        return Ok(Value::Keyword(Keyword::new(
            Some(ns.as_str()),
            name.as_str(),
        )));
    }
    check_arity("clojure.core/keyword", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::Keyword(k) => Ok(Value::Keyword(k.clone())),
        Value::Str(s) => {
            let (ns, name) = split_ns_name(s);
            Ok(Value::Keyword(Keyword::new(ns, name)))
        }
        Value::Symbol(s) => Ok(Value::Keyword(Keyword::new(s.namespace(), s.name()))),
        // Anything else is nil (`(keyword 5)` → nil, fixprobe8
        // `:kw-num` — only the 2-arg arity casts to `String`).
        _ => Ok(Value::Nil),
    }
}

fn native_symbol(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() > 2 {
        return Err(NativeError::arity("clojure.core/symbol", args.len()));
    }
    if args.len() == 2 {
        let ns = match &args[0] {
            Value::Str(s) => s.to_string(),
            v => return Err(cce_err(v, "java.lang.String")),
        };
        let name = match &args[1] {
            Value::Str(s) => s.to_string(),
            v => return Err(cce_err(v, "java.lang.String")),
        };
        return Ok(Value::Symbol(Symbol::new(Some(ns.as_str()), name.as_str())));
    }
    check_arity("clojure.core/symbol", args, &[1])?;
    match &args[0] {
        Value::Str(s) => {
            let (ns, name) = split_ns_name(s);
            Ok(Value::Symbol(Symbol::new(ns, name)))
        }
        Value::Keyword(k) => Ok(Value::Symbol(Symbol::new(k.namespace(), k.name()))),
        Value::Symbol(s) => Ok(Value::Symbol(s.clone())),
        _ => Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "no conversion to symbol".to_owned(),
        )),
    }
}

/// `find-keyword` (S1): the port never interns keywords (value
/// equality, `keyword.rs`), so a well-shaped lookup builds the
/// keyword — only the JVM's nil shapes stay nil (nil and
/// anything-else 1-arity; probes `p-findkw-*`). The 2-arity
/// casts to `String` like `keyword`, with a nil namespace
/// meaning no namespace (probe `p-findkw-2nilns`) and a nil
/// name the `hashCode` NPE (probe `p-findkw-2nilname`).
fn native_find_keyword(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() > 2 {
        return Err(NativeError::arity("clojure.core/find-keyword", args.len()));
    }
    if args.len() == 2 {
        let ns = match &args[0] {
            Value::Nil => None,
            Value::Str(s) => Some(s.to_string()),
            v => return Err(cce_err(v, "java.lang.String")),
        };
        let name = match &args[1] {
            Value::Str(s) => s.to_string(),
            Value::Nil => {
                return Err(NativeError::new(
                    ThrowClass::NullPointerException,
                    "Cannot invoke \"String.hashCode()\" because \"this.name\" is null".to_owned(),
                ));
            }
            v => return Err(cce_err(v, "java.lang.String")),
        };
        return Ok(Value::Keyword(Keyword::new(ns.as_deref(), name.as_str())));
    }
    check_arity("clojure.core/find-keyword", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::Keyword(k) => Ok(Value::Keyword(k.clone())),
        Value::Symbol(s) => Ok(Value::Keyword(Keyword::new(s.namespace(), s.name()))),
        Value::Str(s) => {
            let (ns, name) = split_ns_name(s);
            Ok(Value::Keyword(Keyword::new(ns, name)))
        }
        // Anything else is nil (like `keyword`'s 1-arity).
        _ => Ok(Value::Nil),
    }
}

/// The `Named` name (`name` works on strings too).
fn native_name(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/name", args, &[1])?;
    match &args[0] {
        Value::Str(s) => Ok(Value::Str(Arc::clone(s))),
        Value::Keyword(k) => Ok(Value::Str(Arc::from(k.name()))),
        Value::Symbol(s) => Ok(Value::Str(Arc::from(s.name()))),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.Named.getName()\" because \"x\" is null".to_owned(),
        )),
        v => Err(cce_err(v, "clojure.lang.Named")),
    }
}

/// The `Named` namespace (nil for simple names; strings fail —
/// only `name` accepts them).
fn native_namespace(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/namespace", args, &[1])?;
    match &args[0] {
        Value::Keyword(k) => Ok(k
            .namespace()
            .map_or(Value::Nil, |ns| Value::Str(Arc::from(ns)))),
        Value::Symbol(s) => Ok(s
            .namespace()
            .map_or(Value::Nil, |ns| Value::Str(Arc::from(ns)))),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.Named.getNamespace()\" because \"x\" is null".to_owned(),
        )),
        v => Err(cce_err(v, "clojure.lang.Named")),
    }
}

/// The `charValue` NPE (`int` of nil — `int` falls back to the
/// `Character` cast, fixprobe10 `:int-nil`).
pub(crate) fn npe_char_value() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Character.charValue()\" because \"x\" is null".to_owned(),
    )
}

/// A `Ratio` truncated toward zero (`int`/`long` narrow from
/// there, like the JVM's `intValue`/`longValue`).
fn trunc_ratio(r: &num_rational::BigRational) -> num_bigint::BigInt {
    r.numer() / r.denom()
}

/// A truncated tower integer as a `long`: out-of-`i64` fails
/// `Value out of range for long` with the exact digits (oracle
/// `l1`/`l2` — no wrap, unlike `longValue`).
fn bigint_long(n: &num_bigint::BigInt) -> Result<i64, NativeError> {
    n.to_i64()
        .ok_or_else(|| out_of_range("long", Num::BigInt(n.clone())))
}

/// The `doubleValue` NPE (`long`/`double`/`float`/`short`/`byte`
/// of nil).
pub(crate) fn npe_double_value() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Number.doubleValue()\" because \"x\" is null".to_owned(),
    )
}

/// The `longValue` NPE (`char` of nil).
pub(crate) fn npe_long_value() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Number.longValue()\" because \"x\" is null".to_owned(),
    )
}

/// `IAE | Value out of range for {kind}: {x}` — doubles render
/// via `Double.toString`, ints raw (fixprobe10 `:int-huge`),
/// tower values exactly.
pub(crate) fn out_of_range(kind: &str, n: Num) -> NativeError {
    let x = match n {
        Num::Int(i) => i.to_string(),
        Num::Double(d) => clj_value::java_double_string(d),
        Num::BigInt(n) => n.to_string(),
        Num::Ratio(r) => format!("{r}"),
        Num::BigDec(d) => d.to_bigdec_string(),
    };
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("Value out of range for {kind}: {x}"),
    )
}

/// The `(int x)` cast as an `i64`: longs narrow exactly,
/// doubles range-check before truncating, tower narrows through
/// the `long` stop, characters convert, nil trips the
/// `charValue` NPE, anything else fails to `Character`. Shared
/// with the S12 `seque` capacities and `chunk-buffer` (which pin
/// the same `Character` cast — oracles `sq-cap-str`,
/// `fixprobe10`).
pub(crate) fn int_cast_value(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(i) => int_exact_long(*i).map(i64::from),
        Value::Double(d) => double_int(*d),
        Value::Char(c) => Ok(*c as i64),
        // Tower integers take the `long` range check (exact
        // digits) then `toIntExact` (oracles `conv-i1`,
        // `conv-hugeint`); decimals narrow through `double`
        // first (oracle `conv-hugedecint`: `1.0E30`).
        Value::BigInt(n) | Value::BigInteger(n) => tower_intcast(n).map(i64::from),
        Value::Ratio(r) => tower_intcast(&trunc_ratio(r)).map(i64::from),
        Value::BigDec(d) => bigdec_intcast(d).map(i64::from),
        Value::Nil => Err(npe_char_value()),
        v => Err(cce_err(v, "java.lang.Character")),
    }
}

/// An `RT.intCast`-shaped `Number` narrowing to `i32` (S12
/// `mix-collection-hash`/`hash-combine` args): numbers narrow
/// checked (the `long` range check, then `toIntExact`), tower
/// truncates first, nil trips the `doubleValue` NPE, anything
/// else fails to `Number` (oracles `mh-mix-str`,
/// `mh-comb-seed-str`).
pub(crate) fn number_int_cast(v: &Value) -> Result<i32, NativeError> {
    match v {
        Value::Int(i) => int_exact_long(*i),
        Value::Double(d) => int_cast_double(*d),
        Value::BigInt(n) => tower_intcast(n),
        Value::Ratio(r) => tower_intcast(&trunc_ratio(r)),
        Value::BigDec(d) => bigdec_intcast(d),
        Value::Nil => Err(npe_double_value()),
        v => Err(cce_err(v, "java.lang.Number")),
    }
}

pub(crate) fn native_int(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/int", args, &[1])?;
    // `int` keeps the value a `Long` (no 32-bit ints exist), but
    // checks the int range exactly: longs past it are `AE |
    // integer overflow`, doubles past it `IAE | Value out of
    // range for int` (checked BEFORE truncating), `NaN` is 0,
    // and characters convert (fixprobe10/11).
    Ok(Value::Int(int_cast_value(&args[0])?))
}

/// An `int` from a `double`: `NaN` is 0, past-`i32` fails `Value
/// out of range for int` (checked BEFORE truncating), else
/// truncation toward zero.
fn double_int(d: f64) -> Result<i64, NativeError> {
    if d.is_nan() {
        Ok(0)
    } else if d > f64::from(i32::MAX) || d < f64::from(i32::MIN) {
        Err(out_of_range("int", Num::Double(d)))
    } else {
        Ok(d.trunc() as i64)
    }
}

pub(crate) fn native_long(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/long", args, &[1])?;
    Ok(Value::Int(match &args[0] {
        Value::Int(i) => *i,
        Value::Double(d) => double_long(*d)?,
        Value::Char(c) => *c as i64,
        Value::BigInt(n) | Value::BigInteger(n) => bigint_long(n)?,
        Value::Ratio(r) => bigint_long(&trunc_ratio(r))?,
        // Decimals narrow through `double` (oracle `conv-declong`:
        // `(long 1E+30M)` fails `1.0E30`).
        Value::BigDec(d) => double_long(d.to_f64())?,
        Value::Nil => return Err(npe_double_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    }))
}

pub(crate) fn native_double(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/double", args, &[1])?;
    Ok(Value::Double(match &args[0] {
        Value::Int(i) => *i as f64,
        Value::Double(d) => *d,
        Value::BigInt(n) | Value::BigInteger(n) => n.to_f64().unwrap_or(f64::NAN),
        Value::Ratio(r) => r.to_f64().unwrap_or(f64::NAN),
        Value::BigDec(d) => d.to_f64(),
        Value::Nil => return Err(npe_double_value()),
        // Characters rejected (fixprobe11 `:dbl-ch`).
        v => return Err(cce_err(v, "java.lang.Number")),
    }))
}

/// `bigint`: `BigInt` passes through; `Int` widens; `Double`
/// and `BigDec` truncate toward zero through the canonical
/// decimal (`NaN`/`Inf` fail like `valueOf`); `Ratio` truncates;
/// strings parse like `BigInteger(String)`; nil trips the
/// `String.length` NPE; anything else is the `No matching ctor`
/// `IllegalArgumentException` (all oracle-verbatim).
fn native_bigint(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/bigint", args, &[1])?;
    match &args[0] {
        // `BigInt` passes through (the same object — upstream
        // answers `x`); `BigInteger` coerces to a fresh `BigInt`
        // (S2 oracle `r8-bigrd`).
        Value::BigInt(n) => Ok(Value::BigInt(n.clone())),
        Value::BigInteger(n) => Ok(Value::BigInt(Arc::new(n.as_ref().clone()))),
        Value::Int(i) => Ok(Value::BigInt(Arc::new(num_bigint::BigInt::from(*i)))),
        Value::Double(d) => {
            let dec = clj_value::BigDec::value_of_f64(*d).map_err(number_format)?;
            Ok(Value::BigInt(Arc::new(dec.to_bigint_trunc())))
        }
        Value::BigDec(d) => Ok(Value::BigInt(Arc::new(d.to_bigint_trunc()))),
        Value::Ratio(r) => Ok(Value::BigInt(Arc::new(r.numer() / r.denom()))),
        Value::Str(s) => clj_value::bigint_parse(s)
            .map(|n| Value::BigInt(Arc::new(n)))
            .map_err(number_format),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"String.length()\" because \"val\" is null".to_owned(),
        )),
        _ => Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "No matching ctor found for class java.math.BigInteger".to_owned(),
        )),
    }
}

/// `abs`: exact over every numeric kind (`Long/MIN_VALUE` is a
/// fixed point, doubles take `abs` — so `-0.0` → `+0.0` and
/// `NaN`/`Inf` ride through, `BigInteger` absorbs into `BigInt`);
/// `BigDec` rounds to the bound context (S2 oracle `wp-abs`);
/// nil is the numbers NPE, anything else the `Number` CCE (all
/// oracle-verbatim, `abs-*`/`r9-absbi`).
fn native_abs(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/abs", args, &[1])?;
    Ok(match as_num(&args[0])? {
        Num::Int(i) => Value::Int(if i == i64::MIN { i } else { i.abs() }),
        Num::Double(d) => Value::Double(d.abs()),
        Num::BigInt(n) => Value::BigInt(Arc::new(n.abs())),
        Num::Ratio(r) => num_value(num_ratio(r.abs())),
        // The JDK `abs` answers `this` when non-negative
        // (identity preserved); negatives negate (a computed
        // result — canonical when scale-0 `0..=10`).
        Num::BigDec(d) => match math_context_of(eval)? {
            None if d.sign() != std::cmp::Ordering::Less => args[0].clone(),
            None => bigdec_result(d.abs()),
            Some(mc) => bigdec_result(d.abs_mc(&mc).map_err(arithmetic)?),
        },
    })
}

/// `biginteger`: the upstream `cond` chain — `BigDec`/`Double`
/// truncate toward zero through the canonical decimal
/// (`NaN`/`Inf` fail like `valueOf`), `Ratio` truncates,
/// `Int`/`BigInt` widen, strings parse like
/// `BigInteger(String)`, nil trips the `String.length` NPE,
/// anything else is the `No matching ctor`
/// `IllegalArgumentException` (all oracle-verbatim, `bi-*`).
/// Results print bare (`5`, never `5N`).
fn native_biginteger(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/biginteger", args, &[1])?;
    match &args[0] {
        // `BigInteger` passes through (the same object —
        // upstream answers `x`); `BigInt` coerces to a fresh
        // `BigInteger`.
        Value::BigInteger(n) => Ok(Value::BigInteger(n.clone())),
        Value::BigInt(n) => Ok(Value::BigInteger(Arc::new(n.as_ref().clone()))),
        Value::Int(i) => Ok(Value::BigInteger(Arc::new(num_bigint::BigInt::from(*i)))),
        Value::Double(d) => {
            let dec = clj_value::BigDec::value_of_f64(*d).map_err(number_format)?;
            Ok(Value::BigInteger(Arc::new(dec.to_bigint_trunc())))
        }
        Value::BigDec(d) => Ok(Value::BigInteger(Arc::new(d.to_bigint_trunc()))),
        Value::Ratio(r) => Ok(Value::BigInteger(Arc::new(r.numer() / r.denom()))),
        Value::Str(s) => clj_value::bigint_parse(s)
            .map(|n| Value::BigInteger(Arc::new(n)))
            .map_err(number_format),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"String.length()\" because \"val\" is null".to_owned(),
        )),
        _ => Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "No matching ctor found for class java.math.BigInteger".to_owned(),
        )),
    }
}

/// `numerator` / `denominator`: the reduced `Ratio` parts as
/// `BigInteger` (the denominator is always positive);
/// non-`Ratio` args fail the `clojure.lang.Ratio` checkcast,
/// nil the field-read NPE (S2 oracles `nd-*`/`r2-den-nil`/
/// `r9-numbi`).
fn ratio_part(display: &str, args: &[Value], numer: bool) -> Result<Value, NativeError> {
    check_arity(display, args, &[1])?;
    let field = if numer { "numerator" } else { "denominator" };
    match &args[0] {
        Value::Ratio(r) => Ok(Value::BigInteger(Arc::new(if numer {
            r.numer().clone()
        } else {
            r.denom().clone()
        }))),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            format!("Cannot read field \"{field}\" because \"r\" is null"),
        )),
        v => Err(cce_err(v, "clojure.lang.Ratio")),
    }
}

fn native_numerator(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ratio_part("clojure.core/numerator", args, true)
}

fn native_denominator(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    ratio_part("clojure.core/denominator", args, false)
}

/// Port-only `math-context` ctor behind `with-precision` (S2,
/// installed but never referred): `(math-context precision)` /
/// `(math-context precision mode)`. Precision mirrors
/// `(MathContext. precision)`
/// reflection: `Int` range-checks like `intCast`
/// (past-`i32` is `integer overflow`, negative is `Digits <
/// 0` — oracles `r2-hugeprec`/`wp-negprec`), `String` parses
/// the strict `MathContext(String)` grammar (oracles
/// `r2-str*`/`r3-*`/`r4-*`), nil is the `null String` NPE
/// (oracle `wp-nilprec`), anything else is `No matching ctor`
/// (oracles `wp-fracprec`/`wp-bigprec`/`r4-boolprec`). A string
/// precision with an explicit mode fails the `Number` CCE
/// (oracles `r7-strmode`/`r8-strmodedirect`). Modes name a
/// `RoundingMode` exactly (symbol/keyword/string — a deliberate
/// port superset of upstream's static-field symbols); unknown
/// modes fail the `valueOf` text (upstream fails at compile
/// with a path-laden `CompilerException` — oracles
/// `wp-badmode`/`wp-lowmode`/`r3-nomode`).
fn native_math_context(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/math-context", args, &[1, 2])?;
    let mode = args.get(1);
    let ctx = match &args[0] {
        Value::Int(i) => {
            let p = int_exact_long(*i)?;
            if p < 0 {
                return Err(NativeError::new(
                    ThrowClass::IllegalArgumentException,
                    clj_value::DIGITS_LT_0.to_owned(),
                ));
            }
            clj_value::MathContext::new(p as u32, ctx_rounding(mode)?)
        }
        Value::Str(s) => {
            // No `(String, RoundingMode)` ctor exists: reflection
            // lands on `(int, RoundingMode)` and the unbox CCEs
            // to `Number` (oracles `r7-strmode`/`r8-strmodedirect`).
            if mode.is_some() {
                return Err(cce_err(&args[0], "java.lang.Number"));
            }
            clj_value::MathContext::parse(s)
                .map_err(|m| NativeError::new(ThrowClass::IllegalArgumentException, m))?
        }
        Value::Nil => {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "null String".to_owned(),
            ));
        }
        _ => {
            return Err(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "No matching ctor found for class java.math.MathContext".to_owned(),
            ));
        }
    };
    Ok(Value::Extension(Arc::new(MathContextVal {
        id: eval.next_id(),
        ctx,
    })))
}

/// The `:rounding` mode (default `HALF_UP`): symbols, keywords,
/// and strings match by exact name; anything else fails the
/// `RoundingMode.valueOf` text.
fn ctx_rounding(mode: Option<&Value>) -> Result<clj_value::RoundingMode, NativeError> {
    let Some(m) = mode else {
        return Ok(clj_value::RoundingMode::HalfUp);
    };
    let name = match m {
        Value::Symbol(s) => s.name().to_owned(),
        Value::Keyword(k) => k.name().to_owned(),
        Value::Str(s) => s.to_string(),
        _ => return Err(rounding_value_of(&crate::ns::str_of(m))),
    };
    clj_value::RoundingMode::parse(&name).ok_or_else(|| rounding_value_of(&name))
}

/// `RoundingMode.valueOf` failure for a bad mode name.
fn rounding_value_of(name: &str) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("No enum constant java.math.RoundingMode.{name}"),
    )
}

/// `bigdec`: `BigDec` passes through; `Int`/`Double` go through
/// `valueOf` (so `(bigdec 5)` is `5.0M`); `BigInt` widens;
/// `Ratio` must terminate exactly; strings parse like
/// `BigDecimal(String)`; nil trips the `toCharArray` NPE;
/// anything else is the `No matching ctor`
/// `IllegalArgumentException` (all oracle-verbatim).
fn native_bigdec(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/bigdec", args, &[1])?;
    bigdec_convert(eval, &args[0])
}

/// Numeric `bigdec` inputs, split from `native_bigdec` for the
/// complexity budget.
fn bigdec_convert(eval: &mut Evaluator, v: &Value) -> Result<Value, NativeError> {
    match v {
        // `BigDec` passes through (the same object — upstream
        // answers `x`); every other arm boxes fresh (the JDK's
        // `new BigDecimal` ctors — never `valueOf`).
        Value::BigDec(d) => Ok(Value::BigDec(d.clone())),
        Value::Int(i) => Ok(Value::BigDec(Arc::new(
            clj_value::BigDec::value_of_f64(*i as f64).map_err(number_format)?,
        ))),
        Value::Double(d) => Ok(Value::BigDec(Arc::new(
            clj_value::BigDec::value_of_f64(*d).map_err(number_format)?,
        ))),
        Value::BigInt(n) | Value::BigInteger(n) => Ok(Value::BigDec(Arc::new(
            clj_value::BigDec::from_bigint(n.as_ref().clone()),
        ))),
        other => bigdec_convert_rest(eval, other),
    }
}

/// Non-numeric `bigdec` inputs (ratio/string/nil/other).
fn bigdec_convert_rest(eval: &mut Evaluator, v: &Value) -> Result<Value, NativeError> {
    match v {
        // Upstream `(bigdec ratio)` divides through `/`, so a
        // bound context rounds it (S2 oracle `wp-bigdec`).
        Value::Ratio(r) => match math_context_of(eval)? {
            None => clj_value::BigDec::from_rational_exact(r)
                .map(|d| Value::BigDec(Arc::new(d)))
                .map_err(arithmetic),
            Some(mc) => clj_value::BigDec::from_ratio_mc(r, &mc)
                .map(|d| Value::BigDec(Arc::new(d)))
                .map_err(arithmetic),
        },
        Value::Str(s) => clj_value::BigDec::parse(s)
            .map(|d| Value::BigDec(Arc::new(d)))
            .map_err(number_format),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"String.toCharArray()\" because \"val\" is null".to_owned(),
        )),
        _ => Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "No matching ctor found for class java.math.BigDecimal".to_owned(),
        )),
    }
}

/// `num`: numbers (tower included) pass through, nil is nil,
/// anything else fails the `Number` cast (oracle `num-str`).
fn native_num(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/num", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::Int(_)
        | Value::Double(_)
        | Value::BigInt(_)
        | Value::BigInteger(_)
        | Value::BigDec(_)
        | Value::Ratio(_) => Ok(args[0].clone()),
        other => Err(cce_err(other, "java.lang.Number")),
    }
}

/// `rationalize`: integers pass through, nil is nil, doubles go
/// through `BigDecimal.valueOf` first (`(rationalize 0.1)` is
/// `1/10`, `(rationalize 1e100)` is exact `10N^100`;
/// `NaN`/`Inf` fail), integer-valued decimals fold to `BigInt`
/// (`(rationalize 5.0M)` → `5N`), anything else fails the
/// `Number` cast (all oracle-verbatim).
fn native_rationalize(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/rationalize", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::Int(_) | Value::BigInt(_) | Value::BigInteger(_) | Value::Ratio(_) => {
            Ok(args[0].clone())
        }
        Value::Double(d) => {
            let dec = clj_value::BigDec::value_of_f64(*d).map_err(number_format)?;
            let r = dec.to_rational();
            if r.denom() == &num_bigint::BigInt::from(1) {
                Ok(Value::BigInt(Arc::new(r.numer().clone())))
            } else {
                Ok(Value::Ratio(Arc::new(r)))
            }
        }
        Value::BigDec(d) => {
            let r = d.to_rational();
            if r.denom() == &num_bigint::BigInt::from(1) {
                Ok(Value::BigInt(Arc::new(r.numer().clone())))
            } else {
                Ok(Value::Ratio(Arc::new(r)))
            }
        }
        other => Err(cce_err(other, "java.lang.Number")),
    }
}

/// A `NumberFormatException` for a message.
pub(crate) fn number_format(message: impl Into<String>) -> NativeError {
    NativeError::new(ThrowClass::NumberFormatException, message.into())
}

/// An `ArithmeticException` for a message.
pub(crate) fn arithmetic(message: impl Into<String>) -> NativeError {
    NativeError::new(ThrowClass::ArithmeticException, message.into())
}

pub(crate) fn native_float(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/float", args, &[1])?;
    // No `Float` exists: the float range is checked, then the
    // EXACT double is returned (no `f32` rounding — printing and
    // simple values match; the class/precision gap is documented).
    Ok(Value::Double(match &args[0] {
        Value::Int(i) => *i as f64,
        // `NaN` falls through (all comparisons false).
        Value::Double(d) => float_checked(*d, Num::Double(*d))?,
        // Tower narrows like `double`, then takes the same
        // range check (the class/precision gap is documented).
        Value::BigInt(n) | Value::BigInteger(n) => float_checked(
            n.to_f64().unwrap_or(f64::NAN),
            Num::BigInt(n.as_ref().clone()),
        )?,
        Value::Ratio(r) => float_checked(
            r.to_f64().unwrap_or(f64::NAN),
            Num::Ratio(r.as_ref().clone()),
        )?,
        Value::BigDec(d) => float_checked(d.to_f64(), Num::BigDec(d.as_ref().clone()))?,
        Value::Nil => return Err(npe_double_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    }))
}

/// A `float` from a narrowed `f64` with its error rendering: past
/// the `f32` range fails `Value out of range for float`, else the
/// exact `double` (no `f32` rounding).
fn float_checked(f: f64, err: Num) -> Result<f64, NativeError> {
    if f > f64::from(f32::MAX) || f < -f64::from(f32::MAX) {
        Err(out_of_range("float", err))
    } else {
        Ok(f)
    }
}

pub(crate) fn native_short(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/short", args, &[1])?;
    Ok(Value::Int(match &args[0] {
        Value::Char(c) => *c as i64,
        Value::Nil => return Err(npe_double_value()),
        // Numbers narrow through `narrow_num` (doubles check
        // directly, tower through the `long` stop — oracles
        // `tower-r2e`, `tower-r2f`).
        v => narrow_num(&as_num(v)?, -32768, 32767, "short")?,
    }))
}

pub(crate) fn native_byte(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/byte", args, &[1])?;
    Ok(Value::Int(match &args[0] {
        Value::Char(c) => *c as i64,
        Value::Nil => return Err(npe_double_value()),
        // Numbers narrow through `narrow_num` (oracles `tower-r2e`,
        // `tower-r2f`).
        v => narrow_num(&as_num(v)?, -128, 127, "byte")?,
    }))
}

pub(crate) fn native_char(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/char", args, &[1])?;
    Ok(Value::Char(match &args[0] {
        Value::Char(c) => *c,
        Value::Nil => return Err(npe_long_value()),
        // Numbers narrow through `char_of_num` (oracles
        // `tower-r2f`, `tower-r2g`).
        v => char_of_num(&as_num(v)?)?,
    }))
}

/// Upstream `parsing-err` (S4a): `Expected string, got nil` for
/// nil, else the value's JVM class name (oracle `pb-nil`…).
fn parsing_err(v: &Value) -> NativeError {
    let got = if matches!(v, Value::Nil) {
        "nil".to_owned()
    } else {
        jvm_class_of(v)
    };
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("Expected string, got {got}"),
    )
}

/// Every Unicode decimal-digit block start: each Nd run is exactly
/// 10 code points valued 0–9, so the value is `cp - start`
/// (`Long.parseLong` accepts them all — oracle `pl-arabic`,
/// `pl-fullwidth`; enumerated from the Unicode database).
const ND_STARTS: &[u32] = &[
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x16A60, 0x16B50, 0x1D7CE,
    0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E950, 0x1FBF0,
];

/// `Character.digit(c, 10)`: the decimal value of any Nd char.
fn decimal_value(c: char) -> Option<u32> {
    let cp = c as u32;
    let i = ND_STARTS.partition_point(|&s| s <= cp).checked_sub(1)?;
    let d = cp.checked_sub(ND_STARTS[i])?;
    (d < 10).then_some(d)
}

/// `Long.parseLong`: one ASCII sign, 1+ decimal digits (any
/// script), overflow → `None` (the caller answers nil).
pub(crate) fn parse_long_str(s: &str) -> Option<i64> {
    let (neg, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    if digits.is_empty() {
        return None;
    }
    let mut acc: u64 = 0;
    for c in digits.chars() {
        acc = acc
            .checked_mul(10)?
            .checked_add(u64::from(decimal_value(c)?))?;
    }
    if neg {
        (acc <= i64::MAX as u64 + 1).then_some((acc as i64).wrapping_neg())
    } else {
        (acc <= i64::MAX as u64).then_some(acc as i64)
    }
}

/// A signed `NaN`/`Infinity` spelling (exact case, like the JVM).
fn signed_special(t: &str) -> Option<f64> {
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    match body {
        "NaN" => Some(f64::NAN),
        "Infinity" if neg => Some(f64::NEG_INFINITY),
        "Infinity" => Some(f64::INFINITY),
        _ => None,
    }
}

/// One trailing `d`/`f` suffix (never on an empty core).
fn strip_float_suffix(t: &str) -> &str {
    match t.strip_suffix(['d', 'D', 'f', 'F']) {
        Some(core) if !core.is_empty() => core,
        _ => t,
    }
}

/// A `p`-exponent: ASCII sign plus 1+ digits, clamped past the
/// f64 range (huge exponents are ±Inf/0 either way).
fn parse_hex_exp(exp: &str) -> Option<i32> {
    let digits = exp.strip_prefix(['+', '-']).unwrap_or(exp);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut acc: i32 = 0;
    for b in digits.bytes() {
        acc = acc.saturating_mul(10).saturating_add(i32::from(b - b'0'));
        if acc > 10_000 {
            break;
        }
    }
    Some(if exp.starts_with('-') { -acc } else { acc })
}

/// A hex significand digit (ASCII only, like `FloatingDecimal`).
fn hex_val(c: char) -> Option<u32> {
    c.to_digit(16)
}

/// A `0x` float: hex significand, mandatory `p` exponent
/// (Horner accumulation — exact through 13 significand digits).
fn parse_hex_float(neg: bool, h: &str) -> Option<f64> {
    let p = h.find(['p', 'P'])?;
    let (sig, exp) = (&h[..p], &h[p + 1..]);
    let exp = parse_hex_exp(exp)?;
    let (int_part, frac_part) = sig.split_once('.').unwrap_or((sig, ""));
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    let mut int_val: f64 = 0.0;
    for c in int_part.chars() {
        int_val = int_val.mul_add(16.0, f64::from(hex_val(c)?));
    }
    let mut frac_val: f64 = 0.0;
    for c in frac_part.chars().rev() {
        frac_val = (frac_val + f64::from(hex_val(c)?)) / 16.0;
    }
    let v = (int_val + frac_val) * 2f64.powi(exp);
    Some(if neg { -v } else { v })
}

/// `Double.valueOf`: Java-`trim`, exact `NaN`/`Infinity`, decimal
/// via the host parser (overflow → ±Inf like the JVM), hex
/// floats manual, one `d`/`f` suffix — never on a bare
/// `NaN`/`Infinity` (oracle `pd-nansuf`/`pd-infsuf`).
/// Rejects the `nan`/`inf` spellings the host parser takes but
/// the JVM refuses, and all non-ASCII digits (oracle
/// `pd-arabic`/`pd-fullwidth`).
pub(crate) fn parse_double_str(s: &str) -> Option<f64> {
    let t = s.trim_matches(|c: char| c <= ' ');
    if t.is_empty() {
        return None;
    }
    if let Some(v) = signed_special(t) {
        return Some(v);
    }
    let core = strip_float_suffix(t);
    if signed_special(core).is_some() {
        return None;
    }
    let body = core.strip_prefix(['+', '-']).unwrap_or(core);
    if body.len() >= 2 && (body.starts_with("0x") || body.starts_with("0X")) {
        return parse_hex_float(core.starts_with('-'), &body[2..]);
    }
    if ["nan", "inf", "infinity"].contains(&body.to_ascii_lowercase().as_str()) {
        return None;
    }
    core.parse::<f64>().ok()
}

fn native_parse_boolean(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/parse-boolean", args, &[1])?;
    match &args[0] {
        Value::Str(s) => match s.as_ref() {
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            _ => Ok(Value::Nil),
        },
        other => Err(parsing_err(other)),
    }
}

fn native_parse_long(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/parse-long", args, &[1])?;
    match &args[0] {
        Value::Str(s) => Ok(parse_long_str(s).map(Value::Int).unwrap_or(Value::Nil)),
        other => Err(parsing_err(other)),
    }
}

fn native_parse_double(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/parse-double", args, &[1])?;
    match &args[0] {
        Value::Str(s) => Ok(parse_double_str(s).map(Value::Double).unwrap_or(Value::Nil)),
        other => Err(parsing_err(other)),
    }
}

/// A `reduced` wrapper: `reduce` unwraps it for early exit.
/// Identity equality (the JVM never overrides `equals` here).
#[derive(Debug)]
pub struct Reduced {
    id: u64,
    pub value: Value,
}

/// Fresh `Reduced` identities (print-only; never compared).
static REDUCED_ID: AtomicU64 = AtomicU64::new(1);

impl Reduced {
    pub(crate) fn wrap(value: Value) -> Value {
        Value::Extension(Arc::new(Self {
            id: REDUCED_ID.fetch_add(1, Ordering::Relaxed),
            value,
        }))
    }
}

/// Unwrap a `reduced` value, if any (for `reduce`).
pub(crate) fn as_reduced(v: &Value) -> Option<&Value> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<Reduced>().map(|r| &r.value),
        _ => None,
    }
}

impl ExtensionValue for Reduced {
    fn class_name(&self) -> String {
        "clojure.lang.Reduced".to_owned()
    }

    fn print(&self) -> String {
        format!(
            "#object[clojure.lang.Reduced 0x{:x} \"clojure.lang.Reduced@{:x}\"]",
            self.id, self.id
        )
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| o.id == self.id)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_u64(self.id);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn native_reduced(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/reduced", args, &[1])?;
    Ok(Reduced::wrap(args[0].clone()))
}

fn native_reducedp(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/reduced?", args, &[1])?;
    Ok(Value::Bool(as_reduced(&args[0]).is_some()))
}

/// `ex-info`: an `ExceptionInfo` value (nil messages stay nil).
fn native_ex_info(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(NativeError::arity("clojure.core/ex-info", args.len()));
    }
    let message = match &args[0] {
        Value::Nil => None,
        Value::Str(s) => Some(Arc::clone(s)),
        v => return Err(cce_err(v, "java.lang.String")),
    };
    let data = match &args[1] {
        Value::Nil => None,
        Value::Map(m) => Some(Box::new(Value::Map(m.clone()))),
        v => return Err(cce_err(v, "clojure.lang.IPersistentMap")),
    };
    let cause = if args.len() == 3 {
        match &args[2] {
            Value::Nil => None,
            Value::Throwable(t) => Some(Box::new(t.clone())),
            v => return Err(cce_err(v, "java.lang.Throwable")),
        }
    } else {
        None
    };
    Ok(Value::Throwable(Throwable {
        id: eval.next_id(),
        class: ThrowClass::ExceptionInfo,
        message,
        data,
        cause,
    }))
}

/// `ex-data`/`ex-message`/`ex-cause`: throwables project, anything
/// else (including nil) is nil.
fn native_ex_data(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/ex-data", args, &[1])?;
    Ok(match &args[0] {
        Value::Throwable(t) => t.data.as_deref().cloned().unwrap_or(Value::Nil),
        _ => Value::Nil,
    })
}

fn native_ex_message(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/ex-message", args, &[1])?;
    Ok(match &args[0] {
        Value::Throwable(t) => t
            .message
            .as_ref()
            .map_or(Value::Nil, |m| Value::Str(Arc::clone(m))),
        _ => Value::Nil,
    })
}

fn native_ex_cause(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/ex-cause", args, &[1])?;
    Ok(match &args[0] {
        Value::Throwable(t) => t
            .cause
            .as_ref()
            .map_or(Value::Nil, |c| Value::Throwable((**c).clone())),
        _ => Value::Nil,
    })
}

/// S20 oddballs (#55): one cause-chain link's emulated frame line —
/// the link ex-data's `:clojure.error/line` when it is an `Int`
/// (reader diagnostics ride there), else `-1` (the JVM unknown-line
/// convention).
fn s20_frame_line(link: &Throwable) -> i64 {
    if let Some(Value::Map(m)) = link.data.as_deref() {
        let key = Value::Keyword(Keyword::new(Some("clojure.error"), "line"));
        if let Some(Value::Int(n)) = m.get(&key) {
            return *n;
        }
    }
    -1
}

/// S20: one chain link's emulated `[class method file line]` frame.
/// The class is the link's JVM name as a symbol; the method is the
/// stable symbol `unknown` (no method names exist in port
/// diagnostics — N3 stable-form); the file is nil (a JVM-legal
/// nullable slot). Slot classes stay JVM-shaped: symbol, symbol,
/// string-or-nil, int (probes `a-via0-at`, `e-ste`).
fn s20_frame(link: &Throwable) -> Value {
    Value::Vector(Vector::from_vec(vec![
        Value::Symbol(Symbol::new(None, link.class.jvm_name())),
        Value::Symbol(Symbol::new(None, "unknown")),
        Value::Nil,
        Value::Int(s20_frame_line(link)),
    ]))
}

/// S20: an `ExceptionInfo` link's `:data` value — the alpha10
/// `ex-info` nil→`{}` normalization (probe `l-raw-data`), applied at
/// the projection because the port's `ex-info` still stores nil.
fn s20_link_data(link: &Throwable) -> Value {
    link.data
        .as_deref()
        .cloned()
        .unwrap_or_else(|| Value::Map(new_map()))
}

/// S20: one `:via` entry for a chain link (`:type`/`:at` always,
/// `:message` iff non-nil, `:data` iff `ExceptionInfo`).
fn s20_via_entry(link: &Throwable) -> Value {
    let mut entry = new_map();
    entry = entry.insert(
        Value::Keyword(Keyword::new(None, "type")),
        Value::Symbol(Symbol::new(None, link.class.jvm_name())),
    );
    entry = entry.insert(Value::Keyword(Keyword::new(None, "at")), s20_frame(link));
    if let Some(message) = link.message.as_ref() {
        entry = entry.insert(
            Value::Keyword(Keyword::new(None, "message")),
            Value::Str(Arc::clone(message)),
        );
    }
    if link.class == ThrowClass::ExceptionInfo {
        entry = entry.insert(
            Value::Keyword(Keyword::new(None, "data")),
            s20_link_data(link),
        );
    }
    Value::Map(entry)
}

/// S20 `Throwable->map`: `{:cause :data :via :trace}` over the cause
/// chain. Top `:cause`/`:data` are conditional keys (root message
/// iff non-nil; root ex-data iff the root is an `ExceptionInfo`);
/// `:trace` runs the per-link `:at` frames root-first, so its first
/// frame is the root link's `:at` (probes `errors*.out` pin every
/// rule, including the root-stack `:trace` identity).
fn native_throwable_map(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/Throwable->map", args, &[1])?;
    let Value::Throwable(outer) = &args[0] else {
        if matches!(args[0], Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"java.lang.Throwable.getStackTrace()\" because the return value of \"clojure.core$identity.invokeStatic(Object)\" is null"
                    .to_owned(),
            ));
        }
        return Err(cce_err(&args[0], "java.lang.Throwable"));
    };
    let mut chain: Vec<&Throwable> = Vec::new();
    let mut cursor = Some(outer);
    while let Some(link) = cursor {
        chain.push(link);
        cursor = link.cause.as_deref();
    }
    let mut root = outer;
    while let Some(cause) = root.cause.as_deref() {
        root = cause;
    }
    let via: Vec<Value> = chain.iter().map(|link| s20_via_entry(link)).collect();
    let trace: Vec<Value> = chain.iter().rev().map(|link| s20_frame(link)).collect();
    let mut top = new_map();
    top = top.insert(
        Value::Keyword(Keyword::new(None, "via")),
        Value::Vector(Vector::from_vec(via)),
    );
    top = top.insert(
        Value::Keyword(Keyword::new(None, "trace")),
        Value::Vector(Vector::from_vec(trace)),
    );
    if let Some(message) = root.message.as_ref() {
        top = top.insert(
            Value::Keyword(Keyword::new(None, "cause")),
            Value::Str(Arc::clone(message)),
        );
    }
    if root.class == ThrowClass::ExceptionInfo {
        top = top.insert(
            Value::Keyword(Keyword::new(None, "data")),
            s20_link_data(root),
        );
    }
    Ok(Value::Map(top))
}

/// S20 `StackTraceElement->vec`: the argument throwable's emulated
/// `:at` frame. The port has no `StackTraceElement` values, so the
/// domain widens to throwables (the documented emulation — spec);
/// errors still mirror the JVM hints.
fn native_ste_vec(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/StackTraceElement->vec", args, &[1])?;
    let Value::Throwable(t) = &args[0] else {
        if matches!(args[0], Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"java.lang.StackTraceElement.getClassName()\" because \"o\" is null"
                    .to_owned(),
            ));
        }
        return Err(cce_err(&args[0], "java.lang.StackTraceElement"));
    };
    Ok(s20_frame(t))
}

/// `trampoline`: call, then bounce zero-arg while the result is a
/// `fn?` (the initial call takes arguments; anything non-`IFn` is
/// the `IFn` cast failure).
fn native_trampoline(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/trampoline", 0));
    }
    check_ifn(eval, &args[0]).map_err(signal_to_native)?;
    let mut result = eval.call(&args[0], &args[1..]).map_err(signal_to_native)?;
    while is_fn_val(&result) {
        result = eval.call(&result, &[]).map_err(signal_to_native)?;
    }
    Ok(result)
}

/// `hash`: `hasheq` after a full realize (like the JVM walk).
/// Unbounded `Repeat`/`Range`, zero-step `Range`, any
/// `Cycle`/`Iterate` — top-level, nested, or revealed by forcing —
/// throw bare `UnsupportedOperationException` (the JVM `hasheq`
/// overrides); forcing a genuinely infinite lazy thunk hangs, also
/// like the JVM.
fn native_hash(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/hash", args, &[1])?;
    Ok(Value::Int(i64::from(hasheq_one(eval, &args[0])?)))
}

/// One value's `hasheq`: unhashable infinites (top-level, nested, or
/// revealed by forcing) throw bare `UnsupportedOperationException`;
/// anything else hashes after a full realize (like the JVM walk).
fn hasheq_one(eval: &mut Evaluator, v: &Value) -> Result<i32, NativeError> {
    if clj_value::seq_hash_unsupported(v) {
        return Err(hash_unsupported());
    }
    let realized = realize_deep(eval, v).map_err(signal_to_native)?;
    if clj_value::seq_hash_unsupported(&realized) {
        return Err(hash_unsupported());
    }
    Ok(clj_value::hasheq(&realized))
}

/// The bare `UnsupportedOperationException` for unhashable infinite
/// seqs (the JVM `hasheq` overrides throw with an empty message).
fn hash_unsupported() -> NativeError {
    NativeError::new(ThrowClass::UnsupportedOperationException, String::new())
}

/// The `hash-*-coll` nil NPE (`Iterable.iterator` on a null `xs`).
const ITERABLE_NIL: &str = "Cannot invoke \"java.lang.Iterable.iterator()\" because \"xs\" is null";

/// Whether the port seqs the value (the `Iterable` analog): the
/// `seq` walk's accepted set — `is_seqable` plus subvectors and
/// eductions, which `step_unit` walks but `is_seqable` omits.
fn is_port_iterable(v: &Value) -> bool {
    crate::lazy::is_seqable(v)
        || as_subvector(v).is_some()
        || crate::xform::Eduction::parts(v).is_some()
}

/// A `hash-*-coll` argument's items: nil trips the `Iterable` NPE;
/// strings fail the `Iterable` cast (seqable here, not `Iterable`
/// on the JVM); anything else the port cannot seq fails the cast
/// too; the rest walks like `seq` (maps yield entries, entries
/// their pair).
fn hash_coll_items(eval: &mut Evaluator, v: &Value) -> Result<Vec<Value>, NativeError> {
    match v {
        Value::Nil => Err(npe(ITERABLE_NIL)),
        Value::Str(_) => Err(cce_err(v, "java.lang.Iterable")),
        _ if !is_port_iterable(v) => Err(cce_err(v, "java.lang.Iterable")),
        _ => collect_all(eval, v.clone()).map_err(signal_to_native),
    }
}

/// `hash-ordered-coll`: the Murmur3 ordered mix over the arg's
/// items, each hashed exactly like `(hash item)` (map entries hash
/// as their `[k v]` pair, like `(hash entry)`).
fn native_hash_ordered_coll(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/hash-ordered-coll", args, &[1])?;
    let items = hash_coll_items(eval, &args[0])?;
    let mut hashes = Vec::with_capacity(items.len());
    for item in &items {
        hashes.push(hasheq_one(eval, item)?);
    }
    Ok(Value::Int(i64::from(clj_value::hash_ordered(
        hashes.into_iter(),
    ))))
}

/// `hash-unordered-coll`: the Murmur3 unordered mix over the arg's
/// items, each hashed exactly like `(hash item)`.
fn native_hash_unordered_coll(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/hash-unordered-coll", args, &[1])?;
    let items = hash_coll_items(eval, &args[0])?;
    let mut hashes = Vec::with_capacity(items.len());
    for item in &items {
        hashes.push(hasheq_one(eval, item)?);
    }
    Ok(Value::Int(i64::from(clj_value::hash_unordered(
        hashes.into_iter(),
    ))))
}

/// `hash-code` (PRIVATE, port-internal): Java `.hashCode` for the
/// `case` macro's dispatch tables (`Util/hash`, not murmur —
/// `case-map`/`merge-hash-collisions` call it where the JVM calls
/// `clojure.lang.Util/hash`). Same realize + infinite-UOE contract
/// as `hash` (the JVM `hashCode` overrides throw too).
fn native_hash_code(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/hash-code", args, &[1])?;
    if clj_value::seq_hash_unsupported(&args[0]) {
        return Err(NativeError::new(
            ThrowClass::UnsupportedOperationException,
            String::new(),
        ));
    }
    let v = realize_deep(eval, &args[0]).map_err(signal_to_native)?;
    if clj_value::seq_hash_unsupported(&v) {
        return Err(NativeError::new(
            ThrowClass::UnsupportedOperationException,
            String::new(),
        ));
    }
    Ok(Value::Int(i64::from(clj_value::hash_code(&v))))
}

/// `throw-illegal-arg!` (PRIVATE, port-internal): `core.clj` macros
/// have no `new`, so `cond`/`assert-args`/`defn`/`for`/`case`/
/// `condp` expansion and no-match failures throw through here to keep
/// the exact JVM `IllegalArgumentException` class.
fn native_throw_iae(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/throw-illegal-arg!", args, &[1])?;
    let s = str_concat(eval, args)?;
    Err(NativeError::new(ThrowClass::IllegalArgumentException, s))
}

/// `throw-npe!` (PRIVATE, port-internal): `doseq`'s invalid-keyword
/// path throws the JVM's `steppair` `NullPointerException` verbatim
/// (a JVM bug the corpus pins — `doseq` never validates modifiers).
fn native_throw_npe(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/throw-npe!", args, &[1])?;
    let s = str_concat(eval, args)?;
    Err(NativeError::new(ThrowClass::NullPointerException, s))
}

/// `odd?`/`even?`: the low bit of every integer kind (`Int`,
/// `BigInt`, `BigInteger` — upstream checks `integer?`, so
/// bigints answer too); two's complement throughout, so
/// negatives work. Anything else is `Argument must be an
/// integer: <str>`.
fn odd_even_bit(v: &Value) -> Option<bool> {
    match v {
        Value::Int(n) => Some((n & 1) == 1),
        Value::BigInt(n) | Value::BigInteger(n) => Some(n.bit(0)),
        _ => None,
    }
}

fn odd_even_miss(v: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("Argument must be an integer: {}", crate::ns::str_of(v)),
    )
}

fn native_odd(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/odd?", args, &[1])?;
    match odd_even_bit(&args[0]) {
        Some(bit) => Ok(Value::Bool(bit)),
        None => Err(odd_even_miss(&args[0])),
    }
}

fn native_even(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/even?", args, &[1])?;
    match odd_even_bit(&args[0]) {
        Some(bit) => Ok(Value::Bool(!bit)),
        None => Err(odd_even_miss(&args[0])),
    }
}

/// `clojure-version`: the port release string (S7 P3). A literal,
/// not `env!("CARGO_PKG_VERSION")`: versions live per-crate while
/// the release version belongs to the workspace — the literal keeps
/// every crate reporting the release (bump it with the release;
/// `clj-repl` tests pin the semver shape).
fn native_clojure_version(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/clojure-version", args, &[0])?;
    Ok(Value::Str(Arc::from("1.13.0-alpha11")))
}

/// `destructure` (S4b): the public expansion helper over
/// `destructure_expand::destructure_public` (oracle `dt-*`).
fn native_destructure(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/destructure", args, &[1])?;
    crate::destructure_expand::destructure_public(eval, &args[0])
}

/// `seq-to-map-for-destructuring` (S4b, oracle `sm-*`).
fn native_seq_to_map(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/seq-to-map-for-destructuring", args, &[1])?;
    crate::destructure_expand::seq_to_map_for_destructuring(eval, &args[0])
}

/// `munge` (S4b): `(str s)` through `Compiler/munge`
/// (`ext::compiler_munge`), symbol-preserving (oracle
/// `mg-*`). The munged text never contains `/`, so the
/// rebuilt symbol is always unqualified (like the JVM's).
fn native_munge(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/munge", args, &[1])?;
    let s = str_concat(eval, args)?;
    let munged = crate::ext::compiler_munge(&s);
    if matches!(args[0], Value::Symbol(_)) {
        Ok(Value::Symbol(Symbol::intern(&munged)))
    } else {
        Ok(Value::Str(Arc::from(munged.as_str())))
    }
}

/// `namespace-munge` (S4b): `(str ns)` with `-` → `_`,
/// nothing else (oracle `nm-*`).
fn native_namespace_munge(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/namespace-munge", args, &[1])?;
    let s = str_concat(eval, args)?;
    Ok(Value::Str(Arc::from(s.replace('-', "_").as_str())))
}

/// `__local-var*` (S4b): the `with-local-vars` helper — a
/// fresh anonymous dynamic var (see
/// `Registry::fresh_anonymous`; unlisted, `__multifn*`
/// precedent).
fn native_fresh_local_var(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/__local-var*", args, &[0])?;
    let id = eval.registry.fresh_anonymous();
    Ok(crate::ns::var_value(id, &Symbol::new(None, "--unnamed--")))
}

/// Promoted `+'`: the checked [`add`] lattice with `Int`
/// arms that grow into `BigInt` on overflow (oracles
/// `ap-max1`, `ap-fold`, `mc-addp`; the checked fold is
/// called, not altered).
fn add_prime(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => Ok(x.checked_add(*y).map(Num::Int).unwrap_or_else(|| {
            Num::BigInt(num_bigint::BigInt::from(*x) + num_bigint::BigInt::from(*y))
        })),
        _ => add(eval, a, b),
    }
}

/// Promoted `-'` (same lattice as [`add_prime`]).
fn sub_prime(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => Ok(x.checked_sub(*y).map(Num::Int).unwrap_or_else(|| {
            Num::BigInt(num_bigint::BigInt::from(*x) - num_bigint::BigInt::from(*y))
        })),
        _ => sub(eval, a, b),
    }
}

/// Promoted `*'` (same lattice as [`add_prime`]).
fn mul_prime(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => Ok(x.checked_mul(*y).map(Num::Int).unwrap_or_else(|| {
            Num::BigInt(num_bigint::BigInt::from(*x) * num_bigint::BigInt::from(*y))
        })),
        _ => mul(eval, a, b),
    }
}

/// The `+'`/`*'` 1-arity `(cast Number x)`: numbers (and nil)
/// pass through, anything else fails `Cannot cast {from} to
/// java.lang.Number` (oracles `ap-ar1s`, `ap-ar1nil`,
/// `a2-addchar`, `g-mulnil`).
fn cast_to_number(v: &Value) -> Result<Value, NativeError> {
    match v {
        Value::Int(_)
        | Value::Double(_)
        | Value::BigInt(_)
        | Value::BigInteger(_)
        | Value::Ratio(_)
        | Value::BigDec(_)
        | Value::Nil => Ok(v.clone()),
        v => Err(NativeError::cast(&jvm_class_of(v), "java.lang.Number")),
    }
}

fn native_add_prime(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Int(0));
    }
    if args.len() == 1 {
        return cast_to_number(&args[0]);
    }
    fold(eval, args, add_prime)
}

/// Promoting negate (the `-'` 1-arity): `MIN` grows to
/// `BigInt`, everything else keeps the checked shape
/// (oracles `sp-negmin`, `sp-ar1*`).
fn negate_prime(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) => Ok(x
            .checked_neg()
            .map(Value::Int)
            .unwrap_or_else(|| Value::BigInt(Arc::new(-num_bigint::BigInt::from(x))))),
        Num::BigInt(x) => Ok(Value::BigInt(Arc::new(-x))),
        Num::Ratio(x) => Ok(num_value(num_ratio(-x))),
        Num::BigDec(x) => Ok(bigdec_result(match math_context_of(eval)? {
            None => x.neg(),
            Some(mc) => x.neg_mc(&mc).map_err(arithmetic)?,
        })),
        Num::Double(x) => Ok(Value::Double(-x)),
    }
}

fn native_sub_prime(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/-'", 0));
    }
    if args.len() == 1 {
        return negate_prime(eval, as_num(&args[0])?);
    }
    fold(eval, args, sub_prime)
}

fn native_mul_prime(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Int(1));
    }
    if args.len() == 1 {
        return cast_to_number(&args[0]);
    }
    fold(eval, args, mul_prime)
}

/// One promoting step up: only `MAX` overflows into `BigInt`
/// — the checked [`inc_one`] answers everything else
/// (oracles `ip-*`).
fn inc_prime_one(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) if x == i64::MAX => {
            Ok(Value::BigInt(Arc::new(num_bigint::BigInt::from(x) + 1)))
        }
        n => inc_one(eval, n),
    }
}

/// One promoting step down (mirror of [`inc_prime_one`];
/// oracles `dp-*`).
fn dec_prime_one(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) if x == i64::MIN => {
            Ok(Value::BigInt(Arc::new(num_bigint::BigInt::from(x) - 1)))
        }
        n => dec_one(eval, n),
    }
}

fn native_dec_prime(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/dec'", args, &[1])?;
    dec_prime_one(eval, as_num(&args[0])?)
}

/// Unchecked `+`: the checked [`add`] lattice with WRAPPING
/// `Int` arms (tower/double results are value-identical to
/// the checked op — oracles `ua-wrap`, `ua-big`, `ua-rat`,
/// `ua-dec`, `xm-addu`).
fn add_unchecked(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => Ok(Num::Int(x.wrapping_add(*y))),
        _ => add(eval, a, b),
    }
}

/// Unchecked `-` (same lattice as [`add_unchecked`]).
fn sub_unchecked(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => Ok(Num::Int(x.wrapping_sub(*y))),
        _ => sub(eval, a, b),
    }
}

/// Unchecked `*` (same lattice as [`add_unchecked`]).
fn mul_unchecked(eval: &mut Evaluator, a: Num, b: Num) -> Result<Num, NativeError> {
    match (&a, &b) {
        (Num::Int(x), Num::Int(y)) => Ok(Num::Int(x.wrapping_mul(*y))),
        _ => mul(eval, a, b),
    }
}

/// One unchecked step up: `Int` wraps, the checked
/// [`inc_one`] answers every other kind (oracles `ui-*`).
fn inc_unchecked_one(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) => Ok(Value::Int(x.wrapping_add(1))),
        n => inc_one(eval, n),
    }
}

/// One unchecked step down (mirror of [`inc_unchecked_one`];
/// oracles `ud-*`).
fn dec_unchecked_one(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) => Ok(Value::Int(x.wrapping_sub(1))),
        n => dec_one(eval, n),
    }
}

/// Unchecked negate: `Int` wraps (`MIN` is a fixed point),
/// every other kind keeps the checked shape (oracles `un-*`).
fn negate_unchecked(eval: &mut Evaluator, n: Num) -> Result<Value, NativeError> {
    match n {
        Num::Int(x) => Ok(Value::Int(x.wrapping_neg())),
        Num::BigInt(x) => Ok(Value::BigInt(Arc::new(-x))),
        Num::Ratio(x) => Ok(num_value(num_ratio(-x))),
        Num::BigDec(x) => Ok(bigdec_result(match math_context_of(eval)? {
            None => x.neg(),
            Some(mc) => x.neg_mc(&mc).map_err(arithmetic)?,
        })),
        Num::Double(x) => Ok(Value::Double(-x)),
    }
}

fn native_unchecked_add(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-add", args, &[2])?;
    add_unchecked(eval, as_num(&args[0])?, as_num(&args[1])?).map(num_value)
}

fn native_unchecked_subtract(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-subtract", args, &[2])?;
    sub_unchecked(eval, as_num(&args[0])?, as_num(&args[1])?).map(num_value)
}

fn native_unchecked_multiply(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-multiply", args, &[2])?;
    mul_unchecked(eval, as_num(&args[0])?, as_num(&args[1])?).map(num_value)
}

fn native_unchecked_inc(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-inc", args, &[1])?;
    inc_unchecked_one(eval, as_num(&args[0])?)
}

fn native_unchecked_dec(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-dec", args, &[1])?;
    dec_unchecked_one(eval, as_num(&args[0])?)
}

fn native_unchecked_negate(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-negate", args, &[1])?;
    negate_unchecked(eval, as_num(&args[0])?)
}

/// The `-int` op argument coercion (CHECKED — oracles
/// `uai-long`, `xi-*`, `g-int*`): `Int` takes `toIntExact`,
/// `Double` the `long` range check first, tower integers the
/// `long` stop with exact digits, decimals go through
/// `double`; nil is the `charValue` NPE quirk, chars and
/// everything else fail to `Number` (chars do NOT convert
/// here, unlike `int` — oracle `xi-char`).
pub(crate) fn intcast(v: &Value) -> Result<i32, NativeError> {
    match v {
        Value::Int(i) => int_exact_long(*i),
        Value::Double(d) => int_cast_double(*d),
        Value::BigInt(n) | Value::BigInteger(n) => tower_intcast(n),
        Value::Ratio(r) => tower_intcast(&trunc_ratio(r)),
        Value::BigDec(d) => bigdec_intcast(d),
        Value::Nil => Err(npe_char_value()),
        v => Err(cce_err(v, "java.lang.Number")),
    }
}

/// Two `-int` operands (fixed 2-arity).
fn int_args2(display: &str, args: &[Value]) -> Result<(i32, i32), NativeError> {
    check_arity(display, args, &[2])?;
    Ok((intcast(&args[0])?, intcast(&args[1])?))
}

/// One `-int` operand (fixed 1-arity).
fn int_arg1(display: &str, args: &[Value]) -> Result<i32, NativeError> {
    check_arity(display, args, &[1])?;
    intcast(&args[0])
}

fn native_unchecked_add_int(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let (x, y) = int_args2("clojure.core/unchecked-add-int", args)?;
    Ok(Value::Int(i64::from(x.wrapping_add(y))))
}

fn native_unchecked_subtract_int(
    _eval: &mut Evaluator,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (x, y) = int_args2("clojure.core/unchecked-subtract-int", args)?;
    Ok(Value::Int(i64::from(x.wrapping_sub(y))))
}

fn native_unchecked_multiply_int(
    _eval: &mut Evaluator,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (x, y) = int_args2("clojure.core/unchecked-multiply-int", args)?;
    Ok(Value::Int(i64::from(x.wrapping_mul(y))))
}

fn native_unchecked_divide_int(
    _eval: &mut Evaluator,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (x, y) = int_args2("clojure.core/unchecked-divide-int", args)?;
    if y == 0 {
        return Err(slash_by_zero());
    }
    Ok(Value::Int(i64::from(x.wrapping_div(y))))
}

fn native_unchecked_remainder_int(
    _eval: &mut Evaluator,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (x, y) = int_args2("clojure.core/unchecked-remainder-int", args)?;
    if y == 0 {
        return Err(slash_by_zero());
    }
    Ok(Value::Int(i64::from(x.wrapping_rem(y))))
}

fn native_unchecked_inc_int(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let x = int_arg1("clojure.core/unchecked-inc-int", args)?;
    Ok(Value::Int(i64::from(x.wrapping_add(1))))
}

fn native_unchecked_dec_int(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let x = int_arg1("clojure.core/unchecked-dec-int", args)?;
    Ok(Value::Int(i64::from(x.wrapping_sub(1))))
}

fn native_unchecked_negate_int(
    _eval: &mut Evaluator,
    args: &[Value],
) -> Result<Value, NativeError> {
    let x = int_arg1("clojure.core/unchecked-negate-int", args)?;
    Ok(Value::Int(i64::from(x.wrapping_neg())))
}

/// A Java `(int)` cast: `NaN` is 0, past-`i32` saturates,
/// else truncation toward zero (oracles `uci-nan`,
/// `uci-inf`, `uci-hugedbl`, `xr-intwrap`).
fn java_double_to_int(d: f64) -> i32 {
    if d.is_nan() {
        0
    } else if d >= 2147483648.0 {
        i32::MAX
    } else if d < -2147483648.0 {
        i32::MIN
    } else {
        d.trunc() as i32
    }
}

/// A Java `(long)` cast: `NaN` is 0, past-`i64` saturates,
/// else truncation toward zero (the saturating sibling of
/// [`double_long`] — oracles `ucl-nan`, `ucl-inf`,
/// `ucl-hugedbl`).
fn java_double_to_long(d: f64) -> i64 {
    if d.is_nan() {
        0
    } else if d >= 9223372036854775808.0 {
        i64::MAX
    } else if d < -9223372036854775808.0 {
        i64::MIN
    } else {
        d.trunc() as i64
    }
}

/// Java `longValue` over an unbounded integer: low 64 bits,
/// sign-extended (the 8-byte sibling of [`bigint_int_value`];
/// oracles `ucl-big`, `ucl-negbig`).
pub(crate) fn bigint_long_value(n: &num_bigint::BigInt) -> i64 {
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

/// The `floatValue` NPE (`unchecked-float` of nil).
fn npe_float_value() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Number.floatValue()\" because \"x\" is null".to_owned(),
    )
}

/// The `shortValue` NPE (`unchecked-short` of nil).
fn npe_short_value() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Number.shortValue()\" because \"x\" is null".to_owned(),
    )
}

/// The `byteValue` NPE (`unchecked-byte` of nil).
fn npe_byte_value() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Number.byteValue()\" because \"x\" is null".to_owned(),
    )
}

fn native_unchecked_int(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-int", args, &[1])?;
    // Truncating `intValue` (ratios saturate through `double`
    // — oracle `xr-intwrap`; characters convert, nil is the
    // `charValue` NPE, anything else fails to `Character`).
    Ok(Value::Int(i64::from(match &args[0] {
        Value::Int(i) => *i as i32,
        Value::Double(d) => java_double_to_int(*d),
        Value::BigInt(n) | Value::BigInteger(n) => bigint_int_value(n),
        Value::Ratio(r) => java_double_to_int(r.to_f64().unwrap_or(f64::NAN)),
        Value::BigDec(d) => bigint_int_value(&d.to_bigint_trunc()),
        Value::Char(c) => *c as i32,
        Value::Nil => return Err(npe_char_value()),
        v => return Err(cce_err(v, "java.lang.Character")),
    })))
}

fn native_unchecked_long(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-long", args, &[1])?;
    // Truncating `longValue` (characters do NOT convert —
    // oracle `ucl-char`).
    Ok(Value::Int(match &args[0] {
        Value::Int(i) => *i,
        Value::Double(d) => java_double_to_long(*d),
        Value::BigInt(n) | Value::BigInteger(n) => bigint_long_value(n),
        Value::Ratio(r) => bigint_long_value(&trunc_ratio(r)),
        Value::BigDec(d) => bigint_long_value(&d.to_bigint_trunc()),
        Value::Nil => return Err(npe_long_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    }))
}

fn native_unchecked_float(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-float", args, &[1])?;
    // No `Float` exists: like `float` but unchecked — the
    // EXACT double, so common values print identically
    // (oracle `ucf-dbl` prints `0.1` either way);
    // out-of-`f32` values stay exact instead of `Inf`
    // (oracle `ucf-huge` — documented, pinned).
    Ok(Value::Double(match &args[0] {
        Value::Int(i) => *i as f64,
        Value::Double(d) => *d,
        Value::BigInt(n) | Value::BigInteger(n) => n.to_f64().unwrap_or(f64::NAN),
        Value::Ratio(r) => r.to_f64().unwrap_or(f64::NAN),
        Value::BigDec(d) => d.to_f64(),
        Value::Nil => return Err(npe_float_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    }))
}

fn native_unchecked_double(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-double", args, &[1])?;
    // Extensionally `double` (oracles `ucd-*`, `xc-dchar`).
    Ok(Value::Double(match &args[0] {
        Value::Int(i) => *i as f64,
        Value::Double(d) => *d,
        Value::BigInt(n) | Value::BigInteger(n) => n.to_f64().unwrap_or(f64::NAN),
        Value::Ratio(r) => r.to_f64().unwrap_or(f64::NAN),
        Value::BigDec(d) => d.to_f64(),
        Value::Nil => return Err(npe_double_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    }))
}

fn native_unchecked_short(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-short", args, &[1])?;
    // Truncating `shortValue` (doubles and ratios narrow
    // through the Java `(int)` cast first — oracles
    // `xc-sinf`, `g-shortrat`).
    Ok(Value::Int(i64::from(match &args[0] {
        Value::Int(i) => *i as i16,
        Value::Double(d) => java_double_to_int(*d) as i16,
        Value::BigInt(n) | Value::BigInteger(n) => bigint_int_value(n) as i16,
        Value::Ratio(r) => java_double_to_int(r.to_f64().unwrap_or(f64::NAN)) as i16,
        Value::BigDec(d) => bigint_int_value(&d.to_bigint_trunc()) as i16,
        Value::Nil => return Err(npe_short_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    })))
}

fn native_unchecked_byte(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-byte", args, &[1])?;
    // Truncating `byteValue` (the `unchecked-short` shape at
    // 8 bits — oracles `ucb-*`, `g-byterat`, `xc-binf`).
    Ok(Value::Int(i64::from(match &args[0] {
        Value::Int(i) => *i as i8,
        Value::Double(d) => java_double_to_int(*d) as i8,
        Value::BigInt(n) | Value::BigInteger(n) => bigint_int_value(n) as i8,
        Value::Ratio(r) => java_double_to_int(r.to_f64().unwrap_or(f64::NAN)) as i8,
        Value::BigDec(d) => bigint_int_value(&d.to_bigint_trunc()) as i8,
        Value::Nil => return Err(npe_byte_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    })))
}

fn native_unchecked_char(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/unchecked-char", args, &[1])?;
    // `(char)longValue` over numbers (oracles `ucc-*`,
    // `xc-charhuge`, `xr-char`); lone surrogates share the
    // checked-`char` divergence (house range error).
    let w = match &args[0] {
        Value::Char(c) => return Ok(Value::Char(*c)),
        Value::Int(i) => *i,
        Value::Double(d) => java_double_to_long(*d),
        Value::BigInt(n) | Value::BigInteger(n) => bigint_long_value(n),
        Value::Ratio(r) => bigint_long_value(&trunc_ratio(r)),
        Value::BigDec(d) => bigint_long_value(&d.to_bigint_trunc()),
        Value::Nil => return Err(npe_long_value()),
        v => return Err(cce_err(v, "java.lang.Number")),
    };
    match char::from_u32(w as u16 as u32) {
        Some(c) => Ok(Value::Char(c)),
        None => Err(out_of_range("char", Num::Int(w))),
    }
}

/// A bitwise operand: `Int` only (the JVM also accepts
/// `Integer`/`Short`/`Byte` — oracles `jt-*` — kinds the
/// port lacks); nil is the `getClass` NPE, anything else is
/// the `bit operation not supported for` IAE.
fn bit_operand(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Nil => Err(npe_getclass()),
        v => Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("bit operation not supported for: class {}", jvm_class_of(v)),
        )),
    }
}

/// A variadic (`[x y & more]`) bitwise left fold; operands
/// validate left to right, so the FIRST offender is named
/// (oracles `g-order`, `g-order2`, `va-andbad3`).
fn bit_fold(display: &str, args: &[Value], op: fn(i64, i64) -> i64) -> Result<Value, NativeError> {
    if args.len() < 2 {
        return Err(NativeError::arity(display, args.len()));
    }
    let mut acc = bit_operand(&args[0])?;
    for arg in &args[1..] {
        acc = op(acc, bit_operand(arg)?);
    }
    Ok(Value::Int(acc))
}

fn native_bit_and(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    bit_fold("clojure.core/bit-and", args, |x, y| x & y)
}

fn native_bit_or(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    bit_fold("clojure.core/bit-or", args, |x, y| x | y)
}

fn native_bit_xor(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    bit_fold("clojure.core/bit-xor", args, |x, y| x ^ y)
}

fn native_bit_and_not(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    bit_fold("clojure.core/bit-and-not", args, |x, y| x & !y)
}

fn native_bit_not(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/bit-not", args, &[1])?;
    Ok(Value::Int(!bit_operand(&args[0])?))
}

/// Two bitwise operands (fixed 2-arity).
fn bit_args2(display: &str, args: &[Value]) -> Result<(i64, i64), NativeError> {
    check_arity(display, args, &[2])?;
    Ok((bit_operand(&args[0])?, bit_operand(&args[1])?))
}

fn native_bit_shift_left(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let (x, n) = bit_args2("clojure.core/bit-shift-left", args)?;
    // `wrapping_shl` masks the count to 6 bits exactly like
    // Java `<<` (oracles `bsl-mask`, `bsl-negsh`).
    Ok(Value::Int(x.wrapping_shl(n as u32)))
}

fn native_bit_shift_right(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let (x, n) = bit_args2("clojure.core/bit-shift-right", args)?;
    Ok(Value::Int(x.wrapping_shr(n as u32)))
}

fn native_unsigned_bit_shift_right(
    _eval: &mut Evaluator,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (x, n) = bit_args2("clojure.core/unsigned-bit-shift-right", args)?;
    Ok(Value::Int((x as u64).wrapping_shr(n as u32) as i64))
}

fn native_bit_set(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let (x, n) = bit_args2("clojure.core/bit-set", args)?;
    Ok(Value::Int(x | 1i64.wrapping_shl(n as u32)))
}

fn native_bit_clear(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let (x, n) = bit_args2("clojure.core/bit-clear", args)?;
    Ok(Value::Int(x & !1i64.wrapping_shl(n as u32)))
}

fn native_bit_flip(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let (x, n) = bit_args2("clojure.core/bit-flip", args)?;
    Ok(Value::Int(x ^ 1i64.wrapping_shl(n as u32)))
}

fn native_bit_test(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let (x, n) = bit_args2("clojure.core/bit-test", args)?;
    Ok(Value::Bool(x & 1i64.wrapping_shl(n as u32) != 0))
}

/// Genuine core arith/compare natives by fn pointer (the
/// direct-call inline tags — see [`FastArith`]).
const FAST_ARITH_TABLE: &[(NativeFnPtr, FastArith)] = &[
    (native_add, FastArith::Add),
    (native_sub, FastArith::Sub),
    (native_mul, FastArith::Mul),
    (native_lt, FastArith::Lt),
    (native_lte, FastArith::Lte),
    (native_gt, FastArith::Gt),
    (native_gte, FastArith::Gte),
    (native_inc, FastArith::Inc),
    (native_dec, FastArith::Dec),
];

/// The inline integer op for an installing fn pointer (`None`
/// for everything else — closures, test doubles, and fakes
/// installed under arith names never match). Identity is the
/// code address ([`std::ptr::fn_addr_eq`], never the name): a
/// match means `f` shares the genuine fn's machine code, and
/// code-identical functions behave identically (identical-code
/// folding only equates equal bytes — address _equality_ stays
/// a behavioral claim, which is all the tag asserts; every
/// non-`Int` operand delegates to the installed `func` anyway).
/// A missed match (an address the toolchain duplicated) merely
/// runs the general snapshot call.
pub(crate) fn fast_arith_for(f: NativeFnPtr) -> FastArith {
    FAST_ARITH_TABLE
        .iter()
        .find_map(|(g, op)| std::ptr::fn_addr_eq(f, *g).then_some(*op))
        .unwrap_or(FastArith::None)
}

/// The S5D scalar natives installed into `clojure.core`.
pub(crate) const CORE_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("+", native_add),
    ("-", native_sub),
    ("*", native_mul),
    ("/", native_div),
    ("quot", native_quot),
    ("rem", native_rem),
    ("mod", native_mod),
    ("inc", native_inc),
    ("inc'", native_inc_prime),
    ("dec", native_dec),
    ("+'", native_add_prime),
    ("-'", native_sub_prime),
    ("*'", native_mul_prime),
    ("dec'", native_dec_prime),
    ("unchecked-add", native_unchecked_add),
    ("unchecked-add-int", native_unchecked_add_int),
    ("unchecked-byte", native_unchecked_byte),
    ("unchecked-char", native_unchecked_char),
    ("unchecked-dec", native_unchecked_dec),
    ("unchecked-dec-int", native_unchecked_dec_int),
    ("unchecked-divide-int", native_unchecked_divide_int),
    ("unchecked-double", native_unchecked_double),
    ("unchecked-float", native_unchecked_float),
    ("unchecked-inc", native_unchecked_inc),
    ("unchecked-inc-int", native_unchecked_inc_int),
    ("unchecked-int", native_unchecked_int),
    ("unchecked-long", native_unchecked_long),
    ("unchecked-multiply", native_unchecked_multiply),
    ("unchecked-multiply-int", native_unchecked_multiply_int),
    ("unchecked-negate", native_unchecked_negate),
    ("unchecked-negate-int", native_unchecked_negate_int),
    ("unchecked-remainder-int", native_unchecked_remainder_int),
    ("unchecked-short", native_unchecked_short),
    ("unchecked-subtract", native_unchecked_subtract),
    ("unchecked-subtract-int", native_unchecked_subtract_int),
    ("bit-and", native_bit_and),
    ("bit-and-not", native_bit_and_not),
    ("bit-clear", native_bit_clear),
    ("bit-flip", native_bit_flip),
    ("bit-not", native_bit_not),
    ("bit-or", native_bit_or),
    ("bit-set", native_bit_set),
    ("bit-shift-left", native_bit_shift_left),
    ("bit-shift-right", native_bit_shift_right),
    ("bit-test", native_bit_test),
    ("bit-xor", native_bit_xor),
    ("unsigned-bit-shift-right", native_unsigned_bit_shift_right),
    ("max", native_max),
    ("min", native_min),
    ("=", native_eq),
    ("not=", native_not_eq),
    ("==", native_num_eq),
    ("<", native_lt),
    (">", native_gt),
    ("<=", native_lte),
    (">=", native_gte),
    ("zero?", native_zero),
    ("pos?", native_pos),
    ("neg?", native_neg),
    ("compare", native_compare),
    ("nil?", native_nil),
    ("not", native_not),
    ("boolean", native_boolean),
    ("some?", native_somep),
    ("true?", native_truep),
    ("false?", native_falsep),
    ("boolean?", native_booleanp),
    ("integer?", native_integerp),
    ("int?", native_intp),
    ("float?", native_floatp),
    ("double?", native_doublep),
    ("number?", native_numberp),
    ("ratio?", native_ratiop),
    ("decimal?", native_decimalp),
    ("rational?", native_rationalp),
    ("string?", native_stringp),
    ("char?", native_charp),
    ("keyword?", native_keywordp),
    ("symbol?", native_symbolp),
    ("ident?", native_identp),
    ("qualified-ident?", native_qualified_identp),
    ("simple-ident?", native_simple_identp),
    ("qualified-keyword?", native_qualified_keywordp),
    ("simple-keyword?", native_simple_keywordp),
    ("qualified-symbol?", native_qualified_symbolp),
    ("simple-symbol?", native_simple_symbolp),
    ("coll?", native_collp),
    ("sequential?", native_sequentialp),
    ("associative?", native_associativep),
    ("counted?", native_countedp),
    ("list?", native_listp),
    ("vector?", native_vectorp),
    ("map?", native_mapp),
    ("set?", native_setp),
    ("sorted?", native_sortedp),
    ("reversible?", native_reversiblep),
    ("fn?", native_fnp),
    ("ifn?", native_ifnp),
    ("var?", native_varp),
    ("seq?", native_seqp),
    ("seqable?", native_seqablep),
    ("record?", native_recordp),
    ("any?", native_anyp),
    ("delay?", native_delayp),
    ("indexed?", native_indexedp),
    ("map-entry?", native_map_entryp),
    ("nat-int?", native_nat_intp),
    ("neg-int?", native_neg_intp),
    ("pos-int?", native_pos_intp),
    ("special-symbol?", native_special_symbolp),
    ("NaN?", native_nanp),
    ("infinite?", native_infinitep),
    ("find-keyword", native_find_keyword),
    ("tagged-literal", native_tagged_literal),
    ("tagged-literal?", native_tagged_literalp),
    ("reader-conditional", native_reader_conditional),
    ("reader-conditional?", native_reader_conditionalp),
    ("str", native_str),
    ("subs", native_subs),
    ("pr-str", native_pr_str),
    ("print-str", native_print_str),
    ("keyword", native_keyword),
    ("symbol", native_symbol),
    ("name", native_name),
    ("namespace", native_namespace),
    ("int", native_int),
    ("long", native_long),
    ("double", native_double),
    ("float", native_float),
    ("bigint", native_bigint),
    ("bigdec", native_bigdec),
    ("num", native_num),
    ("rationalize", native_rationalize),
    ("abs", native_abs),
    ("biginteger", native_biginteger),
    ("numerator", native_numerator),
    ("denominator", native_denominator),
    ("short", native_short),
    ("byte", native_byte),
    ("char", native_char),
    ("parse-boolean", native_parse_boolean),
    ("parse-double", native_parse_double),
    ("parse-long", native_parse_long),
    ("format", crate::fmt::native_format),
    ("reduced", native_reduced),
    ("reduced?", native_reducedp),
    ("ex-info", native_ex_info),
    ("ex-data", native_ex_data),
    ("ex-message", native_ex_message),
    ("ex-cause", native_ex_cause),
    ("Throwable->map", native_throwable_map),
    ("StackTraceElement->vec", native_ste_vec),
    ("trampoline", native_trampoline),
    ("hash", native_hash),
    ("hash-code", native_hash_code),
    ("hash-ordered-coll", native_hash_ordered_coll),
    ("hash-unordered-coll", native_hash_unordered_coll),
    ("throw-illegal-arg!", native_throw_iae),
    ("throw-npe!", native_throw_npe),
    ("odd?", native_odd),
    ("even?", native_even),
    ("clojure-version", native_clojure_version),
    ("destructure", native_destructure),
    ("seq-to-map-for-destructuring", native_seq_to_map),
    ("munge", native_munge),
    ("namespace-munge", native_namespace_munge),
    ("__local-var*", native_fresh_local_var),
];

/// Installed but never referred (like `__defmethod*`): the
/// port-only `math-context` ctor behind `with-precision` —
/// resolvable qualified, callable cross-namespace (a `:private`
/// mark would reject the macro's own expansion).
pub(crate) const TOWER_NATIVES: &[(&str, NativeFnPtr)] = &[("math-context", native_math_context)];

#[cfg(test)]
mod tests {
    use super::*;

    /// `num_rank` still orders `Double` top (only called past the
    /// `either_double` guards, so user code never reaches this
    /// arm — the pin keeps the promotion lattice total).
    #[test]
    fn num_rank_orders_double_top() {
        assert_eq!(num_rank(&Num::Double(0.5)), 4);
        assert_eq!(num_rank(&Num::Int(1)), 0);
    }

    /// The exact paths reject `Double` (callers dispatch on
    /// `either_double` first — the panic is the contract).
    #[test]
    #[should_panic(expected = "doubles take the f64 path, never the exact one")]
    fn num_rational_rejects_double() {
        let _ = num_rational(&Num::Double(0.5));
    }

    /// See [`num_rational_rejects_double`].
    #[test]
    #[should_panic(expected = "rank dispatch keeps doubles out of BigDec ops")]
    fn num_bigdec_rejects_double() {
        let _ = num_bigdec(&Num::Double(0.5));
    }

    /// Instants order by millis (unit-tested: `#inst` reads
    /// a `TaggedLiteral`, so no surface form builds a real
    /// instant).
    #[test]
    fn scalar_pair_orders_instants() {
        let instant = |ms: i64| Value::Instant(clj_value::Instant::from_millis(ms));
        assert_eq!(
            jvm_compare(&instant(0), &instant(1000)).unwrap(),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            jvm_compare(&instant(1000), &instant(1000)).unwrap(),
            std::cmp::Ordering::Equal
        );
    }

    /// An instant against a non-instant fails the `String`
    /// checkcast to `java.util.Date` (unit-tested: no
    /// surface form builds a real instant).
    #[test]
    fn instant_compare_cast_names_date_class() {
        let instant = Value::Instant(clj_value::Instant::from_millis(0));
        let err = jvm_compare(&instant, &Value::Str("s".into())).expect_err("inst vs str");
        assert_eq!(err.class, clj_value::ThrowClass::ClassCastException);
        assert!(
            err.message
                .starts_with("class java.lang.String cannot be cast to class java.util.Date")
        );
    }

    /// URIs order by parts (scheme, authority, path, query,
    /// fragment): `compare` over two URIs is unit-tested
    /// because upstream has no URI constructor, so no
    /// surface form builds one.
    #[test]
    fn scalar_pair_orders_uris() {
        let uri = |s: &str| Value::Uri(Box::new(clj_value::Uri::parse(s).unwrap()));
        assert_eq!(
            jvm_compare(&uri("http://h/a"), &uri("http://h/b")).unwrap(),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            jvm_compare(&uri("http://h/a"), &uri("http://h/a")).unwrap(),
            std::cmp::Ordering::Equal
        );
    }

    /// A URI against a non-URI fails the `String` checkcast to
    /// `java.net.URI` (unit-tested: no surface form builds a
    /// URI).
    #[test]
    fn uri_compare_cast_names_uri_class() {
        let uri = Value::Uri(Box::new(clj_value::Uri::parse("http://h/a").unwrap()));
        let err = jvm_compare(&uri, &Value::Str("s".into())).expect_err("uri vs str");
        assert_eq!(err.class, clj_value::ThrowClass::ClassCastException);
        assert!(
            err.message
                .starts_with("class java.lang.String cannot be cast to class java.net.URI")
        );
    }

    /// The scalar gate only admits same-kind S9 pairs (the
    /// caller partitions — the panic is the contract).
    #[test]
    #[should_panic(expected = "scalar gate")]
    fn scalar_pair_rejects_mixed() {
        let _ = compare_scalar_pair(&Value::Int(1), &Value::Int(2));
    }

    /// The entry gate only admits vector/entry pairs (the
    /// caller partitions — the panic is the contract).
    #[test]
    #[should_panic(expected = "entry gate")]
    fn entry_pair_rejects_scalars() {
        let _ = compare_entry_pair(&Value::Int(1), &Value::Int(2));
    }

    /// A denominator-1 rational folds to `BigInt` (user code only
    /// builds reduced ratios, so this arm is the safety net).
    #[test]
    fn num_value_folds_unit_denominator() {
        let unreduced = num_rational::BigRational::new(
            num_bigint::BigInt::from(2),
            num_bigint::BigInt::from(2),
        );
        assert_eq!(
            num_value(Num::Ratio(unreduced)),
            Value::BigInt(Arc::new(num_bigint::BigInt::from(1)))
        );
    }

    /// `MathContextVal` hashing is structural like its `=`
    /// (precision + rounding feed the hasher).
    #[test]
    fn math_context_hash_is_structural() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        fn hash_of(ctx: clj_value::MathContext) -> u64 {
            let v = Value::Extension(Arc::new(MathContextVal { id: 1, ctx }));
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }
        let a = clj_value::MathContext::new(10, clj_value::RoundingMode::HalfUp);
        let b = clj_value::MathContext::new(10, clj_value::RoundingMode::HalfUp);
        let c = clj_value::MathContext::new(11, clj_value::RoundingMode::HalfUp);
        assert_eq!(hash_of(a), hash_of(b));
        assert_ne!(hash_of(a), hash_of(c));
    }

    #[test]
    fn hash_code_post_realize_uoe() {
        // `hash-code` is private (unreachable from user code): like
        // `hash`, an infinite revealed by forcing throws bare UOE.
        let mut eval = crate::boot();
        let mut vs = eval
            .eval_source("(map identity [(repeat 1)])")
            .expect("shape evaluates");
        let v = vs.pop().expect("one form");
        let err = native_hash_code(&mut eval, &[v]).expect_err("post-check throws");
        assert_eq!(
            err.class,
            clj_value::ThrowClass::UnsupportedOperationException
        );
        assert!(err.message.is_empty());
    }

    /// Snapshotting over a bare evaluator skips the autoload merge
    /// (no `*data-readers*` var installed) but still enables defer
    /// mode with an empty consult.
    #[test]
    fn snapshot_bare_evaluator_skips_autoload() {
        let mut eval = Evaluator::new_bare("user");
        let mut opts = ReaderOpts::default();
        snapshot_data_readers(&mut eval, &mut opts).expect("bare snapshot");
        assert!(opts.data_readers);
        assert!(opts.data_reader_keys.is_empty());
        assert!(!opts.default_reader_fn);
    }

    /// An unreadable mapping file fails the scan (a directory in
    /// the mapping slot reads as an error on every platform, so
    /// the failure arm needs no permission juggling).
    #[test]
    fn autoload_scan_reports_unreadable_files() {
        let dir =
            std::env::temp_dir().join(format!("clj-zerofence-unreadable-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("data_readers.clj")).unwrap();
        let mut eval = crate::boot();
        let dirs = vec![dir.to_string_lossy().replace('\\', "/")];
        let mut acc = Value::Map(new_map());
        let mut entries = Vec::new();
        let mut fresh = Vec::new();
        let err = autoload_scan(&mut eval, &dirs, &mut acc, &mut entries, &mut fresh)
            .expect_err("unreadable file fails");
        assert_eq!(err.class, ThrowClass::FileNotFoundException);
        assert!(fresh.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// B2: a merge failure (mid-scan user code turned the live
    /// root non-assocable) reverts the optimistic mark and records
    /// nothing, so the next scan retries the files instead of
    /// silently skipping them forever. The bork must land
    /// mid-scan (a tag firing while the mapping file reads): a
    /// pre-borked root fails the scan instead, and healing
    /// through reads would advance the mark and mask the revert.
    #[test]
    fn autoload_merge_failure_reverts_mark_for_retry() {
        let dir =
            std::env::temp_dir().join(format!("clj-zerofence-mergerevert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut eval = crate::boot();
        // Setup reads scan nothing — no mapping files exist yet.
        eval.eval_source(
            "(do (ns my.lib) (defn borker [x] (do (alter-var-root #'clojure.core/*data-readers* (constantly 5)) 'my.lib/reader)) (in-ns 'user))",
        )
        .expect("borker defined");
        let lit = dir
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        eval.eval_source(&format!(
            "(alter-var-root #'clojure.core/*load-path* (constantly [\"{lit}\"]))"
        ))
        .expect("load-path root set");
        eval.eval_source(
            "(alter-var-root #'clojure.core/*data-readers* assoc 'bork/tag #'my.lib/borker)",
        )
        .expect("bork tag registered");
        eval.data_reader_scan_path = None;
        std::fs::write(dir.join("data_readers.clj"), "{my/tag #bork/tag [1]}").unwrap();
        let readers = crate::ns::core_var(&eval, "*data-readers*").expect("readers var");
        let err = ensure_data_readers_autoloaded(&mut eval).expect_err("merge must fail");
        assert_eq!(err.class, ThrowClass::ClassCastException);
        assert_eq!(eval.data_reader_scan_path, None, "mark reverts");
        assert!(
            eval.loaded_data_reader_files.is_empty(),
            "failed files stay unrecorded"
        );
        // Heal (directly — reads would advance the mark) and
        // rewrite the file clean; the retry must pick it up.
        eval.registry.with_state_mut(readers, |st| {
            st.root = Some(Value::Map(new_map()));
        });
        std::fs::write(dir.join("data_readers.clj"), "{my/tag my.lib/reader}").unwrap();
        ensure_data_readers_autoloaded(&mut eval).expect("retry merges");
        assert_eq!(eval.loaded_data_reader_files.len(), 1);
        let root = eval.registry.root_cloned(readers);
        // One covered assertion (a `let-else` panic arm would be a
        // cold line under the 100% floor): the merged root is a map
        // carrying `my/tag`.
        assert!(
            matches!(&root, Some(Value::Map(m)) if m.iter().any(|(k, _)| matches!(
                k,
                Value::Symbol(s) if s.namespace() == Some("my") && s.name() == "tag"
            ))),
            "merged root is a map carrying my/tag"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_arith_for_tags_genuine_fns_only() {
        use crate::ext::FastArith;
        assert_eq!(fast_arith_for(native_add), FastArith::Add);
        assert_eq!(fast_arith_for(native_sub), FastArith::Sub);
        assert_eq!(fast_arith_for(native_mul), FastArith::Mul);
        assert_eq!(fast_arith_for(native_lt), FastArith::Lt);
        assert_eq!(fast_arith_for(native_lte), FastArith::Lte);
        assert_eq!(fast_arith_for(native_gt), FastArith::Gt);
        assert_eq!(fast_arith_for(native_gte), FastArith::Gte);
        assert_eq!(fast_arith_for(native_inc), FastArith::Inc);
        assert_eq!(fast_arith_for(native_dec), FastArith::Dec);
        // Neighbors and lookalike fakes never tag.
        assert_eq!(fast_arith_for(native_zero), FastArith::None);
        fn fake(_eval: &mut Evaluator, _args: &[Value]) -> Result<Value, NativeError> {
            Ok(Value::Nil)
        }
        assert_eq!(fast_arith_for(fake), FastArith::None);
        // The fake itself runs (its body counts as covered code).
        let mut eval = Evaluator::new_bare("user");
        assert!(matches!(fake(&mut eval, &[]), Ok(Value::Nil)));
    }
}
