// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Shape-specialized direct calls for the register VM (#226).
//!
//! The general intrinsic call (see [`crate::vm_intrinsics`]) matches
//! each operand's kind at run time and reaches its inline answer
//! through a function pointer. The hottest one- and two-argument
//! `clojure.core` ops compile here per operand shape instead — a
//! register, a captured value, a constant, or a computed form — into
//! closures whose fast path is the op itself, inline. Every miss
//! (another argument type, an overflow, a zero divisor, an
//! invalidated site) leaves through one cold function that runs the
//! native over the same values, so results and errors stay the
//! native's.
//!
//! The op comes from the analyzed native's own name (its
//! `clojure.core/…` display), never the var's: a var re-rooted to
//! another native keeps that native's semantics. Operands read in
//! place (locals, constants) are read after the computed ones, like
//! the intrinsics: locals have no effects, and the last-use analysis
//! never moves them out ([`in_place`]).

use std::sync::Arc;

use clj_value::Value;
use smallvec::{SmallVec, smallvec};

use crate::error::Signal;
use crate::eval::Evaluator;
use crate::expr::DirectCall;
use crate::vm::{Code, Exec, Frame, Loc, Operand, direct_slow_values, local, next_site};

/// A two-argument op's inline answer (`None` runs the native).
pub(crate) trait Op2: Send + Sync + 'static {
    fn apply(a: &Value, b: &Value) -> Option<Value>;

    /// The answer's truth, for a branch test.
    #[inline(always)]
    fn test(a: &Value, b: &Value) -> Option<bool> {
        Self::apply(a, b).map(|v| v.is_truthy())
    }
}

/// A one-argument op's inline answer (`None` runs the native).
pub(crate) trait Op1: Send + Sync + 'static {
    fn apply(a: &Value) -> Option<Value>;

    /// The answer's truth, for a branch test.
    #[inline(always)]
    fn test(a: &Value) -> Option<bool> {
        Self::apply(a).map(|v| v.is_truthy())
    }
}

/// What a specialized call produces: a value ([`AsValue`]) or a
/// branch test's truth ([`AsTest`]).
pub(crate) trait Out: Send + Sync + 'static {
    type T: 'static;
    fn hit2<O: Op2>(a: &Value, b: &Value) -> Option<Self::T>;
    fn hit1<O: Op1>(a: &Value) -> Option<Self::T>;
    fn of(v: Value) -> Self::T;
    fn of_bool(b: bool) -> Self::T;
}

pub(crate) struct AsValue;

impl Out for AsValue {
    type T = Value;

    #[inline(always)]
    fn hit2<O: Op2>(a: &Value, b: &Value) -> Option<Value> {
        O::apply(a, b)
    }

    #[inline(always)]
    fn hit1<O: Op1>(a: &Value) -> Option<Value> {
        O::apply(a)
    }

    #[inline(always)]
    fn of(v: Value) -> Value {
        v
    }

    #[inline(always)]
    fn of_bool(b: bool) -> Value {
        Value::Bool(b)
    }
}

pub(crate) struct AsTest;

impl Out for AsTest {
    type T = bool;

    #[inline(always)]
    fn hit2<O: Op2>(a: &Value, b: &Value) -> Option<bool> {
        O::test(a, b)
    }

    #[inline(always)]
    fn hit1<O: Op1>(a: &Value) -> Option<bool> {
        O::test(a)
    }

    #[inline(always)]
    fn of(v: Value) -> bool {
        v.is_truthy()
    }

    #[inline(always)]
    fn of_bool(b: bool) -> bool {
        b
    }
}

/// An operand read in place.
trait Src: Send + Sync + 'static {
    fn read<'v>(&'v self, f: &'v Frame<'_>) -> &'v Value;
}

/// A register of the running frame.
struct Reg(usize);

/// A captured value of the running closure.
struct Up(usize);

/// A literal.
struct Kst(Value);

impl Src for Reg {
    #[inline(always)]
    fn read<'v>(&'v self, f: &'v Frame<'_>) -> &'v Value {
        &f.regs[self.0]
    }
}

impl Src for Up {
    #[inline(always)]
    fn read<'v>(&'v self, f: &'v Frame<'_>) -> &'v Value {
        &f.up[self.0]
    }
}

impl Src for Kst {
    #[inline(always)]
    fn read<'v>(&'v self, _: &'v Frame<'_>) -> &'v Value {
        &self.0
    }
}

/// An operand's shape (`letfn` group members compute: they are
/// never arithmetic operands in practice).
enum Shape {
    Reg(usize),
    Up(usize),
    Kst(Value),
    Exec(Exec),
}

fn shape(op: Operand) -> Shape {
    match op {
        Operand::Const(v) => Shape::Kst(v),
        Operand::Loc(Loc::Reg(r), _) => Shape::Reg(r),
        Operand::Loc(Loc::Up(u), _) => Shape::Up(u),
        Operand::Loc(loc, _) => Shape::Exec(local(loc)),
        Operand::Exec(e) => Shape::Exec(e),
    }
}

/// A specialized call site: its validation slot and record.
pub(crate) struct Site {
    sid: usize,
    call: Arc<DirectCall>,
}

impl Site {
    pub(crate) fn new(call: Arc<DirectCall>) -> Self {
        Self {
            sid: next_site(),
            call,
        }
    }
}

/// The miss: the native over the operands' values (re-validated:
/// an invalid site re-resolves its head, like the general call).
#[cold]
#[inline(never)]
fn native(ev: &mut Evaluator, c: &Site, mut values: SmallVec<[Value; 4]>) -> Result<Value, Signal> {
    if ev.site_direct_valid(c.sid, &c.call) {
        ev.call_direct_values(&c.call, &mut values)
    } else {
        direct_slow_values(ev, &c.call, values)
    }
}

#[cold]
#[inline(never)]
fn native2(ev: &mut Evaluator, c: &Site, a: &Value, b: &Value) -> Result<Value, Signal> {
    native(ev, c, smallvec![a.clone(), b.clone()])
}

#[cold]
#[inline(never)]
fn native1(ev: &mut Evaluator, c: &Site, a: &Value) -> Result<Value, Signal> {
    native(ev, c, smallvec![a.clone()])
}

/// Both operands in place.
fn ss<A: Src, B: Src, O: Op2, T: Out>(c: Site, a: A, b: B) -> Code<T::T> {
    Box::new(move |ev, f| {
        if ev.site_direct_valid(c.sid, &c.call) {
            if let Some(v) = T::hit2::<O>(a.read(f), b.read(f)) {
                return Ok(v);
            }
        }
        native2(ev, &c, a.read(f), b.read(f)).map(T::of)
    })
}

/// A computed first operand (validated before it runs).
fn xs<B: Src, O: Op2, T: Out>(c: Site, a: Exec, b: B) -> Code<T::T> {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let va = a(ev, f)?;
        if valid {
            if let Some(v) = T::hit2::<O>(&va, b.read(f)) {
                return Ok(v);
            }
        }
        native2(ev, &c, &va, b.read(f)).map(T::of)
    })
}

/// A computed second operand.
fn sx<A: Src, O: Op2, T: Out>(c: Site, a: A, b: Exec) -> Code<T::T> {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let vb = b(ev, f)?;
        if valid {
            if let Some(v) = T::hit2::<O>(a.read(f), &vb) {
                return Ok(v);
            }
        }
        native2(ev, &c, a.read(f), &vb).map(T::of)
    })
}

/// Both operands computed (in order).
fn xx<O: Op2, T: Out>(c: Site, a: Exec, b: Exec) -> Code<T::T> {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let va = a(ev, f)?;
        let vb = b(ev, f)?;
        if valid {
            if let Some(v) = T::hit2::<O>(&va, &vb) {
                return Ok(v);
            }
        }
        native2(ev, &c, &va, &vb).map(T::of)
    })
}

/// The one operand in place.
fn s1<A: Src, O: Op1, T: Out>(c: Site, a: A) -> Code<T::T> {
    Box::new(move |ev, f| {
        if ev.site_direct_valid(c.sid, &c.call) {
            if let Some(v) = T::hit1::<O>(a.read(f)) {
                return Ok(v);
            }
        }
        native1(ev, &c, a.read(f)).map(T::of)
    })
}

/// The one operand computed.
fn x1<O: Op1, T: Out>(c: Site, a: Exec) -> Code<T::T> {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let va = a(ev, f)?;
        if valid {
            if let Some(v) = T::hit1::<O>(&va) {
                return Ok(v);
            }
        }
        native1(ev, &c, &va).map(T::of)
    })
}

fn shape2<O: Op2, T: Out>(c: Site, a: Operand, b: Operand) -> Code<T::T> {
    match shape(a) {
        Shape::Reg(r) => with_first::<Reg, O, T>(c, Reg(r), b),
        Shape::Up(u) => with_first::<Up, O, T>(c, Up(u), b),
        Shape::Kst(v) => with_first::<Kst, O, T>(c, Kst(v), b),
        Shape::Exec(e) => with_computed::<O, T>(c, e, b),
    }
}

fn with_first<A: Src, O: Op2, T: Out>(c: Site, a: A, b: Operand) -> Code<T::T> {
    match shape(b) {
        Shape::Reg(r) => ss::<A, Reg, O, T>(c, a, Reg(r)),
        Shape::Up(u) => ss::<A, Up, O, T>(c, a, Up(u)),
        Shape::Kst(v) => ss::<A, Kst, O, T>(c, a, Kst(v)),
        Shape::Exec(e) => sx::<A, O, T>(c, a, e),
    }
}

fn with_computed<O: Op2, T: Out>(c: Site, a: Exec, b: Operand) -> Code<T::T> {
    match shape(b) {
        Shape::Reg(r) => xs::<Reg, O, T>(c, a, Reg(r)),
        Shape::Up(u) => xs::<Up, O, T>(c, a, Up(u)),
        Shape::Kst(v) => xs::<Kst, O, T>(c, a, Kst(v)),
        Shape::Exec(e) => xx::<O, T>(c, a, e),
    }
}

fn shape1<O: Op1, T: Out>(c: Site, a: Operand) -> Code<T::T> {
    match shape(a) {
        Shape::Reg(r) => s1::<Reg, O, T>(c, Reg(r)),
        Shape::Up(u) => s1::<Up, O, T>(c, Up(u)),
        Shape::Kst(v) => s1::<Kst, O, T>(c, Kst(v)),
        Shape::Exec(e) => x1::<O, T>(c, e),
    }
}

/// An n-ary call's computed operands, evaluated in source order
/// into `out` (filled in place: a returned buffer would copy its
/// inline storage per call).
fn computed(
    ops: &[Operand],
    ev: &mut Evaluator,
    f: &mut Frame<'_>,
    out: &mut SmallVec<[Value; 4]>,
) -> Result<(), Signal> {
    for op in ops {
        if let Operand::Exec(e) = op {
            out.push(e(ev, f)?);
        }
    }
    Ok(())
}

/// Every operand in order: locals and literals in place, computed
/// ones from `comp`.
fn operand_refs<'v>(
    ops: &'v [Operand],
    f: &'v Frame<'_>,
    comp: &'v [Value],
) -> SmallVec<[&'v Value; 6]> {
    let mut comp = comp.iter();
    ops.iter()
        .map(|op| match op {
            Operand::Const(v) => v,
            Operand::Loc(loc, _) => f.get(*loc),
            Operand::Exec(_) => comp.next().expect("one value per computed operand"),
        })
        .collect()
}

/// A left fold of a binary op (`(+ a b c)` is `(+ (+ a b) c)`, every
/// step checked like the native's).
#[inline(always)]
fn fold<O: Op2>(args: &[&Value]) -> Option<Value> {
    let mut acc = O::apply(args[0], args[1])?;
    for a in &args[2..] {
        acc = O::apply(&acc, a)?;
    }
    Some(acc)
}

/// A comparison chain (`(< a b c)`: every adjacent pair holds; the
/// first failing pair answers, like the native, before any later
/// operand is checked).
#[inline(always)]
fn chain<O: Op2>(args: &[&Value]) -> Option<bool> {
    for pair in args.windows(2) {
        if !O::test(pair[0], pair[1])? {
            return Some(false);
        }
    }
    Some(true)
}

/// An n-ary op's code: validated once, computed operands first,
/// then `step` over every operand in place; a miss runs the native
/// over all the values (one call, like the general road).
fn nary<T: Out>(c: Site, ops: Vec<Operand>, step: fn(&[&Value]) -> Option<T::T>) -> Code<T::T> {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let mut comp: SmallVec<[Value; 4]> = SmallVec::new();
        computed(&ops, ev, f, &mut comp)?;
        let args = operand_refs(&ops, f, &comp);
        if valid {
            if let Some(v) = step(&args) {
                return Ok(v);
            }
        }
        let values: SmallVec<[Value; 4]> = args.iter().map(|v| (*v).clone()).collect();
        native(ev, &c, values).map(T::of)
    })
}

fn fold_n<O: Op2, T: Out>(c: Site, ops: Vec<Operand>) -> Code<T::T> {
    nary::<T>(c, ops, |args| fold::<O>(args).map(T::of))
}

fn chain_n<O: Op2, T: Out>(c: Site, ops: Vec<Operand>) -> Code<T::T> {
    nary::<T>(c, ops, |args| chain::<O>(args).map(T::of_bool))
}

type CtorN<T> = fn(Site, Vec<Operand>) -> Code<T>;

/// An op table row: the core name, the value code, and the test
/// code (predicates only) — by operand count.
type RowN = (&'static str, CtorN<Value>, Option<CtorN<bool>>);

/// The ops with a 3+-argument form by core name: folds of their
/// binary op, and comparison chains (value and test code).
const NARY: &[RowN] = &[
    ("+", fold_n::<Add, AsValue>, None),
    ("-", fold_n::<Sub, AsValue>, None),
    ("*", fold_n::<Mul, AsValue>, None),
    ("+'", fold_n::<Add, AsValue>, None),
    ("-'", fold_n::<Sub, AsValue>, None),
    ("*'", fold_n::<Mul, AsValue>, None),
    ("min", fold_n::<Min, AsValue>, None),
    ("max", fold_n::<Max, AsValue>, None),
    ("bit-and", fold_n::<BitAnd, AsValue>, None),
    ("bit-or", fold_n::<BitOr, AsValue>, None),
    ("bit-xor", fold_n::<BitXor, AsValue>, None),
    ("<", chain_n::<Lt, AsValue>, Some(chain_n::<Lt, AsTest>)),
    ("<=", chain_n::<Lte, AsValue>, Some(chain_n::<Lte, AsTest>)),
    (">", chain_n::<Gt, AsValue>, Some(chain_n::<Gt, AsTest>)),
    (">=", chain_n::<Gte, AsValue>, Some(chain_n::<Gte, AsTest>)),
    (
        "==",
        chain_n::<NumEq, AsValue>,
        Some(chain_n::<NumEq, AsTest>),
    ),
];

type Ctor2<T> = fn(Site, Operand, Operand) -> Code<T>;
type Ctor1<T> = fn(Site, Operand) -> Code<T>;

/// See [`RowN`].
type Row2 = (&'static str, Ctor2<Value>, Option<Ctor2<bool>>);

/// See [`RowN`].
type Row1 = (&'static str, Ctor1<Value>, Option<Ctor1<bool>>);

/// The two-argument ops by core name: the value code, and the
/// test code for the predicates.
const BINARY: &[Row2] = &[
    ("+", shape2::<Add, AsValue>, None),
    ("-", shape2::<Sub, AsValue>, None),
    ("*", shape2::<Mul, AsValue>, None),
    // The promoting ops answer exactly like `+`/`-`/`*` until a long
    // overflows — where the inline op misses and the call's own
    // native promotes.
    ("+'", shape2::<Add, AsValue>, None),
    ("-'", shape2::<Sub, AsValue>, None),
    ("*'", shape2::<Mul, AsValue>, None),
    ("<", shape2::<Lt, AsValue>, Some(shape2::<Lt, AsTest>)),
    ("<=", shape2::<Lte, AsValue>, Some(shape2::<Lte, AsTest>)),
    (">", shape2::<Gt, AsValue>, Some(shape2::<Gt, AsTest>)),
    (">=", shape2::<Gte, AsValue>, Some(shape2::<Gte, AsTest>)),
    (
        "==",
        shape2::<NumEq, AsValue>,
        Some(shape2::<NumEq, AsTest>),
    ),
    ("=", shape2::<Eq, AsValue>, Some(shape2::<Eq, AsTest>)),
    (
        "not=",
        shape2::<NotEq, AsValue>,
        Some(shape2::<NotEq, AsTest>),
    ),
    ("quot", shape2::<Quot, AsValue>, None),
    ("rem", shape2::<Rem, AsValue>, None),
    ("mod", shape2::<Mod, AsValue>, None),
    ("bit-and", shape2::<BitAnd, AsValue>, None),
    ("bit-or", shape2::<BitOr, AsValue>, None),
    ("bit-xor", shape2::<BitXor, AsValue>, None),
    ("bit-shift-left", shape2::<Shl, AsValue>, None),
    ("bit-shift-right", shape2::<Shr, AsValue>, None),
    ("unsigned-bit-shift-right", shape2::<Ushr, AsValue>, None),
    ("min", shape2::<Min, AsValue>, None),
    ("max", shape2::<Max, AsValue>, None),
    ("nth", shape2::<Nth, AsValue>, None),
    ("get", shape2::<Get, AsValue>, None),
    (
        "contains?",
        shape2::<Contains, AsValue>,
        Some(shape2::<Contains, AsTest>),
    ),
    ("conj!", shape2::<ConjBang, AsValue>, None),
];

/// The one-argument ops by core name (see [`BINARY`]).
const UNARY: &[Row1] = &[
    ("inc", shape1::<Inc, AsValue>, None),
    ("-", shape1::<Negate, AsValue>, None),
    ("dec", shape1::<Dec, AsValue>, None),
    ("inc'", shape1::<Inc, AsValue>, None),
    ("-'", shape1::<Negate, AsValue>, None),
    ("dec'", shape1::<Dec, AsValue>, None),
    (
        "zero?",
        shape1::<Zero, AsValue>,
        Some(shape1::<Zero, AsTest>),
    ),
    ("pos?", shape1::<Pos, AsValue>, Some(shape1::<Pos, AsTest>)),
    ("neg?", shape1::<Neg, AsValue>, Some(shape1::<Neg, AsTest>)),
    (
        "even?",
        shape1::<Even, AsValue>,
        Some(shape1::<Even, AsTest>),
    ),
    ("odd?", shape1::<Odd, AsValue>, Some(shape1::<Odd, AsTest>)),
    ("not", shape1::<Not, AsValue>, Some(shape1::<Not, AsTest>)),
    (
        "nil?",
        shape1::<IsNil, AsValue>,
        Some(shape1::<IsNil, AsTest>),
    ),
    (
        "some?",
        shape1::<IsSome, AsValue>,
        Some(shape1::<IsSome, AsTest>),
    ),
    ("count", shape1::<Count, AsValue>, None),
    (
        "first",
        shape1::<First, AsValue>,
        Some(shape1::<First, AsTest>),
    ),
    (
        "seq",
        shape1::<SeqOp, AsValue>,
        Some(shape1::<SeqOp, AsTest>),
    ),
    (
        "empty?",
        shape1::<EmptyP, AsValue>,
        Some(shape1::<EmptyP, AsTest>),
    ),
    ("int", shape1::<IntCast, AsValue>, None),
    ("long", shape1::<LongCast, AsValue>, None),
    ("double", shape1::<DoubleCast, AsValue>, None),
];

/// The analyzed native's `clojure.core` name (its display).
pub(crate) fn core_name(call: &DirectCall) -> Option<&str> {
    let native = call
        .expect
        .as_any()
        .downcast_ref::<crate::ext::NativeFn>()?;
    native.display.strip_prefix("clojure.core/")
}

/// A specialized op's code constructor (by operand count).
#[derive(Clone, Copy)]
pub(crate) enum Ctor<T: 'static> {
    One(Ctor1<T>),
    Two(Ctor2<T>),
    N(CtorN<T>),
}

impl<T: 'static> Ctor<T> {
    /// The code over the call's operands (one per argument).
    pub(crate) fn code(self, site: Arc<DirectCall>, ops: Vec<Operand>) -> Code<T> {
        let site = Site::new(site);
        let mut ops = ops.into_iter();
        let mut next = || ops.next().expect("one operand per argument");
        match self {
            Ctor::One(c) => c(site, next()),
            Ctor::Two(c) => {
                let a = next();
                c(site, a, next())
            }
            Ctor::N(c) => c(site, ops.collect()),
        }
    }
}

/// The specialized value code for `call`, if its op has one. Its
/// operands are then read in place, after the computed ones — the
/// last-use analysis must not move those locals.
pub(crate) fn value_ctor(call: &DirectCall) -> Option<Ctor<Value>> {
    let name = core_name(call)?;
    match call.args.len() {
        1 => UNARY
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, v, _)| Ctor::One(*v)),
        2 => BINARY
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, v, _)| Ctor::Two(*v)),
        3.. => NARY
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, v, _)| Ctor::N(*v)),
        _ => None,
    }
}

/// The specialized test code for `call`, if its op is a predicate.
pub(crate) fn test_ctor(call: &DirectCall) -> Option<Ctor<bool>> {
    let name = core_name(call)?;
    match call.args.len() {
        1 => UNARY.iter().find(|(n, ..)| *n == name)?.2.map(Ctor::One),
        2 => BINARY.iter().find(|(n, ..)| *n == name)?.2.map(Ctor::Two),
        3.. => NARY.iter().find(|(n, ..)| *n == name)?.2.map(Ctor::N),
        _ => None,
    }
}

/// Whether `call` compiles to specialized code (see [`value_ctor`]).
pub(crate) fn in_place(call: &DirectCall) -> bool {
    value_ctor(call).is_some()
}

/// An update op's inline answer over its owned operands (the
/// collection first): `None` leaves them in place for the native.
pub(crate) trait Upd: Send + Sync + 'static {
    fn apply(args: &mut [Value]) -> Option<Value>;
}

/// Where an update op's collection comes from.
pub(crate) enum Coll {
    /// A last-use register: moves out (unique storage updates in
    /// place).
    Take(usize),
    /// A computed form (runs first, in source order).
    Exec(Exec),
    /// Any other local or literal (cloned).
    Read(Operand),
}

/// An update op (`assoc`, `conj`): a computed collection runs
/// first, then the other operands in order, then a local
/// collection is read — last, so the last-use analysis can move it
/// even when the other operands read it ([`update_order`]).
fn upd<O: Upd>(c: Site, coll: Coll, rest: Vec<Operand>) -> Exec {
    if rest.is_empty() {
        return upd1::<O>(c, coll);
    }
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let mut args: SmallVec<[Value; 4]> = SmallVec::new();
        args.push(match &coll {
            Coll::Exec(e) => e(ev, f)?,
            _ => Value::Nil,
        });
        for o in &rest {
            args.push(o.eval(ev, f)?);
        }
        match &coll {
            Coll::Take(r) => args[0] = std::mem::replace(&mut f.regs[*r], Value::Nil),
            Coll::Read(o) => args[0] = o.eval(ev, f)?,
            Coll::Exec(_) => {}
        }
        if valid {
            if let Some(v) = O::apply(&mut args) {
                return Ok(v);
            }
        }
        native(ev, &c, args)
    })
}

/// A one-operand update op (`next`, `rest`): the collection alone,
/// with no argument buffer unless the native runs.
fn upd1<O: Upd>(c: Site, coll: Coll) -> Exec {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let mut a = match &coll {
            Coll::Exec(e) => e(ev, f)?,
            Coll::Take(r) => std::mem::replace(&mut f.regs[*r], Value::Nil),
            Coll::Read(o) => o.eval(ev, f)?,
        };
        if valid {
            if let Some(v) = O::apply(std::slice::from_mut(&mut a)) {
                return Ok(v);
            }
        }
        native(ev, &c, smallvec![a])
    })
}

/// The update ops by core name and arg count.
/// An update-op table row: core name, arg count, code constructor.
type UpdRow = (&'static str, usize, UpdCtor);

const UPDATES: &[UpdRow] = &[
    ("assoc", 3, upd::<Assoc>),
    ("conj", 2, upd::<Conj>),
    ("next", 1, upd::<Next>),
    ("rest", 1, upd::<Rest>),
];

/// An update op's code constructor (see [`upd`]).
pub(crate) type UpdCtor = fn(Site, Coll, Vec<Operand>) -> Exec;

/// The update op for `call`, if it is one: its collection operand
/// is then read after the others (see [`upd`]).
pub(crate) fn update_ctor(call: &DirectCall) -> Option<UpdCtor> {
    let name = core_name(call)?;
    UPDATES
        .iter()
        .find(|(n, argc, _)| *n == name && *argc == call.args.len())
        .map(|(.., ctor)| *ctor)
}

/// Whether `call` compiles as an update op (see [`update_ctor`]).
pub(crate) fn update_order(call: &DirectCall) -> bool {
    update_ctor(call).is_some()
}

/// An update op's code over its compiled operands (the collection
/// operand a last-use register moves out).
pub(crate) fn update_code(ctor: UpdCtor, site: Arc<DirectCall>, mut ops: Vec<Operand>) -> Exec {
    let rest = ops.split_off(1);
    let coll = match ops.pop().expect("the collection operand") {
        Operand::Loc(Loc::Reg(r), true) => Coll::Take(r),
        Operand::Exec(e) => Coll::Exec(e),
        other => Coll::Read(other),
    };
    ctor(Site::new(site), coll, rest)
}

/// Move a value out of its slot (leaving `nil`).
#[inline(always)]
fn take(slot: &mut Value) -> Value {
    std::mem::replace(slot, Value::Nil)
}

/// `(assoc coll k v)`: vectors at an `int` index up to their count,
/// unsorted maps at scalar keys.
pub(crate) struct Assoc;

impl Upd for Assoc {
    #[inline(always)]
    fn apply(args: &mut [Value]) -> Option<Value> {
        let [coll, k, v] = args else { return None };
        match (std::mem::replace(coll, Value::Nil), &*k) {
            (Value::Vector(x), Value::Int(i)) => match usize::try_from(*i as i32) {
                Ok(n) if n < x.len() => x.set_nth_owned(n, take(v)).map(Value::Vector),
                Ok(n) if n == x.len() => Some(Value::Vector(x.push_back(take(v)))),
                _ => {
                    *coll = Value::Vector(x);
                    None
                }
            },
            (Value::Map(m), k) if !m.is_sorted() && crate::vm_intrinsics::is_scalar_key(k) => {
                Some(Value::Map(m.insert(k.clone(), take(v))))
            }
            (other, _) => {
                *coll = other;
                None
            }
        }
    }
}

/// `(conj coll x)`: vectors, lists, `nil`, and unsorted sets of
/// scalars.
pub(crate) struct Conj;

impl Upd for Conj {
    #[inline(always)]
    fn apply(args: &mut [Value]) -> Option<Value> {
        let [coll, x] = args else { return None };
        match std::mem::replace(coll, Value::Nil) {
            Value::Vector(v) => Some(Value::Vector(v.push_back(take(x)))),
            Value::List(l) => Some(Value::List(l.push_front(take(x)))),
            Value::Nil => Some(Value::List(clj_value::List::from_vec(vec![take(x)]))),
            Value::Set(s) if !s.is_sorted() && crate::vm_intrinsics::is_scalar_key(x) => {
                Some(Value::Set(s.insert(take(x))))
            }
            other => {
                *coll = other;
                None
            }
        }
    }
}

/// `+`/`-`/`*` over numbers the inline path covers: checked `Int`
/// pairs, and `Double` with either side (`Int` widened, like the
/// native's `to_f64`).
#[inline(always)]
fn arith(
    a: &Value,
    b: &Value,
    int: fn(i64, i64) -> Option<i64>,
    dbl: fn(f64, f64) -> f64,
) -> Option<Value> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => int(*x, *y).map(Value::Int),
        (Value::Double(x), Value::Double(y)) => Some(Value::Double(dbl(*x, *y))),
        (Value::Int(x), Value::Double(y)) => Some(Value::Double(dbl(*x as f64, *y))),
        (Value::Double(x), Value::Int(y)) => Some(Value::Double(dbl(*x, *y as f64))),
        _ => None,
    }
}

/// A numeric comparison (`Int` pairs exact, any `Double` as `f64`).
#[inline(always)]
fn compare(
    a: &Value,
    b: &Value,
    int: fn(i64, i64) -> bool,
    dbl: fn(f64, f64) -> bool,
) -> Option<bool> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(int(*x, *y)),
        (Value::Double(x), Value::Double(y)) => Some(dbl(*x, *y)),
        (Value::Int(x), Value::Double(y)) => Some(dbl(*x as f64, *y)),
        (Value::Double(x), Value::Int(y)) => Some(dbl(*x, *y as f64)),
        _ => None,
    }
}

/// An `Int`-pair op (`None` for any other pair).
#[inline(always)]
fn ints(a: &Value, b: &Value, op: fn(i64, i64) -> Option<i64>) -> Option<Value> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => op(*x, *y).map(Value::Int),
        _ => None,
    }
}

/// A numeric predicate over `Int` and `Double`.
#[inline(always)]
fn sign(a: &Value, int: fn(i64) -> bool, dbl: fn(f64) -> bool) -> Option<bool> {
    match a {
        Value::Int(x) => Some(int(*x)),
        Value::Double(x) => Some(dbl(*x)),
        _ => None,
    }
}

macro_rules! op2 {
    ($name:ident, |$a:ident, $b:ident| $body:expr) => {
        pub(crate) struct $name;

        impl Op2 for $name {
            #[inline(always)]
            fn apply($a: &Value, $b: &Value) -> Option<Value> {
                $body
            }
        }
    };
}

macro_rules! cmp2 {
    ($name:ident, |$a:ident, $b:ident| $body:expr) => {
        pub(crate) struct $name;

        impl Op2 for $name {
            #[inline(always)]
            fn apply($a: &Value, $b: &Value) -> Option<Value> {
                Self::test($a, $b).map(Value::Bool)
            }

            #[inline(always)]
            fn test($a: &Value, $b: &Value) -> Option<bool> {
                $body
            }
        }
    };
}

macro_rules! pred1 {
    ($name:ident, |$a:ident| $body:expr) => {
        pub(crate) struct $name;

        impl Op1 for $name {
            #[inline(always)]
            fn apply($a: &Value) -> Option<Value> {
                Self::test($a).map(Value::Bool)
            }

            #[inline(always)]
            fn test($a: &Value) -> Option<bool> {
                $body
            }
        }
    };
}

op2!(Add, |a, b| arith(a, b, i64::checked_add, |x, y| x + y));
op2!(Sub, |a, b| arith(a, b, i64::checked_sub, |x, y| x - y));
op2!(Mul, |a, b| arith(a, b, i64::checked_mul, |x, y| x * y));
cmp2!(Lt, |a, b| compare(a, b, |x, y| x < y, |x, y| x < y));
cmp2!(Lte, |a, b| compare(a, b, |x, y| x <= y, |x, y| x <= y));
cmp2!(Gt, |a, b| compare(a, b, |x, y| x > y, |x, y| x > y));
cmp2!(Gte, |a, b| compare(a, b, |x, y| x >= y, |x, y| x >= y));
cmp2!(NumEq, |a, b| compare(a, b, |x, y| x == y, |x, y| x == y));
cmp2!(Eq, |a, b| crate::vm_intrinsics::scalar_eq(a, b));
cmp2!(NotEq, |a, b| crate::vm_intrinsics::scalar_eq(a, b)
    .map(|e| !e));
cmp2!(Contains, |a, b| crate::vm_intrinsics::contains_test(a, b));
// `quot`/`rem` are raw Java `/`/`%` (`MIN/-1` wraps); `mod` adjusts
// to the divisor's sign. A zero divisor runs the native's error.
op2!(Quot, |a, b| ints(a, b, |x, y| (y != 0)
    .then(|| x.wrapping_div(y))));
op2!(Rem, |a, b| ints(a, b, |x, y| (y != 0)
    .then(|| x.wrapping_rem(y))));
op2!(Mod, |a, b| ints(a, b, crate::vm_intrinsics::java_mod));
op2!(BitAnd, |a, b| ints(a, b, |x, y| Some(x & y)));
op2!(BitOr, |a, b| ints(a, b, |x, y| Some(x | y)));
op2!(BitXor, |a, b| ints(a, b, |x, y| Some(x ^ y)));
// Shift counts mask to 6 bits like Java's (the natives'
// `wrapping_*` over the count's low 32 bits).
op2!(Shl, |a, b| ints(a, b, |x, n| Some(
    x.wrapping_shl(n as u32)
)));
op2!(Shr, |a, b| ints(a, b, |x, n| Some(
    x.wrapping_shr(n as u32)
)));
op2!(Ushr, |a, b| ints(a, b, |x, n| Some(
    (x as u64).wrapping_shr(n as u32) as i64
)));
op2!(Min, |a, b| ints(a, b, |x, y| Some(x.min(y))));
op2!(Max, |a, b| ints(a, b, |x, y| Some(x.max(y))));
op2!(Nth, |a, b| crate::vm_intrinsics::nth_of(a, b, None));
op2!(Get, |a, b| crate::vm_intrinsics::get_of(a, b, None));

pub(crate) struct Inc;

impl Op1 for Inc {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Int(x) => x.checked_add(1).map(Value::Int),
            Value::Double(x) => Some(Value::Double(x + 1.0)),
            _ => None,
        }
    }
}

/// `(- x)`: a checked `Int` negation (`MIN` runs the native's
/// overflow), or a `Double`'s.
pub(crate) struct Negate;

impl Op1 for Negate {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Int(x) => x.checked_neg().map(Value::Int),
            Value::Double(x) => Some(Value::Double(-x)),
            _ => None,
        }
    }
}

pub(crate) struct Dec;

impl Op1 for Dec {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Int(x) => x.checked_sub(1).map(Value::Int),
            Value::Double(x) => Some(Value::Double(x - 1.0)),
            _ => None,
        }
    }
}

pub(crate) struct Count;

impl Op1 for Count {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        crate::vm_intrinsics::count_of(a)
    }
}

/// `(first coll)` over nil, lists, vectors, strings and realized
/// seqs ([`crate::lazy::first_fast`]).
pub(crate) struct First;

impl Op1 for First {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        crate::lazy::first_fast(a)
    }
}

/// `(seq coll)` over nil, lists, vectors, strings and realized seqs;
/// as a test, its truth without building the seq.
pub(crate) struct SeqOp;

impl Op1 for SeqOp {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        crate::lazy::seq_fast(a)
    }

    #[inline(always)]
    fn test(a: &Value) -> Option<bool> {
        crate::lazy::empty_fast(a).map(|e| !e)
    }
}

/// `(empty? coll)` over nil, the counted collections, strings and
/// realized seqs.
pub(crate) struct EmptyP;

impl Op1 for EmptyP {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        crate::lazy::empty_fast(a).map(Value::Bool)
    }

    #[inline(always)]
    fn test(a: &Value) -> Option<bool> {
        crate::lazy::empty_fast(a)
    }
}

/// `(next coll)` over an owned operand ([`crate::lazy::next_owned`]):
/// a last-use seq steps in place.
pub(crate) struct Next;

impl Upd for Next {
    #[inline(always)]
    fn apply(args: &mut [Value]) -> Option<Value> {
        let [coll] = args else { return None };
        crate::lazy::next_owned(coll, false)
    }
}

/// `(rest coll)`, like [`Next`] but `()` at the end.
pub(crate) struct Rest;

impl Upd for Rest {
    #[inline(always)]
    fn apply(args: &mut [Value]) -> Option<Value> {
        let [coll] = args else { return None };
        crate::lazy::next_owned(coll, true)
    }
}

/// `(int x)`: an `Int` in `int` range or a character (the native
/// keeps longs, checking the range).
pub(crate) struct IntCast;

impl Op1 for IntCast {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Int(x) if i32::try_from(*x).is_ok() => Some(Value::Int(*x)),
            Value::Char(c) => Some(Value::Int(i64::from(u32::from(*c)))),
            _ => None,
        }
    }
}

/// `(long x)` of an `Int` or a character.
pub(crate) struct LongCast;

impl Op1 for LongCast {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Int(x) => Some(Value::Int(*x)),
            Value::Char(c) => Some(Value::Int(i64::from(u32::from(*c)))),
            _ => None,
        }
    }
}

/// `(double x)` of an `Int` or a `Double`.
pub(crate) struct DoubleCast;

impl Op1 for DoubleCast {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Int(x) => Some(Value::Double(*x as f64)),
            Value::Double(x) => Some(Value::Double(*x)),
            _ => None,
        }
    }
}

/// `(conj! tv x)` onto a live transient vector (the cell updates in
/// place; anything else — or a frozen transient — runs the native).
pub(crate) struct ConjBang;

impl Op2 for ConjBang {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        let Value::Transient(t) = a else {
            return None;
        };
        let mut guard = t.lock();
        let clj_value::TransientKind::Vector(tv) = &mut *guard else {
            return None;
        };
        tv.conj(b.clone()).ok()?;
        drop(guard);
        Some(a.clone())
    }
}

pred1!(Zero, |a| sign(a, |x| x == 0, |x| x == 0.0));
pred1!(Pos, |a| sign(a, |x| x > 0, |x| x > 0.0));
pred1!(Neg, |a| sign(a, |x| x < 0, |x| x < 0.0));
pred1!(Even, |a| match a {
    Value::Int(x) => Some(x % 2 == 0),
    _ => None,
});
pred1!(Odd, |a| match a {
    Value::Int(x) => Some(x % 2 != 0),
    _ => None,
});
pred1!(Not, |a| Some(!a.is_truthy()));
pred1!(IsNil, |a| Some(matches!(a, Value::Nil)));
pred1!(IsSome, |a| Some(!matches!(a, Value::Nil)));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_predicates_answer_their_truth_by_default() {
        // Only predicates compile as branch tests (each overrides
        // `test`); the defaults still answer a value's truth.
        assert_eq!(
            <Add as Op2>::test(&Value::Int(1), &Value::Int(2)),
            Some(true)
        );
        assert_eq!(<Get as Op2>::test(&Value::Nil, &Value::Int(0)), Some(false));
        assert_eq!(<Inc as Op1>::test(&Value::Int(1)), Some(true));
        assert_eq!(<Count as Op1>::test(&Value::Bool(true)), None);
    }
}
