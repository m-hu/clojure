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

/// A three-argument op's inline answer (`None` runs the native).
pub(crate) trait Op3: Send + Sync + 'static {
    fn apply(a: &Value, b: &Value, c: &Value) -> Option<Value>;
}

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

    /// Whether the site still resolves to its analyzed native
    /// (the same verdict the shape closures check per call).
    pub(crate) fn valid(&self, ev: &mut Evaluator) -> bool {
        ev.site_direct_valid(self.sid, &self.call)
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
pub(crate) fn native3(
    ev: &mut Evaluator,
    c: &Site,
    a: &Value,
    b: &Value,
    d: &Value,
) -> Result<Value, Signal> {
    native(ev, c, smallvec![a.clone(), b.clone(), d.clone()])
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

fn unshape(s: Shape) -> Operand {
    match s {
        Shape::Reg(r) => Operand::Loc(Loc::Reg(r), false),
        Shape::Up(u) => Operand::Loc(Loc::Up(u), false),
        Shape::Kst(v) => Operand::Const(v),
        Shape::Exec(e) => Operand::Exec(e),
    }
}

fn sss<A: Src, B: Src, C: Src, O: Op3>(c: Site, a: A, b: B, d: C) -> Exec {
    Box::new(move |ev, f| {
        if ev.site_direct_valid(c.sid, &c.call) {
            if let Some(v) = O::apply(a.read(f), b.read(f), d.read(f)) {
                return Ok(v);
            }
        }
        native3(ev, &c, a.read(f), b.read(f), d.read(f))
    })
}

fn sxs<A: Src, C: Src, O: Op3>(c: Site, a: A, b: Exec, d: C) -> Exec {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let vb = b(ev, f)?;
        if valid {
            if let Some(v) = O::apply(a.read(f), &vb, d.read(f)) {
                return Ok(v);
            }
        }
        native3(ev, &c, a.read(f), &vb, d.read(f))
    })
}

fn ssx<A: Src, B: Src, O: Op3>(c: Site, a: A, b: B, d: Exec) -> Exec {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let vd = d(ev, f)?;
        if valid {
            if let Some(v) = O::apply(a.read(f), b.read(f), &vd) {
                return Ok(v);
            }
        }
        native3(ev, &c, a.read(f), b.read(f), &vd)
    })
}

fn any3<O: Op3>(c: Site, a: Operand, b: Operand, d: Operand) -> Exec {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let ca = a.computed(ev, f)?;
        let cb = b.computed(ev, f)?;
        let cd = d.computed(ev, f)?;
        let va = a.read(f, ca.as_ref());
        let vb = b.read(f, cb.as_ref());
        let vd = d.read(f, cd.as_ref());
        if valid {
            if let Some(v) = O::apply(va, vb, vd) {
                return Ok(v);
            }
        }
        native3(ev, &c, va, vb, vd)
    })
}

fn shape3<O: Op3>(c: Site, a: Operand, b: Operand, d: Operand) -> Exec {
    match (shape(a), shape(b), shape(d)) {
        (Shape::Reg(ra), Shape::Reg(rb), Shape::Reg(rc)) => {
            sss::<Reg, Reg, Reg, O>(c, Reg(ra), Reg(rb), Reg(rc))
        }
        (Shape::Reg(ra), Shape::Reg(rb), Shape::Kst(rc)) => {
            sss::<Reg, Reg, Kst, O>(c, Reg(ra), Reg(rb), Kst(rc))
        }
        (Shape::Reg(ra), Shape::Kst(rb), Shape::Reg(rc)) => {
            sss::<Reg, Kst, Reg, O>(c, Reg(ra), Kst(rb), Reg(rc))
        }
        (Shape::Reg(ra), Shape::Kst(rb), Shape::Kst(rc)) => {
            sss::<Reg, Kst, Kst, O>(c, Reg(ra), Kst(rb), Kst(rc))
        }
        (Shape::Reg(ra), Shape::Exec(eb), Shape::Kst(rc)) => {
            sxs::<Reg, Kst, O>(c, Reg(ra), eb, Kst(rc))
        }
        (Shape::Reg(ra), Shape::Exec(eb), Shape::Reg(rc)) => {
            sxs::<Reg, Reg, O>(c, Reg(ra), eb, Reg(rc))
        }
        (Shape::Reg(ra), Shape::Reg(rb), Shape::Exec(ed)) => {
            ssx::<Reg, Reg, O>(c, Reg(ra), Reg(rb), ed)
        }
        (sa, sb, sd) => any3::<O>(c, unshape(sa), unshape(sb), unshape(sd)),
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

type Ctor3<T> = fn(Site, Operand, Operand, Operand) -> Code<T>;
type Ctor2<T> = fn(Site, Operand, Operand) -> Code<T>;
type Ctor1<T> = fn(Site, Operand) -> Code<T>;

/// See [`RowN`].
type Row3 = (&'static str, Ctor3<Value>);

/// See [`RowN`].
type Row2 = (&'static str, Ctor2<Value>, Option<Ctor2<bool>>);

/// See [`RowN`].
type Row1 = (&'static str, Ctor1<Value>, Option<Ctor1<bool>>);

/// A one-argument inline answer over a borrowed arg.
pub(crate) type Apply1 = fn(&Value) -> Option<Value>;
/// A two-argument inline answer over borrowed args.
pub(crate) type Apply2 = fn(&Value, &Value) -> Option<Value>;
/// A three-argument inline answer over borrowed args.
pub(crate) type Apply3 = fn(&Value, &Value, &Value) -> Option<Value>;

/// The same inline ops over borrowed arguments, for natively compiled
/// code (issue #259): by core name per arity, `None` runs the native.
/// Kept in step with [`UNARY`]/[`BINARY`]/[`TERNARY`] (pinned by
/// `apply_tables_cover_every_specialized_op`).
pub(crate) const APPLY1: &[(&str, Apply1)] = &[
    ("inc", <Inc as Op1>::apply),
    ("-", <Negate as Op1>::apply),
    ("dec", <Dec as Op1>::apply),
    ("inc'", <Inc as Op1>::apply),
    ("-'", <Negate as Op1>::apply),
    ("dec'", <Dec as Op1>::apply),
    ("zero?", <Zero as Op1>::apply),
    ("pos?", <Pos as Op1>::apply),
    ("neg?", <Neg as Op1>::apply),
    ("even?", <Even as Op1>::apply),
    ("odd?", <Odd as Op1>::apply),
    ("not", <Not as Op1>::apply),
    ("nil?", <IsNil as Op1>::apply),
    ("some?", <IsSome as Op1>::apply),
    ("string?", <IsString as Op1>::apply),
    ("parse-long", <ParseLong as Op1>::apply),
    ("count", <Count as Op1>::apply),
    ("peek", <Peek as Op1>::apply),
    ("pop!", <PopBang as Op1>::apply),
    ("bit-not", <BitNot as Op1>::apply),
    ("first", <First as Op1>::apply),
    ("seq", <SeqOp as Op1>::apply),
    ("empty?", <EmptyP as Op1>::apply),
    ("int", <IntCast as Op1>::apply),
    ("long", <LongCast as Op1>::apply),
    ("double", <DoubleCast as Op1>::apply),
    ("unchecked-inc", <UncheckedInc as Op1>::apply),
    ("unchecked-dec", <UncheckedDec as Op1>::apply),
    ("unchecked-negate", <UncheckedNegate as Op1>::apply),
    ("deref", <DerefCell as Op1>::apply),
    ("char", <CharOf as Op1>::apply),
    ("second", <Second as Op1>::apply),
];

pub(crate) const APPLY2: &[(&str, Apply2)] = &[
    ("+", <Add as Op2>::apply),
    ("-", <Sub as Op2>::apply),
    ("*", <Mul as Op2>::apply),
    ("+'", <Add as Op2>::apply),
    ("-'", <Sub as Op2>::apply),
    ("*'", <Mul as Op2>::apply),
    ("<", <Lt as Op2>::apply),
    ("<=", <Lte as Op2>::apply),
    (">", <Gt as Op2>::apply),
    (">=", <Gte as Op2>::apply),
    ("==", <NumEq as Op2>::apply),
    ("=", <Eq as Op2>::apply),
    ("not=", <NotEq as Op2>::apply),
    ("quot", <Quot as Op2>::apply),
    ("rem", <Rem as Op2>::apply),
    ("mod", <Mod as Op2>::apply),
    ("bit-and", <BitAnd as Op2>::apply),
    ("bit-or", <BitOr as Op2>::apply),
    ("bit-xor", <BitXor as Op2>::apply),
    ("bit-shift-left", <Shl as Op2>::apply),
    ("bit-shift-right", <Shr as Op2>::apply),
    ("unsigned-bit-shift-right", <Ushr as Op2>::apply),
    ("min", <Min as Op2>::apply),
    ("max", <Max as Op2>::apply),
    ("nth", <Nth as Op2>::apply),
    ("get", <Get as Op2>::apply),
    ("dissoc", <Dissoc as Op2>::apply),
    ("contains?", <Contains as Op2>::apply),
    ("conj!", <ConjBang as Op2>::apply),
    ("aget", <Aget as Op2>::apply),
    ("unchecked-add", <UncheckedAdd as Op2>::apply),
    ("unchecked-subtract", <UncheckedSub as Op2>::apply),
    ("unchecked-multiply", <UncheckedMul as Op2>::apply),
    ("compare", <CompareNum as Op2>::apply),
    ("subvec", <Subvec2 as Op2>::apply),
];

/// Specialized regex matcher kept outside [`APPLY2`] so non-regex AOT binaries
/// calling `rt_n_op2` strip `regex_automata`.
pub(crate) const RE_MATCHES_OP: (&str, Apply2) = ("re-matches", <ReMatches as Op2>::apply);

pub(crate) const APPLY3: &[(&str, Apply3)] = &[
    ("assoc!", <AssocBang as Op3>::apply),
    ("nth", <Nth3 as Op3>::apply),
    ("get", <Get3 as Op3>::apply),
    ("aset", <Aset as Op3>::apply),
    ("subvec", <Subvec3 as Op3>::apply),
];

/// The 3+-argument forms over borrowed args for native code (#259):
/// each row's binary op, folded (`false`) or chained like a comparison
/// (`true`). Kept in step with [`NARY`] (pinned by
/// `apply_tables_cover_every_specialized_op`).
pub(crate) const APPLYN: &[(&str, Apply2, bool)] = &[
    ("+", <Add as Op2>::apply, false),
    ("-", <Sub as Op2>::apply, false),
    ("*", <Mul as Op2>::apply, false),
    ("+'", <Add as Op2>::apply, false),
    ("-'", <Sub as Op2>::apply, false),
    ("*'", <Mul as Op2>::apply, false),
    ("min", <Min as Op2>::apply, false),
    ("max", <Max as Op2>::apply, false),
    ("bit-and", <BitAnd as Op2>::apply, false),
    ("bit-or", <BitOr as Op2>::apply, false),
    ("bit-xor", <BitXor as Op2>::apply, false),
    ("<", <Lt as Op2>::apply, true),
    ("<=", <Lte as Op2>::apply, true),
    (">", <Gt as Op2>::apply, true),
    (">=", <Gte as Op2>::apply, true),
    ("==", <NumEq as Op2>::apply, true),
];

/// The three-argument ops by core name (value code).
const TERNARY: &[Row3] = &[
    ("assoc!", shape3::<AssocBang>),
    ("nth", shape3::<Nth3>),
    ("get", shape3::<Get3>),
    ("aset", shape3::<Aset>),
    ("subvec", shape3::<Subvec3>),
];

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
    ("dissoc", shape2::<Dissoc, AsValue>, None),
    (
        "contains?",
        shape2::<Contains, AsValue>,
        Some(shape2::<Contains, AsTest>),
    ),
    ("conj!", shape2::<ConjBang, AsValue>, None),
    ("aget", shape2::<Aget, AsValue>, None),
    ("unchecked-add", shape2::<UncheckedAdd, AsValue>, None),
    ("unchecked-subtract", shape2::<UncheckedSub, AsValue>, None),
    ("unchecked-multiply", shape2::<UncheckedMul, AsValue>, None),
    ("compare", shape2::<CompareNum, AsValue>, None),
    ("subvec", shape2::<Subvec2, AsValue>, None),
    (
        "re-matches",
        shape2::<ReMatches, AsValue>,
        Some(shape2::<ReMatches, AsTest>),
    ),
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
    (
        "string?",
        shape1::<IsString, AsValue>,
        Some(shape1::<IsString, AsTest>),
    ),
    ("parse-long", shape1::<ParseLong, AsValue>, None),
    ("count", shape1::<Count, AsValue>, None),
    ("peek", shape1::<Peek, AsValue>, None),
    ("pop!", shape1::<PopBang, AsValue>, None),
    ("bit-not", shape1::<BitNot, AsValue>, None),
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
    ("unchecked-inc", shape1::<UncheckedInc, AsValue>, None),
    ("unchecked-dec", shape1::<UncheckedDec, AsValue>, None),
    ("unchecked-negate", shape1::<UncheckedNegate, AsValue>, None),
    ("deref", shape1::<DerefCell, AsValue>, None),
    ("char", shape1::<CharOf, AsValue>, None),
    ("second", shape1::<Second, AsValue>, None),
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
    Three(Ctor3<T>),
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
            Ctor::Three(c) => {
                let a = next();
                let b = next();
                c(site, a, b, next())
            }
            Ctor::N(c) => c(site, ops.collect()),
        }
    }
}

fn unary_value_ctor(name: &str) -> Option<Ctor<Value>> {
    UNARY
        .iter()
        .find(|(n, ..)| *n == name)
        .map(|(_, v, _)| Ctor::One(*v))
}

fn binary_value_ctor(name: &str) -> Option<Ctor<Value>> {
    BINARY
        .iter()
        .find(|(n, ..)| *n == name)
        .map(|(_, v, _)| Ctor::Two(*v))
}

fn ternary_value_ctor(name: &str) -> Option<Ctor<Value>> {
    TERNARY
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| Ctor::Three(*v))
        .or_else(|| {
            NARY.iter()
                .find(|(n, ..)| *n == name)
                .map(|(_, v, _)| Ctor::N(*v))
        })
}

fn nary_value_ctor(name: &str) -> Option<Ctor<Value>> {
    NARY.iter()
        .find(|(n, ..)| *n == name)
        .map(|(_, v, _)| Ctor::N(*v))
}

/// The specialized value code for `call`, if its op has one. Its
/// operands are then read in place, after the computed ones — the
/// last-use analysis must not move those locals.
pub(crate) fn value_ctor(call: &DirectCall) -> Option<Ctor<Value>> {
    let name = core_name(call)?;
    match call.args.len() {
        1 => unary_value_ctor(name),
        2 => binary_value_ctor(name),
        3 => ternary_value_ctor(name),
        4.. => nary_value_ctor(name),
        _ => None,
    }
}

fn unary_test_ctor(name: &str) -> Option<Ctor<bool>> {
    UNARY.iter().find(|(n, ..)| *n == name)?.2.map(Ctor::One)
}

fn binary_test_ctor(name: &str) -> Option<Ctor<bool>> {
    BINARY.iter().find(|(n, ..)| *n == name)?.2.map(Ctor::Two)
}

fn ternary_test_ctor(name: &str) -> Option<Ctor<bool>> {
    NARY.iter()
        .find(|(n, ..)| *n == name)
        .and_then(|(_, _, t)| t.map(Ctor::N))
}

fn nary_test_ctor(name: &str) -> Option<Ctor<bool>> {
    NARY.iter().find(|(n, ..)| *n == name)?.2.map(Ctor::N)
}

/// The specialized test code for `call`, if its op is a predicate.
pub(crate) fn test_ctor(call: &DirectCall) -> Option<Ctor<bool>> {
    let name = core_name(call)?;
    match call.args.len() {
        1 => unary_test_ctor(name),
        2 => binary_test_ctor(name),
        3 => ternary_test_ctor(name),
        4.. => nary_test_ctor(name),
        _ => None,
    }
}

/// Whether `call` compiles to specialized code (see [`value_ctor`]).
pub(crate) fn in_place(call: &DirectCall) -> bool {
    value_ctor(call).is_some()
}

/// `pop!` for effect only: the mutation, no transient clone.
struct PopBangFx;

impl Op1 for PopBangFx {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        PopBang::apply_mut(a)?;
        Some(Value::Nil)
    }
}

/// `conj!` for effect only (see [`PopBangFx`]).
struct ConjBangFx;

impl Op2 for ConjBangFx {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        ConjBang::apply_mut(a, b)?;
        Some(Value::Nil)
    }
}

/// `assoc!` for effect only (see [`PopBangFx`]).
struct AssocBangFx;

impl Op3 for AssocBangFx {
    #[inline(always)]
    fn apply(a: &Value, b: &Value, c: &Value) -> Option<Value> {
        AssocBang::apply_mut(a, b, c)?;
        Some(Value::Nil)
    }
}

/// The effect-position code for `call`, when its op is a
/// transient mutator whose returned transient the caller
/// discards: the mutation runs (a miss still runs the native,
/// whose return value the caller drops either way), but the
/// inline hit skips cloning the transient just to drop it.
pub(crate) fn effect_ctor(call: &DirectCall) -> Option<Ctor<Value>> {
    let name = core_name(call)?;
    match (name, call.args.len()) {
        ("pop!", 1) => Some(Ctor::One(shape1::<PopBangFx, AsValue>)),
        ("conj!", 2) => Some(Ctor::Two(shape2::<ConjBangFx, AsValue>)),
        ("assoc!", 3) => Some(Ctor::Three(shape3::<AssocBangFx>)),
        _ => None,
    }
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
fn upd<O: Upd>(c: Site, coll: Coll, mut rest: Vec<Operand>) -> Exec {
    match rest.len() {
        0 => upd1::<O>(c, coll),
        1 => upd2::<O>(c, coll, rest.pop().unwrap()),
        _ => {
            let o2 = rest.pop().unwrap();
            let o1 = rest.pop().unwrap();
            upd3::<O>(c, coll, o1, o2)
        }
    }
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

/// A two-operand update op (`conj`): stack-allocated args without SmallVec.
fn upd2<O: Upd>(c: Site, coll: Coll, o1: Operand) -> Exec {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let mut a = match &coll {
            Coll::Exec(e) => e(ev, f)?,
            _ => Value::Nil,
        };
        let b = o1.eval(ev, f)?;
        match &coll {
            Coll::Take(r) => a = std::mem::replace(&mut f.regs[*r], Value::Nil),
            Coll::Read(o) => a = o.eval(ev, f)?,
            Coll::Exec(_) => {}
        }
        let mut args = [a, b];
        if valid {
            if let Some(v) = O::apply(&mut args) {
                return Ok(v);
            }
        }
        let [a, b] = args;
        native(ev, &c, smallvec![a, b])
    })
}

/// A three-operand update op (`assoc`): stack-allocated args without SmallVec.
fn upd3<O: Upd>(c: Site, coll: Coll, o1: Operand, o2: Operand) -> Exec {
    Box::new(move |ev, f| {
        let valid = ev.site_direct_valid(c.sid, &c.call);
        let mut a = match &coll {
            Coll::Exec(e) => e(ev, f)?,
            _ => Value::Nil,
        };
        let b = o1.eval(ev, f)?;
        let c_val = o2.eval(ev, f)?;
        match &coll {
            Coll::Take(r) => a = std::mem::replace(&mut f.regs[*r], Value::Nil),
            Coll::Read(o) => a = o.eval(ev, f)?,
            Coll::Exec(_) => {}
        }
        let mut args = [a, b, c_val];
        if valid {
            if let Some(v) = O::apply(&mut args) {
                return Ok(v);
            }
        }
        let [a, b, c_val] = args;
        native(ev, &c, smallvec![a, b, c_val])
    })
}

/// An update op whose fast path is its owned native (`assoc-in`,
/// `update`, `update-in`, and multi-pair `assoc`).
fn upd_native(c: Site, coll: Coll, rest: Vec<Operand>) -> Exec {
    Box::new(move |ev, f| {
        let mut args: SmallVec<[Value; 4]> = SmallVec::with_capacity(rest.len() + 1);
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
        native(ev, &c, args)
    })
}

/// The update ops by core name and arg count.
/// An update-op table row: core name, arg count, code constructor.
type UpdRow = (&'static str, usize, UpdCtor);

const UPDATES: &[UpdRow] = &[
    ("assoc", 3, upd::<Assoc>),
    ("conj", 2, upd::<Conj>),
    ("pop", 1, upd::<Pop>),
    ("next", 1, upd::<Next>),
    ("rest", 1, upd::<Rest>),
];

/// An update op's code constructor (see [`upd`]).
pub(crate) type UpdCtor = fn(Site, Coll, Vec<Operand>) -> Exec;

/// The update op for `call`, if it is one: its collection operand
/// is then read after the others (see [`upd`]).
pub(crate) fn update_ctor(call: &DirectCall) -> Option<UpdCtor> {
    let name = core_name(call)?;
    let argc = call.args.len();
    if let Some((_, _, ctor)) = UPDATES.iter().find(|(n, a, _)| *n == name && *a == argc) {
        return Some(*ctor);
    }
    match name {
        "assoc" if argc >= 5 && argc % 2 == 1 => Some(upd_native),
        "assoc-in" if argc == 3 => Some(upd_native),
        "update" | "update-in" if argc >= 3 => Some(upd_native),
        "transient" | "persistent!" if argc == 1 => Some(upd_native),
        _ => None,
    }
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
        match try_assoc_leaf(take(coll), k, v) {
            Ok(out) => Some(out),
            Err(orig) => {
                *coll = orig;
                None
            }
        }
    }
}

#[cold]
#[inline(never)]
fn try_assoc_subvec(
    mut ext: Arc<dyn clj_value::ExtensionValue>,
    i: i64,
    v: &mut Value,
) -> Result<Value, Value> {
    let Some((start, end, orig_start, compact)) = ext
        .as_any()
        .downcast_ref::<clj_value::SubVectorValue>()
        .map(|sv| {
            let len = sv.len();
            (
                sv.start(),
                sv.end(),
                sv.orig_start(),
                len <= 256 && sv.vec().len() > len,
            )
        })
    else {
        return Err(Value::Extension(ext));
    };
    let len = end - start;
    let Ok(idx) = usize::try_from(i as i32) else {
        return Err(Value::Extension(ext));
    };
    if idx >= len {
        return Err(Value::Extension(ext));
    }
    let val = take(v);
    if !compact {
        if let Some(sv_mut) = Arc::get_mut(&mut ext)
            .and_then(|e| e.as_any_mut())
            .and_then(|a| a.downcast_mut::<clj_value::SubVectorValue>())
        {
            sv_mut.set_abs_in_place(start + idx, val);
            return Ok(Value::Extension(ext));
        }
    }
    let sv = ext
        .as_any()
        .downcast_ref::<clj_value::SubVectorValue>()
        .expect("checked SubVectorValue");
    let next = if compact {
        let backing = sv
            .vec()
            .slice_packed(start, end)
            .set_nth_owned(idx, val)
            .expect("idx < len");
        clj_value::SubVectorValue::new(backing, 0, len)
            .with_orig_start(orig_start)
            .with_meta(sv.meta().cloned())
    } else {
        let backing = sv
            .vec()
            .set_nth(start + idx, val)
            .expect("abs < backing len");
        clj_value::SubVectorValue::new(backing, start, end)
            .with_orig_start(orig_start)
            .with_meta(sv.meta().cloned())
    };
    Ok(Value::Extension(Arc::new(next)))
}

fn try_assoc_leaf(coll: Value, k: &mut Value, v: &mut Value) -> Result<Value, Value> {
    match (coll, &*k) {
        (Value::Vector(x), Value::Int(i)) => match usize::try_from(*i as i32) {
            Ok(n) if n < x.len() => Ok(Value::Vector(
                x.set_nth_owned(n, take(v)).expect("n < x.len()"),
            )),
            Ok(n) if n == x.len() => Ok(Value::Vector(x.push_back(take(v)))),
            _ => Err(Value::Vector(x)),
        },
        (Value::Map(m), k_ref) if !m.is_sorted() && crate::vm_intrinsics::is_scalar_key(k_ref) => {
            Ok(Value::Map(m.insert(take(k), take(v))))
        }
        (Value::Extension(ext), Value::Int(i)) => try_assoc_subvec(ext, *i, v),
        (other, _) => Err(other),
    }
}

fn fast_inc_dec_delta(f: &Value) -> Option<i64> {
    match crate::ext::as_native(f)?.fast {
        crate::ext::FastArith::Inc => Some(1),
        crate::ext::FastArith::Dec => Some(-1),
        _ => None,
    }
}

fn try_update_leaf_int(coll: Value, k: &mut Value, delta: i64) -> Result<Value, Value> {
    match (coll, &*k) {
        (Value::Vector(x), Value::Int(i)) => {
            let Some(n) = usize::try_from(*i).ok().filter(|&n| n < x.len()) else {
                return Err(Value::Vector(x));
            };
            let Some(Value::Int(old)) = x.get(n) else {
                return Err(Value::Vector(x));
            };
            let Some(new) = old.checked_add(delta) else {
                return Err(Value::Vector(x));
            };
            Ok(Value::Vector(
                x.set_nth_owned(n, Value::Int(new)).expect("n < x.len()"),
            ))
        }
        (Value::Map(mut m), k_ref)
            if !m.is_sorted() && crate::vm_intrinsics::is_scalar_key(k_ref) =>
        {
            if m.update_existing_mut(k_ref, |v| match v {
                Value::Int(old) => old.checked_add(delta).map(Value::Int).ok_or(v),
                other => Err(other),
            }) {
                Ok(Value::Map(m))
            } else {
                Err(Value::Map(m))
            }
        }
        (other, _) => Err(other),
    }
}

fn try_step_in2(
    outer: Value,
    k1: &mut Value,
    step: impl FnOnce(Value) -> Result<Value, Value>,
) -> Result<Value, Value> {
    match (outer, &*k1) {
        (Value::Map(mut m), k1_ref)
            if !m.is_sorted() && crate::vm_intrinsics::is_scalar_key(k1_ref) =>
        {
            if m.update_existing_mut(k1_ref, step) {
                Ok(Value::Map(m))
            } else {
                Err(Value::Map(m))
            }
        }
        (Value::Vector(mut v), Value::Int(i1)) => {
            let Some(n1) = usize::try_from(*i1).ok().filter(|&n1| n1 < v.len()) else {
                return Err(Value::Vector(v));
            };
            let inner = v.take_nth(n1).expect("n1 < v.len()");
            match step(inner) {
                Ok(updated) => Ok(Value::Vector(
                    v.set_nth_owned(n1, updated).expect("n1 < v.len()"),
                )),
                Err(orig_inner) => {
                    let v = v.set_nth_owned(n1, orig_inner).expect("n1 < v.len()");
                    Err(Value::Vector(v))
                }
            }
        }
        (other, _) => Err(other),
    }
}

/// Owned fast path for 3-arg `(update coll k inc/dec)`.
pub(crate) fn apply_update3(args: &mut [Value]) -> Option<Value> {
    let [coll, k, f] = args else { return None };
    let delta = fast_inc_dec_delta(f)?;
    match try_update_leaf_int(take(coll), k, delta) {
        Ok(out) => Some(out),
        Err(orig) => {
            *coll = orig;
            None
        }
    }
}

/// Owned fast path for 2-key `(assoc-in coll [k1 k2] v)` lowered as
/// 4 args `[coll, k1, k2, v]`.
pub(crate) fn apply_assoc_in2(args: &mut [Value]) -> Option<Value> {
    let [coll, k1, k2, v] = args else { return None };
    match try_step_in2(take(coll), k1, |inner| try_assoc_leaf(inner, k2, v)) {
        Ok(out) => Some(out),
        Err(orig) => {
            *coll = orig;
            None
        }
    }
}

/// Owned fast path for 2-key `(update-in coll [k1 k2] inc/dec)` lowered as
/// 4 args `[coll, k1, k2, f]`.
pub(crate) fn apply_update_in2(args: &mut [Value]) -> Option<Value> {
    let [coll, k1, k2, f] = args else { return None };
    let delta = fast_inc_dec_delta(f)?;
    match try_step_in2(take(coll), k1, |inner| {
        try_update_leaf_int(inner, k2, delta)
    }) {
        Ok(out) => Some(out),
        Err(orig) => {
            *coll = orig;
            None
        }
    }
}

fn finish_swap(val: &mut Value, res: Result<Value, Value>, keep_ret: bool) -> Option<Value> {
    match res {
        Ok(out) => {
            if keep_ret {
                *val = out.clone();
                Some(out)
            } else {
                *val = out;
                Some(Value::Nil)
            }
        }
        Err(orig) => {
            *val = orig;
            None
        }
    }
}

/// In-place `swap!` fast path for unwatched, unvalidated atoms with
/// pure core update functions (`assoc`, 2-key `assoc-in`, `+`).
pub(crate) fn try_swap_in_place(
    val: &mut Value,
    f: &Value,
    extra: &[Value],
    keep_ret: bool,
) -> Option<Value> {
    let nat = crate::ext::as_native(f)?;
    match (&*nat.display, extra) {
        ("clojure.core/assoc", [k, v]) => {
            let mut kc = k.clone();
            let mut vc = v.clone();
            let res = try_assoc_leaf(take(val), &mut kc, &mut vc);
            finish_swap(val, res, keep_ret)
        }
        ("clojure.core/assoc-in", [Value::Vector(ks), v]) if ks.len() == 2 => {
            let mut k1 = ks.get(0).expect("len 2");
            let mut k2 = ks.get(1).expect("len 2");
            let mut vc = v.clone();
            let res = try_step_in2(take(val), &mut k1, |inner| {
                try_assoc_leaf(inner, &mut k2, &mut vc)
            });
            finish_swap(val, res, keep_ret)
        }
        ("clojure.core/+", [Value::Int(d)]) => {
            let Value::Int(old) = *val else { return None };
            let new = Value::Int(old.checked_add(*d)?);
            *val = new.clone();
            Some(if keep_ret { new } else { Value::Nil })
        }
        _ => None,
    }
}

/// Owned effect-position fast path for `(swap! atom f extra...)` (skips
/// cloning the atom's new value when the caller discards the return value).
pub(crate) fn apply_swap_fx(args: &mut [Value]) -> Option<Value> {
    let [Value::Atom(atom), f, extra @ ..] = args else {
        return None;
    };
    let mut state = atom.lock();
    if state.validator.is_some() || !state.watches.is_empty() {
        return None;
    }
    try_swap_in_place(&mut state.value, f, extra, false)
}

/// Owned fast path for 2-key `(swap! atom assoc-in [k1 k2] v)` lowered as
/// 5 args `[atom, assoc-in, k1, k2, v]` without allocating `[k1 k2]`.
pub(crate) fn apply_swap_assoc_in2(args: &mut [Value], keep_ret: bool) -> Option<Value> {
    let [Value::Atom(atom), f, k1, k2, v] = args else {
        return None;
    };
    if crate::ext::as_native(f)?.display.as_ref() != "clojure.core/assoc-in" {
        return None;
    }
    let mut state = atom.lock();
    if state.validator.is_some() || !state.watches.is_empty() {
        return None;
    }
    let res = try_step_in2(take(&mut state.value), k1, |inner| {
        try_assoc_leaf(inner, k2, v)
    });
    finish_swap(&mut state.value, res, keep_ret)
}

pub(crate) fn apply_swap_assoc_in2_val(args: &mut [Value]) -> Option<Value> {
    apply_swap_assoc_in2(args, true)
}

pub(crate) fn apply_swap_assoc_in2_fx(args: &mut [Value]) -> Option<Value> {
    apply_swap_assoc_in2(args, false)
}

/// A local or literal operand of the fused transient update (see
/// [`assoc_conj_code`]), read in place.
pub(crate) enum InPlace {
    Reg(usize),
    Up(usize),
    Kst(Value),
}

impl InPlace {
    #[inline(always)]
    fn get<'v>(&'v self, f: &'v Frame<'_>) -> &'v Value {
        match self {
            InPlace::Reg(r) => &f.regs[*r],
            InPlace::Up(u) => &f.up[*u],
            InPlace::Kst(v) => v,
        }
    }
}

/// The fused update's appended operand: side-effect free, so it
/// may evaluate before the slot's item is read.
pub(crate) enum PureSrc {
    One(InPlace),
    /// A vector literal (no metadata) of locals and literals.
    Vec(Box<[InPlace]>),
}

impl PureSrc {
    fn value(&self, f: &Frame<'_>) -> Value {
        match self {
            PureSrc::One(s) => s.get(f).clone(),
            PureSrc::Vec(items) => Value::Vector(clj_value::Vector::from_vec(
                items.iter().map(|s| s.get(f).clone()).collect(),
            )),
        }
    }
}

/// `(assoc! t k (conj (nth t k) x))` over a transient local `t`, the
/// same key `k` in both places, and a pure `x` (see
/// `Compiler::try_assoc_conj`). The generic road's `nth` clones the
/// slot's item, so `conj` sees it shared and copies it whole (an
/// O(n) append per edge in adjacency builds). Here the item moves
/// out of the transient, takes `x` in place through [`Conj`], and
/// moves back. The three sites validate first, and any miss runs
/// `generic`, whose results and errors are the natives'.
pub(crate) fn assoc_conj_code(
    sites: [Site; 3],
    t: InPlace,
    k: InPlace,
    x: PureSrc,
    effect: bool,
    generic: Exec,
) -> Exec {
    Box::new(move |ev, f| {
        if sites.iter().all(|s| s.valid(ev)) && assoc_conj_hit(t.get(f), k.get(f), &x, f) {
            return Ok(if effect { Value::Nil } else { t.get(f).clone() });
        }
        generic(ev, f)
    })
}

/// The fused update's inline step (`false` leaves the transient as
/// it was).
/// Update slot `i` of a live transient vector by `conj`ing `*xv` in
/// place; restores `*xv` if the slot or transient cannot be updated.
#[inline]
pub(crate) fn assoc_conj_in_cell(cell: &clj_value::Transient, i: usize, xv: &mut Value) -> bool {
    let mut guard = cell.lock();
    let clj_value::TransientKind::Vector(tv) = &mut *guard else {
        return false;
    };
    let Some(item) = tv.take_at(i) else {
        return false;
    };
    let val = std::mem::replace(xv, Value::Nil);
    let mut args = [item, val];
    let (updated, hit) = match Conj::apply(&mut args) {
        Some(v) => (v, true),
        None => (std::mem::replace(&mut args[0], Value::Nil), false),
    };
    // The slot was in bounds a moment ago, so the write-back holds.
    let ok = tv.assoc(i, updated).is_ok() && hit;
    if !ok {
        *xv = std::mem::replace(&mut args[1], Value::Nil);
    }
    ok
}

#[inline]
fn assoc_conj_hit(t: &Value, k: &Value, x: &PureSrc, f: &Frame<'_>) -> bool {
    let (Value::Transient(cell), Value::Int(n)) = (t, k) else {
        return false;
    };
    let Ok(i) = usize::try_from(*n) else {
        return false;
    };
    let mut xv = x.value(f);
    assoc_conj_in_cell(cell, i, &mut xv)
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

/// Owned `dissoc` of one scalar key (#271): a uniquely held map (a
/// moved loop accumulator) drops the key in place; the borrowed
/// [`Dissoc`] op path-copies.
pub(crate) struct DissocOwn;

impl Upd for DissocOwn {
    #[inline(always)]
    fn apply(args: &mut [Value]) -> Option<Value> {
        let [coll, k] = args else { return None };
        if !crate::vm_intrinsics::is_scalar_key(k) {
            return None;
        }
        match std::mem::replace(coll, Value::Nil) {
            Value::Map(m) if !m.is_sorted() => Some(Value::Map(m.without(k))),
            Value::Nil => Some(Value::Nil),
            other => {
                *coll = other;
                None
            }
        }
    }
}

/// Owned `disj` of one scalar member (#271), like [`DissocOwn`].
pub(crate) struct DisjOwn;

impl Upd for DisjOwn {
    #[inline(always)]
    fn apply(args: &mut [Value]) -> Option<Value> {
        let [coll, x] = args else { return None };
        if !crate::vm_intrinsics::is_scalar_key(x) {
            return None;
        }
        match std::mem::replace(coll, Value::Nil) {
            Value::Set(s) if !s.is_sorted() => Some(Value::Set(s.without(x))),
            Value::Nil => Some(Value::Nil),
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
fn arith<F: Fn(i64, i64) -> Option<i64>, G: Fn(f64, f64) -> f64>(
    a: &Value,
    b: &Value,
    int: F,
    dbl: G,
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
fn compare<F: Fn(i64, i64) -> bool, G: Fn(f64, f64) -> bool>(
    a: &Value,
    b: &Value,
    int: F,
    dbl: G,
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
fn ints<F: Fn(i64, i64) -> Option<i64>>(a: &Value, b: &Value, op: F) -> Option<Value> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => op(*x, *y).map(Value::Int),
        _ => None,
    }
}

/// A numeric predicate over `Int` and `Double`.
#[inline(always)]
fn sign<F: Fn(i64) -> bool, G: Fn(f64) -> bool>(a: &Value, int: F, dbl: G) -> Option<bool> {
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

/// A one-argument op over unboxed ints: the `i64` mirror of its
/// [`Op1`] namesake's hit on an `Int` (`None` exactly where the
/// op misses, so the caller deopts to the boxed road). A test
/// pins the mirrors against the ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntOp1 {
    Inc,
    Dec,
    Negate,
    BitNot,
    UInc,
    UDec,
    UNegate,
}

impl IntOp1 {
    #[inline(always)]
    pub(crate) fn apply(self, x: i64) -> Option<i64> {
        match self {
            IntOp1::Inc => x.checked_add(1),
            IntOp1::Dec => x.checked_sub(1),
            IntOp1::Negate => x.checked_neg(),
            IntOp1::BitNot => Some(!x),
            IntOp1::UInc => Some(x.wrapping_add(1)),
            IntOp1::UDec => Some(x.wrapping_sub(1)),
            IntOp1::UNegate => Some(x.wrapping_neg()),
        }
    }
}

/// A two-argument op over unboxed ints (see [`IntOp1`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntOp2 {
    Add,
    Sub,
    Mul,
    UAdd,
    USub,
    UMul,
    Quot,
    Rem,
    RemUnsigned,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Ushr,
    Min,
    Max,
}

impl IntOp2 {
    #[inline(always)]
    pub(crate) fn apply(self, x: i64, y: i64) -> Option<i64> {
        match self {
            IntOp2::Add | IntOp2::Sub | IntOp2::Mul => self.apply_checked(x, y),
            IntOp2::UAdd | IntOp2::USub | IntOp2::UMul => self.apply_wrapping(x, y),
            IntOp2::Quot | IntOp2::Rem | IntOp2::RemUnsigned | IntOp2::Mod => self.apply_div(x, y),
            _ => self.apply_bits(x, y),
        }
    }

    #[inline(always)]
    fn apply_checked(self, x: i64, y: i64) -> Option<i64> {
        match self {
            IntOp2::Add => x.checked_add(y),
            IntOp2::Sub => x.checked_sub(y),
            _ => x.checked_mul(y),
        }
    }

    #[inline(always)]
    fn apply_wrapping(self, x: i64, y: i64) -> Option<i64> {
        Some(match self {
            IntOp2::UAdd => x.wrapping_add(y),
            IntOp2::USub => x.wrapping_sub(y),
            _ => x.wrapping_mul(y),
        })
    }

    #[inline(always)]
    fn apply_div(self, x: i64, y: i64) -> Option<i64> {
        match self {
            IntOp2::Quot => (y != 0).then(|| x.wrapping_div(y)),
            IntOp2::Rem => (y != 0).then(|| x.wrapping_rem(y)),
            IntOp2::RemUnsigned => (y != 0).then(|| ((x as u64) % (y as u64)) as i64),
            _ => crate::vm_intrinsics::java_mod(x, y),
        }
    }

    #[inline(always)]
    fn apply_bits(self, x: i64, y: i64) -> Option<i64> {
        Some(match self {
            IntOp2::BitAnd => x & y,
            IntOp2::BitOr => x | y,
            IntOp2::BitXor => x ^ y,
            IntOp2::Shl => x.wrapping_shl(y as u32),
            IntOp2::Shr => x.wrapping_shr(y as u32),
            IntOp2::Ushr => (x as u64).wrapping_shr(y as u32) as i64,
            IntOp2::Min => x.min(y),
            _ => x.max(y),
        })
    }
}

/// A one-argument predicate over an unboxed int (see [`IntOp1`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntTest1 {
    Zero,
    Pos,
    Neg,
}

impl IntTest1 {
    #[inline(always)]
    pub(crate) fn test(self, x: i64) -> bool {
        match self {
            IntTest1::Zero => x == 0,
            IntTest1::Pos => x > 0,
            IntTest1::Neg => x < 0,
        }
    }
}

/// A two-argument predicate over unboxed ints (see [`IntOp1`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntTest2 {
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
    NotEq,
}

impl IntTest2 {
    #[inline(always)]
    pub(crate) fn test(self, x: i64, y: i64) -> bool {
        match self {
            IntTest2::Lt => x < y,
            IntTest2::Lte => x <= y,
            IntTest2::Gt => x > y,
            IntTest2::Gte => x >= y,
            IntTest2::Eq => x == y,
            IntTest2::NotEq => x != y,
        }
    }
}

/// An unboxed value op by operand count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntCtor {
    One(IntOp1),
    Two(IntOp2),
}

/// An unboxed test op by operand count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntTestCtor {
    One(IntTest1),
    Two(IntTest2),
}

/// The unboxed one-argument value ops by core name (the primed
/// spellings share the op; the miss road differs, as in
/// [`UNARY`]).
const INT_UNARY: &[(&str, IntOp1)] = &[
    ("inc", IntOp1::Inc),
    ("inc'", IntOp1::Inc),
    ("dec", IntOp1::Dec),
    ("dec'", IntOp1::Dec),
    ("-", IntOp1::Negate),
    ("-'", IntOp1::Negate),
    ("bit-not", IntOp1::BitNot),
    ("unchecked-inc", IntOp1::UInc),
    ("unchecked-dec", IntOp1::UDec),
    ("unchecked-negate", IntOp1::UNegate),
];

/// The unboxed two-argument value ops by core name ([`BINARY`]).
const INT_BINARY: &[(&str, IntOp2)] = &[
    ("+", IntOp2::Add),
    ("+'", IntOp2::Add),
    ("-", IntOp2::Sub),
    ("-'", IntOp2::Sub),
    ("*", IntOp2::Mul),
    ("*'", IntOp2::Mul),
    ("unchecked-add", IntOp2::UAdd),
    ("unchecked-subtract", IntOp2::USub),
    ("unchecked-multiply", IntOp2::UMul),
    ("quot", IntOp2::Quot),
    ("rem", IntOp2::Rem),
    ("mod", IntOp2::Mod),
    ("bit-and", IntOp2::BitAnd),
    ("bit-or", IntOp2::BitOr),
    ("bit-xor", IntOp2::BitXor),
    ("bit-shift-left", IntOp2::Shl),
    ("bit-shift-right", IntOp2::Shr),
    ("unsigned-bit-shift-right", IntOp2::Ushr),
    ("min", IntOp2::Min),
    ("max", IntOp2::Max),
];

/// The unboxed one-argument tests by core name.
const INT_TEST1: &[(&str, IntTest1)] = &[
    ("zero?", IntTest1::Zero),
    ("pos?", IntTest1::Pos),
    ("neg?", IntTest1::Neg),
];

/// The unboxed two-argument tests by core name (`=` and `==`
/// agree on `Int` pairs).
const INT_TEST2: &[(&str, IntTest2)] = &[
    ("<", IntTest2::Lt),
    ("<=", IntTest2::Lte),
    (">", IntTest2::Gt),
    (">=", IntTest2::Gte),
    ("=", IntTest2::Eq),
    ("==", IntTest2::Eq),
    ("not=", IntTest2::NotEq),
];

fn int_value_named(name: &str, arity: usize) -> Option<IntCtor> {
    match arity {
        1 => INT_UNARY
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, op)| IntCtor::One(*op)),
        2 => INT_BINARY
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, op)| IntCtor::Two(*op)),
        _ => None,
    }
}

fn int_test_named(name: &str, arity: usize) -> Option<IntTestCtor> {
    match arity {
        1 => INT_TEST1
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, op)| IntTestCtor::One(*op)),
        2 => INT_TEST2
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, op)| IntTestCtor::Two(*op)),
        _ => None,
    }
}

/// The unboxed value op for `call`, when its fixed-arity op has
/// an `Int`-pair hit the driver can mirror. The caller still
/// validates the call's site per stretch, like the shapes.
pub(crate) fn int_value_op(call: &DirectCall) -> Option<IntCtor> {
    int_value_named(core_name(call)?, call.args.len())
}

/// The binary op an n-ary (three or more arguments) call folds
/// left with, for the ops `clojure.core` defines as that fold
/// (`(+ a b c)` is `(+ (+ a b) c)`, each step checked).
pub(crate) fn int_fold_op(call: &DirectCall) -> Option<IntOp2> {
    const FOLDS: &[&str] = &[
        "+", "+'", "-", "-'", "*", "*'", "bit-and", "bit-or", "bit-xor", "min", "max",
    ];
    let name = core_name(call)?;
    if call.args.len() < 3 || !FOLDS.contains(&name) {
        return None;
    }
    INT_BINARY
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, op)| *op)
}

/// The unboxed test op for `call` (see [`int_value_op`]).
pub(crate) fn int_test_op(call: &DirectCall) -> Option<IntTestCtor> {
    int_test_named(core_name(call)?, call.args.len())
}
pub(crate) struct Nth;

impl Op2 for Nth {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        let Value::Int(n) = b else {
            return None;
        };
        if *n < 0 || *n > i32::MAX as i64 {
            return None;
        }
        let idx = *n as usize;
        match a {
            Value::Vector(v) => v.get(idx),
            Value::Str(s) => crate::coll::str_char_at(s, idx).map(Value::Char),
            Value::Transient(t) => {
                let guard = t.lock();
                match &*guard {
                    clj_value::TransientKind::Vector(tv) => tv.nth_opt(idx),
                    _ => None,
                }
            }
            Value::Extension(e) => {
                let any = e.as_any();
                if let Some(sv) = any.downcast_ref::<clj_value::SubVectorValue>() {
                    return sv.nth(idx);
                }
                if let Some(sp) = any.downcast_ref::<clj_value::SplitVecValue>() {
                    return sp.nth(idx);
                }
                crate::vm_intrinsics::nth_of(a, b, None)
            }
            _ => crate::vm_intrinsics::nth_of(a, b, None),
        }
    }
}

pub(crate) struct Get;

impl Op2 for Get {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        match a {
            Value::Map(m) if !m.is_sorted() && crate::vm_intrinsics::is_lookup_key(b) => {
                Some(m.get(b).cloned().unwrap_or(Value::Nil))
            }
            _ => crate::vm_intrinsics::get_of(a, b, None),
        }
    }
}

pub(crate) struct Dissoc;

impl Op2 for Dissoc {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        match a {
            Value::Map(m) if !m.is_sorted() && crate::vm_intrinsics::is_scalar_key(b) => {
                Some(Value::Map(m.clone().without(b)))
            }
            Value::Nil => Some(Value::Nil),
            _ => None,
        }
    }
}

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

/// `(bit-not x)` on an integer.
/// `unchecked-add`/`-subtract`/`-multiply` over two longs (#271): the
/// wrapping op (`(unchecked-add Long/MAX_VALUE 1)` is `Long/MIN_VALUE`);
/// doubles and the tower are the native's.
pub(crate) struct UncheckedAdd;
pub(crate) struct UncheckedSub;
pub(crate) struct UncheckedMul;

impl Op2 for UncheckedAdd {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        wrapping2(a, b, i64::wrapping_add)
    }
}

impl Op2 for UncheckedSub {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        wrapping2(a, b, i64::wrapping_sub)
    }
}

impl Op2 for UncheckedMul {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        wrapping2(a, b, i64::wrapping_mul)
    }
}

#[inline(always)]
fn wrapping2(a: &Value, b: &Value, op: fn(i64, i64) -> i64) -> Option<Value> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(Value::Int(op(*x, *y))),
        _ => None,
    }
}

/// `unchecked-inc`/`-dec`/`-negate` over a long (#271): wrapping.
pub(crate) struct UncheckedInc;
pub(crate) struct UncheckedDec;
pub(crate) struct UncheckedNegate;

impl Op1 for UncheckedInc {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        wrapping1(a, |x| x.wrapping_add(1))
    }
}

impl Op1 for UncheckedDec {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        wrapping1(a, |x| x.wrapping_sub(1))
    }
}

impl Op1 for UncheckedNegate {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        wrapping1(a, i64::wrapping_neg)
    }
}

#[inline(always)]
fn wrapping1(a: &Value, op: fn(i64) -> i64) -> Option<Value> {
    match a {
        Value::Int(x) => Some(Value::Int(op(*x))),
        _ => None,
    }
}

/// `@atom`/`@volatile` (#271): the cell's value, read under its lock as
/// the native reads it; every other derefable (refs, delays, futures,
/// vars, the timeout arity) is the native's.
pub(crate) struct DerefCell;

impl Op1 for DerefCell {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Atom(c) => Some(c.lock().value.clone()),
            Value::Volatile(v) => Some(v.lock().clone()),
            _ => None,
        }
    }
}

/// `(char n)` for a long code point (#271): `char::from_u32` as the
/// native narrows it; out of range (and every other shape) is the
/// native's, error included.
pub(crate) struct CharOf;

impl Op1 for CharOf {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        let Value::Int(i) = a else {
            return None;
        };
        u32::try_from(*i)
            .ok()
            .and_then(char::from_u32)
            .map(Value::Char)
    }
}

/// `compare` over longs and doubles (#271): `-1`/`0`/`1` from
/// `jvm_compare`'s own numeric order; other operands are the native's.
pub(crate) struct CompareNum;

impl Op2 for CompareNum {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        crate::core::compare_long_double(a, b).map(|o| Value::Int(o as i64))
    }
}

pub(crate) struct BitNot;

impl Op1 for BitNot {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        let Value::Int(x) = a else {
            return None;
        };
        Some(Value::Int(!x))
    }
}

pub(crate) struct Count;

impl Op1 for Count {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        crate::vm_intrinsics::count_of(a)
    }
}

/// `(peek coll)` over `nil`, lists, and plain (unmarked) vectors.
pub(crate) struct Peek;

impl Op1 for Peek {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Nil => Some(Value::Nil),
            Value::List(l) => Some(l.first().cloned().unwrap_or(Value::Nil)),
            Value::Vector(x) if x.meta().is_none() => Some(if x.is_empty() {
                Value::Nil
            } else {
                x.get(x.len() - 1).unwrap_or(Value::Nil)
            }),
            _ => None,
        }
    }
}

/// `(pop coll)` over a non-empty plain (unmarked) vector: a last-use
/// operand moves out and drops its last item in place.
pub(crate) struct Pop;

impl Upd for Pop {
    #[inline(always)]
    fn apply(args: &mut [Value]) -> Option<Value> {
        let [coll] = args else { return None };
        match std::mem::replace(coll, Value::Nil) {
            Value::Vector(x) if !x.is_empty() && x.meta().is_none() => {
                x.drop_last_owned().map(Value::Vector)
            }
            other => {
                *coll = other;
                None
            }
        }
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

/// `(subvec v start)` of a vector (#271 Slice 6p): `subvec3` to its
/// count; anything else, and every error, takes the native.
pub(crate) struct Subvec2;

impl Op2 for Subvec2 {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        let Value::Vector(v) = a else { return None };
        crate::coll::subvec3(a, b, &Value::Int(v.len() as i64)).ok()
    }
}

/// `(subvec v start end)` (#271 Slice 6p): `subvec3`; every error takes
/// the native.
pub(crate) struct Subvec3;

impl Op3 for Subvec3 {
    #[inline(always)]
    fn apply(a: &Value, b: &Value, c: &Value) -> Option<Value> {
        crate::coll::subvec3(a, b, c).ok()
    }
}

/// `(second coll)` of a vector or nil (#271 Slice 6n): the item at 1, else
/// nil; other colls take the native.
pub(crate) struct Second;

impl Op1 for Second {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        match a {
            Value::Vector(v) => Some(v.get(1).unwrap_or(Value::Nil)),
            Value::Nil => Some(Value::Nil),
            _ => None,
        }
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

/// `(pop! t)` over a live transient vector (drops its last item in
/// place; anything else runs the native).
pub(crate) struct PopBang;

impl PopBang {
    /// The mutation alone (see [`effect_ctor`]).
    #[inline(always)]
    fn apply_mut(a: &Value) -> Option<()> {
        let Value::Transient(t) = a else {
            return None;
        };
        let mut guard = t.lock();
        let clj_value::TransientKind::Vector(tv) = &mut *guard else {
            return None;
        };
        tv.pop().ok()?;
        drop(guard);
        Some(())
    }
}

impl Op1 for PopBang {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        Self::apply_mut(a)?;
        Some(a.clone())
    }
}

/// `(conj! tv x)` onto a live transient vector (the cell updates in
/// place; anything else — or a frozen transient — runs the native).
pub(crate) struct ConjBang;

impl ConjBang {
    /// The mutation alone (see [`effect_ctor`]).
    #[inline(always)]
    fn apply_mut(a: &Value, b: &Value) -> Option<()> {
        let Value::Transient(t) = a else {
            return None;
        };
        let mut guard = t.lock();
        let clj_value::TransientKind::Vector(tv) = &mut *guard else {
            return None;
        };
        if let Value::Int(n) = b {
            if !tv.conj_int_opt(*n) {
                return None;
            }
        } else {
            tv.conj(b.clone()).ok()?;
        }
        drop(guard);
        Some(())
    }
}

impl Op2 for ConjBang {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        Self::apply_mut(a, b)?;
        Some(a.clone())
    }
}

impl ConjBang {
    /// In-place mutation without cloning the transient wrapper.
    #[inline(always)]
    pub(crate) fn apply_fx(a: &Value, b: &Value) -> bool {
        Self::apply_mut(a, b).is_some()
    }
}

/// `(assoc! t k v)` on a live transient vector or transient map
/// (the cell updates in place; anything else runs the native).
pub(crate) struct AssocBang;

impl AssocBang {
    /// The mutation alone (see [`effect_ctor`]).
    #[inline(always)]
    fn apply_mut(a: &Value, b: &Value, c: &Value) -> Option<()> {
        let Value::Transient(t) = a else {
            return None;
        };
        let mut guard = t.lock();
        match &mut *guard {
            clj_value::TransientKind::Vector(tv) => {
                let Value::Int(n) = b else {
                    return None;
                };
                if *n < 0 || *n > i32::MAX as i64 {
                    return None;
                }
                let i = *n as usize;
                if let Value::Int(x) = c {
                    if !tv.assoc_int_opt(i, *x) {
                        return None;
                    }
                } else {
                    tv.assoc(i, c.clone()).ok()?;
                }
            }
            clj_value::TransientKind::Map(tm) => {
                if !crate::vm_intrinsics::is_scalar_key(b) {
                    return None;
                }
                tm.assoc(b.clone(), c.clone()).ok()?;
            }
            _ => return None,
        }
        drop(guard);
        Some(())
    }
}

impl Op3 for AssocBang {
    #[inline(always)]
    fn apply(a: &Value, b: &Value, c: &Value) -> Option<Value> {
        Self::apply_mut(a, b, c)?;
        Some(a.clone())
    }
}

/// `(aget a i)` inline (#271): an array read at an in-bounds long
/// index. Any other shape (a negative or out-of-bounds index, a
/// coerced index, nil, a non-array) takes the native for its exact
/// answer or error.
pub(crate) struct Aget;

impl Op2 for Aget {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        let (Value::Array(arr), Value::Int(i)) = (a, b) else {
            return None;
        };
        crate::arrays::aget_fast(arr, usize::try_from(*i).ok()?)
    }
}

/// `(aset a i v)` inline (#271): a primitive slot taking its own value
/// type at an in-bounds long index (`aset_prim_fast`); the rest takes
/// the native.
pub(crate) struct Aset;

impl Op3 for Aset {
    #[inline(always)]
    fn apply(a: &Value, b: &Value, c: &Value) -> Option<Value> {
        let (Value::Array(arr), Value::Int(i)) = (a, b) else {
            return None;
        };
        let i = usize::try_from(*i).ok()?;
        crate::arrays::aset_prim_fast(arr, i, c).then(|| c.clone())
    }
}

pub(crate) struct Nth3;

impl Op3 for Nth3 {
    #[inline(always)]
    fn apply(a: &Value, b: &Value, c: &Value) -> Option<Value> {
        let Value::Int(n) = b else {
            return None;
        };
        let Ok(i32_n) = i32::try_from(*n) else {
            return None;
        };
        if i32_n < 0 {
            return Some(c.clone());
        }
        let idx = i32_n as usize;
        match a {
            Value::Vector(v) => Some(v.get(idx).unwrap_or_else(|| c.clone())),
            Value::Str(s) => {
                Some(crate::coll::str_char_at(s, idx).map_or_else(|| c.clone(), Value::Char))
            }
            Value::Transient(t) => {
                let guard = t.lock();
                if let clj_value::TransientKind::Vector(tv) = &*guard {
                    let v = tv.nth(idx).ok()?.unwrap_or_else(|| c.clone());
                    drop(guard);
                    return Some(v);
                }
                None
            }
            Value::Extension(e) => {
                if let Some(sv) = e.as_any().downcast_ref::<clj_value::SubVectorValue>() {
                    return Some(sv.nth(idx).unwrap_or_else(|| c.clone()));
                }
                crate::vm_intrinsics::nth_of(a, b, Some(c))
            }
            _ => crate::vm_intrinsics::nth_of(a, b, Some(c)),
        }
    }
}

pub(crate) struct Get3;

impl Op3 for Get3 {
    #[inline(always)]
    fn apply(a: &Value, b: &Value, c: &Value) -> Option<Value> {
        match a {
            Value::Map(m) if !m.is_sorted() && crate::vm_intrinsics::is_lookup_key(b) => {
                Some(m.get(b).cloned().unwrap_or_else(|| c.clone()))
            }
            _ => crate::vm_intrinsics::get_of(a, b, Some(c)),
        }
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
pred1!(IsString, |a| Some(matches!(a, Value::Str(_))));

pub(crate) struct ParseLong;

impl Op1 for ParseLong {
    #[inline(always)]
    fn apply(a: &Value) -> Option<Value> {
        crate::core::parse_long_1(a).ok()
    }
}

pub(crate) struct ReMatches;

impl Op2 for ReMatches {
    #[inline(always)]
    fn apply(a: &Value, b: &Value) -> Option<Value> {
        crate::re::re_matches_2(a, b).ok()
    }

    #[inline(always)]
    fn test(a: &Value, b: &Value) -> Option<bool> {
        if let (Value::Regex(src), Value::Str(s)) = (a, b) {
            if let Some(m) = crate::re::fast_re_matches(src, s) {
                return Some(m);
            }
        }
        Self::apply(a, b).map(|v| v.is_truthy())
    }
}

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

    #[test]
    fn rematches_op2() {
        let fast_pat = Value::Regex(Arc::from("-?[0-9]+"));
        let s_123 = Value::Str(Arc::from("123"));
        let s_abc = Value::Str(Arc::from("abc"));
        assert_eq!(<ReMatches as Op2>::test(&fast_pat, &s_123), Some(true));
        assert_eq!(<ReMatches as Op2>::test(&fast_pat, &s_abc), Some(false));
        let slow_pat = Value::Regex(Arc::from("a+"));
        let s_aaa = Value::Str(Arc::from("aaa"));
        let s_bbb = Value::Str(Arc::from("bbb"));
        assert_eq!(<ReMatches as Op2>::test(&slow_pat, &s_aaa), Some(true));
        assert_eq!(<ReMatches as Op2>::test(&slow_pat, &s_bbb), Some(false));
        assert_eq!(<ReMatches as Op2>::test(&slow_pat, &Value::Int(1)), None);
        assert_eq!(
            <ReMatches as Op2>::apply(&fast_pat, &s_123),
            Some(s_123.clone())
        );
        assert_eq!(
            <ReMatches as Op2>::apply(&fast_pat, &s_abc),
            Some(Value::Nil)
        );
    }

    #[test]
    fn parse_long_and_is_string() {
        let s_42 = Value::Str(Arc::from("42"));
        let s_bad = Value::Str(Arc::from("bad"));
        let s_hi = Value::Str(Arc::from("hi"));
        assert_eq!(<ParseLong as Op1>::apply(&s_42), Some(Value::Int(42)));
        assert_eq!(<ParseLong as Op1>::apply(&s_bad), Some(Value::Nil));
        assert_eq!(<ParseLong as Op1>::apply(&Value::Int(1)), None);
        assert_eq!(<IsString as Op1>::apply(&s_hi), Some(Value::Bool(true)));
        assert_eq!(
            <IsString as Op1>::apply(&Value::Int(1)),
            Some(Value::Bool(false))
        );
        assert_eq!(<IsString as Op1>::test(&s_hi), Some(true));
        assert_eq!(<IsString as Op1>::test(&Value::Int(1)), Some(false));
    }

    #[test]
    fn peek_pop_and_lookup_fast_paths() {
        let v_empty = Value::Vector(clj_value::Vector::new_sync());
        let v_one = Value::Vector(clj_value::Vector::from_vec(vec![Value::Int(7)]));
        let l_empty = Value::List(clj_value::List::new_sync());
        let l_one = Value::List(clj_value::List::from_vec(vec![Value::Int(9)]));
        assert_eq!(<Peek as Op1>::apply(&Value::Nil), Some(Value::Nil));
        assert_eq!(<Peek as Op1>::apply(&l_empty), Some(Value::Nil));
        assert_eq!(<Peek as Op1>::apply(&l_one), Some(Value::Int(9)));
        assert_eq!(<Peek as Op1>::apply(&v_empty), Some(Value::Nil));
        assert_eq!(<Peek as Op1>::apply(&v_one), Some(Value::Int(7)));
        assert_eq!(<Peek as Op1>::apply(&Value::Int(1)), None);

        assert_eq!(<Pop as Upd>::apply(&mut []), None);
        assert_eq!(<Pop as Upd>::apply(&mut [v_empty.clone()]), None);
        assert_eq!(
            <Pop as Upd>::apply(&mut [v_one.clone()]),
            Some(v_empty.clone())
        );

        let mut ev = crate::boot();
        let res = ev
            .eval_source(
                r#"
                (let [s #{:a :b}
                      v [10 20 30]
                      st "hello"]
                  [(s :a) (s :c :nf) (get s :b) (get s :c :nf)
                   (v 1) (get st 1) (get st 99 :nf) (get st :bad)
                   (get (transient [1]) 0) (get (transient {:a 1}) [1 2])
                   (nth (subvec v 1) 0) (count (subvec v 1))
                   (get (subvec v 1) 0) (get (subvec v 1) 99 :nf)
                   (reduce + 0 (subvec v 1))
                   (apply nth [(subvec v 1) 0])
                   (apply peek [nil]) (apply peek [(vec (range 5))])
                   (#{[1 2]} [1 2] :nf) (#{[1 2]} [3 4] :nf)
                   (vec "abc") (into [] "xy")
                   (vec (map s [:a :c])) (vec (map #(inc %) [1 2]))
                   (vec (map (var inc) [1 2]))
                   (vec (filter s [:a :c])) (vec (filter seq ["a" ""]))
                   (vec (filter (var pos?) [1 -1]))
                   (every? s [:a :b]) (every? (var pos?) [1 2])
                   (some s [:c :a]) (some (var pos?) [-1 2])])
                "#,
            )
            .unwrap();
        assert!(matches!(res.last(), Some(Value::Vector(_))));
    }

    #[test]
    fn pop_bang_and_bit_not() {
        assert_eq!(<BitNot as Op1>::apply(&Value::Int(0)), Some(Value::Int(-1)));
        assert_eq!(<BitNot as Op1>::apply(&Value::Int(-1)), Some(Value::Int(0)));
        assert_eq!(<BitNot as Op1>::apply(&Value::Nil), None);

        let tv = Value::Transient(clj_value::Transient::new(clj_value::TransientKind::Vector(
            clj_value::TransientVector::new(),
        )));
        assert_eq!(<PopBang as Op1>::apply(&tv), None);
        assert_eq!(<PopBang as Op1>::apply(&Value::Nil), None);

        assert!(<ConjBang as Op2>::apply(&tv, &Value::Int(123)).is_some());
        assert_eq!(<PopBang as Op1>::apply(&tv), Some(tv.clone()));

        let tm = Value::Transient(clj_value::Transient::new(clj_value::TransientKind::Map(
            clj_value::TransientMap::new(),
        )));
        assert_eq!(<PopBang as Op1>::apply(&tm), None);
    }

    /// The unboxed mirrors hit and miss exactly where their ops
    /// do, over edge-valued `Int` inputs.
    #[test]
    fn int_ops_mirror_op_hits() {
        const EDGE: &[i64] = &[
            0,
            1,
            -1,
            2,
            -2,
            3,
            63,
            64,
            -64,
            65,
            i64::MAX,
            i64::MIN,
            i64::MAX - 1,
            i64::MIN + 1,
            1 << 32,
            -(1 << 32),
        ];
        fn unbox(v: Option<Value>) -> Option<i64> {
            match v {
                Some(Value::Int(n)) => Some(n),
                _ => None,
            }
        }
        for &x in EDGE {
            let vx = Value::Int(x);
            assert_eq!(IntOp1::Inc.apply(x), unbox(<Inc as Op1>::apply(&vx)));
            assert_eq!(IntOp1::Dec.apply(x), unbox(<Dec as Op1>::apply(&vx)));
            assert_eq!(IntOp1::Negate.apply(x), unbox(<Negate as Op1>::apply(&vx)));
            assert_eq!(IntOp1::BitNot.apply(x), unbox(<BitNot as Op1>::apply(&vx)));
            assert_eq!(IntTest1::Zero.test(x), <Zero as Op1>::test(&vx).unwrap());
            assert_eq!(IntTest1::Pos.test(x), <Pos as Op1>::test(&vx).unwrap());
            assert_eq!(IntTest1::Neg.test(x), <Neg as Op1>::test(&vx).unwrap());
            for &y in EDGE {
                let vy = Value::Int(y);
                macro_rules! pair {
                    ($int:expr, $op:ty) => {
                        assert_eq!(
                            $int.apply(x, y),
                            unbox(<$op as Op2>::apply(&vx, &vy)),
                            "{:?} on ({x}, {y})",
                            $int
                        )
                    };
                }
                pair!(IntOp2::Add, Add);
                pair!(IntOp2::Sub, Sub);
                pair!(IntOp2::Mul, Mul);
                pair!(IntOp2::Quot, Quot);
                pair!(IntOp2::Rem, Rem);
                pair!(IntOp2::Mod, Mod);
                pair!(IntOp2::BitAnd, BitAnd);
                pair!(IntOp2::BitOr, BitOr);
                pair!(IntOp2::BitXor, BitXor);
                pair!(IntOp2::Shl, Shl);
                pair!(IntOp2::Shr, Shr);
                pair!(IntOp2::Ushr, Ushr);
                pair!(IntOp2::Min, Min);
                pair!(IntOp2::Max, Max);
                macro_rules! cmp {
                    ($int:expr, $op:ty) => {
                        assert_eq!(
                            $int.test(x, y),
                            <$op as Op2>::test(&vx, &vy).unwrap(),
                            "{:?} on ({x}, {y})",
                            $int
                        )
                    };
                }
                cmp!(IntTest2::Lt, Lt);
                cmp!(IntTest2::Lte, Lte);
                cmp!(IntTest2::Gt, Gt);
                cmp!(IntTest2::Gte, Gte);
                cmp!(IntTest2::Eq, Eq);
                cmp!(IntTest2::Eq, NumEq);
                cmp!(IntTest2::NotEq, NotEq);
            }
        }
        // The name mapping covers the primed spellings and stays
        // inside the fixed-arity int ops.
        assert!(matches!(
            int_value_named("inc'", 1),
            Some(IntCtor::One(IntOp1::Inc))
        ));
        assert!(matches!(
            int_value_named("+'", 2),
            Some(IntCtor::Two(IntOp2::Add))
        ));
        assert!(matches!(
            int_test_named("==", 2),
            Some(IntTestCtor::Two(IntTest2::Eq))
        ));
        assert_eq!(int_value_named("nth", 2), None);
        assert_eq!(int_value_named("inc", 2), None);
        assert_eq!(int_value_named("+", 3), None);
        assert_eq!(int_test_named("zero?", 2), None);
        assert_eq!(int_test_named("<", 3), None);
    }
}

#[cfg(test)]
mod apply_tests {
    use super::{APPLY1, APPLY2, APPLY3, APPLYN, BINARY, NARY, RE_MATCHES_OP, TERNARY, UNARY};
    use super::{
        Assoc, CharOf, CompareNum, DerefCell, Op1, Op2, UncheckedAdd, UncheckedDec, UncheckedInc,
        UncheckedMul, UncheckedNegate, UncheckedSub, Upd,
    };
    use clj_value::Value;
    use std::sync::Arc;

    #[test]
    fn unchecked_deref_char_and_compare_ops_271() {
        let (int, s) = (Value::Int, Value::Str(std::sync::Arc::from("s")));
        assert_eq!(
            UncheckedAdd::apply(&int(i64::MAX), &int(1)),
            Some(int(i64::MIN))
        );
        assert_eq!(
            UncheckedSub::apply(&int(i64::MIN), &int(1)),
            Some(int(i64::MAX))
        );
        assert_eq!(UncheckedMul::apply(&int(i64::MAX), &int(2)), Some(int(-2)));
        assert_eq!(UncheckedAdd::apply(&int(1), &Value::Double(1.0)), None);
        assert_eq!(UncheckedInc::apply(&int(i64::MAX)), Some(int(i64::MIN)));
        assert_eq!(UncheckedDec::apply(&int(i64::MIN)), Some(int(i64::MAX)));
        assert_eq!(UncheckedNegate::apply(&int(i64::MIN)), Some(int(i64::MIN)));
        assert_eq!(UncheckedInc::apply(&s), None);
        use super::{IntCtor, IntOp1, IntOp2, int_value_named};
        assert_eq!(IntOp1::UInc.apply(i64::MAX), Some(i64::MIN));
        assert_eq!(IntOp1::UDec.apply(i64::MIN), Some(i64::MAX));
        assert_eq!(IntOp1::UNegate.apply(i64::MIN), Some(i64::MIN));
        assert_eq!(IntOp2::UAdd.apply(i64::MAX, 1), Some(i64::MIN));
        assert_eq!(IntOp2::USub.apply(i64::MIN, 1), Some(i64::MAX));
        assert_eq!(IntOp2::UMul.apply(i64::MAX, 2), Some(-2));
        assert_eq!(IntOp2::RemUnsigned.apply(-1, 10), Some(5));
        assert_eq!(IntOp2::RemUnsigned.apply(10, 0), None);
        assert!(matches!(
            int_value_named("unchecked-inc", 1),
            Some(IntCtor::One(IntOp1::UInc))
        ));
        assert!(matches!(
            int_value_named("unchecked-add", 2),
            Some(IntCtor::Two(IntOp2::UAdd))
        ));
        let atom = Value::Atom(clj_value::Atom::new(int(3)));
        let vol = Value::Volatile(clj_value::Volatile::new(int(4)));
        assert_eq!(DerefCell::apply(&atom), Some(int(3)));
        assert_eq!(DerefCell::apply(&vol), Some(int(4)));
        assert_eq!(DerefCell::apply(&Value::Nil), None);
        assert_eq!(CharOf::apply(&int(97)), Some(Value::Char('a')));
        assert_eq!(CharOf::apply(&int(0xD800)), None);
        assert_eq!(CharOf::apply(&int(-1)), None);
        assert_eq!(CharOf::apply(&Value::Char('a')), None);
        assert_eq!(
            CompareNum::apply(&int(1), &Value::Double(2.5)),
            Some(int(-1))
        );
        assert_eq!(CompareNum::apply(&int(2), &int(2)), Some(int(0)));
        assert_eq!(CompareNum::apply(&s, &s), None);
        // The natives answer the same when called as values.
        let mut ev = crate::boot();
        let v = ev
            .eval_source(
                "(= [(reduce unchecked-add [9223372036854775807 1])
                     (reduce unchecked-subtract [-9223372036854775808 1])
                     (reduce unchecked-multiply [9223372036854775807 2])
                     (first (map unchecked-inc [9223372036854775807]))
                     (first (map unchecked-dec [-9223372036854775808]))
                     (first (map unchecked-negate [-9223372036854775808]))
                     (first (map deref [(volatile! 5)]))]
                    [-9223372036854775808 9223372036854775807 -2
                     -9223372036854775808 9223372036854775807 -9223372036854775808 5])",
            )
            .expect("eval");
        assert_eq!(v.last(), Some(&Value::Bool(true)));
    }

    #[test]
    fn apply_tables_cover_every_specialized_op() {
        let a1: Vec<&str> = APPLY1.iter().map(|r| r.0).collect();
        let mut a2: Vec<&str> = APPLY2.iter().map(|r| r.0).collect();
        a2.push(RE_MATCHES_OP.0);
        let a3: Vec<&str> = APPLY3.iter().map(|r| r.0).collect();
        assert_eq!(a1, UNARY.iter().map(|r| r.0).collect::<Vec<_>>());
        assert_eq!(a2, BINARY.iter().map(|r| r.0).collect::<Vec<_>>());
        assert_eq!(a3, TERNARY.iter().map(|r| r.0).collect::<Vec<_>>());
        let an: Vec<&str> = APPLYN.iter().map(|r| r.0).collect();
        assert_eq!(an, NARY.iter().map(|r| r.0).collect::<Vec<_>>());
    }

    #[test]
    fn owned_update_in2_and_swap_in_place_271() {
        use super::{apply_assoc_in2, apply_update_in2, apply_update3, try_swap_in_place};
        use crate::native_abi::{
            TAKE_ARGS, apply_own, own_op, repack_own_args, single_take_arg_bit, take_arg_mask,
        };
        use clj_value::{Keyword, Map, Vector};

        assert_eq!(take_arg_mask(0), TAKE_ARGS | single_take_arg_bit(0));
        let ev = crate::boot();
        let core_fn = |name: &str| {
            let id = crate::ns::core_var(&ev, name).expect("core var");
            ev.registry
                .with_state_mut(id, |st| st.root.clone())
                .flatten()
                .expect("root")
        };
        let inc_fn = core_fn("inc");
        let key = Value::Keyword(Keyword::new(None, "k"));
        let map = Value::Map(Map::from_kvs_last_wins(vec![(key.clone(), Value::Nil)]));
        let mut args = [map.clone(), key, inc_fn.clone()];
        assert_eq!(apply_update3(&mut args), None);
        assert_eq!(args[0], map);
        let dec_fn = core_fn("dec");
        let add_fn = core_fn("+");
        let assoc_fn = core_fn("assoc");
        let assoc_in_fn = core_fn("assoc-in");
        let ka = Value::Keyword(Keyword::new(None, "a"));
        let kb = Value::Keyword(Keyword::new(None, "b"));

        // apply_update3 branches
        assert_eq!(apply_update3(&mut []), None);
        assert_eq!(
            apply_update3(&mut [Value::Nil, ka.clone(), add_fn.clone()]),
            None
        );
        assert_eq!(
            apply_update3(&mut [Value::Nil, ka.clone(), inc_fn.clone()]),
            None
        );
        let v_ok = Value::Vector(Vector::from_vec(vec![
            Value::Int(10),
            Value::Nil,
            Value::Int(i64::MAX),
        ]));
        let mut a = [v_ok.clone(), Value::Int(0), inc_fn.clone()];
        assert_eq!(
            apply_update3(&mut a),
            Some(Value::Vector(Vector::from_vec(vec![
                Value::Int(11),
                Value::Nil,
                Value::Int(i64::MAX)
            ])))
        );
        assert_eq!(
            apply_update3(&mut [v_ok.clone(), Value::Int(5), inc_fn.clone()]),
            None
        );
        assert_eq!(
            apply_update3(&mut [v_ok.clone(), Value::Int(1), inc_fn.clone()]),
            None
        );
        assert_eq!(
            apply_update3(&mut [v_ok, Value::Int(2), inc_fn.clone()]),
            None
        );

        let m_ok = Value::Map(Map::from_kvs_last_wins(vec![
            (ka.clone(), Value::Int(5)),
            (kb.clone(), Value::Int(i64::MIN)),
        ]));
        assert_eq!(
            apply_update3(&mut [m_ok.clone(), ka.clone(), dec_fn.clone()]),
            Some(Value::Map(Map::from_kvs_last_wins(vec![
                (ka.clone(), Value::Int(4)),
                (kb.clone(), Value::Int(i64::MIN)),
            ])))
        );
        assert_eq!(
            apply_update3(&mut [
                m_ok.clone(),
                Value::Keyword(Keyword::new(None, "missing")),
                inc_fn.clone()
            ]),
            None
        );
        assert_eq!(apply_update3(&mut [m_ok, kb.clone(), dec_fn.clone()]), None);

        // apply_assoc_in2 & apply_update_in2 branches
        assert_eq!(apply_assoc_in2(&mut []), None);
        assert_eq!(apply_update_in2(&mut []), None);
        assert_eq!(
            apply_update_in2(&mut [Value::Nil, ka.clone(), kb.clone(), add_fn.clone()]),
            None
        );
        let nested_map = Value::Map(Map::from_kvs_last_wins(vec![
            (
                ka.clone(),
                Value::Map(Map::from_kvs_last_wins(vec![(kb.clone(), Value::Int(1))])),
            ),
            (kb.clone(), Value::Int(9)),
        ]));
        let upd_map =
            apply_assoc_in2(&mut [nested_map.clone(), ka.clone(), kb.clone(), Value::Int(2)])
                .expect("assoc-in2 map");
        assert_eq!(
            apply_update_in2(&mut [upd_map, ka.clone(), kb.clone(), inc_fn.clone()]),
            Some(Value::Map(Map::from_kvs_last_wins(vec![
                (
                    ka.clone(),
                    Value::Map(Map::from_kvs_last_wins(vec![(kb.clone(), Value::Int(3))])),
                ),
                (kb.clone(), Value::Int(9)),
            ])))
        );
        // Missing outer key in map, failing inner step in map, non-map/vec outer
        assert_eq!(
            apply_assoc_in2(&mut [
                nested_map.clone(),
                Value::Keyword(Keyword::new(None, "missing")),
                kb.clone(),
                Value::Int(2),
            ]),
            None
        );
        assert_eq!(
            apply_assoc_in2(&mut [nested_map.clone(), kb.clone(), ka.clone(), Value::Int(2),]),
            None
        );
        assert_eq!(
            apply_assoc_in2(&mut [Value::Nil, ka.clone(), kb.clone(), Value::Int(2)]),
            None
        );
        assert_eq!(
            apply_update_in2(&mut [Value::Nil, ka.clone(), kb.clone(), inc_fn.clone()]),
            None
        );

        // Outer vector in try_step_in2
        let nested_vec = Value::Vector(Vector::from_vec(vec![
            Value::Vector(Vector::from_vec(vec![Value::Int(7)])),
            Value::Nil,
        ]));
        assert_eq!(
            apply_assoc_in2(&mut [
                nested_vec.clone(),
                Value::Int(0),
                Value::Int(1),
                Value::Int(8),
            ]),
            Some(Value::Vector(Vector::from_vec(vec![
                Value::Vector(Vector::from_vec(vec![Value::Int(7), Value::Int(8)])),
                Value::Nil,
            ])))
        );
        assert_eq!(
            apply_assoc_in2(&mut [
                nested_vec.clone(),
                Value::Int(9),
                Value::Int(0),
                Value::Int(8),
            ]),
            None
        );
        assert_eq!(
            apply_assoc_in2(&mut [nested_vec, Value::Int(1), Value::Int(0), Value::Int(8),]),
            None
        );

        // repack_own_args & apply_own
        let op_ain2 = own_op("assoc-in2", 4).expect("assoc-in2 op");
        let mut sv = smallvec::smallvec![Value::Nil, ka.clone(), kb.clone(), Value::Int(1)];
        repack_own_args(op_ain2, &mut sv);
        assert_eq!(sv.len(), 3);
        repack_own_args(u32::MAX, &mut sv);
        assert_eq!(sv.len(), 3);
        let op_upd3 = own_op("update", 3).expect("update op");
        assert!(apply_own(op_upd3, &mut [Value::Nil, ka.clone(), inc_fn.clone()]).is_none());

        // try_swap_in_place branches
        let mut v = Value::Int(10);
        assert_eq!(try_swap_in_place(&mut v, &Value::Nil, &[], true), None);
        assert_eq!(try_swap_in_place(&mut v, &inc_fn, &[], true), None);
        assert_eq!(
            try_swap_in_place(&mut v, &add_fn, &[Value::Int(5)], true),
            Some(Value::Int(15))
        );
        let mut v_max = Value::Int(i64::MAX);
        assert_eq!(
            try_swap_in_place(&mut v_max, &add_fn, &[Value::Int(1)], true),
            None
        );
        let mut v_nil = Value::Nil;
        assert_eq!(
            try_swap_in_place(&mut v_nil, &add_fn, &[Value::Int(1)], true),
            None
        );
        assert_eq!(
            try_swap_in_place(&mut v_nil, &assoc_fn, &[ka.clone(), Value::Int(1)], true),
            None
        );
        let mut v_map = Value::Map(Map::from_kvs_last_wins(vec![(ka.clone(), Value::Int(1))]));
        assert!(
            try_swap_in_place(&mut v_map, &assoc_fn, &[ka.clone(), Value::Int(2)], true).is_some()
        );
        let path2 = Value::Vector(Vector::from_vec(vec![ka.clone(), kb.clone()]));
        assert_eq!(
            try_swap_in_place(
                &mut v_nil,
                &assoc_in_fn,
                &[path2.clone(), Value::Int(1)],
                true
            ),
            None
        );
        let mut v_nmap = nested_map;
        assert!(
            try_swap_in_place(&mut v_nmap, &assoc_in_fn, &[path2, Value::Int(42)], true).is_some()
        );
        let atom = clj_value::Atom::new(v_nmap);
        let mut args = smallvec::smallvec![
            Value::Atom(atom.clone()),
            assoc_in_fn.clone(),
            ka.clone(),
            kb.clone(),
            Value::Int(50),
        ];
        let swap_val = own_op("swap-assoc-in2!", 5).expect("swap assoc-in op");
        let swap_fx = own_op("swap-assoc-in2-fx!", 5).expect("swap assoc-in effect op");
        args[1] = inc_fn.clone();
        assert_eq!(apply_own(swap_val, &mut args), None);
        args[1] = assoc_in_fn;
        assert_eq!(
            apply_own(swap_val, &mut args),
            Some(atom.lock().value.clone())
        );
        args[3] = kb.clone();
        args[4] = Value::Int(60);
        assert_eq!(apply_own(swap_fx, &mut args), Some(Value::Nil));
        assert_eq!(
            try_swap_in_place(&mut Value::Int(1), &add_fn, &[Value::Int(2)], false),
            Some(Value::Nil)
        );
        atom.lock().validator = Some(inc_fn);
        assert_eq!(apply_own(swap_fx, &mut args), None);
        repack_own_args(swap_fx, &mut args);
        assert_eq!(args.len(), 4);
        assert!(matches!(&args[2], Value::Vector(v) if v.len() == 2));
        let swap_fx = own_op("swap-fx!", 3).expect("swap effect op");
        assert_eq!(apply_own(swap_fx, &mut args[..3]), None);
        args[0] = Value::Nil;
        assert_eq!(apply_own(swap_fx, &mut args[..3]), None);
        assert_eq!(apply_own(swap_val, &mut args), None);

        // try_assoc_subvec branches: non-subvec ext, negative/OOB idx, compacting,
        // unique in-place, and shared non-compacting.
        let non_sv = Value::Extension(Arc::new(clj_value::SplitVecValue::new(
            Arc::from("a b"),
            Arc::new(vec![(0, 1), (2, 1)]),
        )));
        assert_eq!(
            Assoc::apply(&mut [non_sv, Value::Int(0), Value::Int(1)]),
            None
        );
        let sv_wide = Value::Extension(Arc::new(clj_value::SubVectorValue::new(
            Vector::from_ints(vec![10, 20, 30, 40]),
            1,
            3,
        )));
        assert_eq!(
            Assoc::apply(&mut [sv_wide.clone(), Value::Int(-1), Value::Int(9)]),
            None
        );
        assert_eq!(
            Assoc::apply(&mut [sv_wide.clone(), Value::Int(2), Value::Int(9)]),
            None
        );
        let compacted =
            Assoc::apply(&mut [sv_wide, Value::Int(0), Value::Int(99)]).expect("compact");
        let shared = compacted.clone();
        let mutated_shared =
            Assoc::apply(&mut [compacted, Value::Int(1), Value::Int(88)]).expect("shared");
        drop(shared);
        let mutated_unique =
            Assoc::apply(&mut [mutated_shared, Value::Int(0), Value::Int(77)]).expect("unique");
        assert_eq!(
            mutated_unique,
            Value::Vector(Vector::from_ints(vec![77, 88]))
        );
    }
}
