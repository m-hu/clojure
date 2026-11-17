// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S5D seq + higher-order natives: lazy transformers over
//! [`crate::lazy`] thunks, eager consumers, and the fn-returning
//! combinators (built as [`NativeFn`] closures).
//!
//! Transducer (`xform`) arities live in [`crate::xform`] (T1200):
//! single-argument `map`/`filter`/etc. return transducer fns.

use std::cmp::Ordering;
use std::sync::Arc;

use clj_value::{List, ThrowClass, Value, Vector, new_map};
use smallvec::SmallVec;

use crate::core::{as_reduced, cce_err, check_ifn, jvm_compare, repeat_count, take_count};
use crate::error::{NativeError, jvm_class_of};
use crate::eval::Evaluator;
use crate::ext::{FastArith, NativeFn, NativeImpl};
use crate::lazy::{
    StepUnit, as_subvector, collect_all, collect_packed, concat_thunk, cycle_thunk, dedupe_thunk,
    distinct_thunk, drop_value, drop_while_thunk, empty_list_value, filter_thunk, interleave_thunk,
    interpose_thunk, iterate_thunk, keep_indexed_thunk, keep_thunk, keys_thunk, map_indexed_thunk,
    map_thunk, mapcat_thunk, next_value, nthrest_value, partition_all_thunk, partition_by_thunk,
    partition_thunk, range1, range2, range3, realized_value, reductions_thunk, repeat_thunk,
    repeatedly_thunk, seq_value, step_unit, take_nth_thunk, take_thunk, take_walk_coll,
    take_while_thunk,
};
use crate::ns::{NativeFnPtr, check_arity, deref_core, signal_to_native};
use crate::vars::check_min_arity;

/// Call a user fn with the `IFn` check first (bad callees fail
/// with the JVM `checkcast` text, matching what the JVM's call
/// sites raise).
pub(crate) fn call_checked(
    eval: &mut Evaluator,
    f: &Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    check_ifn(eval, f).map_err(signal_to_native)?;
    eval.call(f, args).map_err(signal_to_native)
}

/// [`call_checked`] with an owned arg buffer: accumulator step
/// calls (`reduce`, transducers) move the accumulator through,
/// so a uniquely-owned acc stays unique into an owned native
/// (`conj`) instead of pinning shared behind the borrow (#227).
/// Anything else borrows the buffer, exactly as before.
pub(crate) fn call_checked_owned(
    eval: &mut Evaluator,
    f: &Value,
    args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_ifn(eval, f).map_err(signal_to_native)?;
    eval.call_owned(f, args).map_err(signal_to_native)
}

/// A fresh [`NativeFn`] value capturing its environment.
fn native_fn(
    eval: &mut Evaluator,
    display: &str,
    func: impl Fn(&mut Evaluator, &[Value]) -> Result<Value, NativeError> + Send + Sync + 'static,
) -> Value {
    Value::Extension(Arc::new(NativeFn {
        id: eval.next_id(),
        display: Arc::from(display),
        func: Arc::new(func) as Arc<NativeImpl>,
        owned: None,
        meta: None,
        fast: FastArith::None,
    }))
}

/// Head-by-head seq walking (short-circuitable; errors map to
/// native errors at each step).
pub(crate) enum Walk {
    Continue,
    Stop,
}

pub(crate) fn walk_seq(
    eval: &mut Evaluator,
    coll: Value,
    mut visit: impl FnMut(&mut Evaluator, &Value) -> Result<Walk, NativeError>,
) -> Result<(), NativeError> {
    if let Some(done) = walk_direct(eval, &coll, &mut visit) {
        return done;
    }
    if let Some(done) = walk_sole_map(eval, &coll, &mut visit) {
        return done;
    }
    let mut seq = seq_value(eval, &coll).map_err(signal_to_native)?;
    // The walk holds `seq`, never `coll`: owned callers nilled the
    // arg slot (`take_walk_coll`), so dropping the head here frees
    // memoized chunks as the walk advances (borrow callers pass a
    // cheap clone, retaining as before).
    drop(coll);
    // Step-wise, not `first`/`next` per head: a chunked step hands
    // over its whole chunk (a range rebuilt its 32-item chunk for
    // every `first`). Forcing order is unchanged: each step forces
    // exactly what `first` on that seq would, after the previous
    // head's visit.
    loop {
        match step_unit(eval, &seq).map_err(signal_to_native)? {
            StepUnit::Empty => return Ok(()),
            StepUnit::One { head, tail } => {
                if matches!(visit(eval, &head)?, Walk::Stop) {
                    return Ok(());
                }
                seq = tail;
            }
            StepUnit::Chunk { items, tail } => {
                for head in items.iter() {
                    if matches!(visit(eval, head)?, Walk::Stop) {
                        return Ok(());
                    }
                }
                seq = tail;
            }
        }
    }
}

/// Walk a sole unforced `map` seq without its lazy cells (#271, see
/// [`crate::lazy::walk_sole_map`]; `None`: walk the seq).
fn walk_sole_map(
    eval: &mut Evaluator,
    coll: &Value,
    visit: &mut impl FnMut(&mut Evaluator, &Value) -> Result<Walk, NativeError>,
) -> Option<Result<(), NativeError>> {
    let mut failed = None;
    let mut sole = |eval: &mut Evaluator, head: Value| match visit(eval, &head) {
        Ok(walk) => Ok(matches!(walk, Walk::Continue)),
        Err(e) => {
            failed = Some(e);
            Ok(false)
        }
    };
    let done = crate::lazy::walk_sole_map(eval, coll, &mut sole)?;
    Some(
        done.map_err(signal_to_native)
            .and(failed.map_or(Ok(()), Err)),
    )
}

/// Walk a vector, a finite repeat, or a counted `long` range in
/// place (#226): no chunk seq copies the items 32 at a time — the
/// order and items are the seq's (`None`: walk the seq).
fn walk_direct(
    eval: &mut Evaluator,
    coll: &Value,
    visit: &mut impl FnMut(&mut Evaluator, &Value) -> Result<Walk, NativeError>,
) -> Option<Result<(), NativeError>> {
    let mut each = |items: &mut dyn Iterator<Item = Value>| {
        for head in items {
            if matches!(visit(eval, &head)?, Walk::Stop) {
                break;
            }
        }
        Ok(())
    };
    if let Value::Vector(v) = coll {
        return Some(each(&mut v.iter()));
    }
    if let Some(sv) = crate::lazy::as_subvector(coll) {
        return Some(each(&mut sv.vec().iter().skip(sv.start()).take(sv.len())));
    }
    if let Value::Str(s) = coll {
        return Some(each(&mut s.chars().map(Value::Char)));
    }
    if let Value::Array(a) = coll {
        return Some(each(&mut a.snapshot().into_iter()));
    }
    if let Some((x, n)) = finite_repeat(coll) {
        return Some(each(&mut (0..n).map(|_| x.clone())));
    }
    let (cur, step, n) = long_range(coll)?;
    Some(each(&mut (0..n).map(|k| Value::Int(cur + k * step))))
}

/// A finite `repeat`'s item and count.
fn finite_repeat(coll: &Value) -> Option<(Value, i64)> {
    let Value::Extension(e) = coll else {
        return None;
    };
    let r = e.as_any().downcast_ref::<clj_value::RepeatValue>()?;
    Some((r.x().clone(), r.n()?.max(0)))
}

/// A counted `long` range's start, step, and count (items stay in
/// `i64`: the range's own bound checked its last item).
pub(crate) fn long_range(coll: &Value) -> Option<(i64, i64, i64)> {
    let Value::Extension(e) = coll else {
        return None;
    };
    let r = e.as_any().downcast_ref::<clj_value::RangeValue>()?;
    match (r.cur(), r.step(), r.count().filter(|_| r.is_long())) {
        (clj_value::RangeNum::Int(cur), clj_value::RangeNum::Int(step), Some(n)) => {
            Some((cur, step, n.max(0)))
        }
        _ => None,
    }
}

/// Force one step plus `n` heads (mirrors `dorun`'s walk: the
/// initial `seq` always realizes one step, then `n` `next`s).
fn force_n(eval: &mut Evaluator, coll: Value, n: i64) -> Result<(), NativeError> {
    let mut seq = seq_value(eval, &coll).map_err(signal_to_native)?;
    // Owned callers nilled the arg slot: dropping the head here
    // frees memoized chunks as the walk advances (#221).
    drop(coll);
    for _ in 0..n.max(0) {
        if matches!(seq, Value::Nil) {
            return Ok(());
        }
        seq = next_value(eval, &seq).map_err(signal_to_native)?;
    }
    Ok(())
}

fn native_map(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() == 1 {
        return Ok(crate::xform::map_xf(eval, args[0].clone()));
    }
    check_min_arity("clojure.core/map", args, 2)?;
    Ok(map_thunk(args[0].clone(), args[1..].to_vec()))
}

fn native_filter(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/filter", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::filter_xf(
            eval,
            "clojure.core/filter$fn",
            args[0].clone(),
            false,
        ));
    }
    Ok(filter_thunk(args[0].clone(), args[1].clone(), false))
}

fn native_remove(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/remove", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::filter_xf(
            eval,
            "clojure.core/remove$fn",
            args[0].clone(),
            true,
        ));
    }
    Ok(filter_thunk(args[0].clone(), args[1].clone(), true))
}

fn native_take(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/take", args, &[1, 2])?;
    if args.len() == 1 {
        return crate::xform::take_xf(eval, args[0].clone());
    }
    Ok(take_thunk(args[0].clone(), args[1].clone()))
}

fn native_drop(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/drop", args, &[1, 2])?;
    if args.len() == 1 {
        return crate::xform::drop_xf(eval, args[0].clone());
    }
    drop_value(eval, &args[0], args[1].clone()).map_err(signal_to_native)
}

fn native_nthrest(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/nthrest", args, &[2])?;
    nthrest_value(eval, args[0].clone(), &args[1]).map_err(signal_to_native)
}

fn native_take_while(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/take-while", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::take_while_xf(eval, args[0].clone()));
    }
    Ok(take_while_thunk(args[0].clone(), args[1].clone()))
}

fn native_drop_while(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/drop-while", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::drop_while_xf(eval, args[0].clone()));
    }
    Ok(drop_while_thunk(args[0].clone(), args[1].clone()))
}

fn native_concat(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    Ok(concat_thunk(args.to_vec()))
}

fn native_cycle(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/cycle", args, &[1])?;
    cycle_thunk(eval, args[0].clone()).map_err(signal_to_native)
}

fn native_interleave(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(empty_list_value());
    }
    Ok(interleave_thunk(args.to_vec()))
}

fn native_interpose(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/interpose", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::interpose_xf(eval, args[0].clone()));
    }
    Ok(interpose_thunk(args[0].clone(), args[1].clone()))
}

fn native_repeat(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/repeat", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(repeat_thunk(args[0].clone(), None));
    }
    let n = repeat_count(&args[0])?;
    Ok(repeat_thunk(args[1].clone(), Some(n)))
}

fn native_repeatedly(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/repeatedly", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(repeatedly_thunk(args[0].clone()));
    }
    // `(take n (repeatedly f))`: the count stays lazy (validating
    // per force through `take`, ceil rule included).
    Ok(take_thunk(
        args[0].clone(),
        repeatedly_thunk(args[1].clone()),
    ))
}

fn native_iterate(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/iterate", args, &[2])?;
    // Like `Iterate.create`'s `IFn` cast: nil passes (failing on
    // first advance instead), non-callables fail now.
    if !matches!(&args[0], Value::Nil) {
        check_ifn(eval, &args[0]).map_err(signal_to_native)?;
    }
    Ok(iterate_thunk(args[0].clone(), args[1].clone()))
}

fn native_range(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        // `(iterate inc' 0)` (class `Iterate`, not a range).
        0 => Ok(iterate_thunk(deref_core(eval, "inc'"), Value::Int(0))),
        1 => range1(args[0].clone()),
        2 => range2(args[0].clone(), args[1].clone()),
        3 => range3(args[0].clone(), args[1].clone(), args[2].clone()),
        _ => Err(NativeError::arity("clojure.core/range", args.len())),
    }
}

fn native_mapcat(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() == 1 {
        return Ok(crate::xform::mapcat_xf(eval, args[0].clone()));
    }
    check_min_arity("clojure.core/mapcat", args, 2)?;
    Ok(mapcat_thunk(args[0].clone(), args[1..].to_vec()))
}

fn native_keep(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/keep", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::keep_xf(
            eval,
            "clojure.core/keep$fn",
            args[0].clone(),
            false,
        ));
    }
    Ok(keep_thunk(args[0].clone(), args[1].clone()))
}

fn native_keep_indexed(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/keep-indexed", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::keep_xf(
            eval,
            "clojure.core/keep-indexed$fn",
            args[0].clone(),
            true,
        ));
    }
    Ok(keep_indexed_thunk(args[0].clone(), args[1].clone()))
}

fn native_map_indexed(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/map-indexed", args, &[1, 2])?;
    if args.len() == 1 {
        return Ok(crate::xform::map_indexed_xf(eval, args[0].clone()));
    }
    Ok(map_indexed_thunk(args[0].clone(), args[1].clone()))
}

fn native_distinct(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/distinct", args, &[0, 1])?;
    if args.is_empty() {
        return Ok(crate::xform::distinct_xf(eval));
    }
    Ok(distinct_thunk(args[0].clone()))
}

fn native_dedupe(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/dedupe", args, &[0, 1])?;
    if args.is_empty() {
        return Ok(crate::xform::dedupe_xf(eval));
    }
    Ok(dedupe_thunk(args[0].clone()))
}

fn native_partition(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/partition", args, &[2, 3, 4])?;
    let (n, step, pad, coll) = match args.len() {
        2 => (args[0].clone(), args[0].clone(), None, args[1].clone()),
        3 => (args[0].clone(), args[1].clone(), None, args[2].clone()),
        _ => (
            args[0].clone(),
            args[1].clone(),
            Some(args[2].clone()),
            args[3].clone(),
        ),
    };
    Ok(partition_thunk(n, step, pad, coll))
}

fn native_partition_all(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/partition-all", args, &[2, 3])?;
    let (n, step, coll) = match args.len() {
        2 => (args[0].clone(), args[0].clone(), args[1].clone()),
        _ => (args[0].clone(), args[1].clone(), args[2].clone()),
    };
    Ok(partition_all_thunk(n, step, coll))
}

fn native_partition_by(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/partition-by", args, &[2])?;
    Ok(partition_by_thunk(args[0].clone(), args[1].clone()))
}

fn native_take_nth(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/take-nth", args, &[1, 2])?;
    if args.len() == 1 {
        return crate::xform::take_nth_xf(eval, args[0].clone());
    }
    Ok(take_nth_thunk(args[0].clone(), args[1].clone()))
}

fn native_reductions(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/reductions", args, &[2, 3])?;
    if args.len() == 2 {
        return Ok(reductions_thunk(args[0].clone(), None, args[1].clone()));
    }
    if let Some(inner) = as_reduced(&args[1]) {
        return Ok(Value::List(List::from_vec(vec![inner.clone()])));
    }
    Ok(reductions_thunk(
        args[0].clone(),
        Some(args[1].clone()),
        args[2].clone(),
    ))
}

fn native_split_at(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/split-at", args, &[2])?;
    let dropped = drop_value(eval, &args[0], args[1].clone()).map_err(signal_to_native)?;
    Ok(Value::Vector(Vector::from_vec(vec![
        take_thunk(args[0].clone(), args[1].clone()),
        dropped,
    ])))
}

fn native_split_with(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/split-with", args, &[2])?;
    Ok(Value::Vector(Vector::from_vec(vec![
        take_while_thunk(args[0].clone(), args[1].clone()),
        drop_while_thunk(args[0].clone(), args[1].clone()),
    ])))
}

fn native_reduce(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim: correct, retaining (the head stays
    // in the caller's buffer). Owned callers run `owned` directly.
    native_reduce_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_reduce_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/reduce", &args, &[2, 3])?;
    let f = args[0].clone();
    // Eductions reduce without `IReduce` (2-arg takes the JVM cast)
    // and complete through `(completing f)` in 3-arg (oracle
    // `eduction-reduce*`, `core.clj` source).
    let coll = if args.len() == 2 { &args[1] } else { &args[2] };
    if let Some((xform, ecoll)) = crate::xform::Eduction::parts(coll) {
        if args.len() == 2 {
            return Err(NativeError::cast(
                &jvm_class_of(coll),
                "clojure.lang.IReduce",
            ));
        }
        let completing = crate::xform::completing_fn(eval, f.clone(), None);
        return crate::xform::transduce_into(
            eval,
            &xform.clone(),
            &completing,
            args[1].clone(),
            ecoll.clone(),
        );
    }
    // The init moves out (nothing reads `args[1]` again): a kept
    // copy would pin every subtree the first step leaves untouched,
    // so later steps would path-copy instead of updating in place.
    let (mut acc, has_acc) = if args.len() == 3 {
        (std::mem::replace(&mut args[1], Value::Nil), true)
    } else {
        (Value::Nil, false)
    };
    let coll_idx = if args.len() == 3 { 2 } else { 1 };
    let coll = take_walk_coll(&mut args, coll_idx);
    let mut first = true;
    let mut empty = true;
    // The step fn resolves once (#226): a VM fn takes each step's
    // args straight into registers, a native runs its entry.
    let prepared = crate::vm::Prepared::of(&f, 2);
    walk_seq(eval, coll, |eval, head| {
        empty = false;
        if !has_acc && first {
            first = false;
            acc = head.clone();
            return Ok(Walk::Continue);
        }
        first = false;
        // Owned step call (#227): the accumulator moves through
        // (no clone, no borrow pin), so a uniquely-owned acc
        // mutates in place inside an owned `conj` instead of
        // path-copying per step.
        let acc_taken = std::mem::replace(&mut acc, Value::Nil);
        acc = reduce_step(eval, &f, prepared.as_ref(), acc_taken, head)?;
        if let Some(inner) = as_reduced(&acc) {
            acc = inner.clone();
            return Ok(Walk::Stop);
        }
        Ok(Walk::Continue)
    })?;
    if empty && !has_acc {
        return call_checked(eval, &f, &[]);
    }
    Ok(acc)
}

/// One `reduce` step over the moved accumulator: through the
/// prepared callee, else the checked general call.
fn reduce_step(
    eval: &mut Evaluator,
    f: &Value,
    prepared: Option<&crate::vm::Prepared<'_>>,
    acc: Value,
    head: &Value,
) -> Result<Value, NativeError> {
    match prepared {
        Some(p) => p.call2(eval, acc, head.clone()).map_err(signal_to_native),
        None => {
            let mut step_args = SmallVec::new();
            step_args.push(acc);
            step_args.push(head.clone());
            call_checked_owned(eval, f, step_args)
        }
    }
}

fn reduce_kv_step(
    eval: &mut Evaluator,
    f: &Value,
    acc: &mut Value,
    k: Value,
    v: Value,
) -> Result<bool, NativeError> {
    let mut step_args = SmallVec::new();
    step_args.push(std::mem::replace(acc, Value::Nil));
    step_args.push(k);
    step_args.push(v);
    // `core.clj`'s `(f acc k v)` is a plain invoke: a non-fn `f` raises
    // the interpreter's own invoke error, never the `IFn` pre-check's.
    let next = eval.call_owned(f, step_args).map_err(signal_to_native)?;
    if let Some(inner) = as_reduced(&next) {
        *acc = inner.clone();
        return Ok(true);
    }
    *acc = next;
    Ok(false)
}

fn native_reduce_kv(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    native_reduce_kv_owned(eval, args.iter().cloned().collect())
}

fn reduce_kv_vector(
    eval: &mut Evaluator,
    f: &Value,
    acc: &mut Value,
    v: &Vector,
) -> Result<(), NativeError> {
    for (i, elem) in v.iter().enumerate() {
        if reduce_kv_step(eval, f, acc, Value::Int(i as i64), elem)? {
            break;
        }
    }
    Ok(())
}

/// `core.clj`'s index loop for every other `vector?` (subvecs and
/// `vector-of` vectors): `(f acc i (nth coll i))` for `i` below `(count coll)`.
fn reduce_kv_indexed(
    eval: &mut Evaluator,
    f: &Value,
    acc: &mut Value,
    coll: &Value,
) -> Result<(), NativeError> {
    let n = crate::coll::count_of(eval, coll)?;
    for i in 0..n {
        let elem = crate::coll::native_nth(eval, &[coll.clone(), Value::Int(i)])?;
        if reduce_kv_step(eval, f, acc, Value::Int(i), elem)? {
            break;
        }
    }
    Ok(())
}

/// `core.clj`'s `:else` arm: `(reduce (fn [acc e] (f acc (key e) (val e)))
/// init coll)` over the coll's seq (records, entries, lists, sorted maps,
/// and non-colls alike — each head must be a real map entry).
fn reduce_kv_seq(
    eval: &mut Evaluator,
    f: &Value,
    acc: &mut Value,
    coll: Value,
) -> Result<(), NativeError> {
    walk_seq(eval, coll, |eval, head| {
        let k = crate::coll::entry_key(head)?;
        let v = crate::coll::entry_val(head)?;
        if reduce_kv_step(eval, f, acc, k, v)? {
            return Ok(Walk::Stop);
        }
        Ok(Walk::Continue)
    })
}

pub(crate) fn native_reduce_kv_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/reduce-kv", &args, &[3])?;
    let f = args[0].clone();
    let mut acc = std::mem::replace(&mut args[1], Value::Nil);
    let coll = take_walk_coll(&mut args, 2);
    match &coll {
        Value::Nil => Ok(acc),
        Value::Vector(v) => {
            reduce_kv_vector(eval, &f, &mut acc, v)?;
            Ok(acc)
        }
        Value::Map(m) => {
            for (k, v) in m.iter() {
                if reduce_kv_step(eval, &f, &mut acc, k.clone(), v.clone())? {
                    break;
                }
            }
            Ok(acc)
        }
        c if crate::core::is_vector(c) && !matches!(c, Value::MapEntry { .. }) => {
            reduce_kv_indexed(eval, &f, &mut acc, c)?;
            Ok(acc)
        }
        _ => {
            reduce_kv_seq(eval, &f, &mut acc, coll)?;
            Ok(acc)
        }
    }
}

fn native_run_bang(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    native_run_bang_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_run_bang_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/run!", &args, &[2])?;
    let f = args[0].clone();
    let coll = take_walk_coll(&mut args, 1);
    let prepared = crate::vm::Prepared::of(&f, 1);
    walk_seq(eval, coll, |eval, head| {
        let res = match prepared.as_ref() {
            Some(p) => p.call1(eval, head.clone()).map_err(signal_to_native)?,
            // `#(proc %2)` invokes `proc` directly (see `reduce_kv_step`).
            None => eval
                .call(&f, std::slice::from_ref(head))
                .map_err(signal_to_native)?,
        };
        if as_reduced(&res).is_some() {
            return Ok(Walk::Stop);
        }
        Ok(Walk::Continue)
    })?;
    Ok(Value::Nil)
}

fn native_take_last(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_take_last_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_take_last_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/take-last", &args, &[2])?;
    // Exactly the `loop`: `(seq (drop n coll))` validates `n`
    // eagerly, then both sides walk `next` in lockstep (which keeps
    // the result kind-preserving).
    let n = std::mem::replace(&mut args[0], Value::Nil);
    let coll = std::mem::replace(&mut args[1], Value::Nil);
    let mut s = seq_value(eval, &coll).map_err(signal_to_native)?;
    let dropped = drop_value(eval, &n, coll).map_err(signal_to_native)?;
    let mut lead = seq_value(eval, &dropped).map_err(signal_to_native)?;
    while !matches!(lead, Value::Nil) {
        s = next_value(eval, &s).map_err(signal_to_native)?;
        lead = next_value(eval, &lead).map_err(signal_to_native)?;
    }
    Ok(s)
}

fn native_drop_last(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_drop_last_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_drop_last_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/drop-last", &args, &[1, 2])?;
    let (n, coll) = if args.len() == 1 {
        (Value::Int(1), take_walk_coll(&mut args, 0))
    } else {
        let n = std::mem::replace(&mut args[0], Value::Nil);
        let coll = std::mem::replace(&mut args[1], Value::Nil);
        (n, coll)
    };
    let project = native_fn(eval, "clojure.core/drop-last", |_, args: &[Value]| {
        Ok(args.first().cloned().unwrap_or(Value::Nil))
    });
    let dropped = drop_value(eval, &n, coll.clone()).map_err(signal_to_native)?;
    Ok(map_thunk(project, vec![coll, dropped]))
}

fn native_butlast(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_butlast_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_butlast_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/butlast", &args, &[1])?;
    let coll = take_walk_coll(&mut args, 0);
    let items = collect_all(eval, coll).map_err(signal_to_native)?;
    if items.len() < 2 {
        return Ok(Value::Nil);
    }
    // `(seq ret)`: a vector seq (`ChunkedSeq`, counted).
    Ok(realized_value(
        items[..items.len() - 1].to_vec(),
        true,
        true,
        true,
        clj_value::SeqKind::Chunked,
    ))
}

fn native_keys(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/keys", args, &[1])?;
    keys_thunk(eval, args[0].clone(), true).map_err(signal_to_native)
}

fn native_vals(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/vals", args, &[1])?;
    keys_thunk(eval, args[0].clone(), false).map_err(signal_to_native)
}

fn native_reverse(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_reverse_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_reverse_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/reverse", &args, &[1])?;
    let coll = take_walk_coll(&mut args, 0);
    let mut items = collect_all(eval, coll).map_err(signal_to_native)?;
    items.reverse();
    Ok(Value::List(List::from_vec(items)))
}

/// `rseq`: vectors, subvectors, and entries answer their `RSeq`
/// (empty answers nil); nil is the `rev` NPE, anything else the
/// `Reversible` cast (Oracle FX2 `rsub`, `rseq-vec`,
/// `rseq-entry-class`, `rseq-subvec`, `rseq-list`, `rseq-map`,
/// `rseq-nil`, `rseq-empty`).
fn native_rseq(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/rseq", args, &[1])?;
    match &args[0] {
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.Reversible.rseq()\" because \"rev\" is null".to_owned(),
        )),
        Value::Vector(x) => Ok(rseq_of(x.iter().collect())),
        Value::PrimitiveVector(p) => Ok(rseq_of(p.items())),
        Value::MapEntry { key, val, .. } => Ok(rseq_of(vec![(**key).clone(), (**val).clone()])),
        Value::Map(m) if m.is_sorted() => Ok(rseq_of(
            m.iter()
                .map(|(k, v)| clj_value::map_entry(k.clone(), v.clone()))
                .collect(),
        )),
        Value::Set(s) if s.is_sorted() => Ok(rseq_of(s.iter().cloned().collect())),
        other => {
            if let Some(sv) = as_subvector(other) {
                return Ok(rseq_of(sv.iter().collect()));
            }
            Err(cce_err(other, "clojure.lang.Reversible"))
        }
    }
}

/// An `RSeq` over reversed items (counted, never chunked, never
/// `IDrop`; empty answers nil via `realized_value`).
fn rseq_of(mut items: Vec<Value>) -> Value {
    items.reverse();
    realized_value(items, false, true, false, clj_value::SeqKind::RSeq)
}

/// `nthnext`: `(seq (nthrest coll n))` (Oracle FX2 `nthnext`).
fn native_nthnext(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/nthnext", args, &[2])?;
    let rest = nthrest_value(eval, args[0].clone(), &args[1]).map_err(signal_to_native)?;
    seq_value(eval, &rest).map_err(signal_to_native)
}

/// Interpret a sort-comparator result the way `AFunction.compare`
/// does: booleans order (false consults the reversed call),
/// numbers truncate to their sign, anything else fails to
/// `Number`.
pub(crate) fn afunction_order(
    r: Value,
    rev: impl FnOnce() -> Result<Value, NativeError>,
) -> Result<Ordering, NativeError> {
    match r {
        Value::Bool(true) => Ok(Ordering::Less),
        Value::Bool(false) => Ok(if rev()?.is_truthy() {
            Ordering::Greater
        } else {
            Ordering::Equal
        }),
        Value::Int(n) => Ok((n as i32).cmp(&0)),
        Value::Double(d) => {
            let t = d.trunc();
            Ok(if t > 0.0 {
                Ordering::Greater
            } else if t < 0.0 {
                Ordering::Less
            } else {
                Ordering::Equal
            })
        }
        _ => Err(cce_err(&r, "java.lang.Number")),
    }
}

/// A fallible `sort_by` (first failure aborts the order).
fn sort_by_checked(
    items: &mut [Value],
    mut cmp: impl FnMut(&Value, &Value) -> Result<Ordering, NativeError>,
) -> Result<(), NativeError> {
    let mut err: Option<NativeError> = None;
    items.sort_by(|a, b| {
        if err.is_some() {
            return Ordering::Equal;
        }
        match cmp(a, b) {
            Ok(o) => o,
            Err(e) => {
                err = Some(e);
                Ordering::Equal
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// A stable sort of 64-byte items through an index permutation:
/// comparisons and the sort's moves traffic in `usize`s, then
/// each item moves once when the permutation is applied. Small
/// inputs sort in place. The comparator must be total (the
/// specialized lanes below).
fn sort_by_index(items: &mut [Value], cmp: fn(&Value, &Value) -> std::cmp::Ordering) {
    if items.len() < 64 {
        items.sort_by(cmp);
        return;
    }
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by(|&a, &b| cmp(&items[a], &items[b]));
    permute(items, &idx);
}

/// Apply a sort permutation in place (`idx[pos]` is the item that
/// belongs at `pos`): each item moves once.
fn permute(items: &mut [Value], idx: &[usize]) {
    // dest[i]: where the item now at `i` belongs.
    let mut dest = vec![0usize; items.len()];
    for (pos, &src) in idx.iter().enumerate() {
        dest[src] = pos;
    }
    for i in 0..items.len() {
        while dest[i] != i {
            let j = dest[i];
            items.swap(i, j);
            dest.swap(i, j);
        }
    }
}

/// Order two `Int` values (any other pair ties, for the
/// guard-bypassing direct unit tests).
fn cmp_ints(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

/// Sort all-`Int` items as `i64`; `false` when any item is not `Int`.
fn sort_all_ints(items: &mut [Value]) -> bool {
    if !items.iter().all(|v| matches!(v, Value::Int(_))) {
        return false;
    }
    sort_by_index(items, cmp_ints);
    true
}

/// Order two string values by bytes (any other pair ties).
fn cmp_strs(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

/// Sort all-ASCII-string items by bytes (identical to UTF-16 unit
/// order on ASCII); `false` when any item is not an ASCII string.
fn sort_all_ascii_strs(items: &mut [Value]) -> bool {
    if !items
        .iter()
        .all(|v| matches!(v, Value::Str(s) if s.is_ascii()))
    {
        return false;
    }
    sort_by_index(items, cmp_strs);
    true
}

/// Whether `v` is a 2-vector `[ascii-string int]` (borrowed
/// reads: typed storage never holds a mixed pair).
fn is_str_int_pair(v: &Value) -> bool {
    match v {
        Value::Vector(vec) => {
            vec.len() == 2
                && matches!(vec.get_ref(0), Some(Value::Str(s)) if s.is_ascii())
                && matches!(vec.get_ref(1), Some(Value::Int(_)))
        }
        _ => false,
    }
}

/// Order two `[ascii-string int]` pairs (borrowed reads; a
/// typed-storage pair cannot match the `Str` slot either way).
fn cmp_str_int_pair(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (Value::Vector(x), Value::Vector(y)) = (a, b) else {
        return Ordering::Equal;
    };
    let (Some(Value::Str(p)), Some(Value::Str(q))) = (x.get_ref(0), y.get_ref(0)) else {
        return Ordering::Equal;
    };
    match p.cmp(q) {
        Ordering::Equal => match (x.get_ref(1), y.get_ref(1)) {
            (Some(Value::Int(u)), Some(Value::Int(v))) => u.cmp(v),
            _ => Ordering::Equal,
        },
        ord => ord,
    }
}

/// Sort all-`[ascii-string int]` pair items; `false` otherwise.
fn sort_str_int_pairs(items: &mut [Value]) -> bool {
    if items.is_empty() || !items.iter().all(is_str_int_pair) {
        return false;
    }
    sort_by_index(items, cmp_str_int_pair);
    true
}

/// Whether `v` is a vector whose items are all `Int`.
fn is_int_vector(v: &Value) -> bool {
    match v {
        Value::Vector(vec) => vec.iter().all(|e| matches!(e, Value::Int(_))),
        _ => false,
    }
}

/// Order two all-`Int` vectors as `compare` does: the shorter
/// first, else itemwise.
fn cmp_int_vectors(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (Value::Vector(x), Value::Vector(y)) = (a, b) else {
        return Ordering::Equal;
    };
    if x.len() != y.len() {
        return x.len().cmp(&y.len());
    }
    for (u, w) in x.iter().zip(y.iter()) {
        if let (Value::Int(p), Value::Int(q)) = (&u, &w) {
            match p.cmp(q) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
    }
    Ordering::Equal
}

/// Sort all-int-vector items; `false` otherwise.
fn sort_int_vectors(items: &mut [Value]) -> bool {
    if items.is_empty() || !items.iter().all(is_int_vector) {
        return false;
    }
    sort_by_index(items, cmp_int_vectors);
    true
}

/// The longest vectors [`sort_int_tuples`] keys.
const MAX_TUPLE: usize = 8;

/// Sort vectors of one length (at most [`MAX_TUPLE`]) whose items
/// are all `Int` by a flat copy of their items, read once: each
/// comparison is then a slice compare, which is `compare`'s order
/// on vectors of one length. Stable, like the other lanes; `false`
/// for anything else.
fn sort_int_tuples(items: &mut [Value]) -> bool {
    let k = match items.first() {
        Some(Value::Vector(v)) if (1..=MAX_TUPLE).contains(&v.len()) => v.len(),
        _ => return false,
    };
    let mut keys: Vec<i64> = Vec::with_capacity(items.len() * k);
    for v in items.iter() {
        let Value::Vector(vec) = v else {
            return false;
        };
        if vec.len() != k {
            return false;
        }
        for i in 0..k {
            match vec.read_at(i, int_item) {
                Some(n) => keys.push(n),
                None => return false,
            }
        }
    }
    let key = |i: usize| &keys[i * k..(i + 1) * k];
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by(|&a, &b| key(a).cmp(key(b)));
    permute(items, &idx);
    true
}

/// An `Int` item's value.
fn int_item(v: &Value) -> Option<i64> {
    match v {
        Value::Int(n) => Some(*n),
        _ => None,
    }
}

/// Default `sort` order: homogeneous fast paths first, then the
/// generic `jvm_compare` machinery for mixed or other items.
fn sort_default(items: &mut [Value]) -> Result<(), NativeError> {
    if sort_all_ints(items)
        || sort_all_ascii_strs(items)
        || sort_str_int_pairs(items)
        || sort_int_tuples(items)
        || sort_int_vectors(items)
    {
        return Ok(());
    }
    sort_by_checked(items, jvm_compare)
}

fn native_sort(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_sort_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_sort_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/sort", &args, &[1, 2])?;
    let comp = if args.len() == 1 {
        None
    } else {
        Some(args[0].clone())
    };
    let coll_idx = if args.len() == 1 { 0 } else { 1 };
    let coll = take_walk_coll(&mut args, coll_idx);
    let mut items = collect_all(eval, coll).map_err(signal_to_native)?;
    match comp {
        None => sort_default(&mut items)?,
        Some(comp) => {
            sort_by_checked(&mut items, |a, b| {
                let r = call_checked(eval, &comp, &[a.clone(), b.clone()])?;
                afunction_order(r, || call_checked(eval, &comp, &[b.clone(), a.clone()]))
            })?;
        }
    }
    // `ArraySeq` (counted, never `IDrop`, no metadata);
    // empty/nil inputs answer `()`.
    if items.is_empty() {
        return Ok(empty_list_value());
    }
    Ok(realized_value(
        items,
        false,
        true,
        false,
        clj_value::SeqKind::Array,
    ))
}

fn native_sort_by(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_sort_by_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_sort_by_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/sort-by", &args, &[2, 3])?;
    let keyfn = args[0].clone();
    let comp = if args.len() == 2 {
        deref_core(eval, "compare")
    } else {
        args[1].clone()
    };
    let coll_idx = if args.len() == 2 { 1 } else { 2 };
    let coll = take_walk_coll(&mut args, coll_idx);
    let mut items = collect_all(eval, coll).map_err(signal_to_native)?;
    sort_by_checked(&mut items, |a, b| {
        // `(comp (keyfn x) (keyfn y))` directly (the keyfn runs per
        // comparison; no int mapping).
        let ka = call_checked(eval, &keyfn, std::slice::from_ref(a))?;
        let kb = call_checked(eval, &keyfn, std::slice::from_ref(b))?;
        let r = call_checked(eval, &comp, &[ka.clone(), kb.clone()])?;
        afunction_order(r, || call_checked(eval, &comp, &[kb, ka]))
    })?;
    if items.is_empty() {
        return Ok(empty_list_value());
    }
    Ok(realized_value(
        items,
        false,
        true,
        false,
        clj_value::SeqKind::Array,
    ))
}

fn native_dorun(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_dorun_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_dorun_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/dorun", &args, &[1, 2])?;
    if args.len() == 1 {
        let coll = take_walk_coll(&mut args, 0);
        walk_seq(eval, coll, |_, _| Ok(Walk::Continue))?;
        return Ok(Value::Nil);
    }
    let n = take_count(&args[0])?;
    let coll = take_walk_coll(&mut args, 1);
    force_n(eval, coll, n)?;
    Ok(Value::Nil)
}

fn native_doall(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_doall_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_doall_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/doall", &args, &[1, 2])?;
    // `doall` returns the (realized) head, so the walk clones it
    // (the result legitimately holds the head, like the JVM).
    if args.len() == 1 {
        let coll = take_walk_coll(&mut args, 0);
        walk_seq(eval, coll.clone(), |_, _| Ok(Walk::Continue))?;
        return Ok(coll);
    }
    let n = take_count(&args[0])?;
    let coll = take_walk_coll(&mut args, 1);
    force_n(eval, coll.clone(), n)?;
    Ok(coll)
}

fn native_mapv(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_mapv_owned(eval, args.iter().cloned().collect())
}

fn parse_long_items<I: Iterator<Item = Value>>(iter: I, cap: usize) -> Option<Value> {
    let mut ints: Vec<i64> = Vec::with_capacity(cap);
    let mut mixed: Option<Vec<Value>> = None;
    for item in iter {
        let Value::Str(s) = item else {
            return None;
        };
        if let Some(m) = &mut mixed {
            m.push(
                crate::core::parse_long_str(&s)
                    .map(Value::Int)
                    .unwrap_or(Value::Nil),
            );
        } else if let Some(n) = crate::core::parse_long_str(&s) {
            ints.push(n);
        } else {
            let mut m: Vec<Value> = Vec::with_capacity(cap);
            m.extend(ints.drain(..).map(Value::Int));
            m.push(Value::Nil);
            mixed = Some(m);
        }
    }
    Some(Value::Vector(match mixed {
        Some(m) => Vector::from_vec(m),
        None => Vector::from_flat_ints(ints),
    }))
}

fn try_mapv_parse_long(v: &Value) -> Option<Value> {
    match v {
        Value::Vector(vec) => parse_long_items(vec.iter(), vec.len()),
        Value::Extension(_) => {
            if let Some(sp) = crate::lazy::as_splitvec(v) {
                return parse_long_pieces(sp);
            }
            let sv = as_subvector(v)?;
            parse_long_items(sv.iter(), sv.len())
        }
        _ => None,
    }
}

/// `(mapv parse-long split-vec)`: parse the spans in place, no
/// piece string is ever materialized.
fn parse_long_pieces(sp: &clj_value::SplitVecValue) -> Option<Value> {
    let cap = sp.len();
    let mut ints: Vec<i64> = Vec::with_capacity(cap);
    let mut mixed: Option<Vec<Value>> = None;
    for i in 0..cap {
        let s = sp.piece_str(i)?;
        if let Some(m) = &mut mixed {
            m.push(crate::core::parse_long_str(s).map_or(Value::Nil, Value::Int));
        } else if let Some(n) = crate::core::parse_long_str(s) {
            ints.push(n);
        } else {
            let mut m: Vec<Value> = Vec::with_capacity(cap);
            m.extend(ints.drain(..).map(Value::Int));
            m.push(Value::Nil);
            mixed = Some(m);
        }
    }
    Some(Value::Vector(match mixed {
        Some(m) => Vector::from_vec(m),
        None => Vector::from_flat_ints(ints),
    }))
}

pub(crate) fn native_mapv_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/mapv", &args, 2)?;
    // The inputs move into the thunk (drained from `args`), so the
    // walk holds the only handles and input heads free as it
    // advances (#221).
    let inputs: Vec<Value> = args.drain(1..).collect();
    // `(mapv parse-long string-vector)` parses in one Rust loop:
    // token-heavy readers otherwise pay a prepared call per token.
    // A non-string item falls through to the generic paths, which
    // raise the same cast error `parse-long` would.
    if inputs.len() == 1
        && crate::ext::as_native(&args[0]).is_some_and(|n| &*n.display == "clojure.core/parse-long")
    {
        if let Some(res) = try_mapv_parse_long(&inputs[0]) {
            return Ok(res);
        }
    }
    // One input under a VM fn walks it in place (#226): no lazy map
    // in between, the fn resolved once.
    if let ([_], Some(p)) = (&inputs[..], crate::vm::Prepared::of(&args[0], 1)) {
        let input = inputs.into_iter().next().unwrap_or(Value::Nil);
        let mut tv = clj_value::TransientVector::new();
        walk_seq(eval, input, |eval, head| {
            let out = p.call1(eval, head.clone()).map_err(signal_to_native)?;
            tv.conj(out).expect("fresh transient accepts conj");
            Ok(Walk::Continue)
        })?;
        let vec = tv.persistent().expect("fresh transient is valid at freeze");
        return Ok(Value::Vector(vec));
    }
    let mapped = map_thunk(args[0].clone(), inputs);
    // Packed streaming (#227): no intermediate `Vec<Value>`, and
    // all-`Int` results freeze to packed storage zero-copy.
    let vec = collect_packed(eval, mapped).map_err(signal_to_native)?;
    Ok(Value::Vector(vec))
}

fn filterv_vector(eval: &mut Evaluator, f: &Value, v: Vector) -> Result<Value, NativeError> {
    if v.is_empty() {
        return Ok(Value::Vector(v.with_meta(None)));
    }
    crate::core::check_ifn(eval, f).map_err(signal_to_native)?;
    let prep = crate::vm::Prepared::of(f, 1);
    let mut tv: Option<clj_value::TransientVector> = None;
    for i in 0..v.len() {
        let item = v.get(i).expect("index in bounds");
        let pass = match &prep {
            Some(p) => p.call1(eval, item.clone()).map(|r| r.is_truthy()),
            None => crate::lazy::filter_pred(eval, f, &item),
        }
        .map_err(signal_to_native)?;
        if let Some(tv) = &mut tv {
            if pass {
                tv.conj(item).expect("fresh transient accepts conj");
            }
        } else if !pass {
            let mut new_tv = clj_value::TransientVector::new();
            for j in 0..i {
                new_tv
                    .conj(v.get(j).expect("index in bounds"))
                    .expect("fresh transient accepts conj");
            }
            tv = Some(new_tv);
        }
    }
    Ok(Value::Vector(match tv {
        Some(mut tv) => tv.persistent().expect("fresh transient is valid at freeze"),
        None => v.with_meta(None),
    }))
}

fn native_filterv(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_filterv_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_filterv_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/filterv", &args, &[2])?;
    let input = take_walk_coll(&mut args, 1);
    if let Value::Vector(v) = input {
        return filterv_vector(eval, &args[0], v);
    }
    // Split vectors materialize once into the vector lane (the
    // lazy filter thunk steps per item through pending machinery).
    if let Some(sp) = crate::lazy::as_splitvec(&input) {
        let v = clj_value::Vector::from_vec(sp.items());
        return filterv_vector(eval, &args[0], v);
    }
    let filtered = filter_thunk(args[0].clone(), input, false);
    // Packed streaming (#227): no intermediate `Vec<Value>`, and
    // all-`Int` results freeze to packed storage zero-copy.
    let vec = collect_packed(eval, filtered).map_err(signal_to_native)?;
    Ok(Value::Vector(vec))
}

fn native_group_by(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_group_by_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_group_by_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/group-by", &args, &[2])?;
    let f = args[0].clone();
    let mut groups = new_map();
    let coll = take_walk_coll(&mut args, 1);
    walk_seq(eval, coll, |eval, head| {
        let key = call_checked(eval, &f, std::slice::from_ref(head))?;
        let key = crate::lazy::realize_key(eval, &key).map_err(signal_to_native)?;
        let group = match groups.get(&key) {
            Some(Value::Vector(v)) => v.clone(),
            _ => Vector::from_vec(Vec::new()),
        };
        // In place (#227): no `mem::replace` dummy map per item.
        groups.insert_mut(key, Value::Vector(group.push_back(head.clone())));
        Ok(Walk::Continue)
    })?;
    Ok(Value::Map(groups))
}

fn native_frequencies(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_frequencies_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_frequencies_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/frequencies", &args, &[1])?;
    let mut counts = new_map();
    let coll = take_walk_coll(&mut args, 0);
    walk_seq(eval, coll, |eval, head| {
        let head = crate::lazy::realize_key(eval, head).map_err(signal_to_native)?;
        let count = match counts.get(&head) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        // In place (#227): no `mem::replace` dummy map per item.
        counts.insert_mut(head, Value::Int(count + 1));
        Ok(Walk::Continue)
    })?;
    Ok(Value::Map(counts))
}

fn native_every_pred(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/every-pred", args, 1)?;
    let preds = args.to_vec();
    Ok(native_fn(
        eval,
        "clojure.core/every-pred",
        move |eval, args| every_pred_apply(eval, &preds, args),
    ))
}

/// The combined `every-pred`: every pred truthy on every arg
/// (vacuous `true` on zero args; the first three args run
/// pred-major, the rest dispatch on the pred count).
fn every_pred_apply(
    eval: &mut Evaluator,
    preds: &[Value],
    args: &[Value],
) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Bool(true));
    }
    let fixed = args.len().min(3);
    if !every_pred_fixed(eval, preds, &args[..fixed])? {
        return Ok(Value::Bool(false));
    }
    let rest_ok = if preds.len() <= 3 {
        every_pred_rest_by_arg(eval, preds, &args[fixed..])?
    } else {
        every_pred_rest_by_pred(eval, preds, &args[fixed..])?
    };
    Ok(Value::Bool(rest_ok))
}

/// Every pred truthy on each fixed arg (pred-major).
fn every_pred_fixed(
    eval: &mut Evaluator,
    preds: &[Value],
    fixed: &[Value],
) -> Result<bool, NativeError> {
    for p in preds {
        for a in fixed {
            if !call_checked(eval, p, std::slice::from_ref(a))?.is_truthy() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Every pred truthy on each rest arg (arg-major: ≤3 preds).
fn every_pred_rest_by_arg(
    eval: &mut Evaluator,
    preds: &[Value],
    rest: &[Value],
) -> Result<bool, NativeError> {
    for a in rest {
        for p in preds {
            if !call_checked(eval, p, std::slice::from_ref(a))?.is_truthy() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Every pred truthy on each rest arg (pred-major: >3 preds).
fn every_pred_rest_by_pred(
    eval: &mut Evaluator,
    preds: &[Value],
    rest: &[Value],
) -> Result<bool, NativeError> {
    for p in preds {
        for a in rest {
            if !call_checked(eval, p, std::slice::from_ref(a))?.is_truthy() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn native_some_fn(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/some-fn", args, 1)?;
    let preds = args.to_vec();
    Ok(native_fn(
        eval,
        "clojure.core/some-fn",
        move |eval, args| some_fn_apply(eval, &preds, args),
    ))
}

/// The combined `some-fn`: the first truthy pred result (`nil` on
/// zero args or no hit; the first three args run pred-major, the
/// rest dispatch on the pred count).
fn some_fn_apply(
    eval: &mut Evaluator,
    preds: &[Value],
    args: &[Value],
) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Nil);
    }
    let fixed = args.len().min(3);
    if let Some(hit) = some_fn_fixed(eval, preds, &args[..fixed])? {
        return Ok(hit);
    }
    if preds.len() <= 3 {
        some_fn_rest_by_arg(eval, preds, &args[fixed..])
    } else {
        some_fn_rest_by_pred(eval, preds, &args[fixed..])
    }
}

/// The first truthy pred result over the fixed args (pred-major).
fn some_fn_fixed(
    eval: &mut Evaluator,
    preds: &[Value],
    fixed: &[Value],
) -> Result<Option<Value>, NativeError> {
    for p in preds {
        for a in fixed {
            let r = call_checked(eval, p, std::slice::from_ref(a))?;
            if r.is_truthy() {
                return Ok(Some(r));
            }
        }
    }
    Ok(None)
}

/// The first truthy pred result over the rest args (arg-major: ≤3
/// preds; `nil` when nothing hits).
fn some_fn_rest_by_arg(
    eval: &mut Evaluator,
    preds: &[Value],
    rest: &[Value],
) -> Result<Value, NativeError> {
    for a in rest {
        for p in preds {
            let r = call_checked(eval, p, std::slice::from_ref(a))?;
            if r.is_truthy() {
                return Ok(r);
            }
        }
    }
    Ok(Value::Nil)
}

/// The first truthy pred result over the rest args (pred-major: >3
/// preds; `nil` when nothing hits).
fn some_fn_rest_by_pred(
    eval: &mut Evaluator,
    preds: &[Value],
    rest: &[Value],
) -> Result<Value, NativeError> {
    for p in preds {
        for a in rest {
            let r = call_checked(eval, p, std::slice::from_ref(a))?;
            if r.is_truthy() {
                return Ok(r);
            }
        }
    }
    Ok(Value::Nil)
}

fn native_max_key(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/max-key", args, 2)?;
    let k = args[0].clone();
    let mut best = args[1].clone();
    let mut kbest = call_checked(eval, &k, std::slice::from_ref(&best))?;
    for cand in &args[2..] {
        let kcand = call_checked(eval, &k, std::slice::from_ref(cand))?;
        let ge = deref_core(eval, ">=");
        if call_checked(eval, &ge, &[kcand.clone(), kbest.clone()])?.is_truthy() {
            best = cand.clone();
            kbest = kcand;
        }
    }
    Ok(best)
}

fn native_min_key(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/min-key", args, 2)?;
    let k = args[0].clone();
    let mut best = args[1].clone();
    let mut kbest = call_checked(eval, &k, std::slice::from_ref(&best))?;
    for cand in &args[2..] {
        let kcand = call_checked(eval, &k, std::slice::from_ref(cand))?;
        let le = deref_core(eval, "<=");
        if call_checked(eval, &le, &[kcand.clone(), kbest.clone()])?.is_truthy() {
            best = cand.clone();
            kbest = kcand;
        }
    }
    Ok(best)
}

fn native_comp(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(deref_core(eval, "identity"));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    let fns = args.to_vec();
    Ok(native_fn(eval, "clojure.core/comp", move |eval, args| {
        let mut val = call_checked(eval, &fns[fns.len() - 1], args)?;
        for f in fns[..fns.len() - 1].iter().rev() {
            val = call_checked(eval, f, &[val])?;
        }
        Ok(val)
    }))
}

fn native_partial(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/partial", args, 1)?;
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    let f = args[0].clone();
    let bound = args[1..].to_vec();
    Ok(native_fn(
        eval,
        "clojure.core/partial",
        move |eval, args| {
            let mut full = bound.clone();
            full.extend(args.iter().cloned());
            call_checked(eval, &f, &full)
        },
    ))
}

fn native_complement(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/complement", args, &[1])?;
    let f = args[0].clone();
    Ok(native_fn(
        eval,
        "clojure.core/complement",
        move |eval, args| Ok(Value::Bool(!call_checked(eval, &f, args)?.is_truthy())),
    ))
}

fn native_juxt(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/juxt", args, 1)?;
    let fns = args.to_vec();
    Ok(native_fn(eval, "clojure.core/juxt", move |eval, args| {
        let mut out = Vec::with_capacity(fns.len());
        for f in &fns {
            out.push(call_checked(eval, f, args)?);
        }
        Ok(Value::Vector(Vector::from_vec(out)))
    }))
}

fn native_constantly(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/constantly", args, &[1])?;
    let x = args[0].clone();
    Ok(native_fn(eval, "clojure.core/constantly", move |_, _| {
        Ok(x.clone())
    }))
}

fn native_identity(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/identity", args, &[1])?;
    Ok(args[0].clone())
}

fn native_fnil(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/fnil", args, &[2, 3, 4])?;
    let f = args[0].clone();
    let defaults = args[1..].to_vec();
    let min = if defaults.len() >= 2 { 2 } else { 1 };
    Ok(native_fn(eval, "clojure.core/fnil", move |eval, args| {
        if args.len() < min {
            return Err(NativeError::arity("clojure.core/fnil", args.len()));
        }
        let filled: Vec<Value> = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if matches!(a, Value::Nil) {
                    defaults.get(i).cloned().unwrap_or(Value::Nil)
                } else {
                    a.clone()
                }
            })
            .collect();
        call_checked(eval, &f, &filled)
    }))
}

fn call_pred1(
    eval: &mut Evaluator,
    pred: &Value,
    prep: Option<&crate::vm::Prepared<'_>>,
    head: &Value,
) -> Result<Value, NativeError> {
    if let Some(p) = prep {
        return p
            .call1(eval, crate::vm::clone_value(head))
            .map_err(signal_to_native);
    }
    let slice = std::slice::from_ref(head);
    if let Some(v) = crate::vm::invoke_lookup(pred, slice) {
        return Ok(v);
    }
    call_checked(eval, pred, slice)
}

fn native_every_p(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_every_p_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_every_p_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/every?", &args, &[2])?;
    let pred = args[0].clone();
    let prep = crate::vm::Prepared::of(&pred, 1);
    let mut all = true;
    let coll = take_walk_coll(&mut args, 1);
    walk_seq(eval, coll, |eval, head| {
        if !call_pred1(eval, &pred, prep.as_ref(), head)?.is_truthy() {
            all = false;
            return Ok(Walk::Stop);
        }
        Ok(Walk::Continue)
    })?;
    Ok(Value::Bool(all))
}

fn native_some(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_some_owned(eval, args.iter().cloned().collect())
}

fn some_kw_step(found: &mut Value, kw: &Value, head: &Value) -> Walk {
    match head {
        Value::Map(m) => {
            if let Some(v) = m.get(kw) {
                if v.is_truthy() {
                    *found = v.clone();
                    return Walk::Stop;
                }
            }
        }
        _ => {
            if let Some(v) = crate::vm_intrinsics::get_of(head, kw, None) {
                if v.is_truthy() {
                    *found = v;
                    return Walk::Stop;
                }
            }
        }
    }
    Walk::Continue
}

pub(crate) fn native_some_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/some", &args, &[2])?;
    let pred = args[0].clone();
    let mut found = Value::Nil;
    let coll = take_walk_coll(&mut args, 1);
    if let Value::Keyword(_) = &pred {
        walk_seq(eval, coll, |_eval, head| {
            Ok(some_kw_step(&mut found, &pred, head))
        })?;
        return Ok(found);
    }
    let prep = crate::vm::Prepared::of(&pred, 1);
    walk_seq(eval, coll, |eval, head| {
        let r = call_pred1(eval, &pred, prep.as_ref(), head)?;
        if r.is_truthy() {
            found = r;
            return Ok(Walk::Stop);
        }
        Ok(Walk::Continue)
    })?;
    Ok(found)
}

fn native_not_any_p(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_not_any_p_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_not_any_p_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/not-any?", &args, &[2])?;
    let pred = args[0].clone();
    let prep = crate::vm::Prepared::of(&pred, 1);
    let mut none = true;
    let coll = take_walk_coll(&mut args, 1);
    walk_seq(eval, coll, |eval, head| {
        if call_pred1(eval, &pred, prep.as_ref(), head)?.is_truthy() {
            none = false;
            return Ok(Walk::Stop);
        }
        Ok(Walk::Continue)
    })?;
    Ok(Value::Bool(none))
}

fn native_not_every_p(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_not_every_p_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_not_every_p_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/not-every?", &args, &[2])?;
    let pred = args[0].clone();
    let prep = crate::vm::Prepared::of(&pred, 1);
    let mut all = true;
    let coll = take_walk_coll(&mut args, 1);
    walk_seq(eval, coll, |eval, head| {
        if !call_pred1(eval, &pred, prep.as_ref(), head)?.is_truthy() {
            all = false;
            return Ok(Walk::Stop);
        }
        Ok(Walk::Continue)
    })?;
    Ok(Value::Bool(!all))
}

/// The S5D seq + higher-order natives installed into
/// `clojure.core` by [`boot`](crate::ns::boot) (`concat` here
/// shadows the eager S5C install with the lazy seq).
pub(crate) const SEQS_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("map", native_map),
    ("filter", native_filter),
    ("remove", native_remove),
    ("take", native_take),
    ("drop", native_drop),
    ("nthrest", native_nthrest),
    ("nthnext", native_nthnext),
    ("take-while", native_take_while),
    ("drop-while", native_drop_while),
    ("take-last", native_take_last),
    ("drop-last", native_drop_last),
    ("butlast", native_butlast),
    ("split-at", native_split_at),
    ("split-with", native_split_with),
    ("concat", native_concat),
    ("cycle", native_cycle),
    ("interleave", native_interleave),
    ("interpose", native_interpose),
    ("repeat", native_repeat),
    ("repeatedly", native_repeatedly),
    ("iterate", native_iterate),
    ("range", native_range),
    ("mapcat", native_mapcat),
    ("keep", native_keep),
    ("keep-indexed", native_keep_indexed),
    ("map-indexed", native_map_indexed),
    ("distinct", native_distinct),
    ("dedupe", native_dedupe),
    ("partition", native_partition),
    ("partition-all", native_partition_all),
    ("partition-by", native_partition_by),
    ("take-nth", native_take_nth),
    ("reductions", native_reductions),
    ("keys", native_keys),
    ("vals", native_vals),
    ("reduce", native_reduce),
    ("reduce-kv", native_reduce_kv),
    ("reverse", native_reverse),
    ("run!", native_run_bang),
    ("rseq", native_rseq),
    ("sort", native_sort),
    ("sort-by", native_sort_by),
    ("dorun", native_dorun),
    ("doall", native_doall),
    ("mapv", native_mapv),
    ("filterv", native_filterv),
    ("group-by", native_group_by),
    ("frequencies", native_frequencies),
    ("every-pred", native_every_pred),
    ("some-fn", native_some_fn),
    ("max-key", native_max_key),
    ("min-key", native_min_key),
    ("comp", native_comp),
    ("partial", native_partial),
    ("complement", native_complement),
    ("juxt", native_juxt),
    ("constantly", native_constantly),
    ("identity", native_identity),
    ("fnil", native_fnil),
    ("every?", native_every_p),
    ("some", native_some),
    ("not-any?", native_not_any_p),
    ("not-every?", native_not_every_p),
];

#[cfg(test)]
mod sort_fast_path_tests {
    use super::{
        MAX_TUPLE, cmp_int_vectors, cmp_ints, cmp_str_int_pair, cmp_strs, sort_by_index,
        sort_int_tuples,
    };
    use clj_value::{Value, Vector};
    use std::cmp::Ordering;
    use std::sync::Arc;

    fn ints(xs: &[i64]) -> Value {
        Value::Vector(Vector::from_vec(
            xs.iter().map(|n| Value::Int(*n)).collect(),
        ))
    }

    #[test]
    fn defensive_arms_tie() {
        assert_eq!(cmp_ints(&Value::Int(1), &Value::Int(2)), Ordering::Less);
        assert_eq!(
            cmp_ints(&Value::Bool(true), &Value::Int(2)),
            Ordering::Equal
        );
        assert_eq!(
            cmp_strs(&Value::Str(Arc::from("b")), &Value::Str(Arc::from("a"))),
            Ordering::Greater
        );
        assert_eq!(
            cmp_strs(&Value::Int(1), &Value::Str(Arc::from("a"))),
            Ordering::Equal
        );
        assert_eq!(
            cmp_str_int_pair(&Value::Int(1), &Value::Int(2)),
            Ordering::Equal
        );
        assert_eq!(
            cmp_str_int_pair(&ints(&[1, 2]), &ints(&[1, 3])),
            Ordering::Equal
        );
        let p = |s: &str, n: &str| {
            Value::Vector(Vector::from_vec(vec![
                Value::Str(Arc::from(s)),
                Value::Str(Arc::from(n)),
            ]))
        };
        assert_eq!(
            cmp_str_int_pair(&p("a", "x"), &p("a", "y")),
            Ordering::Equal
        );
        assert_eq!(
            cmp_int_vectors(&Value::Int(1), &Value::Int(2)),
            Ordering::Equal
        );
        assert_eq!(
            cmp_int_vectors(&ints(&[1, 2]), &ints(&[1, 3])),
            Ordering::Less
        );
        assert_eq!(cmp_int_vectors(&ints(&[1]), &ints(&[1, 0])), Ordering::Less);
        let strs = |s: &str| Value::Vector(Vector::from_vec(vec![Value::Str(Arc::from(s))]));
        assert_eq!(cmp_int_vectors(&strs("a"), &strs("b")), Ordering::Equal);
    }

    /// Keyed tuples sort as the int-vector lane does on vectors of
    /// one length, stably: duplicate keys keep input order (every
    /// item is a distinct vector, told apart by its identity, which
    /// `=` ignores); every other shape falls through.
    #[test]
    fn int_tuples_match_the_vector_lane() {
        let item = |i: usize| ints(&[(i as i64 * 7) % 5, (i as i64 * 3) % 4, i as i64 % 2]);
        let ids = |items: &[Value]| -> Vec<u64> {
            let id = |v: &Value| if let Value::Vector(v) = v { v.id() } else { 0 };
            items.iter().map(id).collect()
        };
        for n in [1usize, 2, 63, 64, 500] {
            let mut keyed: Vec<Value> = (0..n).map(item).collect();
            let mut lane = keyed.clone();
            assert!(sort_int_tuples(&mut keyed));
            lane.sort_by(cmp_int_vectors);
            assert_eq!(keyed, lane, "n={n}");
            assert_eq!(ids(&keyed), ids(&lane), "stable n={n}");
        }
        let long = vec![0i64; MAX_TUPLE + 1];
        for mut items in [
            vec![],
            vec![ints(&[])],
            vec![ints(&long)],
            vec![ints(&[1, 2]), ints(&[1])],
            vec![ints(&[1]), Value::Int(1)],
            vec![
                ints(&[1]),
                Value::Vector(Vector::from_vec(vec![Value::Bool(true)])),
            ],
            vec![Value::Int(1)],
        ] {
            assert!(!sort_int_tuples(&mut items));
        }
    }

    /// The index permutation applies the same stable order as a
    /// direct sort (duplicate keys keep input order), above and
    /// below the in-place threshold.
    #[test]
    fn index_sort_matches_direct_sort() {
        fn pair(word: String, tag: i64) -> Value {
            Value::Vector(Vector::from_vec(vec![
                Value::Str(Arc::from(word)),
                Value::Int(tag),
            ]))
        }
        for n in [0usize, 1, 63, 64, 65, 500] {
            let mut by_index: Vec<Value> = (0..n)
                .map(|i| pair(format!("w{}", (i * 37) % 11), i as i64))
                .collect();
            let mut direct = by_index.clone();
            sort_by_index(&mut by_index, cmp_str_int_pair);
            direct.sort_by(cmp_str_int_pair);
            assert_eq!(by_index, direct, "n={n}");
        }
    }
}

#[cfg(test)]
mod sole_map_walk_tests {
    use super::{Walk, walk_seq};
    use crate::error::NativeError;
    use crate::eval::Evaluator;
    use crate::lazy::{collect_packed, filter_thunk, map_thunk, seq_value};
    use clj_value::{Value, Vector};

    fn source(eval: &mut Evaluator, text: &str) -> Value {
        eval.eval_source(text)
            .expect("evaluates")
            .pop()
            .expect("a value")
    }

    fn ints(xs: &[i64]) -> Value {
        Value::Vector(Vector::from_vec(
            xs.iter().map(|n| Value::Int(*n)).collect(),
        ))
    }

    /// The items a walk visits (stopping after `stop_after`).
    fn walked(eval: &mut Evaluator, coll: Value, stop_after: usize) -> Result<Value, NativeError> {
        let mut out = Vec::new();
        walk_seq(eval, coll, |_, v| {
            out.push(v.clone());
            Ok(if out.len() >= stop_after {
                Walk::Stop
            } else {
                Walk::Continue
            })
        })?;
        Ok(Value::Vector(Vector::from_vec(out)))
    }

    /// #271: a sole unforced `map` seq walks without building its lazy
    /// cells; the items, early stops, and errors are the seq's own.
    #[test]
    fn sole_map_seqs_walk_like_their_cells() {
        let mut eval = crate::ns::boot();
        let inc = source(&mut eval, "inc");
        let add = source(&mut eval, "+");
        let list = source(&mut eval, "'(1 2)");
        let odd = source(&mut eval, "odd?");
        let kw = source(&mut eval, "[:a]");
        let mut walk = |coll: Value, stop: usize| walked(&mut eval, coll, stop).expect("walks");
        // One input: vector chunks, list items, nothing; an early stop.
        assert_eq!(
            walk(map_thunk(inc.clone(), vec![ints(&[1, 2, 3])]), 9),
            ints(&[2, 3, 4])
        );
        assert_eq!(
            walk(map_thunk(inc.clone(), vec![list.clone()]), 9),
            ints(&[2, 3])
        );
        assert_eq!(walk(map_thunk(inc.clone(), vec![ints(&[])]), 9), ints(&[]));
        assert_eq!(
            walk(map_thunk(inc.clone(), vec![ints(&[1, 2, 3])]), 1),
            ints(&[2])
        );
        // Vector inputs: one call per item, the shortest bounds the walk.
        let both = || map_thunk(add.clone(), vec![ints(&[1, 2, 3]), ints(&[10, 20])]);
        assert_eq!(walk(both(), 9), ints(&[11, 22]));
        assert_eq!(walk(both(), 1), ints(&[11]));
        // Not sole: shared, forced, other thunks, a non-vector input among
        // several, a plain list — each walks through the seq itself.
        let shared = map_thunk(inc.clone(), vec![ints(&[1])]);
        assert_eq!(walk(shared.clone(), 9), ints(&[2]));
        drop(shared);
        assert_eq!(
            walk(filter_thunk(odd, ints(&[1, 2, 3]), false), 9),
            ints(&[1, 3])
        );
        assert_eq!(
            walk(map_thunk(add.clone(), vec![ints(&[1, 2]), list.clone()]), 9),
            ints(&[2, 4])
        );
        assert_eq!(walk(list.clone(), 9), ints(&[1, 2]));
        let forced = map_thunk(inc.clone(), vec![ints(&[1])]);
        seq_value(&mut eval, &forced).expect("forces");
        assert_eq!(walked(&mut eval, forced, 9).expect("walks"), ints(&[2]));
        // The map fn's errors and the visitor's surface.
        assert!(walked(&mut eval, map_thunk(inc.clone(), vec![kw]), 9).is_err());
        let mut calls = 0;
        let failed = walk_seq(
            &mut eval,
            map_thunk(add.clone(), vec![ints(&[1, 2]), ints(&[3, 4])]),
            |_, _| {
                calls += 1;
                Err(NativeError::arity("visit", 0))
            },
        );
        assert!(failed.is_err() && calls == 1);
        // `collect_packed` (`mapv`) takes the same walk.
        let two = collect_packed(
            &mut eval,
            map_thunk(add, vec![ints(&[1, 2]), ints(&[3, 4])]),
        );
        assert_eq!(Value::Vector(two.expect("collects")), ints(&[4, 6]));
        let one = collect_packed(&mut eval, map_thunk(inc, vec![ints(&[5])]));
        assert_eq!(Value::Vector(one.expect("collects")), ints(&[6]));
    }
}
