// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S5D collection natives: the `Collections` matrix rows over
//! persistent vectors/maps/sets/lists, subvectors, and transients
//! (`docs/spec/t1174/s5d-coll-pins.md`). Shapes follow `core.clj` + the
//! `RT`/`APersistent*` sources exactly; the `into`/`merge-with`
//! transient fast paths are plain loops (same values and errors —
//! transients never escape).

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{Arc, Mutex, Weak};

use clj_value::{
    Array, ConsValue, Keyword, List, Map, PrimitiveVector, Set, SortSpec, SubVectorValue,
    ThrowClass, Transient, TransientError, TransientKind, TransientMap, TransientSet,
    TransientVector, Value, Vector, new_map, new_set,
};
use smallvec::SmallVec;

use crate::core::{cce_err, cce_message, is_counted, jvm_compare, nth_index};
use crate::error::{NativeError, Signal, jvm_class_of, simple_name_of};
use crate::eval::Evaluator;
use crate::ext::as_native;
use crate::lazy::{
    StepUnit, as_range, as_realized, as_repeat, as_subvector, collect_all, collect_packed,
    empty_list_value, equiv_vals, first_value, is_seq_value, is_seqable, next_value,
    realized_value, replace_thunk, seq_value, step_unit, take_walk_coll,
};
use crate::ns::{NativeFnPtr, check_arity, signal_to_native, str_of};
use crate::seqs::{afunction_order, call_checked};
use crate::vars::check_min_arity;

/// Whether the value is a transient buffer.
pub(crate) fn is_transient(v: &Value) -> bool {
    matches!(v, Value::Transient(_))
}

/// A `TransientError` as a native error (exact JVM class + text).
fn transient_err(e: TransientError) -> NativeError {
    let class = match e {
        TransientError::UseAfterPersistent => ThrowClass::IllegalAccessError,
        TransientError::EmptyPop => ThrowClass::IllegalStateException,
        TransientError::IndexOutOfBounds => ThrowClass::IndexOutOfBoundsException,
    };
    NativeError::new(class, e.message().to_owned())
}

/// `Util.isInteger`: only our `Int` (doubles — even integral ones —
/// are not integers, probed).
fn int_key(k: &Value) -> Option<i64> {
    match k {
        Value::Int(i) => Some(*i),
        _ => None,
    }
}

/// A string index: any `Number` truncated (`intValue`; NaN → 0,
/// out-of-range saturates, like the JVM cast).
fn num_index(k: &Value) -> Option<i32> {
    match k {
        Value::Int(i) => Some(*i as i32),
        Value::Double(d) => Some(*d as i32),
        _ => None,
    }
}

/// Java `getSimpleName` over a JVM binary name: strip the package,
/// then a member-class owner (`$` + uppercase strips to the inner;
/// `$` + lowercase is a generated fn name and stays).
fn java_simple_name(v: &Value) -> String {
    simple_name_of(&jvm_class_of(v))
}

/// Present-or-missing lookup mirroring `RT.get` exactly (`Some`
/// even for present-nil; `None` ⟺ `not-found`). Vector-family keys
/// truncate through `i32` (`intValue` wraps, like the JVM).
pub(crate) fn lookup_present(
    eval: &mut Evaluator,
    coll: &Value,
    key: &Value,
) -> Result<Option<Value>, NativeError> {
    // Maps dispatch before the record/struct probe (disjoint
    // types — `get` over maps is a hot loop staple, #235).
    if let Value::Map(m) = coll {
        if m.is_sorted() {
            return sorted_lookup_map(eval, m, key);
        }
        let key = crate::lazy::realize_key(eval, key).map_err(signal_to_native)?;
        return Ok(m.get(&key).cloned());
    }
    // Records read their content map (oracle `rec-pos`,
    // `rec-get`); structs likewise (S12 oracle `sm-get-basis`);
    // every other host value misses (oracle `dtor-get`).
    if let Some(i) = crate::types::maplike_inst(coll) {
        let key = crate::lazy::realize_key(eval, key).map_err(signal_to_native)?;
        return Ok(i.content.get(&key).cloned());
    }
    match coll {
        Value::Nil => Ok(None),
        Value::Vector(x) => Ok(lookup_vector_slot(x, key)),
        Value::MapEntry {
            key: ek, val: ev, ..
        } => Ok(lookup_entry_slot(ek, ev, key)),
        // Arrays narrow any `Number` key (`intValue`, lenient —
        // OOB/non-`Number`/nil miss; S15 oracles `b-get`,
        // `f-get-*`, `e-get-*`).
        Value::Array(a) => Ok(lookup_array_slot(a, key)),
        Value::PrimitiveVector(p) => Ok(lookup_pv_slot(p, key)),
        Value::Set(s) => lookup_set_slot(eval, s, key),
        Value::Str(s) => Ok(lookup_str_slot(s, key)),
        Value::Transient(t) => transient_get(t, key),
        other => lookup_present_fallback(other, key),
    }
}

/// A set slot (sorted sets resolve through the comparator; split from
/// [`lookup_present`] for the complexity budget).
fn lookup_set_slot(
    eval: &mut Evaluator,
    s: &Set,
    key: &Value,
) -> Result<Option<Value>, NativeError> {
    if s.is_sorted() {
        Ok(sorted_lookup_set(eval, s, key)?)
    } else {
        Ok(s.get(key).cloned())
    }
}

fn lookup_vector_slot(x: &Vector, key: &Value) -> Option<Value> {
    int_key(key).and_then(|i| usize::try_from(i as i32).ok().and_then(|idx| x.get(idx)))
}

fn lookup_entry_slot(ek: &Value, ev: &Value, key: &Value) -> Option<Value> {
    int_key(key).and_then(|i| {
        usize::try_from(i as i32).ok().and_then(|idx| match idx {
            0 => Some(ek.clone()),
            1 => Some(ev.clone()),
            _ => None,
        })
    })
}

fn lookup_pv_slot(p: &PrimitiveVector, key: &Value) -> Option<Value> {
    int_key(key).and_then(|i| {
        usize::try_from(i as i32)
            .ok()
            .and_then(|idx| p.get(idx).cloned())
    })
}

fn lookup_str_slot(s: &Arc<str>, key: &Value) -> Option<Value> {
    num_index(key).and_then(|i| {
        usize::try_from(i)
            .ok()
            .and_then(|idx| str_char_at(s, idx).map(Value::Char))
    })
}

/// Array lookup through `intValue` keys (split from
/// [`lookup_present`] for the complexity budget): lenient —
/// OOB/non-`Number`/nil miss.
fn lookup_array_slot(a: &clj_value::Array, key: &Value) -> Option<Value> {
    crate::arrays::array_intvalue_key(key).and_then(|i| {
        usize::try_from(i).ok().and_then(|idx| {
            if idx < a.len() {
                Some(crate::arrays::array_read(a, idx))
            } else {
                None
            }
        })
    })
}

fn lookup_present_fallback(coll: &Value, key: &Value) -> Result<Option<Value>, NativeError> {
    if let Some(sp) = crate::lazy::as_splitvec(coll) {
        return Ok(
            int_key(key).and_then(|i| usize::try_from(i as i32).ok().and_then(|idx| sp.nth(idx)))
        );
    }
    if let Some(sv) = as_subvector(coll) {
        return Ok(
            int_key(key).and_then(|i| usize::try_from(i as i32).ok().and_then(|idx| sv.nth(idx)))
        );
    }
    Ok(None)
}

/// Read a transient (`get`/`contains?` share this; use after
/// `persistent!` is the `IllegalAccessError`).
fn transient_get(t: &Transient, key: &Value) -> Result<Option<Value>, NativeError> {
    let guard = t.lock();
    match &*guard {
        TransientKind::Vector(tv) => {
            let Some(i) = int_key(key) else {
                return Ok(None);
            };
            let Some(idx) = usize::try_from(i as i32).ok() else {
                return Ok(None);
            };
            Ok(tv.nth(idx).map_err(transient_err)?)
        }
        TransientKind::Map(tm) => Ok(tm.get(key).map_err(transient_err)?.cloned()),
        TransientKind::Set(ts) => {
            if ts.contains(key).map_err(transient_err)? {
                Ok(Some(key.clone()))
            } else {
                Ok(None)
            }
        }
    }
}

/// `RT.find`: the `[k v]` entry for a present key (`None` when
/// missing; the key reads back as passed). Transient sets and
/// anything else is the `find not supported` error.
/// `RT.find` over a vector (the `[k v]` entry at the truncated index).
fn find_entry_vector(x: &Vector, key: &Value) -> Option<(Value, Value)> {
    int_key(key).and_then(|i| {
        usize::try_from(i as i32)
            .ok()
            .and_then(|idx| x.get(idx).map(|v| (key.clone(), v.clone())))
    })
}

/// `RT.find` over a primitive vector (like vectors).
fn find_entry_pv(p: &PrimitiveVector, key: &Value) -> Option<(Value, Value)> {
    int_key(key).and_then(|i| {
        usize::try_from(i as i32)
            .ok()
            .and_then(|idx| p.get(idx).map(|v| (key.clone(), v.clone())))
    })
}

/// `RT.find` over a transient (maps/vectors read; sets fail).
fn find_entry_transient(
    coll: &Value,
    t: &Transient,
    key: &Value,
) -> Result<Option<(Value, Value)>, NativeError> {
    // Read the class before locking (`jvm_class_of` locks
    // transients itself — locking twice deadlocks).
    let from = jvm_class_of(coll);
    let guard = t.lock();
    match &*guard {
        TransientKind::Map(tm) => Ok(tm
            .get(key)
            .map_err(transient_err)?
            .map(|v| (key.clone(), v.clone()))),
        TransientKind::Vector(tv) => {
            let Some(i) = int_key(key) else {
                return Ok(None);
            };
            let Some(idx) = usize::try_from(i as i32).ok() else {
                return Ok(None);
            };
            Ok(tv
                .nth(idx)
                .map_err(transient_err)?
                .map(|v| (key.clone(), v.clone())))
        }
        TransientKind::Set(_) => Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("find not supported on type: {from}"),
        )),
    }
}

/// `RT.find` over anything else (subvectors read; the rest fail).
fn find_entry_fallback(coll: &Value, key: &Value) -> Result<Option<(Value, Value)>, NativeError> {
    if let Some(sp) = crate::lazy::as_splitvec(coll) {
        return Ok(int_key(key).and_then(|i| {
            usize::try_from(i as i32)
                .ok()
                .and_then(|idx| sp.nth(idx).map(|v| (key.clone(), v)))
        }));
    }
    if let Some(sv) = as_subvector(coll) {
        return Ok(int_key(key).and_then(|i| {
            usize::try_from(i as i32)
                .ok()
                .and_then(|idx| sv.nth(idx).map(|v| (key.clone(), v.clone())))
        }));
    }
    Err(find_unsupported(coll))
}

/// `find` on maps (sorted maps resolve through the comparator).
fn find_entry_map(
    eval: &mut Evaluator,
    m: &Map,
    key: &Value,
) -> Result<Option<(Value, Value)>, NativeError> {
    if m.is_sorted() {
        Ok(sorted_lookup_map(eval, m, key)?.map(|v| (key.clone(), v)))
    } else {
        let key = crate::lazy::realize_key(eval, key).map_err(signal_to_native)?;
        Ok(m.get(&key).map(|v| (key.clone(), v.clone())))
    }
}

pub(crate) fn find_entry(
    eval: &mut Evaluator,
    coll: &Value,
    key: &Value,
) -> Result<Option<(Value, Value)>, NativeError> {
    // Records find through their content (oracle `rec-find`);
    // structs likewise (S12); anything else host takes the
    // unsupported path (oracle `i-find`).
    if let Some(i) = crate::types::maplike_inst(coll) {
        return find_entry_map(eval, &i.content, key);
    }
    match coll {
        Value::Nil => Ok(None),
        Value::Map(m) => find_entry_map(eval, m, key),
        Value::Vector(x) => Ok(find_entry_vector(x, key)),
        Value::PrimitiveVector(p) => Ok(find_entry_pv(p, key)),
        Value::MapEntry {
            key: ek, val: ev, ..
        } => Ok(int_key(key).and_then(|i| {
            usize::try_from(i as i32).ok().and_then(|idx| match idx {
                0 => Some((key.clone(), (**ek).clone())),
                1 => Some((key.clone(), (**ev).clone())),
                _ => None,
            })
        })),
        Value::Transient(t) => find_entry_transient(coll, t, key),
        _ => find_entry_fallback(coll, key),
    }
}

/// The `find not supported` error (full class name, like the JVM).
fn find_unsupported(coll: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("find not supported on type: {}", jvm_class_of(coll)),
    )
}

/// How one value conjs onto a map (persistent and transient share
/// this): nil passes through, pairs assoc, maps merge entry-wise,
/// anything else seqs into one pair per entry (the first non-entry
/// element is the `Map$Entry` cast error — computed in the
/// unlocked plan phase, since resolving the head's class under the
/// transient lock would re-lock when the head aliases the
/// transient).
enum MapConjPlan {
    Nil,
    Pair(Value, Value),
    Entries(Vec<(Value, Value)>),
}

/// A vector-family conj plan: exactly a pair assocs,
/// anything else is the JVM's pair error (`Vec` is an
/// `IPersistentVector`, like vectors).
fn pair_plan(items: &[Value]) -> Result<MapConjPlan, NativeError> {
    match items {
        [k, v] => Ok(MapConjPlan::Pair(k.clone(), v.clone())),
        _ => Err(pair_err()),
    }
}

/// The pair plan for a vector-like extension conj arg (split
/// vectors answer from their spans; non-pairs are the pair error).
fn pair_plan_ext(x: &Value) -> Option<Result<MapConjPlan, NativeError>> {
    if let Some(sp) = crate::lazy::as_splitvec(x) {
        if sp.len() != 2 {
            return Some(Err(pair_err()));
        }
        return Some(Ok(MapConjPlan::Pair(
            sp.nth(0).unwrap_or(Value::Nil),
            sp.nth(1).unwrap_or(Value::Nil),
        )));
    }
    as_subvector(x).map(|sv| {
        if sv.len() != 2 {
            return Err(pair_err());
        }
        let items = sv.items();
        Ok(MapConjPlan::Pair(items[0].clone(), items[1].clone()))
    })
}

fn map_conj_plan(eval: &mut Evaluator, x: &Value) -> Result<MapConjPlan, NativeError> {
    match x {
        Value::Nil => Ok(MapConjPlan::Nil),
        Value::Map(other) => Ok(MapConjPlan::Entries(
            other.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        )),
        Value::Vector(v) => {
            let items: Vec<Value> = v.iter().collect();
            pair_plan(&items)
        }
        // `Vec` pairs assoc like vectors (S15); non-pairs are
        // the pair error, never the entry cast.
        Value::PrimitiveVector(p) => pair_plan(&p.items()),
        Value::MapEntry { key, val, .. } => Ok(MapConjPlan::Pair((**key).clone(), (**val).clone())),
        _ => {
            if let Some(plan) = pair_plan_ext(x) {
                return plan;
            }
            // A sequential arg contributes one pair per entry
            // (Oracle FX2 `conj-seq-entries`, `conj-list-entry`,
            // `conj-multi`); the first non-entry element is the
            // `Map$Entry` cast error (`conj-listpair`,
            // `conj-vec-in-list`, `conj-nest1`, `conj-str`).
            let mut pairs = Vec::new();
            let mut cur = seq_value(eval, x).map_err(signal_to_native)?;
            loop {
                if matches!(cur, Value::Nil) {
                    return Ok(MapConjPlan::Entries(pairs));
                }
                let head = first_value(eval, &cur).map_err(signal_to_native)?;
                let Value::MapEntry { key, val, .. } = head else {
                    return Err(cce_err(&head, "java.util.Map$Entry"));
                };
                pairs.push(((*key).clone(), (*val).clone()));
                cur = next_value(eval, &cur).map_err(signal_to_native)?;
            }
        }
    }
}

/// `Vector arg to map conj must be a pair`.
fn pair_err() -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        "Vector arg to map conj must be a pair".to_owned(),
    )
}

/// Conj onto a sorted persistent map (split from
/// [`conj_map_into`] for the complexity budget).
fn conj_map_into_sorted(eval: &mut Evaluator, m: &Map, x: &Value) -> Result<Map, NativeError> {
    match map_conj_plan(eval, x)? {
        MapConjPlan::Nil => Ok(m.clone()),
        MapConjPlan::Pair(k, v) => Ok(sorted_assoc_map(eval, m, k, v)?),
        MapConjPlan::Entries(entries) => {
            let mut out = m.clone();
            for (k, v) in entries {
                out = sorted_assoc_map(eval, &out, k, v)?;
            }
            Ok(out)
        }
    }
}

/// Conj onto a persistent map (`APersistentMap.cons`), answering
/// the map (records rewrap the answer around their content).
/// Owned `m` (#227): multi-step callers move their accumulator
/// through, so each step after the first mutates uniquely-owned
/// storage in place instead of path-copying per step.
pub(crate) fn conj_map_into(eval: &mut Evaluator, m: Map, x: &Value) -> Result<Map, NativeError> {
    if m.is_sorted() {
        return conj_map_into_sorted(eval, &m, x);
    }
    match map_conj_plan(eval, x)? {
        MapConjPlan::Nil => Ok(m),
        MapConjPlan::Pair(k, v) => {
            let k = crate::lazy::realize_key(eval, &k).map_err(signal_to_native)?;
            Ok(m.insert(k, v))
        }
        MapConjPlan::Entries(entries) => {
            let mut out = m;
            for (k, v) in entries {
                let k = crate::lazy::realize_key(eval, &k).map_err(signal_to_native)?;
                out = out.insert(k, v);
            }
            Ok(out)
        }
    }
}

/// Conj onto a subvector (stays a subvector; writes at the window
/// end, clobbering backing items past the window — like `assocN`).
fn subvector_conj(sv: &SubVectorValue, x: Value) -> Value {
    let v = sv.vec();
    let backing = if sv.end() < v.len() {
        v.set_nth(sv.end(), x.clone())
            .expect("subvector end is below the backing length")
    } else {
        v.clone().push_back(x)
    };
    Value::Extension(Arc::new(
        SubVectorValue::new(backing, sv.start(), sv.end() + 1)
            .with_orig_start(sv.orig_start())
            .with_meta(sv.meta().cloned()),
    ))
}

/// Conj onto a primitive vector coerces through the type (the
/// `ArrayManager` rule — split from [`conj_one`] for the
/// complexity budget).
fn conj_pv(eval: &mut Evaluator, p: PrimitiveVector, x: Value) -> Result<Value, NativeError> {
    let co = crate::arrays::vector_coerce(eval, p.elem(), &x)?;
    Ok(Value::PrimitiveVector(p.push_back(co)))
}

/// `RT.conj` one element (`conj`/`into` share this).
/// Owned `coll` (#227, like `assoc3` in #215): multi-step callers
/// move their accumulator through, so each step after the first
/// mutates uniquely-owned storage in place instead of
/// path-copying per step.
fn conj_one(eval: &mut Evaluator, coll: Value, x: Value) -> Result<Value, NativeError> {
    match coll {
        Value::Nil => Ok(Value::List(List::from_vec(vec![x]))),
        Value::List(l) => Ok(Value::List(l.push_front(x))),
        Value::Vector(v) => Ok(Value::Vector(v.push_back(x))),
        // Primitive vectors coerce through the type (the
        // `ArrayManager` rule — S15 oracles `f-conj-bad`,
        // `j-byte-vec`, `j-bool-num`, `j-conj-nil`).
        Value::PrimitiveVector(p) => conj_pv(eval, p, x),
        // Conj onto an entry appends like its 2-vector (Oracle FX2
        // `conj`, `merge-entry`).
        Value::MapEntry { key, val, .. } => {
            let tmp = Vector::from_vec(vec![*key, *val]);
            Ok(Value::Vector(tmp.push_back(x)))
        }
        Value::Map(m) => Ok(Value::Map(conj_map_into(eval, m, &x)?)),
        // Records conj through their content map and stay records
        // (oracle `h-conj-vec`, `h-conj-map-type`); structs
        // likewise (S12); anything else host falls to the
        // collection cast (oracle `i-conj`).
        Value::HostType(h) => {
            let coll = Value::HostType(h);
            conj_maplike(eval, coll, &x)
        }
        Value::Set(s) if s.is_sorted() => Ok(Value::Set(sorted_conj_set(eval, &s, x)?)),
        Value::Set(s) => {
            let x = crate::lazy::realize_key(eval, &x).map_err(signal_to_native)?;
            Ok(Value::Set(s.insert(x)))
        }
        other => conj_other(other, x),
    }
}

/// Conj onto anything else (split from [`conj_one`] for the
/// complexity budget): subvectors conj through their window, seqs
/// `cons`, and anything else takes the collection cast.
fn conj_other(other: Value, x: Value) -> Result<Value, NativeError> {
    if let Some(sv) = as_subvector(&other) {
        return Ok(subvector_conj(sv, x));
    }
    if is_seq_value(&other) {
        return Ok(Value::Extension(Arc::new(ConsValue::new(x, other))));
    }
    Err(cce_err(&other, "clojure.lang.IPersistentCollection"))
}

/// Conj onto a map-like host value (split from [`conj_one`] for
/// the complexity budget): through the content map, rewrapped so
/// the answer stays a record (or a struct, S12); anything else
/// host takes the collection cast. The content still clones per
/// step (it borrows out of the owned host), so record `conj`
/// chains keep their first-step path copy.
fn conj_maplike(eval: &mut Evaluator, coll: Value, x: &Value) -> Result<Value, NativeError> {
    match crate::types::maplike_inst(&coll) {
        Some(i) => Ok(crate::types::rewrap_maplike(
            i,
            conj_map_into(eval, i.content.clone(), x)?,
        )),
        None => Err(cce_err(&coll, "clojure.lang.IPersistentCollection")),
    }
}

/// `assoc` on a vector (integer keys only; at-count appends;
/// anything else is the empty `IndexOutOfBoundsException`).
fn assoc_vector(x: Vector, i: i64, v: Value) -> Result<Value, NativeError> {
    if let Ok(n) = usize::try_from(i as i32) {
        if n < x.len() {
            let out = x.set_nth_owned(n, v).expect("index is below the length");
            return Ok(Value::Vector(out));
        } else if n == x.len() {
            return Ok(Value::Vector(x.push_back(v)));
        }
    }
    Err(NativeError::new(
        ThrowClass::IndexOutOfBoundsException,
        String::new(),
    ))
}

/// `assoc` on a primitive vector (bounds first, then the coerce
/// — `Vec.assocN`; at-count appends; OOB is the empty `IOOBE`).
fn assoc_pv(
    eval: &mut Evaluator,
    p: PrimitiveVector,
    i: i64,
    v: Value,
) -> Result<Value, NativeError> {
    let Ok(n) = usize::try_from(i as i32) else {
        return Err(ioobe());
    };
    if n > p.len() {
        return Err(ioobe());
    }
    let co = crate::arrays::vector_coerce(eval, p.elem(), &v)?;
    if n == p.len() {
        return Ok(Value::PrimitiveVector(p.push_back(co)));
    }
    Ok(Value::PrimitiveVector(
        p.set_nth(n, co).expect("index is below the length"),
    ))
}

const SUBVEC_COMPACT_MAX: usize = 256;

/// `assoc` on a subvector (stays a subvector; writes at the absolute
/// index — even before the window, like the JVM).
fn assoc_subvector(sv: &SubVectorValue, k: &Value, v: Value) -> Result<Value, NativeError> {
    let Some(i) = int_key(k) else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "Key must be integer".to_owned(),
        ));
    };
    let rel = i64::from(i as i32);
    let len = sv.len() as i64;
    if sv.orig_start() as i64 + rel < 0 || rel > len {
        return Err(NativeError::new(
            ThrowClass::IndexOutOfBoundsException,
            String::new(),
        ));
    }
    if rel == len {
        return Ok(subvector_conj(sv, v));
    }
    let next = if rel < 0 {
        sv.clone()
    } else {
        let idx = rel as usize;
        if sv.len() <= SUBVEC_COMPACT_MAX && sv.vec().len() > sv.len() {
            let backing = sv
                .vec()
                .slice_packed(sv.start(), sv.end())
                .set_nth_owned(idx, v)
                .expect("window index is below the compacted length");
            SubVectorValue::new(backing, 0, sv.len())
                .with_orig_start(sv.orig_start())
                .with_meta(sv.meta().cloned())
        } else {
            let backing = sv
                .vec()
                .set_nth(sv.start() + idx, v)
                .expect("window index is below the backing length");
            SubVectorValue::new(backing, sv.start(), sv.end())
                .with_orig_start(sv.orig_start())
                .with_meta(sv.meta().cloned())
        }
    };
    Ok(Value::Extension(Arc::new(next)))
}

/// `assoc` on a map-like host value (split from [`assoc3`] for
/// the complexity budget): through the content map, rewrapped so
/// the answer stays a record (or a struct, S12); anything else
/// host is not associative.
fn assoc_maplike(coll: &Value, k: &Value, v: Value) -> Result<Value, NativeError> {
    match crate::types::maplike_inst(coll) {
        Some(i) => Ok(crate::types::rewrap_maplike(
            i,
            i.content.clone().insert(k.clone(), v),
        )),
        None => Err(cce_err(coll, "clojure.lang.Associative")),
    }
}

/// `RT.assoc` one triple (`assoc`/`update`/`merge-with` share this;
/// sorted maps position by comparator, FT6).
fn assoc3(eval: &mut Evaluator, coll: Value, k: &Value, v: Value) -> Result<Value, NativeError> {
    // Owned `coll` (#215): multi-step callers (`assoc`, `replace`,
    // `merge-with`) move their accumulator through, so each step
    // after the first mutates uniquely-owned storage in place
    // instead of path-copying per step.
    match coll {
        Value::Nil => Ok(Value::Map(new_map().insert(
            crate::lazy::realize_key(eval, k).map_err(signal_to_native)?,
            v,
        ))),
        Value::Map(m) if m.is_sorted() => Ok(Value::Map(sorted_assoc_map(eval, &m, k.clone(), v)?)),
        Value::Map(m) => Ok(Value::Map(m.insert(
            crate::lazy::realize_key(eval, k).map_err(signal_to_native)?,
            v,
        ))),
        Value::Vector(x) => {
            let Some(i) = int_key(k) else {
                return Err(NativeError::new(
                    ThrowClass::IllegalArgumentException,
                    "Key must be integer".to_owned(),
                ));
            };
            assoc_vector(x, i, v)
        }
        Value::PrimitiveVector(p) => {
            let Some(i) = int_key(k) else {
                return Err(NativeError::new(
                    ThrowClass::IllegalArgumentException,
                    "Key must be integer".to_owned(),
                ));
            };
            assoc_pv(eval, p, i, v)
        }
        // `assoc` on an entry answers a plain vector (Oracle FX2
        // `assoc0`, `assoc1`, `assoc2`).
        Value::MapEntry { key, val, .. } => {
            let Some(i) = int_key(k) else {
                return Err(NativeError::new(
                    ThrowClass::IllegalArgumentException,
                    "Key must be integer".to_owned(),
                ));
            };
            let tmp = Vector::from_vec(vec![*key, *val]);
            assoc_vector(tmp, i, v)
        }
        // Records assoc through their content map and stay records
        // (oracle `rec-assoc-type`, `h-assoc-existing`); structs
        // always stay structs (S12 oracles `sm-assoc-class`,
        // `sm-assoc-extra-class`); anything else host is not
        // associative (oracle `i-assoc`).
        Value::HostType(h) => {
            let coll = Value::HostType(h);
            assoc_maplike(&coll, k, v)
        }
        other => {
            if let Some(sv) = as_subvector(&other) {
                return assoc_subvector(sv, k, v);
            }
            Err(cce_err(&other, "clojure.lang.Associative"))
        }
    }
}

/// Walk a seq counting items (hangs on infinite seqs, like the JVM).
fn walk_count(eval: &mut Evaluator, v: &Value) -> Result<usize, NativeError> {
    let mut n = 0usize;
    let mut bump = |_: &mut Evaluator, _: Value| {
        n += 1;
        Ok(true)
    };
    if let Some(done) = crate::lazy::walk_sole_map(eval, v, &mut bump) {
        done.map_err(signal_to_native)?;
        return Ok(n);
    }
    let mut cur = v.clone();
    loop {
        match step_unit(eval, &cur).map_err(signal_to_native)? {
            StepUnit::Empty => return Ok(n),
            StepUnit::One { tail, .. } => {
                n += 1;
                cur = tail;
            }
            StepUnit::Chunk { items, tail } => {
                n += items.len();
                cur = tail;
            }
        }
    }
}

/// `RT.count` over a transient buffer.
fn count_transient(t: &Transient) -> Result<i64, NativeError> {
    let guard = t.lock();
    let n = match &*guard {
        TransientKind::Vector(tv) => tv.len(),
        TransientKind::Map(tm) => tm.len(),
        TransientKind::Set(ts) => ts.len(),
    }
    .map_err(transient_err)?;
    Ok(n as i64)
}

/// `RT.count` over the extension/seq tail (subvectors, ranges,
/// realized seqs, repeats, then a seq walk).
fn count_fallback(eval: &mut Evaluator, v: &Value) -> Result<i64, NativeError> {
    if let Some(sp) = crate::lazy::as_splitvec(v) {
        return Ok(sp.len() as i64);
    }
    if let Some(sv) = as_subvector(v) {
        return Ok(sv.len() as i64);
    }
    if let Some(items) = crate::emucoll::list_snapshot(v) {
        return Ok(items.len() as i64);
    }
    if let Some(r) = as_range(v) {
        // A `BigDec`/ratio step with no exact decimal raises on
        // the first advance (oracle `range-mixed`), so `count`
        // raises instead of walking short.
        if r.step_inexact() {
            return Err(crate::core::arithmetic(clj_value::NON_TERMINATING));
        }
        return Ok(r.walked_len() as i64);
    }
    if let Some(s) = as_realized(v) {
        return Ok(s.seq().count() as i64);
    }
    if let Some(n) = as_repeat(v).and_then(|r| r.n()) {
        return Ok(n.max(0));
    }
    if is_seq_value(v) {
        return Ok(walk_count(eval, v)? as i64);
    }
    Err(NativeError::new(
        ThrowClass::UnsupportedOperationException,
        format!("count not supported on this type: {}", java_simple_name(v)),
    ))
}

/// `RT.count` as an `i64` (`count`/`empty?`/arities share this).
pub(crate) fn count_of(eval: &mut Evaluator, v: &Value) -> Result<i64, NativeError> {
    // Records count their content (oracle `rec-count`); structs
    // likewise (S12 oracle `sm-count`); chunks count their
    // appended items (S12 oracle `ch-build`); chunk buffers
    // count their used slots, dead or alive (S12 oracles
    // `ch-buf-count-*`); anything else host takes the
    // unsupported path (oracle `dtor-count`).
    if let Some(i) = crate::types::maplike_inst(v) {
        return Ok(i.content.size() as i64);
    }
    if let Some(c) = crate::collmisc::as_chunk(v) {
        return crate::collmisc::chunk_count(c);
    }
    if let Some(b) = crate::collmisc::as_chunk_buffer(v) {
        return Ok(crate::collmisc::chunk_buffer_count(b));
    }
    match v {
        Value::Nil => Ok(0),
        Value::Str(s) => Ok(str_utf16_len(s) as i64),
        Value::List(l) => Ok(l.len() as i64),
        Value::Vector(x) => Ok(x.len() as i64),
        Value::PrimitiveVector(p) => Ok(p.len() as i64),
        Value::Array(a) => Ok(a.len() as i64),
        Value::MapEntry { .. } => Ok(2),
        Value::Map(m) => Ok(m.size() as i64),
        Value::Set(s) => Ok(s.size() as i64),
        Value::Transient(t) => count_transient(t),
        _ => count_fallback(eval, v),
    }
}

/// Plain `into` (`reduce conj`; the transient/kvreduce fast paths
/// are unobservable — same values, same errors).
/// Owned `to` (#227): the accumulator moves through [`conj_one`],
/// so each step after the first mutates uniquely-owned storage in
/// place instead of path-copying per step.
fn try_into_vector(to: Vector, from: &Value) -> Result<Vector, Vector> {
    match from {
        Value::Vector(src) => {
            if to.is_empty() && to.meta().is_none() {
                return Ok(src.clone().with_meta(None));
            }
            to.try_extend_flat_window(src, 0, src.len())
        }
        Value::Extension(_) => {
            let Some(sv) = crate::lazy::as_subvector(from) else {
                return Err(to);
            };
            to.try_extend_flat_window(sv.vec(), sv.start(), sv.end())
        }
        _ => Err(to),
    }
}

fn into_inner(eval: &mut Evaluator, to: Value, from: Value) -> Result<Value, NativeError> {
    let mut acc = match to {
        Value::Vector(v) => match try_into_vector(v, &from) {
            Ok(out) => return Ok(Value::Vector(out)),
            Err(orig) => Value::Vector(orig),
        },
        other => other,
    };
    crate::seqs::walk_seq(eval, from, |eval, head| {
        let cur = std::mem::replace(&mut acc, Value::Nil);
        acc = conj_one(eval, cur, head.clone())?;
        Ok(crate::seqs::Walk::Continue)
    })?;
    Ok(acc)
}

/// Two-way `merge` (`([x y] (if (and x y) (into x y) (or x y)))`).
/// Owned `x` (#227): the accumulator moves through [`into_inner`].
fn merge2(eval: &mut Evaluator, x: Value, y: &Value) -> Result<Value, NativeError> {
    if x.is_truthy() && y.is_truthy() {
        into_inner(eval, x, y.clone())
    } else if x.is_truthy() {
        Ok(x)
    } else {
        Ok(y.clone())
    }
}

/// One `merge-with` key/value pair (`merge-kv`: missing keys assoc,
/// present keys combine through `f`).
fn merge_kv(
    eval: &mut Evaluator,
    m: Value,
    k: Value,
    v: Value,
    f: &Value,
) -> Result<Value, NativeError> {
    match lookup_present(eval, &m, &k)? {
        Some(mv) => {
            let combined = eval.call(f, &[mv, v]).map_err(signal_to_native)?;
            assoc3(eval, m, &k, combined)
        }
        None => assoc3(eval, m, &k, v),
    }
}

/// Three-way `merge-with` (maps and vectors take the `kvreduce`
/// path; anything else seqs and casts its first element to
/// `Map.Entry`).
/// Early `merge-with` exit when either side is falsy.
fn merge_with3_early(x: &Value, y: &Value) -> Option<Value> {
    if !(x.is_truthy() && y.is_truthy()) {
        if x.is_truthy() {
            return Some(x.clone());
        }
        return Some(y.clone());
    }
    None
}

/// `merge-with` over a map right-hand side (the `kvreduce` path).
fn merge_with3_map(
    eval: &mut Evaluator,
    f: &Value,
    x: Value,
    ym: &Map,
) -> Result<Value, NativeError> {
    let mut m = x;
    for (k, v) in ym.iter() {
        m = merge_kv(eval, m, k.clone(), v.clone(), f)?;
    }
    Ok(m)
}

/// `merge-with` index/value pairs for a vector-family right-hand side.
fn merge_with3_indexed_pairs(y: &Value) -> Option<Vec<(Value, Value)>> {
    match y {
        Value::Vector(yv) => Some(
            yv.iter()
                .enumerate()
                .map(|(i, v)| (Value::Int(i as i64), v.clone()))
                .collect(),
        ),
        Value::PrimitiveVector(p) => Some(
            p.iter()
                .enumerate()
                .map(|(i, v)| (Value::Int(i as i64), v.clone()))
                .collect(),
        ),
        _ => as_subvector(y).map(|sv| {
            sv.iter()
                .enumerate()
                .map(|(i, v)| (Value::Int(i as i64), v.clone()))
                .collect()
        }),
    }
}

/// `merge-with` folding one pair per key through `merge_kv`.
fn merge_with3_fold_pairs(
    eval: &mut Evaluator,
    f: &Value,
    x: Value,
    pairs: Vec<(Value, Value)>,
) -> Result<Value, NativeError> {
    let mut m = x;
    for (k, v) in pairs {
        m = merge_kv(eval, m, k, v, f)?;
    }
    Ok(m)
}

/// `merge-with` over a sequential right-hand side (seqs and casts).
fn merge_with3_seq_tail(
    eval: &mut Evaluator,
    f: &Value,
    x: Value,
    y: &Value,
) -> Result<Value, NativeError> {
    // A sequential right-hand side contributes one pair per entry
    // (Oracle FX2 `conj-seq-entries`, `merge-with-entry` CCEs on
    // the `:a` head — entries are real `MapEntry` values now).
    let mut m = x;
    let mut cur = seq_value(eval, y).map_err(signal_to_native)?;
    loop {
        if matches!(cur, Value::Nil) {
            return Ok(m);
        }
        let head = first_value(eval, &cur).map_err(signal_to_native)?;
        let Value::MapEntry { key, val, .. } = head else {
            return Err(cce_err(&head, "java.util.Map$Entry"));
        };
        m = merge_kv(eval, m, (*key).clone(), (*val).clone(), f)?;
        cur = next_value(eval, &cur).map_err(signal_to_native)?;
    }
}

/// Owned `x` (#227): the accumulator moves through [`merge_kv`].
fn merge_with3(eval: &mut Evaluator, f: &Value, x: Value, y: &Value) -> Result<Value, NativeError> {
    if let Some(v) = merge_with3_early(&x, y) {
        return Ok(v);
    }
    if let Value::Map(ym) = y {
        return merge_with3_map(eval, f, x, ym);
    }
    if let Some(pairs) = merge_with3_indexed_pairs(y) {
        return merge_with3_fold_pairs(eval, f, x, pairs);
    }
    merge_with3_seq_tail(eval, f, x, y)
}

fn take_lookup(
    eval: &mut Evaluator,
    coll: &mut Value,
    key: &Value,
) -> Result<Option<Value>, NativeError> {
    match coll {
        Value::Map(m) if !m.is_sorted() => {
            let key = crate::lazy::realize_key(eval, key).map_err(signal_to_native)?;
            Ok(m.take_value(&key))
        }
        Value::Vector(x) => Ok(int_key(key).and_then(|i| {
            usize::try_from(i as i32)
                .ok()
                .and_then(|idx| x.take_nth(idx))
        })),
        other => lookup_present(eval, other, key),
    }
}

/// Cursor over a keypath (`update-in` / `assoc-in` / `get-in`):
/// indexes `Value::Vector` directly without allocating `VectorSeq`
/// nodes per depth level.
enum KeyStep {
    Vec(Vector, usize),
    Seq(Value),
}

impl KeyStep {
    fn from_val(eval: &mut Evaluator, v: &Value) -> Result<Option<(Value, Self)>, NativeError> {
        if let Value::Vector(vec) = v {
            return Ok(vec.get(0).map(|k| (k, Self::Vec(vec.clone(), 1))));
        }
        let s = seq_value(eval, v).map_err(signal_to_native)?;
        if matches!(s, Value::Nil) {
            return Ok(None);
        }
        let k = first_value(eval, &s).map_err(signal_to_native)?;
        let rest = next_value(eval, &s).map_err(signal_to_native)?;
        Ok(Some((k, Self::Seq(rest))))
    }

    fn next(self, eval: &mut Evaluator) -> Result<Option<(Value, Self)>, NativeError> {
        match self {
            Self::Vec(vec, i) => Ok(vec.get(i).map(|k| (k, Self::Vec(vec, i + 1)))),
            Self::Seq(Value::Nil) => Ok(None),
            Self::Seq(s) => {
                let k = first_value(eval, &s).map_err(signal_to_native)?;
                let rest = next_value(eval, &s).map_err(signal_to_native)?;
                Ok(Some((k, Self::Seq(rest))))
            }
        }
    }
}

/// The `update-in` recursion (`ks` already stepped once).
fn update_in_up(
    eval: &mut Evaluator,
    mut m: Value,
    k: &Value,
    ks: KeyStep,
    f: &Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    if let Some((k2, ks2)) = ks.next(eval)? {
        let inner = take_lookup(eval, &mut m, k)?.unwrap_or(Value::Nil);
        let updated = update_in_up(eval, inner, &k2, ks2, f, args)?;
        assoc3(eval, m, k, updated)
    } else {
        let old = take_lookup(eval, &mut m, k)?.unwrap_or(Value::Nil);
        let mut call_args: SmallVec<[Value; 4]> = SmallVec::with_capacity(args.len() + 1);
        call_args.push(old);
        call_args.extend(args.iter().cloned());
        let new = eval.call_owned(f, call_args).map_err(signal_to_native)?;
        assoc3(eval, m, k, new)
    }
}

/// The `assoc-in` recursion (`ks` already stepped once).
fn assoc_in_up(
    eval: &mut Evaluator,
    mut m: Value,
    k: &Value,
    ks: KeyStep,
    v: Value,
) -> Result<Value, NativeError> {
    if let Some((k2, ks2)) = ks.next(eval)? {
        let inner = take_lookup(eval, &mut m, k)?.unwrap_or(Value::Nil);
        let updated = assoc_in_up(eval, inner, &k2, ks2, v)?;
        assoc3(eval, m, k, updated)
    } else {
        assoc3(eval, m, k, v)
    }
}

fn native_count(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_count_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_count_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/count", &args, &[1])?;
    let coll = take_walk_coll(&mut args, 0);
    Ok(Value::Int(count_of(eval, &coll)?))
}

fn native_emptyp(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_emptyp_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_emptyp_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/empty?", &args, &[1])?;
    let coll = take_walk_coll(&mut args, 0);
    if is_counted(&coll) {
        Ok(Value::Bool(count_of(eval, &coll)? == 0))
    } else {
        let s = seq_value(eval, &coll).map_err(signal_to_native)?;
        Ok(Value::Bool(matches!(s, Value::Nil)))
    }
}

fn native_not_empty(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/not-empty", args, &[1])?;
    let s = seq_value(eval, &args[0]).map_err(signal_to_native)?;
    if matches!(s, Value::Nil) {
        Ok(Value::Nil)
    } else {
        Ok(args[0].clone())
    }
}

/// `empty` of a vector-like extension (split vectors and
/// subvectors fall back to a plain vector, metadata kept).
fn empty_vector_like(v: &Value) -> Option<Value> {
    if let Some(sp) = crate::lazy::as_splitvec(v) {
        return Some(Value::Vector(
            Vector::new_sync().with_meta(sp.meta().cloned()),
        ));
    }
    as_subvector(v).map(|sv| Value::Vector(Vector::new_sync().with_meta(sv.meta().cloned())))
}

fn native_empty(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/empty", args, &[1])?;
    // `RT.empty`: maps are class-sticky (array stays array, hash
    // stays hash), subvectors fall back to plain vectors, every
    // seq empties to a bare `()`, and anything else (nil, strings,
    // scalars, transients) is nil; metadata is kept on the strict
    // kinds (fixprobe `:empty-*`, corpus 08 `:empty-*`). Records
    // have no empty (oracle `h-empty`); structs empty to the same
    // basis nil-defaulted with extras dropped (S12); anything else
    // host is nil (oracle `i-empty`).
    if let Some(i) = crate::types::record_inst(&args[0]) {
        return Err(NativeError::new(
            ThrowClass::UnsupportedOperationException,
            format!("Can't create empty: {}", i.def.name),
        ));
    }
    if let Some(i) = crate::types::struct_inst(&args[0]) {
        return Ok(crate::types::empty_struct(i));
    }
    if let Some(v) = empty_vector_like(&args[0]) {
        return Ok(v);
    }
    Ok(match &args[0] {
        Value::Map(m) => Value::Map(
            new_map()
                .with_hashy(m.is_hashy())
                .with_meta(m.meta().cloned())
                .with_sort(m.sort_spec().cloned()),
        ),
        Value::Vector(v) => Value::Vector(Vector::new_sync().with_meta(v.meta().cloned())),
        // `empty` stays typed (S15 oracles `j-empty-cls`,
        // `j-empty-conj`); arrays fall through to nil (oracle
        // `b-empty`).
        Value::PrimitiveVector(p) => {
            Value::PrimitiveVector(PrimitiveVector::empty(p.elem()).with_meta(p.meta().cloned()))
        }
        Value::Set(s) => Value::Set(
            new_set()
                .with_meta(s.meta().cloned())
                .with_sort(s.sort_spec().cloned()),
        ),

        v if is_seq_value(v) => Value::List(List::new_sync()),
        _ => Value::Nil,
    })
}

fn native_disj(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_disj_owned(eval, args.iter().cloned().collect())
}

/// `disj` over an owned set (#271): a uniquely held set (a loop
/// accumulator the AOT moves in) drops members in place instead of
/// path-copying its index and order sidecar per call.
pub(crate) fn native_disj_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/disj", 0));
    }
    let coll = std::mem::replace(&mut args[0], Value::Nil);
    if args.len() == 1 {
        return Ok(coll);
    }
    match coll {
        Value::Nil => Ok(Value::Nil),
        Value::Set(s) if s.is_sorted() => {
            let mut out = s;
            for k in &args[1..] {
                out = sorted_without_set(eval, &out, k)?;
            }
            Ok(Value::Set(out))
        }
        Value::Set(mut out) => {
            for k in &args[1..] {
                let k = crate::lazy::realize_key(eval, k).map_err(signal_to_native)?;
                out = out.without(&k);
            }
            Ok(Value::Set(out))
        }
        v => Err(cce_err(&v, "clojure.lang.IPersistentSet")),
    }
}

fn native_set(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_set_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_set_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/set", &args, &[1])?;
    let coll = take_walk_coll(&mut args, 0);
    let items = collect_all(eval, coll).map_err(signal_to_native)?;
    let mut out = new_set();
    for item in items {
        out = out.insert(item);
    }
    Ok(Value::Set(out))
}

fn native_second(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/second", args, &[1])?;
    let n = next_value(eval, &args[0]).map_err(signal_to_native)?;
    first_value(eval, &n).map_err(signal_to_native)
}

fn native_last(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_last_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_last_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/last", &args, &[1])?;
    // Split vectors answer from the last span (stepping would
    // materialize every piece just to walk to the end).
    if let Some(sp) = crate::lazy::as_splitvec(&args[0]) {
        return Ok(sp
            .len()
            .checked_sub(1)
            .and_then(|i| sp.nth(i))
            .unwrap_or(Value::Nil));
    }
    let mut cur = take_walk_coll(&mut args, 0);
    let mut last = Value::Nil;
    loop {
        match step_unit(eval, &cur).map_err(signal_to_native)? {
            StepUnit::Empty => return Ok(last),
            StepUnit::One { head, tail } => {
                last = head;
                cur = tail;
            }
            StepUnit::Chunk { items, tail } => {
                if let Some(head) = items.last() {
                    last = head.clone();
                }
                cur = tail;
            }
        }
    }
}

fn native_nnext(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/nnext", args, &[1])?;
    let n = next_value(eval, &args[0]).map_err(signal_to_native)?;
    next_value(eval, &n).map_err(signal_to_native)
}

fn native_fnext(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/fnext", args, &[1])?;
    let n = next_value(eval, &args[0]).map_err(signal_to_native)?;
    first_value(eval, &n).map_err(signal_to_native)
}

fn native_nfirst(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/nfirst", args, &[1])?;
    let f = first_value(eval, &args[0]).map_err(signal_to_native)?;
    next_value(eval, &f).map_err(signal_to_native)
}

fn native_list_star(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/list*", args, 1)?;
    if args.len() == 1 {
        return seq_value(eval, &args[0]).map_err(signal_to_native);
    }
    let mut out = args[args.len() - 1].clone();
    for arg in args[..args.len() - 1].iter().rev() {
        out = crate::lazy::native_cons(eval, &[arg.clone(), out])?;
    }
    Ok(out)
}

fn native_conj(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_conj_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_conj_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Vector(Vector::from_vec(Vec::new())));
    }
    if args.len() == 1 {
        return Ok(std::mem::replace(&mut args[0], Value::Nil));
    }
    // Owned accumulator (#227): the coll moves through `conj_one`,
    // so uniquely-owned chains mutate in place after the first
    // step instead of path-copying per step.
    let mut acc = std::mem::replace(&mut args[0], Value::Nil);
    for arg in args.drain(1..) {
        acc = conj_one(eval, acc, arg)?;
    }
    Ok(acc)
}

fn native_assoc(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_assoc_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_assoc_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/assoc", &args, 3)?;
    // Owned coll (#227): fresh-arg chains skip even the first
    // step's path copy (the buffer is the only handle).
    let coll = std::mem::replace(&mut args[0], Value::Nil);
    let mut acc = assoc3(eval, coll, &args[1], args[2].clone())?;
    for pair in args[3..].chunks(2) {
        if pair.len() != 2 {
            return Err(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "assoc expects even number of arguments after map/vector, found odd number"
                    .to_owned(),
            ));
        }
        acc = assoc3(eval, acc, &pair[0], pair[1].clone())?;
    }
    Ok(acc)
}

/// Struct `dissoc`: extras drop and the value stays a struct,
/// but dropping a basis key throws (S12 oracles
/// `sm-dissoc-basis`, `sm-dissoc-extra*`; split from
/// `native_dissoc` for the complexity budget).
fn dissoc_struct(inst: &clj_value::HostInst, keys: &[Value]) -> Result<Value, NativeError> {
    if keys.iter().any(|k| crate::types::is_struct_key(inst, k)) {
        return Err(NativeError::new(
            ThrowClass::RuntimeException,
            "Can't remove struct key".to_owned(),
        ));
    }
    let mut out = inst.content.clone();
    for k in keys {
        out = out.without(k);
    }
    Ok(crate::types::rewrap_maplike(inst, out))
}

/// Record `dissoc` (split from [`native_dissoc`] for the
/// complexity budget): through the content map — dropping a
/// declared field loses recordness (a plain map); anything else
/// stays a record. Metadata travels in both cases.
fn dissoc_record(
    eval: &mut Evaluator,
    inst: &clj_value::HostInst,
    keys: &[Value],
) -> Result<Value, NativeError> {
    let drops_field = keys.iter().any(|k| {
        inst.def
            .fields
            .iter()
            .any(|f| crate::types::is_field_key(f, k))
    });
    let mut out = inst.content.clone();
    for k in keys {
        let k = crate::lazy::realize_key(eval, k).map_err(signal_to_native)?;
        out = out.without(&k);
    }
    if drops_field {
        return Ok(Value::Map(out.with_meta(inst.meta.clone())));
    }
    Ok(crate::types::rewrap_record(inst, out))
}

fn native_dissoc(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_dissoc_owned(eval, args.iter().cloned().collect())
}

/// `dissoc` over an owned map (#271): a uniquely held map (a loop
/// accumulator the AOT moves in) drops keys in place instead of
/// path-copying per call.
pub(crate) fn native_dissoc_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/dissoc", &args, 1)?;
    if args.len() == 1 || matches!(args[0], Value::Nil) {
        return Ok(std::mem::replace(&mut args[0], Value::Nil));
    }
    if let Some(i) = crate::types::struct_inst(&args[0]) {
        return dissoc_struct(i, &args[1..]);
    }
    // Records dissoc through their content map: dropping a declared
    // field loses recordness (a plain map, oracle `rec-dissoc-type`);
    // anything else stays a record (oracle `h-dissoc-missing-type`).
    // Metadata travels in both cases (ad-hoc `dissoc-meta`).
    if let Some(i) = crate::types::record_inst(&args[0]) {
        return dissoc_record(eval, i, &args[1..]);
    }
    let mut out = match std::mem::replace(&mut args[0], Value::Nil) {
        Value::Map(m) => m,
        other => return Err(cce_err(&other, "clojure.lang.IPersistentMap")),
    };
    if out.is_sorted() {
        for k in &args[1..] {
            out = sorted_without_map(eval, &out, k)?;
        }
        return Ok(Value::Map(out));
    }
    for k in &args[1..] {
        let k = crate::lazy::realize_key(eval, k).map_err(signal_to_native)?;
        out = out.without(&k);
    }
    Ok(Value::Map(out))
}

fn native_get(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/get", args, &[2, 3])?;
    let found = lookup_present(eval, &args[0], &args[1])?;
    match found {
        Some(v) => Ok(v),
        None => Ok(args.get(2).cloned().unwrap_or(Value::Nil)),
    }
}

/// `contains?` over a vector (integer keys in range).
fn contains_vector(x: &Vector, key: &Value) -> bool {
    int_key(key).is_some_and(|i| usize::try_from(i as i32).is_ok_and(|idx| idx < x.len()))
}

/// `contains?` over a primitive vector (like vectors).
fn contains_pv(p: &PrimitiveVector, key: &Value) -> bool {
    int_key(key).is_some_and(|i| usize::try_from(i as i32).is_ok_and(|idx| idx < p.len()))
}

/// `contains?` over a string (any numeric index in UTF-16 range).
fn contains_str(s: &str, coll: &Value, key: &Value) -> Result<Value, NativeError> {
    let Some(i) = num_index(key) else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("contains? not supported on type: {}", jvm_class_of(coll)),
        ));
    };
    let len = s.encode_utf16().count();
    Ok(Value::Bool(usize::try_from(i).is_ok_and(|idx| idx < len)))
}

/// `contains?` over anything else (subvectors read; the rest fail).
fn contains_fallback(coll: &Value, key: &Value) -> Result<Value, NativeError> {
    if let Some(sp) = crate::lazy::as_splitvec(coll) {
        return Ok(Value::Bool(int_key(key).is_some_and(|i| {
            usize::try_from(i as i32).is_ok_and(|idx| idx < sp.len())
        })));
    }
    if let Some(sv) = as_subvector(coll) {
        return Ok(Value::Bool(int_key(key).is_some_and(|i| {
            usize::try_from(i as i32).is_ok_and(|idx| idx < sv.len())
        })));
    }
    Err(NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("contains? not supported on type: {}", jvm_class_of(coll)),
    ))
}

/// `contains?` on maps (sorted maps resolve through the comparator).
fn contains_map(eval: &mut Evaluator, m: &Map, key: &Value) -> Result<Value, NativeError> {
    if m.is_sorted() {
        Ok(Value::Bool(sorted_lookup_map(eval, m, key)?.is_some()))
    } else {
        let key = crate::lazy::realize_key(eval, key).map_err(signal_to_native)?;
        Ok(Value::Bool(m.contains_key(&key)))
    }
}

/// `contains?` on sets (sorted sets resolve through the comparator).
fn contains_set(eval: &mut Evaluator, s: &Set, key: &Value) -> Result<Value, NativeError> {
    if s.is_sorted() {
        Ok(Value::Bool(sorted_contains_set(eval, s, key)?))
    } else {
        let key = crate::lazy::realize_key(eval, key).map_err(signal_to_native)?;
        Ok(Value::Bool(s.contains(&key)))
    }
}

fn native_contains(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/contains?", args, &[2])?;
    let (coll, key) = (&args[0], &args[1]);
    // Records contain through their content (oracle `rec-contains`,
    // `h-contains-missing`); structs likewise (S12 oracle
    // `sm-contains`); anything else host takes the unsupported
    // path (oracle `i-contains`).
    if let Some(i) = crate::types::maplike_inst(coll) {
        return contains_map(eval, &i.content, key);
    }
    match coll {
        Value::Nil => Ok(Value::Bool(false)),
        Value::Map(m) => contains_map(eval, m, key),
        Value::Vector(x) => Ok(Value::Bool(contains_vector(x, key))),
        // Entries contain integer keys 0 and 1 (Oracle FX2 `contains`).
        Value::MapEntry { .. } => Ok(Value::Bool(contains_entry(key))),
        Value::Set(s) => contains_set(eval, s, key),
        Value::Str(s) => contains_str(s, coll, key),
        Value::Transient(t) => Ok(Value::Bool(transient_get(t, key)?.is_some())),
        Value::Array(a) => contains_array(a, coll, key),
        Value::PrimitiveVector(p) => Ok(Value::Bool(contains_pv(p, key))),
        _ => contains_fallback(coll, key),
    }
}

/// `contains?` over an entry (split from [`native_contains`] for
/// the complexity budget): integer keys 0 and 1 (Oracle FX2
/// `contains`).
fn contains_entry(key: &Value) -> bool {
    int_key(key).is_some_and(|i| usize::try_from(i as i32).is_ok_and(|idx| idx < 2))
}

/// `contains?` over an array (split from [`native_contains`] for
/// the complexity budget): `Number` keys narrow (`intValue`) to a
/// bounds check, anything else is unsupported (S15 oracles
/// `b-contains`, `f-contains-*`, `e-contains-nil`).
fn contains_array(a: &Array, coll: &Value, key: &Value) -> Result<Value, NativeError> {
    let Some(i) = crate::arrays::array_intvalue_key(key) else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("contains? not supported on type: {}", jvm_class_of(coll)),
        ));
    };
    Ok(Value::Bool(
        usize::try_from(i).is_ok_and(|idx| idx < a.len()),
    ))
}

fn native_find(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/find", args, &[2])?;
    match find_entry(eval, &args[0], &args[1])? {
        // `find` answers real entries (Oracle FX2 `find1class`,
        // `class-f`, `class-fv`).
        Some((k, v)) => Ok(clj_value::map_entry(k, v)),
        None => Ok(Value::Nil),
    }
}

/// `nth` over an indexed value (nil-`not_found` throws, like the
/// 2-arity; vectors, subvectors, and transient vectors share this).
/// `nth` over a transient (only transient vectors are indexed;
/// maps/sets are `nth not supported`, like the JVM).
fn nth_indexed_transient(t: &Transient, n: i32) -> Result<Option<Value>, NativeError> {
    let guard = t.lock();
    match &*guard {
        TransientKind::Vector(tv) => match usize::try_from(n).ok() {
            Some(idx) => Ok(tv.nth(idx).map_err(transient_err)?),
            None => Ok(None),
        },
        kind => Err(nth_unsupported_class(kind.jvm_class())),
    }
}

/// An indexed `nth` hit, else `not-found` (nil-`not-found` throws).
fn nth_indexed_found(
    found: Option<Value>,
    not_found: Option<&Value>,
) -> Result<Value, NativeError> {
    match found {
        Some(v) => Ok(v),
        None => match not_found {
            Some(nf) => Ok(nf.clone()),
            None => Err(NativeError::new(
                ThrowClass::IndexOutOfBoundsException,
                String::new(),
            )),
        },
    }
}

fn nth_indexed(coll: &Value, n: i32, not_found: Option<&Value>) -> Result<Value, NativeError> {
    // Split vectors answer from their spans (the subvector view
    // would materialize every piece).
    if let Some(sp) = crate::lazy::as_splitvec(coll) {
        let found = usize::try_from(n).ok().and_then(|idx| sp.nth(idx));
        return nth_indexed_found(found, not_found);
    }
    nth_indexed_found(nth_indexed_in(coll, n)?, not_found)
}

/// The indexed probe behind [`nth_indexed`] (every shape except
/// split vectors, which answer from their spans).
fn nth_indexed_in(coll: &Value, n: i32) -> Result<Option<Value>, NativeError> {
    Ok(match coll {
        Value::Vector(x) => usize::try_from(n).ok().and_then(|idx| x.get(idx)),
        Value::PrimitiveVector(p) => usize::try_from(n).ok().and_then(|idx| p.get(idx).cloned()),
        Value::MapEntry { key, val, .. } => usize::try_from(n).ok().and_then(|idx| match idx {
            0 => Some((**key).clone()),
            1 => Some((**val).clone()),
            _ => None,
        }),
        Value::Transient(t) => nth_indexed_transient(t, n)?,
        _ => as_subvector(coll).and_then(|sv| usize::try_from(n).ok().and_then(|idx| sv.nth(idx))),
    })
}

/// `nth` over a sequential value by walking (nil-`not_found`
/// throws; negative walks nothing and throws).
fn nth_walk(
    eval: &mut Evaluator,
    coll: &Value,
    n: i32,
    not_found: Option<&Value>,
) -> Result<Value, NativeError> {
    let throw_or_nf = |not_found: Option<&Value>| match not_found {
        Some(nf) => Ok(nf.clone()),
        None => Err(NativeError::new(
            ThrowClass::IndexOutOfBoundsException,
            String::new(),
        )),
    };
    if n < 0 {
        return throw_or_nf(not_found);
    }
    let mut cur = seq_value(eval, coll).map_err(signal_to_native)?;
    for _ in 0..n {
        cur = next_value(eval, &cur).map_err(signal_to_native)?;
        if matches!(cur, Value::Nil) {
            return throw_or_nf(not_found);
        }
    }
    if matches!(cur, Value::Nil) {
        return throw_or_nf(not_found);
    }
    first_value(eval, &cur).map_err(signal_to_native)
}

/// `nth` over a string (chars; `StringIndexOutOfBounds` without `not-found`).
/// A cached string scan: ASCII strings index by byte; other
/// strings carry char→byte offsets plus the UTF-16 length
/// (`count` units). Built once per `Arc` — recipes parse
/// multi-MB inputs char-by-char through `nth`, where a fresh
/// scan per call is quadratic (the 37MB dijkstra input never
/// finishes without this).
#[derive(Clone)]
enum StrScan {
    Ascii,
    Indexed {
        // Shared (not a `Vec`): scans clone per `nth` hit, and a
        // deep table copy per read would dwarf the read itself
        // (#235).
        byte_at: Arc<[usize]>,
        utf16_len: usize,
    },
}

/// One scan-cache slot: the string weakly (a strong ref would
/// pin multi-MB inputs past their use; a bare pointer would
/// ABA when the address is reused).
struct StrScanEntry {
    key: Weak<str>,
    scan: StrScan,
}

/// The string scan cache (8 MRU slots): `nth`/`count` over big
/// strings consult it at every index; small strings decode
/// directly (see [`STR_SCAN_SMALL_MAX`]).
static STR_SCAN_CACHE: Mutex<Vec<StrScanEntry>> = Mutex::new(Vec::new());

/// Strings at or below this byte length never consult the scan
/// cache: a direct decode is bounded (64 chars), and parse loops
/// `nth` thousands of distinct small tokens whose cache entries
/// would churn the global mutex for no reuse (#235).
const STR_SCAN_SMALL_MAX: usize = 64;

// The last string-scan hits, per thread: parse loops `nth` one
// `Arc<str>` millions of times, and search loops alternate two (a
// text and a pattern — #226: one slot ping-ponged through the global
// mutex per read). The memo answers repeats without the global
// cache mutex — and without touching refcounts on hits. The memo
// stores `Weak<str>` so multi-megabyte string buffers are deallocated
// immediately when dropped by the evaluator, avoiding peak RSS bloat.
const LAST_SCAN_SLOTS: usize = 4;

/// The memo slots plus the next slot to reuse.
type ScanMemo = (
    [Option<(std::sync::Weak<str>, StrScan)>; LAST_SCAN_SLOTS],
    usize,
);
const ASCII_CACHE_SLOTS: usize = 4;
type AsciiCache = [(usize, usize, bool); ASCII_CACHE_SLOTS];

thread_local! {
    static LAST_SCAN: std::cell::RefCell<ScanMemo> =
        const { std::cell::RefCell::new(([None, None, None, None], 0)) };
    static ASCII_SLOTS: std::cell::Cell<AsciiCache> =
        const { std::cell::Cell::new([(0, 0, false); ASCII_CACHE_SLOTS]) };
}

#[inline(always)]
fn set_ascii_slot(slot: usize, ptr: usize, len: usize, is_ascii: bool) {
    debug_assert!(slot < ASCII_CACHE_SLOTS);
    let mut slots = ASCII_SLOTS.get();
    slots[slot] = (ptr, len, is_ascii);
    ASCII_SLOTS.set(slots);
}

/// Answer `f` over the memoized scan for a string, if it is one of
/// the last few scanned (no `StrScan` clone on a hit, slot 0 MRU).
fn promote_scan_hit<T>(
    memo: &mut ScanMemo,
    s: &Arc<str>,
    f: impl FnOnce(&StrScan) -> T,
) -> Option<T> {
    let ptr = Arc::as_ptr(s) as *const () as usize;
    let len = s.len();
    for i in 2..LAST_SCAN_SLOTS {
        if let Some((weak, scan)) = &memo.0[i] {
            if std::ptr::eq(weak.as_ptr(), Arc::as_ptr(s)) {
                let hit = f(scan);
                let is_ascii = matches!(scan, StrScan::Ascii);
                memo.0.swap(1, i);
                set_ascii_slot(1, ptr, len, is_ascii);
                return Some(hit);
            }
        }
    }
    None
}

fn with_scan_hit<T>(s: &Arc<str>, f: impl FnOnce(&StrScan) -> T) -> Option<T> {
    LAST_SCAN.with(|memo| {
        let mut memo = memo.borrow_mut();
        if let Some((weak, scan)) = &memo.0[0] {
            if std::ptr::eq(weak.as_ptr(), Arc::as_ptr(s)) {
                return Some(f(scan));
            }
        }
        if let Some((weak, scan)) = &memo.0[1] {
            if std::ptr::eq(weak.as_ptr(), Arc::as_ptr(s)) {
                return Some(f(scan));
            }
        }
        promote_scan_hit(&mut memo, s, f)
    })
}

/// Memoize a string's scan (called on every global-cache hit and
/// fresh build, so repeats skip the mutex), reusing slots in turn.
fn memoize_scan(s: &Arc<str>, scan: &StrScan) {
    let ptr = Arc::as_ptr(s) as *const () as usize;
    let len = s.len();
    let is_ascii = matches!(scan, StrScan::Ascii);
    LAST_SCAN.with(|memo| {
        let mut memo = memo.borrow_mut();
        let slot = memo.1;
        memo.0[slot] = Some((Arc::downgrade(s), scan.clone()));
        memo.1 = (slot + 1) % LAST_SCAN_SLOTS;
        set_ascii_slot(slot, ptr, len, is_ascii);
    });
}

/// Indexed strings past this many chars skip the offset table
/// (a memory cap — the slow path stays correct).
const STR_SCAN_INDEX_CAP: usize = 2_000_000;

/// The cached scan for a string, building it on first use
/// (one O(n) pass; ASCII costs no table). `None` reads over
/// the index cap (uncached slow path).
fn str_scan(s: &Arc<str>) -> Option<StrScan> {
    // (Callers consult the per-thread memo first.)
    {
        let mut cache = STR_SCAN_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = cache
            .iter()
            .position(|e| e.key.upgrade().is_some_and(|a| Arc::ptr_eq(&a, s)))
        {
            let entry = cache.remove(pos);
            let scan = entry.scan.clone();
            cache.insert(0, entry);
            memoize_scan(s, &scan);
            return Some(scan);
        }
    }
    let scan = build_str_scan(s)?;
    {
        let mut cache = STR_SCAN_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache.insert(
            0,
            StrScanEntry {
                key: Arc::downgrade(s),
                scan: scan.clone(),
            },
        );
        cache.truncate(8);
    }
    memoize_scan(s, &scan);
    Some(scan)
}

/// Whether `s` is all ASCII (so `nth` is a byte read), through the
#[inline(always)]
fn ascii_slot_lookup(ptr: usize, len: usize) -> Option<bool> {
    ASCII_SLOTS
        .get()
        .into_iter()
        .find(|&(p, l, _)| p == ptr && l == len)
        .map(|(_, _, is_ascii)| is_ascii)
}

/// scan memo for big strings: O(1) after the first ask, so a loop
/// entered once per item never rescans a long text.
pub(crate) fn str_ascii(s: &Arc<str>) -> bool {
    if s.len() <= STR_SCAN_SMALL_MAX {
        return s.is_ascii();
    }
    let ptr = Arc::as_ptr(s) as *const () as usize;
    if let Some(is_ascii) = ascii_slot_lookup(ptr, s.len()) {
        return is_ascii;
    }
    matches!(str_scan(s), Some(StrScan::Ascii))
}

/// Scan a string once: ASCII short-circuits (no table);
/// otherwise tabulate char→byte offsets (capped).
fn build_str_scan(s: &str) -> Option<StrScan> {
    if s.is_ascii() {
        return Some(StrScan::Ascii);
    }
    let mut byte_at = Vec::new();
    let mut utf16_len = 0usize;
    for (b, c) in s.char_indices() {
        if byte_at.len() >= STR_SCAN_INDEX_CAP {
            return None;
        }
        byte_at.push(b);
        utf16_len += c.len_utf16();
    }
    Some(StrScan::Indexed {
        byte_at: byte_at.into(),
        utf16_len,
    })
}

fn slot_scan_char(
    entry: &Option<(std::sync::Weak<str>, StrScan)>,
    s: &Arc<str>,
    idx: usize,
) -> Option<char> {
    if let Some((weak, scan)) = entry {
        if std::ptr::eq(weak.as_ptr(), Arc::as_ptr(s)) {
            return scan_char_at(s, scan, idx);
        }
    }
    None
}

/// Direct fast-path read from the thread-local scan slots
/// without closure indirection or mutable borrow overhead.
#[inline(always)]
pub(crate) fn fast_str_char_at(s: &Arc<str>, idx: usize) -> Option<char> {
    let ptr = Arc::as_ptr(s) as *const () as usize;
    let len = s.len();
    if let Some(is_ascii) = ascii_slot_lookup(ptr, len) {
        if is_ascii {
            return s.as_bytes().get(idx).map(|b| *b as char);
        }
    } else if len <= STR_SCAN_SMALL_MAX && s.is_ascii() {
        set_ascii_slot(0, ptr, len, true);
        return s.as_bytes().get(idx).map(|b| *b as char);
    }
    LAST_SCAN.with(|memo| {
        let memo = memo.borrow();
        slot_scan_char(&memo.0[0], s, idx).or_else(|| slot_scan_char(&memo.0[1], s, idx))
    })
}

/// The char at `idx` (`None` reads out of bounds). Small strings
/// decode directly — ASCII short-circuits through one SIMD check
/// to a byte load, skipping per-char decode entirely; big
/// strings ride the memoized scan at every index (a cached
/// `Ascii` flag answers in one byte load even for small
/// indexes, where a direct scan would re-decode, #235).
pub(crate) fn str_char_at(s: &Arc<str>, idx: usize) -> Option<char> {
    if let Some(c) = fast_str_char_at(s, idx) {
        return Some(c);
    }
    str_char_at_slow(s, idx)
}

fn str_char_at_slow(s: &Arc<str>, idx: usize) -> Option<char> {
    if s.len() <= STR_SCAN_SMALL_MAX {
        return s.chars().nth(idx);
    }
    if let Some(hit) = with_scan_hit(s, |scan| scan_char_at(s, scan, idx)) {
        return hit;
    }
    match str_scan(s) {
        Some(scan) => scan_char_at(s, &scan, idx),
        None => s.chars().nth(idx),
    }
}

/// The char at `idx` through a string's scan.
fn scan_char_at(s: &str, scan: &StrScan, idx: usize) -> Option<char> {
    match scan {
        StrScan::Ascii => s.as_bytes().get(idx).map(|b| *b as char),
        StrScan::Indexed { byte_at, .. } => byte_at.get(idx).and_then(|b| s[*b..].chars().next()),
    }
}

/// The UTF-16 length (`count` units), via the scan cache for
/// big strings. Shared with the `clj-rt` string fast path (same
/// function, zero drift — the fast path skips only the
/// mutex/evaluator roundtrip, never this).
pub fn str_utf16_len(s: &Arc<str>) -> usize {
    if s.len() <= 256 {
        if s.is_ascii() {
            return s.len();
        }
        return s.encode_utf16().count();
    }
    let memo = with_scan_hit(s, |scan| match scan {
        StrScan::Ascii => s.len(),
        StrScan::Indexed { utf16_len, .. } => *utf16_len,
    });
    if let Some(n) = memo {
        return n;
    }
    match str_scan(s) {
        Some(StrScan::Ascii) => s.len(),
        Some(StrScan::Indexed { utf16_len, .. }) => utf16_len,
        None => s.encode_utf16().count(),
    }
}

/// `nth` over a string (chars; `StringIndexOutOfBounds` without
/// `not-found`). Shared with the `clj-rt` string fast path (same
/// function, zero drift — the caller coerces the index first via
/// [`crate::core::nth_index`], exactly like [`native_nth`]).
pub fn nth_str(s: &Arc<str>, n: i32, not_found: Option<&Value>) -> Result<Value, NativeError> {
    let found = usize::try_from(n)
        .ok()
        .and_then(|idx| str_char_at(s, idx).map(Value::Char));
    match found {
        Some(c) => Ok(c),
        None => {
            if let Some(nf) = not_found {
                return Ok(nf.clone());
            }
            Err(NativeError::new(
                ThrowClass::StringIndexOutOfBoundsException,
                format!("Index {} out of bounds for length {}", n, str_utf16_len(s)),
            ))
        }
    }
}

pub(crate) fn native_nth(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_nth_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_nth_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/nth", &args, &[2, 3])?;
    // The index coerces first (before the collection dispatches).
    let n = nth_index(&args[1])?;
    let not_found = args.get(2).cloned();
    let coll_v = take_walk_coll(&mut args, 0);
    let coll = &coll_v;
    if let Some(fast) = nth_fast(coll, n, &not_found) {
        return fast;
    }
    if let Value::Array(a) = coll {
        return nth_array(a, n, not_found);
    }
    // (Strings dispatch in `nth_fast`, above.)
    if matches!(coll, Value::List(_)) || is_seq_value(coll) {
        return nth_walk(eval, coll, n, not_found.as_ref());
    }
    Err(nth_unsupported(coll))
}

/// The `nth` fast paths (`nil`/string/negative-`not-found`/
/// chunk/indexed; `None` falls through to array/seq dispatch).
/// Strings dispatch before the chunk/subvector probes (disjoint
/// types — search loops `nth` strings millions of times, #235).
fn nth_fast(coll: &Value, n: i32, not_found: &Option<Value>) -> Option<Result<Value, NativeError>> {
    if matches!(coll, Value::Nil) {
        return Some(Ok(not_found.clone().unwrap_or(Value::Nil)));
    }
    if let Value::Str(s) = coll {
        return Some(nth_str(s, n, not_found.as_ref()));
    }
    if not_found.is_some() && n < 0 {
        return Some(Ok(not_found.clone().unwrap_or(Value::Nil)));
    }
    // Chunks read their raw slots (nil holes below capacity, S12
    // oracles `ch-nth-oob`, `ch-nth-cap`).
    if let Some(c) = crate::collmisc::as_chunk(coll) {
        return Some(crate::collmisc::chunk_nth(c, n, not_found.as_ref()));
    }
    if matches!(
        coll,
        Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || crate::lazy::as_splitvec(coll).is_some()
        || as_subvector(coll).is_some()
        || matches!(coll, Value::Transient(_))
    {
        return Some(nth_indexed(coll, n, not_found.as_ref()));
    }
    None
}

/// The `nth` array arm (`not-found` on OOB, else the empty
/// `AIOOBE` — S15 oracles `a-nth`, `a-nth-nf`, `a-nth-oob`,
/// `e-nth-float`, `n-arr-str`, `n-arr-nil`).
fn nth_array(a: &clj_value::Array, n: i32, not_found: Option<Value>) -> Result<Value, NativeError> {
    let found = usize::try_from(n).ok().and_then(|idx| {
        if idx < a.len() {
            Some(crate::arrays::array_read(a, idx))
        } else {
            None
        }
    });
    match (found, not_found) {
        (Some(v), _) => Ok(v),
        (None, Some(nf)) => Ok(nf),
        (None, None) => Err(NativeError::new(
            ThrowClass::ArrayIndexOutOfBoundsException,
            String::new(),
        )),
    }
}

/// The `nth not supported` error (`getSimpleName` rule).
fn nth_unsupported(coll: &Value) -> NativeError {
    nth_unsupported_class(&jvm_class_of(coll))
}

/// `nth not supported` over a precomputed class (the transient
/// path holds the buffer lock while reporting).
fn nth_unsupported_class(class: &str) -> NativeError {
    NativeError::new(
        ThrowClass::UnsupportedOperationException,
        format!("nth not supported on this type: {}", simple_name_of(class)),
    )
}

fn native_vec(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_vec_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_vec_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/vec", &args, &[1])?;
    if matches!(args[0], Value::Vector(_))
        || crate::lazy::as_splitvec(&args[0]).is_some()
        || as_subvector(&args[0]).is_some()
    {
        return Ok(std::mem::replace(&mut args[0], Value::Nil).with_meta(None));
    }
    // `RT.toArray`: non-seqables fail the conversion, not the seq
    // (`:vec-num`, `:vec-trans`).
    if !is_seqable(&args[0]) {
        return Err(NativeError::new(
            ThrowClass::RuntimeException,
            format!(
                "Unable to convert: class {} to Object[]",
                jvm_class_of(&args[0])
            ),
        ));
    }
    let coll = take_walk_coll(&mut args, 0);
    let vec = collect_packed(eval, coll).map_err(signal_to_native)?;
    Ok(Value::Vector(vec))
}

fn native_merge(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Nil);
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    let mut acc = merge2(eval, args[0].clone(), &args[1])?;
    for arg in &args[2..] {
        acc = merge2(eval, acc, arg)?;
    }
    Ok(acc)
}

fn native_merge_with(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/merge-with", args, 1)?;
    if args.len() == 1 {
        return Ok(Value::Nil);
    }
    if args.len() == 2 {
        return Ok(args[1].clone());
    }
    let mut acc = merge_with3(eval, &args[0], args[1].clone(), &args[2])?;
    for arg in &args[3..] {
        acc = merge_with3(eval, &args[0], acc, arg)?;
    }
    Ok(acc)
}

fn native_select_keys(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/select-keys", args, &[2])?;
    let mut ret = new_map();
    let mut cur = seq_value(eval, &args[1]).map_err(signal_to_native)?;
    loop {
        if matches!(cur, Value::Nil) {
            break;
        }
        let k = first_value(eval, &cur).map_err(signal_to_native)?;
        if let Some(v) = lookup_present(eval, &args[0], &k)? {
            ret = ret.insert(k, v);
        }
        cur = next_value(eval, &cur).map_err(signal_to_native)?;
    }
    if let Some(md) = args[0].meta() {
        ret = ret.with_meta(Some(md.clone()));
    }
    Ok(Value::Map(ret))
}

fn native_update(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    native_update_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_update_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/update", &args, 3)?;
    let mut m = std::mem::replace(&mut args[0], Value::Nil);
    let k = std::mem::replace(&mut args[1], Value::Nil);
    let f = std::mem::replace(&mut args[2], Value::Nil);
    let old = take_lookup(eval, &mut m, &k)?.unwrap_or(Value::Nil);
    let mut call_args: SmallVec<[Value; 4]> = SmallVec::with_capacity(args.len() - 2);
    call_args.push(old);
    call_args.extend(args.drain(3..));
    let new = eval.call_owned(&f, call_args).map_err(signal_to_native)?;
    assoc3(eval, m, &k, new)
}

fn native_update_in(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    native_update_in_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_update_in_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/update-in", &args, 3)?;
    let m = std::mem::replace(&mut args[0], Value::Nil);
    let (k, ks) =
        KeyStep::from_val(eval, &args[1])?.unwrap_or((Value::Nil, KeyStep::Seq(Value::Nil)));
    update_in_up(eval, m, &k, ks, &args[2], &args[3..])
}

fn native_assoc_in(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    native_assoc_in_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_assoc_in_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/assoc-in", &args, &[3])?;
    let m = std::mem::replace(&mut args[0], Value::Nil);
    let v = std::mem::replace(&mut args[2], Value::Nil);
    let (k, ks) =
        KeyStep::from_val(eval, &args[1])?.unwrap_or((Value::Nil, KeyStep::Seq(Value::Nil)));
    assoc_in_up(eval, m, &k, ks, v)
}

fn native_get_in(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/get-in", args, &[2, 3])?;
    let mut cur = args[0].clone();
    let mut step = KeyStep::from_val(eval, &args[1])?;
    while let Some((k, next)) = step {
        match lookup_present(eval, &cur, &k)? {
            Some(v) => cur = v,
            None => return Ok(args.get(2).cloned().unwrap_or(Value::Nil)),
        }
        step = next.next(eval)?;
    }
    Ok(cur)
}

fn native_zipmap(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/zipmap", args, &[2])?;
    if let Some(map) = zipmap_chars_longs(&args[0], &args[1]) {
        return Ok(Value::Map(map));
    }
    let mut ks = seq_value(eval, &args[0]).map_err(signal_to_native)?;
    let mut vs = seq_value(eval, &args[1]).map_err(signal_to_native)?;
    let mut out = ZipmapBuild::default();
    while let Some((k, v)) = zipmap_next(eval, &mut ks, &mut vs).map_err(signal_to_native)? {
        out.push(k, v);
    }
    Ok(Value::Map(out.finish()))
}

/// `(zipmap "0123456789" (range))` (most snippets) without the walk
/// (#271): a string's chars against the core `(range)` iterate or a
/// counted long range run no user code and memoize nothing anyone else
/// sees, so the pairs are the walk's own, fed to the same build. `None`:
/// anything else, or a value past the long range (the walk promotes or
/// throws there).
fn zipmap_chars_longs(keys: &Value, vals: &Value) -> Option<Map> {
    let Value::Str(s) = keys else {
        return None;
    };
    let (start, step, count) = crate::lazy::core_long_iterate_head(vals)
        .map(|start| (start, 1, i64::MAX))
        .or_else(|| crate::seqs::long_range(vals))?;
    let mut out = ZipmapBuild::default();
    for (i, c) in (0..count).zip(s.chars()) {
        let v = i.checked_mul(step).and_then(|d| start.checked_add(d))?;
        out.push(Value::Char(c), Value::Int(v));
    }
    Some(out.finish())
}

/// `zipmap`'s next pair, forcing like `core.clj`'s loop: both heads,
/// then both tails (`None` once either side runs out).
fn zipmap_next(
    eval: &mut Evaluator,
    ks: &mut Value,
    vs: &mut Value,
) -> Result<Option<(Value, Value)>, Signal> {
    if matches!(ks, Value::Nil) || matches!(vs, Value::Nil) {
        return Ok(None);
    }
    let k = first_value(eval, ks)?;
    let v = first_value(eval, vs)?;
    let k = crate::lazy::realize_key(eval, &k)?;
    *ks = next_value(eval, ks)?;
    *vs = next_value(eval, vs)?;
    Ok(Some((k, v)))
}

/// `zipmap`'s result under construction: pairs first and one build
/// after (the `assoc` chain's exact map — a small result interns its
/// key shape once instead of once per key; `(zipmap "0123456789"
/// (range))` opens most snippets), in-place inserts past
/// [`ZIPMAP_BULK`] pairs.
#[derive(Default)]
struct ZipmapBuild {
    pairs: Vec<(Value, Value)>,
    map: Option<Map>,
}

impl ZipmapBuild {
    fn push(&mut self, k: Value, v: Value) {
        match &mut self.map {
            Some(map) => map.insert_mut(k, v),
            None if self.pairs.len() < ZIPMAP_BULK => self.pairs.push((k, v)),
            None => {
                let mut map = Map::from_kvs_last_wins(std::mem::take(&mut self.pairs));
                map.insert_mut(k, v);
                self.map = Some(map);
            }
        }
    }

    fn finish(self) -> Map {
        self.map
            .unwrap_or_else(|| Map::from_kvs_last_wins(self.pairs))
    }
}

/// The pairs [`ZipmapBuild`] builds in one step (the shaped-map
/// ceiling: `from_kvs_last_wins` dedupes by scanning, fine this
/// small); longer inputs continue with in-place inserts.
const ZIPMAP_BULK: usize = 64;

fn native_array_map(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() % 2 == 1 {
        let last = args.last().unwrap_or(&Value::Nil);
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("No value supplied for key: {}", str_of(last)),
        ));
    }
    let mut out = new_map();
    for pair in args.chunks(2) {
        out = out.insert(pair[0].clone(), pair[1].clone());
    }
    Ok(Value::Map(out.with_hashy(false)))
}

/// The empty `IndexOutOfBoundsException`.
fn ioobe() -> NativeError {
    NativeError::new(ThrowClass::IndexOutOfBoundsException, String::new())
}

pub(crate) fn native_transient_owned(
    _eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/transient", &args, &[1])?;
    let val = args.pop().unwrap();
    match val {
        Value::Vector(x) => {
            let meta = x.meta().cloned();
            Ok(Value::Transient(Transient::with_meta(
                TransientKind::Vector(TransientVector::from_vector_owned(x)),
                meta,
            )))
        }
        // Sorted maps/sets are not editable on the JVM either
        // (probe `trans-sorted`).
        Value::Map(m) if m.is_sorted() => {
            Err(cce_err(&Value::Map(m), "clojure.lang.IEditableCollection"))
        }
        Value::Map(m) => {
            let meta = m.meta().cloned();
            Ok(Value::Transient(Transient::with_meta(
                TransientKind::Map(TransientMap::from_map(&m)),
                meta,
            )))
        }
        Value::Set(s) if s.is_sorted() => {
            Err(cce_err(&Value::Set(s), "clojure.lang.IEditableCollection"))
        }
        Value::Set(s) => {
            let meta = s.meta().cloned();
            Ok(Value::Transient(Transient::with_meta(
                TransientKind::Set(TransientSet::from_set(&s)),
                meta,
            )))
        }
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.IEditableCollection.asTransient()\" because \"coll\" is null"
                .to_owned(),
        )),
        other => Err(cce_err(&other, "clojure.lang.IEditableCollection")),
    }
}

fn native_transient(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    native_transient_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_persistent_owned(
    _eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/persistent!", &args, &[1])?;
    let val = args.pop().unwrap();
    match val {
        Value::Transient(t) => {
            let meta = t.source_meta().cloned();
            let mut guard = t.lock();
            match &mut *guard {
                TransientKind::Vector(tv) => Ok(Value::Vector(
                    tv.persistent().map_err(transient_err)?.with_meta(meta),
                )),
                TransientKind::Map(tm) => Ok(Value::Map(
                    tm.persistent().map_err(transient_err)?.with_meta(meta),
                )),
                TransientKind::Set(ts) => Ok(Value::Set(
                    ts.persistent().map_err(transient_err)?.with_meta(meta),
                )),
            }
        }
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.ITransientCollection.persistent()\" because \"coll\" is null"
                .to_owned(),
        )),
        other => Err(cce_err(&other, "clojure.lang.ITransientCollection")),
    }
}

fn native_persistent(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    native_persistent_owned(eval, args.iter().cloned().collect())
}

/// The `conj!` map plan (computed unlocked: seqing the
/// element runs user code, and the head can be this transient
/// itself — locking twice deadlocks).
fn conj_bang_plan(
    eval: &mut Evaluator,
    t: &Transient,
    x: &Value,
) -> Result<Option<MapConjPlan>, NativeError> {
    // Bind the discriminant first: a `t.lock()` temporary in the
    // `if` condition would live through the branches, holding the
    // lock while `map_conj_plan` seqs (self-deadlock on `(conj! t
    // t)`).
    let is_map = matches!(&*t.lock(), TransientKind::Map(_));
    let plan = if is_map {
        Some(map_conj_plan(eval, x)?)
    } else {
        None
    };
    Ok(plan)
}

/// `conj!` applying a plan to a transient map (lock held).
fn conj_bang_apply_map(
    tm: &mut TransientMap,
    plan: Option<MapConjPlan>,
) -> Result<(), NativeError> {
    match plan {
        Some(MapConjPlan::Nil) | None => {}
        Some(MapConjPlan::Pair(k, v)) => {
            tm.assoc(k, v).map_err(transient_err)?;
        }
        Some(MapConjPlan::Entries(entries)) => {
            for (k, v) in entries {
                tm.assoc(k, v).map_err(transient_err)?;
            }
        }
    }
    Ok(())
}

/// `conj!` applying one element to a locked transient (lock held).
fn conj_bang_apply(
    kind: &mut TransientKind,
    plan: Option<MapConjPlan>,
    x: &Value,
) -> Result<(), NativeError> {
    match kind {
        TransientKind::Vector(tv) => {
            tv.conj(x.clone()).map_err(transient_err)?;
        }
        TransientKind::Map(tm) => conj_bang_apply_map(tm, plan)?,
        TransientKind::Set(ts) => {
            ts.conj(x.clone()).map_err(transient_err)?;
        }
    }
    Ok(())
}

/// `conj!` onto a transient (plan first, then apply under the lock).
fn conj_bang_transient(
    eval: &mut Evaluator,
    t: &Transient,
    x: &Value,
) -> Result<Value, NativeError> {
    let plan = conj_bang_plan(eval, t, x)?;
    let mut guard = t.lock();
    conj_bang_apply(&mut guard, plan, x)?;
    Ok(Value::Transient(t.clone()))
}

fn native_conj_bang(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // JVM arities: `([] [coll] [coll x] [coll x & xs])` — variadic.
    check_min_arity("clojure.core/conj!", args, 0)?;
    if args.is_empty() {
        return Ok(Value::Transient(Transient::new(TransientKind::Vector(
            TransientVector::new(),
        ))));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    match &args[0] {
        Value::Transient(t) => {
            for x in &args[1..] {
                conj_bang_transient(eval, t, x)?;
            }
            Ok(Value::Transient(t.clone()))
        }
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.ITransientCollection.conj(Object)\" because \"coll\" is null"
                .to_owned(),
        )),
        other => Err(cce_err(other, "clojure.lang.ITransientCollection")),
    }
}

fn native_assoc_bang(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/assoc!", args, 3)?;
    let Value::Transient(t) = &args[0] else {
        if matches!(args[0], Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"clojure.lang.ITransientAssociative.assoc(Object, Object)\" because \"coll\" is null"
                    .to_owned(),
            ));
        }
        return Err(cce_err(&args[0], "clojure.lang.ITransientAssociative"));
    };
    let is_set = {
        let mut guard = t.lock();
        match &mut *guard {
            TransientKind::Vector(tv) => {
                assoc_bang_one(tv, &args[1], &args[2])?;
                for chunk in args[3..].chunks(2) {
                    let v = chunk.get(1).unwrap_or(&Value::Nil);
                    assoc_bang_one(tv, &chunk[0], v)?;
                }
                false
            }
            TransientKind::Map(tm) => {
                tm.assoc(args[1].clone(), args[2].clone())
                    .map_err(transient_err)?;
                for chunk in args[3..].chunks(2) {
                    let v = chunk.get(1).cloned().unwrap_or(Value::Nil);
                    tm.assoc(chunk[0].clone(), v).map_err(transient_err)?;
                }
                false
            }
            TransientKind::Set(_) => true,
        }
    };
    if is_set {
        // The class is read after unlocking (`jvm_class_of` locks
        // transients itself — locking twice deadlocks), and only on
        // this error path (naming the class allocates).
        let from = jvm_class_of(&args[0]);
        return Err(NativeError::new(
            ThrowClass::ClassCastException,
            cce_message(&from, "clojure.lang.ITransientAssociative"),
        ));
    }
    Ok(Value::Transient(t.clone()))
}

/// One `assoc!` triple on a transient vector (integer keys only).
fn assoc_bang_one(tv: &mut TransientVector, k: &Value, v: &Value) -> Result<(), NativeError> {
    let Some(i) = int_key(k) else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "Key must be integer".to_owned(),
        ));
    };
    let Some(idx) = usize::try_from(i as i32).ok() else {
        return Err(ioobe());
    };
    tv.assoc(idx, v.clone()).map_err(transient_err)
}

fn native_dissoc_bang(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/dissoc!", args, 2)?;
    let Value::Transient(t) = &args[0] else {
        if matches!(args[0], Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"clojure.lang.ITransientMap.without(Object)\" because \"map\" is null"
                    .to_owned(),
            ));
        }
        return Err(cce_err(&args[0], "clojure.lang.ITransientMap"));
    };
    // Read the class before locking (`jvm_class_of` locks
    // transients itself — locking twice deadlocks).
    let from = jvm_class_of(&args[0]);
    {
        let mut guard = t.lock();
        match &mut *guard {
            TransientKind::Map(tm) => {
                for k in &args[1..] {
                    tm.dissoc(k).map_err(transient_err)?;
                }
            }
            _ => {
                return Err(NativeError::new(
                    ThrowClass::ClassCastException,
                    cce_message(&from, "clojure.lang.ITransientMap"),
                ));
            }
        }
    }
    Ok(Value::Transient(t.clone()))
}

fn native_disj_bang(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/disj!", args, 1)?;
    // The 1-arity is the identity over ANY value (the JVM decl
    // carries no hint there — S5, oracle `e-arity1`).
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    let Value::Transient(t) = &args[0] else {
        if matches!(args[0], Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"clojure.lang.ITransientSet.disjoin(Object)\" because \"set\" is null"
                    .to_owned(),
            ));
        }
        return Err(cce_err(&args[0], "clojure.lang.ITransientSet"));
    };
    // Read the class before locking (`jvm_class_of` locks
    // transients itself — locking twice deadlocks).
    let from = jvm_class_of(&args[0]);
    {
        let mut guard = t.lock();
        match &mut *guard {
            TransientKind::Set(ts) => {
                for k in &args[1..] {
                    ts.disj(k).map_err(transient_err)?;
                }
            }
            _ => {
                return Err(NativeError::new(
                    ThrowClass::ClassCastException,
                    cce_message(&from, "clojure.lang.ITransientSet"),
                ));
            }
        }
    }
    Ok(Value::Transient(t.clone()))
}

fn native_pop_bang(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/pop!", args, &[1])?;
    let Value::Transient(t) = &args[0] else {
        if matches!(args[0], Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"clojure.lang.ITransientVector.pop()\" because \"coll\" is null"
                    .to_owned(),
            ));
        }
        return Err(cce_err(&args[0], "clojure.lang.ITransientVector"));
    };
    // Read the class before locking (`jvm_class_of` locks
    // transients itself — locking twice deadlocks).
    let from = jvm_class_of(&args[0]);
    {
        let mut guard = t.lock();
        match &mut *guard {
            TransientKind::Vector(tv) => {
                tv.pop().map_err(transient_err)?;
            }
            _ => {
                return Err(NativeError::new(
                    ThrowClass::ClassCastException,
                    cce_message(&from, "clojure.lang.ITransientVector"),
                ));
            }
        }
    }
    Ok(Value::Transient(t.clone()))
}

/// The `PersistentQueue` emulation mark: queues ride on
/// vectors (insertion-ordered, `conj`/`count`/`seq`/`nth`
/// already correct), distinguished by this metadata key so
/// `peek`/`pop` take the front. `conj` and `pop` preserve
/// metadata, so queues stay queues through BFS loops. The mark
/// is visible to `meta` (a documented narrowing — real queues
/// print `#object[…]`; these print as vectors).
fn queue_mark_key() -> Value {
    Value::Keyword(Keyword::new(Some("clojure-rs"), "queue"))
}

/// Whether a vector is a queue emulation (see
/// [`empty_queue_value`]).
fn is_queue_vec(x: &Vector) -> bool {
    x.meta().and_then(|m| m.get(&queue_mark_key())).is_some()
}

/// `clojure.lang.PersistentQueue/EMPTY`: a marked empty vector
/// (FIFO via the `peek`/`pop` arms below; 4 recipes BFS with
/// it).
pub(crate) fn empty_queue_value() -> Value {
    Value::Vector(
        Vector::from_vec(vec![])
            .with_meta(Some(new_map().insert(queue_mark_key(), Value::Bool(true)))),
    )
}

/// `pop` over a queue emulation: drops the FRONT (empty pops
/// answer the queue itself, like the JVM — oracle).
fn pop_queue(x: &Vector) -> Value {
    if x.is_empty() {
        return Value::Vector(x.clone());
    }
    Value::Vector(Vector::from_vec(x.iter().skip(1).collect()).with_meta(x.meta().cloned()))
}

fn native_peek(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/peek", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::List(l) => Ok(l.first().cloned().unwrap_or(Value::Nil)),
        Value::Vector(x) if is_queue_vec(x) => Ok(x.iter().next().unwrap_or(Value::Nil)),
        Value::Vector(x) => Ok(x.iter().next_back().unwrap_or(Value::Nil)),
        Value::PrimitiveVector(p) => Ok(p.iter().last().cloned().unwrap_or(Value::Nil)),
        Value::MapEntry { val, .. } => Ok((**val).clone()),
        _ => {
            if let Some(sp) = crate::lazy::as_splitvec(&args[0]) {
                return Ok(sp
                    .len()
                    .checked_sub(1)
                    .and_then(|i| sp.nth(i))
                    .unwrap_or(Value::Nil));
            }
            if let Some(sv) = as_subvector(&args[0]) {
                return Ok(sv.iter().last().unwrap_or(Value::Nil));
            }
            Err(cce_err(&args[0], "clojure.lang.IPersistentStack"))
        }
    }
}

/// `pop` over a list (empty lists throw; empty tails drop metadata-clean).
fn pop_list(l: &List) -> Result<Value, NativeError> {
    if l.is_empty() {
        return Err(NativeError::new(
            ThrowClass::IllegalStateException,
            "Can't pop empty list".to_owned(),
        ));
    }
    match l.drop_first() {
        Some(tail) if !tail.is_empty() => Ok(Value::List(tail)),
        _ => Ok(Value::List(List::new_sync().with_meta(l.meta().cloned()))),
    }
}

/// `pop` over a vector (empty vectors throw).
fn pop_vector(x: Vector) -> Result<Value, NativeError> {
    if x.is_empty() {
        return Err(NativeError::new(
            ThrowClass::IllegalStateException,
            "Can't pop empty vector".to_owned(),
        ));
    }
    Ok(Value::Vector(
        x.drop_last_owned().expect("non-empty vector has a tail"),
    ))
}

/// `pop` over a primitive vector (stays typed; empty pops throw
/// like vectors).
fn pop_pv(p: PrimitiveVector) -> Result<Value, NativeError> {
    if p.is_empty() {
        return Err(NativeError::new(
            ThrowClass::IllegalStateException,
            "Can't pop empty vector".to_owned(),
        ));
    }
    Ok(Value::PrimitiveVector(
        p.drop_last().expect("non-empty vector has a tail"),
    ))
}

/// `pop` over anything else (subvectors shrink; the rest fail).
fn pop_fallback(arg: &Value) -> Result<Value, NativeError> {
    if let Some(sp) = crate::lazy::as_splitvec(arg) {
        if sp.len() == 1 {
            return Ok(Value::Vector(Vector::from_vec(Vec::new())));
        }
        return Ok(Value::Extension(Arc::new(sp.drop_last())));
    }
    if let Some(sv) = as_subvector(arg) {
        if sv.len() == 1 {
            return Ok(Value::Vector(Vector::from_vec(Vec::new())));
        }
        // Empty subvectors never exist (`subvec` returns a plain
        // vector when `start == end`), so `end >= 1` here.
        let end = sv.end().checked_sub(1).expect("subvector is never empty");
        return Ok(Value::Extension(Arc::new(
            SubVectorValue::new(sv.vec().clone(), sv.start(), end)
                .with_orig_start(sv.orig_start())
                .with_meta(sv.meta().cloned()),
        )));
    }
    Err(cce_err(arg, "clojure.lang.IPersistentStack"))
}

fn native_pop(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/pop", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::List(l) => pop_list(l),
        Value::Vector(x) if is_queue_vec(x) => Ok(pop_queue(x)),
        Value::Vector(x) => pop_vector(x.clone()),
        Value::PrimitiveVector(p) => pop_pv(p.clone()),
        // Popping an entry answers its 1-vector (Oracle FX2
        // `peek-pop`, `pop-class`).
        Value::MapEntry { key, .. } => Ok(Value::Vector(Vector::from_vec(vec![(**key).clone()]))),
        _ => pop_fallback(&args[0]),
    }
}

/// Three-way `subvec` over an already-defaulted end.
/// `subvec` on a split vector: window in place (no piece
/// materializes). `None` passes through to the generic path.
fn splitvec_subvec(
    v: &Value,
    start_v: &Value,
    end_v: &Value,
) -> Option<Result<Value, NativeError>> {
    let sp = crate::lazy::as_splitvec(v)?;
    Some((|| {
        let s = nth_index(start_v)?;
        let e = nth_index(end_v)?;
        let (Ok(su), Ok(eu)) = (usize::try_from(s), usize::try_from(e)) else {
            return Err(ioobe());
        };
        if eu < su || eu > sp.len() {
            return Err(ioobe());
        }
        if su == eu {
            return Ok(Value::Vector(Vector::from_vec(Vec::new())));
        }
        Ok(Value::Extension(Arc::new(
            sp.window(su, eu).with_meta(None),
        )))
    })())
}

pub(crate) fn subvec3(v: &Value, start_v: &Value, end_v: &Value) -> Result<Value, NativeError> {
    if let Some(r) = splitvec_subvec(v, start_v, end_v) {
        return r;
    }
    let (backing, base, orig_base, len) = match v {
        Value::Vector(x) => (x.clone(), 0, 0, x.len()),
        // `subvec` of a primitive vector is a plain subvector
        // (S15 oracle `j-subvec-cls`).
        Value::PrimitiveVector(p) => (Vector::from_vec(p.items()), 0, 0, p.len()),
        Value::MapEntry { key, val, .. } => (
            Vector::from_vec(vec![(**key).clone(), (**val).clone()]),
            0,
            0,
            2,
        ),
        _ => match as_subvector(v) {
            Some(sv) => (sv.vec().clone(), sv.start(), sv.orig_start(), sv.len()),
            None => return Err(cce_err(v, "clojure.lang.IPersistentVector")),
        },
    };
    let s = nth_index(start_v)?;
    let e = nth_index(end_v)?;
    let (Ok(su), Ok(eu)) = (usize::try_from(s), usize::try_from(e)) else {
        return Err(ioobe());
    };
    if eu < su || eu > len {
        return Err(ioobe());
    }
    if su == eu {
        return Ok(Value::Vector(Vector::from_vec(Vec::new())));
    }
    let win = eu - su;
    let sv = if win <= SUBVEC_COMPACT_MAX && !backing.is_flat() {
        let flat = backing.slice_packed(base + su, base + eu);
        SubVectorValue::new(flat, 0, win).with_orig_start(orig_base + su)
    } else {
        SubVectorValue::new(backing, base + su, base + eu).with_orig_start(orig_base + su)
    };
    Ok(Value::Extension(Arc::new(sv)))
}

fn native_subvec(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/subvec", args, &[2, 3])?;
    if args.len() == 2 {
        let end = Value::Int(count_of(eval, &args[0])?);
        return subvec3(&args[0], &args[1], &end);
    }
    subvec3(&args[0], &args[1], &args[2])
}

fn native_replace(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/replace", args, &[2])?;
    let (smap, coll) = (&args[0], &args[1]);
    if !(matches!(
        coll,
        Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || as_subvector(coll).is_some())
    {
        return Ok(replace_thunk(smap.clone(), coll.clone()));
    }
    let n = match coll {
        Value::Vector(x) => x.len(),
        Value::MapEntry { .. } => 2,
        Value::PrimitiveVector(p) => p.len(),
        _ => as_subvector(coll).map(|sv| sv.len()).unwrap_or(0),
    };
    let mut acc = coll.clone();
    for i in 0..n {
        let k = Value::Int(i as i64);
        let x = lookup_present(eval, &acc, &k)?.unwrap_or(Value::Nil);
        if let Some((_, v)) = find_entry(eval, smap, &x)? {
            acc = assoc3(eval, acc, &k, v)?;
        }
    }
    Ok(acc)
}

fn native_into(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    // Borrow-path compat shim (see [`OwnedNativeImpl`]).
    native_into_owned(eval, args.iter().cloned().collect())
}

pub(crate) fn native_into_owned(
    eval: &mut Evaluator,
    mut args: SmallVec<[Value; 4]>,
) -> Result<Value, NativeError> {
    check_arity("clojure.core/into", &args, &[0, 1, 2, 3])?;
    if args.is_empty() {
        return Ok(Value::Vector(Vector::from_vec(Vec::new())));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    if args.len() == 3 {
        // `(into to xform from)` (T1200 D1, oracle `into-*`).
        let from = take_walk_coll(&mut args, 2);
        return crate::xform::into_xform(eval, args[0].clone(), &args[1], from);
    }
    let from = take_walk_coll(&mut args, 1);
    let to = take_walk_coll(&mut args, 0);
    into_inner(eval, to, from)
}

#[allow(clippy::mutable_key_type)]
fn native_distinctp(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/distinct?", args, 1)?;
    if args.len() == 1 {
        return Ok(Value::Bool(true));
    }
    if args.len() == 2 {
        let eq = equiv_vals(eval, &args[0], &args[1]).map_err(signal_to_native)?;
        return Ok(Value::Bool(!eq));
    }
    if equiv_vals(eval, &args[0], &args[1]).map_err(signal_to_native)? {
        return Ok(Value::Bool(false));
    }
    let mut seen = HashSet::new();
    seen.insert(args[0].clone());
    seen.insert(args[1].clone());
    for arg in &args[2..] {
        if seen.contains(arg) {
            return Ok(Value::Bool(false));
        }
        seen.insert(arg.clone());
    }
    Ok(Value::Bool(true))
}

/// `key`/`val`: there is no `MapEntry` type (map seqs and `find`
/// yield `[k v]` vectors), so these read slots 0/1 through `nth` —
/// entry-shaped inputs work, anything else fails exactly like the
/// same `nth`. (On the JVM, non-entries fail the `Map$Entry` cast
/// instead — the one observable edge of the missing type.)
/// Extract the key from a `MapEntry`, or raise the JVM's `Map$Entry` error.
pub(crate) fn entry_key(v: &Value) -> Result<Value, NativeError> {
    match v {
        Value::MapEntry { key, .. } => Ok((**key).clone()),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.util.Map$Entry.getKey()\" because \"e\" is null".to_owned(),
        )),
        other => Err(cce_err(other, "java.util.Map$Entry")),
    }
}

/// Extract the value from a `MapEntry`, or raise the JVM's `Map$Entry` error.
pub(crate) fn entry_val(v: &Value) -> Result<Value, NativeError> {
    match v {
        Value::MapEntry { val, .. } => Ok((**val).clone()),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.util.Map$Entry.getValue()\" because \"e\" is null".to_owned(),
        )),
        other => Err(cce_err(other, "java.util.Map$Entry")),
    }
}

fn native_key(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/key", args, &[1])?;
    // Only real entries answer (`key-vec`, `key-nil`): 2-vectors
    // are NOT entries on the JVM.
    entry_key(&args[0])
}

fn native_val(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/val", args, &[1])?;
    entry_val(&args[0])
}

/// The S5D collection natives installed into `clojure.core` by
/// [`boot`](crate::ns::boot).
/// Sorted collections (FT6, issue #31): maps and sets carry
/// their order in the `keys`/`members` sidecar (`SortSpec` on the
/// value), so printing, seqs, equality, hashing, count, and
/// metadata need no sorted branch. Only order-sensitive
/// operations position through the helpers below.
/// Compare two keys under a sort order: natural order is
/// `jvm_compare`; a custom comparator is called as `(cmp a b)`
/// with `sort-by` number/boolean semantics.
fn compare_sort_keys(
    eval: &mut Evaluator,
    spec: &SortSpec,
    a: &Value,
    b: &Value,
) -> Result<Ordering, NativeError> {
    match spec {
        SortSpec::Natural => jvm_compare(a, b),
        SortSpec::Custom(f) => {
            let r = call_checked(eval, f, &[a.clone(), b.clone()])?;
            afunction_order(r, || call_checked(eval, f, &[b.clone(), a.clone()]))
        }
    }
}

/// Position of `k` in sorted `keys`: the found index, or the
/// insertion point that keeps the order.
enum SortPos {
    Found(usize),
    Insert(usize),
}

/// Binary search a sorted collection's `len` keys (`key_at`) for `k`,
/// in place (no snapshot of the order sidecar).
fn sort_position<'a>(
    eval: &mut Evaluator,
    spec: &SortSpec,
    len: usize,
    key_at: impl Fn(usize) -> Option<&'a Value>,
    k: &Value,
) -> Result<SortPos, NativeError> {
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let key = key_at(mid).expect("sorted position in range");
        match compare_sort_keys(eval, spec, key, k)? {
            Ordering::Less => lo = mid + 1,
            Ordering::Greater => hi = mid,
            Ordering::Equal => return Ok(SortPos::Found(mid)),
        }
    }
    Ok(SortPos::Insert(lo))
}

/// Sorted `assoc`: a compare-0 hit replaces the value but keeps
/// the original key object (probe `sorted-keyobj`); a miss
/// inserts at the sort position.
fn sorted_assoc_map(eval: &mut Evaluator, m: &Map, k: Value, v: Value) -> Result<Map, NativeError> {
    Ok(match map_position(eval, m, &k)? {
        SortPos::Found(i) => m.clone().insert(map_key_at(m, i), v),
        SortPos::Insert(i) => m.clone().insert_sorted_at(i, k, v),
    })
}

/// `k`'s position in sorted map `m`.
fn map_position(eval: &mut Evaluator, m: &Map, k: &Value) -> Result<SortPos, NativeError> {
    let spec = m
        .sort_spec()
        .cloned()
        .expect("a sorted map has a sort order");
    sort_position(eval, &spec, m.sorted_len(), |i| m.sorted_key(i), k)
}

/// The stored key at a found sorted position.
fn map_key_at(m: &Map, i: usize) -> Value {
    m.sorted_key(i).cloned().expect("found position in range")
}

/// Sorted `dissoc`: the found stored key drives the removal, so
/// compare-0-but-unequal keys still drop.
fn sorted_without_map(eval: &mut Evaluator, m: &Map, k: &Value) -> Result<Map, NativeError> {
    Ok(match map_position(eval, m, k)? {
        SortPos::Found(i) => m.clone().without(&map_key_at(m, i)),
        SortPos::Insert(_) => m.clone(),
    })
}

/// Sorted `get`: compare-0 search, then the stored entry.
fn sorted_lookup_map(
    eval: &mut Evaluator,
    m: &Map,
    k: &Value,
) -> Result<Option<Value>, NativeError> {
    Ok(match map_position(eval, m, k)? {
        SortPos::Found(i) => m.get(&map_key_at(m, i)).cloned(),
        SortPos::Insert(_) => None,
    })
}

/// Sorted `conj`: a compare-0 hit is a no-op (the member stays).
fn sorted_conj_set(eval: &mut Evaluator, s: &Set, v: Value) -> Result<Set, NativeError> {
    Ok(match set_position(eval, s, &v)? {
        SortPos::Found(_) => s.clone(),
        SortPos::Insert(i) => s.clone().insert_sorted_at(i, v),
    })
}

/// `v`'s position in sorted set `s`.
fn set_position(eval: &mut Evaluator, s: &Set, v: &Value) -> Result<SortPos, NativeError> {
    let spec = s
        .sort_spec()
        .cloned()
        .expect("a sorted set has a sort order");
    sort_position(eval, &spec, s.sorted_len(), |i| s.sorted_member(i), v)
}

/// Sorted `disj` / `contains?` through the stored member.
fn sorted_without_set(eval: &mut Evaluator, s: &Set, v: &Value) -> Result<Set, NativeError> {
    Ok(match set_position(eval, s, v)? {
        SortPos::Found(i) => s.clone().without_sorted_at(i),
        SortPos::Insert(_) => s.clone(),
    })
}

fn sorted_contains_set(eval: &mut Evaluator, s: &Set, v: &Value) -> Result<bool, NativeError> {
    Ok(matches!(set_position(eval, s, v)?, SortPos::Found(_)))
}

/// Sorted `get` on sets: the stored member, like `Set::get`.
fn sorted_lookup_set(
    eval: &mut Evaluator,
    s: &Set,
    v: &Value,
) -> Result<Option<Value>, NativeError> {
    Ok(match set_position(eval, s, v)? {
        SortPos::Found(i) => s.sorted_member(i).cloned(),
        SortPos::Insert(_) => None,
    })
}

/// The odd-argument error for map constructors (JVM `RT.map` text).
pub(crate) fn odd_keyvals_err(last: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("No value supplied for key: {}", str_of(last)),
    )
}

/// Build a sorted map from pairs under `spec` (duplicates collapse
/// like repeated `assoc`).
fn sorted_map_from(
    eval: &mut Evaluator,
    spec: SortSpec,
    kvs: &[Value],
) -> Result<Value, NativeError> {
    if kvs.len() % 2 == 1 {
        let last = kvs.last().unwrap_or(&Value::Nil);
        return Err(odd_keyvals_err(last));
    }
    let mut out = new_map().with_sort(Some(spec));
    for pair in kvs.chunks(2) {
        out = sorted_assoc_map(eval, &out, pair[0].clone(), pair[1].clone())?;
    }
    Ok(Value::Map(out))
}

/// Build a sorted set from members under `spec`.
fn sorted_set_from(
    eval: &mut Evaluator,
    spec: SortSpec,
    members: &[Value],
) -> Result<Value, NativeError> {
    let mut out = new_set().with_sort(Some(spec));
    for v in members {
        out = sorted_conj_set(eval, &out, v.clone())?;
    }
    Ok(Value::Set(out))
}

fn native_sorted_map(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    sorted_map_from(eval, SortSpec::Natural, args)
}

fn native_sorted_map_by(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/sorted-map-by", args, 1)?;
    let spec = SortSpec::Custom(Box::new(args[0].clone()));
    sorted_map_from(eval, spec, &args[1..])
}

fn native_sorted_set(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    sorted_set_from(eval, SortSpec::Natural, args)
}

fn native_sorted_set_by(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/sorted-set-by", args, 1)?;
    let spec = SortSpec::Custom(Box::new(args[0].clone()));
    sorted_set_from(eval, spec, &args[1..])
}

/// A `subseq`/`rsubseq` source: ascending entry keys with their
/// seq items (map entries for maps, members for sets) plus the
/// sort order (so callers never re-match the collection).
fn sorted_subseq_source(coll: &Value) -> Result<(Vec<Value>, Vec<Value>, SortSpec), NativeError> {
    match coll {
        Value::Map(m) if m.is_sorted() => {
            let mut keys = Vec::with_capacity(m.ordered_keys().len());
            let mut items = Vec::with_capacity(m.ordered_keys().len());
            for (k, v) in m.iter() {
                keys.push(k.clone());
                items.push(clj_value::map_entry(k.clone(), v.clone()));
            }
            let spec = m
                .sort_spec()
                .cloned()
                .expect("sorted map carries a sort order");
            Ok((keys, items, spec))
        }
        Value::Set(s) if s.is_sorted() => {
            let members = s.ordered_members();
            let spec = s
                .sort_spec()
                .cloned()
                .expect("sorted set carries a sort order");
            Ok((members.clone(), members, spec))
        }
        other => Err(cce_err(other, "clojure.lang.Sorted")),
    }
}

/// Whether `test` is one of the two named core fns (identity, like
/// the JVM's `#{> >=}` / `#{< <=}` sets — a lookalike never is).
fn is_named_test(test: &Value, a: &str, b: &str) -> bool {
    as_native(test).is_some_and(|n| n.display.as_ref() == a || n.display.as_ref() == b)
}

/// `mk-bound-fn`: `(test (compare entry-key bound) 0)`.
fn bound_include(
    eval: &mut Evaluator,
    spec: &SortSpec,
    test: &Value,
    bound: &Value,
    entry_key: &Value,
) -> Result<bool, NativeError> {
    let cmp = compare_sort_keys(eval, spec, entry_key, bound)?;
    let n = match cmp {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    };
    Ok(call_checked(eval, test, &[Value::Int(n.into()), Value::Int(0)])?.is_truthy())
}

/// Take matching items from the head (`take-while include`).
fn bound_take_while(
    eval: &mut Evaluator,
    spec: &SortSpec,
    test: &Value,
    bound: &Value,
    keys: &[Value],
    items: &[Value],
) -> Result<Vec<Value>, NativeError> {
    let mut out = Vec::new();
    for (k, item) in keys.iter().zip(items.iter()) {
        if !bound_include(eval, spec, test, bound, k)? {
            break;
        }
        out.push(item.clone());
    }
    Ok(out)
}

/// `seqFrom`: the first index with `compare(entry, key) >= 0`
/// (ascending) or `<= 0` (descending).
fn seq_from(
    eval: &mut Evaluator,
    spec: &SortSpec,
    keys: &[Value],
    key: &Value,
    ascending: bool,
) -> Result<usize, NativeError> {
    for (i, k) in keys.iter().enumerate() {
        let cmp = compare_sort_keys(eval, spec, k, key)?;
        let hit = if ascending {
            cmp != Ordering::Less
        } else {
            cmp != Ordering::Greater
        };
        if hit {
            return Ok(i);
        }
    }
    Ok(keys.len())
}

fn seq_items(items: Vec<Value>) -> Value {
    if items.is_empty() {
        empty_list_value()
    } else {
        realized_value(items, false, true, false, clj_value::SeqKind::Array)
    }
}

/// One `subseq`/`rsubseq` bound walk over ascending `keys`/`items`
/// (`core.clj` formulas verbatim).
#[allow(clippy::too_many_arguments)]
fn subseq_walk(
    eval: &mut Evaluator,
    spec: &SortSpec,
    test: &Value,
    bound: &Value,
    end_test: Option<(&Value, &Value)>,
    keys: &[Value],
    items: &[Value],
    descending: bool,
    start_high: bool,
) -> Result<Value, NativeError> {
    let order: Vec<usize> = if descending {
        (0..keys.len()).rev().collect()
    } else {
        (0..keys.len()).collect()
    };
    let ordered_keys: Vec<Value> = order.iter().map(|&i| keys[i].clone()).collect();
    let ordered_items: Vec<Value> = order.iter().map(|&i| items[i].clone()).collect();
    if start_high {
        // `seqFrom` + head-drop unless included.
        let start = seq_from(eval, spec, &ordered_keys, bound, !descending)?;
        if start >= ordered_keys.len() {
            return Ok(Value::Nil);
        }
        let mut from = start;
        if !bound_include(eval, spec, test, bound, &ordered_keys[from])? {
            from += 1;
            if from >= ordered_keys.len() {
                return Ok(Value::Nil);
            }
        }
        let rest_keys = &ordered_keys[from..];
        let rest_items = &ordered_items[from..];
        match end_test {
            // `from` always lands inside (the empty start returned
            // above), so the rest is a non-empty seq.
            None => {
                debug_assert!(!rest_items.is_empty());
                Ok(realized_value(
                    rest_items.to_vec(),
                    false,
                    true,
                    false,
                    clj_value::SeqKind::Array,
                ))
            }
            Some((end_t, end_b)) => {
                let taken = bound_take_while(eval, spec, end_t, end_b, rest_keys, rest_items)?;
                Ok(seq_items(taken))
            }
        }
    } else {
        // `take-while include` over the whole direction (3-arg
        // only — 5-arg always starts from `seqFrom` above, so
        // `end_test` is always `None` here).
        debug_assert!(end_test.is_none());
        let taken = bound_take_while(eval, spec, test, bound, &ordered_keys, &ordered_items)?;
        Ok(seq_items(taken))
    }
}

fn native_subseq(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/subseq", args, &[3, 5])?;
    let (keys, items, spec) = sorted_subseq_source(&args[0])?;
    let (test, bound) = (&args[1], &args[2]);
    let end_test = if args.len() == 5 {
        Some((&args[3], &args[4]))
    } else {
        None
    };
    // 5-arg always starts from `seqFrom`.
    let start_high = args.len() == 5 || is_named_test(test, "clojure.core/>", "clojure.core/>=");
    subseq_walk(
        eval, &spec, test, bound, end_test, &keys, &items, false, start_high,
    )
}

fn native_rsubseq(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/rsubseq", args, &[3, 5])?;
    let (keys, items, spec) = sorted_subseq_source(&args[0])?;
    let (test, bound) = (&args[1], &args[2]);
    let end_test = if args.len() == 5 {
        Some((&args[3], &args[4]))
    } else {
        None
    };
    let start_high = is_named_test(test, "clojure.core/<", "clojure.core/<=");
    if args.len() == 5 {
        // `(when-let [[e :as s] (seqFrom end-key false)]
        // (take-while start-include (if (end-include e) s (next
        // s))))` (`core.clj` `rsubseq` verbatim): the
        // descending-from-end walk, head-dropped once by the END
        // test, bounded by the START test.
        let (end_t, end_b) = end_test.expect("rsubseq 5-arg carries an end test");
        let mut desc: Vec<usize> = Vec::new();
        for (i, k) in keys.iter().enumerate().rev() {
            if compare_sort_keys(eval, &spec, k, end_b)? != Ordering::Greater {
                desc.push(i);
            }
        }
        if desc.is_empty() {
            return Ok(Value::Nil);
        }
        let mut from = 0usize;
        if !bound_include(eval, &spec, end_t, end_b, &keys[desc[from]])? {
            from += 1;
        }
        // A dropped-to-empty head takes `take-while` over nil,
        // which is `()`, not nil (probe `rsubseq-empty`).
        let rest_keys: Vec<Value> = desc[from..].iter().map(|&i| keys[i].clone()).collect();
        let rest_items: Vec<Value> = desc[from..].iter().map(|&i| items[i].clone()).collect();
        let taken = bound_take_while(eval, &spec, test, bound, &rest_keys, &rest_items)?;
        return Ok(seq_items(taken));
    }
    subseq_walk(
        eval, &spec, test, bound, None, &keys, &items, true, start_high,
    )
}

pub(crate) const COLL_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("count", native_count),
    ("empty?", native_emptyp),
    ("not-empty", native_not_empty),
    ("empty", native_empty),
    ("disj", native_disj),
    ("set", native_set),
    ("second", native_second),
    ("last", native_last),
    ("nnext", native_nnext),
    ("fnext", native_fnext),
    ("nfirst", native_nfirst),
    ("list*", native_list_star),
    ("conj", native_conj),
    ("assoc", native_assoc),
    ("dissoc", native_dissoc),
    ("get", native_get),
    ("contains?", native_contains),
    ("find", native_find),
    ("nth", native_nth),
    ("key", native_key),
    ("val", native_val),
    ("vec", native_vec),
    ("merge", native_merge),
    ("merge-with", native_merge_with),
    ("select-keys", native_select_keys),
    ("update", native_update),
    ("update-in", native_update_in),
    ("assoc-in", native_assoc_in),
    ("get-in", native_get_in),
    ("zipmap", native_zipmap),
    ("array-map", native_array_map),
    ("sorted-map", native_sorted_map),
    ("sorted-map-by", native_sorted_map_by),
    ("sorted-set", native_sorted_set),
    ("sorted-set-by", native_sorted_set_by),
    ("subseq", native_subseq),
    ("rsubseq", native_rsubseq),
    ("transient", native_transient),
    ("persistent!", native_persistent),
    ("conj!", native_conj_bang),
    ("assoc!", native_assoc_bang),
    ("dissoc!", native_dissoc_bang),
    ("disj!", native_disj_bang),
    ("pop!", native_pop_bang),
    ("peek", native_peek),
    ("pop", native_pop),
    ("subvec", native_subvec),
    ("replace", native_replace),
    ("into", native_into),
    ("distinct?", native_distinctp),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zipmap_builds_small_maps_at_once_and_grows_past_the_bulk() {
        let mut eval = crate::boot();
        let show = |eval: &mut Evaluator, src: &str| {
            let out = eval.eval_source(src).expect(src);
            clj_value::pr_to_string(out.last().expect("a form"), &clj_value::PrintOpts::pr())
        };
        assert_eq!(
            show(&mut eval, "(zipmap [:a :b :a] [1 2 3])"),
            "{:a 3, :b 2}"
        );
        assert_eq!(show(&mut eval, "(zipmap [] [])"), "{}");
        // Past the bulk: in-place inserts, a key repeated after it wins.
        assert_eq!(
            show(&mut eval, "(count (zipmap (range 100) (range)))"),
            "100"
        );
        assert_eq!(
            show(&mut eval, "((zipmap (concat (range 70) [3]) (range)) 3)"),
            "70"
        );
    }

    #[test]
    fn str_scan_global_cache_answers_memo_evictions() {
        // More big strings than memo slots: the first falls out of
        // the per-thread memo but stays in the global cache, which
        // answers (and re-memoizes) it.
        let strings: Vec<Arc<str>> = (0..LAST_SCAN_SLOTS + 2)
            .map(|i| Arc::from(format!("{i}").repeat(100)))
            .collect();
        for s in &strings {
            assert!(str_char_at(s, 10).is_some());
        }
        assert_eq!(str_char_at(&strings[0], 99), Some('0'));
        assert_eq!(str_utf16_len(&strings[0]), 100);
    }

    #[test]
    fn str_scan_memo_answers_repeats() {
        // Past the direct-scan threshold: the first read builds and
        // memoizes, repeats hit the memo, and a second string
        // replaces it (all without touching the global-cache mutex
        // path twice for the same string).
        let s: Arc<str> = Arc::from("x".repeat(200));
        assert_eq!(str_char_at(&s, 100), Some('x'));
        assert_eq!(str_char_at(&s, 150), Some('x'));
        let t: Arc<str> = Arc::from("y".repeat(200));
        assert_eq!(str_char_at(&t, 100), Some('y'));
        assert_eq!(str_char_at(&s, 120), Some('x'));
        // A dropped memo key never answers (no ABA): `t2` may reuse
        // `t`'s address, but the `Weak` is dead.
        let t2: Arc<str> = Arc::from("z".repeat(200));
        drop(t);
        assert_eq!(str_char_at(&t2, 100), Some('z'));
    }

    #[test]
    fn str_scan_small_ascii_memoizes_and_repeats() {
        let s: Arc<str> = Arc::from("hello_world");
        assert_eq!(str_char_at(&s, 0), Some('h'));
        assert_eq!(str_char_at(&s, 1), Some('e'));
        assert_eq!(str_char_at(&s, 10), Some('d'));
        assert_eq!(str_char_at(&s, 11), None);
    }

    #[test]
    fn str_scan_memo_slot_swap() {
        let s0: Arc<str> = Arc::from("0".repeat(200));
        let s1: Arc<str> = Arc::from("1".repeat(200));
        let s2: Arc<str> = Arc::from("2".repeat(200));
        assert_eq!(str_char_at(&s0, 0), Some('0'));
        assert_eq!(str_char_at(&s1, 0), Some('1'));
        assert_eq!(str_char_at(&s2, 0), Some('2'));
        assert_eq!(str_char_at(&s2, 10), Some('2'));
        assert_eq!(fast_str_char_at(&s0, 5), Some('0'));
        assert_eq!(fast_str_char_at(&s2, 5), Some('2'));
        assert_eq!(str_char_at(&s1, 5), Some('1'));
    }

    #[test]
    fn native_persistent_borrowed_fallback() {
        let mut eval = crate::boot();
        let tv = Value::Transient(clj_value::Transient::new(clj_value::TransientKind::Vector(
            clj_value::TransientVector::new(),
        )));
        let res = native_persistent(&mut eval, &[tv]).expect("persistent!");
        assert!(matches!(res, Value::Vector(_)));
    }

    #[test]
    fn str_scan_fast_indexed_slot0() {
        let s: Arc<str> = Arc::from("é".repeat(200));
        let scan = build_str_scan(&s).unwrap();
        LAST_SCAN.with(|memo| {
            let mut memo = memo.borrow_mut();
            memo.0[0] = Some((Arc::downgrade(&s), scan));
        });
        assert_eq!(fast_str_char_at(&s, 5), Some('é'));
        assert_eq!(fast_str_char_at(&s, 250), None);
        assert!(!str_ascii(&s));
        assert!(!str_ascii(&s));
        assert_eq!(fast_str_char_at(&s, 10), Some('é'));
        let long_ascii: Arc<str> = Arc::from("a".repeat(200));
        assert!(str_ascii(&long_ascii));
        assert!(str_ascii(&long_ascii));
    }

    #[test]
    fn last_walks_empty_and_nonempty_chunks() {
        let mut eval = crate::boot();
        assert_eq!(
            eval.eval_source("(last (chunk-cons (chunk (chunk-buffer 4)) '(1 2)))")
                .unwrap()
                .last(),
            Some(&Value::Int(2))
        );
        assert_eq!(
            eval.eval_source(
                "(last (chunk-cons (chunk (doto (chunk-buffer 4) (chunk-append 10) (chunk-append 20))) nil))"
            )
            .unwrap()
            .last(),
            Some(&Value::Int(20))
        );
    }

    #[test]
    fn subvec_compaction_and_assoc_orig_start() {
        let mut eval = crate::boot();
        let base_i = Vector::from_ints((0..1030_i64).collect());
        let trie_v = Value::Vector(base_i.clone().push_back(Value::Int(1030)));
        assert!(!matches!(&trie_v, Value::Vector(v) if v.is_flat()));
        // Compacts small window of a Trie backing while preserving orig_start.
        let win = subvec3(&trie_v, &Value::Int(500), &Value::Int(510)).expect("subvec trie");
        let sv = as_subvector(&win).expect("subvec");
        assert_eq!(sv.start(), 0);
        assert_eq!(sv.end(), 10);
        assert_eq!(sv.orig_start(), 500);
        assert!(sv.vec().is_flat());
        // Negative index within orig_start succeeds without changing window.
        let neg_ok = assoc_subvector(sv, &Value::Int(-1), Value::Int(99)).expect("neg assoc");
        assert_eq!(
            as_subvector(&neg_ok).expect("sv").nth(0),
            Some(Value::Int(500))
        );
        // Negative index before 0 throws IOOBE.
        assert!(assoc_subvector(sv, &Value::Int(-501), Value::Int(99)).is_err());
        // Small window of a flat vector compacts on first assoc.
        let flat_v = Value::Vector(Vector::from_ints((0..40_i64).collect()));
        let flat_win = subvec3(&flat_v, &Value::Int(5), &Value::Int(10)).expect("flat win");
        let flat_sv = as_subvector(&flat_win).expect("flat sv");
        let assoc_c = assoc_subvector(flat_sv, &Value::Int(1), Value::Int(77)).expect("compact");
        let c_sv = as_subvector(&assoc_c).expect("compacted");
        assert_eq!(c_sv.vec().len(), 5);
        assert_eq!(c_sv.nth(1), Some(Value::Int(77)));
        // Large window (> 256) updates without compacting.
        let big_win = subvec3(&trie_v, &Value::Int(10), &Value::Int(300)).expect("big win");
        let big_sv = as_subvector(&big_win).expect("big sv");
        let assoc_b = assoc_subvector(big_sv, &Value::Int(0), Value::Int(88)).expect("big assoc");
        assert_eq!(
            as_subvector(&assoc_b).expect("sv").nth(0),
            Some(Value::Int(88))
        );
        // native_vec_owned moves subvec cleanly.
        let moved = native_vec_owned(&mut eval, smallvec::smallvec![assoc_c]).expect("vec");
        assert!(as_subvector(&moved).is_some());
    }
}
