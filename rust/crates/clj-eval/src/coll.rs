// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! S5D collection natives: the `Collections` matrix rows over
//! persistent vectors/maps/sets/lists, subvectors, and transients
//! (`docs/t1174/s5d-coll-pins.md`). Shapes follow `core.clj` + the
//! `RT`/`APersistent*` sources exactly; the `into`/`merge-with`
//! transient fast paths are plain loops (same values and errors —
//! transients never escape).

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use clj_value::{
    Array, ConsValue, List, Map, PrimitiveVector, Set, SortSpec, SubVectorValue, ThrowClass,
    Transient, TransientError, TransientKind, TransientMap, TransientSet, TransientVector, Value,
    Vector, new_map, new_set,
};

use crate::core::{cce_err, cce_message, is_counted, jvm_compare, nth_index};
use crate::error::{NativeError, jvm_class_of, simple_name_of};
use crate::eval::Evaluator;
use crate::ext::as_native;
use crate::lazy::{
    StepUnit, as_range, as_realized, as_repeat, as_subvector, collect_all, empty_list_value,
    equiv_vals, first_value, is_seq_value, is_seqable, next_value, realized_value, replace_thunk,
    seq_value, step_unit,
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
    // Records read their content map (oracle `rec-pos`,
    // `rec-get`); structs likewise (S12 oracle `sm-get-basis`);
    // every other host value misses (oracle `dtor-get`).
    if let Some(i) = crate::types::maplike_inst(coll) {
        return Ok(i.content.get(key).cloned());
    }
    match coll {
        Value::Nil => Ok(None),
        Value::Map(m) => lookup_map_slot(eval, m, key),
        Value::Vector(x) => Ok(lookup_vector_slot(x, key)),
        Value::MapEntry {
            key: ek, val: ev, ..
        } => Ok(lookup_entry_slot(ek, ev, key)),
        Value::Array(a) => Ok(lookup_array_slot(a, key)),
        Value::PrimitiveVector(p) => Ok(lookup_pv_slot(p, key)),
        Value::Set(s) => lookup_set_slot(eval, s, key),
        Value::Str(s) => Ok(lookup_str_slot(s, key)),
        Value::Transient(t) => transient_get(t, key),
        other => lookup_present_fallback(other, key),
    }
}

/// A map slot (sorted maps resolve through the comparator; split
/// from [`lookup_present`] for the complexity budget).
fn lookup_map_slot(
    eval: &mut Evaluator,
    m: &Map,
    key: &Value,
) -> Result<Option<Value>, NativeError> {
    if m.is_sorted() {
        sorted_lookup_map(eval, m, key)
    } else {
        Ok(m.get(key).cloned())
    }
}

/// A set slot (sorted sets resolve through the comparator; split
/// from [`lookup_present`] for the complexity budget).
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

/// An array slot (split from [`lookup_present`] for the complexity
/// budget): any `Number` key narrows (`intValue`, lenient —
/// OOB/non-`Number`/nil miss; S15 oracles `b-get`, `f-get-*`,
/// `e-get-*`).
fn lookup_array_slot(a: &Array, key: &Value) -> Option<Value> {
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

fn lookup_vector_slot(x: &Vector, key: &Value) -> Option<Value> {
    int_key(key).and_then(|i| {
        usize::try_from(i as i32)
            .ok()
            .and_then(|idx| x.iter().nth(idx).cloned())
    })
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

fn lookup_str_slot(s: &str, key: &Value) -> Option<Value> {
    num_index(key).and_then(|i| {
        usize::try_from(i)
            .ok()
            .and_then(|idx| s.chars().nth(idx).map(Value::Char))
    })
}

fn lookup_present_fallback(coll: &Value, key: &Value) -> Result<Option<Value>, NativeError> {
    if let Some(sv) = as_subvector(coll) {
        return Ok(int_key(key).and_then(|i| {
            usize::try_from(i as i32)
                .ok()
                .and_then(|idx| sv.nth(idx).cloned())
        }));
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
            Ok(tv.nth(idx).map_err(transient_err)?.cloned())
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
            .and_then(|idx| x.iter().nth(idx).map(|v| (key.clone(), v.clone())))
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
        Ok(m.get(key).map(|v| (key.clone(), v.clone())))
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

fn map_conj_plan(eval: &mut Evaluator, x: &Value) -> Result<MapConjPlan, NativeError> {
    match x {
        Value::Nil => Ok(MapConjPlan::Nil),
        Value::Map(other) => Ok(MapConjPlan::Entries(
            other.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        )),
        Value::Vector(v) => {
            let items: Vec<Value> = v.iter().cloned().collect();
            pair_plan(&items)
        }
        // `Vec` pairs assoc like vectors (S15); non-pairs are
        // the pair error, never the entry cast.
        Value::PrimitiveVector(p) => pair_plan(&p.items()),
        Value::MapEntry { key, val, .. } => Ok(MapConjPlan::Pair((**key).clone(), (**val).clone())),
        _ => {
            if let Some(sv) = as_subvector(x) {
                if sv.len() != 2 {
                    return Err(pair_err());
                }
                let items = sv.items();
                return Ok(MapConjPlan::Pair(items[0].clone(), items[1].clone()));
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

/// Conj onto a persistent map (`APersistentMap.cons`).
fn conj_map(eval: &mut Evaluator, m: &Map, x: &Value) -> Result<Value, NativeError> {
    Ok(Value::Map(conj_map_into(eval, m, x)?))
}

/// Conj onto a persistent map, answering the map (records rewrap
/// the answer around their content).
pub(crate) fn conj_map_into(eval: &mut Evaluator, m: &Map, x: &Value) -> Result<Map, NativeError> {
    if m.is_sorted() {
        return match map_conj_plan(eval, x)? {
            MapConjPlan::Nil => Ok(m.clone()),
            MapConjPlan::Pair(k, v) => Ok(sorted_assoc_map(eval, m, k, v)?),
            MapConjPlan::Entries(entries) => {
                let mut out = m.clone();
                for (k, v) in entries {
                    out = sorted_assoc_map(eval, &out, k, v)?;
                }
                Ok(out)
            }
        };
    }
    match map_conj_plan(eval, x)? {
        MapConjPlan::Nil => Ok(m.clone()),
        MapConjPlan::Pair(k, v) => Ok(m.clone().insert(k, v)),
        MapConjPlan::Entries(entries) => {
            let mut out = m.clone();
            for (k, v) in entries {
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
        SubVectorValue::new(backing, sv.start(), sv.end() + 1).with_meta(sv.meta().cloned()),
    ))
}

/// `RT.conj` one element (`conj`/`into` share this).
fn conj_one(eval: &mut Evaluator, coll: &Value, x: Value) -> Result<Value, NativeError> {
    match coll {
        Value::Nil => Ok(Value::List(List::from_vec(vec![x]))),
        Value::List(l) => Ok(Value::List(l.clone().push_front(x))),
        Value::Vector(v) => Ok(Value::Vector(v.clone().push_back(x))),
        // Primitive vectors coerce through the type (the
        // `ArrayManager` rule — S15 oracles `f-conj-bad`,
        // `j-byte-vec`, `j-bool-num`, `j-conj-nil`).
        Value::PrimitiveVector(p) => Ok(Value::PrimitiveVector(
            p.push_back(crate::arrays::vector_coerce(eval, p.elem(), &x)?),
        )),
        // Conj onto an entry appends like its 2-vector (Oracle FX2
        // `conj`, `merge-entry`).
        Value::MapEntry { key, val, .. } => {
            let tmp = Vector::from_vec(vec![(**key).clone(), (**val).clone()]);
            Ok(Value::Vector(tmp.push_back(x)))
        }
        Value::Map(m) => conj_map(eval, m, &x),
        // Records conj through their content map and stay records
        // (oracle `h-conj-vec`, `h-conj-map-type`); structs
        // likewise (S12); anything else host falls to the
        // collection cast (oracle `i-conj`).
        Value::HostType(_) => conj_maplike(eval, coll, &x),
        Value::Set(s) if s.is_sorted() => Ok(Value::Set(sorted_conj_set(eval, s, x)?)),
        Value::Set(s) => Ok(Value::Set(s.clone().insert(x))),
        _ => conj_one_fallback(coll, x),
    }
}

/// Conj onto anything else (split from [`conj_one`] for the
/// complexity budget): subvectors stay subvectors, seqs cons,
/// anything else takes the collection cast.
fn conj_one_fallback(coll: &Value, x: Value) -> Result<Value, NativeError> {
    if let Some(sv) = as_subvector(coll) {
        return Ok(subvector_conj(sv, x));
    }
    if is_seq_value(coll) {
        return Ok(Value::Extension(Arc::new(ConsValue::new(x, coll.clone()))));
    }
    Err(cce_err(coll, "clojure.lang.IPersistentCollection"))
}

/// Conj onto a map-like host value (split from [`conj_one`] for
/// the complexity budget): through the content map, rewrapped so
/// the answer stays a record (or a struct, S12); anything else
/// host takes the collection cast.
fn conj_maplike(eval: &mut Evaluator, coll: &Value, x: &Value) -> Result<Value, NativeError> {
    match crate::types::maplike_inst(coll) {
        Some(i) => Ok(crate::types::rewrap_maplike(
            i,
            conj_map_into(eval, &i.content, x)?,
        )),
        None => Err(cce_err(coll, "clojure.lang.IPersistentCollection")),
    }
}

/// `assoc` on a vector (integer keys only; at-count appends;
/// anything else is the empty `IndexOutOfBoundsException`).
fn assoc_vector(x: &Vector, i: i64, v: Value) -> Result<Value, NativeError> {
    if let Ok(n) = usize::try_from(i as i32) {
        if n < x.len() {
            let out = x.set_nth(n, v).expect("index is below the length");
            return Ok(Value::Vector(out));
        } else if n == x.len() {
            return Ok(Value::Vector(x.clone().push_back(v)));
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
    p: &PrimitiveVector,
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

/// `assoc` on a subvector (stays a subvector; writes at the absolute
/// index — even before the window, like the JVM).
fn assoc_subvector(sv: &SubVectorValue, k: &Value, v: Value) -> Result<Value, NativeError> {
    let Some(i) = int_key(k) else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "Key must be integer".to_owned(),
        ));
    };
    let abs = sv.start() as i64 + (i as i32) as i64;
    let end = sv.end() as i64;
    if abs < 0 || abs > end {
        return Err(NativeError::new(
            ThrowClass::IndexOutOfBoundsException,
            String::new(),
        ));
    }
    if abs == end {
        return Ok(subvector_conj(sv, v));
    }
    let n = abs as usize;
    // `abs < end <= backing len` on every live subvector, so this
    // `set_nth` cannot fail.
    let backing = sv
        .vec()
        .set_nth(n, v)
        .expect("window index is below the backing length");
    Ok(Value::Extension(Arc::new(
        SubVectorValue::new(backing, sv.start(), sv.end()).with_meta(sv.meta().cloned()),
    )))
}

/// `RT.assoc` one triple (`assoc`/`update`/`merge-with` share this;
/// sorted maps position by comparator, FT6).
fn assoc3(eval: &mut Evaluator, coll: &Value, k: &Value, v: Value) -> Result<Value, NativeError> {
    match coll {
        Value::Nil => Ok(Value::Map(new_map().insert(k.clone(), v))),
        Value::Map(m) if m.is_sorted() => Ok(Value::Map(sorted_assoc_map(eval, m, k.clone(), v)?)),
        Value::Map(m) => Ok(Value::Map(m.clone().insert(k.clone(), v))),
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
            let tmp = Vector::from_vec(vec![(**key).clone(), (**val).clone()]);
            assoc_vector(&tmp, i, v)
        }
        // Records assoc through their content map and stay records
        // (oracle `rec-assoc-type`, `h-assoc-existing`); structs
        // always stay structs (S12 oracles `sm-assoc-class`,
        // `sm-assoc-extra-class`); anything else host is not
        // associative (oracle `i-assoc`).
        Value::HostType(_) => match crate::types::maplike_inst(coll) {
            Some(i) => Ok(crate::types::rewrap_maplike(
                i,
                i.content.clone().insert(k.clone(), v),
            )),
            None => Err(cce_err(coll, "clojure.lang.Associative")),
        },
        _ => {
            if let Some(sv) = as_subvector(coll) {
                return assoc_subvector(sv, k, v);
            }
            Err(cce_err(coll, "clojure.lang.Associative"))
        }
    }
}

/// Walk a seq counting items (hangs on infinite seqs, like the JVM).
fn walk_count(eval: &mut Evaluator, v: &Value) -> Result<usize, NativeError> {
    let mut n = 0usize;
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
    if let Some(sv) = as_subvector(v) {
        return Ok(sv.len() as i64);
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
        return Ok(s.items().len() as i64);
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
        Value::Str(s) => Ok(s.encode_utf16().count() as i64),
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
fn into_inner(eval: &mut Evaluator, to: &Value, from: &Value) -> Result<Value, NativeError> {
    let mut acc = to.clone();
    let mut cur = seq_value(eval, from).map_err(signal_to_native)?;
    loop {
        if matches!(cur, Value::Nil) {
            return Ok(acc);
        }
        let head = first_value(eval, &cur).map_err(signal_to_native)?;
        acc = conj_one(eval, &acc, head)?;
        cur = next_value(eval, &cur).map_err(signal_to_native)?;
    }
}

/// Two-way `merge` (`([x y] (if (and x y) (into x y) (or x y)))`).
fn merge2(eval: &mut Evaluator, x: &Value, y: &Value) -> Result<Value, NativeError> {
    if x.is_truthy() && y.is_truthy() {
        into_inner(eval, x, y)
    } else if x.is_truthy() {
        Ok(x.clone())
    } else {
        Ok(y.clone())
    }
}

/// One `merge-with` key/value pair (`merge-kv`: missing keys assoc,
/// present keys combine through `f`).
fn merge_kv(
    eval: &mut Evaluator,
    m: &Value,
    k: Value,
    v: Value,
    f: &Value,
) -> Result<Value, NativeError> {
    match lookup_present(eval, m, &k)? {
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
    x: &Value,
    ym: &Map,
) -> Result<Value, NativeError> {
    let mut m = x.clone();
    for (k, v) in ym.iter() {
        m = merge_kv(eval, &m, k.clone(), v.clone(), f)?;
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
    x: &Value,
    pairs: Vec<(Value, Value)>,
) -> Result<Value, NativeError> {
    let mut m = x.clone();
    for (k, v) in pairs {
        m = merge_kv(eval, &m, k, v, f)?;
    }
    Ok(m)
}

/// `merge-with` over a sequential right-hand side (seqs and casts).
fn merge_with3_seq_tail(
    eval: &mut Evaluator,
    f: &Value,
    x: &Value,
    y: &Value,
) -> Result<Value, NativeError> {
    // A sequential right-hand side contributes one pair per entry
    // (Oracle FX2 `conj-seq-entries`, `merge-with-entry` CCEs on
    // the `:a` head — entries are real `MapEntry` values now).
    let mut m = x.clone();
    let mut cur = seq_value(eval, y).map_err(signal_to_native)?;
    loop {
        if matches!(cur, Value::Nil) {
            return Ok(m);
        }
        let head = first_value(eval, &cur).map_err(signal_to_native)?;
        let Value::MapEntry { key, val, .. } = head else {
            return Err(cce_err(&head, "java.util.Map$Entry"));
        };
        m = merge_kv(eval, &m, (*key).clone(), (*val).clone(), f)?;
        cur = next_value(eval, &cur).map_err(signal_to_native)?;
    }
}

fn merge_with3(
    eval: &mut Evaluator,
    f: &Value,
    x: &Value,
    y: &Value,
) -> Result<Value, NativeError> {
    if let Some(v) = merge_with3_early(x, y) {
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

/// Split a keyseq into its head key and tail (`update-in` /
/// `assoc-in` destructure `[k & ks]` by seqing once).
fn seq_first_next(eval: &mut Evaluator, ks: &Value) -> Result<(Value, Value), NativeError> {
    let s = seq_value(eval, ks).map_err(signal_to_native)?;
    if matches!(s, Value::Nil) {
        return Ok((Value::Nil, Value::Nil));
    }
    let k = first_value(eval, &s).map_err(signal_to_native)?;
    let rest = next_value(eval, &s).map_err(signal_to_native)?;
    Ok((k, rest))
}

/// The `update-in` recursion (`ks` already seqed).
fn update_in_up(
    eval: &mut Evaluator,
    m: &Value,
    k: &Value,
    ks: &Value,
    f: &Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    if !matches!(ks, Value::Nil) {
        let inner = lookup_present(eval, m, k)?.unwrap_or(Value::Nil);
        let (k2, ks2) = seq_first_next(eval, ks)?;
        let updated = update_in_up(eval, &inner, &k2, &ks2, f, args)?;
        assoc3(eval, m, k, updated)
    } else {
        let old = lookup_present(eval, m, k)?.unwrap_or(Value::Nil);
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(old);
        call_args.extend(args.iter().cloned());
        let new = eval.call(f, &call_args).map_err(signal_to_native)?;
        assoc3(eval, m, k, new)
    }
}

/// The `assoc-in` recursion (`ks` already seqed).
fn assoc_in_up(
    eval: &mut Evaluator,
    m: &Value,
    k: &Value,
    ks: &Value,
    v: &Value,
) -> Result<Value, NativeError> {
    if !matches!(ks, Value::Nil) {
        let inner = lookup_present(eval, m, k)?.unwrap_or(Value::Nil);
        let (k2, ks2) = seq_first_next(eval, ks)?;
        let updated = assoc_in_up(eval, &inner, &k2, &ks2, v)?;
        assoc3(eval, m, k, updated)
    } else {
        assoc3(eval, m, k, v.clone())
    }
}

fn native_count(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/count", args, &[1])?;
    Ok(Value::Int(count_of(eval, &args[0])?))
}

fn native_emptyp(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/empty?", args, &[1])?;
    if is_counted(&args[0]) {
        Ok(Value::Bool(count_of(eval, &args[0])? == 0))
    } else {
        let s = seq_value(eval, &args[0]).map_err(signal_to_native)?;
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
        v if as_subvector(v).is_some() => {
            let meta = as_subvector(v).and_then(|sv| sv.meta().cloned());
            Value::Vector(Vector::new_sync().with_meta(meta))
        }
        v if is_seq_value(v) => Value::List(List::new_sync()),
        _ => Value::Nil,
    })
}

fn native_disj(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/disj", 0));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::Set(s) if s.is_sorted() => {
            let mut out = s.clone();
            for k in &args[1..] {
                out = sorted_without_set(eval, &out, k)?;
            }
            Ok(Value::Set(out))
        }
        Value::Set(s) => {
            let mut out = s.clone();
            for k in &args[1..] {
                out = out.without(k);
            }
            Ok(Value::Set(out))
        }
        v => Err(cce_err(v, "clojure.lang.IPersistentSet")),
    }
}

fn native_set(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/set", args, &[1])?;
    let items = collect_all(eval, &args[0]).map_err(signal_to_native)?;
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
    check_arity("clojure.core/last", args, &[1])?;
    let mut cur = args[0].clone();
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
    if args.is_empty() {
        return Ok(Value::Vector(Vector::from_vec(Vec::new())));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    let mut acc = args[0].clone();
    for arg in &args[1..] {
        acc = conj_one(eval, &acc, arg.clone())?;
    }
    Ok(acc)
}

fn native_assoc(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/assoc", args, 3)?;
    let mut acc = assoc3(eval, &args[0], &args[1], args[2].clone())?;
    for pair in args[3..].chunks(2) {
        if pair.len() != 2 {
            return Err(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "assoc expects even number of arguments after map/vector, found odd number"
                    .to_owned(),
            ));
        }
        acc = assoc3(eval, &acc, &pair[0], pair[1].clone())?;
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

fn native_dissoc(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/dissoc", args, 1)?;
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    if matches!(args[0], Value::Nil) {
        return Ok(Value::Nil);
    }
    if let Some(i) = crate::types::struct_inst(&args[0]) {
        return dissoc_struct(i, &args[1..]);
    }
    // Records dissoc through their content map: dropping a declared
    // field loses recordness (a plain map, oracle `rec-dissoc-type`);
    // anything else stays a record (oracle `h-dissoc-missing-type`).
    // Metadata travels in both cases (ad-hoc `dissoc-meta`).
    if let Some(i) = crate::types::record_inst(&args[0]) {
        let drops_field = args[1..].iter().any(|k| {
            i.def
                .fields
                .iter()
                .any(|f| crate::types::is_field_key(f, k))
        });
        let mut out = i.content.clone();
        for k in &args[1..] {
            out = out.without(k);
        }
        if drops_field {
            return Ok(Value::Map(out.with_meta(i.meta.clone())));
        }
        return Ok(crate::types::rewrap_record(i, out));
    }
    let Value::Map(m) = &args[0] else {
        return Err(cce_err(&args[0], "clojure.lang.IPersistentMap"));
    };
    if m.is_sorted() {
        let mut out = m.clone();
        for k in &args[1..] {
            out = sorted_without_map(eval, &out, k)?;
        }
        return Ok(Value::Map(out));
    }
    let mut out = m.clone();
    for k in &args[1..] {
        out = out.without(k);
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
        Ok(Value::Bool(m.contains_key(key)))
    }
}

/// `contains?` on sets (sorted sets resolve through the comparator).
fn contains_set(eval: &mut Evaluator, s: &Set, key: &Value) -> Result<Value, NativeError> {
    if s.is_sorted() {
        Ok(Value::Bool(sorted_contains_set(eval, s, key)?))
    } else {
        Ok(Value::Bool(s.contains(key)))
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
            Some(idx) => Ok(tv.nth(idx).map_err(transient_err)?.cloned()),
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
    let found = match coll {
        Value::Vector(x) => usize::try_from(n)
            .ok()
            .and_then(|idx| x.iter().nth(idx).cloned()),
        Value::PrimitiveVector(p) => usize::try_from(n).ok().and_then(|idx| p.get(idx).cloned()),
        Value::MapEntry { key, val, .. } => usize::try_from(n).ok().and_then(|idx| match idx {
            0 => Some((**key).clone()),
            1 => Some((**val).clone()),
            _ => None,
        }),
        Value::Transient(t) => nth_indexed_transient(t, n)?,
        _ => as_subvector(coll)
            .and_then(|sv| usize::try_from(n).ok().and_then(|idx| sv.nth(idx).cloned())),
    };
    nth_indexed_found(found, not_found)
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

/// `nth` over an array (split from [`native_nth`] for the
/// complexity budget): `not-found` on OOB, else the empty
/// `AIOOBE` (S15 oracles `a-nth`, `a-nth-nf`, `a-nth-oob`,
/// `e-nth-float`, `n-arr-str`, `n-arr-nil`).
fn nth_array(a: &Array, n: i32, not_found: Option<&Value>) -> Result<Value, NativeError> {
    let found = usize::try_from(n).ok().and_then(|idx| {
        if idx < a.len() {
            Some(crate::arrays::array_read(a, idx))
        } else {
            None
        }
    });
    match (found, not_found) {
        (Some(v), _) => Ok(v),
        (None, Some(nf)) => Ok(nf.clone()),
        (None, None) => Err(NativeError::new(
            ThrowClass::ArrayIndexOutOfBoundsException,
            String::new(),
        )),
    }
}

/// `nth` over a string (chars; `StringIndexOutOfBounds` without `not-found`).
fn nth_str(s: &str, n: i32, not_found: Option<&Value>) -> Result<Value, NativeError> {
    let found = usize::try_from(n)
        .ok()
        .and_then(|idx| s.chars().nth(idx).map(Value::Char));
    match found {
        Some(c) => Ok(c),
        None => {
            if let Some(nf) = not_found {
                return Ok(nf.clone());
            }
            Err(NativeError::new(
                ThrowClass::StringIndexOutOfBoundsException,
                format!(
                    "Index {} out of bounds for length {}",
                    n,
                    s.encode_utf16().count()
                ),
            ))
        }
    }
}

pub(crate) fn native_nth(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/nth", args, &[2, 3])?;
    // The index coerces first (before the collection dispatches).
    let n = nth_index(&args[1])?;
    let not_found = args.get(2);
    let coll = &args[0];
    if matches!(coll, Value::Nil) {
        return Ok(not_found.cloned().unwrap_or(Value::Nil));
    }
    if not_found.is_some() && n < 0 {
        return Ok(not_found.cloned().unwrap_or(Value::Nil));
    }
    // Chunks read their raw slots (nil holes below capacity, S12
    // oracles `ch-nth-oob`, `ch-nth-cap`).
    if let Some(c) = crate::collmisc::as_chunk(coll) {
        return crate::collmisc::chunk_nth(c, n, not_found);
    }
    if matches!(
        coll,
        Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_)
    ) || as_subvector(coll).is_some()
        || matches!(coll, Value::Transient(_))
    {
        return nth_indexed(coll, n, not_found);
    }
    if let Value::Array(a) = coll {
        return nth_array(a, n, not_found);
    }
    if let Value::Str(s) = coll {
        return nth_str(s, n, not_found);
    }
    if matches!(coll, Value::List(_)) || is_seq_value(coll) {
        return nth_walk(eval, coll, n, not_found);
    }
    Err(nth_unsupported(coll))
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
    check_arity("clojure.core/vec", args, &[1])?;
    if matches!(args[0], Value::Vector(_)) || as_subvector(&args[0]).is_some() {
        return Ok(args[0].clone().with_meta(None));
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
    let items = collect_all(eval, &args[0]).map_err(signal_to_native)?;
    Ok(Value::Vector(Vector::from_vec(items)))
}

fn native_merge(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Ok(Value::Nil);
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    let mut acc = merge2(eval, &args[0], &args[1])?;
    for arg in &args[2..] {
        acc = merge2(eval, &acc, arg)?;
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
    let mut acc = merge_with3(eval, &args[0], &args[1], &args[2])?;
    for arg in &args[3..] {
        acc = merge_with3(eval, &args[0], &acc, arg)?;
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
    check_min_arity("clojure.core/update", args, 3)?;
    let old = lookup_present(eval, &args[0], &args[1])?.unwrap_or(Value::Nil);
    let mut call_args = Vec::with_capacity(args.len() - 2);
    call_args.push(old);
    call_args.extend(args[3..].iter().cloned());
    let new = eval.call(&args[2], &call_args).map_err(signal_to_native)?;
    assoc3(eval, &args[0], &args[1], new)
}

fn native_update_in(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_min_arity("clojure.core/update-in", args, 3)?;
    let (k, ks) = seq_first_next(eval, &args[1])?;
    update_in_up(eval, &args[0], &k, &ks, &args[2], &args[3..])
}

fn native_assoc_in(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/assoc-in", args, &[3])?;
    let (k, ks) = seq_first_next(eval, &args[1])?;
    assoc_in_up(eval, &args[0], &k, &ks, &args[2])
}

fn native_get_in(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/get-in", args, &[2, 3])?;
    let mut cur = args[0].clone();
    let mut ks = seq_value(eval, &args[1]).map_err(signal_to_native)?;
    loop {
        if matches!(ks, Value::Nil) {
            return Ok(cur);
        }
        let k = first_value(eval, &ks).map_err(signal_to_native)?;
        match lookup_present(eval, &cur, &k)? {
            Some(v) => cur = v,
            None => {
                return Ok(args.get(2).cloned().unwrap_or(Value::Nil));
            }
        }
        ks = next_value(eval, &ks).map_err(signal_to_native)?;
    }
}

fn native_zipmap(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/zipmap", args, &[2])?;
    let mut ks = seq_value(eval, &args[0]).map_err(signal_to_native)?;
    let mut vs = seq_value(eval, &args[1]).map_err(signal_to_native)?;
    let mut out = new_map();
    loop {
        if matches!(ks, Value::Nil) || matches!(vs, Value::Nil) {
            return Ok(Value::Map(out));
        }
        let k = first_value(eval, &ks).map_err(signal_to_native)?;
        let v = first_value(eval, &vs).map_err(signal_to_native)?;
        out = out.insert(k, v);
        ks = next_value(eval, &ks).map_err(signal_to_native)?;
        vs = next_value(eval, &vs).map_err(signal_to_native)?;
    }
}

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

fn native_transient(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/transient", args, &[1])?;
    match &args[0] {
        Value::Vector(x) => Ok(Value::Transient(Transient::with_meta(
            TransientKind::Vector(TransientVector::from_vector(x)),
            x.meta().cloned(),
        ))),
        // Sorted maps/sets are not editable on the JVM either
        // (probe `trans-sorted`).
        Value::Map(m) if m.is_sorted() => {
            Err(cce_err(&args[0], "clojure.lang.IEditableCollection"))
        }
        Value::Map(m) => Ok(Value::Transient(Transient::with_meta(
            TransientKind::Map(TransientMap::from_map(m)),
            m.meta().cloned(),
        ))),
        Value::Set(s) if s.is_sorted() => {
            Err(cce_err(&args[0], "clojure.lang.IEditableCollection"))
        }
        Value::Set(s) => Ok(Value::Transient(Transient::with_meta(
            TransientKind::Set(TransientSet::from_set(s)),
            s.meta().cloned(),
        ))),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.IEditableCollection.asTransient()\" because \"coll\" is null"
                .to_owned(),
        )),
        other => Err(cce_err(other, "clojure.lang.IEditableCollection")),
    }
}

fn native_persistent(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/persistent!", args, &[1])?;
    match &args[0] {
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
        other => Err(cce_err(other, "clojure.lang.ITransientCollection")),
    }
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
    check_arity("clojure.core/conj!", args, &[0, 1, 2])?;
    if args.is_empty() {
        return Ok(Value::Transient(Transient::new(TransientKind::Vector(
            TransientVector::new(),
        ))));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    match &args[0] {
        Value::Transient(t) => conj_bang_transient(eval, t, &args[1]),
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
    // Read the class before locking (`jvm_class_of` locks
    // transients itself — locking twice deadlocks).
    let from = jvm_class_of(&args[0]);
    {
        let mut guard = t.lock();
        match &mut *guard {
            TransientKind::Vector(tv) => {
                assoc_bang_one(tv, &args[1], &args[2])?;
                for chunk in args[3..].chunks(2) {
                    let v = chunk.get(1).unwrap_or(&Value::Nil);
                    assoc_bang_one(tv, &chunk[0], v)?;
                }
            }
            TransientKind::Map(tm) => {
                tm.assoc(args[1].clone(), args[2].clone())
                    .map_err(transient_err)?;
                for chunk in args[3..].chunks(2) {
                    let v = chunk.get(1).cloned().unwrap_or(Value::Nil);
                    tm.assoc(chunk[0].clone(), v).map_err(transient_err)?;
                }
            }
            TransientKind::Set(_) => {
                return Err(NativeError::new(
                    ThrowClass::ClassCastException,
                    cce_message(&from, "clojure.lang.ITransientAssociative"),
                ));
            }
        }
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

fn native_peek(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/peek", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::List(l) => Ok(l.first().cloned().unwrap_or(Value::Nil)),
        Value::Vector(x) => Ok(x.iter().last().cloned().unwrap_or(Value::Nil)),
        Value::PrimitiveVector(p) => Ok(p.iter().last().cloned().unwrap_or(Value::Nil)),
        Value::MapEntry { val, .. } => Ok((**val).clone()),
        _ => {
            if let Some(sv) = as_subvector(&args[0]) {
                return Ok(sv.iter().last().cloned().unwrap_or(Value::Nil));
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
fn pop_vector(x: &Vector) -> Result<Value, NativeError> {
    if x.is_empty() {
        return Err(NativeError::new(
            ThrowClass::IllegalStateException,
            "Can't pop empty vector".to_owned(),
        ));
    }
    Ok(Value::Vector(
        x.drop_last().expect("non-empty vector has a tail"),
    ))
}

/// `pop` over a primitive vector (stays typed; empty pops throw
/// like vectors).
fn pop_pv(p: &PrimitiveVector) -> Result<Value, NativeError> {
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
    if let Some(sv) = as_subvector(arg) {
        if sv.len() == 1 {
            return Ok(Value::Vector(Vector::from_vec(Vec::new())));
        }
        // Empty subvectors never exist (`subvec` returns a plain
        // vector when `start == end`), so `end >= 1` here.
        let end = sv.end().checked_sub(1).expect("subvector is never empty");
        return Ok(Value::Extension(Arc::new(
            SubVectorValue::new(sv.vec().clone(), sv.start(), end).with_meta(sv.meta().cloned()),
        )));
    }
    Err(cce_err(arg, "clojure.lang.IPersistentStack"))
}

fn native_pop(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/pop", args, &[1])?;
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::List(l) => pop_list(l),
        Value::Vector(x) => pop_vector(x),
        Value::PrimitiveVector(p) => pop_pv(p),
        // Popping an entry answers its 1-vector (Oracle FX2
        // `peek-pop`, `pop-class`).
        Value::MapEntry { key, .. } => Ok(Value::Vector(Vector::from_vec(vec![(**key).clone()]))),
        _ => pop_fallback(&args[0]),
    }
}

/// Three-way `subvec` over an already-defaulted end.
fn subvec3(v: &Value, start_v: &Value, end_v: &Value) -> Result<Value, NativeError> {
    let (backing, base, len) = match v {
        Value::Vector(x) => (x.clone(), 0, x.len()),
        // `subvec` of a primitive vector is a plain subvector
        // (S15 oracle `j-subvec-cls`).
        Value::PrimitiveVector(p) => (Vector::from_vec(p.items()), 0, p.len()),
        Value::MapEntry { key, val, .. } => (
            Vector::from_vec(vec![(**key).clone(), (**val).clone()]),
            0,
            2,
        ),
        _ => match as_subvector(v) {
            Some(sv) => (sv.vec().clone(), sv.start(), sv.len()),
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
    Ok(Value::Extension(Arc::new(SubVectorValue::new(
        backing,
        base + su,
        base + eu,
    ))))
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
            acc = assoc3(eval, &acc, &k, v)?;
        }
    }
    Ok(acc)
}

fn native_into(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/into", args, &[0, 1, 2, 3])?;
    if args.is_empty() {
        return Ok(Value::Vector(Vector::from_vec(Vec::new())));
    }
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    if args.len() == 3 {
        // `(into to xform from)` (T1200 D1, oracle `into-*`).
        return crate::xform::into_xform(eval, args[0].clone(), &args[1], &args[2]);
    }
    into_inner(eval, &args[0], &args[1])
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
fn native_key(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/key", args, &[1])?;
    // Only real entries answer (`key-vec`, `key-nil`): 2-vectors
    // are NOT entries on the JVM.
    match &args[0] {
        Value::MapEntry { key, .. } => Ok((**key).clone()),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.util.Map$Entry.getKey()\" because \"e\" is null".to_owned(),
        )),
        other => Err(cce_err(other, "java.util.Map$Entry")),
    }
}

fn native_val(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/val", args, &[1])?;
    match &args[0] {
        Value::MapEntry { val, .. } => Ok((**val).clone()),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.util.Map$Entry.getValue()\" because \"e\" is null".to_owned(),
        )),
        other => Err(cce_err(other, "java.util.Map$Entry")),
    }
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

fn sort_position(
    eval: &mut Evaluator,
    spec: &SortSpec,
    keys: &[Value],
    k: &Value,
) -> Result<SortPos, NativeError> {
    let mut lo = 0usize;
    let mut hi = keys.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match compare_sort_keys(eval, spec, &keys[mid], k)? {
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
    let spec = m
        .sort_spec()
        .cloned()
        .expect("sorted assoc needs a sort order");
    let mut entries: Vec<(Value, Value)> = m.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
    match sort_position(eval, &spec, m.ordered_keys(), &k)? {
        SortPos::Found(i) => entries[i].1 = v,
        SortPos::Insert(i) => entries.insert(i, (k, v)),
    }
    Ok(m.clone().with_ordered_entries(entries))
}

/// Sorted `dissoc`: the found stored key drives the removal, so
/// compare-0-but-unequal keys still drop.
fn sorted_without_map(eval: &mut Evaluator, m: &Map, k: &Value) -> Result<Map, NativeError> {
    let spec = m
        .sort_spec()
        .cloned()
        .expect("sorted dissoc needs a sort order");
    match sort_position(eval, &spec, m.ordered_keys(), k)? {
        SortPos::Found(i) => {
            let found = m.ordered_keys()[i].clone();
            Ok(m.clone().without(&found))
        }
        SortPos::Insert(_) => Ok(m.clone()),
    }
}

/// Sorted `get`: compare-0 search, then the stored entry.
fn sorted_lookup_map(
    eval: &mut Evaluator,
    m: &Map,
    k: &Value,
) -> Result<Option<Value>, NativeError> {
    let spec = m
        .sort_spec()
        .cloned()
        .expect("sorted get needs a sort order");
    match sort_position(eval, &spec, m.ordered_keys(), k)? {
        SortPos::Found(i) => Ok(m.get(&m.ordered_keys()[i]).cloned()),
        SortPos::Insert(_) => Ok(None),
    }
}

/// Sorted `conj`: a compare-0 hit is a no-op (the member stays).
fn sorted_conj_set(eval: &mut Evaluator, s: &Set, v: Value) -> Result<Set, NativeError> {
    let spec = s
        .sort_spec()
        .cloned()
        .expect("sorted conj needs a sort order");
    match sort_position(eval, &spec, s.ordered_members(), &v)? {
        SortPos::Found(_) => Ok(s.clone()),
        SortPos::Insert(i) => {
            let mut members = s.ordered_members().to_vec();
            members.insert(i, v);
            Ok(s.clone().with_ordered_members(members))
        }
    }
}

/// Sorted `disj` / `contains?` through the stored member.
fn sorted_without_set(eval: &mut Evaluator, s: &Set, v: &Value) -> Result<Set, NativeError> {
    let spec = s
        .sort_spec()
        .cloned()
        .expect("sorted disj needs a sort order");
    match sort_position(eval, &spec, s.ordered_members(), v)? {
        SortPos::Found(i) => {
            let found = s.ordered_members()[i].clone();
            Ok(s.clone().without(&found))
        }
        SortPos::Insert(_) => Ok(s.clone()),
    }
}

fn sorted_contains_set(eval: &mut Evaluator, s: &Set, v: &Value) -> Result<bool, NativeError> {
    let spec = s
        .sort_spec()
        .cloned()
        .expect("sorted contains needs a sort order");
    match sort_position(eval, &spec, s.ordered_members(), v)? {
        SortPos::Found(_) => Ok(true),
        SortPos::Insert(_) => Ok(false),
    }
}

/// Sorted `get` on sets: the stored member, like `Set::get`.
fn sorted_lookup_set(
    eval: &mut Evaluator,
    s: &Set,
    v: &Value,
) -> Result<Option<Value>, NativeError> {
    let spec = s
        .sort_spec()
        .cloned()
        .expect("sorted set get needs a sort order");
    match sort_position(eval, &spec, s.ordered_members(), v)? {
        SortPos::Found(i) => Ok(Some(s.ordered_members()[i].clone())),
        SortPos::Insert(_) => Ok(None),
    }
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
            let members = s.ordered_members().to_vec();
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
