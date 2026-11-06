// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Transients (S3): bulk-build buffers mirroring
//! `TransientVector`/`TransientArrayMap`/`TransientHashSet`
//! (`docs/spec/t1174/probes/probe19.clj`).
//!
//! Every op validates the edit flag, exactly like the JVM: any use
//! after [`TransientVector::persistent`] (mutating ops, reads, or a
//! second `persistent`) raises
//! [`TransientError::UseAfterPersistent`]. Methods take `&mut self`,
//! so the borrow checker enforces the JVM's single-threaded-use
//! rule statically. `persistent` drops metadata (a fresh transient
//! carries none).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{FxBuildHasher, Map, Set, Value, Vector, new_map, new_set};

/// A transient failure: exact JVM message + class (probe19). The
/// evaluator (S4) raises these with the same text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransientError {
    /// `Transient used after persistent! call` (any op after
    /// `persistent`, including reads and a second call).
    UseAfterPersistent,
    /// `Can't pop empty vector`.
    EmptyPop,
    /// `assoc!` past the end (empty message, like the JDK).
    IndexOutOfBounds,
}

impl TransientError {
    /// The JVM message text.
    #[must_use]
    pub fn message(&self) -> &'static str {
        match self {
            TransientError::UseAfterPersistent => "Transient used after persistent! call",
            TransientError::EmptyPop => "Can't pop empty vector",
            TransientError::IndexOutOfBounds => "",
        }
    }

    /// The JVM exception class.
    #[must_use]
    pub fn jvm_class(&self) -> &'static str {
        match self {
            TransientError::UseAfterPersistent => "java.lang.IllegalAccessError",
            TransientError::EmptyPop => "java.lang.IllegalStateException",
            TransientError::IndexOutOfBounds => "java.lang.IndexOutOfBoundsException",
        }
    }
}

impl fmt::Display for TransientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for TransientError {}

fn ensure(valid: bool) -> Result<(), TransientError> {
    if valid {
        Ok(())
    } else {
        Err(TransientError::UseAfterPersistent)
    }
}

/// Transient vector storage: all-`Int` buffers ride packed (8
/// bytes per element), all-char buffers too (4 bytes); an empty
/// buffer takes the type of its first item, and the first foreign
/// value converts to generic (which stays generic).
#[derive(Clone, Debug)]
enum TransientItems {
    Ints(Vec<i64>),
    Chars(Vec<char>),
    Generic(Vec<Value>),
}

impl TransientItems {
    /// Put `v` at `idx` (`idx == len` appends; callers bound `idx`):
    /// an empty packed buffer retypes to `v`'s type, a foreign value
    /// converts a packed buffer to generic.
    fn put(&mut self, idx: usize, v: Value) {
        let converted = match self {
            TransientItems::Generic(x) => return put_at(x, idx, v),
            TransientItems::Ints(x) => match v {
                Value::Int(n) => return put_at(x, idx, n),
                Value::Char(c) if x.is_empty() => TransientItems::Chars(vec![c]),
                v => TransientItems::Generic(boxed_with(x, Value::Int, idx, v)),
            },
            TransientItems::Chars(x) => match v {
                Value::Char(c) => return put_at(x, idx, c),
                Value::Int(n) if x.is_empty() => TransientItems::Ints(vec![n]),
                v => TransientItems::Generic(boxed_with(x, Value::Char, idx, v)),
            },
        };
        *self = converted;
    }

    fn len(&self) -> usize {
        match self {
            TransientItems::Ints(x) => x.len(),
            TransientItems::Chars(x) => x.len(),
            TransientItems::Generic(x) => x.len(),
        }
    }
}

/// A packed buffer boxed to values, with `v` put at `idx`.
fn boxed_with<T: Copy>(x: &[T], box_item: fn(T) -> Value, idx: usize, v: Value) -> Vec<Value> {
    let mut g: Vec<Value> = x.iter().map(|n| box_item(*n)).collect();
    put_at(&mut g, idx, v);
    g
}

/// Set `x[idx]`, or append at `idx == len`.
fn put_at<T>(x: &mut Vec<T>, idx: usize, v: T) {
    if idx == x.len() {
        x.push(v);
    } else {
        x[idx] = v;
    }
}

/// A transient vector: `conj!`/`pop!`/`assoc!` buffer.
#[derive(Clone, Debug)]
pub struct TransientVector {
    items: TransientItems,
    valid: bool,
}

impl TransientVector {
    /// An empty transient vector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: TransientItems::Ints(Vec::new()),
            valid: true,
        }
    }

    /// A transient over `v`'s items.
    #[must_use]
    pub fn from_vector(v: &Vector) -> Self {
        let items = match v.typed_items() {
            Some(crate::value::TypedItems::Ints(x)) => TransientItems::Ints(x),
            Some(crate::value::TypedItems::Chars(x)) => TransientItems::Chars(x),
            None => TransientItems::Generic(v.iter().collect()),
        };
        Self { items, valid: true }
    }

    fn check(&self) -> Result<(), TransientError> {
        ensure(self.valid)
    }

    /// Append one value (`conj!` takes exactly one; arity is the
    /// core fn's job in S5). A foreign value into a packed buffer
    /// converts it to generic in place (the foreign value stays, so
    /// the buffer never re-packs).
    pub fn conj(&mut self, v: Value) -> Result<(), TransientError> {
        self.check()?;
        let len = self.items.len();
        self.items.put(len, v);
        Ok(())
    }

    /// Drop the last value (empty errors).
    pub fn pop(&mut self) -> Result<(), TransientError> {
        self.check()?;
        let empty = match &mut self.items {
            TransientItems::Ints(x) => x.pop().is_none(),
            TransientItems::Chars(x) => x.pop().is_none(),
            TransientItems::Generic(x) => x.pop().is_none(),
        };
        if empty {
            return Err(TransientError::EmptyPop);
        }
        Ok(())
    }

    /// Set index `idx` (`idx == len` appends, past that errors).
    pub fn assoc(&mut self, idx: usize, v: Value) -> Result<(), TransientError> {
        self.check()?;
        if idx > self.items.len() {
            return Err(TransientError::IndexOutOfBounds);
        }
        self.items.put(idx, v);
        Ok(())
    }

    /// Read index `idx` (`None` past the end).
    pub fn nth(&self, idx: usize) -> Result<Option<Value>, TransientError> {
        self.check()?;
        Ok(match &self.items {
            TransientItems::Ints(x) => x.get(idx).map(|n| Value::Int(*n)),
            TransientItems::Chars(x) => x.get(idx).map(|c| Value::Char(*c)),
            TransientItems::Generic(x) => x.get(idx).cloned(),
        })
    }

    /// The item count.
    pub fn len(&self) -> Result<usize, TransientError> {
        self.check()?;
        Ok(self.items.len())
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> Result<bool, TransientError> {
        Ok(self.len()? == 0)
    }

    /// Whether the buffer is still packed (all-`Int` or all-char so
    /// far).
    #[must_use]
    pub fn is_packed(&self) -> bool {
        !matches!(self.items, TransientItems::Generic(_))
    }

    /// Freeze into a persistent vector (invalidates this transient).
    pub fn persistent(&mut self) -> Result<Vector, TransientError> {
        self.check()?;
        self.valid = false;
        let items = std::mem::replace(&mut self.items, TransientItems::Ints(Vec::new()));
        // Zero-copy for packed runs (no re-scan); growth slack is shed
        // first — the frozen vector never grows.
        Ok(match items {
            TransientItems::Ints(mut x) => {
                x.shrink_to_fit();
                Vector::from_ints(x)
            }
            TransientItems::Chars(mut x) => {
                x.shrink_to_fit();
                Vector::from_chars(x)
            }
            TransientItems::Generic(x) => Vector::from_vec(x),
        })
    }
}

impl Default for TransientVector {
    fn default() -> Self {
        Self::new()
    }
}

/// The simulated array capacity growth
/// (`TransientArrayMap.newCapacity`: `min(ceil(n*1.65),128)`, even).
fn transient_capacity(n: usize) -> usize {
    let grown = ((n as f64 * 1.65).ceil() as usize).min(128);
    grown + (grown & 1)
}

/// A transient map: `assoc!`/`dissoc!` buffer with insertion-ordered
/// freeze (re-assoc keeps the original key and position).
///
/// `len`/`cap` simulate the JVM array (slots, 2 per pair):
/// appends below capacity never promote; a full array promotes to
/// hash when holding 128 slots or receiving a non-keyword key,
/// else grows (`newCapacity`) for keyword keys. A hash source
/// starts promoted. `dissoc!` swap-removes (the JVM moves the last
/// pair into the hole).
#[derive(Clone, Debug)]
pub struct TransientMap {
    entries: Vec<(Value, Value)>,
    index: HashMap<Value, usize, FxBuildHasher>,
    valid: bool,
    len: usize,
    cap: usize,
    promoted: bool,
}

impl TransientMap {
    /// An empty transient map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::default(),
            valid: true,
            len: 0,
            cap: 16,
            promoted: false,
        }
    }

    /// A transient over `m`'s entries (insertion order kept; hash
    /// sources start promoted with their exact array size).
    #[must_use]
    pub fn from_map(m: &Map) -> Self {
        let mut t = Self::new();
        for (k, v) in m.iter() {
            t.index.insert(k.clone(), t.entries.len());
            t.entries.push((k.clone(), v.clone()));
        }
        t.len = t.entries.len() * 2;
        t.cap = t.len.max(16);
        t.promoted = m.is_hashy();
        t
    }

    fn check(&self) -> Result<(), TransientError> {
        ensure(self.valid)
    }

    /// Whether this is a hash transient (promoted or hash-born).
    #[must_use]
    pub fn is_hashy(&self) -> bool {
        self.promoted
    }

    /// Assoc a key/value pair.
    pub fn assoc(&mut self, k: Value, v: Value) -> Result<(), TransientError> {
        self.check()?;
        if let Some(&i) = self.index.get(&k) {
            self.entries[i].1 = v;
            return Ok(());
        }
        if !self.promoted {
            if self.len < self.cap {
                self.len += 2;
            } else if self.len >= 128 || !matches!(k, Value::Keyword(_)) {
                self.promoted = true;
                self.len += 2;
            } else {
                self.cap = transient_capacity(self.cap);
                self.len += 2;
            }
        }
        self.index.insert(k.clone(), self.entries.len());
        self.entries.push((k, v));
        Ok(())
    }

    /// Drop a key (missing is a no-op; the last pair moves into the
    /// hole, like the JVM).
    pub fn dissoc(&mut self, k: &Value) -> Result<(), TransientError> {
        self.check()?;
        if let Some(i) = self.index.remove(k) {
            self.entries.swap_remove(i);
            self.len = self.len.saturating_sub(2);
            if let Some((moved, _)) = self.entries.get(i) {
                let moved = moved.clone();
                self.index.insert(moved, i);
            }
        }
        Ok(())
    }

    /// Look up a key.
    pub fn get(&self, k: &Value) -> Result<Option<&Value>, TransientError> {
        self.check()?;
        Ok(self.index.get(k).map(|&i| &self.entries[i].1))
    }

    /// The entry count.
    pub fn len(&self) -> Result<usize, TransientError> {
        self.check()?;
        Ok(self.entries.len())
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> Result<bool, TransientError> {
        Ok(self.len()? == 0)
    }

    /// Freeze into a persistent map (invalidates this transient).
    /// The class comes from the promotion simulation, never the
    /// content (an unpromoted array keeps array class at any size).
    /// The lookup index is dropped before the build (not after), so
    /// its duplicate keys are not live alongside the persistent map
    /// under construction (#221 R3: peak trim for transient builds).
    pub fn persistent(&mut self) -> Result<Map, TransientError> {
        self.check()?;
        self.valid = false;
        let entries = std::mem::take(&mut self.entries);
        self.index.clear();
        self.index.shrink_to_fit();
        let mut map = new_map();
        for (k, v) in entries {
            map = map.insert(k, v);
        }
        Ok(map.with_hashy(self.promoted))
    }
}

impl Default for TransientMap {
    fn default() -> Self {
        Self::new()
    }
}

/// A transient set: `conj!`/`disj!` buffer with insertion-ordered
/// freeze (re-conj keeps the original member).
#[derive(Clone, Debug)]
pub struct TransientSet {
    members: Vec<Value>,
    index: HashSet<Value, FxBuildHasher>,
    valid: bool,
}

impl TransientSet {
    /// An empty transient set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            index: HashSet::default(),
            valid: true,
        }
    }

    /// A transient over `s`'s members (insertion order kept).
    #[must_use]
    pub fn from_set(s: &Set) -> Self {
        let mut t = Self::new();
        for m in s.iter() {
            t.index.insert(m.clone());
            t.members.push(m.clone());
        }
        t
    }

    fn check(&self) -> Result<(), TransientError> {
        ensure(self.valid)
    }

    /// Add a member (present is a no-op, keeping the original).
    pub fn conj(&mut self, v: Value) -> Result<(), TransientError> {
        self.check()?;
        if self.index.insert(v.clone()) {
            self.members.push(v);
        }
        Ok(())
    }

    /// Drop a member (missing is a no-op).
    pub fn disj(&mut self, v: &Value) -> Result<(), TransientError> {
        self.check()?;
        if self.index.remove(v) {
            self.members.retain(|m| m != v);
        }
        Ok(())
    }

    /// Whether a member is present.
    pub fn contains(&self, v: &Value) -> Result<bool, TransientError> {
        self.check()?;
        Ok(self.index.contains(v))
    }

    /// The member count.
    pub fn len(&self) -> Result<usize, TransientError> {
        self.check()?;
        Ok(self.members.len())
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> Result<bool, TransientError> {
        Ok(self.len()? == 0)
    }

    /// Freeze into a persistent set (invalidates this transient).
    /// The lookup index is dropped before the build (not after), so
    /// its duplicate members are not live alongside the persistent
    /// set under construction (#221 R3: peak trim for transient
    /// builds).
    pub fn persistent(&mut self) -> Result<Set, TransientError> {
        self.check()?;
        self.valid = false;
        let members = std::mem::take(&mut self.members);
        self.index.clear();
        self.index.shrink_to_fit();
        let mut set = new_set();
        for v in members {
            set = set.insert(v);
        }
        Ok(set)
    }
}

impl Default for TransientSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Which persistent type a transient builds.
#[derive(Clone, Debug)]
pub enum TransientKind {
    /// `clojure.lang.PersistentVector$TransientVector`.
    Vector(TransientVector),
    /// `clojure.lang.PersistentArrayMap$TransientArrayMap`.
    Map(TransientMap),
    /// `clojure.lang.PersistentHashSet$TransientHashSet`.
    Set(TransientSet),
}

impl TransientKind {
    /// The JVM class name (for casts, `seq` errors, and printing).
    /// Map transients report hash past promotion.
    #[must_use]
    pub fn jvm_class(&self) -> &'static str {
        match self {
            TransientKind::Vector(_) => "clojure.lang.PersistentVector$TransientVector",
            TransientKind::Map(m) if m.is_hashy() => {
                "clojure.lang.PersistentHashMap$TransientHashMap"
            }
            TransientKind::Map(_) => "clojure.lang.PersistentArrayMap$TransientArrayMap",
            TransientKind::Set(_) => "clojure.lang.PersistentHashSet$TransientHashSet",
        }
    }
}

/// Fresh transient identities (print-only; equality is shared-cell).
static TRANSIENT_ID: AtomicU64 = AtomicU64::new(1);

/// A first-class transient value: shared mutable buffer plus the
/// source metadata (invisible to `meta`, restored by
/// `persistent!`). Identity equality (same cell); only `counted?`
/// among the type predicates. The metadata rides behind an `Arc`:
/// an inline `Map` would inflate this struct past 100 bytes, and
/// every `Value` pays the largest variant's size inline (#221 R1).
#[derive(Clone, Debug)]
pub struct Transient {
    id: u64,
    cell: Arc<Mutex<TransientKind>>,
    meta: Option<Arc<Map>>,
}

impl Transient {
    /// A transient over `kind` (fresh identity, no metadata).
    #[must_use]
    pub fn new(kind: TransientKind) -> Self {
        Self::with_meta(kind, None)
    }

    /// A transient over `kind` carrying the source metadata.
    #[must_use]
    pub fn with_meta(kind: TransientKind, meta: Option<Map>) -> Self {
        Self {
            id: TRANSIENT_ID.fetch_add(1, Ordering::Relaxed),
            cell: Arc::new(Mutex::new(kind)),
            meta: meta.map(Arc::new),
        }
    }

    /// The print identity.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Whether `other` shares this transient's buffer.
    #[must_use]
    pub fn same_cell(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cell, &other.cell)
    }

    /// The source metadata (restored by `persistent!`).
    #[must_use]
    pub fn source_meta(&self) -> Option<&Map> {
        self.meta.as_deref()
    }

    /// Lock the buffer (single-threaded use; poison is unreachable).
    pub fn lock(&self) -> MutexGuard<'_, TransientKind> {
        self.cell.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The JVM class name of the buffer kind (live: map
    /// transients report hash past promotion).
    #[must_use]
    pub fn jvm_class(&self) -> &'static str {
        self.lock().jvm_class()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Transient, TransientError, TransientKind, TransientMap, TransientSet, TransientVector,
    };
    use crate::{Keyword, Value};

    fn kw(name: &str) -> Value {
        Value::Keyword(Keyword::new(None, name))
    }

    #[test]
    fn error_display_matches_message() {
        assert_eq!(
            TransientError::UseAfterPersistent.to_string(),
            "Transient used after persistent! call"
        );
        assert_eq!(
            TransientError::EmptyPop.to_string(),
            "Can't pop empty vector"
        );
        assert_eq!(TransientError::IndexOutOfBounds.to_string(), "");
    }

    #[test]
    fn vector_is_empty_and_default() {
        let mut t = TransientVector::default();
        assert!(t.is_empty().unwrap());
        t.conj(Value::Int(1)).unwrap();
        assert!(!t.is_empty().unwrap());
        assert_eq!(t.len().unwrap(), 1);
    }

    #[test]
    fn map_assoc_grows_for_keyword_keys() {
        let mut t = TransientMap::new();
        for i in 0..8 {
            t.assoc(kw(&format!("k{i}")), Value::Int(i)).unwrap();
        }
        assert!(!t.is_hashy());
        t.assoc(kw("k8"), Value::Int(8)).unwrap();
        assert!(!t.is_hashy());
        assert_eq!(t.len().unwrap(), 9);
        assert_eq!(t.get(&kw("k8")).unwrap(), Some(&Value::Int(8)));
    }

    #[test]
    fn map_assoc_promotes_for_non_keyword_keys() {
        let mut t = TransientMap::new();
        for i in 0..8 {
            t.assoc(kw(&format!("k{i}")), Value::Int(i)).unwrap();
        }
        assert!(!t.is_hashy());
        t.assoc(Value::Int(99), Value::Int(99)).unwrap();
        assert!(t.is_hashy());
        assert_eq!(t.len().unwrap(), 9);
        // Assoc past promotion skips the capacity check (the `!promoted` guard).
        t.assoc(Value::Int(100), Value::Int(100)).unwrap();
        assert_eq!(t.len().unwrap(), 10);
        assert_eq!(t.get(&Value::Int(100)).unwrap(), Some(&Value::Int(100)));
    }

    #[test]
    fn map_dissoc_moves_last_into_hole() {
        let mut t = TransientMap::new();
        t.assoc(kw("a"), Value::Int(1)).unwrap();
        t.assoc(kw("b"), Value::Int(2)).unwrap();
        t.assoc(kw("c"), Value::Int(3)).unwrap();
        t.dissoc(&kw("a")).unwrap();
        assert_eq!(t.len().unwrap(), 2);
        assert_eq!(t.get(&kw("a")).unwrap(), None);
        assert_eq!(t.get(&kw("b")).unwrap(), Some(&Value::Int(2)));
        assert_eq!(t.get(&kw("c")).unwrap(), Some(&Value::Int(3)));
        t.dissoc(&kw("missing")).unwrap();
        assert_eq!(t.len().unwrap(), 2);
    }

    #[test]
    fn map_is_empty_and_default() {
        let mut t = TransientMap::default();
        assert!(t.is_empty().unwrap());
        t.assoc(kw("a"), Value::Int(1)).unwrap();
        assert!(!t.is_empty().unwrap());
    }

    #[test]
    fn set_is_empty_and_default() {
        let mut t = TransientSet::default();
        assert!(t.is_empty().unwrap());
        t.conj(Value::Int(1)).unwrap();
        assert!(!t.is_empty().unwrap());
    }

    #[test]
    fn transient_same_cell() {
        let t = Transient::new(TransientKind::Vector(TransientVector::new()));
        assert!(t.same_cell(&t.clone()));
        let u = Transient::new(TransientKind::Vector(TransientVector::new()));
        assert!(!t.same_cell(&u));
    }
}
