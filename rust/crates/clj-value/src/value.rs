// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! The [`Value`] enum: every runtime value is one variant, immutable
//! and `Send + Sync`, shared by `Arc` (no tracing GC — drops reclaim).
//!
//! Equality is Clojure `=` ([`PartialEq`]): same-type numerics
//! (`Int`/`Int` by value, `Double`/`Double` by `==` — `(= 1 1.0)`
//! is false, `(= 0.0 -0.0)` true, `(= ##NaN ##NaN)` false),
//! sequential comparison across [`Value::List`]/[`Value::Vector`],
//! and identity comparison for [`Value::Regex`] (mirroring
//! `java.util.regex.Pattern`, which has no value equality: two
//! separately read patterns are never `=`, while two handles to the
//! same value are).
//!
//! [`Hash`] is equiv-consistent: values that are `=` hash equally,
//! so `rpds` maps and sets obey Clojure lookup semantics (mixed
//! numerics are distinct keys: `(get {1 :a} 1.0)` misses and
//! `{1 :a 1.0 :b}` keeps both). NaN hashes deterministically
//! (canonical bits); `-0.0` hashes as `0.0` (they are `=`).
//!
//! Collection literals are duplicate-checked on construction
//! ([`Map::from_kvs_checked`], [`Set::from_items_checked`]) with the
//! JVM `RT.map` / `PersistentHashSet.createWithCheck` rules
//! (`docs/spec/t1174/reader-contract.md` §19): a small-map pairwise scan
//! reporting the FIRST key, otherwise an incremental scan reporting
//! the NEW key, comparing with a line-by-line port of
//! `Util.equiv`/`Util.equals` (see [`jvm_equiv`]).

use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Weak};

use num_traits::ToPrimitive;

use crate::{
    Agent, Array, Atom, ClassVal, Delay, Executor, ExtensionValue, Future, HostInst, HostKind,
    HostVal, Keyword, MultiFn, PrimitiveVector, Promise, Ref, Symbol, Throwable, Transient,
    Unbound, Var, Volatile, host_eq,
};

/// Deterministic Fx hasher for maps/sets (rpds's default RandomState
/// is not stable across runs). Same non-crypto posture as the JVM's
/// murmur-based hashing. (Iteration order comes from the insertion
/// sidecars below, not the hasher.)
pub type FxBuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;

type RpdsList = rpds::ListSync<Value>;
/// The persistent vector trie (#226: items inline, an O(1) tail).
type RpdsVector = crate::trie::PVec;
type RpdsMap = rpds::HashTrieMapSync<Value, Value, FxBuildHasher>;
type RpdsSet = rpds::HashTrieSetSync<Value, FxBuildHasher>;
/// A set's live members, each mapped to its slot in the order sidecar.
type RpdsIndex = rpds::HashTrieMapSync<Value, usize, FxBuildHasher>;
/// A set's order sidecar: one slot per member added since the last
/// compaction; a removed member's slot is a hole (`None`).
type SetSlots = crate::trie::PVec<Option<Value>>;

/// Identity counters for `identical?` (S16): every freshly
/// constructed [`List`]/[`Vector`]/[`Set`] mints an id from its
/// counter (the `ATOM_ID` precedent); `Clone` preserves it, so two
/// evaluations of the same binding compare identical while two
/// separately built collections never do. Equality and hashing
/// ignore the ids. [`Map`] instead identifies by an allocation
/// address (see [`Map::id`]) — no counter, no field.
static LIST_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static VECTOR_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static SET_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_list_id() -> u64 {
    LIST_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn next_vector_id() -> u64 {
    VECTOR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn next_set_id() -> u64 {
    SET_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Shared identity counter (fence 4a): every freshly constructed
/// [`Symbol`](crate::Symbol)/[`Uuid`](crate::Uuid)/[`Instant`](crate::Instant)/
/// [`PrimitiveVector`](crate::PrimitiveVector)/`MapEntry`/`TaggedLiteral`/
/// `ReaderConditional` mints an id here (`Clone` preserves it, like
/// the per-kind counters above; equality and hashing ignore the
/// ids). One counter for all seven: ids are only ever compared
/// within a kind.
static IDENTITY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(crate) fn next_identity_id() -> u64 {
    IDENTITY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// `BigDecimal.ZERO_THROUGH_TEN` (fence 4a-big, oracle P5):
/// scale-0 `0..=10` decimals share these instances — but ONLY
/// for arithmetic results (see [`bigdec_result`]), like the
/// JDK's `BigDecimal.valueOf` compact paths. Literals, parses,
/// and coercions always box fresh (`Arc::new`), like the JDK's
/// `new BigDecimal` ctors.
static BIGDEC_ZERO_THROUGH_TEN: std::sync::LazyLock<[Arc<crate::BigDec>; 11]> =
    std::sync::LazyLock::new(|| {
        std::array::from_fn(|i| Arc::new(crate::BigDec::from_bigint(num_bigint::BigInt::from(i))))
    });

/// The `ZERO_THROUGH_TEN` slot for a scale-0 `0..=10` decimal
/// (`None` otherwise — nonzero scales, negatives, `11+`, and
/// past-`i64` magnitudes all stay fresh, like the JDK).
fn canonical_bigdec_index(d: &crate::BigDec) -> Option<usize> {
    (d.scale() == 0)
        .then(|| d.unscaled().to_i64())
        .flatten()
        .filter(|n| (0..=10).contains(n))
        .map(|n| n as usize)
}

/// Wrap an arithmetic-result decimal (see
/// [`BIGDEC_ZERO_THROUGH_TEN`]): scale-0 `0..=10` results
/// share the canonical instance (the JVM's `valueOf` compact
/// paths — `+`/`-`/`*`/`/` and friends all route here);
/// anything else boxes fresh. Literals, parses, and coercions
/// must NOT call this (they box fresh, like `new BigDecimal`).
#[must_use]
pub fn bigdec_result(d: crate::BigDec) -> Value {
    match canonical_bigdec_index(&d) {
        Some(n) => Value::BigDec(BIGDEC_ZERO_THROUGH_TEN[n].clone()),
        None => Value::BigDec(Arc::new(d)),
    }
}

/// The JVM string-pool analog (fence 4a): every string LITERAL the
/// analyzer walks canonicalizes here, so equal literals share one
/// `Arc` (like the JVM's interned class constants) while runtime
/// constructions (`str`, `subs`, `read-string` results) stay fresh.
/// `identical?` on strings is `Arc::ptr_eq`.
static STRING_POOL: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<Arc<str>, Arc<str>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Canonicalize one analyzed string literal (see [`STRING_POOL`]).
/// The analyzer calls this for every literal string it walks
/// (including inside `quote` forms via
/// [`intern_literal_strings`]); runtime string constructions must
/// NOT call it (they stay fresh, like the JVM's `new String`).
#[must_use]
pub fn intern_string(s: &Arc<str>) -> Arc<str> {
    let mut pool = STRING_POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = pool.get(s) {
        return hit.clone();
    }
    pool.insert(s.clone(), s.clone());
    s.clone()
}

/// Canonicalize a literal form's top-level string, if any (see
/// [`intern_string`]): the analyzer's catch-all literal arm runs
/// every non-collection scalar through here, so string literals
/// pool without the analyzer gaining a match arm (complexity
/// ratchet).
#[must_use]
pub fn intern_string_value(form: &Value) -> Value {
    match form {
        Value::Str(s) => Value::Str(intern_string(s)),
        _ => form.clone(),
    }
}

/// Rebuild a quoted form with every literal string canonicalized
/// (see [`intern_string`]): `(identical? (quote "ab") "ab")` is
/// `true` on the JVM (the quote answers the interned constant),
/// so the analyzer runs this over `quote` payloads. Collections
/// rebuild with fresh identity ids (each quote form is its own
/// constant); nested scalars keep their semantics (symbols stay
/// fresh — the JVM news them per read).
#[must_use]
pub fn intern_literal_strings(form: &Value) -> Value {
    match form {
        Value::Str(s) => Value::Str(intern_string(s)),
        Value::List(l) => {
            let items = l.iter().map(intern_literal_strings).collect();
            let mut out = List::from_vec(items);
            if let Some(meta) = l.meta() {
                out = out.with_meta(Some(meta.clone()));
            }
            Value::List(out)
        }
        Value::Vector(v) => {
            let items = v.iter().map(|v| intern_literal_strings(&v)).collect();
            let mut out = Vector::from_vec(items);
            if let Some(meta) = v.meta() {
                out = out.with_meta(Some(meta.clone()));
            }
            Value::Vector(out)
        }
        // The ordered rebuilds preserve metadata, key order, and
        // the map/set class flags (each quote form stays its own
        // constant with a fresh identity id).
        Value::Map(m) => {
            let entries = m
                .iter()
                .map(|(k, v)| (intern_literal_strings(k), intern_literal_strings(v)))
                .collect();
            Value::Map(m.clone().with_ordered_entries(entries))
        }
        Value::Set(s) => {
            let members = s.iter().map(intern_literal_strings).collect();
            Value::Set(s.clone().with_ordered_members(members))
        }
        Value::MapEntry { key, val, .. } => {
            map_entry(intern_literal_strings(key), intern_literal_strings(val))
        }
        Value::TaggedLiteral { tag, form, .. } => {
            tagged_literal((**tag).clone(), intern_literal_strings(form))
        }
        Value::ReaderConditional { form, splicing, .. } => {
            reader_conditional(intern_literal_strings(form), *splicing)
        }
        other => other.clone(),
    }
}

/// A list's fields, behind the handle's `Arc` so a `List`
/// (inline in every `Value`) stays one word (#254: `Value`
/// shrinks to 32 bytes).
#[derive(Clone)]
struct ListData {
    inner: RpdsList,
    meta: Option<Arc<Map>>,
    id: u64,
}

/// A persistent list of values (thread-safe rpds list) with optional
/// metadata. Method names match the bare rpds type so readers keep
/// compiling; equality and hashing ignore the metadata.
#[derive(Clone)]
pub struct List(Arc<ListData>);

impl std::fmt::Debug for List {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("List")
            .field("inner", &self.0.inner)
            .field("meta", &self.0.meta)
            .field("id", &self.0.id)
            .finish()
    }
}

/// A persistent vector of values (thread-safe rpds vector) with
/// optional metadata. See [`List`].
///
/// Homogeneous `Int` vectors ride packed (8 bytes/element instead
/// of 72 — #221 P1); the representation is invisible (every method
/// presents `Value`s), so packing never changes `=`/`hash`/
/// `identical?`/seq behavior (`identical?` on `Int` is
/// value-based inside the `Long` cache range, fresh-boxed outside —
/// packed reads box identically to generic reads).
/// A vector's fields, behind the handle's `Arc` so a `Vector`
/// (inline in every `Value`) stays one word (#254: `Value`
/// shrinks to 32 bytes). Writers go through `Arc::make_mut`, so
/// a shared handle's data clones before mutation, exactly like
/// the old struct copy.
#[derive(Clone)]
struct VectorData {
    inner: VectorInner,
    meta: Option<Arc<Map>>,
    id: u64,
}

#[derive(Clone)]
pub struct Vector(Arc<VectorData>);

impl std::fmt::Debug for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vector")
            .field("inner", &self.0.inner)
            .field("meta", &self.0.meta)
            .field("id", &self.0.id)
            .finish()
    }
}

/// Flat byte buffer + end-offset table for homogeneous string vectors (#271).
/// A run of adjacent identical strings can reference the same physical
/// span, using an index in addition to the original end-offset table.
#[derive(Clone, Debug, Default)]
pub struct PackedStrings {
    buf: String,
    ends: Vec<u32>,
    runs: Option<Box<StringRuns>>,
}

#[derive(Clone, Debug, Default)]
struct StringRuns {
    /// Each run is (physical item index, cumulative duplicate count).
    entries: Vec<(usize, usize)>,
    extra_bytes: usize,
    expanded: std::sync::OnceLock<String>,
}

impl StringRuns {
    #[inline(always)]
    fn extra_items(&self) -> usize {
        self.entries.last().map_or(0, |&(_, extra)| extra)
    }

    #[inline(always)]
    fn physical_index(&self, i: usize) -> usize {
        let after = self.entries.partition_point(|&(at, extra)| at + extra < i);
        let skipped = after.checked_sub(1).map_or(0, |p| self.entries[p].1);
        match self.entries.get(after) {
            Some(&(at, _)) if i >= at + skipped => at,
            _ => i - skipped,
        }
    }

    fn push(&mut self, physical: usize, bytes: usize) {
        let extra = self.extra_items() + 1;
        match self.entries.last_mut() {
            Some((at, count)) if *at == physical => *count = extra,
            _ => self.entries.push((physical, extra)),
        }
        self.extra_bytes += bytes;
        self.expanded.take();
    }

    fn pop(&mut self, physical: usize, bytes: usize) -> bool {
        self.expanded.take();
        let prev = self
            .entries
            .len()
            .checked_sub(2)
            .map_or(0, |p| self.entries[p].1);
        let Some(last) = self.entries.last_mut().filter(|r| r.0 == physical) else {
            return false;
        };
        last.1 -= 1;
        self.extra_bytes -= bytes;
        if last.1 == prev {
            self.entries.pop();
        }
        true
    }
}

impl PackedStrings {
    /// An empty packed string buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-sized empty buffer for `n` items totaling `bytes` UTF-8 bytes.
    #[must_use]
    pub(crate) fn with_capacity(n: usize, bytes: usize) -> Self {
        Self {
            buf: String::with_capacity(bytes),
            ends: Vec::with_capacity(n),
            runs: None,
        }
    }

    /// Number of stored strings.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ends.len() + self.runs.as_ref().map_or(0, |r| r.extra_items())
    }

    /// Whether no strings are stored.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The concatenated byte buffer of all stored strings.
    #[inline]
    #[must_use]
    pub fn raw_buf(&self) -> &str {
        match &self.runs {
            Some(runs) => runs.expanded.get_or_init(|| self.join("")),
            None => &self.buf,
        }
    }

    /// Borrow string `idx` (`None` past the end; panic-free for `clj-bc` bitcode).
    #[inline]
    #[must_use]
    pub fn get_str(&self, idx: usize) -> Option<&str> {
        self.physical_str(self.runs.as_ref().map_or(idx, |r| r.physical_index(idx)))
    }

    #[inline(always)]
    fn physical_str(&self, idx: usize) -> Option<&str> {
        let end = *self.ends.get(idx)? as usize;
        let start = match idx.checked_sub(1) {
            Some(prev) => *self.ends.get(prev)? as usize,
            None => 0,
        };
        self.buf.get(start..end)
    }

    /// Append `s` when total bytes stay within `limit` (`false` otherwise).
    #[inline]
    pub(crate) fn push_bounded(&mut self, s: &str, limit: usize) -> bool {
        let total = self.buf.len() + self.runs.as_ref().map_or(0, |r| r.extra_bytes);
        if s.len() > limit.saturating_sub(total) {
            return false;
        }
        if self.buf.len() >= 4096
            && s.len() >= 12
            && self.physical_str(self.ends.len() - 1) == Some(s)
        {
            self.runs
                .get_or_insert_with(Box::default)
                .push(self.ends.len() - 1, s.len());
            return true;
        }
        if let Some(runs) = &mut self.runs {
            runs.expanded.take();
        }
        if self.buf.capacity() >= 256 * 1024 && s.len() > self.buf.capacity() - self.buf.len() {
            self.buf
                .reserve_exact((self.buf.capacity() / 2).max(s.len()));
        }
        self.buf.push_str(s);
        if self.ends.len() == self.ends.capacity() && self.ends.capacity() >= 64 * 1024 {
            self.ends.reserve_exact(self.ends.capacity() / 2);
        }
        self.ends.push(self.buf.len() as u32);
        true
    }

    /// Append `s` when total bytes stay within `u32::MAX`.
    #[inline]
    pub fn try_push(&mut self, s: &str) -> bool {
        self.push_bounded(s, u32::MAX as usize)
    }

    /// Drop the last string (`false` when empty).
    #[inline]
    pub fn pop(&mut self) -> bool {
        if self.pop_repeated() {
            return true;
        }
        if self.ends.pop().is_none() {
            return false;
        }
        let new_len = self.ends.last().copied().unwrap_or(0) as usize;
        self.buf.truncate(new_len);
        true
    }

    fn pop_repeated(&mut self) -> bool {
        if self.runs.is_none() {
            return false;
        }
        let i = self.ends.len() - 1;
        let bytes = self.physical_str(i).expect("last stored item").len();
        let runs = self.runs.as_mut().expect("has run index");
        let popped = runs.pop(i, bytes);
        if runs.entries.is_empty() {
            self.runs = None;
        }
        popped
    }

    /// Release excess capacity only when more than half the buffer is slack
    /// (avoiding a `realloc` copy-peak on freshly grown buffers before `join`).
    pub(crate) fn shed_slack(&mut self) {
        if self.buf.capacity() - self.buf.len() > self.buf.len().max(4096) {
            self.buf.shrink_to_fit();
        }
        if self.ends.capacity() - self.ends.len() > self.ends.len().max(1024) {
            self.ends.shrink_to_fit();
        }
    }

    /// Join every stored string with `sep` directly from the flat buffer.
    #[must_use]
    pub fn join(&self, sep: &str) -> String {
        if let Some(runs) = &self.runs {
            let cap = self
                .buf
                .len()
                .saturating_add(runs.extra_bytes)
                .saturating_add(sep.len().saturating_mul(self.len() - 1));
            let mut out = String::with_capacity(cap);
            self.push_runs(&mut out, sep, runs);
            return out;
        }
        let n = self.ends.len();
        if n == 0 {
            return String::new();
        }
        let cap = self
            .buf
            .len()
            .saturating_add(sep.len().saturating_mul(n - 1));
        let mut out = String::with_capacity(cap);
        let mut start = 0usize;
        for (i, &end) in self.ends.iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            let end = end as usize;
            out.push_str(&self.buf[start..end]);
            start = end;
        }
        out
    }

    fn push_runs(&self, out: &mut String, sep: &str, runs: &StringRuns) {
        let mut repeats = runs.entries.iter().peekable();
        let mut extra = 0;
        let mut start = 0;
        for (i, &end) in self.ends.iter().enumerate() {
            if i != 0 {
                out.push_str(sep);
            }
            let end = end as usize;
            let text = &self.buf[start..end];
            out.push_str(text);
            if let Some(&(_, total)) = repeats.next_if(|&&(at, _)| at == i) {
                for _ in extra..total {
                    out.push_str(sep);
                    out.push_str(text);
                }
                extra = total;
            }
            start = end;
        }
    }

    fn push_to(&self, out: &mut String) {
        if let Some(runs) = &self.runs {
            self.push_runs(out, "", runs);
        } else {
            out.push_str(&self.buf);
        }
    }

    /// Join every stored string with `sep` in place inside `self.buf`, dropping
    /// `self.ends` before returning so peak RSS avoids holding both the packed
    /// strings buffer and a second output buffer simultaneously.
    #[must_use]
    pub fn into_joined(self, sep: &str) -> String {
        if self.runs.is_some() {
            return self.join(sep);
        }
        let n = self.ends.len();
        if n == 0 {
            return String::new();
        }
        if sep.is_empty() {
            return self.buf;
        }
        let sep_bytes = sep.as_bytes();
        let total = self
            .buf
            .len()
            .saturating_add(sep_bytes.len().saturating_mul(n - 1));
        let mut bytes = self.buf.into_bytes();
        if total > bytes.capacity() {
            bytes.reserve_exact(total - bytes.len());
        }
        bytes.resize(total, 0);
        let mut write = total;
        for i in (0..n).rev() {
            let end = self.ends[i] as usize;
            let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
            let len = end - start;
            write -= len;
            bytes.copy_within(start..end, write);
            if i > 0 {
                write -= sep_bytes.len();
                bytes[write..write + sep_bytes.len()].copy_from_slice(sep_bytes);
            }
        }
        drop(self.ends);
        String::from_utf8(bytes).unwrap_or_default()
    }

    /// Box all items into a `Vec<Value>`.
    #[must_use]
    pub fn to_values(&self) -> Vec<Value> {
        if self.runs.is_some() {
            return (0..self.len())
                .map(|i| Value::Str(Arc::from(self.get_str(i).expect("in bounds"))))
                .collect();
        }
        let mut out = Vec::with_capacity(self.ends.len());
        let mut start = 0usize;
        for &end in &self.ends {
            let end = end as usize;
            out.push(Value::Str(Arc::from(&self.buf[start..end])));
            start = end;
        }
        out
    }

    /// Up to 32 items starting at `index` (matching `Packed::chunk_from`).
    fn chunk_from(&self, index: usize) -> Vec<Value> {
        if index >= self.len() {
            return Vec::new();
        }
        let end = ((index / 32 + 1) * 32).min(self.len());
        let mut out = Vec::with_capacity(end - index);
        for i in index..end {
            out.push(Value::Str(Arc::from(self.get_str(i).expect("in bounds"))));
        }
        out
    }

    fn iter(&self) -> PackedStringsIter<'_> {
        self.iter_from(0)
    }

    fn iter_from(&self, start: usize) -> PackedStringsIter<'_> {
        let len = self.len();
        PackedStringsIter {
            ps: self,
            lo: start.min(len),
            hi: len,
        }
    }
}

struct PackedStringsIter<'a> {
    ps: &'a PackedStrings,
    lo: usize,
    hi: usize,
}

impl Iterator for PackedStringsIter<'_> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        if self.lo >= self.hi {
            return None;
        }
        let s = self.ps.get_str(self.lo).expect("in bounds");
        self.lo += 1;
        Some(Value::Str(Arc::from(s)))
    }

    fn nth(&mut self, n: usize) -> Option<Value> {
        self.lo = self.lo.saturating_add(n).min(self.hi);
        self.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.hi - self.lo;
        (rem, Some(rem))
    }
}

impl DoubleEndedIterator for PackedStringsIter<'_> {
    fn next_back(&mut self) -> Option<Value> {
        if self.lo >= self.hi {
            return None;
        }
        self.hi -= 1;
        let s = self.ps.get_str(self.hi).expect("in bounds");
        Some(Value::Str(Arc::from(s)))
    }
}

/// Borrowed or copy-out element reference returned by vector lookup
/// without constructing (and dropping) an owned temporary [`Value`].
#[derive(Clone, Copy)]
pub enum ValueRef<'a> {
    /// Packed integer element.
    Int(i64),
    /// Packed double element.
    Double(f64),
    /// Packed character element.
    Char(char),
    /// Packed string slice element.
    Str(&'a str),
    /// Borrowed element from generic storage.
    Ref(&'a Value),
}

impl Vector {
    /// Borrow or copy out the element at `index` without building a
    /// temporary [`Value`] (`None` past the end).
    #[inline(always)]
    #[must_use]
    pub fn get_val_ref(&self, index: usize) -> Option<ValueRef<'_>> {
        match &self.0.inner {
            VectorInner::Generic(t) => t.get(index).map(ValueRef::Ref),
            VectorInner::Ints(t) => t.get(index).copied().map(ValueRef::Int),
            VectorInner::Doubles(t) => t.get(index).copied().map(ValueRef::Double),
            VectorInner::Chars(t) => t.get(index).copied().map(ValueRef::Char),
            VectorInner::Strs(ps) => ps.get_str(index).map(ValueRef::Str),
        }
    }

    /// [`get_val_ref`](Self::get_val_ref) over flat long, double and
    /// generic storage only (`None` for tries, chars, strings and
    /// indices past the end): small enough to inline into compiled
    /// `nth` (#271).
    #[inline(always)]
    #[must_use]
    pub fn flat_val_ref(&self, index: usize) -> Option<ValueRef<'_>> {
        match &self.0.inner {
            VectorInner::Ints(t) => t.as_flat()?.get(index).copied().map(ValueRef::Int),
            VectorInner::Doubles(t) => t.as_flat()?.get(index).copied().map(ValueRef::Double),
            VectorInner::Generic(t) => t.as_flat()?.get(index).map(ValueRef::Ref),
            _ => None,
        }
    }

    /// Borrow the flat packed `i64` slice when this vector is backed by flat ints.
    #[inline(always)]
    #[must_use]
    pub fn as_flat_ints(&self) -> Option<&[i64]> {
        match &self.0.inner {
            VectorInner::Ints(t) => t.as_flat(),
            _ => None,
        }
    }

    /// Borrow the flat packed `f64` slice when this vector is backed by flat doubles.
    #[inline(always)]
    #[must_use]
    pub fn as_flat_doubles(&self) -> Option<&[f64]> {
        match &self.0.inner {
            VectorInner::Doubles(t) => t.as_flat(),
            _ => None,
        }
    }

    /// Borrow the contiguous packed Unicode scalar values.
    #[inline]
    #[must_use]
    pub fn as_flat_chars(&self) -> Option<&[char]> {
        match &self.0.inner {
            VectorInner::Chars(items) => items.as_flat(),
            _ => None,
        }
    }

    /// Borrow the flat generic [`Value`] slice when this vector is backed by
    /// flat generic storage (panic-free for `clj-bc` bitcode).
    #[inline(always)]
    #[must_use]
    pub fn as_flat_values(&self) -> Option<&[Value]> {
        match &self.0.inner {
            VectorInner::Generic(t) => t.as_flat(),
            _ => None,
        }
    }

    /// Whether this vector's backing storage is a contiguous flat buffer
    /// (so `as_flat_*` probes succeed without trie traversal).
    #[inline]
    #[must_use]
    pub fn is_flat(&self) -> bool {
        match &self.0.inner {
            VectorInner::Ints(t) => t.as_flat().is_some(),
            VectorInner::Doubles(t) => t.as_flat().is_some(),
            VectorInner::Chars(t) => t.as_flat().is_some(),
            VectorInner::Generic(t) => t.as_flat().is_some(),
            VectorInner::Strs(_) => true,
        }
    }

    /// Copy the window `start..end` into a fresh [`Vector`], preserving
    /// packed scalar storage (flat when `end - start <= FLAT_MAX`).
    #[must_use]
    pub fn slice_packed(&self, start: usize, end: usize) -> Self {
        let len = end - start;
        match &self.0.inner {
            VectorInner::Ints(t) => {
                Self::from_ints(t.iter_from(start).take(len).copied().collect())
            }
            VectorInner::Doubles(t) => {
                Self::from_doubles(t.iter_from(start).take(len).copied().collect())
            }
            VectorInner::Chars(t) => {
                Self::from_chars(t.iter_from(start).take(len).copied().collect())
            }
            VectorInner::Strs(_) | VectorInner::Generic(_) => {
                Self::from_vec(self.iter_from(start).take(len).collect())
            }
        }
    }

    /// Borrow the packed string storage when this vector holds packed strings.
    #[inline(always)]
    #[must_use]
    pub fn as_packed_strs(&self) -> Option<&PackedStrings> {
        match &self.0.inner {
            VectorInner::Strs(ps) => Some(ps),
            _ => None,
        }
    }

    /// Unwrap uniquely owned packed string storage, returning `Err(self)` when
    /// shared or backed by another vector representation.
    pub fn into_packed_strs(self) -> Result<PackedStrings, Self> {
        let data = Arc::try_unwrap(self.0).map_err(Self)?;
        let VectorData { id, inner, meta } = data;
        match inner {
            VectorInner::Strs(ps) => match Arc::try_unwrap(ps) {
                Ok(owned) => Ok(owned),
                Err(shared) => Err(Self(Arc::new(VectorData {
                    id,
                    inner: VectorInner::Strs(shared),
                    meta,
                }))),
            },
            other => Err(Self(Arc::new(VectorData {
                id,
                inner: other,
                meta,
            }))),
        }
    }
}

/// Vector storage: the generic trie of values, or packed `i64`, `f64`, `char`,
/// or flat string storage (#226, #271).
#[derive(Clone, Debug)]
enum VectorInner {
    Generic(crate::trie::Packed<Value>),
    Ints(crate::trie::Packed<i64>),
    Doubles(crate::trie::Packed<f64>),
    Chars(crate::trie::Packed<char>),
    Strs(Arc<PackedStrings>),
}

/// Owned-value vector iteration (typed items box on read).
pub struct VectorIter<'a> {
    inner: VectorIterInner<'a>,
}

enum VectorIterInner<'a> {
    Generic(crate::trie::PackedIter<'a, Value>),
    Ints(crate::trie::PackedIter<'a, i64>),
    Doubles(crate::trie::PackedIter<'a, f64>),
    Chars(crate::trie::PackedIter<'a, char>),
    Strs(PackedStringsIter<'a>),
}

impl Iterator for VectorIter<'_> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        match &mut self.inner {
            VectorIterInner::Generic(it) => it.next().cloned(),
            VectorIterInner::Ints(it) => it.next().map(|n| Value::Int(*n)),
            VectorIterInner::Doubles(it) => it.next().map(|d| Value::Double(*d)),
            VectorIterInner::Chars(it) => it.next().map(|c| Value::Char(*c)),
            VectorIterInner::Strs(it) => it.next(),
        }
    }

    /// Skips by reference (no clone and drop per skipped item: a walk
    /// over a `subvec` view starts mid-vector, #271).
    fn nth(&mut self, n: usize) -> Option<Value> {
        match &mut self.inner {
            VectorIterInner::Generic(it) => it.nth(n).cloned(),
            VectorIterInner::Ints(it) => it.nth(n).map(|n| Value::Int(*n)),
            VectorIterInner::Doubles(it) => it.nth(n).map(|d| Value::Double(*d)),
            VectorIterInner::Chars(it) => it.nth(n).map(|c| Value::Char(*c)),
            VectorIterInner::Strs(it) => it.nth(n),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            VectorIterInner::Generic(it) => it.size_hint(),
            VectorIterInner::Ints(it) => it.size_hint(),
            VectorIterInner::Doubles(it) => it.size_hint(),
            VectorIterInner::Chars(it) => it.size_hint(),
            VectorIterInner::Strs(it) => it.size_hint(),
        }
    }
}

impl DoubleEndedIterator for VectorIter<'_> {
    fn next_back(&mut self) -> Option<Value> {
        match &mut self.inner {
            VectorIterInner::Generic(it) => it.next_back().cloned(),
            VectorIterInner::Ints(it) => it.next_back().map(|n| Value::Int(*n)),
            VectorIterInner::Doubles(it) => it.next_back().map(|d| Value::Double(*d)),
            VectorIterInner::Chars(it) => it.next_back().map(|c| Value::Char(*c)),
            VectorIterInner::Strs(it) => it.next_back(),
        }
    }
}

/// A persistent map of values (thread-safe rpds hash-trie map) with
/// optional metadata. See [`List`].
///
/// Iteration order is observable — syntax-quote flattens maps in
/// seq order, and `pr` prints in seq order — and follows the JVM
/// per class: shaped maps iterate in insertion order (the JVM's
/// array-maps; re-inserting an existing key keeps its original
/// position and key object, like `assoc`), while plain hash-maps
/// iterate the trie's own content-determined order (like the JVM's
/// murmur-hash order — never insertion). (Byte-exact JVM order
/// would need a from-scratch `PersistentHashMap`; the harness
/// canonicalizes map/set order away, so only syntax-quote
/// expansion and rendering observe this.)
///
/// `keys` is the comparator-order sidecar, kept for sorted maps
/// only (plain hash-maps iterate the trie, shaped maps the shape).
/// It rides behind an `Arc` (writes go through `Arc::make_mut`,
/// which clones the O(1) persistent header when shared): the rpds
/// vector header is 24 bytes, and every `Value` pays the largest
/// variant's size inline (#221 R1). Cloning a map stays
/// allocation-free, like the `meta` arc beside it.
///
/// `hashy` reports `PersistentHashMap` instead of
/// `PersistentArrayMap`: promotion is one-way and history-dependent
/// (content alone cannot decide — `dissoc` never demotes and
/// `array-map` forces array), so it is stored, not derived.
///
/// `sort` carries the FT6 sorted-map order: `None` is an ordinary
/// map; `Some` keeps `keys` in comparator order (compare-0
/// duplicates collapse at insertion, like repeated `assoc`). The
/// inner trie stays content-exact, so equality, hashing, printing,
/// and seqs need no sorted branch — only order-sensitive writers
/// (`assoc` position) and readers (`get`/`dissoc` by compare-0 key,
/// `jvm_class_of`, `nth`, `sorted?`) consult it.
/// Entries at or below this count ride shaped (an interned
/// key-set plus a flat values vec, no HAMT or sidecar allocs) —
/// the JVM's `PersistentArrayMap` converts past 16 pairs.
/// Representation is orthogonal to the reported class
/// (`hashy`): a 9-pair all-keyword map stays array-classed
/// whether shaped or hashed.
pub(crate) const INLINE_MAP_MAX: usize = 16;

/// A small map's shared key-set (#221 suffix states: 200k maps
/// share ONE `[:len :link :next]` shape, so per-map memory is
/// values-only). Keys ride in iteration order; maps with the
/// same `==`-equal keys (in the same order) share one shape.
/// First-interned key objects win: keys that CAN carry metadata
/// (symbols — keywords/strings cannot) keep the first map's key
/// object for iteration, so `(meta (first (keys m)))` may see a
/// different lineage than the JVM's own-key object. Values and
/// `=`/`seq` semantics are unaffected.
#[derive(Debug)]
struct MapShape {
    keys: Box<[Value]>,
    ascii_map: [u8; 128],
    has_big_int_keys: bool,
}

fn build_ascii_map(keys: &[Value]) -> [u8; 128] {
    let mut ascii_map = [0xFFu8; 128];
    for (idx, k) in keys.iter().enumerate() {
        if let Value::Char(c) = k {
            let cu = *c as u32;
            if cu < 128 {
                ascii_map[cu as usize] = idx as u8;
            }
        }
    }
    ascii_map
}

impl MapShape {
    #[inline]
    fn find_key(&self, k: &Value) -> Option<usize> {
        if let Value::Char(c) = k {
            let cu = *c as u32;
            if cu < 128 {
                let idx = self.ascii_map[cu as usize];
                if idx != 0xFF {
                    return Some(idx as usize);
                }
                return None;
            }
        }
        // Keywords are interned and equal only to themselves (#271): an
        // address compare per key instead of `Value::eq`'s dispatch.
        if let Value::Keyword(kw) = k {
            return (self.keys.iter())
                .position(|ek| matches!(ek, Value::Keyword(e) if e.ptr_eq(kw)));
        }
        self.keys.iter().position(|ek| ek == k)
    }
}

/// A small map's entries: the shared shape plus this version's
/// values (parallel to `shape.keys`). Boxed slices, not vecs: no
/// capacity field (a 3-pair map buckets at 256B instead of 288B)
/// and no growth slack ever. Inline in the map's data (#271: one
/// allocation fewer per map version than behind its own `Arc`).
#[derive(Clone, Debug)]
struct SmallMap {
    shape: Arc<MapShape>,
    vals: Box<[Value]>,
}

/// Interned shapes by key-set. Entries are `Weak`: a shape dies
/// with its last map, and the registry drops the dead entry at
/// the next cleanup — a fresh-keys loop (ever-changing key
/// sets) retains at most ~1/8th of the table plus
/// [`SHAPE_CLEANUP_EVERY`] dead entries, never the shapes
/// themselves.
struct ShapeRegistry {
    map: HashMap<Vec<Value>, Weak<MapShape>, FxBuildHasher>,
    since_cleanup: usize,
    /// The last exact shapes interned (MRU slots 0 and 1): runs of
    /// same-shape maps or alternating shapes skip the hash probe.
    /// Kept alive here, so their table entries never go stale.
    last: [Option<Arc<MapShape>>; 2],
}

#[inline]
fn is_exact_key(k: &Value) -> bool {
    matches!(
        k,
        Value::Keyword(_)
            | Value::Char(_)
            | Value::Str(_)
            | Value::Symbol(_)
            | Value::Bool(_)
            | Value::Nil
    )
}

/// Dead-entry cleanup cadence: a cleanup scans the whole table,
/// so past this many misses it additionally waits until the
/// scan amortizes to [`SHAPE_CLEANUP_AMORT`] visits per miss (a
/// fixed cadence re-scans a big live table per handful of
/// misses — 109M entry-visits for one 150k-char suffix-tree
/// input, #235). Small enough that tests reach it (coverage),
/// large enough that cleanup never dominates.
const SHAPE_CLEANUP_EVERY: usize = 16;
/// Amortized retain-scan visits per miss past the
/// [`SHAPE_CLEANUP_EVERY`] floor (see above).
const SHAPE_CLEANUP_AMORT: usize = 8;
/// Maximum entries in the thread-local shape registry map to prevent
/// unbounded memory growth on high-cardinality ad-hoc keys.
const SHAPE_REGISTRY_CAP: usize = 512;

impl ShapeRegistry {
    fn insert_shape(&mut self, keys: &[Value], shape: &Arc<MapShape>) {
        if self.map.len() >= SHAPE_REGISTRY_CAP {
            self.map.retain(|_, weak| weak.strong_count() > 0);
        }
        if self.map.len() < SHAPE_REGISTRY_CAP {
            self.map.insert(keys.to_vec(), Arc::downgrade(shape));
        }
        self.since_cleanup += 1;
        if self.since_cleanup >= SHAPE_CLEANUP_EVERY
            && self.since_cleanup.saturating_mul(SHAPE_CLEANUP_AMORT) >= self.map.len()
        {
            self.map.retain(|_, weak| weak.strong_count() > 0);
            self.since_cleanup = 0;
        }
    }
}

thread_local! {
    static SHAPE_REGISTRY: RefCell<ShapeRegistry> = RefCell::new(ShapeRegistry {
        map: HashMap::default(),
        since_cleanup: 0,
        last: [None, None],
    });
}

/// Intern a key-set: the canonical [`MapShape`] for these keys
/// (order-sensitive — `[:a :b]` and `[:b :a]` are different
/// shapes, matching iteration order). Hits clone one `Arc` and
/// allocate nothing; misses allocate one shape. Never
/// re-enters: hashing/cloning keys interns nothing.
fn intern_shape(keys: &[Value]) -> Arc<MapShape> {
    SHAPE_REGISTRY.with(|reg| {
        let mut reg = reg.borrow_mut();
        // Exact keys compare unambiguously (no cross-type `=`), so
        // the shortcut agrees with the table's own `Eq`.
        let exact = keys.iter().all(is_exact_key);
        if exact {
            if let Some(l0) = reg.last[0].as_ref().filter(|l| *l.keys == *keys) {
                return Arc::clone(l0);
            }
            if let Some(l1) = reg.last[1].as_ref().filter(|l| *l.keys == *keys) {
                return Arc::clone(l1);
            }
        }
        if let Some(shape) = reg.map.get(keys).and_then(Weak::upgrade) {
            if exact {
                reg.last[1] = reg.last[0].take();
                reg.last[0] = Some(Arc::clone(&shape));
            }
            return shape;
        }
        let shape = Arc::new(MapShape {
            keys: keys.to_vec().into_boxed_slice(),
            ascii_map: build_ascii_map(keys),
            has_big_int_keys: keys
                .iter()
                .any(|k| matches!(k, Value::BigInt(_) | Value::BigInteger(_))),
        });
        reg.insert_shape(keys, &shape);
        if exact {
            reg.last[1] = reg.last[0].take();
            reg.last[0] = Some(Arc::clone(&shape));
        }
        shape
    })
}

/// Shared empty shape (every empty shaped map's entries).
static EMPTY_MAP_SHAPE: LazyLock<Arc<MapShape>> = LazyLock::new(|| intern_shape(&[]));

/// Empty entries (the shared empty shape, no values: no allocation —
/// the map's identity is its data's address, see [`Map::id`]).
fn fresh_empty_small() -> SmallMap {
    SmallMap {
        shape: EMPTY_MAP_SHAPE.clone(),
        vals: Box::default(),
    }
}

/// A map's fields, behind the handle's `Arc` so a `Map` (inline
/// in every `Value`) stays one word (#254: `Value` shrinks to 32
/// bytes). Writers go through `Arc::make_mut` on the handle data
/// first — a shared handle clones the fields (a shaped map's
/// entries included, a hash map's trie `Arc` shared) before any
/// in-place update; [`Map::set_small`] replaces the entries
/// without copying the old ones.
#[derive(Clone)]
struct MapData {
    /// The entries when `small` is `None` (`None` for shaped maps
    /// — no dummy trie. Behind an `Arc` whose address is the
    /// allocation identity (see [`Map::id`]).
    inner: Option<Arc<RpdsMap>>,
    /// `Some` = shaped authoritative (an interned key-set plus
    /// this version's values, at most [`INLINE_MAP_MAX`] entries
    /// in iteration order); `None` = `inner` authoritative (plus
    /// `keys` when sorted). Inline: a shaped map's identity is its
    /// data's address (see [`Map::id`]).
    small: Option<SmallMap>,
    /// Comparator-order sidecar for sorted maps when `small` is
    /// `None` (`None` otherwise — plain hash-maps iterate the
    /// trie, never this). Writers maintain it only when sorted.
    keys: Option<Arc<RpdsVector>>,
    meta: Option<Arc<Map>>,
    hashy: bool,
    /// Boxed: sorted maps are rare (identity rides the entries
    /// `Arc` instead of a minted field, and the trie rides
    /// behind an `Arc` rather than inline).
    sort: Option<Box<SortSpec>>,
}

#[derive(Clone)]
pub struct Map(Arc<MapData>);

impl std::fmt::Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map")
            .field("inner", &self.0.inner)
            .field("small", &self.0.small)
            .field("keys", &self.0.keys)
            .field("meta", &self.0.meta)
            .field("hashy", &self.0.hashy)
            .field("sort", &self.0.sort)
            .finish()
    }
}

impl Map {
    /// Fast lookup for scalar keys (`Char`, `Keyword`, `Int`, `Bool`,
    /// `Nil`, `Str`) in inline-shaped maps without hashing or
    /// recursive [`Value::eq`]. Returns `None` for unsupported keys
    /// and representations so bitcode helpers can delegate to `rt_n_*`.
    #[inline]
    #[must_use]
    pub fn get_small_scalar(&self, k: &Value) -> Option<Option<&Value>> {
        if self.is_sorted() {
            return None;
        }
        let small = self.0.small.as_ref()?;
        if let Value::Keyword(kw) = k {
            let pos = small
                .shape
                .keys
                .iter()
                .position(|ek| matches!(ek, Value::Keyword(e) if e.ptr_eq(kw)));
            return Some(pos.and_then(|idx| small.vals.get(idx)));
        }
        if let Value::Char(c) = k {
            return small
                .shape
                .ascii_map
                .get(*c as usize)
                .map(|&idx| small.vals.get(idx as usize));
        }
        // Long keys can equal BigInt and BigInteger keys.
        if !is_small_scalar_key(k) || (matches!(k, Value::Int(_)) && small.shape.has_big_int_keys) {
            return None;
        }
        let pos = small
            .shape
            .keys
            .iter()
            .position(|ek| small_scalar_eq(ek, k));
        Some(pos.and_then(|idx| small.vals.get(idx)))
    }

    /// Entry count without panicking on missing invariants.
    #[inline]
    #[must_use]
    pub fn len_opt(&self) -> Option<usize> {
        if let Some(small) = &self.0.small {
            return Some(small.vals.len());
        }
        self.0.inner.as_deref().map(|t| t.size())
    }
}

#[inline]
fn is_small_scalar_key(k: &Value) -> bool {
    matches!(
        k,
        Value::Keyword(_) | Value::Int(_) | Value::Bool(_) | Value::Nil | Value::Str(_)
    )
}

#[inline]
fn small_scalar_eq(ek: &Value, k: &Value) -> bool {
    match (ek, k) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (Value::Str(x), Value::Str(y)) => **x == **y,
        _ => false,
    }
}

/// `m = k` for a nil, boolean, long, char or keyword key, decided
/// without the full `=`: only the same kind can equal such a key, except
/// that big integers can equal a long (`None`, like any other key kind).
#[inline(always)]
fn scalar_member_eq(m: &Value, k: &Value) -> Option<bool> {
    match (m, k) {
        (Value::Int(x), Value::Int(y)) => Some(x == y),
        (Value::Keyword(x), Value::Keyword(y)) => Some(x == y),
        (Value::Char(x), Value::Char(y)) => Some(x == y),
        (Value::Bool(x), Value::Bool(y)) => Some(x == y),
        (Value::Nil, Value::Nil) => Some(true),
        (Value::BigInt(_) | Value::BigInteger(_), Value::Int(_)) => None,
        (_, Value::Int(_) | Value::Keyword(_) | Value::Char(_) | Value::Bool(_) | Value::Nil) => {
            Some(false)
        }
        _ => None,
    }
}

/// The sort order of a sorted map or set (FT6): natural
/// (`Util.compare`, nil-first) or a custom comparator fn called as
/// `(cmp a b)` with `sort-by` number/boolean semantics.
#[derive(Clone, Debug)]
pub enum SortSpec {
    /// Natural key order.
    Natural,
    /// A custom comparator value (called per comparison).
    Custom(Box<Value>),
}

/// A persistent set of values (thread-safe rpds hash-trie set) with
/// optional metadata. See [`List`]. `members` is the insertion-order
/// sidecar (small JVM sets seq in insertion order); re-inserting a
/// member is a no-op, like `conj`. Like [`Map`]'s `keys`, it rides
/// behind an `Arc` so the 24-byte rpds header does not inflate
/// every `Value` (#221 R1). `sort` is the FT6 sorted-set
/// order (same contract as [`Map`]'s): `members` stays in
/// comparator order and compare-0 duplicates collapse.
/// Maximum member count kept inline in [`SetStorage::Small`] (mirrors
/// [`SMALL_MAP_MAX`]): avoids `rpds` trie + sidecar allocations and
/// `compact()` churn on small sets (#271 Slice 5q).
const SMALL_SET_MAX: usize = 8;

/// Hash-indexed persistent storage for sets past [`SMALL_SET_MAX`].
#[derive(Clone, Debug)]
struct LargeSet {
    /// Live members, each mapped to its slot in `members`.
    inner: RpdsIndex,
    /// Insertion order (comparator order when sorted). `disj` leaves a
    /// hole in its member's slot (O(log n), no shifting); the sidecar
    /// compacts once holes outnumber members (amortized O(1)).
    members: Arc<SetSlots>,
    /// How many slots are holes, and the first slot that can be live
    /// (every slot before it is a hole: `first`/`disj` pops stay O(1)).
    holes: usize,
    head: usize,
}

#[derive(Clone, Debug)]
enum SetStorage {
    Small(Vec<Value>),
    Large(Box<LargeSet>),
}

impl SetStorage {
    fn large_from_vec(members: Vec<Value>) -> Self {
        let (inner, slots) = slots_of(members);
        Self::Large(Box::new(LargeSet {
            inner,
            members: Arc::new(slots),
            holes: 0,
            head: 0,
        }))
    }
}

/// A set's fields, behind the handle's `Arc` so a `Set` (inline
/// in every `Value`) stays one word (#254: `Value` shrinks to 32
/// bytes). Writers go through `Arc::make_mut`, exactly like the
/// old struct copy.
#[derive(Clone)]
struct SetData {
    storage: SetStorage,
    meta: Option<Arc<Map>>,
    /// Boxed (like [`Map`]'s): sorted sets are rare.
    sort: Option<Box<SortSpec>>,
    id: u64,
}

#[derive(Clone)]
pub struct Set(Arc<SetData>);

impl std::fmt::Debug for Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Set")
            .field("inner", &self.0.storage)
            .field("meta", &self.0.meta)
            .field("sort", &self.0.sort)
            .field("id", &self.0.id)
            .finish()
    }
}

enum SetIter<'a> {
    Small(std::slice::Iter<'a, Value>),
    Large(std::iter::Flatten<crate::trie::Iter<'a, Option<Value>>>),
}

impl<'a> Iterator for SetIter<'a> {
    type Item = &'a Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SetIter::Small(it) => it.next(),
            SetIter::Large(it) => it.next(),
        }
    }
}

impl List {
    /// An empty list.
    #[must_use]
    pub fn new_sync() -> Self {
        Self(Arc::new(ListData {
            inner: RpdsList::new_sync(),
            meta: None,
            id: next_list_id(),
        }))
    }

    /// Build a list from items in order: `from_vec([a, b])` is `(a b)`.
    #[must_use]
    pub fn from_vec(items: Vec<Value>) -> Self {
        let mut list = Self::new_sync();
        for v in items.into_iter().rev() {
            list = list.push_front(v);
        }
        list
    }

    /// Prepend a value.
    #[must_use]
    pub fn push_front(self, v: Value) -> Self {
        Self(Arc::new(ListData {
            inner: self.0.inner.clone().push_front(v),
            meta: self.0.meta.clone(),
            id: next_list_id(),
        }))
    }

    /// The first value, if any.
    #[must_use]
    pub fn first(&self) -> Option<&Value> {
        self.0.inner.iter().next()
    }

    /// The list without its first value (`None` when empty;
    /// metadata is preserved — `rest` strips it separately).
    #[must_use]
    pub fn drop_first(&self) -> Option<Self> {
        self.0.inner.drop_first().map(|inner| {
            Self(Arc::new(ListData {
                inner,
                meta: self.0.meta.clone(),
                id: next_list_id(),
            }))
        })
    }

    /// Iterate front to back.
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.0.inner.iter()
    }

    /// The element count.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.inner.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.inner.len() == 0
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.0.meta.as_deref()
    }

    /// Return this list with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        let data = Arc::make_mut(&mut self.0);
        data.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news), so a new id.
        data.id = next_list_id();
        self
    }

    /// The allocation identity for `identical?` (S16): minted at
    /// every construction, preserved by `Clone`.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.0.id
    }
}

impl Vector {
    /// An empty vector (typed by its first item: `conj` chains over
    /// `Int`s or chars stay packed from the first push).
    #[must_use]
    pub fn new_sync() -> Self {
        Self::of(VectorInner::Ints(crate::trie::Packed::from_vec(Vec::new())))
    }

    /// The vector's data for writing (`Arc::make_mut`): a shared handle
    /// copies it first, except that a shared flat run past
    /// [`FLAT_MAX`](crate::trie::FLAT_MAX) moves into the trie (one
    /// pass over its items, then log-time updates: #227) instead of
    /// being copied whole on every update of a loop-threaded vector.
    fn data_mut(&mut self) -> &mut VectorData {
        if Arc::strong_count(&self.0) > 1 {
            let long = match &self.0.inner {
                VectorInner::Generic(t) => {
                    t.is_long_flat().then(|| VectorInner::Generic(t.to_trie()))
                }
                VectorInner::Ints(t) => t.is_long_flat().then(|| VectorInner::Ints(t.to_trie())),
                VectorInner::Doubles(t) => {
                    t.is_long_flat().then(|| VectorInner::Doubles(t.to_trie()))
                }
                VectorInner::Chars(t) => t.is_long_flat().then(|| VectorInner::Chars(t.to_trie())),
                VectorInner::Strs(_) => None,
            };
            if let Some(inner) = long {
                self.0 = Arc::new(VectorData {
                    inner,
                    meta: self.0.meta.clone(),
                    id: self.0.id,
                });
            }
        }
        Arc::make_mut(&mut self.0)
    }

    fn of(inner: VectorInner) -> Self {
        Self(Arc::new(VectorData {
            inner,
            meta: None,
            id: next_vector_id(),
        }))
    }

    /// Build a vector from items in order (packed when every item is
    /// an `Int`, every item a `Double`, or every item a char; empty
    /// vectors start typed for their first `conj`).
    #[must_use]
    pub fn from_vec(items: Vec<Value>) -> Self {
        if items.is_empty() {
            return Self::new_sync();
        }
        pack_typed_vec(&items)
            .unwrap_or_else(|| Self::of(VectorInner::Generic(crate::trie::Packed::flat(items))))
    }

    /// Build a packed `Int` vector from `i64`s in order (no scan — for
    /// callers that already know, like ranges and packed transients).
    #[must_use]
    pub fn from_ints(ints: Vec<i64>) -> Self {
        Self::of(VectorInner::Ints(crate::trie::Packed::flat(ints)))
    }

    /// Build a packed `Int` vector from `i64`s in order (no scan).
    #[must_use]
    pub fn from_flat_ints(ints: Vec<i64>) -> Self {
        Self::of(VectorInner::Ints(crate::trie::Packed::flat(ints)))
    }

    /// Build a packed `Double` vector from `f64`s in order (no scan).
    #[must_use]
    pub fn from_doubles(doubles: Vec<f64>) -> Self {
        Self::of(VectorInner::Doubles(crate::trie::Packed::flat(doubles)))
    }

    /// Build a packed `Double` vector from `f64`s in order (no scan).
    #[must_use]
    pub fn from_flat_doubles(doubles: Vec<f64>) -> Self {
        Self::of(VectorInner::Doubles(crate::trie::Packed::flat(doubles)))
    }

    /// Build a packed char vector from `char`s in order (no scan).
    #[must_use]
    pub fn from_chars(chars: Vec<char>) -> Self {
        Self::of(VectorInner::Chars(crate::trie::Packed::flat(chars)))
    }

    /// Build a packed char vector from `char`s in order (no scan).
    #[must_use]
    pub fn from_flat_chars(chars: Vec<char>) -> Self {
        Self::of(VectorInner::Chars(crate::trie::Packed::flat(chars)))
    }

    /// Build a packed string vector from [`PackedStrings`] zero-copy.
    #[must_use]
    pub fn from_packed_strs(strs: PackedStrings) -> Self {
        Self::of(VectorInner::Strs(Arc::new(strs)))
    }

    /// Whether this vector rides on packed storage (`i64`, `char`, or `str`).
    #[must_use]
    pub fn is_packed(&self) -> bool {
        !matches!(self.0.inner, VectorInner::Generic(_))
    }

    /// The packed items when typed (for transients that freeze back
    /// to the same type without re-scanning).
    pub(crate) fn typed_items(&self) -> Option<TypedItems> {
        match &self.0.inner {
            VectorInner::Generic(_) => None,
            VectorInner::Ints(t) => Some(TypedItems::Ints(t.iter().copied().collect())),
            VectorInner::Doubles(t) => Some(TypedItems::Doubles(t.iter().copied().collect())),
            VectorInner::Chars(t) => Some(TypedItems::Chars(t.iter().copied().collect())),
            VectorInner::Strs(ps) => Some(TypedItems::Strs((**ps).clone())),
        }
    }

    /// The packed items when typed, moving out of an owned vector
    /// without re-allocating where possible.
    pub(crate) fn into_typed_items(self) -> Result<TypedItems, Self> {
        let data = Arc::try_unwrap(self.0).unwrap_or_else(|arc| (*arc).clone());
        match data.inner {
            VectorInner::Generic(g) => Err(Self(Arc::new(VectorData {
                inner: VectorInner::Generic(g),
                meta: data.meta,
                id: data.id,
            }))),
            VectorInner::Ints(t) => Ok(TypedItems::Ints(t.into_vec())),
            VectorInner::Doubles(t) => Ok(TypedItems::Doubles(t.into_vec())),
            VectorInner::Chars(t) => Ok(TypedItems::Chars(t.into_vec())),
            VectorInner::Strs(ps) => Ok(TypedItems::Strs(Arc::unwrap_or_clone(ps))),
        }
    }

    /// Extract all items into a flat Vec<Value>, unwrapping unshared nodes.
    #[must_use]
    pub fn into_vec(self) -> Vec<Value> {
        let data = Arc::try_unwrap(self.0).unwrap_or_else(|arc| (*arc).clone());
        match data.inner {
            VectorInner::Generic(pvec) => pvec.into_vec(),
            VectorInner::Ints(t) => t.into_vec().into_iter().map(Value::Int).collect(),
            VectorInner::Doubles(t) => t.into_vec().into_iter().map(Value::Double).collect(),
            VectorInner::Chars(t) => t.into_vec().into_iter().map(Value::Char).collect(),
            VectorInner::Strs(ps) => ps.to_values(),
        }
    }

    /// If this vector is packed chars or strings, append directly to `out`
    /// without boxing items. Returns true if handled.
    pub fn push_to_string(&self, out: &mut String) -> bool {
        match &self.0.inner {
            VectorInner::Chars(packed) => {
                out.reserve(packed.len());
                for &c in packed.iter() {
                    out.push(c);
                }
                true
            }
            VectorInner::Strs(ps) => {
                ps.push_to(out);
                true
            }
            _ => false,
        }
    }

    /// An empty typed vector takes the type of a packable first item.
    fn retype_empty(&mut self, v: &Value) {
        if !self.is_packed() || !self.is_empty() {
            return;
        }
        let next = match (&self.0.inner, v) {
            (VectorInner::Ints(_) | VectorInner::Doubles(_), Value::Char(_)) => Some(
                VectorInner::Chars(crate::trie::Packed::from_vec(Vec::new())),
            ),
            (VectorInner::Chars(_) | VectorInner::Doubles(_), Value::Int(_)) => {
                Some(VectorInner::Ints(crate::trie::Packed::from_vec(Vec::new())))
            }
            (VectorInner::Ints(_) | VectorInner::Chars(_), Value::Double(_)) => Some(
                VectorInner::Doubles(crate::trie::Packed::from_vec(Vec::new())),
            ),
            _ => None,
        };
        if let Some(inner) = next {
            self.data_mut().inner = inner;
        }
    }

    /// Append a value.
    #[must_use]
    pub fn push_back(mut self, v: Value) -> Self {
        // Through `&mut`, so uniquely-owned `conj` chains mutate in
        // place while versions shared with other handles path-copy;
        // a foreign value converts typed storage once (to generic,
        // which stays generic).
        self.retype_empty(&v);
        let data = self.data_mut();
        let generic = push_inner(&mut data.inner, v);
        if let Some(g) = generic {
            data.inner = VectorInner::Generic(g);
        }
        data.id = next_vector_id();
        self
    }

    /// Replace the item at `index` (`None` when out of bounds).
    ///
    /// The shared-storage form, for callers holding only a `&Vector`
    /// (the subvector writers): clones the header, then applies
    /// [`set_nth_owned`](Self::set_nth_owned). Owners should call
    /// `set_nth_owned` directly.
    #[must_use]
    pub fn set_nth(&self, index: usize, v: Value) -> Option<Self> {
        self.clone().set_nth_owned(index, v)
    }

    /// `assoc` in place on a uniquely held flat vector (#271 lean
    /// owned path, inlined into AOT programs as bitcode, so it runs no
    /// drop glue and touches no runtime statics): writes `*v` at
    /// `index` and returns `true` only when this handle and its flat
    /// storage are unique, `index` is in bounds, and the value fits the
    /// packing without retyping (a long into longs, a double into
    /// doubles, anything over a scalar item of generic storage, which
    /// takes `*v`, leaving nil). Otherwise nothing changes and the
    /// general [`set_nth_owned`](Self::set_nth_owned) path applies. The
    /// identity stays: no other handle can observe the old version,
    /// like a uniquely owned map updated in place.
    #[inline]
    pub fn assoc_unique(&mut self, index: usize, v: &mut Value) -> bool {
        let Some(data) = Arc::get_mut(&mut self.0) else {
            return false;
        };
        match (&mut data.inner, &mut *v) {
            (VectorInner::Ints(p), Value::Int(n)) => put_unique(p, index, *n),
            (VectorInner::Doubles(p), Value::Double(d)) => put_unique(p, index, *d),
            (VectorInner::Generic(p), v) => {
                match p.unique_flat_mut().and_then(|xs| xs.get_mut(index)) {
                    Some(slot)
                        if matches!(
                            slot,
                            Value::Nil
                                | Value::Bool(_)
                                | Value::Int(_)
                                | Value::Double(_)
                                | Value::Char(_)
                        ) =>
                    {
                        // A scalar needs no drop (none runs here).
                        std::mem::forget(std::mem::replace(slot, std::mem::replace(v, Value::Nil)));
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// `conj` in place on a uniquely held, non-empty flat vector below
    /// [`FLAT_MAX`](crate::trie::FLAT_MAX) items, within its capacity (#271 lean owned path),
    /// under [`assoc_unique`](Self::assoc_unique)'s rules: `false`
    /// changes nothing and leaves the general [`push_back`](Self::push_back).
    #[inline(always)]
    pub fn conj_unique(&mut self, v: &mut Value) -> bool {
        let Some(data) = Arc::get_mut(&mut self.0) else {
            return false;
        };
        match (&mut data.inner, &mut *v) {
            (VectorInner::Ints(p), Value::Int(n)) => push_unique(p, *n),
            (VectorInner::Doubles(p), Value::Double(d)) => push_unique(p, *d),
            (VectorInner::Generic(p), v) => match pushable(p) {
                Some(xs) => {
                    xs.push(std::mem::replace(v, Value::Nil));
                    true
                }
                None => false,
            },
            _ => false,
        }
    }

    /// Replace the item at `index` in place (`false` when out of bounds).
    #[inline]
    pub fn set_nth_mut(&mut self, index: usize, v: Value) -> bool {
        if index >= self.len() {
            return false;
        }
        let data = self.data_mut();
        let generic = set_inner(&mut data.inner, index, v);
        if let Some(g) = generic {
            data.inner = VectorInner::Generic(g);
        }
        data.id = next_vector_id();
        true
    }

    /// Replace the item at `index` on an owned vector (`None` when
    /// out of bounds — `self` is consumed either way, so check
    /// [`len`](Self::len) first when the vector must survive).
    ///
    /// The #212 pattern (`mut self` + `set_mut`): the trie
    /// path-copies only nodes shared with *other* versions, so
    /// threading one accumulator through an `assoc` chain mutates in
    /// place after the first step. A foreign value converts typed
    /// storage once, like [`push_back`](Self::push_back).
    #[inline]
    #[must_use]
    pub fn set_nth_owned(mut self, index: usize, v: Value) -> Option<Self> {
        self.set_nth_mut(index, v).then_some(self)
    }

    /// Extract the item at `index` on an owned vector (`None` when out of
    /// bounds), replacing a generic slot with `Value::Nil` so nested
    /// collections keep a refcount of 1 across `assoc-in` / `update`.
    pub fn take_nth(&mut self, index: usize) -> Option<Value> {
        if let VectorInner::Generic(t) = &mut self.data_mut().inner {
            t.replace(index, Value::Nil)
        } else {
            self.get(index)
        }
    }

    /// The vector without its last item (`None` when empty).
    ///
    /// The shared-storage form, for callers holding only a `&Vector`:
    /// clones the header, then applies
    /// [`drop_last_owned`](Self::drop_last_owned). Owners should call
    /// `drop_last_owned` directly.
    #[must_use]
    pub fn drop_last(&self) -> Option<Self> {
        self.clone().drop_last_owned()
    }

    /// The owned vector without its last item (`None` when empty —
    /// `self` is consumed either way, so check
    /// [`is_empty`](Self::is_empty) first when the vector must survive).
    ///
    /// The #212 pattern (`mut self` + `drop_last_mut`): `pop` chains
    /// over uniquely-owned vectors mutate in place while versions
    /// shared with other handles path-copy.
    #[must_use]
    pub fn drop_last_owned(mut self) -> Option<Self> {
        let data = self.data_mut();
        let popped = match &mut data.inner {
            VectorInner::Generic(t) => t.pop(),
            VectorInner::Ints(t) => t.pop(),
            VectorInner::Doubles(t) => t.pop(),
            VectorInner::Chars(t) => t.pop(),
            VectorInner::Strs(ps) => {
                if Arc::strong_count(ps) > 1 && ps.len() > crate::trie::FLAT_MAX {
                    let mut g = crate::trie::Packed::from_vec(ps.to_values());
                    let ok = g.pop();
                    data.inner = VectorInner::Generic(g);
                    ok
                } else {
                    Arc::make_mut(ps).pop()
                }
            }
        };
        if !popped {
            return None;
        }
        data.id = next_vector_id();
        Some(self)
    }

    /// Iterate front to back (owned values; typed items box on read).
    pub fn iter(&self) -> VectorIter<'_> {
        VectorIter {
            inner: match &self.0.inner {
                VectorInner::Generic(t) => VectorIterInner::Generic(t.iter()),
                VectorInner::Ints(t) => VectorIterInner::Ints(t.iter()),
                VectorInner::Doubles(t) => VectorIterInner::Doubles(t.iter()),
                VectorInner::Chars(t) => VectorIterInner::Chars(t.iter()),
                VectorInner::Strs(ps) => VectorIterInner::Strs(ps.iter()),
            },
        }
    }

    /// Owned-value iteration from `start` (clamped) to the back: packed
    /// storage positions its cursor instead of walking the skipped
    /// items, so a window deep into a large vector iterates only its
    /// own items (#271 Slice 5m).
    #[must_use]
    pub fn iter_from(&self, start: usize) -> VectorIter<'_> {
        VectorIter {
            inner: match &self.0.inner {
                VectorInner::Generic(t) => VectorIterInner::Generic(t.iter_from(start)),
                VectorInner::Ints(t) => VectorIterInner::Ints(t.iter_from(start)),
                VectorInner::Doubles(t) => VectorIterInner::Doubles(t.iter_from(start)),
                VectorInner::Chars(t) => VectorIterInner::Chars(t.iter_from(start)),
                VectorInner::Strs(ps) => VectorIterInner::Strs(ps.iter_from(start)),
            },
        }
    }

    /// The item at `index` (`None` past the end; logarithmic — indexed
    /// callers must use this, not `iter().nth()`).
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        match &self.0.inner {
            VectorInner::Generic(t) => t.get(index).cloned(),
            VectorInner::Ints(t) => t.get(index).copied().map(Value::Int),
            VectorInner::Doubles(t) => t.get(index).copied().map(Value::Double),
            VectorInner::Chars(t) => t.get(index).copied().map(Value::Char),
            VectorInner::Strs(ps) => ps.get_str(index).map(|s| Value::Str(Arc::from(s))),
        }
    }

    /// The item at `index` read in place through `f` (typed storage
    /// passes a stack copy; `None` past the end): the clone-free
    /// read for callers that only inspect the item.
    #[inline(always)]
    pub fn read_at<R>(&self, index: usize, f: impl FnOnce(&Value) -> Option<R>) -> Option<R> {
        match &self.0.inner {
            VectorInner::Generic(t) => f(t.get(index)?),
            VectorInner::Ints(t) => f(&Value::Int(*t.get(index)?)),
            VectorInner::Doubles(t) => f(&Value::Double(*t.get(index)?)),
            VectorInner::Chars(t) => f(&Value::Char(*t.get(index)?)),
            VectorInner::Strs(ps) => f(&Value::Str(Arc::from(ps.get_str(index)?))),
        }
    }

    /// The item at `index` borrowed (`None` past the end, and
    /// `None` for typed storage, whose items box on read): the
    /// clone-free read for comparison-style callers that only
    /// inspect generic elements.
    #[inline]
    #[must_use]
    pub fn get_ref(&self, index: usize) -> Option<&Value> {
        match &self.0.inner {
            VectorInner::Generic(t) => t.get(index),
            _ => None,
        }
    }

    /// The items from `index` to the end of its 32-item chunk (the
    /// JVM `ChunkedSeq` chunk; empty past the end): borrowed from
    /// generic storage, boxed from typed.
    #[must_use]
    pub fn chunk_from(&self, index: usize) -> std::borrow::Cow<'_, [Value]> {
        use std::borrow::Cow;
        match &self.0.inner {
            VectorInner::Generic(t) => Cow::Borrowed(t.chunk_from(index)),
            VectorInner::Ints(t) => {
                Cow::Owned(t.chunk_from(index).iter().map(|n| Value::Int(*n)).collect())
            }
            VectorInner::Doubles(t) => Cow::Owned(
                t.chunk_from(index)
                    .iter()
                    .map(|d| Value::Double(*d))
                    .collect(),
            ),
            VectorInner::Chars(t) => Cow::Owned(
                t.chunk_from(index)
                    .iter()
                    .map(|c| Value::Char(*c))
                    .collect(),
            ),
            VectorInner::Strs(ps) => Cow::Owned(ps.chunk_from(index)),
        }
    }

    /// Whether the vector's packed representation guarantees no item is `nil`.
    #[inline]
    #[must_use]
    pub fn cannot_contain_nil(&self) -> bool {
        matches!(
            self.0.inner,
            VectorInner::Ints(_)
                | VectorInner::Doubles(_)
                | VectorInner::Chars(_)
                | VectorInner::Strs(_)
        )
    }

    /// Bulk-append `other[start..end]` when both sides share a flat packed
    /// representation (`Err(self)` otherwise so the caller falls back).
    pub fn try_extend_flat_window(
        mut self,
        other: &Vector,
        start: usize,
        end: usize,
    ) -> Result<Self, Self> {
        if start == end {
            return Ok(self);
        }
        let data = self.data_mut();
        let appended = match (&mut data.inner, &other.0.inner) {
            (VectorInner::Ints(dst), VectorInner::Ints(src)) => {
                dst.extend_flat_window(src, start, end)
            }
            (VectorInner::Doubles(dst), VectorInner::Doubles(src)) => {
                dst.extend_flat_window(src, start, end)
            }
            (VectorInner::Generic(dst), VectorInner::Generic(src)) => {
                dst.extend_flat_window(src, start, end)
            }
            _ => false,
        };
        if appended {
            data.id = next_vector_id();
            Ok(self)
        } else {
            Err(self)
        }
    }

    /// The element count.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        vector_inner_len(&self.0.inner)
    }

    /// Whether the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.0.meta.as_deref()
    }

    /// Return this vector with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        let data = self.data_mut();
        data.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news), so a new id.
        data.id = next_vector_id();
        self
    }

    /// The allocation identity for `identical?` (S16): minted at
    /// every construction, preserved by `Clone`.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.0.id
    }
}

/// The item count of a vector storage.
#[inline]
fn vector_inner_len(inner: &VectorInner) -> usize {
    match inner {
        VectorInner::Generic(t) => t.len(),
        VectorInner::Ints(t) => t.len(),
        VectorInner::Doubles(t) => t.len(),
        VectorInner::Chars(t) => t.len(),
        VectorInner::Strs(ps) => ps.len(),
    }
}

fn pack_typed_vec(items: &[Value]) -> Option<Vector> {
    match items.first() {
        Some(Value::Int(_)) => pack_slice(items, |v| match v {
            Value::Int(n) => Some(*n),
            _ => None,
        })
        .map(Vector::from_ints),
        Some(Value::Double(_)) => pack_slice(items, |v| match v {
            Value::Double(d) => Some(*d),
            _ => None,
        })
        .map(Vector::from_doubles),
        Some(Value::Char(_)) => pack_slice(items, |v| match v {
            Value::Char(c) => Some(*c),
            _ => None,
        })
        .map(Vector::from_chars),
        _ => None,
    }
}

fn pack_slice<T>(items: &[Value], extract: impl Fn(&Value) -> Option<T>) -> Option<Vec<T>> {
    items.iter().map(extract).collect()
}

fn push_packed_str(ps: &mut Arc<PackedStrings>, s: Arc<str>) -> Option<crate::trie::Packed<Value>> {
    let shared_big = Arc::strong_count(ps) > 1 && ps.len() > crate::trie::FLAT_MAX;
    if !shared_big && Arc::make_mut(ps).try_push(&s) {
        return None;
    }
    let mut g = crate::trie::Packed::from_vec(ps.to_values());
    g.push(Value::Str(s));
    Some(g)
}

/// Write `x` at `i` of a uniquely held flat run (see
/// [`Vector::assoc_unique`]; `T` is a scalar, so nothing drops).
#[inline]
fn put_unique<T: Clone + Copy>(p: &mut crate::trie::Packed<T>, i: usize, x: T) -> bool {
    match p.unique_flat_mut().and_then(|xs| xs.get_mut(i)) {
        Some(slot) => {
            *slot = x;
            true
        }
        None => false,
    }
}

/// Append `x` to a uniquely held, non-empty flat run below
/// [`FLAT_MAX`](crate::trie::FLAT_MAX) (see [`Vector::conj_unique`]).
#[inline(always)]
fn push_unique<T: Clone + Copy>(p: &mut crate::trie::Packed<T>, x: T) -> bool {
    match pushable(p) {
        Some(xs) => {
            xs.push(x);
            true
        }
        None => false,
    }
}

/// The run's items when [`Vector::conj_unique`] may append in place:
/// within capacity too, so the append never grows the buffer (the
/// general path grows it, keeping appends amortized O(1)) and the
/// inlined bitcode carries no reallocation call.
#[inline(always)]
fn pushable<T: Clone>(p: &mut crate::trie::Packed<T>) -> Option<&mut Vec<T>> {
    p.unique_flat_mut()
        .filter(|xs| !xs.is_empty() && xs.len() < crate::trie::FLAT_MAX && xs.len() < xs.capacity())
}

fn push_inner(inner: &mut VectorInner, v: Value) -> Option<crate::trie::Packed<Value>> {
    match (inner, v) {
        (VectorInner::Generic(t), v) => {
            t.push(v);
            None
        }
        (VectorInner::Ints(t), Value::Int(n)) => {
            t.push(n);
            None
        }
        (VectorInner::Doubles(t), Value::Double(d)) => {
            t.push(d);
            None
        }
        (VectorInner::Chars(t), Value::Char(c)) => {
            t.push(c);
            None
        }
        (VectorInner::Strs(ps), Value::Str(s)) => push_packed_str(ps, s),
        (VectorInner::Ints(t), v) => Some(boxed(t, Value::Int, |g| g.push(v))),
        (VectorInner::Doubles(t), v) => Some(boxed(t, Value::Double, |g| g.push(v))),
        (VectorInner::Chars(t), v) => Some(boxed(t, Value::Char, |g| g.push(v))),
        (VectorInner::Strs(ps), v) => {
            let mut g = crate::trie::Packed::from_vec(ps.to_values());
            g.push(v);
            Some(g)
        }
    }
}

fn set_inner(
    inner: &mut VectorInner,
    index: usize,
    v: Value,
) -> Option<crate::trie::Packed<Value>> {
    match (inner, v) {
        (VectorInner::Generic(t), v) => {
            t.set(index, v);
            None
        }
        (VectorInner::Ints(t), Value::Int(n)) => {
            t.set(index, n);
            None
        }
        (VectorInner::Doubles(t), Value::Double(d)) => {
            t.set(index, d);
            None
        }
        (VectorInner::Chars(t), Value::Char(c)) => {
            t.set(index, c);
            None
        }
        (VectorInner::Ints(t), v) => Some(boxed(t, Value::Int, |g| {
            g.set(index, v);
        })),
        (VectorInner::Doubles(t), v) => Some(boxed(t, Value::Double, |g| {
            g.set(index, v);
        })),
        (VectorInner::Chars(t), v) => Some(boxed(t, Value::Char, |g| {
            g.set(index, v);
        })),
        (VectorInner::Strs(ps), v) => {
            let mut g = crate::trie::Packed::from_vec(ps.to_values());
            g.set(index, v);
            Some(g)
        }
    }
}

/// Typed storage as a generic trie (`box_item` boxes each item),
/// then `then` applied: the one-time conversion when a foreign value
/// meets packed storage.
fn boxed<T: Copy>(
    t: &crate::trie::Packed<T>,
    box_item: fn(T) -> Value,
    then: impl FnOnce(&mut crate::trie::Packed<Value>),
) -> crate::trie::Packed<Value> {
    let mut g = crate::trie::Packed::from_vec(t.iter().map(|x| box_item(*x)).collect());
    then(&mut g);
    g
}

/// A typed vector's items, copied out for a transient.
pub(crate) enum TypedItems {
    Ints(Vec<i64>),
    Doubles(Vec<f64>),
    Chars(Vec<char>),
    Strs(PackedStrings),
}

/// Entries in iteration order (trie order for plain hash-maps,
/// insertion order for shaped maps, comparator order for sorted
/// ones) — one concrete type over all representations (`TrieIter`
/// boxed so `MapIter` stays 40 bytes on the shaped hot path).
pub struct MapIter<'a> {
    inner: MapIterInner<'a>,
}

enum MapIterInner<'a> {
    Shaped {
        keys: std::slice::Iter<'a, Value>,
        vals: std::slice::Iter<'a, Value>,
    },
    Ordered(OrderedIter<'a>),
    Trie(Box<TrieIter<'a>>),
}

/// Sorted hash-map iteration: position over the comparator-order
/// sidecar, resolving each key against the trie (every sorted
/// writer keeps both in sync, so every key resolves).
struct OrderedIter<'a> {
    keys: &'a RpdsVector,
    map: &'a RpdsMap,
    pos: usize,
}

impl<'a> Iterator for OrderedIter<'a> {
    type Item = (&'a Value, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.keys.len() {
            return None;
        }
        let k = self.keys.get(self.pos)?;
        self.pos += 1;
        // Every sorted writer keeps the sidecar and the trie in
        // sync, so every iterated key resolves.
        let v = self.map.get(k).expect("keys sidecar and trie stay in sync");
        Some((k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.keys.len().saturating_sub(self.pos)))
    }
}

/// Plain hash-map iteration: the trie's own order (deterministic
/// per content, like the JVM's hash order — never insertion).
type TrieIter<'a> = rpds::map::hash_trie_map::Iter<'a, Value, Value, archery::ArcTK>;

impl<'a> Iterator for MapIter<'a> {
    type Item = (&'a Value, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            // Keys and vals stay parallel by construction, so
            // both sides end together; `zip` keeps the tail
            // from over-running on any divergence.
            MapIterInner::Shaped { keys, vals } => keys.next().zip(vals.next()),
            MapIterInner::Ordered(it) => it.next(),
            MapIterInner::Trie(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            MapIterInner::Shaped { keys, vals } => {
                let (lk, uk) = keys.size_hint();
                let (lv, uv) = vals.size_hint();
                (lk.min(lv), uk.zip(uv).map(|(a, b)| a.min(b)))
            }
            MapIterInner::Ordered(it) => it.size_hint(),
            MapIterInner::Trie(it) => it.size_hint(),
        }
    }
}

impl Map {
    /// Build a map from key/value pairs, duplicate-checking them with
    /// the JVM literal rules: `Err` is the offending key (the FIRST
    /// of the pair for small maps, the NEW key otherwise). Literals
    /// take their class from `canBePAM` (`RT.map`/`mapUniqueKeys`).
    /// The hash trie (hash arm only — shaped maps hold `None`).
    fn trie(&self) -> &RpdsMap {
        self.0.inner.as_deref().expect("hash map holds a trie")
    }

    /// The hash trie, mutably (hash arm only). `make_mut`: in
    /// place when unique; a header clone plus a path copy when
    /// shared (exactly the old clone-then-write) — and fresh
    /// identity exactly when shared (see [`Map::id`]).
    fn trie_mut(&mut self) -> &mut RpdsMap {
        Arc::make_mut(
            self.data_mut()
                .inner
                .as_mut()
                .expect("hash map holds a trie"),
        )
    }

    /// The handle fields, mutably: shared data clones first (a
    /// shaped map's entries copy with it — a fresh identity, see
    /// [`Map::id`]; a hash map's trie `Arc` stays shared).
    #[inline]
    fn data_mut(&mut self) -> &mut MapData {
        Arc::make_mut(&mut self.0)
    }

    /// This map's data with its shaped entries replaced by `small`: a
    /// shared map copies its other fields only (never the old entries),
    /// a uniquely held one updates in place.
    fn set_small(&mut self, small: SmallMap) -> &mut MapData {
        if Arc::get_mut(&mut self.0).is_none() {
            let d = &*self.0;
            self.0 = Arc::new(MapData {
                inner: None,
                small: None,
                keys: d.keys.clone(),
                meta: d.meta.clone(),
                hashy: d.hashy,
                sort: d.sort.clone(),
            });
        }
        let data = Arc::get_mut(&mut self.0).expect("unique after the copy");
        data.small = Some(small);
        data
    }

    pub fn from_kvs_checked(pairs: Vec<(Value, Value)>) -> Result<Self, Value> {
        let keys: Vec<&Value> = pairs.iter().map(|(k, _)| k).collect();
        if let Some(dup) = find_map_dup(&keys) {
            return Err(dup.clone());
        }
        let hashy = !can_be_small_map(&keys);
        let mut keys_owned = Vec::with_capacity(pairs.len());
        let mut vals = Vec::with_capacity(pairs.len());
        for (k, v) in pairs {
            keys_owned.push(k);
            vals.push(v);
        }
        Ok(Self::from_parts(keys_owned, vals, hashy))
    }

    /// Build a map from evaluated pairs, last-wins (map literals
    /// with computed keys): the first occurrence sets the position,
    /// the last sets the value — exactly the `new_map` + `insert`
    /// loop, but with one intern and one values alloc instead of a
    /// transient version per pair (literals in hot loops shed two
    /// allocs and two registry lookups per pair).
    #[must_use]
    pub fn from_kvs_last_wins(pairs: Vec<(Value, Value)>) -> Map {
        let mut keys: Vec<Value> = Vec::with_capacity(pairs.len());
        let mut vals: Vec<Value> = Vec::with_capacity(pairs.len());
        // The class simulates the incremental `insert` promotion
        // rule pair by pair (overwrites don't advance it).
        let mut hashy = false;
        for (k, v) in pairs {
            if let Some(idx) = keys.iter().position(|ek| ek == &k) {
                vals[idx] = v;
                continue;
            }
            let distinct = keys.len();
            if !hashy && (distinct >= 64 || (distinct >= 8 && !matches!(k, Value::Keyword(_)))) {
                hashy = true;
            }
            keys.push(k);
            vals.push(v);
        }
        Self::from_parts(keys, vals, hashy)
    }

    /// A small map over distinct keyword `keys` (in order) and their
    /// `vals`: no dedupe scan, and a run of same-keyed maps reuses
    /// the last interned shape without probing (every fresh var's
    /// `{:ns … :name …}`). Callers guarantee the keys are distinct
    /// keywords, at most 8 (array-classed, like the JVM's).
    #[must_use]
    pub fn from_distinct_keywords(keys: &[Value], vals: Vec<Value>) -> Self {
        debug_assert!(keys.len() == vals.len() && keys.len() <= 8 && !keys.is_empty());
        debug_assert!(keys.iter().all(|k| matches!(k, Value::Keyword(_))));
        Self(Arc::new(MapData {
            inner: None,
            small: Some(SmallMap {
                shape: intern_shape(keys),
                vals: vals.into_boxed_slice(),
            }),
            keys: None,
            meta: None,
            hashy: false,
            sort: None,
        }))
    }

    /// A map with `template`'s keys over `vals`, paired in the template's
    /// iteration order (#271: keyword map literals). A shaped, unsorted
    /// template lends its interned shape: no shape probe, dedupe scan or
    /// pair vector, just the values. Any other template (or a count
    /// mismatch) builds through [`Map::from_kvs_last_wins`].
    #[must_use]
    pub fn with_vals(template: &Map, vals: Vec<Value>) -> Self {
        if let Some(small) = template
            .0
            .small
            .as_ref()
            .filter(|s| s.shape.keys.len() == vals.len() && template.0.sort.is_none())
        {
            return Self(Arc::new(MapData {
                inner: None,
                small: Some(SmallMap {
                    shape: Arc::clone(&small.shape),
                    vals: vals.into_boxed_slice(),
                }),
                keys: None,
                meta: None,
                hashy: template.0.hashy,
                sort: None,
            }));
        }
        Map::from_kvs_last_wins(template.iter().map(|(k, _)| k.clone()).zip(vals).collect())
    }

    /// Build from deduped parallel keys/vals (big maps take the
    /// hash trie directly; small ones intern one shape).
    pub(crate) fn from_parts(keys: Vec<Value>, vals: Vec<Value>, hashy: bool) -> Self {
        if keys.len() > INLINE_MAP_MAX {
            let mut inner = RpdsMap::new_with_hasher_and_ptr_kind(FxBuildHasher::default());
            for (k, v) in keys.into_iter().zip(vals) {
                inner.insert_mut(k, v);
            }
            return Self(Arc::new(MapData {
                inner: Some(Arc::new(inner)),
                small: None,
                // Plain (never sorted here): iteration is the
                // trie's own order, so no sidecar.
                keys: None,
                meta: None,
                hashy,
                sort: None,
            }));
        }
        if keys.is_empty() {
            return new_map().with_hashy(hashy);
        }
        Self(Arc::new(MapData {
            inner: None,
            small: Some(SmallMap {
                shape: intern_shape(&keys),
                vals: vals.into_boxed_slice(),
            }),
            keys: None,
            meta: None,
            hashy,
            sort: None,
        }))
    }

    /// Assoc a key/value pair. A new key appends to the insertion
    /// order; an existing key keeps its position (and key object).
    /// Promotion follows `PersistentArrayMap.assoc`: past 8 pairs a
    /// non-keyword key promotes, past 64 pairs anything does. Past
    /// [`INLINE_MAP_MAX`] pairs the representation converts to the
    /// hash trie (the class flag stays orthogonal).
    #[must_use]
    pub fn insert(mut self, k: Value, v: Value) -> Self {
        self.insert_mut(k, v);
        self
    }

    /// Assoc a key/value pair in place (the [`insert`](Self::insert)
    /// logic through `&mut`, for accumulators that cannot move —
    /// the `frequencies`/`group-by` walk closures — so they skip
    /// the `mem::replace` dummy map per item).
    ///
    /// Uniquely-owned storage mutates in place (same
    /// [`id`](Self::id) — unobservable: the in-place arms run only
    /// with no live alias, and the consumed input is dead); shared
    /// storage copies into fresh allocations, so live aliases keep
    /// their versions and identities.
    pub fn insert_mut(&mut self, k: Value, v: Value) {
        if self.0.small.is_some() {
            return self.insert_shaped_mut(k, v);
        }
        // Owned `self`: mutate the trie in place (same rationale as
        // `Vector::push_back` — rpds still path-copies nodes shared
        // with *other* versions, so persistence is preserved).
        // `trie_mut` mints fresh identity exactly when shared (see
        // [`Map::id`]). For an unordered trie, whether the size
        // grew also tells us whether the insert can promote its class.
        if self.0.keys.is_none() {
            return self.insert_unordered_trie(k, v);
        }
        if self.trie().get(&k).is_some() {
            self.trie_mut().insert_mut(k, v);
            return;
        }
        let size = self.trie().size();
        let promote =
            !self.0.hashy && (size >= 64 || (size >= 8 && !matches!(k, Value::Keyword(_))));
        // Sorted maps keep the comparator-order sidecar alongside.
        // The Clojure sorted-map path normally computes the position
        // first. Direct callers can append through this method.
        let keys = self
            .data_mut()
            .keys
            .as_mut()
            .expect("ordered trie has keys");
        Arc::make_mut(keys).push_back_mut(k.clone());
        self.trie_mut().insert_mut(k, v);
        self.data_mut().hashy |= promote;
    }

    fn insert_unordered_trie(&mut self, k: Value, v: Value) {
        let size = self.trie().size();
        let promote =
            !self.0.hashy && (size >= 64 || (size >= 8 && !matches!(k, Value::Keyword(_))));
        let trie = self.trie_mut();
        trie.insert_mut(k, v);
        if promote && trie.size() > size {
            self.data_mut().hashy = true;
        }
    }

    /// The shaped arm of [`Map::insert_mut`] (linear scan — the `==`
    /// relation agrees with the hash trie's, so both arms find
    /// exactly the same keys). Uniquely-owned same-key assocs
    /// write the value in place (no allocs); shared same-key
    /// assocs allocate only the values vec (the shape `Arc` is
    /// shared); new keys intern the extended key-set (hits
    /// allocate nothing past values).
    fn insert_shaped_mut(&mut self, k: Value, v: Value) {
        let small = self.0.small.as_ref().expect("shaped checked above");
        if let Some(idx) = small.shape.find_key(&k) {
            // A shared map copies its data (values included) once; a
            // uniquely held one writes in place, its identity (see
            // [`Map::id`]) unobservable: no live alias.
            let data = self.data_mut();
            data.small.as_mut().expect("shaped checked above").vals[idx] = v;
            return;
        }
        let size = small.vals.len();
        let promote = !self.0.hashy && (size >= 64 || (size >= 8 && !matches!(k, Value::Keyword(_))));
        if size < INLINE_MAP_MAX {
            // Exact-size: the new version is frozen, so no growth
            // slack (see [`SmallMap`]).
            let mut keys = Vec::with_capacity(size + 1);
            keys.extend(small.shape.keys.iter().cloned());
            keys.push(k);
            let mut vals = Vec::with_capacity(size + 1);
            vals.extend(small.vals.iter().cloned());
            vals.push(v);
            let next = SmallMap {
                shape: intern_shape(&keys),
                vals: vals.into_boxed_slice(),
            };
            self.set_small(next).hashy |= promote;
            return;
        }
        let data = self.data_mut();
        data.hashy |= promote;
        let small = data.small.take().expect("shaped checked above");
        self.promote_with_mut(&small, k, v);
    }

    /// Update an existing key's value in place (one lookup, one `Arc::make_mut`,
    /// and one `Arc::get_mut` on shaped maps), returning `true` when updated
    /// and `false` when the key is missing or `f` returns `Err(orig)`.
    pub fn update_existing_mut(
        &mut self,
        k: &Value,
        f: impl FnOnce(Value) -> Result<Value, Value>,
    ) -> bool {
        if let Some(found) = self.0.small.as_ref().map(|s| s.shape.find_key(k)) {
            let Some(idx) = found else {
                return false;
            };
            if let Some(data) = Arc::get_mut(&mut self.0) {
                let vals = &mut data.small.as_mut().expect("shaped checked above").vals;
                let old = std::mem::replace(&mut vals[idx], Value::Nil);
                match f(old) {
                    Ok(new) => {
                        vals[idx] = new;
                        return true;
                    }
                    Err(orig) => {
                        vals[idx] = orig;
                        return false;
                    }
                }
            }
            let current = self.0.small.as_ref().expect("shaped checked above").vals[idx].clone();
            let Ok(new) = f(current) else {
                return false;
            };
            self.data_mut().small.as_mut().expect("shaped checked above").vals[idx] = new;
            return true;
        }
        let Some(slot) = self.trie_mut().get_mut(k) else {
            return false;
        };
        match f(std::mem::replace(slot, Value::Nil)) {
            Ok(new) => {
                *slot = new;
                true
            }
            Err(original) => {
                *slot = original;
                false
            }
        }
    }

    /// Convert a shaped map to the hash trie, then insert the
    /// overflowing pair (iteration becomes the trie's own order,
    /// or the comparator order when sorted).
    fn promote_with_mut(&mut self, small: &SmallMap, k: Value, v: Value) {
        let mut inner = RpdsMap::new_with_hasher_and_ptr_kind(FxBuildHasher::default());
        // Sorted maps keep the comparator-order sidecar (entries
        // arrive pre-sorted); plain maps iterate the trie, so no
        // sidecar. In-repo sorted writes position through
        // `with_ordered_entries`, so the sorted arm only fires for
        // direct out-of-crate writes.
        let mut keys = RpdsVector::new_sync();
        for (ek, ev) in small.shape.keys.iter().zip(small.vals.iter()) {
            if self.0.sort.is_some() {
                keys.push_back_mut(ek.clone());
            }
            inner.insert_mut(ek.clone(), ev.clone());
        }
        if self.0.sort.is_some() {
            keys.push_back_mut(k.clone());
        }
        inner.insert_mut(k, v);
        let sorted = self.0.sort.is_some();
        let data = self.data_mut();
        data.inner = Some(Arc::new(inner));
        data.small = None;
        if sorted {
            data.keys = Some(Arc::new(keys));
        }
    }

    /// Remove a key (`dissoc` one key). A missing key is a no-op;
    /// otherwise the entries drop it (and the comparator-order
    /// sidecar when sorted). The class never demotes (a shrunk
    /// hash-map stays a hash-map); neither does the
    /// representation (a shrunk hash-map stays hashed). A hit on a
    /// shared map mints fresh identity (`dissoc` news on the JVM); a
    /// miss returns the map itself.
    #[must_use]
    pub fn without(mut self, k: &Value) -> Self {
        if let Some(small) = &self.0.small {
            let Some(idx) = small.shape.find_key(k) else {
                return self;
            };
            if small.vals.len() == 1 {
                self.set_small(fresh_empty_small());
                return self;
            }
            let mut keys = Vec::with_capacity(small.vals.len() - 1);
            keys.extend(
                small
                    .shape
                    .keys
                    .iter()
                    .enumerate()
                    .filter_map(|(i, ek)| if i == idx { None } else { Some(ek.clone()) }),
            );
            let mut vals = Vec::with_capacity(small.vals.len() - 1);
            vals.extend(
                small
                    .vals
                    .iter()
                    .enumerate()
                    .filter_map(|(i, ev)| if i == idx { None } else { Some(ev.clone()) }),
            );
            self.set_small(SmallMap {
                shape: intern_shape(&keys),
                vals: vals.into_boxed_slice(),
            });
            return self;
        }
        if self.trie().get(k).is_none() {
            return self;
        }
        // Unwrap-or-clone into a fresh `Arc`: uniquely-owned tries
        // move (no clone), shared ones header-clone — and the
        // fresh `Arc` mints identity either way (see [`Map::id`]).
        let inner = self.data_mut().inner.take().expect("hash map holds a trie");
        let mut trie = match Arc::try_unwrap(inner) {
            Ok(t) => t,
            Err(shared) => (*shared).clone(),
        };
        trie.remove_mut(k);
        self.data_mut().inner = Some(Arc::new(trie));
        if let Some(old_keys) = &self.0.keys {
            // Sorted `dissoc` positions through here: prune the
            // comparator-order sidecar alongside.
            let mut keys = RpdsVector::new_sync();
            for key in old_keys.iter() {
                if key != k {
                    keys.push_back_mut(key.clone());
                }
            }
            self.data_mut().keys = Some(Arc::new(keys));
        }
        self
    }

    /// Whether this reports `PersistentHashMap` (else array-map).
    #[must_use]
    pub fn is_hashy(&self) -> bool {
        self.0.hashy
    }

    /// Whether entries ride shaped (an interned key-set plus a
    /// flat values vec, no HAMT or sidecar allocs) rather than
    /// in the hash trie.
    #[must_use]
    pub fn is_shaped(&self) -> bool {
        self.0.small.is_some()
    }

    /// Force the reported class (`hash-map` forces hash,
    /// `array-map`/transient-array forces array).
    #[must_use]
    pub fn with_hashy(mut self, hashy: bool) -> Self {
        self.data_mut().hashy = hashy;
        self
    }

    /// Whether this is a sorted map (FT6 `sorted-map` family).
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        self.0.sort.is_some()
    }

    /// The sort order, if sorted.
    #[must_use]
    pub fn sort_spec(&self) -> Option<&SortSpec> {
        self.0.sort.as_deref()
    }

    /// Mark this map sorted (or clear the order), preserving entries.
    /// Callers maintain `keys` in comparator order alongside:
    /// attaching `Some` to a map whose sidecar isn't (a plain
    /// hash-map's dummy, an unsorted build) breaks iteration —
    /// build through [`Map::with_ordered_entries`] or attach
    /// before inserting.
    #[must_use]
    pub fn with_sort(mut self, sort: Option<SortSpec>) -> Self {
        Arc::make_mut(&mut self.0).sort = sort.map(Box::new);
        self
    }

    /// Keys in iteration order (insertion order for shaped maps,
    /// the trie's own order for plain hash-maps, comparator order
    /// when sorted). Owned: only sorted paths (already linear)
    /// snapshot the sidecar; the trie arm clones each key.
    #[must_use]
    pub fn ordered_keys(&self) -> Vec<Value> {
        match (&self.0.small, &self.0.keys) {
            (Some(small), _) => small.shape.keys.to_vec(),
            (None, Some(keys)) => keys.iter().cloned().collect(),
            (None, None) => self.trie().keys().cloned().collect(),
        }
    }

    /// How many keys a sorted map's binary search spans.
    #[must_use]
    pub fn sorted_len(&self) -> usize {
        match &self.0.small {
            Some(small) => small.vals.len(),
            None => self.sorted_keys().len(),
        }
    }

    /// A sorted map's key at sorted position `i` (0 = first).
    #[must_use]
    pub fn sorted_key(&self, i: usize) -> Option<&Value> {
        match &self.0.small {
            Some(small) => small.shape.keys.get(i),
            None => self.sorted_keys().get(i),
        }
    }

    /// A sorted map's comparator-order key sidecar (trie arm).
    fn sorted_keys(&self) -> &RpdsVector {
        self.0
            .keys
            .as_deref()
            .expect("a sorted map keeps its key order")
    }

    /// Assoc a new key at sorted position `pos` (the caller's binary
    /// search found no compare-0 key): shaped maps rebuild (at most
    /// [`INLINE_MAP_MAX`] entries); larger ones insert into the trie
    /// (O(log n)) and rebuild only the key sidecar.
    #[must_use]
    pub fn insert_sorted_at(mut self, pos: usize, k: Value, v: Value) -> Self {
        if self.0.small.is_some() {
            let mut entries: Vec<(Value, Value)> =
                self.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
            entries.insert(pos, (k, v));
            return self.with_ordered_entries(entries);
        }
        let keys = self.sorted_keys();
        let mut ordered = Vec::with_capacity(keys.len() + 1);
        ordered.extend(keys.iter().take(pos).cloned());
        ordered.push(k.clone());
        ordered.extend(keys.iter_from(pos).cloned());
        self.data_mut().keys = Some(Arc::new(RpdsVector::from_vec(ordered)));
        self.trie_mut().insert_mut(k, v);
        self
    }

    /// Rebuild from explicitly ordered entries, preserving metadata
    /// and the sort order (sorted `assoc` positions through here).
    /// The given order sticks for shaped and sorted maps; plain
    /// large maps iterate the trie's own order regardless.
    #[must_use]
    pub fn with_ordered_entries(self, entries: Vec<(Value, Value)>) -> Self {
        if entries.len() <= INLINE_MAP_MAX {
            if entries.is_empty() {
                return Self(Arc::new(MapData {
                    inner: None,
                    small: Some(fresh_empty_small()),
                    keys: None,
                    meta: self.0.meta.clone(),
                    hashy: self.0.hashy,
                    sort: self.0.sort.clone(),
                }));
            }
            // Exact-capacity: callers may hand over slack, and
            // frozen versions keep exact-size allocs (see
            // [`SmallMap`]).
            let mut keys = Vec::with_capacity(entries.len());
            let mut vals = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                keys.push(k);
                vals.push(v);
            }
            return Self(Arc::new(MapData {
                inner: None,
                small: Some(SmallMap {
                    shape: intern_shape(&keys),
                    vals: vals.into_boxed_slice(),
                }),
                keys: None,
                meta: self.0.meta.clone(),
                hashy: self.0.hashy,
                sort: self.0.sort.clone(),
            }));
        }
        let mut inner = RpdsMap::new_with_hasher_and_ptr_kind(FxBuildHasher::default());
        // Only sorted maps keep the given order in the sidecar
        // (plain maps iterate the trie, whatever order the
        // entries arrive in).
        let mut keys = RpdsVector::new_sync();
        for (k, v) in entries {
            if self.0.sort.is_some() {
                keys.push_back_mut(k.clone());
            }
            inner.insert_mut(k, v);
        }
        Self(Arc::new(MapData {
            inner: Some(Arc::new(inner)),
            small: None,
            meta: self.0.meta.clone(),
            keys: self.0.sort.is_some().then(|| Arc::new(keys)),
            hashy: self.0.hashy,
            sort: self.0.sort.clone(),
        }))
    }

    /// Look up a key.
    #[must_use]
    pub fn get(&self, k: &Value) -> Option<&Value> {
        match &self.0.small {
            Some(small) => small.shape.find_key(k).map(|idx| &small.vals[idx]),
            None => self.trie().get(k),
        }
    }

    /// Look up a key on an owned map, moving the value out of a uniquely-owned
    /// shaped map (replacing the slot with `Value::Nil`) so nested collections
    /// keep a refcount of 1 across `assoc-in` / `update`.
    pub fn take_value(&mut self, k: &Value) -> Option<Value> {
        if let Some(small) = &self.0.small {
            let idx = small.shape.find_key(k)?;
            return Some(match Arc::get_mut(&mut self.0) {
                Some(data) => std::mem::replace(
                    &mut data.small.as_mut().expect("shaped checked above").vals[idx],
                    Value::Nil,
                ),
                None => self.0.small.as_ref().expect("shaped checked above").vals[idx].clone(),
            });
        }
        self.trie().get(k).cloned()
    }

    /// Whether a key is present.
    #[must_use]
    pub fn contains_key(&self, k: &Value) -> bool {
        match &self.0.small {
            Some(small) => small.shape.find_key(k).is_some(),
            None => self.trie().get(k).is_some(),
        }
    }

    /// The entry count.
    #[must_use]
    pub fn size(&self) -> usize {
        match &self.0.small {
            Some(small) => small.vals.len(),
            None => self.trie().size(),
        }
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Iterate entries in iteration order: insertion order for
    /// shaped maps (the JVM small-map seq order), the trie's own
    /// order for plain hash-maps (deterministic per content, like
    /// the JVM's hash order — never insertion), comparator order
    /// when sorted.
    pub fn iter(&self) -> MapIter<'_> {
        MapIter {
            inner: match (&self.0.small, &self.0.keys) {
                (Some(small), _) => MapIterInner::Shaped {
                    keys: small.shape.keys.iter(),
                    vals: small.vals.iter(),
                },
                (None, Some(keys)) => MapIterInner::Ordered(OrderedIter {
                    keys,
                    map: self.trie(),
                    pos: 0,
                }),
                (None, None) => MapIterInner::Trie(Box::new(self.trie().iter())),
            },
        }
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.0.meta.as_deref()
    }

    /// Return this map with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        let data = self.data_mut();
        data.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news): a shared shaped
        // map's data just copied (its identity, see [`Map::id`]); a
        // hash map mints a fresh trie `Arc`.
        if data.small.is_none() {
            let inner = data.inner.take().expect("map has small or trie");
            data.inner = Some(Arc::new((*inner).clone()));
        }
        self
    }

    /// The allocation identity for `identical?` (S16): the data's
    /// `Arc` address (shaped: the entries live inline, #271) or the
    /// trie `Arc` address (hash).
    /// Writers mint a fresh `Arc` except uniquely-owned in-place
    /// updates (#227 — same address, unobservable: no live alias
    /// shares it), `Clone` shares it, and no-ops return `self` —
    /// so live maps compare exactly like minted ids, with no field
    /// and no counter. Not stable across processes (addresses);
    /// stable per allocation (like the JVM's `identityHashCode`,
    /// which is the only other consumer).
    #[must_use]
    pub fn id(&self) -> u64 {
        match &self.0.small {
            Some(_) => Arc::as_ptr(&self.0).addr() as u64,
            None => {
                Arc::as_ptr(self.0.inner.as_ref().expect("hash map holds a trie")).addr() as u64
            }
        }
    }
}

/// A persistent membership set (no order, no metadata, no
/// identity): O(1) clone, O(log n) insert even when shared — for
/// lazy-seq state that must survive per-step clones (`distinct`).
#[derive(Clone, Debug, Default)]
pub struct MemberSet(RpdsSet);

impl MemberSet {
    /// Insert `v`; `false` when it was already a member.
    pub fn insert(&mut self, v: Value) -> bool {
        if self.0.contains(&v) {
            return false;
        }
        self.0.insert_mut(v);
        true
    }
}

impl Set {
    /// Build a set from items, duplicate-checking them with the JVM
    /// literal rules: `Err` is the NEW (repeated) item.
    pub fn from_items_checked(items: Vec<Value>) -> Result<Self, Value> {
        if let Some(dup) = find_set_dup(&items) {
            return Err(dup.clone());
        }
        let storage = if items.len() <= SMALL_SET_MAX {
            SetStorage::Small(items)
        } else {
            SetStorage::large_from_vec(items)
        };
        Ok(Self(Arc::new(SetData {
            storage,
            meta: None,
            sort: None,
            id: next_set_id(),
        })))
    }

    /// Add a value. A new member appends to the insertion order;
    /// a present member is a no-op (keeps the original object).
    #[must_use]
    pub fn insert(mut self, v: Value) -> Self {
        if self.contains(&v) {
            return self;
        }
        let data = Arc::make_mut(&mut self.0);
        match &mut data.storage {
            SetStorage::Small(items) if items.len() < SMALL_SET_MAX => {
                items.push(v);
            }
            SetStorage::Small(items) => {
                let mut all = std::mem::take(items);
                all.push(v);
                data.storage = SetStorage::large_from_vec(all);
            }
            SetStorage::Large(ls) => {
                let slot = ls.members.len();
                Arc::make_mut(&mut ls.members).push_back_mut(Some(v.clone()));
                ls.inner.insert_mut(v, slot);
            }
        }
        data.id = next_set_id();
        self
    }

    /// Whether a value is present.
    #[inline]
    #[must_use]
    pub fn contains(&self, v: &Value) -> bool {
        match &self.0.storage {
            SetStorage::Small(items) => items.iter().any(|m| m == v),
            SetStorage::Large(ls) => ls.inner.contains_key(v),
        }
    }

    /// [`get`](Self::get) on a small unsorted set by a nil, boolean,
    /// long, char or keyword key, member by member without the full `=`
    /// (#271: inlined into compiled code). `Some(None)`: absent; `None`:
    /// a hashed or sorted set, another key kind, or a big-integer member
    /// that may equal a long key.
    #[inline]
    #[must_use]
    pub fn get_small_scalar(&self, k: &Value) -> Option<Option<&Value>> {
        let SetStorage::Small(items) = &self.0.storage else {
            return None;
        };
        if self.is_sorted() {
            return None;
        }
        for m in items {
            if scalar_member_eq(m, k)? {
                return Some(Some(m));
            }
        }
        Some(None)
    }

    /// The stored member equal to `v`, if present (`get` on a set
    /// returns the member, not the key).
    #[inline]
    #[must_use]
    pub fn get(&self, v: &Value) -> Option<&Value> {
        match &self.0.storage {
            SetStorage::Small(items) => items.iter().find(|m| *m == v),
            SetStorage::Large(ls) => ls.inner.get_key_value(v).map(|(member, _)| member),
        }
    }

    /// Remove a value (`disj` one item). A missing value is a no-op;
    /// otherwise small sets shift in place and large sets leave a hole
    /// in the insertion-order sidecar (preserving member order).
    #[must_use]
    pub fn without(mut self, v: &Value) -> Self {
        let slot = match &self.0.storage {
            SetStorage::Small(items) => match items.iter().position(|m| m == v) {
                Some(pos) => pos,
                None => return self,
            },
            SetStorage::Large(ls) => match ls.inner.get(v) {
                Some(_) if self.0.sort.is_some() => {
                    // Sorted sidecars keep no slots (positional inserts would
                    // shift them): find the member's sorted position.
                    let pos = self
                        .iter()
                        .position(|m| m == v)
                        .expect("a member sits in the sidecar");
                    return self.without_sorted_at(pos);
                }
                Some(&slot) => slot,
                None => return self,
            },
        };
        let data = Arc::make_mut(&mut self.0);
        match &mut data.storage {
            SetStorage::Small(items) => {
                items.remove(slot);
            }
            SetStorage::Large(ls) => ls.remove_unsorted_slot(v, slot),
        }
        data.id = next_set_id();
        self
    }

    /// The element count.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        match &self.0.storage {
            SetStorage::Small(items) => items.len(),
            SetStorage::Large(ls) => ls.inner.size(),
        }
    }

    /// Whether the set is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Iterate members in insertion order (the JVM small-set seq
    /// order; see the `members` sidecar).
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        match &self.0.storage {
            SetStorage::Small(items) => SetIter::Small(items.iter()),
            SetStorage::Large(ls) => SetIter::Large(ls.members.iter_from(ls.head).flatten()),
        }
    }

    /// Whether this is a sorted set (FT6 `sorted-set` family).
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        self.0.sort.is_some()
    }

    /// The sort order, if sorted.
    #[must_use]
    pub fn sort_spec(&self) -> Option<&SortSpec> {
        self.0.sort.as_deref()
    }

    /// Mark this set sorted (or clear the order), preserving members.
    /// Callers maintain `members` in comparator order alongside.
    #[must_use]
    pub fn with_sort(mut self, sort: Option<SortSpec>) -> Self {
        Arc::make_mut(&mut self.0).sort = sort.map(Box::new);
        self
    }

    /// Members in iteration order (insertion order, or comparator
    /// order when sorted). Owned: only sorted paths (already
    /// linear) snapshot the persistent sidecar.
    #[must_use]
    pub fn ordered_members(&self) -> Vec<Value> {
        self.iter().cloned().collect()
    }

    /// How many members a sorted set's binary search spans (its
    /// sidecar past `head`, where a sorted sidecar has no holes).
    #[must_use]
    pub fn sorted_len(&self) -> usize {
        match &self.0.storage {
            SetStorage::Small(items) => items.len(),
            SetStorage::Large(ls) => ls.members.len() - ls.head,
        }
    }

    /// A sorted set's member at sorted position `i` (0 = first).
    #[must_use]
    pub fn sorted_member(&self, i: usize) -> Option<&Value> {
        match &self.0.storage {
            SetStorage::Small(items) => items.get(i),
            SetStorage::Large(ls) => ls.members.get(ls.head + i).and_then(Option::as_ref),
        }
    }

    /// Insert `v` at sorted position `pos` (the caller's binary search
    /// found no compare-0 member): small sets insert directly into the
    /// flat slice; large sets update the index in O(log n) and rebuild
    /// only the sidecar.
    #[must_use]
    pub fn insert_sorted_at(mut self, pos: usize, v: Value) -> Self {
        let data = Arc::make_mut(&mut self.0);
        match &mut data.storage {
            SetStorage::Small(items) if items.len() < SMALL_SET_MAX => {
                items.insert(pos, v);
            }
            SetStorage::Small(items) => {
                let mut all = std::mem::take(items);
                all.insert(pos, v);
                data.storage = SetStorage::large_from_vec(all);
            }
            SetStorage::Large(ls) => {
                let start = ls.head;
                let mut members = Vec::with_capacity(ls.members.len() - start + 1);
                members.extend(ls.members.iter_from(start).take(pos).cloned());
                members.push(Some(v.clone()));
                members.extend(ls.members.iter_from(start + pos).cloned());
                ls.members = Arc::new(SetSlots::from_vec(members));
                ls.inner.insert_mut(v, 0);
                ls.holes = 0;
                ls.head = 0;
            }
        }
        data.id = next_set_id();
        self
    }

    /// Remove the member at sorted position `pos` (0 = first): small sets
    /// remove directly; large sets pop position 0 in O(1) and rebuild only
    /// the sidecar for interior removals.
    #[must_use]
    pub fn without_sorted_at(mut self, pos: usize) -> Self {
        let data = Arc::make_mut(&mut self.0);
        match &mut data.storage {
            SetStorage::Small(items) => {
                items.remove(pos);
            }
            SetStorage::Large(ls) => ls.remove_sorted_at(pos),
        }
        data.id = next_set_id();
        self
    }

    /// Rebuild from explicitly ordered members, preserving metadata
    /// and the sort order.
    #[must_use]
    pub fn with_ordered_members(self, members: Vec<Value>) -> Self {
        let storage = if members.len() <= SMALL_SET_MAX {
            SetStorage::Small(members)
        } else {
            SetStorage::large_from_vec(members)
        };
        Self(Arc::new(SetData {
            storage,
            meta: self.0.meta.clone(),
            sort: self.0.sort.clone(),
            id: next_set_id(),
        }))
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.0.meta.as_deref()
    }

    /// Return this set with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        let data = Arc::make_mut(&mut self.0);
        data.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news), so a new id.
        data.id = next_set_id();
        self
    }

    /// The allocation identity for `identical?` (S16): minted at
    /// every construction, preserved by `Clone`.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.0.id
    }
}

// Equality and hashing ignore metadata and the `identical?` id on
// all four newtypes.
impl PartialEq for List {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl Eq for List {}

impl Hash for List {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        for v in self.iter() {
            v.hash(state);
        }
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        vector_eq(self, other)
    }
}

impl Eq for Vector {}

#[inline]
fn packed_for_each<T: Clone>(p: &crate::trie::Packed<T>, mut f: impl FnMut(&T)) {
    if let Some(xs) = p.as_flat() {
        for x in xs {
            f(x);
        }
    } else {
        for x in p.iter() {
            f(x);
        }
    }
}

impl Hash for Vector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        match &self.0.inner {
            VectorInner::Generic(t) => packed_for_each(t, |v| v.hash(state)),
            VectorInner::Ints(t) => packed_for_each(t, |&i| {
                state.write_u8(2);
                state.write_i64(i);
            }),
            VectorInner::Doubles(t) => packed_for_each(t, |&d| {
                state.write_u8(3);
                state.write_u64(double_hash_bits(d));
            }),
            VectorInner::Chars(t) => packed_for_each(t, |&c| {
                state.write_u8(5);
                c.hash(state);
            }),
            VectorInner::Strs(ps) => {
                for i in 0..ps.len() {
                    state.write_u8(4);
                    ps.get_str(i).unwrap_or("").hash(state);
                }
            }
        }
    }
}

impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        map_value_eq(self, other)
    }
}

impl Eq for Map {}

impl Hash for Map {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.size());
        let mut acc: u64 = 0;
        for (k, v) in self.iter() {
            acc = acc.wrapping_add(entry_hash(k, v));
        }
        state.write_u64(acc);
    }
}

impl PartialEq for Set {
    fn eq(&self, other: &Self) -> bool {
        set_value_eq(self, other)
    }
}

impl Eq for Set {}

impl Hash for Set {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.size());
        let mut acc: u64 = 0;
        for v in self.iter() {
            acc = acc.wrapping_add(entry_hash(v, v));
        }
        state.write_u64(acc);
    }
}

/// An empty [`Map`] (shaped, identity-fresh).
#[inline(never)]
#[must_use]
pub fn new_map() -> Map {
    Map(Arc::new(MapData {
        inner: None,
        small: Some(fresh_empty_small()),
        keys: None,
        meta: None,
        hashy: false,
        sort: None,
    }))
}

/// An empty [`Set`].
#[must_use]
pub fn new_set() -> Set {
    Set(Arc::new(SetData {
        storage: SetStorage::Small(Vec::new()),
        meta: None,
        sort: None,
        id: next_set_id(),
    }))
}

/// A hole-free index and order sidecar over `members` (in order).
fn slots_of(members: Vec<Value>) -> (RpdsIndex, SetSlots) {
    let mut inner = RpdsIndex::new_with_hasher_and_ptr_kind(FxBuildHasher::default());
    let mut slots = SetSlots::new_sync();
    for v in members {
        inner.insert_mut(v.clone(), slots.len());
        slots.push_back_mut(Some(v));
    }
    (inner, slots)
}

impl LargeSet {
    fn remove_unsorted_slot(&mut self, v: &Value, slot: usize) {
        self.inner.remove_mut(v);
        let members = Arc::make_mut(&mut self.members);
        members.set_mut(slot, None);
        self.holes += 1;
        while members.get(self.head).is_some_and(Option::is_none) {
            self.head += 1;
        }
        if self.holes * 2 > members.len() {
            self.compact(false);
        }
    }

    fn remove_sorted_at(&mut self, pos: usize) {
        let slot = self.head + pos;
        let member = self
            .members
            .get(slot)
            .cloned()
            .flatten()
            .expect("sorted position in range");
        self.inner.remove_mut(&member);
        if pos == 0 {
            Arc::make_mut(&mut self.members).set_mut(slot, None);
            self.holes += 1;
            self.head += 1;
            if self.holes * 2 > self.members.len() {
                self.compact(true);
            }
        } else {
            let kept = self.members.iter_from(self.head).enumerate();
            let kept = kept.filter(|&(i, _)| i != pos).map(|(_, m)| m.clone());
            self.members = Arc::new(kept.collect());
            self.holes = 0;
            self.head = 0;
        }
    }

    /// Drop the order sidecar's holes (the members keep their order).
    fn compact(&mut self, sorted: bool) {
        let live: Vec<Value> = self
            .members
            .iter_from(self.head)
            .flatten()
            .cloned()
            .collect();
        if sorted {
            // Sorted sets keep no slots: only the sidecar rebuilds.
            self.members = Arc::new(live.into_iter().map(Some).collect());
        } else {
            let (inner, members) = slots_of(live);
            self.inner = inner;
            self.members = Arc::new(members);
        }
        self.holes = 0;
        self.head = 0;
    }
}

/// A map entry from a key and a value (FX2: `find`, map seqs, and
/// entry conj all build real `MapEntry` values, never 2-vectors).
#[must_use]
pub fn map_entry(key: Value, val: Value) -> Value {
    Value::MapEntry {
        key: Box::new(key),
        val: Box::new(val),
        id: next_identity_id(),
    }
}

/// A tagged literal (fence 4a funnel: the only `TaggedLiteral`
/// constructor — mints the identity id).
#[must_use]
pub fn tagged_literal(tag: Symbol, form: Value) -> Value {
    Value::TaggedLiteral {
        tag: Box::new(tag),
        form: Box::new(form),
        id: next_identity_id(),
    }
}

/// A reader conditional (fence 4a funnel: the only
/// `ReaderConditional` constructor — mints the identity id).
#[must_use]
pub fn reader_conditional(form: Value, splicing: bool) -> Value {
    Value::ReaderConditional {
        form: Box::new(form),
        splicing,
        id: next_identity_id(),
    }
}

/// A Clojure runtime value (S2 foundation; S3+ adds functions, vars,
/// refs, and the concurrency primitives as new variants).
/// `repr(u8)` (issue #259): natively compiled code relies on this
/// layout — tag byte at offset 0 in declaration order (`Nil`=0,
/// `Bool`=1 with its payload at offset 1, `Int`=2 / `Double`=3 with
/// payloads at offset 8, `Char`=9) — pinned by `value_layout_abi`.
/// The repr changes nothing measured: same 32 bytes, same bytes.
#[derive(Clone, Debug)]
#[repr(u8)]
pub enum Value {
    /// `nil`.
    Nil,
    /// `true` / `false`.
    Bool(bool),
    /// 64-bit integer (checked arithmetic; overflow errors, never wraps).
    Int(i64),
    /// IEEE-754 double.
    Double(f64),
    /// Arbitrary-precision integer (`clojure.lang.BigInt`,
    /// allocation-boxed: `identical?` is `Arc::ptr_eq` — every
    /// construction is fresh, like the JVM).
    BigInt(Arc<num_bigint::BigInt>),
    /// Arbitrary-precision integer (`java.math.BigInteger`, S2,
    /// allocation-boxed like [`Value::BigInt`]): value-identical
    /// to it (same `=`/`hash`/arithmetic — ops absorb it into
    /// `BigInt`), but prints bare (`5`, never `5N`).
    BigInteger(Arc<num_bigint::BigInt>),
    /// Arbitrary-precision decimal (`java.math.BigDecimal`,
    /// allocation-boxed: `identical?` is `Arc::ptr_eq` —
    /// literals box fresh, arithmetic results share
    /// `ZERO_THROUGH_TEN` through [`bigdec_result`]).
    BigDec(Arc<crate::BigDec>),
    /// Exact rational (`clojure.lang.Ratio`, always normalized,
    /// allocation-boxed like [`Value::BigInt`]).
    Ratio(Arc<num_rational::BigRational>),
    /// String.
    Str(Arc<str>),
    /// Character.
    Char(char),
    /// Symbol.
    Symbol(Symbol),
    /// Keyword.
    Keyword(Keyword),
    /// Persistent list.
    List(List),
    /// Persistent vector.
    Vector(Vector),
    /// A map entry (`clojure.lang.MapEntry`, FX2): a 2-element
    /// `APersistentVector` — vector-like (`nth`, `conj`, `=`,
    /// `hash`) but never an `IObj` (no metadata slot by
    /// construction) and never an `IPersistentMap`. The `id`
    /// is the allocation identity for `identical?` (fence 4a).
    MapEntry {
        key: Box<Value>,
        val: Box<Value>,
        id: u64,
    },
    /// Persistent map.
    Map(Map),
    /// Persistent set.
    Set(Set),
    /// Regex pattern source (matching arrives with `re-*` in S5).
    Regex(Arc<str>),
    /// A UUID (S9): 128 bits, `java.util.UUID` semantics.
    Uuid(crate::Uuid),
    /// An instant (S9): millis since the epoch, `java.util.Date`
    /// semantics.
    Instant(crate::Instant),
    /// A URI (S9): validated string plus raw parts, `java.net.URI`
    /// semantics. Boxed: the parsed parts are larger than the rest
    /// of the payload, and `Value` travels in `Result` errors.
    Uri(Box<crate::Uri>),
    /// `#tag form` under `:read-cond :preserve` (or suppressed reads).
    /// The `id` is the allocation identity for `identical?` (fence 4a).
    TaggedLiteral {
        // Boxed: tagged literals are cold (reader-only), and this
        // keeps the variant at 24 bytes so `Value` fits 64.
        tag: Box<Symbol>,
        form: Box<Value>,
        id: u64,
    },
    /// `#?` / `#?@` under `:read-cond :preserve`.
    /// The `id` is the allocation identity for `identical?` (fence 4a).
    ReaderConditional {
        form: Box<Value>,
        splicing: bool,
        id: u64,
    },
    /// A namespace Var (S4): identity-compared by `id`, like the JVM.
    Var(Var),
    /// An unbound Var's sentinel root (S4): compared by `var_id`.
    Unbound(Unbound),
    /// A throwable (S4): identity-compared by `id`, like the JVM.
    /// Behind an `Arc` so `Value` stays 32 bytes (#254); field
    /// reads ride the `Arc` deref.
    Throwable(Arc<Throwable>),
    /// A transient buffer (S5D): identity equality (shared cell).
    Transient(Transient),
    /// An atom (S6A): identity equality (shared cell, equal meta).
    Atom(Atom),
    /// A delay (S6A): identity equality (shared cell — no meta).
    Delay(Delay),
    /// A future (S6B): identity equality (shared cell, equal meta).
    Future(Future),
    /// A promise (S6B): identity equality (shared cell, equal meta).
    Promise(Promise),
    /// An agent (S6C): identity equality (shared cell, equal meta).
    Agent(Agent),
    /// A volatile (S10): identity equality (shared cell — no meta).
    Volatile(Volatile),
    /// A ref (S19): identity equality (shared cell, equal meta).
    Ref(Ref),
    /// A host-defined value behind the [`ExtensionValue`] trait (S4).
    Extension(Arc<dyn ExtensionValue>),
    /// A port-defined type descriptor or instance (S-typechain):
    /// `deftype` / `defrecord` / `reify` / `proxy` / `gen-class`
    /// values and `definterface` / `gen-interface` descriptors.
    HostType(Arc<HostVal>),
    /// A multimethod (S-typechain stage 3): the `defmulti` value,
    /// invoked like a function.
    MultiFn(Arc<MultiFn>),
    /// A JVM class as a value (S16): named equality (same name is
    /// `=`), string hashing, no metadata, never seqable.
    Class(Arc<ClassVal>),
    /// A Java array (S15): typed, fixed-length, mutable, identity
    /// equality (see [`Array`]).
    Array(Array),
    /// A primitive vector (S15 `vector-of`): immutable, sequential
    /// equality like vectors (see [`PrimitiveVector`]).
    PrimitiveVector(PrimitiveVector),
    /// An executor (fence 3c): a single-thread pool `send-via`
    /// honors. Identity equality (shared pool — no meta).
    Executor(Executor),
}

/// Read metadata off a sequence extension value (borrowed).
/// Single source for the sequence-type list shared with the
/// rebuild helper below.
fn seq_ext_meta(e: &Arc<dyn ExtensionValue>) -> Option<&Map> {
    let any = e.as_any();
    any.downcast_ref::<crate::RealizedSeq>()
        .and_then(crate::RealizedSeq::meta)
        .or_else(|| {
            any.downcast_ref::<crate::RangeValue>()
                .and_then(crate::RangeValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::RepeatValue>()
                .and_then(crate::RepeatValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::CycleValue>()
                .and_then(crate::CycleValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::IterateValue>()
                .and_then(crate::IterateValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::ConsValue>()
                .and_then(crate::ConsValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::KeySeqValue>()
                .and_then(crate::KeySeqValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::ValSeqValue>()
                .and_then(crate::ValSeqValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::SubVectorValue>()
                .and_then(crate::SubVectorValue::meta)
        })
        .or_else(|| {
            any.downcast_ref::<crate::SplitVecValue>()
                .and_then(crate::SplitVecValue::meta)
        })
}

/// Rebuild a sequence extension value with new metadata (`None`
/// for non-sequence extensions).
fn seq_ext_with_meta(
    e: &Arc<dyn ExtensionValue>,
    meta: Option<Map>,
) -> Option<Arc<dyn ExtensionValue>> {
    let any = e.as_any();
    macro_rules! retag {
        ($t:ty) => {
            if let Some(v) = any.downcast_ref::<$t>() {
                let rebuilt: Arc<dyn ExtensionValue> = Arc::new(v.clone().with_meta(meta));
                return Some(rebuilt);
            }
        };
    }
    retag!(crate::RealizedSeq);
    retag!(crate::RangeValue);
    retag!(crate::RepeatValue);
    retag!(crate::CycleValue);
    retag!(crate::IterateValue);
    retag!(crate::ConsValue);
    retag!(crate::KeySeqValue);
    retag!(crate::ValSeqValue);
    retag!(crate::SubVectorValue);
    retag!(crate::SplitVecValue);
    None
}

/// Whether an extension value supports metadata (the sequence
/// types). A rebuild probe, so the type list lives in the
/// rebuild helper alone.
#[must_use]
pub fn extension_supports_meta(e: &Arc<dyn ExtensionValue>) -> bool {
    seq_ext_with_meta(e, None).is_some()
}

/// Attach metadata to a supported extension value (`None` when
/// unsupported — check `extension_supports_meta` first).
#[must_use]
pub fn extension_with_meta(
    e: &Arc<dyn ExtensionValue>,
    meta: Option<Map>,
) -> Option<Arc<dyn ExtensionValue>> {
    seq_ext_with_meta(e, meta)
}

impl Value {
    /// `*self = v`, without the out-of-line drop-glue call when `self`
    /// holds a plain scalar (nothing to free): the common case for the
    /// runtime shims, which write slots compiled code has just dropped
    /// (#271).
    #[inline(always)]
    pub fn assign(&mut self, v: Value) {
        if matches!(
            self,
            Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Double(_) | Value::Char(_)
        ) {
            std::mem::forget(std::mem::replace(self, v));
        } else {
            *self = v;
        }
    }

    /// Clojure truthiness: only `nil` and `false` are false.
    #[must_use]
    pub fn is_truthy(&self) -> bool {
        !matches!(self, Value::Nil | Value::Bool(false))
    }

    /// The metadata map, if this value carries one (Symbol, List,
    /// Vector, Map, Set, primitive vectors, realized seqs — never
    /// Keyword, never arrays).
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        match self {
            Value::Symbol(s) => s.meta(),
            Value::List(l) => l.meta(),
            Value::Vector(v) => v.meta(),
            Value::Map(m) => m.meta(),
            Value::Set(s) => s.meta(),
            Value::PrimitiveVector(p) => p.meta(),
            Value::Extension(e) => seq_ext_meta(e),
            Value::HostType(h) => h.as_inst().and_then(|i| i.meta.as_ref()),
            _ => None,
        }
    }

    /// Attach `meta` to this value, which must support metadata
    /// ([`meta`](Value::meta)); other values are returned unchanged.
    #[must_use]
    pub fn with_meta(self, meta: Option<Map>) -> Self {
        match self {
            Value::Symbol(s) => Value::Symbol(s.with_meta(meta)),
            Value::List(l) => Value::List(l.with_meta(meta)),
            Value::Vector(v) => Value::Vector(v.with_meta(meta)),
            Value::Map(m) => Value::Map(m.with_meta(meta)),
            Value::Set(s) => Value::Set(s.with_meta(meta)),
            Value::PrimitiveVector(p) => Value::PrimitiveVector(p.with_meta(meta)),
            Value::Extension(e) => seq_ext_with_meta(&e, meta)
                .map(Value::Extension)
                .unwrap_or_else(|| Value::Extension(e)),
            Value::HostType(h) => match h.as_ref() {
                HostVal::Inst(i) => Value::HostType(Arc::new(HostVal::Inst(HostInst {
                    def: i.def.clone(),
                    content: i.content.clone(),
                    meta,
                    proxy_map: i.proxy_map.clone(),
                }))),
                HostVal::Def(_) => Value::HostType(h.clone()),
            },
            other => other,
        }
    }
}

/// Port-defined-type pair equality (split for the complexity
/// budget): identical `Arc`s are `=` (so `(= p p)` holds);
/// otherwise descriptors are never `=`, and records need the same
/// type plus content ([`host_eq`]).
fn host_pair_eq(a: &Arc<HostVal>, b: &Arc<HostVal>) -> bool {
    Arc::ptr_eq(a, b) || host_eq(a, b)
}

/// The content map of a struct instance (S12; `None` for every
/// other value, including record instances — records never equal
/// plain maps).
fn host_struct_content(h: &HostVal) -> Option<&Map> {
    match h {
        HostVal::Inst(i) if i.def.kind == HostKind::Struct => Some(&i.content),
        _ => None,
    }
}

/// The content map when `v` is a struct instance (S12).
fn struct_content(v: &Value) -> Option<&Map> {
    match v {
        Value::HostType(h) => host_struct_content(h),
        _ => None,
    }
}

/// The map behind a value: plain maps read directly, struct
/// instances read their content (S12); anything else is `None`.
fn map_or_struct_content(v: &Value) -> Option<&Map> {
    match v {
        Value::Map(m) => Some(m),
        other => struct_content(other),
    }
}

/// Port-defined pair equality (split from `value_eq_mid` and
/// `plain_equals` for the complexity budget): host pairs follow
/// `host_pair_eq`, multifn pairs compare by identity (like the
/// JVM, which never overrides `equals` on them); anything else is
/// `None` for the caller's fallthrough.
fn port_pair_eq(a: &Value, b: &Value) -> Option<bool> {
    match (a, b) {
        (Value::HostType(x), Value::HostType(y)) => Some(host_pair_eq(x, y)),
        (Value::MultiFn(x), Value::MultiFn(y)) => Some(Arc::ptr_eq(x, y)),
        // Struct instances compare as their content map: to plain
        // maps and across bases alike (S12 oracles `sm-eq-map`,
        // `sm-cross-basis`); against anything else they miss.
        // (Struct pairs land in the `HostType` arm above, which
        // compares content too.)
        (a, b) if struct_content(a).is_some() || struct_content(b).is_some() => {
            Some(match (map_or_struct_content(a), map_or_struct_content(b)) {
                (Some(x), Some(y)) => map_value_eq(x, y),
                _ => false,
            })
        }
        // Arrays are identity-compared (same cell only — never
        // sequential, never cross-type).
        (Value::Array(x), Value::Array(y)) => Some(x.same_cell(y)),
        _ => None,
    }
}

/// Shared-cell identity plus equal metadata (a `with-meta` copy
/// shares the cell but compares unequal — split out so
/// [`Value::eq`] stays within the complexity budget).
fn cell_meta_eq(same_cell: bool, a: Option<Map>, b: Option<Map>) -> bool {
    same_cell && a == b
}

impl PartialEq for Value {
    /// Longs and keywords (the common lookup keys) compare without a
    /// stack frame; every other pair takes [`value_eq_full`] (#271:
    /// its frame cost more than the compare in set and map probes).
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Keyword(a), Value::Keyword(b)) => a == b,
            _ => value_eq_full(self, other),
        }
    }
}

#[inline(never)]
fn value_eq_full(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Nil, Value::Nil) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        // `=` on doubles is `==` (`0.0` ≡ `-0.0`, NaN never).
        // Mixed `Int`/`Double` is never `=` (`Numbers.equal`
        // requires one category; `(== 1 1.0)` is the value
        // comparison, arriving with numerics in S4).
        (Value::Double(a), Value::Double(b)) => a == b,
        (Value::Str(a), Value::Str(b)) => a == b,
        (Value::Char(a), Value::Char(b)) => a == b,
        (Value::Symbol(a), Value::Symbol(b)) => a == b,
        // Sequential equality crosses List/Vector.
        (Value::List(a), Value::List(b)) => list_eq(a, b),
        (Value::Vector(a), Value::Vector(b)) => vector_eq(a, b),
        (Value::List(a), Value::Vector(b)) | (Value::Vector(b), Value::List(a)) => {
            list_vector_eq(a, b)
        }
        // Realized seqs compare sequentially against
        // List/Vector/realized seqs (never maps, sets, or
        // pending seqs, which `=` realizes first).
        (a, b) if either_seq(a, b) => seq_items_eq(a, b),
        pair => value_eq_mid(pair.0, pair.1),
    }
}

/// List equality: the same list without a walk, as the JVM's
/// `Util.equiv` answers `k1 == k2` first (#271: an atom's `swap!`
/// compares its value with itself on every publish), else item by item.
fn list_eq(a: &List, b: &List) -> bool {
    Arc::ptr_eq(&a.0, &b.0) || a.iter().eq(b.iter())
}

#[inline]
fn packed_eq<T: Clone + PartialEq>(x: &crate::trie::Packed<T>, y: &crate::trie::Packed<T>) -> bool {
    if let (Some(xs), Some(ys)) = (x.as_flat(), y.as_flat()) {
        return xs == ys;
    }
    x.iter().eq(y.iter())
}

/// Vector equality: the same vector without a walk (see [`list_eq`]), then
/// zero-clone slice comparisons for matching packed/flat representations.
fn vector_eq(a: &Vector, b: &Vector) -> bool {
    if Arc::ptr_eq(&a.0, &b.0) {
        return true;
    }
    if a.len() != b.len() {
        return false;
    }
    match (&a.0.inner, &b.0.inner) {
        (VectorInner::Ints(x), VectorInner::Ints(y)) => packed_eq(x, y),
        (VectorInner::Doubles(x), VectorInner::Doubles(y)) => packed_eq(x, y),
        (VectorInner::Chars(x), VectorInner::Chars(y)) => packed_eq(x, y),
        (VectorInner::Generic(x), VectorInner::Generic(y)) => packed_eq(x, y),
        (VectorInner::Strs(x), VectorInner::Strs(y)) => {
            (0..x.len()).all(|i| x.get_str(i) == y.get_str(i))
        }
        _ => a.iter().eq(b.iter()),
    }
}

/// List/vector cross equality (split for the budget): lists
/// borrow, vectors yield owned values since packing — compare
/// pairwise without cloning.
fn list_vector_eq(a: &List, b: &Vector) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == &y)
}

/// At least one side is sequential (`seq_items`).
fn either_seq(a: &Value, b: &Value) -> bool {
    seq_items(a).is_some() || seq_items(b).is_some()
}

fn seq_items_eq(a: &Value, b: &Value) -> bool {
    match (seq_items(a), seq_items(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Maps, sets, reader forms, and identity values. Chained so each
/// match stays under the complexity gate.
fn value_eq_mid(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Map(a), Value::Map(b)) => map_value_eq(a, b),
        (Value::Set(a), Value::Set(b)) => set_value_eq(a, b),
        // Identity semantics like java.util.regex.Pattern.
        (Value::Regex(a), Value::Regex(b)) => Arc::ptr_eq(a, b),
        // `TaggedLiteral.equals`/`ReaderConditional.equals`
        // delegate to the form's `.equals` (doubles by bits, so
        // `[0.0]` ≠ `[-0.0]` but `[##NaN]` ≡ `[##NaN]`); `=`
        // reaches them through the `Util.equiv` `.equals`
        // fallback. The `Hash` impl stays `equiv`-based: bits-equal
        // forms hash equally, so the `Eq`/`Hash` contract holds.
        (
            Value::TaggedLiteral {
                tag: t1, form: f1, ..
            },
            Value::TaggedLiteral {
                tag: t2, form: f2, ..
            },
        ) => tagged_literal_eq(t1, f1, t2, f2),
        (
            Value::ReaderConditional {
                form: f1,
                splicing: s1,
                ..
            },
            Value::ReaderConditional {
                form: f2,
                splicing: s2,
                ..
            },
        ) => reader_cond_eq(*s1, f1, *s2, f2),
        // S4 identity semantics, mirroring `=`: ids only, never
        // payloads; extensions dispatch through the trait.
        (Value::Var(a), Value::Var(b)) => a.id == b.id,
        (Value::Unbound(a), Value::Unbound(b)) => a.var_id == b.var_id,
        (Value::Throwable(a), Value::Throwable(b)) => a.id == b.id,
        (Value::Transient(a), Value::Transient(b)) => a.same_cell(b),
        // Shared cell AND equal meta: a `with-meta` copy shares
        // the cell but compares unequal (delays carry no meta,
        // so the cell alone decides there).
        (Value::Atom(a), Value::Atom(b)) => cell_meta_eq(a.same_cell(b), a.meta(), b.meta()),
        (Value::Delay(a), Value::Delay(b)) => a.same_cell(b),
        (Value::Future(a), Value::Future(b)) => cell_meta_eq(a.same_cell(b), a.meta(), b.meta()),
        // Port-defined pairs compare structurally (host content,
        // multifn identity); anything else falls through.
        pair => port_pair_eq(pair.0, pair.1).unwrap_or_else(|| value_eq_rest(pair.0, pair.1)),
    }
}

fn value_eq_rest(a: &Value, b: &Value) -> bool {
    if is_tower(a) || is_tower(b) {
        return tower_equiv(a, b);
    }
    match (a, b) {
        (Value::Promise(a), Value::Promise(b)) => cell_meta_eq(a.same_cell(b), a.meta(), b.meta()),
        (Value::Agent(a), Value::Agent(b)) => cell_meta_eq(a.same_cell(b), a.meta(), b.meta()),
        (Value::Volatile(a), Value::Volatile(b)) => a.same_cell(b),
        (Value::Ref(a), Value::Ref(b)) => cell_meta_eq(a.same_cell(b), a.meta(), b.meta()),
        (Value::Executor(a), Value::Executor(b)) => a.same_pool(b),
        (Value::Extension(a), Value::Extension(b)) => a.dyn_eq(&**b),
        // Classes compare by name alone (same name is `=`, whatever
        // the import path that produced the value).
        (Value::Class(a), Value::Class(b)) => a.name() == b.name(),
        // S9 scalars: UUID by bits, instants by millis, URIs by
        // the JDK `equals` (content only — never the print id).
        (Value::Uuid(a), Value::Uuid(b)) => a == b,
        (Value::Instant(a), Value::Instant(b)) => a == b,
        (Value::Uri(a), Value::Uri(b)) => a.uri_eq(b),
        _ => false,
    }
}

/// Either side is a tower numeric
/// (`BigInt`/`BigInteger`/`BigDec`/`Ratio`).
fn is_tower(v: &Value) -> bool {
    matches!(
        v,
        Value::BigInt(_) | Value::BigInteger(_) | Value::BigDec(_) | Value::Ratio(_)
    )
}

/// Tower `=` (oracle-pinned), shared by `==` (`PartialEq`) and
/// `Util.equiv` (`jvm_equiv`): `BigInt`/`BigInteger` equiv each
/// other and `Int` by value; `BigDec` equivs itself by
/// `compareTo` (scale-insensitive); `Ratio` equivs itself;
/// every other cross pairing is false.
fn tower_equiv(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::BigInt(x) | Value::BigInteger(x), Value::BigInt(y) | Value::BigInteger(y)) => {
            x == y
        }
        (Value::BigInt(x) | Value::BigInteger(x), Value::Int(y)) => {
            x.to_i64().is_some_and(|i| i == *y)
        }
        (Value::Int(x), Value::BigInt(y) | Value::BigInteger(y)) => {
            y.to_i64().is_some_and(|i| i == *x)
        }
        (Value::BigDec(x), Value::BigDec(y)) => x.compare(y) == std::cmp::Ordering::Equal,
        (Value::Ratio(x), Value::Ratio(y)) => x == y,
        _ => false,
    }
}

/// Map equality: the same map without a walk (the JVM's `Util.equiv`
/// identity check; see [`Value`]'s `PartialEq`), else entry by entry.
fn map_value_eq(a: &Map, b: &Map) -> bool {
    Arc::ptr_eq(&a.0, &b.0) || (a.size() == b.size() && a.iter().all(|(k, v)| b.get(k) == Some(v)))
}

/// Set equality: the same set without a walk, else member by member.
fn set_value_eq(a: &Set, b: &Set) -> bool {
    Arc::ptr_eq(&a.0, &b.0) || (a.size() == b.size() && a.iter().all(|v| b.contains(v)))
}

fn tagged_literal_eq(t1: &Symbol, f1: &Value, t2: &Symbol, f2: &Value) -> bool {
    t1 == t2 && plain_equals(f1, f2)
}

fn reader_cond_eq(s1: bool, f1: &Value, s2: bool, f2: &Value) -> bool {
    s1 == s2 && plain_equals(f1, f2)
}

// `Eq`'s reflexivity technically fails for NaN (never `=` to itself,
// exactly as in Clojure); the marker is required for map/set keys,
// where NaN behaves identically to the JVM: it hashes
// deterministically but never matches on lookup.
impl Eq for Value {}

/// Canonical bits for hashing doubles: `-0.0` hashes as `0.0`
/// (they are `=`), NaN its canonical bits, anything else its own
/// bits. (`Int` hashes separately — mixed numerics are never `=`.)
fn double_hash_bits(d: f64) -> u64 {
    if d.is_nan() {
        f64::NAN.to_bits()
    } else if d == 0.0 {
        0.0f64.to_bits()
    } else {
        d.to_bits()
    }
}

/// Tower `Hash` arms. Chained so each match stays under the
/// complexity gate: a `BigInt` that fits is exactly its `Int`
/// (they are `=`); wider ones hash by magnitude bytes; stripped
/// decimals hash equally exactly when `compareTo` says so (the
/// `=` rule); rationals are always normalized, so the pair is
/// unique.
fn hash_tower_value<H: Hasher>(v: &Value, state: &mut H) {
    match v {
        Value::BigInt(n) | Value::BigInteger(n) => {
            if let Some(i) = n.to_i64() {
                state.write_u8(2);
                state.write_i64(i);
            } else {
                state.write_u8(23);
                hash_bigint_bytes(n, state);
            }
        }
        Value::BigDec(d) => {
            state.write_u8(24);
            let (unscaled, scale) = d.stripped();
            hash_bigint_bytes(&unscaled, state);
            state.write_i64(scale);
        }
        Value::Ratio(r) => {
            state.write_u8(25);
            hash_bigint_bytes(r.numer(), state);
            hash_bigint_bytes(r.denom(), state);
        }
        _ => unreachable!("tower gate"),
    }
}

impl Hash for Value {
    /// Nil and longs (the common hashed keys) hash inline into the
    /// trie probe; every other value takes [`hash_value_full`] (#271:
    /// its frame cost several times the two hasher writes).
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Int(i) => {
                state.write_u8(2);
                state.write_i64(*i);
            }
            Value::Nil => state.write_u8(0),
            other => hash_value_full(other, state),
        }
    }
}

#[inline(never)]
fn hash_value_full<H: Hasher>(v: &Value, state: &mut H) {
    // Family markers keep unequal types apart (`Int` and
    // `Double` hash separately — they are never `=`).
    match v {
        Value::Bool(b) => {
            state.write_u8(1);
            state.write_u8(u8::from(*b));
        }
        Value::Double(d) => {
            state.write_u8(3);
            state.write_u64(double_hash_bits(*d));
        }
        Value::BigInt(_) | Value::BigInteger(_) | Value::BigDec(_) | Value::Ratio(_) => {
            hash_tower_value(v, state);
        }
        Value::Str(s) => {
            state.write_u8(4);
            s.hash(state);
        }
        Value::Char(c) => {
            state.write_u8(5);
            c.hash(state);
        }
        Value::Symbol(s) => {
            state.write_u8(6);
            s.hash(state);
        }
        Value::Keyword(k) => {
            state.write_u8(7);
            k.hash(state);
        }
        other => hash_value_mid(other, state),
    }
}

/// Length-prefixed magnitude bytes (ambiguity-free across
/// adjacent fields).
fn hash_bigint_bytes<H: Hasher>(n: &num_bigint::BigInt, state: &mut H) {
    let bytes = n.to_signed_bytes_be();
    state.write_usize(bytes.len());
    state.write(&bytes);
}

/// A primitive vector hashes exactly like its item vector (split
/// from [`hash_value_mid`] for the complexity budget).
fn hash_pv<H: Hasher>(p: &crate::PrimitiveVector, state: &mut H) {
    state.write_u8(8);
    state.write_usize(p.len());
    for v in p.iter() {
        v.hash(state);
    }
}

fn hash_value_mid<H: Hasher>(v: &Value, state: &mut H) {
    match v {
        // Sequential hash ignores List-vs-Vector (they can be `=`).
        Value::List(l) => {
            state.write_u8(8);
            l.hash(state);
        }
        Value::Vector(v) => {
            state.write_u8(8);
            v.hash(state);
        }
        // Primitive vectors hash exactly like their item vector
        // (they are `=` to it).
        Value::PrimitiveVector(p) => hash_pv(p, state),
        // A map entry hashes exactly like its 2-vector (they
        // are `=`, so the `Eq`/`Hash` contract requires it;
        // Oracle FX2 `hash-eq`, `distinct`, `frequencies`).
        Value::MapEntry { key, val, .. } => {
            state.write_u8(8);
            state.write_usize(2);
            key.hash(state);
            val.hash(state);
        }
        // Order-insensitive: sums commute, so iteration order
        // cannot leak into the hash.
        Value::Map(m) => {
            state.write_u8(9);
            m.hash(state);
        }
        Value::Set(s) => {
            state.write_u8(10);
            s.hash(state);
        }
        // Same-pointer values share the pattern string, so the
        // string hash is consistent with pointer equality.
        Value::Regex(pattern) => {
            state.write_u8(11);
            pattern.hash(state);
        }
        Value::TaggedLiteral { tag, form, .. } => {
            state.write_u8(12);
            tag.hash(state);
            form.hash(state);
        }
        Value::ReaderConditional { form, splicing, .. } => {
            state.write_u8(13);
            splicing.hash(state);
            form.hash(state);
        }
        Value::Var(v) => {
            state.write_u8(14);
            state.write_u64(v.id);
        }
        Value::Unbound(u) => {
            state.write_u8(15);
            state.write_u64(u.var_id);
        }
        // Throwables and transients are id-hashed cells (tags
        // 16/17 live in `id_cell_tag`, reached via
        // `hash_value_rest`).
        other => hash_value_rest(other, state),
    }
}

/// The tag byte plus minted id for id-hashed cells (throwables,
/// transients, atoms, delays, futures, promises, agents,
/// volatiles, refs, executors — each hashes its tag plus its id),
/// or `None` for every other variant (split from
/// [`hash_value_rest`] for the complexity budget, like
/// [`hash_value_scalars`]).
fn id_cell_tag(v: &Value) -> Option<(u8, u64)> {
    match v {
        Value::Throwable(t) => Some((16, t.id)),
        Value::Transient(t) => Some((17, t.id())),
        Value::Atom(a) => Some((18, a.id())),
        Value::Delay(d) => Some((19, d.id())),
        Value::Future(f) => Some((20, f.id())),
        Value::Promise(p) => Some((21, p.id())),
        Value::Agent(a) => Some((22, a.id())),
        Value::Volatile(v) => Some((27, v.id())),
        Value::Ref(r) => Some((28, r.id())),
        Value::Executor(x) => Some((30, x.id())),
        _ => None,
    }
}

fn hash_value_rest<H: Hasher>(v: &Value, state: &mut H) {
    if let Some((tag, id)) = id_cell_tag(v) {
        state.write_u8(tag);
        state.write_u64(id);
        return;
    }
    match v {
        Value::Extension(e) => hash_extension(e, state),
        // Port-defined types: the class name plus the content map
        // (order-insensitive, like `Map`, so equal records hash
        // equally whatever the extension order). Struct instances
        // hash EXACTLY like their content map (they are `=` to
        // plain maps — oracle `sm-eq-map` — so the `Eq`/`Hash`
        // contract requires identical writes to `Map`).
        Value::HostType(h) => {
            if let Some(content) = host_struct_content(h) {
                state.write_u8(9);
                content.hash(state);
            } else {
                state.write_u8(23);
                h.type_name().hash(state);
                if let Some(i) = h.as_inst() {
                    i.content.hash(state);
                }
            }
        }
        // Multimethods hash their name (identity equality, so
        // identical `Arc`s hash equally by construction).
        Value::MultiFn(m) => {
            state.write_u8(26);
            m.name.hash(state);
        }
        // Classes hash their name (`=`-equal classes share buckets).
        // (Discriminant 28 is shared with `Instant` in
        // `hash_value_scalars`, and 27 with `Uuid`/`Array` —
        // mixing bytes, not unique tags; both sides keep their
        // base discriminants so pinned `(hash …)` values do not
        // move.)
        Value::Class(c) => {
            state.write_u8(28);
            c.name().hash(state);
        }
        // Arrays hash their identity (identity equality).
        Value::Array(a) => {
            state.write_u8(27);
            state.write_u64(a.id());
        }
        // S9 scalars and anything else upstream-missed funnel to the
        // scalar split (which rejects non-scalars the same way).
        other => hash_value_scalars(other, state),
    }
}

/// S9 scalar hashes (split from [`hash_value_rest`] for the
/// complexity budget): UUID by bits, instants by millis, URIs by
/// normalized content (consistent with `uri_eq`).
fn hash_value_scalars<H: Hasher>(v: &Value, state: &mut H) {
    match v {
        Value::Uuid(u) => {
            state.write_u8(27);
            state.write_u64(u.hi);
            state.write_u64(u.lo);
        }
        Value::Instant(i) => {
            state.write_u8(28);
            state.write_i64(i.millis);
        }
        Value::Uri(u) => {
            state.write_u8(29);
            u.hash_content(state);
        }
        // All other variants are handled upstream (the single
        // caller partitions the variants).
        _ => unreachable!("hash partition"),
    }
}

fn hash_extension<H: Hasher>(e: &Arc<dyn ExtensionValue>, state: &mut H) {
    // Realized seqs hash as sequential family (marker 8
    // plus length-prefixed elements — identical writes
    // to `List`, so sequential equals hash equally).
    if let Some(s) = e.as_any().downcast_ref::<crate::RealizedSeq>() {
        state.write_u8(8);
        let items = s.items();
        items.len().hash(state);
        for item in &items {
            item.hash(state);
        }
    } else if let Some(r) = e.as_any().downcast_ref::<crate::RangeValue>() {
        state.write_u8(8);
        r.walked_len().hash(state);
        r.for_each(&mut |v| v.hash(state));
    } else if let Some(r) = e.as_any().downcast_ref::<crate::RepeatValue>() {
        state.write_u8(8);
        r.walked_len().hash(state);
        r.for_each(&mut |v| v.hash(state));
    } else if let Some(sv) = e.as_any().downcast_ref::<crate::SubVectorValue>() {
        state.write_u8(8);
        sv.len().hash(state);
        for item in sv.iter() {
            item.hash(state);
        }
    } else if let Some(sp) = e.as_any().downcast_ref::<crate::SplitVecValue>() {
        state.write_u8(8);
        sp.len().hash(state);
        for item in sp.iter() {
            item.hash(state);
        }
    } else {
        state.write_u8(17);
        e.dyn_hash(state);
    }
}

/// Hash of one map/set entry with a fixed seed, for the
/// order-insensitive sums above.
fn entry_hash(k: &Value, v: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    k.hash(&mut h);
    v.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Duplicate-key rules for collection literals
// ---------------------------------------------------------------------------

/// `PersistentArrayMap.canBePAM`: at most 16 forms, or at most 128
/// forms whose every key at even index ≥ 16 is a Keyword.
fn can_be_small_map(keys: &[&Value]) -> bool {
    let forms = keys.len() * 2;
    if forms <= 16 {
        return true;
    }
    if forms > 128 {
        return false;
    }
    keys.iter()
        .enumerate()
        .all(|(i, k)| i * 2 < 16 || matches!(k, Value::Keyword(_)))
}

/// Small-map scan (`PersistentArrayMap.createWithCheck`): pairwise
/// over earlier/later pairs, reporting the FIRST key.
fn find_map_dup<'a>(keys: &[&'a Value]) -> Option<&'a Value> {
    if can_be_small_map(keys) {
        for (i, a) in keys.iter().enumerate() {
            for b in &keys[i + 1..] {
                if equal_key(a, b) {
                    return Some(a);
                }
            }
        }
        None
    } else {
        // Big-map scan (`PersistentHashMap.createWithCheck`):
        // incremental, reporting the NEW key. The JVM looks keys up
        // by hash, but equivalent keys always hash equally, so a
        // linear scan finds exactly the same duplicates.
        for (i, b) in keys.iter().enumerate() {
            if keys[..i].iter().any(|a| equal_key(a, b)) {
                return Some(b);
            }
        }
        None
    }
}

/// Set scan (`PersistentHashSet.createWithCheck`, transient
/// `indexOf`): incremental over previous members, reporting the NEW
/// item — same hash/equiv equivalence argument as big maps.
fn find_set_dup(items: &[Value]) -> Option<&Value> {
    for (i, b) in items.iter().enumerate() {
        if items[..i].iter().any(|a| equal_key(a, b)) {
            return Some(b);
        }
    }
    None
}

/// `PersistentArrayMap.equalKey`: keywords compare by identity (value
/// equality under interning); everything else by `Util.equiv`.
fn equal_key(a: &Value, b: &Value) -> bool {
    if matches!(a, Value::Keyword(_)) {
        a == b
    } else {
        jvm_equiv(a, b)
    }
}

/// Port of `Util.equiv`, the comparison behind every duplicate check
/// (small maps pairwise, big maps and sets incrementally):
///
/// - Doubles compare by `==` (`0.0` ≡ `-0.0`); NaN ≡ NaN holds
///   because every S2 NaN is the one `##NaN` (on the JVM, the shared
///   `Double.NaN` instance, caught by the `k1 == k2` identity probe).
/// - Mixed `Int`/`Double` never compare: `Numbers.equal` requires
///   the same category.
/// - Collections compare through `pcequiv` (`ASeq.equiv` /
///   `doEquiv` / `APersistentMap.equiv` / `APersistentSet.equiv`),
///   recursing with full `equiv` — sequential comparison crosses
///   `List`/`Vector`/`MapEntry` in every order, at every nesting
///   level (an entry is its 2-vector).
/// - Anything else falls back to Java `.equals` ([`plain_equals`]).
fn jvm_equiv(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Double(x), Value::Double(y)) => {
            if x.is_nan() && y.is_nan() {
                true
            } else {
                x == y
            }
        }
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Int(_), Value::Double(_)) | (Value::Double(_), Value::Int(_)) => false,
        // Tower rules live in `tower_equiv` (shared with `==`).
        (a, b) if is_tower(a) || is_tower(b) => tower_equiv(a, b),
        // Port-defined types `equiv` exactly like `=`.
        (Value::HostType(a), Value::HostType(b)) => host_pair_eq(a, b),
        // Multimethods `equiv` by identity (like `=`).
        (Value::MultiFn(a), Value::MultiFn(b)) => Arc::ptr_eq(a, b),
        _ => {
            if is_collection(a) {
                coll_equiv(a, b)
            } else if is_collection(b) {
                coll_equiv(b, a)
            } else {
                plain_equals(a, b)
            }
        }
    }
}

fn is_collection(v: &Value) -> bool {
    matches!(
        v,
        Value::List(_)
            | Value::Vector(_)
            | Value::MapEntry { .. }
            | Value::Map(_)
            | Value::Set(_)
            | Value::PrimitiveVector(_)
    ) || struct_content(v).is_some()
}

/// Same length plus pairwise comparison (either comparator). The
/// sides borrow independently (`Vector::iter` yields owned values
/// since packing; lists and entries still borrow).
fn seq_eq<I, J>(
    a_len: usize,
    a: I,
    b_len: usize,
    b: J,
    mut eq: impl FnMut(&Value, &Value) -> bool,
) -> bool
where
    I: Iterator,
    I::Item: std::borrow::Borrow<Value>,
    J: Iterator,
    J::Item: std::borrow::Borrow<Value>,
{
    if a_len != b_len {
        return false;
    }
    let mut count = 0;
    for (x, y) in a.zip(b) {
        if !eq(x.borrow(), y.borrow()) {
            return false;
        }
        count += 1;
    }
    count == a_len
}

/// `pcequiv` with `c` the collection side: sequential `equiv` crosses
/// `List`/`Vector`/`MapEntry` (an entry is its 2-vector); maps match
/// same-size with equiv keys and values; sets match same-size with
/// every member of the OTHER side equiv-contained
/// (`APersistentSet.equiv` iterates its argument).
fn coll_equiv(c: &Value, x: &Value) -> bool {
    match c {
        Value::List(items) => seq_coll_equiv(items.len(), items.iter(), x),
        Value::Vector(items) => seq_coll_equiv(items.len(), items.iter(), x),
        Value::PrimitiveVector(p) => seq_coll_equiv(p.len(), p.iter(), x),
        Value::MapEntry { key, val, .. } => {
            seq_coll_equiv(2, [key.as_ref(), val.as_ref()].into_iter(), x)
        }
        Value::Map(m) => map_coll_equiv(m, x),
        Value::Set(s) => set_coll_equiv(s, x),
        // Struct instances equiv as their content map (S12; the
        // only host value `is_collection` admits, so the `None`
        // arm is unreachable — `is_some_and` keeps it branchless).
        Value::HostType(h) => host_struct_content(h).is_some_and(|m| map_coll_equiv(m, x)),
        _ => false,
    }
}

fn seq_coll_equiv<I>(len: usize, items: I, x: &Value) -> bool
where
    I: Iterator,
    I::Item: std::borrow::Borrow<Value>,
{
    match x {
        Value::List(o) => seq_eq(len, items, o.len(), o.iter(), jvm_equiv),
        Value::Vector(o) => seq_eq(len, items, o.len(), o.iter(), jvm_equiv),
        Value::PrimitiveVector(p) => seq_eq(len, items, p.len(), p.iter(), jvm_equiv),
        Value::MapEntry { key, val, .. } => seq_eq(
            len,
            items,
            2,
            [key.as_ref(), val.as_ref()].into_iter(),
            jvm_equiv,
        ),
        _ => false,
    }
}

fn map_coll_equiv(m: &Map, x: &Value) -> bool {
    let Some(o) = map_or_struct_content(x) else {
        return false;
    };
    if m.size() != o.size() {
        return false;
    }
    // `containsKey` + equiv value per entry: hash lookup
    // finds exactly the equiv keys, so scan linearly.
    m.iter().all(|(k, v)| {
        o.iter()
            .find(|(k2, _)| jvm_equiv(k, k2))
            .is_some_and(|(_, v2)| jvm_equiv(v, v2))
    })
}

fn set_coll_equiv(s: &Set, x: &Value) -> bool {
    let Value::Set(o) = x else {
        return false;
    };
    s.size() == o.size() && o.iter().all(|y| s.iter().any(|z| jvm_equiv(z, y)))
}

/// Port of Java `.equals` (`Util.equals` + the collection
/// Sequential items for equality: lists, vectors, and realized
/// seqs (anything else — maps, sets, pending seqs — is not
/// sequential here).
fn seq_items(v: &Value) -> Option<Vec<Value>> {
    match v {
        Value::List(l) => Some(l.iter().cloned().collect()),
        Value::Vector(v) => Some(v.iter().collect()),
        // A map entry compares sequentially as its 2-vector
        // (Oracle FX2 `eq`, `entry-vs-list`, `entry-vs-seq`).
        Value::MapEntry { key, val, .. } => Some(vec![(**key).clone(), (**val).clone()]),
        // Primitive vectors compare sequentially (cross-type too,
        // like the JVM `Vec.equals`).
        Value::PrimitiveVector(p) => Some(p.items()),
        Value::Extension(e) => {
            if let Some(s) = e.as_any().downcast_ref::<crate::RealizedSeq>() {
                return Some(s.items());
            }
            // Strict pure seqs walk without the evaluator
            // (infinite ones hang, like the JVM's `Util.equiv`).
            if let Some(r) = e.as_any().downcast_ref::<crate::RangeValue>() {
                let mut out = Vec::new();
                r.for_each(&mut |v| out.push(v.clone()));
                return Some(out);
            }
            if let Some(r) = e.as_any().downcast_ref::<crate::RepeatValue>() {
                let mut out = Vec::new();
                r.for_each(&mut |v| out.push(v.clone()));
                return Some(out);
            }
            if let Some(sv) = e.as_any().downcast_ref::<crate::SubVectorValue>() {
                return Some(sv.items());
            }
            if let Some(sp) = e.as_any().downcast_ref::<crate::SplitVecValue>() {
                return Some(sp.items());
            }
            None
        }
        _ => None,
    }
}

/// `equals`/`mapEquals`/`setEquals`): like [`jvm_equiv`] except
/// doubles compare by bits, and it is only reached for scalar pairs
/// or nested inside [`Value::TaggedLiteral`]/[`Value::ReaderConditional`]
/// forms (whose `equals` delegates to the form's `.equals`).
fn plain_equals(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Nil, Value::Nil) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Double(x), Value::Double(y)) => x.to_bits() == y.to_bits(),
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Char(x), Value::Char(y)) => x == y,
        (Value::Symbol(x), Value::Symbol(y)) => x == y,
        (Value::Keyword(x), Value::Keyword(y)) => x == y,
        (Value::Regex(x), Value::Regex(y)) => Arc::ptr_eq(x, y),
        (Value::Var(a), Value::Var(b)) => a.id == b.id,
        (Value::Unbound(a), Value::Unbound(b)) => a.var_id == b.var_id,
        (Value::Throwable(a), Value::Throwable(b)) => a.id == b.id,
        other => {
            port_pair_eq(other.0, other.1).unwrap_or_else(|| plain_equals_rest(other.0, other.1))
        }
    }
}

fn plain_equals_rest(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (
            Value::TaggedLiteral {
                tag: t1, form: f1, ..
            },
            Value::TaggedLiteral {
                tag: t2, form: f2, ..
            },
        ) => t1 == t2 && plain_equals(f1, f2),
        (
            Value::ReaderConditional {
                form: f1,
                splicing: s1,
                ..
            },
            Value::ReaderConditional {
                form: f2,
                splicing: s2,
                ..
            },
        ) => s1 == s2 && plain_equals(f1, f2),
        (Value::Extension(a), Value::Extension(b)) => a.dyn_eq(&**b),
        (Value::Map(m), Value::Map(o)) => plain_equals_maps(m, o),
        (Value::Set(s), Value::Set(o)) => plain_equals_sets(s, o),
        // S9 scalars: `.equals` is the `=` rule here too.
        (Value::Uuid(a), Value::Uuid(b)) => a == b,
        (Value::Instant(a), Value::Instant(b)) => a == b,
        (Value::Uri(a), Value::Uri(b)) => a.uri_eq(b),
        // `.equals` is loose across List/Vector too (`doEquals` takes
        // the `java.util.List` branch; `ASeq.equals` accepts `List`;
        // an entry is its 2-vector; primitive vectors join the
        // family, like the JVM `Vec.equals`).
        (
            Value::List(_) | Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_),
            _,
        )
        | (
            _,
            Value::List(_) | Value::Vector(_) | Value::MapEntry { .. } | Value::PrimitiveVector(_),
        ) => plain_equals_seq(a, b),
        _ => false,
    }
}

/// The `.equals` items of a sequential value (`None` otherwise).
fn plain_seq_items(v: &Value) -> Option<Vec<Value>> {
    match v {
        Value::List(l) => Some(l.iter().cloned().collect()),
        Value::Vector(v) => Some(v.iter().collect()),
        Value::MapEntry { key, val, .. } => Some(vec![(**key).clone(), (**val).clone()]),
        Value::PrimitiveVector(p) => Some(p.items()),
        _ => None,
    }
}

fn plain_equals_seq(a: &Value, b: &Value) -> bool {
    match (plain_seq_items(a), plain_seq_items(b)) {
        (Some(x), Some(y)) => {
            x.len() == y.len() && x.iter().zip(&y).all(|(u, v)| plain_equals(u, v))
        }
        _ => false,
    }
}

fn plain_equals_maps(m: &Map, o: &Map) -> bool {
    m.size() == o.size()
        && m.iter().all(|(k, v)| {
            o.iter()
                .find(|(k2, _)| plain_equals(k, k2))
                .is_some_and(|(_, v2)| plain_equals(v, v2))
        })
}

fn plain_equals_sets(s: &Set, o: &Set) -> bool {
    s.size() == o.size() && o.iter().all(|y| s.iter().any(|z| plain_equals(z, y)))
}

#[cfg(test)]
mod tests {
    use super::{
        Value, bigdec_result, find_map_dup, find_set_dup, intern_literal_strings, intern_string,
        new_map, new_set, plain_equals,
    };
    use crate::{HostDef, HostInst, HostKind, HostVal, Keyword, Symbol, Vector};

    #[test]
    fn vector_iter_nth_skips_flat_and_trie_vectors() {
        // Flat storage (constructed), then trie storage (pushed past
        // `trie::FLAT_MAX`), for every packing.
        let generic: Vec<Value> = (0..10)
            .map(|i| Value::Str(Arc::from(i.to_string())))
            .collect();
        let generic = super::Vector::from_vec(generic);
        let mut it = generic.iter();
        assert_eq!(it.nth(3), Some(Value::Str(Arc::from("3"))));
        assert_eq!(it.next(), Some(Value::Str(Arc::from("4"))));
        let ints = super::Vector::from_ints((0..10).collect());
        assert_eq!(ints.iter().nth(9), Some(Value::Int(9)));
        assert_eq!(ints.iter().nth(10), None);
        let chars = super::Vector::from_chars(vec!['a', 'b', 'c', 'd']);
        assert_eq!(chars.iter().nth(3), Some(Value::Char('d')));
        let (mut big_ints, mut big_chars) = (
            super::Vector::from_ints(Vec::new()),
            super::Vector::from_chars(Vec::new()),
        );
        let mut big = super::Vector::from_vec(Vec::new());
        for i in 0..2_000i64 {
            big_ints = big_ints.push_back(Value::Int(i));
            big_chars = big_chars.push_back(Value::Char(if i % 2 == 0 { 'a' } else { 'b' }));
            big = big.push_back(Value::Str(Arc::from(i.to_string())));
        }
        assert_eq!(big_ints.iter().nth(1_500), Some(Value::Int(1_500)));
        assert_eq!(big_chars.iter().nth(1_501), Some(Value::Char('b')));
        assert_eq!(big.iter().nth(1_999), Some(Value::Str(Arc::from("1999"))));
    }

    /// Seeking iteration (#271 Slice 5m): `iter_from` starts at the
    /// index for every packing (flat and trie), so a window deep into a
    /// large vector never walks the items before it.
    #[test]
    fn vector_iter_from_seeks_every_packing() {
        let collect =
            |v: &super::Vector, from: usize| v.iter_from(from).take(3).collect::<Vec<Value>>();
        let generic = super::Vector::from_vec((0..2_000).map(Value::Int).collect());
        let mut ints = super::Vector::from_ints(Vec::new());
        let mut doubles = super::Vector::from_doubles(Vec::new());
        let mut chars = super::Vector::from_chars(Vec::new());
        for i in 0..2_000i64 {
            ints = ints.push_back(Value::Int(i));
            doubles = doubles.push_back(Value::Double(i as f64 / 2.0));
            chars = chars.push_back(Value::Char('a'));
        }
        let strs = super::Vector::from_vec(
            (0..2_000)
                .map(|i| Value::Str(Arc::from(i.to_string())))
                .collect(),
        );
        assert_eq!(
            collect(&generic, 1_500),
            (1_500..1_503).map(Value::Int).collect::<Vec<_>>()
        );
        assert_eq!(
            collect(&ints, 1_024),
            (1_024..1_027).map(Value::Int).collect::<Vec<_>>()
        );
        assert_eq!(
            collect(&doubles, 31),
            (31..34)
                .map(|i| Value::Double(i as f64 / 2.0))
                .collect::<Vec<_>>()
        );
        assert_eq!(collect(&chars, 40), vec![Value::Char('a'); 3]);
        assert_eq!(collect(&strs, 1_999), vec![Value::Str(Arc::from("1999"))]);
        // A seek past the end is empty, for flat and trie storage.
        assert_eq!(generic.iter_from(2_000).next(), None);
        assert_eq!(ints.iter_from(9_999).next(), None);
        assert_eq!(
            super::Vector::from_vec(Vec::new()).iter_from(3).next(),
            None
        );
    }

    /// The layout natively compiled code relies on (issue #259).
    #[test]
    fn value_layout_abi() {
        fn bytes(v: &Value) -> [u8; 32] {
            // SAFETY: `Value` is 32 bytes (asserted below); reading its
            // bytes is a plain copy of initialized-or-padding memory
            // used only for the tag/payload bytes checked here.
            unsafe { std::mem::transmute_copy::<Value, [u8; 32]>(v) }
        }
        assert_eq!(std::mem::size_of::<Value>(), 32);
        // A set's order sidecar slot (`None` = a `disj` hole) costs no
        // more than the member itself.
        assert_eq!(std::mem::size_of::<Option<Value>>(), 32);
        assert_eq!(std::mem::align_of::<Value>(), 8);
        assert_eq!(bytes(&Value::Nil)[0], 0);
        let t = bytes(&Value::Bool(true));
        assert_eq!((t[0], t[1]), (1, 1));
        assert_eq!(bytes(&Value::Bool(false))[1], 0);
        let i = bytes(&Value::Int(-2));
        assert_eq!(i[0], 2);
        assert_eq!(i64::from_ne_bytes(i[8..16].try_into().expect("8")), -2);
        let d = bytes(&Value::Double(1.5));
        assert_eq!(d[0], 3);
        assert_eq!(f64::from_ne_bytes(d[8..16].try_into().expect("8")), 1.5);
        let c = bytes(&Value::Char('\u{1F600}'));
        assert_eq!(c[0], 9);
        // The AOT compiler reads a char's code point at offset 4 (#259).
        assert_eq!(u32::from_ne_bytes(c[4..8].try_into().expect("4")), 0x1F600);
        // ... and tests for a vector before its flat `nth` front (#271).
        assert_eq!(bytes(&Value::Vector(Vector::from_ints(vec![1])))[0], 13);
    }
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    // Fence 4a: the literal pool canonicalizes (equal literals
    // share one `Arc`); direct `Arc::from` constructions stay
    // fresh (the JVM's `new String`).
    #[test]
    fn string_pool_canonicalizes() {
        let a = intern_string(&Arc::from("f4a-pool"));
        let b = intern_string(&Arc::from("f4a-pool"));
        assert!(Arc::ptr_eq(&a, &b));
        let fresh = Arc::from("f4a-pool");
        assert!(!Arc::ptr_eq(&a, &fresh));
    }

    // Fence 4a-big: arithmetic-result decimals at scale-0
    // `0..=10` share the canonical instance (the JVM's
    // `BigDecimal.valueOf` rule); negatives, `11+`, nonzero
    // scales, and past-`i64` magnitudes box fresh — as do
    // literals (direct `Arc::new`, the `new BigDecimal` analog).
    #[test]
    fn bigdec_result_canonicalizes_zero_through_ten() {
        use num_bigint::BigInt;
        let boxed = |d: crate::BigDec| opt_bigdec(&bigdec_result(d)).unwrap().clone();
        let share = |n: i32| {
            Arc::ptr_eq(
                &boxed(crate::BigDec::from_bigint(BigInt::from(n))),
                &boxed(crate::BigDec::from_bigint(BigInt::from(n))),
            )
        };
        assert!(share(0));
        assert!(share(5));
        assert!(share(10));
        assert!(!share(11));
        assert!(!share(-1));
        assert!(!Arc::ptr_eq(
            &boxed(crate::BigDec::parse("5.0").unwrap()),
            &boxed(crate::BigDec::parse("5.0").unwrap()),
        ));
        let wide = BigInt::from(2).pow(100);
        assert!(!Arc::ptr_eq(
            &boxed(crate::BigDec::from_bigint(wide.clone())),
            &boxed(crate::BigDec::from_bigint(wide)),
        ));
        let literal = || Value::BigDec(Arc::new(crate::BigDec::from_bigint(BigInt::from(5))));
        assert!(!Arc::ptr_eq(
            opt_bigdec(&literal()).unwrap(),
            opt_bigdec(&literal()).unwrap(),
        ));
    }

    // Fence 4a test extractors: `Option` + `.unwrap()` is the
    // only guard shape the 100% line floor accepts — the floor
    // counts test code, `if let` without `else` leaves its
    // closing brace uncovered, and rustfmt always explodes
    // single-line `let-else`. Both arms of each helper stay hot
    // (pins take `Some`, the rejection test takes `None`).
    fn opt_vector(v: &Value) -> Option<&super::Vector> {
        match v {
            Value::Vector(inner) => Some(inner),
            _ => None,
        }
    }

    fn opt_str(v: &Value) -> Option<&Arc<str>> {
        match v {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    fn opt_entry(v: &Value) -> Option<(&Value, &Value)> {
        match v {
            Value::MapEntry { key, val, .. } => Some((key.as_ref(), val.as_ref())),
            _ => None,
        }
    }

    fn opt_tagged(v: &Value) -> Option<(&Symbol, &Value)> {
        match v {
            Value::TaggedLiteral { tag, form, .. } => Some((tag.as_ref(), form.as_ref())),
            _ => None,
        }
    }

    fn opt_rc(v: &Value) -> Option<(&Value, bool)> {
        match v {
            Value::ReaderConditional { form, splicing, .. } => Some((form.as_ref(), *splicing)),
            _ => None,
        }
    }

    fn opt_bigdec(v: &Value) -> Option<&Arc<crate::BigDec>> {
        match v {
            Value::BigDec(d) => Some(d),
            _ => None,
        }
    }

    #[test]
    fn opt_extractors_reject_wrong_kinds() {
        assert!(opt_vector(&Value::Int(1)).is_none());
        assert!(opt_str(&Value::Int(1)).is_none());
        assert!(opt_entry(&Value::Int(1)).is_none());
        assert!(opt_tagged(&Value::Int(1)).is_none());
        assert!(opt_rc(&Value::Int(1)).is_none());
        assert!(opt_bigdec(&Value::Int(1)).is_none());
    }

    // Fence 4a: the quote walk interns nested strings and keeps
    // the spine (metadata, key order) intact.
    #[test]
    fn quote_walk_interns_nested_strings() {
        let payload = Value::Vector(super::Vector::from_vec(vec![
            Value::Str(Arc::from("f4a-q")),
            Value::Int(1),
        ]));
        let walked = intern_literal_strings(&payload);
        let items = opt_vector(&walked).unwrap().iter().collect::<Vec<_>>();
        assert_eq!(items.len(), 2);
        let pooled = opt_str(&items[0]).unwrap();
        assert!(Arc::ptr_eq(pooled, &intern_string(&Arc::from("f4a-q"))));
        assert_eq!(items[1], Value::Int(1));
        // Non-string leaves clone through untouched.
        assert_eq!(intern_literal_strings(&Value::Int(7)), Value::Int(7));
    }

    // Fence 4a coverage: the quote walk recurses into map
    // entries, interning both sides.
    #[test]
    fn quote_walk_interns_map_entry_sides() {
        let payload = super::map_entry(
            Value::Str(Arc::from("f4a-k")),
            Value::Str(Arc::from("f4a-v")),
        );
        let walked = intern_literal_strings(&payload);
        let (key, val) = opt_entry(&walked).unwrap();
        let k = opt_str(key).unwrap();
        assert!(Arc::ptr_eq(k, &intern_string(&Arc::from("f4a-k"))));
        let v = opt_str(val).unwrap();
        assert!(Arc::ptr_eq(v, &intern_string(&Arc::from("f4a-v"))));
    }

    // Fence 4a coverage: the quote walk recurses into tagged
    // forms, keeping the tag.
    #[test]
    fn quote_walk_interns_tagged_form() {
        let payload =
            super::tagged_literal(Symbol::new(None, "inst"), Value::Str(Arc::from("f4a-t")));
        let walked = intern_literal_strings(&payload);
        let (tag, form) = opt_tagged(&walked).unwrap();
        assert_eq!(tag, &Symbol::new(None, "inst"));
        let pooled = opt_str(form).unwrap();
        assert!(Arc::ptr_eq(pooled, &intern_string(&Arc::from("f4a-t"))));
    }

    // Fence 4a coverage: the quote walk recurses into reader
    // conditional forms, keeping the splicing flag.
    #[test]
    fn quote_walk_interns_reader_conditional_form() {
        let payload = super::reader_conditional(Value::Str(Arc::from("f4a-r")), true);
        let walked = intern_literal_strings(&payload);
        let (form, splicing) = opt_rc(&walked).unwrap();
        assert!(splicing);
        let pooled = opt_str(form).unwrap();
        assert!(Arc::ptr_eq(pooled, &intern_string(&Arc::from("f4a-r"))));
    }

    fn hash_of(v: &Value) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    fn assert_hash_consistent(a: &Value, b: &Value) {
        assert_eq!(a, b, "{a:?} should equal {b:?}");
        assert_eq!(hash_of(a), hash_of(b), "equal values must hash equally");
    }

    /// Past-`i64` big integers hash by magnitude bytes (the
    /// `hasheq` contract: equal values hash equally).
    #[test]
    fn wide_bigint_hashes_by_magnitude() {
        let wide = || Value::BigInt(Arc::new(num_bigint::BigInt::from(2).pow(100)));
        assert_hash_consistent(&wide(), &wide());
    }

    #[test]
    fn map_entry_sequential_contract() {
        // Oracle T1192 FX2 `eq`, `hash-eq`, `distinct`,
        // `frequencies`: an entry equals its 2-vector (and 2-list)
        // and hashes with it.
        let e = super::map_entry(Value::Keyword(Keyword::new(None, "a")), Value::Int(1));
        let v = Value::Vector(crate::Vector::from_vec(vec![
            Value::Keyword(Keyword::new(None, "a")),
            Value::Int(1),
        ]));
        assert_hash_consistent(&e, &v);
        let l = Value::List(crate::List::from_vec(vec![
            Value::Keyword(Keyword::new(None, "a")),
            Value::Int(1),
        ]));
        assert_hash_consistent(&e, &l);
        assert_ne!(e, Value::Int(1));
    }

    #[test]
    fn scalars_equiv() {
        assert_eq!(Value::Nil, Value::Nil);
        assert_eq!(Value::Bool(true), Value::Bool(true));
        assert_ne!(Value::Bool(true), Value::Bool(false));
        assert_eq!(Value::Int(1), Value::Int(1));
        assert_eq!(Value::Double(1.5), Value::Double(1.5));
        assert_eq!(Value::Str(Arc::from("a")), Value::Str(Arc::from("a")));
        assert_eq!(Value::Char('a'), Value::Char('a'));
        assert_eq!(
            Value::Symbol(Symbol::new(None, "a")),
            Value::Symbol(Symbol::new(None, "a"))
        );
        assert_eq!(
            Value::Keyword(Keyword::new(None, "a")),
            Value::Keyword(Keyword::new(None, "a"))
        );
        // Cross-type values are never `=`.
        assert_ne!(Value::Int(1), Value::Bool(true));
        assert_ne!(Value::Nil, Value::Bool(false));
        assert_ne!(
            Value::Symbol(Symbol::new(None, "a")),
            Value::Keyword(Keyword::new(None, "a"))
        );
        assert_ne!(Value::Str(Arc::from("a")), Value::Char('a'));
    }

    #[test]
    fn numerics_equiv_and_hash() {
        // Same-type comparison (`(== 1 1.0)` is the value
        // comparison; `=` requires one category).
        assert_ne!(Value::Int(1), Value::Double(1.0));
        assert_ne!(Value::Int(0), Value::Double(-0.0));
        assert_ne!(
            Value::Int(9_007_199_254_740_993),
            Value::Double(9_007_199_254_740_993.0)
        );
        assert_hash_consistent(&Value::Int(1), &Value::Int(1));
        assert_hash_consistent(&Value::Double(1.5), &Value::Double(1.5));
        assert_hash_consistent(&Value::Double(0.0), &Value::Double(-0.0));
        assert_ne!(Value::Int(1), Value::Double(1.5));
        assert_ne!(Value::Double(1.5), Value::Double(2.5));
    }

    #[test]
    fn nan_never_equiv_but_hashes() {
        let nan = Value::Double(f64::NAN);
        assert_ne!(nan, nan);
        // Deterministic: same bits every time.
        assert_eq!(hash_of(&nan), hash_of(&Value::Double(f64::NAN)));
        assert_ne!(Value::Double(f64::NAN), Value::Double(1.0));
    }

    #[test]
    fn sequential_cross_equiv_and_hash() {
        let list = Value::List(vec_into_list(vec![Value::Int(1), Value::Int(2)]));
        let vector = Value::Vector(vec_into_vector(vec![Value::Int(1), Value::Int(2)]));
        assert_hash_consistent(&list, &vector);
        let other = Value::Vector(vec_into_vector(vec![Value::Int(2), Value::Int(1)]));
        assert_ne!(list, other);
        let empty_list = Value::List(super::List::new_sync());
        let empty_vector = Value::Vector(super::Vector::new_sync());
        assert_hash_consistent(&empty_list, &empty_vector);
    }

    #[test]
    fn map_equiv_and_lookup() {
        // Bind before wrapping: no destructure, no dead fallback
        // (100% line gate counts test lines too).
        let inner = new_map()
            .insert(Value::Int(1), Value::Str(Arc::from("a")))
            .insert(Value::Keyword(Keyword::new(None, "k")), Value::Nil);
        let m = Value::Map(inner.clone());
        // Mixed numerics are distinct keys: cross-type lookup misses.
        assert_eq!(inner.get(&Value::Double(1.0)), None);
        assert_eq!(inner.get(&Value::Int(1)), Some(&Value::Str(Arc::from("a"))));
        assert_eq!(inner.size(), 2);
        let swapped = Value::Map(
            new_map()
                .insert(Value::Keyword(Keyword::new(None, "k")), Value::Nil)
                .insert(Value::Int(1), Value::Str(Arc::from("a"))),
        );
        assert_hash_consistent(&m, &swapped);
        // `{1 :a 1.0 :b}` keeps both keys.
        let mixed = new_map()
            .insert(Value::Int(1), Value::Str(Arc::from("a")))
            .insert(Value::Double(1.0), Value::Str(Arc::from("b")));
        assert_eq!(mixed.size(), 2);
        assert_eq!(mixed.get(&Value::Int(1)), Some(&Value::Str(Arc::from("a"))));
        assert_eq!(
            mixed.get(&Value::Double(1.0)),
            Some(&Value::Str(Arc::from("b")))
        );
    }

    #[test]
    fn set_equiv() {
        // Bind before wrapping (see map_equiv_and_lookup).
        let inner = new_set().insert(Value::Int(1)).insert(Value::Int(2));
        let a = Value::Set(inner.clone());
        let b = Value::Set(new_set().insert(Value::Int(2)).insert(Value::Int(1)));
        assert_hash_consistent(&a, &b);
        assert!(!inner.contains(&Value::Double(1.0)));
        assert!(inner.contains(&Value::Int(1)));
        assert!(!inner.contains(&Value::Int(3)));
        // `#{1 1.0}` keeps both members.
        let mixed = new_set().insert(Value::Int(1)).insert(Value::Double(1.0));
        assert_eq!(mixed.size(), 2);
    }

    #[test]
    fn regex_is_identity_compared() {
        let re = Value::Regex(Arc::from("a+"));
        assert_eq!(re, re.clone());
        // Separately built identical patterns are NOT `=` (mirrors
        // java.util.regex.Pattern, which has no value equality).
        assert_ne!(re, Value::Regex(Arc::from("a+")));
    }

    #[test]
    fn tagged_and_conditional_equiv_and_hash() {
        let tl = |form| super::tagged_literal(Symbol::new(None, "inst"), form);
        assert_hash_consistent(&tl(Value::Int(1)), &tl(Value::Int(1)));
        assert_ne!(tl(Value::Int(1)), tl(Value::Int(2)));
        assert_ne!(
            tl(Value::Int(1)),
            super::tagged_literal(Symbol::new(None, "uuid"), Value::Int(1))
        );
        let rc = |splicing| super::reader_conditional(Value::Int(1), splicing);
        assert_hash_consistent(&rc(false), &rc(false));
        assert_ne!(rc(false), rc(true));
        assert_ne!(tl(Value::Int(1)), rc(false));
    }

    #[test]
    fn meta_ignored_by_collection_eq_and_hash() {
        let meta = new_map().insert(Value::Keyword(Keyword::new(None, "k")), Value::Bool(true));
        let a = Value::Vector(vec_into_vector(vec![Value::Int(1)]));
        let b = Value::Vector(vec_into_vector(vec![Value::Int(1)]).with_meta(Some(meta)));
        assert_hash_consistent(&a, &b);
        assert!(a.meta().is_none());
        assert!(b.meta().is_some());
        // Keywords and scalars never carry meta.
        assert!(Value::Keyword(Keyword::new(None, "k")).meta().is_none());
        assert!(Value::Int(1).meta().is_none());
    }

    #[test]
    fn truthiness() {
        assert!(!Value::Nil.is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::Bool(true).is_truthy());
        assert!(Value::Int(0).is_truthy());
        assert!(Value::Str(Arc::from("")).is_truthy());
    }

    #[test]
    fn value_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Value>();
    }

    fn vec_into_list(items: Vec<Value>) -> super::List {
        super::List::from_vec(items)
    }

    fn vec_into_vector(items: Vec<Value>) -> super::Vector {
        super::Vector::from_vec(items)
    }

    // -- duplicate rules --

    fn kw(name: &str) -> Value {
        Value::Keyword(Keyword::new(None, name))
    }

    fn map_keys(pairs: &[(Value, Value)]) -> Vec<&Value> {
        pairs.iter().map(|(k, _)| k).collect()
    }

    #[test]
    fn small_map_reports_first() {
        let pairs = vec![
            (Value::Double(0.0), kw("a")),
            (Value::Double(-0.0), kw("b")),
        ];
        let dup = find_map_dup(&map_keys(&pairs)).unwrap();
        assert_eq!(dup, &Value::Double(0.0));
        // Keywords by value.
        let pairs = vec![(kw("a"), Value::Int(1)), (kw("a"), Value::Int(2))];
        assert_eq!(find_map_dup(&map_keys(&pairs)), Some(&kw("a")));
        // No duplicates.
        let pairs = vec![(Value::Int(1), kw("a")), (Value::Double(1.0), kw("b"))];
        assert_eq!(find_map_dup(&map_keys(&pairs)), None);
    }

    #[test]
    fn small_map_seq_cross_both_orders() {
        // Oracle: `{[1] :a (1) :b}` and `{(1) :a [1] :b}` both report
        // the FIRST key (full `equiv`, no order asymmetry).
        let list1 = Value::List(vec_into_list(vec![Value::Int(1)]));
        let vec1 = Value::Vector(vec_into_vector(vec![Value::Int(1)]));
        let pairs = vec![(vec1.clone(), kw("a")), (list1.clone(), kw("b"))];
        assert_eq!(find_map_dup(&map_keys(&pairs)), Some(&vec1));
        let pairs = vec![(list1.clone(), kw("a")), (vec1.clone(), kw("b"))];
        assert_eq!(find_map_dup(&map_keys(&pairs)), Some(&list1));
        // Nested doubles use `==`, not bits: `{(0.0) :a [-0.0] :b}` dups.
        let l0 = Value::List(vec_into_list(vec![Value::Double(0.0)]));
        let vn0 = Value::Vector(vec_into_vector(vec![Value::Double(-0.0)]));
        let pairs = vec![(l0.clone(), kw("a")), (vn0.clone(), kw("b"))];
        assert_eq!(find_map_dup(&map_keys(&pairs)), Some(&l0));
    }

    #[test]
    fn big_map_reports_new() {
        // 18 forms: past the 16-form small-map cutoff.
        let mut pairs: Vec<(Value, Value)> =
            (1..8).map(|i| (Value::Int(i), Value::Int(i))).collect();
        pairs.insert(0, (Value::Double(0.0), Value::Int(0)));
        pairs.push((Value::Double(-0.0), kw("b")));
        let keys = map_keys(&pairs);
        assert_eq!(keys.len(), 9);
        let dup = find_map_dup(&keys).unwrap();
        assert_eq!(dup, &Value::Double(-0.0));
    }

    #[test]
    fn keyword_run_stays_small() {
        // 18 forms, all keys keywords: small-map rule reports FIRST.
        let mut pairs: Vec<(Value, Value)> = (0..8)
            .map(|i| (kw(&format!("k{i}")), Value::Int(i)))
            .collect();
        pairs.push((kw("k0"), Value::Int(9)));
        let dup = find_map_dup(&map_keys(&pairs)).unwrap();
        assert_eq!(dup, &kw("k0"));
    }

    #[test]
    fn sets_report_new() {
        let items = vec![Value::Double(0.0), Value::Double(-0.0)];
        assert_eq!(find_set_dup(&items), Some(&Value::Double(-0.0)));
        assert_eq!(find_set_dup(&[Value::Int(1), Value::Double(1.0)]), None);
        // Distinct regex sources never collide (identity).
        assert_eq!(
            find_set_dup(&[Value::Regex(Arc::from("a")), Value::Regex(Arc::from("a"))]),
            None
        );
        // NaN ≡ NaN in dup checks (the shared `##NaN`).
        assert!(matches!(
            find_set_dup(&[Value::Double(f64::NAN), Value::Double(f64::NAN)]),
            Some(Value::Double(d)) if d.is_nan()
        ));
    }

    #[test]
    fn tagged_forms_compare_by_bits() {
        // Oracle: `{#inst [0.0] 1 #inst [-0.0] 2}` reads fine — the
        // tagged form compares with Java `.equals` (doubles by bits).
        let tl = |v: f64| {
            super::tagged_literal(
                Symbol::new(None, "inst"),
                Value::Vector(vec_into_vector(vec![Value::Double(v)])),
            )
        };
        let pairs = vec![(tl(0.0), Value::Int(1)), (tl(-0.0), Value::Int(2))];
        assert_eq!(find_map_dup(&map_keys(&pairs)), None);
        let pairs = vec![(tl(0.0), Value::Int(1)), (tl(0.0), Value::Int(2))];
        assert_eq!(find_map_dup(&map_keys(&pairs)).unwrap(), &tl(0.0));
        // `=` is bits too (`tl[NaN]` ≡ `tl[NaN]`, unlike plain NaN).
        assert_ne!(tl(0.0), tl(-0.0));
        assert_hash_consistent(&tl(f64::NAN), &tl(f64::NAN));
        // The checked constructors must not collapse bits-distinct
        // tagged keys when inserting (rpds `Eq`-based overwrite).
        let map =
            super::Map::from_kvs_checked(vec![(tl(0.0), Value::Int(1)), (tl(-0.0), Value::Int(2))])
                .unwrap();
        assert_eq!(map.size(), 2);
        let set = super::Set::from_items_checked(vec![tl(0.0), tl(-0.0)]).unwrap();
        assert_eq!(set.size(), 2);
    }

    // #221 shaped maps: same key-sets share one canonical shape
    // (pointer-equal), so per-map memory is values-only; the
    // shape is order-sensitive (`[1 2]` and `[2 1]` iterate
    // differently, like the JVM).
    #[test]
    fn shapes_shared_across_same_key_maps() {
        let kv = |v: i64| {
            super::Map::from_kvs_checked(vec![
                (Value::Int(1), Value::Int(v)),
                (Value::Int(2), Value::Int(v + 1)),
            ])
            .unwrap()
        };
        let a = kv(10);
        let b = kv(20);
        let shape_of = |m: &super::Map| Arc::clone(&m.0.small.as_ref().unwrap().shape);
        assert!(Arc::ptr_eq(&shape_of(&a), &shape_of(&b)));
        // Same-keys assoc keeps the shape (values only are new).
        let a2 = a.clone().insert(Value::Int(1), Value::Int(99));
        assert!(Arc::ptr_eq(&shape_of(&a), &shape_of(&a2)));
        assert_eq!(a2.get(&Value::Int(1)), Some(&Value::Int(99)));
        // Reversed keys are a different shape.
        let c = super::Map::from_kvs_checked(vec![
            (Value::Int(2), Value::Int(1)),
            (Value::Int(1), Value::Int(2)),
        ])
        .unwrap();
        assert!(!Arc::ptr_eq(&shape_of(&a), &shape_of(&c)));
    }

    // Dead shapes leave only a `Weak` behind: re-interning after
    // the last map drops works, and cleanup evicts dead entries
    // (a fresh-keys loop retains at most one window) while live
    // shapes survive.
    #[test]
    fn shape_registry_reclaims_dead_shapes() {
        let one = |i: i64| {
            super::Map::from_kvs_checked(vec![(Value::Int(1000 + i), Value::Int(i))]).unwrap()
        };
        // Re-intern after drop (dead `Weak` or evicted — same arm).
        let t = one(99);
        drop(t);
        assert_eq!(one(99).get(&Value::Int(1099)), Some(&Value::Int(99)));
        // Churn two dropped batches past several cleanups (40
        // consecutive misses guarantee cleanups whatever this
        // thread interned before).
        let live = one(7);
        for i in 0..40 {
            let _dropped = one(10_000 + i);
        }
        for i in 0..40 {
            let _dropped = one(20_000 + i);
        }
        assert_eq!(live.get(&Value::Int(1007)), Some(&Value::Int(7)));
        let mine = |lo: i64, hi: i64| {
            super::SHAPE_REGISTRY.with(|r| {
                r.borrow()
                    .map
                    .iter()
                    .filter(|(k, _)| {
                        k.len() == 1 && matches!(k[0], Value::Int(i) if (lo..hi).contains(&i))
                    })
                    .count()
            })
        };
        // The first batch died before the second batch's first
        // cleanup, so none of its entries linger ...
        assert_eq!(mine(11_000, 11_040), 0);
        // ... and the second batch keeps at most one window.
        assert!(mine(21_000, 21_040) < super::SHAPE_CLEANUP_EVERY);
        // The held map's entry is live.
        let live_strong = super::SHAPE_REGISTRY.with(|r| {
            r.borrow()
                .map
                .get(&vec![Value::Int(1007)][..])
                .map(|w| w.strong_count())
        });
        assert!(live_strong.is_some_and(|n| n >= 1));
    }

    #[test]
    fn insertion_order_shaped_maps_and_sets() {
        // Shaped maps iterate in insertion order (JVM small-map seq).
        let m =
            super::Map::from_kvs_checked(vec![(kw("b"), Value::Int(1)), (kw("a"), Value::Int(2))])
                .unwrap();
        let keys: Vec<Value> = m.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys, vec![kw("b"), kw("a")]);
        // Re-inserting keeps the original position, new value.
        let m2 = m.insert(kw("b"), Value::Int(9));
        assert_eq!(m2.get(&kw("b")), Some(&Value::Int(9)));
        let pairs: Vec<(Value, Value)> = m2.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        assert_eq!(
            pairs,
            vec![(kw("b"), Value::Int(9)), (kw("a"), Value::Int(2))]
        );
        // Sets iterate in insertion order; re-inserting is a no-op.
        let s = super::Set::from_items_checked(vec![Value::Int(2), Value::Int(1)]).unwrap();
        let items: Vec<Value> = s.iter().cloned().collect();
        assert_eq!(items, vec![Value::Int(2), Value::Int(1)]);
        let s2 = s.insert(Value::Int(2));
        assert_eq!(s2.size(), 2);
        // Order-insensitive equality still holds across orders.
        let a =
            super::Map::from_kvs_checked(vec![(kw("b"), Value::Int(1)), (kw("a"), Value::Int(2))])
                .unwrap();
        let b =
            super::Map::from_kvs_checked(vec![(kw("a"), Value::Int(2)), (kw("b"), Value::Int(1))])
                .unwrap();
        assert_hash_consistent(&Value::Map(a), &Value::Map(b));
    }

    #[test]
    fn persistent_sidecar_without_and_snapshot() {
        // `dissoc` drops the key from the sidecar, keeps the rest in
        // order; a missing key is a no-op.
        let m = super::new_map()
            .insert(kw("a"), Value::Int(1))
            .insert(kw("b"), Value::Int(2))
            .insert(kw("c"), Value::Int(3));
        let dropped = m.clone().without(&kw("b"));
        assert_eq!(dropped.ordered_keys(), vec![kw("a"), kw("c")]);
        assert_eq!(dropped.size(), 2);
        let kept = m.clone().without(&kw("zzz"));
        assert_eq!(kept.ordered_keys(), m.ordered_keys());
        // Clones share structure: writing through a clone never
        // mutates the original.
        let grown = m.clone().insert(kw("d"), Value::Int(4));
        assert_eq!(m.ordered_keys(), vec![kw("a"), kw("b"), kw("c")]);
        assert_eq!(grown.get(&kw("c")), Some(&Value::Int(3)));
        // Explicit rebuilds restore the given order.
        let rebuilt =
            m.with_ordered_entries(vec![(kw("c"), Value::Int(30)), (kw("a"), Value::Int(10))]);
        assert_eq!(rebuilt.ordered_keys(), vec![kw("c"), kw("a")]);
        assert_eq!(rebuilt.get(&kw("a")), Some(&Value::Int(10)));
        // Sets mirror maps: `disj` rebuilds, snapshots clone.
        let s = super::new_set()
            .insert(Value::Int(1))
            .insert(Value::Int(2))
            .insert(Value::Int(3));
        let s_dropped = s.clone().without(&Value::Int(2));
        assert_eq!(
            s_dropped.ordered_members(),
            vec![Value::Int(1), Value::Int(3)]
        );
        let s_kept = s.clone().without(&Value::Int(99));
        assert_eq!(s_kept.ordered_members(), s.ordered_members());
        let s_grown = s.clone().insert(Value::Int(4));
        assert_eq!(
            s.ordered_members(),
            vec![Value::Int(1), Value::Int(2), Value::Int(3)]
        );
        assert_eq!(s_grown.size(), 4);
        let s_rebuilt = s.with_ordered_members(vec![Value::Int(3), Value::Int(1)]);
        assert_eq!(
            s_rebuilt.ordered_members(),
            vec![Value::Int(3), Value::Int(1)]
        );
    }

    #[test]
    fn set_disj_holes_keep_insertion_order() {
        let ints = |s: &super::Set| s.iter().cloned().collect::<Vec<_>>();
        let mut s = super::new_set();
        for i in 0..10 {
            s = s.insert(Value::Int(i));
        }
        // Front pops advance past the holes; inner holes keep order.
        let popped = s.clone().without(&Value::Int(0)).without(&Value::Int(1));
        assert_eq!(popped.size(), 8);
        assert_eq!(ints(&popped)[0], Value::Int(2));
        let holed = s.clone().without(&Value::Int(4)).without(&Value::Int(6));
        assert_eq!(
            ints(&holed),
            [0, 1, 2, 3, 5, 7, 8, 9].map(Value::Int).to_vec()
        );
        // A re-added member goes to the back; `get` answers the stored
        // member, a removed one nothing.
        let back = holed.clone().insert(Value::Int(4));
        assert_eq!(ints(&back).last(), Some(&Value::Int(4)));
        assert_eq!(back.get(&Value::Int(4)), Some(&Value::Int(4)));
        assert_eq!(holed.get(&Value::Int(4)), None);
        assert!(!holed.contains(&Value::Int(6)));
        // Removing most members compacts (order kept); the original is
        // untouched.
        let few = (0..8).fold(s.clone(), |acc, i| acc.without(&Value::Int(i * 7 % 10)));
        assert_eq!(ints(&few), [3, 6].map(Value::Int).to_vec());
        assert_eq!(ints(&s), (0..10).map(Value::Int).collect::<Vec<_>>());
        let gone = (0..10).fold(s.clone(), |acc, i| acc.without(&Value::Int(i)));
        assert!(gone.is_empty());
        assert_eq!(ints(&gone.insert(Value::Int(3))), [Value::Int(3)]);
        // Small sets scan their slots, holes included.
        let small = super::new_set()
            .insert(Value::Int(1))
            .insert(Value::Int(2))
            .without(&Value::Int(1));
        assert!(small.contains(&Value::Int(2)) && !small.contains(&Value::Int(1)));
    }

    #[test]
    fn sorted_positional_set_and_map_updates() {
        let ints = |s: &super::Set| s.iter().cloned().collect::<Vec<_>>();
        let sorted = || super::new_set().with_sort(Some(super::SortSpec::Natural));
        // Positional inserts keep the order; lookups read in place.
        let mut s = sorted();
        for (pos, v) in [(0, 5), (0, 1), (1, 3), (3, 9), (3, 7)] {
            s = s.insert_sorted_at(pos, Value::Int(v));
        }
        assert_eq!(ints(&s), [1, 3, 5, 7, 9].map(Value::Int).to_vec());
        assert_eq!(
            (s.sorted_len(), s.sorted_member(2)),
            (5, Some(&Value::Int(5)))
        );
        // Front pops leave holes before `head` (then compact); any
        // other removal rebuilds; the generic `without` finds the
        // member's position.
        let popped = s.clone().without_sorted_at(0).without_sorted_at(0);
        assert_eq!(ints(&popped), [5, 7, 9].map(Value::Int).to_vec());
        assert_eq!(popped.sorted_member(0), Some(&Value::Int(5)));
        let drained = popped.clone().without_sorted_at(0).without_sorted_at(0);
        assert_eq!(
            (ints(&drained), drained.sorted_len()),
            (vec![Value::Int(9)], 1)
        );
        let mid = s.clone().without_sorted_at(2);
        assert_eq!(ints(&mid), [1, 3, 7, 9].map(Value::Int).to_vec());
        let generic = popped
            .clone()
            .without(&Value::Int(7))
            .without(&Value::Int(99));
        assert_eq!(ints(&generic), [5, 9].map(Value::Int).to_vec());
        assert!(!generic.contains(&Value::Int(7)) && generic.contains(&Value::Int(9)));
        assert_eq!(ints(&s), [1, 3, 5, 7, 9].map(Value::Int).to_vec());
        // Maps: shaped inserts rebuild, trie-backed ones keep the trie.
        let mut m = super::new_map().with_sort(Some(super::SortSpec::Natural));
        for i in 0..12 {
            m = m.insert_sorted_at(0, Value::Int(11 - i), Value::Int(i));
            assert_eq!(m.sorted_len(), usize::try_from(i + 1).expect("small"));
        }
        assert_eq!(m.sorted_key(0), Some(&Value::Int(0)));
        assert_eq!(m.sorted_key(11), Some(&Value::Int(11)));
        assert_eq!(m.get(&Value::Int(11)), Some(&Value::Int(0)));
        let small = super::new_map()
            .with_sort(Some(super::SortSpec::Natural))
            .insert_sorted_at(0, Value::Int(2), Value::Int(20))
            .insert_sorted_at(0, Value::Int(1), Value::Int(10));
        assert_eq!(
            (small.sorted_len(), small.sorted_key(1)),
            (2, Some(&Value::Int(2)))
        );
    }

    #[test]
    fn sorted_and_plain_large_map_sidecar() {
        let mut sm = super::new_map().with_sort(Some(super::SortSpec::Natural));
        for i in 0..18 {
            sm = sm.insert(Value::Int(i), Value::Int(i * 10));
        }
        let sm_dropped = sm.clone().without(&Value::Int(5));
        assert_eq!(sm_dropped.ordered_keys().len(), 17);
        let it = sm_dropped.iter();
        assert_eq!(it.size_hint(), (0, Some(17)));
        assert_eq!(it.count(), 17);
        let entries: Vec<(Value, Value)> =
            (0..18).map(|i| (Value::Int(i), Value::Int(i))).collect();
        let sm_rebuilt = sm.with_ordered_entries(entries.clone());
        assert_eq!(sm_rebuilt.ordered_keys().len(), 18);
        let pm = super::new_map().with_ordered_entries(entries);
        assert_eq!(pm.ordered_keys().len(), 18);
        let pm_it = pm.iter();
        assert_eq!(pm_it.size_hint().0, 18);
        assert_eq!(pm_it.count(), 18);
        assert_eq!(pm.without(&Value::Int(5)).size(), 17);
        assert!(sm_rebuilt.with_ordered_entries(Vec::new()).is_empty());
    }

    #[test]
    fn checked_constructors() {
        assert!(super::Map::from_kvs_checked(vec![(kw("a"), Value::Int(1))]).is_ok());
        assert_eq!(
            super::Map::from_kvs_checked(vec![(kw("a"), Value::Int(1)), (kw("a"), Value::Int(2))]),
            Err(kw("a"))
        );
        assert!(super::Set::from_items_checked(vec![Value::Int(1)]).is_ok());
        assert_eq!(
            super::Set::from_items_checked(vec![Value::Int(1), Value::Int(1)]),
            Err(Value::Int(1))
        );
    }

    #[derive(Debug)]
    struct OpaqueExt;

    impl crate::ExtensionValue for OpaqueExt {
        fn class_name(&self) -> String {
            "test.Opaque".to_owned()
        }

        fn print(&self) -> String {
            "#test.Opaque".to_owned()
        }

        fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
            other.as_any().downcast_ref::<Self>().is_some()
        }

        fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
            std::hash::Hasher::write_u8(state, 0x4f);
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn set_is_empty_and_list_eq() {
        assert!(new_set().is_empty());
        assert!(!new_set().insert(Value::Int(1)).is_empty());
        let a = super::List::from_vec(vec![Value::Int(1)]);
        assert_eq!(a, super::List::from_vec(vec![Value::Int(1)]));
        assert_ne!(a, super::List::new_sync());
    }

    #[test]
    fn with_meta_ignores_plain_values() {
        let meta = new_map().insert(kw("k"), Value::Bool(true));
        assert_eq!(Value::Int(1).with_meta(Some(meta.clone())), Value::Int(1));
        assert_eq!(Value::Nil.with_meta(Some(meta)), Value::Nil);
    }

    #[test]
    fn with_meta_ignores_opaque_extensions() {
        let inner = Arc::new(OpaqueExt);
        let v = Value::Extension(inner.clone());
        let meta = new_map().insert(kw("k"), Value::Bool(true));
        assert_eq!(v.clone().with_meta(Some(meta)), v);
        assert_eq!(crate::ExtensionValue::class_name(&*inner), "test.Opaque");
        assert_eq!(crate::ExtensionValue::print(&*inner), "#test.Opaque");
        assert_eq!(hash_of(&v), hash_of(&v.clone()));
    }

    #[test]
    fn seq_ext_meta_helpers() {
        use super::{extension_supports_meta, extension_with_meta};
        use crate::{RealizedSeq, Seq, SubVectorValue, Vector};
        // First and last types in the chain (the last walks every
        // downcast arm); opaque extensions are unsupported.
        let first: Arc<dyn crate::ExtensionValue> = Arc::new(RealizedSeq::new(Seq::empty()));
        let last: Arc<dyn crate::ExtensionValue> =
            Arc::new(SubVectorValue::new(Vector::new_sync(), 0, 0));
        let opaque: Arc<dyn crate::ExtensionValue> = Arc::new(OpaqueExt);
        assert!(extension_supports_meta(&first));
        assert!(extension_supports_meta(&last));
        assert!(!extension_supports_meta(&opaque));
        assert!(extension_with_meta(&opaque, None).is_none());
        // Roundtrip through `Value::{meta, with_meta}`.
        let meta = new_map().insert(kw("a"), Value::Int(1));
        let tagged = Value::Extension(first).with_meta(Some(meta.clone()));
        assert_eq!(tagged.meta(), Some(&meta));
        assert_eq!(tagged.clone().with_meta(None).meta(), None);
        assert!(extension_with_meta(&last, Some(meta)).is_some());
    }

    #[test]
    fn realized_seq_equiv() {
        use crate::{RealizedSeq, Seq};
        let items = vec![Value::Int(1), Value::Int(2)];
        let seq = Value::Extension(Arc::new(RealizedSeq::new(
            Seq::eager(items.clone()).unwrap(),
        )));
        assert_eq!(seq, Value::List(super::List::from_vec(items.clone())));
        assert_eq!(seq, Value::Vector(super::Vector::from_vec(items)));
        assert_ne!(seq, Value::Int(1));
    }

    #[test]
    fn walked_seq_equiv() {
        use crate::{RangeNum, RangeValue, RepeatValue, SubVectorValue};
        let range = Value::Extension(Arc::new(RangeValue::new(
            RangeNum::Int(0),
            Some(RangeNum::Int(2)),
            RangeNum::Int(1),
            true,
            Some(2),
        )));
        let expect = Value::List(super::List::from_vec(vec![Value::Int(0), Value::Int(1)]));
        assert_eq!(range, expect);
        let repeat = Value::Extension(Arc::new(RepeatValue::new(Value::Int(7), Some(2))));
        let expect = Value::Vector(super::Vector::from_vec(vec![Value::Int(7), Value::Int(7)]));
        assert_eq!(repeat, expect);
        let backing = super::Vector::from_vec(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        let sv = Value::Extension(Arc::new(SubVectorValue::new(backing, 1, 3)));
        let expect = Value::List(super::List::from_vec(vec![Value::Int(2), Value::Int(3)]));
        assert_eq!(sv, expect);
    }

    #[test]
    fn transient_eq_is_shared_cell() {
        use crate::{Transient, TransientKind, TransientVector};
        let t = Value::Transient(Transient::new(
            TransientKind::Vector(TransientVector::new()),
        ));
        assert_eq!(t, t.clone());
        let u = Value::Transient(Transient::new(
            TransientKind::Vector(TransientVector::new()),
        ));
        assert_ne!(t, u);
        assert_ne!(t, Value::Int(1));
    }

    #[test]
    fn identity_variants_hash() {
        use crate::{
            Agent, Atom, Delay, Future, Promise, Transient, TransientKind, TransientVector,
        };
        let vals = vec![
            Value::Transient(Transient::new(
                TransientKind::Vector(TransientVector::new()),
            )),
            Value::Atom(Atom::new(Value::Int(0))),
            Value::Delay(Delay::new(Value::Int(0))),
            Value::Future(Future::new()),
            Value::Promise(Promise::new()),
            Value::Agent(Agent::new(Value::Int(0))),
        ];
        for v in &vals {
            assert_eq!(hash_of(v), hash_of(&v.clone()));
        }
    }

    #[test]
    fn seq_extension_hash_matches_sequential() {
        use crate::{RangeNum, RangeValue, RealizedSeq, RepeatValue, Seq, SubVectorValue};
        let items = vec![Value::Int(1), Value::Int(2)];
        let list = Value::List(super::List::from_vec(items.clone()));
        let seq = Value::Extension(Arc::new(RealizedSeq::new(
            Seq::eager(items.clone()).unwrap(),
        )));
        assert_eq!(hash_of(&seq), hash_of(&list));
        let range = Value::Extension(Arc::new(RangeValue::new(
            RangeNum::Int(1),
            Some(RangeNum::Int(3)),
            RangeNum::Int(1),
            true,
            Some(2),
        )));
        assert_eq!(hash_of(&range), hash_of(&list));
        let repeat = Value::Extension(Arc::new(RepeatValue::new(Value::Int(5), Some(2))));
        let fives = Value::List(super::List::from_vec(vec![Value::Int(5), Value::Int(5)]));
        assert_eq!(hash_of(&repeat), hash_of(&fives));
        let backing = super::Vector::from_vec(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        let sv = Value::Extension(Arc::new(SubVectorValue::new(backing, 0, 2)));
        assert_eq!(hash_of(&sv), hash_of(&list));
    }

    #[test]
    fn coll_equiv_rejects_non_collections() {
        assert!(!super::coll_equiv(&Value::Int(1), &Value::Int(1)));
        assert!(!super::coll_equiv(
            &Value::Int(1),
            &Value::Str(Arc::from("x"))
        ));
    }

    /// A struct instance equivs as its content map on the
    /// collection side (S12; duplicate-key policy stays the
    /// constructor's business).
    #[test]
    fn coll_equiv_struct_reads_content_map() {
        let def = Arc::new(HostDef {
            name: Arc::from("clojure.lang.PersistentStructMap"),
            kind: HostKind::Struct,
            fields: Vec::new(),
            methods: std::collections::HashMap::new(),
            interfaces: Vec::new(),
            prefix: None,
            impl_ns: None,
            sigs: std::collections::HashMap::new(),
            takes_this: true,
            struct_keys: Vec::new(),
        });
        let content = new_map().insert(Value::Keyword(Keyword::new(None, "a")), Value::Int(1));
        let sm = Value::HostType(Arc::new(HostVal::Inst(HostInst {
            def,
            content: content.clone(),
            meta: None,
            proxy_map: None,
        })));
        assert!(super::coll_equiv(&sm, &Value::Map(content)));
        assert!(!super::coll_equiv(&sm, &Value::Int(1)));
    }

    #[test]
    fn tagged_identity_forms() {
        use crate::{ThrowClass, Throwable, Unbound, Var};
        let tl = |form| super::tagged_literal(Symbol::new(None, "t"), form);
        let var = |id| {
            Value::Var(Var {
                id,
                sym: Symbol::new(None, "x"),
            })
        };
        assert_eq!(tl(var(1)), tl(var(1)));
        assert_ne!(tl(var(1)), tl(var(2)));
        let unb = |var_id| {
            Value::Unbound(Unbound {
                sym: Symbol::new(None, "y"),
                var_id,
            })
        };
        assert_eq!(tl(unb(1)), tl(unb(1)));
        assert_ne!(tl(unb(1)), tl(unb(2)));
        let thr = |id| {
            Value::Throwable(Arc::new(Throwable {
                id,
                class: ThrowClass::Exception,
                message: None,
                data: None,
                cause: None,
            }))
        };
        assert_eq!(tl(thr(1)), tl(thr(1)));
        assert_ne!(tl(thr(1)), tl(thr(2)));
    }

    #[test]
    fn tagged_collection_forms() {
        let tl = |form| super::tagged_literal(Symbol::new(None, "t"), form);
        let l1 = Value::List(super::List::from_vec(vec![Value::Int(1)]));
        let v1 = Value::Vector(super::Vector::from_vec(vec![Value::Int(1)]));
        let v2 = Value::Vector(super::Vector::from_vec(vec![Value::Int(2)]));
        assert_eq!(tl(l1.clone()), tl(v1.clone()));
        assert_eq!(tl(v1.clone()), tl(l1.clone()));
        assert_ne!(tl(l1.clone()), tl(v2));
        let m1 = Value::Map(new_map().insert(kw("a"), Value::Int(1)));
        let m2 = Value::Map(new_map().insert(kw("a"), Value::Int(1)));
        let m3 = Value::Map(new_map().insert(kw("a"), Value::Int(2)));
        assert_eq!(tl(m1.clone()), tl(m2));
        assert_ne!(tl(m1.clone()), tl(m3));
        let s1 = Value::Set(new_set().insert(Value::Int(1)));
        let s2 = Value::Set(new_set().insert(Value::Int(1)));
        let s3 = Value::Set(new_set().insert(Value::Int(2)));
        assert_eq!(tl(s1.clone()), tl(s2));
        assert_ne!(tl(s1), tl(s3));
        // A sequential form against a scalar (or map) form is never
        // equal: the `plain_equals_seq` fallback.
        assert_ne!(tl(l1.clone()), tl(Value::Int(1)));
        assert_ne!(tl(l1), tl(m1.clone()));
    }

    #[test]
    fn tagged_extension_forms() {
        let tl = |form| super::tagged_literal(Symbol::new(None, "t"), form);
        let e1 = Value::Extension(Arc::new(OpaqueExt));
        let e2 = Value::Extension(Arc::new(OpaqueExt));
        assert_eq!(tl(e1), tl(e2));
        assert_ne!(tl(Value::Extension(Arc::new(OpaqueExt))), tl(Value::Int(1)));
    }

    #[test]
    #[should_panic(expected = "hash partition")]
    fn hash_value_rest_rejects_upstream_variants() {
        // Defensive arm: the single caller partitions the variants,
        // so scalars never arrive here (only a dispatch bug could).
        let mut h = DefaultHasher::new();
        super::hash_value_rest(&Value::Nil, &mut h);
    }

    #[test]
    #[should_panic(expected = "tower gate")]
    fn hash_tower_value_rejects_non_tower() {
        let mut h = DefaultHasher::new();
        super::hash_tower_value(&Value::Nil, &mut h);
    }

    fn host_def(kind: HostKind) -> Arc<HostDef> {
        Arc::new(HostDef {
            name: Arc::from("user.T"),
            kind,
            fields: Vec::new(),
            methods: std::collections::HashMap::new(),
            interfaces: Vec::new(),
            prefix: None,
            impl_ns: None,
            sigs: std::collections::HashMap::new(),
            takes_this: true,
            struct_keys: Vec::new(),
        })
    }

    fn host_inst(def: &Arc<HostDef>) -> Value {
        Value::HostType(Arc::new(HostVal::Inst(HostInst {
            def: def.clone(),
            content: new_map(),
            meta: None,
            proxy_map: None,
        })))
    }

    #[test]
    fn with_meta_on_host_values() {
        // Instances take (or clear) metadata; descriptors ignore it
        // (no caller passes host values today — pinned directly).
        let def = host_def(HostKind::Record);
        let meta = new_map();
        let out = host_inst(&def).with_meta(Some(meta.clone()));
        assert_eq!(out.meta(), Some(&meta));
        let desc = Value::HostType(Arc::new(HostVal::Def(def)));
        assert_eq!(desc.with_meta(Some(meta)).meta(), None);
    }

    #[test]
    fn set_dup_check_compares_host_values() {
        // Duplicate checks compare host values by `equiv`:
        // distinct instances pass, equal records report (only
        // reachable with hand-built values — literals never carry
        // host values — so pinned directly).
        let typ = host_def(HostKind::Type);
        let rec = host_def(HostKind::Record);
        assert_eq!(find_set_dup(&[host_inst(&typ), host_inst(&rec)]), None);
        let items = [host_inst(&rec), host_inst(&rec)];
        assert_eq!(find_set_dup(&items), Some(&items[1]));
    }

    fn test_multifn() -> Value {
        Value::MultiFn(Arc::new(crate::MultiFn {
            name: Arc::from("m"),
            dispatch: Value::Nil,
            default: Value::Keyword(Keyword::new(None, "default")),
            methods: new_map(),
            prefers: new_map(),
            hierarchy: None,
        }))
    }

    #[test]
    fn set_dup_check_compares_multifns() {
        // Duplicate checks compare multifns by identity (only
        // reachable with hand-built values — literals never carry
        // them — so pinned directly).
        let mf = test_multifn();
        assert_eq!(find_set_dup(&[mf.clone(), test_multifn()]), None);
        let items = [mf.clone(), mf.clone()];
        assert_eq!(find_set_dup(&items), Some(&items[1]));
    }

    #[test]
    fn plain_equals_on_multifn_pairs() {
        // `.equals` follows identity too (reachable only through
        // tagged-literal forms, so pinned directly).
        let mf = test_multifn();
        assert!(plain_equals(&mf, &mf));
        assert!(!plain_equals(&mf, &test_multifn()));
    }

    #[test]
    fn plain_equals_on_host_pairs() {
        // `.equals` follows `host_pair_eq`: same-type records with
        // equal content hold; distinct non-records never do (only
        // reachable through tagged-literal forms, so pinned
        // directly).
        let rec = host_def(HostKind::Record);
        assert!(plain_equals(&host_inst(&rec), &host_inst(&rec)));
        let typ = host_def(HostKind::Type);
        assert!(!plain_equals(&host_inst(&typ), &host_inst(&typ)));
    }

    #[test]
    fn identical_collections_are_equal_without_a_walk() {
        // The same collection is equal to itself even holding a NaN (the
        // JVM's `Util.equiv` identity check); equal copies still walk.
        use super::{List, Map, Set, Vector};
        let nan = Value::Double(f64::NAN);
        let m = Value::Map(Map::from_kvs_last_wins(vec![(Value::Int(1), nan.clone())]));
        let v = Value::Vector(Vector::from_vec(vec![nan.clone()]));
        let l = Value::List(List::from_vec(vec![nan.clone()]));
        for x in [&m, &v, &l] {
            assert_eq!(x, &x.clone());
        }
        let s = Set::from_items_checked(vec![Value::Int(1)]).expect("distinct");
        assert_eq!(s, s.clone());
        let other = Value::Vector(Vector::from_vec(vec![nan]));
        assert_ne!(v, other);
        assert_eq!(
            Value::Vector(Vector::from_vec(vec![Value::Int(1)])),
            Value::Vector(Vector::from_vec(vec![Value::Int(1)]))
        );
    }

    #[test]
    fn map_with_vals_shares_the_template_shape() {
        use super::Map;
        let kw = |n: &str| Value::Keyword(crate::Keyword::new(None, n));
        let template = Map::from_kvs_last_wins(vec![(kw("x"), Value::Nil), (kw("y"), Value::Nil)]);
        let built = Map::with_vals(&template, vec![Value::Int(1), Value::Int(2)]);
        let literal =
            Map::from_kvs_last_wins(vec![(kw("x"), Value::Int(1)), (kw("y"), Value::Int(2))]);
        assert_eq!(built, literal);
        let order: Vec<Value> = built.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(order, vec![kw("x"), kw("y")]);
        assert!(Arc::ptr_eq(
            &built.0.small.as_ref().expect("shaped").shape,
            &template.0.small.as_ref().expect("shaped").shape
        ));
        // A count mismatch (or an unshaped template) builds by pairs.
        let short = Map::with_vals(&template, vec![Value::Int(5)]);
        assert_eq!(
            short,
            Map::from_kvs_last_wins(vec![(kw("x"), Value::Int(5))])
        );
        let big: Vec<(Value, Value)> = (0..20).map(|i| (Value::Int(i), Value::Nil)).collect();
        let big = Map::from_kvs_last_wins(big);
        // The values pair with the template's keys in its iteration order.
        let vals: Vec<Value> = (100..120).map(Value::Int).collect();
        let rebuilt = Map::with_vals(&big, vals.clone());
        for (i, (k, _)) in big.iter().enumerate() {
            assert_eq!(rebuilt.get(k), Some(&vals[i]));
        }
    }

    #[test]
    fn bulk_append_preserves_shared_vectors_and_uses_flat_windows() {
        use super::Vector;

        for items in [
            vec![Value::Int(1), Value::Int(2), Value::Int(3)],
            vec![Value::Double(1.0), Value::Double(2.0), Value::Double(3.0)],
            vec![Value::Nil, Value::Bool(true), Value::Bool(false)],
        ] {
            let v = Vector::from_vec(items.clone());
            assert_eq!(v.cannot_contain_nil(), !items.contains(&Value::Nil));
            let mut want = items.clone();
            want.extend_from_slice(&items[1..]);
            let merged = v.clone().try_extend_flat_window(&v, 1, 3).unwrap();
            assert_eq!(merged.iter().collect::<Vec<_>>(), want);
            assert_eq!(v.iter().collect::<Vec<_>>(), items);
            assert!(v.clone().try_extend_flat_window(&v, 1, 4).is_err());
            assert_eq!(v.clone().try_extend_flat_window(&v, 2, 2).unwrap().len(), 3);
        }

        let v = Vector::from_flat_ints((0..2048).collect());
        let merged = v.clone().try_extend_flat_window(&v, 0, v.len()).unwrap();
        assert_eq!(merged.len(), 4096);
        assert_eq!(merged.get(2048), Some(Value::Int(0)));
        let other = Vector::from_doubles(vec![1.0]);
        assert!(v.clone().try_extend_flat_window(&other, 0, 1).is_err());
        let merged = merged.try_extend_flat_window(&v, 0, 1).unwrap();
        assert_eq!(merged.get(4096), Some(Value::Int(0)));
        assert_eq!(v.len(), 2048);
    }

    #[test]
    fn update_existing_map_value_preserves_sharing_and_rolls_back_on_miss() {
        use super::Map;

        for size in [1, 20] {
            let key = Value::Int(0);
            let mut map = Map::from_kvs_last_wins(
                (0..size).map(|i| (Value::Int(i), Value::Int(i))).collect(),
            );
            assert!(!map.update_existing_mut(&key, Err));
            assert_eq!(map.get(&key), Some(&key));
            let original = map.clone();
            assert!(map.update_existing_mut(&key, |old| {
                assert_eq!(old, key);
                Ok(Value::Int(41))
            }));
            assert_eq!(map.get(&key), Some(&Value::Int(41)));
            assert_eq!(original.get(&key), Some(&key));
            assert!(!map.update_existing_mut(&Value::Int(-1), |_| panic!("missing key")));
        }
    }

    #[test]
    fn value_ref_and_small_map_scalar_lookup() {
        use super::{Map, ValueRef, Vector};
        let vi = Vector::from_ints(vec![11]);
        let vc = Vector::from_chars(vec!['q']);
        let vg = Vector::from_vec(vec![Value::Bool(true)]);
        assert!(matches!(vi.get_val_ref(0), Some(ValueRef::Int(11))));
        assert_eq!(vi.as_flat_ints(), Some(&[11][..]));
        assert!(vc.as_flat_ints().is_none());
        assert!(vg.as_flat_ints().is_none());
        let mut vt = Vector::from_ints((0..1025).collect());
        let _shared = vt.clone();
        vt = vt.push_back(Value::Int(1025));
        assert!(vt.as_flat_ints().is_none());
        assert!(vi.get_val_ref(1).is_none());
        assert!(matches!(vc.get_val_ref(0), Some(ValueRef::Char('q'))));
        assert!(matches!(
            vg.get_val_ref(0),
            Some(ValueRef::Ref(Value::Bool(true)))
        ));

        let kw = Value::Keyword(Keyword::new(None, "a"));
        let s = Value::Str(Arc::from("k"));
        let small = Map::from_kvs_last_wins(vec![
            (Value::Char('a'), Value::Int(1)),
            (kw.clone(), Value::Int(2)),
            (Value::Int(7), Value::Int(3)),
            (Value::Bool(false), Value::Int(4)),
            (Value::Nil, Value::Int(5)),
            (s.clone(), Value::Int(6)),
        ]);
        assert_eq!(small.len_opt(), Some(6));
        assert_eq!(
            small.get_small_scalar(&Value::Char('a')),
            Some(Some(&Value::Int(1)))
        );
        assert_eq!(small.get_small_scalar(&Value::Char('b')), Some(None));
        assert_eq!(small.get_small_scalar(&Value::Char('é')), None);
        assert_eq!(small.get_small_scalar(&kw), Some(Some(&Value::Int(2))));
        assert_eq!(
            small.get_small_scalar(&Value::Int(7)),
            Some(Some(&Value::Int(3)))
        );
        assert_eq!(
            small.get_small_scalar(&Value::Bool(false)),
            Some(Some(&Value::Int(4)))
        );
        assert_eq!(
            small.get_small_scalar(&Value::Nil),
            Some(Some(&Value::Int(5)))
        );
        assert_eq!(small.get_small_scalar(&s), Some(Some(&Value::Int(6))));
        assert_eq!(small.get_small_scalar(&Value::Vector(vi)), None);
        let big_key = Value::BigInt(Arc::new(7.into()));
        let big = Map::from_kvs_last_wins(vec![(big_key, Value::Int(3))]);
        assert_eq!(big.get_small_scalar(&Value::Int(7)), None);
        assert_eq!(big.get(&Value::Int(7)), Some(&Value::Int(3)));
        assert_eq!(
            small
                .with_sort(Some(super::SortSpec::Natural))
                .get_small_scalar(&kw),
            None
        );

        let large = Map::from_kvs_last_wins(
            (0..20)
                .map(|i| (Value::Int(i), Value::Int(i * 10)))
                .collect(),
        );
        assert_eq!(large.len_opt(), Some(20));
        assert_eq!(large.get_small_scalar(&Value::Int(0)), None);
    }

    #[test]
    fn packed_strings_preserve_order_and_bounds_across_repeated_runs() {
        use super::{PackedStrings, Vector};

        let prefix = "é".repeat(2048);
        let first = "first repeated string";
        let second = "another repeated string";
        let mut items = vec![
            prefix.as_str(),
            first,
            first,
            "",
            second,
            second,
            second,
            "last",
        ];
        let mut packed = PackedStrings::new();
        for text in &items {
            assert!(packed.try_push(text));
        }
        assert_eq!(packed.runs.as_ref().unwrap().entries.len(), 2);
        assert_eq!(packed.ends.len(), 5);
        assert_eq!(packed.len(), items.len());
        assert_eq!(
            packed.to_values(),
            items
                .iter()
                .map(|s| Value::Str(Arc::from(*s)))
                .collect::<Vec<_>>()
        );
        assert_eq!(packed.get_str(items.len()), None);
        assert_eq!(packed.raw_buf(), items.concat());
        assert_eq!(packed.clone().into_joined("·"), items.join("·"));
        let vector = Vector::from_packed_strs(packed.clone());
        let tail: Vec<_> = vector.iter_from(6).collect();
        assert_eq!(
            tail,
            [Value::Str(Arc::from(second)), Value::Str(Arc::from("last"))]
        );
        assert!(vector.iter_from(items.len()).next().is_none());
        let mut text = String::new();
        assert!(vector.push_to_string(&mut text));
        assert_eq!(text, items.concat());

        let len = text.len();
        assert!(!packed.push_bounded(first, len));
        assert!(packed.try_push("after"));
        assert_ne!(packed.raw_buf(), text);
        assert!(packed.pop());
        while items.len() > 1 {
            assert_eq!(packed.raw_buf(), items.concat());
            assert!(packed.pop());
            items.pop();
        }
        assert!(packed.runs.is_none());
        assert_eq!(packed.raw_buf(), prefix);
    }

    #[test]
    fn packed_strings_vector_operations_and_bounds() {
        use super::{PackedStrings, ValueRef, Vector};
        let mut ps = PackedStrings::new();
        assert!(ps.is_empty());
        assert_eq!(ps.join(","), "");
        assert!(!ps.pop());
        assert!(!ps.push_bounded("abcd", 2));
        assert!(ps.try_push("aa"));
        assert!(ps.try_push("bbb"));
        assert!(ps.try_push("c"));
        assert_eq!(ps.len(), 3);
        assert!(!ps.is_empty());
        assert_eq!(ps.raw_buf(), "aabbbc");
        assert_eq!(ps.get_str(0), Some("aa"));
        assert_eq!(ps.get_str(1), Some("bbb"));
        assert_eq!(ps.get_str(3), None);
        assert_eq!(ps.join("-"), "aa-bbb-c");
        assert!(ps.pop());
        assert_eq!(ps.join("-"), "aa-bbb");

        let mut big_ps = PackedStrings::with_capacity(3000, 12000);
        for i in 0..1030 {
            assert!(big_ps.try_push(&i.to_string()));
        }
        big_ps.shed_slack();
        let mut grown = PackedStrings::with_capacity(2, 256 * 1024);
        let block = "a".repeat(256 * 1024);
        assert!(grown.try_push(&block));
        assert!(grown.try_push("b"));
        grown.shed_slack();
        let vs = Vector::from_packed_strs(big_ps);
        assert!(vs.is_packed());
        assert!(vs.as_packed_strs().is_some());
        assert!(vs.as_flat_ints().is_none());
        assert!(matches!(vs.get_val_ref(0), Some(ValueRef::Str("0"))));
        assert_eq!(vs.get(1), Some(Value::Str(Arc::from("1"))));
        assert_eq!(
            vs.read_at(2, |v| Some(v.clone())),
            Some(Value::Str(Arc::from("2")))
        );
        assert!(vs.get_ref(0).is_none());
        assert_eq!(vs.chunk_from(0).len(), 32);
        assert!(vs.chunk_from(2000).is_empty());
        let mut joined = String::new();
        assert!(vs.push_to_string(&mut joined));
        assert!(joined.starts_with("0123"));

        // Iter: next, nth, size_hint, next_back, exhausted
        let mut it = vs.iter();
        assert_eq!(it.size_hint(), (1030, Some(1030)));
        assert_eq!(it.next(), Some(Value::Str(Arc::from("0"))));
        assert_eq!(it.nth(1), Some(Value::Str(Arc::from("2"))));
        assert_eq!(it.next_back(), Some(Value::Str(Arc::from("1029"))));
        assert_eq!(it.nth(2000), None);
        assert_eq!(it.next_back(), None);

        // Shared > FLAT_MAX push_back and drop_last convert to generic
        let shared = vs.clone();
        let pushed_shared = vs.clone().push_back(Value::Str(Arc::from("extra")));
        assert!(!pushed_shared.is_packed());
        assert_eq!(pushed_shared.len(), 1031);
        let popped_shared = vs.clone().drop_last_owned().expect("popped");
        assert!(!popped_shared.is_packed());
        assert_eq!(popped_shared.len(), 1029);
        drop(shared);

        // Unique push_back (str + non-str), set_nth, drop_last, into_vec
        let mut small = PackedStrings::new();
        assert!(small.try_push("x"));
        let v1 = Vector::from_packed_strs(small);
        assert!(v1.typed_items().is_some());
        let v2 = v1.push_back(Value::Str(Arc::from("y")));
        assert!(v2.is_packed());
        assert_eq!(v2.len(), 2);
        let v3 = v2.clone().drop_last_owned().expect("pop");
        assert!(v3.is_packed());
        assert_eq!(v3.len(), 1);
        let v4 = v2.clone().set_nth_owned(0, Value::Int(7)).expect("set");
        assert!(!v4.is_packed());
        assert_eq!(v4.get(0), Some(Value::Int(7)));
        let v5 = v2.clone().push_back(Value::Int(9));
        assert!(!v5.is_packed());
        assert_eq!(v2.clone().into_vec().len(), 2);
        assert!(v2.into_typed_items().is_ok());

        let empty_v = Vector::from_packed_strs(PackedStrings::new());
        assert!(empty_v.as_flat_values().is_none());
        assert!(empty_v.drop_last_owned().is_none());

        // as_flat_values / as_flat_doubles: generic flat vs packed ints/doubles/chars/strs
        let fv = Vector::from_vec(vec![Value::Double(1.5), Value::Int(2)]);
        assert_eq!(fv.as_flat_values().map(<[Value]>::len), Some(2));
        assert!(fv.as_flat_doubles().is_none());
        let dv = Vector::from_vec(vec![Value::Double(1.5), Value::Double(2.5)]);
        assert_eq!(dv.as_flat_doubles(), Some([1.5, 2.5].as_slice()));
        assert!(dv.as_flat_values().is_none());
        assert!(Vector::from_ints(vec![1, 2]).as_flat_values().is_none());
        assert!(Vector::from_chars(vec!['a']).as_flat_values().is_none());

        // ends 1.5x growth past 64K entries
        let mut many_ends = PackedStrings::new();
        for _ in 0..=65_536 {
            assert!(many_ends.try_push("a"));
        }
        assert_eq!(many_ends.len(), 65_537);

        // into_joined: empty, empty sep, in-place (fits in cap), fallback (exceeds cap)
        assert_eq!(PackedStrings::new().into_joined(","), "");
        let mut p_empty_sep = PackedStrings::with_capacity(2, 8);
        assert!(p_empty_sep.try_push("hi"));
        assert!(p_empty_sep.try_push("lo"));
        assert_eq!(p_empty_sep.into_joined(""), "hilo");

        let mut p_inplace = PackedStrings::with_capacity(3, 16);
        assert!(p_inplace.try_push("aa"));
        assert!(p_inplace.try_push("bbb"));
        assert!(p_inplace.try_push("c"));
        assert_eq!(p_inplace.into_joined("-"), "aa-bbb-c");

        let mut p_grow = PackedStrings::with_capacity(2, 2);
        assert!(p_grow.try_push("a"));
        assert!(p_grow.try_push("b"));
        assert_eq!(p_grow.into_joined("--------"), "a--------b");

        // Vector::into_packed_strs: unique Ok, shared outer Err, shared inner via with_meta Err, non-Strs Err
        let mut ps_u = PackedStrings::new();
        assert!(ps_u.try_push("ok"));
        let vu = Vector::from_packed_strs(ps_u);
        assert_eq!(vu.into_packed_strs().expect("unique").join(","), "ok");

        let mut ps_s = PackedStrings::new();
        assert!(ps_s.try_push("s"));
        let vs_outer = Vector::from_packed_strs(ps_s);
        let vs_clone = vs_outer.clone();
        let vs_back = vs_outer.into_packed_strs().expect_err("shared outer");
        assert_eq!(vs_back.len(), 1);
        let vs_meta = vs_clone
            .clone()
            .with_meta(Some(crate::Map::from_kvs_last_wins(vec![])));
        let vs_meta_back = vs_meta.into_packed_strs().expect_err("shared inner");
        assert_eq!(vs_meta_back.len(), 1);
        drop(vs_clone);

        let non_strs = Vector::from_ints(vec![1, 2]);
        assert!(non_strs.into_packed_strs().is_err());
    }

    #[test]
    fn packed_doubles_vector_and_fast_eq_hash() {
        use super::{List, PackedStrings, TypedItems, Value, ValueRef, Vector};

        // Empty push retypes into Doubles; push/set preserve or widen cleanly.
        let d0 = Vector::new_sync().push_back(Value::Double(1.25));
        assert_eq!(d0.as_flat_doubles(), Some([1.25].as_slice()));
        let d1 = d0.push_back(Value::Double(2.5));
        assert_eq!(d1.as_flat_doubles(), Some([1.25, 2.5].as_slice()));
        assert_eq!(d1.get(1), Some(Value::Double(2.5)));
        assert_eq!(d1.get(9), None);
        assert!(matches!(d1.get_val_ref(0), Some(ValueRef::Double(x)) if x == 1.25));
        assert_eq!(d1.read_at(1, |v| Some(v.clone())), Some(Value::Double(2.5)));
        assert_eq!(d1.read_at(9, |_| Some(Value::Nil)), None);
        assert_eq!(
            d1.chunk_from(0),
            vec![Value::Double(1.25), Value::Double(2.5)]
        );
        assert!(matches!(d1.typed_items(), Some(TypedItems::Doubles(_))));

        // Double-ended iterator + size_hint + nth on Doubles
        let mut it = d1.iter();
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.next_back(), Some(Value::Double(2.5)));
        assert_eq!(it.next(), Some(Value::Double(1.25)));
        assert_eq!(it.next(), None);
        assert_eq!(d1.iter().nth(1), Some(Value::Double(2.5)));

        // set_nth_owned: same-type keeps Doubles, foreign widens to Generic
        let d2 = d1
            .clone()
            .set_nth_owned(0, Value::Double(3.75))
            .expect("set");
        assert_eq!(d2.as_flat_doubles(), Some([3.75, 2.5].as_slice()));
        let g_set = d1
            .clone()
            .set_nth_owned(0, Value::Int(7))
            .expect("widen set");
        assert!(g_set.as_flat_values().is_some());
        let g_push = d1.clone().push_back(Value::Int(9));
        assert!(g_push.as_flat_values().is_some());

        // Ints/Chars widening via set_inner / push_inner
        let i_widen = Vector::from_ints(vec![1])
            .set_nth_owned(0, Value::Double(1.0))
            .expect("i");
        assert!(i_widen.as_flat_values().is_some());
        let c_widen = Vector::from_chars(vec!['a'])
            .set_nth_owned(0, Value::Double(1.0))
            .expect("c");
        assert!(c_widen.as_flat_values().is_some());

        // drop_last_owned, into_vec (unique and shared), into_typed_items (unique and shared)
        let d_popped = d1.clone().drop_last_owned().expect("pop");
        assert_eq!(d_popped.as_flat_doubles(), Some([1.25].as_slice()));
        let empty_d = d_popped.drop_last_owned().expect("pop to empty");
        assert!(empty_d.drop_last_owned().is_none());
        let d_shared = d1.clone();
        assert_eq!(d1.clone().into_vec().len(), 2);
        assert!(matches!(d1.into_typed_items(), Ok(TypedItems::Doubles(_))));
        assert_eq!(d_shared.clone().into_vec().len(), 2);
        assert!(matches!(
            d_shared.into_typed_items(),
            Ok(TypedItems::Doubles(_))
        ));

        // Shared push past FLAT_MAX = 1024 promotes to Trie (exercises non-flat as_flat_doubles, eq, hash)
        let base_d = Vector::from_doubles((0..1025).map(f64::from).collect());
        let big_d1 = base_d.clone().push_back(Value::Double(1025.0));
        let big_d2 = Vector::from_flat_doubles((0..1026).map(f64::from).collect());
        assert!(big_d1.as_flat_doubles().is_none());
        assert_eq!(big_d1, big_d2);
        assert_eq!(
            hash_of(&Value::Vector(big_d1)),
            hash_of(&Value::Vector(big_d2))
        );
        drop(base_d);

        let base_i = Vector::from_ints((0..1025_i64).collect());
        let big_i1 = base_i.clone().push_back(Value::Int(1025));
        let big_i2 = Vector::from_ints((0..1026_i64).collect());
        assert_eq!(big_i1, big_i2);
        assert_eq!(
            hash_of(&Value::Vector(big_i1)),
            hash_of(&Value::Vector(big_i2))
        );
        drop(base_i);

        // Fast eq & hash consistency across all 5 VectorInner variants + mismatched kinds/lengths
        let v_int = Vector::from_ints(vec![1, 2]);
        let v_dbl = Vector::from_doubles(vec![1.0, 2.0]);
        let v_chr = Vector::from_chars(vec!['a', 'b']);
        let v_gen = Vector::from_vec(vec![Value::Int(1), Value::Double(2.0)]);
        let mut ps = PackedStrings::new();
        assert!(ps.try_push("a"));
        assert!(ps.try_push("b"));
        let v_str = Vector::from_packed_strs(ps);
        let mut ps2 = PackedStrings::new();
        assert!(ps2.try_push("a"));
        assert!(ps2.try_push("b"));
        let v_str2 = Vector::from_packed_strs(ps2);

        assert_eq!(v_int, Vector::from_ints(vec![1, 2]));
        assert_eq!(v_dbl, Vector::from_doubles(vec![1.0, 2.0]));
        assert_eq!(v_chr, Vector::from_chars(vec!['a', 'b']));
        assert!(v_int.is_flat());
        assert!(v_dbl.is_flat());
        assert!(v_chr.is_flat());
        assert!(v_gen.is_flat());
        assert!(v_str.is_flat());
        assert_eq!(v_int.slice_packed(0, 1).as_flat_ints(), Some(&[1][..]));
        assert_eq!(v_dbl.slice_packed(1, 2).as_flat_doubles(), Some(&[2.0][..]));
        assert_eq!(v_chr.slice_packed(1, 2), Vector::from_chars(vec!['b']));
        assert_eq!(
            v_gen.slice_packed(0, 1),
            Vector::from_vec(vec![Value::Int(1)])
        );
        assert_eq!(
            v_str.slice_packed(1, 2),
            Vector::from_vec(vec![Value::Str(Arc::from("b"))])
        );
        let mut mut_v = v_int.clone();
        assert!(mut_v.set_nth_mut(1, Value::Int(9)));
        assert!(!mut_v.set_nth_mut(9, Value::Int(0)));
        assert_eq!(mut_v.as_flat_ints(), Some(&[1, 9][..]));
        assert_eq!(
            v_gen,
            Vector::from_vec(vec![Value::Int(1), Value::Double(2.0)])
        );
        assert_eq!(v_str, v_str2);
        assert_ne!(v_int, Vector::from_ints(vec![1]));
        assert_ne!(v_int, v_dbl);

        assert_eq!(
            hash_of(&Value::Vector(v_int)),
            hash_of(&Value::List(List::from_vec(vec![
                Value::Int(1),
                Value::Int(2)
            ])))
        );
        assert_eq!(
            hash_of(&Value::Vector(v_dbl)),
            hash_of(&Value::List(List::from_vec(vec![
                Value::Double(1.0),
                Value::Double(2.0)
            ])))
        );
        assert_eq!(
            hash_of(&Value::Vector(v_chr)),
            hash_of(&Value::List(List::from_vec(vec![
                Value::Char('a'),
                Value::Char('b')
            ])))
        );
        assert_eq!(
            hash_of(&Value::Vector(v_gen)),
            hash_of(&Value::List(List::from_vec(vec![
                Value::Int(1),
                Value::Double(2.0)
            ])))
        );
        assert_eq!(
            hash_of(&Value::Vector(v_str)),
            hash_of(&Value::List(List::from_vec(vec![
                Value::Str(Arc::from("a")),
                Value::Str(Arc::from("b")),
            ])))
        );
    }

    #[test]
    fn assign_frees_a_heap_slot_and_overwrites_a_scalar_one_271() {
        let held = Arc::<str>::from("held");
        let mut slot = Value::Str(Arc::clone(&held));
        slot.assign(Value::Int(1));
        assert_eq!(Arc::strong_count(&held), 1, "the old string was dropped");
        slot.assign(Value::Bool(true));
        assert!(matches!(slot, Value::Bool(true)));
    }

    #[test]
    fn small_set_scalar_lookups_match_full_equality_271() {
        use super::{Set, SortSpec};
        let kw = |n: &str| Value::Keyword(Keyword::new(None, n));
        let s = Set::from_items_checked(vec![
            Value::Int(1),
            kw("a"),
            Value::Char('x'),
            Value::Bool(true),
            Value::Nil,
            Value::Str("s".into()),
        ])
        .unwrap();
        // Hits return the stored member, each scalar kind alone.
        for k in [
            Value::Int(1),
            kw("a"),
            Value::Char('x'),
            Value::Bool(true),
            Value::Nil,
        ] {
            assert_eq!(s.get_small_scalar(&k), Some(s.get(&k)));
            assert!(s.get(&k).is_some());
        }
        // Misses: no other kind equals such a key.
        for k in [Value::Int(2), kw("b"), Value::Char('y'), Value::Bool(false)] {
            assert_eq!(s.get_small_scalar(&k), Some(None));
        }
        // Other key kinds take the full `=`.
        assert_eq!(s.get_small_scalar(&Value::Str("s".into())), None);
        // A big integer may equal a long key.
        let big = Set::from_items_checked(vec![Value::BigInt(Arc::new(7.into()))]).unwrap();
        assert_eq!(big.get_small_scalar(&Value::Int(7)), None);
        assert!(big.contains(&Value::Int(7)));
        // Sorted and hashed sets keep their own lookups.
        let sorted = s.clone().with_sort(Some(SortSpec::Natural));
        assert_eq!(sorted.get_small_scalar(&Value::Int(1)), None);
        let large = Set::from_items_checked((0..20).map(Value::Int).collect()).unwrap();
        assert_eq!(large.get_small_scalar(&Value::Int(1)), None);
        assert!(large.contains(&Value::Int(19)));
    }

    #[test]
    fn shared_long_flat_vectors_update_through_the_trie_271() {
        let n = crate::trie::FLAT_MAX + 1;
        for v in [
            Vector::from_flat_ints((0..n as i64).collect()),
            Vector::from_flat_doubles((0..n).map(|i| i as f64).collect()),
            Vector::from_flat_chars(vec!['a'; n]),
            Vector::from_vec(vec![Value::Nil; n]).with_meta(None),
        ] {
            let flat = v.is_packed().then(|| v.clone());
            let kept = v.clone();
            let first = v.get(0);
            // The shared update leaves `kept` alone and moves `next` into the
            // trie (no flat items left to read straight).
            let next = v.set_nth_owned(0, Value::Bool(true)).expect("in bounds");
            assert_eq!(kept.get(0), first);
            assert_eq!(next.get(0), Some(Value::Bool(true)));
            assert!(next.as_flat_ints().is_none());
            assert_eq!(next.len(), n);
            drop(flat);
        }
        // Strings keep their own storage; short flat runs copy.
        let s = Vector::from_packed_strs({
            let mut ps = super::PackedStrings::new();
            assert!(ps.try_push("x"));
            ps
        });
        let s2 = s.clone().push_back(Value::Str("y".into()));
        assert_eq!((s.len(), s2.len()), (1, 2));
        let short = Vector::from_flat_ints(vec![1, 2]);
        let kept = short.clone();
        let next = short.set_nth_owned(1, Value::Int(5)).expect("in bounds");
        assert_eq!(next.as_flat_ints(), Some(&[1, 5][..]));
        assert_eq!(kept.as_flat_ints(), Some(&[1, 2][..]));
    }

    #[test]
    fn unique_vector_updates_only_touch_unshared_flat_storage_271() {
        let kw = |n: &str| Value::Keyword(Keyword::new(None, n));
        // Longs: in place when unique, in bounds, and a long.
        let mut v = Vector::from_ints(vec![1, 2, 3]);
        let id = v.id();
        assert!(v.assoc_unique(1, &mut Value::Int(9)));
        assert_eq!(v.id(), id, "the identity stays");
        assert!(!v.assoc_unique(3, &mut Value::Int(9)), "past the end");
        assert!(!v.assoc_unique(0, &mut Value::Double(1.5)), "would retype");
        assert_eq!(v.iter().collect::<Vec<_>>(), [1, 9, 3].map(Value::Int));
        // A second handle or shared storage: nothing changes.
        let shared = v.clone();
        assert!(!v.assoc_unique(0, &mut Value::Int(7)));
        assert!(!v.conj_unique(&mut Value::Int(7)));
        drop(shared);
        // A new metadata view holds its own copy of the items (#271: the
        // flat run lives in the vector's data).
        let mut meta_view = v.clone().with_meta(None);
        assert!(meta_view.assoc_unique(0, &mut Value::Int(7)));
        assert_eq!(meta_view.get(0), Some(Value::Int(7)));
        assert_eq!(v.get(0), Some(Value::Int(1)));
        // Doubles.
        let mut d = Vector::from_doubles(vec![0.5, 1.5]);
        assert!(d.assoc_unique(0, &mut Value::Double(2.5)));
        assert!(!d.assoc_unique(0, &mut Value::Int(2)));
        assert_eq!(d.get(0), Some(Value::Double(2.5)));
        // Generic: replaces scalar items only, taking the value.
        let mut g = Vector::from_vec(vec![kw("a"), Value::Nil]);
        let mut x = kw("b");
        assert!(g.assoc_unique(1, &mut x));
        assert_eq!(x, Value::Nil, "the value moved in");
        assert!(
            !g.assoc_unique(0, &mut Value::Int(1)),
            "a heap item would drop"
        );
        assert!(!g.assoc_unique(5, &mut Value::Int(1)));
        assert_eq!(g.get(1), Some(kw("b")));
        // Other storages and the trie decline.
        let mut c = Vector::from_chars(vec!['a']);
        assert!(!c.assoc_unique(0, &mut Value::Char('b')));
        assert!(!c.conj_unique(&mut Value::Char('b')));
        let mut big = Vector::from_ints(vec![0; 1024]).push_back(Value::Int(0));
        assert!(!big.assoc_unique(0, &mut Value::Int(1)));
        // conj: within capacity only, never on empty vectors.
        let mut e = Vector::from_ints(Vec::new());
        assert!(!e.conj_unique(&mut Value::Int(1)));
        let mut p = Vector::from_ints(Vec::with_capacity(4));
        p = p.push_back(Value::Int(1));
        assert!(p.conj_unique(&mut Value::Int(2)));
        assert!(!p.conj_unique(&mut Value::Double(2.0)));
        let mut pd = Vector::from_doubles(Vec::with_capacity(4)).push_back(Value::Double(1.0));
        assert!(pd.conj_unique(&mut Value::Double(2.0)));
        let mut full = Vector::from_ints(vec![1]);
        while full.conj_unique(&mut Value::Int(0)) {}
        let at_cap = full.len();
        full = full.push_back(Value::Int(0));
        assert_eq!(full.len(), at_cap + 1, "the general path grows it");
        let mut pg = Vector::from_vec(Vec::with_capacity(4)).push_back(kw("a"));
        let mut y = kw("c");
        assert!(pg.conj_unique(&mut y));
        assert_eq!(y, Value::Nil);
        let mut z = kw("z");
        while pg.conj_unique(&mut z) {
            z = kw("z");
        }
        assert_eq!(z, kw("z"), "a full generic run keeps the value");
        assert_eq!(p.iter().collect::<Vec<_>>(), [1, 2].map(Value::Int));
        let mut flat_max = Vector::from_ints(vec![0; 1023]).push_back(Value::Int(0));
        assert!(
            !flat_max.conj_unique(&mut Value::Int(1)),
            "a full flat run moves to the trie"
        );
    }

    #[test]
    fn small_and_large_set_storage_paths_271() {
        use super::{PackedStrings, Set, SortSpec, Vector};
        // Small set: from_items_checked, insert, dup insert, get, contains, without, iter
        let s = Set::from_items_checked(vec![Value::Int(1), Value::Int(2)]).unwrap();
        assert_eq!(s.size(), 2);
        assert!(!s.is_empty());
        assert!(s.contains(&Value::Int(1)));
        assert_eq!(s.get(&Value::Int(2)), Some(&Value::Int(2)));
        assert_eq!(s.get(&Value::Int(9)), None);
        let s_dup = s.clone().insert(Value::Int(2));
        assert_eq!(s_dup.id(), s.id());
        let s_miss = s.clone().without(&Value::Int(99));
        assert_eq!(s_miss.id(), s.id());
        let s_rem = s.clone().without(&Value::Int(1));
        assert_eq!(s_rem.size(), 1);
        assert!(Set::from_items_checked(vec![Value::Int(1), Value::Int(1)]).is_err());

        // Promotion > SMALL_SET_MAX via insert and via from_items_checked
        let mut big = new_set();
        for i in 0..10 {
            big = big.insert(Value::Int(i));
        }
        assert_eq!(big.size(), 10);
        assert_eq!(big.get(&Value::Int(9)), Some(&Value::Int(9)));
        assert_eq!(big.iter().count(), 10);
        assert_eq!(big.clone().without(&Value::Int(99)).id(), big.id());
        let mut shrunk = big.clone();
        for i in 0..7 {
            shrunk = shrunk.without(&Value::Int(i));
        }
        assert_eq!(shrunk.size(), 3);
        let big2 = Set::from_items_checked((0..10).map(Value::Int).collect()).unwrap();
        assert_eq!(big, big2);

        // Small & promoted sorted sets
        let mut ss = new_set().with_sort(Some(SortSpec::Natural));
        for i in 0..10 {
            ss = ss.insert_sorted_at(i, Value::Int(i as i64));
        }
        assert_eq!(ss.sorted_len(), 10);
        assert_eq!(ss.sorted_member(0), Some(&Value::Int(0)));
        let ss_sm = new_set()
            .with_sort(Some(SortSpec::Natural))
            .with_ordered_members(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert_eq!(ss_sm.sorted_len(), 3);
        assert_eq!(ss_sm.sorted_member(1), Some(&Value::Int(2)));
        let ss_sm2 = ss_sm.without_sorted_at(1);
        assert_eq!(ss_sm2.sorted_len(), 2);
        let mut ss_lg = ss;
        for i in 0..7 {
            ss_lg = ss_lg.without(&Value::Int(i));
        }
        ss_lg = ss_lg.without(&Value::Int(8));
        assert_eq!(ss_lg.sorted_len(), 2);
        let ss_rebuilt = new_set().with_ordered_members((0..10).map(Value::Int).collect());
        assert_eq!(ss_rebuilt.size(), 10);

        // String-packed Vector::iter_from and PackedStrings::into_joined reserve path
        let mut ps = PackedStrings::new();
        assert!(ps.try_push("a"));
        assert!(ps.try_push("b"));
        assert!(ps.try_push("c"));
        assert_eq!(ps.join("--"), "a--b--c");
        let ps_grow = ps.clone();
        assert_eq!(ps_grow.into_joined("----"), "a----b----c");
        let sv = Vector::from_packed_strs(ps);
        assert_eq!(sv.iter_from(1).count(), 2);
    }
}
