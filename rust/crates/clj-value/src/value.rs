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

/// Identity counters for `identical?` (S16): every freshly
/// constructed [`List`]/[`Vector`]/[`Set`] mints an id from its
/// counter (the `ATOM_ID` precedent); `Clone` preserves it, so two
/// evaluations of the same binding compare identical while two
/// separately built collections never do. Equality and hashing
/// ignore the ids. [`Map`] instead identifies by its entries `Arc`
/// (see [`Map::id`]) — no counter, no field.
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

/// A persistent list of values (thread-safe rpds list) with optional
/// metadata. Method names match the bare rpds type so readers keep
/// compiling; equality and hashing ignore the metadata.
#[derive(Clone, Debug)]
pub struct List {
    inner: RpdsList,
    meta: Option<Arc<Map>>,
    id: u64,
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
#[derive(Clone, Debug)]
pub struct Vector {
    inner: VectorInner,
    meta: Option<Arc<Map>>,
    id: u64,
}

/// Vector storage: the generic trie of values, or the same trie over
/// packed `i64`s or `char`s (#226: 8 or 4 bytes per element instead
/// of a 64-byte `Value` — a vector of a string's chars was 16x its
/// text). Typing is decided at construction (`from_vec` scans,
/// `from_ints`/`from_chars` and typed transients skip the scan) and by
/// the first item into an empty vector; a foreign value converts once
/// to generic, which then stays generic. All three are the same
/// persistent trie, so shared updates path-copy in log time
/// whatever the type.
#[derive(Clone, Debug)]
enum VectorInner {
    Generic(RpdsVector),
    Ints(crate::trie::Packed<i64>),
    Chars(crate::trie::Packed<char>),
}

/// Owned-value vector iteration (typed items box on read).
pub struct VectorIter<'a> {
    inner: VectorIterInner<'a>,
}

enum VectorIterInner<'a> {
    Generic(crate::trie::Iter<'a>),
    Ints(crate::trie::PackedIter<'a, i64>),
    Chars(crate::trie::PackedIter<'a, char>),
}

impl Iterator for VectorIter<'_> {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        match &mut self.inner {
            VectorIterInner::Generic(it) => it.next().cloned(),
            VectorIterInner::Ints(it) => it.next().map(|n| Value::Int(*n)),
            VectorIterInner::Chars(it) => it.next().map(|c| Value::Char(*c)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            VectorIterInner::Generic(it) => it.size_hint(),
            VectorIterInner::Ints(it) => it.size_hint(),
            VectorIterInner::Chars(it) => it.size_hint(),
        }
    }
}

impl DoubleEndedIterator for VectorIter<'_> {
    fn next_back(&mut self) -> Option<Value> {
        match &mut self.inner {
            VectorIterInner::Generic(it) => it.next_back().cloned(),
            VectorIterInner::Ints(it) => it.next_back().map(|n| Value::Int(*n)),
            VectorIterInner::Chars(it) => it.next_back().map(|c| Value::Char(*c)),
        }
    }
}

/// A persistent map of values (thread-safe rpds hash-trie map) with
/// optional metadata. See [`List`].
///
/// `keys` is the insertion-order sidecar: the JVM's small maps are
/// insertion-ordered array-maps, and that order is observable —
/// syntax-quote flattens maps in seq order, and `pr` prints small
/// maps in insertion order. Re-inserting an existing key keeps its
/// original position (and key object), like `assoc`. (Big JVM maps
/// iterate in murmur-hash order, which no insertion record can
/// reproduce; the harness canonicalizes map/set order away, so only
/// syntax-quote expansion and rendering observe this.)
///
/// The sidecar rides behind an `Arc` (writes go through
/// `Arc::make_mut`, which clones the O(1) persistent header when
/// shared): the rpds vector header is 24 bytes, and every `Value`
/// pays the largest variant's size inline (#221 R1). Cloning a map
/// stays allocation-free, like the `meta` arc beside it.
///
/// `hashy` reports `PersistentHashMap` instead of
/// `PersistentArrayMap`: promotion is one-way and history-dependent
/// (content alone cannot decide — `dissoc` never demotes and
/// `array-map` forces array), so it is stored, not derived.
///
/// `sort` carries the FT6 sorted-map order: `None` is an ordinary
/// map; `Some` keeps `keys` in comparator order instead of
/// insertion order (compare-0 duplicates collapse at insertion,
/// like repeated `assoc`). The inner trie stays content-exact, so
/// equality, hashing, printing, and seqs need no sorted branch —
/// only order-sensitive writers (`assoc` position) and readers
/// (`get`/`dissoc` by compare-0 key, `jvm_class_of`, `nth`,
/// `sorted?`) consult it.
/// Entries at or below this count ride shaped (an interned
/// key-set plus a flat values vec, no HAMT or sidecar allocs) —
/// the JVM's `PersistentArrayMap` converts past 16 pairs.
/// Representation is orthogonal to the reported class
/// (`hashy`): a 9-pair all-keyword map stays array-classed
/// whether shaped or hashed.
const INLINE_MAP_MAX: usize = 16;

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
}

/// A small map's entries: the shared shape plus this version's
/// values (parallel to `shape.keys`). Boxed slices, not vecs: no
/// capacity field (a 3-pair map buckets at 256B instead of 288B)
/// and no growth slack ever.
#[derive(Debug)]
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
    /// The last all-keyword shape interned: runs of same-shape maps
    /// (every fresh var's `{:ns … :name …}`) skip the hash probe.
    /// Kept alive here, so its table entry never goes stale.
    last: Option<Arc<MapShape>>,
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

thread_local! {
    static SHAPE_REGISTRY: RefCell<ShapeRegistry> = RefCell::new(ShapeRegistry {
        map: HashMap::default(),
        since_cleanup: 0,
        last: None,
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
        // Keyword keys compare unambiguously (no cross-type `=`), so
        // the shortcut agrees with the table's own `Eq`.
        let keywords = keys.iter().all(|k| matches!(k, Value::Keyword(_)));
        if keywords {
            if let Some(last) = reg.last.as_ref().filter(|l| *l.keys == *keys) {
                return Arc::clone(last);
            }
        }
        if let Some(shape) = reg.map.get(keys).and_then(Weak::upgrade) {
            if keywords {
                reg.last = Some(Arc::clone(&shape));
            }
            return shape;
        }
        let shape = Arc::new(MapShape {
            keys: keys.to_vec().into_boxed_slice(),
        });
        reg.map.insert(keys.to_vec(), Arc::downgrade(&shape));
        if keywords {
            reg.last = Some(Arc::clone(&shape));
        }
        reg.since_cleanup += 1;
        if reg.since_cleanup >= SHAPE_CLEANUP_EVERY
            && reg.since_cleanup.saturating_mul(SHAPE_CLEANUP_AMORT) >= reg.map.len()
        {
            reg.map.retain(|_, weak| weak.strong_count() > 0);
            reg.since_cleanup = 0;
        }
        shape
    })
}

/// Shared dummies for the authority a map isn't using: shaped
/// maps hold `inner: None` and ignore `keys`, hash maps hold
/// `small: None`, so small maps never allocate a HAMT root or a
/// keys vector. The empty shape is shared, but every empty map
/// carries a FRESH [`SmallMap`] allocation: identity is the entries
/// `Arc` (see [`Map::id`]), so sharing one static would identify
/// all empties.
static EMPTY_MAP_KEYS: LazyLock<Arc<RpdsVector>> =
    LazyLock::new(|| Arc::new(RpdsVector::new_sync()));
static EMPTY_MAP_SHAPE: LazyLock<Arc<MapShape>> = LazyLock::new(|| intern_shape(&[]));

/// A fresh empty entries allocation (identity-fresh; the shape is
/// the shared empty shape).
fn fresh_empty_small() -> Arc<SmallMap> {
    Arc::new(SmallMap {
        shape: EMPTY_MAP_SHAPE.clone(),
        vals: Box::default(),
    })
}

#[derive(Clone, Debug)]
pub struct Map {
    /// The entries when `small` is `None` (`None` for shaped maps
    /// — no dummy trie. Boxed (not inline) so `Map` fits 48 bytes
    /// and `Value` fits 64).
    inner: Option<Box<RpdsMap>>,
    /// `Some` = shaped authoritative (an interned key-set plus
    /// this version's values, at most [`INLINE_MAP_MAX`] entries
    /// in iteration order); `None` = `inner`/`keys`
    /// authoritative. Writers mint a fresh entries `Arc` except
    /// uniquely-owned in-place updates (see [`Map::id`]).
    small: Option<Arc<SmallMap>>,
    /// Insertion-order sidecar when `small` is `None` (a
    /// shared-empty dummy otherwise). Writers mint a fresh
    /// sidecar `Arc` except uniquely-owned in-place updates
    /// (see [`Map::id`]).
    keys: Arc<RpdsVector>,
    meta: Option<Arc<Map>>,
    hashy: bool,
    /// Boxed: sorted maps are rare, and the 8 bytes this saves
    /// keep `Map` — inline in `Value` — at 48 bytes (the R1
    /// budget, down from 72: identity rides the entries `Arc`
    /// instead of a minted field, and the trie rides boxed).
    sort: Option<Box<SortSpec>>,
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
#[derive(Clone, Debug)]
pub struct Set {
    inner: RpdsSet,
    meta: Option<Arc<Map>>,
    members: Arc<RpdsVector>,
    /// Boxed (like [`Map`]'s): sorted sets are rare, and this
    /// keeps `Set` at 56 bytes so `Value` fits 64.
    sort: Option<Box<SortSpec>>,
    id: u64,
}

impl List {
    /// An empty list.
    #[must_use]
    pub fn new_sync() -> Self {
        Self {
            inner: RpdsList::new_sync(),
            meta: None,
            id: next_list_id(),
        }
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
        Self {
            inner: self.inner.push_front(v),
            meta: self.meta,
            id: next_list_id(),
        }
    }

    /// The first value, if any.
    #[must_use]
    pub fn first(&self) -> Option<&Value> {
        self.inner.iter().next()
    }

    /// The list without its first value (`None` when empty;
    /// metadata is preserved — `rest` strips it separately).
    #[must_use]
    pub fn drop_first(&self) -> Option<Self> {
        self.inner.drop_first().map(|inner| Self {
            inner,
            meta: self.meta.clone(),
            id: next_list_id(),
        })
    }

    /// Iterate front to back.
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.inner.iter()
    }

    /// The element count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_deref()
    }

    /// Return this list with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news), so a new id.
        self.id = next_list_id();
        self
    }

    /// The allocation identity for `identical?` (S16): minted at
    /// every construction, preserved by `Clone`.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Vector {
    /// An empty vector (typed by its first item: `conj` chains over
    /// `Int`s or chars stay packed from the first push).
    #[must_use]
    pub fn new_sync() -> Self {
        Self::of(VectorInner::Ints(crate::trie::Packed::from_vec(Vec::new())))
    }

    fn of(inner: VectorInner) -> Self {
        Self {
            inner,
            meta: None,
            id: next_vector_id(),
        }
    }

    /// Build a vector from items in order (packed when every item is
    /// an `Int`, or every item a char; empty vectors start typed for
    /// their first `conj`).
    #[must_use]
    pub fn from_vec(items: Vec<Value>) -> Self {
        let typed = match items.first() {
            None => return Self::new_sync(),
            Some(Value::Int(_)) => items
                .iter()
                .map(|v| {
                    if let Value::Int(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<i64>>>()
                .map(Self::from_ints),
            Some(Value::Char(_)) => items
                .iter()
                .map(|v| {
                    if let Value::Char(c) = v {
                        Some(*c)
                    } else {
                        None
                    }
                })
                .collect::<Option<Vec<char>>>()
                .map(Self::from_chars),
            Some(_) => None,
        };
        typed.unwrap_or_else(|| Self::of(VectorInner::Generic(RpdsVector::from_vec(items))))
    }

    /// Build a packed `Int` vector from `i64`s in order (no scan — for
    /// callers that already know, like ranges and packed transients).
    #[must_use]
    pub fn from_ints(ints: Vec<i64>) -> Self {
        Self::of(VectorInner::Ints(crate::trie::Packed::from_vec(ints)))
    }

    /// Build a packed char vector from `char`s in order (no scan).
    #[must_use]
    pub fn from_chars(chars: Vec<char>) -> Self {
        Self::of(VectorInner::Chars(crate::trie::Packed::from_vec(chars)))
    }

    /// Whether this vector rides on packed `i64` or `char` storage.
    #[must_use]
    pub fn is_packed(&self) -> bool {
        !matches!(self.inner, VectorInner::Generic(_))
    }

    /// The packed items when typed (for transients that freeze back
    /// to the same type without re-scanning).
    pub(crate) fn typed_items(&self) -> Option<TypedItems> {
        match &self.inner {
            VectorInner::Generic(_) => None,
            VectorInner::Ints(t) => Some(TypedItems::Ints(t.iter().copied().collect())),
            VectorInner::Chars(t) => Some(TypedItems::Chars(t.iter().copied().collect())),
        }
    }

    /// An empty typed vector takes the type of a packable first item.
    fn retype_empty(&mut self, v: &Value) {
        if !self.is_packed() || !self.is_empty() {
            return;
        }
        match (&self.inner, v) {
            (VectorInner::Ints(_), Value::Char(_)) => {
                self.inner = VectorInner::Chars(crate::trie::Packed::from_vec(Vec::new()));
            }
            (VectorInner::Chars(_), Value::Int(_)) => {
                self.inner = VectorInner::Ints(crate::trie::Packed::from_vec(Vec::new()));
            }
            _ => {}
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
        let generic = match (&mut self.inner, v) {
            (VectorInner::Generic(t), v) => {
                t.push_back_mut(v);
                None
            }
            (VectorInner::Ints(t), Value::Int(n)) => {
                t.push(n);
                None
            }
            (VectorInner::Chars(t), Value::Char(c)) => {
                t.push(c);
                None
            }
            (VectorInner::Ints(t), v) => Some(boxed(t, Value::Int, |g| g.push_back_mut(v))),
            (VectorInner::Chars(t), v) => Some(boxed(t, Value::Char, |g| g.push_back_mut(v))),
        };
        if let Some(g) = generic {
            self.inner = VectorInner::Generic(g);
        }
        self.id = next_vector_id();
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

    /// Replace the item at `index` on an owned vector (`None` when
    /// out of bounds — `self` is consumed either way, so check
    /// [`len`](Self::len) first when the vector must survive).
    ///
    /// The #212 pattern (`mut self` + `set_mut`): the trie
    /// path-copies only nodes shared with *other* versions, so
    /// threading one accumulator through an `assoc` chain mutates in
    /// place after the first step. A foreign value converts typed
    /// storage once, like [`push_back`](Self::push_back).
    #[must_use]
    pub fn set_nth_owned(mut self, index: usize, v: Value) -> Option<Self> {
        if index >= self.len() {
            return None;
        }
        let generic = match (&mut self.inner, v) {
            (VectorInner::Generic(t), v) => {
                t.set_mut(index, v);
                None
            }
            (VectorInner::Ints(t), Value::Int(n)) => {
                t.set(index, n);
                None
            }
            (VectorInner::Chars(t), Value::Char(c)) => {
                t.set(index, c);
                None
            }
            (VectorInner::Ints(t), v) => Some(boxed(t, Value::Int, |g| {
                g.set_mut(index, v);
            })),
            (VectorInner::Chars(t), v) => Some(boxed(t, Value::Char, |g| {
                g.set_mut(index, v);
            })),
        };
        if let Some(g) = generic {
            self.inner = VectorInner::Generic(g);
        }
        self.id = next_vector_id();
        Some(self)
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
        let popped = match &mut self.inner {
            VectorInner::Generic(t) => t.drop_last_mut(),
            VectorInner::Ints(t) => t.pop(),
            VectorInner::Chars(t) => t.pop(),
        };
        if !popped {
            return None;
        }
        self.id = next_vector_id();
        Some(self)
    }

    /// Iterate front to back (owned values; typed items box on read).
    pub fn iter(&self) -> VectorIter<'_> {
        VectorIter {
            inner: match &self.inner {
                VectorInner::Generic(t) => VectorIterInner::Generic(t.iter()),
                VectorInner::Ints(t) => VectorIterInner::Ints(t.iter()),
                VectorInner::Chars(t) => VectorIterInner::Chars(t.iter()),
            },
        }
    }

    /// The item at `index` (`None` past the end; logarithmic — indexed
    /// callers must use this, not `iter().nth()`).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        match &self.inner {
            VectorInner::Generic(t) => t.get(index).cloned(),
            VectorInner::Ints(t) => t.get(index).map(Value::Int),
            VectorInner::Chars(t) => t.get(index).map(Value::Char),
        }
    }

    /// The items from `index` to the end of its 32-item chunk (the
    /// JVM `ChunkedSeq` chunk; empty past the end): borrowed from
    /// generic storage, boxed from typed.
    #[must_use]
    pub fn chunk_from(&self, index: usize) -> std::borrow::Cow<'_, [Value]> {
        use std::borrow::Cow;
        match &self.inner {
            VectorInner::Generic(t) => Cow::Borrowed(t.chunk_from(index)),
            VectorInner::Ints(t) => {
                Cow::Owned(t.chunk_from(index).iter().map(|n| Value::Int(*n)).collect())
            }
            VectorInner::Chars(t) => Cow::Owned(
                t.chunk_from(index)
                    .iter()
                    .map(|c| Value::Char(*c))
                    .collect(),
            ),
        }
    }

    /// The element count.
    #[must_use]
    pub fn len(&self) -> usize {
        vector_inner_len(&self.inner)
    }

    /// Whether the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_deref()
    }

    /// Return this vector with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news), so a new id.
        self.id = next_vector_id();
        self
    }

    /// The allocation identity for `identical?` (S16): minted at
    /// every construction, preserved by `Clone`.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// The item count of a vector storage.
fn vector_inner_len(inner: &VectorInner) -> usize {
    match inner {
        VectorInner::Generic(t) => t.len(),
        VectorInner::Ints(t) => t.len(),
        VectorInner::Chars(t) => t.len(),
    }
}

/// Typed storage as a generic trie (`box_item` boxes each item),
/// then `then` applied: the one-time conversion when a foreign value
/// meets packed storage.
fn boxed<T: Copy>(
    t: &crate::trie::Packed<T>,
    box_item: fn(T) -> Value,
    then: impl FnOnce(&mut RpdsVector),
) -> RpdsVector {
    let mut g: RpdsVector = t.iter().map(|x| box_item(*x)).collect();
    then(&mut g);
    g
}

/// A typed vector's items, copied out for a transient.
pub(crate) enum TypedItems {
    Ints(Vec<i64>),
    Chars(Vec<char>),
}

/// Entries in iteration order (insertion order, or comparator
/// order when sorted) — one concrete type over both
/// representations so `Map::iter` stays allocation-free.
pub struct MapIter<'a> {
    inner: MapIterInner<'a>,
}

enum MapIterInner<'a> {
    Shaped {
        keys: std::slice::Iter<'a, Value>,
        vals: std::slice::Iter<'a, Value>,
    },
    Hash(HashIter<'a>),
}

/// Hash-map iteration: position over the keys sidecar, resolving
/// each key against the trie (every writer keeps both in sync,
/// so every key resolves).
struct HashIter<'a> {
    keys: &'a RpdsVector,
    map: &'a RpdsMap,
    pos: usize,
}

impl<'a> Iterator for HashIter<'a> {
    type Item = (&'a Value, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.keys.len() {
            return None;
        }
        let k = self.keys.get(self.pos)?;
        self.pos += 1;
        // Every writer keeps the sidecar and the trie in sync, so
        // every iterated key resolves.
        let v = self.map.get(k).expect("keys sidecar and trie stay in sync");
        Some((k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.keys.len().saturating_sub(self.pos)))
    }
}

impl<'a> Iterator for MapIter<'a> {
    type Item = (&'a Value, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            // Keys and vals stay parallel by construction, so
            // both sides end together; `zip` keeps the tail
            // from over-running on any divergence.
            MapIterInner::Shaped { keys, vals } => keys.next().zip(vals.next()),
            MapIterInner::Hash(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            MapIterInner::Shaped { keys, vals } => {
                let (lk, uk) = keys.size_hint();
                let (lv, uv) = vals.size_hint();
                (lk.min(lv), uk.zip(uv).map(|(a, b)| a.min(b)))
            }
            MapIterInner::Hash(it) => it.size_hint(),
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
        self.inner.as_deref().expect("hash map holds a trie")
    }

    /// The hash trie, mutably (hash arm only).
    fn trie_mut(&mut self) -> &mut RpdsMap {
        self.inner.as_deref_mut().expect("hash map holds a trie")
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
        Self {
            inner: None,
            small: Some(Arc::new(SmallMap {
                shape: intern_shape(keys),
                vals: vals.into_boxed_slice(),
            })),
            keys: EMPTY_MAP_KEYS.clone(),
            meta: None,
            hashy: false,
            sort: None,
        }
    }

    /// Build from deduped parallel keys/vals (big maps take the
    /// hash trie directly; small ones intern one shape).
    fn from_parts(keys: Vec<Value>, vals: Vec<Value>, hashy: bool) -> Self {
        if keys.len() > INLINE_MAP_MAX {
            let mut inner = RpdsMap::new_with_hasher_and_ptr_kind(FxBuildHasher::default());
            let mut keys_side = RpdsVector::new_sync();
            for (k, v) in keys.into_iter().zip(vals) {
                keys_side.push_back_mut(k.clone());
                inner.insert_mut(k, v);
            }
            return Self {
                inner: Some(Box::new(inner)),
                small: None,
                keys: Arc::new(keys_side),
                meta: None,
                hashy,
                sort: None,
            };
        }
        if keys.is_empty() {
            return new_map().with_hashy(hashy);
        }
        Self {
            inner: None,
            small: Some(Arc::new(SmallMap {
                shape: intern_shape(&keys),
                vals: vals.into_boxed_slice(),
            })),
            keys: EMPTY_MAP_KEYS.clone(),
            meta: None,
            hashy,
            sort: None,
        }
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
    /// Uniquely-owned storage mutates with no fresh entries or
    /// sidecar `Arc` (same [`id`](Self::id) — unobservable: the
    /// in-place arms run only with no live alias, and the consumed
    /// input is dead); shared storage mints fresh `Arc`s exactly
    /// as before, so live aliases keep their versions and
    /// identities.
    pub fn insert_mut(&mut self, k: Value, v: Value) {
        if let Some(small) = self.small.take() {
            return self.insert_shaped_mut(small, k, v);
        }
        // Owned `self`: mutate the trie in place (same rationale as
        // `Vector::push_back` — rpds still path-copies nodes shared
        // with *other* versions, so persistence is preserved).
        if self.trie().get(&k).is_some() {
            self.trie_mut().insert_mut(k, v);
            // Fresh identity only when shared: a live alias must
            // keep its sidecar `Arc` (see [`Map::id`]).
            if Arc::strong_count(&self.keys) > 1 {
                self.keys = Arc::new((*self.keys).clone());
            }
            return;
        }
        let size = self.trie().size();
        let promote = !self.hashy && (size >= 64 || (size >= 8 && !matches!(k, Value::Keyword(_))));
        // `make_mut`: in place when unique; a header clone plus a
        // path copy when shared (exactly the old clone-then-push).
        Arc::make_mut(&mut self.keys).push_back_mut(k.clone());
        self.trie_mut().insert_mut(k, v);
        self.hashy = self.hashy || promote;
    }

    /// The shaped arm of [`Map::insert_mut`] (linear scan — the `==`
    /// relation agrees with the hash trie's, so both arms find
    /// exactly the same keys). Uniquely-owned same-key assocs
    /// write the value in place (no allocs); shared same-key
    /// assocs allocate only the values vec (the shape `Arc` is
    /// shared); new keys intern the extended key-set (hits
    /// allocate nothing past values).
    fn insert_shaped_mut(&mut self, mut small: Arc<SmallMap>, k: Value, v: Value) {
        if let Some(idx) = small.shape.keys.iter().position(|ek| ek == &k) {
            // `get_mut` succeeds only with no live alias (not even
            // the consumed input survives), so the write is
            // invisible and the entries `Arc` (see [`Map::id`])
            // stays put.
            if let Some(entries) = Arc::get_mut(&mut small) {
                entries.vals[idx] = v;
                self.small = Some(small);
                return;
            }
            let mut vals = Vec::with_capacity(small.vals.len());
            vals.extend(small.vals.iter().cloned());
            vals[idx] = v;
            self.small = Some(Arc::new(SmallMap {
                shape: Arc::clone(&small.shape),
                vals: vals.into_boxed_slice(),
            }));
            return;
        }
        let size = small.vals.len();
        let promote = !self.hashy && (size >= 64 || (size >= 8 && !matches!(k, Value::Keyword(_))));
        self.hashy = self.hashy || promote;
        if size < INLINE_MAP_MAX {
            // Exact-size: the new version is frozen, so no growth
            // slack (see [`SmallMap`]).
            let mut keys = Vec::with_capacity(size + 1);
            keys.extend(small.shape.keys.iter().cloned());
            keys.push(k);
            let mut vals = Vec::with_capacity(size + 1);
            vals.extend(small.vals.iter().cloned());
            vals.push(v);
            self.small = Some(Arc::new(SmallMap {
                shape: intern_shape(&keys),
                vals: vals.into_boxed_slice(),
            }));
            return;
        }
        self.promote_with_mut(&small, k, v);
    }

    /// Convert a shaped map to the hash trie, then insert the
    /// overflowing pair (insertion order preserved through `keys`).
    fn promote_with_mut(&mut self, small: &SmallMap, k: Value, v: Value) {
        let mut inner = RpdsMap::new_with_hasher_and_ptr_kind(FxBuildHasher::default());
        let mut keys = RpdsVector::new_sync();
        for (ek, ev) in small.shape.keys.iter().zip(small.vals.iter()) {
            keys.push_back_mut(ek.clone());
            inner.insert_mut(ek.clone(), ev.clone());
        }
        keys.push_back_mut(k.clone());
        inner.insert_mut(k, v);
        self.inner = Some(Box::new(inner));
        self.small = None;
        self.keys = Arc::new(keys);
    }

    /// Remove a key (`dissoc` one key). A missing key is a no-op;
    /// otherwise both the entries and the insertion-order sidecar
    /// drop it, and later positions shift down. The class never
    /// demotes (a shrunk hash-map stays a hash-map); neither does
    /// the representation (a shrunk hash-map stays hashed).
    #[must_use]
    pub fn without(mut self, k: &Value) -> Self {
        if let Some(small) = self.small.take() {
            let Some(idx) = small.shape.keys.iter().position(|ek| ek == k) else {
                self.small = Some(small);
                return self;
            };
            if small.vals.len() == 1 {
                self.small = Some(fresh_empty_small());
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
            self.small = Some(Arc::new(SmallMap {
                shape: intern_shape(&keys),
                vals: vals.into_boxed_slice(),
            }));
            return self;
        }
        if self.trie().get(k).is_none() {
            return self;
        }
        let mut keys = RpdsVector::new_sync();
        for key in self.keys.iter() {
            if key != k {
                keys.push_back_mut(key.clone());
            }
        }
        self.trie_mut().remove_mut(k);
        Self {
            inner: self.inner,
            small: None,
            meta: self.meta,
            keys: Arc::new(keys),
            hashy: self.hashy,
            sort: self.sort,
        }
    }

    /// Whether this reports `PersistentHashMap` (else array-map).
    #[must_use]
    pub fn is_hashy(&self) -> bool {
        self.hashy
    }

    /// Whether entries ride shaped (an interned key-set plus a
    /// flat values vec, no HAMT or sidecar allocs) rather than
    /// in the hash trie.
    #[must_use]
    pub fn is_shaped(&self) -> bool {
        self.small.is_some()
    }

    /// Force the reported class (`hash-map` forces hash,
    /// `array-map`/transient-array forces array).
    #[must_use]
    pub fn with_hashy(mut self, hashy: bool) -> Self {
        self.hashy = hashy;
        self
    }

    /// Whether this is a sorted map (FT6 `sorted-map` family).
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        self.sort.is_some()
    }

    /// The sort order, if sorted.
    #[must_use]
    pub fn sort_spec(&self) -> Option<&SortSpec> {
        self.sort.as_deref()
    }

    /// Mark this map sorted (or clear the order), preserving entries.
    /// Callers maintain `keys` in comparator order alongside.
    #[must_use]
    pub fn with_sort(mut self, sort: Option<SortSpec>) -> Self {
        self.sort = sort.map(Box::new);
        self
    }

    /// Keys in iteration order (insertion order, or comparator
    /// order when sorted). Owned: the sidecar is a persistent
    /// vector, so only sorted paths (already linear) snapshot it.
    #[must_use]
    pub fn ordered_keys(&self) -> Vec<Value> {
        match &self.small {
            Some(small) => small.shape.keys.to_vec(),
            None => self.keys.iter().cloned().collect(),
        }
    }

    /// Rebuild from explicitly ordered entries, preserving metadata
    /// and the sort order (sorted `assoc` positions through here).
    #[must_use]
    pub fn with_ordered_entries(self, entries: Vec<(Value, Value)>) -> Self {
        if entries.len() <= INLINE_MAP_MAX {
            if entries.is_empty() {
                return Self {
                    inner: None,
                    small: Some(fresh_empty_small()),
                    keys: EMPTY_MAP_KEYS.clone(),
                    meta: self.meta,
                    hashy: self.hashy,
                    sort: self.sort,
                };
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
            return Self {
                inner: None,
                small: Some(Arc::new(SmallMap {
                    shape: intern_shape(&keys),
                    vals: vals.into_boxed_slice(),
                })),
                keys: EMPTY_MAP_KEYS.clone(),
                meta: self.meta,
                hashy: self.hashy,
                sort: self.sort,
            };
        }
        let mut inner = RpdsMap::new_with_hasher_and_ptr_kind(FxBuildHasher::default());
        let mut keys = RpdsVector::new_sync();
        for (k, v) in entries {
            keys.push_back_mut(k.clone());
            inner.insert_mut(k, v);
        }
        Self {
            inner: Some(Box::new(inner)),
            small: None,
            meta: self.meta,
            keys: Arc::new(keys),
            hashy: self.hashy,
            sort: self.sort,
        }
    }

    /// Look up a key.
    #[must_use]
    pub fn get(&self, k: &Value) -> Option<&Value> {
        match &self.small {
            Some(small) => small
                .shape
                .keys
                .iter()
                .position(|ek| ek == k)
                .map(|idx| &small.vals[idx]),
            None => self.trie().get(k),
        }
    }

    /// Whether a key is present.
    #[must_use]
    pub fn contains_key(&self, k: &Value) -> bool {
        match &self.small {
            Some(small) => small.shape.keys.iter().any(|ek| ek == k),
            None => self.trie().get(k).is_some(),
        }
    }

    /// The entry count.
    #[must_use]
    pub fn size(&self) -> usize {
        match &self.small {
            Some(small) => small.vals.len(),
            None => self.trie().size(),
        }
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Iterate entries in insertion order (the JVM small-map seq
    /// order; see the `keys` sidecar).
    pub fn iter(&self) -> MapIter<'_> {
        MapIter {
            inner: match &self.small {
                Some(small) => MapIterInner::Shaped {
                    keys: small.shape.keys.iter(),
                    vals: small.vals.iter(),
                },
                None => MapIterInner::Hash(HashIter {
                    keys: &self.keys,
                    map: self.trie(),
                    pos: 0,
                }),
            },
        }
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_deref()
    }

    /// Return this map with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news), so fresh
        // entries (identity rides the entries `Arc`).
        if let Some(small) = self.small.take() {
            self.small = Some(Arc::new(SmallMap {
                shape: Arc::clone(&small.shape),
                vals: small.vals.clone(),
            }));
        } else {
            self.keys = Arc::new((*self.keys).clone());
        }
        self
    }

    /// The allocation identity for `identical?` (S16): the entries
    /// `Arc` address (shaped) or the sidecar `Arc` address (hash).
    /// Writers mint a fresh `Arc` except uniquely-owned in-place
    /// updates (#227 — same address, unobservable: no live alias
    /// shares it), `Clone` shares it, and no-ops return `self` —
    /// so live maps compare exactly like minted ids, with no field
    /// and no counter. Not stable across processes (addresses);
    /// stable per allocation (like the JVM's `identityHashCode`,
    /// which is the only other consumer).
    #[must_use]
    pub fn id(&self) -> u64 {
        match &self.small {
            Some(small) => Arc::as_ptr(small).addr() as u64,
            None => Arc::as_ptr(&self.keys).addr() as u64,
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
        let mut set = new_set();
        for v in items {
            set = set.insert(v);
        }
        Ok(set)
    }

    /// Add a value. A new member appends to the insertion order;
    /// a present member is a no-op (keeps the original object).
    #[must_use]
    pub fn insert(mut self, v: Value) -> Self {
        if self.inner.contains(&v) {
            self
        } else {
            let mut members = self.members;
            Arc::make_mut(&mut members).push_back_mut(v.clone());
            self.inner.insert_mut(v);
            Self {
                inner: self.inner,
                meta: self.meta,
                members,
                sort: self.sort,
                id: next_set_id(),
            }
        }
    }

    /// Whether a value is present.
    #[must_use]
    pub fn contains(&self, v: &Value) -> bool {
        self.inner.contains(v)
    }

    /// The stored member equal to `v`, if present (`get` on a set
    /// returns the member, not the key).
    #[must_use]
    pub fn get(&self, v: &Value) -> Option<&Value> {
        self.members.iter().find(|m| *m == v)
    }

    /// Remove a value (`disj` one item). A missing value is a no-op;
    /// otherwise both the entries and the insertion-order sidecar
    /// drop it, and later positions shift down.
    #[must_use]
    pub fn without(mut self, v: &Value) -> Self {
        if !self.inner.contains(v) {
            return self;
        }
        let mut members = RpdsVector::new_sync();
        for m in self.members.iter() {
            if m != v {
                members.push_back_mut(m.clone());
            }
        }
        self.inner.remove_mut(v);
        Self {
            inner: self.inner,
            meta: self.meta,
            members: Arc::new(members),
            sort: self.sort,
            id: next_set_id(),
        }
    }

    /// The element count.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.size() == 0
    }

    /// Iterate members in insertion order (the JVM small-set seq
    /// order; see the `members` sidecar).
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.members.iter()
    }

    /// Whether this is a sorted set (FT6 `sorted-set` family).
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        self.sort.is_some()
    }

    /// The sort order, if sorted.
    #[must_use]
    pub fn sort_spec(&self) -> Option<&SortSpec> {
        self.sort.as_deref()
    }

    /// Mark this set sorted (or clear the order), preserving members.
    /// Callers maintain `members` in comparator order alongside.
    #[must_use]
    pub fn with_sort(mut self, sort: Option<SortSpec>) -> Self {
        self.sort = sort.map(Box::new);
        self
    }

    /// Members in iteration order (insertion order, or comparator
    /// order when sorted). Owned: only sorted paths (already
    /// linear) snapshot the persistent sidecar.
    #[must_use]
    pub fn ordered_members(&self) -> Vec<Value> {
        self.members.iter().cloned().collect()
    }

    /// Rebuild from explicitly ordered members, preserving metadata
    /// and the sort order (sorted `conj` positions through here).
    #[must_use]
    pub fn with_ordered_members(self, members: Vec<Value>) -> Self {
        let mut inner = RpdsSet::new_with_hasher_with_ptr_kind(FxBuildHasher::default());
        let mut kept = RpdsVector::new_sync();
        for v in &members {
            kept.push_back_mut(v.clone());
            inner.insert_mut(v.clone());
        }
        let members = Arc::new(kept);
        Self {
            inner,
            meta: self.meta,
            members,
            sort: self.sort,
            id: next_set_id(),
        }
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_deref()
    }

    /// Return this set with `meta` attached (`None` clears it).
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta.map(Arc::new);
        // A new object on the JVM (`withMeta` news), so a new id.
        self.id = next_set_id();
        self
    }

    /// The allocation identity for `identical?` (S16): minted at
    /// every construction, preserved by `Clone`.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
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
        self.iter().eq(other.iter())
    }
}

impl Eq for Vector {}

impl Hash for Vector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        for v in self.iter() {
            v.hash(state);
        }
    }
}

impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.size() == other.size() && self.iter().all(|(k, v)| other.get(k) == Some(v))
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
        self.size() == other.size() && self.iter().all(|v| other.contains(v))
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
#[must_use]
pub fn new_map() -> Map {
    Map {
        inner: None,
        small: Some(fresh_empty_small()),
        keys: EMPTY_MAP_KEYS.clone(),
        meta: None,
        hashy: false,
        sort: None,
    }
}

/// An empty [`Set`].
#[must_use]
pub fn new_set() -> Set {
    Set {
        inner: RpdsSet::new_with_hasher_with_ptr_kind(FxBuildHasher::default()),
        meta: None,
        members: Arc::new(RpdsVector::new_sync()),
        sort: None,
        id: next_set_id(),
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
#[derive(Clone, Debug)]
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
    Throwable(Throwable),
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
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            // `=` on doubles is `==` (`0.0` ≡ `-0.0`, NaN never).
            // Mixed `Int`/`Double` is never `=` (`Numbers.equal`
            // requires one category; `(== 1 1.0)` is the value
            // comparison, arriving with numerics in S4).
            (Value::Double(a), Value::Double(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::Keyword(a), Value::Keyword(b)) => a == b,
            // Sequential equality crosses List/Vector.
            (Value::List(a), Value::List(b)) => a.iter().eq(b.iter()),
            (Value::Vector(a), Value::Vector(b)) => a.iter().eq(b.iter()),
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

fn map_value_eq(a: &Map, b: &Map) -> bool {
    a.size() == b.size() && a.iter().all(|(k, v)| b.get(k) == Some(v))
}

fn set_value_eq(a: &Set, b: &Set) -> bool {
    a.size() == b.size() && a.iter().all(|v| b.contains(v))
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
        _ => {
            if let Value::Ratio(r) = v {
                state.write_u8(25);
                hash_bigint_bytes(r.numer(), state);
                hash_bigint_bytes(r.denom(), state);
            }
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Family markers keep unequal types apart (`Int` and
        // `Double` hash separately — they are never `=`).
        match self {
            Value::Nil => state.write_u8(0),
            Value::Bool(b) => {
                state.write_u8(1);
                state.write_u8(u8::from(*b));
            }
            Value::Int(i) => {
                state.write_u8(2);
                state.write_i64(*i);
            }
            Value::Double(d) => {
                state.write_u8(3);
                state.write_u64(double_hash_bits(*d));
            }
            Value::BigInt(_) | Value::BigInteger(_) | Value::BigDec(_) | Value::Ratio(_) => {
                hash_tower_value(self, state);
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
}

/// Length-prefixed magnitude bytes (ambiguity-free across
/// adjacent fields).
fn hash_bigint_bytes<H: Hasher>(n: &num_bigint::BigInt, state: &mut H) {
    let bytes = n.to_signed_bytes_be();
    state.write_usize(bytes.len());
    state.write(&bytes);
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
        Value::PrimitiveVector(p) => {
            state.write_u8(8);
            state.write_usize(p.len());
            for v in p.iter() {
                v.hash(state);
            }
        }
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
        Value::Throwable(t) => {
            state.write_u8(16);
            state.write_u64(t.id);
        }
        Value::Transient(t) => {
            state.write_u8(17);
            state.write_u64(t.id());
        }
        other => hash_value_rest(other, state),
    }
}

/// The tag byte plus minted id for id-hashed cells (atoms,
/// delays, futures, promises, agents, volatiles, refs — each
/// hashes its tag plus its id), or `None` for every other
/// variant (split from [`hash_value_rest`] for the complexity
/// budget, like [`hash_value_scalars`]).
fn id_cell_tag(v: &Value) -> Option<(u8, u64)> {
    match v {
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
    use crate::{HostDef, HostInst, HostKind, HostVal, Keyword, Symbol};
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
        let shape_of = |m: &super::Map| Arc::clone(&m.small.as_ref().unwrap().shape);
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
    fn insertion_order_sidecars() {
        // Maps iterate in insertion order (JVM small-map seq).
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
            Value::Throwable(Throwable {
                id,
                class: ThrowClass::Exception,
                message: None,
                data: None,
                cause: None,
            })
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
}
