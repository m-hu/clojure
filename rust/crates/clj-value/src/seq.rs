// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Seqs (S3, C4): lazy, chunked (32), nil-punned sequences over
//! collections (`docs/spec/t1174/probes/probe18.clj`).
//!
//! [`Seq::first`]/[`Seq::next`]/[`Seq::more`] mirror the JVM
//! (`next` yields `None` at the end, `more` the empty seq).
//! [`Seq::chunk_next`] exposes the current chunk plus the seq after
//! it, so chunk-aware consumers (S5's `map`/`filter`, the test
//! helpers) realize whole chunks exactly like the JVM. [`Seq::lazy`]
//! memoizes through [`OnceLock`], so thunks run once even under
//! contention, like `synchronized` `LazySeq`.

use std::borrow::Cow;
use std::sync::{Arc, Mutex, OnceLock};

use crate::{List, Map, Set, Value, Vector};

/// The JVM chunk size: 32 items per chunk.
pub const CHUNK_SIZE: usize = 32;

/// `seq` a value into items (`None` is unseqable — callers raise
/// the ISeq coercion error). `nil` seqs to empty, strings to chars,
/// maps to `MapEntry` values, entries to their 2 items, like the JVM.
#[must_use]
pub fn seq_vec(v: &Value) -> Option<Vec<Value>> {
    match v {
        Value::Nil => Some(Vec::new()),
        Value::Str(s) => Some(s.chars().map(Value::Char).collect()),
        Value::List(l) => Some(l.iter().cloned().collect()),
        Value::Vector(v) => Some(v.iter().collect()),
        Value::MapEntry { key, val, .. } => Some(vec![(**key).clone(), (**val).clone()]),
        Value::Map(m) => Some(
            m.iter()
                .map(|(k, v)| crate::map_entry(k.clone(), v.clone()))
                .collect(),
        ),
        Value::Set(s) => Some(s.iter().cloned().collect()),
        _ => None,
    }
}

/// A sequence over values. Clones are cheap (all `Arc`-backed).
#[derive(Clone)]
pub enum Seq {
    /// Eager sequential items (lists, maps, sets, `more`-end).
    Eager { items: Arc<[Value]>, pos: usize },
    /// Chunked items (vectors, chunk-preserving ops).
    Chunked {
        chunks: Arc<[Arc<[Value]>]>,
        chunk: usize,
        pos: usize,
    },
    /// A realized chunk head plus its (possibly lazy) tail.
    ChunkCons {
        chunk: Arc<[Value]>,
        pos: usize,
        tail: Box<Seq>,
    },
    /// One head plus its tail (`None` at the end).
    Cons { head: Value, tail: Option<Box<Seq>> },
    /// A memoized thunk (`None` forces to the empty seq).
    Lazy {
        thunk: Arc<dyn Fn() -> Option<Seq> + Send + Sync>,
        memo: Arc<OnceLock<Option<Seq>>>,
    },
    /// A vector from index `pos` (`pos < v.len()`; `head` is the
    /// item there): `seq`/`next` share the vector — O(1), no copy,
    /// like the JVM's `PersistentVector$ChunkedSeq` — and chunks are
    /// the vector's own 32-item leaves.
    Vec { v: Vector, pos: usize, head: Value },
    /// A string from byte offset `pos`, char `idx` of the walk
    /// (`pos < s.len()`; `head` is the char there): O(1) steps over
    /// the shared string, like the JVM's `StringSeq`. Chunks stay the
    /// 32-char blocks of the walk from its first char.
    Str {
        s: Arc<str>,
        pos: usize,
        idx: usize,
        head: Value,
    },
}

impl std::fmt::Debug for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Seq::Eager { items, pos } => f
                .debug_struct("Eager")
                .field("len", &items.len())
                .field("pos", pos)
                .finish(),
            Seq::Chunked { chunks, chunk, pos } => f
                .debug_struct("Chunked")
                .field("chunks", &chunks.len())
                .field("chunk", chunk)
                .field("pos", pos)
                .finish(),
            Seq::ChunkCons { chunk, pos, .. } => f
                .debug_struct("ChunkCons")
                .field("len", &chunk.len())
                .field("pos", pos)
                .finish(),
            Seq::Cons { head, .. } => f.debug_tuple("Cons").field(head).finish(),
            Seq::Lazy { .. } => f.write_str("Lazy(..)"),
            Seq::Vec { v, pos, .. } => f
                .debug_struct("Vec")
                .field("len", &v.len())
                .field("pos", pos)
                .finish(),
            Seq::Str { s, pos, .. } => f
                .debug_struct("Str")
                .field("len", &s.len())
                .field("pos", pos)
                .finish(),
        }
    }
}

impl Seq {
    /// The empty seq (`more` at the end).
    #[must_use]
    pub fn empty() -> Self {
        Seq::Eager {
            items: Arc::from([]),
            pos: 0,
        }
    }

    /// An eager seq, or `None` when empty (nil-punning).
    #[must_use]
    pub fn eager(items: Vec<Value>) -> Option<Self> {
        if items.is_empty() {
            None
        } else {
            Some(Seq::Eager {
                items: Arc::from(items),
                pos: 0,
            })
        }
    }

    /// A chunked seq (32-item chunks), or `None` when empty.
    #[must_use]
    pub fn chunked(items: Vec<Value>) -> Option<Self> {
        if items.is_empty() {
            return None;
        }
        let chunks: Vec<Arc<[Value]>> = items
            .chunks(CHUNK_SIZE)
            .map(|c| Arc::from(c.to_vec()))
            .collect();
        Some(Seq::Chunked {
            chunks: Arc::from(chunks),
            chunk: 0,
            pos: 0,
        })
    }

    /// A realized chunk head plus its tail (empty chunks collapse to
    /// the tail, keeping chunks non-empty).
    #[must_use]
    pub fn chunk_cons(chunk: Vec<Value>, tail: Seq) -> Self {
        if chunk.is_empty() {
            return tail;
        }
        Seq::ChunkCons {
            chunk: Arc::from(chunk),
            pos: 0,
            tail: Box::new(tail),
        }
    }

    /// One head plus its tail.
    #[must_use]
    pub fn cons(head: Value, tail: Option<Seq>) -> Self {
        Seq::Cons {
            head,
            tail: tail.map(Box::new),
        }
    }

    /// A memoized thunk (runs once, even under contention).
    pub fn lazy<F>(f: F) -> Self
    where
        F: Fn() -> Option<Seq> + Send + Sync + 'static,
    {
        Seq::Lazy {
            thunk: Arc::new(f),
            memo: Arc::new(OnceLock::new()),
        }
    }

    /// A vector's seq from index `pos` (`None` at or past the end).
    #[must_use]
    pub fn vector_from(v: Vector, pos: usize) -> Option<Self> {
        let head = v.get(pos)?;
        Some(Seq::Vec { v, pos, head })
    }

    /// A string's seq from byte offset `pos` (a char boundary), char
    /// `idx` of the walk (`None` at or past the end).
    #[must_use]
    pub fn string_from(s: Arc<str>, pos: usize, idx: usize) -> Option<Self> {
        let ch = s.get(pos..)?.chars().next()?;
        Some(Seq::Str {
            s,
            pos,
            idx,
            head: Value::Char(ch),
        })
    }

    /// Seq over a list's items (sequential, like the JVM list seq).
    #[must_use]
    pub fn from_list(l: &List) -> Option<Self> {
        Self::eager(l.iter().cloned().collect())
    }

    /// Seq over a vector's items (chunked, like the JVM).
    #[must_use]
    pub fn from_vector(v: &Vector) -> Option<Self> {
        Self::chunked(v.iter().collect())
    }

    /// Seq over a map's entries as `MapEntry` values, in seq order.
    #[must_use]
    pub fn from_map(m: &Map) -> Option<Self> {
        Self::eager(
            m.iter()
                .map(|(k, v)| crate::map_entry(k.clone(), v.clone()))
                .collect(),
        )
    }

    /// Seq over a set's members, in seq order.
    #[must_use]
    pub fn from_set(s: &Set) -> Option<Self> {
        Self::eager(s.iter().cloned().collect())
    }

    /// Force a lazy seq (memoized); non-lazy seqs return themselves.
    fn forced(&self) -> Option<&Seq> {
        match self {
            Seq::Lazy { thunk, memo } => memo.get_or_init(|| thunk()).as_ref(),
            other => Some(other),
        }
    }

    /// Whether this seq is exhausted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self.forced() {
            None => true,
            Some(Seq::Eager { items, pos }) => *pos >= items.len(),
            Some(Seq::Chunked { .. }) => false,
            Some(Seq::ChunkCons { chunk, pos, tail }) => *pos >= chunk.len() && tail.is_empty(),
            Some(Seq::Cons { .. } | Seq::Vec { .. } | Seq::Str { .. }) => false,
            Some(l @ Seq::Lazy { .. }) => l.is_empty(),
        }
    }

    /// The first value, if any.
    #[must_use]
    pub fn first(&self) -> Option<&Value> {
        match self.forced()? {
            Seq::Eager { items, pos } => items.get(*pos),
            Seq::Chunked { chunks, chunk, pos } => chunks.get(*chunk).and_then(|c| c.get(*pos)),
            Seq::ChunkCons { chunk, pos, tail } => chunk.get(*pos).or_else(|| tail.first()),
            Seq::Cons { head, .. } | Seq::Vec { head, .. } | Seq::Str { head, .. } => Some(head),
            l @ Seq::Lazy { .. } => l.first(),
        }
    }

    /// The seq after the first value (`None` at the end).
    #[must_use]
    pub fn next(&self) -> Option<Seq> {
        match self.forced()? {
            Seq::Eager { items, pos } => (pos + 1 < items.len()).then(|| Seq::Eager {
                items: Arc::clone(items),
                pos: pos + 1,
            }),
            Seq::Chunked { chunks, chunk, pos } => Seq::next_chunked(chunks, *chunk, *pos),
            Seq::ChunkCons { chunk, pos, tail } => {
                if pos + 1 < chunk.len() {
                    Some(Seq::ChunkCons {
                        chunk: Arc::clone(chunk),
                        pos: pos + 1,
                        tail: tail.clone(),
                    })
                } else {
                    let t = (**tail).clone();
                    if t.is_empty() { None } else { Some(t) }
                }
            }
            Seq::Cons { tail, .. } => tail.clone().map(|t| *t),
            l @ Seq::Lazy { .. } => l.next(),
            Seq::Vec { v, pos, .. } => Seq::vector_from(v.clone(), pos + 1),
            Seq::Str { s, pos, idx, .. } => {
                let step = s[*pos..].chars().next().map_or(1, char::len_utf8);
                Seq::string_from(Arc::clone(s), pos + step, idx + 1)
            }
        }
    }

    /// `next` of a [`Seq::Chunked`] at `chunk`/`pos`.
    fn next_chunked(chunks: &Arc<[Arc<[Value]>]>, chunk: usize, pos: usize) -> Option<Seq> {
        let len = chunks.get(chunk).map_or(0, |c| c.len());
        if pos + 1 < len {
            Some(Seq::Chunked {
                chunks: Arc::clone(chunks),
                chunk,
                pos: pos + 1,
            })
        } else if chunk + 1 < chunks.len() {
            Some(Seq::Chunked {
                chunks: Arc::clone(chunks),
                chunk: chunk + 1,
                pos: 0,
            })
        } else {
            None
        }
    }

    /// Step past the head in place (`false`, unchanged, when nothing
    /// follows it): vector and string seqs bump their position over
    /// the storage they keep; other shapes become their `next`.
    fn step_in_place(&mut self) -> bool {
        match self {
            Seq::Vec { v, pos, head } => match v.get(*pos + 1) {
                Some(x) => {
                    *pos += 1;
                    *head = x;
                    true
                }
                None => false,
            },
            Seq::Str { s, pos, idx, head } => {
                let step = s[*pos..].chars().next().map_or(1, char::len_utf8);
                match s[*pos + step..].chars().next() {
                    Some(c) => {
                        *pos += step;
                        *idx += 1;
                        *head = Value::Char(c);
                        true
                    }
                    None => false,
                }
            }
            other => match other.next() {
                Some(rest) if !rest.is_empty() => {
                    *other = rest;
                    true
                }
                _ => false,
            },
        }
    }

    /// The seq after the first value (empty, never `None`, at the end).
    #[must_use]
    pub fn more(&self) -> Seq {
        self.next().unwrap_or_else(Self::empty)
    }

    /// The remaining item count, read off the backing arrays without
    /// cloning items (the JVM's counted seqs count in O(1), and
    /// `empty?` counts counted seqs). Cons cells and lazy tails still
    /// step, forcing like a [`first`](Self::first)/[`next`](Self::next)
    /// walk, which this matches item for item.
    #[must_use]
    pub fn count(&self) -> usize {
        let mut n = 0;
        let mut cur = self.forced();
        while let Some(s) = cur {
            cur = match s {
                Seq::Eager { items, pos } => {
                    n += items.len().saturating_sub(*pos);
                    None
                }
                Seq::Chunked { chunks, chunk, pos } => {
                    if let Some(c) = chunks.get(*chunk).filter(|c| *pos < c.len()) {
                        n += c.len() - pos;
                        // A walk stops at an empty chunk (`first` is `None`).
                        n += chunks[chunk + 1..]
                            .iter()
                            .take_while(|c| !c.is_empty())
                            .map(|c| c.len())
                            .sum::<usize>();
                    }
                    None
                }
                Seq::ChunkCons { chunk, pos, tail } => {
                    n += chunk.len().saturating_sub(*pos);
                    tail.forced()
                }
                Seq::Cons { tail, .. } => {
                    n += 1;
                    tail.as_deref().and_then(Seq::forced)
                }
                l @ Seq::Lazy { .. } => l.forced(),
                Seq::Vec { v, pos, .. } => {
                    n += v.len() - pos;
                    None
                }
                Seq::Str { s, pos, .. } => {
                    n += s[*pos..].chars().count();
                    None
                }
            };
        }
        n
    }

    /// The current chunk plus the seq after it (`None` unless this
    /// seq is chunked here). The rest is the empty seq past the last
    /// chunk.
    #[must_use]
    pub fn chunk_next(&self) -> Option<(Cow<'_, [Value]>, Seq)> {
        match self.forced()? {
            Seq::Chunked { chunks, chunk, pos } => {
                let rest = if chunk + 1 < chunks.len() {
                    Seq::Chunked {
                        chunks: Arc::clone(chunks),
                        chunk: chunk + 1,
                        pos: 0,
                    }
                } else {
                    Seq::empty()
                };
                Some((Cow::Borrowed(&chunks[*chunk][*pos..]), rest))
            }
            Seq::ChunkCons { chunk, pos, tail } => {
                if *pos < chunk.len() {
                    Some((Cow::Borrowed(&chunk[*pos..]), (**tail).clone()))
                } else {
                    tail.chunk_next()
                }
            }
            // A forced lazy is a nested lazy: force through (like
            // `first`/`next` — the JVM forces recursively).
            Seq::Lazy { .. } => self.forced()?.chunk_next(),
            Seq::Vec { v, pos, .. } => {
                let chunk = v.chunk_from(*pos);
                let rest =
                    Seq::vector_from(v.clone(), pos + chunk.len()).unwrap_or_else(Seq::empty);
                Some((chunk, rest))
            }
            Seq::Str { s, pos, idx, .. } => {
                // Up to the walk's next 32-char boundary.
                let want = (idx / CHUNK_SIZE + 1) * CHUNK_SIZE - idx;
                let mut chunk = Vec::with_capacity(want);
                let mut end = *pos;
                for ch in s[*pos..].chars().take(want) {
                    chunk.push(Value::Char(ch));
                    end += ch.len_utf8();
                }
                let n = chunk.len();
                let rest = Seq::string_from(Arc::clone(s), end, idx + n).unwrap_or_else(Seq::empty);
                Some((Cow::Owned(chunk), rest))
            }
            _ => None,
        }
    }
}

/// Which JVM seq class a [`RealizedSeq`] reports (S5D): the shape
/// alone cannot tell a string seq from a set seq, but cast errors
/// print the class, so values carry it explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqKind {
    /// `PersistentVector$ChunkedSeq` (vectors, `butlast`, vector drops).
    Chunked,
    /// `ArraySeq` (`sort`/`sort-by` results).
    Array,
    /// `StringSeq` (string seqs and their tails).
    Str,
    /// `PersistentArrayMap$Seq` (small-map seqs and their tails).
    Map,
    /// `APersistentMap$KeySeq` (set and big-map seqs and their tails).
    Key,
    /// `ChunkedCons` (`take-last` of a lazy seq).
    ChunkedCons,
    /// `Cons` (pure `Seq::Cons` tails).
    Cons,
    /// `LazySeq` (pure `Seq::Lazy` tails).
    Lazy,
    /// `APersistentVector$Seq` (subvector seqs and their tails:
    /// counted but never `IDrop`).
    Subvec,
    /// `APersistentVector$RSeq` (`rseq` results: counted, never
    /// chunked, never `IDrop` — FX2 probes `rseq-vec-class`,
    /// `chunk-first-rseq`, `drop-rseq-class`).
    RSeq,
}

impl SeqKind {
    /// The JVM binary name.
    #[must_use]
    pub fn jvm_name(self) -> &'static str {
        match self {
            SeqKind::Chunked => "clojure.lang.PersistentVector$ChunkedSeq",
            SeqKind::Array => "clojure.lang.ArraySeq",
            SeqKind::Str => "clojure.lang.StringSeq",
            SeqKind::Map => "clojure.lang.PersistentArrayMap$Seq",
            SeqKind::Key => "clojure.lang.APersistentMap$KeySeq",
            SeqKind::ChunkedCons => "clojure.lang.ChunkedCons",
            SeqKind::Cons => "clojure.lang.Cons",
            SeqKind::Lazy => "clojure.lang.LazySeq",
            SeqKind::Subvec => "clojure.lang.APersistentVector$Seq",
            SeqKind::RSeq => "clojure.lang.APersistentVector$RSeq",
        }
    }

    /// The default kind for a pure shape (what `Seq::eager` /
    /// `Seq::chunked` tails report when nothing more specific is
    /// known).
    #[must_use]
    pub fn of_shape(seq: &Seq) -> Self {
        match seq {
            Seq::Chunked { .. } | Seq::ChunkCons { .. } | Seq::Vec { .. } => SeqKind::Chunked,
            Seq::Str { .. } => SeqKind::Str,
            Seq::Cons { .. } => SeqKind::Cons,
            Seq::Lazy { .. } => SeqKind::Lazy,
            Seq::Eager { .. } => SeqKind::Array,
        }
    }
}

/// A realized (thunk-free) seq as a value (S5D): `seq` on vectors
/// yields [`Seq::Chunked`], on maps/sets/strings [`Seq::Eager`]
/// (lists are already seqs, so `seq` returns them directly).
/// Prints as a list; sequential equality/hash cross `List`/`Vector`
/// (see `Value`'s impls). Carries metadata like the JVM's seqs.
/// `counted`/`idrop` mirror the JVM interfaces (`StringSeq` and
/// small-map seqs are both; set seqs are neither; `sort` results
/// are counted only), and `kind` is the reported JVM class.
#[derive(Clone, Debug)]
pub struct RealizedSeq {
    seq: Seq,
    meta: Option<Map>,
    counted: bool,
    idrop: bool,
    kind: SeqKind,
}

impl RealizedSeq {
    /// Wrap a realized seq (metadata-free, uncounted, non-`IDrop`,
    /// shape-derived kind).
    #[must_use]
    pub fn new(seq: Seq) -> Self {
        let kind = SeqKind::of_shape(&seq);
        Self {
            seq,
            meta: None,
            counted: false,
            idrop: false,
            kind,
        }
    }

    /// Wrap a realized seq with explicit JVM interfaces and class.
    #[must_use]
    pub fn flagged(seq: Seq, counted: bool, idrop: bool, kind: SeqKind) -> Self {
        Self {
            seq,
            meta: None,
            counted,
            idrop,
            kind,
        }
    }

    /// Whether the JVM seq is `Counted` (`counted?`).
    #[must_use]
    pub fn is_counted(&self) -> bool {
        self.counted
    }

    /// Whether the JVM seq is `IDrop` (`drop` validates eagerly).
    #[must_use]
    pub fn is_idrop(&self) -> bool {
        self.idrop
    }

    /// The reported JVM class.
    #[must_use]
    pub fn kind(&self) -> SeqKind {
        self.kind
    }

    /// The remaining items (for printing and sequential equality).
    #[must_use]
    pub fn items(&self) -> Vec<Value> {
        let mut out = Vec::new();
        let mut cur = self.seq.clone();
        while let Some(head) = cur.first() {
            out.push(head.clone());
            match cur.next() {
                Some(rest) => cur = rest,
                None => break,
            }
        }
        out
    }

    /// The underlying seq (chunk-aware consumers step it directly).
    #[must_use]
    pub fn seq(&self) -> &Seq {
        &self.seq
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Attach a metadata map.
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta;
        self
    }

    /// Step past the first item in place, for `next`/`rest` of a seq
    /// held once (no new seq value): `false`, changing nothing, when
    /// nothing follows it. Metadata stays only on string seqs, like
    /// a fresh tail (`StringSeq.next` keeps it, every other `next`
    /// drops it).
    pub fn advance(&mut self) -> bool {
        if !self.seq.step_in_place() {
            return false;
        }
        if self.kind != SeqKind::Str {
            self.meta = None;
        }
        true
    }
}

impl crate::ExtensionValue for RealizedSeq {
    fn class_name(&self) -> String {
        self.kind.jvm_name().to_owned()
    }

    fn print(&self) -> String {
        // The printer downcasts and prints with options instead;
        // this fallback prints readably without limits.
        let parts: Vec<String> = self
            .items()
            .iter()
            .map(|v| crate::pr_to_string(v, &crate::PrintOpts::pr()))
            .collect();
        format!("({})", parts.join(" "))
    }

    fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| self.items() == o.items())
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Fallback (`Value::hash` downcasts and writes the
        // sequential family directly instead): digest through a
        // fresh state, since `dyn Hasher` cannot nest.
        let items = self.items();
        let mut h = DefaultHasher::new();
        8u8.hash(&mut h);
        items.len().hash(&mut h);
        for item in &items {
            item.hash(&mut h);
        }
        state.write_u64(h.finish());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// A persistent subvector view (S5D): `APersistentVector$SubVector`
/// reporting that class — a window over a shared backing vector
/// (nested windows flatten at construction, like the JVM ctor).
/// Prints/counts/compares/calls as its window; `conj`/`assoc` stay
/// subvectors, popping to one element and `empty` fall back to plain
/// vectors. Seqs over the window report `APersistentVector$Seq`
/// ([`SeqKind::Subvec`]).
#[derive(Clone, Debug)]
pub struct SubVectorValue {
    vec: Vector,
    start: usize,
    end: usize,
    meta: Option<Map>,
}

impl SubVectorValue {
    /// A window over `vec` (`start..end`; the caller upholds
    /// `start <= end <= len`, like `RT.subvec` does).
    #[must_use]
    pub fn new(vec: Vector, start: usize, end: usize) -> Self {
        Self {
            vec,
            start,
            end,
            meta: None,
        }
    }

    /// The backing vector.
    #[must_use]
    pub fn vec(&self) -> &Vector {
        &self.vec
    }

    /// The window start (an index into the backing vector).
    #[must_use]
    pub fn start(&self) -> usize {
        self.start
    }

    /// The window end (exclusive).
    #[must_use]
    pub fn end(&self) -> usize {
        self.end
    }

    /// The window length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The window items, in order.
    #[must_use]
    pub fn items(&self) -> Vec<Value> {
        self.iter().collect()
    }

    /// Iterate the window items, in order (owned values since
    /// vector packing).
    pub fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.vec.iter().skip(self.start).take(self.len())
    }

    /// The window item at `index` (`None` past the end).
    #[must_use]
    pub fn nth(&self, index: usize) -> Option<Value> {
        if index < self.len() {
            self.vec.get(self.start + index)
        } else {
            None
        }
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Attach a metadata map.
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta;
        self
    }

    /// Replace the window items (for deep realization): the new
    /// backing keeps the old head and tail, so `start`/`end` still
    /// delimit the window.
    #[must_use]
    pub fn with_window(&self, items: Vec<Value>) -> Self {
        let mut backing: Vec<Value> = self.vec.iter().take(self.start).collect();
        let end = self.start + items.len();
        backing.extend(items);
        backing.extend(self.vec.iter().skip(self.end));
        Self {
            vec: Vector::from_vec(backing),
            start: self.start,
            end,
            meta: self.meta.clone(),
        }
    }
}

impl crate::ExtensionValue for SubVectorValue {
    fn class_name(&self) -> String {
        "clojure.lang.APersistentVector$SubVector".to_owned()
    }

    fn print(&self) -> String {
        // The printer downcasts and prints with options instead;
        // this fallback prints readably without limits.
        let parts: Vec<String> = self
            .iter()
            .map(|v| crate::pr_to_string(&v, &crate::PrintOpts::pr()))
            .collect();
        format!("[{}]", parts.join(" "))
    }

    fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| self.items() == o.items())
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Fallback (`Value::hash` downcasts and writes the
        // sequential family directly instead): digest through a
        // fresh state, since `dyn Hasher` cannot nest.
        let items = self.items();
        let mut h = DefaultHasher::new();
        8u8.hash(&mut h);
        items.len().hash(&mut h);
        for item in &items {
            item.hash(&mut h);
        }
        state.write_u64(h.finish());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A range bound: all-`Int` ranges walk exactly, any `Double`
/// promotes the walk to `f64`, tower bounds walk exactly (the
/// start still yields verbatim).
#[derive(Clone, Debug, PartialEq)]
pub enum RangeNum {
    /// An exact integer bound.
    Int(i64),
    /// An inexact floating bound.
    Double(f64),
    /// An exact bigint bound.
    BigInt(num_bigint::BigInt),
    /// An exact ratio bound.
    Ratio(num_rational::BigRational),
    /// An exact decimal bound.
    BigDec(crate::BigDec),
}

impl RangeNum {
    /// The bound as a value (verbatim, never promoted).
    #[must_use]
    pub fn value(&self) -> Value {
        match self {
            RangeNum::Int(i) => Value::Int(*i),
            RangeNum::Double(d) => Value::Double(*d),
            RangeNum::BigInt(n) => Value::BigInt(Arc::new(n.clone())),
            RangeNum::Ratio(r) => Value::Ratio(Arc::new(r.clone())),
            RangeNum::BigDec(d) => Value::BigDec(Arc::new(d.clone())),
        }
    }

    /// The bound as `f64` (double walks; tower narrows).
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        use num_traits::ToPrimitive;
        match self {
            RangeNum::Int(i) => *i as f64,
            RangeNum::Double(d) => *d,
            RangeNum::BigInt(n) => n.to_f64().unwrap_or(f64::NAN),
            RangeNum::Ratio(r) => r.to_f64().unwrap_or(f64::NAN),
            RangeNum::BigDec(d) => d.to_f64(),
        }
    }

    /// Whether this bound is inexact (drives the `f64` walk).
    fn is_double(&self) -> bool {
        matches!(self, RangeNum::Double(_))
    }
}

/// An exact rational for a non-`Double` bound (`Int` and tower;
/// decimals are exact rationals, ratios as-is).
fn range_rational(n: &RangeNum) -> num_rational::BigRational {
    match n {
        RangeNum::Int(i) => num_rational::BigRational::from_integer(num_bigint::BigInt::from(*i)),
        RangeNum::BigInt(n) => num_rational::BigRational::from_integer(n.clone()),
        RangeNum::Ratio(r) => r.clone(),
        RangeNum::BigDec(d) => d.to_rational(),
        RangeNum::Double(_) => unreachable!("doubles take the f64 walk, never the exact one"),
    }
}

/// The sign of a non-`Double` step as `-1`/`0`/`1` (exact, via
/// the rational numerator).
fn range_step_sign(step: &RangeNum) -> i32 {
    match range_rational(step).numer().sign() {
        num_bigint::Sign::Minus => -1,
        num_bigint::Sign::NoSign => 0,
        num_bigint::Sign::Plus => 1,
    }
}

/// One exact step (`cur + step`, no `Double` involved): `BigDec`
/// wins, else ratios add (integral sums fold to `BigInt`, like
/// `Numbers.add`), else big integers add. `Int`+`Int` stays
/// checked (`None` overflows, like `forceChunk` landing out of
/// bounds). A `BigDec`/ratio pair with no exact decimal is `None`
/// — callers check [`RangeValue::step_inexact`] first and raise
/// the JVM `ArithmeticException` there.
/// Never inlined: the steppers are cold paths, and out-of-line
/// bodies keep their coverage regions on their own lines.
#[inline(never)]
fn range_add_exact(cur: &RangeNum, step: &RangeNum) -> Option<RangeNum> {
    if is_bigdec_num(cur) || is_bigdec_num(step) {
        return range_add_bigdec(cur, step);
    }
    if is_ratio_num(cur) || is_ratio_num(step) {
        return Some(range_add_ratio(cur, step));
    }
    range_add_int(cur, step)
}

/// Whether a bound is a decimal step.
fn is_bigdec_num(n: &RangeNum) -> bool {
    matches!(n, RangeNum::BigDec(_))
}

/// Whether a bound is a ratio step.
fn is_ratio_num(n: &RangeNum) -> bool {
    matches!(n, RangeNum::Ratio(_))
}

/// A `BigDec`-winning step (`None` for a ratio with no exact
/// decimal — see [`RangeValue::step_inexact`]).
#[inline(never)]
fn range_add_bigdec(cur: &RangeNum, step: &RangeNum) -> Option<RangeNum> {
    let (x, y) = (range_bigdec(cur)?, range_bigdec(step)?);
    Some(RangeNum::BigDec(x.add(&y)))
}

/// A ratio-winning step (integral sums fold to `BigInt`).
#[inline(never)]
fn range_add_ratio(cur: &RangeNum, step: &RangeNum) -> RangeNum {
    let sum = range_rational(cur) + range_rational(step);
    if sum.denom() == &num_bigint::BigInt::from(1) {
        RangeNum::BigInt(sum.numer().clone())
    } else {
        RangeNum::Ratio(sum)
    }
}

/// An integer step (`None` on `Int`+`Int` overflow).
#[inline(never)]
fn range_add_int(cur: &RangeNum, step: &RangeNum) -> Option<RangeNum> {
    match (cur, step) {
        (RangeNum::Int(c), RangeNum::Int(s)) => c.checked_add(*s).map(RangeNum::Int),
        (RangeNum::Int(c), RangeNum::BigInt(s)) => {
            Some(RangeNum::BigInt(num_bigint::BigInt::from(*c) + s))
        }
        (RangeNum::BigInt(c), RangeNum::Int(s)) => {
            Some(RangeNum::BigInt(c + num_bigint::BigInt::from(*s)))
        }
        (RangeNum::BigInt(c), RangeNum::BigInt(s)) => Some(RangeNum::BigInt(c + s)),
        _ => unreachable!("doubles and mixed exact kinds are handled above"),
    }
}

/// An exact `BigDec` for a `BigDec`-winning step (`None` for a
/// ratio with no exact decimal — see [`range_add_exact`]).
fn range_bigdec(n: &RangeNum) -> Option<crate::BigDec> {
    match n {
        RangeNum::Int(i) => Some(crate::BigDec::from_bigint(num_bigint::BigInt::from(*i))),
        RangeNum::BigInt(n) => Some(crate::BigDec::from_bigint(n.clone())),
        RangeNum::BigDec(d) => Some(d.clone()),
        RangeNum::Ratio(r) => crate::BigDec::from_rational_exact(r).ok(),
        RangeNum::Double(_) => unreachable!("doubles take the f64 walk, never the exact one"),
    }
}

/// `clojure.lang.LongRange` / `clojure.lang.Range` (S5D): a strict
/// numeric seq. All-`Int` ranges are `LongRange` (`Counted` +
/// `IDrop`, exact `count`); any-`Double` ranges are `Range`
/// (neither). Empty ranges never exist as values (`range`
/// returns `()` instead). Stepping lives in `clj-eval` (chunked,
/// 32 per step); printing/equality realize first, like `LazySeq`.
#[derive(Clone, Debug)]
pub struct RangeValue {
    cur: RangeNum,
    end: Option<RangeNum>,
    step: RangeNum,
    long: bool,
    count: Option<i64>,
    meta: Option<Map>,
}

impl RangeValue {
    /// A range value (`long` selects `LongRange` vs `Range`;
    /// `count` is the exact remaining count for `LongRange`).
    #[must_use]
    pub fn new(
        cur: RangeNum,
        end: Option<RangeNum>,
        step: RangeNum,
        long: bool,
        count: Option<i64>,
    ) -> Self {
        Self {
            cur,
            end,
            step,
            long,
            count,
            meta: None,
        }
    }

    /// The current bound (yields verbatim).
    #[must_use]
    pub fn cur(&self) -> RangeNum {
        self.cur.clone()
    }

    /// The exclusive end bound, if any.
    #[must_use]
    pub fn end(&self) -> Option<RangeNum> {
        self.end.clone()
    }

    /// The step bound.
    #[must_use]
    pub fn step(&self) -> RangeNum {
        self.step.clone()
    }

    /// Whether this is a `LongRange` (`Counted` + `IDrop`).
    #[must_use]
    pub fn is_long(&self) -> bool {
        self.long
    }

    /// The exact remaining count (`LongRange` only).
    #[must_use]
    pub fn count(&self) -> Option<i64> {
        self.count
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Attach a metadata map.
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta;
        self
    }

    /// Whether the walk is done: `LongRange` is count-bounded
    /// (exact, overflow-proof, like `LongRange.next`); double
    /// walks bound-compare in `f64` (`NaN` ends never finish, like
    /// the JVM's bounds check); exact walks compare exactly (tower
    /// bounds never narrow).
    #[must_use]
    pub fn done(&self) -> bool {
        if self.long {
            return self.count.is_none_or(|c| c <= 0);
        }
        match (&self.cur, &self.end, &self.step) {
            (_, None, _) => false,
            (cur, Some(end), step) if cur.is_double() || end.is_double() || step.is_double() => {
                let (c, x, s) = (cur.to_f64(), end.to_f64(), step.to_f64());
                if s >= 0.0 { c >= x } else { c <= x }
            }
            (cur, Some(end), step) => {
                let (c, x) = (range_rational(cur), range_rational(end));
                match range_step_sign(step) {
                    s if s > 0 => c >= x,
                    s if s < 0 => c <= x,
                    _ => false,
                }
            }
        }
    }

    /// Whether the next [`RangeValue::advance`] would add `BigDec`
    /// to a ratio with no exact decimal (the JVM throws
    /// `ArithmeticException` there — oracle `range-mixed`).
    /// Steppers with an error channel check this first and raise;
    /// channel-less walks treat such ranges as unsupported (see
    /// `seq_hash_unsupported`).
    #[must_use]
    pub fn step_inexact(&self) -> bool {
        match (&self.cur, &self.step) {
            (RangeNum::BigDec(_), RangeNum::Ratio(r))
            | (RangeNum::Ratio(r), RangeNum::BigDec(_)) => {
                crate::BigDec::from_rational_exact(r).is_err()
            }
            _ => false,
        }
    }

    /// Advance one step: `LongRange` adds unchecked (wrapping, like
    /// `LongRange.next` — the count bounds the walk); double walks
    /// check `Int`+`Int` (`None` ends the walk, like `forceChunk`
    /// landing out of bounds) and promote anything else to `f64`;
    /// exact walks add exactly (see [`range_add_exact` — a
    /// `BigDec`/ratio pair with no exact decimal is `None`, and
    /// callers raise via [`RangeValue::step_inexact`] instead]).
    pub fn advance(&self) -> Option<RangeNum> {
        match (&self.cur, &self.step) {
            (RangeNum::Int(c), RangeNum::Int(s)) if self.long => {
                Some(RangeNum::Int(c.wrapping_add(*s)))
            }
            (c, s) if c.is_double() || s.is_double() => {
                Some(RangeNum::Double(c.to_f64() + s.to_f64()))
            }
            // Non-`long` `Int`+`Int` stays checked (`None`
            // overflows) via [`range_add_int` below.
            (c, s) => range_add_exact(c, s),
        }
    }

    /// The head plus the range after one step (`None` when done;
    /// the tail is `None` on `Int`+`Int` overflow).
    #[must_use]
    pub fn stepped(&self) -> Option<(Value, Option<RangeValue>)> {
        if self.done() {
            return None;
        }
        let head = self.cur.value();
        // Tails drop metadata (like `Range.next`/`LongRange.next`,
        // whose fresh objects carry none).
        let next = self.advance().map(|nxt| RangeValue {
            cur: nxt,
            end: self.end.clone(),
            step: self.step.clone(),
            long: self.long,
            count: self.count.map(|c| c - 1),
            meta: None,
        });
        Some((head, next))
    }

    /// [`stepped`](Self::stepped) in place: the head, with this range
    /// advanced past it and its metadata dropped (`None` when done).
    /// The flag is `false` on `Int`+`Int` overflow, which leaves the
    /// range unadvanced (the walk ends there). Chunk fills step 32
    /// heads per tail, so they skip `stepped`'s fresh tail per head.
    pub fn step_in_place(&mut self) -> Option<(Value, bool)> {
        if self.done() {
            return None;
        }
        let head = self.cur.value();
        match self.advance() {
            Some(nxt) => {
                self.cur = nxt;
                self.count = self.count.map(|c| c - 1);
                self.meta = None;
                Some((head, true))
            }
            None => Some((head, false)),
        }
    }

    /// Call `f` per item (pure walk; infinite ranges never return,
    /// like the JVM's `hashCode` walk; a `long overflow` ends the
    /// walk early — the `hash` fn realizes first and reports it).
    pub fn for_each(&self, f: &mut dyn FnMut(&Value)) {
        let mut cur = self.clone();
        while let Some((head, next)) = cur.stepped() {
            f(&head);
            match next {
                Some(nxt) => cur = nxt,
                None => break,
            }
        }
    }

    /// The item count (`LongRange` reads its exact count; double
    /// ranges walk — infinite ones never return).
    #[must_use]
    pub fn walked_len(&self) -> usize {
        if let Some(c) = self.count {
            return usize::try_from(c.max(0)).unwrap_or(usize::MAX);
        }
        let mut n = 0usize;
        self.for_each(&mut |_| n += 1);
        n
    }
}

impl crate::ExtensionValue for RangeValue {
    fn class_name(&self) -> String {
        if self.long {
            "clojure.lang.LongRange".to_owned()
        } else {
            "clojure.lang.Range".to_owned()
        }
    }

    fn print(&self) -> String {
        // Out-of-band only (natives realize first): the JVM would
        // stream items, which needs the evaluator for chunking.
        "(...)".to_owned()
    }

    fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
        // Unrealized identity only (`=` realizes first).
        other.as_any().downcast_ref::<Self>().is_some_and(|o| {
            self.cur == o.cur && self.end == o.end && self.step == o.step && self.long == o.long
        })
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Fallback (`Value::hash` downcasts and writes the
        // sequential family directly instead): digest through a
        // fresh state, since `dyn Hasher` cannot nest.
        let mut h = DefaultHasher::new();
        8u8.hash(&mut h);
        self.walked_len().hash(&mut h);
        self.for_each(&mut |v| v.hash(&mut h));
        state.write_u64(h.finish());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `clojure.lang.Repeat` (S5D): a strict constant seq (`IDrop`,
/// never `Counted`, unchunked — one item per step). `n` is the
/// remaining count (`None` is infinite). Empty repeats never exist
/// as values (`repeat` returns `()` instead).
#[derive(Clone, Debug)]
pub struct RepeatValue {
    x: Value,
    n: Option<i64>,
    meta: Option<Map>,
}

impl RepeatValue {
    /// A repeat value (`None` is infinite).
    #[must_use]
    pub fn new(x: Value, n: Option<i64>) -> Self {
        Self { x, n, meta: None }
    }

    /// The repeated value.
    #[must_use]
    pub fn x(&self) -> &Value {
        &self.x
    }

    /// The remaining count (`None` is infinite).
    #[must_use]
    pub fn n(&self) -> Option<i64> {
        self.n
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Attach a metadata map.
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta;
        self
    }

    /// Call `f` per item (infinite repeats never return).
    pub fn for_each(&self, f: &mut dyn FnMut(&Value)) {
        match self.n {
            Some(n) => {
                for _ in 0..n.max(0) {
                    f(&self.x);
                }
            }
            None => loop {
                f(&self.x);
            },
        }
    }

    /// The item count (infinite repeats never return).
    #[must_use]
    pub fn walked_len(&self) -> usize {
        match self.n {
            Some(n) => usize::try_from(n.max(0)).unwrap_or(usize::MAX),
            None => {
                let mut n = 0usize;
                self.for_each(&mut |_| n += 1);
                n
            }
        }
    }
}

impl crate::ExtensionValue for RepeatValue {
    fn class_name(&self) -> String {
        "clojure.lang.Repeat".to_owned()
    }

    fn print(&self) -> String {
        "(...)".to_owned()
    }

    fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| self.x == o.x && self.n == o.n)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Fallback (`Value::hash` downcasts and writes the
        // sequential family directly instead).
        let mut h = DefaultHasher::new();
        8u8.hash(&mut h);
        self.walked_len().hash(&mut h);
        self.for_each(&mut |v| v.hash(&mut h));
        state.write_u64(h.finish());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `clojure.lang.Cycle` (S5D): a strict cycling seq. `prev` is the
/// seqed input (replayed forever), `current` the remaining tail.
/// Empty inputs never reach here (`cycle` returns `()` instead, like
/// `Cycle.create`).
#[derive(Clone, Debug)]
pub struct CycleValue {
    prev: Value,
    current: Value,
    meta: Option<Map>,
}

impl CycleValue {
    /// A cycle value over the seqed input.
    #[must_use]
    pub fn new(prev: Value, current: Value) -> Self {
        Self {
            prev,
            current,
            meta: None,
        }
    }

    /// The seqed input (replayed forever).
    #[must_use]
    pub fn prev(&self) -> &Value {
        &self.prev
    }

    /// The remaining tail.
    #[must_use]
    pub fn current(&self) -> &Value {
        &self.current
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Attach a metadata map.
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta;
        self
    }
}

impl crate::ExtensionValue for CycleValue {
    fn class_name(&self) -> String {
        "clojure.lang.Cycle".to_owned()
    }

    fn print(&self) -> String {
        "(...)".to_owned()
    }

    fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| std::ptr::eq(self, o))
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_usize(self as *const Self as usize);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `clojure.lang.Iterate` (S5D): a strict iteration seq with a
/// lazily realized seed (like the JVM's `_seed`/`prevSeed`):
/// `new` holds a realized head (free, no call), `deferred` holds
/// the previous seed (one `f` call on first read, memoized).
/// Stepping never calls `f` (like `next`, which defers); only the
/// head read realizes (like `first`). The call itself lives in
/// `clj-eval` (it needs the evaluator); this type only owns the
/// memo cell.
#[derive(Debug)]
pub struct IterateValue {
    f: Value,
    prev: Value,
    seed: Mutex<Option<Value>>,
    meta: Option<Map>,
}

impl Clone for IterateValue {
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            prev: self.prev.clone(),
            seed: Mutex::new(self.seed.lock().ok().and_then(|s| s.clone())),
            meta: self.meta.clone(),
        }
    }
}

impl IterateValue {
    /// An iterate value with a realized head.
    #[must_use]
    pub fn new(f: Value, x: Value) -> Self {
        Self {
            f,
            prev: Value::Nil,
            seed: Mutex::new(Some(x)),
            meta: None,
        }
    }

    /// An iterate value whose head is `(f prev)` (realized on first
    /// read, memoized).
    #[must_use]
    pub fn deferred(f: Value, prev: Value) -> Self {
        Self {
            f,
            prev,
            seed: Mutex::new(None),
            meta: None,
        }
    }

    /// The step function.
    #[must_use]
    pub fn f(&self) -> &Value {
        &self.f
    }

    /// The previous seed (the `f` argument while unrealized).
    #[must_use]
    pub fn prev(&self) -> &Value {
        &self.prev
    }

    /// The memoized head, if realized.
    #[must_use]
    pub fn seed_opt(&self) -> Option<Value> {
        self.seed.lock().ok().and_then(|s| s.clone())
    }

    /// Memoize a realized head.
    pub fn store_seed(&self, v: Value) {
        if let Ok(mut seed) = self.seed.lock() {
            *seed = Some(v);
        }
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Attach a metadata map.
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta;
        self
    }
}

impl crate::ExtensionValue for IterateValue {
    fn class_name(&self) -> String {
        "clojure.lang.Iterate".to_owned()
    }

    fn print(&self) -> String {
        "(...)".to_owned()
    }

    fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| std::ptr::eq(self, o))
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_usize(self as *const Self as usize);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `clojure.lang.Cons` (S5D): a strict head-plus-tail seq. `tail`
/// is the seq-or-nil tail (`cons` seqs non-seq tails eagerly, like
/// `RT.cons`; `rest` returns it untouched, preserving its kind).
#[derive(Clone, Debug)]
pub struct ConsValue {
    head: Value,
    tail: Value,
    meta: Option<Map>,
}

impl ConsValue {
    /// A cons value.
    #[must_use]
    pub fn new(head: Value, tail: Value) -> Self {
        Self {
            head,
            tail,
            meta: None,
        }
    }

    /// The head item.
    #[must_use]
    pub fn head(&self) -> &Value {
        &self.head
    }

    /// The raw tail value.
    #[must_use]
    pub fn tail(&self) -> &Value {
        &self.tail
    }

    /// The metadata map, if any.
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Attach a metadata map.
    #[must_use]
    pub fn with_meta(mut self, meta: Option<Map>) -> Self {
        self.meta = meta;
        self
    }
}

impl crate::ExtensionValue for ConsValue {
    fn class_name(&self) -> String {
        "clojure.lang.Cons".to_owned()
    }

    fn print(&self) -> String {
        "(...)".to_owned()
    }

    fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| std::ptr::eq(self, o))
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_usize(self as *const Self as usize);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

macro_rules! entry_seq {
    ($t:ident, $class:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug)]
        pub struct $t {
            underlying: Value,
            meta: Option<Map>,
        }

        impl $t {
            #[doc = "An entry view over the seqed input (entries as 2-vectors)."]
            #[must_use]
            pub fn new(underlying: Value) -> Self {
                Self {
                    underlying,
                    meta: None,
                }
            }

            #[doc = "The seqed input (entries as 2-vectors)."]
            #[must_use]
            pub fn underlying(&self) -> &Value {
                &self.underlying
            }

            #[doc = "The metadata map, if any."]
            #[must_use]
            pub fn meta(&self) -> Option<&Map> {
                self.meta.as_ref()
            }

            #[doc = "Attach a metadata map."]
            #[must_use]
            pub fn with_meta(mut self, meta: Option<Map>) -> Self {
                self.meta = meta;
                self
            }
        }

        impl crate::ExtensionValue for $t {
            fn class_name(&self) -> String {
                $class.to_owned()
            }

            fn print(&self) -> String {
                "(...)".to_owned()
            }

            fn dyn_eq(&self, other: &dyn crate::ExtensionValue) -> bool {
                other
                    .as_any()
                    .downcast_ref::<Self>()
                    .is_some_and(|o| std::ptr::eq(self, o))
            }

            fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
                state.write_usize(self as *const Self as usize);
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

entry_seq!(
    KeySeqValue,
    "clojure.lang.APersistentMap$KeySeq",
    "`clojure.lang.APersistentMap$KeySeq` (S5D): a strict view mapping entries to keys."
);
entry_seq!(
    ValSeqValue,
    "clojure.lang.APersistentMap$ValSeq",
    "`clojure.lang.APersistentMap$ValSeq` (S5D): a strict view mapping entries to values."
);

#[cfg(test)]
mod tests {
    use super::{
        ConsValue, CycleValue, IterateValue, KeySeqValue, RangeNum, RangeValue, RealizedSeq,
        RepeatValue, Seq, SeqKind, SubVectorValue, ValSeqValue, range_add_int, range_bigdec,
        range_rational,
    };
    use crate::{ExtensionValue, Value, Vector};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    fn int(i: i64) -> Value {
        Value::Int(i)
    }

    fn hash_ext(e: &dyn ExtensionValue) -> u64 {
        let mut h = DefaultHasher::new();
        e.dyn_hash(&mut h);
        h.finish()
    }

    fn hash_val(v: &Value) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    fn ints(r: std::ops::Range<i64>) -> Vec<Value> {
        r.map(int).collect()
    }

    /// `count` must equal the `first`/`next` walk length (what it
    /// replaced) on every shape, crafted cursor states included.
    #[test]
    fn count_matches_walk_on_every_shape() {
        let walked = |s: &Seq| RealizedSeq::new(s.clone()).items().len();
        let chunked = Seq::chunked(ints(0..70)).expect("chunked");
        let mut cases = vec![
            Seq::empty(),
            Seq::eager(ints(0..5)).expect("eager"),
            chunked.clone(),
            Seq::chunk_cons(ints(0..3), Seq::eager(ints(3..5)).expect("tail")),
            Seq::cons(int(0), Some(Seq::cons(int(1), None))),
            Seq::cons(int(0), Some(Seq::empty())),
            Seq::lazy(|| None),
            Seq::lazy(|| Some(Seq::lazy(|| Seq::eager(ints(0..4))))),
            // A spent cursor and an empty middle chunk end the walk.
            Seq::Chunked {
                chunks: Arc::from(vec![Arc::from(ints(0..2))]),
                chunk: 0,
                pos: 5,
            },
            Seq::Chunked {
                chunks: Arc::from(vec![
                    Arc::from(ints(0..2)),
                    Arc::from(Vec::new()),
                    Arc::from(ints(2..3)),
                ]),
                chunk: 0,
                pos: 0,
            },
        ];
        // Mid-walk cursors: inside and across chunk boundaries.
        let mut cur = chunked;
        for _ in 0..69 {
            cur = cur.next().expect("70 items");
            cases.push(cur.clone());
        }
        for s in &cases {
            assert_eq!(s.count(), walked(s), "{s:?}");
        }
        assert_eq!(cases[2].count(), 70);
        assert_eq!(cases[9].count(), 2);
    }

    #[test]
    fn seq_debug_names_shapes() {
        assert_eq!(format!("{:?}", Seq::empty()), "Eager { len: 0, pos: 0 }");
        let chunked = Seq::chunked(vec![int(1), int(2), int(3)]).expect("chunked");
        assert_eq!(
            format!("{:?}", chunked),
            "Chunked { chunks: 1, chunk: 0, pos: 0 }"
        );
        let headed = Seq::chunk_cons(vec![int(1), int(2)], Seq::empty());
        assert_eq!(format!("{:?}", headed), "ChunkCons { len: 2, pos: 0 }");
        assert_eq!(format!("{:?}", Seq::cons(int(1), None)), "Cons(Int(1))");
        let lazy = Seq::lazy(|| None::<Seq>);
        assert_eq!(format!("{:?}", lazy), "Lazy(..)");
    }

    #[test]
    fn seq_chunk_cons_empty_collapses() {
        let tail = Seq::eager(vec![int(1)]).expect("tail");
        let s = Seq::chunk_cons(Vec::new(), tail);
        assert_eq!(s.first(), Some(&int(1)));
        assert!(s.next().is_none());
    }

    #[test]
    fn seq_lazy_none_is_empty() {
        let s = Seq::lazy(|| None::<Seq>);
        assert!(s.is_empty());
        assert!(s.first().is_none());
        assert!(s.next().is_none());
        assert!(s.more().is_empty());
    }

    #[test]
    fn seq_nested_lazy_forces() {
        let inner = Seq::lazy(|| Seq::eager(vec![int(1), int(2)]));
        let outer = Seq::lazy(move || Some(inner.clone()));
        assert!(!outer.is_empty());
        assert_eq!(outer.first(), Some(&int(1)));
        let tail = outer.next().expect("tail");
        assert_eq!(tail.first(), Some(&int(2)));
        assert!(tail.next().is_none());
        assert_eq!(outer.more().first(), Some(&int(2)));
    }

    #[test]
    fn seq_chunk_next_nested_lazy_delegates() {
        // `chunk_next` forces through a nested lazy (like
        // `first`/`next` — the JVM forces recursively).
        let items: Vec<Value> = (0i64..40).map(int).collect();
        let chunked = Seq::chunked(items).expect("chunked");
        let inner = Seq::lazy(move || Some(chunked.clone()));
        let outer = Seq::lazy(move || Some(inner.clone()));
        let (head, rest) = outer.chunk_next().expect("delegate");
        assert_eq!(head.len(), 32);
        assert_eq!(rest.first(), Some(&int(32)));
    }

    #[test]
    fn seq_chunk_cons_first_next() {
        let tail = Seq::eager(vec![int(3)]).expect("tail");
        let s = Seq::chunk_cons(vec![int(1), int(2)], tail);
        assert_eq!(s.first(), Some(&int(1)));
        let mid = s.next().expect("mid");
        assert_eq!(mid.first(), Some(&int(2)));
        let rest = mid.next().expect("rest");
        assert_eq!(rest.first(), Some(&int(3)));
        assert!(rest.next().is_none());
    }

    /// The exact range helpers reject `Double` (the walk
    /// dispatches on `is_double` first — the panic is the
    /// contract).
    #[test]
    #[should_panic(expected = "doubles take the f64 walk, never the exact one")]
    fn range_rational_rejects_double() {
        let _ = range_rational(&RangeNum::Double(0.5));
    }

    /// See [`range_rational_rejects_double`].
    #[test]
    #[should_panic(expected = "doubles take the f64 walk, never the exact one")]
    fn range_bigdec_rejects_double() {
        let _ = range_bigdec(&RangeNum::Double(0.5));
    }

    /// See [`range_rational_rejects_double`].
    #[test]
    #[should_panic(expected = "doubles and mixed exact kinds are handled above")]
    fn range_add_int_rejects_double() {
        let _ = range_add_int(&RangeNum::Double(0.5), &RangeNum::Double(0.5));
    }

    /// A zero step never finishes (exact walks only see nonzero
    /// steps from `range` — zero steps repeat instead — so the
    /// pin documents the walk, exercising both `done` arms).
    #[test]
    fn zero_step_exact_walk_never_done() {
        let r = RangeValue::new(
            RangeNum::Int(0),
            Some(RangeNum::Int(5)),
            RangeNum::BigInt(num_bigint::BigInt::from(0)),
            false,
            None,
        );
        assert!(!r.done());
    }

    #[test]
    fn seq_next_crosses_chunks() {
        let items: Vec<Value> = (0i64..33).map(int).collect();
        let mut cur = Seq::chunked(items).expect("chunked");
        for _ in 0..32 {
            cur = cur.next().expect("step");
        }
        assert_eq!(cur.first(), Some(&int(32)));
        assert!(cur.next().is_none());
    }

    #[test]
    fn seq_chunk_next_exhausted_delegates() {
        let items: Vec<Value> = (0i64..40).map(int).collect();
        let tail = Seq::chunked(items).expect("tail");
        let s = Seq::ChunkCons {
            chunk: Arc::from(vec![int(1)]),
            pos: 1,
            tail: Box::new(tail),
        };
        let (head, rest) = s.chunk_next().expect("delegate");
        assert_eq!(head.len(), 32);
        assert_eq!(rest.first(), Some(&int(32)));
    }

    #[test]
    fn seq_kind_jvm_names() {
        assert_eq!(
            SeqKind::Chunked.jvm_name(),
            "clojure.lang.PersistentVector$ChunkedSeq"
        );
        assert_eq!(SeqKind::Array.jvm_name(), "clojure.lang.ArraySeq");
        assert_eq!(SeqKind::Str.jvm_name(), "clojure.lang.StringSeq");
        assert_eq!(
            SeqKind::Map.jvm_name(),
            "clojure.lang.PersistentArrayMap$Seq"
        );
        assert_eq!(
            SeqKind::Key.jvm_name(),
            "clojure.lang.APersistentMap$KeySeq"
        );
        assert_eq!(SeqKind::ChunkedCons.jvm_name(), "clojure.lang.ChunkedCons");
        assert_eq!(SeqKind::Cons.jvm_name(), "clojure.lang.Cons");
        assert_eq!(SeqKind::Lazy.jvm_name(), "clojure.lang.LazySeq");
        assert_eq!(
            SeqKind::Subvec.jvm_name(),
            "clojure.lang.APersistentVector$Seq"
        );
    }

    #[test]
    fn seq_kind_of_shape() {
        let chunked = Seq::chunked(vec![int(1)]).expect("chunked");
        assert_eq!(RealizedSeq::new(chunked).kind(), SeqKind::Chunked);
        assert_eq!(
            RealizedSeq::new(Seq::chunk_cons(vec![int(1)], Seq::empty())).kind(),
            SeqKind::Chunked
        );
        assert_eq!(
            RealizedSeq::new(Seq::cons(int(1), None)).kind(),
            SeqKind::Cons
        );
        assert_eq!(
            RealizedSeq::new(Seq::lazy(|| None::<Seq>)).kind(),
            SeqKind::Lazy
        );
        let eager = Seq::eager(vec![int(1)]).expect("eager");
        assert_eq!(RealizedSeq::new(eager).kind(), SeqKind::Array);
        assert_eq!(RealizedSeq::new(Seq::empty()).kind(), SeqKind::Array);
    }

    #[test]
    fn realized_seq_fallbacks() {
        let seq = RealizedSeq::new(Seq::eager(vec![int(1), int(2)]).expect("seq"));
        assert_eq!(seq.print(), "(1 2)");
        let same = RealizedSeq::new(Seq::eager(vec![int(1), int(2)]).expect("same"));
        assert!(seq.dyn_eq(&same));
        let diff = RealizedSeq::new(Seq::eager(vec![int(1)]).expect("diff"));
        assert!(!seq.dyn_eq(&diff));
        let range = RangeValue::new(
            RangeNum::Int(0),
            Some(RangeNum::Int(2)),
            RangeNum::Int(1),
            true,
            Some(2),
        );
        assert!(!seq.dyn_eq(&range));
        assert_eq!(hash_ext(&seq), hash_ext(&same));
    }

    #[test]
    fn subvector_helpers_and_fallbacks() {
        let backing = Vector::from_vec(vec![int(1), int(2), int(3)]);
        assert!(SubVectorValue::new(backing.clone(), 1, 1).is_empty());
        let sv = SubVectorValue::new(backing, 0, 2);
        assert!(!sv.is_empty());
        assert_eq!(sv.print(), "[1 2]");
        let same = SubVectorValue::new(Vector::from_vec(vec![int(1), int(2)]), 0, 2);
        assert!(sv.dyn_eq(&same));
        let diff = SubVectorValue::new(Vector::from_vec(vec![int(1)]), 0, 1);
        assert!(!sv.dyn_eq(&diff));
        let seq = RealizedSeq::new(Seq::eager(vec![int(1), int(2)]).expect("seq"));
        assert!(!sv.dyn_eq(&seq));
        assert_eq!(hash_ext(&sv), hash_ext(&same));
    }

    #[test]
    fn range_value_meta_done_overflow() {
        let r = RangeValue::new(
            RangeNum::Int(0),
            Some(RangeNum::Int(2)),
            RangeNum::Int(1),
            true,
            Some(2),
        );
        let v = Value::Extension(Arc::new(r));
        assert!(v.meta().is_none());
        let meta = crate::new_map();
        let tagged = v.clone().with_meta(Some(meta.clone()));
        assert_eq!(tagged.meta(), Some(&meta));
        let open = RangeValue::new(RangeNum::Int(0), None, RangeNum::Int(1), false, None);
        assert!(!open.done());
        // Overflow break: the bounds are `f64`-distinct (so
        // `done` is false) but `start + step` overflows `i64` (so
        // the first advance ends the walk). Production builds the
        // `long` path for this count — the shape is constructed
        // directly to pin the break (oracled walk: 1 item).
        let edge = RangeValue::new(
            RangeNum::Int(9223372036854773807),
            Some(RangeNum::Int(9223372036854775807)),
            RangeNum::Int(3000),
            false,
            None,
        );
        assert_eq!(edge.walked_len(), 1);
        let mut out = Vec::new();
        edge.for_each(&mut |item| out.push(item.clone()));
        assert_eq!(out, vec![int(9223372036854773807)]);
    }

    #[test]
    fn range_value_fallbacks() {
        let r = RangeValue::new(
            RangeNum::Double(0.0),
            Some(RangeNum::Double(1.5)),
            RangeNum::Double(0.5),
            false,
            None,
        );
        assert_eq!(r.class_name(), "clojure.lang.Range");
        let same = RangeValue::new(
            RangeNum::Double(0.0),
            Some(RangeNum::Double(1.5)),
            RangeNum::Double(0.5),
            false,
            None,
        );
        assert!(r.dyn_eq(&same));
        let diff = RangeValue::new(
            RangeNum::Double(0.0),
            Some(RangeNum::Double(1.5)),
            RangeNum::Double(1.0),
            false,
            None,
        );
        assert!(!r.dyn_eq(&diff));
        let rep = RepeatValue::new(int(1), Some(2));
        assert!(!r.dyn_eq(&rep));
        assert_eq!(hash_ext(&r), hash_ext(&same));
    }

    #[test]
    fn repeat_value_gaps() {
        let v = Value::Extension(Arc::new(RepeatValue::new(int(7), Some(3))));
        assert!(v.meta().is_none());
        let meta = crate::new_map();
        let tagged = v.clone().with_meta(Some(meta.clone()));
        assert_eq!(tagged.meta(), Some(&meta));
        let r = RepeatValue::new(int(7), Some(3));
        assert_eq!(r.class_name(), "clojure.lang.Repeat");
        assert_eq!(r.print(), "(...)");
        let same = RepeatValue::new(int(7), Some(3));
        assert!(r.dyn_eq(&same));
        let fewer = RepeatValue::new(int(7), Some(2));
        assert!(!r.dyn_eq(&fewer));
        let other = RepeatValue::new(int(8), Some(3));
        assert!(!r.dyn_eq(&other));
        assert_eq!(hash_ext(&r), hash_ext(&same));
    }

    #[test]
    #[should_panic(expected = "stop after two")]
    fn repeat_for_each_infinite_iterates() {
        // The infinite arm never returns (matching the JVM, whose
        // walks hang the same way): iterate twice, then stop with a
        // panic so the loop mechanics genuinely run and the test ends.
        let r = RepeatValue::new(int(1), None);
        let mut count = 0;
        r.for_each(&mut |_| {
            count += 1;
            assert!(count < 3, "stop after two iterations");
        });
    }

    #[test]
    fn repeat_walked_len_infinite_diverges() {
        use std::thread;
        use std::time::Duration;

        // The infinite arm never returns and offers no injection
        // point (its closure is internal): run it on a thread the
        // test abandons, so the arm executes and the process still
        // exits to flush counters. The arm is entered within
        // microseconds of thread start; 50ms is pure margin.
        let r = RepeatValue::new(int(1), None);
        let _handle = thread::spawn(move || r.walked_len());
        thread::sleep(Duration::from_millis(50));
    }

    #[test]
    fn cycle_value_gaps() {
        let v = Value::Extension(Arc::new(CycleValue::new(int(1), int(2))));
        assert!(v.meta().is_none());
        let meta = crate::new_map();
        let tagged = v.clone().with_meta(Some(meta.clone()));
        assert_eq!(tagged.meta(), Some(&meta));
        let c = CycleValue::new(int(1), int(2));
        assert_eq!(c.class_name(), "clojure.lang.Cycle");
        assert_eq!(c.print(), "(...)");
        // Identity equality (same pointer is `=`, distinct
        // instances are not, whatever the items).
        assert!(c.dyn_eq(&c));
        let d = CycleValue::new(int(1), int(2));
        assert!(!c.dyn_eq(&d));
    }

    #[test]
    fn iterate_value_gaps() {
        let v = Value::Extension(Arc::new(IterateValue::new(int(0), int(1))));
        assert!(v.meta().is_none());
        let meta = crate::new_map();
        let tagged = v.clone().with_meta(Some(meta.clone()));
        assert_eq!(tagged.meta(), Some(&meta));
        let it = IterateValue::new(int(0), int(1));
        let fresh = it.clone();
        assert_eq!(fresh.seed_opt(), Some(int(1)));
        it.store_seed(int(9));
        let copied = it.clone();
        assert_eq!(copied.seed_opt(), Some(int(9)));
        let pending = IterateValue::deferred(int(0), int(1));
        assert!(pending.clone().seed_opt().is_none());
        assert_eq!(it.class_name(), "clojure.lang.Iterate");
        assert_eq!(it.print(), "(...)");
        // Identity equality (same pointer is `=`, distinct
        // instances are not, whatever the seeds).
        assert!(it.dyn_eq(&it));
        let other = IterateValue::new(int(0), int(1));
        assert!(!it.dyn_eq(&other));
    }

    #[test]
    fn cons_value_gaps() {
        let v = Value::Extension(Arc::new(ConsValue::new(int(1), int(2))));
        assert!(v.meta().is_none());
        let meta = crate::new_map();
        let tagged = v.clone().with_meta(Some(meta.clone()));
        assert_eq!(tagged.meta(), Some(&meta));
        let c = ConsValue::new(int(1), int(2));
        assert_eq!(c.print(), "(...)");
        assert!(c.dyn_eq(&c));
        let twin = ConsValue::new(int(1), int(2));
        assert!(!c.dyn_eq(&twin));
        let hashed = Value::Extension(Arc::new(c));
        assert_eq!(hash_val(&hashed), hash_val(&hashed));
    }

    #[test]
    fn entry_seq_gaps() {
        let k = Value::Extension(Arc::new(KeySeqValue::new(int(1))));
        assert!(k.meta().is_none());
        let meta = crate::new_map();
        assert_eq!(k.clone().with_meta(Some(meta.clone())).meta(), Some(&meta));
        let key = KeySeqValue::new(int(1));
        assert_eq!(key.class_name(), "clojure.lang.APersistentMap$KeySeq");
        assert_eq!(key.print(), "(...)");
        assert!(key.dyn_eq(&key));
        let other_key = KeySeqValue::new(int(1));
        assert!(!key.dyn_eq(&other_key));
        assert_eq!(hash_val(&k), hash_val(&k));
        let g = Value::Extension(Arc::new(ValSeqValue::new(int(1))));
        assert_eq!(g.clone().with_meta(Some(meta.clone())).meta(), Some(&meta));
        let val = ValSeqValue::new(int(1));
        assert_eq!(val.class_name(), "clojure.lang.APersistentMap$ValSeq");
        assert_eq!(val.print(), "(...)");
        assert!(val.dyn_eq(&val));
        let other_val = ValSeqValue::new(int(1));
        assert!(!val.dyn_eq(&other_val));
        assert_eq!(hash_val(&g), hash_val(&g));
    }

    /// #226: O(1) vector and string seqs — their shapes, `Debug`,
    /// counts, chunk boundaries (borrowed from generic storage, boxed
    /// from packed), and the in-place step (`RealizedSeq::advance`)
    /// over every shape, including the ends that leave a seq as is.
    #[test]
    fn vector_and_string_seqs_step_in_place() {
        let generic = Vector::from_vec(vec![Value::Nil; 40]);
        let packed = Vector::from_vec(ints(0..40));
        assert!(packed.is_packed() && !generic.is_packed());
        for v in [&generic, &packed] {
            let s = Seq::vector_from(v.clone(), 0).expect("non-empty");
            assert_eq!(SeqKind::of_shape(&s), SeqKind::Chunked);
            assert_eq!(format!("{s:?}"), "Vec { len: 40, pos: 0 }");
            assert_eq!(s.count(), 40);
            let (chunk, rest) = s.chunk_next().expect("vector seqs chunk");
            assert_eq!(chunk.len(), 32);
            assert_eq!(rest.count(), 8);
            let (tail, end) = rest.chunk_next().expect("the last chunk");
            assert_eq!(tail.len(), 8);
            assert!(end.is_empty());
        }
        assert!(Seq::vector_from(packed.clone(), 40).is_none());
        assert!(packed.chunk_from(41).is_empty());
        let s = Seq::string_from(Arc::from("aé"), 0, 0).expect("non-empty");
        assert_eq!(SeqKind::of_shape(&s), SeqKind::Str);
        assert_eq!(format!("{s:?}"), "Str { len: 3, pos: 0 }");
        assert_eq!(s.count(), 2);
        assert!(Seq::string_from(Arc::from("aé"), 2, 1).is_none());
        // In place: vector and string seqs walk to their last item and
        // stop there; metadata survives string seqs only.
        let meta = crate::Map::from_kvs_last_wins(vec![(int(1), int(2))]);
        let mut r = RealizedSeq::flagged(
            Seq::string_from(Arc::from("xyz"), 0, 0).expect("chars"),
            true,
            true,
            SeqKind::Str,
        )
        .with_meta(Some(meta.clone()));
        assert!(r.advance() && r.advance());
        assert!(!r.advance());
        assert_eq!(r.seq().first(), Some(&Value::Char('z')));
        assert!(r.meta().is_some());
        let mut r = RealizedSeq::flagged(
            Seq::vector_from(packed, 38).expect("items"),
            true,
            true,
            SeqKind::Chunked,
        )
        .with_meta(Some(meta));
        assert!(r.advance());
        assert!(!r.advance());
        assert_eq!(r.seq().first(), Some(&int(39)));
        assert!(r.meta().is_none());
        // Other shapes step to their `next`, and stop at the end.
        let mut r = RealizedSeq::new(Seq::eager(ints(0..2)).expect("items"));
        assert!(r.advance());
        assert!(!r.advance());
        assert_eq!(r.seq().first(), Some(&int(1)));
        let mut cell = RealizedSeq::new(Seq::empty());
        assert!(cell.as_any_mut().is_some());
    }
}
