// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! The persistent vector trie (#226): Clojure's `PersistentVector`
//! layout — a 32-way trie of full leaves plus a tail of the last 1–32
//! items — with the items stored inline in the leaves.
//!
//! Every handle shares structure through `Arc`s; the `*_mut` writers
//! update through `Arc::make_mut`, so a uniquely owned vector mutates
//! in place while one sharing nodes with other versions path-copies
//! only the shared nodes (the `rpds` contract this replaces, without
//! its per-item allocation and with an O(1) tail for appends and
//! reads near the end).

use std::sync::Arc;

use crate::Value;

const BITS: u32 = 5;
const WIDTH: usize = 1 << BITS;
const MASK: usize = WIDTH - 1;

/// A trie node: a branch's children, or a full leaf's items (the
/// other side empty — the node's level says which, by construction).
#[derive(Clone, Debug)]
struct Node<T> {
    kids: Vec<Arc<Node<T>>>,
    items: Vec<T>,
}

impl<T> Node<T> {
    fn branch(kids: Vec<Arc<Node<T>>>) -> Self {
        Self {
            kids,
            items: Vec::new(),
        }
    }

    fn leaf(items: Vec<T>) -> Self {
        Self {
            kids: Vec::new(),
            items,
        }
    }
}

/// A persistent vector of values (or of a packed scalar type: the
/// typed `Vector` storages keep `i64`/`char` items in the same trie).
#[derive(Clone, Debug)]
pub struct PVec<T = Value> {
    len: usize,
    /// The root's level: its children index by `(i >> shift) & MASK`
    /// (`BITS` when the root's children are leaves).
    shift: u32,
    /// The trie of full leaves (`None` while every item is in the
    /// tail).
    root: Option<Arc<Node<T>>>,
    tail: Arc<Vec<T>>,
}

impl<T: Clone> Default for PVec<T> {
    fn default() -> Self {
        Self::new_sync()
    }
}

impl<T: Clone> PVec<T> {
    /// An empty vector.
    #[must_use]
    pub fn new_sync() -> Self {
        Self {
            len: 0,
            shift: BITS,
            root: None,
            tail: Arc::new(Vec::new()),
        }
    }

    /// A vector over `items` in order (up to 32 become the tail as
    /// is — no per-item push).
    #[must_use]
    pub fn from_vec(items: Vec<T>) -> Self {
        if items.len() <= WIDTH {
            return Self {
                len: items.len(),
                shift: BITS,
                root: None,
                tail: Arc::new(items),
            };
        }
        items.into_iter().collect()
    }

    /// The item count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Where the tail starts.
    fn tail_offset(&self) -> usize {
        self.len - self.tail.len()
    }

    /// The full leaf holding index `i` (below the tail offset).
    fn leaf(&self, i: usize) -> &[T] {
        let mut node = self
            .root
            .as_deref()
            .expect("indexes below the tail have a trie");
        let mut level = self.shift;
        while level > 0 {
            node = &node.kids[(i >> level) & MASK];
            level -= BITS;
        }
        &node.items
    }

    /// The item at `i` (`None` past the end).
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len {
            return None;
        }
        let off = self.tail_offset();
        if i >= off {
            return self.tail.get(i - off);
        }
        self.leaf(i).get(i & MASK)
    }

    /// The items from `i` to the end of the leaf (or tail) holding
    /// it, borrowed (empty past the end): a vector seq's current
    /// chunk. Leaves and the tail start on multiples of 32, so the
    /// chunks of a walk from 0 are the JVM's `ChunkedSeq` chunks.
    #[must_use]
    pub fn chunk_from(&self, i: usize) -> &[T] {
        if i >= self.len {
            return &[];
        }
        let off = self.tail_offset();
        if i >= off {
            return &self.tail[i - off..];
        }
        &self.leaf(i)[i & MASK..]
    }

    /// Append an item.
    pub fn push_back_mut(&mut self, v: T) {
        if self.tail.len() < WIDTH {
            Arc::make_mut(&mut self.tail).push(v);
            self.len += 1;
            return;
        }
        // The full tail moves into the trie as a leaf.
        let full = std::mem::replace(&mut self.tail, Arc::new(Vec::with_capacity(WIDTH)));
        let leaf = Arc::new(Node::leaf(Arc::unwrap_or_clone(full)));
        let at = self.len - WIDTH;
        match self.root.take() {
            None => self.root = Some(Arc::new(Node::branch(vec![leaf]))),
            Some(root) if (at >> BITS) >= (1 << self.shift) => {
                // The root is full: a new level over it.
                let path = new_path(self.shift, leaf);
                self.root = Some(Arc::new(Node::branch(vec![root, path])));
                self.shift += BITS;
            }
            Some(mut root) => {
                push_leaf(Arc::make_mut(&mut root), self.shift, at, leaf);
                self.root = Some(root);
            }
        }
        Arc::make_mut(&mut self.tail).push(v);
        self.len += 1;
    }

    /// Replace the item at `i` (`false` when out of bounds).
    pub fn set_mut(&mut self, i: usize, v: T) -> bool {
        if i >= self.len {
            return false;
        }
        let off = self.tail_offset();
        if i >= off {
            Arc::make_mut(&mut self.tail)[i - off] = v;
            return true;
        }
        let mut node = Arc::make_mut(
            self.root
                .as_mut()
                .expect("indexes below the tail have a trie"),
        );
        let mut level = self.shift;
        while level > 0 {
            node = Arc::make_mut(&mut node.kids[(i >> level) & MASK]);
            level -= BITS;
        }
        node.items[i & MASK] = v;
        true
    }

    /// Drop the last item (`false` when empty).
    pub fn drop_last_mut(&mut self) -> bool {
        if self.len == 0 {
            return false;
        }
        if self.tail.len() > 1 || self.root.is_none() {
            Arc::make_mut(&mut self.tail).pop();
            self.len -= 1;
            return true;
        }
        // The tail empties: the trie's last leaf becomes the tail.
        let mut root = self.root.take().expect("guarded above");
        let leaf = pop_leaf(Arc::make_mut(&mut root), self.shift);
        self.tail = Arc::new(leaf);
        self.len -= 1;
        // A root left with one child sheds a level (no root at all
        // when the trie emptied).
        self.root = match root.kids.len() {
            0 => None,
            1 if self.shift > BITS => {
                self.shift -= BITS;
                Some(Arc::clone(&root.kids[0]))
            }
            _ => Some(root),
        };
        true
    }

    /// Items front to back.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            vec: self,
            front: 0,
            back: self.len,
            front_leaf: &[],
            front_base: 0,
            back_leaf: &[],
            back_base: 0,
        }
    }
}

/// A path of branches from `level` down to `leaf`.
fn new_path<T>(level: u32, leaf: Arc<Node<T>>) -> Arc<Node<T>> {
    if level == 0 {
        return leaf;
    }
    Arc::new(Node::branch(vec![new_path(level - BITS, leaf)]))
}

/// Push a full leaf for index `at` into `node` (a branch at `level`
/// with room below it).
fn push_leaf<T: Clone>(node: &mut Node<T>, level: u32, at: usize, leaf: Arc<Node<T>>) {
    let kids = &mut node.kids;
    let slot = (at >> level) & MASK;
    if level == BITS {
        kids.push(leaf);
    } else if slot < kids.len() {
        push_leaf(Arc::make_mut(&mut kids[slot]), level - BITS, at, leaf);
    } else {
        kids.push(new_path(level - BITS, leaf));
    }
}

/// Remove the last leaf under `node` (a branch at `level`), pruning
/// branches it empties; answers the leaf's items.
fn pop_leaf<T: Clone>(node: &mut Node<T>, level: u32) -> Vec<T> {
    let kids = &mut node.kids;
    if level == BITS {
        let leaf = kids.pop().expect("a branch holds at least one child");
        return Arc::unwrap_or_clone(leaf).items;
    }
    let last = kids.last_mut().expect("a branch holds at least one child");
    let items = pop_leaf(Arc::make_mut(last), level - BITS);
    if last.kids.is_empty() {
        kids.pop();
    }
    items
}

/// Front-to-back item iteration (double-ended), one leaf lookup per
/// 32 items.
pub struct Iter<'a, T = Value> {
    vec: &'a PVec<T>,
    front: usize,
    back: usize,
    front_leaf: &'a [T],
    front_base: usize,
    back_leaf: &'a [T],
    back_base: usize,
}

impl<'a, T: Clone> Iter<'a, T> {
    /// The leaf (or tail) holding index `i`, with its first index.
    fn chunk(&self, i: usize) -> (&'a [T], usize) {
        let off = self.vec.tail_offset();
        if i >= off {
            (&self.vec.tail[..], off)
        } else {
            (self.vec.leaf(i), i & !MASK)
        }
    }
}

impl<'a, T: Clone> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.front >= self.back {
            return None;
        }
        let i = self.front;
        if i < self.front_base || i >= self.front_base + self.front_leaf.len() {
            (self.front_leaf, self.front_base) = self.chunk(i);
        }
        self.front += 1;
        Some(&self.front_leaf[i - self.front_base])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.back - self.front;
        (n, Some(n))
    }
}

impl<T: Clone> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let i = self.back - 1;
        if i < self.back_base || i >= self.back_base + self.back_leaf.len() {
            (self.back_leaf, self.back_base) = self.chunk(i);
        }
        self.back = i;
        Some(&self.back_leaf[i - self.back_base])
    }
}

impl<T: Clone> ExactSizeIterator for Iter<'_, T> {}

impl<T: Clone> FromIterator<T> for PVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        let mut out = PVec::new_sync();
        for v in items {
            out.push_back_mut(v);
        }
        out
    }
}

/// Longest flat packed run: past it packed items live in the trie.
/// A shared update clones a flat run whole (`Arc::make_mut`), and
/// loop-threaded vectors are shared at update time, so an unbounded
/// flat run turns `conj`/`assoc` chains quadratic (#227); at or below
/// this a shared update copies at most 8 KB, and reads index directly.
pub(crate) const FLAT_MAX: usize = 1024;

/// Packed scalar storage (`i64`/`char` vector items, #226): a flat
/// run up to [`FLAT_MAX`] items (direct reads, in-place writes when
/// unique), the trie past it (log-time shared updates, still 8 or 4
/// bytes per item).
#[derive(Clone, Debug)]
pub(crate) enum Packed<T> {
    Flat(Arc<Vec<T>>),
    Trie(PVec<T>),
}

impl<T: Copy> Packed<T> {
    /// Packed storage over `items` in order.
    pub(crate) fn from_vec(items: Vec<T>) -> Self {
        if items.len() <= FLAT_MAX {
            Packed::Flat(Arc::new(items))
        } else {
            Packed::Trie(PVec::from_vec(items))
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Packed::Flat(a) => a.len(),
            Packed::Trie(t) => t.len(),
        }
    }

    pub(crate) fn get(&self, i: usize) -> Option<T> {
        match self {
            Packed::Flat(a) => a.get(i).copied(),
            Packed::Trie(t) => t.get(i).copied(),
        }
    }

    /// Append (a full flat run moves into the trie first).
    pub(crate) fn push(&mut self, x: T) {
        match self {
            Packed::Flat(a) if a.len() < FLAT_MAX => Arc::make_mut(a).push(x),
            Packed::Flat(a) => {
                let mut t: PVec<T> = a.iter().copied().collect();
                t.push_back_mut(x);
                *self = Packed::Trie(t);
            }
            Packed::Trie(t) => t.push_back_mut(x),
        }
    }

    /// Replace the item at `i` (`false` when out of bounds).
    pub(crate) fn set(&mut self, i: usize, x: T) -> bool {
        match self {
            Packed::Flat(a) => match Arc::make_mut(a).get_mut(i) {
                Some(slot) => {
                    *slot = x;
                    true
                }
                None => false,
            },
            Packed::Trie(t) => t.set_mut(i, x),
        }
    }

    /// Drop the last item (`false` when empty).
    pub(crate) fn pop(&mut self) -> bool {
        match self {
            Packed::Flat(a) => !a.is_empty() && Arc::make_mut(a).pop().is_some(),
            Packed::Trie(t) => t.drop_last_mut(),
        }
    }

    /// The items from `i` to the end of its 32-item chunk.
    pub(crate) fn chunk_from(&self, i: usize) -> &[T] {
        match self {
            Packed::Flat(a) => {
                let end = ((i / WIDTH + 1) * WIDTH).min(a.len());
                a.get(i..end).unwrap_or(&[])
            }
            Packed::Trie(t) => t.chunk_from(i),
        }
    }

    pub(crate) fn iter(&self) -> PackedIter<'_, T> {
        match self {
            Packed::Flat(a) => PackedIter::Flat(a.iter()),
            Packed::Trie(t) => PackedIter::Trie(t.iter()),
        }
    }
}

/// Front-to-back (double-ended) iteration over [`Packed`] storage.
pub(crate) enum PackedIter<'a, T> {
    Flat(std::slice::Iter<'a, T>),
    Trie(Iter<'a, T>),
}

impl<'a, T: Clone> Iterator for PackedIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match self {
            PackedIter::Flat(it) => it.next(),
            PackedIter::Trie(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            PackedIter::Flat(it) => it.size_hint(),
            PackedIter::Trie(it) => it.size_hint(),
        }
    }
}

impl<T: Clone> DoubleEndedIterator for PackedIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            PackedIter::Flat(it) => it.next_back(),
            PackedIter::Trie(it) => it.next_back(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &PVec) -> Vec<i64> {
        v.iter()
            .filter_map(|x| match x {
                Value::Int(n) => Some(*n),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn grows_reads_and_shrinks_across_levels() {
        // Past 32 + 32*32 + 32*32*32 items: three trie levels plus
        // the tail, then all the way back down.
        let n = 40_000usize;
        let mut v = PVec::new_sync();
        let mut versions = Vec::new();
        for i in 0..n {
            if i % 9_973 == 0 {
                versions.push((i, v.clone()));
            }
            v.push_back_mut(Value::Int(i as i64));
        }
        assert_eq!(v.len(), n);
        assert!(!v.is_empty());
        for i in [0, 31, 32, 1_055, 1_056, 33_823, 33_824, n - 1] {
            assert!(matches!(v.get(i), Some(Value::Int(x)) if *x == i as i64));
        }
        assert!(v.get(n).is_none());
        assert_eq!(ints(&v), (0..n as i64).collect::<Vec<_>>());
        let back: Vec<i64> = ints(&v.iter().rev().cloned().collect());
        assert_eq!(back, (0..n as i64).rev().collect::<Vec<_>>());
        // Older versions are untouched by later pushes.
        for (len, old) in &versions {
            assert_eq!(old.len(), *len);
            assert_eq!(ints(old), (0..*len as i64).collect::<Vec<_>>());
        }
        let shared = v.clone();
        assert!(v.set_mut(5, Value::Int(-5)));
        assert!(v.set_mut(n - 1, Value::Int(-1)));
        assert!(!v.set_mut(n, Value::Nil));
        assert!(matches!(v.get(5), Some(Value::Int(-5))));
        assert!(matches!(shared.get(5), Some(Value::Int(5))));
        while v.drop_last_mut() {}
        assert!(v.is_empty());
        assert!(!v.drop_last_mut());
        assert_eq!(ints(&shared).len(), n);
    }

    #[test]
    fn iterates_from_both_ends() {
        let v: PVec = (0..100).map(Value::Int).collect();
        let mut it = v.iter();
        assert_eq!(it.len(), 100);
        assert!(matches!(it.next(), Some(Value::Int(0))));
        assert!(matches!(it.next_back(), Some(Value::Int(99))));
        let rest: Vec<_> = it.cloned().collect();
        assert_eq!(rest.len(), 98);
        let mut it = v.iter();
        for _ in 0..50 {
            it.next_back();
        }
        for _ in 0..50 {
            it.next();
        }
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());
        assert_eq!(PVec::<Value>::default().len(), 0);
        assert!(format!("{:?}", PVec::<Value>::new_sync()).contains("PVec"));
        let small = PVec::from_vec(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(ints(&small), vec![1, 2]);
        let big = PVec::from_vec((0..40).map(Value::Int).collect());
        assert_eq!(ints(&big), (0..40).collect::<Vec<_>>());
        let mixed = PVec::from_vec(vec![Value::Nil, Value::Int(3)]);
        assert_eq!(ints(&mixed), vec![3]);
    }

    #[test]
    fn chunks_borrow_leaves_and_tail() {
        let v: PVec = (0..70).map(Value::Int).collect();
        // A leaf from its middle, the second leaf, then the tail.
        assert_eq!(v.chunk_from(5).len(), 27);
        assert_eq!(v.chunk_from(32).len(), 32);
        assert_eq!(v.chunk_from(65).len(), 5);
        assert!(v.chunk_from(70).is_empty());
    }

    /// [`Packed`]: flat up to `FLAT_MAX` items, the trie past it —
    /// every operation on both shapes, and the move between them.
    #[test]
    fn packed_runs_go_flat_then_trie() {
        let mut p: Packed<i64> = Packed::from_vec((0..FLAT_MAX as i64).collect());
        assert!(matches!(p, Packed::Flat(_)));
        assert_eq!(p.get(5), Some(5));
        assert!(p.set(5, -5) && p.get(5) == Some(-5));
        assert!(!p.set(FLAT_MAX, 0));
        assert_eq!(p.chunk_from(30), &[30, 31][..]);
        assert!(p.chunk_from(FLAT_MAX).is_empty());
        assert_eq!(p.iter().size_hint(), (FLAT_MAX, Some(FLAT_MAX)));
        assert_eq!(p.iter().next_back(), Some(&(FLAT_MAX as i64 - 1)));
        assert!(p.pop() && p.len() == FLAT_MAX - 1);
        p.push(7);
        p.push(8);
        assert!(matches!(p, Packed::Trie(_)));
        assert_eq!(p.len(), FLAT_MAX + 1);
        assert_eq!(p.get(FLAT_MAX), Some(8));
        assert!(p.set(0, 9) && p.get(0) == Some(9));
        assert_eq!(p.chunk_from(FLAT_MAX - 1).len(), 1);
        assert_eq!(p.chunk_from(FLAT_MAX).len(), 1);
        assert_eq!(p.iter().size_hint().0, FLAT_MAX + 1);
        assert_eq!(p.iter().next_back(), Some(&8));
        assert_eq!(p.iter().next(), Some(&9));
        p.push(10);
        assert!(p.pop() && p.len() == FLAT_MAX + 1);
        let big: Packed<char> = Packed::from_vec(vec!['x'; FLAT_MAX + 1]);
        assert!(matches!(big, Packed::Trie(_)));
        let mut empty: Packed<char> = Packed::from_vec(Vec::new());
        assert!(!empty.pop());
    }
}
