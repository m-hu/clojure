// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Java arrays (S15): typed, fixed-length, mutable values with
//! identity equality. An [`Array`] is a descriptor (base type +
//! dimensions, `into-array` narrowing tag) plus a shared cell, so
//! `aset` mutates through every handle, like the JVM. Nesting is
//! structural: `dims > 1` stores `Option<Array>` slots (`None` is
//! a nil inner, except rectangular `make-array` which pre-fills).

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::Value;

static ARRAY_ID: AtomicU64 = AtomicU64::new(1);

/// Payload bytes past which allocation throws
/// `OutOfMemoryError` instead of attempting the `Vec` (S15:
/// multi-dim products would otherwise thrash the machine; the
/// JVM OOMs from its heap the same way).
pub const MAX_ARRAY_BYTES: u64 = 256 * 1024 * 1024;

/// The nine JVM element types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseType {
    Boolean,
    Byte,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
    Object,
}

impl BaseType {
    /// The descriptor letter (`ZBC SJIFD`, `L…;` built by the caller).
    #[must_use]
    pub fn letter(self) -> &'static str {
        match self {
            Self::Boolean => "Z",
            Self::Byte => "B",
            Self::Char => "C",
            Self::Short => "S",
            Self::Int => "I",
            Self::Long => "J",
            Self::Float => "F",
            Self::Double => "D",
            Self::Object => "Ljava.lang.Object;",
        }
    }

    /// Payload bytes per slot (for the allocation guard).
    #[must_use]
    pub fn slot_bytes(self) -> u64 {
        match self {
            Self::Boolean | Self::Byte => 1,
            Self::Char | Self::Short => 2,
            Self::Int | Self::Float => 4,
            Self::Long | Self::Double => 8,
            // A pointer per slot (values/arrays live behind it).
            Self::Object => 8,
        }
    }
}

/// The full array type: base + dimensions + the `into-array`
/// narrowing tag (`Some` JVM class name, e.g.
/// `java.lang.Long`, for inferred reference arrays — `None` is
/// plain `Object`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayDesc {
    /// The element base type.
    pub base: BaseType,
    /// Dimension count (`>= 1`).
    pub dims: u32,
    /// Narrowed reference component (dims-1 `into-array` only).
    pub component: Option<Arc<str>>,
}

impl ArrayDesc {
    /// A 1-dim descriptor over `base`.
    #[must_use]
    pub fn flat(base: BaseType) -> Self {
        Self {
            base,
            dims: 1,
            component: None,
        }
    }

    /// The JVM descriptor (`[I`, `[[I`, `[Ljava.lang.Object;`,
    /// `[Ljava.lang.Long;`, …).
    #[must_use]
    pub fn descriptor(&self) -> String {
        let mut out = "[".repeat(self.dims as usize);
        match self.base {
            BaseType::Object => match &self.component {
                None => out.push_str("Ljava.lang.Object;"),
                Some(c) => {
                    out.push('L');
                    out.push_str(c);
                    out.push(';');
                }
            },
            base => out.push_str(base.letter()),
        }
        out
    }

    /// The expected inner descriptor (`dims - 1`, `None` at
    /// dims 1 — callers only invoke this past the dims check).
    #[must_use]
    pub fn inner(&self) -> Self {
        Self {
            base: self.base,
            dims: self.dims - 1,
            component: None,
        }
    }
}

/// The mutable payload, one variant per base type (`Floats`
/// stores `f64` — the port has no `f32`, like its `float`
/// native; `Arrays` slots are `None` until set).
#[derive(Clone, Debug)]
pub enum ArrayStore {
    /// Boolean slots.
    Bools(Vec<bool>),
    /// Byte slots.
    Bytes(Vec<i8>),
    /// Char slots (never lone surrogates — `char` rejects them).
    Chars(Vec<char>),
    /// Short slots.
    Shorts(Vec<i16>),
    /// Int slots.
    Ints(Vec<i32>),
    /// Long slots.
    Longs(Vec<i64>),
    /// Float slots (`f64`, see above).
    Floats(Vec<f64>),
    /// Double slots.
    Doubles(Vec<f64>),
    /// Reference slots.
    Objs(Vec<Value>),
    /// Inner arrays (`dims > 1`).
    Arrays(Vec<Option<Array>>),
}

impl ArrayStore {
    /// The slot count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Bools(v) => v.len(),
            Self::Bytes(v) => v.len(),
            Self::Chars(v) => v.len(),
            Self::Shorts(v) => v.len(),
            Self::Ints(v) => v.len(),
            Self::Longs(v) => v.len(),
            Self::Floats(v) => v.len(),
            Self::Doubles(v) => v.len(),
            Self::Objs(v) => v.len(),
            Self::Arrays(v) => v.len(),
        }
    }

    /// Whether the store holds no slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A Java array value: identity plus a shared mutable cell.
#[derive(Clone, Debug)]
pub struct Array {
    id: u64,
    desc: ArrayDesc,
    cell: Arc<Mutex<ArrayStore>>,
}

impl Array {
    /// An array over `store` (fresh identity; the descriptor
    /// must match the store variant — debug-asserted).
    #[must_use]
    pub fn new(desc: ArrayDesc, store: ArrayStore) -> Self {
        debug_assert_eq!(desc.dims > 1, matches!(store, ArrayStore::Arrays(_)));
        Self {
            id: ARRAY_ID.fetch_add(1, Ordering::Relaxed),
            desc,
            cell: Arc::new(Mutex::new(store)),
        }
    }

    /// The print/hash identity.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The full type descriptor.
    #[must_use]
    pub fn desc(&self) -> &ArrayDesc {
        &self.desc
    }

    /// The JVM descriptor string (`[I`, …).
    #[must_use]
    pub fn descriptor(&self) -> String {
        self.desc.descriptor()
    }

    /// Whether `other` shares this array's cell (`=`).
    #[must_use]
    pub fn same_cell(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cell, &other.cell)
    }

    /// Lock the cell (short critical sections only; poison is
    /// unreachable — natives never panic under the lock).
    pub fn lock(&self) -> MutexGuard<'_, ArrayStore> {
        self.cell.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The outer slot count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the outer dimension is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Every slot boxed (`nil` for unset inners), under one
    /// lock (the `seq`/`to-array` snapshot).
    #[must_use]
    pub fn snapshot(&self) -> Vec<Value> {
        match &*self.lock() {
            ArrayStore::Bools(v) => v.iter().map(|b| Value::Bool(*b)).collect(),
            ArrayStore::Chars(v) => v.iter().map(|c| Value::Char(*c)).collect(),
            ArrayStore::Floats(v) | ArrayStore::Doubles(v) => {
                v.iter().map(|d| Value::Double(*d)).collect()
            }
            ArrayStore::Objs(v) => v.clone(),
            ArrayStore::Arrays(v) => v
                .iter()
                .map(|o| o.clone().map(Value::Array).unwrap_or(Value::Nil))
                .collect(),
            // The integer stores box through `snapshot_int_store`
            // (split for the complexity budget).
            other => snapshot_int_store(other),
        }
    }

    /// A shallow copy: fresh identity and cell, same
    /// descriptor, cloned slots (`aclone` — inners shared).
    #[must_use]
    pub fn shallow_clone(&self) -> Self {
        Self::new(self.desc.clone(), self.lock().clone())
    }
}

/// The integer-store arms of [`Array::snapshot`] (split for the
/// complexity budget): every slot widened to `Int`.
fn snapshot_int_store(store: &ArrayStore) -> Vec<Value> {
    match store {
        ArrayStore::Bytes(v) => v.iter().map(|b| Value::Int(i64::from(*b))).collect(),
        ArrayStore::Shorts(v) => v.iter().map(|s| Value::Int(i64::from(*s))).collect(),
        ArrayStore::Ints(v) => v.iter().map(|i| Value::Int(i64::from(*i))).collect(),
        ArrayStore::Longs(v) => v.iter().map(|l| Value::Int(*l)).collect(),
        // All other stores are handled upstream (the single
        // caller partitions the variants).
        _ => unreachable!("snapshot partition"),
    }
}
