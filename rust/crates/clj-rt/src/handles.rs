// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Generational handle table: the integer-only FFI substrate (Z3).
//!
//! No pointer crosses the boundary — every value is a `u64`
//! handle. `0` is the null/error sentinel; nonzero handles split
//! by low 3 tag bits into inline scalars (no table traffic) and
//! generational slab references:
//!
//! | Tag   | Meaning                              |
//! |-------|--------------------------------------|
//! | `000` | reserved (`0` itself is the sentinel)|
//! | `001` | value slot (`index \|\| generation`)  |
//! | `010` | inline scalar (nil/false/true)       |
//! | `011` | inline char (21-bit payload)         |
//! | `100` | inline small-int (61-bit payload)    |
//! | `101` | arg-accumulator slot                 |
//! | `110` | string-builder slot                  |
//! | `111` | reserved (never minted)              |
//!
//! Table payloads pack a 32-bit slot index (bits 3..35) with a
//! 29-bit generation (bits 35..64). A generation mismatch
//! (use-after-drop, double-drop) resolves to stale —
//! deterministic graceful degradation, never UB. Small ints
//! cover ±2^60 (full-range `i64` and doubles ride the table — a
//! full `i64` alone consumes all 2^64 codepoints, so no tagged
//! scheme can inline it; NaN-boxing is out per the spec).
//!
//! Every function here is total (no panics): invalid handles
//! resolve to `None`/`false`, and callers degrade gracefully.

// The value-handle consumers (`resolve`, `clone`/`drop`, table
// `get`) land with the flip commit (ii) — until then they are
// exercised by the unit tests only, so the rlib build allows
// the temporary dead code (libtest keeps the full deny).
#![cfg_attr(not(test), allow(dead_code))]

use clj_value::Value;

/// Low-3-bit tag mask.
const TAG_MASK: u64 = 0b111;
/// Reserved: only `0` (the sentinel) is valid here.
const TAG_RESERVED0: u64 = 0b000;
/// Value slot (`index || generation`).
pub(crate) const TAG_VALUE: u64 = 0b001;
/// Inline scalar (nil/false/true by payload).
const TAG_SCALAR: u64 = 0b010;
/// Inline char.
const TAG_CHAR: u64 = 0b011;
/// Inline small-int (61-bit two's complement).
const TAG_INT: u64 = 0b100;
/// Arg-accumulator slot.
pub(crate) const TAG_ACC: u64 = 0b101;
/// String-builder slot.
pub(crate) const TAG_BUILDER: u64 = 0b110;
/// Reserved (never minted).
const TAG_RESERVED7: u64 = 0b111;

/// Scalar payloads under [`TAG_SCALAR`].
const SCALAR_NIL: u64 = 0;
const SCALAR_FALSE: u64 = 1;
const SCALAR_TRUE: u64 = 2;

/// Inline small-int bounds (±2^60 — every smaller magnitude
/// round-trips through the 61-bit payload).
pub(crate) const SMALL_MIN: i64 = -(1 << 60);
pub(crate) const SMALL_MAX: i64 = (1 << 60) - 1;

/// Inline payload bits (61) below the generation/tag split.
const PAYLOAD_MASK: u64 = (1 << 61) - 1;

/// Generation bits (29) carried in a table handle.
const GEN_BITS: u32 = 29;
const GEN_MASK: u32 = (1 << GEN_BITS) - 1;
/// Payload bit where the generation starts (3 tag + 32 index).
const GEN_SHIFT: u32 = 35;

/// Encode an inline scalar (nil/bool/small-int/char); `None`
/// means table-bound (everything else, including full-range
/// ints and doubles).
#[must_use]
pub(crate) fn encode_inline(value: &Value) -> Option<u64> {
    match value {
        Value::Nil => Some(TAG_SCALAR | (SCALAR_NIL << 3)),
        Value::Bool(false) => Some(TAG_SCALAR | (SCALAR_FALSE << 3)),
        Value::Bool(true) => Some(TAG_SCALAR | (SCALAR_TRUE << 3)),
        Value::Int(n) if (SMALL_MIN..=SMALL_MAX).contains(n) => {
            Some((((*n as u64) & PAYLOAD_MASK) << 3) | TAG_INT)
        }
        Value::Char(c) => Some((u64::from(*c as u32) << 3) | TAG_CHAR),
        _ => None,
    }
}

/// Decode one scalar payload (nil/false/true); anything else is
/// not an inline handle.
fn decode_scalar(payload: u64) -> Option<Value> {
    match payload {
        SCALAR_NIL => Some(Value::Nil),
        SCALAR_FALSE => Some(Value::Bool(false)),
        SCALAR_TRUE => Some(Value::Bool(true)),
        _ => None,
    }
}

/// Decode an inline handle; `None` means table-bound or invalid
/// (the caller resolves table tags next, else treats it stale).
#[must_use]
pub(crate) fn decode_inline(handle: u64) -> Option<Value> {
    match handle & TAG_MASK {
        TAG_INT => Some(Value::Int((handle as i64) >> 3)),
        TAG_SCALAR => decode_scalar(handle >> 3),
        TAG_CHAR => char::from_u32((handle >> 3) as u32).map(Value::Char),
        _ => None,
    }
}

/// Split a table handle into tag, index, and generation; `None`
/// for inline and reserved tags.
#[must_use]
pub(crate) fn split_table(handle: u64) -> Option<(u64, u32, u32)> {
    let tag = handle & TAG_MASK;
    match tag {
        TAG_VALUE | TAG_ACC | TAG_BUILDER => {
            let index = (handle >> 3) as u32;
            let generation = (handle >> GEN_SHIFT) as u32;
            Some((tag, index, generation))
        }
        _ => None,
    }
}

/// Mint a table handle from tag, index, and generation.
#[must_use]
pub(crate) fn mint_table(tag: u64, index: u32, generation: u32) -> u64 {
    tag | (u64::from(index) << 3) | (u64::from(generation & GEN_MASK) << GEN_SHIFT)
}

/// Use both reserved tags so the encoding table stays greppable
/// (they are never minted — see the module docs).
#[must_use]
pub(crate) fn is_reserved_tag(handle: u64) -> bool {
    matches!(handle & TAG_MASK, TAG_RESERVED0 | TAG_RESERVED7)
}

/// Bump a slot generation, skipping the zero value (fresh slots
/// start at zero, so zero must never recur after a free).
fn next_generation(generation: u32) -> u32 {
    let next = generation.wrapping_add(1);
    if next & GEN_MASK == 0 {
        next.wrapping_add(1)
    } else {
        next
    }
}

/// One slab slot: an optional payload (frees release it
/// promptly), a refcount, and the generation.
struct Slot<T> {
    payload: Option<T>,
    refcount: u32,
    generation: u32,
}

/// A generational slab: dense slots plus a freelist (hand-rolled
/// per the spec — ~30 lines, zero new deps).
pub(crate) struct Slab<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> Slab<T> {
    /// An empty slab.
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    /// Insert a payload (refcount 1), returning slot index and
    /// generation for [`mint_table`].
    pub(crate) fn insert(&mut self, payload: T) -> (u32, u32) {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.payload = Some(payload);
            slot.refcount = 1;
            (index, slot.generation & GEN_MASK)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                payload: Some(payload),
                refcount: 1,
                generation: 0,
            });
            (index, 0)
        }
    }

    /// Borrow a live payload (`None` for stale/OOB handles).
    pub(crate) fn get(&self, index: u32, generation: u32) -> Option<&T> {
        let slot = self.slots.get(index as usize)?;
        if slot.generation & GEN_MASK != generation {
            return None;
        }
        slot.payload.as_ref()
    }

    /// Borrow a live payload mutably (`None` for stale/OOB).
    pub(crate) fn get_mut(&mut self, index: u32, generation: u32) -> Option<&mut T> {
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation & GEN_MASK != generation {
            return None;
        }
        slot.payload.as_mut()
    }

    /// Bump a live refcount (saturating — absurd counts pin the
    /// slot rather than overflow); `false` for stale/OOB.
    pub(crate) fn clone_ref(&mut self, index: u32, generation: u32) -> bool {
        match self.slots.get_mut(index as usize) {
            Some(slot) if slot.generation & GEN_MASK == generation && slot.payload.is_some() => {
                slot.refcount = slot.refcount.saturating_add(1);
                true
            }
            _ => false,
        }
    }

    /// Drop one refcount; frees at zero (payload released,
    /// generation bumped). Returns whether the slot freed;
    /// stale/OOB/double-drops return `false`.
    pub(crate) fn drop_ref(&mut self, index: u32, generation: u32) -> bool {
        let slot = match self.slots.get_mut(index as usize) {
            Some(slot) if slot.generation & GEN_MASK == generation && slot.payload.is_some() => {
                slot
            }
            _ => return false,
        };
        slot.refcount -= 1;
        if slot.refcount == 0 {
            slot.payload = None;
            slot.generation = next_generation(slot.generation);
            self.free.push(index);
            true
        } else {
            false
        }
    }

    /// Take a live payload, freeing its slot (`None` for
    /// stale/OOB — the linear consume for accs/builders).
    pub(crate) fn take(&mut self, index: u32, generation: u32) -> Option<T> {
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation & GEN_MASK != generation {
            return None;
        }
        let payload = slot.payload.take()?;
        slot.generation = next_generation(slot.generation);
        self.free.push(index);
        Some(payload)
    }

    /// Live (unfreed) slot count — the leak-counter input.
    pub(crate) fn live(&self) -> usize {
        self.slots.len() - self.free.len()
    }
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Slab;
    use super::{
        GEN_MASK, SMALL_MAX, SMALL_MIN, TAG_ACC, TAG_BUILDER, TAG_CHAR, TAG_INT, TAG_RESERVED0,
        TAG_RESERVED7, TAG_SCALAR, TAG_VALUE, decode_inline, encode_inline, is_reserved_tag,
        mint_table, split_table,
    };
    use clj_value::Value;

    /// Encode then decode must round-trip every inline shape.
    #[test]
    fn inline_scalars_round_trip() {
        let cases = [
            Value::Nil,
            Value::Bool(false),
            Value::Bool(true),
            Value::Int(0),
            Value::Int(1),
            Value::Int(-1),
            Value::Int(42),
            Value::Int(-42),
            Value::Int(SMALL_MIN),
            Value::Int(SMALL_MAX),
            Value::Char('a'),
            Value::Char('λ'),
            Value::Char('\0'),
            Value::Char(char::MAX),
        ];
        for value in cases {
            let handle = encode_inline(&value).expect("inline shape");
            assert_ne!(handle, 0, "{value:?} must never mint the sentinel");
            assert_eq!(decode_inline(handle).as_ref(), Some(&value));
        }
    }

    /// Out-of-range ints and heap shapes are table-bound (never
    /// inline — full-range `i64` cannot fit beside the tags).
    #[test]
    fn heap_shapes_are_table_bound() {
        let cases = [
            Value::Int(SMALL_MIN - 1),
            Value::Int(SMALL_MAX + 1),
            Value::Int(i64::MIN),
            Value::Int(i64::MAX),
            Value::Double(1.5),
            Value::Double(f64::NAN),
            Value::Str("s".into()),
            Value::Vector(clj_value::Vector::from_vec(Vec::new())),
        ];
        for value in cases {
            assert_eq!(encode_inline(&value), None, "{value:?}");
        }
    }

    /// Tag layout pins: each shape mints its documented tag.
    #[test]
    fn tags_pin_to_shapes() {
        assert_eq!(encode_inline(&Value::Nil).unwrap() & 0b111, TAG_SCALAR);
        assert_eq!(
            encode_inline(&Value::Bool(true)).unwrap() & 0b111,
            TAG_SCALAR
        );
        assert_eq!(encode_inline(&Value::Int(7)).unwrap() & 0b111, TAG_INT);
        assert_eq!(encode_inline(&Value::Char('z')).unwrap() & 0b111, TAG_CHAR);
        assert_eq!(mint_table(TAG_VALUE, 1, 0) & 0b111, TAG_VALUE);
        assert_eq!(mint_table(TAG_ACC, 1, 0) & 0b111, TAG_ACC);
        assert_eq!(mint_table(TAG_BUILDER, 1, 0) & 0b111, TAG_BUILDER);
    }

    /// The sentinel and every reserved/invalid pattern decodes to
    /// nothing and splits to nothing (total, never panics).
    #[test]
    fn invalid_handles_are_total() {
        let bad = [
            0,
            0b000,
            0b1000,
            0b111,
            (3 << 3) | TAG_SCALAR,
            (0xD800_u64 << 3) | TAG_CHAR,
            (0x110000_u64 << 3) | TAG_CHAR,
            u64::MAX,
        ];
        for handle in bad {
            assert_eq!(decode_inline(handle), None, "{handle:#x}");
            if handle & 0b111 == TAG_RESERVED0 || handle & 0b111 == TAG_RESERVED7 {
                assert_eq!(split_table(handle), None, "{handle:#x}");
            }
        }
        assert!(is_reserved_tag(0));
        assert!(is_reserved_tag(0b1000));
        assert!(is_reserved_tag(0b111));
        assert!(!is_reserved_tag(mint_table(TAG_VALUE, 0, 0)));
        assert!(!is_reserved_tag(encode_inline(&Value::Int(1)).unwrap()));
    }

    /// Mint/split round-trips index and generation per table tag.
    #[test]
    fn table_handles_round_trip() {
        for tag in [TAG_VALUE, TAG_ACC, TAG_BUILDER] {
            let minted = mint_table(tag, 5, 7);
            assert_eq!(split_table(minted), Some((tag, 5, 7)));
            assert_eq!(decode_inline(minted), None);
        }
        let big = mint_table(TAG_VALUE, u32::MAX, GEN_MASK);
        assert_eq!(split_table(big), Some((TAG_VALUE, u32::MAX, GEN_MASK)));
        // Inline handles never split.
        assert_eq!(split_table(encode_inline(&Value::Nil).unwrap()), None);
        assert_eq!(split_table(encode_inline(&Value::Int(9)).unwrap()), None);
    }

    /// Refcount discipline: clone bumps, drops free at zero, and a
    /// freed slot's old handle goes stale on reuse (generation).
    #[test]
    fn slab_refcounts_and_generations() {
        let mut slab = Slab::new();
        let (index, generation) = slab.insert(Value::Int(1));
        assert_eq!((index, generation), (0, 0));
        assert_eq!(slab.live(), 1);
        assert!(slab.clone_ref(index, generation));
        assert!(!slab.drop_ref(index, generation));
        assert_eq!(slab.live(), 1);
        assert!(slab.drop_ref(index, generation));
        assert_eq!(slab.live(), 0);
        // Double-drop is a silent no-op, never a second free.
        assert!(!slab.drop_ref(index, generation));
        assert!(!slab.clone_ref(index, generation));
        assert_eq!(slab.get(index, generation), None);
        // Reuse bumps the generation: the old handle stays stale.
        let (index2, generation2) = slab.insert(Value::Int(2));
        assert_eq!(index2, index);
        assert_ne!(generation2, generation);
        assert_eq!(slab.get(index, generation), None);
        assert_eq!(slab.get(index2, generation2), Some(&Value::Int(2)));
    }

    /// OOB indices and wrong generations are silent `None`/`false`
    /// (total — stale handles degrade, never panic).
    #[test]
    fn slab_stale_and_oob_are_total() {
        let mut slab = Slab::new();
        slab.insert(Value::Int(1));
        assert_eq!(slab.get(99, 0), None);
        assert_eq!(slab.get_mut(99, 0), None);
        assert!(!slab.clone_ref(99, 0));
        assert!(!slab.drop_ref(99, 0));
        assert_eq!(slab.take(99, 0), None);
        assert_eq!(slab.get(0, 77), None);
        assert!(!slab.clone_ref(0, 77));
        assert!(!slab.drop_ref(0, 77));
        assert_eq!(slab.take(0, 77), None);
        assert_eq!(slab.live(), 1);
    }

    /// `take` consumes linearly (acc/builder discipline): payload
    /// out, slot freed, old handle stale.
    #[test]
    fn slab_take_consumes() {
        let mut accs: Slab<Vec<u64>> = Slab::new();
        let (index, generation) = accs.insert(vec![1, 2]);
        assert_eq!(accs.take(index, generation), Some(vec![1, 2]));
        assert_eq!(accs.live(), 0);
        assert_eq!(accs.take(index, generation), None);
        let mut builders: Slab<Vec<u8>> = Slab::new();
        let (index, generation) = builders.insert(vec![b'a']);
        assert_eq!(builders.take(index, generation), Some(vec![b'a']));
        assert_eq!(builders.live(), 0);
    }

    /// In-place mutation through a live handle (the `vec_push`
    /// shape); stale handles cannot reach the slot.
    #[test]
    fn slab_mutates_live_only() {
        let mut slab = Slab::new();
        let (index, generation) = slab.insert(vec![1_u64]);
        slab.get_mut(index, generation).expect("live").push(2);
        assert_eq!(slab.get(index, generation), Some(&vec![1, 2]));
        slab.take(index, generation);
        assert_eq!(slab.get_mut(index, generation), None);
    }

    /// Generation wrap skips zero (fresh slots start at zero, so
    /// zero must never recur — else a never-reused handle could
    /// validate against a wrapped slot).
    #[test]
    fn slab_generation_wrap_skips_zero() {
        let mut slab = Slab::new();
        let (index, _) = slab.insert(Value::Int(1));
        slab.slots[index as usize].generation = GEN_MASK;
        assert!(slab.drop_ref(index, GEN_MASK));
        let wrapped = slab.slots[index as usize].generation;
        assert_ne!(wrapped & GEN_MASK, 0);
        let (index2, generation2) = slab.insert(Value::Int(2));
        assert_eq!(index2, index);
        assert_eq!(generation2, wrapped & GEN_MASK);
    }
}
