// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! JVM-exact `(hash x)` ([`hasheq`]) and `(format "%h" x)` ([`hash_code`]).
//!
//! Line-by-line port of `clojure.lang.Murmur3` (`hashInt`, `hashLong`,
//! `hashUnencodedChars`, `mixCollHash`, `hashOrdered`, `hashUnordered`),
//! `Util.hasheq` / `Util.hash` / `Util.hashCombine`, `Symbol.hasheq` /
//! `.hashCode`, `Keyword.hasheq` / `.hashCode`, the Long/Double branches
//! of `Numbers.hasheq`, and the `APersistentVector` / `ASeq` /
//! `APersistentMap` / `APersistentSet` / `AMapEntry` hash methods. Every
//! oracle vector in the unit tests is pinned against JVM output (see
//! `/tmp/s5d-out7g7.txt`, `/tmp/s5d-out7g8.txt`, the `%h` pins in
//! `/tmp/s5d-out7h.txt` / `/tmp/s5d-out7h2.txt`, and the edge probes in
//! `/tmp/s5d-hasheq-edge.clj`).
//!
//! Two different hashes, like the JVM:
//!
//! - [`hasheq`] is Clojure `hasheq` (`(hash x)`): murmur-based for
//!   strings, integers, and collections; `Double.hashCode` for doubles
//!   (`-0.0` hashes as `0.0`); identity-derived for identity types.
//! - [`hash_code`] is Java `.hashCode` (`(format "%h" x)` renders it as
//!   lowercase hex): the `Long`/`Double` bit mixes, the `AbstractList`
//!   31-chain over element `.hashCode`s (recursively — never `hasheq`),
//!   sums for sets, `key ^ value` sums for maps.
//!
//! # Caller contract (clj-eval)
//!
//! - Realize before hashing: only [`RealizedSeq`](crate::RealizedSeq) is
//!   understood among sequence extensions (hashed in order, like a
//!   list). Pending/lazy seqs owned by `clj-eval` (which cannot be
//!   realized here) hash to a fixed tag, so the `hash` and `format`
//!   natives MUST realize seqs before calling either function.
//! - Never assert exact values for identity hashes: [`Value::Transient`],
//!   [`Value::Atom`], [`Value::Delay`], [`Value::Future`],
//!   [`Value::Promise`], [`Value::Agent`], [`Value::Volatile`], [`Value::Var`],
//!   [`Value::Unbound`], [`Value::Throwable`], and unrecognized
//!   extensions have NO stable JVM hash (identity hashCode). Here
//!   they hash from the per-process `id` (or a fixed tag), so values
//!   are stable only within one process. The differential corpus must
//!   not assert them — the same rule as the `#object ... 0x<id>`
//!   print hex.
//! - [`Value::Regex`] is a deliberate deterministic stand-in: the JVM
//!   hashes patterns by identity, but here a regex hashes as its source
//!   string. Do not assert it against JVM output either.
//! - Tower numerics ([`Value::BigInt`]/[`Value::BigInteger`]/
//!   [`Value::BigDec`]/[`Value::Ratio`]) hash to JVM-identical
//!   integers: `BigInt`/`BigInteger` in `long` range hash exactly
//!   like `Int` (`Murmur3.hashLong`), wider ones as
//!   `BigInteger.hashCode`; decimals hash as the
//!   trailing-zero-stripped `BigDecimal.hashCode` (zeros of any
//!   scale hash `0`); ratios hash as `numerator ^ denominator`.
//!   `=`-equal tower values always hash equally (assert both the
//!   integers and the consistency — see the `tower_*` tests).
//! - `(format "%h" nil)` rendering `"null"` is `format`-level behavior
//!   (Java renders a null `%h` argument as `"null"`); [`hash_code`] of
//!   nil is `0`, for use inside collections.

use num_traits::{ToPrimitive, Zero};

use crate::{
    CycleValue, ExtensionValue, IterateValue, Keyword, RangeNum, RangeValue, RealizedSeq,
    RepeatValue, SubVectorValue, Symbol, Value,
};

/// The `hashCombine` golden ratio, also the keyword salt (`0x9e3779b9`).
const GOLDEN: i32 = 0x9e37_79b9_u32 as i32;
/// Murmur3 `C1` (`0xcc9e2d51`).
const C1: i32 = 0xcc9e_2d51_u32 as i32;
/// Murmur3 `C2` (`0x1b873593`).
const C2: i32 = 0x1b87_3593;
/// Murmur3 `mixH1` addend (`0xe6546b64`).
const H1_ADD: i32 = 0xe654_6b64_u32 as i32;
/// Murmur3 `fmix` multipliers (`0x85ebca6b`, `0xc2b2ae35`).
const FMIX_M1: i32 = 0x85eb_ca6b_u32 as i32;
/// Murmur3 `fmix` multipliers (`0x85ebca6b`, `0xc2b2ae35`).
const FMIX_M2: i32 = 0xc2b2_ae35_u32 as i32;

/// Fixed hash for extensions this crate cannot realize (pending seqs
/// owned by `clj-eval`). Callers must realize seqs before hashing; this
/// tag only keeps the functions total. Never asserted differentially.
const UNREALIZED_EXTENSION_HASH: i32 = 0x51e9_0001_u32 as i32;

/// Clojure `hasheq` (`(hash x)`), bit-exact with the JVM for every
/// deterministic type. See the module docs for the caller contract.
#[must_use]
pub fn hasheq(v: &Value) -> i32 {
    match v {
        Value::Nil => 0,
        Value::Bool(b) => bool_hash(*b),
        Value::Int(i) => hash_long(*i),
        Value::Double(d) => double_hasheq(*d),
        Value::Str(s) => hash_int(str_hashcode(s)),
        Value::Char(c) => (*c) as i32,
        Value::Symbol(s) => symbol_hasheq(s),
        Value::Keyword(k) => keyword_hasheq(k),
        Value::List(l) => hash_ordered(l.iter().map(hasheq)),
        Value::Vector(v) => hash_ordered(v.iter().map(hasheq)),
        // Primitive vectors hash exactly like their item vector
        // (they are `=` to it — S15 oracle `f-hash-eq`).
        Value::PrimitiveVector(p) => hash_ordered(p.iter().map(hasheq)),
        Value::BigInt(_) | Value::BigInteger(_) | Value::BigDec(_) | Value::Ratio(_) => {
            hasheq_tower(v)
        }
        // S9 scalars are not `IHashEq`, so `Util.hasheq` falls
        // back to `.hashCode` (the `TaggedLiteral` rule below).
        Value::Uuid(_) | Value::Instant(_) | Value::Uri(_) => scalar_hash(v),
        other => hasheq_mid(other),
    }
}

/// S9 scalar hashing (split for the budget): the JDK `.hashCode`
/// of each kind.
fn scalar_hash(v: &Value) -> i32 {
    match v {
        Value::Uuid(u) => u.hash_code(),
        Value::Instant(i) => i.hash_code(),
        Value::Uri(u) => u.hash_code(),
        _ => unreachable!("caller passes scalars only"),
    }
}

/// `java.math.BigInteger.hashCode`: the `31`-chain over the
/// big-endian magnitude words times the sign (the `ArraysSupport`
/// chain; zero hashes `0` — its magnitude is empty).
fn bigint_hashcode(n: &num_bigint::BigInt) -> i32 {
    use num_bigint::Sign;
    let (sign, mut bytes) = n.to_bytes_be();
    if sign == Sign::NoSign {
        return 0;
    }
    // Left-pad to whole words: the first group is the most
    // significant word, like `BigInteger.mag` past leading zeros
    // (`to_bytes_be` carries no leading zero bytes, so the ragged
    // first group holds exactly the top word's value).
    while bytes.len() % 4 != 0 {
        bytes.insert(0, 0);
    }
    let mut h: i32 = 0;
    for w in bytes.chunks_exact(4) {
        let word = u32::from_be_bytes([w[0], w[1], w[2], w[3]]) as i32;
        h = h.wrapping_mul(31).wrapping_add(word);
    }
    if sign == Sign::Minus {
        h.wrapping_neg()
    } else {
        h
    }
}

/// `java.math.BigDecimal.hashCode`: `31 * unscaled.hashCode +
/// scale` (the compact fast path computes the same value — the
/// general formula covers both; `as i32` wraps exactly like the
/// JDK's `(int)` cast on astronomical parsed scales).
fn bigdec_hashcode(unscaled: &num_bigint::BigInt, scale: i64) -> i32 {
    bigint_hashcode(unscaled)
        .wrapping_mul(31)
        .wrapping_add(scale as i32)
}

/// `BigInt.hashCode` (split from [`hash_code_rest`] for the
/// complexity budget): the `Long` bit mix in range, else the
/// `BigInteger` word formula (`BigInteger.hashCode` always uses
/// the word formula, so it needs no helper).
fn bigint_code(n: &num_bigint::BigInt) -> i32 {
    match n.to_i64() {
        Some(i) => long_hash_code(i as u64),
        None => bigint_hashcode(n),
    }
}

/// Tower `hasheq`, bit-exact with the JVM (`Numbers.hasheqFrom` +
/// `BigInt.hasheq` + `Ratio.hashCode`): `BigInt`/`BigInteger` in
/// `long` range hash exactly like `Int` (`Murmur3.hashLong`, so
/// `{5N :a}` looks up by `5`); wider ones hash as
/// `BigInteger.hashCode`; decimals hash as the trailing-zero-stripped
/// `BigDecimal.hashCode` (zeros of any scale hash `0`, like
/// `BigDecimal.ZERO.hashCode`); ratios — not `IHashEq` — fall
/// through to `Ratio.hashCode` (`numerator ^ denominator`).
fn hasheq_tower(v: &Value) -> i32 {
    match v {
        Value::BigInt(n) | Value::BigInteger(n) => match n.to_i64() {
            Some(i) => hash_long(i),
            None => bigint_hashcode(n),
        },
        Value::BigDec(d) => {
            if d.is_zero() {
                0
            } else {
                let (unscaled, scale) = d.stripped();
                bigdec_hashcode(&unscaled, scale)
            }
        }
        Value::Ratio(r) => bigint_hashcode(r.numer()) ^ bigint_hashcode(r.denom()),
        _ => unreachable!("caller passes tower values only"),
    }
}

fn hasheq_mid(v: &Value) -> i32 {
    match v {
        // A map entry hashes exactly like its 2-vector (Oracle FX2
        // `hash-eq`: `(hash e)` ≡ `(hash [:a 1])`).
        Value::MapEntry { key, val, .. } => {
            hash_ordered([key.as_ref(), val.as_ref()].into_iter().map(hasheq))
        }
        Value::Map(m) => hasheq_map(m),
        Value::Set(s) => hash_unordered(s.iter().map(hasheq)),
        // Deterministic stand-in (JVM: identity) — see module docs.
        Value::Regex(source) => hash_int(str_hashcode(source)),
        // Neither JVM class is `IHashEq`, so `Util.hasheq` falls back
        // to `.hashCode()`: `(hash x)` IS `.hashCode` here.
        Value::TaggedLiteral { .. } | Value::ReaderConditional { .. } => hash_code(v),
        // Identity types: per-process `id` mix — see module docs.
        Value::Var(var) => id_hash(var.id),
        Value::Unbound(u) => id_hash(u.var_id),
        Value::Throwable(t) => id_hash(t.id),
        Value::Transient(t) => id_hash(t.id()),
        Value::Atom(a) => id_hash(a.id()),
        // Arrays hash their identity (the JVM uses
        // `identityHashCode` — S15 oracles `c-hash-*`).
        Value::Array(a) => id_hash(a.id()),
        // Port-defined types: records use the solved content-XOR
        // formula (oracle `l-hash-rec`); anything else mixes its
        // content deterministically (the JVM uses identity here).
        Value::HostType(h) => hasheq_host(h.as_ref()),
        // Multimethods hash their name symbol (identity equality,
        // so identical `Arc`s hash equally by construction; the JVM
        // uses identity here).
        Value::MultiFn(m) => hasheq_multifn(m),
        other => hasheq_rest(other),
    }
}

fn hasheq_rest(v: &Value) -> i32 {
    match v {
        // Classes hash their name's string hash (deterministic
        // stand-in — the JVM uses identity here; `=`-equal classes
        // share buckets either way; split from `hasheq_mid` for the
        // complexity budget).
        Value::Class(c) => hash_int(str_hashcode(c.name())),
        Value::Delay(d) => id_hash(d.id()),
        Value::Future(f) => id_hash(f.id()),
        Value::Promise(p) => id_hash(p.id()),
        Value::Agent(a) => id_hash(a.id()),
        Value::Volatile(v) => id_hash(v.id()),
        Value::Ref(r) => id_hash(r.id()),
        Value::Executor(x) => id_hash(x.id()),
        Value::Extension(ext) => hasheq_extension(&**ext),
        // All other variants are handled by `hasheq`/`hasheq_mid`
        // (the single caller partitions the variants).
        _ => unreachable!("hasheq partition"),
    }
}

fn hasheq_map(m: &crate::Map) -> i32 {
    hash_unordered(m.iter().map(|(k, v)| map_entry_hasheq(k, v)))
}

/// `(hash record)` (split from [`hasheq_mid`] for the complexity
/// budget): the content-map hash XOR the munged-class-name symbol
/// hash (oracle `l-hash-rec`); struct instances hash exactly like
/// their content map (S12 oracle `sm-hash-eq`); anything else mixes
/// deterministically.
fn hasheq_host(h: &crate::HostVal) -> i32 {
    match h {
        crate::HostVal::Inst(i) if i.def.kind == crate::HostKind::Record => {
            crate::record_hasheq(&i.content, &i.def.name)
        }
        crate::HostVal::Inst(i) if i.def.kind == crate::HostKind::Struct => hasheq_map(&i.content),
        other => crate::inst_hasheq(other),
    }
}

fn hasheq_extension(ext: &dyn ExtensionValue) -> i32 {
    match realized_items(ext) {
        Some(items) => hash_ordered(items.iter().map(hasheq)),
        None => UNREALIZED_EXTENSION_HASH,
    }
}

/// Java `.hashCode`, bit-exact with the JVM for every deterministic
/// type. `(format "%h" x)` renders this as lowercase hex. Collection
/// elements hash recursively with `.hashCode` semantics, never
/// `hasheq`. See the module docs for the caller contract.
#[must_use]
pub fn hash_code(v: &Value) -> i32 {
    match v {
        Value::Nil => 0,
        Value::Bool(b) => bool_hash(*b),
        Value::Int(i) => long_hash_code(*i as u64),
        Value::Double(d) => double_hash_code(*d),
        Value::Str(s) => str_hashcode(s),
        Value::Char(c) => (*c) as i32,
        Value::Symbol(s) => symbol_hash_code(s),
        Value::Keyword(k) => keyword_hash_code(k),
        Value::List(l) => list_hash_code(l.iter().map(hash_code)),
        Value::Vector(v) => list_hash_code(v.iter().map(hash_code)),
        Value::Uuid(_) | Value::Instant(_) | Value::Uri(_) => scalar_hash(v),
        Value::PrimitiveVector(p) => list_hash_code(p.iter().map(hash_code)),
        other => hash_code_mid(other),
    }
}

fn hash_code_mid(v: &Value) -> i32 {
    match v {
        // `APersistentVector.hashCode` is inherited, so `.hashCode`
        // matches the 2-vector exactly (Oracle FX2 `pr-e`/`str-e`
        // print shape plus `hash-eq`).
        Value::MapEntry { key, val, .. } => {
            list_hash_code([key.as_ref(), val.as_ref()].into_iter().map(hash_code))
        }
        Value::Map(m) => hash_code_map(m),
        Value::Set(s) => hash_code_set(s),
        // Deterministic stand-in (JVM: identity) — see module docs.
        Value::Regex(source) => str_hashcode(source),
        // `TaggedLiteral.hashCode`: `31 * hash(tag) + hash(form)`.
        Value::TaggedLiteral { tag, form, .. } => symbol_hash_code(tag)
            .wrapping_mul(31)
            .wrapping_add(hash_code(form)),
        // `ReaderConditional.hashCode`: `31 * hash(form) + hash(splicing)`.
        Value::ReaderConditional { form, splicing, .. } => hash_code(form)
            .wrapping_mul(31)
            .wrapping_add(bool_hash(*splicing)),
        // Identity types: per-process `id` mix — see module docs.
        Value::Var(var) => id_hash(var.id),
        Value::Unbound(u) => id_hash(u.var_id),
        Value::Throwable(t) => id_hash(t.id),
        Value::Transient(t) => id_hash(t.id()),
        Value::Array(a) => id_hash(a.id()),
        // Port-defined types: records hash exactly like their
        // content map (oracle `l-rec-hashcode`); anything else
        // mixes deterministically (the JVM uses identity here).
        Value::HostType(h) => hash_code_host(h.as_ref()),
        // Multimethods hash their name (deterministic stand-in for
        // the JVM's identity hash).
        Value::MultiFn(m) => hash_code_multifn(m),
        other => hash_code_rest(other),
    }
}

/// `(hash multifn)` (split from [`hasheq_mid`] for the complexity
/// budget): the name symbol's hash.
fn hasheq_multifn(m: &crate::MultiFn) -> i32 {
    hasheq(&Value::Symbol(Symbol::new(None, &m.name)))
}

/// `(.hashCode multifn)` (split from [`hash_code_mid`] for the
/// complexity budget): the name's Java hash.
fn hash_code_multifn(m: &crate::MultiFn) -> i32 {
    str_hashcode(&m.name)
}

/// `(.hashCode record)` (split from [`hash_code_mid`] for the
/// complexity budget): exactly the content map's hash (oracle
/// `l-rec-hashcode`); struct instances likewise (S12 — map-identical
/// in both hashes); anything else mixes deterministically.
fn hash_code_host(h: &crate::HostVal) -> i32 {
    match h {
        crate::HostVal::Inst(i) if i.def.kind == crate::HostKind::Record => {
            hash_code_map(&i.content)
        }
        crate::HostVal::Inst(i) if i.def.kind == crate::HostKind::Struct => {
            hash_code_map(&i.content)
        }
        crate::HostVal::Inst(i) => hash_combine(
            str_hashcode(&i.def.name),
            hash_ordered(i.content.iter().map(|(_, v)| hash_code(v))),
        ),
        crate::HostVal::Def(d) => str_hashcode(&d.name),
    }
}

fn hash_code_rest(v: &Value) -> i32 {
    match v {
        // Classes hash their name's `String.hashCode` (deterministic
        // stand-in — the JVM uses identity here; split from
        // `hash_code_mid` for the complexity budget).
        Value::Class(c) => str_hashcode(c.name()),
        Value::Atom(a) => id_hash(a.id()),
        Value::Delay(d) => id_hash(d.id()),
        Value::Future(f) => id_hash(f.id()),
        Value::Promise(p) => id_hash(p.id()),
        Value::Agent(a) => id_hash(a.id()),
        Value::Volatile(v) => id_hash(v.id()),
        Value::Ref(r) => id_hash(r.id()),
        Value::Executor(x) => id_hash(x.id()),
        // Numerics and extensions live in `hash_code_rest_num`
        // (split for the complexity budget).
        other => hash_code_rest_num(other),
    }
}

/// The numeric and extension arms of [`hash_code_rest`] (split for
/// the complexity budget).
fn hash_code_rest_num(v: &Value) -> i32 {
    match v {
        // JVM `.hashCode`s: in-range `BigInt` uses the `Long`
        // bit mix (`BigInt.hashCode`), `BigInteger` always uses
        // `BigInteger.hashCode`, decimals use the UNstripped
        // `BigDecimal.hashCode` (scale counts), ratios `num ^ den`.
        Value::BigInt(n) => bigint_code(n),
        Value::BigInteger(n) => bigint_hashcode(n),
        Value::BigDec(d) => bigdec_hashcode(d.unscaled(), d.scale()),
        Value::Ratio(r) => bigint_hashcode(r.numer()) ^ bigint_hashcode(r.denom()),
        Value::Extension(ext) => hash_code_extension(&**ext),
        // All other variants are handled upstream (the single
        // caller partitions the variants).
        _ => unreachable!("hash_code partition"),
    }
}

fn hash_code_map(m: &crate::Map) -> i32 {
    let mut sum: i32 = 0;
    for (k, val) in m.iter() {
        sum = sum.wrapping_add(hash_code(k) ^ hash_code(val));
    }
    sum
}

fn hash_code_set(s: &crate::Set) -> i32 {
    let mut sum: i32 = 0;
    for x in s.iter() {
        sum = sum.wrapping_add(hash_code(x));
    }
    sum
}

fn hash_code_extension(ext: &dyn ExtensionValue) -> i32 {
    match realized_items(ext) {
        Some(items) => list_hash_code(items.iter().map(hash_code)),
        None => UNREALIZED_EXTENSION_HASH,
    }
}

/// Java `String.hashCode` over UTF-16 code units (`31 * h + unit`):
/// astral-plane characters hash as their surrogate pair, exactly like
/// the JVM (`"😀"` is `(0xD83D, 0xDE00)`).
#[must_use]
pub fn str_hashcode(s: &str) -> i32 {
    let mut h: i32 = 0;
    for unit in s.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(i32::from(unit));
    }
    h
}

/// `Boolean.hashCode`: `1231` / `1237` (also the `hasheq`, since
/// `Boolean` is not `IHashEq`).
fn bool_hash(b: bool) -> i32 {
    if b { 1231 } else { 1237 }
}

/// `Util.hashCombine`: `seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2)`
/// (all wrapping; `>>` arithmetic, like Java). Public for the S12
/// `hash-combine` native, which composes it over `hash_code(y)`
/// (never murmur — oracle `mh-util-spot`).
#[must_use]
pub fn hash_combine(seed: i32, hash: i32) -> i32 {
    seed ^ hash
        .wrapping_add(GOLDEN)
        .wrapping_add(seed << 6)
        .wrapping_add(seed >> 2)
}

/// `Util.hash` of the namespace part: `0` when missing (Java hashes
/// `null` as `0`, and `""` also hashes to `0`, so empty and missing
/// namespaces agree without special-casing).
fn ns_hash(ns: Option<&str>) -> i32 {
    ns.map_or(0, str_hashcode)
}

/// `Symbol.hasheq`: `hashCombine(murmur3(name), hash(ns))`.
fn symbol_hasheq(s: &Symbol) -> i32 {
    hash_combine(str_murmur(s.name()), ns_hash(s.namespace()))
}

/// `Symbol.hashCode`: `hashCombine(name.hashCode, hash(ns))`.
fn symbol_hash_code(s: &Symbol) -> i32 {
    hash_combine(str_hashcode(s.name()), ns_hash(s.namespace()))
}

/// `Keyword.hasheq`: `symbolHasheq + 0x9e3779b9` (wrapping).
fn keyword_hasheq(k: &Keyword) -> i32 {
    hash_combine(str_murmur(k.name()), ns_hash(k.namespace())).wrapping_add(GOLDEN)
}

/// `Keyword.hashCode`: `symbolHashCode + 0x9e3779b9` (wrapping).
fn keyword_hash_code(k: &Keyword) -> i32 {
    hash_combine(str_hashcode(k.name()), ns_hash(k.namespace())).wrapping_add(GOLDEN)
}

/// A map entry hashes as the ordered pair `[k v]` (`AMapEntry extends
/// `APersistentVector` on the JVM, so entries hash as 2-vectors).
fn map_entry_hasheq(k: &Value, v: &Value) -> i32 {
    hash_ordered([hasheq(k), hasheq(v)].into_iter())
}

/// Java `Long.hashCode`: `(int)(v ^ (v >>> 32))`.
fn long_hash_code(bits: u64) -> i32 {
    (bits ^ (bits >> 32)) as i32
}

/// Java `Double.hashCode`: `Long.hashCode(doubleToLongBits(d))`, with
/// NaN canonicalized exactly like `Double.doubleToLongBits`.
fn double_hash_code(d: f64) -> i32 {
    let bits = if d.is_nan() {
        0x7ff8_0000_0000_0000u64
    } else {
        d.to_bits()
    };
    long_hash_code(bits)
}

/// `Numbers.hasheq`, Double branch: exactly `-0.0` (bitwise, matching
/// `Double.equals`) hashes as `0`; everything else is `Double.hashCode`
/// (`+0.0` already mixes to `0`).
fn double_hasheq(d: f64) -> i32 {
    if d.to_bits() == 0x8000_0000_0000_0000u64 {
        0
    } else {
        double_hash_code(d)
    }
}

/// Identity-typed values (`Var`, `Unbound`, `Throwable`, `Transient`,
/// `Atom`, `Delay`, `Future`, `Promise`, `Agent`, `Volatile`): the JVM
/// uses identity hashCode; here the per-process `id` mixed as a long.
/// Stable within one process only — see the module docs.
fn id_hash(id: u64) -> i32 {
    long_hash_code(id)
}

/// Whether hashing must throw bare `UnsupportedOperationException`
/// (JVM `Repeat`/`Cycle`/`Iterate.hasheq`/`hashCode`): an unbounded
/// `Repeat`/`Range`, a zero-step `Range`, any `Cycle`/`Iterate` —
/// top-level or nested in a realized collection (elements hash
/// eagerly on the JVM, so a nested infinite throws too). Non-forcing:
/// pending `clj-eval` thunks are skipped (their infiniteness is
/// unknowable without forcing — the JVM hangs forcing them, and so
/// does the realize step). Callers (`hash`, `case*`, `format %h`)
/// check this BEFORE realizing.
#[must_use]
pub fn seq_hash_unsupported(v: &Value) -> bool {
    match v {
        Value::List(l) => l.iter().any(seq_hash_unsupported),
        Value::Vector(v) => v.iter().any(seq_hash_unsupported),
        Value::MapEntry { key, val, .. } => seq_hash_unsupported(key) || seq_hash_unsupported(val),
        Value::Map(m) => m
            .iter()
            .any(|(k, v)| seq_hash_unsupported(k) || seq_hash_unsupported(v)),
        Value::Set(s) => s.iter().any(seq_hash_unsupported),
        Value::Extension(ext) => extension_hash_unsupported(&**ext),
        _ => false,
    }
}

/// Whether hashing an extension value throws (unbounded `repeat` /
/// `cycle` / `iterate` / `range` steps, or a realized seq with a
/// nested infinite).
fn extension_hash_unsupported(ext: &dyn ExtensionValue) -> bool {
    let any = ext.as_any();
    if let Some(r) = any.downcast_ref::<RepeatValue>() {
        return r.n().is_none();
    }
    if any.downcast_ref::<CycleValue>().is_some() || any.downcast_ref::<IterateValue>().is_some() {
        return true;
    }
    if let Some(r) = any.downcast_ref::<RangeValue>() {
        return range_hash_unsupported(r);
    }
    // Realized seqs: a nested infinite still throws.
    realized_items(ext).is_some_and(|items| items.iter().any(seq_hash_unsupported))
}

/// Whether hashing a range throws (open-ended, or a zero step that
/// never reaches its end). A `BigDec`/ratio step with no exact
/// decimal is *not* unsupported: callers realize first, and the
/// realize step raises the JVM `ArithmeticException` (oracle
/// `range-mixed`).
fn range_hash_unsupported(r: &RangeValue) -> bool {
    if r.end().is_none() {
        return true;
    }
    match r.step() {
        RangeNum::Int(n) => n == 0,
        RangeNum::Double(x) => x == 0.0,
        RangeNum::BigInt(n) => n.is_zero(),
        RangeNum::Ratio(r) => r.is_zero(),
        RangeNum::BigDec(d) => d.is_zero(),
    }
}

/// Items of a realized-seq extension (`None` for anything else —
/// pending seqs live in `clj-eval` and must be realized by the caller
/// before hashing).
fn realized_items(ext: &dyn ExtensionValue) -> Option<Vec<Value>> {
    if let Some(sv) = ext.as_any().downcast_ref::<SubVectorValue>() {
        return Some(sv.items());
    }
    ext.as_any()
        .downcast_ref::<RealizedSeq>()
        .map(RealizedSeq::items)
}

/// `Murmur3.hashOrdered`: `31`-chain from `1` over element hasheqs,
/// then `mixCollHash(hash, count)`.
pub fn hash_ordered<I>(hashes: I) -> i32
where
    I: Iterator<Item = i32>,
{
    let mut h: i32 = 1;
    let mut n: i32 = 0;
    for x in hashes {
        h = h.wrapping_mul(31).wrapping_add(x);
        n = n.wrapping_add(1);
    }
    mix_coll_hash(h, n)
}

/// `Murmur3.hashUnordered`: sum of element hasheqs, then
/// `mixCollHash(sum, count)`.
pub fn hash_unordered<I>(hashes: I) -> i32
where
    I: Iterator<Item = i32>,
{
    let mut sum: i32 = 0;
    let mut n: i32 = 0;
    for x in hashes {
        sum = sum.wrapping_add(x);
        n = n.wrapping_add(1);
    }
    mix_coll_hash(sum, n)
}

/// `AbstractList.hashCode`: the `31`-chain from `1` WITHOUT the
/// collection mix.
fn list_hash_code<I>(codes: I) -> i32
where
    I: Iterator<Item = i32>,
{
    let mut h: i32 = 1;
    for x in codes {
        h = h.wrapping_mul(31).wrapping_add(x);
    }
    h
}

fn mix_k1(mut k: i32) -> i32 {
    k = k.wrapping_mul(C1);
    k = k.rotate_left(15);
    k.wrapping_mul(C2)
}

fn mix_h1(h: i32, k: i32) -> i32 {
    let h = (h ^ k).rotate_left(13);
    h.wrapping_mul(5).wrapping_add(H1_ADD)
}

fn fmix(h: i32, len: i32) -> i32 {
    let mut h = (h ^ len) as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(FMIX_M1 as u32);
    h ^= h >> 13;
    h = h.wrapping_mul(FMIX_M2 as u32);
    h ^= h >> 16;
    h as i32
}

/// `Murmur3.hashInt` (with the `0 -> 0` fast path).
fn hash_int(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    fmix(mix_h1(0, mix_k1(v)), 4)
}

/// `Murmur3.hashLong` (with the `0 -> 0` fast path); also the
/// `Numbers.hasheq` Long branch.
fn hash_long(v: i64) -> i32 {
    if v == 0 {
        return 0;
    }
    let low = v as i32;
    let high = ((v as u64) >> 32) as i32;
    let h = mix_h1(0, mix_k1(low));
    let h = mix_h1(h, mix_k1(high));
    fmix(h, 8)
}

/// `Murmur3.mixCollHash`. Public for the S12
/// `mix-collection-hash` native (int-checked longs in, like the
/// JVM wrapper).
#[must_use]
pub fn mix_coll_hash(hash: i32, count: i32) -> i32 {
    fmix(mix_h1(0, mix_k1(hash)), count)
}

/// `Murmur3.hashUnencodedChars` over pre-encoded UTF-16 units (pairs
/// packed little-endian, a trailing lone unit xored in, finalized with
/// twice the unit count).
fn hash_unencoded_chars(units: &[u16]) -> i32 {
    let mut h: i32 = 0;
    // Step through two units at a time, like the JVM loop (`as_chunks`
    // would trip the MSRV 1.85 floor, so this stays a plain loop).
    let mut i = 1;
    while i < units.len() {
        let k = u32::from(units[i - 1]) | (u32::from(units[i]) << 16);
        h = mix_h1(h, mix_k1(k as i32));
        i += 2;
    }
    if units.len() % 2 == 1 {
        h ^= mix_k1(i32::from(units[units.len() - 1]));
    }
    fmix(h, (units.len() as i32).wrapping_mul(2))
}

/// `Murmur3.hashUnencodedChars` over a Rust string's UTF-16 encoding.
fn str_murmur(s: &str) -> i32 {
    hash_unencoded_chars(&s.encode_utf16().collect::<Vec<u16>>())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        List, Map, Seq, Set, ThrowClass, Throwable, Transient, TransientKind, TransientVector,
        Unbound, Var, Vector,
    };

    fn int(v: i64) -> Value {
        Value::Int(v)
    }

    fn dbl(v: f64) -> Value {
        Value::Double(v)
    }

    fn strv(s: &str) -> Value {
        Value::Str(Arc::from(s))
    }

    fn kw(ns: Option<&str>, name: &str) -> Value {
        Value::Keyword(Keyword::new(ns, name))
    }

    fn sym(ns: Option<&str>, name: &str) -> Value {
        Value::Symbol(Symbol::new(ns, name))
    }

    fn vecv(items: Vec<Value>) -> Value {
        Value::Vector(Vector::from_vec(items))
    }

    fn listv(items: Vec<Value>) -> Value {
        Value::List(List::from_vec(items))
    }

    fn mapv(pairs: Vec<(Value, Value)>) -> Value {
        Value::Map(Map::from_kvs_checked(pairs).unwrap())
    }

    fn setv(items: Vec<Value>) -> Value {
        Value::Set(Set::from_items_checked(items).unwrap())
    }

    fn seqv(items: Vec<Value>) -> Value {
        Value::Extension(Arc::new(RealizedSeq::new(Seq::eager(items).unwrap())))
    }

    /// `(format "%h" x)` rendering of a hash: lowercase hex of the bits,
    /// exactly like `Integer.toHexString`.
    fn hex_of(code: i32) -> String {
        format!("{:x}", code as u32)
    }

    #[test]
    fn oracle_fx2_entry_hash() {
        // `docs/t1192/probes/fx2-mapentry3.out` (`entry-hash`,
        // `vec-hash`): an entry hashes exactly like its 2-vector.
        let k = kw(None, "a");
        assert_eq!(hasheq(&crate::map_entry(k.clone(), int(1))), -1_809_393_901);
        assert_eq!(
            hasheq(&crate::map_entry(k.clone(), int(1))),
            hasheq(&vecv(vec![k, int(1)]))
        );
    }

    #[test]
    fn oracle_fx2_entry_percent_h() {
        // `docs/t1192/probes/fx2-mapentry16.out` (`pcth-eq`):
        // `.hashCode` matches the 2-vector exactly.
        let k = kw(None, "a");
        assert_eq!(
            hash_code(&crate::map_entry(k.clone(), int(1))),
            hash_code(&vecv(vec![k, int(1)]))
        );
    }

    #[test]
    fn oracle_7g7_hash() {
        // `/tmp/s5d-out7g7.txt`, re-run from `/tmp/s5d-gaps7g7.clj`.
        assert_eq!(hasheq(&strv("ab")), 1_707_819_681);
        assert_eq!(hasheq(&vecv(vec![strv("ab"), int(1)])), 484_367_275);
        assert_eq!(hasheq(&kw(None, "a")), -2_123_407_586);
        assert_eq!(hasheq(&sym(None, "a")), -482_876_059);
        assert_eq!(hasheq(&vecv(vec![int(1), int(2)])), 156_247_261);
        assert_eq!(hasheq(&mapv(vec![(kw(None, "a"), int(1))])), 1_772_842_048);
        assert_eq!(hasheq(&int(1)), 1_392_991_556);
        assert_eq!(hasheq(&dbl(1.0)), 1_072_693_248);
        assert_eq!(hasheq(&Value::Nil), 0);
        assert_eq!(hasheq(&dbl(1.5)), 1_073_217_536);
        assert_eq!(hasheq(&dbl(-0.0)), 0);
        assert_eq!(hasheq(&Value::Bool(true)), 1231);
        assert_eq!(hasheq(&Value::Char('A')), 65);
        assert_eq!(hasheq(&setv(vec![int(1)])), 1_038_464_948);
        assert_eq!(hasheq(&listv(vec![int(1), int(2)])), 156_247_261);
        assert_eq!(
            hasheq(&mapv(vec![(kw(None, "a"), vecv(vec![int(1), strv("x")]))])),
            -1_503_654_198
        );
    }

    #[test]
    fn oracle_7g8_hash() {
        // `/tmp/s5d-out7g8.txt`, re-run from `/tmp/s5d-gaps7g8.clj`.
        assert_eq!(hasheq(&vecv(vec![])), -2_017_569_654);
        assert_eq!(hasheq(&mapv(vec![])), -15_128_758);
        assert_eq!(hasheq(&setv(vec![])), -15_128_758);
        assert_eq!(hasheq(&listv(vec![])), -2_017_569_654);
        assert_eq!(hasheq(&vecv(vec![int(1)])), -1_381_383_523);
        assert_eq!(hasheq(&strv("a")), 1_455_541_201);
        assert_eq!(hasheq(&int(0)), 0);
        assert_eq!(hasheq(&int(-1)), 1_651_860_712);
        assert_eq!(hasheq(&dbl(0.0)), 0);
        assert_eq!(hasheq(&dbl(2.5)), 1_074_003_968);
        assert_eq!(hasheq(&kw(Some("a"), "b")), 1_482_224_565);
        assert_eq!(hasheq(&sym(Some("a"), "b")), -1_172_211_204);
        assert_eq!(hasheq(&vecv(vec![kw(None, "a"), int(1)])), -1_809_393_901);
    }

    #[test]
    fn oracle_7g8_percent_h() {
        // `/tmp/s5d-out7g8.txt`: `(format "%h" x)` strings.
        let cases: Vec<(Value, &str)> = vec![
            (vecv(vec![strv("ab"), int(1)]), "17bc1"),
            (mapv(vec![(kw(None, "a"), vecv(vec![int(1)]))]), "3c6f0c09"),
            (sym(Some("a"), "b"), "9e3792d0"),
            (kw(Some("a"), "b"), "3c6f0c89"),
            (listv(vec![int(1), int(2)]), "3e2"),
            (setv(vec![int(1), int(2)]), "3"),
            (vecv(vec![dbl(1.5)]), "3ff8001f"),
            (dbl(-0.0), "80000000"),
            (strv("a"), "61"),
            (
                mapv(vec![(kw(None, "a"), int(1)), (kw(None, "b"), int(2))]),
                "78de1816",
            ),
            (vecv(vec![]), "1"),
        ];
        for (v, want) in &cases {
            assert_eq!(hex_of(hash_code(v)), *want);
        }
        assert_eq!(hash_code(&sym(Some("a"), "b")), 0x9e37_92d0_u32 as i32);
        assert_eq!(hash_code(&kw(Some("a"), "b")), 0x3c6f_0c89);
        assert_eq!(hash_code(&vecv(vec![dbl(1.5)])), 0x3ff8_001f);
        assert_eq!(hash_code(&dbl(-0.0)), i32::MIN);
        assert_eq!(
            hash_code(&mapv(vec![
                (kw(None, "a"), int(1)),
                (kw(None, "b"), int(2))
            ])),
            0x78de_1816
        );
    }

    #[test]
    fn oracle_7h_7h2_percent_h() {
        // `%h` pins from `/tmp/s5d-out7h.txt` (`:c-h-*`) and
        // `/tmp/s5d-out7h2.txt` (`:h-*`). (`:c-h-nil` renders `"null"`
        // at the `format` level, not here; BigInt pins have no `Value`.)
        let cases: Vec<(Value, &str)> = vec![
            (int(5), "5"),
            (strv("ab"), "c21"),
            (int(-1), "0"),
            (Value::Bool(true), "4cf"),
            (Value::Bool(false), "4d5"),
            (Value::Char('A'), "41"),
            (dbl(1.5), "3ff80000"),
            (kw(None, "a"), "3c6f0c29"),
            (sym(None, "a"), "9e379270"),
            (vecv(vec![int(1), int(2)]), "3e2"),
            (vecv(vec![]), "1"),
            (mapv(vec![(kw(None, "a"), int(1))]), "3c6f0c28"),
            (mapv(vec![]), "0"),
            (setv(vec![int(1)]), "1"),
            (listv(vec![int(1), int(2)]), "3e2"),
            (strv(""), "0"),
            (strv("a\nb"), "16db9"),
            (strv("é"), "e9"),
            (int(0), "0"),
            (dbl(0.0), "0"),
            (dbl(f64::NAN), "7ff80000"),
            (dbl(f64::INFINITY), "7ff00000"),
        ];
        for (v, want) in &cases {
            assert_eq!(hex_of(hash_code(v)), *want);
        }
        assert_eq!(hash_code(&kw(None, "a")), 0x3c6f_0c29);
        assert_eq!(hash_code(&sym(None, "a")), 0x9e37_9270_u32 as i32);
        assert_eq!(hash_code(&mapv(vec![(kw(None, "a"), int(1))])), 0x3c6f_0c28);
        assert_eq!(hash_code(&strv("a\nb")), 0x16db9);
        assert_eq!(hash_code(&dbl(f64::NAN)), 0x7ff8_0000);
    }

    #[test]
    fn edge_nan_and_infinities() {
        // Extra oracle pins (`/tmp/s5d-hasheq-edge.clj`): doubles hash
        // by `Double.hashCode` in both functions.
        assert_eq!(hasheq(&dbl(f64::NAN)), 2_146_959_360);
        assert_eq!(hasheq(&dbl(f64::INFINITY)), 2_146_435_072);
        assert_eq!(hasheq(&dbl(f64::NEG_INFINITY)), -1_048_576);
        assert_eq!(hash_code(&dbl(f64::NAN)), 0x7ff8_0000);
        assert_eq!(hash_code(&dbl(f64::INFINITY)), 0x7ff0_0000);
        assert_eq!(hash_code(&dbl(f64::NEG_INFINITY)), 0xfff0_0000_u32 as i32);
        assert_eq!(hasheq(&vecv(vec![dbl(f64::NAN)])), -298_223_043);
    }

    #[test]
    fn edge_neg_zero_vs_zero() {
        // `hasheq` equates them (`Numbers.hasheq` maps `-0.0` to `0`,
        // and `+0.0` already mixes to `0`); `.hashCode` distinguishes
        // them (`%h`: `"0"` vs `"80000000"`).
        assert_eq!(hasheq(&dbl(0.0)), 0);
        assert_eq!(hasheq(&dbl(-0.0)), 0);
        assert_eq!(hash_code(&dbl(0.0)), 0);
        assert_eq!(hash_code(&dbl(-0.0)), i32::MIN);
        assert_eq!(hex_of(hash_code(&dbl(-0.0))), "80000000");
        assert_eq!(hasheq(&vecv(vec![dbl(-0.0)])), 965_192_274);
        assert_eq!(hasheq(&vecv(vec![dbl(0.0)])), 965_192_274);
    }

    #[test]
    fn edge_astral_plane_utf16() {
        // `"😀"` is the surrogate pair `(0xD83D, 0xDE00)`, hashed as two
        // units exactly like the JVM (oracle: `/tmp/s5d-hasheq-edge.clj`).
        assert_eq!(str_hashcode("😀"), 0x1b_0d63);
        assert_eq!(str_hashcode("😀"), 31 * 0xd83d + 0xde00);
        assert_eq!(hex_of(hash_code(&strv("😀"))), "1b0d63");
        assert_eq!(hasheq(&strv("😀")), -1_934_073_014);
        assert_eq!(hasheq(&strv("é")), -1_107_437_598);
        assert_eq!(hasheq(&strv("hello world")), -1_522_757_468);
        assert_eq!(hex_of(hash_code(&strv("hello world"))), "6aefe2c4");
    }

    #[test]
    fn edge_empty_collections() {
        // Ordered empties share one `hasheq`, unordered empties another;
        // `%h` keeps the raw `AbstractList`/`AbstractMap`/`AbstractSet` values.
        assert_eq!(hasheq(&vecv(vec![])), hasheq(&listv(vec![])));
        assert_eq!(hasheq(&mapv(vec![])), hasheq(&setv(vec![])));
        assert_eq!(hash_code(&vecv(vec![])), 1);
        assert_eq!(hash_code(&listv(vec![])), 1);
        assert_eq!(hash_code(&mapv(vec![])), 0);
        assert_eq!(hash_code(&setv(vec![])), 0);
        // An empty realized seq hashes as the empty list.
        let empty_seq = Value::Extension(Arc::new(RealizedSeq::new(Seq::empty())));
        assert_eq!(hasheq(&empty_seq), hasheq(&listv(vec![])));
        assert_eq!(hash_code(&empty_seq), 1);
    }

    #[test]
    fn sequential_kinds_agree() {
        // Vectors, lists, and realized seqs share both hashes.
        let items = vec![int(1), strv("x"), kw(None, "a")];
        let v = vecv(items.clone());
        let l = listv(items.clone());
        let s = seqv(items);
        assert_eq!(hasheq(&v), hasheq(&l));
        assert_eq!(hasheq(&v), hasheq(&s));
        assert_eq!(hash_code(&v), hash_code(&l));
        assert_eq!(hash_code(&v), hash_code(&s));
    }

    #[test]
    fn unordered_hashes_ignore_order() {
        let ab = mapv(vec![(kw(None, "a"), int(1)), (kw(None, "b"), int(2))]);
        let ba = mapv(vec![(kw(None, "b"), int(2)), (kw(None, "a"), int(1))]);
        assert_eq!(hasheq(&ab), hasheq(&ba));
        assert_eq!(hash_code(&ab), hash_code(&ba));
        let s1 = setv(vec![int(1), int(2), strv("x")]);
        let s2 = setv(vec![strv("x"), int(2), int(1)]);
        assert_eq!(hasheq(&s1), hasheq(&s2));
        assert_eq!(hash_code(&s1), hash_code(&s2));
    }

    #[test]
    fn empty_ns_matches_missing_ns() {
        // Java hashes `""` and `null` alike (`0`), so `Some("")` ≡ `None`.
        assert_eq!(hasheq(&sym(Some(""), "a")), hasheq(&sym(None, "a")));
        assert_eq!(hash_code(&sym(Some(""), "a")), hash_code(&sym(None, "a")));
        assert_eq!(hasheq(&kw(Some(""), "a")), hasheq(&kw(None, "a")));
        assert_eq!(hash_code(&kw(Some(""), "a")), hash_code(&kw(None, "a")));
    }

    #[test]
    fn multichar_names_hash_murmur_pairs() {
        // Multi-unit names run the pair loop in `hash_unencoded_chars`
        // (oracled `(hash :ab)`, `(hash 'ab)`, `(hash :abc)`).
        assert_eq!(hasheq(&kw(None, "ab")), -839573926);
        assert_eq!(hasheq(&sym(None, "ab")), 800957601);
        assert_eq!(hasheq(&kw(None, "abc")), -1232035677);
    }

    #[test]
    fn map_entry_hashes_as_ordered_pair() {
        // `AMapEntry extends APersistentVector`: `{:a 1}` mixes the
        // single entry vector `[k v]` through the unordered path.
        let entry = hash_ordered([hasheq(&kw(None, "a")), hasheq(&int(1))].into_iter());
        let m = mapv(vec![(kw(None, "a"), int(1))]);
        assert_eq!(hasheq(&m), hash_unordered([entry].into_iter()));
    }

    #[test]
    fn tagged_and_conditional_match_jvm_hashcode() {
        // Neither JVM class is `IHashEq`: `(hash x)` IS `.hashCode`.
        let tagged = crate::tagged_literal(Symbol::new(None, "tag"), vecv(vec![int(1)]));
        let want = symbol_hash_code(&Symbol::new(None, "tag"))
            .wrapping_mul(31)
            .wrapping_add(hash_code(&vecv(vec![int(1)])));
        assert_eq!(hash_code(&tagged), want);
        assert_eq!(hasheq(&tagged), want);

        for splicing in [false, true] {
            let rc = crate::reader_conditional(int(7), splicing);
            let want = hash_code(&int(7))
                .wrapping_mul(31)
                .wrapping_add(if splicing { 1231 } else { 1237 });
            assert_eq!(hash_code(&rc), want);
            assert_eq!(hasheq(&rc), want);
        }
    }

    #[test]
    fn regex_hashes_as_its_source() {
        // Deterministic stand-in (JVM: identity) — see module docs.
        let rx = Value::Regex(Arc::from("a+"));
        assert_eq!(hasheq(&rx), hasheq(&strv("a+")));
        assert_eq!(hash_code(&rx), hash_code(&strv("a+")));
    }

    #[test]
    fn identity_types_derive_from_id() {
        // Per-process-stable by contract — the corpus must not pin these.
        let var = Value::Var(Var {
            id: 0x1234_5678_9abc_def0,
            sym: Symbol::new(Some("user"), "x"),
        });
        // `(0x9ABCDEF0 ^ 0x12345678) == 0x88888888`.
        assert_eq!(hasheq(&var), 0x8888_8888_u32 as i32);
        assert_eq!(hash_code(&var), 0x8888_8888_u32 as i32);

        let unbound = Value::Unbound(Unbound {
            sym: Symbol::new(None, "y"),
            var_id: 7,
        });
        assert_eq!(hasheq(&unbound), 7);
        assert_eq!(hash_code(&unbound), 7);

        let throwable = Value::Throwable(Throwable {
            id: 42,
            class: ThrowClass::Exception,
            message: None,
            data: None,
            cause: None,
        });
        assert_eq!(hasheq(&throwable), 42);
        assert_eq!(hash_code(&throwable), 42);

        // Bind before wrapping: no destructure, no dead fallback
        // (100% line gate counts test lines too).
        let t = Transient::new(TransientKind::Vector(TransientVector::new()));
        let id = t.id();
        let transient_ = Value::Transient(t);
        let want = (id ^ (id >> 32)) as i32;
        assert_eq!(hasheq(&transient_), want);
        assert_eq!(hash_code(&transient_), want);
        assert_eq!(hasheq(&transient_.clone()), want);
    }

    #[test]
    fn unrealized_extension_hashes_to_fixed_tag() {
        #[derive(Debug)]
        struct Opaque;
        impl crate::ExtensionValue for Opaque {
            fn class_name(&self) -> String {
                "Opaque".to_owned()
            }
            fn print(&self) -> String {
                "Opaque".to_owned()
            }
            fn dyn_eq(&self, _other: &dyn crate::ExtensionValue) -> bool {
                false
            }
            fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
                state.write_u8(0);
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let v = Value::Extension(Arc::new(Opaque));
        assert_eq!(hasheq(&v), UNREALIZED_EXTENSION_HASH);
        assert_eq!(hash_code(&v), UNREALIZED_EXTENSION_HASH);
        // The stub's own trait methods (hashing only reaches `as_any`).
        use crate::ExtensionValue;
        let o = Opaque;
        assert_eq!(o.class_name(), "Opaque");
        assert_eq!(o.print(), "Opaque");
        assert!(!o.dyn_eq(&o));
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        o.dyn_hash(&mut h1);
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        o.dyn_hash(&mut h2);
        assert_eq!(
            std::hash::Hasher::finish(&h1),
            std::hash::Hasher::finish(&h2)
        );
    }

    #[test]
    fn str_hashcode_matches_java() {
        assert_eq!(str_hashcode(""), 0);
        assert_eq!(str_hashcode("a"), 97);
        assert_eq!(str_hashcode("ab"), 3105);
        assert_eq!(str_hashcode("a\nb"), 0x16db9);
        assert_eq!(str_hashcode("é"), 0xe9);
        assert_eq!(str_hashcode("😀"), 0x1b_0d63);
    }

    #[test]
    fn hash_code_multifn_uses_name() {
        // `(.hashCode multifn)` is the name's Java hash
        // (unreachable from Clojure — `.hashCode` has no
        // non-host fallback and `hash-code` is private — so
        // pinned directly; `'m'` hashes to 109).
        let mf = Value::MultiFn(std::sync::Arc::new(crate::MultiFn {
            name: std::sync::Arc::from("m"),
            dispatch: Value::Nil,
            default: Value::Nil,
            methods: crate::new_map(),
            prefers: crate::new_map(),
            hierarchy: None,
        }));
        assert_eq!(hash_code(&mf), 109);
    }

    #[test]
    fn future_and_promise_hash_from_id() {
        use crate::{Future, Promise};
        let f = Future::new();
        let want = (f.id() ^ (f.id() >> 32)) as i32;
        let v = Value::Future(f);
        assert_eq!(hasheq(&v), want);
        assert_eq!(hash_code(&v), want);
        let p = Promise::new();
        let want = (p.id() ^ (p.id() >> 32)) as i32;
        let v = Value::Promise(p);
        assert_eq!(hasheq(&v), want);
        assert_eq!(hash_code(&v), want);
    }

    #[test]
    fn executor_hash_from_id() {
        use crate::Executor;
        let x = Executor::new().unwrap();
        let want = (x.id() ^ (x.id() >> 32)) as i32;
        let v = Value::Executor(x);
        assert_eq!(hasheq(&v), want);
        // `.hashCode` is unreachable from Clojure (`hash-code` is
        // private), so the `hash_code_rest` arm is pinned directly.
        assert_eq!(hash_code(&v), want);
    }

    #[test]
    fn atom_delay_agent_hash_from_id() {
        use crate::{Agent, Atom, Delay};
        let a = Atom::new(int(0));
        let want = (a.id() ^ (a.id() >> 32)) as i32;
        let v = Value::Atom(a);
        assert_eq!(hasheq(&v), want);
        assert_eq!(hash_code(&v), want);
        let d = Delay::new(int(0));
        let want = (d.id() ^ (d.id() >> 32)) as i32;
        let v = Value::Delay(d);
        assert_eq!(hasheq(&v), want);
        assert_eq!(hash_code(&v), want);
        let g = Agent::new(int(0));
        let want = (g.id() ^ (g.id() >> 32)) as i32;
        let v = Value::Agent(g);
        assert_eq!(hasheq(&v), want);
        assert_eq!(hash_code(&v), want);
    }

    #[test]
    fn seq_hash_unsupported_inside_sets() {
        use crate::RepeatValue;
        // Set members hash at insertion, so a set can only hold
        // supported members: the traversal always answers false
        // (an unbounded member would hang construction itself).
        let rep = Value::Extension(Arc::new(RepeatValue::new(int(1), Some(2))));
        assert!(!seq_hash_unsupported(&setv(vec![int(1), rep])));
        assert!(!seq_hash_unsupported(&setv(vec![int(1), int(2)])));
    }

    #[test]
    fn seq_hash_unsupported_ranges() {
        use crate::{RangeNum, RangeValue};
        let open = RangeValue::new(RangeNum::Int(0), None, RangeNum::Int(1), true, None);
        assert!(seq_hash_unsupported(&Value::Extension(Arc::new(open))));
        let zero_int = RangeValue::new(
            RangeNum::Int(0),
            Some(RangeNum::Int(10)),
            RangeNum::Int(0),
            true,
            Some(3),
        );
        assert!(seq_hash_unsupported(&Value::Extension(Arc::new(zero_int))));
        let zero_dbl = RangeValue::new(
            RangeNum::Double(0.0),
            Some(RangeNum::Double(1.0)),
            RangeNum::Double(0.0),
            false,
            None,
        );
        assert!(seq_hash_unsupported(&Value::Extension(Arc::new(zero_dbl))));
        let finite = RangeValue::new(
            RangeNum::Int(0),
            Some(RangeNum::Int(3)),
            RangeNum::Int(1),
            true,
            Some(3),
        );
        assert!(!seq_hash_unsupported(&Value::Extension(Arc::new(finite))));
    }

    #[test]
    fn subvector_hashes_as_sequential() {
        use crate::SubVectorValue;
        let backing = Vector::from_vec(vec![int(1), int(2), strv("x")]);
        let sv = Value::Extension(Arc::new(SubVectorValue::new(backing, 1, 3)));
        let expect = listv(vec![int(2), strv("x")]);
        assert_eq!(hasheq(&sv), hasheq(&expect));
        assert_eq!(hash_code(&sv), hash_code(&expect));
    }

    #[test]
    fn empty_string_takes_hash_int_fast_path() {
        assert_eq!(str_hashcode(""), 0);
        assert_eq!(hasheq(&strv("")), 0);
    }

    /// `hasheq_tower` rejects non-tower values (the caller
    /// dispatches on kind first — the panic is the contract).
    #[test]
    #[should_panic(expected = "caller passes tower values only")]
    fn hasheq_tower_rejects_non_tower() {
        let _ = super::hasheq_tower(&Value::Nil);
    }

    /// `scalar_hash` rejects non-scalars (both callers dispatch
    /// on kind first — the panic is the contract).
    #[test]
    #[should_panic(expected = "caller passes scalars only")]
    fn scalar_hash_rejects_non_scalar() {
        let _ = super::scalar_hash(&Value::Nil);
    }

    /// `hash_code` walks every tower arm (Java word formulas are
    /// stand-ins — see module docs — so the pins are determinism
    /// and scattering, not magic numbers).
    #[test]
    fn hash_code_tower_arms() {
        use num_bigint::BigInt;
        let big = |n: i32| Value::BigInt(Arc::new(BigInt::from(n)));
        let dec = |n: i32| Value::BigDec(Arc::new(crate::BigDec::from_bigint(BigInt::from(n))));
        let rat = |n: i32, d: i32| {
            Value::Ratio(Arc::new(num_rational::BigRational::new(
                BigInt::from(n),
                BigInt::from(d),
            )))
        };
        for v in [big(5), dec(5), rat(1, 2)] {
            assert_eq!(hash_code(&v), hash_code(&v));
        }
        assert_ne!(hash_code(&big(5)), hash_code(&big(6)));
        assert_ne!(hash_code(&dec(5)), hash_code(&dec(6)));
        assert_ne!(hash_code(&rat(1, 2)), hash_code(&rat(1, 3)));
    }

    #[test]
    fn str_murmur_is_order_sensitive() {
        assert_ne!(hasheq(&strv("ab")), hasheq(&strv("ba")));
        assert_ne!(hasheq(&strv("abcd")), hasheq(&strv("abdc")));
    }

    #[test]
    #[should_panic(expected = "hasheq partition")]
    fn hasheq_rest_rejects_upstream_variants() {
        // Defensive arm: the single caller partitions the variants,
        // so scalars never arrive here (only a dispatch bug could).
        let _ = super::hasheq_rest(&Value::Nil);
    }

    #[test]
    #[should_panic(expected = "hash_code partition")]
    fn hash_code_rest_rejects_upstream_variants() {
        // Defensive arm: the single caller partitions the variants,
        // so scalars never arrive here (only a dispatch bug could).
        let _ = super::hash_code_rest(&Value::Nil);
    }

    fn bigint(v: &str) -> Value {
        Value::BigInt(Arc::new(v.parse::<num_bigint::BigInt>().unwrap()))
    }

    fn biginteger(v: &str) -> Value {
        Value::BigInteger(Arc::new(v.parse::<num_bigint::BigInt>().unwrap()))
    }

    fn bigdec(v: &str) -> Value {
        Value::BigDec(Arc::new(crate::BigDec::parse(v).unwrap()))
    }

    fn ratio(n: &str, d: &str) -> Value {
        Value::Ratio(Arc::new(num_rational::BigRational::new(
            n.parse::<num_bigint::BigInt>().unwrap(),
            d.parse::<num_bigint::BigInt>().unwrap(),
        )))
    }

    /// Tower `hasheq` is JVM-identical: in-`long`-range `BigInt`s hash
    /// exactly like `Int` (`Murmur3.hashLong`); wider ones hash as
    /// `BigInteger.hashCode` (the `31`-chain over magnitude words times
    /// the sign); decimals hash as the trailing-zero-stripped
    /// `BigDecimal.hashCode` (`31 * unscaled + scale`, zeros are `0`);
    /// ratios hash as `numerator ^ denominator` (`Ratio.hashCode`).
    /// Integers below are transcribed from those JDK formulas.
    #[test]
    fn tower_hasheq_matches_jdk() {
        // In-range `BigInt`/`BigInteger` agree with `Int` (murmur).
        assert_eq!(hasheq(&bigint("5")), hasheq(&int(5)));
        assert_eq!(hasheq(&biginteger("5")), hasheq(&int(5)));
        assert_eq!(hasheq(&bigint("-1")), hasheq(&int(-1)));
        assert_eq!(hasheq(&biginteger("-1")), hasheq(&int(-1)));
        assert_eq!(
            hasheq(&bigint("9223372036854775807")),
            hasheq(&int(i64::MAX))
        );
        assert_eq!(
            hasheq(&biginteger("-9223372036854775808")),
            hasheq(&int(i64::MIN))
        );
        // The murmur range covers the whole `long` range (`2^32`
        // is `Murmur3.hashLong`, not the word formula).
        assert_eq!(hasheq(&bigint("4294967296")), 987_256_456);
        assert_eq!(hasheq(&bigint("4294967296")), hasheq(&int(4_294_967_296)));
        assert_eq!(hasheq(&bigint("4294967297")), hasheq(&int(4_294_967_297)));
        // Wide magnitudes are `BigInteger.hashCode`.
        assert_eq!(hasheq(&bigint("18446744073709551616")), 961);
        assert_eq!(hasheq(&bigint("-18446744073709551616")), -961);
        assert_eq!(hasheq(&bigint("9223372036854775808")), -2_147_483_648);
        assert_eq!(hasheq(&bigint("-9223372036854775809")), 2_147_483_647);
        assert_eq!(
            hasheq(&bigint("1000000000000000000000000000000")),
            1_319_562_394
        );
        assert_eq!(hasheq(&biginteger("18446744073709551616")), 961);
        // Decimals strip trailing zeros first (zeros of any scale are 0).
        assert_eq!(hasheq(&bigdec("1")), 31);
        assert_eq!(hasheq(&bigdec("1.0")), 31);
        assert_eq!(hasheq(&bigdec("1.00")), 31);
        assert_eq!(hasheq(&bigdec("0")), 0);
        assert_eq!(hasheq(&bigdec("0.00")), 0);
        assert_eq!(hasheq(&bigdec("0.5")), 156);
        assert_eq!(hasheq(&bigdec("100")), 29);
        assert_eq!(hasheq(&bigdec("1E+3")), 28);
        // Ratios are `numerator.hashCode ^ denominator.hashCode`.
        assert_eq!(hasheq(&ratio("1", "2")), 3);
        assert_eq!(hasheq(&ratio("3", "2")), 1);
        assert_eq!(hasheq(&ratio("-1", "2")), -3);
        assert_eq!(hasheq(&ratio("5", "3")), 6);
        assert_eq!(hasheq(&ratio("18446744073709551616", "3")), 962);
    }

    /// Tower `.hashCode` is JVM-identical too: in-range `BigInt` uses
    /// the `Long` bit mix, `BigInteger` always uses
    /// `BigInteger.hashCode`, decimals use the UNstripped
    /// `BigDecimal.hashCode` (scale counts: `0.0M` is `1`), ratios
    /// the same `num ^ den` as `hasheq`.
    #[test]
    fn tower_hash_code_matches_jdk() {
        assert_eq!(hash_code(&bigint("5")), 5);
        assert_eq!(hash_code(&bigint("-1")), 0);
        assert_eq!(hash_code(&bigint("18446744073709551616")), 961);
        assert_eq!(hash_code(&bigint("-18446744073709551616")), -961);
        assert_eq!(hash_code(&biginteger("5")), 5);
        assert_eq!(hash_code(&biginteger("-1")), -1);
        assert_eq!(hash_code(&biginteger("18446744073709551616")), 961);
        assert_eq!(hash_code(&bigdec("1")), 31);
        assert_eq!(hash_code(&bigdec("1.0")), 311);
        assert_eq!(hash_code(&bigdec("0")), 0);
        assert_eq!(hash_code(&bigdec("0.0")), 1);
        assert_eq!(hash_code(&ratio("1", "2")), 3);
        assert_eq!(hash_code(&ratio("-1", "2")), -3);
    }

    /// Equal values hash equal: the `=`-consistency battery over every
    /// numeric `=` rule (in-range `Int`/`BigInt`/`BigInteger`, wide
    /// magnitudes by value, scale-insensitive decimals, normalized
    /// ratios, `±0.0`). Each pair asserts `=` AND the shared `hasheq`.
    #[test]
    fn hasheq_consistent_with_equiv() {
        let pairs = [
            (int(5), bigint("5")),
            (int(5), biginteger("5")),
            (bigint("5"), biginteger("5")),
            (int(-1), bigint("-1")),
            (int(i64::MAX), bigint("9223372036854775807")),
            (int(i64::MIN), biginteger("-9223372036854775808")),
            (
                bigint("18446744073709551616"),
                biginteger("18446744073709551616"),
            ),
            (
                bigint("-1000000000000000000000000000000"),
                biginteger("-1000000000000000000000000000000"),
            ),
            (bigdec("5.0"), bigdec("5.00")),
            (bigdec("0"), bigdec("0.00")),
            (bigdec("100"), bigdec("1E+2")),
            (ratio("1", "2"), ratio("2", "4")),
            (ratio("-1", "2"), ratio("-2", "4")),
            (dbl(0.0), dbl(-0.0)),
        ];
        for (a, b) in &pairs {
            assert_eq!(a, b, "pair must be `=`: {a:?} {b:?}");
            assert_eq!(b, a, "pair must be `=` both ways");
            assert_eq!(hasheq(a), hasheq(b), "equal values hash equal: {a:?} {b:?}");
        }
    }
}
