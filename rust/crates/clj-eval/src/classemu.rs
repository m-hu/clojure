// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Class emulation (S16, issue #60): `Class` values over the emulated
//! JVM lattice — `instance?`/`type`/`class`/`bases`/`supers`/
//! `class?`/`cast`/`bean`/`identical?`, the `import`/`..`/`memfn`
//! macros (defined in `core.clj`, backed by the hidden `__import*`
//! native here), and the `..` member surface consulted by `__meth*`
//! before the host dispatch.
//!
//! The lattice is one static table (transcribed verbatim from
//! `docs/spec/zero-oos-s16-classemu/probes/lattice.out`, plus the five
//! documented synthetic rows, the two unprobed import-only rows,
//! and the 72 #74 rows)
//! with four dynamic rules: the HostDef
//! rule (host types and their claimed names), the fn rule (closures
//! by rest-presence, natives unconditionally `AFunction`), the future
//! / promise / iteration stable names (static reify rows), and the
//! struct carve-outs (`PersistentStructMap` / `$Def`). Walks recurse
//! through the registry, skip names that no longer resolve (the
//! redefinition leniency), and carry a visited set (redefinition can
//! link `HostDef`s cyclically).

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use clj_value::{
    Array, ArrayDesc, ArrayStore, BaseType, BigDec, ClassVal, ExtensionValue, HostDef, HostKind,
    HostVal, Keyword, List, Map, MathContext, RoundingMode, SeqKind, Symbol, ThrowClass, Value,
    Vector, bigdec_result, new_map, new_set,
};
use num_traits::ToPrimitive;
use rand::Rng;

use crate::error::{NativeError, jvm_class_of, simple_name_of};
use crate::eval::Evaluator;
use crate::ext::{Closure, NativeFn};
use crate::ns::{NativeFnPtr, check_arity, signal_to_native};

/// The S16 natives: class queries, `bean`, and `identical?`
/// (`PrintWriter-on` lives with the writers in [`crate::io`]).
pub(crate) const CLASS_EMU_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("instance?", native_instance),
    ("type", native_type),
    ("class", native_class),
    ("bases", native_bases),
    ("supers", native_supers),
    ("class?", native_classp),
    ("cast", native_cast),
    ("bean", native_bean),
    ("identical?", native_identical),
];

/// Hidden natives: `__import*` is an implementation detail of the
/// `import` macro in the standing of `__new*` (public so macro
/// expansions analyze, unlisted in the matrix); `__static*` is the
/// analyzer's static-call lowering (`Class/forName` plus the
/// `System` statics, #120).
pub(crate) const CLASS_EMU_HIDDEN: &[(&str, NativeFnPtr)] = &[
    ("__import*", native_import_star),
    ("__static*", native_static),
];

/// One static lattice row: binary name, interface flag, and the
/// ordered direct-supertype names, transcribed verbatim from the
/// lattice probe output (the 214 static rows; the 13 dynamic rows
/// live in the construction rules below).
struct StaticRow {
    name: &'static str,
    iface: bool,
    bases: &'static [&'static str],
}

/// The static lattice: the 214 oracle rows plus the documented
/// synthetic rows (the future/promise/iteration reifies, the
/// nonexistent MissingFormatConversionException, and
/// NoSuchElementException for the S16 getFirst/getLast variant),
/// the two unprobed import-only rows (`java.util.Calendar`,
/// `java.sql.Date`), the 72 #74 rows (65 defaults + 7 closure,
/// JDK 21 declarations), and the #205 emulation rows (JDK 21
/// declarations, appended at the table's end).
const STATIC_LATTICE: &[StaticRow] = &[
    StaticRow {
        name: "clojure.core.Eduction",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.lang.Iterable",
            "clojure.lang.IReduceInit",
            "clojure.lang.Sequential",
            "clojure.lang.IType",
        ],
    },
    StaticRow {
        name: "clojure.lang.AFn",
        iface: false,
        bases: &["java.lang.Object", "clojure.lang.IFn"],
    },
    StaticRow {
        name: "clojure.lang.AFunction",
        iface: false,
        bases: &[
            "clojure.lang.AFn",
            "clojure.lang.IObj",
            "java.util.Comparator",
            "clojure.lang.Fn",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "clojure.lang.AMapEntry",
        iface: false,
        bases: &["clojure.lang.APersistentVector", "clojure.lang.IMapEntry"],
    },
    StaticRow {
        name: "clojure.lang.APersistentMap",
        iface: false,
        bases: &[
            "clojure.lang.AFn",
            "clojure.lang.IPersistentMap",
            "java.util.Map",
            "java.lang.Iterable",
            "java.io.Serializable",
            "clojure.lang.MapEquivalence",
            "clojure.lang.IHashEq",
        ],
    },
    StaticRow {
        name: "clojure.lang.APersistentMap$KeySeq",
        iface: false,
        bases: &["clojure.lang.ASeq"],
    },
    StaticRow {
        name: "clojure.lang.APersistentMap$ValSeq",
        iface: false,
        bases: &["clojure.lang.ASeq"],
    },
    StaticRow {
        name: "clojure.lang.APersistentSet",
        iface: false,
        bases: &[
            "clojure.lang.AFn",
            "clojure.lang.IPersistentSet",
            "java.util.Collection",
            "java.util.Set",
            "java.io.Serializable",
            "clojure.lang.IHashEq",
        ],
    },
    StaticRow {
        name: "clojure.lang.APersistentVector",
        iface: false,
        bases: &[
            "clojure.lang.AFn",
            "clojure.lang.IPersistentVector",
            "java.lang.Iterable",
            "java.util.List",
            "java.util.RandomAccess",
            "java.lang.Comparable",
            "java.io.Serializable",
            "clojure.lang.IHashEq",
        ],
    },
    StaticRow {
        name: "clojure.lang.APersistentVector$RSeq",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IndexedSeq",
            "clojure.lang.Counted",
        ],
    },
    StaticRow {
        name: "clojure.lang.APersistentVector$Seq",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IndexedSeq",
            "clojure.lang.IReduce",
        ],
    },
    StaticRow {
        name: "clojure.lang.APersistentVector$SubVector",
        iface: false,
        bases: &[
            "clojure.lang.APersistentVector",
            "clojure.lang.IObj",
            "clojure.lang.IKVReduce",
        ],
    },
    StaticRow {
        name: "clojure.lang.ARef",
        iface: false,
        bases: &["clojure.lang.AReference", "clojure.lang.IRef"],
    },
    StaticRow {
        name: "clojure.lang.AReference",
        iface: false,
        bases: &["java.lang.Object", "clojure.lang.IReference"],
    },
    StaticRow {
        name: "clojure.lang.ASeq",
        iface: false,
        bases: &[
            "clojure.lang.Obj",
            "clojure.lang.ISeq",
            "clojure.lang.Sequential",
            "java.util.List",
            "java.io.Serializable",
            "clojure.lang.IHashEq",
        ],
    },
    StaticRow {
        name: "clojure.lang.ATransientMap",
        iface: false,
        bases: &[
            "clojure.lang.AFn",
            "clojure.lang.ITransientMap",
            "clojure.lang.ITransientAssociative2",
        ],
    },
    StaticRow {
        name: "clojure.lang.ATransientSet",
        iface: false,
        bases: &["clojure.lang.AFn", "clojure.lang.ITransientSet"],
    },
    StaticRow {
        name: "clojure.lang.Agent",
        iface: false,
        bases: &["clojure.lang.ARef"],
    },
    StaticRow {
        name: "clojure.lang.ArityException",
        iface: false,
        bases: &["java.lang.IllegalArgumentException"],
    },
    StaticRow {
        name: "clojure.lang.ArrayChunk",
        iface: false,
        bases: &[
            "java.lang.Object",
            "clojure.lang.IChunk",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "clojure.lang.ArraySeq",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IndexedSeq",
            "clojure.lang.IReduce",
        ],
    },
    StaticRow {
        name: "clojure.lang.Associative",
        iface: true,
        bases: &["clojure.lang.IPersistentCollection", "clojure.lang.ILookup"],
    },
    StaticRow {
        name: "clojure.lang.Atom",
        iface: false,
        bases: &["clojure.lang.ARef", "clojure.lang.IAtom2"],
    },
    StaticRow {
        name: "clojure.lang.BigInt",
        iface: false,
        bases: &["java.lang.Number", "clojure.lang.IHashEq"],
    },
    StaticRow {
        name: "clojure.lang.ChunkBuffer",
        iface: false,
        bases: &["java.lang.Object", "clojure.lang.Counted"],
    },
    StaticRow {
        name: "clojure.lang.ChunkedCons",
        iface: false,
        bases: &["clojure.lang.ASeq", "clojure.lang.IChunkedSeq"],
    },
    StaticRow {
        name: "clojure.lang.Compiler$CompilerException",
        iface: false,
        bases: &["java.lang.RuntimeException", "clojure.lang.IExceptionInfo"],
    },
    StaticRow {
        name: "clojure.lang.Cons",
        iface: false,
        bases: &["clojure.lang.ASeq", "java.io.Serializable"],
    },
    StaticRow {
        name: "clojure.lang.Counted",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.Cycle",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IReduce",
            "clojure.lang.IPending",
        ],
    },
    StaticRow {
        name: "clojure.lang.Delay",
        iface: false,
        bases: &[
            "java.lang.Object",
            "clojure.lang.IDeref",
            "clojure.lang.IPending",
        ],
    },
    StaticRow {
        name: "clojure.lang.ExceptionInfo",
        iface: false,
        bases: &["java.lang.RuntimeException", "clojure.lang.IExceptionInfo"],
    },
    StaticRow {
        name: "clojure.lang.Fn",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IAtom",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IAtom2",
        iface: true,
        bases: &["clojure.lang.IAtom"],
    },
    StaticRow {
        name: "clojure.lang.IBlockingDeref",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IChunk",
        iface: true,
        bases: &["clojure.lang.Indexed"],
    },
    StaticRow {
        name: "clojure.lang.IChunkedSeq",
        iface: true,
        bases: &["clojure.lang.ISeq", "clojure.lang.Sequential"],
    },
    StaticRow {
        name: "clojure.lang.IDeref",
        iface: true,
        bases: &[
            "java.util.function.Supplier",
            "java.util.function.BooleanSupplier",
            "java.util.function.IntSupplier",
            "java.util.function.LongSupplier",
            "java.util.function.DoubleSupplier",
        ],
    },
    StaticRow {
        name: "clojure.lang.IDrop",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IEditableCollection",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IExceptionInfo",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IFn",
        iface: true,
        bases: &["java.util.concurrent.Callable", "java.lang.Runnable"],
    },
    StaticRow {
        name: "clojure.lang.IHashEq",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IKVReduce",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IKeywordLookup",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.ILookup",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IMapEntry",
        iface: true,
        bases: &["java.util.Map$Entry"],
    },
    StaticRow {
        name: "clojure.lang.IMapIterable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IMeta",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IObj",
        iface: true,
        bases: &["clojure.lang.IMeta"],
    },
    StaticRow {
        name: "clojure.lang.IPending",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IPersistentCollection",
        iface: true,
        bases: &["clojure.lang.Seqable"],
    },
    StaticRow {
        name: "clojure.lang.IPersistentList",
        iface: true,
        bases: &["clojure.lang.Sequential", "clojure.lang.IPersistentStack"],
    },
    StaticRow {
        name: "clojure.lang.IPersistentMap",
        iface: true,
        bases: &[
            "java.lang.Iterable",
            "clojure.lang.Associative",
            "clojure.lang.Counted",
        ],
    },
    StaticRow {
        name: "clojure.lang.IPersistentSet",
        iface: true,
        bases: &[
            "clojure.lang.IPersistentCollection",
            "clojure.lang.Counted",
            "clojure.lang.ILookup",
        ],
    },
    StaticRow {
        name: "clojure.lang.IPersistentStack",
        iface: true,
        bases: &["clojure.lang.IPersistentCollection"],
    },
    StaticRow {
        name: "clojure.lang.IPersistentVector",
        iface: true,
        bases: &[
            "clojure.lang.Associative",
            "clojure.lang.Sequential",
            "clojure.lang.IPersistentStack",
            "clojure.lang.Reversible",
            "clojure.lang.Indexed",
        ],
    },
    StaticRow {
        name: "clojure.lang.IRecord",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IReduce",
        iface: true,
        bases: &["clojure.lang.IReduceInit"],
    },
    StaticRow {
        name: "clojure.lang.IReduceInit",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.IRef",
        iface: true,
        bases: &["clojure.lang.IDeref"],
    },
    StaticRow {
        name: "clojure.lang.IReference",
        iface: true,
        bases: &["clojure.lang.IMeta"],
    },
    StaticRow {
        name: "clojure.lang.ISeq",
        iface: true,
        bases: &["clojure.lang.IPersistentCollection"],
    },
    StaticRow {
        name: "clojure.lang.ITransientAssociative",
        iface: true,
        bases: &["clojure.lang.ITransientCollection", "clojure.lang.ILookup"],
    },
    StaticRow {
        name: "clojure.lang.ITransientAssociative2",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.ITransientCollection",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.ITransientMap",
        iface: true,
        bases: &["clojure.lang.ITransientAssociative", "clojure.lang.Counted"],
    },
    StaticRow {
        name: "clojure.lang.ITransientSet",
        iface: true,
        bases: &[
            "clojure.lang.ITransientCollection",
            "clojure.lang.Counted",
            "clojure.lang.ILookup",
        ],
    },
    StaticRow {
        name: "clojure.lang.ITransientVector",
        iface: true,
        bases: &["clojure.lang.ITransientAssociative", "clojure.lang.Indexed"],
    },
    StaticRow {
        name: "clojure.lang.IType",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.Indexed",
        iface: true,
        bases: &["clojure.lang.Counted"],
    },
    StaticRow {
        name: "clojure.lang.IndexedSeq",
        iface: true,
        bases: &[
            "clojure.lang.ISeq",
            "clojure.lang.Sequential",
            "clojure.lang.Counted",
        ],
    },
    StaticRow {
        name: "clojure.lang.Iterate",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IReduce",
            "clojure.lang.IPending",
        ],
    },
    StaticRow {
        name: "clojure.lang.Keyword",
        iface: false,
        bases: &[
            "java.lang.Object",
            "clojure.lang.IFn",
            "java.lang.Comparable",
            "clojure.lang.Named",
            "java.io.Serializable",
            "clojure.lang.IHashEq",
        ],
    },
    StaticRow {
        name: "clojure.lang.LazilyPersistentVector",
        iface: false,
        bases: &[
            "clojure.lang.APersistentVector",
            "clojure.lang.ISeq",
            "clojure.lang.Sequential",
            "java.util.List",
        ],
    },
    StaticRow {
        name: "clojure.lang.LazySeq",
        iface: false,
        bases: &[
            "clojure.lang.Obj",
            "clojure.lang.ISeq",
            "clojure.lang.Sequential",
            "java.util.List",
            "clojure.lang.IPending",
            "clojure.lang.IHashEq",
        ],
    },
    StaticRow {
        name: "clojure.lang.LineNumberingPushbackReader",
        iface: false,
        bases: &["java.io.PushbackReader"],
    },
    StaticRow {
        name: "clojure.lang.LispReader$ReaderException",
        iface: false,
        bases: &["java.lang.RuntimeException", "clojure.lang.IExceptionInfo"],
    },
    StaticRow {
        name: "clojure.lang.LongRange",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.Counted",
            "clojure.lang.IChunkedSeq",
            "clojure.lang.IReduce",
            "clojure.lang.IDrop",
        ],
    },
    StaticRow {
        name: "clojure.lang.MapEntry",
        iface: false,
        bases: &["clojure.lang.AMapEntry"],
    },
    StaticRow {
        name: "clojure.lang.MapEquivalence",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.MultiFn",
        iface: false,
        bases: &["clojure.lang.AFn"],
    },
    StaticRow {
        name: "clojure.lang.Named",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.Namespace",
        iface: false,
        bases: &["clojure.lang.AReference", "java.io.Serializable"],
    },
    StaticRow {
        name: "clojure.lang.Obj",
        iface: false,
        bases: &[
            "java.lang.Object",
            "clojure.lang.IObj",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentArrayMap",
        iface: false,
        bases: &[
            "clojure.lang.APersistentMap",
            "clojure.lang.IObj",
            "clojure.lang.IEditableCollection",
            "clojure.lang.IMapIterable",
            "clojure.lang.IKVReduce",
            "clojure.lang.IDrop",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentArrayMap$Seq",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.Counted",
            "clojure.lang.IReduce",
            "clojure.lang.IDrop",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentArrayMap$TransientArrayMap",
        iface: false,
        bases: &["clojure.lang.ATransientMap"],
    },
    StaticRow {
        name: "clojure.lang.PersistentHashMap",
        iface: false,
        bases: &[
            "clojure.lang.APersistentMap",
            "clojure.lang.IEditableCollection",
            "clojure.lang.IObj",
            "clojure.lang.IMapIterable",
            "clojure.lang.IKVReduce",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentHashMap$ArrayNode$Seq",
        iface: false,
        bases: &["clojure.lang.ASeq"],
    },
    StaticRow {
        name: "clojure.lang.PersistentHashMap$TransientHashMap",
        iface: false,
        bases: &["clojure.lang.ATransientMap"],
    },
    StaticRow {
        name: "clojure.lang.PersistentHashSet",
        iface: false,
        bases: &[
            "clojure.lang.APersistentSet",
            "clojure.lang.IObj",
            "clojure.lang.IEditableCollection",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentHashSet$TransientHashSet",
        iface: false,
        bases: &["clojure.lang.ATransientSet"],
    },
    StaticRow {
        name: "clojure.lang.PersistentList",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IPersistentList",
            "clojure.lang.IReduce",
            "java.util.List",
            "clojure.lang.Counted",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentList$EmptyList",
        iface: false,
        bases: &[
            "clojure.lang.Obj",
            "clojure.lang.IPersistentList",
            "java.util.List",
            "clojure.lang.ISeq",
            "clojure.lang.Counted",
            "clojure.lang.IHashEq",
        ],
    },
    // Oracle: superclass `Obj`, interfaces `Counted`, `IHashEq`,
    // `IPersistentList`, `Collection` (no `List`, no
    // `Serializable`).
    StaticRow {
        name: "clojure.lang.PersistentQueue",
        iface: false,
        bases: &[
            "clojure.lang.Obj",
            "clojure.lang.Counted",
            "clojure.lang.IHashEq",
            "clojure.lang.IPersistentList",
            "java.util.Collection",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentStructMap",
        iface: false,
        bases: &["clojure.lang.APersistentMap", "clojure.lang.IObj"],
    },
    StaticRow {
        name: "clojure.lang.PersistentStructMap$1",
        iface: false,
        bases: &["clojure.lang.AFn"],
    },
    StaticRow {
        name: "clojure.lang.PersistentStructMap$Def",
        iface: false,
        bases: &["java.lang.Object", "java.io.Serializable"],
    },
    StaticRow {
        name: "clojure.lang.PersistentTreeMap",
        iface: false,
        bases: &[
            "clojure.lang.APersistentMap",
            "clojure.lang.IObj",
            "clojure.lang.Reversible",
            "clojure.lang.Sorted",
            "clojure.lang.IKVReduce",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentTreeMap$Seq",
        iface: false,
        bases: &["clojure.lang.ASeq"],
    },
    StaticRow {
        name: "clojure.lang.PersistentTreeSet",
        iface: false,
        bases: &[
            "clojure.lang.APersistentSet",
            "clojure.lang.IObj",
            "clojure.lang.Reversible",
            "clojure.lang.Sorted",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentVector",
        iface: false,
        bases: &[
            "clojure.lang.APersistentVector",
            "clojure.lang.IObj",
            "clojure.lang.IEditableCollection",
            "clojure.lang.IReduce",
            "clojure.lang.IKVReduce",
            "clojure.lang.IDrop",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentVector$ChunkedSeq",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IChunkedSeq",
            "clojure.lang.Counted",
            "clojure.lang.IReduce",
            "clojure.lang.IDrop",
        ],
    },
    StaticRow {
        name: "clojure.lang.PersistentVector$TransientVector",
        iface: false,
        bases: &[
            "clojure.lang.AFn",
            "clojure.lang.ITransientVector",
            "clojure.lang.ITransientAssociative2",
            "clojure.lang.Counted",
        ],
    },
    StaticRow {
        name: "clojure.lang.Range",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IChunkedSeq",
            "clojure.lang.IReduce",
        ],
    },
    StaticRow {
        name: "clojure.lang.Ratio",
        iface: false,
        bases: &["java.lang.Number", "java.lang.Comparable"],
    },
    StaticRow {
        name: "clojure.lang.ReaderConditional",
        iface: false,
        bases: &["java.lang.Object", "clojure.lang.ILookup"],
    },
    StaticRow {
        name: "clojure.lang.Reduced",
        iface: false,
        bases: &["java.lang.Object", "clojure.lang.IDeref"],
    },
    StaticRow {
        name: "clojure.lang.Repeat",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IReduce",
            "clojure.lang.IDrop",
        ],
    },
    StaticRow {
        name: "clojure.lang.RestFn",
        iface: false,
        bases: &["clojure.lang.AFunction"],
    },
    StaticRow {
        name: "clojure.lang.Reversible",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.Seqable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.Sequential",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.Settable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.Sorted",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "clojure.lang.StringSeq",
        iface: false,
        bases: &[
            "clojure.lang.ASeq",
            "clojure.lang.IndexedSeq",
            "clojure.lang.IDrop",
            "clojure.lang.IReduceInit",
        ],
    },
    StaticRow {
        name: "clojure.lang.Symbol",
        iface: false,
        bases: &[
            "clojure.lang.AFn",
            "clojure.lang.IObj",
            "java.lang.Comparable",
            "clojure.lang.Named",
            "java.io.Serializable",
            "clojure.lang.IHashEq",
        ],
    },
    StaticRow {
        name: "clojure.lang.TaggedLiteral",
        iface: false,
        bases: &["java.lang.Object", "clojure.lang.ILookup"],
    },
    StaticRow {
        name: "clojure.lang.Var",
        iface: false,
        bases: &[
            "clojure.lang.ARef",
            "clojure.lang.IFn",
            "clojure.lang.IRef",
            "clojure.lang.Settable",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "clojure.lang.Var$Unbound",
        iface: false,
        bases: &["clojure.lang.AFn"],
    },
    StaticRow {
        name: "clojure.lang.Volatile",
        iface: false,
        bases: &["java.lang.Object", "clojure.lang.IDeref"],
    },
    StaticRow {
        name: "java.io.Closeable",
        iface: true,
        bases: &["java.lang.AutoCloseable"],
    },
    StaticRow {
        name: "java.io.File",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.io.Serializable",
            "java.lang.Comparable",
        ],
    },
    StaticRow {
        name: "java.io.FileInputStream",
        iface: false,
        bases: &["java.io.InputStream"],
    },
    StaticRow {
        name: "java.io.FileNotFoundException",
        iface: false,
        bases: &["java.io.IOException"],
    },
    StaticRow {
        name: "java.io.FileOutputStream",
        iface: false,
        bases: &["java.io.OutputStream"],
    },
    StaticRow {
        name: "java.io.FilterReader",
        iface: false,
        bases: &["java.io.Reader"],
    },
    StaticRow {
        name: "java.io.Flushable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.io.IOException",
        iface: false,
        bases: &["java.lang.Exception"],
    },
    StaticRow {
        name: "java.io.InputStream",
        iface: false,
        bases: &["java.lang.Object", "java.io.Closeable"],
    },
    StaticRow {
        name: "java.io.OutputStream",
        iface: false,
        bases: &["java.lang.Object", "java.io.Closeable", "java.io.Flushable"],
    },
    StaticRow {
        name: "java.io.OutputStreamWriter",
        iface: false,
        bases: &["java.io.Writer"],
    },
    StaticRow {
        name: "java.io.PrintWriter",
        iface: false,
        bases: &["java.io.Writer"],
    },
    StaticRow {
        name: "java.io.PushbackReader",
        iface: false,
        bases: &["java.io.FilterReader"],
    },
    StaticRow {
        name: "java.io.Reader",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.lang.Readable",
            "java.io.Closeable",
        ],
    },
    StaticRow {
        name: "java.io.Serializable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.io.StringWriter",
        iface: false,
        bases: &["java.io.Writer"],
    },
    StaticRow {
        name: "java.io.Writer",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.lang.Appendable",
            "java.io.Closeable",
            "java.io.Flushable",
        ],
    },
    StaticRow {
        name: "java.lang.Appendable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.ArithmeticException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.ArrayIndexOutOfBoundsException",
        iface: false,
        bases: &["java.lang.IndexOutOfBoundsException"],
    },
    StaticRow {
        name: "java.lang.AssertionError",
        iface: false,
        bases: &["java.lang.Error"],
    },
    StaticRow {
        name: "java.lang.AutoCloseable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.Boolean",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.io.Serializable",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
        ],
    },
    StaticRow {
        name: "java.lang.CharSequence",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.Character",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.io.Serializable",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
        ],
    },
    StaticRow {
        name: "java.lang.Class",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.io.Serializable",
            "java.lang.reflect.GenericDeclaration",
            "java.lang.reflect.Type",
            "java.lang.reflect.AnnotatedElement",
            "java.lang.invoke.TypeDescriptor$OfField",
            "java.lang.constant.Constable",
        ],
    },
    StaticRow {
        name: "java.lang.ClassCastException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.ClassNotFoundException",
        iface: false,
        bases: &["java.lang.ReflectiveOperationException"],
    },
    StaticRow {
        name: "java.lang.Cloneable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.Comparable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.Double",
        iface: false,
        bases: &[
            "java.lang.Number",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
            "java.lang.constant.ConstantDesc",
        ],
    },
    StaticRow {
        name: "java.lang.Error",
        iface: false,
        bases: &["java.lang.Throwable"],
    },
    StaticRow {
        name: "java.lang.Exception",
        iface: false,
        bases: &["java.lang.Throwable"],
    },
    StaticRow {
        name: "java.lang.IllegalAccessError",
        iface: false,
        bases: &["java.lang.IncompatibleClassChangeError"],
    },
    StaticRow {
        name: "java.lang.IllegalArgumentException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.IllegalStateException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.IncompatibleClassChangeError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.IndexOutOfBoundsException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.Iterable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.LinkageError",
        iface: false,
        bases: &["java.lang.Error"],
    },
    StaticRow {
        name: "java.lang.Long",
        iface: false,
        bases: &[
            "java.lang.Number",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
            "java.lang.constant.ConstantDesc",
        ],
    },
    StaticRow {
        name: "java.lang.NegativeArraySizeException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.NullPointerException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.Number",
        iface: false,
        bases: &["java.lang.Object", "java.io.Serializable"],
    },
    StaticRow {
        name: "java.lang.NumberFormatException",
        iface: false,
        bases: &["java.lang.IllegalArgumentException"],
    },
    StaticRow {
        name: "java.lang.Object",
        iface: false,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.Readable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.ReflectiveOperationException",
        iface: false,
        bases: &["java.lang.Exception"],
    },
    StaticRow {
        name: "java.lang.Runnable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.RuntimeException",
        iface: false,
        bases: &["java.lang.Exception"],
    },
    StaticRow {
        name: "java.lang.String",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.io.Serializable",
            "java.lang.Comparable",
            "java.lang.CharSequence",
            "java.lang.constant.Constable",
            "java.lang.constant.ConstantDesc",
        ],
    },
    StaticRow {
        name: "java.lang.StringIndexOutOfBoundsException",
        iface: false,
        bases: &["java.lang.IndexOutOfBoundsException"],
    },
    StaticRow {
        name: "java.lang.Throwable",
        iface: false,
        bases: &["java.lang.Object", "java.io.Serializable"],
    },
    StaticRow {
        name: "java.lang.UnsupportedOperationException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.constant.Constable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.constant.ConstantDesc",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.invoke.TypeDescriptor",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.invoke.TypeDescriptor$OfField",
        iface: true,
        bases: &["java.lang.invoke.TypeDescriptor"],
    },
    StaticRow {
        name: "java.lang.reflect.AnnotatedElement",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.reflect.GenericDeclaration",
        iface: true,
        bases: &["java.lang.reflect.AnnotatedElement"],
    },
    StaticRow {
        name: "java.lang.reflect.Type",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.math.BigDecimal",
        iface: false,
        bases: &["java.lang.Number", "java.lang.Comparable"],
    },
    StaticRow {
        name: "java.math.BigInteger",
        iface: false,
        bases: &["java.lang.Number", "java.lang.Comparable"],
    },
    StaticRow {
        name: "java.math.MathContext",
        iface: false,
        bases: &["java.lang.Object", "java.io.Serializable"],
    },
    StaticRow {
        name: "java.math.RoundingMode",
        iface: false,
        bases: &["java.lang.Enum"],
    },
    // Import-only (unprobed; ancestry verified live on the JVM):
    // `i-conflict-class` needs a second `Date` to collide with.
    StaticRow {
        name: "java.sql.Date",
        iface: false,
        bases: &["java.util.Date"],
    },
    // Import-only (unprobed; ancestry verified live on the JVM):
    // `i-import-multi-ret` needs a second `java.util` class.
    StaticRow {
        name: "java.util.Calendar",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.io.Serializable",
            "java.lang.Cloneable",
            "java.lang.Comparable",
        ],
    },
    StaticRow {
        name: "java.util.Collection",
        iface: true,
        bases: &["java.lang.Iterable"],
    },
    StaticRow {
        name: "java.util.Comparator",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.Date",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.io.Serializable",
            "java.lang.Cloneable",
            "java.lang.Comparable",
        ],
    },
    StaticRow {
        name: "java.util.DuplicateFormatFlagsException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.FormatFlagsConversionMismatchException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.IllegalFormatArgumentIndexException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.IllegalFormatCodePointException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.IllegalFormatConversionException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.IllegalFormatException",
        iface: false,
        bases: &["java.lang.IllegalArgumentException"],
    },
    StaticRow {
        name: "java.util.IllegalFormatFlagsException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.IllegalFormatPrecisionException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.IllegalFormatWidthException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.Locale",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.lang.Cloneable",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.util.List",
        iface: true,
        bases: &["java.util.SequencedCollection"],
    },
    StaticRow {
        name: "java.util.Map",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.Map$Entry",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.MissingFormatArgumentException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.MissingFormatWidthException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.RandomAccess",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.SequencedCollection",
        iface: true,
        bases: &["java.util.Collection"],
    },
    StaticRow {
        name: "java.util.Set",
        iface: true,
        bases: &["java.util.Collection"],
    },
    StaticRow {
        name: "java.util.UnknownFormatConversionException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    StaticRow {
        name: "java.util.concurrent.Callable",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.concurrent.CancellationException",
        iface: false,
        bases: &["java.lang.IllegalStateException"],
    },
    // Fence 3c: `(executor)` values report the classic delegated
    // service (ancestry probed from `newSingleThreadExecutor` on an
    // older JDK — newer JDKs interpose an `AutoShutdown` subclass;
    // port-fixed).
    StaticRow {
        name: "java.util.concurrent.Executor",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.concurrent.ExecutorService",
        iface: true,
        bases: &["java.util.concurrent.Executor"],
    },
    StaticRow {
        name: "java.util.concurrent.Executors$DelegatedExecutorService",
        iface: false,
        bases: &["java.lang.Object", "java.util.concurrent.ExecutorService"],
    },
    StaticRow {
        name: "java.util.concurrent.Executors$FinalizableDelegatedExecutorService",
        iface: false,
        bases: &["java.util.concurrent.Executors$DelegatedExecutorService"],
    },
    StaticRow {
        name: "java.util.concurrent.ExecutionException",
        iface: false,
        bases: &["java.lang.Exception"],
    },
    StaticRow {
        name: "java.util.concurrent.Future",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.concurrent.RejectedExecutionException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.util.function.BooleanSupplier",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.function.DoubleSupplier",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.function.IntSupplier",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.function.LongSupplier",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.function.Supplier",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.regex.MatchResult",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.util.regex.Matcher",
        iface: false,
        bases: &["java.lang.Object", "java.util.regex.MatchResult"],
    },
    StaticRow {
        name: "java.util.regex.Pattern",
        iface: false,
        bases: &["java.lang.Object", "java.io.Serializable"],
    },
    StaticRow {
        name: "java.util.regex.PatternSyntaxException",
        iface: false,
        bases: &["java.lang.IllegalArgumentException"],
    },
    // Synthetic: future reify (JVM ancestry copied from the l-future probe row).
    StaticRow {
        name: "clojure.core$future_call$reify",
        iface: false,
        bases: &[
            "java.lang.Object",
            "clojure.lang.IPending",
            "clojure.lang.IBlockingDeref",
            "java.util.concurrent.Future",
            "clojure.lang.IDeref",
            "clojure.lang.IObj",
        ],
    },
    // Synthetic: promise reify (JVM ancestry copied from the l-promise probe row).
    StaticRow {
        name: "clojure.core$promise$reify",
        iface: false,
        bases: &[
            "java.lang.Object",
            "clojure.lang.IPending",
            "clojure.lang.IBlockingDeref",
            "clojure.lang.IDeref",
            "clojure.lang.IFn",
            "clojure.lang.IObj",
        ],
    },
    // Synthetic: iteration (JVM ancestry copied from the l-iteration probe row).
    StaticRow {
        name: "clojure.core$iteration",
        iface: false,
        bases: &[
            "java.lang.Object",
            "clojure.lang.IReduceInit",
            "clojure.lang.Seqable",
            "clojure.lang.IObj",
        ],
    },
    // Synthetic: the nonexistent MissingFormatConversionException (throw-chain shape).
    StaticRow {
        name: "java.util.MissingFormatConversionException",
        iface: false,
        bases: &["java.util.IllegalFormatException"],
    },
    // Synthetic: NoSuchElementException (live-JVM ancestry; the S16 getFirst/getLast variant).
    StaticRow {
        name: "java.util.NoSuchElementException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    // #74: the default-import rows with no oracle probe (JDK 21
    // class declarations in superclass-first, declaration-order
    // shape — the same source the 214 oracle rows agree with where
    // they overlap, e.g. `Character`, `Boolean`, `Long`, `BigDecimal`).
    // `lattice_defaults_rows` pins each `[isInterface, bases]`.
    // Surprises cross-checked against jdk21u sources: no
    // `java.lang.Compiler` row (the class was removed in 21), and
    // the `Appendable` in the `StringBuffer`/`Builder` bases is
    // explicit in the JDK 21 `implements` clause.
    StaticRow {
        name: "java.lang.AbstractMethodError",
        iface: false,
        bases: &["java.lang.IncompatibleClassChangeError"],
    },
    StaticRow {
        name: "java.lang.ArrayStoreException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.BootstrapMethodError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.Byte",
        iface: false,
        bases: &[
            "java.lang.Number",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
        ],
    },
    StaticRow {
        name: "java.lang.ClassCircularityError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.ClassFormatError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.ClassLoader",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.ClassValue",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.CloneNotSupportedException",
        iface: false,
        bases: &["java.lang.Exception"],
    },
    StaticRow {
        name: "java.lang.Deprecated",
        iface: true,
        bases: &["java.lang.annotation.Annotation"],
    },
    StaticRow {
        name: "java.lang.Enum",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.lang.constant.Constable",
            "java.lang.Comparable",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.lang.EnumConstantNotPresentException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.ExceptionInInitializerError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.Float",
        iface: false,
        bases: &[
            "java.lang.Number",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
            "java.lang.constant.ConstantDesc",
        ],
    },
    StaticRow {
        name: "java.lang.IllegalAccessException",
        iface: false,
        bases: &["java.lang.ReflectiveOperationException"],
    },
    StaticRow {
        name: "java.lang.IllegalCallerException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.IllegalMonitorStateException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.IllegalThreadStateException",
        iface: false,
        bases: &["java.lang.IllegalArgumentException"],
    },
    StaticRow {
        name: "java.lang.InheritableThreadLocal",
        iface: false,
        bases: &["java.lang.ThreadLocal"],
    },
    StaticRow {
        name: "java.lang.InstantiationError",
        iface: false,
        bases: &["java.lang.IncompatibleClassChangeError"],
    },
    StaticRow {
        name: "java.lang.InstantiationException",
        iface: false,
        bases: &["java.lang.ReflectiveOperationException"],
    },
    StaticRow {
        name: "java.lang.Integer",
        iface: false,
        bases: &[
            "java.lang.Number",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
            "java.lang.constant.ConstantDesc",
        ],
    },
    StaticRow {
        name: "java.lang.InternalError",
        iface: false,
        bases: &["java.lang.VirtualMachineError"],
    },
    StaticRow {
        name: "java.lang.InterruptedException",
        iface: false,
        bases: &["java.lang.Exception"],
    },
    StaticRow {
        name: "java.lang.LayerInstantiationException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.Math",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.ModuleLayer",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.NoClassDefFoundError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.NoSuchFieldError",
        iface: false,
        bases: &["java.lang.IncompatibleClassChangeError"],
    },
    StaticRow {
        name: "java.lang.NoSuchFieldException",
        iface: false,
        bases: &["java.lang.ReflectiveOperationException"],
    },
    StaticRow {
        name: "java.lang.NoSuchMethodError",
        iface: false,
        bases: &["java.lang.IncompatibleClassChangeError"],
    },
    StaticRow {
        name: "java.lang.NoSuchMethodException",
        iface: false,
        bases: &["java.lang.ReflectiveOperationException"],
    },
    StaticRow {
        name: "java.lang.OutOfMemoryError",
        iface: false,
        bases: &["java.lang.VirtualMachineError"],
    },
    StaticRow {
        name: "java.lang.VirtualMachineError",
        iface: false,
        bases: &["java.lang.Error"],
    },
    StaticRow {
        name: "java.lang.Override",
        iface: true,
        bases: &["java.lang.annotation.Annotation"],
    },
    StaticRow {
        name: "java.lang.Package",
        iface: false,
        bases: &[
            "java.lang.NamedPackage",
            "java.lang.reflect.AnnotatedElement",
        ],
    },
    StaticRow {
        name: "java.lang.Process",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.ProcessBuilder",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.ProcessHandle",
        iface: true,
        bases: &["java.lang.Comparable"],
    },
    StaticRow {
        name: "java.lang.Runtime",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.RuntimePermission",
        iface: false,
        bases: &["java.security.BasicPermission"],
    },
    StaticRow {
        name: "java.lang.SecurityException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.SecurityManager",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.Short",
        iface: false,
        bases: &[
            "java.lang.Number",
            "java.lang.Comparable",
            "java.lang.constant.Constable",
        ],
    },
    StaticRow {
        name: "java.lang.StackOverflowError",
        iface: false,
        bases: &["java.lang.VirtualMachineError"],
    },
    StaticRow {
        name: "java.lang.StackTraceElement",
        iface: false,
        bases: &["java.lang.Object", "java.io.Serializable"],
    },
    StaticRow {
        name: "java.lang.StackWalker",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.StrictMath",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.StringBuffer",
        iface: false,
        bases: &[
            "java.lang.AbstractStringBuilder",
            "java.lang.Appendable",
            "java.io.Serializable",
            "java.lang.Comparable",
            "java.lang.CharSequence",
        ],
    },
    StaticRow {
        name: "java.lang.StringBuilder",
        iface: false,
        bases: &[
            "java.lang.AbstractStringBuilder",
            "java.lang.Appendable",
            "java.io.Serializable",
            "java.lang.Comparable",
            "java.lang.CharSequence",
        ],
    },
    StaticRow {
        name: "java.lang.SuppressWarnings",
        iface: true,
        bases: &["java.lang.annotation.Annotation"],
    },
    StaticRow {
        name: "java.lang.System",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.Thread",
        iface: false,
        bases: &["java.lang.Object", "java.lang.Runnable"],
    },
    StaticRow {
        name: "java.lang.Thread$State",
        iface: false,
        bases: &["java.lang.Enum"],
    },
    StaticRow {
        name: "java.lang.Thread$UncaughtExceptionHandler",
        iface: true,
        bases: &[],
    },
    StaticRow {
        name: "java.lang.ThreadDeath",
        iface: false,
        bases: &["java.lang.Error"],
    },
    StaticRow {
        name: "java.lang.ThreadGroup",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.lang.Thread$UncaughtExceptionHandler",
        ],
    },
    StaticRow {
        name: "java.lang.ThreadLocal",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.lang.TypeNotPresentException",
        iface: false,
        bases: &["java.lang.RuntimeException"],
    },
    StaticRow {
        name: "java.lang.UnknownError",
        iface: false,
        bases: &["java.lang.VirtualMachineError"],
    },
    StaticRow {
        name: "java.lang.UnsatisfiedLinkError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.UnsupportedClassVersionError",
        iface: false,
        bases: &["java.lang.ClassFormatError"],
    },
    StaticRow {
        name: "java.lang.VerifyError",
        iface: false,
        bases: &["java.lang.LinkageError"],
    },
    StaticRow {
        name: "java.lang.VirtualMachineError",
        iface: false,
        bases: &["java.lang.Error"],
    },
    StaticRow {
        name: "java.lang.Void",
        iface: false,
        bases: &["java.lang.Object"],
    },
    // #74 closure: annotation superinterface (JDK 21 declaration).
    StaticRow {
        name: "java.lang.annotation.Annotation",
        iface: true,
        bases: &[],
    },
    // #74 closure: Package superclass (JDK 21 declaration).
    StaticRow {
        name: "java.lang.NamedPackage",
        iface: false,
        bases: &["java.lang.Object"],
    },
    // #74 closure: StringBuffer/Builder superclass (JDK 21 declaration).
    StaticRow {
        name: "java.lang.AbstractStringBuilder",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.lang.Appendable",
            "java.lang.CharSequence",
        ],
    },
    // #74 closure: RuntimePermission superclass (JDK 21 declaration).
    StaticRow {
        name: "java.security.BasicPermission",
        iface: false,
        bases: &["java.security.Permission"],
    },
    // #74 closure: BasicPermission superclass (JDK 21 declaration).
    StaticRow {
        name: "java.security.Permission",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.security.Guard",
            "java.io.Serializable",
        ],
    },
    // #74 closure: Permission base (JDK 21 declaration).
    StaticRow {
        name: "java.security.Guard",
        iface: true,
        bases: &[],
    },
    // #74 default import: same-package shadowing — `Compiler` in
    // `RT.java` resolves to `clojure.lang.Compiler`, never
    // `java.lang.Compiler` (JLS: single-type > package > java.lang).
    StaticRow {
        name: "clojure.lang.Compiler",
        iface: false,
        bases: &["java.lang.Object", "clojure.asm.Opcodes"],
    },
    // #74 closure: Compiler base (relocated ASM constants).
    StaticRow {
        name: "clojure.asm.Opcodes",
        iface: true,
        bases: &[],
    },
    // #205: the algorithms suite's emulated classes (JDK 21
    // declarations; bases transcribed like the oracle rows).
    StaticRow {
        name: "java.util.Random",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.util.random.RandomGenerator",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.security.SecureRandom",
        iface: false,
        bases: &["java.util.Random"],
    },
    StaticRow {
        name: "java.time.LocalDate",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.time.temporal.Temporal",
            "java.time.temporal.TemporalAdjuster",
            "java.time.chrono.ChronoLocalDate",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.time.LocalDateTime",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.time.temporal.Temporal",
            "java.time.temporal.TemporalAdjuster",
            "java.time.chrono.ChronoLocalDateTime",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.time.Instant",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.time.temporal.Temporal",
            "java.time.temporal.TemporalAdjuster",
            "java.lang.Comparable",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.time.ZoneOffset",
        iface: false,
        bases: &[
            "java.time.ZoneId",
            "java.time.temporal.TemporalAccessor",
            "java.time.temporal.TemporalAdjuster",
            "java.lang.Comparable",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.time.ZonedDateTime",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.time.temporal.Temporal",
            "java.time.chrono.ChronoZonedDateTime",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.time.DayOfWeek",
        iface: false,
        bases: &[
            "java.lang.Enum",
            "java.time.temporal.TemporalAccessor",
            "java.time.temporal.TemporalAdjuster",
        ],
    },
    StaticRow {
        name: "java.time.format.DateTimeFormatter",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.security.MessageDigest",
        iface: false,
        bases: &["java.security.MessageDigestSpi"],
    },
    StaticRow {
        name: "java.util.Arrays",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.net.URI",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.net.URLEncoder",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.util.ArrayDeque",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.util.Deque",
            "java.lang.Cloneable",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.util.ArrayList",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.util.List",
            "java.util.RandomAccess",
            "java.lang.Cloneable",
            "java.io.Serializable",
        ],
    },
    StaticRow {
        name: "java.util.Collections",
        iface: false,
        bases: &["java.lang.Object"],
    },
    StaticRow {
        name: "java.util.PriorityQueue",
        iface: false,
        bases: &[
            "java.lang.Object",
            "java.util.Queue",
            "java.io.Serializable",
        ],
    },
];

/// A per-static-row memo (classes, supertype sets), by row name.
type StaticCache<T> = std::collections::HashMap<&'static str, T>;

/// A static lattice row by binary name. Indexed once (#226): every
/// `class`/`instance?` looks rows up — the value's class and each
/// supertype — and a scan of the 325 rows cost each lookup hundreds
/// of string compares. The first row of a name wins, as the scan did.
fn static_row(name: &str) -> Option<&'static StaticRow> {
    static INDEX: std::sync::OnceLock<std::collections::HashMap<&'static str, &'static StaticRow>> =
        std::sync::OnceLock::new();
    INDEX
        .get_or_init(|| STATIC_LATTICE.iter().rev().map(|r| (r.name, r)).collect())
        .get(name)
        .copied()
}

/// A static class's transitive supertype names (memoized: the static
/// lattice never changes, and static names resolve before user types,
/// so this is the set [`closure_contains`] walks — dangling names
/// included).
fn static_supers(row: &'static StaticRow) -> Arc<HashSet<&'static str>> {
    static SUPERS: std::sync::OnceLock<Mutex<StaticCache<Arc<HashSet<&'static str>>>>> =
        std::sync::OnceLock::new();
    let mut cache = SUPERS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let set = cache.entry(row.name).or_insert_with(|| {
        let mut out: HashSet<&'static str> = HashSet::new();
        let mut stack: Vec<&'static str> = row.bases.to_vec();
        while let Some(name) = stack.pop() {
            if out.insert(name) {
                if let Some(r) = static_row(name) {
                    stack.extend(r.bases.iter().copied());
                }
            }
        }
        Arc::new(out)
    });
    Arc::clone(set)
}

/// The JVM default imports (`RT.DEFAULT_IMPORTS`, #74) in upstream
/// order. Five keys are not `java.lang` (name resolution inside
/// `RT.java`): the `java.math` pair (single-type imports), `Callable`
/// (`j.u.c`, single-type import — the only `Callable` in the JDK),
/// `Compiler` (`clojure.lang`, same package wins), and
/// `ExceptionInfo` (explicit). Keys are oracle-pinned (`import.out`
/// i-defaults); every name is tabled, so seeding resolves through
/// `static_class` with no evaluator needed.
const DEFAULT_IMPORTS: &[&str] = &[
    "java.lang.AbstractMethodError",
    "java.lang.Appendable",
    "java.lang.ArithmeticException",
    "java.lang.ArrayIndexOutOfBoundsException",
    "java.lang.ArrayStoreException",
    "java.lang.AssertionError",
    "java.lang.AutoCloseable",
    "java.math.BigDecimal",
    "java.math.BigInteger",
    "java.lang.Boolean",
    "java.lang.BootstrapMethodError",
    "java.lang.Byte",
    "java.util.concurrent.Callable",
    "java.lang.CharSequence",
    "java.lang.Character",
    "java.lang.Class",
    "java.lang.ClassCastException",
    "java.lang.ClassCircularityError",
    "java.lang.ClassFormatError",
    "java.lang.ClassLoader",
    "java.lang.ClassNotFoundException",
    "java.lang.ClassValue",
    "java.lang.CloneNotSupportedException",
    "java.lang.Cloneable",
    "java.lang.Comparable",
    "clojure.lang.Compiler",
    "java.lang.Deprecated",
    "java.lang.Double",
    "java.lang.Enum",
    "java.lang.EnumConstantNotPresentException",
    "java.lang.Error",
    "java.lang.Exception",
    "java.lang.ExceptionInInitializerError",
    "java.lang.Float",
    "java.lang.IllegalAccessError",
    "java.lang.IllegalAccessException",
    "java.lang.IllegalArgumentException",
    "java.lang.IllegalCallerException",
    "java.lang.IllegalMonitorStateException",
    "java.lang.IllegalStateException",
    "java.lang.IllegalThreadStateException",
    "java.lang.IncompatibleClassChangeError",
    "java.lang.IndexOutOfBoundsException",
    "java.lang.InheritableThreadLocal",
    "java.lang.InstantiationError",
    "java.lang.InstantiationException",
    "java.lang.Integer",
    "java.lang.InternalError",
    "java.lang.InterruptedException",
    "java.lang.Iterable",
    "java.lang.LayerInstantiationException",
    "java.lang.LinkageError",
    "java.lang.Long",
    "java.lang.Math",
    "java.lang.ModuleLayer",
    "java.lang.NegativeArraySizeException",
    "java.lang.NoClassDefFoundError",
    "java.lang.NoSuchFieldError",
    "java.lang.NoSuchFieldException",
    "java.lang.NoSuchMethodError",
    "java.lang.NoSuchMethodException",
    "java.lang.NullPointerException",
    "java.lang.Number",
    "java.lang.NumberFormatException",
    "java.lang.Object",
    "java.lang.OutOfMemoryError",
    "java.lang.Override",
    "java.lang.Package",
    "java.lang.Process",
    "java.lang.ProcessBuilder",
    "java.lang.ProcessHandle",
    "java.lang.Readable",
    "java.lang.ReflectiveOperationException",
    "java.lang.Runnable",
    "java.lang.Runtime",
    "java.lang.RuntimeException",
    "java.lang.RuntimePermission",
    "java.lang.SecurityException",
    "java.lang.SecurityManager",
    "java.lang.Short",
    "java.lang.StackOverflowError",
    "java.lang.StackTraceElement",
    "java.lang.StackWalker",
    "java.lang.StrictMath",
    "java.lang.String",
    "java.lang.StringBuffer",
    "java.lang.StringBuilder",
    "java.lang.StringIndexOutOfBoundsException",
    "java.lang.SuppressWarnings",
    "java.lang.System",
    "java.lang.Thread",
    "java.lang.Thread$State",
    "java.lang.Thread$UncaughtExceptionHandler",
    "java.lang.ThreadDeath",
    "java.lang.ThreadGroup",
    "java.lang.ThreadLocal",
    "java.lang.Throwable",
    "java.lang.TypeNotPresentException",
    "java.lang.UnknownError",
    "java.lang.UnsatisfiedLinkError",
    "java.lang.UnsupportedClassVersionError",
    "java.lang.UnsupportedOperationException",
    "java.lang.VerifyError",
    "java.lang.VirtualMachineError",
    "java.lang.Void",
    "clojure.lang.ExceptionInfo",
];

/// The default imports as `(simple-name, class)` pairs for
/// namespace seeding. Every name resolves: a missing row is a
/// lattice bug and fails loud here rather than seeding a
/// divergent fallback.
/// [`default_imports`] as a namespace's starting imports table:
/// built once per process and shared (namespaces copy it on their
/// first import change), so namespace creation skips ~170 class
/// lookups.
pub(crate) fn default_import_map() -> Arc<crate::env::FxHashMap<String, Arc<ClassVal>>> {
    static MAP: std::sync::LazyLock<Arc<crate::env::FxHashMap<String, Arc<ClassVal>>>> =
        std::sync::LazyLock::new(|| Arc::new(default_imports().collect()));
    Arc::clone(&MAP)
}

pub(crate) fn default_imports() -> impl Iterator<Item = (String, Arc<ClassVal>)> {
    DEFAULT_IMPORTS.iter().map(|fqn| {
        let simple = fqn.rsplit('.').next().unwrap_or(fqn).to_owned();
        let class = static_class(fqn).expect("default import is tabled");
        (simple, class)
    })
}

/// A static lattice row as a class value.
fn static_class(name: &str) -> Option<Arc<ClassVal>> {
    // Built once per row (#226): a fresh class per lookup allocated
    // its name and every supertype name (`class`/`instance?` hot).
    static CLASSES: std::sync::OnceLock<Mutex<StaticCache<Arc<ClassVal>>>> =
        std::sync::OnceLock::new();
    let r = static_row(name)?;
    let mut cache = CLASSES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let class = cache.entry(r.name).or_insert_with(|| {
        Arc::new(ClassVal::new(
            Arc::from(r.name),
            r.iface,
            r.bases.iter().map(|b| Arc::from(*b)).collect(),
        ))
    });
    Some(Arc::clone(class))
}

/// A static field read in value position (`Long/MAX_VALUE`):
/// the constant for a known `(binary class name, field)` pair.
/// Boxed numbers surface as the port's unboxed values (`Int`
/// for every integral bound, `Double` for floating ones —
/// `Float` bounds convert through `f32` exactly as the JVM's
/// widening does); `Character` bounds are `Char`s.
pub(crate) fn static_field_value(class: &str, field: &str) -> Option<Value> {
    // Two-level dispatch (complexity ratchet): one arm per class,
    // then a per-class field match below; the tail classes live
    // in the helper so neither match breaches the budget.
    match class {
        "java.lang.Long" => static_field_long(field),
        "java.lang.Integer" => static_field_integer(field),
        "java.lang.Short" => static_field_short(field),
        "java.lang.Byte" => static_field_byte(field),
        "java.lang.Double" => static_field_double(field),
        "java.lang.Float" => static_field_float(field),
        "java.lang.Character" => static_field_character(field),
        "java.math.RoundingMode" => static_field_rounding(field),
        "java.lang.Boolean" => static_field_boolean(field),
        "java.lang.Math" => static_field_math(field),
        _ => static_field_value_rest(class, field),
    }
}

/// The tail half of [`static_field_value`]'s class dispatch.
fn static_field_value_rest(class: &str, field: &str) -> Option<Value> {
    match class {
        "java.util.Locale" => static_field_locale(field),
        "clojure.lang.PersistentQueue" => static_field_queue(field),
        "java.math.BigInteger" => static_field_bigint(field),
        "java.time.ZoneOffset" => crate::emutime::static_field_zoneoffset(field),
        _ => None,
    }
}

/// `BigInteger` constants (#205): fresh values per read (like
/// every other field — `=` holds, `identical?` does not).
fn static_field_bigint(field: &str) -> Option<Value> {
    let n = match field {
        "ZERO" => num_bigint::BigInt::from(0),
        "ONE" => num_bigint::BigInt::from(1),
        "TWO" => num_bigint::BigInt::from(2),
        "TEN" => num_bigint::BigInt::from(10),
        _ => return None,
    };
    Some(bigint_value(n))
}

fn static_field_rounding(field: &str) -> Option<Value> {
    // #205: all eight enum constants; unknown names miss.
    RoundingMode::parse(field).map(RoundingModeVal::value)
}

fn static_field_long(field: &str) -> Option<Value> {
    match field {
        "MIN_VALUE" => Some(Value::Int(i64::MIN)),
        "MAX_VALUE" => Some(Value::Int(i64::MAX)),
        _ => None,
    }
}

fn static_field_integer(field: &str) -> Option<Value> {
    match field {
        "MIN_VALUE" => Some(Value::Int(i64::from(i32::MIN))),
        "MAX_VALUE" => Some(Value::Int(i64::from(i32::MAX))),
        _ => None,
    }
}

fn static_field_short(field: &str) -> Option<Value> {
    match field {
        "MIN_VALUE" => Some(Value::Int(i64::from(i16::MIN))),
        "MAX_VALUE" => Some(Value::Int(i64::from(i16::MAX))),
        _ => None,
    }
}

fn static_field_byte(field: &str) -> Option<Value> {
    match field {
        "MIN_VALUE" => Some(Value::Int(i64::from(i8::MIN))),
        "MAX_VALUE" => Some(Value::Int(i64::from(i8::MAX))),
        _ => None,
    }
}

fn static_field_double(field: &str) -> Option<Value> {
    match field {
        // JDK `Double/MIN_VALUE` is the smallest positive *subnormal*
        // (4.9e-324), not `MIN_NORMAL` (`f64::MIN_POSITIVE`, 2.2e-308).
        "MIN_VALUE" => Some(Value::Double(f64::from_bits(1))),
        "MAX_VALUE" => Some(Value::Double(f64::MAX)),
        "POSITIVE_INFINITY" => Some(Value::Double(f64::INFINITY)),
        "NEGATIVE_INFINITY" => Some(Value::Double(f64::NEG_INFINITY)),
        "NaN" => Some(Value::Double(f64::NAN)),
        // The `double` class (#205: `(make-array Double/TYPE …)` —
        // the name feeds `class_designator`'s primitive table).
        "TYPE" => Some(Value::Class(Arc::new(ClassVal::new(
            Arc::from("double"),
            false,
            Arc::from(vec![]),
        )))),
        _ => None,
    }
}

fn static_field_float(field: &str) -> Option<Value> {
    match field {
        // JDK `Float/MIN_VALUE` is 1.4e-45 (subnormal), not MIN_NORMAL.
        "MIN_VALUE" => Some(Value::Double(f64::from(f32::from_bits(1)))),
        "MAX_VALUE" => Some(Value::Double(f64::from(f32::MAX))),
        "POSITIVE_INFINITY" => Some(Value::Double(f64::INFINITY)),
        "NEGATIVE_INFINITY" => Some(Value::Double(f64::NEG_INFINITY)),
        "NaN" => Some(Value::Double(f64::NAN)),
        _ => None,
    }
}

fn static_field_character(field: &str) -> Option<Value> {
    match field {
        "MIN_VALUE" => Some(Value::Char('\u{0}')),
        "MAX_VALUE" => Some(Value::Char('\u{FFFF}')),
        _ => None,
    }
}

fn static_field_boolean(field: &str) -> Option<Value> {
    match field {
        "TRUE" => Some(Value::Bool(true)),
        "FALSE" => Some(Value::Bool(false)),
        _ => None,
    }
}

fn static_field_math(field: &str) -> Option<Value> {
    match field {
        "PI" => Some(Value::Double(std::f64::consts::PI)),
        "E" => Some(Value::Double(std::f64::consts::E)),
        _ => None,
    }
}

fn static_field_locale(field: &str) -> Option<Value> {
    match field {
        "US" => Some(Value::Extension(Arc::new(LocaleHandle::us()))),
        _ => None,
    }
}

fn static_field_queue(field: &str) -> Option<Value> {
    match field {
        "EMPTY" => Some(crate::coll::empty_queue_value()),
        _ => None,
    }
}

/// A class value from parts (the dynamic rules).
fn dynamic_class(name: &str, iface: bool, bases: &[&str]) -> Arc<ClassVal> {
    Arc::new(ClassVal::new(
        Arc::from(name),
        iface,
        bases.iter().map(|b| Arc::from(*b)).collect(),
    ))
}

/// A static row, falling back to a bare `Object` child for names
/// with no row and no rule (totality: every name the port reports is
/// tabled, so the fallback only fires for foreign extensions —
/// unit-tested directly).
fn static_or_fallback(name: &str) -> Arc<ClassVal> {
    static_class(name).unwrap_or_else(|| dynamic_class(name, false, &["java.lang.Object"]))
}

/// The record implicits, in bases order (the JVM adds all eight to
/// every record class).
const RECORD_IMPLICITS: &[&str] = &[
    "clojure.lang.IRecord",
    "clojure.lang.IHashEq",
    "clojure.lang.IObj",
    "clojure.lang.ILookup",
    "clojure.lang.IKeywordLookup",
    "clojure.lang.IPersistentMap",
    "java.util.Map",
    "java.io.Serializable",
];

/// The deftype implicits: `IType` alone (deftypes carry no `IObj` —
/// they reject `with-meta`, like the JVM).
const TYPE_IMPLICITS: &[&str] = &["clojure.lang.IType"];

/// The kind implicits of the HostDef rule (interfaces and gen-class
/// carry none; structs never reach here — the carve-outs resolve
/// first — and share the empty arm to stay total).
fn implicit_bases(kind: HostKind) -> &'static [&'static str] {
    match kind {
        HostKind::Record => RECORD_IMPLICITS,
        HostKind::Type => TYPE_IMPLICITS,
        _ => &[],
    }
}

/// The HostDef rule: `[Object]` (absent for interfaces) plus the
/// claimed names verbatim plus the kind implicits.
fn hostdef_class(d: &HostDef) -> Arc<ClassVal> {
    let iface = d.kind == HostKind::Interface;
    let mut bases: Vec<Arc<str>> = Vec::new();
    if !iface {
        bases.push(Arc::from("java.lang.Object"));
    }
    bases.extend(d.interfaces.iter().cloned());
    bases.extend(implicit_bases(d.kind).iter().map(|n| Arc::from(*n)));
    Arc::new(ClassVal::new(d.name.clone(), iface, bases.into()))
}

/// The class value for a supertype name: a static hit, else the
/// HostDef rule by qualified name (struct names carve to
/// `PersistentStructMap$Def`), else `None` (callers skip — the
/// redefinition leniency).
fn class_for_name(eval: &Evaluator, name: &str) -> Option<Arc<ClassVal>> {
    if let Some(class) = static_class(name) {
        return Some(class);
    }
    let def = crate::types::hostdef_by_name(eval, name)?;
    if def.kind == HostKind::Struct {
        return static_class("clojure.lang.PersistentStructMap$Def");
    }
    Some(hostdef_class(&def))
}

/// The class of a host value: the struct carve-outs (instances
/// report `PersistentStructMap`, descriptors `$Def`), else the
/// HostDef rule off the definition itself.
fn host_class_of(h: &Arc<HostVal>) -> Arc<ClassVal> {
    match h.as_ref() {
        HostVal::Def(d) if d.kind == HostKind::Struct => {
            static_or_fallback("clojure.lang.PersistentStructMap$Def")
        }
        HostVal::Inst(i) if i.def.kind == HostKind::Struct => {
            static_or_fallback("clojure.lang.PersistentStructMap")
        }
        HostVal::Def(d) => hostdef_class(d),
        HostVal::Inst(i) => hostdef_class(&i.def),
    }
}

/// The class of an extension value: the fn rule for closures (any
/// rest arity → `RestFn`, else `AFunction`) and natives
/// (unconditionally `AFunction`), both under their own display
/// names; every other kind reports a tabled name.
fn extension_class_of(e: &Arc<dyn clj_value::ExtensionValue>) -> Arc<ClassVal> {
    let any = e.as_any();
    if let Some(c) = any.downcast_ref::<Closure>() {
        let rest = c.arities.iter().any(|a| a.rest.is_some());
        let base = if rest {
            "clojure.lang.RestFn"
        } else {
            "clojure.lang.AFunction"
        };
        return dynamic_class(&e.class_name(), false, &[base]);
    }
    if any.downcast_ref::<NativeFn>().is_some() {
        return dynamic_class(&e.class_name(), false, &["clojure.lang.AFunction"]);
    }
    static_or_fallback(&e.class_name())
}

/// The class of a value (`None` for nil): classes answer
/// `java.lang.Class`, host values fold through [`host_class_of`],
/// extensions through [`extension_class_of`], and everything else
/// reports its `jvm_class_of` name (tabled).
pub(crate) fn class_of_value(v: &Value) -> Option<Arc<ClassVal>> {
    match v {
        Value::Nil => None,
        Value::Class(_) => static_class("java.lang.Class"),
        Value::HostType(h) => Some(host_class_of(h)),
        Value::Extension(e) => Some(extension_class_of(e)),
        _ => Some(static_or_fallback(&jvm_class_of(v))),
    }
}

/// The class of a (non-nil) value as a value.
fn class_value_of(v: &Value) -> Value {
    class_of_value(v).map(Value::Class).unwrap_or(Value::Nil)
}

/// Whether `target` names the class of `x` or one of its transitive
/// supertypes (the walk recurses through the registry, skips names
/// that no longer resolve, and carries a visited set).
pub(crate) fn is_instance_of(eval: &Evaluator, target: &str, x: &Value) -> bool {
    let Some(class) = class_of_value(x) else {
        return false;
    };
    if class.name() == target {
        return true;
    }
    // Built-in values have static classes: their supertypes are fixed.
    match static_row(class.name()).filter(|_| !matches!(x, Value::HostType(_))) {
        Some(row) => static_supers(row).contains(target),
        None => closure_contains(eval, class.bases(), target),
    }
}

/// Whether the transitive supertypes of `bases` contain `target`.
fn closure_contains(eval: &Evaluator, bases: &[Arc<str>], target: &str) -> bool {
    let mut visited: HashSet<Arc<str>> = HashSet::new();
    let mut stack: Vec<Arc<str>> = bases.to_vec();
    while let Some(name) = stack.pop() {
        if name.as_ref() == target {
            return true;
        }
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(shape) = class_for_name(eval, &name) {
            stack.extend(shape.bases().iter().cloned());
        }
    }
    false
}

/// The transitive supertype names of `bases` (dangling names are
/// collected here and skipped at materialization).
fn closure_set(eval: &Evaluator, bases: &[Arc<str>]) -> HashSet<Arc<str>> {
    let mut out: HashSet<Arc<str>> = HashSet::new();
    let mut stack: Vec<Arc<str>> = bases.to_vec();
    while let Some(name) = stack.pop() {
        if !out.insert(name.clone()) {
            continue;
        }
        if let Some(shape) = class_for_name(eval, &name) {
            stack.extend(shape.bases().iter().cloned());
        }
    }
    out
}

/// A class-position argument: nil is the caller's NPE, classes pass
/// through, descriptors fold through the registry (struct
/// descriptors and instances are cast failures, like any other
/// non-class).
fn class_position(
    v: &Value,
    nil_msg: &'static str,
    eval: &Evaluator,
) -> Result<Arc<ClassVal>, NativeError> {
    match v {
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            nil_msg.to_owned(),
        )),
        Value::Class(c) => Ok(c.clone()),
        Value::HostType(h) => fold_descriptor(h, v, eval),
        _ => Err(crate::core::cce_err(v, "java.lang.Class")),
    }
}

/// Fold a descriptor in class position to its registry shape (a
/// folded name with no registry entry is the cast failure, like a
/// struct descriptor or an instance).
fn fold_descriptor(
    h: &Arc<HostVal>,
    v: &Value,
    eval: &Evaluator,
) -> Result<Arc<ClassVal>, NativeError> {
    match h.as_ref() {
        HostVal::Def(d) if d.kind != HostKind::Struct => {
            class_for_name(eval, &d.name).ok_or_else(|| crate::core::cce_err(v, "java.lang.Class"))
        }
        _ => Err(crate::core::cce_err(v, "java.lang.Class")),
    }
}

/// `instance?`: the class of `x` is `c` or has it as a supertype.
fn native_instance(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/instance?", args, &[2])?;
    let class = class_position(
        &args[0],
        "Cannot invoke \"java.lang.Class.isInstance(Object)\" because \"c\" is null",
        eval,
    )?;
    Ok(Value::Bool(is_instance_of(eval, class.name(), &args[1])))
}

/// Cast `x` to `c`: nil passes through, members pass through, and
/// anything else is the short-shape miss.
fn cast_to_class(eval: &Evaluator, c: &Arc<ClassVal>, x: &Value) -> Result<Value, NativeError> {
    if matches!(x, Value::Nil) {
        return Ok(x.clone());
    }
    if is_instance_of(eval, c.name(), x) {
        Ok(x.clone())
    } else {
        Err(NativeError::cast(&jvm_class_of(x), c.name()))
    }
}

/// `cast`: like the `Class.cast` member through [`cast_to_class`].
fn native_cast(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/cast", args, &[2])?;
    let class = class_position(
        &args[0],
        "Cannot invoke \"java.lang.Class.cast(Object)\" because \"c\" is null",
        eval,
    )?;
    cast_to_class(eval, &class, &args[1])
}

/// The `:type` metadata entry, when present.
fn meta_type(v: &Value) -> Option<Value> {
    let meta = v.meta()?;
    let probe = Value::Keyword(Keyword::new(None, "type"));
    meta.get(&probe).cloned()
}

/// `type`: a truthy `:type` entry wins, else the class.
fn native_type(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/type", args, &[1])?;
    if let Some(t) = meta_type(&args[0]) {
        if t.is_truthy() {
            return Ok(t);
        }
    }
    Ok(class_value_of(&args[0]))
}

/// `class`: the class of the value, nil for nil.
fn native_class(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/class", args, &[1])?;
    Ok(class_value_of(&args[0]))
}

/// A `bases`/`supers` argument: nil is nil (not the NPE — unlike
/// `instance?`/`cast`), classes pass through, descriptors fold.
fn bases_class_arg(v: &Value, eval: &Evaluator) -> Result<Option<Arc<ClassVal>>, NativeError> {
    match v {
        Value::Nil => Ok(None),
        Value::Class(c) => Ok(Some(c.clone())),
        Value::HostType(h) => Ok(Some(fold_descriptor(h, v, eval)?)),
        _ => Err(crate::core::cce_err(v, "java.lang.Class")),
    }
}

/// `bases`: the ordered direct supertypes as a seq, nil for empty
/// ancestry (dangling names skip — the redefinition leniency).
fn native_bases(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/bases", args, &[1])?;
    let Some(class) = bases_class_arg(&args[0], eval)? else {
        return Ok(Value::Nil);
    };
    if class.bases().is_empty() {
        return Ok(Value::Nil);
    }
    let mut items = Vec::new();
    for base in class.bases() {
        if let Some(c) = class_for_name(eval, base) {
            items.push(Value::Class(c));
        }
    }
    Ok(Value::List(List::from_vec(items)))
}

/// `supers`: the transitive supertypes as a set, nil for empty
/// ancestry (dangling names skip — the redefinition leniency).
fn native_supers(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/supers", args, &[1])?;
    let Some(class) = bases_class_arg(&args[0], eval)? else {
        return Ok(Value::Nil);
    };
    let names = closure_set(eval, class.bases());
    if names.is_empty() {
        return Ok(Value::Nil);
    }
    let mut set = new_set();
    for name in &names {
        if let Some(c) = class_for_name(eval, name) {
            set = set.insert(Value::Class(c));
        }
    }
    Ok(Value::Set(set))
}

/// `class?`: whether the value is a class.
fn native_classp(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/class?", args, &[1])?;
    Ok(Value::Bool(matches!(&args[0], Value::Class(_))))
}

/// `identical?`: the JVM `==` model (fence 4a — oracle battery
/// `docs/spec/fence-4a-identity/probes.md`).
///
/// Unboxed numbers/characters follow the boxing caches, exactly
/// (JVM locals rebox per use, so outside the cache there is no
/// object to share — not even for the same binding): longs in
/// `[-128, 127]` compare by value, anything else long is never
/// identical; chars in `[\u{0}, \u{7f}]` compare by value, the
/// rest never; doubles are never identical (even same-binding
/// `NaN`). `nil`/`Bool` are canonical (value-compared).
///
/// Heap kinds compare allocation identity: strings are
/// `Arc::ptr_eq` (literals pool through the analyzer's interner,
/// runtime constructions stay fresh); symbols, uuids, instants,
/// uris, map entries, tagged literals, reader conditionals, and
/// primitive vectors compare minted ids; collections compare
/// their minted ids; extensions / host values / multimethods /
/// regexes / tower numbers compare allocations; classes compare
/// names; vars, unbounds, throwables, transients, atoms, delays,
/// futures, promises, agents, volatiles, and refs compare
/// through `=` (their equality already IS cell identity).
///
/// Tower numbers (`BigInt`/`BigInteger`/`BigDec`/`Ratio`) are
/// `Arc::ptr_eq` per kind (fence 4a-big, oracle P5):
/// `BigInt`/`BigInteger`/`Ratio` box fresh per construction,
/// `BigDec` literals box fresh while arithmetic results share
/// `ZERO_THROUGH_TEN` (scale-0 `0..=10`) through
/// `bigdec_result` — the JVM's `BigDecimal.valueOf` rule.
/// Cross-kind pairs are never identical.
pub(crate) fn identical_values(a: &Value, b: &Value) -> bool {
    match (a, b) {
        // The `Long` cache (`LongCache`, -128..127): inside, the
        // box IS the value; outside, every box is fresh.
        (Value::Int(x), Value::Int(y)) => x == y && (-128..=127).contains(x),
        // Doubles never cache (`Double.valueOf` news).
        (Value::Double(_), Value::Double(_)) => false,
        // The `Character` cache (`\u{0}`..`\u{7f}`).
        (Value::Char(x), Value::Char(y)) => x == y && (*x as u32) < 128,
        // Literals pool (analyzer interner); constructions fresh.
        (Value::Str(x), Value::Str(y)) => Arc::ptr_eq(x, y),
        (Value::Symbol(x), Value::Symbol(y)) => x.id() == y.id(),
        // Keywords intern globally (`Keyword.intern` analog).
        (Value::Keyword(x), Value::Keyword(y)) => x.ptr_eq(y),
        _ => identical_heap(a, b),
    }
}

/// `identical?` for the minted-id kinds (fence 4a): collections,
/// uuids/instants/uris, entries, tagged literals, reader
/// conditionals, and primitive vectors compare allocation ids.
fn identical_heap(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::List(x), Value::List(y)) => x.id() == y.id(),
        (Value::Vector(x), Value::Vector(y)) => x.id() == y.id(),
        (Value::Map(x), Value::Map(y)) => x.id() == y.id(),
        (Value::Set(x), Value::Set(y)) => x.id() == y.id(),
        (Value::Uuid(x), Value::Uuid(y)) => x.id() == y.id(),
        (Value::Instant(x), Value::Instant(y)) => x.id() == y.id(),
        (Value::Uri(x), Value::Uri(y)) => x.id() == y.id(),
        (Value::MapEntry { id: x, .. }, Value::MapEntry { id: y, .. }) => x == y,
        (Value::TaggedLiteral { id: x, .. }, Value::TaggedLiteral { id: y, .. }) => x == y,
        (Value::ReaderConditional { id: x, .. }, Value::ReaderConditional { id: y, .. }) => x == y,
        (Value::PrimitiveVector(x), Value::PrimitiveVector(y)) => x.id() == y.id(),
        _ => identical_alloc(a, b),
    }
}

/// `identical?` for the allocation/name kinds (fence 4a):
/// extensions / host values / multimethods / regexes / tower
/// numbers compare allocations, classes compare names, and
/// every other same-kind pair compares like `=` (vars,
/// unbounds, throwables, transients, and all cells compare
/// cell identity through `=`).
fn identical_alloc(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Extension(x), Value::Extension(y)) => Arc::ptr_eq(x, y),
        (Value::HostType(x), Value::HostType(y)) => Arc::ptr_eq(x, y),
        (Value::MultiFn(x), Value::MultiFn(y)) => Arc::ptr_eq(x, y),
        (Value::BigInt(x), Value::BigInt(y)) => Arc::ptr_eq(x, y),
        (Value::BigInteger(x), Value::BigInteger(y)) => Arc::ptr_eq(x, y),
        (Value::BigDec(x), Value::BigDec(y)) => Arc::ptr_eq(x, y),
        (Value::Ratio(x), Value::Ratio(y)) => Arc::ptr_eq(x, y),
        (Value::Class(x), Value::Class(y)) => x.name() == y.name(),
        _ => std::mem::discriminant(a) == std::mem::discriminant(b) && a == b,
    }
}

/// `identical?` (the analyzer's direct-call rewrite lowers to this
/// same var, so the two cannot drift).
fn native_identical(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/identical?", args, &[2])?;
    Ok(Value::Bool(identical_values(&args[0], &args[1])))
}

/// Whether a dotted name sits in class position (resolves as a
/// class or fails `ClassNotFoundException`): dotted without a
/// leading dot (oracle `val-fq-missing`, `val-dotdot`,
/// `v-traildot`, `v-dashdot` — leading-dot names like `.-a`/`.foo`
/// keep the unresolvable error, oracle `v-dotdash-a`,
/// `v-leaddot`).
pub(crate) fn is_class_position(name: &str) -> bool {
    name.contains('.') && !name.starts_with('.')
}

/// Resolve a class name: a static hit, else a `HostDef` of that
/// qualified name (struct names carve to
/// `PersistentStructMap$Def`), else unresolvable. Shared by
/// `import`, static calls, dotted literals, `resolve`, and
/// `Class/forName`, so all five agree on every name.
pub(crate) fn resolve_class_name(eval: &Evaluator, fqn: &str) -> Option<Arc<ClassVal>> {
    if let Some(class) = static_class(fqn) {
        return Some(class);
    }
    let def = crate::types::hostdef_by_name(eval, fqn)?;
    if def.kind == HostKind::Struct {
        return static_class("clojure.lang.PersistentStructMap$Def");
    }
    Some(hostdef_class(&def))
}

/// Whether a var root is the descriptor of the imported name itself
/// (importing a type in its defining namespace no-ops instead of
/// conflicting).
fn def_of_same_name(root: &Value, fqn: &str) -> bool {
    match root {
        Value::HostType(h) => match h.as_ref() {
            HostVal::Def(d) => d.name.as_ref() == fqn,
            HostVal::Inst(_) => false,
        },
        _ => false,
    }
}

/// `__import*`: import one fully-qualified name into the current
/// namespace, returning the class. Unknown names are
/// `ClassNotFoundException` (bare name); a var mapping (def or
/// refer) is the `Var` → `Class` cast failure, except the same-name
/// descriptor leniency (no-op success); an import of the same class
/// is a no-op, and of a different class the `already refers to`
/// state error.
fn native_import_star(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/__import*", args, &[1])?;
    let Value::Str(fqn) = &args[0] else {
        return Err(crate::core::cce_err(&args[0], "java.lang.String"));
    };
    import_class(eval, fqn)
}

/// Import `fqn` into the current namespace (split for the complexity
/// budget).
pub(crate) fn import_class(eval: &mut Evaluator, fqn: &str) -> Result<Value, NativeError> {
    let class = resolve_class_name(eval, fqn)
        .ok_or_else(|| NativeError::new(ThrowClass::ClassNotFoundException, fqn.to_owned()))?;
    let simple = fqn.rsplit('.').next().unwrap_or(fqn);
    let cur = eval.current_ns_name();
    let sym = Symbol::new(None, simple);
    // Nested `if let` (not a let-chain): MSRV 1.85 predates
    // let-chains (E0658 on the msrv leg).
    if let Some(target) = eval.registry.resolve_in(&cur, &sym) {
        if let Some(id) = eval.registry.lookup(&target) {
            let root = eval.deref_var(id, &target);
            if def_of_same_name(&root, fqn) {
                return Ok(Value::Class(class));
            }
            let var = crate::ns::var_value(id, &target);
            return Err(crate::core::cce_err(&var, "java.lang.Class"));
        }
    }
    if let Some(existing) = import_in(eval, &cur, simple) {
        if existing.name() == class.name() {
            return Ok(Value::Class(class));
        }
        return Err(NativeError::new(
            ThrowClass::IllegalStateException,
            format!(
                "{simple} already refers to: {} in namespace: {cur}",
                clj_value::class_java_debug(&existing),
            ),
        ));
    }
    eval.registry
        .with_ns_mut(&cur, |ns| ns.import_class(simple, class.clone()));
    Ok(Value::Class(class))
}

/// The emulated static surface (#120 slices 1+3): exact `(class,
/// member)` entries with their fixed arities. The analyzer
/// validates arities for known members at compile time (oracle
/// `forName-0arg`, `forName-4arg`), while unknown members on
/// resolved classes defer to the runtime [`native_static`] miss (a
/// documented phase narrowing — the JVM rejects those at compile
/// time too, oracle `static-miss-0`). Wire new statics here plus
/// an arm in [`native_static`] (and, for class values, in the
/// instance-position surface below).
pub(crate) fn static_member_arities(fqn: &str, member: &str) -> Option<&'static [usize]> {
    // Two-level dispatch (complexity ratchet): one arm per class,
    // then a per-class member match below; the tail classes live
    // in the helper so neither match breaches the budget.
    match fqn {
        "java.lang.Class" => static_arities_class(member),
        "java.lang.System" => static_arities_system(member),
        "java.lang.Character" => static_arities_character(member),
        "clojure.lang.LazilyPersistentVector" => static_arities_lpv(member),
        "java.lang.Long" => static_arities_long(member),
        "java.lang.Integer" => static_arities_integer(member),
        "java.lang.Double" => static_arities_double(member),
        "java.lang.Boolean" => static_arities_boolean(member),
        "java.lang.Float" => static_arities_float(member),
        "java.math.BigInteger" => static_arities_bigint(member),
        "java.math.BigDecimal" => static_arities_bigdec(member),
        "java.util.Arrays" => static_arities_arrays(member),
        _ => static_member_arities_rest(fqn, member),
    }
}

/// The tail half of [`static_member_arities`]' class dispatch.
fn static_member_arities_rest(fqn: &str, member: &str) -> Option<&'static [usize]> {
    match fqn {
        "java.util.Collections" => static_arities_collections(member),
        "java.security.MessageDigest" => static_arities_digest(member),
        "java.util.regex.Pattern" => static_arities_pattern(member),
        "java.io.File" => static_arities_file(member),
        "java.net.URLEncoder" => static_arities_urlencoder(member),
        "java.lang.String" => static_arities_string(member),
        "java.lang.Math" => static_arities_math(member),
        "java.time.LocalDate" => crate::emutime::static_arities_localdate(member),
        "java.time.LocalDateTime" => crate::emutime::static_arities_datetime(member),
        "java.time.Instant" => crate::emutime::static_arities_instant(member),
        "java.time.format.DateTimeFormatter" => crate::emutime::static_arities_formatter(member),
        _ => None,
    }
}

fn static_arities_class(member: &str) -> Option<&'static [usize]> {
    match member {
        "forName" => Some(&[1, 2, 3]),
        _ => None,
    }
}

fn static_arities_system(member: &str) -> Option<&'static [usize]> {
    match member {
        "getProperty" => Some(&[1, 2]),
        "lineSeparator" => Some(&[0]),
        "identityHashCode" => Some(&[1]),
        // #205: the algorithms suite measures and copies.
        "nanoTime" | "currentTimeMillis" => Some(&[0]),
        "getenv" => Some(&[1]),
        "arraycopy" => Some(&[5]),
        _ => None,
    }
}

fn static_arities_character(member: &str) -> Option<&'static [usize]> {
    match member {
        "isWhitespace" | "isLetterOrDigit" | "isLowerCase" | "isUpperCase" | "isDigit"
        | "isLetter" | "isAlphabetic" | "toUpperCase" | "toLowerCase" => Some(&[1]),
        "digit" => Some(&[2]),
        _ => None,
    }
}

fn static_arities_lpv(member: &str) -> Option<&'static [usize]> {
    match member {
        "createOwning" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_long(member: &str) -> Option<&'static [usize]> {
    match member {
        // #205: `lempel_ziv` parses binary with a radix.
        "parseLong" => Some(&[1, 2]),
        "remainderUnsigned" => Some(&[2]),
        // #205: bit-string formatting and population counts.
        "toBinaryString" | "toHexString" | "bitCount" | "numberOfTrailingZeros" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_integer(member: &str) -> Option<&'static [usize]> {
    match member {
        "parseInt" => Some(&[1, 2]),
        // #205: bit-string formatting.
        "toBinaryString" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_double(member: &str) -> Option<&'static [usize]> {
    match member {
        "parseDouble" => Some(&[1]),
        // #205: finiteness guards in finance/neural code.
        "isNaN" | "isInfinite" | "isFinite" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_boolean(member: &str) -> Option<&'static [usize]> {
    // #205: `comb_sort` boxes predicates explicitly.
    match member {
        "valueOf" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_float(member: &str) -> Option<&'static [usize]> {
    // #205: `fast_inverse_sqrt` bit-twiddles through floats.
    match member {
        "floatToIntBits" | "intBitsToFloat" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_bigint(member: &str) -> Option<&'static [usize]> {
    // #205: RSA/Diffie-Hellman lift longs into `BigInteger`.
    match member {
        "valueOf" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_bigdec(member: &str) -> Option<&'static [usize]> {
    // #205: the `(long)`/`(double)` overloads (boxed by type).
    match member {
        "valueOf" => Some(&[1]),
        _ => None,
    }
}

fn static_arities_string(member: &str) -> Option<&'static [usize]> {
    match member {
        "format" => Some(&[3]),
        _ => None,
    }
}

fn static_arities_math(member: &str) -> Option<&'static [usize]> {
    // Only `pow`/`atan2`/`hypot`/`floorDiv`/`floorMod` take 2
    // args; `random` takes none; the rest of the emulated
    // surface is 1-arg (checked at compile time, so the linear
    // scan below never runs hot).
    const TWO: &[&str] = &["pow", "atan2", "hypot", "floorDiv", "floorMod"];
    const ONE: &[&str] = &[
        "abs",
        "sqrt",
        "ceil",
        "log",
        "exp",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "cbrt",
        "floor",
        "log10",
        "round",
        "rint",
        "toRadians",
        "toDegrees",
    ];
    if TWO.contains(&member) {
        Some(&[2])
    } else if ONE.contains(&member) {
        Some(&[1])
    } else if member == "random" {
        Some(&[0])
    } else {
        None
    }
}

/// Whether a lattice name is an interface (the uniform ctor miss
/// reads `for interface` for interfaces, `for class` otherwise —
/// oracles `ctor-iface`, `ctor-abstract`). Unknown names read
/// `class` (only lattice classes and `HostDef`s reach the miss).
pub(crate) fn lattice_is_iface(fqn: &str) -> bool {
    static_row(fqn).is_some_and(|r| r.iface)
}
/// `__static*`: `(FQN member args…)` — the analyzer's static-call
/// lowering plus the instance-position consult for class values.
/// Known members dispatch; 0-arg field names read the field
/// (#205); anything else is the oracle's missing-method error
/// (oracle `static-miss-args` text, at runtime per the
/// table-gated narrowing).
fn native_static(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() < 2 {
        return Err(NativeError::arity("clojure.core/__static*", args.len()));
    }
    let Value::Str(fqn) = &args[0] else {
        return Err(crate::core::cce_err(&args[0], "java.lang.String"));
    };
    let Value::Str(member) = &args[1] else {
        return Err(crate::core::cce_err(&args[1], "java.lang.String"));
    };
    let rest = &args[2..];
    // Two-level dispatch (complexity ratchet): one arm per class,
    // then a per-class member match below (Math's 21 members ride
    // a name table); the later classes live in the mid/tail
    // helpers so no single match breaches the budget.
    match fqn.as_ref() {
        "java.lang.Class" => static_call_class(eval, member.as_ref(), rest),
        "java.lang.System" => static_call_system(member.as_ref(), rest),
        "java.lang.Character" => static_call_character(member.as_ref(), rest),
        "clojure.lang.LazilyPersistentVector" => static_call_lpv(member.as_ref(), rest),
        "java.lang.Long" => static_call_long(member.as_ref(), rest),
        "java.lang.Integer" => static_call_integer(member.as_ref(), rest),
        "java.lang.Double" => static_call_double(member.as_ref(), rest),
        "java.lang.Boolean" => static_call_boolean(member.as_ref(), rest),
        _ => native_static_mid(eval, fqn.as_ref(), member.as_ref(), rest),
    }
}

/// The middle third of [`native_static`]'s class dispatch.
fn native_static_mid(
    eval: &mut Evaluator,
    fqn: &str,
    member: &str,
    rest: &[Value],
) -> Result<Value, NativeError> {
    match fqn {
        "java.lang.Float" => static_call_float(member, rest),
        "java.math.BigInteger" => static_call_bigint(member, rest),
        "java.math.BigDecimal" => static_call_bigdec(member, rest),
        "java.util.Arrays" => static_call_arrays(member, rest),
        "java.util.Collections" => static_call_collections(member, rest),
        "java.security.MessageDigest" => static_call_digest(eval, member, rest),
        "java.util.regex.Pattern" => static_call_pattern(member, rest),
        "java.io.File" => static_call_file(eval, member, rest),
        _ => native_static_tail(eval, fqn, member, rest),
    }
}

/// The tail third of [`native_static`]'s class dispatch.
fn native_static_tail(
    eval: &mut Evaluator,
    fqn: &str,
    member: &str,
    rest: &[Value],
) -> Result<Value, NativeError> {
    match fqn {
        "java.net.URLEncoder" => static_call_urlencoder(member, rest),
        "java.lang.String" => static_call_string(eval, member, rest),
        "java.lang.Math" => static_call_math(member, rest),
        "java.time.LocalDate" => crate::emutime::static_call_localdate(member, rest),
        "java.time.LocalDateTime" => crate::emutime::static_call_datetime(member, rest),
        "java.time.Instant" => crate::emutime::static_call_instant(member, rest),
        "java.time.format.DateTimeFormatter" => crate::emutime::static_call_formatter(member, rest),
        _ => static_field_or_miss(fqn, member, rest),
    }
}

/// A static miss that is really a field read (#205): a 0-arg
/// `Class/member` call where `member` is a static field answers
/// the field (the JVM compiles `(Integer/MAX_VALUE)` to the
/// field access — the algorithms suite calls
/// `(PersistentQueue/EMPTY)`). Methods always win (this runs
/// only on the miss path); fields with args still miss.
fn static_field_or_miss(fqn: &str, member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    if rest.is_empty() {
        if let Some(value) = static_field_value(fqn, member) {
            return Ok(value);
        }
    }
    Err(crate::types::no_method(member, rest.len(), fqn))
}

fn static_call_class(
    eval: &mut Evaluator,
    member: &str,
    rest: &[Value],
) -> Result<Value, NativeError> {
    match member {
        "forName" => static_forname(eval, rest),
        _ => static_field_or_miss("java.lang.Class", member, rest),
    }
}

fn static_call_system(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "getProperty" => static_getproperty(rest),
        "lineSeparator" => static_lineseparator(rest),
        "identityHashCode" => static_identity_hashcode(rest),
        "nanoTime" => static_system_nano_time(rest),
        "currentTimeMillis" => static_system_current_time_millis(rest),
        "getenv" => static_system_getenv(rest),
        "arraycopy" => static_system_arraycopy(rest),
        _ => static_field_or_miss("java.lang.System", member, rest),
    }
}

fn static_call_character(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "isWhitespace" => static_character_is_whitespace(rest),
        "isLetterOrDigit" => static_character_is_letter_or_digit(rest),
        "digit" => static_character_digit(rest),
        "isLowerCase" => static_character_is_lower_case(rest),
        "isUpperCase" => static_character_is_upper_case(rest),
        "isDigit" => static_character_is_digit(rest),
        "isLetter" => static_character_is_letter(rest),
        "isAlphabetic" => static_character_is_alphabetic(rest),
        "toUpperCase" => static_character_to_upper_case(rest),
        "toLowerCase" => static_character_to_lower_case(rest),
        _ => static_field_or_miss("java.lang.Character", member, rest),
    }
}

fn static_call_lpv(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "createOwning" => static_lpv_create_owning(rest),
        _ => static_field_or_miss("clojure.lang.LazilyPersistentVector", member, rest),
    }
}

fn static_call_long(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "parseLong" => static_long_parse_long(rest),
        "toBinaryString" => static_long_to_binary_string(rest),
        "toHexString" => static_long_to_hex_string(rest),
        "bitCount" => static_long_bit_count(rest),
        "numberOfTrailingZeros" => static_long_number_of_trailing_zeros(rest),
        "remainderUnsigned" => static_long_remainder_unsigned(rest),
        _ => static_field_or_miss("java.lang.Long", member, rest),
    }
}

fn static_call_integer(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "parseInt" => static_integer_parse_int(rest),
        "toBinaryString" => static_integer_to_binary_string(rest),
        _ => static_field_or_miss("java.lang.Integer", member, rest),
    }
}

fn static_call_double(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "parseDouble" => static_double_parse_double(rest),
        "isNaN" => static_double_is_nan(rest),
        "isInfinite" => static_double_is_infinite(rest),
        "isFinite" => static_double_is_finite(rest),
        _ => static_field_or_miss("java.lang.Double", member, rest),
    }
}

fn static_call_boolean(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "valueOf" => static_boolean_value_of(rest),
        _ => static_field_or_miss("java.lang.Boolean", member, rest),
    }
}

fn static_call_float(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "floatToIntBits" => static_float_to_int_bits(rest),
        "intBitsToFloat" => static_int_bits_to_float(rest),
        _ => static_field_or_miss("java.lang.Float", member, rest),
    }
}

fn static_call_bigint(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "valueOf" => static_bigint_value_of(rest),
        _ => static_field_or_miss("java.math.BigInteger", member, rest),
    }
}

fn static_call_bigdec(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "valueOf" => static_bigdec_value_of(rest),
        _ => static_field_or_miss("java.math.BigDecimal", member, rest),
    }
}

/// `BigDecimal/valueOf` (#205): `(long)` is exact at scale 0
/// (through the canonical `valueOf` compact paths);
/// `(double)` is the short rendering (`Infinite`/`NaN` fail
/// like the JVM). Anything else (including nil — both overloads
/// take primitives) misses.
fn static_bigdec_value_of(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 1 {
        return Err(crate::types::no_method(
            "valueOf",
            args.len(),
            "java.math.BigDecimal",
        ));
    }
    match &args[0] {
        Value::Int(n) => Ok(bigdec_result(BigDec::from_bigint(
            num_bigint::BigInt::from(*n),
        ))),
        Value::Double(d) => BigDec::value_of_f64(*d)
            .map(bigdec_result)
            .map_err(crate::core::number_format),
        _ => Err(crate::types::no_method(
            "valueOf",
            1,
            "java.math.BigDecimal",
        )),
    }
}

fn static_arities_arrays(member: &str) -> Option<&'static [usize]> {
    match member {
        "copyOfRange" => Some(&[3]),
        _ => None,
    }
}

fn static_call_arrays(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "copyOfRange" => static_arrays_copy_of_range(rest),
        _ => static_field_or_miss("java.util.Arrays", member, rest),
    }
}

fn static_arities_collections(member: &str) -> Option<&'static [usize]> {
    match member {
        "shuffle" => Some(&[2]),
        _ => None,
    }
}

fn static_call_collections(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "shuffle" => static_collections_shuffle(rest),
        _ => static_field_or_miss("java.util.Collections", member, rest),
    }
}

fn static_arities_digest(member: &str) -> Option<&'static [usize]> {
    match member {
        "getInstance" => Some(&[1]),
        _ => None,
    }
}

fn static_call_digest(
    eval: &mut Evaluator,
    member: &str,
    rest: &[Value],
) -> Result<Value, NativeError> {
    match member {
        "getInstance" => crate::emudigest::digest_get_instance(eval, rest),
        _ => static_field_or_miss("java.security.MessageDigest", member, rest),
    }
}

/// `Collections/shuffle` (#205): the `(List, Random)` overload
/// shuffles in place (anything else misses at the arity).
fn static_collections_shuffle(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 2 {
        return Err(crate::types::no_method(
            "shuffle",
            args.len(),
            "java.util.Collections",
        ));
    }
    crate::emucoll::shuffle_list(&args[0], &args[1])
}

fn static_arities_urlencoder(member: &str) -> Option<&'static [usize]> {
    match member {
        "encode" => Some(&[2]),
        _ => None,
    }
}

fn static_arities_pattern(member: &str) -> Option<&'static [usize]> {
    match member {
        "quote" => Some(&[1]),
        _ => None,
    }
}

fn static_call_pattern(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "quote" => static_pattern_quote(rest),
        _ => static_field_or_miss("java.util.regex.Pattern", member, rest),
    }
}

fn static_arities_file(member: &str) -> Option<&'static [usize]> {
    match member {
        "createTempFile" => Some(&[2]),
        _ => None,
    }
}

fn static_call_file(
    eval: &mut Evaluator,
    member: &str,
    rest: &[Value],
) -> Result<Value, NativeError> {
    match member {
        "createTempFile" => static_file_create_temp_file(eval, rest),
        _ => static_field_or_miss("java.io.File", member, rest),
    }
}

/// `File/createTempFile` (#205): the `(prefix, suffix)` overload —
/// a uniquely-named empty file in the system temp dir (nil
/// prefixes NPE, short prefixes are `Prefix string too short`,
/// nil suffixes default to `.tmp` — all the JDK shapes).
fn static_file_create_temp_file(
    eval: &mut Evaluator,
    args: &[Value],
) -> Result<Value, NativeError> {
    if args.len() != 2 {
        return Err(crate::types::no_method(
            "createTempFile",
            args.len(),
            "java.io.File",
        ));
    }
    let prefix = match str_text_arg(&args[0])? {
        Some(s) => s,
        None => {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"java.io.File.createTempFile(String, String)\" because \"prefix\" is null"
                    .to_owned(),
            ));
        }
    };
    if prefix.chars().count() < 3 {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "Prefix string too short".to_owned(),
        ));
    }
    let suffix = str_text_arg(&args[1])?.unwrap_or(".tmp");
    let name = format!("{prefix}{}{suffix}", temp_unique());
    let path = std::env::temp_dir().join(&name);
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| {
            NativeError::new(ThrowClass::IOException, format!("{}: {e}", path.display()))
        })?;
    Ok(crate::jdkio::file(eval, &path.to_string_lossy()))
}

/// A temp-file uniqueness stamp (process + nanos + counter —
/// collisions retry nowhere: one attempt, like `File.createTempFile`
/// semantics minus the JDK's internal retry loop).
fn temp_unique() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{}{seq}{nanos}", std::process::id())
}

/// `Pattern/quote` (#205): `\Q...\E` with inner `\E`s escaped
/// (nil NPEs on the parameter, like the JDK).
fn static_pattern_quote(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 1 {
        return Err(crate::types::no_method(
            "quote",
            args.len(),
            "java.util.regex.Pattern",
        ));
    }
    match str_text_arg(&args[0])? {
        Some(s) => Ok(Value::Str(Arc::from(
            format!("\\Q{}\\E", s.replace("\\E", "\\E\\\\E\\Q")).as_str(),
        ))),
        None => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.util.regex.Pattern.quote(String)\" because \"s\" is null"
                .to_owned(),
        )),
    }
}

fn static_call_urlencoder(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    match member {
        "encode" => static_urlencoder_encode(rest),
        _ => static_field_or_miss("java.net.URLEncoder", member, rest),
    }
}

fn static_call_string(
    eval: &mut Evaluator,
    member: &str,
    rest: &[Value],
) -> Result<Value, NativeError> {
    match member {
        "format" => static_string_format(eval, rest),
        _ => static_field_or_miss("java.lang.String", member, rest),
    }
}

/// One row of the `java.lang.Math` static dispatch table.
type MathStatic = (&'static str, fn(&[Value]) -> Result<Value, NativeError>);

fn static_call_math(member: &str, rest: &[Value]) -> Result<Value, NativeError> {
    const TABLE: &[MathStatic] = &[
        ("abs", static_math_abs),
        ("sqrt", static_math_sqrt),
        ("ceil", static_math_ceil),
        ("log", static_math_log),
        ("exp", static_math_exp),
        ("pow", static_math_pow),
        ("sin", static_math_sin),
        ("cos", static_math_cos),
        ("tan", static_math_tan),
        ("asin", static_math_asin),
        ("acos", static_math_acos),
        ("atan", static_math_atan),
        ("sinh", static_math_sinh),
        ("cosh", static_math_cosh),
        ("tanh", static_math_tanh),
        ("atan2", static_math_atan2),
        ("hypot", static_math_hypot),
        ("cbrt", static_math_cbrt),
        ("floor", static_math_floor),
        ("log10", static_math_log10),
        ("round", static_math_round),
        ("rint", static_math_rint),
        ("toRadians", static_math_to_radians),
        ("toDegrees", static_math_to_degrees),
        ("floorDiv", static_math_floor_div),
        ("floorMod", static_math_floor_mod),
        ("random", static_math_random),
    ];
    TABLE
        .iter()
        .find(|(name, _)| *name == member)
        .map(|(_, f)| f(rest))
        .unwrap_or_else(|| static_field_or_miss("java.lang.Math", member, rest))
}

/// `Class/forName` over the emulated lattice. The 1- and 3-arg
/// overloads resolve (the 3-arg `initialize` flag is unobservable
/// without a runtime and the loader narrows to nil-only — both
/// still cast); the 2-arg `(Module, String)` overload can never
/// resolve port-natively (no `Module` values), so it casts and
/// NPEs exactly as the JVM does for every port-representable
/// input. Other arities miss (reachable only through instance
/// position or direct `__static*` calls — static syntax
/// arity-checks at compile time).
fn static_forname(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => forname_resolve(eval, forname_name_str(&args[0])?).map(Value::Class),
        2 => forname_two(&args[0], &args[1]),
        3 => forname_three(eval, &args[0], &args[1], &args[2]),
        n => Err(crate::types::no_method("forName", n, "java.lang.Class")),
    }
}

/// The 3-arg overload: the JVM coerces static-call args
/// left-to-right (name checkcast, `initialize` unbox, loader
/// checkcast — oracle `o3-init-unbox-first`), so the unbox NPEs
/// before a bad loader CCEs. Name/init casts stay nil-permissive
/// (oracle `o3-name-first`, `o3-init-first`,
/// `o3-missing-badinit`, `o3-nilname-badinit`); then the name
/// deref NPEs, and only then does the name resolve.
fn forname_three(
    eval: &mut Evaluator,
    name: &Value,
    init: &Value,
    loader: &Value,
) -> Result<Value, NativeError> {
    let s = forname_name_str(name)?;
    if !matches!(init, Value::Nil | Value::Bool(_)) {
        return Err(crate::core::cce_err(init, "java.lang.Boolean"));
    }
    if matches!(init, Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.lang.Boolean.booleanValue()\" because \"initialize\" is null"
                .to_owned(),
        ));
    }
    if !matches!(loader, Value::Nil) {
        return Err(crate::core::cce_err(loader, "java.lang.ClassLoader"));
    }
    // The resolve NPEs a nil name after the unbox (oracle
    // `o3-nilnilnil`).
    forname_resolve(eval, s).map(Value::Class)
}

/// The 2-arg `(Module, String)` overload: the module casts (nil
/// passes — oracle `forName-2arg-nilmod`, `forName-nn`), the name
/// casts (oracle `forName-2arg-nil-bad`), then the null module
/// NPEs with the empty message (oracle `forName-2arg-nilmod`,
/// `forName-nn`, `forName-order` for the cast order).
fn forname_two(module: &Value, name: &Value) -> Result<Value, NativeError> {
    if !matches!(module, Value::Nil) {
        return Err(crate::core::cce_err(module, "java.lang.Module"));
    }
    if !matches!(name, Value::Nil | Value::Str(_)) {
        return Err(crate::core::cce_err(name, "java.lang.String"));
    }
    Err(NativeError::new(
        ThrowClass::NullPointerException,
        String::new(),
    ))
}

/// Cast a `forName` name to `String` (nil passes — the NPE fires at
/// deref, oracle `o3-nilname-badinit`; anything else is the
/// bootstrap cast failure, oracle `forName-num`).
fn forname_name_str(name: &Value) -> Result<Option<Arc<str>>, NativeError> {
    match name {
        Value::Nil => Ok(None),
        Value::Str(s) => Ok(Some(Arc::clone(s))),
        _ => Err(crate::core::cce_err(name, "java.lang.String")),
    }
}

/// The nil-name NPE (port-pinned text in house style: the JDK
/// message comes from `Class.forName0` internals and varies by
/// JDK — JDK 27 says `Cannot invoke "String.length()" because
/// "str" is null`, oracle `forName-nil`).
fn forname_name_npe() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"java.lang.Class.forName(String)\" because \"name\" is null".to_owned(),
    )
}

/// Resolve a cast-checked name: the shared lattice (so `forName`
/// and `import` agree on every name, including the struct carve
/// and `HostDef` rows — oracle `deftype-forName`), else
/// `ClassNotFoundException` (oracle `forName-missing`,
/// `forName-int`, `o1-empty`). Array descriptors (`[`-led names)
/// never resolve here (slice 2: array class semantics, oracle
/// `forName-arr*`) and error exactly like any other unresolvable
/// name.
fn forname_resolve(eval: &Evaluator, name: Option<Arc<str>>) -> Result<Arc<ClassVal>, NativeError> {
    match name {
        None => Err(forname_name_npe()),
        Some(s) => resolve_class_name(eval, &s)
            .ok_or_else(|| NativeError::new(ThrowClass::ClassNotFoundException, s.to_string())),
    }
}

/// The platform line separator (JDK `System.java`: `"\n"` on UNIX,
/// `"\r\n"` on Windows — the `line.separator` property, which the
/// port never rebinds, so `lineSeparator` and the property agree
/// exactly, like upstream `core.clj` `system-newline`).
#[cfg(windows)]
const LINE_SEP: &str = "\r\n";
#[cfg(not(windows))]
const LINE_SEP: &str = "\n";

/// The platform path separator (`":"` on UNIX, `";"` on Windows —
/// the `path.separator` property; `std` exposes no constant, so the
/// `cfg` pair pins it per platform).
#[cfg(windows)]
const PATH_SEP: &str = ";";
#[cfg(not(windows))]
const PATH_SEP: &str = ":";

/// The emulated `os.name` (JDK `SystemProps` reads the kernel's OS
/// name; the port maps `std::env::consts::OS` to the JVM spellings).
/// `Windows` reads the family only, where the JVM reports the full
/// marketing version such as `Windows 10` — a documented narrowing
/// (upstream `process.clj` branches only on the `Windows` prefix).
#[cfg(target_os = "linux")]
const OS_NAME: &str = "Linux";
#[cfg(target_os = "macos")]
const OS_NAME: &str = "Mac OS X";
#[cfg(target_os = "windows")]
const OS_NAME: &str = "Windows";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const OS_NAME: &str = std::env::consts::OS;

/// The emulated `os.arch` (JDK `SystemProps` reads the kernel's
/// arch; `x86_64` takes the JVM spelling `amd64` — `aarch64`
/// already matches, and anything else reports the Rust arch name).
#[cfg(target_arch = "x86_64")]
const OS_ARCH: &str = "amd64";
#[cfg(target_arch = "x86")]
const OS_ARCH: &str = "x86";
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
const OS_ARCH: &str = std::env::consts::ARCH;

/// The well-known system properties (#120 slice 3): the keys the
/// port answers exactly and port-natively (JDK `SystemProps`
/// standard keys). Everything else ousts precisely — `None` here,
/// hence nil (1-arg) or the default (2-arg), exactly like the
/// JVM's unset keys (`System.java`: `props.getProperty` answers
/// null / the default). Ousted on purpose: `os.version` (no `std`
/// kernel-version source), `user.*` (dynamic identity — the port
/// pins constants only), `java.*` (there is no JVM to version),
/// and every `clojure.*` key (unset unless `-D`-supplied, so the
/// oust feeds upstream's own defaults — `main.clj`
/// `clojure.compile.path` → `"classes"`, `clojure.main.report` →
/// `"file"`, `basis` `clojure.basis` → nil).
fn system_property(key: &str) -> Option<&'static str> {
    match key {
        "line.separator" => Some(LINE_SEP),
        "file.separator" => Some(std::path::MAIN_SEPARATOR_STR),
        "path.separator" => Some(PATH_SEP),
        "file.encoding" => Some("UTF-8"),
        "os.name" => Some(OS_NAME),
        "os.arch" => Some(OS_ARCH),
        _ => None,
    }
}

/// Cast a `getProperty` key to `String` (nil passes — `checkKey`
/// NPEs it with the JDK text; anything else is the bootstrap cast
/// failure, mirroring [`forname_name_str`]).
fn property_key_str(key: &Value) -> Result<Option<Arc<str>>, NativeError> {
    match key {
        Value::Nil => Ok(None),
        Value::Str(s) => Ok(Some(Arc::clone(s))),
        _ => Err(crate::core::cce_err(key, "java.lang.String")),
    }
}

/// Cast a `getProperty` default to `String` (nil passes — the JDK
/// returns it as-is for missing keys; anything else is the
/// bootstrap cast failure).
fn property_def_str(def: &Value) -> Result<Value, NativeError> {
    match def {
        Value::Nil => Ok(Value::Nil),
        Value::Str(s) => Ok(Value::Str(Arc::clone(s))),
        _ => Err(crate::core::cce_err(def, "java.lang.String")),
    }
}

/// JDK `System.checkKey`, verbatim: nil keys NPE (`"key can't be
/// null"`), empty keys IAE (`"key can't be empty"`).
fn check_property_key(key: Option<&str>) -> Result<&str, NativeError> {
    match key {
        None => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "key can't be null".to_owned(),
        )),
        Some("") => Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "key can't be empty".to_owned(),
        )),
        Some(k) => Ok(k),
    }
}

/// `System/getProperty` over the well-known table. Static-call args
/// coerce left-to-right (the slice-2 `forName` order: key cast,
/// default cast, both nil-permissive), then `checkKey` fires, then
/// the lookup — so a bad default CCEs before a nil key NPEs. Other
/// arities miss (reachable only through instance position or direct
/// `__static*` calls — static syntax arity-checks at compile time).
fn static_getproperty(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => {
            let key = property_key_str(&args[0])?;
            let key = check_property_key(key.as_deref())?;
            Ok(system_property(key).map_or(Value::Nil, |v| Value::Str(Arc::from(v))))
        }
        2 => {
            let key = property_key_str(&args[0])?;
            let def = property_def_str(&args[1])?;
            let key = check_property_key(key.as_deref())?;
            Ok(system_property(key).map_or(def, |v| Value::Str(Arc::from(v))))
        }
        n => Err(crate::types::no_method(
            "getProperty",
            n,
            "java.lang.System",
        )),
    }
}

/// `System/lineSeparator`: the `line.separator` property (JDK
/// `System.java` is `return lineSeparator` — the two agree by
/// construction here too). Other arities miss (compile-time for
/// static syntax, runtime through instance position and direct
/// `__static*` calls).
fn static_lineseparator(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        0 => Ok(Value::Str(Arc::from(LINE_SEP))),
        n => Err(crate::types::no_method(
            "lineSeparator",
            n,
            "java.lang.System",
        )),
    }
}

/// `System/identityHashCode`: nil is 0 (JDK javadoc: "The hash code
/// for the null reference is zero"); anything else hashes its
/// allocation identity where the port has one (fence-4a ids and
/// `Arc` allocations) and its content where identity is the value
/// (cached/pooled kinds). `identical?` values always agree, and
/// distinct live allocations (all but) always differ. Upstream
/// consumes it for the `#object[`/`#<` print slots
/// (`core_print.clj`, `pprint/dispatch.clj`). Other arities miss
/// (compile-time for static syntax, runtime through instance
/// position and direct `__static*` calls).
fn static_identity_hashcode(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Int(i64::from(identity_hash(&args[0])))),
        n => Err(crate::types::no_method(
            "identityHashCode",
            n,
            "java.lang.System",
        )),
    }
}

/// `Character/isWhitespace` (issue #201 S6): the `(char)` and
/// `(int)` overloads. Java's set is Rust's `White_Space` minus
/// NEL (U+0085 — `White_Space` but not Java whitespace, JVM
/// oracle) and the non-breaking spaces (U+00A0, U+2007,
/// U+202F), plus the controls U+001C–U+001F. Out-of-range ints
/// and lone surrogates read false; anything else is the
/// `Character` cast failure.
fn static_character_is_whitespace(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(match &args[0] {
            Value::Char(c) => java_is_whitespace(*c),
            Value::Int(i) => u32::try_from(*i)
                .ok()
                .and_then(char::from_u32)
                .is_some_and(java_is_whitespace),
            other => {
                return Err(crate::core::cce_err(other, "java.lang.Character"));
            }
        })),
        n => Err(crate::types::no_method(
            "isWhitespace",
            n,
            "java.lang.Character",
        )),
    }
}

/// `Character/digit` (2-arg): the digit value of `ch` in
/// `radix`, or -1 (non-digits read -1, out-of-range radixes
/// read -1 — oracle). `Char` and `Int` (code point) spellings
/// both apply; only ASCII alphanumerics convert (a documented
/// narrowing — full `CharacterData` tables stay out).
fn static_character_digit(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 2 {
        return Err(crate::types::no_method(
            "digit",
            args.len(),
            "java.lang.Character",
        ));
    }
    let ch = match &args[0] {
        Value::Char(c) => Some(*c),
        Value::Int(i) => u32::try_from(*i).ok().and_then(char::from_u32),
        other => return Err(crate::core::cce_err(other, "java.lang.Character")),
    };
    let radix = str_int_arg(&args[1])?;
    let digit = match (ch, u32::try_from(radix).ok()) {
        (Some(c), Some(r)) if (2..=36).contains(&r) => c.to_digit(r).map(|d| d as i64),
        _ => None,
    };
    Ok(Value::Int(digit.unwrap_or(-1)))
}

/// The Java whitespace predicate (see `static_character_is_whitespace`).
pub(crate) fn java_is_whitespace(c: char) -> bool {
    matches!(c, '\u{1C}'..='\u{1F}')
        || (c.is_whitespace() && !matches!(c, '\u{85}' | '\u{A0}' | '\u{2007}' | '\u{202F}'))
}

/// `LazilyPersistentVector/createOwning` (issue #201 S2): upstream
/// wraps a `String[]` into an owning vector. Our `Pattern/.split`
/// already returns a `Value::Vector`, so vectors pass through;
/// arrays snapshot into an owned vector (`(class ...)` reads
/// `PersistentVector`, like the JVM — oracle).
fn static_lpv_create_owning(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Vector(_) => Ok(args[0].clone()),
            Value::Array(a) => Ok(Value::Vector(Vector::from_vec(a.snapshot()))),
            _ => Err(crate::types::no_method(
                "createOwning",
                1,
                "clojure.lang.LazilyPersistentVector",
            )),
        },
        n => Err(crate::types::no_method(
            "createOwning",
            n,
            "clojure.lang.LazilyPersistentVector",
        )),
    }
}

/// The `NumberFormatException` for a bad parse (the JDK's
/// `For input string` text; nil parses as the literal `null`,
/// like `Long.parseLong((String) null)`).
fn parse_nfe(text: &str) -> NativeError {
    NativeError::new(
        ThrowClass::NumberFormatException,
        format!("For input string: \"{text}\""),
    )
}

/// `Long/parseLong` (issue #201 S12): the shared `parse_long_str`
/// body (sign + digits + i64 range, like `parse-long`), throwing
/// instead of answering nil. #205 adds the `(String, int)`
/// radix overload (`lempel_ziv` parses binary).
fn static_long_parse_long(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Str(s) => crate::core::parse_long_str(s)
                .map(Value::Int)
                .ok_or_else(|| parse_nfe(s)),
            Value::Nil => Err(parse_nfe("null")),
            other => Err(crate::core::cce_err(other, "java.lang.String")),
        },
        2 => {
            let Value::Str(s) = &args[0] else {
                if matches!(&args[0], Value::Nil) {
                    return Err(parse_nfe("null"));
                }
                return Err(crate::core::cce_err(&args[0], "java.lang.String"));
            };
            let radix = str_int_arg(&args[1])?;
            parse_long_radix(s, radix)
                .map(Value::Int)
                .ok_or_else(|| parse_radix_nfe(s, radix))
        }
        n => Err(crate::types::no_method("parseLong", n, "java.lang.Long")),
    }
}

/// Parse `text` in `radix` (the JDK's negative-accumulation
/// algorithm, so `MIN_VALUE` parses): `None` on empty input,
/// lone signs, bad digits, overflow, and out-of-range radixes
/// (the caller picks the NFE text).
fn parse_long_radix(text: &str, radix: i32) -> Option<i64> {
    if !(2..=36).contains(&radix) {
        return None;
    }
    let radix = i64::from(radix);
    let mut chars = text.chars();
    let first = chars.next()?;
    let negative = first == '-';
    let rest = if first == '-' || first == '+' {
        chars.as_str()
    } else {
        text
    };
    if rest.is_empty() {
        return None;
    }
    let limit = if negative { i64::MIN } else { -i64::MAX };
    let multmin = limit / radix;
    let mut result: i64 = 0;
    for c in rest.chars() {
        let d = i64::from(c.to_digit(radix as u32)?);
        if result < multmin {
            return None;
        }
        result *= radix;
        if result < limit + d {
            return None;
        }
        result -= d;
    }
    if negative {
        Some(result)
    } else {
        result.checked_neg()
    }
}

/// The radix-parse NFE: out-of-range radixes name the radix
/// (verbatim JDK texts), everything else names the input.
fn parse_radix_nfe(text: &str, radix: i32) -> NativeError {
    if radix < 2 {
        NativeError::new(
            ThrowClass::NumberFormatException,
            format!("radix {radix} less than Character.MIN_RADIX"),
        )
    } else if radix > 36 {
        NativeError::new(
            ThrowClass::NumberFormatException,
            format!("radix {radix} greater than Character.MAX_RADIX"),
        )
    } else {
        parse_nfe(text)
    }
}

/// The boxArg `long` coercion (the `str_int_arg` sibling:
/// every `Number` truncates, nil NPEs on the primitive, the
/// rest miss the param).
pub(crate) fn long_arg(v: &Value) -> Result<i64, NativeError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Double(d) => Ok(*d as i64),
        Value::BigInt(n) | Value::BigInteger(n) => Ok(crate::core::bigint_long_value(n)),
        Value::Ratio(r) => Ok(r.to_f64().unwrap_or(f64::NAN) as i64),
        Value::BigDec(d) => Ok(crate::core::bigint_long_value(&d.to_bigint_trunc())),
        Value::Nil => Err(boxarg_npe()),
        _ => Err(unexpected_param_long(v)),
    }
}

/// The boxArg `long`-param miss (the `unexpected_param`
/// sibling for 64-bit params).
fn unexpected_param_long(v: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!(
            "Unexpected param type, expected: long, given: {}",
            jvm_class_of(v)
        ),
    )
}

/// `Long/remainderUnsigned` (2-arg): the unsigned remainder
/// (zero divisor is `/ by zero` — oracle).
fn static_long_remainder_unsigned(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 2 {
        return Err(crate::types::no_method(
            "remainderUnsigned",
            args.len(),
            "java.lang.Long",
        ));
    }
    let x = long_arg(&args[0])?;
    let m = long_arg(&args[1])?;
    if m == 0 {
        return Err(NativeError::new(
            ThrowClass::ArithmeticException,
            "/ by zero".to_owned(),
        ));
    }
    Ok(Value::Int((x as u64).wrapping_rem(m as u64) as i64))
}

/// `String/format` (3-arg): `(format locale fmt args-array)`
/// (44 recipes pin `Locale/US` explicitly). Only the `US`
/// locale exists here, so the locale is accepted and the
/// shared `clojure.core/format` engine renders (it always
/// emits `.` decimals, like `Formatter` under `US`).
fn static_string_format(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 3 {
        return Err(crate::types::no_method(
            "format",
            args.len(),
            "java.lang.String",
        ));
    }
    if as_locale(&args[0]).is_none() {
        return Err(crate::core::cce_err(&args[0], "java.util.Locale"));
    }
    let Value::Array(a) = &args[2] else {
        return Err(crate::core::cce_err(&args[2], "[Ljava.lang.Object;"));
    };
    let mut full = Vec::with_capacity(a.len() + 1);
    full.push(args[1].clone());
    full.extend(a.snapshot());
    crate::fmt::native_format(eval, &full)
}

/// `Integer/parseInt` (issue #201 S12): the long body narrowed to
/// i32 (out-of-range is the same NFE the JDK raises). #205 adds
/// the `(String, int)` radix overload.
fn static_integer_parse_int(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Str(s) => crate::core::parse_long_str(s)
                .and_then(|n| i32::try_from(n).ok())
                .map(|n| Value::Int(i64::from(n)))
                .ok_or_else(|| parse_nfe(s)),
            Value::Nil => Err(parse_nfe("null")),
            other => Err(crate::core::cce_err(other, "java.lang.String")),
        },
        2 => {
            let Value::Str(s) = &args[0] else {
                if matches!(&args[0], Value::Nil) {
                    return Err(parse_nfe("null"));
                }
                return Err(crate::core::cce_err(&args[0], "java.lang.String"));
            };
            let radix = str_int_arg(&args[1])?;
            parse_long_radix(s, radix)
                .and_then(|n| i32::try_from(n).ok())
                .map(|n| Value::Int(i64::from(n)))
                .ok_or_else(|| parse_radix_nfe(s, radix))
        }
        n => Err(crate::types::no_method("parseInt", n, "java.lang.Integer")),
    }
}

/// `Double/parseDouble` (issue #201 S12): the shared
/// `parse_double_str` body (trim + specials + hex floats, like
/// `parse-double`), throwing instead of answering nil.
fn static_double_parse_double(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Str(s) => crate::core::parse_double_str(s)
                .map(Value::Double)
                .ok_or_else(|| parse_nfe(s)),
            Value::Nil => Err(parse_nfe("null")),
            other => Err(crate::core::cce_err(other, "java.lang.String")),
        },
        n => Err(crate::types::no_method(
            "parseDouble",
            n,
            "java.lang.Double",
        )),
    }
}

/// A `Math/*` double argument (widening, like the `(double)`
/// overload): ints convert exactly when small, ratios/bigints
/// through f64 (NaN stays NaN).
fn math_double_arg(v: &Value, member: &str) -> Result<f64, NativeError> {
    match v {
        Value::Double(d) => Ok(*d),
        Value::Int(i) => Ok(*i as f64),
        Value::Ratio(r) => Ok(r.to_f64().unwrap_or(f64::NAN)),
        Value::BigInt(n) | Value::BigInteger(n) => Ok(n.to_f64().unwrap_or(f64::NAN)),
        Value::BigDec(d) => Ok(d.to_f64()),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            format!(
                "Cannot invoke \"java.lang.Math.{member}(double)\" because the argument is null"
            ),
        )),
        other => Err(crate::core::cce_err(other, "java.lang.Double")),
    }
}

/// `Math/abs` (issue #201 S12): ints wrap like the JDK
/// (`abs(MIN_VALUE)` is itself), everything else goes double.
fn static_math_abs(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Int(i) => Ok(Value::Int(i.wrapping_abs())),
            other => Ok(Value::Double(math_double_arg(other, "abs")?.abs())),
        },
        n => Err(crate::types::no_method("abs", n, "java.lang.Math")),
    }
}

/// `Math/sqrt` (issue #201 S12).
fn static_math_sqrt(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Double(math_double_arg(&args[0], "sqrt")?.sqrt())),
        n => Err(crate::types::no_method("sqrt", n, "java.lang.Math")),
    }
}

/// `Math/ceil` (issue #201 S12).
fn static_math_ceil(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Double(math_double_arg(&args[0], "ceil")?.ceil())),
        n => Err(crate::types::no_method("ceil", n, "java.lang.Math")),
    }
}

fn static_math_log(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Double(math_double_arg(&args[0], "log")?.ln())),
        n => Err(crate::types::no_method("log", n, "java.lang.Math")),
    }
}

fn static_math_exp(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Double(math_double_arg(&args[0], "exp")?.exp())),
        n => Err(crate::types::no_method("exp", n, "java.lang.Math")),
    }
}

fn static_math_pow(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        2 => Ok(Value::Double(
            math_double_arg(&args[0], "pow")?.powf(math_double_arg(&args[1], "pow")?),
        )),
        n => Err(crate::types::no_method("pow", n, "java.lang.Math")),
    }
}

/// The trig/hyperbolic/inverse family (recipes `optics/*`,
/// `finance/var_es`, `search/backprop` call `sin`/`cos`/`tanh`;
/// the siblings ride along so the next script cannot miss):
/// plain double math through `math_double_arg`, exactly like
/// `log`/`exp` above. HotSpot implements these with the system
/// libm (not `StrictMath`), so on the same platform the bits
/// match the host libm Rust calls.
macro_rules! static_math_unary {
    ($name:ident, $member:literal, $method:ident) => {
        fn $name(args: &[Value]) -> Result<Value, NativeError> {
            match args.len() {
                1 => Ok(Value::Double(math_double_arg(&args[0], $member)?.$method())),
                n => Err(crate::types::no_method($member, n, "java.lang.Math")),
            }
        }
    };
}

static_math_unary!(static_math_sin, "sin", sin);
static_math_unary!(static_math_cos, "cos", cos);
static_math_unary!(static_math_tan, "tan", tan);
static_math_unary!(static_math_asin, "asin", asin);
static_math_unary!(static_math_acos, "acos", acos);
static_math_unary!(static_math_atan, "atan", atan);
static_math_unary!(static_math_sinh, "sinh", sinh);
static_math_unary!(static_math_cosh, "cosh", cosh);
static_math_unary!(static_math_tanh, "tanh", tanh);
static_math_unary!(static_math_cbrt, "cbrt", cbrt);
static_math_unary!(static_math_floor, "floor", floor);
static_math_unary!(static_math_log10, "log10", log10);

/// `Math/atan2` and `Math/hypot` (2-arg): `bitonic_tsp` computes
/// distances with `hypot` over int coords (widened by
/// `math_double_arg`).
fn static_math_atan2(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        2 => Ok(Value::Double(
            math_double_arg(&args[0], "atan2")?.atan2(math_double_arg(&args[1], "atan2")?),
        )),
        n => Err(crate::types::no_method("atan2", n, "java.lang.Math")),
    }
}

fn static_math_hypot(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        2 => Ok(Value::Double(
            math_double_arg(&args[0], "hypot")?.hypot(math_double_arg(&args[1], "hypot")?),
        )),
        n => Err(crate::types::no_method("hypot", n, "java.lang.Math")),
    }
}

/// `Math/round` (1-arg, the `(double)` overload returning
/// `long`): `floor(a + 0.5)` narrowed Java-style (NaN becomes 0,
/// out-of-range saturates — Rust's `as i64` does exactly that).
fn static_math_round(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Int(
            (math_double_arg(&args[0], "round")? + 0.5).floor() as i64,
        )),
        n => Err(crate::types::no_method("round", n, "java.lang.Math")),
    }
}

static_math_unary!(static_math_rint, "rint", round_ties_even);
static_math_unary!(static_math_to_radians, "toRadians", to_radians);
static_math_unary!(static_math_to_degrees, "toDegrees", to_degrees);

/// `Math/floorDiv` (2-arg, the `(long,long)` overload): division
/// rounded toward negative infinity. Computed in `i128` and
/// wrapped: `floorDiv(MIN,-1)` is `MIN` on the JVM (no throw —
/// only a zero divisor throws `/ by zero`).
fn static_math_floor_div(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        2 => {
            let x = long_arg(&args[0])?;
            let y = long_arg(&args[1])?;
            if y == 0 {
                return Err(NativeError::new(
                    ThrowClass::ArithmeticException,
                    "/ by zero".to_owned(),
                ));
            }
            Ok(Value::Int(floor_div128(x, y) as i64))
        }
        n => Err(crate::types::no_method("floorDiv", n, "java.lang.Math")),
    }
}

/// `Math/floorMod` (2-arg): `x - floorDiv(x,y)*y` (the sign of
/// the divisor, zero for `MIN % -1`). Zero divisor is `/ by
/// zero`, like `floorDiv`.
fn static_math_floor_mod(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        2 => {
            let x = long_arg(&args[0])?;
            let y = long_arg(&args[1])?;
            if y == 0 {
                return Err(NativeError::new(
                    ThrowClass::ArithmeticException,
                    "/ by zero".to_owned(),
                ));
            }
            Ok(Value::Int(
                (x as i128 - floor_div128(x, y) * y as i128) as i64,
            ))
        }
        n => Err(crate::types::no_method("floorMod", n, "java.lang.Math")),
    }
}

/// Exact floored division over 128-bit intermediates.
fn floor_div128(x: i64, y: i64) -> i128 {
    let (x, y) = (x as i128, y as i128);
    let d = x / y;
    let r = x % y;
    if r != 0 && (r < 0) != (y < 0) {
        d - 1
    } else {
        d
    }
}

/// `Math/random`: a `Double` in `[0, 1)` (same draw as
/// `clojure.core/rand` — the JVM backs both with one `Random`).
fn static_math_random(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        0 => Ok(Value::Double(rand::rng().random_range(0.0..1.0))),
        n => Err(crate::types::no_method("random", n, "java.lang.Math")),
    }
}

/// `Long/toBinaryString` (#205): the unsigned base-2 digits (no
/// leading zeros; negatives print all 64 bits, like the JDK).
fn static_long_to_binary_string(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Str(Arc::from(format!(
            "{:b}",
            long_arg(&args[0])? as u64
        )))),
        n => Err(crate::types::no_method(
            "toBinaryString",
            n,
            "java.lang.Long",
        )),
    }
}

/// `Long/toHexString` (#205): the unsigned base-16 digits
/// (lowercase, like the JDK).
fn static_long_to_hex_string(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Str(Arc::from(format!(
            "{:x}",
            long_arg(&args[0])? as u64
        )))),
        n => Err(crate::types::no_method("toHexString", n, "java.lang.Long")),
    }
}

/// `Long/bitCount` (#205): the population count.
fn static_long_bit_count(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Int(i64::from(
            (long_arg(&args[0])? as u64).count_ones(),
        ))),
        n => Err(crate::types::no_method("bitCount", n, "java.lang.Long")),
    }
}

/// `Long/numberOfTrailingZeros` (#205): 64 for zero, like the JDK.
fn static_long_number_of_trailing_zeros(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Int(i64::from(
            (long_arg(&args[0])? as u64).trailing_zeros(),
        ))),
        n => Err(crate::types::no_method(
            "numberOfTrailingZeros",
            n,
            "java.lang.Long",
        )),
    }
}

/// `Integer/toBinaryString` (#205): the `(int)` overload —
/// the arg truncates to 32 bits first (negatives print 32
/// digits, while `Long/toBinaryString` prints 64).
fn static_integer_to_binary_string(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Str(Arc::from(format!(
            "{:b}",
            str_int_arg(&args[0])? as u32
        )))),
        n => Err(crate::types::no_method(
            "toBinaryString",
            n,
            "java.lang.Integer",
        )),
    }
}

/// `Double/isNaN` / `isInfinite` / `isFinite` (#205): through
/// the widening double coercion (ints read finite).
fn static_double_is_nan(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(math_double_arg(&args[0], "isNaN")?.is_nan())),
        n => Err(crate::types::no_method("isNaN", n, "java.lang.Double")),
    }
}

fn static_double_is_infinite(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(
            math_double_arg(&args[0], "isInfinite")?.is_infinite(),
        )),
        n => Err(crate::types::no_method("isInfinite", n, "java.lang.Double")),
    }
}

fn static_double_is_finite(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(
            math_double_arg(&args[0], "isFinite")?.is_finite(),
        )),
        n => Err(crate::types::no_method("isFinite", n, "java.lang.Double")),
    }
}

/// `Boolean/valueOf` (#205): the `(boolean)` overload passes
/// through; the `(String)` overload is ASCII
/// case-insensitive `true` (nil answers false — the JVM's
/// `(String)` overload with a null, never an NPE). Anything
/// else has no overload (the miss shape).
fn static_boolean_value_of(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            Value::Str(s) => Ok(Value::Bool(s.eq_ignore_ascii_case("true"))),
            Value::Nil => Ok(Value::Bool(false)),
            _ => Err(crate::types::no_method("valueOf", 1, "java.lang.Boolean")),
        },
        n => Err(crate::types::no_method("valueOf", n, "java.lang.Boolean")),
    }
}

/// `Float/floatToIntBits` (#205): the `(float)` overload over a
/// narrowed double (NaN canonicalizes to `0x7fc00000`, like
/// the JDK).
fn static_float_to_int_bits(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => {
            let f = math_double_arg(&args[0], "floatToIntBits")? as f32;
            let bits = if f.is_nan() {
                0x7fc0_0000u32
            } else {
                f.to_bits()
            };
            Ok(Value::Int(i64::from(bits as i32)))
        }
        n => Err(crate::types::no_method(
            "floatToIntBits",
            n,
            "java.lang.Float",
        )),
    }
}

/// `Float/intBitsToFloat` (#205): quiet NaN payloads survive
/// the widening (bit-exact through `f32`, like the JDK).
fn static_int_bits_to_float(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Double(f64::from(f32::from_bits(
            str_int_arg(&args[0])? as u32,
        )))),
        n => Err(crate::types::no_method(
            "intBitsToFloat",
            n,
            "java.lang.Float",
        )),
    }
}

/// `BigInteger/valueOf` (#205): the `(long)` overload lifts
/// through the boxArg coercion.
fn static_bigint_value_of(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => long_arg(&args[0]).map(|n| bigint_value(num_bigint::BigInt::from(n))),
        n => Err(crate::types::no_method(
            "valueOf",
            n,
            "java.math.BigInteger",
        )),
    }
}

/// `Arrays/copyOfRange` (#205): the `(byte[],int,int)`
/// overload only (the suite's single use). `from > to` is the
/// JDK's `IllegalArgumentException` text; a nil array is the
/// bare NPE (the `arraycopy` precedent); a negative or
/// over-long `from` is `ArrayIndexOutOfBoundsException` on the
/// index; a `to` past the end zero-pads. Anything else misses.
fn static_arrays_copy_of_range(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 3 {
        return Err(crate::types::no_method(
            "copyOfRange",
            args.len(),
            "java.util.Arrays",
        ));
    }
    if matches!(&args[0], Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            String::new(),
        ));
    }
    let Value::Array(a) = &args[0] else {
        return Err(crate::types::no_method(
            "copyOfRange",
            3,
            "java.util.Arrays",
        ));
    };
    let (from, to) = (str_int_arg(&args[1])?, str_int_arg(&args[2])?);
    if to < from {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("{from} > {to}"),
        ));
    }
    let guard = a.lock();
    let clj_value::ArrayStore::Bytes(slots) = &*guard else {
        return Err(crate::types::no_method(
            "copyOfRange",
            3,
            "java.util.Arrays",
        ));
    };
    if from < 0 || from as usize > slots.len() {
        return Err(NativeError::new(
            ThrowClass::ArrayIndexOutOfBoundsException,
            from.to_string(),
        ));
    }
    let (from, new_len) = (from as usize, (to - from) as usize);
    let mut out = vec![0i8; new_len];
    let take = new_len.min(slots.len().saturating_sub(from));
    out[..take].copy_from_slice(&slots[from..from + take]);
    Ok(Value::Array(clj_value::Array::new(
        clj_value::ArrayDesc::flat(clj_value::BaseType::Byte),
        clj_value::ArrayStore::Bytes(out),
    )))
}

/// `URLEncoder/encode` (#205): `application/x-www-form-urlencoded`
/// over the named charset's bytes (`lookup_charset`, so unknown
/// names are `UnsupportedEncodingException`, like the JDK).
/// Unreserved bytes (`0-9A-Za-z.-*`) pass, space becomes `+`,
/// the rest `%HH` uppercase. Nils NPE on their parameter.
fn static_urlencoder_encode(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 2 {
        return Err(crate::types::no_method(
            "encode",
            args.len(),
            "java.net.URLEncoder",
        ));
    }
    let (Some(s), Some(enc)) = (str_text_arg(&args[0])?, str_text_arg(&args[1])?) else {
        let param = if matches!(&args[0], Value::Nil) {
            "s"
        } else {
            "enc"
        };
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            format!(
                "Cannot invoke \"java.net.URLEncoder.encode(String, String)\" because \"{param}\" is null"
            ),
        ));
    };
    let cs = crate::ctor::lookup_charset(enc)?;
    let mut out = String::new();
    for b in crate::ctor::encode(cs, s, true) {
        match b {
            b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | b'.' | b'-' | b'*' | b'_' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                use std::fmt::Write as _;
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    Ok(Value::Str(Arc::from(out.as_str())))
}

/// `Character/isDigit` (#205): `Char` and `Int` (code point)
/// spellings. `is_numeric` is exact over ASCII; non-ASCII
/// letter-numbers and number-others (Roman numerals, vulgar
/// fractions) read true — a documented narrowing (the JDK
/// counts `Nd` only).
fn static_character_is_digit(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(match &args[0] {
            Value::Char(c) => c.is_numeric(),
            Value::Int(i) => u32::try_from(*i)
                .ok()
                .and_then(char::from_u32)
                .is_some_and(|c| c.is_numeric()),
            other => {
                return Err(crate::core::cce_err(other, "java.lang.Character"));
            }
        })),
        n => Err(crate::types::no_method("isDigit", n, "java.lang.Character")),
    }
}

/// `Character/isLetter` (#205): same shape. `is_alphabetic`
/// is exact except letter-numbers (Roman numerals read true)
/// — pinned in the cover test.
fn static_character_is_letter(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(match &args[0] {
            Value::Char(c) => c.is_alphabetic(),
            Value::Int(i) => u32::try_from(*i)
                .ok()
                .and_then(char::from_u32)
                .is_some_and(|c| c.is_alphabetic()),
            other => {
                return Err(crate::core::cce_err(other, "java.lang.Character"));
            }
        })),
        n => Err(crate::types::no_method(
            "isLetter",
            n,
            "java.lang.Character",
        )),
    }
}

/// `Character/isAlphabetic` (#205): exact (both are the
/// Unicode `Alphabetic` property).
fn static_character_is_alphabetic(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(match &args[0] {
            Value::Char(c) => c.is_alphabetic(),
            Value::Int(i) => u32::try_from(*i)
                .ok()
                .and_then(char::from_u32)
                .is_some_and(|c| c.is_alphabetic()),
            other => {
                return Err(crate::core::cce_err(other, "java.lang.Character"));
            }
        })),
        n => Err(crate::types::no_method(
            "isAlphabetic",
            n,
            "java.lang.Character",
        )),
    }
}

/// Simple-uppercase one char (ASCII exact; non-ASCII takes the
/// first scalar of the full mapping — `ß` maps to `S` where
/// the JDK's simple mapping keeps `ß`, pinned in the test).
fn simple_upper(c: char) -> char {
    if c.is_ascii() {
        c.to_ascii_uppercase()
    } else {
        c.to_uppercase().next().unwrap_or(c)
    }
}

/// Simple-lowercase one char (same shape; `Σ` maps to `σ`,
/// like the JDK's simple mapping).
fn simple_lower(c: char) -> char {
    if c.is_ascii() {
        c.to_ascii_lowercase()
    } else {
        c.to_lowercase().next().unwrap_or(c)
    }
}

/// `Character/toUpperCase` (#205): the `(char)` overload
/// answers a `Char`, the `(int)` overload a code-point `Int`
/// (out-of-range code points pass through, like the JDK).
fn static_character_to_upper_case(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Char(c) => Ok(Value::Char(simple_upper(*c))),
            Value::Int(i) => Ok(Value::Int(
                match u32::try_from(*i).ok().and_then(char::from_u32) {
                    Some(c) => i64::from(simple_upper(c) as u32),
                    None => *i,
                },
            )),
            other => Err(crate::core::cce_err(other, "java.lang.Character")),
        },
        n => Err(crate::types::no_method(
            "toUpperCase",
            n,
            "java.lang.Character",
        )),
    }
}

/// `Character/toLowerCase` (#205): the mirror image.
fn static_character_to_lower_case(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Char(c) => Ok(Value::Char(simple_lower(*c))),
            Value::Int(i) => Ok(Value::Int(
                match u32::try_from(*i).ok().and_then(char::from_u32) {
                    Some(c) => i64::from(simple_lower(c) as u32),
                    None => *i,
                },
            )),
            other => Err(crate::core::cce_err(other, "java.lang.Character")),
        },
        n => Err(crate::types::no_method(
            "toLowerCase",
            n,
            "java.lang.Character",
        )),
    }
}

/// `System/nanoTime` (#205): epoch nanos (wrapping `as` —
/// callers difference two readings, like the JDK's
/// unspecified origin).
fn static_system_nano_time(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        0 => Ok(Value::Int(epoch_nanos() as i64)),
        n => Err(crate::types::no_method("nanoTime", n, "java.lang.System")),
    }
}

/// `System/currentTimeMillis` (#205).
fn static_system_current_time_millis(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        0 => Ok(Value::Int(epoch_nanos() as i64 / 1_000_000)),
        n => Err(crate::types::no_method(
            "currentTimeMillis",
            n,
            "java.lang.System",
        )),
    }
}

/// Nanos since the Unix epoch (zero before it — unreachable on
/// real clocks, pinned only through the millis wrapper).
fn epoch_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// `System/getenv` (#205): the var's value, or nil when unset
/// (non-UTF8 values read nil — the port has no byte strings).
fn static_system_getenv(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => match &args[0] {
            Value::Str(name) => Ok(std::env::var(name.as_ref())
                .ok()
                .map(|v| Value::Str(Arc::from(v.as_str())))
                .unwrap_or(Value::Nil)),
            Value::Nil => Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"java.lang.System.getenv(String)\" because \"name\" is null"
                    .to_owned(),
            )),
            other => Err(crate::core::cce_err(other, "java.lang.String")),
        },
        n => Err(crate::types::no_method("getenv", n, "java.lang.System")),
    }
}

/// `System/arraycopy` (#205): across same-store arrays (the
/// `hashes/*` padding loops copy `byte[]`→`byte[]`). Source
/// bytes snapshot first, so overlapping same-array copies
/// move correctly. Answers nil (the JVM returns void).
fn static_system_arraycopy(args: &[Value]) -> Result<Value, NativeError> {
    if args.len() != 5 {
        return Err(crate::types::no_method(
            "arraycopy",
            args.len(),
            "java.lang.System",
        ));
    }
    let (Value::Array(src), Value::Array(dst)) = (&args[0], &args[2]) else {
        let bad = if !matches!(&args[0], Value::Array(_)) {
            &args[0]
        } else {
            &args[2]
        };
        if matches!(bad, Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                String::new(),
            ));
        }
        return Err(NativeError::new(
            ThrowClass::ArrayStoreException,
            "arraycopy".to_owned(),
        ));
    };
    let s_pos = str_int_arg(&args[1])?;
    let d_pos = str_int_arg(&args[3])?;
    let len = str_int_arg(&args[4])?;
    arraycopy_slots(src, s_pos, dst, d_pos, len)?;
    Ok(Value::Nil)
}

/// A snapshotted `arraycopy` source range, one variant per
/// flat store (multidimensional `Arrays` sources never
/// snapshot — they take the store-mismatch error, as before).
enum ArrayCells {
    Bools(Vec<bool>),
    Bytes(Vec<i8>),
    Chars(Vec<char>),
    Shorts(Vec<i16>),
    Ints(Vec<i32>),
    Longs(Vec<i64>),
    Floats(Vec<f64>),
    Doubles(Vec<f64>),
    Objs(Vec<Value>),
}

/// Snapshot an `arraycopy` source range (the lock drops before
/// the destination locks — same-array copies never deadlock).
fn arraycopy_snapshot(store: &clj_value::ArrayStore, s: usize, n: usize) -> Option<ArrayCells> {
    match store {
        clj_value::ArrayStore::Bools(sv) => Some(ArrayCells::Bools(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Bytes(sv) => Some(ArrayCells::Bytes(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Chars(sv) => Some(ArrayCells::Chars(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Shorts(sv) => Some(ArrayCells::Shorts(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Ints(sv) => Some(ArrayCells::Ints(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Longs(sv) => Some(ArrayCells::Longs(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Floats(sv) => Some(ArrayCells::Floats(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Doubles(sv) => Some(ArrayCells::Doubles(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Objs(sv) => Some(ArrayCells::Objs(sv[s..s + n].to_vec())),
        clj_value::ArrayStore::Arrays(_) => None,
    }
}

/// The cell count of a snapshotted `arraycopy` source range.
fn arraycells_len(cells: &ArrayCells) -> usize {
    match cells {
        ArrayCells::Bools(v) => v.len(),
        ArrayCells::Bytes(v) => v.len(),
        ArrayCells::Chars(v) => v.len(),
        ArrayCells::Shorts(v) => v.len(),
        ArrayCells::Ints(v) => v.len(),
        ArrayCells::Longs(v) => v.len(),
        ArrayCells::Floats(v) => v.len(),
        ArrayCells::Doubles(v) => v.len(),
        ArrayCells::Objs(v) => v.len(),
    }
}

/// Write a snapshotted range into an `arraycopy` destination
/// (same store or the mismatch error).
fn arraycopy_write(
    store: &mut clj_value::ArrayStore,
    d: usize,
    cells: ArrayCells,
) -> Result<(), NativeError> {
    let n = arraycells_len(&cells);
    match (store, cells) {
        (clj_value::ArrayStore::Bools(dv), ArrayCells::Bools(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Bytes(dv), ArrayCells::Bytes(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Chars(dv), ArrayCells::Chars(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Shorts(dv), ArrayCells::Shorts(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Ints(dv), ArrayCells::Ints(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Longs(dv), ArrayCells::Longs(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Floats(dv), ArrayCells::Floats(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Doubles(dv), ArrayCells::Doubles(tmp)) => {
            dv[d..d + n].copy_from_slice(&tmp);
        }
        (clj_value::ArrayStore::Objs(dv), ArrayCells::Objs(tmp)) => {
            dv[d..d + n].clone_from_slice(&tmp);
        }
        _ => {
            return Err(NativeError::new(
                ThrowClass::ArrayStoreException,
                "arraycopy: array type mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

/// The slot move behind `arraycopy` (split for the complexity
/// budget): negative positions/lengths and overruns are
/// `IndexOutOfBoundsException`, cross-store copies are
/// `ArrayStoreException`. The source snapshots under its own
/// lock and the destination writes under its own — sequential
/// locks, so same-array copies (memmove) never deadlock.
fn arraycopy_slots(
    src: &clj_value::Array,
    s_pos: i32,
    dst: &clj_value::Array,
    d_pos: i32,
    len: i32,
) -> Result<(), NativeError> {
    if s_pos < 0 || d_pos < 0 || len < 0 {
        return Err(arraycopy_oob());
    }
    let (s, d, n) = (s_pos as usize, d_pos as usize, len as usize);
    let cells = {
        let guard = src.lock();
        if s.checked_add(n).is_none_or(|e| e > guard.len()) {
            return Err(arraycopy_oob());
        }
        arraycopy_snapshot(&guard, s, n)
    };
    let mut guard = dst.lock();
    if d.checked_add(n).is_none_or(|e| e > guard.len()) {
        return Err(arraycopy_oob());
    }
    match cells {
        Some(cells) => arraycopy_write(&mut guard, d, cells),
        None => Err(NativeError::new(
            ThrowClass::ArrayStoreException,
            "arraycopy: array type mismatch".to_owned(),
        )),
    }
}

/// The `arraycopy` bounds failure (the JDK throws
/// `IndexOutOfBoundsException` with no message from the
/// intrinsic).
fn arraycopy_oob() -> NativeError {
    NativeError::new(ThrowClass::IndexOutOfBoundsException, String::new())
}

/// `Character/isLetterOrDigit` (1-arg): `Char` and `Int`
/// (code point) spellings (a documented narrowing —
/// `is_alphanumeric` covers a slightly wider Unicode range
/// than `isLetter||isDigit`, identical over ASCII).
fn static_character_is_letter_or_digit(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(match &args[0] {
            Value::Char(c) => c.is_alphanumeric(),
            Value::Int(i) => u32::try_from(*i)
                .ok()
                .and_then(char::from_u32)
                .is_some_and(|c| c.is_alphanumeric()),
            other => {
                return Err(crate::core::cce_err(other, "java.lang.Character"));
            }
        })),
        n => Err(crate::types::no_method(
            "isLetterOrDigit",
            n,
            "java.lang.Character",
        )),
    }
}

/// `Character/isLowerCase` / `isUpperCase` (1-arg): `Char` and
/// `Int` (code point) spellings, same shape as
/// `isLetterOrDigit` above (`fol_unify` tests vars with
/// `isLowerCase` over `(.charAt sym 0)`). A documented
/// narrowing — Rust's Unicode case properties differ from
/// `java.lang.Character` on a handful of non-ASCII code points,
/// identical over ASCII.
fn static_character_is_lower_case(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(match &args[0] {
            Value::Char(c) => c.is_lowercase(),
            Value::Int(i) => u32::try_from(*i)
                .ok()
                .and_then(char::from_u32)
                .is_some_and(|c| c.is_lowercase()),
            other => {
                return Err(crate::core::cce_err(other, "java.lang.Character"));
            }
        })),
        n => Err(crate::types::no_method(
            "isLowerCase",
            n,
            "java.lang.Character",
        )),
    }
}

fn static_character_is_upper_case(args: &[Value]) -> Result<Value, NativeError> {
    match args.len() {
        1 => Ok(Value::Bool(match &args[0] {
            Value::Char(c) => c.is_uppercase(),
            Value::Int(i) => u32::try_from(*i)
                .ok()
                .and_then(char::from_u32)
                .is_some_and(|c| c.is_uppercase()),
            other => {
                return Err(crate::core::cce_err(other, "java.lang.Character"));
            }
        })),
        n => Err(crate::types::no_method(
            "isUpperCase",
            n,
            "java.lang.Character",
        )),
    }
}

/// Java `Long.hashCode` over an identity id (the private `id_hash`
/// mix in `hasheq`, restated: small ids survive intact, so distinct
/// allocation ids hash distinctly).
fn id_mix(id: u64) -> i32 {
    (id ^ (id >> 32)) as i32
}

/// The low 32 address bits of an allocation (stable within one
/// process; distinct live allocations differ unless 4G apart).
fn ptr_mix<T: ?Sized>(p: *const T) -> i32 {
    p.addr() as i32
}

/// The identity hash (split for the complexity budget): the id
/// tiers first, then the allocation-pointer tier, else the content
/// hash — `identical?` values always agree (same id, same
/// allocation, or equal values in the cached/pooled tiers), and nil
/// is 0 (shared with `hash_code`, per the JDK).
fn identity_hash(v: &Value) -> i32 {
    identity_hash_heap(v)
        .or_else(|| identity_hash_alloc(v))
        .or_else(|| identity_hash_ptr(v))
        .unwrap_or_else(|| clj_value::hash_code(v))
}

/// `identityHashCode` over the `identical_heap` id kinds plus
/// `Symbol` (id-compared like them): collections, entries, tagged
/// and conditional reads, primitive vectors, and the scalar ids.
/// `None` for every other kind.
fn identity_hash_heap(v: &Value) -> Option<i32> {
    match v {
        Value::Symbol(s) => Some(id_mix(s.id())),
        Value::List(l) => Some(id_mix(l.id())),
        Value::Vector(x) => Some(id_mix(x.id())),
        Value::Map(m) => Some(id_mix(m.id())),
        Value::Set(s) => Some(id_mix(s.id())),
        Value::MapEntry { id, .. } => Some(id_mix(*id)),
        Value::TaggedLiteral { id, .. } => Some(id_mix(*id)),
        Value::ReaderConditional { id, .. } => Some(id_mix(*id)),
        Value::PrimitiveVector(p) => Some(id_mix(p.id())),
        Value::Uuid(u) => Some(id_mix(u.id())),
        Value::Instant(i) => Some(id_mix(i.id())),
        Value::Uri(u) => Some(id_mix(u.id())),
        _ => None,
    }
}

/// `identityHashCode` over the cell/var/throwable ids (one tier of
/// [`identity_hash`]): vars, unbounds, throwables, transients,
/// atoms, delays, futures, promises, agents, volatiles, refs,
/// arrays, and executors. `None` for every other kind.
fn identity_hash_alloc(v: &Value) -> Option<i32> {
    match v {
        Value::Var(var) => Some(id_mix(var.id)),
        Value::Unbound(u) => Some(id_mix(u.var_id)),
        Value::Throwable(t) => Some(id_mix(t.id)),
        Value::Transient(t) => Some(id_mix(t.id())),
        Value::Atom(a) => Some(id_mix(a.id())),
        Value::Delay(d) => Some(id_mix(d.id())),
        Value::Future(f) => Some(id_mix(f.id())),
        Value::Promise(p) => Some(id_mix(p.id())),
        Value::Agent(a) => Some(id_mix(a.id())),
        Value::Volatile(x) => Some(id_mix(x.id())),
        Value::Ref(r) => Some(id_mix(r.id())),
        Value::Array(a) => Some(id_mix(a.id())),
        Value::Executor(x) => Some(id_mix(x.id())),
        _ => None,
    }
}

/// `identityHashCode` over the `Arc` allocations (one tier of
/// [`identity_hash`]): strings, tower numbers, extensions, host
/// values, and multimethods — every kind whose `identical?` is
/// `Arc::ptr_eq`. `None` for every other kind.
fn identity_hash_ptr(v: &Value) -> Option<i32> {
    match v {
        Value::Str(s) => Some(ptr_mix(Arc::as_ptr(s))),
        Value::BigInt(n) => Some(ptr_mix(Arc::as_ptr(n))),
        Value::BigInteger(n) => Some(ptr_mix(Arc::as_ptr(n))),
        Value::BigDec(d) => Some(ptr_mix(Arc::as_ptr(d))),
        Value::Ratio(r) => Some(ptr_mix(Arc::as_ptr(r))),
        Value::Extension(e) => Some(ptr_mix(Arc::as_ptr(e))),
        Value::HostType(h) => Some(ptr_mix(Arc::as_ptr(h))),
        Value::MultiFn(m) => Some(ptr_mix(Arc::as_ptr(m))),
        _ => None,
    }
}

/// Read an import mapping (shared with name resolution).
pub(crate) fn import_in(eval: &Evaluator, ns: &str, simple: &str) -> Option<Arc<ClassVal>> {
    eval.registry.import_in(ns, simple)
}

/// A bean key.
fn kw(name: &str) -> Value {
    Value::Keyword(Keyword::new(None, name))
}

/// The throwable message as a value (nil when absent).
fn throwable_message(t: &clj_value::Throwable) -> Value {
    t.message.clone().map(Value::Str).unwrap_or(Value::Nil)
}

/// The throwable cause as a value (nil when absent).
fn throwable_cause(t: &clj_value::Throwable) -> Value {
    t.cause
        .clone()
        .map(|c| Value::Throwable(*c))
        .unwrap_or(Value::Nil)
}

/// The superclass name (interfaces and baseless classes have none).
fn superclass_name(c: &Arc<ClassVal>) -> Option<&str> {
    if c.is_iface() || c.bases().is_empty() {
        None
    } else {
        Some(c.bases()[0].as_ref())
    }
}

/// The superclass as a value (nil when absent or dangling).
fn superclass_value(eval: &Evaluator, c: &Arc<ClassVal>) -> Value {
    superclass_name(c)
        .and_then(|n| class_for_name(eval, n))
        .map(Value::Class)
        .unwrap_or(Value::Nil)
}

/// The interfaces as a vector in lattice order (dangling names skip).
fn interfaces_vector(eval: &Evaluator, c: &Arc<ClassVal>) -> Value {
    let rest = if c.is_iface() || c.bases().is_empty() {
        c.bases()
    } else {
        &c.bases()[1..]
    };
    let mut items = Vec::new();
    for base in rest {
        if let Some(ic) = class_for_name(eval, base) {
            items.push(Value::Class(ic));
        }
    }
    Value::Vector(Vector::from_vec(items))
}

/// The class keys of `bean` (`:class` itself is pushed by the caller).
fn class_bean_keys(eval: &Evaluator, c: &Arc<ClassVal>, out: &mut Vec<(Value, Value)>) {
    out.push((kw("array"), Value::Bool(false)));
    out.push((kw("interface"), Value::Bool(c.is_iface())));
    out.push((kw("interfaces"), interfaces_vector(eval, c)));
    out.push((kw("name"), Value::Str(Arc::from(c.name()))));
    out.push((
        kw("simpleName"),
        Value::Str(Arc::from(simple_name_of(c.name()).as_str())),
    ));
    out.push((kw("superclass"), superclass_value(eval, c)));
}

/// The throwable keys of `bean` (`:data` for `ExceptionInfo` only).
fn throwable_bean_keys(t: &clj_value::Throwable, out: &mut Vec<(Value, Value)>) {
    out.push((kw("cause"), throwable_cause(t)));
    out.push((kw("localizedMessage"), throwable_message(t)));
    out.push((kw("message"), throwable_message(t)));
    if t.class == ThrowClass::ExceptionInfo {
        out.push((kw("data"), t.data.clone().map(|d| *d).unwrap_or(Value::Nil)));
    }
}

/// Whether a host method name is a bean getter (`get`/`is` plus an
/// uppercase letter, like the JVM's property reflection).
fn bean_suffix(name: &str) -> Option<&str> {
    let rest = name
        .strip_prefix("get")
        .or_else(|| name.strip_prefix("is"))?;
    let first = rest.chars().next()?;
    if first.is_ascii_uppercase() {
        Some(rest)
    } else {
        None
    }
}

/// JavaBeans decapitalization: a leading all-caps run stays (`URL`
/// → `URL`), else the first letter lowercases (`Count` → `count`).
fn decapitalize(s: &str) -> String {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(first), Some(second))
            if first.is_ascii_uppercase() && second.is_ascii_uppercase() =>
        {
            s.to_owned()
        }
        (Some(first), _) => first.to_lowercase().collect::<String>() + &s[first.len_utf8()..],
        (None, _) => s.to_owned(),
    }
}

/// The host-method keys of `bean`: one eager 0-arg call per
/// getter-named method (sorted for determinism; throwing getters
/// fail `bean` itself) — except getters colliding with the
/// already-seeded fixed keys (`:class` via `getClass` on every
/// host instance, `:empty` via `isEmpty`/`getEmpty` on
/// records/structs): the fixed seeds win and the colliding
/// getter is never called, like the member surface shadowing
/// host members (§2.1.19).
fn host_method_bean_keys(
    eval: &mut Evaluator,
    h: &Arc<HostVal>,
    target: &Value,
    out: &mut Vec<(Value, Value)>,
) -> Result<(), NativeError> {
    let HostVal::Inst(i) = h.as_ref() else {
        return Ok(());
    };
    let mut getters: Vec<(&str, &str)> = i
        .def
        .methods
        .iter()
        .filter_map(|(name, overloads)| {
            let suffix = bean_suffix(name.as_ref())?;
            overloads
                .iter()
                .any(|o| o.arity == 0)
                .then_some((name.as_ref(), suffix))
        })
        .collect();
    getters.sort_unstable();
    let seeded: Vec<Value> = out.iter().map(|(k, _)| k.clone()).collect();
    for (name, suffix) in getters {
        let key = kw(&decapitalize(suffix));
        if seeded.contains(&key) {
            continue;
        }
        let val = crate::types::call_zero_arg_method(eval, h, target.clone(), name)?;
        out.push((key, val));
    }
    Ok(())
}

/// The host keys of `bean`: `:empty` for records and structs, plus
/// the 0-arg getter methods of instances (descriptors carry `:class`
/// alone).
fn host_bean_keys(
    eval: &mut Evaluator,
    h: &Arc<HostVal>,
    target: &Value,
    out: &mut Vec<(Value, Value)>,
) -> Result<(), NativeError> {
    if let HostVal::Inst(i) = h.as_ref() {
        if matches!(i.def.kind, HostKind::Record | HostKind::Struct) {
            out.push((kw("empty"), Value::Bool(i.content.is_empty())));
        }
    }
    host_method_bean_keys(eval, h, target, out)
}

/// The gate keys of `bean`: `:empty` for strings and
/// Collection/Map members, `:first`/`:last` for List members,
/// `:realized` for IPending members.
fn collection_bean_keys(
    eval: &mut Evaluator,
    v: &Value,
    out: &mut Vec<(Value, Value)>,
) -> Result<(), NativeError> {
    if matches!(v, Value::Str(_))
        || is_instance_of(eval, "java.util.Collection", v)
        || is_instance_of(eval, "java.util.Map", v)
    {
        out.push((kw("empty"), Value::Bool(surface_is_empty(eval, v)?)));
    }
    if is_instance_of(eval, "java.util.List", v) {
        out.push((kw("first"), surface_first(eval, v)?));
        out.push((kw("last"), surface_last(eval, v)?));
    }
    if is_instance_of(eval, "clojure.lang.IPending", v) {
        out.push((kw("realized"), Value::Bool(realized_flag(v))));
    }
    Ok(())
}

/// The non-`:class` keys of `bean` (split for the complexity
/// budget): classes, throwables, host values, and named values take
/// their fixed keys; everything else takes the gate keys (`:blank`
/// for strings, `:key`/`:value` for entries, and the
/// Collection/List/Pending gates).
fn bean_gated_keys(
    eval: &mut Evaluator,
    v: &Value,
    out: &mut Vec<(Value, Value)>,
) -> Result<(), NativeError> {
    match v {
        Value::Class(c) => {
            class_bean_keys(eval, c, out);
            Ok(())
        }
        Value::Throwable(t) => {
            throwable_bean_keys(t, out);
            Ok(())
        }
        Value::HostType(h) => host_bean_keys(eval, h, v, out),
        Value::Keyword(k) => {
            named_bean_keys(k.name(), k.namespace(), out);
            Ok(())
        }
        Value::Symbol(s) => {
            named_bean_keys(s.name(), s.namespace(), out);
            Ok(())
        }
        _ => {
            if let Value::Str(s) = v {
                out.push((kw("blank"), Value::Bool(is_blank(s))));
            }
            collection_bean_keys(eval, v, out)?;
            if let Value::MapEntry { key, val, .. } = v {
                out.push((kw("key"), key.as_ref().clone()));
                out.push((kw("value"), val.as_ref().clone()));
            }
            Ok(())
        }
    }
}

/// The named keys of `bean` (all three keys, always — the namespace
/// is nil when absent).
fn named_bean_keys(name: &str, namespace: Option<&str>, out: &mut Vec<(Value, Value)>) {
    out.push((kw("name"), Value::Str(Arc::from(name))));
    out.push((
        kw("namespace"),
        namespace
            .map(|n| Value::Str(Arc::from(n)))
            .unwrap_or(Value::Nil),
    ));
}

/// `bean`: an eager plain map of the value's getters — `:class`
/// always, plus the kind's fixed keys (nil is the `getClass` NPE,
/// like the JVM).
fn native_bean(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/bean", args, &[1])?;
    let v = &args[0];
    if matches!(v, Value::Nil) {
        return Err(crate::core::npe_getclass());
    }
    let mut entries: Vec<(Value, Value)> = vec![(kw("class"), class_value_of(v))];
    bean_gated_keys(eval, v, &mut entries)?;
    let mut map = new_map();
    for (k, val) in entries {
        map = map.insert(k, val);
    }
    Ok(Value::Map(map))
}

/// Whether either side of `equals` is a numeric-tower value (the
/// Java-exact rule covers `Long`/`Double`/`BigInt`/`BigDec`/`Ratio`).
fn is_tower_value(v: &Value) -> bool {
    matches!(
        v,
        Value::Int(_) | Value::Double(_) | Value::BigInt(_) | Value::BigDec(_) | Value::Ratio(_)
    )
}

/// `equals` on a numeric-tower pair (Java-exact): same classes
/// compare (`double` by bits — NaN equals NaN, signed zeros differ;
/// `BigDec` scale-sensitively), cross classes never do.
fn tower_equals(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        // Bit-equality is exact here: `Double.equals`
        // canonicalizes all NaN payloads, but the port has no
        // payload constructors, so every NaN is canonical.
        (Value::Double(x), Value::Double(y)) => x.to_bits() == y.to_bits(),
        (Value::BigInt(x), Value::BigInt(y)) => x == y,
        (Value::BigDec(x), Value::BigDec(y)) => x == y,
        (Value::Ratio(x), Value::Ratio(y)) => x == y,
        _ => false,
    }
}

/// The map content for `equals` branch 2 (records and structs
/// compare contents, maps compare themselves).
fn record_content(v: &Value) -> Option<Map> {
    match v {
        Value::Map(m) => Some(m.clone()),
        Value::HostType(h) => match h.as_ref() {
            HostVal::Inst(i) if matches!(i.def.kind, HostKind::Record | HostKind::Struct) => {
                Some(i.content.clone())
            }
            _ => None,
        },
        _ => None,
    }
}

/// `equals`: the tower rule for numerics, the map-content rule when
/// either side is a record, else port `=` (realizing, like the
/// native — pending seqs compare by content).
fn equals_member(eval: &mut Evaluator, a: &Value, b: &Value) -> Result<bool, NativeError> {
    if is_tower_value(a) || is_tower_value(b) {
        return Ok(tower_equals(a, b));
    }
    if crate::types::is_host_record(a) || crate::types::is_host_record(b) {
        return Ok(match (record_content(a), record_content(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        });
    }
    crate::lazy::equiv_vals(eval, a, b).map_err(signal_to_native)
}

/// `hashCode`: Java `.hashCode` over the realized value (mirroring
/// the private `hash-code` native's realize-then-hash contract,
/// including the infinite-seq refusal).
fn hashcode_member(eval: &mut Evaluator, target: &Value) -> Result<Value, NativeError> {
    if clj_value::seq_hash_unsupported(target) {
        return Err(NativeError::new(
            ThrowClass::UnsupportedOperationException,
            String::new(),
        ));
    }
    let v = eval.realize_deep(target).map_err(signal_to_native)?;
    if clj_value::seq_hash_unsupported(&v) {
        return Err(NativeError::new(
            ThrowClass::UnsupportedOperationException,
            String::new(),
        ));
    }
    Ok(Value::Int(i64::from(clj_value::hash_code(&v))))
}

/// The empty-list `NoSuchElementException` (empty message, like the
/// JVM's nil).
fn nsee_empty() -> NativeError {
    NativeError::new(ThrowClass::NoSuchElementException, String::new())
}

/// `getFirst` by position (empty lists throw).
fn surface_first(eval: &mut Evaluator, v: &Value) -> Result<Value, NativeError> {
    let seq = crate::lazy::seq_value(eval, v).map_err(signal_to_native)?;
    if matches!(seq, Value::Nil) {
        return Err(nsee_empty());
    }
    crate::lazy::first_value(eval, &seq).map_err(signal_to_native)
}

/// `getLast` by position (empty lists throw).
fn surface_last(eval: &mut Evaluator, v: &Value) -> Result<Value, NativeError> {
    let seq = crate::lazy::seq_value(eval, v).map_err(signal_to_native)?;
    if matches!(seq, Value::Nil) {
        return Err(nsee_empty());
    }
    let mut cur = seq;
    loop {
        let nxt = crate::lazy::next_value(eval, &cur).map_err(signal_to_native)?;
        if matches!(nxt, Value::Nil) {
            return crate::lazy::first_value(eval, &cur).map_err(signal_to_native);
        }
        cur = nxt;
    }
}

/// `isEmpty` (mirroring `empty?`: counted kinds compare the count,
/// the rest check for a first element).
fn surface_is_empty(eval: &mut Evaluator, v: &Value) -> Result<bool, NativeError> {
    if crate::core::is_counted(v) {
        Ok(crate::coll::count_of(eval, v)? == 0)
    } else {
        let s = crate::lazy::seq_value(eval, v).map_err(signal_to_native)?;
        Ok(matches!(s, Value::Nil))
    }
}

/// `isBlank` (empty or all Java whitespace — the JDK reads
/// `Character.isWhitespace`, not Unicode `White_Space`: NEL
/// and the non-breaking spaces are non-blank, FS–US blank).
fn is_blank(s: &str) -> bool {
    s.chars().all(java_is_whitespace)
}

/// The `isRealized` flag (the IPending gate admits only delays,
/// futures, promises, lazy seqs, and cycles/iterates).
fn realized_flag(v: &Value) -> bool {
    match v {
        Value::Delay(d) => d.is_realized(),
        Value::Promise(p) => p.is_realized(),
        Value::Future(f) => f.is_done(),
        Value::Extension(_) => extension_realized(v),
        _ => false,
    }
}

/// The `isRealized` flag for extensions: pending seqs report their
/// flag, realized Lazy-kind seqs are realized by construction, and
/// cycles/iterates report realized (the JVM's
/// `Cycle/Iterate.isRealized` is unconditionally true —
/// JVM-verified live; the only other IPending extensions).
fn extension_realized(v: &Value) -> bool {
    if let Some(p) = crate::lazy::as_pending(v) {
        return p.is_realized();
    }
    crate::lazy::as_realized(v).is_none_or(|r| r.kind() == SeqKind::Lazy)
}

/// The universal members (`toString` is `str`, `equals` the three
/// rules, `hashCode` Java's, `getClass` the class).
fn surface_universal(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("toString", 0) => {
            Some(crate::core::str_of_one(eval, target).map(|s| Value::Str(Arc::from(s.as_str()))))
        }
        ("equals", 1) => Some(equals_member(eval, target, &rest[0]).map(Value::Bool)),
        ("hashCode", 0) => Some(hashcode_member(eval, target)),
        ("getClass", 0) => Some(Ok(class_value_of(target))),
        _ => None,
    }
}

/// `isAssignableFrom`: whether the argument class has `c` as itself
/// or a supertype (nil is the empty-message NPE, non-classes the
/// short cast failure).
fn assignable_from(eval: &Evaluator, c: &Arc<ClassVal>, arg: &Value) -> Result<Value, NativeError> {
    let Value::Class(d) = arg else {
        if matches!(arg, Value::Nil) {
            return Err(NativeError::new(
                ThrowClass::NullPointerException,
                String::new(),
            ));
        }
        return Err(NativeError::cast(
            &crate::error::jvm_class_of(arg),
            "java.lang.Class",
        ));
    };
    Ok(Value::Bool(
        d.name() == c.name() || closure_contains(eval, d.bases(), c.name()),
    ))
}

/// The class members (classes only — anything else misses through).
fn surface_class_member(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let Value::Class(c) = target else {
        return None;
    };
    if let Some(hit) = surface_class_static(eval, meth, c, rest) {
        return Some(hit);
    }
    match (meth, rest.len()) {
        ("getName", 0) => Some(Ok(Value::Str(Arc::from(c.name())))),
        ("getSimpleName", 0) => Some(Ok(Value::Str(Arc::from(simple_name_of(c.name()).as_str())))),
        ("getSuperclass", 0) => Some(Ok(superclass_value(eval, c))),
        ("getInterfaces", 0) => Some(Ok(interfaces_vector(eval, c))),
        ("isInterface", 0) => Some(Ok(Value::Bool(c.is_iface()))),
        ("isArray", 0) => Some(Ok(Value::Bool(false))),
        ("isInstance", 1) => Some(Ok(Value::Bool(is_instance_of(eval, c.name(), &rest[0])))),
        ("isAssignableFrom", 1) => Some(assignable_from(eval, c, &rest[0])),
        ("cast", 1) => Some(cast_to_class(eval, c, &rest[0])),
        // Classes miss method-shape at every arity (unlike other
        // values, whose 0-arg misses read field-first).
        _ => Some(Err(crate::types::no_method(
            meth,
            rest.len(),
            &crate::error::jvm_class_of(target),
        ))),
    }
}

/// Statics resolvable through instance position on class values
/// (oracle `dotdot-static`); `Class` and `System` have tabled
/// statics, so anything else is `None` (the caller falls through
/// to the reflection members and their miss arm, like any
/// non-class target misses through). Split out so both halves
/// stay under the complexity ratchet; the arm sets are disjoint
/// by method name, so checking statics first is order-irrelevant.
fn surface_class_static(
    eval: &mut Evaluator,
    meth: &str,
    c: &Arc<ClassVal>,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("forName", 1..=3) if c.name() == "java.lang.Class" => Some(static_forname(eval, rest)),
        ("getProperty", 1..=2) if c.name() == "java.lang.System" => Some(static_getproperty(rest)),
        ("lineSeparator", 0) if c.name() == "java.lang.System" => Some(static_lineseparator(rest)),
        ("identityHashCode", 1) if c.name() == "java.lang.System" => {
            Some(static_identity_hashcode(rest))
        }
        _ => None,
    }
}

/// `isEmpty`: strings and Collection/Map members.
fn gated_empty(eval: &mut Evaluator, target: &Value) -> Option<Result<Value, NativeError>> {
    if matches!(target, Value::Str(_))
        || is_instance_of(eval, "java.util.Collection", target)
        || is_instance_of(eval, "java.util.Map", target)
    {
        Some(surface_is_empty(eval, target).map(Value::Bool))
    } else {
        None
    }
}

/// `getName` on named values (classes take the class member above).
fn gated_named_name(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Keyword(k) => Some(Ok(Value::Str(Arc::from(k.name())))),
        Value::Symbol(s) => Some(Ok(Value::Str(Arc::from(s.name())))),
        _ => None,
    }
}

/// `getNamespace` on named values.
fn gated_named_ns(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Keyword(k) => Some(Ok(k
            .namespace()
            .map(|ns| Value::Str(Arc::from(ns)))
            .unwrap_or(Value::Nil))),
        Value::Symbol(s) => Some(Ok(s
            .namespace()
            .map(|ns| Value::Str(Arc::from(ns)))
            .unwrap_or(Value::Nil))),
        _ => None,
    }
}

/// `getFirst`/`getLast` on List members.
fn gated_list_end(
    eval: &mut Evaluator,
    target: &Value,
    first: bool,
) -> Option<Result<Value, NativeError>> {
    if is_instance_of(eval, "java.util.List", target) {
        Some(if first {
            surface_first(eval, target)
        } else {
            surface_last(eval, target)
        })
    } else {
        None
    }
}

/// `getData` on `ExceptionInfo` (other throwables miss).
fn gated_exinfo_data(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Throwable(t) if t.class == ThrowClass::ExceptionInfo => {
            Some(Ok(t.data.clone().map(|d| *d).unwrap_or(Value::Nil)))
        }
        _ => None,
    }
}

/// The lattice-gated members plus the writer `close`/`flush`
/// (fields and host methods fall through to `__meth*`).
fn surface_gated(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    // Split by family for the complexity budget (arms are disjoint,
    // so order is irrelevant).
    surface_gated_coll(eval, meth, target, rest)
        .or_else(|| surface_gated_misc(eval, meth, target, rest))
        .or_else(|| surface_gated_jcoll(eval, meth, target, rest))
        .or_else(|| surface_gated_str(meth, target, rest))
        .or_else(|| surface_gated_builder(eval, meth, target, rest))
        .or_else(|| surface_gated_regex(meth, target, rest))
        .or_else(|| surface_gated_matcher(meth, target, rest))
        .or_else(|| surface_gated_io(meth, target, rest))
        .or_else(|| surface_gated_bigint(meth, target, rest))
        .or_else(|| surface_gated_bigdec(meth, target, rest))
        .or_else(|| surface_gated_uri(meth, target, rest))
        .or_else(|| crate::emurand::random_method(meth, target, rest))
        .or_else(|| crate::emutime::time_method(meth, target, rest))
}

/// The `StringBuilder` members (issue #201 S5, #205
/// `length`/`reverse`): `append`, the UTF-16 length, and the
/// in-place reversal — `toString` rides the universal via
/// `print_str`.
fn surface_gated_builder(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("append", 1) => gated_builder_append(eval, target, rest),
        ("length", 0) => gated_builder_length(target),
        ("reverse", 0) => gated_builder_reverse(target),
        _ => None,
    }
}

/// The `java.net.URI` members (#205): `getHost` only (the
/// suite's single use — `emails-from-url`).
fn surface_gated_uri(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("getHost", 0) => {
            let h = as_uri(target)?;
            Some(Ok(h
                .host()
                .map_or(Value::Nil, |host| Value::Str(Arc::from(host)))))
        }
        _ => None,
    }
}

/// The `Pattern` members (issue #201 S7): `split` only.
fn surface_gated_regex(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("split", 1 | 2) => gated_regex_split(target, rest),
        ("replaceAll", 1) => gated_matcher_replace_all(target, rest),
        _ => None,
    }
}

/// The `java.util` collection members (#205): deque ends
/// and queue heap ops (each gates its own target type).
fn surface_gated_jcoll(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    crate::emucoll::deque_method(meth, target, rest)
        .or_else(|| crate::emucoll::pqueue_method(eval, meth, target, rest))
        .or_else(|| crate::emudigest::digest_method(meth, target, rest))
}

/// The file/IO members (#205): `exists`/`length` on files,
/// `readAllBytes` on file input streams, `write` on writers
/// (strings, chars, ints) and file output streams (ints),
/// `readLine` on readers. Anything else misses through.
fn surface_gated_io(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("exists", 0) => gated_file_exists(target),
        ("length", 0) => gated_file_length(target),
        ("getPath", 0) => gated_file_get_path(target),
        ("getAbsolutePath", 0) => gated_file_get_absolute_path(target),
        ("delete", 0) => gated_file_delete(target),
        ("readAllBytes", 0) => crate::jio::file_input_read_all(target),
        ("readLine", 0) => gated_reader_read_line(target),
        ("write", 1) => gated_write(target, rest),
        _ => None,
    }
}

/// `File.exists`: missing paths answer false (like the JDK).
fn gated_file_exists(target: &Value) -> Option<Result<Value, NativeError>> {
    let path = crate::jdkio::file_path(target)?;
    Some(Ok(Value::Bool(std::path::Path::new(&path).exists())))
}

/// `File.length` on files (other `length` targets fall through
/// to their own gates): missing paths answer 0, like the JDK.
fn gated_file_length(target: &Value) -> Option<Result<Value, NativeError>> {
    let path = crate::jdkio::file_path(target)?;
    let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    Some(Ok(Value::Int(len as i64)))
}

/// `File.getPath` (#205): the path as given.
fn gated_file_get_path(target: &Value) -> Option<Result<Value, NativeError>> {
    let path = crate::jdkio::file_path(target)?;
    Some(Ok(Value::Str(Arc::from(path.as_str()))))
}

/// `File.getAbsolutePath` (#205): absolute paths pass through;
/// relative ones resolve against the working directory (the
/// `user.dir` mirror — the algorithms harness runs in a temp
/// work dir).
fn gated_file_get_absolute_path(target: &Value) -> Option<Result<Value, NativeError>> {
    let path = crate::jdkio::file_path(target)?;
    if std::path::Path::new(&path).is_absolute() {
        return Some(Ok(Value::Str(Arc::from(path.as_str()))));
    }
    let abs = std::env::current_dir()
        .map(|cwd| cwd.join(&path).to_string_lossy().into_owned())
        .unwrap_or(path);
    Some(Ok(Value::Str(Arc::from(abs.as_str()))))
}

/// `File.delete` (#205): true when something was removed (files
/// and empty dirs alike, like the JDK); missing paths answer
/// false.
fn gated_file_delete(target: &Value) -> Option<Result<Value, NativeError>> {
    let path = crate::jdkio::file_path(target)?;
    let ok = if std::path::Path::new(&path).is_dir() {
        std::fs::remove_dir(&path).is_ok()
    } else {
        std::fs::remove_file(&path).is_ok()
    };
    Some(Ok(Value::Bool(ok)))
}

/// `BufferedReader.readLine` on stream readers.
fn gated_reader_read_line(target: &Value) -> Option<Result<Value, NativeError>> {
    let r = crate::reader::as_reader(target)?;
    Some(crate::reader::reader_read_line(r))
}

/// `write` on writers and file output streams (disjoint
/// targets — writers take strings/chars/ints, streams take the
/// low byte of ints).
fn gated_write(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    if crate::io::as_writer(target).is_some() {
        return Some(gated_writer_write(target, &rest[0]));
    }
    if let Value::Int(n) = &rest[0] {
        if let hit @ Some(_) = crate::jio::file_output_write(target, *n as u8) {
            return hit;
        }
    }
    None
}

/// `Writer.write` (#205): strings append, chars and ints write
/// one UTF-16 unit (lone surrogates become U+FFFD — Rust
/// strings cannot hold them), nil NPEs on the parameter.
fn gated_writer_write(target: &Value, arg: &Value) -> Result<Value, NativeError> {
    match arg {
        Value::Str(s) => crate::io::write_to_writer(target, s).map(|()| Value::Nil),
        Value::Char(c) => crate::io::write_to_writer(target, &c.to_string()).map(|()| Value::Nil),
        Value::Int(n) => {
            let unit = char::from_u32((*n & 0xFFFF) as u32).unwrap_or('\u{FFFD}');
            crate::io::write_to_writer(target, &unit.to_string()).map(|()| Value::Nil)
        }
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.io.Writer.write(String)\" because \"str\" is null".to_owned(),
        )),
        _ => Err(crate::types::no_method("write", 1, "java.io.Writer")),
    }
}

/// The `Matcher` members (#205): `find`/`group`/`start`/`end`
/// (non-handles miss through to the uniform error).
fn surface_gated_matcher(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let h = crate::collmisc::as_matcher(target)?;
    match (meth, rest.len()) {
        ("find", 0) => Some(Ok(Value::Bool(h.find_bool()))),
        ("find", 1) => Some(str_int_arg(&rest[0]).and_then(|s| h.find_from(s).map(Value::Bool))),
        ("group", 0) => Some(h.group0().map(|g| Value::Str(Arc::from(g.as_str())))),
        ("group", 1) => Some(str_int_arg(&rest[0]).and_then(|i| h.group_n(i))),
        ("start", 0) => Some(h.last_span().map(|(s, _)| Value::Int(s as i64))),
        ("end", 0) => Some(h.last_span().map(|(_, e)| Value::Int(e as i64))),
        _ => None,
    }
}

/// `Matcher.replaceAll` (1-arg): `string/replace` over patterns
/// lowers here. Nil is ambiguous (`String` vs `Function`
/// overloads — the JVM reports the ambiguity); anything else
/// unhandled falls through to the missing-method error (the
/// `Function` overload stays unimplemented).
fn gated_matcher_replace_all(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let h = crate::collmisc::as_matcher(target)?;
    match &rest[0] {
        Value::Str(r) => Some(Ok(Value::Str(Arc::from(h.replace_all(r).as_str())))),
        Value::Nil => Some(Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "More than one matching method found: replaceAll".to_owned(),
        ))),
        _ => None,
    }
}

/// The collection-gated members (`String`, `Named`, `List`,
/// `MapEntry`).
fn surface_gated_coll(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("isEmpty", 0) => gated_empty(eval, target),
        ("isBlank", 0) => gated_blank(target),
        ("getName", 0) => gated_named_name(target),
        ("getNamespace", 0) => gated_named_ns(target),
        ("getFirst", 0) => gated_list_end(eval, target, true),
        ("getLast", 0) => gated_list_end(eval, target, false),
        ("getKey", 0) => gated_entry_key(target),
        ("getValue", 0) => gated_entry_val(target),
        ("indexOf", 1) => gated_coll_indexof(eval, target, rest),
        _ => None,
    }
}

/// `List.indexOf` (1-arg): the first slot `.equals` to the
/// needle, -1 absent (vectors and lists; other `List`-likes
/// keep the missing-method error).
fn gated_coll_indexof(
    eval: &mut Evaluator,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    // Subvectors compare like lists (#205:
    // `sorted_linked_list` indexes one as a `List`).
    if let Some(sv) = crate::lazy::as_subvector(target) {
        return Some(
            indexof_scan(eval, sv.iter(), &rest[0]).map(|at| Value::Int(at.unwrap_or(-1))),
        );
    }
    let scanned = match target {
        Value::Vector(x) => indexof_scan(eval, x.iter(), &rest[0]),
        Value::List(l) => indexof_scan(eval, l.iter().cloned(), &rest[0]),
        _ => return None,
    };
    Some(scanned.map(|at| Value::Int(at.unwrap_or(-1))))
}

/// Scan items for the needle with `=` semantics (the shared
/// `indexOf` body for vectors, lists, and subvectors).
fn indexof_scan(
    eval: &mut Evaluator,
    items: impl Iterator<Item = Value>,
    needle: &Value,
) -> Result<Option<i64>, NativeError> {
    for (i, item) in items.enumerate() {
        if equals_member(eval, &item, needle)? {
            return Ok(Some(i as i64));
        }
    }
    Ok(None)
}

/// The throwable/pending/writer-gated members.
fn surface_gated_misc(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("getMessage" | "getLocalizedMessage", 0) => gated_throwable_msg(target),
        ("getCause", 0) => gated_throwable_cause(target),
        ("getData", 0) => gated_exinfo_data(target),
        ("isRealized", 0) => gated_realized(eval, target),
        ("close", 0) => gated_writer_close(eval, target),
        ("flush", 0) => gated_writer_flush(eval, target),
        _ => None,
    }
}

/// `isBlank` on strings.
fn gated_blank(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(Ok(Value::Bool(is_blank(s)))),
        _ => None,
    }
}

/// `getKey`/`getValue` on entries.
fn gated_entry_key(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::MapEntry { key, .. } => Some(Ok(key.as_ref().clone())),
        _ => None,
    }
}

/// `getValue` on entries.
fn gated_entry_val(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::MapEntry { val, .. } => Some(Ok(val.as_ref().clone())),
        _ => None,
    }
}

/// `getMessage`/`getLocalizedMessage` on throwables.
fn gated_throwable_msg(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Throwable(t) => Some(Ok(throwable_message(t))),
        _ => None,
    }
}

/// `getCause` on throwables.
fn gated_throwable_cause(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Throwable(t) => Some(Ok(throwable_cause(t))),
        _ => None,
    }
}

/// `isRealized` on IPending members.
fn gated_realized(eval: &Evaluator, target: &Value) -> Option<Result<Value, NativeError>> {
    if is_instance_of(eval, "clojure.lang.IPending", target) {
        Some(Ok(Value::Bool(realized_flag(target))))
    } else {
        None
    }
}

/// `close` on writers, stream readers, and facsimile streams
/// (void — callable, never a bean key).
fn gated_writer_close(eval: &mut Evaluator, target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Extension(e) => crate::io::close_writer_value(eval, e)
            .or_else(|| crate::reader::close_reader_value(e))
            .or_else(|| crate::jio::close_stream_value(e)),
        _ => None,
    }
}

/// `flush` on writers and facsimile output streams (void —
/// callable, never a bean key).
fn gated_writer_flush(eval: &mut Evaluator, target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Extension(e) => {
            crate::io::flush_writer_value(eval, e).or_else(|| crate::jio::flush_stream_value(e))
        }
        _ => None,
    }
}

/// The boxArg `int` coercion (upstream
/// `Reflector.java#boxArg` — every `String` member below is the
/// lone overload at its arity, so there is no congruence
/// filter): every `Number` truncates through `intValue` (longs
/// keep the low 32 bits, doubles saturate-and-truncate like a
/// Java `(int)` cast, tower values go through the shared
/// low-bits helper); nil trips the `arg.getClass()` NPE inside
/// the `Unexpected param type` throw, anything else is that
/// IAE.
fn str_int_arg(v: &Value) -> Result<i32, NativeError> {
    match v {
        Value::Int(i) => Ok(*i as i32),
        Value::Double(d) => Ok(*d as i32),
        Value::BigInt(n) | Value::BigInteger(n) => Ok(crate::core::bigint_int_value(n)),
        Value::Ratio(r) => Ok(r.to_f64().unwrap_or(f64::NAN) as i32),
        Value::BigDec(d) => Ok(crate::core::bigint_int_value(&d.to_bigint_trunc())),
        Value::Nil => Err(boxarg_npe()),
        _ => Err(unexpected_param(v)),
    }
}

/// The boxArg `String` coercion (upstream
/// `Reflector.java#boxArg` casts): strings pass, nil passes
/// (the method body NPEs with its own text), anything else
/// fails the `String` cast (the port's short shape for the
/// JVM's module shape — the `m-cls-assign-nonclass`
/// precedent).
fn str_text_arg(v: &Value) -> Result<Option<&str>, NativeError> {
    match v {
        Value::Str(s) => Ok(Some(s)),
        Value::Nil => Ok(None),
        other => Err(NativeError::cast(&jvm_class_of(other), "java.lang.String")),
    }
}

/// The boxArg nil-for-`int` NPE (upstream
/// `Reflector.java#boxArg` evaluates `arg.getClass()` while
/// building the `Unexpected param type` throw, so nil reports
/// the `arg` receiver).
fn boxarg_npe() -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        "Cannot invoke \"Object.getClass()\" because \"arg\" is null".to_owned(),
    )
}

/// The boxArg non-`Number` for-`int` failure (upstream
/// `Reflector.java#boxArg` text, verbatim).
fn unexpected_param(v: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!(
            "Unexpected param type, expected: int, given: {}",
            jvm_class_of(v)
        ),
    )
}

/// The JDK `checkIndex` template (`charAt` — upstream
/// `Preconditions.java#outOfBoundsMessage`, `checkIndex` arm).
fn sioobe_index(index: i32, len: usize) -> NativeError {
    NativeError::new(
        ThrowClass::StringIndexOutOfBoundsException,
        format!("Index {index} out of bounds for length {len}"),
    )
}

/// The JDK `checkFromToIndex` template (`substring` — upstream
/// `Preconditions.java#outOfBoundsMessage`; the `subs` twin).
fn sioobe_range(begin: i64, end: i64, len: i64) -> NativeError {
    NativeError::new(
        ThrowClass::StringIndexOutOfBoundsException,
        format!("Range [{begin}, {end}) out of bounds for length {len}"),
    )
}

/// The method-body nil-`String` NPE (the body dereferences the
/// parameter first — upstream `String.java#startsWith`,
/// `#endsWith`, and `#split` all read `param.length()`).
fn str_body_npe(param: &str) -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        format!("Cannot invoke \"String.length()\" because \"{param}\" is null"),
    )
}

/// `length` (UTF-16 units — the `count` twin).
fn gated_str_length(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(Ok(Value::Int(s.encode_utf16().count() as i64))),
        _ => None,
    }
}

/// `trim` (leading/trailing codepoints `<= U+0020` only —
/// NBSP and friends stay, unlike `strip`/Rust `trim`).
fn gated_str_trim(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(Ok(Value::Str(Arc::from(s.trim_matches(|c| c <= '\u{20}'))))),
        _ => None,
    }
}

/// `toUpperCase` (the Unicode default mapping — the port has no
/// locales, so `tr`/`az`/`lt` rules and context-sensitive
/// mappings like final sigma stay unmapped).
fn gated_str_upper(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(Ok(Value::Str(Arc::from(
            s.chars().flat_map(char::to_uppercase).collect::<String>(),
        )))),
        _ => None,
    }
}

/// `toLowerCase` (same locale narrowing as `toUpperCase`).
fn gated_str_lower(target: &Value) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(Ok(Value::Str(Arc::from(
            s.chars().flat_map(char::to_lowercase).collect::<String>(),
        )))),
        _ => None,
    }
}

/// `charAt` (UTF-16 units, like `length`; astral halves answer
/// the unit as an int — a lone surrogate is no `Char`, so
/// `(int (charAt ...))` reads the half, issue #201 S4).
fn gated_str_charat(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(str_char_at(s, &rest[0])),
        _ => None,
    }
}

/// The `charAt` body (split for the complexity budget).
fn str_char_at(s: &str, v: &Value) -> Result<Value, NativeError> {
    let index = str_int_arg(v)?;
    let units: Vec<u16> = s.encode_utf16().collect();
    let unit = usize::try_from(index)
        .ok()
        .and_then(|i| units.get(i).copied());
    match unit {
        Some(u) => Ok(utf16_unit_value(u)),
        None => Err(sioobe_index(index, units.len())),
    }
}

/// A UTF-16 unit as a value: scalars read `Char`, surrogate
/// halves the unit as an int (a lone surrogate is no `Char`,
/// so `(int (charAt ...))` reads the half — issue #201 S4).
fn utf16_unit_value(u: u16) -> Value {
    char::from_u32(u32::from(u)).map_or(Value::Int(i64::from(u)), Value::Char)
}

/// `substring/1,2` (char-indexed — the `subs` twin, same
/// astral divergence; the JDK `checkFromToIndex` template).
fn gated_str_substring(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(str_substring(s, rest)),
        _ => None,
    }
}

/// The `substring` body (split for the complexity budget).
fn str_substring(s: &str, rest: &[Value]) -> Result<Value, NativeError> {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len() as i64;
    let begin = i64::from(str_int_arg(&rest[0])?);
    let end = if rest.len() == 2 {
        i64::from(str_int_arg(&rest[1])?)
    } else {
        len
    };
    if begin < 0 || begin > end || end > len {
        return Err(sioobe_range(begin, end, len));
    }
    Ok(Value::Str(Arc::from(
        chars[begin as usize..end as usize]
            .iter()
            .collect::<String>(),
    )))
}

/// `startsWith/1,2` (UTF-16 units, like `length`; a negative
/// `toffset` is false before the nil-prefix NPE — the JDK
/// short-circuit in `String.java#startsWith`).
fn gated_str_startswith(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(str_starts_with(s, rest)),
        _ => None,
    }
}

/// The `startsWith` body (split for the complexity budget).
fn str_starts_with(s: &str, rest: &[Value]) -> Result<Value, NativeError> {
    let prefix = str_text_arg(&rest[0])?;
    let to = if rest.len() == 2 {
        str_int_arg(&rest[1])?
    } else {
        0
    };
    if to < 0 {
        return Ok(Value::Bool(false));
    }
    let Some(prefix) = prefix else {
        return Err(str_body_npe("prefix"));
    };
    Ok(Value::Bool(units_start_with(s, prefix, to)))
}

/// The unit-slice prefix test (the `startsWith` body past the
/// bounds check — `to >= 0` by construction).
fn units_start_with(s: &str, prefix: &str, to: i32) -> bool {
    let text: Vec<u16> = s.encode_utf16().collect();
    let pre: Vec<u16> = prefix.encode_utf16().collect();
    if i64::from(to) > text.len() as i64 - pre.len() as i64 {
        return false;
    }
    text[to as usize..].starts_with(&pre)
}

/// `endsWith` (literally `startsWith(suffix, len - suflen)` —
/// upstream `String.java#endsWith`, including the `suffix`
/// receiver on the nil NPE).
fn gated_str_endswith(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(str_ends_with(s, &rest[0])),
        _ => None,
    }
}

/// The `endsWith` body (split for the complexity budget).
fn str_ends_with(s: &str, v: &Value) -> Result<Value, NativeError> {
    let Some(suffix) = str_text_arg(v)? else {
        return Err(str_body_npe("suffix"));
    };
    let to = s.encode_utf16().count() as i64 - suffix.encode_utf16().count() as i64;
    if to < 0 {
        return Ok(Value::Bool(false));
    }
    Ok(Value::Bool(units_start_with(s, suffix, to as i32)))
}

/// `split/1,2` (limit `0` drops trailing empties, positive caps
/// the parts, negative keeps all — upstream
/// `Pattern.java#split`; the port has no arrays, so the parts
/// come back as a vector, like `clojure.string/split`).
fn gated_str_split(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    match target {
        Value::Str(s) => Some(str_split(s, rest)),
        _ => None,
    }
}

/// The `split` body (split for the complexity budget).
fn str_split(s: &str, rest: &[Value]) -> Result<Value, NativeError> {
    let Some(regex) = str_text_arg(&rest[0])? else {
        return Err(str_body_npe("regex"));
    };
    let limit = if rest.len() == 2 {
        str_int_arg(&rest[1])?
    } else {
        0
    };
    crate::re::pattern_split(s, regex, limit)
}

/// The 0-arg string members (`length`, `trim`, case).
fn surface_gated_str0(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("length", 0) => gated_str_length(target),
        ("trim", 0) => gated_str_trim(target),
        ("toUpperCase", 0) => gated_str_upper(target),
        ("toLowerCase", 0) => gated_str_lower(target),
        _ => None,
    }
}

/// The string members with arguments.
fn surface_gated_strn(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("charAt", 1) => gated_str_charat(target, rest),
        ("substring", 1 | 2) => gated_str_substring(target, rest),
        // `subSequence` shares the substring body (same bounds
        // template; `CharSequence` results are port strings).
        ("subSequence", 2) => gated_str_substring(target, rest),
        ("startsWith", 1 | 2) => gated_str_startswith(target, rest),
        ("endsWith", 1) => gated_str_endswith(target, rest),
        ("split", 1 | 2) => gated_str_split(target, rest),
        ("indexOf", 1) => gated_str_indexof(target, rest),
        ("indexOf", 2) => gated_str_indexof_from(target, rest),
        ("lastIndexOf", 1) => gated_str_last_indexof(target, rest),
        _ => surface_gated_strn_rest(meth, target, rest),
    }
}

/// The tail half of [`surface_gated_strn`]'s method dispatch.
fn surface_gated_strn_rest(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("lastIndexOf", 2) => gated_str_last_indexof_from(target, rest),
        ("contains", 1) => gated_str_contains(target, rest),
        ("replace", 2) => gated_str_replace(target, rest),
        ("getBytes", 0 | 1) => gated_str_getbytes(target, rest),
        ("codePointCount", 2) => gated_str_codepoint_count(target, rest),
        ("compareTo", 1) => gated_str_compare_to(target, rest),
        _ => None,
    }
}

/// `String.indexOf` (1-arg): the first UTF-16 offset of the
/// `String` (or code-point) needle, -1 absent. Empty needles
/// read 0 (the JDK rule).
fn gated_str_indexof(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    let needle = str_needle(&rest[0])?;
    Some(Ok(Value::Int(str_index_search(s, needle, 0))))
}

/// `String.lastIndexOf` (1-arg, #205): the last UTF-16 offset
/// of the needle, -1 absent. Empty needles read the length
/// (the JDK rule).
fn gated_str_last_indexof(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    let needle = str_needle(&rest[0])?;
    let len = s.encode_utf16().count() as i32;
    Some(Ok(Value::Int(str_last_index_search(s, needle, len))))
}

/// `String.lastIndexOf` (2-arg, #205): the backward scan from
/// `fromIndex` (the JDK clamp rules — see
/// `str_last_index_search`). The needle validates before the
/// index, like the `indexOf` pair.
fn gated_str_last_indexof_from(
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    let needle = str_needle(&rest[0])?;
    let from = match str_int_arg(&rest[1]) {
        Ok(from) => from,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(Value::Int(str_last_index_search(s, needle, from))))
}

/// `String.indexOf` (2-arg, #205): the same scan from
/// `fromIndex` (negative reads 0; past-the-end reads the
/// length for empty needles, -1 otherwise). The needle
/// validates before the index (a mistyped pair is the
/// overload miss, like the 1-arg shape).
fn gated_str_indexof_from(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    let needle = str_needle(&rest[0])?;
    let from = match str_int_arg(&rest[1]) {
        Ok(from) => from,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(Value::Int(str_index_search(s, needle, from))))
}

/// An `indexOf` needle: UTF-16 units, or `Absent` for an
/// unrepresentable code point (no match, like the JDK).
enum StrNeedle {
    Units(Vec<u16>),
    Absent,
}

/// Parse an `indexOf` needle (`String`, code-point `Int`, or
/// `Char`): `None` for a mistyped needle (the generic miss).
fn str_needle(v: &Value) -> Option<StrNeedle> {
    match v {
        Value::Str(p) => Some(StrNeedle::Units(p.encode_utf16().collect())),
        Value::Int(i) => match u32::try_from(*i).ok().and_then(char::from_u32) {
            Some(c) => Some(StrNeedle::Units(c.encode_utf16(&mut [0; 2]).to_vec())),
            None => Some(StrNeedle::Absent),
        },
        Value::Char(c) => Some(StrNeedle::Units(c.encode_utf16(&mut [0; 2]).to_vec())),
        _ => None,
    }
}

/// The `indexOf` scan over UTF-16 units from `from` (the JDK
/// `fromIndex` rules — see `gated_str_indexof_from`).
fn str_index_search(s: &str, needle: StrNeedle, from: i32) -> i64 {
    let hay: Vec<u16> = s.encode_utf16().collect();
    let from = from.max(0) as usize;
    let units = match needle {
        StrNeedle::Absent => return -1,
        StrNeedle::Units(units) => units,
    };
    if units.is_empty() {
        return from.min(hay.len()) as i64;
    }
    if from > hay.len() {
        return -1;
    }
    hay[from..]
        .windows(units.len())
        .position(|w| w == units.as_slice())
        .map(|i| (from + i) as i64)
        .unwrap_or(-1)
}

/// The `lastIndexOf` scan over UTF-16 units (the JDK clamp:
/// `fromIndex` past the last start clamps, negative reads -1,
/// empty needles read the clamped start).
fn str_last_index_search(s: &str, needle: StrNeedle, from: i32) -> i64 {
    let hay: Vec<u16> = s.encode_utf16().collect();
    let units = match needle {
        StrNeedle::Absent => return -1,
        StrNeedle::Units(units) => units,
    };
    let last_start = hay.len() as i64 - units.len() as i64;
    let from = (from as i64).min(last_start);
    if from < 0 {
        return -1;
    }
    if units.is_empty() {
        return from;
    }
    let from = from as usize;
    hay[..=from + units.len() - 1]
        .windows(units.len())
        .rposition(|w| w == units.as_slice())
        .map(|i| i as i64)
        .unwrap_or(-1)
}

/// `String.contains` (#205): a `CharSequence` substring test
/// (literal, not regex). Nil is the body's NPE; anything else
/// mistyped is the overload miss.
fn gated_str_contains(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    match &rest[0] {
        Value::Str(p) => Some(Ok(Value::Bool(s.contains(p.as_ref())))),
        Value::Nil => Some(Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.lang.CharSequence.toString()\" because \"s\" is null".to_owned(),
        ))),
        _ => None,
    }
}

/// `String.replace` (#205): the `(char, char)` and
/// `(CharSequence, CharSequence)` overloads (both literal —
/// regex lives in `replaceAll`). Nil halves are the body's
/// NPEs; mixed/mistyped pairs are the overload miss.
fn gated_str_replace(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    match (&rest[0], &rest[1]) {
        (Value::Char(a), Value::Char(b)) => {
            let out: String = s.chars().map(|c| if c == *a { *b } else { c }).collect();
            Some(Ok(Value::Str(Arc::from(out.as_str()))))
        }
        (Value::Str(a), Value::Str(b)) => Some(Ok(Value::Str(Arc::from(
            s.replace(a.as_ref(), b.as_ref()).as_str(),
        )))),
        (Value::Nil, _) => Some(Err(str_replace_npe("target"))),
        (_, Value::Nil) => Some(Err(str_replace_npe("replacement"))),
        _ => None,
    }
}

/// The `replace` body's nil-half NPE (verbatim shape).
fn str_replace_npe(param: &str) -> NativeError {
    NativeError::new(
        ThrowClass::NullPointerException,
        format!("Cannot invoke \"java.lang.CharSequence.toString()\" because \"{param}\" is null"),
    )
}

/// `String.getBytes` (#205): the platform-default (UTF-8)
/// bytes, or the named charset's (`MessageDigest` feeds on
/// these). Bare `UTF-16`/`UTF-32` take the BOM, like the JDK;
/// unknown charsets are `UnsupportedEncodingException`, and a
/// nil name is ambiguous (`(String)` vs `(Charset)`).
fn gated_str_getbytes(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    if rest.is_empty() {
        return Some(Ok(str_bytes_value(crate::ctor::encode(
            crate::ctor::Charset::Utf8,
            s,
            false,
        ))));
    }
    match &rest[0] {
        Value::Str(name) => {
            let cs = match crate::ctor::lookup_charset(name) {
                Ok(cs) => cs,
                Err(e) => return Some(Err(e)),
            };
            Some(Ok(str_bytes_value(crate::ctor::encode(cs, s, true))))
        }
        Value::Nil => Some(Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "More than one matching method found: getBytes".to_owned(),
        ))),
        _ => None,
    }
}

/// Mint a `byte[]` from encoded bytes.
fn str_bytes_value(bytes: Vec<u8>) -> Value {
    let cells: Vec<i8> = bytes.into_iter().map(|b| b as i8).collect();
    Value::Array(Array::new(
        ArrayDesc::flat(BaseType::Byte),
        ArrayStore::Bytes(cells),
    ))
}

/// `String.codePointCount` (#205): code points over the
/// UTF-16 `[begin, end)` window (surrogate pairs count once).
/// Bounds are the `substring` template.
fn gated_str_codepoint_count(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    let (begin, end) = match (str_int_arg(&rest[0]), str_int_arg(&rest[1])) {
        (Ok(begin), Ok(end)) => (i64::from(begin), i64::from(end)),
        (Err(e), _) | (_, Err(e)) => return Some(Err(e)),
    };
    let units: Vec<u16> = s.encode_utf16().collect();
    let len = units.len() as i64;
    if begin < 0 || end > len || begin > end {
        return Some(Err(sioobe_range(begin, end, len)));
    }
    let (begin, end) = (begin as usize, end as usize);
    let mut count: i64 = (end - begin) as i64;
    for pair in units[begin..end].windows(2) {
        if (0xD800..0xDC00).contains(&pair[0]) && (0xDC00..0xE000).contains(&pair[1]) {
            count -= 1;
        }
    }
    Some(Ok(Value::Int(count)))
}

/// `String.compareTo` (#205): lexicographic by UTF-16 units
/// (first difference, else the length difference). Nil is the
/// body's NPE; non-strings are the overload miss.
fn gated_str_compare_to(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Str(s) = target else {
        return None;
    };
    match &rest[0] {
        Value::Str(o) => {
            let (a, b): (Vec<u16>, Vec<u16>) =
                (s.encode_utf16().collect(), o.encode_utf16().collect());
            let width = a.len().min(b.len());
            for i in 0..width {
                if a[i] != b[i] {
                    return Some(Ok(Value::Int(i64::from(a[i]) - i64::from(b[i]))));
                }
            }
            Some(Ok(Value::Int(a.len() as i64 - b.len() as i64)))
        }
        Value::Nil => Some(Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"String.compareTo(String)\" because \"anotherString\" is null"
                .to_owned(),
        ))),
        _ => None,
    }
}

/// The string-gated members (split 0-arg/with-args for the
/// complexity budget; arms are disjoint).
fn surface_gated_str(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    surface_gated_str0(meth, target, rest).or_else(|| surface_gated_strn(meth, target, rest))
}

/// The `BigInteger` members (#205): arithmetic, shifts,
/// modular exponentiation, and conversions over
/// `Value::BigInteger` targets (strict — `BigInt` twins take
/// the generic miss, like the JVM, which has no such
/// methods on `clojure.lang.BigInt`).
fn surface_gated_bigint(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    match (meth, rest.len()) {
        ("add" | "subtract" | "multiply" | "divide", 1) => gated_bigint_binop(meth, target, rest),
        ("and" | "or" | "xor", 1) => gated_bigint_bitop(meth, target, rest),
        ("not", 0) => gated_bigint_not(target),
        ("shiftLeft" | "shiftRight", 1) => gated_bigint_shift(meth, target, rest),
        ("modPow", 2) => gated_bigint_modpow(target, rest),
        ("mod" | "modInverse", 1) => gated_bigint_mod(meth, target, rest),
        ("pow", 1) => gated_bigint_pow(target, rest),
        ("compareTo", 1) => gated_bigint_compare(target, rest),
        ("signum" | "bitLength", 0) => gated_bigint_measure(meth, target),
        ("testBit" | "setBit", 1) => gated_bigint_bit(meth, target, rest),
        ("toByteArray" | "intValue" | "longValue" | "longValueExact", 0) => {
            gated_bigint_narrow(meth, target)
        }
        ("toString", 1) => gated_bigint_tostring(target, rest),
        _ => None,
    }
}

/// The `BigInteger` target, if it is one.
fn bigint_self(target: &Value) -> Option<&num_bigint::BigInt> {
    match target {
        Value::BigInteger(n) => Some(n),
        _ => None,
    }
}

/// A `BigInteger` parameter (nil is the body's NPE, anything
/// else mistyped the overload miss at the call's arity).
fn bigint_param<'a>(
    v: &'a Value,
    meth: &str,
    argc: usize,
) -> Result<&'a num_bigint::BigInt, NativeError> {
    match v {
        Value::BigInteger(n) => Ok(n),
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            format!("Cannot invoke \"java.math.BigInteger.{meth}\" because \"val\" is null"),
        )),
        _ => Err(crate::types::no_method(meth, argc, "java.math.BigInteger")),
    }
}

/// Mint a `BigInteger` value.
fn bigint_value(n: num_bigint::BigInt) -> Value {
    Value::BigInteger(Arc::new(n))
}

/// `add`/`subtract`/`multiply`/`divide` (division truncates
/// toward zero, like `/`; zero divisors are the JDK text).
fn gated_bigint_binop(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let y = match bigint_param(&rest[0], meth, 1) {
        Ok(y) => y,
        Err(e) => return Some(Err(e)),
    };
    let out = match meth {
        "add" => x + y,
        "subtract" => x - y,
        "multiply" => x * y,
        _ => {
            if y.sign() == num_bigint::Sign::NoSign {
                return Some(Err(crate::core::arithmetic("BigInteger divide by zero")));
            }
            x / y
        }
    };
    Some(Ok(bigint_value(out)))
}

/// `and`/`or`/`xor` (infinite two's complement, like the JDK).
fn gated_bigint_bitop(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let y = match bigint_param(&rest[0], meth, 1) {
        Ok(y) => y,
        Err(e) => return Some(Err(e)),
    };
    let out = match meth {
        "and" => x & y,
        "or" => x | y,
        _ => x ^ y,
    };
    Some(Ok(bigint_value(out)))
}

/// `not` (`-x-1`, the infinite complement).
fn gated_bigint_not(target: &Value) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    Some(Ok(bigint_value(-x - 1)))
}

/// `shiftLeft`/`shiftRight` (negative distances flip, like the
/// JDK — there is no `>>>` on `BigInteger`).
fn gated_bigint_shift(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let dist = match str_int_arg(&rest[0]) {
        Ok(dist) => dist,
        Err(e) => return Some(Err(e)),
    };
    let left = (meth == "shiftLeft") != (dist < 0);
    let dist = dist.unsigned_abs();
    Some(Ok(bigint_value(if left { x << dist } else { x >> dist })))
}

/// `modPow(exponent, modulus)` (guards first — the power
/// routine panics on negative exponents and zero moduli).
fn gated_bigint_modpow(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let exp = match bigint_param(&rest[0], "modPow", 2) {
        Ok(exp) => exp,
        Err(e) => return Some(Err(e)),
    };
    let modulus = match bigint_param(&rest[1], "modPow", 2) {
        Ok(modulus) => modulus,
        Err(e) => return Some(Err(e)),
    };
    if exp.sign() == num_bigint::Sign::Minus {
        return Some(Err(crate::core::arithmetic(
            "BigInteger: negative exponent",
        )));
    }
    if modulus.sign() != num_bigint::Sign::Plus {
        return Some(Err(crate::core::arithmetic(
            "BigInteger: modulus not positive",
        )));
    }
    Some(Ok(bigint_value(x.modpow(exp, modulus))))
}

/// `mod` (floored, non-negative for positive moduli) and
/// `modInverse` (in `[0, m)`, absent when not coprime).
fn gated_bigint_mod(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let m = match bigint_param(&rest[0], meth, 1) {
        Ok(m) => m,
        Err(e) => return Some(Err(e)),
    };
    if m.sign() != num_bigint::Sign::Plus {
        return Some(Err(crate::core::arithmetic(
            "BigInteger: modulus not positive",
        )));
    }
    if meth == "mod" {
        let r = x % m;
        let out = if r.sign() == num_bigint::Sign::Minus {
            r + m
        } else {
            r
        };
        return Some(Ok(bigint_value(out)));
    }
    match x.modinv(m) {
        Some(inv) => Some(Ok(bigint_value(inv))),
        None => Some(Err(crate::core::arithmetic("BigInteger not invertible"))),
    }
}

/// `pow` (negative exponents are the JDK text).
fn gated_bigint_pow(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let exp = match str_int_arg(&rest[0]) {
        Ok(exp) => exp,
        Err(e) => return Some(Err(e)),
    };
    if exp < 0 {
        return Some(Err(crate::core::arithmetic("Negative exponent")));
    }
    Some(Ok(bigint_value(x.pow(exp as u32))))
}

/// `compareTo` (the `BigInt` order).
fn gated_bigint_compare(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let y = match bigint_param(&rest[0], "compareTo", 1) {
        Ok(y) => y,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(Value::Int(match x.cmp(y) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    })))
}

/// `signum` and `bitLength` (negatives measure `~x`, like the
/// JDK's minimal-complement rule).
fn gated_bigint_measure(meth: &str, target: &Value) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    if meth == "signum" {
        return Some(Ok(Value::Int(match x.sign() {
            num_bigint::Sign::Minus => -1,
            num_bigint::Sign::NoSign => 0,
            num_bigint::Sign::Plus => 1,
        })));
    }
    let bits = if x.sign() == num_bigint::Sign::Minus {
        let comp: num_bigint::BigInt = -x - 1;
        comp.bits()
    } else {
        x.bits()
    };
    Some(Ok(Value::Int(bits as i64)))
}

/// `testBit`/`setBit` (two's complement; negative indices are
/// the JDK text).
fn gated_bigint_bit(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let bit = match str_int_arg(&rest[0]) {
        Ok(bit) => bit,
        Err(e) => return Some(Err(e)),
    };
    if bit < 0 {
        return Some(Err(crate::core::arithmetic("Negative bit address")));
    }
    if meth == "testBit" {
        return Some(Ok(Value::Bool(x.bit(bit as u64))));
    }
    let mut out = x.clone();
    out.set_bit(bit as u64, true);
    Some(Ok(bigint_value(out)))
}

/// `toByteArray` (minimal two's complement) and the narrowing
/// conversions (`intValue`/`longValue` wrap; `longValueExact`
/// is the JDK text outside `long` range).
fn gated_bigint_narrow(meth: &str, target: &Value) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    match meth {
        "toByteArray" => Some(Ok(str_bytes_value(x.to_signed_bytes_be()))),
        "intValue" => Some(Ok(Value::Int(i64::from(bigint_wrap_i32(x))))),
        "longValue" => Some(Ok(Value::Int(bigint_wrap_i64(x)))),
        _ => match x.to_i64() {
            Some(n) => Some(Ok(Value::Int(n))),
            None => Some(Err(crate::core::arithmetic("BigInteger out of long range"))),
        },
    }
}

/// The low 32 bits as an `i32` (wrapping, like the JDK).
fn bigint_wrap_i32(x: &num_bigint::BigInt) -> i32 {
    let mut le = x.to_signed_bytes_le();
    le.resize(
        4,
        if x.sign() == num_bigint::Sign::Minus {
            0xFF
        } else {
            0
        },
    );
    i32::from_le_bytes([le[0], le[1], le[2], le[3]])
}

/// The low 64 bits as an `i64` (wrapping, like the JDK).
fn bigint_wrap_i64(x: &num_bigint::BigInt) -> i64 {
    let mut le = x.to_signed_bytes_le();
    le.resize(
        8,
        if x.sign() == num_bigint::Sign::Minus {
            0xFF
        } else {
            0
        },
    );
    i64::from_le_bytes([le[0], le[1], le[2], le[3], le[4], le[5], le[6], le[7]])
}

/// `toString(radix)` (lowercase, like the JDK; out-of-range
/// radixes read 10, the `Long.toString` rule).
fn gated_bigint_tostring(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let x = bigint_self(target)?;
    let radix = match str_int_arg(&rest[0]) {
        Ok(radix) => radix,
        Err(e) => return Some(Err(e)),
    };
    let radix = if (2..=36).contains(&radix) {
        radix as u32
    } else {
        10
    };
    Some(Ok(Value::Str(Arc::from(x.to_str_radix(radix).as_str()))))
}

/// A `java.math.RoundingMode` enum constant (#205): fresh per
/// static-field read (like the `BigInteger` constants), so `=`
/// compares modes, not ids (the JVM hands out singletons —
/// same mode is the same constant).
#[derive(Debug)]
pub(crate) struct RoundingModeVal {
    id: u64,
    mode: RoundingMode,
}

static NEXT_ROUNDING: AtomicU64 = AtomicU64::new(1);

impl RoundingModeVal {
    pub(crate) fn value(mode: RoundingMode) -> Value {
        Value::Extension(Arc::new(Self {
            id: NEXT_ROUNDING.fetch_add(1, Ordering::Relaxed),
            mode,
        }))
    }
}

impl ExtensionValue for RoundingModeVal {
    fn class_name(&self) -> String {
        "java.math.RoundingMode".to_owned()
    }

    fn print(&self) -> String {
        format!(
            "#object[java.math.RoundingMode 0x{:x} \"{}\"]",
            self.id,
            self.mode.name()
        )
    }

    fn print_str(&self) -> Option<String> {
        Some(self.mode.name().to_owned())
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<RoundingModeVal>()
            .is_some_and(|o| o.mode == self.mode)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write(self.mode.name().as_bytes());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Downcast a value to a rounding mode, if it is one.
pub(crate) fn as_rounding(v: &Value) -> Option<RoundingMode> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<RoundingModeVal>().map(|r| r.mode),
        _ => None,
    }
}

/// Downcast a value to a math context, if it is one (reuses the
/// `with-precision` value — same class, same shape).
fn as_math_context(v: &Value) -> Option<MathContext> {
    match v {
        Value::Extension(e) => e
            .as_any()
            .downcast_ref::<crate::core::MathContextVal>()
            .map(|m| m.ctx),
        _ => None,
    }
}

/// The `BigDec` target, when the value is one.
fn bigdec_self(target: &Value) -> Option<&BigDec> {
    match target {
        Value::BigDec(d) => Some(d),
        _ => None,
    }
}

/// A `BigDecimal` method arg: anything else mistypes the
/// overload miss (only `BigDecimal` overloads exist).
fn bigdec_param<'a>(v: &'a Value, meth: &str, argc: usize) -> Result<&'a BigDec, NativeError> {
    match v {
        Value::BigDec(d) => Ok(d),
        _ => Err(crate::types::no_method(meth, argc, "java.math.BigDecimal")),
    }
}

/// A `MathContext` method arg: nil is the bare NPE (the body
/// derefs it); anything else mistypes the overload miss.
fn mc_param(v: &Value, meth: &str, argc: usize) -> Result<MathContext, NativeError> {
    match v {
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            String::new(),
        )),
        _ => as_math_context(v)
            .ok_or_else(|| crate::types::no_method(meth, argc, "java.math.BigDecimal")),
    }
}

/// A `RoundingMode` method arg: nil is the bare NPE; anything
/// else mistypes the overload miss.
fn rounding_param(v: &Value, meth: &str, argc: usize) -> Result<RoundingMode, NativeError> {
    match v {
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            String::new(),
        )),
        _ => as_rounding(v)
            .ok_or_else(|| crate::types::no_method(meth, argc, "java.math.BigDecimal")),
    }
}

/// A tower arithmetic failure as the JDK error: the tower's
/// divide-by-zero text is the Clojure shape, the method needs
/// the JDK's; every other message already matches.
fn bigdec_arith(err: &'static str) -> NativeError {
    if err == "Divide by zero" {
        crate::core::arithmetic("Division by zero")
    } else {
        crate::core::arithmetic(err)
    }
}

fn surface_gated_bigdec(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    bigdec_self(target)?;
    match (meth, rest.len()) {
        ("add" | "subtract" | "multiply", 1) => gated_bigdec_exact(meth, target, rest),
        ("add" | "subtract" | "multiply", 2) => gated_bigdec_mc(meth, target, rest),
        ("divide", 1 | 2) => gated_bigdec_div(target, rest),
        ("sqrt", 1) => gated_bigdec_sqrt(target, rest),
        ("setScale", 2) => gated_bigdec_setscale(target, rest),
        ("toPlainString" | "doubleValue" | "toBigInteger", 0) => gated_bigdec_conv(meth, target),
        _ => None,
    }
}

/// Exact `add`/`subtract`/`multiply` (no context).
fn gated_bigdec_exact(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let x = bigdec_self(target)?;
    let y = match bigdec_param(&rest[0], meth, 1) {
        Ok(y) => y,
        Err(e) => return Some(Err(e)),
    };
    let out = match meth {
        "add" => x.add(y),
        "subtract" => x.sub(y),
        _ => x.mul(y),
    };
    Some(Ok(bigdec_result(out)))
}

/// Context `add`/`subtract`/`multiply` (chudnovsky threads one).
fn gated_bigdec_mc(
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let x = bigdec_self(target)?;
    let y = match bigdec_param(&rest[0], meth, 2) {
        Ok(y) => y,
        Err(e) => return Some(Err(e)),
    };
    let mc = match mc_param(&rest[1], meth, 2) {
        Ok(mc) => mc,
        Err(e) => return Some(Err(e)),
    };
    let out = match meth {
        "add" => x.add_mc(y, &mc),
        "subtract" => x.sub_mc(y, &mc),
        _ => x.mul_mc(y, &mc),
    };
    Some(out.map(bigdec_result).map_err(bigdec_arith))
}

/// `divide`, exact or under a context.
fn gated_bigdec_div(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let x = bigdec_self(target)?;
    let y = match bigdec_param(&rest[0], "divide", rest.len()) {
        Ok(y) => y,
        Err(e) => return Some(Err(e)),
    };
    if rest.len() == 1 {
        return Some(x.div_exact(y).map(bigdec_result).map_err(bigdec_arith));
    }
    let mc = match mc_param(&rest[1], "divide", 2) {
        Ok(mc) => mc,
        Err(e) => return Some(Err(e)),
    };
    Some(x.div_mc(y, &mc).map(bigdec_result).map_err(bigdec_arith))
}

/// `sqrt(mc)`.
fn gated_bigdec_sqrt(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let x = bigdec_self(target)?;
    let mc = match mc_param(&rest[0], "sqrt", 1) {
        Ok(mc) => mc,
        Err(e) => return Some(Err(e)),
    };
    Some(x.sqrt_mc(&mc).map(bigdec_result).map_err(bigdec_arith))
}

/// `setScale(scale, rounding)` (same scale answers the target
/// itself, like the JVM).
fn gated_bigdec_setscale(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let x = bigdec_self(target)?;
    let scale = match str_int_arg(&rest[0]) {
        Ok(scale) => scale as i64,
        Err(e) => return Some(Err(e)),
    };
    if scale == x.scale() {
        return Some(Ok(target.clone()));
    }
    let mode = match rounding_param(&rest[1], "setScale", 2) {
        Ok(mode) => mode,
        Err(e) => return Some(Err(e)),
    };
    Some(
        x.set_scale(scale, mode)
            .map(bigdec_result)
            .map_err(bigdec_arith),
    )
}

/// `toPlainString`/`doubleValue`/`toBigInteger` (`toString`
/// rides the universal gate, like `StringBuilder`).
fn gated_bigdec_conv(meth: &str, target: &Value) -> Option<Result<Value, NativeError>> {
    let x = bigdec_self(target)?;
    Some(Ok(match meth {
        "toPlainString" => Value::Str(Arc::from(x.to_plain_string().as_str())),
        "doubleValue" => Value::Double(x.to_f64()),
        // The caller admits only the three conversions, so the
        // wildcard is `toBigInteger`.
        _ => Value::BigInteger(Arc::new(x.to_bigint_trunc())),
    }))
}

/// A `java.lang.StringBuilder` (issue #201 S5): a growable
/// buffer behind the `Extension` facade. Identity is the
/// sequence `id` (like `Writer`); `str`/`toString` render the
/// buffer via `print_str`, `print` keeps the `#object[…]` form.
#[derive(Debug)]
pub(crate) struct StringBuilder {
    id: u64,
    buf: Mutex<String>,
}

static NEXT_BUILDER: AtomicU64 = AtomicU64::new(1);

/// The text of a `CharSequence` argument (strings and
/// `StringBuilder`s; `None` for anything else).
pub(crate) fn char_sequence_text(v: &Value) -> Option<Arc<str>> {
    match v {
        Value::Str(s) => Some(Arc::clone(s)),
        Value::Extension(e) => e
            .as_any()
            .downcast_ref::<StringBuilder>()
            .map(|sb| Arc::from(sb.lock().as_str())),
        _ => None,
    }
}

impl StringBuilder {
    pub(crate) fn fresh(content: String) -> Self {
        Self {
            id: NEXT_BUILDER.fetch_add(1, Ordering::Relaxed),
            buf: Mutex::new(content),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, String> {
        self.buf.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A `java.util.Locale` (issue #201 recipes: 44 scripts pin
/// `Locale/US` for `String/format`). Identity is the tag; `str`
/// renders the JVM `toString` (`en_US`), `print` the
/// `#object[…]` form.
#[derive(Debug)]
pub(crate) struct LocaleHandle {
    tag: &'static str,
}

impl LocaleHandle {
    /// The `Locale/US` constant (`en_US`).
    fn us() -> Self {
        Self { tag: "en_US" }
    }
}

/// Downcast a locale value to its handle, if any.
fn as_locale(v: &Value) -> Option<&LocaleHandle> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<LocaleHandle>(),
        _ => None,
    }
}

impl ExtensionValue for LocaleHandle {
    fn class_name(&self) -> String {
        "java.util.Locale".to_owned()
    }

    fn print(&self) -> String {
        format!(
            "#object[java.util.Locale 0x{:x} \"{}\"]",
            self as *const Self as usize, self.tag
        )
    }

    fn print_str(&self) -> Option<String> {
        Some(self.tag.to_owned())
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<LocaleHandle>()
            .is_some_and(|o| o.tag == self.tag)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write(self.tag.as_bytes());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A `java.net.URI` (#205): the raw string plus its parsed
/// host (`None` when the URI has no registry authority —
/// `getHost` answers nil, like the JDK). Identity is the
/// string; `str` renders it.
#[derive(Debug)]
pub(crate) struct UriHandle {
    raw: String,
    host: Option<String>,
}

impl UriHandle {
    /// Parse `raw`: illegal characters fail with the JDK's
    /// `URISyntaxException` shape (`Illegal character … at
    /// index N: <input>`); the host is the `://` authority
    /// minus userinfo/port/brackets (nil without `://`).
    pub(crate) fn parse(raw: &str) -> Result<Self, NativeError> {
        for (i, c) in raw.char_indices() {
            if c.is_control()
                || matches!(
                    c,
                    ' ' | '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`'
                )
            {
                return Err(NativeError::new(
                    ThrowClass::URISyntaxException,
                    format!("Illegal character in path at index {i}: {raw}"),
                ));
            }
        }
        let host = raw
            .split_once("://")
            .map(|(_, rest)| rest)
            .and_then(|rest| rest.split(['/', '?', '#']).next())
            .and_then(|auth| auth.rsplit('@').next())
            .map(|hostport| {
                if let Some(rest) = hostport.strip_prefix('[') {
                    rest.split(']').next().unwrap_or(rest).to_owned()
                } else {
                    hostport.split(':').next().unwrap_or(hostport).to_owned()
                }
            })
            .filter(|host| !host.is_empty());
        Ok(Self {
            raw: raw.to_owned(),
            host,
        })
    }

    fn host(&self) -> Option<&str> {
        self.host.as_deref()
    }
}

/// Downcast a URI value to its handle, if any.
fn as_uri(v: &Value) -> Option<&UriHandle> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<UriHandle>(),
        _ => None,
    }
}

impl ExtensionValue for UriHandle {
    fn class_name(&self) -> String {
        "java.net.URI".to_owned()
    }

    fn print(&self) -> String {
        format!(
            "#object[java.net.URI 0x{:x} \"{}\"]",
            self as *const Self as usize, self.raw
        )
    }

    fn print_str(&self) -> Option<String> {
        Some(self.raw.clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<UriHandle>()
            .is_some_and(|o| o.raw == self.raw)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write(self.raw.as_bytes());
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ExtensionValue for StringBuilder {
    fn class_name(&self) -> String {
        "java.lang.StringBuilder".to_owned()
    }

    /// `print-object`: the `toString` (the buffer) printed
    /// readably.
    fn print(&self) -> String {
        let text = Value::Str(Arc::from(self.lock().as_str()));
        format!(
            "#object[java.lang.StringBuilder 0x{:x} {}]",
            self.id,
            clj_value::pr_to_string(&text, &clj_value::PrintOpts::pr())
        )
    }

    /// Not readably, the buffer goes in unquoted.
    fn print_str(&self) -> Option<String> {
        Some(format!(
            "#object[java.lang.StringBuilder 0x{:x} {}]",
            self.id,
            self.lock()
        ))
    }

    /// `toString` is the buffer.
    fn java_str(&self) -> Option<String> {
        Some(self.lock().clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<StringBuilder>()
            .is_some_and(|o| o.id == self.id)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_u64(self.id);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Downcast a value to a string builder, if it is one.
fn as_builder(v: &Value) -> Option<&StringBuilder> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<StringBuilder>(),
        _ => None,
    }
}

/// `append`: strings/chars push directly, nil appends `"null"`
/// (`String.valueOf`), anything else renders through `str` —
/// and returns the builder itself, like the JVM.
fn gated_builder_append(
    eval: &mut Evaluator,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    let b = as_builder(target)?;
    let text = match &rest[0] {
        Value::Nil => "null".to_owned(),
        Value::Str(s) => s.to_string(),
        Value::Char(c) => c.to_string(),
        other => match crate::core::str_of_one(eval, other) {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        },
    };
    b.lock().push_str(&text);
    Some(Ok(target.clone()))
}

/// `StringBuilder.length` (#205): UTF-16 code units (astral
/// chars count twice, like the JDK).
fn gated_builder_length(target: &Value) -> Option<Result<Value, NativeError>> {
    let b = as_builder(target)?;
    Some(Ok(Value::Int(b.lock().encode_utf16().count() as i64)))
}

/// `StringBuilder.reverse` (#205): the in-place reversal,
/// answering the builder (like `append`). Reversing by scalar
/// value matches the JDK's surrogate-preserving unit reversal
/// on every representable string (the reader rejects lone
/// surrogates, so no half survives to be split).
fn gated_builder_reverse(target: &Value) -> Option<Result<Value, NativeError>> {
    let b = as_builder(target)?;
    let mut guard = b.lock();
    let rev: String = guard.chars().rev().collect();
    *guard = rev;
    Some(Ok(target.clone()))
}

/// `Pattern/split`: the `re` split loop over the pattern source
/// (same body as `String/split`, pattern-first argument order).
fn gated_regex_split(target: &Value, rest: &[Value]) -> Option<Result<Value, NativeError>> {
    let Value::Regex(source) = target else {
        return None;
    };
    let text = match str_text_arg(&rest[0]) {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Some(Err(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke \"java.lang.CharSequence.length()\" because \"input\" is null"
                    .to_owned(),
            )));
        }
        Err(e) => return Some(Err(e)),
    };
    let limit = if rest.len() == 2 {
        match str_int_arg(&rest[1]) {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        }
    } else {
        0
    };
    Some(crate::re::pattern_split(text, source, limit))
}

/// The member surface for `__meth*`: `None` when the member misses
/// (the caller falls through to the host dispatch and its miss
/// shapes).
pub(crate) fn surface_meth(
    eval: &mut Evaluator,
    meth: &str,
    target: &Value,
    rest: &[Value],
) -> Option<Result<Value, NativeError>> {
    if let Some(hit) = surface_universal(eval, meth, target, rest) {
        return Some(hit);
    }
    if let Some(hit) = surface_class_member(eval, meth, target, rest) {
        return Some(hit);
    }
    surface_gated(eval, meth, target, rest)
}

/// The surface as `(member, argc)` pairs (sorted for the unit test's
/// nil-target sweep).
pub(crate) const SURFACE_MEMBERS: &[(&str, usize)] = &[
    ("add", 1),
    ("addFirst", 1),
    ("addLast", 1),
    ("and", 1),
    ("append", 1),
    ("bitLength", 0),
    ("cast", 1),
    ("charAt", 1),
    ("close", 0),
    ("codePointCount", 2),
    ("compareTo", 1),
    ("contains", 1),
    ("delete", 0),
    ("digest", 0),
    ("digest", 1),
    ("divide", 1),
    ("end", 0),
    ("endsWith", 1),
    ("equals", 1),
    ("exists", 0),
    ("find", 0),
    ("find", 1),
    ("flush", 0),
    ("getAbsolutePath", 0),
    ("getBytes", 0),
    ("getBytes", 1),
    ("getCause", 0),
    ("getClass", 0),
    ("getData", 0),
    ("getFirst", 0),
    ("getHost", 0),
    ("getInterfaces", 0),
    ("getKey", 0),
    ("getLast", 0),
    ("getLocalizedMessage", 0),
    ("getMessage", 0),
    ("getName", 0),
    ("getNamespace", 0),
    ("getPath", 0),
    ("getSimpleName", 0),
    ("getSuperclass", 0),
    ("getValue", 0),
    ("group", 0),
    ("group", 1),
    ("hashCode", 0),
    ("indexOf", 1),
    ("indexOf", 2),
    ("intValue", 0),
    ("isArray", 0),
    ("isAssignableFrom", 1),
    ("isBlank", 0),
    ("isEmpty", 0),
    ("isInstance", 1),
    ("isInterface", 0),
    ("isRealized", 0),
    ("lastIndexOf", 1),
    ("lastIndexOf", 2),
    ("length", 0),
    ("longValue", 0),
    ("longValueExact", 0),
    ("mod", 1),
    ("modInverse", 1),
    ("modPow", 2),
    ("multiply", 1),
    ("nextBytes", 1),
    ("nextDouble", 0),
    ("nextGaussian", 0),
    ("nextInt", 0),
    ("nextInt", 1),
    ("not", 0),
    ("offer", 1),
    ("offerFirst", 1),
    ("offerLast", 1),
    ("or", 1),
    ("peek", 0),
    ("peekFirst", 0),
    ("peekLast", 0),
    ("poll", 0),
    ("pollFirst", 0),
    ("pollLast", 0),
    ("pow", 1),
    ("readAllBytes", 0),
    ("readLine", 0),
    ("replace", 2),
    ("replaceAll", 1),
    ("reverse", 0),
    ("setBit", 1),
    ("shiftLeft", 1),
    ("shiftRight", 1),
    ("signum", 0),
    ("split", 1),
    ("split", 2),
    ("start", 0),
    ("startsWith", 1),
    ("startsWith", 2),
    ("subSequence", 2),
    ("substring", 1),
    ("substring", 2),
    ("subtract", 1),
    ("testBit", 1),
    ("toByteArray", 0),
    ("toLowerCase", 0),
    ("toString", 0),
    ("toString", 1),
    ("toUpperCase", 0),
    ("trim", 0),
    ("update", 1),
    ("write", 1),
    ("xor", 1),
];

/// Whether `meth` names any surface member (any arity).
pub(crate) fn surface_member_name(meth: &str) -> bool {
    SURFACE_MEMBERS.iter().any(|(name, _)| *name == meth)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fence 4a: kinds with no Clojure surface (upstream has no URI
    // constructor either) — same allocation true, distinct false.
    #[test]
    fn identical_unreachable_kinds() {
        let instant = Value::Instant(clj_value::Instant::from_millis(1000));
        assert!(identical_values(&instant, &instant.clone()));
        assert!(!identical_values(
            &instant,
            &Value::Instant(clj_value::Instant::from_millis(1000))
        ));
        let uri = Value::Uri(Box::new(clj_value::Uri::parse("http://x/").unwrap()));
        assert!(identical_values(&uri, &uri.clone()));
        assert!(!identical_values(
            &uri,
            &Value::Uri(Box::new(clj_value::Uri::parse("http://x/").unwrap()))
        ));
        let cond = clj_value::reader_conditional(Value::Int(1), false);
        assert!(identical_values(&cond, &cond.clone()));
        assert!(!identical_values(
            &cond,
            &clj_value::reader_conditional(Value::Int(1), false)
        ));
    }

    #[test]
    fn identity_hash_unreachable_uri() {
        // `Uri` has no surface constructor (like the unreachable
        // row above): the id tier still answers — the same
        // allocation agrees, distinct allocations differ.
        let uri = Value::Uri(Box::new(clj_value::Uri::parse("http://x/").unwrap()));
        assert_eq!(identity_hash(&uri), identity_hash(&uri.clone()));
        let other = Value::Uri(Box::new(clj_value::Uri::parse("http://x/").unwrap()));
        assert_ne!(identity_hash(&uri), identity_hash(&other));
    }

    #[test]
    fn static_or_fallback_mints_for_missing() {
        // Every reported name is tabled (§1.4), so the fallback only
        // fires defensively: a missing name mints a plain
        // `Object`-child class of that name.
        let c = static_or_fallback("missing.Name");
        assert_eq!(c.name(), "missing.Name");
        assert!(!c.is_iface());
        assert_eq!(c.bases(), &["java.lang.Object".into()]);
    }

    #[test]
    fn decapitalize_total_on_empty() {
        // Defensive arm: `bean_suffix` never yields an empty suffix
        // (it needs a first char), so the empty input only arrives
        // here — total by construction.
        assert_eq!(decapitalize(""), "");
        assert_eq!(decapitalize("URL"), "URL");
        assert_eq!(decapitalize("Count"), "count");
    }

    #[test]
    fn realized_flag_defensive_arm() {
        // Defensive arm: the IPending gate admits only
        // delays/promises/futures/extensions, so anything else only
        // arrives here — never realized.
        assert!(!realized_flag(&Value::Int(1)));
        assert!(!realized_flag(&Value::Nil));
    }

    #[test]
    fn static_arity_defensive_arms() {
        // Defensive arms: the analyzer checks arities before
        // lowering a static call, so a wrong argc only arrives
        // here — the uniform no-method miss, per member.
        let one = [Value::Int(1)];
        let two = [Value::Int(1), Value::Int(2)];
        for (member, class, err) in [
            (
                "isWhitespace",
                "java.lang.Character",
                static_character_is_whitespace(&[]),
            ),
            ("digit", "java.lang.Character", static_character_digit(&one)),
            (
                "createOwning",
                "clojure.lang.LazilyPersistentVector",
                static_lpv_create_owning(&[]),
            ),
            ("parseLong", "java.lang.Long", static_long_parse_long(&[])),
            (
                "remainderUnsigned",
                "java.lang.Long",
                static_long_remainder_unsigned(&one),
            ),
            (
                "parseInt",
                "java.lang.Integer",
                static_integer_parse_int(&[]),
            ),
            (
                "parseDouble",
                "java.lang.Double",
                static_double_parse_double(&[]),
            ),
            ("abs", "java.lang.Math", static_math_abs(&[])),
            ("sqrt", "java.lang.Math", static_math_sqrt(&[])),
            ("ceil", "java.lang.Math", static_math_ceil(&[])),
            ("log", "java.lang.Math", static_math_log(&[])),
            ("exp", "java.lang.Math", static_math_exp(&[])),
            ("pow", "java.lang.Math", static_math_pow(&one)),
            ("sin", "java.lang.Math", static_math_sin(&[])),
            ("atan2", "java.lang.Math", static_math_atan2(&one)),
            ("hypot", "java.lang.Math", static_math_hypot(&one)),
            ("round", "java.lang.Math", static_math_round(&[])),
            (
                "isLetterOrDigit",
                "java.lang.Character",
                static_character_is_letter_or_digit(&two),
            ),
            (
                "isLowerCase",
                "java.lang.Character",
                static_character_is_lower_case(&two),
            ),
            (
                "isUpperCase",
                "java.lang.Character",
                static_character_is_upper_case(&two),
            ),
        ] {
            let argc = if matches!(
                member,
                "digit" | "pow" | "atan2" | "hypot" | "remainderUnsigned"
            ) {
                1
            } else if matches!(member, "isLetterOrDigit" | "isLowerCase" | "isUpperCase") {
                2
            } else {
                0
            };
            let e = err.expect_err("wrong argc misses");
            assert_eq!(
                e.message,
                format!("No matching method {member} found taking {argc} args for class {class}")
            );
        }
        // `String/format` takes the evaluator (unused on the arity
        // path — a bare evaluator suffices).
        let mut eval = Evaluator::new_bare("user");
        let e = static_string_format(&mut eval, &[]).expect_err("wrong argc misses");
        assert_eq!(
            e.message,
            "No matching method format found taking 0 args for class java.lang.String"
        );
    }

    #[test]
    fn locale_builder_hash_stable() {
        // `dyn_hash` feeds the `Hash` impl (Rust-keyed maps), not
        // `hasheq`: same tag hashes alike, builders hash by id.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let (mut h1, mut h2) = (DefaultHasher::new(), DefaultHasher::new());
        LocaleHandle::us().dyn_hash(&mut h1);
        LocaleHandle::us().dyn_hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
        let sb = StringBuilder::fresh(String::new());
        let (mut h1, mut h2) = (DefaultHasher::new(), DefaultHasher::new());
        sb.dyn_hash(&mut h1);
        sb.dyn_hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }
}
