// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Lightweight AOT runtime boot (`BootMode::Lite`) and direct CLI frame
//! execution without embedded stdlib evaluation (`core_lazy`) or startup
//! `(eval-source ...)` parsing.
//!
//! Callers in [`boot_lite`] must NOT reference [`crate::core_lazy`],
//! [`crate::compiler`], [`crate::ns::boot`], or [`crate::printsys::boot_print_system`],
//! so `--gc-sections` / `-dead_strip` drops ~300 KB of embedded Clojure
//! sources and the analyzer/reader/eval subsystems from Lite binaries.

use std::sync::Arc;

use clj_value::{Keyword, List, Map, Symbol, Value, new_map};

use crate::error::Signal;
use crate::eval::{Evaluator, SourceError};
use crate::ns::{
    NativeFnPtr, OwnedNativeFnPtr, core_var, core_var_handle, load_native, ns_value,
    owned_entry_in, push_core_binding, set_core_ns,
};

/// One slice of `BootMode::Lite` native function entries `(name, ptr)`.
pub type LiteNativeSlice = &'static [(&'static str, NativeFnPtr)];

/// One sorted slice of `BootMode::Lite` owned-args native entries.
pub type LiteOwnedSlice = &'static [(&'static str, OwnedNativeFnPtr)];

/// `BootMode::Lite` core/collection owned-args entries.
pub const LITE_OWNED_CORE: LiteOwnedSlice = crate::ns::OWNED_CORE_ENTRIES;

/// `BootMode::Lite` sequence/transducer owned-args entries.
pub const LITE_OWNED_SEQ: LiteOwnedSlice = crate::ns::OWNED_SEQ_ENTRIES;

/// `BootMode::Lite` `clojure.string` owned-args entries.
pub const LITE_OWNED_STRING: LiteOwnedSlice = &[("join", crate::stdlib_natives::native_join_owned)];

/// Bitmask of runtime native capability groups required by a `BootMode::Lite`
/// program so link-time GC strips unreferenced subsystems (`re`, `num_bigint`,
/// `classemu`, `seqs`/`lazy`, `clojure.string`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiteCaps(u8);

impl LiteCaps {
    /// Core I/O (`print`, `println`, `pr`, `prn`, `newline`, `flush`, `slurp`, `spit`, `read-line`).
    pub const IO: Self = Self(1 << 0);
    /// Core collections, arithmetic, vars, macros, RNG, scalars, arrays, tower.
    pub const CORE: Self = Self(1 << 1);
    /// Lazy/sequence/transducer/collmisc/jdkio natives + internal helpers.
    pub const SEQ: Self = Self(1 << 2);
    /// Regular expression engine (`re-*` + `regex_automata`).
    pub const RE: Self = Self(1 << 3);
    /// Host/class emulation, concurrency, STM, protocols, multimethods.
    pub const HOST: Self = Self(1 << 4);
    /// `clojure.string` native subset (`STRING_NATIVES`).
    pub const STRING: Self = Self(1 << 5);

    /// Whether all bits in `cap` are enabled in `self`.
    #[must_use]
    pub const fn contains(self, cap: Self) -> bool {
        (self.0 & cap.0) == cap.0
    }

    /// Union two capability masks.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// `BootMode::Lite` I/O capability natives.
pub const LITE_CAP_IO: &[LiteNativeSlice] = &[
    crate::io::IO_LITE_NATIVES,
    &[("read-line", crate::reader::native_read_line)],
];

/// `BootMode::Lite` core language capability natives.
pub const LITE_CAP_CORE: &[LiteNativeSlice] = &[
    crate::core::CORE_NATIVES,
    crate::core::TOWER_NATIVES,
    crate::coll::COLL_NATIVES,
    crate::vars::VAR_NATIVES,
    crate::macros::MACRO_LITE_NATIVES,
    crate::rng::RNG_NATIVES,
    crate::scalars::SCALAR_NATIVES,
    crate::arrays::ARRAY_NATIVES,
];

/// `BootMode::Lite` lazy-seq, transducer, collection-misc, and JDK-IO natives.
pub const LITE_CAP_SEQ: &[LiteNativeSlice] = &[
    crate::seqs::SEQS_NATIVES,
    crate::lazy::LAZY_NATIVES,
    crate::xform::XFORM_NATIVES,
    crate::collmisc::COLLMISC_NATIVES,
    crate::jdkio::JDKIO_NATIVES,
];

/// `BootMode::Lite` regular-expression capability natives.
pub const LITE_CAP_RE: &[LiteNativeSlice] = &[crate::re::RE_NATIVES];

/// `BootMode::Lite` host/concurrency/STM/protocol/multimethod natives.
pub const LITE_CAP_HOST: &[LiteNativeSlice] = &[
    crate::concurrency::CONC_NATIVES,
    crate::stm::STM_NATIVES,
    crate::fx4_host::FX4_NATIVES,
    crate::types::TYPES_NATIVES,
    crate::typeinternals::TYPEINTERNALS_NATIVES,
    crate::classemu::CLASS_EMU_NATIVES,
    crate::classemu::CLASS_EMU_HIDDEN,
    crate::protocol::PROTOCOL_NATIVES,
    crate::multi::MULTI_NATIVES,
];

/// `BootMode::Lite` `clojure.string` capability natives.
pub const LITE_CAP_STRING: &[LiteNativeSlice] = &[crate::stdlib_natives::STRING_NATIVES];

/// Compiler-side mapping from each `clojure.core` Lite capability bit to its
/// native slices (never referenced by [`boot_lite_with_caps`]).
const LITE_CAP_CORE_GROUPS: &[(LiteCaps, &[LiteNativeSlice])] = &[
    (LiteCaps::IO, LITE_CAP_IO),
    (LiteCaps::CORE, LITE_CAP_CORE),
    (LiteCaps::SEQ, LITE_CAP_SEQ),
    (LiteCaps::RE, LITE_CAP_RE),
    (LiteCaps::HOST, LITE_CAP_HOST),
];

/// `def`-name metadata keys only var-metadata readers observe (no
/// runtime path consults them): a Lite program without such readers
/// ([`needs_var_meta`]) drops them at compile time, so a `defn` name
/// pools as a plain symbol instead of a read-back form (#271).
pub const LITE_INERT_DEF_META: &[&str] = &["arglists", "doc", "line", "column", "file"];

/// The `clojure.core` capability groups a compiled binary registers, each
/// with its hint id (its [`LITE_CAP_CORE_GROUPS`] position, which
/// [`lite_native_hint`] encodes).
pub const LITE_CORE_IO: LiteCoreGroup = (0, LITE_CAP_IO);
/// See [`LITE_CORE_IO`].
pub const LITE_CORE_CORE: LiteCoreGroup = (1, LITE_CAP_CORE);
/// See [`LITE_CORE_IO`].
pub const LITE_CORE_SEQ: LiteCoreGroup = (2, LITE_CAP_SEQ);
/// See [`LITE_CORE_IO`].
pub const LITE_CORE_RE: LiteCoreGroup = (3, LITE_CAP_RE);
/// See [`LITE_CORE_IO`].
pub const LITE_CORE_HOST: LiteCoreGroup = (4, LITE_CAP_HOST);

/// Pure core helper functions looked up internally by seq/coll/xform/jdkio
/// natives via `deref_core` (kept resident in Lite even when user code does
/// not reference them directly).
const LITE_INTERNAL_HELPERS: &[&str] = &[
    "conj",
    "transient",
    "conj!",
    "persistent!",
    "identity",
    "some?",
    "comp",
    "compare",
    ">=",
    "<=",
    "inc'",
    "take",
    "+",
];

/// `clojure.core` names natives resolve by name (`deref_core` /
/// `core_var`) besides [`LITE_INTERNAL_HELPERS`]: a compiled program that
/// declares one installs it at boot, while every other declared native
/// installs when compiled code first resolves its var ([`install_lazy`]:
/// about half of the declared natives never run for a given input).
/// `by_name_lookups_are_eager` pins this list against the sources.
pub const LITE_BY_NAME: &[&str] = &[
    "assoc",
    "commute",
    "default-data-readers",
    "global-hierarchy",
    "print-dup",
    "print-method",
    "print-simple",
];

/// Dynamic core var names installed by [`boot_lite`].
const LITE_DYNAMIC_NAMES: &[&str] = &[
    "*ns*",
    "*out*",
    "*err*",
    "*in*",
    "*flush-on-newline*",
    "*print-length*",
    "*print-level*",
    "*print-namespace-maps*",
    "*print-readably*",
    "*print-meta*",
    "*print-dup*",
    "*verbose-defrecords*",
    "*command-line-args*",
    "*file*",
    "*source-path*",
    "*read-eval*",
];

/// Look up `name` in `tables`.
fn find_native(tables: &[LiteNativeSlice], name: &str) -> Option<NativeFnPtr> {
    tables
        .iter()
        .flat_map(|t| t.iter())
        .find(|&&(k, _)| k == name)
        .map(|&(_, f)| f)
}

/// Look up `name` across active `core` capability slices.
fn find_native_in_groups(groups: &[LiteCoreGroup], name: &str) -> Option<NativeFnPtr> {
    groups.iter().find_map(|&(_, g)| find_native(g, name))
}

/// One registered `clojure.core` capability group: its hint id (the
/// `LITE_CAP_CORE_GROUPS` position) and its native slices.
pub type LiteCoreGroup = (u32, &'static [LiteNativeSlice]);

/// "No native" for [`lite_native_hint`] (decodes to no group, so a
/// var without one takes the name scan).
pub const NO_LITE_HINT: u32 = u32::MAX;

/// Hint bit: the name has an owned-args entry (boot searches the owned
/// tables only then; a stale bit only costs the fast path, never
/// correctness).
const OWNED_HINT_BIT: u32 = 1 << 15;

/// The owned-args tables [`OWNED_HINT_BIT`] is computed against (every
/// owned group a binary can register).
const LITE_OWNED_ALL: [LiteOwnedSlice; 3] = [LITE_OWNED_CORE, LITE_OWNED_SEQ, LITE_OWNED_STRING];

/// The hint group of `clojure.string`'s natives (after the core groups).
const STRING_HINT_GROUP: u32 = 5;

/// Where the Lite native behind `ns/name` sits — `group << 24 | table <<
/// 16 | entry`, in the first-match order the runtime scan uses, plus
/// [`OWNED_HINT_BIT`] — so a compiled binary installs it without
/// scanning the capability tables (#271); [`NO_LITE_HINT`] when there
/// is none. Compiler-side only.
#[must_use]
pub fn lite_native_hint(ns: &str, name: &str) -> u32 {
    match ns {
        "clojure.core" => core_groups_hint(LITE_CAP_CORE_GROUPS, name),
        "clojure.string" => {
            core_groups_hint(&[(LiteCaps::STRING, LITE_CAP_STRING)], name)
                | (STRING_HINT_GROUP << 24)
        }
        _ => NO_LITE_HINT,
    }
}

/// `a == b`, usable in constants.
const fn const_str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The first `name` entry across `groups` as `group << 24 | table << 16 |
/// entry` ([`NO_LITE_HINT`] when absent) — const, so the internal helpers'
/// hints resolve at compile time without linking the tables.
const fn core_groups_hint(groups: &[(LiteCaps, &[LiteNativeSlice])], name: &str) -> u32 {
    let mut g = 0;
    while g < groups.len() {
        let tables = groups[g].1;
        let mut t = 0;
        while t < tables.len() {
            let mut e = 0;
            while e < tables[t].len() {
                if const_str_eq(tables[t][e].0, name) {
                    let owned = if has_owned_entry(name) {
                        OWNED_HINT_BIT
                    } else {
                        0
                    };
                    return ((g as u32) << 24) | ((t as u32) << 16) | owned | e as u32;
                }
                e += 1;
            }
            t += 1;
        }
        g += 1;
    }
    NO_LITE_HINT
}

/// Whether any owned-args table has an entry for `name`.
const fn has_owned_entry(name: &str) -> bool {
    let mut g = 0;
    while g < LITE_OWNED_ALL.len() {
        let entries = LITE_OWNED_ALL[g];
        let mut e = 0;
        while e < entries.len() {
            if const_str_eq(entries[e].0, name) {
                return true;
            }
            e += 1;
        }
        g += 1;
    }
    false
}

/// [`LITE_INTERNAL_HELPERS`]' hints, resolved at compile time.
const LITE_HELPER_HINTS: [u32; LITE_INTERNAL_HELPERS.len()] = {
    let mut out = [NO_LITE_HINT; LITE_INTERNAL_HELPERS.len()];
    let mut i = 0;
    while i < out.len() {
        out[i] = core_groups_hint(LITE_CAP_CORE_GROUPS, LITE_INTERNAL_HELPERS[i]);
        i += 1;
    }
    out
};

/// The native a compiler [`lite_native_hint`] points at in the registered
/// tables — `lib`'s (a `clojure.string` var) or the core groups' —
/// when that entry still names `name` (a hint of the other kind, or a
/// runtime archive from another build, falls back to the scan).
fn hinted_native(
    hint: u32,
    name: &str,
    core_groups: &[LiteCoreGroup],
    lib: Option<&[LiteNativeSlice]>,
) -> Option<NativeFnPtr> {
    let (group, table, entry) = (hint >> 24, (hint >> 16) & 0xff, hint & 0x7fff);
    let tables = match lib {
        Some(lib) => (group == STRING_HINT_GROUP).then_some(lib)?,
        None => core_groups.iter().find(|&&(id, _)| id == group)?.1,
    };
    let &(k, f) = tables.get(table as usize)?.get(entry as usize)?;
    (k == name).then_some(f)
}

/// Look up a pure runtime `clojure.core` native compatible with `Lite` boot.
fn find_lite_core_native(name: &str) -> Option<NativeFnPtr> {
    LITE_CAP_CORE_GROUPS
        .iter()
        .find_map(|&(_, g)| find_native(g, name))
}

/// Look up a `clojure.string` native compatible with `Lite` boot.
fn find_lite_string_native(name: &str) -> Option<NativeFnPtr> {
    find_native(LITE_CAP_STRING, name)
}

/// Classify the minimal set of `BootMode::Lite` runtime capabilities needed
/// for a compiled module's declared `(ns, name)` var list.
#[must_use]
pub fn lite_caps_for(vars: &[(&str, &str)]) -> LiteCaps {
    let mut caps = LiteCaps::default();
    for &(ns, name) in vars {
        if ns == "clojure.string" {
            if find_lite_string_native(name).is_some() {
                caps = caps.union(LiteCaps::STRING);
            }
            continue;
        }
        if ns == "clojure.core" {
            if let Some(&(cap, _)) = LITE_CAP_CORE_GROUPS
                .iter()
                .find(|(_, g)| find_native(g, name).is_some())
            {
                caps = caps.union(cap);
            }
        }
    }
    // `SEQ` natives resolve internal helpers (`conj`, `transient`, `+`, comparisons)
    // via `deref_core`, which live in `LITE_CAP_CORE`.
    if caps.contains(LiteCaps::SEQ) {
        caps = caps.union(LiteCaps::CORE);
    }
    caps
}

/// Whether `name` is one of the dynamic `clojure.core` vars bootstrapped
/// directly by [`boot_lite_with_caps`].
fn is_lite_dynamic(name: &str) -> bool {
    LITE_DYNAMIC_NAMES.contains(&name)
}

/// Whether `clojure.core/<name>` is natively supported under `BootMode::Lite`.
#[must_use]
pub fn is_lite_core_var(name: &str) -> bool {
    is_lite_dynamic(name) || find_lite_core_native(name).is_some()
}

/// Whether `clojure.string/<name>` is natively supported under `BootMode::Lite`.
#[must_use]
pub fn is_lite_string_var(name: &str) -> bool {
    find_lite_string_native(name).is_some()
}

/// Whether `(<ns>/<name>)` requires initializing stdin (`*in*` buffered reader).
#[must_use]
pub fn is_lite_stdin_var(ns: &str, name: &str) -> bool {
    ns == "clojure.core" && matches!(name, "*in*" | "read-line")
}

/// Install a dynamic `clojure.core` var with root `val`, adding
/// `:dynamic true` to its metadata when `dynamic_meta`.
fn install_core_dynamic(eval: &mut Evaluator, name: &str, val: Value, dynamic_meta: bool) {
    // (The `:dynamic` stamp, like every var's metadata, only matters
    // to a program that can read it.)
    let bare = eval.skip_var_meta;
    let q = Symbol::new(Some("clojure.core"), name);
    let id = eval.registry.intern(&q);
    eval.registry.with_state_mut(id, |st| {
        st.root = Some(val);
        st.dynamic = true;
        if dynamic_meta && !bare {
            let meta: Map = st.meta.clone().unwrap_or_else(new_map).insert(
                Value::Keyword(Keyword::new(None, "dynamic")),
                Value::Bool(true),
            );
            st.meta = Some(meta);
        }
    });
}

/// Install a declared var's Lite native when compiled code first
/// resolves it (the boot installed only [`LITE_BY_NAME`] and the
/// helpers): a no-op for other namespaces, an installed var, or a name
/// with no native in the registered tables.
pub fn install_lazy(eval: &mut Evaluator, (ns, name, hint): (&str, &str, u32), tables: LiteTables) {
    let native_ns = ns == "clojure.core" || tables.lib_tables.iter().any(|&(lib, _)| lib == ns);
    if !native_ns || eval.registry.lookup_str(ns, name).is_some() {
        return;
    }
    if let Some(row) = lite_row((ns, name, hint), tables) {
        eval.install_native_rows(ns, vec![row]);
    }
}

/// A declared Lite native's entries (fn and owned-args) without
/// installing its var — what a direct call needs (`None`: a dynamic, a
/// foreign namespace, or no native in the registered tables).
pub(crate) fn lite_entry(
    decl: (&str, &str, u32),
    tables: LiteTables,
) -> Option<(
    std::sync::Arc<crate::ext::NativeImpl>,
    Option<std::sync::Arc<crate::ext::OwnedNativeImpl>>,
)> {
    let (_, f, owned) = lite_row(decl, tables)?;
    Some((std::sync::Arc::new(f), owned))
}

/// The native a declared Lite var installs — `None` for a dynamic
/// (installed apart) or a name the registered tables lack — with its
/// owned-args entry (searched only when the hint says one exists).
fn lite_row<'n>(
    (ns, name, hint): (&str, &'n str, u32),
    tables: LiteTables,
) -> Option<crate::eval::NativeRow<'n>> {
    let f = if ns == "clojure.core" {
        if name.starts_with('*') && is_lite_dynamic(name) {
            return None;
        }
        hinted_native(hint, name, tables.core_groups, None)
            .or_else(|| find_native_in_groups(tables.core_groups, name))?
    } else {
        let &(_, lib) = tables
            .lib_tables
            .iter()
            .find(|&&(lib_ns, _)| lib_ns == ns)?;
        hinted_native(hint, name, tables.core_groups, Some(lib))
            .or_else(|| find_native(lib, name))?
    };
    let owned = (hint & OWNED_HINT_BIT != 0)
        .then(|| {
            tables
                .owned_groups
                .iter()
                .find_map(|entries| owned_entry_in(entries, name))
        })
        .flatten();
    Some((name, f, owned))
}

/// Construct a minimal evaluator for AOT `BootMode::Lite` programs with every
/// capability group enabled.
#[must_use]
pub fn boot_lite(vars: &[(&str, &str)], need_stdin: bool) -> Evaluator {
    let tables = LiteTables {
        core_groups: &[
            LITE_CORE_IO,
            LITE_CORE_CORE,
            LITE_CORE_SEQ,
            LITE_CORE_RE,
            LITE_CORE_HOST,
        ],
        owned_groups: &[LITE_OWNED_CORE, LITE_OWNED_SEQ, LITE_OWNED_STRING],
        lib_tables: &[("clojure.string", LITE_CAP_STRING)],
    };
    boot_lite_impl(
        vars,
        &[],
        need_stdin,
        tables,
        true,
        true,
        needs_var_meta(vars, true),
    )
}

/// The capability tables a Lite boot installs natives from (the ones
/// the compiled binary registered).
#[derive(Clone, Copy)]
pub struct LiteTables<'a> {
    /// `clojure.core` groups, with their hint ids.
    pub core_groups: &'a [LiteCoreGroup],
    /// Owned-argument entries of the registered natives.
    pub owned_groups: &'a [LiteOwnedSlice],
    /// Library namespaces' natives (`clojure.string`).
    pub lib_tables: &'a [(&'a str, &'a [LiteNativeSlice])],
}

/// Construct a minimal evaluator for AOT `BootMode::Lite` programs using only
/// the capability slices registered by the compiled binary (`boot_lite` keeps
/// every group for unit tests; [`boot_lite_with_caps`] never references any
/// `LITE_CAP_*` constant directly so unreferenced subsystems dead-strip).
#[must_use]
pub fn boot_lite_with_caps(
    vars: &[(&str, &str)],
    hints: &[u32],
    need_stdin: bool,
    tables: LiteTables,
    seq_helpers: bool,
    var_meta: bool,
) -> Evaluator {
    boot_lite_impl(
        vars,
        hints,
        need_stdin,
        tables,
        seq_helpers,
        false,
        var_meta,
    )
}

fn has_core_ref(vars: &[(&str, &str)], name: &str) -> bool {
    vars.iter()
        .any(|&(vn, vname)| vn == "clojure.core" && vname == name)
}

fn needs_stdout_dynamics(vars: &[(&str, &str)]) -> bool {
    vars.iter().any(|&(ns, name)| {
        ns == "clojure.core"
            && matches!(
                name,
                "*out*"
                    | "*flush-on-newline*"
                    | "*print-readably*"
                    | "*print-namespace-maps*"
                    | "*print-length*"
                    | "*print-level*"
                    | "*print-meta*"
                    | "*print-dup*"
                    | "println"
                    | "print"
                    | "pr"
                    | "prn"
                    | "pr-str"
                    | "prn-str"
                    | "print-str"
                    | "newline"
                    | "flush"
                    | "with-out-str"
                    | "format"
                    | "printf"
                    | "time"
            )
    })
}

/// Whether a Lite program can observe var metadata (it references a
/// reader of it). When it cannot, `def`s keep no metadata at all, and
/// the compiler drops [`LITE_INERT_DEF_META`] from `def` names. The
/// by-name readers (`doc`, `macro?`, the var-resolving fns) find the var
/// themselves; the value readers (`meta`, `vary-meta`, `alter-meta!`,
/// `reset-meta!`) see a var's metadata only when the program can hold a
/// var object (`var_objects`: a `#'v` outside `defn`'s own inert forms),
/// so `(meta coll)` alone keeps none (#271: `clojure.set`'s `select` and
/// `rename` read their relation's metadata).
#[must_use]
pub fn needs_var_meta(vars: &[(&str, &str)], var_objects: bool) -> bool {
    vars.iter().any(|&(ns, name)| {
        ns == "clojure.core"
            && match name {
                "meta" | "alter-meta!" | "reset-meta!" | "vary-meta" => var_objects,
                "doc" | "macro?" | "resolve" | "ns-resolve" | "ns-publics" | "ns-interns"
                | "ns-refers" => true,
                _ => false,
            }
    })
}

enum DynInit {
    Nil,
    Bool(bool),
    NoSource,
}

impl DynInit {
    fn into_value(self) -> Value {
        match self {
            Self::Nil => Value::Nil,
            Self::Bool(b) => Value::Bool(b),
            Self::NoSource => Value::Str(Arc::from("NO_SOURCE_PATH")),
        }
    }
}

fn install_optional_dynamics(eval: &mut Evaluator, vars: &[(&str, &str)], need_all: bool) {
    let entries: &[(&str, DynInit, bool)] = &[
        ("*print-length*", DynInit::Nil, true),
        ("*print-level*", DynInit::Nil, true),
        ("*print-meta*", DynInit::Bool(false), false),
        ("*print-dup*", DynInit::Bool(false), false),
        ("*verbose-defrecords*", DynInit::Bool(false), true),
        ("*command-line-args*", DynInit::Nil, true),
        ("*file*", DynInit::NoSource, true),
        ("*source-path*", DynInit::NoSource, false),
    ];
    for (name, init, has_meta) in entries {
        if need_all || has_core_ref(vars, name) {
            let v = match init {
                DynInit::Nil => DynInit::Nil.into_value(),
                DynInit::Bool(b) => DynInit::Bool(*b).into_value(),
                DynInit::NoSource => DynInit::NoSource.into_value(),
            };
            install_core_dynamic(eval, name, v, *has_meta);
        }
    }
    let read_needed = ["*read-eval*", "read", "read-string"]
        .iter()
        .any(|n| has_core_ref(vars, n));
    if need_all || read_needed {
        install_core_dynamic(eval, "*read-eval*", Value::Bool(true), false);
    }
}

/// Install the declared vars' natives (the compiler's list is
/// distinct), one batch per namespace — under `lazy` (compiled
/// binaries) only the [`LITE_BY_NAME`] ones, the rest on first use —
/// then the internal helpers the program did not declare.
fn install_declared(
    eval: &mut Evaluator,
    vars: &[(&str, &str)],
    hints: &[u32],
    tables: LiteTables,
    (seq_helpers, lazy): (bool, bool),
) {
    let mut batches: Vec<(&str, Vec<crate::eval::NativeRow>)> = Vec::new();
    for (i, &(ns, name)) in vars.iter().enumerate() {
        if lazy && !(ns == "clojure.core" && LITE_BY_NAME.contains(&name)) {
            continue;
        }
        let hint = hints.get(i).copied().unwrap_or(NO_LITE_HINT);
        let Some(row) = lite_row((ns, name, hint), tables) else {
            continue;
        };
        match batches.iter_mut().find(|(b, _)| *b == ns) {
            Some((_, rows)) => rows.push(row),
            None => batches.push((ns, vec![row])),
        }
    }
    for (ns, rows) in batches {
        eval.install_native_rows(ns, rows);
    }
    if seq_helpers {
        let helpers = LITE_INTERNAL_HELPERS
            .iter()
            .zip(&LITE_HELPER_HINTS)
            .filter(|&(&name, _)| core_var(eval, name).is_none())
            .filter_map(|(&name, &hint)| lite_row(("clojure.core", name, hint), tables))
            .collect();
        eval.install_native_rows("clojure.core", helpers);
    }
}

fn boot_lite_impl(
    vars: &[(&str, &str)],
    hints: &[u32],
    need_stdin: bool,
    tables: LiteTables,
    seq_helpers: bool,
    all_dynamics: bool,
    var_meta: bool,
) -> Evaluator {
    let mut eval = Evaluator::new_bare("user");
    eval.print_defaults_only = true;
    eval.skip_var_meta = !all_dynamics && !var_meta;
    eval.registry.set_bare_vars(eval.skip_var_meta);
    eval.registry.ensure_ns("clojure.core");
    let user_id = eval.registry.ensure_ns("user");
    let user_handle = ns_value("user", user_id);
    let helper_cap = usize::from(seq_helpers) * LITE_INTERNAL_HELPERS.len();
    let need_all = all_dynamics
        || ["resolve", "ns-resolve", "find-var", "var-get", "intern"]
            .iter()
            .any(|n| has_core_ref(vars, n));
    eval.registry.reserve_vars(
        "clojure.core",
        helper_cap
            + vars.len()
            + if need_all {
                LITE_DYNAMIC_NAMES.len()
            } else {
                6
            },
    );

    install_core_dynamic(&mut eval, "*ns*", user_handle.clone(), true);
    if need_all || needs_stdout_dynamics(vars) {
        let out = crate::io::stdout_writer(&mut eval);
        install_core_dynamic(&mut eval, "*out*", out, false);
        install_core_dynamic(&mut eval, "*flush-on-newline*", Value::Bool(true), false);
        install_core_dynamic(&mut eval, "*print-readably*", Value::Bool(true), false);
        install_core_dynamic(&mut eval, "*print-namespace-maps*", Value::Bool(true), true);
    }
    if need_all || has_core_ref(vars, "*err*") {
        let err = crate::io::stderr_writer(&mut eval);
        install_core_dynamic(&mut eval, "*err*", err, false);
    }
    if need_all || need_stdin || has_core_ref(vars, "*in*") {
        let stdin = if need_stdin {
            crate::reader::stdin_reader(&mut eval)
        } else {
            Value::Nil
        };
        install_core_dynamic(&mut eval, "*in*", stdin, false);
    }
    install_optional_dynamics(&mut eval, vars, need_all);

    // Base thread-local `*ns*` binding frame (mirrors full `boot`: `in-ns`
    // mutates the frame rather than the root).
    let id = core_var(&eval, "*ns*").expect("*ns* installed");
    eval.push_binding(core_var_handle(id, "*ns*"), user_handle);

    // Compiled binaries (`boot_lite_with_caps`) install on first use; the
    // interpreter-side `boot_lite` resolves names, so it installs all.
    install_declared(&mut eval, vars, hints, tables, (seq_helpers, !all_dynamics));

    eval.registry.set_current("user");
    eval.reset_gensym();
    eval
}

/// Lightweight `(ns name …)` transition in `BootMode::Lite`: ensures the
/// namespace exists, switches the current registry namespace, and updates `*ns*`.
pub fn ensure_ns_lite(eval: &mut Evaluator, name: &str) {
    let id = eval.registry.ensure_ns(name);
    eval.registry.set_current(name);
    set_core_ns(eval, ns_value(name, id));
}

/// Lightweight `(require 'clojure.string)` / `(require '[clojure.string :as s])`
/// in `BootMode::Lite`: ensures `clojure.string` exists and installs any `:as`
/// alias on the current namespace without loading embedded stdlib sources.
pub fn require_string_lite(eval: &mut Evaluator, alias: Option<&str>) {
    let target_id = eval.registry.ensure_ns("clojure.string");
    if let Some(a) = alias.filter(|s| !s.is_empty()) {
        let cur = eval.registry.current().to_owned();
        eval.registry.with_ns_mut(&cur, |ns| {
            let _ = ns.add_alias(a, "clojure.string", target_id);
        });
    }
}

/// Convert CLI arguments into Clojure's `*command-line-args*` value (`nil` when
/// empty, list of strings otherwise).
fn cli_args_value(args: &[String]) -> Value {
    if args.is_empty() {
        Value::Nil
    } else {
        Value::List(List::from_vec(
            args.iter()
                .map(|a| Value::Str(Arc::from(a.as_str())))
                .collect(),
        ))
    }
}

/// Execute a compiled AOT program inside a direct CLI binding frame
/// (`*command-line-args*` + `load_native` `*file*`/`*ns*`), eliminating
/// `eval_source(run_script(...))` overhead on both `Lite` and `Full` modes.
pub fn run_with_cli_frame<F>(eval: &mut Evaluator, path: &str, args: &[String], body: F) -> i32
where
    F: FnOnce(&mut Evaluator) -> Result<Value, Signal>,
{
    let depth = eval.bindings_depth();
    push_core_binding(eval, "*command-line-args*", cli_args_value(args));
    let code = match load_native(eval, path, body) {
        Ok(_) => 0,
        Err(err) => {
            let src_err = SourceError::Uncaught(eval.throwable(err));
            eprintln!("{}", crate::script::run_diagnostic(path, &src_err));
            1
        }
    };
    eval.pop_bindings(depth);
    eval.join_drainers();
    let _flushed = std::io::Write::flush(&mut std::io::stdout());
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ns::deref_core;
    use clj_value::Vector;

    /// A native's address (fn pointers compare by address only).
    fn addr(f: Option<NativeFnPtr>) -> Option<usize> {
        f.map(|f| f as usize)
    }

    #[test]
    fn lite_native_hints_point_at_the_scan_result() {
        let all = [
            LITE_CORE_IO,
            LITE_CORE_CORE,
            LITE_CORE_SEQ,
            LITE_CORE_RE,
            LITE_CORE_HOST,
        ];
        // The registered ids are the hint encoding's group positions.
        for (i, (&(id, tables), &(_, compiler))) in all.iter().zip(LITE_CAP_CORE_GROUPS).enumerate()
        {
            assert_eq!(id as usize, i);
            assert!(std::ptr::eq(tables, compiler));
        }
        let inc = lite_native_hint("clojure.core", "inc");
        assert!(inc >> 24 < STRING_HINT_GROUP);
        assert_eq!(
            addr(hinted_native(inc, "inc", &all, None)),
            addr(find_native_in_groups(&all, "inc"))
        );
        let trim = lite_native_hint("clojure.string", "trim");
        assert_eq!(trim >> 24, STRING_HINT_GROUP);
        assert_eq!(
            addr(hinted_native(trim, "trim", &all, Some(LITE_CAP_STRING))),
            addr(find_native(LITE_CAP_STRING, "trim"))
        );
        // Unknown names and namespaces have no hint.
        assert_eq!(lite_native_hint("clojure.core", "no-such"), NO_LITE_HINT);
        assert_eq!(lite_native_hint("clojure.string", "no-such"), NO_LITE_HINT);
        assert_eq!(lite_native_hint("user", "inc"), NO_LITE_HINT);
        // Every mismatch falls back to the scan (`None`): another name,
        // an unregistered group, a hint of the other kind, and entries
        // past the tables.
        assert!(hinted_native(inc, "dec", &all, None).is_none());
        assert!(hinted_native(inc, "inc", &all[..1], None).is_none());
        assert!(hinted_native(inc, "inc", &all, Some(LITE_CAP_STRING)).is_none());
        assert!(hinted_native(trim, "trim", &all, None).is_none());
        assert!(hinted_native(NO_LITE_HINT, "inc", &all, None).is_none());
        let past_tables = (inc & 0xff00_0000) | (0xff << 16);
        assert!(hinted_native(past_tables, "inc", &all, None).is_none());
        assert!(hinted_native(inc | 0xffff, "inc", &all, None).is_none());
        // The internal helpers' compile-time hints are the same encoding.
        for (&name, &hint) in LITE_INTERNAL_HELPERS.iter().zip(&LITE_HELPER_HINTS) {
            assert_eq!(hint, lite_native_hint("clojure.core", name), "{name}");
        }
    }

    /// Whether `ns/name`'s installed native carries an owned-args entry.
    fn installed_owned(eval: &Evaluator, ns: &str, name: &str) -> bool {
        let sym = Symbol::new(Some(ns), name);
        let id = eval
            .registry
            .lookup_in_live_or_removed(ns, &sym)
            .expect("installed");
        let root = eval.deref_var_unforced(id, &sym);
        crate::ext::as_native(&root)
            .expect("a native root")
            .owned
            .is_some()
    }

    /// Resolve every declared var the way compiled code does on first
    /// use ([`install_lazy`]).
    fn resolve_all(eval: &mut Evaluator, vars: &[(&str, &str)], hints: &[u32], tables: LiteTables) {
        for (i, &(ns, name)) in vars.iter().enumerate() {
            let hint = hints.get(i).copied().unwrap_or(NO_LITE_HINT);
            install_lazy(eval, (ns, name, hint), tables);
        }
    }

    /// Lite binaries install a declared native only when compiled code
    /// first resolves its var, so every name a native resolves by name
    /// must be installed eagerly: a helper, a dynamic (installed apart),
    /// or [`LITE_BY_NAME`].
    #[test]
    fn by_name_lookups_are_eager() {
        let call = regex::Regex::new(r#"(?:deref_core|core_var)\(\s*&?[a-z_.]+\s*,\s*"([^"]+)""#)
            .expect("pattern");
        let mut dirs = vec![std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src"
        ))];
        let mut names = std::collections::BTreeSet::new();
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).expect("read src") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    // (CRLF checkouts, e.g. Windows, normalized for the marker.)
                    let text = std::fs::read_to_string(&path)
                        .expect("read")
                        .replace("\r\n", "\n");
                    let body = text
                        .split("#[cfg(test)]\nmod tests")
                        .next()
                        .unwrap_or_default();
                    names.extend(call.captures_iter(body).map(|c| c[1].to_owned()));
                }
            }
        }
        assert!(
            names.contains("assoc") && names.contains("conj"),
            "{names:?}"
        );
        let lazy: Vec<&String> = names
            .iter()
            .filter(|n| {
                !n.starts_with('*')
                    && !LITE_INTERNAL_HELPERS.contains(&n.as_str())
                    && !LITE_BY_NAME.contains(&n.as_str())
            })
            .collect();
        assert!(
            lazy.is_empty(),
            "resolved by name but installed lazily: {lazy:?}"
        );
    }

    #[test]
    fn value_readers_see_var_meta_only_given_a_var() {
        let core = |name| ("clojure.core", name);
        for name in ["meta", "vary-meta", "alter-meta!", "reset-meta!"] {
            assert!(!needs_var_meta(&[core(name)], false), "{name}");
            assert!(needs_var_meta(&[core(name)], true), "{name}");
        }
        for name in ["doc", "macro?", "resolve", "ns-publics"] {
            assert!(needs_var_meta(&[core(name)], false), "{name}");
        }
        assert!(!needs_var_meta(&[core("map"), ("user", "meta")], true));
    }

    #[test]
    fn lazy_installs_skip_installed_foreign_and_unknown_names() {
        let tables = LiteTables {
            core_groups: &[LITE_CORE_CORE],
            owned_groups: &[],
            lib_tables: &[("clojure.string", LITE_CAP_STRING)],
        };
        let mut eval =
            boot_lite_with_caps(&[("clojure.core", "inc")], &[], false, tables, false, false);
        assert!(core_var(&eval, "inc").is_none(), "installed on first use");
        install_lazy(&mut eval, ("clojure.core", "inc", NO_LITE_HINT), tables);
        let id = core_var(&eval, "inc").expect("installed");
        install_lazy(&mut eval, ("clojure.core", "inc", NO_LITE_HINT), tables);
        assert_eq!(core_var(&eval, "inc"), Some(id), "installed once");
        install_lazy(&mut eval, ("user", "inc", NO_LITE_HINT), tables);
        assert!(eval.registry.lookup_str("user", "inc").is_none());
        install_lazy(&mut eval, ("clojure.core", "no-such", NO_LITE_HINT), tables);
        assert!(core_var(&eval, "no-such").is_none());
    }

    #[test]
    fn owned_hint_bits_steer_the_owned_search() {
        for (ns, name, owned) in [
            ("clojure.core", "assoc", true),
            ("clojure.core", "inc", false),
            ("clojure.string", "join", true),
            ("clojure.string", "trim", false),
        ] {
            let hint = lite_native_hint(ns, name);
            assert_eq!(hint & OWNED_HINT_BIT != 0, owned, "{ns}/{name}");
        }
        let vars = [
            ("clojure.core", "assoc"),
            ("clojure.core", "inc"),
            ("clojure.core", "conj"),
            ("clojure.string", "join"),
            ("clojure.core", "*out*"),
        ];
        let tables = LiteTables {
            core_groups: &[LITE_CORE_CORE, LITE_CORE_SEQ],
            owned_groups: &[LITE_OWNED_CORE, LITE_OWNED_SEQ, LITE_OWNED_STRING],
            lib_tables: &[("clojure.string", LITE_CAP_STRING)],
        };
        let hints: Vec<u32> = vars
            .iter()
            .map(|&(ns, n)| lite_native_hint(ns, n))
            .collect();
        // Hinted and scanned (no hints) boots install the same entries;
        // `conj` is declared and an internal helper (installed once).
        for hints in [&hints[..], &[]] {
            let mut eval = boot_lite_with_caps(
                &vars,
                hints,
                true,
                tables,
                true,
                needs_var_meta(&vars, true),
            );
            // Boot installs the by-name `assoc` and the helpers; `inc`
            // and `join` wait for first use.
            assert!(core_var(&eval, "inc").is_none());
            assert!(installed_owned(&eval, "clojure.core", "assoc"));
            resolve_all(&mut eval, &vars, hints, tables);
            assert!(!installed_owned(&eval, "clojure.core", "inc"));
            assert!(installed_owned(&eval, "clojure.core", "conj"));
            assert!(installed_owned(&eval, "clojure.string", "join"));
            assert!(core_var(&eval, "persistent!").is_some());
        }
    }

    #[test]
    fn hinted_boot_installs_and_falls_back_on_stale_hints() {
        let vars = [
            ("clojure.core", "inc"),
            ("clojure.string", "trim"),
            ("clojure.core", "dec"),
            ("clojure.string", "join"),
        ];
        // `dec` and `join` carry stale hints (another entry's):
        // the name check rejects them and the scan installs them.
        let hints = [
            lite_native_hint("clojure.core", "inc"),
            lite_native_hint("clojure.string", "trim"),
            lite_native_hint("clojure.core", "inc"),
            lite_native_hint("clojure.string", "trim"),
        ];
        let tables = LiteTables {
            core_groups: &[LITE_CORE_CORE],
            owned_groups: &[],
            lib_tables: &[("clojure.string", LITE_CAP_STRING)],
        };
        let mut eval = boot_lite_with_caps(
            &vars,
            &hints,
            false,
            tables,
            false,
            needs_var_meta(&vars, true),
        );
        resolve_all(&mut eval, &vars, &hints, tables);
        assert!(core_var(&eval, "inc").is_some());
        assert!(core_var(&eval, "dec").is_some());
        for name in ["trim", "join"] {
            let sym = Symbol::new(Some("clojure.string"), name);
            assert!(
                eval.registry
                    .lookup_in_live_or_removed("clojure.string", &sym)
                    .is_some(),
                "{name}"
            );
        }
    }

    #[test]
    fn lite_queries_classify_core_string_and_stdin_vars() {
        assert!(is_lite_core_var("println"));
        assert!(is_lite_core_var("vector"));
        assert!(is_lite_core_var("list"));
        assert!(is_lite_core_var("apply"));
        assert!(is_lite_core_var("instance?"));
        assert!(is_lite_core_var("__static*"));
        assert!(is_lite_core_var("*out*"));
        assert!(is_lite_core_var("read-line"));
        assert!(!is_lite_core_var("eval"));
        assert!(!is_lite_core_var("read-string"));
        assert!(is_lite_core_var("run!"));
        assert!(is_lite_core_var("reduce-kv"));
        assert!(is_lite_core_var("fnil"));

        assert!(is_lite_string_var("split"));
        assert!(is_lite_string_var("join"));
        assert!(!is_lite_string_var("replace"));
        assert!(!is_lite_string_var("no-such-string-fn"));

        assert!(is_lite_stdin_var("clojure.core", "*in*"));
        assert!(is_lite_stdin_var("clojure.core", "read-line"));
        assert!(!is_lite_stdin_var("clojure.core", "slurp"));
        assert!(!is_lite_stdin_var("clojure.string", "read-line"));
    }

    #[test]
    fn boot_lite_installs_dynamics_requested_vars_and_ns() {
        let mut eval = boot_lite(
            &[
                ("clojure.core", "println"),
                ("clojure.core", "*out*"),
                ("clojure.core", "conj"),
                ("clojure.string", "split"),
                ("clojure.string", "join"),
                ("clojure.string", "join"),
                ("user", "helper"),
                ("my.app", "solve"),
            ],
            false,
        );
        assert!(eval.print_defaults_only);
        assert!(eval.lazy_hooks.is_none());
        assert_eq!(deref_core(&eval, "*in*"), Value::Nil);
        assert!(core_var(&eval, "println").is_some());
        assert!(core_var(&eval, "conj").is_some());
        assert!(core_var(&eval, "eval").is_none());
        assert_eq!(eval.registry.current(), "user");

        ensure_ns_lite(&mut eval, "my.app");
        assert_eq!(eval.registry.current(), "my.app");
        let ns_val = deref_core(&eval, "*ns*");
        assert_eq!(
            crate::ext::as_ns(&ns_val).expect("ns handle").name.as_ref(),
            "my.app"
        );
        require_string_lite(&mut eval, None);
        require_string_lite(&mut eval, Some("str"));
        let has_alias = eval
            .registry
            .with_ns_mut("my.app", |ns| {
                ns.aliases()
                    .any(|(a, t, _)| a == "str" && t == "clojure.string")
            })
            .unwrap_or(false);
        assert!(has_alias);

        let boot_with_in = boot_lite(&[("clojure.core", "read-line")], true);
        assert_ne!(deref_core(&boot_with_in, "*in*"), Value::Nil);
    }

    #[test]
    fn run_with_cli_frame_binds_args_handles_errors_and_lazy_forks() {
        let mut eval = boot_lite(&[("clojure.core", "println")], false);
        let code = run_with_cli_frame(
            &mut eval,
            "app.clj",
            &["x".to_owned(), "y".to_owned()],
            |e| {
                let cli = deref_core(e, "*command-line-args*");
                assert_eq!(
                    cli,
                    Value::List(List::from_vec(vec![
                        Value::Str(Arc::from("x")),
                        Value::Str(Arc::from("y")),
                    ]))
                );
                Ok(Value::Nil)
            },
        );
        assert_eq!(code, 0);
        assert_eq!(deref_core(&eval, "*command-line-args*"), Value::Nil);

        let err_code = run_with_cli_frame(&mut eval, "app.clj", &[], |e| {
            Err(e.throw(crate::error::NativeError::new(
                clj_value::ThrowClass::RuntimeException,
                "boom".to_owned(),
            )))
        });
        assert_eq!(err_code, 1);
        assert!(!eval.force_meta(1));
        assert!(eval.lazy_interpreted(1).is_none());

        let forked = eval.fork_for_thread(crate::env::Bindings::new());
        assert!(forked.lazy_hooks.is_none());

        let ns_expr = crate::expr::Expr::Ns {
            name: "demo.ns".to_owned(),
            doc: None,
            attr: None,
            clauses: Vec::new(),
        };
        assert!(eval.eval(&ns_expr, &crate::env::Env::new()).is_err());

        let closure = Value::Extension(Arc::new(crate::ext::Closure {
            id: 1,
            display: Arc::from("demo"),
            arities: Arc::from([]),
            env: crate::env::Env::new(),
            meta: None,
            conveyed: None,
            vm: None,
        }));
        assert!(eval.call(&closure, &[]).is_err());
        assert!(
            eval.call_owned(&closure, smallvec::SmallVec::new())
                .is_err()
        );

        let mut full = crate::ns::boot();
        let vm_fn = full
            .eval_source("(fn ([x] x) ([x y] x))")
            .expect("vm fn")
            .pop()
            .expect("closure value");
        let p1 = crate::vm::Prepared::of(&vm_fn, 1).expect("p1");
        assert!(p1.call1(&mut eval, Value::Int(1)).is_err());
        let p2 = crate::vm::Prepared::of(&vm_fn, 2).expect("p2");
        assert!(p2.call2(&mut eval, Value::Int(1), Value::Int(2)).is_err());

        let mut lazy_eval = boot_lite(&[("clojure.core", "first")], false);
        let first_id = core_var(&lazy_eval, "first").expect("first var");
        let first_fn = lazy_eval.registry.root_cloned(first_id).expect("first fn");
        let thunk = crate::lazy::thunk0_value(
            Arc::new(vec![crate::expr::Expr::Literal(Value::Nil)]),
            crate::env::Env::new(),
        );
        assert!(lazy_eval.call(&first_fn, &[thunk]).is_err());
    }

    #[test]
    fn lite_string_fallback_covers_join_and_error_paths() {
        let mut eval = boot_lite(
            &[("clojure.string", "join"), ("clojure.string", "split")],
            false,
        );
        let sym = Symbol::new(Some("clojure.string"), "join");
        let join_id = eval
            .registry
            .lookup_in_live_or_removed("clojure.string", &sym)
            .expect("join var");
        let join_fn = eval.registry.root_cloned(join_id).expect("join fn");

        // Non-vector seq & 1-arg join route through lite_string_fallback without core_lazy.
        let one = eval
            .call(
                &join_fn,
                &[Value::List(List::from_vec(vec![
                    Value::Int(1),
                    Value::Int(2),
                ]))],
            )
            .expect("1-arg join");
        assert_eq!(one, Value::Str(Arc::from("12")));

        let two = eval
            .call(
                &join_fn,
                &[
                    Value::Str(Arc::from(",")),
                    Value::List(List::from_vec(vec![
                        Value::Str(Arc::from("a")),
                        Value::Nil,
                        Value::Str(Arc::from("b")),
                    ])),
                ],
            )
            .expect("2-arg join");
        assert_eq!(two, Value::Str(Arc::from("a,,b")));

        let custom_sep = eval
            .call(
                &join_fn,
                &[
                    Value::Int(0),
                    Value::Vector(Vector::from_vec(vec![Value::Int(1), Value::Int(2)])),
                ],
            )
            .expect("custom-sep fallback join");
        assert_eq!(custom_sep, Value::Str(Arc::from("102")));

        assert!(eval.call(&join_fn, &[]).is_err());

        // Arity and nil/type errors on string natives in Lite mode.
        let sym_trim = Symbol::new(Some("clojure.string"), "trim");
        eval.install_native(
            "clojure.string",
            "trim",
            Arc::new(find_lite_string_native("trim").expect("trim")),
        );
        let trim_id = eval
            .registry
            .lookup_in_live_or_removed("clojure.string", &sym_trim)
            .expect("trim id");
        let trim_fn = eval.registry.root_cloned(trim_id).expect("trim fn");
        assert!(eval.call(&trim_fn, &[Value::Nil]).is_err());
        assert!(eval.call(&trim_fn, &[]).is_err());
        assert!(eval.call(&trim_fn, &[Value::Int(1)]).is_err());

        let sym_split = Symbol::new(Some("clojure.string"), "split");
        let split_id = eval
            .registry
            .lookup_in_live_or_removed("clojure.string", &sym_split)
            .expect("split id");
        let split_fn = eval.registry.root_cloned(split_id).expect("split fn");
        assert!(
            eval.call(&split_fn, &[Value::Str(Arc::from("a")), Value::Nil])
                .is_err()
        );
        assert!(eval.call(&split_fn, &[Value::Str(Arc::from("a"))]).is_err());

        let sym_sw = Symbol::new(Some("clojure.string"), "starts-with?");
        eval.install_native(
            "clojure.string",
            "starts-with?",
            Arc::new(find_lite_string_native("starts-with?").expect("starts-with?")),
        );
        let sw_id = eval
            .registry
            .lookup_in_live_or_removed("clojure.string", &sym_sw)
            .expect("sw id");
        let sw_fn = eval.registry.root_cloned(sw_id).expect("sw fn");
        assert!(
            eval.call(&sw_fn, &[Value::Vector(Vector::from_vec(Vec::new()))])
                .is_err()
        );
    }

    #[test]
    fn lite_caps_for_classifies_minimal_capability_sets() {
        assert_eq!(
            lite_caps_for(&[("clojure.core", "println"), ("user", "solve")]),
            LiteCaps::IO
        );
        // Unknown dynamic or user-defined symbols request zero extra capability tables.
        assert_eq!(
            lite_caps_for(&[
                ("clojure.core", "*out*"),
                ("clojure.core", "user-defined"),
                ("clojure.string", "not-native"),
            ]),
            LiteCaps::default()
        );
        // `SEQ` implies `CORE` (because internal seq helpers live in `CORE`).
        let seq_caps = lite_caps_for(&[("clojure.core", "map")]);
        assert!(seq_caps.contains(LiteCaps::SEQ));
        assert!(seq_caps.contains(LiteCaps::CORE));
        assert!(!seq_caps.contains(LiteCaps::RE));
        assert_eq!(
            lite_caps_for(&[
                ("clojure.string", "trim"),
                ("clojure.set", "union"),
                ("clojure.core", "re-find"),
                ("clojure.core", "atom"),
            ]),
            LiteCaps::STRING.union(LiteCaps::RE).union(LiteCaps::HOST)
        );
        // Booting with `seq_helpers = false` omits `LITE_INTERNAL_HELPERS` (`conj`),
        // and unknown core/string names stay uninterned cleanly.
        let io_vars = [
            ("clojure.core", "println"),
            ("clojure.core", "unknown-core"),
            ("clojure.string", "unknown-str"),
        ];
        let io_tables = LiteTables {
            core_groups: &[LITE_CORE_IO],
            owned_groups: &[],
            lib_tables: &[],
        };
        let mut io_only = boot_lite_with_caps(
            &io_vars,
            &[],
            false,
            io_tables,
            false,
            needs_var_meta(&io_vars, true),
        );
        resolve_all(&mut io_only, &io_vars, &[], io_tables);
        assert!(core_var(&io_only, "println").is_some());
        assert!(core_var(&io_only, "conj").is_none());
        // A program that never prints installs no stdout dynamics, and a
        // `clojure.string` name with no native stays uninterned.
        let quiet_vars = [
            ("clojure.core", "inc"),
            ("clojure.string", "no-such-native"),
        ];
        let quiet_tables = LiteTables {
            core_groups: &[LITE_CORE_CORE],
            owned_groups: &[],
            lib_tables: &[("clojure.string", LITE_CAP_STRING)],
        };
        let mut quiet = boot_lite_with_caps(
            &quiet_vars,
            &[],
            false,
            quiet_tables,
            false,
            needs_var_meta(&quiet_vars, true),
        );
        resolve_all(&mut quiet, &quiet_vars, &[], quiet_tables);
        assert!(core_var(&quiet, "inc").is_some());
        assert!(core_var(&quiet, "*out*").is_none());
        let missing = Symbol::new(Some("clojure.string"), "no-such-native");
        assert!(
            quiet
                .registry
                .lookup_in_live_or_removed("clojure.string", &missing)
                .is_none()
        );
    }
}
