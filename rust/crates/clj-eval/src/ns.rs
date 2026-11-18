// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S5A2 namespace vars, the loader, and `boot` (normative:
//! `docs/spec/t1174/namespaces-contract.md`, oracles probe35-36).
//!
//! Natives receive the live evaluator (§N), so `in-ns` switches the
//! registry current, `require` evaluates files, and `eval` re-enters
//! [`Evaluator::eval_top_form`]. Stale handles read the registry's
//! tombstones, like the JVM's detached namespace objects.

use std::sync::Arc;

use clj_reader::{ReaderError, Span};
use clj_value::{
    Keyword, List, Map, Ref, Set, SortSpec, Symbol, ThrowClass, Throwable, Value, Var, Vector,
    new_map, new_set, seq_vec,
};

use crate::concurrency::DEREF_NIL;
use crate::core::cce_err;
use crate::env::{Env, Namespace, ReferTable, SharedRegistry};
use crate::error::{NativeError, Signal, jvm_class_of, site_of};
use crate::eval::{Evaluator, SourceError};
use crate::expr::{Expr, NsClause};
use crate::ext::{NsHandle, as_ns};
use crate::stm::{eval_sync, tx_deref};

/// `clojure.lang.Symbol`, the cast target for namespace arguments.
pub(crate) const SYMBOL_CLASS: &str = "clojure.lang.Symbol";

/// `clojure.lang.Named`, the cast target for lib names.
const NAMED_CLASS: &str = "clojure.lang.Named";

/// The `find`/`findOrCreate` nil shape (a null key into the
/// namespace table, probe35 `find-ns-nil`).
const NIL_KEY_NPE: &str = "Cannot invoke \"Object.hashCode()\" because \"key\" is null";

/// Reading `.ns` off a nil symbol (probe35 `resolve-nil`).
const NIL_SYM_NPE: &str = "Cannot read field \"ns\" because \"sym\" is null";

/// Reading `.getName` off a nil nameable (probe36 `req-nil-lib`).
const NIL_NAME_NPE: &str = "Cannot invoke \"clojure.lang.Named.getName()\" because \"x\" is null";

/// The embedded `clojure.core` composites: `boot` loads this with
/// the current namespace already `clojure.core` (so the file carries
/// intentionally no `ns` form), then refers the publics into `user`.
pub(crate) static CORE_CLJ: &str = include_str!("core.clj");

/// The embedded `clojure.spec.alpha` slice-1 source (gap-122):
/// `resolve_source` serves this for `clojure/spec/alpha` (unlike
/// `CORE_CLJ` it carries its own `ns` form, so it loads on demand
/// through the ordinary `require` path).
const SPEC_ALPHA_CLJ: &str = include_str!("spec_alpha.clj");

/// The embedded `clojure.spec.gen.alpha` slice-F1 source (gap-122):
/// `resolve_source` serves this for `clojure/spec/gen/alpha`, same
/// on-demand shape as [`SPEC_ALPHA_CLJ`].
const SPEC_GEN_ALPHA_CLJ: &str = include_str!("gen_alpha.clj");

/// The embedded `clojure.spec.test.alpha` slice-FDEF source (gap-122):
/// `resolve_source` serves this for `clojure/spec/test/alpha`, same
/// on-demand shape as [`SPEC_ALPHA_CLJ`].
const SPEC_TEST_ALPHA_CLJ: &str = include_str!("spec_test_alpha.clj");

/// The embedded `clojure.java.shell` facsimile (#120 slice 3):
/// served on demand for `clojure/java/shell`, same shape as
/// [`SPEC_ALPHA_CLJ`].
const JAVA_SHELL_CLJ: &str = include_str!("java_shell.clj");

/// The embedded tools.deps config-layer sources (gap-124 slice 1):
/// each carries its own `ns` form and loads on demand through the
/// ordinary `require` path (the `spec_alpha.clj` shape).
/// The embedded upstream stdlib (issue #201 S10): verbatim
/// `clojure/string.clj` + `clojure/set.clj`, served for
/// `clojure/string` + `clojure/set` through the ordinary
/// `require` path (the `spec_alpha.clj` shape), so scripts work
/// with zero install steps.
pub(crate) static STDLIB_STRING_CLJ: &str = include_str!("stdlib/clojure/string.clj");
pub(crate) static STDLIB_SET_CLJ: &str = include_str!("stdlib/clojure/set.clj");
// W1 (verbatim+pure) — do not merge by hand, integrator resolves.
const STDLIB_W1_TEMPLATE_CLJ: &str = include_str!("stdlib/clojure/template.clj");
const STDLIB_W1_WALK_CLJ: &str = include_str!("stdlib/clojure/walk.clj");
const STDLIB_W1_ZIP_CLJ: &str = include_str!("stdlib/clojure/zip.clj");
const STDLIB_W1_DATA_CLJ: &str = include_str!("stdlib/clojure/data.clj");
const STDLIB_W1_DATAFY_CLJ: &str = include_str!("stdlib/clojure/datafy.clj");
// W2 (test) — do not merge by hand, integrator resolves
const STDLIB_TEST_CLJ: &str = include_str!("stdlib/clojure/test.clj");
const STDLIB_TEST_JUNIT_CLJ: &str = include_str!("stdlib/clojure/test/junit.clj");
const STDLIB_TEST_TAP_CLJ: &str = include_str!("stdlib/clojure/test/tap.clj");

// W4 (entry/IO) — do not merge by hand, integrator resolves
const STDLIB_EDN_CLJ: &str = include_str!("stdlib/clojure/edn.clj");
const STDLIB_REPL_CLJ: &str = include_str!("stdlib/clojure/repl.clj");
const STDLIB_MAIN_CLJ: &str = include_str!("stdlib/clojure/main.clj");
const STDLIB_STACKTRACE_CLJ: &str = include_str!("stdlib/clojure/stacktrace.clj");
const STDLIB_CORE_PRINT_CLJ: &str = include_str!("stdlib/clojure/core_print.clj");
// W7 (tools) — do not merge by hand, integrator resolves
const STDLIB_JAVA_BASIS_CLJ: &str = include_str!("stdlib/clojure/java/basis.clj");
const STDLIB_JAVA_BASIS_IMPL_CLJ: &str = include_str!("stdlib/clojure/java/basis/impl.clj");
const STDLIB_TOOLS_DEPS_INTEROP_CLJ: &str = include_str!("stdlib/clojure/tools/deps/interop.clj");
const STDLIB_REPL_DEPS_CLJ: &str = include_str!("stdlib/clojure/repl/deps.clj");
// W8 (core) — do not merge by hand, integrator resolves
const STDLIB_CORE_PROTOCOLS_CLJ: &str = include_str!("stdlib/clojure/core/protocols.clj");
const STDLIB_INSPECTOR_CLJ: &str = include_str!("stdlib/clojure/inspector.clj");
const STDLIB_BROWSE_UI_CLJ: &str = include_str!("stdlib/clojure/java/browse_ui.clj");

// W3 (pprint) — do not merge by hand, integrator resolves
const STDLIB_PPRINT_CLJ: &str = include_str!("stdlib/clojure/pprint.clj");
const STDLIB_PPRINT_UTILITIES_CLJ: &str = include_str!("stdlib/clojure/pprint/utilities.clj");
const STDLIB_PPRINT_COLUMN_WRITER_CLJ: &str =
    include_str!("stdlib/clojure/pprint/column_writer.clj");
const STDLIB_PPRINT_PRETTY_WRITER_CLJ: &str =
    include_str!("stdlib/clojure/pprint/pretty_writer.clj");
const STDLIB_PPRINT_BASE_CLJ: &str = include_str!("stdlib/clojure/pprint/pprint_base.clj");
const STDLIB_PPRINT_CL_FORMAT_CLJ: &str = include_str!("stdlib/clojure/pprint/cl_format.clj");
const STDLIB_PPRINT_DISPATCH_CLJ: &str = include_str!("stdlib/clojure/pprint/dispatch.clj");
const STDLIB_PPRINT_PRINT_TABLE_CLJ: &str = include_str!("stdlib/clojure/pprint/print_table.clj");
// W6 (runtime-ish) — do not merge by hand, integrator resolves.
const W6_SERVER_CLJ: &str = include_str!("stdlib/clojure/core/server.clj");
const W6_REDUCERS_CLJ: &str = include_str!("stdlib/clojure/core/reducers.clj");
const W6_PARALLEL_CLJ: &str = include_str!("stdlib/clojure/parallel.clj");
const W6_CORE_PROXY_CLJ: &str = include_str!("stdlib/clojure/core_proxy.clj");
const W6_XML_CLJ: &str = include_str!("stdlib/clojure/xml.clj");
// W5 (small libs) — do not merge by hand, integrator resolves
const STDLIB_MATH_CLJ: &str = include_str!("stdlib/clojure/math.clj");
const STDLIB_UUID_CLJ: &str = include_str!("stdlib/clojure/uuid.clj");
const STDLIB_INSTANT_CLJ: &str = include_str!("stdlib/clojure/instant.clj");
const STDLIB_JAVADOC_CLJ: &str = include_str!("stdlib/clojure/java/javadoc.clj");
const STDLIB_BROWSE_CLJ: &str = include_str!("stdlib/clojure/java/browse.clj");
const STDLIB_PROCESS_CLJ: &str = include_str!("stdlib/clojure/java/process.clj");
const STDLIB_GVEC_CLJ: &str = include_str!("stdlib/clojure/gvec.clj");

const TOOLS_DEPS_UTIL_DIR_CLJ: &str = include_str!("tools_deps_util_dir.clj");
const TOOLS_DEPS_EDN_CLJ: &str = include_str!("tools_deps_edn.clj");
const TOOLS_DEPS_EXTENSIONS_CLJ: &str = include_str!("tools_deps_extensions.clj");

/// The embedded tools.deps SPI-layer sources (gap-124 slice 2):
/// `util.io` + `util.session` and the `local` / `git` / `deps`
/// extension namespaces, same on-demand shape as slice 1.
const TOOLS_DEPS_UTIL_IO_CLJ: &str = include_str!("tools_deps_util_io.clj");
const TOOLS_DEPS_UTIL_SESSION_CLJ: &str = include_str!("tools_deps_util_session.clj");
const TOOLS_DEPS_EXTENSIONS_LOCAL_CLJ: &str = include_str!("tools_deps_extensions_local.clj");
const TOOLS_DEPS_EXTENSIONS_GIT_CLJ: &str = include_str!("tools_deps_extensions_git.clj");
const TOOLS_DEPS_EXTENSIONS_DEPS_CLJ: &str = include_str!("tools_deps_extensions_deps.clj");

/// The embedded tools.deps API-layer sources (gap-124 slice 3):
/// `util.concurrent` and the main `clojure.tools.deps` namespace,
/// same on-demand shape as slices 1-2.
const TOOLS_DEPS_UTIL_CONCURRENT_CLJ: &str = include_str!("tools_deps_util_concurrent.clj");
const TOOLS_DEPS_CLJ: &str = include_str!("tools_deps.clj");
/// The embedded `clojure.reflect` facsimile (#120 sub-slice 4):
/// upstream map shapes over the emulated lattice, loaded on demand
/// through the ordinary `require` path (the `spec_alpha.clj` shape).
const REFLECT_CLJ: &str = include_str!("reflect.clj");

pub(crate) fn npe(message: &str) -> NativeError {
    NativeError::new(ThrowClass::NullPointerException, message.to_owned())
}

pub(crate) fn ex(message: String) -> NativeError {
    NativeError::new(ThrowClass::Exception, message)
}

pub(crate) fn cast_err(v: &Value, to: &str) -> NativeError {
    NativeError::cast(&jvm_class_of(v), to)
}

/// `str` a value (`nil` prints empty, unlike `print`).
pub(crate) fn str_of(v: &Value) -> String {
    match v {
        Value::Nil => String::new(),
        _ => clj_value::str_of_value(v),
    }
}

/// A symbol as written (`a/b` or `a`).
pub(crate) fn sym_str(s: &Symbol) -> String {
    match s.namespace() {
        Some(ns) => format!("{ns}/{}", s.name()),
        None => s.name().to_owned(),
    }
}

/// A strict namespace-name symbol: nil is the null-key NPE, anything
/// else non-symbol is the `Symbol` cast (the `find`/`findOrCreate`
/// path, probe35 `find-ns-ns-arg`).
pub(crate) fn strict_sym(v: &Value) -> Result<&Symbol, NativeError> {
    match v {
        Value::Symbol(s) => Ok(s),
        Value::Nil => Err(npe(NIL_KEY_NPE)),
        _ => Err(cast_err(v, SYMBOL_CLASS)),
    }
}

/// `clojure.core/the-ns`: a handle resolves to its name (handles
/// never miss, like the JVM's identity return — stale ones read
/// tombstones downstream); a symbol must name a live namespace.
fn the_ns_name(reg: &SharedRegistry, v: &Value) -> Result<String, NativeError> {
    if let Some(h) = as_ns(v) {
        return Ok(h.name.to_string());
    }
    let key = sym_str(strict_sym(v)?);
    if reg.ns_id(&key).is_some() {
        Ok(key)
    } else {
        Err(ex(format!("No namespace: {key} found")))
    }
}

/// A namespace handle value for a live-or-removed namespace.
pub(crate) fn ns_value(name: &str, id: u64) -> Value {
    Value::Extension(Arc::new(NsHandle {
        name: Arc::from(name),
        id,
    }))
}

/// A var value for an id and qualified symbol.
pub(crate) fn var_value(id: u64, sym: &Symbol) -> Value {
    Value::Var(Var {
        id,
        sym: sym.clone(),
    })
}

/// Stamp the `:ns`/`:name` keys onto a var meta map (JVM
/// `Var.intern`: always present, winning over adopted meta).
pub(crate) fn stamped_meta(base: Map, ns_name: &str, ns_id: u64, name: &str) -> Map {
    let ns_key = Value::Keyword(Keyword::new(None, "ns"));
    let name_key = Value::Keyword(Keyword::new(None, "name"));
    base.insert(ns_key, ns_value(ns_name, ns_id))
        .insert(name_key, Value::Symbol(Symbol::new(None, name)))
}

/// Whether a var id is public (`^:private` meta is truthiness-tested,
/// `Var.isPublic`).
pub(crate) fn is_public(reg: &SharedRegistry, id: u64) -> bool {
    let probe = Value::Keyword(Keyword::new(None, "private"));
    !reg.meta_cloned(id)
        .is_some_and(|m| m.get(&probe).is_some_and(Value::is_truthy))
}

/// Resolve a symbol against an explicit namespace home, returning the
/// qualified target and the var id. Refers yield their install-pinned
/// id (the referred object survives re-interns, like the JVM); alias
/// targets may dangle into tombstones, but real-namespace qualifiers
/// must be live. Qualified symbols always resolve aliases-then-global
/// (upstream `Compiler.namespaceFor` — no self-name shortcut, so a
/// same-name alias wins and a removed home misses even for stale
/// handles; T73 round 2).
fn resolve_var_in(home: &Namespace, reg: &SharedRegistry, sym: &Symbol) -> Option<(Symbol, u64)> {
    match sym.namespace() {
        Some(part) => resolve_qualified(home, reg, part, sym),
        None => {
            let q = Symbol::new(Some(home.name()), sym.name());
            if let Some(id) = home.lookup(&q) {
                return Some((q, id));
            }
            home.refers()
                .find(|(n, _, _)| *n == sym.name())
                .map(|(_, t, i)| (t.clone(), i))
        }
    }
}

/// The qualified arm of [`resolve_var_in`]: aliases (which may
/// dangle into tombstones) win over live namespaces.
fn resolve_qualified(
    home: &Namespace,
    reg: &SharedRegistry,
    part: &str,
    sym: &Symbol,
) -> Option<(Symbol, u64)> {
    let target = home
        .aliases()
        .find(|(a, _, _)| *a == part)
        .map(|(_, t, _)| t.to_owned())
        .or_else(|| reg.ns_id(part).map(|_| part.to_owned()))?;
    let q = Symbol::new(Some(target.as_str()), sym.name());
    reg.lookup_in_live_or_removed(&target, &q).map(|id| (q, id))
}

/// Build a symbol-keyed map from `(simple-name, value)` pairs, sorted
/// by name (deterministic; JVM hash order is unspecified).
fn sorted_map(pairs: Vec<(String, Value)>) -> Map {
    let mut pairs = pairs;
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = new_map();
    for (name, v) in pairs {
        out = out.insert(Value::Symbol(Symbol::new(None, &name)), v);
    }
    out
}

/// Fixed arities (`display` is the qualified `clojure.core/name`,
/// probe35 `arity-intern-1`).
pub(crate) fn check_arity(
    display: &str,
    args: &[Value],
    want: &[usize],
) -> Result<(), NativeError> {
    if want.contains(&args.len()) {
        Ok(())
    } else {
        Err(NativeError::arity(display, args.len()))
    }
}

/// The ISeq coercion error (probe36 `req-string`).
fn iseq_err(v: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("Don't know how to create ISeq from: {}", jvm_class_of(v)),
    )
}

/// A `throw` signal as a native error (the class, message, data, and
/// cause survive; the evaluator stamps a fresh identity).
pub(crate) fn throwable_to_native(t: &Throwable) -> NativeError {
    let mut err = NativeError::new(t.class, t.message.as_deref().unwrap_or("").to_owned());
    err.data = t.data.clone().map(|b| *b);
    err.cause = t.cause.clone().map(|b| *b);
    err
}

/// An invoked value's failure as a native error (an escaping `recur`
/// is unreachable — the analyzer cages every one).
pub(crate) fn signal_to_native(s: Signal) -> NativeError {
    match s {
        Signal::Throw(t) => throwable_to_native(&t),
        Signal::Recur(_) => ex("recur escaped its loop or fn".to_owned()),
    }
}

fn native_create_ns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/create-ns", args, &[1])?;
    let key = sym_str(strict_sym(&args[0])?);
    let id = eval.registry.ensure_ns(&key);
    Ok(ns_value(&key, id))
}

fn native_in_ns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/in-ns", args, &[1])?;
    let key = sym_str(strict_sym(&args[0])?);
    let id = eval.registry.ensure_ns(&key);
    eval.registry.set_current(&key);
    // `RT.inNamespace` is a thread-local `set`: the top frame when
    // bound (boot pushes a base frame, so the root never moves),
    // else the root.
    set_core_ns(eval, ns_value(&key, id));
    Ok(ns_value(&key, id))
}

/// Overwrite one var's root.
fn set_root(reg: &SharedRegistry, id: u64, v: Value) {
    reg.with_state_mut(id, |st| {
        st.root = Some(v);
    });
}

/// `in-ns` assignment: the top `*ns*` frame when bound, else the
/// root (`RT.inNamespace` is a thread-local `set`).
fn set_ns_current(eval: &mut Evaluator, var_id: u64, handle: Value) {
    if !eval.set_binding_top(var_id, handle.clone()) {
        set_root(&eval.registry, var_id, handle);
    }
}

/// Point `clojure.core/*ns*` (top frame or root) at `handle`, if booted.
pub(crate) fn set_core_ns(eval: &mut Evaluator, handle: Value) {
    if let Some(var_id) = core_var(eval, "*ns*") {
        set_ns_current(eval, var_id, handle);
    }
}

fn native_ns_name(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/ns-name", args, &[1])?;
    let name = the_ns_name(&eval.registry, &args[0])?;
    Ok(Value::Symbol(Symbol::new(None, &name)))
}

fn native_the_ns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/the-ns", args, &[1])?;
    if as_ns(&args[0]).is_some() {
        return Ok(args[0].clone());
    }
    // Handles returned above; symbols take one live lookup.
    let key = sym_str(strict_sym(&args[0])?);
    match eval.registry.ns_id(&key) {
        Some(id) => Ok(ns_value(&key, id)),
        None => Err(ex(format!("No namespace: {key} found"))),
    }
}

fn native_find_ns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/find-ns", args, &[1])?;
    let key = sym_str(strict_sym(&args[0])?);
    match eval.registry.ns_id(&key) {
        Some(id) => Ok(ns_value(&key, id)),
        None => Ok(Value::Nil),
    }
}

fn native_remove_ns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/remove-ns", args, &[1])?;
    let sym = match &args[0] {
        Value::Nil => {
            return Err(npe(
                "Cannot invoke \"clojure.lang.Symbol.equals(Object)\" because \"name\" is null",
            ));
        }
        Value::Symbol(s) => s,
        v => return Err(cast_err(v, SYMBOL_CLASS)),
    };
    let key = sym_str(sym);
    if key == "clojure.core" {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "Cannot remove clojure namespace".to_owned(),
        ));
    }
    match eval.registry.ns_id(&key) {
        Some(id) => {
            eval.registry.remove_ns(&key);
            Ok(ns_value(&key, id))
        }
        None => Ok(Value::Nil),
    }
}

fn native_all_ns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/all-ns", args, &[0])?;
    let mut out = Vec::new();
    for (name, id) in eval.registry.all_ns_sorted() {
        out.push(ns_value(&name, id));
    }
    Ok(Value::List(List::from_vec(out)))
}

/// The home namespace for the `ns-*` family (handles read
/// tombstones; the `None` arm is unreachable — `the-ns` checked).
fn family_home(eval: &Evaluator, args: &[Value], display: &str) -> Result<Namespace, NativeError> {
    check_arity(display, args, &[1])?;
    let name = the_ns_name(&eval.registry, &args[0])?;
    // Handles pin their generation (stale ones read the tombstone);
    // symbols resolve by name.
    let found = match as_ns(&args[0]).map(|h| h.id) {
        Some(id) => eval.registry.ns_for_handle(&name, id),
        None => eval.registry.ns_live_or_removed(&name),
    };
    match found {
        Some(ns) => Ok(ns),
        None => Err(ex(format!("No namespace: {name} found"))),
    }
}

fn native_ns_map(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let home = family_home(eval, args, "clojure.core/ns-map")?;
    // Imports first, then refers, then interns: refers overwrite
    // imports and interns overwrite both on collision (stable sort
    // keeps that order for equal names).
    let mut pairs = Vec::new();
    for (n, c) in home.imports() {
        pairs.push((n.to_owned(), Value::Class(c.clone())));
    }
    for (n, t, i) in home.refers() {
        pairs.push((n.to_owned(), var_value(i, t)));
    }
    for (s, i) in home.interns() {
        pairs.push((s.name().to_owned(), var_value(i, s)));
    }
    Ok(Value::Map(sorted_map(pairs)))
}

fn native_ns_interns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let home = family_home(eval, args, "clojure.core/ns-interns")?;
    let pairs: Vec<(String, Value)> = home
        .interns()
        .map(|(s, i)| (s.name().to_owned(), var_value(i, s)))
        .collect();
    Ok(Value::Map(sorted_map(pairs)))
}

fn native_ns_publics(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let home = family_home(eval, args, "clojure.core/ns-publics")?;
    let pairs: Vec<(String, Value)> = home
        .interns()
        .filter(|(_, i)| is_public(&eval.registry, *i))
        .map(|(s, i)| (s.name().to_owned(), var_value(i, s)))
        .collect();
    Ok(Value::Map(sorted_map(pairs)))
}

fn native_ns_refers(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let home = family_home(eval, args, "clojure.core/ns-refers")?;
    let pairs: Vec<(String, Value)> = home
        .refers()
        .map(|(n, t, i)| (n.to_owned(), var_value(i, t)))
        .collect();
    Ok(Value::Map(sorted_map(pairs)))
}

fn native_ns_imports(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let home = family_home(eval, args, "clojure.core/ns-imports")?;
    let pairs: Vec<(String, Value)> = home
        .imports()
        .map(|(n, c)| (n.to_owned(), Value::Class(c.clone())))
        .collect();
    Ok(Value::Map(sorted_map(pairs)))
}

fn native_ns_aliases(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let home = family_home(eval, args, "clojure.core/ns-aliases")?;
    let pairs: Vec<(String, Value)> = home
        .aliases()
        .map(|(a, t, i)| (a.to_owned(), ns_value(t, i)))
        .collect();
    // Alias keys keep the installed spelling (qualified aliases
    // re-split on the first `/`, like the reader).
    let mut pairs = pairs;
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = new_map();
    for (name, v) in pairs {
        out = out.insert(Value::Symbol(Symbol::intern(&name)), v);
    }
    Ok(Value::Map(out))
}

fn native_alias(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/alias", args, &[2])?;
    // The target resolves first: call-argument order beats the name
    // cast (core.clj `alias` evaluates `(the-ns ...)` before the
    // `addAlias` checkcast runs).
    let target = the_ns_name(&eval.registry, &args[1])?;
    // The target is live (symbols passed `the-ns`) or tombstoned
    // (handles); tombstones persist, so this lookup cannot miss.
    let target_id = eval
        .registry
        .ns_live_or_removed_id(&target)
        .expect("alias target is live or tombstoned");
    let alias_sym = match &args[0] {
        Value::Nil => return Err(npe("Expecting Symbol + Namespace")),
        Value::Symbol(s) => s,
        v => return Err(cast_err(v, SYMBOL_CLASS)),
    };
    let a = sym_str(alias_sym);
    let cur = eval.current_ns_name();
    eval.registry.ensure_ns(&cur);
    match eval
        .registry
        .with_ns_mut(&cur, |home| home.add_alias(&a, &target, target_id))
    {
        Some(Ok(())) | None => Ok(Value::Nil),
        Some(Err(old)) => Err(NativeError::new(
            ThrowClass::IllegalStateException,
            format!("Alias {a} already exists in namespace {cur}, aliasing {old}"),
        )),
    }
}

/// Option pairs (`(apply hash-map filters)` fails odd counts naming
/// the last item, probe36 `refer-odd-filters`).
fn opt_pairs(filters: &[Value]) -> Result<Vec<(&Value, &Value)>, NativeError> {
    if filters.len() % 2 == 1 {
        let last = &filters[filters.len() - 1];
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("No value supplied for key: {}", str_of(last)),
        ));
    }
    Ok((0..filters.len())
        .step_by(2)
        .map(|i| (&filters[i], &filters[i + 1]))
        .collect())
}

/// Look up an exact (unqualified) keyword option (later pairs win).
fn opt_get<'a>(pairs: &[(&'a Value, &'a Value)], name: &str) -> Option<&'a Value> {
    pairs
        .iter()
        .rev()
        .find(|(k, _)| {
            matches!(k, Value::Keyword(kw) if kw.namespace().is_none() && kw.name() == name)
        })
        .map(|(_, v)| *v)
}

/// The `:only`/`:refer` sequential check (lists, vectors, and
/// entries — strings, maps, and sets fail like the JVM's
/// `Sequential` test).
fn check_sequential(v: &Value) -> Result<Vec<Value>, NativeError> {
    match v {
        Value::List(l) => Ok(l.iter().cloned().collect()),
        Value::Vector(x) => Ok(x.iter().collect()),
        Value::MapEntry { key, val, .. } => Ok(vec![(**key).clone(), (**val).clone()]),
        _ => Err(ex(
            ":only/:refer value must be a sequential collection of symbols".to_owned(),
        )),
    }
}

/// `refer` (and `refer-clojure`, which fixes the lib): refer the
/// source's publics into the current namespace, subject to filters.
/// A namespace's public vars, sorted by name: qualified symbol
/// plus the id pinned for each refer.
/// The `refer` work list: `(or (:refer fs) (:only fs) (keys
/// nspublics))`, with nils falling through and `:all` meaning every
/// public (core.clj `refer`).
fn refer_todo(
    public_names: Vec<Value>,
    pairs: &[(&Value, &Value)],
) -> Result<Vec<Value>, NativeError> {
    let refer_v = opt_get(pairs, "refer").filter(|v| !matches!(*v, Value::Nil));
    let only_v = opt_get(pairs, "only").filter(|v| !matches!(*v, Value::Nil));
    match refer_v {
        Some(Value::Keyword(kw)) if kw.namespace().is_none() && kw.name() == "all" => {
            Ok(public_names)
        }
        Some(v) => check_sequential(v),
        None => match only_v {
            Some(v) => check_sequential(v),
            None => Ok(public_names),
        },
    }
}

/// The `refer` exclusion set (`(set (:exclude fs))`).
fn refer_exclude(pairs: &[(&Value, &Value)]) -> Result<Vec<Value>, NativeError> {
    match opt_get(pairs, "exclude") {
        None | Some(Value::Nil) => Ok(Vec::new()),
        Some(v) => seq_vec(v).ok_or_else(|| iseq_err(v)),
    }
}

/// Resolve one `refer` item to its simple name plus the target var
/// (private and absent vars are the two `IllegalAccessError`s).
fn refer_target(
    eval: &Evaluator,
    lib_key: &str,
    publics: &ReferTable,
    item: &Value,
) -> Result<(Symbol, Symbol, u64), NativeError> {
    let item_sym = match item {
        Value::Symbol(s) if s.namespace().is_none() => s,
        _ => {
            return Err(NativeError::new(
                ThrowClass::IllegalAccessError,
                format!("{} does not exist", str_of(item)),
            ));
        }
    };
    match publics.get(item_sym.name()) {
        Some((target, id)) => Ok(((*item_sym).clone(), target.clone(), *id)),
        None => {
            let interned = eval
                .registry
                .lookup_in_ns(lib_key, &Symbol::new(Some(lib_key), item_sym.name()))
                .is_some();
            let why = if interned {
                "is not public"
            } else {
                "does not exist"
            };
            Err(NativeError::new(
                ThrowClass::IllegalAccessError,
                format!("{} {why}", item_sym.name()),
            ))
        }
    }
}

/// Apply the `:rename` option to one item (a map looks up, anything
/// else invokes with nil falling back to the plain name).
fn refer_renamed(
    eval: &mut Evaluator,
    rename: Option<&Value>,
    item_sym: &Symbol,
    item: &Value,
) -> Result<Symbol, NativeError> {
    let renamed = match rename {
        None => item_sym.clone(),
        Some(Value::Map(m)) => match m.get(item) {
            None => item_sym.clone(),
            Some(Value::Symbol(s)) => s.clone(),
            Some(v) => return Err(cast_err(v, SYMBOL_CLASS)),
        },
        Some(f) => match eval.call(f, std::slice::from_ref(item)) {
            Ok(Value::Nil) => item_sym.clone(),
            Ok(Value::Symbol(s)) => s,
            Ok(v) => return Err(cast_err(&v, SYMBOL_CLASS)),
            Err(s) => return Err(signal_to_native(s)),
        },
    };
    if renamed.namespace().is_some() {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "Can't intern namespace-qualified symbol".to_owned(),
        ));
    }
    Ok(renamed)
}

fn refer_into(eval: &mut Evaluator, lib: &Value, filters: &[Value]) -> Result<Value, NativeError> {
    let lib_key = sym_str(strict_sym(lib)?);
    let Some(publics) = eval.registry.publics_table(&lib_key) else {
        return Err(ex(format!("No namespace: {lib_key}")));
    };
    let pairs = opt_pairs(filters)?;
    match refer_all_exclude(&pairs)? {
        Some(exclude) => {
            refer_all_into_current(eval, &publics, &exclude);
            Ok(Value::Nil)
        }
        None => refer_filtered(eval, &lib_key, &publics.table, &pairs),
    }
}

/// Every public minus `:exclude` (boot's `user`, each `ns` form's
/// `refer-clojure`): the lib's shared table.
fn refer_all_into_current(
    eval: &mut Evaluator,
    publics: &crate::env::PublicsTable,
    exclude: &[Value],
) {
    let exclude: Vec<&str> = exclude
        .iter()
        .filter_map(|e| match e {
            Value::Symbol(x) if x.namespace().is_none() => Some(x.name()),
            _ => None,
        })
        .collect();
    let cur = eval.current_ns_name();
    eval.registry.ensure_ns(&cur);
    eval.registry
        .with_ns_mut(&cur, |home| home.refer_all(publics, &exclude));
}

/// `refer` with `:only`/`:refer`/`:rename`: item by item, in name
/// order (the per-item order — `:rename` fn calls, the stop at a
/// failing item — stays deterministic).
fn refer_filtered(
    eval: &mut Evaluator,
    lib_key: &str,
    publics: &ReferTable,
    pairs: &[(&Value, &Value)],
) -> Result<Value, NativeError> {
    let mut names: Vec<&Arc<str>> = publics.keys().collect();
    names.sort_unstable();
    let public_names: Vec<Value> = names
        .into_iter()
        .map(|n| Value::Symbol(Symbol::new(None, n)))
        .collect();
    let to_do = refer_todo(public_names, pairs)?;
    let exclude = refer_exclude(pairs)?;
    let rename = match opt_get(pairs, "rename") {
        None | Some(Value::Nil) | Some(Value::Bool(false)) => None,
        Some(v) => Some(v),
    };
    // A `:rename` fn runs user code per item, which may observe the
    // refers installed so far; otherwise one batch installs them
    // (`user` refers every `clojure.core` public at boot). Items
    // before a failing one stay installed either way.
    let per_item = rename.is_some_and(|r| !matches!(r, Value::Map(_)));
    let mut pending = Vec::new();
    let mut failure = None;
    for item in to_do.iter().filter(|item| !exclude.contains(item)) {
        let step = refer_target(eval, lib_key, publics, item).and_then(|(item_sym, target, id)| {
            Ok((refer_renamed(eval, rename, &item_sym, item)?, target, id))
        });
        match step {
            Ok(refer) => pending.push(refer),
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
        if per_item {
            install_refers(eval, &mut pending);
        }
    }
    install_refers(eval, &mut pending);
    failure.map_or(Ok(Value::Nil), Err)
}

/// Install pending `(name, target, var id)` refers into the current
/// namespace (only the first symbol's name is read). Interns win
/// over refers (the JVM's REJECTED warning waits for `*err*` in
/// S5D); a namespace without interns skips the check.
fn install_refers(eval: &mut Evaluator, pending: &mut Vec<(Symbol, Symbol, u64)>) {
    if pending.is_empty() {
        return;
    }
    let cur = eval.current_ns_name();
    eval.registry.ensure_ns(&cur);
    eval.registry.with_ns_mut(&cur, |home| {
        let check = home.interns().next().is_some();
        for (renamed, target, id) in pending.drain(..) {
            if check
                && home
                    .lookup(&Symbol::new(Some(home.name()), renamed.name()))
                    .is_some()
            {
                continue;
            }
            home.add_refer(renamed.name(), target, id);
        }
    });
}

/// `refer`'s options when they select every public (no `:refer`,
/// `:only`, or `:rename`): the `:exclude` list, else `None`.
fn refer_all_exclude(pairs: &[(&Value, &Value)]) -> Result<Option<Vec<Value>>, NativeError> {
    let unset = |k| opt_get(pairs, k).is_none_or(|v| matches!(v, Value::Nil));
    let no_rename =
        opt_get(pairs, "rename").is_none_or(|v| matches!(v, Value::Nil | Value::Bool(false)));
    if unset("refer") && unset("only") && no_rename {
        return refer_exclude(pairs).map(Some);
    }
    Ok(None)
}

fn native_refer(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.is_empty() {
        return Err(NativeError::arity("clojure.core/refer", 0));
    }
    refer_into(eval, &args[0], &args[1..])
}

fn native_refer_clojure(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let lib = Value::Symbol(Symbol::new(None, "clojure.core"));
    refer_into(eval, &lib, args)
}

/// Route an `intern` through a handle pin (stale handles land in
/// the tombstone, which becomes the var's home) or by name (live
/// first). Returns the var id and the home namespace id.
pub(crate) fn intern_routed(
    eval: &mut Evaluator,
    ns_name: &str,
    pinned: Option<u64>,
    q: &Symbol,
) -> Result<(u64, u64), NativeError> {
    match pinned {
        Some(expected) => eval
            .registry
            .intern_in_pinned(ns_name, expected, q)
            .ok_or_else(|| ex(format!("No namespace: {ns_name} found"))),
        None => {
            let id = eval
                .registry
                .intern_in(ns_name, q)
                .ok_or_else(|| ex(format!("No namespace: {ns_name} found")))?;
            Ok((id, eval.registry.ns_id(ns_name).unwrap_or(0)))
        }
    }
}

fn native_intern(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/intern", args, &[2, 3])?;
    let ns_name = the_ns_name(&eval.registry, &args[0])?;
    let name = match &args[1] {
        Value::Nil => return Err(npe(NIL_SYM_NPE)),
        Value::Symbol(s) if s.namespace().is_some() => {
            return Err(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Can't intern namespace-qualified symbol".to_owned(),
            ));
        }
        Value::Symbol(s) => s,
        v => return Err(cast_err(v, SYMBOL_CLASS)),
    };
    let meta = name.meta().cloned();
    let q = Symbol::new(Some(ns_name.as_str()), name.name());
    let pinned = as_ns(&args[0]).map(|h| h.id);
    let (id, home) = intern_routed(eval, &ns_name, pinned, &q)?;
    // A lazy `core.clj` var forces first (a later force would undo
    // this intern's root and meta).
    eval.force_lazy(id);
    eval.registry.take_meta_lazy(id);
    // The var adopts the name symbol's meta, if any (core.clj
    // `intern`); 3-arity resets the root on the same var. The
    // `:ns`/`:name` keys are always (re)stamped (JVM
    // `Var.intern`; a cleared meta stays cleared).
    let reset = args.get(2).cloned();
    eval.registry.with_state_mut(id, |st| {
        if let Some(m) = meta {
            st.meta = Some(m);
        }
        if let Some(m) = st.meta.clone() {
            st.meta = Some(stamped_meta(m, &ns_name, home, name.name()));
        }
        if let Some(v) = reset {
            st.root = Some(v);
        }
    });
    Ok(var_value(id, &q))
}

/// `ns-resolve`/`resolve` through an explicit home (the env check
/// runs before symbol validation, like the `when-not` in
/// `ns-resolve`).
pub(crate) fn do_resolve(
    eval: &Evaluator,
    ns_name: &str,
    pinned: Option<u64>,
    env: Option<&Value>,
    sym_v: &Value,
) -> Result<Value, NativeError> {
    if let Some(e) = env {
        if env_shadows(e, sym_v) {
            return Ok(Value::Nil);
        }
    }
    let sym = match sym_v {
        Value::Nil => return Err(npe(NIL_SYM_NPE)),
        Value::Symbol(s) => s,
        v => return Err(cast_err(v, SYMBOL_CLASS)),
    };
    // Handles pin their generation (stale ones resolve the tombstone
    // for unqualified symbols; qualified ones go aliases-then-global
    // per `resolve_var_in`, like upstream `namespaceFor`).
    let home = match pinned {
        Some(id) => eval.registry.ns_for_handle(ns_name, id),
        None => eval.registry.ns_live_or_removed(ns_name),
    };
    if let Some((q, id)) = home
        .as_ref()
        .and_then(|h| resolve_var_in(h, &eval.registry, sym))
    {
        return Ok(var_value(id, &q));
    }
    // Imports resolve after vars (unqualified only — imports are
    // simple names; the value is the class, not a var).
    if sym.namespace().is_none() {
        if let Some(class) = home.as_ref().and_then(|h| h.get_import(sym.name())) {
            return Ok(Value::Class(class));
        }
        if let Some(class) = resolve_class_position(eval, home.as_ref(), sym) {
            return Ok(Value::Class(class));
        }
    }
    Ok(Value::Nil)
}

/// FQ class-position resolve (split from [`do_resolve`] for the
/// complexity budget, like [`env_shadows`]): dotted names resolve
/// as classes like the analyzer (oracle `resolve-fq`); missing
/// ones are nil (oracle `resolve-missing` — `resolve` never
/// throws here).
fn resolve_class_position(
    eval: &Evaluator,
    home: Option<&Namespace>,
    sym: &Symbol,
) -> Option<std::sync::Arc<clj_value::ClassVal>> {
    let _ = home?;
    if crate::classemu::is_class_position(sym.name()) {
        crate::classemu::resolve_class_name(eval, sym.name())
    } else {
        None
    }
}

/// Whether the `&env` argument shadows a resolve (split from
/// [`do_resolve`] for the complexity budget): a map or set
/// containing the symbol resolves `nil` (locals win).
fn env_shadows(env: &Value, sym_v: &Value) -> bool {
    match env {
        Value::Map(m) => m.contains_key(sym_v),
        Value::Set(s) => s.contains(sym_v),
        _ => false,
    }
}

fn native_ns_resolve(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/ns-resolve", args, &[2, 3])?;
    let ns_name = the_ns_name(&eval.registry, &args[0])?;
    let (env, sym_v) = if args.len() == 3 {
        (Some(&args[1]), &args[2])
    } else {
        (None, &args[1])
    };
    let pinned = as_ns(&args[0]).map(|h| h.id);
    do_resolve(eval, &ns_name, pinned, env, sym_v)
}

fn native_resolve(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/resolve", args, &[1, 2])?;
    let cur = eval.current_ns_name();
    let (env, sym_v) = if args.len() == 2 {
        (Some(&args[0]), &args[1])
    } else {
        (None, &args[0])
    };
    do_resolve(eval, &cur, None, env, sym_v)
}

/// `find-var`: the interned var named by a namespace-qualified
/// symbol, or nil when the namespace exists but the var does not
/// (S5; oracle `f-*`). Refers are blind (interns only, like
/// `Var.find`); a missing namespace and an unqualified symbol
/// throw `IllegalArgumentException`, like the JVM.
fn native_find_var(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/find-var", args, &[1])?;
    let sym = match &args[0] {
        Value::Nil => {
            return Err(npe(
                "Cannot read field \"ns\" because \"nsQualifiedSym\" is null",
            ));
        }
        Value::Symbol(s) => s,
        other => return Err(cce_err(other, SYMBOL_CLASS)),
    };
    let Some(ns_part) = sym.namespace() else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            "Symbol must be namespace-qualified".to_owned(),
        ));
    };
    let Some(home) = eval.registry.get(ns_part) else {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!("No such namespace: {ns_part}"),
        ));
    };
    match home.lookup(sym) {
        Some(id) => Ok(var_value(id, sym)),
        None => Ok(Value::Nil),
    }
}

/// `requiring-resolve`: resolve a namespace-qualified symbol in
/// the current ns, requiring its namespace first when the initial
/// resolve misses (S5; oracle `g-*`). Loader failures propagate
/// in port shapes.
fn native_requiring_resolve(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/requiring-resolve", args, &[1])?;
    let ns_part = match &args[0] {
        Value::Symbol(s) if s.namespace().is_some() => s.namespace().unwrap_or("").to_owned(),
        _ => {
            return Err(NativeError::new(
                ThrowClass::IllegalArgumentException,
                format!("Not a qualified symbol: {}", str_of(&args[0])),
            ));
        }
    };
    let cur = eval.current_ns_name();
    let first = do_resolve(eval, &cur, None, None, &args[0])?;
    if !matches!(first, Value::Nil) {
        return Ok(first);
    }
    let lib = Value::Symbol(Symbol::new(None, &ns_part));
    load_libs(eval, &["require"], std::slice::from_ref(&lib))?;
    do_resolve(eval, &cur, None, None, &args[0])
}

fn native_ns_unmap(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/ns-unmap", args, &[2])?;
    let ns_name = the_ns_name(&eval.registry, &args[0])?;
    let simple = match &args[1] {
        Value::Nil => return Err(npe(NIL_SYM_NPE)),
        Value::Symbol(s) if s.namespace().is_none() => s.name().to_owned(),
        Value::Symbol(_) => {
            return Err(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Can't unintern namespace-qualified symbol".to_owned(),
            ));
        }
        v => return Err(cast_err(v, SYMBOL_CLASS)),
    };
    // Handles pin their generation (stale ones unmap the tombstone).
    match as_ns(&args[0]).map(|h| h.id) {
        Some(id) => eval
            .registry
            .with_handle_mut(&ns_name, id, |home| home.unmap(&simple)),
        None => eval
            .registry
            .with_live_or_removed_mut(&ns_name, |home| home.unmap(&simple)),
    };
    Ok(Value::Nil)
}

fn native_ns_unalias(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/ns-unalias", args, &[2])?;
    let ns_name = the_ns_name(&eval.registry, &args[0])?;
    let simple = match &args[1] {
        Value::Nil => return Ok(Value::Nil),
        Value::Symbol(s) => sym_str(s),
        v => return Err(cast_err(v, SYMBOL_CLASS)),
    };
    // Handles pin their generation (stale ones unalias the tombstone).
    match as_ns(&args[0]).map(|h| h.id) {
        Some(id) => eval
            .registry
            .with_handle_mut(&ns_name, id, |home| home.remove_alias(&simple)),
        None => eval
            .registry
            .with_live_or_removed_mut(&ns_name, |home| home.remove_alias(&simple)),
    };
    Ok(Value::Nil)
}

/// Simple JVM exception names mapped to [`ThrowClass`].
const THROW_CLASSES: &[(&str, ThrowClass)] = &[
    ("Exception", ThrowClass::Exception),
    ("Error", ThrowClass::Error),
    ("RuntimeException", ThrowClass::RuntimeException),
    (
        "IllegalArgumentException",
        ThrowClass::IllegalArgumentException,
    ),
    ("IllegalStateException", ThrowClass::IllegalStateException),
    (
        "IllegalMonitorStateException",
        ThrowClass::IllegalMonitorStateException,
    ),
    (
        "UnsupportedOperationException",
        ThrowClass::UnsupportedOperationException,
    ),
    ("AssertionError", ThrowClass::AssertionError),
    ("ArityException", ThrowClass::ArityException),
    ("NullPointerException", ThrowClass::NullPointerException),
    ("ClassCastException", ThrowClass::ClassCastException),
    ("ClassNotFoundException", ThrowClass::ClassNotFoundException),
    (
        "IndexOutOfBoundsException",
        ThrowClass::IndexOutOfBoundsException,
    ),
    (
        "ArrayIndexOutOfBoundsException",
        ThrowClass::ArrayIndexOutOfBoundsException,
    ),
    ("ArrayStoreException", ThrowClass::ArrayStoreException),
    (
        "NegativeArraySizeException",
        ThrowClass::NegativeArraySizeException,
    ),
    ("OutOfMemoryError", ThrowClass::OutOfMemoryError),
    ("ArithmeticException", ThrowClass::ArithmeticException),
    ("ExceptionInfo", ThrowClass::ExceptionInfo),
    ("FileNotFoundException", ThrowClass::FileNotFoundException),
    ("ExecutionException", ThrowClass::ExecutionException),
    ("CancellationException", ThrowClass::CancellationException),
    (
        "RejectedExecutionException",
        ThrowClass::RejectedExecutionException,
    ),
    ("IllegalAccessError", ThrowClass::IllegalAccessError),
    ("NumberFormatException", ThrowClass::NumberFormatException),
    (
        "StringIndexOutOfBoundsException",
        ThrowClass::StringIndexOutOfBoundsException,
    ),
    (
        "IllegalFormatConversionException",
        ThrowClass::IllegalFormatConversionException,
    ),
    (
        "UnknownFormatConversionException",
        ThrowClass::UnknownFormatConversionException,
    ),
    (
        "MissingFormatConversionException",
        ThrowClass::MissingFormatConversionException,
    ),
    (
        "MissingFormatArgumentException",
        ThrowClass::MissingFormatArgumentException,
    ),
    (
        "IllegalFormatPrecisionException",
        ThrowClass::IllegalFormatPrecisionException,
    ),
    (
        "IllegalFormatCodePointException",
        ThrowClass::IllegalFormatCodePointException,
    ),
    ("IOException", ThrowClass::IOException),
    ("Compiler$CompilerException", ThrowClass::CompilerException),
    ("LispReader$ReaderException", ThrowClass::ReaderException),
    ("ReaderException", ThrowClass::ReaderException),
];

/// A JVM class name (simple or fully qualified) to its
/// [`ThrowClass`]. Unknown names — including the port-only
/// `ReaderError` deferrals — fall back to `Exception`.
pub(crate) fn throw_class_for(name: &str) -> ThrowClass {
    let short = name.rsplit('.').next().unwrap_or(name);
    THROW_CLASSES
        .iter()
        .find_map(|&(k, cls)| (k == short).then_some(cls))
        .unwrap_or(ThrowClass::Exception)
}

/// A load/eval failure as a native error: a real
/// `CompilerException` wrapper with the structured cause (so `catch`
/// and `ex-cause` walk the JVM chain); bare failures (macro arity)
/// escape unwrapped, cause class and cause message only.
pub(crate) fn source_to_native(eval: &mut Evaluator, source: &str, e: SourceError) -> NativeError {
    match e {
        SourceError::Reader(e) => {
            let span = e.span();
            let mut outer = NativeError::new(
                ThrowClass::CompilerException,
                format!(
                    "Syntax error reading source at ({}).",
                    site_of(source, span)
                ),
            );
            // S7: a `#=`/resolver throwable rides as the cause
            // intact (its `ex-data` and cause chain survive the
            // wrap, like the JVM).
            outer.cause = Some(match e {
                ReaderError::EvalFailed {
                    error: clj_reader::EvalReadError::Failed(t),
                    ..
                } => t.clone(),
                _ => Throwable {
                    id: eval.next_id(),
                    class: throw_class_for(e.jvm_class()),
                    message: Some(Arc::from(e.message().as_str())),
                    data: None,
                    cause: None,
                },
            });
            outer
        }
        SourceError::Compile(e) => {
            // `compile_err` converts every bare failure to `Uncaught`
            // before it escapes (`eval_top_form`/`load_source` map
            // through it), so `e.bare` is always false here (all four
            // callers pass converted errors only).
            let mut outer = NativeError::new(ThrowClass::CompilerException, e.wrapper());
            outer.cause = Some(Throwable {
                id: eval.next_id(),
                class: throw_class_for(e.class),
                message: Some(Arc::from(e.message.as_str())),
                data: None,
                cause: None,
            });
            outer
        }
        SourceError::Uncaught(t) => throwable_to_native(&t),
        SourceError::Internal(m) => ex(format!("internal error: {m}")),
    }
}

/// A `clojure.core` var id, if installed.
/// A `Var` handle for a `clojure.core` var id (binding frames
/// carry the handle so snapshots print without re-lookup).
pub(crate) fn core_var_handle(id: u64, name: &str) -> Var {
    Var {
        id,
        sym: Symbol::new(Some("clojure.core"), name),
    }
}

pub(crate) fn core_var(eval: &Evaluator, name: &str) -> Option<u64> {
    eval.registry.lookup_str("clojure.core", name)
}

/// Push a thread binding for `clojure.core/<name>` if that core var exists.
pub(crate) fn push_core_binding(eval: &mut Evaluator, name: &str, value: Value) {
    if let Some(id) = core_var(eval, name) {
        eval.push_binding(core_var_handle(id, name), value);
    }
}

/// Stamp `:arglists` meta on an installed `clojure.core` var
/// (one vector per arity; S8 single-arity calls route through
/// here too).
pub(crate) fn set_arglists_multi(eval: &mut Evaluator, name: &str, arities: &[&[&str]]) {
    let q = Symbol::new(Some("clojure.core"), name);
    let id = eval.registry.intern(&q);
    let arglists = Value::List(List::from_vec(
        arities
            .iter()
            .map(|params| {
                Value::Vector(Vector::from_vec(
                    params
                        .iter()
                        .map(|p| Value::Symbol(Symbol::new(None, p)))
                        .collect(),
                ))
            })
            .collect(),
    ));
    eval.registry.with_state_mut(id, |st| {
        let meta = st
            .meta
            .clone()
            .unwrap_or_else(new_map)
            .insert(Value::Keyword(Keyword::new(None, "arglists")), arglists);
        st.meta = Some(meta);
    });
}

/// The visible value of a `clojure.core` var (total fallback nil —
/// boot installs every var the loader touches).
pub(crate) fn deref_core(eval: &Evaluator, name: &'static str) -> Value {
    eval.core_read(name)
}

/// The visible `*loaded-libs*` set: the var holds a `Ref`
/// (JVM core.clj `(ref (sorted-set))`), so read exactly like
/// `@` — a transaction-aware deref, and `deref`'s own cast
/// error on anything else (probe H1).
fn libs_set(eval: &mut Evaluator) -> Result<Set, NativeError> {
    let derefed = match deref_core(eval, "*loaded-libs*") {
        Value::Ref(reff) => tx_deref(eval, &reff),
        Value::Nil => return Err(npe(DEREF_NIL)),
        v => return Err(cce_err(&v, "java.util.concurrent.Future")),
    };
    match derefed {
        Value::Set(s) => Ok(s),
        v => Err(cast_err(&v, "clojure.lang.IPersistentSet")),
    }
}

/// Mark a lib loaded, exactly like the JVM's `load-one`:
/// `(dosync (commute *loaded-libs* conj lib))` (core.clj).
/// `dosync` joins an enclosing transaction instead of nesting.
fn mark_loaded(eval: &mut Evaluator, lib: &Symbol) -> Result<(), NativeError> {
    mark_member(eval, &Value::Symbol(lib.clone()))
}

/// Conjoin one member into `*loaded-libs*` through the real `sync`
/// machinery: `(commute REF conj MEMBER)` built as expressions and
/// run through `eval_sync` (join-or-begin plus retries — exactly
/// what the `dosync` form expands to, minus the source round-trip;
/// the member passes at the Rust level, so every value conjoins —
/// nothing is ever dropped or left unrenderable). Union-or-throw,
/// like the JVM's `reduce1 conj`: symbols and nil commit, while an
/// incomparable exotic (reachable only via blank-ref `ref-set`
/// sabotage, which skips the commute's own comparison) fails the
/// load with the comparator's CCE. Marks commit one member per
/// transaction, so a sabotaged union can land earlier members
/// before the CCE aborts the load (the JVM's single union commute
/// is all-or-nothing — accepted narrowing, sabotage-only).
fn mark_member(eval: &mut Evaluator, member: &Value) -> Result<(), NativeError> {
    let commute = deref_core(eval, "commute");
    let conj = deref_core(eval, "conj");
    let reff = deref_core(eval, "*loaded-libs*");
    let call = Expr::Invoke {
        head: Box::new(Expr::Literal(commute)),
        args: vec![
            Expr::Literal(reff),
            Expr::Literal(conj),
            Expr::Literal(member.clone()),
        ],
        span: Span::new(1, 1),
    };
    let env = Env::new();
    eval_sync(eval, &|ev: &mut Evaluator| {
        ev.eval_body(std::slice::from_ref(&call), &env)
    })
    .map(|_| ())
    .map_err(signal_to_native)
}

/// `(name x)`: strings pass through, symbols and keywords give their
/// name, nil is the `getName` NPE, anything else the `Named` cast
/// (core.clj `name`, probe35 `name-num`).
pub(crate) fn name_of(v: &Value) -> Result<String, NativeError> {
    match v {
        Value::Str(s) => Ok(s.to_string()),
        Value::Symbol(s) => Ok(s.name().to_owned()),
        Value::Keyword(k) => Ok(k.name().to_owned()),
        Value::Nil => Err(npe(NIL_NAME_NPE)),
        _ => Err(cast_err(v, NAMED_CLASS)),
    }
}

/// The root resource for a lib name: dashes to underscores, dots to
/// slashes, with a leading slash (core.clj `root-resource`).
pub(crate) fn root_resource(lib_name: &str) -> String {
    let mapped: String = lib_name
        .chars()
        .map(|c| match c {
            '-' => '_',
            '.' => '/',
            _ => c,
        })
        .collect();
    format!("/{mapped}")
}

/// The root directory for a namespace name (core.clj
/// `root-directory`): the resource minus its last segment.
fn root_directory(ns_name: &str) -> String {
    let r = root_resource(ns_name);
    r.rfind('/').map_or(String::new(), |i| r[..i].to_owned())
}

/// The missing-file error, verbatim (`RT.load`: the three tried
/// names plus the underscore hint, probe36 `req-missing`).
fn could_not_locate(root: &str) -> NativeError {
    let mut msg =
        format!("Could not locate {root}__init.class, {root}.clj or {root}.cljc on classpath.");
    if root.contains('_') {
        msg.push_str(
            " Please check that namespaces with dashes use underscores in the Clojure file name.",
        );
    }
    NativeError::new(ThrowClass::FileNotFoundException, msg)
}

/// A filesystem failure as `FileNotFoundException` with the path as
/// given (Rust's ` (os error N)` suffix is stripped — the JVM shape
/// is the bare OS message, probe36 `loadfile-missing`).
pub(crate) fn fs_err(path: &str, e: &std::io::Error) -> NativeError {
    let mut msg = e.to_string();
    if let Some(i) = msg.rfind(" (os error ").filter(|_| msg.ends_with(')')) {
        msg.truncate(i);
    }
    NativeError::new(ThrowClass::FileNotFoundException, format!("{path} ({msg})"))
}

/// The `*load-path*` directories (seqable, lenient nil; entries must
/// be strings). Shared with `io/resource` (#205).
pub(crate) fn load_dirs(eval: &Evaluator) -> Result<Vec<String>, NativeError> {
    let v = deref_core(eval, "*load-path*");
    let items = seq_vec(&v).ok_or_else(|| cast_err(&v, "clojure.lang.IPersistentVector"))?;
    let mut out = Vec::new();
    for item in &items {
        match item {
            Value::Str(s) => out.push(s.to_string()),
            _ => return Err(cast_err(item, "java.lang.String")),
        }
    }
    Ok(out)
}

/// Embedded library sources by load root (no leading slash):
/// the `*file*` path and text served without touching
/// `*load-path*`. New embedded namespaces add a ROW here, never
/// a branch in [`resolve_source`] (complexity ratchet: the
/// ten-`if` chain tripped it at cognitive 20).
const EMBEDDED_SOURCES: &[(&str, &str, &str)] = &[
    (
        "clojure/spec/alpha",
        "clojure/spec/alpha.clj",
        SPEC_ALPHA_CLJ,
    ),
    (
        "clojure/spec/gen/alpha",
        "clojure/spec/gen/alpha.clj",
        SPEC_GEN_ALPHA_CLJ,
    ),
    (
        "clojure/spec/test/alpha",
        "clojure/spec/test/alpha.clj",
        SPEC_TEST_ALPHA_CLJ,
    ),
    (
        "clojure/java/shell",
        "clojure/java/shell.clj",
        JAVA_SHELL_CLJ,
    ),
    (
        "clojure/tools/deps/util/dir",
        "clojure/tools/deps/util/dir.clj",
        TOOLS_DEPS_UTIL_DIR_CLJ,
    ),
    (
        "clojure/tools/deps/edn",
        "clojure/tools/deps/edn.clj",
        TOOLS_DEPS_EDN_CLJ,
    ),
    (
        "clojure/tools/deps/extensions",
        "clojure/tools/deps/extensions.clj",
        TOOLS_DEPS_EXTENSIONS_CLJ,
    ),
    (
        "clojure/tools/deps/util/io",
        "clojure/tools/deps/util/io.clj",
        TOOLS_DEPS_UTIL_IO_CLJ,
    ),
    (
        "clojure/tools/deps/util/session",
        "clojure/tools/deps/util/session.clj",
        TOOLS_DEPS_UTIL_SESSION_CLJ,
    ),
    (
        "clojure/tools/deps/extensions/local",
        "clojure/tools/deps/extensions/local.clj",
        TOOLS_DEPS_EXTENSIONS_LOCAL_CLJ,
    ),
    (
        "clojure/tools/deps/extensions/git",
        "clojure/tools/deps/extensions/git.clj",
        TOOLS_DEPS_EXTENSIONS_GIT_CLJ,
    ),
    (
        "clojure/tools/deps/extensions/deps",
        "clojure/tools/deps/extensions/deps.clj",
        TOOLS_DEPS_EXTENSIONS_DEPS_CLJ,
    ),
    (
        "clojure/tools/deps/util/concurrent",
        "clojure/tools/deps/util/concurrent.clj",
        TOOLS_DEPS_UTIL_CONCURRENT_CLJ,
    ),
    (
        "clojure/tools/deps",
        "clojure/tools/deps.clj",
        TOOLS_DEPS_CLJ,
    ),
    ("clojure/reflect", "clojure/reflect.clj", REFLECT_CLJ),
    ("clojure/string", "clojure/string.clj", STDLIB_STRING_CLJ),
    ("clojure/set", "clojure/set.clj", STDLIB_SET_CLJ),
    // W1 (verbatim+pure) — do not merge by hand, integrator resolves.
    (
        "clojure/template",
        "clojure/template.clj",
        STDLIB_W1_TEMPLATE_CLJ,
    ),
    ("clojure/walk", "clojure/walk.clj", STDLIB_W1_WALK_CLJ),
    ("clojure/zip", "clojure/zip.clj", STDLIB_W1_ZIP_CLJ),
    ("clojure/data", "clojure/data.clj", STDLIB_W1_DATA_CLJ),
    ("clojure/datafy", "clojure/datafy.clj", STDLIB_W1_DATAFY_CLJ),
    // W2 (test) — do not merge by hand, integrator resolves
    ("clojure/test", "clojure/test.clj", STDLIB_TEST_CLJ),
    (
        "clojure/test/junit",
        "clojure/test/junit.clj",
        STDLIB_TEST_JUNIT_CLJ,
    ),
    (
        "clojure/test/tap",
        "clojure/test/tap.clj",
        STDLIB_TEST_TAP_CLJ,
    ),
    // W4 (entry/IO) — do not merge by hand, integrator resolves
    ("clojure/edn", "clojure/edn.clj", STDLIB_EDN_CLJ),
    ("clojure/repl", "clojure/repl.clj", STDLIB_REPL_CLJ),
    ("clojure/main", "clojure/main.clj", STDLIB_MAIN_CLJ),
    (
        "clojure/stacktrace",
        "clojure/stacktrace.clj",
        STDLIB_STACKTRACE_CLJ,
    ),
    (
        "clojure/core_print",
        "clojure/core_print.clj",
        STDLIB_CORE_PRINT_CLJ,
    ),
    // W7 (tools) — do not merge by hand, integrator resolves
    (
        "clojure/java/basis",
        "clojure/java/basis.clj",
        STDLIB_JAVA_BASIS_CLJ,
    ),
    (
        "clojure/java/basis/impl",
        "clojure/java/basis/impl.clj",
        STDLIB_JAVA_BASIS_IMPL_CLJ,
    ),
    (
        "clojure/tools/deps/interop",
        "clojure/tools/deps/interop.clj",
        STDLIB_TOOLS_DEPS_INTEROP_CLJ,
    ),
    (
        "clojure/repl/deps",
        "clojure/repl/deps.clj",
        STDLIB_REPL_DEPS_CLJ,
    ),
    // W8 (core) — do not merge by hand, integrator resolves
    (
        "clojure/core/protocols",
        "clojure/core/protocols.clj",
        STDLIB_CORE_PROTOCOLS_CLJ,
    ),
    (
        "clojure/inspector",
        "clojure/inspector.clj",
        STDLIB_INSPECTOR_CLJ,
    ),
    (
        "clojure/java/browse_ui",
        "clojure/java/browse_ui.clj",
        STDLIB_BROWSE_UI_CLJ,
    ),
    // W3 (pprint) — do not merge by hand, integrator resolves
    ("clojure/pprint", "clojure/pprint.clj", STDLIB_PPRINT_CLJ),
    (
        "clojure/pprint/utilities",
        "clojure/pprint/utilities.clj",
        STDLIB_PPRINT_UTILITIES_CLJ,
    ),
    (
        "clojure/pprint/column_writer",
        "clojure/pprint/column_writer.clj",
        STDLIB_PPRINT_COLUMN_WRITER_CLJ,
    ),
    (
        "clojure/pprint/pretty_writer",
        "clojure/pprint/pretty_writer.clj",
        STDLIB_PPRINT_PRETTY_WRITER_CLJ,
    ),
    (
        "clojure/pprint/pprint_base",
        "clojure/pprint/pprint_base.clj",
        STDLIB_PPRINT_BASE_CLJ,
    ),
    (
        "clojure/pprint/cl_format",
        "clojure/pprint/cl_format.clj",
        STDLIB_PPRINT_CL_FORMAT_CLJ,
    ),
    (
        "clojure/pprint/dispatch",
        "clojure/pprint/dispatch.clj",
        STDLIB_PPRINT_DISPATCH_CLJ,
    ),
    (
        "clojure/pprint/print_table",
        "clojure/pprint/print_table.clj",
        STDLIB_PPRINT_PRINT_TABLE_CLJ,
    ),
    // W6 (runtime-ish) — do not merge by hand, integrator resolves.
    (
        "clojure/core/server",
        "clojure/core/server.clj",
        W6_SERVER_CLJ,
    ),
    (
        "clojure/core/reducers",
        "clojure/core/reducers.clj",
        W6_REDUCERS_CLJ,
    ),
    ("clojure/parallel", "clojure/parallel.clj", W6_PARALLEL_CLJ),
    (
        "clojure/core_proxy",
        "clojure/core_proxy.clj",
        W6_CORE_PROXY_CLJ,
    ),
    ("clojure/xml", "clojure/xml.clj", W6_XML_CLJ),
    // W5 (small libs) — do not merge by hand, integrator resolves
    ("clojure/math", "clojure/math.clj", STDLIB_MATH_CLJ),
    ("clojure/uuid", "clojure/uuid.clj", STDLIB_UUID_CLJ),
    ("clojure/instant", "clojure/instant.clj", STDLIB_INSTANT_CLJ),
    (
        "clojure/java/javadoc",
        "clojure/java/javadoc.clj",
        STDLIB_JAVADOC_CLJ,
    ),
    (
        "clojure/java/browse",
        "clojure/java/browse.clj",
        STDLIB_BROWSE_CLJ,
    ),
    (
        "clojure/java/process",
        "clojure/java/process.clj",
        STDLIB_PROCESS_CLJ,
    ),
    ("clojure/gvec", "clojure/gvec.clj", STDLIB_GVEC_CLJ),
];

/// Resolve a load root (no leading slash) to the `*file*` path and
/// text: embedded libs first (gap-122 slices 1+F1:
/// `clojure/spec/alpha` + `clojure/spec/gen/alpha`; gap-124 slices
/// 1+2: the three tools.deps config namespaces plus the five SPI
/// namespaces; slice 3: `util.concurrent` + the API namespace;
/// #120 sub-slice 4: `clojure/reflect`; the flat set lives here
/// when S5D lands), then `.clj`
/// across every `*load-path*` dir, then `.cljc` (`RT.load` lookup
/// order).
fn resolve_source(eval: &Evaluator, root: &str) -> Result<(String, String), NativeError> {
    if let Some(&(_, file, text)) = EMBEDDED_SOURCES.iter().find(|(r, _, _)| *r == root) {
        return Ok((file.to_owned(), text.to_owned()));
    }
    let dirs = load_dirs(eval)?;
    for ext in ["clj", "cljc"] {
        for dir in &dirs {
            // `*file*` pins `/` separators on every platform (S5A2
            // file_star_during_load); Win32 file access accepts them.
            let candidate = format!("{dir}/{root}.{ext}").replace('\\', "/");
            if std::path::Path::new(&candidate).is_file() {
                match std::fs::read_to_string(&candidate) {
                    Ok(text) => return Ok((candidate, text)),
                    Err(e) => return Err(fs_err(&candidate, &e)),
                }
            }
        }
    }
    Err(could_not_locate(root))
}

/// Evaluate loaded text: `*file*` bound, the current namespace saved
/// and restored, and (for `load`, like `RT.load`) an isolating `*ns*`
/// frame so the file's `in-ns` never escapes.
fn run_loaded(
    eval: &mut Evaluator,
    file_star: Value,
    spans: &str,
    text: &str,
    isolate_ns: bool,
) -> Result<Vec<Value>, SourceError> {
    in_load_frame(eval, file_star, isolate_ns, |eval| {
        eval.load_source(spans, text)
    })
}

/// `load-file` of a natively compiled program (issue #259): the same
/// load frame, source naming, and error mapping as
/// [`load_text_as_file`], with `body` (the compiled top-level
/// thunks) in place of reading and evaluating text.
pub fn load_native(
    eval: &mut Evaluator,
    path: &str,
    body: impl FnOnce(&mut Evaluator) -> Result<Value, Signal>,
) -> Result<Value, NativeError> {
    let abs = absolutize(path);
    let out = in_load_frame(eval, Value::Str(Arc::from(abs.as_str())), false, |eval| {
        let prev = std::mem::replace(&mut eval.source, Arc::from(abs.as_str()));
        let out = body(eval);
        eval.source = prev;
        out
    });
    out.map_err(|s| source_to_native(eval, &abs, crate::eval::uncaught(s)))
}

/// [`run_loaded`]'s frame around any load body (the lazy library
/// load shares it).
fn in_load_frame<R>(
    eval: &mut Evaluator,
    file_star: Value,
    isolate_ns: bool,
    body: impl FnOnce(&mut Evaluator) -> R,
) -> R {
    let cur = eval.registry.current().to_owned();
    let depth = eval.bindings_depth();
    let source_path = source_path_value(&file_star);
    push_core_binding(eval, "*file*", file_star);
    // S18: every load binds `*source-path*` alongside `*file*`
    // (no existing test can observe the new var).
    push_core_binding(eval, "*source-path*", source_path);
    if isolate_ns {
        // Always `Some`: the depth is captured up front, so the
        // isolation frame never records here.
        let mut keep = Some(depth);
        push_ns_frame(eval, &mut keep);
    }
    // S7: load forces `*read-eval*` true, visibly (probes `l-gate`,
    // `m-*` — `Compiler.load` binds it the same way). The push rides
    // the `depth` pop below (captured up front, so the pop restores
    // exactly this frame's pushes).
    push_core_binding(eval, "*read-eval*", Value::Bool(true));
    let out = body(eval);
    eval.registry.set_current(&cur);
    eval.pop_bindings(depth);
    out
}

/// Embedded libraries that load lazily (see `core_lazy::load_lazy`):
/// every `run` requires `clojure.string`, and its defs are plain.
const LAZY_LIBS: &[&str] = &["clojure/string"];

/// Load an allow-listed embedded library lazily (`None` for any
/// other root): its prefix and non-def forms run in the ordinary
/// load frame, its defs defer.
fn load_lazy_lib(
    eval: &mut Evaluator,
    root: &str,
    file_star: &str,
) -> Option<Result<Vec<Value>, SourceError>> {
    if !LAZY_LIBS.contains(&root) {
        return None;
    }
    let &(_, _, text) = EMBEDDED_SOURCES.iter().find(|(r, _, _)| *r == root)?;
    let file = Value::Str(Arc::from(file_star));
    Some(in_load_frame(eval, file, true, |eval| {
        let prev = eval.swap_source(Arc::from(file_star));
        let lazy = crate::core_lazy::load_lazy(eval, file_star, text);
        eval.swap_source(prev);
        match lazy {
            Ok(true) => Ok(Vec::new()),
            Ok(false) => eval.load_source(file_star, text),
            Err(e) => Err(e),
        }
    }))
}

/// The `*source-path*` value for a load (S18): file loads bind the
/// last path segment (even `load-file` absolutes); `load-string`
/// (`*file*` nil) binds the root.
fn source_path_value(file_star: &Value) -> Value {
    match file_star {
        Value::Str(path) => Value::Str(Arc::from(last_segment(path))),
        _ => Value::Str(Arc::from("NO_SOURCE_FILE")),
    }
}

/// The last path segment (split on `/` or `\`: root loads pin
/// `/` in `*file*`, but `load-file` binds the path as opened, so a
/// native Windows path still yields its basename). Both separators
/// split on every platform — a deliberate narrowing (Unix backslash
/// filenames are out of scope; spec §2.1 D10).
fn last_segment(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Push the isolating `*ns*` frame for a load (recording the
/// depth when it is the first push).
fn push_ns_frame(eval: &mut Evaluator, depth: &mut Option<usize>) {
    if let Some(id) = core_var(eval, "*ns*") {
        let v = deref_core(eval, "*ns*");
        let d = eval.push_binding(core_var_handle(id, "*ns*"), v);
        if depth.is_none() {
            *depth = Some(d);
        }
    }
}

/// Install the hidden natives behind the W6 ported-source
/// namespaces (#217): `clojure.xml` and `clojure.core.server` are
/// source files plus natives, so `load_one` installs first (a
/// `:reload` no-op) and the source load below resolves the calls.
fn install_w6_natives(eval: &mut Evaluator, lib: &Symbol) {
    if sym_str(lib) == crate::w6_runtime::XML_NS {
        crate::w6_runtime::install_xml_natives(eval);
    }
    if sym_str(lib) == crate::w6_runtime::SERVER_NS {
        crate::w6_runtime::install_server_natives(eval);
    }
}

/// `load-one`: load a lib's root resource, check the namespace when
/// `need_ns`, then mark (core.clj `load-one` — the mark runs after
/// the check, probe36 `req-mismatch-as`; callers always require).
pub(crate) fn load_one(
    eval: &mut Evaluator,
    lib: &Symbol,
    need_ns: bool,
) -> Result<(), NativeError> {
    // Gap-123 slice 1: `clojure.core.async` is a native namespace
    // (no source file): requiring it installs the natives and
    // marks it loaded (reinstalling is a `:reload` no-op).
    if sym_str(lib) == crate::async_ns::ASYNC_NS {
        crate::async_ns::install_async_ns(eval);
        mark_loaded(eval, lib)?;
        return Ok(());
    }
    // Gap-124 slice 1: `clojure.tools.deps.util.port` is a native
    // namespace (the file/env substrate, no source file): same
    // install-and-mark shape.
    if sym_str(lib) == crate::portdeps::PORT_NS {
        crate::portdeps::install_port_ns(eval);
        mark_loaded(eval, lib)?;
        return Ok(());
    }
    // Gap-120 slice 3: `clojure.java.io` is a native namespace
    // (the file/stream facsimile, no source file): same
    // install-and-mark shape.
    if sym_str(lib) == crate::jio::JIO_NS {
        crate::jio::install_jio_ns(eval);
        mark_loaded(eval, lib)?;
        return Ok(());
    }
    // #120 slice 3: `clojure.java.shell.impl` is a native
    // namespace (the spawn helper, no source file): same
    // install-and-mark shape.
    if sym_str(lib) == crate::shell::SHELL_NS {
        crate::shell::install_shell_ns(eval);
        mark_loaded(eval, lib)?;
        return Ok(());
    }
    install_w6_natives(eval, lib);
    let root = root_resource(lib.name());
    load_paths(
        eval,
        std::slice::from_ref(&Value::Str(Arc::from(root.as_str()))),
    )?;
    if need_ns && eval.registry.ns_id(&sym_str(lib)).is_none() {
        return Err(ex(format!(
            "namespace '{}' not found after loading '{}'",
            sym_str(lib),
            root
        )));
    }
    mark_loaded(eval, lib)?;
    Ok(())
}

/// `load-all`: `load-one` against a blank `*loaded-libs*` ref,
/// unioning whatever the load pulled in (core.clj `load-all` —
/// the port's embedded set is flat, so this behaves as
/// `:reload`). The union re-marks member by member: marks are
/// monotonic set additions, so one commute per member commits
/// the same set as the JVM's single union commute.
fn load_all(eval: &mut Evaluator, lib: &Symbol, need_ns: bool) -> Result<(), NativeError> {
    let id = core_var(eval, "*loaded-libs*").expect("*loaded-libs* is bound during load");
    let blank = Value::Ref(Ref::new(Value::Set(
        new_set().with_sort(Some(SortSpec::Natural)),
    )));
    let depth = eval.push_binding(core_var_handle(id, "*loaded-libs*"), blank);
    let out = load_one(eval, lib, need_ns);
    let new = libs_set(eval);
    eval.pop_bindings(depth);
    out?;
    for m in new.unwrap_or(new_set()).iter() {
        mark_member(eval, m)?;
    }
    Ok(())
}

/// How `load-lib` loads (or skips) a lib.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LoadKind {
    One,
    All,
    Stub,
}

/// `load-lib`: flags, the load-or-skip, and `:as`/`:refer`
/// application (core.clj `load-lib`, mirrored branch for branch).
/// A libspec's parsed options (core.clj `load-lib` destructuring).
struct LibOpts<'a> {
    as_: Option<&'a Value>,
    use_: bool,
    reload: bool,
    reload_all: bool,
    verbose: bool,
    as_alias: Option<&'a Value>,
    need_ns: bool,
}

fn lib_opts<'a>(pairs: &'a [(&'a Value, &'a Value)]) -> LibOpts<'a> {
    let truthy = |name: &str| opt_get(pairs, name).is_some_and(Value::is_truthy);
    let as_ = opt_get(pairs, "as");
    let use_ = truthy("use");
    LibOpts {
        as_,
        use_,
        reload: truthy("reload"),
        reload_all: truthy("reload-all"),
        verbose: truthy("verbose"),
        as_alias: opt_get(pairs, "as-alias"),
        need_ns: as_.is_some_and(Value::is_truthy) || use_,
    }
}

/// Which load (if any) a libspec selects (core.clj `load-lib` cond).
fn select_load(opts: &LibOpts, loaded: bool) -> Option<LoadKind> {
    if opts.reload_all {
        Some(LoadKind::All)
    } else if opts.reload {
        Some(LoadKind::One)
    } else if !loaded {
        if opts.need_ns {
            Some(LoadKind::One)
        } else if opts.as_alias.is_some_and(Value::is_truthy) {
            Some(LoadKind::Stub)
        } else {
            Some(LoadKind::One)
        }
    } else {
        None
    }
}

/// The `load-lib` load-or-skip phase: the verbose binding, the load
/// with its `Exception` cleanup (errors pass through — a failed load
/// unmakes a namespace it introduced), or the skip's `need-ns` check.
fn load_lib_phase(
    eval: &mut Evaluator,
    joined_sym: &Symbol,
    opts: &LibOpts,
    load: Option<LoadKind>,
    undefined_on_entry: bool,
) -> Result<(), NativeError> {
    // The verbose binding is observable; the echo itself waits for
    // `*out*` printing in S5D.
    let outer_verbose = deref_core(eval, "*loading-verbosely*").is_truthy() || opts.verbose;
    let depth = eval.bindings_depth();
    push_core_binding(eval, "*loading-verbosely*", Value::Bool(outer_verbose));
    let out = match load {
        Some(kind) => {
            let r = match kind {
                LoadKind::One => load_one(eval, joined_sym, opts.need_ns),
                LoadKind::All => load_all(eval, joined_sym, opts.need_ns),
                LoadKind::Stub => {
                    eval.registry.ensure_ns(&sym_str(joined_sym));
                    Ok(())
                }
            };
            match r {
                Err(e) if e.class.matches("java.lang.Exception") => {
                    if undefined_on_entry {
                        eval.registry.remove_ns(&sym_str(joined_sym));
                    }
                    Err(e)
                }
                other => other,
            }
        }
        None => {
            if opts.need_ns && eval.registry.ns_id(&sym_str(joined_sym)).is_none() {
                Err(ex(format!("namespace '{}' not found", sym_str(joined_sym))))
            } else {
                Ok(())
            }
        }
    };
    eval.pop_bindings(depth);
    out
}

/// The `load-lib` finish: `:as`/`:as-alias` aliases, then the
/// `:refer` application for `use` and explicit `:refer`.
fn load_lib_finish(
    eval: &mut Evaluator,
    joined_sym: &Symbol,
    pairs: &[(&Value, &Value)],
    opts: &LibOpts,
) -> Result<(), NativeError> {
    if let Some(a) = opts.as_.filter(|a| a.is_truthy()) {
        native_alias(eval, &[(*a).clone(), Value::Symbol(joined_sym.clone())])?;
    }
    if let Some(a) = opts.as_alias.filter(|a| a.is_truthy()) {
        native_alias(eval, &[(*a).clone(), Value::Symbol(joined_sym.clone())])?;
    }
    if opts.use_ || opt_get(pairs, "refer").is_some() {
        let mut filter = Vec::new();
        for name in ["exclude", "only", "rename", "refer"] {
            if let Some(v) = opt_get(pairs, name) {
                filter.push(Value::Keyword(Keyword::new(None, name)));
                filter.push(v.clone());
            }
        }
        refer_into(eval, &Value::Symbol(joined_sym.clone()), &filter)?;
    }
    Ok(())
}

fn load_lib(
    eval: &mut Evaluator,
    prefix: Option<String>,
    lib: &Value,
    options: &[Value],
) -> Result<(), NativeError> {
    // Eager `(name lib)`: `throw-if` is a function, so the format
    // args evaluate before the predicate (probe36 `req-num-lib`).
    let lib_name = name_of(lib)?;
    if let Some(p) = prefix.as_deref().filter(|_| lib_name.contains('.')) {
        return Err(ex(format!(
            "Found lib name '{lib_name}' containing period with prefix '{p}'.  lib names inside prefix lists must not contain periods"
        )));
    }
    let joined: Value = match &prefix {
        Some(p) => Value::Symbol(Symbol::intern(&format!("{p}.{}", str_of(lib)))),
        None => lib.clone(),
    };
    let pairs = opt_pairs(options)?;
    let opts = lib_opts(&pairs);
    let loaded = libs_set(eval)?.contains(&joined);
    let joined_sym = strict_sym(&joined)?.clone();
    let undefined_on_entry = eval.registry.ns_id(&sym_str(&joined_sym)).is_none();
    let load = select_load(&opts, loaded);
    load_lib_phase(eval, &joined_sym, &opts, load, undefined_on_entry)?;
    load_lib_finish(eval, &joined_sym, &pairs, &opts)
}

/// A libspec: a symbol, or a vector (or entry — `vector?` on
/// the JVM) whose second item is nil or a keyword (core.clj
/// `libspec?` — lists and strings take the prefix path, probe36
/// `req-string`).
fn is_libspec(v: &Value) -> bool {
    match v {
        Value::Symbol(_) => true,
        Value::Vector(x) => matches!(x.get(1), None | Some(Value::Nil) | Some(Value::Keyword(_))),
        Value::MapEntry { val, .. } => {
            matches!(val.as_ref(), Value::Nil | Value::Keyword(_))
        }
        _ => false,
    }
}

/// `prependss`: a symbol conses, anything else concats (seqing it —
/// the ISeq error fires here for scalars, probe36 `req-string`).
fn expand_libspec(v: &Value, flag_opts: &[Value]) -> Result<Vec<Value>, NativeError> {
    let mut out = match v {
        Value::Symbol(_) => vec![v.clone()],
        _ => seq_vec(v).ok_or_else(|| iseq_err(v))?,
    };
    out.extend(flag_opts.iter().cloned());
    Ok(out)
}

/// `load-libs`: flags, libspecs, and prefix lists (core.clj
/// `load-libs`; `require` injects `:require`, `use` `:require :use`).
/// Split `require`/`use` args into top-level flags and libspecs,
/// rejecting unknown flags and empty arg lists (core.clj
/// `load-libs`).
fn split_flags(args: &[Value]) -> Result<(Vec<&Keyword>, Vec<&Value>), NativeError> {
    let mut flags: Vec<&Keyword> = Vec::new();
    let mut rest: Vec<&Value> = Vec::new();
    for a in args {
        if let Value::Keyword(kw) = a {
            flags.push(kw);
        } else {
            rest.push(a);
        }
    }
    const SUPPORTED: &[&str] = &[
        "as",
        "reload",
        "reload-all",
        "require",
        "use",
        "verbose",
        "refer",
        "as-alias",
    ];
    let mut unsupported = Vec::new();
    for kw in &flags {
        let ok = kw.namespace().is_none() && SUPPORTED.contains(&kw.name());
        if !ok {
            unsupported.push(str_of(&Value::Keyword((*kw).clone())));
        }
    }
    if !unsupported.is_empty() {
        return Err(ex(format!(
            "Unsupported option(s) supplied: {}",
            unsupported.join(",")
        )));
    }
    if rest.is_empty() {
        return Err(ex("Nothing specified to load".to_owned()));
    }
    Ok((flags, rest))
}

/// Load one expanded libspec (`(apply load-lib prefix ...)`).
fn load_expanded(
    eval: &mut Evaluator,
    prefix: Option<String>,
    expanded: &[Value],
) -> Result<(), NativeError> {
    // Both callers pass `expand_libspec` output, which always carries
    // the injected flags — never empty.
    let (lib, options) = expanded
        .split_first()
        .expect("load-libs always injects flags");
    load_lib(eval, prefix, lib, options)
}

/// Load one prefix-list arg: the first item is the prefix, the rest
/// are libspecs joined under it.
fn load_prefix_arg(
    eval: &mut Evaluator,
    arg: &Value,
    flag_opts: &[Value],
) -> Result<(), NativeError> {
    let items = seq_vec(arg).ok_or_else(|| iseq_err(arg))?;
    let mut it = items.into_iter();
    let prefix = match it.next() {
        None | Some(Value::Nil) => {
            return Err(ex("prefix cannot be nil".to_owned()));
        }
        Some(p) => str_of(&p),
    };
    for sub in it {
        let expanded = expand_libspec(&sub, flag_opts)?;
        load_expanded(eval, Some(prefix.clone()), &expanded)?;
    }
    Ok(())
}

fn load_libs(
    eval: &mut Evaluator,
    injected: &[&str],
    args: &[Value],
) -> Result<Value, NativeError> {
    let (flags, rest) = split_flags(args)?;
    let mut flag_opts: Vec<Value> = Vec::new();
    for kw in &flags {
        flag_opts.push(Value::Keyword((*kw).clone()));
        flag_opts.push(Value::Bool(true));
    }
    for name in injected {
        flag_opts.push(Value::Keyword(Keyword::new(None, name)));
        flag_opts.push(Value::Bool(true));
    }
    for arg in rest {
        if is_libspec(arg) {
            let expanded = expand_libspec(arg, &flag_opts)?;
            load_expanded(eval, None, &expanded)?;
        } else {
            load_prefix_arg(eval, arg, &flag_opts)?;
        }
    }
    Ok(Value::Nil)
}

fn native_require(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    load_libs(eval, &["require"], args)
}

/// Evaluate an `(ns …)` form: `in-ns`, unconditional `resetMeta`
/// (the attrmap wins over the docstring for `:doc`; no attrmap
/// clears), default-or-clause `refer-clojure`, each
/// `:require`/`:use` clause in order through the loader (failures
/// surface raw), then mark the lib loaded (probe37 `ns-*`).
/// `with-loading-context` needs no analog: the loader already
/// scopes `*loaded-libs*` per load.
pub(crate) fn eval_ns_form(
    eval: &mut Evaluator,
    name: &str,
    doc: Option<&str>,
    attr: Option<&Map>,
    clauses: &[NsClause],
) -> Result<Value, Signal> {
    let name_arg = Value::Symbol(Symbol::new(None, name));
    native_in_ns(eval, &[name_arg]).map_err(|e| eval.throw(e))?;
    let meta = ns_form_meta(doc, attr);
    eval.registry.with_ns_mut(name, |ns| {
        ns.set_meta(meta);
    });
    run_ns_clauses(eval, clauses)?;
    let lib = Symbol::new(None, name);
    mark_loaded(eval, &lib).map_err(|e| eval.throw(e))?;
    Ok(Value::Nil)
}

/// An `(ns …)` form's meta: the attrmap (winning over the
/// docstring for `:doc`), the docstring alone, or nothing (which
/// clears, like the unconditional `resetMeta`).
fn ns_form_meta(doc: Option<&str>, attr: Option<&Map>) -> Option<Map> {
    let doc_key = Value::Keyword(Keyword::new(None, "doc"));
    match attr {
        None => doc.map(|d| new_map().insert(doc_key, Value::Str(Arc::from(d)))),
        Some(a) => {
            let mut m = a.clone();
            if let Some(d) = doc.filter(|_| m.get(&doc_key).is_none()) {
                m = m.insert(doc_key, Value::Str(Arc::from(d)));
            }
            Some(m)
        }
    }
}

/// An `(ns …)` form's clauses: the default `refer-clojure` unless
/// a clause overrides it, then each `:require`/`:use` clause in
/// order through the loader (failures surface raw).
fn run_ns_clauses(eval: &mut Evaluator, clauses: &[NsClause]) -> Result<(), Signal> {
    let has_refer_clojure = clauses
        .iter()
        .any(|c| matches!(c, NsClause::ReferClojure(_)));
    if !has_refer_clojure {
        native_refer_clojure(eval, &[]).map_err(|e| eval.throw(e))?;
    }
    for clause in clauses {
        run_one_ns_clause(eval, clause)?;
    }
    Ok(())
}

/// One `(ns …)` clause through its loader entry (failures surface
/// raw, like the sequential loop it replaces).
fn run_one_ns_clause(eval: &mut Evaluator, clause: &NsClause) -> Result<(), Signal> {
    let out = match clause {
        NsClause::ReferClojure(filters) => native_refer_clojure(eval, filters),
        NsClause::Require(libspecs) => native_require(eval, libspecs),
        NsClause::Use(libspecs) => native_use(eval, libspecs),
        NsClause::Import(specs) => native_import_clause(eval, specs),
    };
    out.map(|_| ()).map_err(|e| eval.throw(e))
}

fn import_spec_error() -> NativeError {
    NativeError::new(
        ThrowClass::ExceptionInfo,
        "Call to clojure.core/import did not conform to spec.".to_owned(),
    )
}

/// One `(package Class*)` import list (either sequential shape —
/// the JVM takes lists and vectors alike): every element is a
/// bare symbol and at least the package plus one class are
/// present.
fn import_package_list(eval: &mut Evaluator, parts: &[&Value]) -> Result<Value, NativeError> {
    use crate::classemu::import_class;
    let ok = parts.len() >= 2
        && parts
            .iter()
            .all(|e| matches!(e, Value::Symbol(s) if s.namespace().is_none()));
    if !ok {
        return Err(import_spec_error());
    }
    let pkg = sym_str(symbol_of(parts[0]));
    let mut last = Value::Nil;
    for class in &parts[1..] {
        let fqn = format!("{pkg}.{}", sym_str(symbol_of(class)));
        last = import_class(eval, &fqn)?;
    }
    Ok(last)
}

fn import_one_spec(eval: &mut Evaluator, spec: &Value) -> Result<Value, NativeError> {
    use crate::classemu::import_class;
    // Vectors are package lists directly (the `(quote x)` strip
    // below is list-only: a vector IS the spec, never a form).
    if let Value::Vector(x) = spec {
        let owned: Vec<Value> = x.iter().collect();
        let parts: Vec<&Value> = owned.iter().collect();
        return import_package_list(eval, &parts);
    }
    let items: Vec<&Value> = match spec {
        Value::List(l) => l.iter().collect(),
        _ => Vec::new(),
    };
    // The `import` macro's `(quote x)` strip, over raw forms.
    let bare: &Value = match items.as_slice() {
        [Value::Symbol(q), inner] if q.namespace().is_none() && q.name() == "quote" => inner,
        _ => spec,
    };
    match bare {
        Value::Symbol(s) if s.namespace().is_none() => import_class(eval, &sym_str(s)),
        Value::List(l) => {
            let parts: Vec<&Value> = l.iter().collect();
            import_package_list(eval, &parts)
        }
        _ => Err(import_spec_error()),
    }
}

/// An `(ns … (:import spec…))` clause (issue #201 S1): the same
/// shapes the `import` macro takes — a bare unqualified symbol or
/// a package list of ≥2 unqualified symbols (`(quote x)` stripped
/// the same way) — fanned out to `import_class`. Malformed specs
/// replay the macro's ex-info text verbatim.
fn native_import_clause(eval: &mut Evaluator, specs: &[Value]) -> Result<Value, NativeError> {
    let mut last = Value::Nil;
    for spec in specs {
        last = import_one_spec(eval, spec)?;
    }
    Ok(last)
}

/// The symbol inside an import spec element (callers validate first).
fn symbol_of(v: &Value) -> &Symbol {
    match v {
        Value::Symbol(s) => s,
        _ => unreachable!("import specs validate before use"),
    }
}

fn native_use(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    load_libs(eval, &["require", "use"], args)
}

/// `load` over paths (shared by the `load` native and `load-one`):
/// root resolution, the cycle guard, and the isolated file eval.
/// Resolve one `load` path: a leading slash is absolute, otherwise
/// relative to the current namespace's root directory (core.clj
/// `load`). Returns the full path and the visible pending stack.
/// The relative-load root namespace: the `*ns*` handle's name,
/// the registry current when `*ns*` is unbound (bare evaluators
/// have no `*ns*` root), else the JVM's `ClassCastException`
/// (oracle REL42: `(binding [*ns* 42] (load "x"))` fails the
/// `Symbol` cast instead of falling back).
fn load_ns_root(eval: &Evaluator, starred: &Value) -> Result<String, NativeError> {
    if let Some(h) = as_ns(starred) {
        return Ok(h.name.to_string());
    }
    if matches!(starred, Value::Nil) {
        return Ok(eval.registry.current().to_owned());
    }
    Err(cce_err(starred, "clojure.lang.Symbol"))
}

fn resolve_load_path(eval: &Evaluator, p: &Value) -> Result<(String, List), NativeError> {
    let path = match p {
        Value::Str(s) => s.to_string(),
        _ => return Err(cast_err(p, "java.lang.String")),
    };
    let full = if path.starts_with('/') {
        path
    } else {
        let starred = deref_core(eval, "*ns*");
        let ns_name = load_ns_root(eval, &starred)?;
        format!("{}/{}", root_directory(&ns_name), path)
    };
    match deref_core(eval, "*pending-paths*") {
        Value::List(l) => Ok((full, l)),
        v => Err(cast_err(&v, "clojure.lang.IPersistentList")),
    }
}

/// The `*pending-paths*` cycle guard: a repeat below the top errors
/// (`check-cyclic-dependency`), the top itself is exempt (self-load),
/// anything else proceeds.
fn cycle_action(pending: &List, full: &str) -> Result<bool, NativeError> {
    let is_full = |x: &Value| matches!(x, Value::Str(s) if s.as_ref() == full);
    if pending.iter().skip(1).any(is_full) {
        let mut chain = vec![full.to_owned()];
        chain.extend(pending.iter().map(str_of));
        let marked: Vec<String> = chain
            .iter()
            .map(|x| {
                if x == full {
                    format!("[ {x} ]")
                } else {
                    x.clone()
                }
            })
            .collect();
        return Err(ex(format!("Cyclic load dependency: {}", marked.join("->"))));
    }
    Ok(!pending.first().is_some_and(is_full))
}

/// Load, bind, and evaluate one resolved path.
fn load_one_path(eval: &mut Evaluator, full: &str, pending: &List) -> Result<(), NativeError> {
    let root = full.strip_prefix('/').unwrap_or(full).to_owned();
    let (file_star, text) = resolve_source(eval, &root)?;
    // S18: the flag-gated plan phase (`None` when `*compile-files*`
    // is falsy — the load below then runs exactly as today).
    let planned = crate::compiler::maybe_plan_root(eval, &root, &file_star, &text)?;
    // `resolve_load_path` just read this var as a list (no user code
    // runs between), so the var exists and still holds a list.
    let id = core_var(eval, "*pending-paths*").expect("*pending-paths* is bound during load");
    let pending_depth = eval.push_binding(
        core_var_handle(id, "*pending-paths*"),
        Value::List(pending.clone().push_front(Value::Str(Arc::from(full)))),
    );
    let out = match load_lazy_lib(eval, &root, &file_star) {
        Some(out) => out,
        None => run_loaded(
            eval,
            Value::Str(Arc::from(file_star.as_str())),
            &file_star,
            &text,
            true,
        ),
    };
    eval.pop_bindings(pending_depth);
    match out {
        Ok(_) => {
            // S18: emit the planned object (skipped when `None` —
            // eval failures never reach here, so T12 leaves nothing).
            if let Some(plan) = planned {
                crate::compiler::emit_planned(&file_star, &plan)?;
            }
            Ok(())
        }
        Err(e) => Err(source_to_native(eval, &file_star, e)),
    }
}

pub(crate) fn load_paths(eval: &mut Evaluator, paths: &[Value]) -> Result<(), NativeError> {
    for p in paths {
        let (full, pending) = resolve_load_path(eval, p)?;
        if cycle_action(&pending, &full)? {
            load_one_path(eval, &full, &pending)?;
        }
    }
    Ok(())
}

fn native_load(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    load_paths(eval, args)?;
    Ok(Value::Nil)
}

/// The `load-file` `*file*`: the path absolutized, `..` preserved
/// (`java.io.File.getAbsolutePath`, probe36 `file-dotdot`).
fn absolutize(path: &str) -> String {
    if std::path::Path::new(path).is_absolute() {
        return path.to_owned();
    }
    // `current_dir` fails only when the cwd was deleted under us;
    // the empty path then joins back to the relative path itself.
    std::env::current_dir()
        .unwrap_or_default()
        .join(path)
        .to_string_lossy()
        .into_owned()
}

fn native_load_file(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/load-file", args, &[1])?;
    let path = match &args[0] {
        Value::Str(s) => s.to_string(),
        v => return Err(cast_err(v, "java.lang.String")),
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return Err(fs_err(&path, &e)),
    };
    load_text_as_file(eval, &path, &text)
}

/// `load-file` over text already in hand, as if read from `path`
/// (issue #256: AOT-built binaries load their embedded source
/// through this). `Compiler.loadFile`: no `*ns*` isolation, no
/// cycle guard — only `*file*`. Returns the last value (nil when
/// empty).
pub fn load_text_as_file(
    eval: &mut Evaluator,
    path: &str,
    text: &str,
) -> Result<Value, NativeError> {
    let abs = absolutize(path);
    match run_loaded(eval, Value::Str(Arc::from(abs.as_str())), &abs, text, false) {
        Ok(vals) => Ok(vals.last().cloned().unwrap_or(Value::Nil)),
        Err(e) => Err(source_to_native(eval, &abs, e)),
    }
}

fn native_load_string(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/load-string", args, &[1])?;
    let text = match &args[0] {
        Value::Str(s) => s.to_string(),
        v => return Err(cast_err(v, "java.lang.String")),
    };
    // `*file*` is nil and spans blame the empty source (probe36
    // `file-loadstring`, `loadstring-errspan`). Returns the last
    // value (nil when empty).
    match run_loaded(eval, Value::Nil, "", &text, false) {
        Ok(vals) => Ok(vals.last().cloned().unwrap_or(Value::Nil)),
        Err(e) => Err(source_to_native(eval, "", e)),
    }
}

/// One `load-reader` read step: snapshot the data-reader table,
/// read a form, drain the fired tags. Returns the read outcome
/// plus the post-read `*suppress-read*` gate. (Split out so
/// `native_load_reader` stays at the cyclo-15 ratchet — the two
/// consult branches live here instead of in the loop.)
fn load_reader_step(
    eval: &mut Evaluator,
    opts: &mut clj_reader::ReaderOpts,
    parser: &mut clj_reader::Reader,
) -> Result<(Result<Option<Value>, clj_reader::ReaderError>, bool), NativeError> {
    crate::core::snapshot_data_readers(eval, opts)?;
    parser.set_opts(opts);
    let outcome = {
        let mut hooks = crate::reader::LiveReadHooks::new(&mut *eval);
        parser.read_form(false, &mut hooks)
    };
    let fired = parser.take_fired();
    let suppressed = opts.suppress_read;
    crate::core::drain_fired_tags(eval, fired, suppressed)?;
    Ok((outcome, suppressed))
}

/// `load-reader` entry checks: arity, then nil (NPE) vs
/// non-reader (CCE) dispatch. (Split out so
/// `native_load_reader` stays at the cyclo-15 ratchet.)
fn load_reader_arg(args: &[Value]) -> Result<&crate::reader::StreamReader, NativeError> {
    check_arity("clojure.core/load-reader", args, &[1])?;
    match &args[0] {
        Value::Nil => Err(NativeError::new(
            ThrowClass::NullPointerException,
            String::new(),
        )),
        v => match crate::reader::as_reader(v) {
            Some(r) => Ok(r),
            None => Err(cce_err(v, "java.io.Reader")),
        },
    }
}

/// `load-reader`: read and evaluate every form from a reader,
/// returning the last value (nil when empty — S7; probes `l-*`).
/// `*file*` is nil, `*ns*` is isolated, `*read-eval*` is forced true
/// (visibly), and `*pending-paths*` is untouched.
fn native_load_reader(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let reader = load_reader_arg(args)?;
    crate::reader::check_reader_open(reader)?;
    // `Compiler.load` consumes leading whitespace before its entry
    // checks, so entry failures blame the post-skip position.
    let mut opts = clj_reader::ReaderOpts {
        read_cond: clj_reader::ReadCond::Allow,
        ..clj_reader::ReaderOpts::default()
    };
    let (mut parser, _) = crate::reader::take_parser(reader, &opts);
    parser.consume_whitespaces();
    let (line, col) = (parser.line(), parser.col());
    crate::reader::restore_parser(reader, parser);
    let seed = match crate::reader::seed_load_gates(eval) {
        Ok(seed) => seed,
        Err(cce) => {
            let wrap =
                crate::reader::executing_wrap(line, col, &cce, eval.next_id(), eval.next_id());
            return Err(source_to_native(eval, "", wrap));
        }
    };
    opts.read_eval = seed.read_eval;
    opts.suppress_read = seed.suppress_read;
    opts.resolver_bound = seed.resolver_bound;
    let frame = push_load_frame(eval);
    let out = load_reader_forms(eval, reader, &mut opts);
    pop_load_frame(eval, frame);
    out.map_err(|e| source_to_native(eval, "", e))
}

/// Read and evaluate one form at a time, like `Compiler.load`: each
/// form reads after the previous one evaluates (a later read error
/// leaves the earlier defs standing — oracle `l-err-read-defined`).
fn load_reader_forms(
    eval: &mut Evaluator,
    reader: &crate::reader::StreamReader,
    opts: &mut clj_reader::ReaderOpts,
) -> Result<Value, SourceError> {
    let mut out = Value::Nil;
    while let Some(form) = load_reader_next(eval, reader, opts)? {
        out = eval.eval_top_form(&form)?;
    }
    Ok(out)
}

/// One `load-reader` read: the namespace view re-synced, the parser
/// out of its lock for the read (the hook runs user code, which may
/// read the same reader again), the per-form S6 consult (B1 —
/// snapshot, read, drain, resolve). Data-reader failures surface
/// like `load`'s (raw uncaught throwables, not `CompilerException`
/// wraps).
fn load_reader_next(
    eval: &mut Evaluator,
    reader: &crate::reader::StreamReader,
    opts: &mut clj_reader::ReaderOpts,
) -> Result<Option<Value>, SourceError> {
    crate::reader::sync_reader_ns(eval, opts);
    let (mut parser, _) = crate::reader::take_parser(reader, opts);
    let read = match load_reader_step(eval, opts, &mut parser) {
        Ok((Ok(Some(form)), suppressed)) => {
            // `Compiler.load` skips the whitespace after a form before
            // evaluating it (a `read-line` there starts past it).
            parser.consume_whitespaces();
            crate::core::resolve_data_readers(eval, form, suppressed).map(Some)
        }
        Ok((Ok(None), _)) => Ok(None),
        Ok((Err(e), _)) => {
            crate::reader::restore_parser(reader, parser);
            return Err(SourceError::Reader(e));
        }
        Err(e) => Err(e),
    };
    crate::reader::restore_parser(reader, parser);
    read.map_err(|e| SourceError::Uncaught(eval.throwable(e)))
}

/// The `load-reader` evaluation frame (`*file*` nil, `*ns*`
/// isolated, `*read-eval*` forced true): the namespace and binding
/// depth [`pop_load_frame`] restores.
fn push_load_frame(eval: &mut Evaluator) -> (String, usize) {
    let cur = eval.registry.current().to_owned();
    let depth = eval.bindings_depth();
    if let Some(id) = core_var(eval, "*file*") {
        eval.push_binding(core_var_handle(id, "*file*"), Value::Nil);
    }
    // S18: `load-reader` binds `*source-path*` to the root (its
    // `*file*` is nil — the `load-string` shape, never "leave",
    // §2.2).
    if let Some(id) = core_var(eval, "*source-path*") {
        eval.push_binding(
            core_var_handle(id, "*source-path*"),
            source_path_value(&Value::Nil),
        );
    }
    // Always `Some`: the depth is captured up front (the
    // `run_loaded` shape), so the isolation frame never records
    // here.
    let mut keep = Some(depth);
    push_ns_frame(eval, &mut keep);
    if let Some(id) = core_var(eval, "*read-eval*") {
        eval.push_binding(core_var_handle(id, "*read-eval*"), Value::Bool(true));
    }
    (cur, depth)
}

fn pop_load_frame(eval: &mut Evaluator, (cur, depth): (String, usize)) {
    eval.registry.set_current(&cur);
    eval.pop_bindings(depth);
}

fn native_loaded_libs(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/loaded-libs", args, &[0])?;
    Ok(Value::Set(libs_set(eval)?))
}

fn native_eval(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/eval", args, &[1])?;
    let source = eval.source().to_owned();
    match eval.eval_top_form(&args[0]) {
        Ok(v) => Ok(v),
        Err(e) => Err(source_to_native(eval, &source, e)),
    }
}

/// A native fn pointer (all §N natives share the signature).
pub(crate) type NativeFnPtr = fn(&mut Evaluator, &[Value]) -> Result<Value, NativeError>;

/// An owned-args native fn pointer (head-releasing loopers).
pub(crate) type OwnedNativeFnPtr =
    fn(&mut Evaluator, smallvec::SmallVec<[Value; 4]>) -> Result<Value, NativeError>;

/// Owned-args entries for `BootMode::Lite` core/collection natives (sorted for
/// binary search; kept separate from sequence/transducer entries so non-seq
/// AOT binaries dead-strip `seqs`/`xform`).
pub(crate) const OWNED_CORE_ENTRIES: &[(&str, OwnedNativeFnPtr)] = &[
    ("assoc", crate::coll::native_assoc_owned),
    ("assoc-in", crate::coll::native_assoc_in_owned),
    ("conj", crate::coll::native_conj_owned),
    ("count", crate::coll::native_count_owned),
    ("empty?", crate::coll::native_emptyp_owned),
    ("into", crate::coll::native_into_owned),
    ("last", crate::coll::native_last_owned),
    ("nth", crate::coll::native_nth_owned),
    ("persistent!", crate::coll::native_persistent_owned),
    ("set", crate::coll::native_set_owned),
    ("str", crate::core::native_str_owned),
    ("transient", crate::coll::native_transient_owned),
    ("update", crate::coll::native_update_owned),
    ("update-in", crate::coll::native_update_in_owned),
    ("vec", crate::coll::native_vec_owned),
];

/// Owned-args entries for sequence and transducer natives (sorted for binary search).
pub(crate) const OWNED_SEQ_ENTRIES: &[(&str, OwnedNativeFnPtr)] = &[
    ("butlast", crate::seqs::native_butlast_owned),
    ("doall", crate::seqs::native_doall_owned),
    ("dorun", crate::seqs::native_dorun_owned),
    ("drop-last", crate::seqs::native_drop_last_owned),
    ("every?", crate::seqs::native_every_p_owned),
    ("filterv", crate::seqs::native_filterv_owned),
    ("frequencies", crate::seqs::native_frequencies_owned),
    ("group-by", crate::seqs::native_group_by_owned),
    ("mapv", crate::seqs::native_mapv_owned),
    ("not-any?", crate::seqs::native_not_any_p_owned),
    ("not-every?", crate::seqs::native_not_every_p_owned),
    ("reduce", crate::seqs::native_reduce_owned),
    ("reduce-kv", crate::seqs::native_reduce_kv_owned),
    ("reverse", crate::seqs::native_reverse_owned),
    ("run!", crate::seqs::native_run_bang_owned),
    ("some", crate::seqs::native_some_owned),
    ("sort", crate::seqs::native_sort_owned),
    ("sort-by", crate::seqs::native_sort_by_owned),
    ("take-last", crate::seqs::native_take_last_owned),
    ("transduce", crate::xform::native_transduce_owned),
];

/// Lookup `name` inside a sorted owned-args table (`None` for borrow-only natives).
pub(crate) fn owned_entry_in(
    entries: &[(&str, OwnedNativeFnPtr)],
    name: &str,
) -> Option<std::sync::Arc<crate::ext::OwnedNativeImpl>> {
    let idx = entries.binary_search_by(|(n, _)| n.cmp(&name)).ok()?;
    Some(std::sync::Arc::new(entries[idx].1))
}

/// The owned-args entry for a looping native by name (`None`
/// for borrow-only natives — test doubles install literally with `owned: None`).
pub(crate) fn owned_entry(name: &str) -> Option<std::sync::Arc<crate::ext::OwnedNativeImpl>> {
    owned_entry_in(OWNED_CORE_ENTRIES, name).or_else(|| owned_entry_in(OWNED_SEQ_ENTRIES, name))
}

/// The §N natives installed into `clojure.core` by [`boot`].
const NATIVES: &[(&str, NativeFnPtr)] = &[
    ("create-ns", native_create_ns),
    ("in-ns", native_in_ns),
    ("ns-name", native_ns_name),
    ("the-ns", native_the_ns),
    ("find-ns", native_find_ns),
    ("remove-ns", native_remove_ns),
    ("all-ns", native_all_ns),
    ("ns-aliases", native_ns_aliases),
    ("ns-map", native_ns_map),
    ("ns-interns", native_ns_interns),
    ("ns-publics", native_ns_publics),
    ("ns-refers", native_ns_refers),
    ("ns-imports", native_ns_imports),
    ("alias", native_alias),
    ("refer", native_refer),
    ("refer-clojure", native_refer_clojure),
    ("intern", native_intern),
    ("ns-resolve", native_ns_resolve),
    ("resolve", native_resolve),
    ("find-var", native_find_var),
    ("requiring-resolve", native_requiring_resolve),
    ("ns-unmap", native_ns_unmap),
    ("ns-unalias", native_ns_unalias),
    ("require", native_require),
    ("use", native_use),
    ("load", native_load),
    ("load-string", native_load_string),
    ("load-file", native_load_file),
    ("load-reader", native_load_reader),
    ("loaded-libs", native_loaded_libs),
    ("eval", native_eval),
    ("add-classpath", crate::compiler::native_add_classpath),
    ("compile", crate::compiler::native_compile),
];

/// The `clojure.core` vars installed by [`boot`]: name, root,
/// dynamic, private.
fn core_vars(user_ns: Value) -> Vec<(String, Value, bool, bool)> {
    vec![
        ("*ns*".to_owned(), user_ns, true, false),
        (
            "*file*".to_owned(),
            Value::Str(Arc::from("NO_SOURCE_PATH")),
            true,
            false,
        ),
        (
            "*loaded-libs*".to_owned(),
            Value::Ref(Ref::new(Value::Set(
                new_set().with_sort(Some(SortSpec::Natural)),
            ))),
            true,
            true,
        ),
        (
            "*pending-paths*".to_owned(),
            Value::List(List::from_vec(Vec::new())),
            true,
            true,
        ),
        (
            "*loading-verbosely*".to_owned(),
            Value::Bool(false),
            true,
            true,
        ),
        (
            "*load-path*".to_owned(),
            Value::Vector(Vector::from_vec(Vec::new())),
            true,
            false,
        ),
        // `core.clj` `^:dynamic` vars: slot AND `:dynamic` meta
        // (corpus 11 `:print-vars-dynamic` pins all three `true`).
        ("*print-length*".to_owned(), Value::Nil, true, false),
        ("*print-level*".to_owned(), Value::Nil, true, false),
        (
            "*print-namespace-maps*".to_owned(),
            Value::Bool(true),
            true,
            false,
        ),
        // S7 REPL dynamics (P1/P2): nil roots; the session binds and
        // shifts them (`^:dynamic`, like core.clj).
        ("*1".to_owned(), Value::Nil, true, false),
        ("*2".to_owned(), Value::Nil, true, false),
        ("*3".to_owned(), Value::Nil, true, false),
        ("*e".to_owned(), Value::Nil, true, false),
        // S20 oddballs (#55, probes `r-*`): false root; sessions
        // bind it true (`^:dynamic`, like the window above).
        ("*repl*".to_owned(), Value::Bool(false), true, false),
        // S7 CLI (P4): nil root; `run` binds the arg seq.
        ("*command-line-args*".to_owned(), Value::Nil, true, false),
        // S6 data readers: the tag table (empty root) and the
        // default-fn fallback (nil root); both `^:dynamic` with
        // `:dynamic` meta (probes `a-roots`, `a-meta`).
        (
            "*data-readers*".to_owned(),
            Value::Map(new_map()),
            true,
            false,
        ),
        (
            "*default-data-reader-fn*".to_owned(),
            Value::Nil,
            true,
            false,
        ),
        // S18 compiler knob (`def ^:dynamic` shape: slot AND
        // `:dynamic` meta, like the print vars above).
        (
            "*verbose-defrecords*".to_owned(),
            Value::Bool(false),
            true,
            false,
        ),
        // S7 environment (P3): a plain (non-dynamic) root map over
        // the workspace release version.
        (
            "*clojure-version*".to_owned(),
            clojure_version_map(),
            false,
            false,
        ),
    ]
}

/// The `*clojure-version*` root (P3): same keys as the JVM map
/// for the tracked release (T1197 REL probe `rel-version`:
/// `{:major 1, :minor 13, :incremental 0, :qualifier "alpha11"}`).
fn clojure_version_map() -> Value {
    let mut map = new_map();
    for (key, value) in [
        ("major", Value::Int(1)),
        ("minor", Value::Int(13)),
        ("incremental", Value::Int(0)),
        ("qualifier", Value::Str(Arc::from("alpha11"))),
    ] {
        map = map.insert(Value::Keyword(Keyword::new(None, key)), value);
    }
    Value::Map(map)
}

/// Install one `clojure.core` var with the dynamic SLOT set but no
/// `:dynamic` meta — the JVM `RT.java` interns (`Var.intern(...).
/// setDynamic()`): `*assert*`, the `*unchecked-math*` /
/// `*warn-on-reflection*` / `*allow-unresolved-vars*` flags, `*out*` /
/// `*err*`, and the `*print-meta*` / `*print-readably*` /
/// `*print-dup*` / `*flush-on-newline*` print vars (corpus 11/15
/// pin the meta `false` while `binding` still works).
fn def_dynamic_root(eval: &mut Evaluator, name: &str, root: Value) {
    let q = Symbol::new(Some("clojure.core"), name);
    let id = eval.registry.intern(&q);
    eval.registry.with_state_mut(id, |st| {
        st.root = Some(root);
        st.dynamic = true;
    });
}

/// Install one `clojure.core` var (private vars carry
/// `^:private` meta, core.clj `defonce`).
fn def_var(eval: &mut Evaluator, name: &str, root: Value, dynamic: bool, private: bool) {
    let q = Symbol::new(Some("clojure.core"), name);
    let id = eval.registry.intern(&q);
    eval.registry.with_state_mut(id, |st| {
        st.root = Some(root);
        st.dynamic = dynamic;
        // `intern` stamped `:ns`/`:name`; boot adds `:dynamic`
        // (JVM `def^:dynamic` keeps it in the meta) and `:private`.
        let mut meta = st.meta.clone().unwrap_or_else(new_map);
        if dynamic {
            meta = meta.insert(
                Value::Keyword(Keyword::new(None, "dynamic")),
                Value::Bool(true),
            );
        }
        if private {
            meta = meta.insert(
                Value::Keyword(Keyword::new(None, "private")),
                Value::Bool(true),
            );
        }
        st.meta = Some(meta);
    });
}

/// Boot a fresh evaluator: `clojure.core` plus `user` (current),
/// the §N natives and vars installed, every public install referred
/// into `user` (JVM boot refers `clojure.core`), a permanent `*ns*`
/// base frame (so `in-ns` writes the top frame and the root never
/// moves, like `RT.inNamespace` under `clojure.main`), and the
/// embedded `core.clj` composites loaded into `clojure.core` with
/// their publics referred into `user`.
fn vm_call_hook(
    eval: &mut Evaluator,
    callee: &Value,
    args: &[Value],
) -> Option<Result<Value, Signal>> {
    let (code, arity) = crate::vm::fixed_target(callee, args.len())?;
    Some(crate::vm::call_borrowed(eval, code, arity, args))
}

fn vm_apply_moved_hook(
    eval: &mut Evaluator,
    callee: &Value,
    args: &mut smallvec::SmallVec<[Value; 4]>,
) -> Option<Result<Value, Signal>> {
    let closure = crate::ext::as_closure(callee)?;
    let (Some(code), None) = (&closure.vm, &closure.conveyed) else {
        return None;
    };
    Some(crate::vm::apply_moved(eval, closure, code, args))
}

pub(crate) static FULL_LAZY_HOOKS: crate::eval::LazyHooks = crate::eval::LazyHooks {
    force: crate::core_lazy::force,
    force_meta: crate::core_lazy::force_meta,
    interpreted: crate::core_lazy::interpreted,
    eval_ns: eval_ns_form,
    apply_closure: Evaluator::apply_closure,
    vm_call: vm_call_hook,
    vm_apply_moved: vm_apply_moved_hook,
    vm_call1: crate::vm::vm_call1,
    vm_call2: crate::vm::vm_call2,
    eval_expr: Evaluator::eval,
};

#[must_use]
pub fn boot() -> Evaluator {
    let mut eval = Evaluator::new_bare("user");
    eval.lazy_hooks = Some(&FULL_LAZY_HOOKS);
    eval.registry.ensure_ns("clojure.core");
    let user_ns = eval.registry.ensure_ns("user");
    eval.registry.set_current("user");
    boot_install_natives(&mut eval);
    boot_install_builtin_macros(&mut eval);
    boot_mark_private(&mut eval);
    let user_handle = ns_value("user", user_ns);
    boot_core_vars(&mut eval, user_handle.clone());
    crate::printsys::boot_print_system(&mut eval);
    crate::reader::boot_read_arglists(&mut eval);
    push_core_binding(&mut eval, "*ns*", user_handle);
    // Embedded `core.clj` composites (their privates resolve in
    // `clojure.core`), then `user` refers the publics — the file
    // carries no `ns` form by design. Its defs register lazily: the
    // first resolution of each var evaluates its form (#226). The
    // namespace switches and the refer run directly (evaluating
    // `(in-ns …)`/`(refer …)` needed `user` to refer them first).
    boot_in_ns(&mut eval, "clojure.core");
    crate::core_lazy::boot_lazy(&mut eval, CORE_CLJ);
    boot_in_ns(&mut eval, "user");
    let core = Value::Symbol(Symbol::new(None, "clojure.core"));
    refer_into(&mut eval, &core, &[]).expect("boot refers clojure.core into user");
    // The load's macroexpansions consume `gensym` numbers; the
    // user-visible counter restarts from 1 (S5C pins `G__1` first).
    eval.reset_gensym();
    eval
}

/// `in-ns` at boot (see [`native_in_ns`]), without evaluating a form.
fn boot_in_ns(eval: &mut Evaluator, name: &str) {
    let id = eval.registry.ensure_ns(name);
    eval.registry.set_current(name);
    set_core_ns(eval, ns_value(name, id));
}

const IO_EXTRA_NATIVES: &[(&str, NativeFnPtr)] = &[("read-string", crate::io::native_read_string)];

/// Install the §N natives into `clojure.core`.
fn boot_install_natives(eval: &mut Evaluator) {
    let natives = NATIVES
        .iter()
        .chain(crate::vars::VAR_NATIVES.iter())
        .chain(crate::macros::MACRO_LITE_NATIVES.iter())
        .chain(crate::macros::MACRO_NATIVES.iter())
        .chain(crate::lazy::LAZY_NATIVES.iter())
        .chain(crate::core::CORE_NATIVES.iter())
        .chain(crate::xform::XFORM_NATIVES.iter())
        .chain(crate::seqs::SEQS_NATIVES.iter())
        .chain(crate::coll::COLL_NATIVES.iter())
        .chain(crate::re::RE_NATIVES.iter())
        .chain(crate::rng::RNG_NATIVES.iter())
        .chain(crate::io::IO_LITE_NATIVES.iter())
        .chain(IO_EXTRA_NATIVES.iter())
        .chain(crate::jdkio::JDKIO_NATIVES.iter())
        .chain(crate::reader::READER_NATIVES.iter())
        .chain(crate::concurrency::CONC_NATIVES.iter())
        .chain(crate::stm::STM_NATIVES.iter())
        .chain(crate::fx4_host::FX4_NATIVES.iter())
        .chain(crate::types::TYPES_NATIVES.iter())
        .chain(crate::typeinternals::TYPEINTERNALS_NATIVES.iter())
        .chain(crate::protocol::PROTOCOL_NATIVES.iter())
        .chain(crate::multi::MULTI_NATIVES.iter())
        .chain(crate::hierarchy::HIERARCHY_NATIVES.iter())
        .chain(crate::core::TOWER_NATIVES.iter())
        .chain(crate::collmisc::COLLMISC_NATIVES.iter())
        .chain(crate::printsys::PRINTSYS_NATIVES.iter())
        .chain(crate::scalars::SCALAR_NATIVES.iter())
        .chain(crate::arrays::ARRAY_NATIVES.iter())
        .chain(crate::classemu::CLASS_EMU_NATIVES.iter())
        .chain(crate::classemu::CLASS_EMU_HIDDEN.iter())
        .chain(crate::reflect::REFLECT_HIDDEN.iter())
        .collect::<Vec<_>>();
    eval.registry.reserve_vars("clojure.core", natives.len());
    for (name, f) in natives {
        eval.install_native_ptr("clojure.core", name, *f);
    }
}

/// Builtin-macro vars. The FX4 nine run real expanders; the
/// fence seven stay placeholders that fail on direct invocation.
fn boot_install_builtin_macros(eval: &mut Evaluator) {
    for name in crate::macros::BUILTIN_MACROS {
        eval.install_native("clojure.core", name, builtin_macro_root(name));
        let q = Symbol::new(Some("clojure.core"), name);
        let id = eval.registry.intern(&q);
        eval.registry.with_state_mut(id, |st| {
            let meta = st.meta.clone().unwrap_or_else(new_map).insert(
                Value::Keyword(Keyword::new(None, "macro")),
                Value::Bool(true),
            );
            st.meta = Some(meta);
        });
    }
}

/// The nine FX4 heads are real macro fns. The fence seven stay
/// placeholders (direct invocation fails loudly).
fn builtin_macro_root(name: &str) -> Arc<crate::ext::NativeImpl> {
    if let Some(f) = crate::builtin_expand::fx4_macro_fn(name) {
        return Arc::new(f);
    }
    let display = format!("clojure.core/{name}");
    Arc::new(move |_, _| {
        Err(NativeError::new(
            ThrowClass::IllegalStateException,
            format!("Can't invoke builtin macro: {display}"),
        ))
    })
}

/// Port-internal natives (macro-expansion errors, `case` dispatch
/// hashing, FX4 host helpers): usable but never referred into
/// `user` (the refers filter on `is_public`).
fn boot_mark_private(eval: &mut Evaluator) {
    for name in ["throw-illegal-arg!", "throw-npe!", "hash-code"]
        .into_iter()
        .chain(crate::fx4_host::FX4_PRIVATE.iter().copied())
        .chain(crate::xform::XFORM_PRIVATE.iter().copied())
    {
        let q = Symbol::new(Some("clojure.core"), name);
        let id = eval.registry.intern(&q);
        eval.registry.with_state_mut(id, |st| {
            let meta = st.meta.clone().unwrap_or_else(new_map).insert(
                Value::Keyword(Keyword::new(None, "private")),
                Value::Bool(true),
            );
            st.meta = Some(meta);
        });
    }
}

/// Install the `clojure.core` vars and the `RT.java` dynamics (slot
/// set, no `:dynamic` meta).
fn boot_core_vars(eval: &mut Evaluator, user_handle: Value) {
    for (name, root, dynamic, private) in core_vars(user_handle) {
        def_var(eval, &name, root, dynamic, private);
    }
    let stdout = crate::io::stdout_writer(eval);
    let stderr = crate::io::stderr_writer(eval);
    let stdin = crate::reader::stdin_reader(eval);
    def_dynamic_root(eval, "*out*", stdout);
    def_dynamic_root(eval, "*err*", stderr);
    def_dynamic_root(eval, "*in*", stdin);
    def_dynamic_root(eval, "*assert*", Value::Bool(true));
    def_dynamic_root(eval, "*unchecked-math*", Value::Bool(false));
    def_dynamic_root(eval, "*warn-on-reflection*", Value::Bool(false));
    def_dynamic_root(eval, "*allow-unresolved-vars*", Value::Bool(false));
    def_dynamic_root(eval, "*print-meta*", Value::Bool(false));
    def_dynamic_root(eval, "*print-readably*", Value::Bool(true));
    def_dynamic_root(eval, "*print-dup*", Value::Bool(false));
    // S7 read gates (issue #50; probes `a-*`): the roots the JVM
    // interns them with (no `:dynamic` meta, like the others here).
    def_dynamic_root(eval, "*read-eval*", Value::Bool(true));
    def_dynamic_root(eval, "*suppress-read*", Value::Nil);
    def_dynamic_root(eval, "*reader-resolver*", Value::Nil);
    def_dynamic_root(eval, "*flush-on-newline*", Value::Bool(true));
    // `RT.java`: `*agent*` is a dynamic slot with a nil root (bound
    // to the agent inside each action by `send`).
    def_dynamic_root(eval, "*agent*", Value::Nil);
    // `RT.java`: `*math-context*` is a dynamic slot with a nil
    // root, bound by `with-precision` (S2; no `:dynamic` meta —
    // oracle `mc-meta`).
    def_dynamic_root(eval, "*math-context*", Value::Nil);
    // S6: the default tag table (a plain map var — neither dynamic
    // nor private, probe `a-ddr-meta`).
    let defaults = crate::core::default_data_readers_map(eval);
    def_var(eval, "default-data-readers", defaults, false, false);
    // S18 compiler knobs (slot-only roots owned by `compiler.rs`)
    // plus the plain `primitives-classnames` root map.
    for (name, root) in crate::compiler::knob_roots() {
        def_dynamic_root(eval, name, root);
    }
    def_var(
        eval,
        "primitives-classnames",
        crate::compiler::prims_map(),
        false,
        false,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unmapped class names fall back to `Exception` (user code
    /// only produces mapped classes — the pin keeps the totality).
    #[test]
    fn throw_class_for_falls_back_to_exception() {
        assert_eq!(
            throw_class_for("NoSuchClass"),
            clj_value::ThrowClass::Exception
        );
    }

    #[test]
    fn signal_recur_to_native_is_internal() {
        // Defense in depth: an escaping `recur` (impossible — the
        // analyzer cages every one) surfaces as an internal error.
        let err = signal_to_native(Signal::Recur(vec![]));
        assert!(
            err.into_inner()
                .message
                .contains("recur escaped its loop or fn")
        );
    }

    #[test]
    #[should_panic(expected = "import specs validate before use")]
    fn symbol_of_non_symbol_panics_loudly() {
        // Defensive arm: import specs validate before use, so a
        // non-symbol only arrives here — loud, never silent.
        symbol_of(&Value::Int(1));
    }

    #[test]
    fn source_to_native_internal_is_loud() {
        // Direct total-function pin: the `Internal` arm renders loud
        // (only an analyzer bug could produce it from source).
        let mut eval = Evaluator::new_bare("user");
        let err = source_to_native(&mut eval, "f", SourceError::Internal("boom".to_owned()));
        assert_eq!(err.message, "internal error: boom");
    }

    #[test]
    fn push_ns_frame_records_first_depth() {
        let mut eval = Evaluator::new_bare("user");
        let q = Symbol::new(Some("clojure.core"), "*ns*");
        eval.registry.intern(&q);
        let mut depth = None;
        push_ns_frame(&mut eval, &mut depth);
        assert!(depth.is_some());
    }

    #[test]
    fn push_ns_frame_without_var_is_idle() {
        let mut eval = Evaluator::new_bare("user");
        let mut depth = None;
        push_ns_frame(&mut eval, &mut depth);
        assert!(depth.is_none());
    }

    #[test]
    fn deref_core_missing_var_is_nil() {
        let eval = Evaluator::new_bare("user");
        assert_eq!(deref_core(&eval, "*ns*"), Value::Nil);
    }

    #[test]
    fn throw_class_for_maps_unthrown_format_classes() {
        // Direct total-table pin: no producer throws these two
        // (the fmt paths that would are unimplemented), so only a
        // direct call exercises their arms.
        assert_eq!(
            throw_class_for("MissingFormatConversionException"),
            ThrowClass::MissingFormatConversionException
        );
        assert_eq!(
            throw_class_for("IllegalFormatCodePointException"),
            ThrowClass::IllegalFormatCodePointException
        );
    }

    #[test]
    fn throw_class_for_maps_reader_exception_spellings() {
        // Both the FQN and the simple name resolve (the
        // `catch`-name spellings, issue #36).
        assert_eq!(
            throw_class_for("clojure.lang.LispReader$ReaderException"),
            ThrowClass::ReaderException
        );
        assert_eq!(
            throw_class_for("ReaderException"),
            ThrowClass::ReaderException
        );
    }

    #[test]
    fn resolve_load_path_falls_back_when_ns_unbound() {
        // A bare evaluator has no `*ns*` root: the relative root
        // falls back to the registry current. `*pending-paths*`
        // is missing too, so the resolve fails after the fallback
        // — the fallback is the pin.
        let eval = Evaluator::new_bare("user");
        assert!(resolve_load_path(&eval, &Value::Str(Arc::from("foo"))).is_err());
    }

    #[test]
    fn last_segment_splits_either_separator() {
        // S18 `*source-path*`: root loads pin `/` in `*file*`, but
        // `load-file` binds the path as opened.
        assert_eq!(last_segment("my/ctx.clj"), "ctx.clj");
        assert_eq!(last_segment("ctx.clj"), "ctx.clj");
        assert_eq!(last_segment("C:\\my\\ctx.clj"), "ctx.clj");
        assert_eq!(last_segment("a/b\\c.clj"), "c.clj");
        assert_eq!(last_segment("a/"), "");
        assert_eq!(last_segment(""), "");
    }

    #[test]
    fn load_frame_without_vars_is_idle() {
        // S18 `load-reader`: a bare evaluator binds nothing (the
        // missing-var arms are the pin — every `if let` here must
        // tolerate absence).
        let mut eval = Evaluator::new_bare("user");
        let depth = eval.bindings_depth();
        let cur = eval.registry.current().to_owned();
        let frame = push_load_frame(&mut eval);
        assert_eq!(eval.bindings_depth(), depth);
        pop_load_frame(&mut eval, frame);
        assert_eq!(eval.registry.current(), cur);
    }

    #[test]
    fn set_core_ns_on_bare_eval_is_idle() {
        let mut eval = Evaluator::new_bare("user");
        set_core_ns(&mut eval, Value::Nil);
        assert_eq!(deref_core(&eval, "*ns*"), Value::Nil);
    }

    #[test]
    fn intern_routed_errors_and_bare_intern_skips_meta() {
        let mut eval = boot();
        let q = Symbol::new(Some("missing.ns"), "x");
        assert!(intern_routed(&mut eval, "missing.ns", None, &q).is_err());
        assert!(intern_routed(&mut eval, "user", Some(999_999), &q).is_err());

        let args = [
            Value::Symbol(Symbol::new(None, "user")),
            Value::Symbol(Symbol::new(None, "plain-unmetad")),
        ];
        assert!(native_intern(&mut eval, &args).is_ok());
        let plain_id = eval
            .registry
            .lookup(&Symbol::new(Some("user"), "plain-unmetad"))
            .unwrap();
        eval.registry.with_state_mut(plain_id, |st| st.meta = None);
        assert!(native_intern(&mut eval, &args).is_ok());

        let m = Map::from_kvs_last_wins(vec![(
            Value::Keyword(Keyword::new(None, "tag")),
            Value::Keyword(Keyword::new(None, "k")),
        )]);
        let meta_sym = Symbol::new(None, "metad-var").with_meta(Some(m));
        let args_meta = [
            Value::Symbol(Symbol::new(None, "user")),
            Value::Symbol(meta_sym),
            Value::Int(7),
        ];
        assert!(native_intern(&mut eval, &args_meta).is_ok());
    }

    #[test]
    fn throw_class_for_covers_runtime_exception_subclasses() {
        let cases = [
            ("Error", ThrowClass::Error),
            ("RuntimeException", ThrowClass::RuntimeException),
            (
                "IllegalMonitorStateException",
                ThrowClass::IllegalMonitorStateException,
            ),
            (
                "UnsupportedOperationException",
                ThrowClass::UnsupportedOperationException,
            ),
            ("NullPointerException", ThrowClass::NullPointerException),
            (
                "IndexOutOfBoundsException",
                ThrowClass::IndexOutOfBoundsException,
            ),
            (
                "ArrayIndexOutOfBoundsException",
                ThrowClass::ArrayIndexOutOfBoundsException,
            ),
            ("ArrayStoreException", ThrowClass::ArrayStoreException),
            (
                "NegativeArraySizeException",
                ThrowClass::NegativeArraySizeException,
            ),
            ("ClassNotFoundException", ThrowClass::ClassNotFoundException),
            ("OutOfMemoryError", ThrowClass::OutOfMemoryError),
        ];
        for (name, expected) in cases {
            assert_eq!(throw_class_for(name), expected, "mismatch for {name}");
        }
    }

    #[test]
    fn set_ns_current_without_frame_sets_root() {
        let mut eval = Evaluator::new_bare("user");
        let star = Symbol::new(Some("clojure.core"), "*ns*");
        let var_id = eval.registry.intern(&star);
        let handle = ns_value("user", 1);
        set_ns_current(&mut eval, var_id, handle.clone());
        assert_eq!(eval.deref_var(var_id, &star), handle);
    }

    #[test]
    fn owned_entries_sorted_and_total() {
        // Binary search in both split owned tables needs sorted unique names;
        // every entry must resolve and unknown names must miss.
        for table in [super::OWNED_CORE_ENTRIES, super::OWNED_SEQ_ENTRIES] {
            let names: Vec<&str> = table.iter().map(|(n, _)| *n).collect();
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(names, sorted, "owned entries out of order");
            for name in &names {
                assert!(super::owned_entry(name).is_some(), "missing {name}");
            }
        }
        assert!(super::owned_entry("map").is_none());
        assert!(super::owned_entry("").is_none());
    }

    #[test]
    fn push_core_binding_skips_missing_and_pushes_present() {
        let mut eval = Evaluator::new_bare("user");
        super::push_core_binding(&mut eval, "*missing*", Value::Int(1));
        assert_eq!(eval.bindings_depth(), 0);
        let id = eval
            .registry
            .intern(&Symbol::new(Some("clojure.core"), "*x*"));
        super::push_core_binding(&mut eval, "*x*", Value::Int(42));
        assert_eq!(eval.bindings_depth(), 1);
        assert_eq!(
            eval.deref_var(id, &Symbol::new(Some("clojure.core"), "*x*")),
            Value::Int(42)
        );
    }
}
