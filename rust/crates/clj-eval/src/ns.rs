// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! S5A2 namespace vars, the loader, and `boot` (normative:
//! `docs/t1174/namespaces-contract.md`, oracles probe35-36).
//!
//! Natives receive the live evaluator (§N), so `in-ns` switches the
//! registry current, `require` evaluates files, and `eval` re-enters
//! [`Evaluator::eval_top_form`]. Stale handles read the registry's
//! tombstones, like the JVM's detached namespace objects.

use std::sync::Arc;

use clj_reader::{ReaderError, Span};
use clj_value::{
    Keyword, List, Map, PrintOpts, Ref, Set, SortSpec, Symbol, ThrowClass, Throwable, Value, Var,
    Vector, new_map, new_set, pr_to_string, seq_vec,
};

use crate::concurrency::DEREF_NIL;
use crate::core::cce_err;
use crate::env::{Env, Namespace, SharedRegistry};
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
const CORE_CLJ: &str = include_str!("core.clj");

/// The embedded `clojure.spec.alpha` slice-1 source (gap-122):
/// `resolve_source` serves this for `clojure/spec/alpha` (unlike
/// `CORE_CLJ` it carries its own `ns` form, so it loads on demand
/// through the ordinary `require` path).
const SPEC_ALPHA_CLJ: &str = include_str!("spec_alpha.clj");

/// The embedded tools.deps config-layer sources (gap-124 slice 1):
/// each carries its own `ns` form and loads on demand through the
/// ordinary `require` path (the `spec_alpha.clj` shape).
const TOOLS_DEPS_UTIL_DIR_CLJ: &str = include_str!("tools_deps_util_dir.clj");
const TOOLS_DEPS_EDN_CLJ: &str = include_str!("tools_deps_edn.clj");
const TOOLS_DEPS_EXTENSIONS_CLJ: &str = include_str!("tools_deps_extensions.clj");

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
        _ => pr_to_string(v, &PrintOpts::print()),
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
    if reg.get(&key).is_some() {
        Ok(key)
    } else {
        Err(ex(format!("No namespace: {key} found")))
    }
}

/// A namespace handle value for a live-or-removed namespace.
fn ns_value(name: &str, id: u64) -> Value {
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
    !reg.state(id).is_some_and(|s| {
        s.meta
            .as_ref()
            .and_then(|m| m.get(&probe))
            .is_some_and(Value::is_truthy)
    })
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
        .or_else(|| reg.get(part).map(|_| part.to_owned()))?;
    let target_ns = reg.ns_live_or_removed(&target)?;
    let q = Symbol::new(Some(target.as_str()), sym.name());
    target_ns.lookup(&q).map(|id| (q, id))
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

/// A qualified symbol's var id, without the alias step (the target
/// is already resolved).
fn id_of(reg: &SharedRegistry, q: &Symbol) -> Option<u64> {
    reg.get(q.namespace()?).and_then(|ns| ns.lookup(q))
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
    let id = eval.registry.ensure_ns(&key).id();
    Ok(ns_value(&key, id))
}

fn native_in_ns(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/in-ns", args, &[1])?;
    let key = sym_str(strict_sym(&args[0])?);
    let id = eval.registry.ensure_ns(&key).id();
    eval.registry.set_current(&key);
    // `RT.inNamespace` is a thread-local `set`: the top frame when
    // bound (boot pushes a base frame, so the root never moves),
    // else the root.
    let star = Symbol::new(Some("clojure.core"), "*ns*");
    if let Some(var_id) = id_of(&eval.registry, &star) {
        set_ns_current(eval, var_id, ns_value(&key, id));
    }
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
        .ns_live_or_removed(&target)
        .expect("alias target is live or tombstoned")
        .id();
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
        Value::Vector(x) => Ok(x.iter().cloned().collect()),
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
fn refer_publics(eval: &Evaluator, home: &Namespace) -> Vec<(Symbol, u64)> {
    let mut publics = Vec::new();
    for (s, i) in home.interns() {
        if is_public(&eval.registry, i) {
            publics.push((s.clone(), i));
        }
    }
    publics.sort_by(|a, b| a.0.name().cmp(b.0.name()));
    publics
}

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
    publics: &[(Symbol, u64)],
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
    match publics.iter().find(|(s, _)| s.name() == item_sym.name()) {
        Some((s, i)) => Ok(((*item_sym).clone(), s.clone(), *i)),
        None => {
            let interned = eval
                .registry
                .get(lib_key)
                .and_then(|home| home.lookup(&Symbol::new(Some(lib_key), item_sym.name())))
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
    let Some(home) = eval.registry.get(&lib_key) else {
        return Err(ex(format!("No namespace: {lib_key}")));
    };
    let pairs = opt_pairs(filters)?;
    let publics = refer_publics(eval, &home);
    let public_names: Vec<Value> = publics
        .iter()
        .map(|(s, _)| Value::Symbol(Symbol::new(None, s.name())))
        .collect();
    let to_do = refer_todo(public_names, &pairs)?;
    let exclude = refer_exclude(&pairs)?;
    let rename = match opt_get(&pairs, "rename") {
        None | Some(Value::Nil) | Some(Value::Bool(false)) => None,
        Some(v) => Some(v),
    };
    for item in &to_do {
        if exclude.contains(item) {
            continue;
        }
        let (item_sym, target_q, var_id) = refer_target(eval, &lib_key, &publics, item)?;
        let renamed = refer_renamed(eval, rename, &item_sym, item)?;
        // Interns win over refers (the JVM's REJECTED warning waits
        // for `*err*` in S5D).
        let cur = eval.current_ns_name();
        eval.registry.ensure_ns(&cur);
        let q = Symbol::new(Some(cur.as_str()), renamed.name());
        eval.registry.with_ns_mut(&cur, |home| {
            if home.lookup(&q).is_none() {
                home.add_refer(renamed.name(), target_q, var_id);
            }
        });
    }
    Ok(Value::Nil)
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
            Ok((id, eval.registry.get(ns_name).map_or(0, |ns| ns.id())))
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
    }
    Ok(Value::Nil)
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

/// A JVM class name (simple or fully qualified) to its
/// [`ThrowClass`]. Unknown names — including the port-only
/// `ReaderError` deferrals — fall back to `Exception`.
pub(crate) fn throw_class_for(name: &str) -> ThrowClass {
    match name.rsplit('.').next().unwrap_or(name) {
        "Exception" => ThrowClass::Exception,
        "Error" => ThrowClass::Error,
        "RuntimeException" => ThrowClass::RuntimeException,
        "IllegalArgumentException" => ThrowClass::IllegalArgumentException,
        "IllegalStateException" => ThrowClass::IllegalStateException,
        "IllegalMonitorStateException" => ThrowClass::IllegalMonitorStateException,
        "UnsupportedOperationException" => ThrowClass::UnsupportedOperationException,
        "AssertionError" => ThrowClass::AssertionError,
        "ArityException" => ThrowClass::ArityException,
        "NullPointerException" => ThrowClass::NullPointerException,
        "ClassCastException" => ThrowClass::ClassCastException,
        "ClassNotFoundException" => ThrowClass::ClassNotFoundException,
        "IndexOutOfBoundsException" => ThrowClass::IndexOutOfBoundsException,
        "ArrayIndexOutOfBoundsException" => ThrowClass::ArrayIndexOutOfBoundsException,
        "ArrayStoreException" => ThrowClass::ArrayStoreException,
        "NegativeArraySizeException" => ThrowClass::NegativeArraySizeException,
        "OutOfMemoryError" => ThrowClass::OutOfMemoryError,
        "ArithmeticException" => ThrowClass::ArithmeticException,
        "ExceptionInfo" => ThrowClass::ExceptionInfo,
        "FileNotFoundException" => ThrowClass::FileNotFoundException,
        "ExecutionException" => ThrowClass::ExecutionException,
        "CancellationException" => ThrowClass::CancellationException,
        "RejectedExecutionException" => ThrowClass::RejectedExecutionException,
        "IllegalAccessError" => ThrowClass::IllegalAccessError,
        "NumberFormatException" => ThrowClass::NumberFormatException,
        "StringIndexOutOfBoundsException" => ThrowClass::StringIndexOutOfBoundsException,
        "IllegalFormatConversionException" => ThrowClass::IllegalFormatConversionException,
        "UnknownFormatConversionException" => ThrowClass::UnknownFormatConversionException,
        "MissingFormatConversionException" => ThrowClass::MissingFormatConversionException,
        "MissingFormatArgumentException" => ThrowClass::MissingFormatArgumentException,
        "IllegalFormatPrecisionException" => ThrowClass::IllegalFormatPrecisionException,
        "IllegalFormatCodePointException" => ThrowClass::IllegalFormatCodePointException,
        "IOException" => ThrowClass::IOException,
        "Compiler$CompilerException" => ThrowClass::CompilerException,
        "LispReader$ReaderException" | "ReaderException" => ThrowClass::ReaderException,
        _ => ThrowClass::Exception,
    }
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
    id_of(&eval.registry, &Symbol::new(Some("clojure.core"), name))
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
pub(crate) fn deref_core(eval: &Evaluator, name: &str) -> Value {
    match core_var(eval, name) {
        Some(id) => eval.deref_var(id, &Symbol::new(Some("clojure.core"), name)),
        None => Value::Nil,
    }
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
    eval_sync(eval, std::slice::from_ref(&call), &env)
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
/// be strings).
fn load_dirs(eval: &Evaluator) -> Result<Vec<String>, NativeError> {
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

/// Resolve a load root (no leading slash) to the `*file*` path and
/// text: embedded libs first (gap-122 slice 1: `clojure/spec/alpha`;
/// gap-124 slice 1: the three tools.deps config namespaces; the
/// flat set lives here when S5D lands), then `.clj` across every
/// `*load-path*` dir, then `.cljc` (`RT.load` lookup order).
fn resolve_source(eval: &Evaluator, root: &str) -> Result<(String, String), NativeError> {
    if root == "clojure/spec/alpha" {
        return Ok((
            "clojure/spec/alpha.clj".to_owned(),
            SPEC_ALPHA_CLJ.to_owned(),
        ));
    }
    if root == "clojure/tools/deps/util/dir" {
        return Ok((
            "clojure/tools/deps/util/dir.clj".to_owned(),
            TOOLS_DEPS_UTIL_DIR_CLJ.to_owned(),
        ));
    }
    if root == "clojure/tools/deps/edn" {
        return Ok((
            "clojure/tools/deps/edn.clj".to_owned(),
            TOOLS_DEPS_EDN_CLJ.to_owned(),
        ));
    }
    if root == "clojure/tools/deps/extensions" {
        return Ok((
            "clojure/tools/deps/extensions.clj".to_owned(),
            TOOLS_DEPS_EXTENSIONS_CLJ.to_owned(),
        ));
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
    let cur = eval.registry.current().to_owned();
    let depth = eval.bindings_depth();
    let source_path = source_path_value(&file_star);
    if let Some(id) = core_var(eval, "*file*") {
        eval.push_binding(core_var_handle(id, "*file*"), file_star);
    }
    // S18: every load binds `*source-path*` alongside `*file*`
    // (no existing test can observe the new var).
    if let Some(id) = core_var(eval, "*source-path*") {
        eval.push_binding(core_var_handle(id, "*source-path*"), source_path);
    }
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
    if let Some(id) = core_var(eval, "*read-eval*") {
        eval.push_binding(core_var_handle(id, "*read-eval*"), Value::Bool(true));
    }
    let out = eval.load_source(spans, text);
    eval.registry.set_current(&cur);
    eval.pop_bindings(depth);
    out
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
    let root = root_resource(lib.name());
    load_paths(
        eval,
        std::slice::from_ref(&Value::Str(Arc::from(root.as_str()))),
    )?;
    if need_ns && eval.registry.get(&sym_str(lib)).is_none() {
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
    let depth = core_var(eval, "*loading-verbosely*").map(|id| {
        eval.push_binding(
            core_var_handle(id, "*loading-verbosely*"),
            Value::Bool(outer_verbose),
        )
    });
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
            if opts.need_ns && eval.registry.get(&sym_str(joined_sym)).is_none() {
                Err(ex(format!("namespace '{}' not found", sym_str(joined_sym))))
            } else {
                Ok(())
            }
        }
    };
    if let Some(d) = depth {
        eval.pop_bindings(d);
    }
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
    let undefined_on_entry = eval.registry.get(&sym_str(&joined_sym)).is_none();
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
        Value::Vector(x) => matches!(
            x.iter().nth(1),
            None | Some(Value::Nil) | Some(Value::Keyword(_))
        ),
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
        match clause {
            NsClause::ReferClojure(filters) => {
                native_refer_clojure(eval, filters).map_err(|e| eval.throw(e))?;
            }
            NsClause::Require(libspecs) => {
                native_require(eval, libspecs).map_err(|e| eval.throw(e))?;
            }
            NsClause::Use(libspecs) => {
                native_use(eval, libspecs).map_err(|e| eval.throw(e))?;
            }
        }
    }
    Ok(())
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
    let out = run_loaded(
        eval,
        Value::Str(Arc::from(file_star.as_str())),
        &file_star,
        &text,
        true,
    );
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
    let abs = absolutize(&path);
    // `Compiler.loadFile`: no `*ns*` isolation, no cycle guard —
    // only `*file*`. Returns the last value (nil when empty).
    match run_loaded(
        eval,
        Value::Str(Arc::from(abs.as_str())),
        &abs,
        &text,
        false,
    ) {
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

/// `load-reader`: read every form from a reader and evaluate
/// them, returning the last value (nil when empty — S7; probes
/// `l-*`). `*file*` is nil, `*ns*` is isolated, `*read-eval*` is
/// forced true (visibly), and `*pending-paths*` is untouched.
fn native_load_reader(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let reader = load_reader_arg(args)?;
    // `Compiler.load` consumes leading whitespace before its entry
    // checks, so entry failures blame the post-skip position; the
    // parser stays out of its lock for the whole read phase (the
    // hook runs user code, which may read the same reader again).
    let mut opts = clj_reader::ReaderOpts {
        read_cond: clj_reader::ReadCond::Allow,
        ..clj_reader::ReaderOpts::default()
    };
    let (mut parser, _) = crate::reader::take_parser(reader, &opts);
    parser.consume_whitespaces();
    let (line, col) = (parser.line(), parser.col());
    let seed = match crate::reader::seed_load_gates(eval) {
        Ok(seed) => seed,
        Err(cce) => {
            let wrap =
                crate::reader::executing_wrap(line, col, &cce, eval.next_id(), eval.next_id());
            crate::reader::restore_parser(reader, parser);
            return Err(source_to_native(eval, "", wrap));
        }
    };
    crate::reader::sync_reader_ns(eval, &mut opts);
    opts.read_eval = seed.read_eval;
    opts.suppress_read = seed.suppress_read;
    opts.resolver_bound = seed.resolver_bound;
    parser.set_opts(&opts);
    let mut forms = Vec::new();
    loop {
        // Per-form S6 consult (B1 — mirrors `read_forms_loop`):
        // snapshot, read, drain, resolve. Data-reader failures
        // surface like `load`'s (raw uncaught throwables, not
        // `CompilerException` wraps).
        let (outcome, suppressed) = match load_reader_step(eval, &mut opts, &mut parser) {
            Ok(step) => step,
            Err(e) => {
                crate::reader::restore_parser(reader, parser);
                let thrown = eval.throwable(e);
                return Err(source_to_native(eval, "", SourceError::Uncaught(thrown)));
            }
        };
        match outcome {
            Ok(Some(form)) => match crate::core::resolve_data_readers(eval, form, suppressed) {
                Ok(form) => forms.push(form),
                Err(e) => {
                    crate::reader::restore_parser(reader, parser);
                    let thrown = eval.throwable(e);
                    return Err(source_to_native(eval, "", SourceError::Uncaught(thrown)));
                }
            },
            Ok(None) => break,
            Err(e) => {
                crate::reader::restore_parser(reader, parser);
                return Err(source_to_native(eval, "", SourceError::Reader(e)));
            }
        }
    }
    crate::reader::restore_parser(reader, parser);
    run_loaded_values(eval, forms)
}

/// Evaluate loaded forms under the `load-reader` frame (`*file*`
/// nil, `*ns*` isolated, `*read-eval*` forced true) and return the
/// last value (nil when empty).
fn run_loaded_values(eval: &mut Evaluator, forms: Vec<Value>) -> Result<Value, NativeError> {
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
    let mut out = Value::Nil;
    for form in &forms {
        match eval.eval_top_form(form) {
            Ok(v) => out = v,
            Err(e) => {
                eval.registry.set_current(&cur);
                eval.pop_bindings(depth);
                return Err(source_to_native(eval, "", e));
            }
        }
    }
    eval.registry.set_current(&cur);
    eval.pop_bindings(depth);
    Ok(out)
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

/// Evaluate one boot form: the embedded boot sequence is fixed, so
/// any failure is a port bug (pinned by `core_loads`), never user
/// input.
fn boot_eval(eval: &mut Evaluator, src: &str) {
    if let Err(e) = eval.eval_source(src) {
        unreachable!("boot form failed ({src}): {e:?}");
    }
}

/// Boot a fresh evaluator: `clojure.core` plus `user` (current),
/// the §N natives and vars installed, every public install referred
/// into `user` (JVM boot refers `clojure.core`), a permanent `*ns*`
/// base frame (so `in-ns` writes the top frame and the root never
/// moves, like `RT.inNamespace` under `clojure.main`), and the
/// embedded `core.clj` composites loaded into `clojure.core` with
/// their publics referred into `user`.
#[must_use]
pub fn boot() -> Evaluator {
    let mut eval = Evaluator::new_bare("user");
    eval.registry.ensure_ns("clojure.core");
    let user_ns = eval.registry.ensure_ns("user").id();
    eval.registry.set_current("user");
    boot_install_natives(&mut eval);
    boot_install_builtin_macros(&mut eval);
    boot_mark_private(&mut eval);
    let user_handle = ns_value("user", user_ns);
    boot_core_vars(&mut eval, user_handle.clone());
    crate::printsys::boot_print_system(&mut eval);
    crate::reader::boot_read_arglists(&mut eval);
    boot_refer_user(&mut eval);
    if let Some(id) = core_var(&eval, "*ns*") {
        eval.push_binding(core_var_handle(id, "*ns*"), user_handle);
    }
    // Embedded `core.clj` composites (their privates resolve in
    // `clojure.core`), then `user` refers the publics — the file
    // carries no `ns` form by design.
    boot_eval(&mut eval, "(in-ns 'clojure.core)");
    boot_eval(&mut eval, CORE_CLJ);
    boot_eval(&mut eval, "(in-ns 'user)");
    boot_eval(&mut eval, "(refer 'clojure.core)");
    // The load's macroexpansions consume `gensym` numbers; the
    // user-visible counter restarts from 1 (S5C pins `G__1` first).
    eval.reset_gensym();
    eval
}

/// Install the §N natives into `clojure.core`.
fn boot_install_natives(eval: &mut Evaluator) {
    for (name, f) in NATIVES
        .iter()
        .chain(crate::vars::VAR_NATIVES.iter())
        .chain(crate::macros::MACRO_NATIVES.iter())
        .chain(crate::lazy::LAZY_NATIVES.iter())
        .chain(crate::core::CORE_NATIVES.iter())
        .chain(crate::xform::XFORM_NATIVES.iter())
        .chain(crate::seqs::SEQS_NATIVES.iter())
        .chain(crate::coll::COLL_NATIVES.iter())
        .chain(crate::re::RE_NATIVES.iter())
        .chain(crate::rng::RNG_NATIVES.iter())
        .chain(crate::io::IO_NATIVES.iter())
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
    {
        let func: Arc<crate::ext::NativeImpl> = Arc::new(*f);
        eval.install_native("clojure.core", name, func);
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

/// Refer every public install into `user` (JVM boot refers
/// `clojure.core`).
fn boot_refer_user(eval: &mut Evaluator) {
    let mut names: Vec<String> = NATIVES
        .iter()
        .chain(crate::vars::VAR_NATIVES.iter())
        .chain(crate::macros::MACRO_NATIVES.iter())
        .chain(crate::lazy::LAZY_NATIVES.iter())
        .chain(crate::core::CORE_NATIVES.iter())
        .chain(crate::seqs::SEQS_NATIVES.iter())
        .chain(crate::coll::COLL_NATIVES.iter())
        .chain(crate::re::RE_NATIVES.iter())
        .chain(crate::rng::RNG_NATIVES.iter())
        .chain(crate::io::IO_NATIVES.iter())
        .chain(crate::reader::READER_NATIVES.iter())
        .chain(crate::concurrency::CONC_NATIVES.iter())
        .chain(crate::stm::STM_NATIVES.iter())
        .chain(crate::fx4_host::FX4_NATIVES.iter())
        .chain(crate::collmisc::COLLMISC_NATIVES.iter())
        .chain(crate::printsys::PRINTSYS_NATIVES.iter())
        .chain(crate::classemu::CLASS_EMU_NATIVES.iter())
        .map(|(n, _)| (*n).to_owned())
        .collect();
    names.extend(
        crate::macros::BUILTIN_MACROS
            .iter()
            .map(|s| (*s).to_owned()),
    );
    names.extend(
        [
            "*ns*",
            "*file*",
            "*loaded-libs*",
            "*pending-paths*",
            "*loading-verbosely*",
            "*load-path*",
            "*out*",
            "*err*",
            "*in*",
            "*assert*",
            "*unchecked-math*",
            "*warn-on-reflection*",
            "*allow-unresolved-vars*",
            "*print-meta*",
            "*print-readably*",
            "*print-dup*",
            "*read-eval*",
            "*suppress-read*",
            "*reader-resolver*",
            "*flush-on-newline*",
            "print-method",
            "print-dup",
            "char-escape-string",
            "char-name-string",
            "*agent*",
            "*math-context*",
            "*print-length*",
            "*print-level*",
            "*print-namespace-maps*",
            "*1",
            "*2",
            "*3",
            "*e",
            "*command-line-args*",
            "*clojure-version*",
            "*data-readers*",
            "*default-data-reader-fn*",
            "default-data-readers",
            "*compile-path*",
            "*compile-files*",
            "*compiler-options*",
            "*source-path*",
            "*fn-loader*",
            "*use-context-classloader*",
            "*verbose-defrecords*",
            "primitives-classnames",
        ]
        .iter()
        .map(|s| (*s).to_owned()),
    );
    let mut refers: Vec<(String, Symbol, u64)> = Vec::new();
    for name in &names {
        let q = Symbol::new(Some("clojure.core"), name);
        if let Some(id) = id_of(&eval.registry, &q).filter(|id| is_public(&eval.registry, *id)) {
            refers.push((name.clone(), q, id));
        }
    }
    eval.registry.with_ns_mut("user", |user| {
        for (name, q, id) in refers {
            user.add_refer(&name, q, id);
        }
    });
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
    #[should_panic(expected = "boot form failed")]
    fn boot_eval_panics_loudly() {
        let mut eval = boot();
        boot_eval(&mut eval, "(defn broken [)");
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
    fn run_loaded_values_without_vars_is_idle() {
        // S18 `load-reader`: a bare evaluator binds nothing and
        // answers nil for empty input (the missing-var arms are the
        // pin — every `if let` here must tolerate absence).
        let mut eval = Evaluator::new_bare("user");
        let out = run_loaded_values(&mut eval, Vec::new()).expect("bare empty load-reader");
        assert_eq!(out, Value::Nil);
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
}
