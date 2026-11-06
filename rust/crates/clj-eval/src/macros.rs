// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S5C macroexpansion (`macroexpand-1`, `macroexpand`, `gensym`)
//! plus the early list-building natives syntax-quote expansions
//! call (`list`, `apply`, `vector`, `hash-map`, `hash-set` —
//! `seq`/`concat` live in `lazy`/`seqs` since S5D went lazy per
//! C4). Oracle: probe38; rules:
//! `docs/spec/t1174/macros-contract.md`. The wrap partition is
//! source-read from `Compiler.macroexpand1`.

use std::sync::Arc;

use clj_reader::Span;
use clj_value::{Keyword, List, Symbol, ThrowClass, Value, Vector, new_map, new_set};

use crate::error::{CompileError, NativeError, Signal, Verb, internal_throwable};
use crate::eval::Evaluator;
use crate::expr::is_special;
use crate::ns::{NativeFnPtr, check_arity, signal_to_native, str_of, throw_class_for};

/// Builtins whose vars resolve as macros (the analyzer
/// intercepts every form, so the roots only matter to `resolve`;
/// `boot` installs the vars from this table).
pub(crate) const BUILTIN_MACROS: &[&str] = &[
    "let",
    "loop",
    "fn",
    "binding",
    "letfn",
    "defmacro",
    "declare",
    "ns",
    "with-redefs",
    "with-bindings",
    "bound-fn",
    "assert",
    "time",
    "with-out-str",
    "with-in-str",
    "locking",
    "vswap!",
    "dosync",
    "sync",
    "io!",
];

/// A resolved macro's current root (the call runs it with
/// `(form, env, args...)`).
pub(crate) struct MacroRef {
    pub root: Value,
}

/// Whether a var id is a macro (`:macro` meta truthy).
pub(crate) fn is_macro_var(eval: &Evaluator, id: u64) -> bool {
    eval.registry
        .meta_flag(id, &clj_value::static_keyword!("macro"))
}

/// Whether a var id is private (`:private` meta truthy; cleared
/// metas read public).
pub(crate) fn is_private_var(eval: &Evaluator, id: u64) -> bool {
    eval.registry
        .meta_flag(id, &clj_value::static_keyword!("private"))
}

/// Fence macros that stay identity (real macros on the JVM; the
/// port still analyzes them). The FX4 nine expand.
const FENCE_MACROS: &[&str] = &[
    "defmacro",
    "declare",
    "ns",
    "assert",
    "time",
    "with-out-str",
    "with-in-str",
];

fn is_expand_macro(name: &str) -> bool {
    matches!(
        name,
        "let"
            | "loop"
            | "fn"
            | "binding"
            | "letfn"
            | "with-redefs"
            | "with-bindings"
            | "bound-fn"
            | "locking"
    )
}

fn is_fence_sym(sym: &Symbol) -> bool {
    sym.namespace() == Some("clojure.core") && FENCE_MACROS.contains(&sym.name())
}

/// Heads `macroexpand-1` leaves alone: real specials (bare or
/// `clojure.core`-qualified) and the seven fence-macro vars. The
/// nine FX4 heads are specials too, but they expand.
fn is_identity_head(eval: &Evaluator, op: &Value) -> bool {
    match op {
        Value::Symbol(sym) => {
            if sym.namespace().is_none_or(|ns| ns == "clojure.core")
                && is_special(sym.name())
                && !is_expand_macro(sym.name())
            {
                return true;
            }
            let cur = eval.current_ns_name();
            eval.registry
                .resolve_in(&cur, sym)
                .is_some_and(|target| is_fence_sym(&target))
        }
        Value::Var(v) => is_fence_sym(&v.sym),
        _ => false,
    }
}

/// `isMacro` (`Compiler.java`): resolve the operator — a symbol
/// via the synced current namespace, or a direct var value — to a
/// macro var. Private cross-namespace macros are the bare
/// `IllegalStateException` (callers wrap per position); anything
/// unresolvable or non-macro is `Ok(None)`.
pub(crate) fn lookup_macro(
    eval: &mut Evaluator,
    op: &Value,
) -> Result<Option<MacroRef>, NativeError> {
    let (id, target) = match op {
        Value::Symbol(sym) => {
            let cur = eval.current_ns_name();
            let Some((target, id)) = eval.registry.resolve_target_and_id(&cur, sym) else {
                return Ok(None);
            };
            (id, target)
        }
        Value::Var(v) => (v.id, v.sym.clone()),
        _ => return Ok(None),
    };
    // Every list head resolves here first: a lazy `core.clj` var
    // (fn or macro) forces before any metadata decides its role.
    eval.force_lazy(id);
    if !is_macro_var(eval, id) {
        return Ok(None);
    }
    let home = target.namespace().unwrap_or("");
    if home != eval.current_ns_name() && is_private_var(eval, id) {
        return Err(NativeError::new(
            ThrowClass::IllegalStateException,
            format!("var: #'{} is not public", target),
        ));
    }
    let root = eval.registry.root_or_unbound(id, &target);
    Ok(Some(MacroRef { root }))
}

/// Display name of a callable root (for the arity-strip check).
fn root_display(root: &Value) -> Option<&str> {
    match root {
        Value::Extension(e) => {
            if let Some(c) = e.as_any().downcast_ref::<crate::ext::Closure>() {
                return Some(&c.display);
            }
            if let Some(n) = e.as_any().downcast_ref::<crate::ext::NativeFn>() {
                return Some(&n.display);
            }
            None
        }
        _ => None,
    }
}

/// Wrap an expansion failure per `Compiler.macroexpand1`: the
/// macro's *own* `ArityException` (name matches the root display)
/// rethrows bare with the 2 hidden `&form`/`&env` params stripped;
/// any other arity error rethrows bare as-is; `IAE` (including
/// `NumberFormatException`), `ISE`, `ExceptionInfo`, and exactly
/// `Exception` wrap as `Syntax error`; everything else wraps as
/// `Unexpected error`. The wrapper names the operator as written
/// (absent for `Var` operators).
fn wrap_expansion(
    t: &clj_value::Throwable,
    op: &Value,
    root: &Value,
    call_argc: usize,
    span: Span,
    source: &Arc<str>,
) -> CompileError {
    let sym = match op {
        Value::Symbol(s) => Some(match s.namespace() {
            Some(ns) => format!("{ns}/{}", s.name()),
            None => s.name().to_owned(),
        }),
        _ => None,
    };
    if t.class == ThrowClass::ArityException {
        // The failing name rides after `passed to: ` (displays
        // never contain the marker).
        let failing = t
            .message
            .as_deref()
            .and_then(|m| m.rsplit("passed to: ").next());
        let own = failing.is_some_and(|name| Some(name) == root_display(root));
        if !own {
            // Not ours (an inner call failed): the original error
            // escapes bare and untouched (probed Q1: `(max)` of
            // nothing inside `case` keeps its `(0)` count).
            return CompileError {
                class: "clojure.lang.ArityException",
                message: t.message.clone().unwrap_or_default().to_string(),
                span,
                source: Arc::clone(source),
                verb: Verb::Compiling,
                sym: None,
                unexpected: false,
                bare: true,
            };
        }
        // The macro received `(form env & user-args)`: strip the 2
        // hidden params from OUR count (exact, no message parse).
        let actual = call_argc.saturating_sub(2);
        let name = failing.unwrap_or("clojure.core/fn");
        return CompileError {
            class: "clojure.lang.ArityException",
            message: NativeError::arity(name, actual).into_inner().message,
            span,
            source: Arc::clone(source),
            verb: Verb::Compiling,
            sym: None,
            unexpected: false,
            bare: true,
        };
    }
    // `Exception` exactly (not subclasses) joins the `Syntax` side.
    let syntax = matches!(
        t.class,
        ThrowClass::IllegalArgumentException
            | ThrowClass::NumberFormatException
            | ThrowClass::IllegalStateException
            | ThrowClass::ExceptionInfo
            | ThrowClass::Exception
    );
    CompileError {
        class: t.class.jvm_name(),
        message: t.message.as_deref().unwrap_or("").to_owned(),
        span,
        source: Arc::clone(source),
        verb: Verb::Macroexpanding,
        sym,
        unexpected: !syntax,
        bare: false,
    }
}

/// Expand one macro step: run the root with `(form, env,
/// args...)`. `env` is the analyzer's local map (or nil at top);
/// natives pass nil. The wrap blames the evaluator's current
/// source (every caller passed that same source).
pub(crate) fn expand1(
    eval: &mut Evaluator,
    op: &Value,
    root: &Value,
    form: &Value,
    env: &Value,
    args: &[Value],
    span: Span,
) -> Result<Value, CompileError> {
    let mut call_args = Vec::with_capacity(args.len() + 2);
    call_args.push(form.clone());
    call_args.push(env.clone());
    call_args.extend(args.iter().cloned());
    let argc = call_args.len();
    let source = Arc::from(eval.source());
    match eval.call(root, &call_args) {
        Ok(v) => Ok(v),
        Err(s) => Err(expansion_err(s, op, root, argc, span, &source)),
    }
}

/// A macro-call failure as a compile error (an escaping `recur`
/// is impossible — the call runs inside the expansion frame —
/// and surfaces as an internal error, never a panic).
fn expansion_err(
    s: Signal,
    op: &Value,
    root: &Value,
    call_argc: usize,
    span: Span,
    source: &Arc<str>,
) -> CompileError {
    match s {
        Signal::Throw(t) => wrap_expansion(&t, op, root, call_argc, span, source),
        Signal::Recur(_) => wrap_expansion(
            &internal_throwable("recur escaped a macro call".to_owned()),
            op,
            root,
            call_argc,
            span,
            source,
        ),
    }
}

/// `tagOf` (`Compiler.java`): the `:tag` meta entry as a
/// symbol — symbols read directly, strings intern bare,
/// anything else (including absent) reads nil.
fn tag_of(v: &Value) -> Option<Symbol> {
    let tag = v.meta()?.get(&Value::Keyword(Keyword::new(None, "tag")))?;
    match tag {
        Value::Symbol(s) => Some(s.clone()),
        Value::Str(s) => Some(Symbol::new(None, s)),
        _ => None,
    }
}

/// Upstream 8aa45d128 (`Compiler.macroexpand1`): when the input
/// form carries a type-hint tag the expansion drops (tags
/// differ), wrap as `(clojure.core/identity <exp>)` carrying the input form's
/// meta, so the hint survives expansion. Equal tags — and
/// unhinted forms — pass through untouched. The analyzer's
/// macro path shares this (`analyzeSeq` funnels through
/// `macroexpand1`).
pub(crate) fn maybe_wrap_hint(form: &Value, exp: Value) -> Value {
    let Some(tag) = tag_of(form) else {
        return exp;
    };
    if Some(&tag) == tag_of(&exp).as_ref() {
        return exp;
    }
    let head = Value::Symbol(Symbol::new(Some("clojure.core"), "identity"));
    Value::List(List::from_vec(vec![head, exp])).with_meta(form.meta().cloned())
}

/// One `macroexpand-1` step over any form (natives and
/// `eval_top_form` share this): non-seqs, `()`, special/builtin
/// operators (bare or `clojure.core`-qualified), unresolvable
/// operators, and non-macro `Var` operators return unchanged;
/// macros expand with a nil `&env`.
pub(crate) fn macroexpand1_value(
    eval: &mut Evaluator,
    form: &Value,
    span: Span,
    source: &Arc<str>,
) -> Result<Value, CompileError> {
    let Value::List(l) = form else {
        return Ok(form.clone());
    };
    let Some(op) = l.first() else {
        return Ok(form.clone());
    };
    if is_identity_head(eval, op) {
        return Ok(form.clone());
    }
    let keen = match lookup_macro(eval, op) {
        Err(private) => {
            let inner = private.into_inner();
            return Err(CompileError {
                class: inner.class.jvm_name(),
                message: inner.message,
                span,
                source: Arc::clone(source),
                verb: Verb::Compiling,
                sym: None,
                unexpected: false,
                bare: true,
            });
        }
        Ok(keen) => keen,
    };
    let Some(mac) = keen else {
        return Ok(form.clone());
    };
    let args: Vec<Value> = l.iter().skip(1).cloned().collect();
    let exp = expand1(eval, op, &mac.root, form, &Value::Nil, &args, span)?;
    Ok(maybe_wrap_hint(form, exp))
}

/// `Compiler.macroexpand`: loop `macroexpand-1` while the head is
/// a macro (equivalent to the JVM's identity loop in every
/// terminating case, hanging in the same self-similar cases).
pub(crate) fn macroexpand_full(
    eval: &mut Evaluator,
    form: &Value,
    span: Span,
    source: &Arc<str>,
) -> Result<Value, CompileError> {
    let mut current = form.clone();
    loop {
        let expanded = macroexpand1_value(eval, &current, span, source)?;
        let headed = match &expanded {
            Value::List(l) => match l.first() {
                None => false,
                Some(op) => {
                    if is_identity_head(eval, op) {
                        return Ok(expanded);
                    }
                    match lookup_macro(eval, op) {
                        Err(_) => return macroexpand1_value(eval, &expanded, span, source),
                        Ok(keen) => keen.is_some(),
                    }
                }
            },
            _ => false,
        };
        if !headed {
            return Ok(expanded);
        }
        current = expanded;
    }
}

/// A `CompileError` as a native error (bare causes stay bare;
/// wrapped chains render `wrapper\ncause`, like `eval`).
fn expansion_to_native(e: CompileError) -> NativeError {
    if e.bare {
        NativeError::new(throw_class_for(e.class), e.message)
    } else {
        NativeError::new(throw_class_for(e.class), e.to_string())
    }
}

fn native_macroexpand_1(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/macroexpand-1", args, &[1])?;
    // The wrap blames the call site (probed: a nested call
    // reports the call's position), never the form's own meta.
    let span = eval.native_call_span().unwrap_or(Span::new(1, 1));
    let source = Arc::from(eval.source());
    macroexpand1_value(eval, &args[0], span, &source).map_err(expansion_to_native)
}

fn native_macroexpand(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/macroexpand", args, &[1])?;
    let span = eval.native_call_span().unwrap_or(Span::new(1, 1));
    let source = Arc::from(eval.source());
    macroexpand_full(eval, &args[0], span, &source).map_err(expansion_to_native)
}

fn native_gensym(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/gensym", args, &[0, 1])?;
    let prefix = args.first().map_or_else(|| "G__".to_owned(), str_of);
    let n = eval.next_gensym();
    // `Symbol.intern` semantics: a `/` in the prefix splits off a
    // namespace (JVM-oracle-verified: `(gensym "a/")` namespaces
    // `"a"`).
    Ok(Value::Symbol(Symbol::intern(&format!("{prefix}{n}"))))
}

// ---------------------------------------------------------------------------
// Early list natives (`list`, `apply`, `vector`, `hash-map`,
// `hash-set`; `seq`/`concat` live in `lazy`/`seqs` since S5D).
// ---------------------------------------------------------------------------

fn native_list(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::List(List::from_vec(args.to_vec())))
}

fn native_vector(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Vector(Vector::from_vec(args.to_vec())))
}

fn native_hash_map(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() % 2 == 1 {
        return Err(NativeError::new(
            ThrowClass::IllegalArgumentException,
            format!(
                "No value supplied for key: {}",
                str_of(&args[args.len() - 1])
            ),
        ));
    }
    let mut out = new_map();
    for i in (0..args.len()).step_by(2) {
        out = out.insert(args[i].clone(), args[i + 1].clone());
    }
    Ok(Value::Map(out.with_hashy(true)))
}

fn native_hash_set(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    let mut out = new_set();
    for arg in args {
        out = out.insert(arg.clone());
    }
    Ok(Value::Set(out))
}

fn native_apply(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    if args.len() < 2 {
        return Err(NativeError::arity("clojure.core/apply", args.len()));
    }
    if crate::ext::as_native(&args[0]).is_some_and(|n| &*n.display == "clojure.core/str") {
        if let Some(s) = crate::core::apply_str_direct(eval, &args[1..])? {
            return Ok(s);
        }
    }
    // The spread realizes (JVM `spreadToArray`); any seq value —
    // including the lazy ones syntax-quote expansions build — seqs.
    // With no leading args it is the argument array as is (one copy
    // of the items, not two).
    let mut spread =
        crate::lazy::collect_all(eval, args[args.len() - 1].clone()).map_err(signal_to_native)?;
    if args.len() > 2 {
        let mut call_args = Vec::with_capacity(args.len() - 2 + spread.len());
        call_args.extend_from_slice(&args[1..args.len() - 1]);
        call_args.append(&mut spread);
        spread = call_args;
    }
    eval.call(&args[0], &spread).map_err(signal_to_native)
}

/// The S5C natives installed into `clojure.core` by [`boot`](crate::ns::boot).
pub(crate) const MACRO_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("macroexpand-1", native_macroexpand_1),
    ("macroexpand", native_macroexpand),
    ("gensym", native_gensym),
    // Early (S5C rows): syntax-quote expansions call these.
    ("list", native_list),
    ("apply", native_apply),
    ("vector", native_vector),
    ("hash-map", native_hash_map),
    ("hash-set", native_hash_set),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_recur_is_internal() {
        // Defense in depth: an escaping `recur` (impossible — the
        // call runs inside the expansion frame) surfaces as an
        // internal error.
        let source: Arc<str> = Arc::from("NO_SOURCE_PATH");
        let err = expansion_err(
            Signal::Recur(vec![]),
            &Value::Nil,
            &Value::Nil,
            2,
            Span::new(1, 1),
            &source,
        );
        assert!(err.message.contains("recur escaped a macro call"));
    }
}
