// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Build-time front end of the native compiler (issue #259 §2.1).
//!
//! Reads a program one top-level form at a time with the live
//! reader (data readers, `::kw`, `*ns*` exactly as `run` sees them),
//! macroexpands and splits top-level `do` like `eval_top_form`, and
//! analyzes each unit into an [`Expr`] with the interpreter's own
//! analyzer. Between units it applies the **environment effects**
//! later reads and expansions depend on — namespaces, requires,
//! macros, and plain fn definitions (so macros calling helper fns
//! expand) — by evaluating exactly those units on the build
//! evaluator. Every other unit is only analyzed: nothing like
//! `(slurp *in*)` ever runs during `build`.

use std::sync::Arc;

use clj_reader::Span;
use clj_value::Value;

use crate::error::span_of;
use crate::eval::{Evaluator, SourceError, split_top_do};
use crate::expr::{Arity, Expr, FunExpr, NsClause, Pattern};

/// Core fns whose top-level calls change what later forms read or
/// resolve (evaluated at build time as well as compiled).
const ENV_FNS: [&str; 9] = [
    "require",
    "use",
    "import",
    "refer",
    "refer-clojure",
    "in-ns",
    "alias",
    "load",
    "create-ns",
];

/// One analyzed top-level unit.
#[derive(Debug)]
pub struct TopUnit {
    /// The analyzed unit.
    pub expr: Expr,
    /// Whether the build evaluator also evaluated it (environment
    /// effect).
    pub env_effect: bool,
    /// The unit's source span.
    pub span: Span,
}

/// The front end's output: units in load order.
#[derive(Debug, Default)]
pub struct FrontEnd {
    /// Every top-level unit, `do`-split, in load order.
    pub units: Vec<TopUnit>,
    /// How many leading units are spliced library units (#271, see
    /// [`SPLICED_LIBS`]): a full-boot build drops them, its `require`
    /// loading the interpreted library instead.
    pub lib_units: usize,
    /// The namespaces those units define, in splice order.
    pub libs: Vec<&'static str>,
}

/// Vendored pure-Clojure stdlib namespaces a compiled program carries as
/// its own leading units when its leading `ns` form requires them
/// (#271): `(namespace, file, source)`. They compile like user code (so
/// they behave exactly as the interpreted library under `run`), and a
/// program requiring them keeps `BootMode::Lite`.
const SPLICED_LIBS: &[(&str, &str, &str)] =
    &[("clojure.set", "clojure/set.clj", crate::ns::STDLIB_SET_CLJ)];

/// Run the front end over `text` (named `path` in spans) on a
/// booted build evaluator, inside `run`'s own binding frame
/// ([`run_script`](crate::script::run_script): `*ns*`,
/// `*warn-on-reflection*`, … plus the `clojure.string` preload), so
/// every unit expands and analyzes exactly as under `run`. The first
/// read, expansion, or analysis failure stops the front end (the
/// caller decides the fallback).
pub fn front_end(eval: &mut Evaluator, text: &str, path: &str) -> Result<FrontEnd, SourceError> {
    type Slot = Arc<std::sync::Mutex<Option<Result<FrontEnd, SourceError>>>>;
    let slot: Slot = Arc::default();
    let inner = Arc::clone(&slot);
    let text: Arc<str> = Arc::from(text);
    let path: Arc<str> = Arc::from(path);
    eval.install_native(
        FRONT_NS,
        "front-end",
        Arc::new(move |eval: &mut Evaluator, _: &[Value]| {
            let prev = std::mem::replace(&mut eval.source, Arc::clone(&path));
            let out = front_end_inner(eval, &text);
            eval.source = prev;
            *inner.lock().expect("front-end slot") = Some(out);
            Ok(Value::Nil)
        }),
    );
    let script = crate::script::run_script(&[], &format!("({FRONT_NS}/front-end)"));
    eval.eval_source(&script)?;
    slot.lock()
        .expect("front-end slot")
        .take()
        .expect("front-end ran")
}

/// The build-time namespace holding the front-end entry.
const FRONT_NS: &str = "clj.aot.build";

fn front_end_inner(eval: &mut Evaluator, text: &str) -> Result<FrontEnd, SourceError> {
    let mut front = read_units(eval, text)?;
    splice_libs(eval, &mut front)?;
    Ok(front)
}

/// Read, expand, and analyze every top-level form of `text`.
fn read_units(eval: &mut Evaluator, text: &str) -> Result<FrontEnd, SourceError> {
    let mut front = FrontEnd::default();
    let mut forms = eval.live_forms(text);
    while let Some(form) = eval.next_live_form(&mut forms)? {
        unit(eval, &form, &mut front)?;
    }
    Ok(front)
}

/// Prepend the units of every [`SPLICED_LIBS`] library the program's
/// leading `ns` form requires (#271). They run before that form, as the
/// interpreted library would have loaded during its `require`; the build
/// evaluator (which already loaded the library through that `require`)
/// re-evaluates their environment effects, the same definitions.
fn splice_libs(eval: &mut Evaluator, front: &mut FrontEnd) -> Result<(), SourceError> {
    let mut units = Vec::new();
    for &(ns, file, text) in SPLICED_LIBS {
        if !leading_ns_requires(front, ns) {
            continue;
        }
        let prev = eval.swap_source(Arc::from(file));
        let lib = read_units(eval, text);
        eval.swap_source(prev);
        units.extend(lib?.units);
        front.libs.push(ns);
    }
    front.lib_units = units.len();
    units.append(&mut front.units);
    front.units = units;
    Ok(())
}

/// Whether the program's first unit is an `ns` form whose `:require`
/// names `lib`.
fn leading_ns_requires(front: &FrontEnd, lib: &str) -> bool {
    let Some(Expr::Ns { clauses, .. }) = front.units.first().map(|u| &u.expr) else {
        return false;
    };
    clauses.iter().any(|c| match c {
        NsClause::Require(specs) => specs.iter().any(|s| names_lib(s, lib)),
        _ => false,
    })
}

/// Whether a `:require` spec (`lib` or `[lib …]`) names `lib`.
#[must_use]
pub fn names_lib(spec: &Value, lib: &str) -> bool {
    let head = match spec {
        Value::Vector(items) => items.get(0),
        other => Some(other.clone()),
    };
    matches!(head, Some(Value::Symbol(s)) if s.namespace().is_none() && s.name() == lib)
}

/// Expand, split, analyze (and maybe evaluate) one top-level form.
fn unit(eval: &mut Evaluator, form: &Value, front: &mut FrontEnd) -> Result<(), SourceError> {
    let source = Arc::clone(&eval.source);
    let span = span_of(form, Span::new(1, 1));
    let expanded = crate::macros::macroexpand_full(eval, form, span, &source)
        .map_err(|e| eval.compile_err(e))?;
    if let Some(elements) = split_top_do(&expanded) {
        for element in elements {
            unit(eval, element, front)?;
        }
        return Ok(());
    }
    eval.registry.bump_eval_seq();
    let expr = eval.analyze(&expanded).map_err(|e| eval.compile_err(e))?;
    let env_effect = is_env_effect(&expr);
    if env_effect {
        let code = crate::vm::compile_top(&expr);
        crate::vm::run_top(eval, &code).map_err(crate::eval::uncaught)?;
    }
    front.units.push(TopUnit {
        expr,
        env_effect,
        span,
    });
    Ok(())
}

/// Whether a unit changes the environment later units read or
/// expand against (and is safe to run at build time).
#[must_use]
pub fn is_env_effect(expr: &Expr) -> bool {
    match expr {
        Expr::Ns { .. } | Expr::DefMacro { .. } | Expr::Declare { .. } => true,
        // A plain fn definition whose name metadata is side-effect
        // free: macros may call it during later expansions.
        Expr::Def { init, meta, .. } => {
            init.as_deref().is_none_or(|e| matches!(e, Expr::Fn(_)))
                && meta.iter().flatten().all(|(k, v)| is_pure(k) && is_pure(v))
        }
        Expr::Invoke { head, .. } => {
            matches!(head.as_ref(), Expr::Global(var) if is_env_var(var))
        }
        Expr::DirectNative(call) => is_env_var(&call.var),
        _ => false,
    }
}

/// Whether evaluating `expr` cannot have side effects (constants and
/// collection literals of constants).
fn is_pure(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_) | Expr::Quote(_) => true,
        Expr::VecLit(items, _) | Expr::SetLit(items, _) => items.iter().all(is_pure),
        Expr::MapLit(pairs, _) => pairs.iter().all(|(k, v)| is_pure(k) && is_pure(v)),
        _ => false,
    }
}

fn is_env_var(var: &clj_value::Var) -> bool {
    var.sym.namespace() == Some("clojure.core") && ENV_FNS.contains(&var.sym.name())
}

/// Call `f` on every direct child expression of `expr` (patterns'
/// `:or` defaults and fn arity bodies included).
pub fn for_each_child<'a>(expr: &'a Expr, f: &mut dyn FnMut(&'a Expr)) {
    if for_each_simple_child(expr, f) || for_each_binding_child(expr, f) {
        return;
    }
    for_each_scope_child(expr, f);
}

/// Leaf, linear-sequence, and direct-call forms; returns `true` when `expr`
/// matched and was visited.
fn for_each_simple_child<'a>(expr: &'a Expr, f: &mut dyn FnMut(&'a Expr)) -> bool {
    match expr {
        Expr::Literal(_)
        | Expr::Global(_)
        | Expr::Local(_)
        | Expr::Quote(_)
        | Expr::VarRef(_)
        | Expr::NewThrowable { .. }
        | Expr::Declare { .. }
        | Expr::Ns { .. } => true,
        Expr::VecLit(items, _) | Expr::SetLit(items, _) | Expr::Do(items) | Expr::Recur(items) => {
            items.iter().for_each(f);
            true
        }
        Expr::WithOutStr { body, .. }
        | Expr::DelayStar { body, .. }
        | Expr::Sync { body }
        | Expr::Io { body } => {
            body.iter().for_each(f);
            true
        }
        Expr::LazySeqStar { body } => {
            body.iter().for_each(f);
            true
        }
        Expr::Throw(e) | Expr::Time { expr: e } | Expr::Set { val: e, .. } => {
            f(e);
            true
        }
        Expr::If { test, then, els } => {
            f(test);
            f(then);
            f(els);
            true
        }
        Expr::Invoke { head, args, .. } => {
            f(head);
            args.iter().for_each(f);
            true
        }
        Expr::DirectNative(call) => {
            call.args.iter().for_each(f);
            true
        }
        _ => false,
    }
}

/// Binding, definition, and fn-literal forms; returns `true` when `expr`
/// matched and was visited.
fn for_each_binding_child<'a>(expr: &'a Expr, f: &mut dyn FnMut(&'a Expr)) -> bool {
    match expr {
        Expr::MapLit(pairs, _) => {
            visit_expr_pairs(pairs, f);
            true
        }
        Expr::Def { init, meta, .. } => {
            if let Some(e) = init.as_deref() {
                f(e);
            }
            if let Some(pairs) = meta.as_deref() {
                visit_expr_pairs(pairs, f);
            }
            true
        }
        Expr::Let { bindings, body } => {
            visit_expr_bindings(bindings, body, f);
            true
        }
        Expr::Loop { bindings, body } => {
            visit_expr_bindings(bindings, body, f);
            true
        }
        Expr::LetFn { fns, body } => {
            for (_, fun) in fns {
                fun_children(fun, f);
            }
            body.iter().for_each(f);
            true
        }
        Expr::Fn(fun) | Expr::BoundFn { fun, .. } | Expr::DefMacro { fun, .. } => {
            fun_children(fun, f);
            true
        }
        _ => false,
    }
}

/// Dynamic-scope, try/catch, assert, and `case*` forms.
fn for_each_scope_child<'a>(expr: &'a Expr, f: &mut dyn FnMut(&'a Expr)) {
    match expr {
        Expr::Try {
            body,
            catches,
            finally_body,
        } => {
            body.iter().for_each(&mut *f);
            for catch in catches {
                catch.body.iter().for_each(&mut *f);
            }
            finally_body.iter().for_each(f);
        }
        Expr::Binding { pairs, body } | Expr::WithRedefs { pairs, body } => {
            for (_, e) in pairs {
                f(e);
            }
            body.iter().for_each(f);
        }
        Expr::WithBindings { map, body }
        | Expr::WithInStr { s: map, body, .. }
        | Expr::Locking { lockee: map, body } => {
            f(map);
            body.iter().for_each(f);
        }
        Expr::Assert { test, message, .. } => {
            f(test);
            if let Some(m) = message.as_deref() {
                f(m);
            }
        }
        Expr::CaseStar {
            expr,
            default,
            cases,
            ..
        } => {
            f(expr);
            f(default);
            for (_, test, then) in cases {
                f(test);
                f(then);
            }
        }
        _ => {}
    }
}

fn visit_expr_pairs<'a>(pairs: &'a [(Expr, Expr)], f: &mut dyn FnMut(&'a Expr)) {
    for (k, v) in pairs {
        f(k);
        f(v);
    }
}

fn visit_expr_bindings<'a>(
    bindings: &'a [(Pattern, Expr)],
    body: &'a [Expr],
    f: &mut dyn FnMut(&'a Expr),
) {
    for (pat, init) in bindings {
        pattern_children(pat, f);
        f(init);
    }
    body.iter().for_each(f);
}

fn fun_children<'a>(fun: &'a FunExpr, f: &mut dyn FnMut(&'a Expr)) {
    for Arity { fixed, rest, body } in fun.arities.iter() {
        fixed.iter().for_each(|p| pattern_children(p, f));
        rest.iter().for_each(|p| pattern_children(p, f));
        body.iter().for_each(&mut *f);
    }
}

/// Visit every `:or` default expression nested anywhere in `pat`.
pub fn pattern_children<'a>(pat: &'a Pattern, f: &mut dyn FnMut(&'a Expr)) {
    match pat {
        Pattern::Symbol(_) => {}
        Pattern::Vec(v) => {
            v.fixed.iter().for_each(|p| pattern_children(p, f));
            v.rest.iter().for_each(|p| pattern_children(p, f));
        }
        Pattern::Map(m) => {
            m.entries
                .iter()
                .for_each(|(p, _, _, _)| pattern_children(p, f));
            m.or.iter().for_each(|(_, e)| f(e));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::front_end;
    use crate::eval::SourceError;
    use crate::ns::boot;

    #[test]
    fn front_end_reports_macroexpansion_errors() {
        let mut eval = boot();
        let err = front_end(
            &mut eval,
            "(defmacro bad [] (throw (ex-info \"boom\" {}))) (bad)",
            "t.clj",
        )
        .unwrap_err();
        assert!(matches!(err, SourceError::Compile(_)), "{err:?}");
    }

    #[test]
    fn front_end_splices_libs_a_leading_ns_requires() {
        use crate::expr::Expr;
        use clj_value::{Symbol, Value, Vector};
        let mut eval = boot();
        let front = front_end(
            &mut eval,
            "(ns t (:require [clojure.set :as s] clojure.string)) (s/union #{1} #{2})",
            "t.clj",
        )
        .expect("front");
        assert_eq!(front.libs, ["clojure.set"]);
        assert!(matches!(&front.units[0].expr, Expr::Ns { name, .. } if name == "clojure.set"));
        assert!(matches!(
            &front.units[front.lib_units].expr,
            Expr::Ns { name, .. } if name == "t"
        ));
        // Only a leading `ns` form's `:require` splices.
        for src in [
            "",
            "(require 'clojure.set) (ns u (:require clojure.set))",
            "(ns v (:require [clojure.string :as str]) (:refer-clojure :exclude [get]))",
        ] {
            let front = front_end(&mut boot(), src, "t.clj").expect("front");
            assert_eq!((front.lib_units, front.libs.len()), (0, 0), "{src}");
        }
        let lib = || Value::Symbol(Symbol::new(None, "clojure.set"));
        assert!(super::names_lib(&lib(), "clojure.set"));
        assert!(super::names_lib(
            &Value::Vector(Vector::from_vec(vec![lib()])),
            "clojure.set"
        ));
        assert!(!super::names_lib(
            &Value::Vector(Vector::from_vec(Vec::new())),
            "clojure.set"
        ));
        assert!(!super::names_lib(&Value::Int(1), "clojure.set"));
        assert!(!super::names_lib(&lib(), "clojure.string"));
    }

    #[test]
    fn scope_child_skips_non_scope_expr() {
        let mut visited = false;
        super::for_each_scope_child(&clj_eval_expr_literal(), &mut |_| visited = true);
        assert!(!visited);
    }

    fn clj_eval_expr_literal() -> crate::expr::Expr {
        crate::expr::Expr::Literal(clj_value::Value::Nil)
    }
}
