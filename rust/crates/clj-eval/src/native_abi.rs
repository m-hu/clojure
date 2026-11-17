// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! The safe evaluation steps natively compiled code calls (issue
//! #259 §2.4). Each function is one step of [`Evaluator::eval`]
//! with its sub-expressions already evaluated by native code (values
//! in, value out), so compiled programs and `run` share every
//! semantic — errors, laziness, var resolution — by construction.
//!
//! Bodies the interpreter evaluates under a dynamic extent (`try`,
//! `binding`, …) arrive as zero-argument fn values compiled
//! natively (`body` below): the step sets up the extent and calls
//! the body through [`Evaluator::call`].

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use clj_reader::Span;
use clj_value::{
    BindCounts, Keyword, List, Map, Set, Symbol, ThrowClass, Throwable, Value, Var, Vector, new_map,
};

use crate::error::{NativeError, Signal};
use crate::eval::Evaluator;
use crate::expr::{CaseTest, NsClause};

/// The result of every step: a value or a pending throw.
pub type Step = Result<Value, Signal>;

/// Call `f` with `args` exactly like an interpreted call.
pub fn invoke(eval: &mut Evaluator, f: &Value, args: &[Value]) -> Step {
    eval.call(f, args)
}

/// The runtime var for a fully qualified symbol (interned like the
/// analyzer does when it resolves a def target or a global).
pub fn intern_var(eval: &mut Evaluator, ns: &Arc<str>, name: &Arc<str>) -> Var {
    let sym = Symbol::from_shared(Some(Arc::clone(ns)), Arc::clone(name));
    let id = eval.registry.intern(&sym);
    Var { id, sym }
}

/// A var's current value (thread binding, else root; unbound vars
/// deref like the interpreter does).
pub fn deref(eval: &mut Evaluator, var: &Var) -> Value {
    eval.deref_var_cached(var.id, &var.sym)
}

/// `(def sym …)` first half: the target var's id, before the init
/// runs (recursive `defn`s see their own var).
pub fn def_begin(eval: &mut Evaluator, sym: &Symbol) -> u64 {
    eval.def_begin(sym)
}

/// [`def_begin`] for a target var already interned (a compiled plain
/// def's var slot: no symbol constant, no second intern).
pub fn def_begin_var(eval: &mut Evaluator, id: u64) -> u64 {
    eval.def_begin_id(id)
}

/// `(def sym …)` second half: root, metadata, dynamic flag; returns
/// the var.
#[allow(clippy::too_many_arguments)]
pub fn def_finish(
    eval: &mut Evaluator,
    id: u64,
    sym: &Symbol,
    doc: Option<String>,
    value: Option<Value>,
    name_meta: Option<Map>,
    line: u32,
    col: u32,
) -> Value {
    eval.def_finish(id, sym, doc, value, name_meta, Span::new(line, col))
}

/// A vector literal.
#[must_use]
pub fn vec_lit(items: Vec<Value>, meta: Option<Map>) -> Value {
    Value::Vector(Vector::from_vec(items).with_meta(meta))
}

/// A map literal over evaluated `[k0 v0 k1 v1 …]` (keys realized,
/// last wins, like the interpreter).
pub fn map_lit(eval: &mut Evaluator, kvs: &[Value], meta: Option<Map>) -> Step {
    Ok(Value::Map(map_pairs(eval, kvs)?.with_meta(meta)))
}

/// `[k0 v0 …]` as a map (shared by map literals and `def` metadata).
pub fn map_pairs(eval: &mut Evaluator, kvs: &[Value]) -> Result<Map, Signal> {
    let mut pairs = Vec::with_capacity(kvs.len() / 2);
    for kv in kvs.chunks(2) {
        let key = crate::lazy::realize_key(eval, &kv[0])?;
        pairs.push((key, kv.get(1).cloned().unwrap_or(Value::Nil)));
    }
    Ok(Map::from_kvs_last_wins(pairs))
}

/// A set literal (elements realized, like the interpreter).
pub fn set_lit(eval: &mut Evaluator, items: &[Value], meta: Option<Map>) -> Step {
    let mut out = clj_value::new_set();
    for item in items {
        out = out.insert(crate::lazy::realize_key(eval, item)?);
    }
    Ok(Value::Set(out.with_meta(meta)))
}

/// `(throw v)`.
pub fn throw(eval: &mut Evaluator, value: Value) -> Signal {
    eval.throw_value(value)
}

/// A throwable as the value a `catch` binds.
#[must_use]
pub fn caught(thrown: Throwable) -> Value {
    Value::Throwable(Arc::new(thrown))
}

/// A fresh throwable of `class` (the analyzer's `NewThrowable`).
pub fn new_throwable(eval: &mut Evaluator, class: ThrowClass, message: &str) -> Value {
    Value::Throwable(Arc::new(Throwable {
        id: eval.next_id(),
        class,
        message: Some(Arc::from(message)),
        data: None,
        cause: None,
    }))
}

/// `(try body (catch C e handler)… (finally fin))` over natively
/// compiled zero-arg `body`/`fin` fns and one-arg handlers, with the
/// interpreter's ordering: finally after the body or the handler,
/// a throwing finally wins.
pub fn try_call(
    eval: &mut Evaluator,
    body: &Value,
    catches: &[(String, Value)],
    fin: Option<&Value>,
) -> Step {
    let run_fin = |eval: &mut Evaluator| -> Result<(), Signal> {
        match fin {
            Some(f) => eval.call(f, &[]).map(|_| ()),
            None => Ok(()),
        }
    };
    let signal = match eval.call(body, &[]) {
        Ok(value) => {
            run_fin(eval)?;
            return Ok(value);
        }
        Err(signal) => signal,
    };
    // Catches match throws only; anything else unwinds through the
    // finally like an unmatched throw.
    let thrown = thrown_of(&signal);
    for (class, handler) in catches {
        if let Some(thrown) = thrown.filter(|t| t.class.matches(class)) {
            let result = eval.call(handler, &[caught(thrown.clone())]);
            run_fin(eval)?;
            return result;
        }
    }
    run_fin(eval)?;
    Err(signal)
}

/// The throwable a signal carries (`None` for a non-throw signal).
fn thrown_of(signal: &Signal) -> Option<&Throwable> {
    match signal {
        Signal::Throw(t) => Some(t),
        Signal::Recur(_) => None,
    }
}

/// `(binding [v x …] body)` over evaluated values.
pub fn binding_call(eval: &mut Evaluator, pairs: Vec<(Var, Value)>, body: &Value) -> Step {
    binding_call_with(eval, pairs, &|ev: &mut Evaluator| ev.call(body, &[]))
}

/// The `binding` core: non-dynamic vars refuse, then one frame
/// around `body`.
pub(crate) fn binding_call_with(
    eval: &mut Evaluator,
    pairs: Vec<(Var, Value)>,
    body: crate::stm::TxBody<'_>,
) -> Step {
    for (var, _) in &pairs {
        if !eval.registry.is_dynamic(var.id) {
            return Err(eval.throw(NativeError::new(
                ThrowClass::IllegalStateException,
                format!("Can't dynamically bind non-dynamic var: {}", var.sym),
            )));
        }
    }
    let depth = eval.bindings_depth();
    eval.push_frame(pairs);
    let result = body(eval);
    eval.pop_bindings(depth);
    result
}

/// `(set! var v)`.
pub fn set_var(eval: &mut Evaluator, var: &Var, value: Value) -> Step {
    if eval.set_binding_top(var.id, value.clone()) {
        Ok(value)
    } else {
        Err(eval.throw(NativeError::new(
            ThrowClass::IllegalStateException,
            format!(
                "Can't change/establish root binding of: {} with set",
                var.sym.name()
            ),
        )))
    }
}

/// `(declare a b …)`: each name defined rootless with `:declared`.
pub fn declare(eval: &mut Evaluator, syms: &[Symbol], line: u32, col: u32) -> Value {
    let mut last = Value::Nil;
    for sym in syms {
        let meta = sym.meta().cloned().unwrap_or_else(new_map).insert(
            Value::Keyword(Keyword::new(None, "declared")),
            Value::Bool(true),
        );
        let sym = sym.clone().with_meta(Some(meta));
        let id = eval.def_begin(&sym);
        last = eval.def_finish(id, &sym, None, None, None, Span::new(line, col));
    }
    last
}

/// `(ns name …)` with its data clauses.
pub fn ns(
    eval: &mut Evaluator,
    name: &str,
    doc: Option<&str>,
    attr: Option<&Map>,
    clauses: &[NsClause],
) -> Step {
    crate::ns::eval_ns_form(eval, name, doc, attr, clauses)
}

/// The `case*` dispatch: `Ok(None)` takes the default; `Ok(Some(i))`
/// takes case `i`, whose test (when `checked`) the caller must still
/// confirm with [`equiv`] against the returned realized key.
pub fn case_select(
    eval: &mut Evaluator,
    value: &Value,
    shift: i32,
    mask: i32,
    keys: &[i32],
    skip: &[i32],
    test: CaseTest,
) -> Result<CaseHit, Signal> {
    if test == CaseTest::Int {
        let Some(key) = crate::eval::case_int_value(value) else {
            return Ok(CaseHit::Default);
        };
        let key = crate::eval::case_shift_mask(key, shift, mask);
        return Ok(match keys.iter().position(|k| *k == key) {
            None => CaseHit::Default,
            Some(i) => CaseHit::Check(i, value.clone()),
        });
    }
    if clj_value::seq_hash_unsupported(value) {
        return Err(eval.unsupported_hash());
    }
    let realized = crate::lazy::realize_deep(eval, value)?;
    if clj_value::seq_hash_unsupported(&realized) {
        return Err(eval.unsupported_hash());
    }
    let key = crate::eval::case_shift_mask(clj_value::hash_code(&realized), shift, mask);
    Ok(match keys.iter().position(|k| *k == key) {
        None => CaseHit::Default,
        Some(i) if skip.contains(&keys[i]) => CaseHit::Direct(i),
        Some(i) => CaseHit::Check(i, realized),
    })
}

/// A [`case_select`] outcome.
#[derive(Debug)]
pub enum CaseHit {
    /// Take the default branch.
    Default,
    /// Take case `i` without a test check.
    Direct(usize),
    /// Take case `i` only if its test `equiv`s the realized key.
    Check(usize, Value),
}

/// Clojure `=` (`equiv`) as used by `case*`.
pub fn equiv(eval: &mut Evaluator, a: &Value, b: &Value) -> Result<bool, Signal> {
    crate::lazy::equiv_vals(eval, a, b)
}

/// Positional destructuring: `(nth coll i nil)`.
pub fn nth_or_nil(eval: &mut Evaluator, coll: &Value, i: usize) -> Step {
    crate::eval::nth_or_nil(coll, i, eval)
}

/// `&`-destructuring's `gseq`.
pub fn gseq(eval: &mut Evaluator, value: &Value) -> Step {
    eval.destructure_gseq(value)
}

/// `(first s)`.
pub fn first(eval: &mut Evaluator, s: &Value) -> Step {
    crate::lazy::first_value(eval, s)
}

/// `(next s)`.
pub fn next(eval: &mut Evaluator, s: &Value) -> Step {
    crate::lazy::next_value(eval, s)
}

/// Rest args of a variadic call as the interpreter passes them.
#[must_use]
pub fn rest_args(args: &[Value]) -> Value {
    crate::eval::rest_list(args)
}

/// The arity error text the interpreter raises for `display`.
pub fn arity_error(eval: &mut Evaluator, display: &str, got: usize) -> Signal {
    eval.throw(NativeError::arity(display, got))
}

/// A `NativeError` as a pending throw.
pub fn native_throw(eval: &mut Evaluator, err: NativeError) -> Signal {
    eval.throw(err)
}

/// A pending throw as the `NativeError` a native fn returns to the
/// interpreter (the throwable rides along unchanged).
#[must_use]
pub fn signal_to_native(signal: Signal) -> NativeError {
    crate::ns::signal_to_native(signal)
}

/// Run one native fn invocation under the interpreter's call guard
/// (call-depth limit raising `StackOverflowError`, segmented stack
/// growth), so deep native recursion behaves exactly like `run`.
pub fn guarded(eval: &mut Evaluator, f: impl FnOnce(&mut Evaluator) -> Step) -> Step {
    eval.guarded_call(f)
}

/// The throwable class a catch/`NewThrowable` names (JVM spelling,
/// simple or qualified; unknown names fall back like the
/// interpreter).
#[must_use]
pub fn throw_class(name: &str) -> ThrowClass {
    crate::ns::throw_class_for(name)
}

/// Read one printed constant (the reader fallback for constants with
/// no direct builder: collections, metadata, big numbers…).
pub fn read_constant(text: &str) -> Result<Value, String> {
    clj_reader::read_str(
        text,
        &clj_reader::ReaderOpts::default(),
        &mut clj_reader::NoHooks,
    )
    .map_err(|e| e.message())
}

/// `^{k v …} ns/name` exactly as [`read_constant`] builds it — the
/// metadata pairs (`kvs`, interleaved) inserted one by one into an empty
/// map, like the reader's merge — so pooled symbol constants with
/// metadata (`defn-`'s `{:private true}`) skip the reader (#271).
#[must_use]
pub fn meta_symbol(ns: Option<&str>, name: &str, kvs: &[Value]) -> Value {
    let mut meta = new_map();
    for pair in kvs.chunks(2) {
        meta = meta.insert(pair[0].clone(), pair[1].clone());
    }
    Value::Symbol(Symbol::new(ns, name)).with_meta(Some(meta))
}

/// A metadata-free constant collection over its items, built by the
/// reader's own constructors (#271: pooled constants skip the reader).
/// `kind`: 0 list, 1 vector, 2 map (`items` interleave keys and
/// values), anything else a set.
#[must_use]
pub fn const_coll(kind: u64, items: &[Value]) -> Value {
    match kind {
        0 if items.is_empty() => Value::List(List::new_sync()),
        0 => Value::List(List::from_vec(items.to_vec())),
        1 => Value::Vector(Vector::from_vec(items.to_vec())),
        2 => {
            let pairs = items
                .chunks(2)
                .map(|p| (p[0].clone(), p[1].clone()))
                .collect();
            Value::Map(Map::from_kvs_checked(pairs).expect("constant map keys are distinct"))
        }
        _ => Value::Set(
            Set::from_items_checked(items.to_vec()).expect("constant set items are distinct"),
        ),
    }
}

/// Print a constant for [`read_constant`], with metadata.
#[must_use]
pub fn print_constant(value: &Value) -> String {
    let opts = clj_value::PrintOpts {
        meta: true,
        ..clj_value::PrintOpts::pr()
    };
    clj_value::pr_to_string(value, &opts)
}

/// The `(class, field)` a static-field constant came from, when the
/// analyzer folded one into a literal (so native code can re-resolve
/// it at startup instead of printing it).
#[must_use]
pub fn static_field_origin(value: &Value) -> Option<(&'static str, &'static str)> {
    crate::classemu::static_field_origin(value)
}

/// Re-resolve a static field (`java.util.Locale/US`, …).
#[must_use]
pub fn static_field(class: &str, field: &str) -> Option<Value> {
    crate::classemu::static_field_value(class, field)
}

// ---- generic destructuring ----------------------------------------------

/// Encode a binding pattern as constant data for
/// [`destructure_encoded`], returning the `:or` default expressions in
/// the order they must be evaluated (pre-order) and passed back.
#[must_use]
pub fn encode_pattern(pat: &crate::expr::Pattern) -> (Value, Vec<&crate::expr::Expr>) {
    let mut defaults = Vec::new();
    let v = enc_pattern(pat, &mut defaults);
    (v, defaults)
}

fn kw(name: &str) -> Value {
    Value::Keyword(Keyword::new(None, name))
}

fn opt_str(s: Option<&String>) -> Value {
    s.map_or(Value::Nil, |s| Value::Str(Arc::from(s.as_str())))
}

fn enc_pattern<'a>(
    pat: &'a crate::expr::Pattern,
    defaults: &mut Vec<&'a crate::expr::Expr>,
) -> Value {
    use crate::expr::Pattern;
    match pat {
        Pattern::Symbol(name) => Value::Str(Arc::from(name.as_str())),
        Pattern::Vec(v) => {
            let fixed = v.fixed.iter().map(|p| enc_pattern(p, defaults)).collect();
            let rest = v
                .rest
                .as_ref()
                .map_or(Value::Nil, |p| enc_pattern(p, defaults));
            vec_lit(
                vec![
                    kw("vec"),
                    vec_lit(fixed, None),
                    rest,
                    opt_str(v.as_sym.as_ref()),
                ],
                None,
            )
        }
        Pattern::Map(m) => {
            let entries = m
                .entries
                .iter()
                .map(|(p, key, or_key, req)| {
                    vec_lit(
                        vec![
                            enc_pattern(p, defaults),
                            key.clone(),
                            or_key.clone(),
                            Value::Bool(*req),
                        ],
                        None,
                    )
                })
                .collect();
            let keys = m
                .keys
                .iter()
                .map(|k| {
                    vec_lit(
                        vec![
                            Value::Str(Arc::from(k.local.as_str())),
                            k.key.clone(),
                            Value::Bool(k.required),
                        ],
                        None,
                    )
                })
                .collect();
            let or_keys =
                m.or.iter()
                    .map(|(key, e)| {
                        defaults.push(e);
                        key.clone()
                    })
                    .collect();
            vec_lit(
                vec![
                    kw("map"),
                    vec_lit(entries, None),
                    vec_lit(keys, None),
                    vec_lit(m.extra_keys.clone(), None),
                    opt_str(m.as_sym.as_ref()),
                    vec_lit(or_keys, None),
                    opt_str(m.select.as_ref()),
                    opt_str(m.all.as_ref()),
                    opt_str(m.excess.as_ref()),
                    opt_str(m.missing.as_ref()),
                ],
                None,
            )
        }
    }
}

fn dec_str(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::Str(s)) => Some(s.to_string()),
        _ => None,
    }
}

fn dec_items(v: Option<&Value>) -> Vec<Value> {
    match v {
        Some(Value::Vector(v)) => v.iter().collect(),
        _ => Vec::new(),
    }
}

fn dec_pattern(v: &Value, defaults: &mut std::slice::Iter<'_, Value>) -> crate::expr::Pattern {
    use crate::expr::{Expr, KeyBinding, MapPattern, Pattern, VecPattern};
    let items = match v {
        Value::Str(s) => return Pattern::Symbol(s.to_string()),
        Value::Vector(v) => v.iter().collect::<Vec<_>>(),
        _ => return Pattern::Symbol(String::new()),
    };
    let is_vec = matches!(items.first(), Some(Value::Keyword(k)) if k.name() == "vec");
    if is_vec {
        let fixed = dec_items(items.get(1))
            .iter()
            .map(|p| dec_pattern(p, defaults))
            .collect();
        let rest = match items.get(2) {
            Some(Value::Nil) | None => None,
            Some(p) => Some(Box::new(dec_pattern(p, defaults))),
        };
        return Pattern::Vec(VecPattern {
            fixed,
            rest,
            as_sym: dec_str(items.get(3)),
        });
    }
    let entries = dec_items(items.get(1))
        .iter()
        .map(|e| {
            let parts = dec_items(Some(e));
            let p = dec_pattern(&parts[0], defaults);
            (p, parts[1].clone(), parts[2].clone(), parts[3].is_truthy())
        })
        .collect();
    let keys = dec_items(items.get(2))
        .iter()
        .map(|k| {
            let parts = dec_items(Some(k));
            KeyBinding {
                local: dec_str(parts.first()).unwrap_or_default(),
                key: parts[1].clone(),
                required: parts[2].is_truthy(),
            }
        })
        .collect();
    let or = dec_items(items.get(5))
        .into_iter()
        .map(|key| {
            (
                key,
                Expr::Literal(defaults.next().cloned().unwrap_or(Value::Nil)),
            )
        })
        .collect();
    Pattern::Map(MapPattern {
        entries,
        keys,
        extra_keys: dec_items(items.get(3)),
        as_sym: dec_str(items.get(4)),
        or,
        select: dec_str(items.get(6)),
        all: dec_str(items.get(7)),
        excess: dec_str(items.get(8)),
        missing: dec_str(items.get(9)),
    })
}

/// Destructure `value` against an [`encode_pattern`]-encoded pattern
/// with its `:or` defaults already evaluated, returning the bound
/// values in binding order — the interpreter's own `destructure`.
pub fn destructure_encoded(
    eval: &mut Evaluator,
    encoded: &Value,
    value: &Value,
    defaults: &[Value],
) -> Result<Vec<Value>, Signal> {
    let pat = dec_pattern(encoded, &mut defaults.iter());
    let mut out = Vec::new();
    eval.destructure_in(&pat, value, &crate::env::Env::new(), &mut out)?;
    Ok(out)
}

/// How many locals a pattern binds.
#[must_use]
pub fn pattern_local_count(pat: &crate::expr::Pattern) -> usize {
    crate::expr::pattern_names(pat).len()
}

/// A class value by fully qualified name (folded class constants).
pub fn class_named(eval: &mut Evaluator, name: &str) -> Value {
    crate::classemu::resolve_class_name(eval, name).map_or(Value::Nil, Value::Class)
}

/// The name of a class constant.
#[must_use]
pub fn class_name_of(value: &Value) -> Option<String> {
    match value {
        Value::Class(c) => Some(c.name().to_owned()),
        _ => None,
    }
}

/// The registry's root epoch (bumps on every var root change).
#[must_use]
pub fn root_epoch(eval: &Evaluator) -> (u64, u64) {
    (eval.registry.epoch(), eval.bindings.version())
}

/// The registry's standalone root-generation counter (even outside
/// mutations).
#[must_use]
pub fn registry_epoch(eval: &Evaluator) -> u64 {
    eval.registry.epoch()
}

/// Share the evaluator's atomic root-generation counter.
#[must_use]
pub fn registry_epoch_handle(eval: &Evaluator) -> Arc<AtomicU64> {
    eval.registry.epoch_handle()
}

/// Share the process-wide per-var binding refcount table.
#[must_use]
pub fn bind_counts() -> Arc<BindCounts> {
    crate::env::bind_counts()
}

/// The argument-count bit a caller of compiled code sets when it hands
/// its argument array over (`call_owned`): the callee may then take a
/// param it consumes (moving it out, leaving nil) instead of cloning it,
/// so e.g. `reduce`'s accumulator reaches `assoc` uniquely owned (#271).
/// The count itself is the low bits.
pub const TAKE_ARGS: u64 = 1 << 62;

/// Parse a JVM `Long/parseLong` base-10 string (`None` on malformed or out-of-range).
#[must_use]
pub fn parse_long_str(s: &str) -> Option<i64> {
    crate::core::parse_long_str(s)
}

/// Parse a JVM `Double/parseDouble` string (`None` on malformed input).
#[must_use]
pub fn parse_double_str(s: &str) -> Option<f64> {
    crate::core::parse_double_str(s)
}

/// Call `callee` (a cached var root) with `args` by value: a native
/// runs its entry directly (an owned-args entry gets them moved in —
/// no per-arg clone/drop, and head-releasing loopers can free as they
/// walk); anything else takes the general call.
pub fn call_value_owned(
    eval: &mut Evaluator,
    callee: &Value,
    args: smallvec::SmallVec<[Value; 4]>,
) -> Step {
    if let Some(native) = crate::ext::as_native(callee) {
        return match &native.owned {
            Some(owned) => owned(eval, args).map_err(|e| eval.throw(e)),
            None => (native.func)(eval, &args).map_err(|e| eval.throw(e)),
        };
    }
    eval.call(callee, &args)
}

// ---- intrinsic steps (borrowed args, no var lookup) -------------------
// Valid only where the compiled program cannot rebind the core var;
// each mirrors the core native's own fast path and falls back to it.

/// Enter a direct native-to-native call (see
/// `Evaluator::enter_native_call`).
pub fn enter_call(eval: &mut Evaluator) -> bool {
    eval.enter_native_call()
}

/// Leave a call entered with [`enter_call`].
pub fn leave_call(eval: &mut Evaluator) {
    eval.leave_native_call();
}

/// A core native's entries, resolved once (the compiled program
/// cannot rebind core vars when it uses these).
pub struct CoreEntry {
    func: Arc<crate::ext::NativeImpl>,
    owned: Option<Arc<crate::ext::OwnedNativeImpl>>,
}

/// A Lite native's [`CoreEntry`] straight from the registered tables —
/// compiled code calling it directly needs no var, so its install stays
/// lazy (`None`: not a Lite native; see `aot_boot::lite_entry`).
#[must_use]
pub fn lite_core_entry(
    decl: (&str, &str, u32),
    tables: crate::aot_boot::LiteTables,
) -> Option<CoreEntry> {
    crate::aot_boot::lite_entry(decl, tables).map(|(func, owned)| CoreEntry { func, owned })
}

/// Resolve var `var`'s root as a [`CoreEntry`] (`None`: not a native).
pub fn core_entry(eval: &mut Evaluator, var: &Var) -> Option<CoreEntry> {
    let root = eval.deref_var_cached(var.id, &var.sym);
    crate::ext::as_native(&root).map(|n| CoreEntry {
        func: Arc::clone(&n.func),
        owned: n.owned.clone(),
    })
}

/// Call a resolved core entry with moved args.
pub fn call_core(
    eval: &mut Evaluator,
    entry: &CoreEntry,
    args: smallvec::SmallVec<[Value; 4]>,
) -> Step {
    match &entry.owned {
        Some(owned) => owned(eval, args).map_err(|e| eval.throw(e)),
        None => (entry.func)(eval, &args).map_err(|e| eval.throw(e)),
    }
}

/// The VM's inline op for core fn `name` called with `arity` args (the
/// id for [`apply_inline`]), if it has one.
#[must_use]
pub fn inline_op(name: &str, arity: usize) -> Option<u32> {
    use crate::vm_ops::{APPLY1, APPLY2, APPLY3, APPLYN, RE_MATCHES_OP};
    if arity == 2 && name == RE_MATCHES_OP.0 {
        return Some(RE_MATCHES_TAG);
    }
    let pos = match arity {
        1 => APPLY1.iter().position(|r| r.0 == name),
        2 => APPLY2.iter().position(|r| r.0 == name),
        3 => APPLY3.iter().position(|r| r.0 == name),
        _ => None,
    };
    let nary = || {
        APPLYN
            .iter()
            .position(|r| r.0 == name)
            .filter(|_| arity >= 3)
            .map(|i| i as u32 | NARY_OP)
    };
    pos.map(|i| i as u32).or_else(nary)
}

/// [`inline_op`]'s mark for a 3+-argument fold or comparison chain.
const NARY_OP: u32 = 1 << 30;
/// Dedicated inline tag for `(re-matches pat s)` so `APPLY2` stays regex-free.
const RE_MATCHES_TAG: u32 = 1 << 29;

/// Bit mask covering disjoint op-namespace tag bits so [`RE_MATCHES_TAG`],
/// [`NARY_OP`], and [`OWN_BIT`] can never alias inside shared `u32` dispatch helpers.
const TAG_MASK: u32 = RE_MATCHES_TAG | NARY_OP | OWN_BIT;

/// Whether `id` is the dedicated regex inline op (`(re-matches ...)`).
#[must_use]
pub const fn is_re_op(id: u32) -> bool {
    id == RE_MATCHES_TAG
}

/// Apply inline op `id` ([`inline_op`] at `args.len()`) to borrowed
/// args — the same inline answers the VM gives (excluding regex ops so
/// non-regex AOT binaries strip `regex_automata`).
#[must_use]
pub fn apply_inline(id: u32, args: &[&Value]) -> Option<Value> {
    use crate::vm_ops::APPLYN;
    match args {
        [a] => apply1(id, a),
        [a, b] => apply2_no_re(id, a, b),
        [a, b, c] => apply3(id, a, b, c),
        _ if id != u32::MAX && (id & TAG_MASK) == NARY_OP => {
            let (_, op, chain) = APPLYN.get((id & !NARY_OP) as usize)?;
            nary(*op, *chain, args)
        }
        _ => None,
    }
}

/// [`apply_inline`] for one arg.
#[must_use]
pub fn apply1(id: u32, a: &Value) -> Option<Value> {
    crate::vm_ops::APPLY1.get(id as usize)?.1(a)
}

/// [`apply_inline`] for two args (including `re-matches`).
#[must_use]
pub fn apply2(id: u32, a: &Value, b: &Value) -> Option<Value> {
    if id == RE_MATCHES_TAG {
        return apply_re_matches(a, b);
    }
    apply2_no_re(id, a, b)
}

/// [`apply_inline`] for two args excluding `re-matches` (keeps `regex_automata`
/// stripped from non-regex AOT binaries).
#[must_use]
pub fn apply2_no_re(id: u32, a: &Value, b: &Value) -> Option<Value> {
    crate::vm_ops::APPLY2.get(id as usize)?.1(a, b)
}

/// Inline `(re-matches pat s)` fast path for `rt_n_op2_re`.
#[must_use]
pub fn apply_re_matches(a: &Value, b: &Value) -> Option<Value> {
    (crate::vm_ops::RE_MATCHES_OP.1)(a, b)
}

/// Cached O(1) ASCII check for strings (memoizes long strings after one scan).
#[must_use]
pub fn str_is_ascii(s: &Arc<str>) -> bool {
    crate::coll::str_ascii(s)
}

/// [`apply_inline`] for three args (a ternary op, or a 3-arg fold or
/// chain).
#[must_use]
pub fn apply3(id: u32, a: &Value, b: &Value, c: &Value) -> Option<Value> {
    use crate::vm_ops::{APPLY3, APPLYN};
    if id != u32::MAX && (id & TAG_MASK) == NARY_OP {
        let (_, op, chain) = APPLYN.get((id & !NARY_OP) as usize)?;
        return nary(*op, *chain, &[a, b, c]);
    }
    APPLY3.get(id as usize)?.1(a, b, c)
}

/// A fold of binary `op` over `args`, or (`chain`) a comparison chain
/// stopping at the first false pair (like `clojure.core/<`).
fn nary(op: crate::vm_ops::Apply2, chain: bool, args: &[&Value]) -> Option<Value> {
    let (first, rest) = args.split_first()?;
    if chain {
        let mut prev = *first;
        for x in rest {
            if !op(prev, x)?.is_truthy() {
                return Some(Value::Bool(false));
            }
            prev = x;
        }
        return Some(Value::Bool(true));
    }
    rest.iter().try_fold((*first).clone(), |acc, x| op(&acc, x))
}

// ---- extent forms over compiled body fns --------------------------------

fn body_of(body: &Value) -> impl Fn(&mut Evaluator) -> Step + '_ {
    move |ev: &mut Evaluator| ev.call(body, &[])
}

/// `with-redefs`.
pub fn with_redefs_call(eval: &mut Evaluator, vals: Vec<(Var, Value)>, body: &Value) -> Step {
    eval.with_redefs_core(vals, &body_of(body))
}

/// `with-bindings*`.
pub fn with_bindings_call(eval: &mut Evaluator, map: &Value, body: &Value) -> Step {
    eval.with_bindings_core(map, &body_of(body))
}

/// `time`.
pub fn time_call(eval: &mut Evaluator, body: &Value) -> Step {
    eval.time_core(&body_of(body))
}

/// `with-out-str` (binding `out`, `*out*` as written).
pub fn with_out_str_call(eval: &mut Evaluator, out: &Symbol, body: &Value) -> Step {
    eval.with_out_str_core(out, &body_of(body))
}

/// `with-in-str`.
pub fn with_in_str_call(eval: &mut Evaluator, input: &Symbol, s: &Value, body: &Value) -> Step {
    eval.with_in_str_core(input, s, &body_of(body))
}

/// `locking`.
pub fn locking_call(eval: &mut Evaluator, target: &Value, body: &Value) -> Step {
    eval.locking_core(target, &body_of(body))
}

/// `sync` (`dosync`): the body re-runs on retry.
pub fn sync_call(eval: &mut Evaluator, body: &Value) -> Step {
    crate::stm::eval_sync(eval, &body_of(body))
}

/// `io!`.
pub fn io_call(eval: &mut Evaluator, body: &Value) -> Step {
    crate::stm::eval_io(eval, &body_of(body))
}

/// `lazy-seq` over a compiled zero-arg body fn.
#[must_use]
pub fn lazy_seq(f: Value) -> Value {
    crate::lazy::call0_value(f)
}

/// `delay` over a compiled zero-arg body fn.
#[must_use]
pub fn delay(f: Value) -> Value {
    Value::Delay(clj_value::Delay::new(f))
}

/// `bound-fn`: `f` re-pushes the creation-time thread bindings around
/// every call.
pub fn bound_fn(eval: &mut Evaluator, f: Value) -> Value {
    let conveyed = eval.bindings.clone();
    let display: Arc<str> = Arc::from("clojure.core/bound-fn");
    let func = Arc::new(move |ev: &mut Evaluator, args: &[Value]| {
        let depth = ev.bindings_depth();
        let frame: Vec<(Var, Value)> = conveyed
            .pairs()
            .map(|(v, x)| (v.clone(), x.clone()))
            .collect();
        ev.push_frame(frame);
        let out = ev.call(&f, args);
        ev.pop_bindings(depth);
        out.map_err(signal_to_native)
    });
    Value::Extension(Arc::new(crate::ext::NativeFn {
        id: eval.next_id(),
        display,
        func,
        owned: None,
        meta: None,
        fast: crate::ext::FastArith::None,
    }))
}

/// `defmacro` over a compiled macro fn.
#[allow(clippy::too_many_arguments)]
pub fn defmacro(
    eval: &mut Evaluator,
    sym: &Symbol,
    doc: Option<&str>,
    attr: Option<&Map>,
    arglists: &[Value],
    line: u32,
    col: u32,
    f: Value,
) -> Step {
    eval.defmacro_with(sym, doc, attr, arglists, Span::new(line, col), f)
}

/// A failed `assert` (`message` already evaluated).
pub fn assert_failed(eval: &mut Evaluator, message: Option<Value>, form: &str) -> Signal {
    eval.assert_failed(message, form)
}

/// Fused `(assoc! t1 k (conj (nth t2 k) x))` in place when `t1` and
/// `t2` alias the same live transient vector and `k` is in bounds.
pub fn assoc_conj(t1: &Value, t2: &Value, k: &Value, x: &mut Value, take_x: bool) -> bool {
    let (Value::Transient(a), Value::Transient(b), Value::Int(n)) = (t1, t2, k) else {
        return false;
    };
    if !a.same_cell(b) {
        return false;
    }
    let Ok(i) = usize::try_from(*n) else {
        return false;
    };
    if take_x {
        crate::vm_ops::assoc_conj_in_cell(a, i, x)
    } else {
        let mut xv = x.clone();
        crate::vm_ops::assoc_conj_in_cell(a, i, &mut xv)
    }
}

/// Effect-only `(conj! coll x)` in place on a live transient vector.
pub fn conj_bang_fx(a: &Value, b: &Value) -> bool {
    crate::vm_ops::ConjBang::apply_fx(a, b)
}

type OwnFn = fn(&mut [Value]) -> Option<Value>;

const OWN_BIT: u32 = 1 << 31;

const OWN_OPS: &[(&str, usize, OwnFn)] = &[
    ("pop", 1, <crate::vm_ops::Pop as crate::vm_ops::Upd>::apply),
    (
        "conj",
        2,
        <crate::vm_ops::Conj as crate::vm_ops::Upd>::apply,
    ),
    (
        "assoc",
        3,
        <crate::vm_ops::Assoc as crate::vm_ops::Upd>::apply,
    ),
    (
        "next",
        1,
        <crate::vm_ops::Next as crate::vm_ops::Upd>::apply,
    ),
    (
        "rest",
        1,
        <crate::vm_ops::Rest as crate::vm_ops::Upd>::apply,
    ),
];

/// Opcode for [`apply_own`], matching core fns whose in-place `Upd`
/// fast path consumes owned operands.
#[must_use]
pub fn own_op(name: &str, arity: usize) -> Option<u32> {
    OWN_OPS
        .iter()
        .position(|(n, a, _)| *n == name && *a == arity)
        .map(|i| OWN_BIT | (i as u32))
}

/// Apply an owned-operand fast op (`None`: fall back; `args` remain
/// valid on `None`).
pub fn apply_own(id: u32, args: &mut [Value]) -> Option<Value> {
    if id == u32::MAX || (id & OWN_BIT) == 0 {
        return None;
    }
    OWN_OPS
        .get((id & !OWN_BIT) as usize)
        .and_then(|(_, _, f)| f(args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_symbols_match_their_read_back() {
        let kw = |n: &str| Value::Keyword(Keyword::new(None, n));
        for (ns, name, kvs, text) in [
            (
                Some("user"),
                "f",
                vec![kw("private"), Value::Bool(true)],
                "^{:private true} user/f",
            ),
            (
                None,
                "g",
                vec![
                    kw("tag"),
                    Value::Symbol(Symbol::new(None, "String")),
                    kw("dynamic"),
                    Value::Bool(true),
                    kw("doc"),
                    Value::Str(std::sync::Arc::from("d")),
                    kw("n"),
                    Value::Int(3),
                ],
                "^{:tag String :dynamic true :doc \"d\" :n 3} g",
            ),
        ] {
            let built = meta_symbol(ns, name, &kvs);
            let read = read_constant(text).expect("reads");
            assert_eq!(print_constant(&built), print_constant(&read));
            assert_eq!(built.meta(), read.meta());
        }
    }

    #[test]
    fn const_colls_match_the_reader() {
        let kw = |n: &str| Value::Keyword(Keyword::new(None, n));
        let sym = |n: &str| Value::Symbol(Symbol::new(None, n));
        let vec_of = |xs: Vec<Value>| const_coll(1, &xs);
        for (kind, items, text) in [
            (0, vec![], "()"),
            (
                0,
                vec![
                    vec_of(vec![sym("x")]),
                    vec_of(vec![sym("x"), sym("&"), sym("y")]),
                ],
                "([x] [x & y])",
            ),
            (
                1,
                vec![Value::Str(std::sync::Arc::from("java.lang.Exception"))],
                "[\"java.lang.Exception\"]",
            ),
            (
                2,
                vec![
                    kw("added"),
                    Value::Str(std::sync::Arc::from("1.0")),
                    kw("n"),
                    Value::Int(2),
                ],
                "{:added \"1.0\", :n 2}",
            ),
            (3, vec![kw("a"), Value::Int(1)], "#{:a 1}"),
        ] {
            let built = const_coll(kind, &items);
            let read = read_constant(text).expect("reads");
            assert_eq!(built, read);
            // Read lists carry their read position; built ones carry none.
            let shown = |v: &Value| print_constant(&v.clone().with_meta(None));
            assert_eq!(shown(&built), shown(&read));
            assert!(built.meta().is_none());
        }
    }

    #[test]
    fn only_throws_carry_a_throwable() {
        assert!(thrown_of(&Signal::Recur(Vec::new())).is_none());
    }

    #[test]
    fn assoc_conj_and_own_ops_handle_hits_and_misses() {
        let tv = Value::Transient(clj_value::Transient::new(clj_value::TransientKind::Vector(
            clj_value::TransientVector::from_vector_owned(Vector::from_vec(vec![
                Value::Vector(Vector::from_vec(vec![Value::Int(1)])),
                Value::Int(42),
            ])),
        )));
        let tv2 = Value::Transient(clj_value::Transient::new(clj_value::TransientKind::Vector(
            clj_value::TransientVector::new(),
        )));
        let mut x = Value::Int(9);
        assert!(assoc_conj(&tv, &tv, &Value::Int(0), &mut x, true));
        assert!(matches!(x, Value::Nil));
        let mut x2 = Value::Int(8);
        assert!(assoc_conj(&tv, &tv, &Value::Int(0), &mut x2, false));
        assert_eq!(x2, Value::Int(8));
        // Non-conjable element misses and restores `x`.
        let mut x3 = Value::Int(7);
        assert!(!assoc_conj(&tv, &tv, &Value::Int(1), &mut x3, true));
        assert_eq!(x3, Value::Int(7));
        // Mismatched transients / bad key / non-transient miss cleanly.
        assert!(!assoc_conj(&tv, &tv2, &Value::Int(0), &mut x3, false));
        assert!(!assoc_conj(&tv, &tv, &Value::Int(-1), &mut x3, false));
        assert!(!assoc_conj(
            &Value::Nil,
            &tv,
            &Value::Int(0),
            &mut x3,
            false
        ));
        assert!(conj_bang_fx(&tv, &Value::Int(99)));
        assert!(!conj_bang_fx(&Value::Nil, &Value::Int(99)));

        let pop_id = own_op("pop", 1).expect("pop op");
        let mut pop_args = [Value::Vector(Vector::from_vec(vec![Value::Int(1)]))];
        assert!(apply_own(pop_id, &mut pop_args).is_some());
        let mut empty_pop = [Value::Vector(Vector::new_sync())];
        assert!(apply_own(pop_id, &mut empty_pop).is_none());
        assert!(own_op("unknown", 2).is_none());
        let add_id = inline_op("+", 2).expect("+");
        assert_eq!(
            apply_inline(add_id, &[&Value::Int(2), &Value::Int(3)]),
            Some(Value::Int(5))
        );
        let assoc_id = own_op("assoc", 3).expect("assoc op");
        assert_eq!(
            apply_inline(assoc_id, &[&Value::Int(2), &Value::Int(3), &Value::Int(4)]),
            None
        );
        assert_eq!(
            apply3(assoc_id, &Value::Int(2), &Value::Int(3), &Value::Int(4)),
            None
        );
        let mul3_id = inline_op("*", 3).expect("*/3");
        let mut mul_args = [
            Value::Vector(Vector::from_vec(vec![Value::Int(1)])),
            Value::Int(0),
            Value::Int(9),
        ];
        assert_eq!(apply_own(mul3_id, &mut mul_args), None);
        let mut assoc_miss = [Value::Int(2), Value::Int(3), Value::Int(4)];
        assert_eq!(apply_own(assoc_id, &mut assoc_miss), None);

        let re_id = inline_op("re-matches", 2).expect("re-matches");
        assert!(is_re_op(re_id));
        assert!(!is_re_op(add_id));
        let re_pattern = crate::re::RE_NATIVES
            .iter()
            .find(|&&(k, _)| k == "re-pattern")
            .expect("re-pattern")
            .1;
        let pat = re_pattern(
            &mut Evaluator::new_bare("user"),
            &[Value::Str(Arc::from("a+"))],
        )
        .expect("pat");
        let aa = Value::Str(Arc::from("aa"));
        assert_eq!(apply2_no_re(re_id, &pat, &aa), None);
        assert_eq!(apply2(re_id, &pat, &aa), Some(aa.clone()));
        assert_eq!(
            apply2(add_id, &Value::Int(2), &Value::Int(3)),
            Some(Value::Int(5))
        );
        assert_eq!(apply_re_matches(&pat, &aa), Some(aa));
    }
}
