// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! [`Expr`] plus the [`Analyzer`]: reader [`Value`]s become typed
//! expressions, enforcing every §E2–E6/E11 shape rule with
//! oracle-verbatim classes, messages, spans, and wrappers.
//!
//! `eval` macroexpands the nine FX4 heads (`let`/`loop`/`fn`/
//! `binding`/`letfn`/`with-redefs`/`with-bindings`/`bound-fn`/
//! `locking`) before analyze. Direct [`Analyzer`] calls still
//! parse those unexpanded specials (spec-check then body).

use std::collections::HashMap;
use std::sync::Arc;

use clj_reader::Span;
use clj_value::{
    ClassVal, Keyword, List, Map, PrintOpts, Symbol, Value, Var, Vector, map_entry, new_map,
    pr_to_string, print_to_string,
};

use crate::core::cce_message;
use crate::error::{CompileError, NativeError, Signal, Verb, jvm_class_of, span_of};
use crate::eval::Evaluator;

/// A `catch` clause: resolved class name, local slot, body.
#[derive(Clone, Debug)]
pub struct Catch {
    pub class: String,
    pub slot: usize,
    pub body: Vec<Expr>,
}

/// One `:keys`/`:strs`/`:syms` entry (`!` = required).
#[derive(Clone, Debug)]
pub struct KeyBinding {
    pub local: String,
    pub key: Value,
    pub required: bool,
}

/// A map-destructure pattern (`destmap*`, incl. the 1.13 directives).
#[derive(Clone, Debug)]
pub struct MapPattern {
    /// `(local-pattern, lookup-key, or-key, required?)` entries
    /// (`:req!` locals bind required, like `:keys!` elements). The
    /// lookup key is unquoted (map keys are values); the or-key is
    /// the key form as written (`:or` matches forms, #158).
    pub entries: Vec<(Pattern, Value, Value, bool)>,
    /// `:keys`/`:strs`/`:syms` (and `!`) bindings.
    pub keys: Vec<KeyBinding>,
    /// Post-`&` key-list keys (selection only, never bound).
    pub extra_keys: Vec<Value>,
    /// `:as` (whole post-conversion map).
    pub as_sym: Option<String>,
    /// `:or` (key as written → default expression, eager).
    pub or: Vec<(Value, Expr)>,
    /// 1.13 outlets (`:defaults` is silently ignored).
    pub select: Option<String>,
    pub all: Option<String>,
    pub excess: Option<String>,
    pub missing: Option<String>,
}

/// A sequential-destructure pattern (`destvec*`).
#[derive(Clone, Debug)]
pub struct VecPattern {
    pub fixed: Vec<Pattern>,
    pub rest: Option<Box<Pattern>>,
    pub as_sym: Option<String>,
}

/// A binding pattern: plain symbol, sequential, or map.
#[derive(Clone, Debug)]
pub enum Pattern {
    Symbol(String),
    Vec(VecPattern),
    Map(MapPattern),
}

/// One `fn` arity: parameters plus the desugared body (leading
/// pre-asserts, then the body — or the post `%` wrapper).
#[derive(Clone, Debug)]
pub struct Arity {
    pub fixed: Vec<Pattern>,
    pub rest: Option<Pattern>,
    pub body: Vec<Expr>,
}

/// An analyzed `fn`: display name plus arities.
#[derive(Clone, Debug)]
pub struct FunExpr {
    /// `user/eval7/fn--8` or `user/eval7/name--9`.
    pub display: String,
    pub arities: Vec<Arity>,
}

/// One `ns` clause (raw unevaluated items, like the macro's
/// quoted args).
#[derive(Clone, Debug)]
pub enum NsClause {
    ReferClojure(Vec<Value>),
    Require(Vec<Value>),
    Use(Vec<Value>),
}

/// A `case*` dispatch mode: `:int` funnels numbers through
/// `intValue` (non-numbers miss, then an `equiv` post-check);
/// `:hash-equiv`/`:hash-identity` hash and post-check (`identity`
/// is `equiv` here — keywords are not interned, and equal keywords
/// must match either way).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseTest {
    Int,
    HashEquiv,
    HashIdentity,
}

/// An analyzed expression.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Self-evaluating (atoms, quotes' values, `()`).
    Literal(Value),
    /// Collection literals evaluate their elements (carrying the
    /// reader form's metadata, like the JVM).
    VecLit(Vec<Expr>, Option<Map>),
    MapLit(Vec<(Expr, Expr)>, Option<Map>),
    SetLit(Vec<Expr>, Option<Map>),
    /// A resolved global (the captured `Var`: analysis resolves
    /// the id once, like the JVM's compiled `Var` reference, so
    /// evaluation never re-resolves or resurrects — #79).
    Global(Var),
    /// A local slot index.
    Local(usize),
    /// `(def sym doc? init?)` (created in the ns at analyze time).
    Def {
        sym: Symbol,
        doc: Option<String>,
        init: Option<Box<Expr>>,
        span: Span,
    },
    If {
        test: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
    },
    Do(Vec<Expr>),
    Let {
        bindings: Vec<(Pattern, Expr)>,
        body: Vec<Expr>,
    },
    Loop {
        bindings: Vec<(Pattern, Expr)>,
        body: Vec<Expr>,
    },
    LetFn {
        fns: Vec<(String, FunExpr)>,
        body: Vec<Expr>,
    },
    Fn(FunExpr),
    Recur(Vec<Expr>),
    Quote(Value),
    /// `(var sym)` (the captured `Var`, resolved at analyze time).
    VarRef(Var),
    Throw(Box<Expr>),
    /// A freshly constructed throwable (pre/post `assert` failures;
    /// the evaluator stamps the identity).
    NewThrowable {
        class: clj_value::ThrowClass,
        message: String,
    },
    Try {
        body: Vec<Expr>,
        catches: Vec<Catch>,
        finally_body: Vec<Expr>,
    },
    /// `(set! global init)` (locals rejected at analyze time;
    /// the captured `Var`).
    Set {
        var: Var,
        val: Box<Expr>,
    },
    /// `(binding [var init …] body…)` (S4 minimal surface form;
    /// pairs carry captured `Var`s).
    Binding {
        pairs: Vec<(Var, Expr)>,
        body: Vec<Expr>,
    },
    /// `(with-redefs [var init …] body…)` (S5B parallel root
    /// swap; pairs carry captured `Var`s, like `var`).
    WithRedefs {
        pairs: Vec<(Var, Expr)>,
        body: Vec<Expr>,
    },
    /// `(with-bindings map-expr body…)` (S5B).
    WithBindings {
        map: Box<Expr>,
        body: Vec<Expr>,
    },
    /// `(bound-fn name? argv body…)` (S5B: like `fn`, snapshots
    /// the thread bindings at creation).
    BoundFn {
        name: Option<String>,
        fun: FunExpr,
    },
    /// `(defmacro name doc? attr? fdecl)` (S5C: like `def` of a
    /// `fn` with `&form`/`&env` prepended, plus `:macro` and
    /// `:arglists` meta).
    DefMacro {
        sym: Symbol,
        doc: Option<String>,
        attr: Option<Map>,
        arglists: Vec<Value>,
        fun: FunExpr,
        span: Span,
    },
    /// `(declare name…)` (S5C: per-name `def`-with-no-init plus
    /// `:declared`, returning the last var).
    Declare {
        syms: Vec<Symbol>,
        span: Span,
    },
    /// `(ns name doc? attr? clause…)` (S5C: `in-ns`, `resetMeta`,
    /// `refer-clojure`, `:require`/`:use` clauses, mark loaded).
    Ns {
        name: String,
        doc: Option<String>,
        attr: Option<Map>,
        clauses: Vec<NsClause>,
    },
    /// `(assert x)` / `(assert x message)` (S5D analyzer builtin):
    /// `*assert*` was truthy at analysis (else analysis emits
    /// `Literal(Nil)` without touching `test`). `form` is the
    /// `pr-str` of the test form.
    Assert {
        test: Box<Expr>,
        message: Option<Box<Expr>>,
        form: String,
    },
    /// `(time expr)` (S5D analyzer builtin).
    Time {
        expr: Box<Expr>,
    },
    /// `(with-out-str body…)` (S5D analyzer builtin): `out` is the
    /// `*out*` binding target (resolved like `binding`, so user
    /// shadows apply).
    WithOutStr {
        out: Symbol,
        body: Vec<Expr>,
    },
    /// `(with-in-str s body…)` (S-instream analyzer builtin):
    /// `input` is the `*in*` binding target (resolved like
    /// `binding`, so user shadows apply), `s` the string
    /// expression, and the body returns its value.
    WithInStr {
        input: Symbol,
        s: Box<Expr>,
        body: Vec<Expr>,
    },
    /// `(lazy-seq* body…)` (S5D analyzer form — the JVM has no such
    /// var; `lazy-seq` expands to this).
    LazySeqStar {
        body: Vec<Expr>,
    },
    /// `(delay* body…)` (S6A analyzer form — the JVM has no such
    /// var; `delay` expands to this). `display` names the trapped
    /// zero-arg closure, as `fn` displays do.
    DelayStar {
        display: String,
        body: Vec<Expr>,
    },
    /// `(locking lockee body…)` (S6D analyzer form — the JVM's
    /// `locking` is a `core.clj` macro over `monitor-enter/-exit`,
    /// and the port's analyzer takes the lockee's per-object
    /// monitor directly (fence 3b stripes)).
    Locking {
        lockee: Box<Expr>,
        body: Vec<Expr>,
    },
    /// `(sync flags body…)` (S19 analyzer form — the flags form is
    /// dropped before analysis, never evaluated; the body joins or
    /// begins a transaction).
    Sync {
        body: Vec<Expr>,
    },
    /// `(io! body…)` (S19 analyzer form — the gate throws inside a
    /// transaction before the body evaluates).
    Io {
        body: Vec<Expr>,
    },
    /// `(case* ge shift mask default imap switch-type test-type
    /// skip?)` (S5D analyzer form, emitted by the `case` macro):
    /// `cases` maps dispatch keys to `(test, then)` (`test` is an
    /// analyzed expr, evaluated on a hash hit like the JVM
    /// `test.emit`); `skip` holds case ints whose `then` is already
    /// a collision-`condp` (no post-check).
    CaseStar {
        expr: Box<Expr>,
        shift: i32,
        mask: i32,
        default: Box<Expr>,
        cases: Vec<(i32, Expr, Expr)>,
        skip: Vec<i32>,
        test: CaseTest,
    },
    Invoke {
        head: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
}

/// JVM classes used by analyzer causes.
pub(crate) mod class {
    pub const STACK_OVERFLOW: &str = "java.lang.StackOverflowError";
    pub const RUNTIME: &str = "java.lang.RuntimeException";
    pub const ILLEGAL_ARG: &str = "java.lang.IllegalArgumentException";
    pub const ILLEGAL_STATE: &str = "java.lang.IllegalStateException";
    pub const UNSUPPORTED: &str = "java.lang.UnsupportedOperationException";
    pub const NULL_POINTER: &str = "java.lang.NullPointerException";
    pub const CLASS_CAST: &str = "java.lang.ClassCastException";
    pub const EX_INFO: &str = "clojure.lang.ExceptionInfo";
    pub const EXCEPTION: &str = "java.lang.Exception";
    pub const CLASS_NOT_FOUND: &str = "java.lang.ClassNotFoundException";
}

/// `java.lang` short names that resolve as `catch` classes: the
/// builtin throwable shorts, every other `java.lang` throwable
/// (bulk-verified on the oracle, probe32 — dead handlers in S4 since
/// nothing throws them), and the real-but-unthrowable shorts
/// (accepted, never match — the JVM `VerifyError`s those instead,
/// §E6 documents the divergence).
const JAVA_LANG_CLASSES: &[(&str, &str)] = &[
    ("Throwable", "java.lang.Throwable"),
    ("Exception", "java.lang.Exception"),
    ("Error", "java.lang.Error"),
    ("RuntimeException", "java.lang.RuntimeException"),
    (
        "IllegalArgumentException",
        "java.lang.IllegalArgumentException",
    ),
    ("IllegalStateException", "java.lang.IllegalStateException"),
    (
        "UnsupportedOperationException",
        "java.lang.UnsupportedOperationException",
    ),
    ("AssertionError", "java.lang.AssertionError"),
    ("NullPointerException", "java.lang.NullPointerException"),
    ("ClassCastException", "java.lang.ClassCastException"),
    (
        "IndexOutOfBoundsException",
        "java.lang.IndexOutOfBoundsException",
    ),
    ("ArithmeticException", "java.lang.ArithmeticException"),
    ("InterruptedException", "java.lang.InterruptedException"),
    ("NumberFormatException", "java.lang.NumberFormatException"),
    (
        "IllegalMonitorStateException",
        "java.lang.IllegalMonitorStateException",
    ),
    (
        "IllegalThreadStateException",
        "java.lang.IllegalThreadStateException",
    ),
    (
        "ArrayIndexOutOfBoundsException",
        "java.lang.ArrayIndexOutOfBoundsException",
    ),
    (
        "StringIndexOutOfBoundsException",
        "java.lang.StringIndexOutOfBoundsException",
    ),
    (
        "NegativeArraySizeException",
        "java.lang.NegativeArraySizeException",
    ),
    ("ArrayStoreException", "java.lang.ArrayStoreException"),
    ("NoSuchMethodError", "java.lang.NoSuchMethodError"),
    ("NoSuchFieldError", "java.lang.NoSuchFieldError"),
    ("AbstractMethodError", "java.lang.AbstractMethodError"),
    ("UnsatisfiedLinkError", "java.lang.UnsatisfiedLinkError"),
    (
        "ExceptionInInitializerError",
        "java.lang.ExceptionInInitializerError",
    ),
    ("StackOverflowError", "java.lang.StackOverflowError"),
    ("OutOfMemoryError", "java.lang.OutOfMemoryError"),
    ("ClassFormatError", "java.lang.ClassFormatError"),
    ("NoClassDefFoundError", "java.lang.NoClassDefFoundError"),
    ("VerifyError", "java.lang.VerifyError"),
    ("LinkageError", "java.lang.LinkageError"),
    ("ClassCircularityError", "java.lang.ClassCircularityError"),
    (
        "IncompatibleClassChangeError",
        "java.lang.IncompatibleClassChangeError",
    ),
    ("InstantiationError", "java.lang.InstantiationError"),
    ("InternalError", "java.lang.InternalError"),
    ("UnknownError", "java.lang.UnknownError"),
    ("ThreadDeath", "java.lang.ThreadDeath"),
    (
        "ReflectiveOperationException",
        "java.lang.ReflectiveOperationException",
    ),
    ("ClassNotFoundException", "java.lang.ClassNotFoundException"),
    ("IllegalAccessException", "java.lang.IllegalAccessException"),
    ("InstantiationException", "java.lang.InstantiationException"),
    ("NoSuchMethodException", "java.lang.NoSuchMethodException"),
    ("NoSuchFieldException", "java.lang.NoSuchFieldException"),
    (
        "CloneNotSupportedException",
        "java.lang.CloneNotSupportedException",
    ),
    ("SecurityException", "java.lang.SecurityException"),
    (
        "TypeNotPresentException",
        "java.lang.TypeNotPresentException",
    ),
    (
        "EnumConstantNotPresentException",
        "java.lang.EnumConstantNotPresentException",
    ),
    ("Object", "java.lang.Object"),
    ("String", "java.lang.String"),
    ("Number", "java.lang.Number"),
    ("Boolean", "java.lang.Boolean"),
    ("Character", "java.lang.Character"),
    ("Byte", "java.lang.Byte"),
    ("Short", "java.lang.Short"),
    ("Integer", "java.lang.Integer"),
    ("Long", "java.lang.Long"),
    ("Float", "java.lang.Float"),
    ("Double", "java.lang.Double"),
    ("Void", "java.lang.Void"),
    ("Class", "java.lang.Class"),
    ("Thread", "java.lang.Thread"),
];

/// The span a form binds: only lists rebind it — vectors, maps,
/// sets, and atoms inherit the fallback, like the JVM (which binds
/// `LINE`/`COLUMN` for seq forms only, probe31 `span-*`).
fn analyze_span(form: &Value, fallback: Span) -> Span {
    match form {
        Value::List(_) => span_of(form, fallback),
        _ => fallback,
    }
}

/// A `java.lang` short name's JVM binary name, if it resolves.
fn resolve_java_lang(name: &str) -> Option<&'static str> {
    JAVA_LANG_CLASSES
        .iter()
        .find(|(short, _)| *short == name)
        .map(|(_, jvm)| *jvm)
}

/// Curated throwable FQNs outside `java.lang` (each oracle-verified,
/// probe32 `catch-fqn-*`): the port has no class loader, so only
/// these resolve; anything else dotted is `ClassNotFoundException`
/// (§E6 fences the exotic remainder).
fn is_curated_fqn(fqn: &str) -> bool {
    matches!(
        fqn,
        "java.io.IOException"
            | "java.io.FileNotFoundException"
            | "java.io.EOFException"
            | "java.io.UncheckedIOException"
            | "java.util.NoSuchElementException"
            | "java.util.concurrent.ExecutionException"
            | "java.util.concurrent.CancellationException"
            | "java.util.concurrent.RejectedExecutionException"
            | "java.lang.reflect.InvocationTargetException"
            | "clojure.lang.Compiler$CompilerException"
            | "clojure.lang.LispReader$ReaderException"
    )
}

/// Clojure `str` of a value (`nil` → `""`): the `Unsupported binding
/// form` formatter.
fn str_of(v: &Value) -> String {
    if matches!(v, Value::Nil) {
        String::new()
    } else {
        print_to_string(v)
    }
}

/// Java `String.valueOf`: `nil` → `"null"`, else `str`. The `got:`
/// and `classname:` formatter.
fn value_of(v: &Value) -> String {
    if matches!(v, Value::Nil) {
        "null".to_owned()
    } else {
        str_of(v)
    }
}

/// `pr` of a value (forms, assert conditions, `:or` key sets print
/// readably).
fn pr_of(v: &Value) -> String {
    pr_to_string(v, &PrintOpts::pr())
}

/// A special-form operator (parsed once, at the dispatch guard).
#[derive(Clone, Copy, Debug)]
enum SpecialOp {
    Def,
    If,
    Do,
    Let,
    LetStar,
    Loop,
    LoopStar,
    LetFn,
    Fn,
    FnStar,
    Recur,
    Quote,
    Var,
    Throw,
    Try,
    Set,
    Binding,
    WithRedefs,
    WithBindings,
    BoundFn,
    DefMacro,
    Declare,
    Ns,
    Assert,
    Time,
    WithOutStr,
    WithInStr,
    LazySeqStar,
    DelayStar,
    CaseStar,
    Locking,
    Sync,
    Io,
}

/// The special-form table: name → op (one row per form; names
/// are unique so the linear scan is a pure dispatch).
const SPECIAL_OPS: &[(&str, SpecialOp)] = &[
    ("def", SpecialOp::Def),
    ("if", SpecialOp::If),
    ("do", SpecialOp::Do),
    ("let", SpecialOp::Let),
    ("let*", SpecialOp::LetStar),
    ("loop", SpecialOp::Loop),
    ("loop*", SpecialOp::LoopStar),
    ("letfn", SpecialOp::LetFn),
    ("fn", SpecialOp::Fn),
    ("fn*", SpecialOp::FnStar),
    ("recur", SpecialOp::Recur),
    ("quote", SpecialOp::Quote),
    ("var", SpecialOp::Var),
    ("throw", SpecialOp::Throw),
    ("try", SpecialOp::Try),
    ("set!", SpecialOp::Set),
    ("binding", SpecialOp::Binding),
    ("with-redefs", SpecialOp::WithRedefs),
    ("with-bindings", SpecialOp::WithBindings),
    ("bound-fn", SpecialOp::BoundFn),
    ("defmacro", SpecialOp::DefMacro),
    ("declare", SpecialOp::Declare),
    ("ns", SpecialOp::Ns),
    ("assert", SpecialOp::Assert),
    ("time", SpecialOp::Time),
    ("with-out-str", SpecialOp::WithOutStr),
    ("with-in-str", SpecialOp::WithInStr),
    ("lazy-seq*", SpecialOp::LazySeqStar),
    ("delay*", SpecialOp::DelayStar),
    ("case*", SpecialOp::CaseStar),
    ("locking", SpecialOp::Locking),
    ("sync", SpecialOp::Sync),
    ("io!", SpecialOp::Io),
];

/// Parse a special-form operator (the single table behind
/// [`is_special`] and dispatch).
fn special_op(name: &str) -> Option<SpecialOp> {
    SPECIAL_OPS
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, op)| *op)
}

/// Whether a symbol names a special form (unqualified only).
pub(crate) fn is_special(name: &str) -> bool {
    special_op(name).is_some()
}

/// The macro-implemented specials: a local or global binding
/// with one of these names shadows the macro (real specials never
/// shadow, probe31 `shadow-*`). `case*` is not shadowable (the JVM
/// hardcodes it); `lazy-seq*` and `delay*` are (the JVM has no such
/// forms, so a user definition wins there), as are `locking`,
/// `sync`, and `io!` (JVM `core.clj` macros).
fn is_shadowable_macro(name: &str) -> bool {
    matches!(
        name,
        "let"
            | "loop"
            | "letfn"
            | "fn"
            | "binding"
            | "with-redefs"
            | "with-bindings"
            | "bound-fn"
            | "defmacro"
            | "declare"
            | "ns"
            | "assert"
            | "time"
            | "with-out-str"
            | "with-in-str"
            | "lazy-seq*"
            | "delay*"
            | "locking"
            | "sync"
            | "io!"
    )
}

/// The analyzer: reader values become [`Expr`]s. Holds the local
/// scope, the `recur` target, the `try` fence, and the live
/// evaluator (namespace resolve checks plus `def` interning like
/// the JVM, macro calls, and the `*ns*`-synced current name).
pub struct Analyzer<'a> {
    source: Arc<str>,
    eval: &'a mut Evaluator,
    locals: Vec<String>,
    recur: Option<usize>,
    no_recur: bool,
    eval_seq: u64,
    /// Current analyze nesting (siblings share levels via inc/dec).
    depth: usize,
    /// Plan-path recursion budget (`None` on the run path, which
    /// keeps today's unbounded behavior).
    limit: Option<usize>,
}

impl<'a> Analyzer<'a> {
    /// Analyze top-level forms from `source` against the evaluator
    /// (`eval_seq` numbers the form for anonymous-`fn` display
    /// names).
    pub fn new(source: &Arc<str>, eval: &'a mut Evaluator, eval_seq: u64) -> Self {
        let limit = eval.analyze_depth_limit;
        Self {
            source: Arc::clone(source),
            eval,
            locals: Vec::new(),
            recur: None,
            no_recur: false,
            eval_seq,
            depth: 0,
            limit,
        }
    }

    /// The `*ns*`-synced current namespace name (every resolve
    /// check and installation reads this, never the raw registry
    /// current).
    fn current_ns(&self) -> String {
        self.eval.current_ns_name()
    }

    fn err(
        &self,
        class: &'static str,
        message: String,
        span: Span,
        verb: Verb,
        sym: Option<String>,
    ) -> CompileError {
        CompileError::new(class, message, span, &self.source, verb, sym)
    }

    /// Analyze one form (`fallback` = enclosing span; `tail` = tail
    /// position for `recur`). On the plan path (`limit` set) the
    /// nesting trips before the machine stack does: analyze-time
    /// macroexpansion can deepen forms the rescan measured
    /// shallow, so the budget is enforced here, at every level.
    pub fn analyze(
        &mut self,
        form: &Value,
        fallback: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        // Refuse to descend past the limit (`depth` counts live
        // ancestors: a 40-deep nest analyzes its leaf at depth 40,
        // so `> limit` trips exactly where `value_depth` would).
        if matches!(self.limit, Some(limit) if self.depth > limit) {
            let span = analyze_span(form, fallback);
            return Err(self.err(
                class::STACK_OVERFLOW,
                "maximum analyzer depth exceeded".to_owned(),
                span,
                Verb::Compiling,
                None,
            ));
        }
        self.depth += 1;
        let out = self.analyze_inner(form, fallback, tail);
        self.depth -= 1;
        out
    }

    /// The analyze body (see [`Analyzer::analyze`] for the budget).
    fn analyze_inner(
        &mut self,
        form: &Value,
        fallback: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let span = analyze_span(form, fallback);
        match form {
            Value::Symbol(sym) => self.analyze_symbol(sym, span),
            Value::List(list) if !list.is_empty() => {
                let items: Vec<&Value> = list.iter().collect();
                self.analyze_list(form, &items, span, tail)
            }
            form if crate::lazy::is_seq_value(form) => self.analyze_seq_form(form, span, tail),
            Value::Vector(v) => self.analyze_vec_lit(v, span),
            // Entries in code (syntax-quote can embed them) analyze
            // like vector literals, answering plain vectors (Oracle
            // FX2 `eval-entry-class`).
            Value::MapEntry { key, val, .. } => {
                let tmp = clj_value::Vector::from_vec(vec![(**key).clone(), (**val).clone()]);
                self.analyze_vec_lit(&tmp, span)
            }
            Value::Map(m) => self.analyze_map_lit(m, span),
            Value::Set(s) => self.analyze_set_lit(s, span),
            // String literals canonicalize through the JVM
            // string-pool analog (fence 4a — inside the helper,
            // so this match gains no arm).
            _ => Ok(Expr::Literal(clj_value::intern_string_value(form))),
        }
    }

    /// Any other seq value (macro expansions built at runtime by
    /// `seq`/`concat` arrive as realized or pending seq extensions;
    /// the JVM analyzes every `ISeq` as a form).
    fn analyze_seq_form(
        &mut self,
        form: &Value,
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let items = match crate::lazy::collect_all(self.eval, form) {
            Ok(items) => items,
            Err(s) => return Err(form_force_err(&self.source, s, span)),
        };
        if items.is_empty() {
            // `EmptyExpr`: an empty form evaluates to itself.
            return Ok(Expr::Literal(form.clone()));
        }
        let refs: Vec<&Value> = items.iter().collect();
        self.analyze_list(form, &refs, span, tail)
    }

    /// A vector literal (elements analyze left to right).
    fn analyze_vec_lit(&mut self, v: &Vector, span: Span) -> Result<Expr, CompileError> {
        let mut out = Vec::with_capacity(v.len());
        for elem in v.iter() {
            out.push(self.analyze(elem, span, false)?);
        }
        Ok(Expr::VecLit(out, v.meta().cloned()))
    }

    /// A map literal (keys and values analyze left to right).
    fn analyze_map_lit(&mut self, m: &Map, span: Span) -> Result<Expr, CompileError> {
        let mut out = Vec::with_capacity(m.size());
        for (k, val) in m.iter() {
            out.push((
                self.analyze(k, span, false)?,
                self.analyze(val, span, false)?,
            ));
        }
        Ok(Expr::MapLit(out, m.meta().cloned()))
    }

    /// A set literal (elements analyze left to right).
    fn analyze_set_lit(&mut self, s: &clj_value::Set, span: Span) -> Result<Expr, CompileError> {
        let mut out = Vec::with_capacity(s.size());
        for elem in s.iter() {
            out.push(self.analyze(elem, span, false)?);
        }
        Ok(Expr::SetLit(out, s.meta().cloned()))
    }

    /// A body: every form non-tail except the last.
    fn analyze_body(
        &mut self,
        forms: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Vec<Expr>, CompileError> {
        let mut out = Vec::with_capacity(forms.len());
        for (i, form) in forms.iter().enumerate() {
            let last = i + 1 == forms.len();
            out.push(self.analyze(form, span, tail && last)?);
        }
        Ok(out)
    }

    /// A local slot by name (innermost wins).
    fn local_idx(&self, name: &str) -> Option<usize> {
        self.locals.iter().rposition(|l| l == name)
    }

    /// Resolve a symbol in value position.
    fn analyze_symbol(&mut self, sym: &Symbol, span: Span) -> Result<Expr, CompileError> {
        if let Some(idx) = self
            .local_idx(sym.name())
            .filter(|_| sym.namespace().is_none())
        {
            return Ok(Expr::Local(idx));
        }
        if let Some(empty) = crate::fx4_host::pam_empty(sym) {
            return Ok(Expr::Literal(empty));
        }
        let cur = self.current_ns();
        if let Some((target, id)) = self.eval.registry.resolve_target_and_id(&cur, sym) {
            // Value-position checks (`analyzeSymbol` order: private
            // first, then macro). Neither wrapper names an operator
            // (probed); the private message spells the symbol as
            // written without `#'`, the macro one the `#'` target.
            // (Invoke position has its own check in `analyze_list`.)
            let home = target.namespace().unwrap_or("");
            if home != cur && crate::macros::is_private_var(self.eval, id) {
                return Err(self.err(
                    class::ILLEGAL_STATE,
                    format!("var: {sym} is not public"),
                    span,
                    Verb::Compiling,
                    None,
                ));
            }
            if crate::macros::is_macro_var(self.eval, id) {
                return Err(self.err(
                    class::RUNTIME,
                    format!("Can't take value of a macro: #'{}", target),
                    span,
                    Verb::Compiling,
                    None,
                ));
            }
            return Ok(Expr::Global(Var { id, sym: target }));
        }
        // S16: imports resolve after vars (the class, not a var) —
        // operator position shares this path (#120: dotted bare
        // names resolve as classes or fail `CNFE`).
        if let Some(expr) = self.analyze_symbol_class(sym, span, &cur)? {
            return Ok(expr);
        }
        match sym.namespace() {
            // Aliases count as namespaces even when dangling
            // (probe34 `ns-alias-removed`); both miss shapes blame
            // the symbol as written (probe34 `ns-alias-novar`).
            // An unresolvable dotted namespace part is
            // `ClassNotFoundException` (oracle `removed-ns-val`),
            // unless it names a class — value-position static
            // members (method fns, field reads) are a later slice,
            // so those keep `No such namespace` for now.
            Some(ns) if self.eval.registry.namespace_target_in(&cur, ns).is_none() => {
                Err(self.dangling_alias_err(&cur, ns, span))
            }
            Some(_) => Err(self.err(
                class::RUNTIME,
                format!("No such var: {sym}"),
                span,
                Verb::Compiling,
                None,
            )),
            None => Err(self.err(
                class::RUNTIME,
                format!("Unable to resolve symbol: {sym} in this context"),
                span,
                Verb::Compiling,
                None,
            )),
        }
    }

    /// Error for a dangling alias / unresolvable dotted head
    /// (split from [`analyze_symbol`](Self::analyze_symbol) for
    /// the complexity budget): `ClassNotFoundException` when the
    /// head sits in class position with no static class behind
    /// it, else `No such namespace`.
    fn dangling_alias_err(&self, cur: &str, ns: &str, span: Span) -> CompileError {
        if crate::classemu::is_class_position(ns) && self.static_head_class(cur, ns).is_none() {
            self.err(
                class::CLASS_NOT_FOUND,
                ns.to_owned(),
                span,
                Verb::Compiling,
                None,
            )
        } else {
            self.err(
                class::RUNTIME,
                format!("No such namespace: {ns}"),
                span,
                Verb::Compiling,
                None,
            )
        }
    }

    /// Unqualified class-position resolve (split from
    /// [`analyze_symbol`](Self::analyze_symbol) for the complexity
    /// budget): imports, then dotted bare names as classes
    /// (`ClassNotFoundException` when missing — oracles
    /// `val-fq-String`, `bare-deftype`, `val-fq-missing`,
    /// `val-dotdot`). `None` falls through to the namespace match.
    fn analyze_symbol_class(
        &self,
        sym: &Symbol,
        span: Span,
        cur: &str,
    ) -> Result<Option<Expr>, CompileError> {
        if sym.namespace().is_some() {
            return Ok(None);
        }
        if let Some(class) = crate::classemu::import_in(self.eval, cur, sym.name()) {
            return Ok(Some(Expr::Literal(Value::Class(class))));
        }
        if !crate::classemu::is_class_position(sym.name()) {
            return Ok(None);
        }
        match crate::classemu::resolve_class_name(self.eval, sym.name()) {
            Some(class) => Ok(Some(Expr::Literal(Value::Class(class)))),
            None => Err(self.err(
                class::CLASS_NOT_FOUND,
                sym.name().to_owned(),
                span,
                Verb::Compiling,
                None,
            )),
        }
    }

    /// Analyze a non-empty list: special form, static-`nil` check,
    /// or invocation.
    fn analyze_list(
        &mut self,
        form: &Value,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let head = items[0];
        if let Some(expr) = self.try_head_intercept(head, items, span, tail)? {
            return Ok(expr);
        }
        // Macro expansion (`analyzeSeq` order: specials, then
        // macros): an expanded form re-analyzes from the top.
        if let Some(expanded) = self.try_expand_macro(form, head, &items[1..], span)? {
            let out = self.analyze(&expanded, span, tail);
            // B1: macro outputs are the one unbounded-depth class
            // (30k threading steps build 30k-deep output). Dropping
            // one recurses per level and can overflow _during
            // unwinding_ — after the budget already tripped. On the
            // plan path the build fails right after this either
            // way, so leak the output instead of dropping it; the
            // run path is untouched (no budget, same as before).
            if out.is_err() && self.limit.is_some() {
                std::mem::forget(expanded);
            }
            return out;
        }
        // Private cross-namespace invocation (the `isInline`
        // analog): the wrapper names the operator as written, the
        // message the `#'`-qualified target. Macros took the
        // expansion path above and never reach this.
        if let Value::Symbol(sym) = head {
            self.check_private_invoke(sym, span)?;
        }
        if matches!(head, Value::Nil) {
            return Err(self.err(
                class::ILLEGAL_ARG,
                format!("Can't call nil, form: {}", pr_of(form)),
                span,
                Verb::Compiling,
                None,
            ));
        }
        if let Value::Symbol(sym) = head {
            self.check_class_operator(sym, span)?;
        }
        let head_expr = self.analyze(head, span, false)?;
        let mut args = Vec::with_capacity(items.len() - 1);
        for arg in &items[1..] {
            args.push(self.analyze(arg, span, false)?);
        }
        Ok(Expr::Invoke {
            head: Box::new(head_expr),
            args,
            span,
        })
    }

    /// Head interceptions before macro expansion (split from
    /// [`analyze_list`] for the complexity budget): FX conditionals,
    /// then dot forms, then special/builtin heads.
    fn try_head_intercept(
        &mut self,
        head: &Value,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Option<Expr>, CompileError> {
        if let Some(expr) = self.try_fx4_host(items, span, tail)? {
            return Ok(Some(expr));
        }
        if let Some(expr) = self.try_dot_form(items, span)? {
            return Ok(Some(expr));
        }
        if let Some(expr) = self.try_static_call(items, span)? {
            return Ok(Some(expr));
        }
        self.intercept_head(head, items, span, tail)
    }

    /// Intercept dot heads (S-typechain fence): `(Type. args…)` is a
    /// port-type constructor, `(.-field x)` a positional field read,
    /// `(.method x args…)` a method-table call. Each rewrites to an
    /// `Expr::Invoke` of the hidden native with the name quoted, so
    /// no new `Expr` variant is needed; resolution (and its errors)
    /// happens at call time through ordinary vars. `..` is the
    /// threading macro (expanded later) and bare `.` keeps its
    /// unresolvable error, so neither is intercepted. `Some` when a
    /// form parsed, `None` to continue to specials.
    fn try_dot_form(&mut self, items: &[&Value], span: Span) -> Result<Option<Expr>, CompileError> {
        let Value::Symbol(sym) = items[0] else {
            return Ok(None);
        };
        let name = sym.name();
        if name.len() < 2 || name == ".." {
            return Ok(None);
        }
        if let Some(base) = name.strip_suffix('.') {
            let quoted = Symbol::new(sym.namespace(), base);
            return Ok(Some(self.dot_call("__new*", quoted, &items[1..], span)?));
        }
        if !name.starts_with('.') {
            return Ok(None);
        }
        if let Some(field) = name.strip_prefix(".-") {
            if !field.is_empty() && items.len() == 2 {
                let expr = self.dot_call_str("__field*", field, &items[1..], span)?;
                return Ok(Some(expr));
            }
            return Ok(None);
        }
        let expr = self.dot_call_str("__meth*", &name[1..], &items[1..], span)?;
        Ok(Some(expr))
    }

    /// Intercept static calls (`Class/forName`,
    /// `pkg.Cls/member`): qualified heads whose namespace part
    /// names no namespace or alias but resolves to an emulated
    /// class rewrite to `__static*` — namespaces and aliases always
    /// win (oracle `var-vs-class`, `alias-vs-static`), so macro and
    /// var heads never reach here. Known members validate arities
    /// at compile time (oracle `forName-0arg`, `forName-4arg`);
    /// unknown members lower anyway and miss at runtime (a
    /// documented phase narrowing — the JVM rejects those at
    /// compile time too, oracle `static-miss-0`). An unresolvable
    /// dotted namespace part is `ClassNotFoundException` (oracle
    /// `static-unknown-fq`, `removed-ns-call`); simple ones fall
    /// through to `No such namespace`. FX4-owned heads are exempt
    /// (their shapes and misses are the builtin's).
    fn try_static_call(
        &mut self,
        items: &[&Value],
        span: Span,
    ) -> Result<Option<Expr>, CompileError> {
        let Value::Symbol(sym) = items[0] else {
            return Ok(None);
        };
        let Some(ns_part) = sym.namespace() else {
            return Ok(None);
        };
        if crate::fx4_host::owns_call_head(ns_part, sym.name()) {
            return Ok(None);
        }
        let cur = self.current_ns();
        if self
            .eval
            .registry
            .namespace_target_in(&cur, ns_part)
            .is_some()
        {
            return Ok(None);
        }
        let Some(class) = self.static_head_class(&cur, ns_part) else {
            if crate::classemu::is_class_position(ns_part) {
                return Err(self.err(
                    class::CLASS_NOT_FOUND,
                    ns_part.to_owned(),
                    span,
                    Verb::Compiling,
                    None,
                ));
            }
            return Ok(None);
        };
        let argc = items.len() - 1;
        if let Some(arities) = crate::classemu::static_member_arities(class.name(), sym.name()) {
            if !arities.contains(&argc) {
                let message = crate::types::no_method(sym.name(), argc, class.name())
                    .into_inner()
                    .message;
                return Err(self.err(
                    class::ILLEGAL_ARG,
                    message,
                    span,
                    Verb::Compiling,
                    Some(sym.to_string()),
                ));
            }
        }
        Ok(Some(self.static_call(
            class.name(),
            sym.name(),
            &items[1..],
            span,
        )?))
    }

    /// The class a static head's namespace part names: simple names
    /// resolve through the import table (which carries the JVM's
    /// 106 default imports, so `Class/forName` needs no explicit
    /// import), dotted names through the shared lattice (which
    /// also carves struct names, like `import`).
    fn static_head_class(&self, cur: &str, ns_part: &str) -> Option<Arc<ClassVal>> {
        if ns_part.contains('.') {
            crate::classemu::resolve_class_name(self.eval, ns_part)
        } else {
            crate::classemu::import_in(self.eval, cur, ns_part)
        }
    }

    /// Synthesize `(__static* "FQN" "member" args…)` over an
    /// already-analyzed tail.
    fn static_call(
        &mut self,
        fqn: &str,
        member: &str,
        args: &[&Value],
        span: Span,
    ) -> Result<Expr, CompileError> {
        let head = Expr::Global(self.synth_var("__static*"));
        let mut out = Vec::with_capacity(args.len() + 2);
        out.push(Expr::Quote(Value::Str(Arc::from(fqn))));
        out.push(Expr::Quote(Value::Str(Arc::from(member))));
        for arg in args {
            out.push(self.analyze(arg, span, false)?);
        }
        Ok(Expr::Invoke {
            head: Box::new(head),
            args: out,
            span,
        })
    }

    /// A synthesized `clojure.core` head's captured var: resolve,
    /// interning unbound when missing (bare evaluators analyze
    /// dot/host forms without a booted core — the miss surfaces as
    /// the usual unbound error at invoke, like before; #79).
    fn synth_var(&mut self, op: &str) -> Var {
        let sym = Symbol::new(Some("clojure.core"), op);
        let id = self.eval.var_id(&sym);
        Var { id, sym }
    }

    /// Synthesize `(__op* 'Sym args…)` over an already-analyzed tail.
    fn dot_call(
        &mut self,
        op: &str,
        sym: Symbol,
        args: &[&Value],
        span: Span,
    ) -> Result<Expr, CompileError> {
        let head = Expr::Global(self.synth_var(op));
        let mut out = Vec::with_capacity(args.len() + 1);
        out.push(Expr::Quote(Value::Symbol(sym)));
        for arg in args {
            out.push(self.analyze(arg, span, false)?);
        }
        Ok(Expr::Invoke {
            head: Box::new(head),
            args: out,
            span,
        })
    }

    /// Synthesize `(__op* "name" args…)` (field/method names travel
    /// as strings).
    fn dot_call_str(
        &mut self,
        op: &str,
        name: &str,
        args: &[&Value],
        span: Span,
    ) -> Result<Expr, CompileError> {
        let head = Expr::Global(self.synth_var(op));
        let mut out = Vec::with_capacity(args.len() + 1);
        out.push(Expr::Quote(Value::Str(Arc::from(name))));
        for arg in args {
            out.push(self.analyze(arg, span, false)?);
        }
        Ok(Expr::Invoke {
            head: Box::new(head),
            args: out,
            span,
        })
    }

    /// Intercept special/builtin heads: bare specials (shadowable
    /// macros defer to any binding; real specials never do), and
    /// `clojure.core`-qualified builtins while the core var is
    /// intact (syntax-quote emits `(clojure.core/let …)`).
    /// Qualified real specials stay var lookups that miss. `Some`
    /// when a form parsed, `None` to continue to expansion.
    fn intercept_head(
        &mut self,
        head: &Value,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Option<Expr>, CompileError> {
        let Value::Symbol(sym) = head else {
            return Ok(None);
        };
        // Shadowable macros defer to any binding (a local, a
        // current-ns global, or a referred one); real specials
        // never do (probe31 `shadow-*`). The builtin's own
        // `clojure.core` macro var is the builtin, not a shadow
        // (only a non-macro re-`def` there shadows).
        // MSRV 1.85: let-chains are 1.88+, so the nesting stays.
        #[allow(clippy::collapsible_if)]
        if sym.namespace().is_none() {
            if let Some(op) = special_op(sym.name()) {
                let shadowed = is_shadowable_macro(sym.name())
                    && (self.local_idx(sym.name()).is_some() || self.global_shadows(sym));
                if shadowed {
                    return Ok(None);
                }
                let expr = self.analyze_special(op, items, span, tail)?;
                return Ok(Some(expr));
            }
        }
        if sym.namespace() == Some("clojure.core")
            && is_shadowable_macro(sym.name())
            && self.core_macro_intact(sym)
        {
            let op = special_op(sym.name()).expect("shadowable macros are special forms");
            let expr = self.analyze_special(op, items, span, tail)?;
            return Ok(Some(expr));
        }
        Ok(None)
    }

    /// The `&env` for macro calls: nil at top level, else the
    /// in-scope local names mapped to `true` (the JVM's
    /// `LocalBinding` values are unobservable here — keys,
    /// `contains?`, and truthiness behave — fence).
    fn macro_env(&self) -> Value {
        if self.locals.is_empty() {
            Value::Nil
        } else {
            let mut out = new_map();
            for name in &self.locals {
                out = out.insert(Value::Symbol(Symbol::new(None, name)), Value::Bool(true));
            }
            Value::Map(out)
        }
    }

    /// Whether `clojure.core/{name}` still resolves to its
    /// builtin macro var (a re-`def` there falls through to
    /// ordinary invocation).
    fn core_macro_intact(&self, sym: &Symbol) -> bool {
        let cur = self.current_ns();
        let Some(target) = self.eval.registry.resolve_in(&cur, sym) else {
            return false;
        };
        if target.namespace() != Some("clojure.core") || target.name() != sym.name() {
            return false;
        }
        self.eval
            .registry
            .get("clojure.core")
            .and_then(|ns| ns.lookup(&target))
            .is_some_and(|id| crate::macros::is_macro_var(self.eval, id))
    }

    /// Invoke-position private check (the `isInline` analog):
    /// cross-namespace private non-macro vars fail with the
    /// operator as written and the `#'` target (probed).
    fn check_private_invoke(&self, sym: &Symbol, span: Span) -> Result<(), CompileError> {
        let cur = self.current_ns();
        let Some((target, id)) = self.eval.registry.resolve_target_and_id(&cur, sym) else {
            return Ok(());
        };
        let home = target.namespace().unwrap_or("");
        if home == cur {
            return Ok(());
        }
        if crate::macros::is_macro_var(self.eval, id) {
            return Ok(());
        }
        if crate::macros::is_private_var(self.eval, id) {
            return Err(self.err(
                class::ILLEGAL_STATE,
                format!("var: #'{} is not public", target),
                span,
                Verb::Compiling,
                Some(sym.to_string()),
            ));
        }
        Ok(())
    }

    /// An imported class in operator position (`(String)`): the
    /// JVM's bare `Expecting var` (oracle `invoke-class`, `op-args`
    /// — bare, so it escapes `eval` like any uncaught throwable).
    /// Only unqualified import mappings reach here: locals shadow
    /// imports, vars resolve first, and fully-qualified heads take
    /// the ordinary invoke path (oracle `call-fq-noargs`).
    fn check_class_operator(&self, sym: &Symbol, span: Span) -> Result<(), CompileError> {
        if sym.namespace().is_some() || self.local_idx(sym.name()).is_some() {
            return Ok(());
        }
        let cur = self.current_ns();
        if self.eval.registry.resolve_in(&cur, sym).is_some() {
            return Ok(());
        }
        if let Some(class) = crate::classemu::import_in(self.eval, &cur, sym.name()) {
            return Err(CompileError::bare(
                class::RUNTIME,
                format!(
                    "Expecting var, but {} is mapped to class {}",
                    sym.name(),
                    class.name()
                ),
                span,
                &self.source,
            ));
        }
        Ok(())
    }

    /// Expand a macro-headed list (`isMacro` + call, then the
    /// 8aa45d128 hint wrap — upstream `analyzeSeq` funnels through
    /// `macroexpand1`): locals shadow macros; private
    /// cross-namespace macros escape bare (probed: direct and
    /// nested invocations alike surface the raw
    /// `IllegalStateException`).
    fn try_expand_macro(
        &mut self,
        form: &Value,
        head: &Value,
        args: &[&Value],
        span: Span,
    ) -> Result<Option<Value>, CompileError> {
        if matches!(head, Value::Symbol(sym) if sym.namespace().is_none() && self.local_idx(sym.name()).is_some())
        {
            return Ok(None);
        }
        let mac = match crate::macros::lookup_macro(self.eval, head) {
            Err(private) => {
                return Err(CompileError::bare(
                    class::ILLEGAL_STATE,
                    private.into_inner().message,
                    span,
                    &self.source,
                ));
            }
            Ok(mac) => mac,
        };
        let Some(mac) = mac else {
            return Ok(None);
        };
        let env = self.macro_env();
        let arg_vals: Vec<Value> = args.iter().map(|a| (*a).clone()).collect();
        let expanded =
            crate::macros::expand1(self.eval, head, &mac.root, form, &env, &arg_vals, span)?;
        Ok(Some(crate::macros::maybe_wrap_hint(form, expanded)))
    }

    /// Whether a global binding shadows a shadowable-macro
    /// builtin (the builtin's own `clojure.core` macro var does
    /// not — only locals, other-namespace targets, and non-macro
    /// re-`def`s do).
    fn global_shadows(&self, sym: &Symbol) -> bool {
        let cur = self.current_ns();
        let Some(target) = self.eval.registry.resolve_in(&cur, sym) else {
            return false;
        };
        if target.namespace() == Some("clojure.core") && target.name() == sym.name() {
            let id = self
                .eval
                .registry
                .get("clojure.core")
                .and_then(|ns| ns.lookup(&target));
            return id.is_none_or(|id| !crate::macros::is_macro_var(self.eval, id));
        }
        true
    }

    /// Dispatch one special form (operator already parsed).
    fn analyze_special(
        &mut self,
        op: SpecialOp,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        self.check_bare_arity(op, items, span)?;
        match op {
            SpecialOp::Def => self.parse_def(items, span),
            SpecialOp::If => self.parse_if(items, span, tail),
            SpecialOp::Do => Ok(Expr::Do(self.analyze_body(&items[1..], span, tail)?)),
            SpecialOp::Quote => self.parse_quote(items, span),
            SpecialOp::Var => self.parse_var(items, span),
            SpecialOp::Throw => self.parse_throw(items, span),
            SpecialOp::Recur => self.parse_recur(items, span, tail),
            SpecialOp::Set => self.parse_set(items, span),
            other => self.analyze_special_bind(other, items, span, tail),
        }
    }

    /// Binding/fn special forms (C1: an `analyze_special`
    /// sub-dispatch; one arm per form, like `analyze_special`).
    fn analyze_special_bind(
        &mut self,
        op: SpecialOp,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        match op {
            SpecialOp::Try => self.parse_try(items, span, tail),
            SpecialOp::Let => self.parse_let("let", items, span, tail),
            SpecialOp::LetStar => self.parse_let("let*", items, span, tail),
            SpecialOp::Loop => self.parse_let("loop", items, span, tail),
            SpecialOp::LoopStar => self.parse_let("loop*", items, span, tail),
            SpecialOp::LetFn => self.parse_letfn("letfn", items, span, tail),
            SpecialOp::Fn => self.parse_fn(items, span),
            SpecialOp::FnStar => self.parse_fn_star(items, span),
            other => self.analyze_special_scope(other, items, span, tail),
        }
    }

    /// Dynamic-scope/ns special forms (C1: an `analyze_special`
    /// sub-dispatch).
    fn analyze_special_scope(
        &mut self,
        op: SpecialOp,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        match op {
            SpecialOp::Binding => self.parse_binding(items, span, tail),
            SpecialOp::WithRedefs => self.parse_with_redefs(items, span, tail),
            SpecialOp::WithBindings => self.parse_with_bindings(items, span, tail),
            SpecialOp::BoundFn => self.parse_bound_fn(items, span),
            SpecialOp::DefMacro => self.parse_defmacro(items, span),
            SpecialOp::Declare => self.parse_declare(items, span),
            SpecialOp::Ns => self.parse_ns(items, span),
            SpecialOp::Assert => self.parse_assert(items, span),
            other => self.analyze_special_effect(other, items, span, tail),
        }
    }

    /// Effect/utility special forms (C1: an `analyze_special`
    /// sub-dispatch; the tail of the chain).
    fn analyze_special_effect(
        &mut self,
        op: SpecialOp,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        match op {
            SpecialOp::Time => self.parse_time(items, span),
            SpecialOp::WithOutStr => self.parse_with_out_str(items, span),
            SpecialOp::WithInStr => self.parse_with_in_str(items, span),
            SpecialOp::LazySeqStar => self.parse_lazy_seq_star(items, span),
            SpecialOp::DelayStar => self.parse_delay_star(items, span),
            SpecialOp::Locking | SpecialOp::Sync | SpecialOp::Io => {
                self.parse_locking_or_tx(op, items, span, tail)
            }
            SpecialOp::CaseStar => self.parse_case_star(items, span, tail),
            // Core, binding, and scope forms dispatch through
            // `analyze_special`/`analyze_special_bind`/
            // `analyze_special_scope` above.
            _ => unreachable!("analyze_special_effect: non-effect op"),
        }
    }

    /// Bare `(loop)`/`(binding)`/`(letfn)`: the macro fn itself
    /// rejects 0 args before any body runs (probe25). Unspec'd
    /// macros with no arguments fail arity, unwrapped (spec'd ones
    /// reach their spec/assert checks instead); longer forms skip.
    fn check_bare_arity(
        &self,
        op: SpecialOp,
        items: &[&Value],
        span: Span,
    ) -> Result<(), CompileError> {
        if items.len() != 1 {
            return Ok(());
        }
        let name = match op {
            SpecialOp::Loop => "clojure.core/loop",
            SpecialOp::Binding => "clojure.core/binding",
            SpecialOp::LetFn => "clojure.core/letfn",
            SpecialOp::WithRedefs => "clojure.core/with-redefs",
            SpecialOp::WithBindings => "clojure.core/with-bindings",
            SpecialOp::DefMacro => "clojure.core/defmacro",
            SpecialOp::Locking => "clojure.core/locking",
            SpecialOp::Sync => "clojure.core/sync",
            _ => return Ok(()),
        };
        Err(CompileError::macro_arity(name, span, &self.source))
    }

    /// `(def sym doc? init?)`: count routing first (a 3-count with a
    /// non-string second is "Too many", probe25), then the symbol
    /// check. Interns before analyzing the init (self-`def` stays
    /// unbound, probe25).
    /// `def` target rules shared by `def` and `declare`: same-ns
    /// symbols create, other-ns symbols fail — "Can't create defs"
    /// when the var exists, "Can't refer" when it doesn't (probe34
    /// `ns-intern-def`, `ns-crossdef-missing`, probe37
    /// `declare-qualified`).
    fn check_def_target(&self, sym: &Symbol, span: Span) -> Result<(), CompileError> {
        let cur = self.current_ns();
        if sym.namespace().is_some_and(|ns| ns != cur) {
            let message = if self.eval.registry.resolve_in(&cur, sym).is_some() {
                "Can't create defs outside of current ns"
            } else {
                "Can't refer to qualified var that doesn't exist"
            };
            return Err(self.err(
                class::RUNTIME,
                message.to_owned(),
                span,
                Verb::Compiling,
                Some("def".to_owned()),
            ));
        }
        Ok(())
    }

    fn parse_def(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let (sym, doc, init) = self.def_parts(items, span)?;
        // `def` always defines in the current namespace.
        self.check_def_target(sym, span)?;
        let cur = self.current_ns();
        // S16: `def` over an imported name is a compile error (the
        // var is never interned).
        if let Some(class) = crate::classemu::import_in(self.eval, &cur, sym.name()) {
            return Err(self.def_error(
                &format!(
                    "Expecting var, but {} is mapped to class {}",
                    sym.name(),
                    class.name()
                ),
                span,
            ));
        }
        let qualified = self
            .eval
            .registry
            .qualify_in(&cur, sym)
            .with_meta(sym.meta().cloned());
        self.eval.registry.intern_def_target(&qualified);
        let init = init
            .map(|form| self.analyze(form, span, false).map(Box::new))
            .transpose()?;
        Ok(Expr::Def {
            sym: qualified,
            doc,
            init,
            span,
        })
    }

    /// `def` arity shapes: bare, init, or docstring + init (a
    /// 3-arg form without a string second arg is `Too many`).
    fn def_parts<'v>(
        &self,
        items: &[&'v Value],
        span: Span,
    ) -> Result<(&'v Symbol, Option<String>, Option<&'v Value>), CompileError> {
        match items.len() - 1 {
            0 => Err(self.def_error("Too few arguments to def", span)),
            1 => Ok((self.def_target(items[1], span)?, None, None)),
            2 => Ok((self.def_target(items[1], span)?, None, Some(items[2]))),
            3 => {
                let Value::Str(doc) = items[2] else {
                    return Err(self.def_error("Too many arguments to def", span));
                };
                Ok((
                    self.def_target(items[1], span)?,
                    Some(doc.to_string()),
                    Some(items[3]),
                ))
            }
            _ => Err(self.def_error("Too many arguments to def", span)),
        }
    }

    /// The `def` target symbol (anything else is the
    /// `First argument` error).
    fn def_target<'v>(&self, form: &'v Value, span: Span) -> Result<&'v Symbol, CompileError> {
        as_symbol(form)
            .ok_or_else(|| self.def_error("First argument to def must be a Symbol", span))
    }

    /// A `def` (`Compiling`) shape error.
    fn def_error(&self, message: &str, span: Span) -> CompileError {
        self.err(
            class::RUNTIME,
            message.to_owned(),
            span,
            Verb::Compiling,
            Some("def".to_owned()),
        )
    }

    /// `(if test then else?)`.
    fn parse_if(&mut self, items: &[&Value], span: Span, tail: bool) -> Result<Expr, CompileError> {
        let argc = items.len() - 1;
        if argc < 2 {
            return Err(self.err(
                class::RUNTIME,
                "Too few arguments to if".to_owned(),
                span,
                Verb::Compiling,
                Some("if".to_owned()),
            ));
        }
        if argc > 3 {
            return Err(self.err(
                class::RUNTIME,
                "Too many arguments to if".to_owned(),
                span,
                Verb::Compiling,
                Some("if".to_owned()),
            ));
        }
        let test = self.analyze(items[1], span, false)?;
        let then = self.analyze(items[2], span, tail)?;
        let els = match items.get(3) {
            Some(form) => self.analyze(form, span, tail)?,
            None => Expr::Literal(Value::Nil),
        };
        Ok(Expr::If {
            test: Box::new(test),
            then: Box::new(then),
            els: Box::new(els),
        })
    }

    /// `(quote form)`: exactly one argument (`ExceptionInfo`).
    fn parse_quote(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let argc = items.len() - 1;
        if argc != 1 {
            return Err(self.err(
                class::EX_INFO,
                format!("Wrong number of args ({argc}) passed to quote"),
                span,
                Verb::Compiling,
                Some("quote".to_owned()),
            ));
        }
        // The payload's literal strings canonicalize (fence 4a):
        // `(identical? (quote "ab") "ab")` is true on the JVM
        // (the quote answers the interned constant).
        Ok(Expr::Quote(clj_value::intern_literal_strings(items[1])))
    }

    /// Resolve a symbol as a var (shared by `var` and `binding`
    /// var-forms): locals are invisible, refers and other
    /// namespaces are visible (probe34 `ns-var-refer`,
    /// `ns-var-cross`), and every miss is the single `Unable to
    /// resolve var` shape (probe25). Captures the id (like the
    /// JVM's compiled `Var` reference — #79).
    fn resolve_var(&self, sym: &Symbol, span: Span, op: &str) -> Result<Var, CompileError> {
        // Locals are invisible: only namespace membership matters.
        let cur = self.current_ns();
        if let Some((target, id)) = self.eval.registry.resolve_target_and_id(&cur, sym) {
            Ok(Var { id, sym: target })
        } else {
            Err(self.err(
                class::RUNTIME,
                format!("Unable to resolve var: {sym} in this context"),
                span,
                Verb::Compiling,
                Some(op.to_owned()),
            ))
        }
    }

    /// `(var sym)`: missing is the modern-JDK `NullPointerException`,
    /// non-symbols are `ClassCastException`s, extras ignored.
    fn parse_var(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let Some(second) = items.get(1) else {
            return Err(self.err(
                class::NULL_POINTER,
                "Cannot read field \"ns\" because \"sym\" is null".to_owned(),
                span,
                Verb::Compiling,
                Some("var".to_owned()),
            ));
        };
        let Value::Symbol(sym) = second else {
            return Err(self.err(
                class::CLASS_CAST,
                NativeError::cast(&jvm_class_of(second), "clojure.lang.Symbol")
                    .into_inner()
                    .message,
                span,
                Verb::Compiling,
                Some("var".to_owned()),
            ));
        };
        Ok(Expr::VarRef(self.resolve_var(sym, span, "var")?))
    }

    /// `(throw expr)`: exactly one argument.
    fn parse_throw(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let argc = items.len() - 1;
        if argc != 1 {
            let few = argc < 1;
            return Err(self.err(
                class::RUNTIME,
                format!(
                    "Too {} arguments to throw, throw expects a single Throwable instance",
                    if few { "few" } else { "many" }
                ),
                span,
                Verb::Compiling,
                Some("throw".to_owned()),
            ));
        }
        Ok(Expr::Throw(Box::new(self.analyze(items[1], span, false)?)))
    }

    /// `(recur expr*)`: across-`try` first (beats tail, target,
    /// and arity alike), then tail (targeted only — targetless never
    /// consults it), then arg analysis (failures keep their own
    /// wrappers), then arity (§E2). A targetless `(recur)` is the
    /// constant tail-position error (the JVM hangs; §E2 documents the
    /// divergence).
    fn parse_recur(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let argc = items.len() - 1;
        if self.no_recur {
            return Err(self.err(
                class::UNSUPPORTED,
                "Cannot recur across try".to_owned(),
                span,
                Verb::Compiling,
                Some("recur".to_owned()),
            ));
        }
        if self.recur.is_some() && !tail {
            return Err(self.tail_err(span));
        }
        let mut args = Vec::with_capacity(argc);
        for arg in &items[1..] {
            args.push(self.analyze(arg, span, false)?);
        }
        let count = self.recur.unwrap_or(0);
        if argc == 0 && self.recur.is_none() {
            return Err(self.tail_err(span));
        }
        if argc != count {
            return Err(self.err(
                class::ILLEGAL_ARG,
                format!("Mismatched argument count to recur, expected: {count} args, got: {argc}"),
                span,
                Verb::Compiling,
                Some("recur".to_owned()),
            ));
        }
        Ok(Expr::Recur(args))
    }

    /// The constant tail-position error.
    fn tail_err(&self, span: Span) -> CompileError {
        self.err(
            class::UNSUPPORTED,
            "Can only recur from tail position".to_owned(),
            span,
            Verb::Compiling,
            Some("recur".to_owned()),
        )
    }

    /// `(set! target val)`: the target analyzes first (failures
    /// propagate with their own wrappers); locals are the constant
    /// `fn*` error; anything analyzed-but-unassignable is `Invalid
    /// assignment target`.
    fn parse_set(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        if items.len() != 3 {
            return Err(self.err(
                class::ILLEGAL_ARG,
                "Malformed assignment, expecting (set! target val)".to_owned(),
                span,
                Verb::Compiling,
                Some("set!".to_owned()),
            ));
        }
        let target = self.analyze(items[1], span, false)?;
        match target {
            Expr::Local(_) => Err(self.err(
                class::ILLEGAL_ARG,
                format!(
                    "Cannot assign to non-mutable: {}",
                    local_name(items[1]).unwrap_or("?")
                ),
                span,
                Verb::Compiling,
                Some("fn*".to_owned()),
            )),
            Expr::Global(var) => Ok(Expr::Set {
                var,
                val: Box::new(self.analyze(items[2], span, false)?),
            }),
            _ => Err(self.err(
                class::ILLEGAL_ARG,
                "Invalid assignment target".to_owned(),
                span,
                Verb::Compiling,
                Some("set!".to_owned()),
            )),
        }
    }
}

/// A form as a symbol, if it is one.
fn as_symbol(form: &Value) -> Option<&Symbol> {
    match form {
        Value::Symbol(sym) => Some(sym),
        _ => None,
    }
}

/// The name of a local-target form (always a symbol here).
fn local_name(form: &Value) -> Option<&str> {
    as_symbol(form).map(Symbol::name)
}

/// A plain binding symbol: unqualified and not `&` (`::local-name`).
fn is_local_name(sym: &Symbol) -> bool {
    sym.namespace().is_none() && sym.name() != "&"
}

/// `fn`/`letfn` names: any unqualified symbol — including `&`,
/// which the JVM mangles (`_AMPERSAND_`) but otherwise treats
/// as a name (probe33 `fn-amp-name`, `letfn-amp`).
fn is_fn_name(sym: &Symbol) -> bool {
    sym.namespace().is_none()
}

/// Characters allowed after `&` in key lists are unrestricted; the
/// bindings before it must be idents (symbols or keywords).
fn is_ident(v: &Value) -> bool {
    matches!(v, Value::Symbol(_) | Value::Keyword(_))
}

/// Bound names of a pattern in canonical slot order (entries, then
/// keys, then `:as`/outlets; fixed, then rest, then `:as`). The
/// builder and the evaluator share this order.
pub(crate) fn pattern_names(pat: &Pattern) -> Vec<String> {
    match pat {
        Pattern::Symbol(name) => vec![name.clone()],
        Pattern::Vec(v) => {
            let mut out = Vec::new();
            for fixed in &v.fixed {
                out.extend(pattern_names(fixed));
            }
            if let Some(rest) = &v.rest {
                out.extend(pattern_names(rest));
            }
            out.extend(v.as_sym.clone());
            out
        }
        Pattern::Map(m) => {
            let mut out = Vec::new();
            for (local, _, _, _) in &m.entries {
                out.extend(pattern_names(local));
            }
            for key in &m.keys {
                out.push(key.local.clone());
            }
            out.extend(m.as_sym.clone());
            out.extend(m.select.clone());
            out.extend(m.all.clone());
            out.extend(m.excess.clone());
            out.extend(m.missing.clone());
            out
        }
    }
}

impl<'a> Analyzer<'a> {
    /// `(try body* catch* finally?)`: leading non-clauses are the
    /// body; everything after must be clauses. The whole `try` sets
    /// the `recur` fence; catches and `finally` analyze non-tail
    /// (probe25).
    fn parse_try(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let mut splits = items.len();
        for (i, form) in items.iter().enumerate().skip(1) {
            if is_clause(form) {
                splits = i;
                break;
            }
        }
        let saved_recur = self.no_recur;
        self.no_recur = true;
        let body = self.analyze_body(&items[1..splits], span, tail)?;
        // Catch/finally bodies reset the fence and hide the recur
        // target (empty, never `nil`): a `recur` there is always the
        // tail-position error, whatever the outer fence or target
        // (probe30 `fence-catch-*`, `fence-finally-*`). Nested `try`
        // bodies re-arm the fence; nested `fn`/`loop` re-establish a
        // target.
        self.no_recur = false;
        let saved_target = self.recur;
        self.recur = Some(0);
        let mut catches = Vec::new();
        let mut finally_body = Vec::new();
        let mut seen_finally = false;
        for form in &items[splits..] {
            let clause = as_clause(form);
            match clause {
                Some(("catch", list)) => {
                    if seen_finally {
                        return Err(self.err(
                            class::RUNTIME,
                            "finally clause must be last in try expression".to_owned(),
                            span,
                            Verb::Compiling,
                            Some("try".to_owned()),
                        ));
                    }
                    catches.push(self.parse_catch(list, span)?);
                }
                Some(("finally", list)) => {
                    if seen_finally {
                        return Err(self.err(
                            class::RUNTIME,
                            "finally clause must be last in try expression".to_owned(),
                            span,
                            Verb::Compiling,
                            Some("try".to_owned()),
                        ));
                    }
                    seen_finally = true;
                    let parts: Vec<&Value> = list.iter().collect();
                    finally_body = self.analyze_body(&parts[1..], span, false)?;
                }
                _ => {
                    // Past a `finally`, anything is the must-be-last
                    // error; otherwise the only-catch-or-finally one
                    // (probe32 `try-after-*`).
                    let message = if seen_finally {
                        "finally clause must be last in try expression"
                    } else {
                        "Only catch or finally clause can follow catch in try expression"
                    };
                    return Err(self.err(
                        class::RUNTIME,
                        message.to_owned(),
                        span,
                        Verb::Compiling,
                        Some("try".to_owned()),
                    ));
                }
            }
        }
        self.no_recur = saved_recur;
        self.recur = saved_target;
        Ok(Expr::Try {
            body,
            catches,
            finally_body,
        })
    }

    /// One `(catch class binding body*)`. Shape errors blame the
    /// `try` (the clause is parsed inline, without rebinding —
    /// probe25 `catch-nonsym`, probe32 `catch-*-nope`); only the
    /// clause body descends (via [`Analyzer::analyze`]).
    fn parse_catch(&mut self, list: &List, span: Span) -> Result<Catch, CompileError> {
        let parts: Vec<&Value> = list.iter().collect();
        let class = match parts.get(1) {
            None => {
                return Err(self.err(
                    class::ILLEGAL_ARG,
                    "Unable to resolve classname: null".to_owned(),
                    span,
                    Verb::Compiling,
                    Some("try".to_owned()),
                ));
            }
            Some(Value::Symbol(sym)) => self.resolve_classname(sym, span)?,
            Some(other) => {
                return Err(self.err(
                    class::ILLEGAL_ARG,
                    format!("Unable to resolve classname: {}", value_of(other)),
                    span,
                    Verb::Compiling,
                    Some("try".to_owned()),
                ));
            }
        };
        let binding = match parts.get(2) {
            None => {
                return Err(self.err(
                    class::ILLEGAL_ARG,
                    "Bad binding form, expected symbol, got: null".to_owned(),
                    span,
                    Verb::Compiling,
                    Some("try".to_owned()),
                ));
            }
            Some(Value::Symbol(sym)) if sym.namespace().is_some() => {
                return Err(self.err(
                    class::RUNTIME,
                    format!("Can't bind qualified name:{sym}"),
                    span,
                    Verb::Compiling,
                    Some("try".to_owned()),
                ));
            }
            Some(Value::Symbol(sym)) => sym.name().to_owned(),
            Some(other) => {
                return Err(self.err(
                    class::ILLEGAL_ARG,
                    format!(
                        "Bad binding form, expected symbol, got: {}",
                        value_of(other)
                    ),
                    span,
                    Verb::Compiling,
                    Some("try".to_owned()),
                ));
            }
        };
        self.locals.push(binding);
        let slot = self.locals.len() - 1;
        let body = self.analyze_body(&parts[3..], span, false)?;
        self.locals.pop();
        Ok(Catch { class, slot, body })
    }

    /// Resolve a `catch` class: default-import shorts (`java.lang`
    /// throwables, the real-but-unthrowable names — which resolve
    /// and never match, the JVM `VerifyError`s those instead — plus
    /// clojure's `ExceptionInfo` short and the `ReaderException`
    /// short), and dotted FQNs (curated
    /// throwables, `java.lang.*` shorts, the two `clojure.lang.*`
    /// throwables). Unknown shorts are `Unable to resolve
    /// classname`; unknown dotted names are `ClassNotFoundException`
    /// (probe32; §E6).
    fn resolve_classname(&self, sym: &Symbol, span: Span) -> Result<String, CompileError> {
        match sym.namespace() {
            // Default imports (`ArityException` short does NOT
            // resolve — oracle-verified, probe32). Dotted bare names
            // are FQNs: the curated set resolves, `java.lang.*` and
            // `clojure.lang.*` prefixes strip to their short rules,
            // anything else dotted is `ClassNotFoundException`.
            None => {
                if sym.name().contains('.') {
                    return self.resolve_dotted(sym.name(), span);
                }
                match resolve_java_lang(sym.name()) {
                    Some(jvm) => Ok(jvm.to_owned()),
                    None if sym.name() == "ExceptionInfo" => {
                        Ok("clojure.lang.ExceptionInfo".to_owned())
                    }
                    // The short `ReaderException` is a port
                    // extension (the JVM only resolves the FQN —
                    // `clojure.core` never imports it — but the
                    // S-instream spec pins the short spelling).
                    None if sym.name() == "ReaderException" => {
                        Ok("clojure.lang.LispReader$ReaderException".to_owned())
                    }
                    None => Err(self.err(
                        class::ILLEGAL_ARG,
                        format!("Unable to resolve classname: {sym}"),
                        span,
                        Verb::Compiling,
                        Some("try".to_owned()),
                    )),
                }
            }
            // Slash-qualified: dotless namespaces (`q/Nope`) never
            // resolve; dotted ones (`java.lang`, `clojure.lang`,
            // `java.io`, …) join with a dot and run the FQN rule.
            Some(ns) if !ns.contains('.') => Err(self.err(
                class::ILLEGAL_ARG,
                format!("Unable to resolve classname: {sym}"),
                span,
                Verb::Compiling,
                Some("try".to_owned()),
            )),
            Some(ns) => self.resolve_dotted(&format!("{ns}.{}", sym.name()), span),
        }
    }

    /// A dotted fully-qualified classname: the curated set resolves,
    /// `java.lang.*` strips to the short rule, `clojure.lang.*`
    /// takes the two throwables, anything else is
    /// `ClassNotFoundException` (probe32).
    fn resolve_dotted(&self, fqn: &str, span: Span) -> Result<String, CompileError> {
        if is_curated_fqn(fqn) {
            return Ok(fqn.to_owned());
        }
        if let Some(jvm) = fqn.strip_prefix("java.lang.").and_then(resolve_java_lang) {
            return Ok(jvm.to_owned());
        }
        if fqn
            .strip_prefix("clojure.lang.")
            .is_some_and(|short| matches!(short, "ExceptionInfo" | "ArityException"))
        {
            return Ok(fqn.to_owned());
        }
        Err(self.err(
            class::CLASS_NOT_FOUND,
            fqn.to_owned(),
            span,
            Verb::Compiling,
            Some("try".to_owned()),
        ))
    }

    /// `(binding [var init …] body…)`: assert-args shapes, then each
    /// var-form validated exactly like `var` (the macro emits `(var
    /// v)`, probe25).
    fn parse_binding(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let op = as_symbol(items[0])
            .expect("binding dispatches on the operator symbol")
            .name()
            .to_owned();
        let bindings = items
            .get(1)
            .expect("bare binding fails arity before dispatch");
        let Some(forms) = vector_shape_items(bindings) else {
            return Err(self.err(
                class::ILLEGAL_ARG,
                "binding requires a vector for its binding".to_owned(),
                span,
                Verb::Macroexpanding,
                Some(op),
            ));
        };
        // MSRV 1.85: `usize::is_multiple_of` is 1.87+.
        #[allow(clippy::manual_is_multiple_of)]
        if forms.len() % 2 != 0 {
            return Err(self.err(
                class::ILLEGAL_ARG,
                "binding requires an even number of forms in binding vector".to_owned(),
                span,
                Verb::Macroexpanding,
                Some(op),
            ));
        }
        let mut pairs = Vec::with_capacity(forms.len() / 2);
        // Step indexing, not `chunks_exact` (new clippy wants
        // `as_chunks`, which needs Rust 1.88 > MSRV 1.85).
        for i in (0..forms.len()).step_by(2) {
            let var_sym = match forms[i] {
                Value::Symbol(sym) => self.resolve_var(sym, span, "var")?,
                other => {
                    return Err(self.err(
                        class::CLASS_CAST,
                        NativeError::cast(&jvm_class_of(other), "clojure.lang.Symbol")
                            .into_inner()
                            .message,
                        span,
                        Verb::Compiling,
                        Some("var".to_owned()),
                    ));
                }
            };
            pairs.push((var_sym, self.analyze(forms[i + 1], span, false)?));
        }
        Ok(Expr::Binding {
            pairs,
            body: self.analyze_body(&items[2..], span, tail)?,
        })
    }

    /// `(with-redefs bindings body…)`: bindings seq like the
    /// macro's `zipmap` — vectors, lists, sets, nil (empty),
    /// strings (chars, like the JVM's `seq`), and maps (entries
    /// as `MapEntry` values, like the JVM's `seq` — a lone entry
    /// odd-drops, paired entries fail the `var` `Symbol` cast as
    /// `MapEntry`); anything else is the `ISeq` error under
    /// `macroexpanding with-redefs`. An odd trailing binding
    /// silently drops (probe37 `wrd-odd`).
    fn parse_with_redefs(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let op = as_symbol(items[0])
            .expect("with-redefs dispatches on the operator symbol")
            .name()
            .to_owned();
        let bindings = items.get(1).copied().unwrap_or(&Value::Nil);
        let forms = self.with_redefs_forms(bindings, span, &op)?;
        let mut pairs = Vec::with_capacity(forms.len() / 2);
        // Step indexing, not `chunks_exact` (new clippy wants
        // `as_chunks`, which needs Rust 1.88 > MSRV 1.85); the
        // `len() - len() % 2` drops an odd tail like `zipmap`.
        for i in (0..forms.len() - forms.len() % 2).step_by(2) {
            let var_sym = match &forms[i] {
                Value::Symbol(sym) => self.resolve_var(sym, span, "var")?,
                other => {
                    return Err(self.err(
                        class::CLASS_CAST,
                        NativeError::cast(&jvm_class_of(other), "clojure.lang.Symbol")
                            .into_inner()
                            .message,
                        span,
                        Verb::Compiling,
                        Some("var".to_owned()),
                    ));
                }
            };
            pairs.push((var_sym, self.analyze(&forms[i + 1], span, false)?));
        }
        Ok(Expr::WithRedefs {
            pairs,
            body: self.analyze_body(&items[2..], span, tail)?,
        })
    }

    /// The `with-redefs` binding items: vectors, lists, sets,
    /// strings (chars), and nil (empty) seq; a non-empty map
    /// fails its first entry's `Symbol` cast under `compiling
    /// var` (entries aren't symbols); anything else is the `ISeq`
    /// error under `macroexpanding with-redefs`.
    fn with_redefs_forms(
        &self,
        bindings: &Value,
        span: Span,
        op: &str,
    ) -> Result<Vec<Value>, CompileError> {
        match bindings {
            v @ (Value::Vector(_) | Value::MapEntry { .. }) => Ok(vector_shape_items(v)
                .expect("vector shape checked")
                .into_iter()
                .cloned()
                .collect()),
            Value::List(l) => Ok(l.iter().cloned().collect()),
            Value::Nil => Ok(Vec::new()),
            Value::Set(s) => Ok(s.iter().cloned().collect()),
            // Fence 4b A4: strings seq to chars (like the JVM's
            // `seq`, which feeds each char to `(var …)` — the
            // `Symbol` cast fails as `Character`, not `ISeq`).
            Value::Str(s) => Ok(s.chars().map(Value::Char).collect()),
            // Fence 4b E1e/E1q: maps seq to their entries (as
            // `MapEntry` values, like the JVM's `seq`) — a lone
            // entry odd-drops downstream (no error); paired entries
            // fail the `var` Symbol cast as `MapEntry`.
            Value::Map(m) => Ok(m
                .iter()
                .map(|(k, v)| map_entry(k.clone(), v.clone()))
                .collect()),
            other => Err(self.err(
                class::ILLEGAL_ARG,
                format!(
                    "Don't know how to create ISeq from: {}",
                    jvm_class_of(other)
                ),
                span,
                Verb::Macroexpanding,
                Some(op.to_owned()),
            )),
        }
    }

    /// `(with-bindings map-expr body…)`: the map is an evaluated
    /// expression (computed maps work, probe37 `wb-computed-map`).
    /// Bare `(with-bindings)` is intercepted by `check_bare_arity`.
    fn parse_with_bindings(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let map = items
            .get(1)
            .expect("bare with-bindings fails arity before dispatch");
        Ok(Expr::WithBindings {
            map: Box::new(self.analyze(map, span, false)?),
            body: self.analyze_body(&items[2..], span, tail)?,
        })
    }

    /// `(bound-fn name? argv body…)`: parsed exactly like `fn` (the
    /// macro expands to one, probe37 `bf-expansion`) — including
    /// the named single-`letfn` shape, whose slot layout the
    /// evaluator replays with a conveyed closure.
    fn parse_bound_fn(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let (name, raw) = self.parse_fn_front(&items[1..], span)?;
        let (fun, _) = self.build_fun(name.clone(), &raw, span)?;
        Ok(Expr::BoundFn { name, fun })
    }

    /// `(defmacro name doc? attr? fdecl)`: the name must be an
    /// unqualified symbol and the fdecl a vector-headed single
    /// arity or list-of-clauses (else the `defmacro` spec shape —
    /// the JVM routes through `defn`, the port names its own
    /// validator); clause *contents* parse through `parse_fn`
    /// with `fn`-flavored errors. `&form`/`&env` prepend
    /// positionally (an explicit leading `&form` shadows, probe38
    /// `83-explicit-ampersand`); the display is the qualified var
    /// name (arity errors name `user/m`, probe38 `06`).
    /// A `defmacro` fdecl's user arglists (without `&form`/`&env`,
    /// for `:arglists`): one vector, or one vector per list clause
    /// (anything else is the `defmacro` spec error).
    fn defmacro_arglists(&self, rest: &[&Value], span: Span) -> Result<Vec<Value>, CompileError> {
        let spec = || self.spec_error("clojure.core/defmacro", span);
        match rest[0] {
            v if vector_shape_items(v).is_some() => Ok(vec![v.clone()]),
            Value::List(_) => {
                let mut out = Vec::with_capacity(rest.len());
                for clause in rest.iter().copied() {
                    match clause {
                        Value::List(l) => match l.first() {
                            Some(v) if vector_shape_items(v).is_some() => {
                                out.push(v.clone());
                            }
                            _ => return Err(spec()),
                        },
                        _ => return Err(spec()),
                    }
                }
                Ok(out)
            }
            _ => Err(spec()),
        }
    }

    fn parse_defmacro(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let spec = || self.spec_error("clojure.core/defmacro", span);
        let name_value = items
            .get(1)
            .expect("bare defmacro fails arity before dispatch");
        let Value::Symbol(name_sym) = name_value else {
            return Err(spec());
        };
        if name_sym.namespace().is_some() {
            return Err(spec());
        }
        let mut rest = &items[2..];
        let mut doc = None;
        if let Some(Value::Str(s)) = rest.first() {
            doc = Some(s.to_string());
            rest = &rest[1..];
        }
        let mut attr = None;
        if let Some(Value::Map(m)) = rest.first() {
            attr = Some((*m).clone());
            rest = &rest[1..];
        }
        if rest.is_empty() {
            return Err(spec());
        }
        // The `:arglists` (user arglists, without `&form`/`&env`).
        let arglists = self.defmacro_arglists(rest, span)?;
        // Clause contents through the `fn` parser stages, with
        // `&form`/`&env` consed onto every arglist *before*
        // analysis (like the JVM `defmacro`), so bodies resolve
        // user params at their shifted slots. The shape above
        // guarantees no leading name symbol, so no self-recursion
        // binding.
        let owned = self.collect_fn_clauses(rest, span)?;
        let slices: Vec<&[&Value]> = owned.iter().map(Vec::as_slice).collect();
        let mut raw = self.parse_clause_slices(&slices, span)?;
        let ampform = Value::Symbol(Symbol::new(None, "&form"));
        let ampenv = Value::Symbol(Symbol::new(None, "&env"));
        for clause in &mut raw {
            clause.fixed.insert(0, &ampenv);
            clause.fixed.insert(0, &ampform);
        }
        let (mut fun, _) = self.build_fun(None, &raw, span)?;
        let cur = self.current_ns();
        fun.display = format!("{cur}/{}", name_sym.name());
        let qualified =
            Symbol::new(Some(&cur), name_sym.name()).with_meta(name_sym.meta().cloned());
        self.eval.registry.intern_def_target(&qualified);
        Ok(Expr::DefMacro {
            sym: qualified,
            doc,
            attr,
            arglists,
            fun,
            span,
        })
    }

    /// `(declare name…)`: every name must be a symbol (else the
    /// JDK `IObj` checkcast text — the JVM reaches it through
    /// `vary-meta` during expansion, probe37 `declare-num`,
    /// fx2-mapentry18 `declare-5`) and pass the `def` target
    /// rules (probe37 `declare-qualified`).
    fn parse_declare(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let mut syms = Vec::with_capacity(items.len().saturating_sub(1));
        for form in items[1..].iter().copied() {
            let Value::Symbol(sym) = form else {
                return Err(CompileError::bare(
                    class::CLASS_CAST,
                    cce_message(&jvm_class_of(form), "clojure.lang.IObj"),
                    span,
                    &self.source,
                ));
            };
            self.check_def_target(sym, span)?;
            // `declare` is `(def name)` per name: qualify and
            // intern exactly like `parse_def`.
            let cur = self.current_ns();
            let qualified = self
                .eval
                .registry
                .qualify_in(&cur, sym)
                .with_meta(sym.meta().cloned());
            self.eval.registry.intern_def_target(&qualified);
            syms.push(qualified);
        }
        Ok(Expr::Declare { syms, span })
    }

    /// `(ns name doc? attr? clause…)`: the name must be an
    /// unqualified symbol and every clause a list headed by
    /// `:refer-clojure`/`:require`/`:use` (else the `ns` spec
    /// shape, probe37 `ns-bad-clause`, `ns-num-name`).
    /// `:import`/`:gen-class`/`:load` are spec errors too (N1:
    /// no host classes — loud, never skipped).
    fn parse_ns(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let spec = || self.spec_error("clojure.core/ns", span);
        let Some(name_value) = items.get(1) else {
            return Err(spec());
        };
        let Value::Symbol(name_sym) = name_value else {
            return Err(spec());
        };
        if name_sym.namespace().is_some() {
            return Err(spec());
        }
        let mut rest = &items[2..];
        let mut doc = None;
        if let Some(Value::Str(s)) = rest.first() {
            doc = Some(s.to_string());
            rest = &rest[1..];
        }
        let mut attr = None;
        if let Some(Value::Map(m)) = rest.first() {
            attr = Some((*m).clone());
            rest = &rest[1..];
        }
        let mut clauses = Vec::with_capacity(rest.len());
        for form in rest.iter().copied() {
            let Value::List(l) = form else {
                return Err(spec());
            };
            let mut elems = l.iter();
            match elems.next() {
                Some(Value::Keyword(kw)) if kw.namespace().is_none() => match kw.name() {
                    "refer-clojure" => {
                        clauses.push(NsClause::ReferClojure(elems.cloned().collect()));
                    }
                    "require" => clauses.push(NsClause::Require(elems.cloned().collect())),
                    "use" => clauses.push(NsClause::Use(elems.cloned().collect())),
                    _ => return Err(spec()),
                },
                _ => return Err(spec()),
            }
        }
        Ok(Expr::Ns {
            name: name_sym.name().to_owned(),
            doc,
            attr,
            clauses,
        })
    }
}

/// A `try` clause opener: a list headed by unqualified
/// `catch`/`finally`.
fn is_clause(form: &Value) -> bool {
    as_clause(form).is_some()
}

/// A form-forcing failure as a compile error (an escaping
/// `recur` is impossible — analyzed forms cage every one — and
/// surfaces as an internal error, never a panic).
fn form_force_err(source: &Arc<str>, s: Signal, span: Span) -> CompileError {
    match s {
        Signal::Throw(t) => CompileError::bare(
            t.class.jvm_name(),
            t.message.as_deref().unwrap_or("").to_owned(),
            span,
            source,
        ),
        Signal::Recur(_) => CompileError::bare(
            class::RUNTIME,
            "recur escaped forcing a form".to_owned(),
            span,
            source,
        ),
    }
}

fn as_clause(form: &Value) -> Option<(&str, &List)> {
    match form {
        Value::List(list) => match list.first() {
            Some(Value::Symbol(sym)) if sym.namespace().is_none() => match sym.name() {
                "catch" | "finally" => Some((sym.name(), list)),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// A binding-pattern shape (spec- or star-validated before building).
enum Pat<'v> {
    Symbol(&'v Symbol),
    Vector(Vec<&'v Value>),
    Map(&'v Map),
}

/// The items of a vector-shape form: a vector's elements, or an
/// entry's key and value (`MapEntry extends APersistentVector`, so
/// every binding/param/pattern position that takes a vector takes
/// an entry too). `None` for anything else.
fn vector_shape_items(v: &Value) -> Option<Vec<&Value>> {
    match v {
        Value::Vector(vec) => Some(vec.iter().collect()),
        Value::MapEntry { key, val, .. } => Some(vec![key.as_ref(), val.as_ref()]),
        _ => None,
    }
}

/// The pattern shape of a form (`None` is not a pattern).
fn as_pat(form: &Value) -> Option<Pat<'_>> {
    match form {
        Value::Symbol(sym) => Some(Pat::Symbol(sym)),
        Value::Map(m) => Some(Pat::Map(m)),
        _ => vector_shape_items(form).map(Pat::Vector),
    }
}

/// A literal map key from its code form (#112): the analyzer reads
/// pattern keys as values, but upstream spells symbol keys quoted
/// (`'k` — macro keys are code, evaluated at runtime). Unquote
/// `(quote k)` so both spellings denote `k`. Bare symbols stay
/// literal (fenced: key position holds values here, never var
/// references, so bare never resolves).
fn unquote_key(v: &Value) -> Value {
    if let Value::List(l) = v {
        if l.len() == 2 {
            let mut items = l.iter();
            if let (Some(Value::Symbol(q)), Some(inner)) = (items.next(), items.next()) {
                if q.namespace().is_none() && q.name() == "quote" {
                    return inner.clone();
                }
            }
        }
    }
    v.clone()
}

/// Build `(quote k)` as a value (test-only fixture helper).
#[cfg(test)]
fn quote_form(key: &Value) -> Value {
    Value::List(List::from_vec(vec![
        Value::Symbol(Symbol::new(None, "quote")),
        key.clone(),
    ]))
}

/// The deep `::binding-form` spec mirror: plain symbol, sequential
/// (`forms*` + optional `& rest` + optional `:as sym`), or map
/// (binding entries + known directives with shaped values).
fn spec_valid_binding(form: &Value) -> bool {
    match as_pat(form) {
        Some(Pat::Symbol(sym)) => is_local_name(sym),
        Some(Pat::Vector(forms)) => spec_valid_seq(&forms),
        Some(Pat::Map(m)) => spec_valid_map(m),
        None => false,
    }
}

fn spec_valid_seq(forms: &[&Value]) -> bool {
    let Some(rest) = strip_spec_as(forms) else {
        return false;
    };
    let Some(rest) = strip_spec_rest(rest) else {
        return false;
    };
    // Fixed forms, none `&`/`:as`.
    rest.iter().all(|f| match f {
        Value::Symbol(sym) if sym.namespace().is_none() && sym.name() == "&" => false,
        Value::Keyword(kw) if kw.namespace().is_none() && kw.name() == "as" => false,
        other => spec_valid_binding(other),
    })
}

/// Strip an optional trailing `:as sym` (`None` = spec failure).
fn strip_spec_as<'a, 'v>(forms: &'a [&'v Value]) -> Option<&'a [&'v Value]> {
    let as_at_end = forms.len() >= 2
        && matches!(forms[forms.len() - 2], Value::Keyword(kw) if kw.namespace().is_none() && kw.name() == "as");
    if !as_at_end {
        return Some(forms);
    }
    if !matches!(forms[forms.len() - 1], Value::Symbol(sym) if is_local_name(sym)) {
        return None;
    }
    Some(&forms[..forms.len() - 2])
}

/// Strip an optional `& rest` (`None` = dangling/invalid `&`).
fn strip_spec_rest<'a, 'v>(forms: &'a [&'v Value]) -> Option<&'a [&'v Value]> {
    let Some(pos) = forms.iter().position(
        |f| matches!(f, Value::Symbol(sym) if sym.namespace().is_none() && sym.name() == "&"),
    ) else {
        return Some(forms);
    };
    if pos + 1 != forms.len() - 1 {
        return None;
    }
    if !spec_valid_binding(forms[pos + 1]) {
        return None;
    }
    Some(&forms[..pos])
}

fn spec_valid_map(m: &Map) -> bool {
    for (k, v) in m.iter() {
        if let Value::Keyword(kw) = k {
            if !spec_valid_directive(kw, v) {
                return false;
            }
        } else if !spec_valid_binding(k) {
            return false;
        }
    }
    true
}

/// One directive's value shape (`::as`/`::or`/`::keys-form`/…).
/// Qualified `::ns-keys` take `::simple-syms-form` regardless of
/// kind (simple symbols only — stricter than unqualified `:keys`
/// idents or `:syms` symbols). Anything else in keyword position
/// fails `let`/`fn` spec (spec lines 44-50), or is `Unsupported
/// map directive` on the `loop` path. Qualified names are
/// keys/syms only (`:foo/strs` and `:foo/strs!` are not
/// `::ns-keys`).
fn spec_valid_directive(kw: &Keyword, v: &Value) -> bool {
    if kw.namespace().is_some() {
        return matches!(kw.name(), "keys" | "syms" | "keys!" | "syms!")
            && spec_valid_key_list(v, KeyKind::Strs);
    }
    match kw.name() {
        "as" | "select" | "excess" | "missing" | "defaults" | "all" => {
            matches!(v, Value::Symbol(sym) if is_local_name(sym))
        }
        "or" => matches!(v, Value::Map(_)),
        "keys" | "keys!" => spec_valid_key_list(v, KeyKind::Keys),
        "syms" | "syms!" => spec_valid_key_list(v, KeyKind::Syms),
        "strs" | "strs!" => spec_valid_key_list(v, KeyKind::Strs),
        "req!" => spec_valid_req(v),
        _ => false,
    }
}

/// `::req-bindings` (M2, upstream `core.specs.alpha`): a map of
/// local-name (or nested map form) to any key. The runtime splice
/// (`pb`) accepts vector locals too, and the expansion path takes
/// them — so the analyzer follows the runtime, not the spec text
/// (#112): any binding form passes here.
fn spec_valid_req(v: &Value) -> bool {
    let Value::Map(m) = v else {
        return false;
    };
    m.iter().all(|(local, _)| spec_valid_binding(local))
}

#[derive(Clone, Copy)]
enum KeyKind {
    Keys,
    Syms,
    Strs,
}

/// `::keys-form`: idents, then an optional `&` tail of anything
/// (dangling `&` is fine, probe25).
fn spec_valid_key_list(v: &Value, kind: KeyKind) -> bool {
    let Some(forms) = vector_shape_items(v) else {
        return false;
    };
    let tail = match forms.iter().position(
        |f| matches!(f, Value::Symbol(sym) if sym.namespace().is_none() && sym.name() == "&"),
    ) {
        None => &forms[..],
        Some(pos) => &forms[..pos],
    };
    tail.iter().all(|f| match kind {
        KeyKind::Keys => is_ident(f),
        KeyKind::Syms => matches!(f, Value::Symbol(_)),
        KeyKind::Strs => matches!(f, Value::Symbol(sym) if is_local_name(sym)),
    })
}

/// A destructure-body failure before wrapper attribution: the
/// `destructure`/`destvec*`/`destmap*` layer runs at macroexpansion,
/// so `Exception`/`IllegalArgumentException` render `Syntax error`
/// while anything else (our `ClassCastException`s) renders
/// `Unexpected error` (probe25).
struct WalkError {
    class: &'static str,
    message: String,
}

impl WalkError {
    fn unsupported(form: &Value) -> Self {
        Self {
            class: class::EXCEPTION,
            message: format!("Unsupported binding form: {}", str_of(form)),
        }
    }

    fn iseq(class_name: &str) -> Self {
        Self {
            class: class::ILLEGAL_ARG,
            message: format!("Don't know how to create ISeq from: {class_name}"),
        }
    }
}

/// Whether a value seqs (`keys`/key-lists accept these; anything
/// else is the `ISeq` error).
fn is_seqable(v: &Value) -> bool {
    matches!(
        v,
        Value::Nil
            | Value::List(_)
            | Value::Vector(_)
            | Value::MapEntry { .. }
            | Value::Map(_)
            | Value::Set(_)
            | Value::Str(_)
    )
}

/// The directive kind of a `:keys`-family keyword: `(tr bb)`
/// mapping plus requiredness. Qualified `:ns/keys` carry their ns
/// (probe25); anything else is `Unsupported map directive`.
struct KeyDirective {
    kind: KeyKind,
    ns: Option<String>,
    required: bool,
}

fn as_key_directive(kw: &Keyword) -> Option<KeyDirective> {
    let name = kw.name();
    let (base, required) = match name.strip_suffix('!') {
        Some(base) => (base, true),
        None => (name, false),
    };
    let kind = if base.starts_with("keys") {
        KeyKind::Keys
    } else if base.starts_with("syms") {
        KeyKind::Syms
    } else if base.starts_with("strs") {
        KeyKind::Strs
    } else {
        return None;
    };
    Some(KeyDirective {
        kind,
        ns: kw.namespace().map(str::to_owned),
        required,
    })
}

/// Validate one binding form the way `destructure` does (the `loop`
/// path and the shared semantic layer): symbols pass through, nested
/// shapes recurse, anything else is `Unsupported binding form`.
fn walk_binding(form: &Value) -> Result<(), WalkError> {
    match form {
        Value::Symbol(_) => Ok(()),
        Value::Map(m) => walk_map(m),
        _ => match vector_shape_items(form) {
            Some(items) => walk_seq(&items),
            None => Err(WalkError::unsupported(form)),
        },
    }
}

/// `destvec*`: fixed shapes recurse; `&` takes the next form as the
/// rest pattern (dangling `&` walks `nil`); after `&` only `:as`
/// may follow; `:as` validates its value (dangling walks `nil`)
/// immediately — before emission — then ends the scan.
fn walk_seq(forms: &[&Value]) -> Result<(), WalkError> {
    let mut rest = forms;
    let mut seen_rest = false;
    while let Some((first, tail)) = rest.split_first() {
        if is_amp(first) {
            let second = tail.first().copied().unwrap_or(&Value::Nil);
            walk_binding(second)?;
            rest = tail.get(1..).unwrap_or(&[]);
            seen_rest = true;
        } else if is_as(first) {
            // The `:as` value walks as a binding form right away
            // (probe31 `loop-as-*`): scalars fail here, while shapes
            // pass and die at the emitted `let`'s spec instead.
            let value = tail.first().copied().unwrap_or(&Value::Nil);
            walk_binding(value)?;
            return Ok(());
        } else {
            if seen_rest {
                return Err(WalkError {
                    class: class::EXCEPTION,
                    message: "Unsupported binding form, only :as can follow & parameter".to_owned(),
                });
            }
            walk_binding(first)?;
            rest = tail;
        }
    }
    Ok(())
}

fn is_amp(v: &Value) -> bool {
    matches!(v, Value::Symbol(sym) if sym.namespace().is_none() && sym.name() == "&")
}

fn is_as(v: &Value) -> bool {
    matches!(v, Value::Keyword(kw) if kw.namespace().is_none() && kw.name() == "as")
}

/// `destmap*` validation: directives known, `:or` seqable,
/// key-lists well-formed, `:or` semantics (`Multiple`, strict).
fn walk_map(m: &Map) -> Result<(), WalkError> {
    // `(keys defaults)`: maps are defaults, `nil` is none, empty
    // seqables proceed default-less, nonempty seqables throw on
    // their first element (`key` assumes entries), anything else is
    // the `ISeq` error (probe25).
    let or_map = match map_get(m, "or") {
        None | Some(Value::Nil) => None,
        Some(Value::Map(or)) => Some(or),
        Some(other) if !is_seqable(other) => {
            return Err(WalkError::iseq(&jvm_class_of(other)));
        }
        Some(other) => match seq_values(other).first() {
            None => None,
            Some(first) => {
                return Err(WalkError {
                    class: class::CLASS_CAST,
                    message: NativeError::cast(&jvm_class_of(first), "java.util.Map$Entry")
                        .into_inner()
                        .message,
                });
            }
        },
    };
    let mut sel: Vec<Value> = Vec::new();
    let mut bound_keys: HashMap<String, Value> = HashMap::new();
    for (k, v) in m.iter() {
        walk_map_pair(k, v, or_map, &mut sel, &mut bound_keys)?;
    }
    check_strict_or(m, or_map, &sel, &bound_keys)?;
    Ok(())
}

/// One map-pattern pair: outlet directives pass, key-list
/// directives validate, binding entries recurse.
fn walk_map_pair(
    k: &Value,
    v: &Value,
    or_map: Option<&Map>,
    sel: &mut Vec<Value>,
    bound_keys: &mut HashMap<String, Value>,
) -> Result<(), WalkError> {
    if let Value::Keyword(kw) = k {
        if is_outlet_directive(kw) {
            return Ok(());
        }
        if kw.namespace().is_none() && kw.name() == "req!" {
            return walk_req_list(v, or_map, sel, bound_keys);
        }
        let Some(dir) = as_key_directive(kw) else {
            return Err(WalkError {
                class: class::EXCEPTION,
                message: format!("Unsupported map directive: {kw}"),
            });
        };
        return walk_key_list(v, kw, &dir, or_map, sel, bound_keys);
    }
    walk_binding(k)?;
    // #112: the entry key unquotes (quoted spells the same key).
    // The `:or` check compares the form as written (#158).
    let key = unquote_key(v);
    sel.push(key.clone());
    if let Value::Symbol(local) = k {
        bound_keys.insert(local.name().to_owned(), key.clone());
        check_or_pair(or_map, local.name(), v, false)?;
    }
    Ok(())
}

/// One `:keys`-family list: pre-`&` idents bind (non-`Named`s are
/// the `ClassCastException`, probe25), a second `&` or a post-`&`
/// symbol is the `IllegalArgumentException`, post-`&` forms join
/// the selection.
fn walk_key_list(
    v: &Value,
    dir_kw: &Keyword,
    dir: &KeyDirective,
    or_map: Option<&Map>,
    sel: &mut Vec<Value>,
    bound_keys: &mut HashMap<String, Value>,
) -> Result<(), WalkError> {
    if !is_seqable(v) {
        return Err(WalkError::iseq(&jvm_class_of(v)));
    }
    let forms: Vec<Value> = seq_values(v);
    let mut preamp = true;
    for form in &forms {
        if is_amp(form) {
            if !preamp {
                return Err(WalkError {
                    class: class::ILLEGAL_ARG,
                    message: format!("& can only appear once in {dir_kw}"),
                });
            }
            preamp = false;
            continue;
        }
        if !preamp {
            if let Value::Symbol(sym) = form {
                return Err(WalkError {
                    class: class::ILLEGAL_ARG,
                    message: format!(
                        "'{}' - binding symbols can only appear before '&', use keys after",
                        sym.name()
                    ),
                });
            }
            // #112: post-`&` keys unquote like entries.
            sel.push(unquote_key(form));
            continue;
        }
        // `:strs` never casts (`core.clj:4556` bare `str`):
        // vectors and maps take the `push1`/`pb` recurse —
        // validated as nested patterns here, rejected by the spec
        // downstream (probe33 `strs-shape-*`). The `sel` push is
        // unobservable (the spec always fails after), so only
        // validation runs.
        if matches!(dir.kind, KeyKind::Strs)
            && (matches!(form, Value::Map(_)) || vector_shape_items(form).is_some())
        {
            walk_binding(form)?;
            continue;
        }
        let (key, local) = key_for(dir, form)?;
        sel.push(key.clone());
        bound_keys.insert(local.clone(), key.clone());
        check_or_pair(or_map, &local, &key, dir.required)?;
    }
    Ok(())
}

/// One `:req!` map (M2): locals validate as binding forms with
/// required `:or` semantics; keys join the selection, mirroring
/// the upstream splice back into the binding loop. The spec text
/// (`::req-bindings`) is local-names or nested map forms, but the
/// runtime splice takes vector locals too, like the expansion path
/// (#112); anything else is `Unsupported binding form`. Non-maps
/// take the expansion-stage errors the JVM hits: non-seqables the
/// `ISeq` error, other shapes the first element's `Map$Entry` cast
/// (probe `loop-req-num`, `loop-req-bad-full` on alpha10).
fn walk_req_list(
    v: &Value,
    or_map: Option<&Map>,
    sel: &mut Vec<Value>,
    bound_keys: &mut HashMap<String, Value>,
) -> Result<(), WalkError> {
    if !is_seqable(v) {
        return Err(WalkError::iseq(&jvm_class_of(v)));
    }
    if let Value::Map(m) = v {
        for (local, key) in m.iter() {
            walk_req_entry(local, key, or_map, sel, bound_keys)?;
        }
        return Ok(());
    }
    // A seqable non-map fails on `(key %)` of its first element —
    // only real map entries pass, and the reader builds no map-entry
    // literals, so every reachable shape takes the cast.
    match seq_values(v).into_iter().next() {
        Some(first) => Err(WalkError {
            class: class::CLASS_CAST,
            message: NativeError::cast(&jvm_class_of(&first), "java.util.Map$Entry")
                .into_inner()
                .message,
        }),
        None => Ok(()),
    }
}

fn walk_req_entry(
    local: &Value,
    key: &Value,
    or_map: Option<&Map>,
    sel: &mut Vec<Value>,
    bound_keys: &mut HashMap<String, Value>,
) -> Result<(), WalkError> {
    // #112: the key unquotes (quoted spells the same key); vector
    // locals validate as nested patterns (the runtime `pb` splice
    // accepts any binding form, like the expansion path). The `:or`
    // check compares the form as written (#158).
    let ukey = unquote_key(key);
    match local {
        Value::Symbol(sym) => {
            bound_keys.insert(sym.name().to_owned(), ukey.clone());
            check_or_pair(or_map, sym.name(), key, true)?;
        }
        Value::Map(nested) => walk_map(nested)?,
        _ if vector_shape_items(local).is_some() => walk_binding(local)?,
        _ => return Err(WalkError::unsupported(local)),
    }
    sel.push(ukey);
    Ok(())
}

/// `(tr bb)`: the lookup key plus the bound local for one
/// pre-`&` element. Only idents pass (probe32): keywords ride
/// the fast path untouched, and the local is the plain name.
/// Anything else takes the `(namespace x)` cast — except under
/// `:strs`, which never casts (see `key_cast_err`). The element's
/// own namespace wins over the directive namespace (upstream `xf`,
/// M2).
fn key_for(dir: &KeyDirective, form: &Value) -> Result<(Value, String), WalkError> {
    let (elem_ns, name) = match form {
        Value::Symbol(s) => (s.namespace().map(str::to_owned), s.name().to_owned()),
        Value::Keyword(k) => (k.namespace().map(str::to_owned), k.name().to_owned()),
        _ => return Err(key_cast_err(dir, form)),
    };
    let key = match dir.kind {
        KeyKind::Keys => {
            let ns = elem_ns.or(dir.ns.clone());
            Value::Keyword(Keyword::new(ns.as_deref(), &name))
        }
        KeyKind::Syms => {
            let ns = elem_ns.or(dir.ns.clone());
            Value::Symbol(Symbol::new(ns.as_deref(), &name))
        }
        KeyKind::Strs => Value::Str(Arc::from(name.as_str())),
    };
    Ok((key, name))
}

/// Upstream `xfk` (M2): the post-`&` key for one key-list element.
/// Under a namespaced `:keys`/`:keys!` directive a keyword takes
/// the directive namespace unless already qualified; anything else
/// passes through untouched.
fn post_amp_key(dir: &KeyDirective, form: &Value) -> Value {
    match form {
        Value::Keyword(k) if matches!(dir.kind, KeyKind::Keys) && dir.ns.is_some() => {
            let ns = k.namespace().map(str::to_owned).or(dir.ns.clone());
            Value::Keyword(Keyword::new(ns.as_deref(), k.name()))
        }
        _ => form.clone(),
    }
}

/// The `(tr bb)` cast for a non-ident key-list element
/// (probe32): `ClassCastException … to clojure.lang.Named`,
/// the helpful-NPE text for `nil` — but `:strs` never casts
/// and reports `Unsupported binding form` (str-rendered)
/// instead. `(namespace %)` always runs now (M2, upstream
/// 4610bb46 flipped `(or mkns (namespace %))` to `(or
/// (namespace %) mkns)`), so strings under a qualified directive
/// take the cast and `nil` always takes the `getNamespace` NPE
/// text (probe `s`, `nil` on alpha10).
fn key_cast_err(dir: &KeyDirective, form: &Value) -> WalkError {
    if matches!(dir.kind, KeyKind::Strs) {
        return WalkError::unsupported(form);
    }
    if matches!(form, Value::Nil) {
        return WalkError {
            class: class::NULL_POINTER,
            message: "Cannot invoke \"clojure.lang.Named.getNamespace()\" because \"x\" is null"
                .to_owned(),
        };
    }
    WalkError {
        class: class::CLASS_CAST,
        message: NativeError::cast(&jvm_class_of(form), "clojure.lang.Named")
            .into_inner()
            .message,
    }
}

/// `:or` semantics for one binding (compile time): `Multiple :or
/// defaults` when both the local and the key have defaults, `Can't
/// supply default value for required key` when a required binding
/// has either. The key arrives unquoted (#112) while `:or` keys
/// stay as written, so a quoted `:or` spelling matches too.
/// Upstream `push1` (`local-default?`/`key-default?`): `:or`
/// matches the local symbol or the key form as written (#158) —
/// callers pass the raw form, never the unquoted key.
fn check_or_pair(
    or_map: Option<&Map>,
    local: &str,
    key: &Value,
    required: bool,
) -> Result<(), WalkError> {
    let Some(or) = or_map else {
        return Ok(());
    };
    let local_key = Value::Symbol(Symbol::new(None, local));
    let has_local = or.get(&local_key).is_some();
    let has_key = or.get(key).is_some();
    if has_local && has_key {
        return Err(WalkError {
            class: class::EXCEPTION,
            message: format!(
                "Multiple :or defaults for same key: {} '{local}'",
                str_of(key)
            ),
        });
    }
    if required && (has_local || has_key) {
        return Err(WalkError {
            class: class::EXCEPTION,
            message: format!(
                "Can't supply default value for required key: {}",
                pr_of(key)
            ),
        });
    }
    Ok(())
}

/// Strict `:or` with `:select`/`:all` (compile time).
fn check_strict_or(
    m: &Map,
    or_map: Option<&Map>,
    sel: &[Value],
    bound_keys: &HashMap<String, Value>,
) -> Result<(), WalkError> {
    let has_select = map_get(m, "select").is_some();
    let has_all = map_get(m, "all").is_some();
    if !has_select && !has_all {
        return Ok(());
    }
    let Some(or) = or_map else {
        return Ok(());
    };
    let mut mapped: Vec<Value> = Vec::new();
    for (k, _) in or.iter() {
        if let Value::Symbol(sym) = k {
            match bound_keys.get(sym.name()) {
                Some(key) => mapped.push(key.clone()),
                None => {
                    return Err(WalkError {
                        class: class::ILLEGAL_ARG,
                        message: format!("symbol {sym} in :or does not refer to a binding"),
                    });
                }
            }
        } else {
            // #112: the selection holds unquoted keys, so quoted
            // `:or` spellings compare unquoted (bare symbols above
            // still resolve through bound locals only).
            mapped.push(unquote_key(k));
        }
    }
    let extras: Vec<&Value> = mapped.iter().filter(|k| !sel.contains(k)).collect();
    if !extras.is_empty() {
        let mut set = clj_value::new_set();
        for k in extras {
            set = set.insert((*k).clone());
        }
        return Err(WalkError {
            class: class::ILLEGAL_ARG,
            message: format!("keys {} appear only in :or", pr_of(&Value::Set(set))),
        });
    }
    Ok(())
}

fn map_get<'m>(m: &'m Map, key: &str) -> Option<&'m Value> {
    let probe = Value::Keyword(Keyword::new(None, key));
    m.get(&probe)
}

/// One raw `fn` clause (or `letfn` fnspec tail clause): parameter
/// forms plus pre/post/body forms.
struct RawClause<'v> {
    fixed: Vec<&'v Value>,
    rest: Option<&'v Value>,
    pre: Vec<Value>,
    post: Vec<Value>,
    body: Vec<&'v Value>,
}

/// One validated `letfn` fnspec: the name plus its tail forms.
struct FnSpec {
    name: String,
    tail: Vec<Value>,
}

/// One analyzed `letfn*` init, before the `ObjExpr` cast: a
/// `fn`-headed init parses straight to its closure, anything else
/// analyzes to an expression the post-body cast checks (A3).
enum LetFnInit {
    Fun(FunExpr),
    Expr(Expr),
}

impl<'a> Analyzer<'a> {
    /// The `CheckSpecs` failure for `let`/`fn`.
    fn spec_error(&self, macro_name: &str, span: Span) -> CompileError {
        self.err(
            class::EX_INFO,
            format!("Call to {macro_name} did not conform to spec."),
            span,
            Verb::Macroexpanding,
            Some(macro_name.to_owned()),
        )
    }

    /// A bare analyzer-builtin arity failure (`assert`/`time`
    /// reject 0 and 3+ args like the JVM macros — the `ArityException`
    /// escapes `eval` unwrapped).
    fn builtin_arity(&self, name: &'static str, argc: usize, span: Span) -> CompileError {
        CompileError::bare(
            "clojure.lang.ArityException",
            NativeError::arity(name, argc).into_inner().message,
            span,
            &self.source,
        )
    }

    /// `(assert x)` / `(assert x message)`: the expansion-time
    /// `*assert*` check reads the DYNAMIC value (a `binding` around
    /// `eval` disables, corpus 15 `:assert-bound-false-eval`); when
    /// off, the test is neither analyzed nor evaluated (the
    /// expansion is `nil`). A missing/unbound `*assert*` reads
    /// truthy (assert on).
    fn parse_assert(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let argc = items.len() - 1;
        if argc != 1 && argc != 2 {
            return Err(self.builtin_arity("clojure.core/assert", argc, span));
        }
        let sym = Symbol::new(Some("clojure.core"), "*assert*");
        let on = self
            .eval
            .registry
            .get("clojure.core")
            .and_then(|ns| ns.lookup(&sym))
            .is_none_or(|id| self.eval.deref_var(id, &sym).is_truthy());
        if !on {
            return Ok(Expr::Literal(Value::Nil));
        }
        let test = self.analyze(items[1], span, false)?;
        let message = match items.get(2) {
            Some(m) => Some(Box::new(self.analyze(m, span, false)?)),
            None => None,
        };
        Ok(Expr::Assert {
            test: Box::new(test),
            message,
            form: pr_to_string(items[1], &PrintOpts::pr()),
        })
    }

    /// `(time expr)`: exactly one arg (else the bare macro arity).
    /// The expr is not in tail position (it binds `ret#` on the
    /// JVM), so an inner `recur` is the tail-position error.
    fn parse_time(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let argc = items.len() - 1;
        if argc != 1 {
            return Err(self.builtin_arity("clojure.core/time", argc, span));
        }
        Ok(Expr::Time {
            expr: Box::new(self.analyze(items[1], span, false)?),
        })
    }

    /// `(with-out-str body…)`: the body never runs in tail position
    /// (the JVM expansion ends in `(str s#)`). `*out*` resolves like
    /// a `binding` target, so user shadows apply.
    fn parse_with_out_str(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let body = self.analyze_body(&items[1..], span, false)?;
        Ok(Expr::WithOutStr {
            out: Symbol::new(None, "*out*"),
            body,
        })
    }

    /// `(with-in-str s body…)`: `s` plus a body (the JVM macro
    /// takes `[s & body]`); the body never runs in tail position
    /// (the expansion ends in the `binding` pop). `*in*` resolves
    /// like a `binding` target, so user shadows apply.
    fn parse_with_in_str(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let argc = items.len() - 1;
        if argc < 1 {
            return Err(self.builtin_arity("clojure.core/with-in-str", argc, span));
        }
        Ok(Expr::WithInStr {
            input: Symbol::new(None, "*in*"),
            s: Box::new(self.analyze(items[1], span, false)?),
            body: self.analyze_body(&items[2..], span, false)?,
        })
    }

    /// `(lazy-seq* body…)`: a zero-arg `fn` boundary for `recur`
    /// (an empty `recur` re-invokes the body at force time, like the
    /// JVM closure; anything else is the mismatch error).
    fn parse_lazy_seq_star(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let saved_recur = self.recur;
        let saved_no_recur = self.no_recur;
        self.recur = Some(0);
        self.no_recur = false;
        let body = self.analyze_body(&items[1..], span, true);
        self.recur = saved_recur;
        self.no_recur = saved_no_recur;
        Ok(Expr::LazySeqStar { body: body? })
    }

    /// `(delay* body…)`: a zero-arg `fn` boundary for `recur` (like
    /// `lazy-seq*` — the body runs at force time, as the JVM's
    /// `(fn* [] …)` does). The display names the trapped closure.
    fn parse_delay_star(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let seq = self.eval.next_fn_seq();
        let display = format!("user/eval{}/delay--{seq}", self.eval_seq);
        let saved_recur = self.recur;
        let saved_no_recur = self.no_recur;
        self.recur = Some(0);
        self.no_recur = false;
        let body = self.analyze_body(&items[1..], span, true);
        self.recur = saved_recur;
        self.no_recur = saved_no_recur;
        Ok(Expr::DelayStar {
            display,
            body: body?,
        })
    }

    /// `(locking lockee body…)`: the lockee evaluates once (like the
    /// macro's `lockee#`); the body is an implicit `do` under the
    /// monitor. `recur` is fenced like `try` (the JVM expansion wraps
    /// the body in `try`/`finally`, which `recur` cannot cross).
    fn parse_locking(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let lockee = self.analyze(items[1], span, false)?;
        let saved_no_recur = self.no_recur;
        self.no_recur = true;
        let body = self.analyze_body(&items[2..], span, tail);
        self.no_recur = saved_no_recur;
        Ok(Expr::Locking {
            lockee: Box::new(lockee),
            body: body?,
        })
    }

    /// `(sync flags body…)`: the flags form drops before analysis
    /// (never evaluated — or analyzed); the body is an implicit `do`
    /// under the transaction. `recur` is fenced (the JVM expansion
    /// runs the body in a fresh `fn`, which `recur` cannot cross).
    fn parse_sync(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let saved_no_recur = self.no_recur;
        self.no_recur = true;
        let body = self.analyze_body(&items[2..], span, tail);
        self.no_recur = saved_no_recur;
        Ok(Expr::Sync { body: body? })
    }

    /// `(io! body…)`: an implicit `do` under the I/O gate (no `recur`
    /// fence — the JVM expansion is a transparent `if`/`do`).
    fn parse_io(&mut self, items: &[&Value], span: Span, tail: bool) -> Result<Expr, CompileError> {
        Ok(Expr::Io {
            body: self.analyze_body(&items[1..], span, tail)?,
        })
    }

    /// The lock-adjacent special forms (split from
    /// [`analyze_special`](Self::analyze_special) for the
    /// complexity budget — S19's `sync`/`io!` share the
    /// `locking` dispatch arm, the precedent they follow).
    fn parse_locking_or_tx(
        &mut self,
        op: SpecialOp,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        if matches!(op, SpecialOp::Locking) {
            self.parse_locking(items, span, tail)
        } else if matches!(op, SpecialOp::Sync) {
            self.parse_sync(items, span, tail)
        } else {
            self.parse_io(items, span, tail)
        }
    }

    /// `(case* ge shift mask default imap switch-type test-type
    /// skip?)`: the `case` macro's expansion shape. Wrong-typed
    /// fields are the JVM's deterministic casts (`case_star_cast`);
    /// missing/extra fields, NPE-accident shapes, and
    /// macro-unreachable out-of-range ints keep the spec error
    /// (frozen wedge, T1192 FX1). `default`/`then`s inherit tail
    /// position; `test`s analyze as exprs (evaluated on a hash hit,
    /// like the JVM `test.emit`); `skip` defaults to empty.
    fn parse_case_star(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        if items.len() != 8 && items.len() != 9 {
            return Err(self.spec_error("clojure.core/case*", span));
        }
        let (shift, mask, imap, test) = self.parse_case_star_head(items, span)?;
        let skip = self.parse_case_star_skip(items, span)?;
        let expr = self.analyze(items[1], span, false)?;
        let default = self.analyze(items[4], span, tail)?;
        let cases = self.parse_case_star_cases(imap, span, tail)?;
        Ok(Expr::CaseStar {
            expr: Box::new(expr),
            shift,
            mask,
            default: Box::new(default),
            cases,
            skip,
            test,
        })
    }

    /// A `case*` integer slot (`i32`-ranged `Int`, else `None`).
    fn case_star_int(v: &Value) -> Option<i32> {
        match v {
            Value::Int(i) => i32::try_from(*i).ok(),
            _ => None,
        }
    }

    /// A `case*` wrong-type field: the JVM's deterministic cast
    /// (T1192 FX1 probes), not the spec error. Missing/extra fields
    /// and NPE-accident shapes keep the spec error (frozen wedge).
    fn case_star_cast(&self, v: &Value, to: &'static str, span: Span) -> CompileError {
        let message = NativeError::cast(&jvm_class_of(v), to).into_inner().message;
        self.err(
            class::CLASS_CAST,
            message,
            span,
            Verb::Compiling,
            Some("case*".to_owned()),
        )
    }

    /// The `case*` header: shift, mask, the case map, the test kind,
    /// and the switch-kind check (all spec errors).
    fn parse_case_star_head<'v>(
        &self,
        items: &[&'v Value],
        span: Span,
    ) -> Result<(i32, i32, &'v Map, CaseTest), CompileError> {
        let (Some(shift), Some(mask)) =
            (Self::case_star_int(items[2]), Self::case_star_int(items[3]))
        else {
            // Wrong-typed (non-`Int`) fields are the JVM's `Number`
            // cast; out-of-range `Int`s are macro-unreachable (the
            // macro routes big constants to hash mode) and keep the
            // spec error — the JVM shape there is unknowable (every
            // probe crashes elsewhere first).
            let bad = [items[2], items[3]]
                .into_iter()
                .find(|v| !matches!(v, Value::Int(_)));
            match bad {
                Some(v) => return Err(self.case_star_cast(v, "java.lang.Number", span)),
                None => return Err(self.spec_error("clojure.core/case*", span)),
            }
        };
        let Value::Map(imap) = items[5] else {
            return Err(self.case_star_cast(items[5], "java.util.Map", span));
        };
        let test = self.parse_case_star_test(items[7], span)?;
        if !matches!(items[6], Value::Keyword(_)) {
            return Err(self.case_star_cast(items[6], "clojure.lang.Keyword", span));
        }
        if !matches!(items[6], Value::Keyword(kw) if kw.namespace().is_none() && matches!(kw.name(), "sparse" | "compact"))
        {
            return Err(self.spec_error("clojure.core/case*", span));
        }
        Ok((shift, mask, imap, test))
    }

    /// The `case*` test kind (`:int` / `:hash-equiv` /
    /// `:hash-identity`, unqualified).
    fn parse_case_star_test(&self, v: &Value, span: Span) -> Result<CaseTest, CompileError> {
        match v {
            Value::Keyword(kw) if kw.namespace().is_none() => match kw.name() {
                "int" => Ok(CaseTest::Int),
                "hash-equiv" => Ok(CaseTest::HashEquiv),
                "hash-identity" => Ok(CaseTest::HashIdentity),
                _ => Err(self.spec_error("clojure.core/case*", span)),
            },
            _ => Err(self.case_star_cast(v, "clojure.lang.Keyword", span)),
        }
    }

    /// The optional `case*` skip set (absent or `nil` means empty).
    fn parse_case_star_skip(&self, items: &[&Value], span: Span) -> Result<Vec<i32>, CompileError> {
        let mut skip = Vec::new();
        // (Nested `if`, not a let-chain: MSRV 1.85.)
        if let Some(s) = items.get(8).copied() {
            if !matches!(s, Value::Nil) {
                let Value::Set(set) = s else {
                    return Err(self.case_star_cast(s, "java.util.Set", span));
                };
                for v in set.iter() {
                    let Some(k) = Self::case_star_int(v) else {
                        return Err(self.spec_error("clojure.core/case*", span));
                    };
                    skip.push(k);
                }
            }
        }
        Ok(skip)
    }

    /// The `case*` case map: dispatch keys to `(test, then)` (test
    /// constants are literal data — probed: symbols, lists, and
    /// defined-var names all match as data, never evaluated; only
    /// the `then` is an expression).
    fn parse_case_star_cases(
        &mut self,
        imap: &Map,
        span: Span,
        tail: bool,
    ) -> Result<Vec<(i32, Expr, Expr)>, CompileError> {
        let mut cases = Vec::new();
        for (k, v) in imap.iter() {
            let Some(key) = Self::case_star_int(k) else {
                // Wrong-typed keys are the JVM's `Number` cast;
                // out-of-range `Int` keys are macro-unreachable (big
                // constants route to hash mode) and keep the spec
                // error.
                if matches!(k, Value::Int(_)) {
                    return Err(self.spec_error("clojure.core/case*", span));
                }
                return Err(self.case_star_cast(k, "java.lang.Number", span));
            };
            let Value::Vector(pair) = v else {
                return Err(self.spec_error("clojure.core/case*", span));
            };
            let mut pair_forms = pair.iter();
            let (Some(test_form), Some(then_form), None) =
                (pair_forms.next(), pair_forms.next(), pair_forms.next())
            else {
                return Err(self.spec_error("clojure.core/case*", span));
            };
            let test_expr = Expr::Literal(test_form.clone());
            let then_expr = self.analyze(then_form, span, tail)?;
            cases.push((key, test_expr, then_expr));
        }
        Ok(cases)
    }

    /// Attribute a destructure-body failure: `Syntax error` for
    /// `Exception`/`IllegalArgumentException` causes, `Unexpected
    /// error` otherwise (mirroring the macroexpand `catch` ladder).
    fn walk_error(&self, err: WalkError, span: Span, op: &str) -> CompileError {
        let mut e = self.err(
            err.class,
            err.message,
            span,
            Verb::Macroexpanding,
            Some(op.to_owned()),
        );
        e.unexpected = !matches!(
            err.class,
            class::EXCEPTION | class::ILLEGAL_ARG | class::ILLEGAL_STATE | class::EX_INFO
        );
        e
    }

    /// `(let [pat init …] body…)` / `(loop …)`: `let` runs the deep
    /// spec first, then the shared walk (as-written wrapper);
    /// `loop` runs asserts, the walk (`loop` wrapper), the emission
    /// re-validation (inner-`let` spec), or the `loop*` star checks.
    fn parse_let(
        &mut self,
        op: &str,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let is_loop = op == "loop" || op == "loop*";
        let forms = self.let_binding_forms(op, items, span)?;
        let pats: Vec<&Value> = forms.iter().step_by(2).copied().collect();
        self.validate_let_pats(op, &pats, span)?;
        let saved_recur = self.recur;
        if is_loop {
            self.recur = None;
        }
        let base = self.locals.len();
        let mut bindings_out = Vec::with_capacity(pats.len());
        for i in (0..forms.len()).step_by(2) {
            let init = self.analyze(forms[i + 1], span, false)?;
            let validated = as_pat(forms[i]).expect("patterns validated before building");
            let (pat, names) = self.build_pattern(validated, span)?;
            self.locals.extend(names);
            bindings_out.push((pat, init));
        }
        // A `loop` body resets the try/recur fence; the inits keep
        // the outer fence (probe30 `fence-loop-*`). The body is
        // self-tail (like `fn`): a `loop` in non-tail position still
        // takes `recur` (probe30 `fence-loop-in-catch`).
        let saved_no_recur = self.no_recur;
        if is_loop {
            self.recur = Some(pats.len());
            self.no_recur = false;
        }
        let body_tail = if is_loop { true } else { tail };
        let body = self.analyze_body(&items[2..], span, body_tail)?;
        self.locals.truncate(base);
        self.recur = saved_recur;
        self.no_recur = saved_no_recur;
        if is_loop {
            Ok(Expr::Loop {
                bindings: bindings_out,
                body,
            })
        } else {
            Ok(Expr::Let {
                bindings: bindings_out,
                body,
            })
        }
    }

    /// The `let`/`loop`/`let*`/`loop*` bindings vector,
    /// shape-checked: star shapes fail `Bad binding form`, macro
    /// shapes route to the spec/assert checks.
    fn let_binding_forms<'v>(
        &self,
        op: &str,
        items: &[&'v Value],
        span: Span,
    ) -> Result<Vec<&'v Value>, CompileError> {
        let is_loop = op == "loop" || op == "loop*";
        let is_star = matches!(op, "let*" | "loop*");
        let bindings = items.get(1).copied().unwrap_or(&Value::Nil);
        let Some(forms) = vector_shape_items(bindings) else {
            if is_star {
                return Err(self.star_shape_error(op, "expected vector", span));
            }
            return Err(self.shape_error(is_loop, "a vector for its binding", span, op));
        };
        // MSRV 1.85: `usize::is_multiple_of` is 1.87+.
        #[allow(clippy::manual_is_multiple_of)]
        if forms.len() % 2 != 0 {
            if is_star {
                return Err(self.star_shape_error(
                    op,
                    "expected matched symbol expression pairs",
                    span,
                ));
            }
            return Err(self.shape_error(
                is_loop,
                "an even number of forms in binding vector",
                span,
                op,
            ));
        }
        Ok(forms)
    }

    /// Binding-pattern validation for `parse_let`: the star
    /// checks, the `loop` checks, or the `let` spec+walk.
    fn validate_let_pats(&self, op: &str, pats: &[&Value], span: Span) -> Result<(), CompileError> {
        if matches!(op, "let*" | "loop*") {
            self.validate_star_bindings(pats, span, op)
        } else if op == "loop" {
            self.validate_loop_bindings(pats, span, op)
        } else {
            self.validate_let_bindings(pats, span, op)
        }
    }

    /// `let` validation: deep spec first, then the shared walk
    /// (as-written wrapper).
    fn validate_let_bindings(
        &self,
        pats: &[&Value],
        span: Span,
        op: &str,
    ) -> Result<(), CompileError> {
        for pat in pats {
            if !spec_valid_binding(pat) {
                return Err(self.spec_error("clojure.core/let", span));
            }
        }
        for pat in pats {
            if let Err(e) = walk_binding(pat) {
                return Err(self.walk_error(e, span, op));
            }
        }
        Ok(())
    }

    /// `loop` validation: asserts, the walk (`loop` wrapper), then
    /// the emission re-validation (inner-`let` spec) for
    /// destructuring patterns, or the `loop*` star checks for
    /// all-symbol bindings.
    fn validate_loop_bindings(
        &self,
        pats: &[&Value],
        span: Span,
        op: &str,
    ) -> Result<(), CompileError> {
        for pat in pats {
            if let Err(e) = walk_binding(pat) {
                return Err(self.walk_error(e, span, op));
            }
        }
        if pats.iter().any(|p| !matches!(p, Value::Symbol(_))) {
            for pat in pats {
                if !spec_valid_binding(pat) {
                    return Err(self.spec_error("clojure.core/let", span));
                }
            }
            return Ok(());
        }
        self.validate_star_bindings(pats, span, "loop*")
    }

    /// Raw `let*`/`loop*` binding checks (probe33): symbols only
    /// (`Bad binding form, expected symbol, got: <pr>`), no `&`,
    /// no qualified names.
    fn validate_star_bindings(
        &self,
        pats: &[&Value],
        span: Span,
        op: &str,
    ) -> Result<(), CompileError> {
        for pat in pats {
            let Value::Symbol(sym) = pat else {
                return Err(self.err(
                    class::ILLEGAL_ARG,
                    format!("Bad binding form, expected symbol, got: {}", pr_of(pat)),
                    span,
                    Verb::Compiling,
                    Some(op.to_owned()),
                ));
            };
            if is_amp(pat) {
                return Err(self.err(
                    class::RUNTIME,
                    "Can't use & as a local binding".to_owned(),
                    span,
                    Verb::Compiling,
                    Some(op.to_owned()),
                ));
            }
            if sym.namespace().is_some() {
                return Err(self.err(
                    class::RUNTIME,
                    format!("Can't let qualified name: {sym}"),
                    span,
                    Verb::Compiling,
                    Some(op.to_owned()),
                ));
            }
        }
        Ok(())
    }

    /// A raw `let*`/`loop*` bindings-shape error (`Compiling`).
    fn star_shape_error(&self, op: &str, what: &str, span: Span) -> CompileError {
        self.err(
            class::ILLEGAL_ARG,
            format!("Bad binding form, {what}"),
            span,
            Verb::Compiling,
            Some(op.to_owned()),
        )
    }

    /// The bindings-shape failure: spec for `let`, assert-args for
    /// `loop` (volatile `in NS:LINE` suffix dropped, §E4).
    fn shape_error(&self, is_loop: bool, what: &str, span: Span, op: &str) -> CompileError {
        if is_loop {
            self.err(
                class::ILLEGAL_ARG,
                format!("loop requires {what}"),
                span,
                Verb::Macroexpanding,
                Some(op.to_owned()),
            )
        } else {
            self.spec_error("clojure.core/let", span)
        }
    }

    /// Build a validated pattern, analyzing `:or` defaults in the
    /// current scope. Names come from the shared
    /// [`pattern_names`] order.
    fn build_pattern(
        &mut self,
        shape: Pat<'_>,
        span: Span,
    ) -> Result<(Pattern, Vec<String>), CompileError> {
        let pat = match shape {
            Pat::Symbol(sym) => Pattern::Symbol(sym.name().to_owned()),
            Pat::Vector(forms) => self.build_seq(&forms, span)?,
            Pat::Map(m) => self.build_map(m, span)?,
        };
        let names = pattern_names(&pat);
        Ok((pat, names))
    }

    /// Split a validated sequential pattern (`&` rest, trailing
    /// `:as`; junk after `:as` ignored like the walk).
    fn build_seq(&mut self, forms: &[&Value], span: Span) -> Result<Pattern, CompileError> {
        let mut fixed = Vec::new();
        let mut rest_pat = None;
        let mut as_sym = None;
        let mut rest = forms;
        while let Some((first, tail)) = rest.split_first() {
            if is_amp(first) {
                let second = tail.first().copied().unwrap_or(&Value::Nil);
                let validated = as_pat(second).expect("patterns validated before building");
                let (pat, _) = self.build_pattern(validated, span)?;
                rest_pat = Some(Box::new(pat));
                rest = tail.get(1..).unwrap_or(&[]);
            } else if is_as(first) {
                if let Some(Value::Symbol(sym)) = tail.first() {
                    as_sym = Some(sym.name().to_owned());
                }
                break;
            } else {
                let validated = as_pat(first).expect("patterns validated before building");
                let (pat, _) = self.build_pattern(validated, span)?;
                fixed.push(pat);
                rest = tail;
            }
        }
        Ok(Pattern::Vec(VecPattern {
            fixed,
            rest: rest_pat,
            as_sym,
        }))
    }

    /// Build a validated map pattern, analyzing `:or` defaults.
    fn build_map(&mut self, m: &Map, span: Span) -> Result<Pattern, CompileError> {
        let mut acc = MapAcc::default();
        for (k, v) in m.iter() {
            self.build_map_pair(k, v, span, &mut acc)?;
        }
        let or = self.build_or(m, span)?;
        Ok(Pattern::Map(MapPattern {
            entries: acc.entries,
            keys: acc.keys,
            extra_keys: acc.extra_keys,
            as_sym: acc.as_sym,
            or,
            select: acc.select,
            all: acc.all,
            excess: acc.excess,
            missing: acc.missing,
        }))
    }

    /// One map-pattern pair: outlet directive, key-list
    /// directive, or a binding entry.
    fn build_map_pair(
        &mut self,
        k: &Value,
        v: &Value,
        span: Span,
        acc: &mut MapAcc,
    ) -> Result<(), CompileError> {
        if let Value::Keyword(kw) = k {
            // `:req!` (M2): each entry binds required with its key
            // (unquoted, #112), exactly like the upstream splice
            // back into the binding loop (validation ran already, so
            // only maps reach the builder; anything else never
            // does). Hoisted ahead of the chain so the chain keeps
            // its master shape; every non-`req!` keyword takes the
            // false edge past the block, keeping its brace
            // attributed.
            if kw.namespace().is_none() && kw.name() == "req!" {
                return self.build_req_entries(v, span, acc);
            }
            if is_outlet_directive(kw) {
                set_outlet(kw, v, acc);
            } else if let Some(dir) = as_key_directive(kw) {
                build_keys(v, &dir, &mut acc.keys, &mut acc.extra_keys);
            }
            return Ok(());
        }
        let validated = as_pat(k).expect("patterns validated before building");
        let (pat, _) = self.build_pattern(validated, span)?;
        // #112: the entry key unquotes (quoted spells the same key).
        push_entry(&mut acc.entries, pat, unquote_key(v), v.clone(), false);
        Ok(())
    }

    /// The analyzed `:or` defaults (eager at runtime). Keys stay as
    /// written: `:or` matches forms (upstream `push1`), so a quoted
    /// spelling matches only the quoted entry key (#158).
    fn build_or(&mut self, m: &Map, span: Span) -> Result<Vec<(Value, Expr)>, CompileError> {
        let mut or = Vec::new();
        if let Some(Value::Map(or_map)) = map_get(m, "or") {
            for (k, default) in or_map.iter() {
                or.push((k.clone(), self.analyze(default, span, false)?));
            }
        }
        Ok(or)
    }
}

/// Push one `(pattern, lookup-key, or-key, required?)` entry.
fn push_entry(
    acc: &mut Vec<(Pattern, Value, Value, bool)>,
    pat: Pattern,
    key: Value,
    or_key: Value,
    required: bool,
) {
    acc.push((pat, key, or_key, required));
}

impl<'a> Analyzer<'a> {
    /// Build one `:req!` map into required entries (M2).
    fn build_req_entries(
        &mut self,
        v: &Value,
        span: Span,
        acc: &mut MapAcc,
    ) -> Result<(), CompileError> {
        if let Value::Map(req) = v {
            for (local, key) in req.iter() {
                let validated = as_pat(local).expect("patterns validated before building");
                let (pat, _) = self.build_pattern(validated, span)?;
                push_entry(&mut acc.entries, pat, unquote_key(key), key.clone(), true);
            }
            return Ok(());
        }
        Ok(())
    }
}

/// A map pattern under construction.
#[derive(Default)]
struct MapAcc {
    entries: Vec<(Pattern, Value, Value, bool)>,
    keys: Vec<KeyBinding>,
    extra_keys: Vec<Value>,
    as_sym: Option<String>,
    select: Option<String>,
    all: Option<String>,
    excess: Option<String>,
    missing: Option<String>,
}

/// Unqualified `:as`/`:or`/outlet directives (validation ran
/// already; unknown directives never reach the builder).
fn is_outlet_directive(kw: &Keyword) -> bool {
    kw.namespace().is_none()
        && matches!(
            kw.name(),
            "as" | "or" | "select" | "excess" | "missing" | "defaults" | "all"
        )
}

/// Record one outlet binding (`:or`/`:defaults` carry no local).
fn set_outlet(kw: &Keyword, v: &Value, acc: &mut MapAcc) {
    let slot = match kw.name() {
        "as" => &mut acc.as_sym,
        "select" => &mut acc.select,
        "all" => &mut acc.all,
        "excess" => &mut acc.excess,
        "missing" => &mut acc.missing,
        _ => return,
    };
    *slot = as_symbol(v).map(|s| s.name().to_owned());
}

/// One validated key-list: pre-`&` elements bind, post-`&`
/// elements join the selection.
fn build_keys(
    v: &Value,
    dir: &KeyDirective,
    keys: &mut Vec<KeyBinding>,
    extra_keys: &mut Vec<Value>,
) {
    let mut preamp = true;
    for form in seq_values(v) {
        if is_amp(&form) {
            preamp = false;
            continue;
        }
        if !preamp {
            // Upstream `xfk` (M2): under a namespaced `:keys`/`:keys!`
            // directive an unqualified post-`&` key takes the directive
            // namespace, a qualified one keeps its own. Post-`&` keys
            // unquote like entries (#112).
            extra_keys.push(unquote_key(&post_amp_key(dir, &form)));
            continue;
        }
        if let Ok((key, local)) = key_for(dir, &form) {
            keys.push(KeyBinding {
                local,
                key,
                required: dir.required,
            });
        }
    }
}

/// The unrealistically generous `seq` for key-lists (only the
/// seqable check matters; elements drive validation).
fn seq_values(v: &Value) -> Vec<Value> {
    match v {
        Value::Nil => Vec::new(),
        Value::List(l) => l.iter().cloned().collect(),
        v @ (Value::Vector(_) | Value::MapEntry { .. }) => vector_shape_items(v)
            .expect("vector shape checked")
            .into_iter()
            .cloned()
            .collect(),
        Value::Map(m) => m
            .iter()
            .map(|(k, v)| Value::Vector(clj_value::Vector::from_vec(vec![k.clone(), v.clone()])))
            .collect(),
        Value::Set(s) => s.iter().cloned().collect(),
        Value::Str(s) => s.chars().map(Value::Char).collect(),
        _ => Vec::new(),
    }
}

/// Split parameter forms into fixed + optional rest (`&` must be
/// second-to-last; `fn` params take no `:as`). `None` = spec failure.
fn split_params<'v>(forms: &[&'v Value]) -> Option<(Vec<&'v Value>, Option<&'v Value>)> {
    match forms.iter().position(|f| is_amp(f)) {
        None => {
            if forms.iter().any(|f| matches!(f, Value::Keyword(_))) {
                return None;
            }
            Some((forms.to_vec(), None))
        }
        Some(pos) => {
            if pos + 1 != forms.len() - 1 {
                return None;
            }
            if forms[..pos].iter().any(|f| matches!(f, Value::Keyword(_))) {
                return None;
            }
            Some((forms[..pos].to_vec(), Some(forms[pos + 1])))
        }
    }
}

impl<'a> Analyzer<'a> {
    /// `(fn name? params body* | name? (params body*)+)`: the deep
    /// spec first, then per-clause validation in source order
    /// (20-count, parameter walk under the emitted `let`'s wrapper,
    /// body, dup checks; fixed-vs-variadic last).
    fn parse_fn(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let (name, raw) = self.parse_fn_front(&items[1..], span)?;
        self.finish_fn(name, &raw, span)
    }

    /// `(fn …)`/`(bound-fn …)` front: the optional name plus the
    /// analyzed clauses (callers wrap: `fn` in `Fn`/`LetFn`,
    /// `bound-fn` in `BoundFn`).
    fn parse_fn_front<'v>(
        &mut self,
        mut rest: &[&'v Value],
        span: Span,
    ) -> Result<(Option<String>, Vec<RawClause<'v>>), CompileError> {
        let mut name: Option<String> = None;
        match rest.split_first() {
            Some((Value::Symbol(sym), tail))
                if tail.first().is_some_and(|f| {
                    matches!(f, Value::List(_)) || vector_shape_items(f).is_some()
                }) =>
            {
                if !is_fn_name(sym) {
                    return Err(self.spec_error("clojure.core/fn", span));
                }
                name = Some(sym.name().to_owned());
                rest = tail;
            }
            _ => {}
        }
        let owned = self.collect_fn_clauses(rest, span)?;
        let slices: Vec<&[&Value]> = owned.iter().map(Vec::as_slice).collect();
        let raw = self.parse_clause_slices(&slices, span)?;
        Ok((name, raw))
    }

    /// `(fn* name? params body* | name? (params body*)+)`: raw
    /// `FnExpr` semantics — no spec, no destructuring, no
    /// pre/post. A leading symbol is always the name (any symbol;
    /// the simple name binds for self-recursion, probe33
    /// `fn*-selfsimple`); a leading vector is the single clause,
    /// otherwise every item is a list clause. `nil` passes the
    /// seq cast and dies at the arglist-NPE like `()` (probe33
    /// `fn*-nil`); anything else is the `ISeq` CCE.
    fn parse_fn_star(&mut self, items: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let mut rest = &items[1..];
        let mut name: Option<String> = None;
        if let Some((Value::Symbol(sym), tail)) = rest.split_first() {
            name = Some(sym.name().to_owned());
            rest = tail;
        }
        let owned: Vec<Vec<&Value>> = if rest.is_empty() {
            Vec::new()
        } else if rest
            .first()
            .is_some_and(|f| vector_shape_items(f).is_some())
        {
            vec![rest.to_vec()]
        } else {
            let mut out = Vec::with_capacity(rest.len());
            for clause in rest {
                match clause {
                    Value::List(list) => out.push(list.iter().collect()),
                    Value::Nil => out.push(Vec::new()),
                    other => {
                        return Err(self.fn_star_error(
                            class::CLASS_CAST,
                            &NativeError::cast(&jvm_class_of(other), "clojure.lang.ISeq")
                                .into_inner()
                                .message,
                            span,
                        ));
                    }
                }
            }
            out
        };
        let slices: Vec<&[&Value]> = owned.iter().map(Vec::as_slice).collect();
        let raw = self.parse_star_clauses(&slices, span)?;
        self.finish_fn(name, &raw, span)
    }

    /// Shared `fn`/`fn*` tail: display name, the self-recursion
    /// slot, per-clause analysis, the `LetFn` wrap for names.
    /// Shared `fn`/`fn*` tail: display name, the self-recursion
    /// slot, and per-clause analysis. Returns the fun plus the
    /// self-recursion binding (named callers wrap in `LetFn`).
    fn build_fun(
        &mut self,
        name: Option<String>,
        raw: &[RawClause],
        span: Span,
    ) -> Result<(FunExpr, Option<(String, usize)>), CompileError> {
        let seq = self.eval.next_fn_seq();
        let display = match &name {
            Some(n) => format!("user/eval{}/{}--{seq}", self.eval_seq, n),
            None => format!("user/eval{}/fn--{seq}", self.eval_seq),
        };
        let base = self.locals.len();
        let name_slot = if let Some(n) = &name {
            self.locals.push(n.clone());
            Some(self.locals.len() - 1)
        } else {
            None
        };
        let arities = self.analyze_arities(raw, span)?;
        self.locals.truncate(base);
        let fun = FunExpr { display, arities };
        match (name, name_slot) {
            (Some(n), Some(slot)) => Ok((fun, Some((n, slot)))),
            _ => Ok((fun, None)),
        }
    }

    /// Shared `fn`/`fn*` tail: the `LetFn` wrap for names.
    fn finish_fn(
        &mut self,
        name: Option<String>,
        raw: &[RawClause],
        span: Span,
    ) -> Result<Expr, CompileError> {
        let (fun, bound) = self.build_fun(name, raw, span)?;
        // A named `fn` is a one-function `letfn`: the frame extends
        // first, the closure captures it, then the slot fills
        // (self-recursion). A `let` wrap would capture the
        // pre-extension frame instead.
        Ok(match bound {
            Some((n, slot)) => Expr::LetFn {
                fns: vec![(n, fun)],
                body: vec![Expr::Local(slot)],
            },
            None => Expr::Fn(fun),
        })
    }

    /// One raw `fn*` clause per slice: the params vector plus the
    /// whole body (no pre/post extraction — a leading map is a
    /// body form, probe33 `fn*-pre`). Missing params are the
    /// arglist-NPE; non-vectors are the `IPersistentVector` CCE.
    fn parse_star_clauses<'v>(
        &self,
        clauses: &[&[&'v Value]],
        span: Span,
    ) -> Result<Vec<RawClause<'v>>, CompileError> {
        let mut raw = Vec::with_capacity(clauses.len());
        for clause in clauses {
            let Some((params, body)) = clause.split_first() else {
                return Err(self.fn_star_error(
                    class::NULL_POINTER,
                    "Cannot invoke \"clojure.lang.IPersistentVector.count()\" because \"arglist\" is null",
                    span,
                ));
            };
            let Some(param_forms) = vector_shape_items(params) else {
                return Err(self.fn_star_error(
                    class::CLASS_CAST,
                    &NativeError::cast(&jvm_class_of(params), "clojure.lang.IPersistentVector")
                        .message,
                    span,
                ));
            };
            let (fixed, rest) = self.star_params(&param_forms, span)?;
            raw.push(RawClause {
                fixed,
                rest,
                pre: Vec::new(),
                post: Vec::new(),
                body: body.to_vec(),
            });
        }
        Ok(raw)
    }

    /// Raw `fn*` parameter splitting (probe33): a trailing `&`
    /// drops (fixed arity); `&` as the rest name, or after the
    /// rest, is `Invalid parameter list`; a plain parameter after
    /// the rest is `Unexpected parameter`.
    fn star_params<'v>(
        &self,
        forms: &[&'v Value],
        span: Span,
    ) -> Result<(Vec<&'v Value>, Option<&'v Value>), CompileError> {
        let mut fixed = Vec::new();
        let mut rest = None;
        let mut i = 0;
        while i < forms.len() {
            let form = forms[i];
            if is_amp(form) {
                if rest.is_some() {
                    return Err(self.fn_star_error(class::RUNTIME, "Invalid parameter list", span));
                }
                i += 1;
                match forms.get(i) {
                    None => {}
                    Some(f) if is_amp(f) => {
                        return Err(self.fn_star_error(
                            class::RUNTIME,
                            "Invalid parameter list",
                            span,
                        ));
                    }
                    Some(f) => {
                        self.check_star_param(f, span)?;
                        rest = Some(*f);
                    }
                }
            } else if rest.is_some() {
                return Err(self.fn_star_error(class::RUNTIME, "Unexpected parameter", span));
            } else {
                self.check_star_param(form, span)?;
                fixed.push(form);
            }
            i += 1;
        }
        Ok((fixed, rest))
    }

    /// One raw `fn*` parameter: symbols only, unqualified.
    fn check_star_param(&self, form: &Value, span: Span) -> Result<(), CompileError> {
        let Value::Symbol(sym) = form else {
            return Err(self.fn_star_error(class::ILLEGAL_ARG, "fn params must be Symbols", span));
        };
        if sym.namespace().is_some() {
            return Err(self.fn_star_error(
                class::RUNTIME,
                &format!("Can't use qualified name as parameter: {sym}"),
                span,
            ));
        }
        Ok(())
    }

    /// A `fn*` (`Compiling`) shape error.
    fn fn_star_error(&self, class: &'static str, message: &str, span: Span) -> CompileError {
        self.err(
            class,
            message.to_owned(),
            span,
            Verb::Compiling,
            Some("fn*".to_owned()),
        )
    }

    /// Collect one clause per arity: a leading vector is the
    /// single clause, otherwise every item is a list clause
    /// (non-lists and empty arities fail spec).
    fn collect_fn_clauses<'v>(
        &self,
        rest: &[&'v Value],
        span: Span,
    ) -> Result<Vec<Vec<&'v Value>>, CompileError> {
        if rest
            .first()
            .is_some_and(|f| vector_shape_items(f).is_some())
        {
            return Ok(vec![rest.to_vec()]);
        }
        let mut out = Vec::with_capacity(rest.len());
        for clause in rest {
            let Value::List(list) = clause else {
                return Err(self.spec_error("clojure.core/fn", span));
            };
            out.push(list.iter().collect());
        }
        if out.is_empty() {
            return Err(self.spec_error("clojure.core/fn", span));
        }
        Ok(out)
    }

    /// One clause's parameter walk (the fn macro's
    /// `maybe-destructured` runs before `FnExpr`): carries the
    /// emitted `let`'s wrapper.
    fn walk_clause(&self, clause: &RawClause, span: Span) -> Result<(), CompileError> {
        for pat in clause.fixed.iter().chain(clause.rest.iter()) {
            if let Err(e) = walk_binding(pat) {
                return Err(self.walk_error(e, span, "clojure.core/let"));
            }
        }
        Ok(())
    }

    /// Per-clause `fn*` validation in source order: 20-count, the
    /// parameter walk, the body, then the dup/variadic checks;
    /// fixed-vs-variadic last (probe32 `fn-*`).
    fn analyze_arities(
        &mut self,
        raw: &[RawClause],
        span: Span,
    ) -> Result<Vec<Arity>, CompileError> {
        let mut arities = Vec::with_capacity(raw.len());
        let mut fixed_counts = std::collections::HashSet::new();
        let mut variadic_fixed: Option<usize> = None;
        for clause in raw {
            if clause.fixed.len() > 20 {
                return Err(self.err(
                    class::RUNTIME,
                    "Can't specify more than 20 params".to_owned(),
                    span,
                    Verb::Compiling,
                    Some("fn*".to_owned()),
                ));
            }
            self.walk_clause(clause, span)?;
            let arity = self.analyze_clause(clause, span)?;
            if clause.rest.is_some() {
                if variadic_fixed.is_some() {
                    return Err(self.err(
                        class::RUNTIME,
                        "Can't have more than 1 variadic overload".to_owned(),
                        span,
                        Verb::Compiling,
                        Some("fn*".to_owned()),
                    ));
                }
                variadic_fixed = Some(clause.fixed.len());
            } else if !fixed_counts.insert(clause.fixed.len()) {
                return Err(self.err(
                    class::RUNTIME,
                    "Can't have 2 overloads with same arity".to_owned(),
                    span,
                    Verb::Compiling,
                    Some("fn*".to_owned()),
                ));
            }
            arities.push(arity);
        }
        if variadic_fixed.is_some_and(|vfixed| fixed_counts.iter().any(|c| *c > vfixed)) {
            return Err(self.err(
                class::RUNTIME,
                "Can't have fixed arity function with more params than variadic function"
                    .to_owned(),
                span,
                Verb::Compiling,
                Some("fn*".to_owned()),
            ));
        }
        Ok(arities)
    }

    /// Analyze one clause: parameters, pre-asserts, body (non-tail
    /// under `:post`, which wraps it in the `%` `let`), post-asserts.
    fn analyze_clause(&mut self, clause: &RawClause, span: Span) -> Result<Arity, CompileError> {
        let base = self.locals.len();
        let mut fixed = Vec::with_capacity(clause.fixed.len());
        for pat in &clause.fixed {
            let validated = as_pat(pat).expect("patterns validated before building");
            let (pattern, names) = self.build_pattern(validated, span)?;
            self.locals.extend(names);
            fixed.push(pattern);
        }
        let mut rest_pat = None;
        if let Some(rest) = clause.rest {
            let validated = as_pat(rest).expect("patterns validated before building");
            let (pattern, names) = self.build_pattern(validated, span)?;
            self.locals.extend(names);
            rest_pat = Some(pattern);
        }
        let saved_recur = self.recur;
        let count = clause.fixed.len() + usize::from(clause.rest.is_some());
        self.recur = Some(count);
        // A `fn` boundary resets the try/recur fence (`Compiler.java`
        // binds `NO_RECUR` null per `FnExpr`), for the whole clause
        // (pre/post included, probe30 `fence-fn-*`).
        let saved_no_recur = self.no_recur;
        self.no_recur = false;
        let mut body = Vec::with_capacity(clause.pre.len() + clause.body.len() + 1);
        for cond in &clause.pre {
            body.push(self.analyze_assert(cond, span)?);
        }
        if clause.post.is_empty() {
            body.extend(self.analyze_body(&clause.body, span, true)?);
        } else {
            let inner = self.analyze_body(&clause.body, span, false)?;
            let pct = self.locals.len();
            self.locals.push("%".to_owned());
            let mut post = Vec::with_capacity(clause.post.len() + 1);
            for cond in &clause.post {
                post.push(self.analyze_assert(cond, span)?);
            }
            post.push(Expr::Local(pct));
            self.locals.pop();
            body.push(Expr::Let {
                bindings: vec![(Pattern::Symbol("%".to_owned()), Expr::Do(inner))],
                body: post,
            });
        }
        self.locals.truncate(base);
        self.recur = saved_recur;
        self.no_recur = saved_no_recur;
        Ok(Arity {
            fixed,
            rest: rest_pat,
            body,
        })
    }

    /// `(assert cond)`: falsy throws
    /// `AssertionError | Assert failed: COND`.
    fn analyze_assert(&mut self, cond: &Value, span: Span) -> Result<Expr, CompileError> {
        let message = format!("Assert failed: {}", pr_of(cond));
        Ok(Expr::If {
            test: Box::new(self.analyze(cond, span, false)?),
            then: Box::new(Expr::Literal(Value::Nil)),
            els: Box::new(Expr::Throw(Box::new(Expr::NewThrowable {
                class: clj_value::ThrowClass::AssertionError,
                message,
            }))),
        })
    }

    /// `(letfn [name tail …] body…)`: fnspecs seq (the `ISeq` error
    /// for non-seqables), name (the `letfn*` error), tails as `fn`
    /// clauses (the `fn`/`fn*` wrappers).
    fn parse_letfn(
        &mut self,
        op: &str,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let bindings = items.get(1).copied().unwrap_or(&Value::Nil);
        // Fence 4b E1e: fnspecs seq like the expansion's `mapcat`
        // (any seqable — lists, maps, sets, strings — not just
        // vectors); only non-seqables take the `ISeq` error.
        if !is_seqable(bindings) {
            return Err(self.err(
                class::ILLEGAL_ARG,
                format!(
                    "Don't know how to create ISeq from: {}",
                    jvm_class_of(bindings)
                ),
                span,
                Verb::Macroexpanding,
                Some(op.to_owned()),
            ));
        }
        let specs: Vec<Value> = seq_values(bindings);
        // Validate every fnspec before analyzing any tail.
        let validated = self.validate_fnspecs(&specs, span, op)?;
        let base = self.locals.len();
        for spec in &validated {
            self.locals.push(spec.name.clone());
        }
        let mut fns = Vec::with_capacity(validated.len());
        for spec in &validated {
            let seq = self.eval.next_fn_seq();
            let display = format!("user/eval{}/{}--{seq}", self.eval_seq, spec.name);
            let tail_refs: Vec<&Value> = spec.tail.iter().collect();
            let owned = self.collect_fn_clauses(&tail_refs, span)?;
            let slices: Vec<&[&Value]> = owned.iter().map(Vec::as_slice).collect();
            let raw = self.parse_clause_slices(&slices, span)?;
            let arities = self.analyze_arities(&raw, span)?;
            fns.push((spec.name.clone(), FunExpr { display, arities }));
        }
        let body = self.analyze_body(&items[2..], span, tail)?;
        self.locals.truncate(base);
        Ok(Expr::LetFn { fns, body })
    }

    /// Validate every fnspec (`(map first fnspecs)`: strict
    /// `seq`, then a plain-symbol head) before any tail analysis.
    fn validate_fnspecs(
        &self,
        specs: &[Value],
        span: Span,
        op: &str,
    ) -> Result<Vec<FnSpec>, CompileError> {
        let mut validated = Vec::with_capacity(specs.len());
        for spec in specs {
            if !is_seqable(spec) {
                return Err(self.err(
                    class::ILLEGAL_ARG,
                    format!("Don't know how to create ISeq from: {}", jvm_class_of(spec)),
                    span,
                    Verb::Macroexpanding,
                    Some(op.to_owned()),
                ));
            }
            let parts = seq_values(spec);
            validated.push(self.fnspec_head(&parts, span)?);
        }
        Ok(validated)
    }

    /// One fnspec head: a plain symbol (anything else is the
    /// `letfn*` error).
    fn fnspec_head(&self, parts: &[Value], span: Span) -> Result<FnSpec, CompileError> {
        let head = parts.first();
        let Some(Value::Symbol(name)) = head else {
            return Err(self.err(
                class::ILLEGAL_ARG,
                format!(
                    "Bad binding form, expected symbol, got: {}",
                    head.map_or("null".to_owned(), value_of)
                ),
                span,
                Verb::Compiling,
                Some("letfn*".to_owned()),
            ));
        };
        // Qualified names hit the `let` pre-seed rule, not the
        // binding-form check (probe31 `letfn-qual-name`).
        if name.namespace().is_some() {
            return Err(self.err(
                class::RUNTIME,
                format!("Can't let qualified name: {name}"),
                span,
                Verb::Compiling,
                Some("letfn*".to_owned()),
            ));
        }
        // Non-symbols and qualified names are rejected above;
        // `&` is a valid function name (mangled, probe33
        // `letfn-amp`).
        Ok(FnSpec {
            name: name.name().to_owned(),
            tail: parts.get(1..).unwrap_or(&[]).to_vec(),
        })
    }

    /// Parse `fn` clause slices (shared by `fn` and `letfn`):
    /// spec-checked into raw clauses.
    fn parse_clause_slices<'v>(
        &mut self,
        clauses: &[&[&'v Value]],
        span: Span,
    ) -> Result<Vec<RawClause<'v>>, CompileError> {
        let mut raw = Vec::with_capacity(clauses.len());
        for clause in clauses {
            let Some((params, body)) = clause.split_first() else {
                return Err(self.spec_error("clojure.core/fn", span));
            };
            let Some(param_forms) = vector_shape_items(params) else {
                return Err(self.spec_error("clojure.core/fn", span));
            };
            let Some((fixed, rest_pat)) = split_params(&param_forms) else {
                return Err(self.spec_error("clojure.core/fn", span));
            };
            for pat in fixed.iter().chain(rest_pat.iter()) {
                if !spec_valid_binding(pat) {
                    return Err(self.spec_error("clojure.core/fn", span));
                }
            }
            // A leading map is pre/post only when a body form follows
            // it; a lone map is the body (probe31 `fn-lone-*`).
            let (pre, post, body) = match body.split_first() {
                Some((Value::Map(prepost), rest)) if !rest.is_empty() => {
                    let pre = self.prepost_conds(prepost, "pre", span)?;
                    let post = self.prepost_conds(prepost, "post", span)?;
                    (pre, post, rest.to_vec())
                }
                _ => (Vec::new(), Vec::new(), body.to_vec()),
            };
            raw.push(RawClause {
                fixed,
                rest: rest_pat,
                pre,
                post,
                body,
            });
        }
        Ok(raw)
    }

    /// One pre/post condition list: `nil` is empty, seqables are
    /// conditions, anything else is the `fn*` `ISeq` error. Map
    /// values seq to whole entry vectors (evaluated in map order),
    /// never flattened to k/v conds (probe33 `pre-map-*`).
    fn prepost_conds(
        &self,
        prepost: &Map,
        which: &str,
        span: Span,
    ) -> Result<Vec<Value>, CompileError> {
        match map_get(prepost, which) {
            None | Some(Value::Nil) => Ok(Vec::new()),
            Some(Value::Map(m)) => Ok(m
                .iter()
                .map(|(k, v)| Value::Vector(Vector::from_vec(vec![k.clone(), v.clone()])))
                .collect()),
            Some(v) if is_seqable(v) => Ok(seq_refs(v).into_iter().cloned().collect()),
            Some(other) => Err(self.err(
                class::ILLEGAL_ARG,
                format!(
                    "Don't know how to create ISeq from: {}",
                    jvm_class_of(other)
                ),
                span,
                Verb::Compiling,
                Some("fn*".to_owned()),
            )),
        }
    }

    /// Rewrite an expansion call (`instance?`/`identical?`/`new`/
    /// array-map/`monitor-*`/`letfn*`) before specials.
    fn try_fx4_host(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Option<Expr>, CompileError> {
        let op = crate::fx4_host::classify(items);
        self.lower_host(op, items, span, tail)
    }

    fn lower_host(
        &mut self,
        op: crate::fx4_host::HostOp<'_>,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Option<Expr>, CompileError> {
        match op {
            crate::fx4_host::HostOp::None => Ok(None),
            crate::fx4_host::HostOp::SeqQ(arg) => Ok(Some(self.host_call("seq?", &[arg], span)?)),
            crate::fx4_host::HostOp::Identical(a, b) => {
                Ok(Some(self.host_call("identical?", &[a, b], span)?))
            }
            crate::fx4_host::HostOp::NewObject => Ok(Some(self.fresh_object())),
            crate::fx4_host::HostOp::Pam(inner) => {
                Ok(Some(self.host_call("fx4-pam-create", &[inner], span)?))
            }
            rest => self.lower_host_rest(rest, items, span, tail),
        }
    }

    /// The monitor/`letfn*`/arity-error branch of [`Analyzer::lower_host`]
    /// (split for the complexity budget).
    fn lower_host_rest(
        &mut self,
        op: crate::fx4_host::HostOp<'_>,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Option<Expr>, CompileError> {
        match op {
            crate::fx4_host::HostOp::MonitorEnter(arg) => {
                Ok(Some(self.monitor_call("fx4-monitor-enter", arg, span)?))
            }
            crate::fx4_host::HostOp::MonitorExit(arg) => {
                Ok(Some(self.monitor_call("fx4-monitor-exit", arg, span)?))
            }
            crate::fx4_host::HostOp::LetFnStar => {
                Ok(Some(self.parse_letfn_star(items, span, tail)?))
            }
            crate::fx4_host::HostOp::InstanceArity(n) => {
                Err(self.host_arity_err("clojure.core/instance?", n, span))
            }
            crate::fx4_host::HostOp::IdenticalArity(n) => {
                Err(self.host_arity_err("clojure.core/identical?", n, span))
            }
            // Handled by `lower_host` (the single caller partitions
            // the ops).
            _ => unreachable!("host-op partition"),
        }
    }

    /// A compile-time arity error for a direct qualified
    /// `instance?`/`identical?` call: the runtime text, at compile
    /// time, non-bare so it surfaces as `SourceError::Compile`
    /// (bare errors escape as uncaught throwables).
    fn host_arity_err(&self, name: &'static str, argc: usize, span: Span) -> CompileError {
        let message = crate::error::NativeError::arity(name, argc)
            .into_inner()
            .message;
        self.err(
            "clojure.lang.ArityException",
            message,
            span,
            Verb::Compiling,
            None,
        )
    }

    fn fresh_object(&self) -> Expr {
        Expr::Literal(Value::Extension(Arc::new(
            crate::fx4_host::Fx4Object::fresh(),
        )))
    }

    /// Invoke a core var by qualified symbol (skips the private check).
    fn host_call(&mut self, name: &str, args: &[&Value], span: Span) -> Result<Expr, CompileError> {
        let head = Expr::Global(self.synth_var(name));
        let mut analyzed = Vec::with_capacity(args.len());
        for arg in args {
            analyzed.push(self.analyze(arg, span, false)?);
        }
        Ok(Expr::Invoke {
            head: Box::new(head),
            args: analyzed,
            span,
        })
    }

    /// A `monitor-enter`/`monitor-exit` call with the (expanded)
    /// form quoted alongside (fence 4b E1r–E1y — the nil-message
    /// builder reads shapes post-expansion, like the JVM).
    /// Expansion failure falls back to the raw form (analysis
    /// re-raises it properly downstream).
    fn monitor_call(&mut self, name: &str, arg: &Value, span: Span) -> Result<Expr, CompileError> {
        let expanded = crate::macros::macroexpand_full(self.eval, arg, span, &self.source);
        let form = expanded.as_ref().unwrap_or(arg);
        let quoted = Value::List(List::from_vec(vec![
            Value::Symbol(Symbol::new(None, "quote")),
            (*form).clone(),
        ]));
        self.host_call(name, &[form, &quoted], span)
    }

    /// `(letfn* [name (fn name …) …] body…)`, lowered to [`Expr::LetFn`].
    /// The inits analyze before the body, but the `ObjExpr` cast
    /// waits until after it: upstream `LetFnExpr` analyzes inits
    /// then body in `Parser.parse` and only casts in `emit`, so a
    /// body analysis error beats a non-`fn` init CCE (fence 4b A3).
    fn parse_letfn_star(
        &mut self,
        items: &[&Value],
        span: Span,
        tail: bool,
    ) -> Result<Expr, CompileError> {
        let bindings = items.get(1).copied().unwrap_or(&Value::Nil);
        let pairs = self.letfn_star_pairs(bindings, span)?;
        let base = self.locals.len();
        for (name, _) in &pairs {
            self.locals.push(name.clone());
        }
        let inits = self.letfn_star_inits(&pairs, span)?;
        let body = self.analyze_body(&items[2..], span, tail)?;
        let mut fns = Vec::with_capacity(pairs.len());
        for ((name, form), init) in pairs.iter().zip(inits) {
            fns.push((name.clone(), self.letfn_star_cast(form, init, span)?));
        }
        self.locals.truncate(base);
        Ok(Expr::LetFn { fns, body })
    }

    fn letfn_star_pairs(
        &self,
        bindings: &Value,
        span: Span,
    ) -> Result<Vec<(String, Value)>, CompileError> {
        let forms = self.letfn_star_vec(bindings, span)?;
        if forms.len() % 2 != 0 {
            return Err(self.letfn_star_err(
                class::ILLEGAL_ARG,
                "letfn* requires an even number of forms in binding vector",
                span,
            ));
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i + 1 < forms.len() {
            out.push((self.letfn_star_name(forms[i], span)?, forms[i + 1].clone()));
            i += 2;
        }
        Ok(out)
    }

    fn letfn_star_vec<'v>(
        &self,
        bindings: &'v Value,
        span: Span,
    ) -> Result<Vec<&'v Value>, CompileError> {
        vector_shape_items(bindings).ok_or_else(|| {
            self.letfn_star_err(
                class::ILLEGAL_ARG,
                "letfn* requires a vector for its binding",
                span,
            )
        })
    }

    fn letfn_star_name(&self, v: &Value, span: Span) -> Result<String, CompileError> {
        let Value::Symbol(sym) = v else {
            return Err(self.letfn_star_err(
                class::ILLEGAL_ARG,
                &format!("Bad binding form, expected symbol, got: {}", value_of(v)),
                span,
            ));
        };
        if sym.namespace().is_some() {
            return Err(self.letfn_star_err(
                class::RUNTIME,
                &format!("Can't let qualified name: {sym}"),
                span,
            ));
        }
        Ok(sym.name().to_owned())
    }

    fn letfn_star_err(&self, class: &'static str, message: &str, span: Span) -> CompileError {
        self.err(
            class,
            message.to_owned(),
            span,
            Verb::Compiling,
            Some("letfn*".to_owned()),
        )
    }

    fn letfn_star_inits(
        &mut self,
        pairs: &[(String, Value)],
        span: Span,
    ) -> Result<Vec<LetFnInit>, CompileError> {
        let mut inits = Vec::with_capacity(pairs.len());
        for (_, form) in pairs {
            inits.push(self.letfn_star_init(form, span)?);
        }
        Ok(inits)
    }

    fn letfn_star_init(&mut self, form: &Value, span: Span) -> Result<LetFnInit, CompileError> {
        // Fence 4b E1k/E1r: `letfn*` inits compile as expressions,
        // then cast to `ObjExpr`. A `fn`-headed init parses
        // directly (a named `fn` would otherwise analyze to the
        // self-recursion `LetFn` wrap, hiding the `FunExpr`);
        // anything else analyzes unchecked — the cast waits for the
        // body (A3).
        if let Value::List(list) = form {
            let items: Vec<Value> = list.iter().cloned().collect();
            let rest = strip_fn_head(&items);
            if rest.len() != items.len() {
                let refs: Vec<&Value> = rest.iter().collect();
                let (name, raw) = self.parse_fn_front(&refs, span)?;
                let (mut fun, _) = self.build_fun(None, &raw, span)?;
                if let Some(n) = name {
                    fun.display = fun.display.replacen("/fn--", &format!("/{n}--"), 1);
                }
                return Ok(LetFnInit::Fun(fun));
            }
        }
        Ok(LetFnInit::Expr(self.analyze(form, span, false)?))
    }

    /// The emit-time `ObjExpr` cast (fence 4b A3): a `fn` result
    /// qualifies (macros expanding to `fn` included) or the CCE
    /// names the source form's class (the compiler-internal
    /// `NumberExpr`/`LetExpr` kinds have no port counterpart).
    fn letfn_star_cast(
        &self,
        form: &Value,
        init: LetFnInit,
        span: Span,
    ) -> Result<FunExpr, CompileError> {
        match init {
            LetFnInit::Fun(fun) => Ok(fun),
            LetFnInit::Expr(Expr::Fn(fun)) => Ok(fun),
            LetFnInit::Expr(_) => Err(self.err(
                class::CLASS_CAST,
                NativeError::cast(&jvm_class_of(form), "clojure.lang.Compiler$ObjExpr")
                    .into_inner()
                    .message,
                span,
                Verb::Compiling,
                Some("fn*".to_owned()),
            )),
        }
    }
}

fn strip_fn_head(items: &[Value]) -> &[Value] {
    match items.first() {
        Some(Value::Symbol(s))
            if s.name() == "fn" && s.namespace().is_none_or(|ns| ns == "clojure.core") =>
        {
            &items[1..]
        }
        _ => items,
    }
}

/// Borrowed `seq` for pre/post conditions (`nil` and maps are
/// filtered by the caller; anything else seqable-but-unlisted
/// yields no condition refs).
fn seq_refs(v: &Value) -> Vec<&Value> {
    match v {
        Value::List(l) => l.iter().collect(),
        v @ (Value::Vector(_) | Value::MapEntry { .. }) => {
            vector_shape_items(v).expect("vector shape checked")
        }
        Value::Set(s) => s.iter().collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_values_non_seqable_is_empty() {
        // Defensive arm: non-seqables yield no key-list elements.
        assert!(seq_values(&Value::Int(1)).is_empty());
    }

    #[test]
    #[should_panic(expected = "host-op partition")]
    fn lower_host_rest_rejects_outer_ops() {
        // Defensive arm: the single caller (`lower_host`)
        // partitions the ops, so outer-handled ops never arrive
        // here (only a dispatch bug could send one).
        let source: Arc<str> = Arc::from("NO_SOURCE_PATH");
        let mut eval = crate::ns::boot();
        let mut analyzer = Analyzer::new(&source, &mut eval, 0);
        let items: Vec<&Value> = vec![];
        let _ = analyzer.lower_host_rest(
            crate::fx4_host::HostOp::None,
            &items,
            Span::new(1, 1),
            false,
        );
    }

    fn unqualified_bb() -> Value {
        Value::Keyword(Keyword::new(None, "bb"))
    }

    #[test]
    fn unquote_key_strips_quote_forms() {
        // #112: `(quote k)` denotes `k`; anything else passes
        // through untouched (bare symbols stay literal).
        let quoted = quote_form(&Value::Symbol(Symbol::new(None, "k")));
        assert_eq!(unquote_key(&quoted), Value::Symbol(Symbol::new(None, "k")));
        let bare = Value::Symbol(Symbol::new(None, "k"));
        assert_eq!(unquote_key(&bare), bare);
        let kw = Value::Keyword(Keyword::new(None, "k"));
        assert_eq!(unquote_key(&kw), kw);
        let call = Value::List(List::from_vec(vec![
            Value::Symbol(Symbol::new(None, "f")),
            Value::Symbol(Symbol::new(None, "k")),
        ]));
        assert_eq!(unquote_key(&call), call);
        // None arm of the guard: a 2-list key whose head is not a
        // symbol (here `((quote q) v)`) passes through untouched.
        let nested = Value::List(List::from_vec(vec![
            quote_form(&Value::Symbol(Symbol::new(None, "q"))),
            Value::Symbol(Symbol::new(None, "v")),
        ]));
        assert_eq!(unquote_key(&nested), nested);
        // Length-guard false edge: a list key with length != 2
        // (here a 1-list) passes through untouched.
        let singleton = Value::List(List::from_vec(vec![Value::Symbol(Symbol::new(None, "k"))]));
        assert_eq!(unquote_key(&singleton), singleton);
    }

    #[test]
    fn spec_valid_req_takes_vector_locals() {
        // #112: the runtime splice accepts vector locals (the spec
        // text is map/local-only, but the expansion path takes
        // vectors, so the analyzer follows the runtime).
        let pair = Value::Vector(Vector::from_vec(vec![
            Value::Symbol(Symbol::new(None, "a")),
            Value::Symbol(Symbol::new(None, "b")),
        ]));
        let req = Value::Map(new_map().insert(pair, Value::Keyword(Keyword::new(None, "pair"))));
        assert!(spec_valid_req(&req));
        let bad =
            Value::Map(new_map().insert(Value::Int(5), Value::Keyword(Keyword::new(None, "a"))));
        assert!(!spec_valid_req(&bad));
    }

    #[test]
    fn build_req_entries_ignores_vacuous_non_map() {
        // Defensive: validation rejects non-map `:req!` first, so
        // only maps reach the builder; a vacuous value still builds
        // clean (covers the skip path).
        let source: Arc<str> = Arc::from("NO_SOURCE_PATH");
        let mut eval = crate::ns::boot();
        let mut analyzer = Analyzer::new(&source, &mut eval, 0);
        let m = new_map().insert(
            Value::Keyword(Keyword::new(None, "req!")),
            Value::Vector(Vector::from_vec(vec![])),
        );
        let pat = analyzer
            .build_map(&m, Span::new(1, 1))
            .expect("vacuous :req! builds");
        assert!(
            matches!(pat, Pattern::Map(ref mp) if mp.entries.is_empty()),
            "vacuous :req! builds no entries"
        );
    }

    #[test]
    fn post_amp_key_qualifies_under_ns_keys() {
        // Upstream `xfk` (M2): under a namespaced `:keys`/`:keys!`
        // directive an unqualified post-`&` key takes the directive
        // namespace, a qualified one keeps its own; anything else
        // passes through untouched.
        let dir = KeyDirective {
            kind: KeyKind::Keys,
            ns: Some("foo".to_owned()),
            required: false,
        };
        assert_eq!(
            post_amp_key(&dir, &unqualified_bb()),
            Value::Keyword(Keyword::new(Some("foo"), "bb"))
        );
        assert_eq!(
            post_amp_key(&dir, &Value::Keyword(Keyword::new(Some("other"), "bb"))),
            Value::Keyword(Keyword::new(Some("other"), "bb"))
        );
        let syms = KeyDirective {
            kind: KeyKind::Syms,
            ns: Some("foo".to_owned()),
            required: false,
        };
        assert_eq!(post_amp_key(&syms, &unqualified_bb()), unqualified_bb());
        let plain = KeyDirective {
            kind: KeyKind::Keys,
            ns: None,
            required: false,
        };
        assert_eq!(post_amp_key(&plain, &unqualified_bb()), unqualified_bb());
    }

    #[test]
    fn form_force_recur_is_internal() {
        // Defense in depth: an escaping `recur` (impossible — the
        // analyzer cages every one) surfaces as an internal error.
        let source: Arc<str> = Arc::from("NO_SOURCE_PATH");
        let err = form_force_err(&source, Signal::Recur(vec![]), Span::new(1, 1));
        assert!(err.bare);
        assert_eq!(err.class, super::class::RUNTIME);
        assert!(err.message.contains("recur escaped forcing a form"));
    }
}
