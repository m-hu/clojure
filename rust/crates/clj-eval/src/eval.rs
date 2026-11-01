// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! The tree-walking [`Evaluator`]: [`Expr`]s become values, throwing
//! [`Signal`]s. Covers invocation (§E7), `try`, `loop`/`recur`,
//! `def`/`set!`/`binding`, and the full destructure engine (§E12).
//!
//! S4 ships no prelude: tests inject [`NativeFn`](crate::ext::NativeFn)
//! doubles; `ex-info` et al. arrive with the S5 core subset.

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

use clj_reader::{Reader, ReaderOpts, Span};
use clj_value::{
    Delay, Keyword, List, Map, PrimitiveVector, Promise, Symbol, ThrowClass, Throwable, Value, Var,
    Vector, new_map,
};

use crate::coll::native_nth;
use crate::env::{Bindings, Env, SharedRegistry};
use crate::error::{CompileError, NativeError, Signal, jvm_class_of, span_of};
use crate::expr::{Analyzer, CaseTest, Catch, Expr, MapPattern, Pattern, pattern_names};
use crate::ext::{Closure, NativeFn, NsHandle, as_closure, as_native};

/// `eval_source` failures: reader errors, analyzer errors, uncaught
/// throwables, and (unreachable by construction) escaping `recur`s.
#[derive(Debug)]
pub enum SourceError {
    Reader(clj_reader::ReaderError),
    Compile(CompileError),
    Uncaught(Throwable),
    Internal(String),
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reader(e) => write!(f, "{e}"),
            Self::Compile(e) => write!(f, "{e}"),
            Self::Uncaught(t) => write!(
                f,
                "{} | {}",
                t.class.jvm_name(),
                t.message.as_deref().unwrap_or("")
            ),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for SourceError {}

/// An uncaught top-level signal (an escaping `recur` is
/// impossible — every loop/fn frame catches its own — and
/// surfaces as an internal error, never a panic).
fn uncaught(s: Signal) -> SourceError {
    match s {
        Signal::Throw(t) => SourceError::Uncaught(t),
        Signal::Recur(_) => SourceError::Internal("recur escaped its loop or fn".to_owned()),
    }
}

/// The throwable of a signal leaving a `try` region (`None`
/// for `recur`, which the analyzer fences from crossing `try`
/// — expr.rs `no_recur` — so `try` bodies only ever throw).
fn try_thrown(signal: Signal) -> Option<Throwable> {
    match signal {
        Signal::Throw(t) => Some(t),
        Signal::Recur(_) => None,
    }
}

/// The body of a top-level `do`-headed list (head unqualified,
/// mirroring the analyzer's special-form check), `None` for any
/// other form or a bare `(do)`. Shared with AOT planning, which
/// validates the same pipeline.
pub(crate) fn split_top_do(form: &Value) -> Option<Vec<&Value>> {
    let Value::List(list) = form else {
        return None;
    };
    let mut items = list.iter();
    match items.next() {
        Some(Value::Symbol(sym)) if sym.namespace().is_none() && sym.name() == "do" => {}
        _ => return None,
    }
    let rest: Vec<&Value> = items.collect();
    if rest.is_empty() { None } else { Some(rest) }
}

/// `Number.intValue` for `:int` `case*` dispatch (`None` for
/// non-numbers — characters included, which miss like on the JVM).
/// `Int` truncates; `Double` narrows exactly like Java (`as`
/// saturates and maps `NaN` to 0, same as `(int)`).
fn case_int_value(v: &Value) -> Option<i32> {
    match v {
        Value::Int(i) => Some(*i as i32),
        Value::Double(d) => Some(*d as i32),
        _ => None,
    }
}

/// `case*` dispatch-key masking (a zero mask skips the shift).
fn case_shift_mask(key: i32, shift: i32, mask: i32) -> i32 {
    if mask == 0 {
        key
    } else {
        key.wrapping_shr((shift & 31) as u32) & mask
    }
}

/// The evaluator: shared lineage state (namespace registry plus
/// identity counters), per-thread bindings, load path, spawned
/// future threads, and the ambient native call span. `Send + Sync`
/// (the shared state travels by `Arc`); per-thread evaluators fork
/// over one lineage (see [`fork_for_thread`](Self::fork_for_thread)).
pub struct Evaluator {
    pub registry: SharedRegistry,
    bindings: Bindings,
    source: Arc<str>,
    future_threads: Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>,
    agent_system: Arc<crate::concurrency::AgentSystem>,
    tap_system: Arc<crate::concurrency::TapSystem>,
    native_call_span: Option<Span>,
    /// Sends buffered while an agent action fn runs (the JVM's
    /// thread-local `nested`): released on success, discarded on
    /// failure. `None` outside an action fn.
    pub(crate) agent_outbox: Option<Vec<(clj_value::Agent, clj_value::QueuedAction)>>,
    /// The running STM transaction (S19): `Some` inside `sync`,
    /// `None` outside. Lives outside `Bindings` (§1h) — same-thread
    /// `bound-fn` calls observe it, cross-thread conveyance drops it.
    pub(crate) tx: Option<crate::stm::Transaction>,
    /// Analyze recursion budget (`Some` on the AOT plan path only;
    /// the run path stays unbounded). Set once on the throwaway
    /// planning evaluator, read by every `Analyzer` it builds.
    pub(crate) analyze_depth_limit: Option<usize>,
    /// `*out*` flushes served (the `*flush-on-newline*` test hook:
    /// fresh evaluators read 0; per-evaluator, so parallel-safe).
    flush_count: u64,
    /// The print driver renders through the booted defaults only
    /// (user table entries bypassed) while set — `print-simple`'s
    /// object half. Save/restore around the region, never nested.
    pub(crate) print_defaults_only: bool,
    /// The `*load-path*` value at the last `data_readers.clj`
    /// autoload scan (`None` before the first read): an unchanged
    /// value skips the scan outright (upstream scans once at
    /// startup; the port re-scans only when the path moves).
    pub(crate) data_reader_scan_path: Option<Vec<String>>,
    /// Mapping files already merged into the `*data-readers*` root
    /// (`/`-joined paths): files merge once each, so a root the
    /// user replaced afterwards is never re-merged.
    pub(crate) loaded_data_reader_files: HashSet<String>,
    /// The pristine boot `inst`/`uuid` default natives (#94r):
    /// the snapshot compares the live root values against these by
    /// reference, so a replaced canonical skips read-side
    /// validation (`None` before boot installs the root).
    pub(crate) canonical_default_readers: Option<(Value, Value)>,
}

/// A `defmacro` definition's parts (bundles `eval_defmacro`'s
/// parameters).
#[derive(Clone, Copy)]
struct DefMacroParts<'a> {
    sym: &'a Symbol,
    doc: Option<&'a str>,
    attr: Option<&'a Map>,
    arglists: &'a [Value],
    fun: &'a crate::expr::FunExpr,
    span: Span,
}

impl Evaluator {
    /// A fresh evaluator loading from `source` (`:file`, used in
    /// spans and `def` meta), with a lone `user` namespace current
    /// (tests install doubles into it; `boot` builds the real
    /// registry from [`new_bare`](Self::new_bare) instead).
    #[must_use]
    pub fn new(source: &str) -> Self {
        let this = Self::new_bare(source);
        this.registry.ensure_ns("user");
        this.registry.set_current("user");
        this
    }

    /// A fresh evaluator with an empty registry (boot ensures
    /// `clojure.core` first, so it takes id 1 like the JVM's
    /// creation order, then `user`).
    #[must_use]
    pub fn new_bare(source: &str) -> Self {
        Self {
            registry: SharedRegistry::new(),
            bindings: Bindings::new(),
            source: Arc::from(source),
            future_threads: Arc::new(std::sync::Mutex::new(Vec::new())),
            agent_system: Arc::new(crate::concurrency::AgentSystem::new()),
            tap_system: Arc::new(crate::concurrency::TapSystem::new()),
            native_call_span: None,
            agent_outbox: None,
            tx: None,
            analyze_depth_limit: None,
            flush_count: 0,
            print_defaults_only: false,
            data_reader_scan_path: None,
            loaded_data_reader_files: HashSet::new(),
            canonical_default_readers: None,
        }
    }

    /// Fork a per-thread evaluator over this evaluator's lineage
    /// (S6B D1): the namespace registry, identity counters, and
    /// future-thread list are shared, while `bindings` (the
    /// conveyed snapshot — like `bound-fn`'s `resetThreadBindingFrame`)
    /// is installed as the whole thread-binding stack. `def`s on any
    /// fork are globally visible; bindings stay thread-local. The
    /// current-namespace *pointer* is lineage-global (the JVM keeps
    /// it thread-local — a port fence), but resolution reads the
    /// thread-local `*ns*` binding first, so an `in-ns` on one
    /// thread never redirects another's lookups.
    pub(crate) fn fork_for_thread(&self, bindings: Bindings) -> Self {
        Self {
            registry: self.registry.clone(),
            bindings,
            source: Arc::clone(&self.source),
            future_threads: Arc::clone(&self.future_threads),
            agent_system: Arc::clone(&self.agent_system),
            tap_system: Arc::clone(&self.tap_system),
            native_call_span: None,
            agent_outbox: None,
            // Transactions never cross threads (the JVM's
            // `LockingTransaction` is a thread-local the binding
            // conveyor does not copy — S19 §1h).
            tx: None,
            // Budgets never cross threads (limits are set on
            // single-threaded planning evaluators, which never fork).
            analyze_depth_limit: None,
            // Flushes are counted per evaluator (fresh forks read 0);
            // defaults-only mode never crosses threads either.
            flush_count: 0,
            print_defaults_only: false,
            // The registry (hence the merged `*data-readers*` root)
            // is shared, so the merge history crosses with it —
            // otherwise a fork would re-merge (and re-fail on) the
            // same files.
            data_reader_scan_path: self.data_reader_scan_path.clone(),
            loaded_data_reader_files: self.loaded_data_reader_files.clone(),
            // The pristine natives are registry-global values, so
            // the baseline crosses with the fork.
            canonical_default_readers: self.canonical_default_readers.clone(),
        }
    }

    /// Snapshot the thread bindings for conveyance (the
    /// `binding-conveyor-fn` frame clone, installed by
    /// [`fork_for_thread`](Self::fork_for_thread)).
    pub(crate) fn clone_bindings(&self) -> Bindings {
        self.bindings.clone()
    }

    /// Record a spawned future thread (S6C's `shutdown-agents`
    /// drains and joins these).
    pub(crate) fn push_future_thread(&self, handle: std::thread::JoinHandle<()>) {
        self.future_threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(handle);
    }

    /// Whether `shutdown-agents` has latched (post-shutdown
    /// futures reject instead of spawning).
    pub(crate) fn agent_system_shutdown(&self) -> bool {
        self.agent_system.is_shutdown()
    }

    /// This lineage's tap system (shared by every forked evaluator,
    /// like the agent system).
    pub(crate) fn tap_system(&self) -> &Arc<crate::concurrency::TapSystem> {
        &self.tap_system
    }

    /// Install the `send` pool field (validating, like the JVM's
    /// `set!` — a rejection installs nothing, gap-96).
    pub(crate) fn set_send_executor(&self, executor: Value) -> Result<(), NativeError> {
        self.agent_system.set_send_executor(executor)
    }

    /// Install the `send-off` pool field (same contract).
    pub(crate) fn set_send_off_executor(&self, executor: Value) -> Result<(), NativeError> {
        self.agent_system.set_send_off_executor(executor)
    }

    /// Where a `send`/`send-off` action executes: the installed
    /// field when set (read per send), else the fixed pools
    /// (gap-96 — the JVM's volatile pool fields).
    pub(crate) fn agent_target_for(&self, pool: clj_value::AgentPool) -> clj_value::ExecTarget {
        self.agent_system.target_for(pool)
    }

    /// This lineage's agent pools, creating them on first use
    /// (`None` after [`shutdown_agents`](Self::shutdown_agents):
    /// dispatches drop like the JVM's rejected `execute`).
    pub(crate) fn agent_pools_for_send(
        &self,
    ) -> Result<Option<Arc<crate::concurrency::AgentPools>>, NativeError> {
        self.agent_system.pools_for_send()
    }

    /// This lineage's agent pools for a custom-executor dispatch:
    /// shutdown-blind (the custom target may still be live) —
    /// born-shut when created post-shutdown.
    pub(crate) fn agent_pools_for_custom(
        &self,
    ) -> Result<Arc<crate::concurrency::AgentPools>, NativeError> {
        self.agent_system.pools_for_custom()
    }

    /// Stop the agent pools (running actions complete; queued
    /// ones drop; new sends fail). Returns at once, like the
    /// JVM — draining happens on the workers; the exit join (see
    /// `join_drainers`) waits them out (T1192 FX1). Shuts the
    /// current pool fields, customs included — a nil field throws
    /// its NPE instead (gap-96).
    pub(crate) fn shutdown_agents(&self) -> Result<(), NativeError> {
        self.agent_system.shutdown()
    }

    /// Latch the pools shut (idempotent — like `clojure.main`'s
    /// exit-time `shutdown-agents`) and join every parked worker
    /// and spawned future thread. The process-exit join: mirrors
    /// the JVM, whose non-daemon pool threads keep the runtime
    /// alive past main. The `run`, `eval`, and REPL drivers call
    /// this before exiting (T1192 FX1).
    pub fn join_drainers(&self) {
        self.agent_system.shutdown_for_exit();
        self.agent_system.join_parked();
        let handles = std::mem::take(
            &mut *self
                .future_threads
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        );
        crate::concurrency::join_all(handles);
    }

    /// The innermost call form's span while a native runs (set by
    /// `Invoke` evaluation; natives needing the call site — the
    /// `macroexpand` wrap position — read this).
    pub(crate) fn native_call_span(&self) -> Option<Span> {
        self.native_call_span
    }

    /// The next `fn` display sequence number (the analyzer
    /// numbers every `fn` it analyzes; lineage-shared, so forked
    /// evaluators never collide).
    pub(crate) fn next_fn_seq(&mut self) -> u64 {
        self.registry.next_fn_seq()
    }

    /// The next `gensym` counter value (per-boot lineage, from 1).
    pub(crate) fn next_gensym(&mut self) -> u64 {
        self.registry.next_gensym()
    }

    /// Reset the `gensym` counter to 1: `boot` loads the embedded
    /// `core.clj` (whose macroexpansions consume numbers), but the
    /// user-visible counter starts from 1 (`macros-contract.md`
    /// pins `G__1` first). Sound: boot-time expansions evaluate to
    /// closed values, so reused numbers can never capture them.
    pub(crate) fn reset_gensym(&mut self) {
        self.registry.reset_gensym();
    }

    /// The `*ns*`-synced current namespace name: the thread-local
    /// `*ns*` handle when bound (boot binds a base frame and
    /// `in-ns` keeps both in sync), else the registry current.
    /// Public for S7 drivers (the REPL prompt tracks this).
    pub fn current_ns_name(&self) -> String {
        let star_ns = Symbol::new(Some("clojure.core"), "*ns*");
        let bound = self
            .registry
            .get("clojure.core")
            .and_then(|ns| ns.lookup(&star_ns))
            .and_then(|id| self.bindings.get(id))
            .and_then(|v| match v {
                Value::Extension(h) => h.as_any().downcast_ref::<NsHandle>(),
                _ => None,
            })
            .map(|ns| ns.name.to_string());
        bound.unwrap_or_else(|| self.registry.current().to_owned())
    }

    /// The next lineage identity (S7 drivers stamp
    /// driver-synthesized throwables — e.g. the REPL's `*e` on
    /// read errors — with these, like [`throw`](Self::throw)).
    pub fn next_id(&mut self) -> u64 {
        self.registry.next_id()
    }

    /// `*out*` flushes served by this evaluator (the
    /// `*flush-on-newline*` test hook; fresh evaluators read 0).
    pub fn flush_count(&self) -> u64 {
        self.flush_count
    }

    /// Record one served `*out*` flush.
    pub(crate) fn bump_flush_count(&mut self) {
        self.flush_count += 1;
    }

    /// Analyze one form against the current namespace (tests use
    /// this for contract errors without evaluating).
    pub fn analyze(&mut self, form: &Value) -> Result<Expr, CompileError> {
        let source = Arc::clone(&self.source);
        let seq = self.registry.eval_seq();
        Analyzer::new(&source, self, seq).analyze(form, Span::new(1, 1), true)
    }

    /// Read every form in `text` against the synced current
    /// namespace (syntax-quote and `::` resolve read-time, like the
    /// JVM). S7 drivers use this + [`eval_top_form`](Self::eval_top_form)
    /// to iterate per-form (print/shift each value, stop at the
    /// first error). Like `Compiler.load`, leading whitespace is
    /// consumed before the zero-oos-S7 entry checks (which blame
    /// the post-skip position), and `*read-eval*` is forced true.
    /// The hook applies `#=` and consults the resolver (hookless
    /// callers pass `NoHooks`, which reports `#=` unevaluated).
    /// Data readers resolve per form (S6): the consult
    /// re-snapshots every iteration (user code runs between forms),
    /// fired discards drain before each form resolves, and
    /// resolution failures surface as uncaught throwables.
    pub fn read_forms(
        &mut self,
        text: &str,
        hook: &mut dyn clj_reader::ReadHooks,
    ) -> Result<Vec<Value>, SourceError> {
        let (reader, seed) = Self::read_forms_pre(self, text);
        match seed {
            Ok(seed) => {
                let mut reader = reader;
                let mut opts = Self::read_forms_opts(self, &seed);
                Self::read_forms_loop(self, &mut reader, hook, &mut opts)
            }
            Err(cce) => Err(crate::reader::executing_wrap(
                reader.line(),
                reader.col(),
                &cce,
                hook.fresh_id(),
                hook.fresh_id(),
            )),
        }
    }

    /// [`read_forms`](Self::read_forms) with the live evaluator
    /// hooks (what `eval`/`load` use). The loop is inlined (not
    /// shared with `read_forms_loop`): the live hooks borrow the
    /// evaluator, which the per-form consult also needs mutably,
    /// so each attempt scopes its own hooks.
    pub fn read_forms_live(&mut self, text: &str) -> Result<Vec<Value>, SourceError> {
        let (reader, seed) = Self::read_forms_pre(self, text);
        match seed {
            Ok(seed) => {
                let mut reader = reader;
                let mut opts = Self::read_forms_opts(self, &seed);
                let mut forms = Vec::new();
                loop {
                    Self::snap_data_readers(self, &mut opts)?;
                    reader.set_opts(&opts);
                    let outcome = {
                        let mut hooks = crate::reader::LiveReadHooks::new(self);
                        reader.read_form(false, &mut hooks)
                    };
                    match outcome {
                        Ok(Some(v)) => {
                            Self::drain_data_readers(self, &mut reader, opts.suppress_read)?;
                            forms.push(Self::resolve_data_readers(self, v, opts.suppress_read)?);
                        }
                        Ok(None) => {
                            Self::drain_data_readers(self, &mut reader, opts.suppress_read)?;
                            return Ok(forms);
                        }
                        Err(e) => {
                            // Effects precede the failure (fired tags ran
                            // during the read), so drain first.
                            Self::drain_data_readers(self, &mut reader, opts.suppress_read)?;
                            return Err(SourceError::Reader(e));
                        }
                    }
                }
            }
            Err(cce) => {
                let outer = self.next_id();
                let cause = self.next_id();
                Err(crate::reader::executing_wrap(
                    reader.line(),
                    reader.col(),
                    &cce,
                    outer,
                    cause,
                ))
            }
        }
    }

    /// The shared read prologue: synced options, leading
    /// whitespace consumed, gates seeded (entry failures keep the
    /// post-skip reader, so callers blame the position like
    /// `Compiler.load`).
    fn read_forms_pre(
        eval: &mut Evaluator,
        text: &str,
    ) -> (
        clj_reader::Reader,
        Result<crate::reader::ReadGateSeed, NativeError>,
    ) {
        let mut reader = clj_reader::Reader::new(text, &ReaderOpts::default());
        reader.consume_whitespaces();
        let seed = crate::reader::seed_load_gates(eval);
        if let Ok(seed) = &seed {
            reader.set_opts(&Self::read_forms_opts(eval, seed));
        }
        (reader, seed)
    }

    /// The synced base options for [`read_forms`](Self::read_forms):
    /// the current namespace maps plus the seeded read gates (the
    /// prologue installs them on the reader; the per-form loops
    /// re-snapshot the data-reader fields from this base).
    fn read_forms_opts(eval: &mut Evaluator, seed: &crate::reader::ReadGateSeed) -> ReaderOpts {
        let mut opts = ReaderOpts {
            current_ns: eval.current_ns_name(),
            ..Default::default()
        };
        // Aliases/refers/imports (the detached `*ns*` keeps them,
        // like the JVM; #80) — shared with the stream readers.
        crate::reader::sync_reader_ns(eval, &mut opts);
        opts.read_eval = seed.read_eval;
        opts.suppress_read = seed.suppress_read;
        opts.resolver_bound = seed.resolver_bound;
        opts
    }

    /// Read every form from a primed reader, resolving data
    /// readers per form (S6 consult inside the S7 hook loop).
    fn read_forms_loop(
        eval: &mut Evaluator,
        reader: &mut clj_reader::Reader,
        hook: &mut dyn clj_reader::ReadHooks,
        opts: &mut ReaderOpts,
    ) -> Result<Vec<Value>, SourceError> {
        let mut forms = Vec::new();
        loop {
            Self::snap_data_readers(eval, opts)?;
            reader.set_opts(opts);
            match reader.read_form(false, hook) {
                Ok(Some(v)) => {
                    Self::drain_data_readers(eval, reader, opts.suppress_read)?;
                    forms.push(Self::resolve_data_readers(eval, v, opts.suppress_read)?);
                }
                Ok(None) => {
                    Self::drain_data_readers(eval, reader, opts.suppress_read)?;
                    return Ok(forms);
                }
                Err(e) => {
                    // Effects precede the failure (fired tags ran
                    // during the read), so drain first.
                    Self::drain_data_readers(eval, reader, opts.suppress_read)?;
                    return Err(SourceError::Reader(e));
                }
            }
        }
    }

    /// Re-snapshot the tag consult for one form (S6), reporting
    /// autoload failures as uncaught throwables (shared by both
    /// per-form loops, so the mapping is pinned once).
    fn snap_data_readers(eval: &mut Evaluator, opts: &mut ReaderOpts) -> Result<(), SourceError> {
        crate::core::snapshot_data_readers(eval, opts)
            .map_err(|e| SourceError::Uncaught(eval.throwable(e)))
    }

    /// Drain one read's fired discards (S6), reporting failures as
    /// uncaught throwables. `suppressed` is the entry seed (M2).
    fn drain_data_readers(
        &mut self,
        reader: &mut Reader,
        suppressed: bool,
    ) -> Result<(), SourceError> {
        let fired = reader.take_fired();
        crate::core::drain_fired_tags(self, fired, suppressed)
            .map_err(|e| SourceError::Uncaught(self.throwable(e)))
    }

    /// Resolve one read form's deferred tags (S6), reporting
    /// failures as uncaught throwables. `suppressed` is the entry
    /// seed (M2 — `snapshot_data_readers` never touches it, so the
    /// loops' `opts.suppress_read` stays the entry value).
    fn resolve_data_readers(
        &mut self,
        form: Value,
        suppressed: bool,
    ) -> Result<Value, SourceError> {
        crate::core::resolve_data_readers(self, form, suppressed)
            .map_err(|e| SourceError::Uncaught(self.throwable(e)))
    }

    /// Read every form in `text` (default `:read-cond :allow`
    /// options) and evaluate each in turn, collecting one value per
    /// read form. Uncaught throwables (including bare macro-arity
    /// failures) abort the load.
    pub fn eval_source(&mut self, text: &str) -> Result<Vec<Value>, SourceError> {
        let forms = self.read_forms_live(text)?;
        let mut out = Vec::with_capacity(forms.len());
        for form in &forms {
            out.push(self.eval_top_form(form)?);
        }
        Ok(out)
    }

    /// Deep-realize a value for printing (like the JVM, whose
    /// print walks — forcing errors surface here, not in `eval`).
    pub fn realize_deep(&mut self, v: &Value) -> Result<Value, crate::error::Signal> {
        crate::lazy::realize_deep(self, v)
    }

    /// Evaluate one top-level form. A `do`-headed list splits like
    /// `Compiler.eval`: each element analyzes after the previous
    /// evaluates, and only the last value survives (§do-split).
    pub fn eval_top_form(&mut self, form: &Value) -> Result<Value, SourceError> {
        // `Compiler.eval` macroexpands first (fully), then splits
        // `do`-headed expansions; each element re-enters `eval`
        // (macros defining macros before later-element analysis
        // see them).
        let source = Arc::clone(&self.source);
        let span = span_of(form, Span::new(1, 1));
        let expanded = crate::macros::macroexpand_full(self, form, span, &source)
            .map_err(|e| self.compile_err(e))?;
        if let Some(elements) = split_top_do(&expanded) {
            let mut last = Value::Nil;
            for element in elements {
                last = self.eval_top_form(element)?;
            }
            return Ok(last);
        }
        self.registry.bump_eval_seq();
        let expr = self.analyze(&expanded).map_err(|e| self.compile_err(e))?;
        match self.eval(&expr, &Env::new()) {
            Ok(v) => Ok(v),
            Err(s) => Err(uncaught(s)),
        }
    }

    /// A compile failure as a source error: bare causes escape
    /// like uncaught throwables of their own class (macro arities,
    /// `declare`'s cast, stripped macro arities); wrapped chains
    /// stay compile errors. Shared with AOT planning, which mirrors
    /// `eval_top_form`'s failure mapping.
    pub(crate) fn compile_err(&mut self, e: CompileError) -> SourceError {
        if e.bare {
            let id = self.next_id();
            SourceError::Uncaught(Throwable {
                id,
                class: crate::ns::throw_class_for(e.class),
                message: Some(Arc::from(e.message.as_str())),
                data: None,
                cause: None,
            })
        } else {
            SourceError::Compile(e)
        }
    }

    /// Evaluate `text` as a load from `path`: spans blame the
    /// path while the load runs, then the previous source returns
    /// (callers bind `*file*` and save the current namespace).
    pub(crate) fn load_source(
        &mut self,
        path: &str,
        text: &str,
    ) -> Result<Vec<Value>, SourceError> {
        let prev = std::mem::replace(&mut self.source, Arc::from(path));
        let out = self.eval_source(text);
        self.source = prev;
        out
    }

    /// Push one single-entry thread-binding frame, returning the
    /// depth to pass to [`pop_bindings`](Self::pop_bindings)
    /// (natives bind `*file*`/`*pending-paths*` around loads).
    /// Public for S7 drivers (the REPL session frame, the `run`
    /// `*command-line-args*` binding); the var should be dynamic.
    pub fn push_binding(&mut self, var: Var, value: Value) -> usize {
        let depth = self.bindings.depth();
        self.bindings.push_frame(vec![(var, value)]);
        depth
    }

    /// Push one thread-binding frame (validated entries).
    pub(crate) fn push_frame(&mut self, entries: Vec<(Var, Value)>) {
        self.bindings.push_frame(entries);
    }

    /// Pop one thread-binding frame (`None` when the stack is
    /// empty — `pop-thread-bindings` raises then).
    pub(crate) fn pop_frame(&mut self) -> Option<Vec<(Var, Value)>> {
        self.bindings.pop_frame()
    }

    /// Pop thread-binding frames back to `depth`. Public for S7
    /// drivers (restores the depth a [`push_binding`](Self::push_binding)
    /// returned).
    pub fn pop_bindings(&mut self, depth: usize) {
        self.bindings.truncate(depth);
    }

    /// The current thread-binding stack depth (natives pushing
    /// several frames pop back to this).
    pub(crate) fn bindings_depth(&self) -> usize {
        self.bindings.depth()
    }

    /// `set!` on the topmost frame for a var (false when no frame
    /// exists — `in-ns` falls back to the root then). Public for
    /// S7 drivers (the REPL shifts `*1`/`*2`/`*3`/`*e` through
    /// this, like `var-set`).
    pub fn set_binding_top(&mut self, id: u64, value: Value) -> bool {
        self.bindings.set_top(id, value)
    }

    /// Whether a var has a thread-local entry (`thread-bound?`).
    pub(crate) fn is_thread_bound(&self, id: u64) -> bool {
        self.bindings.get(id).is_some()
    }

    /// Every binding entry oldest-first (for snapshots).
    pub(crate) fn binding_pairs(&self) -> Vec<(Var, Value)> {
        self.bindings
            .pairs()
            .map(|(var, v)| (var.clone(), v.clone()))
            .collect()
    }

    /// The current load path (`:file`), for spans.
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    /// Install a native fn under `ns`/`name` (test doubles land
    /// in `user`; S5 installs the core subset into `clojure.core`).
    pub fn install_native(
        &mut self,
        ns: &str,
        name: &str,
        func: Arc<crate::ext::NativeImpl>,
    ) -> Value {
        let id = self.next_id();
        let sym = Symbol::new(Some(ns), name);
        let var_id = self.registry.intern(&sym);
        let native = Value::Extension(Arc::new(NativeFn {
            id,
            display: Arc::from(format!("{ns}/{name}").as_str()),
            func,
            meta: None,
        }));
        self.registry.with_state_mut(var_id, |state| {
            state.root = Some(native.clone());
        });
        native
    }

    /// Stamp a runtime failure with an identity and throw it.
    pub(crate) fn throw(&mut self, err: NativeError) -> Signal {
        Signal::Throw(self.throwable(err))
    }

    /// Stamp a runtime failure with an identity (S6: read paths
    /// report data-reader failures as uncaught throwables).
    pub(crate) fn throwable(&mut self, err: NativeError) -> Throwable {
        let inner = err.into_inner();
        // A missing message stays missing (the JVM's no-arg
        // `CancellationException` reports a nil message, and
        // `NativeError` encodes that as the empty string).
        let message = if inner.message.is_empty() {
            None
        } else {
            Some(Arc::from(inner.message.as_str()))
        };
        Throwable {
            id: self.next_id(),
            class: inner.class,
            message,
            data: inner.data.map(Box::new),
            cause: inner.cause.map(Box::new),
        }
    }

    /// The visible value of a var: thread binding, else root (or
    /// the var's own `Unbound` sentinel, never a throw). Public
    /// for S7 drivers (the REPL reads the current `*1`/`*2` to
    /// shift the window).
    pub fn deref_var(&self, id: u64, sym: &Symbol) -> Value {
        if let Some(v) = self.bindings.get(id) {
            return v.clone();
        }
        self.registry.root_or_unbound(id, sym)
    }

    /// Evaluate an expression to a value (or a control signal).
    pub fn eval(&mut self, expr: &Expr, env: &Env) -> Result<Value, Signal> {
        match expr {
            Expr::Literal(v) | Expr::Quote(v) => Ok(v.clone()),
            Expr::VecLit(elems, meta) => self.eval_vec(elems, meta.clone(), env),
            Expr::MapLit(pairs, meta) => self.eval_map_lit(pairs, meta.clone(), env),
            Expr::SetLit(elems, meta) => self.eval_set_lit(elems, meta.clone(), env),
            Expr::Global(var) => Ok(self.deref_var(var.id, &var.sym)),
            Expr::Local(idx) => Ok(env.get(*idx)),
            Expr::VarRef(var) => Ok(Value::Var(var.clone())),
            Expr::Def {
                sym,
                doc,
                init,
                span,
            } => self.eval_def(sym, doc.clone(), init.as_deref(), *span, env),
            Expr::If { test, then, els } => self.eval_if(test, then, els, env),
            Expr::Do(exprs) => self.eval_body(exprs, env),
            Expr::Let { bindings, body } => {
                let env = self.bind_all(bindings, env)?;
                self.eval_body(body, &env)
            }
            Expr::Loop { bindings, body } => self.eval_loop(bindings, body, env),
            Expr::LetFn { fns, body } => self.eval_letfn(fns, body, env),
            Expr::Fn(fun) => Ok(Value::Extension(Arc::new(self.make_closure(fun, env)))),
            Expr::Recur(args) => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.eval(arg, env)?);
                }
                Err(Signal::Recur(values))
            }
            Expr::Throw(inner) => self.eval_throw(inner, env),
            Expr::NewThrowable { class, message } => Ok(Value::Throwable(Throwable {
                id: self.next_id(),
                class: *class,
                message: Some(Arc::from(message.as_str())),
                data: None,
                cause: None,
            })),
            Expr::Try {
                body,
                catches,
                finally_body,
            } => self.eval_try(body, catches, finally_body, env),
            Expr::Set { var, val } => self.eval_set(var, val, env),
            Expr::Binding { pairs, body } => self.eval_binding(pairs, body, env),
            Expr::WithRedefs { pairs, body } => self.eval_with_redefs(pairs, body, env),
            Expr::WithBindings { map, body } => self.eval_with_bindings(map, body, env),
            Expr::BoundFn { name, fun } => self.eval_bound_fn(name.as_deref(), fun, env),
            Expr::DefMacro {
                sym,
                doc,
                attr,
                arglists,
                fun,
                span,
            } => {
                let parts = DefMacroParts {
                    sym,
                    doc: doc.as_deref(),
                    attr: attr.as_ref(),
                    arglists,
                    fun,
                    span: *span,
                };
                self.eval_defmacro(&parts, env)
            }
            Expr::Declare { syms, span } => self.eval_declare(syms, *span, env),
            Expr::Ns {
                name,
                doc,
                attr,
                clauses,
            } => crate::ns::eval_ns_form(self, name, doc.as_deref(), attr.as_ref(), clauses),
            Expr::Assert {
                test,
                message,
                form,
            } => self.eval_assert(test, message.as_deref(), form, env),
            Expr::Time { expr } => self.eval_time(expr, env),
            Expr::WithOutStr { out, body } => self.eval_with_out_str(out, body, env),
            Expr::WithInStr { input, s, body } => self.eval_with_in_str(input, s, body, env),
            Expr::LazySeqStar { body } => Ok(crate::lazy::thunk0_value(
                Arc::new(body.clone()),
                env.clone(),
            )),
            Expr::DelayStar { display, body } => Ok(self.eval_delay_star(display, body, env)),
            Expr::Locking { lockee, body } => self.eval_locking(lockee, body, env),
            Expr::Sync { body } | Expr::Io { body } => {
                crate::stm::eval_sync_or_io(self, expr, body, env)
            }
            Expr::CaseStar {
                expr,
                shift,
                mask,
                default,
                cases,
                skip,
                test,
            } => self.eval_case_star(expr, *shift, *mask, default, cases, skip, *test, env),
            Expr::Invoke { head, args, span } => {
                let callee = self.eval(head, env)?;
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.eval(arg, env)?);
                }
                // The call site stays visible to natives for the
                // duration of the call (nesting saves/restores).
                let saved = self.native_call_span.replace(*span);
                let out = self.invoke(&callee, &values);
                self.native_call_span = saved;
                out
            }
        }
    }

    /// A var id, interning on first reference. Only synthesized
    /// analyzer heads and VM-internal bindings reach here;
    /// analyzer-captured paths (`Global`/`VarRef`/`Set`/`Binding`/
    /// `WithRedefs`) carry their ids and never re-resolve (#79).
    pub(crate) fn var_id(&mut self, sym: &Symbol) -> u64 {
        self.registry
            .lookup(sym)
            .or_else(|| self.tombstone_id(sym))
            .unwrap_or_else(|| self.registry.intern(sym))
    }

    /// A var id through a removed namespace's tombstone: every
    /// caller feeds analyzer-resolved targets, so a lookup miss
    /// means the home went away after analysis — and the
    /// analyzer-captured var (ids are never reused) still answers
    /// from frozen mappings, like the JVM's compiled `Var`
    /// reference, without resurrecting the name (T73 finding 4
    /// round 2; live homes win by name inside
    /// `ns_live_or_removed`, so live behavior is untouched).
    fn tombstone_id(&self, sym: &Symbol) -> Option<u64> {
        let ns = sym.namespace()?;
        self.registry.ns_live_or_removed(ns)?.lookup(sym)
    }

    /// A vector literal (elements evaluate left to right).
    fn eval_vec(&mut self, elems: &[Expr], meta: Option<Map>, env: &Env) -> Result<Value, Signal> {
        let mut out = Vec::with_capacity(elems.len());
        for elem in elems {
            out.push(self.eval(elem, env)?);
        }
        Ok(Value::Vector(Vector::from_vec(out).with_meta(meta)))
    }

    /// A map literal (keys and values evaluate left to right).
    fn eval_map_lit(
        &mut self,
        pairs: &[(Expr, Expr)],
        meta: Option<Map>,
        env: &Env,
    ) -> Result<Value, Signal> {
        let mut out = new_map();
        for (k, v) in pairs {
            let key = self.eval(k, env)?;
            let val = self.eval(v, env)?;
            out = out.insert(key, val);
        }
        Ok(Value::Map(out.with_meta(meta)))
    }

    /// A set literal (elements evaluate left to right).
    fn eval_set_lit(
        &mut self,
        elems: &[Expr],
        meta: Option<Map>,
        env: &Env,
    ) -> Result<Value, Signal> {
        let mut out = clj_value::new_set();
        for elem in elems {
            out = out.insert(self.eval(elem, env)?);
        }
        Ok(Value::Set(out.with_meta(meta)))
    }

    /// `letfn`: extend the frame, fill each slot with its closure
    /// (self/mutual recursion), run the body.
    fn eval_letfn(
        &mut self,
        fns: &[(String, crate::expr::FunExpr)],
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        let env = env.extend(vec![Value::Nil; fns.len()]);
        let base = env.len() - fns.len();
        for (i, (_, fun)) in fns.iter().enumerate() {
            let closure = self.make_closure(fun, &env);
            env.set(base + i, Value::Extension(Arc::new(closure)));
        }
        self.eval_body(body, &env)
    }

    /// `throw`: throwables signal, anything else is the
    /// `Throwable` cast error.
    fn eval_throw(&mut self, inner: &Expr, env: &Env) -> Result<Value, Signal> {
        let value = self.eval(inner, env)?;
        match value {
            Value::Throwable(t) => Err(Signal::Throw(t)),
            // `nil` is not the cast error: the JVM throws a
            // Clojure-constructed NPE with stable text (probe29).
            Value::Nil => Err(self.throw(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot throw exception because \"null\" is null".to_owned(),
            ))),
            other => Err(self.throw(NativeError::cast(
                &jvm_class_of(&other),
                "java.lang.Throwable",
            ))),
        }
    }

    /// `set!`: rebind the thread-local top, else the root-binding
    /// error.
    fn eval_set(&mut self, var: &Var, val: &Expr, env: &Env) -> Result<Value, Signal> {
        let value = self.eval(val, env)?;
        if self.bindings.set_top(var.id, value.clone()) {
            Ok(value)
        } else {
            Err(self.throw(NativeError::new(
                ThrowClass::IllegalStateException,
                format!(
                    "Can't change/establish root binding of: {} with set",
                    var.sym.name()
                ),
            )))
        }
    }

    /// `binding`: evaluate every init, validate every var, then
    /// push all and run the body, always popping. The JVM builds
    /// the push map (all inits) before `push-thread-bindings`
    /// validates, so an init throw beats a dynamic-ness error
    /// (probe29 `binding-*`); nothing is pushed before the body,
    /// so pre-body failures need no unwind.
    fn eval_binding(
        &mut self,
        pairs: &[(Var, Expr)],
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        let mut values = Vec::with_capacity(pairs.len());
        for (_, init) in pairs {
            values.push(self.eval(init, env)?);
        }
        for (var, _) in pairs {
            let dynamic = self.registry.state(var.id).is_some_and(|s| s.dynamic);
            if !dynamic {
                return Err(self.throw(NativeError::new(
                    ThrowClass::IllegalStateException,
                    format!("Can't dynamically bind non-dynamic var: {}", var.sym),
                )));
            }
        }
        let depth = self.bindings.depth();
        let mut frame = Vec::with_capacity(pairs.len());
        for ((var, _), value) in pairs.iter().zip(values) {
            frame.push((var.clone(), value));
        }
        self.bindings.push_frame(frame);
        let result = self.eval_body(body, env);
        self.bindings.truncate(depth);
        result
    }

    /// `(assert x)` / `(assert x message)`: a falsy test throws
    /// `AssertionError` (`Assert failed: <form>` /
    /// `Assert failed: <msg>\n<form>`, the message evaluated only on
    /// failure); a truthy test yields `nil` (the `when-not`
    /// expansion).
    fn eval_assert(
        &mut self,
        test: &Expr,
        message: Option<&Expr>,
        form: &str,
        env: &Env,
    ) -> Result<Value, Signal> {
        if self.eval(test, env)?.is_truthy() {
            return Ok(Value::Nil);
        }
        let text = match message {
            None => format!("Assert failed: {form}"),
            Some(m) => {
                let mv = self.eval(m, env)?;
                // `native_str` of one arg is `str_of_one`.
                let rendered = crate::core::str_of_one(self, &mv).map_err(|e| self.throw(e))?;
                format!("Assert failed: {rendered}\n{form}")
            }
        };
        Err(self.throw(NativeError::new(ThrowClass::AssertionError, text)))
    }

    /// `(time expr)`: wall-clock the expr, `prn` the JVM line to
    /// `*out*` (`"Elapsed time: <millis> msecs"`, quoted by `prn`),
    /// return the value.
    fn eval_time(&mut self, expr: &Expr, env: &Env) -> Result<Value, Signal> {
        let start = std::time::Instant::now();
        let v = self.eval(expr, env)?;
        let line = Self::format_elapsed_millis(start.elapsed().as_nanos());
        crate::io::prn_str(self, &line).map_err(|e| self.throw(e))?;
        Ok(v)
    }

    /// The `time` line for a fixed nanosecond count: `nanos/1e6`
    /// as an f64, printed like the JVM's `Double/toString`
    /// (T1192 FX1 fx1-time probe pins the digits; the live path
    /// passes `Instant::elapsed` here).
    fn format_elapsed_millis(nanos: u128) -> String {
        let millis = nanos as f64 / 1_000_000.0;
        let dstr = clj_value::pr_to_string(&Value::Double(millis), &clj_value::PrintOpts::print());
        format!("Elapsed time: {dstr} msecs")
    }

    /// `(with-out-str body…)`: bind `*out*` to a fresh string writer
    /// (the `binding` push/pop discipline, so throws still pop),
    /// discard the body, return the captured output.
    fn eval_with_out_str(
        &mut self,
        out: &Symbol,
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        let writer = crate::io::string_writer(self);
        let out_id = self.var_id(out);
        let pairs = vec![(
            Var {
                id: out_id,
                sym: out.clone(),
            },
            Expr::Literal(writer.clone()),
        )];
        self.eval_binding(&pairs, body, env)?;
        let contents = crate::io::writer_contents(&writer).unwrap_or_default();
        Ok(Value::Str(Arc::from(contents.as_str())))
    }

    /// `(with-in-str s body…)`: bind `*in*` to a fresh string
    /// reader (the `binding` push/pop discipline, so throws still
    /// pop) and return the body's value. A non-string `s` fails
    /// the `String` cast (probe `wis-num`).
    fn eval_with_in_str(
        &mut self,
        input: &Symbol,
        s: &Expr,
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        let v = self.eval(s, env)?;
        let Value::Str(text) = &v else {
            return Err(self.throw(crate::core::cce_err(&v, "java.lang.String")));
        };
        let reader = crate::reader::string_reader(self, text);
        let input_id = self.var_id(input);
        let pairs = vec![(
            Var {
                id: input_id,
                sym: input.clone(),
            },
            Expr::Literal(reader),
        )];
        self.eval_binding(&pairs, body, env)
    }

    /// `(case* …)`: the interpreter's `CaseExpr.doEmit` on the boxed
    /// path (the only one here — no primitives). `:int` funnels
    /// numbers through `intValue` (non-numbers miss, then an `equiv`
    /// post-check); hash modes realize, hash, and post-check (skip
    /// members run their collision-`condp` unchecked).
    #[allow(clippy::too_many_arguments)]
    fn eval_case_star(
        &mut self,
        expr: &Expr,
        shift: i32,
        mask: i32,
        default: &Expr,
        cases: &[(i32, Expr, Expr)],
        skip: &[i32],
        test: CaseTest,
        env: &Env,
    ) -> Result<Value, Signal> {
        let v = self.eval(expr, env)?;
        if test == CaseTest::Int {
            self.eval_case_star_int(&v, shift, mask, default, cases, env)
        } else {
            self.eval_case_star_hash(&v, shift, mask, default, cases, skip, env)
        }
    }

    /// `:int` `case*` dispatch: `intValue` the target (non-numbers
    /// miss), shift-mask, then an `equiv` post-check on a hit.
    fn eval_case_star_int(
        &mut self,
        v: &Value,
        shift: i32,
        mask: i32,
        default: &Expr,
        cases: &[(i32, Expr, Expr)],
        env: &Env,
    ) -> Result<Value, Signal> {
        let Some(key) = case_int_value(v) else {
            return self.eval(default, env);
        };
        let key = case_shift_mask(key, shift, mask);
        let hit = cases.iter().find(|(k, _, _)| *k == key);
        match hit {
            None => self.eval(default, env),
            Some((_, t, then)) => {
                let tv = self.eval(t, env)?;
                if crate::lazy::equiv_vals(self, v, &tv)? {
                    self.eval(then, env)
                } else {
                    self.eval(default, env)
                }
            }
        }
    }

    /// Hash `case*` dispatch: reject unbounded seqs, realize, hash,
    /// shift-mask, then a hit runs directly (`skip`) or behind an
    /// `equiv` post-check.
    #[allow(clippy::too_many_arguments)]
    fn eval_case_star_hash(
        &mut self,
        v: &Value,
        shift: i32,
        mask: i32,
        default: &Expr,
        cases: &[(i32, Expr, Expr)],
        skip: &[i32],
        env: &Env,
    ) -> Result<Value, Signal> {
        if clj_value::seq_hash_unsupported(v) {
            return Err(self.unsupported_hash());
        }
        let r = crate::lazy::realize_deep(self, v)?;
        if clj_value::seq_hash_unsupported(&r) {
            return Err(self.unsupported_hash());
        }
        let key = case_shift_mask(clj_value::hash_code(&r), shift, mask);
        let hit = cases.iter().find(|(k, _, _)| *k == key);
        match hit {
            None => self.eval(default, env),
            Some((k, _, then)) if skip.contains(k) => self.eval(then, env),
            Some((_, t, then)) => {
                let tv = self.eval(t, env)?;
                if crate::lazy::equiv_vals(self, &r, &tv)? {
                    self.eval(then, env)
                } else {
                    self.eval(default, env)
                }
            }
        }
    }

    /// Bare `UnsupportedOperationException` (the JVM `hasheq` /
    /// `hashCode` overrides on unbounded seqs).
    fn unsupported_hash(&mut self) -> Signal {
        self.throw(NativeError::new(
            ThrowClass::UnsupportedOperationException,
            String::new(),
        ))
    }

    /// `with-redefs`: a PARALLEL ROOT SWAP, not thread bindings.
    /// Every init evaluates against the old roots (a repeated var
    /// keeps the last value, map-style), the roots install, the body
    /// runs, and the saved roots — or rootlessness — restore in a
    /// `finally`, even on throw (probe37 `wrd-*`). The restore also
    /// runs on panic unwind (an S19 retry unwinding through here
    /// must not leak redefined roots globally).
    fn eval_with_redefs(
        &mut self,
        pairs: &[(Var, Expr)],
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        let mut saves: Vec<(u64, Option<Value>, Value)> = Vec::with_capacity(pairs.len());
        for (var, init) in pairs {
            let value = self.eval(init, env)?;
            let id = var.id;
            match saves.iter_mut().find(|(i, _, _)| *i == id) {
                Some(slot) => slot.2 = value,
                None => {
                    let saved = self.registry.state(id).and_then(|s| s.root.clone());
                    saves.push((id, saved, value));
                }
            }
        }
        for (id, _, value) in &saves {
            self.registry.with_state_mut(*id, |st| {
                st.root = Some(value.clone());
            });
        }
        let result = catch_unwind(AssertUnwindSafe(|| self.eval_body(body, env)));
        for (id, saved, _) in saves.iter().rev() {
            self.registry.with_state_mut(*id, |st| {
                st.root = saved.clone();
            });
        }
        match result {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        }
    }

    /// `with-bindings`: evaluate the map expression and run the body
    /// over one pushed frame (the macro expands to
    /// `with-bindings*`, probe37 `wb-expansion`).
    fn eval_with_bindings(
        &mut self,
        map: &Expr,
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        let bindings = self.eval(map, env)?;
        let entries = crate::vars::push_entries(self, &bindings).map_err(|e| self.throw(e))?;
        let depth = self.bindings.depth();
        self.bindings.push_frame(entries);
        let result = self.eval_body(body, env);
        self.bindings.truncate(depth);
        result
    }

    /// `bound-fn`: like `fn`, but the closure snapshots the thread
    /// bindings at creation and re-pushes them around every
    /// invocation. Named `bound-fn` replays the single-`letfn`
    /// layout (same slot math, conveyed closure).
    fn eval_bound_fn(
        &mut self,
        name: Option<&str>,
        fun: &crate::expr::FunExpr,
        env: &Env,
    ) -> Result<Value, Signal> {
        let conveyed = self.bindings.clone();
        if name.is_none() {
            let mut closure = self.make_closure(fun, env);
            closure.conveyed = Some(conveyed);
            return Ok(Value::Extension(Arc::new(closure)));
        }
        let env = env.extend(vec![Value::Nil]);
        let mut closure = self.make_closure(fun, &env);
        closure.conveyed = Some(conveyed);
        let value = Value::Extension(Arc::new(closure));
        env.set(env.len() - 1, value.clone());
        Ok(value)
    }

    /// `defmacro`: like `def` of the `&form`/`&env`-prepended
    /// `fn`, plus `:macro true` and `:arglists` meta (name-symbol
    /// meta under the attrmap; the attrmap wins over the docstring
    /// for `:doc`; autos win over all).
    fn eval_defmacro(&mut self, parts: &DefMacroParts<'_>, env: &Env) -> Result<Value, Signal> {
        let DefMacroParts {
            sym,
            doc,
            attr,
            arglists,
            fun,
            span,
        } = *parts;
        let id = self.registry.intern_def_target(sym);
        let closure = self.make_closure(fun, env);
        let dynamic = Self::dynamic_of(sym);
        let cur = self.current_ns_name();
        let cur_id = self.registry.get(&cur).map_or(0, |ns| ns.id());
        let mut meta = sym.meta().cloned().unwrap_or_else(new_map);
        if let Some(attr) = attr {
            for (k, v) in attr.iter() {
                meta = meta.insert(k.clone(), v.clone());
            }
        }
        meta = meta
            .insert(
                Value::Keyword(Keyword::new(None, "macro")),
                Value::Bool(true),
            )
            .insert(
                Value::Keyword(Keyword::new(None, "arglists")),
                Value::List(List::from_vec(arglists.to_vec())),
            );
        let doc_key = Value::Keyword(Keyword::new(None, "doc"));
        if let Some(doc) = doc.filter(|_| meta.get(&doc_key).is_none()) {
            meta = meta.insert(doc_key, Value::Str(Arc::from(doc)));
        }
        let auto = [
            ("line", Value::Int(i64::from(span.line))),
            ("column", Value::Int(i64::from(span.col))),
            ("file", Value::Str(Arc::clone(&self.source))),
            ("name", Value::Symbol(Symbol::new(None, sym.name()))),
            (
                "ns",
                Value::Extension(Arc::new(NsHandle {
                    name: Arc::from(cur.as_str()),
                    id: cur_id,
                })),
            ),
        ];
        for (key, value) in auto {
            let key = Value::Keyword(Keyword::new(None, key));
            meta = meta.insert(key, value);
        }
        let root = Value::Extension(Arc::new(closure));
        self.registry.with_state_mut(id, |state| {
            state.root = Some(root);
            state.meta = Some(meta);
            state.dynamic = dynamic;
        });
        Ok(Value::Var(Var {
            id,
            sym: sym.clone(),
        }))
    }

    /// `declare`: per-name `def`-with-no-init plus `:declared`
    /// true, returning the last var (roots survive: `def` only
    /// sets given inits; empty `declare` is nil).
    fn eval_declare(&mut self, syms: &[Symbol], span: Span, env: &Env) -> Result<Value, Signal> {
        let mut last = Value::Nil;
        for sym in syms {
            let mut meta = sym.meta().cloned().unwrap_or_else(new_map);
            meta = meta.insert(
                Value::Keyword(Keyword::new(None, "declared")),
                Value::Bool(true),
            );
            let sym = sym.clone().with_meta(Some(meta));
            last = self.eval_def(&sym, None, None, span, env)?;
        }
        Ok(last)
    }

    /// Dynamic is the truthiness of the sym's `:dynamic` meta,
    /// reset by every `def` (shared by `def`/`defmacro`).
    fn dynamic_of(sym: &Symbol) -> bool {
        sym.meta()
            .and_then(|meta| {
                let probe = Value::Keyword(Keyword::new(None, "dynamic"));
                meta.get(&probe)
            })
            .is_some_and(Value::is_truthy)
    }

    /// A body: the last value, or `nil` when empty.
    pub(crate) fn eval_body(&mut self, body: &[Expr], env: &Env) -> Result<Value, Signal> {
        let mut value = Value::Nil;
        for expr in body {
            value = self.eval(expr, env)?;
        }
        Ok(value)
    }

    /// `if`: the taken branch (split from [`eval`](Self::eval)
    /// for the complexity budget).
    fn eval_if(
        &mut self,
        test: &Expr,
        then: &Expr,
        els: &Expr,
        env: &Env,
    ) -> Result<Value, Signal> {
        if self.eval(test, env)?.is_truthy() {
            self.eval(then, env)
        } else {
            self.eval(els, env)
        }
    }

    /// Bind `let`/`loop` pairs left to right (each init sees the
    /// previous pairs), returning the extended frame.
    fn bind_all(&mut self, bindings: &[(Pattern, Expr)], env: &Env) -> Result<Env, Signal> {
        let mut env = env.clone();
        for (pat, init) in bindings {
            let value = self.eval(init, &env)?;
            env = self.bind_pattern(pat, value, &env)?;
        }
        Ok(env)
    }

    /// Bind one pattern: extend the frame, evaluate `:or` defaults
    /// against the pre-extension frame, destructure into the slots.
    /// The pre-extension frame is the env the analyzer saw when it
    /// analyzed the defaults (the pair's/param's own names extend
    /// only after), so default closures capture the right length —
    /// evaluating against the extended frame shifts every captured
    /// local by the pair's slot count (#171: the callee arrives as
    /// its own first arg).
    fn bind_pattern(&mut self, pat: &Pattern, value: Value, env: &Env) -> Result<Env, Signal> {
        let count = pattern_names(pat).len();
        let extended = env.extend(vec![Value::Nil; count]);
        let base = extended.len() - count;
        let mut out = Vec::with_capacity(count);
        self.destructure(pat, &value, env, &mut out, ForceSet::default())?;
        debug_assert_eq!(out.len(), count, "destructure fills every slot");
        for (i, v) in out.into_iter().enumerate() {
            extended.set(base + i, v);
        }
        Ok(extended)
    }

    /// `(def …)`: evaluate the init (or leave rootless), refresh
    /// root/meta/dynamic, return the var. Dynamic is the truthiness
    /// of the sym's `:dynamic` meta, reset by every `def`
    /// (probe25).
    fn eval_def(
        &mut self,
        sym: &Symbol,
        doc: Option<String>,
        init: Option<&Expr>,
        span: Span,
        env: &Env,
    ) -> Result<Value, Signal> {
        let id = self.registry.intern_def_target(sym);
        let value = match init {
            Some(init) => Some(self.eval(init, env)?),
            None => None,
        };
        let dynamic = Self::dynamic_of(sym);
        let cur = self.current_ns_name();
        let cur_id = self.registry.get(&cur).map_or(0, |ns| ns.id());
        let mut meta = sym.meta().cloned().unwrap_or_else(new_map);
        if let Some(doc) = doc {
            let key = Value::Keyword(Keyword::new(None, "doc"));
            meta = meta.insert(key, Value::Str(Arc::from(doc.as_str())));
        }
        let auto = [
            ("line", Value::Int(i64::from(span.line))),
            ("column", Value::Int(i64::from(span.col))),
            ("file", Value::Str(Arc::clone(&self.source))),
            ("name", Value::Symbol(Symbol::new(None, sym.name()))),
            (
                "ns",
                Value::Extension(Arc::new(NsHandle {
                    name: Arc::from(cur.as_str()),
                    id: cur_id,
                })),
            ),
        ];
        for (key, val) in auto {
            let key = Value::Keyword(Keyword::new(None, key));
            meta = meta.insert(key, val);
        }
        self.registry.with_state_mut(id, |state| {
            if let Some(v) = value {
                state.root = Some(v);
            }
            state.meta = Some(meta);
            state.dynamic = dynamic;
        });
        Ok(Value::Var(Var {
            id,
            sym: sym.clone(),
        }))
    }

    /// `(loop …)`: bind once, then run the body, rebinding through
    /// the patterns on every `recur`.
    fn eval_loop(
        &mut self,
        bindings: &[(Pattern, Expr)],
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        // Bind sequentially (each init sees the previous pairs),
        // then run the body against the frame.
        let outer = env.clone();
        let mut env = env.clone();
        for (pat, init) in bindings {
            let value = self.eval(init, &env)?;
            env = self.bind_pattern(pat, value, &env)?;
        }
        loop {
            match self.eval_body(body, &env) {
                Ok(v) => return Ok(v),
                Err(Signal::Recur(values)) => {
                    // Fresh frame per iteration (like the copy it
                    // replaces, closures from this iteration keep
                    // observing the old frame): rebind through
                    // `bind_pattern` so `:or` defaults evaluate
                    // against each pair's pre-extension frame (#171).
                    let mut frame = outer.clone();
                    for ((pat, _), value) in bindings.iter().zip(values) {
                        frame = self.bind_pattern(pat, value, &frame)?;
                    }
                    env = frame;
                }
                Err(signal) => return Err(signal),
            }
        }
    }

    /// `(try …)`: the first matching `catch` wins (by hierarchy);
    /// `finally` runs on every path and its value is discarded.
    /// A panic unwinding through the body (an S19 retry is now
    /// routine control flow, not just port bugs) still runs
    /// `finally` — catches never see it — then resumes; a throwing
    /// `finally` masks the in-flight panic like the JVM.
    fn eval_try(
        &mut self,
        body: &[Expr],
        catches: &[Catch],
        finally_body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        let result = match catch_unwind(AssertUnwindSafe(|| self.eval_body(body, env))) {
            Ok(result) => result,
            Err(payload) => {
                if let Err(signal @ Signal::Throw(_)) = self.eval_body(finally_body, env) {
                    return Err(signal);
                }
                resume_unwind(payload);
            }
        };
        match result {
            Ok(value) => {
                self.eval_body(finally_body, env)?;
                Ok(value)
            }
            Err(signal) => {
                // `recur` cannot cross `try` (analyzer-fenced),
                // so every error here is a throw.
                let thrown = try_thrown(signal).expect("recur cannot cross try");
                for catch in catches {
                    if thrown.class.matches(&catch.class) {
                        let env = env.extend(vec![Value::Throwable(thrown.clone())]);
                        debug_assert_eq!(env.len() - 1, catch.slot, "catch binds one slot");
                        let result = self.eval_body(&catch.body, &env);
                        // `finally` still runs when the handler throws.
                        match result {
                            Ok(value) => {
                                self.eval_body(finally_body, &env)?;
                                return Ok(value);
                            }
                            // A failed handler always throws (`recur`
                            // cannot cross `try`: the analyzer fences
                            // it — expr.rs `no_recur`). A throwing
                            // `finally` masks the in-flight exception
                            // (probe29).
                            Err(_) => {
                                if let Err(signal @ Signal::Throw(_)) =
                                    self.eval_body(finally_body, &env)
                                {
                                    return Err(signal);
                                }
                                return result;
                            }
                        }
                    }
                }
                if let Err(signal @ Signal::Throw(_)) = self.eval_body(finally_body, env) {
                    return Err(signal);
                }
                Err(Signal::Throw(thrown))
            }
        }
    }

    /// Build a closure over the current frame.
    fn make_closure(&mut self, fun: &crate::expr::FunExpr, env: &Env) -> Closure {
        Closure {
            id: self.next_id(),
            display: Arc::from(fun.display.as_str()),
            arities: fun.arities.clone(),
            env: env.clone(),
            meta: None,
            conveyed: None,
        }
    }

    /// `(delay* body…)`: trap a zero-arg closure over the current
    /// frame in an unrealized delay (forced by `deref`/`force`).
    fn eval_delay_star(&mut self, display: &str, body: &[Expr], env: &Env) -> Value {
        let fun = crate::expr::FunExpr {
            display: display.to_owned(),
            arities: vec![crate::expr::Arity {
                fixed: Vec::new(),
                rest: None,
                body: body.to_vec(),
            }],
        };
        let closure = self.make_closure(&fun, env);
        Value::Delay(Delay::new(Value::Extension(Arc::new(closure))))
    }

    /// `(locking lockee body…)`: evaluate the lockee once, hold its
    /// per-object monitor over the implicit-`do` body, return its
    /// value (nil for an empty body). A nil lockee is the JVM's
    /// synchronized-block NPE — with a port-fixed lock name (the JVM
    /// embeds a gensym'd local, P6). The guard drops on every exit
    /// path (values, throws, `recur` signals, panics), so the
    /// monitor always releases.
    fn eval_locking(&mut self, lockee: &Expr, body: &[Expr], env: &Env) -> Result<Value, Signal> {
        let target = self.eval(lockee, env)?;
        if matches!(target, Value::Nil) {
            return Err(self.throw(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot enter synchronized block because \"lock\" is null".to_owned(),
            )));
        }
        let key = crate::concurrency::monitor_key(&target);
        let _guard = crate::concurrency::locking_acquire_key(key);
        self.eval_body(body, env)
    }

    /// The §E7 `IFn` table: closures/natives, keywords, symbols,
    /// vectors, maps, sets, vars (deref-and-delegate), unbounds
    /// (the `IllegalStateException`), promises (one arg delivers),
    /// and the `ClassCastException` for everything else.
    /// Invoke any value (natives reuse this for `:rename`
    /// fallbacks and other higher-order calls).
    pub(crate) fn call(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        self.invoke(callee, args)
    }

    fn invoke(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        // Struct instances invoke as maps (S12 oracles `sm-call*`):
        // a pre-dispatch, not a new table arm, so the complexity
        // budget holds (`invoke_map` names the plain struct class
        // for arity errors and `lookup` already reads structs, so
        // 1/2-arity, `apply`, and seq-fn-as-fn all follow).
        if crate::types::is_host_struct(callee) {
            return self.invoke_map(callee, args);
        }
        match callee {
            Value::Nil => Err(self.throw(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke nil as a function".to_owned(),
            ))),
            Value::Extension(_) => self.invoke_extension(callee, args),
            Value::Keyword(_) | Value::Symbol(_) => self.invoke_keyed_name(callee, args),
            Value::Vector(_)
            | Value::PrimitiveVector(_)
            | Value::MapEntry { .. }
            | Value::Map(_)
            | Value::Set(_) => self.invoke_coll(callee, args),
            Value::Var(v) => self.invoke_var(v, args),
            Value::Unbound(u) => Err(self.throw(NativeError::new(
                ThrowClass::IllegalStateException,
                format!("Attempting to call unbound fn: #'{}", u.sym),
            ))),
            Value::Promise(p) => self.invoke_promise(p, args),
            Value::MultiFn(m) => crate::multi::invoke_multifn(self, m, args),
            Value::Ref(r) => {
                let value = crate::stm::tx_deref(self, r);
                self.invoke(&value, args)
            }
            _ => Err(self.throw(NativeError::cast(&jvm_class_of(callee), "clojure.lang.IFn"))),
        }
    }

    /// Invoke a collection as a function (split from
    /// [`invoke`](Self::invoke) for the complexity budget).
    fn invoke_coll(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        match callee {
            Value::Vector(v) => self.invoke_vector(v, args),
            Value::PrimitiveVector(p) => self.invoke_pv(p, args),
            Value::MapEntry { key, val, .. } => self.invoke_mapentry(key, val, args),
            Value::Map(_) => self.invoke_map(callee, args),
            Value::Set(s) => self.invoke_set(s, args),
            _ => unreachable!("invoke collection partition"),
        }
    }

    /// Invoke a keyword/symbol by name (split from [`invoke`](Self::invoke)
    /// for the complexity budget — the shared [`invoke_keyed`](Self::invoke_keyed)
    /// names arity errors per kind).
    fn invoke_keyed_name(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        let name = match callee {
            Value::Keyword(kw) => kw.to_string(),
            _ => "clojure.lang.Symbol".to_owned(),
        };
        self.invoke_keyed(callee, &name, args)
    }

    fn invoke_keyed(
        &mut self,
        callee: &Value,
        name: &str,
        args: &[Value],
    ) -> Result<Value, Signal> {
        match args {
            [coll] => Ok(lookup(coll, callee).unwrap_or(Value::Nil)),
            [coll, not_found] => Ok(lookup(coll, callee).unwrap_or_else(|| not_found.clone())),
            _ => Err(self.throw(NativeError::arity(name, args.len()))),
        }
    }

    fn invoke_map(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        let name = jvm_class_of(callee);
        match args {
            [key] => Ok(lookup(callee, key).unwrap_or(Value::Nil)),
            [key, not_found] => Ok(lookup(callee, key).unwrap_or_else(|| not_found.clone())),
            _ => Err(self.throw(NativeError::arity(&name, args.len()))),
        }
    }

    fn invoke_set(&mut self, s: &clj_value::Set, args: &[Value]) -> Result<Value, Signal> {
        match args {
            [key] => Ok(member(s, key).unwrap_or(Value::Nil)),
            [key, not_found] => Ok(member(s, key).unwrap_or_else(|| not_found.clone())),
            _ => Err(self.throw(NativeError::arity(
                "clojure.lang.PersistentHashSet",
                args.len(),
            ))),
        }
    }

    /// Invoke a promise: exactly one argument delivers it (the
    /// first delivery wins, returning the promise; later ones
    /// return `nil`). Any other arity is the `ArityException`
    /// (the JVM's undeclared reify overloads throw
    /// `AbstractMethodError` instead — a pinned divergence, since
    /// the port has no such class).
    fn invoke_promise(&mut self, p: &Promise, args: &[Value]) -> Result<Value, Signal> {
        if args.len() != 1 {
            return Err(self.throw(NativeError::arity(p.jvm_class(), args.len())));
        }
        if p.deliver(args[0].clone()) {
            Ok(Value::Promise(p.clone()))
        } else {
            Ok(Value::Nil)
        }
    }

    /// Invoke an extension value: closures apply, natives run,
    /// struct accessors read, anything else is the `IFn` cast error.
    fn invoke_extension(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        if let Some(closure) = as_closure(callee) {
            return self.apply_closure(closure, args);
        }
        if let Some(native) = as_native(callee) {
            let func = Arc::clone(&native.func);
            return func(self, args).map_err(|e| self.throw(e));
        }
        if let Some(sv) = crate::lazy::as_subvector(callee) {
            return self.invoke_subvector(sv, args);
        }
        if let Some(accessor) = crate::collmisc::as_accessor(callee) {
            return self.invoke_accessor(accessor, args);
        }
        Err(self.throw(NativeError::cast(&jvm_class_of(callee), "clojure.lang.IFn")))
    }

    /// Invoke a struct accessor (S12): exactly one struct argument
    /// of the accessor's own basis, answered with the basis key's
    /// value (oracles `sm-accessor-arity0`, `sm-accessor-arity2`,
    /// `sm-accessor-nil`, `sm-accessor-mismatch`).
    fn invoke_accessor(
        &mut self,
        accessor: &crate::collmisc::AccessorValue,
        args: &[Value],
    ) -> Result<Value, Signal> {
        if args.len() != 1 {
            return Err(self.throw(NativeError::arity(
                "clojure.lang.PersistentStructMap/1",
                args.len(),
            )));
        }
        if matches!(&args[0], Value::Nil) {
            return Err(self.throw(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot read field \"def\" because \"m\" is null".to_owned(),
            )));
        }
        match crate::types::struct_inst(&args[0]) {
            Some(i) if Arc::ptr_eq(accessor.def(), &i.def) => {
                Ok(i.content.get(accessor.key()).cloned().unwrap_or(Value::Nil))
            }
            Some(_) => Err(self.throw(NativeError::new(
                ThrowClass::RuntimeException,
                "Accessor/struct mismatch".to_owned(),
            ))),
            None => Err(self.throw(crate::core::cce_err(
                &args[0],
                "clojure.lang.PersistentStructMap",
            ))),
        }
    }

    /// Invoke a vector: exactly one integer key (arity, key-type,
    /// then bounds, §E7).
    fn invoke_vector(&mut self, v: &clj_value::Vector, args: &[Value]) -> Result<Value, Signal> {
        if args.len() != 1 {
            return Err(self.throw(NativeError::arity(
                "clojure.lang.PersistentVector",
                args.len(),
            )));
        }
        let Value::Int(n) = &args[0] else {
            return Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Key must be integer".to_owned(),
            )));
        };
        match usize::try_from(*n).ok().and_then(|i| v.iter().nth(i)) {
            Some(elem) => Ok(elem.clone()),
            None => Err(self.throw(NativeError::new(
                ThrowClass::IndexOutOfBoundsException,
                String::new(),
            ))),
        }
    }

    /// Invoke a primitive vector: exactly one `isInteger` key
    /// (`Int`/`BigInt`/`BigInteger` — `Short`/`Double`/nil and the
    /// rest fail `Key must be integer`, OOB is the empty `IOOBE`,
    /// any other arity is the `ArityException` naming
    /// `clojure.core.Vec` — the JVM throws `AbstractMethodError`
    /// for the missing overloads, which the port pins as
    /// `ArityException`, promise precedent; S15 oracles `f-invoke`,
    /// `f-invoke-2`, `f-invoke-bad`, `f-invoke-oob`).
    fn invoke_pv(&mut self, p: &PrimitiveVector, args: &[Value]) -> Result<Value, Signal> {
        if args.len() != 1 {
            return Err(self.throw(NativeError::arity("clojure.core.Vec", args.len())));
        }
        // The `isInteger` gate passes `Int`/`BigInt`/`BigInteger`
        // only; the key then coerces through core-`int` (huge
        // `BigInt` fails the `long` stop with exact digits —
        // oracle `g-vo-hugekey`).
        if !matches!(
            &args[0],
            Value::Int(_) | Value::BigInt(_) | Value::BigInteger(_)
        ) {
            return Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Key must be integer".to_owned(),
            )));
        }
        // `int_cast_value` is core-`int` minus the arity check
        // (the key slice is always exactly one arg): it
        // answers the `i64` directly, so no `Int` unwrap
        // remains (the old `unreachable!` could never
        // execute — `int` only answers `Int`).
        let i = crate::core::int_cast_value(&args[0]).map_err(|e| self.throw(e))?;
        // Core-`int` range-checks, so the `i64` always fits `i32`.
        match usize::try_from(i as i32).ok().and_then(|idx| p.get(idx)) {
            Some(elem) => Ok(elem.clone()),
            None => Err(self.throw(NativeError::new(
                ThrowClass::IndexOutOfBoundsException,
                String::new(),
            ))),
        }
    }

    /// Invoke a map entry: exactly one integer key (the arity error
    /// names `clojure.lang.MapEntry`, like the JVM — Oracle FX2
    /// `invoke`, `invoke-0`, `invoke-2`, `invoke-kw`).
    fn invoke_mapentry(
        &mut self,
        key: &Value,
        val: &Value,
        args: &[Value],
    ) -> Result<Value, Signal> {
        if args.len() != 1 {
            return Err(self.throw(NativeError::arity("clojure.lang.MapEntry", args.len())));
        }
        let Value::Int(n) = &args[0] else {
            return Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Key must be integer".to_owned(),
            )));
        };
        let found = match usize::try_from(*n).ok() {
            Some(0) => Some(key),
            Some(1) => Some(val),
            _ => None,
        };
        match found {
            Some(elem) => Ok(elem.clone()),
            None => Err(self.throw(NativeError::new(
                ThrowClass::IndexOutOfBoundsException,
                String::new(),
            ))),
        }
    }

    /// Invoke a subvector: exactly one integer key (the arity error
    /// names `APersistentVector/SubVector`, like the JVM).
    fn invoke_subvector(
        &mut self,
        sv: &clj_value::SubVectorValue,
        args: &[Value],
    ) -> Result<Value, Signal> {
        if args.len() != 1 {
            return Err(self.throw(NativeError::arity(
                "clojure.lang.APersistentVector/SubVector",
                args.len(),
            )));
        }
        let Value::Int(n) = &args[0] else {
            return Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Key must be integer".to_owned(),
            )));
        };
        match usize::try_from(*n).ok().and_then(|i| sv.nth(i)) {
            Some(elem) => Ok(elem.clone()),
            None => Err(self.throw(NativeError::new(
                ThrowClass::IndexOutOfBoundsException,
                String::new(),
            ))),
        }
    }

    /// Invoke a var: deref and delegate (unbounds are the
    /// `IllegalStateException`).
    fn invoke_var(&mut self, v: &Var, args: &[Value]) -> Result<Value, Signal> {
        let root = self.deref_var(v.id, &v.sym);
        if matches!(root, Value::Unbound(_)) {
            return Err(self.throw(NativeError::new(
                ThrowClass::IllegalStateException,
                format!("Attempting to call unbound fn: #'{}", v.sym),
            )));
        }
        self.invoke(&root, args)
    }

    /// Apply a closure: fixed arities by count, else the variadic
    /// arity (`rest` binds `nil` when empty, §E5). A `bound-fn`
    /// closure re-pushes its conveyed bindings as one frame around
    /// the call (popped in a `finally`, like the JVM's
    /// `resetThreadBindingFrame`).
    fn apply_closure(&mut self, closure: &Closure, args: &[Value]) -> Result<Value, Signal> {
        if let Some(conveyed) = closure.conveyed.clone() {
            let depth = self.bindings.depth();
            let frame: Vec<(Var, Value)> = conveyed
                .pairs()
                .map(|(var, v)| (var.clone(), v.clone()))
                .collect();
            self.bindings.push_frame(frame);
            let result = self.apply_closure_inner(closure, args);
            self.bindings.truncate(depth);
            return result;
        }
        self.apply_closure_inner(closure, args)
    }

    /// The arity dispatch behind [`apply_closure`](Self::apply_closure).
    fn apply_closure_inner(&mut self, closure: &Closure, args: &[Value]) -> Result<Value, Signal> {
        let mut variadic: Option<&crate::expr::Arity> = None;
        for arity in &closure.arities {
            if arity.rest.is_none() && arity.fixed.len() == args.len() {
                return self.apply_arity(closure, arity, args, None);
            }
            if arity.rest.is_some() {
                variadic = Some(arity);
            }
        }
        if let Some(arity) = variadic.filter(|a| args.len() >= a.fixed.len()) {
            let rest = rest_list(&args[arity.fixed.len()..]);
            return self.apply_arity(closure, arity, args, Some(rest));
        }
        Err(self.throw(NativeError::arity(&closure.display, args.len())))
    }

    /// Bind one arity's parameters (sequentially, like `let`) and
    /// run the body, looping on `recur`.
    fn apply_arity(
        &mut self,
        closure: &Closure,
        arity: &crate::expr::Arity,
        args: &[Value],
        rest: Option<Value>,
    ) -> Result<Value, Signal> {
        let mut values: Vec<Value> = args[..arity.fixed.len()].to_vec();
        if let Some(rest) = rest {
            values.push(rest);
        }
        let pats: Vec<&Pattern> = arity.fixed.iter().chain(arity.rest.iter()).collect();
        // Bind through `bind_pattern` (initial and `recur` alike)
        // so each param's `:or` defaults evaluate against the
        // pre-extension frame the analyzer saw (#171). A fresh
        // frame per `recur` keeps the old copy-on-recur promise:
        // closures from this iteration keep the old frame.
        let outer = closure.env.clone();
        let mut frame = outer.clone();
        for (pat, value) in pats.iter().zip(values) {
            frame = self.bind_pattern(pat, value, &frame)?;
        }
        loop {
            match self.eval_body(&arity.body, &frame) {
                Ok(v) => return Ok(v),
                Err(Signal::Recur(values)) => {
                    let mut next = outer.clone();
                    for (pat, value) in pats.iter().zip(values) {
                        next = self.bind_pattern(pat, value, &next)?;
                    }
                    frame = next;
                }
                Err(signal) => return Err(signal),
            }
        }
    }

    /// Destructure a value against a pattern, pushing bound values
    /// in [`pattern_names`] order. Returns the computed outlet
    /// values (requested or forced) for `subs` projection.
    fn destructure(
        &mut self,
        pat: &Pattern,
        value: &Value,
        env: &Env,
        out: &mut Vec<Value>,
        force: ForceSet,
    ) -> Result<OutletVals, Signal> {
        match pat {
            Pattern::Symbol(_) => {
                out.push(value.clone());
                Ok(OutletVals::default())
            }
            Pattern::Vec(v) => {
                // Vectors neither force nor project (`(map? bb)`).
                self.destructure_seq(v, value, env, out)?;
                Ok(OutletVals::default())
            }
            Pattern::Map(m) => self.bind_map(m, value, env, out, force),
        }
    }

    /// `destvec*`: without `&`, positional `nth`-with-`nil` (wrong
    /// types are `nth not supported`); with `&`, `first`/`next`
    /// chaining; `:as` takes the whole.
    fn destructure_seq(
        &mut self,
        pat: &crate::expr::VecPattern,
        value: &Value,
        env: &Env,
        out: &mut Vec<Value>,
    ) -> Result<(), Signal> {
        // `Some` first: the binding proves the shape with no
        // second discrimination.
        if let Some(rest) = pat.rest.as_ref() {
            // `destvec*` binds `gseq` to `(seq gvec)`: the `&`
            // pattern gets a seq (`nil` when empty) and
            // non-seqables raise the `ISeq` error (§E12).
            let mut current = self.destructure_gseq(value)?;
            for fixed in &pat.fixed {
                let first = crate::lazy::first_value(self, &current)?;
                let _ = self.destructure(fixed, &first, env, out, ForceSet::default())?;
                current = crate::lazy::next_value(self, &current)?;
            }
            let _ = self.destructure(rest, &current, env, out, ForceSet::default())?;
        } else {
            for (i, fixed) in pat.fixed.iter().enumerate() {
                let nth = nth_or_nil(value, i, self)?;
                let _ = self.destructure(fixed, &nth, env, out, ForceSet::default())?;
            }
        }
        if pat.as_sym.is_some() {
            out.push(value.clone());
        }
        Ok(())
    }

    /// The `&`-pattern `gseq`: seq values (S6D: `pmap`'s `rets`)
    /// coerce through the canonical `seq` (which forces), while
    /// concrete collections keep `seq_val`'s eager lists.
    fn destructure_gseq(&mut self, value: &Value) -> Result<Value, Signal> {
        match value {
            Value::Extension(_) => crate::lazy::seq_value(self, value),
            _ => self.seq_val(value),
        }
    }

    /// `destmap*`: seq-convert, eager `:or`, entries, keys,
    /// outlets. Pushes values in [`pattern_names`] order (entries,
    /// keys, `:as`, `:select`, `:all`, `:excess`, `:missing`) and
    /// returns every computed outlet (requested or forced) for the
    /// parent's `subs` projection — each nested pattern binds once,
    /// so `:or` defaults fire exactly once.
    fn bind_map(
        &mut self,
        pat: &MapPattern,
        value: &Value,
        env: &Env,
        out: &mut Vec<Value>,
        force: ForceSet,
    ) -> Result<OutletVals, Signal> {
        let converted = self.seq_to_map(value)?;
        let defaults = self.eval_defaults(pat, env)?;
        let ctx = MapBind {
            pat,
            env,
            converted: &converted,
            defaults: &defaults,
            force,
        };
        // Entries and keys (missing keys take the local-then-key
        // default; required keys `req!`). `:missing` starts absent:
        // nothing missing means `nil`, not `{}`.
        let mut missing: Option<Map> = None;
        let mut subs = SubMaps::default();
        self.bind_entries(&ctx, &mut missing, &mut subs, out)?;
        self.bind_key_list(&ctx, &mut missing, out)?;
        if pat.as_sym.is_some() {
            out.push(converted.clone());
        }
        self.bind_outlets(&ctx, &mut missing, &subs, out)
    }

    /// The eager `:or` defaults (evaluated once, before any lookup).
    fn eval_defaults(&mut self, pat: &MapPattern, env: &Env) -> Result<Map, Signal> {
        let mut defaults: Map = new_map();
        for (key, default) in &pat.or {
            let val = self.eval(default, env)?;
            defaults = defaults.insert((*key).clone(), val);
        }
        Ok(defaults)
    }

    /// The `symbol keyword` entries, accumulating `subs` (entry
    /// key → outlet value) from each nested map's single bind.
    fn bind_entries(
        &mut self,
        ctx: &MapBind,
        missing: &mut Option<Map>,
        subs: &mut SubMaps,
        out: &mut Vec<Value>,
    ) -> Result<(), Signal> {
        for (local, key, or_key, required) in &ctx.pat.entries {
            // `:or` is keyed by the binding local first, then the
            // key form as written (upstream `destmap*` `push1`;
            // #158); nested locals have no `Value` form, so only
            // symbols get the local lookup.
            let local_val = match local {
                Pattern::Symbol(name) => Some(Value::Symbol(Symbol::new(None, name))),
                _ => None,
            };
            let val = self.map_binding(ctx, local_val.as_ref(), key, or_key, *required, missing)?;
            // Forcing (`destmap*` gensym outlets): an effective
            // outlet the nested map lacks is still computed, for
            // `subs` (never bound).
            let force = ForceSet {
                select: ctx.effective(Outlet::Select) && !has_outlet(local, Outlet::Select),
                all: ctx.effective(Outlet::All) && !has_outlet(local, Outlet::All),
                excess: ctx.effective(Outlet::Excess) && !has_outlet(local, Outlet::Excess),
                missing: ctx.effective(Outlet::Missing) && !has_outlet(local, Outlet::Missing),
            };
            let ov = self.destructure(local, &val, ctx.env, out, force)?;
            if matches!(local, Pattern::Map(_)) {
                insert_sub(&mut subs.select, key, ov.select);
                insert_sub(&mut subs.all, key, ov.all);
                insert_sub(&mut subs.excess, key, ov.excess);
                insert_sub(&mut subs.missing, key, ov.missing);
            }
        }
        Ok(())
    }

    /// The `:keys`/`:strs`/`:syms` bindings.
    fn bind_key_list(
        &mut self,
        ctx: &MapBind,
        missing: &mut Option<Map>,
        out: &mut Vec<Value>,
    ) -> Result<(), Signal> {
        for key in &ctx.pat.keys {
            let local_key = Value::Symbol(Symbol::new(None, &key.local));
            // Directive keys are built, never quoted: the or-key is
            // the key itself (#158).
            let val = self.map_binding(
                ctx,
                Some(&local_key),
                &key.key,
                &key.key,
                key.required,
                missing,
            )?;
            out.push(val);
        }
        Ok(())
    }

    /// The 1.13 outlets, computed exactly when effective
    /// (requested or forced), bound only when requested. Merges use
    /// JVM `merge` semantics over the *converted* target
    /// (`nil`-skipping; non-map survivors pass through).
    fn bind_outlets(
        &mut self,
        ctx: &MapBind,
        missing: &mut Option<Map>,
        subs: &SubMaps,
        out: &mut Vec<Value>,
    ) -> Result<OutletVals, Signal> {
        let pat = ctx.pat;
        let dm = or_mapped(pat, ctx.defaults);
        // `(if (empty? dm) nil dm)`.
        let dm_v = if dm.is_empty() {
            Value::Nil
        } else {
            Value::Map(dm)
        };
        let sel = selection(pat);
        let mut ov = OutletVals::default();
        if ctx.effective(Outlet::Select) {
            let mm = self.jvm_merge3(
                dm_v.clone(),
                ctx.converted.clone(),
                opt_val(subs.select.clone()),
            )?;
            // `when-let`: an all-absent merge binds `nil`.
            let v = if matches!(mm, Value::Nil) {
                Value::Nil
            } else {
                self.select_keys_general(&mm, &sel)?
            };
            emit_outlet(out, &mut ov.select, pat.select.is_some(), v);
        }
        if ctx.effective(Outlet::All) {
            let v = self.jvm_merge3(
                dm_v.clone(),
                ctx.converted.clone(),
                opt_val(subs.all.clone()),
            )?;
            emit_outlet(out, &mut ov.all, pat.all.is_some(), v);
        }
        if ctx.effective(Outlet::Excess) {
            let base = self.dissoc_opt(ctx.converted, &sel)?;
            let v = self.jvm_merge2(opt_val(base), opt_val(subs.excess.clone()))?;
            emit_outlet(out, &mut ov.excess, pat.excess.is_some(), v);
        }
        if ctx.effective(Outlet::Missing) {
            let v = self.jvm_merge2(opt_val(missing.take()), opt_val(subs.missing.clone()))?;
            emit_outlet(out, &mut ov.missing, pat.missing.is_some(), v);
        }
        Ok(ov)
    }

    /// One map binding: present keys read, missing keys take the
    /// local-then-key default, missing required keys `req!` (or
    /// record into `:missing` and bind `nil`). Collection honors
    /// *forced* `:missing` too (upstream a1041abf1): an enclosing
    /// `:missing` injects a real outlet into nested maps, so their
    /// required keys collect instead of throwing.
    fn map_binding(
        &mut self,
        ctx: &MapBind,
        local: Option<&Value>,
        key: &Value,
        or_key: &Value,
        required: bool,
        missing: &mut Option<Map>,
    ) -> Result<Value, Signal> {
        if let Some(val) = lookup(ctx.converted, key) {
            return Ok(val);
        }
        let default = local
            .and_then(|local| ctx.defaults.get(local).cloned())
            .or_else(|| ctx.defaults.get(or_key).cloned());
        if required && default.is_none() {
            if ctx.effective(Outlet::Missing) {
                let m = missing.take().unwrap_or_else(new_map);
                *missing = Some(m.insert(key.clone(), Value::Nil));
                return Ok(Value::Nil);
            }
            return Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                format!("Missing required key: {}", pr_of(key)),
            )));
        }
        Ok(default.unwrap_or(Value::Nil))
    }

    /// JVM 2-`merge`: `nil`s skipped, else `into1`.
    fn jvm_merge2(&mut self, x: Value, y: Value) -> Result<Value, Signal> {
        match (&x, &y) {
            (Value::Nil, Value::Nil) => Ok(Value::Nil),
            (Value::Nil, _) => Ok(y),
            (_, Value::Nil) => Ok(x),
            _ => self.into1(x, &y),
        }
    }

    /// JVM 3-`merge`: left-associated `jvm_merge2`.
    fn jvm_merge3(&mut self, x: Value, y: Value, z: Value) -> Result<Value, Signal> {
        let merged = self.jvm_merge2(x, y)?;
        self.jvm_merge2(merged, z)
    }

    /// `select-keys` over a merge result: present keys read
    /// (`lookup`: maps, int-keyed vectors/strings, sets),
    /// absent keys skipped — non-map targets select `{}`.
    fn select_keys_general(&mut self, merged: &Value, sel: &[Value]) -> Result<Value, Signal> {
        let mut out = new_map();
        for key in sel {
            if let Some(val) = lookup(merged, key) {
                out = out.insert(key.clone(), val.clone());
            }
        }
        Ok(Value::Map(out))
    }

    /// A `dissoc` base: `nil` is absent, maps shed the selection
    /// (`not-empty`), anything else is the `IPersistentMap` cast error.
    fn dissoc_opt(&mut self, converted: &Value, sel: &[Value]) -> Result<Option<Map>, Signal> {
        match converted {
            Value::Nil => Ok(None),
            Value::Map(m) => {
                let mut out = m.clone();
                for key in sel {
                    out = out.without(key);
                }
                Ok((!out.is_empty()).then_some(out))
            }
            other => Err(self.throw(NativeError::cast(
                &jvm_class_of(other),
                "clojure.lang.IPersistentMap",
            ))),
        }
    }

    /// `into1`: map `from`s conj each entry pair; seq `from`s
    /// conj each element (real `merge`: `nil`s skip,
    /// pair-vectors assoc, unseqables are the `ISeq` error).
    /// `to` is always the `:or` map on the seq path (the second
    /// merge arg is a map unless it is the converted target,
    /// and then the first is `:or` defaults).
    fn into1(&mut self, to: Value, from: &Value) -> Result<Value, Signal> {
        if let Value::Map(m) = from {
            let mut out = to;
            for (k, v) in m.iter() {
                out = self.conj_entry_val(out, k.clone(), v.clone())?;
            }
            return Ok(out);
        }
        match self.seq_opt(from)? {
            None => Ok(to),
            Some(l) => {
                let mut out = to;
                for elem in l.iter() {
                    out = self.conj_one(out, elem.clone())?;
                }
                Ok(out)
            }
        }
    }

    /// One real entry conjed: maps assoc, vectors/lists/sets take
    /// the `[k v]` form, `nil` starts a list, anything else is the
    /// `IPersistentCollection` cast error.
    fn conj_entry_val(&mut self, coll: Value, k: Value, v: Value) -> Result<Value, Signal> {
        if let Value::Map(m) = coll {
            return Ok(Value::Map(m.insert(k, v)));
        }
        let entry = clj_value::map_entry(k, v);
        self.conj_one(coll, entry)
    }

    /// `conj` of one element: lists prepend, vectors append, sets
    /// insert, maps take entries, `nil` starts a list, anything else
    /// is the `IPersistentCollection` cast error.
    fn conj_one(&mut self, coll: Value, elem: Value) -> Result<Value, Signal> {
        // `nil` never reaches this conj (`into1` merge args are
        // non-`nil`), so it falls into the cast error below.
        match coll {
            Value::Map(m) => self.conj_map(m, elem).map(Value::Map),
            Value::Vector(v) => Ok(Value::Vector(v.push_back(elem))),
            Value::PrimitiveVector(p) => {
                let co = crate::arrays::vector_coerce(self, p.elem(), &elem)
                    .map_err(|e| self.throw(e))?;
                Ok(Value::PrimitiveVector(p.push_back(co)))
            }
            Value::MapEntry { key, val, .. } => {
                let tmp = Vector::from_vec(vec![*key, *val]);
                Ok(Value::Vector(tmp.push_back(elem)))
            }
            Value::List(l) => Ok(Value::List(l.push_front(elem))),
            Value::Set(s) => Ok(Value::Set(s.insert(elem))),
            other => Err(self.throw(NativeError::cast(
                &jvm_class_of(&other),
                "clojure.lang.IPersistentCollection",
            ))),
        }
    }

    /// `conj` onto a map: `nil` is identity, maps merge, 2-vectors
    /// assoc, other vectors are the pair error, seqables contribute
    /// entries (non-entries are the `Map$Entry` cast error).
    fn conj_map(&mut self, mut map: Map, elem: Value) -> Result<Map, Signal> {
        match elem {
            Value::Nil => Ok(map),
            Value::Map(other) => {
                for (k, v) in other.iter() {
                    map = map.insert(k.clone(), v.clone());
                }
                Ok(map)
            }
            Value::Vector(v) => {
                let items: Vec<Value> = v.iter().cloned().collect();
                match items.as_slice() {
                    [k, val] => Ok(map.insert(k.clone(), val.clone())),
                    _ => Err(self.throw(NativeError::new(
                        ThrowClass::IllegalArgumentException,
                        "Vector arg to map conj must be a pair".to_owned(),
                    ))),
                }
            }
            // `Vec` is an `IPersistentVector`: pairs assoc like vectors.
            Value::PrimitiveVector(p) => match p.items().as_slice() {
                [k, val] => Ok(map.insert(k.clone(), val.clone())),
                _ => Err(self.throw(NativeError::new(
                    ThrowClass::IllegalArgumentException,
                    "Vector arg to map conj must be a pair".to_owned(),
                ))),
            },
            Value::MapEntry { key, val, .. } => Ok(map.insert(*key, *val)),
            other => self.conj_map_seq(map, other),
        }
    }

    /// `conj` a seqable element onto a map (split from
    /// [`conj_map`](Self::conj_map) for the complexity budget):
    /// one pair per entry (Oracle FX2 `conj-seq-entries`); the
    /// first non-entry is the `Map$Entry` cast error.
    fn conj_map_seq(&mut self, mut map: Map, elem: Value) -> Result<Map, Signal> {
        match self.seq_opt(&elem)? {
            None => Ok(map),
            Some(l) => {
                for head in l.iter() {
                    let Value::MapEntry { key, val, .. } = head else {
                        return Err(self.throw(cast_entry(head)));
                    };
                    map = map.insert((**key).clone(), (**val).clone());
                }
                Ok(map)
            }
        }
    }

    /// `seq-to-map-for-destructuring`: only lists convert (S4 has no
    /// other `ISeq`s). Empty is `{}`, one element is the element,
    /// longer reads pairs plus the trailing rule (§E12).
    pub(crate) fn seq_to_map(&mut self, value: &Value) -> Result<Value, Signal> {
        let Value::List(list) = value else {
            return Ok(value.clone());
        };
        let items: Vec<&Value> = list.iter().collect();
        match items.as_slice() {
            [] => Ok(Value::Map(new_map())),
            [single] => Ok((*single).clone()),
            _ => {
                let mut map = new_map();
                let mut pairs = &items[..];
                // Pairs first, trailing last (`(conj (apply hash-map
                // (pop v)) (peek v))`): the trailer wins duplicates.
                let trailing = if pairs.len() % 2 == 1 {
                    let last = pairs[pairs.len() - 1];
                    pairs = &pairs[..pairs.len() - 1];
                    Some(last)
                } else {
                    None
                };
                for i in (0..pairs.len()).step_by(2) {
                    map = map.insert(pairs[i].clone(), pairs[i + 1].clone());
                }
                if let Some(last) = trailing {
                    self.cons_trailing(&mut map, last)?;
                }
                Ok(Value::Map(map))
            }
        }
    }

    /// The odd-tail rule: maps merge, 2-vectors assoc, `nil`/empty
    /// seqables vanish, seqable elements are entries-or-`CCE`,
    /// anything else is the `ISeq` error (§E12).
    fn cons_trailing(&mut self, map: &mut Map, trailing: &Value) -> Result<(), Signal> {
        match trailing {
            Value::Map(m) => {
                for (k, v) in m.iter() {
                    *map = std::mem::replace(map, new_map()).insert(k.clone(), v.clone());
                }
                Ok(())
            }
            Value::Vector(v) => {
                let items: Vec<&Value> = v.iter().collect();
                if items.len() != 2 {
                    return Err(self.throw(NativeError::new(
                        ThrowClass::IllegalArgumentException,
                        "Vector arg to map conj must be a pair".to_owned(),
                    )));
                }
                *map = std::mem::replace(map, new_map()).insert(items[0].clone(), items[1].clone());
                Ok(())
            }
            // `Vec` is an `IPersistentVector`: pairs assoc like vectors.
            Value::PrimitiveVector(p) => {
                let items = p.items();
                if items.len() != 2 {
                    return Err(self.throw(NativeError::new(
                        ThrowClass::IllegalArgumentException,
                        "Vector arg to map conj must be a pair".to_owned(),
                    )));
                }
                *map = std::mem::replace(map, new_map()).insert(items[0].clone(), items[1].clone());
                Ok(())
            }
            Value::Nil => Ok(()),
            // An entry trailer conjs as its pair (Oracle FX2
            // `trailer-entry`).
            Value::MapEntry { key, val, .. } => {
                *map = std::mem::replace(map, new_map()).insert((**key).clone(), (**val).clone());
                Ok(())
            }
            Value::List(l) => {
                let elems: Vec<Value> = l.iter().cloned().collect();
                self.fold_seqable_trailing(map, elems)
            }
            Value::Array(a) => {
                let elems: Vec<Value> = a.snapshot();
                self.fold_seqable_trailing(map, elems)
            }
            Value::Set(s) => {
                let elems: Vec<Value> = s.iter().cloned().collect();
                self.fold_seqable_trailing(map, elems)
            }
            Value::Str(s) => {
                let elems: Vec<Value> = s.chars().map(Value::Char).collect();
                self.fold_seqable_trailing(map, elems)
            }
            other => Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                format!(
                    "Don't know how to create ISeq from: {}",
                    jvm_class_of(other)
                ),
            ))),
        }
    }

    /// The odd-tail seqable rule: empties vanish, every entry
    /// conjs as its pair (Oracle FX2 `trailer-listentry`), and the
    /// first non-entry is the entry-cast error (issue #70: the JVM
    /// `APersistentMap.cons` loop folds EVERY seqed element).
    fn fold_seqable_trailing(&mut self, map: &mut Map, elems: Vec<Value>) -> Result<(), Signal> {
        for elem in elems {
            match elem {
                Value::MapEntry { key, val, .. } => {
                    *map = std::mem::replace(map, new_map()).insert(*key, *val);
                }
                other => {
                    return Err(self.throw(NativeError::cast(
                        &jvm_class_of(&other),
                        "java.util.Map$Entry",
                    )));
                }
            }
        }
        Ok(())
    }
}

/// The shared inputs to one map-pattern bind.
struct MapBind<'a> {
    pat: &'a MapPattern,
    env: &'a Env,
    converted: &'a Value,
    defaults: &'a Map,
    force: ForceSet,
}

impl MapBind<'_> {
    /// An outlet is effective when requested or forced.
    fn effective(&self, outlet: Outlet) -> bool {
        let requested = match outlet {
            Outlet::Select => self.pat.select.is_some(),
            Outlet::All => self.pat.all.is_some(),
            Outlet::Excess => self.pat.excess.is_some(),
            Outlet::Missing => self.pat.missing.is_some(),
        };
        requested || self.force.get(outlet)
    }
}

/// The four nested outlets (`:subs`/`:suba`/`:subexcess`/`:submissing`).
#[derive(Clone, Copy)]
enum Outlet {
    Select,
    All,
    Excess,
    Missing,
}

/// Inherited outlet forcing: an effective outlet the nested map
/// lacks is still computed, for `subs` (never bound).
#[derive(Clone, Copy, Default)]
struct ForceSet {
    select: bool,
    all: bool,
    excess: bool,
    missing: bool,
}

impl ForceSet {
    fn get(self, outlet: Outlet) -> bool {
        match outlet {
            Outlet::Select => self.select,
            Outlet::All => self.all,
            Outlet::Excess => self.excess,
            Outlet::Missing => self.missing,
        }
    }
}

/// Computed outlet values (`Some` ⟺ effective; values may be `nil`).
#[derive(Default)]
struct OutletVals {
    select: Option<Value>,
    all: Option<Value>,
    excess: Option<Value>,
    missing: Option<Value>,
}

/// `subs`: entry key → nested outlet value (`some-vals`: non-`nil`
/// only, hence never `Some`-empty).
#[derive(Default)]
struct SubMaps {
    select: Option<Map>,
    all: Option<Map>,
    excess: Option<Map>,
    missing: Option<Map>,
}

/// Whether a pattern is a map requesting an outlet (forcing and
/// `subs` only concern direct map locals).
fn has_outlet(local: &Pattern, outlet: Outlet) -> bool {
    let Pattern::Map(nested) = local else {
        return false;
    };
    match outlet {
        Outlet::Select => nested.select.is_some(),
        Outlet::All => nested.all.is_some(),
        Outlet::Excess => nested.excess.is_some(),
        Outlet::Missing => nested.missing.is_some(),
    }
}

/// Record one `subs` contribution (`nil`s dropped).
fn insert_sub(slot: &mut Option<Map>, key: &Value, value: Option<Value>) {
    let Some(v) = value else {
        return;
    };
    if !matches!(v, Value::Nil) {
        let map = slot.take().unwrap_or_else(new_map);
        *slot = Some(map.insert(key.clone(), v));
    }
}

/// An optional map as a merge argument (absent ⟺ `nil`).
fn opt_val(map: Option<Map>) -> Value {
    map.map(Value::Map).unwrap_or(Value::Nil)
}

/// Record a computed outlet value, binding it only when requested.
fn emit_outlet(out: &mut Vec<Value>, slot: &mut Option<Value>, requested: bool, v: Value) {
    if requested {
        out.push(v.clone());
    }
    *slot = Some(v);
}

/// Cast one element to `Map$Entry` (`nil` is the entry-NPE;
/// anything else fails — entries are real `MapEntry` values since
/// FX2, so callers check for them first).
fn cast_entry(entry: &Value) -> NativeError {
    match entry {
        Value::Nil => NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.util.Map$Entry.getKey()\" because \"e\" is null".to_owned(),
        ),
        other => NativeError::cast(&jvm_class_of(other), "java.util.Map$Entry"),
    }
}

/// `get` without the not-found: maps, int-keyed vectors/strings
/// (in range), sets (membership), else `None` (§E12). Never throws.
fn lookup(coll: &Value, key: &Value) -> Option<Value> {
    // Records read their content map (oracle `h-destructure`
    // rides this path); structs likewise (S12); every other host
    // value misses.
    if let Some(i) = crate::types::maplike_inst(coll) {
        return i.content.get(key).cloned();
    }
    match coll {
        Value::Map(m) => m.get(key).cloned(),
        Value::Vector(v) => lookup_vector(v, key),
        // Entries read like their 2-vector (map destructuring
        // passes integer keys — Oracle FX2 `mapdest-plain`).
        Value::MapEntry {
            key: ek, val: ev, ..
        } => lookup_entry(ek, ev, key),
        Value::Str(s) => lookup_string(s, key),
        Value::Set(s) => member(s, key),
        _ => None,
    }
}

fn lookup_vector(v: &Vector, key: &Value) -> Option<Value> {
    match key {
        Value::Int(n) => usize::try_from(*n)
            .ok()
            .and_then(|i| v.iter().nth(i).cloned()),
        _ => None,
    }
}

fn lookup_entry(key: &Value, val: &Value, index: &Value) -> Option<Value> {
    match index {
        Value::Int(n) => match usize::try_from(*n).ok() {
            Some(0) => Some(key.clone()),
            Some(1) => Some(val.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn lookup_string(s: &str, key: &Value) -> Option<Value> {
    match key {
        Value::Int(n) => usize::try_from(*n)
            .ok()
            .and_then(|i| s.chars().nth(i))
            .map(Value::Char),
        _ => None,
    }
}

/// Set membership: the key when present (`get` returns the equal
/// key; ours is identical by construction).
fn member(s: &clj_value::Set, key: &Value) -> Option<Value> {
    s.contains(key).then(|| key.clone())
}

/// `pr` of a value (assert conditions, `:or` key sets).
fn pr_of(v: &Value) -> String {
    clj_value::pr_to_string(v, &clj_value::PrintOpts::pr())
}

/// Variadic rest: `nil` when empty, else the list (§E5).
fn rest_list(args: &[Value]) -> Value {
    if args.is_empty() {
        Value::Nil
    } else {
        Value::List(List::from_vec(args.to_vec()))
    }
}

/// `(nth coll i nil)`: vectors/lists/strings (OOB is `nil`),
/// `nil` is `nil`; anything else delegates to `nth` (lazy
/// seqs force and walk, subvectors index, the rest is
/// `nth not supported`) (§E12).
fn nth_or_nil(coll: &Value, i: usize, eval: &mut Evaluator) -> Result<Value, Signal> {
    match coll {
        Value::Nil => Ok(Value::Nil),
        Value::Vector(v) => Ok(v.iter().nth(i).cloned().unwrap_or(Value::Nil)),
        Value::MapEntry { key, val, .. } => Ok(match i {
            0 => (**key).clone(),
            1 => (**val).clone(),
            _ => Value::Nil,
        }),
        Value::List(l) => Ok(l.iter().nth(i).cloned().unwrap_or(Value::Nil)),
        Value::Str(s) => Ok(s.chars().nth(i).map(Value::Char).unwrap_or(Value::Nil)),
        other => {
            let idx = Value::Int(i64::try_from(i).unwrap_or(i64::MAX));
            native_nth(eval, &[other.clone(), idx, Value::Nil]).map_err(|err| eval.throw(err))
        }
    }
}

impl Evaluator {
    /// `clojure.lang.RT.seq`: `nil` in, `nil` out; empties to
    /// `nil`; vectors/strings/sets to element lists; maps to
    /// entry lists; anything else is the `ISeq` error.
    fn seq_val(&mut self, value: &Value) -> Result<Value, Signal> {
        Ok(self.seq_opt(value)?.map(Value::List).unwrap_or(Value::Nil))
    }

    /// `seq_val` as an option-list (`None` ⟺ `nil`/empty, so
    /// match sites stay two-armed with no dead wildcard).
    fn seq_opt(&mut self, value: &Value) -> Result<Option<List>, Signal> {
        match value {
            Value::Nil => Ok(None),
            Value::List(l) => {
                if l.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(l.clone()))
                }
            }
            Value::Vector(v) => Ok(Self::seq_list_opt(v.iter().cloned().collect())),
            Value::PrimitiveVector(p) => Ok(Self::seq_list_opt(p.items())),
            Value::Array(a) => Ok(Self::seq_list_opt(a.snapshot())),
            Value::Set(s) => Ok(Self::seq_list_opt(s.iter().cloned().collect())),
            Value::Str(s) => Ok(Self::seq_list_opt(s.chars().map(Value::Char).collect())),
            Value::Map(m) => Ok(Self::seq_list_opt(
                m.iter()
                    .map(|(k, v)| clj_value::map_entry(k.clone(), v.clone()))
                    .collect(),
            )),
            Value::MapEntry { key, val, .. } => {
                Ok(Self::seq_list_opt(vec![(**key).clone(), (**val).clone()]))
            }
            other => Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                format!(
                    "Don't know how to create ISeq from: {}",
                    jvm_class_of(other)
                ),
            ))),
        }
    }

    fn seq_list_opt(elems: Vec<Value>) -> Option<List> {
        if elems.is_empty() {
            None
        } else {
            Some(List::from_vec(elems))
        }
    }
}

/// The selection: entry keys, key-list keys, and `&`-tail keys in
/// first-seen order (source order; the JVM iterates a hash-ordered
/// set, §E12 documents the divergence).
fn selection(pat: &MapPattern) -> Vec<Value> {
    let mut sel = Vec::new();
    for (_, key, _, _) in &pat.entries {
        if !sel.contains(key) {
            sel.push(key.clone());
        }
    }
    for key in &pat.keys {
        if !sel.contains(&key.key) {
            sel.push(key.key.clone());
        }
    }
    for key in &pat.extra_keys {
        if !sel.contains(key) {
            sel.push(key.clone());
        }
    }
    sel
}

/// `:or` defaults keyed by lookup key (upstream `push1`, #158): a
/// symbol `:or` key matches a local only; any other `:or` key must
/// equal an entry key form as written. Non-matching `:or` keys are
/// dead like upstream's unreferenced `gdefaults` (excluded here, so
/// the `:select`/`:all` merge base holds applied defaults only).
fn or_mapped(pat: &MapPattern, defaults: &Map) -> Map {
    let mut b_to_k: std::collections::HashMap<&str, &Value> = std::collections::HashMap::new();
    for (local, key, _, _) in &pat.entries {
        if let Pattern::Symbol(name) = local {
            b_to_k.insert(name.as_str(), key);
        }
    }
    for key in &pat.keys {
        b_to_k.insert(key.local.as_str(), &key.key);
    }
    // Raw entry forms alongside their lookup keys (directive keys
    // are built, never quoted, so each is its own raw form).
    let mut raw_keys: Vec<(&Value, &Value)> = Vec::new();
    for (_, key, raw, _) in &pat.entries {
        raw_keys.push((raw, key));
    }
    for key in &pat.keys {
        raw_keys.push((&key.key, &key.key));
    }
    let mut dm = new_map();
    for (key, _) in defaults.iter() {
        let mapped = match key {
            Value::Symbol(sym) => b_to_k.get(sym.name()).copied(),
            _ => raw_keys
                .iter()
                .find(|(raw, _)| *raw == key)
                .map(|(_, key)| *key),
        };
        if let (Some(mapped), Some(val)) = (mapped, defaults.get(key)) {
            dm = dm.insert(mapped.clone(), val.clone());
        }
    }
    dm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_millis_digits_match_jvm() {
        // Fixed-nanos digit table (T1192 FX1 fx1-time probe):
        // `nanos/1e6` prints exactly like the JVM macro.
        for (nanos, millis) in [
            (0u128, "0.0"),
            (1, "1.0E-6"),
            (10, "1.0E-5"),
            (100, "1.0E-4"),
            (999, "9.99E-4"),
            (1000, "0.001"),
            (1234, "0.001234"),
            (2341699, "2.341699"),
            (123456789, "123.456789"),
            (999999500000000, "9.999995E8"),
            (9007199254740993, "9.007199254740992E9"),
            (123456789012345678, "1.2345678901234567E11"),
        ] {
            assert_eq!(
                Evaluator::format_elapsed_millis(nanos),
                format!("Elapsed time: {millis} msecs"),
                "{nanos}"
            );
        }
    }

    #[test]
    fn uncaught_recur_is_internal() {
        // Defense in depth: an escaping `recur` (impossible —
        // every loop/fn frame catches its own) surfaces as an
        // internal error.
        assert!(matches!(
            uncaught(Signal::Recur(vec![])),
            SourceError::Internal(msg) if msg.contains("recur escaped its loop or fn")
        ));
    }

    #[test]
    fn try_thrown_none_for_recur() {
        // The `try`-region impossible case: pins the helper's
        // total contract (`try` bodies only ever throw).
        assert!(try_thrown(Signal::Recur(vec![])).is_none());
    }

    #[test]
    fn locking_nil_lockee_is_npe() {
        // Direct analyze+evaluate (bypasses macroexpansion, which
        // lowers `locking` to `monitor-enter`): the analyzer's
        // `Locking` arm with a nil lockee is the JVM's
        // synchronized-block NPE (port-fixed lock name, P6).
        let mut eval = Evaluator::new("test");
        let form = Value::List(List::from_vec(vec![
            Value::Symbol(Symbol::new(None, "locking")),
            Value::Nil,
            Value::Int(1),
        ]));
        let expr = eval.analyze(&form).expect("analyze locking");
        let err = eval.eval(&expr, &Env::new()).expect_err("nil lockee");
        let t = try_thrown(err).expect("locking nil throws");
        assert_eq!(t.class, ThrowClass::NullPointerException);
        assert_eq!(
            t.message.as_deref(),
            Some("Cannot enter synchronized block because \"lock\" is null")
        );
    }

    #[test]
    fn read_forms_seed_cce_wraps() {
        // Coverage for `read_forms`' seed-failure arm (the
        // `load-string` twin is oracle-pinned by
        // `m-load-resolver`; this is the hook-taking twin the
        // AOT subset drives): a non-fn `*reader-resolver*`
        // fails the entry cast, wrapped as an executing error
        // at the seed position.
        let mut eval = crate::ns::boot();
        eval.eval_source("(alter-var-root #'clojure.core/*reader-resolver* (constantly 42))")
            .expect("root flip");
        let err = eval
            .read_forms("(+ 1 2)", &mut clj_reader::NoHooks)
            .expect_err("seed cce");
        // No root restore: with 42 still bound, even the restore
        // form would trip the entry cast; the evaluator is dropped
        // with the test. (`Display` renders `Class | message`,
        // like the integration `uncaught` helper. The `const`
        // keeps the assert on one line — multi-line assert
        // literals attract dead macro regions.)
        const WANT: &str =
            "clojure.lang.Compiler$CompilerException | Syntax error executing at (1:1).";
        assert_eq!(err.to_string(), WANT);
    }

    #[test]
    fn seq_to_map_pv_and_array_trailers() {
        // No suite feeds `cons_trailing` a PV/Array odd
        // trailer, so the arms are covered directly with a
        // hand-built list: a PV pair assocs, a PV triple is
        // the pair error, an array folds its entries.
        let mut eval = Evaluator::new("test");
        let pv = |items: Vec<Value>| {
            Value::PrimitiveVector(PrimitiveVector::new(clj_value::Elem8::Int, items, None))
        };
        let list = Value::List(List::from_vec(vec![
            Value::Int(1),
            Value::Int(2),
            pv(vec![Value::Int(3), Value::Int(4)]),
        ]));
        let trailer = eval.seq_to_map(&list).expect("pv pair trailer");
        assert!(
            matches!(&trailer, Value::Map(m)
                if matches!(m.get(&Value::Int(1)), Some(Value::Int(2)))
                    && matches!(m.get(&Value::Int(3)), Some(Value::Int(4)))),
            "expected a map trailer, got {trailer:?}"
        );
        let bad = Value::List(List::from_vec(vec![
            Value::Int(1),
            Value::Int(2),
            pv(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        ]));
        let err = eval.seq_to_map(&bad).expect_err("pv triple trailer");
        let t = try_thrown(err).expect("throws");
        assert_eq!(t.class, ThrowClass::IllegalArgumentException);
        assert_eq!(
            t.message.as_deref(),
            Some("Vector arg to map conj must be a pair")
        );
        let arr = Value::Array(clj_value::Array::new(
            clj_value::ArrayDesc::flat(clj_value::BaseType::Object),
            clj_value::ArrayStore::Objs(vec![clj_value::map_entry(Value::Int(3), Value::Int(4))]),
        ));
        let list = Value::List(List::from_vec(vec![Value::Int(1), Value::Int(2), arr]));
        let trailer = eval.seq_to_map(&list).expect("array trailer");
        assert!(
            matches!(&trailer, Value::Map(m)
                if matches!(m.get(&Value::Int(1)), Some(Value::Int(2)))
                    && matches!(m.get(&Value::Int(3)), Some(Value::Int(4)))),
            "expected a map trailer, got {trailer:?}"
        );
    }

    #[test]
    fn conj_map_pv_pair_and_nonpair() {
        // `Evaluator::conj_map` is the destructuring-`into1`
        // path (surface `conj` plans in `coll.rs`), so its PV
        // arms are covered directly: pairs assoc, non-pairs
        // are the pair error.
        let mut eval = Evaluator::new("test");
        let pv = Value::PrimitiveVector(PrimitiveVector::new(
            clj_value::Elem8::Int,
            vec![Value::Int(1), Value::Int(2)],
            None,
        ));
        let m = eval.conj_map(new_map(), pv).expect("pv pair");
        assert!(matches!(m.get(&Value::Int(1)), Some(Value::Int(2))));
        let bad = Value::PrimitiveVector(PrimitiveVector::new(
            clj_value::Elem8::Int,
            vec![Value::Int(1)],
            None,
        ));
        let err = eval.conj_map(new_map(), bad).expect_err("pv non-pair");
        let t = try_thrown(err).expect("throws");
        assert_eq!(t.class, ThrowClass::IllegalArgumentException);
        assert_eq!(
            t.message.as_deref(),
            Some("Vector arg to map conj must be a pair")
        );
    }

    #[test]
    fn conj_one_pv_coerces_and_appends() {
        // `Evaluator::conj_one` is the destructuring-`into1`
        // path (surface `conj` plans in `coll.rs`), so its PV
        // arm is covered directly: ints append, strings fail
        // the core-`int` coerce (same text as surface
        // `f-conj-bad`).
        let mut eval = Evaluator::new("test");
        let pv = Value::PrimitiveVector(PrimitiveVector::new(
            clj_value::Elem8::Int,
            vec![Value::Int(1)],
            None,
        ));
        let got = eval.conj_one(pv, Value::Int(2)).expect("pv conj");
        assert!(
            matches!(&got, Value::PrimitiveVector(p)
                if p.elem() == clj_value::Elem8::Int
                    && p.items() == vec![Value::Int(1), Value::Int(2)]),
            "expected int pv [1 2], got {got:?}"
        );
        let pv = Value::PrimitiveVector(PrimitiveVector::new(
            clj_value::Elem8::Int,
            vec![Value::Int(1)],
            None,
        ));
        let err = eval
            .conj_one(pv, Value::Str("a".into()))
            .expect_err("pv conj coerce");
        let t = try_thrown(err).expect("throws");
        assert_eq!(t.class, ThrowClass::ClassCastException);
        assert_eq!(
            t.message.as_deref(),
            Some(
                "class java.lang.String cannot be cast to class java.lang.Character (java.lang.String and java.lang.Character are in module java.base of loader 'bootstrap')"
            )
        );
    }
}
