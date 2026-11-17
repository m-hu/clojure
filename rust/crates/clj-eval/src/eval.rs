// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! The tree-walking [`Evaluator`]: [`Expr`]s become values, throwing
//! [`Signal`]s. Covers invocation (§E7), `try`, `loop`/`recur`,
//! `def`/`set!`/`binding`, and the full destructure engine (§E12).
//!
//! S4 ships no prelude: tests inject [`NativeFn`](crate::ext::NativeFn)
//! doubles; `ex-info` et al. arrive with the S5 core subset.

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, LazyLock};

use clj_reader::{Reader, ReaderOpts, Span};
use clj_value::{
    Delay, Keyword, List, Map, PrimitiveVector, Promise, Symbol, ThrowClass, Throwable, Value, Var,
    Vector, new_map,
};
use smallvec::SmallVec;

use crate::coll::native_nth;
use crate::env::{Bindings, Env, FxHashMap, SharedRegistry};
use crate::error::{CompileError, NativeError, Signal, jvm_class_of, span_of};
use crate::expr::{
    Analyzer, CaseTest, Catch, DirectCall, Expr, MapPattern, NsClause, Pattern, pattern_names,
};
use crate::ext::{Closure, FastArith, NativeFn, NsHandle, as_closure, as_native};

/// A form-by-form live read of one source text (see
/// [`Evaluator::live_forms`]).
pub struct LiveForms {
    reader: clj_reader::Reader,
    opts: ReaderOpts,
    fresh: bool,
    /// The entry failure (the load-path resolver cast), reported by
    /// the first read.
    failed: Option<SourceError>,
}

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
pub(crate) fn uncaught(s: Signal) -> SourceError {
    match s {
        Signal::Throw(t) => SourceError::Uncaught(t),
        Signal::Recur(_) => SourceError::Internal("recur escaped its loop or fn".to_owned()),
    }
}

/// The throwable of a signal leaving a `try` region (`None`
/// for `recur`, which the analyzer fences from crossing `try`
/// — expr.rs `no_recur` — so `try` bodies only ever throw).
/// One VM site's cached answer (see `Evaluator::vm_sites`); the
/// default never matches a real epoch.
#[derive(Clone)]
struct SiteEntry {
    epoch: u64,
    bind: u64,
    value: Value,
}

impl Default for SiteEntry {
    fn default() -> Self {
        Self {
            epoch: u64::MAX,
            bind: u64::MAX,
            value: Value::Nil,
        }
    }
}

pub(crate) fn try_thrown(signal: Signal) -> Option<Throwable> {
    match signal {
        Signal::Throw(t) => Some(t),
        Signal::Recur(_) => None,
    }
}

/// A list headed by a symbol whose name starts with `def` (the
/// `Compiler.eval` test, after expansion, for forms that skip the
/// `eval` wrapper).
fn is_def_headed(form: &Value) -> bool {
    match form {
        Value::List(list) => {
            matches!(list.first(), Some(Value::Symbol(sym)) if sym.name().starts_with("def"))
        }
        _ => false,
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
pub(crate) fn case_int_value(v: &Value) -> Option<i32> {
    match v {
        Value::Int(i) => Some(*i as i32),
        Value::Double(d) => Some(*d as i32),
        _ => None,
    }
}

/// `case*` dispatch-key masking (a zero mask skips the shift).
pub(crate) fn case_shift_mask(key: i32, shift: i32, mask: i32) -> i32 {
    if mask == 0 {
        key
    } else {
        key.wrapping_shr((shift & 31) as u32) & mask
    }
}

/// Hook signature for evaluating a `(ns ...)` form in full boot mode.
pub(crate) type EvalNsHook =
    fn(&mut Evaluator, &str, Option<&str>, Option<&Map>, &[NsClause]) -> Result<Value, Signal>;
/// Hook signature for borrowed VM invocation.
pub(crate) type VmCallHook = fn(&mut Evaluator, &Value, &[Value]) -> Option<Result<Value, Signal>>;
/// Hook signature for moved-args VM invocation.
pub(crate) type VmApplyMovedHook =
    fn(&mut Evaluator, &Value, &mut SmallVec<[Value; 4]>) -> Option<Result<Value, Signal>>;

/// Indirect hooks into `crate::core_lazy`, `crate::vm`, and interpreter
/// closure/ns evaluation installed only by full [`crate::ns::boot`], keeping
/// embedded stdlib sources, the VM, and interpreter eval out of Lite AOT links.
pub(crate) struct LazyHooks {
    pub(crate) force: fn(&mut Evaluator, u64) -> bool,
    pub(crate) force_meta: fn(&mut Evaluator, u64) -> bool,
    pub(crate) interpreted: fn(&mut Evaluator, u64) -> Option<Value>,
    pub(crate) eval_ns: EvalNsHook,
    pub(crate) apply_closure: fn(&mut Evaluator, &Closure, &[Value]) -> Result<Value, Signal>,
    pub(crate) vm_call: VmCallHook,
    pub(crate) vm_apply_moved: VmApplyMovedHook,
    pub(crate) vm_call1: fn(
        &mut Evaluator,
        &crate::vm::VmCode,
        &crate::vm::CompiledArity,
        Value,
    ) -> Result<Value, Signal>,
    pub(crate) vm_call2: fn(
        &mut Evaluator,
        &crate::vm::VmCode,
        &crate::vm::CompiledArity,
        Value,
        Value,
    ) -> Result<Value, Signal>,
    pub(crate) eval_expr: fn(&mut Evaluator, &Expr, &crate::env::Env) -> Result<Value, Signal>,
}

/// The evaluator: shared lineage state (namespace registry plus
/// identity counters), per-thread bindings, load path, spawned
/// future threads, and the ambient native call span. `Send + Sync`
/// (the shared state travels by `Arc`); per-thread evaluators fork
/// over one lineage (see [`fork_for_thread`](Self::fork_for_thread)).
pub struct Evaluator {
    pub registry: SharedRegistry,
    pub(crate) bindings: Bindings,
    pub(crate) source: Arc<str>,
    /// `clojure.core/*ns*`'s var id once [`current_ns_name`] found it
    /// (0 = not yet; var ids start at 1).
    star_ns_id: std::sync::atomic::AtomicU64,
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
    /// Var-root cache (see [`deref_var_cached`](Self::deref_var_cached)):
    /// var id → root snapshot. Valid only while `root_epoch` equals
    /// the lineage [`epoch`](SharedRegistry::epoch) — every root
    /// write bumps the lineage epoch, ending this generation.
    root_cache: FxHashMap<u64, Value>,
    /// The lineage epoch [`root_cache`](Self::root_cache) was filled under.
    root_epoch: u64,
    /// The VM's per-site caches (`vm` numbers its global reads and
    /// direct calls): a value or a validation verdict, valid while
    /// both the lineage epoch and the bindings version match.
    vm_sites: Vec<SiteEntry>,
    /// Direct-call sites last validated under `(epoch, bindings
    /// version)` (only valid verdicts are recorded; a mismatch
    /// revalidates).
    vm_direct: Vec<(u64, u64)>,
    /// Pending closure applications (#205): the JVM's bounded
    /// stack as a catchable `StackOverflowError` instead of a
    /// process abort. STM retries restore the entry depth (see
    /// `sync_begin`), like the bindings stack.
    pub(crate) call_depth: usize,
    /// Pooled `recur` arg buffers (#235): `Expr::Recur` checks
    /// one out (or allocates cold) and the landing site returns
    /// the drained buffer, so steady-state loops allocate
    /// nothing here. A LIFO stack (nested loops check out
    /// several without colliding); pooled buffers are always
    /// empty (drained by `try_rebind`, cleared on checkout).
    /// Capped ([`RECUR_POOL_MAX`]): pathological nesting never
    /// parks megabytes on a long-lived evaluator.
    recur_pool: Vec<Vec<Value>>,
    /// Whether a `loop` compiles its unboxed program (`vm_scalar`) at
    /// its first entry rather than once it runs hot (see
    /// [`set_eager_tier`](Self::set_eager_tier)).
    pub(crate) eager_tier: bool,
    /// Skip synthesizing default var metadata maps in [`def_finish`]
    /// when `BootMode::Lite` verified no metadata readers exist (#271).
    pub(crate) skip_var_meta: bool,
    /// The namespace name [`install_native_inner`](Self::install_native_inner)
    /// last qualified a native with (boot installs whole namespaces in
    /// a row: their symbols share it instead of copying it).
    install_home: Option<Arc<str>>,
    /// Optional `core_lazy`/interpreter dispatch hooks (`None` in Lite AOT boot).
    pub(crate) lazy_hooks: Option<&'static LazyHooks>,
}

/// Pooled `recur` buffers retained per evaluator (see
/// `recur_pool`): real nests run a few deep; past this the
/// landing site drops the spare.
const RECUR_POOL_MAX: usize = 8;

/// Pending closure applications past this point throw the
/// catchable `StackOverflowError` (#205, raised #220): the
/// JVM runs ~10-20k deep, so the backstop sits at 2.5x the
/// acceptance depth — far above legitimate suite depths
/// (hundreds) while bounding the grown segments' transient
/// memory (~440 MiB release / ~1 GiB dev, measured at the trip
/// point, freed on the unwind). The apply path grows its own
/// segments
/// ([`STACK_RED_ZONE`]), so native overflow can no longer
/// precede this guard on any thread stack size.
pub(crate) const MAX_CALL_DEPTH: usize = 50_000;

/// The apply path's stack red zone (#210): `maybe_grow` moves
/// the call onto a fresh heap segment when fewer than this many
/// bytes remain. Measured total per-level cost (native + heap)
/// is ~8 KiB (release) to ~20 KiB (dev), so 1 MiB holds any
/// single level's native frames with 40x+ margin.
const STACK_RED_ZONE: usize = 1024 * 1024;

/// Fresh-segment size behind [`STACK_RED_ZONE`]: 8 MiB matches
/// the main-thread stack, so deep runs switch segments about as
/// often as a native-sized stack would exhaust.
const STACK_SEGMENT: usize = 8 * 1024 * 1024;

/// Guarded calls probe the native stack at every this-many-th
/// level only (#226: the probe reads thread-local state on each
/// call). Between probes at most 15 levels run — ~120 KiB at the
/// measured ~8 KiB per level — so a probe that found
/// [`STACK_RED_ZONE`] free still leaves 8x margin at the next.
const STACK_PROBE_EVERY: usize = 16;

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
        let mut this = Self::new_bare(source);
        this.lazy_hooks = Some(&crate::ns::FULL_LAZY_HOOKS);
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
            star_ns_id: std::sync::atomic::AtomicU64::new(0),
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
            root_cache: FxHashMap::default(),
            vm_sites: Vec::new(),
            vm_direct: Vec::new(),
            root_epoch: 0,
            call_depth: 0,
            recur_pool: Vec::new(),
            eager_tier: false,
            skip_var_meta: false,
            install_home: None,
            lazy_hooks: None,
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
            star_ns_id: std::sync::atomic::AtomicU64::new(
                self.star_ns_id.load(std::sync::atomic::Ordering::Relaxed),
            ),
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
            // A fork starts with an empty root cache at the current
            // lineage epoch (sibling writes end its generation like
            // any other evaluator's).
            root_cache: FxHashMap::default(),
            vm_sites: Vec::new(),
            vm_direct: Vec::new(),
            root_epoch: self.registry.epoch(),
            // A fork starts unwound (its stack is its own).
            call_depth: 0,
            // A fork starts with an empty pool (buffers never
            // cross threads).
            recur_pool: Vec::new(),
            eager_tier: self.eager_tier,
            skip_var_meta: self.skip_var_meta,
            install_home: None,
            lazy_hooks: self.lazy_hooks,
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
        clj_value::unpoison(self.future_threads.lock()).push(handle);
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
        let handles = std::mem::take(&mut *clj_value::unpoison(self.future_threads.lock()));
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
        // The `*ns*` id lookup is static and cached (equality/hash
        // ignore the id/meta, so sharing one symbol is sound; vars
        // never change ids; an absent `*ns*` re-probes).
        let id = self.star_ns_var();
        self.bound_ns_name(id)
            .map_or_else(|| self.registry.current(), |n| n.to_string())
    }

    /// [`current_ns_name`](Self::current_ns_name) as the bound
    /// namespace handle's shared name (no copy on the common path:
    /// the analyzer asks per symbol).
    pub(crate) fn current_ns_arc(&self) -> Arc<str> {
        let id = self.star_ns_var();
        self.bound_ns_name(id)
            .map_or_else(|| Arc::from(self.registry.current()), Arc::clone)
    }

    /// The `*ns*` var id (cached once found).
    fn star_ns_var(&self) -> Option<u64> {
        static STAR_NS: LazyLock<Symbol> =
            LazyLock::new(|| Symbol::new(Some("clojure.core"), "*ns*"));
        let cached = self.star_ns_id.load(std::sync::atomic::Ordering::Relaxed);
        if cached != 0 {
            return Some(cached);
        }
        let found = self.registry.lookup_in_ns("clojure.core", &STAR_NS);
        if let Some(id) = found {
            self.star_ns_id
                .store(id, std::sync::atomic::Ordering::Relaxed);
        }
        found
    }

    /// The thread-bound `*ns*` handle's name, if any.
    fn bound_ns_name(&self, id: Option<u64>) -> Option<&Arc<str>> {
        id.and_then(|id| self.bindings.get(id))
            .and_then(|v| match v {
                Value::Extension(h) => h.as_any().downcast_ref::<NsHandle>(),
                _ => None,
            })
            .map(|ns| &ns.name)
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
        let mut analyzer = Analyzer::new(&source, self, seq);
        // `Compiler.eval` wraps every form but `def…`-headed ones
        // in an `eval{N}` fn, which names the fns inside it.
        analyzer.top_wrapped = !is_def_headed(form);
        analyzer.analyze(form, Span::new(1, 1), true)
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
        self.read_forms_into(text, hook, None)
    }

    /// [`read_forms`](Self::read_forms), optionally recording each
    /// form's char range.
    fn read_forms_into(
        &mut self,
        text: &str,
        hook: &mut dyn clj_reader::ReadHooks,
        ranges: Option<&mut Vec<std::ops::Range<usize>>>,
    ) -> Result<Vec<Value>, SourceError> {
        let (reader, opts) = Self::read_forms_pre(self, text, 1);
        match opts {
            Ok(mut opts) => {
                let mut reader = reader;
                Self::read_forms_loop(self, &mut reader, hook, &mut opts, ranges)
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

    /// [`read_forms`](Self::read_forms) plus each top-level form's
    /// char range in the `\r\n`-normalized text (issue #256: the
    /// AOT build masks native `defn`s out of the embedded source).
    /// A range starts after the previous form (or the skipped
    /// leading whitespace), so trivia before a form belongs to it.
    pub fn read_forms_ranged(
        &mut self,
        text: &str,
        hook: &mut dyn clj_reader::ReadHooks,
    ) -> Result<(Vec<Value>, Vec<std::ops::Range<usize>>), SourceError> {
        let mut ranges = Vec::new();
        let forms = self.read_forms_into(text, hook, Some(&mut ranges))?;
        Ok((forms, ranges))
    }

    /// [`read_forms`](Self::read_forms) with the live evaluator
    /// hooks (what `eval`/`load` use). The loop is inlined (not
    /// shared with `read_forms_loop`): the live hooks borrow the
    /// evaluator, which the per-form consult also needs mutably,
    /// so each attempt scopes its own hooks.
    pub fn read_forms_live(&mut self, text: &str) -> Result<Vec<Value>, SourceError> {
        self.read_forms_live_at(text, 1)
    }

    /// [`read_forms_live`](Self::read_forms_live) of a slice that
    /// starts at `line` of its file.
    fn read_forms_live_at(&mut self, text: &str, line: u32) -> Result<Vec<Value>, SourceError> {
        let mut forms = self.live_forms_at(text, line);
        let mut out = Vec::new();
        while let Some(form) = self.next_live_form(&mut forms)? {
            out.push(form);
        }
        Ok(out)
    }

    /// One live read: the next form with its data readers resolved
    /// (`None` at EOF).
    fn read_next_live(
        &mut self,
        reader: &mut clj_reader::Reader,
        opts: &mut ReaderOpts,
    ) -> Result<Option<Value>, SourceError> {
        Self::snap_data_readers(self, opts)?;
        reader.set_opts(opts);
        let outcome = {
            let mut hooks = crate::reader::LiveReadHooks::new(self);
            reader.read_form(false, &mut hooks)
        };
        // Effects precede every exit (fired tags ran during the
        // read, failures included), so drain first.
        Self::drain_data_readers(self, reader, opts.suppress_read)?;
        match outcome {
            Ok(Some(v)) => Ok(Some(Self::resolve_data_readers(
                self,
                v,
                opts.suppress_read,
            )?)),
            Ok(None) => Ok(None),
            Err(e) => Err(SourceError::Reader(e)),
        }
    }

    /// The shared read prologue: synced options, leading
    /// whitespace consumed, gates seeded (entry failures keep the
    /// post-skip reader, so callers blame the position like
    /// `Compiler.load`). Hands back the installed options so the
    /// per-form loops start from them instead of re-syncing the
    /// namespace maps.
    fn read_forms_pre(
        eval: &mut Evaluator,
        text: &str,
        line: u32,
    ) -> (clj_reader::Reader, Result<ReaderOpts, NativeError>) {
        let mut reader = clj_reader::Reader::new(text, &ReaderOpts::default());
        reader.set_start_line(line);
        reader.consume_whitespaces();
        let opts = crate::reader::seed_load_gates(eval).map(|seed| {
            let opts = Self::read_forms_opts(eval, &seed);
            reader.set_opts(&opts);
            opts
        });
        (reader, opts)
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
        mut ranges: Option<&mut Vec<std::ops::Range<usize>>>,
    ) -> Result<Vec<Value>, SourceError> {
        let mut forms = Vec::new();
        loop {
            Self::snap_data_readers(eval, opts)?;
            reader.set_opts(opts);
            let start = reader.capture_mark();
            match reader.read_form(false, hook) {
                Ok(Some(v)) => {
                    if let Some(ranges) = ranges.as_deref_mut() {
                        ranges.push(start..reader.capture_mark());
                    }
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

    /// Return a drained `recur` argument buffer to `recur_pool` up to [`RECUR_POOL_MAX`].
    fn recycle_recur_buf(&mut self, values: Vec<Value>) {
        if self.recur_pool.len() < RECUR_POOL_MAX {
            self.recur_pool.push(values);
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
        self.eval_source_at(text, 1)
    }

    /// [`eval_source`](Self::eval_source) of a slice that starts at
    /// `line` of its file (embedded forms evaluated one by one keep
    /// their file lines).
    pub(crate) fn eval_source_at(
        &mut self,
        text: &str,
        line: u32,
    ) -> Result<Vec<Value>, SourceError> {
        let mut forms = self.live_forms_at(text, line);
        let mut out = Vec::new();
        while let Some(form) = self.next_live_form(&mut forms)? {
            out.push(self.eval_top_form(&form)?);
        }
        Ok(out)
    }

    /// Start reading `text` form by form with the live hooks, for
    /// callers that evaluate each form before reading the next
    /// (`load`, `-e`): see [`next_live_form`](Self::next_live_form).
    pub fn live_forms(&mut self, text: &str) -> LiveForms {
        self.live_forms_at(text, 1)
    }

    /// An entry failure (the load-path resolver cast) is the
    /// `Syntax error executing` wrap at the post-skip position,
    /// answered by the first read.
    fn live_forms_at(&mut self, text: &str, line: u32) -> LiveForms {
        let (reader, opts) = Self::read_forms_pre(self, text, line);
        let (opts, failed) = match opts {
            Ok(opts) => (opts, None),
            Err(cce) => {
                let outer = self.next_id();
                let cause = self.next_id();
                let wrap =
                    crate::reader::executing_wrap(reader.line(), reader.col(), &cce, outer, cause);
                (ReaderOpts::default(), Some(wrap))
            }
        };
        LiveForms {
            reader,
            opts,
            fresh: true,
            failed,
        }
    }

    /// The next form of a [`LiveForms`] read (`None` at EOF). Like
    /// `Compiler.load`, each read after the first re-syncs the
    /// namespace view, so an `ns`/`in-ns` switch and new aliases,
    /// refers and imports from the forms evaluated so far reach the
    /// next one (`::kw` after `(ns foo)` reads `:foo/kw`).
    pub fn next_live_form(&mut self, forms: &mut LiveForms) -> Result<Option<Value>, SourceError> {
        if let Some(failed) = forms.failed.take() {
            return Err(failed);
        }
        if !std::mem::replace(&mut forms.fresh, false) {
            crate::reader::sync_reader_ns(self, &mut forms.opts);
        }
        self.read_next_live(&mut forms.reader, &mut forms.opts)
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
        let code = crate::vm::compile_top(&expr);
        match crate::vm::run_top(self, &code) {
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

    /// Compile each `loop`'s unboxed program (`vm_scalar`) at the
    /// loop's first entry instead of once it has run hot (#254): the
    /// tier then runs every loop it covers from the start, as tests
    /// driving it with single runs need. Forks inherit the setting.
    pub fn set_eager_tier(&mut self, on: bool) {
        self.eager_tier = on;
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
        self.install_native_inner(
            ns,
            name,
            func,
            FastArith::None,
            crate::ns::owned_entry(name),
        )
    }

    /// Install a table-driven `fn`-pointer native (like
    /// [`install_native`](Self::install_native), but the inline
    /// integer tag reads off the installing pointer — only the
    /// genuine core arith/compare fns tag, never lookalikes).
    pub fn install_native_ptr(&mut self, ns: &str, name: &str, f: crate::ns::NativeFnPtr) -> Value {
        let owned = crate::ns::owned_entry(name);
        self.install_native_ptr_with_owned(ns, name, f, owned)
    }

    /// Install a namespace's table-driven natives in one registry pass
    /// (Lite boot: the namespace resolves once and the registry takes one
    /// mutation for the batch, instead of per native).
    pub(crate) fn install_native_rows(&mut self, ns: &str, rows: Vec<NativeRow>) {
        let home: Arc<str> = Arc::from(ns);
        let mut roots = Vec::with_capacity(rows.len());
        for (name, f, owned) in rows {
            let native = Value::Extension(Arc::new(NativeFn {
                id: self.next_id(),
                display: qualified_display(ns, name),
                func: Arc::new(f),
                owned,
                meta: None,
                fast: crate::core::fast_arith_for(f),
            }));
            let sym = Symbol::from_shared(Some(Arc::clone(&home)), Arc::from(name));
            roots.push((sym, native));
        }
        self.registry.intern_roots(ns, roots);
    }

    /// Install a table-driven `fn`-pointer native with an explicit owned-args
    /// handler (`None` for borrow-only or stripped `Lite` builds).
    pub(crate) fn install_native_ptr_with_owned(
        &mut self,
        ns: &str,
        name: &str,
        f: crate::ns::NativeFnPtr,
        owned: Option<Arc<crate::ext::OwnedNativeImpl>>,
    ) -> Value {
        let func: Arc<crate::ext::NativeImpl> = Arc::new(f);
        self.install_native_inner(ns, name, func, crate::core::fast_arith_for(f), owned)
    }

    /// Install one native root (shared by the `Arc` and
    /// fn-pointer entries).
    fn install_native_inner(
        &mut self,
        ns: &str,
        name: &str,
        func: Arc<crate::ext::NativeImpl>,
        fast: FastArith,
        owned: Option<Arc<crate::ext::OwnedNativeImpl>>,
    ) -> Value {
        let id = self.next_id();
        let home = match self.install_home.take() {
            Some(home) if *home == *ns => home,
            _ => Arc::from(ns),
        };
        let sym = Symbol::from_shared(Some(Arc::clone(&home)), Arc::from(name));
        self.install_home = Some(home);
        let native = Value::Extension(Arc::new(NativeFn {
            id,
            display: qualified_display(ns, name),
            func,
            owned,
            meta: None,
            fast,
        }));
        self.registry.intern_with_root(&sym, native.clone());
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
    pub fn deref_var(&mut self, id: u64, sym: &Symbol) -> Value {
        if let Some(v) = self.bindings.get(id) {
            return v.clone();
        }
        self.force_lazy(id);
        self.registry.root_or_unbound(id, sym)
    }

    /// [`deref_var`](Self::deref_var) for shared-borrow readers of
    /// boot-installed vars (never lazy `core.clj` ones, so nothing
    /// to force).
    pub(crate) fn deref_var_unforced(&self, id: u64, sym: &Symbol) -> Value {
        if let Some(v) = self.bindings.get(id) {
            return v.clone();
        }
        self.registry.root_or_unbound(id, sym)
    }

    /// A VM global read at `site`: the cached value while nothing
    /// changed (no root write anywhere, the same thread bindings),
    /// else [`deref_var_cached`](Self::deref_var_cached), recorded.
    #[inline(always)]
    pub(crate) fn site_global(&mut self, site: usize, var: &Var) -> Value {
        if let Some(e) = self.vm_sites.get(site) {
            if e.epoch == self.registry.epoch() && e.bind == self.bindings.version() {
                return crate::vm::clone_value(&e.value);
            }
        }
        self.site_global_miss(site, var)
    }

    /// [`site_global`](Self::site_global)'s miss: read and record.
    #[cold]
    #[inline(never)]
    fn site_global_miss(&mut self, site: usize, var: &Var) -> Value {
        let value = self.deref_var_cached(var.id, &var.sym);
        self.site_store(site, value.clone());
        value
    }

    /// A VM direct call's validation at `site` (cached like
    /// [`site_global`](Self::site_global)): no thread binding
    /// shadows the var and its root is still the analyzed native.
    /// The cached verdict checks inline; a miss re-validates out of
    /// line.
    #[inline(always)]
    pub(crate) fn site_direct_valid(&mut self, site: usize, call: &DirectCall) -> bool {
        self.site_root_valid(site, &call.var, &call.expect)
    }

    /// [`site_direct_valid`](Self::site_direct_valid) over a var and
    /// the root it must still hold (also an inlined closure's site,
    /// see `vm_scalar`).
    #[inline(always)]
    pub(crate) fn site_root_valid(
        &mut self,
        site: usize,
        var: &Var,
        expect: &Arc<dyn clj_value::ExtensionValue>,
    ) -> bool {
        let now = (self.registry.epoch(), self.bindings.version());
        self.vm_direct.get(site) == Some(&now) || self.site_root_check(site, var, expect)
    }

    /// Whether `depth` more nested calls fit under the call-depth
    /// limit (an inlined call skips [`guarded_call`](Self::guarded_call)'s
    /// check, so a program with inlined calls enters only when every
    /// real call would have passed it).
    pub(crate) fn calls_fit(&self, depth: usize) -> bool {
        self.call_depth + depth <= MAX_CALL_DEPTH
    }

    /// Whether a group of sites (one id for many) still validates:
    /// the cached verdict answers inline, else `check` re-validates
    /// every member and a valid verdict is recorded under the
    /// current epochs (read after the check, which may have forced
    /// a lazy var).
    pub(crate) fn group_valid(
        &mut self,
        gid: usize,
        check: impl FnOnce(&mut Self) -> bool,
    ) -> bool {
        let now = (self.registry.epoch(), self.bindings.version());
        if self.vm_direct.get(gid) == Some(&now) {
            return true;
        }
        let ok = check(self);
        if ok {
            if self.vm_direct.len() <= gid {
                self.vm_direct.resize(gid + 1, (u64::MAX, u64::MAX));
            }
            self.vm_direct[gid] = (self.registry.epoch(), self.bindings.version());
        }
        ok
    }

    /// The epochs a cached direct verdict is keyed by: while a
    /// snapshot is unchanged, every site validated under it
    /// still validates (no root in the lineage was written and
    /// this evaluator's bindings did not change).
    #[inline(always)]
    pub(crate) fn direct_snapshot(&self) -> (u64, u64) {
        (self.registry.epoch(), self.bindings.version())
    }

    /// [`site_root_valid`](Self::site_root_valid)'s miss: check the
    /// var and record a valid verdict under the current epochs.
    #[cold]
    #[inline(never)]
    fn site_root_check(
        &mut self,
        site: usize,
        var: &Var,
        expect: &Arc<dyn clj_value::ExtensionValue>,
    ) -> bool {
        let valid = self.bindings.get(var.id).is_none()
            && matches!(self.deref_var_cached(var.id, &var.sym),
                Value::Extension(e) if Arc::ptr_eq(&e, expect));
        if valid {
            if self.vm_direct.len() <= site {
                self.vm_direct.resize(site + 1, (u64::MAX, u64::MAX));
            }
            // Read after the check, which may have forced a lazy var.
            self.vm_direct[site] = (self.registry.epoch(), self.bindings.version());
        }
        valid
    }

    /// Record a site answer under the current epochs (read after the
    /// computation, which may have forced a lazy var).
    fn site_store(&mut self, site: usize, value: Value) {
        if self.vm_sites.len() <= site {
            self.vm_sites.resize(site + 1, SiteEntry::default());
        }
        self.vm_sites[site] = SiteEntry {
            epoch: self.registry.epoch(),
            bind: self.bindings.version(),
            value,
        };
    }

    /// The visible value of a var through the per-evaluator root
    /// cache (the `Expr::Global` fast path): thread bindings win
    /// (checked first, like [`deref_var`](Self::deref_var));
    /// otherwise a cached root snapshot when the lineage epoch is
    /// unchanged, else the locked read + insert. Every root write
    /// bumps the lineage epoch, so a hit always serves the current
    /// root: single-threaded behavior is identical to
    /// [`deref_var`](Self::deref_var). The insert files under the
    /// pre-read epoch, so even a cross-thread write racing the
    /// locked read cannot poison the cache (the next deref sees a
    /// newer epoch and drops the generation).
    pub(crate) fn deref_var_cached(&mut self, id: u64, sym: &Symbol) -> Value {
        if let Some(v) = self.bindings.get(id) {
            return v.clone();
        }
        let epoch = self.registry.epoch();
        if epoch != self.root_epoch {
            // A root changed somewhere in the lineage: drop the whole
            // generation (keeps the cache bounded across re-def-heavy
            // boots) and fall through to the locked read.
            self.root_cache.clear();
            self.root_epoch = epoch;
        } else if let Some(v) = self.root_cache.get(&id) {
            return v.clone();
        }
        let mut value = self.registry.root_or_unbound(id, sym);
        // Only unbound roots can be lazy `core.clj` vars (forcing
        // bumps the epoch, so the stale-epoch insert below misses
        // once and refills).
        if matches!(value, Value::Unbound(_)) && self.force_lazy(id) {
            value = self.registry.root_or_unbound(id, sym);
        }
        self.root_cache.insert(id, value.clone());
        self.root_epoch = epoch;
        value
    }

    /// Force a lazy `core.clj` var (see `core_lazy`): `true` when
    /// this call evaluated its form. A lock-free miss for every
    /// other var.
    pub(crate) fn force_lazy(&mut self, id: u64) -> bool {
        self.registry.is_lazy(id) && self.lazy_hooks.is_some_and(|h| (h.force)(self, id))
    }

    /// Force a native-backed var's `core.clj` metadata when full boot installed
    /// lazy hooks (`false` in Lite mode).
    pub(crate) fn force_meta(&mut self, id: u64) -> bool {
        self.lazy_hooks.is_some_and(|h| (h.force_meta)(self, id))
    }

    /// Evaluate the upstream interpreted fallback for native var `id`, if full
    /// boot installed lazy hooks.
    pub(crate) fn lazy_interpreted(&mut self, id: u64) -> Option<Value> {
        let hooks = self.lazy_hooks?;
        (hooks.interpreted)(self, id)
    }

    /// Swap the source name (spans and `:file` metadata), returning
    /// the old one.
    pub(crate) fn swap_source(&mut self, source: Arc<str>) -> Arc<str> {
        std::mem::replace(&mut self.source, source)
    }

    /// Swap the binding stack and source name (lazy forcing runs
    /// isolated from the caller's bindings), returning the old pair.
    pub(crate) fn swap_context(
        &mut self,
        bindings: Bindings,
        source: Arc<str>,
    ) -> (Bindings, Arc<str>) {
        (
            std::mem::replace(&mut self.bindings, bindings),
            std::mem::replace(&mut self.source, source),
        )
    }

    /// Evaluate an expression to a value (or a control signal).
    pub fn eval(&mut self, expr: &Expr, env: &Env) -> Result<Value, Signal> {
        match expr {
            Expr::Literal(v) | Expr::Quote(v) => Ok(v.clone()),
            Expr::Global(var) => Ok(self.deref_var_cached(var.id, &var.sym)),
            Expr::Local(idx) => Ok(env.get(*idx)),
            Expr::VarRef(var) => Ok(Value::Var(var.clone())),
            Expr::If { test, then, els } => self.eval_if(test, then, els, env),
            Expr::Do(exprs) => self.eval_body(exprs, env),
            Expr::Let { bindings, body } => self.eval_let_or_loop(expr, bindings, body, env),
            Expr::Loop { bindings, body } => self.eval_let_or_loop(expr, bindings, body, env),
            Expr::Recur(args) => self.eval_recur_expr(args, env),
            Expr::Invoke { head, args, span } => self.eval_invoke_expr(head, args, *span, env),
            Expr::DirectNative(call) => self.eval_direct_native(call, env),
            other => self.eval_non_hot(other, env),
        }
    }

    #[inline]
    fn eval_recur_expr(&mut self, args: &[Expr], env: &Env) -> Result<Value, Signal> {
        // Pooled arg buffer (see `recur_pool`): steady-state loops
        // never allocate here.
        let mut values = self.recur_pool.pop().unwrap_or_default();
        values.clear();
        values.reserve(args.len());
        for arg in args {
            values.push(self.eval(arg, env)?);
        }
        Err(Signal::Recur(values))
    }

    #[inline]
    fn eval_invoke_expr(
        &mut self,
        head: &Expr,
        args: &[Expr],
        span: Span,
        env: &Env,
    ) -> Result<Value, Signal> {
        let callee = self.eval(head, env)?;
        // Stack-inline arg buffer (spills past 4 args, like the old
        // heap `Vec`): calls never escape this buffer, so most
        // arities allocate nothing.
        let mut values: SmallVec<[Value; 4]> = SmallVec::new();
        for arg in args {
            values.push(self.eval(arg, env)?);
        }
        self.invoke_args(&callee, values, span)
    }

    /// Literal collections, fns, defs, and control flow off the fast
    /// scalar/call path (split from [`Evaluator::eval`] for the
    /// complexity budget).
    #[inline]
    fn eval_non_hot(&mut self, expr: &Expr, env: &Env) -> Result<Value, Signal> {
        match expr {
            Expr::VecLit(elems, meta) => self.eval_vec(elems, meta.clone(), env),
            Expr::MapLit(pairs, meta) => self.eval_map_lit(pairs, meta.clone(), env),
            Expr::SetLit(elems, meta) => self.eval_set_lit(elems, meta.clone(), env),
            Expr::Fn(fun) => Ok(Value::Extension(Arc::new(self.make_closure(fun, env)))),
            Expr::LetFn { fns, body } => self.eval_letfn(fns, body, env),
            Expr::Def {
                sym,
                doc,
                init,
                meta,
                span,
            } => self.eval_def(
                sym,
                doc.clone(),
                init.as_deref(),
                meta.as_deref(),
                *span,
                env,
            ),
            Expr::Set { var, val } => self.eval_set(var, val, env),
            Expr::Throw(inner) => self.eval_throw(inner, env),
            Expr::NewThrowable { class, message } => Ok(Value::Throwable(Arc::new(Throwable {
                id: self.next_id(),
                class: *class,
                message: Some(Arc::from(message.as_str())),
                data: None,
                cause: None,
            }))),
            Expr::Try {
                body,
                catches,
                finally_body,
            } => self.eval_try(body, catches, finally_body, env),
            other => self.eval_scope_form(other, env),
        }
    }

    /// Dynamic-binding, namespace, and macro forms (split from
    /// [`Evaluator::eval`] for the complexity budget).
    fn eval_scope_form(&mut self, expr: &Expr, env: &Env) -> Result<Value, Signal> {
        match expr {
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
            } => self.eval_ns_hook(name, doc.as_deref(), attr.as_ref(), clauses),
            Expr::Assert {
                test,
                message,
                form,
            } => self.eval_assert(test, message.as_deref(), form, env),
            Expr::Time { expr } => self.eval_time(expr, env),
            Expr::WithOutStr { out, body } => self.eval_with_out_str(out, body, env),
            other => self.eval_tail_form(other, env),
        }
    }

    fn eval_ns_hook(
        &mut self,
        name: &str,
        doc: Option<&str>,
        attr: Option<&Map>,
        clauses: &[crate::expr::NsClause],
    ) -> Result<Value, Signal> {
        let Some(hooks) = self.lazy_hooks else {
            return Err(self.throw(NativeError::new(
                ThrowClass::IllegalStateException,
                "ns requires full boot".to_owned(),
            )));
        };
        (hooks.eval_ns)(self, name, doc, attr, clauses)
    }

    /// Leaf stream/tx/dispatch forms (split from [`Evaluator::eval`]
    /// for the complexity budget).
    fn eval_tail_form(&mut self, expr: &Expr, env: &Env) -> Result<Value, Signal> {
        match expr {
            Expr::WithInStr { input, s, body } => self.eval_with_in_str(input, s, body, env),
            Expr::LazySeqStar { body } => {
                Ok(crate::lazy::thunk0_value(Arc::clone(body), env.clone()))
            }
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
            _ => unreachable!("eval_tail_form partition"),
        }
    }

    /// `let`/`loop` (one merged arm, so the [`eval`](Self::eval)
    /// dispatch keeps its arm count under the complexity ratchet):
    /// `loop` installs the recur target, `let` binds plainly.
    fn eval_let_or_loop(
        &mut self,
        expr: &Expr,
        bindings: &[(Pattern, Expr)],
        body: &[Expr],
        env: &Env,
    ) -> Result<Value, Signal> {
        if matches!(expr, Expr::Loop { .. }) {
            self.eval_loop(bindings, body, env)
        } else {
            let env = self.bind_all(bindings, env)?;
            self.eval_body(body, &env)
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
        self.registry.lookup_in_live_or_removed(ns, sym)
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
        Ok(Value::Map(self.eval_map_pairs(pairs, env)?.with_meta(meta)))
    }

    /// A map literal's pairs, evaluated left to right.
    fn eval_map_pairs(&mut self, pairs: &[(Expr, Expr)], env: &Env) -> Result<Map, Signal> {
        let mut pairs_out = Vec::with_capacity(pairs.len());
        for (k, v) in pairs {
            let key = self.eval(k, env)?;
            let val = self.eval(v, env)?;
            let key = crate::lazy::realize_key(self, &key)?;
            pairs_out.push((key, val));
        }
        Ok(Map::from_kvs_last_wins(pairs_out))
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
            let elem = self.eval(elem, env)?;
            let elem = crate::lazy::realize_key(self, &elem)?;
            out = out.insert(elem);
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
        let env = env.extend_patchable(fns.len());
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
        Err(self.throw_value(value))
    }

    /// The signal `throw` raises for a value: throwables as-is,
    /// anything else the cast (or `nil`) error.
    pub(crate) fn throw_value(&mut self, value: Value) -> Signal {
        match value {
            Value::Throwable(t) => {
                Signal::Throw(Arc::try_unwrap(t).unwrap_or_else(|a| (*a).clone()))
            }
            // `nil` is not the cast error: the JVM throws a
            // Clojure-constructed NPE with stable text (probe29).
            Value::Nil => self.throw(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot throw exception because \"null\" is null".to_owned(),
            )),
            other => self.throw(NativeError::cast(
                &jvm_class_of(&other),
                "java.lang.Throwable",
            )),
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
            let dynamic = self.registry.is_dynamic(var.id);
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
        let message = match message {
            None => None,
            Some(m) => Some(self.eval(m, env)?),
        };
        Err(self.assert_failed(message, form))
    }

    /// The failed `assert`'s `AssertionError` (the message value,
    /// when given, renders like `str`).
    pub(crate) fn assert_failed(&mut self, message: Option<Value>, form: &str) -> Signal {
        let text = match message {
            None => format!("Assert failed: {form}"),
            Some(mv) => {
                // `native_str` of one arg is `str_of_one`.
                match crate::core::str_of_one(self, &mv) {
                    Ok(rendered) => format!("Assert failed: {rendered}\n{form}"),
                    Err(e) => return self.throw(e),
                }
            }
        };
        self.throw(NativeError::new(ThrowClass::AssertionError, text))
    }

    /// `(time expr)`: wall-clock the expr, `prn` the JVM line to
    /// `*out*` (`"Elapsed time: <millis> msecs"`, quoted by `prn`),
    /// return the value.
    fn eval_time(&mut self, expr: &Expr, env: &Env) -> Result<Value, Signal> {
        self.time_core(&|ev: &mut Self| ev.eval(expr, env))
    }

    /// `time` around a body.
    pub(crate) fn time_core(&mut self, body: crate::stm::TxBody<'_>) -> Result<Value, Signal> {
        let start = std::time::Instant::now();
        let v = body(self)?;
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
        self.with_out_str_core(out, &|ev: &mut Self| ev.eval_body(body, env))
    }

    /// `with-out-str` around a body.
    pub(crate) fn with_out_str_core(
        &mut self,
        out: &Symbol,
        body: crate::stm::TxBody<'_>,
    ) -> Result<Value, Signal> {
        let writer = crate::io::string_writer(self);
        let var = Var {
            id: self.var_id(out),
            sym: out.clone(),
        };
        crate::native_abi::binding_call_with(self, vec![(var, writer.clone())], body)?;
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
        self.with_in_str_core(input, &v, &|ev: &mut Self| ev.eval_body(body, env))
    }

    /// `with-in-str` over an evaluated string around a body.
    pub(crate) fn with_in_str_core(
        &mut self,
        input: &Symbol,
        v: &Value,
        body: crate::stm::TxBody<'_>,
    ) -> Result<Value, Signal> {
        let Value::Str(text) = v else {
            return Err(self.throw(crate::core::cce_err(v, "java.lang.String")));
        };
        let reader = crate::reader::string_reader(self, text);
        let var = Var {
            id: self.var_id(input),
            sym: input.clone(),
        };
        crate::native_abi::binding_call_with(self, vec![(var, reader)], body)
    }

    /// The `case*` dispatch probe for an evaluated target: the value
    /// the hit's post-check compares (realized for hash tests) and
    /// the shift-masked key — `None` when an `:int` test misses
    /// outright (non-numbers).
    pub(crate) fn case_probe(
        &mut self,
        v: Value,
        shift: i32,
        mask: i32,
        test: CaseTest,
    ) -> Result<Option<(Value, i32)>, Signal> {
        if test == CaseTest::Int {
            return Ok(case_int_value(&v).map(|key| (v, case_shift_mask(key, shift, mask))));
        }
        if clj_value::seq_hash_unsupported(&v) {
            return Err(self.unsupported_hash());
        }
        let r = crate::lazy::realize_deep(self, &v)?;
        if clj_value::seq_hash_unsupported(&r) {
            return Err(self.unsupported_hash());
        }
        let key = case_shift_mask(clj_value::hash_code(&r), shift, mask);
        Ok(Some((r, key)))
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
    pub(crate) fn unsupported_hash(&mut self) -> Signal {
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
        let mut vals = Vec::with_capacity(pairs.len());
        for (var, init) in pairs {
            vals.push((var.clone(), self.eval(init, env)?));
        }
        self.with_redefs_core(vals, &|ev: &mut Self| ev.eval_body(body, env))
    }

    /// `with-redefs` over evaluated values around a body (the
    /// interpreter's or a compiled body fn, #259).
    pub(crate) fn with_redefs_core(
        &mut self,
        vals: Vec<(Var, Value)>,
        body: crate::stm::TxBody<'_>,
    ) -> Result<Value, Signal> {
        let mut saves: Vec<(u64, Option<Value>, Value)> = Vec::with_capacity(vals.len());
        for (var, value) in vals {
            let id = var.id;
            match saves.iter_mut().find(|(i, _, _)| *i == id) {
                Some(slot) => slot.2 = value,
                None => {
                    let saved = self.registry.root_cloned(id);
                    saves.push((id, saved, value));
                }
            }
        }
        for (id, _, value) in &saves {
            self.registry.with_state_mut(*id, |st| {
                st.root = Some(value.clone());
            });
        }
        let result = catch_unwind(AssertUnwindSafe(|| body(self)));
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
        self.with_bindings_core(&bindings, &|ev: &mut Self| ev.eval_body(body, env))
    }

    /// `with-bindings*` over an evaluated binding map.
    pub(crate) fn with_bindings_core(
        &mut self,
        bindings: &Value,
        body: crate::stm::TxBody<'_>,
    ) -> Result<Value, Signal> {
        let entries = crate::vars::push_entries(self, bindings).map_err(|e| self.throw(e))?;
        let depth = self.bindings.depth();
        self.bindings.push_frame(entries);
        let result = body(self);
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
        let env = env.extend_patchable(1);
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
        let closure = Value::Extension(Arc::new(self.make_closure(fun, env)));
        self.defmacro_with(sym, doc, attr, arglists, span, closure)
    }

    /// `defmacro` over a ready macro fn (the interpreter's closure or
    /// a natively compiled one, #259).
    pub(crate) fn defmacro_with(
        &mut self,
        sym: &Symbol,
        doc: Option<&str>,
        attr: Option<&Map>,
        arglists: &[Value],
        span: Span,
        closure: Value,
    ) -> Result<Value, Signal> {
        let id = self.registry.intern_def_target(sym);
        let dynamic = Self::dynamic_of(sym);
        let cur = self.current_ns_name();
        let cur_id = self.registry.ns_id(&cur).unwrap_or(0);
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
        self.registry.with_state_mut(id, |state| {
            state.root = Some(closure);
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
            last = self.eval_def(&sym, None, None, None, span, env)?;
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
        // Fast path: a bare symbol binds one slot directly (no
        // placeholder `Vec`, no `pattern_names` `Vec`, no
        // destructure `Vec`, no `set`).
        if matches!(pat, Pattern::Symbol(_)) {
            return Ok(env.extend_one(value));
        }
        let count = pattern_names(pat).len();
        let mut out = Vec::with_capacity(count);
        self.destructure(pat, &value, env, &mut out, ForceSet::default())?;
        debug_assert_eq!(out.len(), count, "destructure fills every slot");
        Ok(env.extend(out))
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
        meta: Option<&[(Expr, Expr)]>,
        span: Span,
        env: &Env,
    ) -> Result<Value, Signal> {
        let id = self.def_begin(sym);
        let value = match init {
            Some(init) => Some(self.eval(init, env)?),
            None => None,
        };
        // The evaluated name metadata (after the init, like
        // `DefExpr.eval`), else the constant map on the name.
        let name_meta = match meta {
            Some(pairs) => Some(self.eval_map_pairs(pairs, env)?),
            None => sym.meta().cloned(),
        };
        Ok(self.def_finish(id, sym, doc, value, name_meta, span))
    }

    /// A `def`'s first step, before the init evaluates: the target
    /// var's id. Redefining a lazy `core.clj` var forces it first, so
    /// a later force can never overwrite this def (a native-backed
    /// macro's pending metadata source is dropped for the same
    /// reason).
    pub(crate) fn def_begin(&mut self, sym: &Symbol) -> u64 {
        let id = self.registry.intern_def_target(sym);
        self.def_begin_id(id)
    }

    /// [`def_begin`](Self::def_begin) past the intern (a compiled plain
    /// def's target is its already-interned var slot): a lazily
    /// installed root forced, pending metadata taken.
    pub(crate) fn def_begin_id(&mut self, id: u64) -> u64 {
        self.force_lazy(id);
        self.registry.take_meta_lazy(id);
        id
    }

    /// A `def`'s last step: the root (when an init ran), the var
    /// metadata, and the dynamic flag.
    pub(crate) fn def_finish(
        &mut self,
        id: u64,
        sym: &Symbol,
        doc: Option<String>,
        value: Option<Value>,
        name_meta: Option<Map>,
        span: Span,
    ) -> Value {
        let dynamic = Self::dynamic_of(sym);
        // Nothing reads var metadata in a `skip_var_meta` (Lite) program:
        // only the name's own keys stay (`refer` consults `:private`),
        // without the auto keys (#271).
        let meta = match name_meta {
            name_meta if self.skip_var_meta => name_meta,
            name_meta => Some(self.def_var_meta(sym, name_meta.as_ref(), doc, span)),
        };
        self.registry.with_state_mut(id, |state| {
            if let Some(v) = value {
                state.root = Some(v);
            }
            if let Some(m) = meta {
                state.meta = Some(m);
            }
            state.dynamic = dynamic;
        });
        Value::Var(Var {
            id,
            sym: sym.clone(),
        })
    }

    /// A `def`'s var metadata: the name's map, then `:doc`, then the
    /// auto keys — one build with the `insert` chain's exact layout
    /// (first occurrence keeps its position, the last value wins).
    fn def_var_meta(
        &self,
        sym: &Symbol,
        name_meta: Option<&clj_value::Map>,
        doc: Option<String>,
        span: Span,
    ) -> clj_value::Map {
        let cur = self.current_ns_name();
        let cur_id = self.registry.ns_id(&cur).unwrap_or(0);
        let mut pairs: Vec<(Value, Value)> = name_meta
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        if let Some(doc) = doc {
            pairs.push((
                Value::Keyword(clj_value::static_keyword!("doc")),
                Value::Str(Arc::from(doc.as_str())),
            ));
        }
        pairs.extend([
            (
                Value::Keyword(clj_value::static_keyword!("line")),
                Value::Int(i64::from(span.line)),
            ),
            (
                Value::Keyword(clj_value::static_keyword!("column")),
                Value::Int(i64::from(span.col)),
            ),
            (
                Value::Keyword(clj_value::static_keyword!("file")),
                Value::Str(Arc::clone(&self.source)),
            ),
            (
                Value::Keyword(clj_value::static_keyword!("name")),
                Value::Symbol(Symbol::new(None, sym.name())),
            ),
            (
                Value::Keyword(clj_value::static_keyword!("ns")),
                Value::Extension(Arc::new(NsHandle {
                    name: Arc::from(cur.as_str()),
                    id: cur_id,
                })),
            ),
        ]);
        match name_meta {
            // Keep a hashy (or sorted) name map's own class.
            Some(m) if m.is_hashy() => pairs
                .into_iter()
                .skip(m.size())
                .fold(m.clone(), |acc, (k, v)| acc.insert(k, v)),
            _ => clj_value::Map::from_kvs_last_wins(pairs),
        }
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
        // The patterns, collected once (each iteration rebinds in
        // place when unobserved, else binds a fresh frame, so
        // closures from this iteration keep observing the old
        // frame).
        let pats: SmallVec<[&Pattern; 8]> = bindings.iter().map(|(pat, _)| pat).collect();
        loop {
            match self.eval_body(body, &env) {
                Ok(v) => return Ok(v),
                Err(Signal::Recur(values)) => {
                    // Rebind in place when unobserved (#221 R2: no
                    // allocation on the hot path); a captured frame
                    // falls back to the fresh-frame copy-on-`recur`.
                    let mut values = values;
                    if env.try_rebind(&outer, &mut values) {
                        self.recycle_recur_buf(values);
                    } else {
                        env = self.bind_values(&pats, values, &outer)?;
                    }
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
                        let env = env.extend(vec![Value::Throwable(Arc::new(thrown.clone()))]);
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
            vm: None,
        }
    }

    /// `(delay* body…)`: trap a zero-arg closure over the current
    /// frame in an unrealized delay (forced by `deref`/`force`).
    fn eval_delay_star(&mut self, display: &str, body: &[Expr], env: &Env) -> Value {
        let fun = crate::expr::FunExpr {
            display: display.to_owned(),
            arities: Arc::from([crate::expr::Arity {
                fixed: Vec::new(),
                rest: None,
                body: body.to_vec(),
            }]),
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
        self.locking_core(&target, &|ev: &mut Self| ev.eval_body(body, env))
    }

    /// `locking` over an evaluated lockee around a body.
    pub(crate) fn locking_core(
        &mut self,
        target: &Value,
        body: crate::stm::TxBody<'_>,
    ) -> Result<Value, Signal> {
        if matches!(target, Value::Nil) {
            return Err(self.throw(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot enter synchronized block because \"lock\" is null".to_owned(),
            )));
        }
        let key = crate::concurrency::monitor_key(target);
        let _guard = crate::concurrency::locking_acquire_key(key);
        body(self)
    }

    /// The §E7 `IFn` table: closures/natives, keywords, symbols,
    /// vectors, maps, sets, vars (deref-and-delegate), unbounds
    /// (the `IllegalStateException`), promises (one arg delivers),
    /// and the `ClassCastException` for everything else.
    /// Invoke any value (natives reuse this for `:rename`
    /// fallbacks and other higher-order calls).
    pub(crate) fn call(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        // A VM closure's fixed arity runs directly (#226: natives
        // call back per item; no apply dispatch, no arg buffer).
        if let Some(hooks) = self.lazy_hooks {
            if let Some(result) = (hooks.vm_call)(self, callee, args) {
                return result;
            }
        }
        self.invoke(callee, args)
    }

    /// Invoke with an owned arg buffer from natives (accumulator
    /// step calls move the accumulator through, so uniquely-owned
    /// accs stay unique — see [`invoke_args`](Self::invoke_args)).
    pub(crate) fn call_owned(
        &mut self,
        callee: &Value,
        args: SmallVec<[Value; 4]>,
    ) -> Result<Value, Signal> {
        let span = self.native_call_span.unwrap_or(Span::new(1, 1));
        self.invoke_args(callee, args, span)
    }

    /// Invoke with an owned arg buffer: head-releasing looping
    /// natives take the buffer (nil-ing walked coll slots so
    /// memoized seq chunks free as they advance, like the JVM's
    /// dead locals); everything else borrows it (the buffer drops
    /// at return, as before).
    pub(crate) fn invoke_args(
        &mut self,
        callee: &Value,
        mut args: SmallVec<[Value; 4]>,
        span: Span,
    ) -> Result<Value, Signal> {
        // VM closures take the buffer by move (accumulators stay
        // uniquely owned through `reduce`-style step calls).
        if let Some(hooks) = self.lazy_hooks {
            if let Some(result) = (hooks.vm_apply_moved)(self, callee, &mut args) {
                return result;
            }
        }
        // The call site stays visible to natives for the
        // duration of the call (nesting saves/restores).
        let saved = self.native_call_span.replace(span);
        let out = match crate::ext::as_native(callee) {
            Some(native) => self.invoke_native(native, args),
            None => self.invoke(callee, &args),
        };
        self.native_call_span = saved;
        out
    }

    /// Invoke an already-downcast native (split from
    /// [`invoke_args`](Self::invoke_args) for the complexity
    /// budget): owned-args loopers take the buffer, borrow-only
    /// natives borrow it — through the borrow, so the single
    /// downcast above is the whole dispatch (no `Arc` bumps, no
    /// second downcast inside [`invoke`](Self::invoke)).
    fn invoke_native(
        &mut self,
        native: &NativeFn,
        args: SmallVec<[Value; 4]>,
    ) -> Result<Value, Signal> {
        match &native.owned {
            Some(owned) => owned(self, args).map_err(|e| self.throw(e)),
            None => (native.func)(self, &args).map_err(|e| self.throw(e)),
        }
    }

    /// A direct-threaded native call (see [`Expr::DirectNative`]):
    /// the validated fast road calls the snapshot entries, any
    /// validation miss takes the ordinary `Invoke` road.
    fn eval_direct_native(&mut self, call: &DirectCall, env: &Env) -> Result<Value, Signal> {
        if self.direct_valid(call) {
            self.invoke_direct(call, env)
        } else {
            self.invoke_direct_slow(call, env)
        }
    }

    /// Direct-call validation: no thread binding shadows the var,
    /// the lineage epoch still matches (no root write since the
    /// cache generation), and the cached root is still the
    /// analyzed native by identity.
    pub(crate) fn direct_valid(&self, call: &DirectCall) -> bool {
        if self.bindings.get(call.var.id).is_some() {
            return false;
        }
        if self.registry.epoch() != self.root_epoch {
            return false;
        }
        matches!(self.root_cache.get(&call.var.id), Some(Value::Extension(e)) if Arc::ptr_eq(e, &call.expect))
    }

    /// The slow road: re-resolve the head and invoke generally
    /// (the `Invoke` arm over a `Global` head, step for step).
    fn invoke_direct_slow(&mut self, call: &DirectCall, env: &Env) -> Result<Value, Signal> {
        let callee = self.deref_var_cached(call.var.id, &call.var.sym);
        let mut values: SmallVec<[Value; 4]> = SmallVec::new();
        for arg in &call.args {
            values.push(self.eval(arg, env)?);
        }
        self.invoke_args(&callee, values, call.span)
    }

    /// The validated road: inline integer ops compute directly,
    /// everything else calls the snapshot entries.
    fn invoke_direct(&mut self, call: &DirectCall, env: &Env) -> Result<Value, Signal> {
        match call.fast {
            FastArith::None => self.call_direct_general(call, env),
            FastArith::Inc | FastArith::Dec => self.call_direct_arith1(call, env),
            _ => self.call_direct_arith2(call, env),
        }
    }

    /// A validated general call: evaluate into the stack-inline
    /// buffer and run the snapshot entries (no downcast, no `Arc`
    /// bump — the `Invoke` road minus the dispatch).
    fn call_direct_general(&mut self, call: &DirectCall, env: &Env) -> Result<Value, Signal> {
        let mut values: SmallVec<[Value; 4]> = SmallVec::new();
        for arg in &call.args {
            values.push(self.eval(arg, env)?);
        }
        self.call_direct_values(call, &mut values)
    }

    /// A validated direct call over evaluated args (the snapshot
    /// entries — owned-args loopers take the buffer).
    pub(crate) fn call_direct_values(
        &mut self,
        call: &DirectCall,
        values: &mut SmallVec<[Value; 4]>,
    ) -> Result<Value, Signal> {
        let saved = self.native_call_span.replace(call.span);
        let out = match &call.owned {
            Some(owned) => owned(self, std::mem::take(values)).map_err(|e| self.throw(e)),
            None => (call.func)(self, values).map_err(|e| self.throw(e)),
        };
        self.native_call_span = saved;
        out
    }

    /// Read an integer operand without cloning a [`Value`] (#226:
    /// local reads and integer literals evaluate directly as `i64`).
    fn eval_as_i64(&self, expr: &Expr, env: &Env) -> Option<i64> {
        match expr {
            Expr::Literal(Value::Int(n)) => Some(*n),
            Expr::Local(idx) => match env.get_ref(*idx) {
                Some(Value::Int(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        }
    }

    /// A validated `inc`/`dec` call: the `Int` step computes
    /// inline, anything else (non-`Int`, overflow, arity) runs
    /// the genuine native.
    fn call_direct_arith1(&mut self, call: &DirectCall, env: &Env) -> Result<Value, Signal> {
        let [arg] = call.args.as_slice() else {
            return self.call_direct_general(call, env);
        };
        if let Some(x) = self.eval_as_i64(arg, env) {
            // Dispatch routes only `Inc`/`Dec` here, so the `else`
            // is `Dec` — kept as `if/else` (not `match`) so no
            // unreachable arm can trip the 100% line-coverage floor.
            let res = if call.fast == FastArith::Inc {
                x.checked_add(1)
            } else {
                x.checked_sub(1)
            };
            if let Some(y) = res {
                return Ok(Value::Int(y));
            }
        }
        let value = self.eval(arg, env)?;
        match (call.fast, &value) {
            (FastArith::Inc, Value::Int(x)) => match x.checked_add(1) {
                Some(y) => Ok(Value::Int(y)),
                None => self.call_direct_fallback(call, &[value]),
            },
            (FastArith::Dec, Value::Int(x)) => match x.checked_sub(1) {
                Some(y) => Ok(Value::Int(y)),
                None => self.call_direct_fallback(call, &[value]),
            },
            _ => self.call_direct_fallback(call, &[value]),
        }
    }

    /// A validated 2-arg arith/compare call: the `Int`/`Int` pair
    /// computes inline, anything else (non-`Int`, overflow,
    /// arity) runs the genuine native.
    fn call_direct_arith2(&mut self, call: &DirectCall, env: &Env) -> Result<Value, Signal> {
        let [lhs, rhs] = call.args.as_slice() else {
            return self.call_direct_general(call, env);
        };
        if let (Some(x), Some(y)) = (self.eval_as_i64(lhs, env), self.eval_as_i64(rhs, env)) {
            if let Some(v) = Self::fast_int2(call.fast, x, y) {
                return Ok(v);
            }
        }
        let a = self.eval(lhs, env)?;
        let b = self.eval(rhs, env)?;
        if let (Value::Int(x), Value::Int(y)) = (&a, &b) {
            if let Some(v) = Self::fast_int2(call.fast, *x, *y) {
                return Ok(v);
            }
        }
        self.call_direct_fallback(call, &[a, b])
    }

    /// A 2-arg inline integer op (`None` = overflow: the caller
    /// falls back to the native for the `long overflow` error, so
    /// `*unchecked-math*` and every error text stay the native's;
    /// the trailing arm only serves direct unit tests — dispatch
    /// never routes those tags here).
    pub(crate) fn fast_int2(op: FastArith, x: i64, y: i64) -> Option<Value> {
        match op {
            FastArith::Add => x.checked_add(y).map(Value::Int),
            FastArith::Sub => x.checked_sub(y).map(Value::Int),
            FastArith::Mul => x.checked_mul(y).map(Value::Int),
            FastArith::Div => {
                if y != 0 && x != i64::MIN && y != i64::MIN && x % y == 0 {
                    Some(Value::Int(x / y))
                } else {
                    None
                }
            }
            FastArith::Lt => Some(Value::Bool(x < y)),
            FastArith::Lte => Some(Value::Bool(x <= y)),
            FastArith::Gt => Some(Value::Bool(x > y)),
            FastArith::Gte => Some(Value::Bool(x >= y)),
            FastArith::None | FastArith::Inc | FastArith::Dec => None,
        }
    }

    /// A validated call over already-evaluated args (fast-path
    /// fallback: overflow or non-`Int` operands run the genuine
    /// native, so errors and tower results stay exact — and every
    /// tagged native installs with `owned: None`, so the borrow
    /// entry is the whole story).
    pub(crate) fn call_direct_fallback(
        &mut self,
        call: &DirectCall,
        args: &[Value],
    ) -> Result<Value, Signal> {
        let saved = self.native_call_span.replace(call.span);
        let out = (call.func)(self, args).map_err(|e| self.throw(e));
        self.native_call_span = saved;
        out
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
            Value::Extension(_) => self.invoke_extension(callee, args),
            Value::Keyword(_) | Value::Symbol(_) => self.invoke_keyed_name(callee, args),
            Value::Vector(v) => self.invoke_vector(v, args),
            Value::PrimitiveVector(p) => self.invoke_pv(p, args),
            Value::MapEntry { key, val, .. } => self.invoke_mapentry(key, val, args),
            Value::Map(_) => self.invoke_map(callee, args),
            Value::Set(s) => self.invoke_set(s, args),
            Value::Var(v) => self.invoke_var(v, args),
            Value::Promise(p) => self.invoke_promise(p, args),
            Value::MultiFn(m) => crate::multi::invoke_multifn(self, m, args),
            other => self.invoke_fallback(other, args),
        }
    }

    /// Cold invoke arms (split from [`Evaluator::invoke`] for the
    /// complexity budget): nil/unbound throws, `Ref` deref, and `IFn` cast.
    fn invoke_fallback(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        match callee {
            Value::Nil => Err(self.throw(NativeError::new(
                ThrowClass::NullPointerException,
                "Cannot invoke nil as a function".to_owned(),
            ))),
            Value::Unbound(u) => Err(self.throw(NativeError::new(
                ThrowClass::IllegalStateException,
                format!("Attempting to call unbound fn: #'{}", u.sym),
            ))),
            Value::Ref(r) => {
                let value = crate::stm::tx_deref(self, r);
                self.invoke(&value, args)
            }
            _ => Err(self.throw(NativeError::cast(&jvm_class_of(callee), "clojure.lang.IFn"))),
        }
    }

    /// Invoke a keyword/symbol by name (split from [`invoke`](Self::invoke)
    /// for the complexity budget). The arity-error name builds lazily —
    /// `Keyword::to_string` never runs on the hit path.
    fn invoke_keyed_name(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        match args {
            [coll] => Ok(lookup(coll, callee).unwrap_or(Value::Nil)),
            [coll, not_found] => Ok(lookup(coll, callee).unwrap_or_else(|| not_found.clone())),
            _ => {
                let name = match callee {
                    Value::Keyword(kw) => kw.to_string(),
                    _ => "clojure.lang.Symbol".to_owned(),
                };
                Err(self.throw(NativeError::arity(&name, args.len())))
            }
        }
    }

    fn invoke_map(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        match args {
            [key] => Ok(lookup(callee, key).unwrap_or(Value::Nil)),
            [key, not_found] => Ok(lookup(callee, key).unwrap_or_else(|| not_found.clone())),
            _ => Err(self.throw(NativeError::arity(&jvm_class_of(callee), args.len()))),
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

    /// Invoke an extension value: natives run, closures apply,
    /// struct accessors read, anything else is the `IFn` cast error.
    /// Natives dispatch first (every core-fn call is one — a closure
    /// callee pays one extra failed downcast instead).
    fn invoke_extension(&mut self, callee: &Value, args: &[Value]) -> Result<Value, Signal> {
        if let Some(native) = as_native(callee) {
            // The caller holds `callee` for the call (no `Arc` bump).
            return (native.func)(self, args).map_err(|e| self.throw(e));
        }
        if let Some(closure) = as_closure(callee) {
            let Some(hooks) = self.lazy_hooks else {
                return Err(self.throw(NativeError::new(
                    ThrowClass::UnsupportedOperationException,
                    "Closure invocation requires full boot".to_owned(),
                )));
            };
            return (hooks.apply_closure)(self, closure, args);
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
        match usize::try_from(*n).ok().and_then(|i| v.get(i)) {
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
        let root = self.deref_var_cached(v.id, &v.sym);
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
    pub(crate) fn apply_closure(
        &mut self,
        closure: &Closure,
        args: &[Value],
    ) -> Result<Value, Signal> {
        // The #205 call-depth guard: infinite recursion throws
        // the catchable `StackOverflowError` (like the JVM)
        // instead of aborting the process. Every unbounded
        // application path (closures, host methods, multimethod
        // targets, forced thunks) funnels through here; `loop` /
        // `recur` never touch the counter (they iterate). The
        // frame runs on a grown segment when the native stack
        // runs low (#210), so the backstop trips by count on
        // every thread stack size — and `maybe_grow` inlines
        // when stack suffices, keeping the hot path
        // allocation-free.
        self.guarded_call(|ev| ev.apply_closure_frame(closure, args))
    }

    /// Run one unbounded application under the #205 call-depth
    /// guard, on a grown stack segment when the native stack runs
    /// low (see [`apply_closure`](Self::apply_closure)).
    /// Enter one native-to-native call under the interpreter's
    /// depth limit and stack probe (`false`: take the guarded path,
    /// which raises `StackOverflowError` or grows the stack).
    pub(crate) fn enter_native_call(&mut self) -> bool {
        if self.call_depth >= MAX_CALL_DEPTH {
            return false;
        }
        self.call_depth += 1;
        if self.call_depth % STACK_PROBE_EVERY == 0
            && stacker::remaining_stack().is_some_and(|r| r < STACK_RED_ZONE)
        {
            self.call_depth -= 1;
            return false;
        }
        true
    }

    /// Leave a call entered by [`enter_native_call`](Self::enter_native_call).
    pub(crate) fn leave_native_call(&mut self) {
        self.call_depth -= 1;
    }

    pub(crate) fn guarded_call(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<Value, Signal>,
    ) -> Result<Value, Signal> {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(self.throw(NativeError::new(
                ThrowClass::StackOverflowError,
                String::new(),
            )));
        }
        self.call_depth += 1;
        let result = if self.call_depth % STACK_PROBE_EVERY == 0 {
            stacker::maybe_grow(STACK_RED_ZONE, STACK_SEGMENT, || f(self))
        } else {
            f(self)
        };
        self.call_depth -= 1;
        result
    }

    /// The conveyed-bindings frame behind [`apply_closure`](Self::apply_closure).
    fn apply_closure_frame(&mut self, closure: &Closure, args: &[Value]) -> Result<Value, Signal> {
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
        if let Some(code) = &closure.vm {
            return crate::vm::apply(self, closure, code, args);
        }
        let mut variadic: Option<&crate::expr::Arity> = None;
        for arity in closure.arities.iter() {
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
    /// Bind precomputed values against patterns (call params, `recur`
    /// rebinds — no inits evaluate here, so no observer can see an
    /// intermediate frame): all-symbol patterns extend once, with a
    /// single frame for every slot; destructuring patterns still bind
    /// one by one through [`bind_pattern`](Self::bind_pattern) so each
    /// `:or` default evaluates against the pre-extension frame the
    /// analyzer saw (#171).
    fn bind_values(
        &mut self,
        pats: &[&Pattern],
        values: Vec<Value>,
        env: &Env,
    ) -> Result<Env, Signal> {
        if pats.iter().all(|p| matches!(p, Pattern::Symbol(_))) {
            return Ok(env.extend(values));
        }
        let mut frame = env.clone();
        for (pat, value) in pats.iter().zip(values) {
            frame = self.bind_pattern(pat, value, &frame)?;
        }
        Ok(frame)
    }

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
        // Bind through `bind_values` initially; each `recur`
        // rebinds in place when the frame is unobserved, else binds
        // fresh — the copy-on-`recur` promise: closures from this
        // iteration keep the old frame.
        let outer = closure.env.clone();
        // The pattern list borrows the arity (no per-call `Vec`;
        // its length is bounded by the parameter count).
        let pats: SmallVec<[&Pattern; 8]> = arity.fixed.iter().chain(arity.rest.iter()).collect();
        let mut frame = self.bind_values(&pats, values, &outer)?;
        loop {
            match self.eval_body(&arity.body, &frame) {
                Ok(v) => return Ok(v),
                Err(Signal::Recur(values)) => {
                    // In-place rebind when unobserved (#221 R2);
                    // captured frames keep the fresh-frame promise.
                    let mut values = values;
                    if frame.try_rebind(&outer, &mut values) {
                        self.recycle_recur_buf(values);
                    } else {
                        frame = self.bind_values(&pats, values, &outer)?;
                    }
                }
                Err(signal) => return Err(signal),
            }
        }
    }

    /// [`destructure`](Self::destructure) for a binding (`:or`
    /// defaults evaluate against `env`).
    pub(crate) fn destructure_in(
        &mut self,
        pat: &Pattern,
        value: &Value,
        env: &Env,
        out: &mut Vec<Value>,
    ) -> Result<(), Signal> {
        self.destructure(pat, value, env, out, ForceSet::default())
            .map(|_| ())
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
    pub(crate) fn destructure_gseq(&mut self, value: &Value) -> Result<Value, Signal> {
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
                let items: Vec<Value> = v.iter().collect();
                self.conj_map_pair(map, &items)
            }
            // `Vec` is an `IPersistentVector`: pairs assoc like vectors.
            Value::PrimitiveVector(p) => self.conj_map_pair(map, &p.items()),
            Value::MapEntry { key, val, .. } => Ok(map.insert(*key, *val)),
            other => self.conj_map_seq(map, &other),
        }
    }

    fn conj_map_pair(&mut self, map: Map, items: &[Value]) -> Result<Map, Signal> {
        match items {
            [k, val] => Ok(map.insert(k.clone(), val.clone())),
            _ => Err(self.throw(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Vector arg to map conj must be a pair".to_owned(),
            ))),
        }
    }

    fn conj_map_seq(&mut self, mut map: Map, other: &Value) -> Result<Map, Signal> {
        let Some(l) = self.seq_opt(other)? else {
            return Ok(map);
        };
        // One pair per entry (Oracle FX2 `conj-seq-entries`); the
        // first non-entry is the `Map$Entry` cast error.
        for head in l.iter() {
            let Value::MapEntry { key, val, .. } = head else {
                return Err(self.throw(cast_entry(head)));
            };
            map = map.insert((**key).clone(), (**val).clone());
        }
        Ok(map)
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
                let items: Vec<Value> = v.iter().collect();
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
        Value::Int(n) => usize::try_from(*n).ok().and_then(|i| v.get(i)),
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
pub(crate) fn rest_list(args: &[Value]) -> Value {
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
pub(crate) fn nth_or_nil(coll: &Value, i: usize, eval: &mut Evaluator) -> Result<Value, Signal> {
    match coll {
        Value::Nil => Ok(Value::Nil),
        Value::Vector(v) => Ok(v.get(i).unwrap_or(Value::Nil)),
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
            Value::Vector(v) => Ok(Self::seq_list_opt(v.iter().collect())),
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

/// One table-driven native to install: name, entry, owned-args entry.
pub(crate) type NativeRow<'a> = (
    &'a str,
    crate::ns::NativeFnPtr,
    Option<Arc<crate::ext::OwnedNativeImpl>>,
);

/// `ns/name` as one shared string, built in a stack buffer (boot
/// qualifies every native it installs: the heap `String` a `format!`
/// would copy from is a second allocation).
fn qualified_display(ns: &str, name: &str) -> Arc<str> {
    let mut buf = [0u8; 128];
    let len = ns.len() + 1 + name.len();
    match buf.get_mut(..len) {
        Some(out) => {
            let (head, tail) = out.split_at_mut(ns.len());
            head.copy_from_slice(ns.as_bytes());
            tail[0] = b'/';
            tail[1..].copy_from_slice(name.as_bytes());
            Arc::from(std::str::from_utf8(out).unwrap_or_default())
        }
        None => Arc::from(format!("{ns}/{name}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_display_joins_short_and_long_names() {
        assert_eq!(
            &*qualified_display("clojure.core", "inc"),
            "clojure.core/inc"
        );
        let long = "x".repeat(200);
        assert_eq!(&*qualified_display("user", &long), format!("user/{long}"));
    }

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

    #[test]
    fn root_cache_tracks_bindings_defs_and_redefs() {
        let mut eval = Evaluator::new("root-cache-test");
        let sym = Symbol::new(Some("user"), "rc-x");
        let id = eval.registry.intern(&sym);
        // Rootless: the `Unbound` sentinel (miss + insert).
        assert!(matches!(eval.deref_var_cached(id, &sym), Value::Unbound(_)));
        // A root write ends the generation: the next deref drops the
        // cached sentinel and serves the root, then hits the cache.
        eval.registry
            .with_state_mut(id, |st| st.root = Some(Value::Int(1)));
        assert_eq!(eval.deref_var_cached(id, &sym), Value::Int(1));
        assert_eq!(eval.deref_var_cached(id, &sym), Value::Int(1));
        // Re-def invalidates again (no stale root survives).
        eval.registry
            .with_state_mut(id, |st| st.root = Some(Value::Int(2)));
        assert_eq!(eval.deref_var_cached(id, &sym), Value::Int(2));
        // Thread bindings always win over the cached root.
        eval.bindings.push_frame(vec![(
            Var {
                id,
                sym: sym.clone(),
            },
            Value::Int(3),
        )]);
        assert_eq!(eval.deref_var_cached(id, &sym), Value::Int(3));
        eval.bindings.pop_frame();
        assert_eq!(eval.deref_var_cached(id, &sym), Value::Int(2));
    }

    #[test]
    fn fast_int2_rows() {
        use FastArith::{Add, Dec, Div, Gt, Gte, Inc, Lt, Lte, Mul, Sub};
        assert_eq!(Evaluator::fast_int2(Add, 1, 2), Some(Value::Int(3)));
        assert_eq!(Evaluator::fast_int2(Sub, 5, 3), Some(Value::Int(2)));
        assert_eq!(Evaluator::fast_int2(Mul, 4, -3), Some(Value::Int(-12)));
        assert_eq!(Evaluator::fast_int2(Div, 6, -2), Some(Value::Int(-3)));
        assert_eq!(Evaluator::fast_int2(Div, 5, 2), None);
        assert_eq!(Evaluator::fast_int2(Div, 5, 0), None);
        assert_eq!(Evaluator::fast_int2(Div, i64::MIN, 2), None);
        assert_eq!(Evaluator::fast_int2(Div, 2, i64::MIN), None);
        assert_eq!(Evaluator::fast_int2(Lt, 1, 2), Some(Value::Bool(true)));
        assert_eq!(Evaluator::fast_int2(Lt, 2, 2), Some(Value::Bool(false)));
        assert_eq!(Evaluator::fast_int2(Lte, 2, 2), Some(Value::Bool(true)));
        assert_eq!(Evaluator::fast_int2(Gt, 3, 2), Some(Value::Bool(true)));
        assert_eq!(Evaluator::fast_int2(Gte, 1, 2), Some(Value::Bool(false)));
        // Overflow reports `None` (the caller falls back to the
        // native for the `long overflow` error).
        assert_eq!(Evaluator::fast_int2(Add, i64::MAX, 1), None);
        assert_eq!(Evaluator::fast_int2(Sub, i64::MIN, 1), None);
        assert_eq!(Evaluator::fast_int2(Mul, i64::MAX, 2), None);
        // Dispatch never routes these tags here (direct pins for
        // the defensive arm).
        assert_eq!(Evaluator::fast_int2(FastArith::None, 1, 2), None);
        assert_eq!(Evaluator::fast_int2(Inc, 1, 2), None);
        assert_eq!(Evaluator::fast_int2(Dec, 1, 2), None);
    }

    #[test]
    fn recycle_recur_buf_drops_past_pool_cap() {
        let mut eval = Evaluator::new_bare("user");
        for _ in 0..=RECUR_POOL_MAX {
            eval.recycle_recur_buf(Vec::new());
        }
        assert_eq!(eval.recur_pool.len(), RECUR_POOL_MAX);
    }

    #[test]
    fn cov_edge_cases_across_eval_and_builtins() {
        let mut eval = crate::ns::boot();
        // async duplicate binding shadow
        assert!(
            eval.eval_source(
                "(require 'clojure.core.async) (clojure.core.async/go (let [x 1 x 2] x))"
            )
            .is_ok()
        );
        // classemu: Character/MIN_VALUE, Double/parseDouble bad, Named/.getNamespace, BigDecimal setScale/subtract
        assert!(eval.eval_source("Character/MIN_VALUE").is_ok());
        assert!(eval.eval_source("(Double/parseDouble \"bad\")").is_err());
        assert!(
            eval.eval_source(
                "[(let [f (fn [^clojure.lang.Named n] (.getNamespace n))] [(f :k) (f :a/k) (f 's) (f 'a/s)])
                  (bean :k) (bean :a/k) (bean 's) (bean 'a/s)]"
            )
            .is_ok()
        );
        assert!(
            eval.eval_source("(.setScale (bigdec 1.5) 0 (Object.))")
                .is_err()
        );
        assert!(
            eval.eval_source(
                "(let [mc (java.math.MathContext. 2)]
                   [(.subtract (bigdec \"2.5\") (bigdec \"1.2\") mc)
                    (.add (bigdec \"2.5\") (bigdec \"1.2\") mc)
                    (.multiply (bigdec \"2.5\") (bigdec \"1.2\") mc)])"
            )
            .is_ok()
        );
        // collmisc: pr-str on matched re-matcher
        assert!(
            eval.eval_source("(let [m (re-matcher #\"(\\d+)\" \"a42b\")] (re-find m) (pr-str m))")
                .is_ok()
        );
        // ctor: MathContext bad string
        assert!(
            eval.eval_source("(java.math.MathContext. \"bad\")")
                .is_err()
        );
        // emurand: SecureRandom.nextBytes & BigInteger(int, SecureRandom)
        assert!(eval.eval_source("(let [sr (java.security.SecureRandom.) b (byte-array 4)] (.nextBytes sr b) [(java.math.BigInteger. 16 sr) b])").is_ok());
        // fmt: half-way rounding tie & longer tail
        assert!(
            eval.eval_source(
                "[(format \"%.1f\" 1.25) (format \"%.1f\" 1.251) (format \"%.1f\" 1.2501)]"
            )
            .is_ok()
        );
        // eval: seq_to_map on empty list + duplicate selection keys across entries/keys/extra_keys
        assert_eq!(
            eval.seq_to_map(&Value::List(List::new_sync())).unwrap(),
            Value::Map(new_map())
        );
        assert!(
            eval.eval_source(
                "[(let [{a :x b :x} {:x 1}] [a b])
                  (let [{:keys [x x]} {:x 1}] x)
                  (let [{x :x :keys [& :x]} {:x 1}] x)]"
            )
            .is_ok()
        );
        let kx = Value::Keyword(Keyword::new(None, "x"));
        let sel = super::selection(&MapPattern {
            entries: vec![
                (
                    Pattern::Symbol("a".to_owned()),
                    kx.clone(),
                    kx.clone(),
                    false,
                ),
                (
                    Pattern::Symbol("b".to_owned()),
                    kx.clone(),
                    kx.clone(),
                    false,
                ),
            ],
            keys: vec![
                crate::expr::KeyBinding {
                    local: "x1".to_owned(),
                    key: kx.clone(),
                    required: false,
                },
                crate::expr::KeyBinding {
                    local: "x2".to_owned(),
                    key: Value::Keyword(Keyword::new(None, "y")),
                    required: false,
                },
            ],
            extra_keys: vec![kx.clone()],
            as_sym: None,
            or: Vec::new(),
            select: None,
            all: None,
            excess: None,
            missing: None,
        });
        assert_eq!(sel.len(), 2);
        // eval: drain_data_readers error via firing discard reader
        eval.eval_source(
            "(alter-var-root #'*data-readers* (constantly {'t/boom (fn [_] (throw (ex-info \"dr\" {})))}))",
        )
        .unwrap();
        assert!(matches!(
            eval.eval_source("#_#t/boom 1").unwrap_err(),
            SourceError::Uncaught(_)
        ));
        eval.eval_source("(alter-var-root #'*data-readers* (constantly {}))")
            .unwrap();
        // eval: snap_data_readers autoload failure with broken temp data_readers.clj
        let tmp = std::env::temp_dir().join(format!("clj_bad_dr_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("data_readers.clj"), "{").unwrap();
        let prev_lp = eval
            .eval_source("*load-path*")
            .unwrap()
            .last()
            .cloned()
            .unwrap_or(Value::Nil);
        eval.data_reader_scan_path = None;
        let set_lp = format!(
            "(alter-var-root #'*load-path* (constantly [{:?}]))",
            tmp.to_string_lossy()
        );
        let _ = eval.eval_source(&set_lp);
        assert!(eval.eval_source("1").is_err());
        let restore_id = eval
            .registry
            .lookup(&Symbol::new(Some("clojure.core"), "*load-path*"))
            .unwrap();
        eval.registry
            .with_state_mut(restore_id, |st| st.root = Some(prev_lp));
        eval.data_reader_scan_path = None;
        let _ = std::fs::remove_dir_all(&tmp);
        // lazy: take_count errors on drop(lazy + IDrop)/partition/partition-all, sequence non-vec/reduced-non-vec, eduction throwing xf
        assert!(eval.eval_source("(doall (drop :bad '(1 2 3)))").is_err());
        assert!(eval.eval_source("(doall (drop :bad [1 2 3]))").is_err());
        assert!(
            eval.eval_source("(doall (partition :bad [1 2 3]))")
                .is_err()
        );
        assert!(
            eval.eval_source("(doall (partition-all :bad [1 2 3]))")
                .is_err()
        );
        assert!(
            eval.eval_source(
                "[(doall (sequence (fn [rf] (fn ([] (rf)) ([acc] (rf acc)) ([_ _] 42))) [1]))
                  (doall (sequence (fn [rf] (fn ([] (rf)) ([acc] (rf acc)) ([_ _] (reduced 42)))) [1]))]"
            )
            .is_ok()
        );
        assert!(
            eval.eval_source("(doall (eduction (fn [_] (throw (ex-info \"xf\" {}))) [1]))")
                .is_err()
        );
        // types: Object & interface symbols in gen-class :implements and proxy bases
        eval.eval_source(
            "(gen-class :name cov.ObjImpl :implements [Object])
             (assert (= :closed (.close (proxy [Object java.io.Closeable Object] [] (close [] :closed)))))"
        )
        .unwrap();
        // vm_scalar: miss bailout restoring scalar loop reg while skipping boxed reg
        eval.set_eager_tier(true);
        let scalar_res = eval
            .eval_source(
                "(loop [i 0 v [1 2]]
                   (if (< i 2)
                     (if (= i 1)
                       (do (list i) (+ i (count v)))
                       (recur (+ i 1) v))
                     (count v)))",
            )
            .unwrap();
        assert_eq!(scalar_res.last().unwrap(), &Value::Int(3));
    }

    #[test]
    #[should_panic(expected = "eval_tail_form partition")]
    fn eval_tail_form_rejects_upstream_expr() {
        let mut eval = crate::ns::boot();
        let _ = eval.eval_tail_form(&Expr::Literal(Value::Nil), &Env::new());
    }
}
