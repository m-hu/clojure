// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Runtime of natively compiled programs (issue #259), the safe half.
//!
//! Generated code holds `Value`s by value in 32-byte slots and calls
//! the `extern "C"` shims in [`crate::abi`] (`rt_n_*`), which convert
//! raw slots and delegate here. Everything here is safe Rust over
//! owned values and the active evaluator ([`crate::runtime::with_active`]).
//!
//! Program tables (vars, constants, fn descriptors) are declared by
//! the generated `main` before the program runs and frozen after
//! constant construction, so the hot path reads them lock-free.
//!
//! Errors: a failing step stores its `Signal` in the thread's pending
//! slot and the shim returns 1; generated code propagates the 1 to
//! its caller, and the boundary back into the interpreter (closure
//! calls, the program entry) takes the pending signal again.

use std::cell::RefCell;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, OnceLock};

use clj_eval::native_abi::{self, Step};
use clj_eval::{Evaluator, NativeError, NativeFn, Signal};
use clj_value::{FastVars, Value, Var};

use crate::runtime::{ActiveScope, with_active};

/// A compiled arity: `(ret, args, argc, captured) -> status`.
pub type Code = extern "C" fn(*mut Value, *const Value, usize, *const Value) -> i32;

/// One arity of a compiled fn.
#[derive(Clone, Copy)]
pub(crate) struct ArityDesc {
    pub(crate) fixed: usize,
    pub(crate) variadic: bool,
    pub(crate) code: Code,
}

/// A compiled fn's static description.
#[derive(Clone)]
pub(crate) struct FnDesc {
    pub(crate) display: Arc<str>,
    pub(crate) arities: Arc<[ArityDesc]>,
}

/// One declared var: its qualified name (interned on first use) and
/// the compiler's Lite native hint for it.
pub(crate) struct VarSlot {
    ns: Arc<str>,
    name: Arc<str>,
    hint: u32,
    var: OnceLock<Var>,
    core: OnceLock<Option<native_abi::CoreEntry>>,
}

/// A Lite program's registered capability tables, kept past boot for
/// the natives that install on first use (`aot_boot::install_lazy`).
#[derive(Default)]
struct LiteSet {
    core: Vec<clj_eval::aot_boot::LiteCoreGroup>,
    owned: Vec<clj_eval::aot_boot::LiteOwnedSlice>,
    libs: Vec<(&'static str, &'static [clj_eval::aot_boot::LiteNativeSlice])>,
}

impl LiteSet {
    fn tables(&self) -> clj_eval::aot_boot::LiteTables<'_> {
        clj_eval::aot_boot::LiteTables {
            core_groups: &self.core,
            owned_groups: &self.owned,
            lib_tables: &self.libs,
        }
    }
}

/// Tables under construction (startup only).
#[derive(Default)]
struct Building {
    vars: Vec<VarSlot>,
    consts: Vec<Value>,
    fns: Vec<Option<FnDesc>>,
    lite_core_caps: Vec<clj_eval::aot_boot::LiteCoreGroup>,
    lite_owned_caps: Vec<clj_eval::aot_boot::LiteOwnedSlice>,
    lite_libs: Vec<(&'static str, &'static [clj_eval::aot_boot::LiteNativeSlice])>,
    lite_seq_helpers: bool,
    /// Set by a Lite run: freezing keeps the tables above as a [`LiteSet`].
    lite: bool,
}

static BUILDING: Mutex<Option<Building>> = Mutex::new(None);

/// The frozen program tables.
pub(crate) struct Tables {
    /// A Lite run's capability tables (its declared natives install on
    /// first use); `None` under the full boot.
    lite: Option<LiteSet>,
    vars: Box<[VarSlot]>,
    consts: Box<[Value]>,
    fns: Box<[FnDesc]>,
    fast_vars: FastVars,
}

/// The current program's frozen tables with their generation
/// (leaked per run: a binary runs one program; the in-process JIT may
/// run several).
static TABLES: Mutex<Option<(u64, &'static Tables)>> = Mutex::new(None);

/// The generation of [`TABLES`] (bumped under its lock on freeze).
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// The throw a failing step left for generated code to propagate.
    static PENDING: RefCell<Option<Signal>> = const { RefCell::new(None) };
    /// This thread's view of [`TABLES`] (no lock on the hot path).
    static CACHED: std::cell::Cell<Option<(u64, &'static Tables)>> =
        const { std::cell::Cell::new(None) };
}

fn building<R>(f: impl FnOnce(&mut Building) -> R) -> R {
    let mut guard = BUILDING.lock().expect("clj-rt native tables lock");
    f(guard.get_or_insert_with(Building::default))
}

fn tables() -> &'static Tables {
    let generation = GENERATION.load(std::sync::atomic::Ordering::Acquire);
    if let Some((_, t)) = CACHED.get().filter(|(g, _)| *g == generation) {
        return t;
    }
    let current = TABLES.lock().expect("clj-rt native tables lock");
    let (g, t) = current.expect("native program tables frozen before use");
    CACHED.set(Some((g, t)));
    t
}

/// A string constant's text (empty for anything else).
pub(crate) fn str_of(v: &Value) -> String {
    match v {
        Value::Str(s) => s.to_string(),
        _ => String::new(),
    }
}

/// A symbol constant (`_` for anything else).
pub(crate) fn sym_of(v: &Value) -> clj_value::Symbol {
    match v {
        Value::Symbol(s) => s.clone(),
        _ => clj_value::Symbol::new(None, "_"),
    }
}

/// The items of a constant vector (empty for anything else).
pub(crate) fn items_of(v: &Value) -> Vec<Value> {
    match v {
        Value::Vector(items) => items.iter().collect(),
        _ => Vec::new(),
    }
}

/// A shim status: 0 on success, else 1 with the signal pending.
pub(crate) fn status(result: Result<(), Signal>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(signal) => fail(signal),
    }
}

/// Store a failing step's signal; returns the shim status (1).
pub(crate) fn fail(signal: Signal) -> i32 {
    PENDING.with(|p| *p.borrow_mut() = Some(signal));
    1
}

/// Take the pending signal (a missing one is an internal error).
pub(crate) fn take_pending() -> Signal {
    PENDING.with(|p| p.borrow_mut().take()).unwrap_or_else(|| {
        with_active(|eval| {
            native_abi::native_throw(
                eval,
                NativeError::new(
                    clj_value::ThrowClass::Error,
                    "internal error: native code failed without a pending throw".to_owned(),
                ),
            )
        })
    })
}

/// A step's outcome as `(status, value)`.
pub(crate) fn settle(step: Step) -> (i32, Value) {
    match step {
        Ok(v) => (0, v),
        Err(s) => (fail(s), Value::Nil),
    }
}

/// Declare the program's vars — `(ns, name, Lite native hint)` in
/// index order — in one step (one table, each namespace name shared).
pub(crate) fn declare_vars<S: AsRef<str>>(decls: impl ExactSizeIterator<Item = (S, S, u32)>) {
    let mut homes: Vec<Arc<str>> = Vec::new();
    let mut vars = Vec::with_capacity(decls.len());
    for (ns, name, hint) in decls {
        let ns = ns.as_ref();
        let home = match homes.iter().find(|h| ***h == *ns) {
            Some(h) => Arc::clone(h),
            None => {
                let h: Arc<str> = Arc::from(ns);
                homes.push(Arc::clone(&h));
                h
            }
        };
        vars.push(VarSlot {
            ns: home,
            name: Arc::from(name.as_ref()),
            hint,
            var: OnceLock::new(),
            core: OnceLock::new(),
        });
    }
    building(|b| b.vars = vars);
}

/// Register one `BootMode::Lite` core native capability group.
pub(crate) fn declare_lite_core_cap(
    group: clj_eval::aot_boot::LiteCoreGroup,
    owned: Option<clj_eval::aot_boot::LiteOwnedSlice>,
    seq_helpers: bool,
) {
    building(|b| {
        b.lite_core_caps.push(group);
        if let Some(o) = owned {
            b.lite_owned_caps.push(o);
        }
        b.lite_seq_helpers |= seq_helpers;
    });
}

/// Register a `BootMode::Lite` library native capability group (e.g.
/// `clojure.string`).
pub(crate) fn declare_lite_lib_cap(
    ns: &'static str,
    caps: &'static [clj_eval::aot_boot::LiteNativeSlice],
    owned: clj_eval::aot_boot::LiteOwnedSlice,
) {
    building(|b| {
        b.lite_libs.push((ns, caps));
        b.lite_owned_caps.push(owned);
    });
}

/// Declare fn descriptor `k`.
pub(crate) fn declare_fn(k: usize, desc: FnDesc) {
    building(|b| {
        if b.fns.len() <= k {
            b.fns.resize(k + 1, None);
        }
        b.fns[k] = Some(desc);
    });
}

/// Store constant `k` (constant construction, before freezing).
pub(crate) fn set_const(k: usize, value: Value) {
    building(|b| {
        if b.consts.len() <= k {
            b.consts.resize(k + 1, Value::Nil);
        }
        b.consts[k] = value;
    });
}

/// Freeze the tables (after constant construction).
pub(crate) fn freeze(reg_epoch: Arc<AtomicU64>) {
    let b = building(std::mem::take);
    let fns: Vec<FnDesc> = b
        .fns
        .into_iter()
        .map(|d| d.expect("every declared fn index is filled"))
        .collect();
    let fast_vars = FastVars::new(reg_epoch, native_abi::bind_counts(), b.vars.len());
    let lite = b.lite.then_some(LiteSet {
        core: b.lite_core_caps,
        owned: b.lite_owned_caps,
        libs: b.lite_libs,
    });
    let tables: &'static Tables = Box::leak(Box::new(Tables {
        lite,
        vars: b.vars.into_boxed_slice(),
        consts: b.consts.into_boxed_slice(),
        fns: fns.into_boxed_slice(),
        fast_vars,
    }));
    let mut current = TABLES.lock().expect("clj-rt native tables lock");
    let generation = GENERATION.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
    *current = Some((generation, tables));
}

/// Lock-free process-wide fast-var cache for the frozen program.
pub(crate) fn fast_vars() -> &'static FastVars {
    &tables().fast_vars
}

/// Constant `k`.
pub(crate) fn constant(k: usize) -> &'static Value {
    &tables().consts[k]
}

/// The frozen constant table's first slot: the table never moves or
/// changes after [`freeze`], so `base + k` is constant `k` for the rest
/// of the run.
pub(crate) fn consts_base() -> *const Value {
    tables().consts.as_ptr()
}

/// Var `k`, interned on first use (in the then-current registry).
pub(crate) fn var(k: usize) -> &'static Var {
    let t = tables();
    let slot = &t.vars[k];
    slot.var.get_or_init(|| {
        with_active(|eval| {
            if let Some(lite) = &t.lite {
                let decl = (&*slot.ns, &*slot.name, slot.hint);
                clj_eval::aot_boot::install_lazy(eval, decl, lite.tables());
            }
            native_abi::intern_var(eval, &slot.ns, &slot.name)
        })
    })
}

/// A var-cache validity key: (registry epoch, binding version).
type CacheKey = (u64, u64);

thread_local! {
    /// Per-var value cache `(key, value)`: valid while the registry
    /// epoch and the binding version are unchanged (equal versions
    /// mean equal bindings) — the interpreter's own direct-call
    /// validity rule.
    static ROOTS: RefCell<Vec<Option<(CacheKey, Value)>>> = const { RefCell::new(Vec::new()) };
}

/// The current value of var `k` (cached root fast path).
pub(crate) fn root(eval: &mut Evaluator, k: usize) -> Value {
    let epoch = native_abi::root_epoch(eval);
    let hit = ROOTS.with(|r| match r.borrow().get(k) {
        Some(Some((e, v))) if *e == epoch => Some(v.clone()),
        _ => None,
    });
    if let Some(v) = hit {
        return v;
    }
    let var_ref = var(k);
    let e1 = native_abi::registry_epoch(eval);
    let v = native_abi::deref(eval, var_ref);
    let e2 = native_abi::registry_epoch(eval);
    if e1 == e2 {
        fast_vars().record(k, e1, var_ref.id, &v);
    }
    ROOTS.with(|r| {
        let mut r = r.borrow_mut();
        if r.len() <= k {
            r.resize(k + 1, None);
        }
        r[k] = Some((epoch, v.clone()));
    });
    v
}

/// A closure over compiled fn `k` with its captured values: a
/// `NativeFn` the interpreter calls like any fn (arity dispatch and
/// errors as for interpreted closures).
/// A closure's captured values as its shared slice: capture-free
/// closures (every top-level `defn`, plain lambdas) share one empty
/// slice instead of allocating their own.
fn captured_slice(captured: Vec<Value>) -> Arc<[Value]> {
    static NO_CAPTURES: std::sync::LazyLock<Arc<[Value]>> =
        std::sync::LazyLock::new(|| Arc::from(Vec::new()));
    if captured.is_empty() {
        Arc::clone(&NO_CAPTURES)
    } else {
        captured.into()
    }
}

pub(crate) fn make_fn(eval: &mut Evaluator, k: usize, captured: Vec<Value>) -> Value {
    let desc = tables().fns[k].clone();
    let display = Arc::clone(&desc.display);
    let captured = captured_slice(captured);
    // The common shape — one fixed arity — calls its code straight away
    // (no arity search); anything else, wrong counts included, takes
    // the general dispatch.
    let single = match &*desc.arities {
        [a] if !a.variadic => Some((a.fixed, a.code)),
        _ => None,
    };
    // Owned calls (`reduce`'s step, …) hand the argument array over, so
    // the code may take the params it consumes (`native_abi::TAKE_ARGS`).
    let owned = single.map(|(fixed, code)| {
        let (captured, desc) = (Arc::clone(&captured), desc.clone());
        Arc::new(
            move |eval: &mut Evaluator, args: clj_eval::ext::OwnedArgs| {
                if args.len() == fixed {
                    call_code_owned(eval, code, &captured, args)
                } else {
                    call_compiled(eval, &desc, &captured, &args)
                }
                .map_err(native_abi::signal_to_native)
            },
        ) as Arc<clj_eval::ext::OwnedNativeImpl>
    });
    let func: Arc<clj_eval::ext::NativeImpl> = match single {
        Some((fixed, code)) => Arc::new(move |eval: &mut Evaluator, args: &[Value]| {
            if args.len() == fixed {
                call_code(eval, code, &captured, args)
            } else {
                call_compiled(eval, &desc, &captured, args)
            }
            .map_err(native_abi::signal_to_native)
        }),
        None => Arc::new(move |eval: &mut Evaluator, args: &[Value]| {
            call_compiled(eval, &desc, &captured, args).map_err(native_abi::signal_to_native)
        }),
    };
    let ext: Arc<dyn clj_value::ExtensionValue> = Arc::new(NativeFn {
        id: eval.next_id(),
        display,
        func,
        owned,
        meta: None,
        fast: Default::default(),
    });
    CREATED.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() <= k {
            c.resize(k + 1, 0);
        }
        c[k] = Arc::as_ptr(&ext).cast::<()>() as usize;
    });
    Value::Extension(ext)
}

thread_local! {
    /// The latest closure made per compiled fn (by data address): a
    /// direct call site for `defn` k may call its code only while the
    /// var's root is exactly that closure.
    static CREATED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Var `k`'s native entry, resolved once (core vars of a program
/// that cannot rebind them).
pub(crate) fn core_entry(eval: &mut Evaluator, k: usize) -> Option<&'static native_abi::CoreEntry> {
    let t = tables();
    let slot = &t.vars[k];
    slot.core
        .get_or_init(|| {
            // A Lite native called directly needs no var: its entry comes
            // from the tables and its install stays lazy (value uses and
            // `#'f` still install it through `var`).
            let decl = (&*slot.ns, &*slot.name, slot.hint);
            t.lite
                .as_ref()
                .and_then(|lite| native_abi::lite_core_entry(decl, lite.tables()))
                .or_else(|| native_abi::core_entry(eval, var(k)))
        })
        .as_ref()
}

/// Call var `k` with moved args: its resolved native entry, or (when
/// the root is not a native) the root through the general call.
pub(crate) fn call_core_var(eval: &mut Evaluator, k: usize, args: clj_eval::SmallArgs) -> Step {
    match core_entry(eval, k) {
        Some(entry) => native_abi::call_core(eval, entry, args),
        None => {
            let callee = root(eval, k);
            native_abi::call_value_owned(eval, &callee, args)
        }
    }
}

/// Whether var `k`'s current root is the latest closure of compiled
/// fn `desc` (no clone: reads the root cache).
pub(crate) fn root_is_fn(eval: &mut Evaluator, k: usize, desc: usize) -> bool {
    let want = CREATED.with(|c| c.borrow().get(desc).copied().unwrap_or(0));
    if want == 0 {
        return false;
    }
    let epoch = native_abi::root_epoch(eval);
    let hit = ROOTS.with(|r| match r.borrow().get(k) {
        Some(Some((e, v))) if *e == epoch => Some(is_closure_at(v, want)),
        _ => None,
    });
    hit.unwrap_or_else(|| is_closure_at(&root(eval, k), want))
}

/// Whether `v` is the closure at address `want`.
fn is_closure_at(v: &Value, want: usize) -> bool {
    matches!(v, Value::Extension(e) if Arc::as_ptr(e).cast::<()>() as usize == want)
}

/// Select the arity for `args` and run its code on this evaluator.
fn call_compiled(eval: &mut Evaluator, desc: &FnDesc, captured: &[Value], args: &[Value]) -> Step {
    let exact = desc
        .arities
        .iter()
        .find(|a| !a.variadic && a.fixed == args.len());
    let arity = match exact {
        Some(a) => *a,
        None => match desc
            .arities
            .iter()
            .find(|a| a.variadic && args.len() >= a.fixed)
        {
            Some(a) => *a,
            None => return Err(native_abi::arity_error(eval, &desc.display, args.len())),
        },
    };
    if !arity.variadic {
        return call_code(eval, arity.code, captured, args);
    }
    let mut packed = args[..arity.fixed].to_vec();
    packed.push(native_abi::rest_args(&args[arity.fixed..]));
    call_code(eval, arity.code, captured, &packed)
}

/// Run compiled `code` on `argv` (already shaped for its arity) under
/// the call-depth guard, with `eval` active for the shims.
fn call_code(eval: &mut Evaluator, code: Code, captured: &[Value], argv: &[Value]) -> Step {
    native_abi::guarded(eval, |eval| {
        let mut ret = Value::Nil;
        let status = {
            let _active = ActiveScope::enter(eval);
            code(&mut ret, argv.as_ptr(), argv.len(), captured.as_ptr())
        };
        if status == 0 {
            Ok(ret)
        } else {
            Err(take_pending())
        }
    })
}

/// [`call_code`] taking the argument array over: the code may move any
/// param it consumes out of its slot (leaving nil) instead of cloning it
/// (`native_abi::TAKE_ARGS`); whatever it leaves drops here.
fn call_code_owned(
    eval: &mut Evaluator,
    code: Code,
    captured: &[Value],
    mut argv: clj_eval::ext::OwnedArgs,
) -> Step {
    native_abi::guarded(eval, |eval| {
        let mut ret = Value::Nil;
        let status = {
            let _active = ActiveScope::enter(eval);
            let n = argv.len() | native_abi::TAKE_ARGS as usize;
            code(
                &mut ret,
                argv.as_mut_ptr().cast_const(),
                n,
                captured.as_ptr(),
            )
        };
        drop(argv);
        if status == 0 {
            Ok(ret)
        } else {
            Err(take_pending())
        }
    })
}

fn reset_thread_caches() {
    ROOTS.with(|r| r.borrow_mut().clear());
    CREATED.with(|c| c.borrow_mut().clear());
}

fn execute_units(
    eval: &mut Evaluator,
    init: Code,
    tops: &[Code],
) -> Result<Value, clj_eval::error::Signal> {
    run_code(eval, init)?;
    freeze(native_abi::registry_epoch_handle(eval));
    let mut last = Value::Nil;
    for &top in tops {
        last = run_code(eval, top)?;
    }
    Ok(last)
}

/// Run a compiled program using full Clojure boot: `boot`, `run`'s binding
/// frame (plus `clojure.string` preload), constants, then top-level thunks.
pub(crate) fn run_main(path: &str, args: &[String], init: Code, tops: Vec<Code>) -> i32 {
    reset_thread_caches();
    let mut eval = clj_eval::boot();
    let file: Arc<str> = Arc::from(path);
    let tops: Arc<[Code]> = tops.into();
    eval.install_native(
        clj_eval::MIXED_INTERN_NS,
        "run-native",
        Arc::new(move |eval: &mut Evaluator, _: &[Value]| {
            let tops = Arc::clone(&tops);
            clj_eval::load_native(eval, &file, move |eval| execute_units(eval, init, &tops))
        }),
    );
    let script =
        clj_eval::script::run_script(args, &format!("({}/run-native)", clj_eval::MIXED_INTERN_NS));
    let code = match eval.eval_source(&script) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("{}", clj_eval::script::run_diagnostic(path, &e));
            1
        }
    };
    eval.join_drainers();
    std::mem::forget(eval);
    let _flushed = std::io::Write::flush(&mut std::io::stdout());
    code
}

/// Run a compiled program using `BootMode::Lite`: bootstraps only the minimal
/// IO/print/CLI dynamics and the subset vars declared in `BUILDING.vars`,
/// inside a direct CLI binding frame (no `eval_source` / `core_lazy` boot).
pub(crate) fn run_main_lite(
    path: &str,
    args: &[String],
    init: Code,
    tops: Vec<Code>,
    need_stdin: bool,
) -> i32 {
    reset_thread_caches();
    let (declared, core_caps, owned_caps, lib_caps, seq_helpers) = building(|b| {
        b.lite = true;
        let declared: Vec<(Arc<str>, Arc<str>, u32)> = b
            .vars
            .iter()
            .map(|v| (Arc::clone(&v.ns), Arc::clone(&v.name), v.hint))
            .collect();
        (
            declared,
            b.lite_core_caps.clone(),
            b.lite_owned_caps.clone(),
            b.lite_libs.clone(),
            b.lite_seq_helpers,
        )
    });
    let refs: Vec<(&str, &str)> = declared
        .iter()
        .map(|(ns, name, _)| (&**ns, &**name))
        .collect();
    let hints: Vec<u32> = declared.iter().map(|&(_, _, hint)| hint).collect();
    let mut eval = clj_eval::aot_boot::boot_lite_with_caps(
        &refs,
        &hints,
        need_stdin,
        clj_eval::aot_boot::LiteTables {
            core_groups: &core_caps,
            owned_groups: &owned_caps,
            lib_tables: &lib_caps,
        },
        seq_helpers,
    );
    let code = clj_eval::aot_boot::run_with_cli_frame(&mut eval, path, args, move |eval| {
        execute_units(eval, init, &tops)
    });
    std::mem::forget(eval);
    code
}

/// Run one zero-argument compiled code (a top-level thunk or the
/// constant initializer) on `eval`.
fn run_code(eval: &mut Evaluator, code: Code) -> Step {
    let mut ret = Value::Nil;
    let status = {
        let _active = ActiveScope::enter(eval);
        code(&mut ret, std::ptr::null(), 0, std::ptr::null())
    };
    if status == 0 {
        Ok(ret)
    } else {
        Err(take_pending())
    }
}

/// Constant encoding of an `ns` form: `[name doc attr [kind vals…]…]`
/// (`doc`/`attr` nil when absent; kind 0 refer-clojure, 1 require,
/// 2 use, 3 import).
pub(crate) fn run_ns(desc: &Value) -> Step {
    use clj_eval::expr::NsClause;
    let Value::Vector(v) = desc else {
        return Ok(Value::Nil);
    };
    let items: Vec<Value> = v.iter().collect();
    let name = match items.first() {
        Some(Value::Str(s)) => s.to_string(),
        _ => String::new(),
    };
    let doc = match items.get(1) {
        Some(Value::Str(s)) => Some(s.to_string()),
        _ => None,
    };
    let attr = match items.get(2) {
        Some(Value::Map(m)) => Some(m.clone()),
        _ => None,
    };
    let mut clauses = Vec::new();
    for clause in items.iter().skip(3) {
        let Value::Vector(c) = clause else { continue };
        let mut parts = c.iter();
        let kind = match parts.next() {
            Some(Value::Int(k)) => k,
            _ => continue,
        };
        let vals: Vec<Value> = parts.collect();
        clauses.push(match kind {
            0 => NsClause::ReferClojure(vals),
            1 => NsClause::Require(vals),
            2 => NsClause::Use(vals),
            _ => NsClause::Import(vals),
        });
    }
    with_active(|eval| native_abi::ns(eval, &name, doc.as_deref(), attr.as_ref(), &clauses))
}

/// Constant encoding of a `case*` dispatch: `[shift mask test keys
/// skip]` (test 0 int, 1 hash-equiv, 2 hash-identity). Returns the
/// selector (see `rt_n_case`) and the realized key.
pub(crate) fn case_select(desc: &Value, v: &Value) -> Result<(i64, Value), Signal> {
    use clj_eval::expr::CaseTest;
    let Value::Vector(d) = desc else {
        return Ok((-1, Value::Nil));
    };
    let items: Vec<Value> = d.iter().collect();
    let int = |i: usize| int_at(&items, i);
    let ints = |i: usize| ints_at(&items, i);
    let test = match int(2) {
        0 => CaseTest::Int,
        1 => CaseTest::HashEquiv,
        _ => CaseTest::HashIdentity,
    };
    let (keys, skip) = (ints(3), ints(4));
    let hit = with_active(|eval| {
        native_abi::case_select(eval, v, int(0) as i32, int(1) as i32, &keys, &skip, test)
    })?;
    Ok(match hit {
        native_abi::CaseHit::Default => (-1, Value::Nil),
        native_abi::CaseHit::Direct(i) => (i as i64, Value::Nil),
        native_abi::CaseHit::Check(i, r) => (i as i64 | (1 << 32), r),
    })
}

/// Constant encoding of a `defmacro`: `[sym doc attr arglists line col]`.
pub(crate) fn run_defmacro(desc: &Value, f: Value) -> Step {
    let Value::Vector(v) = desc else {
        return Ok(Value::Nil);
    };
    let items: Vec<Value> = v.iter().collect();
    let sym = match items.first() {
        Some(Value::Symbol(s)) => s.clone(),
        _ => clj_value::Symbol::new(None, "_"),
    };
    let doc = match items.get(1) {
        Some(Value::Str(s)) => Some(s.to_string()),
        _ => None,
    };
    let attr = match items.get(2) {
        Some(Value::Map(m)) => Some(m.clone()),
        _ => None,
    };
    let arglists: Vec<Value> = match items.get(3) {
        Some(Value::Vector(a)) => a.iter().collect(),
        _ => Vec::new(),
    };
    let num = |i: usize| match items.get(i) {
        Some(Value::Int(n)) => *n as u32,
        _ => 0,
    };
    with_active(|eval| {
        native_abi::defmacro(
            eval,
            &sym,
            doc.as_deref(),
            attr.as_ref(),
            &arglists,
            num(4),
            num(5),
            f,
        )
    })
}

/// `items[i]` as an int (0 when absent or not an int).
fn int_at(items: &[Value], i: usize) -> i64 {
    match items.get(i) {
        Some(Value::Int(n)) => *n,
        _ => 0,
    }
}

/// `items[i]` as a vector of `i32`s (empty when absent).
fn ints_at(items: &[Value], i: usize) -> Vec<i32> {
    match items.get(i) {
        Some(Value::Vector(v)) => v
            .iter()
            .map(|x| int_at(std::slice::from_ref(&x), 0) as i32)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ActiveScope;

    #[test]
    fn constant_readers_fall_back_on_malformed_constants() {
        assert_eq!(str_of(&Value::Str(Arc::from("s"))), "s");
        assert_eq!(str_of(&Value::Nil), "");
        assert_eq!(sym_of(&Value::Int(1)), clj_value::Symbol::new(None, "_"));
        assert!(items_of(&Value::Nil).is_empty());
    }

    fn vector(items: Vec<Value>) -> Value {
        Value::Vector(clj_value::Vector::from_vec(items))
    }

    #[test]
    fn decoders_tolerate_malformed_descriptors() {
        let mut eval = clj_eval::boot();
        let _scope = ActiveScope::enter(&mut eval);
        // Not a vector: nothing to do.
        assert!(matches!(run_ns(&Value::Nil), Ok(Value::Nil)));
        assert!(matches!(
            case_select(&Value::Nil, &Value::Nil),
            Ok((-1, Value::Nil))
        ));
        assert!(matches!(
            run_defmacro(&Value::Nil, Value::Nil),
            Ok(Value::Nil)
        ));
        // Malformed parts decode to their neutral defaults.
        let ns = vector(vec![
            Value::Str(Arc::from("user.decoded")),
            Value::Int(0),
            Value::Int(0),
            Value::Int(9),
            vector(vec![Value::Nil]),
        ]);
        assert!(run_ns(&ns).is_ok());
        let case = vector(vec![
            Value::Int(0),
            Value::Int(0),
            Value::Int(1),
            Value::Nil,
        ]);
        assert!(matches!(case_select(&case, &Value::Int(1)), Ok((-1, _))));
        let mac = vector(vec![
            Value::Nil,
            Value::Nil,
            Value::Nil,
            Value::Nil,
            Value::Nil,
        ]);
        assert!(run_defmacro(&mac, Value::Nil).is_ok());
        assert_eq!(ints_at(&[vector(vec![Value::Nil])], 0), [0]);
        // A nameless ns and a macro with an attribute map.
        let _ = run_ns(&vector(vec![Value::Nil]));
        let attr = Value::Map(clj_value::Map::from_kvs_last_wins(Vec::new()));
        let mac = vector(vec![
            Value::Symbol(clj_value::Symbol::new(None, "decoded-macro")),
            Value::Str(Arc::from("doc")),
            attr,
            vector(vec![]),
            Value::Int(1),
            Value::Int(2),
        ]);
        assert!(run_defmacro(&mac, Value::Nil).is_ok());
    }

    #[test]
    fn a_missing_pending_throw_is_an_internal_error() {
        let mut eval = clj_eval::boot();
        let _scope = ActiveScope::enter(&mut eval);
        assert!(matches!(take_pending(), Signal::Throw(_)));
    }

    #[test]
    fn statuses_mark_failures_pending() {
        let mut eval = clj_eval::boot();
        let _scope = ActiveScope::enter(&mut eval);
        assert_eq!(status(Ok(())), 0);
        let signal = take_pending();
        assert_eq!(status(Err(signal)), 1);
        assert!(matches!(take_pending(), Signal::Throw(_)));
    }

    #[test]
    fn unknown_fns_are_never_the_root() {
        let mut eval = clj_eval::boot();
        assert!(!root_is_fn(&mut eval, 0, usize::MAX));
    }

    // (The only test touching the global program tables.)
    #[test]
    fn core_var_calls_take_the_native_entry_or_the_root() {
        let mut eval = clj_eval::boot();
        eval.eval_source("(def plain (fn [x] (* 2 x)))")
            .expect("def");
        // Index order; the repeated namespace shares its name.
        let none = clj_eval::aot_boot::NO_LITE_HINT;
        declare_vars(
            [
                ("clojure.core", "inc", none),
                ("user", "plain", none),
                ("clojure.core", "dec", none),
            ]
            .into_iter(),
        );
        building(|b| assert!(Arc::ptr_eq(&b.vars[0].ns, &b.vars[2].ns)));
        extern "C" fn stub_code(_: *mut Value, _: *const Value, _: usize, _: *const Value) -> i32 {
            0
        }
        assert_eq!(
            stub_code(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null()),
            0
        );
        let desc = FnDesc {
            display: Arc::from("stub"),
            arities: Arc::from([ArityDesc {
                fixed: 0,
                variadic: false,
                code: stub_code,
            }]),
        };
        declare_fn(1, desc.clone());
        declare_fn(0, desc);
        set_const(1, Value::Int(1));
        set_const(0, Value::Int(0));
        freeze(native_abi::registry_epoch_handle(&eval));
        let _scope = ActiveScope::enter(&mut eval);
        with_active(|eval| {
            let one: clj_eval::SmallArgs = [Value::Int(1)].into_iter().collect();
            assert!(matches!(call_core_var(eval, 0, one), Ok(Value::Int(2))));
            let three: clj_eval::SmallArgs = [Value::Int(3)].into_iter().collect();
            assert!(matches!(call_core_var(eval, 1, three), Ok(Value::Int(6))));
            ROOTS.with(|r| r.borrow_mut().clear());
            let out = root(eval, 0);
            assert!(fast_vars().get(0).is_some());
            let out2 = root(eval, 0);
            assert!(fast_vars().get(0) == Some(&out2));
            assert!(out == out2);
            let depth = eval.push_binding(var(1).clone(), Value::Int(99));
            assert_eq!(root(eval, 1), Value::Int(99));
            assert!(fast_vars().get(1).is_none());
            eval.pop_bindings(depth);
            let _plain = root(eval, 1);
            assert!(fast_vars().get(1).is_some());
        });
    }
}
