// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Safe half of the AOT runtime: the process-global booted
//! evaluator, primitive dispatch through the interpreter's own
//! natives, and the shared error/print rendering.
//!
//! Zero semantic drift by construction: every helper calls THE
//! SAME [`NativeImpl`](clj_eval::NativeImpl) the interpreter
//! calls, resolved from a booted evaluator's `clojure.core`
//! (comparison chains, `quot` zero rules, `nth`/`get`/`conj`
//! edges, arity texts — all shared, never reimplemented).
//!
//! Concurrency: one [`Mutex`] guards the global evaluator (unit
//! tests share the process; built binaries are single-threaded).
//! A single lock with no nesting cannot deadlock, and the panic
//! catch sits INSIDE the lock scope so the guard always drops on
//! the normal path — a panicking native never poisons the lock.
//! (That single-lock argument is the whole model: there is no lock
//! ordering to check, so a loom model would be vacuous.)

#![forbid(unsafe_code)]

use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, LazyLock, Mutex};

use clj_eval::{Evaluator, NativeError, NativeImpl, as_native, boot};
use clj_value::{List, ThrowClass, Value, Vector, print_to_string};

use crate::handles::{self, Slab};

/// `clojure.core` names in [`Prim`](clj_eval::Prim) declaration order: the table
/// index IS the discriminant (`prim as usize`). The order test
/// below pins the pairing; each FFI differential test pins its
/// shim to its native end to end.
pub(crate) const PRIM_NAMES: [&str; 13] = [
    "+", "-", "*", "quot", "<", "<=", ">", ">=", "=", "count", "nth", "get", "conj",
];

/// The global runtime: one booted evaluator plus the resolved
/// primitive implementations, indexed by [`Prim`](clj_eval::Prim) discriminant.
pub(crate) struct Runtime {
    eval: Evaluator,
    prims: [Arc<NativeImpl>; 13],
    handles: Slab<Value>,
    accs: Slab<Vec<u64>>,
    builders: Slab<Vec<u8>>,
    strings: Vec<Option<Arc<str>>>,
}

impl Runtime {
    /// Boot once: a stock evaluator, then each [`PRIM_NAMES`]
    /// entry resolved exactly like the interpreter resolves it
    /// (evaluating the qualified symbol and taking its native).
    fn boot() -> Self {
        let mut eval = boot();
        let prims = PRIM_NAMES.map(|name| Self::resolve(&mut eval, name));
        Self {
            eval,
            prims,
            handles: Slab::new(),
            accs: Slab::new(),
            builders: Slab::new(),
            strings: Vec::new(),
        }
    }

    /// Resolve one `clojure.core` native through the booted
    /// evaluator (the `expect`s name boot bugs — every core prim
    /// resolves, pinned by the smoke test).
    fn resolve(eval: &mut Evaluator, name: &str) -> Arc<NativeImpl> {
        let probe = format!("clojure.core/{name}");
        let value = eval
            .eval_source(&probe)
            .expect("boot resolves every core prim")
            .into_iter()
            .next()
            .expect("a symbol evaluates to one value");
        Arc::clone(&as_native(&value).expect("core prims are natives").func)
    }

    /// Mint a handle for an owned value (inline scalars skip the
    /// table entirely — no lock-adjacent traffic for literals).
    pub(crate) fn give(&mut self, value: Value) -> u64 {
        if let Some(handle) = handles::encode_inline(&value) {
            return handle;
        }
        let (index, generation) = self.handles.insert(value);
        handles::mint_table(handles::TAG_VALUE, index, generation)
    }

    /// Resolve a handle to a value clone (`None` for stale,
    /// wrong-tag, and invalid handles).
    pub(crate) fn resolve_handle(&self, handle: u64) -> Option<Value> {
        if let Some(value) = handles::decode_inline(handle) {
            return Some(value);
        }
        let (tag, index, generation) = handles::split_table(handle)?;
        if tag != handles::TAG_VALUE {
            return None;
        }
        self.handles.get(index, generation).cloned()
    }

    /// Clone a handle (inline: identity; table: bump; anything
    /// else: `0` — fail-stop at the caller's null check).
    pub(crate) fn clone_handle(&mut self, handle: u64) -> u64 {
        if handles::decode_inline(handle).is_some() {
            return handle;
        }
        let Some((tag, index, generation)) = handles::split_table(handle) else {
            return 0;
        };
        if tag != handles::TAG_VALUE || !self.handles.clone_ref(index, generation) {
            return 0;
        }
        handle
    }

    /// Drop a handle (table: dec, free at zero; inline, stale,
    /// and wrong-tag: silent no-op).
    pub(crate) fn drop_handle(&mut self, handle: u64) {
        if handles::decode_inline(handle).is_some() {
            return;
        }
        if let Some((tag, index, generation)) = handles::split_table(handle) {
            if tag == handles::TAG_VALUE {
                self.handles.drop_ref(index, generation);
            }
        }
    }

    /// Take a handle (resolve + drop — the consume for truthy
    /// tests, push operands, keyword parts); `None` for stale,
    /// wrong-tag, and invalid handles.
    pub(crate) fn take_handle(&mut self, handle: u64) -> Option<Value> {
        if let Some(value) = handles::decode_inline(handle) {
            return Some(value);
        }
        let (tag, index, generation) = handles::split_table(handle)?;
        if tag != handles::TAG_VALUE {
            return None;
        }
        let value = self.handles.get(index, generation)?.clone();
        self.handles.drop_ref(index, generation);
        Some(value)
    }

    /// A fresh arg accumulator (linear: pushed, then consumed
    /// exactly once by a prim/print call).
    pub(crate) fn acc_new(&mut self) -> u64 {
        let (index, generation) = self.accs.insert(Vec::new());
        handles::mint_table(handles::TAG_ACC, index, generation)
    }

    /// Take an acc's arg handles, freeing the slot (`None` for
    /// stale, wrong-tag, and invalid accs).
    pub(crate) fn acc_take(&mut self, acc: u64) -> Option<Vec<u64>> {
        let (tag, index, generation) = handles::split_table(acc)?;
        if tag != handles::TAG_ACC {
            return None;
        }
        self.accs.take(index, generation)
    }

    /// Push an owned element onto a live value slot in place
    /// (non-vectors are overwritten — parity with the old
    /// wrong-type arm); `Err` returns the element unless the slot
    /// is live.
    fn vec_push_live(&mut self, vec: u64, elem: Value) -> Result<u64, Value> {
        let Some((tag, index, generation)) = handles::split_table(vec) else {
            return Err(elem);
        };
        if tag != handles::TAG_VALUE {
            return Err(elem);
        }
        let Some(slot) = self.handles.get_mut(index, generation) else {
            return Err(elem);
        };
        let old = std::mem::replace(slot, Value::Nil);
        *slot = match old {
            Value::Vector(items) => Value::Vector(items.push_back(elem)),
            _ => Value::Vector(Vector::from_vec(vec![elem])),
        };
        Ok(vec)
    }

    /// Push an owned element onto the `vec` slot (live slots
    /// update in place — non-vectors are overwritten, parity with
    /// the old wrong-type arm; dead, slotless, or wrong-tag
    /// `vec`s degrade to a fresh single-element vector).
    pub(crate) fn vec_push_value(&mut self, vec: u64, elem: Value) -> u64 {
        match self.vec_push_live(vec, elem) {
            Ok(handle) => handle,
            Err(elem) => self.give(Value::Vector(Vector::from_vec(vec![elem]))),
        }
    }

    /// Take print args (infallible: stale accs read empty, stale
    /// args read nil — print sites carry no null check, so this
    /// must never fail).
    pub(crate) fn print_take_args(&mut self, acc: u64) -> Vec<Value> {
        let Some(arg_handles) = self.acc_take(acc) else {
            return Vec::new();
        };
        let mut owned = Vec::with_capacity(arg_handles.len());
        for handle in &arg_handles {
            owned.push(self.resolve_handle(*handle).unwrap_or(Value::Nil));
        }
        for handle in &arg_handles {
            self.drop_handle(*handle);
        }
        owned
    }

    /// Push a handle onto an acc (ownership transfers — no
    /// refcount traffic; the consumer drops each arg exactly
    /// once). Returns the acc, `0`-propagating on stale accs.
    pub(crate) fn acc_push(&mut self, acc: u64, handle: u64) -> u64 {
        let Some((tag, index, generation)) = handles::split_table(acc) else {
            return 0;
        };
        if tag != handles::TAG_ACC {
            return 0;
        }
        match self.accs.get_mut(index, generation) {
            Some(args) => {
                args.push(handle);
                acc
            }
            None => 0,
        }
    }

    /// A fresh string builder (chunked scalar construction — no
    /// byte slice ever crosses the boundary).
    pub(crate) fn builder_new(&mut self) -> u64 {
        let (index, generation) = self.builders.insert(Vec::new());
        handles::mint_table(handles::TAG_BUILDER, index, generation)
    }

    /// Append the low `nbytes` of `chunk` (little-endian,
    /// clamped to 8); returns the builder, `0`-propagating on
    /// stale builders.
    pub(crate) fn builder_chunk(&mut self, builder: u64, chunk: u64, nbytes: u8) -> u64 {
        let Some((tag, index, generation)) = handles::split_table(builder) else {
            return 0;
        };
        if tag != handles::TAG_BUILDER {
            return 0;
        }
        let Some(bytes) = self.builders.get_mut(index, generation) else {
            return 0;
        };
        bytes.extend_from_slice(&chunk.to_le_bytes()[..usize::from(nbytes.min(8))]);
        builder
    }

    /// Take a builder's text (lossy — parity with the old
    /// byte-slice decode); `None` for stale builders.
    fn builder_take_text(&mut self, builder: u64) -> Option<String> {
        let (tag, index, generation) = handles::split_table(builder)?;
        if tag != handles::TAG_BUILDER {
            return None;
        }
        let bytes = self.builders.take(index, generation)?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Intern built text under a startup str-id (the M1 prologue
    /// path; repeats overwrite); stale builders intern empty.
    /// Returns the id for chaining.
    pub(crate) fn str_register(&mut self, id: u64, builder: u64) -> u64 {
        let text = self.builder_take_text(builder).unwrap_or_default();
        if let Ok(slot) = usize::try_from(id) {
            self.strings_store(slot, text);
        }
        id
    }

    /// Store interned text at a slot (grows the table; the
    /// `usize::MAX` slot is unaddressable — ids never reach it).
    fn strings_store(&mut self, slot: usize, text: String) {
        if slot == usize::MAX {
            return;
        }
        if self.strings.len() <= slot {
            self.strings.resize(slot + 1, None);
        }
        self.strings[slot] = Some(Arc::from(text.as_str()));
    }

    /// Fetch interned text by str-id (`None` for missing ids).
    pub(crate) fn str_get(&self, id: u64) -> Option<Arc<str>> {
        let slot = usize::try_from(id).ok()?;
        self.strings.get(slot)?.clone()
    }

    /// Finish a builder into an owned string value (tests plus
    /// the M1 per-use fallback); stale builders yield `0`.
    pub(crate) fn str_finish(&mut self, builder: u64) -> u64 {
        match self.builder_take_text(builder) {
            Some(text) => self.give(Value::Str(Arc::from(text.as_str()))),
            None => 0,
        }
    }

    /// Live handle occupancy (values + accs + builders; the
    /// string registry is process-lifetime by design and
    /// excluded) — the leak-counter input.
    pub(crate) fn live_count(&self) -> u64 {
        (self.handles.live() + self.accs.live() + self.builders.live()) as u64
    }
}

/// The process-global runtime, booted on first helper call.
pub(crate) static RUNTIME: LazyLock<Mutex<Runtime>> = LazyLock::new(|| Mutex::new(Runtime::boot()));

/// Dispatch primitive `index` (a [`Prim`](clj_eval::Prim)
/// discriminant) on owned `args` against an already-held
/// runtime: the SAME native the interpreter would call, with the
/// SAME [`NativeError`] on failure. The FFI entry resolves its
/// acc under the same hold — one lock, no nesting, no TOCTOU
/// between arg resolution and dispatch.
pub(crate) fn call_prim_with(
    rt: &mut Runtime,
    index: usize,
    args: Vec<Value>,
) -> Result<Value, NativeError> {
    let func = Arc::clone(&rt.prims[index]);
    catch_panic(AssertUnwindSafe(|| func(&mut rt.eval, &args)))
}

/// The `*command-line-args*` value from program args (T1192 FX1
/// D-argv, shared by the zero-param `rt_argv` shim): index 0
/// (the program name) is skipped; no remaining args is nil, else
/// a list of strings; bytes decode lossy like the JVM's platform
/// decode.
pub(crate) fn argv_from<I>(args: I) -> Value
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
{
    let mut items = Vec::new();
    for arg in args.into_iter().skip(1) {
        let text = arg.as_ref().to_string_lossy().into_owned();
        items.push(Value::Str(Arc::from(text.as_str())));
    }
    if items.is_empty() {
        Value::Nil
    } else {
        Value::List(List::from_vec(items))
    }
}

/// Run a native call, converting a panic (an internal bug — the
/// subset natives return [`NativeError`], never panic) into a
/// `java.lang.Error` so native unwinding never crosses the FFI
/// boundary; the shim prints it and returns null like any error.
/// (Two impossible-by-construction paths still could: a
/// `LazyLock`-init panic on boot failure, or a `prims[index]`
/// OOB panic on a bad codegen index. Both fail stop — boot is
/// smoke-pinned, indices are macro literals — so no silent
/// wrong-code hides behind them. R1 finding 9.)
pub(crate) fn catch_panic(
    call: AssertUnwindSafe<impl FnOnce() -> Result<Value, NativeError>>,
) -> Result<Value, NativeError> {
    match catch_unwind(call) {
        Ok(outcome) => outcome,
        // `as_ref` (NOT `&payload`): unsized coercion would wrap
        // the Box itself as the `Any`, hiding the real payload.
        Err(payload) => Err(NativeError::new(
            ThrowClass::Error,
            panic_text(payload.as_ref()),
        )),
    }
}

/// Render a caught panic payload (String, &str, or unknown).
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        return format!("internal error: {text}");
    }
    if let Some(text) = payload.downcast_ref::<&str>() {
        return format!("internal error: {text}");
    }
    "internal error: native panicked".to_owned()
}

/// The helper error line: `path:line:col: Class: message` (spans
/// baked at the call site; the class/message come straight from
/// the shared [`NativeError`]).
pub(crate) fn error_line(path: &str, line: u32, col: u32, err: &NativeError) -> String {
    format!(
        "{path}:{line}:{col}: {}: {}",
        err.class.jvm_name(),
        err.message
    )
}

/// The `print` intrinsic's output: space-joined
/// human-readable, unquoted strings, NO trailing newline.
pub(crate) fn print_join(args: &[Value]) -> String {
    args.iter()
        .map(print_to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::{
        PRIM_NAMES, RUNTIME, Runtime, call_prim_with, catch_panic, error_line, print_join,
    };
    use clj_eval::{NativeError, Prim};
    use clj_value::{ThrowClass, Value, Vector};
    use std::panic::AssertUnwindSafe;

    #[test]
    fn prim_names_follow_prim_order() {
        let pairs = [
            (Prim::Add, "+"),
            (Prim::Sub, "-"),
            (Prim::Mul, "*"),
            (Prim::Quot, "quot"),
            (Prim::Lt, "<"),
            (Prim::Le, "<="),
            (Prim::Gt, ">"),
            (Prim::Ge, ">="),
            (Prim::Eq, "="),
            (Prim::Count, "count"),
            (Prim::Nth, "nth"),
            (Prim::Get, "get"),
            (Prim::Conj, "conj"),
        ];
        for (prim, name) in pairs {
            assert_eq!(
                PRIM_NAMES.iter().position(|n| *n == name),
                Some(prim as usize),
                "{name}"
            );
            assert_eq!(PRIM_NAMES[prim as usize], name);
        }
    }

    #[test]
    fn boot_resolves_every_prim() {
        let rt = RUNTIME.lock().expect("test lock");
        assert_eq!(rt.prims.len(), PRIM_NAMES.len());
    }

    #[test]
    fn call_prim_reaches_the_shared_native() {
        let mut rt = Runtime::boot();
        let out = call_prim_with(
            &mut rt,
            Prim::Add as usize,
            vec![Value::Int(1), Value::Int(2)],
        );
        assert_eq!(format!("{out:?}"), "Ok(Int(3))");
        let out = call_prim_with(&mut rt, Prim::Sub as usize, Vec::new());
        assert!(out.is_err());
    }

    #[test]
    fn error_line_pins() {
        assert_eq!(
            error_line("f.clj", 1, 2, &NativeError::arity("user/f", 0),),
            "f.clj:1:2: clojure.lang.ArityException: Wrong number of args (0) passed to: user/f"
        );
        assert_eq!(
            error_line(
                "g.clj",
                12,
                30,
                &NativeError::new(ThrowClass::IndexOutOfBoundsException, String::new()),
            ),
            "g.clj:12:30: java.lang.IndexOutOfBoundsException: "
        );
        assert_eq!(
            error_line(
                "h.clj",
                7,
                1,
                &NativeError::new(ThrowClass::Error, "internal error: boom".to_owned()),
            ),
            "h.clj:7:1: java.lang.Error: internal error: boom"
        );
    }

    #[test]
    fn catch_panic_passes_success_through() {
        let out = catch_panic(AssertUnwindSafe(|| Ok(Value::Int(7))));
        assert_eq!(format!("{out:?}"), "Ok(Int(7))");
        let out: Result<Value, NativeError> =
            catch_panic(AssertUnwindSafe(|| Err(NativeError::arity("user/f", 2))));
        assert!(out.is_err());
    }

    #[test]
    fn catch_panic_renders_every_payload() {
        let out = catch_panic(AssertUnwindSafe(|| -> Result<Value, NativeError> {
            std::panic::panic_any("boom-12".to_owned());
        }));
        assert!(
            matches!(&out, Err(e) if e.message == "internal error: boom-12" && e.class == ThrowClass::Error)
        );
        let out = catch_panic(AssertUnwindSafe(|| -> Result<Value, NativeError> {
            std::panic::panic_any("static");
        }));
        assert!(matches!(&out, Err(e) if e.message == "internal error: static"));
        // Real `panic!`s land in `internal error` text too (the
        // exact payload type is toolchain-version-dependent, so
        // these pin the prefix; `panic_any` above pins each arm
        // exactly, version-proof).
        let out = catch_panic(AssertUnwindSafe(|| -> Result<Value, NativeError> {
            panic!("boom");
        }));
        assert!(matches!(&out, Err(e) if e.message.starts_with("internal error: ")));
        let out = catch_panic(AssertUnwindSafe(|| -> Result<Value, NativeError> {
            panic!("boom-{}", 12);
        }));
        assert!(matches!(&out, Err(e) if e.message.starts_with("internal error: ")));
        let out = catch_panic(AssertUnwindSafe(|| -> Result<Value, NativeError> {
            std::panic::panic_any(7_i32);
        }));
        assert!(matches!(&out, Err(e) if e.message == "internal error: native panicked"));
    }

    #[test]
    fn print_join_pins() {
        assert_eq!(print_join(&[]), "");
        assert_eq!(print_join(&[Value::Int(1)]), "1");
        assert_eq!(
            print_join(&[Value::Int(1), Value::Str("a b".into()), Value::Bool(true)]),
            "1 a b true"
        );
    }

    /// Inline scalars skip the table (no occupancy); heap shapes
    /// mint value-slot handles (tagged, counted).
    #[test]
    fn give_inlines_scalars_and_tables_the_rest() {
        let mut rt = Runtime::boot();
        for value in [
            Value::Nil,
            Value::Bool(true),
            Value::Int(7),
            Value::Char('x'),
        ] {
            let handle = rt.give(value);
            assert_ne!(handle, 0);
            assert_eq!(crate::handles::split_table(handle), None);
        }
        assert_eq!(rt.live_count(), 0);
        let tabled = rt.give(Value::Str("s".into()));
        let (tag, _, _) = crate::handles::split_table(tabled).expect("value slot");
        assert_eq!(tag, crate::handles::TAG_VALUE);
        assert_eq!(rt.live_count(), 1);
        let doubled = rt.give(Value::Double(1.5));
        assert_ne!(doubled, tabled);
        assert_eq!(rt.live_count(), 2);
    }

    /// Acc push chains the acc; stale/wrong-tag accs propagate 0.
    #[test]
    fn acc_push_chains_and_propagates_zero() {
        let mut rt = Runtime::boot();
        let acc = rt.acc_new();
        assert_ne!(acc, 0);
        let one = rt.give(Value::Int(1));
        assert_eq!(rt.acc_push(acc, one), acc);
        let two = rt.give(Value::Int(2));
        assert_eq!(rt.acc_push(acc, two), acc);
        assert_eq!(rt.acc_push(0, 1), 0);
        let inline = rt.give(Value::Int(1));
        assert_eq!(rt.acc_push(inline, 2), 0);
        let tabled = rt.give(Value::Str("v".into()));
        assert_eq!(rt.acc_push(tabled, 2), 0);
        assert_eq!(rt.live_count(), 2);
    }

    /// Builders chunk bytes (partial tails, clamped widths)
    /// and finish into owned strings (the registry test pins the
    /// chunk→text content; here the shape and the stale arms).
    #[test]
    fn builders_chunk_and_finish() {
        let mut rt = Runtime::boot();
        let builder = rt.builder_new();
        assert_ne!(builder, 0);
        assert_eq!(
            rt.builder_chunk(builder, u64::from_le_bytes(*b"hello\0\0\0"), 5),
            builder
        );
        // Widths clamp to 8 (no over-read, no panic).
        assert_eq!(rt.builder_chunk(builder, 0, 9), builder);
        assert_eq!(rt.live_count(), 1);
        let finished = rt.str_finish(builder);
        let (tag, _, _) = crate::handles::split_table(finished).expect("value slot");
        assert_eq!(tag, crate::handles::TAG_VALUE);
        assert_eq!(rt.live_count(), 1);
        // Stale builders finish to 0 (fail-stop at the null check).
        assert_eq!(rt.str_finish(0), 0);
        assert_eq!(rt.str_finish(builder), 0);
        assert_eq!(rt.builder_chunk(0, 1, 1), 0);
        assert_eq!(rt.builder_chunk(builder, 1, 1), 0);
        let inline = rt.give(Value::Int(1));
        assert_eq!(rt.builder_chunk(inline, 1, 1), 0);
    }

    /// The M1 registry: register under an id, fetch by id;
    /// repeats overwrite, misses are `None`, stale builders
    /// intern empty (graceful, never panics).
    #[test]
    fn registry_registers_and_gets() {
        let mut rt = Runtime::boot();
        assert_eq!(rt.str_get(0), None);
        let first = rt.builder_new();
        rt.builder_chunk(first, u64::from_le_bytes(*b"f.clj\0\0\0"), 5);
        assert_eq!(rt.str_register(0, first), 0);
        assert_eq!(rt.str_get(0).as_deref(), Some("f.clj"));
        // Registration consumes the builder (registry entries are
        // process-lifetime and stay out of the leak counter).
        assert_eq!(rt.live_count(), 0);
        let second = rt.builder_new();
        rt.builder_chunk(second, u64::from_le_bytes(*b"g.clj\0\0\0"), 5);
        assert_eq!(rt.str_register(0, second), 0);
        assert_eq!(rt.str_get(0).as_deref(), Some("g.clj"));
        assert_eq!(rt.str_register(7, 0), 7);
        assert_eq!(rt.str_get(7).as_deref(), Some(""));
        assert_eq!(rt.str_get(6), None);
        // Invalid bytes decode lossy (parity with the old slice
        // decode — codegen only ever passes valid UTF-8).
        let lossy = rt.builder_new();
        rt.builder_chunk(lossy, 0x62ff61, 3);
        assert_eq!(rt.str_register(8, lossy), 8);
        assert_eq!(rt.str_get(8).as_deref(), Some("a�b"));
    }

    /// The `usize::MAX` slot is unaddressable (ids never reach
    /// it — graceful no-store, never a `resize` overflow).
    #[test]
    fn registry_max_slot_is_unaddressable() {
        let mut rt = Runtime::boot();
        rt.strings_store(usize::MAX, "x".to_owned());
        assert_eq!(rt.live_count(), 0);
    }

    /// The leak counter tracks values + accs + builders (the
    /// registry stays out by design).
    #[test]
    fn live_count_tracks_occupancy() {
        let mut rt = Runtime::boot();
        assert_eq!(rt.live_count(), 0);
        rt.give(Value::Str("a".into()));
        rt.acc_new();
        rt.builder_new();
        assert_eq!(rt.live_count(), 3);
        rt.give(Value::Nil);
        assert_eq!(rt.live_count(), 3);
    }

    /// Resolve/clone/drop/take round-trip both inline and table
    /// handles (clone-then-drop balances; take consumes).
    #[test]
    fn value_handles_round_trip() {
        let mut rt = Runtime::boot();
        let inline = rt.give(Value::Int(9));
        assert_eq!(rt.resolve_handle(inline), Some(Value::Int(9)));
        assert_eq!(rt.clone_handle(inline), inline);
        assert_eq!(rt.take_handle(inline), Some(Value::Int(9)));
        rt.drop_handle(inline);
        assert_eq!(rt.live_count(), 0);
        let tabled = rt.give(Value::Str("v".into()));
        assert!(matches!(rt.resolve_handle(tabled), Some(Value::Str(_))));
        assert_eq!(rt.clone_handle(tabled), tabled);
        assert_eq!(rt.live_count(), 1);
        rt.drop_handle(tabled);
        assert_eq!(rt.live_count(), 1);
        assert!(matches!(rt.take_handle(tabled), Some(Value::Str(_))));
        assert_eq!(rt.live_count(), 0);
    }

    /// Stale, wrong-tag, and invalid handles degrade gracefully:
    /// clones yield `0`, drops are silent, resolves take `None`.
    #[test]
    fn value_handles_degrade_gracefully() {
        let mut rt = Runtime::boot();
        let tabled = rt.give(Value::Str("v".into()));
        rt.drop_handle(tabled);
        for stale in [0, tabled, u64::MAX] {
            assert_eq!(rt.resolve_handle(stale), None);
            assert_eq!(rt.clone_handle(stale), 0);
            assert_eq!(rt.take_handle(stale), None);
            rt.drop_handle(stale);
        }
        let acc = rt.acc_new();
        let builder = rt.builder_new();
        for wrong in [acc, builder] {
            assert_eq!(rt.resolve_handle(wrong), None);
            assert_eq!(rt.clone_handle(wrong), 0);
            assert_eq!(rt.take_handle(wrong), None);
            rt.drop_handle(wrong);
        }
        assert_eq!(rt.live_count(), 2);
    }

    /// Acc take consumes linearly (payload out, slot freed, old
    /// handle stale); wrong-tag takes are `None`.
    #[test]
    fn acc_take_consumes() {
        let mut rt = Runtime::boot();
        let acc = rt.acc_new();
        let one = rt.give(Value::Int(1));
        let two = rt.give(Value::Int(2));
        rt.acc_push(acc, one);
        rt.acc_push(acc, two);
        assert_eq!(rt.acc_take(acc), Some(vec![one, two]));
        assert_eq!(rt.live_count(), 0);
        assert_eq!(rt.acc_take(acc), None);
        assert_eq!(rt.acc_take(0), None);
        assert_eq!(rt.acc_take(one), None);
    }

    /// Vector push appends in place (same handle back), while a
    /// non-vector slot is overwritten and stale/wrong-tag vecs
    /// degrade to fresh single-element vectors.
    #[test]
    fn vec_push_appends_or_degrades() {
        use clj_value::print_to_string;
        let mut rt = Runtime::boot();
        let vec = rt.give(Value::Vector(Vector::from_vec(Vec::new())));
        let one = rt.give(Value::Int(1));
        let elem = rt.take_handle(one).expect("owned elem");
        assert_eq!(rt.vec_push_value(vec, elem), vec);
        let two = rt.give(Value::Int(2));
        let elem = rt.take_handle(two).expect("owned elem");
        assert_eq!(rt.vec_push_value(vec, elem), vec);
        let pushed = rt.resolve_handle(vec).expect("live vec");
        assert_eq!(print_to_string(&pushed), "[1 2]");
        let not_vec = rt.give(Value::Str("s".into()));
        let uno = rt.give(Value::Int(1));
        let elem = rt.take_handle(uno).expect("owned elem");
        assert_eq!(rt.vec_push_value(not_vec, elem), not_vec);
        let degraded = rt.resolve_handle(not_vec).expect("live slot");
        assert_eq!(print_to_string(&degraded), "[1]");
        rt.drop_handle(vec);
        rt.drop_handle(not_vec);
        let fresh = rt.give(Value::Int(3));
        let elem = rt.take_handle(fresh).expect("owned elem");
        let degraded = rt.vec_push_value(vec, elem);
        assert_ne!(degraded, vec);
        let rendered = rt.resolve_handle(degraded).expect("fresh vec");
        assert_eq!(print_to_string(&rendered), "[3]");
        let acc = rt.acc_new();
        let last = rt.give(Value::Int(4));
        let elem = rt.take_handle(last).expect("owned elem");
        let via_acc = rt.vec_push_value(acc, elem);
        assert_ne!(via_acc, acc);
        let rendered = rt.resolve_handle(via_acc).expect("fresh vec");
        assert_eq!(print_to_string(&rendered), "[4]");
    }

    /// Print takes args infallibly (stale accs read empty, stale
    /// args read nil) and always consumes the acc.
    #[test]
    fn print_take_args_is_infallible() {
        use clj_value::print_to_string;
        let mut rt = Runtime::boot();
        assert!(rt.print_take_args(0).is_empty());
        let acc = rt.acc_new();
        let one = rt.give(Value::Int(1));
        let tabled = rt.give(Value::Str("s".into()));
        rt.acc_push(acc, one);
        rt.acc_push(acc, tabled);
        rt.drop_handle(tabled);
        let owned = rt.print_take_args(acc);
        let rendered: Vec<String> = owned.iter().map(print_to_string).collect();
        assert_eq!(rendered, vec!["1".to_owned(), "nil".to_owned()]);
        assert_eq!(rt.live_count(), 0);
    }

    /// `argv_from` skips the program name, nils when empty, and
    /// decodes lossy (the T1192 FX1 shape, unit-pinned without
    /// touching the process environment).
    #[test]
    fn argv_from_pins() {
        use super::argv_from;
        use clj_value::print_to_string;
        assert!(matches!(argv_from(["prog"]), Value::Nil));
        let empty: [&str; 0] = [];
        assert!(matches!(argv_from(empty), Value::Nil));
        assert_eq!(print_to_string(&argv_from(["prog", "a", "b"])), "(a b)");
        let owned = ["prog".to_owned(), "a b".to_owned()];
        assert_eq!(print_to_string(&argv_from(owned)), "(a b)");
    }

    /// Non-UTF8 argv bytes decode lossy (Unix-only: only there can
    /// an `OsStr` hold invalid UTF-8).
    #[test]
    #[cfg(unix)]
    fn argv_from_decodes_lossy() {
        use super::argv_from;
        use clj_value::print_to_string;
        use std::os::unix::ffi::OsStrExt;
        let prog = std::ffi::OsStr::new("prog");
        let bad = std::ffi::OsStr::from_bytes(b"a\xffb");
        assert_eq!(print_to_string(&argv_from([prog, bad])), "(a�b)");
    }

    /// `call_prim_with` dispatches against a held runtime (the
    /// FFI entry's single-hold path — success and error arms).
    #[test]
    fn call_prim_with_dispatches() {
        let mut rt = Runtime::boot();
        let held = call_prim_with(
            &mut rt,
            Prim::Add as usize,
            vec![Value::Int(1), Value::Int(2)],
        );
        assert_eq!(format!("{held:?}"), "Ok(Int(3))");
        let err = call_prim_with(&mut rt, Prim::Sub as usize, Vec::new());
        assert!(err.is_err());
    }
}
