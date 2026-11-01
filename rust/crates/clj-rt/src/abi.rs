// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! The safe `extern "C"` boundary: helpers over `u64` handles
//! (Z3; S8 plan D1/D2/D7 retargeted — no pointer crosses).
//!
//! Contract (enforced by S8C codegen, documented per shim):
//! helper arguments are consumed (each handle dropped exactly
//! once); the return is owned (`0` on error — the generated code
//! zero-checks and exits 1, exactly where the old null check ran).
//! Strings arrive chunked as scalars and live in the startup
//! registry under str-ids; arg lists ride an accumulator the
//! prim/print call always consumes, on error paths too.
//!
//! Every shim is total: stale, wrong-tag, and invalid handles
//! degrade gracefully and deterministically (drops are silent,
//! value positions yield `0` or a substitute — never UB, never a
//! panic across the boundary). Generations make stale use
//! fail-stop at the caller's zero check instead of touching a
//! recycled slot. The shims never panic: unexpected shapes are
//! caller (codegen) bugs, and the graceful arms keep them loud
//! (fail-stop `0`) or documented (silent substitutes).
//!
//! Every shim is `#[unsafe(no_mangle)]` so the platform link finds the
//! plain `rt_*` symbols in the static archive.

use std::sync::Arc;

use clj_eval::{NativeError, Prim};
use clj_value::{Keyword, Value, Vector};

use crate::handles;
use crate::runtime::{RUNTIME, Runtime, argv_from, call_prim_with, error_line, print_join};

/// The `extern "C"` symbol lowering [`Prim`] calls to (S8C
/// codegen's single source of truth — no parallel table).
#[must_use]
pub fn prim_symbol(prim: Prim) -> &'static str {
    match prim {
        Prim::Add => "rt_add",
        Prim::Sub => "rt_sub",
        Prim::Mul => "rt_mul",
        Prim::Quot => "rt_quot",
        Prim::Lt => "rt_lt",
        Prim::Le => "rt_le",
        Prim::Gt => "rt_gt",
        Prim::Ge => "rt_ge",
        Prim::Eq => "rt_eq",
        Prim::Count => "rt_count",
        Prim::Nth => "rt_nth",
        Prim::Get => "rt_get",
        Prim::Conj => "rt_conj",
    }
}

/// Resolve a baked path id for an error line (missing ids —
/// codegen bugs — degrade to a marker, never panic).
fn error_path(rt: &Runtime, path_id: u64) -> Arc<str> {
    rt.str_get(path_id)
        .unwrap_or_else(|| Arc::from("<bad-str-id>"))
}

/// Take a keyword part handle into text (stale handles and
/// non-`Str` parts degrade to empty — codegen always passes
/// interned strings).
fn keyword_part(rt: &mut Runtime, handle: u64) -> String {
    match rt.take_handle(handle) {
        Some(Value::Str(text)) => text.to_string(),
        _ => String::new(),
    }
}

/// One `nil` handle (inline constant — no lock, no table).
#[unsafe(no_mangle)]
pub extern "C" fn rt_nil() -> u64 {
    handles::NIL_HANDLE
}

/// One boolean handle: `0` is false, anything else true (inline
/// constant — no lock, no table).
#[unsafe(no_mangle)]
pub extern "C" fn rt_bool(flag: u8) -> u64 {
    if flag != 0 {
        handles::TRUE_HANDLE
    } else {
        handles::FALSE_HANDLE
    }
}

/// One integer handle (small ints inline lock-free; out-of-range
/// ints ride a value slot — same `Value::Int` either way).
#[unsafe(no_mangle)]
pub extern "C" fn rt_int(n: i64) -> u64 {
    match handles::encode_int(n) {
        Some(handle) => handle,
        None => RUNTIME
            .lock()
            .expect("clj-rt evaluator lock")
            .give(Value::Int(n)),
    }
}

/// One double handle (doubles ride value slots — a full `i64`
/// beside them leaves no tag room inline).
#[unsafe(no_mangle)]
pub extern "C" fn rt_double(x: f64) -> u64 {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .give(Value::Double(x))
}

/// One keyword handle from two string handles (the L1 pick):
/// `name` is required; the `ns` handle is consumed but only read
/// when `has_ns` is nonzero (`0` is unqualified — codegen passes
/// the registered empty string, mirroring the old slice shape).
#[unsafe(no_mangle)]
pub extern "C" fn rt_keyword(name: u64, ns: u64, has_ns: u8) -> u64 {
    let mut rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    let name = keyword_part(&mut rt, name);
    let namespace = keyword_part(&mut rt, ns);
    let keyword = if has_ns != 0 {
        Keyword::new(Some(namespace.as_str()), name.as_str())
    } else {
        Keyword::new(None, name.as_str())
    };
    rt.give(Value::Keyword(keyword))
}

/// One empty vector handle.
#[unsafe(no_mangle)]
pub extern "C" fn rt_vec_new() -> u64 {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .give(Value::Vector(Vector::from_vec(Vec::new())))
}

/// Push the owned `elem` handle onto the owned `vec` slot in
/// place and return `vec` for chaining (a live non-vector slot
/// is overwritten, parity with the old wrong-type arm; a dead,
/// slotless, or wrong-tag `vec` is a caller bug — it degrades to
/// a fresh single-element vector).
#[unsafe(no_mangle)]
pub extern "C" fn rt_vec_push(vec: u64, elem: u64) -> u64 {
    let mut rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    let elem = rt.take_handle(elem).unwrap_or(Value::Nil);
    rt.vec_push_value(vec, elem)
}

/// The `*command-line-args*` value, read from the process
/// environment (T1192 FX1 D-argv via [`argv_from`]): `argv[0]`
/// is skipped; no remaining args is nil, else a list of strings.
/// Zero params — generated `main` ignores its C `argc`/`argv`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_argv() -> u64 {
    let mut rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    let value = argv_from(std::env::args_os());
    rt.give(value)
}

/// Clone a handle (inline: identity; table: refcount bump;
/// stale: `0` — fail-stop at the caller's null check).
#[unsafe(no_mangle)]
pub extern "C" fn rt_clone(handle: u64) -> u64 {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .clone_handle(handle)
}

/// Drop a handle (table: dec, free at zero; inline and stale:
/// silent no-op).
#[unsafe(no_mangle)]
pub extern "C" fn rt_drop(handle: u64) {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .drop_handle(handle);
}

/// Consume the handle and report Clojure truthiness: `0` for nil
/// and false, `1` for everything else (the `if` test; stale
/// handles read false).
#[unsafe(no_mangle)]
pub extern "C" fn rt_is_truthy(handle: u64) -> u8 {
    let mut rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    u8::from(
        rt.take_handle(handle)
            .is_some_and(|value| value.is_truthy()),
    )
}

/// The `print` intrinsic: consume the acc, write the
/// space-joined rendering with NO trailing newline, return nil.
/// Infallible (stale accs read empty, stale args read nil —
/// print sites carry no null check, so this must never fail).
#[unsafe(no_mangle)]
pub extern "C" fn rt_print(acc: u64) -> u64 {
    let mut rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    let owned = rt.print_take_args(acc);
    print!("{}", print_join(&owned));
    let _ = std::io::Write::flush(&mut std::io::stdout());
    rt.give(Value::Nil)
}

/// A sibling call with the wrong arg count: always fails with the
/// shared arity text over the stable `user/name` display (both
/// baked as str-ids), returning `0`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_arity_error(
    display_id: u64,
    got: u64,
    path_id: u64,
    line: u32,
    col: u32,
) -> u64 {
    let rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    let display = error_path(&rt, display_id);
    let path = error_path(&rt, path_id);
    let err = NativeError::arity(&display, usize::try_from(got).unwrap_or(usize::MAX));
    eprintln!("{}", error_line(&path, line, col, &err));
    0
}

/// The shared primitive body: take the acc, resolve every arg,
/// drop every arg handle (on every path — the leak counter never
/// trips), call the shared native, give the result — or print
/// `path:line:col: Class: message` and return `0`.
fn prim_call(index: usize, acc: u64, path_id: u64, line: u32, col: u32) -> u64 {
    let mut rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    let Some(arg_handles) = rt.acc_take(acc) else {
        let path = error_path(&rt, path_id);
        eprintln!("{path}:{line}:{col}: internal error: stale arg accumulator");
        return 0;
    };
    let mut owned = Vec::with_capacity(arg_handles.len());
    for handle in &arg_handles {
        match rt.resolve_handle(*handle) {
            Some(value) => owned.push(value),
            None => {
                for handle in &arg_handles {
                    rt.drop_handle(*handle);
                }
                let path = error_path(&rt, path_id);
                eprintln!("{path}:{line}:{col}: internal error: stale arg handle");
                return 0;
            }
        }
    }
    for handle in &arg_handles {
        rt.drop_handle(*handle);
    }
    match call_prim_with(&mut rt, index, owned) {
        Ok(value) => rt.give(value),
        Err(err) => {
            let path = error_path(&rt, path_id);
            eprintln!("{}", error_line(&path, line, col, &err));
            0
        }
    }
}

macro_rules! def_prim {
    ($shim:ident, $index:expr) => {
        #[doc = "One closed-set primitive (see `prim_call`): take the acc, call the shared native."]
        #[unsafe(no_mangle)]
        pub extern "C" fn $shim(acc: u64, path_id: u64, line: u32, col: u32) -> u64 {
            prim_call($index, acc, path_id, line, col)
        }
    };
}

def_prim!(rt_add, 0);
def_prim!(rt_sub, 1);
def_prim!(rt_mul, 2);
def_prim!(rt_quot, 3);
def_prim!(rt_lt, 4);
def_prim!(rt_le, 5);
def_prim!(rt_gt, 6);
def_prim!(rt_ge, 7);
def_prim!(rt_eq, 8);
def_prim!(rt_count, 9);
def_prim!(rt_nth, 10);
def_prim!(rt_get, 11);
def_prim!(rt_conj, 12);

// The handle-table construction shims (landed additive in Z3
// step 1): integer-only signatures, safe bodies — no pointer
// crosses the boundary.

/// A fresh arg accumulator (pushed, then consumed exactly once
/// by a prim/print call — the call drops it on every path
/// including errors, so the leak counter never trips).
#[unsafe(no_mangle)]
pub extern "C" fn rt_args_new() -> u64 {
    RUNTIME.lock().expect("clj-rt evaluator lock").acc_new()
}

/// Push a value handle onto an acc (ownership transfers);
/// returns the acc, `0`-propagating on stale accs.
#[unsafe(no_mangle)]
pub extern "C" fn rt_args_push(acc: u64, handle: u64) -> u64 {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .acc_push(acc, handle)
}

/// A fresh string builder (chunked scalar construction).
#[unsafe(no_mangle)]
pub extern "C" fn rt_str_new() -> u64 {
    RUNTIME.lock().expect("clj-rt evaluator lock").builder_new()
}

/// Append the low `nbytes` of `chunk` (little-endian, clamped
/// to 8); returns the builder, `0`-propagating on stale ones.
#[unsafe(no_mangle)]
pub extern "C" fn rt_str_chunk(builder: u64, chunk: u64, nbytes: u8) -> u64 {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .builder_chunk(builder, chunk, nbytes)
}

/// Intern built text under a startup str-id (the `main`
/// prologue path); returns the id for chaining.
#[unsafe(no_mangle)]
pub extern "C" fn rt_str_register(id: u64, builder: u64) -> u64 {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .str_register(id, builder)
}

/// Fetch interned text as a fresh owned string (`0` for
/// missing ids — fail-stop at the caller's null check).
#[unsafe(no_mangle)]
pub extern "C" fn rt_str_get(id: u64) -> u64 {
    let mut rt = RUNTIME.lock().expect("clj-rt evaluator lock");
    match rt.str_get(id) {
        Some(text) => rt.give(Value::Str(text)),
        None => 0,
    }
}

/// Finish a builder into an owned string value (the per-use
/// fallback); stale builders yield `0`.
#[unsafe(no_mangle)]
pub extern "C" fn rt_str_finish(builder: u64) -> u64 {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .str_finish(builder)
}

/// Live handle occupancy for the leak-counter tests (test
/// support — codegen never emits it).
#[unsafe(no_mangle)]
pub extern "C" fn rt_live_count() -> u64 {
    RUNTIME.lock().expect("clj-rt evaluator lock").live_count()
}

/// Take a handle into an owned value for the FFI tests (panics
/// on stale handles — test-loud by design). Test support only:
/// plain Rust, not `extern`, never emitted by codegen.
#[doc(hidden)]
pub fn test_take(handle: u64) -> Value {
    RUNTIME
        .lock()
        .expect("clj-rt evaluator lock")
        .take_handle(handle)
        .expect("test handle resolves")
}
