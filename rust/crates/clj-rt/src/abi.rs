// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! The audited `unsafe` boundary: `extern "C"` helpers over opaque
//! `*mut Value` boxes (S8 plan D1/D2/D7).
//!
//! Contract (enforced by S8C codegen, documented per shim):
//! pointers are non-null and valid; helper arguments are consumed
//! (each box freed exactly once); the return is owned (null on
//! error — the generated code null-checks and exits 1); byte
//! slices are valid for their length. Violating the contract is a
//! caller bug and instant UB — the shims do not null-check (the
//! FFI boundary is internal to a built binary, not a public C
//! API), except [`rt_vec_push`]'s wrong-type arm, which degrades
//! to a fresh vector so the line stays meaningful.
//!
//! Every shim is `#[unsafe(no_mangle)]` so the platform link finds the
//! plain `rt_*` symbols in the static archive.

use std::ffi::{CStr, c_char};
use std::ptr;
use std::sync::Arc;

use clj_eval::{NativeError, Prim};
use clj_value::{Keyword, List, Value, Vector};

use crate::runtime::{RUNTIME, call_prim, error_line, print_join};

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

/// Borrow a caller byte slice.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    // Safety: the caller guarantees validity (see module docs).
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Decode a caller byte string (lossy — codegen only ever passes
/// valid UTF-8, so the lossy arm never fires in practice).
unsafe fn text(ptr: *const u8, len: usize) -> String {
    String::from_utf8_lossy(unsafe { bytes(ptr, len) }).into_owned()
}

/// Box a fresh owned value for the caller.
fn give(value: Value) -> *mut Value {
    Box::into_raw(Box::new(value))
}

/// Take ownership of a caller box.
unsafe fn take(ptr: *mut Value) -> Value {
    // Safety: the caller transfers ownership (see module docs).
    unsafe { *Box::from_raw(ptr) }
}

/// Take ownership of every box in a caller array (the array
/// itself stays caller-owned — only the boxes move).
unsafe fn take_args(args: *const *mut Value, argc: usize) -> Vec<Value> {
    // Safety: the caller guarantees a valid array (see module docs).
    let items = unsafe { std::slice::from_raw_parts(args, argc) };
    items.iter().map(|ptr| unsafe { take(*ptr) }).collect()
}

/// One `nil` box.
#[unsafe(no_mangle)]
pub extern "C" fn rt_nil() -> *mut Value {
    give(Value::Nil)
}

/// One boolean box: `0` is false, anything else true.
#[unsafe(no_mangle)]
pub extern "C" fn rt_bool(flag: u8) -> *mut Value {
    give(Value::Bool(flag != 0))
}

/// One integer box.
#[unsafe(no_mangle)]
pub extern "C" fn rt_int(n: i64) -> *mut Value {
    give(Value::Int(n))
}

/// One double box.
#[unsafe(no_mangle)]
pub extern "C" fn rt_double(x: f64) -> *mut Value {
    give(Value::Double(x))
}

/// One string box decoding `len` bytes at `ptr`.
///
/// # Safety
///
/// `ptr` must be valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_str(ptr: *const u8, len: usize) -> *mut Value {
    give(Value::Str(Arc::from(unsafe { text(ptr, len) }.as_str())))
}

/// One keyword box: `name` is required; the `ns` slice is read
/// only when `has_ns` is nonzero (`0` is unqualified — the slice
/// must still be valid, but is never read).
///
/// # Safety
///
/// Both byte slices must be valid for their lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_keyword(
    name_ptr: *const u8,
    name_len: usize,
    ns_ptr: *const u8,
    ns_len: usize,
    has_ns: u8,
) -> *mut Value {
    let name = unsafe { text(name_ptr, name_len) };
    let ns = if has_ns != 0 {
        Some(unsafe { text(ns_ptr, ns_len) })
    } else {
        None
    };
    give(Value::Keyword(Keyword::new(ns.as_deref(), &name)))
}

/// One empty vector box.
#[unsafe(no_mangle)]
pub extern "C" fn rt_vec_new() -> *mut Value {
    give(Value::Vector(Vector::from_vec(Vec::new())))
}

/// Push the owned `elem` box onto the owned `vec` box in place
/// and return `vec` for chaining (a non-vector `vec` is a caller
/// bug — it degrades to a fresh single-element vector).
///
/// # Safety
///
/// Both boxes must be valid; `elem` is consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_vec_push(vec: *mut Value, elem: *mut Value) -> *mut Value {
    let elem = unsafe { take(elem) };
    // Safety: the caller guarantees a valid box (see module docs).
    let slot = unsafe { &mut *vec };
    let old = std::mem::replace(slot, Value::Nil);
    *slot = match old {
        Value::Vector(items) => Value::Vector(items.push_back(elem)),
        _ => Value::Vector(Vector::from_vec(vec![elem])),
    };
    vec
}

/// The `*command-line-args*` value from C `argc`/`argv` (T1192
/// FX1 D-argv): `argv[0]` (the program name) is skipped; no
/// remaining args is nil, else a list of strings — the JVM
/// shape (fx1-argv probe). Bytes decode lossy (like the JVM's
/// platform decode).
///
/// # Safety
///
/// `argv` must be valid for `argc` entries (`argc >= 0`), each a
/// valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_argv(argv: *const *const c_char, argc: i32) -> *mut Value {
    let mut items = Vec::new();
    // Safety: the caller guarantees validity (see module docs).
    for i in 1..argc {
        let entry = unsafe { *argv.offset(i as isize) };
        let text = unsafe { CStr::from_ptr(entry) }
            .to_string_lossy()
            .into_owned();
        items.push(Value::Str(Arc::from(text.as_str())));
    }
    if items.is_empty() {
        give(Value::Nil)
    } else {
        give(Value::List(List::from_vec(items)))
    }
}

/// Clone the borrowed box into a fresh owned box.
///
/// # Safety
///
/// `value` must be a valid box (borrowed, never freed here).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_clone(value: *const Value) -> *mut Value {
    // Safety: the caller guarantees a valid box (see module docs).
    give(unsafe { (*value).clone() })
}

/// Free the owned box.
///
/// # Safety
///
/// `value` must be a valid box, freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_drop(value: *mut Value) {
    drop(unsafe { take(value) });
}

/// Consume the box and report Clojure truthiness: `0` for nil
/// and false, `1` for everything else (the `if` test).
///
/// # Safety
///
/// `value` must be a valid box, consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_is_truthy(value: *mut Value) -> u8 {
    u8::from(unsafe { take(value) }.is_truthy())
}

/// The `print` intrinsic: consume the boxes, write the
/// space-joined rendering with NO trailing newline, return nil.
///
/// # Safety
///
/// `args` must be a valid array of `argc` valid boxes, consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_print(args: *const *mut Value, argc: usize) -> *mut Value {
    let owned = unsafe { take_args(args, argc) };
    print!("{}", print_join(&owned));
    let _ = std::io::Write::flush(&mut std::io::stdout());
    give(Value::Nil)
}

/// A sibling call with the wrong arg count: always fails with the
/// shared arity text over the stable `user/name` display.
///
/// # Safety
///
/// Both byte slices must be valid for their lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rt_arity_error(
    display: *const u8,
    display_len: usize,
    got: usize,
    path: *const u8,
    path_len: usize,
    line: u32,
    col: u32,
) -> *mut Value {
    let err = NativeError::arity(&unsafe { text(display, display_len) }, got);
    let path = unsafe { text(path, path_len) };
    eprintln!("{}", error_line(&path, line, col, &err));
    ptr::null_mut()
}

/// The shared primitive body: consume the boxes, call the shared
/// native, box the result — or print `path:line:col: Class:
/// message` and return null.
///
/// # Safety
///
/// `args` must be a valid array of `argc` valid boxes,
/// consumed; the path slice must be valid for its length.
unsafe fn prim_call(
    index: usize,
    args: *const *mut Value,
    argc: usize,
    path: *const u8,
    path_len: usize,
    line: u32,
    col: u32,
) -> *mut Value {
    let owned = unsafe { take_args(args, argc) };
    match call_prim(index, owned) {
        Ok(value) => give(value),
        Err(err) => {
            let path = unsafe { text(path, path_len) };
            eprintln!("{}", error_line(&path, line, col, &err));
            ptr::null_mut()
        }
    }
}

macro_rules! def_prim {
    ($shim:ident, $index:expr) => {
        #[doc = "One closed-set primitive (see `prim_call`): consume the boxes, call the shared native."]
        #[doc = ""]
        #[doc = " # Safety"]
        #[doc = ""]
        #[doc = " `args` must be a valid array of `argc` valid boxes, consumed;"]
        #[doc = " the path slice must be valid for its length."]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $shim(
            args: *const *mut Value,
            argc: usize,
            path: *const u8,
            path_len: usize,
            line: u32,
            col: u32,
        ) -> *mut Value {
            unsafe { prim_call($index, args, argc, path, path_len, line, col) }
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

// The handle-table shims (Z3 step 1, additive): integer-only
// signatures, safe bodies — no pointer crosses the boundary.
// They coexist with the pointer shims above; the flip commit
// deletes the pointer half and retargets codegen here.

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
