// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Shared handle helpers for the `clj-rt` FFI tests (safe — no
//! `unsafe` anywhere in these binaries).

use std::sync::atomic::{AtomicU64, Ordering};

use clj_value::{Value, print_to_string};
use cljrt::{
    rt_args_new, rt_args_push, rt_bool, rt_double, rt_int, rt_keyword, rt_nil, rt_str_chunk,
    rt_str_finish, rt_str_new, rt_str_register, rt_vec_new, rt_vec_push, test_take,
};

/// Test str-id allocator (the registry is process-global; ids
/// come from a counter so parallel tests never collide — no
/// codegen runs in these binaries, so counter ids are safe).
static NEXT_STR_ID: AtomicU64 = AtomicU64::new(0);

/// Chunk Rust text into a builder handle (8 bytes per call).
fn chunks(text: &str) -> u64 {
    let builder = rt_str_new();
    for piece in text.as_bytes().chunks(8) {
        let mut chunk = [0_u8; 8];
        chunk[..piece.len()].copy_from_slice(piece);
        rt_str_chunk(builder, u64::from_le_bytes(chunk), piece.len() as u8);
    }
    builder
}

/// Build an owned string handle (per-use construction — no
/// registry traffic, so leak-counter tests stay balanced).
/// (Each test binary compiles its own support copy: `acc` and
/// `inbox_str` are unused in some, hence the allows.)
#[allow(dead_code)]
pub fn inbox_str(text: &str) -> u64 {
    rt_str_finish(chunks(text))
}

/// Register text under a fresh str-id (the prologue path, for
/// prim paths and arity displays).
pub fn str_id(text: &str) -> u64 {
    let id = NEXT_STR_ID.fetch_add(1, Ordering::SeqCst);
    rt_str_register(id, chunks(text))
}

/// Pack owned handles into an acc for a prim/print call.
#[allow(dead_code)]
pub fn acc(handles: Vec<u64>) -> u64 {
    let acc = rt_args_new();
    for handle in handles {
        rt_args_push(acc, handle);
    }
    acc
}

/// Box a value into an owned handle for a helper call (the
/// helper consumes it). Shapes without a constructor shim
/// (chars, collections beyond vectors) panic — the suites only
/// inbox what generated code can build.
pub fn inbox(value: Value) -> u64 {
    match value {
        Value::Nil => rt_nil(),
        Value::Bool(b) => rt_bool(u8::from(b)),
        Value::Int(n) => rt_int(n),
        Value::Double(x) => rt_double(x),
        Value::Str(text) => inbox_str(&text),
        Value::Vector(items) => {
            let vec = rt_vec_new();
            for item in items.iter() {
                rt_vec_push(vec, inbox(item.clone()));
            }
            vec
        }
        Value::Keyword(keyword) => {
            let name = inbox_str(keyword.name());
            let (ns, has_ns) = match keyword.namespace() {
                Some(ns) => (inbox_str(ns), 1),
                None => (inbox_str(""), 0),
            };
            rt_keyword(name, ns, has_ns)
        }
        other => panic!("inbox: unsupported shape {other:?}"),
    }
}

/// Render a helper return to its `print` string (takes the
/// handle — panics on `0`, so callers check errors first).
pub fn render(handle: u64) -> String {
    print_to_string(&test_take(handle))
}
