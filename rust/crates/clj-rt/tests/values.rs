// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

#![forbid(unsafe_code)]
//! Value helpers, called through the real FFI boundary (no
//! LLVM): every pin renders through `print` (the specified
//! observable — never derived `Debug` on collections).

mod support;

use clj_value::Value;
use cljrt::{
    rt_argv, rt_bool, rt_clone, rt_double, rt_drop, rt_int, rt_is_truthy, rt_keyword, rt_nil,
    rt_str_get, rt_vec_new, rt_vec_push,
};
use support::{inbox, inbox_str, render, str_id};

#[test]
fn scalar_handles_render() {
    assert_eq!(render(rt_nil()), "nil");
    assert_eq!(render(rt_bool(0)), "false");
    assert_eq!(render(rt_bool(1)), "true");
    assert_eq!(render(rt_bool(7)), "true");
    assert_eq!(render(rt_int(0)), "0");
    assert_eq!(render(rt_int(i64::MIN)), "-9223372036854775808");
    assert_eq!(render(rt_int(i64::MAX)), "9223372036854775807");
    assert_eq!(render(rt_double(1.5)), "1.5");
    assert_eq!(render(rt_double(f64::NAN)), "##NaN");
}

#[test]
fn string_handles_decode_chunks() {
    assert_eq!(render(inbox_str("héllo")), "héllo");
    assert_eq!(render(inbox_str("")), "");
    assert_eq!(render(inbox_str("a�b")), "a�b");
    // Nine bytes span two chunks (8 + 1).
    assert_eq!(render(inbox_str("123456789")), "123456789");
    // The registry path: register once, fetch per use.
    let id = str_id("shared");
    assert_eq!(render(rt_str_get(id)), "shared");
    assert_eq!(render(rt_str_get(id)), "shared");
    assert_eq!(rt_str_get(u64::MAX), 0);
}

#[test]
fn keyword_handles_decode_parts() {
    // T73 item 8: plain, namespaced, and non-ASCII keywords; any
    // nonzero flag reads the ns handle, zero ignores it (but
    // still consumes it — the leak test pins the balance).
    let miss = inbox_str("miss");
    let empty = inbox_str("");
    assert_eq!(render(rt_keyword(miss, empty, 0)), ":miss");
    let ns = inbox_str("user");
    let id = inbox_str("id");
    assert_eq!(render(rt_keyword(id, ns, 7)), ":user/id");
    let uni = inbox_str("λ");
    let empty = inbox_str("");
    assert_eq!(render(rt_keyword(uni, empty, 0)), ":λ");
    // Non-string parts degrade to empty (graceful, documented).
    let num = rt_int(1);
    let num2 = rt_int(2);
    assert_eq!(render(rt_keyword(num, num2, 1)), ":/");
    let num = rt_int(3);
    let empty = inbox_str("");
    assert_eq!(render(rt_keyword(num, empty, 0)), ":");
}

#[test]
fn vectors_build_in_order() {
    let vec = rt_vec_new();
    assert_eq!(render(rt_clone(vec)), "[]");
    let vec = rt_vec_push(vec, inbox(Value::Int(1)));
    let vec = rt_vec_push(vec, inbox(Value::Str("s".into())));
    assert_eq!(render(vec), "[1 s]");
}

#[test]
fn argv_reads_the_process_environment() {
    // The zero-param shim reads `std::env::args_os` (shape-only
    // pin — the test process's own args are harness-dependent);
    // exact skip/lossy/nil semantics live in `argv_from` unit pins.
    let rendered = render(rt_argv());
    assert!(rendered == "nil" || rendered.starts_with('('), "{rendered}");
}

#[test]
fn vec_push_onto_non_vector_degrades_to_fresh() {
    // A live non-vector slot is overwritten in place (same
    // handle back); a slotless inline is reborn fresh.
    let slot = inbox(Value::Str("s".into()));
    let vec = rt_vec_push(slot, inbox(Value::Int(1)));
    assert_eq!(vec, slot);
    assert_eq!(render(vec), "[1]");
    let inline = inbox(Value::Int(9));
    let vec = rt_vec_push(inline, inbox(Value::Int(1)));
    assert_ne!(vec, inline);
    assert_eq!(render(vec), "[1]");
}

#[test]
fn truthiness_consumes() {
    let cases = [
        (Value::Nil, 0),
        (Value::Bool(false), 0),
        (Value::Bool(true), 1),
        (Value::Int(0), 1),
        (Value::Str("".into()), 1),
    ];
    for (value, want) in cases {
        assert_eq!(rt_is_truthy(inbox(value)), want);
    }
}

#[test]
fn clone_bumps_and_drop_frees() {
    let original = inbox(Value::Str("v".into()));
    let copy = rt_clone(original);
    assert_eq!(render(copy), "v");
    assert_eq!(render(original), "v");
    rt_drop(inbox(Value::Int(1)));
    // Stale clones fail stop at `0`; stale drops are silent.
    assert_eq!(rt_clone(original), 0);
    assert_eq!(rt_clone(0), 0);
    rt_drop(original);
    rt_drop(0);
}
