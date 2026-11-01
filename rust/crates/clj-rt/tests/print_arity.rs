// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

#![forbid(unsafe_code)]
//! `print` and arity-error helpers through the real FFI boundary
//! (no LLVM): return-shape pins here, exact stdout/stderr bytes in
//! the S8D corpus replay.

mod support;

use clj_value::Value;
use cljrt::{rt_arity_error, rt_print};
use support::{acc, inbox, render, str_id};

#[test]
fn print_returns_nil() {
    assert_eq!(render(rt_print(acc(Vec::new()))), "nil");
    assert_eq!(render(rt_print(acc(vec![inbox(Value::Int(1))]))), "nil");
    assert_eq!(
        render(rt_print(acc(vec![
            inbox(Value::Int(1)),
            inbox(Value::Str("x".into())),
            inbox(Value::Bool(false)),
        ]))),
        "nil"
    );
}

#[test]
fn arity_error_returns_zero() {
    let display = str_id("user/f");
    let path = str_id("f.clj");
    assert_eq!(rt_arity_error(display, 0, path, 1, 16), 0);
    // Missing str-ids degrade to a marker (still `0`, never a panic).
    assert_eq!(rt_arity_error(u64::MAX, 2, u64::MAX, 1, 16), 0);
}
