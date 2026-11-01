// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

#![forbid(unsafe_code)]
//! The L1 leak-counter test: every entry point consumes its acc
//! (with every arg handle) on every path including errors, so the
//! live count returns to its entry value. ONE test in its OWN
//! binary: the runtime is process-global, so any sibling test
//! holding handles concurrently would trip the counter.

mod support;

use clj_value::Value;
use cljrt::{
    rt_args_new, rt_arity_error, rt_clone, rt_drop, rt_keyword, rt_live_count, rt_print, rt_quot,
    rt_str_get, rt_sub, rt_vec_new, rt_vec_push,
};
use support::{acc, inbox, inbox_str, render, str_id};

/// The live count after draining this binary's traffic (entry
/// baseline — the binary runs only this test, so it holds).
fn live() -> u64 {
    rt_live_count()
}

#[test]
fn every_entry_point_balances() {
    let entry = live();
    // Prim arity-error: `(-)` takes the acc and fails.
    let path = str_id("f.clj");
    assert_eq!(rt_sub(rt_args_new(), path, 1, 1), 0);
    assert_eq!(live(), entry);
    // Prim domain-error: `(quot 1 0)` resolves, drops, then fails.
    let quot_acc = acc(vec![inbox(Value::Int(1)), inbox(Value::Int(0))]);
    assert_eq!(rt_quot(quot_acc, path, 1, 1), 0);
    assert_eq!(live(), entry);
    // Stale accs and stale args fail without leaking.
    assert_eq!(rt_quot(0, path, 1, 1), 0);
    assert_eq!(live(), entry);
    let stale = inbox(Value::Str("gone".into()));
    rt_drop(stale);
    assert_eq!(rt_quot(acc(vec![stale]), path, 1, 1), 0);
    assert_eq!(live(), entry);
    // Print consumes (missing ids still balance — infallible).
    let print_acc = acc(vec![inbox(Value::Int(1)), inbox(Value::Str("x".into()))]);
    assert_eq!(render(rt_print(print_acc)), "nil");
    assert_eq!(live(), entry);
    assert_eq!(render(rt_print(0)), "nil");
    assert_eq!(live(), entry);
    // Sibling arity-errors carry no acc (codegen creates accs only
    // after args lower) — still balanced by construction.
    let display = str_id("user/f");
    assert_eq!(rt_arity_error(display, 2, path, 1, 16), 0);
    assert_eq!(live(), entry);
    // Clones balance against drops; double-drops are silent.
    let tabled = inbox(Value::Str("v".into()));
    assert_eq!(live(), entry + 1);
    let copy = rt_clone(tabled);
    rt_drop(copy);
    assert_eq!(live(), entry + 1);
    rt_drop(tabled);
    rt_drop(tabled);
    assert_eq!(live(), entry);
    // Vectors and keywords balance through construction.
    let vec = rt_vec_new();
    let vec = rt_vec_push(vec, inbox(Value::Int(1)));
    assert_eq!(render(vec), "[1]");
    assert_eq!(live(), entry);
    let name = inbox_str("id");
    let ns = inbox_str("user");
    assert_eq!(render(rt_keyword(name, ns, 1)), ":user/id");
    assert_eq!(live(), entry);
    // Registry fetches are owned (each `get` balances one drop).
    let id = str_id("registry");
    let first = rt_str_get(id);
    let second = rt_str_get(id);
    assert_eq!(live(), entry + 2);
    rt_drop(first);
    rt_drop(second);
    assert_eq!(live(), entry);
}
