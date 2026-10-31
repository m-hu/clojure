// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-122 slice-1 tests: the `clojure.spec.alpha` minimal core
//! (registry, `def`, predicate specs, `s/and`, `valid?`,
//! `conform`/`unform`, basic `explain`).
//!
//! Oracle: `docs/gap-122/probes/slice1.out`, produced by
//! `docs/gap-122/probes/slice1.clj` under
//! `CP=clojure-1.12.6.jar:spec.alpha-0.5.238.jar:core.specs.alpha-0.4.74.jar`
//! (`java -cp $CP clojure.main docs/gap-122/probes/slice1.clj`).
//! Pins that diverge from the oracle say so (each divergence is
//! listed in `docs/gap-122/slice-1.md`).

use clj_eval::{Evaluator, SourceError, boot};
use clj_value::{PrintOpts, Value, pr_to_string};

fn pr(v: &Value) -> String {
    pr_to_string(v, &PrintOpts::pr())
}

/// Evaluate and return the pr-string of the LAST top-level form.
fn run(eval: &mut Evaluator, src: &str) -> String {
    eval.eval_source(src)
        .unwrap()
        .iter()
        .map(|v| match eval.realize_deep(v) {
            Ok(r) => pr(&r),
            Err(e) => panic!("realize failed: {e:?}"),
        })
        .last()
        .unwrap()
}

/// The `Class | message` of an uncaught throw.
fn uncaught(eval: &mut Evaluator, src: &str) -> String {
    match eval.eval_source(src) {
        Err(SourceError::Uncaught(t)) => format!(
            "{} | {}",
            t.class.jvm_name(),
            t.message.as_deref().unwrap_or("")
        ),
        Err(SourceError::Compile(e)) => format!("{} | {}", e.class, e.message),
        Err(other) => panic!("expected an uncaught throw, got {other:?}"),
        Ok(vs) => panic!("expected a throw, got values: {vs:?}"),
    }
}

/// Require spec with the `s` alias (as the oracle does).
fn require_spec(eval: &mut Evaluator) {
    eval.eval_source("(require '[clojure.spec.alpha :as s])")
        .unwrap();
}

// ---- load ----

#[test]
fn spec_alpha_loads_via_require() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(some? (find-ns 'clojure.spec.alpha))"),
        "true"
    );
}

#[test]
fn spec_alpha_marks_loaded() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(contains? (loaded-libs) 'clojure.spec.alpha)"),
        "true"
    );
}

// ---- valid? / conform / unform over predicate specs ----

#[test]
fn valid_predicate_spec() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/even 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/even 5)"), "false");
}

#[test]
fn valid_bare_predicate() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? even? 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? even? 5)"), "false");
}

#[test]
fn conform_unform_predicate() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(run(&mut eval, "(s/conform :user/even 4)"), "4");
    assert_eq!(
        run(&mut eval, "(s/conform :user/even 5)"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(run(&mut eval, "(s/unform :user/even 4)"), "4");
    assert_eq!(
        run(&mut eval, "(s/invalid? (s/conform :user/even 5))"),
        "true"
    );
    assert_eq!(run(&mut eval, "(s/invalid? 4)"), "false");
}

#[test]
fn valid_set_spec() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/small #{1 2 3})").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/small 2)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/small 9)"), "false");
}

#[test]
fn valid_anonymous_predicate() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/big #(> % 100))").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/big 101)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/big 99)"), "false");
}

#[test]
fn spec_macro_builds_predicate_spec() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? (s/spec even?) 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? (s/spec even?) 5)"), "false");
    assert_eq!(run(&mut eval, "(s/describe (s/spec even?))"), "even?");
}

// ---- s/and ----

#[test]
fn valid_and_spec() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even-small (s/and even? #(< % 10)))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/even-small 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/even-small 5)"), "false");
    assert_eq!(run(&mut eval, "(s/valid? :user/even-small 42)"), "false");
}

#[test]
fn conform_unform_and() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even-small (s/and even? #(< % 10)))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/conform :user/even-small 4)"), "4");
    assert_eq!(
        run(&mut eval, "(s/conform :user/even-small 5)"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(run(&mut eval, "(s/unform :user/even-small 4)"), "4");
}

#[test]
fn and_three_preds() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/tri (s/and integer? even? #(< % 10)))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/tri 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/tri 5)"), "false");
    assert_eq!(run(&mut eval, "(s/valid? :user/tri 42)"), "false");
    assert_eq!(run(&mut eval, "(s/valid? :user/tri 4.0)"), "false");
}

#[test]
fn and_forwards_registry_refs() {
    // The `and` macro embeds idents unevaluated; the AndSpec
    // resolves them lazily, so building the `and` before its target
    // exists must not throw (upstream delay behavior).
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/late (s/and :user/later even?))")
        .unwrap();
    eval.eval_source("(s/def :user/later integer?)").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/late 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/late 4.0)"), "false");
}

// ---- explain ----

#[test]
fn explain_str_predicate() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/even 5)"),
        "\"5 - failed: even? spec: :user/even\\n\""
    );
}

#[test]
fn explain_str_and_first_pred() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even-small (s/and even? #(< % 10)))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/even-small 5)"),
        "\"5 - failed: even? spec: :user/even-small\\n\""
    );
}

#[test]
fn explain_str_and_second_pred() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even-small (s/and even? #(< % 10)))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/even-small 42)"),
        "\"42 - failed: (< % 10) spec: :user/even-small\\n\""
    );
}

#[test]
fn explain_str_bare_predicate() {
    // DIVERGENCE (pinned): upstream prints the resolved fn name
    // (`even?`) via JVM reflection (`fn-sym`); the port has no
    // fn-name primitive, so bare-fn specs explain as
    // `:clojure.spec.alpha/unknown`.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str even? 5)"),
        "\"5 - failed: :clojure.spec.alpha/unknown\\n\""
    );
}

#[test]
fn explain_str_success() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/even 4)"),
        "\"Success!\\n\""
    );
}

#[test]
fn explain_data_shape() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-data :user/even 5)"),
        "#:clojure.spec.alpha{:problems [{:path [], :pred clojure.core/even?, :val 5, :via [:user/even], :in []}], :spec :user/even, :value 5}"
    );
    assert_eq!(run(&mut eval, "(s/explain-data :user/even 4)"), "nil");
}

#[test]
fn explain_returns_nil() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(
        run(&mut eval, "(with-out-str (s/explain :user/even 5))"),
        "\"5 - failed: even? spec: :user/even\\n\""
    );
}

// ---- registry ----

#[test]
fn registry_and_get_spec() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(
        run(&mut eval, "(contains? (s/registry) :user/even)"),
        "true"
    );
    assert_eq!(run(&mut eval, "(some? (s/get-spec :user/even))"), "true");
    assert_eq!(run(&mut eval, "(s/get-spec :user/nope)"), "nil");
}

#[test]
fn spec_predicate() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(
        run(&mut eval, "(some? (s/spec? (s/get-spec :user/even)))"),
        "true"
    );
    assert_eq!(run(&mut eval, "(s/spec? even?)"), "nil");
    assert_eq!(run(&mut eval, "(s/spec? :user/even)"), "nil");
}

#[test]
fn def_returns_key_and_nil_removes() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/def :user/tmp string?)"), ":user/tmp");
    assert_eq!(run(&mut eval, "(s/valid? :user/tmp \"a\")"), "true");
    eval.eval_source("(s/def :user/tmp nil)").unwrap();
    assert_eq!(run(&mut eval, "(s/get-spec :user/tmp)"), "nil");
}

#[test]
fn def_symbol_key() {
    // A resolvable symbol key registers under the qualified symbol.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/def user/sym-spec string?)"),
        "user/sym-spec"
    );
    assert_eq!(run(&mut eval, "(s/valid? 'user/sym-spec \"a\")"), "true");
}

#[test]
fn def_bare_keyword_throws_assert() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/def :bare string?)"),
        "java.lang.AssertionError | Assert failed: k must be namespaced keyword or resolvable symbol\n(c/and (ident? k) (namespace k))"
    );
}

#[test]
fn unresolvable_spec_throws() {
    // DIVERGENCE (pinned): upstream throws plain `Exception`; the
    // port has no `Exception.` constructor path, so the port throws
    // `ExceptionInfo` with the same message (still caught by
    // `catch Exception`).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/valid? :user/definitely-missing 1)"),
        "clojure.lang.ExceptionInfo | Unable to resolve spec: :user/definitely-missing"
    );
    assert_eq!(
        uncaught(&mut eval, "(s/conform :user/definitely-missing 1)"),
        "clojure.lang.ExceptionInfo | Unable to resolve spec: :user/definitely-missing"
    );
}

#[test]
fn explain_non_fn_pred_throws() {
    // `(s/and 42)`: explaining hits `dt`'s "not a fn" guard.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/explain-str (s/and 42) 1)"),
        "clojure.lang.ExceptionInfo | 42 is not a fn, expected predicate fn"
    );
}

// ---- form / describe ----

#[test]
fn form_and_describe_predicate() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    assert_eq!(run(&mut eval, "(s/form :user/even)"), "clojure.core/even?");
    assert_eq!(run(&mut eval, "(s/describe :user/even)"), "even?");
}

#[test]
fn form_and_describe_and() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even-small (s/and even? #(< % 10)))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/form :user/even-small)"),
        "(clojure.spec.alpha/and clojure.core/even? (clojure.core/fn [%] (clojure.core/< % 10)))"
    );
    assert_eq!(
        run(&mut eval, "(s/describe :user/even-small)"),
        "(and even? (< % 10))"
    );
}

// ---- out-of-scope errors ----

#[test]
fn regex_spec_out_of_scope() {
    // Gap-122 slice E retired the slice-1 guard: a regex-op-shaped
    // map is now a usable spec (see spec_regex.rs for the full
    // pins). Same input, new behavior: 1 is not sequential, so the
    // accept op simply does not match (no throw).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? {:clojure.spec.alpha/op :accept} 1)"),
        "false"
    );
}

#[test]
fn gen_methods_unextended() {
    // `gen*`/`with-gen*` are declared but have no impls yet.
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/even even?)").unwrap();
    let msg = uncaught(
        &mut eval,
        "(clojure.spec.alpha/gen* (s/get-spec :user/even) nil [] {})",
    );
    assert!(
        msg.contains("No implementation of method: :gen*"),
        "unexpected: {msg}"
    );
}
