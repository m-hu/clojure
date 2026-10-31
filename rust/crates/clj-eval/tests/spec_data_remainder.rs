// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-122 slice-D2 tests: the remaining data specs (`s/tuple`,
//! `s/multi-spec`, `s/conformer`, `s/nonconforming`) over the
//! embedded `clojure.spec.alpha`.
//!
//! Oracle: upstream `clojure/spec/alpha.clj` 0.6.249, read as TEXT
//! (sha256 `68925dc5...`, 1996 lines — byte-identical to the 0.5.238
//! file pinned by slice 1). The JVM was NEVER executed for this
//! slice: every expected value below is derived by reading the
//! upstream source (see `docs/gap-122/slice-D2.md` for the line
//! pins). Pins that diverge from the oracle say so.

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

// ---- s/tuple ----

#[test]
fn tuple_conform_valid_basics() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/pair (s/tuple integer? string?))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/conform :user/pair [1 \"a\"])"),
        "[1 \"a\"]"
    );
    assert_eq!(run(&mut eval, "(s/valid? :user/pair [1 \"a\"])"), "true");
    // Non-vectors never conform (oracle: `vector?` + count guard).
    assert_eq!(run(&mut eval, "(s/valid? :user/pair '(\"a\" 1))"), "false");
    assert_eq!(run(&mut eval, "(s/valid? :user/pair {:a 1})"), "false");
    // Wrong count.
    assert_eq!(run(&mut eval, "(s/valid? :user/pair [1])"), "false");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/pair [1 \"a\" :x])"),
        "false"
    );
    // Element failures.
    assert_eq!(run(&mut eval, "(s/valid? :user/pair [1 2])"), "false");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/pair [\"a\" \"b\"])"),
        "false"
    );
    // Unchanged conform returns the identical vector (oracle
    // `identical?` fast path: pred conform returns the value itself).
    assert_eq!(
        run(
            &mut eval,
            "(let [v [1 \"a\"]] (identical? v (s/conform :user/pair v)))"
        ),
        "true"
    );
}

#[test]
fn tuple_conforming_elements_thread() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/conv (s/tuple (s/conformer inc dec) string?))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/conform :user/conv [1 \"a\"])"),
        "[2 \"a\"]"
    );
    assert_eq!(
        run(&mut eval, "(s/unform :user/conv [2 \"a\"])"),
        "[1 \"a\"]"
    );
}

#[test]
fn tuple_unform_asserts_shape() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/pair (s/tuple integer? string?))")
        .unwrap();
    assert_eq!(
        uncaught(&mut eval, "(s/unform :user/pair [1])"),
        "java.lang.AssertionError | Assert failed: (c/and (vector? x) (= (count x) (count (:preds this))))"
    );
}

#[test]
fn tuple_forward_refs_resolve_lazily() {
    // Like `and`: the tuple builds before its ident target exists.
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/late (s/tuple :user/later string?))")
        .unwrap();
    eval.eval_source("(s/def :user/later integer?)").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/late [1 \"a\"])"), "true");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/late [\"a\" \"b\"])"),
        "false"
    );
}

#[test]
fn tuple_zero_preds_asserts() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/tuple)"),
        "java.lang.AssertionError | Assert failed: (not (empty? preds))"
    );
}

#[test]
fn tuple_explain_data_shapes() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/pair (s/tuple integer? string?))")
        .unwrap();
    // Non-vector names `vector?`.
    assert_eq!(
        run(&mut eval, "(s/explain-data :user/pair '(\"a\" 1))"),
        "#:clojure.spec.alpha{:problems [{:path [], :pred clojure.core/vector?, :val (\"a\" 1), :via [:user/pair], :in []}], :spec :user/pair, :value (\"a\" 1)}"
    );
    // Wrong count names the count form.
    assert_eq!(
        run(&mut eval, "(s/explain-data :user/pair [1])"),
        "#:clojure.spec.alpha{:problems [{:path [], :pred (clojure.core/= (clojure.core/count %) 2), :val [1], :via [:user/pair], :in []}], :spec :user/pair, :value [1]}"
    );
    // Element failure carries the ordinal path/in (oracle: the
    // per-element branch is `(apply concat ...)`, so `:problems`
    // prints as a seq here, exactly as upstream).
    assert_eq!(
        run(&mut eval, "(s/explain-data :user/pair [1 2])"),
        "#:clojure.spec.alpha{:problems ({:path [1], :pred clojure.core/string?, :val 2, :via [:user/pair], :in [1]}), :spec :user/pair, :value [1 2]}"
    );
}

#[test]
fn tuple_explain_str_element() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/pair (s/tuple integer? string?))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/pair [1 2])"),
        "\"2 - failed: string? in: [1] at: [1] spec: :user/pair\\n\""
    );
}

#[test]
fn tuple_form_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/pair (s/tuple integer? string?))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/form :user/pair)"),
        "(clojure.spec.alpha/tuple clojure.core/integer? clojure.core/string?)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe :user/pair)"),
        "(tuple integer? string?)"
    );
}

// ---- s/multi-spec ----

/// `(defmulti num-kind ...)` + methods, then the spec (separate
/// batches so the macro's `res` resolves the var at expansion).
fn setup_num_kind(eval: &mut Evaluator) {
    eval.eval_source(
        "(defmulti num-kind (fn [x] (if (integer? x) :int :str)))
         (defmethod num-kind :int [_] integer?)
         (defmethod num-kind :str [_] string?)",
    )
    .unwrap();
    eval.eval_source("(s/def :user/nk (s/multi-spec num-kind :tag))")
        .unwrap();
}

fn setup_mspec(eval: &mut Evaluator) {
    eval.eval_source(
        "(defmulti mspec :tag)
         (defmethod mspec :m [_] map?)",
    )
    .unwrap();
    eval.eval_source("(s/def :user/ms (s/multi-spec mspec :tag))")
        .unwrap();
}

#[test]
fn multi_dispatch_conforms() {
    let mut eval = boot();
    require_spec(&mut eval);
    setup_num_kind(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? :user/nk 5)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/nk \"a\")"), "true");
    // Dispatches `:str`; `string?` fails on 5.5.
    assert_eq!(run(&mut eval, "(s/valid? :user/nk 5.5)"), "false");
    assert_eq!(run(&mut eval, "(s/conform :user/nk 5)"), "5");
}

#[test]
fn multi_keyword_dispatch_over_maps() {
    let mut eval = boot();
    require_spec(&mut eval);
    setup_mspec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? :user/ms {:tag :m})"), "true");
}

#[test]
fn multi_no_method_explains() {
    let mut eval = boot();
    require_spec(&mut eval);
    setup_mspec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? :user/ms {:tag :nope})"), "false");
    assert_eq!(
        run(&mut eval, "(s/explain-data :user/ms {:tag :nope})"),
        "#:clojure.spec.alpha{:problems [{:path [:nope], :pred user/mspec, :val {:tag :nope}, :reason \"no method\", :via [:user/ms], :in []}], :spec :user/ms, :value {:tag :nope}}"
    );
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/ms {:tag :nope})"),
        "\"{:tag :nope} - failed: no method at: [:nope] spec: :user/ms\\n\""
    );
}

#[test]
fn multi_explain_path_carries_dispatch_value() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source(
        "(defmulti tv (fn [x] (if (vector? x) :vec :other)))
         (defmethod tv :vec [_] (s/tuple integer? string?))
         (defmethod tv :other [_] string?)",
    )
    .unwrap();
    eval.eval_source("(s/def :user/tv (s/multi-spec tv :tag))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(let [p (first (:clojure.spec.alpha/problems (s/explain-data :user/tv [1 2])))] [(:path p) (:in p)])"
        ),
        "[[:vec 1] [1]]"
    );
}

#[test]
fn multi_unform_dispatches() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source(
        "(defmulti cf (fn [x] :num))
         (defmethod cf :num [_] (s/conformer inc dec))",
    )
    .unwrap();
    eval.eval_source("(s/def :user/cf (s/multi-spec cf :tag))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/conform :user/cf 5)"), "6");
    assert_eq!(run(&mut eval, "(s/unform :user/cf 6)"), "5");
}

#[test]
fn multi_unform_no_method_throws() {
    // DIVERGENCE (pinned): upstream throws `IllegalStateException`;
    // the port throws `ExceptionInfo` with the same message (still
    // caught by `catch Exception`).
    let mut eval = boot();
    require_spec(&mut eval);
    setup_mspec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/unform :user/ms {:tag :nope})"),
        "clojure.lang.ExceptionInfo | No method of: user/mspec for dispatch value: :nope"
    );
}

#[test]
fn multi_late_method_applies() {
    // The mm var is derefed fresh per operation (upstream).
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(defmulti late-mm :tag)").unwrap();
    eval.eval_source("(s/def :user/lm (s/multi-spec late-mm :tag))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/lm {:tag :a})"), "false");
    eval.eval_source("(defmethod late-mm :a [_] map?)").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/lm {:tag :a})"), "true");
}

#[test]
fn multi_form_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    setup_mspec(&mut eval);
    // Raw form keeps the resolved multimethod symbol; `describe`
    // abbreviates namespaced symbols (oracle `abbrev`).
    assert_eq!(
        run(&mut eval, "(s/form :user/ms)"),
        "(clojure.spec.alpha/multi-spec user/mspec :tag)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe :user/ms)"),
        "(multi-spec mspec :tag)"
    );
}

// ---- s/conformer (+ s/unform) ----

#[test]
fn conformer_conform_and_invalid() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(defn keep-int [x] (if (integer? x) x :clojure.spec.alpha/invalid))")
        .unwrap();
    eval.eval_source("(s/def :user/ki (s/conformer keep-int))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/conform :user/ki 5)"), "5");
    assert_eq!(
        run(&mut eval, "(s/conform :user/ki \"a\")"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(run(&mut eval, "(s/valid? :user/ki 5)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/ki \"a\")"), "false");
}

#[test]
fn conformer_falsy_result_is_valid() {
    // Oracle `conform*`: a conformer returns its result as-is —
    // only `::invalid` fails, so `false` is valid.
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/ff (s/conformer (fn [_] false)))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/conform :user/ff :anything)"), "false");
    assert_eq!(run(&mut eval, "(s/valid? :user/ff :anything)"), "true");
    assert_eq!(run(&mut eval, "(s/explain-data :user/ff :anything)"), "nil");
}

#[test]
fn conformer_unform_reverses_or_throws() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/ud (s/conformer inc dec))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/unform :user/ud 6)"), "5");
    eval.eval_source("(s/def :user/nu (s/conformer inc))")
        .unwrap();
    // DIVERGENCE (pinned): upstream throws `IllegalStateException`;
    // the port throws `ExceptionInfo` with the same message.
    assert_eq!(
        uncaught(&mut eval, "(s/unform :user/nu 6)"),
        "clojure.lang.ExceptionInfo | no unform fn for conformer"
    );
}

#[test]
fn conformer_threads_through_and() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/ac (s/and (s/conformer inc dec) #(< % 10)))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/conform :user/ac 5)"), "6");
    assert_eq!(
        run(&mut eval, "(s/conform :user/ac 50)"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(run(&mut eval, "(s/unform :user/ac 6)"), "5");
}

#[test]
fn conformer_explain_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(defn keep-int [x] (if (integer? x) x :clojure.spec.alpha/invalid))")
        .unwrap();
    eval.eval_source("(s/def :user/ki (s/conformer keep-int))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-data :user/ki \"a\")"),
        "#:clojure.spec.alpha{:problems [{:path [], :pred (clojure.spec.alpha/conformer user/keep-int), :val \"a\", :via [:user/ki], :in []}], :spec :user/ki, :value \"a\"}"
    );
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/ki \"a\")"),
        "\"\\\"a\\\" - failed: (conformer keep-int) spec: :user/ki\\n\""
    );
    eval.eval_source("(s/def :user/ud (s/conformer inc dec))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/form :user/ud)"),
        "(clojure.spec.alpha/conformer clojure.core/inc clojure.core/dec)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe :user/ud)"),
        "(conformer inc dec)"
    );
}

// ---- s/nonconforming ----

#[test]
fn nonconforming_returns_original() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/nc (s/nonconforming (s/conformer inc dec)))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/conform :user/nc 5)"), "5");
    assert_eq!(run(&mut eval, "(s/valid? :user/nc 5)"), "true");
    // Unform is identity.
    assert_eq!(run(&mut eval, "(s/unform :user/nc 6)"), "6");
    // Invalid stays invalid.
    eval.eval_source("(defn keep-int [x] (if (integer? x) x :clojure.spec.alpha/invalid))")
        .unwrap();
    eval.eval_source("(s/def :user/nk2 (s/nonconforming (s/conformer keep-int)))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/conform :user/nk2 \"a\")"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(run(&mut eval, "(s/valid? :user/nk2 \"a\")"), "false");
}

#[test]
fn nonconforming_explain_delegates_and_describes() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(defn keep-int [x] (if (integer? x) x :clojure.spec.alpha/invalid))")
        .unwrap();
    eval.eval_source("(s/def :user/nk2 (s/nonconforming (s/conformer keep-int)))")
        .unwrap();
    // Same problem pred as the inner spec.
    assert_eq!(
        run(
            &mut eval,
            "(let [p (first (:clojure.spec.alpha/problems (s/explain-data :user/nk2 \"a\")))] (:pred p))"
        ),
        "(clojure.spec.alpha/conformer user/keep-int)"
    );
    eval.eval_source("(s/def :user/nc (s/nonconforming (s/conformer inc dec)))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/describe :user/nc)"),
        "(nonconforming (conformer inc dec))"
    );
}

// ---- no s/invalid var ----

#[test]
fn no_invalid_var_by_oracle() {
    // Upstream has no `s/invalid` var (only the `::invalid`
    // keyword + `invalid?`): it must stay absent.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(nil? (resolve 'clojure.spec.alpha/invalid))"),
        "true"
    );
}

// ---- __dispatch_fn* glue native ----

#[test]
fn dispatch_fn_native_contract() {
    // The gap-122 slice-D2 embedding glue: answers the multifn's
    // dispatch fn (upstream `(.dispatchFn mm)`).
    let mut eval = boot();
    eval.eval_source("(defmulti dm :tag)").unwrap();
    assert_eq!(
        run(&mut eval, "((clojure.core/__dispatch_fn* dm) {:tag :a})"),
        ":a"
    );
    assert_eq!(
        uncaught(&mut eval, "(clojure.core/__dispatch_fn*)"),
        "clojure.lang.ArityException | Wrong number of args (0) passed to: clojure.core/__dispatch_fn*"
    );
    assert_eq!(
        uncaught(&mut eval, "(clojure.core/__dispatch_fn* 42)"),
        "java.lang.ClassCastException | class java.lang.Long cannot be cast to class clojure.lang.MultiFn (java.lang.Long is in module java.base of loader 'bootstrap'; clojure.lang.MultiFn is in unnamed module of loader 'app')"
    );
}
