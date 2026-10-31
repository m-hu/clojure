// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-122 slice-B tests: the `clojure.spec.alpha` combinators
//! (`s/or`, `s/nilable`, `s/coll-of`, `s/map-of`, `s/every`, plus
//! the supporting `s/tuple`/`s/every-kv` that `s/map-of` expands
//! through).
//!
//! Oracle: upstream `spec.alpha-0.5.238` `clojure/spec/alpha.clj`,
//! used as a TEXT oracle only (no JVM was executed for this
//! slice; pins are derived by code-reading per
//! `docs/gap-122/slice-B.md`). Upstream line references name the
//! construct each test pins.

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

/// Require spec with the `s` alias.
fn require_spec(eval: &mut Evaluator) {
    eval.eval_source("(require '[clojure.spec.alpha :as s])")
        .unwrap();
}

// ---- s/or (upstream alpha.clj 476-491, 1061-1126) ----

#[test]
fn or_valid_branches() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/oor (s/or :even even? :small #(< % 10)))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/oor 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/oor 5)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/oor 11)"), "false");
}

#[test]
fn or_conform_tagged_first_match() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/or :even even? :small #(< % 10)) 4)"
        ),
        "[:even 4]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/or :even even? :small #(< % 10)) 5)"
        ),
        "[:small 5]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/or :even even? :small #(< % 10)) 11)"
        ),
        ":clojure.spec.alpha/invalid"
    );
}

#[test]
fn or_conform_nested() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/or :o (s/or :e even?)) 2)"),
        "[:o [:e 2]]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/unform (s/or :o (s/or :e even?)) [:o [:e 2]])"
        ),
        "2"
    );
}

#[test]
fn or_unform_and_key_val() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/unform (s/or :even even? :s string?) [:s \"a\"])"
        ),
        "\"a\""
    );
    assert_eq!(
        run(&mut eval, "(key (s/conform (s/or :even even?) 4))"),
        ":even"
    );
    assert_eq!(
        run(&mut eval, "(val (s/conform (s/or :even even?) 4))"),
        "4"
    );
}

#[test]
fn or_three_branches() {
    // The general loop path (upstream has 2-/3-branch fast paths;
    // the port runs one loop for all counts).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/or :s string? :k keyword? :m map?) :a)"
        ),
        "[:k :a]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/or :s string? :k keyword? :m map?) 5)"
        ),
        "false"
    );
}

#[test]
fn or_explain_str_branches() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/or :even even? :small #(< % 10)) 11)"
        ),
        "\"11 - failed: even? at: [:even]\\n11 - failed: (< % 10) at: [:small]\\n\""
    );
}

#[test]
fn or_explain_str_named() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/oor (s/or :even even? :small #(< % 10)))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/oor 11)"),
        "\"11 - failed: even? at: [:even] spec: :user/oor\\n11 - failed: (< % 10) at: [:small] spec: :user/oor\\n\""
    );
}

#[test]
fn or_explain_data_paths() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(:clojure.spec.alpha/problems (s/explain-data (s/or :even even? :neg neg?) 5))"
        ),
        "({:path [:even], :pred clojure.core/even?, :val 5, :via [], :in []} {:path [:neg], :pred clojure.core/neg?, :val 5, :via [], :in []})"
    );
}

#[test]
fn or_forwards_registry_refs() {
    // As with `s/and`: idents resolve lazily via the delay.
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/olate (s/or :s string? :later :user/olater))")
        .unwrap();
    eval.eval_source("(s/def :user/olater even?)").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/olate 4)"), "true");
    assert_eq!(run(&mut eval, "(s/conform :user/olate 4)"), "[:later 4]");
    assert_eq!(run(&mut eval, "(s/valid? :user/olate \"a\")"), "true");
}

#[test]
fn or_macro_asserts() {
    let mut eval = boot();
    require_spec(&mut eval);
    for src in ["(s/or :a)", "(s/or \"a\" even?)"] {
        assert_eq!(
            uncaught(&mut eval, src),
            "java.lang.AssertionError | Assert failed: spec/or expects k1 p1 k2 p2..., where ks are keywords\n(c/and (even? (count key-pred-forms)) (every? keyword? keys))"
        );
    }
}

#[test]
fn or_form_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/form (s/or :even even? :small #(< % 10)))"),
        "(clojure.spec.alpha/or :even clojure.core/even? :small (clojure.core/fn [%] (clojure.core/< % 10)))"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/describe (s/or :even even? :small #(< % 10)))"
        ),
        "(or :even even? :small (< % 10))"
    );
}

// ---- s/nilable (upstream alpha.clj 1834-1864) ----

#[test]
fn nilable_valid_conform() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? (s/nilable even?) nil)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? (s/nilable even?) 4)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? (s/nilable even?) 5)"), "false");
    assert_eq!(run(&mut eval, "(s/conform (s/nilable even?) nil)"), "nil");
    assert_eq!(run(&mut eval, "(s/conform (s/nilable even?) 4)"), "4");
    assert_eq!(
        run(&mut eval, "(s/conform (s/nilable even?) 5)"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(run(&mut eval, "(s/unform (s/nilable even?) nil)"), "nil");
    assert_eq!(run(&mut eval, "(s/unform (s/nilable even?) 4)"), "4");
}

#[test]
fn nilable_explain_str() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/nilable even?) 5)"),
        "\"5 - failed: even? at: [:clojure.spec.alpha/pred]\\n5 - failed: nil? at: [:clojure.spec.alpha/nil]\\n\""
    );
    // `string?` (not `even?`) for the nil case: the port's
    // numeric predicates throw on nil where the JVM answers false.
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/nilable string?) nil)"),
        "\"Success!\\n\""
    );
}

#[test]
fn nilable_form_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/form (s/nilable even?))"),
        "(clojure.spec.alpha/nilable clojure.core/even?)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe (s/nilable even?))"),
        "(nilable even?)"
    );
}

// ---- s/tuple (upstream alpha.clj 694-700, 996-1056) ----

#[test]
fn tuple_valid_conform() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? (s/tuple even? string?) [2 \"a\"])"),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/tuple even? string?) [2])"),
        "false"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/tuple even? string?) '(2 \"a\"))"),
        "false"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/tuple even? string?) [3 \"a\"])"),
        "false"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/tuple (s/or :e even?) string?) [2 \"a\"])"
        ),
        "[[:e 2] \"a\"]"
    );
}

#[test]
fn tuple_explain_str() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/tuple even? string?) [2])"),
        "\"[2] - failed: (= (count %) 2)\\n\""
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/tuple even? string?) '(2 \"a\"))"
        ),
        "\"(2 \\\"a\\\") - failed: vector?\\n\""
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/tuple even? string?) [3 \"a\"])"
        ),
        "\"3 - failed: even? in: [0] at: [0]\\n\""
    );
}

#[test]
fn tuple_unform() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/unform (s/tuple (s/or :e even?) string?) [[:e 2] \"a\"])"
        ),
        "[2 \"a\"]"
    );
    assert_eq!(
        uncaught(&mut eval, "(s/unform (s/tuple even?) [1 2])"),
        "java.lang.AssertionError | Assert failed: (c/and (vector? x) (= (count x) (count (:preds this))))"
    );
}

#[test]
fn tuple_macro_assert_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/tuple)"),
        "java.lang.AssertionError | Assert failed: (not (empty? preds))"
    );
    assert_eq!(
        run(&mut eval, "(s/form (s/tuple even? string?))"),
        "(clojure.spec.alpha/tuple clojure.core/even? clojure.core/string?)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe (s/tuple even? string?))"),
        "(tuple even? string?)"
    );
}

// ---- s/coll-of (upstream alpha.clj 579-590 via every-impl 1245-1372) ----

#[test]
fn coll_of_valid() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? (s/coll-of even?) [2 4])"), "true");
    assert_eq!(
        run(&mut eval, "(s/valid? (s/coll-of even?) [2 3])"),
        "false"
    );
    assert_eq!(run(&mut eval, "(s/valid? (s/coll-of even?) [])"), "true");
    assert_eq!(run(&mut eval, "(s/valid? (s/coll-of even?) 5)"), "false");
    assert_eq!(
        run(&mut eval, "(s/valid? (s/coll-of even?) #{2 4})"),
        "true"
    );
}

#[test]
fn coll_of_conform() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/coll-of even?) [2 4])"),
        "[2 4]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/coll-of (s/or :e even? :o odd?)) [2 3])"
        ),
        "[[:e 2] [:o 3]]"
    );
    assert_eq!(
        run(&mut eval, "(s/conform (s/coll-of even?) [2 3])"),
        ":clojure.spec.alpha/invalid"
    );
}

#[test]
fn coll_of_into_and_shape() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/coll-of even? :into []) '(2 4))"),
        "[2 4]"
    );
    // Default: match the input collection shape.
    assert_eq!(
        run(&mut eval, "(s/conform (s/coll-of even?) '(2 4))"),
        "(2 4)"
    );
}

#[test]
fn coll_of_kind() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/coll-of even? :kind vector?) [2 4])"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/coll-of even? :kind vector?) '(2 4))"
        ),
        "false"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/coll-of even? :kind vector?) '(2 4))"
        ),
        "\"(2 4) - failed: vector?\\n\""
    );
}

#[test]
fn coll_of_count() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? (s/coll-of even? :count 2) [2 4])"),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/coll-of even? :count 2) [2 4 6])"),
        "false"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/coll-of even? :count 2) [2 4 6])"
        ),
        "\"[2 4 6] - failed: (= 2 (count %))\\n\""
    );
}

#[test]
fn coll_of_min_max_count() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/coll-of even? :min-count 1 :max-count 2) [2 4])"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/coll-of even? :min-count 1 :max-count 2) [2 4 6])"
        ),
        "\"[2 4 6] - failed: (<= 1 (count %) 2)\\n\""
    );
    // Min-only: the open upper bound prints abbreviated, as upstream.
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/coll-of even? :min-count 2) [2])"
        ),
        "\"[2] - failed: (<= 2 (count %) MAX_VALUE)\\n\""
    );
}

#[test]
fn coll_of_distinct() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/coll-of even? :distinct true) [2 4])"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/coll-of even? :distinct true) [2 2])"
        ),
        "false"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/coll-of even? :distinct true) [2 2])"
        ),
        "\"[2 2] - failed: distinct?\\n\""
    );
}

#[test]
fn coll_of_explain_element() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/ce (s/coll-of even?))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/ce [2 3 4])"),
        "\"3 - failed: even? in: [1] spec: :user/ce\\n\""
    );
    // Ident element preds extend :via (spec prints the last entry).
    eval.eval_source("(s/def :user/even even?)").unwrap();
    eval.eval_source("(s/def :user/cee (s/coll-of :user/even))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/cee [3])"),
        "\"3 - failed: even? in: [0] spec: :user/even\\n\""
    );
}

#[test]
fn coll_of_form_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/form (s/coll-of even? :kind vector?))"),
        "(clojure.spec.alpha/coll-of clojure.core/even? :kind clojure.core/vector?)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe (s/coll-of even? :kind vector?))"),
        "(coll-of even? :kind vector?)"
    );
}

// ---- s/every (upstream alpha.clj 520-566, 1245-1372) ----

#[test]
fn every_valid_basic() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? (s/every even?) [2 4])"), "true");
    assert_eq!(run(&mut eval, "(s/valid? (s/every even?) [2 3])"), "false");
    assert_eq!(run(&mut eval, "(s/valid? (s/every even?) '(2 4))"), "true");
    assert_eq!(run(&mut eval, "(s/valid? (s/every even?) '(2 3))"), "false");
    assert_eq!(run(&mut eval, "(s/valid? (s/every even?) #{2 4})"), "true");
    assert_eq!(
        run(&mut eval, "(s/valid? (s/every even? :count 2) [2 4])"),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/every even? :count 2) [2 4 6])"),
        "false"
    );
}

#[test]
fn every_does_not_conform() {
    // Unlike coll-of, every returns the input unchanged.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/every (s/or :e even? :o odd?)) [2 3])"
        ),
        "[2 3]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/unform (s/every (s/or :e even? :o odd?)) [2 3])"
        ),
        "[2 3]"
    );
}

#[test]
fn every_sampling_limit() {
    let mut eval = boot();
    require_spec(&mut eval);
    // A bad element past the sample still validates (indexed step).
    assert_eq!(
        run(
            &mut eval,
            "(binding [clojure.spec.alpha/*coll-check-limit* 1] (s/valid? (s/every even?) [2 3]))"
        ),
        "true"
    );
    assert_eq!(run(&mut eval, "(s/valid? (s/every even?) [2 3])"), "false");
    // Same for the sequential (non-indexed) walk.
    assert_eq!(
        run(
            &mut eval,
            "(binding [clojure.spec.alpha/*coll-check-limit* 1] (s/valid? (s/every even?) '(2 3)))"
        ),
        "true"
    );
}

#[test]
fn every_error_limit() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(count (:clojure.spec.alpha/problems (s/explain-data (s/every even?) [1 3])))"
        ),
        "2"
    );
    assert_eq!(
        run(
            &mut eval,
            "(binding [clojure.spec.alpha/*coll-error-limit* 1] (count (:clojure.spec.alpha/problems (s/explain-data (s/every even?) [1 3]))))"
        ),
        "1"
    );
}

#[test]
fn every_form_and_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/form (s/every even?))"),
        "(clojure.spec.alpha/every clojure.core/even?)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe (s/every even?))"),
        "(every even?)"
    );
    // Multi-opt describes keep kwargs order, as upstream.
    assert_eq!(
        run(
            &mut eval,
            "(s/describe (s/every even? :count 2 :distinct true))"
        ),
        "(every even? :count 2 :distinct true)"
    );
}

// ---- s/map-of / s/every-kv (upstream alpha.clj 568-604) ----

#[test]
fn map_of_valid() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? (s/map-of keyword? even?) {:a 2})"),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/map-of keyword? even?) {:a 3})"),
        "false"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/map-of keyword? even?) {\"a\" 2})"),
        "false"
    );
    // :kind defaults to map?.
    assert_eq!(
        run(&mut eval, "(s/valid? (s/map-of keyword? even?) [[:a 2]])"),
        "false"
    );
    // An explicit :kind wins over the default.
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/map-of string? even? :kind vector?) [[\"a\" 2]])"
        ),
        "true"
    );
}

#[test]
fn map_of_conform() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/map-of keyword? (s/or :e even? :o odd?)) {:a 3})"
        ),
        "{:a [:o 3]}"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/unform (s/map-of keyword? (s/or :e even? :o odd?)) {:a [:o 3]})"
        ),
        "{:a 3}"
    );
    // Keys conform only with :conform-keys.
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/map-of (s/or :k keyword?) even? :conform-keys true) {:a 2})"
        ),
        "{[:k :a] 2}"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/map-of (s/or :k keyword?) even?) {:a 2})"
        ),
        "{:a 2}"
    );
}

#[test]
fn map_of_explain_str() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/mm (s/map-of keyword? even?))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/mm {:a 3})"),
        "\"3 - failed: even? in: [:a 1] at: [1] spec: :user/mm\\n\""
    );
}

#[test]
fn map_of_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/describe (s/map-of keyword? even?))"),
        "(map-of keyword? even?)"
    );
}

#[test]
fn every_kv_valid_no_conform() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? (s/every-kv keyword? even?) {:a 2})"),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/every-kv keyword? even?) {\"a\" 2})"
        ),
        "false"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/every-kv keyword? (s/or :e even?)) {:a 2})"
        ),
        "{:a 2}"
    );
}
