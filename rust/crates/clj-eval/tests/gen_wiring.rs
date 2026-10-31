// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-122 slice-F2 tests: `gen*`/`with-gen*` wiring (`s/gen`,
//! `s/with-gen`, `s/exercise`, `s/exercise-fn`) plus the three
//! F1-deferred gen fns the wiring needs (`gen/delay`,
//! `gen/hash-map`, `gen/vector-distinct`).
//!
//! Oracles (text only, never executed): upstream
//! `spec.alpha-0.6.249` `clojure/spec/alpha.clj` (1996 lines) for the
//! `gen*` arms, `gensub`/`gen`/`with-gen`/`exercise`, and the macro
//! `:gen` kwargs; `test.check-1.1.3` `generators.cljc` for
//! `hash-map`/`vector-distinct` shapes and assert texts; upstream
//! `gen/alpha.clj` for `delay-impl`/`delay`. Seed streams
//! intentionally do NOT match the JVM (slice-F1 divergence 1): these
//! tests pin distribution/shape properties and determinism (same
//! seed ⇒ same value), never JVM stream equality. Every generated
//! value is also checked with `s/valid?` against its spec.

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

/// Require spec (as `s`) and gen (as `gen`).
fn require_spec_gen(eval: &mut Evaluator) {
    eval.eval_source("(require '[clojure.spec.alpha :as s] '[clojure.spec.gen.alpha :as gen])")
        .unwrap();
}

/// Assert an uncaught throw whose class and message contain the parts
/// (pins the oracle text without overfitting the port's assert wrapper).
fn assert_throws(eval: &mut Evaluator, src: &str, class_part: &str, msg_part: &str) {
    let got = uncaught(eval, src);
    assert!(
        got.contains(class_part),
        "class mismatch for {src}: got {got}"
    );
    assert!(
        got.contains(msg_part),
        "message mismatch for {src}: got {got}"
    );
}

// ---- F2 gen additions: delay / hash-map / vector-distinct ----

#[test]
fn f2_gen_publics_resolve() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? some? (map resolve '[gen/delay gen/delay-impl gen/hash-map gen/vector-distinct]))"
        ),
        "true"
    );
}

#[test]
fn delay_defers_body_until_generate() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(def calls (atom 0))
             (def g (gen/delay (swap! calls inc) (gen/return 7)))
             [@calls (gen/generate g 30 1) @calls]"
        ),
        "[0 7 1]"
    );
}

#[test]
fn delay_recursive_grammar_terminates() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    // Leaf-biased recursive grammar: must realize without hanging.
    assert_eq!(
        run(
            &mut eval,
            "(def tree-gen (gen/frequency [[9 (gen/return :leaf)]
                                            [1 (gen/delay (gen/fmap (fn [t] [:node t]) tree-gen))]]))
             (every? (fn [s] (let [v (gen/generate tree-gen 30 s)]
                               (or (= v :leaf) (= :node (first v)))))
                      (range 25))"
        ),
        "true"
    );
}

#[test]
fn hash_map_draws_maps() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (gen/generate (gen/hash-map :a (gen/return 1) :b (gen/return 2)) 30 1)
                {:a 1 :b 2})"
        ),
        "true"
    );
    assert_eq!(run(&mut eval, "(gen/generate (gen/hash-map) 30 1)"), "{}");
}

#[test]
fn hash_map_assert_texts() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/hash-map :a (gen/return 1) :b)",
        "AssertionError",
        "Assert failed",
    );
    assert_throws(
        &mut eval,
        "(gen/hash-map :a 42)",
        "AssertionError",
        "Value args to hash-map must be generators",
    );
}

#[test]
fn vector_distinct_fixed_count() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(def g (gen/vector-distinct (gen/choose 0 9) {:num-elements 5}))
             (every? (fn [s] (let [v (gen/generate g 30 s)]
                               (and (= 5 (count v))
                                    (apply distinct? v)
                                    (every? #(<= 0 % 9) v))))
                      (range 50))"
        ),
        "true"
    );
}

#[test]
fn vector_distinct_sized_range() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(def g (gen/vector-distinct (gen/choose 0 9) {:min-elements 2 :max-elements 4}))
             (every? (fn [s] (let [v (gen/generate g 30 s)]
                               (and (<= 2 (count v) 4)
                                    (apply distinct? v))))
                      (range 50))"
        ),
        "true"
    );
}

#[test]
fn vector_distinct_impossible_throws_oracle_text() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/generate (gen/vector-distinct (gen/return 1) {:num-elements 2}) 30 1)",
        "ExceptionInfo",
        "Couldn't generate enough distinct elements!",
    );
    assert_throws(
        &mut eval,
        "(gen/vector-distinct 42)",
        "AssertionError",
        "First arg to vector-distinct must be a generator!",
    );
}

// ---- s/gen over predicate and s/and specs ----

#[test]
fn gen_pred_spec_draws_valid() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/n int?)").unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? int? (map #(gen/generate (s/gen :user/n) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/s #{1 2 3})").unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #{1 2 3} (map #(gen/generate (s/gen :user/s) 30 %) (range 25)))"
        ),
        "true"
    );
}

#[test]
fn gen_and_spec_draws_valid() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/a (s/and int? pos?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(s/valid? :user/a %) (map #(gen/generate (s/gen :user/a) 30 %) (range 25)))"
        ),
        "true"
    );
}

#[test]
fn gen_is_deterministic_per_seed() {
    let mut eval1 = boot();
    require_spec_gen(&mut eval1);
    let v1 = run(
        &mut eval1,
        "(s/def :user/n int?) (gen/generate (s/gen :user/n) 30 7)",
    );
    let mut eval2 = boot();
    require_spec_gen(&mut eval2);
    let v2 = run(
        &mut eval2,
        "(s/def :user/n int?) (gen/generate (s/gen :user/n) 30 7)",
    );
    assert_eq!(v1, v2);
}

#[test]
fn with_gen_overrides_draws() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/w (s/with-gen int? (fn [] (gen/return 7))))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= 7 %) (map #(gen/generate (s/gen :user/w) 30 %) (range 10)))"
        ),
        "true"
    );
    // Conform behavior is unaffected by the generator override.
    assert_eq!(run(&mut eval, "(s/valid? :user/w 7)"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/w :x)"), "false");
}

#[test]
fn spec_gen_kwarg_overrides_draws() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/g (s/spec int? :gen (fn [] (gen/return 8))))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= 8 %) (map #(gen/generate (s/gen :user/g) 30 %) (range 10)))"
        ),
        "true"
    );
}

#[test]
fn exercise_returns_valid_pairs() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/n int?)").unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(let [xs (s/exercise :user/n 5)]
               (and (= 5 (count xs))
                    (every? (fn [[v c]] (and (s/valid? :user/n v) (= v c))) xs)))"
        ),
        "true"
    );
    // Default arity generates 10 pairs.
    assert_eq!(run(&mut eval, "(count (s/exercise :user/n))"), "10");
}

#[test]
fn gen_missing_generator_throws_oracle() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(defn fresh-pred-xyz [x] (int? x))")
        .unwrap();
    eval.eval_source("(s/def :user/ng fresh-pred-xyz)").unwrap();
    assert_throws(
        &mut eval,
        "(s/gen :user/ng)",
        "ExceptionInfo",
        "Unable to construct gen at: []",
    );
    assert_eq!(
        run(
            &mut eval,
            "(try (s/gen :user/ng) :no-throw
               (catch Exception e (:clojure.spec.alpha/failure (ex-data e))))"
        ),
        ":no-gen"
    );
    assert_throws(
        &mut eval,
        "(s/gen :nope/missing)",
        "ExceptionInfo",
        "Unable to resolve spec: :nope/missing",
    );
}

// ---- s/or, s/nilable, s/tuple, s/every family ----

#[test]
fn gen_or_spec_draws_valid() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/o (s/or :i int? :s string?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(s/valid? :user/o %) (map #(gen/generate (s/gen :user/o) 30 %) (range 25)))"
        ),
        "true"
    );
    // Both branches realize across draws.
    assert_eq!(
        run(
            &mut eval,
            "(let [vs (map #(gen/generate (s/gen :user/o) 30 %) (range 50))]
               [(boolean (some int? vs)) (boolean (some string? vs))])"
        ),
        "[true true]"
    );
}

#[test]
fn gen_recursion_limit_default_is_four() {
    // Oracle alpha.clj:26-30. Pinned as a value (drawing at the
    // default depth can exceed the port's call-stack budget — see
    // gen_or_recursive_terminates).
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_eq!(run(&mut eval, "s/*recursion-limit*"), "4");
}

#[test]
fn gen_or_recursive_terminates() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/tree (s/or :leaf int? :node (s/tuple keyword? :user/tree)))")
        .unwrap();
    // The limit binds to a small depth: each spec nesting level costs
    // ~30 interpreter calls and the port overflows past ~100-200
    // nested calls (pre-existing evaluator characteristic — plain
    // `(down 200)` overflows too), so depth-4 default-limit draws can
    // abort. Binding the limit exercises the same termination
    // mechanism within budget, robust to RNG-stream changes.
    assert_eq!(
        run(
            &mut eval,
            "(binding [s/*recursion-limit* 1]
               (let [vs (map #(gen/generate (s/gen :user/tree) 30 %) (range 10))]
                 [(every? #(s/valid? :user/tree %) vs)
                  (boolean (some vector? vs))]))"
        ),
        "[true true]"
    );
}

#[test]
fn gen_nilable_draws_nil_and_values() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/nl (s/nilable int?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(let [vs (map #(gen/generate (s/gen :user/nl) 30 %) (range 200))]
               [(boolean (some nil? vs))
                (every? #(s/valid? :user/nl %) vs)])"
        ),
        "[true true]"
    );
}

#[test]
fn gen_tuple_draws_shaped_valid() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/t (s/tuple int? string?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [v] (and (vector? v) (= 2 (count v))
                                   (int? (nth v 0)) (string? (nth v 1))
                                   (s/valid? :user/t v)))
                      (map #(gen/generate (s/gen :user/t) 30 %) (range 25)))"
        ),
        "true"
    );
}

#[test]
fn gen_every_family_honors_opts() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/c (s/coll-of int?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [v] (and (s/valid? :user/c v) (<= (count v) 20)))
                      (map #(gen/generate (s/gen :user/c) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/m (s/map-of keyword? int?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(s/valid? :user/m %) (map #(gen/generate (s/gen :user/m) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/e3 (s/every int? :count 3))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= 3 (count %)) (map #(gen/generate (s/gen :user/e3) 30 %) (range 25)))"
        ),
        "true"
    );
    // :kind without :into resolves the into-coll from the kind form.
    eval.eval_source("(s/def :user/ev (s/every int? :kind vector?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (vector? %) (s/valid? :user/ev %))
                      (map #(gen/generate (s/gen :user/ev) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/d (s/coll-of int? :distinct true :min-count 2 :max-count 4))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [v] (and (<= 2 (count v) 4) (apply distinct? v)
                                   (s/valid? :user/d v)))
                      (map #(gen/generate (s/gen :user/d) 30 %) (range 25)))"
        ),
        "true"
    );
    // :gen on s/every.
    eval.eval_source("(s/def :user/eg (s/every int? :gen (fn [] (gen/return [1 2]))))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= [1 2] %) (map #(gen/generate (s/gen :user/eg) 30 %) (range 10)))"
        ),
        "true"
    );
}

#[test]
fn gen_name_and_path_overrides() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/po2 int?)").unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(gen/generate (s/gen :user/po2 {:user/po2 (fn [] (gen/return 42))}) 30 1)"
        ),
        "42"
    );
    // Path override reaches the tuple element at [1].
    eval.eval_source("(s/def :user/po (s/tuple int? int?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(nth (gen/generate (s/gen :user/po {[1] (fn [] (gen/return 99))}) 30 1) 1)"
        ),
        "99"
    );
}

// ---- s/keys, s/merge ----

#[test]
fn gen_keys_draws_valid_maps() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/a int?) (s/def :user/b string?)")
        .unwrap();
    eval.eval_source("(s/def :user/m (s/keys :req [:user/a] :opt [:user/b]))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(let [vs (map #(gen/generate (s/gen :user/m) 30 %) (range 25))]
               [(every? #(s/valid? :user/m %) vs)
                (every? #(contains? % :user/a) vs)
                (boolean (some #(contains? % :user/b) vs))])"
        ),
        "[true true true]"
    );
}

#[test]
fn gen_keys_un_draws_valid_maps() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/x int?)").unwrap();
    eval.eval_source("(s/def :user/mu (s/keys :req-un [:user/x]))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [m] (and (s/valid? :user/mu m)
                                   (contains? m :x)
                                   (int? (:x m))))
                      (map #(gen/generate (s/gen :user/mu) 30 %) (range 25)))"
        ),
        "true"
    );
}

#[test]
fn gen_keys_gen_kwarg_overrides() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/a int?)").unwrap();
    eval.eval_source(
        "(s/def :user/mg (s/keys :req [:user/a] :gen (fn [] (gen/return {:user/a 1}))))",
    )
    .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= {:user/a 1} %) (map #(gen/generate (s/gen :user/mg) 30 %) (range 10)))"
        ),
        "true"
    );
}

#[test]
fn gen_merge_draws_valid() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/a int?) (s/def :user/b string?)")
        .unwrap();
    eval.eval_source("(s/def :user/m1 (s/keys :req [:user/a]))")
        .unwrap();
    eval.eval_source("(s/def :user/m2 (s/keys :req [:user/b]))")
        .unwrap();
    eval.eval_source("(s/def :user/mm (s/merge :user/m1 :user/m2))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [m] (and (s/valid? :user/mm m)
                                   (int? (:user/a m))
                                   (string? (:user/b m))))
                      (map #(gen/generate (s/gen :user/mm) 30 %) (range 25)))"
        ),
        "true"
    );
}

// ---- s/multi-spec, s/nonconforming ----

#[test]
fn gen_multi_spec_draws_retagged_valid() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/tag keyword?) (s/def :user/i int?) (s/def :user/s string?)")
        .unwrap();
    eval.eval_source(
        "(defmulti mspec-g :tag)
         (defmethod mspec-g :int [_] (s/keys :req [:user/tag :user/i]))
         (defmethod mspec-g :str [_] (s/keys :req [:user/tag :user/s]))",
    )
    .unwrap();
    eval.eval_source("(s/def :user/mg (s/multi-spec mspec-g :tag))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [m] (and (s/valid? :user/mg m)
                                   (contains? #{:int :str} (:tag m))
                                   (if (= :int (:tag m))
                                     (int? (:user/i m))
                                     (string? (:user/s m)))))
                      (map #(gen/generate (s/gen :user/mg) 30 %) (range 10)))"
        ),
        "true"
    );
}

#[test]
fn gen_nonconforming_delegates_to_inner() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/nc (s/nonconforming (s/tuple int? string?)))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [v] (and (vector? v)
                                   (int? (nth v 0))
                                   (string? (nth v 1))))
                      (map #(gen/generate (s/gen :user/nc) 30 %) (range 25)))"
        ),
        "true"
    );
    // with-gen threads through the wrapper.
    eval.eval_source(
        "(s/def :user/ncw (s/with-gen (s/nonconforming int?) (fn [] (gen/return 11))))",
    )
    .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= 11 %) (map #(gen/generate (s/gen :user/ncw) 30 %) (range 10)))"
        ),
        "true"
    );
}

// ---- regex gen* (re-gen), exercise-fn, keys* ----

#[test]
fn gen_regex_cat_draws_valid_seqs() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/c (s/cat :a int? :b string?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [v] (and (sequential? v)
                                   (= 2 (count v))
                                   (s/valid? :user/c v)))
                      (map #(gen/generate (s/gen :user/c) 30 %) (range 25)))"
        ),
        "true"
    );
}

#[test]
fn gen_regex_rep_alt_maybe_amp_draws_valid() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/r (s/* int?))").unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(s/valid? :user/r %) (map #(gen/generate (s/gen :user/r) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/p (s/+ int?))").unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [v] (and (seq v) (s/valid? :user/p v)))
                      (map #(gen/generate (s/gen :user/p) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/q (s/? int?))").unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? (fn [v] (and (<= (count v) 1) (s/valid? :user/q v)))
                      (map #(gen/generate (s/gen :user/q) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/al (s/alt :i int? :s string?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(s/valid? :user/al %) (map #(gen/generate (s/gen :user/al) 30 %) (range 25)))"
        ),
        "true"
    );
    eval.eval_source("(s/def :user/am (s/& (s/* int?) vector?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(s/valid? :user/am %) (map #(gen/generate (s/gen :user/am) 30 %) (range 25)))"
        ),
        "true"
    );
}

#[test]
fn gen_regex_with_gen_on_op() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/wr (s/with-gen (s/cat :a int?) (fn [] (gen/return [7]))))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= [7] %) (map #(gen/generate (s/gen :user/wr) 30 %) (range 10)))"
        ),
        "true"
    );
}

#[test]
fn gen_regex_path_override_wraps_vector() {
    // Override on a :cat branch path hits re-gen's `(:accept nil)`
    // arm: the override generator is fmap-vectored.
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/po3 (s/cat :a int? :b int?))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(first (gen/generate (s/gen :user/po3 {[:a] (fn [] (gen/return 5))}) 30 1))"
        ),
        "5"
    );
}

#[test]
fn exercise_fn_over_args_map() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(let [xs (s/exercise-fn '+ 5 {:args (s/cat :x int? :y int?)})]
               (and (= 5 (count xs))
                    (every? (fn [[[x y] r]] (= r (+ x y))) xs)))"
        ),
        "true"
    );
    assert_throws(
        &mut eval,
        "(s/exercise-fn '+ 5 {})",
        "ExceptionInfo",
        "No :args spec found, can't generate",
    );
}

#[test]
fn gen_keys_star_runtime() {
    let mut eval = boot();
    require_spec_gen(&mut eval);
    eval.eval_source("(s/def :user/a int?) (s/def :user/b string?)")
        .unwrap();
    eval.eval_source("(s/def :user/ks (s/keys* :req-un [:user/a :user/b]))")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(every? #(s/valid? :user/ks %) (map #(gen/generate (s/gen :user/ks) 30 %) (range 10)))"
        ),
        "true"
    );
}
