// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-122 slice-F1 tests: the `clojure.spec.gen.alpha` narrow
//! surface (combinators, scalar prims, `gen-for-pred` builtins,
//! `gen-for-name`) over the seeded splitmix64 RNG.
//!
//! Oracles (text only, never executed): upstream
//! `spec.alpha-0.6.249` `clojure/spec/gen/alpha.clj` (227 lines) and
//! `test.check-1.1.3` `generators.cljc` for combinator shapes, default
//! sizes, and assert/exception texts (listed in
//! `docs/gap-122/slice-F1.md`). Seed streams intentionally do NOT
//! match the JVM (documented divergence 1): these tests pin
//! distribution/shape properties and determinism (same seed ⇒ same
//! value), never JVM stream equality. Values pinned exactly are
//! forced by construction (`return`, degenerate `choose`, fixed
//! arities), never raw RNG output.

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

/// Require gen with the `gen` alias (as the oracle does).
fn require_gen(eval: &mut Evaluator) {
    eval.eval_source("(require '[clojure.spec.gen.alpha :as gen])")
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

// ---- load ----

#[test]
fn gen_alpha_loads_via_require() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(&mut eval, "(some? (find-ns 'clojure.spec.gen.alpha))"),
        "true"
    );
}

#[test]
fn gen_alpha_marks_loaded() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(contains? (loaded-libs) 'clojure.spec.gen.alpha)"
        ),
        "true"
    );
}

#[test]
fn gen_alpha_publics_resolve() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? some? (map resolve '[gen/return gen/bind gen/fmap gen/one-of
               gen/such-that gen/elements gen/choose gen/tuple gen/vector gen/map
               gen/set gen/list gen/shuffle gen/frequency gen/not-empty gen/sized
               gen/resize gen/cat gen/sample gen/generate gen/gen-for-pred
               gen/gen-for-name gen/any gen/any-printable gen/boolean gen/bytes
               gen/char gen/char-alpha gen/char-alphanumeric gen/char-ascii
               gen/double gen/double* gen/int gen/keyword gen/keyword-ns
               gen/large-integer gen/large-integer* gen/ratio gen/simple-type
               gen/simple-type-printable gen/string gen/string-ascii
               gen/string-alphanumeric gen/symbol gen/symbol-ns gen/uuid]))"
        ),
        "true"
    );
}

#[test]
fn gen_alpha_generator_pred_stays_private() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(:private (meta #'clojure.spec.gen.alpha/generator?))"
        ),
        "true"
    );
}

#[test]
fn gen_alpha_does_not_pull_spec() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(&mut eval, "(some? (find-ns 'clojure.spec.alpha))"),
        "false"
    );
}

// ---- return / fmap / bind / sized / resize ----

#[test]
fn return_generates_constantly() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(run(&mut eval, "(gen/generate (gen/return 42))"), "42");
    assert_eq!(run(&mut eval, "(gen/generate (gen/return 42) 30 7)"), "42");
    assert_eq!(
        run(&mut eval, "(gen/sample (gen/return 42))"),
        "(42 42 42 42 42 42 42 42 42 42)"
    );
}

#[test]
fn fmap_bind_thread_values() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(&mut eval, "(gen/generate (gen/fmap inc (gen/return 1)))"),
        "2"
    );
    assert_eq!(
        run(
            &mut eval,
            "(gen/generate (gen/bind (gen/return 1) (fn [x] (gen/return (inc x)))))"
        ),
        "2"
    );
}

#[test]
fn fmap_bind_reject_non_generators() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/fmap inc 5)",
        "AssertionError",
        "Second arg to fmap must be a generator",
    );
    assert_throws(
        &mut eval,
        "(gen/bind 5 (fn [x] (gen/return x)))",
        "AssertionError",
        "First arg to bind must be a generator",
    );
}

#[test]
fn sized_resize_thread_size_exactly() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(gen/generate (gen/sized (fn [s] (gen/return s))) 30 1)"
        ),
        "30"
    );
    assert_eq!(
        run(
            &mut eval,
            "(gen/generate (gen/resize 7 (gen/sized (fn [s] (gen/return s)))) 30 1)"
        ),
        "7"
    );
    assert_throws(
        &mut eval,
        "(gen/resize 7 8)",
        "AssertionError",
        "Second arg to resize must be a generator",
    );
}

// ---- sample / generate / determinism ----

#[test]
fn sample_defaults_and_counts() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(&mut eval, "(count (gen/sample (gen/choose 0 1)))"),
        "10"
    );
    assert_eq!(
        run(&mut eval, "(count (gen/sample (gen/choose 0 1) 25))"),
        "25"
    );
    assert_throws(
        &mut eval,
        "(gen/sample 5)",
        "AssertionError",
        "First arg to sample must be a generator",
    );
}

#[test]
fn sample_size_cycle_starts_at_zero() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(&mut eval, "(first (gen/sample (gen/vector (gen/boolean))))"),
        "[]"
    );
}

#[test]
fn generate_is_deterministic_per_seed() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (gen/generate (gen/vector (gen/choose 0 100)) 30 11)
                (gen/generate (gen/vector (gen/choose 0 100)) 30 11))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(= (gen/generate (gen/choose 0 1000000) 30 1)
                (gen/generate (gen/choose 0 1000000) 30 2))"
        ),
        "false"
    );
}

#[test]
fn generate_is_deterministic_across_evaluators() {
    let mut a = boot();
    require_gen(&mut a);
    let mut b = boot();
    require_gen(&mut b);
    let expr = "(gen/generate (gen/map (gen/choose 0 9) (gen/vector (gen/boolean))) 30 77)";
    assert_eq!(run(&mut a, expr), run(&mut b, expr));
}

// ---- choose ----

#[test]
fn choose_stays_in_range() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (integer? %) (<= -5 % 7))
               (mapv #(gen/generate (gen/choose -5 7) 30 %) (range 100)))"
        ),
        "true"
    );
}

#[test]
fn choose_degenerate_is_constant() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= 5 %) (mapv #(gen/generate (gen/choose 5 5) 30 %) (range 10)))"
        ),
        "true"
    );
}

#[test]
fn choose_full_long_range_stays_in_range() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (integer? %) (<= -9223372036854775808 % 9223372036854775807))
               (mapv #(gen/generate (gen/choose -9223372036854775808 9223372036854775807) 30 %)
                     (range 20)))"
        ),
        "true"
    );
}

#[test]
fn choose_casts_double_bounds_to_long() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(&mut eval, "(gen/generate (gen/choose 1.9 1.9) 30 3)"),
        "1"
    );
}

#[test]
fn choose_inverted_bounds_assert() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/generate (gen/choose 5 1) 30 1)",
        "AssertionError",
        "Assert failed",
    );
}

// ---- elements / one-of / frequency ----

#[test]
fn elements_draws_members() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #{:a :b :c}
               (mapv #(gen/generate (gen/elements [:a :b :c]) 30 %) (range 50)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= :only %) (mapv #(gen/generate (gen/elements [:only]) 30 %) (range 5)))"
        ),
        "true"
    );
}

#[test]
fn elements_rejects_empty() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/elements [])",
        "AssertionError",
        "elements cannot be called with an empty collection",
    );
    assert_throws(
        &mut eval,
        "(gen/elements nil)",
        "AssertionError",
        "elements cannot be called with an empty collection",
    );
}

#[test]
fn one_of_hits_every_arm() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= #{:a :b} (set (mapv #(gen/generate (gen/one-of [(gen/return :a) (gen/return :b)]) 30 %)
                                    (range 100))))"
        ),
        "true"
    );
}

#[test]
fn one_of_rejects_bad_collections() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/one-of [(gen/return 1) 2])",
        "AssertionError",
        "Arg to one-of must be a collection of generators",
    );
    assert_throws(
        &mut eval,
        "(gen/one-of [])",
        "AssertionError",
        "one-of cannot be called with an empty collection",
    );
}

#[test]
fn frequency_hits_every_arm() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= #{:a :b} (set (mapv #(gen/generate (gen/frequency [[1 (gen/return :a)] [1 (gen/return :b)]]) 30 %)
                                    (range 100))))"
        ),
        "true"
    );
}

#[test]
fn frequency_respects_weights() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(let [draws (mapv #(gen/generate (gen/frequency [[3 (gen/return :a)] [1 (gen/return :b)]]) 30 %)
                               (range 2000))
                   n (count (filter #(= :b %) draws))]
               (<= 350 n 650))"
        ),
        "true"
    );
}

#[test]
fn frequency_rejects_bad_pairs() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/frequency [[1 (gen/return 1)] [:x (gen/return 2)]])",
        "AssertionError",
        "Arg to frequency must be a list of [num generator] pairs",
    );
    assert_throws(
        &mut eval,
        "(gen/frequency [[0 (gen/return 1)] [-2 (gen/return 2)]])",
        "AssertionError",
        "frequency must be called with at least one non-zero weight",
    );
}

// ---- such-that / not-empty ----

#[test]
fn such_that_satisfies_predicate() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? odd? (mapv #(gen/generate (gen/such-that odd? (gen/choose 0 100)) 30 %)
                                (range 100)))"
        ),
        "true"
    );
}

#[test]
fn such_that_accepts_int_and_map_opts() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(odd? (gen/generate (gen/such-that odd? (gen/choose 0 100) 50) 30 1))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(odd? (gen/generate (gen/such-that odd? (gen/choose 0 100) {:max-tries 50}) 30 1))"
        ),
        "true"
    );
}

#[test]
fn such_that_failure_modes() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/generate (gen/such-that even? (gen/return 1)) 30 1)",
        "ExceptionInfo",
        "Couldn't satisfy such-that predicate after 10 tries.",
    );
    assert_throws(
        &mut eval,
        "(gen/generate (gen/such-that even? (gen/return 1) 3) 30 1)",
        "ExceptionInfo",
        "Couldn't satisfy such-that predicate after 3 tries.",
    );
    assert_throws(
        &mut eval,
        "(gen/such-that even? (gen/return 1) :nope)",
        "ExceptionInfo",
        "Bad argument to such-that!",
    );
    assert_throws(
        &mut eval,
        "(gen/such-that even? 7)",
        "AssertionError",
        "Second arg to such-that must be a generator",
    );
}

#[test]
fn not_empty_never_empty() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(not (empty? %))
               (mapv #(gen/generate (gen/not-empty (gen/vector (gen/boolean))) 30 %)
                     (range 100)))"
        ),
        "true"
    );
    assert_throws(
        &mut eval,
        "(gen/not-empty 9)",
        "AssertionError",
        "Arg to not-empty must be a generator",
    );
}

// ---- tuple / vector / list ----

#[test]
fn tuple_positions_come_from_arm_generators() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(gen/generate (gen/tuple (gen/return 1) (gen/choose 0 0) (gen/elements [:k])) 30 2)"
        ),
        "[1 0 :k]"
    );
    assert_eq!(
        run(&mut eval, "(gen/generate (apply gen/tuple []) 30 1)"),
        "[]"
    );
    assert_throws(
        &mut eval,
        "(gen/tuple (gen/return 1) 2)",
        "AssertionError",
        "Args to tuple must be generators",
    );
}

#[test]
fn vector_arities_and_bounds() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(gen/generate (gen/vector (gen/return 8) 3) 30 1)"
        ),
        "[8 8 8]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(<= 2 (count %) 4)
               (mapv #(gen/generate (gen/vector (gen/return 1) 2 4) 30 %) (range 50)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (vector? %) (<= 0 (count %) 30))
               (mapv #(gen/generate (gen/vector (gen/boolean)) 30 %) (range 50)))"
        ),
        "true"
    );
    assert_throws(
        &mut eval,
        "(gen/vector 3)",
        "AssertionError",
        "Arg to vector must be a generator",
    );
    assert_throws(
        &mut eval,
        "(gen/vector 3 2)",
        "AssertionError",
        "First arg to vector must be a generator",
    );
}

#[test]
fn list_generates_bounded_lists() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (list? %) (<= 0 (count %) 30))
               (mapv #(gen/generate (gen/list (gen/boolean)) 30 %) (range 50)))"
        ),
        "true"
    );
    assert_throws(
        &mut eval,
        "(gen/list 3)",
        "AssertionError",
        "First arg to list must be a generator",
    );
}

// ---- set / map / shuffle / cat ----

#[test]
fn set_map_honor_num_elements() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (set? %) (= 5 (count %)))
               (mapv #(gen/generate (gen/set (gen/choose 0 100) {:num-elements 5}) 30 %)
                     (range 30)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (map? %) (= 3 (count %)) (every? (fn [[k v]] (and (<= 0 k 100) (= :v v))) %))
               (mapv #(gen/generate (gen/map (gen/choose 0 100) (gen/return :v) {:num-elements 3}) 30 %)
                     (range 30)))"
        ),
        "true"
    );
}

#[test]
fn set_map_default_bounds() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (set? %) (<= 0 (count %) 30))
               (mapv #(gen/generate (gen/set (gen/boolean)) 30 %) (range 30)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (map? %) (<= 0 (count %) 30))
               (mapv #(gen/generate (gen/map (gen/choose 0 9) (gen/choose 0 9)) 30 %) (range 30)))"
        ),
        "true"
    );
}

#[test]
fn set_map_distinctness_failure() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/generate (gen/set (gen/return 1) {:num-elements 5}) 30 1)",
        "ExceptionInfo",
        "Couldn't generate enough distinct elements!",
    );
    assert_throws(
        &mut eval,
        "(gen/generate (gen/map (gen/return 1) (gen/return 2) {:num-elements 5}) 30 1)",
        "ExceptionInfo",
        "Couldn't generate enough distinct elements!",
    );
    assert_throws(
        &mut eval,
        "(gen/set 3)",
        "AssertionError",
        "First arg to set must be a generator!",
    );
}

#[test]
fn shuffle_permutes() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(= [1 2 3 4 5] (sort %))
               (mapv #(gen/generate (gen/shuffle [1 2 3 4 5]) 30 %) (range 30)))"
        ),
        "true"
    );
    assert_eq!(run(&mut eval, "(gen/generate (gen/shuffle []) 30 1)"), "[]");
    assert_eq!(
        run(
            &mut eval,
            "(let [v (gen/generate (gen/shuffle '(1 2)) 30 1)] [(vector? v) (sort v)])"
        ),
        "[true (1 2)]"
    );
}

#[test]
fn cat_concatenates_arm_draws() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(gen/generate (gen/cat (gen/vector (gen/return 1) 2) (gen/vector (gen/return 2) 1)) 30 1)"
        ),
        "(1 1 2)"
    );
    assert_eq!(run(&mut eval, "(gen/generate (gen/cat) 30 1)"), "()");
}

// ---- large-integer / double ----

#[test]
fn large_integer_full_range() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? integer? (mapv #(gen/generate (gen/large-integer) 30 %) (range 100)))"
        ),
        "true"
    );
}

#[test]
fn large_integer_star_honors_bounds() {
    let mut eval = boot();
    require_gen(&mut eval);
    for (opts, lo, hi) in [
        ("{:min 9000 :max 10000}", 9000, 10000),
        ("{:min 1}", 1, 9223372036854775807i64),
        ("{:max -1}", -9223372036854775808i64, -1),
        ("{:min 0}", 0, 9223372036854775807i64),
        ("{:min -5 :max 5}", -5, 5),
    ] {
        assert_eq!(
            run(
                &mut eval,
                &format!(
                    "(every? #(and (integer? %) (<= {lo} % {hi}))
                        (mapv #(gen/generate (gen/large-integer* {opts}) 30 %) (range 50)))"
                )
            ),
            "true",
            "bounds {opts}"
        );
    }
    assert_throws(
        &mut eval,
        "(gen/large-integer* {:min 2 :max 1})",
        "AssertionError",
        "Assert failed",
    );
}

#[test]
fn double_covers_edges() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(let [draws (mapv #(gen/generate (gen/double) 30 %) (range 1500))]
               [(every? double? draws)
                (some #(not (= % %)) draws)
                (some #(= ##Inf %) draws)
                (some #(= ##-Inf %) draws)])"
        ),
        "[true true true true]"
    );
}

#[test]
fn double_star_honors_opts() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (= % %) (< (abs %) 1.8e308))
               (mapv #(gen/generate (gen/double* {:infinite? false :NaN? false}) 30 %)
                     (range 300)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(<= 1.5 % 2.5)
               (mapv #(gen/generate (gen/double* {:infinite? false :NaN? false :min 1.5 :max 2.5}) 30 %)
                     (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(<= 0.0 %)
               (mapv #(gen/generate (gen/double* {:infinite? false :NaN? false :min 0.0}) 30 %)
                     (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(<= % 0.0)
               (mapv #(gen/generate (gen/double* {:infinite? false :NaN? false :max 0.0}) 30 %)
                     (range 100)))"
        ),
        "true"
    );
}

// ---- scalar prims ----

#[test]
fn boolean_char_bytes_prims() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= #{true false} (set (mapv #(gen/generate (gen/boolean) 30 %) (range 20))))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? char? (mapv #(gen/generate (gen/char) 30 %) (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? bytes? (mapv #(gen/generate (gen/bytes) 30 %) (range 20)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? integer? (mapv #(gen/generate (gen/int) 30 %) (range 50)))"
        ),
        "true"
    );
}

#[test]
fn char_range_prims() {
    let mut eval = boot();
    require_gen(&mut eval);
    for (prim, lo, hi) in [
        ("(gen/char-ascii)", 32, 126),
        ("(gen/char-alphanumeric)", 0, 122),
        ("(gen/char-alpha)", 65, 122),
    ] {
        assert_eq!(
            run(
                &mut eval,
                &format!(
                    "(every? #(let [c (int %)] (<= {lo} c {hi}))
                        (mapv #(gen/generate {prim} 30 %) (range 100)))"
                )
            ),
            "true",
            "{prim} range"
        );
    }
    assert_eq!(
        run(
            &mut eval,
            "(every? #(let [c (int %)] (or (<= 48 c 57) (<= 65 c 90) (<= 97 c 122)))
               (mapv #(gen/generate (gen/char-alphanumeric) 30 %) (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(let [c (int %)] (or (<= 65 c 90) (<= 97 c 122)))
               (mapv #(gen/generate (gen/char-alpha) 30 %) (range 100)))"
        ),
        "true"
    );
}

#[test]
fn string_prims() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? string? (mapv #(gen/generate (gen/string) 30 %) (range 20)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(every? (fn [ch] (let [c (int ch)] (<= 32 c 126))) %)
               (mapv #(gen/generate (gen/string-ascii) 30 %) (range 20)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(every? (fn [ch] (let [c (int ch)]
                                              (or (<= 48 c 57) (<= 65 c 90) (<= 97 c 122)))) %)
               (mapv #(gen/generate (gen/string-alphanumeric) 30 %) (range 20)))"
        ),
        "true"
    );
}

#[test]
fn keyword_symbol_prims() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (keyword? %) (nil? (namespace %)))
               (mapv #(gen/generate (gen/keyword) 30 %) (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? qualified-keyword? (mapv #(gen/generate (gen/keyword-ns) 30 %) (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (symbol? %) (nil? (namespace %)))
               (mapv #(gen/generate (gen/symbol) 30 %) (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? qualified-symbol? (mapv #(gen/generate (gen/symbol-ns) 30 %) (range 100)))"
        ),
        "true"
    );
}

#[test]
fn uuid_prim_is_version_4() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (uuid? %) (= \"4\" (subs (str %) 14 15))
                                 (contains? #{\"8\" \"9\" \"a\" \"b\"} (subs (str %) 19 20)))
               (mapv #(gen/generate (gen/uuid) 30 %) (range 50)))"
        ),
        "true"
    );
}

#[test]
fn ratio_prim_yields_ratios_or_ints() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(or (ratio? %) (integer? %))
               (mapv #(gen/generate (gen/ratio) 30 %) (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(some? (some ratio? (mapv #(gen/generate (gen/ratio) 30 %) (range 100))))"
        ),
        "true"
    );
}

// ---- simple-type / any ----

#[test]
fn simple_type_draws_scalars() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(or (number? %) (char? %) (string? %) (boolean? %)
                            (keyword? %) (symbol? %) (uuid? %))
               (mapv #(gen/generate (gen/simple-type) 30 %) (range 100)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? some? (mapv #(gen/generate (gen/simple-type-printable) 30 %) (range 50)))"
        ),
        "true"
    );
}

#[test]
fn any_draws_nested_non_nil_values() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? some? (mapv #(gen/generate (gen/any) 30 %) (range 30)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(every? some? (mapv #(gen/generate (gen/any-printable) 30 %) (range 30)))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(some? (some coll? (mapv #(gen/generate (gen/any-printable) 30 %) (range 30))))"
        ),
        "true"
    );
}

// ---- inst / decimal ctors ----

#[test]
fn inst_builtin_draws_bounded_instants() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #(and (inst? %) (<= 0 (inst-ms %) 4102444800000))
               (mapv #(gen/generate (gen/gen-for-pred inst?) 30 %) (range 50)))"
        ),
        "true"
    );
}

#[test]
fn decimal_builtin_draws_decimals() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? decimal? (mapv #(gen/generate (gen/gen-for-pred decimal?) 30 %) (range 50)))"
        ),
        "true"
    );
}

// ---- gen-for-pred ----

/// Every `gen-builtins` entry (all 45 oracle preds except `uri?`):
/// resolves to a generator whose draws satisfy the pred.
#[test]
fn gen_for_pred_table() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(let [fails (mapv (fn [sym]
                                 (let [pred @(resolve sym)
                                       g (gen/gen-for-pred pred)]
                                   (when-not (and (some? g)
                                                  (every? pred (mapv #(gen/generate g 30 %) (range 30))))
                                     sym)))
                               '[any? some? number? integer? int? pos-int? neg-int? nat-int?
                                 float? double? boolean? string? ident? simple-ident?
                                 qualified-ident? keyword? simple-keyword? qualified-keyword?
                                 symbol? simple-symbol? qualified-symbol? uuid? decimal? inst?
                                 seqable? indexed? map? vector? list? seq? char? set? nil?
                                 false? true? zero? rational? coll? empty? associative?
                                 sequential? ratio? bytes?])]
               (vec (remove nil? fails)))"
        ),
        "[]"
    );
}

#[test]
fn gen_for_pred_set_and_unknown() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(every? #{1 2 3} (mapv #(gen/generate (gen/gen-for-pred #{1 2 3}) 30 %) (range 30)))"
        ),
        "true"
    );
    assert_eq!(run(&mut eval, "(gen/gen-for-pred odd?)"), "nil");
}

#[test]
fn gen_for_pred_uri_has_no_builtin() {
    // Documented divergence 3: the port has no host URI constructor,
    // so `uri?` is absent from `gen-builtins` until F2-or-later.
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(run(&mut eval, "(gen/gen-for-pred uri?)"), "nil");
}

// ---- gen-for-name ----

#[test]
fn gen_for_name_resolves_generator_vars() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(do (in-ns 'clojure.spec.gen.alpha)
                 (def test-gen-xyz (clojure.spec.gen.alpha/vector
                                    (clojure.spec.gen.alpha/choose 0 5)))
                 (in-ns 'user)
                 (= (gen/generate (gen/gen-for-name 'clojure.spec.gen.alpha/test-gen-xyz) 30 4)
                    (gen/generate @#'clojure.spec.gen.alpha/test-gen-xyz 30 4)))"
        ),
        "true"
    );
}

#[test]
fn gen_for_name_failure_modes() {
    let mut eval = boot();
    require_gen(&mut eval);
    assert_throws(
        &mut eval,
        "(gen/gen-for-name 'unqualified)",
        "AssertionError",
        "Assert failed",
    );
    assert_throws(
        &mut eval,
        "(gen/gen-for-name 'clojure.core/+)",
        "ExceptionInfo",
        "Var clojure.core/+ is not a generator",
    );
    assert_throws(
        &mut eval,
        "(gen/gen-for-name 'clojure.core/name-does-not-exist)",
        "ExceptionInfo",
        "Var clojure.core/name-does-not-exist is not on the classpath",
    );
    assert_throws(
        &mut eval,
        "(gen/gen-for-name 'ns.does.not.exist/f)",
        "FileNotFoundException",
        "Could not locate",
    );
    // Prims are fns in this port (as upstream), not generator values.
    assert_throws(
        &mut eval,
        "(gen/gen-for-name 'clojure.spec.gen.alpha/boolean)",
        "ExceptionInfo",
        "Var clojure.spec.gen.alpha/boolean is not a generator",
    );
}
