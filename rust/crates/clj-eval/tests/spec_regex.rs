// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-122 slice-E tests: regex specs (`s/cat`, `s/alt`, `s/*`,
//! `s/+`, `s/?`, `s/&`) over the derivative-matching engine, plus
//! `s/keys*` expansion.
//!
//! Oracle: upstream `clojure/spec/alpha.clj` from
//! `org.clojure/spec.alpha-0.6.249.jar`, read as TEXT ONLY (the JVM
//! was never executed for this slice). Every expected value is
//! derived by reading the upstream source; derivations live in
//! `docs/gap-122/slice-E.md`, whose scenario numbers prefix each
//! test name below.

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

// ---- macro expansions (scenarios 1-6) ----

#[test]
fn star_expands_to_rep_impl() {
    // Scenario 1 (oracle 607-611).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (macroexpand-1 '(s/* even?)) '(clojure.spec.alpha/rep-impl (quote clojure.core/even?) even?))"
        ),
        "true"
    );
}

#[test]
fn plus_expands_to_rep_plus_impl() {
    // Scenario 2 (oracle 613-617).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (macroexpand-1 '(s/+ even?)) '(clojure.spec.alpha/rep+impl (quote clojure.core/even?) even?))"
        ),
        "true"
    );
}

#[test]
fn maybe_expands_to_maybe_impl() {
    // Scenario 3 (oracle 619-623).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (macroexpand-1 '(s/? even?)) '(clojure.spec.alpha/maybe-impl even? (quote clojure.core/even?)))"
        ),
        "true"
    );
}

#[test]
fn cat_expands_to_cat_impl() {
    // Scenario 4 (oracle 642-656).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (macroexpand-1 '(s/cat :e even? :o odd?)) '(clojure.spec.alpha/cat-impl [:e :o] [even? odd?] (quote [clojure.core/even? clojure.core/odd?])))"
        ),
        "true"
    );
}

#[test]
fn alt_expands_to_alt_impl() {
    // Scenario 5 (oracle 625-640).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (macroexpand-1 '(s/alt :e even? :o odd?)) '(clojure.spec.alpha/alt-impl [:e :o] [even? odd?] (quote [clojure.core/even? clojure.core/odd?])))"
        ),
        "true"
    );
}

#[test]
fn amp_expands_to_amp_impl() {
    // Scenario 6 (oracle 658-665).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (macroexpand-1 '(s/& (s/* integer?) vector?)) '(clojure.spec.alpha/amp-impl (s/* integer?) (quote (clojure.spec.alpha/* clojure.core/integer?)) [vector?] (quote [clojure.core/vector?])))"
        ),
        "true"
    );
}

// ---- macro asserts (scenarios 7-8) ----

#[test]
fn cat_asserts_pairing() {
    // Scenario 7 (oracle 654).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/cat :e)"),
        "java.lang.AssertionError | Assert failed: cat expects k1 p1 k2 p2..., where ks are keywords\n(c/and (even? (count key-pred-forms)) (every? keyword? keys))"
    );
}

#[test]
fn alt_asserts_pairing() {
    // Scenario 8 (oracle 638).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        uncaught(&mut eval, "(s/alt \"e\" even?)"),
        "java.lang.AssertionError | Assert failed: alt expects k1 p1 k2 p2..., where ks are keywords\n(c/and (even? (count key-pred-forms)) (every? keyword? keys))"
    );
}

// ---- cat / valid? + conform + unform (scenarios 9-17) ----

#[test]
fn cat_conform_destructures() {
    // Scenario 9 (oracle re-conform 1658-1669, deriv 1526-1544,
    // pcat* 1385-1394).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (s/conform (s/cat :e even? :o odd?) [4 5]) {:e 4 :o 5})"
        ),
        "true"
    );
}

#[test]
fn cat_element_failure_stays_invalid() {
    // Scenario 10.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/cat :e even? :o odd?) [5 4])"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/cat :e even? :o odd?) [5 4])"),
        "false"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/cat :e even? :o odd?) [4 5])"),
        "true"
    );
}

#[test]
fn cat_empty_input_is_invalid() {
    // Scenario 11 (oracle accept-nil? 1463-1473).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? (s/cat :e even?) [])"), "false");
}

#[test]
fn cat_takes_any_sequential() {
    // Scenario 12 (oracle regex-spec-impl 1709-1712).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (s/conform (s/cat :e even? :o odd?) '(4 5)) {:e 4 :o 5})"
        ),
        "true"
    );
}

#[test]
fn cat_rejects_non_sequential() {
    // Scenario 13.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? (s/cat :e even?) {:e 4})"),
        "false"
    );
    assert_eq!(run(&mut eval, "(s/valid? (s/cat :e even?) 42)"), "false");
}

#[test]
fn nil_input_follows_accept_nil() {
    // Scenario 14.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? (s/cat :e even?) nil)"), "false");
    assert_eq!(run(&mut eval, "(s/valid? (s/* integer?) nil)"), "true");
}

#[test]
fn cat_unform_flattens_in_key_order() {
    // Scenario 15 (oracle op-unform 1492-1511).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/unform (s/cat :e even? :o odd?) {:e 4 :o 5})"),
        "(4 5)"
    );
}

#[test]
fn cat_conform_unform_round_trip() {
    // Scenario 16.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (s/unform (s/cat :e even? :o odd?) (s/conform (s/cat :e even? :o odd?) [4 5])) [4 5])"
        ),
        "true"
    );
}

#[test]
fn empty_cat() {
    // Scenario 17.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/conform (s/cat) [])"), "{}");
    assert_eq!(run(&mut eval, "(s/valid? (s/cat) [])"), "true");
    assert_eq!(run(&mut eval, "(some? (s/regex? (s/cat)))"), "true");
}

// ---- nested op (scenario 18) ----

#[test]
fn cat_with_nested_alt_element() {
    // Scenario 18 (oracle deriv 1535 recurses into the op).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (s/conform (s/cat :a (s/alt :x integer?)) [5]) {:a [:x 5]})"
        ),
        "true"
    );
}

// ---- alt / conform + unform (scenarios 19-22) ----

#[test]
fn alt_conform_tags() {
    // Scenario 19 (oracle tagged-ret 1058-1059).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/alt :e even? :o odd?) [4])"),
        "[:e 4]"
    );
    assert_eq!(
        run(&mut eval, "(key (s/conform (s/alt :e even? :o odd?) [5]))"),
        ":o"
    );
    assert_eq!(
        run(&mut eval, "(val (s/conform (s/alt :e even? :o odd?) [5]))"),
        "5"
    );
}

#[test]
fn alt_no_branch_matches() {
    // Scenario 20.
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/alt :s string? :k keyword?) [42])"),
        ":clojure.spec.alpha/invalid"
    );
    // Throwing preds propagate out of deriv (upstream dt calls
    // the pred directly).
    assert_eq!(
        uncaught(&mut eval, "(s/conform (s/alt :e even? :o odd?) [4.5])"),
        "java.lang.IllegalArgumentException | Argument must be an integer: 4.5"
    );
}

#[test]
fn alt_unform_unwraps_tag() {
    // Scenario 21 (oracle op-unform 1508-1511).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/unform (s/alt :e even? :o odd?) [:e 4])"),
        "[4]"
    );
}

#[test]
fn bare_alt_is_not_a_regex() {
    // Scenario 22: verbatim upstream quirk (oracle alt* 1432-1442,
    // alt-impl 1447-1449).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/regex? (s/alt))"), "nil");
}

// ---- * / + / ? (scenarios 23-26) ----

#[test]
fn star_collects() {
    // Scenario 23 (oracle rep* 1402-1407).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/* integer?) [1 2 3])"),
        "[1 2 3]"
    );
    assert_eq!(run(&mut eval, "(s/conform (s/* integer?) [])"), "[]");
    assert_eq!(run(&mut eval, "(s/valid? (s/* integer?) [])"), "true");
    assert_eq!(
        run(&mut eval, "(s/conform (s/* integer?) [1 :a])"),
        ":clojure.spec.alpha/invalid"
    );
}

#[test]
fn plus_needs_one() {
    // Scenario 24 (oracle rep+impl 1413-1416).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/conform (s/+ integer?) [7])"), "[7]");
    assert_eq!(
        run(&mut eval, "(s/conform (s/+ integer?) [])"),
        ":clojure.spec.alpha/invalid"
    );
}

#[test]
fn maybe_is_single_or_nil() {
    // Scenario 25 (oracle re-conform 1662-1666).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/conform (s/? integer?) [])"), "nil");
    assert_eq!(run(&mut eval, "(s/conform (s/? integer?) [1])"), "1");
    assert_eq!(
        run(&mut eval, "(s/conform (s/? integer?) [1 2])"),
        ":clojure.spec.alpha/invalid"
    );
}

#[test]
fn rep_unforms() {
    // Scenario 26 (oracle op-unform 1496-1507).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/unform (s/* integer?) [1 2])"), "(1 2)");
    assert_eq!(run(&mut eval, "(s/unform (s/+ integer?) [1 2])"), "(1 2)");
    assert_eq!(run(&mut eval, "(s/unform (s/? integer?) 5)"), "[5]");
    assert_eq!(run(&mut eval, "(s/unform (s/? integer?) nil)"), "[nil]");
}

// ---- & (scenarios 27-28) ----

#[test]
fn amp_filters() {
    // Scenario 27 (oracle deriv 1534-1539, and-preds 1128-1137).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/& (s/* integer?) #(> (count %) 2)) [1 2 3])"
        ),
        "[1 2 3]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/conform (s/& (s/* integer?) #(> (count %) 2)) [1])"
        ),
        ":clojure.spec.alpha/invalid"
    );
}

#[test]
fn amp_unform() {
    // Scenario 28 (oracle op-unform 1496-1498).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/unform (s/& (s/* integer?) vector?) [1 2])"),
        "(1 2)"
    );
}

// ---- form / describe (scenarios 29-30) ----

#[test]
fn cat_form_and_describe() {
    // Scenario 29 (oracle op-describe 1546-1560).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/form (s/cat :e even? :o odd?))"),
        "(clojure.spec.alpha/cat :e clojure.core/even? :o clojure.core/odd?)"
    );
    assert_eq!(
        run(&mut eval, "(s/describe (s/cat :e even? :o odd?))"),
        "(cat :e even? :o odd?)"
    );
}

#[test]
fn other_describes() {
    // Scenario 30 (oracle op-describe 1546-1560).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/describe (s/alt :e even? :o odd?))"),
        "(alt :e even? :o odd?)"
    );
    assert_eq!(run(&mut eval, "(s/describe (s/* even?))"), "(* even?)");
    assert_eq!(
        run(&mut eval, "(s/describe (s/+ integer?))"),
        "(+ integer?)"
    );
    assert_eq!(run(&mut eval, "(s/describe (s/? odd?))"), "(? odd?)");
    assert_eq!(
        run(
            &mut eval,
            "(s/describe (s/& (s/* integer?) #(> (count %) 1)))"
        ),
        "(& (* integer?) (> (count %) 1))"
    );
}

// ---- explain (scenarios 31-38) ----

#[test]
fn explain_cat_element_failure() {
    // Scenario 31 (oracle re-explain 1671-1695, op-explain
    // 1586-1597).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/cat :e even? :o odd?) [5 4])"),
        "\"5 - failed: even? in: [0] at: [:e]\\n\""
    );
}

#[test]
fn explain_insufficient_input() {
    // Scenario 32 (oracle op-explain 1569-1575).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/cat :e even? :o odd?) [4])"),
        "\"() - failed: Insufficient input at: [:o]\\n\""
    );
}

#[test]
fn explain_extra_input() {
    // Scenario 33 (oracle re-explain 1682-1688).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/cat :e even?) [4 5])"),
        "\"(5) - failed: Extra input in: [1]\\n\""
    );
}

#[test]
fn explain_non_sequential() {
    // Scenario 34 (oracle regex-spec-impl explain* 1714-1717).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/cat :e even?) 42)"),
        "\"42 - failed: (or (nil? %) (sequential? %))\\n\""
    );
}

#[test]
fn explain_alt_covers_every_branch() {
    // Scenario 35 (oracle op-explain 1598-1608).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/alt :s string? :k keyword?) [42])"
        ),
        "\"42 - failed: string? in: [0] at: [:s]\\n42 - failed: keyword? in: [0] at: [:k]\\n\""
    );
}

#[test]
fn explain_star_mid_sequence_failure() {
    // Scenario 36 (oracle op-explain 1609-1612).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/* integer?) [1 :a])"),
        "\":a - failed: integer? in: [1]\\n\""
    );
}

#[test]
fn explain_amp_predicate_failure() {
    // Scenario 37 (oracle op-explain 1579-1585).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/& (s/cat :e even?) #(> (:e %) 10)) [4])"
        ),
        "\"{:e 4} - failed: (> (:e %) 10) in: [0]\\n\""
    );
}

#[test]
fn explain_registered_name_suffix_and_via() {
    // Scenario 38 (oracle op-explain 1566 conj's the op's own
    // :name onto the caller via — verbatim doubled via).
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/n (s/cat :e even?))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/n [5])"),
        "\"5 - failed: even? in: [0] at: [:e] spec: :user/n\\n\""
    );
    assert_eq!(
        run(
            &mut eval,
            "(:via (first (:clojure.spec.alpha/problems (s/explain-data :user/n [5]))))"
        ),
        "[:user/n :user/n]"
    );
}

// ---- empty-input edges (scenarios 43-46) ----

#[test]
fn amp_over_empty_match_conforms_nil() {
    // Scenario 43 (oracle preturn 1477-1490, noret? 1455-1459,
    // re-conform 1662-1666).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/conform (s/& (s/* integer?) vector?) [])"),
        "nil"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? (s/& (s/* integer?) vector?) [])"),
        "true"
    );
}

#[test]
fn amp_over_empty_input_explains_predicates() {
    // Scenario 44 (oracle op-explain 1579-1583).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/& (s/* integer?) #(> (count %) 5)) [])"
        ),
        "\"[] - failed: (> (count %) 5)\\n\""
    );
}

#[test]
fn amp_over_empty_input_can_be_insufficient() {
    // Scenario 45 (oracle op-explain 1584).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/explain-str (s/& (s/cat :e even?) vector?) [])"
        ),
        "\"() - failed: Insufficient input\\n\""
    );
}

#[test]
fn alt_over_empty_input_can_be_insufficient() {
    // Scenario 46 (oracle op-explain 1598-1599).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str (s/alt :a (s/cat :x even?)) [])"),
        "\"() - failed: Insufficient input\\n\""
    );
}

// ---- registry / integration (scenarios 39-40) ----

#[test]
fn regex_through_the_registry() {
    // Scenario 39 (oracle maybe-spec 110-119, def-impl stores ops
    // raw).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/def :user/e (s/cat :e even?))"),
        ":user/e"
    );
    assert_eq!(run(&mut eval, "(s/valid? :user/e [4])"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/e [5])"), "false");
    assert_eq!(
        run(&mut eval, "(some? (s/regex? (s/get-spec :user/e)))"),
        "true"
    );
    assert_eq!(run(&mut eval, "(s/spec? (s/cat :e even?))"), "nil");
    assert_eq!(
        run(&mut eval, "(some? (s/spec? (s/spec (s/cat :e even?))))"),
        "true"
    );
}

#[test]
fn regex_forward_refs_and_aliases() {
    // Scenario 39 continued: ops hold idents, resolved lazily.
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/w (s/cat :x :user/late))")
        .unwrap();
    eval.eval_source("(s/def :user/late integer?)").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/w [4])"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/w [4.5])"), "false");
    eval.eval_source("(s/def :user/e2 (s/cat :e even?))")
        .unwrap();
    eval.eval_source("(s/def :user/al :user/e2)").unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/al [4])"), "true");
}

#[test]
fn spec_macro_wraps_regex() {
    // Scenario 40 (oracle spec-impl 919).
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? (s/spec (s/cat :e even?)) [4])"),
        "true"
    );
}

// ---- generators wired (scenario 41) ----

#[test]
fn gen_paths_wired() {
    // Scenario 41 (oracle re-gen 1617-1655): slice F2 wires the real
    // generator (this test asserted the slice-E stub before F2).
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(require '[clojure.spec.gen.alpha :as gen])")
        .unwrap();
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? (s/cat :e int?)
                (gen/generate (clojure.spec.alpha/gen* (s/spec (s/cat :e int?)) nil [] {}) 30 1))"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(def gspec (clojure.spec.alpha/with-gen* (s/spec (s/cat :e int?)) (fn [] (gen/return [7]))))
             (gen/generate (clojure.spec.alpha/gen* gspec nil [] {}) 30 1)"
        ),
        "[7]"
    );
}

// ---- keys* (scenario 42) ----

#[test]
fn keys_star_expands() {
    // Scenario 42 (oracle 1794-1812): the skeleton is pinned
    // structurally (each check expands afresh — gensyms differ
    // across expansions, so shared-gensym pins bind one `e`).
    // Runtime use needs slices A + D + F and is NOT pinned here.
    let mut eval = boot();
    require_spec(&mut eval);
    let exp = "(macroexpand-1 '(s/keys* :req-un [:user/a]))";
    // `(let [G (keys ...)] ...)`.
    assert_eq!(
        run(&mut eval, &format!("(let [e {exp}] (= (first e) 'let))")),
        "true"
    );
    assert_eq!(
        run(&mut eval, &format!("(let [e {exp}] (vector? (second e)))")),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp}] (= (second (second e)) '(clojure.spec.alpha/keys :req-un [:user/a])))"
            )
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!("(let [e {exp}] (symbol? (first (second e))))")
        ),
        "true"
    );
    // `(with-gen (& (* (cat :k keyword? :v any?)) :kvs->map G) ...)`.
    assert_eq!(
        run(
            &mut eval,
            &format!("(let [e {exp}] (= (first (nth e 2)) 'clojure.spec.alpha/with-gen))")
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} amp (second (nth e 2))] (= (first amp) 'clojure.spec.alpha/&))"
            )
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} amp (second (nth e 2))] (= (first (second amp)) 'clojure.spec.alpha/*))"
            )
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} amp (second (nth e 2))] (= (second (second amp)) '(clojure.spec.alpha/cat :clojure.spec.alpha/k keyword? :clojure.spec.alpha/v any?)))"
            )
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} amp (second (nth e 2))] (= (nth amp 2) :clojure.spec.alpha/kvs->map))"
            )
        ),
        "true"
    );
    // The same gensym threads the binding, the `&` tail, and the
    // `(gen G)` call.
    assert_eq!(
        run(
            &mut eval,
            &format!("(let [e {exp} amp (second (nth e 2))] (= (nth amp 3) (first (second e))))")
        ),
        "true"
    );
    // `(fn [] (fmap (fn [m] (apply concat m)) (gen G)))`.
    assert_eq!(
        run(
            &mut eval,
            &format!("(let [e {exp} g (nth (nth e 2) 2)] (= (take 2 g) '(fn [])))")
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} g (nth (nth e 2) 2)] (= (first (nth g 2)) 'clojure.spec.gen.alpha/fmap))"
            )
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} g (nth (nth e 2) 2) fmc (nth g 2)] (= (first (nth fmc 2)) 'clojure.spec.alpha/gen))"
            )
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} g (nth (nth e 2) 2) fmc (nth g 2)] (= (second (nth fmc 2)) (first (second e))))"
            )
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [e {exp} g (nth (nth e 2) 2) f (second (nth g 2))] (= (nth f 2) (list 'apply 'concat (first (second f)))))"
            )
        ),
        "true"
    );
}
