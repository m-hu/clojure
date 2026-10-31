// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-122 slice-A tests: map specs (`s/keys` + `s/merge`).
//!
//! Oracle: upstream `clojure/spec/alpha.clj` from
//! `org.clojure/spec.alpha-0.5.238` (see `docs/gap-122/slice-A.md`
//! for the jar pin + hashes). The JVM was never executed: every
//! expectation below is derived by reading the upstream source,
//! and each test cites the oracle lines. Scenario numbers refer to
//! `docs/gap-122/slice-A.md`.

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

fn def_person(eval: &mut Evaluator, keys_form: &str) {
    eval.eval_source("(s/def :user/name string?)").unwrap();
    eval.eval_source("(s/def :user/age integer?)").unwrap();
    eval.eval_source(&format!("(s/def :user/person {keys_form})"))
        .unwrap();
}

// ---- keys / valid? (scenarios 1-4) ----
// Oracle: `keys` macro 442-474, `map-spec-impl` `conform*` 835-849.

#[test]
fn keys_valid_req() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:user/name \"n\"})"),
        "true"
    );
    assert_eq!(run(&mut eval, "(s/valid? :user/person {})"), "false");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:user/name 42})"),
        "false"
    );
    assert_eq!(run(&mut eval, "(s/valid? :user/person [:a])"), "false");
}

// Scenario 5. Oracle: `conform*` 837-848 loops over present
// entries; nothing rejects extras.
#[test]
fn keys_maps_are_open() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? :user/person {:user/name \"n\" :other 1})"
        ),
        "true"
    );
}

// Scenario 6. Oracle: `opt-keys`/`opt-specs` 450-451 feed value
// checking only (via `k->s` 826), never `keys-pred`.
#[test]
fn keys_opt() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name] :opt [:user/age])");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:user/name \"n\"})"),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? :user/person {:user/name \"n\" :user/age 3})"
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? :user/person {:user/name \"n\" :user/age \"old\"})"
        ),
        "false"
    );
}

// Scenario 7. Oracle: docstring 436-438, `conform*` 840-841
// consults the registry for every present key.
#[test]
fn keys_qualified_values_always_validated() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? :user/person {:user/name \"n\" :user/age \"old\"})"
        ),
        "false"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? :user/person {:user/name \"n\" :user/age 3})"
        ),
        "true"
    );
}

// Scenario 8. Oracle: docstring 424-434, `unk` + `req-un`
// wiring 443/445/449/463.
#[test]
fn keys_req_un() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req-un [:user/name])");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:name \"n\"})"),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:name 42})"),
        "false"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:user/name \"n\"})"),
        "false"
    );
}

// Scenario 9. Oracle: `opt-un` wiring 450-451 + `unk` 443.
#[test]
fn keys_opt_un() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(
        &mut eval,
        "(s/keys :req-un [:user/name] :opt-un [:user/age])",
    );
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:name \"n\"})"),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:name \"n\" :age 3})"),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            "(s/valid? :user/person {:name \"n\" :age \"old\"})"
        ),
        "false"
    );
}

// Scenarios 10-11. Oracle: docstring 420-422, `parse-req`
// 453-460 (non-keyword req entries keep `or`/`and` heads while
// keywords become `contains?` checks).
#[test]
fn keys_key_groups() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/a integer?)").unwrap();
    eval.eval_source("(s/def :user/b integer?)").unwrap();
    eval.eval_source("(s/def :user/or-ab (s/keys :req [(or :user/a :user/b)]))")
        .unwrap();
    eval.eval_source("(s/def :user/and-ab (s/keys :req [(and :user/a :user/b)]))")
        .unwrap();
    assert_eq!(run(&mut eval, "(s/valid? :user/or-ab {:user/a 1})"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/or-ab {:user/b 2})"), "true");
    assert_eq!(run(&mut eval, "(s/valid? :user/or-ab {})"), "false");
    assert_eq!(
        run(&mut eval, "(s/valid? :user/and-ab {:user/a 1 :user/b 2})"),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/valid? :user/and-ab {:user/a 1})"),
        "false"
    );
}

// Scenario 12. Oracle: `keys-pred` 461-464 degrades to a lone
// `map?` check when every kwarg is empty.
#[test]
fn keys_empty() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(run(&mut eval, "(s/valid? (s/keys) {:a 1})"), "true");
    assert_eq!(run(&mut eval, "(s/valid? (s/keys) 42)"), "false");
}

// ---- keys / conform + unform (scenarios 13-14) ----
// Oracle: `conform*` 835-849, `unform*` 850-860.

#[test]
fn keys_conform_unform() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    assert_eq!(
        run(&mut eval, "(s/conform :user/person {:user/name \"n\"})"),
        "#:user{:name \"n\"}"
    );
    assert_eq!(
        run(&mut eval, "(s/conform :user/person {})"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(
        run(&mut eval, "(s/conform :user/person {:user/name 42})"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(
        run(&mut eval, "(s/conform :user/person [:a])"),
        ":clojure.spec.alpha/invalid"
    );
    assert_eq!(
        run(
            &mut eval,
            "(= (s/unform :user/person {:user/name \"n\"}) {:user/name \"n\"})"
        ),
        "true"
    );
}

// ---- keys / explain (scenarios 15-19, 21) ----
// Oracle: `explain*` 861-877, `explain-1` 786-791,
// `pred-forms` via `res`/`unfn` 466.

#[test]
fn keys_explain_missing() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/person {})"),
        "\"{} - failed: (contains? % :user/name) spec: :user/person\\n\""
    );
}

#[test]
fn keys_explain_invalid_value() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/person {:user/name 42})"),
        "\"42 - failed: string? in: [:user/name] at: [:user/name] spec: :user/name\\n\""
    );
}

#[test]
fn keys_explain_non_map() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/person [:not :a :map])"),
        "\"[:not :a :map] - failed: map? spec: :user/person\\n\""
    );
}

#[test]
fn keys_explain_unqualified() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req-un [:user/name])");
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/person {:name 42})"),
        "\"42 - failed: string? in: [:name] at: [:name] spec: :user/name\\n\""
    );
}

#[test]
fn keys_explain_data_missing() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_person(&mut eval, "(s/keys :req [:user/name])");
    // `:problems` is a seq, not a vector: `explain*` returns
    // `(apply concat ...)` (oracle 865-877), printed with parens.
    assert_eq!(
        run(&mut eval, "(s/explain-data :user/person {})"),
        "#:clojure.spec.alpha{:problems ({:path [], :pred (clojure.core/fn [%] (clojure.core/contains? % :user/name)), :val {}, :via [:user/person], :in []}), :spec :user/person, :value {}}"
    );
}

// Scenario 20. Oracle: group `pred-forms` keep the `or` head
// (453-460) and abbreviate like scenario 15.
#[test]
fn keys_explain_group() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/a integer?)").unwrap();
    eval.eval_source("(s/def :user/b integer?)").unwrap();
    eval.eval_source("(s/def :user/ab (s/keys :req [(or :user/a :user/b)]))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/ab {})"),
        "\"{} - failed: (or (contains? % :user/a) (contains? % :user/b)) spec: :user/ab\\n\""
    );
}

// Scenario 21. Oracle: `explain-1` 786-791 appends the nested
// spec name to `via` while `path`/`in` extend with the key.
#[test]
fn keys_explain_nested() {
    let mut eval = boot();
    require_spec(&mut eval);
    eval.eval_source("(s/def :user/city string?)").unwrap();
    eval.eval_source("(s/def :user/addr (s/keys :req [:user/city]))")
        .unwrap();
    eval.eval_source("(s/def :user/person (s/keys :req [:user/addr]))")
        .unwrap();
    assert_eq!(
        run(&mut eval, "(s/valid? :user/person {:user/addr {}})"),
        "false"
    );
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/person {:user/addr {}})"),
        "\"{} - failed: (contains? % :user/city) in: [:user/addr] at: [:user/addr] spec: :user/addr\\n\""
    );
}

// ---- keys / form + errors (scenarios 22-23) ----
// Oracle: `describe*` 903-908, macro assert 446-447.

#[test]
fn keys_form_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(s/form (s/keys :req [:user/name] :opt [:user/age]))"
        ),
        "(clojure.spec.alpha/keys :req [:user/name] :opt [:user/age])"
    );
    assert_eq!(
        run(&mut eval, "(s/describe (s/keys :req [:user/name]))"),
        "(keys :req [:user/name])"
    );
}

#[test]
fn keys_rejects_unqualified() {
    let mut eval = boot();
    require_spec(&mut eval);
    let msg = uncaught(&mut eval, "(s/keys :req [:bare])");
    assert!(
        msg.contains("all keys must be namespace-qualified keywords"),
        "unexpected: {msg}"
    );
}

// ---- merge (scenarios 24-29) ----
// Oracle: `merge` macro 503-509, `merge-spec-impl` 1195-1221.

fn def_merge_ab(eval: &mut Evaluator) {
    eval.eval_source("(s/def :user/x integer?)").unwrap();
    eval.eval_source("(s/def :user/y integer?)").unwrap();
    eval.eval_source("(s/def :user/a (s/keys :req [:user/x]))")
        .unwrap();
    eval.eval_source("(s/def :user/b (s/keys :req [:user/y]))")
        .unwrap();
    eval.eval_source("(s/def :user/ab (s/merge :user/a :user/b))")
        .unwrap();
}

#[test]
fn merge_valid_conform() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_merge_ab(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/valid? :user/ab {:user/x 1 :user/y 2})"),
        "true"
    );
    assert_eq!(run(&mut eval, "(s/valid? :user/ab {:user/x 1})"), "false");
    assert_eq!(
        run(
            &mut eval,
            "(= (s/conform :user/ab {:user/x 1 :user/y 2}) {:user/x 1 :user/y 2})"
        ),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/conform :user/ab {:user/x 1})"),
        ":clojure.spec.alpha/invalid"
    );
}

#[test]
fn merge_explain_missing() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_merge_ab(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/ab {:user/x 1})"),
        "\"#:user{:x 1} - failed: (contains? % :user/y) spec: :user/b\\n\""
    );
}

#[test]
fn merge_explain_non_map() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_merge_ab(&mut eval);
    assert_eq!(
        run(&mut eval, "(s/explain-str :user/ab 42)"),
        "\"42 - failed: map? spec: :user/a\\n42 - failed: map? spec: :user/b\\n\""
    );
}

#[test]
fn merge_unform_describe() {
    let mut eval = boot();
    require_spec(&mut eval);
    def_merge_ab(&mut eval);
    assert_eq!(
        run(
            &mut eval,
            "(= (s/unform :user/ab {:user/x 1 :user/y 2}) {:user/x 1 :user/y 2})"
        ),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(s/describe (s/merge :user/a :user/b))"),
        "(merge :user/a :user/b)"
    );
}
