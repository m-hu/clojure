# Gap-122 slice A: map specs (`s/keys` + `s/merge`)

Part of #122 (clojure.spec port). This slice adds map-validating
specs to the embedded `clojure.spec.alpha` (started in
[slice 1](slice-1.md)): the `s/keys` macro + `map-spec-impl`, the
`s/merge` macro + `merge-spec-impl`, and the `pvalid?` helper they
share. `valid?`/`conform`/`unform`/`explain` integrate through the
existing `Spec` protocol; no new protocol methods.

## Oracle pins (text oracle only)

Upstream `clojure/spec/alpha.clj` from
`org.clojure/spec.alpha-0.5.238.jar`
(https://repo1.maven.org/maven2/org/clojure/spec.alpha/0.5.238/spec.alpha-0.5.238.jar,
sha256 `94cd99b6ea639641f37af4860a643b6ed399ee5a8be5d717cff0b663c8d75077`;
inner `clojure/spec/alpha.clj` sha256
`68925dc581d37e034aee2179e3c666e32e513bcc40771ca5497ef600d60f94c1`,
1996 lines). The JVM was NEVER executed for this slice: every
expected value below is derived by reading the upstream source.
(`spec.alpha-0.6.249`'s `alpha.clj` is byte-identical, verified by
diff; the pin stays on 0.5.238 to match slice 1.)

Upstream line references (all in `clojure/spec/alpha.clj`):

- `keys` macro: 414-474 (docstring semantics 414-441, kwargs
  parse + assert 442-451, `parse-req` 453-460, `keys-pred` /
  `pred-exprs` / `pred-forms` 461-466, `map-spec-impl` call 468-474).
- `merge` macro: 503-509.
- `pvalid?`: 779-784. `explain-1`: 786-791.
- `map-spec-impl`: 822-908 (`conform*` 835-849, `unform*` 850-860,
  `explain*` 861-877, `describe*` 903-908).
- `merge-spec-impl`: 1195-1221 (`conform*` 1204-1207, `unform*`
  1208, `explain*` 1209-1212, `describe*` 1221).

## Problem

Slice 1 covers predicate specs and `s/and` only. Realistic specs
are maps: `(s/keys :req [:user/name])` must require keys,
validate each present value against its registered spec,
combine via `(s/merge ...)`, and explain failures with
missing-key (`(contains? % :k)`) and invalid-value (path/in/via)
problem shapes. None of that exists in the port yet.

## Solution

Port the oracle sections above into
`rust/crates/clj-eval/src/spec_alpha.clj`, narrowed to slice 1's
no-generators / no-conformers / no-regex subset:

- `MapSpec` / `MergeSpec` `defrecord`s + `Spec` extensions
  (records carry metadata for `with-name`, as in slice 1;
  `gen*`/`with-gen*` stay unextended).
- `map-spec-impl` takes the upstream argm map minus `:gfn` and
  precomputes `k->s` (`zipmap` of `req-keys`/`opt-keys` to
  `req-specs`/`opt-specs`); `keys->specnames` is
  `(get k->s k k)`.
- `keys` macro built with `list` (no syntax-quote, per slice-1
  rule 1): emits `(fn* [gx] ...)` predicates with fully-qualified
  `clojure.core/contains?` / `clojure.core/map?` /
  `clojure.core/and` heads, and calls
  `clojure.spec.alpha/map-spec-impl` with a map whose data entries
  are quoted and whose `:pred-exprs` / `:keys-pred` entries are
  evaluated fns. `walk/postwalk` is the local `spec-postwalk`.
  `c/assert` is bare `assert` (slice-1 rule 9).
- `merge` macro + `merge-spec-impl` mirror the `and` macro +
  `and-spec-impl` shape (`forms` res'd, `preds` raw).
- `pvalid?` (2- and 3-arity) over the narrowed `dt`.
- `ns` grows `:exclude [and def keys merge]` (upstream excludes
  `[+ * and assert or cat def keys merge]`; the rest belongs to
  sibling slices). Internal code uses `c/keys` / `c/merge`.
- The `keys` docstring drops the `:gen` paragraph (narrowed out;
  a `:gen` kwarg is accepted and ignored). Everything else in the
  docstring holds, including "the values of *all*
  namespace-qualified keys will be validated".

Out of scope (sibling worker): `s/or`, `s/nilable`,
`s/coll-of`, `s/map-of`, `s/every`, regex ops, generators,
conformers, `s/multi-spec`, `s/fspec`/`fdef`, `keys*`.

## BDD scenarios

Conventions: `(s/def :user/name string?)`,
`(s/def :user/age integer?)` unless stated. `=>` is the
pr-string of the last form (or `explain-str` output).

### keys / valid?

1. Required key present and valid
   Given `(s/def :user/person (s/keys :req [:user/name]))`
   When `(s/valid? :user/person {:user/name "n"})` Then `true`.
2. Required key missing
   When `(s/valid? :user/person {})` Then `false`.
3. Required value invalid
   When `(s/valid? :user/person {:user/name 42})` Then `false`.
4. Non-map never valid
   When `(s/valid? :user/person [:a])` Then `false`.
5. Maps are open
   When `(s/valid? :user/person {:user/name "n" :other 1})`
   Then `true` (oracle: `conform*` loops over present entries;
   nothing rejects extras).
6. Optional key absent is fine, present is checked
   Given `(s/keys :req [:user/name] :opt [:user/age])`
   When `{:user/name "n"}` Then `true`;
   When `{:user/name "n" :user/age 3}` Then `true`;
   When `{:user/name "n" :user/age "old"}` Then `false`.
7. All qualified values are validated even when unlisted
   Given `:req [:user/name]` only
   When `{:user/name "n" :user/age "old"}` Then `false`
   (oracle docstring 436-438; `conform*` consults the registry
   for every key via `keys->specnames`).
8. Unqualified keys connect to qualified specs
   Given `(s/keys :req-un [:user/name])`
   When `{:name "n"}` Then `true`;
   When `{:name 42}` Then `false`;
   When `{:user/name "n"}` Then `false` (the required key is
   the unqualified `:name`; oracle 424-434).
9. Optional unqualified keys
   Given `(s/keys :req-un [:user/name] :opt-un [:user/age])`
   When `{:name "n"}` Then `true`;
   When `{:name "n" :age 3}` Then `true`;
   When `{:name "n" :age "old"}` Then `false`.
10. `or` key groups
    Given `(s/keys :req [(or :user/a :user/b)])` with `:user/a`,
    `:user/b` registered as `integer?`
    When `{:user/a 1}` Then `true`; When `{:user/b 2}` Then
    `true`; When `{}` Then `false` (oracle 420-422).
11. `and` key groups
    Given `(s/keys :req [(and :user/a :user/b)])`
    When `{:user/a 1 :user/b 2}` Then `true`;
    When `{:user/a 1}` Then `false`.
12. Empty keys spec
    When `(s/valid? (s/keys) {:a 1})` Then `true`;
    When `(s/valid? (s/keys) 42)` Then `false`.

### keys / conform + unform

13. Conform returns the map, invalid stays invalid
    When `(s/conform :user/person {:user/name "n"})`
    Then `#:user{:name "n"}` (namespaced-map print, as on the
    JVM);
    When `(s/conform :user/person {})`
    Then `:clojure.spec.alpha/invalid`;
    When `(s/conform :user/person {:user/name 42})`
    Then `:clojure.spec.alpha/invalid`;
    When `(s/conform :user/person [:a])`
    Then `:clojure.spec.alpha/invalid`.
14. Unform round-trips
    When `(= (s/unform :user/person {:user/name "n"})
    {:user/name "n"})` Then `true`.

### keys / explain

15. Missing key
    When `(s/explain-str :user/person {})`
    Then `"{} - failed: (contains? % :user/name) spec:
    :user/person\n"` (oracle `explain*` 866-872 + `abbrev`
    `(fn [%] e)` collapse).
16. Invalid value
    When `(s/explain-str :user/person {:user/name 42})`
    Then `"42 - failed: string? in: [:user/name] at:
    [:user/name] spec: :user/name\n"` (oracle `explain*`
    873-876 via `explain-1`, which appends the value spec's
    name to `via`).
17. Non-map value
    When `(s/explain-str :user/person [:not :a :map])`
    Then `"[:not :a :map] - failed: map? spec:
    :user/person\n"` (oracle `explain*` 862-863; `` `map? `` is
    `clojure.core/map?`).
18. Unqualified paths, qualified spec name
    Given scenario 8
    When `(s/explain-str :user/person {:name 42})`
    Then `"42 - failed: string? in: [:name] at: [:name]
    spec: :user/name\n"` (`path`/`in` extend with the present
    key `:name`; `via` with the spec name `:user/name`).
19. Missing-key problem-map shape
    When `(s/explain-data :user/person {})`
    Then `#:clojure.spec.alpha{:problems ({:path [], :pred
    (clojure.core/fn [%] (clojure.core/contains? %
    :user/name)), :val {}, :via [:user/person], :in []}),
    :spec :user/person, :value {}}` (`:problems` is the
    `(apply concat ...)` seq, printed with parens;
    `pred-forms` go through `res`, so `unfn` restores
    `(fn [%] ...)` with resolved symbols).
20. Group failure
    Given scenario 10
    When `(s/explain-str :user/ab {})`
    Then `"{} - failed: (or (contains? % :user/a)
    (contains? % :user/b)) spec: :user/ab\n"`.
21. Nested map specs
    Given `(s/def :user/addr (s/keys :req [:user/city]))`,
    `(s/def :user/city string?)`,
    `(s/def :user/person2 (s/keys :req [:user/addr]))`
    When `(s/explain-str :user/person2 {:user/addr {}})`
    Then `"{} - failed: (contains? % :user/city) in:
    [:user/addr] at: [:user/addr] spec: :user/addr\n"`.

### keys / form + errors

22. Form and describe
    When `(s/form (s/keys :req [:user/name] :opt [:user/age]))`
    Then `(clojure.spec.alpha/keys :req [:user/name] :opt
    [:user/age])`;
    When `(s/describe (s/keys :req [:user/name]))`
    Then `(keys :req [:user/name])` (oracle `describe*`
    903-908).
23. Unqualified keys are rejected
    When `(s/keys :req [:bare])` Then `AssertionError` with
    `all keys must be namespace-qualified keywords`
    (oracle 446-447).

### merge

24. Merge requires every branch
    Given `(s/def :user/a (s/keys :req [:user/x]))`,
    `(s/def :user/b (s/keys :req [:user/y]))`,
    `(s/def :user/x integer?)`, `(s/def :user/y integer?)`,
    `(s/def :user/ab (s/merge :user/a :user/b))`
    When `(s/valid? :user/ab {:user/x 1 :user/y 2})` Then
    `true`; When `(s/valid? :user/ab {:user/x 1})` Then
    `false`.
25. Merge conforms to the union map
    When `(= (s/conform :user/ab {:user/x 1 :user/y 2})
    {:user/x 1 :user/y 2})` Then `true` (oracle `conform*`
    1204-1207: each branch conforms the whole map, results
    `merge`d);
    When `(s/conform :user/ab {:user/x 1})`
    Then `:clojure.spec.alpha/invalid`.
26. Merge explains every branch
    When `(s/explain-str :user/ab {:user/x 1})`
    Then `"#:user{:x 1} - failed: (contains? % :user/y)
    spec: :user/b\n"` (oracle `explain*` 1209-1212: concat
    over branches; the passing branch contributes nothing).
27. Merge over a non-map explains per branch
    When `(s/explain-str :user/ab 42)`
    Then `"42 - failed: map? spec: :user/a\n42 - failed:
    map? spec: :user/b\n"`.
28. Merge unforms to the union map
    When `(= (s/unform :user/ab {:user/x 1 :user/y 2})
    {:user/x 1 :user/y 2})` Then `true` (oracle `unform*`
    1208: reverse-order merge).
29. Merge describe
    When `(s/describe (s/merge :user/a :user/b))`
    Then `(merge :user/a :user/b)` (oracle `describe*` 1221).

## TDD plan

Red-green in `rust/crates/clj-eval/tests/spec_maps.rs`
(harness mirrors `spec_alpha.rs`; each test cites the oracle
lines above):

1. RED: scenarios 1-4, 13 (basic `keys` valid?/conform) ->
   GREEN: `keys` macro + `MapSpec` + `map-spec-impl` +
   `pvalid?`, `ns` exclude grows `keys`.
2. RED: scenarios 6-9, 12 (opt/un families, open maps) ->
   GREEN: covered by step 1's `k->s` (no new code expected;
   if red, fix `opt-keys`/`opt-specs` wiring).
3. RED: scenarios 10-11, 20 (key groups) -> GREEN:
   `parse-req` group postwalk (part of step 1, pinned here).
4. RED: scenarios 15-19, 21 (explain shapes) -> GREEN:
   `MapSpec explain*` + `pred-forms` via `res`/`unfn`.
5. RED: scenarios 22-23 (form/describe/assert) -> GREEN:
   `describe*`, macro assert.
6. RED: scenarios 24-29 (`merge`) -> GREEN: `merge` macro +
   `MergeSpec` + `merge-spec-impl`, `ns` exclude grows
   `merge`.
7. Full gates: `cargo fmt --check`, `cargo clippy` clean,
   `clj-eval` suite green, llvm-cov floor holds (no non-test
   Rust code added), complexity ratchet holds (Rust
   test-only additions).

## Deviations from upstream (all by narrowing)

1. No `:gen` support (`map-spec-impl` drops `:gfn`;
   `gen*`/`with-gen*` unextended, as in slice 1).
2. `Exception.` reads `ex-info` (slice-1 rule 5); the keys
   assert path is unaffected (bare `assert`, slice-1 rule 9).
3. `fn-sym` folds to nil (slice-1 rule 4): only affects
   bare-fn `:pred` forms, none of which this slice adds.
