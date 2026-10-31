# Gap-122 slice D2: remaining data specs (`s/tuple` + `s/multi-spec` + `s/conformer` + `s/nonconforming`)

Part of #122 (clojure.spec port). This slice adds the remaining
data-shape specs to the embedded `clojure.spec.alpha` (started in
[slice 1](slice-1.md)): the `s/tuple` macro + `tuple-impl`, the
`s/multi-spec` macro + `multi-spec-impl` (multimethod integration),
the `s/conformer` macro (+ `spec-impl`/`dt` conformer machinery, which
makes the existing `s/unform` meaningful), and `s/nonconforming`.
`valid?`/`conform`/`unform`/`explain` integrate through the existing
`Spec` protocol; no new protocol methods.

## Oracle pins (text oracle only)

Upstream `clojure/spec/alpha.clj` from
`org.clojure/spec.alpha-0.6.249.jar`
(`~/.m2/repository/org/clojure/spec.alpha/0.6.249/`,
inner `clojure/spec/alpha.clj` sha256
`68925dc581d37e034aee2179e3c666e32e513bcc40771ca5497ef600d60f94c1`,
1996 lines — byte-identical to the `0.5.238` file pinned by slice 1).
The JVM was NEVER executed for this slice: every expected value below
is derived by reading the upstream source.

Upstream line references (all in `clojure/spec/alpha.clj`):

- `multi-spec` macro: 385-412 (expansion at 412).
- `conformer` macro: 666-672. `tuple` macro: 694-700.
- `dt`: 757-768 (2 arities; `cpred?` branch at 764).
- `pvalid?`: 779-784. (`explain-1`: 786-791, already ported.)
- `spec-impl`: 913-944 (2 arities; `cpred?`/`unc` at 928-935).
- `multi-spec-impl`: 946-994 (`conform*` 962-964, `unform*`
  965-967, `explain*` 968-974, `describe*` 994).
- `tuple-impl`: 996-1059 (`conform*` 1007-1023, `unform*`
  1024-1033, `explain*` 1034-1051, `describe*` 1059).
- `nonconforming`: 1813-1832.

There is NO `s/invalid` var upstream (only the `::invalid` keyword
and the `invalid?` predicate, both already ported): it is
deliberately not added, and a test pins its absence.

## Problem

Slice 1 covers predicate specs and `s/and` only. Fixed-shape
vectors (`(s/tuple int? string?)`), dispatch-driven specs
(`(s/multi-spec event-type :event/type)`), value-converting specs
(`(s/conformer parse unparse)`), and conform-suppressing wrappers
(`(s/nonconforming spec)`) do not exist in the port yet. Without
the conformer machinery, `s/unform` is also a no-op for every
spec.

## Solution

Port the oracle sections above into
`rust/crates/clj-eval/src/spec_alpha.clj`, narrowed to slice 1's
no-generators / no-regex subset (every `gen*`/`with-gen*` stays
unextended, as in slice 1):

- `TupleSpec` / `MultiSpec` / `NonconformingSpec` `defrecord`s +
  `Spec` extensions (records carry metadata for `with-name`, as in
  slice 1). `tuple-impl`/`multi-spec-impl` mirror the
  `and-spec-impl` shape (`forms` res'd, `preds` raw, specs resolved
  through a `delay`).
- `PredSpec` grows `cpred?`/`unc` fields and `spec-impl` grows
  `[form pred cpred?]` / `[form pred cpred? unc]` arities (the
  upstream `[form pred gfn cpred?]` / `[form pred gfn cpred? unc]`
  arities minus `:gen`, which belongs to the generators slice);
  the existing `[form pred]` arity delegates with `nil`s, so
  slice-1 behavior is unchanged.
- `dt` grows the upstream 4-arity `[pred x form cpred?]` (the
  3-arity delegates with `nil`); `pvalid?` is ported verbatim
  (modulo the written-out `:clojure.spec.alpha/unknown` keyword —
  no `::` auto-resolve, per the slice-1 batch-read rule).
- `tuple` / `multi-spec` / `conformer` macros are built with
  `list` (no syntax-quote, per slice-1 rule 1) and bare `assert`
  (slice-1 rule 9). `nonconforming` is a plain fn, as upstream.
- `multi-spec-impl` resolves methods through
  `clojure.core/get-method` on the dispatch value, exactly like
  the upstream `.getMethod` check (hierarchy-aware, `nil` on
  miss, throws on ambiguity — no try/catch, so method-body throws
  still propagate).
- One embedding-glue native is REQUIRED:
  `clojure.core/__dispatch_fn*` (`rust/crates/clj-eval/src/multi.rs`,
  next to `__multifn*`): the port exposes no accessor for a
  multifn's dispatch fn, and upstream `multi-spec-impl` applies
  `.dispatchFn` on every path (`predx`, `dval`, the `unform*`
  error). Without it the explain path could not conj the dispatch
  value (an upstream-documented behavior: "the dispatch values of
  the multimethod will be included in the path"). The `__*`
  hidden-native shape follows the existing `__multifn*` /
  `__defmethod*` precedent; it is covered by the new tests like
  any other native.
- `retag` is accepted and carried into `describe` but otherwise
  ignored (it is only consumed by `gen*`, out of scope) — the
  same accept-and-ignore narrowing slice A uses for `:gen`.

Out of scope (sibling workers): `s/keys`, `s/merge`, `s/or`,
`s/nilable`, `s/coll-of`, `s/map-of`, `s/every`, regex ops
(`cat`/`alt`/`*`/`+`/`?`/`&`/`keys*`), generators, `s/fspec` /
`fdef`, instrumentation. OVERLAP NOTE for integration: `pvalid?`
is also needed by slice A (`keys`/`merge`); whichever slice lands
second drops its copy (the definitions are identical).

## Port adaptations (all pinned by tests)

1. `IllegalStateException` construction (the conformer "no unform
   fn" throw, oracle line 935; the multi-spec "No method of"
   throw, oracle line 967) is `ex-info` with the same message
   (same adaptation as slice 1's `Exception.` → `ex-info`;
   still caught by `catch Exception`).
2. Upstream `.getMethod`/`.dispatchFn` interop becomes
   `clojure.core/get-method` + the `__dispatch_fn*` glue native
   (same dispatch-table semantics, JVM ambiguity error included).
3. `nonconforming` "will specize regex ops" (oracle docstring)
   does not hold yet: regex input still hits the slice-1 guard
   (regex specs are a sibling slice).
4. `tuple-impl`/`multi-spec-impl`/`spec-impl` drop the `gfn`
   parameter (generators slice); `gen*`/`with-gen*` stay
   unextended on the new records.

## BDD scenarios

Conventions: `(require '[clojure.spec.alpha :as s])` as in the
slice-1 tests. `=>` is the pr-string of the last form (or
`explain-str` output). Multi-specs use
`(defmulti mspec :tag)` / `(defmulti num-kind tag-of ...)`-style
multimethods defined in the test setup.

### tuple / conform / valid?

1. Element-wise conform
   Given `(s/def :user/pair (s/tuple integer? string?))`
   When `(s/conform :user/pair [1 "a"])` Then `[1 "a"]`.
2. Non-vector is invalid
   When `(s/valid? :user/pair '("a" 1))` Then `false`;
   `(s/valid? :user/pair {:a 1})` Then `false`.
3. Wrong count is invalid
   When `(s/valid? :user/pair [1])` Then `false`;
   When `(s/valid? :user/pair [1 "a" :x])` Then `false`.
4. Element failure is invalid
   When `(s/valid? :user/pair [1 2])` Then `false`;
   When `(s/valid? :user/pair ["a" "b"])` Then `false`.
5. Conform threads through converting elements
   Given `(s/def :user/conv (s/tuple (s/conformer inc dec) string?))`
   When `(s/conform :user/conv [1 "a"])` Then `[2 "a"]`.
6. Unchanged conform is identical (oracle `identical?` fast path)
   When `(identical? v (s/conform :user/pair v))` for
   `v = [1 "a"]` Then `true`.
7. Forward registry references resolve lazily
   Given `(s/def :user/late (s/tuple :user/later string?))` built
   BEFORE `:user/later` exists, then
   `(s/def :user/later integer?)`
   When `(s/valid? :user/later ...)` — i.e.
   `(s/valid? :user/late [1 "a"])` Then `true`;
   `(s/valid? :user/late ["a" "b"])` Then `false`.
8. Zero preds rejected at macroexpansion
   When `(s/tuple)` Then `AssertionError` (`Assert failed: ...`).

### tuple / unform / explain / describe

9. Unform reverses per element
   Given scenario 5 When `(s/unform :user/conv [2 "a"])`
   Then `[1 "a"]`.
10. Unform asserts shape
    When `(s/unform :user/pair [1])` Then `AssertionError`.
11. Explain non-vector names `vector?`
    `(s/explain-data :user/pair '("a" 1))` has the single problem
    `{:path [] :pred clojure.core/vector? :val ("a" 1)
    :via [:user/pair] :in []}`.
12. Explain wrong count names the count form
    `(s/explain-data :user/pair [1])` has the single problem
    `{:path [] :pred (clojure.core/= (clojure.core/count %) 2)
    :val [1] :via [:user/pair] :in []}`.
13. Explain element failure carries ordinal path/in
    `(s/explain-data :user/pair [1 2])` has the single problem
    `{:path [1] :pred clojure.core/string? :val 2
    :via [:user/pair] :in [1]}`; `explain-str` renders
    `"2 - failed: string? in: [1] at: [1] spec: :user/pair\n"`.
14. describe/form
    `(s/form :user/pair)` =>
    `(clojure.spec.alpha/tuple clojure.core/integer?
    clojure.core/string?)`; `(s/describe :user/pair)` =>
    `(tuple integer? string?)`.

### multi-spec

15. Dispatch conforms through the winning method spec
    Given `(defmulti num-kind (fn [x] (if (integer? x) :int :str)))`,
    `(defmethod num-kind :int [_] integer?)`,
    `(defmethod num-kind :str [_] string?)`,
    `(s/def :user/nk (s/multi-spec num-kind :tag))`
    When `(s/valid? :user/nk 5)` Then `true`;
    `(s/valid? :user/nk "a")` Then `true`;
    `(s/valid? :user/nk 5.5)` Then `false` (dispatched `:str`,
    `string?` fails).
16. Keyword dispatch over maps
    Given `(defmulti mspec :tag)`,
    `(defmethod mspec :m [_] map?)`,
    `(s/def :user/ms (s/multi-spec mspec :tag))`
    When `(s/valid? :user/ms {:tag :m})` Then `true`.
17. Missing method is invalid with a "no method" reason
    When `(s/valid? :user/ms {:tag :nope})` Then `false`;
    `(s/explain-data :user/ms {:tag :nope})` has the single
    problem `{:path [:nope] :pred user/mspec :val {:tag :nope}
    :reason "no method" :via [:user/ms] :in []}`;
    `explain-str` renders
    `"{:tag :nope} - failed: no method at: [:nope] spec: :user/ms\n"`.
18. Explain path includes the dispatch value on method failure
    Given `(defmulti tv (fn [x] (if (vector? x) :vec :other)))`,
    `(defmethod tv :vec [_] (s/tuple integer? string?))`,
    `(defmethod tv :other [_] string?)`,
    `(s/def :user/tv (s/multi-spec tv :tag))`
    When `(s/explain-data :user/tv [1 2])` the single problem has
    `:path [:vec 1]` and `:in [1]`.
19. Unform dispatches too
    Given `(defmulti cf (fn [x] :num))`,
    `(defmethod cf :num [_] (s/conformer inc dec))`,
    `(s/def :user/cf (s/multi-spec cf :tag))`
    When `(s/conform :user/cf 5)` Then `6`;
    When `(s/unform :user/cf 6)` Then `5`.
20. Unform with no method throws naming form and dispatch value
    When `(s/unform :user/ms {:tag :nope})` Then `ExceptionInfo`
    `No method of: user/mspec for dispatch value: :nope`
    (adaptation 1: `ex-info`, same message).
21. Late method installation is picked up (fresh deref)
    Given `(s/def :user/lm (s/multi-spec late-mm :tag))` built
    before any method exists, then
    `(defmethod late-mm :a [_] integer?)`
    When `(s/valid? :user/lm ...)` — dispatching `:a` — Then the
    new method applies (no stale miss).
22. form/describe
    `(s/form :user/ms)` =>
    `(clojure.spec.alpha/multi-spec user/mspec :tag)`;
    `(s/describe :user/ms)` => `(multi-spec mspec :tag)` (oracle
    `abbrev` strips the namespace).

### conformer (+ unform)

23. Conform applies f, invalid passes through
    Given `(defn keep-int [x] (if (integer? x) x
    :clojure.spec.alpha/invalid))`,
    `(s/def :user/ki (s/conformer keep-int))`
    When `(s/conform :user/ki 5)` Then `5`;
    When `(s/conform :user/ki "a")`
    Then `:clojure.spec.alpha/invalid`;
    When `(s/valid? :user/ki "a")` Then `false`.
24. Falsy-but-valid conform (oracle: only `::invalid` fails)
    Given `(s/def :user/ff (s/conformer (fn [_] false)))`
    When `(s/conform :user/ff :anything)` Then `false`;
    When `(s/valid? :user/ff :anything)` Then `true`.
25. Unform reverses with unf, throws without
    Given `(s/def :user/ud (s/conformer inc dec))`
    When `(s/unform :user/ud 6)` Then `5`;
    Given `(s/def :user/nu (s/conformer inc))`
    When `(s/unform :user/nu 6)` Then `ExceptionInfo`
    `no unform fn for conformer` (adaptation 1).
26. Conform threads through `s/and`
    Given `(s/def :user/ac (s/and (s/conformer inc dec) #(< % 10)))`
    When `(s/conform :user/ac 5)` Then `6`;
    When `(s/conform :user/ac 50)` Then invalid (conforms to 51,
    fails `< 10`); When `(s/unform :user/ac 6)` Then `5`.
27. Explain names the conformer form
    `(s/explain-data :user/ki "a")` has the single problem with
    `:pred (clojure.spec.alpha/conformer user/keep-int)` and
    `:via [:user/ki]`; `explain-str` renders
    `"\"a\" - failed: (conformer keep-int) spec: :user/ki\n"`.
28. describe/form
    `(s/describe :user/ud)` => `(conformer inc dec)`;
    `(s/form :user/ud)` =>
    `(clojure.spec.alpha/conformer clojure.core/inc clojure.core/dec)`.

### nonconforming

29. Conform returns the original value
    Given `(s/def :user/nc (s/nonconforming (s/conformer inc dec)))`
    When `(s/conform :user/nc 5)` Then `5` (not `6`);
    When `(s/valid? :user/nc 5)` Then `true`.
30. Invalid stays invalid
    Given `(s/def :user/nk2 (s/nonconforming
    (s/conformer keep-int)))` (scenario 23's fn)
    When `(s/conform :user/nk2 "a")`
    Then `:clojure.spec.alpha/invalid`.
31. Unform is identity
    When `(s/unform :user/nc 6)` Then `6`.
32. Explain delegates to the inner spec
    `(s/explain-data :user/nk2 "a")` equals
    `(s/explain-data (s/get-spec :user/nk2-inner) "a")`-shaped
    problems — i.e. the single problem has
    `:pred (clojure.spec.alpha/conformer user/keep-int)`.
33. describe wraps the inner description
    `(s/describe :user/nc)` =>
    `(nonconforming (conformer inc dec))`.

### invalid

34. No `s/invalid` var (oracle has none)
    When `(nil? (resolve 'clojure.spec.alpha/invalid))`
    Then `true`.

## TDD

- RED: `rust/crates/clj-eval/tests/spec_data_remainder.rs` pins
  every scenario above (helpers mirror `spec_alpha.rs`: `run`,
  `uncaught`, `require_spec`). New `__dispatch_fn*` native paths
  (happy path, arity error, non-multifn cast error) are covered
  by multi-spec tests plus two error tests, keeping the llvm-cov
  floor at 100.
- GREEN: extend `spec_alpha.clj` (PredSpec `cpred?`/`unc`,
  `spec-impl`/`dt` arities, `pvalid?`, `tuple`/`tuple-impl`,
  `multi-spec`/`multi-spec-impl`, `conformer`, `nonconforming`)
  + the `__dispatch_fn*` glue native.
- Gates before push: `cargo fmt --check`, `cargo clippy` clean,
  `clj-eval` suite green (slice-1 tests untouched and passing),
  llvm-cov floor holds, complexity ratchet holds.
