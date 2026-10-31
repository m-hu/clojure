# Gap-122 slice E: regex specs (`s/cat`/`s/alt`/`s/*`/`s/+`/`s/?`/`s/&` + `s/keys*`)

Part of #122 (clojure.spec port). This slice adds regex
(sequential) specs to the embedded `clojure.spec.alpha` (started in
[slice 1](slice-1.md)): the six regex-op macros, the op
constructors, the derivative-matching engine
(`accept-nil?`/`preturn`/`deriv`/`add-ret`), `re-conform` /
`re-explain` / `op-unform` / `op-describe` / `op-explain`, the
`regex-spec-impl` Spec glide, and the `s/keys*` macro.
`valid?`/`conform`/`unform`/`explain`/`form`/`describe` integrate
through the existing `Spec` protocol; no new protocol methods.

## Oracle pins (text oracle only)

Upstream `clojure/spec/alpha.clj` from
`org.clojure/spec.alpha-0.6.249.jar`
(https://repo1.maven.org/maven2/org/clojure/spec.alpha/0.6.249/spec.alpha-0.6.249.jar,
sha256
`022d864baa5d943c19ed9d5858f1909a66029ccaf88396e391a5236f6e5c5364`;
inner `clojure/spec/alpha.clj` sha256
`68925dc581d37e034aee2179e3c666e32e513bcc40771ca5497ef600d60f94c1`,
1996 lines). The JVM was NEVER executed for this slice: every
expected value below is derived by reading the upstream source.
(The inner-file sha256 matches the 0.5.238 pin published in
[slice A](slice-A.md), corroborating identical content across
versions; line numbers below are the same in both.)

Upstream line references (all in `clojure/spec/alpha.clj`):

- ns `:exclude`: 17 (`[+ * and assert or cat def keys merge]`).
- `maybe-spec` regex wrap: 110-119.
- `*` `+` `?` `alt` `cat` `&` macros: 607-665.
- `tagged-ret`: 1058-1059. `and-preds`: 1128-1137.
- `spec-impl` regex branch: 919.
- `accept`/`accept?`: 1380-1383. `pcat*`/`pcat`: 1385-1395.
- `cat-impl`: 1397-1400. `rep*`: 1402-1407. `rep-impl`:
  1409-1411. `rep+impl`: 1413-1416. `amp-impl`: 1418-1421.
- `filter-alt`: 1423-1430. `alt*`: 1432-1442. `alts`/`alt2`:
  1444-1445. `alt-impl`: 1447-1449. `maybe-impl`: 1451-1453.
- `noret?`: 1455-1459. `accept-nil?`: 1463-1473 (`declare
  preturn` 1461). `preturn`: 1477-1490 (`declare add-ret`
  1475). `op-unform`: 1492-1511. `add-ret`: 1513-1524.
- `deriv`: 1526-1544. `op-describe`: 1546-1560. `op-explain`:
  1562-1615. `re-gen`: 1617-1655 (STUBBED, see below).
- `re-conform`: 1658-1669. `re-explain`: 1671-1695.
- `regex-spec-impl`: 1697-1723.
- `::kvs->map`: 1792 (NOT ported: needs `conformer`, slice D).
- `keys*` macro: 1794-1812.

Recon: `docs/gap-122/recon-fdef-generators.md` §1d + §2 option E
(this slice), on branch `gap-122-recon-fdef`.

## Problem

Slice 1 covers predicate specs and `s/and` only, and fails loudly
on anything regex-shaped (`maybe-spec` / `spec-impl` throw
`regex specs are out of scope in gap-122 slice 1`). Sequential
specs do not exist: `(s/cat :e even? :o odd?)`,
`(s/alt :e even? :o odd?)`, `(s/* integer?)`, `(s/+ integer?)`,
`(s/? integer?)`, `(s/& re pred ...)` are all unresolvable, so
`(s/conform (s/cat :e even?) [4])` cannot work, nor can
`describe` / `explain` of sequential data, nor the `s/keys*`
regex op over key/value sequences.

## Solution

Port the oracle sections above into
`rust/crates/clj-eval/src/spec_alpha.clj`, narrowed to slice 1's
no-generators subset (`re-gen` is a throwing stub; `::kvs->map`
and `keys`/`conformer`/`with-gen`/`gen` belong to sibling slices):

- Regex-op macros `*` `+` `?` `alt` `cat` `&` after the `and`
  macro (mirroring upstream macro-before-impl order), built with
  `list` (no syntax-quote, slice-1 rule 1), fully-qualified
  `-impl` heads, `res`'d forms quoted. `c/assert` in the `alt` /
  `cat` bodies is bare `assert` (slice-1 rule 9); the `c/and`
  inside stays (bare `and` would hit spec's own macro).
- `ns` grows `:exclude [and def + * cat]` (upstream line 17
  excludes `+ * ... cat ...`; `?` / `alt` / `&` / `keys*` need
  no exclude — no core clash — same as upstream).
- `maybe-spec` regex branch becomes the upstream wrap
  (`(with-name (regex-spec-impl s) (spec-name s))`); `spec-impl`
  regex branch becomes `(regex-spec-impl pred)`; the top
  `declare` gains `regex-spec-impl`.
- Engine at end of file (after `and-spec-impl`):
  `tagged-ret`, `and-preds`, `accept` / `accept?` / `pcat*` /
  `pcat` / `cat-impl` / `rep*` / `rep-impl` / `rep+impl` /
  `amp-impl` / `filter-alt` / `alt*` / `alts` / `alt2` /
  `alt-impl` / `maybe-impl` / `noret?` / `accept-nil?` /
  `preturn` / `op-unform` / `add-ret` / `deriv` /
  `op-describe` / `op-explain` / `re-conform` / `re-explain`,
  then `RegexSpec` + `regex-spec-impl`, then `keys*` last
  (mirroring upstream). `case` ports verbatim (keyword + `nil`
  + grouped-list tests verified supported); `(declare
  preturn)` / `(declare add-ret)` mirror upstream.
- `RegexSpec` is a `defrecord` (metadata-carrying for
  `with-name`, slice-1 rule 2). `regex-spec-impl` narrows to
  `[re]` (gfn dropped, slice-1 rule 8 precedent).
- `re-gen` keeps its upstream `[p overrides path rmap f]`
  signature but throws `(ex-info "generators slice pending"
  {})`; `RegexSpec` `gen*` calls it (upstream call shape, so
  slice F's diff is the body) and `with-gen*` throws the same
  stub (both gen paths fail honestly until F threads `gfn`).
- The non-sequential `explain*` `:pred` is the literal form
  `(clojure.core/fn [%] (clojure.core/or (clojure.core/nil? %)
  (clojure.core/sequential? %)))` — the deterministic value of
  upstream's `(res `#(c/or (nil? %) (sequential? %)))`
  (syntax-quote qualifies, `unfn` restores `(fn [%] ...)`,
  `res` re-resolves `fn`/`or`/`nil?`/`sequential?` via core
  refers); it abbreviates to `(or (nil? %) (sequential? %))`.
- `keys*` mirrors the upstream expansion with `list`: a
  `gensym`'d `mspec` binding over a
  `(clojure.spec.alpha/keys ...)` call with the raw kspecs, a
  `(clojure.spec.alpha/with-gen (clojure.spec.alpha/& ...))`
  body whose inner `*` / `cat` use fully-qualified heads and
  `:clojure.spec.alpha/k` / `:clojure.spec.alpha/v`, the
  `:clojure.spec.alpha/kvs->map` ident, and a `(fn [] ...)`
  gen fn over `clojure.spec.gen.alpha/fmap` /
  `clojure.spec.gen.alpha/gen`. `let` / `fn` / `apply` /
  `concat` / `keyword?` / `any?` stay bare (resolved at the use
  site; the port only treats the unqualified `let` spelling as
  the special form). Runtime use needs slices A (`keys`) + D
  (`conformer` for `::kvs->map`) + F (`with-gen`/`gen`), so
  only the expansion shape is pinned here.

Out of scope (sibling workers): data specs (`or` / `keys` /
`merge` / `every` / `coll-of` / `map-of` / `tuple` /
`multi-spec` / `conformer` / `nilable` / `nonconforming`),
generators (`re-gen` body, `gen*` wiring, `s/gen`,
`s/exercise`, `with-gen`), `fspec` / `fdef` / instrument /
`check`, `core.specs`. No embedding glue is needed (`ns.rs`
already serves `spec_alpha.clj`; no new namespaces).

## BDD scenarios

Conventions: `=>` is the pr-string of the last form (or
`explain-str` output, or `Class | message` for throws). `s` is
the `clojure.spec.alpha` alias. Every value is derived from the
oracle lines above; derivations cite the mechanism.

### Macros / expansion + asserts

1. `*` expands
   When `(= (macroexpand-1 '(s/* even?))
   '(clojure.spec.alpha/rep-impl (quote clojure.core/even?)
   even?))` Then `true` (oracle 607-611: form `res`'d and
   quoted, pred embedded raw).
2. `+` expands
   When `(= (macroexpand-1 '(s/+ even?))
   '(clojure.spec.alpha/rep+impl (quote clojure.core/even?)
   even?))` Then `true` (oracle 613-617).
3. `?` expands
   When `(= (macroexpand-1 '(s/? even?))
   '(clojure.spec.alpha/maybe-impl even? (quote
   clojure.core/even?)))` Then `true` (oracle 619-623; note
   the `[p form]` arg order).
4. `cat` expands
   When `(= (macroexpand-1 '(s/cat :e even? :o odd?))
   '(clojure.spec.alpha/cat-impl [:e :o] [even? odd?]
   (quote [clojure.core/even? clojure.core/odd?])))` Then
   `true` (oracle 642-656: keys + raw pred-forms + quoted
   `res`'d forms).
5. `alt` expands
   When `(= (macroexpand-1 '(s/alt :e even? :o odd?))
   '(clojure.spec.alpha/alt-impl [:e :o] [even? odd?]
   (quote [clojure.core/even? clojure.core/odd?])))` Then
   `true` (oracle 625-640).
6. `&` expands
   When `(= (macroexpand-1 '(s/& (s/* integer?) vector?))
   '(clojure.spec.alpha/amp-impl (s/* integer?) (quote
   (clojure.spec.alpha/* clojure.core/integer?)) [vector?]
   (quote [clojure.core/vector?])))` Then `true` (oracle
   658-665: re embedded raw, `res`'d re-form quoted, preds
   raw + quoted).
7. `cat` asserts pairing
   When `(s/cat :e)` Then `AssertionError` mentioning `cat
   expects k1 p1 k2 p2..., where ks are keywords` (oracle
   654; fires at macroexpansion; exact port surface pinned by
   the test).
8. `alt` asserts pairing
   When `(s/alt "e" even?)` Then `AssertionError` mentioning
   `alt expects k1 p1 k2 p2..., where ks are keywords`
   (oracle 638).

### cat / valid? + conform + unform

9. Conform destructures
   When `(= (s/conform (s/cat :e even? :o odd?) [4 5]) {:e 4
   :o 5})` Then `true` (oracle `re-conform` 1658-1669 over
   `deriv` 1526-1544; `pcat*` 1385-1394 conjs `{k v}` onto
   `{}`).
10. Element failure stays invalid
    When `(s/conform (s/cat :e even? :o odd?) [5 4])` Then
    `:clojure.spec.alpha/invalid`;
    When `(s/valid? (s/cat :e even? :o odd?) [5 4])` Then
    `false`;
    When `(s/valid? (s/cat :e even? :o odd?) [4 5])` Then
    `true`.
11. Empty input is invalid
    When `(s/valid? (s/cat :e even?) [])` Then `false`
    (`accept-nil?` 1463-1473: `pcat` needs every element).
12. Sequential input, not just vectors
    When `(= (s/conform (s/cat :e even? :o odd?) '(4 5)) {:e
    4 :o 5})` Then `true` (`regex-spec-impl` 1709-1712 takes
    any `sequential?` via `seq`).
13. Non-sequential input is invalid
    When `(s/valid? (s/cat :e even?) {:e 4})` Then `false`;
    When `(s/valid? (s/cat :e even?) 42)` Then `false`.
14. Nil input follows accept-nil
    When `(s/valid? (s/cat :e even?) nil)` Then `false`;
    When `(s/valid? (s/* integer?) nil)` Then `true`
    (`(seq nil)` is empty; `*` accepts nil, `cat` does not).
15. Unform flattens in key order
    When `(s/unform (s/cat :e even? :o odd?) {:e 4 :o 5})`
    Then `(4 5)` (`op-unform` 1492-1511 `mapcat`s one
    single-element vector per key — a seq, not a vector).
16. Conform/unform round-trip
    When `(= (s/unform (s/cat :e even? :o odd?) (s/conform
    (s/cat :e even? :o odd?) [4 5])) [4 5])` Then `true`.
17. Empty cat
    When `(s/conform (s/cat) [])` Then `{}`;
    When `(s/valid? (s/cat) [])` Then `true`;
    When `(some? (s/regex? (s/cat)))` Then `true`
    (`(every? accept-nil? [])` is true; `preturn` of the
    empty `pcat` is `{}`).
18. Nested op as element
    When `(= (s/conform (s/cat :a (s/alt :x integer?)) [5])
    {:a [:x 5]})` Then `true` (`deriv` 1535 recurses into
    the element op; one value conforms through `alt`).

### alt / conform + unform

19. Conform tags
    When `(s/conform (s/alt :e even? :o odd?) [4])` Then
    `[:e 4]` (a `MapEntry`: `tagged-ret` 1058-1059; the
    input is the one-element sequence the branch matches);
    When `(key (s/conform (s/alt :e even? :o odd?) [5]))`
    Then `:o`;
    When `(val (s/conform (s/alt :e even? :o odd?) [5]))`
    Then `5`.
20. No branch matches
    When `(s/conform (s/alt :s string? :k keyword?) [42])`
    Then `:clojure.spec.alpha/invalid`;
    When `(s/conform (s/alt :e even? :o odd?) [4.5])` Then
    `java.lang.IllegalArgumentException | Argument must be
    an integer: 4.5` (throwing preds propagate out of
    `deriv`, as upstream: `dt` calls the pred directly, so
    `even?` on a non-integer throws instead of failing).
21. Unform unwraps the tag
    When `(s/unform (s/alt :e even? :o odd?) [:e 4])` Then
    `[4]` (`op-unform` 1508-1511 destructures `[k v]` and
    returns the branch unform — a vector).
22. Bare `(s/alt)` is not a regex
    When `(s/regex? (s/alt))` Then `nil` (verbatim upstream
    quirk: `alt*` 1432-1442 of nothing is `nil`, so
    `alt-impl` 1447-1449 yields `{:id ...}` with no `::op`).

### `*` / `+` / `?` / conform + unform

23. `*` collects, empty is `[]`
    When `(s/conform (s/* integer?) [1 2 3])` Then `[1 2
    3]`;
    When `(s/conform (s/* integer?) [])` Then `[]`;
    When `(s/valid? (s/* integer?) [])` Then `true`;
    When `(s/conform (s/* integer?) [1 :a])` Then
    `:clojure.spec.alpha/invalid` (`rep*` 1402-1407 with
    `identical? p1 p2` accepts nil, returning `[]`).
24. `+` needs one
    When `(s/conform (s/+ integer?) [7])` Then `[7]`;
    When `(s/conform (s/+ integer?) [])` Then
    `:clojure.spec.alpha/invalid` (`rep+impl` 1413-1416 is a
    `pcat` of the pred plus the `rep`).
25. `?` is single-or-nil
    When `(s/conform (s/? integer?) [])` Then `nil`
    (`preturn` of the `::nil` accept is `nil` per
    `re-conform` 1662-1666);
    When `(s/conform (s/? integer?) [1])` Then `1`;
    When `(s/conform (s/? integer?) [1 2])` Then
    `:clojure.spec.alpha/invalid`.
26. Rep unforms
    When `(s/unform (s/* integer?) [1 2])` Then `(1 2)`
    (`mapcat` — a seq);
    When `(s/unform (s/+ integer?) [1 2])` Then `(1 2)`
    (the `rep+` `pcat` branch 1499-1500);
    When `(s/unform (s/? integer?) 5)` Then `[5]` (the
    `maybe` `alt` branch 1507);
    When `(s/unform (s/? integer?) nil)` Then `[nil]`.

### `&` / conform + unform

27. `&` filters
    When `(s/conform (s/& (s/* integer?) #(> (count %) 2))
    [1 2 3])` Then `[1 2 3]` (`deriv` 1534-1539 runs
    `and-preds` 1128-1137 over the `re` result);
    When `(s/conform (s/& (s/* integer?) #(> (count %) 2))
    [1])` Then `:clojure.spec.alpha/invalid`.
28. `&` unform unforms preds then `re`
    When `(s/unform (s/& (s/* integer?) vector?) [1 2])`
    Then `(1 2)` (`op-unform` 1496-1498 reduces `unform`
    over the reversed preds, then unforms `p1`).

### form / describe

29. cat form + describe
    When `(s/form (s/cat :e even? :o odd?))` Then
    `(clojure.spec.alpha/cat :e clojure.core/even? :o
    clojure.core/odd?)`;
    When `(s/describe (s/cat :e even? :o odd?))` Then
    `(cat :e even? :o odd?)` (`op-describe` 1546-1560 +
    `abbrev`).
30. Other describes
    When `(s/describe (s/alt :e even? :o odd?))` Then
    `(alt :e even? :o odd?)`;
    When `(s/describe (s/* even?))` Then `(* even?)`;
    When `(s/describe (s/+ integer?))` Then `(+ integer?)`;
    When `(s/describe (s/? odd?))` Then `(? odd?)`;
    When `(s/describe (s/& (s/* integer?) #(> (count %)
    1)))` Then `(& (* integer?) (> (count %) 1))`
    (`list*` of `&`, the `res`'d re-form, and the pred
    forms).

### explain

31. cat element failure
    When `(s/explain-str (s/cat :e even? :o odd?) [5 4])`
    Then `"5 - failed: even? in: [0] at: [:e]\n"`
    (`re-explain` 1671-1695 conjs the failing index onto
    `:in`; `op-explain` 1586-1597 picks the first
    non-accept-nil element and conjs its key onto `:path`).
32. Insufficient input
    When `(s/explain-str (s/cat :e even? :o odd?) [4])`
    Then `"() - failed: Insufficient input at: [:o]\n"`
    (the `insufficient` problem 1569-1575: `:val ()`,
    empty `:in`, `:path` extended with `:o`).
33. Extra input
    When `(s/explain-str (s/cat :e even?) [4 5])` Then
    `"(5) - failed: Extra input in: [1]\n"` (the accept
    with leftover data 1682-1688: `:val` is the remaining
    seq, `:in [1]`).
34. Non-sequential input
    When `(s/explain-str (s/cat :e even?) 42)` Then `"42 -
    failed: (or (nil? %) (sequential? %))\n"`
    (`regex-spec-impl` `explain*` 1714-1717; the `:pred`
    abbreviates as derived in Solution).
35. alt explains every branch
    When `(s/explain-str (s/alt :s string? :k keyword?)
    [42])` Then `"42 - failed: string? in: [0] at:
    [:s]\n42 - failed: keyword? in: [0] at: [:k]\n"`
    (`op-explain` 1598-1608 concats one problem per branch,
    in order; `string?`/`keyword?` never throw, unlike
    `even?`/`odd?` on non-integers — see scenario 20).
36. `*` mid-sequence failure
    When `(s/explain-str (s/* integer?) [1 :a])` Then `":a
    - failed: integer? in: [1]\n"` (the `rep` branch
    1609-1612 re-explains the identical-`p1` case with the
    bare pred form).
37. `&` predicate failure
    When `(s/explain-str (s/& (s/cat :e even?) #(> (:e %)
    10)) [4])` Then `"{:e 4} - failed: (> (:e %) 10) in:
    [0]\n"` (the `amp` branch 1579-1585 runs
    `explain-pred-list` over the `re` result).
38. Registered name suffix + doubled via
    Given `(s/def :user/n (s/cat :e even?))`
    When `(s/explain-str :user/n [5])` Then `"5 - failed:
    even? in: [0] at: [:e] spec: :user/n\n"`;
    When `(:via (first (:clojure.spec.alpha/problems
    (s/explain-data :user/n [5]))))` Then `[:user/n
    :user/n]` (verbatim upstream: `op-explain` 1566 conj's
    the op's own `:name` onto the caller `via`, which
    already holds the name; the printer shows the last).

### empty-input edges

43. `&` over an empty match conforms to nil
    When `(s/conform (s/& (s/* integer?) vector?) [])`
    Then `nil` (`preturn` 1477-1490 of the `amp` hits
    `noret?` 1455-1459: an empty `rep` result carries no
    return, so `re-conform` 1662-1666 yields `nil`);
    When `(s/valid? (s/& (s/* integer?) vector?) [])`
    Then `true`.
44. `&` over empty input explains its predicates
    When `(s/explain-str (s/& (s/* integer?) #(> (count %)
    5)) [])` Then `"[] - failed: (> (count %) 5)\n"` (the
    `amp` branch 1579-1583 with an accept-nil `re` runs
    `explain-pred-list` over the empty `re` result).
45. `&` over empty input can be insufficient
    When `(s/explain-str (s/& (s/cat :e even?) vector?)
    [])` Then `"() - failed: Insufficient input\n"` (the
    `amp` branch 1584: the `cat` `re` accepts no nil).
46. `alt` over empty input can be insufficient
    When `(s/explain-str (s/alt :a (s/cat :x even?)) [])`
    Then `"() - failed: Insufficient input\n"` (the `alt`
    branch 1598-1599 with empty input).

### registry / integration

39. Regex through the registry
    Given `(s/def :user/e (s/cat :e even?))`
    When `(s/def :user/e (s/cat :e even?))` Then `:user/e`;
    When `(s/valid? :user/e [4])` Then `true`;
    When `(s/valid? :user/e [5])` Then `false`;
    When `(some? (s/regex? (s/get-spec :user/e)))` Then
    `true` (`def-impl` stores the op raw);
    When `(s/spec? (s/cat :e even?))` Then `nil` (the op is
    a map, not a `Spec`);
    When `(some? (s/spec? (s/spec (s/cat :e even?))))`
    Then `true`;
    Given `(s/def :user/w (s/cat :x :user/late))` before
    `:user/late` exists, then `(s/def :user/late
    integer?)`
    When `(s/valid? :user/w [4])` Then `true` (ops hold
    idents; resolution is lazy at use);
    Given `(s/def :user/al :user/e)`
    When `(s/valid? :user/al [4])` Then `true` (alias
    chains resolve through `reg-resolve!`).
40. `s/spec` wraps regex
    When `(s/valid? (s/spec (s/cat :e even?)) [4])` Then
    `true` (the `spec-impl` regex branch, oracle 919).

### Generators stub

41. gen paths throw honestly
    When `(clojure.spec.alpha/gen* (s/spec (s/cat :e
    even?)) nil [] {})` Then `clojure.lang.ExceptionInfo |
    generators slice pending`;
    When `(clojure.spec.alpha/with-gen* (s/spec (s/cat :e
    even?)) (fn [] :g))` Then the same (slice F replaces
    the `re-gen` body and threads `gfn`).

### keys*

42. `keys*` expands
    When `(macroexpand-1 '(s/keys* :req-un [:user/a]))`
    Then a `(let [G (clojure.spec.alpha/keys :req-un
    [:user/a])] (clojure.spec.alpha/with-gen
    (clojure.spec.alpha/& (clojure.spec.alpha/*
    (clojure.spec.alpha/cat :clojure.spec.alpha/k
    keyword? :clojure.spec.alpha/v any?))
    :clojure.spec.alpha/kvs->map G) (fn [] ...)))` skeleton
    with one shared gensym `G` (also threaded into the
    `(clojure.spec.gen.alpha/gen G)` call inside the
    `clojure.spec.gen.alpha/fmap`), pinned structurally
    (oracle 1794-1812). Runtime use is NOT pinned: it needs
    slices A (`keys`) + D (`conformer` for `::kvs->map`) +
    F (`with-gen`/`gen`).

## TDD plan

Red-green in `rust/crates/clj-eval/tests/spec_regex.rs`
(harness mirrors `spec_alpha.rs`; each test cites the oracle
lines above):

1. RED: scenarios 1-8 (macro shapes + asserts) ->
   GREEN: `ns` exclude grows `+ * cat`; `*` `+` `?` `alt`
   `cat` `&` macros after `and`.
2. RED: scenarios 9-17 (cat valid/conform/unform) ->
   GREEN: `tagged-ret`, `and-preds`, `accept`/`accept?`/
   `pcat*`/`pcat`/`cat-impl`, `noret?`/`accept-nil?`/
   `preturn`/`add-ret`/`deriv`, `re-conform`, `RegexSpec` +
   `regex-spec-impl`, `maybe-spec` + `spec-impl`
   integration, `declare` updates.
3. RED: scenarios 18-28 + 43 (alt/`*`/`+`/`?`/`&`) ->
   GREEN: `rep*`/`rep-impl`/`rep+impl`/`amp-impl`/
   `filter-alt`/`alt*`/`alts`/`alt2`/`alt-impl`/
   `maybe-impl`, `op-unform`.
4. RED: scenarios 29-30 (form/describe) -> GREEN:
   `op-describe`.
5. RED: scenarios 31-38 + 44-46 (explain) -> GREEN:
   `op-explain`, `re-explain`, the guard-form literal.
6. RED: scenarios 39-40 (registry integration) -> GREEN: no
   new code expected (pins step 2's integration); if red,
   fix the `maybe-spec`/`spec-impl` wiring.
7. RED: scenario 41 (gen stub) -> GREEN: `re-gen` stub +
   `gen*`/`with-gen*` on `RegexSpec`. (Shipped with step
   2's record; pinned here.)
8. RED: scenario 42 (`keys*` expansion) -> GREEN: `keys*`
   macro.
9. Full gates: `cargo fmt --check`, `cargo clippy` clean,
   `clj-eval` suite green, llvm-cov floor holds (no non-test
   Rust code added), complexity ratchet holds (Rust
   test-only additions).

## Deviations from upstream (all by narrowing or port rule)

1. Macros build expansions with `list`, no syntax-quote
   (slice-1 rule 1); macro-body asserts are bare `assert`
   (slice-1 rule 9) with `c/and` kept inside.
2. `tagged-ret` is `(first {tag ret})`: the JVM builds a
   `MapEntry` object, which the port cannot construct, but
   the first entry of a single-entry map IS a port
   `MapEntry` — verified identical for `pr` (`[k v]`),
   `key`/`val`/`nth`/`count`, `=` against vectors,
   `vector?`, and sequential destructuring (probed live;
   oracle is text-only, the port is not).
3. `re-gen` (oracle 1617-1655) is a stub throwing
   `(ex-info "generators slice pending" {})`; slice F
   replaces the body and threads `gfn` (see Solution).
4. `regex-spec-impl` narrows to `[re]` (slice-1 rule 8
   precedent); the Spec glide is a `RegexSpec` record, not
   `reify` (slice-1 rule 2).
5. The non-sequential `:pred` is a hardcoded literal (see
   Solution for the derivation); upstream computes the same
   constant via syntax-quote + `res`.
6. `keys*` expansion keeps `let`/`fn`/`apply`/`concat`/
   `keyword?`/`any?` bare (use-site core refers; the port
   only treats the unqualified `let` as the special form)
   while spec/gen heads are fully qualified (as upstream's
   syntax-quote would). `::kvs->map` is NOT defined (needs
   `conformer`, slice D).
7. `case` ports verbatim, including `nil` tests and grouped
   lists (verified supported by a live port probe).
8. The `&` `(next forms)` shift in `accept-nil?` (oracle
   1469) and `deriv` (oracle 1536) ports verbatim even
   though `preturn` (oracle 1485) and `op-explain` (oracle
   1583) use the unshifted `forms`: an upstream quirk that
   only affects the `dt` "not a fn" message path.
9. `Exception.` reads `ex-info` (slice-1 rule 5) — no new
   throw sites except the `re-gen` stub; all other throws
   ride the inherited paths.
10. `explain-data` `:via` for a registered regex holds the
    name twice (scenario 38) — verbatim upstream mechanism,
    pinned against "helpful" fixes.
