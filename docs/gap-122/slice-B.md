# Gap-122 slice B: combinator specs (`s/or`, `s/nilable`, `s/coll-of`, `s/map-of`, `s/every`)

Part of #122 (clojure.spec port). This is the second slice: the
branching/collection combinators over the slice-1 core (registry,
`def`, predicate specs, `s/and`, `valid?`, `conform`/`unform`,
`explain`).

## Source

Ported from upstream `spec.alpha-0.5.238`
(`clojure/spec/alpha.clj`, 1996 lines), fetched as the
`spec.alpha-0.5.238-sources.jar` from Maven Central and used as a
**text oracle only** (no JVM was executed for this slice; every pin
below is derived by code-reading the oracle). Upstream line
references are against that file. The port lives in
`rust/crates/clj-eval/src/spec_alpha.clj` (embedded library, no glue
changes needed); tests in
`rust/crates/clj-eval/tests/spec_combinators.rs`.

Oracle pins:

| Construct | Upstream lines |
|---|---|
| `*coll-check-limit*` / `*coll-error-limit*` | 36-42 |
| `or` macro | 476-491 |
| `every` macro | 520-566 |
| `every-kv` macro | 568-577 |
| `coll-of` macro | 579-590 |
| `map-of` macro | 592-604 |
| `tuple` macro | 694-700 |
| `pvalid?` | 779-784 |
| `tuple-impl` | 996-1056 |
| `tagged-ret` | 1058-1059 |
| `or-spec-impl` | 1061-1126 |
| `coll-prob` | 1223-1241 |
| `every-impl` | 1245-1372 |
| `nilable-impl` | 1834-1858 |
| `nilable` macro | 1860-1864 |

## Problem

Slice 1 covers only predicate specs and `s/and`. Real spec usage
needs branching (`s/or`: tagged conformed values, per-branch
explain), nullable specs (`s/nilable`), and collection specs with
options (`s/every` sampling, `s/coll-of` exhaustive conform,
`s/map-of` key/val specs, `:kind`/`:count`/`:min-count`/`:max-count`/
`:distinct`/`:into`).

## Solution

Port the five macros plus the two supporting constructs upstream
`map-of` is built on (`s/tuple` + `tuple-impl`, `s/every-kv`;
`map-of` expands through `every-kv`, which uses `tuple` as its
element pred). New spec records `OrSpec`/`NilableSpec`/`TupleSpec`/
`EverySpec` extend the slice-1 `Spec` protocol, so `valid?`/
`conform`/`unform`/`explain-data`/`form`/`describe` integrate with
no changes to existing code. Generator params (`gfn`, `:gen`) are
narrowed out, as in slice 1 (`gen*`/`with-gen*` stay unextended).

Port adaptations (each pinned by tests):

1. `or-spec-impl` uses one conform loop (the 2-/3-branch fast paths
   are behavior-identical), mirroring slice 1's `AndSpec`.
2. `tagged-ret` builds the map entry via `(find {tag ret} tag)`: the
   port has no `MapEntry.` interop. Entries print as `[tag ret]`,
   and `key`/`val`/`nth`/destructuring all work on them.
3. `Integer/MAX_VALUE` in evaluated positions (the `every` cpred,
   `coll-prob`'s min/max check) is the literal `2147483647`: the
   port has no `Integer/MAX_VALUE`. Explain *data* still carries
   the `Integer/MAX_VALUE` symbol (abbreviated to `MAX_VALUE` by
   `abbrev`, exactly as upstream).
4. `long` does not exist in the port: the indexed sampling step
   `(long (/ count limit))` is `(quot count limit)`, identical for
   non-negative counts.
5. Macros take raw `& opts` and `(apply hash-map opts)` explicitly
   instead of relying on `& {:keys ...}`/seq-as-map destructuring
   conversion; `res-kind` is a `defn-` over the opts seq.
6. The `or` macro joins the `:refer-clojure :exclude` list
   (`[and def or]`); all combinator code uses `c/and`/`c/or` and
   bare `assert`, per the slice-1 file conventions. New code is
   appended after the slice-1 code so no forward `declare`s are
   needed.
7. `:gen` is accepted and dropped (generators are a later slice).

Out of scope (unchanged): `s/keys`/`s/merge` (sibling worker),
regex ops (`cat`/`alt`/`*`/`+`/`?`/`&`, later slice),
`clojure.spec.gen.alpha`, `fdef`/instrumentation, `conformer`.

## BDD scenarios

### `s/or`

- valid?: first branch matches, second branch matches, neither
  matches.
- conform: `[tag value]` entry of the FIRST matching branch;
  nested ors nest entries (`[:o [:e 2]]`); no match is
  `:clojure.spec.alpha/invalid`.
- unform: `[tag val]` back through the branch (nested: to the
  raw value).
- explain: one problem per failing branch, `path` extended with
  the branch tag (`at: [:even]`), in branch order; named specs add
  the `spec: <name>` suffix.
- Registry idents resolve lazily (forward refs, as with `s/and`).
- Macro asserts (even args, keyword tags) throw `AssertionError`.
- `form`/`describe`: `(or :even even? :small (< % 10))`.

### `s/nilable`

- valid?: nil ok, satisfying value ok, other values rejected.
- conform/unform: nil passes through untouched.
- explain: two problems, the pred at `[:clojure.spec.alpha/pred]`
  and `nil?` at `[:clojure.spec.alpha/nil]`.
- `form`/`describe`: `(nilable even?)`.

### `s/coll-of` (exhaustive)

- valid?: all-valid, one bad element, empty coll, non-coll input.
- conform: elementwise conformed rebuild (`[[:e 2] [:o 3]]`);
  `:into []` rebuilds a list as a vector; default preserves the
  input shape (list stays a list).
- Opts: `:kind` (incl. explain `vector?` failure), `:count`
  (explain `(= 2 (count %))`), `:min-count`/`:max-count` (explain
  `(<= 1 (count %) 2)`, min-only `(<= 2 (count %) MAX_VALUE)`),
  `:distinct` (explain `distinct?`).
- Element failures report `in: [<index>]`; ident element preds
  extend `:via` (`spec: <pred-name>`).
- `describe`: `(coll-of even? :kind vector?)`.

### `s/every` (sampling, no conform)

- valid? over vectors, lists, sets.
- conform returns the input UNCHANGED (no element conforming).
- `*coll-check-limit*` sampling: a bad element past the sample
  still validates (indexed step + sequential walk).
- `*coll-error-limit*` caps explain problems.
- Same opt checks as `coll-of` (shared `every-impl`).
- `describe`: `(every even?)`.

### `s/map-of` / `s/every-kv` / `s/tuple`

- map-of valid?: good map, bad key, bad val, non-map (`:kind`
  defaults to `map?`).
- map-of conform: values conformed (`{:a [:o 3]}`), keys raw by
  default, `:conform-keys true` conforms keys (`{[:k :a] 2}`);
  unform reverses.
- map-of explain: `in: [:a 1] at: [1]`.
- `describe`: `(map-of keyword? even?)`.
- every-kv validates entries, never conforms.
- tuple: per-element valid?/conform, count mismatch (`(=
  (count %) 2)`), non-vector (`vector?`), element paths
  (`in: [0] at: [0]`), unform (+ assert), `(tuple even?
  string?)`.

## TDD plan

1. Helpers: `*coll-check-limit*`, `*coll-error-limit*`,
   `pvalid?`, `tagged-ret`, `res-kind` table, `coll-prob` (red:
   limit vars resolve; green: direct calls).
2. `s/or` + `or-spec-impl` (red: valid?/conform/unform/explain/
   assert/form tests; green: record + methods + macro).
3. `s/nilable` + `nilable-impl` (same cycle).
4. `s/tuple` + `tuple-impl` (same cycle).
5. `every-impl` + the four macros (`every`, `every-kv`,
   `coll-of`, `map-of`) (red: full opt matrix; green: impl +
   expansions).
6. Gates: `cargo fmt --check`, `cargo clippy` clean, full
   `clj-eval` suite green; coverage floor + complexity ratchet
   unaffected (no Rust sources touched).
