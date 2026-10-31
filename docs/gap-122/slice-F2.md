# Gap-122 slice F2: `gen*`/`with-gen*` wiring (`s/gen`, `s/with-gen`, `s/exercise`)

Part of #122 (clojure.spec port). Recon: `docs/gap-122/recon-fdef-generators.md`
§1c+§2 option F, **PART 2**: wire every spec's `gen*`/`with-gen*` to the
slice-F1 `clojure.spec.gen.alpha` surface, add the F1-deferred gen fns the
wiring needs (`delay`, `hash-map`, `vector-distinct`), and ship `s/gen`,
`s/with-gen`, `s/exercise`, `s/exercise-fn`. F1 landed the gen namespace
with `gen*`/`with-gen*` as stubs ("generators slice pending"); F2 replaces
every stub with the real generator. No `spec_alpha.clj` behavior outside
generation changes (conform/unform/explain/describe untouched).

## Oracles (text only — NEVER execute a JVM)

- `~/.m2/repository/org/clojure/spec.alpha/0.6.249/spec.alpha-0.6.249.jar`:
  `clojure/spec/alpha.clj` (1996 lines). Line refs below are against it:
  `with-gen` 210-216, `gensub`/`gen` 279-303, `*recursion-limit*` 26-30,
  `recur-limit?`/`inck` 750-755, `k-gen` family 793-819,
  `map-spec-impl` gen 878-902, `spec-impl` 913-944, `multi-spec-impl`
  gen 976-993, `tuple-impl` gen 1047-1055, `or-spec-impl` gen 1114-1125,
  `and-spec-impl` gen 1191, `merge-spec-impl` gen 1213-1220,
  `every-impl` gen 1341-1371, `re-gen` 1617-1655, `regex-spec-impl`
  1699-1723, `nonconforming` 1813-1832, `nilable-impl` gen 1848-1855,
  `exercise`/`exercise-fn` 1866-1888, `:gen` macro args (`spec` 366-383,
  `keys` 414-474, `every` 520-566).
- `~/.m2/repository/org/clojure/test.check/1.1.3/test.check-1.1.3.jar`:
  `generators.cljc` `hash-map` (691), `the-shuffle-fn` (777),
  `coll-distinct-by` (792), `vector-distinct` (841); `gen/alpha.clj`
  `delay-impl`/`delay` (52-66).
- Port cites are `rust/crates/clj-eval/…` at the base commit.

## Problem

Every `gen*`/`with-gen*` in `spec_alpha.clj` is a stub: PredSpec/AndSpec/
MapSpec/MergeSpec/TupleSpec/MultiSpec/OrSpec/NilableSpec/EverySpec/
NonconformingSpec leave them unextended (protocol-throw on call) and
RegexSpec/`re-gen` throw `ex-info` "generators slice pending". There is
no `s/gen`, `s/with-gen`, `s/exercise`; the `:gen` kwargs of `s/spec`,
`s/keys`, `s/every` are dropped (`every` dissocs it, the other two don't
even parse it); `s/keys*` expands to `s/with-gen` calls that cannot
resolve. Slice F1 deferred exactly the three gen fns this wiring needs:
`delay`/`delay-impl` (recursive-spec `gen*`), `hash-map` (map-spec
`gen*`), `vector-distinct` (`:distinct` every `gen*`).

## Solution

Pure `.clj` additions; **zero Rust diff** (no new natives, no `ns.rs`
change — the F1 embedding already serves both files).

### `gen_alpha.clj`: the three deferred fns

- `delay-impl` + `delay` macro (oracle `gen/alpha.clj:52-66`): the macro
  builds `(delay-impl (clojure.core/delay …))` with `list` (no
  syntax-quote, per the file header); the `clojure.core/` qualification
  is load-bearing (bare `delay` would self-expand). `delay-impl`
  derefs and delegates via `call-gen` (the `(:gen @gfnd)` equivalent
  without rose trees).
- `hash-map` (oracle `generators.cljc:691-713`, verbatim incl. both
  asserts): `take-nth`/`zipmap` over `(apply tuple vs)`.
- `the-shuffle-fn` (oracle `:777-790`, verbatim; `rand-range` is F1's)
  and `vector-distinct` (oracle `:841-855` incl. the assert text).
  `coll-distinct-by` regains the upstream
  `[empty-coll key-fn allows-dupes? ordered? gen opts]` shape:
  `ordered?` picks `the-shuffle-fn` vs identity; `allows-dupes?` stays
  shrink-only (accepted, still unused — shrinking is slice I). `set`/
  `map` pass `false false` exactly as upstream.
- The ns `:exclude` grows `hash-map` + `delay`, making it identical to
  upstream's list.

### `spec_alpha.clj`: require + `gfn`/`id` + `gen*` per record

- The ns form gains `(:require [clojure.spec.gen.alpha :as gen])`
  (the gap-124 `tools_deps_edn.clj` precedent for embedded-lib
  requires). Call sites use the `gen/` alias exactly as upstream —
  including inside `let [gen (fn …)]` locals (or/multi/tuple `gen*`),
  where the qualified head still resolves via the alias.
- Every spec record gains a `gfn` field; OrSpec/MultiSpec/MapSpec also
  gain the upstream `id` (a `random-uuid` minted in the `-impl` ctor,
  as upstream). Every `-impl` ctor gains the upstream `gfn` arity
  (`every-impl`/`regex-spec-impl`/etc. keep their short arities
  delegating with nil, exactly as upstream does); `spec-impl` is
  restored to upstream's `[form pred gfn cpred?]`/`[form pred gfn
  cpred? unc]` plus the port-convenience `[form pred]`, and the
  `conformer` macro is updated to the upstream arities. `nilable`/
  `tuple`/`multi-spec`/`or`/`and`/`merge` macros are unchanged (they
  pass nil upstream too); `spec`/`keys`/`every` accept `:gen` and
  thread it through (`every` keeps dissocing `:gen` from describe
  opts, as upstream).
- New fns (upstream-verbatim unless noted): `*recursion-limit*` (4),
  `recur-limit?`, `inck`, `with-gen`, `gensub` (incl. the "Unable to
  construct gen at: …" throw with `::path`/`::form`/`::failure
  `:no-gen`), `gen` (both arities, `::recursion-limit` rmap seed),
  `k-gen`/`or-k-gen`/`and-k-gen`, `empty-coll` (keyed on
  `clojure.core/*` symbols, the evaluated value of upstream's
  syntax-quoted keys), `exercise` (all 3 arities), `exercise-fn`
  (throw via `ex-info`, the file precedent; full fspec use waits for
  slice G but the fn ports verbatim and works over `{:args …}` maps).
- `re-gen` replaces the stub with the full oracle port (1617-1655):
  overrides lookup (`spec-name` of orig/resolved + path), `::gfn`,
  `accept`/nil/`amp`/`pcat`/`alt`/`rep` arms, `inck`/`recur-limit?`
  guards, `gen/delay` under `:id`-bearing ops.
- `with-gen` on a regex op assocs `:clojure.spec.alpha/gfn` (as
  upstream assocs `::gfn`), consumed by `re-gen`. `NonconformingSpec`
  delegates `gen*` to the inner spec and `with-gen*` re-wraps, as
  upstream. `conformer` one-arg specs (`gen-for-pred` of a fn → nil)
  fail through `gensub` with the oracle no-gen throw.
- Forward refs (`gensub`, `k-gen` family, `inck`, `recur-limit?`,
  `-impl` ctors used by `with-gen*`, `empty-coll`) join the `declare`
  (the port resolves globals at analyze time).

### Known divergences (pinned, not hidden)

1. Seed streams do not match the JVM (F1 divergence 1 covers the RNG;
   F2 pins shape/determinism properties, never JVM streams).
2. `uri?`-rooted specs have no generator (`gen-for-pred` nil → the
   oracle "Unable to construct gen" throw), per F1 divergence 3.
3. `s/spec` still lacks upstream's `(when form …)` guard (pre-existing
   slice-1 narrowing, out of F2 scope).
4. `:gen` on `s/every` with `:distinct` + impossible counts throws
   "Couldn't generate enough distinct elements!" from F1's
   `coll-distinct-by` (same text as upstream's default `ex-fn`).
5. `exercise-fn` over real fspecs waits for slice G (no `fdef` yet);
   the ported fn is tested over `{:args …}` maps.
6. Generation depth ceiling: each spec nesting level costs ~30
   interpreter calls and the port overflows past ~100-200 nested
   calls (pre-existing evaluator characteristic — plain `(down 200)`
   overflows too), so deeply recursive grammars drawn at the default
   `*recursion-limit*` 4 can abort where the JVM survives. Same
   failure *mode* as upstream (stack exhaustion on pathological
   grammars), lower threshold. The recursion test binds the limit
   small (same termination mechanism, within budget); the default 4
   is pinned as a value.

## BDD (acceptance; each is a test in `rust/crates/clj-eval/tests/gen_wiring.rs`)

Gen additions (F1-deferred, oracle `generators.cljc` + `gen/alpha.clj`):

- Given `(gen/delay …)`, the body does not run until first `generate`;
  recursive grammars via delay terminate and every value realizes.
- Given `(gen/hash-map :a (gen/return 1))`, draws are `{:a 1}`; odd
  arg counts and non-generator values fire the oracle assert texts.
- Given `(gen/vector-distinct (gen/choose 0 9) {:num-elements 5})`,
  50 seeded draws are distinct 5-vectors in range; impossible
  distinctness fires "Couldn't generate enough distinct elements!";
  non-generator first arg fires the oracle assert text.

`gen*` wiring (oracle `alpha.clj` arms above; `generate` with fixed
seeds; every value must also `s/valid?` against its spec):

- Given `(s/def ::n int?)`, `(s/gen ::n)` draws satisfy `int?`; given
  `(s/def ::s #{1 2 3})`, draws are members; given an unresolvable
  spec, the oracle "Unable to resolve spec: …" throw fires.
- Given `(s/and int? pos?)`, draws satisfy both; given
  `(s/or :i int? :s string?)`, draws are `[tag val]` entries valid
  for the tagged branch; given `(s/nilable int?)`, both nil and ints
  appear across 200 seeded draws.
- Given `(s/tuple int? string?)`, draws are 2-vectors of the right
  shape; given `(s/coll-of int?)`, `(s/map-of keyword? int?)`,
  `(s/every int? :count 3)`, `(s/every int? :kind vector? :into [])`,
  `(s/coll-of int? :distinct true :min-count 2 :max-count 4)`, draws
  are valid colls honoring count/distinct/into.
- Given `(s/def ::m (s/keys :req [::a] :opt [::b]))` (+ int specs for
  `::a`/`::b`), draws are maps containing `:user/a`-style keys…
  (concretely: required key always present, values valid, optional
  key sometimes present); given two `keys` specs, `(s/merge …)`
  draws satisfy both.
- Given `(s/def ::c (s/cat :a int? :b string?))`, `(s/gen ::c)` draws
  are 2-seqs valid for the regex; `s/*`/`s/+`/`s/?`/`s/alt`/`s/&`
  specs generate valid seqs (possibly empty where the op allows).
- Given a recursive `(s/def ::tree (s/or :leaf int? :node (s/tuple
  keyword? ::tree)))` and a small bound `*recursion-limit*`,
  generation terminates (recursion limit) and draws are valid; the
  default limit 4 is pinned as a value (divergence 6).
- Given a `multi-spec` over a 2-method multi, draws carry the retag
  key and are valid; given `(s/nonconforming spec)`, draws equal
  inner-spec draws in distribution (valid for the inner spec).
- Given `(s/with-gen int? #(gen/return 7))`, every draw is 7; given
  `(s/spec int? :gen #(gen/return 8))` / `:gen` on `keys`/`every`,
  draws use the override; given a name override
  `(s/gen ::n {::n #(gen/return 42)})`, draws are 42; given a path
  override (`{[1] …}` into a tuple, `{[:a] …}` into a `cat`), the
  sub-path value uses it.
- Given `(s/exercise ::n 5)`, five `[val conform]` pairs return with
  `val` valid; given `(s/exercise-fn 'f 5 {:args (s/cat :x int?)})`,
  five `[args ret]` pairs return with `ret` = `(apply f args)`.
- Given a spec with no generator (`(s/conformer str)` bare fn… more
  precisely a pred with no builtin, e.g. `(s/spec some-new-pred)`),
  `s/gen` throws "Unable to construct gen at: …" with `:failure
  :no-gen` in the ex-data.
- Given `s/keys*`, the macroexpansion's `with-gen` resolves and a
  `keys*` regex over `:req-un` keys generates valid kv seqs.

## TDD (implementation order)

1. `gen_alpha.clj`: `delay-impl`/`delay` + `hash-map` +
   `the-shuffle-fn`/`vector-distinct` (+ `coll-distinct-by` reshape,
   `:exclude` growth) + gen-side tests (red → green).
2. `spec_alpha.clj` ns `:require` + `*recursion-limit*`/
   `recur-limit?`/`inck` + `with-gen`/`gensub`/`gen` + PredSpec/
   AndSpec `gen*`/`with-gen*` (`spec-impl`/`and-spec-impl` arities)
   + tests.
3. Or/Nilable/Tuple/Every (`or-spec-impl`/`nilable-impl`/`tuple-impl`/
   `every-impl` + `empty-coll`, `every` macro `:gen`) + tests.
4. Map/Merge (`k-gen` family, `map-spec-impl`/`merge-spec-impl`,
   `keys`/`spec` macros `:gen`, `conformer` arity update) + tests.
5. Multi/Nonconforming (`multi-spec-impl` + `nonconforming`) + tests.
6. `re-gen` + RegexSpec `with-gen*` (`regex-spec-impl` arity) + tests
   (incl. recursion-limit + `keys*` runtime).
7. `exercise`/`exercise-fn` + overrides/path-override tests.
8. Gates: `cargo fmt --check`, `cargo clippy -p clj-eval --all-targets
   -- -D warnings`, new suite + full `clj-eval` suite green.

## Oracle pins (upstream texts preserved verbatim)

"Unable to construct gen at: …" (+ `::path`/`::form`/`::failure`
`:no-gen` ex-data), "Var %s is not a generator" (n/a here),
"First arg to vector-distinct must be a generator!", "Value args to
hash-map must be generators", "Couldn't generate enough distinct
elements!", "No :args spec found, can't generate". Defaults kept:
`*recursion-limit*` 4, `gen-max` 20, `such-that` 100 in `gensub`,
nilable frequency `[1 9]`, exercise 10, every `:max-tries` 100 in
the distinct arms.
