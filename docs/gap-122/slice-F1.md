# Gap-122 slice F1: `clojure.spec.gen.alpha` narrow surface + seeded RNG

Part of #122 (clojure.spec port). Recon: `docs/gap-122/recon-fdef-generators.md`
§1c+§2 option F, **PART 1 ONLY**: the `clojure.spec.gen.alpha` namespace with
its generator combinators, `gen-for-pred` builtins, and `gen-for-name`, backed
by a small seeded RNG. No `spec_alpha.clj` changes (no `gen*` wiring — that is
F2, a later slice).

## Oracles (text only — NEVER execute a JVM)

- `~/.m2/repository/org/clojure/spec.alpha/0.6.249/spec.alpha-0.6.249.jar`:
  `clojure/spec/alpha.clj` (1996 lines), `clojure/spec/gen/alpha.clj`
  (**227 lines**). `gen/alpha.clj` line refs below are against that file.
- `~/.m2/repository/org/clojure/test.check/1.1.3/test.check-1.1.3.jar`:
  `generators.cljc` (**1787**), `random.clj` (**191**), `rose_tree.cljc`
  (**208**), `properties.cljc` (**95**). Refs `generators.cljc:<line>`.
- Port cites are `rust/crates/clj-eval/…` at the base commit.

## Problem

Every spec `gen*` impl, `s/gen`, `s/exercise`, fspec conform, and
instrument `:stub` need generators, but the port has no `clojure.spec.gen.alpha`
and no test.check (`gen*`/`with-gen*` are declared-but-unextended since slice 1).
A full test.check port is ≈2641 lines plus ~8 natives (recon §1c); F1 instead
ships the narrow surface spec actually uses, with a self-contained seeded RNG.

## Solution

New embedded library `rust/crates/clj-eval/src/gen_alpha.clj`, served by
`resolve_source` (`ns.rs`, same pattern as `SPEC_ALPHA_CLJ`), so
`(require '[clojure.spec.gen.alpha :as gen])` works with no files on disk.
Zero new Rust natives: the only Rust change is the 6-line embedding
(const + `resolve_source` branch).

### Public surface

Combinators (all `gen/alpha.clj:95-97` `lazy-combinators` except the deferred
two): `return` `bind` `fmap` `one-of` `such-that` `elements` `choose` `tuple`
`vector` `map` `set` `list` `shuffle` `frequency` `not-empty` `sized` `resize`
`cat` (`gen/alpha.clj:122-129`) `sample` `generate`, plus `gen-for-pred`
(`gen/alpha.clj:190-197`) and `gen-for-name` (`gen/alpha.clj:68-74`).

Scalar prims (all `gen/alpha.clj:118-120` `lazy-prims`, needed by the builtins
map): `any` `any-printable` `boolean` `bytes` `char` `char-alpha`
`char-alphanumeric` `char-ascii` `double` `double*` `int` `keyword` `keyword-ns`
`large-integer` `large-integer*` `ratio` `simple-type` `simple-type-printable`
`string` `string-ascii` `string-alphanumeric` `symbol` `symbol-ns` `uuid`.
Each prim is a variadic fn returning the (once-built) generator value, exactly
like the upstream `lazy-prim` expansion (`(let [g …] (defn s [& _] g))`), so
callers write `(boolean)`, `(string-alphanumeric)`, etc. — and `gen-for-name`
on a prim name fails with "not a generator", as upstream.

Deferred (documented, not silent): `hash-map`, `vector-distinct` (F-remainder);
`delay`/`delay-impl` (`gen/alpha.clj:55-66`, needed by F2 recursive-spec `gen*`);
`quick-check`/`for-all*` (`gen/alpha.clj:28-39`, needed by slice I).

### RNG decision: pure-Clojure splitmix64, no seeded native

The recon offered "a small seeded native (or `rand`)". F1 implements the RNG
in pure Clojure instead, for three reasons:

1. A native *immutable splittable* RNG needs a new opaque value type plus
   natives for split/rand-long/rand-double — more Rust surface, more coverage
   burden, for no behavioral gain: seed streams cannot match the JVM anyway
   (see divergences).
2. The port already has every primitive splitmix64 needs, all probe-verified:
   `unchecked-add`/`unchecked-multiply` (wrapping, `core.rs:4397-4418`),
   `bit-xor`, `unsigned-bit-shift-right`, `bit-shift-right`, `bit-and`.
   The three 64-bit mix constants have the high bit set, so they are written
   as negative long literals (no hex-reader dependency).
3. Unseeded entry points (`generate`/`sample` without a seed) seed from the
   existing unseeded `rand-int`, keeping `rng.rs` untouched.

Design (mirrors `random.clj:15-37` `IRandom`, not its SplittableRandom bits):
an RNG is one `long` of state, threaded immutably exactly like upstream
generators thread `rnd`. `rand-long(r) = mix64(r + G)`, `split` mixes with
two further odd multiples of the golden gamma (`3G`, `5G`) so the draw path
and both child paths are disjoint by construction; `rand-double` takes the
top 53 bits over 2^53. `split-n` chains splits; `lazy-random-states` ports
`generators.cljc:69-80` verbatim. Every consumer mixes before observing, so
raw state (including seed 0) is never observable.

### Generator representation (no rose trees in F1)

Upstream generator fns return rose trees (`rose_tree.cljc`, shrinking support).
Shrinking is only exercised by `check` failure paths (slice I), so F1 generator
fns return plain values; every `rose/*` call in the ported combinators is
replaced by its root-value equivalent (`rose/fmap`→plain call,
`rose/filter`→such-that-style retry where needed, shrink arms dropped).
`Generator` is a `defrecord` holding the `gen` fn (as upstream
`generators.cljc:28`), with a one-method `GeneratorP` protocol (`call-gen`)
extended to it; `generator?` is private `satisfies?`, following the `spec?`
precedent (`spec_alpha.clj:130-135`) — the port cannot `instance?` record
classes (`Unable to resolve symbol: user.Foo`, probe-verified).

### Port adaptations (beyond no-syntax-quote / fully-qualified keywords, which
follow the `spec_alpha.clj` file header)

- `choose` needs `Math/floor`: `(long (* factor width))` truncates toward
  zero, which equals floor on the non-negative product — no native needed.
  `double_long` *throws* past 2^63 (`core.rs:225-233`, unlike JVM saturation),
  so the extreme-width arm clamps in double space before `long`, mapping NaN
  (from `0.0 * +Inf`) to `lower`.
- `Math/pow`/`log`/`exp` (sizing helpers only): `n^0.6` in
  `resize-symbolish-generator` → `(quot n 4)` (exact at the default size 30);
  `size^1.1` in `size->max-leaf-count` → `(+ size (quot size 3))`;
  `random-pseudofactoring` draws uniform instead of log-uniform. All three are
  sizing-distribution approximations, pinned as shape properties by tests.
- `clojure.string` is absent: `join` → `(apply str …)`; the two colon
  cleanups in `symbol-name-or-namespace` → a 10-line local `collapse-colons`
  over `(seq s)`; `Character/isDigit` → `(int ch)` range check (`<=` on chars
  throws CCE, probe-verified).
- `inst?` builtin: upstream builds `(java.util.Date. millis)`; the port has
  no millis→Instant constructor, so F1 formats millis (Hinnant
  `civil-from-days` + time-of-day split, zero-padded by hand) and calls the
  `inst` fn from `default-data-readers` (probe-verified round-trip, including
  `inst-ms` 0 at the epoch). Range is bounded to 1970–2100 (upstream: full
  long range).
- `decimal?` builtin: `(bigdec d)` over finite-only `double*` (probe-verified
  `decimal?`).
- `uuid` builtin: two drawn longs → 32 hex nibbles via `subs` → version `4`
  + variant `[89ab]` → `parse-uuid` (probe-verified round-trip). No
  `UUID.` ctor exists in the port.
- `bytes` builtin: `(fmap byte-array (vector byte))` ports verbatim
  (`byte`/`byte-array`/`bytes?` probe-verified).
- `(+ 1.0 0.0)`-style edge values: `(/ 1.0 0.0)` → `##Inf` and `(/ 0.0 0.0)`
  → `##NaN` are probe-verified, and the `##Inf`/`##NaN` literals read, so
  `double*` uses the literals directly.
- `gen-for-name` ports `dynaload`+`gen-for-name` (`gen/alpha.clj:17-26,68-74`)
  with `require`+`resolve`+`deref` (all probe-verified, including requiring
  the already-loaded self ns); `RuntimeException.` construction becomes
  `ex-info` with identical messages (the `spec_alpha.clj` precedent). One
  deliberate micro-divergence: the `require` is guarded by `find-ns`, because
  the port's `require` (unlike the JVM's, where `clojure.core` sits in
  `loaded-libs`) throws for boot namespaces — without the guard the oracle
  "not a generator" mode would be unreachable for `clojure.core/*` vars.
  Side effect (superset, pinned by test): `user`-ns vars also resolve.
- `float?` builtin maps to `(double)` exactly as upstream (`gen/alpha.clj:143`,
  upstream quirk preserved).
- `size-bounded-bigint` is replaced by `large-integer` inside
  `base-simple-type` (the `BigInteger/shiftLeft` construction has no port
  equivalent); bigints past the long range never appear in `simple-type`.
- `long->large-integer`/`large-integer**`/`large-integer*` port near-verbatim
  (`Long/MIN_VALUE`/`MAX_VALUE` → long literals); plain `-` throws on long
  overflow (probe-verified `ArithmeticException`), so `calc-long` uses `-'`
  for the width.
- `recursive-gen`/`container-type`/`randomized` port with the integer
  substitutes above; `scale` exists only as the private two use sites need it
  (`(quot % 2)` in `container-type` is exact).
- `coll-distinct-by` ports without the ordered/duplicate arms (only `set`/`map`
  use it in F1; `the-shuffle-fn` and the `allows-dupes?`/`ordered?` params move
  with `vector-distinct` to F-remainder). Its option defaults use a merged
  default map (like `such-that`) instead of upstream's arglist `:or`: the port
  miscompiles calls to `:or`-defaulted fn bindings (pre-existing bug —
  `(let [{:keys [e] :or {e (fn [m] …)}} …] (e x))` passes the callee as its own
  first arg), so upstream's `:or` shape is avoided for fns. Untouched by F1
  beyond the workaround; the port bug itself is out of slice scope.
- `sample`/`sample-seq` keep the size-cycling shape (`(cycle (range 0 200))`,
  `generators.cljc:143-180`); `sample` keeps its assert, `generate` keeps its
  3 arities (default size 30) and no assert, as upstream.

### Known divergences (pinned, not hidden)

1. Seed streams will not match the JVM (different RNG: splitmix64 vs
   SplittableRandom; different call shapes without rose trees). Tests pin
   distribution/shape properties and determinism (same seed ⇒ same value),
   never JVM stream equality.
2. `ratio`/`double` edge distributions are approximate until slice K
   (`double-finite` is a sized small/large mix, not the bit-level
   construction; `ratio` mirrors upstream exactly but its inputs are the
   approximate `small-integer`/`nat` streams).
3. `uri?` has NO builtin: `gen-for-pred` returns `nil` for it. There is no
   pure-Clojure URI constructor in the port (no `URI/create` emulation, no
   `#uri` reader). Needs one small host constructor native in F2-or-later.
4. `inst?` range is years 1970–2100 (see above), not the full long range.
5. No shrinking: values are final. `no-shrink`/`shrink-2` are meaningless and
   stay out with rose trees (slice I).
6. `assert` failure texts follow the port (`Assert failed: …`), not the JVM.

## BDD (acceptance; each is a test in `rust/crates/clj-eval/tests/gen_alpha.rs`)

- Given the `gen` alias, `(generate (return 42))` is 42 and
  `(sample (return 42))` is ten 42s.
- Given a fixed seed, `generate` is deterministic: same seed ⇒ same value,
  across fresh evaluators; `(generate (sized #(return %)) 30 s)` is 30 and
  `resize` overrides it exactly.
- Given `(choose lo hi)`, 100 seeded draws are all ints within `[lo, hi]`;
  `(choose 5 5)` is always 5; full-long-range `choose` stays in range.
- Given `(elements [7])`, every draw is 7; given `[]`, the oracle assert text
  fires; given a set, draws are members.
- Given `one-of`/`frequency` over recognizable arms, draws come from the arms;
  empty `one-of` and all-zero-weight `frequency` fire the oracle assert texts.
- Given `(such-that pred gen)`, 100 seeded draws satisfy `pred`; given an
  unsatisfiable pair, the oracle "Couldn't satisfy…" text fires; int, map,
  and bad-opts third args behave per the oracle.
- Given `tuple`/`vector`/`list`, draws are vectors/lists of the right shape
  and per-position generators; `vector` honors fixed and min/max arities;
  `(sample (vector boolean))` starts with `[]` (size cycle starts at 0).
- Given `set`/`map` with `:num-elements`, draws have exactly that count with
  distinct keys; default draws are sets/maps within size bounds; impossible
  distinctness fires "Couldn't generate enough distinct elements!".
- Given `(shuffle coll)`, draws are permutations of `coll`; `(shuffle [])`
  is `[]`.
- Given `not-empty`, 100 seeded draws are all non-empty.
- Given `cat`, draws are the concatenation of the arm draws.
- Given each scalar prim (`boolean` `char` `bytes` `string-alphanumeric`
  `keyword` `keyword-ns` `symbol` `symbol-ns` `uuid` `ratio` `int`
  `large-integer` `large-integer*` `double` `double*`), 100 seeded draws
  satisfy the matching predicate; `uuid` draws print with version `4`;
  `double*` honors `:infinite?`/`:NaN?`/`:min`/`:max`.
- Given `simple-type`/`any`/`any-printable`, 50 seeded draws realize without
  throwing and `any` draws are never nil.
- Given `inst?`/`decimal?` builtins, draws satisfy the predicates and inst
  millis stay in `[0, 4102444800000]`.
- Given `gen-for-pred`, every builtin pred (all 45 oracle entries except
  `uri?`) resolves to a generator whose draws satisfy the pred; set preds
  become `elements`; `uri?` resolves to nil (documented, divergence 3).
- Given `gen-for-name`, qualified names of generator-valued vars resolve to
  working generators (same seed ⇒ same value as the direct var); unqualified /
  non-generator / missing-var / missing-ns inputs fail with the oracle failure
  modes — including prim names, which are fns (as upstream) and fail with
  "not a generator".

## TDD (implementation order)

1. `resolve_source` + `gen_alpha.clj` skeleton (`ns`, `Generator`, `return`,
   `generate`) + load test.
2. RNG core (`mix64`, `rand-long`, `rand-double`, `split`, `split-n`,
   `rand-range`, `calc-long`) + `choose` + determinism tests.
3. `gen-pure`/`gen-fmap`/`gen-bind`, `fmap`/`bind`, `sized`/`resize`,
   `sample`/`sample-seq` + tests.
4. `one-of`/`frequency`/`elements`/`such-that`/`not-empty` + tests.
5. `gen-tuple`, `tuple`/`vector`/`list`, `coll-distinct-by`, `set`/`map`,
   `shuffle`, `cat` + tests.
6. Scalar prims (char/string/keyword/symbol/uuid/bytes/boolean/int/ratio/
   large-integer*/double*) + `simple-type`/`any` families + `inst`/`decimal`
   ctors + tests.
7. `gen-builtins`, `gen-for-pred`, `gen-for-name` + table tests.
8. Gates: `cargo fmt --check`, `cargo clippy -- -D warnings`, full
   `clj-eval` suite, `coverage-patch` 100 on the Rust diff,
   `complexity` ratchet holds.

## Oracle pins (upstream texts preserved verbatim)

Assert/exception messages kept: "Second arg to fmap must be a generator",
"First arg to bind must be a generator", "First arg to sample must be a
generator", "Second arg to resize must be a generator", "Args to tuple must
be generators", "First arg to vector must be a generator", "First arg to
list must be a generator", "Arg to vector must be a generator",
"First arg to set must be a generator!", "Arg to not-empty must be a
generator", "Arg to one-of must be a collection of generators",
"one-of cannot be called with an empty collection", "Arg to frequency must
be a list of [num generator] pairs",
"frequency must be called with at least one non-zero weight",
"elements cannot be called with an empty collection",
"Second arg to such-that must be a generator",
"Couldn't satisfy such-that predicate after %s tries.",
"Couldn't generate enough distinct elements!", "Bad argument to such-that!",
"Var %s is not a generator", "Var %s is not on the classpath".
Defaults kept: sample 10 / sample-seq max-size 200 / generate size 30 /
such-that max-tries 10 / set+map max-tries 10.
