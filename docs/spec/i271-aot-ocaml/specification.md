# Issue #271: AOT Performance Catch-Up with OCaml — Specification

Status: Draft (Spec-first commit)
Issue: https://github.com/hhh-pa/clojure/issues/271
Baseline: `ca27a18e` (`master`, post-#259/#260)

## 1. Goal

**Requirement: AOT performance must outperform OCaml** (issue #271, updated 2026-11-17). On the Linux `h2h-cpu-mem` workflow, native binaries produced by `clojure-rs build` must beat native OCaml (`ocamlopt -O2`) across the shared snippet suite. On every compared case, AOT's median Pin instruction count and median peak RSS must both be strictly below OCaml's: every `aot-ocaml` case is a win in `h2h-compare`'s terms, with no losses and no ties (`--metrics mem,ins`). Staying within the 1.15x margin (parity) is not enough. Enforcing this needs an opt-in `h2h-compare` mode that fails on ties as well as losses; the other gates keep their loss-only rule.

To establish the authoritative Linux baseline and enforce every subsequent PR, first extend `.github/workflows/h2h-cpu-mem.yml` and `scripts/ci/go/cmd/h2h-compare` to measure `AOT from`, `AOT to`, and `OCAML`, and add the `AOT from` vs `AOT to` (`aot-rsrs`) and `AOT` vs `OCAML` (`aot-ocaml`) quality gates.

## 2. Invariants and Non-Goals

1. **Interpreter invariance:** `rs-from` and `rs-to` inside `h2h-cpu-mem.yml` MUST remain built with `cargo build --release --no-default-features --locked` and run via `clojure-rs run`, preserving exact parity on existing `recipes` (`rs-from` vs `python`) and `recipes-rsrs` (`rs-from` vs `rs-to`) gates.
2. **Semantic equivalence:** `clojure-rs build` optimizations MUST preserve observable output, exit codes, and Clojure error semantics of `clojure-rs run`.
3. **No benchmark gaming:** Snippet sources in `hhh-pa/clojure-code-snippet` and `hhh-pa/ocaml-code-snippet` MUST NOT be altered to work around compiler/runtime bottlenecks inside `clojure-rs`.
4. **Workspace quality gates unchanged:** All existing Rust and Go gates apply without relaxation (100.00% canonical line and patch coverage, `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, MSRV `1.85` on non-`aot` crates, `--no-default-features`, complexity ratchet, `geiger`, `udeps`, `deny`, `audit`, `go test ./...`).
5. **Non-goals for Slice 1:** Slice 1 delivers measurement, gating, tests, and the authoritative Linux Pin baseline report; compiler/runtime optimization changes belong to Slice 2+ so the baseline instrumentation can be reviewed cleanly and verified first.

## 3. Suite Parity Context

Direct repository inventory at spec time:

- `hhh-pa/clojure-code-snippet`: 394 scripts, 1,754 test cases.
- `hhh-pa/ocaml-code-snippet`: 393 scripts, 1,743 test cases.
- Every script in `ocaml-code-snippet` has a matching `(label, stem)` and case set in `clojure-code-snippet`; `string_related/suffix_tree/string_function_calculation` (11 cases) is Clojure-only.
- `ocaml-code-snippet` uses `ocamlopt -O2` and only OCaml standard library modules (`open Complex` appears in `optics/jones_mueller.ml` and `optics/thin_film.ml`). Its `discover.go` differs slightly from `clojure-code-snippet/scripts/ci/go/internal/snipcore/discover.go` on nested paths, so `h2h-cpu-mem.yml` MUST run `clojure-code-snippet`'s `snip measure --script-extension .ml` (mirroring `python`'s `--script-extension .py`) so shards and `(label, stem, case)` keys match deterministically.

## 4. Slice 1 — `h2h-compare` Contract

Extend `scripts/ci/go/cmd/h2h-compare`:

1. **New flag:** `--allow-extra-challenger` (boolean, default `false`).
2. **Default behavior (`false`):** Preserved byte-for-byte. Any non-empty `onlyBase` or `onlyChal` fails with `leg mismatch (only-<base> N, only-<chal> M)`. Existing gates (`recipes`, `recipes-rsrs`, `aot-rsrs`) MUST leave this flag off.
3. **Opt-in behavior (`--allow-extra-challenger`):**
   - `onlyChal` cases (challenger keys missing from base, e.g. the 11 Clojure-only cases of `string_function_calculation`) are still printed in the informational summary (`only-<chal>: N` and listed keys) and are **excluded from failing `problems` as a leg mismatch**.
   - `onlyBase` cases (any base/OCaml key missing from challenger/AOT) **MUST STILL FAIL** with `leg mismatch (only-<base> N)`.
   - If the intersection of base and challenger keys is empty (`len(common) == 0`), comparison **MUST FAIL** (`no common cases between <base> and <chal>`).
   - Metric comparisons (`MEM`, `INS`), `not-ok`, `test_counts` drift, `failures`, and `missing` checks run strictly over the compared cases and leg summaries using the exact same rules and `--margin` threshold.
4. **Tests:** Unit tests and golden/fixture tests in `scripts/ci/go/cmd/h2h-compare/main_test.go` MUST cover:
   - flag parsing (`--allow-extra-challenger`),
   - passing run when only `onlyChal` is non-empty and all common cases pass,
   - failing run when `onlyBase` is non-empty even with `--allow-extra-challenger`,
   - failing run when `len(common) == 0` with `--allow-extra-challenger`,
   - default strict mode unaffected.

## 5. Slice 1 — `.github/workflows/h2h-cpu-mem.yml` Contract

### 5.1 Inputs & `resolve` job

- Resolve `ocaml_snippets_sha` from `hhh-pa/ocaml-code-snippet` (default `master`, overridable via `workflow_dispatch` input alongside `snippets_sha` and `python_snippets_sha`).
- Compute `waive_ocaml` boolean alongside `waive_recipes` and `waive_h2h`:
  - Triggered by `[waive-ocaml]` in PR head commit subject/body (or `waive_ocaml` `workflow_dispatch` input).
  - Independent of `[waive-recipes]` so `recipes` stays controlled solely by `[waive-recipes]`/`[waive-h2h]` and `aot-rsrs` is controlled only by `[waive-h2h]`.

### 5.2 `measure` job (32 shards)

1. **Toolchains & dependencies:**
   - Provision LLVM 23 via `./.github/actions/setup-llvm`.
   - Install `ocaml` and `ocaml-native-compilers` via `awalsh128/cache-apt-pkgs-action` (same package pair as `ocaml-code-snippet`).
   - Check out `hhh-pa/ocaml-code-snippet` at `ocaml_snippets_sha` alongside `snippets` (`clojure-code-snippet`) and `py` (`python-code-snippet`).
   - Decrypt and extract shared test resources into `ocaml/tests/resources` using `RESOURCES_PASSPHRASE` (same archive already extracted for `snippets` and `py`).
2. **Isolated binary builds:**
   - Build `--no-default-features` interpreter binaries (`rs-from`, `rs-to`) and copy/stage them **before or in separate target dirs from** default-feature `aot` builds so feature toggling never contaminates `run` measurements.
   - Build `aot-from` (from `work/`) and `aot-to` (from `base/` when distinct from `work/`) with default `aot` feature:
     - `cargo build --release --locked`
     - `cargo rustc --release --locked -p clj-rt --crate-type staticlib`
     - Stage `clojure-rs`, `libcljrt.a`, and `hot-text.ld` together into dedicated directories (`$WORK/aot-from-bin/`, `$WORK/aot-to-bin/`) so `NativeBuildCache` / `clojure-rs build` automatically discovers matching `libcljrt.a` and Linux `hot-text.ld` next to the compiler binary.
3. **Isolated OCaml wrapper:**
   - Provide an executable wrapper in `$RUNNER_TEMP/bin/ocaml-build`:
     ```bash
     #!/usr/bin/env bash
     set -euo pipefail
     script=$1
     out=$2
     work=$(mktemp -d)
     trap 'rm -rf "$work"' EXIT
     cp "$script" "$work/$(basename "$script")"
     ocamlopt -O2 -o "$work/a.out" "$work/$(basename "$script")"
     mv "$work/a.out" "$out"
     ```
   - This prevents `ocamlopt` from writing `.cmi`/`.cmx`/`.o` artifacts into `ocaml/src/...` and isolates concurrent `--jobs` workers cleanly.
4. **Measurement runs:**
   - Measure `aot-from`: `./scripts/ci/go/snip measure --runtime aot-from --resources tests/resources --build-template "$AOT_FROM_DIR/clojure-rs build {script} -o {out}" --pin ...` writing `aot-from-results.json`.
   - Measure `aot-to` (using `aot-from-results.json` directly if `base == work` and non-PR dispatch reuses the single commit, OR building and measuring `base/`): writing `aot-to-results.json`.
   - Measure `ocaml`: inside `ocaml/`, `../snippets/scripts/ci/go/snip measure --runtime ocaml --script-extension .ml --resources tests/resources --build-template "$RUNNER_TEMP/bin/ocaml-build {script} {out}" --pin ...` writing `ocaml-results.json`.
   - Upload shard artifacts `aot-from-results-<shard>`, `aot-to-results-<shard>`, `ocaml-results-<shard>` (or bundled alongside shard outputs) for `h2h-compare`.

### 5.3 `h2h-compare` job

Run five comparisons with `--metrics mem,ins --margin 0.15` (and pass `--report-only` only per the corresponding waive flag):

1. `recipes`: `python` (base) vs `rs-from` (challenger) — existing gate, strict key match.
2. `recipes-rsrs`: `rs-to` (base) vs `rs-from` (challenger) — existing gate, strict key match.
3. `aot-rsrs`: `AOT to` (`aot-to`, base) vs `AOT from` (`aot-from`, challenger) — regression gate for `AOT from` vs `AOT to`, strict key match, fails on any MEM/INS loss unless `waive-h2h`.
4. `aot-ocaml`: `OCAML` (`ocaml`, base) vs `AOT` (`aot-from`, challenger) — `--allow-extra-challenger`, compares all 1,743 OCaml-counterpart cases cleanly, honors `waive-ocaml` / `waive-h2h` during Slice 1 baseline capture and optimization bring-up, then becomes strictly enforced (unwaived) before closing #271.
5. `aot-run`: `run` (`rs-from`, base) vs `AOT` (`aot-from`, challenger) — the same revision interpreted vs compiled, strict key match (both legs run all 394 scripts), fails on any MEM/INS loss unless `waive-aot-run` / `waive-h2h` (#259: compiled binaries must outperform `clojure-rs run`).

Upload merged JSONs (`merged-aot-from.json`, `merged-aot-to.json`, `merged-ocaml.json` alongside existing merged artifacts) so local analysis scripts can inspect exact per-case Pin instruction and RSS ratios.

## 6. Slice 2+ — Optimization Methodology

Once Slice 1 yields authoritative Linux Pin (`pin_instructions`) and RSS (`maxrss_kb`) distributions for `AOT` vs `OCAML`:

1. Categorize losses by:
   - small-case startup/runtime RSS floor vs OCaml (~8–10 MB OCaml floor),
   - steady-state allocation / peak heap working set,
   - instruction-count hot families across `clj-aot`, `clj-bc`, `clj-rt`, `clj-value`.
2. Ship focused, well-tested slices, each verified against:
   - `rust.yml` full gate suite (100.00% line & patch coverage, MSRV, `--no-default-features`, complexity, geiger),
   - `h2h-cpu-mem.yml` showing measurable progress on `aot-ocaml` and zero regressions (`0 MEM`, `0 INS`) on `aot-rsrs`, `recipes`, and `recipes-rsrs`.
3. Remove `[waive-ocaml]` usage once unwaived `aot-ocaml` shows AOT outperforming OCaml on every compared case (wins only on MEM and INS: no losses, no ties).

### 6.1 `BootMode::Lite` semantic parity contract

`BootMode::Lite` binaries keep no `clojure.core` / `clojure.string` source resident, so every native that stands in for an interpreted definition under Lite MUST reproduce `clojure-rs run` for **every** input — values, exception classes, and messages — not only the inputs its fast path takes (Invariant 2):

1. **Natives replacing `core.clj` definitions** (`reduce-kv`, `run!`) transcribe the definition literally: a non-fn is invoked exactly as the interpreted `(f …)` invokes it (the interpreter's own invoke error, never an `IFn` pre-check), and `reduce-kv` takes `core.clj`'s index loop for every `vector?` (subvecs, `vector-of`), not only persistent vectors.
2. **`clojure.string` slow paths** (inputs outside a native's fast path): `run` keeps calling the interpreted `string.clj` definition; Lite runs that definition's literal transcription through the same interop entry points (`__meth*` for `.length` / `.charAt` / `.toString` / `.startsWith` …, the `Character/isWhitespace` emulation, the `zero?` / `dec` / `inc` / `split` natives), so `java.io.File`, `StringBuilder`, nil, and non-`CharSequence` receivers fail or succeed exactly as under `run`.
3. **Enforcement:** `crates/clj-eval/tests/lite_parity.rs` evaluates every case in a full-boot evaluator (the oracle) and a Lite evaluator and requires identical printed outcomes; a case either side cannot read, analyze, or realize fails the suite (no vacuous matches).
4. **Deferred (not Lite-eligible in Slice 2):** `clojure.set` and `clojure.string/replace`. Programs referencing them keep the full boot. A native `replace` / `clojure.set` cannot be exact yet: `run`'s own `replace-by` throws `No matching ctor found for class java.lang.StringBuffer` on any fn replacement that matches, and the `Matcher.replaceAll` emulation diverges from the JVM on `$10` / `\$` / empty matches; fixing `run` itself is a separate change, after which these natives land with `lite_parity` coverage. (`clojure.set` became Lite-eligible in Slice 3k without natives, by compiling its source into the program: §6.5.)

5. **Stream vars of `with-out-str` / `with-in-str`.** These analyzer builtins name an unqualified `*out*` / `*in*`, which the runtime resolves when the body runs. Under `run`, `user` refers `clojure.core`, so the name finds `clojure.core/*out*`. A Lite program refers nothing: the lookup missed and interned a fresh, non-dynamic `user/*out*`, and `binding` it threw `Can't dynamically bind non-dynamic var: *out*`. The stream var now resolves as the current namespace maps the name, so a user shadow still wins as under `run`, else through `clojure.core`. The compiler declares `clojure.core/*out*` / `*in*`, so the Lite boot installs them as dynamic vars. Corpus `docs/spec/i259-native-compiler/corpus/with_out_str.clj`.

### 6.2 Slice 2 contents

- Lite startup: dynamic vars installed only when referenced, var metadata synthesized only when a reader (`meta`, `doc`, `resolve`, …) is referenced, lazily allocated binding-count / lazy-index / tap-queue tables.
- Codegen: inline static fast paths (`Long/parseLong`, `Integer/parseInt`, `Double/parseDouble`, `Long/remainderUnsigned`, `Math/*`) that fall back to the generic `__static*` invoke for any input they do not take, variadic `+` / `-` / `*` folding (up to 8 args), exact integer `/`, numeric comparison chains, and `min` / `max`.
- New Lite natives: `reduce-kv`, `run!` (contract 6.1).

### 6.3 Slice 3a: static Linux AOT links

Linux callgrind probes (issue #271 comment) put the dynamic loader at ~43% of a small AOT program's instructions (eager GOT symbol lookups) and the shared C libraries at ~1.65 MB of its RSS. `clojure-rs build` on Linux therefore links static non-PIE first and keeps the dynamic link as the fallback (static C libraries missing):

1. **Flags:** `-static -Wl,--eh-frame-hdr -Wl,--wrap=__register_frame_info -Wl,--wrap=__deregister_frame_info`, on top of the existing `-no-pie`, hot-text script, and `--gc-sections`.
2. **No eager FDE registration:** crtbeginT's `.eh_frame` registration (GCC 13+ classifies every FDE at registration — ~415K instructions plus the whole `.eh_frame` faulted in) is wrapped to no-ops the compiler emits into every module (unreferenced and dead-stripped elsewhere); `--eh-frame-hdr` keeps unwinding (`catch_panic`) on the `dl_iterate_phdr` path dynamic executables use.
3. **Observable behavior is unchanged:** output, exit codes, and error text are the same as the dynamic binary's; only the startup floor and RSS move (probe: `(println 1)` 356K → 170K instructions, 3.2 → 1.5 MB RSS).

### 6.4 Slice 3d: per-program startup work

Linux callgrind profiles of real snippets run on empty stdin (issue #271 comment) put a typical program's startup at ~960K instructions, about 800K of it above the `(println 1)` floor and ~25% of it in `malloc`/`free`. The dominant costs were per-var and per-`defn`, so this slice removes them:

1. **Native lookup hints.** The compiler records, per declared var, where its Lite native sits in the capability tables (`aot_boot::lite_native_hint`: group, table, entry — first-match order, like the runtime scan). The binary declares all vars in one `rt_n_declare_vars` table (`ns`, `name`, hint; each namespace name shared), and boot installs a hinted native only after checking that the entry still carries the var's name, falling back to the scan otherwise (a runtime archive from another build stays correct).
2. **Var metadata only where it can be seen.** A Lite program that references no var-metadata reader (`aot_boot::needs_var_meta`: `meta`, `doc`, `alter-meta!`, `reset-meta!`, `vary-meta`, `macro?`, `resolve`, `ns-resolve`, `ns-publics`, `ns-interns`, `ns-refers`) cannot observe var metadata, so:
   - `defn`'s discarded `(alter-meta! (var f) assoc :arglists '…)` statements (and `merge`s of constant maps) are not lowered when they only write `aot_boot::LITE_INERT_DEF_META` keys (`:arglists`, `:doc`, `:line`, `:column`, `:file`) with constant values; a reader-free program then references no `alter-meta!` at all;
   - `def` names and their evaluated metadata pairs drop the same inert keys, so names pool as plain symbols instead of read-back forms; every other key (`:private`, `:dynamic`, `:tag`, `:const`, …) stays, since runtime paths (`refer`, `binding`) consult some of them;
   - interned vars carry no `{:ns :name}` stamp and the Lite dynamics no `:dynamic` stamp (binding semantics use the var's own `dynamic` flag).
   - a `def` whose name carries metadata (`defn-`'s `{:private true}`) keeps exactly that map as the var's metadata, without the `:line` / `:column` / `:file` / `:name` / `:ns` auto keys a reader would see (Slice 3n; `refer` consults `:private`, nothing reads the rest).

   A program that references any reader keeps every metadata key exactly as `run` builds it.
3. **Cheaper installs:** native symbols share their namespace's name and are qualified in one allocation; compiled var slots resolve through symbols built from their shared names.
4. **Smaller fixes:** `split-lines` takes the `\r?\n` byte cut (its source spelled the pattern with literal CR/LF, which compiled a full regex — ~275K instructions — on first use), and the static class lattice index is Fx-hashed and pre-sized.
5. **Batched installs:** boot installs each namespace's declared natives in one registry pass (`install_native_rows` → `intern_roots`: one lock, one namespace resolution, one mutation epoch per batch) with no per-var "already installed" probe (the compiler's list is distinct; only the internal helpers the program did not declare are probed). Hints carry an "owned-args entry exists" bit, so most installs skip the owned-table searches; a stale bit can only drop the owned fast path, never change results.
6. **`zipmap` and literal strings:** `zipmap` collects its pairs and builds a small result once (`Map::from_kvs_last_wins`, the `assoc` chain's exact map) instead of interning a key shape per pair — `(zipmap "0123456789" (range))` opens 238 of the 394 snippets — and continues with in-place inserts past 64 pairs; its forcing order (both heads, then both tails) is unchanged. Compiler-emitted strings decode through `str::from_utf8`'s fast path, with the lossy decoder as the fallback. Later (Slice 3n), a string's chars zipped against the core `(range)` iterate or a counted long range pair up directly. No user code runs and nothing anyone else sees is memoized, so the pairs feed the same build; overflow takes the walk. An iterate stepping the core `inc'` / `inc` natives over a long below the max computes `n + 1` without the call (corpus `zipmap_ranges.clj`).
7. **Names with metadata pool without the reader.** A pooled constant the compiler cannot build directly is printed to text and read back with the full reader at startup (about 12K instructions per constant on macOS). Every `defn-` name is such a constant: it is a symbol carrying `{:private true}`, which Lite keeps because `refer` consults it. `^String` and `^:dynamic` names are too. A symbol whose metadata maps keywords to scalars, keywords, plain symbols, or strings is now a `Const::MetaSymbol`. The constant init writes its pairs into slots by name (`rt_n_kw` / `rt_n_sym` / `rt_n_str`, scalars inline; it cannot read other pool constants before the pool freezes). `rt_n_meta_sym` then inserts them pair by pair into an empty map, exactly as the reader merges `^{…}` (`native_abi::meta_symbol`, checked against the reader). Other metadata, such as an `:arglists` list when the program reads var metadata, still reads back. Locally, no_prefix_set (ten `defn-`s) saves 110K–210K instructions per run (corpus `meta_names.clj`). Metadata-free collections of such forms build the same way: quoted data, catch class lists, and `:arglists` lists become `Const::Coll`, built by the reader's own constructors (`native_abi::const_coll` via `rt_n_coll`). Each candidate is kept only if building it at compile time prints exactly as the original does, and it is capped at 256 forms. That cuts startup reads across the snippets from 212 to 9 (only the queue constant, which carries metadata, still reads). It also fixes a divergence: the reader stamped `:line`/`:column` on macro-built lists such as `:arglists`, which carry none under `run` (corpus `const_colls.clj`).

### 6.5 Slice 3k: spliced libraries

The full boot costs ~14M instructions per run (Linux Pin), against OCaml's ~0.6M for a whole small case, so every program forced onto it loses all its cases. Four snippets took the full boot only because their leading `ns` form requires `clojure.set` (three call `union` / `intersection` / `difference`; one never calls it).

1. **Splice:** when a program's first unit is an `ns` form whose `:require` names a library in `aot_front::SPLICED_LIBS` (`clojure.set`), the front end reads, expands, and analyzes that library's vendored source (the same embedded text `run`'s `require` loads) and prepends its units, recorded as `FrontEnd::lib_units` / `FrontEnd::libs`. A `require` anywhere else (a later unit, a `(require …)` call) does not splice.
2. **Exact by construction:** the library's definitions compile like user code, so they behave exactly as `run`'s interpreted library: values, thrown classes and messages, non-set arguments, the variadic arities, and the one-value-per-form `#{}` of `union`'s zero arity. No native stands in for them, so contract 6.1's transcription rule does not apply. The i259 corpus program `clojure_set.clj` covers every `clojure.set` fn (`jit` ≡ `run`, the same module `build` links).
3. **Boot selection:** the Lite check accepts a `:require` spec naming a spliced library (the namespace's vars resolve to the spliced definitions); at runtime a Lite `ns` form only switches namespaces, as for `clojure.string`. A program that needs the full boot for another reason drops the spliced units and loads the interpreted library through its own `require`, exactly as before (corpus `clojure_set_full.clj`). The `aot: native N/N defns` report counts the program's own definitions only.
4. **Effect (macOS instructions, unchanged output):** ssa_dominance, dfa_minimization, hindley_milner, and ll1_parser each drop ~14M per case (e.g. ssa_dominance 37.1M → 22.7M, including macOS's ~20M process floor).
5. **Selective libraries (`clojure.string`):** a library whose vars are mostly Lite natives splices only what the program reaches that no native covers. The front end collects the library vars the program names without a native (`aot_boot::is_lite_string_var`), then keeps the top-level forms defining them and, transitively, the forms those reach. It also keeps the library's `ns` form with its clauses dropped: the build evaluator applied them, and a Lite `ns` form only switches namespaces. `TopUnit::form` groups each `defn`'s split units, so `def`, metadata write, and `var` stay together. If a needed var has no definition there, nothing is spliced. A program that looks vars up at runtime (`resolve`, `ns-resolve`, `ns-publics`, `ns-interns`, `ns-map`, `find-var`) splices nothing either, since a subset would show there. A `def` into `clojure.string` blocks Lite only when it redefines a native-served var, so spliced definitions keep it. knuth_div and cantor_expansion (`str/replace`) leave the full boot: macOS instructions −9M to −12M per case and RSS about −2 MB, with unchanged output (corpus `clojure_string_replace.clj`). The spliced `replace` reproduces `run` exactly, including #273's divergences from the JVM.

### 6.6 Slice 4a: stable var roots and borrowed operands

Per-operation costs, not startup, separate AOT from OCaml on most remaining cases. Linux callgrind on loop microbenchmarks (`probe/mb`) priced a global var read at about 100 instructions: `rt_bc_deref` (epoch, slot, and binding checks plus an atomic `Arc` increment) and the matching drop. Every pooled constant used as an argument also went through an `rt_n_const` clone and a drop. `(:k m)` paid the general `rt_n_invoke` (about 450 instructions on a three-key map).

1. **Stable vars.** In a Lite program in which no unit can re-root an arbitrary var (`mentions_rebinding`: `alter-var-root`, `with-redefs`, the var-resolving fns, …), a var is stable when exactly one `def` with an init names it anywhere in the program. That `def` must be a whole top-level unit, the var must not be `^:dynamic`, and the program must never `set!` it or take it as a var object (`#'v`) outside the inert units `defn` emits. Its root cannot change once that `def` has run.
2. **Root caches.** Each stable var gets an internal module global, holding `nil` until its `def` copies the root in (`rt_n_deref`), and a set flag stored with release ordering. A call that only reads its arguments (bitcode helpers, `rt_n_op*`, the borrowed core call, direct `defn` calls) is passed the cache itself; other reads clone from it. Code in a later unit than the `def` can only run after the `def` has run, so it skips the flag. Code in the `def`'s unit or earlier checks the flag (acquire) and falls back to the var while the cache is unset.
3. **Borrowed constants.** `cljtop_0` stores the frozen constant table's base (`rt_n_consts_base`; the table never moves after `freeze`), and read-only call operands borrow `&consts[k]` in place.
4. **Keyword invokes.** `(:k coll)` and `(:k coll not-found)` with a literal keyword go through the bitcode `get` with the keyword borrowed from the pool. Any shape the helper leaves to the runtime takes the keyword's own invoke. Both use the interpreter's `lookup`, so results are identical.
5. **Keyword equality is pointer equality.** Every keyword is interned (`Keyword::new` is the only constructor, and the table keeps each entry for the life of the process), so `Keyword::eq` no longer falls back to comparing strings when the pointers differ. `Hash` is unchanged, because hash-trie iteration order depends on it.
6. **Effect (Linux callgrind, per iteration):** a vector `nth` through a global drops from 271 to about 170 instructions. On macOS, `(:b rec)` falls from 661 to 139, `contains?` on a global set from 485 to 361, and `get` on a global map from 560 to 440. Snippet output is unchanged (corpus `stable_vars.clj`: forward references, dynamic, re-`def`ed, nested, and `#'`-taken vars, and every keyword-invoke shape).

### 6.7 Slice 4b: cheaper direct calls

A direct call of a compiled `defn` cost about 290 instructions on macOS beyond its body. `rt_n_enter_fn` checked that the var still held that fn (thread-local caches, the registry epoch) and the call depth. The callee also cloned every parameter whose last use the move analysis recorded, even when that use only read it.

1. **No root check for stable `defn`s.** A direct-call target's `def` init is its `fn` literal, so a call site in that `def`'s own unit (the body, including recursion) or a later unit can only run once the root is the compiled fn. If the var is stable (§6.6), such a site calls `rt_n_enter`, which only checks call depth (the `MAX_CALL_DEPTH` / stack red-zone fallback is unchanged). Forward calls through `declare` keep `rt_n_enter_fn`.
2. **Borrowed parameters.** A parameter whose every use is an operand of a core call outside the owned-move ops, or of an invoke, keeps reading the caller's argument array. An owned copy at entry can buy no in-place update for it. Parameters reaching an owned-move op, a `let`, `recur`, or the result still take or clone their value as before, including `reduce`'s handed-over accumulator.
3. **Effect (macOS, per iteration):** `(defn pick [w i] (nth w i))` called in a loop falls from 422 to 231 instructions (call overhead about 290 → 96). Corpus `direct_calls.clj` covers recursion at depth 20,000, forward calls, and every parameter shape.

### 6.8 Slice 4c: inline traversal loops

`doseq` expands to `(dorun (map (fn [x] …) xs))` and `for` to a lazy `map` of a `fn`, so every iteration of these and of `mapv` / `reduce` with a literal `fn` was a lazy-seq step plus a callback into a compiled closure. That cost 400–550 instructions per item beyond the body (Linux callgrind: `doseq` over a range 531, `for` 548).

1. **Shapes.** `(dorun (map F xs))`, `(mapv F xs)`, `(vec (map F xs))`, `(into [] (map F xs))`, and `(reduce F init xs)`, all `clojure.core` in a program that cannot rebind core vars, where `F` is a literal `fn` with one arity of exactly the needed symbol params (one, or two for `reduce`) that never `recur`s to itself. A named `fn` arrives as a `letfn`, so `F` never refers to itself.
2. **Direct loop.** The collection (and `reduce`'s init) evaluate in argument order. `rt_n_iter_plan` (`native_abi::iter_plan`) then classifies the collection. A persistent vector iterates by index (`rt_n_vec_item`) and a long range by arithmetic (wrapping, so the unused value after the last cannot trap). The `fn`'s body runs in place with its params bound like a `let` at the env indices the analyzer gave them. It is lowered with its own last-use moves: a param or a body local may move, while an enclosing local is read afresh every item. `mapv` and `vec`/`into` accumulate into a uniquely held vector (`rt_n_vec_push`). `reduce` stops on a `reduced` value (`rt_n_unreduce`).
3. **General path.** Any other collection (lazy seqs, lists, maps, sets, strings, nil) makes the `fn` a closure and calls the original core functions over the already-evaluated collection, with the arguments moved over as the original call took its temporaries. `run`'s semantics therefore hold exactly, lazy forcing order and `walk_sole_map` fusion included.
4. **Moves.** The analysis saw the closure capture its free locals before the collection evaluated, so it may have let the collection move a local the body still reads. While the collection and init lower, those locals' moves are suppressed.
5. **Effect (macOS, per item):** `doseq` over a range 423 → 22 instructions, `reduce` with a literal `fn` 460 → 138, `mapv` over a vector 449 → 307 (the accumulator push dominates). Corpus `hof_loops.clj` covers every shape and collection kind, early `reduced`, exceptions in the body, nested loops, captured locals, and lazy forcing order.

### 6.9 Slice 4d: fewer runtime boundaries for small programs

70% of the instruction non-wins (458 of 655 at `b4a57992`) are cases where OCaml's whole run is under 0.7M instructions. Linux callgrind of every losing script with empty input puts AOT's per-program startup at a median 293K (trivial floor 96K). Exact per-case call counts (an instrumented runtime archive, macOS) show the small cases crossing these runtime boundaries most: `rt_n_nth_nil` (146 per case: vector destructuring), `rt_n_vec_lit` (66), and closure creation (48).

1. **Destructuring reads inline.** `(let [[a b] v] …)` and destructured params read each position with the inlined bitcode `(nth v i nil)` (vectors, strings, nil), falling back to `rt_n_nth_nil` for every shape the helper leaves to the runtime. macOS: a two-element destructure in a loop drops from 481 to 352 instructions per iteration.
2. **Vector literals take their items.** `rt_n_vec_lit` moves the evaluated items out of their slots (each left nil) instead of cloning them (`native_abi::vec_lit_owned`). All-long items go straight to packed storage, the storage `Vector::from_vec` would choose, without an intermediate boxed vector.
3. **Loop accumulators without copy-on-write.** A compiled `mapv` / `vec` / `into []` loop (§6.8) gathers its items in a `VecBuilder`. Only the loop holds it, so a push is a uniqueness check and a buffer push: no `Arc::make_mut`, and no nil drops. Items stay packed while every one is a long, and the builder freezes into one persistent vector at the end (`rt_n_vec_done`). `mapv` over a vector: 307 → 233 macOS instructions per item.
4. **Var metadata only where it can be read.** The by-name readers (`doc`, `macro?`, the var-resolving fns) find vars themselves, so they still count. The value readers (`meta`, `vary-meta`, `alter-meta!`, `reset-meta!`) now count only when the program can hold a var object, meaning a `#'v` outside `defn`'s own forms (`needs_var_meta(vars, var_objects)`). A discarded constant metadata write whose keys the runtime never acts on (anything but `:dynamic`, `:private`, `:macro`, `:redef`) is inert: an attr map's `:added`, a `:tag` hint. The compiler passes its decision to the boot (`rt_n_cap_var_meta`), so both agree. `clojure.set`'s `select` and `rename` read their relations' metadata and its attr maps write `:added` / `:tag`, so every program that spliced it kept full def metadata. hindley_milner, ssa_dominance, ll1_parser and dfa_minimization each save 0.6–0.9M macOS instructions per run (Linux startup about 1.2M before). Corpus `var_meta_reads.clj` and `set_meta_data.clj`.

### 6.10 Slice 4e: printing reads its settings vars through a cache

Exact Linux costs of single operations come from callgrind on synthetic programs on the fork, by slope between two iteration counts. `println` cost about 14.2K instructions per call, even for a constant string. About 12K of that went to resolving about 13 `clojure.core` settings vars by name per call: `*out*` per argument and again for the newline and the flush, `*flush-on-newline*`, the six print options, and `*print-readably*` for the `print` binding. Each `deref_core` string-hashed the namespace and the name, allocated a fresh `Symbol`, then looked the var's state up again through its home namespace.

`Evaluator::core_read` keeps each probed `clojure.core` name's id, or its absence, and its root while the lineage epoch holds. Thread bindings are still checked first on every read. Every root write and namespace mutation bumps the epoch. So now do `remove-ns` and any intern that creates a var: a Lite program installs a print var on first reference, after printing may already have cached its absence. Names are literals, so entries match by address before by text. Linux, exact: `println` 14.2K → 8.5K per call with found names cached; caching misses saves about 2.6K more per `println` on macOS. Most cases print 1-4 lines (median 2). Corpus `print_settings.clj` pins bindings, root writes and nested `with-out-str` against `run`; `print_late_vars.clj` pins print vars first referenced after a print.

### 6.11 Slice 4f: less fixed work per program

Linux callgrind of every losing script's empty-input run, profiled function by function, prices each program's fixed work:

1. **`slurp` of stdin.** It read into an 8 KiB zero-filled first chunk. Pin and callgrind count the bulk zero-fill per byte, so that cost about 8.4K instructions per program, although inputs are tiny (median 26 bytes; 94% at most 8 KiB). Closing the reader then built a fresh parser (about 3.4K) over one `slurp` had just drained. Stdin now reads through `read_to_end` into one buffer that is never zero-filled. std probes small inputs on the stack and grows large buffers by `realloc`. The text is validated once, with `\r\n` and a lone `\r` normalized in one pass, and `slurp` only latches the reader closed.
2. **Declared var names.** `rt_n_declare_vars` received an `ns`/`name` string pair per var: two UTF-8 validations and an `Arc<str>` allocation per var, about 22K for a typical program's 42 vars. The compiler now emits one names text, each namespace once, then every var name, plus a `(ns span, name span, hint)` row per var. The runtime validates and copies the text once, and slots hold spans. Only the vars a program actually interns build a `Symbol`.
3. **Docstrings.** `def` built its docstring even when the program keeps no var metadata, which never reads it (§6.9 item 4).

Exact Linux startup (callgrind, empty input, 365 scripts) for the three together: median 256K → 225K instructions. Every script saves between 22K and 41K (median 29K).

### 6.12 Slice 4g: unread pure defs drop

Most snippets carry the same input boilerplate. A `digit-values` table `(zipmap "0123456789" (range))` appears in 220 of 393 scripts and is never read in 177. Likewise `separators` / `whitespace-chars` character sets (unread in 165) and `max-long` `(bit-not min-long)` (unread in 140; it holds the only read of `min-long`). Each unread def still built its value at startup (a `zipmap` costs about 8.8K instructions on Linux, a 6-char set literal about 4.1K) and ran the def machinery (about 2.8K per def).

A Lite program drops a top-level `def` (`dead_defs`) when:

1. its var is stable (§6.6);
2. nothing reads, binds, sets, declares or takes the var as an object (`visit_var_reads`);
3. the program cannot reach vars by name (no `var-get`, `refer`, `load-string`, …; the Lite check already excludes `resolve`, `eval`, `intern`, `ns-publics`, …);
4. its init is pure and total (`is_pure_total`). That means a constant, a collection literal of constants, a long built from long literals, known-long vars and the bit operations, or `(zipmap "text" (range n?))`.

A dropped def's own reads no longer count, so the set grows to a fixpoint: `max-long` frees `min-long`. An unread def whose init has an effect still runs it. macOS empty-input startup: −87K or more for a quarter of the scripts (median −12K). Emit test `native_emit_ir_drops_unread_pure_defs_271`; corpus `dead_defs.clj`.

### 6.13 Slice 4h: generic core calls without the argument buffer

A core fn the compiler neither inlines nor calls directly goes through `rt_n_core_call` (and the `rt_n_op*` fallbacks). That path cloned the arguments into a `SmallVec<[Value; 4]>`, moved the 72-byte buffer through three call layers, and returned a `Step` (`Result<Value, Signal>`, about 60 bytes). On Linux the call cost about 470 instructions; for `identity`, the native's own share was 10. The non-win scripts make about 1300 such calls per case.

A native that takes its arguments borrowed, which is most of them, now gets 1-3 clones in a stack array and writes the result straight to the output slot. A `Signal` is built only on failure, exactly as the general path throws. Owned-args natives (`reduce`, `into`, …) and anything else keep the general path. macOS, per call: `identity` 401 → 304 instructions, `subs` 837 → 737, `name` 608 → 519.

### 6.14 Slice 4i: common callbacks without the native call

Higher-order natives (`mapv`, `filterv`, `reduce`, the lazy `map` / `filter` steps) resolve their fn once (`Prepared`) and call it per item. Two common callbacks paid for a native call that their results did not need:

1. **`seq` as a filter predicate.** `(filterv seq tokens)` drops empty strings after a split, and each call allocated a string seq only to test its truthiness. The lazy `filter` already tests emptiness instead (`filter_pred`: nil, lists, vectors, maps, sets, strings, realized seqs; anything else still calls `seq`), and `filterv` now shares that predicate preparation (`filter_prep`). macOS: 639 → 201 instructions per item.
2. **`inc` / `dec`.** `Prepared::call1` now steps a long inline when the native carries the `inc`/`dec` arithmetic tag, as `call2` already did for the binary tagged ops. Overflow and every other operand still call the native, so its errors and numeric tower stay exact. macOS: `(mapv inc v)` 404 → 261 instructions per element, the lazy `map` and `reduce` over it alike.

### 6.15 Slice 4j: a fresh hot-text order, from generic workloads only

Linux AOT links apply `rust/hot-text-aot.ld` (embedded in clj-aot): a linker script listing the runtime functions that typical programs execute, which the linker packs into one `.text.hot` section. The kernel faults file-backed text in 64 KiB windows, so a run touches a few windows of the hot section instead of most of the runtime's 5-6 MB of code. Code pages dominate a short program's peak RSS.

The order names functions, so it goes stale as runtime code moves: renamed or new functions fall back to the cold `.text`. The committed order (generated from the 30 workloads in `scripts/bench/hot-text` in Slice 3c, and last regenerated for Slices 3k-3n) had drifted, and MEM wins fell from 1372 (b4a57992) to 1280 (90509c29).

**Training set.** The order is trained only on `scripts/bench/hot-text/workloads` and `aot-workloads`: 30 small generic programs (map-assoc, regex, sort, strings, stdin readers, …) committed in #246, before this goal, plus `core-ops.clj` (Slice 6o, 6.72). It is never trained on the benchmark snippets or their test inputs. AOT will run other code bases, and an order fitted to the measured programs would not carry over.

**Measured** with the m-hu fork probe (151 programs, empty stdin, RSS at exit, the same runtime with four orders):

| Order | Median RSS | vs committed | Projected aot-ocaml MEM (win/tie/loss) |
|---|---|---|---|
| committed (stale) | 2936 KB | — | 1280/408/55 (= h2h 90509c29) |
| none | 3464 KB | +512 KB (worse for 151/151) | 297/869/577 |
| regenerated from the workloads | 2436 KB | −448 KB (better for 151/151) | 1701/32/10 |
| trained on the snippets (rejected) | 1940 KB | −960 KB | 1729/6/8 |

The projection adds each probed program's RSS change to its h2h 90509c29 cases, and the median change to the rest.

**Tooling.** `scripts/bench/hot-text/regen-aot.sh RUST_WORKSPACE [OUT.ld]` works as follows:

1. Builds the revision's runtime with a blank order.
2. Compiles each workload with `clojure-rs build` and profiles it under callgrind (stdin from `NAME.in`, else empty).
3. Lists the functions they run with `hot_text.py`, the generator of the interpreter's `rust/hot-text.ld` and of the committed AOT order: the functions most workloads run come first, ties in the order the profiles first ran them. (Until f955bd39 a separate `aot_order.py` packed 62 KiB windows of co-used functions; see the second correction below.)

Ties break in a fixed order, so every job that links the order gets the same one. `run.sh --aot` delegates to the script.

**Correction after the first CI runs.** The probe above ran on empty stdin, so it measured mostly startup. With real inputs, the h2h for f9a16ff5, the first run with a CI-regenerated order, regressed instead: MEM 1280/408/55 → 1046/597/100, and AOT median peak RSS rose 116 KB.

The runtime changes in between were not the cause: on macOS, which has no layout order, peak RSS on the same inputs fell (median −80 KB). A same-runtime probe on empty stdin still ranked the CI order 192 KB better than the committed one.

The first suspect was the generator's name filter. `aot_order.py` kept only Rust-mangled names, so it dropped the runtime's `rt_*` C-ABI entry points (`rt_n_core_call`, `rt_n_drop`, `rt_n_deref`, `rt_n_core_own`, …). Every compiled program calls these while it processes input, but empty-stdin runs barely touch them. The committed order (from `hot_text.py`) listed them. f955bd39 kept `rt_*` names alongside Rust ones.

**Second correction.** That was not the whole cause. The h2h for f955bd39 (run 44411733078) still gave MEM 1024/616/103 and an AOT median of 2992 KB, against 2876 KB with the committed order at 90509c29. Per case, the shift from the committed order is broad, not a constant: median +116 KB, 5th to 95th percentile −160 to +368 KB. 300 cases that won MEM at 90509c29 no longer win.

Both orders come from the same 30 workload profiles. The one remaining difference is the layout: the committed order lists functions by how many workloads run them (`hot_text.py`), while `aot_order.py` packed 62 KiB windows of co-used functions. `regen-aot.sh` therefore lays out with `hot_text.py` itself, and `aot_order.py` is gone. The next h2h checks the result.

**Third correction: coverage, not layout.** With `hot_text.py` itself, the h2h for fdc5a900 gave MEM 1037/612/94 (AOT median 2976 KB). After the next runtime slices, dd228091 dropped to 627/857/259 (3176 KB), while the master leg without an order stayed at 7196 KB. On macOS, which links no order, the worst-hit scripts used the same or less RSS at dd228091. The jump was text pages, not heap.

Two fork probes on Linux (generic programs, no snippet inputs) settled the cause:

- **Revision by order.** On the i259 corpus programs, dd228091 with a fresh order (2968 KB median) matched 90509c29 with its own orders (2996 fresh, 2968 committed). Code growth alone cost about 120 KB (3680 → 3800 KB with no order).
- **Held-out training.** Orders trained on the workloads alone, and on the workloads plus half of the corpus, were measured on the other half of the corpus, which neither saw. The wider training cut median peak RSS by 508 KB there (p10 −576, p90 −256). It cost 62 KB on the workloads themselves.

The snippets run runtime paths the 30 workloads never reach. Each such function sits in the cold text and faults in its own 64 KiB window. `regen-aot.sh` now also profiles the compiler's test corpus (`docs/spec/i259-native-compiler/corpus`, empty stdin): generic programs, still never the benchmark snippets or their inputs.

**CI keeps both orders fresh.**

- The h2h workflow's `hot-text` job regenerates each side's interpreter order (`rust/hot-text.ld` via `regen-run.sh`, generic 26 workloads) and AOT order (`rust/hot-text-aot.ld` via `regen-aot.sh`) once, using this run's tooling and workloads, and the 32 measure shards link with them. (At ae942dbd, the stale committed `rust/hot-text.ld` had let `rs-from` drift +526 KB above `rs-to`, tripping 6 `recipes` MEM losses against CPython.)
- `rust-release.yml` regenerates both orders before the Linux release build, so shipped binaries never carry a stale order.
- A failed regeneration keeps the committed order.
- The committed copies only serve local builds and are refreshed from the same pipeline.

### 6.16 Slice 4k: map destructuring by direct lookups

The AOT lowered every map pattern (`{:keys [a b] :or {…} :as m}`) to `rt_n_destructure`. That call decoded the pattern constant back into a `Pattern` and ran the interpreter's whole destructure in a fresh `Env`, on every bind: about 7.8K macOS instructions for `(fn [s {:keys [a b]}] …)`, against about 700 for the vector pattern `[a b]`.

A plain map pattern now binds by direct lookups, as the interpreter binds it. Plain means: no required (`!`) keys, no `&` key list, and no 1.13 outlets.

1. `rt_n_destr_map` converts the source: a list becomes its map, and anything else is itself.
2. Every `:or` default is evaluated once in the enclosing frame.
3. Each entry and each `:keys`/`:strs`/`:syms` local is read by `rt_n_destr_get`, which uses the interpreter's own `lookup` with the key passed in place from the constant table. A present key wins (a present `nil` stays `nil`). Otherwise the local takes the default keyed by its symbol, then the default keyed by its key form, then nil.
4. `:as` binds the converted source.

Locals follow `pattern_names` order, and nested patterns lower recursively. Patterns that are not plain keep the generic path.

Measured on macOS, instructions per element: `{:keys [a b]}` 7777 → 1069; `{:keys [a b c] :or {c 1} :as m}` 12598 → 1371; `{:strs [a b]}` in a `let` 7580 → 761.

The corpus file `map_destructure.clj` replays the JIT against `run` across every source shape: lists, records, vectors, strings, sets, numbers and nil. It also covers present `nil`, both kinds of `:or` key, eager defaults, nested patterns, and required keys.

### 6.17 Slice 4l: lazy steps walk chunks without copying them

`step_head` serves every lazy op that consumes its input one item at a time: multi-input `map`, `take-while`, `drop-while`, `drop`, `take-nth`, the `partition` family, `distinct`, `reductions`, `interleave`, `sequence` and `cycle`. It used to re-chain the rest of each stepped chunk as a fresh copy (`chain_tail`), so walking a 32-item chunk item by item copied about 16 values per item and allocated a vector for each.

The rest of a chunk is now a forced cursor cell sharing the chunk: `SeqState::Chunk { items, pos, tail }`, the cell `rest` already uses (the JVM's `ChunkedCons`). `step_head` peels a cursor cell in place without forcing it. Values, laziness, `realized?` and the seq classes are unchanged (the remainder keeps the `LazySeq` class the re-chained copy had), and `run`, AOT and the previous binary agree on every op above.

Measured on macOS, instructions per element over a 100-item vector:

| Op | Before | After |
|---|---|---|
| `(partition 2 v)` | 6509 | 4285 |
| `(partition-all 3 v)` | 5676 | 3335 |
| `drop-while` | 4249 | 1508 |
| `take-while` | 6311 | 3551 |
| `distinct` | 7474 | 4694 |
| `(map + v v)` | 10784 | 5199 |
| `interleave` | 12662 | 7071 |
| `reductions` | 6634 | 3878 |

### 6.18 Slice 4m: keyword map literals share a shaped template

A map literal lowered every key into a value slot. It then went through `map_pairs` (a pairs vector, `realize_key` per key) and `from_kvs_last_wins` (a dedupe scan plus separate key and value vectors), then `from_parts`, which probes the shape registry. That is about 1.5K macOS instructions more than a two-element vector literal for `{:x x :y y}`, the usual record-like map.

A literal without metadata whose keys are 1 to 8 distinct constant keywords now lowers to `rt_n_kw_map`. The call takes a `{k nil …}` template constant (rebuilt at startup like any constant map, so it holds the interned shape) and the evaluated values, in order. `Map::with_vals` shares the template's shape `Arc` and stores only the values. Keys, iteration order, equality, class (`PersistentArrayMap`), and the order in which values evaluate are unchanged. Any other literal keeps the pairs path.

Measured on macOS, instructions per iteration: `(count {:x i :y i})` 2098 → 652 (a two-element vector literal is 609); four keys 3205 → 785; `(:x {:x i :y i})` 2148 → 697.

### 6.19 Slice 4n: `aget` and `aset` as inline ops

`(aget a i)` and `(aset a i v)` went through the generic core call: the arguments were cloned into a buffer, and the native's `Result` plumbing came back. That cost about 740 macOS instructions per element of an `aset` loop. Both are now inline ops (`APPLY2`/`BINARY` "aget", `APPLY3`/`TERNARY` "aset"), so the AOT lowers them to `rt_n_op2`/`rt_n_op3` and the VM to its op steps.

- `aget` reads an in-bounds long index under one lock, with `array_read`'s slot mapping.
- `aset` stores only the unambiguous cases: an in-bounds long index, with a `long`, `int` (in range), `double`, `boolean` or `char` slot taking its own value type.
- Every other shape returns `None`, so the native gives its exact answer or error: negative or out-of-bounds indices, coerced indices, `int` overflow, mismatched value types, object and nested arrays, nil.

The corpus file `arrays_inline.clj` pins every case against `run` and against the previous binary.

Measured on macOS, instructions per element:

| Loop | Before | After |
|---|---|---|
| `long-array` `aset` loop | 738 | 291 |
| `double-array` `aset` loop | 746 | 298 |
| `int-array` sum by `aget` | 644 | 267 |
| a DP step `(aset a i (+ (aget a (dec i)) i))` | 1331 | 509 |

### 6.20 Slice 4o: identical collections compare without a walk

Value equality on maps, sets, vectors and lists walked every entry, even when both sides were the same collection. `swap!` publishes only when the atom still holds the value it read (`state.value != old`), so every swap on an atom holding a collection walked the whole collection. The common memo and accumulator pattern `(swap! cache assoc k v)` was quadratic: 2000 swaps into a growing map cost 721M macOS instructions, about 360K per swap at the end.

Equality now answers true first for the same collection (`Arc::ptr_eq`), as the JVM's `Util.equiv` checks `k1 == k2` before anything else. The same 2000 swaps cost 14.5M (500 cost 5.2M: linear). Clojure's `=` goes through its own `equiv` and is unchanged; the shortcut serves internal equality: atom publishes, map keys and set members.

### 6.21 Slice 4p: eager walks fuse `mapcat` (`for` over several bindings)

`for` with several bindings expands to a `mapcat` over inner `map`s, and so does a `mapcat` call. Every element went through lazy cells even when an eager consumer was about to walk the whole seq. `for` is used by 47% of the scripts that lose by more than 2×, against 13% of all-win scripts. A sole unforced `map` was already walked without its cells (`walk_sole_map`).

The fused walk now also takes a sole unforced `mapcat` whose outer `map` is sole and unforced. It walks the outer `map` as before, then each of its results in turn:

- nil, lists and vectors item by item (as `splice_realized` splices them);
- a sole `map`/`mapcat` fused again, recursively;
- anything else unit by unit through its seq.

These are the steps forcing the cells runs, in the same order. A visitor's stop ends the whole walk.

The fusion only applies where the walk owns the seq, so nothing else can see the cells. Its consumers are the `walk_seq` callers: `reduce`, `into`, `mapv`, `vec`, `some`, `every?`, `not-any?`, `frequencies`, `group-by`, `doall`, `dorun` and `run!`. The corpus file `for_fused.clj` pins items, side-effect order and counts (including chunked over-evaluation before an early stop), errors, and reuse of a held seq against `run` and the previous binary.

Measured on macOS, instructions per element:

| Consumer | Before | After |
|---|---|---|
| `reduce` over a 2-D `for` | 2135 | 1147 |
| `into []` over a 2-D `for` | 2204 | 1221 |
| `vec` over a 2-D `for` | 2637 | 1586 |
| `some` over a 2-D `for` | 2340 | 1354 |
| `(reduce + (mapcat …))` | 979 | 862 |

`count` keeps the cells for now: its AOT slow path passes the seq borrowed, and a borrowed seq may be used again.

### 6.22 Slice 4q: calls through a value

A call whose head is not a known var, such as a local closure or a fn argument (`(f x)` inside a higher-order fn), went through `rt_n_invoke` and the general `Evaluator::call`. That path ran a VM-closure hook, a struct check and two downcasts, then `call_code`'s depth guard and evaluator scope, and finally the compiled code. It cost about 4× a direct `defn` call.

1. `native_abi::invoke` calls a native callee's `func` straight away. Compiled closures are natives, and for them the general call reached the same `func` anyway.
2. A compiled closure carries its single fixed arity's entry (`NativeFn::compiled`: code address, captured values, parameter count). At a value call site, the AOT asks `rt_n_enter_closure` for it. That function checks the callee and its arity and performs the same call-depth check as `rt_n_enter`. On a hit, the site calls the code directly with the argument slots and closes the call with `rt_n_leave`, exactly as a `defn`'s direct call does. Every other callee and arity, and the depth error, keep `rt_n_invoke`.

The corpus file `closure_calls.clj` pins captured values, nested and returned closures, multi-arity, variadic and wrong-arity calls, non-closure callees (keywords, maps, sets, vectors, natives, nil), exceptions crossing the call, and the `StackOverflowError` of deep recursion through a local fn, against `run` and the previous binary.

3. As a `defn`'s direct call does, plain locals ride in place. A local head is called where it lives, and local arguments are bitwise aliases in the argument array (the callee only reads its args), cleared after the call. A local moved by its last use drops after the call. Moves never target captured values, so nothing the enclosing closure owns is released.

Measured on macOS, instructions per call:

| Call | Before | Direct entry | + in-place locals |
|---|---|---|---|
| local closure in a `conj` loop | 710 | 587 | 478 |
| a fn parameter called per item | 477 | 354 | 245 |
| a captured-adder closure | 475 | 353 | 245 |

### 6.23 Slice 4r: scalar loops keep exits over outer locals

The scalar loop compiler (`vm_scalar`, shared by the VM and the AOT) gave up on the whole loop when an exit value had no scalar form. The common case is a loop that fills a transient and returns it persistent:

```clojure
(let [pi (transient (vec (repeat m 0)))]
  (loop [i 1 j 0]
    (if (>= i m)
      (persistent! pi)
      (do (assoc! pi i j) (recur (inc i) …)))))
```

That exit misses on every run, and the AOT does not lower a loop with transient effects that may deopt (a resumed iteration would replay its effects). So the loop stayed boxed, and every `nth`, comparison and `assoc!` in it went through boxed values.

An exit now runs boxed when the loop ends (`ExitVal::Boxed`) if all of these hold:

- its scalar form always misses;
- it is the outer loop's own exit, not an inner loop's value;
- it reads only locals bound before the loop: literals, vars, calls, collection literals, and `if` over such forms. An exit that reads a loop variable or a body binding keeps its scalar form.

The AOT lowers such an exit as ordinary code after the scalar loop. The effects it can observe already happened in place, in order. The VM's scalar loops deopt at that exit as before, and the boxed loop evaluates it.

The corpus file `scalar_boxed_exits.clj` pins `persistent!`, `count`, map and vector exits, an exit that reads the loop variable, and a throwing exit, against `run`.

Measured on macOS: KMP substring search (`string_related/kmp/substring_searching`) fell from 163M to 67M instructions on one input; its prefix-table loop is now scalar.

### 6.24 Slice 4s: lookups without the general call

Three lookup paths still went through generic machinery that the lookup itself never needed:

1. `(:k coll)` and `(:k coll nf)` with a literal keyword try the bitcode `get`, which answers nil, vectors, strings and shaped maps with scalar keys. Every other shape (a hash map past 16 entries, a record, a set) fell back to the keyword's own invoke through `rt_n_invoke` and `Evaluator::call`. That invoke is the interpreter's `lookup` with the default (`invoke_keyed_name`), so the fallback now calls `rt_n_destr_get`, the same `lookup` with the default, directly.
2. A call through a value (`(m k)`, `(s x)`, `(k m)` with the callee in a local) that misses the compiled-closure entry now tries `rt_n_lookup_call` before `rt_n_invoke`. A map, set, keyword or symbol callee with one argument, or two with a default, answers there with the lookup `Evaluator::invoke` ends in (`lookup`, `member`). Any other callee or arity takes `rt_n_invoke`, whose `native_abi::invoke` makes the same shortcut for its other callers.
3. The inline `get` and `contains?` ops (VM and `rt_n_op2`/`rt_n_op3`) answered unsorted maps and sets only for scalar keys, so a tuple key such as `[r c]` or a double paid the generic core call. They now accept any key except an extension value. The natives realize only a pending seq key (an extension) before the same `Map::get`, `Map::contains_key` and `Set::contains`, so the answers are the natives' own.
4. A shaped map finds a keyword key by address: keywords are interned and equal only to themselves, so this is `Value::eq`'s answer without its dispatch per key.

Measured on macOS, instructions per element: `(:k m)` on a 40-key hash map 713 → 489; a 4-key map called as a function 690 → 520.

The corpus file `lookup_calls.clj` pins keyword, map and set calls with and without defaults on shaped and hash maps, records, sets, sorted maps, nil, vectors and strings, tuple, double and lazy-seq keys, and the arity errors, against `run`.

### 6.25 Slice 4t: core-var reads by name address

Printing reads about ten `clojure.core` vars per call: `*out*` three times, six `*print-*` settings, `*flush-on-newline*` and `*print-dup*`. `Evaluator::core_read` cached each name's var id and root per lineage epoch, but found a name by scanning the cache, which by then held some 50 names. The scan cost every `println` and every `format` about 2.2K Linux instructions (`deref_core` in the probe below). `print` also resolved `*print-readably*` by name through the registry for its binding on every call.

The cache now keeps a 64-way direct-mapped table from the name's address to its position. Call sites pass their own literal, so a probe hits without a scan; a collision falls back to the scan. `push_core_binding` takes its var id from the same cache. Ids and roots still live only as long as the lineage epoch, as before.

### 6.26 Slice 4u: longs and doubles compare without the numeric tower

`compare`, `sort-by`, sorted collections and vector comparison order numbers through `jvm_compare`. Two longs took the exact path, which converted both to `Ratio<BigInt>` (heap digits) and cross-multiplied them on every compare. Over the 348 tiny-input non-win cases, the instrumented runtime counted 583K such compares and 1.17M rational conversions.

`jvm_compare` now orders long and double pairs first, exactly as `compare_numbers` does: longs exactly, and a double on either side as `f64` with `<`/`>` (`NaN` ties). Every other pair keeps the tower. A unit test checks the fast path against `compare_numbers` over longs, doubles, signed zeros, infinities, `NaN` and the long extremes.

Measured on macOS, instructions per element: `(compare i 50)` 761 → 418; `sort-by` on a key 1584 → 1183; sorted-set inserts 12297 → 10482.

### 6.27 Slice 4v: scalar prints in one write

`print`, `println`, `pr` and `prn` validated the print options, pushed a `*print-readably*` binding frame (a fresh `Var`, a frame vector and a count-map entry, all undone after), and then dispatched every argument through `pr-on`, the static `print-method` and `scalar_default`. That path re-read the options and the writer, and wrote each argument, separator and newline separately.

When every argument is nil, a boolean, long, double, string, char or keyword, the booted print methods are in place (`print_defaults_only`) and `*print-dup*` is off, `scalar_default` renders each argument as `pr_to_string` under the ambient options: none of these carries meta. Those calls now build the line once from the validated options (`readably` already carries the `print`/`println` override) and write it in one emission, under the same `*out*` check, emission lock and `*flush-on-newline*` flush. Any other argument, a live `print-method`, `*print-dup*`, or a call with no arguments takes the general road.

The corpus file `print_scalars.clj` pins every scalar shape (unicode, `NaN` and infinities, readable strings and chars), `with-out-str`, the `*print-readably*`, `*print-dup*` and `*print-length*` bindings, a nil `*out*`, empty calls, and a user `print-method`, against `run` and the previous binary.

Measured on macOS (kernel `write` included): `println` of a long 22.4K → 18.2K instructions; a two-argument `prn` 25.2K → 19.2K; `println` into `with-out-str` 13.4K → 9.4K.

The line is one buffer presized for the string arguments, and each argument renders into it through `pr_into`. A first version rendered each argument to a temporary and copied it in, so `println` of a 1.27 MB string cost 1.57M macOS instructions instead of 0.19M; it now costs 0.25M. Readable strings also copy the runs between escapes whole: every `CHAR_ESCAPES` entry is ASCII, so a byte scan finds the same cuts the per-char table lookup did. `prn` of that string went from 49.9M to 41.7M (it escapes a newline every 6 bytes).

### 6.28 Slice 4w: unchecked arithmetic, `deref`, `char` and `compare` inline

A core call with no inline op goes through `core_fallback`: the evaluator, the var's entry, cloned arguments and the native. That is about 240 Linux instructions even for `identity` (probe `identity_call`). A trace of these generic calls over the 348 tiny-input non-win cases counted 210K. Some of the most frequent were trivial: `unchecked-multiply` and `unchecked-add` (38.6K each, a pseudo-random generator), `deref` (14.3K, `@atom`), `char` (2.8K) and `compare` (1.7K).

These are now VM inline ops, which the AOT's slow path reaches through `rt_n_op1`/`rt_n_op2` before any generic call:

- `unchecked-add`, `-subtract`, `-multiply`, `-inc`, `-dec` and `-negate` over longs wrap. Doubles and the tower stay the natives'.
- `deref` of an atom or a volatile reads the cell under its lock, as the native does. Refs, delays, futures, promises, vars and the timeout arity stay the native's.
- `char` of a long code point narrows through `char::from_u32`. Out-of-range values (errors) and other shapes stay the native's.
- `compare` of longs and doubles answers -1, 0 or 1 from `jvm_compare`'s numeric order. Other operands stay the native's.

The corpus file `core_inline_ops.clj` pins the wrapping edges, double operands, every derefable, the `char` range errors and mixed compares against `run` and the previous binary.

### 6.29 Slice 4x: sorts call their functions directly

`sort` with a comparator and `sort-by` made a checked general call (`check_ifn`, then `Evaluator::invoke`) for every key function and comparator call: three per comparison for `sort-by`. On Linux, `(sort-by first ts)` over 50 tuples cost 351K instructions.

- The key function and any comparator go through their prepared entry (`vm::Prepared`: a native, or a closure of that fixed arity). Keywords, maps, vars and other callees keep the checked general call.
- The booted `compare` (the native itself; a rebound var holds something else) orders through `jvm_compare` directly.
- The key function still runs twice per comparison and the comparator once, as on the JVM. The corpus pins the call counts.
- `first` of a vector and `compare_vectors` read items by index instead of building a trie iterator per call.

The corpus file `sort_calls.clj` pins key functions (natives, closures, keywords, `juxt`), boolean and three-way comparators, var comparators, the error cases and the call counts against `run` and the previous binary.

Measured on macOS, instructions per element: `sort >` 5665 → 2464; `sort-by` on a keyword key 1183 → 800; `sort-by first` on tuples 8442 → 4763; `sort-by` with a closure key 1502 → 1078.

### 6.30 Slice 4y: collection printers render scalar children directly

A printed vector, list, set or map dispatched every child through `pr-on`: `*print-dup*` read, method lookup, then `scalar_default`, which re-read all six print settings, cloned the child and wrote it. Each child and each separator was a separate write. On Linux, `prn` of a three-item vector cost 15.2K instructions, 4K of them in core-var reads.

Under the booted print methods with `*print-dup*` off, a scalar child (nil, a boolean, long, double, string, char or keyword) now renders from the parent's options through `pr_into`. That is `scalar_default`'s own answer: no meta to prefix, and levels and lengths never apply to a scalar. The collection's text gathers in one buffer, flushed before any child that prints itself (nested collections, records, seqs), so the output order and every other path are unchanged.

The corpus file `print_collections.clj` pins nested shapes, records, sorted and namespace maps, `*print-length*`, `*print-level*`, `*print-meta*` with meta, `*print-dup*`, `*print-readably*`, `with-out-str`, and a user `print-method` that turns the direct path off, against `run` and the previous binary.

Measured on macOS, instructions per call: `prn` of a 5-item vector into `with-out-str` 31.1K → 17.9K; `prn` of a map 39.5K → 29.1K (kernel `write` included).

### 6.31 Slice 4z: print settings read once per binding state

Even on the scalar path, a `println` still read nine core vars: the six behind `print_opts`, plus `*print-dup*`, `*out*` and `*flush-on-newline*`. On Linux that was about 950 of its 2690 instructions (`deref_core` and the cache slot lookup). Each collection level read the six again.

The evaluator keeps a `PrintSnap`: the validated print options, `*print-dup*`, `*out*` and `*flush-on-newline*`, stamped with the lineage epoch and the binding version it was read under. Every binding push, pop and `set!` bumps the version, and every root write bumps the epoch, so a matching stamp means the vars still hold those values. `print_opts` and the scalar print path take the snapshot. A stale stamp reads afresh and keeps the result. An invalid `*print-length*` or `*print-level*` is never kept, so it errors on every call as before. `pr-on`'s own `*print-dup*` read is unchanged, so a user `print-method` that ignores the print options sees no new error.

Measured on macOS, instructions per call: `println` of a long 18.0K → 16.6K; `prn` of a 3-item vector 25.6K → 23.5K (kernel `write` included). A print inside `with-out-str` rebinds `*out*` each time, so it misses: 9.2K → 9.5K. No snippet uses `with-out-str`.

### 6.32 Slice 5a: generic core calls resolve their native once

`core_fallback`, behind `rt_n_core_call` and the `rt_n_op*` misses, resolved var `k`'s native entry and offered it the arguments as a borrowed stack array. A native with an owned-args entry (`range`, `repeat`, `concat`, `map` …) refused that array. Its clones were dropped, the arguments were cloned again into a buffer, and a second `with_active` resolved the entry a second time before the call.

`CoreEntry::call_refs` now takes the resolved entry once. An owned-args native gets its clones by value directly; a borrowing one gets them in a stack array (one to three args) or a buffer.

The re-resolving fallback is gone. Direct core calls are emitted only for programs that cannot rebind core vars; a rebinding program calls each var's current root (`core_rebind.clj`). So var `k` always holds the native the analyzer saw, and `core_fallback` states that invariant (`expect`). Undetected rebinding already broke the inline ops' answers.

Measured on macOS, instructions per call: `(range 3)` 1680 → 1510, `(repeat 2 :x)` 1750 → 1600, `(concat a b)` 9360 → 9190 (owned natives); `subvec` (borrowing) 1170 → 1140.

### 6.33 Slice 5b: natives call compiled closures over their own argument array

`Prepared::call1`/`call2` (the step calls of `map`, `filter`, `mapv`, `reduce`, `sort-by` …) reached a compiled closure through `call_native`. That built a 144-byte `SmallVec` argument buffer and moved it by value through the owned `dyn Fn` entry into `guarded`'s closure. On Linux those moves were out-of-line `memcpy` calls: 126 instructions per element of `(count (for [x (range n)] x))`.

`Prepared` now calls a compiled closure whose single fixed arity matches through `InPlaceCall`: its code (`CompiledEntry::code`, now a typed `CodePtr`) over an argument array the caller owns. `Prepared` passes one or two args as a stack array (a slice). The code may take its params from the slots, exactly as on the owned path (`TAKE_ARGS`), and the caller drops what is left. The native runtime registers the one `InPlaceCall` (it publishes the evaluator to compiled code), so no closure is allocated per compiled fn. At startup each allocation costs about 190 Linux instructions in glibc's slow path. Other natives, other arities, VM closures and owned calls (`update`, `reduce-kv`, transducers) keep their paths. The corpus file `native_closure_calls.clj` pins both paths, their exceptions and arity errors against `run` and the previous binary.

Measured on macOS: `sort-by` with a closure key 1077 → 877 instructions per element. On macOS the buffer moves are inline copies, so elsewhere the gain shows only on Linux.

### 6.34 Slice 5c: zero-closure native dispatch, bulk fn tables, and lazy Lite helpers

A Linux callgrind startup profile (`probe-initline`, 34 near-miss scripts vs trivial) put per-script startup at ~300 `malloc` calls (~48K glibc instructions) plus ~19K self instructions in `rt_n_main_lite`: `install_native_rows` (58.9 mallocs), `rt_n_declare_fn` (33.9), `Symbol::new` (30.4), `rt_n_make_fn` (27.4), and 19 repeated linear scans of declared vars in `boot_lite_impl`.

- **`NativeTarget` / `OwnedTarget` / `DisplayName` (`clj-eval::ext`):** table-driven natives carry bare function pointers (`NativeTarget::Ptr` / `OwnedTarget::Ptr`) instead of allocating `Arc<dyn Fn>` wrappers; compiled AOT closures carry `(desc, captured)` dispatched through one process-wide `CompiledDispatch` table and slice their display name (`DisplayName::Span`) straight from `Tables::fn_names`. `rt_n_make_fn` drops from 3 allocations to 1 (no per-fn closures or display string), and direct core-call resolution (`lite_entry`) allocates nothing.
- **Bulk `rt_n_declare_fns` (`clj-aot`, `clj-rt`):** mirrors `rt_n_declare_vars`, declaring every compiled fn's display name span and arities slice in one table call and pre-sizing `b.consts`; `rt_n_set_const` moves constants out of the builder slot instead of cloning and dropping them.
- **Single-pass Lite boot + lazy `LITE_INTERNAL_HELPERS` (`clj-eval::aot_boot`, `clj-rt`):** `boot_lite_impl` scans declared vars once (`scan_lite_vars`), reserves tight `clojure.core` capacity when reflection vars are absent, and defers the 13 `LITE_INTERNAL_HELPERS` until `Evaluator::core_read` actually misses on one. `run_main_lite` passes `&[Code]` and moves `BUILDING` vectors without cloning.

Measured on macOS (mimalloc, `libinsn.dylib` shim): median empty-stdin startup across 12 near-miss scripts −82K instructions (larger on Linux glibc); closure creation + call −391 instructions per capturing closure (17.07M → 13.16M for 10K) and −388 per capture-free closure (13.48M → 9.60M).

### 6.35 Slice 5d: hoisted flat int-vector and locked transient pointers in scalar loops

Scalar `loop`/`recur` functions over integer vectors and transient DP tables already lowered their arithmetic unboxed (`clj-aot/src/scalar.rs`), but every `(nth v i)` and `(assoc! t i x)` still called `rt_bc_scalar_nth` / `rt_bc_scalar_assoc`, re-acquiring `TransientLock` and re-dispatching `Value` tags per iteration (e.g. `dp/knapsack/solution` spent 295.7M Pin instructions vs OCaml's 50.0M).

- **Zero-`unsafe` packed-int slice & raw transient lock (`clj-value`, `clj-bc`):** `Vector::as_flat_ints` exposes `Packed::Flat<i64>` directly (`rt_bc_scalar_ivec`); `TransientLock::lock_ints_ptr` acquires the existing spinlock for a valid non-empty packed-`Int` `TransientVector`, retains it via `std::mem::forget(guard)` (zero new `unsafe`), and pairs with `TransientLock::unlock_raw` (`rt_bc_scalar_tvec_lock` / `rt_bc_scalar_tvec_unlock`), alongside `rt_bc_scalar_same_transient` for preheader aliasing checks.
- **Preheader hoisting & single-compare bounds checks (`clj-aot/src/scalar.rs`):** `AstScan` separates growing mutations (`conj!`/`pop!`) from in-bounds updates (`assoc!`) and skips probes a loop does not read. Read-only flat `Int` vectors hoist `(ptr, len)` once; unaliased non-growing `TransientVector`s lock once in the preheader (also caching `count`). Setting inactive `str_len`/`ivec_len` to `0` collapses each fast `nth` and `assoc!` check to one unsigned `idx < len` compare plus a direct `i64` load or store. Every miss, slow call (before any potential throw), deopt, and outer exit unlocks all locked transients and zeroes their cached lengths cleanly.
- **Verification:** `docs/spec/i259-native-compiler/corpus/scalar_hoisted_vecs.clj` pins flat int vectors, nested knapsack DP, aliased transients, append/type-widening fallback, read-only transients, OOB throws, and overflow deopt against `run`; `native_emit_ir_hoists_int_vecs_and_locks_transients_271` pins the IR.

Measured on macOS (`libinsn.dylib` shim): `dp/knapsack/solution` drops 290.4M → 61.2M instructions (0.211x, −229.2M).

### 6.36 Slice 5e: lazy combinator fusion, `count` fusion, and `Double` fast arithmetic in `Prepared`

Eager consumers (`vec`, `into`, `reduce`, `some`, `mapv`, `count`) only fused sole unforced `Thunk::Map` and `Thunk::MapCat` seqs (`lazy.rs::walk_sole_map`). Pipelines using `filter`, `remove`, `take`, `concat`, `repeat`, or `count` still allocated `PendingSeq` cells and `Arc<[Value]>` chunks per step (e.g. `binomial_tree`'s `(vec (concat nvals (repeat k 0.0)))`), and `walk_map_vectors` / `Prepared::call2` fell back to general `eval.call` / `native_add` for 2-vector `map` and `Double` arithmetic (`backprop`, `policy_iteration`, `value_iteration`).

- **`Prepared` `Double` & mixed `Double`/`Int` fast arithmetic (`clj-eval::eval`, `clj-eval::vm`):** `Evaluator::fast_double2` handles binary `+`/`-`/`*`/`/`/`<`/`<=`/`>`/`>=` on `Double` pairs and `+`/`-`/`*`/`/` on mixed `Double`/`Int` pairs; `Prepared::call1` handles `inc`/`dec` on `Double`.
- **Extended `SoleMap` fusion (`clj-eval::lazy`, `clj-eval::coll`):** `sole_map_parts` / `walk_parts` now fuse sole unforced `Thunk::Filter`, `Thunk::Remove`, `Thunk::Take`, and iterative `Thunk::Concat`; `walk_all` walks `SubVector`, finite/infinite `RepeatValue`, and counted `LongRange` directly; `walk_map_one` and `walk_filter` walk `Vector` and `LongRange` inputs in stack-buffered 32-item chunks preserving exact chunked forcing and side-effect order; `walk_map_vectors` specializes 2-vector inputs via `Prepared::of(f, 2)`; `coll::walk_count` enters `walk_sole_map` before stepping seq cells.
- **2-input `mapv` fast path (`clj-eval::seqs`):** `native_mapv_owned` walks `(mapv f vec coll)` directly through `Prepared::of(f, 2)` without allocating an intermediate lazy `map` cell.
- **Verification:** `docs/spec/i259-native-compiler/corpus/lazy_fusion.clj` pins chunking/side-effect order under early termination, nested `concat`, infinite `repeat` under `take`, mixed `Double`/`Int` ops, `count` fusion, 2-input `mapv`, and error cases against `run`.

Measured on macOS (`libinsn.dylib` shim, vs Slice 5d): `binomial_tree` 20.64M → 7.94M (0.385x, −12.70M), `backprop` 8.45M → 7.08M (0.839x, −1.36M), `policy_iteration` 6.94M → 6.32M (0.911x, −619K), `value_iteration` 6.56M → 6.16M (0.939x, −403K).

### 6.37 Slice 5f: `PackedStrings` vector storage, direct `clojure.string/join`, and zero-copy full-string `subs`

On `30f6ab42`, `aot-ocaml` MEM jumped to `1499/7/237` strict (`551/1163/29` at ±15%, AOT median RSS `2,760 KB` vs OCaml `3,032 KB`), leaving `data_structure/disjoint_set/connected_component` (`test_case_7..9`, 23.3–27.9 MB vs OCaml 5.5–6.3 MB, up to `4.88x`) as the sole >2x MEM outlier. Its `render-chunk` collected ~250K short strings into a `TransientVector` (`Generic(Vec<Value>)`, ~16.8 MB of 32-byte `Value` slots plus ~8 MB of individual `Arc<str>` heap blocks) before passing the frozen vector to `clojure.string/join`.

- **`PackedStrings` (`clj-value`):** flat `buf: String` plus `ends: Vec<u32>` (`VectorInner::Strs(Arc<PackedStrings>)` and `TransientItems::Strs(PackedStrings)`), with 1.5x growth past 256 KiB and copy-spike-free `shed_slack`. `TransientVector` stays `Generic` below `PACK_STR_MIN = 256` (keeping small string vectors zero-alloc on `nth` / `rt_bc_2_nth`) and compacts once at length 256 when every item is a string; non-string items, in-place `assoc!`, or shared updates past `FLAT_MAX` convert cleanly to `Generic`. `ValueRef::Str(&str)` preserves exact in-bounds vs out-of-bounds dispatch in `clj-bc` without allocating in bitcode.
- **Direct `clojure.string/join` & `apply str` (`clj-eval`, `clj-value`):** `join_vector_buf` joins `Vector::as_packed_strs()` directly from the flat buffer in one pass without boxing items, and `Vector::push_to_string` copies `raw_buf()` directly.
- **Zero-copy full-string `subs` (`clj-eval::core`):** `(subs s 0 (count s))` clones the `Arc<str>` handle instead of copying multi-megabyte input windows.
- **Verification:** `docs/spec/i259-native-compiler/corpus/packed_strings.clj` plus `clj-value` and `clj-bc` unit tests pin every vector/transient/seq/equality/hash/join/subs operation against `run`.

Measured on macOS (`connected_component`, vs Slice 5e): `test_case_7` 34,624 → 23,152 KB (−11.5 MB, 0.669x), `test_case_8` 37,904 → 27,792 KB (−10.1 MB, 0.733x), `test_case_9` 44,864 → 28,880 KB (−16.0 MB, 0.644x) and INS 748.3M → 716.4M (0.957x).

### 6.38 Slice 5g: per-arg ownership transfer, 2-key `assoc-in`/`update-in` & 3-arg `update` fast paths, atom `swap!` in-place, slot recycling, and `HofLoop` `filter`/`some` fusion

Multi-step state simulations (`shared_memory_mutex`, `dining_philosophers`, `max_density_subgraph`, `q_learning`) lost ownership (`Arc::strong_count > 1`) across function boundaries and `loop`/`recur` back-edges, forcing persistent map/vector copy-on-write on every step, and allocated closures + lazy seqs for per-step `filterv`, `(count (filter ...))`, and `(some ...)` traversals.

- **Per-arg ownership transfer & tail-recur parameter moves (`clj-eval::native_abi`, `clj-aot/src/native.rs`):** `TAKE_ARG_SHIFT = 30`, `single_take_arg_bit(i)`, and `take_arg_mask(i)` encode per-argument ownership in the `n` call word when a compiled caller (`lower_direct_invoke`, `lower_value_invoke`) passes a temporary or last-use moved local; `lower_arity` and `own_consumed_params` move owned symbol parameters directly out of `argv[i]` (`take_or_clone`) and include self-recursive (`owned == true`) parameters in `taken` so last-use moves fire inside tail-recursive `defn`s; `end_fn` moves `result` into `ret` instead of cloning.
- **Destructuring, `let`-on-`recur` drop, and stack-slot recycling (`clj-aot/src/native.rs`, `clj-aot/src/scalar.rs`):** `bind_owned_slot` eliminates redundant slot allocation + `clone_v` on symbol positions in vector/map destructuring; `lower_recur` drops active `let_slots` above `target.let_depth` before jumping to the loop header (releasing extra `Arc` references held by loop-body `let` bindings before the next iteration); `FnCtx` recycles dead temporary `free_slots` and `free_arrays` across expressions (cutting `shared_memory_mutex` `one-step` stack frame from 350+ slots to 8 scalar slots + 3 arrays).
- **Zero-vector 2-key `assoc-in2`/`update-in2`, 3-arg `update`, and in-place atom `swap!` (`clj-eval::vm_ops`, `clj-eval::concurrency`, `clj-rt`):** `lower_direct_moving` unpacks 2-element vector literals in `(assoc-in m [k1 k2] v)` and `(update-in m [k1 k2] f)` into 4-arg `"assoc-in2"` / `"update-in2"` calls without allocating the path vector (`repack_own_args` repacks to 3 args on general fallback); `apply_assoc_in2`, `apply_update_in2`, and `apply_update3` (`inc`/`dec`) mutate uniquely owned outer and inner maps/vectors in place via `Map::take_value` / `Vector::take_nth`; `native_swap` updates unwatched, unvalidated atoms in place (`try_swap_in_place`) for `assoc`, 2-key `assoc-in`, and `+`.
- **`HofLoop` `FilterCollect`, `CountFilter`, `Some`, and owned 1-arg `count` slow path (`clj-aot/src/native.rs`):** `hof_loop` fuses `(filterv F xs)`, `(vec (filter F xs))`, `(into [] (filter F xs))` (`HofKind::FilterCollect`), `(count (filter|filterv F xs))` (`HofKind::CountFilter`, branchless `zext i1` counter increment), and `(some F xs)` (`HofKind::Some`, short-circuit exit on truthy result) inline over vectors and long ranges with zero closure or lazy-seq allocation; 1-arg `count` on a temporary or moved local routes its `rt_bc_1_count` miss through `rt_n_core_own` (`emit_owned_slow1`) so `native_count_owned` fuses lazy pipelines with `Arc::strong_count == 1`.
- **Verification:** `docs/spec/i259-native-compiler/corpus/ownership_in2_hof.clj` plus `clj-eval::vm_ops` and `clj-aot::emit` unit tests pin every fast path and fallback branch against `run`.

Measured on macOS (`libinsn.dylib` shim, vs Slice 5f): `shared_memory_mutex` `test_case_0` 6.19M → 3.17M (−48.8%), `test_case_1` 18.87M → 5.99M (−12.88M, −68.2%), `test_case_2` 18.49M → 6.20M (−12.29M, −66.5%), `test_case_3` 18.39M → 5.93M (−12.46M, −67.7%), `test_case_4` 6.29M → 3.19M (−49.2%); `max_density_subgraph` `test_case_0` 5.98M → 4.80M (−19.8%), `test_case_1` 4.55M → 3.89M (−14.5%), `test_case_2` 12.84M → 9.27M (−3.57M, −27.8%); `dining_philosophers` `test_case_3` 9.62M → 8.65M (−966K, −10.0%).

### 6.39 Slice 5h: zero-alloc `SplitVec` parsing, fused dot product, 2-collection/`every?`/`reduce +` `HofLoop`, and panic-free `PackedStrings` bitcode

Linux `h2h-cpu-mem` on `f6444c2b` (`44592735684`) surfaced a bitcode link error (`core::str::slice_error_fail` pulled into `clj_bc-*.bc` by `&self.buf[start..end]` in `PackedStrings::get_str`), while input-parsing and vector-math benchmarks (`stock_span`, `histogram`, `sliding_window_max`, `kd_tree`, `backprop`, `q_learning`, `approx_inference`) spent tens of millions of instructions materializing per-token `Arc<str>` slices from `clojure.string/split` and allocating closures + lazy seqs for 2-collection `map`/`mapv`, `(reduce + init (map * a b))`, `(reduce + [init] (map F ...))`, and `(every? F xs)`.

- **Panic-free `PackedStrings::get_str` (`clj-value/src/value.rs`):** replaced unchecked slice/index ops with `idx.checked_sub(1)`, `self.ends.get(prev)?`, and `self.buf.get(start..end)`, eliminating `core::str::slice_error_fail` from `clj_bc-*.bc` (pinned by a `clj-aot::emit` regression guard).
- **Zero-materialization `SplitVecValue` pipelines (`clj-value/src/seq.rs`, `clj-eval/src/{seqs,lazy}.rs`):** added `SplitVecValue::filter_non_empty` (O(1) window shift for all-non-empty or single-leading-empty splits; span-only filter otherwise) wired into `native_filterv_owned` for `(filterv seq split-vec)`, and walked `SplitVecValue` directly via `sp.iter()` in `seqs::walk_direct` and `lazy::walk_all_other`.
- **Direct string-to-number `mapv` parsing (`clj-eval::native_abi`, `clj-rt`, `clj-aot/src/native.rs`):** `parse_num_arity` + `emit_parse_num_probe` probe `rt_n_mapv_parse_num` (`try_mapv_parse_num`) for `(mapv #(Long/parseLong|Integer/parseInt|Double/parseDouble %) coll)` over `Vector` (including `PackedStrings`), `SplitVecValue`, and `SubVectorValue`, parsing straight into flat `Int` or `Double` vectors without allocating per-token `Arc<str>`s and falling back to the general loop on malformed tokens or `i32` overflow.
- **Fused dot product & 2-collection / `every?` / `reduce +` `HofLoop` (`clj-eval::native_abi`, `clj-rt`, `clj-aot/src/native.rs`):** `try_lower_dot_product` emits `rt_n_dot` (`try_dot_product`, with exact `dot_ints` and `dot_doubles` including `i64` overflow fallback) for `(reduce + init (map * c1 c2))`; `HofLoop` now supports 2-collection `mapv`/`vec`/`into []`/`dorun`/`reduce +`, short-circuit `HofKind::Every` (`every?`), and `HofKind::Sum` (`(reduce + [init] (map F c1 [c2]))` via `rt_n_add_acc`, preserving 1-element identity semantics when `init` is omitted).
- **Verification:** `docs/spec/i259-native-compiler/corpus/splitvec_dot_hof2.clj`, `clj-value::seq`, `clj-eval::native_abi`, and `clj-aot::emit` unit tests pin every fast path, fallback, and panic-free bitcode invariant against `run`.

Measured on macOS (`libinsn.dylib` shim, vs Slice 5f): `backprop` `test_case_0` 17.72M → 10.45M (−7.27M, −41.0%), `test_case_1` 7.13M → 4.89M (−31.4%), `test_case_2` 11.11M → 7.16M (−35.5%), `test_case_3` 6.04M → 4.43M (−26.8%); `stock_span` `test_case_7` 482.78M → 422.61M (−60.17M, −12.5%); `histogram` `test_case_9` 431.97M → 369.78M (−62.19M, −14.4%); `sliding_window_max` `test_case_7` 1,771.63M → 1,705.87M (−65.76M, −3.7%); `kd_tree` `test_case_6` 352.53M → 336.33M (−16.20M, −4.6%); `q_learning` `test_case_2` 20.20M → 18.84M (−6.7%); `approx_inference` `test_case_1` 11.76M → 10.96M (−6.8%).

### 6.40 Slice 5i: unboxed `Double` scalar loops, `Math/*` intrinsics/shims, hoisted `gvec`, top-level `Global` scalars, 2D `Nth2`, and in-place `Double` `assoc!`

Twenty-two of the top 40 non-tiny `aot-ocaml` INS losers on `30f6ab42` are floating-point loop / finance / ML / geometry recipes (`merton_vasicek`, `garch`, `q_learning`, `binomial_tree`, `em_gmm`, `backprop`, etc.) whose `loop`/`recur` bodies previously exited the scalar-loop fast path on the first `Double`, `/`, `Math/*`, top-level constant `def`, or nested `(nth (nth m i) j)` read, and whose flat `Double` vectors fell back to boxed `rt_bc_scalar_nth` per element.

- **Unboxed `K::Double`, `FpCalc`/`FpStep`, `IExpr::Global`, and generalized 2D `Nth2` (`clj-eval/src/vm_scalar.rs`, `clj-eval/src/lib.rs`):** extended `Sv` with `K::Double` (payload = IEEE 754 `f64::to_bits() as i64`) across `+`/`-`/`*`/`/` (unary & n-ary, deopting on int/int to preserve `Ratio`), `inc`/`dec`/unary `-`, type- and NaN-preserving `max`/`min`, `<`/`<=`/`>`/`>=`/`==`/`=` (`-0.0 == 0.0`, `NaN != NaN`, `1 != 1.0`), `zero?`/`pos?`/`neg?`, `double`/`long`/`int` casts (`NaN -> 0`, pre-truncation range checks matching `core.rs`), `clojure.core/abs`, `java.lang.Math/*` (`abs`, `sqrt`, 15 `MATH_UNARY` and 3 `MATH_BINARY` ops), invariant top-level `Expr::Global(Var)` loads, and arbitrary-index nested `(nth (nth coll i) j)` (`Calc::Nth2`).
- **AOT SSA `Double` lowering with `allow_fp` gating & hoisted `gvec` (`clj-aot/src/scalar.rs`, `clj-bc`, `clj-value`):** lowers `K::Double` loops directly to LLVM `f64` arithmetic, `llvm.sqrt`/`llvm.fabs`/`llvm.abs` intrinsics, and `rt_n_math_*` shims; gates FP dispatch behind `allow_fp` (`scan.has_fp || !scan.globals.is_empty() || non-literal scalar init`, plus `k != 3` guards on `rt_bc_scalar_nth`/`rt_bc_scalar_nth2` when `!allow_fp`) so pure-integer loops emit zero floating-point branches; hoists flat generic `Value` vectors via panic-free `Vector::as_flat_values` (`rt_bc_scalar_gvec`, 32-byte indexed direct load + zero-branch unbox); and adds drop-free, panic-free `TransientVector::assoc_double_in_bounds` to `clj-bc::ops::assoc_bang_fast`.
- **Verification:** `docs/spec/i259-native-compiler/corpus/scalar_double_math.clj` plus `clj-value`, `clj-bc`, `clj-eval::vm_scalar`, and `clj-aot::emit` unit tests pin every arithmetic/cast/NaN/deopt/global/2D-`nth`/gvec/`allow_fp` invariant and zero panic symbols in `clj_bc-*.bc` against `run`; full `sweep.py` passes `510/510` scripts (`2,548/2,548` cases, `0` failures).

Measured on macOS (`libinsn.dylib` shim, vs Slice 5h): `finance/merton_vasicek` `test_case_1` 16.06M → 3.96M (−12.10M, 0.247x — 4.05x faster), `finance/garch` `test_case_2` 12.33M → 7.19M (−5.14M, 0.583x), `search/q_learning` `test_case_1` 10.86M → 10.54M (−315K, 0.971x), with zero regression on pure-integer loops (`simulated_annealing` `test_case_2` 9.85M → 9.83M, 0.999x).

### 6.41 Slice 5j: pure-int `allow_fp` regression recovery, streaming `print`/`println` `join`, in-place `PackedStrings::into_joined`, and 1.5x large-transient growth

Linux Pin CI on `376afc2f` (`44643289207`) confirmed Slice 5i's FP gains (`finance/merton_vasicek` 13.84M → 1.57M, 0.48x of OCaml; `count_strings` 683M → 388M; strict MEM wins 1,492 → 1,510, BOTH wins 1,286 → 1,297), but surfaced two pure-integer regressions where Slice 5i's broad `allow_fp` predicate (`non-literal init` / `globals`) and an extra `k != 3` branch in `lower_sc_nth` emitted FP code or disrupted LLVM `JumpThreading`/`InstCombine` (`dp/knapsack` +38%, `dp/edit_distance` +19%, `hill_climbing` +23%, `min_conflicts` +21%, `boyer_moore` +7.3%). Meanwhile, the top two MEM outliers (`connected_component` 22.4 MB vs 6.3 MB, `suffix_array` 12.3 MB vs 10.0 MB) allocated redundant full-output copies across `clojure.string/join` + `println` and doubled large transient vectors past 256 KiB.

- **Exact pure-int `allow_fp` gating (`clj-aot/src/scalar.rs`, `clj-bc`):** tightened `allow_fp = scan.has_fp` and moved the `!allow_fp` rejection of `Value::Double` (`k == 3`) inside `rt_bc_scalar_nth` / `rt_bc_scalar_nth2` (`allow_fp: u8`), removing the extra branch from `lower_sc_nth`/`lower_sc_nth2` so pure-integer loops emit identical CFG to Slice 5h while preserving `-0.0 == 0.0` deopt safety.
- **Streaming `print`/`println` of `clojure.string/join` & zero-copy single-string print/join (`clj-aot/src/native.rs`, `clj-eval/src/{io,stdlib_natives,native_abi}.rs`, `clj-rt/src/nabi.rs`):** `try_lower_print_join` matches `(println (clojure.string/join sep coll))`, `(print (clojure.string/join sep coll))`, and `(print (str (clojure.string/join sep coll) "\n"))` and streams `PackedStrings`, flat `Ints`, flat plain-scalar `Value`s, and `SplitVecValue` directly to `*out*` in 16 KiB chunks (`rt_n_print_join` / `try_print_join`) without allocating the joined `String` or `Arc<str>`, preserving `*out*`, `EmissionGuard`, `check_writer`, `*print-dup*`, and `*flush-on-newline*` and falling back to the general call on any miss. `print_scalars` writes single plain `Value::Str` arguments directly without a `scalar_line` copy; `single_str_join` returns the sole `Arc<str>` zero-copy.
- **In-place `PackedStrings::into_joined` & 1.5x large-buffer growth (`clj-value/src/{value,transient}.rs`):** `Vector::into_packed_strs` unwraps uniquely held `PackedStrings` so `native_join_owned` shifts bytes in place via zero-`unsafe` `copy_within` whenever `total <= buf.capacity()` (falling back to `join` when reallocation would be required); `TransientVector` (`push_tight`) and `PackedStrings::ends` grow 1.5x past 256 KiB (`64K` `u32` entries), matching `PackedStrings::buf`'s existing threshold.
- **Verification:** `docs/spec/i259-native-compiler/corpus/print_join_stream.clj` plus `clj-value`, `clj-bc`, `clj-eval`, and `clj-aot::emit` unit tests pin every streaming/fallback/in-place/growth/`allow_fp` path and zero panic symbols in `clj_bc-*.bc` against `run`; full `sweep.py` passes `510/510` scripts (`2,548/2,548` cases, `0` failures).

Measured on macOS (vs Slice 5h/5i): `dp/edit_distance` `test_case_5` 952.35M → 919.94M (1.000x of 5h), `boyer_moore` `test_case_11` 165.85M → 158.73M (1.006x of 5h), `hill_climbing` `test_case_2` 19.90M → 19.76M (1.000x of 5h), `min_conflicts` `test_case_2` 20.91M → 20.64M (1.001x of 5h), `count_strings` `test_case_6` 321.59M → 318.58M (0.701x of 5h), while retaining `merton_vasicek` (0.634x) and `garch` (0.826x); with immediate purge (`MIMALLOC_PURGE_DELAY=0`, mirroring Linux glibc `munmap`), `connected_component` `test_case_9` drops 20,352 → 17,504 KB (−2,848 KB) and `suffix_array` `test_case_7` drops 19,024 → 18,400 KB (−624 KB; −2,160 KB under default mimalloc).

### 6.42 Slice 5k: packed `f64` vectors & transients, zero-alloc vector `compare`/`=`/`hash`, `SubVector`/`SplitVec` HOF loops, and `drop`/`take-while`/`drop-while`/`collect_all` fusion

Profiling the top `aot-ocaml` INS losers on `376afc2f` (`beam_search` 232M, `q_learning` 15.4M, `steiner_triple` 6.5M, `binomial_tree` 6.0M, `fortune_voronoi` 5.4M, `delaunay` 5.2M, `garch` 5.2M) showed four compounding bottlenecks: sorting and hashing vectors of tuples/ints cloned and dropped every element through `Vector::get` and `VectorIter::next`; homogeneous `Double` vectors and transients stayed in 32-byte `Generic(Vec<Value>)` slots instead of flat `f64` storage; compiled `HofLoop`s missed `SubVectorValue` and `SplitVecValue`; and `drop`/`take-while`/`drop-while` plus `collect_all` (`set`/`sort`/`reverse`/`frequencies`) missed `SoleMap` fusion.

- **Packed `f64` vectors & locked `dtvec` scalar lowering (`clj-value`, `clj-bc`, `clj-eval`, `clj-aot/src/scalar.rs`):** added `VectorInner::Doubles(Packed<f64>)`, `TransientItems::Doubles(Vec<f64>)`, `ValueRef::Double(f64)`, `TypedItems::Doubles(Vec<f64>)`, `Vector::as_flat_doubles`, `Transient::lock_doubles_ptr`, and panic-free bitcode shims `rt_bc_scalar_dvec`, `rt_bc_scalar_dtvec_lock`, `rt_bc_scalar_assoc_d`, and `rt_bc_scalar_conj_d`; upgraded `VecBuilder`, `RepeatValue` collection, `try_mapv_parse_num`, `dot_doubles`, and streaming `print-join` to stay flat `f64` end-to-end; wired optional `dvec`/`dtvec` hoisting & direct 8-byte stores in `scalar.rs` with zero IR perturbation for `!allow_fp` pure-int loops and targeted `allow_fp` detection for `Double`-literal / `(double ...)` / read-only `nth`-initialized loops (`best-action`/`max-q`).
- **Zero-alloc vector `compare`, `=`, and `hash` (`clj-value/src/{value,compare}.rs`, `clj-eval/src/core.rs`):** `jvm_compare` dispatches `(Value::Vector, Value::Vector)` immediately after `compare_long_double` and both `core::compare_vectors` and `clj_value::compare::compare_vectors` compare flat `Ints`, flat `Doubles`, and flat `Generic` slices directly without cloning or dropping elements; `vector_eq`, `PartialEq for Vector`, and `Hash for Vector` walk flat/packed buffers directly (`packed_eq`/`packed_for_each`) with zero `VectorIter` boxing or `Value::clone`/`drop` while preserving exact Clojure hash/equality parity.
- **`SubVector`/`SplitVec` HOF iteration & extended `SoleMap`/`collect_all` fusion (`clj-eval/src/{native_abi,lazy}.rs`):** `iter_plan`/`vec_item` index `SubVectorValue` and `SplitVecValue` directly for all compiled `HofLoop`s; `SoleMap` fuses `Drop`, `TakeWhile`, and `DropWhile` reusing `filter_prep`/`test_filter_item`; and `collect_all` enters `walk_sole_map` so `set`, `sort`, `sort-by`, `reverse`, `frequencies`, `group-by`, and `apply` fuse unforced lazy pipelines.
- **Verification:** `docs/spec/i259-native-compiler/corpus/packed_doubles_and_fusion.clj` plus `clj-value`, `clj-bc`, `clj-eval`, and `clj-aot::emit` unit tests pin every new fast path, fallback, and panic-free bitcode invariant against `run`; full `sweep.py` passes `510/510` scripts (`2,548/2,548` cases, `0` failures).

Measured on macOS (work instructions above empty baseline, vs Slice 5j): `search/beam_search` `test_case_0` 206.33M → 88.44M (0.429x, −117.9M), `string/count_strings` `test_case_10` 372.96M → 349.40M (0.937x, −23.6M), `fortune_voronoi` `test_case_3` 9.59M → 8.51M (0.888x), `steiner_triple` `test_case_2` 11.29M → 10.03M (0.889x), `finance/garch` `test_case_2` 8.92M → 8.07M (0.904x, −32 KB), `delaunay` `test_case_3` 9.06M → 8.32M (0.918x), `bentley_ottmann` `test_case_3` 8.59M → 7.99M (0.930x), `finance/binomial_tree` `test_case_0` 10.08M → 9.50M (0.942x, −16 KB), `sort/bucket_sort` `test_case_2` 21.98M → 21.18M (0.963x), with zero regression on pure-int loops (`dp/edit_distance` `test_case_5` 920.09M → 919.63M, 0.9995x; `boyer_moore` `test_case_14` 1.000x).

### 6.43 Slice 5l: fused loops over `(range …)` classify by their bounds, without the range value

A 2-second macOS `sample` profile of the `approx_inference` inner loops (the worst `aot-ocaml` INS loser at Slice 5k, 9.4x) put ~15% of samples in `(range k)` machinery that a fused loop never needed: every call of a `defn` whose body reduces over a small range (`prow`, ten times per sample) materialized a fresh `RangeValue` (`native_range` → `long_range`, one `Arc` allocation), cloned it, classified it (`rt_n_iter_plan`), and dropped it — per-call setup, not the walk, dominated.

- **Bounds-only plans (`clj-eval::native_abi`, `clj-rt::nabi`):** `range_plan(argc, [a0..a2])` mirrors `lazy::long_range`'s classification from the long bounds alone — `(2, count, start, step)` for a finite long range, `(2, 0, start, step)` for the empty shapes (which walk zero items exactly as the general call walks the empty list), and `None` for the zero-step repeat and the `i32`-count-overflow shapes that stay general. `rt_n_range_plan` exports it to compiled code; a `clj-eval` unit test checks every shape against the plan of the value `long_range` builds, and `clj-rt` pins the shim.
- **Fused-loop emission (`clj-aot::native.rs`):** a fused HofLoop whose collection (first or second) is a 1-3-argument direct `clojure.core/range` call lowers its argument exprs into slots, checks the tags are all `Int`, and calls `rt_n_range_plan` for the plan — no range value is built, cloned, classified, or dropped. Any other shape (a non-`Int` bound, a zero step, an overflow) takes the existing general path, which materializes the range once over the already-lowered slots (`core_call_owned`), exactly as the old lowering did; the parse-num `mapv` probe keeps the materialized path (it reads the collection value). Argument evaluation order and the general path's closure-then-call order are unchanged.
- **Verification:** `docs/spec/i259-native-compiler/corpus/range_plans.clj` pins every fused shape (`reduce`/`mapv`/`vec`/`into`/`some`/`every?`/`filterv`/`count`/`dorun`, 1-3 range args, both collections of a 2-input `mapv`) over finite, empty, negative-step, double-bound, zero-step (infinite under `take`), and overflow ranges, plus the error cases, against `run`; `native_emit_ir_plans_range_loops_directly_271` pins the IR (`rt_n_range_plan` present, `rt_n_iter_plan` gone, two plans for a two-range `mapv`); a local run-vs-build parity sweep passed `394/394` snippet scripts on three generic inputs.

Measured on macOS (arm64, min of 7 wall runs): `approx_inference` on a CI-shaped input (two cases, `s` ≈ 2-3k) 46.6ms → 36.2ms (0.778x); on a heavy input (three cases, `s` = 150k-250k) 6.37s → 4.74s (0.745x); empty-stdin startup unchanged (2.1ms → 2.0ms). Output byte-identical to `run` and to the previous binary on both inputs.

### 6.44 Slice 5m: windows into a large vector seek instead of skipping

The `sudoku` cluster (the largest `aot-ocaml` INS loser family, 20 cases at 6.3-6.6x) printed each solved board with `(str/join " " (subvec board …))`, where `board` is `(vec (subvec all base (+ base 81)))` — a window over the whole parsed input vector. Every join materialized its items through `vec.iter().skip(start)`: `Skip` pulls the default `Iterator::nth`, and the trie-backed `VectorIter`/`PackedIter` inherit the default too, so each window walked **every item from the vector's start** — O(base) per join, quadratic in the input. A bisected run (1500 puzzles) spent 77% of its wall time in that walk (2.2s of 2.9s; the solver itself was 0.55s).

- **Seeking iteration (`clj-value::trie`, `clj_value::value`):** `PVec::iter_from` already positioned its cursor; `Packed::iter_from` now reaches it (a flat run advances its slice iterator by index — `slice::Iter::nth` is O(1) — and a trie positions its cursor, one descent on the first `next`), `Vector::iter_from` exposes it for every packing, and `PackedStrings` seeks its `lo`.
- **The walks (`clj-value::seq`, `clj-eval::lazy`, `clj-eval::seqs`):** `SubVectorValue::iter` seeks to its `start` (`.take(len)` unchanged), `with_window` and `idrop_shift_vector` (`drop` on a vector) take `iter_from`, and the two subvector walk sites (`seqs::walk_direct`'s each-helper, `lazy::walk_all_other`'s visit) seek as well. Item values, laziness, `realized?`, seq classes, and side-effect order are unchanged — only the skipped items are no longer stepped.
- **Verification:** `clj-value` unit tests seek flat and trie `Packed`/`Vector` storage of every packing (double-ended from a seek, clamped and past-end starts, empty storage); corpus `docs/spec/i259-native-compiler/corpus/subvector_seeks.clj` pins join/print/seq/reduce/`into`/`drop`/`doseq`/lazy-realization order over deep windows, windows over windows, and the error shapes, against `run`; the local run-vs-build parity sweep passed `394/394` snippet scripts.

Measured on macOS (arm64, min of 5 wall runs, byte-identical output): `sudoku` 1500-puzzle input 2.20s → 0.51s (**0.231x**), 500-puzzle input 269ms → 80ms (**0.298x**); parse-only and solve-only bisect variants put the eliminated cost in the join walk.

### 6.45 Slice 5n: vector literals classify in one pass and skip the meta re-stamp

A 2-second `sample` of `q_learning` (the worst remaining `aot-ocaml` INS loser after Slice 5m, 9.5x) put ~7% of samples in per-step small vector literals (`[rng u]` pair returns): `vec_lit_owned` first collected an `Option<Vec<i64>>`, so every mixed-shape literal allocated (and dropped) a partial longs buffer before the generic path allocated again, and every literal ended in `with_meta(None)` — an `Arc::make_mut` plus a second identity mint over a freshly constructed vector (the first is stamped at construction).

- **One classification pass (`clj-eval::native_abi::vec_lit_owned`):** all-long and all-double literals pack directly into `Vector::from_ints` / `Vector::from_doubles` without touching the caller's POD slots; everything else moves the items out as before — no partial buffer for the mixed shapes. `vec_lit` and `vec_lit_owned` stamp metadata only when there is any; identity semantics are unchanged (each literal is a fresh object either way, and `with-meta` proper still re-stamps).
- **Verification:** `clj-eval::native_abi` unit tests cover `None` and `Some(meta)` for both `vec_lit` and `vec_lit_owned`, plus all-long, all-double, mixed, char, and empty shapes at 100% line coverage.

Measured on macOS (arm64, min of 3 wall runs, byte-identical output): `q_learning` heavy workload (80k episodes) 7.04s → 6.48s (**0.92x**); the local run-vs-build parity sweep passed `394/394` snippet scripts.

### 6.46 Slice 5o: primed `FastArith`, pre-effect primed scalar loops, inline `double`/`int`/`long`/`char`/`bit-not`/`unchecked-*`/`Character/digit`, and scalar `Long/remainderUnsigned`

Profiling the remaining top `aot-ocaml` INS losers (`unbounded_knapsack` 2.93x, `coin_change` 2.46x, `knuth_div` 8.03x, `q_learning` 9.47x, `approx_inference` 5.88x, `suffix_array`) showed four high-frequency gaps in inline codegen and scalar-loop eligibility: promoting ops (`+'`, `-'`, `*'`, `inc'`, `dec'`) were absent from `FAST_ARITH_TABLE` and disqualified every stateful scalar loop even when no mutation preceded the primed op; `double`, `int`, `long`, `char`, `bit-not`, and `unchecked-*` fell back to runtime calls outside scalar loops; `unchecked-*` and `Long/remainderUnsigned` were absent from `vm_scalar`; and `java.lang.Character/digit` went through `rt_n_static_call`.

- **Primed `FastArith` & pre-effect primed scalar loops (`clj-eval/src/{core,vm_scalar}.rs`, `clj-aot/src/scalar.rs`):** registered `native_{add,sub,mul,inc,dec}_prime` in `FAST_ARITH_TABLE`; added flow-sensitive `seen_effect` / `primed_after_effect` tracking in `vm_scalar::Builder` (including branch-scoped `branch_effects`, argument-evaluation ordering before the primed check, and nested-loop `inner_eff && self.primed` rejection so outer deopt can never replay an earlier mutation); relaxed `can_lower_scalar` from `ast.primed` to `ast.primed_after_effect`, unlocking hoisted `itvec` SSA loops for DP loops like `(assoc! row j (+' (nth row j) (nth row (- j c))))` while still deopting cleanly on `i64` overflow.
- **Unchecked ops & `Long/remainderUnsigned` in scalar loops (`clj-eval/src/{vm_ops,vm_scalar}.rs`, `clj-aot/src/scalar.rs`):** added `IntOp1::{UInc, UDec, UNegate}` and `IntOp2::{UAdd, USub, UMul, RemUnsigned}` across the VM scalar tier and AOT SSA lowering (both integer wrapping and `allow_fp` floating-point paths).
- **Inline scalar casts, `bit-not`, `unchecked-*`, and `Character/digit` (`clj-aot/src/native.rs`):** lowered `(double x)` (Int/Double), `(long x)` and `(int x)` (Int/Double/Char with exact pre-truncation range checks and `NaN` fallback), `(char x)` (Char and BMP Int `< 0xD800`), `bit-not`, `unchecked-{inc,dec,negate,add,subtract,multiply}`, and `java.lang.Character/digit` (branchless ASCII `0-9`/`A-Z`/`a-z` check matching `char::to_digit`) directly to LLVM IR with general fallback.
- **Verification:** `docs/spec/i259-native-compiler/corpus/inline_casts_unchecked_primed.clj`, `clj-eval::{core,vm_ops,vm_scalar}`, and `clj-aot::emit` unit tests pin every fast path, overflow/deopt boundary, and error case against `run`.

Measured on macOS (`libinsn.dylib`, vs Slice 5n, byte-identical output on all 18 scripts): `dp/unbounded_knapsack` 246.20M → 32.15M (**0.131x**, −214.1M), `dp/coin_change` 5.89M → 3.06M (**0.519x**, −2.83M, −16 KB), `math/knuth_div` 164.21M → 97.54M (**0.594x**, −66.68M, −32 KB), `suffix_array` 762.98M → 710.98M (**0.932x**, −52.00M), `search/q_learning` 175.63M → 169.77M (**0.967x**, −5.86M, −16 KB), `search/approx_inference` 212.33M → 208.04M (**0.980x**, −4.29M, −16 KB).

### 6.47 Slice 5p: zero-alloc vector-literal destructuring elision and small leaf stable-`defn` inlining

Across the 117 remaining `aot-ocaml` INS-losing scripts, 114 destructured short tuple vectors (`[r2 u] (lcg-double r)`, `[r2 a] (egreedy …)`, `[r3 nst rw] (trans-step …)`, `[a r] (loop …)`, `[c2 cnt] (loop …)`) and called small non-recursive helper `defn`s (`lcg-next`, `orient`, `best-action`, `max-q`, `sigmoid`) inside tight loops — paying per-iteration `rt_n_enter`/`rt_n_leave`, heap vector allocation, `nth` indexing, and drop overhead solely to pass 2–4 scalars across a `let` or leaf call boundary.

- **Small leaf stable-`defn` inlining (`clj-aot/src/native.rs`):** `inline_callee` / `inline_callee_into` inline fixed-arity, non-`recur`, strictly-DAG (`sr.unit < callee_unit`) stable `defn`s bounded by `INLINE_MAX_DEPTH = 3`, `INLINE_MAX_PER_FN = 24`, and `INLINE_GEN_MAX_NODES = 32` (`INLINE_DESTR_MAX_NODES = 80` at destructuring sites). Caller args evaluate left-to-right before swapping `env`/`recur`/`moves`/`captured`; non-moved caller locals bind directly in place as borrowed `LocalRef::Slot`s (zero `Arc` refcount churn) while computed/moved args participate in callee-local `collect_body_moves`.
- **Zero-alloc vector-literal destructuring elision (`clj-aot/src/native.rs`):** in `lower_let_bindings`, plain `Pattern::Vec` bindings (`rest.is_none() && as_sym.is_none()`) whose init has a destructurable vector-literal exit (`has_destr_vec`) lower via `lower_vec_destructure` directly into the `n` target slots through `VecLit`, `If` (including partial-literal `If` via `lower_and_unpack`), `Let`, `Do`, non-scalar `Loop` (keeping `try_lower_scalar_loop` priority intact), and inlineable `Invoke` (`InlineDest::Destructure`), eliminating both the intermediate vector allocation and the `nth` extractions.
- **Verification:** `docs/spec/i259-native-compiler/corpus/inline_destructure.clj` and `clj-aot::emit::native_emit_ir_inlines_helpers_and_elides_destructuring_271` pin multi-level inlining, destructuring elision, depth/count caps, scalar-loop priority, non-inlineable recursive/forward/closure/variadic/big callees, and exception/side-effect evaluation order against `run`.

Measured on macOS (`libinsn.dylib`, vs Slice 5o, byte-identical output on all 18 scripts): `search/q_learning` 169.77M → 62.95M (**0.371x**, −106.8M; **0.335x** vs 5k), `search/approx_inference` 208.04M → 159.41M (**0.766x**, −48.6M; **0.551x** vs 5k), `geometry/fortune_voronoi` 55.91M → 47.33M (**0.847x**, −8.58M, −16 KB), `suffix_array` 710.98M → 620.36M (**0.873x**, −90.6M; **0.813x** vs 5k), with minimal `.text` delta (+0 to +16 KB on 15/18 scripts).

### 6.48 Slice 5q: `SmallSet`, owned `fnil`, fast Lite `clojure.string/replace`, and in-place CRLF `slurp`

Profiling the set-heavy geometry/compiler clusters (`fortune_voronoi`, `delaunay`, `bentley_ottmann`, `steiner_triple`, `hindley_milner`, `ssa_dominance`, `dfa_minimization`) and `knuth_div` exposed three runtime bottlenecks: every `Set` (even 0–8 elements) allocated both an `rpds::HashTrieMapSync` and an `rpds::VectorSync` sidecar and triggered full `compact()` rebuilds on small `disj`s; `fnil` used borrow-only `native_fn` with a fresh `Vec<Value>`, breaking `native_update_owned`'s `take_lookup` unique ownership; and `clojure.string/replace` was spliced as Clojure code running `re-matcher` + full regex engine on every `(str/replace s #"^0+" "")` call in `knuth_div`.

- **Inline `SmallSet` (`clj-value/src/value.rs`):** split `SetData` storage into `SetStorage::Small(Vec<Value>)` (up to `SMALL_SET_MAX = 8`, both unsorted and sorted) and `SetStorage::Large(Box<LargeSet>)`, eliminating `rpds` trie/sidecar allocations, double-cloning, and `compact()` churn on small sets while preserving exact insertion/comparator order, equality, hashing, and identity semantics.
- **Owned `fnil` (`clj-eval/src/{ext,seqs}.rs`):** added `OwnedTarget::Dyn(Arc<OwnedNativeImpl>)` and `native_fn_owned` so `fnil` fills `nil` defaults in place inside `SmallVec<[Value; 4]>` and dispatches via `call_checked_owned`, keeping uniquely-held sets and vectors from `update`'s `take_lookup` unique all the way through `conj`.
- **Fast Lite `clojure.string/replace` & in-place CRLF `slurp` (`clj-eval/src/{re,stdlib_natives,aot_boot,reader}.rs`):** added `re::replace_all_fast` (zero-copy/direct `trim_start_matches`/`trim_end_matches` for simple anchored `^c+`/`c+$` patterns without `$`/`\` escapes, cached `Regex::replace_all` otherwise) and registered `native_replace` in `STRING_NATIVES` (with fast `Str`/`Regex`/`Str`, `Str`/`Char`/`Char`, `Str`/`Str`/`Str`, and callable `lite_replace_by` paths plus exact `upstream`/`lite_replace` fallback); normalized `\r\n`/`\r` in place inside `slurp_buf_reader` without allocating a second full-file string.
- **Verification:** `docs/spec/i259-native-compiler/corpus/small_set_fnil_replace.clj`, `clj-value::value::small_and_large_set_storage_paths_271`, `clj-eval::stdlib_natives::native_and_lite_replace_cover_fast_paths_and_fallbacks_271`, and `aot_boot`/`aot_front`/`emit` tests pin every path against `run`.

Measured on macOS (`libinsn.dylib`, vs Slice 5p, byte-identical output on all 18 scripts): `compiler/hindley_milner` 5.86M → 5.26M (**0.899x**, −10.1%), `math/knuth_div` 96.76M → 87.48M (**0.904x**, −9.28M, −32 KB), `geometry/fortune_voronoi` 47.33M → 44.11M (**0.932x**, −3.22M), `combinatorics/steiner_triple` 7.49M → 7.42M (**0.992x**).

### 6.49 Slice 5r: selective `clojure.set` splicing, targeted `rt_bc_*` inline hints, and lean `end_fn` / cold `fail` cleanup

Inspecting the Slice 5l Linux RSS gap showed 219 of 229 MEM losses sitting within `+0..+584 KB` of OCaml (`hindley_milner` +484 KB, `bentley_ottmann` +524 KB, `point_in_polygon` +584 KB, `delaunay` +368 KB), driven by `.text` page bloat from three codegen sources: splicing all 19 definitions of `clojure.set` regardless of reachability (~50 KB dead `join`/`project`/`rename`/`index`/`select` code), marking every `rt_bc_*` bitcode helper `inlinehint` (forcing multi-hundred-instruction `nth`/`get`/`eq`/`first`/`peek` bodies into every call site), and emitting inline POD-branch drop loops in `end_fn` for both already-nilled slots and cold exception-exit blocks.

- **Selective `clojure.set` splicing (`clj-eval/src/aot_front.rs`):** changed `SplicedLib::select` from `Option<fn(&str) -> bool>` to `fn(&str) -> bool` with `clojure.set` using `no_lite_native`, reusing `needed_defs` + `select_units` so only reachable `clojure.set` definitions and their transitive dependencies are compiled.
- **Targeted `rt_bc_*` `inlinehint` (`clj-aot/src/lib.rs`):** restricted `mark_internal_inline` (`should_inline_hint`) to `rt_bc_scalar_*`, `rt_bc_is_pod`, `rt_bc_deref`, `rt_bc_clone`, `rt_bc_1_count`, and the transient mutators (`rt_bc_1_pop_bang[_fx]`, `rt_bc_2_conj_bang[_fx]`, `rt_bc_3_assoc_bang[_fx]`), letting LLVM's O2 cost model inline tiny scalar ops automatically while outlining large `nth`/`get`/`eq`/`first`/`peek`/`contains` helpers.
- **Lean `end_fn` and cold `fail` cleanup (`clj-aot/src/native.rs`):** skipped normal-exit `drop_v` for slots already moved or released (`f.free_slots`, `f.free_arrays`, `result`), and emitted direct `cold`-attributed `@rt_n_drop` calls in `f.fail` so exception cleanup stays out of hot `.text` pages.
- **Verification:** `clj-aot::emit::native_emit_splices_clojure_set_under_lite_only` pins tree-shaking of unused `clojure.set/join`; `clj-eval::aot_front`, `clj-aot::emit`, `fmt`, `clippy`, and `complexity` all pass with 0 violations.

Measured on macOS (`libinsn.dylib`, vs Slice 5q, byte-identical output on all 18 scripts): `hindley_milner.o` `__TEXT` 301 KB → 212 KB (−89 KB, −29.6%; binary TEXT −96 KB, PEAK_FP −32 KB, INS −1.8%); `suffix_array` 620.57M → 593.69M (**0.957x**, −26.88M), `count_strings` 56.86M → 53.28M (**0.937x**, −3.58M), `approx_inference` 159.39M → 155.78M (**0.977x**, −3.60M), `q_learning` 62.84M → 61.10M (**0.972x**, −1.74M), `sudoku` 357.59M → 355.42M (−2.17M), `connected_component` 918.76M → 916.84M (−1.93M, PEAK_FP −16 KB), `fortune_voronoi` −48 KB TEXT, `backprop` −32 KB TEXT, 17/18 benchmarks strictly faster than 5q.

### 6.50 Slice 5s: JDK-ASCII regex classes, owned `disj`/`dissoc`, and a suffix-compare default-import check

Three general fixes found while auditing per-operation and startup costs (macOS `libinsn` microbenchmarks; Linux callgrind of 55 snippet programs on empty stdin, fork probe `probe/init2-271`).

- **JDK-ASCII predefined regex classes (`clj-eval/src/re.rs`):** the JDK matches `\d \D \w \W \s \S \b \B` over ASCII only (without `UNICODE_CHARACTER_CLASS`); the `regex` crate defaults them to Unicode. `ascii_classes` rewrites them before every compile (`[[:digit:]]`, `[:digit:]` inside a bracket class, `(?-u:\b)`), so `(re-seq #"\d+" "12 ٣٤")` is `("12")` like the JVM, and a compile no longer builds the Unicode class tables (`\w` ~2.8M instructions, `\d` ~0.9M). The hand-scanned `\s+` split no longer needs all-ASCII text.
- **Owned `disj`/`dissoc` (`clj-eval/src/{vm_ops,native_abi,coll,ns}.rs`):** both were on the AOT's owned-move list but only had borrowing implementations, so a moved loop accumulator was cloned and then path-copied on every call. `DisjOwn`/`DissocOwn` join `OWN_OPS` (unsorted collection, scalar key; anything else restores the operand and falls back), and `native_disj_owned`/`native_dissoc_owned` join `OWNED_CORE_ENTRIES` for the general path. A 64-member set `disj`+`conj` cycle: 8.2K -> 3.5K instructions; 8-member: 2.1K -> 0.9K; a 64-entry map `dissoc`+`assoc`: 3.9K -> 2.4K.
- **Default-import check (`clj-eval/src/classemu.rs`):** every `def` of an uppercase name (`EPS`, `INF`) scanned ~170 default imports with a reverse split (~26K instructions per such script on Linux); a suffix compare replaces it.
- **Verification:** `re::tests::ascii_classes_follow_the_jdk_defaults`; corpus `owned_disj_dissoc.clj` (moved and shared operands, nil, sorted, non-scalar keys, records, wrong types, every arity) replayed against `run`.

Measured on macOS (interleaved A/B over all 1754 snippet cases): neutral on the gate (median +1.1K per case, within noise; projected class moves 4 T->W, 4 L->T, 3 T->L, 1 W->T). These are correctness and complexity fixes for general code more than gate wins: the startup profile puts the typical Linux startup excess at 30-190K instructions spread over keyword/metadata constants and top-level initializers.

### 6.51 Slice 5t: the AOT hot-text order packs the C runtime's archive members

The AOT order (`rust/hot-text-aot.ld`) placed only Rust functions: glibc's static archives are built without function sections, so the profiled C functions (`malloc`, `memcpy` variants, `strlen`, start-up code) stayed scattered over the cold `.text`, each pulling in its own 64 KiB fault-around window. `scripts/bench/hot-text/hot_text.py --archive LIB.a` now maps the executed functions to the archive members defining them (`nm -A --defined-only`) and lists those members first in `.text.hot` (`*libc.a:malloc.o(.text .text.*)`), most workloads first; `regen-aot.sh` passes the toolchain's `libc.a`, `libm.a` and `libgcc.a`. The training set is unchanged (the generic workloads and the compiler corpus, never the snippets).

- **Measured (fork probe, Linux, 98 snippet programs, empty stdin, peak RSS best of 3):** median -122 KB (p10 -256 KB, p90 +48 KB) against the order without members, both regenerated from the same profiles; 133 members.
- **Verification:** fork run of `regen-aot.sh` + link: 72/72 corpus programs build and match `run`; the member mapping is exercised against GNU `nm -A` output.

### 6.52 Slice 5u: in-place owned vector `assoc`/`conj` in compiled code, and a numeric `(reduce + …)` fold

Measuring operation costs outside the scalar loops (macOS `libinsn`, loops of 100K) put an owned vector `assoc` at ~280 instructions and a `conj` at ~330 for a few instructions of real work: the moved accumulator went through `rt_n_assoc_own3`/`rt_n_core_own`, the `OWN_OPS` dispatch, `Vector::set_nth_owned`/`push_back` (two `Arc::make_mut`s, retyping checks, a fresh identity), and argument drops.

- **Lean owned vector updates (`clj-value/src/{value,trie}.rs`, `clj-bc/src/{ops,lib}.rs`, `clj-aot/src/{native,lib}.rs`):** `Vector::assoc_unique`/`conj_unique` update in place only when the handle and its flat storage are both unique (`Arc::get_mut`), the index is in bounds (or, for `conj`, the vector is non-empty, below `FLAT_MAX`, and within capacity), and the value fits the packing without retyping (a long into longs, a double into doubles, any value over a scalar item or appended to generic storage). `rt_bc_assoc_own3`/`rt_bc_conj_own2` expose them as bitcode, and `lower_own_call` tries them first: a hit moves the updated vector to the result, and a miss leaves every argument untouched for the general owned call. Bitcode cannot reach runtime statics or out-of-line drop glue, so the lean path never drops (a replaced generic item must be a scalar), never grows a buffer (the general path grows it: appends stay amortized O(1)), and keeps the vector's identity: no other handle can observe the old version, as for a uniquely owned map updated in place.
- **`(reduce + [init] v)` over a vector (`clj-eval/src/seqs.rs`):** with the genuine `+`/`+'` (`FastArith::Add`) and a vector of longs and doubles, `reduce` folds left in the general order (doubles round alike) without a boxed step call per item; any other operand, an empty vector without an init, or a long overflow hands the unchanged walk the whole reduction (`+` is pure).
- **Verification:** `clj-value::value::unique_vector_updates_only_touch_unshared_flat_storage_271` (shared handles, shared storage after `with-meta`, retyping, trie storage, full capacity, `FLAT_MAX`), `clj-bc` unit and shim tests, corpus `owned_vector_updates.clj` (aliases kept in history, bound twice, held by a var; retyping; out-of-range; metadata) and `reduce_add_vector.clj` (overflow, `+'`, BigInt, ratio, NaN, signed zero) replayed against `run`, both byte-identical to the previous runtime's output.

Measured (macOS, loops of 100K): vector `assoc` ~330 -> ~130 instructions per iteration, `conj` of a long 364 -> 167, of a keyword 465 -> 298; `(reduce + 0.0 v)` on two items 1419 -> 1034. Program `__text` +0..+2 KB. Interleaved A/B over all 1754 cases: median -1.7K instructions (-0.1%), `sudoku` -8%, `dining_philosophers` -14%; projected class moves 5 loss->tie, 1 tie->win, 1 tie->loss, 1 win->tie.

### 6.53 Slice 5v: borrowing read chains in compiled code

A read feeding a read op cloned its result only for the outer read to drop it: `(count (nth adj u))`, `(nth (get g :adj) r)`, `(nth (nth grid r) c)` each paid a clone and a drop of the intermediate collection (two atomic refcount updates plus the boxed write and drop checks) on every evaluation.

- **Chains (`clj-aot/src/native.rs`, `clj-bc/src/{ops,lib}.rs`):** for a read op (`nth`/`get` of 2-3 args, `count`/`first`/`peek`, `contains?`) whose later args are trivial (locals, constants, var roots), a collection argument `(nth|get coll key)` with a trivial key over a borrowable local, a pooled constant, a stable global, or another such link is lowered by `direct_arg_refs`/`borrow_read` to a pointer into its collection's storage: `rt_bc_ref_nth`/`rt_bc_ref_get` return the element's address when it is stored as a `Value` (generic vector storage, a small map's value), and null otherwise, in which case the read is computed the general way into a temporary. Nothing runs between the borrow and the consuming call (move checks key on the original nodes), so the collection outlives the pointer. With bitcode, `(nth (nth …))` takes the chain instead of the out-of-line `rt_n_nth2`.
- **Ownership:** a borrowed first argument never reaches `count`'s owned slow path (`emit_owned_slow1`), which takes its argument: through a chain that would move an element out of its parent collection (a sorted-set element of `ssa_dominance`'s dominator vector, caught by the full A/B).
- **Verification:** `clj-bc` unit tests (`read_ref` borrows in place, misses on packed items, `nth` of a map, absent keys, nil); corpus `read_chains.clj` (nested chains, defaults, misses, errors, a base local moved later in the same call or at its last use, var and literal bases, loops, `count` over sorted-set and lazy-seq elements with the parent printed after) replayed against `run`, byte-identical to the previous runtime.

Measured (macOS, loops of 100K): `(count (nth grid r))` 316 -> 199 instructions per iteration, `(nth (get g :adj) r)` 478 -> 420, `(nth (nth grid r) c)` 365 -> 338. Program `__text` unchanged on the sampled programs. Interleaved A/B over all 1754 cases: median -761 instructions, mean -17.6K, `ternary_search_trie` -14%, `tarjan_lca` -9%; projected class moves 3 loss->tie, 3 win->tie, 1 tie->loss.

### 6.54 Slice 5w: cheaper set and map probes, and a flat-vector `nth` front

Profiles of the losing cases put lookups and element reads at the top: `contains?` on a 4-element set cost ~310 instructions, on a 100-element set ~370, `get` on a 100-entry map ~400, and each boxed `(nth v i)` ~70 inside its helper plus the call around it (a two-point cross product: ~1470 instructions for 15 reads and 8 arithmetic ops).

- **Equality and hashing (`clj-value/src/value.rs`):** `Value::eq` compares longs and keywords (interned: pointer equality) without a stack frame and hands every other pair to `value_eq_full`; `Value::hash` writes nil and longs inline into the trie probe and hands the rest to `hash_value_full`. Before, both set up a frame of 7 register pairs and 24 hoisted hasher constants for every call: one long's hash cost 63 instructions, a long compare 36 (now ~7 and 11). Same writes and the same answers, so hashed collections keep their shape and order.
- **Small sets (`Set::get_small_scalar`, `clj-bc/src/ops.rs`):** `contains?`, `get` and the 5v read chains answer a nil, boolean, long, char or keyword key over an unsorted small set (≤ 8 members) inline, member by member: only the same kind can equal such a key, except that a big integer can equal a long, which (like hashed and sorted sets and other key kinds) takes the runtime path. Before, every set missed the bitcode helper.
- **Runtime op shims (`clj-rt/src/nabi.rs`):** `rt_n_op1/2/3`, `rt_n_core_ref` and the core fallback write their result with `Value::assign`, which skips the drop-glue call when the slot holds a plain scalar (compiled code drops a slot before calling out); the atom-`nth` paths move out of line so `rt_n_op2`/`rt_n_op3` lose their frame; `apply1`/`apply2_no_re` inline into the shims.
- **`nth` front (`rt_bc_nth_flat`, `clj-aot/src/native.rs`):** before the general `rt_bc_2_nth`/`rt_bc_3_nth`, compiled code tests the collection's tag (`TAG_VECTOR` = 13, pinned by `value_layout_abi`) and, for a vector, calls one shared out-of-line front that writes a flat item (longs and doubles in place, generic items cloned) and returns; tries, strings, transients, subvectors, misses and errors take the general read unchanged. Variants measured and rejected: inlining the front at every site (+32 KB of program text on `visibility_graph`, ~450 bytes per `nth`), and splitting the general read out of line (string and transient reads lost LLVM's inlining of the whole op in small functions: `edit_distance` +2.2%, `knapsack` +3.2%).
- **Verification:** `clj-value` unit tests (small-set lookups against the full `=`, a big-integer member, sorted and hashed sets; `assign` over heap and scalar slots; the vector tag); `clj-bc` unit and shim tests (`nth_flat` hits and misses over long, double, generic, trie, char and string storage; set arms of `get`/`contains?`/`read_ref`); `build_exe` `native_nth_front_and_small_set_lookups_match_general_reads_271`; corpus `flat_nth_set_lookups.clj` (every `nth` storage kind, defaults, negative and out-of-range indices, sets with `1N`, `1.0`, keywords, chars, nil, booleans, strings, vectors, sorted and hashed sets, a set called as a function), output identical between `run` and the native binary.

Measured (macOS, loops of 10K, final build): `contains?` on a 4-element set 361 -> 149 instructions per iteration, on a 100-element set 368 -> 285, `get` on a 100-entry map 402 -> 320; the cross product 1469 -> 1093; a boxed vector `nth` loop 141 -> 118. Interleaved A/B over all 1754 cases (the build before the final coverage refactor, which costs 6-8 instructions per small-collection lookup): median -232 instructions, mean -217K; `count_strings` -9% to -10% on all 11 cases, best cases of `visibility_graph` -9.1%, `dining_philosophers` -8.4%, `binomial_tree` -6.8%, `perfect_hashing` -7.6%; worst long cases `suffix_array`, `kmp` and `boyer_moore` +1.2-1.3% (the front's tag test before an `nth` of a string). Projected class moves against the common OCaml reference: 2 loss->tie, 1 tie->win, none worse. macOS instruction counts include kernel page-fault work: a binary whose layout touches more pages at startup reads ~20K higher on small cases, which Linux Pin does not count.

### 6.55 Slice 5x: core fns passed by name run traversals inline

The native compiler ran a traversal inline (`mapv`, `filterv`, `some`, `every?`, `reduce` with an init, and `map`/`filter` under `vec`, `into []`, `count`, `dorun` or `reduce +`) only when its fn was a literal `fn`. A `clojure.core` fn passed by name — `(mapv inc xs)`, `(reduce + v)`, `(reduce max v)`, `(mapv count tables)` — took the runtime traversal, which calls the fn through the generic apply per item: ~340 instructions per item for `(mapv inc (range 32))` against ~45 for `(mapv (fn [x] (inc x)) …)`, and ~93 per item for `(reduce + v)` over 32 doubles. 93 of the 105 programs with an INS non-win pass a core fn by name to a traversal (`reduce +` 62 times, `reduce min`/`max` 31, `map *` 12, …), and several build their lookup tables that way at load time.

- **Eta templates (`clj-eval/src/aot_front.rs`):** after reading the program, the front end collects every `clojure.core` fn the units pass by name as a traversal's fn (`eta_wanted`) and analyzes `(fn* [a…] (clojure.core/f a…))` once per name and arg count on the build evaluator. A template is kept (`FrontEnd::eta`) only when its body is one direct native call over exactly its params, in order; a fn defined in `core.clj`, an intercepted form, or an arity the analysis rejects keeps the general path.
- **Inline traversal (`clj-aot/src/native.rs`):** the traversal matcher resolves its fn argument through `LoopFns`: a literal `fn`'s arity as before, or the template's direct call re-based onto the loop's param slots (`eta_fn`, built once per traversal site), so the item loop, the scalar traversal (`(reduce + v)` over doubles or longs runs unboxed) and the moves analysis treat it like a literal `fn`. The general path (other collections, `reduce` over no items) passes the named fn itself, never a closure. A program that can rebind core vars (`mentions_rebinding`) gets no templates: its traversals call the current root.
- **Folds:** `(reduce + [init] xs)` over the items themselves runs as the existing sum loop over an identity body (the scalar sum when the items are numbers); `(reduce f xs)` without an init folds from the first item, and over no items takes the general path, so `(reduce + [])` is `0` and `(reduce max [])` throws the same arity error.
- **Runtime-answered traversals stay in the runtime:** `(filter|filterv seq xs)` tests emptiness without building seqs and keeps a split string's tokens unmaterialized, `(some nil? xs)` answers a packed vector without a walk, and `(mapv parse-long xs)` parses in one native loop (`runtime_traversal`). An earlier build that inlined them measured a tokenizer call 6.5K -> 12.2K instructions and `(some nil? ints)` 766 -> 6959 (`dijkstrashortreach` +23%).
- **Lazy collections keep the plain call (`lazy_colls`):** over a lazy sequence a core call makes (`(reduce bit-or 0 (map f xs))`, `(map * xs (map g ys))`), the runtime walks the sequence fused with its consumer, while the inline loop would take its general path over the realized sequence (`least_squares`, `perm_rank`, `graphplan` +1-2%): a traversal of a core fn by name over such a collection is not expanded. (Literal `fn`s keep their inline loops as before.)
- **Verification:** `clj-eval` unit test (which names get templates, a `core.clj` fn and non-core fns do not); `build_exe` `native_core_fns_by_name_run_traversals_inline_271` (sums of longs, doubles, ranges and lists, with and without an init, `reduce max` over one item, a list, and no items, `mapv` over vectors and ranges, the runtime-answered pairs); corpus `eta_traversals.clj` (every traversal kind, empty and one-item folds, `+'`, ratios, overflow, `reduced`, cast and nil errors) and `eta_rebind.clj` (rebound `inc` and `+` show through), output identical between `run` and the native binary, error messages included.

Measured (macOS, loops of 10K): `(mapv inc (range 32))` 10828 -> 1451 instructions, `(reduce + v)` over 32 doubles 2958 -> 978, a binomial-tree step (`mapv` of a closure then `reduce +`) 5495 -> 3315. Interleaved A/B over all 1754 cases against a baseline built in the same session (same toolchain, fat-LTO runtime archive): median -1.4K instructions, mean -9.8K, ratio quantiles q05 0.981, q50 1.000, q95 1.007; best `quickselect` -6.9%, `counting_sort` -5%, `min_conflicts`/`walksat`/`sudoku` -3.5%; the worst cases (+1.2-1.8%) are small programs within the macOS page-fault noise. Projected against the common OCaml reference: INS 1486/47/210 -> 1488/49/206. (A first A/B against a baseline whose runtime archive was built without LTO reported a -98.5K median: that was the archive, not this slice; see the PR correction.)

### 6.56 Slice 5y: a flat vector's items live in its own data (two allocations, not three)

A vector was `Arc<VectorData>` holding `Packed::Flat(Arc<Vec<T>>)`: three heap blocks per flat vector (the data, the `Arc` around the items, and the item buffer), freed one by one, and an extra pointer hop on every read. On Linux glibc malloc and free average 18% of the instructions of the profiled losing programs (fork probe `probe/mb6-271`), and a three-long vector literal costs ~594 instructions per iteration, 183 of them in malloc and free (`probe/mb7-271`).

- **Storage (`clj-value/src/trie.rs`):** `Packed::Flat(Vec<T>)` holds the items inline in the vector's data, so a flat vector is two blocks (data and buffer). The vector's `Arc` is now the only sharing: every `Packed` write runs on storage its vector holds alone, so the per-run shared checks go away and `unique_flat_mut` is the flat run itself.
- **Shared updates (`Vector::data_mut`, `clj-value/src/value.rs`):** every vector writer (`conj`, `assoc`, `pop`, typed-storage retyping, `take_nth`, window appends, `with-meta`) takes its data through `data_mut`, which copies a shared vector's data once (`Arc::make_mut`, as before: a flat run up to `FLAT_MAX` copies at most 8 KB) except that a shared flat run past `FLAT_MAX` (built in one piece) moves into the trie in one pass, so loop-threaded updates of a long vector stay logarithmic (#227) instead of copying it whole each time.
- **Costs that move:** `with-meta` on a shared vector now copies its flat items (it shared them before); a uniquely held vector still updates in place, and the in-place `assoc`/`conj` bitcode paths are unchanged.
- **Verification:** `clj-value` unit tests (`shared_long_flat_vectors_update_through_the_trie_271`: long, double, char and generic flat runs past `FLAT_MAX` update through the trie and leave the shared original intact, packed strings keep their storage, short runs copy; `packed_replace_coverage`; the unique-update test now shows a metadata view owning its items); every vector test and the snippet sweep unchanged.

Measured (macOS, interleaved A/B over all 1754 cases against the same-session baseline): median -11.1K instructions, mean -147K; ratio quantiles q05 0.983, q50 0.997, q95 1.006; `beam_search` and `trie/no_prefix_set` -6%, `sudoku` -4%. Projected INS against the common OCaml reference 1486/47/210 -> 1489/48/206 (Linux glibc allocations cost more than macOS mimalloc's, so the Linux effect should be larger).

### 6.57 Slice 5z: small map entries inline, stripped release binaries, and an x86-64-v3 runtime archive

- **Small maps (`clj-value/src/value.rs`):** a shaped map's entries (`SmallMap`: the interned shape plus its values) live inline in the map's data instead of behind their own `Arc`, so a map version is two blocks (data and values) instead of three, as 5y did for vectors. A shaped map's identity (`Map::id`) is its data's address (a hash map's stays its trie's): a shared map's writers copy its data once (fresh identity), a uniquely held one updates in place (unobservable, as before), `Map::set_small` installs new entries without copying a shared map's old ones, and a `dissoc` or `update` miss returns the map itself (no copy, same identity). Empty maps allocate no entries. Linux A/B (fork probe `probe/ab-map-271`, callgrind, public programs on synthetic inputs): `count_strings` -1.8%, `mutex` -1.2%, `delaunay`/`voronoi` -0.5%, the rest within +-0.2%; peak RSS within noise.
- **Stripped binaries:** the release profile strips the symbol table (`strip = true`) and `clojure-rs build` links stripped binaries (`link::strip_args`: `--strip-all` on Linux, local and debug symbols on macOS; Windows keeps symbols in a separate PDB). Symbols are never loaded, so only the files shrink. `--keep-symbols` keeps them for profilers; the hot-text regenerations, which profile release builds by function name, keep them (`regen-run.sh` overrides the profile, `regen-aot.sh` passes the flag when the revision's compiler has it).
- **x86-64-v3 runtime archive:** on Linux x86-64 the release and the h2h AOT legs also build the runtime with `-C target-cpu=x86-64-v3` (`libcljrt-v3.a`, next to `libcljrt.a`); `clojure-rs build` links it when the host CPU has every v3 feature (`link::host_has_x86_64_v3`, std feature detection) and `--portable` is not given, else the baseline archive (an explicit `--runtime-archive` always wins), so binaries built on older CPUs still run there. Linux probes (`probe/opt-271`, `probe/cpu-271`, callgrind on public programs): the v3 runtime cuts 2-4% of instructions on most programs with input (`lzw` -4.1%, `count_strings` and `lexer` -3%, `steiner` -3.1%, `sudoku` -1.8%), x86-64-v2 gains nothing, compiling the program module for v3 adds at most 1%; peak RSS within noise. Also measured and not adopted: the `default<O3>` AOT pipeline with aggressive codegen (neutral on macOS and Linux) and Linux mimalloc (+370-480K startup instructions, +200-700 KB RSS).
- **Verification:** `clj-value` unit test `shaped_map_identity_follows_its_data_271` (misses keep identity, shared writes mint it and leave the original intact, value takes clone or move, updates over shared maps, emptying shared and unique maps); `clj-aot` unit tests for `strip_args` and the archive choice (override, v3 present or absent, preferred or not); `build_exe` `native_binaries_strip_symbols_unless_kept_271` (same output, the stripped binary smaller).

### 6.58 Slice 6a: nested and filtered `for` comprehensions under `vec` run as native loops

A `for` with more than one binding, or with `:when`/`:let`, expands (`clj-eval/src/core_macros.rs`, `comprehension`) to lazy `mapcat` and `__for-keep*` stages, each with a closure made per outer item and a lazy cell per item: a 10x10 nested `(vec (for [a … b …] [a b]))` cost ~1220 instructions per item on macOS against ~480 for the single-binding form (already a traversal loop, 5x), and a nested `for` with `:when` ~2170. 42 of the `for` sites in the programs with INS non-wins nest or filter; 77 of all 99 are consumed by `vec` or `into []`.

- **Lowering (`clj-aot/src/comprehension.rs`):** `(vec E)` and `(into [] E)` over such a comprehension `E` match it whole first (`stage`/`step`: a plan of stages, `:let`s, `:when`s and the item; any other shape keeps the general path, nothing emitted), then run it as nested loops pushing into one vector builder: a stage over a vector or long range indexes its items and binds its `fn` param in place, `:let`s bind as `let` does (destructuring, dead drops), `:when`s branch, the item is lowered inline; a stage over any other collection makes the stage's closure and calls its own core function over the evaluated collection, appending the items it yields (`rt_n_vec_push_all`, new). The analysis's last-use moves assume each stage `fn` is a closure made once, so the loops take no moves and no tail position.
- **Semantics:** `vec`/`into []` realize the whole sequence in order, so the items, their order, every side effect's order and the first exception are the lazy form's. An item that names the `__for-keep*` sentinel (which the lazy stage drops) keeps the general path. Consumers that may stop early (`first`, `some`, `every?`) are not lowered: the lazy form realizes chunks of 32, so a side-effecting body would run a different number of times.
- **Verification:** corpus `comprehensions.clj` (nested, `:when`, `:let` with destructuring, three levels, empty collections, lists, sets and double ranges at each level, an exception mid-way, side-effect order, and the shapes that keep the general path), output identical between `run` and the native binary.

Measured (macOS): a nested 10x10 `(vec (for …))` 121.8K -> 52.7K instructions per call, a nested `for` with `:when` 119.2K -> 16.3K. Interleaved A/B over all 1754 cases against the same-session 5z build: median +0.6K, mean +2.6K (no case with a nested comprehension worse; the largest rises are programs without any `for`, which swing both ways across their own cases); `pl_resolution` -14%, `center_star` -6 to -10%, `line_arrangement` -5%, `generalized_suffix_tree` -5%.

### 6.59 Slice 6b: more consumers of native comprehensions

6a lowered nested and filtered comprehensions under `vec` and `into []` only. Among the losing programs' nested or filtered `for` sites, the other consumers are `reduce` (6), `into {}` (4), `first` (3), `let` (3), `every?` (2), `count`, `into #{}`/`(sorted-set)` and `some` (1 each).

- **Whole-sequence consumers (`clj-aot/src/comprehension.rs`):** `(into to E)` for any `to`, `(count E)`, `(sort E)` and `(set E)` always realize the whole sequence in order, so they run the comprehension into a vector natively and then call the consumer over it (`to` evaluated first, as the call would): same result, items, effects and first exception.
- **`(reduce + [init] E)`:** `+` never returns `reduced`, so the items fold into the accumulator as they come (`rt_n_add_acc`, no intermediate vector); without an init the first item becomes the accumulator untouched, as `reduce` takes it, and an empty sequence gives `0`; a stage over a collection it cannot index folds the stage's own sequence (`rt_n_add_all`, new).
- **Not lowered:** `first`, `some` and `every?` may stop early, and the lazy form realizes chunks of 32, so a side-effecting body would run a different number of times; `reduce` with any other function may stop at `reduced`.
- **Verification:** corpus `comprehensions.clj` gains every consumer, `+` folds with and without an init, over no items and one non-number item, on the general path, and with cast errors raised mid-fold; `native_corpus` replays it through the JIT.

Measured (macOS, per call): `(count (into {} (for [a (range 8) b (range 8) :when (< a b)] …)))` 260K -> 178K instructions, `(reduce + 0.0 (for [k (range 50) :when …] (nth w k)))` 37.3K -> 4.9K.

### 6.60 Slice 6c: multi-pair `assoc` as a chain of owned updates

`(assoc m k1 v1 k2 v2 …)` went through the generic core call (args borrowed and cloned, so every pair copied a shared map or vector again): on a shared 6-key map the second pair cost ~1300 instructions against ~500 for a chained `(-> m (assoc k1 v1) (assoc k2 v2))`, whose second step updates the copy the first made in place. The programs with INS non-wins have 31 multi-pair `assoc` sites in 8 files (82 across all programs).

- **Lowering (`try_lower_assoc_chain`, `clj-aot/src/native.rs`):** with 5 or more args and an odd count, every arg is evaluated first, in order (a local at its last use moved into its slot, as owned calls take it), then one owned three-arg `assoc` per pair (`rt_n_assoc_own3`, after the in-place bitcode path for vectors), each over the previous result: exactly `assoc`'s definition, so values, errors (out-of-range index, non-associative target) and metadata are the call's. An even count (a missing value) keeps the general call and its message.
- **Verification:** corpus `assoc_chain.clj` (maps, vectors, nil, repeated keys, shared originals left intact, loop accumulators moved and not moved, argument side-effect order, out-of-range and cast errors, the odd-argument error, metadata), output identical between `run` and the native binary.

Measured (macOS, per iteration over a shared 6-key map): two pairs 2859 -> 2133 instructions, three pairs 3340 -> 2530.

### 6.61 Slice 6d: single `map` stages and `apply` under whole-sequence consumers

The traversal loops (5x) cover a single `(map F xs)` under `vec`, `into []`, `reduce +`, `dorun` and `count`/`filter`, but not under the other consumers that realize a whole sequence, and not `(apply g (map F xs))`: 32 `(apply str (map …))` sites sit in the programs with INS non-wins.

- **Lowering (`clj-aot/src/comprehension.rs`):** a single unfiltered `map` stage now lowers under the whole-sequence consumers of 6b (`into` with any target, `count`, `sort`, `set`), and `(apply g E)` joins them for `g` among `str`, `max`, `min`, `+`, `*`, `merge`, `vector`, `hash-set` (`applies_whole`: their variadic arity consumes every arg in order; `concat` walks its rest args lazily and stays out): the items run natively into a vector, then the original call takes it. `vec`/`into []` and `reduce +` over a single `map` keep their traversal loops.
- **Verification:** corpus `comprehensions.clj` (every `apply` fn, `into {}` and `into #{}` over a map, `set`, `count` with side effects in order, `sort`, lists, empty inputs, `(apply max ())`'s arity error, `apply concat` left lazy), output identical between `run` and the native binary.

Measured (macOS): interleaved A/B over all 1754 cases against the same-session 6c build: median -0.4K, mean -4.2K; `value_iteration` -13%, `purged_cv` -5%, `gray_code` -4%, `jones_mueller` -3.7%; no rise above 1.5% outside the randomized quick_sort/quickselect cases.

### 6.62 Slice 6e: fns passed by name, program `defn`s included

A traversal (`mapv`, `filterv`, `reduce`, `some`, `every?`, a `map` under `vec`/`into []`/`dorun`, …) of a literal `fn` or a `clojure.core` fn by name runs as a native loop (5x), but a program `defn` by name (`(mapv parse-row lines)`, `(reduce step acc xs)`) took the generic core call: one generic invoke per item, about 390 instructions per item on macOS against 15–55 for the native loop. Programs pass their own fns by name to traversals at 139 sites, 32 `mapv`, 13 `map`, 4 `reduce` and 4 `every?` of them in the programs with INS non-wins. Comprehension stages (6a/6d) took only literal `fn`s.

- **Resolution (`Lowerer::named_fn`, `clj-aot/src/native.rs`):** a fn passed by name becomes the one-arity eta `fn` calling it with the traversal's args: a `clojure.core` fn through its eta template (as before), or a program `defn` whose var is stable at the site (`stable_of`: its `def` has run, as for the cheap direct-call entry, and nothing in the program rebinds it) and which has a compiled fixed arity for the arg count (`direct_target`). The eta's call is the ordinary guarded direct call (or the callee inlined), and its args get last-use moves like a literal `fn`'s (a `reduce` accumulator moves into the call). The traversal would have read the var's root once; a stable var's root cannot change, so calling it per item is the same fn. A var the program redefines, a fn used before its `def` has run, and a variadic or missing arity keep the generic call. The general path (a non-vector, non-range collection) evaluates the name as before.
- **Comprehension stages (`clj-aot/src/comprehension.rs`):** a `map` stage whose fn is a name that resolves (`StageFn::Named`, `Step::NamedItem`) lowers like a literal one: `(apply str (map label xs))`, `(into #{} (map f xs))`, `(count (map f xs))`.
- **Verification:** corpus `named_fns.clj` (mapv, filterv, reduce with and without init and an accumulator updated in place, every?, some, two collections, multi-arity and variadic defns, a private defn, side effects in order, exceptions inside the traversal, empty inputs and `reduce`'s zero-arg error, traversals inside compiled fns and of a fn parameter, a forward-declared fn used before its `def`, a self-redefining var whose root the traversal reads once), output identical between `run` and the native binary.

Measured (macOS, per item of `(mapv f (range n))` with `f` a one-line `defn`): 393 -> 15 instructions at top level, 384 -> 21 inside a compiled fn.

### 6.63 Slice 6f: runtime declarations say what the runtime does with their params

A native program links against the runtime archive without cross-module optimization, and the compiler declared every runtime ABI fn bare. So once a slot's address reached any runtime call (every boxed op's slow path takes its operands by address), LLVM had to assume every later call could rewrite it. It kept every tag reload and every drop check after each call: a boxed `(inc d)` in a loop costs a tag-switch drop of a temporary the loop had just set to nil.

- **Attributes (`Lowerer::annotate_runtime_decl`, `clj-aot/src/native.rs`):** each `rt_n_*` declaration is annotated from the fn's Rust signature in `clj-rt/src/nabi.rs`, parsed once from that source (`runtime_abi`), so the declarations cannot drift from the definitions:
  - `nounwind`: an `extern "C"` Rust fn aborts on a panic instead of unwinding.
  - `captures(none)` on reference params (`&T`, `&mut T`, `Option<&…>`): safe Rust cannot keep a non-`'static` reference past the call, and the ABI's only `unsafe` code builds slices bounded by the borrow.
  - `readonly` on shared references: every interior-mutable part of a `Value` (atoms, refs, transients, volatiles, …) sits behind an `Arc`, so a `&Value` never writes the 32-byte slot.
  - Nothing on raw or fn pointer params: `Code`, which `rt_n_make_fn` and the boot keep.
  - The `f64` math fns (`rt_n_math_*`) are pure: `memory(none) willreturn nounwind`.
- **Effect:** in the `mask-digits` table loop of the sudoku programs the top-level function's IR shrinks from 1622 to 1249 lines (68 to 53 drop calls, 61 to 41 tag switches).
- **Verification:**
  - Unit test `runtime_abi_params_follow_their_rust_types_271`: a synthetic signature with every param shape, plus the real table.
  - Emit test `runtime_declarations_carry_their_abi_attributes_271`: the declared `rt_n_drop`, `rt_n_op2` and `rt_n_math_sin`, with their attribute groups.
  - The full native corpus and the snippet sweep are unchanged.
  - Three emit tests pinned block names inside traversal loops (`some_hit`, `every_fail`) and the scalar loop's `sc_deopt`; LLVM now sinks those blocks' stores into the loop exit and drops the emptied blocks. The tests instead check that each traversal keeps its general-path entry block (`loop_general`, one per lowered traversal) and that the loop keeps its unboxed counter and in-place `conj!` (`sc_int_ok`, `sc_conj_try`).

Measured (macOS, interleaved A/B over all 1754 cases against the same-session 6e build): median -5.4K instructions, mean -115K; ratio quantiles q05 0.985, q50 0.998, q95 1.009; `convex_hull` -12%, `segment_tree` -10%, `boyer_moore` -5.5%, the sudoku programs -5%; `edit_distance` +7 to +8% (its scalar loop now unrolls differently; it runs at 0.39x OCaml).

### 6.64 Slice 6g: closed integer tables folded at build time

Some programs build lookup tables at startup: `(def mask-digits (mapv (fn [m] (loop … (conj acc d) …)) (range 1024)))` in the two sudoku programs, `(mapv #(quot % 9) (range 81))`. These are the same longs every run, but on Linux they cost the sudoku programs about 3.05M of their ~3.97M instructions per case. The table's loop runs boxed because its accumulator is a persistent vector, and OCaml solves the whole case in 1.28M.

- **Fold (`clj-aot/src/fold.rs`, `Lowerer::try_fold_init`):** a `def` whose init is `(mapv F (range …))` with literal long bounds, or `(mapv F table)` over an earlier folded `def` whose var is stable at the site, is evaluated at build time when the program cannot rebind core vars.
  - `F` must be a literal one-arg `fn` or a fn by name.
  - A bounded evaluator runs `F` per item: `let`, `if`, `do`, `loop`/`recur`, an empty `[]`, `conj` of a long onto a vector, `count`, `not`, and the core long natives, called through their own entries so values and overflow errors are the runtime's.
  - It declines on any other form, any error, 4M evaluated expressions, more than 65536 longs, an empty range, or a result other than all longs or all rows of longs.
  - The program then builds the value from the longs in one runtime call (`rt_n_int_table`), and a declined init runs as compiled.
- **Identity:** at run time every empty row is the one `[]` object of the site it came from, and each non-empty row is a new vector from `conj`, so `rt_n_int_table` builds one shared empty vector and a new vector per non-empty row.
  - A fold whose empty rows come from two `[]` sites declines.
  - So does one whose results might be shared objects: a vector literal that is not empty, or rows returned over a folded table.
- **Verification:** corpus `int_tables.clj`, with output identical between `run` and the native binary. It covers:
  - Folds: flat, row, and over-a-table tables (over rows and over a flat table), negative steps, `nil` locals, empty rows and their identity, equal rows staying distinct, and appending a keyword or double to a folded row.
  - Every decline: a program fn, a string, a double, a bool result, a boolean operand of a long op, a pooled literal, two `[]` sites, a var bound, an enclosing local, an empty range, a literal collection source, rows over a table, an overflow caught, a printing init, the step budget, the item cap, `conj` of a string, destructuring, and a non-long native.

Measured (macOS): the sudoku programs' startup with empty stdin falls from 3.83M to 1.90M instructions above the process floor, and every case drops by about 1.9M with identical output.

### 6.65 Slice 6h: `into` takes a comprehension a chunk at a time

6b made `(into to E)` over a native comprehension push every item into one vector and then call `into` with it. The lazy form `into` walked a chunk at a time, and the items it had poured into `to` were garbage, so peak memory held the target plus one chunk. Natively it held the target plus every item. The 6a–6c head-to-head showed it: `chan_hull` (`(into {} (for [[gi g] (map-indexed vector groups) p g] [p gi]))`) rose 276 KB of peak RSS on all four cases, and `bentley_ottmann`'s `(into #{} (mapcat …))` sets rose up to 320 KB, giving 7 new MEM losses.

- **Sink (`Sink::Chunk`, `clj-aot/src/comprehension.rs`):** `(into to E)` with any target other than `[]` (the vector itself) pushes items into a builder, and every 32 items `to = (into to chunk)` runs through `into`'s own native (`core_own_args`). The pending items are poured before a stage's lazy general path pours its sequence, and once more at the end.
  - `into` over a concatenation is `into` over its parts in turn, `conj` by `conj` for every target type (lists prepend the same way), so values, metadata and error messages are `into`'s.
  - Peak memory is back to the target plus a chunk, and `chan_hull` also runs about 1.7% fewer instructions than under 6b (macOS).
- **Verification:** corpus `comprehensions.clj`, with output identical between `run` and the native binary. It covers flush boundaries (0, 1, 31, 32, 33, 64 and 65 items), map, set and list targets, a `nil` target, a sorted-map target, a vector target with metadata, list-backed stages taking the general path, a nested mapcat into a set, and a bad map entry's error.

### 6.66 Slice 6i: tests and seq steps read their locals in place

A walk over a seq — `(loop [s (seq v) a 0] (if s (recur (next s) (+ a (first s))) a))` — cost ~1180 instructions per step on a vector (95 such loops in 71 programs, 48 in the programs with INS non-wins), against 64 for `(reduce + v)`. Two of its costs were copies the step never needed:

- **Local tests (`lower_cond`, `clj-aot/src/native.rs`):** `(if s …)` copied `s` into a temporary to test it, then dropped the copy (a refcount bump and release for a heap value). A test that is a local's read (not its last use) now tests the local's own slot.
- **Shared `next` (`is_direct_moving`):** `(next s)` where `s` is read again later went down the owned path, which copied `s` into the call. The copy shares the seq, so the in-place step can never apply, and the copy only costs a bump and a drop. It now takes the borrowing call. `rest` keeps the owned path: its general native costs more than the copy.
- **Verification:** every corpus test of `if`, `when`, `and`/`or` over locals, the seq walks, and the snippet sweep are unchanged.

Measured (macOS):
- **Per step:** the vector walk 1182 -> 975 instructions, a walk with `first` taken before `next` 655 -> 550.
- **Interleaved A/B of the local tests alone**, over all 1754 cases against 6h: median -0.4K instructions, mean -16K; `edit_distance` -1.5 to -2%, `interval_tree` -1.2%.

### 6.67 Slice 6j: map patterns read their source in place

A plain map pattern (`{:keys [a b]}`, `{a :a}`, `:strs`, `:syms`, `:or`) converted its source first (`rt_n_destr_map`: a list becomes its map, as the interpreter's `seq?` test does), and that conversion copied every other source into a temporary (a refcount bump and release for a map), read the keys from the copy, then dropped it: ~580 instructions per two-key bind.

- **Lowering (`Lowerer::destr_source`, `clj-aot/src/native.rs`):** without `:as`, the source's tag is tested inline. A list still converts into its map, and any other value is read in place (a `select` between the two). With `:as` the converted source is bound, as before.
  - `TAG_LIST` (12) is pinned by `value_layout_abi` beside the other tags the compiler tests.
- **Verification:** corpus `destructure_sources.clj`, with output identical between `run` and the native binary. It covers maps, lists, `nil`, vectors, sorted maps, strings, `:or` with and without a present `nil`, `:strs`/`:syms`, `:as`, nested patterns, an odd-length list, and a loop over mixed sources.

Measured (macOS): a two-key `{:keys [a b]}` bind in a loop 580 -> 472 instructions per iteration.

### 6.68 Slice 6k: variadic `conj` as a chain of owned updates

A trace of the generic core calls the near-line programs make (cases at 1.0–1.4x OCaml, a temporary local probe counting `core_fallback`/`call_core_var` by var) showed about 170 generic calls per run. Of those, 2002 of 4402 were `conj`, and 2000 of them came from one site in `var_es`, `(conj out z0 z1)`. Only the two-arg `conj` has an owned op, so the variadic call took the general native every time, about 1000 times per case.

- **Lowering (`try_lower_conj_chain`, `clj-aot/src/native.rs`):** `(conj coll x1 x2 …)` with two or more items evaluates every arg first, in order, with a local at its last use moved. It then runs one owned two-arg `conj` per item (the lean in-place bitcode path first), each over the previous result. That is `conj`'s own definition, so values, metadata and errors are the call's. `assoc` and `conj` now share one dispatcher (`try_lower_update_call`).
- **Verification:** corpus `conj_chain.clj`, with output identical between `run` and the native binary. It covers vectors, lists, `nil`, sets, map entries and maps, sorted sets, metadata, `nil` items, subvectors, a 40-item vector, argument side effects in order, a loop accumulator, a shared original left intact, and the pair, string and number errors.

Measured (macOS): `var_es` 23.99M -> 23.54M instructions on its first case, identical output.

### 6.69 Slice 6l: stable maps, sets and keywords passed by name

The token parsers many programs share call `(every? digit-values digits)`, passing a map-valued var as the predicate. 6e resolves a program fn passed by name only when it is a `defn` with a compiled arity, so these traversals took the generic core call.

- **Resolution (`Lowerer::named_callable`, `clj-aot/src/native.rs`):** a stable var whose root is a map, set or keyword (`StableKind::Lookup`), used after its `def` has run and called with one or two args, now resolves as well. Its eta `fn`'s call `(v x)` lowers as the stable lookup (`try_lower_stable_lookup`, `rt_n_lookup_call`).
  - The traversal calls `(pred x)` on the root it read, and a stable root cannot change, so the eta is the same call.
- **Verification:** corpus `named_fns.clj`, with output identical between `run` and the native binary. It covers `every?`, `mapv`, `filterv`, `some`, `reduce`, `count`/`filter` and `apply str`/`map` over a `zipmap` map, a set literal and a keyword var, including missing keys and the two-arg lookup with a default.

Measured (macOS, interleaved A/B of 6k and 6l over all 1754 cases against 6j): median -1.2K instructions, mean -3.1K; `var_es` -6 to -7% (6k).

### 6.70 Slice 6m: `=` of flat long vectors inline

The same trace found `=` as the next generic call. `var_es`'s `(count (filter #(= % [0 0]) tr))` compares each pair with a vector literal about 96 times per case, and neither the bitcode `=` (`rt_bc_2_eq`) nor the runtime's inline op (`scalar_eq`) knew vectors, so every comparison took the general native.

- **Equality (`clj-bc/src/ops.rs` `apply2_eq`, `clj-eval/src/vm_intrinsics.rs` `scalar_eq`):** two vectors whose storage is flat packed longs compare their long slices. Any other vector pair still takes the general native, with its sequential, numeric-category and nested semantics.
- **Verification:**
  - Unit tests `flat_long_vectors_compare_inline_271` (clj-bc and clj-eval): equal, unequal, different lengths, and a non-long vector declining.
  - Corpus `vector_equality.clj`, with output identical between `run` and the native binary: the filter tallies, vector against list, nested vectors, a double against a long, empty vectors, 2000-long vectors, `not=`, and a subvector.

Measured (macOS): `var_es` 23.48M -> 23.22M instructions on its first case (with 6k, 23.99M -> 23.22M), identical output.

### 6.71 Slice 6n: `second` of a vector inline

The generic-call trace counted 95 `(second node)` calls in `pratt_parser`, which reads its AST nodes as vectors (`[:num 5]`). `second` had no inline op, so every call took the general native through the var table.

- **Op (`Second`, `clj-eval/src/vm_ops.rs`, appended to the one-arg inline table so no other op's index moves):** a vector gives its item at 1, or `nil` when it is shorter, and `nil` gives `nil`. Every other collection (lists, strings, maps, seqs, subvectors, sets) takes the native. The VM and the native runtime share the table.
- **Verification:** corpus `second_inline.clj`, with output identical between `run` and the native binary. It covers vectors with fewer than two items, `nil` items, lists, strings, maps, ranges, subvectors and sets.

Measured (macOS): `pratt_parser` -23K and -85K instructions on its first two cases.

### 6.72 Slice 6o: the interpreter's training covers the core ops every program reaches

The strict `recipes` leg failed twice on one case. `bentley_ottmann` measured the interpreter's peak RSS at 1.151–1.153x CPython against the 1.15 limit, while master's own interpreter sat at 1.144–1.148x on the same runners. A Linux probe showed why: most of that RSS is resident binary text, and the run executed 100 functions outside `.text.hot`. Those functions total only 76 KB, but they touched 24 of the 61 64 KiB windows of cold `.text`, and the kernel faults each window in whole (fork `probe/rss-271`, `probe/cold.sh`).

They were not specific to the program. They were the generic paths the 26 training workloads never run:
- reading stdin (`slurp *in*`, which no interpreter workload called);
- the loop tier's specializations of the comparison, arithmetic and collection ops for each operand shape (register, constant, upvalue, computed);
- vector equality, hashing and ordering;
- the lazy `filter`/`concat`/`drop-while` walks;
- number parsing;
- comparator-sorted collections.

- **Workload (`scripts/bench/hot-text/workloads/core-ops.clj`):** a generic program that exercises those core operations, each binary op in every operand shape generated by a macro, the unary ops and collection reads in loops, and seqs consumed one item at a time. It contains no snippet code or inputs. Like every workload it trains both orders: `regen-run.sh` for the interpreter, and `regen-aot.sh` compiles it natively.
- **Measured (the same Linux probe, the PR head with and without the workload):**
  - `bentley_ottmann`'s interpreter RSS fell from 10312 to 9412 KB (-900 KB), with cold executed functions down from 100 to 39.
  - A trivial program stayed at 5312 KB, against 5320 without the workload.
  - Its output is identical between `run` and the native binary.

### 6.73 Slice 6p: `subvec` inline

The token parser many programs share opens with `(let [cs (vec tok) … digits (subvec cs (if neg 1 0))] …)`. In the generic-call trace, `point_in_polygon` and `closest_pair` made about 25 `subvec` calls per case, each through the general native and the var table.

- **Ops (`Subvec2`, `Subvec3`, `clj-eval/src/vm_ops.rs`, appended to the two- and three-arg inline tables so no other op's index moves):** `(subvec v start [end])` calls the native's own `subvec3`. The two-arg form does so only for a vector, with its count as the end. Any other argument, and every error, takes the native, so its exceptions are unchanged. The VM and the native runtime share the tables.
- **Not taken:** an inline `vec` of a string or vector. Routing `vec` through the inline table moved some calls off the native that consumes the lazy seqs it owns: a later test run no longer reached those walks.
- **Verification:** corpus `vec_subvec_inline.clj`, with output identical between `run` and the native binary, and between this `run` and 6n's. It covers the parser shape, nested subvectors, empty windows, the out-of-range, negative, reversed and non-vector errors, the subvector class, and equality.

Measured (macOS): `point_in_polygon` -10K and -72K instructions on its first two cases, `closest_pair` -56K and -20K.

## 7. Acceptance Checklist (Issue #271)

- [ ] `h2h-compare` `--allow-extra-challenger` implemented and covered by Go tests.
- [ ] `.github/workflows/h2h-cpu-mem.yml` measures `AOT from`, `AOT to`, and `OCAML` and runs `aot-rsrs` and `aot-ocaml`.
- [ ] Slice 1 CI run completes cleanly and captures Linux Pin `AOT` vs `OCAML` baseline artifacts.
- [ ] Unwaived `h2h-cpu-mem.yml` passes `recipes`, `recipes-rsrs`, `aot-rsrs`, and `aot-run` with 0 MEM and 0 INS losses at `--margin 0.15`, and `aot-ocaml` shows AOT outperforming OCaml on every compared case (wins only on MEM and INS).
- [ ] All CI quality gates in `rust.yml` remain green at 100.00% coverage.
