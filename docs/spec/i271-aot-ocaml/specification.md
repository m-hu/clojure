# Issue #271: AOT Performance Catch-Up with OCaml — Specification

Status: Draft (Spec-first commit)
Issue: https://github.com/hhh-pa/clojure/issues/271
Baseline: `ca27a18e` (`master`, post-#259/#260)

## 1. Goal

Make native binaries produced by `clojure-rs build` match or outperform native OCaml (`ocamlopt -O2`) on the Linux `h2h-cpu-mem` workflow across the shared snippet suite: **zero instruction-count losses and zero peak-RSS losses at the 1.15x margin (`--margin 0.15`, `--metrics mem,ins`)**.

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
3. Remove `[waive-ocaml]` usage once unwaived `aot-ocaml` reports `0 MEM` and `0 INS` losses across all compared cases.

### 6.1 `BootMode::Lite` semantic parity contract

`BootMode::Lite` binaries keep no `clojure.core` / `clojure.string` source resident, so every native that stands in for an interpreted definition under Lite MUST reproduce `clojure-rs run` for **every** input — values, exception classes, and messages — not only the inputs its fast path takes (Invariant 2):

1. **Natives replacing `core.clj` definitions** (`reduce-kv`, `run!`) transcribe the definition literally: a non-fn is invoked exactly as the interpreted `(f …)` invokes it (the interpreter's own invoke error, never an `IFn` pre-check), and `reduce-kv` takes `core.clj`'s index loop for every `vector?` (subvecs, `vector-of`), not only persistent vectors.
2. **`clojure.string` slow paths** (inputs outside a native's fast path): `run` keeps calling the interpreted `string.clj` definition; Lite runs that definition's literal transcription through the same interop entry points (`__meth*` for `.length` / `.charAt` / `.toString` / `.startsWith` …, the `Character/isWhitespace` emulation, the `zero?` / `dec` / `inc` / `split` natives), so `java.io.File`, `StringBuilder`, nil, and non-`CharSequence` receivers fail or succeed exactly as under `run`.
3. **Enforcement:** `crates/clj-eval/tests/lite_parity.rs` evaluates every case in a full-boot evaluator (the oracle) and a Lite evaluator and requires identical printed outcomes; a case either side cannot read, analyze, or realize fails the suite (no vacuous matches).
4. **Deferred (not Lite-eligible in Slice 2):** `clojure.set` and `clojure.string/replace`. Programs referencing them keep the full boot. A native `replace` / `clojure.set` cannot be exact yet: `run`'s own `replace-by` throws `No matching ctor found for class java.lang.StringBuffer` on any fn replacement that matches, and the `Matcher.replaceAll` emulation diverges from the JVM on `$10` / `\$` / empty matches; fixing `run` itself is a separate change, after which these natives land with `lite_parity` coverage. (`clojure.set` became Lite-eligible in Slice 3k without natives, by compiling its source into the program: §6.5.)

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

   A program that references any reader keeps every metadata key exactly as `run` builds it.
3. **Cheaper installs:** native symbols share their namespace's name and are qualified in one allocation; compiled var slots resolve through symbols built from their shared names.
4. **Smaller fixes:** `split-lines` takes the `\r?\n` byte cut (its source spelled the pattern with literal CR/LF, which compiled a full regex — ~275K instructions — on first use), and the static class lattice index is Fx-hashed and pre-sized.
5. **Batched installs:** boot installs each namespace's declared natives in one registry pass (`install_native_rows` → `intern_roots`: one lock, one namespace resolution, one mutation epoch per batch) with no per-var "already installed" probe (the compiler's list is distinct; only the internal helpers the program did not declare are probed). Hints carry an "owned-args entry exists" bit, so most installs skip the owned-table searches; a stale bit can only drop the owned fast path, never change results.
6. **`zipmap` and literal strings:** `zipmap` collects its pairs and builds a small result once (`Map::from_kvs_last_wins`, the `assoc` chain's exact map) instead of interning a key shape per pair — `(zipmap "0123456789" (range))` opens 238 of the 394 snippets — and continues with in-place inserts past 64 pairs; its forcing order (both heads, then both tails) is unchanged. Compiler-emitted strings decode through `str::from_utf8`'s fast path, with the lossy decoder as the fallback.
7. **Names with metadata pool without the reader.** A pooled constant the compiler cannot build directly is printed to text and read back with the full reader at startup (about 12K instructions per constant on macOS). Every `defn-` name is such a constant: it is a symbol carrying `{:private true}`, which Lite keeps because `refer` consults it. `^String` and `^:dynamic` names are too. A symbol whose metadata maps keywords to scalars, keywords, plain symbols, or strings is now a `Const::MetaSymbol`. The constant init writes its pairs into slots by name (`rt_n_kw` / `rt_n_sym` / `rt_n_str`, scalars inline; it cannot read other pool constants before the pool freezes). `rt_n_meta_sym` then inserts them pair by pair into an empty map, exactly as the reader merges `^{…}` (`native_abi::meta_symbol`, checked against the reader). Other metadata, such as an `:arglists` list when the program reads var metadata, still reads back. Locally, no_prefix_set (ten `defn-`s) saves 110K–210K instructions per run (corpus `meta_names.clj`).

### 6.5 Slice 3k: spliced libraries

The full boot costs ~14M instructions per run (Linux Pin), against OCaml's ~0.6M for a whole small case, so every program forced onto it loses all its cases. Four snippets took the full boot only because their leading `ns` form requires `clojure.set` (three call `union` / `intersection` / `difference`; one never calls it).

1. **Splice:** when a program's first unit is an `ns` form whose `:require` names a library in `aot_front::SPLICED_LIBS` (`clojure.set`), the front end reads, expands, and analyzes that library's vendored source (the same embedded text `run`'s `require` loads) and prepends its units, recorded as `FrontEnd::lib_units` / `FrontEnd::libs`. A `require` anywhere else (a later unit, a `(require …)` call) does not splice.
2. **Exact by construction:** the library's definitions compile like user code, so they behave exactly as `run`'s interpreted library: values, thrown classes and messages, non-set arguments, the variadic arities, and the one-value-per-form `#{}` of `union`'s zero arity. No native stands in for them, so contract 6.1's transcription rule does not apply. The i259 corpus program `clojure_set.clj` covers every `clojure.set` fn (`jit` ≡ `run`, the same module `build` links).
3. **Boot selection:** the Lite check accepts a `:require` spec naming a spliced library (the namespace's vars resolve to the spliced definitions); at runtime a Lite `ns` form only switches namespaces, as for `clojure.string`. A program that needs the full boot for another reason drops the spliced units and loads the interpreted library through its own `require`, exactly as before (corpus `clojure_set_full.clj`). The `aot: native N/N defns` report counts the program's own definitions only.
4. **Effect (macOS instructions, unchanged output):** ssa_dominance, dfa_minimization, hindley_milner, and ll1_parser each drop ~14M per case (e.g. ssa_dominance 37.1M → 22.7M, including macOS's ~20M process floor).
5. **Selective libraries (`clojure.string`):** a library whose vars are mostly Lite natives splices only what the program reaches that no native covers. The front end collects the library vars the program names without a native (`aot_boot::is_lite_string_var`), then keeps the top-level forms defining them and, transitively, the forms those reach. It also keeps the library's `ns` form with its clauses dropped: the build evaluator applied them, and a Lite `ns` form only switches namespaces. `TopUnit::form` groups each `defn`'s split units, so `def`, metadata write, and `var` stay together. If a needed var has no definition there, nothing is spliced. A program that looks vars up at runtime (`resolve`, `ns-resolve`, `ns-publics`, `ns-interns`, `ns-map`, `find-var`) splices nothing either, since a subset would show there. A `def` into `clojure.string` blocks Lite only when it redefines a native-served var, so spliced definitions keep it. knuth_div and cantor_expansion (`str/replace`) leave the full boot: macOS instructions −9M to −12M per case and RSS about −2 MB, with unchanged output (corpus `clojure_string_replace.clj`). The spliced `replace` reproduces `run` exactly, including #273's divergences from the JVM.

## 7. Acceptance Checklist (Issue #271)

- [ ] `h2h-compare` `--allow-extra-challenger` implemented and covered by Go tests.
- [ ] `.github/workflows/h2h-cpu-mem.yml` measures `AOT from`, `AOT to`, and `OCAML` and runs `aot-rsrs` and `aot-ocaml`.
- [ ] Slice 1 CI run completes cleanly and captures Linux Pin `AOT` vs `OCAML` baseline artifacts.
- [ ] Unwaived `h2h-cpu-mem.yml` passes `recipes`, `recipes-rsrs`, `aot-rsrs`, `aot-ocaml`, and `aot-run` with 0 MEM and 0 INS losses at `--margin 0.15`.
- [ ] All CI quality gates in `rust.yml` remain green at 100.00% coverage.
