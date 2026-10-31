# Gap 124, slice 1 — tools.deps config layer (source-only redesign)

Upstream oracle (text only, never executed): `org.clojure/tools.deps.edn`
`0.9.57` (`clojure/tools/deps/edn.clj`, `specs.clj`, `util/dir.clj`) and
`org.clojure/tools.deps` `0.12.635` (`extensions/local.clj`), read from
Maven Central jars. No JDK/JVM anywhere in this slice.

No specification/TDD/BDD skills are installed in this workspace, so this
spec follows the standard spec-first shape: Problem / Solution / BDD
acceptance scenarios / TDD test plan, with every behavior pinned to an
upstream source line.

## 1. Problem

The port has no declarative dependency configuration. JVM Clojure resolves
`deps.edn` files through tools.deps; a source-compatible port must read,
validate, canonicalize, chain (root/user/project), merge, and alias-combine
the same `deps.edn` data — minus everything that downloads or consumes JVM
bytecode (decision: source-only redesign, issue #124).

## 2. Solution

Port the config layer as embedded Clojure namespaces plus a minimal
port-substrate ns (new Rust natives that upstream gets from `java.io` /
`System`, which the port deliberately lacks). Resolution/expansion comes in
later slices; this slice ends at `:local lib-location`.

### 2.1 Namespaces and functions

`clojure.tools.deps.util.dir` (from `util/dir.clj`):

| fn | upstream | port note |
|---|---|---|
| `*the-dir*` | dynamic, default `user.dir` | default `(current-dir)` substrate |
| `canonicalize` | `.getCanonicalFile` | substrate `canonical-path`, prefix-walk when the leaf is missing (A3) |
| `with-dir` | macro over `binding` | faithful (`binding` is native) |
| `sub-path?` | `Files/isSameFile` walk | canonical-string comparison (A4) |

`clojure.tools.deps.edn` (from `edn.clj`):

| fn | upstream | port note |
|---|---|---|
| `read-edn` | `PushbackReader`, single value, blank→nil, multi→throw | string-based `(read-edn s & opts)` over substrate `read-edn-string` (A1); `:path` opt kept |
| `validate` | `specs/valid-deps?` (spec.alpha `s/keys`/`s/or`/`s/merge`, not yet ported) | hand-written structural validator, same verdicts (A2) |
| `canonicalize` | postwalk + stderr deprecation for simple lib syms | scoped map/vector walk (A5); stderr warning faithful via `*err*` |
| `read-deps` | read→validate→canonicalize, missing file→nil | faithful over `file-exists?`+`slurp` |
| `root-deps` | classpath resource `clojure/tools/deps/deps.edn` | embedded `{}` (D1) |
| `user-config-dir` | `CLJ_CONFIG`/`XDG_CONFIG_HOME`/`~/.clojure` | faithful over substrate `getenv` |
| `user-deps-path`, `user-deps` | path join + read | faithful |
| `project-dir`, `project-deps-path`, `project-deps` | `*the-dir*` + read | faithful |
| `create-edn-maps` | `:standard`/string/nil/map sources | faithful, incl. "Unexpected dep source" error |
| `merge-edns` | `merge-with merge-or-replace`, nils dropped | faithful |
| `combine-aliases` | full rule table (last-wins/append/append-unique/merge/merge-or-replace, unknown map→merge, unknown scalar→replace) | faithful |

`clojure.tools.deps.extensions` (new ns shell; full ns is later slices):

| fn | note |
|---|---|
| `lib-location` defmulti | dispatch on coord kind (`:local/root`→`:local`, `:mvn/version`→`:mvn`, `:git/url`/sha/tag→`:git`) |
| `:local` method | faithful: `{:base root :path "" :type :local}` |
| `:mvn` method | **refusal** (redesign, permanent): ex-info naming the lib + "source-only" rationale (R1) |
| `:git` method | **not-yet**: ex-info "deferred to the git slice" (temporary) |

`clojure.tools.deps.util.port` (PORT SUBSTRATE, no upstream counterpart —
upstream gets this from the JDK; documented as such, never in `clojure.core`):

| native | behavior |
|---|---|
| `read-edn-string` | whole-string single-value EDN read: blank→nil, second value→throw, unmatched delimiter→throw; ex-info carries `:path` |
| `file-exists?` | path string → bool |
| `canonical-path` | absolute+normalized path string (A3) |
| `current-dir` | process cwd string |
| `getenv` | env var string or nil |

Drive-by core addition: `reduce-kv` (faithful upstream `clojure.core`
`reduce-kv`, incl. `reduced` short-circuit) — a real upstream core fn the
port lacks; the tools ns needs it. Local helpers inside the tools ns only:
`join-space` (`clojure.string` is unported), `canonicalize-walk` (A5),
`as->` stays (already a port macro).

### 2.2 Adaptation ledger (upstream → port, each test-pinned)

- A1 `read-edn` takes a string, not a `Reader`; callers `slurp` first.
  Error *shapes* match (ex-info + `:path`); message *texts* follow port
  reader wording, pinned as port-canonical.
- A2 `validate` is hand-written (spec.alpha lacks `s/keys`/`s/or`/`s/merge`).
  Same accept/reject verdicts over the grammar in `specs.clj`: lib symbols,
  `:mvn`/`:local`/`:git` coord shapes, `:deps`/`:paths`/alias maps,
  resolve/classpath args. `explain` text is port wording, pinned.
- A3 `canonical-path` on a missing leaf canonicalizes the existing prefix
  and joins the remainder lexically (Rust `fs::canonicalize` requires
  existence; the JDK does not).
- A4 `sub-path?` compares canonical path strings instead of inode identity
  (`Files/isSameFile`); differs only across symlinked/hardlinked prefixes.
  Unreadable paths → false, as upstream.
- A5 `canonicalize-walk` recurses maps/vectors only (≡ `postwalk` on valid
  `deps.edn`: the grammar has no maps under other nodes; `read-deps`
  validates before canonicalizing, so invalid shapes never reach it).

### 2.3 Design decisions and refusals

- D1 `root-deps` is the embedded value `{}`. Upstream root ships JVM
  defaults (a `org.clojure/clojure` dep, stock aliases); the port binary IS
  the runtime, so the port root carries no defaults. Revisitable if a later
  slice needs root defaults.
- R1 `:mvn` coords are ACCEPTED by `read-deps`/`validate` (config
  compatibility: stock `deps.edn` files must parse) and REFUSED at
  `lib-location` with an ex-info that names the lib and states the
  source-only rule. Permanent.
- R2 `:local/root` pointing at a jar file is refused at manifest detection
  (a later slice; noted here because upstream assigns `:jar` manifests to
  file roots). Permanent.
- R3 Error message texts that quote the JVM reader/spec machinery are port
  wording, pinned as port-canonical; error *kinds* (ex-info keys) match.

## 3. BDD acceptance scenarios

```gherkin
Feature: deps.edn config layer

  Scenario: read a project deps.edn
    Given a project dir with deps.edn {:paths ["src"] :deps {my/lib {:local/root "../lib"}}}
    When the client calls project-deps inside (with-dir project-dir)
    Then it returns the validated, canonicalized map

  Scenario: missing files are nil, not errors
    Given no deps.edn at the user or project path
    When the client calls user-deps and project-deps
    Then both return nil

  Scenario: blank deps.edn is nil
    Given a deps.edn containing only whitespace/comments
    When the client calls read-deps on it
    Then it returns nil

  Scenario: trailing second value is rejected
    Given a deps.edn with two top-level forms
    When the client calls read-deps on it
    Then it throws ex-info naming the path

  Scenario: invalid coords are rejected with the path
    Given a deps.edn {:deps {my/lib {:local/root 42}}}
    When the client calls read-deps on it
    Then it throws ex-info whose message names the path

  Scenario: simple lib symbols canonicalize with a stderr warning
    Given a deps.edn {:deps {mylib {:local/root "lib"}}}
    When the client calls read-deps on it
    Then the lib becomes mylib/mylib and *err* carries the DEPRECATED line

  Scenario: config chain merges root < user < project
    Given user deps {:paths ["u"]} and project deps {:paths ["p"]}
    When the client merges (create-edn-maps {}) with merge-edns
    Then :paths is ["p"] and nil sources are skipped

  Scenario: aliases combine per the rule table
    Given an alias {:extra-paths ["a"] :jvm-opts ["-X"]} and an alias {:extra-paths ["a" "b"] :main-opts ["-m"]}
    When the client calls combine-aliases on both
    Then extra-paths append-unique and jvm-opts append

  Scenario: :local lib-location resolves
    Given coord {:local/root "/abs/lib"}
    When the client calls lib-location
    Then it returns {:base "/abs/lib" :path "" :type :local}

  Scenario: :mvn lib-location is refused, not resolved
    Given a deps.edn with {:mvn/version "1.2.3"} (which reads fine)
    When the client calls lib-location on it
    Then it throws ex-info naming the lib and the source-only rule
```

## 4. TDD test plan

- `rust/crates/clj-eval/tests/tools_deps_config.rs` (new): one test per
  BDD scenario plus edge rows — unmatched delimiter; non-map top level;
  `create-edn-maps` with `:standard`/string/nil/map sources incl. the
  "Unexpected dep source" throw; `merge-edns` nil-skipping and
  replace-vs-merge per value kind; every `combine-aliases` rule row
  (last-wins/append/append-unique/merge/merge-or-replace/unknown-key);
  `with-dir` nesting/restore; `sub-path?` true/false/nil; `user-config-dir`
  precedence (`CLJ_CONFIG` > `XDG_CONFIG_HOME` > home; serial env
  save/restore in Rust); `:mvn` refusal data; `:git` not-yet data.
- `reduce-kv` tests: value/ordering parity incl. `reduced`, ported against
  upstream `clojure/core.clj` `reduce-kv` doc semantics.
- Oracle pins: each test cites the upstream file/line behavior it pins
  (edn.clj `read-edn`/`validate`/`canonicalize`/`choose-deps`/rule table,
  dir.clj `with-dir`/`sub-path?`, local.clj `lib-location`).
- Gates: `cargo fmt --check`, `cargo clippy`, full `clj-eval` suite, llvm-cov
  floor holds (new natives + embedded clj fully covered), complexity ratchet
  holds (validator fns split if needed).

## 5. Risks

- Env-mutating tests are process-global: keep them in one serial test (or a
  serial module) with save/restore; never run `user-config-dir` tests in
  parallel with other env readers.
- `read-edn-string` must reject a trailing second value even when separated
  by comments/whitespace: implement as two `read_form` calls on one reader,
  not two `read-string` calls.
