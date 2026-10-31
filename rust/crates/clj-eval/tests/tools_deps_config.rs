// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-124 slice-1 tests: the tools.deps config layer — the
//! `clojure.tools.deps.util.port` substrate natives, the embedded
//! `util.dir` / `edn` / `extensions` namespaces, and the
//! `reduce-kv` drive-by.
//!
//! Oracles (text only, never executed): `org.clojure/tools.deps.edn`
//! 0.9.57 (`edn.clj`, `specs.clj`, `util/dir.clj`) and
//! `org.clojure/tools.deps` (`extensions.clj`,
//! `extensions/local.clj`); each test cites the upstream
//! file/behavior it pins. Divergences are the spec's A1-A5/D1/R1-R3
//! (`docs/gap-124/slice-1.md`).
//!
//! Env-mutating coverage lives in ONE serial test
//! (`user_config_dir_serial`) with save/restore — env vars are
//! process-global, so no other test here touches them.

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
        Err(other) => panic!("expected an uncaught throw, got {other:?}"),
        Ok(vs) => panic!("expected a throw, got values: {vs:?}"),
    }
}

/// A Rust string as a Clojure string literal.
fn clj_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Require the four slice namespaces with short aliases.
fn require_tools(eval: &mut Evaluator) {
    eval.eval_source(
        "(require '[clojure.tools.deps.util.port :as port]
                  '[clojure.tools.deps.util.dir :as dir]
                  '[clojure.tools.deps.edn :as edn]
                  '[clojure.tools.deps.extensions :as ext])",
    )
    .unwrap();
}

/// A fresh evaluator with the tools namespaces required.
fn tools_eval() -> Evaluator {
    let mut eval = boot();
    require_tools(&mut eval);
    eval
}

/// A scratch dir unique to this test (parallel-safe), wiped first.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("clj-td124-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write `name` under `dir`, returning the absolute path string.
fn write_dep(dir: &std::path::Path, name: &str, content: &str) -> String {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&path, content).unwrap();
    path.to_string_lossy().into_owned()
}

// ---- load (ns.rs resolve_source/load_one) ----

#[test]
fn tools_deps_nss_load_via_require() {
    // The loader serves the three embedded libs and installs the
    // native port ns (the spec_alpha.clj / install_async_ns shapes).
    let mut eval = boot();
    require_tools(&mut eval);
    for ns in [
        "clojure.tools.deps.util.port",
        "clojure.tools.deps.util.dir",
        "clojure.tools.deps.edn",
        "clojure.tools.deps.extensions",
    ] {
        assert_eq!(
            run(&mut eval, &format!("(some? (find-ns '{ns}))")),
            "true",
            "{ns} loads"
        );
        assert_eq!(
            run(&mut eval, &format!("(contains? (loaded-libs) '{ns})")),
            "true",
            "{ns} marks loaded"
        );
    }
    // The port natives resolve under their own ns only.
    assert_eq!(
        run(
            &mut eval,
            "(some? (ns-resolve 'clojure.tools.deps.util.port 'getenv))"
        ),
        "true"
    );
    assert_eq!(
        run(&mut eval, "(nil? (ns-resolve 'clojure.core 'getenv))"),
        "true"
    );
}

#[test]
fn port_ns_reload_is_noop() {
    // Reinstalling over itself is a `:reload` no-op (async shape).
    let mut eval = tools_eval();
    eval.eval_source("(require '[clojure.tools.deps.util.port] :reload)")
        .unwrap();
    assert_eq!(run(&mut eval, "(port/file-exists? \".\")"), "true");
    eval.eval_source("(require '[clojure.tools.deps.edn :as edn] :reload)")
        .unwrap();
    assert_eq!(run(&mut eval, "(edn/root-deps)"), "{}");
}

// ---- port substrate: read-edn-string (edn.clj read-edn) ----

#[test]
fn read_edn_string_rows() {
    // Upstream edn.clj `read-edn`: a single value reads; blank
    // (empty/whitespace/comments) answers nil.
    let mut eval = tools_eval();
    assert_eq!(
        run(&mut eval, "(port/read-edn-string \"{:a 1}\")"),
        "{:a 1}"
    );
    assert_eq!(run(&mut eval, "(port/read-edn-string \"42\")"), "42");
    assert_eq!(run(&mut eval, "(port/read-edn-string \"[1 2]\")"), "[1 2]");
    assert_eq!(run(&mut eval, "(port/read-edn-string \"\")"), "nil");
    assert_eq!(
        run(&mut eval, "(port/read-edn-string \"  \\n\\t \")"),
        "nil"
    );
    assert_eq!(
        run(
            &mut eval,
            "(port/read-edn-string \";; just a comment\\n;; another\")"
        ),
        "nil"
    );
    assert_eq!(
        run(
            &mut eval,
            "(port/read-edn-string \"  ;; c\\n {:a 1} ;; trailing\\n \")"
        ),
        "{:a 1}"
    );
    // Tagged literals resolve through the default readers.
    assert_eq!(
        run(
            &mut eval,
            "(port/read-edn-string \"#inst \\\"2020-01-01\\\"\")"
        ),
        "#inst \"2020-01-01T00:00:00.000-00:00\""
    );
}

#[test]
fn read_edn_string_multi_throws() {
    // Upstream edn.clj `read-edn`: a second value throws ex-info
    // naming the path (the inner "single value" text survives the
    // `io-err` wrap, so the port keeps it verbatim).
    let mut eval = tools_eval();
    let cwd = std::env::current_dir()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    // No path: `io-err` parity reports "unknown path".
    assert_eq!(
        uncaught(&mut eval, "(port/read-edn-string \"1 2\")"),
        "clojure.lang.ExceptionInfo | Expected edn to contain a single value. (unknown path)"
    );
    assert_eq!(
        run(
            &mut eval,
            "(try (port/read-edn-string \"1 2\") (catch clojure.lang.ExceptionInfo e (ex-data e)))"
        ),
        "{:path \"unknown path\"}"
    );
    // Explicit nil path reports the same.
    assert_eq!(run(&mut eval, "(port/read-edn-string \"{}\" nil)"), "{}");
    assert_eq!(
        uncaught(&mut eval, "(port/read-edn-string \"1 2\" nil)"),
        "clojure.lang.ExceptionInfo | Expected edn to contain a single value. (unknown path)"
    );
    // A second value hides behind no comments/whitespace (spec §5:
    // two `read_form` calls on one reader).
    let msg = uncaught(
        &mut eval,
        "(port/read-edn-string \"{:a 1} ;; note\\n ;; more\\n [2]\" \"/abs/deps.edn\")",
    );
    assert_eq!(
        msg,
        "clojure.lang.ExceptionInfo | Expected edn to contain a single value. (/abs/deps.edn)"
    );
    assert_eq!(
        run(
            &mut eval,
            "(try (port/read-edn-string \"1 2\" \"/abs/deps.edn\") (catch clojure.lang.ExceptionInfo e (ex-data e)))"
        ),
        "{:path \"/abs/deps.edn\"}"
    );
    // A relative path absolutizes `getAbsolutePath`-style (joined
    // onto the cwd, never normalized).
    let rel_msg = uncaught(&mut eval, "(port/read-edn-string \"1 2\" \"rel/x.edn\")");
    let abs = std::path::Path::new(&cwd)
        .join("rel/x.edn")
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        rel_msg,
        format!("clojure.lang.ExceptionInfo | Expected edn to contain a single value. ({abs})")
    );
}

#[test]
fn read_edn_string_bad_input_throws() {
    // Upstream edn.clj `read-edn`: reader failures (incl. unmatched
    // delimiters) throw ex-info naming the path; texts are port
    // reader wording (R3).
    let mut eval = tools_eval();
    let msg = uncaught(
        &mut eval,
        "(port/read-edn-string \"{:a 1\" \"/abs/deps.edn\")",
    );
    assert!(msg.starts_with("clojure.lang.ExceptionInfo | "), "{msg}");
    assert!(msg.ends_with("(/abs/deps.edn)"), "{msg}");
    // Trailing garbage after a first value throws too.
    let trailing = uncaught(
        &mut eval,
        "(port/read-edn-string \"{:a 1} (oops\" \"/abs/deps.edn\")",
    );
    assert!(trailing.ends_with("(/abs/deps.edn)"), "{trailing}");
    // `#=` never evaluates (clojure.edn parity — `read-edn` reads
    // through `clojure.edn`, never `clojure.core/read`): the gate
    // text lands in the ex-info instead of running the form.
    let eval_msg = uncaught(
        &mut eval,
        "(port/read-edn-string \"#=(+ 1 2)\" \"/abs/deps.edn\")",
    );
    assert_eq!(
        eval_msg,
        "clojure.lang.ExceptionInfo | EvalReader not allowed when *read-eval* is false. (/abs/deps.edn)"
    );
    // Read-conds are denied like `read-string`.
    let rc = uncaught(
        &mut eval,
        "(port/read-edn-string \"#?(:clj 1)\" \"/abs/deps.edn\")",
    );
    assert!(rc.ends_with("(/abs/deps.edn)"), "{rc}");
}

#[test]
fn read_edn_string_arg_errors() {
    // Port-canonical arg errors (read-string shapes).
    let mut eval = tools_eval();
    assert_eq!(
        uncaught(&mut eval, "(port/read-edn-string nil)"),
        "java.lang.NullPointerException | "
    );
    assert!(uncaught(&mut eval, "(port/read-edn-string 42)").contains("java.lang.String"));
    assert!(uncaught(&mut eval, "(port/read-edn-string \"{}\" 42)").contains("java.lang.String"));
    assert!(uncaught(&mut eval, "(port/read-edn-string)").contains("Wrong number of args"));
    assert!(
        uncaught(&mut eval, "(port/read-edn-string \"{}\" \"a\" \"b\")")
            .contains("Wrong number of args")
    );
}

// ---- port substrate: file-exists? / canonical-path / current-dir / getenv ----

#[test]
fn file_exists_rows() {
    // Upstream `File/exists` (edn.clj `read-deps` missing-file gate).
    let mut eval = tools_eval();
    let dir = scratch("file-exists");
    let file = write_dep(&dir, "x.edn", "{}");
    assert_eq!(
        run(
            &mut eval,
            &format!("(port/file-exists? {})", clj_str(&file))
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!("(port/file-exists? {})", clj_str(&dir.to_string_lossy()))
        ),
        "true"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(port/file-exists? {})",
                clj_str(&dir.join("nope.edn").to_string_lossy())
            )
        ),
        "false"
    );
    assert!(uncaught(&mut eval, "(port/file-exists? 42)").contains("java.lang.String"));
    assert!(uncaught(&mut eval, "(port/file-exists?)").contains("Wrong number of args"));
}

#[test]
fn canonical_path_rows() {
    // Upstream `File/getCanonicalFile` (dir.clj `canonicalize`),
    // spec A3: a missing leaf canonicalizes the existing prefix
    // and joins the remainder lexically.
    let mut eval = tools_eval();
    let dir = scratch("canonical-path");
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    let canon = |p: &std::path::Path| {
        std::fs::canonicalize(p)
            .unwrap()
            .to_string_lossy()
            .into_owned()
    };
    let dir_c = canon(&dir);
    // An existing path answers its canonical form (1- and 2-arity).
    assert_eq!(
        run(
            &mut eval,
            &format!("(port/canonical-path {})", clj_str(&dir.to_string_lossy()))
        ),
        clj_str(&dir_c)
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(port/canonical-path {} {})",
                clj_str(&dir.to_string_lossy()),
                clj_str(&sub.to_string_lossy())
            )
        ),
        clj_str(&canon(&sub))
    );
    // A missing leaf joins lexically onto the canonical prefix.
    let missing = dir.join("nope.edn");
    let missing_s = missing.to_string_lossy().into_owned();
    let expect = std::path::Path::new(&dir_c)
        .join("nope.edn")
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        run(
            &mut eval,
            &format!("(port/canonical-path {})", clj_str(&missing_s))
        ),
        clj_str(&expect)
    );
    // Missing nested tails join whole.
    let deep = dir.join("a").join("b").join("c.edn");
    let deep_expect = std::path::Path::new(&dir_c)
        .join("a")
        .join("b")
        .join("c.edn")
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        run(
            &mut eval,
            &format!("(port/canonical-path {})", clj_str(&deep.to_string_lossy()))
        ),
        clj_str(&deep_expect)
    );
    // `.` drops and `..` pops lexically (the JDK normalizes those
    // lexically too).
    let dotty = format!("{}/./sub/../nope.edn", dir.to_string_lossy());
    assert_eq!(
        run(
            &mut eval,
            &format!("(port/canonical-path {})", clj_str(&dotty))
        ),
        clj_str(&expect)
    );
    // A `.` inside the missing tail drops (during the walk — a
    // `..` there pops, `ghost/../sub` in the dir tests pins it).
    let dot_tail = std::path::Path::new(&dir_c)
        .join("ghost")
        .join("x")
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(port/canonical-path {} \"ghost/./x\")",
                clj_str(&dir.to_string_lossy())
            )
        ),
        clj_str(&dot_tail)
    );
    // A relative base resolves against the cwd (2-arity).
    let rel_base = run(&mut eval, "(port/canonical-path \"td124-relbase\" \"x\")");
    let cwd = std::env::current_dir().unwrap();
    let cwd_c = canon(&cwd);
    let rel_expect = std::path::Path::new(&cwd_c)
        .join("td124-relbase")
        .join("x")
        .to_string_lossy()
        .into_owned();
    assert_eq!(rel_base, clj_str(&rel_expect));
    // Arg errors (2-arity is `(base, path)`).
    assert!(uncaught(&mut eval, "(port/canonical-path 42)").contains("java.lang.String"));
    assert!(
        uncaught(
            &mut eval,
            &format!("(port/canonical-path {} 42)", clj_str(&dir_c))
        )
        .contains("java.lang.String")
    );
    assert!(uncaught(&mut eval, "(port/canonical-path 42 \"x\")").contains("java.lang.String"));
    assert!(uncaught(&mut eval, "(port/canonical-path)").contains("Wrong number of args"));
}

#[test]
fn current_dir_rows() {
    // Upstream `user.dir` (dir.clj `*the-dir*` default).
    let mut eval = tools_eval();
    let cwd = std::env::current_dir()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(run(&mut eval, "(port/current-dir)"), clj_str(&cwd));
    assert!(uncaught(&mut eval, "(port/current-dir 1)").contains("Wrong number of args"));
}

#[test]
fn getenv_rows() {
    // Upstream `System/getenv` (edn.clj `user-config-dir`).
    let mut eval = tools_eval();
    // SAFETY: a unique var name no other test touches.
    unsafe { std::env::set_var("CLJ_TD124_PROBE", "probe-value") };
    assert_eq!(
        run(&mut eval, "(port/getenv \"CLJ_TD124_PROBE\")"),
        "\"probe-value\""
    );
    unsafe { std::env::remove_var("CLJ_TD124_PROBE") };
    assert_eq!(run(&mut eval, "(port/getenv \"CLJ_TD124_PROBE\")"), "nil");
    assert_eq!(
        run(&mut eval, "(port/getenv \"CLJ_TD124_NEVER_SET\")"),
        "nil"
    );
    assert!(uncaught(&mut eval, "(port/getenv 42)").contains("java.lang.String"));
    assert!(uncaught(&mut eval, "(port/getenv)").contains("Wrong number of args"));
}

// ---- util.dir (util/dir.clj) ----

#[test]
fn the_dir_default_and_canonicalize() {
    // Upstream dir.clj: `*the-dir*` defaults to the working
    // directory; `canonicalize` resolves abs/relative against it.
    let mut eval = tools_eval();
    assert_eq!(
        run(&mut eval, "dir/*the-dir*"),
        run(&mut eval, "(port/current-dir)")
    );
    let dir = scratch("the-dir");
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    let dir_lit = clj_str(&dir.to_string_lossy());
    let canon_sub = std::fs::canonicalize(&sub)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    // Absolute passes through to canonical form.
    assert_eq!(
        run(
            &mut eval,
            &format!("(dir/with-dir {dir_lit} (dir/canonicalize {dir_lit}))")
        ),
        clj_str(&std::fs::canonicalize(&dir).unwrap().to_string_lossy())
    );
    // Relative resolves against the pushed context.
    assert_eq!(
        run(
            &mut eval,
            &format!("(dir/with-dir {dir_lit} (dir/canonicalize \"sub\"))")
        ),
        clj_str(&canon_sub)
    );
    // `..` against the context (and through a missing middle).
    let dir_c = std::fs::canonicalize(&dir)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {} (dir/canonicalize \"..\"))",
                clj_str(&canon_sub)
            )
        ),
        clj_str(&dir_c)
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {} (dir/canonicalize \"ghost/../sub\"))",
                clj_str(&dir.to_string_lossy())
            )
        ),
        clj_str(&canon_sub)
    );
}

#[test]
fn with_dir_nesting_and_restore() {
    // Upstream dir.clj `with-dir`: push/pop nesting over `binding`.
    let mut eval = tools_eval();
    let outer = scratch("with-dir-outer");
    let inner = outer.join("inner");
    std::fs::create_dir_all(&inner).unwrap();
    let outer_lit = clj_str(&outer.to_string_lossy());
    let outer_c = std::fs::canonicalize(&outer)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let inner_c = std::fs::canonicalize(&inner)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    // The body value returns; nesting restores level by level.
    let cwd = run(&mut eval, "dir/*the-dir*");
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(let [before dir/*the-dir* \
                       triple (dir/with-dir {outer_lit} \
                                (let [o dir/*the-dir* \
                                      i (dir/with-dir \"inner\" dir/*the-dir*) \
                                      back dir/*the-dir*] \
                                  [o i back])) \
                       after dir/*the-dir*] \
                   [before triple after])"
            )
        ),
        format!(
            "[{cwd} [{} {} {}] {cwd}]",
            clj_str(&outer_c),
            clj_str(&inner_c),
            clj_str(&outer_c)
        )
    );
    // Popping restores the entry context.
    let before = run(&mut eval, "dir/*the-dir*");
    run(&mut eval, &format!("(dir/with-dir {outer_lit} 1)"));
    assert_eq!(run(&mut eval, "dir/*the-dir*"), before);
}

#[test]
fn sub_path_rows() {
    // Upstream dir.clj `sub-path?`: true when the path has a
    // parent that is the context (canonical-string comparison,
    // spec A4); nil and missing contexts answer false.
    let mut eval = tools_eval();
    let root = scratch("sub-path");
    let child = root.join("child");
    let nested = child.join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    let sibling = scratch("sub-path-sib");
    let root_lit = clj_str(&root.to_string_lossy());
    let child_lit = clj_str(&child.to_string_lossy());
    let nested_lit = clj_str(&nested.to_string_lossy());
    let sib_lit = clj_str(&sibling.join("x").to_string_lossy());
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {root_lit} \
                   [(dir/sub-path? {child_lit}) \
                    (dir/sub-path? {nested_lit}) \
                    (dir/sub-path? {root_lit}) \
                    (dir/sub-path? \"child/nested\") \
                    (dir/sub-path? \"child/../child\") \
                    (dir/sub-path? {sib_lit}) \
                    (dir/sub-path? nil)])"
            )
        ),
        "[true true true true true false false]"
    );
    // A missing path still resolves under an existing context...
    assert_eq!(
        run(
            &mut eval,
            &format!("(dir/with-dir {root_lit} (dir/sub-path? \"ghost/deep\"))")
        ),
        "true"
    );
    // ...while a missing context answers false (upstream:
    // unreadable paths answer false).
    let ghost = root.join("ghost-ctx");
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {} (dir/sub-path? {}))",
                clj_str(&ghost.to_string_lossy()),
                child_lit
            )
        ),
        "false"
    );
}

// ---- BDD scenarios (spec §3) ----

#[test]
fn bdd_read_project_deps_edn() {
    // Scenario: read a project deps.edn (edn.clj `project-deps`).
    let mut eval = tools_eval();
    let proj = scratch("bdd-project");
    write_dep(
        &proj,
        "deps.edn",
        "{:paths [\"src\"] :deps {my/lib {:local/root \"../lib\"}}}",
    );
    let got = run(
        &mut eval,
        &format!(
            "(dir/with-dir {} (edn/project-deps))",
            clj_str(&proj.to_string_lossy())
        ),
    );
    assert_eq!(
        got,
        "{:paths [\"src\"], :deps #:my{lib #:local{:root \"../lib\"}}}"
    );
    // The project path helpers join under the context (canonical).
    let proj_c = std::fs::canonicalize(&proj)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {} (edn/project-deps-path))",
                clj_str(&proj.to_string_lossy())
            )
        ),
        clj_str(&format!("{proj_c}/deps.edn"))
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {} (edn/project-dir))",
                clj_str(&proj.to_string_lossy())
            )
        ),
        clj_str(&std::fs::canonicalize(&proj).unwrap().to_string_lossy())
    );
}

#[test]
fn bdd_missing_project_file_is_nil() {
    // Scenario: missing files are nil, not errors (edn.clj
    // `read-deps` `when exists` gate) — the project half (the user
    // half needs env control, so it lives in `user_config_dir_serial`).
    let mut eval = tools_eval();
    let proj = scratch("bdd-missing");
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {} (edn/project-deps))",
                clj_str(&proj.to_string_lossy())
            )
        ),
        "nil"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(edn/read-deps {})",
                clj_str(&proj.join("nope.edn").to_string_lossy())
            )
        ),
        "nil"
    );
}

#[test]
fn bdd_blank_deps_edn_is_nil() {
    // Scenario: blank deps.edn is nil (edn.clj `read-edn` blank
    // arm + nilable `::deps-map`).
    let mut eval = tools_eval();
    let proj = scratch("bdd-blank");
    let ws = write_dep(&proj, "ws.edn", "  \n\t ;; nothing here\n");
    let empty = write_dep(&proj, "empty.edn", "");
    for f in [ws, empty] {
        assert_eq!(
            run(&mut eval, &format!("(edn/read-deps {})", clj_str(&f))),
            "nil",
            "{f} reads nil"
        );
    }
    assert_eq!(run(&mut eval, "(edn/read-edn \"  ;; c\\n \")"), "nil");
    assert_eq!(run(&mut eval, "(edn/validate nil)"), "nil");
}

#[test]
fn bdd_trailing_second_value_rejected() {
    // Scenario: trailing second value is rejected (edn.clj
    // `read-edn` single-value throw, ex-info naming the path).
    let mut eval = tools_eval();
    let proj = scratch("bdd-multi");
    let file = write_dep(&proj, "deps.edn", "{:paths [\"src\"]}\n{:deps {}}\n");
    let lit = clj_str(&file);
    assert_eq!(
        uncaught(&mut eval, &format!("(edn/read-deps {lit})")),
        format!("clojure.lang.ExceptionInfo | Expected edn to contain a single value. ({file})")
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(try (edn/read-deps {lit}) (catch clojure.lang.ExceptionInfo e [(ex-message e) (ex-data e)]))"
            )
        ),
        format!(
            "[\"Expected edn to contain a single value. ({file})\" {{:path {}}}]",
            clj_str(&file)
        )
    );
}

#[test]
fn bdd_invalid_coords_rejected_with_path() {
    // Scenario: invalid coords are rejected with the path (edn.clj
    // `validate` + the hand-written specs.clj verdicts — A2).
    let mut eval = tools_eval();
    let proj = scratch("bdd-invalid");
    let file = write_dep(&proj, "deps.edn", "{:deps {my/lib {:local/root 42}}}");
    let lit = clj_str(&file);
    let msg = uncaught(&mut eval, &format!("(edn/read-deps {lit})"));
    assert_eq!(
        msg,
        format!(
            "clojure.lang.ExceptionInfo | Error validating deps in {file}: \
             invalid :local/root for my/lib: expected a string, found 42"
        )
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(try (edn/read-deps {lit}) (catch clojure.lang.ExceptionInfo e (ex-data e)))"
            )
        ),
        format!("{{:path {}}}", clj_str(&file))
    );
}

#[test]
fn bdd_simple_lib_canonicalizes_with_warning() {
    // Scenario: simple lib symbols canonicalize with a stderr
    // warning (edn.clj `canonicalize` + `printerrln`).
    let mut eval = tools_eval();
    let proj = scratch("bdd-canon");
    let file = write_dep(&proj, "deps.edn", "{:deps {mylib {:local/root \"lib\"}}}");
    let lit = clj_str(&file);
    assert_eq!(
        run(&mut eval, &format!("(edn/read-deps {lit})")),
        "{:deps #:mylib{mylib #:local{:root \"lib\"}}}"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!("(with-out-str (binding [*err* *out*] (edn/read-deps {lit})))")
        ),
        format!("\"DEPRECATED: Libs must be qualified, change mylib => mylib/mylib ({file})\\n\"")
    );
}

#[test]
fn bdd_config_chain_merges() {
    // Scenario: config chain merges root < user < project (edn.clj
    // `create-edn-maps` + `merge-edns`; nil sources skipped).
    let mut eval = tools_eval();
    assert_eq!(
        run(
            &mut eval,
            "(edn/create-edn-maps {:root nil :user {:paths [\"u\"]} :project {:paths [\"p\"]}})"
        ),
        "{:user {:paths [\"u\"]}, :project {:paths [\"p\"]}}"
    );
    assert_eq!(
        run(
            &mut eval,
            "(edn/merge-edns (vals (edn/create-edn-maps {:root nil :user {:paths [\"u\"]} :project {:paths [\"p\"]}})))"
        ),
        "{:paths [\"p\"]}"
    );
    // Nils drop out of the merge entirely.
    assert_eq!(run(&mut eval, "(edn/merge-edns [nil nil])"), "nil");
    assert_eq!(
        run(&mut eval, "(edn/merge-edns [nil {:paths [\"p\"]} nil])"),
        "{:paths [\"p\"]}"
    );
}

#[test]
fn bdd_aliases_combine() {
    // Scenario: aliases combine per the rule table (edn.clj
    // `combine-aliases` — extra-paths append-unique, jvm-opts
    // append).
    let mut eval = tools_eval();
    assert_eq!(
        run(
            &mut eval,
            "(edn/combine-aliases {:aliases {:a {:extra-paths [\"a\"] :jvm-opts [\"-X\"]} \
                                              :b {:extra-paths [\"a\" \"b\"] :main-opts [\"-m\"]}}} \
                                 [:a :b])"
        ),
        "{:extra-paths [\"a\" \"b\"], :jvm-opts [\"-X\"], :main-opts [\"-m\"]}"
    );
}

#[test]
fn bdd_local_lib_location() {
    // Scenario: :local lib-location resolves (extensions/local.clj
    // `lib-location` :local — faithful `{:base root :path ""
    // :type :local}`).
    let mut eval = tools_eval();
    assert_eq!(
        run(
            &mut eval,
            "(ext/lib-location 'my/lib {:local/root \"/abs/lib\"} nil)"
        ),
        "{:base \"/abs/lib\", :path \"\", :type :local}"
    );
}

#[test]
fn bdd_mvn_lib_location_refused() {
    // Scenario: :mvn lib-location is refused, not resolved
    // (redesign R1 — config parses, resolution throws naming the
    // lib with the source-only rule).
    let mut eval = tools_eval();
    let proj = scratch("bdd-mvn");
    let file = write_dep(
        &proj,
        "deps.edn",
        "{:deps {my/lib {:mvn/version \"1.2.3\"}}}",
    );
    assert_eq!(
        run(&mut eval, &format!("(edn/read-deps {})", clj_str(&file))),
        "{:deps #:my{lib #:mvn{:version \"1.2.3\"}}}"
    );
    assert_eq!(
        uncaught(
            &mut eval,
            "(ext/lib-location 'my/lib {:mvn/version \"1.2.3\"} nil)"
        ),
        "clojure.lang.ExceptionInfo | Source-only port: cannot resolve Maven coordinate for my/lib"
    );
    assert_eq!(
        run(
            &mut eval,
            "(try (ext/lib-location 'my/lib {:mvn/version \"1.2.3\"} nil) \
              (catch clojure.lang.ExceptionInfo e [(ex-message e) (ex-data e)]))"
        ),
        "[\"Source-only port: cannot resolve Maven coordinate for my/lib\" \
          {:lib my/lib, :coord #:mvn{:version \"1.2.3\"}}]"
    );
}

// ---- validate (specs.clj grammar verdicts — A2) ----

/// Assert `(edn/validate 'SRC)` returns its input unchanged (data
/// is quoted — bare lib symbols would read as var refs).
fn assert_valid(eval: &mut Evaluator, src: &str) {
    assert_eq!(
        run(eval, &format!("(edn/validate '{src})")),
        run(eval, &format!("'{src}")),
        "{src} validates"
    );
}

/// Assert `(edn/validate 'SRC :path "/p/deps.edn")` throws with the
/// exact port-canonical problem message.
fn assert_invalid(eval: &mut Evaluator, src: &str, problem: &str) {
    assert_eq!(
        uncaught(
            eval,
            &format!("(edn/validate '{src} :path \"/p/deps.edn\")")
        ),
        format!("clojure.lang.ExceptionInfo | Error validating deps in /p/deps.edn: {problem}"),
        "{src} rejects"
    );
}

#[test]
fn validate_accepts() {
    // Upstream specs.clj `valid-deps?` accept rows: nil (nilable
    // `::deps-map`), `{}`, every coord kind, paths/aliases, the
    // namespaced opts, open extra keys, and simple lib symbols
    // (which `canonicalize` fixes — validation must not reject).
    let mut eval = tools_eval();
    assert_valid(&mut eval, "nil");
    assert_valid(&mut eval, "{}");
    assert_valid(&mut eval, "{:paths [\"src\" \"test\"]}");
    assert_valid(&mut eval, "{:paths [\"src\" :extra-alias]}");
    assert_valid(&mut eval, "{:aliases {:a {:extra-paths [\"x\"]}}}");
    assert_valid(&mut eval, "{:aliases {:a [1 2]}}");
    assert_valid(&mut eval, "{:deps {my/lib {:local/root \"lib\"}}}");
    assert_valid(&mut eval, "{:deps {my/lib {:mvn/version \"1.2.3\"}}}");
    assert_valid(
        &mut eval,
        "{:deps {my/lib {:git/url \"https://x\" :git/sha \"abc\" :git/tag \"v1\"}}}",
    );
    assert_valid(&mut eval, "{:deps {my/lib {:git/url \"https://x\"}}}");
    assert_valid(&mut eval, "{:deps {my/lib {}}}");
    assert_valid(&mut eval, "{:deps {my/lib nil}}");
    assert_valid(&mut eval, "{:deps {my/lib {:deps/manifest :deps}}}");
    assert_valid(
        &mut eval,
        "{:deps {my/lib {:local/root \"l\" :deps/root \"r\"}}}",
    );
    assert_valid(
        &mut eval,
        "{:deps {my/lib {:mvn/version \"1\" :exclusions [other/lib]}}}",
    );
    assert_valid(&mut eval, "{:deps {mylib {:local/root \"lib\"}}}");
    assert_valid(
        &mut eval,
        "{:deps {my/lib {:local/root \"x\"}} :tool/extra 42}",
    );
    assert_valid(&mut eval, "{:mvn/local-repo \"/m2\"}");
    assert_valid(&mut eval, "{:mvn/repos {\"central\" {:url \"https://x\"}}}");
    assert_valid(&mut eval, "{:mvn/repos {\"c\" nil}}");
    assert_valid(
        &mut eval,
        "{:mvn/repos {\"c\" {:mvn/releases {:enabled true :update :daily :checksum :warn} \
                              :mvn/snapshots {:enabled false :update 60 :checksum :fail}}}}",
    );
    assert_valid(&mut eval, "{:tools/usage {:ns-default my-ns}}");
    assert_valid(&mut eval, "{:tools/usage {:ns-aliases {a b}}}");
    assert_valid(
        &mut eval,
        "{:deps/prep-lib {:ensure \"x\" :alias :a :fn my/fn}}",
    );
}

#[test]
fn validate_rejects_top_shapes() {
    // Non-map tops, mistyped sections, present-nil sections (absent
    // is fine — the keys are `:opt`), open alias values unchecked.
    let mut eval = tools_eval();
    assert_invalid(
        &mut eval,
        "[1 2]",
        "expected a deps.edn map or nil, found [1 2]",
    );
    assert_invalid(&mut eval, "42", "expected a deps.edn map or nil, found 42");
    assert_invalid(
        &mut eval,
        "\"x\"",
        "expected a deps.edn map or nil, found \"x\"",
    );
    assert_invalid(
        &mut eval,
        "{:paths (\"src\")}",
        "expected :paths to be a vector, found (\"src\")",
    );
    assert_invalid(
        &mut eval,
        "{:paths nil}",
        "expected :paths to be a vector, found nil",
    );
    assert_invalid(
        &mut eval,
        "{:paths [\"src\" 42]}",
        "expected :paths entries to be path strings or alias keywords, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps []}",
        "expected :deps to be a map of lib to coord, found []",
    );
    assert_invalid(
        &mut eval,
        "{:deps nil}",
        "expected :deps to be a map of lib to coord, found nil",
    );
    assert_invalid(
        &mut eval,
        "{:deps {:kw {:local/root \"x\"}}}",
        "expected :deps keys to be lib symbols, found :kw",
    );
    assert_invalid(
        &mut eval,
        "{:aliases []}",
        "expected :aliases to be a map of alias keyword to alias map, found []",
    );
    assert_invalid(
        &mut eval,
        "{:aliases {\"a\" {}}}",
        "expected :aliases keys to be keywords, found \"a\"",
    );
    // First problem wins in check order (:paths before :deps).
    assert_invalid(
        &mut eval,
        "{:paths [42] :deps {:kw {}}}",
        "expected :paths entries to be path strings or alias keywords, found 42",
    );
}

#[test]
fn validate_rejects_coords() {
    // Coord-kind keys must be well-typed when present (BDD-5 pins
    // `:local/root 42` rejected — the spec-letter `:git` catch-all
    // is narrowed); generic attrs keep spec verdicts.
    let mut eval = tools_eval();
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:local/root 42}}}",
        "invalid :local/root for my/lib: expected a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:mvn/version 42}}}",
        "invalid :mvn/version for my/lib: expected a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:git/url 42}}}",
        "invalid :git/url for my/lib: expected a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:git/sha 42}}}",
        "invalid :git/sha for my/lib: expected a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:git/tag 42}}}",
        "invalid :git/tag for my/lib: expected a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:local/root \"x\" :deps/root 42}}}",
        "invalid :deps/root for my/lib: expected a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:deps/manifest \"deps\"}}}",
        "invalid :deps/manifest for my/lib: expected a keyword, found \"deps\"",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:exclusions 42}}}",
        "invalid :exclusions for my/lib: expected a collection of lib symbols, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib {:exclusions [42]}}}",
        "invalid :exclusions for my/lib: expected lib symbols, found [42]",
    );
    assert_invalid(
        &mut eval,
        "{:deps {my/lib 42}}",
        "invalid coord for my/lib: expected a map, found 42",
    );
}

#[test]
fn validate_rejects_namespaced_opts() {
    // The `:opt` namespaced tops: `:mvn/repos`, `:mvn/local-repo`,
    // `:tools/usage`, `:deps/prep-lib` (required `:ensure`,
    // `:alias`, `:fn` — specs.clj `:req-un`).
    let mut eval = tools_eval();
    assert_invalid(
        &mut eval,
        "{:mvn/repos []}",
        "expected :mvn/repos to be a map of repo id to repo, found []",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/repos {42 {}}}",
        "expected :mvn/repos keys to be strings, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/repos {\"c\" 42}}",
        "expected mvn repo \"c\" to be a map or nil, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/repos {\"c\" {:url 42}}}",
        "expected mvn repo \"c\" :url to be a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/repos {\"c\" {:mvn/releases {:enabled \"x\"}}}}",
        "expected mvn repo \"c\" :mvn/releases :enabled to be boolean, found \"x\"",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/repos {\"c\" {:mvn/snapshots {:update :weekly}}}}",
        "expected mvn repo \"c\" :mvn/snapshots :update to be a policy or interval int, found :weekly",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/repos {\"c\" {:mvn/snapshots {:update 1.5}}}}",
        "expected mvn repo \"c\" :mvn/snapshots :update to be a policy or interval int, found 1.5",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/repos {\"c\" {:mvn/releases {:checksum :ok}}}}",
        "expected mvn repo \"c\" :mvn/releases :checksum to be :warn, :fail, or :ignore, found :ok",
    );
    assert_invalid(
        &mut eval,
        "{:mvn/local-repo 42}",
        "expected :mvn/local-repo to be a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:tools/usage []}",
        "expected :tools/usage to be a map, found []",
    );
    assert_invalid(
        &mut eval,
        "{:tools/usage {:ns-default foo/bar}}",
        "expected :tools/usage :ns-default to be a simple symbol, found foo/bar",
    );
    assert_invalid(
        &mut eval,
        "{:tools/usage {:ns-aliases {a 1}}}",
        "expected :ns-aliases entries to be simple symbol to simple symbol, found [a 1]",
    );
    assert_invalid(
        &mut eval,
        "{:deps/prep-lib []}",
        "expected :deps/prep-lib to be a map, found []",
    );
    assert_invalid(
        &mut eval,
        "{:deps/prep-lib {:alias :a :fn my/fn}}",
        "expected :deps/prep-lib to contain :ensure",
    );
    assert_invalid(
        &mut eval,
        "{:deps/prep-lib {:ensure 42 :alias :a :fn my/fn}}",
        "expected :deps/prep-lib :ensure to be a string, found 42",
    );
    assert_invalid(
        &mut eval,
        "{:deps/prep-lib {:ensure \"x\" :fn my/fn}}",
        "expected :deps/prep-lib to contain :alias",
    );
    assert_invalid(
        &mut eval,
        "{:deps/prep-lib {:ensure \"x\" :alias \"a\" :fn my/fn}}",
        "expected :deps/prep-lib :alias to be a keyword, found \"a\"",
    );
    assert_invalid(
        &mut eval,
        "{:deps/prep-lib {:ensure \"x\" :alias :a}}",
        "expected :deps/prep-lib to contain :fn",
    );
    assert_invalid(
        &mut eval,
        "{:deps/prep-lib {:ensure \"x\" :alias :a :fn \"f\"}}",
        "expected :deps/prep-lib :fn to be a symbol, found \"f\"",
    );
}

#[test]
fn validate_opts_forms() {
    // Opts arrive kwargs or as a single map (upstream's own
    // `read-deps` passes a map); a missing path reports
    // "unknown path" (`io-err` parity).
    let mut eval = tools_eval();
    assert_eq!(
        run(&mut eval, "(edn/validate {:paths [\"s\"]} :path \"/p\")"),
        "{:paths [\"s\"]}"
    );
    assert_eq!(
        run(&mut eval, "(edn/validate {:paths [\"s\"]} {:path \"/p\"})"),
        "{:paths [\"s\"]}"
    );
    assert_eq!(
        uncaught(&mut eval, "(edn/validate {:paths [42]})"),
        "clojure.lang.ExceptionInfo | Error validating deps in unknown path: \
         expected :paths entries to be path strings or alias keywords, found 42"
    );
    assert_eq!(
        run(
            &mut eval,
            "(try (edn/validate {:paths [42]}) (catch clojure.lang.ExceptionInfo e (ex-data e)))"
        ),
        "{:path \"unknown path\"}"
    );
    assert_eq!(
        run(&mut eval, "(edn/read-edn \"{:a 1}\" {:path \"/p\"})"),
        "{:a 1}"
    );
}

// ---- canonicalize (edn.clj canonicalize) ----

#[test]
fn canonicalize_rows() {
    // Upstream edn.clj `canonicalize`: simple lib syms qualify
    // (with the DEPRECATED stderr line) at every dep-map level;
    // qualified libs pass silent; the walk is maps/vectors only
    // (spec A5).
    let mut eval = tools_eval();
    // A qualified map canonicalizes silent.
    assert_eq!(
        run(
            &mut eval,
            "(edn/canonicalize '{:deps {my/lib {:local/root \"l\"}}})"
        ),
        "{:deps #:my{lib #:local{:root \"l\"}}}"
    );
    assert_eq!(
        run(
            &mut eval,
            "(with-out-str (binding [*err* *out*] (edn/canonicalize '{:deps {my/lib {:local/root \"l\"}}})))"
        ),
        "\"\""
    );
    // Simple libs qualify at top level and under aliases.
    assert_eq!(
        run(
            &mut eval,
            "(edn/canonicalize '{:deps {mylib {:local/root \"l\"}} \
                                 :aliases {:a {:extra-deps {other {:mvn/version \"1\"}}}}})"
        ),
        "{:deps #:mylib{mylib #:local{:root \"l\"}}, \
          :aliases {:a {:extra-deps #:other{other #:mvn{:version \"1\"}}}}}"
    );
    // Exclusions qualify too (mixed-ns coord maps print unlifted).
    assert_eq!(
        run(
            &mut eval,
            "(edn/canonicalize '{:deps {my/lib {:mvn/version \"1\" :exclusions [plain]}}})"
        ),
        "{:deps #:my{lib {:mvn/version \"1\", :exclusions [plain/plain]}}}"
    );
    // `:classpath-overrides` values are path strings, not coords —
    // the node fn still descends without crashing (destructuring a
    // string yields no exclusions).
    assert_eq!(
        run(
            &mut eval,
            "(edn/canonicalize '{:classpath-overrides {my/lib \"/o\"}})"
        ),
        "{:classpath-overrides #:my{lib \"/o\"}}"
    );
    // Warning without a path carries no parens (upstream trailing
    // empty-string shape).
    assert_eq!(
        run(
            &mut eval,
            "(with-out-str (binding [*err* *out*] (edn/canonicalize '{:deps {mylib {:local/root \"l\"}}})))"
        ),
        "\"DEPRECATED: Libs must be qualified, change mylib => mylib/mylib \\n\""
    );
    // Nil and scalars pass through (blank flow; postwalk parity).
    assert_eq!(run(&mut eval, "(edn/canonicalize nil)"), "nil");
    assert_eq!(run(&mut eval, "(edn/canonicalize 42)"), "42");
}

// ---- create-edn-maps / merge-edns (edn.clj) ----

#[test]
fn root_deps_is_empty() {
    // Redesign D1: the port root carries no defaults.
    let mut eval = tools_eval();
    assert_eq!(run(&mut eval, "(edn/root-deps)"), "{}");
}

#[test]
fn create_edn_maps_sources() {
    // Upstream edn.clj `create-edn-maps`/`choose-deps`: `:standard`
    // (root here — user/project standards need env control, see
    // `user_config_dir_serial`), string paths, nil omission, literal
    // maps, and the "Unexpected dep source" throw.
    let mut eval = tools_eval();
    let dir = scratch("edn-maps");
    let file = write_dep(&dir, "custom.edn", "{:paths [\"c\"]}");
    let lit = clj_str(&file);
    // `:standard` root is the embedded `{}` (user/project pinned
    // to nil — their standards read the machine's real config).
    assert_eq!(
        run(
            &mut eval,
            "(edn/create-edn-maps {:root :standard :user nil :project nil})"
        ),
        "{:root {}}"
    );
    // String sources read through the dir context.
    assert_eq!(
        run(
            &mut eval,
            &format!("(edn/create-edn-maps {{:root nil :user nil :project nil :extra {lit}}})")
        ),
        "{:extra {:paths [\"c\"]}}"
    );
    // String sources resolve relative to the context.
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(dir/with-dir {} (edn/create-edn-maps {{:root nil :user nil :project nil :extra \"custom.edn\"}}))",
                clj_str(&dir.to_string_lossy())
            )
        ),
        "{:extra {:paths [\"c\"]}}"
    );
    // Nil omits; missing files omit; maps pass through.
    assert_eq!(
        run(
            &mut eval,
            "(edn/create-edn-maps {:root nil :user nil :project nil})"
        ),
        "{}"
    );
    assert_eq!(
        run(
            &mut eval,
            &format!(
                "(edn/create-edn-maps {{:root nil :user nil :project nil :extra {}}})",
                clj_str(&dir.join("ghost.edn").to_string_lossy())
            )
        ),
        "{}"
    );
    assert_eq!(
        run(
            &mut eval,
            "(edn/create-edn-maps {:root nil :user nil :project nil :extra {:paths [\"e\"]}})"
        ),
        "{:extra {:paths [\"e\"]}}"
    );
    // Anything else throws naming the request.
    assert_eq!(
        uncaught(
            &mut eval,
            "(edn/create-edn-maps {:root nil :user nil :project nil :extra 42})"
        ),
        "clojure.lang.ExceptionInfo | Unexpected dep source: 42"
    );
    assert_eq!(
        run(
            &mut eval,
            "(try (edn/create-edn-maps {:root nil :user nil :project nil :extra 42}) \
              (catch clojure.lang.ExceptionInfo e (ex-data e)))"
        ),
        "{:requested 42}"
    );
}

#[test]
fn merge_edns_rows() {
    // Upstream edn.clj `merge-edns`/`merge-or-replace`: maps merge,
    // everything else replaces (rightmost non-nil wins); an
    // all-nil value merges to a nil-valued key.
    let mut eval = tools_eval();
    assert_eq!(
        run(
            &mut eval,
            "(edn/merge-edns [{:deps {'a/a {}}} {:deps {'b/b {}}}])"
        ),
        "{:deps {a/a {}, b/b {}}}"
    );
    assert_eq!(
        run(
            &mut eval,
            "(edn/merge-edns [{:paths [\"u\"]} {:paths [\"p\"]}])"
        ),
        "{:paths [\"p\"]}"
    );
    assert_eq!(
        run(&mut eval, "(edn/merge-edns [{:a nil} {:a {:x 1}}])"),
        "{:a {:x 1}}"
    );
    assert_eq!(
        run(&mut eval, "(edn/merge-edns [{:a {:x 1}} {:a nil}])"),
        "{:a {:x 1}}"
    );
    assert_eq!(
        run(&mut eval, "(edn/merge-edns [{:a nil} {:a nil}])"),
        "{:a nil}"
    );
    assert_eq!(run(&mut eval, "(edn/merge-edns [])"), "nil");
    assert_eq!(run(&mut eval, "(edn/merge-edns [{:a 1}])"), "{:a 1}");
    // Nested maps merge one level (merge, not deep-merge).
    assert_eq!(
        run(&mut eval, "(edn/merge-edns [{:a {:x 1 :y 1}} {:a {:y 2}}])"),
        "{:a {:x 1, :y 2}}"
    );
}

// ---- combine-aliases rule table (edn.clj) ----

/// Combine two alias maps under `:a`/`:b` and return the result.
fn combine(eval: &mut Evaluator, a: &str, b: &str) -> String {
    run(
        eval,
        &format!("(edn/combine-aliases '{{:aliases {{:a {a} :b {b}}}}} [:a :b])"),
    )
}

#[test]
fn combine_aliases_rules() {
    // Upstream edn.clj `merge-alias-rules`: last-wins, append,
    // append-unique, merge, merge-or-replace, plus the unknown-key
    // fallback (map merges, scalar replaces).
    let mut eval = tools_eval();
    // last-wins: rightmost non-nil (both nil stays nil).
    assert_eq!(
        combine(&mut eval, "{:main-opts [\"-a\"]}", "{:main-opts [\"-b\"]}"),
        "{:main-opts [\"-b\"]}"
    );
    assert_eq!(
        combine(&mut eval, "{:main-opts [\"-a\"]}", "{}"),
        "{:main-opts [\"-a\"]}"
    );
    assert_eq!(
        combine(&mut eval, "{}", "{:main-opts nil}"),
        "{:main-opts nil}"
    );
    assert_eq!(
        combine(&mut eval, "{:exec-fn a/f}", "{:exec-fn b/g}"),
        "{:exec-fn b/g}"
    );
    assert_eq!(
        combine(&mut eval, "{:ns-default a}", "{:ns-default b}"),
        "{:ns-default b}"
    );
    // append keeps duplicates.
    assert_eq!(
        combine(
            &mut eval,
            "{:jvm-opts [\"-X\" \"-Y\"]}",
            "{:jvm-opts [\"-X\"]}"
        ),
        "{:jvm-opts [\"-X\" \"-Y\" \"-X\"]}"
    );
    // append-unique dedups.
    assert_eq!(
        combine(
            &mut eval,
            "{:extra-paths [\"a\"]}",
            "{:extra-paths [\"a\" \"b\"]}"
        ),
        "{:extra-paths [\"a\" \"b\"]}"
    );
    assert_eq!(
        combine(&mut eval, "{:paths [\"a\"]}", "{:paths [\"a\" \"b\"]}"),
        "{:paths [\"a\" \"b\"]}"
    );
    assert_eq!(
        combine(
            &mut eval,
            "{:replace-paths [\"a\"]}",
            "{:replace-paths [\"b\"]}"
        ),
        "{:replace-paths [\"a\" \"b\"]}"
    );
    // merge for dep maps and classpath overrides.
    assert_eq!(
        combine(
            &mut eval,
            "{:extra-deps {a/a {}}}",
            "{:extra-deps {b/b {}}}"
        ),
        "{:extra-deps {a/a {}, b/b {}}}"
    );
    assert_eq!(
        combine(
            &mut eval,
            "{:replace-deps {a/a {}}}",
            "{:replace-deps {b/b {}}}"
        ),
        "{:replace-deps {a/a {}, b/b {}}}"
    );
    assert_eq!(
        combine(&mut eval, "{:deps {a/a {}}}", "{:deps {b/b {}}}"),
        "{:deps {a/a {}, b/b {}}}"
    );
    assert_eq!(
        combine(
            &mut eval,
            "{:override-deps {a/a {}}}",
            "{:override-deps {b/b {}}}"
        ),
        "{:override-deps {a/a {}, b/b {}}}"
    );
    assert_eq!(
        combine(
            &mut eval,
            "{:default-deps {a/a {}}}",
            "{:default-deps {b/b {}}}"
        ),
        "{:default-deps {a/a {}, b/b {}}}"
    );
    assert_eq!(
        combine(
            &mut eval,
            "{:classpath-overrides {a/a \"/o\"}}",
            "{:classpath-overrides {b/b \"/n\"}}"
        ),
        "{:classpath-overrides {a/a \"/o\", b/b \"/n\"}}"
    );
    assert_eq!(
        combine(&mut eval, "{:ns-aliases {a x}}", "{:ns-aliases {b y}}"),
        "{:ns-aliases {a x, b y}}"
    );
    // merge-or-replace: maps merge, scalars replace.
    assert_eq!(
        combine(&mut eval, "{:exec-args {:x 1}}", "{:exec-args {:y 2}}"),
        "{:exec-args {:x 1, :y 2}}"
    );
    assert_eq!(
        combine(&mut eval, "{:exec-args {:x 1}}", "{:exec-args 42}"),
        "{:exec-args 42}"
    );
    // Unknown keys: maps merge, scalars replace.
    assert_eq!(
        combine(&mut eval, "{:custom {:x 1}}", "{:custom {:y 2}}"),
        "{:custom {:x 1, :y 2}}"
    );
    assert_eq!(
        combine(&mut eval, "{:custom 1}", "{:custom 2}"),
        "{:custom 2}"
    );
    assert_eq!(
        combine(&mut eval, "{:custom [1]}", "{:custom [2]}"),
        "{:custom [2]}"
    );
    // Missing aliases merge as nil (nil maps contribute nothing).
    assert_eq!(
        run(&mut eval, "(edn/combine-aliases {:aliases {}} [:a])"),
        "{}"
    );
    assert_eq!(
        run(
            &mut eval,
            "(edn/combine-aliases {:aliases {:a {:x 1}}} [:a :ghost])"
        ),
        "{:x 1}"
    );
    assert_eq!(
        run(&mut eval, "(edn/combine-aliases {:aliases {:a {:x 1}}} [])"),
        "{}"
    );
}

// ---- extensions (extensions.clj + extensions/local.clj) ----

#[test]
fn git_lib_location_defers() {
    // The `:git` method is not-yet: it throws deferring to the git
    // slice (temporary — unlike the permanent `:mvn` refusal).
    let mut eval = tools_eval();
    for coord in [
        "{:git/url \"https://x\"}",
        "{:git/sha \"abc\"}",
        "{:git/tag \"v1\"}",
    ] {
        assert_eq!(
            uncaught(
                &mut eval,
                &format!("(ext/lib-location 'my/lib {coord} nil)")
            ),
            "clojure.lang.ExceptionInfo | Cannot resolve git coordinate for my/lib: deferred to the git slice",
            "{coord} defers"
        );
    }
    assert_eq!(
        run(
            &mut eval,
            "(try (ext/lib-location 'my/lib {:git/url \"https://x\"} nil) \
              (catch clojure.lang.ExceptionInfo e (ex-data e)))"
        ),
        "{:lib my/lib, :coord #:git{:url \"https://x\"}}"
    );
}

#[test]
fn lib_location_dispatch_rows() {
    // Dispatch keys on the coord kind (`:local/root` -> `:local`,
    // `:mvn/version` -> `:mvn`, git keys -> `:git`); mixed-kind
    // coords take the first match (pinned, port-canonical);
    // unknown/non-map coords throw `throw-bad-coord`-shaped
    // ex-infos (extensions.clj parity).
    let mut eval = tools_eval();
    // Mixed kinds: local wins over mvn and git.
    assert_eq!(
        run(
            &mut eval,
            "(ext/lib-location 'my/lib {:local/root \"/l\" :mvn/version \"1\"} nil)"
        ),
        "{:base \"/l\", :path \"\", :type :local}"
    );
    // Unknown map coords name the lib and coord.
    assert_eq!(
        uncaught(&mut eval, "(ext/lib-location 'my/lib {} nil)"),
        "clojure.lang.ExceptionInfo | Coord of unknown type for my/lib: {}"
    );
    assert_eq!(
        uncaught(
            &mut eval,
            "(ext/lib-location 'my/lib {:deps/manifest :deps} nil)"
        ),
        "clojure.lang.ExceptionInfo | Coord of unknown type for my/lib: #:deps{:manifest :deps}"
    );
    // Non-map coords (incl. nil) take the bad-coordinate shape.
    assert_eq!(
        uncaught(&mut eval, "(ext/lib-location 'my/lib 42 nil)"),
        "clojure.lang.ExceptionInfo | Bad coordinate for library my/lib, expected map: 42"
    );
    assert_eq!(
        uncaught(&mut eval, "(ext/lib-location 'my/lib nil nil)"),
        "clojure.lang.ExceptionInfo | Bad coordinate for library my/lib, expected map: nil"
    );
    assert_eq!(
        run(
            &mut eval,
            "(try (ext/lib-location 'my/lib 42 nil) \
              (catch clojure.lang.ExceptionInfo e [(ex-message e) (ex-data e)]))"
        ),
        "[\"Bad coordinate for library my/lib, expected map: 42\" {:lib my/lib, :coord 42}]"
    );
}

// ---- user config dir (edn.clj user-config-dir; SERIAL env) ----

/// Save an env var's current value for later restore.
fn save_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Restore an env var to a saved value (or unset it).
fn restore_env(name: &str, saved: Option<String>) {
    // SAFETY: single-threaded within this test; every mutated var
    // is restored before it returns.
    unsafe {
        match saved {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
}

/// Set an env var for one precedence row.
fn set_env(name: &str, val: &str) {
    // SAFETY: single-threaded within this test; restored on exit.
    unsafe { std::env::set_var(name, val) };
}

/// Unset an env var for one precedence row.
fn unset_env(name: &str) {
    // SAFETY: single-threaded within this test; restored on exit.
    unsafe { std::env::remove_var(name) };
}

#[test]
fn user_config_dir_serial() {
    // Upstream edn.clj `user-config-dir`: `CLJ_CONFIG` >
    // `XDG_CONFIG_HOME` > home. THE serial env test (spec §5): the
    // only test here that mutates process env, with save/restore —
    // it also hosts every `:standard` user/project behavior plus
    // the user half of BDD scenario 2 (missing user file is nil).
    // Home reads `$HOME`, else `%USERPROFILE%`, else the cwd
    // (port-canonical — no `user.home` property here).
    let saved_clj = save_env("CLJ_CONFIG");
    let saved_xdg = save_env("XDG_CONFIG_HOME");
    let saved_home = save_env("HOME");
    let saved_profile = save_env("USERPROFILE");
    let mut eval = tools_eval();
    // `CLJ_CONFIG` wins over everything.
    set_env("CLJ_CONFIG", "/cfg/clj");
    set_env("XDG_CONFIG_HOME", "/xdg");
    set_env("HOME", "/home/u");
    assert_eq!(run(&mut eval, "(edn/user-config-dir)"), "\"/cfg/clj\"");
    assert_eq!(
        run(&mut eval, "(edn/user-deps-path)"),
        "\"/cfg/clj/deps.edn\""
    );
    // `XDG_CONFIG_HOME` wins when `CLJ_CONFIG` is unset.
    unset_env("CLJ_CONFIG");
    assert_eq!(run(&mut eval, "(edn/user-config-dir)"), "\"/xdg/clojure\"");
    // `$HOME` wins when both are unset.
    unset_env("XDG_CONFIG_HOME");
    assert_eq!(
        run(&mut eval, "(edn/user-config-dir)"),
        "\"/home/u/.clojure\""
    );
    // `%USERPROFILE%` backs `$HOME` (Windows shape).
    unset_env("HOME");
    set_env("USERPROFILE", "C:\\Users\\u");
    assert_eq!(
        run(&mut eval, "(edn/user-config-dir)"),
        "\"C:\\\\Users\\\\u/.clojure\""
    );
    // The cwd backs both when neither is set.
    unset_env("USERPROFILE");
    let cwd = std::env::current_dir()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        run(&mut eval, "(edn/user-config-dir)"),
        clj_str(&format!("{cwd}/.clojure"))
    );
    // A user deps.edn reads through `CLJ_CONFIG`...
    let cfg = scratch("user-cfg");
    write_dep(&cfg, "deps.edn", "{:paths [\"u\"]}");
    set_env("CLJ_CONFIG", &cfg.to_string_lossy());
    assert_eq!(run(&mut eval, "(edn/user-deps)"), "{:paths [\"u\"]}");
    // ...and a missing user file is nil (BDD scenario 2, user half).
    let empty_cfg = scratch("user-cfg-empty");
    set_env("CLJ_CONFIG", &empty_cfg.to_string_lossy());
    assert_eq!(run(&mut eval, "(edn/user-deps)"), "nil");
    // `:standard` user/project resolve under the controlled env and
    // dir context (both create-edn-maps arities).
    let proj = scratch("user-cfg-proj");
    write_dep(&proj, "deps.edn", "{:paths [\"p\"]}");
    set_env("CLJ_CONFIG", &cfg.to_string_lossy());
    for call in ["(edn/create-edn-maps)", "(edn/create-edn-maps {})"] {
        assert_eq!(
            run(
                &mut eval,
                &format!("(dir/with-dir {} {call})", clj_str(&proj.to_string_lossy()))
            ),
            "{:root {}, :user {:paths [\"u\"]}, :project {:paths [\"p\"]}}",
            "{call} chains root < user < project"
        );
    }
    restore_env("CLJ_CONFIG", saved_clj.clone());
    restore_env("XDG_CONFIG_HOME", saved_xdg.clone());
    restore_env("HOME", saved_home.clone());
    restore_env("USERPROFILE", saved_profile.clone());
    // Restore verified: the saved values read back.
    assert_eq!(save_env("CLJ_CONFIG"), saved_clj);
    assert_eq!(save_env("XDG_CONFIG_HOME"), saved_xdg);
    assert_eq!(save_env("HOME"), saved_home);
    assert_eq!(save_env("USERPROFILE"), saved_profile);
}

// ---- reduce-kv drive-by (clojure.core) ----

#[test]
fn reduce_kv_rows() {
    // Upstream `clojure.core/reduce-kv` doc semantics (the port
    // already carries it — this pins value/ordering parity incl.
    // `reduced` short-circuit on both the vector and map paths).
    let mut eval = tools_eval();
    // Nil answers init without calling f.
    assert_eq!(
        run(&mut eval, "(reduce-kv (fn [a k v] (/ 1 0)) :init nil)"),
        ":init"
    );
    // Vectors reduce over ordinals, in order.
    assert_eq!(
        run(
            &mut eval,
            "(reduce-kv (fn [acc k v] (conj acc [k v])) [] [:a :b :c])"
        ),
        "[[0 :a] [1 :b] [2 :c]]"
    );
    // Maps reduce over entries, in seq order.
    assert_eq!(
        run(
            &mut eval,
            "(reduce-kv (fn [acc k v] (conj acc [k v])) [] {:a 1 :b 2})"
        ),
        "[[:a 1] [:b 2]]"
    );
    assert_eq!(
        run(&mut eval, "(reduce-kv (fn [acc k v] (conj acc k)) [] {})"),
        "[]"
    );
    // `reduced` short-circuits both paths (call counts pinned via
    // an atom: the vector run stops at index 1, the map run after
    // the first entry).
    assert_eq!(
        run(
            &mut eval,
            "(let [n (atom 0)] \
               [(reduce-kv (fn [acc k v] (swap! n inc) (if (= k 1) (reduced :stop) acc)) \
                            :go [:a :b :c :d]) \
                @n])"
        ),
        "[:stop 2]"
    );
    assert_eq!(
        run(
            &mut eval,
            "(let [n (atom 0)] \
               [(reduce-kv (fn [acc k v] (swap! n inc) (reduced :stop)) \
                            :go {:a 1 :b 2 :c 3}) \
                @n])"
        ),
        "[:stop 1]"
    );
    // init returns untouched on empty input without calling f.
    assert_eq!(
        run(&mut eval, "(reduce-kv (fn [a k v] (/ 1 0)) :init {})"),
        ":init"
    );
    assert_eq!(
        run(&mut eval, "(reduce-kv (fn [a k v] (/ 1 0)) :init [])"),
        ":init"
    );
}
