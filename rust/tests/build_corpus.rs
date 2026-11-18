// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S8D corpus replay: every `docs/spec/t1174/s8-corpus` file plus every
//! `docs/spec/i214-jit-subset/corpus` file, differentially `run` vs
//! `build`+execute vs `jit` (rules in each `MANIFEST.md`).
//! Supported files must print byte-identical stdout on all paths
//! (exit 0, empty stderr); error programs pin the abort stderr
//! (true call-site spans, exit 1) on the `build` AND `jit` paths;
//! unsupported files pin the exact plan diagnostic (exit 1) on
//! both native paths while `run` still works (exit 0).

#![cfg(feature = "aot")]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::Command;

/// A `Command` for the `clojure-rs` binary under test.
fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_clojure-rs"))
}

/// The corpus dir (absolute, canonicalized — diagnostics embed the
/// exact path passed to `build`).
fn corpus_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/spec/t1174/s8-corpus");
    dir.canonicalize().expect("corpus dir")
}

/// The #214 subset-expansion corpus dir (same replay contract as
/// [`corpus_dir`]; lives outside `docs/spec/t1174`, which is frozen).
fn corpus_i214_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/spec/i214-jit-subset/corpus");
    dir.canonicalize().expect("i214 corpus dir")
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("clj-rs-corpus-{}-{name}", std::process::id()))
}

/// The active Cargo profile dir (e.g. `<target>/debug`), supporting
/// both legacy `<target>/<profile>/deps/<bin>` and Cargo >=1.100
/// `<target>/<profile>/build/<pkg>/<hash>/out/<bin>`.
fn target_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current exe");
    let bin_dir = exe.parent().expect("test binary parent");
    if bin_dir.file_name() == Some(std::ffi::OsStr::new("deps")) {
        return bin_dir.parent().expect("profile").to_path_buf();
    }
    if let Some(profile) = bin_dir
        .file_name()
        .filter(|n| *n == "out")
        .and_then(|_| bin_dir.parent())
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .filter(|p| p.file_name() == Some(std::ffi::OsStr::new("build")))
        .and_then(|b| b.parent())
    {
        return profile.to_path_buf();
    }
    bin_dir
        .ancestors()
        .find(|p| p.join(".cargo-lock").is_file())
        .map(PathBuf::from)
        .unwrap_or_else(|| bin_dir.parent().expect("profile").to_path_buf())
}

/// Ensure the static `clj-rt` archive exists (twin of the
/// `clj-aot/tests/build_exe.rs` helper: plain `cargo test` only
/// builds the rlib, so build the staticlib on demand in an isolated
/// `cljrt-archive` target dir with coverage instrumentation
/// stripped — sharing the test target dir would reuse instrumented
/// dependency objects, whose profile-runtime symbols no plain `cc`
/// link can resolve).
fn ensure_archive() -> PathBuf {
    static ARCHIVE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ARCHIVE.get_or_init(build_archive).clone()
}

struct DirLock(PathBuf);

impl Drop for DirLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

fn acquire_lock(lock_dir: &std::path::Path) -> DirLock {
    let stale = std::time::Duration::from_secs(600);
    let wait = std::time::Duration::from_millis(25);
    // Windows refuses to create a directory whose previous instance is
    // still pending deletion ("Access is denied", os error 5): a lock
    // just released by another test process. That is a held lock, not a
    // failure; a denial lasting this long is a real permission problem.
    let denied_limit = std::time::Duration::from_secs(120);
    let mut denied_since = None;
    loop {
        match std::fs::create_dir(lock_dir) {
            Ok(()) => return DirLock(lock_dir.to_path_buf()),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                let expired = std::fs::metadata(lock_dir)
                    .and_then(|m| m.modified())
                    .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > stale));
                if expired {
                    let _ = std::fs::remove_dir(lock_dir);
                    continue;
                }
                std::thread::sleep(wait);
            }
            Err(err)
                if err.kind() == std::io::ErrorKind::PermissionDenied
                    && denied_since
                        .get_or_insert_with(std::time::Instant::now)
                        .elapsed()
                        < denied_limit =>
            {
                std::thread::sleep(wait);
            }
            Err(err) => panic!("create lock {}: {err}", lock_dir.display()),
        }
    }
}

fn prune_old_published(pub_root: &std::path::Path) {
    let max_age = std::time::Duration::from_secs(3600);
    let Ok(entries) = std::fs::read_dir(pub_root) else {
        return;
    };
    for entry in entries.flatten() {
        let expired = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > max_age));
        if expired {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

fn build_archive() -> PathBuf {
    let dir = target_dir();
    let name = if cfg!(windows) {
        "cljrt.lib"
    } else {
        "libcljrt.a"
    };
    let tops = dir.parent().expect("target root").join("cljrt-archive");
    let run_key = std::env::var("NEXTEST_RUN_ID")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("pid-{}", std::process::id()));
    let pub_root = tops.join("published");
    let pub_dir = pub_root.join(&run_key);
    let published = pub_dir.join(name);
    if published.is_file() {
        return published;
    }
    std::fs::create_dir_all(&tops).expect("create cljrt-archive");
    let _guard = acquire_lock(&tops.join(".archive-lock"));
    if published.is_file() {
        return published;
    }
    let archive = tops.join("debug").join(name);
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--locked", "-p", "clj-rt", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&tops);
    cmd.env_remove("RUSTC_WRAPPER")
        .env_remove("LLVM_PROFILE_FILE");
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().contains("LLVM_COV") {
            cmd.env_remove(key);
        }
    }
    let built = cmd.status().expect("run cargo build");
    assert!(built.success(), "cargo build -p clj-rt failed");
    assert!(archive.is_file(), "missing {}", archive.display());
    std::fs::create_dir_all(&pub_dir).expect("create published dir");
    let staged = pub_dir.join(format!(".tmp-{}-{name}", std::process::id()));
    std::fs::copy(&archive, &staged).expect("stage runtime archive");
    std::fs::rename(&staged, &published).expect("publish runtime archive");
    prune_old_published(&pub_root);
    published
}

fn exe_suffix() -> &'static str {
    if cfg!(windows) { ".exe" } else { "" }
}

/// `run` a corpus file: (stdout bytes, stderr bytes, exit code).
fn run_file(path: &std::path::Path) -> (Vec<u8>, Vec<u8>, i32) {
    let out = bin()
        .arg("run")
        .arg(path)
        .output()
        .expect("spawn clojure-rs run");
    (
        out.stdout,
        out.stderr,
        out.status.code().expect("exit code"),
    )
}

/// `build` a corpus file, then run the binary:
/// (build code, build stderr, exe stdout, exe stderr, exe code).
fn build_and_run(name: &str, path: &std::path::Path) -> (i32, Vec<u8>, Vec<u8>, Vec<u8>, i32) {
    build_and_run_with(name, path, &[])
}

/// `build_and_run` with argv for the binary (T1192 FX1 D-argv).
fn build_and_run_with(
    name: &str,
    path: &std::path::Path,
    args: &[&str],
) -> (i32, Vec<u8>, Vec<u8>, Vec<u8>, i32) {
    build_and_run_flags(name, path, args, &[])
}

/// `build_and_run_with` plus extra `build` flags (`--strict`).
fn build_and_run_flags(
    name: &str,
    path: &std::path::Path,
    args: &[&str],
    flags: &[&str],
) -> (i32, Vec<u8>, Vec<u8>, Vec<u8>, i32) {
    let dir = tmp(name);
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let prog = dir.join(format!("prog{}", exe_suffix()));
    let archive = ensure_archive();
    let built = bin()
        .args([
            OsStr::new("build"),
            path.as_os_str(),
            OsStr::new("-o"),
            prog.as_os_str(),
            OsStr::new("--runtime-archive"),
            archive.as_os_str(),
        ])
        .args(flags)
        .output()
        .expect("spawn clojure-rs build");
    let build_code = built.status.code().expect("build exit code");
    let build_err = built.stderr;
    let (exe_out, exe_err, exe_code) = if build_code == 0 {
        let ran = Command::new(&prog).args(args).output().expect("run binary");
        (
            ran.stdout,
            ran.stderr,
            ran.status.code().expect("exe exit code"),
        )
    } else {
        (Vec::new(), Vec::new(), -1)
    };
    std::fs::remove_dir_all(&dir).expect("cleanup");
    (build_code, build_err, exe_out, exe_err, exe_code)
}

/// `jit` a corpus file: (stdout bytes, stderr bytes, exit code).
fn jit_file(path: &std::path::Path, args: &[&str]) -> (Vec<u8>, Vec<u8>, i32) {
    let mut cmd = bin();
    cmd.arg("jit").arg(path);
    for arg in args {
        cmd.arg(arg);
    }
    let out = cmd.output().expect("spawn clojure-rs jit");
    (
        out.stdout,
        out.stderr,
        out.status.code().expect("exit code"),
    )
}

#[test]
fn supported_files_match_run_byte_for_byte() {
    // 00-05: both paths exit 0, empty stderr, stdout == the
    // `run`-seeded golden (which itself matches the JVM golden).
    // `11-loop`/`12-when` replay from the i214 corpus instead
    // (`i214_supported_files_match`).
    let corpus = corpus_dir();
    let files = [
        "00-empty",
        "01-arith",
        "02-recursion",
        "03-types",
        "04-let-do-if",
        "05-print",
    ];
    for stem in files {
        let path = corpus.join(format!("{stem}.clj"));
        let golden = std::fs::read(corpus.join(format!("expected/{stem}.out"))).expect("golden");
        let (run_out, run_err, run_code) = run_file(&path);
        assert_eq!(run_code, 0, "{stem}: run exit");
        assert!(run_err.is_empty(), "{stem}: run stderr: {run_err:?}");
        assert_eq!(run_out, golden, "{stem}: run stdout vs golden");
        let (build_code, build_err, exe_out, exe_err, exe_code) = build_and_run(stem, &path);
        assert_eq!(build_code, 0, "{stem}: build exit, stderr: {build_err:?}");
        assert!(build_err.is_empty(), "{stem}: build stderr: {build_err:?}");
        assert_eq!(exe_code, 0, "{stem}: exe exit");
        assert!(exe_err.is_empty(), "{stem}: exe stderr: {exe_err:?}");
        assert_eq!(exe_out, golden, "{stem}: exe stdout vs golden");
        let (jit_out, jit_err, jit_code) = jit_file(&path, &[]);
        assert_eq!(jit_code, 0, "{stem}: jit exit");
        assert!(jit_err.is_empty(), "{stem}: jit stderr: {jit_err:?}");
        assert_eq!(jit_out, golden, "{stem}: jit stdout vs golden");
    }
}

#[test]
fn dash_main_receives_argv_after_tops() {
    // 10-argv (T1192 FX1 D-argv): `run` never invokes `-main`
    // (script mode); the binary runs the tops, then `-main`
    // with nil (no args) or the argv list.
    let corpus = corpus_dir();
    let path = corpus.join("10-argv.clj");
    let (run_out, run_err, run_code) = run_file(&path);
    assert_eq!(run_code, 0, "run exit");
    assert!(run_err.is_empty(), "run stderr: {run_err:?}");
    assert_eq!(
        run_out,
        std::fs::read(corpus.join("expected/10-argv.runout")).expect("runout golden"),
        "run stdout"
    );
    for (name, args, golden) in [
        ("argv-noargs", &[][..], "expected/10-argv.exe-noargs.out"),
        (
            "argv-args",
            &["a", "b"][..],
            "expected/10-argv.exe-args.out",
        ),
    ] {
        let (build_code, build_err, exe_out, exe_err, exe_code) =
            build_and_run_with(name, &path, args);
        assert_eq!(build_code, 0, "{name}: build exit, stderr: {build_err:?}");
        assert!(build_err.is_empty(), "{name}: build stderr: {build_err:?}");
        assert_eq!(exe_code, 0, "{name}: exe exit");
        assert!(exe_err.is_empty(), "{name}: exe stderr: {exe_err:?}");
        assert_eq!(
            exe_out,
            std::fs::read(corpus.join(golden)).expect("exe golden"),
            "{name}: exe stdout"
        );
        let (jit_out, jit_err, jit_code) = jit_file(&path, args);
        assert_eq!(jit_code, 0, "{name}: jit exit");
        assert!(jit_err.is_empty(), "{name}: jit stderr: {jit_err:?}");
        assert_eq!(
            jit_out,
            std::fs::read(corpus.join(golden)).expect("exe golden"),
            "{name}: jit stdout"
        );
    }
}

#[test]
fn error_programs_pin_the_aot_abort() {
    // 06-09: `run` reports `{path}:1:1:` (synthesized span);
    // the AOT binary reports the TRUE call-site span (line 2,
    // col 8 in each file) — class/message match the oracle.
    let corpus = corpus_dir();
    let cases: &[(&str, &str, &str)] = &[
        ("06-err-quot0", "java.lang.ArithmeticException", "/ by zero"),
        ("07-err-nth", "java.lang.IndexOutOfBoundsException", ""),
        (
            "08-err-count",
            "java.lang.UnsupportedOperationException",
            "count not supported on this type: Long",
        ),
        (
            "09-err-arity",
            "clojure.lang.ArityException",
            "Wrong number of args (0) passed to: clojure.core/-",
        ),
    ];
    for (stem, class, message) in cases {
        let path = corpus.join(format!("{stem}.clj"));
        let golden = std::fs::read(corpus.join(format!("expected/{stem}.out"))).expect("golden");
        let runerr =
            std::fs::read(corpus.join(format!("expected/{stem}.runerr"))).expect("runerr golden");
        let (run_out, run_err, run_code) = run_file(&path);
        assert_eq!(run_code, 1, "{stem}: run exit");
        assert_eq!(run_out, golden, "{stem}: run stdout vs golden");
        // The `.runerr` golden pins the class/message after its
        // relative-path prefix; the replayed stderr carries the
        // same text after the absolute replay path (split at the
        // span, never at `/` — Windows paths are backslashed).
        let golden_text = lossy(&runerr);
        let (_, run_message) = golden_text
            .split_once(":1:1: ")
            .expect("runerr golden shape");
        let want = format!("{}:1:1: {run_message}", path.display());
        assert_eq!(lossy(&run_err), want, "{stem}: run stderr");
        let (build_code, build_err, exe_out, exe_err, exe_code) = build_and_run(stem, &path);
        assert_eq!(build_code, 0, "{stem}: build exit, stderr: {build_err:?}");
        assert_eq!(exe_code, 1, "{stem}: exe exit");
        assert_eq!(exe_out, golden, "{stem}: exe stdout vs golden");
        let want = format!("{}:2:8: {class}: {message}\n", path.display());
        assert_eq!(lossy(&exe_err), want, "{stem}: exe stderr");
        let (jit_out, jit_err, jit_code) = jit_file(&path, &[]);
        assert_eq!(jit_code, 1, "{stem}: jit exit");
        assert_eq!(jit_out, golden, "{stem}: jit stdout vs golden");
        assert_eq!(lossy(&jit_err), want, "{stem}: jit stderr");
    }
}

#[test]
fn unsupported_files_fail_build_but_run() {
    // 01-16: `run` still works (exit 0, `.runout` stdout, empty
    // stderr — the BDD clause); `build` exits 1 naming the first
    // unsupported form: `{path}:{line}:{col}: unsupported form in
    // AOT subset: {form} ({hint})`. Forms/hints are the S8A
    // rejection-table contract; spans read off the sources.
    let corpus = corpus_dir();
    // (`02-loop`/`10-when` graduated out of this table: plain
    // loops and `when` compile now. Their t1174 fixtures stay
    // byte-frozen (t1174-empty-diff rule) but unreferenced; the
    // live loop/when corpus — `11-loop`, `12-when`, and the
    // still-unsupported destructuring `02-loop` — replays from
    // the i214 corpus instead.)
    let cases: &[(&str, u32, u32, &str, &str)] = &[
        ("01-def", 1, 1, "def", "only defn defines in the AOT subset"),
        (
            "03-recur",
            1,
            27,
            "recur",
            "recur outside loop is outside the subset; call the function by name",
        ),
        (
            "04-try",
            1,
            8,
            "try",
            "error handling is outside the AOT subset",
        ),
        ("05-var", 1, 8, "var", "vars are outside the AOT subset"),
        (
            "06-future",
            1,
            1,
            "clojure.core/deref",
            "concurrency is outside the AOT subset",
        ),
        ("07-lazy-seq", 1, 8, "take", "not in the closed call set"),
        (
            "08-println",
            1,
            1,
            "println",
            "only print writes in the AOT subset",
        ),
        (
            "09-ns",
            1,
            1,
            "ns",
            "AOT programs are single-file, without namespaces",
        ),
        (
            "11-defmacro",
            1,
            1,
            "defmacro",
            "macros at runtime are outside the subset",
        ),
        (
            "12-fn",
            1,
            9,
            "fn",
            "only defn defines functions in the AOT subset",
        ),
        (
            "13-quote",
            1,
            8,
            "quote",
            "quoted values are outside the value flow",
        ),
        (
            "14-defn-multi",
            1,
            1,
            "defn",
            "multi-arity; use a single vector clause",
        ),
        (
            "15-defn-prepost",
            1,
            1,
            "defn",
            "pre/post conditions are runtime checks",
        ),
        (
            "16-let-destructure",
            1,
            1,
            "let",
            "destructuring is outside the subset; bind plain symbols",
        ),
    ];
    for (stem, line, col, form, hint) in cases {
        let path = corpus.join(format!("unsupported/{stem}.clj"));
        let runout = std::fs::read(corpus.join(format!("expected/unsupported/{stem}.runout")))
            .expect("runout golden");
        let (run_out, run_err, run_code) = run_file(&path);
        assert_eq!(run_code, 0, "{stem}: run exit");
        assert!(run_err.is_empty(), "{stem}: run stderr: {run_err:?}");
        assert_eq!(run_out, runout, "{stem}: run stdout vs golden");
        let (build_code, build_err, _, _, _) = build_and_run_flags(stem, &path, &[], &["--strict"]);
        assert_eq!(build_code, 1, "{stem}: build exit");
        let want = format!(
            "{}:{line}:{col}: unsupported form in AOT subset: {form} ({hint})\n",
            path.display()
        );
        assert_eq!(lossy(&build_err), want, "{stem}: build stderr");
        // #256: without `--strict` the same program builds mixed
        // and prints exactly what `run` prints.
        let (mixed_code, mixed_err, mixed_out, mixed_run_err, mixed_run_code) =
            build_and_run(stem, &path);
        assert_eq!(mixed_code, 0, "{stem}: mixed build: {}", lossy(&mixed_err));
        assert_eq!(mixed_out, runout, "{stem}: mixed stdout vs run golden");
        assert!(mixed_run_err.is_empty(), "{stem}: mixed stderr");
        assert_eq!(mixed_run_code, 0, "{stem}: mixed exit");
        // #259: `jit` runs it through the native compiler like `run`.
        let (jit_out, jit_err, jit_code) = jit_file(&path, &[]);
        assert_eq!(jit_code, 0, "{stem}: jit exit: {}", lossy(&jit_err));
        assert_eq!(jit_out, runout, "{stem}: jit stdout vs run golden");
        assert!(jit_err.is_empty(), "{stem}: jit stderr");
    }
}

#[test]
fn malformed_files_report_direct_spans() {
    // 17: the analyzer diagnostic is byte-identical on both paths
    // (same invocation path). 18: `build` reports the raw reader
    // message WITHOUT `run`'s load-file wrapper — pinned divergence.
    let corpus = corpus_dir();
    let p17 = corpus.join("unsupported/17-analyzer.clj");
    let (run_out, run_err, run_code) = run_file(&p17);
    assert_eq!(run_code, 1, "17: run exit");
    assert!(run_out.is_empty(), "17: run stdout: {run_out:?}");
    let (build_code, build_err, _, _, _) =
        build_and_run_flags("17-analyzer", &p17, &[], &["--strict"]);
    assert_eq!(build_code, 1, "17: build exit");
    assert_eq!(build_err, run_err, "17: build stderr == run stderr");
    let want17 = format!(
        "{}:1:1: Syntax error compiling if: Too few arguments to if\n",
        p17.display()
    );
    assert_eq!(lossy(&build_err), want17, "17: exact text");
    let (jit_out, jit_err, jit_code) = jit_file(&p17, &[]);
    assert_eq!(jit_code, 1, "17: jit exit");
    assert!(jit_out.is_empty(), "17: jit stdout: {jit_out:?}");
    assert_eq!(jit_err, build_err, "17: jit stderr == build stderr");
    // #256: a mixed build defers the failure to runtime, where the
    // binary fails exactly like `run`.
    let (mixed_code, _, mixed_out, mixed_err, mixed_exit) = build_and_run("17-analyzer", &p17);
    assert_eq!(mixed_code, 0, "17: mixed build exit");
    assert!(mixed_out.is_empty(), "17: mixed stdout: {mixed_out:?}");
    assert_eq!(mixed_err, run_err, "17: mixed stderr == run stderr");
    assert_eq!(mixed_exit, 1, "17: mixed exit");

    let p18 = corpus.join("unsupported/18-reader.clj");
    let (run_out, run_err, run_code) = run_file(&p18);
    assert_eq!(run_code, 1, "18: run exit");
    assert!(run_out.is_empty(), "18: run stdout: {run_out:?}");
    assert!(
        lossy(&run_err).contains("Syntax error reading source: "),
        "18: run keeps the load wrapper: {run_err:?}"
    );
    let (build_code, build_err, _, _, _) =
        build_and_run_flags("18-reader", &p18, &[], &["--strict"]);
    assert_eq!(build_code, 1, "18: build exit");
    let want18 = format!("{}:1:2: Unmatched delimiter: )\n", p18.display());
    assert_eq!(lossy(&build_err), want18, "18: exact text");
    let (jit_out, jit_err, jit_code) = jit_file(&p18, &[]);
    assert_eq!(jit_code, 1, "18: jit exit");
    assert!(jit_out.is_empty(), "18: jit stdout: {jit_out:?}");
    assert_eq!(jit_err, build_err, "18: jit stderr == build stderr");
    // #256: the mixed binary keeps `run`'s load wrapper (no
    // divergence once the reader runs at startup).
    let (mixed_code, _, mixed_out, mixed_err, mixed_exit) = build_and_run("18-reader", &p18);
    assert_eq!(mixed_code, 0, "18: mixed build exit");
    assert!(mixed_out.is_empty(), "18: mixed stdout: {mixed_out:?}");
    assert_eq!(mixed_err, run_err, "18: mixed stderr == run stderr");
    assert_eq!(mixed_exit, 1, "18: mixed exit");
}

#[test]
fn nesting_limit_reports_clean_diagnostic() {
    // 19 (S9C/S9E): 41-deep input — one past the AOT build limit of
    // 40 — still runs (exit 0, `.runout` stdout, empty stderr);
    // `build` exits 1 with the verbatim nesting diagnostic (no
    // `build: ` prefix), pinning the boundary instead of a machine
    // stack overflow at compile time.
    let corpus = corpus_dir();
    let path = corpus.join("unsupported/19-nesting.clj");
    let runout = std::fs::read(corpus.join("expected/unsupported/19-nesting.runout"))
        .expect("runout golden");
    let (run_out, run_err, run_code) = run_file(&path);
    assert_eq!(run_code, 0, "19: run exit");
    assert!(run_err.is_empty(), "19: run stderr: {run_err:?}");
    assert_eq!(run_out, runout, "19: run stdout vs golden");
    let (build_code, build_err, _, _, _) =
        build_and_run_flags("19-nesting", &path, &[], &["--strict"]);
    assert_eq!(build_code, 1, "19: build exit");
    let want = format!(
        "{}:1:164: nesting depth 41 exceeds the AOT build limit (40)\n",
        path.display()
    );
    assert_eq!(lossy(&build_err), want, "19: exact text");
    let (jit_out, jit_err, jit_code) = jit_file(&path, &[]);
    assert_eq!(jit_code, 0, "19: jit exit: {}", lossy(&jit_err));
    assert_eq!(jit_out, runout, "19: jit stdout vs golden");
    // #256: past the limit nothing goes native; the embedded
    // program runs like `run`.
    let (mixed_code, _, mixed_out, mixed_err, mixed_exit) = build_and_run("19-nesting", &path);
    assert_eq!(mixed_code, 0, "19: mixed build exit");
    assert_eq!(mixed_out, runout, "19: mixed stdout vs golden");
    assert!(mixed_err.is_empty(), "19: mixed stderr: {mixed_err:?}");
    assert_eq!(mixed_exit, 0, "19: mixed exit");
}

#[test]
fn i214_supported_files_match() {
    // `11-loop`/`12-when` (same contract as
    // `supported_files_match_run_byte_for_byte`, i214 corpus):
    // all paths exit 0, empty stderr, stdout == the
    // `run`-seeded golden (which itself matches the JVM golden).
    let corpus = corpus_i214_dir();
    for stem in ["11-loop", "12-when"] {
        let path = corpus.join(format!("{stem}.clj"));
        let golden = std::fs::read(corpus.join(format!("expected/{stem}.out"))).expect("golden");
        let (run_out, run_err, run_code) = run_file(&path);
        assert_eq!(run_code, 0, "{stem}: run exit");
        assert!(run_err.is_empty(), "{stem}: run stderr: {run_err:?}");
        assert_eq!(run_out, golden, "{stem}: run stdout vs golden");
        let (build_code, build_err, exe_out, exe_err, exe_code) = build_and_run(stem, &path);
        assert_eq!(build_code, 0, "{stem}: build exit, stderr: {build_err:?}");
        assert!(build_err.is_empty(), "{stem}: build stderr: {build_err:?}");
        assert_eq!(exe_code, 0, "{stem}: exe exit");
        assert!(exe_err.is_empty(), "{stem}: exe stderr: {exe_err:?}");
        assert_eq!(exe_out, golden, "{stem}: exe stdout vs golden");
        let (jit_out, jit_err, jit_code) = jit_file(&path, &[]);
        assert_eq!(jit_code, 0, "{stem}: jit exit");
        assert!(jit_err.is_empty(), "{stem}: jit stderr: {jit_err:?}");
        assert_eq!(jit_out, golden, "{stem}: jit stdout vs golden");
    }
}

#[test]
fn i214_unsupported_files_fail_build_but_run() {
    // Destructuring `02-loop` (same contract as
    // `unsupported_files_fail_build_but_run`, i214 corpus):
    // `run` still works (exit 0, `.runout` stdout); `build`
    // exits 1 naming `loop` (plain-symbol loops compile).
    let corpus = corpus_i214_dir();
    let (stem, line, col, form, hint) = (
        "02-loop",
        1,
        8,
        "loop",
        "destructuring is outside the subset; bind plain symbols",
    );
    let path = corpus.join(format!("unsupported/{stem}.clj"));
    let runout = std::fs::read(corpus.join(format!("expected/unsupported/{stem}.runout")))
        .expect("runout golden");
    let (run_out, run_err, run_code) = run_file(&path);
    assert_eq!(run_code, 0, "{stem}: run exit");
    assert!(run_err.is_empty(), "{stem}: run stderr: {run_err:?}");
    assert_eq!(run_out, runout, "{stem}: run stdout vs golden");
    let (build_code, build_err, _, _, _) = build_and_run_flags(stem, &path, &[], &["--strict"]);
    assert_eq!(build_code, 1, "{stem}: build exit");
    let want = format!(
        "{}:{line}:{col}: unsupported form in AOT subset: {form} ({hint})\n",
        path.display()
    );
    assert_eq!(lossy(&build_err), want, "{stem}: build stderr");
    let (mixed_code, mixed_err, mixed_out, _, mixed_run_code) = build_and_run(stem, &path);
    assert_eq!(mixed_code, 0, "{stem}: mixed build: {}", lossy(&mixed_err));
    assert_eq!(mixed_out, runout, "{stem}: mixed stdout vs run golden");
    assert_eq!(mixed_run_code, 0, "{stem}: mixed exit");
    let (jit_out, jit_err, jit_code) = jit_file(&path, &[]);
    assert_eq!(jit_code, 0, "{stem}: jit exit: {}", lossy(&jit_err));
    assert_eq!(jit_out, runout, "{stem}: jit stdout vs run golden");
}

/// Lossy text of captured bytes.
fn lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Write `source` to a fresh temp file named `name.clj`.
fn temp_source(name: &str, source: &str) -> PathBuf {
    let dir = tmp(&format!("src-{name}"));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join(format!("{name}.clj"));
    std::fs::write(&path, source).expect("write source");
    path
}

/// Issue #256 BDD scenarios: a mixed build prints exactly what
/// `run` prints (stdout, stderr, exit code), with argv, stdin-less
/// runs, and failures included.
#[test]
fn mixed_builds_match_run_256() {
    let cases: [(&str, &str, &[&str]); 8] = [
        (
            "native-and-embedded",
            "(defn sq [x] (* x x))\n(println (map sq [1 2 3]))\n",
            &[],
        ),
        (
            "main-runs-once",
            "(defn -main [s] (+ s 1))\n(println (-main 1))\n(defn tail [] :t)\n",
            &[],
        ),
        ("argv", "(println *command-line-args*)\n", &["a", "b"]),
        (
            "caught-native-error",
            "(defn f [v] (nth v 5))\n\
             (println (try (f [1]) (catch IndexOutOfBoundsException e :caught)))\n",
            &[],
        ),
        (
            "uncaught-native-error",
            "(defn f [v] (nth v 5))\n(println :before)\n(f [1])\n",
            &[],
        ),
        ("uncaught-embedded-error", "(println 1)\n(nth [] 5)\n", &[]),
        ("trailing-print", "(print \"no newline\")\n", &[]),
        (
            "deep-self-recursion",
            "(defn f [n] (if (= n 0) 0 (+ 1 (f (- n 1)))))\n\
             (println (try (f 10000000) (catch StackOverflowError e :so)))\n",
            &[],
        ),
    ];
    for (name, source, args) in cases {
        let path = temp_source(name, source);
        let mut run = bin();
        run.arg("run").arg(&path).args(args);
        let want = run.output().expect("spawn clojure-rs run");
        let (build_code, build_err, out, err, code) = build_and_run_with(name, &path, args);
        assert_eq!(build_code, 0, "{name}: build: {}", lossy(&build_err));
        assert_eq!(lossy(&out), lossy(&want.stdout), "{name}: stdout");
        assert_eq!(lossy(&err), lossy(&want.stderr), "{name}: stderr");
        assert_eq!(Some(code), want.status.code(), "{name}: exit");
        std::fs::remove_dir_all(path.parent().expect("dir")).expect("cleanup");
    }
}

/// `--report` prints the coverage line; the binary needs no source
/// at runtime (spec 1.3 #4/#5).
#[test]
fn mixed_build_reports_and_runs_without_source_256() {
    let path = temp_source(
        "report",
        "(defn sq [x] (* x x))\n(defn bad [x] (str x))\n(println (sq 3) (bad 4))\n",
    );
    let dir = tmp("report-out");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let prog = dir.join(format!("prog{}", exe_suffix()));
    let archive = ensure_archive();
    let built = bin()
        .args([
            OsStr::new("build"),
            OsStr::new("--report"),
            path.as_os_str(),
            OsStr::new("-o"),
            prog.as_os_str(),
            OsStr::new("--runtime-archive"),
            archive.as_os_str(),
        ])
        .output()
        .expect("spawn clojure-rs build");
    assert_eq!(built.status.code(), Some(0), "{}", lossy(&built.stderr));
    let report_out = lossy(&built.stderr);
    assert!(
        report_out.starts_with("aot: native 2/2 defns, embedded 0 top-level forms, helpers: ")
            && report_out.contains("rustc LLVM ")
            && report_out.ends_with('\n'),
        "{report_out}"
    );
    std::fs::remove_dir_all(path.parent().expect("dir")).expect("remove source");
    let ran = Command::new(&prog).output().expect("run binary");
    assert_eq!(lossy(&ran.stdout), "9 4\n");
    assert_eq!(ran.status.code(), Some(0));
    // A pure program reports full native coverage.
    let pure = temp_source("pure", "(defn sq [x] (* x x))\n(print (sq 3))\n");
    let built = bin()
        .args([
            OsStr::new("build"),
            OsStr::new("--report"),
            pure.as_os_str(),
            OsStr::new("-o"),
            prog.as_os_str(),
            OsStr::new("--runtime-archive"),
            archive.as_os_str(),
        ])
        .output()
        .expect("spawn clojure-rs build");
    let pure_report = lossy(&built.stderr);
    assert!(
        pure_report.starts_with("aot: native 1/1 defns, embedded 0 top-level forms, helpers: ")
            && pure_report.contains("rustc LLVM ")
            && pure_report.ends_with('\n'),
        "{pure_report}"
    );
    std::fs::remove_dir_all(pure.parent().expect("dir")).expect("cleanup");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}
