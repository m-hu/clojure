// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Full link + execute: real `cc`/`link` against the real archive,
//! then run the binary and pin stdout/exit (the S8D corpus replay
//! generalizes this over every corpus file).

use std::path::PathBuf;
use std::process::Command;

use clj_aot::{EmitOptions, LinkOptions, compile_str};

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("clj-aot-exe-{}-{name}", std::process::id()))
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

/// Ensure the static archive exists (plain `cargo test` only builds
/// the rlib — build the staticlib on demand, cached by fingerprints).
/// Builds in an isolated `cljrt-archive` target dir with coverage
/// instrumentation stripped: sharing the test target dir would reuse
/// instrumented dependency objects, whose profile-runtime symbols no
/// plain `cc` link can resolve. Built once per test process — parallel
/// tests must never spawn competing nested cargo builds.
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
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates")
        .parent()
        .expect("workspace")
        .join("Cargo.toml");
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

/// Build `source` to a binary and run it (stdout + exit pinned).
fn build_and_run(name: &str, source: &str) -> (String, String, i32) {
    let dir = tmp(name);
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let out = dir.join(format!("prog{}", exe_suffix()));
    compile_str(
        source,
        "f.clj",
        &out,
        &EmitOptions::default(),
        &LinkOptions {
            runtime_archive: Some(ensure_archive()),
            exe_dir: dir.clone(),
            linker: None,
        },
    )
    .expect("compile");
    let ran = Command::new(&out).output().expect("run binary");
    let code = ran.status.code().expect("exit code");
    let stdout = String::from_utf8_lossy(&ran.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&ran.stderr).into_owned();
    if code != 0 || !stderr.is_empty() {
        eprintln!("binary failed: {code} stdout={stdout:?} stderr={stderr:?}");
    }
    std::fs::remove_dir_all(&dir).expect("cleanup");
    (stdout, stderr, code)
}

#[test]
fn linked_binary_prints() {
    let (stdout, stderr, code) =
        build_and_run("hello", "(defn f [x] (+ x 1)) (print (f 41) \"done\")");
    assert_eq!((stdout.as_str(), stderr.as_str(), code), ("42 done", "", 0));
}

#[test]
fn linked_binary_runs_control_flow() {
    let (stdout, stderr, code) = build_and_run(
        "control",
        "(let [x 7] (if (> x 5) (print \"big\" x) (print \"small\"))) (print (nth [1 2] 1))",
    );
    assert_eq!((stdout.as_str(), stderr.as_str(), code), ("big 72", "", 0));
}

#[test]
fn linked_binary_survives_untaken_let_branches() {
    // T73 item 9: a `let` in an untaken branch never stores — the
    // scope-end drop must see nil-inited slots, not stack garbage
    // (pre-fix: correct stdout, then SIGABRT/SIGSEGV in teardown).
    let (stdout, stderr, code) = build_and_run(
        "cond-let",
        "(defn f [c] (if c (let [x 1] x) (let [y 2] y))) \
         (print (f false) (f true) (if false (let [z 3] z) 4))",
    );
    assert_eq!((stdout.as_str(), stderr.as_str(), code), ("2 1 4", "", 0));
}

#[test]
fn linked_binary_flows_keywords() {
    // T73 item 8: keywords are subset values — the vector-001
    // shape that used to fail the build with `unsupported form
    // in AOT subset: :miss` (JVM prints the same line).
    let (stdout, stderr, code) = build_and_run(
        "keyword",
        "(print [(get [1 2] 5 :miss) (= :a :a) (= :a :b) :x :u/id])",
    );
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("[:miss true false :x :u/id]", "", 0)
    );
}

#[test]
fn linked_binary_runs_slice_a_forms() {
    // Slice A: `cond`/`and`/`or`/`when`/`when-not` desugar with
    // short-circuit semantics (the trailing statements print
    // nothing — their branches never evaluate), `long`/`int` cast
    // through the shared natives (JVM prints the same line).
    let (stdout, stderr, code) = build_and_run(
        "slice-a",
        "(print (cond false 1 (= 1 2) 2 :else 3) (and 1 2) (and 1 nil 3) \
         (or false nil 7) (or) (when (> 2 1) \"y\") (when-not (= 1 1) \"n\") \
         (long 5.9) (int 7) (clojure.core/when true :q)) \
         (and false (print \"X\")) (or true (print \"Y\")) (when false (print \"W\")) \
         (cond true 1 false (print \"Z\"))",
    );
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("3 2 nil 7 nil y nil 5 7 :q", "", 0)
    );
}

#[test]
fn linked_binary_reports_runtime_errors() {
    let (stdout, stderr, code) = build_and_run("divzero", "(print (quot 1 0))");
    assert_eq!(stdout, "");
    assert_eq!(code, 1);
    assert_eq!(
        stderr,
        "f.clj:1:8: java.lang.ArithmeticException: / by zero\n"
    );
}

/// `build_str` in `mode`, then run the binary with `args`.
fn build_mode_and_run(name: &str, source: &str, args: &[&str]) -> (String, String, i32, String) {
    let dir = tmp(name);
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let out = dir.join(format!("prog{}", exe_suffix()));
    let report = clj_aot::build_str(
        source,
        "f.clj",
        &out,
        &EmitOptions::default(),
        &LinkOptions {
            runtime_archive: Some(ensure_archive()),
            exe_dir: dir.clone(),
            linker: None,
        },
        clj_aot::BuildMode::Mixed,
    )
    .expect("build");
    let ran = Command::new(&out).args(args).output().expect("run binary");
    std::fs::remove_dir_all(&dir).expect("cleanup");
    (
        String::from_utf8_lossy(&ran.stdout).into_owned(),
        String::from_utf8_lossy(&ran.stderr).into_owned(),
        ran.status.code().expect("exit code"),
        report,
    )
}

#[test]
fn mixed_binary_interns_natives_and_runs_embedded_256() {
    // `add` (arity 2) and `twice` (arity 1, calls `add`) go native;
    // `fmt` and the tops stay embedded and call them through vars.
    let source = "(defn add [a b] (+ a b))\n\
                  (defn twice [x] (add x x))\n\
                  (defn fmt [x] (str \"<\" x \">\"))\n\
                  (println (fmt (twice (add 1 2))) *command-line-args*)\n\
                  (print (try (add :a 1) (catch ClassCastException _ :cce)))\n\
                  (eval '(def zz 5))\n(print zz)\n";
    // `zz` only exists at run time (via `eval`), so the native front
    // end cannot analyze the last form: the mixed path takes over.
    let (stdout, stderr, code, report) = build_mode_and_run("mixed", source, &["x"]);
    assert!(
        report.starts_with("aot: native 2/3 defns, embedded 5 top-level forms (native: front end"),
        "{report}"
    );
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("<6> (x)\n:cce5", "", 0)
    );
    // Without the runtime-only var the whole program goes native.
    let native = source.replace("(eval '(def zz 5))\n(print zz)\n", "");
    let (stdout, stderr, code, report) = build_mode_and_run("native", &native, &["x"]);
    assert_eq!(
        report,
        format!(
            "aot: native 3/3 defns, embedded 0 top-level forms, {}",
            clj_aot::helpers_summary()
        )
    );
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("<6> (x)\n:cce", "", 0)
    );
}

#[test]
fn mixed_binary_native_failure_exits_like_run_256() {
    let (stdout, stderr, code, _) =
        build_mode_and_run("mixed-fail", "(defn f [v] (nth v 3))\n(f [1])\n", &[]);
    assert_eq!(stdout, "");
    assert_eq!(code, 1);
    assert!(stderr.starts_with("f.clj:"), "{stderr}");
    assert!(stderr.contains("IndexOutOfBoundsException"), "{stderr}");
}

#[test]
fn native_binary_runs_assoc_conj_and_fx_259() {
    let source = "\
(let [t (transient [[1] [2]])
      k 1
      x 9
      r (assoc! t 0 (conj (nth t 0) x))]
  (assoc! r k (conj (nth r k) [x]))
  (pop! (transient [1 2]))
  (assoc! (transient [1 2]) 0 9)
  (println (persistent! r)))
";
    let (stdout, stderr, code, report) = build_mode_and_run("assoc-conj-fx", source, &[]);
    assert_eq!(
        report,
        format!(
            "aot: native 0/0 defns, embedded 0 top-level forms, {}",
            clj_aot::helpers_summary()
        )
    );
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("[[1 9] [2 [9]]]\n", "", 0)
    );
}

#[test]
fn native_binary_runs_nested_loop_move_and_string_preds() {
    let source = "\
(require '[clojure.string :as str])
(defn sweep [words w]
  (loop [k 0 stack [] best -1]
    (if (= k (count words))
      [best (count stack)]
      (let [cur (nth words k)
            stack (loop [st stack]
                    (if (and (seq st)
                             (not (str/starts-with? w (peek st))))
                      (recur (pop st))
                      st))
            best (if (and (seq stack)
                          (str/ends-with? cur \"b\")
                          (str/includes? cur \"a\"))
                   k
                   best)]
        (recur (inc k) (conj stack cur) best)))))
(println (sweep [\"a\" \"ab\" \"c\"] \"ab\")
         (try (str/starts-with? nil \"a\") (catch Exception _ :npe))
         (with-redefs [str/starts-with? (fn [_ _] true)]
           (str/starts-with? \"a\" \"z\")))
";
    let (stdout, stderr, code, _report) = build_mode_and_run("nested-loop-str", source, &[]);
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("[1 3] :npe true\n", "", 0)
    );
}

#[test]
fn native_binary_preserves_nested_or_defaults_and_assoc_miss_parity() {
    let source = "\
(defn nested-vec-or []
  (loop [n 1 acc nil]
    (if (zero? n)
      acc
      (let [x [10 20]
            [_ {:keys [b] :or {b x}}] [x {}]]
        (recur 0 b)))))
(defn nested-map-or []
  (loop [n 1 acc nil]
    (if (zero? n)
      acc
      (let [x [30 40]
            {a :a {:keys [b] :or {b x}} :k} {:a x :k {}}]
        (recur 0 [a b])))))
(println (nested-vec-or)
         (nested-map-or)
         (try (assoc 2 3 4) (catch Exception _ :err))
         (try (* {} :k (count [1 2])) (catch Exception _ :err)))
";
    let (stdout, stderr, code, _report) = build_mode_and_run("nested-or-assoc-miss", source, &[]);
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("[10 20] [[30 40] [30 40]] :err :err\n", "", 0)
    );
}

#[test]
fn native_binary_exercises_lite_ns_stdin_and_full_cli() {
    use std::io::Write;
    use std::process::Stdio;

    let dir = tmp("lite-stdin");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let out = dir.join(format!("prog{}", exe_suffix()));
    let lite_src = "\
(ns demo.app (:require [clojure.string :as s]))
(println (read-line) (s/join \",\" *command-line-args*))
";
    let report = clj_aot::build_str(
        lite_src,
        "f.clj",
        &out,
        &EmitOptions::default(),
        &LinkOptions {
            runtime_archive: Some(ensure_archive()),
            exe_dir: dir.clone(),
            linker: None,
        },
        clj_aot::BuildMode::Mixed,
    )
    .expect("compile lite stdin");
    assert!(report.starts_with("aot: native "), "{report}");
    let mut child = Command::new(&out)
        .args(["x", "y"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lite");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(b"hi\n")
        .expect("write stdin");
    let ran = child.wait_with_output().expect("wait lite");
    std::fs::remove_dir_all(&dir).expect("cleanup");
    assert_eq!(
        (
            String::from_utf8_lossy(&ran.stdout).into_owned().as_str(),
            String::from_utf8_lossy(&ran.stderr).into_owned().as_str(),
            ran.status.code()
        ),
        ("hi x,y\n", "", Some(0))
    );

    let req_src = "\
(require 'clojure.string '[clojure.string :as rq] '[clojure.string :refer [trim]])
(println (trim (rq/join \",\" [(Long/parseLong \"7\") (instance? Long 1)])))
";
    let (stdout, stderr, code, _report) = build_mode_and_run("lite-require-string", req_src, &[]);
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("7,true\n", "", 0)
    );

    let caps_src = "\
(println (deref (atom [(re-find #\"a+\" \"baa\") (re-matches #\"a+\" \"aa\") (vec (take 2 (map inc [1 2]))) (Long/parseLong \"7\")])))
";
    let (stdout, stderr, code, _report) =
        build_mode_and_run("lite-caps-seq-re-host", caps_src, &[]);
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("[aa aa [2 3] 7]\n", "", 0)
    );

    let lite_str_src = "\
(ns demo.str (:require [clojure.string :as s]))
(println (s/blank? \" \") (vec *command-line-args*))
";
    let (stdout, stderr, code, _report) = build_mode_and_run("lite-str-cli", lite_str_src, &["a"]);
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("true [a]\n", "", 0)
    );

    let full_src = "\
(ns demo.full (:require [clojure.walk :as w]))
(println (w/postwalk identity [1 2]) (vec *command-line-args*))
";
    let (stdout, stderr, code, _report) = build_mode_and_run("full-ns-cli", full_src, &["a"]);
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("[1 2] [a]\n", "", 0)
    );

    let full_err_src = "(println (eval '(/ 1 0)))";
    let (stdout, stderr, code, _report) = build_mode_and_run("full-err", full_err_src, &[]);
    assert_eq!(stdout, "");
    assert_eq!(code, 1);
    assert!(stderr.contains("ArithmeticException"), "{stderr}");

    let scalar_src = "\
(defn step [x] (+ x 1))
(defn run-scalar [s u pairs]
  (let [t (transient [])]
    (loop [i 0 acc 0]
      (if (< i 20)
        (do
          (conj! t i)
          (if (> i 0) (assoc! t (- i 1) (* i 2)) nil)
          (recur (step i) (+ acc (if (= (nth s (mod i (count s))) \\a) 1 0))))
        (do
          (pop! t)
          [acc
           (count (persistent! t))
           (loop [j 0 uacc 0]
             (if (< j (count u))
               (recur (+ j 1) (+ uacc (if (= (nth u j) \\é) 10 1)))
               uacc))
           (loop [k 0 pacc 0]
             (if (< k (count pairs))
               (let [[a b] (nth pairs k)]
                 (recur (+ k 1) (+ pacc a b)))
               pacc))
           (loop [m 0 hi 0 lo 1000]
             (if (< m 20)
               (recur (+ m 1) (max hi (rem (* m 37) 101)) (min lo (rem (* m 53) 97)))
               (+ hi lo)))])))))
(println (run-scalar \"aba\" \"aé\" [[3 4] [5 6]]))
";
    let (stdout, stderr, code, _report) = build_mode_and_run("scalar-271", scalar_src, &[]);
    assert_eq!(
        (stdout.as_str(), stderr.as_str(), code),
        ("[13 19 11 18 97]\n", "", 0)
    );
}
