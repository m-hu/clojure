// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Hermetic `--emit-ir` / `--emit-obj` paths (no linker, no
//! archive): outputs land, linking is skipped, failures name
//! their cause.

use std::path::PathBuf;

use clj_aot::{CompileError, EmitOptions, LinkOptions, compile_file, compile_str};

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("clj-aot-emit-{}-{name}", std::process::id()))
}

fn no_link() -> LinkOptions {
    LinkOptions {
        runtime_archive: None,
        exe_dir: tmp("no-such-dir"),
        linker: Some("clj-aot-definitely-not-a-linker".to_owned()),
    }
}

#[test]
fn emit_ir_writes_module_and_skips_link() {
    let dir = tmp("ir");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let src = dir.join("f.clj");
    std::fs::write(&src, "(defn f [x] (+ x 1)) (print (f 41))").expect("src");
    let ir = dir.join("f.ll");
    let out = dir.join("f");
    compile_file(
        &src,
        &out,
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect("emit ir");
    let text = std::fs::read_to_string(&ir).expect("read ir");
    assert!(text.contains("define ptr @cljfn_f_0"), "{text}");
    assert!(text.contains("define i32 @main(i32"), "{text}");
    assert!(!out.exists(), "linking must be skipped");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn emit_obj_writes_object_and_skips_link() {
    let dir = tmp("obj");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let obj = dir.join("f.o");
    let out = dir.join("f");
    compile_str(
        "(print 1)",
        "f.clj",
        &out,
        &EmitOptions {
            emit_ir: None,
            emit_obj: Some(obj.clone()),
        },
        &no_link(),
    )
    .expect("emit obj");
    assert!(!std::fs::read(&obj).expect("read obj").is_empty());
    assert!(!out.exists(), "linking must be skipped");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn emit_covers_arity_branches_through_the_rlib() {
    // The unit snapshots verify these shapes; emitting them here
    // executes the rlib copy of the arity joins too (the libtest
    // copy's profile is dropped as mismatched on merge).
    for (i, source) in [
        "(defn f [x] x) (if true (f 1 2) 3)",
        "(defn f [x] x) (if true 3 (f 1 2))",
        "(defn f [x] x) (if true (f 1 2) (f 1 2))",
        "(defn f [x] x) (if (f 1 2) 1 2)",
        "(defn f [x] x) (+ (f 1 2) 4)",
    ]
    .iter()
    .enumerate()
    {
        let dir = tmp(&format!("arity-{i}"));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let ir = dir.join("f.ll");
        compile_str(
            source,
            "f.clj",
            &dir.join("f"),
            &EmitOptions {
                emit_ir: Some(ir.clone()),
                emit_obj: None,
            },
            &no_link(),
        )
        .expect("emit arity branch");
        let text = std::fs::read_to_string(&ir).expect("read ir");
        assert!(
            text.contains("call ptr @rt_arity_error"),
            "{source}\n{text}"
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}

#[test]
fn emit_covers_conditional_let_slot_init_through_the_rlib() {
    // T73 item 9: the unit snapshot pins the shape; emitting here
    // executes the rlib copy of the slot nil-init and the `let`
    // drop-old (the libtest copy's profile is dropped as
    // mismatched on merge).
    let dir = tmp("let-init");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let ir = dir.join("f.ll");
    compile_str(
        "(defn f [c] (if c (let [x 1] x) 2)) (print (f false) (if true 1 (let [y 2] y)))",
        "f.clj",
        &dir.join("f"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect("emit conditional lets");
    let text = std::fs::read_to_string(&ir).expect("read ir");
    assert!(text.contains("call ptr @rt_nil"), "{text}");
    assert!(text.contains("call void @rt_drop"), "{text}");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn emit_covers_keyword_literals_through_the_rlib() {
    // T73 item 8: the unit snapshots pin the shape; emitting here
    // executes the rlib copy of the planner keyword arm and the
    // lowering, both ns shapes (the libtest copy's profile is
    // dropped as mismatched on merge).
    let dir = tmp("kw");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let ir = dir.join("f.ll");
    compile_str(
        "(print (get [1 2] 5 :miss) [:a :u/id])",
        "f.clj",
        &dir.join("f"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect("emit keywords");
    let text = std::fs::read_to_string(&ir).expect("read ir");
    assert_eq!(
        text.match_indices("call ptr @rt_keyword").count(),
        3,
        "{text}"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn emit_covers_slice_a_forms_through_the_rlib() {
    // Slice A: the planner unit tests pin the desugared shapes;
    // emitting here executes the rlib copies of the new planner
    // arms and the new prim declarations (the libtest copy's
    // profile is dropped as mismatched on merge).
    let dir = tmp("slice-a");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let ir = dir.join("f.ll");
    compile_str(
        "(defn and [x] x) (print (and 1) (cond false 1 :else 2) (or nil 3) \
         (when true 4) (when-not false 5) (long 6.5) (int 7) \
         (clojure.core/cond false 8 true 9) (clojure.core/when-not true 10 11))",
        "f.clj",
        &dir.join("f"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect("emit slice-a forms");
    let text = std::fs::read_to_string(&ir).expect("read ir");
    assert!(text.contains("call ptr @rt_long"), "{text}");
    assert!(text.contains("call ptr @rt_int_cast"), "{text}");
    assert!(text.contains("call i8 @rt_is_truthy"), "{text}");
    assert!(text.contains("call ptr @cljfn_and_0"), "{text}");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn out_of_subset_names_the_form() {
    let err = compile_str(
        "(def x 1)",
        "f.clj",
        &tmp("nope"),
        &EmitOptions::default(),
        &no_link(),
    )
    .expect_err("must reject def");
    assert_eq!(
        err,
        CompileError::Plan(
            "f.clj:1:1: unsupported form in AOT subset: def (only defn defines in the AOT subset)"
                .to_owned()
        )
    );
    assert_eq!(
        format!("{err}"),
        "f.clj:1:1: unsupported form in AOT subset: def (only defn defines in the AOT subset)"
    );
}

#[test]
fn io_failures_name_the_path() {
    let missing = tmp("no-such-dir").join("f.clj");
    let err = compile_file(&missing, &tmp("nope"), &EmitOptions::default(), &no_link())
        .expect_err("missing input");
    assert!(matches!(err, CompileError::Io { .. }), "{err:?}");
    assert!(format!("{err}").contains("f.clj"), "{err}");

    let err = compile_str(
        "(print 1)",
        "f.clj",
        &tmp("nope"),
        &EmitOptions {
            emit_ir: Some(tmp("no-such-dir").join("f.ll")),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect_err("unwritable ir");
    assert!(matches!(err, CompileError::Io { .. }), "{err:?}");
}

#[test]
fn display_covers_every_variant() {
    assert_eq!(format!("{}", CompileError::Target("t".to_owned())), "t");
    assert_eq!(format!("{}", CompileError::Emit("e".to_owned())), "e");
    assert_eq!(
        format!(
            "{}",
            CompileError::Link(clj_aot::link::LinkError::LinkFailed("l".to_owned()))
        ),
        "l"
    );
}

/// `depth` nested parens around `1`.
fn parens(depth: usize) -> String {
    format!("{}1{}", "(".repeat(depth), ")".repeat(depth))
}

/// `depth` nested `(+ 1 ...)` calls (in-subset at any depth).
fn plus_nest(depth: usize) -> String {
    debug_assert!(depth > 0);
    format!("{}(+ 1 1{}", "(+ 1 ".repeat(depth - 1), ")".repeat(depth))
}

fn no_emit() -> EmitOptions {
    EmitOptions {
        emit_ir: None,
        emit_obj: None,
    }
}

#[test]
fn nesting_rejects_forty_one_deep() {
    let err = compile_str(&parens(41), "f.clj", &tmp("nope"), &no_emit(), &no_link())
        .expect_err("41 deep rejects");
    assert_eq!(
        err,
        CompileError::Plan(
            "f.clj:1:41: nesting depth 41 exceeds the AOT build limit (40)".to_owned()
        )
    );
}

#[test]
fn nesting_rejects_mixed_delimiters() {
    let source = "([{".repeat(14);
    let err = compile_str(&source, "f.clj", &tmp("nope"), &no_emit(), &no_link())
        .expect_err("mixed 42 rejects");
    assert_eq!(
        err,
        CompileError::Plan(
            "f.clj:1:41: nesting depth 41 exceeds the AOT build limit (40)".to_owned()
        )
    );
}

#[test]
fn nesting_rejects_quote_runs() {
    let source = format!("{}x", "'".repeat(41));
    let err = compile_str(&source, "f.clj", &tmp("nope"), &no_emit(), &no_link())
        .expect_err("41 quotes reject");
    assert_eq!(
        err,
        CompileError::Plan(
            "f.clj:1:41: nesting depth 41 exceeds the AOT build limit (40)".to_owned()
        )
    );
}

#[test]
fn nesting_rejects_deep_expansion() {
    let threaded = format!("(-> 0{} )", " (+ 1)".repeat(300));
    let err = compile_str(&threaded, "f.clj", &tmp("nope"), &no_emit(), &no_link())
        .expect_err("300-threaded rejects");
    assert_eq!(
        err,
        CompileError::Plan(
            "f.clj:1:1: nesting depth 300 after expansion exceeds the AOT build limit (40)"
                .to_owned()
        )
    );
}

#[test]
fn nesting_accepts_boundary_through_emit_ir() {
    let dir = tmp("nest-ok");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let ir = dir.join("f.ll");
    let out = dir.join("f");
    compile_str(
        &plus_nest(40),
        "f.clj",
        &out,
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect("40 deep compiles");
    let text = std::fs::read_to_string(&ir).expect("read ir");
    assert!(text.contains("define i32 @main(i32"), "{text}");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn nesting_accepts_boundary_on_small_stack() {
    // The 1 MB thread models the smallest main-thread stack
    // (Windows); the limit holds a 4x margin under the probed
    // 161-level overflow.
    let dir = tmp("nest-1mb");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let ir = dir.join("f.ll");
    let out = dir.join("f");
    let source = plus_nest(40);
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            compile_str(
                &source,
                "f.clj",
                &out,
                &EmitOptions {
                    emit_ir: Some(ir.clone()),
                    emit_obj: None,
                },
                &no_link(),
            )
            .expect("boundary compiles on a 1 MB stack");
        })
        .expect("spawn")
        .join()
        .expect("worker panicked");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn nesting_keeps_deep_unsupported_clean_on_small_stack() {
    let source = parens(40);
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let err = compile_str(&source, "f.clj", &tmp("nope"), &no_emit(), &no_link())
                .expect_err("unsupported, not overflow");
            assert_eq!(
                err,
                CompileError::Plan(
                    "f.clj:1:40: unsupported form in AOT subset: 1 (higher-order calls are outside the subset)"
                        .to_owned()
                )
            );
        })
        .expect("spawn")
        .join()
        .expect("worker panicked");
}

#[test]
fn nesting_rejects_backquote_runs_on_small_stack() {
    // M2: five ticks read 121 deep (rescan phase), eight reject
    // at the scan — both must diagnose, not overflow, on 1 MB.
    for (source, want) in [
        (
            "`````x".to_owned(),
            "f.clj:1:1: nesting depth 121 after expansion exceeds the AOT build limit (40)",
        ),
        (
            "````````x".to_owned(),
            "f.clj:1:6: nesting depth 48 exceeds the AOT build limit (40)",
        ),
    ] {
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || {
                let err = compile_str(&source, "f.clj", &tmp("nope"), &no_emit(), &no_link())
                    .expect_err("backquote run must diagnose");
                assert_eq!(err, CompileError::Plan(want.to_owned()));
            })
            .expect("spawn")
            .join()
            .expect("worker panicked");
    }
}

#[test]
fn nesting_accepts_parens_in_strings() {
    let dir = tmp("nest-str");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let source = format!("(print \"{}{}\")", "(".repeat(50), ")".repeat(50));
    let ir = dir.join("f.ll");
    let out = dir.join("f");
    compile_str(
        &source,
        "f.clj",
        &out,
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect("string parens stay inert");
    assert!(ir.exists(), "ir written");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn nesting_accepts_wide_shallow() {
    let dir = tmp("nest-wide");
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let source = format!("(do {} )", "(print 1) ".repeat(100));
    let ir = dir.join("f.ll");
    let out = dir.join("f");
    compile_str(
        &source,
        "f.clj",
        &out,
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
    )
    .expect("wide but shallow compiles");
    assert!(ir.exists(), "ir written");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn nesting_backstops_user_macros() {
    let err = compile_str(
        "(defmacro m [x] x)",
        "f.clj",
        &tmp("nope"),
        &no_emit(),
        &no_link(),
    )
    .expect_err("defmacro rejects");
    assert_eq!(
        err,
        CompileError::Plan(
            "f.clj:1:1: unsupported form in AOT subset: defmacro (macros at runtime are outside the subset)"
                .to_owned()
        )
    );
}

#[test]
fn mixed_emit_ir_lowers_trampolines_and_runtime_main_256() {
    let ir = tmp("mixed.ll");
    let report = clj_aot::build_str(
        "(defn add [a b] (+ a b))\n(println (add 1 2))\n(eval '(def zz 1))\n(println zz)\n",
        "f.clj",
        &tmp("nope"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("mixed build");
    assert!(
        report.starts_with("aot: native 1/1 defns, embedded 3 top-level forms (native: front end"),
        "{report}"
    );
    let text = std::fs::read_to_string(&ir).expect("ir");
    std::fs::remove_file(&ir).expect("cleanup");
    for needle in [
        "define ptr @cljtramp_0(ptr",
        "call void @rt_register_native(i64 0",
        "call i32 @rt_run_program(",
        "(clj.aot/n 0)",
    ] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }
    // Strict mode keeps the first-unsupported-form failure.
    let err = clj_aot::build_str(
        "(println 1)\n",
        "f.clj",
        &tmp("nope"),
        &EmitOptions::default(),
        &no_link(),
        clj_aot::BuildMode::Strict,
    )
    .unwrap_err();
    assert!(matches!(err, CompileError::Plan(_)), "{err:?}");
    assert_eq!(clj_aot::BuildMode::default(), clj_aot::BuildMode::Mixed);
}

#[test]
fn native_emit_ir_lowers_units_constants_and_main_259() {
    let ir = tmp("native.ll");
    let report = clj_aot::build_str(
        "(def k :x)\n(defn add [a b] (+ a b))\n(println (add 1 2) k \"s\" 1.5)\n",
        "f.clj",
        &tmp("nope"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("native build");
    assert_eq!(
        report,
        format!(
            "aot: native 2/2 defns, embedded 0 top-level forms, {}",
            clj_aot::helpers_summary()
        )
    );
    let text = std::fs::read_to_string(&ir).expect("ir");
    std::fs::remove_file(&ir).expect("cleanup");
    for needle in [
        "@rt_n_main_lite(",
        "@rt_n_declare_vars(",
        "@rt_n_declare_fn(",
        "@rt_n_set_const(",
        "@cljconst_init",
    ] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }
}

#[test]
fn native_unreadable_constant_falls_back_to_mixed_259() {
    // A macro expanding to a live object (an atom) leaves a constant
    // with no readable form: the native compiler declines and the
    // build takes the mixed path, naming why.
    let ir = tmp("unreadable.ll");
    let report = clj_aot::build_str(
        "(defmacro live [] (atom 1))\n(defn f [] @(live))\n(println (f))\n",
        "f.clj",
        &tmp("nope"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("mixed build");
    std::fs::remove_file(&ir).expect("cleanup");
    assert!(
        report.contains("(native: constant without a readable form: "),
        "{report}"
    );
}

#[test]
fn native_build_does_not_run_side_effecting_def_metadata_259() {
    // A fn def whose name metadata has a side effect is only analyzed
    // at build time (the effect runs once, in the built program).
    let ir = tmp("side.ll");
    let report = clj_aot::build_str(
        "(defn ^{:k (throw (ex-info \"ran at build time\" {}))} f [] 1)\n(println (f))\n",
        "f.clj",
        &tmp("nope"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("native build");
    std::fs::remove_file(&ir).expect("cleanup");
    // Had the metadata run at build time, the front end would have
    // failed and the build would have fallen back to mixed.
    assert_eq!(
        report,
        format!(
            "aot: native 1/1 defns, embedded 0 top-level forms, {}",
            clj_aot::helpers_summary()
        )
    );
}

#[test]
fn native_emit_ir_covers_assoc_conj_fx_and_nested_moves_259() {
    let ir = tmp("assoc-conj-fx.ll");
    let src = "\
(defn f [k x]
  (let [t (transient [[1] [2]])]
    (assoc! t 0 (conj (nth t 0) x))
    (let [r (assoc! t k (conj (nth t k) [x 9]))]
      (assoc! r 0 (pop [1]))
      (assoc! r 0 (conj [] 1))
      (assoc! r 0 (conj (get r 0) 1))
      (assoc! r 0 (conj (nth [1] 0) 1))
      (assoc! r 0 (conj (nth r 1) 1))
      (assoc! r 0 (conj (nth r 0) (inc 1)))
      ((fn [] [(assoc! r 0 (conj (nth r 0) 1)) x]))
      ((fn [lt] (assoc! lt 0 (conj (nth lt 0) k))))
      (let [y (vector k x)]
        (assoc! r 0 (conj (nth r 0) y)))
      (if (= k 999)
        (letfn [(bad [] 0)]
          (assoc! bad 0 (conj (nth bad 0) 1))
          (assoc! r bad (conj (nth r bad) 1))))
      (pop! (transient [1 2]))
      (assoc! (transient [1 2]) 0 9)
      (loop [i 0 u (transient [1])]
        (if (= i 1)
          (persistent! u)
          (recur (inc i) (if (= i 0) (do (conj! u i) u) u)))))))
(defn g [v]
  (let [t (transient [[1]])
        y (vector v)]
    (assoc! t 0 (conj (nth t 0) y))))
(require '[clojure.string :as str])
(defn h [words w]
  (loop [k 0 stack [] best -1]
    (if (= k (count words))
      best
      (let [cur (nth words k)
            stack (loop [st stack]
                    (if (and (seq st)
                             (not (str/starts-with? w (nth (peek st) 0))))
                      (recur (pop st))
                      st))
            best (if (and (seq stack) (str/ends-with? cur \"a\")
                          (str/includes? w cur))
                   (max best k)
                   best)]
        (recur (inc k) (conj stack [cur k]) best)))))
(println (f 1 7) (persistent! (g 3)) (h [\"ab\" \"abc\"] \"ab\"))
";
    let report = clj_aot::build_str(
        src,
        "f.clj",
        &tmp("nope"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("native build");
    let text = std::fs::read_to_string(&ir).expect("ir");
    std::fs::remove_file(&ir).expect("cleanup");
    assert!(!report.contains("(native:"), "{report}");
    assert!(text.contains("@rt_n_assoc_conj("), "{text}");
    assert!(
        text.contains("@rt_bc_2_str_starts_with")
            || text.contains("memcmp")
            || text.contains("@bcmp("),
        "{text}"
    );
}

#[test]
fn native_emit_ir_marks_clojure_string_def_as_rebinding() {
    let ir = tmp("native-str-rebind.ll");
    let src = "\
(in-ns 'clojure.string)
(def starts-with? (fn [_ _] false))
(in-ns 'user)
(println (clojure.string/starts-with? \"ab\" \"a\"))
";
    let report = clj_aot::build_str(
        src,
        "f.clj",
        &tmp("nope"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("native build");
    let text = std::fs::read_to_string(&ir).expect("ir");
    std::fs::remove_file(&ir).expect("cleanup");
    assert!(!report.contains("(native:"), "{report}");
    assert!(text.contains("@rt_n_fast_vars"), "{text}");
}

#[test]
fn native_emit_ir_covers_owned_move_assoc_update_and_loop_unsupported() {
    let ir = tmp("owned-move-ops.ll");
    let src = "\
(defn mv-ops [m]
  (let [a (assoc m :a 1 :b 2)
        b (assoc-in a [:x] 3)
        c (update b :x inc)
        d (update-in c [:x] inc)]
    d))
(println (mv-ops {:x 0}))
";
    let report = clj_aot::build_str(
        src,
        "f.clj",
        &tmp("nope-mv"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("native build");
    std::fs::remove_file(&ir).expect("cleanup");
    assert!(!report.contains("(native:"), "{report}");

    let bad_loop = "\
(defmacro bad-val [] (list 'quote +))
(defn g [] (loop [x 0] (bad-val)))
(println (g))
";
    let bad_ir = tmp("bad-loop.ll");
    let bad_report = clj_aot::build_str(
        bad_loop,
        "f.clj",
        &tmp("nope-bad-loop"),
        &EmitOptions {
            emit_ir: Some(bad_ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("mixed fallback");
    std::fs::remove_file(&bad_ir).expect("cleanup");
    assert!(
        bad_report.contains("(native: constant without a readable form:"),
        "{bad_report}"
    );
}

#[test]
fn native_emit_ir_selects_lite_vs_full_boot() {
    let lite_src = "\
(ns demo.app
  (:refer-clojure :exclude [get])
  (:require [clojure.string :as s] clojure.string))
(require 'clojure.string '[clojure.string :as rq] '[clojure.string :refer [trim]])
(declare fwd)
(def ^:dynamic *d* 1)
(defn fwd [x] (trim (s/join \",\" [x (Long/parseLong \"7\")])))
(defn g [] (binding [*d* 2] (set! *d* 3) (vector *d* #'fwd (fwd \"ok\") (rq/blank? \" \") (instance? Long 1))))
(println (g))
";
    let lite_ir = tmp("boot-lite.ll");
    let lite_report = clj_aot::build_str(
        lite_src,
        "f.clj",
        &tmp("nope-lite"),
        &EmitOptions {
            emit_ir: Some(lite_ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("lite native");
    let lite_text = std::fs::read_to_string(&lite_ir).expect("ir");
    std::fs::remove_file(&lite_ir).expect("cleanup");
    assert!(lite_report.starts_with("aot: native "), "{lite_report}");
    assert!(
        lite_text.contains("@rt_n_require_string_lite("),
        "{lite_text}"
    );
    assert!(lite_text.contains("@rt_n_ns_lite("), "{lite_text}");
    assert!(lite_text.contains("@rt_n_main_lite("), "{lite_text}");
    // The Linux static link's `--wrap` targets ride in every module (#271).
    assert!(
        lite_text.contains("@__wrap___register_frame_info(ptr")
            && lite_text.contains("@__wrap___deregister_frame_info(ptr"),
        "{lite_text}"
    );

    let lite_ext_ir = tmp("boot-lite-ext.ll");
    let lite_ext_report = clj_aot::build_str(
        "(ns demo.s (:require [clojure.string :as str])) \
         (require '[clojure.string :as sq]) \
         (run! println [(sq/trim \" 1 \") ((fnil inc 0) nil) (reduce-kv (fn [a k v] (+ a v)) 0 {:a 1}) (str/join \",\" [1 2])])",
        "e.clj",
        &tmp("nope-lite-ext"),
        &EmitOptions {
            emit_ir: Some(lite_ext_ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("lite ext native");
    let lite_ext_text = std::fs::read_to_string(&lite_ext_ir).expect("ir");
    std::fs::remove_file(&lite_ext_ir).expect("cleanup");
    assert!(
        lite_ext_report.starts_with("aot: native "),
        "{lite_ext_report}"
    );
    assert!(
        lite_ext_text.contains("@rt_n_main_lite("),
        "{lite_ext_text}"
    );
    assert!(
        lite_ext_text.contains("@rt_n_require_string_lite("),
        "{lite_ext_text}"
    );

    for (tag, full_src) in [
        ("macro", "(defmacro m [x] x) (println (m 1))"),
        ("ns-use", "(ns demo.u (:use clojure.string)) (println 1)"),
        (
            "ns-import",
            "(ns demo.i (:import [java.util Locale])) (println Locale/US)",
        ),
        (
            "ns-walk",
            "(ns demo.w (:require [clojure.walk :as w])) (println (w/postwalk identity [1]))",
        ),
        ("top-req-other", "(require 'clojure.walk) (println 1)"),
        ("top-req-dyn", "(let [x 'clojure.string] (require x))"),
        ("top-req-empty", "(defn f [] (require)) (println 1)"),
        ("core-eval", "(println (eval '(+ 1 2)))"),
        ("redef", "(with-redefs [inc dec] (println (inc 1)))"),
    ] {
        let ir = tmp(&format!("boot-full-{tag}.ll"));
        let report = clj_aot::build_str(
            full_src,
            "f.clj",
            &tmp("nope-full"),
            &EmitOptions {
                emit_ir: Some(ir.clone()),
                emit_obj: None,
            },
            &no_link(),
            clj_aot::BuildMode::Mixed,
        )
        .expect("full native");
        let text = std::fs::read_to_string(&ir).expect("ir");
        std::fs::remove_file(&ir).expect("cleanup");
        assert!(report.starts_with("aot: native "), "{tag}: {report}");
        assert!(text.contains("@rt_n_main("), "{tag}: {text}");
    }
}

#[test]
fn native_emit_splices_clojure_set_under_lite_only() {
    // #271: a leading `ns` requiring `clojure.set` compiles the vendored
    // library into the program and keeps the Lite boot; a program that
    // needs the full boot anyway drops those units (its `require` loads
    // the interpreted library).
    let emit = |tag: &str, src: &str| {
        let ir = tmp(&format!("splice-{tag}.ll"));
        let report = clj_aot::build_str(
            src,
            "s.clj",
            &tmp("nope-splice"),
            &EmitOptions {
                emit_ir: Some(ir.clone()),
                emit_obj: None,
            },
            &no_link(),
            clj_aot::BuildMode::Mixed,
        )
        .expect("splice native");
        let text = std::fs::read_to_string(&ir).expect("ir");
        std::fs::remove_file(&ir).expect("cleanup");
        (report, text)
    };
    let (report, lite) = emit(
        "lite",
        "(ns demo.set (:require [clojure.set :as set] [clojure.string :as str]))
         (defn pick [a b] (set/union a b))
         (println (pick #{1} #{2}) (str/join [1]))",
    );
    assert!(report.starts_with("aot: native 1/1 defns"), "{report}");
    assert!(lite.contains("@rt_n_main_lite("), "{lite}");
    assert!(lite.contains("_clojure_set_union"), "{lite}");
    let (report, full) = emit(
        "full",
        "(ns demo.setm (:require [clojure.set :as set]))
         (defmacro m [x] x)
         (println (m (set/union #{1} #{2})))",
    );
    assert!(report.starts_with("aot: native 0/0 defns"), "{report}");
    assert!(full.contains("@rt_n_main("), "{full}");
    assert!(!full.contains("_clojure_set_union"), "{full}");
    // `clojure.string` splices only what no Lite native covers
    // (`replace` and its helper); redefining a native-served var still
    // takes the full boot.
    let (_, lite) = emit(
        "string",
        "(ns demo.str (:require [clojure.string :as str]))
         (println (str/replace \"a1\" #\"\\d\" \"x\") (str/join [1]))",
    );
    assert!(lite.contains("@rt_n_main_lite("), "{lite}");
    assert!(lite.contains("_clojure_string_replace_by"), "{lite}");
    assert!(!lite.contains("_clojure_string_join"), "{lite}");
    let (_, redef) = emit(
        "string-redef",
        "(ns demo.strr (:require [clojure.string :as str]))
         (in-ns 'clojure.string)
         (defn trim [s] s)
         (in-ns 'demo.strr)
         (println (str/trim \" a \"))",
    );
    assert!(redef.contains("@rt_n_main("), "{redef}");
    let (_, core) = emit(
        "core-def",
        "(in-ns 'clojure.core) (def x-271 1) (in-ns 'user) (println clojure.core/x-271)",
    );
    assert!(core.contains("@rt_n_main("), "{core}");
}

#[test]
fn native_emit_ir_lowers_scalar_loops_and_branch_conds_271() {
    let src = "\
(defn step [x]
  (let [y (+ x 1)]
    (if (> y 0) y 0)))

(defn multi
  ([x] x)
  ([x y] (+ x y)))

(defn conds-and-preds [a b]
  (let [v [1]
        _ (if true 1 2)
        _ (if false 1 2)
        _ (if (or (< a b) (<= a b)) 1 2)
        _ (if (and (> a b) (>= a b) (== a b)) 1 2)
        _ (if (if (nil? a) (some? b) (not b)) 1 2)
        _ (if (step a) 1 2)
        c (< a b)
        d (<= a b)
        e (> a b)
        f (>= a b)
        g (== a b)
        h (nil? a)
        i (some? a)
        j (not a)
        k (int a)
        l (long b)]
    (step v)
    (let [x [1]] (vec x))
    [c d e f g h i j k l]))

(defn scalar-pure [s pairs outer-n outer-box]
  (loop [i 0 acc 0 pass s]
    (if (and (< i (count s)) (<= i 100) (> 200 i) (>= 200 i) (== i i))
      (let [ch (nth s i)
            [p0 p1] (nth pairs 0)
            b (if (= ch \\a) true false)
            u (if b 1 (if (not (or (zero? i) (pos? i) (neg? i))) 2 3))
            v (+ (- (* (inc (dec (- (bit-not (int (long outer-n)))))) 1) p0) (quot p1 1))
            w (do (+ v 1) (step i))
            inner-v (loop [j 0] (if (< j 2) (recur (+ j 1)) (+ j w)))
            flag (< i 5)
            cmp-v (if flag 1 0)]
        (if (< i 1)
          (let [z (+ i 1)] (+ z 1))
          (+ i 2))
        (loop [k 0]
          (if (< k 1)
            (recur (+ k 1))
            nil))
        (recur (+ i 1) (+ acc inner-v cmp-v u) pass))
      (if (= i 999)
        :done
        (if (= i 998)
          pass
          (if (= i 997)
            acc
            (if (= i 996)
              outer-box
              (if (= i 995)
                (let [{:keys [k]} i] k)
                (if (= i 994)
                  (println i)
                  (+ acc 1))))))))))

(defn scalar-mut [n]
  (let [t (transient [10 20])]
    (loop [i 0]
      (if (< i n)
        (do
          (if (< i 1)
            (let [z (+ i 1)] (conj! t z))
            (conj! t i))
          (assoc! t 0 i)
          (pop! t)
          (recur (+ i 1)))
        (conj! t 99)))))

(defn scalar-fallbacks [n]
  (let [t (transient [1])]
    (letfn [(cell-fn [x] x)
            (use-cell [m]
              (loop [i 0]
                (if (< i m)
                  (recur (+ i 1))
                  cell-fn)))]
      [(use-cell n)
       (loop [[a b] [1 2]]
         (if a (recur [nil b]) b))
       (loop [i 0]
         (if (< i n)
           (do (conj! t i) (recur (+' i 1)))
           i))])))

(println (conds-and-preds 1 2) (scalar-pure \"ab\" [[3 4]] 5 :box) (scalar-mut 2) (scalar-fallbacks 2))
";
    let ir = tmp("scalar-271.ll");
    let report = clj_aot::build_str(
        src,
        "s271.clj",
        &tmp("nope-s271"),
        &EmitOptions {
            emit_ir: Some(ir.clone()),
            emit_obj: None,
        },
        &no_link(),
        clj_aot::BuildMode::Mixed,
    )
    .expect("scalar 271 native");
    let text = std::fs::read_to_string(&ir).expect("ir");
    std::fs::remove_file(&ir).expect("cleanup");
    assert!(!report.contains("(native:"), "{report}");
    assert!(text.contains("sc_hdr"), "{text}");
    assert!(text.contains("sc_deopt"), "{text}");
}

/// Stable vars (#271): a single-`def` var reads from its root cache —
/// borrowed by calls, cloned elsewhere, checked only where code can run
/// before the `def` — while dynamic, re-`def`ed, nested, `set!`, and
/// `#'`-taken vars keep the deref, and a program that can re-root any
/// var keeps no caches at all. Keyword invokes go through the bitcode
/// `get` with the keyword borrowed from the constant pool.
#[test]
fn native_emit_ir_caches_stable_var_roots_271() {
    let emit = |src: &str, name: &str| {
        let ir = tmp(name);
        clj_aot::build_str(
            src,
            "f.clj",
            &tmp("nope"),
            &EmitOptions {
                emit_ir: Some(ir.clone()),
                emit_obj: None,
            },
            &no_link(),
            clj_aot::BuildMode::Mixed,
        )
        .expect("native build");
        let text = std::fs::read_to_string(&ir).expect("ir");
        std::fs::remove_file(&ir).expect("cleanup");
        text
    };
    let src = "\
(declare tbl)
(defn early [x] (if x (nth tbl 0) (count [tbl])))
(def tbl [10 20 30])
(def ^:dynamic *dyn* 1)
(def ^{:dynamic (not false)} formed 2)
(def twice 1)
(def twice 2)
(defn setter [] (def inner 3))
(def taken 4)
(def m {:a 1 :b 2})
(defn late [i] (+ (nth tbl i) (:a m) (:z m 0) (get m :b) (count tbl)))
(def wrapped [tbl (get '{a 1} 'a)])
(defn bump [] (set! *dyn* 2))
(defn odd-arity [] (:k {} 1 2))
(println (early false) (late 1) *dyn* formed twice (setter) inner @#'taken (:k [1]) wrapped)
";
    let text = emit(src, "stable.ll");
    for needle in [
        "@clj_root0 = internal global [4 x i64] zeroinitializer",
        "load atomic i8, ptr @clj_root_set",
        "store atomic i8 1, ptr @clj_root_set",
        "@rt_n_consts_base",
        "kw_slow",
    ] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }
    // bump, early, late, m, odd-arity, setter, tbl, and wrapped are
    // stable (roots 0-7); the dynamic, re-`def`ed, nested, `set!`, and
    // `#'`-taken vars are not.
    assert!(text.contains("@clj_root7 = "), "{text}");
    assert!(!text.contains("@clj_root8 = "), "{text}");
    // Direct calls (#271 Slice 4b): a stable `defn` called from its own
    // unit or a later one skips the root check; a forward call keeps it.
    // Params only read as call operands keep reading the caller's array.
    let text = emit(
        "\
(declare later)
(defn first-of [w i] (nth w i))
(defn relay [w] (first-of w 0))
(defn grow [w x] (conj w x))
(defn kept [w] (let [v w] (count v)))
(defn ahead [x] (if (pos? x) (later (dec x)) 0))
(defn later [x] (+ x (ahead x)))
(println (relay [1 2]) (grow [1] 2) (kept [3]) (later 2))
",
        "calls.ll",
    );
    for needle in ["@rt_n_enter()", "@rt_n_enter_fn("] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }
    // Inline traversals (#271 Slice 4c): `doseq`, `mapv`, `vec`/`into []`
    // of `map`, and `reduce` over literal `fn`s plan a direct loop with a
    // general fallback; other shapes keep the plain call.
    let text = emit(
        "\
(def v [1 2 3])
(doseq [x v] (print x))
(println (mapv (fn [x] (* x x)) v) (vec (map (fn [x] (inc x)) (range 3))))
(println (into [] (map (fn [x] x) v)) (reduce (fn [a x] (+ a x)) 0 v))
(println (reduce (fn [a x] a) 0 '(1)) (mapv (fn [x & r] x) v) (into [1] (map (fn [x] x) v)))
(println (mapv (fn self [x] (if (pos? x) (self (dec x)) x)) v) (mapv (fn [[a]] a) [[1]]))
(println (mapv (fn ([x] x) ([x y] y)) v))
(println (map (fn [x] x) v) (dorun (map inc v)) (mapv (fn [x] (recur (dec x))) []))
",
        "loops.ll",
    );
    for needle in [
        "@rt_n_iter_plan(",
        "@rt_n_vec_item(",
        "@rt_n_vec_push(",
        "@rt_n_unreduce(",
    ] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }
    // Var metadata (#271): `(meta coll)` alone (clojure.set's `select`)
    // and constant attr-map writes (`{:added …, :tag …}`) keep none; a
    // `#'v` with a value reader keeps it.
    let set_only = emit(
        "(ns user (:require [clojure.set :as set]))\n(println (set/union #{1} #{2}) (meta (with-meta [1] {:a 1})))\n",
        "meta-off.ll",
    );
    assert!(
        !set_only.contains("call void @rt_n_cap_var_meta"),
        "{set_only}"
    );
    let reads = emit(
        "(defn f {:added \"1\"} [^String x] x)\n(println (:added (meta #'f)) (f 1))\n",
        "meta-on.ll",
    );
    assert!(reads.contains("call void @rt_n_cap_var_meta"), "{reads}");
    // Any var-rooting call anywhere disables the caches.
    let text = emit(
        "(def a [1]) (defn f [] (nth a 0)) (alter-var-root #'a conj 2) (println (f))",
        "stable-off.ll",
    );
    assert!(!text.contains("@clj_root0"), "{text}");
}
