// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! In-process LLVM JIT over the AOT subset (gap-121 slice 1).
//!
//! [`run_str`] plans the source ([`clj_eval`] subset planner),
//! lowers it in JIT return mode ([`to_module_jit`]), creates an
//! MCJIT execution engine over the module, maps every `rt_*`
//! declaration to its in-process [`cljrt`] address, and calls
//! the C `main` — returning its exit code. stdout/stderr flow
//! to the process streams, like the AOT binary; nothing is
//! written to disk (no temp files, no archive, no linker).

use std::ffi::{CString, c_char};

use clj_eval::{Prim, plan_source};
use cljrt::prim_symbol;
use inkwell::OptimizationLevel;
use inkwell::context::Context;

use crate::lower::to_module_jit;

/// The JITted C `main(argc, argv) -> i32` signature.
type MainFn = unsafe extern "C" fn(i32, *const *const c_char) -> i32;

/// JIT-run failures (the CLI maps both to exit 1: `Plan`
/// prints verbatim, `Jit` arrives pre-prefixed).
#[derive(Debug)]
pub enum JitError {
    /// The source is outside the AOT subset (rendered diagnostic).
    Plan(String),
    /// Lowering, EE creation, or entry lookup failed (`jit: …`).
    Jit(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plan(msg) | Self::Jit(msg) => write!(f, "{msg}"),
        }
    }
}

/// Prefix an LLVM failure as a [`JitError::Jit`] (single line:
/// the failure arm is environmentally untriggerable, so the
/// construction shares its line with the always-executed call).
fn jit_err<T, E: std::fmt::Display>(result: Result<T, E>, what: &str) -> Result<T, JitError> {
    result.map_err(|e| JitError::Jit(format!("jit: {what}: {e}")))
}

/// The `rt_*` symbol table: every helper `lower.rs` declares
/// paired with its in-process address (branchless literal — an
/// exhaustive `Prim` match would trip the complexity gate).
/// Prim symbols come from [`prim_symbol`], the single source
/// of truth; `exit` is deliberately absent (declared but never
/// emitted in JIT mode, so MCJIT never resolves it), and so are
/// the test-only `rt_str_finish`/`rt_live_count` (never declared).
fn mappings() -> [(&'static str, *const ()); 32] {
    [
        (prim_symbol(Prim::Add), cljrt::rt_add as *const ()),
        (prim_symbol(Prim::Sub), cljrt::rt_sub as *const ()),
        (prim_symbol(Prim::Mul), cljrt::rt_mul as *const ()),
        (prim_symbol(Prim::Quot), cljrt::rt_quot as *const ()),
        (prim_symbol(Prim::Lt), cljrt::rt_lt as *const ()),
        (prim_symbol(Prim::Le), cljrt::rt_le as *const ()),
        (prim_symbol(Prim::Gt), cljrt::rt_gt as *const ()),
        (prim_symbol(Prim::Ge), cljrt::rt_ge as *const ()),
        (prim_symbol(Prim::Eq), cljrt::rt_eq as *const ()),
        (prim_symbol(Prim::Count), cljrt::rt_count as *const ()),
        (prim_symbol(Prim::Nth), cljrt::rt_nth as *const ()),
        (prim_symbol(Prim::Get), cljrt::rt_get as *const ()),
        (prim_symbol(Prim::Conj), cljrt::rt_conj as *const ()),
        ("rt_nil", cljrt::rt_nil as *const ()),
        ("rt_bool", cljrt::rt_bool as *const ()),
        ("rt_int", cljrt::rt_int as *const ()),
        ("rt_double", cljrt::rt_double as *const ()),
        ("rt_str_new", cljrt::rt_str_new as *const ()),
        ("rt_str_chunk", cljrt::rt_str_chunk as *const ()),
        ("rt_str_register", cljrt::rt_str_register as *const ()),
        ("rt_str_get", cljrt::rt_str_get as *const ()),
        ("rt_keyword", cljrt::rt_keyword as *const ()),
        ("rt_vec_new", cljrt::rt_vec_new as *const ()),
        ("rt_vec_push", cljrt::rt_vec_push as *const ()),
        ("rt_clone", cljrt::rt_clone as *const ()),
        ("rt_drop", cljrt::rt_drop as *const ()),
        ("rt_args_new", cljrt::rt_args_new as *const ()),
        ("rt_args_push", cljrt::rt_args_push as *const ()),
        ("rt_argv", cljrt::rt_argv as *const ()),
        ("rt_print", cljrt::rt_print as *const ()),
        ("rt_arity_error", cljrt::rt_arity_error as *const ()),
        ("rt_is_truthy", cljrt::rt_is_truthy as *const ()),
    ]
}

/// Borrow-stable C argv: a static dummy program name (skipped
/// by `rt_argv`, like an AOT binary's `argv[0]`) plus the args
/// (`NUL` in an arg panics, like `std::process::Command`).
fn c_argv(args: &[String]) -> Vec<CString> {
    let mut owned = Vec::with_capacity(args.len() + 1);
    owned.push(c"jit".to_owned());
    for arg in args {
        owned.push(CString::new(arg.as_str()).expect("jit argv entries admit no NUL"));
    }
    owned
}

/// JIT-run `source` (named `source_path` in spans) in-process:
/// plan → JIT module → execution engine → map `rt_*` → call
/// C `main` with `args` → return its exit code (0 or 1).
pub fn run_str(source: &str, source_path: &str, args: &[String]) -> Result<i32, JitError> {
    let program = plan_source(source).map_err(|e| JitError::Plan(e.diagnostic(source_path)))?;
    let ctx = Context::create();
    let module = jit_err(to_module_jit(&ctx, &program, source_path), "lower")?;
    // Split across two statements (not one call): the `?`
    // early-return is environmentally untriggerable, so it must
    // share its line with the always-executed call or llvm-cov
    // marks the bare `)?;` line 0/140 — and rustfmt's call-width
    // rule forbids the one-liner.
    let engine = module.create_jit_execution_engine(OptimizationLevel::None);
    let ee = jit_err(engine, "engine")?;
    for (name, addr) in mappings() {
        let func = module
            .get_function(name)
            .expect("lowered module declares every mapped helper");
        ee.add_global_mapping(&func, addr as usize);
    }
    let owned = c_argv(args);
    let mut ptrs: Vec<*const c_char> = owned.iter().map(|c| c.as_ptr()).collect();
    ptrs.push(std::ptr::null());
    let argc = owned.len() as i32;
    // Safety: `to_module_jit` always emits C `main(argc, argv)
    // -> i32` (verified before return), so `MainFn` matches the
    // looked-up entry; `argv` holds `argc` valid C strings plus
    // a NUL terminator, alive across the call.
    let code =
        unsafe { jit_err(ee.get_function::<MainFn>("main"), "entry")?.call(argc, ptrs.as_ptr()) };
    Ok(code)
}

#[cfg(test)]
mod tests {
    use clj_eval::plan_source;
    use cljrt::prim_symbol;
    use inkwell::OptimizationLevel;
    use inkwell::context::Context;

    use super::{JitError, mappings, run_str};
    use crate::lower::{PRIM_ORDER, to_module, to_module_jit};

    /// Lower `source` in JIT return mode and return verified IR.
    fn compile_ir_jit(source: &str) -> String {
        let program = plan_source(source).expect("plan");
        let ctx = Context::create();
        let module = to_module_jit(&ctx, &program, "jit.clj").expect("verify");
        module.print_to_string().to_string()
    }

    /// Lower `source` in AOT exit mode and return verified IR.
    fn compile_ir_aot(source: &str) -> String {
        let program = plan_source(source).expect("plan");
        let ctx = Context::create();
        let module = to_module(&ctx, &program, "aot.clj").expect("verify");
        module.print_to_string().to_string()
    }

    #[test]
    fn s1_spike_creates_execution_engine_over_empty_module() {
        // Gap-121 spike S1 (recon §2): MCJIT ships on the LLVM 22
        // pin, so EE creation is expected to pass — a failure
        // here (not a code bug) triggers fallback B.
        let ctx = Context::create();
        let module = ctx.create_module("s1");
        let _ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("S1: EE creation on the pin");
    }

    #[test]
    fn jit_aborts_return_while_aot_aborts_exit() {
        let source = "(defn f [x] (+ x 1)) (print (f 41) (quot 1 0))";
        let jit = compile_ir_jit(source);
        assert!(jit.contains("ret i32 1"), "{jit}");
        assert!(jit.contains("ret i64 0"), "{jit}");
        assert!(!jit.contains("call void @exit"), "{jit}");
        let aot = compile_ir_aot(source);
        assert!(aot.contains("call void @exit(i32 1)"), "{aot}");
        assert!(!aot.contains("ret i32 1"), "{aot}");
    }

    #[test]
    fn mapping_table_covers_every_declaration() {
        // Drift guard: every helper the lowering declares must
        // resolve in the JIT table — except `exit`, which JIT
        // codegen never emits (declared, unmapped, unreferenced).
        let program = plan_source("(print 1)").expect("plan");
        let ctx = Context::create();
        let module = to_module_jit(&ctx, &program, "drift.clj").expect("verify");
        let table = mappings();
        let mut decls = 0;
        let mut func = module.get_first_function();
        while let Some(f) = func {
            if f.count_basic_blocks() == 0 {
                let name = f.get_name().to_str().expect("helper name");
                if name != "exit" {
                    assert!(
                        table.iter().any(|(mapped, _)| *mapped == name),
                        "unmapped declaration: {name}"
                    );
                    decls += 1;
                }
            }
            func = f.get_next_function();
        }
        assert_eq!(decls, 32, "expected all 32 helpers declared");
        for prim in PRIM_ORDER {
            let symbol = prim_symbol(prim);
            assert!(
                table.iter().any(|(mapped, _)| *mapped == symbol),
                "unmapped prim: {symbol}"
            );
        }
    }

    #[test]
    fn entry_and_exit_stay_unmapped() {
        // Defined functions (`main`, siblings) and `exit` are
        // never mapped: definitions resolve internally, and no
        // JIT codegen path emits an `exit` call.
        let table = mappings();
        for symbol in ["exit", "main", "cljfn_f_0", "rt_bogus"] {
            assert!(
                !table.iter().any(|(mapped, _)| *mapped == symbol),
                "{symbol} must not be in the mapping table"
            );
        }
    }

    #[test]
    fn errors_render_for_the_cli() {
        // `Plan` verbatim (the interpreter-identical diagnostic);
        // `Jit` arrives pre-prefixed (construction sites own the
        // `jit: ` prefix, so one Display arm serves both).
        let plan = JitError::Plan("f.clj:1:1: boom".to_owned());
        let jit = JitError::Jit("jit: engine: boom".to_owned());
        assert_eq!(plan.to_string(), "f.clj:1:1: boom");
        assert_eq!(jit.to_string(), "jit: engine: boom");
        assert!(format!("{plan:?}").contains("Plan"));
        assert!(format!("{jit:?}").contains("Jit"));
    }

    #[test]
    fn subset_violations_return_the_verbatim_diagnostic() {
        let err = run_str("(def x 1)", "f.clj", &[]).expect_err("outside the subset");
        assert!(matches!(err, JitError::Plan(_)), "{err:?}");
        assert_eq!(
            err.to_string(),
            "f.clj:1:1: unsupported form in AOT subset: def (only defn defines in the AOT subset)"
        );
    }
}
