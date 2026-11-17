// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! LLVM AOT backend for the pure Clojure subset (T1174 S8C).
//!
//! [`compile_file`] plans the source ([`clj_eval`] subset
//! planner), lowers the plan to an LLVM module ([`lower`]), emits
//! a host object ([`target`]), and links it against the `clj-rt`
//! archive ([`link`]). `--emit-ir` / `--emit-obj` stop before
//! linking (hermetic, linker-less); otherwise a temporary object
//! links and is removed. [`jit`] runs the same subset
//! in-process through an execution engine (no object, no link,
//! no temp files).

// The FFI declarations assume 64-bit `usize` (R1 finding 8): a
// 32-bit host would silently miscompile, so refuse to build one
// (a compile-time gate costs no coverage lines).
#[cfg(not(target_pointer_width = "64"))]
compile_error!("clj-aot requires a 64-bit host (usize FFI widths)");

pub mod driver;
pub mod jit;
pub mod link;
pub mod lower;
pub mod native;
pub mod target;

mod helper_bc {
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/helper_bc.rs"));
}

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use clj_eval::{plan_mixed, plan_source};
use inkwell::context::Context;

use link::{LinkError, current_platform, find_archive, link_command};

/// Emit options: `--emit-ir` / `--emit-obj` paths (either one skips
/// linking; both write and skip).
#[derive(Clone, Debug, Default)]
pub struct EmitOptions {
    /// Write LLVM IR here (skips linking).
    pub emit_ir: Option<PathBuf>,
    /// Write the object file here (skips linking).
    pub emit_obj: Option<PathBuf>,
}

/// Link options: runtime archive discovery + linker override.
#[derive(Clone, Debug, Default)]
pub struct LinkOptions {
    /// `--runtime-archive` override file (default: next to the binary).
    pub runtime_archive: Option<PathBuf>,
    /// Directory holding the current executable (archive lookup).
    pub exe_dir: PathBuf,
    /// Linker program override (default: `cc` / `link`).
    pub linker: Option<String>,
}

/// Compilation failures (the CLI maps IO/link/archive to exit 3
/// `build: ...`, plan failures to exit 1 with the diagnostic, and
/// LLVM failures to exit 1 with the message).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// The source is outside the AOT subset (rendered diagnostic).
    Plan(String),
    /// The host target machine failed (broken LLVM install).
    Target(String),
    /// Object emission failed (broken LLVM install).
    Emit(String),
    /// A file read/write failed.
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The OS error text.
        message: String,
    },
    /// Archive discovery or linking failed.
    Link(LinkError),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plan(msg) | Self::Target(msg) | Self::Emit(msg) => write!(f, "{msg}"),
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Self::Link(e) => write!(f, "{e}"),
        }
    }
}

fn io_err(path: &Path, err: std::io::Error) -> CompileError {
    CompileError::Io {
        path: path.to_path_buf(),
        message: err.to_string(),
    }
}

/// A unique temporary object path (removed after linking). The
/// counter — not the clock — makes it unique: parallel compiles
/// in one process can share a timestamp tick on coarse clocks
/// (macOS `build_exe` flake: one test linked another's `.o`).
fn temp_obj_path() -> PathBuf {
    static OBJ_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = OBJ_SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("clj-aot-{}-{seq}-{nanos}.o", std::process::id()))
}

/// How `build` treats a program outside the pure subset (issue
/// #256, spec D8).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BuildMode {
    /// Fall back to a mixed build: subset `defn`s native, the rest
    /// embedded and loaded at startup like `run` (every readable
    /// program builds).
    #[default]
    Mixed,
    /// Fail with the first unsupported form (`build --strict`, and
    /// the library's [`compile_str`]).
    Strict,
}

/// One-line helper-mode summary included in `build --report` (#260).
#[must_use]
pub fn helpers_summary() -> String {
    helper_report(
        helper_bc::HELPER_BC.is_some(),
        helper_bc::RUSTC_LLVM_VERSION,
        helper_bc::BACKEND_LLVM_MAJOR,
    )
}

fn helper_report(bitcode: bool, rustc_version: &str, backend_major: u32) -> String {
    if bitcode {
        format!("helpers: bitcode-lto (rustc LLVM {rustc_version})")
    } else {
        format!("helpers: out-of-line (rustc LLVM {rustc_version}, backend LLVM {backend_major})")
    }
}

/// Compile `source` (named `source_path` in spans) to the `out`
/// binary, falling back per `mode`. Returns the one-line coverage
/// report (`aot: native N/M defns, embedded K top-level forms, helpers: ...`).
pub fn build_str(
    source: &str,
    source_path: &str,
    out: &Path,
    emit: &EmitOptions,
    link: &LinkOptions,
    mode: BuildMode,
) -> Result<String, CompileError> {
    let ctx = Context::create();
    let helpers = helpers_summary();
    let (module, report) = match plan_source(source) {
        Ok(program) => {
            let n = program.fns.len();
            let module =
                lower::to_module(&ctx, &program, source_path).map_err(CompileError::Emit)?;
            (
                module,
                format!("aot: native {n}/{n} defns, embedded 0 top-level forms, {helpers}"),
            )
        }
        Err(e) if mode == BuildMode::Strict => {
            return Err(CompileError::Plan(e.diagnostic(source_path)));
        }
        Err(_) => match native_module(&ctx, source, source_path) {
            Ok((module, defns)) => (
                module,
                format!("aot: native {defns}/{defns} defns, embedded 0 top-level forms, {helpers}"),
            ),
            Err(fail) => {
                let program = plan_mixed(source);
                let module = lower::to_module_mixed(&ctx, &program, source_path)
                    .map_err(CompileError::Emit)?;
                (
                    module,
                    format!("{} (native: {fail}), {helpers}", program.report()),
                )
            }
        },
    };
    finish(&module, out, emit, link)?;
    Ok(report)
}

/// Why the native compiler declined a program.
#[derive(Debug)]
pub(crate) enum NativeFail {
    /// The program itself does not compile (rendered exactly as `run`
    /// reports it).
    Program(String),
    /// A construct the native compiler cannot lower (yet).
    Unsupported(String),
}

impl std::fmt::Display for NativeFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Program(diag) => write!(f, "front end: {diag}"),
            Self::Unsupported(reason) => write!(f, "{reason}"),
        }
    }
}

/// The full native compiler (issue #259): front end, then lowering
/// of every unit. Returns the module and the top-level `defn` count,
/// or why the program cannot be compiled natively (yet).
pub(crate) fn native_module<'ctx>(
    ctx: &'ctx Context,
    source: &str,
    source_path: &str,
) -> Result<(inkwell::module::Module<'ctx>, usize), NativeFail> {
    native_module_with_helpers(ctx, source, source_path, helper_bc::HELPER_BC)
}

pub(crate) fn native_module_with_helpers<'ctx>(
    ctx: &'ctx Context,
    source: &str,
    source_path: &str,
    helpers: Option<&[u8]>,
) -> Result<(inkwell::module::Module<'ctx>, usize), NativeFail> {
    let mut eval = clj_eval::boot();
    let front = clj_eval::aot_front::front_end(&mut eval, source, source_path)
        .map_err(|e| NativeFail::Program(clj_eval::script::run_diagnostic(source_path, &e)))?;
    // The program's own definitions (spliced library units aside, #271).
    let defns = front.units[front.lib_units..]
        .iter()
        .filter(|u| matches!(&u.expr, clj_eval::expr::Expr::Def { init: Some(_), .. }))
        .count();
    // The build evaluator is torn down by leaking, like `run` does.
    std::mem::forget(eval);
    let module = native::Lowerer::with_bitcode(ctx, helpers.is_some())
        .lower_program(&front, source_path)
        .map_err(|native::Unsupported(r)| NativeFail::Unsupported(r))?;
    let machine = target::host_machine().map_err(NativeFail::Unsupported)?;
    let triple = inkwell::targets::TargetMachine::get_default_triple();
    let layout = machine.get_target_data().get_data_layout();
    module.set_triple(&triple);
    module.set_data_layout(&layout);
    if let Some(bc) = helpers {
        link_helpers(ctx, &module, bc, &triple, &layout).map_err(NativeFail::Unsupported)?;
    }
    // Native modules are optimized (inlining, mem2reg over scalars,
    // CFG simplification of the inline value paths + linked helpers).
    optimize_module(&module, &machine, "default<O2>")?;
    Ok((module, defns))
}

fn optimize_module(
    module: &inkwell::module::Module<'_>,
    machine: &inkwell::targets::TargetMachine,
    passes: &str,
) -> Result<(), NativeFail> {
    module
        .run_passes(
            passes,
            machine,
            inkwell::passes::PassBuilderOptions::create(),
        )
        .map_err(|e| NativeFail::Unsupported(e.to_string()))
}

fn helper_exports_to_internalize<'ctx>(helper: &inkwell::module::Module<'ctx>) -> Vec<String> {
    let mut names = Vec::new();
    for func in helper.get_functions() {
        let gv = func.as_global_value();
        if gv.is_declaration() {
            continue;
        }
        func.remove_string_attribute(inkwell::attributes::AttributeLoc::Function, "target-cpu");
        func.remove_string_attribute(
            inkwell::attributes::AttributeLoc::Function,
            "target-features",
        );
        let already_local = matches!(
            func.get_linkage(),
            inkwell::module::Linkage::Internal | inkwell::module::Linkage::Private
        );
        if gv.get_comdat().is_none() && !already_local {
            names.push(func.get_name().to_string_lossy().into_owned());
        }
    }
    names
}

fn mark_internal_inline<'ctx>(
    ctx: &'ctx Context,
    module: &inkwell::module::Module<'ctx>,
    names: &[String],
) {
    let inline_kind = inkwell::attributes::Attribute::get_named_enum_kind_id("inlinehint");
    let inline_attr = (inline_kind != 0).then(|| ctx.create_enum_attribute(inline_kind, 0));
    for name in names {
        let Some(func) = module.get_function(name) else {
            continue;
        };
        func.set_linkage(inkwell::module::Linkage::Internal);
        if name.starts_with("rt_bc_")
            && let Some(attr) = inline_attr
        {
            func.add_attribute(inkwell::attributes::AttributeLoc::Function, attr);
        }
    }
}

fn link_helpers<'ctx>(
    ctx: &'ctx Context,
    module: &inkwell::module::Module<'ctx>,
    bc: &[u8],
    triple: &inkwell::targets::TargetTriple,
    layout: &inkwell::data_layout::DataLayout,
) -> Result<(), String> {
    let buf = inkwell::memory_buffer::MemoryBuffer::create_from_memory_range(bc, "clj_bc");
    let helper =
        inkwell::module::Module::parse_bitcode_from_buffer(&buf, ctx).map_err(|e| e.to_string())?;
    helper.set_triple(triple);
    helper.set_data_layout(layout);
    let to_internalize = helper_exports_to_internalize(&helper);
    module.link_in_module(helper).map_err(|e| e.to_string())?;
    mark_internal_inline(ctx, module, &to_internalize);
    module.verify().map_err(|e| e.to_string())
}

/// Compile `source` (named `source_path` in spans) to the `out`
/// binary: plan → module → object → link, failing on the first
/// form outside the pure subset ([`BuildMode::Strict`]).
pub fn compile_str(
    source: &str,
    source_path: &str,
    out: &Path,
    emit: &EmitOptions,
    link: &LinkOptions,
) -> Result<(), CompileError> {
    build_str(source, source_path, out, emit, link, BuildMode::Strict).map(|_| ())
}

/// Emit the requested artifacts, or link the binary when none was
/// requested (a temporary object, removed after linking).
fn finish(
    module: &inkwell::module::Module<'_>,
    out: &Path,
    emit: &EmitOptions,
    link: &LinkOptions,
) -> Result<(), CompileError> {
    if let Some(ir_path) = &emit.emit_ir {
        std::fs::write(ir_path, module.print_to_string().to_string())
            .map_err(|e| io_err(ir_path, e))?;
    }
    if let Some(obj_path) = &emit.emit_obj {
        let machine = target::host_machine().map_err(CompileError::Target)?;
        target::emit_object(&machine, module, obj_path).map_err(CompileError::Emit)?;
    }
    if emit.emit_ir.is_some() || emit.emit_obj.is_some() {
        return Ok(());
    }
    let machine = target::host_machine().map_err(CompileError::Target)?;
    let tmp = temp_obj_path();
    target::emit_object(&machine, module, &tmp).map_err(CompileError::Emit)?;
    let platform = current_platform();
    let archive = find_archive(&link.exe_dir, link.runtime_archive.as_deref(), platform)
        .map_err(CompileError::Link)?;
    let linux = cfg!(target_os = "linux");
    let script_path = tmp.with_extension("ld");
    let script =
        link::write_layout_script(&script_path, linux).map_err(|e| io_err(&script_path, e))?;
    let mut cmd = link_command(platform, &tmp, &archive, out, link.linker.as_deref());
    cmd.args.extend(link::layout_args(script, linux));
    cmd.args.extend(link::gc_section_args(
        cfg!(target_os = "linux"),
        cfg!(target_os = "macos"),
    ));
    // Linux tries the static link first (see `link::static_args`); the
    // plain command stays the fallback (and the only attempt elsewhere).
    let mut attempts = vec![cmd.clone()];
    attempts[0].args.extend(link::static_args(linux));
    attempts.push(cmd);
    let linked = link::run_link_first(&attempts).map_err(CompileError::Link);
    let _removed = std::fs::remove_file(&tmp);
    let _removed = script.map(std::fs::remove_file);
    linked
}

/// Compile the file at `path` to the `out` binary.
pub fn compile_file(
    path: &Path,
    out: &Path,
    emit: &EmitOptions,
    link: &LinkOptions,
) -> Result<(), CompileError> {
    match std::fs::read_to_string(path) {
        Ok(source) => compile_str(&source, &path.to_string_lossy(), out, emit, link),
        Err(e) => Err(io_err(path, e)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::thread;

    use inkwell::context::Context;
    use inkwell::targets::TargetMachine;

    use super::{
        helper_bc, helper_report, helpers_summary, link_helpers, mark_internal_inline,
        native_module_with_helpers, target, temp_obj_path,
    };

    #[test]
    fn temp_obj_paths_are_unique_across_threads() {
        let paths: Vec<_> = thread::scope(|s| {
            (0..32)
                .map(|_| s.spawn(temp_obj_path))
                .map(|h| h.join().expect("thread panicked"))
                .collect()
        });
        let uniq: HashSet<_> = paths.iter().collect();
        assert_eq!(paths.len(), uniq.len(), "duplicate temp paths: {paths:?}");
    }

    #[test]
    fn helper_report_and_fallback_path() {
        assert_eq!(
            helper_report(true, "23.1.1", 23),
            "helpers: bitcode-lto (rustc LLVM 23.1.1)"
        );
        assert_eq!(
            helper_report(false, "22.1.0", 23),
            "helpers: out-of-line (rustc LLVM 22.1.0, backend LLVM 23)"
        );
        let summary = helpers_summary();
        assert!(summary.contains("rustc LLVM "));

        let src = "(def a \"hello\")\n(defn f [v x] (if (= a \"hello\") (conj v x) (count v)))\n(println (f [1] :k))\n";
        let ctx = Context::create();
        let (out_mod, _) =
            native_module_with_helpers(&ctx, src, "t.clj", None).expect("out-of-line");
        let out_ir = out_mod.print_to_string().to_string();
        assert!(!out_ir.contains("@rt_bc_"), "{out_ir}");
        assert!(out_ir.contains("@rt_n_deref"), "{out_ir}");

        let bc = helper_bc::HELPER_BC.expect("helper bitcode on pinned LLVM 23");
        assert_eq!(
            helper_bc::RUSTC_LLVM_MAJOR,
            Some(helper_bc::BACKEND_LLVM_MAJOR)
        );
        let (lto_mod, _) =
            native_module_with_helpers(&ctx, src, "t.clj", Some(bc)).expect("bitcode-lto");
        let lto_ir = lto_mod.print_to_string().to_string();
        assert!(lto_ir.contains("@rt_n_fast_vars"), "{lto_ir}");
        assert!(!lto_ir.contains("declare i8 @rt_bc_"), "{lto_ir}");
        // Small bitcode ops inline completely into the caller and strip away.
        assert!(!lto_ir.contains("@rt_bc_is_pod"), "{lto_ir}");
        assert!(!lto_ir.contains("@rt_bc_1_count"), "{lto_ir}");
        assert!(!lto_ir.contains("@rt_bc_2_eq"), "{lto_ir}");
        // Shared multi-site clone/deref helpers are internalized cleanly.
        assert!(!lto_ir.contains("define i8 @rt_bc_"), "{lto_ir}");
        assert!(!lto_ir.contains("panicking"), "{lto_ir}");

        let bad_ctx = Context::create();
        let bad_mod = bad_ctx.create_module("empty");
        mark_internal_inline(&bad_ctx, &bad_mod, &["missing_helper".to_owned()]);
        let machine = target::host_machine().expect("machine");
        let triple = TargetMachine::get_default_triple();
        let layout = machine.get_target_data().get_data_layout();
        assert!(link_helpers(&bad_ctx, &bad_mod, b"bad-bc", &triple, &layout).is_err());
        assert!(
            native_module_with_helpers(&bad_ctx, "(println 1)", "t.clj", Some(b"bad-bc")).is_err()
        );
        assert!(super::optimize_module(&bad_mod, &machine, "definitely-invalid-pass").is_err());
    }
}
