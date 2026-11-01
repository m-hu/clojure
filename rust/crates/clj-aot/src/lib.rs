// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

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
// Handle-IR ratchet (Z3): no `unsafe` compiles in the lowering
// (no GEP anywhere — accs and str-ids only). Crate-wide forbid
// is impossible: `jit` keeps inherent MCJIT unsafety (the `MainFn`
// entry type plus the entry-lookup-and-call block), and forbid
// cannot carve out a module — the geiger gate backstops `jit`.
#[forbid(unsafe_code)]
pub mod lower;
pub mod target;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use clj_eval::plan_source;
use inkwell::context::Context;

use link::{LinkError, current_platform, find_archive, link_command, run_link};

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

/// Compile `source` (named `source_path` in spans) to the `out`
/// binary: plan → module → object → link.
pub fn compile_str(
    source: &str,
    source_path: &str,
    out: &Path,
    emit: &EmitOptions,
    link: &LinkOptions,
) -> Result<(), CompileError> {
    let program = plan_source(source).map_err(|e| CompileError::Plan(e.diagnostic(source_path)))?;
    let ctx = Context::create();
    let module = lower::to_module(&ctx, &program, source_path).map_err(CompileError::Emit)?;
    if let Some(ir_path) = &emit.emit_ir {
        std::fs::write(ir_path, module.print_to_string().to_string())
            .map_err(|e| io_err(ir_path, e))?;
    }
    if let Some(obj_path) = &emit.emit_obj {
        let machine = target::host_machine().map_err(CompileError::Target)?;
        target::emit_object(&machine, &module, obj_path).map_err(CompileError::Emit)?;
    }
    if emit.emit_ir.is_some() || emit.emit_obj.is_some() {
        return Ok(());
    }
    let machine = target::host_machine().map_err(CompileError::Target)?;
    let tmp = temp_obj_path();
    target::emit_object(&machine, &module, &tmp).map_err(CompileError::Emit)?;
    let platform = current_platform();
    let archive = find_archive(&link.exe_dir, link.runtime_archive.as_deref(), platform)
        .map_err(CompileError::Link)?;
    let cmd = link_command(platform, &tmp, &archive, out, link.linker.as_deref());
    let linked = run_link(&cmd).map_err(CompileError::Link);
    let _removed = std::fs::remove_file(&tmp);
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

    use super::temp_obj_path;

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
}
