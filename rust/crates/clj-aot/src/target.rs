// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Host target machine + object emission (S8C).
//!
//! Host-only: x86/aarch64 init (never `target-all`), the default
//! triple, generic CPU, PIC reloc (position-independent objects
//! link everywhere, including macOS ARM64 PIE binaries).

use std::path::Path;

use inkwell::OptimizationLevel;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};

/// Build the host target machine (fails only when LLVM itself is
/// broken — the host triple always resolves after x86/aarch64
/// init on every supported runner).
pub fn host_machine() -> Result<TargetMachine, String> {
    machine_for(&TargetMachine::get_default_triple())
}

fn machine_for(triple: &inkwell::targets::TargetTriple) -> Result<TargetMachine, String> {
    Target::initialize_x86(&InitializationConfig::default());
    Target::initialize_aarch64(&InitializationConfig::default());
    let target = Target::from_triple(triple).map_err(|e| e.to_string())?;
    Ok(target
        .create_target_machine(
            triple,
            "generic",
            "",
            OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .expect("initialized target creates host machine"))
}

/// Emit `module` into an object buffer (stamps default triple & layout).
pub fn emit_object_buffer<'ctx>(
    machine: &TargetMachine,
    module: &Module<'ctx>,
) -> inkwell::memory_buffer::MemoryBuffer<'ctx> {
    let triple = TargetMachine::get_default_triple();
    split_function_sections(module, is_elf(&triple.as_str().to_string_lossy()));
    module.set_triple(&triple);
    let layout = machine.get_target_data().get_data_layout();
    module.set_data_layout(&layout);
    machine
        .write_to_memory_buffer(module, FileType::Object)
        .expect("host machine emits object for verified module")
}

/// Whether objects for `triple` are ELF (Linux and other Unix targets;
/// Mach-O and COFF name sections differently).
fn is_elf(triple: &str) -> bool {
    !(triple.contains("apple") || triple.contains("darwin") || triple.contains("windows"))
}

/// On ELF, every function the module defines in its own `.text.<name>`
/// section, as `-ffunction-sections` places Rust's (#271 Slice 6p). The
/// hot-text linker script orders functions by such sections: with the
/// whole program in one `.text`, the order's entries for the program
/// object's own functions (`main`, the frame-info wrappers, the bitcode
/// helpers linked in, which every workload profile names) never matched,
/// so those stayed in cold `.text`, each faulting in a 64 KiB window.
fn split_function_sections(module: &Module<'_>, elf: bool) {
    if !elf {
        return;
    }
    for f in module
        .get_functions()
        .filter(|f| f.count_basic_blocks() > 0)
    {
        let section = format!(".text.{}", f.get_name().to_string_lossy());
        f.as_global_value().set_section(Some(&section));
    }
}

/// Emit `module` as a native object file (stamps the module triple
/// and data layout from the machine first).
pub fn emit_object(
    machine: &TargetMachine,
    module: &Module,
    obj_path: &Path,
) -> Result<(), String> {
    let buffer = emit_object_buffer(machine, module);
    std::fs::write(obj_path, buffer.as_slice()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{emit_object, host_machine};
    use inkwell::context::Context;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("clj-aot-target-{}-{name}", std::process::id()))
    }

    #[test]
    fn host_machine_emits_objects() {
        let machine = host_machine().expect("host machine");
        let ctx = Context::create();
        let module = ctx.create_module("empty");
        let obj = tmp("empty.o");
        emit_object(&machine, &module, &obj).expect("emit");
        let bytes = std::fs::read(&obj).expect("read back");
        assert!(!bytes.is_empty(), "object must not be empty");
        std::fs::remove_file(&obj).expect("cleanup");
    }

    #[test]
    fn emit_reports_unwritable_paths() {
        let machine = host_machine().expect("host machine");
        let ctx = Context::create();
        let module = ctx.create_module("empty");
        let obj = tmp("no-such-dir").join("x.o");
        assert!(emit_object(&machine, &module, &obj).is_err());
    }

    #[test]
    fn elf_objects_give_each_defined_function_its_section_271() {
        assert!(super::is_elf("x86_64-unknown-linux-gnu"));
        assert!(!super::is_elf("arm64-apple-darwin25.0.0"));
        assert!(!super::is_elf("x86_64-pc-windows-msvc"));
        let ctx = Context::create();
        let module = ctx.create_module("sections");
        let ty = ctx.void_type().fn_type(&[], false);
        let defined = module.add_function("defined_fn", ty, None);
        let builder = ctx.create_builder();
        builder.position_at_end(ctx.append_basic_block(defined, "entry"));
        builder.build_return(None).expect("ret");
        let declared = module.add_function("declared_fn", ty, None);
        super::split_function_sections(&module, false);
        assert_eq!(defined.as_global_value().get_section(), None);
        super::split_function_sections(&module, true);
        let global = defined.as_global_value();
        let section = global
            .get_section()
            .map(|s| s.to_string_lossy().into_owned());
        assert_eq!(section.as_deref(), Some(".text.defined_fn"));
        assert_eq!(declared.as_global_value().get_section(), None);
    }

    #[test]
    fn machine_for_reports_uninitialized_triple() {
        let bad_triple = inkwell::targets::TargetTriple::create("riscv64-unknown-none");
        assert!(super::machine_for(&bad_triple).is_err());
    }
}
