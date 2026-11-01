// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! AOT runtime archive for compiled Clojure binaries (T1174 S8B).
//!
//! [`abi`] exposes the `extern "C"` helper ABI over `u64` handles
//! (Z3 — no pointer crosses the boundary); [`runtime`] holds the
//! process-global booted evaluator the primitive helpers delegate
//! to, plus the handle table folded in under the single lock. The
//! crate links LLVM-free (`clj-eval + clj-value` only) so built
//! binaries stay small and link anywhere.
//!
//! Ownership (S8 plan D7/D8): helper arguments are consumed, the
//! return is owned, [`abi::rt_clone`] bumps (inline scalars:
//! identity), [`abi::rt_drop`] frees, and generated code clones
//! locals on every use and drops all frame slots at scope end.
//! Fallible helpers return `0` on error (after printing
//! `path:line:col: Class: message`); the generated code
//! zero-checks and exits 1 — the library itself never exits, so
//! every line stays unit-coverable.

// Literal-zero ratchet (Z3): `handles` and `runtime` carry
// `#![forbid(unsafe_code)]` (see those files). The crate root
// cannot: `#[unsafe(no_mangle)]` counts as `unsafe_code`, so a
// crate-level forbid rejects the ABI exports themselves (probed
// both spellings × editions 2021/2024 — all fail). `abi` stays
// lint-clean by construction instead, backstopped by the geiger
// gate (any `unsafe` there grows the workspace total and fails
// CI) plus eyeball review of the `def_prim!` expansion.

// The shims stay out of the libtest binary: their `#[no_mangle]`
// names would collide with the rlib copies linked into the
// integration binaries (same symbol, different compilation), and
// llvm-cov drops the merged records as mismatched. Unit tests
// never call shims (they pin `runtime` directly); integration
// tests link the normal rlib, where the shims exist.
#[cfg(not(test))]
pub mod abi;
pub(crate) mod handles;
pub(crate) mod runtime;

#[cfg(not(test))]
pub use abi::{
    prim_symbol, rt_add, rt_args_new, rt_args_push, rt_argv, rt_arity_error, rt_bool, rt_clone,
    rt_conj, rt_count, rt_double, rt_drop, rt_eq, rt_ge, rt_get, rt_gt, rt_int, rt_is_truthy,
    rt_keyword, rt_le, rt_live_count, rt_lt, rt_mul, rt_nil, rt_nth, rt_print, rt_quot,
    rt_str_chunk, rt_str_finish, rt_str_get, rt_str_new, rt_str_register, rt_sub, rt_vec_new,
    rt_vec_push, test_take,
};
