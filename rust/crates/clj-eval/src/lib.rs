// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Analyzer and tree-walking interpreter for Clojure (T1174 S4).
//!
//! [`Analyzer`] turns reader values into [`Expr`]s enforcing the
//! `docs/t1174/evaluator-contract.md` shape rules; [`Evaluator`]
//! runs them. Single namespace (`user`), no prelude yet (S5).

pub mod aot_subset;
pub mod arrays;
#[path = "async.rs"]
pub mod async_ns;
pub mod builtin_expand;
pub mod classemu;
pub mod coll;
pub mod collmisc;
pub mod compiler;
pub mod concurrency;
pub mod core;
pub mod destructure_expand;
pub mod env;
pub mod error;
pub mod eval;
pub mod expr;
pub mod ext;
pub mod fmt;
pub mod fx4_form;
pub mod fx4_host;
pub mod hierarchy;
pub mod io;
pub mod jdkio;
pub mod lazy;
pub mod macros;
pub mod multi;
pub mod ns;
pub mod portdeps;
pub(crate) mod printsys;
pub mod protocol;
pub mod re;
pub mod reader;
pub mod rng;
pub mod scalars;
pub mod seqs;
pub mod stm;
pub mod typeinternals;
pub mod types;
pub mod vars;
pub mod xform;

pub use aot_subset::{
    PlanError, Prim, SubsetError, SubsetExpr, SubsetFn, SubsetProgram, plan_source,
};
pub use env::{Bindings, Env, Namespace, NamespaceRegistry, SharedRegistry, VarState};
pub use error::{
    CompileError, NativeError, Signal, Verb, compile_diagnostic, compile_head, jvm_class_of,
    simple_class_of, span_of,
};
pub use eval::{Evaluator, SourceError};
pub use expr::{
    Analyzer, Arity, Catch, Expr, FunExpr, KeyBinding, MapPattern, Pattern, VecPattern,
};
pub use ext::{Closure, NativeFn, NativeImpl, NsHandle, as_closure, as_native, as_ns, munge};
pub use ns::boot;
