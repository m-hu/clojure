// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Plan → LLVM module lowering (S8C; D1/D2/D7/D8).
//!
//! Every value is a `u64` handle (`i64` in IR) to an owned
//! runtime value. One LLVM function per `defn` (`extern`
//! runtime helpers per primitive call); `@main` takes C
//! `argc`/`argv` (ignored — `rt_argv` reads the process
//! environment), drives the tops, invokes `-main` with the argv
//! list when defined, and returns 0. A prologue registers every
//! interned string literal under its str-id before any use.
//! Locals clone on every use; non-param frame slots start
//! nil (a `let` in an untaken branch never stores, so without
//! the init the scope-end drop would free stack garbage);
//! `let` drops the slot's old handle before storing; all frame
//! slots drop at scope end; fallible calls zero-check into an
//! abort block that calls libc `exit(1)`.
//!
//! Builder calls use `.expect()`: the IR is correct by
//! construction (verified by every snapshot test), so a builder
//! failure is a compiler bug and panics loudly at build time.
//! Every builder call here is a safe `fn` (no GEP anywhere —
//! arg lists ride an accumulator, not an array), pinned by the
//! module's `#![forbid(unsafe_code)]`.

use std::collections::HashMap;

use clj_eval::{Prim, SubsetExpr, SubsetFn, SubsetProgram};
use cljrt::prim_symbol;
use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::types::{FloatType, IntType, PointerType};
use inkwell::values::{BasicMetadataValueEnum, FunctionValue, IntValue, PointerValue};

/// Prims in discriminant order (index `== prim as usize`;
/// shared with the JIT symbol table so a new prim cannot miss
/// a mapping silently — the drift test walks this list).
pub(crate) const PRIM_ORDER: [Prim; 13] = [
    Prim::Add,
    Prim::Sub,
    Prim::Mul,
    Prim::Quot,
    Prim::Lt,
    Prim::Le,
    Prim::Gt,
    Prim::Ge,
    Prim::Eq,
    Prim::Count,
    Prim::Nth,
    Prim::Get,
    Prim::Conj,
];

/// The LLVM symbol per planned `defn`: sanitized name plus source
/// index (the index disambiguates; the prefix keeps clear of the
/// `rt_*` runtime and `main`).
#[must_use]
pub fn fn_symbol(index: usize, name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("cljfn_{clean}_{index}")
}

/// Declared runtime helpers (resolved once per module).
struct Decls<'ctx> {
    prims: [FunctionValue<'ctx>; 13],
    nil: FunctionValue<'ctx>,
    boolean: FunctionValue<'ctx>,
    int: FunctionValue<'ctx>,
    double: FunctionValue<'ctx>,
    str_new: FunctionValue<'ctx>,
    str_chunk: FunctionValue<'ctx>,
    str_register: FunctionValue<'ctx>,
    str_get: FunctionValue<'ctx>,
    keyword: FunctionValue<'ctx>,
    vec_new: FunctionValue<'ctx>,
    vec_push: FunctionValue<'ctx>,
    clone: FunctionValue<'ctx>,
    drop_: FunctionValue<'ctx>,
    args_new: FunctionValue<'ctx>,
    args_push: FunctionValue<'ctx>,
    argv: FunctionValue<'ctx>,
    print: FunctionValue<'ctx>,
    arity_error: FunctionValue<'ctx>,
    is_truthy: FunctionValue<'ctx>,
    exit: FunctionValue<'ctx>,
}

/// The lowering session: one module, one builder, interned string
/// ids (registered by the `main` prologue), the baked source path,
/// and sibling declarations.
struct Lower<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    ptr: PointerType<'ctx>,
    i8t: IntType<'ctx>,
    i32t: IntType<'ctx>,
    i64t: IntType<'ctx>,
    f64t: FloatType<'ctx>,
    decls: Decls<'ctx>,
    str_ids: HashMap<String, u64>,
    next_str_id: u64,
    path: String,
    fns: Vec<FunctionValue<'ctx>>,
    /// Whether an arity abort diverged the current straight-line
    /// span (once set, everything downstream is unreachable until
    /// the next `if` join resets it per branch).
    diverged: bool,
    /// JIT return mode (gap-121 slice 1): abort sites drop the
    /// frame slots and return (null from a `defn`, 1 from
    /// `main`) instead of calling `exit(1)` — exiting is fatal
    /// in-process, while callers already null-check every call
    /// result, so null propagates frame by frame to `main`.
    jit: bool,
    /// The function under construction returns `i32` (`main`)
    /// rather than `ptr` (a `defn`) — set at function entry,
    /// read by abort sites in JIT mode only.
    in_main: bool,
}

impl<'ctx> Lower<'ctx> {
    /// New session over a fresh module (declares every helper,
    /// interns the baked source path).
    fn new(ctx: &'ctx Context, source_path: &str) -> Self {
        let module = ctx.create_module("clj");
        let builder = ctx.create_builder();
        let ptr = ctx.ptr_type(AddressSpace::default());
        let i8t = ctx.i8_type();
        let i32t = ctx.i32_type();
        let i64t = ctx.i64_type();
        let f64t = ctx.f64_type();
        let voidt = ctx.void_type();
        let prim_ty = i64t.fn_type(&[i64t.into(), i64t.into(), i32t.into(), i32t.into()], false);
        let decls = Decls {
            prims: PRIM_ORDER.map(|prim| module.add_function(prim_symbol(prim), prim_ty, None)),
            nil: module.add_function("rt_nil", i64t.fn_type(&[], false), None),
            boolean: module.add_function("rt_bool", i64t.fn_type(&[i8t.into()], false), None),
            int: module.add_function("rt_int", i64t.fn_type(&[i64t.into()], false), None),
            double: module.add_function("rt_double", i64t.fn_type(&[f64t.into()], false), None),
            str_new: module.add_function("rt_str_new", i64t.fn_type(&[], false), None),
            str_chunk: module.add_function(
                "rt_str_chunk",
                i64t.fn_type(&[i64t.into(), i64t.into(), i8t.into()], false),
                None,
            ),
            str_register: module.add_function(
                "rt_str_register",
                i64t.fn_type(&[i64t.into(), i64t.into()], false),
                None,
            ),
            str_get: module.add_function("rt_str_get", i64t.fn_type(&[i64t.into()], false), None),
            keyword: module.add_function(
                "rt_keyword",
                i64t.fn_type(&[i64t.into(), i64t.into(), i8t.into()], false),
                None,
            ),
            vec_new: module.add_function("rt_vec_new", i64t.fn_type(&[], false), None),
            vec_push: module.add_function(
                "rt_vec_push",
                i64t.fn_type(&[i64t.into(), i64t.into()], false),
                None,
            ),
            clone: module.add_function("rt_clone", i64t.fn_type(&[i64t.into()], false), None),
            drop_: module.add_function("rt_drop", voidt.fn_type(&[i64t.into()], false), None),
            args_new: module.add_function("rt_args_new", i64t.fn_type(&[], false), None),
            args_push: module.add_function(
                "rt_args_push",
                i64t.fn_type(&[i64t.into(), i64t.into()], false),
                None,
            ),
            argv: module.add_function("rt_argv", i64t.fn_type(&[], false), None),
            print: module.add_function("rt_print", i64t.fn_type(&[i64t.into()], false), None),
            arity_error: module.add_function(
                "rt_arity_error",
                i64t.fn_type(
                    &[
                        i64t.into(),
                        i64t.into(),
                        i64t.into(),
                        i32t.into(),
                        i32t.into(),
                    ],
                    false,
                ),
                None,
            ),
            is_truthy: module.add_function(
                "rt_is_truthy",
                i8t.fn_type(&[i64t.into()], false),
                None,
            ),
            exit: module.add_function("exit", voidt.fn_type(&[i32t.into()], false), None),
        };
        Self {
            ctx,
            module,
            builder,
            ptr,
            i8t,
            i32t,
            i64t,
            f64t,
            decls,
            str_ids: HashMap::new(),
            next_str_id: 0,
            path: source_path.to_owned(),
            fns: Vec::new(),
            diverged: false,
            jit: false,
            in_main: false,
        }
    }

    /// New session in JIT return mode (see [`to_module_jit`]).
    fn new_jit(ctx: &'ctx Context, source_path: &str) -> Self {
        Self {
            jit: true,
            ..Self::new(ctx, source_path)
        }
    }

    /// The baked source path's str-id (interned on first use).
    fn path_id(&mut self) -> u64 {
        let path = self.path.clone();
        self.str_id(&path)
    }

    /// Intern a literal under a str-id (deduplicated across call
    /// sites; the `main` prologue registers every id once).
    fn str_id(&mut self, text: &str) -> u64 {
        if let Some(hit) = self.str_ids.get(text) {
            return *hit;
        }
        let id = self.next_str_id;
        self.next_str_id += 1;
        self.str_ids.insert(text.to_owned(), id);
        id
    }

    /// Call a helper returning an owned handle.
    fn call_handle(
        &self,
        func: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
    ) -> IntValue<'ctx> {
        self.builder
            .build_call(func, args, "call")
            .expect("call")
            .try_as_basic_value()
            .expect_basic("handle call")
            .into_int_value()
    }

    /// Call a helper returning nothing.
    fn call_void(&self, func: FunctionValue<'ctx>, args: &[BasicMetadataValueEnum<'ctx>]) {
        let _ignored = self.builder.build_call(func, args, "").expect("call");
    }

    /// Call a helper returning a byte.
    fn call_byte(
        &self,
        func: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
    ) -> IntValue<'ctx> {
        self.builder
            .build_call(func, args, "call")
            .expect("call")
            .try_as_basic_value()
            .expect_basic("byte call")
            .into_int_value()
    }

    /// Drop one owned handle.
    fn call_drop(&self, value: IntValue<'ctx>) {
        self.call_void(self.decls.drop_, &[value.into()]);
    }

    /// Nil-initialize every non-param frame slot (params are
    /// stored by the caller path below; a `let` slot in an
    /// untaken branch is never stored, so the init is what makes
    /// every slot a valid owned handle on every path).
    fn init_slots(&self, slots: &[PointerValue<'ctx>], params: usize) {
        for slot in &slots[params..] {
            let nil = self.call_handle(self.decls.nil, &[]);
            self.builder.build_store(*slot, nil).expect("store nil");
        }
    }

    /// Drop every frame slot (all hold valid owned handles: params
    /// stored at entry, `let` slots nil-initialized at entry plus
    /// one store per executed binding, never reused).
    fn drop_slots(&self, slots: &[PointerValue<'ctx>]) {
        for slot in slots {
            let value = self
                .builder
                .build_load(self.i64t, *slot, "slot")
                .expect("load")
                .into_int_value();
            self.call_drop(value);
        }
    }

    /// Abort the run: the AOT binary exits 1 plus `unreachable`;
    /// JIT mode drops the frame slots and returns instead (`0`
    /// from a `defn`, 1 from `main` — callers zero-check every
    /// call result, so the failure propagates outward).
    fn emit_abort(&self, slots: &[PointerValue<'ctx>]) {
        if !self.jit {
            self.call_void(self.decls.exit, &[self.i32t.const_int(1, false).into()]);
            self.builder.build_unreachable().expect("unreachable");
            return;
        }
        self.drop_slots(slots);
        if self.in_main {
            self.builder
                .build_return(Some(&self.i32t.const_int(1, false)))
                .expect("return");
        } else {
            self.builder
                .build_return(Some(&self.i64t.const_zero()))
                .expect("return");
        }
    }

    /// Zero-check a fallible result: `0` aborts (exit 1 AOT,
    /// slot-dropping return in JIT mode), else the value flows
    /// on (single-predecessor join, no phi needed).
    fn zero_check(
        &mut self,
        slots: &[PointerValue<'ctx>],
        value: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        let func = self
            .builder
            .get_insert_block()
            .expect("block")
            .get_parent()
            .expect("parent");
        let is_zero = self
            .builder
            .build_int_compare(IntPredicate::EQ, value, self.i64t.const_zero(), "iszero")
            .expect("iszero");
        let abort_bb = self.ctx.append_basic_block(func, "abort");
        let cont_bb = self.ctx.append_basic_block(func, "cont");
        self.builder
            .build_conditional_branch(is_zero, abort_bb, cont_bb)
            .expect("condbr");
        self.builder.position_at_end(abort_bb);
        self.emit_abort(slots);
        self.builder.position_at_end(cont_bb);
        value
    }

    /// Lower call arguments left to right, then pack them into an
    /// accumulator for a helper call. The acc is created only
    /// after all args lower, so no acc is ever live across a
    /// possibly-diverging arg lowering (spec §3.2 ownership).
    fn build_acc(
        &mut self,
        func: FunctionValue<'ctx>,
        slots: &[PointerValue<'ctx>],
        args: &[SubsetExpr],
    ) -> IntValue<'ctx> {
        let mut lowered = Vec::with_capacity(args.len());
        for arg in args {
            lowered.push(self.lower_expr(func, slots, arg));
        }
        let acc = self.call_handle(self.decls.args_new, &[]);
        for value in lowered {
            self.call_handle(self.decls.args_push, &[acc.into(), value.into()]);
        }
        acc
    }

    /// Lower an expression sequence: intermediates dropped, the
    /// last value returned (empty is nil).
    fn lower_seq(
        &mut self,
        func: FunctionValue<'ctx>,
        slots: &[PointerValue<'ctx>],
        exprs: &[SubsetExpr],
    ) -> IntValue<'ctx> {
        if exprs.is_empty() {
            return self.call_handle(self.decls.nil, &[]);
        }
        let mut last = self.i64t.const_zero();
        for (i, expr) in exprs.iter().enumerate() {
            let value = self.lower_expr(func, slots, expr);
            if i + 1 < exprs.len() {
                self.call_drop(value);
            } else {
                last = value;
            }
        }
        last
    }

    /// Lower one planned expression to an owned handle (diverging
    /// forms — arity aborts — terminate the block and continue in
    /// a fresh one with a zero value, so callers never branch).
    /// Literals lower here; compound forms and calls chain on so
    /// each match stays under the complexity gate.
    fn lower_expr(
        &mut self,
        func: FunctionValue<'ctx>,
        slots: &[PointerValue<'ctx>],
        expr: &SubsetExpr,
    ) -> IntValue<'ctx> {
        match expr {
            SubsetExpr::Nil => self.call_handle(self.decls.nil, &[]),
            SubsetExpr::Bool(b) => self.call_handle(
                self.decls.boolean,
                &[self.i8t.const_int(u64::from(*b), false).into()],
            ),
            SubsetExpr::Int(n) => self.call_handle(
                self.decls.int,
                &[self.i64t.const_int(*n as u64, false).into()],
            ),
            SubsetExpr::Double(x) => {
                self.call_handle(self.decls.double, &[self.f64t.const_float(*x).into()])
            }
            SubsetExpr::Str(s) => {
                let id = self.str_id(s);
                self.call_handle(self.decls.str_get, &[self.i64t.const_int(id, false).into()])
            }
            SubsetExpr::Keyword(k) => self.lower_keyword_parts(k.name(), k.namespace()),
            other => self.lower_compound(func, slots, other),
        }
    }

    /// Lower a keyword literal via two string handles (the L1
    /// pick): the name and ns parts are prologue-registered
    /// literals; unqualified keywords pass the registered empty
    /// string with a zero flag.
    fn lower_keyword_parts(&mut self, name: &str, ns: Option<&str>) -> IntValue<'ctx> {
        let name_id = self.str_id(name);
        let (ns_id, has_ns) = match ns {
            Some(ns) => (self.str_id(ns), 1),
            None => (self.str_id(""), 0),
        };
        let name = self.call_handle(
            self.decls.str_get,
            &[self.i64t.const_int(name_id, false).into()],
        );
        let ns = self.call_handle(
            self.decls.str_get,
            &[self.i64t.const_int(ns_id, false).into()],
        );
        self.call_handle(
            self.decls.keyword,
            &[
                name.into(),
                ns.into(),
                self.i8t.const_int(has_ns, false).into(),
            ],
        )
    }

    /// Lower compound forms (vectors, locals, control flow).
    fn lower_compound(
        &mut self,
        func: FunctionValue<'ctx>,
        slots: &[PointerValue<'ctx>],
        expr: &SubsetExpr,
    ) -> IntValue<'ctx> {
        match expr {
            SubsetExpr::VecLit(elems) => {
                let mut acc = self.call_handle(self.decls.vec_new, &[]);
                for elem in elems {
                    let value = self.lower_expr(func, slots, elem);
                    acc = self.call_handle(self.decls.vec_push, &[acc.into(), value.into()]);
                }
                acc
            }
            SubsetExpr::Local(i) => {
                let slot = self
                    .builder
                    .build_load(self.i64t, slots[*i], "local")
                    .expect("load")
                    .into_int_value();
                self.call_handle(self.decls.clone, &[slot.into()])
            }
            SubsetExpr::If { test, then, els } => self.lower_if(func, slots, test, then, els),
            SubsetExpr::Do(exprs) => self.lower_seq(func, slots, exprs),
            SubsetExpr::Let { bindings, body } => {
                for (slot, init) in bindings {
                    let value = self.lower_expr(func, slots, init);
                    let old = self
                        .builder
                        .build_load(self.i64t, slots[*slot], "old")
                        .expect("load old")
                        .into_int_value();
                    self.call_drop(old);
                    self.builder
                        .build_store(slots[*slot], value)
                        .expect("store let");
                }
                self.lower_seq(func, slots, body)
            }
            other => self.lower_call(func, slots, other),
        }
    }

    /// Lower calls (primitives, siblings, arity aborts, print).
    /// Anything else re-dispatches from the top (defense in depth
    /// — the chain only ever routes calls here — terminating after
    /// exactly one hop since `lower_expr` routes correctly).
    fn lower_call(
        &mut self,
        func: FunctionValue<'ctx>,
        slots: &[PointerValue<'ctx>],
        expr: &SubsetExpr,
    ) -> IntValue<'ctx> {
        match expr {
            SubsetExpr::CallPrim { prim, args, span } => {
                let acc = self.build_acc(func, slots, args);
                let path_id = self.path_id();
                let called = self.call_handle(
                    self.decls.prims[*prim as usize],
                    &[
                        acc.into(),
                        self.i64t.const_int(path_id, false).into(),
                        self.i32t.const_int(u64::from(span.line), false).into(),
                        self.i32t.const_int(u64::from(span.col), false).into(),
                    ],
                );
                self.zero_check(slots, called)
            }
            SubsetExpr::CallFn { index, args, .. } => {
                let mut params: Vec<BasicMetadataValueEnum> = Vec::with_capacity(args.len());
                for arg in args {
                    params.push(self.lower_expr(func, slots, arg).into());
                }
                let called = self.call_handle(self.fns[*index], &params);
                self.zero_check(slots, called)
            }
            SubsetExpr::ArityError {
                display, got, span, ..
            } => {
                let display_id = self.str_id(display);
                let path_id = self.path_id();
                let _reported = self
                    .builder
                    .build_call(
                        self.decls.arity_error,
                        &[
                            self.i64t.const_int(display_id, false).into(),
                            self.i64t.const_int(*got as u64, false).into(),
                            self.i64t.const_int(path_id, false).into(),
                            self.i32t.const_int(u64::from(span.line), false).into(),
                            self.i32t.const_int(u64::from(span.col), false).into(),
                        ],
                        "arity",
                    )
                    .expect("arity call");
                self.emit_abort(slots);
                let cont = self.ctx.append_basic_block(func, "cont");
                self.builder.position_at_end(cont);
                self.diverged = true;
                self.i64t.const_zero()
            }
            SubsetExpr::Print { args, .. } => {
                let acc = self.build_acc(func, slots, args);
                self.call_handle(self.decls.print, &[acc.into()])
            }
            other => self.lower_expr(func, slots, other),
        }
    }

    /// Lower `if`: the test consumes through `rt_is_truthy`; each
    /// branch joins only when it reaches (an arity abort diverges
    /// through, never joining) — two joiners share a phi, one
    /// passes its value straight through, none yields null (and the
    /// `if` itself diverges only when both branches do).
    fn lower_if(
        &mut self,
        func: FunctionValue<'ctx>,
        slots: &[PointerValue<'ctx>],
        test: &SubsetExpr,
        then: &SubsetExpr,
        els: &SubsetExpr,
    ) -> IntValue<'ctx> {
        let test_val = self.lower_expr(func, slots, test);
        let test_div = std::mem::replace(&mut self.diverged, false);
        let flag = self.call_byte(self.decls.is_truthy, &[test_val.into()]);
        let cond = self
            .builder
            .build_int_compare(IntPredicate::NE, flag, self.i8t.const_zero(), "truthy")
            .expect("compare");
        let then_bb = self.ctx.append_basic_block(func, "then");
        let else_bb = self.ctx.append_basic_block(func, "else");
        let merge_bb = self.ctx.append_basic_block(func, "merge");
        self.builder
            .build_conditional_branch(cond, then_bb, else_bb)
            .expect("condbr");
        let mut incoming: Vec<(IntValue<'ctx>, inkwell::basic_block::BasicBlock)> =
            Vec::with_capacity(2);
        self.builder.position_at_end(then_bb);
        self.diverged = false;
        let then_val = self.lower_expr(func, slots, then);
        let then_div = std::mem::replace(&mut self.diverged, false);
        if then_div {
            self.builder.build_unreachable().expect("seal");
        } else {
            let pred = self.builder.get_insert_block().expect("then block");
            self.builder
                .build_unconditional_branch(merge_bb)
                .expect("br");
            incoming.push((then_val, pred));
        }
        self.builder.position_at_end(else_bb);
        let else_val = self.lower_expr(func, slots, els);
        let else_div = std::mem::replace(&mut self.diverged, false);
        if else_div {
            self.builder.build_unreachable().expect("seal");
        } else {
            let pred = self.builder.get_insert_block().expect("else block");
            self.builder
                .build_unconditional_branch(merge_bb)
                .expect("br");
            incoming.push((else_val, pred));
        }
        self.builder.position_at_end(merge_bb);
        self.diverged = test_div || (then_div && else_div);
        if incoming.len() > 1 {
            let phi = self.builder.build_phi(self.i64t, "if").expect("phi");
            for (value, pred) in &incoming {
                phi.add_incoming(&[(value, *pred)]);
            }
            phi.as_basic_value().into_int_value()
        } else if let Some((single, _)) = incoming.first() {
            *single
        } else {
            self.i64t.const_zero()
        }
    }

    /// Lower one sibling `defn` (declared before its body so
    /// direct self-calls resolve; siblings only call backwards).
    fn lower_defn(&mut self, index: usize, def: &SubsetFn) {
        let params: Vec<BasicMetadataTypeEnum> = vec![self.i64t.into(); def.arity];
        let ty = self.i64t.fn_type(params.as_slice(), false);
        let func = self
            .module
            .add_function(&fn_symbol(index, &def.name), ty, None);
        self.fns.push(func);
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        self.diverged = false;
        self.in_main = false;
        let mut slots = Vec::with_capacity(def.frame_slots);
        for i in 0..def.frame_slots {
            slots.push(
                self.builder
                    .build_alloca(self.i64t, &format!("slot{i}"))
                    .expect("alloca slot"),
            );
        }
        for (i, slot) in slots.iter().enumerate().take(def.arity) {
            let param = func
                .get_nth_param(i as u32)
                .expect("param")
                .into_int_value();
            self.builder.build_store(*slot, param).expect("store param");
        }
        self.init_slots(&slots, def.arity);
        let retval = self.lower_expr(func, &slots, &def.body);
        self.drop_slots(&slots);
        self.builder.build_return(Some(&retval)).expect("return");
    }

    /// Lower the `main` statements (values discarded, slots
    /// dropped, exit 0 — abnormal exits happen in abort blocks).
    /// `main` takes C `argc`/`argv` (ignored — `rt_argv` reads the
    /// process environment); when the program defines `-main`,
    /// the tops run first (load order, like namespace load), then
    /// `-main` is invoked with the argv list (nil without args)
    /// and its return is discarded, like the JVM.
    fn lower_main(&mut self, main: &[SubsetExpr], main_slots: usize, main_fn: Option<usize>) {
        let ty = self
            .i32t
            .fn_type(&[self.i32t.into(), self.ptr.into()], false);
        let func = self.module.add_function("main", ty, None);
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        self.diverged = false;
        self.in_main = true;
        let mut slots = Vec::with_capacity(main_slots);
        for i in 0..main_slots {
            slots.push(
                self.builder
                    .build_alloca(self.i64t, &format!("slot{i}"))
                    .expect("alloca slot"),
            );
        }
        self.init_slots(&slots, 0);
        for stmt in main {
            let value = self.lower_expr(func, &slots, stmt);
            self.call_drop(value);
        }
        if let Some(index) = main_fn {
            let args = self.call_handle(self.decls.argv, &[]);
            let ret = self.call_handle(self.fns[index], &[args.into()]);
            // JIT only: a failing `-main` returns `0` (AOT exits
            // inside instead), so zero-check before drop —
            // without it a failing `-main` would exit 0. Gated on
            // `self.jit` to keep AOT IR byte-identical; `in_main`
            // is already true, so the abort drops slots and
            // returns 1.
            let ret = if self.jit {
                self.zero_check(&slots, ret)
            } else {
                ret
            };
            self.call_drop(ret);
        }
        self.drop_slots(&slots);
        self.builder
            .build_return(Some(&self.i32t.const_zero()))
            .expect("return");
        self.emit_str_prologue(func);
    }

    /// Emit the string-registration prologue (M1
    /// startup-registration): every interned literal is built once
    /// (chunked `rt_str_new` + N× `rt_str_chunk`) and interned
    /// under its str-id before any `rt_str_get` use site runs. The
    /// calls are inserted before the entry block's first
    /// instruction — all execution flows from `main`, so the
    /// prologue precedes every use, in defns and tops alike.
    fn emit_str_prologue(&mut self, func: FunctionValue<'ctx>) {
        if self.str_ids.is_empty() {
            return;
        }
        let mut interned: Vec<(u64, String)> = self
            .str_ids
            .iter()
            .map(|(text, id)| (*id, text.clone()))
            .collect();
        interned.sort();
        let entry = func.get_first_basic_block().expect("main entry");
        let first = entry.get_first_instruction().expect("entry head");
        self.builder.position_before(&first);
        for (id, text) in &interned {
            let builder = self.call_handle(self.decls.str_new, &[]);
            for (chunk, nbytes) in str_chunks(text) {
                self.call_handle(
                    self.decls.str_chunk,
                    &[
                        builder.into(),
                        self.i64t.const_int(chunk, false).into(),
                        self.i8t.const_int(u64::from(nbytes), false).into(),
                    ],
                );
            }
            self.call_handle(
                self.decls.str_register,
                &[self.i64t.const_int(*id, false).into(), builder.into()],
            );
        }
    }
}

/// Split literal bytes into little-endian 8-byte chunks (the
/// tail carries its own width) for prologue emission.
fn str_chunks(text: &str) -> Vec<(u64, u8)> {
    let mut out = Vec::new();
    for piece in text.as_bytes().chunks(8) {
        let mut chunk = [0_u8; 8];
        chunk[..piece.len()].copy_from_slice(piece);
        out.push((u64::from_le_bytes(chunk), piece.len() as u8));
    }
    out
}

/// Verify a lowered module, mapping LLVM's free-text rejection
/// to an `Emit`-bound message. Uses the inherent lossy
/// `to_string`: `LLVMString`'s `Display` impl debug-quotes the
/// text.
pub(crate) fn verify_module(module: &Module) -> Result<(), String> {
    module
        .verify()
        .map_err(|e| format!("LLVM verification failed: {}", e.to_string().trim()))
}

/// Lower a planned program to an LLVM module (the source path is
/// baked into every error span). The module is verified before
/// return: a rejection means a lowering bug, reported as `Err`
/// instead of reaching emission.
pub fn to_module<'ctx>(
    ctx: &'ctx Context,
    program: &SubsetProgram,
    source_path: &str,
) -> Result<Module<'ctx>, String> {
    let mut lower = Lower::new(ctx, source_path);
    for (index, def) in program.fns.iter().enumerate() {
        lower.lower_defn(index, def);
    }
    lower.lower_main(&program.main, program.main_slots, program.main_fn);
    verify_module(&lower.module)?;
    Ok(lower.module)
}

/// Lower a planned program to a JIT-run module: same IR as
/// [`to_module`] except abort sites return instead of exiting
/// (gap-121 slice 1, spec D-error — `exit(1)` is fatal
/// in-process, so failures propagate as `0` up to `main`,
/// which returns 1). Verified before return, like AOT.
pub fn to_module_jit<'ctx>(
    ctx: &'ctx Context,
    program: &SubsetProgram,
    source_path: &str,
) -> Result<Module<'ctx>, String> {
    let mut lower = Lower::new_jit(ctx, source_path);
    for (index, def) in program.fns.iter().enumerate() {
        lower.lower_defn(index, def);
    }
    lower.lower_main(&program.main, program.main_slots, program.main_fn);
    verify_module(&lower.module)?;
    Ok(lower.module)
}

#[cfg(test)]
mod tests {
    use super::{PRIM_ORDER, fn_symbol, to_module, verify_module};
    use clj_eval::{SubsetExpr, plan_source};
    use cljrt::prim_symbol;
    use inkwell::context::Context;

    /// Lower `source` and return verified IR (every snapshot pins
    /// `contains`, and verification proves the SSA well-formed —
    /// phis, dominance, and the arity-divergence joins).
    fn compile_ir(source: &str) -> String {
        let program = plan_source(source).expect("plan");
        let ctx = Context::create();
        let module = to_module(&ctx, &program, "snap.clj").expect("verify");
        module.print_to_string().to_string()
    }

    #[test]
    fn verify_module_maps_verifier_text() {
        let ctx = Context::create();
        let module = ctx.create_module("broken");
        let ty = ctx.i32_type().fn_type(&[], false);
        let func = module.add_function("no_terminator", ty, None);
        let _entry = ctx.append_basic_block(func, "entry");
        let err = verify_module(&module).unwrap_err();
        assert!(err.starts_with("LLVM verification failed: "), "{err}");
        assert!(err.contains("terminator"), "{err}");
    }

    #[test]
    fn fn_symbol_sanitizes_and_indexes() {
        assert_eq!(fn_symbol(0, "f"), "cljfn_f_0");
        assert_eq!(fn_symbol(1, "my-fn?"), "cljfn_my_fn__1");
        assert_eq!(fn_symbol(12, "a/b.c"), "cljfn_a_b_c_12");
    }

    #[test]
    fn prim_order_matches_discriminants() {
        let names = [
            "rt_add", "rt_sub", "rt_mul", "rt_quot", "rt_lt", "rt_le", "rt_gt", "rt_ge", "rt_eq",
            "rt_count", "rt_nth", "rt_get", "rt_conj",
        ];
        for (index, prim) in PRIM_ORDER.iter().enumerate() {
            assert_eq!(*prim as usize, index);
            assert_eq!(prim_symbol(*prim), names[index]);
        }
        assert_eq!(format!("{PRIM_ORDER:?}").split(',').count(), 13);
    }

    #[test]
    fn every_prim_lowers_to_its_symbol() {
        let ir = compile_ir(
            "(+ 1 2) (- 3 1) (* 2 3) (quot 7 2) (< 1 2) (<= 1 1) (> 2 1) \
             (>= 2 2) (= 1 1) (count [1]) (nth [1] 0) (get [1] 0) (conj [1] 2)",
        );
        for symbol in [
            "rt_add", "rt_sub", "rt_mul", "rt_quot", "rt_lt", "rt_le", "rt_gt", "rt_ge", "rt_eq",
            "rt_count", "rt_nth", "rt_get", "rt_conj",
        ] {
            assert!(
                ir.contains(&format!("call i64 @{symbol}")),
                "{symbol}\n{ir}"
            );
        }
        assert!(ir.contains("call void @exit(i32 1)"), "{ir}");
        assert!(ir.contains("define i32 @main(i32"), "{ir}");
        // Args ride an accumulator (one new + one push per arg).
        assert_eq!(
            ir.match_indices("call i64 @rt_args_new").count(),
            13,
            "{ir}"
        );
        assert_eq!(
            ir.match_indices("call i64 @rt_args_push").count(),
            25,
            "{ir}"
        );
    }

    #[test]
    fn defn_and_call_lower() {
        let ir = compile_ir("(defn f [x] (+ x 1)) (print (f 41))");
        assert!(ir.contains("define i64 @cljfn_f_0(i64"), "{ir}");
        assert!(ir.contains("call i64 @cljfn_f_0"), "{ir}");
        assert!(ir.contains("call i64 @rt_print"), "{ir}");
        assert!(ir.contains("call i64 @rt_clone"), "{ir}");
    }

    #[test]
    fn control_flow_and_literals_lower() {
        let ir = compile_ir("(let [x 1] (if (= x 1) [1 \"s\" 2.5 true nil] (do (print 0) 2)))");
        assert!(ir.contains("call i8 @rt_is_truthy"), "{ir}");
        assert!(ir.contains("phi i64"), "{ir}");
        assert!(ir.contains("call i64 @rt_vec_new"), "{ir}");
        assert!(ir.contains("call i64 @rt_vec_push"), "{ir}");
        assert!(ir.contains("call i64 @rt_str_get"), "{ir}");
        assert!(ir.contains("call i64 @rt_double"), "{ir}");
        assert!(ir.contains("call i64 @rt_bool"), "{ir}");
        assert!(ir.contains("call i64 @rt_nil"), "{ir}");
        assert!(!ir.contains("getelementptr"), "{ir}");
    }

    #[test]
    fn keyword_literals_lower() {
        // T73 item 8: keywords flow (plain and namespaced) via two
        // string handles each (the L1 pick).
        let ir = compile_ir("(print :miss [:a :u/id])");
        assert_eq!(ir.match_indices("call i64 @rt_keyword").count(), 3, "{ir}");
        assert_eq!(ir.match_indices("call i64 @rt_str_get").count(), 6, "{ir}");
    }

    #[test]
    fn string_prologue_registers_each_literal_once() {
        // M1 startup-registration: each unique literal is built
        // once in the `main` prologue (new + chunks + register)
        // no matter how many use sites read it.
        let ir = compile_ir("(print \"ab\" \"ab\" \"c\")");
        assert_eq!(ir.match_indices("call i64 @rt_str_new").count(), 2, "{ir}");
        assert_eq!(
            ir.match_indices("call i64 @rt_str_register").count(),
            2,
            "{ir}"
        );
        assert_eq!(
            ir.match_indices("call i64 @rt_str_chunk").count(),
            2,
            "{ir}"
        );
        assert_eq!(ir.match_indices("call i64 @rt_str_get").count(), 3, "{ir}");
        // Nine bytes span two chunks (8 + 1).
        let ir = compile_ir("(print \"123456789\")");
        assert_eq!(
            ir.match_indices("call i64 @rt_str_chunk").count(),
            2,
            "{ir}"
        );
        // Literally nothing to register: no prologue at all.
        let ir = compile_ir("(print 1)");
        assert!(!ir.contains("call i64 @rt_str_new"), "{ir}");
        assert!(!ir.contains("call i64 @rt_str_register"), "{ir}");
    }

    #[test]
    fn arity_aborts_verify_in_every_branch_shape() {
        for source in [
            "(defn f [x] x) (if true (f 1 2) 3)",
            "(defn f [x] x) (if true 3 (f 1 2))",
            "(defn f [x] x) (if true (f 1 2) (f 1 2))",
            "(defn f [x] x) (if (f 1 2) 1 2)",
            "(defn f [x] x) (+ (f 1 2) 4)",
        ] {
            let ir = compile_ir(source);
            assert!(ir.contains("call i64 @rt_arity_error"), "{source}\n{ir}");
            assert!(ir.contains("unreachable"), "{source}\n{ir}");
        }
    }

    #[test]
    fn empty_forms_lower_to_nil() {
        let ir = compile_ir("(do)");
        assert!(ir.contains("call i64 @rt_nil"), "{ir}");
        let ir = compile_ir("(defn f [x] x)");
        assert!(ir.contains("define i32 @main(i32"), "{ir}");
        assert!(ir.contains("ret i32 0"), "{ir}");
    }

    #[test]
    fn dash_main_receives_argv_after_tops() {
        // T1192 FX1 D-argv: `main` builds the argv list and calls
        // `-main` after the tops; without `-main` no argv call.
        // The call is zero-param (the C `argc`/`argv` are ignored).
        let ir = compile_ir("(print 1) (defn -main [args] (print args))");
        assert!(ir.contains("define i64 @cljfn__main_0"), "{ir}");
        assert!(ir.contains("call i64 @rt_argv()"), "{ir}");
        assert!(ir.contains("call i64 @cljfn__main_0"), "{ir}");
        let ir = compile_ir("(print 1) (defn f [x] x)");
        assert!(!ir.contains("call i64 @rt_argv"), "{ir}");
    }

    #[test]
    fn lower_call_redispatches_defensively() {
        let ctx = Context::create();
        let mut lower = super::Lower::new(&ctx, "snap.clj");
        let ty = lower.i64t.fn_type(&[], false);
        let func = lower.module.add_function("scratch", ty, None);
        let entry = ctx.append_basic_block(func, "entry");
        lower.builder.position_at_end(entry);
        let out = lower.lower_call(func, &[], &SubsetExpr::Nil);
        lower.builder.build_return(Some(&out)).expect("return");
        lower.module.verify().expect("verify");
        let ir = lower.module.print_to_string().to_string();
        assert!(ir.contains("call i64 @rt_nil"), "{ir}");
    }

    #[test]
    fn conditional_let_slots_nil_init_and_drop_old() {
        // T73 item 9: a `let` in an untaken branch never stores,
        // so every non-param slot starts nil (else the scope-end
        // drop frees stack garbage) and each binding drops the
        // slot's old handle before storing.
        let ir = compile_ir("(defn f [c] (if c (let [x 1] x) 2))");
        assert_eq!(ir.match_indices("call i64 @rt_nil").count(), 1, "{ir}");
        assert_eq!(ir.match_indices("call void @rt_drop").count(), 3, "{ir}");
        let ir = compile_ir("(print (if false (let [x 1] x) 2))");
        assert_eq!(ir.match_indices("call i64 @rt_nil").count(), 1, "{ir}");
        assert_eq!(ir.match_indices("call void @rt_drop").count(), 3, "{ir}");
    }

    #[test]
    fn self_calls_are_plain_calls() {
        let ir = compile_ir("(defn f [n] (if (= n 0) 0 (f 1))) (f 3)");
        assert!(ir.contains("call i64 @cljfn_f_0"), "{ir}");
        assert!(!ir.contains("tail call"), "{ir}");
    }

    #[test]
    fn str_chunks_split_little_endian() {
        assert_eq!(super::str_chunks(""), Vec::new());
        assert_eq!(super::str_chunks("ab"), vec![(0x6261, 2)]);
        assert_eq!(
            super::str_chunks("123456789"),
            vec![(0x3837363534333231, 8), (0x39, 1)]
        );
    }
}
