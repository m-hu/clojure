// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Unboxed SSA-promotable scalar `loop`/`recur` lowering for `clj-aot`
//! (issue #271 Slice 3b).
//!
//! Lowers [`ScalarAst`] from `clj_eval::aot_scalar::compile_aot_loop`
//! into unboxed `(k: i8, v: i64)` slots allocated in the function entry
//! block (promoted to SSA registers and `phi` nodes by LLVM's `sroa` /
//! `mem2reg` pass). Invariant strings are probed once in the loop
//! preheader (`rt_bc_scalar_str`) so in-bounds ASCII `(nth s i)` compiles
//! to a direct byte load, and transient vector mutations (`assoc!`,
//! `conj!`, `pop!`) run through zero-alloc bitcode helpers with cold
//! in-place fallback for capacity growth.

use clj_eval::aot_scalar::{
    Calc, Cast, Cmp, Cond, Ctl, Effect, ExitVal, IExpr, Inner, IntOp1, IntOp2, IntTest1, IntTest2,
    K, Leaf, Loc, Node, ScalarAst, Stmt, compile_aot_loop,
};
use clj_eval::expr::{Expr, Pattern};
use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::types::BasicType;
use inkwell::values::{FunctionValue, IntValue, PointerValue};

use super::{FnCtx, LocalRef, Lowerer, RecurTarget, Unsupported};

/// Cached preheader metadata for one invariant collection.
struct CollCache<'ctx> {
    slot: PointerValue<'ctx>,
    str_ok: PointerValue<'ctx>,
    str_ptr: PointerValue<'ctx>,
    str_len: PointerValue<'ctx>,
    cnt_ok: PointerValue<'ctx>,
    cnt_val: PointerValue<'ctx>,
}

/// Active loop target (outer loop vs nested `Inner` loop).
enum LoopTarget<'a, 'ctx> {
    Outer {
        scalar: &'a [bool],
        header: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
        dest: PointerValue<'ctx>,
        bound: &'a [PointerValue<'ctx>],
    },
    Inner {
        slots: &'a [usize],
        header: BasicBlock<'ctx>,
        exit: BasicBlock<'ctx>,
        res_k: PointerValue<'ctx>,
        res_v: PointerValue<'ctx>,
    },
}

/// Pre-scanned AST properties and invariant locals/collections.
#[derive(Default)]
struct AstScan {
    loads: Vec<Loc>,
    colls: Vec<Loc>,
    mutated: Vec<Loc>,
    all_locs: Vec<Loc>,
    has_effects: bool,
    has_deopt: bool,
}

impl AstScan {
    fn push_unique(vec: &mut Vec<Loc>, loc: Loc) {
        if !vec.contains(&loc) {
            vec.push(loc);
        }
    }

    fn record_load(&mut self, loc: Loc) {
        Self::push_unique(&mut self.loads, loc);
        Self::push_unique(&mut self.all_locs, loc);
    }

    fn record_coll(&mut self, loc: Loc, mutated: bool) {
        Self::push_unique(&mut self.colls, loc);
        Self::push_unique(&mut self.all_locs, loc);
        if mutated {
            Self::push_unique(&mut self.mutated, loc);
        }
    }

    fn scan_node(&mut self, node: &Node) {
        match node {
            Node::If(c, a, b) => {
                self.scan_cond(c);
                self.scan_node(a);
                self.scan_node(b);
            }
            Node::Let(binds, b) => {
                self.scan_binds(binds);
                self.scan_node(b);
            }
            Node::Seq(stmts, b) => {
                for s in stmts.iter() {
                    self.scan_stmt(s);
                }
                self.scan_node(b);
            }
            Node::Leaf(leaf) => self.scan_leaf(leaf),
            Node::Deopt => self.has_deopt = true,
        }
    }

    fn scan_leaf(&mut self, leaf: &Leaf) {
        match leaf {
            Leaf::Recur(ops) => {
                for e in ops.iter().flatten() {
                    self.scan_expr(e);
                }
            }
            Leaf::Exit(ExitVal::Expr(e)) => self.scan_expr(e),
            Leaf::Exit(ExitVal::Box(loc)) => Self::push_unique(&mut self.all_locs, *loc),
            Leaf::Exit(ExitVal::Const(_) | ExitVal::Var(_)) => {}
        }
    }

    fn scan_binds(&mut self, binds: &[(usize, IExpr)]) {
        for (_, e) in binds {
            self.scan_expr(e);
        }
    }

    fn scan_stmt(&mut self, st: &Stmt) {
        match st {
            Stmt::Effect(eff) => self.scan_effect(eff),
            Stmt::If(c, a, b) => {
                self.scan_cond(c);
                for x in a.iter().chain(b.iter()) {
                    self.scan_stmt(x);
                }
            }
            Stmt::Let(binds, b) => {
                self.scan_binds(binds);
                for x in b.iter() {
                    self.scan_stmt(x);
                }
            }
            Stmt::Loop(inner) => self.scan_inner(inner),
            Stmt::Expr(e) => self.scan_expr(e),
        }
    }

    fn scan_effect(&mut self, eff: &Effect) {
        self.has_effects = true;
        match eff {
            Effect::Assoc(loc, k, v) => {
                self.record_coll(*loc, true);
                self.scan_expr(k);
                self.scan_expr(v);
            }
            Effect::Conj(loc, v) => {
                self.record_coll(*loc, true);
                self.scan_expr(v);
            }
            Effect::Pop(loc) => self.record_coll(*loc, true),
        }
    }

    fn scan_inner(&mut self, inner: &Inner) {
        for e in inner.inits.iter() {
            self.scan_expr(e);
        }
        self.scan_node(&inner.body);
    }

    fn scan_cond(&mut self, c: &Cond) {
        match c {
            Cond::Leaf(Cmp::Truthy(e) | Cmp::T1(_, e)) => self.scan_expr(e),
            Cond::Leaf(Cmp::T2(_, a, b) | Cmp::Eq(a, b)) => {
                self.scan_expr(a);
                self.scan_expr(b);
            }
            Cond::Not(inner) => self.scan_cond(inner),
            Cond::And(a, b) | Cond::Or(a, b) => {
                self.scan_cond(a);
                self.scan_cond(b);
            }
        }
    }

    fn scan_expr(&mut self, e: &IExpr) {
        match e {
            IExpr::Slot(_) | IExpr::Kst(_) => {}
            IExpr::Loc(loc) => self.record_load(*loc),
            IExpr::Calc(c) => self.scan_calc(c),
            IExpr::Ctl(c) => self.scan_ctl(c),
            IExpr::Deopt => self.has_deopt = true,
        }
    }

    fn scan_calc(&mut self, c: &Calc) {
        match c {
            Calc::Op1(_, a) | Calc::Cast(_, a) => self.scan_expr(a),
            Calc::Op2(_, a, b) => {
                self.scan_expr(a);
                self.scan_expr(b);
            }
            Calc::Count(loc) => self.record_coll(*loc, false),
            Calc::Nth(loc, a) | Calc::Nth2(loc, a, _) => {
                self.record_coll(*loc, false);
                self.scan_expr(a);
            }
        }
    }

    fn scan_ctl(&mut self, c: &Ctl) {
        match c {
            Ctl::If(cond, a, b) => {
                self.scan_cond(cond);
                self.scan_expr(a);
                self.scan_expr(b);
            }
            Ctl::Let(binds, b) => {
                self.scan_binds(binds);
                self.scan_expr(b);
            }
            Ctl::Seq(stmts, b) => {
                for s in stmts.iter() {
                    self.scan_stmt(s);
                }
                self.scan_expr(b);
            }
            Ctl::Loop(inner) => self.scan_inner(inner),
            Ctl::Test(cond) => self.scan_cond(cond),
        }
    }
}

/// Per-loop scalar lowering state.
struct ScalarSession<'ctx> {
    k_slots: Vec<PointerValue<'ctx>>,
    v_slots: Vec<PointerValue<'ctx>>,
    load_locs: Vec<Loc>,
    load_vals: Vec<(IntValue<'ctx>, IntValue<'ctx>)>,
    coll_locs: Vec<Loc>,
    colls: Vec<CollCache<'ctx>>,
    scratch_k: PointerValue<'ctx>,
    scratch_v: PointerValue<'ctx>,
    deopt_bb: BasicBlock<'ctx>,
}

impl<'ctx> ScalarSession<'ctx> {
    fn load_val(&self, loc: Loc) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let idx = self
            .load_locs
            .iter()
            .position(|l| *l == loc)
            .expect("registered load loc");
        self.load_vals[idx]
    }

    fn coll_idx(&self, loc: Loc) -> usize {
        self.coll_locs
            .iter()
            .position(|l| *l == loc)
            .expect("registered coll loc")
    }
}

impl<'ctx> Lowerer<'ctx> {
    /// Attempt to lower `(loop bindings body...)` via unboxed scalar IR,
    /// returning `Ok(false)` when the loop is not eligible.
    pub(super) fn try_lower_scalar_loop(
        &mut self,
        f: &mut FnCtx<'ctx>,
        bindings: &[(Pattern, Expr)],
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<bool, Unsupported> {
        if !self.inline_arith || !self.use_bitcode || bindings.is_empty() {
            return Ok(false);
        }
        if !bindings
            .iter()
            .all(|(p, _)| matches!(p, Pattern::Symbol(_)))
        {
            return Ok(false);
        }
        let depth = f.env.len();
        let body_base = depth + bindings.len();
        let slots: Vec<usize> = (depth..body_base).collect();
        let Some(ast) = compile_aot_loop(body, &slots, body_base, &self.callees) else {
            return Ok(false);
        };
        let mut scan = AstScan::default();
        scan.scan_node(&ast.node);
        if !self.can_lower_scalar(f, &ast, &scan) {
            return Ok(false);
        }
        let (bound, targets) = self.bind_scalar_loop_inits(f, bindings)?;
        let sc_hdr = self.ctx.append_basic_block(f.func, "sc_hdr");
        let deopt_bb = self.ctx.append_basic_block(f.func, "sc_deopt");
        let boxed_hdr = self.ctx.append_basic_block(f.func, "sc_boxed");
        let loop_done = self.ctx.append_basic_block(f.func, "sc_done");
        let session = self.init_scalar_session(f, &ast, &scan, &bound, sc_hdr, deopt_bb, boxed_hdr);
        self.builder.position_at_end(sc_hdr);
        let outer = LoopTarget::Outer {
            scalar: &ast.scalar,
            header: sc_hdr,
            done: loop_done,
            dest,
            bound: &bound,
        };
        self.lower_sc_node(f, &session, &ast.node, &outer)?;
        self.emit_scalar_deopt(f, &session, &ast.scalar, &bound, deopt_bb, boxed_hdr);
        self.emit_scalar_boxed_fallback(f, body, dest, depth, targets, boxed_hdr, loop_done)?;
        self.builder.position_at_end(loop_done);
        self.drop_slots(f.func, &bound);
        Ok(true)
    }

    #[allow(clippy::type_complexity)]
    fn bind_scalar_loop_inits(
        &mut self,
        f: &mut FnCtx<'ctx>,
        bindings: &[(Pattern, Expr)],
    ) -> Result<
        (
            Vec<PointerValue<'ctx>>,
            Vec<(Pattern, Vec<PointerValue<'ctx>>)>,
        ),
        Unsupported,
    > {
        let mut bound = Vec::with_capacity(bindings.len());
        let mut targets = Vec::with_capacity(bindings.len());
        for (pat, init) in bindings {
            let t = self.slot(f);
            self.lower(f, init, t)?;
            f.env.push(Some(LocalRef::Slot(t)));
            bound.push(t);
            targets.push((pat.clone(), vec![t]));
        }
        Ok((bound, targets))
    }

    fn emit_scalar_deopt(
        &self,
        f: &FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        scalar: &[bool],
        bound: &[PointerValue<'ctx>],
        deopt_bb: BasicBlock<'ctx>,
        boxed_hdr: BasicBlock<'ctx>,
    ) {
        self.builder.position_at_end(deopt_bb);
        for (p, &is_sc) in scalar.iter().enumerate() {
            if is_sc {
                let k = self.load_i8(s.k_slots[p]);
                let v = self.load_i64(s.v_slots[p]);
                self.box_scalar(f.func, bound[p], k, v);
            }
        }
        self.builder
            .build_unconditional_branch(boxed_hdr)
            .expect("br");
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_scalar_boxed_fallback(
        &mut self,
        f: &mut FnCtx<'ctx>,
        body: &[Expr],
        dest: PointerValue<'ctx>,
        depth: usize,
        targets: Vec<(Pattern, Vec<PointerValue<'ctx>>)>,
        boxed_hdr: BasicBlock<'ctx>,
        loop_done: BasicBlock<'ctx>,
    ) -> Result<(), Unsupported> {
        self.builder.position_at_end(boxed_hdr);
        f.recur.push(RecurTarget {
            header: boxed_hdr,
            patterns: targets,
        });
        let res = self.lower_body(f, body, dest);
        f.recur.pop();
        f.env.truncate(depth);
        res?;
        self.builder
            .build_unconditional_branch(loop_done)
            .expect("br");
        Ok(())
    }

    fn can_lower_scalar(&self, f: &FnCtx<'ctx>, ast: &ScalarAst, scan: &AstScan) -> bool {
        if !scan.all_locs.iter().all(|loc| self.is_direct_loc(f, *loc)) {
            return false;
        }
        if scan.has_effects && (ast.primed || scan.has_deopt) {
            return false;
        }
        true
    }

    fn is_direct_loc(&self, f: &FnCtx<'ctx>, loc: Loc) -> bool {
        matches!(
            f.env.get(loc_reg_idx(loc)).copied().flatten(),
            Some(LocalRef::Slot(_) | LocalRef::Captured(_))
        )
    }

    fn loc_slot(&self, f: &FnCtx<'ctx>, loc: Loc) -> PointerValue<'ctx> {
        self.local_slot(f, loc_reg_idx(loc))
            .expect("direct local slot")
    }

    pub(super) fn entry_alloca(
        &self,
        f: &FnCtx<'ctx>,
        ty: impl BasicType<'ctx>,
        name: &str,
    ) -> PointerValue<'ctx> {
        let here = self.builder.get_insert_block().expect("bb");
        self.builder.position_at_end(f.entry);
        let p = self.builder.build_alloca(ty, name).expect("alloca");
        self.builder.position_at_end(here);
        p
    }

    fn load_i8(&self, ptr: PointerValue<'ctx>) -> IntValue<'ctx> {
        self.builder
            .build_load(self.i8t, ptr, "k")
            .expect("load")
            .into_int_value()
    }

    fn load_i64(&self, ptr: PointerValue<'ctx>) -> IntValue<'ctx> {
        self.builder
            .build_load(self.i64t, ptr, "v")
            .expect("load")
            .into_int_value()
    }

    /// Unbox `slot` into `(ok: i1, k: i8, v: i64)` with zero branches.
    fn unbox_scalar(
        &self,
        slot: PointerValue<'ctx>,
    ) -> (IntValue<'ctx>, IntValue<'ctx>, IntValue<'ctx>) {
        let b = &self.builder;
        let t = self.tag(slot);
        let is_int = b
            .build_int_compare(IntPredicate::EQ, t, self.i8t.const_int(2, false), "is_i")
            .expect("cmp");
        let is_chr = b
            .build_int_compare(IntPredicate::EQ, t, self.i8t.const_int(9, false), "is_c")
            .expect("cmp");
        let is_bl = b
            .build_int_compare(IntPredicate::EQ, t, self.i8t.const_int(1, false), "is_b")
            .expect("cmp");
        let ok1 = b.build_or(is_int, is_chr, "ok1").expect("or");
        let ok = b.build_or(ok1, is_bl, "sc_ok").expect("or");
        let iv = self.int_payload(slot);
        let cv = b
            .build_int_z_extend(self.char_payload(slot), self.i64t, "cv")
            .expect("zext");
        let raw_b = self.load_i8(self.byte_at(slot, 1));
        let bv = b.build_int_z_extend(raw_b, self.i64t, "bv").expect("zext");
        let v1 = b
            .build_select(is_chr, cv, bv, "v1")
            .expect("sel")
            .into_int_value();
        let v = b
            .build_select(is_int, iv, v1, "sc_v")
            .expect("sel")
            .into_int_value();
        let k1 = b
            .build_select(
                is_chr,
                self.i8t.const_int(1, false),
                self.i8t.const_int(2, false),
                "k1",
            )
            .expect("sel")
            .into_int_value();
        let k = b
            .build_select(is_int, self.i8t.const_zero(), k1, "sc_k")
            .expect("sel")
            .into_int_value();
        (ok, k, v)
    }

    /// Box `(k: i8, v: i64)` into `dest` (`0 = Int, 1 = Char, 2 = Bool`).
    fn box_scalar(
        &self,
        func: FunctionValue<'ctx>,
        dest: PointerValue<'ctx>,
        k: IntValue<'ctx>,
        v: IntValue<'ctx>,
    ) {
        self.drop_v(func, dest);
        let b = &self.builder;
        let is_int = b
            .build_int_compare(IntPredicate::EQ, k, self.i8t.const_zero(), "bi")
            .expect("cmp");
        let is_chr = b
            .build_int_compare(IntPredicate::EQ, k, self.i8t.const_int(1, false), "bc")
            .expect("cmp");
        let t1 = b
            .build_select(
                is_chr,
                self.i8t.const_int(9, false),
                self.i8t.const_int(1, false),
                "t1",
            )
            .expect("sel")
            .into_int_value();
        let tag = b
            .build_select(is_int, self.i8t.const_int(2, false), t1, "btag")
            .expect("sel")
            .into_int_value();
        b.build_store(dest, tag).expect("store");
        let nz = b
            .build_int_compare(IntPredicate::NE, v, self.i64t.const_zero(), "bnz")
            .expect("cmp");
        let b8 = b.build_int_z_extend(nz, self.i8t, "b8").expect("zext");
        b.build_store(self.byte_at(dest, 1), b8).expect("store");
        let c32 = b.build_int_truncate(v, self.i32t, "c32").expect("trunc");
        b.build_store(self.byte_at(dest, 4), c32).expect("store");
        b.build_store(self.byte_at(dest, 8), v).expect("store");
    }

    #[allow(clippy::too_many_arguments)]
    fn init_scalar_session(
        &mut self,
        f: &mut FnCtx<'ctx>,
        ast: &ScalarAst,
        scan: &AstScan,
        bound: &[PointerValue<'ctx>],
        sc_hdr: BasicBlock<'ctx>,
        deopt_bb: BasicBlock<'ctx>,
        boxed_hdr: BasicBlock<'ctx>,
    ) -> ScalarSession<'ctx> {
        let mut k_slots = Vec::with_capacity(ast.nslots);
        let mut v_slots = Vec::with_capacity(ast.nslots);
        for _ in 0..ast.nslots {
            k_slots.push(self.entry_alloca(f, self.i8t, "sck"));
            v_slots.push(self.entry_alloca(f, self.i64t, "scv"));
        }
        let scratch_k = self.entry_alloca(f, self.i8t, "scrk");
        let scratch_v = self.entry_alloca(f, self.i64t, "scrv");
        for (p, &is_sc) in ast.scalar.iter().enumerate() {
            if is_sc {
                let (ok, k, v) = self.unbox_scalar(bound[p]);
                let next = self.ctx.append_basic_block(f.func, "sc_init");
                self.builder
                    .build_conditional_branch(ok, next, boxed_hdr)
                    .expect("br");
                self.builder.position_at_end(next);
                self.builder.build_store(k_slots[p], k).expect("store");
                self.builder.build_store(v_slots[p], v).expect("store");
            }
        }
        let mut load_vals = Vec::with_capacity(scan.loads.len());
        for &loc in &scan.loads {
            let slot = self.loc_slot(f, loc);
            let (ok, k, v) = self.unbox_scalar(slot);
            let next = self.ctx.append_basic_block(f.func, "sc_box_ok");
            self.builder
                .build_conditional_branch(ok, next, boxed_hdr)
                .expect("br");
            self.builder.position_at_end(next);
            load_vals.push((k, v));
        }
        let mut colls = Vec::with_capacity(scan.colls.len());
        for &loc in &scan.colls {
            let mutated = scan.mutated.contains(&loc);
            colls.push(self.probe_coll(f, loc, mutated));
        }
        self.builder.build_unconditional_branch(sc_hdr).expect("br");
        ScalarSession {
            k_slots,
            v_slots,
            load_locs: scan.loads.clone(),
            load_vals,
            coll_locs: scan.colls.clone(),
            colls,
            scratch_k,
            scratch_v,
            deopt_bb,
        }
    }

    fn probe_coll(&self, f: &FnCtx<'ctx>, loc: Loc, mutated: bool) -> CollCache<'ctx> {
        let slot = self.loc_slot(f, loc);
        let str_ok = self.entry_alloca(f, self.i8t, "strok");
        let str_ptr = self.entry_alloca(f, self.ptr, "strp");
        let str_len = self.entry_alloca(f, self.i64t, "strn");
        let cnt_ok = self.entry_alloca(f, self.i8t, "cntok");
        let cnt_val = self.entry_alloca(f, self.i64t, "cntv");
        let probe = self
            .helper("rt_n_str_ascii", Some(self.i8t.into()), &[self.ptr.into()])
            .as_global_value()
            .as_pointer_value();
        let sok = self
            .call(
                "rt_bc_scalar_str",
                Some(self.i8t),
                &[slot.into(), str_ptr.into(), str_len.into(), probe.into()],
            )
            .expect("sok");
        self.builder.build_store(str_ok, sok).expect("store");
        if mutated {
            self.builder
                .build_store(cnt_ok, self.i8t.const_zero())
                .expect("store");
        } else {
            let cok = self
                .call(
                    "rt_bc_scalar_count",
                    Some(self.i8t),
                    &[slot.into(), cnt_val.into(), probe.into()],
                )
                .expect("cok");
            self.builder.build_store(cnt_ok, cok).expect("store");
        }
        CollCache {
            slot,
            str_ok,
            str_ptr,
            str_len,
            cnt_ok,
            cnt_val,
        }
    }

    fn lower_sc_node(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        node: &Node,
        target: &LoopTarget<'_, 'ctx>,
    ) -> Result<(), Unsupported> {
        match node {
            Node::If(c, then_n, else_n) => {
                let then_bb = self.ctx.append_basic_block(f.func, "sc_then");
                let else_bb = self.ctx.append_basic_block(f.func, "sc_else");
                self.lower_sc_cond(f, s, c, then_bb, else_bb)?;
                self.builder.position_at_end(then_bb);
                self.lower_sc_node(f, s, then_n, target)?;
                self.builder.position_at_end(else_bb);
                self.lower_sc_node(f, s, else_n, target)
            }
            Node::Let(binds, body) => {
                self.lower_sc_binds(f, s, binds)?;
                self.lower_sc_node(f, s, body, target)
            }
            Node::Seq(stmts, body) => {
                for st in stmts.iter() {
                    self.lower_sc_stmt(f, s, st)?;
                }
                self.lower_sc_node(f, s, body, target)
            }
            Node::Leaf(leaf) => self.lower_sc_leaf(f, s, leaf, target),
            Node::Deopt => {
                self.builder
                    .build_unconditional_branch(s.deopt_bb)
                    .expect("br");
                Ok(())
            }
        }
    }

    fn lower_sc_binds(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        binds: &[(usize, IExpr)],
    ) -> Result<(), Unsupported> {
        for (slot, expr) in binds {
            let (k, v) = self.lower_sc_expr(f, s, expr)?;
            self.builder.build_store(s.k_slots[*slot], k).expect("st");
            self.builder.build_store(s.v_slots[*slot], v).expect("st");
        }
        Ok(())
    }

    fn lower_sc_leaf(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        leaf: &Leaf,
        target: &LoopTarget<'_, 'ctx>,
    ) -> Result<(), Unsupported> {
        match leaf {
            Leaf::Recur(ops) => self.lower_sc_recur(f, s, ops, target),
            Leaf::Exit(val) => self.lower_sc_exit(f, s, val, target),
        }
    }

    fn lower_sc_recur(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        ops: &[Option<IExpr>],
        target: &LoopTarget<'_, 'ctx>,
    ) -> Result<(), Unsupported> {
        let (slots, header) = match target {
            LoopTarget::Outer { header, .. } => (None, *header),
            LoopTarget::Inner { slots, header, .. } => (Some(*slots), *header),
        };
        let mut next = Vec::with_capacity(ops.len());
        for (j, op) in ops.iter().enumerate() {
            if let Some(e) = op {
                let (nk, nv) = self.lower_sc_expr(f, s, e)?;
                let sl = slots.map_or(j, |ss| ss[j]);
                next.push((sl, nk, nv));
            }
        }
        for (sl, nk, nv) in next {
            self.builder.build_store(s.k_slots[sl], nk).expect("st");
            self.builder.build_store(s.v_slots[sl], nv).expect("st");
        }
        self.builder.build_unconditional_branch(header).expect("br");
        Ok(())
    }

    fn lower_sc_exit(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        val: &ExitVal,
        target: &LoopTarget<'_, 'ctx>,
    ) -> Result<(), Unsupported> {
        match target {
            LoopTarget::Outer {
                scalar,
                done,
                dest,
                bound,
                ..
            } => {
                match val {
                    ExitVal::Const(v) => self.lower_const(f.func, v, *dest)?,
                    ExitVal::Box(loc) => {
                        let src = self.loc_slot(f, *loc);
                        self.clone_v(f.func, *dest, src);
                    }
                    ExitVal::Var(p) if !scalar[*p] => {
                        self.clone_v(f.func, *dest, bound[*p]);
                    }
                    ExitVal::Var(p) => {
                        let k = self.load_i8(s.k_slots[*p]);
                        let v = self.load_i64(s.v_slots[*p]);
                        self.box_scalar(f.func, *dest, k, v);
                    }
                    ExitVal::Expr(e) => {
                        let (k, v) = self.lower_sc_expr(f, s, e)?;
                        self.box_scalar(f.func, *dest, k, v);
                    }
                }
                self.builder.build_unconditional_branch(*done).expect("br");
            }
            LoopTarget::Inner {
                exit, res_k, res_v, ..
            } => {
                let (k, v) = match val {
                    ExitVal::Expr(e) => self.lower_sc_expr(f, s, e)?,
                    _ => (self.i8t.const_zero(), self.i64t.const_zero()),
                };
                self.builder.build_store(*res_k, k).expect("st");
                self.builder.build_store(*res_v, v).expect("st");
                self.builder.build_unconditional_branch(*exit).expect("br");
            }
        }
        Ok(())
    }

    fn lower_sc_stmt(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        st: &Stmt,
    ) -> Result<(), Unsupported> {
        match st {
            Stmt::Effect(eff) => self.lower_sc_effect(f, s, eff),
            Stmt::If(c, then_s, else_s) => self.lower_sc_stmt_if(f, s, c, then_s, else_s),
            Stmt::Let(binds, body) => {
                self.lower_sc_binds(f, s, binds)?;
                for x in body.iter() {
                    self.lower_sc_stmt(f, s, x)?;
                }
                Ok(())
            }
            Stmt::Loop(inner) => {
                self.lower_sc_inner(f, s, inner)?;
                Ok(())
            }
            Stmt::Expr(e) => {
                self.lower_sc_expr(f, s, e)?;
                Ok(())
            }
        }
    }

    fn lower_sc_stmt_if(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        c: &Cond,
        then_s: &[Stmt],
        else_s: &[Stmt],
    ) -> Result<(), Unsupported> {
        let then_bb = self.ctx.append_basic_block(f.func, "sc_st_then");
        let else_bb = self.ctx.append_basic_block(f.func, "sc_st_else");
        let join_bb = self.ctx.append_basic_block(f.func, "sc_st_join");
        self.lower_sc_cond(f, s, c, then_bb, else_bb)?;
        self.builder.position_at_end(then_bb);
        for x in then_s {
            self.lower_sc_stmt(f, s, x)?;
        }
        self.builder
            .build_unconditional_branch(join_bb)
            .expect("br");
        self.builder.position_at_end(else_bb);
        for x in else_s {
            self.lower_sc_stmt(f, s, x)?;
        }
        self.builder
            .build_unconditional_branch(join_bb)
            .expect("br");
        self.builder.position_at_end(join_bb);
        Ok(())
    }

    fn lower_sc_effect(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        eff: &Effect,
    ) -> Result<(), Unsupported> {
        match eff {
            Effect::Conj(loc, val_e) => self.lower_sc_conj(f, s, *loc, val_e),
            Effect::Assoc(loc, idx_e, val_e) => self.lower_sc_assoc(f, s, *loc, idx_e, val_e),
            Effect::Pop(loc) => self.lower_sc_pop(f, s, *loc),
        }
    }

    fn lower_sc_conj(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        loc: Loc,
        val_e: &IExpr,
    ) -> Result<(), Unsupported> {
        let (vk, vv) = self.lower_sc_expr(f, s, val_e)?;
        let c_slot = s.colls[s.coll_idx(loc)].slot;
        let try_bb = self.ctx.append_basic_block(f.func, "sc_conj_try");
        let slow_bb = self.ctx.append_basic_block(f.func, "sc_conj_slow");
        let done_bb = self.ctx.append_basic_block(f.func, "sc_conj_done");
        let is_i = self
            .builder
            .build_int_compare(IntPredicate::EQ, vk, self.i8t.const_zero(), "ci")
            .expect("cmp");
        self.builder
            .build_conditional_branch(is_i, try_bb, slow_bb)
            .expect("br");
        self.builder.position_at_end(try_bb);
        let ok = self
            .call(
                "rt_bc_scalar_conj",
                Some(self.i8t),
                &[c_slot.into(), vv.into()],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "hit")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, done_bb, slow_bb)
            .expect("br");
        self.builder.position_at_end(slow_bb);
        let tv = self.slot(f);
        let out = self.slot(f);
        self.box_scalar(f.func, tv, vk, vv);
        let k = self.var("clojure.core", "conj!");
        let call_bb = self.ctx.append_basic_block(f.func, "sc_conj_call");
        let slow_ok = self.ctx.append_basic_block(f.func, "sc_conj_ok");
        let fx_ok = self
            .call(
                "rt_n_conj_bang_fx",
                Some(self.i8t),
                &[c_slot.into(), tv.into()],
            )
            .expect("ok");
        let fx_hit = self
            .builder
            .build_int_compare(IntPredicate::NE, fx_ok, self.i8t.const_zero(), "fx")
            .expect("cmp");
        self.builder
            .build_conditional_branch(fx_hit, slow_ok, call_bb)
            .expect("br");
        self.emit_direct_slow(f, &[c_slot, tv], k, 2, Some("conj!"), out, call_bb, slow_ok);
        self.drop_and_nil_slots(f.func, &[out, tv]);
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        self.builder.position_at_end(done_bb);
        Ok(())
    }

    fn lower_sc_assoc(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        loc: Loc,
        idx_e: &IExpr,
        val_e: &IExpr,
    ) -> Result<(), Unsupported> {
        let (ik, iv) = self.lower_sc_expr(f, s, idx_e)?;
        let (vk, vv) = self.lower_sc_expr(f, s, val_e)?;
        let c_slot = s.colls[s.coll_idx(loc)].slot;
        let try_bb = self.ctx.append_basic_block(f.func, "sc_assoc_try");
        let slow_bb = self.ctx.append_basic_block(f.func, "sc_assoc_slow");
        let slow_ok = self.ctx.append_basic_block(f.func, "sc_assoc_ok");
        let done_bb = self.ctx.append_basic_block(f.func, "sc_assoc_done");
        let b_or = self.builder.build_or(ik, vk, "ikvk").expect("or");
        let both_i = self
            .builder
            .build_int_compare(IntPredicate::EQ, b_or, self.i8t.const_zero(), "bi")
            .expect("cmp");
        self.builder
            .build_conditional_branch(both_i, try_bb, slow_bb)
            .expect("br");
        self.builder.position_at_end(try_bb);
        let ok = self
            .call(
                "rt_bc_scalar_assoc",
                Some(self.i8t),
                &[c_slot.into(), iv.into(), vv.into()],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "hit")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, done_bb, slow_bb)
            .expect("br");
        self.builder.position_at_end(slow_bb);
        let ti = self.slot(f);
        let tv = self.slot(f);
        let out = self.slot(f);
        self.box_scalar(f.func, ti, ik, iv);
        self.box_scalar(f.func, tv, vk, vv);
        let k = self.var("clojure.core", "assoc!");
        let call_bb = self.ctx.append_basic_block(f.func, "sc_assoc_call");
        self.builder
            .build_unconditional_branch(call_bb)
            .expect("br");
        self.emit_direct_slow(
            f,
            &[c_slot, ti, tv],
            k,
            3,
            Some("assoc!"),
            out,
            call_bb,
            slow_ok,
        );
        self.drop_and_nil_slots(f.func, &[out, ti, tv]);
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        self.builder.position_at_end(done_bb);
        Ok(())
    }

    fn lower_sc_pop(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        loc: Loc,
    ) -> Result<(), Unsupported> {
        let c_slot = s.colls[s.coll_idx(loc)].slot;
        let slow_bb = self.ctx.append_basic_block(f.func, "sc_pop_slow");
        let slow_ok = self.ctx.append_basic_block(f.func, "sc_pop_ok");
        let done_bb = self.ctx.append_basic_block(f.func, "sc_pop_done");
        let ok = self
            .call("rt_bc_scalar_pop", Some(self.i8t), &[c_slot.into()])
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "hit")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, done_bb, slow_bb)
            .expect("br");
        let out = self.slot(f);
        let k = self.var("clojure.core", "pop!");
        self.emit_direct_slow(f, &[c_slot], k, 1, Some("pop!"), out, slow_bb, slow_ok);
        self.drop_and_nil_slots(f.func, &[out]);
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        self.builder.position_at_end(done_bb);
        Ok(())
    }

    fn lower_sc_inner(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        inner: &Inner,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        for (j, init_e) in inner.inits.iter().enumerate() {
            let (ik, iv) = self.lower_sc_expr(f, s, init_e)?;
            let sl = inner.slots[j];
            self.builder.build_store(s.k_slots[sl], ik).expect("st");
            self.builder.build_store(s.v_slots[sl], iv).expect("st");
        }
        let res_k = self.entry_alloca(f, self.i8t, "ink");
        let res_v = self.entry_alloca(f, self.i64t, "inv");
        let hdr = self.ctx.append_basic_block(f.func, "sc_in_hdr");
        let exit = self.ctx.append_basic_block(f.func, "sc_in_exit");
        self.builder.build_unconditional_branch(hdr).expect("br");
        self.builder.position_at_end(hdr);
        let t = LoopTarget::Inner {
            slots: &inner.slots,
            header: hdr,
            exit,
            res_k,
            res_v,
        };
        self.lower_sc_node(f, s, &inner.body, &t)?;
        self.builder.position_at_end(exit);
        Ok((self.load_i8(res_k), self.load_i64(res_v)))
    }

    fn lower_sc_cond(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        c: &Cond,
        yes: BasicBlock<'ctx>,
        no: BasicBlock<'ctx>,
    ) -> Result<(), Unsupported> {
        match c {
            Cond::Leaf(cmp) => {
                let bit = self.lower_sc_cmp(f, s, cmp)?;
                self.builder
                    .build_conditional_branch(bit, yes, no)
                    .expect("br");
                Ok(())
            }
            Cond::Not(inner) => self.lower_sc_cond(f, s, inner, no, yes),
            Cond::And(a, b) => {
                let mid = self.ctx.append_basic_block(f.func, "sc_and");
                self.lower_sc_cond(f, s, a, mid, no)?;
                self.builder.position_at_end(mid);
                self.lower_sc_cond(f, s, b, yes, no)
            }
            Cond::Or(a, b) => {
                let mid = self.ctx.append_basic_block(f.func, "sc_or");
                self.lower_sc_cond(f, s, a, yes, mid)?;
                self.builder.position_at_end(mid);
                self.lower_sc_cond(f, s, b, yes, no)
            }
        }
    }

    fn lower_sc_cmp(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        cmp: &Cmp,
    ) -> Result<IntValue<'ctx>, Unsupported> {
        match cmp {
            Cmp::Truthy(e) => {
                let (k, v) = self.lower_sc_expr(f, s, e)?;
                Ok(self.scalar_truthy(k, v))
            }
            Cmp::T1(op, e) => {
                let (k, v) = self.lower_sc_expr(f, s, e)?;
                self.require_int(f, k, s.deopt_bb);
                let name = match op {
                    IntTest1::Zero => "zero?",
                    IntTest1::Pos => "pos?",
                    IntTest1::Neg => "neg?",
                };
                Ok(self.int_pred(name, v))
            }
            Cmp::T2(op, a, b) => {
                let (ak, av) = self.lower_sc_expr(f, s, a)?;
                let (bk, bv) = self.lower_sc_expr(f, s, b)?;
                self.require_ints(f, ak, bk, s.deopt_bb);
                Ok(self.lower_sc_int_test2(*op, av, bv))
            }
            Cmp::Eq(a, b) => {
                let (ak, av) = self.lower_sc_expr(f, s, a)?;
                let (bk, bv) = self.lower_sc_expr(f, s, b)?;
                Ok(self.scalar_eq(ak, av, bk, bv))
            }
        }
    }

    fn scalar_truthy(&self, k: IntValue<'ctx>, v: IntValue<'ctx>) -> IntValue<'ctx> {
        let is_b = self
            .builder
            .build_int_compare(IntPredicate::EQ, k, self.i8t.const_int(2, false), "ib")
            .expect("cmp");
        let is_z = self
            .builder
            .build_int_compare(IntPredicate::EQ, v, self.i64t.const_zero(), "iz")
            .expect("cmp");
        let falsy = self.builder.build_and(is_b, is_z, "falsy").expect("and");
        self.builder.build_not(falsy, "truthy").expect("not")
    }

    fn scalar_eq(
        &self,
        ak: IntValue<'ctx>,
        av: IntValue<'ctx>,
        bk: IntValue<'ctx>,
        bv: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        let keq = self
            .builder
            .build_int_compare(IntPredicate::EQ, ak, bk, "keq")
            .expect("cmp");
        let veq = self
            .builder
            .build_int_compare(IntPredicate::EQ, av, bv, "veq")
            .expect("cmp");
        self.builder.build_and(keq, veq, "eq").expect("and")
    }

    fn lower_sc_int_test2(
        &self,
        op: IntTest2,
        av: IntValue<'ctx>,
        bv: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        let pred = match op {
            IntTest2::Lt => IntPredicate::SLT,
            IntTest2::Lte => IntPredicate::SLE,
            IntTest2::Gt => IntPredicate::SGT,
            IntTest2::Gte => IntPredicate::SGE,
            _ => IntPredicate::EQ,
        };
        self.builder
            .build_int_compare(pred, av, bv, "t2")
            .expect("cmp")
    }

    fn lower_sc_expr(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        e: &IExpr,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        match e {
            IExpr::Slot(sl) => Ok((self.load_i8(s.k_slots[*sl]), self.load_i64(s.v_slots[*sl]))),
            IExpr::Loc(loc) => Ok(s.load_val(*loc)),
            IExpr::Kst(sv) => {
                let k = match sv.k {
                    K::Int => 0,
                    K::Char => 1,
                    K::Bool => 2,
                };
                Ok((
                    self.i8t.const_int(k, false),
                    self.i64t.const_int(sv.v as u64, true),
                ))
            }
            IExpr::Calc(c) => self.lower_sc_calc(f, s, c),
            IExpr::Ctl(c) => self.lower_sc_ctl(f, s, c),
            IExpr::Deopt => {
                let dead = self.ctx.append_basic_block(f.func, "sc_dead");
                self.builder
                    .build_unconditional_branch(s.deopt_bb)
                    .expect("br");
                self.builder.position_at_end(dead);
                Ok((self.i8t.const_zero(), self.i64t.const_zero()))
            }
        }
    }

    fn lower_sc_ctl(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        ctl: &Ctl,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        match ctl {
            Ctl::If(c, then_e, else_e) => self.lower_sc_ctl_if(f, s, c, then_e, else_e),
            Ctl::Let(binds, body) => {
                self.lower_sc_binds(f, s, binds)?;
                self.lower_sc_expr(f, s, body)
            }
            Ctl::Seq(stmts, body) => {
                for st in stmts.iter() {
                    self.lower_sc_stmt(f, s, st)?;
                }
                self.lower_sc_expr(f, s, body)
            }
            Ctl::Loop(inner) => self.lower_sc_inner(f, s, inner),
            Ctl::Test(c) => self.lower_sc_ctl_test(f, s, c),
        }
    }

    fn lower_sc_ctl_if(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        c: &Cond,
        then_e: &IExpr,
        else_e: &IExpr,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        let rk = self.entry_alloca(f, self.i8t, "ifk");
        let rv = self.entry_alloca(f, self.i64t, "ifv");
        let then_bb = self.ctx.append_basic_block(f.func, "sc_if_then");
        let else_bb = self.ctx.append_basic_block(f.func, "sc_if_else");
        let join_bb = self.ctx.append_basic_block(f.func, "sc_if_join");
        self.lower_sc_cond(f, s, c, then_bb, else_bb)?;
        self.builder.position_at_end(then_bb);
        let (tk, tv) = self.lower_sc_expr(f, s, then_e)?;
        self.builder.build_store(rk, tk).expect("st");
        self.builder.build_store(rv, tv).expect("st");
        self.builder
            .build_unconditional_branch(join_bb)
            .expect("br");
        self.builder.position_at_end(else_bb);
        let (ek, ev) = self.lower_sc_expr(f, s, else_e)?;
        self.builder.build_store(rk, ek).expect("st");
        self.builder.build_store(rv, ev).expect("st");
        self.builder
            .build_unconditional_branch(join_bb)
            .expect("br");
        self.builder.position_at_end(join_bb);
        Ok((self.load_i8(rk), self.load_i64(rv)))
    }

    fn lower_sc_ctl_test(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        c: &Cond,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        let rv = self.entry_alloca(f, self.i64t, "tstv");
        let yes_bb = self.ctx.append_basic_block(f.func, "sc_tst_y");
        let no_bb = self.ctx.append_basic_block(f.func, "sc_tst_n");
        let join_bb = self.ctx.append_basic_block(f.func, "sc_tst_j");
        self.lower_sc_cond(f, s, c, yes_bb, no_bb)?;
        self.builder.position_at_end(yes_bb);
        self.builder
            .build_store(rv, self.i64t.const_int(1, false))
            .expect("st");
        self.builder
            .build_unconditional_branch(join_bb)
            .expect("br");
        self.builder.position_at_end(no_bb);
        self.builder
            .build_store(rv, self.i64t.const_zero())
            .expect("st");
        self.builder
            .build_unconditional_branch(join_bb)
            .expect("br");
        self.builder.position_at_end(join_bb);
        Ok((self.i8t.const_int(2, false), self.load_i64(rv)))
    }

    fn lower_sc_calc(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        calc: &Calc,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        match calc {
            Calc::Op1(op, a) => {
                let (ak, av) = self.lower_sc_expr(f, s, a)?;
                self.require_int(f, ak, s.deopt_bb);
                Ok((self.i8t.const_zero(), self.lower_sc_op1(f, s, *op, av)))
            }
            Calc::Op2(op, a, b) => {
                let (ak, av) = self.lower_sc_expr(f, s, a)?;
                let (bk, bv) = self.lower_sc_expr(f, s, b)?;
                self.require_ints(f, ak, bk, s.deopt_bb);
                Ok((self.i8t.const_zero(), self.lower_sc_op2(f, s, *op, av, bv)))
            }
            Calc::Cast(c, a) => {
                let (ak, av) = self.lower_sc_expr(f, s, a)?;
                Ok(self.lower_sc_cast(f, s, *c, ak, av))
            }
            Calc::Count(loc) => Ok(self.lower_sc_count(f, s, *loc)),
            Calc::Nth(loc, idx) => self.lower_sc_nth(f, s, *loc, idx),
            Calc::Nth2(loc, idx, item) => self.lower_sc_nth2(f, s, *loc, idx, *item),
        }
    }

    fn lower_sc_op1(
        &self,
        f: &FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        op: IntOp1,
        x: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        let one = self.i64t.const_int(1, false);
        let zero = self.i64t.const_zero();
        match op {
            IntOp1::Inc => self.checked_i64_op(f, "llvm.sadd.with.overflow", x, one, s.deopt_bb),
            IntOp1::Dec => self.checked_i64_op(f, "llvm.ssub.with.overflow", x, one, s.deopt_bb),
            IntOp1::Negate => {
                self.checked_i64_op(f, "llvm.ssub.with.overflow", zero, x, s.deopt_bb)
            }
            IntOp1::BitNot => self.builder.build_not(x, "bnot").expect("not"),
        }
    }

    fn lower_sc_op2(
        &self,
        f: &FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        op: IntOp2,
        x: IntValue<'ctx>,
        y: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        if let Some(intr) = checked_op2_intrinsic(op) {
            return self.checked_i64_op(f, intr, x, y, s.deopt_bb);
        }
        self.int_binop(f, op2_binop_name(op), x, y, s.deopt_bb)
    }

    fn checked_i64_op(
        &self,
        f: &FnCtx<'ctx>,
        intr_name: &str,
        x: IntValue<'ctx>,
        y: IntValue<'ctx>,
        fail: BasicBlock<'ctx>,
    ) -> IntValue<'ctx> {
        let intr = inkwell::intrinsics::Intrinsic::find(intr_name).expect("intrinsic");
        let decl = intr
            .get_declaration(&self.module, &[self.i64t.into()])
            .expect("decl");
        let pair = self
            .builder
            .build_call(decl, &[x.into(), y.into()], "ov")
            .expect("call")
            .try_as_basic_value()
            .expect_basic("pair")
            .into_struct_value();
        let r = self
            .builder
            .build_extract_value(pair, 0, "r")
            .expect("ext")
            .into_int_value();
        let o = self
            .builder
            .build_extract_value(pair, 1, "o")
            .expect("ext")
            .into_int_value();
        let ok_bb = self.ctx.append_basic_block(f.func, "sc_ov_ok");
        self.builder
            .build_conditional_branch(o, fail, ok_bb)
            .expect("br");
        self.builder.position_at_end(ok_bb);
        r
    }

    fn lower_sc_cast(
        &self,
        f: &FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        c: Cast,
        ak: IntValue<'ctx>,
        av: IntValue<'ctx>,
    ) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let b = &self.builder;
        let is_bool = b
            .build_int_compare(IntPredicate::EQ, ak, self.i8t.const_int(2, false), "cb")
            .expect("cmp");
        let ok_kind = match c {
            Cast::Long => b.build_not(is_bool, "cl_ok").expect("not"),
            Cast::Int => {
                let is_chr = b
                    .build_int_compare(IntPredicate::EQ, ak, self.i8t.const_int(1, false), "cc")
                    .expect("cmp");
                let is_int = b
                    .build_int_compare(IntPredicate::EQ, ak, self.i8t.const_zero(), "ci")
                    .expect("cmp");
                let t32 = b.build_int_truncate(av, self.i32t, "t32").expect("trunc");
                let s64 = b.build_int_s_extend(t32, self.i64t, "s64").expect("sext");
                let fits = b
                    .build_int_compare(IntPredicate::EQ, s64, av, "fits")
                    .expect("cmp");
                let ok_i = b.build_and(is_int, fits, "oki").expect("and");
                b.build_or(is_chr, ok_i, "ci_ok").expect("or")
            }
        };
        let ok_bb = self.ctx.append_basic_block(f.func, "sc_cast_ok");
        b.build_conditional_branch(ok_kind, ok_bb, s.deopt_bb)
            .expect("br");
        self.builder.position_at_end(ok_bb);
        (self.i8t.const_zero(), av)
    }

    fn lower_sc_count(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        loc: Loc,
    ) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let c = &s.colls[s.coll_idx(loc)];
        let c_slot = c.slot;
        let cnt_ok = self.load_i8(c.cnt_ok);
        let cnt_val = c.cnt_val;
        let res_v = self.entry_alloca(f, self.i64t, "cntres");
        let hit_bb = self.ctx.append_basic_block(f.func, "sc_cnt_hit");
        let bc_bb = self.ctx.append_basic_block(f.func, "sc_cnt_bc");
        let bc_ok_bb = self.ctx.append_basic_block(f.func, "sc_cnt_bcok");
        let slow_bb = self.ctx.append_basic_block(f.func, "sc_cnt_slow");
        let slow_ok = self.ctx.append_basic_block(f.func, "sc_cnt_slok");
        let done_bb = self.ctx.append_basic_block(f.func, "sc_cnt_done");
        let is_cached = self
            .builder
            .build_int_compare(IntPredicate::NE, cnt_ok, self.i8t.const_zero(), "cc")
            .expect("cmp");
        self.builder
            .build_conditional_branch(is_cached, hit_bb, bc_bb)
            .expect("br");
        self.builder.position_at_end(hit_bb);
        let cv = self.load_i64(cnt_val);
        self.builder.build_store(res_v, cv).expect("st");
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        self.builder.position_at_end(bc_bb);
        let probe = self
            .helper("rt_n_str_ascii", Some(self.i8t.into()), &[self.ptr.into()])
            .as_global_value()
            .as_pointer_value();
        let ok = self
            .call(
                "rt_bc_scalar_count",
                Some(self.i8t),
                &[c_slot.into(), s.scratch_v.into(), probe.into()],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "hit")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, bc_ok_bb, slow_bb)
            .expect("br");
        self.builder.position_at_end(bc_ok_bb);
        let bv = self.load_i64(s.scratch_v);
        self.builder.build_store(res_v, bv).expect("st");
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        let tmp_out = self.slot(f);
        let k_cnt = self.var("clojure.core", "count");
        self.emit_direct_slow(
            f,
            &[c_slot],
            k_cnt,
            1,
            Some("count"),
            tmp_out,
            slow_bb,
            slow_ok,
        );
        let sv = self.int_payload(tmp_out);
        self.builder.build_store(res_v, sv).expect("st");
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        self.builder.position_at_end(done_bb);
        (self.i8t.const_zero(), self.load_i64(res_v))
    }

    fn lower_sc_nth(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        loc: Loc,
        idx_e: &IExpr,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        let (ik, iv) = self.lower_sc_expr(f, s, idx_e)?;
        self.require_int(f, ik, s.deopt_bb);
        let c = &s.colls[s.coll_idx(loc)];
        let c_slot = c.slot;
        let str_ok = self.load_i8(c.str_ok);
        let str_len = self.load_i64(c.str_len);
        let str_ptr_slot = c.str_ptr;
        let res_k = self.entry_alloca(f, self.i8t, "nthk");
        let res_v = self.entry_alloca(f, self.i64t, "nthv");
        let is_ascii = self
            .builder
            .build_int_compare(IntPredicate::NE, str_ok, self.i8t.const_zero(), "asc")
            .expect("cmp");
        let in_b = self
            .builder
            .build_int_compare(IntPredicate::ULT, iv, str_len, "inb")
            .expect("cmp");
        let fast_str = self.builder.build_and(is_ascii, in_b, "fstr").expect("and");
        let str_bb = self.ctx.append_basic_block(f.func, "sc_nth_str");
        let bc_bb = self.ctx.append_basic_block(f.func, "sc_nth_bc");
        let bc_ok_bb = self.ctx.append_basic_block(f.func, "sc_nth_bcok");
        let slow_bb = self.ctx.append_basic_block(f.func, "sc_nth_slow");
        let done_bb = self.ctx.append_basic_block(f.func, "sc_nth_done");
        self.builder
            .build_conditional_branch(fast_str, str_bb, bc_bb)
            .expect("br");
        self.builder.position_at_end(str_bb);
        let base = self
            .builder
            .build_load(self.ptr, str_ptr_slot, "sp")
            .expect("ld")
            .into_pointer_value();
        let addr = self
            .builder
            .build_int_add(
                self.builder
                    .build_ptr_to_int(base, self.i64t, "p2i")
                    .expect("p2i"),
                iv,
                "addr",
            )
            .expect("add");
        let bp = self
            .builder
            .build_int_to_ptr(addr, self.ptr, "i2p")
            .expect("i2p");
        let ch = self
            .builder
            .build_int_z_extend(self.load_i8(bp), self.i64t, "ch64")
            .expect("zext");
        self.builder
            .build_store(res_k, self.i8t.const_int(1, false))
            .expect("st");
        self.builder.build_store(res_v, ch).expect("st");
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        self.builder.position_at_end(bc_bb);
        let probe = self
            .helper("rt_n_str_ascii", Some(self.i8t.into()), &[self.ptr.into()])
            .as_global_value()
            .as_pointer_value();
        let ok = self
            .call(
                "rt_bc_scalar_nth",
                Some(self.i8t),
                &[
                    c_slot.into(),
                    iv.into(),
                    s.scratch_k.into(),
                    s.scratch_v.into(),
                    probe.into(),
                ],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "hit")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, bc_ok_bb, slow_bb)
            .expect("br");
        self.builder.position_at_end(bc_ok_bb);
        self.builder
            .build_store(res_k, self.load_i8(s.scratch_k))
            .expect("st");
        self.builder
            .build_store(res_v, self.load_i64(s.scratch_v))
            .expect("st");
        self.builder
            .build_unconditional_branch(done_bb)
            .expect("br");
        self.emit_sc_nth_slow(f, s, c_slot, iv, res_k, res_v, slow_bb, done_bb);
        self.builder.position_at_end(done_bb);
        Ok((self.load_i8(res_k), self.load_i64(res_v)))
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_sc_nth_slow(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        c_slot: PointerValue<'ctx>,
        iv: IntValue<'ctx>,
        res_k: PointerValue<'ctx>,
        res_v: PointerValue<'ctx>,
        slow_bb: BasicBlock<'ctx>,
        done_bb: BasicBlock<'ctx>,
    ) {
        self.builder.position_at_end(slow_bb);
        let tmp_idx = self.slot(f);
        let tmp_out = self.slot(f);
        self.write_int(f.func, tmp_idx, iv);
        let k_nth = self.var("clojure.core", "nth");
        let call_bb = self.ctx.append_basic_block(f.func, "sc_nth_call");
        let slow_ok = self.ctx.append_basic_block(f.func, "sc_nth_slok");
        self.builder
            .build_unconditional_branch(call_bb)
            .expect("br");
        self.emit_direct_slow(
            f,
            &[c_slot, tmp_idx],
            k_nth,
            2,
            Some("nth"),
            tmp_out,
            call_bb,
            slow_ok,
        );
        let (u_ok, uk, uv) = self.unbox_scalar(tmp_out);
        self.drop_and_nil_slots(f.func, &[tmp_out]);
        self.builder.build_store(res_k, uk).expect("st");
        self.builder.build_store(res_v, uv).expect("st");
        self.builder
            .build_conditional_branch(u_ok, done_bb, s.deopt_bb)
            .expect("br");
    }

    fn lower_sc_nth2(
        &mut self,
        f: &mut FnCtx<'ctx>,
        s: &ScalarSession<'ctx>,
        loc: Loc,
        idx_e: &IExpr,
        item: u16,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), Unsupported> {
        let (ik, iv) = self.lower_sc_expr(f, s, idx_e)?;
        self.require_int(f, ik, s.deopt_bb);
        let c_slot = s.colls[s.coll_idx(loc)].slot;
        let item_v = self.i64t.const_int(u64::from(item), false);
        let ok = self
            .call(
                "rt_bc_scalar_nth2",
                Some(self.i8t),
                &[
                    c_slot.into(),
                    iv.into(),
                    item_v.into(),
                    s.scratch_k.into(),
                    s.scratch_v.into(),
                ],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "hit")
            .expect("cmp");
        let ok_bb = self.ctx.append_basic_block(f.func, "sc_nth2_ok");
        self.builder
            .build_conditional_branch(hit, ok_bb, s.deopt_bb)
            .expect("br");
        self.builder.position_at_end(ok_bb);
        Ok((self.load_i8(s.scratch_k), self.load_i64(s.scratch_v)))
    }

    fn require_int(&self, f: &FnCtx<'ctx>, k: IntValue<'ctx>, fail: BasicBlock<'ctx>) {
        let is_i = self
            .builder
            .build_int_compare(IntPredicate::EQ, k, self.i8t.const_zero(), "req_i")
            .expect("cmp");
        let ok_bb = self.ctx.append_basic_block(f.func, "sc_int_ok");
        self.builder
            .build_conditional_branch(is_i, ok_bb, fail)
            .expect("br");
        self.builder.position_at_end(ok_bb);
    }

    fn require_ints(
        &self,
        f: &FnCtx<'ctx>,
        k1: IntValue<'ctx>,
        k2: IntValue<'ctx>,
        fail: BasicBlock<'ctx>,
    ) {
        let both = self.builder.build_or(k1, k2, "req_or").expect("or");
        self.require_int(f, both, fail);
    }
}

fn loc_reg_idx(loc: Loc) -> usize {
    match loc {
        Loc::Reg(idx) => idx,
        Loc::Up(_) | Loc::Rec(_) => usize::MAX,
    }
}

fn checked_op2_intrinsic(op: IntOp2) -> Option<&'static str> {
    match op {
        IntOp2::Add => Some("llvm.sadd.with.overflow"),
        IntOp2::Sub => Some("llvm.ssub.with.overflow"),
        IntOp2::Mul => Some("llvm.smul.with.overflow"),
        _ => None,
    }
}

fn op2_binop_name(op: IntOp2) -> &'static str {
    const NAMES: [(&IntOp2, &str); 11] = [
        (&IntOp2::Quot, "quot"),
        (&IntOp2::Rem, "rem"),
        (&IntOp2::Mod, "mod"),
        (&IntOp2::Max, "max"),
        (&IntOp2::Min, "min"),
        (&IntOp2::BitAnd, "bit-and"),
        (&IntOp2::BitOr, "bit-or"),
        (&IntOp2::BitXor, "bit-xor"),
        (&IntOp2::Shl, "bit-shift-left"),
        (&IntOp2::Shr, "bit-shift-right"),
        (&IntOp2::Ushr, "unsigned-bit-shift-right"),
    ];
    NAMES
        .iter()
        .find(|(k, _)| **k == op)
        .map(|(_, v)| *v)
        .expect("non-checked IntOp2")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loc_reg_idx_maps_non_reg_to_max() {
        assert_eq!(loc_reg_idx(Loc::Reg(2)), 2);
        assert_eq!(loc_reg_idx(Loc::Up(0)), usize::MAX);
        assert_eq!(loc_reg_idx(Loc::Rec(0)), usize::MAX);
    }
}
