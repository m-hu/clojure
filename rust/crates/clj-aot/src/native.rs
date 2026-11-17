// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Native lowering of analyzed programs (issue #259 S1).
//!
//! Input: the build front end's top-level units ([`clj_eval::aot_front`]),
//! i.e. the interpreter's own analyzed [`Expr`] trees. Output: one
//! LLVM function per top-level unit and per fn arity, a constant
//! initializer, and a `main` that hands them to the `clj-rt` native
//! runtime (`rt_n_*`).
//!
//! Value ABI: every value lives by value in a 32-byte slot
//! (`[4 x i64]`), zero-initialized (= nil) at function entry and
//! dropped at both exits. Helpers borrow inputs and replace outputs.
//! A failing helper returns 1; generated code branches to the
//! function's fail block, which drops every slot and returns 1.
//!
//! Locals follow the analyzer's absolute slot numbering: `Local(i)`
//! indexes the frame chain, so a fn created at frame length `base`
//! captures exactly the `Local(i < base)` its arities mention.

use std::collections::{BTreeSet, HashMap};

use clj_eval::aot_front::{FrontEnd, TopUnit, for_each_child, names_lib, pattern_children};
use clj_eval::expr::{Arity, CaseTest, Expr, FunExpr, NsClause, Pattern};
use clj_eval::native_abi;
use clj_value::{Map, Symbol, Value, Vector};
use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::{
    ArrayType, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType, IntType, PointerType,
};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, FunctionValue, IntValue, PointerValue};

#[path = "scalar.rs"]
mod scalar;

/// "No index" for optional table indices (matches `clj-rt`).
const NONE: u64 = u64::MAX;

/// `Value` tags the inline paths test (pinned by `value_layout_abi`).
const TAG_INT: u64 = 2;
const TAG_CHAR: u64 = 9;

/// Why a program cannot (yet) be lowered natively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// A constant-pool entry.
#[derive(Clone, Debug, PartialEq)]
enum Const {
    // (`nil`, booleans, ints, doubles, and chars are written inline,
    // never pooled.)
    Str(String),
    Keyword(Option<String>, String),
    Symbol(Option<String>, String),
    Regex(String),
    StaticField(String, String),
    Class(String),
    Read(String),
    /// A symbol whose metadata maps keywords to simple values (`defn-`'s
    /// `^{:private true}` names), built without the reader (#271). Its
    /// metadata is self-contained: the init cannot read other pool
    /// constants before the pool freezes.
    MetaSymbol {
        ns: Option<String>,
        name: String,
        meta: Vec<(MetaVal, MetaVal)>,
    },
    /// A metadata-free collection of simple forms (quoted data, catch
    /// class lists, `:arglists`), built in place by the reader's own
    /// constructors (#271); `rt_n_coll`'s kind code.
    Coll(u64, Vec<ConstForm>),
    /// An empty collection literal without metadata (`[]`, `{}`, `#{}`):
    /// one value per form, as under `run` — built once by its literal
    /// shim, `site` keeping each form's value its own.
    EmptyColl {
        helper: &'static str,
        site: usize,
    },
}

/// A [`Const::MetaSymbol`] metadata key or value, built in place: a
/// scalar written inline, or a keyword, symbol, or string made by name.
#[derive(Clone, Debug, PartialEq)]
enum MetaVal {
    Inline(Value),
    Keyword(Option<String>, String),
    Symbol(Option<String>, String),
    Str(String),
}

/// One item of a [`Const::Coll`], built in place.
#[derive(Clone, Debug, PartialEq)]
enum ConstForm {
    Simple(MetaVal),
    MetaSym(Option<String>, String, Vec<(MetaVal, MetaVal)>),
    Coll(u64, Vec<ConstForm>),
}

/// How many forms a [`Const::Coll`] may hold (larger ones read back:
/// their text is the compact encoding).
const COLL_FORMS_MAX: usize = 256;

/// The [`MetaVal`] for `v` (`None`: not a simple metadata value).
fn meta_val(v: &Value) -> Option<MetaVal> {
    Some(match v {
        Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Double(_) | Value::Char(_) => {
            MetaVal::Inline(v.clone())
        }
        Value::Keyword(k) => {
            MetaVal::Keyword(k.namespace().map(str::to_owned), k.name().to_owned())
        }
        Value::Symbol(s) if s.meta().is_none() => {
            MetaVal::Symbol(s.namespace().map(str::to_owned), s.name().to_owned())
        }
        Value::Str(s) => MetaVal::Str(s.to_string()),
        _ => return None,
    })
}

/// One compiled arity: `(fixed, variadic, code)`.
type ArityCode<'ctx> = (usize, bool, FunctionValue<'ctx>);

/// One compiled fn: display plus its arities.
struct FnInfo<'ctx> {
    display: String,
    arities: Vec<(usize, bool, FunctionValue<'ctx>)>,
}

/// How a local slot index is reached inside the current function.
#[derive(Clone, Copy)]
enum LocalRef<'ctx> {
    /// A slot of this function.
    Slot(PointerValue<'ctx>),
    /// A `letfn` cell held in a slot of this function.
    Cell(PointerValue<'ctx>),
    /// Captured value `i` of this closure.
    Captured(usize),
    /// Captured `letfn` cell `i` of this closure.
    CapturedCell(usize),
}

/// A `recur` target: its loop header and the patterns rebound from
/// the recur values (each pattern's slots in binding order).
struct RecurTarget<'ctx> {
    header: BasicBlock<'ctx>,
    patterns: Vec<(Pattern, Vec<PointerValue<'ctx>>)>,
}

/// How a lowered boolean/comparison operation delivers its `i1` result.
enum BoolSink<'ctx> {
    Slot {
        dest: PointerValue<'ctx>,
        done: BasicBlock<'ctx>,
    },
    Jump {
        join: BasicBlock<'ctx>,
        phi: inkwell::values::PhiValue<'ctx>,
    },
}

/// Per-function lowering state.
struct FnCtx<'ctx> {
    func: FunctionValue<'ctx>,
    entry: BasicBlock<'ctx>,
    fail: BasicBlock<'ctx>,
    slots: Vec<PointerValue<'ctx>>,
    env: Vec<Option<LocalRef<'ctx>>>,
    recur: Vec<RecurTarget<'ctx>>,
    captured: PointerValue<'ctx>,
    /// `Expr::Local` occurrences `(node address, local)` in the `recur`
    /// args being lowered that are their local's last use, so may move.
    /// Such a local is never borrowed or aliased at an earlier
    /// occurrence (the move would invalidate the reference).
    moves: Vec<(usize, usize)>,
}

/// The program-level lowering session.
pub struct Lowerer<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    ptr: PointerType<'ctx>,
    i8t: IntType<'ctx>,
    i32t: IntType<'ctx>,
    i64t: IntType<'ctx>,
    slot_ty: ArrayType<'ctx>,
    code_ty: FunctionType<'ctx>,
    consts: Vec<Const>,
    const_index: HashMap<String, usize>,
    vars: Vec<(String, String)>,
    var_index: HashMap<(String, String), usize>,
    fns: Vec<FnInfo<'ctx>>,
    strings: HashMap<String, PointerValue<'ctx>>,
    byte_strings: HashMap<String, PointerValue<'ctx>>,
    fn_seq: usize,
    /// Inline the analyzer's `FastArith` int ops (off when the program
    /// can rebind vars, so core arithmetic always means the native).
    inline_arith: bool,
    /// Whether `def`s keep their whole name metadata: always under the
    /// full boot; in a Lite program only when it references a reader of
    /// var metadata ([`program_reads_var_meta`]) — otherwise the
    /// runtime keeps no `def` metadata it could see (#271).
    var_meta: bool,
    /// Boot the compiled binary via `rt_n_main_lite` / `rt_n_ns_lite` (no
    /// embedded stdlib `core_lazy` evaluation) when the program uses only
    /// Lite-safe core/string natives and user-defined vars.
    lite: bool,
    /// Emit and link LLVM bitcode runtime helpers (`clj-bc`) instead
    /// of going straight to out-of-line `rt_n_*` shims.
    use_bitcode: bool,
    /// Global slot initialized on entry to `cljtop_0` holding `&FastVars`
    /// (`None` when `use_bitcode` is false).
    fast_vars: Option<PointerValue<'ctx>>,
    /// Top-level `defn`s reserved before lowering (by `FunExpr`
    /// address): fn index and pre-declared arity functions.
    reserved: HashMap<usize, (usize, Vec<FunctionValue<'ctx>>)>,
    /// Direct-call targets by var: fn index and `(fixed, variadic,
    /// code)` arities.
    direct: HashMap<(String, String), (usize, Vec<ArityCode<'ctx>>)>,
    /// Single-def top-level fn arities eligible for scalar-loop callee inlining.
    callees: HashMap<(String, String), std::sync::Arc<[Arity]>>,
}

impl<'ctx> Lowerer<'ctx> {
    /// A fresh session selecting bitcode helpers (`true`) vs out-of-line
    /// `rt_n_*` fallback (`false`).
    #[must_use]
    pub fn with_bitcode(ctx: &'ctx Context, use_bitcode: bool) -> Self {
        let ptr = ctx.ptr_type(AddressSpace::default());
        let i64t = ctx.i64_type();
        let i32t = ctx.i32_type();
        let code_ty = i32t.fn_type(&[ptr.into(), ptr.into(), i64t.into(), ptr.into()], false);
        let module = ctx.create_module("clj_native");
        let fast_vars = use_bitcode.then(|| {
            let g = module.add_global(ptr, None, "clj_fast_vars");
            g.set_initializer(&ptr.const_null());
            g.set_linkage(Linkage::Internal);
            g.as_pointer_value()
        });
        Self {
            ctx,
            module,
            builder: ctx.create_builder(),
            ptr,
            i8t: ctx.i8_type(),
            i32t,
            i64t,
            slot_ty: i64t.array_type(4),
            code_ty,
            consts: Vec::new(),
            const_index: HashMap::new(),
            vars: Vec::new(),
            var_index: HashMap::new(),
            fns: Vec::new(),
            strings: HashMap::new(),
            byte_strings: HashMap::new(),
            fn_seq: 0,
            inline_arith: true,
            var_meta: true,
            lite: false,
            use_bitcode,
            fast_vars,
            reserved: HashMap::new(),
            direct: HashMap::new(),
            callees: HashMap::new(),
        }
    }

    /// Lower a whole program to a verified module.
    pub fn lower_program(
        mut self,
        front: &FrontEnd,
        path: &str,
    ) -> Result<Module<'ctx>, Unsupported> {
        self.lite = can_lite_program(front);
        // A full boot loads the spliced libraries through the program's
        // own `require` (#271): their compiled units drop.
        let units = if self.lite {
            &front.units[..]
        } else {
            &front.units[front.lib_units..]
        };
        self.inline_arith = !units.iter().any(|u| mentions_rebinding(&u.expr));
        self.var_meta = !self.lite || program_reads_var_meta(units);
        self.reserve_defns(units);
        let mut tops = Vec::with_capacity(units.len());
        for unit in units {
            if self.drops(&unit.expr) {
                continue;
            }
            let i = tops.len();
            tops.push(self.lower_thunk(&format!("cljtop_{i}"), &unit.expr, i == 0)?);
        }
        let init = self.lower_const_init();
        self.lower_main(path, init, &tops);
        self.lower_frame_registry_wraps();
        crate::lower::verify_module(&self.module).map_err(Unsupported)?;
        Ok(self.module)
    }

    /// Reserve every top-level `(def name (fn …))` whose var is defined
    /// exactly once: calls through such a var may go direct.
    fn reserve_defns(&mut self, units: &[TopUnit]) {
        let key = |sym: &Symbol| {
            (
                sym.namespace().unwrap_or("user").to_owned(),
                sym.name().to_owned(),
            )
        };
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for u in units {
            if let Expr::Def { sym, .. } | Expr::DefMacro { sym, .. } = &u.expr {
                *counts.entry(key(sym)).or_default() += 1;
            }
            if let Expr::Declare { syms, .. } = &u.expr {
                for sym in syms {
                    *counts.entry(key(sym)).or_default() += 0;
                }
            }
        }
        for u in units {
            let Expr::Def {
                sym,
                init: Some(init),
                ..
            } = &u.expr
            else {
                continue;
            };
            let Expr::Fn(fun) = init.as_ref() else {
                continue;
            };
            if counts.get(&key(sym)) != Some(&1) {
                continue;
            }
            let k = self.fns.len();
            self.fns.push(FnInfo {
                display: fun.display.clone(),
                arities: Vec::new(),
            });
            let mut funcs = Vec::new();
            let mut arities = Vec::new();
            for (i, arity) in fun.arities.iter().enumerate() {
                self.fn_seq += 1;
                let name = format!("cljfn_{}_{i}_{}", self.fn_seq, sanitize(&fun.display));
                let func = self
                    .module
                    .add_function(&name, self.code_ty, Some(Linkage::Private));
                funcs.push(func);
                arities.push((arity.fixed.len(), arity.rest.is_some(), func));
            }
            self.reserved
                .insert(std::ptr::from_ref(fun) as usize, (k, funcs));
            self.direct.insert(key(sym), (k, arities));
            if self.inline_arith {
                self.callees
                    .insert(key(sym), std::sync::Arc::clone(&fun.arities));
            }
        }
    }

    // ---- helpers: declarations, constants, strings ---------------------

    fn helper(
        &self,
        name: &str,
        ret: Option<BasicTypeEnum<'ctx>>,
        params: &[BasicMetadataTypeEnum<'ctx>],
    ) -> FunctionValue<'ctx> {
        if let Some(f) = self.module.get_function(name) {
            return f;
        }
        let ty = match ret {
            Some(r) => r.fn_type(params, false),
            None => self.ctx.void_type().fn_type(params, false),
        };
        self.module.add_function(name, ty, Some(Linkage::External))
    }

    fn call_raw(
        &self,
        name: &str,
        ret: Option<BasicTypeEnum<'ctx>>,
        args: &[BasicMetadataValueEnum<'ctx>],
    ) -> inkwell::values::CallSiteValue<'ctx> {
        let params: Vec<BasicMetadataTypeEnum> = args
            .iter()
            .map(|a| match a {
                BasicMetadataValueEnum::IntValue(v) => v.get_type().into(),
                BasicMetadataValueEnum::FloatValue(v) => v.get_type().into(),
                _ => self.ptr.into(),
            })
            .collect();
        let f = self.helper(name, ret, &params);
        self.builder.build_call(f, args, "").expect("call")
    }

    fn call_f64(
        &self,
        name: &str,
        args: &[BasicMetadataValueEnum<'ctx>],
    ) -> inkwell::values::FloatValue<'ctx> {
        self.call_raw(name, Some(self.ctx.f64_type().into()), args)
            .try_as_basic_value()
            .expect_basic("f64")
            .into_float_value()
    }

    fn call(
        &self,
        name: &str,
        ret: Option<IntType<'ctx>>,
        args: &[BasicMetadataValueEnum<'ctx>],
    ) -> Option<IntValue<'ctx>> {
        let site = self.call_raw(name, ret.map(Into::into), args);
        ret.map(|_| {
            site.try_as_basic_value()
                .expect_basic("status")
                .into_int_value()
        })
    }

    fn call_ptr(&self, name: &str, args: &[BasicMetadataValueEnum<'ctx>]) -> PointerValue<'ctx> {
        let site = self.call_raw(name, Some(self.ptr.into()), args);
        site.try_as_basic_value()
            .expect_basic("ptr")
            .into_pointer_value()
    }

    fn u64c(&self, n: u64) -> BasicMetadataValueEnum<'ctx> {
        self.i64t.const_int(n, false).into()
    }

    /// A private constant global holding `value`.
    fn const_global(&mut self, value: &dyn BasicValue<'ctx>, prefix: &str) -> PointerValue<'ctx> {
        self.fn_seq += 1;
        let v = value.as_basic_value_enum();
        let g = self
            .module
            .add_global(v.get_type(), None, &format!("{prefix}{}", self.fn_seq));
        g.set_initializer(&v);
        g.set_constant(true);
        g.set_linkage(Linkage::Private);
        g.as_pointer_value()
    }

    /// The `{ptr, len}` header type of a `clj_rt::nabi::CSlice`.
    fn slice_ty(&self) -> inkwell::types::StructType<'ctx> {
        self.ctx
            .struct_type(&[self.ptr.into(), self.i64t.into()], false)
    }

    /// A constant `CSlice` header over `len` items at `base`.
    fn const_slice(&mut self, base: PointerValue<'ctx>, len: u64) -> PointerValue<'ctx> {
        let header = self
            .slice_ty()
            .const_named_struct(&[base.into(), self.i64t.const_int(len, false).into()]);
        self.const_global(&header, "cljh")
    }

    /// A `CSlice` header over `n` slots at `base` (a slot array of
    /// `f`), built once in the entry block.
    fn slots_ref(
        &mut self,
        f: &FnCtx<'ctx>,
        base: PointerValue<'ctx>,
        n: usize,
    ) -> PointerValue<'ctx> {
        if n == 0 {
            return self.const_slice(self.ptr.const_null(), 0);
        }
        let here = self.builder.get_insert_block().expect("positioned");
        self.builder.position_at_end(f.entry);
        let ty = self.slice_ty();
        let h = self.builder.build_alloca(ty, "sl").expect("alloca");
        let p0 = self.builder.build_struct_gep(ty, h, 0, "sp").expect("gep");
        self.builder.build_store(p0, base).expect("store");
        let p1 = self.builder.build_struct_gep(ty, h, 1, "sn").expect("gep");
        self.builder
            .build_store(p1, self.i64t.const_int(n as u64, false))
            .expect("store");
        self.builder.position_at_end(here);
        h
    }

    /// A constant `CSlice<u8>` header over `text` (cached).
    fn str_ptr(&mut self, text: &str) -> PointerValue<'ctx> {
        if let Some(p) = self.strings.get(text) {
            return *p;
        }
        let bytes = self.str_bytes(text);
        let p = self.const_slice(bytes, text.len() as u64);
        self.strings.insert(text.to_owned(), p);
        p
    }

    /// A private constant global holding `text`'s bytes (cached).
    fn str_bytes(&mut self, text: &str) -> PointerValue<'ctx> {
        if let Some(p) = self.byte_strings.get(text) {
            return *p;
        }
        let value = self.ctx.const_string(text.as_bytes(), false);
        let bytes = self.const_global(&value, "cljs");
        self.byte_strings.insert(text.to_owned(), bytes);
        bytes
    }

    /// A `CSlice<u8>` header value (not a global) over `text`.
    fn str_slice_value(&mut self, text: &str) -> inkwell::values::StructValue<'ctx> {
        let bytes = self.str_bytes(text);
        self.slice_ty().const_named_struct(&[
            bytes.into(),
            self.i64t.const_int(text.len() as u64, false).into(),
        ])
    }

    fn constant(&mut self, c: Const) -> u64 {
        let key = format!("{c:?}");
        if let Some(k) = self.const_index.get(&key) {
            return *k as u64;
        }
        let k = self.consts.len();
        self.consts.push(c);
        self.const_index.insert(key, k);
        k as u64
    }

    /// A value as a constant-pool entry (scalars directly, everything
    /// else through the printed-and-read fallback, checked to round
    /// trip at build time).
    fn value_const(&mut self, v: &Value) -> Result<u64, Unsupported> {
        let c = match scalar_const(v)
            .or_else(|| meta_symbol_const(v))
            .or_else(|| coll_const(v))
        {
            Some(c) => c,
            None => Const::Read(readable_text(v)?),
        };
        Ok(self.constant(c))
    }

    fn var(&mut self, ns: &str, name: &str) -> u64 {
        let key = (ns.to_owned(), name.to_owned());
        if let Some(k) = self.var_index.get(&key) {
            return *k as u64;
        }
        let k = self.vars.len();
        self.vars.push(key.clone());
        self.var_index.insert(key, k);
        k as u64
    }

    fn var_of(&mut self, var: &clj_value::Var) -> u64 {
        let ns = var.sym.namespace().unwrap_or("user").to_owned();
        let name = var.sym.name().to_owned();
        self.var(&ns, &name)
    }

    // ---- function scaffolding ------------------------------------------

    fn begin_fn(&mut self, name: &str, env: Vec<Option<LocalRef<'ctx>>>) -> FnCtx<'ctx> {
        let func = self
            .module
            .add_function(name, self.code_ty, Some(Linkage::Private));
        self.begin_fn_into(func, env)
    }

    fn begin_fn_into(
        &mut self,
        func: FunctionValue<'ctx>,
        env: Vec<Option<LocalRef<'ctx>>>,
    ) -> FnCtx<'ctx> {
        let entry = self.ctx.append_basic_block(func, "entry");
        let start = self.ctx.append_basic_block(func, "start");
        let fail = self.ctx.append_basic_block(func, "fail");
        self.builder.position_at_end(start);
        let captured = func
            .get_nth_param(3)
            .expect("captured")
            .into_pointer_value();
        FnCtx {
            func,
            entry,
            fail,
            slots: Vec::new(),
            env,
            recur: Vec::new(),
            captured,
            moves: Vec::new(),
        }
    }

    /// Finish a function whose result is in `result`: entry inits every
    /// slot to nil; the exit moves the result out and drops every slot.
    fn end_fn(&mut self, f: FnCtx<'ctx>, result: PointerValue<'ctx>) -> FunctionValue<'ctx> {
        let ret = f.func.get_nth_param(0).expect("ret").into_pointer_value();
        self.clone_v(f.func, ret, result);
        self.drop_slots(f.func, &f.slots);
        self.builder
            .build_return(Some(&self.i32t.const_zero()))
            .expect("ret");
        self.builder.position_at_end(f.fail);
        self.drop_slots(f.func, &f.slots);
        self.builder
            .build_return(Some(&self.i32t.const_int(1, false)))
            .expect("ret");
        // Entry: zero-init the slots (nil), then into the body.
        let first = f.func.get_basic_blocks()[1];
        self.builder.position_at_end(f.entry);
        let zero = self.slot_ty.const_zero();
        for slot in &f.slots {
            self.builder.build_store(*slot, zero).expect("init");
        }
        self.builder.build_unconditional_branch(first).expect("br");
        f.func
    }

    fn drop_slots(&self, func: FunctionValue<'ctx>, slots: &[PointerValue<'ctx>]) {
        for slot in slots {
            self.drop_v(func, *slot);
        }
    }

    fn drop_and_nil_slots(&self, func: FunctionValue<'ctx>, slots: &[PointerValue<'ctx>]) {
        for slot in slots {
            self.drop_v(func, *slot);
            self.write_tag(*slot, 0);
        }
    }

    // ---- inline value primitives (`Value` is `repr(u8)`: tag byte at 0,
    // `Bool` payload at 1, `Int`/`Double` payload at 8; POD tags are
    // Nil 0, Bool 1, Int 2, Double 3, Char 9) ---------------------------

    fn tag(&self, slot: PointerValue<'ctx>) -> IntValue<'ctx> {
        self.builder
            .build_load(self.i8t, slot, "tag")
            .expect("load")
            .into_int_value()
    }

    /// A pointer to byte `off` (1: the bool payload, 8: the numeric
    /// payload) of a slot, through the slot's field view
    /// `{i8 tag, i8, i16, i32, i64, i64, i64}` (offsets 0, 1, 2, 4, 8,
    /// 16, 24 — the `Value` layout, see `value_layout_abi`).
    fn byte_at(&self, slot: PointerValue<'ctx>, off: u64) -> PointerValue<'ctx> {
        let i16t = self.ctx.i16_type();
        let view = self.ctx.struct_type(
            &[
                self.i8t.into(),
                self.i8t.into(),
                i16t.into(),
                self.i32t.into(),
                self.i64t.into(),
                self.i64t.into(),
                self.i64t.into(),
            ],
            false,
        );
        let field = match off {
            1 => 1,
            4 => 3,
            _ => 4,
        };
        self.builder
            .build_struct_gep(view, slot, field, "b")
            .expect("gep")
    }

    /// The numeric payload of an `Int` (converted) or `Double` slot as
    /// an `f64` (callers checked the tag is 2 or 3).
    fn num_as_f64(&self, slot: PointerValue<'ctx>) -> inkwell::values::FloatValue<'ctx> {
        let f64t = self.ctx.f64_type();
        let is_int = self
            .builder
            .build_int_compare(
                IntPredicate::EQ,
                self.tag(slot),
                self.i8t.const_int(2, false),
                "isi",
            )
            .expect("cmp");
        let as_int = self
            .builder
            .build_signed_int_to_float(self.int_payload(slot), f64t, "itof")
            .expect("itof");
        let as_dbl = self
            .builder
            .build_load(f64t, self.byte_at(slot, 8), "d")
            .expect("load")
            .into_float_value();
        self.builder
            .build_select(is_int, as_int, as_dbl, "num")
            .expect("sel")
            .into_float_value()
    }

    fn write_double(
        &self,
        func: FunctionValue<'ctx>,
        dest: PointerValue<'ctx>,
        v: inkwell::values::FloatValue<'ctx>,
    ) {
        self.drop_v(func, dest);
        self.write_tag(dest, 3);
        self.builder
            .build_store(self.byte_at(dest, 8), v)
            .expect("store");
    }

    fn is_numeric(&self, slot: PointerValue<'ctx>) -> IntValue<'ctx> {
        let t = self.tag(slot);
        let shifted = self
            .builder
            .build_int_sub(t, self.i8t.const_int(2, false), "t2")
            .expect("sub");
        self.builder
            .build_int_compare(
                IntPredicate::ULE,
                shifted,
                self.i8t.const_int(1, false),
                "num",
            )
            .expect("cmp")
    }

    /// Continue in a fresh block when every slot is `Int` (`2`) or `Double` (`3`),
    /// else branch to `slow`.
    fn when_numeric(&self, f: &FnCtx<'ctx>, elems: &[PointerValue<'ctx>], slow: BasicBlock<'ctx>) {
        let mut numeric = self.ctx.bool_type().const_int(1, false);
        for e in elems {
            let ok = self.is_numeric(*e);
            numeric = self.builder.build_and(numeric, ok, "nums").expect("and");
        }
        let go = self.ctx.append_basic_block(f.func, "num_fast");
        self.builder
            .build_conditional_branch(numeric, go, slow)
            .expect("br");
        self.builder.position_at_end(go);
    }

    /// `true` iff `a` (and `b` when present) hold `TAG_INT`.
    fn both_ints(&self, a: PointerValue<'ctx>, b: Option<PointerValue<'ctx>>) -> IntValue<'ctx> {
        let two = self.i8t.const_int(2, false);
        let a_int = self
            .builder
            .build_int_compare(IntPredicate::EQ, self.tag(a), two, "a_int")
            .expect("cmp");
        match b {
            Some(rhs) => {
                let b_int = self
                    .builder
                    .build_int_compare(IntPredicate::EQ, self.tag(rhs), two, "b_int")
                    .expect("cmp");
                self.builder.build_and(a_int, b_int, "ints").expect("and")
            }
            None => a_int,
        }
    }

    /// `FastArith` over numeric (int/double) args when not all ints:
    /// computes in `f64` (at least one arg is a double), else `slow`.
    fn lower_float_operands(
        &self,
        f: &FnCtx<'ctx>,
        lhs: PointerValue<'ctx>,
        rhs: Option<PointerValue<'ctx>>,
        slow: BasicBlock<'ctx>,
    ) -> (
        inkwell::values::FloatValue<'ctx>,
        inkwell::values::FloatValue<'ctx>,
    ) {
        if let Some(r) = rhs {
            self.when_numeric(f, &[lhs, r], slow);
        } else {
            self.when_numeric(f, &[lhs], slow);
        }
        let x = self.num_as_f64(lhs);
        let one = self.ctx.f64_type().const_float(1.0);
        let y = match rhs {
            Some(r) => self.num_as_f64(r),
            None => one,
        };
        (x, y)
    }

    fn lower_double_cmp(
        &self,
        f: &FnCtx<'ctx>,
        op: clj_eval::ext::FastArith,
        lhs: PointerValue<'ctx>,
        rhs: PointerValue<'ctx>,
        sink: &BoolSink<'ctx>,
        slow: BasicBlock<'ctx>,
    ) {
        use clj_eval::ext::FastArith;
        use inkwell::FloatPredicate;
        let (x, y) = self.lower_float_operands(f, lhs, Some(rhs), slow);
        let pred = match op {
            FastArith::Lt => FloatPredicate::OLT,
            FastArith::Lte => FloatPredicate::OLE,
            FastArith::Gt => FloatPredicate::OGT,
            _ => FloatPredicate::OGE,
        };
        let r = self
            .builder
            .build_float_compare(pred, x, y, "fc")
            .expect("fcmp");
        self.finish_bool(f.func, sink, r);
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_double_arith(
        &self,
        f: &FnCtx<'ctx>,
        op: clj_eval::ext::FastArith,
        lhs: PointerValue<'ctx>,
        rhs: Option<PointerValue<'ctx>>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        use clj_eval::ext::FastArith;
        let (x, y) = self.lower_float_operands(f, lhs, rhs, slow);
        let b = &self.builder;
        let r = match op {
            FastArith::Add | FastArith::Inc => b.build_float_add(x, y, "fa"),
            FastArith::Sub | FastArith::Dec => b.build_float_sub(x, y, "fs"),
            FastArith::Mul => b.build_float_mul(x, y, "fm"),
            _ => b.build_float_div(x, y, "fd"),
        }
        .expect("fop");
        self.write_double(f.func, dest, r);
        self.builder.build_unconditional_branch(done).expect("br");
    }

    /// A `Char` slot's code point (`u32` at offset 4), as an `i32`.
    fn char_payload(&self, slot: PointerValue<'ctx>) -> IntValue<'ctx> {
        let p = self.byte_at(slot, 4);
        self.builder
            .build_load(self.i32t, p, "c")
            .expect("load")
            .into_int_value()
    }

    fn int_payload(&self, slot: PointerValue<'ctx>) -> IntValue<'ctx> {
        let p = self.byte_at(slot, 8);
        self.builder
            .build_load(self.i64t, p, "i")
            .expect("load")
            .into_int_value()
    }

    fn is_pod(&self, tag: IntValue<'ctx>) -> IntValue<'ctx> {
        let small = self
            .builder
            .build_int_compare(
                IntPredicate::ULE,
                tag,
                self.i8t.const_int(3, false),
                "small",
            )
            .expect("cmp");
        let chr = self
            .builder
            .build_int_compare(IntPredicate::EQ, tag, self.i8t.const_int(9, false), "chr")
            .expect("cmp");
        self.builder.build_or(small, chr, "pod").expect("or")
    }

    /// `if !pod(slot) { rt_n_drop(slot) }` (leaves a POD in place).
    fn drop_v(&self, func: FunctionValue<'ctx>, slot: PointerValue<'ctx>) {
        let pod = self.is_pod(self.tag(slot));
        let heap = self.ctx.append_basic_block(func, "drop_heap");
        let done = self.ctx.append_basic_block(func, "drop_done");
        self.builder
            .build_conditional_branch(pod, done, heap)
            .expect("br");
        self.builder.position_at_end(heap);
        self.call("rt_n_drop", None, &[slot.into()]);
        self.builder.build_unconditional_branch(done).expect("br");
        self.builder.position_at_end(done);
    }

    /// `dest = clone(src)`: POD values copy inline, others try the
    /// bitcode helper (`rt_bc_clone`) when enabled, else call out.
    fn clone_v(
        &self,
        func: FunctionValue<'ctx>,
        dest: PointerValue<'ctx>,
        src: PointerValue<'ctx>,
    ) {
        let pod = self.is_pod(self.tag(src));
        let podb = self.ctx.append_basic_block(func, "clone_pod");
        let heap = self.ctx.append_basic_block(func, "clone_heap");
        let done = self.ctx.append_basic_block(func, "clone_done");
        self.builder
            .build_conditional_branch(pod, podb, heap)
            .expect("br");
        self.builder.position_at_end(podb);
        self.drop_v(func, dest);
        let v = self
            .builder
            .build_load(self.slot_ty, src, "v")
            .expect("load");
        self.builder.build_store(dest, v).expect("store");
        self.builder.build_unconditional_branch(done).expect("br");
        self.builder.position_at_end(heap);
        if self.use_bitcode {
            self.drop_v(func, dest);
            let ok = self
                .call("rt_bc_clone", Some(self.i8t), &[dest.into(), src.into()])
                .expect("ok");
            let slow = self.ctx.append_basic_block(func, "clone_slow");
            let hit = self
                .builder
                .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "ok")
                .expect("cmp");
            self.builder
                .build_conditional_branch(hit, done, slow)
                .expect("br");
            self.builder.position_at_end(slow);
        }
        self.call("rt_n_clone", None, &[dest.into(), src.into()]);
        self.builder.build_unconditional_branch(done).expect("br");
        self.builder.position_at_end(done);
    }

    fn emit_deref(&self, func: FunctionValue<'ctx>, dest: PointerValue<'ctx>, k: u64) {
        let Some(gv) = self.fast_vars else {
            self.call("rt_n_deref", None, &[dest.into(), self.u64c(k)]);
            return;
        };
        let fv = self
            .builder
            .build_load(self.ptr, gv, "fv")
            .expect("load")
            .into_pointer_value();
        let non_null = self.builder.build_is_not_null(fv, "has_fv").expect("nn");
        let fast = self.ctx.append_basic_block(func, "deref_fast");
        let slow = self.ctx.append_basic_block(func, "deref_slow");
        let done = self.ctx.append_basic_block(func, "deref_done");
        self.builder
            .build_conditional_branch(non_null, fast, slow)
            .expect("br");
        self.builder.position_at_end(fast);
        self.drop_v(func, dest);
        let ok = self
            .call(
                "rt_bc_deref",
                Some(self.i8t),
                &[dest.into(), fv.into(), self.u64c(k)],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "ok")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, done, slow)
            .expect("br");
        self.builder.position_at_end(slow);
        self.call("rt_n_deref", None, &[dest.into(), self.u64c(k)]);
        self.builder.build_unconditional_branch(done).expect("br");
        self.builder.position_at_end(done);
    }

    fn write_tag(&self, dest: PointerValue<'ctx>, tag: u64) {
        self.builder
            .build_store(dest, self.i8t.const_int(tag, false))
            .expect("store");
    }

    /// Move `src`'s value into `dest` (dropping `dest`'s), leaving `src`
    /// nil without dropping it: ownership transfers.
    fn move_v(&self, func: FunctionValue<'ctx>, dest: PointerValue<'ctx>, src: PointerValue<'ctx>) {
        self.drop_v(func, dest);
        let v = self
            .builder
            .build_load(self.slot_ty, src, "mv")
            .expect("load");
        self.builder.build_store(dest, v).expect("store");
        self.write_tag(src, 0);
    }

    fn write_nil(&self, func: FunctionValue<'ctx>, dest: PointerValue<'ctx>) {
        self.drop_v(func, dest);
        self.write_tag(dest, 0);
    }

    fn write_int(&self, func: FunctionValue<'ctx>, dest: PointerValue<'ctx>, v: IntValue<'ctx>) {
        self.drop_v(func, dest);
        self.write_tag(dest, 2);
        self.builder
            .build_store(self.byte_at(dest, 8), v)
            .expect("store");
    }

    fn write_bool(&self, func: FunctionValue<'ctx>, dest: PointerValue<'ctx>, b: IntValue<'ctx>) {
        self.drop_v(func, dest);
        self.write_tag(dest, 1);
        let byte = self
            .builder
            .build_int_z_extend(b, self.i8t, "b")
            .expect("zext");
        self.builder
            .build_store(self.byte_at(dest, 1), byte)
            .expect("store");
    }

    /// A fresh nil slot of the current function.
    fn slot(&self, f: &mut FnCtx<'ctx>) -> PointerValue<'ctx> {
        let here = self.builder.get_insert_block().expect("positioned");
        self.builder.position_at_end(f.entry);
        let p = self
            .builder
            .build_alloca(self.slot_ty, "v")
            .expect("alloca");
        p.as_instruction_value()
            .expect("alloca instr")
            .set_alignment(8)
            .expect("align");
        self.builder.position_at_end(here);
        f.slots.push(p);
        p
    }

    /// `n` contiguous fresh slots: returns the base pointer and each
    /// element pointer.
    fn slot_array(
        &self,
        f: &mut FnCtx<'ctx>,
        n: usize,
    ) -> (PointerValue<'ctx>, Vec<PointerValue<'ctx>>) {
        if n == 0 {
            return (self.ptr.const_null(), Vec::new());
        }
        let here = self.builder.get_insert_block().expect("positioned");
        self.builder.position_at_end(f.entry);
        let arr_ty = self.slot_ty.array_type(n as u32);
        let base = self.builder.build_alloca(arr_ty, "arr").expect("alloca");
        base.as_instruction_value()
            .expect("alloca instr")
            .set_alignment(8)
            .expect("align");
        let elems: Vec<PointerValue> = (0..n).map(|i| self.elem(base, i)).collect();
        self.builder.position_at_end(here);
        f.slots.extend(elems.iter().copied());
        (base, elems)
    }

    /// Pointer to slot `i` of a slot array (viewed as a struct of
    /// `i + 1` slots: same layout, and the GEP stays in bounds by
    /// construction).
    fn elem(&self, base: PointerValue<'ctx>, i: usize) -> PointerValue<'ctx> {
        let fields: Vec<BasicTypeEnum<'ctx>> = vec![self.slot_ty.into(); i + 1];
        let view = self.ctx.struct_type(&fields, false);
        self.builder
            .build_struct_gep(view, base, i as u32, "e")
            .expect("gep")
    }

    /// Branch to the fail block when `status != 0`.
    fn check(&self, f: &FnCtx<'ctx>, status: IntValue<'ctx>) {
        let bad = self
            .builder
            .build_int_compare(
                IntPredicate::NE,
                status,
                status.get_type().const_zero(),
                "bad",
            )
            .expect("cmp");
        let ok = self.ctx.append_basic_block(f.func, "ok");
        self.builder
            .build_conditional_branch(bad, f.fail, ok)
            .expect("br");
        self.builder.position_at_end(ok);
    }

    /// Continue in a fresh unreachable block after a terminator.
    fn dead_end(&self, f: &FnCtx<'ctx>) {
        let dead = self.ctx.append_basic_block(f.func, "dead");
        self.builder.position_at_end(dead);
    }

    // ---- units and fns ---------------------------------------------------

    fn lower_thunk(
        &mut self,
        name: &str,
        expr: &Expr,
        init_fast_vars: bool,
    ) -> Result<FunctionValue<'ctx>, Unsupported> {
        let mut f = self.begin_fn(name, Vec::new());
        if init_fast_vars && let Some(gv) = self.fast_vars {
            let fv = self.call_ptr("rt_n_fast_vars", &[]);
            self.builder.build_store(gv, fv).expect("store");
        }
        f.moves = collect_body_moves(std::slice::from_ref(expr), 0, &f.env, &[]);
        let result = self.slot(&mut f);
        self.lower(&mut f, expr, result)?;
        Ok(self.end_fn(f, result))
    }

    /// Compile `fun` as a closure created at frame length `base` in
    /// the current function `outer`: returns the fn table index and
    /// the captured local indices (in capture order).
    fn lower_fun(
        &mut self,
        fun: &FunExpr,
        base: usize,
        outer: &FnCtx<'ctx>,
    ) -> Result<(u64, Vec<usize>), Unsupported> {
        let mut free = BTreeSet::new();
        for arity in fun.arities.iter() {
            arity_free(arity, base, &mut free);
        }
        let captured: Vec<usize> = free.into_iter().collect();
        // Locals not captured stay `None` (never referenced here).
        let mut env = vec![None; base];
        for (pos, idx) in captured.iter().enumerate() {
            env[*idx] = Some(match outer.env.get(*idx) {
                Some(Some(LocalRef::Cell(_) | LocalRef::CapturedCell(_))) => {
                    LocalRef::CapturedCell(pos)
                }
                _ => LocalRef::Captured(pos),
            });
        }
        let reserved = self
            .reserved
            .get(&(std::ptr::from_ref(fun) as usize))
            .cloned();
        let k = match &reserved {
            Some((k, _)) => *k,
            None => {
                self.fns.push(FnInfo {
                    display: fun.display.clone(),
                    arities: Vec::new(),
                });
                self.fns.len() - 1
            }
        };
        let here = self.builder.get_insert_block().expect("positioned");
        let mut arities = Vec::new();
        for (i, arity) in fun.arities.iter().enumerate() {
            let func = match &reserved {
                Some((_, funcs)) => funcs[i],
                None => {
                    self.fn_seq += 1;
                    let name = format!("cljfn_{}_{i}_{}", self.fn_seq, sanitize(&fun.display));
                    self.module
                        .add_function(&name, self.code_ty, Some(Linkage::Private))
                }
            };
            let code = self.lower_arity(func, arity, env.clone())?;
            arities.push((arity.fixed.len(), arity.rest.is_some(), code));
        }
        self.fns[k].arities = arities;
        self.builder.position_at_end(here);
        Ok((k as u64, captured))
    }

    fn lower_arity(
        &mut self,
        func: FunctionValue<'ctx>,
        arity: &Arity,
        env: Vec<Option<LocalRef<'ctx>>>,
    ) -> Result<FunctionValue<'ctx>, Unsupported> {
        let mut f = self.begin_fn_into(func, env);
        let args = f.func.get_nth_param(1).expect("args").into_pointer_value();
        let pats: Vec<&Pattern> = arity.fixed.iter().chain(arity.rest.iter()).collect();
        // Params read the caller's argument array in place (the caller
        // owns it for the call; compiled code never drops or moves a
        // param). An arity that `recur`s rebinds its params, so it takes
        // owned copies first.
        let captures_end = f.env.len();
        let owned = self_recurs(&arity.body);
        let mut targets = Vec::with_capacity(pats.len());
        for (i, pat) in pats.iter().enumerate() {
            let arg = self.elem(args, i);
            let src = if owned {
                let s = self.slot(&mut f);
                self.clone_v(f.func, s, arg);
                s
            } else {
                arg
            };
            let slots = if matches!(pat, Pattern::Symbol(_)) {
                f.env.push(Some(LocalRef::Slot(src)));
                vec![src]
            } else {
                self.bind_pattern(&mut f, pat, src)?
            };
            targets.push(((*pat).clone(), slots));
        }
        let min_slot = if owned { captures_end } else { f.env.len() };
        let taken = if owned {
            Vec::new()
        } else {
            self.own_consumed_params(&mut f, &arity.body, captures_end, &pats, &mut targets)
        };
        let min_slot = if taken.is_empty() {
            min_slot
        } else {
            captures_end
        };
        f.moves = collect_body_moves(&arity.body, min_slot, &f.env, &taken);
        let header = self.ctx.append_basic_block(f.func, "body");
        self.builder.build_unconditional_branch(header).expect("br");
        self.builder.position_at_end(header);
        f.recur.push(RecurTarget {
            header,
            patterns: targets,
        });
        let result = self.slot(&mut f);
        self.lower_body(&mut f, &arity.body, result)?;
        Ok(self.end_fn(f, result))
    }

    /// Give each symbol param the body consumes (a last use it would move)
    /// an owned slot: taken from the caller's array when the call handed
    /// it over (`native_abi::TAKE_ARGS` in the count — e.g. `reduce`'s
    /// step, so its accumulator reaches `assoc` uniquely owned), cloned
    /// otherwise (the clone the use itself made before). Params the body
    /// never consumes keep reading the caller's array in place. Returns
    /// the owned params' env indices.
    fn own_consumed_params(
        &mut self,
        f: &mut FnCtx<'ctx>,
        body: &[Expr],
        captures_end: usize,
        pats: &[&Pattern],
        targets: &mut [(Pattern, Vec<PointerValue<'ctx>>)],
    ) -> Vec<usize> {
        let symbols: Vec<usize> = pats
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p, Pattern::Symbol(_)))
            .map(|(i, _)| captures_end + i)
            .collect();
        let probe = collect_body_moves(body, captures_end, &f.env, &symbols);
        let taken: Vec<usize> = symbols
            .into_iter()
            .filter(|i| probe.iter().any(|m| m.1 == *i))
            .collect();
        if taken.is_empty() {
            return taken;
        }
        let n = f.func.get_nth_param(2).expect("n").into_int_value();
        let bit = self
            .builder
            .build_and(
                n,
                self.i64t.const_int(clj_eval::native_abi::TAKE_ARGS, false),
                "take_bit",
            )
            .expect("and");
        let take = self
            .builder
            .build_int_compare(IntPredicate::NE, bit, self.i64t.const_zero(), "take")
            .expect("cmp");
        for &i in &taken {
            let arg = self.local_slot(f, i).expect("param slot");
            let s = self.slot(f);
            self.take_or_clone(f, s, arg, take);
            f.env[i] = Some(LocalRef::Slot(s));
            targets[i - captures_end].1 = vec![s];
        }
        taken
    }

    /// `dest = take ? move(src) : clone(src)`.
    fn take_or_clone(
        &self,
        f: &FnCtx<'ctx>,
        dest: PointerValue<'ctx>,
        src: PointerValue<'ctx>,
        take: IntValue<'ctx>,
    ) {
        let take_bb = self.ctx.append_basic_block(f.func, "param_take");
        let clone_bb = self.ctx.append_basic_block(f.func, "param_clone");
        let join = self.ctx.append_basic_block(f.func, "param_ready");
        self.builder
            .build_conditional_branch(take, take_bb, clone_bb)
            .expect("br");
        self.builder.position_at_end(take_bb);
        self.move_v(f.func, dest, src);
        self.builder.build_unconditional_branch(join).expect("br");
        self.builder.position_at_end(clone_bb);
        self.clone_v(f.func, dest, src);
        self.builder.build_unconditional_branch(join).expect("br");
        self.builder.position_at_end(join);
    }

    // ---- patterns ----------------------------------------------------------

    /// Bind `pat` to the value in `src`: allocates the pattern's local
    /// slots (in `pattern_names` order), pushes them on the frame, and
    /// returns them.
    fn bind_pattern(
        &mut self,
        f: &mut FnCtx<'ctx>,
        pat: &Pattern,
        src: PointerValue<'ctx>,
    ) -> Result<Vec<PointerValue<'ctx>>, Unsupported> {
        match pat {
            Pattern::Symbol(_) => {
                let s = self.slot(f);
                self.clone_v(f.func, s, src);
                f.env.push(Some(LocalRef::Slot(s)));
                Ok(vec![s])
            }
            Pattern::Vec(v) => {
                let mut out = Vec::new();
                if let Some(rest) = &v.rest {
                    let cur = self.slot(f);
                    let st = self
                        .call("rt_n_gseq", Some(self.i32t), &[cur.into(), src.into()])
                        .expect("st");
                    self.check(f, st);
                    for fixed in &v.fixed {
                        let item = self.slot(f);
                        let st = self
                            .call("rt_n_first", Some(self.i32t), &[item.into(), cur.into()])
                            .expect("st");
                        self.check(f, st);
                        out.extend(self.bind_pattern(f, fixed, item)?);
                        let st = self
                            .call("rt_n_next_mut", Some(self.i32t), &[cur.into()])
                            .expect("st");
                        self.check(f, st);
                    }
                    out.extend(self.bind_pattern(f, rest, cur)?);
                } else {
                    for (i, fixed) in v.fixed.iter().enumerate() {
                        let item = self.slot(f);
                        let st = self
                            .call(
                                "rt_n_nth_nil",
                                Some(self.i32t),
                                &[item.into(), src.into(), self.u64c(i as u64)],
                            )
                            .expect("st");
                        self.check(f, st);
                        out.extend(self.bind_pattern(f, fixed, item)?);
                    }
                }
                if v.as_sym.is_some() {
                    let s = self.slot(f);
                    self.clone_v(f.func, s, src);
                    f.env.push(Some(LocalRef::Slot(s)));
                    out.push(s);
                }
                Ok(out)
            }
            Pattern::Map(_) => {
                // The interpreter's own destructure over the encoded
                // pattern, `:or` defaults evaluated here first (eager,
                // in the enclosing frame, like the interpreter).
                let (enc, defaults) = native_abi::encode_pattern(pat);
                let k = self.value_const(&enc)?;
                let dexprs: Vec<Expr> = defaults.into_iter().cloned().collect();
                let darr = self.lower_args(f, &dexprs)?;
                let n = native_abi::pattern_local_count(pat);
                let (oarr, outs) = self.slot_array(f, n);
                let oref = self.slots_ref(f, oarr, n);
                let dref = self.slots_ref(f, darr, dexprs.len());
                let st = self
                    .call(
                        "rt_n_destructure",
                        Some(self.i32t),
                        &[oref.into(), src.into(), self.u64c(k), dref.into()],
                    )
                    .expect("st");
                self.check(f, st);
                for s in &outs {
                    f.env.push(Some(LocalRef::Slot(*s)));
                }
                Ok(outs)
            }
        }
    }

    /// Rebind `pat`'s existing slots from `src` (for `recur`).
    fn rebind_pattern(
        &mut self,
        f: &mut FnCtx<'ctx>,
        pat: &Pattern,
        src: PointerValue<'ctx>,
        slots: &[PointerValue<'ctx>],
    ) -> Result<(), Unsupported> {
        // A plain symbol takes the (dead after `recur`) temp by move.
        if let (Pattern::Symbol(_), [dst]) = (pat, slots) {
            self.move_v(f.func, *dst, src);
            return Ok(());
        }
        let depth = f.env.len();
        let fresh = self.bind_pattern(f, pat, src)?;
        f.env.truncate(depth);
        for (dst, s) in slots.iter().zip(fresh) {
            self.clone_v(f.func, *dst, s);
            self.drop_v(f.func, s);
        }
        Ok(())
    }

    // ---- expressions -------------------------------------------------------

    fn lower_body(
        &mut self,
        f: &mut FnCtx<'ctx>,
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let Some((last, init)) = body.split_last() else {
            self.write_nil(f.func, dest);
            return Ok(());
        };
        for e in init {
            if self.drops(e) {
                continue;
            }
            self.lower_discard(f, e)?;
        }
        self.lower(f, last, dest)
    }

    /// Whether a discarded statement can go unlowered: a value with no
    /// effect (`defn`'s trailing `(var name)`, a literal, a quote), or an
    /// inert metadata write ([`is_inert_meta_statement`]) in a program
    /// that never reads var metadata.
    fn drops(&self, e: &Expr) -> bool {
        matches!(e, Expr::VarRef(_) | Expr::Literal(_) | Expr::Quote(_))
            || (!self.var_meta && is_inert_meta_statement(e))
    }

    fn bc_fx_call<'a>(
        &self,
        expr: &'a Expr,
    ) -> Option<(&'a clj_eval::expr::DirectCall, &'static str)> {
        let Expr::DirectNative(call) = expr else {
            return None;
        };
        (self.inline_arith && self.use_bitcode && call.var.sym.namespace() == Some("clojure.core"))
            .then(|| {
                clj_bc::bc_fx_symbol(call.var.sym.name(), call.args.len()).map(|s| (&**call, s))
            })
            .flatten()
    }

    fn lower_discard(&mut self, f: &mut FnCtx<'ctx>, expr: &Expr) -> Result<(), Unsupported> {
        if let Expr::DirectNative(call) = expr {
            if self.try_lower_assoc_conj(f, call, None)? {
                return Ok(());
            }
            if self.try_lower_conj_bang_fx(f, call)? {
                return Ok(());
            }
        }
        let t = self.slot(f);
        if let Some((call, sym)) = self.bc_fx_call(expr) {
            self.lower_direct_with(f, call, Some(sym), t)?;
        } else {
            self.lower(f, expr, t)?;
        }
        self.drop_v(f.func, t);
        Ok(())
    }

    /// Lower each expression into its own slot of a fresh array.
    fn lower_args(
        &mut self,
        f: &mut FnCtx<'ctx>,
        args: &[Expr],
    ) -> Result<PointerValue<'ctx>, Unsupported> {
        let (base, elems) = self.slot_array(f, args.len());
        for (a, e) in args.iter().zip(elems) {
            self.lower(f, a, e)?;
        }
        Ok(base)
    }

    fn lower_local(
        &mut self,
        f: &FnCtx<'ctx>,
        expr: &Expr,
        idx: usize,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        // The analyzer resolves every local it emits (captures included).
        let local = f.env[idx].expect("analyzed local is visible");
        let p = self.local_slot(f, idx).expect("analyzed local is visible");
        if matches!(local, LocalRef::Cell(_) | LocalRef::CapturedCell(_)) {
            self.call("rt_n_cell_get", None, &[dest.into(), p.into()]);
        } else if is_move(f, expr) && matches!(local, LocalRef::Slot(_)) {
            self.move_v(f.func, dest, p);
        } else {
            self.clone_v(f.func, dest, p);
        }
        Ok(())
    }

    /// The raw slot holding local `idx` (a cell local yields the cell).
    fn local_slot(&self, f: &FnCtx<'ctx>, idx: usize) -> Option<PointerValue<'ctx>> {
        match f.env.get(idx).copied().flatten()? {
            LocalRef::Slot(p) | LocalRef::Cell(p) => Some(p),
            LocalRef::Captured(i) | LocalRef::CapturedCell(i) => Some(self.elem(f.captured, i)),
        }
    }

    fn lower(
        &mut self,
        f: &mut FnCtx<'ctx>,
        expr: &Expr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        match expr {
            Expr::Literal(v) | Expr::Quote(v) => self.lower_const(f.func, v, dest),
            Expr::Local(i) => self.lower_local(f, expr, *i, dest),
            Expr::Global(var) => {
                let k = self.var_of(var);
                self.emit_deref(f.func, dest, k);
                Ok(())
            }
            Expr::VarRef(var) => {
                let k = self.var_of(var);
                self.call("rt_n_varref", None, &[dest.into(), self.u64c(k)]);
                Ok(())
            }
            Expr::If { test, then, els } => self.lower_if(f, test, then, els, dest),
            Expr::Do(body) => self.lower_body(f, body, dest),
            Expr::Let { bindings, body } => self.lower_let(f, bindings, body, dest),
            Expr::Loop { bindings, body } => self.lower_loop(f, bindings, body, dest),
            Expr::Recur(args) => self.lower_recur(f, args),
            Expr::Invoke { head, args, .. } => self.lower_invoke(f, head, args, dest),
            Expr::DirectNative(call) => self.lower_direct(f, call, dest),
            other => self.lower_form(f, other, dest),
        }
    }

    /// Lower literals/collections, fns, defs, and control-flow forms
    /// off the hot scalar/call path.
    fn lower_form(
        &mut self,
        f: &mut FnCtx<'ctx>,
        expr: &Expr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        match expr {
            Expr::VecLit(items, meta) => {
                self.lower_coll(f, "rt_n_vec_lit", items, meta.as_ref(), dest, false)
            }
            Expr::SetLit(items, meta) => {
                self.lower_coll(f, "rt_n_set_lit", items, meta.as_ref(), dest, true)
            }
            Expr::MapLit(pairs, meta) => {
                let flat: Vec<Expr> = pairs
                    .iter()
                    .flat_map(|(k, v)| [k.clone(), v.clone()])
                    .collect();
                self.lower_coll(f, "rt_n_map_lit", &flat, meta.as_ref(), dest, true)
            }
            Expr::Fn(fun) => {
                let base = f.env.len();
                self.make_closure(f, fun, base, dest)
            }
            Expr::LetFn { fns, body } => self.lower_letfn(f, fns, body, dest),
            Expr::Def {
                sym,
                doc,
                init,
                meta,
                span,
            } => self.lower_def(
                f,
                sym,
                doc.as_deref(),
                init.as_deref(),
                meta.as_deref(),
                *span,
                dest,
            ),
            Expr::Throw(e) => self.lower_throw(f, e, dest),
            Expr::NewThrowable { class, message } => {
                let c = self.constant(Const::Str(class.jvm_name().to_owned()));
                let m = self.constant(Const::Str(message.clone()));
                self.call(
                    "rt_n_new_throwable",
                    None,
                    &[dest.into(), self.u64c(c), self.u64c(m)],
                );
                Ok(())
            }
            Expr::Try {
                body,
                catches,
                finally_body,
            } => self.lower_try(f, body, catches, finally_body, dest),
            Expr::Set { var, val } => self.lower_set(f, var, val, dest),
            Expr::CaseStar {
                expr,
                shift,
                mask,
                default,
                cases,
                skip,
                test,
            } => self.lower_case(f, expr, *shift, *mask, default, cases, skip, *test, dest),
            other => self.lower_scope_form(f, other, dest),
        }
    }

    fn lower_throw(
        &mut self,
        f: &mut FnCtx<'ctx>,
        e: &Expr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let _ = dest;
        let t = self.slot(f);
        self.lower(f, e, t)?;
        self.call("rt_n_throw", Some(self.i32t), &[t.into()]);
        self.builder.build_unconditional_branch(f.fail).expect("br");
        self.dead_end(f);
        Ok(())
    }

    fn lower_set(
        &mut self,
        f: &mut FnCtx<'ctx>,
        var: &clj_value::Var,
        val: &Expr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let k = self.var_of(var);
        let t = self.slot(f);
        self.lower(f, val, t)?;
        let st = self
            .call(
                "rt_n_set_var",
                Some(self.i32t),
                &[dest.into(), self.u64c(k), t.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    /// Lower dynamic-binding, namespace, and runtime-wrapper forms.
    fn lower_scope_form(
        &mut self,
        f: &mut FnCtx<'ctx>,
        expr: &Expr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        match expr {
            Expr::Binding { pairs, body } => self.lower_binding(f, pairs, body, dest),
            Expr::Declare { syms, span } => self.lower_declare(syms, *span, dest),
            Expr::Ns {
                name,
                doc,
                attr,
                clauses,
            } => self.lower_ns(f, name, doc.as_deref(), attr.as_ref(), clauses, dest),
            Expr::WithRedefs { pairs, body } => self.lower_with_redefs(f, pairs, body, dest),
            Expr::WithBindings { map, body } => self.lower_with_bindings(f, map, body, dest),
            Expr::Time { expr } => {
                let b = self.slot(f);
                self.body_closure(f, 0, std::slice::from_ref(expr.as_ref()), b)?;
                let st = self
                    .call("rt_n_time", Some(self.i32t), &[dest.into(), b.into()])
                    .expect("st");
                self.check(f, st);
                Ok(())
            }
            Expr::WithOutStr { out, body } => self.lower_with_out_str(f, out, body, dest),
            Expr::WithInStr { input, s, body } => self.lower_with_in_str(f, input, s, body, dest),
            Expr::Locking { lockee, body } => self.lower_locking(f, lockee, body, dest),
            Expr::Sync { body } | Expr::Io { body } => {
                let io = u64::from(matches!(expr, Expr::Io { .. }));
                let b = self.slot(f);
                self.body_closure(f, 0, body, b)?;
                let st = self
                    .call(
                        "rt_n_sync",
                        Some(self.i32t),
                        &[dest.into(), self.i8t.const_int(io, false).into(), b.into()],
                    )
                    .expect("st");
                self.check(f, st);
                Ok(())
            }
            other => self.lower_tail_form(f, other, dest),
        }
    }

    fn lower_declare(
        &mut self,
        syms: &[Symbol],
        span: clj_reader::Span,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let v = Value::Vector(Vector::from_vec(
            syms.iter().cloned().map(Value::Symbol).collect(),
        ));
        let k = self.value_const(&v)?;
        self.call(
            "rt_n_declare",
            None,
            &[
                dest.into(),
                self.u64c(k),
                self.i32t.const_int(u64::from(span.line), false).into(),
                self.i32t.const_int(u64::from(span.col), false).into(),
            ],
        );
        Ok(())
    }

    fn lower_ns(
        &mut self,
        f: &mut FnCtx<'ctx>,
        name: &str,
        doc: Option<&str>,
        attr: Option<&Map>,
        clauses: &[NsClause],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        if self.lite {
            let np = self.str_ptr(name);
            self.call("rt_n_ns_lite", None, &[dest.into(), np.into()]);
            return Ok(());
        }
        let v = ns_desc(name, doc, attr, clauses);
        let k = self.value_const(&v)?;
        let st = self
            .call("rt_n_ns", Some(self.i32t), &[dest.into(), self.u64c(k)])
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    fn lower_with_redefs(
        &mut self,
        f: &mut FnCtx<'ctx>,
        pairs: &[(clj_value::Var, Expr)],
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let exprs: Vec<Expr> = pairs.iter().map(|(_, e)| e.clone()).collect();
        let vals = self.lower_args(f, &exprs)?;
        let ks: Vec<u64> = pairs.iter().map(|(v, _)| self.var_of(v)).collect();
        let karr = self.u64_array(&ks);
        let b = self.slot(f);
        self.body_closure(f, 0, body, b)?;
        let vref = self.slots_ref(f, vals, pairs.len());
        let st = self
            .call(
                "rt_n_with_redefs",
                Some(self.i32t),
                &[dest.into(), karr.into(), vref.into(), b.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    fn lower_with_bindings(
        &mut self,
        f: &mut FnCtx<'ctx>,
        map: &Expr,
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let m = self.slot(f);
        self.lower(f, map, m)?;
        let b = self.slot(f);
        self.body_closure(f, 0, body, b)?;
        let st = self
            .call(
                "rt_n_with_bindings",
                Some(self.i32t),
                &[dest.into(), m.into(), b.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    fn lower_with_out_str(
        &mut self,
        f: &mut FnCtx<'ctx>,
        out: &Symbol,
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let k = self.value_const(&Value::Symbol(out.clone()))?;
        let b = self.slot(f);
        self.body_closure(f, 0, body, b)?;
        let st = self
            .call(
                "rt_n_with_out_str",
                Some(self.i32t),
                &[dest.into(), self.u64c(k), b.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    fn lower_with_in_str(
        &mut self,
        f: &mut FnCtx<'ctx>,
        input: &Symbol,
        s: &Expr,
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let k = self.value_const(&Value::Symbol(input.clone()))?;
        let sv = self.slot(f);
        self.lower(f, s, sv)?;
        let b = self.slot(f);
        self.body_closure(f, 0, body, b)?;
        let st = self
            .call(
                "rt_n_with_in_str",
                Some(self.i32t),
                &[dest.into(), self.u64c(k), sv.into(), b.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    fn lower_locking(
        &mut self,
        f: &mut FnCtx<'ctx>,
        lockee: &Expr,
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let t = self.slot(f);
        self.lower(f, lockee, t)?;
        let b = self.slot(f);
        self.body_closure(f, 0, body, b)?;
        let st = self
            .call(
                "rt_n_locking",
                Some(self.i32t),
                &[dest.into(), t.into(), b.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    /// Lower lazy-seq, delay, bound-fn, defmacro, and assert forms.
    fn lower_tail_form(
        &mut self,
        f: &mut FnCtx<'ctx>,
        expr: &Expr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        match expr {
            Expr::LazySeqStar { body } => self.wrap_body(f, 0, "clojure.core/fn", body, dest),
            Expr::DelayStar { display, body } => self.wrap_body(f, 1, display, body, dest),
            Expr::BoundFn { name, fun } => self.lower_bound_fn(f, name.as_deref(), fun, dest),
            Expr::DefMacro {
                sym,
                doc,
                attr,
                arglists,
                fun,
                span,
            } => self.lower_defmacro(
                f,
                sym,
                doc.as_deref(),
                attr.as_ref(),
                arglists,
                fun,
                *span,
                dest,
            ),
            Expr::Assert {
                test,
                message,
                form,
            } => self.lower_assert(f, test, message.as_deref(), form, dest),
            _ => unreachable!("lower_tail_form partition"),
        }
    }

    fn lower_bound_fn(
        &mut self,
        f: &mut FnCtx<'ctx>,
        name: Option<&str>,
        fun: &FunExpr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let depth = f.env.len();
        let cell = name.map(|_| {
            let c = self.slot(f);
            self.call("rt_n_cell_new", None, &[c.into()]);
            f.env.push(Some(LocalRef::Cell(c)));
            c
        });
        let t = self.slot(f);
        let base = f.env.len();
        self.make_closure(f, fun, base, t)?;
        self.call(
            "rt_n_wrap_fn",
            None,
            &[dest.into(), self.i8t.const_int(2, false).into(), t.into()],
        );
        if let Some(c) = cell {
            self.call("rt_n_cell_set", None, &[c.into(), dest.into()]);
        }
        f.env.truncate(depth);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_defmacro(
        &mut self,
        f: &mut FnCtx<'ctx>,
        sym: &Symbol,
        doc: Option<&str>,
        attr: Option<&Map>,
        arglists: &[Value],
        fun: &FunExpr,
        span: clj_reader::Span,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let desc = Value::Vector(Vector::from_vec(vec![
            Value::Symbol(sym.clone()),
            doc.map_or(Value::Nil, |d| Value::Str(std::sync::Arc::from(d))),
            attr.cloned().map_or(Value::Nil, Value::Map),
            Value::Vector(Vector::from_vec(arglists.to_vec())),
            Value::Int(i64::from(span.line)),
            Value::Int(i64::from(span.col)),
        ]));
        let k = self.value_const(&desc)?;
        let t = self.slot(f);
        let base = f.env.len();
        self.make_closure(f, fun, base, t)?;
        let st = self
            .call(
                "rt_n_defmacro",
                Some(self.i32t),
                &[dest.into(), self.u64c(k), t.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    fn lower_assert(
        &mut self,
        f: &mut FnCtx<'ctx>,
        test: &Expr,
        message: Option<&Expr>,
        form: &str,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let t = self.slot(f);
        self.lower(f, test, t)?;
        let ok = self.truthy(t);
        let fail_b = self.ctx.append_basic_block(f.func, "assert_fail");
        let pass_b = self.ctx.append_basic_block(f.func, "assert_pass");
        self.builder
            .build_conditional_branch(ok, pass_b, fail_b)
            .expect("br");
        self.builder.position_at_end(fail_b);
        let msg = match message {
            Some(m) => {
                let ms = self.slot(f);
                self.lower(f, m, ms)?;
                ms
            }
            None => self.ptr.const_null(),
        };
        let fk = self.constant(Const::Str(form.to_owned()));
        self.call(
            "rt_n_assert_fail",
            Some(self.i32t),
            &[msg.into(), self.u64c(fk)],
        );
        self.builder.build_unconditional_branch(f.fail).expect("br");
        self.builder.position_at_end(pass_b);
        self.write_nil(f.func, dest);
        Ok(())
    }

    /// A `let`: each init lowered into its slot; a symbol binding
    /// takes over the init's slot (no copy), while a destructuring
    /// pattern binds out of it. Bound values release at scope end.
    fn lower_let(
        &mut self,
        f: &mut FnCtx<'ctx>,
        bindings: &[(Pattern, Expr)],
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let depth = f.env.len();
        let mut bound = Vec::new();
        for (pat, init) in bindings {
            let t = self.slot(f);
            self.lower(f, init, t)?;
            if matches!(pat, Pattern::Symbol(_)) {
                // The init's slot becomes the local's (no copy).
                f.env.push(Some(LocalRef::Slot(t)));
                bound.push(t);
            } else {
                bound.extend(self.bind_pattern(f, pat, t)?);
                self.drop_v(f.func, t);
            }
        }
        self.lower_body(f, body, dest)?;
        f.env.truncate(depth);
        // Release let-bound values at scope end (RSS parity).
        self.drop_slots(f.func, &bound);
        Ok(())
    }

    /// A body compiled as a zero-arg fn named `display`, wrapped by
    /// the runtime (`kind`: 0 `lazy-seq`, 1 `delay`).
    fn wrap_body(
        &mut self,
        f: &mut FnCtx<'ctx>,
        kind: u64,
        display: &str,
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let fun = FunExpr {
            display: display.to_owned(),
            arities: std::sync::Arc::from([Arity {
                fixed: Vec::new(),
                rest: None,
                body: body.to_vec(),
            }]),
        };
        let t = self.slot(f);
        let base = f.env.len();
        self.make_closure(f, &fun, base, t)?;
        self.call(
            "rt_n_wrap_fn",
            None,
            &[
                dest.into(),
                self.i8t.const_int(kind, false).into(),
                t.into(),
            ],
        );
        Ok(())
    }

    /// The direct-call target of `(head args…)` with `n` args: var
    /// index, fn index, and the fixed arity's code.
    fn direct_target(
        &self,
        head: &Expr,
        n: usize,
    ) -> Option<(clj_value::Var, usize, FunctionValue<'ctx>)> {
        let Expr::Global(var) = head else { return None };
        let key = (
            var.sym.namespace().unwrap_or("user").to_owned(),
            var.sym.name().to_owned(),
        );
        let (k, arities) = self.direct.get(&key)?;
        let code = arities
            .iter()
            .find(|(fixed, variadic, _)| !variadic && *fixed == n)?
            .2;
        Some((var.clone(), *k, code))
    }

    /// Invoke lowering: handles pure `clojure.string` `require` forms in
    /// `BootMode::Lite`, guarded direct calls of compiled `defn`s, and general
    /// dynamic invocations.
    fn lower_invoke(
        &mut self,
        f: &mut FnCtx<'ctx>,
        head: &Expr,
        args: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        if self.direct_target(head, args.len()).is_some() {
            return self.lower_direct_invoke(f, head, args, dest);
        }
        if self.try_lower_static(f, head, args, dest)? {
            return Ok(());
        }
        let h = self.slot(f);
        self.lower(f, head, h)?;
        let (arr, elems) = self.slot_array(f, args.len());
        for (a, e) in args.iter().zip(&elems) {
            self.lower(f, a, *e)?;
        }
        let aref = self.slots_ref(f, arr, args.len());
        let st = self
            .call(
                "rt_n_invoke",
                Some(self.i32t),
                &[dest.into(), h.into(), aref.into()],
            )
            .expect("st");
        self.check(f, st);
        self.drop_v(f.func, h);
        self.write_tag(h, 0);
        self.drop_and_nil_slots(f.func, &elems);
        Ok(())
    }

    fn try_lower_static(
        &mut self,
        f: &mut FnCtx<'ctx>,
        head: &Expr,
        args: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<bool, Unsupported> {
        if !self.inline_arith {
            return Ok(false);
        }
        let Expr::Global(v) = head else {
            return Ok(false);
        };
        if v.sym.namespace() != Some("clojure.core") || v.sym.name() != "__static*" {
            return Ok(false);
        }
        let Some((op, rest)) = static_call_op(args) else {
            return Ok(false);
        };
        let (refs, temps, moved) = self.arg_refs(f, rest, dest)?;
        let slow = self.ctx.append_basic_block(f.func, "static_slow");
        let done = self.ctx.append_basic_block(f.func, "static_done");
        self.emit_static_fast(f, op, &refs, dest, slow, done);
        self.emit_static_invoke_slow(f, head, args, &refs, dest, slow, done)?;
        self.builder.position_at_end(done);
        self.drop_and_nil_slots(f.func, &temps);
        self.drop_and_nil_slots(f.func, &moved);
        Ok(true)
    }

    fn emit_static_fast(
        &self,
        f: &FnCtx<'ctx>,
        op: StaticOp,
        refs: &[PointerValue<'ctx>],
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        match op {
            StaticOp::Parse(shim) => self.lower_static_parse(shim, refs[0], dest, slow, done),
            StaticOp::RemUnsigned => self.lower_static_urem(f, refs[0], refs[1], dest, slow, done),
            StaticOp::MathAbs => self.lower_static_math_abs(f, refs[0], dest, slow, done),
            StaticOp::MathSqrt => self.lower_static_math_sqrt(f, refs[0], dest, slow, done),
            StaticOp::MathUnary(shim) => {
                self.when_numeric(f, &[refs[0]], slow);
                let x = self.num_as_f64(refs[0]);
                let r = self.call_f64(shim, &[x.into()]);
                self.write_double(f.func, dest, r);
                self.builder.build_unconditional_branch(done).expect("br");
            }
            StaticOp::MathBinary(shim) => {
                self.when_numeric(f, &[refs[0], refs[1]], slow);
                let (x, y) = (self.num_as_f64(refs[0]), self.num_as_f64(refs[1]));
                let r = self.call_f64(shim, &[x.into(), y.into()]);
                self.write_double(f.func, dest, r);
                self.builder.build_unconditional_branch(done).expect("br");
            }
        }
    }

    fn lower_static_parse(
        &self,
        shim: &str,
        arg: PointerValue<'ctx>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        let hit = self
            .call(shim, Some(self.i8t), &[dest.into(), arg.into()])
            .expect("hit");
        let ok = self
            .builder
            .build_int_compare(IntPredicate::NE, hit, self.i8t.const_zero(), "parse_ok")
            .expect("cmp");
        self.builder
            .build_conditional_branch(ok, done, slow)
            .expect("br");
    }

    fn lower_static_urem(
        &self,
        f: &FnCtx<'ctx>,
        lhs: PointerValue<'ctx>,
        rhs: PointerValue<'ctx>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        let ints = self.both_ints(lhs, Some(rhs));
        let int_bb = self.ctx.append_basic_block(f.func, "urem_int");
        self.builder
            .build_conditional_branch(ints, int_bb, slow)
            .expect("br");
        self.builder.position_at_end(int_bb);
        let (x, y) = (self.int_payload(lhs), self.int_payload(rhs));
        let nz = self
            .builder
            .build_int_compare(IntPredicate::NE, y, self.i64t.const_zero(), "nz")
            .expect("cmp");
        let go_bb = self.ctx.append_basic_block(f.func, "urem_go");
        self.builder
            .build_conditional_branch(nz, go_bb, slow)
            .expect("br");
        self.builder.position_at_end(go_bb);
        let r = self
            .builder
            .build_int_unsigned_rem(x, y, "urem")
            .expect("urem");
        self.write_int(f.func, dest, r);
        self.builder.build_unconditional_branch(done).expect("br");
    }

    fn lower_static_math_abs(
        &self,
        f: &FnCtx<'ctx>,
        arg: PointerValue<'ctx>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        let tag = self.tag(arg);
        let is_int = self
            .builder
            .build_int_compare(IntPredicate::EQ, tag, self.i8t.const_int(2, false), "abs_i")
            .expect("cmp");
        let int_bb = self.ctx.append_basic_block(f.func, "abs_int");
        let chk_dbl = self.ctx.append_basic_block(f.func, "abs_chk_dbl");
        self.builder
            .build_conditional_branch(is_int, int_bb, chk_dbl)
            .expect("br");

        self.builder.position_at_end(int_bb);
        let x = self.int_payload(arg);
        let intr = inkwell::intrinsics::Intrinsic::find("llvm.abs").expect("llvm.abs");
        let decl = intr
            .get_declaration(&self.module, &[self.i64t.into()])
            .expect("decl");
        let false_bit = self.ctx.bool_type().const_zero();
        let abs_i = self
            .builder
            .build_call(decl, &[x.into(), false_bit.into()], "iabs")
            .expect("call")
            .try_as_basic_value()
            .expect_basic("iabs")
            .into_int_value();
        self.write_int(f.func, dest, abs_i);
        self.builder.build_unconditional_branch(done).expect("br");

        self.builder.position_at_end(chk_dbl);
        let is_dbl = self
            .builder
            .build_int_compare(IntPredicate::EQ, tag, self.i8t.const_int(3, false), "abs_d")
            .expect("cmp");
        let dbl_bb = self.ctx.append_basic_block(f.func, "abs_dbl");
        self.builder
            .build_conditional_branch(is_dbl, dbl_bb, slow)
            .expect("br");

        self.builder.position_at_end(dbl_bb);
        let f64t = self.ctx.f64_type();
        let d = self
            .builder
            .build_load(f64t, self.byte_at(arg, 8), "d")
            .expect("load")
            .into_float_value();
        let fintr = inkwell::intrinsics::Intrinsic::find("llvm.fabs").expect("llvm.fabs");
        let fdecl = fintr
            .get_declaration(&self.module, &[f64t.into()])
            .expect("decl");
        let abs_d = self
            .builder
            .build_call(fdecl, &[d.into()], "fabs")
            .expect("call")
            .try_as_basic_value()
            .expect_basic("fabs")
            .into_float_value();
        self.write_double(f.func, dest, abs_d);
        self.builder.build_unconditional_branch(done).expect("br");
    }

    fn lower_static_math_sqrt(
        &self,
        f: &FnCtx<'ctx>,
        arg: PointerValue<'ctx>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        self.when_numeric(f, &[arg], slow);
        let x = self.num_as_f64(arg);
        let f64t = self.ctx.f64_type();
        let intr = inkwell::intrinsics::Intrinsic::find("llvm.sqrt").expect("llvm.sqrt");
        let decl = intr
            .get_declaration(&self.module, &[f64t.into()])
            .expect("decl");
        let r = self
            .builder
            .build_call(decl, &[x.into()], "sqrt")
            .expect("call")
            .try_as_basic_value()
            .expect_basic("sqrt")
            .into_float_value();
        self.write_double(f.func, dest, r);
        self.builder.build_unconditional_branch(done).expect("br");
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_static_invoke_slow(
        &mut self,
        f: &mut FnCtx<'ctx>,
        head: &Expr,
        args: &[Expr],
        refs: &[PointerValue<'ctx>],
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) -> Result<(), Unsupported> {
        self.builder.position_at_end(slow);
        let h = self.slot(f);
        self.lower(f, head, h)?;
        let (arr, elems) = self.slot_array(f, args.len());
        self.lower(f, &args[0], elems[0])?;
        self.lower(f, &args[1], elems[1])?;
        for (i, r) in refs.iter().enumerate() {
            self.copy_bits(elems[2 + i], *r);
        }
        let aref = self.slots_ref(f, arr, args.len());
        let st = self
            .call(
                "rt_n_invoke",
                Some(self.i32t),
                &[dest.into(), h.into(), aref.into()],
            )
            .expect("st");
        self.clear_aliases(&elems[2..]);
        self.check(f, st);
        self.drop_v(f.func, h);
        self.write_tag(h, 0);
        self.drop_and_nil_slots(f.func, &elems[..2]);
        self.builder.build_unconditional_branch(done).expect("br");
        Ok(())
    }

    /// A call of a compiled top-level `defn`: guarded direct call of
    /// its arity code (the var still holds that fn; depth/stack under
    /// the interpreter's limits), else the general invoke.
    fn lower_direct_invoke(
        &mut self,
        f: &mut FnCtx<'ctx>,
        head: &Expr,
        args: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let (var, k, code) = self.direct_target(head, args.len()).expect("checked");
        let vk = self.var_of(&var);
        // Plain locals ride in the argument array as bitwise aliases (the
        // callee only reads its args); they are cleared, never dropped,
        // right after the call. Computed args are owned there.
        let (arr, elems) = self.slot_array(f, args.len());
        let mut aliases = Vec::new();
        let mut temps = Vec::new();
        let mut moved = Vec::new();
        for (i, (a, e)) in args.iter().zip(&elems).enumerate() {
            let local = match a {
                Expr::Local(idx) => self.borrowable_before(f, *idx, &args[i + 1..]),
                _ => None,
            };
            if let Some(p) = local {
                self.copy_bits(*e, p);
                aliases.push(*e);
                if is_move(f, a) {
                    moved.push(p);
                }
            } else {
                self.lower(f, a, *e)?;
                temps.push(*e);
            }
        }
        let ok = self
            .call(
                "rt_n_enter_fn",
                Some(self.i8t),
                &[self.u64c(vk), self.u64c(k as u64)],
            )
            .expect("ok");
        let go = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "go")
            .expect("cmp");
        let direct_b = self.ctx.append_basic_block(f.func, "call_direct");
        let slow_b = self.ctx.append_basic_block(f.func, "call_slow");
        let done = self.ctx.append_basic_block(f.func, "call_done");
        self.builder
            .build_conditional_branch(go, direct_b, slow_b)
            .expect("br");
        self.builder.position_at_end(direct_b);
        let st = self
            .builder
            .build_call(
                code,
                &[
                    dest.into(),
                    arr.into(),
                    self.u64c(args.len() as u64),
                    self.ptr.const_null().into(),
                ],
                "st",
            )
            .expect("call")
            .try_as_basic_value()
            .expect_basic("st")
            .into_int_value();
        self.call("rt_n_leave", None, &[]);
        self.clear_aliases(&aliases);
        self.check(f, st);
        self.builder.build_unconditional_branch(done).expect("br");
        self.builder.position_at_end(slow_b);
        let h = self.slot(f);
        self.emit_deref(f.func, h, vk);
        let aref = self.slots_ref(f, arr, args.len());
        let st = self
            .call(
                "rt_n_invoke",
                Some(self.i32t),
                &[dest.into(), h.into(), aref.into()],
            )
            .expect("st");
        self.clear_aliases(&aliases);
        self.check(f, st);
        self.builder.build_unconditional_branch(done).expect("br");
        self.builder.position_at_end(done);
        self.drop_and_nil_slots(f.func, &temps);
        self.drop_and_nil_slots(f.func, &moved);
        Ok(())
    }

    /// `dest = src`'s bits without a refcount change (an alias the
    /// caller clears with [`clear_aliases`](Self::clear_aliases)).
    fn copy_bits(&self, dest: PointerValue<'ctx>, src: PointerValue<'ctx>) {
        let v = self
            .builder
            .build_load(self.slot_ty, src, "alias")
            .expect("load");
        self.builder.build_store(dest, v).expect("store");
    }

    /// Reset alias slots to nil without dropping (they never owned).
    fn clear_aliases(&self, aliases: &[PointerValue<'ctx>]) {
        for a in aliases {
            self.write_tag(*a, 0);
        }
    }

    /// A direct-native call: inline `FastArith` int ops (falling back
    /// to the native on non-int args or overflow), else a direct call
    /// of the var's native.
    fn lower_direct(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        if self.lite
            && let Some(aliases) = lite_direct_require_aliases(call)
        {
            for alias in aliases {
                let ap = self.str_ptr(alias.as_deref().unwrap_or(""));
                self.call("rt_n_require_string_lite", None, &[dest.into(), ap.into()]);
            }
            return Ok(());
        }
        if self.try_lower_assoc_conj(f, call, Some(dest))? {
            return Ok(());
        }
        self.lower_direct_with(f, call, None, dest)
    }

    fn lower_direct_with(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        bc_override: Option<&'static str>,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        if !self.inline_arith {
            return self.lower_direct_owned(f, call, dest);
        }
        let k = self.var_of(&call.var);
        let n = call.args.len();
        let fast = fast_op(call.fast, n);
        let ns = call.var.sym.namespace();
        let name = call.var.sym.name();
        let core_name = (ns == Some("clojure.core")).then(|| name.to_owned());
        let has_own_op = core_name
            .as_deref()
            .and_then(|cname| native_abi::own_op(cname, n))
            .is_some();
        let bc_sym = bc_override.or_else(|| {
            (fast.is_none() && self.use_bitcode)
                .then(|| clj_bc::bc_ns_op_symbol(ns, name, n))
                .flatten()
        });
        let moving = bc_sym.is_none()
            && fast.is_none()
            && (has_own_op
                || (is_owned_move_op(ns, name, n)
                    && call.args.iter().any(|a| is_move(f, a) || is_temporary(a))));
        if moving {
            return self.lower_direct_moving(f, call, k, core_name.as_deref(), dest);
        }
        let (refs, temps, moved) = self.arg_refs(f, &call.args, dest)?;
        let slow = self.ctx.append_basic_block(f.func, "native_slow");
        let done = self.ctx.append_basic_block(f.func, "native_done");
        self.emit_direct_inline(
            f,
            fast,
            core_name.as_deref(),
            bc_sym,
            &refs,
            dest,
            slow,
            done,
        );
        self.emit_direct_slow(f, &refs, k, n, core_name.as_deref(), dest, slow, done);
        self.drop_and_nil_slots(f.func, &temps);
        self.drop_and_nil_slots(f.func, &moved);
        Ok(())
    }

    /// The fast path of a direct-native call: an intrinsic, `FastArith`,
    /// or bitcode helper op that falls through to `done`, else a branch to `slow`.
    #[allow(clippy::too_many_arguments)]
    fn emit_direct_inline(
        &mut self,
        f: &mut FnCtx<'ctx>,
        fast: Option<clj_eval::ext::FastArith>,
        core_name: Option<&str>,
        bc_override: Option<&'static str>,
        refs: &[PointerValue<'ctx>],
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        let bc_sym = if bc_override.is_some() {
            bc_override
        } else if fast.is_none() && self.use_bitcode {
            core_name.and_then(|name| clj_bc::bc_op_symbol(name, refs.len()))
        } else {
            None
        };
        let bc_block = bc_sym.map(|_| self.ctx.append_basic_block(f.func, "bc_try"));
        let miss = bc_block.unwrap_or(slow);
        let inline = match (fast, core_name) {
            (None, Some(name)) => self.lower_intrinsic(f, name, refs, dest, miss, done),
            _ => false,
        };
        if !inline {
            match fast {
                Some(op) => self.lower_fast_arith(f, op, refs, dest, slow, done),
                None => {
                    self.builder.build_unconditional_branch(miss).expect("br");
                }
            }
        }
        if let (Some(sym), Some(bb)) = (bc_sym, bc_block) {
            self.emit_bc_op(f.func, bb, sym, refs, dest, slow, done);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_bc_op(
        &self,
        func: FunctionValue<'ctx>,
        bb: BasicBlock<'ctx>,
        shim: &str,
        refs: &[PointerValue<'ctx>],
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        self.builder.position_at_end(bb);
        self.drop_v(func, dest);
        let mut call_args: Vec<BasicMetadataValueEnum> = vec![dest.into()];
        call_args.extend(refs.iter().map(|r| BasicMetadataValueEnum::from(*r)));
        if matches!(
            shim,
            "rt_bc_1_count" | "rt_bc_2_nth" | "rt_bc_2_get" | "rt_bc_3_nth" | "rt_bc_3_get"
        ) {
            let probe = self.helper("rt_n_str_ascii", Some(self.i8t.into()), &[self.ptr.into()]);
            call_args.push(probe.as_global_value().as_pointer_value().into());
        }
        let ok = self.call(shim, Some(self.i8t), &call_args).expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "ok")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, done, slow)
            .expect("br");
    }

    /// The slow path of a direct-native call: the VM's inline op over the
    /// borrowed args, else the native (args cloned only there). 1-3 args
    /// go as direct operands; more through a slice.
    #[allow(clippy::too_many_arguments)]
    fn emit_direct_slow(
        &mut self,
        f: &mut FnCtx<'ctx>,
        refs: &[PointerValue<'ctx>],
        k: u64,
        n: usize,
        core_name: Option<&str>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        self.builder.position_at_end(slow);
        let op = core_name
            .and_then(|name| native_abi::inline_op(name, n))
            .unwrap_or(u32::MAX);
        let st = if op == u32::MAX {
            let s = self.refs_slice(f, refs);
            self.call(
                "rt_n_core_call",
                Some(self.i32t),
                &[dest.into(), self.u64c(k), s.into()],
            )
            .expect("st")
        } else if native_abi::is_re_op(op) && refs.len() == 2 {
            self.call(
                "rt_n_op2_re",
                Some(self.i32t),
                &[dest.into(), self.u64c(k), refs[0].into(), refs[1].into()],
            )
            .expect("st")
        } else {
            let mut call_args: Vec<BasicMetadataValueEnum> = vec![
                dest.into(),
                self.i32t.const_int(u64::from(op), false).into(),
                self.u64c(k),
            ];
            let shim = match refs.len() {
                1 => "rt_n_op1",
                2 => "rt_n_op2",
                3 => "rt_n_op3",
                _ => "rt_n_core_ref",
            };
            if (1..=3).contains(&refs.len()) {
                call_args.extend(refs.iter().map(|r| BasicMetadataValueEnum::from(*r)));
            } else {
                call_args.push(self.refs_slice(f, refs).into());
            }
            self.call(shim, Some(self.i32t), &call_args).expect("st")
        };
        self.check(f, st);
        self.builder.build_unconditional_branch(done).expect("br");
        self.builder.position_at_end(done);
    }

    /// A core call taking a local at its last use (inside `recur` args)
    /// by move: the args are owned (the moved local's slot is left nil),
    /// so a native with an owned-args entry gets a uniquely held
    /// collection and updates it in place, like the VM's last-use moves.
    fn lower_direct_moving(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        k: u64,
        core_name: Option<&str>,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let n = call.args.len();
        let (arr, elems) = self.slot_array(f, n);
        for (a, e) in call.args.iter().zip(&elems) {
            let moved = match a {
                Expr::Local(i) if is_move(f, a) => self.slot_of(f, *i),
                _ => None,
            };
            if let Some(src) = moved {
                self.move_v(f.func, *e, src);
            } else {
                self.lower(f, a, *e)?;
            }
        }
        let op = core_name
            .and_then(|name| native_abi::own_op(name, n).or_else(|| native_abi::inline_op(name, n)))
            .unwrap_or(u32::MAX);
        let aref = self.slots_ref(f, arr, n);
        let st = self
            .call(
                "rt_n_core_own",
                Some(self.i32t),
                &[
                    dest.into(),
                    self.i32t.const_int(u64::from(op), false).into(),
                    self.u64c(k),
                    aref.into(),
                ],
            )
            .expect("st");
        self.check(f, st);
        self.drop_slots(f.func, &elems);
        Ok(())
    }

    /// [`lower_direct`](Self::lower_direct) for a program that may rebind
    /// core vars: owned args through the var's current root.
    fn lower_direct_owned(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let k = self.var_of(&call.var);
        let n = call.args.len();
        let arr = self.lower_args(f, &call.args)?;
        let aref = self.slots_ref(f, arr, n);
        let st = self
            .call(
                "rt_n_native_mv",
                Some(self.i32t),
                &[dest.into(), self.u64c(k), aref.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    /// Each argument as a slot the call may borrow: a plain local's own
    /// slot (no clone; never `dest`, which the call writes), else a
    /// fresh slot holding the lowered argument. Returns the pointers,
    /// the fresh slots (`temps`), and any borrowed local slots at their
    /// last use (`moved`) to release right after the call completes.
    #[allow(clippy::type_complexity)]
    fn arg_refs(
        &mut self,
        f: &mut FnCtx<'ctx>,
        args: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<
        (
            Vec<PointerValue<'ctx>>,
            Vec<PointerValue<'ctx>>,
            Vec<PointerValue<'ctx>>,
        ),
        Unsupported,
    > {
        let mut refs = Vec::with_capacity(args.len());
        let mut temps = Vec::new();
        let mut moved = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let local = match a {
                Expr::Local(idx) => self.borrowable_before(f, *idx, &args[i + 1..]),
                _ => None,
            };
            if let Some(p) = local.filter(|p| *p != dest) {
                refs.push(p);
                if is_move(f, a) {
                    moved.push(p);
                }
            } else {
                let t = self.slot(f);
                self.lower(f, a, t)?;
                refs.push(t);
                temps.push(t);
            }
        }
        Ok((refs, temps, moved))
    }

    /// The slot holding local `idx`'s value, when it can be borrowed
    /// safely across subsequent call arguments `rest`.
    fn borrowable_before(
        &self,
        f: &FnCtx<'ctx>,
        idx: usize,
        rest: &[Expr],
    ) -> Option<PointerValue<'ctx>> {
        // A local some occurrence in the remaining arguments of the same
        // call moves is copied instead: moving later in arg evaluation
        // would overwrite or invalidate this borrowed slot.
        if rest.iter().any(|e| expr_moves_local(e, idx, &f.moves)) {
            return None;
        }
        self.slot_of(f, idx)
    }

    /// The slot holding local `idx`'s value (not a `letfn` cell).
    fn slot_of(&self, f: &FnCtx<'ctx>, idx: usize) -> Option<PointerValue<'ctx>> {
        match f.env.get(idx).copied().flatten()? {
            LocalRef::Slot(p) => Some(p),
            LocalRef::Captured(i) => Some(self.elem(f.captured, i)),
            LocalRef::Cell(_) | LocalRef::CapturedCell(_) => None,
        }
    }

    /// A `CSlice<&Value>` header over `refs` (the pointer array lives in
    /// the entry block; the pointers are stored here, at the call).
    fn refs_slice(&mut self, f: &FnCtx<'ctx>, refs: &[PointerValue<'ctx>]) -> PointerValue<'ctx> {
        let fields: Vec<BasicTypeEnum<'ctx>> = vec![self.ptr.into(); refs.len()];
        let ty = self.ctx.struct_type(&fields, false);
        let here = self.builder.get_insert_block().expect("positioned");
        self.builder.position_at_end(f.entry);
        let base = self.builder.build_alloca(ty, "refs").expect("alloca");
        self.builder.position_at_end(here);
        for (i, r) in refs.iter().enumerate() {
            let p = self
                .builder
                .build_struct_gep(ty, base, i as u32, "ref")
                .expect("gep");
            self.builder.build_store(p, *r).expect("store");
        }
        self.slots_ref(f, base, refs.len())
    }

    /// The analyzer's `FastArith` op inline: variadic folds over 2..=8 args
    /// step pairwise through a scratch POD slot so mixed int/double or late
    /// overflow falls back cleanly to `slow` with original `elems` intact.
    fn lower_fast_arith(
        &mut self,
        f: &mut FnCtx<'ctx>,
        op: clj_eval::ext::FastArith,
        elems: &[PointerValue<'ctx>],
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        use clj_eval::ext::FastArith;
        if matches!(
            op,
            FastArith::Lt | FastArith::Lte | FastArith::Gt | FastArith::Gte
        ) {
            self.lower_cmp_arith(f, op, elems, &BoolSink::Slot { dest, done }, slow);
            return;
        }
        if elems.len() <= 2 {
            self.lower_fast_arith_step(f, op, elems[0], elems.get(1).copied(), dest, slow, done);
            return;
        }
        let acc = self.slot(f);
        for i in 1..elems.len() {
            let lhs = if i == 1 { elems[0] } else { acc };
            let last = i + 1 == elems.len();
            let out = if last { dest } else { acc };
            let step_done = if last {
                done
            } else {
                self.ctx.append_basic_block(f.func, "arith_step")
            };
            self.lower_fast_arith_step(f, op, lhs, Some(elems[i]), out, slow, step_done);
            if !last {
                self.builder.position_at_end(step_done);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_fast_arith_step(
        &self,
        f: &FnCtx<'ctx>,
        op: clj_eval::ext::FastArith,
        lhs: PointerValue<'ctx>,
        rhs: Option<PointerValue<'ctx>>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        use clj_eval::ext::FastArith;
        let ints = self.both_ints(lhs, rhs);
        let fastb = self.ctx.append_basic_block(f.func, "native_fast");
        let dbl_check = self.ctx.append_basic_block(f.func, "native_dbl_check");
        self.builder
            .build_conditional_branch(ints, fastb, dbl_check)
            .expect("br");
        // Doubles (and int/double mixes): f64 math like the native.
        self.builder.position_at_end(dbl_check);
        self.lower_double_arith(f, op, lhs, rhs, dest, slow, done);
        self.builder.position_at_end(fastb);
        let a = self.int_payload(lhs);
        let b = match rhs {
            Some(r) => self.int_payload(r),
            None => self.i64t.const_int(1, false),
        };
        if op == FastArith::Div {
            self.lower_int_div_exact(f, a, b, dest, slow, done);
            return;
        }
        let name = match op {
            FastArith::Add | FastArith::Inc => "llvm.sadd.with.overflow",
            FastArith::Sub | FastArith::Dec => "llvm.ssub.with.overflow",
            _ => "llvm.smul.with.overflow",
        };
        let intr = inkwell::intrinsics::Intrinsic::find(name).expect("intrinsic");
        let decl = intr
            .get_declaration(&self.module, &[self.i64t.into()])
            .expect("decl");
        let pair = self
            .builder
            .build_call(decl, &[a.into(), b.into()], "ov")
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
        let okb = self.ctx.append_basic_block(f.func, "native_ok");
        self.builder
            .build_conditional_branch(o, slow, okb)
            .expect("br");
        self.builder.position_at_end(okb);
        self.write_int(f.func, dest, r);
        self.builder.build_unconditional_branch(done).expect("br");
    }

    fn lower_int_div_exact(
        &self,
        f: &FnCtx<'ctx>,
        x: IntValue<'ctx>,
        y: IntValue<'ctx>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        let bb = &self.builder;
        let y0 = bb
            .build_int_compare(IntPredicate::EQ, y, self.i64t.const_zero(), "y0")
            .expect("cmp");
        let min_v = self.i64t.const_int(i64::MIN as u64, true);
        let xmin = bb
            .build_int_compare(IntPredicate::EQ, x, min_v, "xmin")
            .expect("cmp");
        let ymin = bb
            .build_int_compare(IntPredicate::EQ, y, min_v, "ymin")
            .expect("cmp");
        let edge = bb
            .build_or(
                y0,
                bb.build_or(xmin, ymin, "anymin").expect("or"),
                "div_edge",
            )
            .expect("or");
        let safe_bb = self.ctx.append_basic_block(f.func, "div_safe");
        bb.build_conditional_branch(edge, slow, safe_bb)
            .expect("br");
        self.builder.position_at_end(safe_bb);
        let bb = &self.builder;
        let rem = bb.build_int_signed_rem(x, y, "rem").expect("rem");
        let exact = bb
            .build_int_compare(IntPredicate::EQ, rem, self.i64t.const_zero(), "exact")
            .expect("cmp");
        let exact_bb = self.ctx.append_basic_block(f.func, "div_exact");
        bb.build_conditional_branch(exact, exact_bb, slow)
            .expect("br");
        self.builder.position_at_end(exact_bb);
        let q = self.builder.build_int_signed_div(x, y, "q").expect("div");
        self.write_int(f.func, dest, q);
        self.builder.build_unconditional_branch(done).expect("br");
    }

    fn emit_int_cmp(
        &self,
        f: &FnCtx<'ctx>,
        op: clj_eval::ext::FastArith,
        a: IntValue<'ctx>,
        b: IntValue<'ctx>,
        sink: &BoolSink<'ctx>,
    ) {
        use clj_eval::ext::FastArith;
        let pred = match op {
            FastArith::Lt => IntPredicate::SLT,
            FastArith::Lte => IntPredicate::SLE,
            FastArith::Gt => IntPredicate::SGT,
            _ => IntPredicate::SGE,
        };
        let r = self
            .builder
            .build_int_compare(pred, a, b, "cmp")
            .expect("cmp");
        self.finish_bool(f.func, sink, r);
    }

    /// The slow block (the var's native, args moved in) and the join.
    #[allow(clippy::too_many_arguments)]
    /// `nil?`/`some?`/`not` of any value (no fallback needed).
    fn any_pred(&self, name: &str, v: PointerValue<'ctx>) -> IntValue<'ctx> {
        if name == "not" {
            let t = self.truthy(v);
            return self.builder.build_not(t, "not").expect("not");
        }
        let pred = if name == "nil?" {
            IntPredicate::EQ
        } else {
            IntPredicate::NE
        };
        self.builder
            .build_int_compare(pred, self.tag(v), self.i8t.const_zero(), "nil")
            .expect("cmp")
    }

    /// `zero?`/`pos?`/`neg?`/`even?`/`odd?` of an `Int` payload.
    fn int_pred(&self, name: &str, x: IntValue<'ctx>) -> IntValue<'ctx> {
        let zero = self.i64t.const_zero();
        let bb = &self.builder;
        match name {
            "zero?" => bb.build_int_compare(IntPredicate::EQ, x, zero, "p"),
            "pos?" => bb.build_int_compare(IntPredicate::SGT, x, zero, "p"),
            "neg?" => bb.build_int_compare(IntPredicate::SLT, x, zero, "p"),
            _ => {
                let low = bb
                    .build_and(x, self.i64t.const_int(1, false), "low")
                    .expect("and");
                let pred = if name == "even?" {
                    IntPredicate::EQ
                } else {
                    IntPredicate::NE
                };
                bb.build_int_compare(pred, low, zero, "p")
            }
        }
        .expect("cmp")
    }

    /// An int-only binary core op over checked `Int` payloads
    /// (division by zero and `MIN/-1` branch to `slow`).
    fn int_binop(
        &self,
        f: &FnCtx<'ctx>,
        name: &str,
        x: IntValue<'ctx>,
        y: IntValue<'ctx>,
        slow: BasicBlock<'ctx>,
    ) -> IntValue<'ctx> {
        match name {
            "quot" | "rem" | "mod" => self.int_div(f, name, x, y, slow),
            "max" | "min" => {
                let pred = if name == "max" {
                    IntPredicate::SGE
                } else {
                    IntPredicate::SLE
                };
                let pick = self
                    .builder
                    .build_int_compare(pred, x, y, "pick")
                    .expect("cmp");
                self.builder
                    .build_select(pick, x, y, "mm")
                    .expect("sel")
                    .into_int_value()
            }
            _ => {
                let bb = &self.builder;
                let sh = bb
                    .build_and(y, self.i64t.const_int(63, false), "sh")
                    .expect("and");
                match name {
                    "bit-and" => bb.build_and(x, y, "r"),
                    "bit-or" => bb.build_or(x, y, "r"),
                    "bit-xor" => bb.build_xor(x, y, "r"),
                    "bit-shift-left" => bb.build_left_shift(x, sh, "r"),
                    "bit-shift-right" => bb.build_right_shift(x, sh, true, "r"),
                    _ => bb.build_right_shift(x, sh, false, "r"),
                }
                .expect("bit")
            }
        }
    }

    /// `quot`/`rem`/`mod` on ints; the native handles zero and overflow.
    fn int_div(
        &self,
        f: &FnCtx<'ctx>,
        name: &str,
        x: IntValue<'ctx>,
        y: IntValue<'ctx>,
        slow: BasicBlock<'ctx>,
    ) -> IntValue<'ctx> {
        let bb = &self.builder;
        let zero = self.i64t.const_zero();
        let y0 = bb
            .build_int_compare(IntPredicate::EQ, y, zero, "y0")
            .expect("cmp");
        let ym1 = bb
            .build_int_compare(IntPredicate::EQ, y, self.i64t.const_all_ones(), "ym1")
            .expect("cmp");
        let xmin = bb
            .build_int_compare(
                IntPredicate::EQ,
                x,
                self.i64t.const_int(i64::MIN as u64, true),
                "xmin",
            )
            .expect("cmp");
        let ovf = bb.build_and(ym1, xmin, "ovf").expect("and");
        let bad = bb.build_or(y0, ovf, "bad").expect("or");
        let ok = self.ctx.append_basic_block(f.func, "div_ok");
        bb.build_conditional_branch(bad, slow, ok).expect("br");
        self.builder.position_at_end(ok);
        let bb = &self.builder;
        match name {
            "quot" => bb.build_int_signed_div(x, y, "q").expect("div"),
            "rem" => bb.build_int_signed_rem(x, y, "r").expect("rem"),
            _ => {
                // Clojure `mod`: the remainder takes the divisor's sign.
                let m = bb.build_int_signed_rem(x, y, "m").expect("rem");
                let xor = bb.build_xor(m, y, "x").expect("xor");
                let diff = bb
                    .build_int_compare(IntPredicate::SLT, xor, zero, "diff")
                    .expect("cmp");
                let nz = bb
                    .build_int_compare(IntPredicate::NE, m, zero, "nz")
                    .expect("cmp");
                let fix = bb.build_and(diff, nz, "fix").expect("and");
                let adj = bb.build_int_add(m, y, "adj").expect("add");
                bb.build_select(fix, adj, m, "mod")
                    .expect("sel")
                    .into_int_value()
            }
        }
    }

    /// Branch to `fast` when every slot holds an `Int`, else `slow`.
    fn when_ints(&self, f: &FnCtx<'ctx>, slots: &[PointerValue<'ctx>], slow: BasicBlock<'ctx>) {
        self.when_tag(f, slots, TAG_INT, slow);
    }

    /// Continue in a fresh block when every slot has tag `tag`, else
    /// branch to `slow`.
    fn when_tag(
        &self,
        f: &FnCtx<'ctx>,
        slots: &[PointerValue<'ctx>],
        tag: u64,
        slow: BasicBlock<'ctx>,
    ) {
        let want = self.i8t.const_int(tag, false);
        let mut all = self.ctx.bool_type().const_int(1, false);
        for s in slots {
            let is = self
                .builder
                .build_int_compare(IntPredicate::EQ, self.tag(*s), want, "is_tag")
                .expect("cmp");
            all = self.builder.build_and(all, is, "ints").expect("and");
        }
        let fast = self.ctx.append_basic_block(f.func, "intr_fast");
        self.builder
            .build_conditional_branch(all, fast, slow)
            .expect("br");
        self.builder.position_at_end(fast);
    }

    /// Inline a core fn by name when it has an intrinsic: returns
    /// `false` (nothing emitted) when it does not. Int-only ops guard
    /// on `Int` args and branch to `slow` otherwise; every emitted
    /// fast path ends by branching to `done`, and the caller positions
    /// at `slow` next.
    fn finish_bool(&self, func: FunctionValue<'ctx>, sink: &BoolSink<'ctx>, bit: IntValue<'ctx>) {
        let cur = self.builder.get_insert_block().expect("bb");
        match sink {
            BoolSink::Slot { dest, done } => {
                self.write_bool(func, *dest, bit);
                self.builder.build_unconditional_branch(*done).expect("br");
            }
            BoolSink::Jump { join, phi } => {
                phi.add_incoming(&[(&bit, cur)]);
                self.builder.build_unconditional_branch(*join).expect("br");
            }
        }
    }

    /// Inline predicate intrinsics (`nil?`, `some?`, `not`, `=`, `not=`, `==`,
    /// `zero?`, `pos?`, `neg?`, `even?`, `odd?`), emitting directly into `sink`.
    fn lower_pred_intrinsic(
        &self,
        f: &FnCtx<'ctx>,
        name: &str,
        a: &[PointerValue<'ctx>],
        sink: &BoolSink<'ctx>,
        slow: BasicBlock<'ctx>,
    ) -> bool {
        match (name, a.len()) {
            ("nil?" | "some?" | "not", 1) => {
                let r = self.any_pred(name, a[0]);
                self.finish_bool(f.func, sink, r);
                true
            }
            ("=" | "not=", 2) => {
                let other = self.ctx.append_basic_block(f.func, "eq_other");
                self.when_ints(f, a, other);
                let (x, y) = (self.int_payload(a[0]), self.int_payload(a[1]));
                let pred = if name == "not=" {
                    IntPredicate::NE
                } else {
                    IntPredicate::EQ
                };
                let r = self
                    .builder
                    .build_int_compare(pred, x, y, "eq")
                    .expect("cmp");
                self.finish_bool(f.func, sink, r);
                self.builder.position_at_end(other);
                self.when_tag(f, a, TAG_CHAR, slow);
                let (x, y) = (self.char_payload(a[0]), self.char_payload(a[1]));
                let r = self
                    .builder
                    .build_int_compare(pred, x, y, "ceq")
                    .expect("cmp");
                self.finish_bool(f.func, sink, r);
                true
            }
            ("==", 2) => {
                self.when_ints(f, a, slow);
                let (x, y) = (self.int_payload(a[0]), self.int_payload(a[1]));
                let r = self
                    .builder
                    .build_int_compare(IntPredicate::EQ, x, y, "eq")
                    .expect("cmp");
                self.finish_bool(f.func, sink, r);
                true
            }
            ("zero?" | "pos?" | "neg?" | "even?" | "odd?", 1) => {
                self.when_ints(f, a, slow);
                let r = self.int_pred(name, self.int_payload(a[0]));
                self.finish_bool(f.func, sink, r);
                true
            }
            _ => false,
        }
    }

    /// Lower a scalar-result comparison (`<`, `<=`, `>`, `>=`) into `sink` without
    /// boxing intermediate booleans when `sink` is a branch jump.
    fn lower_cmp_arith(
        &self,
        f: &FnCtx<'ctx>,
        op: clj_eval::ext::FastArith,
        elems: &[PointerValue<'ctx>],
        sink: &BoolSink<'ctx>,
        slow: BasicBlock<'ctx>,
    ) {
        if elems.len() == 2 {
            self.lower_cmp_pair(f, op, elems[0], elems[1], sink, slow);
            return;
        }
        let false_bb = self.ctx.append_basic_block(f.func, "cmp_false");
        for i in 0..elems.len() - 1 {
            if i + 2 == elems.len() {
                self.lower_cmp_pair(f, op, elems[i], elems[i + 1], sink, slow);
                break;
            }
            let here = self.builder.get_insert_block().expect("bb");
            let step_join = self.ctx.append_basic_block(f.func, "cmp_step");
            self.builder.position_at_end(step_join);
            let phi = self
                .builder
                .build_phi(self.ctx.bool_type(), "step_cmp")
                .expect("phi");
            self.builder.position_at_end(here);
            let next_bb = self.ctx.append_basic_block(f.func, "cmp_next");
            let step_sink = BoolSink::Jump {
                join: step_join,
                phi,
            };
            self.lower_cmp_pair(f, op, elems[i], elems[i + 1], &step_sink, slow);
            self.builder.position_at_end(step_join);
            self.builder
                .build_conditional_branch(phi.as_basic_value().into_int_value(), next_bb, false_bb)
                .expect("br");
            self.builder.position_at_end(next_bb);
        }
        let saved = self.builder.get_insert_block().expect("chain block");
        self.builder.position_at_end(false_bb);
        self.finish_bool(f.func, sink, self.ctx.bool_type().const_zero());
        self.builder.position_at_end(saved);
    }

    fn lower_cmp_pair(
        &self,
        f: &FnCtx<'ctx>,
        op: clj_eval::ext::FastArith,
        lhs: PointerValue<'ctx>,
        rhs: PointerValue<'ctx>,
        sink: &BoolSink<'ctx>,
        slow: BasicBlock<'ctx>,
    ) {
        let ints = self.both_ints(lhs, Some(rhs));
        let fastb = self.ctx.append_basic_block(f.func, "cmp_fast");
        let dbl_check = self.ctx.append_basic_block(f.func, "cmp_dbl_check");
        self.builder
            .build_conditional_branch(ints, fastb, dbl_check)
            .expect("br");
        // Doubles (and int/double mixes): f64 math like the native.
        self.builder.position_at_end(dbl_check);
        self.lower_double_cmp(f, op, lhs, rhs, sink, slow);
        self.builder.position_at_end(fastb);
        self.emit_int_cmp(f, op, self.int_payload(lhs), self.int_payload(rhs), sink);
    }

    fn lower_intrinsic(
        &mut self,
        f: &mut FnCtx<'ctx>,
        name: &str,
        a: &[PointerValue<'ctx>],
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) -> bool {
        let sink = BoolSink::Slot { dest, done };
        if self.lower_pred_intrinsic(f, name, a, &sink, slow) {
            return true;
        }
        let n = a.len();
        match (name, n) {
            ("int" | "long", 1) => {
                // A char's code point (always in range); anything else
                // takes `slow`.
                self.when_tag(f, a, TAG_CHAR, slow);
                let c = self.char_payload(a[0]);
                let n = self
                    .builder
                    .build_int_z_extend(c, self.i64t, "cp")
                    .expect("zext");
                self.write_int(f.func, dest, n);
            }
            ("max" | "min", 2..=8) => {
                self.lower_minmax(f, name == "max", a, dest, slow, done);
                return true;
            }
            ("quot" | "rem" | "mod", 2)
            | (
                "bit-and"
                | "bit-or"
                | "bit-xor"
                | "bit-shift-left"
                | "bit-shift-right"
                | "unsigned-bit-shift-right",
                2,
            ) => {
                self.when_ints(f, a, slow);
                let (x, y) = (self.int_payload(a[0]), self.int_payload(a[1]));
                let r = self.int_binop(f, name, x, y, slow);
                self.write_int(f.func, dest, r);
            }
            _ => return false,
        }
        self.builder.build_unconditional_branch(done).expect("br");
        true
    }

    fn lower_minmax(
        &mut self,
        f: &mut FnCtx<'ctx>,
        is_max: bool,
        elems: &[PointerValue<'ctx>],
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        if elems.len() == 2 {
            self.lower_minmax_step(f, is_max, elems[0], elems[1], dest, slow, done);
            return;
        }
        let acc = self.slot(f);
        for i in 1..elems.len() {
            let lhs = if i == 1 { elems[0] } else { acc };
            let last = i + 1 == elems.len();
            let out = if last { dest } else { acc };
            let step_done = if last {
                done
            } else {
                self.ctx.append_basic_block(f.func, "mm_step")
            };
            self.lower_minmax_step(f, is_max, lhs, elems[i], out, slow, step_done);
            if !last {
                self.builder.position_at_end(step_done);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_minmax_step(
        &self,
        f: &FnCtx<'ctx>,
        is_max: bool,
        lhs: PointerValue<'ctx>,
        rhs: PointerValue<'ctx>,
        dest: PointerValue<'ctx>,
        slow: BasicBlock<'ctx>,
        done: BasicBlock<'ctx>,
    ) {
        use inkwell::FloatPredicate;
        self.when_numeric(f, &[lhs, rhs], slow);
        let ints = self.both_ints(lhs, Some(rhs));
        let int_bb = self.ctx.append_basic_block(f.func, "mm_int");
        let dbl_bb = self.ctx.append_basic_block(f.func, "mm_dbl");
        self.builder
            .build_conditional_branch(ints, int_bb, dbl_bb)
            .expect("br");

        self.builder.position_at_end(int_bb);
        let (ix, iy) = (self.int_payload(lhs), self.int_payload(rhs));
        let pred = if is_max {
            IntPredicate::SGT
        } else {
            IntPredicate::SLT
        };
        let ipick = self
            .builder
            .build_int_compare(pred, ix, iy, "ipick")
            .expect("cmp");
        let iwin = self
            .builder
            .build_select(ipick, ix, iy, "iwin")
            .expect("sel")
            .into_int_value();
        self.write_int(f.func, dest, iwin);
        self.builder.build_unconditional_branch(done).expect("br");

        self.builder.position_at_end(dbl_bb);
        let (fx, fy) = (self.num_as_f64(lhs), self.num_as_f64(rhs));
        // Mirror core::minmax: strict winner keeps incumbent, NaN incumbent wins.
        let x_nan = self
            .builder
            .build_float_compare(FloatPredicate::UNO, fx, fx, "xnan")
            .expect("fcmp");
        let fpred = if is_max {
            FloatPredicate::OGT
        } else {
            FloatPredicate::OLT
        };
        let fcmp = self
            .builder
            .build_float_compare(fpred, fx, fy, "fcmp")
            .expect("fcmp");
        let fpick = self.builder.build_or(x_nan, fcmp, "fpick").expect("or");
        let win_ptr = self
            .builder
            .build_select(fpick, lhs, rhs, "winp")
            .expect("sel")
            .into_pointer_value();
        let win_bits = self
            .builder
            .build_load(self.slot_ty, win_ptr, "winv")
            .expect("load");
        self.drop_v(f.func, dest);
        self.builder.build_store(dest, win_bits).expect("store");
        self.builder.build_unconditional_branch(done).expect("br");
    }

    fn lower_const(
        &mut self,
        func: FunctionValue<'ctx>,
        v: &Value,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        match v {
            Value::Nil => self.write_nil(func, dest),
            Value::Int(n) => self.write_int(func, dest, self.i64t.const_int(*n as u64, true)),
            Value::Bool(b) => {
                let bit = self.ctx.bool_type().const_int(u64::from(*b), false);
                self.write_bool(func, dest, bit);
            }
            Value::Double(d) => {
                let fv = self.ctx.f64_type().const_float(*d);
                self.write_double(func, dest, fv);
            }
            Value::Char(c) => {
                self.drop_v(func, dest);
                self.write_tag(dest, TAG_CHAR);
                let cp = self.i32t.const_int(u64::from(u32::from(*c)), false);
                self.builder
                    .build_store(self.byte_at(dest, 4), cp)
                    .expect("store");
            }
            Value::Var(var) => {
                let k = self.var_of(var);
                self.call("rt_n_varref", None, &[dest.into(), self.u64c(k)]);
            }
            other => {
                let k = self.value_const(other)?;
                self.call("rt_n_const", None, &[dest.into(), self.u64c(k)]);
            }
        }
        Ok(())
    }

    fn meta_const(&mut self, meta: Option<&Map>) -> Result<u64, Unsupported> {
        match meta {
            Some(m) => self.value_const(&Value::Map(m.clone())),
            None => Ok(NONE),
        }
    }

    fn lower_coll(
        &mut self,
        f: &mut FnCtx<'ctx>,
        helper: &'static str,
        items: &[Expr],
        meta: Option<&Map>,
        dest: PointerValue<'ctx>,
        fallible: bool,
    ) -> Result<(), Unsupported> {
        if items.is_empty() && meta.is_none() {
            // One value per form (no allocation per evaluation). (A
            // literal with reader metadata stays per evaluation: its
            // metadata is a pool constant the init cannot read yet.)
            let site = self.consts.len();
            let k = self.constant(Const::EmptyColl { helper, site });
            self.call("rt_n_const", None, &[dest.into(), self.u64c(k)]);
            return Ok(());
        }
        let mk = self.meta_const(meta)?;
        let arr = self.lower_args(f, items)?;
        let aref = self.slots_ref(f, arr, items.len());
        let args = [dest.into(), aref.into(), self.u64c(mk)];
        if fallible {
            let st = self.call(helper, Some(self.i32t), &args).expect("st");
            self.check(f, st);
        } else {
            self.call(helper, None, &args);
        }
        Ok(())
    }

    fn truthy(&self, v: PointerValue<'ctx>) -> IntValue<'ctx> {
        let t = self.tag(v);
        let non_nil = self
            .builder
            .build_int_compare(IntPredicate::NE, t, self.i8t.const_zero(), "nn")
            .expect("cmp");
        let not_bool = self
            .builder
            .build_int_compare(IntPredicate::NE, t, self.i8t.const_int(1, false), "nb")
            .expect("cmp");
        let b = self
            .builder
            .build_load(self.i8t, self.byte_at(v, 1), "b")
            .expect("load")
            .into_int_value();
        let true_bool = self
            .builder
            .build_int_compare(IntPredicate::NE, b, self.i8t.const_zero(), "tb")
            .expect("cmp");
        let ok = self
            .builder
            .build_or(not_bool, true_bool, "ok")
            .expect("or");
        self.builder.build_and(non_nil, ok, "truthy").expect("and")
    }

    fn lower_cond_short_circuit(
        &mut self,
        f: &mut FnCtx<'ctx>,
        a: &Expr,
        b: &Expr,
        is_and: bool,
        then_b: BasicBlock<'ctx>,
        else_b: BasicBlock<'ctx>,
    ) -> Result<(), Unsupported> {
        let next_b = self.ctx.append_basic_block(f.func, "cond.next");
        if is_and {
            self.lower_cond(f, a, next_b, else_b)?;
        } else {
            self.lower_cond(f, a, then_b, next_b)?;
        }
        self.builder.position_at_end(next_b);
        let mark = f.env.len();
        f.env.push(None);
        let r = self.lower_cond(f, b, then_b, else_b);
        f.env.truncate(mark);
        r
    }

    fn lower_cond_if(
        &mut self,
        f: &mut FnCtx<'ctx>,
        test: &Expr,
        then: &Expr,
        els: &Expr,
        then_b: BasicBlock<'ctx>,
        else_b: BasicBlock<'ctx>,
    ) -> Result<(), Unsupported> {
        let tb = self.ctx.append_basic_block(f.func, "cond.then");
        let eb = self.ctx.append_basic_block(f.func, "cond.else");
        self.lower_cond(f, test, tb, eb)?;
        self.builder.position_at_end(tb);
        self.lower_cond(f, then, then_b, else_b)?;
        self.builder.position_at_end(eb);
        self.lower_cond(f, els, then_b, else_b)
    }

    /// Branch to `then_b` when `test` is truthy, else `else_b`, lowering
    /// short-circuit `and`/`or`, `not`, `if`, and inline predicates without
    /// boxing intermediate booleans.
    fn lower_cond(
        &mut self,
        f: &mut FnCtx<'ctx>,
        test: &Expr,
        then_b: BasicBlock<'ctx>,
        else_b: BasicBlock<'ctx>,
    ) -> Result<(), Unsupported> {
        if let Some((a, b, is_and)) = clj_eval::and_or_test(test, f.env.len()) {
            return self.lower_cond_short_circuit(f, a, b, is_and, then_b, else_b);
        }
        match test {
            Expr::Literal(v) => {
                let target = if matches!(v, Value::Nil | Value::Bool(false)) {
                    else_b
                } else {
                    then_b
                };
                self.builder.build_unconditional_branch(target).expect("br");
                return Ok(());
            }
            Expr::If {
                test: itest,
                then,
                els,
            } => {
                return self.lower_cond_if(f, itest, then, els, then_b, else_b);
            }
            Expr::DirectNative(call)
                if self.inline_arith && self.try_lower_cond_call(f, call, then_b, else_b)? =>
            {
                return Ok(());
            }
            _ => {}
        }
        let t = self.slot(f);
        self.lower(f, test, t)?;
        let cond = self.truthy(t);
        self.drop_v(f.func, t);
        self.write_tag(t, 0);
        self.builder
            .build_conditional_branch(cond, then_b, else_b)
            .expect("br");
        Ok(())
    }

    fn try_lower_cond_call(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        then_b: BasicBlock<'ctx>,
        else_b: BasicBlock<'ctx>,
    ) -> Result<bool, Unsupported> {
        use clj_eval::ext::FastArith;
        if call.var.sym.namespace() != Some("clojure.core") {
            return Ok(false);
        }
        let name = call.var.sym.name();
        let n = call.args.len();
        if name == "not" && n == 1 {
            self.lower_cond(f, &call.args[0], else_b, then_b)?;
            return Ok(true);
        }
        let fast = fast_op(call.fast, n);
        let is_cmp = matches!(
            fast,
            Some(FastArith::Lt | FastArith::Lte | FastArith::Gt | FastArith::Gte)
        );
        let is_pred = matches!(
            (name, n),
            (
                "nil?" | "some?" | "zero?" | "pos?" | "neg?" | "even?" | "odd?",
                1
            ) | ("=" | "not=" | "==", 2)
        );
        if !is_cmp && !is_pred {
            return Ok(false);
        }
        let slow_t = self.slot(f);
        let (refs, temps, moved) = self.arg_refs(f, &call.args, slow_t)?;
        let join = self.ctx.append_basic_block(f.func, "cond.join");
        let here = self.builder.get_insert_block().expect("bb");
        self.builder.position_at_end(join);
        let phi = self
            .builder
            .build_phi(self.ctx.bool_type(), "cond")
            .expect("phi");
        self.builder.position_at_end(here);
        let sink = BoolSink::Jump { join, phi };
        let slow = self.ctx.append_basic_block(f.func, "cond.slow");
        let slow_done = self.ctx.append_basic_block(f.func, "cond.slow_done");
        let bc_sym = (fast.is_none() && self.use_bitcode)
            .then(|| clj_bc::bc_op_symbol(name, n))
            .flatten();
        let bc_block = bc_sym.map(|_| self.ctx.append_basic_block(f.func, "cond.bc"));
        let miss = bc_block.unwrap_or(slow);
        if is_cmp {
            self.lower_cmp_arith(f, fast.expect("cmp"), &refs, &sink, miss);
        } else {
            self.lower_pred_intrinsic(f, name, &refs, &sink, miss);
        }
        if let (Some(sym), Some(bb)) = (bc_sym, bc_block) {
            self.emit_bc_op(f.func, bb, sym, &refs, slow_t, slow, slow_done);
        }
        let k = self.var_of(&call.var);
        self.emit_direct_slow(f, &refs, k, n, Some(name), slow_t, slow, slow_done);
        self.builder.position_at_end(slow_done);
        let sc = self.truthy(slow_t);
        self.drop_v(f.func, slow_t);
        self.write_tag(slow_t, 0);
        self.finish_bool(f.func, &sink, sc);
        self.builder.position_at_end(join);
        self.drop_and_nil_slots(f.func, &temps);
        self.drop_and_nil_slots(f.func, &moved);
        let bit = phi.as_basic_value().into_int_value();
        self.builder
            .build_conditional_branch(bit, then_b, else_b)
            .expect("br");
        Ok(true)
    }

    fn lower_if(
        &mut self,
        f: &mut FnCtx<'ctx>,
        test: &Expr,
        then: &Expr,
        els: &Expr,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let then_b = self.ctx.append_basic_block(f.func, "then");
        let else_b = self.ctx.append_basic_block(f.func, "else");
        let merge = self.ctx.append_basic_block(f.func, "merge");
        self.lower_cond(f, test, then_b, else_b)?;
        self.builder.position_at_end(then_b);
        self.lower(f, then, dest)?;
        self.builder.build_unconditional_branch(merge).expect("br");
        self.builder.position_at_end(else_b);
        self.lower(f, els, dest)?;
        self.builder.build_unconditional_branch(merge).expect("br");
        self.builder.position_at_end(merge);
        Ok(())
    }

    fn lower_loop(
        &mut self,
        f: &mut FnCtx<'ctx>,
        bindings: &[(Pattern, Expr)],
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        if self.try_lower_scalar_loop(f, bindings, body, dest)? {
            return Ok(());
        }
        let depth = f.env.len();
        let mut bound = Vec::new();
        let mut targets = Vec::new();
        for (pat, init) in bindings {
            let t = self.slot(f);
            self.lower(f, init, t)?;
            let slots = if matches!(pat, Pattern::Symbol(_)) {
                // The init's slot becomes the loop local's (no extra ref).
                f.env.push(Some(LocalRef::Slot(t)));
                vec![t]
            } else {
                let s = self.bind_pattern(f, pat, t)?;
                self.drop_v(f.func, t);
                s
            };
            bound.extend_from_slice(&slots);
            targets.push((pat.clone(), slots));
        }
        let header = self.ctx.append_basic_block(f.func, "loop");
        self.builder.build_unconditional_branch(header).expect("br");
        self.builder.position_at_end(header);
        f.recur.push(RecurTarget {
            header,
            patterns: targets,
        });
        let result = self.lower_body(f, body, dest);
        f.recur.pop();
        f.env.truncate(depth);
        if result.is_ok() {
            self.drop_slots(f.func, &bound);
        }
        result
    }

    fn lower_recur(&mut self, f: &mut FnCtx<'ctx>, args: &[Expr]) -> Result<(), Unsupported> {
        let fresh: Result<Vec<PointerValue>, Unsupported> = args
            .iter()
            .map(|a| {
                let t = self.slot(f);
                self.lower(f, a, t).map(|()| t)
            })
            .collect();
        let fresh = fresh?;
        // The analyzer only emits `recur` in tail position of a target.
        let target = f.recur.pop().expect("recur has a target");
        let depth = f.env.len();
        // Rebinding pushes env entries per pattern position: rebuild
        // the env so later patterns see earlier ones like a fresh bind.
        for ((pat, slots), src) in target.patterns.iter().zip(&fresh) {
            self.rebind_pattern(f, pat, *src, slots)?;
        }
        f.env.truncate(depth);
        for t in &fresh {
            self.drop_v(f.func, *t);
        }
        self.builder
            .build_unconditional_branch(target.header)
            .expect("br");
        f.recur.push(target);
        self.dead_end(f);
        Ok(())
    }

    /// Build the captured array for a fn created at `base` and make the
    /// closure value in `dest`.
    fn make_closure(
        &mut self,
        f: &mut FnCtx<'ctx>,
        fun: &FunExpr,
        base: usize,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let (k, captured) = self.lower_fun(fun, base, f)?;
        let (arr, elems) = self.slot_array(f, captured.len());
        for (idx, e) in captured.iter().zip(&elems) {
            let src = self.local_slot(f, *idx).expect("analyzed local is visible");
            self.clone_v(f.func, *e, src);
        }
        let cref = self.slots_ref(f, arr, captured.len());
        self.call(
            "rt_n_make_fn",
            None,
            &[dest.into(), self.u64c(k), cref.into()],
        );
        // The closure owns copies; release ours now.
        self.drop_slots(f.func, &elems);
        Ok(())
    }

    fn lower_letfn(
        &mut self,
        f: &mut FnCtx<'ctx>,
        fns: &[(String, FunExpr)],
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let depth = f.env.len();
        let mut cells = Vec::with_capacity(fns.len());
        for _ in fns {
            let c = self.slot(f);
            self.call("rt_n_cell_new", None, &[c.into()]);
            f.env.push(Some(LocalRef::Cell(c)));
            cells.push(c);
        }
        let base = f.env.len();
        for ((_, fun), cell) in fns.iter().zip(&cells) {
            let t = self.slot(f);
            self.make_closure(f, fun, base, t)?;
            self.call("rt_n_cell_set", None, &[(*cell).into(), t.into()]);
            self.drop_v(f.func, t);
        }
        let r = self.lower_body(f, body, dest);
        f.env.truncate(depth);
        r
    }

    /// A body run under a dynamic extent: compiled as a zero-argument
    /// closure (or a one-argument handler binding at the current
    /// frame length) whose value lands in `dest`.
    fn body_closure(
        &mut self,
        f: &mut FnCtx<'ctx>,
        params: usize,
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let fun = FunExpr {
            display: "clojure.core/fn".to_owned(),
            arities: std::sync::Arc::from([Arity {
                fixed: (0..params)
                    .map(|i| Pattern::Symbol(format!("p{i}")))
                    .collect(),
                rest: None,
                body: body.to_vec(),
            }]),
        };
        let base = f.env.len();
        self.make_closure(f, &fun, base, dest)
    }

    fn lower_try(
        &mut self,
        f: &mut FnCtx<'ctx>,
        body: &[Expr],
        catches: &[clj_eval::expr::Catch],
        fin: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let b = self.slot(f);
        self.body_closure(f, 0, body, b)?;
        let (harr, hs) = self.slot_array(f, catches.len());
        let mut classes = Vec::new();
        for (c, h) in catches.iter().zip(&hs) {
            assert_eq!(c.slot, f.env.len(), "a catch binds the next local slot");
            self.body_closure(f, 1, &c.body, *h)?;
            classes.push(Value::Str(std::sync::Arc::from(c.class.as_str())));
        }
        let ck = self.value_const(&Value::Vector(Vector::from_vec(classes)))?;
        let fin_ptr = if fin.is_empty() {
            self.ptr.const_null()
        } else {
            let s = self.slot(f);
            self.body_closure(f, 0, fin, s)?;
            s
        };
        let href = self.slots_ref(f, harr, catches.len());
        let st = self
            .call(
                "rt_n_try",
                Some(self.i32t),
                &[
                    dest.into(),
                    b.into(),
                    self.u64c(ck),
                    href.into(),
                    fin_ptr.into(),
                ],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    fn lower_binding(
        &mut self,
        f: &mut FnCtx<'ctx>,
        pairs: &[(clj_value::Var, Expr)],
        body: &[Expr],
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let exprs: Vec<Expr> = pairs.iter().map(|(_, e)| e.clone()).collect();
        let vals = self.lower_args(f, &exprs)?;
        let ks: Vec<u64> = pairs.iter().map(|(v, _)| self.var_of(v)).collect();
        let karr = self.u64_array(&ks);
        let b = self.slot(f);
        self.body_closure(f, 0, body, b)?;
        let vref = self.slots_ref(f, vals, pairs.len());
        let st = self
            .call(
                "rt_n_binding",
                Some(self.i32t),
                &[dest.into(), karr.into(), vref.into(), b.into()],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    /// A constant `CSlice<u64>` header over `values`.
    fn u64_array(&mut self, values: &[u64]) -> PointerValue<'ctx> {
        if values.is_empty() {
            return self.const_slice(self.ptr.const_null(), 0);
        }
        let consts: Vec<IntValue> = values
            .iter()
            .map(|v| self.i64t.const_int(*v, false))
            .collect();
        let arr = self.i64t.const_array(&consts);
        let g = self.const_global(&arr, "cljk");
        self.const_slice(g, values.len() as u64)
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_def(
        &mut self,
        f: &mut FnCtx<'ctx>,
        sym: &Symbol,
        doc: Option<&str>,
        init: Option<&Expr>,
        meta: Option<&[(Expr, Expr)]>,
        span: clj_reader::Span,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let inert_free;
        let (sym, meta) = if self.var_meta {
            (sym, meta)
        } else {
            inert_free = without_inert_def_meta(sym, meta);
            (&inert_free.0, inert_free.1.as_deref())
        };
        // A plain qualified name targets its declared var slot (no
        // symbol constant, no second intern); a name with metadata keeps
        // its symbol (`:dynamic`, `:private`, … live on it).
        let (begin, finish, key) = match (sym.namespace(), sym.meta()) {
            (Some(ns), None) => (
                "rt_n_def_begin_var",
                "rt_n_def_finish_var",
                self.var(ns, sym.name()),
            ),
            _ => (
                "rt_n_def_begin",
                "rt_n_def_finish",
                self.value_const(&Value::Symbol(sym.clone()))?,
            ),
        };
        let dk = match doc {
            Some(d) => self.constant(Const::Str(d.to_owned())),
            None => NONE,
        };
        let id = self
            .call(begin, Some(self.i64t), &[self.u64c(key)])
            .expect("id");
        let value = match init {
            Some(e) => {
                let t = self.slot(f);
                self.lower(f, e, t)?;
                t
            }
            None => self.ptr.const_null(),
        };
        let mref = match meta {
            Some(pairs) => {
                let flat: Vec<Expr> = pairs
                    .iter()
                    .flat_map(|(k, v)| [k.clone(), v.clone()])
                    .collect();
                let marr = self.lower_args(f, &flat)?;
                self.slots_ref(f, marr, flat.len())
            }
            None => self.ptr.const_null(),
        };
        let st = self
            .call(
                finish,
                Some(self.i32t),
                &[
                    dest.into(),
                    id.into(),
                    self.u64c(key),
                    self.u64c(dk),
                    value.into(),
                    mref.into(),
                    self.i32t.const_int(u64::from(span.line), false).into(),
                    self.i32t.const_int(u64::from(span.col), false).into(),
                ],
            )
            .expect("st");
        self.check(f, st);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_case(
        &mut self,
        f: &mut FnCtx<'ctx>,
        expr: &Expr,
        shift: i32,
        mask: i32,
        default: &Expr,
        cases: &[(i32, Expr, Expr)],
        skip: &[i32],
        test: CaseTest,
        dest: PointerValue<'ctx>,
    ) -> Result<(), Unsupported> {
        let v = self.slot(f);
        self.lower(f, expr, v)?;
        let desc = Value::Vector(Vector::from_vec(vec![
            Value::Int(i64::from(shift)),
            Value::Int(i64::from(mask)),
            Value::Int(match test {
                CaseTest::Int => 0,
                CaseTest::HashEquiv => 1,
                CaseTest::HashIdentity => 2,
            }),
            Value::Vector(Vector::from_vec(
                cases
                    .iter()
                    .map(|(k, _, _)| Value::Int(i64::from(*k)))
                    .collect(),
            )),
            Value::Vector(Vector::from_vec(
                skip.iter().map(|k| Value::Int(i64::from(*k))).collect(),
            )),
        ]));
        let ck = self.value_const(&desc)?;
        let sel = {
            let here = self.builder.get_insert_block().expect("positioned");
            self.builder.position_at_end(f.entry);
            let p = self.builder.build_alloca(self.i64t, "sel").expect("alloca");
            self.builder.position_at_end(here);
            p
        };
        let realized = self.slot(f);
        let st = self
            .call(
                "rt_n_case",
                Some(self.i32t),
                &[sel.into(), realized.into(), v.into(), self.u64c(ck)],
            )
            .expect("st");
        self.check(f, st);
        let selv = self
            .builder
            .build_load(self.i64t, sel, "sel")
            .expect("load")
            .into_int_value();
        let idx = self
            .builder
            .build_and(selv, self.i64t.const_int(0xffff_ffff, false), "idx")
            .expect("and");
        let checked = self
            .builder
            .build_right_shift(selv, self.i64t.const_int(32, false), false, "chk")
            .expect("shr");
        let default_b = self.ctx.append_basic_block(f.func, "case_default");
        let merge = self.ctx.append_basic_block(f.func, "case_end");
        let is_default = self
            .builder
            .build_int_compare(IntPredicate::EQ, selv, self.i64t.const_all_ones(), "dflt")
            .expect("cmp");
        let dispatch = self.ctx.append_basic_block(f.func, "case_dispatch");
        self.builder
            .build_conditional_branch(is_default, default_b, dispatch)
            .expect("br");
        self.builder.position_at_end(dispatch);
        let mut arms = Vec::new();
        let mut blocks = Vec::new();
        for i in 0..cases.len() {
            let b = self.ctx.append_basic_block(f.func, "case_arm");
            arms.push((self.i64t.const_int(i as u64, false), b));
            blocks.push(b);
        }
        self.builder
            .build_switch(idx, default_b, &arms)
            .expect("switch");
        for ((_, t, then), b) in cases.iter().zip(blocks) {
            self.builder.position_at_end(b);
            let do_check = self
                .builder
                .build_int_compare(IntPredicate::NE, checked, self.i64t.const_zero(), "c")
                .expect("cmp");
            let check_b = self.ctx.append_basic_block(f.func, "case_check");
            let then_b = self.ctx.append_basic_block(f.func, "case_then");
            self.builder
                .build_conditional_branch(do_check, check_b, then_b)
                .expect("br");
            self.builder.position_at_end(check_b);
            let tv = self.slot(f);
            self.lower(f, t, tv)?;
            let eq = {
                let here = self.builder.get_insert_block().expect("positioned");
                self.builder.position_at_end(f.entry);
                let p = self.builder.build_alloca(self.i8t, "eq").expect("alloca");
                self.builder.position_at_end(here);
                p
            };
            let st = self
                .call(
                    "rt_n_equiv",
                    Some(self.i32t),
                    &[eq.into(), realized.into(), tv.into()],
                )
                .expect("st");
            self.check(f, st);
            let eqv = self
                .builder
                .build_load(self.i8t, eq, "eqv")
                .expect("load")
                .into_int_value();
            let hit = self
                .builder
                .build_int_compare(IntPredicate::NE, eqv, self.i8t.const_zero(), "hit")
                .expect("cmp");
            self.builder
                .build_conditional_branch(hit, then_b, default_b)
                .expect("br");
            self.builder.position_at_end(then_b);
            self.lower(f, then, dest)?;
            self.builder.build_unconditional_branch(merge).expect("br");
        }
        self.builder.position_at_end(default_b);
        self.lower(f, default, dest)?;
        self.builder.build_unconditional_branch(merge).expect("br");
        self.builder.position_at_end(merge);
        Ok(())
    }

    // ---- program wiring ------------------------------------------------------

    fn lower_const_init(&mut self) -> FunctionValue<'ctx> {
        let mut f = self.begin_fn("cljconst_init", Vec::new());
        let t = self.slot(&mut f);
        let consts = self.consts.clone();
        for (k, c) in consts.iter().enumerate() {
            match c {
                Const::Str(s) => {
                    let p = self.str_ptr(s);
                    self.call("rt_n_str", None, &[t.into(), p.into()]);
                }
                Const::Keyword(ns, name) | Const::Symbol(ns, name) => {
                    let np = match ns {
                        Some(ns) => self.str_ptr(ns),
                        None => self.ptr.const_null(),
                    };
                    let p = self.str_ptr(name);
                    let h = if matches!(c, Const::Keyword(..)) {
                        "rt_n_kw"
                    } else {
                        "rt_n_sym"
                    };
                    self.call(h, None, &[t.into(), np.into(), p.into()]);
                }
                Const::Regex(text) => {
                    let p = self.str_ptr(text);
                    self.call("rt_n_regex", None, &[t.into(), p.into()]);
                }
                Const::StaticField(class, field) => {
                    let cp = self.str_ptr(class);
                    let fp = self.str_ptr(field);
                    self.call("rt_n_static_field", None, &[t.into(), cp.into(), fp.into()]);
                }
                Const::Class(name) => {
                    let p = self.str_ptr(name);
                    self.call("rt_n_class", None, &[t.into(), p.into()]);
                }
                Const::Read(text) => {
                    let p = self.str_ptr(text);
                    self.call("rt_n_read", None, &[t.into(), p.into()]);
                }
                Const::MetaSymbol { ns, name, meta } => {
                    self.init_meta_symbol(&mut f, t, ns.as_deref(), name, meta);
                }
                Const::Coll(kind, items) => self.init_coll(&mut f, t, *kind, items),
                Const::EmptyColl { helper, .. } => {
                    let empty = self.const_slice(self.ptr.const_null(), 0);
                    let args = [t.into(), empty.into(), self.u64c(NONE)];
                    if *helper == "rt_n_vec_lit" {
                        self.call(helper, None, &args);
                    } else {
                        let st = self.call(helper, Some(self.i32t), &args).expect("st");
                        self.check(&f, st);
                    }
                }
            }
            self.call("rt_n_set_const", None, &[self.u64c(k as u64), t.into()]);
        }
        let result = self.slot(&mut f);
        self.end_fn(f, result)
    }

    /// Build a [`Const::MetaSymbol`] into `t`: its metadata pairs into a
    /// slot array, then `rt_n_meta_sym`.
    fn init_meta_symbol(
        &mut self,
        f: &mut FnCtx<'ctx>,
        t: PointerValue<'ctx>,
        ns: Option<&str>,
        name: &str,
        meta: &[(MetaVal, MetaVal)],
    ) {
        let n = meta.len() * 2;
        let (arr, elems) = self.slot_array(f, n);
        for ((key, val), pair) in meta.iter().zip(elems.chunks(2)) {
            self.init_meta_val(f.func, key, pair[0]);
            self.init_meta_val(f.func, val, pair[1]);
        }
        let kvs = self.slots_ref(f, arr, n);
        let np = self.opt_str_ptr(ns);
        let p = self.str_ptr(name);
        self.call(
            "rt_n_meta_sym",
            None,
            &[t.into(), np.into(), p.into(), kvs.into()],
        );
        self.drop_slots(f.func, &elems);
    }

    /// Build a [`Const::Coll`] (or a nested collection) into `t`: its
    /// items into a slot array, then `rt_n_coll`.
    fn init_coll(
        &mut self,
        f: &mut FnCtx<'ctx>,
        t: PointerValue<'ctx>,
        kind: u64,
        items: &[ConstForm],
    ) {
        let (arr, elems) = self.slot_array(f, items.len());
        for (item, slot) in items.iter().zip(&elems) {
            match item {
                ConstForm::Simple(v) => self.init_meta_val(f.func, v, *slot),
                ConstForm::MetaSym(ns, name, meta) => {
                    self.init_meta_symbol(f, *slot, ns.as_deref(), name, meta);
                }
                ConstForm::Coll(k, nested) => self.init_coll(f, *slot, *k, nested),
            }
        }
        let r = self.slots_ref(f, arr, items.len());
        self.call("rt_n_coll", None, &[t.into(), self.u64c(kind), r.into()]);
        self.drop_slots(f.func, &elems);
    }

    /// Write one [`MetaVal`] into `slot`.
    fn init_meta_val(&mut self, func: FunctionValue<'ctx>, v: &MetaVal, slot: PointerValue<'ctx>) {
        match v {
            MetaVal::Inline(v) => self
                .lower_const(func, v, slot)
                .expect("inline scalars lower"),
            MetaVal::Keyword(ns, name) | MetaVal::Symbol(ns, name) => {
                let np = self.opt_str_ptr(ns.as_deref());
                let p = self.str_ptr(name);
                let h = if matches!(v, MetaVal::Keyword(..)) {
                    "rt_n_kw"
                } else {
                    "rt_n_sym"
                };
                self.call(h, None, &[slot.into(), np.into(), p.into()]);
            }
            MetaVal::Str(s) => {
                let p = self.str_ptr(s);
                self.call("rt_n_str", None, &[slot.into(), p.into()]);
            }
        }
    }

    /// A string constant pointer, or null for `None`.
    fn opt_str_ptr(&mut self, s: Option<&str>) -> PointerValue<'ctx> {
        match s {
            Some(s) => self.str_ptr(s),
            None => self.ptr.const_null(),
        }
    }

    /// No-op targets for the Linux static link's `--wrap` of crtbeginT's
    /// `.eh_frame` registration (#271): GCC 13+'s `__register_frame_info`
    /// classifies every FDE eagerly (hundreds of thousands of instructions
    /// and the whole `.eh_frame` faulted in at startup), while unwinding
    /// finds FDEs through `--eh-frame-hdr` + `dl_iterate_phdr` without it.
    /// Unreferenced (and dead-stripped) everywhere else.
    fn lower_frame_registry_wraps(&self) {
        let ptr = self.ptr;
        let register = self.module.add_function(
            "__wrap___register_frame_info",
            self.ctx
                .void_type()
                .fn_type(&[ptr.into(), ptr.into()], false),
            None,
        );
        self.builder
            .position_at_end(self.ctx.append_basic_block(register, "entry"));
        self.builder.build_return(None).expect("ret");
        let deregister = self.module.add_function(
            "__wrap___deregister_frame_info",
            ptr.fn_type(&[ptr.into()], false),
            None,
        );
        self.builder
            .position_at_end(self.ctx.append_basic_block(deregister, "entry"));
        self.builder
            .build_return(Some(&ptr.const_null()))
            .expect("ret");
    }

    fn lower_main(&mut self, path: &str, init: FunctionValue<'ctx>, tops: &[FunctionValue<'ctx>]) {
        let ty = self
            .i32t
            .fn_type(&[self.i32t.into(), self.ptr.into()], false);
        let main = self.module.add_function("main", ty, None);
        let entry = self.ctx.append_basic_block(main, "entry");
        self.builder.position_at_end(entry);
        // One `clj_rt::nabi::VarDecl` table: `ns/name` and the Lite
        // native hint of every var, in index order.
        let slice_ty = self.slice_ty();
        let decl_ty = self
            .ctx
            .struct_type(&[slice_ty.into(), slice_ty.into(), self.i32t.into()], false);
        let vars = self.vars.clone();
        let mut decls = Vec::with_capacity(vars.len());
        for (ns, name) in &vars {
            let ns_slice = self.str_slice_value(ns);
            let name_slice = self.str_slice_value(name);
            let hint = clj_eval::aot_boot::lite_native_hint(ns, name);
            decls.push(decl_ty.const_named_struct(&[
                ns_slice.into(),
                name_slice.into(),
                self.i32t.const_int(u64::from(hint), false).into(),
            ]));
        }
        let table = self.const_global(&decl_ty.const_array(&decls), "cljv");
        let decls_ref = self.const_slice(table, vars.len() as u64);
        self.call("rt_n_declare_vars", None, &[decls_ref.into()]);
        let fns: Vec<(String, Vec<ArityCode>)> = self
            .fns
            .iter()
            .map(|i| (i.display.clone(), i.arities.clone()))
            .collect();
        for (k, (display, arities)) in fns.iter().enumerate() {
            let dp = self.str_ptr(display);
            let arity_ty = self.ctx.struct_type(
                &[self.i64t.into(), self.i64t.into(), self.ptr.into()],
                false,
            );
            let descs: Vec<inkwell::values::StructValue> = arities
                .iter()
                .map(|a| {
                    arity_ty.const_named_struct(&[
                        self.i64t.const_int(a.0 as u64, false).into(),
                        self.i64t.const_int(u64::from(a.1), false).into(),
                        a.2.as_global_value().as_pointer_value().into(),
                    ])
                })
                .collect();
            let arr = arity_ty.const_array(&descs);
            let g = self.const_global(&arr, "clja");
            let aref = self.const_slice(g, arities.len() as u64);
            self.call(
                "rt_n_declare_fn",
                None,
                &[self.u64c(k as u64), dp.into(), aref.into()],
            );
        }
        let top_ptrs: Vec<PointerValue> = tops
            .iter()
            .map(|t| t.as_global_value().as_pointer_value())
            .collect();
        let tops_arr = self.ptr.const_array(&top_ptrs);
        let tops_g = self.const_global(&tops_arr, "cljtops");
        let tops_ref = self.const_slice(tops_g, tops.len() as u64);
        let pp = self.str_ptr(path);
        let argc = main.get_nth_param(0).expect("argc").into_int_value();
        let argv = main.get_nth_param(1).expect("argv");
        let ty = self.slice_ty();
        let argv_ref = self.builder.build_alloca(ty, "argv").expect("alloca");
        let p0 = self
            .builder
            .build_struct_gep(ty, argv_ref, 0, "ap")
            .expect("gep");
        self.builder.build_store(p0, argv).expect("store");
        let n = self
            .builder
            .build_int_z_extend(argc, self.i64t, "argc")
            .expect("zext");
        let p1 = self
            .builder
            .build_struct_gep(ty, argv_ref, 1, "an")
            .expect("gep");
        self.builder.build_store(p1, n).expect("store");
        let code = if self.lite {
            let var_refs: Vec<(&str, &str)> = self
                .vars
                .iter()
                .map(|(ns, n)| (ns.as_str(), n.as_str()))
                .collect();
            let caps = clj_eval::aot_boot::lite_caps_for(&var_refs);
            for (cap, sym) in [
                (clj_eval::aot_boot::LiteCaps::IO, "rt_n_cap_io"),
                (clj_eval::aot_boot::LiteCaps::CORE, "rt_n_cap_core"),
                (clj_eval::aot_boot::LiteCaps::SEQ, "rt_n_cap_seq"),
                (clj_eval::aot_boot::LiteCaps::RE, "rt_n_cap_re"),
                (clj_eval::aot_boot::LiteCaps::HOST, "rt_n_cap_host"),
                (clj_eval::aot_boot::LiteCaps::STRING, "rt_n_cap_string"),
            ] {
                if caps.contains(cap) {
                    self.call(sym, None, &[]);
                }
            }
            let need_stdin = var_refs
                .iter()
                .any(|&(ns, n)| clj_eval::aot_boot::is_lite_stdin_var(ns, n));
            let stdin_flag = self.i8t.const_int(u64::from(need_stdin), false);
            self.call(
                "rt_n_main_lite",
                Some(self.i32t),
                &[
                    pp.into(),
                    argv_ref.into(),
                    init.as_global_value().as_pointer_value().into(),
                    tops_ref.into(),
                    stdin_flag.into(),
                ],
            )
            .expect("code")
        } else {
            self.call(
                "rt_n_main",
                Some(self.i32t),
                &[
                    pp.into(),
                    argv_ref.into(),
                    init.as_global_value().as_pointer_value().into(),
                    tops_ref.into(),
                ],
            )
            .expect("code")
        };
        self.builder.build_return(Some(&code)).expect("ret");
    }
}

#[derive(Clone, Copy)]
enum StaticOp {
    Parse(&'static str),
    RemUnsigned,
    MathAbs,
    MathSqrt,
    MathUnary(&'static str),
    MathBinary(&'static str),
}

/// The inline fast path of a literal `(__static* "Cls" "member" args…)`
/// call plus its tail args (`None`: the generic `__static*` invoke).
fn static_call_op(args: &[Expr]) -> Option<(StaticOp, &[Expr])> {
    let ([fqn, mem], rest) = args.split_first_chunk::<2>()?;
    let op = static_op_spec(expr_str_lit(fqn)?, expr_str_lit(mem)?, rest.len())?;
    Some((op, rest))
}

fn expr_str_lit(e: &Expr) -> Option<&str> {
    match e {
        Expr::Quote(Value::Str(s)) | Expr::Literal(Value::Str(s)) => Some(s.as_ref()),
        _ => None,
    }
}

fn static_op_spec(fqn: &str, member: &str, n_args: usize) -> Option<StaticOp> {
    const TABLE: &[(&str, &str, usize, StaticOp)] = &[
        (
            "java.lang.Long",
            "parseLong",
            1,
            StaticOp::Parse("rt_n_parse_long"),
        ),
        (
            "java.lang.Integer",
            "parseInt",
            1,
            StaticOp::Parse("rt_n_parse_int"),
        ),
        (
            "java.lang.Double",
            "parseDouble",
            1,
            StaticOp::Parse("rt_n_parse_double"),
        ),
        (
            "java.lang.Long",
            "remainderUnsigned",
            2,
            StaticOp::RemUnsigned,
        ),
        ("java.lang.Math", "abs", 1, StaticOp::MathAbs),
        ("java.lang.Math", "sqrt", 1, StaticOp::MathSqrt),
        (
            "java.lang.Math",
            "exp",
            1,
            StaticOp::MathUnary("rt_n_math_exp"),
        ),
        (
            "java.lang.Math",
            "log",
            1,
            StaticOp::MathUnary("rt_n_math_log"),
        ),
        (
            "java.lang.Math",
            "log10",
            1,
            StaticOp::MathUnary("rt_n_math_log10"),
        ),
        (
            "java.lang.Math",
            "ceil",
            1,
            StaticOp::MathUnary("rt_n_math_ceil"),
        ),
        (
            "java.lang.Math",
            "floor",
            1,
            StaticOp::MathUnary("rt_n_math_floor"),
        ),
        (
            "java.lang.Math",
            "sin",
            1,
            StaticOp::MathUnary("rt_n_math_sin"),
        ),
        (
            "java.lang.Math",
            "cos",
            1,
            StaticOp::MathUnary("rt_n_math_cos"),
        ),
        (
            "java.lang.Math",
            "tan",
            1,
            StaticOp::MathUnary("rt_n_math_tan"),
        ),
        (
            "java.lang.Math",
            "asin",
            1,
            StaticOp::MathUnary("rt_n_math_asin"),
        ),
        (
            "java.lang.Math",
            "acos",
            1,
            StaticOp::MathUnary("rt_n_math_acos"),
        ),
        (
            "java.lang.Math",
            "atan",
            1,
            StaticOp::MathUnary("rt_n_math_atan"),
        ),
        (
            "java.lang.Math",
            "sinh",
            1,
            StaticOp::MathUnary("rt_n_math_sinh"),
        ),
        (
            "java.lang.Math",
            "cosh",
            1,
            StaticOp::MathUnary("rt_n_math_cosh"),
        ),
        (
            "java.lang.Math",
            "tanh",
            1,
            StaticOp::MathUnary("rt_n_math_tanh"),
        ),
        (
            "java.lang.Math",
            "cbrt",
            1,
            StaticOp::MathUnary("rt_n_math_cbrt"),
        ),
        (
            "java.lang.Math",
            "pow",
            2,
            StaticOp::MathBinary("rt_n_math_pow"),
        ),
        (
            "java.lang.Math",
            "atan2",
            2,
            StaticOp::MathBinary("rt_n_math_atan2"),
        ),
        (
            "java.lang.Math",
            "hypot",
            2,
            StaticOp::MathBinary("rt_n_math_hypot"),
        ),
    ];
    TABLE
        .iter()
        .find(|&&(c, m, n, _)| c == fqn && m == member && n == n_args)
        .map(|r| r.3)
}

/// The `FastArith` op a direct call inlines (`inc`/`dec` take one
/// arg, binary/variadic ops fold up to 8 args).
fn fast_op(fast: clj_eval::ext::FastArith, n: usize) -> Option<clj_eval::ext::FastArith> {
    use clj_eval::ext::FastArith;
    match (fast, n) {
        (FastArith::Inc | FastArith::Dec, 1) => Some(fast),
        (FastArith::None | FastArith::Inc | FastArith::Dec, _) => None,
        (other, 2..=8) => Some(other),
        _ => None,
    }
}

/// A value with a direct constant builder (`None`: the reader
/// fallback).
/// `v`'s printed form, checked to read back equal (the
/// `rt_n_read` constant path).
fn readable_text(v: &Value) -> Result<String, Unsupported> {
    let text = native_abi::print_constant(v);
    match native_abi::read_constant(&text) {
        Ok(back) if back == *v => Ok(text),
        _ => Err(Unsupported(format!(
            "constant without a readable form: {text}"
        ))),
    }
}

fn scalar_const(v: &Value) -> Option<Const> {
    Some(match v {
        Value::Str(s) => Const::Str(s.to_string()),
        Value::Keyword(k) => Const::Keyword(k.namespace().map(str::to_owned), k.name().to_owned()),
        Value::Symbol(s) if s.meta().is_none() => {
            Const::Symbol(s.namespace().map(str::to_owned), s.name().to_owned())
        }
        _ => return host_const(v),
    })
}

/// A symbol whose metadata maps keywords to simple values, built
/// without the reader (#271; `None`: anything else).
fn meta_symbol_const(v: &Value) -> Option<Const> {
    let Value::Symbol(s) = v else {
        return None;
    };
    let mut meta = Vec::new();
    for (k, val) in s.meta()?.iter() {
        let key = matches!(k, Value::Keyword(_)).then(|| meta_val(k))??;
        meta.push((key, meta_val(val)?));
    }
    Some(Const::MetaSymbol {
        ns: s.namespace().map(str::to_owned),
        name: s.name().to_owned(),
        meta,
    })
}

/// A metadata-free collection of simple forms as a [`Const::Coll`],
/// kept only when building it prints exactly as `v` does (#271).
fn coll_const(v: &Value) -> Option<Const> {
    let mut budget = COLL_FORMS_MAX;
    let (kind, items) = coll_parts(v, &mut budget)?;
    let built =
        clj_eval::native_abi::const_coll(kind, &items.iter().map(form_value).collect::<Vec<_>>());
    let same = native_abi::print_constant(&built) == native_abi::print_constant(v);
    same.then_some(Const::Coll(kind, items))
}

/// `v`'s [`ConstForm`] collection kind and items (within `budget` forms).
fn coll_parts(v: &Value, budget: &mut usize) -> Option<(u64, Vec<ConstForm>)> {
    let (kind, items) = coll_items(v)?;
    *budget = budget.checked_sub(items.len() + 1)?;
    let forms = items
        .iter()
        .map(|item| item_form(item, budget))
        .collect::<Option<_>>()?;
    Some((kind, forms))
}

/// A metadata-free unsorted collection's kind (`rt_n_coll`'s code) and
/// items (a map's keys and values interleaved).
fn coll_items(v: &Value) -> Option<(u64, Vec<Value>)> {
    Some(match v {
        Value::List(l) if l.meta().is_none() => (0, l.iter().cloned().collect()),
        Value::Vector(x) if x.meta().is_none() => (1, x.iter().collect()),
        Value::Map(m) if m.meta().is_none() && !m.is_sorted() => (
            2,
            m.iter()
                .flat_map(|(k, val)| [k.clone(), val.clone()])
                .collect(),
        ),
        Value::Set(s) if s.meta().is_none() && !s.is_sorted() => (3, s.iter().cloned().collect()),
        _ => return None,
    })
}

/// One collection item as a [`ConstForm`].
fn item_form(v: &Value, budget: &mut usize) -> Option<ConstForm> {
    if let Some(item) = meta_val(v) {
        return Some(ConstForm::Simple(item));
    }
    if let Some(Const::MetaSymbol { ns, name, meta }) = meta_symbol_const(v) {
        return Some(ConstForm::MetaSym(ns, name, meta));
    }
    coll_parts(v, budget).map(|(kind, items)| ConstForm::Coll(kind, items))
}

/// The value a [`ConstForm`] builds (the compiler-side twin of the
/// constant init, for [`coll_const`]'s check).
fn form_value(form: &ConstForm) -> Value {
    match form {
        ConstForm::Simple(v) => meta_val_value(v),
        ConstForm::MetaSym(ns, name, meta) => {
            let kvs: Vec<Value> = meta
                .iter()
                .flat_map(|(k, v)| [meta_val_value(k), meta_val_value(v)])
                .collect();
            native_abi::meta_symbol(ns.as_deref(), name, &kvs)
        }
        ConstForm::Coll(kind, items) => {
            native_abi::const_coll(*kind, &items.iter().map(form_value).collect::<Vec<_>>())
        }
    }
}

/// The value a [`MetaVal`] builds.
fn meta_val_value(v: &MetaVal) -> Value {
    match v {
        MetaVal::Inline(v) => v.clone(),
        MetaVal::Keyword(ns, name) => Value::Keyword(clj_value::Keyword::new(ns.as_deref(), name)),
        MetaVal::Symbol(ns, name) => Value::Symbol(Symbol::new(ns.as_deref(), name)),
        MetaVal::Str(s) => Value::Str(std::sync::Arc::from(s.as_str())),
    }
}

/// Host-derived constants rebuilt by name at startup (regexes,
/// classes, folded static fields).
fn host_const(v: &Value) -> Option<Const> {
    match v {
        Value::Regex(p) => Some(Const::Regex(p.to_string())),
        Value::Class(_) => Some(Const::Class(
            native_abi::class_name_of(v).unwrap_or_default(),
        )),
        other => native_abi::static_field_origin(other)
            .map(|(class, field)| Const::StaticField(class.to_owned(), field.to_owned())),
    }
}

/// Whether an arity body `recur`s to the arity itself (a `recur` outside
/// any nested `loop` or `fn`, which have their own targets).
fn self_recurs(body: &[Expr]) -> bool {
    fn walk(e: &Expr) -> bool {
        match e {
            Expr::Recur(_) => true,
            Expr::Loop { bindings, .. } => bindings.iter().any(|(_, init)| walk(init)),
            Expr::Fn(_) => false,
            _ => {
                let mut found = false;
                for_each_child(e, &mut |c| found = found || walk(c));
                found
            }
        }
    }
    body.iter().any(walk)
}

/// Whether a direct call has an owned-args entry that can reuse moved
/// (`Arc::strong_count == 1`) collections or release consumed inputs early.
fn is_owned_move_op(ns: Option<&str>, name: &str, argc: usize) -> bool {
    match (ns, name) {
        (Some("clojure.string"), "join") => argc == 2,
        (Some("clojure.core"), "assoc") => argc >= 3 && argc % 2 == 1,
        (Some("clojure.core"), "conj" | "into" | "merge" | "dissoc" | "disj") => argc >= 2,
        (Some("clojure.core"), "pop" | "next" | "rest" | "transient" | "persistent!") => argc == 1,
        (Some("clojure.core"), "assoc-in") => argc == 3,
        (Some("clojure.core"), "update" | "update-in") => argc >= 3,
        (
            Some("clojure.core"),
            "str" | "vec" | "set" | "butlast" | "doall" | "dorun" | "drop-last" | "take-last"
            | "frequencies" | "reverse" | "sort" | "sort-by" | "group-by" | "some" | "every?"
            | "not-every?" | "not-any?" | "mapv" | "filterv" | "reduce" | "transduce",
        ) => argc >= 1,
        _ => false,
    }
}

/// Whether `e` evaluates into a slot of its own (a call, a constructed
/// collection — not a local, constant, or var): an owned call takes that
/// slot over instead of cloning it (#271), so e.g. `reduce` sees a fresh
/// `(map …)` seq as its only holder.
fn is_temporary(e: &Expr) -> bool {
    !matches!(
        e,
        Expr::Local(_) | Expr::Literal(_) | Expr::Quote(_) | Expr::Global(_) | Expr::VarRef(_)
    )
}

/// Whether `e` is an occurrence the current `recur` args move.
fn is_move(f: &FnCtx<'_>, e: &Expr) -> bool {
    f.moves.iter().any(|m| m.0 == expr_addr(e))
}

/// A node's address (identifies one occurrence).
fn expr_addr(e: &Expr) -> usize {
    std::ptr::from_ref(e) as usize
}

/// Collects `reads(e)` into `later` without emitting moves.
fn reads(e: &Expr, later: &mut BTreeSet<usize>) {
    if let Expr::Local(i) = e {
        later.insert(*i);
    }
    for_each_child(e, &mut |c| reads(c, later));
}

fn is_pure_expr(e: &Expr) -> bool {
    match e {
        Expr::Literal(_) | Expr::Quote(_) | Expr::Local(_) => true,
        Expr::VecLit(items, None) => items.iter().all(is_pure_expr),
        _ => false,
    }
}

fn same_pure_key<'a>(a: &'a Expr, b: &'a Expr) -> Option<&'a Expr> {
    match (a, b) {
        (Expr::Local(i), Expr::Local(j)) if i == j => Some(a),
        (Expr::Literal(Value::Int(x)), Expr::Literal(Value::Int(y))) if x == y => Some(a),
        _ => None,
    }
}

type AssocConjParts<'a> = (usize, usize, &'a Expr, &'a Expr);

fn match_assoc_conj(call: &clj_eval::expr::DirectCall) -> Option<AssocConjParts<'_>> {
    if call.var.sym.namespace() != Some("clojure.core") || call.var.sym.name() != "assoc!" {
        return None;
    }
    let [Expr::Local(ti1), k1, Expr::DirectNative(conj_c)] = call.args.as_slice() else {
        return None;
    };
    if conj_c.var.sym.namespace() != Some("clojure.core") || conj_c.var.sym.name() != "conj" {
        return None;
    }
    let [Expr::DirectNative(nth_c), x] = conj_c.args.as_slice() else {
        return None;
    };
    if nth_c.var.sym.namespace() != Some("clojure.core") || nth_c.var.sym.name() != "nth" {
        return None;
    }
    let [Expr::Local(ti2), k2] = nth_c.args.as_slice() else {
        return None;
    };
    let k_expr = same_pure_key(k1, k2)?;
    is_pure_expr(x).then_some((*ti1, *ti2, k_expr, x))
}

impl<'ctx> Lowerer<'ctx> {
    fn resolve_assoc_conj_operand(
        &mut self,
        f: &mut FnCtx<'ctx>,
        e: &Expr,
        allow_move: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<(PointerValue<'ctx>, u8, Option<PointerValue<'ctx>>), Unsupported> {
        if let Expr::Local(idx) = e
            && let Some(p) = self.slot_of(f, *idx)
        {
            let can_take = allow_move.is_some_and(|pred| {
                is_move(f, e)
                    && matches!(f.env.get(*idx), Some(Some(LocalRef::Slot(_))))
                    && pred(*idx)
            });
            return Ok((p, u8::from(can_take), None));
        }
        let s = self.slot(f);
        let saved = std::mem::take(&mut f.moves);
        let res = self.lower(f, e, s);
        f.moves = saved;
        res?;
        Ok((s, 1, Some(s)))
    }

    fn assoc_conj_slots_ok(&self, f: &FnCtx<'ctx>, e: &Expr) -> bool {
        !matches!(e, Expr::Local(i) if self.slot_of(f, *i).is_none())
    }

    /// Fused `(assoc! t1 k (conj (nth t2 k) x))` in place: returns
    /// `Ok(true)` when matched and lowered, delegating misses to
    /// ordinary lowering.
    fn try_lower_assoc_conj(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        dest: Option<PointerValue<'ctx>>,
    ) -> Result<bool, Unsupported> {
        if !self.inline_arith {
            return Ok(false);
        }
        let Some((ti1, ti2, k_expr, x)) = match_assoc_conj(call) else {
            return Ok(false);
        };
        let (Some(t1p), Some(t2p)) = (self.slot_of(f, ti1), self.slot_of(f, ti2)) else {
            return Ok(false);
        };
        if !self.assoc_conj_slots_ok(f, k_expr) || !self.assoc_conj_slots_ok(f, x) {
            return Ok(false);
        }
        let (kp, _, temp_k) = self.resolve_assoc_conj_operand(f, k_expr, None)?;
        let same_k = |xi: usize| matches!(k_expr, Expr::Local(ki) if *ki == xi);
        let pred = |xi: usize| xi != ti1 && xi != ti2 && !same_k(xi);
        let (xp, take_x, temp_x) = self.resolve_assoc_conj_operand(f, x, Some(&pred))?;
        let ok = self
            .call(
                "rt_n_assoc_conj",
                Some(self.i8t),
                &[
                    t1p.into(),
                    t2p.into(),
                    kp.into(),
                    xp.into(),
                    self.i8t.const_int(u64::from(take_x), false).into(),
                ],
            )
            .expect("ok");
        if let Some(sk) = temp_k {
            self.drop_v(f.func, sk);
        }
        self.emit_assoc_conj_branches(f, call, ok, t1p, temp_x, dest)?;
        Ok(true)
    }

    fn emit_assoc_conj_branches(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        ok: IntValue<'ctx>,
        t1p: PointerValue<'ctx>,
        temp_x: Option<PointerValue<'ctx>>,
        dest: Option<PointerValue<'ctx>>,
    ) -> Result<(), Unsupported> {
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "ac_hit")
            .expect("cmp");
        let fast_b = self.ctx.append_basic_block(f.func, "ac_hit");
        let slow_b = self.ctx.append_basic_block(f.func, "ac_miss");
        let done_b = self.ctx.append_basic_block(f.func, "ac_done");
        self.builder
            .build_conditional_branch(hit, fast_b, slow_b)
            .expect("br");
        self.builder.position_at_end(fast_b);
        if let Some(d) = dest {
            self.clone_v(f.func, d, t1p);
        }
        self.builder.build_unconditional_branch(done_b).expect("br");
        self.builder.position_at_end(slow_b);
        if let Some(s) = temp_x {
            self.drop_v(f.func, s);
        }
        self.lower_assoc_conj_miss(f, call, dest)?;
        self.builder.build_unconditional_branch(done_b).expect("br");
        self.builder.position_at_end(done_b);
        Ok(())
    }

    fn lower_assoc_conj_miss(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
        dest: Option<PointerValue<'ctx>>,
    ) -> Result<(), Unsupported> {
        let out = match dest {
            Some(d) => d,
            None => self.slot(f),
        };
        self.lower_direct_with(f, call, None, out)?;
        if dest.is_none() {
            self.drop_v(f.func, out);
        }
        Ok(())
    }

    /// Effect-only `(conj! coll x)` via inlined `rt_bc_2_conj_bang_fx` (when
    /// bitcode enabled) + `rt_n_conj_bang_fx` (no transient clone/drop), falling
    /// back to ordinary lowering on `0`.
    fn try_lower_conj_bang_fx(
        &mut self,
        f: &mut FnCtx<'ctx>,
        call: &clj_eval::expr::DirectCall,
    ) -> Result<bool, Unsupported> {
        if !self.inline_arith
            || !self.use_bitcode
            || call.var.sym.namespace() != Some("clojure.core")
            || call.var.sym.name() != "conj!"
            || call.args.len() != 2
        {
            return Ok(false);
        }
        let tmp = self.slot(f);
        let (refs, temps, moved) = self.arg_refs(f, &call.args, tmp)?;
        let done = self.ctx.append_basic_block(f.func, "cb_done");
        let nfx = self.ctx.append_basic_block(f.func, "cb_nfx");
        let ok = self
            .call(
                "rt_bc_2_conj_bang_fx",
                Some(self.i8t),
                &[tmp.into(), refs[0].into(), refs[1].into()],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "cb_bc")
            .expect("cmp");
        self.builder
            .build_conditional_branch(hit, done, nfx)
            .expect("br");
        self.builder.position_at_end(nfx);
        let ok = self
            .call(
                "rt_n_conj_bang_fx",
                Some(self.i8t),
                &[refs[0].into(), refs[1].into()],
            )
            .expect("ok");
        let hit = self
            .builder
            .build_int_compare(IntPredicate::NE, ok, self.i8t.const_zero(), "cb_hit")
            .expect("cmp");
        let slow = self.ctx.append_basic_block(f.func, "cb_miss");
        self.builder
            .build_conditional_branch(hit, done, slow)
            .expect("br");
        self.builder.position_at_end(slow);
        let k = self.var_of(&call.var);
        let fast = fast_op(call.fast, 2);
        let inner = self.ctx.append_basic_block(f.func, "cb_slow");
        self.emit_direct_inline(f, fast, Some("conj!"), None, &refs, tmp, inner, done);
        self.emit_direct_slow(f, &refs, k, 2, Some("conj!"), tmp, inner, done);
        self.drop_v(f.func, tmp);
        self.drop_and_nil_slots(f.func, &temps);
        self.drop_and_nil_slots(f.func, &moved);
        Ok(true)
    }
}

fn expr_moves_local(e: &Expr, idx: usize, moves: &[(usize, usize)]) -> bool {
    if let Expr::Local(_) = e
        && moves.iter().any(|&(a, l)| l == idx && a == expr_addr(e))
    {
        return true;
    }
    let mut hit = false;
    for_each_child(e, &mut |c| {
        if !hit && expr_moves_local(c, idx, moves) {
            hit = true;
        }
    });
    hit
}

fn collect_body_moves(
    body: &[Expr],
    min_slot: usize,
    env: &[Option<LocalRef<'_>>],
    owned_params: &[usize],
) -> Vec<(usize, usize)> {
    let mut slots = vec![false; env.len().max(min_slot)];
    for &i in owned_params {
        slots[i] = true;
    }
    let mut later = BTreeSet::new();
    let mut out = Vec::new();
    for e in body.iter().rev() {
        walk_moves(e, &mut slots, min_slot, &mut later, &mut out);
    }
    out
}

fn walk_moves(
    e: &Expr,
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    match e {
        Expr::Local(i) => {
            if !later.contains(i) && *i >= min_slot && slots.get(*i).copied().unwrap_or(false) {
                out.push((expr_addr(e), *i));
            }
            later.insert(*i);
        }
        Expr::If { test, then, els } => walk_if_moves(test, then, els, slots, min_slot, later, out),
        Expr::CaseStar {
            expr,
            default,
            cases,
            ..
        } => walk_case_moves(expr, default, cases, slots, min_slot, later, out),
        Expr::Let { bindings, body } => walk_let_moves(bindings, body, slots, min_slot, later, out),
        Expr::Loop { bindings, body } => {
            walk_loop_moves(bindings, body, slots, min_slot, later, out);
        }
        Expr::LetFn { fns, body } => walk_letfn_moves(fns, body, slots, min_slot, later, out),
        Expr::Recur(args) => walk_recur_moves(args, slots, min_slot, later, out),
        _ if walk_extent_moves(e, slots, min_slot, later, out) => {}
        _ => walk_children_moves(e, slots, min_slot, later, out),
    }
}

fn walk_if_moves(
    test: &Expr,
    then: &Expr,
    els: &Expr,
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    let mut other = later.clone();
    walk_moves(els, slots, min_slot, &mut other, out);
    walk_moves(then, slots, min_slot, later, out);
    later.extend(other);
    walk_moves(test, slots, min_slot, later, out);
}

fn walk_recur_moves(
    args: &[Expr],
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    later.retain(|&i| i < min_slot);
    for a in args.iter().rev() {
        walk_moves(a, slots, min_slot, later, out);
    }
}

fn walk_letfn_moves(
    fns: &[(String, FunExpr)],
    body: &[Expr],
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    let depth = slots.len();
    slots.resize(depth + fns.len(), false);
    for b in body.iter().rev() {
        walk_moves(b, slots, min_slot, later, out);
    }
    slots.truncate(depth);
    for (_, fun) in fns {
        for a in fun.arities.iter() {
            for b in &a.body {
                reads(b, later);
            }
        }
    }
    later.retain(|&i| i < depth);
}

fn walk_children_moves(
    e: &Expr,
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    let mut kids = Vec::new();
    for_each_child(e, &mut |c| kids.push(c));
    for c in kids.into_iter().rev() {
        walk_moves(c, slots, min_slot, later, out);
    }
}

fn walk_case_moves(
    expr: &Expr,
    default: &Expr,
    cases: &[(i32, Expr, Expr)],
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    let base_later = later.clone();
    let mut merged = later.clone();
    walk_moves(default, slots, min_slot, &mut merged, out);
    for (_, test, then) in cases {
        let mut br = base_later.clone();
        walk_moves(then, slots, min_slot, &mut br, out);
        reads(test, &mut br);
        merged.extend(br);
    }
    *later = merged;
    walk_moves(expr, slots, min_slot, later, out);
}

fn walk_bindings_rev(
    bindings: &[(Pattern, Expr)],
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    for (pat, init) in bindings.iter().rev() {
        pattern_children(pat, &mut |def_e| {
            reads(def_e, later);
        });
        let n = native_abi::pattern_local_count(pat);
        let new_len = slots.len() - n;
        slots.truncate(new_len);
        // Shadowed/inner locals at indices >= new_len are out of scope in `init`.
        later.retain(|&i| i < new_len);
        walk_moves(init, slots, min_slot, later, out);
    }
}

fn push_binding_slots(bindings: &[(Pattern, Expr)], slots: &mut Vec<bool>) {
    for (pat, _) in bindings {
        let n = native_abi::pattern_local_count(pat);
        slots.resize(slots.len() + n, true);
    }
}

fn walk_let_moves(
    bindings: &[(Pattern, Expr)],
    body: &[Expr],
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    let base = slots.len();
    push_binding_slots(bindings, slots);
    for b in body.iter().rev() {
        walk_moves(b, slots, min_slot, later, out);
    }
    walk_bindings_rev(bindings, slots, min_slot, later, out);
    later.retain(|&i| i < base);
}

fn walk_loop_moves(
    bindings: &[(Pattern, Expr)],
    body: &[Expr],
    slots: &mut Vec<bool>,
    outer_min: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) {
    let loop_base = slots.len();
    push_binding_slots(bindings, slots);
    let mut body_later = later.clone();
    body_later.retain(|&i| i < loop_base);
    for b in body.iter().rev() {
        walk_moves(b, slots, loop_base, &mut body_later, out);
    }
    later.extend(body_later);
    walk_bindings_rev(bindings, slots, outer_min, later, out);
    later.retain(|&i| i < loop_base);
}

fn walk_extent_moves(
    e: &Expr,
    slots: &mut Vec<bool>,
    min_slot: usize,
    later: &mut BTreeSet<usize>,
    out: &mut Vec<(usize, usize)>,
) -> bool {
    let depth = slots.len();
    match e {
        Expr::Fn(_)
        | Expr::BoundFn { .. }
        | Expr::DefMacro { .. }
        | Expr::LazySeqStar { .. }
        | Expr::DelayStar { .. }
        | Expr::Try { .. }
        | Expr::Sync { .. }
        | Expr::Io { .. }
        | Expr::Time { .. }
        | Expr::WithOutStr { .. } => {
            reads(e, later);
            later.retain(|&i| i < depth);
            true
        }
        Expr::Binding { pairs, body } | Expr::WithRedefs { pairs, body } => {
            body.iter().for_each(|b| reads(b, later));
            later.retain(|&i| i < depth);
            pairs
                .iter()
                .rev()
                .for_each(|(_, v)| walk_moves(v, slots, min_slot, later, out));
            true
        }
        Expr::WithBindings { map: arg, body }
        | Expr::Locking { lockee: arg, body }
        | Expr::WithInStr { s: arg, body, .. } => {
            body.iter().for_each(|b| reads(b, later));
            later.retain(|&i| i < depth);
            walk_moves(arg, slots, min_slot, later, out);
            true
        }
        _ => false,
    }
}

/// The free locals (`< base`) an arity mentions, nested fns included
/// (absolute slot numbering makes nesting transparent).
fn arity_free(arity: &Arity, base: usize, out: &mut BTreeSet<usize>) {
    fn walk(e: &Expr, base: usize, out: &mut BTreeSet<usize>) {
        if let Expr::Local(i) = e
            && *i < base
        {
            out.insert(*i);
        }
        for_each_child(e, &mut |c| walk(c, base, out));
    }
    let probe = Expr::Fn(FunExpr {
        display: String::new(),
        arities: std::sync::Arc::from([arity.clone()]),
    });
    for_each_child(&probe, &mut |c| walk(c, base, out));
}

/// Whether a unit can rebind a core var at run time (then inline
/// arithmetic would not mean the current root).
fn mentions_rebinding(e: &Expr) -> bool {
    // (`alter-meta!` is benign here: `defn` itself uses it for
    // `:arglists`.)
    // (The var-resolving fns hand out any var — `alter-var-root`
    // included — without naming it statically.)
    const REBIND: [&str; 12] = [
        "alter-var-root",
        "with-redefs-fn",
        "intern",
        "ns-unmap",
        "eval",
        "resolve",
        "ns-resolve",
        "find-var",
        "requiring-resolve",
        "ns-interns",
        "ns-publics",
        "ns-map",
    ];
    let core_rebind = |v: &clj_value::Var| {
        v.sym.namespace() == Some("clojure.core") && REBIND.contains(&v.sym.name())
    };
    let hit = match e {
        Expr::WithRedefs { .. } => true,
        Expr::Global(v) | Expr::VarRef(v) => core_rebind(v),
        Expr::DirectNative(c) => core_rebind(&c.var),
        // Redefining a `clojure.string` var only conflicts with a Lite
        // native that serves it (a spliced definition has none, #271).
        Expr::Def { sym, .. } => match sym.namespace() {
            Some("clojure.core") => true,
            Some("clojure.string") => clj_eval::aot_boot::is_lite_string_var(sym.name()),
            _ => false,
        },
        _ => false,
    };
    let mut found = hit;
    if !found {
        for_each_child(e, &mut |c| found = found || mentions_rebinding(c));
    }
    found
}

/// The `ns` constant descriptor (see `clj_rt::native::run_ns`).
fn ns_desc(name: &str, doc: Option<&str>, attr: Option<&Map>, clauses: &[NsClause]) -> Value {
    let mut items = vec![
        Value::Str(std::sync::Arc::from(name)),
        doc.map_or(Value::Nil, |d| Value::Str(std::sync::Arc::from(d))),
        attr.map_or(Value::Nil, |a| Value::Map(a.clone())),
    ];
    for c in clauses {
        let (kind, vals) = match c {
            NsClause::ReferClojure(v) => (0, v),
            NsClause::Require(v) => (1, v),
            NsClause::Use(v) => (2, v),
            NsClause::Import(v) => (3, v),
        };
        let mut row = vec![Value::Int(kind)];
        row.extend(vals.iter().cloned());
        items.push(Value::Vector(Vector::from_vec(row)));
    }
    Value::Vector(Vector::from_vec(items))
}

/// A symbol-safe fragment of a display name.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(40)
        .collect()
}

/// Whether `front` can boot cleanly under `BootMode::Lite` (no embedded
/// stdlib / compiler / runtime eval / non-Lite namespaces or rebinds).
fn can_lite_program(front: &FrontEnd) -> bool {
    let mut defined = BTreeSet::new();
    for u in &front.units {
        if mentions_rebinding(&u.expr) {
            return false;
        }
        collect_lite_defs(&u.expr, &mut defined);
    }
    let scope = LiteScope {
        defined,
        libs: &front.libs,
    };
    front.units.iter().all(|u| expr_lite_ok(&u.expr, &scope))
}

/// What the Lite check resolves against: the program's own definitions
/// (spliced library units included) and the spliced libraries its
/// leading `ns` form may require (#271).
struct LiteScope<'a> {
    defined: BTreeSet<(String, String)>,
    libs: &'a [&'static str],
}

fn sym_key(sym: &Symbol) -> (String, String) {
    (
        sym.namespace().unwrap_or("user").to_owned(),
        sym.name().to_owned(),
    )
}

/// Whether a program references a reader of var metadata
/// ([`clj_eval::aot_boot::needs_var_meta`] over every var it names).
fn program_reads_var_meta(units: &[TopUnit]) -> bool {
    let mut refs = Vec::new();
    for u in units.iter().filter(|u| !is_inert_meta_statement(&u.expr)) {
        collect_var_refs(&u.expr, &mut refs);
    }
    let refs: Vec<(&str, &str)> = refs
        .iter()
        .map(|(ns, name)| (ns.as_str(), name.as_str()))
        .collect();
    clj_eval::aot_boot::needs_var_meta(&refs)
}

fn collect_var_refs(e: &Expr, out: &mut Vec<(String, String)>) {
    match e {
        // Discarded inert metadata writes are what this decides on
        // dropping (they are no reader).
        Expr::Do(body) => {
            let last = body.len().saturating_sub(1);
            let kept = body
                .iter()
                .enumerate()
                .filter(|&(i, s)| i == last || !is_inert_meta_statement(s));
            for (_, s) in kept {
                collect_var_refs(s, out);
            }
            return;
        }
        // (`binding` / `set!` name only dynamic vars, never a reader.)
        Expr::Global(v) | Expr::VarRef(v) => out.push(sym_key(&v.sym)),
        Expr::DirectNative(call) => out.push(sym_key(&call.var.sym)),
        _ => {}
    }
    for_each_child(e, &mut |c| collect_var_refs(c, out));
}

/// Whether `sym` is `clojure.core/name`.
fn is_core_sym(sym: &Symbol, name: &str) -> bool {
    sym.namespace() == Some("clojure.core") && sym.name() == name
}

/// `defn`'s discarded `(alter-meta! (var name) assoc :arglists '…)`
/// (and a `merge` of a constant map): a write of
/// [`clj_eval::aot_boot::LITE_INERT_DEF_META`] keys only, with constant
/// values — nothing but a var-metadata reader can tell it ran.
fn is_inert_meta_statement(e: &Expr) -> bool {
    let Some((head, args)) = global_call(e) else {
        return false;
    };
    let [Expr::VarRef(_), Expr::Global(f), rest @ ..] = args else {
        return false;
    };
    if !is_core_sym(head, "alter-meta!") {
        return false;
    }
    if is_core_sym(&f.sym, "assoc") {
        return is_inert_assoc(rest);
    }
    is_core_sym(&f.sym, "merge") && is_inert_merge(rest)
}

/// The head var's symbol and the arguments of a call to a global.
fn global_call(e: &Expr) -> Option<(&Symbol, &[Expr])> {
    match e {
        Expr::DirectNative(call) => Some((&call.var.sym, call.args.as_slice())),
        Expr::Invoke { head, args, .. } => match &**head {
            Expr::Global(v) => Some((&v.sym, args.as_slice())),
            _ => None,
        },
        _ => None,
    }
}

/// `assoc`'s `k v …` arguments: inert keys with constant values only.
fn is_inert_assoc(kvs: &[Expr]) -> bool {
    !kvs.is_empty()
        && kvs.len().is_multiple_of(2)
        && kvs.chunks(2).all(|kv| is_inert_meta_pair(&kv[0], &kv[1]))
}

/// `merge`'s argument: one constant map of inert keys.
fn is_inert_merge(args: &[Expr]) -> bool {
    match args {
        [Expr::Literal(Value::Map(m)) | Expr::Quote(Value::Map(m))] => {
            m.iter().all(|(k, _)| is_inert_def_meta_key(k))
        }
        _ => false,
    }
}

/// A metadata `key value` pair with an inert key and a constant value.
fn is_inert_meta_pair(k: &Expr, v: &Expr) -> bool {
    matches!(k, Expr::Literal(key) if is_inert_def_meta_key(key))
        && matches!(v, Expr::Literal(_) | Expr::Quote(_))
}

/// Whether `key` names a [`clj_eval::aot_boot::LITE_INERT_DEF_META`] key.
fn is_inert_def_meta_key(key: &Value) -> bool {
    matches!(key, Value::Keyword(k) if k.namespace().is_none()
        && clj_eval::aot_boot::LITE_INERT_DEF_META.contains(&k.name()))
}

/// A `def` name and its evaluated metadata pairs without the keys no
/// runtime path of a program that never reads var metadata can see
/// (only side-effect-free pairs drop; an emptied map goes away).
fn without_inert_def_meta(
    sym: &Symbol,
    pairs: Option<&[(Expr, Expr)]>,
) -> (Symbol, Option<Vec<(Expr, Expr)>>) {
    let meta = sym.meta().map(|m| {
        m.iter()
            .map(|(k, _)| k.clone())
            .filter(is_inert_def_meta_key)
            .fold(m.clone(), |m, k| m.without(&k))
    });
    let sym = sym.clone().with_meta(meta.filter(|m| !m.is_empty()));
    let pairs = pairs.map(|pairs| {
        pairs
            .iter()
            .filter(|(k, v)| !is_inert_meta_pair(k, v))
            .cloned()
            .collect::<Vec<_>>()
    });
    (sym, pairs.filter(|p| !p.is_empty()))
}

fn collect_lite_defs(e: &Expr, defined: &mut BTreeSet<(String, String)>) {
    match e {
        Expr::Def { sym, .. } => {
            defined.insert(sym_key(sym));
        }
        Expr::Declare { syms, .. } => {
            for sym in syms {
                defined.insert(sym_key(sym));
            }
        }
        _ => {}
    }
    for_each_child(e, &mut |c| collect_lite_defs(c, defined));
}

fn var_lite_ok(sym: &Symbol, defined: &BTreeSet<(String, String)>) -> bool {
    let ns = sym.namespace().unwrap_or("user");
    let name = sym.name();
    match ns {
        "clojure.core" => clj_eval::aot_boot::is_lite_core_var(name),
        // A `clojure.string` var without a Lite native may be spliced
        // in (#271): defined by the program's own units.
        "clojure.string" => {
            clj_eval::aot_boot::is_lite_string_var(name)
                || defined.contains(&(ns.to_owned(), name.to_owned()))
        }
        _ => defined.contains(&(ns.to_owned(), name.to_owned())),
    }
}

fn is_clojure_string_sym(v: &Value) -> bool {
    matches!(v, Value::Symbol(s) if s.namespace().is_none() && s.name() == "clojure.string")
}

fn lite_require_spec_alias(v: &Value) -> Option<Option<String>> {
    if is_clojure_string_sym(v) {
        return Some(None);
    }
    let Value::Vector(items) = v else {
        return None;
    };
    let mut it = items.iter();
    if !it.next().is_some_and(|first| is_clojure_string_sym(&first)) {
        return None;
    }
    while let Some(k) = it.next() {
        if matches!(&k, Value::Keyword(kw) if kw.namespace().is_none() && kw.name() == "as")
            && let Some(Value::Symbol(sym)) = it.next()
        {
            return Some(Some(sym.name().to_owned()));
        }
    }
    Some(None)
}

fn lite_direct_require_aliases(call: &clj_eval::expr::DirectCall) -> Option<Vec<Option<String>>> {
    if call.var.sym.namespace() != Some("clojure.core")
        || call.var.sym.name() != "require"
        || call.args.is_empty()
    {
        return None;
    }
    call.args
        .iter()
        .map(|arg| match arg {
            Expr::Quote(val) => lite_require_spec_alias(val),
            _ => None,
        })
        .collect()
}

fn ns_clause_lite_ok(clause: &NsClause, libs: &[&str]) -> bool {
    match clause {
        NsClause::ReferClojure(_) => true,
        NsClause::Require(specs) => specs
            .iter()
            .all(|v| lite_require_spec_alias(v).is_some() || libs.iter().any(|l| names_lib(v, l))),
        NsClause::Use(_) | NsClause::Import(_) => false,
    }
}

fn expr_lite_ok(e: &Expr, scope: &LiteScope<'_>) -> bool {
    if let Expr::DirectNative(call) = e
        && lite_direct_require_aliases(call).is_some()
    {
        return true;
    }
    let here = match e {
        Expr::DefMacro { .. } => false,
        Expr::Ns { clauses, .. } => clauses.iter().all(|c| ns_clause_lite_ok(c, scope.libs)),
        Expr::Global(v) | Expr::VarRef(v) | Expr::Set { var: v, .. } => {
            var_lite_ok(&v.sym, &scope.defined)
        }
        Expr::DirectNative(call) => var_lite_ok(&call.var.sym, &scope.defined),
        Expr::Binding { pairs, .. } | Expr::WithRedefs { pairs, .. } => pairs
            .iter()
            .all(|(v, _)| var_lite_ok(&v.sym, &scope.defined)),
        _ => true,
    };
    if !here {
        return false;
    }
    let mut ok = true;
    for_each_child(e, &mut |c| {
        if ok && !expr_lite_ok(c, scope) {
            ok = false;
        }
    });
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_collections_pool_without_the_reader() {
        use clj_value::{Keyword, List, SortSpec, new_map};
        let reads = |text: &str| native_abi::read_constant(text).expect("reads");
        // Nested data, catch class lists, and arglists with a tagged param.
        for text in [
            "[\"java.lang.Exception\"]",
            "{:a [1 2.5 \\c] \"b\" #{x/y nil}}",
            "([^CharSequence s re f] [])",
        ] {
            let v = reads(text).with_meta(None);
            assert!(matches!(coll_const(&v), Some(Const::Coll(..))), "{text}");
        }
        // Anything else reads back: metadata, a sorted map, an item with
        // no simple form, more forms than the cap, a non-collection.
        let tagged = Value::List(List::from_vec(vec![Value::Int(1)])).with_meta(Some(
            new_map().insert(Value::Keyword(Keyword::new(None, "line")), Value::Int(1)),
        ));
        let sorted = Value::Map(new_map().with_sort(Some(SortSpec::Natural)));
        let big = reads("[1N]");
        let wide = Value::Vector(clj_value::Vector::from_vec(vec![
            Value::Int(0);
            COLL_FORMS_MAX
        ]));
        for v in [tagged, sorted, big, wide, Value::Int(1)] {
            assert!(coll_const(&v).is_none(), "{v:?}");
        }
    }

    #[test]
    fn meta_symbols_pool_without_the_reader() {
        use clj_value::{Keyword, List, new_map};
        let kw = |n: &str| Value::Keyword(Keyword::new(None, n));
        let named = |pairs: Vec<(Value, Value)>| {
            let meta = pairs
                .into_iter()
                .fold(new_map(), |m, (k, v)| m.insert(k, v));
            Value::Symbol(Symbol::new(Some("user"), "f")).with_meta(Some(meta))
        };
        let simple = named(vec![
            (kw("private"), Value::Bool(true)),
            (kw("k"), Value::Keyword(Keyword::new(Some("a"), "b"))),
            (kw("tag"), Value::Symbol(Symbol::new(None, "String"))),
            (kw("doc"), Value::Str(std::sync::Arc::from("d"))),
        ]);
        assert!(matches!(
            meta_symbol_const(&simple),
            Some(Const::MetaSymbol { ns, name, meta })
                if ns.as_deref() == Some("user") && name == "f" && meta.len() == 4
        ));
        // Anything else reads back: a non-keyword key, a nested value, a
        // symbol carrying its own metadata, a plain symbol, a non-symbol.
        let list = Value::List(List::from_vec(vec![Value::Int(1)]));
        assert!(meta_symbol_const(&named(vec![(Value::Int(1), Value::Nil)])).is_none());
        assert!(meta_symbol_const(&named(vec![(kw("arglists"), list)])).is_none());
        assert!(meta_symbol_const(&named(vec![(kw("tag"), simple.clone())])).is_none());
        assert!(meta_symbol_const(&Value::Symbol(Symbol::new(None, "plain"))).is_none());
        assert!(meta_symbol_const(&Value::Int(1)).is_none());
    }

    #[test]
    fn static_call_op_needs_literal_class_and_member() {
        // The analyzer always synthesizes `__static*` with literal strings;
        // anything else (or a short arg list) takes the generic invoke.
        let lit = |s: &str| Expr::Quote(Value::Str(std::sync::Arc::from(s)));
        let abs = [
            lit("java.lang.Math"),
            lit("abs"),
            Expr::Literal(Value::Int(-5)),
        ];
        assert!(matches!(
            static_call_op(&abs),
            Some((StaticOp::MathAbs, [_]))
        ));
        assert!(static_call_op(&abs[..1]).is_none());
        let computed = [
            Expr::Literal(Value::Nil),
            lit("abs"),
            Expr::Literal(Value::Int(-5)),
        ];
        assert!(static_call_op(&computed).is_none());
        assert!(static_call_op(&[lit("java.lang.Math"), lit("signum"), lit("x")]).is_none());
    }

    #[test]
    #[should_panic(expected = "lower_tail_form partition")]
    fn lower_tail_form_rejects_upstream_expr() {
        let ctx = Context::create();
        let mut lowerer = Lowerer::with_bitcode(&ctx, false);
        let func = lowerer
            .module
            .add_function("test_fn", lowerer.code_ty, None);
        let entry = ctx.append_basic_block(func, "entry");
        let dest = lowerer.ptr.const_null();
        let mut f = FnCtx {
            func,
            entry,
            fail: entry,
            slots: Vec::new(),
            env: Vec::new(),
            recur: Vec::new(),
            captured: dest,
            moves: Vec::new(),
        };
        let _ = lowerer.lower_tail_form(&mut f, &Expr::Literal(Value::Nil), dest);
    }
}
