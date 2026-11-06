// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Lazy seqs (S5D, C4): chunked (32), memoized, nil-punning.
//!
//! [`PendingSeq`] is a seq value whose tail needs the evaluator:
//! forcing runs one [`Thunk`] step (a whole input chunk for
//! chunked inputs, one item otherwise — the JVM's observable
//! side-effect order), memoizes the chunk in the shared cell, and
//! chains the tail value (which may be pending again). Realization
//! sees the ambient bindings, never the creation ones (probed).
//! Thunk errors restore the pending state, so a retry re-runs the
//! step like the JVM's `LazySeq`.

use std::sync::{Arc, Mutex};

use num_traits::ToPrimitive;
use smallvec::SmallVec;

use clj_value::{
    CHUNK_SIZE, ConsValue, CycleValue, ExtensionValue, IterateValue, KeySeqValue, List, Map,
    RangeNum, RangeValue, RealizedSeq, RepeatValue, Seq, SeqKind, Set, SubVectorValue, ThrowClass,
    TransientVector, ValSeqValue, Value, Vector,
};

use crate::core::Num;
use crate::env::Env;
use crate::error::{NativeError, Signal, jvm_class_of};
use crate::eval::Evaluator;
use crate::expr::Expr;
use crate::jdkio::JdkCursor;
use crate::ns::{NativeFnPtr, check_arity, signal_to_native};

/// Which lazy class a [`PendingSeq`] reports: almost everything is
/// a `LazySeq`, but `rest` of a forced chunk is a `ChunkedCons`
/// (like the JVM's `more`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LazyKind {
    LazySeq,
    ChunkedCons,
}

impl LazyKind {
    fn jvm_name(self) -> &'static str {
        match self {
            LazyKind::LazySeq => "clojure.lang.LazySeq",
            LazyKind::ChunkedCons => "clojure.lang.ChunkedCons",
        }
    }
}

/// A seq value with an evaluator-backed tail: forcing needs
/// [`force_pending`]. Prints best-effort from memoized state
/// (natives realize first, so this is out-of-band only).
#[derive(Debug)]
pub struct PendingSeq {
    state: Mutex<SeqState>,
    meta: Option<Map>,
    kind: LazyKind,
}

/// One flattened `concat` level: an outer `(rest, idx)` to resume
/// after the descended-into inner concat exhausts (#205:
/// `mapcat` tails nest one concat per element, so the descent is
/// iterative and the resume stack lives here instead of the Rust
/// stack).
#[derive(Clone, Debug)]
pub(crate) struct ConcatFrame {
    rest: Arc<Vec<Value>>,
    idx: usize,
    parent: ConcatStack,
}

/// A `concat` resume stack (persistent: push, pop, and clone are
/// O(1) — `force_pending` clones the whole thunk per force, so an
/// owned `Vec` would make every step O(depth)).
pub(crate) type ConcatStack = Option<Arc<ConcatFrame>>;

impl Drop for ConcatFrame {
    fn drop(&mut self) {
        // Deep stacks (e.g. `(reduce concat ...)` over thousands of
        // inputs) would overflow the stack dropping parent-by-parent
        // recursively, so unwind sole-owned parents iteratively
        // (shared parents resume through their own `Drop`, like
        // `PendingSeq`).
        let mut next = self.parent.take();
        while let Some(node) = next {
            match Arc::try_unwrap(node) {
                Ok(mut frame) => next = frame.parent.take(),
                Err(_) => break,
            }
        }
    }
}

/// The memoized state: pending thunks become realized chunks.
#[derive(Clone, Debug)]
enum SeqState {
    Pending(Box<Thunk>),
    Chunk {
        items: Arc<[Value]>,
        pos: usize,
        tail: Value,
    },
    Empty,
}

/// One lazy step: the captured op plus its (advanced) inputs.
/// Chunks hold at most [`CHUNK_SIZE`] items.
#[derive(Clone, Debug)]
pub(crate) enum Thunk {
    Map {
        f: Value,
        inputs: Vec<Value>,
    },
    Filter {
        f: Value,
        input: Value,
    },
    Remove {
        f: Value,
        input: Value,
    },
    Take {
        n: Value,
        input: Value,
    },
    Concat {
        /// The input currently being stepped (`Nil` when there are
        /// no inputs at all).
        cur: Value,
        /// The remaining inputs behind `Arc`: `force_pending`
        /// clones the whole thunk per force (to restore it when a
        /// step fails), so owned inputs cost O(n) per force —
        /// `(apply concat buckets)` over tens of thousands of
        /// buckets was O(n^2) (recipes
        /// `string_function_calculation` took ~70s at n=32K).
        /// Shared inputs plus the index below make both the clone
        /// and the step O(1).
        rest: Arc<Vec<Value>>,
        /// The next index into `rest` once `cur` exhausts.
        idx: usize,
        /// Outer levels flattened by the iterative descent (#205):
        /// each frame resumes after its inner concat exhausts.
        /// Persistent (O(1) clone — see `ConcatFrame`).
        outer: ConcatStack,
    },
    MapCat {
        outer: Value,
    },
    Chain {
        head: Vec<Value>,
        tail: Value,
    },
    // The `lazy-seq*` control form (S5D): an unevaluated body
    // plus its lexical env (dynamic bindings stay force-time, like
    // the JVM closure).
    Thunk0 {
        body: Arc<Vec<Expr>>,
        env: Env,
    },
    /// The VM's `lazy-seq*`: the body compiled as a zero-arity
    /// closure over its captured locals (`recur` loops inside it).
    Call0 {
        f: Value,
    },
    Repeatedly {
        f: Value,
    },
    Interleave {
        queue: Vec<Value>,
        inputs: Vec<Value>,
    },
    Interpose {
        sep: Value,
        input: Value,
        first: bool,
    },
    TakeWhile {
        f: Value,
        input: Value,
    },
    DropWhile {
        f: Value,
        input: Value,
    },
    Drop {
        n: Value,
        input: Value,
    },
    TakeNth {
        n: Value,
        input: Value,
    },
    Partition {
        n: Value,
        step: Value,
        pad: Option<Value>,
        input: Value,
    },
    PartitionAll {
        n: Value,
        step: Value,
        input: Value,
    },
    PartitionBy {
        f: Value,
        input: Value,
    },
    PartitionByAdvance {
        f: Value,
        run: Value,
        s: Value,
    },
    PartitionRun {
        f: Value,
        fv: Value,
        input: Value,
    },
    Distinct {
        // Persistent: `force_pending` clones the thunk per step (to
        // restore it when a step fails), so an owned set made
        // `distinct` quadratic (recipes `convex_hull`).
        seen: clj_value::MemberSet,
        input: Value,
    },
    Dedupe {
        last: Option<Value>,
        input: Value,
    },
    /// `keep` (`skip: None`, nil results drop) and a `for` stage's
    /// `:when` chain (`__for-keep*`: only results identical to
    /// `skip` drop — nil results stay).
    Keep {
        f: Value,
        input: Value,
        skip: Option<Value>,
    },
    KeepIndexed {
        f: Value,
        input: Value,
        idx: i64,
    },
    MapIndexed {
        f: Value,
        input: Value,
        idx: i64,
    },
    Reductions {
        f: Value,
        acc: Option<Value>,
        has_init: bool,
        input: Value,
        started: bool,
    },
    Replace {
        smap: Value,
        input: Value,
    },
    /// T1200 D1: a transducer stepping fn over lockstep sources
    /// (the `TransformerIterator` role for `sequence`/eduction seq).
    Sequence {
        step: Value,
        srcs: Vec<Value>,
    },
    /// S12: a `seque` drain over the feeder queue (one blocking
    /// take per step; the eos sentinel or a dropped feeder ends
    /// the seq).
    Seque {
        rx: crate::collmisc::SequeRx,
    },
    /// S12: one `iteration` cell (step once; emit through
    /// `:vf` when `:somef` passes, continue at `:kf` unless nil).
    Iteration {
        step: Value,
        somef: Value,
        vf: Value,
        kf: Value,
        k: Value,
    },
    /// S12: a `file-seq` walk (paths left to visit, root first;
    /// each step pops one path and pushes its children when it is
    /// a directory).
    FileSeq {
        stack: Vec<String>,
    },
    /// S12: a `chunk-cons` head (the exact chunk plus its tail;
    /// empty chunks collapse at construction, so heads are
    /// never empty).
    ChunkHead {
        chunk: crate::collmisc::ChunkValue,
        tail: Value,
    },
    /// S17: pulls through a shared JDK-handle cursor (up to
    /// [`CHUNK_SIZE`] items per step when `chunked`, one
    /// otherwise) for `iterator-seq`/`stream-seq!`/
    /// `resultset-seq` tails/`file-seq`.
    JdkPull {
        cursor: JdkCursor,
        chunked: bool,
    },
}

/// A forced step: empty, or a chunk plus its tail value.
enum Forced {
    Empty,
    Chunk { items: Arc<[Value]>, tail: Value },
}

/// One input step: empty, one head plus tail, or a chunk plus tail.
pub(crate) enum StepUnit {
    Empty,
    One { head: Value, tail: Value },
    Chunk { items: Arc<[Value]>, tail: Value },
}

impl PendingSeq {
    /// A pending seq value from a thunk (metadata-free).
    fn pending(thunk: Thunk) -> Value {
        Self::pending_kind(thunk, LazyKind::LazySeq)
    }

    /// A pending seq value with an explicit lazy class.
    fn pending_kind(thunk: Thunk, kind: LazyKind) -> Value {
        Value::Extension(Arc::new(Self {
            state: Mutex::new(SeqState::Pending(Box::new(thunk))),
            meta: None,
            kind,
        }))
    }

    /// The metadata map, if any.
    pub(crate) fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Whether the seq has forced its first step (`realized?` — a
    /// pending thunk reads false without forcing, like the JVM's
    /// `LazySeq.isRealized`).
    pub(crate) fn is_realized(&self) -> bool {
        !matches!(
            *self.state.lock().unwrap_or_else(|e| e.into_inner()),
            SeqState::Pending(_)
        )
    }

    /// Whether this pending seq reports `ChunkedCons` (S12
    /// `chunked-seq?` answers true for these: `chunk-cons`
    /// results and `rest`-of-chunk tails).
    pub(crate) fn is_chunked_cons(&self) -> bool {
        matches!(self.kind, LazyKind::ChunkedCons)
    }

    /// Build a `ChunkedCons`-classed pending seq from a thunk (S12
    /// `chunk-cons`).
    pub(crate) fn pending_chunked_cons(thunk: Thunk) -> Value {
        Self::pending_kind(thunk, LazyKind::ChunkedCons)
    }

    /// A metadata-carrying clone with a FRESH memo cell (like the
    /// JVM, whose `withMeta` re-wraps the fn). Staged for
    /// `with-meta` (S5D).
    #[allow(dead_code)]
    pub(crate) fn with_meta_fresh(&self, meta: Option<Map>) -> Value {
        let state = match self.state.lock().ok().map(|s| s.clone()) {
            Some(SeqState::Pending(t)) => SeqState::Pending(t),
            // Forced states re-wrap as chains (already realized, so
            // no recomputation either way).
            Some(SeqState::Chunk { items, pos, tail }) => {
                SeqState::Pending(Box::new(Thunk::Chain {
                    head: items[pos..].to_vec(),
                    tail,
                }))
            }
            _ => SeqState::Empty,
        };
        Value::Extension(Arc::new(Self {
            state: Mutex::new(state),
            meta,
            kind: self.kind,
        }))
    }
}

/// Swap a seq cell's memoized state out (leaving `Empty`), even
/// when the mutex is poisoned (a panic during `drop` aborts).
fn take_seq_state(state: &Mutex<SeqState>) -> SeqState {
    let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());
    std::mem::replace(&mut *guard, SeqState::Empty)
}

/// The chain-ward link of an unforced concat: `cur` when it is a
/// cell (left nests like `(reduce concat ...)` and `mapcat`
/// tails), else the sole remaining rest input when it is a
/// sole-owned cell (right nests). Sibling cells unwind through
/// their own `Drop`, and `outer` through `ConcatFrame::Drop`.
fn drop_concat_link(cur: Value, rest: &mut Arc<Vec<Value>>, idx: usize) -> Option<Value> {
    if as_pending(&cur).is_some() {
        return Some(cur);
    }
    if idx + 1 == rest.len() {
        if let Some(slot) = Arc::get_mut(rest).and_then(|v| v.get_mut(idx)) {
            if as_pending(slot).is_some() {
                return Some(std::mem::replace(slot, Value::Nil));
            }
        }
    }
    None
}

/// The one chain-ward link of an unforced thunk, if it has exactly
/// one (`None` drops the whole thunk normally — every nested seq
/// cell unwinds through its own `Drop`, so those drops stay
/// shallow too).
fn drop_thunk_link(thunk: Thunk) -> Option<Value> {
    // `SeqState`/`Thunk` destructuring must stay exhaustive: a new
    // thunk arm without a link decision fails closed at compile time.
    let single = |mut links: Vec<Value>| {
        if links.len() == 1 { links.pop() } else { None }
    };
    // Arms group same-shaped variants with OR-patterns (the match
    // stays exhaustive — no wildcard — so new variants still fail
    // closed at compile time).
    match thunk {
        Thunk::Map { inputs, .. } => single(inputs),
        Thunk::Filter { input, .. }
        | Thunk::Remove { input, .. }
        | Thunk::Take { input, .. }
        | Thunk::Interpose { input, .. }
        | Thunk::TakeWhile { input, .. }
        | Thunk::DropWhile { input, .. }
        | Thunk::Drop { input, .. }
        | Thunk::TakeNth { input, .. }
        | Thunk::Partition { input, .. }
        | Thunk::PartitionAll { input, .. }
        | Thunk::PartitionBy { input, .. }
        | Thunk::PartitionRun { input, .. }
        | Thunk::Distinct { input, .. }
        | Thunk::Dedupe { input, .. }
        | Thunk::Keep { input, .. }
        | Thunk::KeepIndexed { input, .. }
        | Thunk::MapIndexed { input, .. }
        | Thunk::Reductions { input, .. }
        | Thunk::Replace { input, .. } => Some(input),
        Thunk::Concat {
            cur, mut rest, idx, ..
        } => drop_concat_link(cur, &mut rest, idx),
        Thunk::MapCat { outer } => Some(outer),
        Thunk::Chain { tail, .. } | Thunk::ChunkHead { tail, .. } => Some(tail),
        Thunk::Thunk0 { .. }
        | Thunk::Call0 { .. }
        | Thunk::Repeatedly { .. }
        | Thunk::Seque { .. }
        | Thunk::FileSeq { .. }
        | Thunk::JdkPull { .. }
        | Thunk::Iteration { .. } => None,
        Thunk::Interleave { queue, inputs } => {
            let mut links = queue;
            links.extend(inputs);
            single(links)
        }
        // The run is a realized prefix; the chain continues at `s`.
        Thunk::PartitionByAdvance { s, .. } => Some(s),
        Thunk::Sequence { srcs, .. } => single(srcs),
    }
}

/// Unwrap a sole-owned successor seq cell behind an extension Arc
/// (shared cells stay for their own `Drop` to unwind later).
fn drop_next_cell(link: &mut Value) -> Option<SeqState> {
    let Value::Extension(arc) = link else {
        return None;
    };
    let cell = Arc::get_mut(arc)?
        .as_any_mut()?
        .downcast_mut::<PendingSeq>()?;
    Some(take_seq_state(&cell.state))
}

impl Drop for PendingSeq {
    fn drop(&mut self) {
        // Long lazy chains (500K+ forced cells with the head held,
        // e.g. `(nth (take 500000 (range ...)) 499999)`) would
        // overflow the stack if each cell's tail dropped
        // recursively, so unwind sole-owned successor cells
        // iteratively: each successor's state is swapped out
        // (leaving `Empty`) before its shell drops, keeping every
        // nested drop shallow.
        let mut cur = take_seq_state(&self.state);
        loop {
            let mut link = match cur {
                SeqState::Empty => break,
                SeqState::Chunk { tail, .. } => tail,
                SeqState::Pending(thunk) => match drop_thunk_link(*thunk) {
                    Some(link) => link,
                    None => break,
                },
            };
            match drop_next_cell(&mut link) {
                Some(next) => cur = next,
                // `link` drops here: a shared cell (its own `Drop`
                // resumes the unwind when its last reference
                // releases) or a strict value (shallow drop).
                None => break,
            }
        }
    }
}

impl ExtensionValue for PendingSeq {
    fn class_name(&self) -> String {
        self.kind.jvm_name().to_owned()
    }

    fn print(&self) -> String {
        // Best-effort over memoized state (natives realize first):
        // forced chunks print their known prefix, pending prints
        // `(...)`, empties `()`.
        let Ok(state) = self.state.lock() else {
            return "(...)".to_owned();
        };
        match &*state {
            SeqState::Empty => "()".to_owned(),
            SeqState::Pending(_) => "(...)".to_owned(),
            SeqState::Chunk { items, pos, tail } => {
                let parts: Vec<String> = items[*pos..]
                    .iter()
                    .map(|v| clj_value::pr_to_string(v, &clj_value::PrintOpts::pr()))
                    .collect();
                let open = parts.join(" ");
                let tail_empty =
                    matches!(tail, Value::Nil) || matches!(tail, Value::List(l) if l.is_empty());
                if tail_empty && !open.is_empty() {
                    format!("({open})")
                } else if open.is_empty() {
                    "(...)".to_owned()
                } else {
                    format!("({open} ...)")
                }
            }
        }
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        // Unrealized identity only (`=` realizes first, so deep
        // equality never reaches here).
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| std::ptr::eq(self, o))
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_usize(self as *const Self as usize);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// Downcast a seq value to its pending cell, if any.
pub(crate) fn as_pending(v: &Value) -> Option<&PendingSeq> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<PendingSeq>(),
        _ => None,
    }
}

/// Downcast a seq value to its realized seq, if any.
pub(crate) fn as_realized(v: &Value) -> Option<&RealizedSeq> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<RealizedSeq>(),
        _ => None,
    }
}

macro_rules! seq_downcast {
    ($name:ident, $t:ty, $doc:literal) => {
        #[doc = $doc]
        pub(crate) fn $name(v: &Value) -> Option<&$t> {
            match v {
                Value::Extension(e) => e.as_any().downcast_ref::<$t>(),
                _ => None,
            }
        }
    };
}

seq_downcast!(
    as_range,
    RangeValue,
    "Downcast a seq value to its range, if any."
);
seq_downcast!(
    as_repeat,
    RepeatValue,
    "Downcast a seq value to its repeat, if any."
);
seq_downcast!(
    as_cycle,
    CycleValue,
    "Downcast a seq value to its cycle, if any."
);
seq_downcast!(
    as_iterate,
    IterateValue,
    "Downcast a seq value to its iterate, if any."
);
seq_downcast!(
    as_cons_value,
    ConsValue,
    "Downcast a seq value to its cons, if any."
);
seq_downcast!(
    as_keys,
    KeySeqValue,
    "Downcast a seq value to its key view, if any."
);
seq_downcast!(
    as_vals,
    ValSeqValue,
    "Downcast a seq value to its value view, if any."
);
seq_downcast!(
    as_subvector,
    SubVectorValue,
    "Downcast a value to its subvector view, if any."
);

/// Whether the value is a seq (a list or any seq object).
pub(crate) fn is_seq_value(v: &Value) -> bool {
    matches!(v, Value::List(_))
        || as_pending(v).is_some()
        || as_realized(v).is_some()
        || as_range(v).is_some()
        || as_repeat(v).is_some()
        || as_cycle(v).is_some()
        || as_iterate(v).is_some()
        || as_cons_value(v).is_some()
        || as_keys(v).is_some()
        || as_vals(v).is_some()
}

/// The `ISeq` coercion error.
fn iseq_err(v: &Value) -> NativeError {
    NativeError::new(
        ThrowClass::IllegalArgumentException,
        format!("Don't know how to create ISeq from: {}", jvm_class_of(v)),
    )
}

/// Whether `vec` accepts the value (mirrors `RT.canSeq` over
/// our value types: transients and throwables are not seqable,
/// so `vec` fails them with the conversion error, not the seq
/// error). `seqable?` keeps its own predicate in `core.rs`;
/// S10's `hash-*-coll` starts from this set (plus the subvector
/// and eduction views `step_unit` walks).
pub(crate) fn is_seqable(v: &Value) -> bool {
    // Map-like hosts seq (oracles `g-seqable-rec`, S12
    // `sm-seq`); anything else host does not (oracle
    // `g-seqable-dt`).
    if crate::types::maplike_inst(v).is_some() {
        return true;
    }
    match v {
        Value::Nil
        | Value::List(_)
        | Value::Vector(_)
        | Value::MapEntry { .. }
        | Value::Map(_)
        | Value::Set(_)
        | Value::Str(_)
        | Value::Array(_)
        | Value::PrimitiveVector(_) => true,
        Value::Extension(e) => {
            let any = e.as_any();
            any.downcast_ref::<PendingSeq>().is_some()
                || any.downcast_ref::<RealizedSeq>().is_some()
                || any.downcast_ref::<RangeValue>().is_some()
                || any.downcast_ref::<RepeatValue>().is_some()
                || any.downcast_ref::<CycleValue>().is_some()
                || any.downcast_ref::<IterateValue>().is_some()
                || any.downcast_ref::<ConsValue>().is_some()
                || any.downcast_ref::<KeySeqValue>().is_some()
                || any.downcast_ref::<ValSeqValue>().is_some()
                || crate::emucoll::as_list(v).is_some()
        }
        _ => false,
    }
}

/// Force a pending cell one step (memoized; errors restore the
/// pending state so retries re-run). The lock never spans
/// evaluation.
fn force_pending(eval: &mut Evaluator, p: &PendingSeq) -> Result<Forced, Signal> {
    let thunk = {
        let mut state = p.state.lock().unwrap_or_else(|e| e.into_inner());
        // A single match: every arm is live (matching twice would
        // leave a dead fallback for the already-seen pending state).
        // The memoized chunk is restored untouched.
        match std::mem::replace(&mut *state, SeqState::Empty) {
            SeqState::Empty => return Ok(Forced::Empty),
            SeqState::Chunk { items, pos, tail } => {
                // Memoized chunks share their items: re-slicing per
                // force copies O(chunk) on every `first`/`next`/`seq`
                // (element-wise walks force ~3x per item —
                // gigabytes of churn on suffix-scale inputs). Fresh
                // chunks always memoize whole (`pos == 0`); only an
                // advanced cursor re-slices.
                let forced = if pos == 0 {
                    Forced::Chunk {
                        items: Arc::clone(&items),
                        tail: tail.clone(),
                    }
                } else {
                    Forced::Chunk {
                        items: Arc::from(&items[pos..]),
                        tail: tail.clone(),
                    }
                };
                *state = SeqState::Chunk { items, pos, tail };
                return Ok(forced);
            }
            SeqState::Pending(t) => t,
        }
    };
    match step_thunk(eval, (*thunk).clone()) {
        Ok(forced @ (Forced::Empty | Forced::Chunk { .. })) => {
            let mut state = p.state.lock().unwrap_or_else(|e| e.into_inner());
            *state = match &forced {
                Forced::Empty => SeqState::Empty,
                Forced::Chunk { items, tail } => SeqState::Chunk {
                    items: Arc::clone(items),
                    pos: 0,
                    tail: tail.clone(),
                },
            };
            Ok(forced)
        }
        Err(e) => {
            let mut state = p.state.lock().unwrap_or_else(|e| e.into_inner());
            *state = SeqState::Pending(thunk);
            Err(e)
        }
    }
}

/// Step a pending seq (forcing one step; forced chunks are
/// non-empty by construction — see the no-empty-chunk proof in
/// `docs/spec/t1174/s8-coverage.md`).
fn step_pending_unit(eval: &mut Evaluator, p: &PendingSeq) -> Result<StepUnit, Signal> {
    match force_pending(eval, p)? {
        Forced::Empty => Ok(StepUnit::Empty),
        Forced::Chunk { items, tail } => Ok(StepUnit::Chunk { items, tail }),
    }
}

/// Step a subvector view (eager; subvectors are never empty —
/// `subvec` answers a plain vector when `start == end`).
fn step_subvector_unit(sv: &SubVectorValue) -> Result<StepUnit, Signal> {
    let s = step_realized(&sv.items(), false).expect("subvectors are never empty");
    step_seq(&s, true, false, SeqKind::Subvec, None)
}

/// Step the entry-view arms (keys take the `true` projection).
fn step_entry_unit(eval: &mut Evaluator, v: &Value) -> Result<Option<StepUnit>, Signal> {
    if let Some(k) = as_keys(v) {
        return step_keys_unit(eval, k.underlying(), true).map(Some);
    }
    if let Some(k) = as_vals(v) {
        return step_keys_unit(eval, k.underlying(), false).map(Some);
    }
    Ok(None)
}

/// Step the late extension hooks (eduction, iteration, and the
/// `ArrayList` snapshot — same hooks as `seq`, which see for the
/// arm coupling).
fn step_unit_extension_tail(eval: &mut Evaluator, v: &Value) -> Result<StepUnit, Signal> {
    // T1200 D1: walking an eduction steps its lazy transform (same
    // hook as `seq`, which see for the arm coupling).
    if let Some((xform, coll)) = crate::xform::Eduction::parts(v) {
        let s = seq_eduction(eval, xform, coll)?;
        return step_unit(eval, &s);
    }
    // S12: walking an iteration steps its eager-first-step seq
    // (same hook as `seq`).
    if let Some(it) = crate::collmisc::as_iteration(v) {
        let s = crate::collmisc::seq_iteration(eval, it)?;
        return step_unit(eval, &s);
    }
    // #205: walking an `ArrayList` steps its snapshot seq (same
    // hook as `seq`).
    if let Some(items) = crate::emucoll::list_snapshot(v) {
        let s = realized_value(items, false, true, false, clj_value::SeqKind::Array);
        return step_unit(eval, &s);
    }
    Err(eval.throw(iseq_err(v)))
}

/// Step an extension input.
fn step_unit_extension(eval: &mut Evaluator, v: &Value) -> Result<StepUnit, Signal> {
    if let Some(p) = as_pending(v) {
        return step_pending_unit(eval, p);
    }
    if let Some(s) = as_realized(v) {
        return step_seq(s.seq(), s.is_counted(), s.is_idrop(), s.kind(), s.meta());
    }
    if let Some(r) = as_range(v) {
        return step_range_value(eval, r);
    }
    if let Some(r) = as_repeat(v) {
        return step_repeat_value(r);
    }
    if let Some(c) = as_cycle(v) {
        return step_cycle_value(eval, c);
    }
    if let Some(it) = as_iterate(v) {
        return step_iterate_value(eval, it);
    }
    if let Some(c) = as_cons_value(v) {
        return step_cons_value(eval, c);
    }
    if let Some(unit) = step_entry_unit(eval, v)? {
        return Ok(unit);
    }
    if let Some(sv) = as_subvector(v) {
        return step_subvector_unit(sv);
    }
    step_unit_extension_tail(eval, v)
}

/// Step a list (one head plus its tail).
fn step_list_unit(l: &List) -> Result<StepUnit, Signal> {
    match l.first() {
        None => Ok(StepUnit::Empty),
        Some(head) => {
            let tail = l
                .drop_first()
                .map(|t| Value::List(t.with_meta(None)))
                .unwrap_or_else(|| Value::List(List::new_sync()));
            Ok(StepUnit::One {
                head: head.clone(),
                tail,
            })
        }
    }
}

/// Step a vector (chunked-lazy: 32 items realize per step, like
/// strings — the old code collected every item eagerly, twice).
fn step_vector_unit(x: &Vector) -> Result<StepUnit, Signal> {
    match Seq::vector_from(x.clone(), 0) {
        None => Ok(StepUnit::Empty),
        Some(s) => step_seq(&s, true, true, SeqKind::Chunked, None),
    }
}

/// Step a map (small maps keep `Map` kind).
fn step_map_unit(m: &Map) -> Result<StepUnit, Signal> {
    let entries: Vec<Value> = m
        .iter()
        .map(|(k, val)| clj_value::map_entry(k.clone(), val.clone()))
        .collect();
    let small = !m.is_hashy();
    let kind = if small { SeqKind::Map } else { SeqKind::Key };
    step_realized(&entries, false).map_or_else(
        || Ok(StepUnit::Empty),
        |s| step_seq(&s, small, small, kind, None),
    )
}

/// Step a set.
fn step_set_unit(s: &Set) -> Result<StepUnit, Signal> {
    step_realized(&s.iter().cloned().collect::<Vec<_>>(), false).map_or_else(
        || Ok(StepUnit::Empty),
        |s| step_seq(&s, false, false, SeqKind::Key, None),
    )
}

/// Step a string (its 32-char chunks realize on demand, O(1) per
/// step over the shared string, like the JVM's `StringSeq`).
fn step_str_unit(s: &Arc<str>) -> Result<StepUnit, Signal> {
    match Seq::string_from(Arc::clone(s), 0, 0) {
        None => Ok(StepUnit::Empty),
        Some(seq) => step_seq(&seq, true, true, SeqKind::Str, None),
    }
}

/// Step one input: empty, one head, or a whole chunk (chunked
/// inputs yield their current chunk up to [`CHUNK_SIZE`]).
pub(crate) fn step_unit(eval: &mut Evaluator, v: &Value) -> Result<StepUnit, Signal> {
    match v {
        Value::Nil => Ok(StepUnit::Empty),
        Value::Extension(_) => step_unit_extension(eval, v),
        Value::List(l) => step_list_unit(l),
        Value::Vector(x) => step_vector_unit(x),
        // Entries step exactly like their 2-vector (chunked).
        Value::MapEntry { key, val, .. } => {
            let tmp = Vector::from_vec(vec![(**key).clone(), (**val).clone()]);
            step_vector_unit(&tmp)
        }
        Value::Map(m) => step_map_unit(m),
        Value::Set(s) => step_set_unit(s),
        Value::Str(s) => step_str_unit(s),
        Value::Array(a) => step_realized(&a.snapshot(), false).map_or_else(
            || Ok(StepUnit::Empty),
            |s| step_seq(&s, true, false, SeqKind::Array, None),
        ),
        Value::PrimitiveVector(p) => step_realized(&p.items(), true).map_or_else(
            || Ok(StepUnit::Empty),
            |s| step_seq(&s, true, true, SeqKind::Chunked, None),
        ),
        // Map-like hosts step their content entries (like
        // `seq_value`; anything else host has no seq).
        Value::HostType(_) => match crate::types::maplike_inst(v) {
            Some(i) => step_map_unit(&i.content),
            None => Err(eval.throw(iseq_err(v))),
        },
        _ => Err(eval.throw(iseq_err(v))),
    }
}

/// Step a pure seq: chunked seqs yield the current chunk, anything
/// else one head. Tails inherit the JVM interfaces and class
/// (`more` preserves them); only `StringSeq` tails keep metadata
/// (like `StringSeq.next` — every other `next` drops it).
fn step_seq(
    s: &Seq,
    counted: bool,
    idrop: bool,
    kind: SeqKind,
    meta: Option<&Map>,
) -> Result<StepUnit, Signal> {
    if let Some((chunk, rest)) = s.chunk_next() {
        // `chunk_next` never yields empty chunks (`Seq::chunked`
        // rejects empties; `next` keeps `pos < len`).
        let tail = Value::Extension(Arc::new(seq_tail(rest, counted, idrop, kind, meta)));
        // One copy into the chunk (a borrowed vector leaf clones its
        // items; a string or packed chunk is already owned).
        let items: Arc<[Value]> = match chunk {
            std::borrow::Cow::Borrowed(b) => Arc::from(b),
            std::borrow::Cow::Owned(v) => Arc::from(v),
        };
        return Ok(StepUnit::Chunk { items, tail });
    }
    match s.first() {
        None => Ok(StepUnit::Empty),
        Some(head) => {
            let tail = Value::Extension(Arc::new(seq_tail(s.more(), counted, idrop, kind, meta)));
            Ok(StepUnit::One {
                head: head.clone(),
                tail,
            })
        }
    }
}

/// A stepped tail seq (metadata kept only for `StringSeq`).
fn seq_tail(s: Seq, counted: bool, idrop: bool, kind: SeqKind, meta: Option<&Map>) -> RealizedSeq {
    let keep = if matches!(kind, SeqKind::Str) {
        meta.cloned()
    } else {
        None
    };
    RealizedSeq::flagged(s, counted, idrop, kind).with_meta(keep)
}

/// Build a pure seq from items (chunked for vectors, eager
/// otherwise), or `None` when empty.
fn step_realized(items: &[Value], chunked: bool) -> Option<Seq> {
    if chunked {
        Seq::chunked(items.to_vec())
    } else {
        Seq::eager(items.to_vec())
    }
}

/// Step a range: up to [`CHUNK_SIZE`] verbatim heads plus the
/// shifted range (nil tail when done). An `Int`+`Int` overflow
/// ends the walk (like `Range.forceChunk`, whose promoting
/// `Numbers.addP` always lands out of bounds past `i64`); a
/// `BigDec`/ratio step with no exact decimal raises the JVM
/// `ArithmeticException` (oracle `range-mixed`).
fn step_range_value(eval: &mut Evaluator, r: &RangeValue) -> Result<StepUnit, Signal> {
    if let Some(unit) = step_long_range(r) {
        return Ok(unit);
    }
    let mut items = Vec::with_capacity(CHUNK_SIZE);
    let mut cur = r.clone();
    // One tail per chunk: `cur` advances in place, boxed once below.
    let mut live = true;
    while items.len() < CHUNK_SIZE {
        let Some((head, advanced)) = cur.step_in_place() else {
            break;
        };
        items.push(head);
        if !advanced {
            if cur.step_inexact() {
                return Err(eval.throw(range_step_error()));
            }
            live = false;
            break;
        }
    }
    // Empty ranges never exist as values (`range` nil-puns), so
    // stepping always yields at least one head.
    debug_assert!(!items.is_empty(), "stepped ranges are non-empty");
    let tail = if live && !cur.done() {
        Value::Extension(Arc::new(cur))
    } else {
        Value::Nil
    };
    Ok(StepUnit::Chunk {
        items: Arc::from(items),
        tail,
    })
}

/// A counted `long` range's next chunk built directly (#226): the
/// items are the range's own (every one is in `i64` range — the
/// count was exact at construction), and the tail is a fresh range
/// past them (no metadata, like `step_in_place`).
fn step_long_range(r: &RangeValue) -> Option<StepUnit> {
    let (RangeNum::Int(cur), RangeNum::Int(step), Some(count)) = (
        r.cur(),
        r.step(),
        r.count().filter(|c| r.is_long() && *c > 0),
    ) else {
        return None;
    };
    let n = count.min(CHUNK_SIZE as i64);
    let items: Arc<[Value]> = (0..n).map(|k| Value::Int(cur + k * step)).collect();
    let rest = count - n;
    let tail = if rest > 0 {
        let next = RangeValue::new(
            RangeNum::Int(cur + n * step),
            r.end(),
            RangeNum::Int(step),
            true,
            Some(rest),
        );
        Value::Extension(Arc::new(next))
    } else {
        Value::Nil
    };
    Some(StepUnit::Chunk { items, tail })
}

/// Step a repeat: one head plus the decremented repeat (infinite
/// repeats return themselves; tails drop metadata like
/// `Repeat.next`).
fn step_repeat_value(r: &RepeatValue) -> Result<StepUnit, Signal> {
    match r.n() {
        Some(n) if n <= 0 => Ok(StepUnit::Empty),
        Some(n) => Ok(StepUnit::One {
            head: r.x().clone(),
            tail: Value::Extension(Arc::new(RepeatValue::new(r.x().clone(), Some(n - 1)))),
        }),
        None => Ok(StepUnit::One {
            head: r.x().clone(),
            tail: Value::Extension(Arc::new(RepeatValue::new(r.x().clone(), None))),
        }),
    }
}

/// Step a cycle: the head of `current`, restarting from `prev`
/// when it runs out (tails drop metadata like `Cycle.next`).
fn step_cycle_value(eval: &mut Evaluator, c: &CycleValue) -> Result<StepUnit, Signal> {
    if let Some((head, tail)) = step_head(eval, c.current())? {
        let next = CycleValue::new(c.prev().clone(), tail);
        return Ok(StepUnit::One {
            head,
            tail: Value::Extension(Arc::new(next)),
        });
    }
    // `prev` is never nil (seq'd non-nil at construction, preserved
    // by tails, fields immutable).
    if let Some((head, tail)) = step_head(eval, c.prev())? {
        let next = CycleValue::new(c.prev().clone(), tail);
        return Ok(StepUnit::One {
            head,
            tail: Value::Extension(Arc::new(next)),
        });
    }
    Ok(StepUnit::Empty)
}

/// Read an iterate head, realizing the seed at most once (like
/// `Iterate.first`, memoized). A nil step fn is the JVM's field
/// NPE (construction only checks non-nil callees, like
/// `Iterate.create`'s reflectively-found `IFn` cast).
fn iterate_seed(eval: &mut Evaluator, it: &IterateValue) -> Result<Value, Signal> {
    if let Some(seed) = it.seed_opt() {
        return Ok(seed);
    }
    if matches!(it.f(), Value::Nil) {
        return Err(eval.throw(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.IFn.invoke(Object)\" because \"this.f\" is null"
                .to_owned(),
        )));
    }
    crate::core::check_ifn(eval, it.f())?;
    let seed = eval.call(it.f(), std::slice::from_ref(it.prev()))?;
    it.store_seed(seed.clone());
    Ok(seed)
}

/// Step an iterate: realize this head (at most one `f` call, like
/// `next` calling `first`) plus a deferred tail (no call, like
/// `next`'s unrealized seed). Tails drop metadata.
fn step_iterate_value(eval: &mut Evaluator, it: &IterateValue) -> Result<StepUnit, Signal> {
    let head = iterate_seed(eval, it)?;
    let tail = IterateValue::deferred(it.f().clone(), head.clone());
    Ok(StepUnit::One {
        head,
        tail: Value::Extension(Arc::new(tail)),
    })
}

/// Step a cons: the head plus the tail untouched (like
/// `Cons.more`, which never seqs — `cons` already did).
fn step_cons_value(_eval: &mut Evaluator, c: &ConsValue) -> Result<StepUnit, Signal> {
    Ok(StepUnit::One {
        head: c.head().clone(),
        tail: c.tail().clone(),
    })
}

/// Extract one entry projection (keys take the key, vals the val).
/// Only real entries project; 2-vectors fail like the JVM's
/// `Map.Entry` cast (Oracle FX2 `keys-of-vec` CCE).
fn entry_of(item: &Value, keys: bool) -> Result<Value, NativeError> {
    match item {
        Value::MapEntry { key, val, .. } => {
            if keys {
                Ok((**key).clone())
            } else {
                Ok((**val).clone())
            }
        }
        _ => Err(crate::core::cce_err(item, "java.util.Map$Entry")),
    }
}

/// Step an entry view: extract the head projection (casting, like
/// `KeySeq.first`) plus the rewrapped underlying tail.
fn step_keys_unit(
    eval: &mut Evaluator,
    underlying: &Value,
    keys: bool,
) -> Result<StepUnit, Signal> {
    let Some((head, tail)) = step_head(eval, underlying)? else {
        return Ok(StepUnit::Empty);
    };
    let head = entry_of(&head, keys).map_err(|e| eval.throw(e))?;
    // Provably-spent underlyings nil-pun (like `KeySeq.create`).
    let tail = if keys_empty(&tail) {
        Value::Nil
    } else if keys {
        Value::Extension(Arc::new(KeySeqValue::new(tail)))
    } else {
        Value::Extension(Arc::new(ValSeqValue::new(tail)))
    };
    Ok(StepUnit::One { head, tail })
}

/// An entry-view head (nil for spent underlyings; casting, like
/// `KeySeq.first`).
fn keys_first(eval: &mut Evaluator, underlying: &Value, keys: bool) -> Result<Value, Signal> {
    match step_head(eval, underlying)? {
        None => Ok(Value::Nil),
        Some((head, _)) => entry_of(&head, keys).map_err(|e| eval.throw(e)),
    }
}

/// An entry-view rest (advancing never casts, like
/// `KeySeq.next`; spent underlyings rest to `()`).
fn keys_rest(eval: &mut Evaluator, underlying: &Value, keys: bool) -> Result<Value, Signal> {
    // Step chunks are never empty, so one step decides (no loop).
    let tail = match step_unit(eval, underlying)? {
        StepUnit::Empty => None,
        StepUnit::One { tail, .. } => Some(tail),
        StepUnit::Chunk { items, tail } => {
            if items.len() > 1 {
                Some(chain_tail(&items[1..], &tail))
            } else {
                // Single-item chunk.
                Some(tail)
            }
        }
    };
    match tail {
        None => Ok(empty_list_value()),
        Some(t) if keys_empty(&t) => Ok(empty_list_value()),
        Some(t) if keys => Ok(Value::Extension(Arc::new(KeySeqValue::new(t)))),
        Some(t) => Ok(Value::Extension(Arc::new(ValSeqValue::new(t)))),
    }
}

/// A realized seq value from items (nil-punning when empty),
/// with explicit JVM interfaces and class.
pub(crate) fn realized_value(
    items: Vec<Value>,
    chunked: bool,
    counted: bool,
    idrop: bool,
    kind: SeqKind,
) -> Value {
    match step_realized(&items, chunked) {
        Some(s) => Value::Extension(Arc::new(RealizedSeq::flagged(s, counted, idrop, kind))),
        None => Value::Nil,
    }
}

/// Run one thunk step: empty, or a chunk (at most [`CHUNK_SIZE`]
/// items) plus its tail value.
fn step_thunk(eval: &mut Evaluator, thunk: Thunk) -> Result<Forced, Signal> {
    match thunk {
        Thunk::Map { f, inputs } => step_map(eval, &f, inputs),
        Thunk::Filter { f, input } => step_filter(eval, &f, input, false),
        Thunk::Remove { f, input } => step_filter(eval, &f, input, true),
        Thunk::Take { n, input } => step_take(eval, &n, input),
        Thunk::Concat {
            cur,
            rest,
            idx,
            outer,
        } => step_concat(eval, cur, rest, idx, outer),
        Thunk::MapCat { outer } => step_mapcat(eval, outer),
        Thunk::Chain { head, tail } => step_chain(eval, head, tail),
        Thunk::Thunk0 { body, env } => step_thunk0(eval, &body, &env),
        Thunk::Call0 { f } => step_call0(eval, &f),
        Thunk::Repeatedly { f } => step_repeatedly(eval, &f),
        Thunk::Interleave { queue, inputs } => step_interleave(eval, queue, inputs),
        Thunk::Interpose { sep, input, first } => step_interpose(eval, sep, input, first),
        Thunk::TakeWhile { f, input } => step_take_while(eval, &f, input),
        other => step_thunk_mid(eval, other),
    }
}

/// Drop, partition, and keep thunks. Chained so each match stays
/// under the complexity gate.
fn step_thunk_mid(eval: &mut Evaluator, thunk: Thunk) -> Result<Forced, Signal> {
    match thunk {
        Thunk::DropWhile { f, input } => step_drop_while(eval, &f, input),
        Thunk::Drop { n, input } => step_drop(eval, &n, input),
        Thunk::TakeNth { n, input } => step_take_nth(eval, &n, input),
        Thunk::Partition {
            n,
            step,
            pad,
            input,
        } => step_partition_collect(eval, &n, &step, pad, input),
        Thunk::PartitionAll { n, step, input } => {
            step_partition_all_collect(eval, &n, &step, input)
        }
        Thunk::PartitionBy { f, input } => step_partition_by(eval, &f, input),
        Thunk::PartitionByAdvance { f, run, s } => step_partition_by_advance(eval, &f, &run, s),
        Thunk::PartitionRun { f, fv, input } => step_partition_run(eval, &f, &fv, input),
        Thunk::Distinct { seen, input } => step_distinct(eval, seen, input),
        Thunk::Dedupe { last, input } => step_dedupe(eval, last, input),
        Thunk::Keep { f, input, skip } => step_keep(eval, &f, input, skip),
        Thunk::KeepIndexed { f, input, idx } => step_keep_indexed(eval, &f, input, idx),
        other => step_thunk_rest(eval, other),
    }
}

/// Indexed and reduction thunks.
fn step_thunk_rest(eval: &mut Evaluator, thunk: Thunk) -> Result<Forced, Signal> {
    match thunk {
        Thunk::MapIndexed { f, input, idx } => step_map_indexed(eval, &f, input, idx),
        Thunk::Reductions {
            f,
            acc,
            has_init,
            input,
            started,
        } => step_reductions(eval, &f, acc, has_init, input, started),
        Thunk::Replace { smap, input } => step_replace(eval, &smap, input),
        Thunk::Sequence { step, srcs } => step_sequence(eval, &step, srcs),
        Thunk::JdkPull { cursor, chunked } => step_jdk_pull(eval, &cursor, chunked),
        other => step_thunk_s12(eval, other),
    }
}

/// S12 thunks (`seque` drains, `iteration` cells, `file-seq`
/// walks). Chained so each match stays under the complexity gate.
fn step_thunk_s12(eval: &mut Evaluator, thunk: Thunk) -> Result<Forced, Signal> {
    match thunk {
        Thunk::Seque { rx } => Ok(step_seque(&rx)),
        Thunk::Iteration {
            step,
            somef,
            vf,
            kf,
            k,
        } => step_iteration_cell(eval, &step, &somef, &vf, &kf, &k),
        Thunk::FileSeq { stack } => Ok(step_fileseq(&stack)),
        Thunk::ChunkHead { chunk, tail } => Ok(step_chunk_head(&chunk, &tail)),
        // All other variants are handled by
        // `step_thunk`/`step_thunk_mid`/`step_thunk_rest` (the
        // single caller partitions the variants).
        _ => unreachable!("thunk partition"),
    }
}

/// Step a `seque` drain (S12): one blocking take per step (the
/// consumer pulls exactly what it consumes, like the JVM's
/// per-element `.take`); the eos sentinel or a dropped feeder
/// ends the seq.
fn step_seque(rx: &crate::collmisc::SequeRx) -> Forced {
    let item = rx.lock().recv().ok().flatten();
    match item {
        Some(v) => Forced::Chunk {
            items: Arc::from(vec![v]),
            tail: pending_value(Thunk::Seque { rx: rx.clone() }),
        },
        None => Forced::Empty,
    }
}

/// Step one `iteration` cell (S12): step once at `k`, emit
/// through `:vf` when `:somef` passes, and continue at `:kf`
/// unless it is nil.
fn step_iteration_cell(
    eval: &mut Evaluator,
    step: &Value,
    somef: &Value,
    vf: &Value,
    kf: &Value,
    k: &Value,
) -> Result<Forced, Signal> {
    match crate::collmisc::iteration_advance(eval, step, somef, vf, kf, k)? {
        Some((v, tail)) => Ok(Forced::Chunk {
            items: Arc::from(vec![v]),
            tail,
        }),
        None => Ok(Forced::Empty),
    }
}

/// Step a `chunk-cons` head (S12): the chunk's appended items
/// plus its tail (never empty — construction collapses).
fn step_chunk_head(chunk: &crate::collmisc::ChunkValue, tail: &Value) -> Forced {
    Forced::Chunk {
        items: chunk.items_arc(),
        tail: tail.clone(),
    }
}

/// Step a `file-seq` walk (S12): pop the next path, emitting it
/// and pushing its children when it is a directory (unreadable
/// directories read as leaves, like the JVM's null `listFiles`).
fn step_fileseq(stack: &[String]) -> Forced {
    let mut stack = stack.to_vec();
    let Some(path) = stack.pop() else {
        return Forced::Empty;
    };
    if std::fs::metadata(&path).is_ok_and(|m| m.is_dir()) {
        // Unreadable directories read as leaves (like the JVM's
        // null `listFiles`): `read_dir` failure pushes nothing
        // (the iterator chain keeps this branchless — an `if let`
        // `Err` arm is untestable without chmod).
        for entry in std::fs::read_dir(&path).into_iter().flatten().flatten() {
            stack.push(entry.path().to_string_lossy().into_owned());
        }
    }
    Forced::Chunk {
        items: Arc::from(vec![Value::Str(Arc::from(path.as_str()))]),
        tail: pending_value(Thunk::FileSeq { stack }),
    }
}

/// S17: one pull through a JDK-handle cursor (a chunk when
/// `chunked`, one item otherwise); the tail re-pulls through
/// the same shared cursor.
fn step_jdk_pull(
    eval: &mut Evaluator,
    cursor: &JdkCursor,
    chunked: bool,
) -> Result<Forced, Signal> {
    let mut cursor = cursor.clone();
    let n = if chunked { CHUNK_SIZE } else { 1 };
    let items = crate::jdkio::pull_cursor(eval, &mut cursor, n).map_err(|e| eval.throw(e))?;
    if items.is_empty() {
        return Ok(Forced::Empty);
    }
    Ok(Forced::Chunk {
        items: Arc::from(items),
        tail: pending_value(Thunk::JdkPull { cursor, chunked }),
    })
}

/// A prebuilt chain: an empty head forces the tail, otherwise the
/// head is the whole chunk.
fn step_chain(eval: &mut Evaluator, head: Vec<Value>, tail: Value) -> Result<Forced, Signal> {
    if head.is_empty() {
        step_value_forced(eval, &tail)
    } else {
        Ok(Forced::Chunk {
            items: Arc::from(head),
            tail,
        })
    }
}

/// Step a value into a forced chunk (for chain tails).
fn step_value_forced(eval: &mut Evaluator, v: &Value) -> Result<Forced, Signal> {
    match step_unit(eval, v)? {
        StepUnit::Empty => Ok(Forced::Empty),
        StepUnit::One { head, tail } => Ok(Forced::Chunk {
            items: Arc::from(vec![head]),
            tail,
        }),
        StepUnit::Chunk { items, tail } => Ok(Forced::Chunk { items, tail }),
    }
}

/// Step one head plus its remainder (chunk remainders re-chain
/// ahead of the tail; step chunks are never empty).
fn step_head(eval: &mut Evaluator, input: &Value) -> Result<Option<(Value, Value)>, Signal> {
    // Step chunks are never empty, so one step decides (no loop).
    match step_unit(eval, input)? {
        StepUnit::Empty => Ok(None),
        StepUnit::One { head, tail } => Ok(Some((head, tail))),
        StepUnit::Chunk { items, tail } => {
            let (head, rest) = items.split_first().expect("step chunks are non-empty");
            Ok(Some((head.clone(), chain_tail(rest, &tail))))
        }
    }
}

/// Whether `drop`/`nthrest` take the eager `IDrop` path: vectors,
/// small maps, `LongRange`, `Repeat`, and the `IDrop` realized
/// seqs (`ChunkedSeq`, `StringSeq`, small-map seqs).
pub(crate) fn is_idrop(v: &Value) -> bool {
    match v {
        Value::Vector(_) => true,
        Value::Map(m) => !m.is_hashy(),
        Value::Extension(_) => {
            if let Some(r) = as_range(v) {
                return r.is_long();
            }
            if as_repeat(v).is_some() {
                return true;
            }
            if let Some(s) = as_realized(v) {
                return matches!(s.kind(), SeqKind::Chunked | SeqKind::Str | SeqKind::Map);
            }
            false
        }
        _ => false,
    }
}

/// Eager `IDrop` shift of a vector (`None` past the end).
fn idrop_shift_vector(x: &Vector, n: i32, n64: i64) -> Option<Value> {
    if n64 < 0 || n64 >= x.len() as i64 {
        return None;
    }
    let rest: Vec<Value> = x.iter().skip(n as usize).collect();
    Some(realized_value(rest, true, true, true, SeqKind::Chunked))
}

/// Eager `IDrop` shift of a small map (`None` past the end).
fn idrop_shift_map(m: &Map, n: i32, n64: i64) -> Option<Value> {
    let entries: Vec<Value> = m
        .iter()
        .map(|(k, val)| clj_value::map_entry(k.clone(), val.clone()))
        .collect();
    if n64 < 0 || n64 >= entries.len() as i64 {
        return None;
    }
    Some(realized_value(
        entries[n as usize..].to_vec(),
        false,
        true,
        true,
        SeqKind::Map,
    ))
}

/// Eager `IDrop` shift of a long range (`None` past the end).
fn idrop_shift_range(r: &RangeValue, n64: i64) -> Option<Value> {
    // Callers pre-check `is_idrop` (long-only for ranges), and long
    // ranges always carry `Int` bounds, so only the past-end `_`
    // arm is live.
    let count = r.count().unwrap_or(0);
    match (r.cur(), r.step(), n64 >= count) {
        (RangeNum::Int(start), RangeNum::Int(step), false) => {
            let shifted = RangeValue::new(
                RangeNum::Int(start.wrapping_add(step.wrapping_mul(n64))),
                r.end(),
                RangeNum::Int(step),
                true,
                Some(count - n64),
            );
            Some(Value::Extension(Arc::new(shifted)))
        }
        _ => None,
    }
}

/// Eager `IDrop` shift of a repeat (infinite drops return `this`,
/// metadata kept; `None` past the end).
fn idrop_shift_repeat(v: &Value, r: &RepeatValue, n64: i64) -> Option<Value> {
    match r.n() {
        None => Some(v.clone()),
        Some(c) if c - n64 > 0 => Some(Value::Extension(Arc::new(RepeatValue::new(
            r.x().clone(),
            Some(c - n64),
        )))),
        _ => None,
    }
}

/// Eager `IDrop` shift of a chunked/string/small-map realized seq
/// (`None` past the end).
fn idrop_shift_realized(s: &RealizedSeq, n64: i64) -> Option<Value> {
    // Callers pre-check `is_idrop` (kind-checked, and `SeqKind` is
    // immutable on realized seqs).
    let mut cur = s.seq().clone();
    for _ in 0..n64 {
        cur = cur.next()?;
    }
    // `next` answers `None` exactly at the end (`eager`/`chunked`
    // are the only constructed `Seq` shapes, and both yield `None`
    // only at exhaustion), so a past-end drop returns above and
    // `cur` is non-empty here.
    Some(Value::Extension(Arc::new(RealizedSeq::flagged(
        cur,
        s.is_counted(),
        s.is_idrop(),
        s.kind(),
    ))))
}

/// Eager `IDrop` shift of an extension value (`None` past the end
/// or for non-`IDrop` extensions).
fn idrop_shift_extension(v: &Value, n64: i64) -> Option<Value> {
    // Callers only pass `IDrop` extensions (long ranges, repeats,
    // idrop-kind realized seqs).
    if let Some(r) = as_range(v) {
        idrop_shift_range(r, n64)
    } else if let Some(r) = as_repeat(v) {
        idrop_shift_repeat(v, r, n64)
    } else {
        let s = as_realized(v).expect("idrop extensions shift eagerly");
        idrop_shift_realized(s, n64)
    }
}

/// Eager `IDrop` shift by a positive `i32` count (metadata always
/// dropped, like every `.drop`; `None` past the end, like the
/// JVM's null). Callers only pass `IDrop` values.
fn idrop_shift(v: &Value, n: i32) -> Option<Value> {
    let n64 = i64::from(n);
    // Callers only pass `IDrop` values (vectors, small maps, idrop
    // extensions).
    if let Value::Vector(x) = v {
        idrop_shift_vector(x, n, n64)
    } else if let Value::Map(m) = v {
        debug_assert!(!m.is_hashy(), "idrop maps are array maps");
        idrop_shift_map(m, n, n64)
    } else {
        idrop_shift_extension(v, n64)
    }
}

/// `(drop n coll)`: `IDrop` inputs validate `n` (like `pos?`) and
/// drop eagerly — non-positive counts `seq` eagerly, past-end
/// drops answer `()`; everything else stays a lazy `Drop` thunk
/// (counts validated when forced).
pub(crate) fn drop_value(eval: &mut Evaluator, n: &Value, coll: Value) -> Result<Value, Signal> {
    if !is_idrop(&coll) {
        return Ok(drop_thunk(n.clone(), coll));
    }
    let count = crate::core::take_count(n).map_err(|e| eval.throw(e))?;
    if count <= 0 {
        return match seq_value(eval, &coll)? {
            Value::Nil => Ok(empty_list_value()),
            s => Ok(s),
        };
    }
    let n32 = crate::core::idrop_coerce(n).map_err(|e| eval.throw(e))?;
    Ok(idrop_shift(&coll, n32).unwrap_or_else(empty_list_value))
}

/// The `nthrest` count coercion (`(pos? n)` first — eager, before
/// coll dispatch; nil is the `getClass` NPE, non-numbers the
/// `Number` CCE).
fn nthrest_count(eval: &mut Evaluator, n: &Value) -> Result<f64, Signal> {
    match n {
        Value::Int(i) => Ok(*i as f64),
        Value::Double(d) => Ok(*d),
        Value::Nil => Err(eval.throw(crate::core::npe_getclass())),
        _ => Err(eval.throw(crate::core::cce_err(n, "java.lang.Number"))),
    }
}

/// The `nthrest` eager `IDrop` shift (out-of-`i32` fails, like `drop`).
fn nthrest_idrop(eval: &mut Evaluator, coll: &Value, n: &Value) -> Result<Value, Signal> {
    let n32 = crate::core::idrop_coerce(n).map_err(|e| eval.throw(e))?;
    Ok(idrop_shift(coll, n32).unwrap_or_else(empty_list_value))
}

/// The `nthrest` strict walk (`(dec n)` per step as `f64`: exact for
/// small counts; huge counts on finite colls exit via exhaustion, on
/// infinite colls hang — both like the JVM loop), plus the `(or … ())`
/// empty-to-`()` mapping.
fn nthrest_walk(eval: &mut Evaluator, coll: Value, mut cur: f64) -> Result<Value, Signal> {
    let mut xs = coll;
    loop {
        if cur <= 0.0 || cur.is_nan() {
            break;
        }
        let s = seq_value(eval, &xs)?;
        if matches!(s, Value::Nil) {
            xs = Value::Nil;
            break;
        }
        cur -= 1.0;
        xs = rest_value(eval, &s)?;
    }
    match seq_value(eval, &xs)? {
        Value::Nil => Ok(empty_list_value()),
        s => Ok(s),
    }
}

/// `nthrest`: `(pos? n)` first — eager, before coll dispatch (nil
/// is the `getClass` NPE, non-numbers the `Number` CCE, on vectors
/// AND lists). Non-positive `n` returns the coll ITSELF (never
/// seqed); `IDrop` colls shift eagerly (out-of-`i32` fails, like
/// `drop`); anything else walks the JVM's strict
/// `(loop [n n xs coll] ...)` — huge `n` walks to the end silently
/// (no `AE`/`IAE`), and the `(or … ())` maps empty to `()`.
pub(crate) fn nthrest_value(eval: &mut Evaluator, coll: Value, n: &Value) -> Result<Value, Signal> {
    let cur = nthrest_count(eval, n)?;
    if cur <= 0.0 || cur.is_nan() {
        return Ok(coll);
    }
    if is_idrop(&coll) {
        return nthrest_idrop(eval, &coll, n);
    }
    nthrest_walk(eval, coll, cur)
}

/// `map`: step every input; the first input's unit sets the width
/// (whole chunks for chunked first inputs, one item otherwise);
/// other inputs walk to the same width; the shortest wins.
fn step_map(eval: &mut Evaluator, f: &Value, inputs: Vec<Value>) -> Result<Forced, Signal> {
    if inputs.len() == 1 {
        return step_map_one(eval, f, inputs.into_iter().next().unwrap_or(Value::Nil));
    }
    // Multi-input `map` is one call per step (inputs are never
    // empty: natives enforce min-arity 2 and tails preserve count;
    // the arities `cons`
    // per item; only the single-input arity chunks).
    let mut heads = Vec::with_capacity(inputs.len());
    let mut tails = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let Some((head, tail)) = step_head(eval, input)? else {
            return Ok(Forced::Empty);
        };
        heads.push(head);
        tails.push(tail);
    }
    crate::core::check_ifn(eval, f)?;
    let out = eval.call(f, &heads)?;
    Ok(Forced::Chunk {
        items: Arc::from(vec![out]),
        tail: PendingSeq::pending(Thunk::Map {
            f: f.clone(),
            inputs: tails,
        }),
    })
}

/// Single-input `map`: chunked inputs apply the whole chunk per
/// step, anything else one item.
fn step_map_one(eval: &mut Evaluator, f: &Value, input: Value) -> Result<Forced, Signal> {
    let (items, tail): (Arc<[Value]>, Value) = match step_unit(eval, &input)? {
        StepUnit::Empty => return Ok(Forced::Empty),
        StepUnit::One { head, tail } => (Arc::from([head]), tail),
        StepUnit::Chunk { items, tail } => (items, tail),
    };
    crate::core::check_ifn(eval, f)?;
    let mut out = Vec::with_capacity(items.len());
    // The fn resolves once per chunk (#226): a VM fn takes each item
    // straight into its register, a native runs its entry.
    match crate::vm::Prepared::of(f, 1) {
        Some(p) => {
            for item in items.iter() {
                out.push(p.call1(eval, crate::vm::clone_value(item))?);
            }
        }
        None => {
            for item in items.iter() {
                out.push(eval.call(f, std::slice::from_ref(item))?);
            }
        }
    }
    Ok(Forced::Chunk {
        items: Arc::from(out),
        tail: PendingSeq::pending(Thunk::Map {
            f: f.clone(),
            inputs: vec![tail],
        }),
    })
}

/// Re-chain an unconsumed remainder ahead of its tail.
fn chain_tail(remainder: &[Value], tail: &Value) -> Value {
    if remainder.is_empty() {
        tail.clone()
    } else {
        PendingSeq::pending(Thunk::Chain {
            head: remainder.to_vec(),
            tail: tail.clone(),
        })
    }
}

/// `filter`/`remove`: consume whole units, testing each item in
/// order, until a non-empty chunk (or the end).
fn step_filter(
    eval: &mut Evaluator,
    f: &Value,
    mut input: Value,
    invert: bool,
) -> Result<Forced, Signal> {
    loop {
        let (items, tail) = match step_unit(eval, &input)? {
            StepUnit::Empty => return Ok(Forced::Empty),
            StepUnit::One { head, tail } => (vec![head], tail),
            StepUnit::Chunk { items, tail } => (items.to_vec(), tail),
        };
        crate::core::check_ifn(eval, f)?;
        let mut kept = Vec::new();
        for item in &items {
            let pass = eval.call(f, std::slice::from_ref(item))?.is_truthy() != invert;
            if pass {
                kept.push(item.clone());
            }
        }
        if kept.is_empty() {
            input = tail;
        } else {
            let next = if invert {
                Thunk::Remove {
                    f: f.clone(),
                    input: tail,
                }
            } else {
                Thunk::Filter {
                    f: f.clone(),
                    input: tail,
                }
            };
            return Ok(Forced::Chunk {
                items: Arc::from(kept),
                tail: PendingSeq::pending(next),
            });
        }
    }
}

/// `take`: one step consumes the head of the current unit.
fn step_take(eval: &mut Evaluator, n: &Value, input: Value) -> Result<Forced, Signal> {
    // `(pos? n)` per step (the count re-validates every force);
    // one item per step (the body `cons`es).
    let n = crate::core::take_count(n).map_err(|e| eval.throw(e))?;
    if n <= 0 {
        return Ok(Forced::Empty);
    }
    let Some((head, tail)) = step_head(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    Ok(Forced::Chunk {
        items: Arc::from(vec![head]),
        tail: PendingSeq::pending(Thunk::Take {
            n: Value::Int(n - 1),
            input: tail,
        }),
    })
}

/// The content of an unforced `Concat` cell, if `v` is one:
/// cloned out (O(1): one `Value` plus `Arc`s) so `step_concat`
/// can descend without disturbing the cell — an aliased inner
/// still forces normally later, yielding identical items. Forced
/// cells (memoized chunks) and non-concat pendings answer `None`
/// and step through the normal path.
fn unforced_concat_parts(v: &Value) -> Option<(Value, Arc<Vec<Value>>, usize, ConcatStack)> {
    let cell = as_pending(v)?;
    let state = cell.state.lock().unwrap_or_else(|e| e.into_inner());
    match &*state {
        SeqState::Pending(thunk) => match &**thunk {
            Thunk::Concat {
                cur,
                rest,
                idx,
                outer,
            } => Some((cur.clone(), Arc::clone(rest), *idx, outer.clone())),
            _ => None,
        },
        _ => None,
    }
}

/// Push `inner`'s resume frames above `outer` (bottom-first, so
/// the inner top stays on top). Fresh tails carry no stack; only
/// re-chained tails do (shallow in practice, iterative
/// regardless).
fn splice_concat_stack(inner: ConcatStack, mut outer: ConcatStack) -> ConcatStack {
    let mut frames = Vec::new();
    let mut node = inner;
    while let Some(frame) = node {
        frames.push((Arc::clone(&frame.rest), frame.idx));
        node = frame.parent.clone();
    }
    for (rest, idx) in frames.into_iter().rev() {
        outer = Some(Arc::new(ConcatFrame {
            rest,
            idx,
            parent: outer,
        }));
    }
    outer
}

/// Pop one resume frame (O(1): two `Arc` clones).
fn pop_concat_frame(stack: &mut ConcatStack) -> Option<(Arc<Vec<Value>>, usize)> {
    let frame = stack.take()?;
    *stack = frame.parent.clone();
    Some((Arc::clone(&frame.rest), frame.idx))
}

/// `concat`: pass the first non-empty input's unit through,
/// chaining its tail back ahead of the rest. The `rest` vec is
/// shared, never mutated: exhausted inputs advance the index,
/// and the current input's tail travels in `cur` — every step is
/// O(1) no matter how many inputs `(apply concat ...)` splices.
/// Nested unforced concats (`mapcat` tails nest one level per
/// element) flatten through the explicit `outer` stack instead of
/// the Rust stack (#205: the old recursion overflowed on large
/// `for`/`mapcat` inputs).
fn step_concat(
    eval: &mut Evaluator,
    mut cur: Value,
    mut rest: Arc<Vec<Value>>,
    mut idx: usize,
    mut outer: ConcatStack,
) -> Result<Forced, Signal> {
    loop {
        // Descend into nested unforced concats iteratively,
        // stashing each resumed level on `outer`. Spent levels
        // (nothing left past `idx`) push nothing: mapcat tails
        // always advance past their single input, so the hot path
        // keeps the stack empty and stays O(1) per step.
        while let Some((icur, irest, iidx, iouter)) = unforced_concat_parts(&cur) {
            if rest.get(idx).is_some() {
                outer = Some(Arc::new(ConcatFrame {
                    rest,
                    idx,
                    parent: outer,
                }));
            }
            outer = splice_concat_stack(iouter, outer);
            cur = icur;
            rest = irest;
            idx = iidx;
        }
        match step_unit(eval, &cur)? {
            StepUnit::Empty => {
                // Advance past the spent `cur`, unwinding spent
                // levels (a resumed level may itself be spent).
                loop {
                    if let Some(next) = rest.get(idx) {
                        cur = next.clone();
                        idx += 1;
                        break;
                    }
                    match pop_concat_frame(&mut outer) {
                        Some((orest, oidx)) => {
                            rest = orest;
                            idx = oidx;
                        }
                        None => return Ok(Forced::Empty),
                    }
                }
            }
            StepUnit::One { head, tail } => {
                return Ok(Forced::Chunk {
                    items: Arc::from(vec![head]),
                    tail: PendingSeq::pending(Thunk::Concat {
                        cur: tail,
                        rest,
                        idx,
                        outer,
                    }),
                });
            }
            StepUnit::Chunk { items, tail } => {
                return Ok(Forced::Chunk {
                    items,
                    tail: PendingSeq::pending(Thunk::Concat {
                        cur: tail,
                        rest,
                        idx,
                        outer,
                    }),
                });
            }
        }
    }
}

/// Step a `mapcat` outer seq: each outer item is seq'd and
/// spliced (`(mapcat f xs)` ≡ `(apply concat (map f xs))`), so
/// nil/empty items vanish and non-seqables fail exactly like
/// `concat` would. Chunks peel one item and re-chain the rest.
fn step_mapcat(eval: &mut Evaluator, outer: Value) -> Result<Forced, Signal> {
    let mut outer = outer;
    loop {
        // `outer` is always a pending seq (`mapcat_thunk` builds it
        // over the lazy map outer; both re-chains below are pending
        // `Chain`/`MapCat` tails), so forcing answers empty or a
        // chunk — never a `One` unit.
        let pending = as_pending(&outer).expect("mapcat outer is always pending");
        match force_pending(eval, pending)? {
            Forced::Empty => return Ok(Forced::Empty),
            Forced::Chunk { items, tail } => {
                let (out, used) = splice_realized(&items);
                if used > 0 {
                    // The leading realized results splice whole; the
                    // rest of the chunk (if any) continues the outer.
                    let rest = items[used..].to_vec();
                    let next = pending_value(Thunk::Chain { head: rest, tail });
                    if out.is_empty() {
                        outer = next;
                        continue;
                    }
                    return Ok(Forced::Chunk {
                        items: Arc::from(out),
                        tail: pending_value(Thunk::MapCat { outer: next }),
                    });
                }
                let mut it = items.iter();
                let head = it.next().expect("step chunks are non-empty");
                let rest: Vec<Value> = it.cloned().collect();
                let cont = pending_value(Thunk::Chain { head: rest, tail });
                let s = seq_value(eval, head)?;
                if matches!(s, Value::Nil) {
                    outer = cont;
                } else {
                    let cont = pending_value(Thunk::MapCat { outer: cont });
                    return splice_first(eval, &s, cont);
                }
            }
        }
    }
}

/// The leading results of a forced `mapcat` chunk that are already
/// realized (nil, lists, vectors), their items in order, and how
/// many results that took. Splicing them forces nothing (the
/// chunked outer `map` has run `f` over the whole chunk already, as
/// on the JVM), and one output chunk replaces a concat step per item
/// — `for` with `:when` yields a one-item list or nil per element.
fn splice_realized(items: &[Value]) -> (Vec<Value>, usize) {
    let mut out = Vec::new();
    let mut used = 0;
    for item in items {
        match item {
            Value::Nil => {}
            Value::List(l) => out.extend(l.iter().cloned()),
            Value::Vector(v) => out.extend(v.iter()),
            _ => break,
        }
        used += 1;
    }
    (out, used)
}

/// Yield the first unit of a spliced seq, chaining its tail ahead
/// of the mapcat continuation.
fn splice_first(eval: &mut Evaluator, s: &Value, cont: Value) -> Result<Forced, Signal> {
    match step_unit(eval, s)? {
        StepUnit::Empty => step_value_forced(eval, &cont),
        StepUnit::One { head, tail } => Ok(Forced::Chunk {
            items: Arc::from(vec![head]),
            tail: PendingSeq::pending(Thunk::Concat {
                cur: tail,
                rest: Arc::new(vec![cont]),
                idx: 0,
                outer: None,
            }),
        }),
        StepUnit::Chunk { items, tail } => Ok(Forced::Chunk {
            items,
            tail: PendingSeq::pending(Thunk::Concat {
                cur: tail,
                rest: Arc::new(vec![cont]),
                idx: 0,
                outer: None,
            }),
        }),
    }
}

/// Step one unit into an item vector plus its tail (empty inputs
/// end the step).
fn step_items(eval: &mut Evaluator, input: &Value) -> Result<Option<(Vec<Value>, Value)>, Signal> {
    match step_unit(eval, input)? {
        StepUnit::Empty => Ok(None),
        StepUnit::One { head, tail } => Ok(Some((vec![head], tail))),
        StepUnit::Chunk { items, tail } => Ok(Some((items.to_vec(), tail))),
    }
}

/// per realization).
fn step_repeatedly(eval: &mut Evaluator, f: &Value) -> Result<Forced, Signal> {
    // Always infinite here (the `n` arity routes through `take`).
    crate::core::check_ifn(eval, f)?;
    let item = eval.call(f, &[])?;
    Ok(Forced::Chunk {
        items: Arc::from(vec![item]),
        tail: PendingSeq::pending(Thunk::Repeatedly { f: f.clone() }),
    })
}

/// `interleave`: one round (one head per input); any exhausted
/// input ends the whole seq.
fn step_interleave(
    eval: &mut Evaluator,
    mut queue: Vec<Value>,
    inputs: Vec<Value>,
) -> Result<Forced, Signal> {
    // One item per step: drain the queued round first, else step
    // every input together (any exhaustion ends the round).
    if !queue.is_empty() {
        let head = queue.remove(0);
        return Ok(Forced::Chunk {
            items: Arc::from(vec![head]),
            tail: PendingSeq::pending(Thunk::Interleave { queue, inputs }),
        });
    }
    // Inputs are never empty (the native answers `()` for 0 args;
    // tails preserve length).
    let mut heads = Vec::with_capacity(inputs.len());
    let mut next = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let Some((head, tail)) = step_head(eval, input)? else {
            return Ok(Forced::Empty);
        };
        heads.push(head);
        next.push(tail);
    }
    let mut rounds = heads.into_iter();
    let head = rounds.next().expect("non-empty inputs yield a head");
    Ok(Forced::Chunk {
        items: Arc::from(vec![head]),
        tail: PendingSeq::pending(Thunk::Interleave {
            queue: rounds.collect(),
            inputs: next,
        }),
    })
}

/// `interpose`: the head alone first, then one item per step
/// (a separator, then its head — no trailing separator).
fn step_interpose(
    eval: &mut Evaluator,
    sep: Value,
    input: Value,
    first: bool,
) -> Result<Forced, Signal> {
    let Some((head, tail)) = step_head(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    if first {
        return Ok(Forced::Chunk {
            items: Arc::from(vec![head]),
            tail: PendingSeq::pending(Thunk::Interpose {
                sep,
                input: tail,
                first: false,
            }),
        });
    }
    Ok(Forced::Chunk {
        items: Arc::from(vec![sep.clone()]),
        tail: chain_tail(
            std::slice::from_ref(&head),
            &PendingSeq::pending(Thunk::Interpose {
                sep,
                input: tail,
                first: false,
            }),
        ),
    })
}

/// `take-while`: one test per step (the JVM tests per item);
/// failure drops the remainder.
fn step_take_while(eval: &mut Evaluator, f: &Value, input: Value) -> Result<Forced, Signal> {
    let Some((head, tail)) = step_head(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    crate::core::check_ifn(eval, f)?;
    if !eval.call(f, std::slice::from_ref(&head))?.is_truthy() {
        return Ok(Forced::Empty);
    }
    Ok(Forced::Chunk {
        items: Arc::from(vec![head]),
        tail: PendingSeq::pending(Thunk::TakeWhile {
            f: f.clone(),
            input: tail,
        }),
    })
}

/// `drop-while`: skip passing heads (every test runs in the first
/// step, like the JVM's walk), then pass the failing head and
/// its remainder through.
fn step_drop_while(eval: &mut Evaluator, f: &Value, mut input: Value) -> Result<Forced, Signal> {
    loop {
        let Some((head, tail)) = step_head(eval, &input)? else {
            return Ok(Forced::Empty);
        };
        crate::core::check_ifn(eval, f)?;
        if !eval.call(f, std::slice::from_ref(&head))?.is_truthy() {
            return Ok(Forced::Chunk {
                items: Arc::from(vec![head]),
                tail,
            });
        }
        input = tail;
    }
}

/// `drop`: validate the count when forced, skip that many heads,
/// then pass the remainder through.
fn step_drop(eval: &mut Evaluator, n: &Value, mut input: Value) -> Result<Forced, Signal> {
    let mut skip = crate::core::take_count(n)
        .map_err(|e| eval.throw(e))?
        .max(0);
    while skip > 0 {
        let Some((_, tail)) = step_head(eval, &input)? else {
            return Ok(Forced::Empty);
        };
        input = tail;
        skip -= 1;
    }
    step_value_forced(eval, &input)
}

/// `take-nth`: every step seqs, takes the head, and evaluates
/// `(drop n s)` for the tail — exactly the JVM body, so `IDrop`
/// inputs validate eagerly on the first force (non-positive `n`
/// replays the head forever, like the JVM's zero-drop loop).
fn step_take_nth(eval: &mut Evaluator, n: &Value, input: Value) -> Result<Forced, Signal> {
    let s = seq_value(eval, &input)?;
    let Some((head, _)) = step_head(eval, &s)? else {
        return Ok(Forced::Empty);
    };
    let dropped = drop_value(eval, n, s)?;
    Ok(Forced::Chunk {
        items: Arc::from(vec![head]),
        tail: PendingSeq::pending(Thunk::TakeNth {
            n: n.clone(),
            input: dropped,
        }),
    })
}

/// The eager `(nthrest s step)` past a collected window (the
/// `cons` arg evaluates on the SAME force as the window, so the
/// step validates here): non-positive steps replay the window
/// start, overlaps re-chain the window tail, gaps skip ahead.
/// `IDrop` inputs also run the `.drop` count coercion (out of
/// `i32` range fails, like `nthrest` on a vector).
fn partition_next(
    eval: &mut Evaluator,
    window: &[Value],
    rest: Value,
    s_idrop: bool,
    step: &Value,
) -> Result<Value, Signal> {
    let step_count = crate::core::take_count(step).map_err(|e| eval.throw(e))?;
    if step_count <= 0 {
        return Ok(chain_tail(window, &rest));
    }
    if s_idrop {
        // The `.drop` count coercion on the RAW step (huge doubles
        // fail the long range check, like `nthrest`). `take_count`
        // above already rejected non-numbers, so only Int/Double
        // arrive here.
        if let Value::Int(i) = step {
            crate::core::int_exact_long(*i).map_err(|e| eval.throw(e))?;
        }
        if let Value::Double(d) = step {
            crate::core::int_cast_double(d.ceil()).map_err(|e| eval.throw(e))?;
        }
    }
    let step_usize = step_count as usize;
    if step_usize >= window.len() {
        let mut skip = step_usize - window.len();
        let mut tail = rest;
        while skip > 0 {
            let Some((_, rest)) = step_head(eval, &tail)? else {
                break;
            };
            tail = rest;
            skip -= 1;
        }
        Ok(tail)
    } else {
        Ok(chain_tail(&window[step_usize..], &rest))
    }
}

/// A `partition` window: a realized lazy seq (the JVM's `doall`
/// of `take`), never a list.
fn lazy_window(eval: &mut Evaluator, items: Vec<Value>) -> Result<Value, Signal> {
    let pending = PendingSeq::pending(Thunk::Chain {
        head: items,
        tail: Value::Nil,
    });
    let p = as_pending(&pending).expect("fresh pending cell");
    force_pending(eval, p)?;
    Ok(pending)
}

/// Whether a collected window counts as full: the JVM compares
/// `(= n (count p))`, and `Numbers.equal` requires one category —
/// a `Double` width never fills an integral count (oracled: `(=
/// 3.0 3)` is false, so `(partition 3.0 [1 2 3])` answers `()`).
fn window_full(n: &Value, len: usize) -> bool {
    // The JVM compares `(= n (count p))` with `Numbers.equal`
    // category rules (oracled: `(= 3.0 3)` is false, so
    // `(partition 3.0 [1 2 3])` answers `()`); `==` implements
    // exactly those, tower included (`(partition 3N [1 2 3])`
    // fills).
    *n == Value::Int(len as i64)
}

/// `partition`: seq-check, collect one window, and advance
/// eagerly (the `(nthrest s step)` arg evaluates on the SAME
/// force, so the step validates with the first window). Fullness
/// compares the raw `n`, so zero widths emit infinite empties on
/// non-empty input. Short windows pad when a pad was given —
/// `(take n (concat p pad))`, ending the run — and drop
/// otherwise (never validating the step).
fn step_partition_collect(
    eval: &mut Evaluator,
    n: &Value,
    step: &Value,
    pad: Option<Value>,
    input: Value,
) -> Result<Forced, Signal> {
    let Some(s) = seq_opt(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    let s_idrop = is_idrop(&s);
    let width = crate::core::take_count(n)
        .map_err(|e| eval.throw(e))?
        .max(0) as usize;
    let mut window = Vec::with_capacity(width.min(CHUNK_SIZE));
    let mut rest = s;
    while window.len() < width {
        let Some((head, tail)) = step_head(eval, &rest)? else {
            break;
        };
        window.push(head);
        rest = tail;
    }
    if !window_full(n, window.len()) {
        let Some(pad) = pad else {
            return Ok(Forced::Empty);
        };
        let padded = PendingSeq::pending(Thunk::Take {
            n: n.clone(),
            input: PendingSeq::pending(Thunk::Concat {
                cur: lazy_window(eval, window)?,
                rest: Arc::new(vec![pad]),
                idx: 0,
                outer: None,
            }),
        });
        return Ok(Forced::Chunk {
            items: Arc::from(vec![padded]),
            tail: Value::Nil,
        });
    }
    let next = partition_next(eval, &window, rest, s_idrop, step)?;
    Ok(Forced::Chunk {
        items: Arc::from(vec![lazy_window(eval, window.clone())?]),
        tail: PendingSeq::pending(Thunk::Partition {
            n: n.clone(),
            step: step.clone(),
            pad,
            input: next,
        }),
    })
}

/// `partition-all`: collect one window (short windows kept — even
/// empty ones, so non-positive widths emit empties until the
/// input runs out) and advance eagerly on the same force.
fn step_partition_all_collect(
    eval: &mut Evaluator,
    n: &Value,
    step: &Value,
    input: Value,
) -> Result<Forced, Signal> {
    let Some(s) = seq_opt(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    let s_idrop = is_idrop(&s);
    let width = crate::core::take_count(n)
        .map_err(|e| eval.throw(e))?
        .max(0) as usize;
    let mut window = Vec::with_capacity(width.min(CHUNK_SIZE));
    let mut rest = s;
    while window.len() < width {
        let Some((head, tail)) = step_head(eval, &rest)? else {
            break;
        };
        window.push(head);
        rest = tail;
    }
    if window.len() < width {
        return Ok(Forced::Chunk {
            items: Arc::from(vec![lazy_window(eval, window)?]),
            tail: Value::Nil,
        });
    }
    let next = partition_next(eval, &window, rest, s_idrop, step)?;
    Ok(Forced::Chunk {
        items: Arc::from(vec![lazy_window(eval, window.clone())?]),
        tail: PendingSeq::pending(Thunk::PartitionAll {
            n: n.clone(),
            step: step.clone(),
            input: next,
        }),
    })
}

/// `partition-by`: one run per step, exactly the JVM shape: the
/// run is a cons of the head over a lazy take-while tail, and
/// the advance drops the (fully forced) run length. Boundary
/// keys recompute (never cached), matching the JVM call counts.
fn step_partition_by(eval: &mut Evaluator, f: &Value, input: Value) -> Result<Forced, Signal> {
    let Some(s) = seq_opt(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    let Some((fst, next)) = step_head(eval, &s)? else {
        return Ok(Forced::Empty);
    };
    crate::core::check_ifn(eval, f)?;
    let fv = eval.call(f, std::slice::from_ref(&fst))?;
    let run_tail = PendingSeq::pending(Thunk::PartitionRun {
        f: f.clone(),
        fv,
        input: next,
    });
    let run = Value::Extension(Arc::new(ConsValue::new(fst, run_tail)));
    Ok(Forced::Chunk {
        items: Arc::from(vec![run.clone()]),
        tail: PendingSeq::pending(Thunk::PartitionByAdvance {
            f: f.clone(),
            run,
            s,
        }),
    })
}

/// `partition-by` advance: `(drop (count run) s)`, evaluated here
/// (the count forces the run, like the JVM's strict `count`).
fn step_partition_by_advance(
    eval: &mut Evaluator,
    f: &Value,
    run: &Value,
    s: Value,
) -> Result<Forced, Signal> {
    let mut k: i64 = 0;
    let mut rest = run.clone();
    while let Some((_, tail)) = step_head(eval, &rest)? {
        k += 1;
        rest = tail;
    }
    let dropped = drop_value(eval, &Value::Int(k), s)?;
    step_partition_by(eval, f, dropped)
}

/// A `partition-by` run tail: `(take-while #(= fv (f %)))`.
fn step_partition_run(
    eval: &mut Evaluator,
    f: &Value,
    fv: &Value,
    input: Value,
) -> Result<Forced, Signal> {
    let Some((head, tail)) = step_head(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    crate::core::check_ifn(eval, f)?;
    let got = eval.call(f, std::slice::from_ref(&head))?;
    if !equiv_vals(eval, fv, &got)? {
        return Ok(Forced::Empty);
    }
    Ok(Forced::Chunk {
        items: Arc::from(vec![head]),
        tail: PendingSeq::pending(Thunk::PartitionRun {
            f: f.clone(),
            fv: fv.clone(),
            input: tail,
        }),
    })
}

/// `seq` as an option (`None` for empty inputs; unseqable
/// inputs still error).
fn seq_opt(eval: &mut Evaluator, v: &Value) -> Result<Option<Value>, Signal> {
    match seq_value(eval, v)? {
        Value::Nil => Ok(None),
        s => Ok(Some(s)),
    }
}

/// `distinct`: one fresh item per step (the body destructures
/// first/rest, never chunks).
#[allow(clippy::mutable_key_type)]
fn step_distinct(
    eval: &mut Evaluator,
    mut seen: clj_value::MemberSet,
    mut input: Value,
) -> Result<Forced, Signal> {
    loop {
        let Some((head, tail)) = step_head(eval, &input)? else {
            return Ok(Forced::Empty);
        };
        if seen.insert(head.clone()) {
            return Ok(Forced::Chunk {
                items: Arc::from(vec![head]),
                tail: PendingSeq::pending(Thunk::Distinct { seen, input: tail }),
            });
        }
        input = tail;
    }
}

/// `dedupe`: whole units, suppressing runs against the last kept.
fn step_dedupe(
    eval: &mut Evaluator,
    mut last: Option<Value>,
    mut input: Value,
) -> Result<Forced, Signal> {
    loop {
        let Some((items, tail)) = step_items(eval, &input)? else {
            return Ok(Forced::Empty);
        };
        let mut out = Vec::new();
        for item in &items {
            if last.as_ref() != Some(item) {
                out.push(item.clone());
            }
            last = Some(item.clone());
        }
        if out.is_empty() {
            input = tail;
        } else {
            return Ok(Forced::Chunk {
                items: Arc::from(out),
                tail: PendingSeq::pending(Thunk::Dedupe { last, input: tail }),
            });
        }
    }
}

/// T1200 D1 `sequence`: pull lockstep heads through the stepping
/// fn into `CHUNK_SIZE` chunks (the `TransformerIterator` role for
/// `sequence` and eduction seq). Each item steps against a fresh
/// empty vector, so every item of the returned vector is a new
/// output (0 for dropped inputs, 2 for `interpose`, N for
/// `mapcat`). A `Reduced` step stops the walk: vector inners adopt
/// their items (the `take` boundary item), anything else (the
/// `halt-when` sentinel) ends silently.
fn step_sequence(
    eval: &mut Evaluator,
    step: &Value,
    mut srcs: Vec<Value>,
) -> Result<Forced, Signal> {
    let mut out = Vec::new();
    let mut stopped = false;
    while out.len() < CHUNK_SIZE && !stopped {
        let mut call = Vec::with_capacity(srcs.len() + 1);
        call.push(Value::Vector(Vector::from_vec(Vec::new())));
        let mut tails = Vec::with_capacity(srcs.len());
        let mut exhausted = false;
        for src in &srcs {
            match step_head(eval, src)? {
                Some((head, tail)) => {
                    call.push(head);
                    tails.push(tail);
                }
                None => {
                    exhausted = true;
                    break;
                }
            }
        }
        if exhausted {
            break;
        }
        srcs = tails;
        let stepped = eval.call(step, &call)?;
        if let Some(inner) = crate::core::as_reduced(&stepped).cloned() {
            stopped = true;
            if let Value::Vector(v) = &inner {
                out.extend(v.iter());
            }
        } else if let Value::Vector(v) = &stepped {
            out.extend(v.iter());
        }
    }
    if out.is_empty() {
        return Ok(Forced::Empty);
    }
    let tail = if stopped {
        empty_list_value()
    } else {
        PendingSeq::pending(Thunk::Sequence {
            step: step.clone(),
            srcs,
        })
    };
    Ok(Forced::Chunk {
        items: Arc::from(out),
        tail,
    })
}

/// `keep`: whole units, dropping nil results — or, under a `skip`
/// sentinel (a `for` stage), dropping only results identical to it.
fn step_keep(
    eval: &mut Evaluator,
    f: &Value,
    mut input: Value,
    skip: Option<Value>,
) -> Result<Forced, Signal> {
    loop {
        let Some((items, tail)) = step_items(eval, &input)? else {
            return Ok(Forced::Empty);
        };
        crate::core::check_ifn(eval, f)?;
        let prepared = crate::vm::Prepared::of(f, 1);
        let mut kept = Vec::new();
        for item in &items {
            let kept_item = match &prepared {
                Some(p) => p.call1(eval, crate::vm::clone_value(item))?,
                None => eval.call(f, std::slice::from_ref(item))?,
            };
            let dropped = match &skip {
                None => matches!(kept_item, Value::Nil),
                Some(s) => matches!((&kept_item, s),
                    (Value::Extension(a), Value::Extension(b)) if Arc::ptr_eq(a, b)),
            };
            if !dropped {
                kept.push(kept_item);
            }
        }
        if kept.is_empty() {
            input = tail;
        } else {
            return Ok(Forced::Chunk {
                items: Arc::from(kept),
                tail: PendingSeq::pending(Thunk::Keep {
                    f: f.clone(),
                    input: tail,
                    skip,
                }),
            });
        }
    }
}

/// `(__for-keep* f coll skip)`: a `for` stage's `:when` chain —
/// `f` maps each item, answering `skip` (this native itself, by the
/// expansion) for an item the chain rejects (no per-item list to
/// splice, unlike a `mapcat` stage).
fn native_for_keep(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/__for-keep*", args, &[3])?;
    Ok(pending_value(Thunk::Keep {
        f: args[0].clone(),
        input: args[1].clone(),
        skip: Some(args[2].clone()),
    }))
}

/// `keep-indexed`: whole units with running indices, dropping nil
/// results.
fn step_keep_indexed(
    eval: &mut Evaluator,
    f: &Value,
    mut input: Value,
    mut idx: i64,
) -> Result<Forced, Signal> {
    loop {
        let Some((items, tail)) = step_items(eval, &input)? else {
            return Ok(Forced::Empty);
        };
        crate::core::check_ifn(eval, f)?;
        let mut kept = Vec::new();
        for item in &items {
            let kept_item = eval.call(f, &[Value::Int(idx), item.clone()])?;
            idx += 1;
            if !matches!(kept_item, Value::Nil) {
                kept.push(kept_item);
            }
        }
        if kept.is_empty() {
            input = tail;
        } else {
            return Ok(Forced::Chunk {
                items: Arc::from(kept),
                tail: PendingSeq::pending(Thunk::KeepIndexed {
                    f: f.clone(),
                    input: tail,
                    idx,
                }),
            });
        }
    }
}

/// `map-indexed`: whole units with running indices.
fn step_map_indexed(
    eval: &mut Evaluator,
    f: &Value,
    input: Value,
    mut idx: i64,
) -> Result<Forced, Signal> {
    let Some((items, tail)) = step_items(eval, &input)? else {
        return Ok(Forced::Empty);
    };
    crate::core::check_ifn(eval, f)?;
    let mut out = Vec::with_capacity(items.len());
    for item in &items {
        out.push(eval.call(f, &[Value::Int(idx), item.clone()])?);
        idx += 1;
    }
    Ok(Forced::Chunk {
        items: Arc::from(out),
        tail: PendingSeq::pending(Thunk::MapIndexed {
            f: f.clone(),
            input: tail,
            idx,
        }),
    })
}

/// `reductions`: one item per step (unstarted with-init emits the
/// init; unstarted no-init seeds from the head, or calls `(f)` on
/// empty inputs).
fn step_reductions(
    eval: &mut Evaluator,
    f: &Value,
    acc: Option<Value>,
    has_init: bool,
    input: Value,
    started: bool,
) -> Result<Forced, Signal> {
    if !started {
        if has_init {
            // `reductions_thunk` ties `has_init` to `acc.is_some()`.
            let init = acc.clone().expect("has_init carries an acc");
            return Ok(Forced::Chunk {
                items: Arc::from(vec![init]),
                tail: PendingSeq::pending(Thunk::Reductions {
                    f: f.clone(),
                    acc,
                    has_init,
                    input,
                    started: true,
                }),
            });
        }
        match step_head(eval, &input)? {
            None => {
                crate::core::check_ifn(eval, f)?;
                let seed = eval.call(f, &[])?;
                return Ok(Forced::Chunk {
                    items: Arc::from(vec![seed]),
                    tail: Value::Nil,
                });
            }
            Some((head, tail)) => {
                return Ok(Forced::Chunk {
                    items: Arc::from(vec![head.clone()]),
                    tail: PendingSeq::pending(Thunk::Reductions {
                        f: f.clone(),
                        acc: Some(head),
                        has_init: false,
                        input: tail,
                        started: true,
                    }),
                });
            }
        }
    }
    // Started reductions always carry `Some` acc (all tails set or
    // preserve it), so only the spent-input `_` arm is live.
    match (step_head(eval, &input)?, acc) {
        (Some((head, tail)), Some(acc)) => {
            crate::core::check_ifn(eval, f)?;
            let next = eval.call(f, &[acc, head])?;
            Ok(Forced::Chunk {
                items: Arc::from(vec![next.clone()]),
                tail: PendingSeq::pending(Thunk::Reductions {
                    f: f.clone(),
                    acc: Some(next),
                    has_init,
                    input: tail,
                    started: true,
                }),
            })
        }
        _ => Ok(Forced::Empty),
    }
}

/// A `lazy-seq` body: evaluate, seq the result, and share its first
/// unit (pending bodies adopt the inner cell's chunk). An empty
/// `recur` re-invokes the body (the zero-arg `fn` boundary, like the
/// JVM closure); nested loops consume their own first.
fn step_thunk0(eval: &mut Evaluator, body: &[Expr], env: &Env) -> Result<Forced, Signal> {
    loop {
        let mut value = Value::Nil;
        let mut recurred = false;
        for expr in body {
            match eval.eval(expr, env) {
                Ok(v) => value = v,
                Err(Signal::Recur(args)) if args.is_empty() => {
                    recurred = true;
                    break;
                }
                Err(other) => return Err(other),
            }
        }
        if !recurred {
            let forced = seq_value(eval, &value)?;
            return step_value_forced(eval, &forced);
        }
    }
}

/// Run a [`Thunk::Call0`] body, then step its value like
/// [`step_thunk0`].
fn step_call0(eval: &mut Evaluator, f: &Value) -> Result<Forced, Signal> {
    let value = eval.call(f, &[])?;
    let forced = seq_value(eval, &value)?;
    step_value_forced(eval, &forced)
}

/// The empty list value (`()`).
pub(crate) fn empty_list_value() -> Value {
    Value::List(List::new_sync())
}

/// Pure emptiness for entry-view underlyings (nil, empty lists
/// and realized seqs, done ranges, spent repeats — pending and
/// other strict seqs count as non-empty without forcing, like
/// the JVM's `.seq()`, which never forces).
fn keys_empty(u: &Value) -> bool {
    // Underlyings and step tails are always seq'd (Nil/List/Extension).
    if let Value::Nil = u {
        return true;
    }
    if let Value::List(l) = u {
        return l.is_empty();
    }
    keys_empty_extension(u)
}

/// Emptiness for extension underlyings (pending and other strict
/// seqs count as non-empty without forcing).
fn keys_empty_extension(u: &Value) -> bool {
    if let Some(s) = as_realized(u) {
        return s.seq().is_empty();
    }
    if let Some(r) = as_range(u) {
        return r.done();
    }
    if let Some(r) = as_repeat(u) {
        return r.n().is_some_and(|n| n <= 0);
    }
    false
}

/// `seq` of a pending seq (forcing one step; nil-punning).
fn seq_pending(eval: &mut Evaluator, v: &Value, p: &PendingSeq) -> Result<Value, Signal> {
    match force_pending(eval, p)? {
        Forced::Empty => Ok(Value::Nil),
        Forced::Chunk { .. } => Ok(v.clone()),
    }
}

/// `seq` of a realized seq (nil-punning empties).
fn seq_realized(v: &Value, s: &RealizedSeq) -> Value {
    if s.seq().is_empty() {
        Value::Nil
    } else {
        v.clone()
    }
}

/// `seq` of a range (nil-punning done ranges).
fn seq_range(v: &Value, r: &RangeValue) -> Value {
    if r.done() { Value::Nil } else { v.clone() }
}

/// `seq` of a repeat (nil-punning spent repeats).
fn seq_repeat(v: &Value, r: &RepeatValue) -> Value {
    let empty = r.n().is_some_and(|n| n <= 0);
    if empty { Value::Nil } else { v.clone() }
}

/// `seq` of a cycle (nil-punning doubly-nil cycles).
fn seq_cycle(v: &Value, c: &CycleValue) -> Value {
    let empty = matches!(c.prev(), Value::Nil) && matches!(c.current(), Value::Nil);
    if empty { Value::Nil } else { v.clone() }
}

/// `seq` of a subvector view (as a realized subvec seq).
fn seq_subvector(sv: &SubVectorValue) -> Value {
    realized_value(sv.items(), false, true, false, SeqKind::Subvec)
}

/// `seq` of iterate/cons values (never empty).
fn is_always_seq(v: &Value) -> bool {
    as_iterate(v).is_some() || as_cons_value(v).is_some()
}

/// S12: whether `chunked-seq?` answers true: a bounded range, a
/// realized seq over a chunked head, or the emulated chunked cons
/// (`chunk-cons` results and `rest`-of-chunk tails).
pub(crate) fn is_chunked_seq_value(v: &Value) -> bool {
    if let Some(r) = as_range(v) {
        return r.end().is_some();
    }
    if let Some(s) = as_realized(v) {
        return matches!(s.kind(), SeqKind::Chunked | SeqKind::ChunkedCons);
    }
    if let Some(p) = as_pending(v) {
        return p.is_chunked_cons();
    }
    false
}

/// S12: decompose a chunked seq into its head chunk plus the tail
/// after it (`None` tail at the end). `Ok(None)` is not chunked
/// (projectors take the `IChunkedSeq` cast there).
pub(crate) fn chunked_parts(
    eval: &mut Evaluator,
    v: &Value,
) -> Result<Option<(crate::collmisc::ChunkValue, Option<Value>)>, Signal> {
    if let Some(s) = as_realized(v) {
        if !matches!(s.kind(), SeqKind::Chunked | SeqKind::ChunkedCons) {
            return Ok(None);
        }
        return Ok(chunked_realized_parts(s));
    }
    if let Some(r) = as_range(v) {
        if r.end().is_none() {
            return Ok(None);
        }
        return chunked_range_parts(eval, r);
    }
    if let Some(p) = as_pending(v) {
        if !p.is_chunked_cons() {
            return Ok(None);
        }
        return chunked_pending_parts(eval, p);
    }
    Ok(None)
}

/// S12: head chunk + tail of a realized chunked seq (the current
/// 32-block; a non-chunked shape under a chunked kind splits the
/// items instead, so projectors stay total).
fn chunked_realized_parts(s: &RealizedSeq) -> Option<(crate::collmisc::ChunkValue, Option<Value>)> {
    if let Some((items, rest)) = s.seq().chunk_next() {
        let tail = if rest.is_empty() {
            None
        } else {
            Some(Value::Extension(Arc::new(RealizedSeq::flagged(
                rest,
                s.is_counted(),
                s.is_idrop(),
                s.kind(),
            ))))
        };
        return Some((crate::collmisc::head_chunk(items.into_owned()), tail));
    }
    // Defensive: chunked kind over an eager shape (no constructor
    // builds this today) — split the first 32 items off.
    let items = s.items();
    if items.is_empty() {
        return None;
    }
    let head: Vec<Value> = items.iter().take(CHUNK_SIZE).cloned().collect();
    let rest: Vec<Value> = items.iter().skip(CHUNK_SIZE).cloned().collect();
    let tail = step_realized(&rest, true).map(|seq| {
        Value::Extension(Arc::new(RealizedSeq::flagged(
            seq,
            s.is_counted(),
            s.is_idrop(),
            s.kind(),
        )))
    });
    Some((crate::collmisc::head_chunk(head), tail))
}

/// S12: head chunk + tail of a bounded range (up to 32 stepped
/// items; an inexact `BigDec`/ratio step raises, like the walk).
fn chunked_range_parts(
    eval: &mut Evaluator,
    r: &RangeValue,
) -> Result<Option<(crate::collmisc::ChunkValue, Option<Value>)>, Signal> {
    let mut items = Vec::new();
    let mut cur = r.clone();
    let mut tail: Option<Value> = None;
    while items.len() < CHUNK_SIZE {
        let Some((head, next)) = cur.stepped() else {
            break;
        };
        items.push(head);
        match next {
            Some(nxt) => {
                tail = Some(Value::Extension(Arc::new(nxt.clone())));
                cur = nxt;
            }
            None => {
                if cur.step_inexact() {
                    return Err(eval.throw(range_step_error()));
                }
                tail = None;
                break;
            }
        }
    }
    if cur.done() {
        tail = None;
    }
    Ok(Some((crate::collmisc::head_chunk(items), tail)))
}

/// One `chunked_pending_parts` decision, made under the memo lock
/// (forcing happens after the lock drops — never nested).
enum PendingChunk {
    Parts(crate::collmisc::ChunkValue, Value),
    Force,
    Gone,
}

/// S12: the `chunked_pending_parts` decision, made under the
/// memo lock (split for the complexity budget).
fn chunked_pending_decision(p: &PendingSeq) -> PendingChunk {
    let state = p.state.lock().unwrap_or_else(|e| e.into_inner());
    match &*state {
        SeqState::Pending(t) => match &**t {
            Thunk::ChunkHead { chunk, tail } => PendingChunk::Parts(chunk.clone(), tail.clone()),
            Thunk::Chain { head, tail } => {
                PendingChunk::Parts(crate::collmisc::head_chunk(head.clone()), tail.clone())
            }
            _ => PendingChunk::Force,
        },
        SeqState::Chunk { items, pos, tail } => PendingChunk::Parts(
            crate::collmisc::head_chunk(items[*pos..].to_vec()),
            tail.clone(),
        ),
        SeqState::Empty => PendingChunk::Gone,
    }
}

/// S12: the `chunked_pending_parts` force half — force one step
/// after the memo lock drops (split for the complexity budget).
fn chunked_pending_forced(
    eval: &mut Evaluator,
    p: &PendingSeq,
) -> Result<Option<(crate::collmisc::ChunkValue, Option<Value>)>, Signal> {
    match force_pending(eval, p)? {
        Forced::Empty => Ok(None),
        Forced::Chunk { items, tail } => {
            let chunk = crate::collmisc::head_chunk(items.to_vec());
            Ok(Some((chunk, tail_seq_opt(eval, &tail)?)))
        }
    }
}

/// S12: head chunk + tail of an emulated chunked cons (forced
/// state reads without forcing; unforced non-chain thunks force
/// one step).
fn chunked_pending_parts(
    eval: &mut Evaluator,
    p: &PendingSeq,
) -> Result<Option<(crate::collmisc::ChunkValue, Option<Value>)>, Signal> {
    match chunked_pending_decision(p) {
        PendingChunk::Gone => Ok(None),
        PendingChunk::Force => chunked_pending_forced(eval, p),
        PendingChunk::Parts(chunk, tail) => Ok(Some((chunk, tail_seq_opt(eval, &tail)?))),
    }
}

/// S12: a chunk tail as a seq value (`None` when empty — `rest`
/// answers `()` there, `next` answers nil).
fn tail_seq_opt(eval: &mut Evaluator, tail: &Value) -> Result<Option<Value>, Signal> {
    let s = seq_value(eval, tail)?;
    if matches!(s, Value::Nil) {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

/// `seq` of an extension value.
fn seq_value_extension(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    if let Some(p) = as_pending(v) {
        return seq_pending(eval, v, p);
    }
    if let Some(s) = as_realized(v) {
        return Ok(seq_realized(v, s));
    }
    if let Some(r) = as_range(v) {
        return Ok(seq_range(v, r));
    }
    if let Some(r) = as_repeat(v) {
        return Ok(seq_repeat(v, r));
    }
    if let Some(c) = as_cycle(v) {
        return Ok(seq_cycle(v, c));
    }
    if is_always_seq(v) {
        return Ok(v.clone());
    }
    // Entry views are always non-empty: every wrap guards
    // `!keys_empty`, and emptiness is time-stable over immutable
    // underlyings.
    if as_keys(v).is_some() {
        return Ok(v.clone());
    }
    if as_vals(v).is_some() {
        return Ok(v.clone());
    }
    if let Some(sv) = as_subvector(v) {
        return Ok(seq_subvector(sv));
    }
    // T1200 D1: eductions seq through the stepping fn per use (like
    // `sequence`; bad collections take the ISeq error via the walk).
    if let Some((xform, coll)) = crate::xform::Eduction::parts(v) {
        return seq_eduction(eval, xform, coll);
    }
    // S12: iterations seq through one eager step per use (fresh
    // from `:initk` every time, like the JVM's `Seqable`).
    if let Some(it) = crate::collmisc::as_iteration(v) {
        return crate::collmisc::seq_iteration(eval, it);
    }
    // #205: `ArrayList` seqs snapshot the slots (empties answer
    // nil — `realized_value` holds that shape).
    if let Some(items) = crate::emucoll::list_snapshot(v) {
        return Ok(realized_value(
            items,
            false,
            true,
            false,
            clj_value::SeqKind::Array,
        ));
    }
    Err(eval.throw(iseq_err(v)))
}

/// `seq` of an eduction value: the lazy transform over its
/// collection (fresh application per `seq`, like the JVM's
/// `TransformerIterator`).
fn seq_eduction(eval: &mut Evaluator, xform: &Value, coll: &Value) -> Result<Value, Signal> {
    check_seqable(eval, coll)?;
    let step = crate::xform::apply_xf(eval, xform, crate::ns::deref_core(eval, "conj"))
        .map_err(|e| eval.throw(e))?;
    Ok(sequence_thunk(step, vec![coll.clone()]))
}

/// Whether an extension value backs a lazy transform without
/// forcing (split from [`check_seqable`] so both stay within the
/// complexity budget).
fn is_extension_seqable(v: &Value) -> bool {
    as_pending(v).is_some()
        || as_realized(v).is_some()
        || as_range(v).is_some()
        || as_repeat(v).is_some()
        || as_cycle(v).is_some()
        || is_always_seq(v)
        || as_keys(v).is_some()
        || as_vals(v).is_some()
        || as_subvector(v).is_some()
        || crate::xform::Eduction::parts(v).is_some()
        || crate::collmisc::as_iteration(v).is_some()
}

/// Whether a value can back a lazy transform *without forcing it*:
/// the strict seqables plus every lazy/extension shape
/// `seq_value_extension` accepts (the JVM's `RT.iter` validates
/// eagerly, before any realization — oracle `err-seq-nonseq`).
/// Kept next to that dispatch so new arms update both.
pub(crate) fn check_seqable(eval: &mut Evaluator, v: &Value) -> Result<(), Signal> {
    let ok = match v {
        Value::Nil
        | Value::List(_)
        | Value::Vector(_)
        | Value::MapEntry { .. }
        | Value::Map(_)
        | Value::Set(_)
        | Value::Str(_) => true,
        Value::Extension(_) => is_extension_seqable(v),
        _ => false,
    };
    if ok {
        return Ok(());
    }
    Err(eval.throw(iseq_err(v)))
}

/// `seq` of a vector (chunked).
fn seq_vector(x: &Vector) -> Value {
    match Seq::vector_from(x.clone(), 0) {
        None => Value::Nil,
        Some(seq) => Value::Extension(Arc::new(RealizedSeq::flagged(
            seq,
            true,
            true,
            SeqKind::Chunked,
        ))),
    }
}

/// `seq` of a map (small maps keep `Map` kind).
fn seq_map(m: &Map) -> Value {
    let small = !m.is_hashy();
    let kind = if small { SeqKind::Map } else { SeqKind::Key };
    realized_value(
        m.iter()
            .map(|(k, val)| clj_value::map_entry(k.clone(), val.clone()))
            .collect::<Vec<_>>(),
        false,
        small,
        small,
        kind,
    )
}

/// `seq` of a set.
fn seq_set(s: &Set) -> Value {
    realized_value(
        s.iter().cloned().collect::<Vec<_>>(),
        false,
        false,
        false,
        SeqKind::Key,
    )
}

/// `seq` of a string: O(1) over the shared string, like the JVM's
/// `StringSeq` (same flags as the old eager seq).
fn seq_str(s: &Arc<str>) -> Value {
    match Seq::string_from(Arc::clone(s), 0, 0) {
        None => Value::Nil,
        Some(seq) => Value::Extension(Arc::new(RealizedSeq::flagged(
            seq,
            true,
            true,
            SeqKind::Str,
        ))),
    }
}

/// Nil-punning `seq`: nil for empties, the seq value otherwise.
/// Pending seqs force one step; strict seqs decide emptiness from
/// their bounds (never stepping, like the JVM's `.seq()` on
/// `ISeq`s); vectors chunk, maps/sets/strings go eager; lists
/// return directly.
pub(crate) fn seq_value(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    match v {
        Value::Nil => Ok(Value::Nil),
        Value::Extension(_) => seq_value_extension(eval, v),
        Value::List(l) => Ok(if l.is_empty() { Value::Nil } else { v.clone() }),
        Value::Vector(x) => Ok(seq_vector(x)),
        Value::MapEntry { key, val, .. } => {
            let tmp = Vector::from_vec(vec![(**key).clone(), (**val).clone()]);
            Ok(seq_vector(&tmp))
        }
        Value::Map(m) => Ok(seq_map(m)),
        Value::Set(s) => Ok(seq_set(s)),
        Value::Str(s) => Ok(seq_str(s)),
        // Arrays snapshot eagerly (`ArraySeq` from `sort` uses the
        // same flags; D1 pins the snapshot — S15 oracles `a-seq`,
        // `a-seq-empty`, `a-seq-obj`, `a-seq-2d`).
        Value::Array(a) => Ok(realized_value(
            a.snapshot(),
            false,
            true,
            false,
            SeqKind::Array,
        )),
        // Primitive vectors seq exactly like vectors (chunked).
        Value::PrimitiveVector(p) => Ok(realized_value(
            p.items(),
            true,
            true,
            true,
            SeqKind::Chunked,
        )),
        // Records seq their content entries (oracle `rec-seq`);
        // structs likewise (S12 oracle `sm-seq`); anything else
        // host has no seq (oracle `i-seq`).
        Value::HostType(_) => match crate::types::maplike_inst(v) {
            Some(i) => Ok(seq_map(&i.content)),
            None => Err(eval.throw(iseq_err(v))),
        },
        _ => Err(eval.throw(iseq_err(v))),
    }
}

/// The first item of an extension collection (the subvector
/// head, else the `ArrayList` snapshot head).
fn first_direct_extension(v: &Value) -> Option<Value> {
    as_subvector(v)
        .map(|sv| sv.iter().next().unwrap_or(Value::Nil))
        .or_else(|| {
            crate::emucoll::list_snapshot(v)
                .map(|items| items.into_iter().next().unwrap_or(Value::Nil))
        })
}

/// The first item of a directly-readable collection (`None` for
/// anything else, including seq values and unseqables).
fn first_direct(v: &Value) -> Option<Value> {
    match v {
        Value::List(l) => Some(l.first().cloned().unwrap_or(Value::Nil)),
        Value::Vector(x) => Some(x.iter().next().unwrap_or(Value::Nil)),
        Value::MapEntry { key, .. } => Some((**key).clone()),
        Value::Map(m) => Some(
            m.iter()
                .next()
                .map(|(k, val)| clj_value::map_entry(k.clone(), val.clone()))
                .unwrap_or(Value::Nil),
        ),
        Value::Set(s) => Some(s.iter().next().cloned().unwrap_or(Value::Nil)),
        Value::Str(s) => Some(s.chars().next().map(Value::Char).unwrap_or(Value::Nil)),
        Value::Array(a) => Some(if a.is_empty() {
            Value::Nil
        } else {
            crate::arrays::array_read(a, 0)
        }),
        Value::PrimitiveVector(p) => Some(p.get(0).cloned().unwrap_or(Value::Nil)),
        // Records read their first content entry (empty reads nil,
        // like `{}`); structs likewise (S12); other host values
        // keep the `ISeq` error.
        Value::HostType(_) => crate::types::maplike_inst(v).map(|i| {
            i.content
                .iter()
                .next()
                .map(|(k, val)| clj_value::map_entry(k.clone(), val.clone()))
                .unwrap_or(Value::Nil)
        }),
        Value::Extension(_) => first_direct_extension(v),
        _ => None,
    }
}

/// The first item of a seq value (memoized chunks and realized
/// seqs read their head without stepping — no chunk copy per
/// `first`; iterate heads read without stepping; entry views
/// cast; anything else steps one unit).
fn first_seq_value(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    if let Some(p) = as_pending(v) {
        let state = p.state.lock().unwrap_or_else(|e| e.into_inner());
        match &*state {
            SeqState::Chunk { items, pos, .. } if *pos < items.len() => {
                return Ok(items[*pos].clone());
            }
            // An exhausted chunk continues in its tail (like the
            // JVM's `ChunkedCons.more()`; `rest` seqs the tail at
            // the last item, so this only sees crafted states).
            SeqState::Chunk { tail, .. } => {
                let tail = tail.clone();
                drop(state);
                return first_value(eval, &tail);
            }
            _ => {}
        }
    }
    if let Some(r) = as_realized(v) {
        return Ok(r.seq().first().cloned().unwrap_or(Value::Nil));
    }
    if let Some(it) = as_iterate(v) {
        return iterate_seed(eval, it);
    }
    if let Some(k) = as_keys(v) {
        return keys_first(eval, k.underlying(), true);
    }
    if let Some(k) = as_vals(v) {
        return keys_first(eval, k.underlying(), false);
    }
    match step_unit(eval, v)? {
        StepUnit::Empty => Ok(Value::Nil),
        StepUnit::One { head, .. } => Ok(head),
        StepUnit::Chunk { items, .. } => Ok(items.first().cloned().unwrap_or(Value::Nil)),
    }
}

/// The first item (nil for empties/unseqable-nils; direct for
/// collections, stepping only for seq values). Iterate heads
/// read without stepping (never calling `f` for realized seeds);
/// entry views cast.
pub(crate) fn first_value(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    if matches!(v, Value::Nil) {
        return Ok(Value::Nil);
    }
    if let Some(direct) = first_direct(v) {
        return Ok(direct);
    }
    if let Some(head) = range_head(v) {
        return Ok(head);
    }
    if is_seq_value(v) {
        return first_seq_value(eval, v);
    }
    Err(eval.throw(iseq_err(v)))
}

/// An exact range's head is its current bound (`Range.first`), read
/// without stepping a whole chunk (32 steps per `first`). Inexact
/// steps keep the stepped path, which raises.
fn range_head(v: &Value) -> Option<Value> {
    as_range(v)
        .filter(|r| !r.step_inexact() && !r.done())
        .map(|r| r.cur().value())
}

/// A range shifted past its head (`()` when spent; metadata
/// dropped, like `Range.next`/`LongRange.next`). A `BigDec`/ratio
/// step with no exact decimal raises (oracle `range-mixed`).
fn range_rest(eval: &mut Evaluator, r: &RangeValue) -> Result<Value, Signal> {
    // Callers seq first (`seq_range` nil-puns done ranges), and
    // ranges are immutable, so the range is never done here.
    if r.step_inexact() {
        return Err(eval.throw(range_step_error()));
    }
    match r.advance() {
        None => Ok(empty_list_value()),
        Some(nxt) => {
            let shifted = RangeValue::new(
                nxt,
                r.end(),
                r.step(),
                r.is_long(),
                r.count().map(|c| c - 1),
            );
            if shifted.done() {
                Ok(empty_list_value())
            } else {
                Ok(Value::Extension(Arc::new(shifted)))
            }
        }
    }
}

/// The JVM `ArithmeticException` for a range step that adds
/// `BigDec` to a ratio with no exact decimal (oracle
/// `range-mixed`).
fn range_step_error() -> NativeError {
    NativeError::new(
        ThrowClass::ArithmeticException,
        clj_value::NON_TERMINATING.to_owned(),
    )
}

/// `rest` past a chunk's last item: seq the tail (nil-punning
/// to `()`).
fn rest_chunk_tail(eval: &mut Evaluator, tail: &Value) -> Result<Value, Signal> {
    match seq_value(eval, tail)? {
        Value::Nil => Ok(empty_list_value()),
        t => Ok(t),
    }
}

/// `rest` of a forced pending chunk: multi-item chunks rest to a
/// `ChunkedCons` (like the JVM's `more`); empty chunks fall through
/// to the tail; single items seq the tail (nil-punning to `()`).
fn rest_pending_chunk(
    eval: &mut Evaluator,
    items: &Arc<[Value]>,
    tail: Value,
) -> Result<Value, Signal> {
    if items.len() > 1 {
        return Ok(PendingSeq::pending_kind(
            Thunk::Chain {
                head: items[1..].to_vec(),
                tail,
            },
            LazyKind::ChunkedCons,
        ));
    }
    // Single-item chunk (forced chunks are never empty).
    rest_chunk_tail(eval, &tail)
}

/// `rest` of a pending seq (a memoized chunk rests by advancing
/// its cursor — a new cell sharing the items, like the JVM's
/// `ChunkedCons`, with no per-element chunk copy; anything else
/// forces one step).
fn rest_pending(eval: &mut Evaluator, p: &PendingSeq) -> Result<Value, Signal> {
    {
        let state = p.state.lock().unwrap_or_else(|e| e.into_inner());
        if let SeqState::Chunk { items, pos, tail } = &*state {
            if *pos + 1 < items.len() {
                // `ChunkedCons` class, like the forcing path below
                // (JVM `more()`; pinned by `lazy_chunked_cons_*`).
                return Ok(Value::Extension(Arc::new(PendingSeq {
                    state: Mutex::new(SeqState::Chunk {
                        items: Arc::clone(items),
                        pos: *pos + 1,
                        tail: tail.clone(),
                    }),
                    meta: p.meta.clone(),
                    kind: LazyKind::ChunkedCons,
                })));
            }
            let tail = tail.clone();
            drop(state);
            return rest_chunk_tail(eval, &tail);
        }
    }
    match force_pending(eval, p)? {
        Forced::Empty => Ok(empty_list_value()),
        Forced::Chunk { items, tail } => rest_pending_chunk(eval, &items, tail),
    }
}

/// `rest` of a realized seq (shifting, preserving kind; `StringSeq`
/// keeps metadata).
fn rest_realized_seq(r: &RealizedSeq) -> Value {
    let more = r.seq().more();
    if more.is_empty() {
        return empty_list_value();
    }
    Value::Extension(Arc::new(seq_tail(
        more,
        r.is_counted(),
        r.is_idrop(),
        r.kind(),
        r.meta(),
    )))
}

/// Map a stepped tail to its `rest` (nil/empty tails answer `()`).
fn rest_tail_value(tail: Value) -> Value {
    // Nil tails (cons over empty colls seqs to Nil).
    if matches!(tail, Value::Nil) {
        return empty_list_value();
    }
    if matches!(&tail, Value::List(l) if l.is_empty()) {
        return empty_list_value();
    }
    tail
}

/// `rest` of a one-step strict seq (stepping once, then mapping the
/// tail).
fn rest_stepped(eval: &mut Evaluator, s: &Value) -> Result<Value, Signal> {
    match step_unit(eval, s)? {
        StepUnit::Empty => Ok(empty_list_value()),
        StepUnit::One { tail, .. } | StepUnit::Chunk { tail, .. } => Ok(rest_tail_value(tail)),
    }
}

/// `rest` of a seqed extension value.
fn rest_extension(eval: &mut Evaluator, s: &Value) -> Result<Value, Signal> {
    // No pending arm: `rest_value` diverts pendings before seqing,
    // and `seq` of anything else never yields a pending cell.
    if let Some(r) = as_realized(s) {
        return Ok(rest_realized_seq(r));
    }
    if let Some(r) = as_range(s) {
        return range_rest(eval, r);
    }
    if let Some(k) = as_keys(s) {
        return keys_rest(eval, k.underlying(), true);
    }
    if let Some(k) = as_vals(s) {
        return keys_rest(eval, k.underlying(), false);
    }
    rest_stepped(eval, s)
}

/// `rest` of a seqed value (never nil): lists drop their head,
/// realized seqs shift (preserving kind, `StringSeq` keeping
/// metadata), ranges shift, one-step strict seqs step, entry
/// views advance without casting, and forced multi-item chunks
/// rest to a `ChunkedCons` (like the JVM's `more`).
fn rest_of_seq(eval: &mut Evaluator, s: &Value) -> Result<Value, Signal> {
    // Inputs are always seq'd (`rest_value` seqs first).
    if let Value::Nil = s {
        return Ok(empty_list_value());
    }
    if let Value::List(l) = s {
        return Ok(l
            .drop_first()
            .map(|t| Value::List(t.with_meta(None)))
            .unwrap_or_else(empty_list_value));
    }
    rest_extension(eval, s)
}

/// A fresh realized seq of `seq` (counted, `IDrop`), as `seq` of a
/// vector or string builds it.
fn realized_of(seq: Seq, kind: SeqKind) -> Value {
    Value::Extension(Arc::new(RealizedSeq::flagged(seq, true, true, kind)))
}

/// `first` without the evaluator (#226 VM op): the native's answer
/// for nil, lists, vectors, strings and realized seqs (`None` runs
/// the native).
pub(crate) fn first_fast(v: &Value) -> Option<Value> {
    match v {
        Value::Nil => Some(Value::Nil),
        Value::List(l) => Some(l.first().cloned().unwrap_or(Value::Nil)),
        Value::Vector(x) => Some(x.get(0).unwrap_or(Value::Nil)),
        Value::Str(s) => Some(s.chars().next().map_or(Value::Nil, Value::Char)),
        _ => as_realized(v).map(|r| r.seq().first().cloned().unwrap_or(Value::Nil)),
    }
}

/// `empty?` without the evaluator (#226 VM op): the native's answer
/// for nil, the counted collections, strings and realized seqs — a
/// realized seq answers from its head, never by counting (`None`
/// runs the native).
pub(crate) fn empty_fast(v: &Value) -> Option<bool> {
    match v {
        Value::Nil => Some(true),
        Value::List(l) => Some(l.is_empty()),
        Value::Vector(x) => Some(x.is_empty()),
        Value::Map(m) => Some(m.is_empty()),
        Value::Set(s) => Some(s.is_empty()),
        Value::Str(s) => Some(s.is_empty()),
        _ => as_realized(v).map(|r| r.seq().is_empty()),
    }
}

/// `seq` without the evaluator (#226 VM op): nil, lists, vectors,
/// strings and realized seqs (`None` runs the native).
pub(crate) fn seq_fast(v: &Value) -> Option<Value> {
    match v {
        Value::Nil => Some(Value::Nil),
        Value::List(l) => Some(if l.is_empty() { Value::Nil } else { v.clone() }),
        Value::Vector(x) => Some(seq_vector(x)),
        Value::Str(s) => Some(seq_str(s)),
        _ => as_realized(v).map(|r| seq_realized(v, r)),
    }
}

/// `next` (`rest` when `rest`) without the evaluator, over an owned
/// operand (#226 VM op; `None` leaves it for the native): a realized
/// seq held once steps in place, with no new seq value; vectors and
/// strings seq from their second item without a copy.
pub(crate) fn next_owned(coll: &mut Value, rest: bool) -> Option<Value> {
    let end = || if rest { empty_list_value() } else { Value::Nil };
    match std::mem::replace(coll, Value::Nil) {
        Value::Nil => Some(end()),
        Value::List(l) => {
            let tail = l
                .drop_first()
                .map(|t| Value::List(t.with_meta(None)))
                .unwrap_or_else(empty_list_value);
            Some(nil_punned(tail, rest))
        }
        Value::Vector(v) => {
            Some(Seq::vector_from(v, 1).map_or_else(end, |s| realized_of(s, SeqKind::Chunked)))
        }
        Value::Str(s) => {
            let step = s.chars().next().map_or(0, char::len_utf8);
            Some(Seq::string_from(s, step, 1).map_or_else(end, |s| realized_of(s, SeqKind::Str)))
        }
        Value::Extension(e) => next_extension(coll, e, rest),
        other => {
            *coll = other;
            None
        }
    }
}

/// [`next_owned`] of an extension value: a realized seq held once
/// steps in place; a shared one gets a fresh tail; anything else goes
/// back in `coll` for the native (`None`).
fn next_extension(
    coll: &mut Value,
    mut e: Arc<dyn clj_value::ExtensionValue>,
    rest: bool,
) -> Option<Value> {
    let stepped = Arc::get_mut(&mut e)
        .and_then(|x| x.as_any_mut())
        .and_then(|a| a.downcast_mut::<RealizedSeq>())
        .map(RealizedSeq::advance);
    match stepped {
        Some(true) => Some(Value::Extension(e)),
        Some(false) => Some(if rest { empty_list_value() } else { Value::Nil }),
        None => {
            *coll = Value::Extension(e);
            as_realized(coll).map(|r| nil_punned(rest_realized_seq(r), rest))
        }
    }
}

/// A `rest` tail as `next` answers it (`rest` keeps it): `nil` for
/// the empty list.
fn nil_punned(tail: Value, rest: bool) -> Value {
    if !rest && matches!(&tail, Value::List(t) if t.is_empty()) {
        Value::Nil
    } else {
        tail
    }
}

/// `rest`: a seq value, never nil (empty lists print `()`).
pub(crate) fn rest_value(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    // Pendings rest directly: `seq` would only force and re-wrap the
    // same cell (memoized either way; empty and error paths agree).
    if let Some(p) = as_pending(v) {
        return rest_pending(eval, p);
    }
    let s = seq_value(eval, v)?;
    rest_of_seq(eval, &s)
}

/// `next`: like `rest` but nil at the end (forcing one step).
pub(crate) fn next_value(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    let rest = rest_value(eval, v)?;
    // `next` nil-puns empty tails (spent repeats, done ranges —
    // like the JVM's `next.seq()`).
    seq_value(eval, &rest)
}

/// Take ownership of a walked coll arg, niling the slot: the walk
/// holds the only handle, so memoized seq chunks free as it
/// advances (like the JVM's dead locals) instead of accumulating
/// behind the head held in `args` (#221).
pub(crate) fn take_walk_coll(args: &mut SmallVec<[Value; 4]>, idx: usize) -> Value {
    std::mem::replace(&mut args[idx], Value::Nil)
}

/// Realize every item (for `apply` spreads, `vec`, `sort`,
/// `count`, printing, and `=`).
/// Streaming `collect_all` into a vector: items flow one by one
/// through a transient, so all-`Int` seqs freeze to packed storage
/// without ever materializing a `Vec<Value>` (8 bytes per element
/// peak instead of 72, single copy).
pub(crate) fn collect_packed(eval: &mut Evaluator, v: Value) -> Result<Vector, Signal> {
    if let Some(direct) = collect_direct(&v) {
        return Ok(direct);
    }
    let mut tv = TransientVector::new();
    // Owned: the walk advances `cur` off the head (owned callers
    // nilled the arg slot, so memoized chunks free as it walks).
    let mut cur = v;
    loop {
        match step_unit(eval, &cur)? {
            StepUnit::Empty => {
                return Ok(tv.persistent().expect("fresh transient is valid at freeze"));
            }
            StepUnit::One { head, tail } => {
                tv.conj(head).expect("fresh transient accepts conj");
                cur = tail;
            }
            StepUnit::Chunk { items, tail } => {
                for item in items.iter().cloned() {
                    tv.conj(item).expect("fresh transient accepts conj");
                }
                cur = tail;
            }
        }
    }
}

/// A finite `repeat` or a counted `long` range built without walking
/// its seq (#226: `(vec (repeat n 0))` builds DP tables; the items
/// are exactly the seq's).
fn collect_direct(v: &Value) -> Option<Vector> {
    let Value::Extension(e) = v else {
        return None;
    };
    let any = e.as_any();
    if let Some(r) = any.downcast_ref::<clj_value::RepeatValue>() {
        let n = usize::try_from(r.n()?.max(0)).ok()?;
        return Some(match r.x() {
            Value::Int(x) => Vector::from_ints(vec![*x; n]),
            Value::Char(c) => Vector::from_chars(vec![*c; n]),
            x => Vector::from_vec(vec![x.clone(); n]),
        });
    }
    let r = any.downcast_ref::<clj_value::RangeValue>()?;
    let (clj_value::RangeNum::Int(cur), clj_value::RangeNum::Int(step), Some(n)) =
        (r.cur(), r.step(), r.count().filter(|_| r.is_long()))
    else {
        return None;
    };
    let ints = (0..n.max(0))
        .map(|k| k.checked_mul(step).and_then(|d| cur.checked_add(d)))
        .collect::<Option<Vec<i64>>>()?;
    Some(Vector::from_ints(ints))
}

pub(crate) fn collect_all(eval: &mut Evaluator, v: Value) -> Result<Vec<Value>, Signal> {
    // A vector's items copy out at their exact count (no growth
    // doubling, no chunk copies).
    if let Value::Vector(x) = &v {
        return Ok(x.iter().collect());
    }
    let mut out = Vec::new();
    // Owned: the walk advances `cur` off the head (owned callers
    // nilled the arg slot, so memoized chunks free as it walks).
    let mut cur = v;
    loop {
        match step_unit(eval, &cur)? {
            StepUnit::Empty => return Ok(out),
            StepUnit::One { head, tail } => {
                out.push(head);
                cur = tail;
            }
            StepUnit::Chunk { items, tail } => {
                out.extend(items.iter().cloned());
                cur = tail;
            }
        }
    }
}

/// Whether hashing throws `UnsupportedOperationException` (the
/// infinite `Cycle`/`Iterate`/`Repeat`, like the JVM's `hasheq`
/// overrides — empty message).
pub(crate) fn is_hash_uoe(v: &Value) -> bool {
    if let Some(r) = as_repeat(v) {
        return r.n().is_none();
    }
    as_cycle(v).is_some() || as_iterate(v).is_some()
}

/// The `UnsupportedOperationException` (empty message) for
/// unhashable infinite seqs.
fn uoe() -> NativeError {
    NativeError::new(ThrowClass::UnsupportedOperationException, String::new())
}

/// Whether `=` walks the value pairwise (lists, vectors, and every
/// seq object — like `Sequential`).
fn is_sequential(v: &Value) -> bool {
    matches!(v, Value::Vector(_) | Value::MapEntry { .. })
        || as_subvector(v).is_some()
        || is_seq_value(v)
}

/// Whether a seq value is `Counted` (probed): long ranges, conses,
/// and flagged realized seqs; repeats/cycles/iterates, entry views,
/// and pending seqs never are.
pub(crate) fn is_counted_seq(v: &Value) -> bool {
    if let Some(r) = as_range(v) {
        return r.is_long();
    }
    // `Cons` is never `Counted` — not even over a vector
    // (fixprobe7 `:cc-cons` reverses the old pin).
    if as_cons_value(v).is_some() {
        return false;
    }
    if let Some(s) = as_realized(v) {
        return s.is_counted();
    }
    false
}

/// Whether a half-walked side is spent (buffer drained, tail
/// empty).
fn equiv_drained(eval: &mut Evaluator, tail: &Value) -> Result<bool, Signal> {
    // Step chunks are never empty, so one step decides.
    match step_unit(eval, tail)? {
        StepUnit::Empty => Ok(true),
        StepUnit::One { .. } | StepUnit::Chunk { .. } => Ok(false),
    }
}

/// Fill a walk buffer from its tail (`Ok(false)` when spent).
fn equiv_fill(
    eval: &mut Evaluator,
    buf: &mut Vec<Value>,
    tail: &mut Value,
) -> Result<bool, Signal> {
    // Step chunks are never empty, so the first unit always fills.
    match step_unit(eval, tail)? {
        StepUnit::Empty => Ok(false),
        StepUnit::One { head, tail: rest } => {
            *buf = vec![head];
            *tail = rest;
            Ok(true)
        }
        StepUnit::Chunk { items, tail: rest } => {
            *buf = items.to_vec();
            *tail = rest;
            Ok(true)
        }
    }
}

/// Pairwise sequential walk with early exit (like
/// `Util.doEquiv`): finite-vs-infinite answers `false` instead of
/// hanging, and errors surface from whichever side forces first.
fn equiv_seq(eval: &mut Evaluator, a: &Value, b: &Value) -> Result<bool, Signal> {
    let mut ta = a.clone();
    let mut tb = b.clone();
    let (mut ba, mut bb) = (Vec::new(), Vec::new());
    let (mut pa, mut pb) = (0usize, 0usize);
    loop {
        if pa >= ba.len() {
            if !equiv_fill(eval, &mut ba, &mut ta)? {
                if pb < bb.len() {
                    return Ok(false);
                }
                return equiv_drained(eval, &tb);
            }
            pa = 0;
        }
        if pb >= bb.len() {
            if !equiv_fill(eval, &mut bb, &mut tb)? {
                return Ok(false);
            }
            pb = 0;
        }
        if !equiv_vals(eval, &ba[pa], &bb[pb])? {
            return Ok(false);
        }
        pa += 1;
        pb += 1;
    }
}

/// Map `=` branch: size plus per-key recursion (missing keys fail
/// even for nil values, like the JVM's sentinel lookup).
fn equiv_map_vals(eval: &mut Evaluator, x: &Map, y: &Map) -> Result<bool, Signal> {
    if x.size() != y.size() {
        return Ok(false);
    }
    for (k, v) in x.iter() {
        if is_hash_uoe(k) {
            return Err(eval.throw(uoe()));
        }
        let k = realize_key(eval, k)?;
        let Some(w) = y.get(&k) else {
            return Ok(false);
        };
        if !equiv_vals(eval, v, w)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Set `=` branch: size plus per-member pairwise search.
fn equiv_set_vals(eval: &mut Evaluator, x: &Set, y: &Set) -> Result<bool, Signal> {
    if x.size() != y.size() {
        return Ok(false);
    }
    for v in x.iter() {
        if is_hash_uoe(v) {
            return Err(eval.throw(uoe()));
        }
        if !equiv_set_member(eval, v, y)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether one set member has a pairwise-equal partner.
fn equiv_set_member(eval: &mut Evaluator, v: &Value, y: &Set) -> Result<bool, Signal> {
    for w in y.iter() {
        if equiv_vals(eval, v, w)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The map behind a value: plain maps read directly, struct
/// instances read their content (S12); anything else is `None`.
fn map_or_struct_content(v: &Value) -> Option<&Map> {
    match v {
        Value::Map(m) => Some(m),
        other => crate::types::struct_inst(other).map(|i| &i.content),
    }
}

/// `Util.equiv`: sequential pairs walk pairwise (early exit);
/// maps compare by size plus per-key recursion (missing keys fail
/// even for nil values, like the JVM's sentinel lookup) — struct
/// instances equiv as their content map, to plain maps and across
/// bases (S12 oracles `sm-eq-map`, `sm-cross-basis`);
/// unhashable-infinite members throw like `hasheq` does during
/// the JVM's hash lookups; anything else falls back to `==`.
pub(crate) fn equiv_vals(eval: &mut Evaluator, a: &Value, b: &Value) -> Result<bool, Signal> {
    if is_sequential(a) && is_sequential(b) {
        return equiv_seq(eval, a, b);
    }
    if let (Some(x), Some(y)) = (map_or_struct_content(a), map_or_struct_content(b)) {
        return equiv_map_vals(eval, x, y);
    }
    if let (Value::Set(x), Value::Set(y)) = (a, b) {
        return equiv_set_vals(eval, x, y);
    }
    Ok(a == b)
}

/// Realize a prospective map/set key (the key invariant):
/// pending seqs deep-realize to `RealizedSeq` (which compares
/// by content under `PartialEq`/`Hash`, like the JVM's memoized
/// `LazySeq.equals`/`hashCode`); anything else passes through
/// untouched. Every map/set key boundary funnels here so stored
/// and lookup keys always agree — without it, `(frequencies
/// (partition …))` never merges (#205 bpe-tokenizer).
pub(crate) fn realize_key(eval: &mut Evaluator, k: &Value) -> Result<Value, Signal> {
    if let Some(p) = as_pending(k) {
        return realize_pending_deep(eval, k, p);
    }
    Ok(k.clone())
}

/// Deep-realize a pending seq into a realized one (metadata preserved).
fn realize_pending_deep(eval: &mut Evaluator, v: &Value, p: &PendingSeq) -> Result<Value, Signal> {
    let items = collect_all(eval, v.clone())?;
    // Recurse like the realized path: collected items may hold
    // nested pending seqs (partition/partition-by chunks), which
    // must print realized, not `(...)`.
    let mut realized = Vec::with_capacity(items.len());
    for item in items {
        realized.push(realize_deep(eval, &item)?);
    }
    let mut seq = RealizedSeq::new(Seq::eager(realized).unwrap_or_else(Seq::empty));
    seq = seq.with_meta(p.meta().cloned());
    Ok(Value::Extension(Arc::new(seq)))
}

/// Deep-realize a realized seq (items rebuilt, metadata preserved).
fn realize_realized_deep(eval: &mut Evaluator, s: &RealizedSeq) -> Result<Value, Signal> {
    let mut items = Vec::new();
    for item in s.items() {
        items.push(realize_deep(eval, &item)?);
    }
    let mut seq = RealizedSeq::new(Seq::eager(items).unwrap_or_else(Seq::empty));
    seq = seq.with_meta(s.meta().cloned());
    Ok(Value::Extension(Arc::new(seq)))
}

/// Whether the value is a strict seq object that deep-realizes by
/// collecting (ranges, repeats, cycles, iterates, conses, entry views).
fn is_collectable_seq(v: &Value) -> bool {
    as_range(v).is_some()
        || as_repeat(v).is_some()
        || as_cycle(v).is_some()
        || as_iterate(v).is_some()
        || as_cons_value(v).is_some()
        || as_keys(v).is_some()
        || as_vals(v).is_some()
}

/// Deep-realize a strict seq object by collecting (metadata preserved).
fn realize_strict_seq_deep(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    let items = collect_all(eval, v.clone())?;
    let mut seq = RealizedSeq::new(Seq::eager(items).unwrap_or_else(Seq::empty));
    seq = seq.with_meta(v.meta().cloned());
    Ok(Value::Extension(Arc::new(seq)))
}

/// Deep-realize a subvector view (window rebuilt with realized items).
fn realize_subvector_deep(eval: &mut Evaluator, sv: &SubVectorValue) -> Result<Value, Signal> {
    let mut items = Vec::with_capacity(sv.len());
    for item in sv.iter() {
        items.push(realize_deep(eval, &item)?);
    }
    Ok(Value::Extension(Arc::new(sv.with_window(items))))
}

/// Deep-realize an extension value into a realized one.
fn realize_deep_extension(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    if let Some(p) = as_pending(v) {
        return realize_pending_deep(eval, v, p);
    }
    if let Some(s) = as_realized(v) {
        return realize_realized_deep(eval, s);
    }
    if is_collectable_seq(v) {
        return realize_strict_seq_deep(eval, v);
    }
    if let Some(sv) = as_subvector(v) {
        return realize_subvector_deep(eval, sv);
    }
    Ok(v.clone())
}

/// Deep-realize a list (metadata preserved).
fn realize_deep_list(eval: &mut Evaluator, l: &List) -> Result<Value, Signal> {
    let mut items = Vec::with_capacity(l.len());
    for item in l.iter() {
        items.push(realize_deep(eval, item)?);
    }
    Ok(Value::List(
        List::from_vec(items).with_meta(l.meta().cloned()),
    ))
}

/// Deep-realize a vector (metadata preserved).
fn realize_deep_vector(eval: &mut Evaluator, x: &Vector) -> Result<Value, Signal> {
    let mut items = Vec::with_capacity(x.len());
    for item in x.iter() {
        items.push(realize_deep(eval, &item)?);
    }
    Ok(Value::Vector(
        Vector::from_vec(items).with_meta(x.meta().cloned()),
    ))
}

/// Deep-realize a map entry (key and value realized, the entry
/// rebuilt — Oracle FX2 `lazy-val-entry`).
fn realize_deep_entry(eval: &mut Evaluator, key: &Value, val: &Value) -> Result<Value, Signal> {
    Ok(clj_value::map_entry(
        realize_deep(eval, key)?,
        realize_deep(eval, val)?,
    ))
}

/// Deep-realize a map (metadata preserved).
fn realize_deep_map(eval: &mut Evaluator, m: &Map) -> Result<Value, Signal> {
    let mut pairs = Vec::with_capacity(m.size());
    for (k, val) in m.iter() {
        pairs.push((realize_deep(eval, k)?, realize_deep(eval, val)?));
    }
    let mut out = clj_value::new_map();
    for (k, val) in pairs {
        out = out.insert(k, val);
    }
    // Realization preserves the map class (sorted stays sorted,
    // hashy stays hashy — the print driver dispatches on it).
    let out = out
        .with_sort(m.sort_spec().cloned())
        .with_hashy(m.is_hashy());
    Ok(Value::Map(out.with_meta(m.meta().cloned())))
}

/// Deep-realize a set (metadata preserved).
fn realize_deep_set(eval: &mut Evaluator, s: &Set) -> Result<Value, Signal> {
    let mut out = clj_value::new_set();
    for item in s.iter() {
        out = out.insert(realize_deep(eval, item)?);
    }
    // Realization preserves the set class (sorted stays sorted —
    // the print driver dispatches on it).
    let out = out.with_sort(s.sort_spec().cloned());
    Ok(Value::Set(out.with_meta(s.meta().cloned())))
}

/// Deep-realize pending seqs into realized ones (for printing,
/// `str`, and `=`); realized seqs rebuild with realized items and
/// preserved metadata.
pub(crate) fn realize_deep(eval: &mut Evaluator, v: &Value) -> Result<Value, Signal> {
    // Unbounded seqs are preserved, never expanded: expanding them
    // hangs, `hash`'s post-check (like the JVM's `hasheq` overrides)
    // must SEE them to throw `UnsupportedOperationException`, and
    // printing renders them `(...)` like top-level infinites.
    if clj_value::seq_hash_unsupported(v) {
        return Ok(v.clone());
    }
    // Records realize their content like maps (a lazy field
    // value forces before `hash`/`pr` walks it); structs likewise
    // (S12); other host values hold no realizable spine.
    if let Some(i) = crate::types::maplike_inst(v) {
        let mut pairs = Vec::with_capacity(i.content.size());
        for (k, val) in i.content.iter() {
            pairs.push((realize_deep(eval, k)?, realize_deep(eval, val)?));
        }
        let mut out = clj_value::new_map();
        for (k, val) in pairs {
            out = out.insert(k, val);
        }
        return Ok(crate::types::rewrap_maplike(i, out));
    }
    match v {
        Value::Extension(_) => realize_deep_extension(eval, v),
        Value::List(l) => realize_deep_list(eval, l),
        Value::Vector(x) => realize_deep_vector(eval, x),
        Value::MapEntry { key, val, .. } => realize_deep_entry(eval, key, val),
        Value::Map(m) => realize_deep_map(eval, m),
        Value::Set(s) => realize_deep_set(eval, s),
        _ => Ok(v.clone()),
    }
}

/// Build a pending seq from a thunk (for the seq natives).
pub(crate) fn pending_value(thunk: Thunk) -> Value {
    PendingSeq::pending(thunk)
}

/// A `Thunk::Map` tail value (for `map`).
pub(crate) fn map_thunk(f: Value, inputs: Vec<Value>) -> Value {
    pending_value(Thunk::Map { f, inputs })
}

/// A `Thunk::MapCat` tail value (for `mapcat`, over the map outer).
pub(crate) fn mapcat_thunk(f: Value, inputs: Vec<Value>) -> Value {
    pending_value(Thunk::MapCat {
        outer: map_thunk(f, inputs),
    })
}

/// A `Thunk::Replace` tail value (for `replace` over non-vectors).
pub(crate) fn replace_thunk(smap: Value, input: Value) -> Value {
    pending_value(Thunk::Replace { smap, input })
}

/// `replace` over a non-vector: swap members found in `smap` (a
/// `find` per item; missing keys pass through; an unsupported
/// `smap` fails on the first item).
fn step_replace(eval: &mut Evaluator, smap: &Value, input: Value) -> Result<Forced, Signal> {
    let (items, tail) = match step_unit(eval, &input)? {
        StepUnit::Empty => return Ok(Forced::Empty),
        StepUnit::One { head, tail } => (vec![head], tail),
        StepUnit::Chunk { items, tail } => (items.to_vec(), tail),
    };
    let mut out = Vec::with_capacity(items.len());
    for item in &items {
        let found = crate::coll::find_entry(eval, smap, item).map_err(|e| eval.throw(e))?;
        out.push(found.map_or_else(|| item.clone(), |(_, v)| v));
    }
    Ok(Forced::Chunk {
        items: Arc::from(out),
        tail: PendingSeq::pending(Thunk::Replace {
            smap: smap.clone(),
            input: tail,
        }),
    })
}

/// A `Thunk::Filter`/`Remove` tail value.
pub(crate) fn filter_thunk(f: Value, input: Value, invert: bool) -> Value {
    pending_value(if invert {
        Thunk::Remove { f, input }
    } else {
        Thunk::Filter { f, input }
    })
}

/// A `Thunk::Take` tail value (the count stays raw — invalid
/// counts fail when forced, like the JVM's lazy `pos?`).
pub(crate) fn take_thunk(n: Value, input: Value) -> Value {
    pending_value(Thunk::Take { n, input })
}

/// A `Thunk::Concat` tail value.
pub(crate) fn concat_thunk(inputs: Vec<Value>) -> Value {
    let mut inputs = inputs.into_iter();
    let cur = inputs.next().unwrap_or(Value::Nil);
    pending_value(Thunk::Concat {
        cur,
        rest: Arc::new(inputs.collect()),
        idx: 0,
        outer: None,
    })
}

/// `int?` over our numbers (fixed-precision integers only).
/// A bound as a number pair (`Int`/`Int` compare exactly, anything
/// else as `f64` — like `Numbers`). Bad sides fail x-first (nil is
/// the numbers NPE, anything else the `Number` cast failure).
fn num_pair(a: &Value, b: &Value) -> Result<(Num, Num), NativeError> {
    Ok((crate::core::as_num(a)?, crate::core::as_num(b)?))
}

/// `Numbers.gt` over our numbers (`Int`/`Int` exact, `Double`
/// pairs as `f64`, tower pairs exactly).
fn num_gt(a: &Value, b: &Value) -> Result<bool, NativeError> {
    match num_pair(a, b)? {
        (Num::Int(x), Num::Int(y)) => Ok(x > y),
        (x, y) if is_double_num(&x) || is_double_num(&y) => Ok(num_f64(x) > num_f64(y)),
        (x, y) => Ok(num_rational(&x) > num_rational(&y)),
    }
}

/// `Numbers.equiv` over our numbers (`NaN` never, cross-type by
/// value — like `Range.create`'s emptiness check; tower pairs
/// use the `==` rules, exactly).
fn num_equiv(a: &Value, b: &Value) -> Result<bool, NativeError> {
    match num_pair(a, b)? {
        (Num::Int(x), Num::Int(y)) => Ok(x == y),
        (x, y) if is_double_num(&x) || is_double_num(&y) => Ok(num_f64(x) == num_f64(y)),
        (x, y) => Ok(num_rational(&x) == num_rational(&y)),
    }
}

/// `Numbers.isPos`/`isNeg` over our numbers (`sign > 0` / `< 0`,
/// tower exactly).
fn num_sign(v: &Value, positive: bool) -> Result<bool, NativeError> {
    match crate::core::as_num(v)? {
        Num::Int(i) => Ok(if positive { i > 0 } else { i < 0 }),
        Num::Double(d) => Ok(if positive { d > 0.0 } else { d < 0.0 }),
        n => Ok(num_signum(&n) == (if positive { 1 } else { -1 })),
    }
}

/// `Numbers.isZero` over our numbers (tower exactly).
fn num_is_zero(v: &Value) -> Result<bool, NativeError> {
    match crate::core::as_num(v)? {
        Num::Int(i) => Ok(i == 0),
        Num::Double(d) => Ok(d == 0.0),
        n => Ok(num_signum(&n) == 0),
    }
}

/// A `Num` as `f64` (tower narrows approximately).
fn num_f64(n: Num) -> f64 {
    match n {
        Num::Int(i) => i as f64,
        Num::BigInt(n) => n.to_f64().unwrap_or(f64::NAN),
        Num::Ratio(r) => r.to_f64().unwrap_or(f64::NAN),
        Num::BigDec(d) => d.to_f64(),
        Num::Double(d) => d,
    }
}

/// Whether a range `Num` is a `Double` (the approximate path).
fn is_double_num(n: &Num) -> bool {
    matches!(n, Num::Double(_))
}

/// The sign of a range `Num` as `-1`/`0`/`1` (tower exactly via
/// the rational numerator; the denominator stays positive).
fn num_signum(n: &Num) -> i32 {
    match num_rational(n).numer().sign() {
        num_bigint::Sign::Minus => -1,
        num_bigint::Sign::NoSign => 0,
        num_bigint::Sign::Plus => 1,
    }
}

/// An exact rational for a range `Num` (`Double` never reaches
/// here — doubles always take the `f64` path above).
fn num_rational(n: &Num) -> num_rational::BigRational {
    match n {
        Num::Int(i) => num_rational::BigRational::from_integer(num_bigint::BigInt::from(*i)),
        Num::BigInt(n) => num_rational::BigRational::from_integer(n.clone()),
        Num::Ratio(r) => r.clone(),
        Num::BigDec(d) => d.to_rational(),
        Num::Double(_) => {
            unreachable!("doubles take the f64 path, never the exact one")
        }
    }
}

/// A numeric value as a range bound (only reached for numbers —
/// the `create` checks reject the rest first; tower bounds walk
/// exactly).
fn range_num(v: &Value) -> RangeNum {
    // Callers validate numbers first; sequential tests keep every
    // line live.
    match v {
        Value::Int(i) => RangeNum::Int(*i),
        Value::Double(d) => RangeNum::Double(*d),
        Value::BigInt(n) | Value::BigInteger(n) => RangeNum::BigInt(n.as_ref().clone()),
        Value::Ratio(r) => RangeNum::Ratio(r.as_ref().clone()),
        Value::BigDec(d) => RangeNum::BigDec(d.as_ref().clone()),
        _ => RangeNum::Int(0),
    }
}

/// `LongRange.rangeCount` (checked; `None` overflows like the
/// JVM's `ArithmeticException`, routing to the double path).
fn range_count(start: i64, end: i64, step: i64) -> Option<i64> {
    let total = end
        .checked_sub(start)?
        .checked_add(step)?
        .checked_add(if step > 0 { -1 } else { 1 })?;
    total.checked_div(step)
}

/// `LongRange.create` (all-`Int` args): empties answer `()`,
/// zero steps repeat the start, overflows fall back to the double
/// path with `Int` bounds.
fn long_range(s: i64, e: i64, st: i64) -> Value {
    if (st > 0 && e <= s) || (st < 0 && e >= s) || (st == 0 && e == s) {
        return empty_list_value();
    }
    if st == 0 {
        return Value::Extension(Arc::new(RepeatValue::new(Value::Int(s), None)));
    }
    match range_count(s, e, st).and_then(|c| i32::try_from(c).ok()) {
        Some(count) => Value::Extension(Arc::new(RangeValue::new(
            RangeNum::Int(s),
            Some(RangeNum::Int(e)),
            RangeNum::Int(st),
            true,
            Some(i64::from(count)),
        ))),
        None => Value::Extension(Arc::new(RangeValue::new(
            RangeNum::Int(s),
            Some(RangeNum::Int(e)),
            RangeNum::Int(st),
            false,
            None,
        ))),
    }
}

/// `Range.create` (any non-`Int` arg): bound checks answer `()`,
/// zero steps repeat the start, else a double-path `Range`.
fn double_range(start: Value, end: Value, step: Value) -> Result<Value, NativeError> {
    if (num_sign(&step, true)? && num_gt(&start, &end)?)
        || (num_sign(&step, false)? && num_gt(&end, &start)?)
        || num_equiv(&start, &end)?
    {
        return Ok(empty_list_value());
    }
    if num_is_zero(&step)? {
        return Ok(Value::Extension(Arc::new(RepeatValue::new(start, None))));
    }
    Ok(Value::Extension(Arc::new(RangeValue::new(
        range_num(&start),
        Some(range_num(&end)),
        range_num(&step),
        false,
        None,
    ))))
}

/// The 1-arity `(range end)`: `int?` dispatches to
/// `LongRange.create`, else `Range.create` (whose `isPos` errors
/// on non-numbers).
pub(crate) fn range1(end: Value) -> Result<Value, NativeError> {
    if let Value::Int(e) = &end {
        return Ok(long_range(0, *e, 1));
    }
    if num_sign(&end, true)? {
        Ok(Value::Extension(Arc::new(RangeValue::new(
            RangeNum::Int(0),
            Some(range_num(&end)),
            RangeNum::Int(1),
            false,
            None,
        ))))
    } else {
        Ok(empty_list_value())
    }
}

/// The 2-arity `(range start end)`.
pub(crate) fn range2(start: Value, end: Value) -> Result<Value, NativeError> {
    if let (Value::Int(s), Value::Int(e)) = (&start, &end) {
        return Ok(long_range(*s, *e, 1));
    }
    double_range(start, end, Value::Int(1))
}

/// The 3-arity `(range start end step)`.
pub(crate) fn range3(start: Value, end: Value, step: Value) -> Result<Value, NativeError> {
    if let (Value::Int(s), Value::Int(e), Value::Int(st)) = (&start, &end, &step) {
        return Ok(long_range(*s, *e, *st));
    }
    double_range(start, end, step)
}

/// A `Thunk::Thunk0` tail value (the `lazy-seq*` body).
pub(crate) fn thunk0_value(body: Arc<Vec<Expr>>, env: Env) -> Value {
    pending_value(Thunk::Thunk0 { body, env })
}

/// The VM's `lazy-seq*` value over a zero-arity body closure.
pub(crate) fn call0_value(f: Value) -> Value {
    pending_value(Thunk::Call0 { f })
}

/// A `repeat` value: non-positive counts answer `()` (like
/// `Repeat.create`), else a strict `Repeat` (`None` is infinite).
pub(crate) fn repeat_thunk(x: Value, n: Option<i64>) -> Value {
    if n.is_some_and(|n| n <= 0) {
        return empty_list_value();
    }
    Value::Extension(Arc::new(RepeatValue::new(x, n)))
}

/// An infinite `repeatedly` tail value (the `n` arity routes
/// through `take`, so counts stay lazy there).
pub(crate) fn repeatedly_thunk(f: Value) -> Value {
    pending_value(Thunk::Repeatedly { f })
}

/// An `iterate` value (the seed emits free, like
/// `Iterate.create` — non-`IFn` callees fail at the call site).
pub(crate) fn iterate_thunk(f: Value, x: Value) -> Value {
    Value::Extension(Arc::new(IterateValue::new(f, x)))
}

/// A `cycle` value: the input seqed eagerly, `()` for empties
/// (like `Cycle.create`), else a strict `Cycle` replaying it.
pub(crate) fn cycle_thunk(eval: &mut Evaluator, input: Value) -> Result<Value, Signal> {
    match seq_value(eval, &input)? {
        Value::Nil => Ok(empty_list_value()),
        s => Ok(Value::Extension(Arc::new(CycleValue::new(s.clone(), s)))),
    }
}

/// A `Thunk::Interleave` tail value (empty round queue).
pub(crate) fn interleave_thunk(inputs: Vec<Value>) -> Value {
    pending_value(Thunk::Interleave {
        queue: Vec::new(),
        inputs,
    })
}

/// A `keys`/`vals` entry view over the seqed input (`None` for
/// empties, like `KeySeq.create`).
pub(crate) fn keys_thunk(eval: &mut Evaluator, input: Value, keys: bool) -> Result<Value, Signal> {
    match seq_value(eval, &input)? {
        Value::Nil => Ok(Value::Nil),
        s => Ok(if keys {
            Value::Extension(Arc::new(KeySeqValue::new(s)))
        } else {
            Value::Extension(Arc::new(ValSeqValue::new(s)))
        }),
    }
}

/// A `Thunk::Interpose` tail value.
pub(crate) fn interpose_thunk(sep: Value, input: Value) -> Value {
    pending_value(Thunk::Interpose {
        sep,
        input,
        first: true,
    })
}

/// A `Thunk::TakeWhile` tail value.
pub(crate) fn take_while_thunk(f: Value, input: Value) -> Value {
    pending_value(Thunk::TakeWhile { f, input })
}

/// A `Thunk::DropWhile` tail value.
pub(crate) fn drop_while_thunk(f: Value, input: Value) -> Value {
    pending_value(Thunk::DropWhile { f, input })
}

/// A `Thunk::Drop` tail value (the count stays raw — validated
/// when forced).
pub(crate) fn drop_thunk(n: Value, input: Value) -> Value {
    pending_value(Thunk::Drop { n, input })
}

/// A `Thunk::TakeNth` tail value (the count stays raw —
/// validated per force through the body's `drop`).
pub(crate) fn take_nth_thunk(n: Value, input: Value) -> Value {
    pending_value(Thunk::TakeNth { n, input })
}

/// A `Thunk::Partition` tail value (counts stay raw; `None` pad
/// drops short windows, `Some` pads and keeps them).
pub(crate) fn partition_thunk(n: Value, step: Value, pad: Option<Value>, input: Value) -> Value {
    pending_value(Thunk::Partition {
        n,
        step,
        pad,
        input,
    })
}

/// A `Thunk::PartitionAll` tail value (counts stay raw).
pub(crate) fn partition_all_thunk(n: Value, step: Value, input: Value) -> Value {
    pending_value(Thunk::PartitionAll { n, step, input })
}

/// A `Thunk::PartitionBy` tail value (runs key off the head's
/// value per step, like the JVM body).
pub(crate) fn partition_by_thunk(f: Value, input: Value) -> Value {
    pending_value(Thunk::PartitionBy { f, input })
}

/// A `Thunk::Distinct` tail value (nothing seen yet).
pub(crate) fn distinct_thunk(input: Value) -> Value {
    pending_value(Thunk::Distinct {
        seen: clj_value::MemberSet::default(),
        input,
    })
}

/// A `Thunk::Dedupe` tail value (no last item yet).
pub(crate) fn dedupe_thunk(input: Value) -> Value {
    pending_value(Thunk::Dedupe { last: None, input })
}

/// T1200 D1: a `Thunk::Sequence` tail value (stepping fn over
/// lockstep sources, for `sequence` and eduction seq).
pub(crate) fn sequence_thunk(step: Value, srcs: Vec<Value>) -> Value {
    pending_value(Thunk::Sequence { step, srcs })
}

/// A `Thunk::Keep` tail value.
pub(crate) fn keep_thunk(f: Value, input: Value) -> Value {
    pending_value(Thunk::Keep {
        f,
        input,
        skip: None,
    })
}

/// A `Thunk::KeepIndexed` tail value (indices from zero).
pub(crate) fn keep_indexed_thunk(f: Value, input: Value) -> Value {
    pending_value(Thunk::KeepIndexed { f, input, idx: 0 })
}

/// A `Thunk::MapIndexed` tail value (indices from zero).
pub(crate) fn map_indexed_thunk(f: Value, input: Value) -> Value {
    pending_value(Thunk::MapIndexed { f, input, idx: 0 })
}

/// A `Thunk::Reductions` tail value (`None` init seeds from the head).
pub(crate) fn reductions_thunk(f: Value, init: Option<Value>, input: Value) -> Value {
    pending_value(Thunk::Reductions {
        f,
        acc: init.clone(),
        has_init: init.is_some(),
        input,
        started: false,
    })
}

fn native_seq(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/seq", args, &[1])?;
    seq_value(eval, &args[0]).map_err(signal_to_native)
}

fn native_first(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/first", args, &[1])?;
    first_value(eval, &args[0]).map_err(signal_to_native)
}

fn native_rest(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/rest", args, &[1])?;
    rest_value(eval, &args[0]).map_err(signal_to_native)
}

fn native_next(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/next", args, &[1])?;
    next_value(eval, &args[0]).map_err(signal_to_native)
}

pub(crate) fn native_cons(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/cons", args, &[2])?;
    // Like `RT.cons`: nil tails answer a one-list, seq tails pass
    // through untouched (never forced), anything else seqs eagerly
    // (unseqables fail now).
    if matches!(args[1], Value::Nil) {
        return Ok(Value::List(List::from_vec(vec![args[0].clone()])));
    }
    if is_seq_value(&args[1]) {
        return Ok(Value::Extension(Arc::new(ConsValue::new(
            args[0].clone(),
            args[1].clone(),
        ))));
    }
    let tail = seq_value(eval, &args[1]).map_err(signal_to_native)?;
    Ok(Value::Extension(Arc::new(ConsValue::new(
        args[0].clone(),
        tail,
    ))))
}

/// The S5D seq-access natives installed into `clojure.core`.
pub(crate) const LAZY_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("__for-keep*", native_for_keep),
    ("seq", native_seq),
    ("first", native_first),
    ("rest", native_rest),
    ("next", native_next),
    ("cons", native_cons),
];

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use super::*;
    use clj_value::new_map;

    fn pending_seq(state: SeqState, kind: LazyKind) -> PendingSeq {
        PendingSeq {
            state: Mutex::new(state),
            meta: None,
            kind,
        }
    }

    fn empty_chain() -> Thunk {
        Thunk::Chain {
            head: Vec::new(),
            tail: Value::Nil,
        }
    }

    /// Non-numbers fall back to an exact zero bound (callers
    /// validate numbers first — the pin keeps the totality).
    #[test]
    fn range_num_falls_back_for_non_numbers() {
        assert_eq!(range_num(&Value::Nil), RangeNum::Int(0));
    }

    /// The exact range rational rejects `Double` (bound checks
    /// dispatch on `is_double_num` first — the panic is the
    /// contract).
    #[test]
    #[should_panic(expected = "doubles take the f64 path, never the exact one")]
    fn range_num_rational_rejects_double() {
        let _ = num_rational(&crate::core::Num::Double(0.5));
    }

    #[test]
    fn print_pending_and_empty() {
        // Best-effort print contract (out-of-band: natives realize first).
        let pending = pending_seq(
            SeqState::Pending(Box::new(empty_chain())),
            LazyKind::LazySeq,
        );
        assert_eq!(pending.print(), "(...)");
        let empty = pending_seq(SeqState::Empty, LazyKind::LazySeq);
        assert_eq!(empty.print(), "()");
    }

    #[test]
    fn memoized_chunk_heads_read_without_copying() {
        // `first` on a memoized chunk answers from the cursor (no
        // force, no chunk copy); `rest` advances a shared cursor
        // (the original keeps its own — seqs are immutable).
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let v = Value::Extension(Arc::new(pending_seq(
            SeqState::Chunk {
                items: Arc::from(vec![Value::Int(1), Value::Int(2)]),
                pos: 0,
                tail: Value::Nil,
            },
            LazyKind::LazySeq,
        )));
        assert_eq!(super::first_value(&mut eval, &v).unwrap(), Value::Int(1));
        let r = super::rest_value(&mut eval, &v).unwrap();
        assert_eq!(super::first_value(&mut eval, &r).unwrap(), Value::Int(2));
        assert_eq!(super::first_value(&mut eval, &v).unwrap(), Value::Int(1));
        // Rest at the last item seqs the (nil) tail to `()`.
        let end = super::rest_value(&mut eval, &r).unwrap();
        assert!(matches!(end, Value::List(l) if l.is_empty()));
        // Realized seqs read their head without stepping either.
        let vec = clj_value::Vector::from_vec(vec![Value::Int(7)]);
        let s = super::seq_value(&mut eval, &Value::Vector(vec)).unwrap();
        assert_eq!(super::first_value(&mut eval, &s).unwrap(), Value::Int(7));
    }

    #[test]
    fn advanced_and_exhausted_chunks_use_the_tail() {
        // Forcing an advanced cursor re-slices (the `pos > 0`
        // arm); `first`/`rest` past the end continue in the tail,
        // like the JVM's `ChunkedCons.more()`.
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let tail = Value::List(clj_value::List::from_vec(vec![Value::Int(9)]));
        let v = Value::Extension(Arc::new(pending_seq(
            SeqState::Chunk {
                items: Arc::from(vec![Value::Int(1)]),
                pos: 1,
                tail: tail.clone(),
            },
            LazyKind::ChunkedCons,
        )));
        // `seq` forces (re-slice arm) and stays non-nil (live tail).
        assert!(matches!(
            super::seq_value(&mut eval, &v).unwrap(),
            Value::Extension(_)
        ));
        assert_eq!(super::first_value(&mut eval, &v).unwrap(), Value::Int(9));
        assert_eq!(super::rest_value(&mut eval, &v).unwrap(), tail);
    }

    #[test]
    fn print_chunk_states() {
        // A closed chunk prints its known items.
        let closed = pending_seq(
            SeqState::Chunk {
                items: Arc::from(vec![Value::Int(1), Value::Int(2)]),
                pos: 0,
                tail: Value::Nil,
            },
            LazyKind::LazySeq,
        );
        assert_eq!(closed.print(), "(1 2)");
        // A consumed chunk (empty remainder) prints open.
        let consumed = pending_seq(
            SeqState::Chunk {
                items: Arc::from(vec![Value::Int(1)]),
                pos: 1,
                tail: Value::Nil,
            },
            LazyKind::LazySeq,
        );
        assert_eq!(consumed.print(), "(...)");
        // A chunk with a live tail prints its prefix plus `...`.
        let tail = PendingSeq::pending(empty_chain());
        let open = pending_seq(
            SeqState::Chunk {
                items: Arc::from(vec![Value::Int(1)]),
                pos: 0,
                tail,
            },
            LazyKind::LazySeq,
        );
        assert_eq!(open.print(), "(1 ...)");
    }

    #[test]
    fn print_poisoned_lock_is_open() {
        let seq = pending_seq(SeqState::Empty, LazyKind::LazySeq);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = seq.state.lock().unwrap();
            panic!("poison the test lock");
        }));
        assert_eq!(seq.print(), "(...)");
    }

    #[test]
    fn identity_eq_and_hash() {
        // Pending seqs compare and hash by identity.
        let a = pending_seq(SeqState::Empty, LazyKind::LazySeq);
        let b = pending_seq(SeqState::Empty, LazyKind::LazySeq);
        assert!(a.dyn_eq(&a));
        assert!(!a.dyn_eq(&b));
        let mut first = std::collections::hash_map::DefaultHasher::new();
        a.dyn_hash(&mut first);
        let mut second = std::collections::hash_map::DefaultHasher::new();
        a.dyn_hash(&mut second);
        assert_eq!(first.finish(), second.finish());
    }

    #[test]
    fn with_meta_fresh_rebuilds_each_state() {
        // Staged machinery: `with-meta` on seqs is not wired yet
        // (`with_meta_val` still reports the IObj cast), so each
        // state's rebuild is covered directly.
        let meta = Some(new_map());
        // Pending stays pending, carrying the new meta.
        let pending = pending_seq(
            SeqState::Pending(Box::new(empty_chain())),
            LazyKind::LazySeq,
        );
        let rebuilt = pending.with_meta_fresh(meta.clone());
        let back = as_pending(&rebuilt).unwrap();
        // Single-line `matches!`: every test line must execute (line gate).
        assert!(matches!(&*back.state.lock().unwrap(), SeqState::Pending(_)));
        assert!(back.meta().is_some());
        // A memoized chunk rechains its unconsumed items.
        let chunk = pending_seq(
            SeqState::Chunk {
                items: Arc::from(vec![Value::Int(1), Value::Int(2)]),
                pos: 1,
                tail: Value::Nil,
            },
            LazyKind::ChunkedCons,
        );
        let rebuilt = chunk.with_meta_fresh(meta.clone());
        let back = as_pending(&rebuilt).unwrap();
        assert!(
            matches!(&*back.state.lock().unwrap(), SeqState::Pending(t) if matches!(&**t, Thunk::Chain { head, .. } if head.as_slice() == [Value::Int(2)]))
        );
        assert_eq!(back.meta().unwrap().size(), 0);
        // An exhausted seq stays empty.
        let empty = pending_seq(SeqState::Empty, LazyKind::LazySeq);
        let rebuilt = empty.with_meta_fresh(meta);
        let back = as_pending(&rebuilt).unwrap();
        assert!(matches!(&*back.state.lock().unwrap(), SeqState::Empty));
        assert!(back.meta().is_some());
    }

    #[test]
    #[should_panic(expected = "thunk partition")]
    fn step_thunk_rest_rejects_upstream_variants() {
        // Defensive arm: the single caller partitions the variants,
        // so `step_thunk`-handled thunks never arrive here (only a
        // dispatch bug could).
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let thunk = Thunk::Map {
            f: Value::Nil,
            inputs: Vec::new(),
        };
        let _ = super::step_thunk_rest(&mut eval, thunk);
    }

    #[test]
    fn chunked_parts_rejects_unbounded_range() {
        // Only `chunked-seq?` on unbounded ranges is pinned
        // behaviorally; the projector `None` is structural (the
        // CCE from-name stays the range slice's business).
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let r = RangeValue::new(RangeNum::Int(0), None, RangeNum::Int(1), true, None);
        let v = Value::Extension(Arc::new(r));
        assert!(super::chunked_parts(&mut eval, &v).unwrap().is_none());
    }

    #[test]
    fn chunked_pending_parts_forces_foreign_thunk_to_empty() {
        // Defensive: only `ChunkHead`/`Chain` thunks sit under a
        // `ChunkedCons` kind in production; anything else forces
        // one step (empty here, so gone).
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let v = PendingSeq::pending_chunked_cons(Thunk::FileSeq { stack: Vec::new() });
        let p = as_pending(&v).expect("chunked cons builds a pending");
        assert!(
            super::chunked_pending_parts(&mut eval, p)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn chunked_pending_parts_reads_forced_foreign_thunk() {
        // Defensive, forcing to a chunk: items plus the seq tail
        // (a missing path walks a singleton leaf — no fixture).
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let v = PendingSeq::pending_chunked_cons(Thunk::FileSeq {
            stack: vec!["/definitely/not/an/s12/dir".to_owned()],
        });
        let p = as_pending(&v).expect("chunked cons builds a pending");
        let (chunk, tail) = super::chunked_pending_parts(&mut eval, p)
            .unwrap()
            .expect("forced chunk reads");
        assert_eq!(chunk.items_arc().len(), 1);
        assert!(tail.is_none());
    }

    #[test]
    fn chunked_pending_parts_reads_chain_thunk() {
        // A `Chain` under `ChunkedCons` reads its head as the
        // chunk without forcing (the `with_meta_fresh` re-wrap
        // shape); the tail seqs behind it.
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let v = PendingSeq::pending_chunked_cons(Thunk::Chain {
            head: vec![Value::Int(1), Value::Int(2)],
            tail: Value::List(List::from_vec(vec![Value::Int(3)])),
        });
        let p = as_pending(&v).expect("chunked cons builds a pending");
        let (chunk, tail) = super::chunked_pending_parts(&mut eval, p)
            .unwrap()
            .expect("chain head reads");
        assert_eq!(chunk.items_arc()[..2], [Value::Int(1), Value::Int(2)]);
        assert!(tail.is_some());
    }

    #[test]
    fn chunked_pending_parts_empty_is_gone() {
        // Defensive: `ChunkedCons` cells never empty in production
        // (heads are never empty); an empty one reads as gone.
        let mut eval = crate::eval::Evaluator::new_bare("user");
        let p = pending_seq(SeqState::Empty, LazyKind::ChunkedCons);
        assert!(
            super::chunked_pending_parts(&mut eval, &p)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn chunked_realized_parts_empty_shape_is_gone() {
        // Defensive: no constructor flags an empty shape chunked;
        // such a seq reads as gone.
        let s = RealizedSeq::flagged(Seq::empty(), true, true, SeqKind::Chunked);
        assert!(super::chunked_realized_parts(&s).is_none());
    }

    #[test]
    fn drop_thunk_link_single_input_variants() {
        // Each single-`input` variant links chain-ward at its
        // input (the drop unwind follows exactly one link).
        let input = Value::Int(7);
        for thunk in [
            Thunk::Interpose {
                sep: Value::Int(0),
                input: input.clone(),
                first: true,
            },
            Thunk::DropWhile {
                f: Value::Nil,
                input: input.clone(),
            },
            Thunk::PartitionBy {
                f: Value::Nil,
                input: input.clone(),
            },
            Thunk::PartitionRun {
                f: Value::Nil,
                fv: Value::Nil,
                input: input.clone(),
            },
            Thunk::Dedupe {
                last: None,
                input: input.clone(),
            },
            Thunk::KeepIndexed {
                f: Value::Nil,
                input: input.clone(),
                idx: 0,
            },
        ] {
            assert_eq!(drop_thunk_link(thunk), Some(input.clone()));
        }
    }

    #[test]
    fn chunked_realized_parts_splits_eager_shape() {
        // Defensive: no constructor flags an eager shape chunked;
        // projectors split the first 32 items off instead.
        let items: Vec<Value> = (0..40).map(Value::Int).collect();
        let s = RealizedSeq::flagged(
            Seq::eager(items).expect("non-empty"),
            true,
            true,
            SeqKind::Chunked,
        );
        let (chunk, tail) = super::chunked_realized_parts(&s).expect("non-empty splits");
        assert_eq!(chunk.items_arc().len(), 32);
        assert!(tail.is_some());
    }
}
