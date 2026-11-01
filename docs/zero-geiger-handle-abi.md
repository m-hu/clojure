# Z3: Handle-table FFI — the honest zero-geiger design (SPEC ONLY)

Status: design spec. No implementation. No code changes on this branch —
this file is the only deliverable.

Parent: [issue #191](https://github.com/hhh-pa/clojure/issues/191)
("Drive complexity + geiger baselines to zero", FFI note: literal 0
requires deleting the pointer FFI). Track Z3 answers that note: it
designs the replacement ABI that keeps AOT compilation while taking
geiger to literal zero.

## 1. Problem statement

`rust/crates/clj-rt/src/abi.rs` crosses raw `*mut Value` pointers
(`Box::into_raw` / `Box::from_raw`, `CStr::from_ptr`,
`slice::from_raw_parts`, raw derefs). Every crossing is inherently
`unsafe`. Shrinking, auditing, or macro-hiding that unsafety lowers the
counter but can never zero it: the only honest zero is an ABI in which
no pointer crosses the boundary at all.

This spec designs that ABI: **u64 handles** (generational slab IDs with
tagged-pointer inline scalars) plus a process-global synchronized
table, with safe `extern "C"` shims doing integer ops plus table
lookup. Integer-only signatures, zero `unsafe` blocks workspace-wide.

## 2. Surveyed surface (measured, this worktree @ `759045a1`)

### 2.1 `abi.rs` shim inventory (328 lines)

| Class | Count | Symbols |
|---|---|---|
| Safe-signature ctors (still `unsafe` inside via `give` → `Box::into_raw`) | 5 | `rt_nil` `rt_bool` `rt_int` `rt_double` `rt_vec_new` |
| Named `pub unsafe extern "C"` shims | 9 | `rt_str` `rt_keyword` `rt_vec_push` `rt_argv` `rt_clone` `rt_drop` `rt_is_truthy` `rt_print` `rt_arity_error` |
| Macro prim shims (`def_prim!`, `pub unsafe extern "C"`) | 13 | `rt_add` `rt_sub` `rt_mul` `rt_quot` `rt_lt` `rt_le` `rt_gt` `rt_ge` `rt_eq` `rt_count` `rt_nth` `rt_get` `rt_conj` |
| Private `unsafe fn` helpers | 5 | `bytes` `text` `take` `take_args` `prim_call` |
| **Total exported `rt_*` symbols** | **27** | (`prim_symbol` maps `Prim` → 13 of them; `lib.rs` re-exports all 27) |

Unsafety primitives used: `Box::into_raw` (1 site, `give`),
`Box::from_raw` (1 site, `take`), `slice::from_raw_parts` (2 sites:
`bytes`, `take_args`), `&mut *vec` deref (`rt_vec_push`),
`argv.offset` + `CStr::from_ptr` (`rt_argv`), `(*value).clone`
(`rt_clone`). Contract today: pointers non-null/valid, args consumed,
returns owned, null-on-error; violation is instant UB (module docs).

### 2.2 `lower.rs` call sites (890 lines)

- `Decls`: 16 `add_function` decl expressions covering 28 symbols
  (13 prims + nil/bool/int/double/str/keyword/vec_new/vec_push/
  clone/drop/argv/print/arity_error/is_truthy/exit).
- ~25 emit sites funnel through 5 helpers: `call_ptr`, `call_void`,
  `call_byte`, `call_drop`, plus 1 direct `build_call`
  (`rt_arity_error`). Every value in generated IR is an opaque `ptr`:
  params, returns, frame slots (`alloca ptr`), phis (`phi ptr`),
  arg arrays (`alloca [N x ptr]` + GEP + store).
- 2 non-FFI `unsafe` expressions: `build_in_bounds_gep` × 2 in
  `pack_args` (inkwell API is `unsafe fn` there). A handle ABI with no
  arg arrays deletes both with no replacement (loads/stores/allocas
  and `build_gep` are safe; after this redesign no GEP is needed at
  all — §4.4).
- `null_check` (fallible-call null → `exit(1)`) becomes an integer
  compare against the 0 sentinel (§4.2).

### 2.3 `Value` shapes (`clj-value/src/value.rs`, `pub enum Value`, ~45 variants)

- Trivially inlinable into a u64 payload: `Nil`, `Bool`, `Int`
  (full i64 fits), `Double` (bitcast), `Char` (fits in 32 bits).
- Everything else is already behind `Arc`/shared-cell indirection
  (`Str`, `BigInt`, `Vector`, `Map`, `Atom`, `Future`, …) or small
  fixed payloads (`Uuid`, `Instant`): `Value::clone` is a cheap
  Arc-bump, so table slots storing owned `Value`s keep clone/drop
  semantics identical to today's box clone/drop. No layout change to
  `Value` is required.
- Identity-equality types (`identical?` via `Arc::ptr_eq` / ids) are
  unaffected: the table stores the same `Value`s, only the outer box
  changes from `Box` to slab slot.

### 2.4 Sync precedents (the table has in-repo models)

- `clj-rt/src/runtime.rs`: `static RUNTIME: LazyLock<Mutex<Runtime>>`,
  single lock, no nesting, panic-catch inside the lock scope. Built
  binaries are single-threaded; unit tests share the process.
- `clj-value/src/executor.rs`: `Mutex<Option<Sender>>` + `mpsc`
  channel + one pool thread. `collmisc.rs` (`SequeRx`):
  `Arc<Mutex<mpsc::Receiver>>>` + feeder thread. `channel.rs`:
  `Mutex` + `Condvar`. Precedent is uniform: **`Mutex` + `mpsc`,
  no lock-free structures, no extra sync deps.** The handle table
  follows suit (§4.3).

### 2.5 What "literal zero" must cover (beyond `abi.rs`)

Geiger counts `unsafe` blocks/expressions in workspace crates (deps
excluded). Deleting `abi.rs` unsafety alone does NOT reach 0:

| Remaining site | Owner | Z3 treatment |
|---|---|---|
| `lower.rs` 2× `build_in_bounds_gep` | Z3 | deleted by §4.4 (no arg arrays) |
| `clj-rt/tests/support`: `into_raw`/`from_raw` + ~26 `unsafe` test call sites (`prims.rs`, `values.rs`, `print_arity.rs`) | Z3 | rewritten to safe handle calls; test `unsafe` → 0 |
| Edition-2024 `set_var`/`remove_var` in tools-deps config tests (Gap-124, +10) | G1 | dependency: Z3 lands on a tree where G1 (subprocess form) has removed these |
| `geiger-baseline.json` (`unsafe_total: 252` in this worktree; issue text cites 268 on master — re-measure at implementation) + rollup double-count | G1/Z3 | ledger split per G1; Z3 zeroes the file |

`#[unsafe(no_mangle)]` is an attribute, not counted unsafe — it stays.
`extern "C" fn` with integer-only signatures and safe bodies contributes
nothing to geiger. (Caveat carried from #191: macro-generated shims are
a counter blind spot — the new `def_prim!` expansion must be eyeball-
reviewed to confirm it contains no `unsafe`, regardless of the count.)

## 3. Design

### 3.1 Handle encoding (u64, 0 reserved)

- `0` = null/error sentinel (replaces null-pointer returns; generated
  code compares `!= 0` exactly where `null_check` runs today).
- Nonzero handles split by tag bits into:
  - **Inline scalars** (no table traffic): `Nil`, `Bool`, `Int`,
    `Double` (bits), `Char`. Pure integer encode/decode in the shim.
    Rationale: these are the hottest shapes (literals, counters,
    truthiness tests); inlining makes them *cheaper* than today (no
    `Box::new`/`free` per value).
  - **Table handles**: generational index (`index || generation`).
    Generation mismatch (use-after-drop, double-drop) = deterministic
    abort/error, never UB — a strict improvement over today's
    use-after-free UB. Slot holds `{ value: Value, refcount: u32,
    generation: u32 }`.
- Refcounted slots preserve today's affine discipline 1:1:
  `rt_clone` = bump (scalars: identity, no-op), `rt_drop` = dec,
  free at zero. Every `drop_slots` / `call_drop` / let-old-drop emit
  site keeps its position; only the type changes (`ptr` → `i64`).

### 3.2 Shim signatures (all integers, all safe `extern "C"`)

Pointers are banned from every signature — including the byte slices
(`rt_str`, `rt_keyword`, prim error paths, `rt_print` arg arrays) and
`argv`, whose derefs are `unsafe` no matter how audited. The rules:

- Value traffic: `u64` in, `u64` out. `rt_nil/bool/int/double` become
  pure encoders; `rt_clone/drop/is_truthy/vec_push`, all 13 prims take
  and return `u64`.
- Strings (literals, keyword parts, error display/path): today's
  `(ptr, len)` cannot survive — any raw-slice deref is `unsafe`, and
  int→pointer casts are equally `unsafe`. Replacement: **chunked
  scalar construction**. Codegen lowers a literal to `rt_str_new()`
  + N× `rt_str_chunk(h: u64, chunk: u64, nbytes: u8)` (8 bytes per
  call, scalars only), or one call for len ≤ 8. Same for keyword
  parts and the baked path (path interned once at startup → `u64`
  str-id; prim call sites then pass `(prim_args_acc, str_id, line,
  col)`, all integers). IR cost: ~1 call per 8 literal bytes;
  literals are rare per program — acceptable.
- Arg arrays (prims, `print`): replaced by an **accumulator**:
  `rt_args_new() → u64`, `rt_args_push(acc: u64, h: u64)`,
  consumed by the prim/print call. No `alloca [N x ptr]`, no GEP,
  no `take_args` deref chain. Args resolve to `Vec<Value>` under one
  table lock hold (§4.3).
- `argv`: `rt_argv(argc: i32, argv: ptr)` cannot survive either.
  Replacement: **`rt_argv() → u64`, zero params**, reading
  `std::env::args_os()` (lossy decode, skip index 0 — same
  T1192 FX1 semantics). Generated `main` ignores its C `argc/argv`.
  Testability: the list-building logic lives in a plain
  (non-FFI) `fn argv_from<I>(args: I) -> Value` unit-tested
  directly; the FFI shim is a 3-line safe wrapper.
- `usize` widths become explicit `u64`/`u32` everywhere (the 64-bit
  `compile_error!` gate in `lib.rs` can then optionally be dropped —
  recommended follow-up, not required).
- `exit` decl stays as-is (inkwell external-call emission is safe API).

New-symbol sketch (final names at implementation; ~30 symbols replace
~27): `rt_nil/bool/int/double` (encoders), `rt_str_new/chunk`,
`rt_keyword_new/push_part` (or keyword-via-two-str-handles),
`rt_vec_new/push`, `rt_clone/drop/is_truthy`, `rt_args_new/push`,
`rt_print`(acc), 13 prims `(acc, str_id, line, col)`,
`rt_arity_error`(str_ids…), `rt_argv()`.

### 3.3 The table: `Mutex<Slab>` folded into `Runtime`

```text
static RUNTIME: LazyLock<Mutex<Runtime>>   // unchanged shape
struct Runtime {
    eval: Evaluator,                       // unchanged
    prims: [Arc<NativeImpl>; 13],          // unchanged
    handles: Slab<Slot>,                   // NEW: the handle table
    strings: Slab<Arc<str>>,               // NEW: baked path / str-ids
}
```

- **Fold into `Runtime`, don't add a second global.** The codebase's
  whole concurrency argument is "a single lock with no nesting cannot
  deadlock" (`runtime.rs` docs). A separate `HANDLE_TABLE` global would
  introduce lock ordering (`table → runtime` vs `runtime → table`)
  across prim calls; folding in keeps exactly one lock and the existing
  argument intact. Cost: table ops serialize with prim dispatch — fine,
  because (a) built binaries are single-threaded, (b) inline scalars
  never touch the lock, (c) uncontended `Mutex` is ~20ns.
- `Slab`: `Vec<Slot>` + freelist (hand-rolled ~30 lines, or the `slab`
  crate if the workspace accepts the dep — decide at implementation;
  hand-rolled keeps the dep tree unchanged).
- No sharding, no lock-free structures in v1. The generational-index
  encoding reserves the option: a future sharded (`[Mutex<Shard>; N]`)
  or epoch-based table fits the same u64 format. Shard only on measured
  contention (none expected: single-threaded binaries + scalar bypass).
- Leak detection: debug/test-mode live-handle counter + a test
  asserting every FFI test ends at its entry count (replaces today's
  implicit leaks-are-UB discipline with an explicit check).

### 3.4 Per-op overhead (estimate)

| Op | Today | Handle design |
|---|---|---|
| Scalar literal (nil/bool/int/double/char) | `Box::new` + deref + `free` (~50–100ns, allocator) | integer encode, no lock, no alloc (~1ns) — **faster** |
| Heap value create/clone/drop | `Box::new`/deref/`free` | 1 uncontended mutex hold + slab index + refcount (~25ns) — **neutral to faster** (no allocator traffic) |
| Prim call (N args) | `take_args` deref chain + 1 `RUNTIME` hold | 1 `RUNTIME` hold resolving acc + args + dispatch — **same lock count as today** |
| String literal (len L) | 1 call + slice deref | 1 + ceil(L/8) scalar calls — **more calls, zero unsafety**; literals are cold |
| Truthiness/arity paths | deref / slice deref | integer ops — **faster** |

Net: single-threaded binaries should be as fast or faster (allocator
traffic drops); multithreaded unit-test processes serialize slightly
more. Implementation must add a micro-bench (compile+run a loop-heavy
corpus program, wall-clock before/after) to confirm — no perf gate
exists today, so this is new measurement, not a regression gate.

## 4. Migration strategy: brief dual-entry, then flag-day flip

Pure flag day (one PR touching `abi.rs` + `lower.rs` + all tests +
snapshots) is reviewable (~1500 lines) but unvalidatable mid-flight.
Pure dual-entry (pointer + handle shims coexisting) doubles the test
surface and keeps geiger > 0 indefinitely. Recommended hybrid, one
tracking issue, three stacked commits:

1. **Additive**: handle table + safe shims alongside the pointer ABI
   (new symbols, e.g. `rt_h_*` or final names with coexisting old
   ones). Geiger unchanged. New unit tests pin encode/decode tables,
   generation-mismatch aborts, refcount/leak-counter behavior.
2. **Flip**: codegen emits handle IR; all `call ptr @rt_*` assertions
   → `call i64 @rt_*`; FFI integration tests rewritten safe;
   `build_exe` end-to-end proves behavioral identity. **Delete the
   pointer shims in the same commit.** Geiger drops to the floor in
   one step; no commit ships two live ABIs to users.
3. **Retire**: zero `geiger-baseline.json` (with G1's ledger split),
   drop the 64-bit gate if widths are explicit, remove string-global
   interning leftovers (`Lower::strings`, `const_str`, `path_const`
   → str-id interning).

`#![forbid(unsafe_code)]` on `clj-rt` and `clj-aot` (and the test
targets via config) lands in step 2 — the ratchet that makes zero
permanent. Workspace-wide `forbid` follows once G1's env-mutation
sites are gone (dependency, §2.5).

## 5. Corpus / IR snapshot impact

- `lower.rs` unit tests (~15 tests): every `call ptr @rt_*`,
  `phi ptr`, `define ptr @cljfn`, `call void @rt_drop`,
  `call i8 @rt_is_truthy` pin changes type/shape; `pack_args` tests
  become accumulator-sequence pins. Mechanical, contained in 1 file.
- `clj-aot/tests/emit.rs` (~77 asserts): same mechanical rewrite of
  IR-substring pins.
- `clj-aot/tests/build_exe.rs` (~22 asserts): compiles real binaries,
  links, runs, checks output — **must pass unchanged in behavior**;
  it is the differential oracle for the whole migration. No assertion
  changes expected beyond symbol-name updates if names change.
- `clj-rt` FFI tests (`prims.rs`, `values.rs`, `print_arity.rs` +
  `tests/support`): full rewrite from box-passing to handle-passing;
  `support::inbox/unbox/render` collapse to `u64` helpers. Test
  `unsafe` count 26+ → 0.
- No `.clj` corpus impact: the Subset language, planner, and runtime
  semantics are untouched — only the lowering target type changes.
- Docs: `abi.rs` module docs (the UB-contract essay), `lib.rs`
  ownership docs, S8 plan docs referencing `*mut Value` need rewrites
  to the handle contract (sentinel-0, generation aborts, str-ids).

## 6. Risks

1. **String-chunking bloat** (large literals → many calls). Mitigate:
   chunked calls are linear and cold; cap literal size with a
   planner error if pathological programs appear.
2. **Handle discipline bugs** (leak/double-drop) become silent logic
   errors instead of loud UB. Mitigate: generations (deterministic
   abort on stale use), debug refcount assertions, test-mode leak
   counter, `build_exe` differential.
3. **Lock-hold duration**: arg resolution + dispatch under one hold
   lengthens the critical section vs today. Single-threaded binaries
   unaffected; contention only across test threads. Measure; the u64
   format reserves sharding if ever needed.
4. **inkwell safe-API coverage**: post-change builder calls must all
   be safe-fn. Audit at implementation: `build_alloca/load/store/
   call/return/branch/phi/int_compare` (safe today) + no GEP at all.
   If any needed inkwell API is `unsafe fn`, that site caps the floor
   above 0 — verify before committing to the design (spike: compile
   one handle-shaped module with `#![forbid(unsafe_code)]`).
5. **`std::env` in `rt_argv`**: reads process state; behavior under
   `-main`-less binaries unchanged (no call emitted). Unit tests use
   the `argv_from` helper, not the shim.
6. **Baseline-number drift**: this worktree's `unsafe_total` (252)
   ≠ issue's master figure (268). Re-run `cargo geiger` workspace-
   wide at implementation kickoff and enumerate every site; the §2.5
   table must reconcile to the run, not to either stale number.
7. **Macro blind spot**: new `def_prim!` expansion eyeball-reviewed
   for `unsafe` regardless of counter (per #191 fraud note — gaming
   the counter via macros is forbidden; the design has nothing to
   hide since the macro emits safe fns).

## 7. Cost estimate

| Phase | Size | Notes |
|---|---|---|
| Table + safe shims + unit tests | ~500 lines | `abi.rs` rewrite, table, generation/refcount logic |
| Codegen retarget | ~300 lines touched | `lower.rs` decls/emit/null_check/pack→acc |
| Test rewrites (rt + aot) | ~400 lines touched | mechanical, type-level |
| Docs + baseline + `forbid` | ~150 lines | module docs, S8 refs, ledger |
| **Total** | **~1300–1500 lines** | 2–4 stacked PRs, each CI-green |

No new dependencies required (hand-rolled slab; std-only). No `Value`
layout changes. No planner/corpus changes.

## 8. Open questions (for implementation kickoff)

1. Final symbol names: keep `rt_*` (churn in every snapshot either
   way) or `rt_h_*` during dual-entry then rename? Recommendation:
   final names from step 1 to avoid a second snapshot churn.
2. `slab` crate vs hand-rolled freelist? Recommendation: hand-rolled
   (~30 lines, zero dep review).
3. Inline `Double`? Bitcast is easy, but NaN-boxing-adjacent tricks
   are NOT proposed — plain tag + 64-bit payload. Confirm tag budget:
   3 tag bits + 61-bit payload covers index space generously.
4. Drop the 64-bit `compile_error!` gate once widths are explicit u64?
   Recommendation: yes, in step 3.
5. Micro-bench harness: wall-clock `build_exe`-style loop program, or
   criterion on shims? Recommendation: wall-clock binary timing (tests
   the real path); criterion optional.
6. G1 sequencing: does Z3 block on G1's env-mutation cleanup for the
   literal-0 commit, or does Z3 land first with `unsafe_total` at the
   G1 remainder? Recommendation: Z3's step-2 commit message records
   the achieved number; the file hits 0 only after both land.

## 9. Feasibility verdict

**Feasible.** The design is complete: every pointer crossing in the
27-symbol ABI has a named integer-only replacement (§3.2), the table
fits the codebase's existing single-`Mutex` concurrency argument
(§3.3), per-op cost is neutral-to-better (§3.4), the migration is
validatable at every step with `build_exe` as oracle (§4), and the
residual risks each have a check (§6). The one must-verify spike
before committing: an inkwell-emitted handle-shaped module compiled
under `#![forbid(unsafe_code)]` (risk 4). Estimated ~1300–1500 lines
over 2–4 stacked PRs, std-only, no `Value`/planner/corpus changes.
