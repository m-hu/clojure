// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S-instream readers: `*in*`-bindable stream readers plus the
//! `read`/`read-line` natives (issue #36).
//!
//! Both reader kinds print as
//! `clojure.lang.LineNumberingPushbackReader` (the JVM class of the
//! `*in*` root and the `with-in-str` binding alike); the id is
//! port-owned like `Writer`'s. `read` and `read-line`
//! share one persistent parser, so interleaving matches the JVM
//! and line numbers survive across calls.

#[cfg(test)]
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::sync::{Arc, Mutex};

use clj_reader::{
    EvalReadError, FiredTag, ReadCond, ReadHooks, Reader, ReaderError, ReaderOpts, ResolverOp, Span,
};
use clj_value::{ExtensionValue, Keyword, Map, Symbol, ThrowClass, Throwable, Value, new_map};

use crate::core::cce_err;
use crate::error::NativeError;
use crate::eval::Evaluator;
use crate::ns::{
    NativeFnPtr, check_arity, core_var, deref_core, do_resolve, intern_routed, signal_to_native,
    sym_str, throw_class_for, throwable_to_native, var_value,
};

/// The JVM class of both the `*in*` root and the `with-in-str`
/// bindings (probes `in-class`, `wis-class`).
const READER_CLASS: &str = "clojure.lang.LineNumberingPushbackReader";

/// A `*in*`-bindable stream reader: process stdin or an in-memory
/// string. Identity is the sequence `id`.
pub(crate) struct StreamReader {
    id: u64,
    state: Arc<Mutex<StreamState>>,
}

impl std::fmt::Debug for StreamReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StreamReader({})", self.id)
    }
}

/// The shared read position: one persistent parser plus the live
/// stdin source (`None` for string readers; stdin stays `Some`
/// and EOF is detected per-read).
struct StreamState {
    parser: Reader,
    stdin: Option<BufReader<Box<dyn Read + Send>>>,
    /// Latched by `.close` (gap-120 slice 3: `with-open` over
    /// `clojure.java.io` readers); reads past it report the JDK
    /// "Stream closed" text.
    closed: bool,
    /// Scripted input lines (tests only): drained before live
    /// stdin, so retry tests never touch process stdin.
    #[cfg(test)]
    script_lines: VecDeque<String>,
}

impl ExtensionValue for StreamReader {
    fn class_name(&self) -> String {
        READER_CLASS.to_owned()
    }

    fn print(&self) -> String {
        format!(
            "#object[{READER_CLASS} 0x{:x} \"{READER_CLASS}@{:x}\"]",
            self.id, self.id
        )
    }

    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
        other
            .as_any()
            .downcast_ref::<StreamReader>()
            .is_some_and(|o| o.id == self.id)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_u64(self.id);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Downcast a value to a stream reader, if it is one.
pub(crate) fn as_reader(v: &Value) -> Option<&StreamReader> {
    match v {
        Value::Extension(e) => e.as_any().downcast_ref::<StreamReader>(),
        _ => None,
    }
}

/// `close` on a stream reader value (`None` for non-readers —
/// the miss shapes; void, so never a bean key). Latches closed
/// (second closes no-op, like the JVM); gap-120 slice 3.
pub(crate) fn close_reader_value(
    e: &Arc<dyn ExtensionValue>,
) -> Option<Result<Value, NativeError>> {
    let r = e.as_any().downcast_ref::<StreamReader>()?;
    Some(close_stream_reader(r).map(|()| Value::Nil))
}

/// Build the standard reader UTF-8 IO error.
fn utf8_io_err() -> NativeError {
    NativeError::new(
        ThrowClass::IOException,
        "stream did not contain valid UTF-8".to_owned(),
    )
}

/// Append `s` onto `out` with `\r\n` and a lone `\r` normalized to `\n`.
fn push_normalized(out: &mut String, s: &str) {
    let mut rest = s;
    while let Some(i) = rest.find('\r') {
        out.push_str(&rest[..i]);
        out.push('\n');
        rest = &rest[i + 1..];
        rest = rest.strip_prefix('\n').unwrap_or(rest);
    }
    out.push_str(rest);
}

/// Drain the live stdin reader into `out`: one buffer grown as the input
/// arrives and never zero-filled, validated once. Most inputs are tens
/// of bytes, and the previous zeroed 8 KiB first chunk cost ~8K
/// instructions per program (#271). std's `read_to_end` probes small
/// inputs on the stack, retries interrupted reads, and grows large
/// buffers by `realloc`.
fn slurp_buf_reader(src: &mut dyn Read, out: &mut String) -> Result<(), NativeError> {
    let mut buf = Vec::new();
    src.read_to_end(&mut buf)
        .map_err(|e| NativeError::new(ThrowClass::IOException, e.to_string()))?;
    if out.is_empty() && !buf.contains(&b'\r') {
        *out = String::from_utf8(buf).map_err(|_| utf8_io_err())?;
        return Ok(());
    }
    let text = std::str::from_utf8(&buf).map_err(|_| utf8_io_err())?;
    out.reserve_exact(text.len());
    push_normalized(out, text);
    Ok(())
}

/// Drain a stream reader fully (issue #201 S8): parser remainder
/// plus any buffered/live feed without intermediate char expansion.
/// Matches upstream `slurp`, which `with-open`s the reader: the stream
/// latches closed on success, even stdin.
pub(crate) fn slurp_reader(r: &StreamReader) -> Result<String, NativeError> {
    check_reader_open(r)?;
    let mut out = String::new();
    {
        let mut state = lock_state(r);
        out.push_str(&state.parser.drain_to_string());
        #[cfg(test)]
        while let Some(line) = state.script_lines.pop_front() {
            state.parser.extend(&line);
            out.push_str(&state.parser.drain_to_string());
        }
        if let Some(src) = state.stdin.as_mut() {
            slurp_buf_reader(src, &mut out)?;
        }
        // Closed like `.close`; the parser was just drained, so it
        // stays (a fresh one costs ~3K instructions to build, #271).
        state.closed = true;
    }
    Ok(out)
}

/// Latch a stream reader closed (idempotent).
fn close_stream_reader(r: &StreamReader) -> Result<(), NativeError> {
    let mut state = lock_state(r);
    state.closed = true;
    state.parser = Reader::new("", &ReaderOpts::default());
    Ok(())
}

/// Reads past `.close` report the JDK text (gap-120 slice 3).
/// Shared with `load-reader`, which takes the parser directly.
pub(crate) fn check_reader_open(r: &StreamReader) -> Result<(), NativeError> {
    if lock_state(r).closed {
        return Err(crate::io::stream_closed());
    }
    Ok(())
}

/// A fresh stdin reader value (the `*in*` root): an empty parser
/// fed line by line from process stdin.
pub(crate) fn stdin_reader(eval: &mut Evaluator) -> Value {
    Value::Extension(Arc::new(StreamReader {
        id: eval.next_id(),
        state: Arc::new(Mutex::new(StreamState {
            parser: Reader::new("", &ReaderOpts::default()),
            stdin: Some(BufReader::with_capacity(
                64 * 1024,
                Box::new(std::io::stdin()),
            )),
            closed: false,
            #[cfg(test)]
            script_lines: VecDeque::new(),
        })),
    }))
}

/// A fresh string reader value (`with-in-str`): the parser starts
/// over `s` (line numbers on, for ex-data spans and form meta).
pub(crate) fn string_reader(eval: &mut Evaluator, s: &str) -> Value {
    Value::Extension(Arc::new(StreamReader {
        id: eval.next_id(),
        state: Arc::new(Mutex::new(StreamState {
            parser: Reader::new(s, &ReaderOpts::default()),
            stdin: None,
            closed: false,
            #[cfg(test)]
            script_lines: VecDeque::new(),
        })),
    }))
}

/// The locked stream state (a poisoned mutex still yields its
/// data, like `Writer`'s buffer).
fn lock_state(reader: &StreamReader) -> std::sync::MutexGuard<'_, StreamState> {
    reader.state.lock().unwrap_or_else(|e| e.into_inner())
}

/// Pull one buffered fill from `src` as text; `None` at EOF
/// (generic so unit tests drive it with in-memory buffers instead
/// of process stdin).
fn pull_chunk<R: BufRead>(src: &mut R) -> Result<Option<String>, NativeError> {
    let io_err = |e: std::io::Error| NativeError::new(ThrowClass::IOException, e.to_string());
    let buf = src.fill_buf().map_err(io_err)?;
    if buf.is_empty() {
        return Ok(None);
    }
    // Take the longest valid UTF-8 prefix of the buffered fill; a
    // char split across the fill boundary stays buffered for the
    // next pull (or, when it is the whole fill, the line reader
    // reads ahead to complete it). Invalid bytes report exactly
    // what `BufRead::read_line` reported.
    let (text, take) = match std::str::from_utf8(buf) {
        Ok(s) => (s.to_owned(), buf.len()),
        Err(e) => {
            let up = e.valid_up_to();
            if e.error_len().is_none() && up > 0 {
                (String::from_utf8_lossy(&buf[..up]).into_owned(), up)
            } else if e.error_len().is_none() {
                let mut chunk = String::new();
                let n = src.read_line(&mut chunk).map_err(io_err)?;
                return Ok(if n == 0 { None } else { Some(chunk) });
            } else {
                return Err(io_err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stream did not contain valid UTF-8",
                )));
            }
        }
    };
    src.consume(take);
    Ok(Some(text))
}

/// Pull one line from `src` into the parser; `false` at EOF (the
/// stdin feed for `read-line`, which never needs the read-retry
/// snapshot).
fn pull_line<R: BufRead>(src: &mut R, parser: &mut Reader) -> Result<bool, NativeError> {
    match pull_chunk(src)? {
        Some(chunk) => {
            parser.extend(&chunk);
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Whether another line may still arrive (live stdin; tests also
/// feed scripted lines). `run_read` snapshots the parser only
/// then, so string reads stay zero-overhead.
fn may_grow(state: &StreamState) -> bool {
    #[cfg(test)]
    if !state.script_lines.is_empty() {
        return true;
    }
    state.stdin.is_some()
}

/// Fetch the next input line without touching the parser:
/// scripted test lines first, then live stdin when present, `None`
/// at EOF (string readers and exhausted stdin). The `read` retry
/// restores its snapshot before extending with this, so a
/// truncated head survives. Combinators, not a `match`: the stdin
/// dispatch stays on lines string-reader tests execute (100% line
/// gate, like `more_input`).
fn next_chunk(state: &mut StreamState) -> Result<Option<String>, NativeError> {
    #[cfg(test)]
    if let Some(line) = state.script_lines.pop_front() {
        return Ok(Some(line));
    }
    state.stdin.as_mut().map_or(Ok(None), pull_chunk)
}

/// One full line from a stream reader, feeding stdin fills until
/// a terminator is buffered (a buffered partial line is not the
/// EOF tail while more input may arrive); `None` at EOF.
fn stream_read_line(state: &mut StreamState) -> Result<Option<String>, NativeError> {
    loop {
        if state.parser.has_complete_line() {
            return Ok(state.parser.read_line());
        }
        if !more_input(state)? {
            return Ok(state.parser.read_line());
        }
    }
}

/// Feed one stdin line into the parser when stdin is live;
/// `false` for string readers and exhausted stdin (the `map`
/// closure shares its line with the chain, so string-reader tests
/// cover the dispatch while `pull_line` unit tests cover the feed
/// itself). Tests drain scripted lines first.
fn more_input(state: &mut StreamState) -> Result<bool, NativeError> {
    state.parser.compact_stream();
    #[cfg(test)]
    if let Some(line) = state.script_lines.pop_front() {
        state.parser.extend(&line);
        return Ok(true);
    }
    Ok(state
        .stdin
        .as_mut()
        .map(|src| pull_line(src, &mut state.parser))
        .transpose()?
        .unwrap_or(false))
}

/// Whether a read failure is an EOF (clean or truncated): stdin
/// readers feed another line and retry these, and throw the rest.
fn is_eof_error(e: &ReaderError) -> bool {
    matches!(
        e,
        ReaderError::Eof { .. }
            | ReaderError::EofDelimited { .. }
            | ReaderError::EofChar { .. }
            | ReaderError::EofString { .. }
            | ReaderError::EofRegex { .. }
    )
}

/// The `ReaderException` ex-data map for a span (line/column under
/// `clojure.error`).
fn reader_ex_data(span: Span) -> Value {
    let line_key = Value::Keyword(Keyword::new(Some("clojure.error"), "line"));
    let col_key = Value::Keyword(Keyword::new(Some("clojure.error"), "column"));
    Value::Map(
        new_map()
            .insert(line_key, Value::Int(i64::from(span.line)))
            .insert(col_key, Value::Int(i64::from(span.col))),
    )
}

/// A read failure as a native error: `recursive?` rethrows the raw
/// root (probe `r-recursive-outer`); otherwise a `ReaderException`
/// (`{root-class}: {root-message}`, ex-data line/column from the
/// span, the root attached as the cause).
pub(crate) fn map_reader_error(
    eval: &mut Evaluator,
    e: &ReaderError,
    recursive: bool,
) -> NativeError {
    let class = throw_class_for(e.jvm_class());
    let message = e.message();
    if recursive {
        // A raw rethrow still rides `#=`/resolver throwables
        // intact (data and cause survive, like `read-string`).
        if let ReaderError::EvalFailed {
            error: EvalReadError::Failed(t),
            ..
        } = e
        {
            return throwable_to_native(t);
        }
        return NativeError::new(class, message);
    }
    let span = e.span();
    let mut outer = NativeError::new(
        ThrowClass::ReaderException,
        format!("{}: {message}", class.jvm_name()),
    );
    outer.data = Some(reader_ex_data(span));
    // S7: a `#=`/resolver throwable rides as the cause intact (its
    // `ex-data` and cause chain survive the wrap, like the JVM).
    outer.cause = Some(match e {
        ReaderError::EvalFailed {
            error: EvalReadError::Failed(t),
            ..
        } => t.clone(),
        _ => Throwable {
            id: eval.next_id(),
            class,
            message: Some(Arc::from(message.as_str())),
            data: None,
            cause: None,
        },
    });
    outer
}

/// S7 read gates (issue #50): the `*reader-resolver*` protocol
/// is a 2-arity fn; anything else bound fails this cast (probes
/// `f-nonresolver*`).
const RESOLVER_CLASS: &str = "clojure.lang.LispReader$Resolver";

/// The seeded gate state for one read call.
pub(crate) struct ReadGateSeed {
    /// `*read-eval*` (load entries force true).
    pub read_eval: bool,
    /// `*suppress-read*` (live entry binding).
    pub suppress_read: bool,
    /// `*reader-resolver*` is bound to a fn.
    pub resolver_bound: bool,
}

/// Whether a bound `*reader-resolver*` can take consults: fns
/// (closures/natives ride `Extension`), vars (derefed), multifns.
fn is_resolver_fn(v: &Value) -> bool {
    matches!(v, Value::Extension(_) | Value::Var(_) | Value::MultiFn(_))
}

/// The resolver entry cast (S7): nil is unbound, fns pass,
/// anything else fails the `Resolver` cast — raw, like the JVM
/// entry check (probes `f-nonresolver*`, `h-*`).
fn check_resolver(eval: &Evaluator) -> Result<bool, NativeError> {
    match deref_core(eval, "*reader-resolver*") {
        Value::Nil => Ok(false),
        v if is_resolver_fn(&v) => Ok(true),
        v => Err(cce_err(&v, RESOLVER_CLASS)),
    }
}

/// Seed the gates from the entry bindings (S7; probes `g-*`,
/// `h-order*`): the resolver cast first, then the `:unknown`
/// refusal. A missing `*read-eval*` (bare evaluators) allows,
/// matching the true root.
pub(crate) fn seed_read_gates(eval: &Evaluator) -> Result<ReadGateSeed, NativeError> {
    let resolver_bound = check_resolver(eval)?;
    let suppress_read = deref_core(eval, "*suppress-read*").is_truthy();
    let Some(_) = core_var(eval, "*read-eval*") else {
        return Ok(ReadGateSeed {
            read_eval: true,
            suppress_read,
            resolver_bound,
        });
    };
    let read_eval = deref_core(eval, "*read-eval*");
    if read_eval == Value::Keyword(Keyword::new(None, "unknown")) {
        return Err(NativeError::new(
            ThrowClass::RuntimeException,
            "Reading disallowed - *read-eval* bound to :unknown".to_owned(),
        ));
    }
    Ok(ReadGateSeed {
        read_eval: read_eval.is_truthy(),
        suppress_read,
        resolver_bound,
    })
}

/// Seed the gates for a load (S7; probes `m-*`): `*read-eval*` is
/// forced true (the binding is pushed separately, visibly);
/// suppression still seeds live. The resolver cast stays raw here —
/// callers report it `Compiler.load`-style (see [`executing_wrap`]).
pub(crate) fn seed_load_gates(eval: &Evaluator) -> Result<ReadGateSeed, NativeError> {
    Ok(ReadGateSeed {
        read_eval: true,
        suppress_read: deref_core(eval, "*suppress-read*").is_truthy(),
        resolver_bound: check_resolver(eval)?,
    })
}

/// A load-path resolver cast as `Compiler.load` reports it (S7;
/// probes `m-load-resolver`, `l-resolver-mid`): `Syntax error
/// executing at (line:col)`, caused by the cast.
pub(crate) fn executing_wrap(
    line: u32,
    col: u32,
    cce: &NativeError,
    outer_id: u64,
    cause_id: u64,
) -> crate::eval::SourceError {
    crate::eval::SourceError::Uncaught(Throwable {
        id: outer_id,
        class: ThrowClass::CompilerException,
        message: Some(Arc::from(format!(
            "Syntax error executing at ({line}:{col})."
        ))),
        data: None,
        cause: Some(Box::new(Throwable {
            id: cause_id,
            class: cce.class,
            message: Some(Arc::from(cce.message.as_str())),
            data: cce.data.clone().map(Box::new),
            cause: None,
        })),
    })
}

/// The live S7 hooks: `#=` application plus resolver consults,
/// backed by the calling evaluator.
pub(crate) struct LiveReadHooks<'a> {
    eval: &'a mut Evaluator,
}

impl<'a> LiveReadHooks<'a> {
    /// Borrow the calling evaluator for one read call.
    pub(crate) fn new(eval: &'a mut Evaluator) -> Self {
        Self { eval }
    }

    /// Lift a native error into a traveling `#=`/resolver failure
    /// (message, class, data, and cause all survive).
    fn lift(&mut self, e: NativeError) -> EvalReadError {
        EvalReadError::Failed(Throwable {
            id: self.eval.next_id(),
            class: e.class,
            message: Some(Arc::from(e.message.as_str())),
            data: e.data.clone().map(Box::new),
            cause: e.cause.clone().map(Box::new),
        })
    }

    /// One `#=` operand (JVM `EvalReader`, probes `c-*`/`d-*`):
    /// symbols name classes (the port has none), lists apply a
    /// resolved head to the raw operand forms, anything else is
    /// unsupported.
    fn apply_eval_read(&mut self, form: &Value) -> Result<Value, NativeError> {
        match form {
            Value::Symbol(s) => Err(class_not_found(&sym_str(s))),
            Value::List(l) => self.apply_eval_list(l),
            _ => Err(NativeError::new(
                ThrowClass::IllegalArgumentException,
                "Unsupported #= form".to_owned(),
            )),
        }
    }

    /// Apply a `#=(head args*)` list: bare `var` interns, `C.`
    /// heads need host classes, unknown qualified heads fail the
    /// class lookup, and resolved vars apply to the raw args.
    fn apply_eval_list(&mut self, l: &clj_value::List) -> Result<Value, NativeError> {
        let mut items = l.iter();
        let Some(head) = items.next() else {
            return Err(npe(
                "Cannot invoke \"clojure.lang.Symbol.equals(Object)\" because \"fs\" is null",
            ));
        };
        let Value::Symbol(hs) = head else {
            return Err(cce_err(head, "clojure.lang.Symbol"));
        };
        if hs.namespace().is_none() && hs.name() == "var" {
            return self.apply_eval_var(items.next());
        }
        if hs.name().ends_with('.') {
            let base = &hs.name()[..hs.name().len() - 1];
            return Err(class_not_found(base));
        }
        let rest: Vec<Value> = items.cloned().collect();
        let cur = self.eval.current_ns_name();
        if let Some(ns_part) = hs.namespace() {
            // `namesStaticMember`: a head outside every known
            // namespace/alias fails the class lookup (probes
            // `c-qualmiss`, `d-static-miss`).
            if !self.ns_or_alias_known(ns_part) {
                return Err(class_not_found(ns_part));
            }
            // Upstream `maybeResolveIn` is interns-only
            // (`findInternedVar` — refers excluded): resolving the
            // full qualified sym from the current ns matches it, so
            // a referred name under a qualified head throws
            // `Can't resolve` (probe `eval-qualified-ignores-refers`).
            let resolved = do_resolve(self.eval, &cur, None, None, &Value::Symbol(hs.clone()))?;
            return self.apply_eval_resolved(resolved, hs, rest);
        }
        let resolved = do_resolve(self.eval, &cur, None, None, &Value::Symbol(hs.clone()))?;
        self.apply_eval_resolved(resolved, hs, rest)
    }

    /// Apply a resolved `#=` head: vars call with the raw args, and
    /// anything else is unresolvable (the JVM's non-`Var` arm —
    /// probes `c-nofn`, `c-if`, `c-classhead`).
    fn apply_eval_resolved(
        &mut self,
        resolved: Value,
        hs: &Symbol,
        rest: Vec<Value>,
    ) -> Result<Value, NativeError> {
        match resolved {
            Value::Var(v) => self.apply_eval_call(v, rest),
            _ => Err(cant_resolve(hs)),
        }
    }

    /// Apply a resolved `#=` head var to the raw operand forms: the
    /// root must be invokable (probe `c-nonfn` pins the module-text
    /// cast); throws propagate (probes `c-div-*`, `d-discard`).
    fn apply_eval_call(
        &mut self,
        var: clj_value::Var,
        rest: Vec<Value>,
    ) -> Result<Value, NativeError> {
        self.eval.force_lazy(var.id);
        let root = self.eval.deref_var(var.id, &var.sym);
        if !is_invokable(&root) {
            return Err(cce_err(&root, "clojure.lang.IFn"));
        }
        let callee = Value::Var(var);
        self.eval.call(&callee, &rest).map_err(signal_to_native)
    }

    /// Apply `#=(var s)`: intern (creating) and return the var
    /// (probes `d-var*`); unqualified and missing seconds NPE like
    /// the JVM (`RT.var` interning accidents).
    fn apply_eval_var(&mut self, second: Option<&Value>) -> Result<Value, NativeError> {
        let Some(second) = second else {
            return Err(npe("Cannot read field \"ns\" because \"vs\" is null"));
        };
        let Value::Symbol(s) = second else {
            return Err(cce_err(second, "clojure.lang.Symbol"));
        };
        let Some(ns_part) = s.namespace() else {
            return Err(npe(
                "Cannot invoke \"String.hashCode()\" because \"this.name\" is null",
            ));
        };
        let name = s.name().to_owned();
        self.eval.registry.ensure_ns(ns_part);
        let target = Symbol::new(Some(ns_part), &name);
        let (id, _) = intern_routed(self.eval, ns_part, None, &target)?;
        Ok(var_value(id, &target))
    }

    /// `Compiler.namespaceFor`: a namespace that exists or is
    /// aliased in the current namespace (probes `c-qq`,
    /// `c-alias-head`).
    fn ns_or_alias_known(&self, ns_part: &str) -> bool {
        if self.eval.registry.ns_id(ns_part).is_some() {
            return true;
        }
        let cur = self.eval.current_ns_name();
        self.eval
            .registry
            .get(&cur)
            .is_some_and(|home| home.aliases().any(|(alias, _, _)| alias == ns_part))
    }
}

impl ReadHooks for LiveReadHooks<'_> {
    fn eval_read(&mut self, form: &Value) -> Result<Value, EvalReadError> {
        self.apply_eval_read(form).map_err(|e| self.lift(e))
    }

    fn resolve(
        &mut self,
        op: ResolverOp,
        sym: Option<&Symbol>,
    ) -> Result<Option<Symbol>, EvalReadError> {
        // The seed cast runs at entry, so only fns arrive here (nil
        // is the unbound no-consult, like the JVM null resolver).
        let bound = deref_core(self.eval, "*reader-resolver*");
        let resolver = match &bound {
            Value::Nil => return Ok(None),
            v if is_resolver_fn(v) => v.clone(),
            v => return Err(self.lift(cce_err(v, RESOLVER_CLASS))),
        };
        let op_name = match op {
            ResolverOp::CurrentNs => "current-ns",
            ResolverOp::ResolveAlias => "resolve-alias",
            ResolverOp::ResolveClass => "resolve-class",
            ResolverOp::ResolveVar => "resolve-var",
        };
        let op_kw = Value::Keyword(Keyword::new(None, op_name));
        let sym_arg = sym.map_or(Value::Nil, |s| Value::Symbol(s.clone()));
        let out = self
            .eval
            .call(&resolver, &[op_kw, sym_arg])
            .map_err(signal_to_native)
            .map_err(|e| self.lift(e))?;
        match out {
            Value::Nil => Ok(None),
            Value::Symbol(s) => Ok(Some(s)),
            other => Err(self.lift(cce_err(&other, "clojure.lang.Symbol"))),
        }
    }

    fn fresh_id(&mut self) -> u64 {
        self.eval.next_id()
    }
}

/// Roots `invoke` dispatches (mirroring its arms): fns,
/// lookupables, vars/promises/multifns, plus nil (the nil-call NPE)
/// and unbound (the unbound-call error); anything else fails the
/// `IFn` cast with module text (probe `c-nonfn`).
fn is_invokable(root: &Value) -> bool {
    matches!(
        root,
        Value::Extension(_)
            | Value::Keyword(_)
            | Value::Symbol(_)
            | Value::Vector(_)
            | Value::MapEntry { .. }
            | Value::Map(_)
            | Value::Set(_)
            | Value::Var(_)
            | Value::Unbound(_)
            | Value::Nil
            | Value::Promise(_)
            | Value::MultiFn(_)
    )
}

/// An unresolvable `#=` head (probes `c-nofn`, `c-qq-miss`,
/// `c-if`, `c-classhead`, `e-eval-sup-tag`).
fn cant_resolve(hs: &Symbol) -> NativeError {
    NativeError::new(ThrowClass::RuntimeException, format!("Can't resolve {hs}"))
}

/// A `#=` class lookup (the port has no classes, so every one
/// fails — probes `d-class-miss`, `c-qualmiss`, `d-ctor`,
/// `d-static-miss`).
fn class_not_found(name: &str) -> NativeError {
    NativeError::new(ThrowClass::ClassNotFoundException, name.to_owned())
}

/// A null-receiver failure with verbatim JDK text (probes
/// `d-var-unqual`, `d-var-nosec`, `c-empty`).
fn npe(message: &str) -> NativeError {
    NativeError::new(ThrowClass::NullPointerException, message.to_owned())
}

/// `String.trim` (S7 `read+string`): strip leading/trailing chars
/// at or below U+0020 (probes `p-ws`, `p-tabs`).
fn java_trim(s: &str) -> String {
    s.trim_matches(|c: char| c <= ' ').to_owned()
}

/// The `:read-cond` option: `:allow`/`:preserve` opt in, anything
/// else (absent included) denies (probes `r-readcond-allow`,
/// `r-readcond-preserve`, `r-cond-opts-absent`, `r-rc-bogus`,
/// `r-rc-nil`).
fn apply_read_cond(m: &Map, opts: &mut ReaderOpts) {
    let key = Value::Keyword(Keyword::new(None, "read-cond"));
    let mode = match m.get(&key) {
        Some(Value::Keyword(kw)) if kw.namespace().is_none() && kw.name() == "allow" => {
            ReadCond::Allow
        }
        Some(Value::Keyword(kw)) if kw.namespace().is_none() && kw.name() == "preserve" => {
            ReadCond::Preserve
        }
        _ => ReadCond::Deny,
    };
    opts.read_cond = mode;
}

/// The `:features` option: a set of keywords (only keywords are
/// kept — anything else can never match a conditional feature);
/// nil/absent keeps the default; anything else fails the
/// `IPersistentSet` cast (probes `r-feat-vec`, `r-feat-nil`).
fn apply_features(m: &Map, opts: &mut ReaderOpts) -> Result<(), NativeError> {
    let key = Value::Keyword(Keyword::new(None, "features"));
    match m.get(&key) {
        None | Some(Value::Nil) => Ok(()),
        Some(Value::Set(s)) => {
            opts.features = s
                .iter()
                .filter_map(|v| match v {
                    Value::Keyword(kw) => Some(kw.clone()),
                    _ => None,
                })
                .collect();
            Ok(())
        }
        Some(other) => Err(cce_err(other, "clojure.lang.IPersistentSet")),
    }
}

/// The `[opts stream]` options: `:read-cond`/`:features` onto the
/// reader options, `:eof` presence (even nil) selecting the
/// no-throw value (probes `r-opts-*`). Nil opts mean defaults;
/// non-map opts fail the `IPersistentMap` cast (probe
/// `r-opts-nonmap`). Shared with `read-string` 2-arity (issue
/// #40: every `rs-opts-*` row agrees with the `read` pins).
pub(crate) fn parse_read_opts(
    opts_arg: &Value,
    opts: &mut ReaderOpts,
) -> Result<(bool, Value, bool), NativeError> {
    if matches!(opts_arg, Value::Nil) {
        return Ok((true, Value::Nil, false));
    }
    let Value::Map(m) = opts_arg else {
        return Err(cce_err(opts_arg, "clojure.lang.IPersistentMap"));
    };
    apply_read_cond(m, opts);
    apply_features(m, opts)?;
    let eof_key = Value::Keyword(Keyword::new(None, "eof"));
    match m.get(&eof_key) {
        Some(v) => Ok((false, v.clone(), false)),
        None => Ok((true, Value::Nil, false)),
    }
}

/// The stream operand: `*in*` at 0-arity, the second arg at
/// 2-arity (`[opts stream]` — there is no `[stream eof-error?]`
/// arity), the first arg otherwise.
fn pick_stream(eval: &Evaluator, args: &[Value]) -> Value {
    match args.len() {
        0 => deref_core(eval, "*in*"),
        2 => args[1].clone(),
        _ => args[0].clone(),
    }
}

/// Stamp the S7 natives' `:arglists` (M4; oracles `p-arglists`,
/// `l-arglists`). Called from `boot` after the natives install.
pub(crate) fn boot_read_arglists(eval: &mut Evaluator) {
    crate::ns::set_arglists_multi(
        eval,
        "read+string",
        &[
            &[],
            &["stream"],
            &["stream", "eof-error?", "eof-value"],
            &["stream", "eof-error?", "eof-value", "recursive?"],
            &["opts", "stream"],
        ],
    );
    crate::ns::set_arglists_multi(eval, "load-reader", &[&["rdr"]]);
}

/// The `PushbackReader` cast: non-nil non-readers fail up front
/// (probe `r-str-stream`); nil streams wait for the `Reader.read()`
/// NPE, after opts validation (probe `r-2arity-boolstream`,
/// plus `r-optsbad-nilstream`/`r-eofopts-nilstream` for the
/// ordering).
fn check_stream_type(stream_value: &Value) -> Result<(), NativeError> {
    if !matches!(stream_value, Value::Nil) && as_reader(stream_value).is_none() {
        return Err(cce_err(stream_value, "java.io.PushbackReader"));
    }
    Ok(())
}

/// The EOF/recursive behavior plus reader options: 0/1-arity throw
/// at EOF, 2-arity parses the opts map, 3/4-arity take
/// truthiness-tested flags (probes `r-opts-*`, `r-eoferr-*`).
fn parse_read_tail(
    args: &[Value],
    opts: &mut ReaderOpts,
) -> Result<(bool, Value, bool), NativeError> {
    match args.len() {
        0 | 1 => Ok((true, Value::Nil, false)),
        2 => parse_read_opts(&args[0], opts),
        3 => Ok((args[1].is_truthy(), args[2].clone(), false)),
        _ => Ok((args[1].is_truthy(), args[2].clone(), args[3].is_truthy())),
    }
}

/// Syntax-quote and `::` resolve against the synced current
/// namespace (read-time, like the JVM), mirroring
/// Populate reader namespace resolution maps (`aliases`, `refers`, and
/// optionally `class_map`) from `cur`, if registered.
pub(crate) fn apply_reader_maps(
    eval: &Evaluator,
    cur: &str,
    opts: &mut ReaderOpts,
    with_classes: bool,
) {
    if let Some(maps) = eval.registry.reader_maps(cur) {
        opts.aliases = maps.aliases;
        opts.refers = maps.refers;
        if with_classes {
            opts.class_map = maps.class_map;
        }
    }
}

/// Sync `opts` against the evaluator's active current namespace for
/// `read`.
pub(crate) fn sync_reader_ns(eval: &Evaluator, opts: &mut ReaderOpts) {
    let cur = eval.current_ns_name();
    opts.current_ns = cur.clone();
    // Live-or-removed: a removed current namespace's tombstone keeps
    // its aliases/refers (the detached `*ns*` object, like the JVM),
    // so syntax-quote still resolves through refers (#80).
    // #120: imports feed syntax-quote's class map, so `` `String ``
    // and `` `String/m `` qualify to the imported class (oracle
    // `rsq-imported`, `rsq-qual-imported`).
    apply_reader_maps(eval, &cur, opts, true);
}

/// Extract one read's drain inputs (fired log + span) from a
/// locked stream state. The caller drops the guard BEFORE running
/// user code (H1): drain/resolve invoke data-reader fns, which
/// may read the same reader again — holding the non-reentrant
/// lock across them deadlocks.
fn take_drain(state: &mut StreamState) -> (Vec<FiredTag>, Span) {
    let at = Span::new(state.parser.line(), state.parser.col());
    (state.parser.take_fired(), at)
}

/// Drain fired-discards outside the stream lock (S6 `Reader.`):
/// effects precede every exit (a fired tag ran during the read).
/// See [`take_drain`] for the lock discipline.
fn run_drain(
    eval: &mut Evaluator,
    fired: Vec<FiredTag>,
    at: Span,
    recursive: bool,
    suppressed: bool,
) -> Result<(), NativeError> {
    crate::core::drain_fired_tags(eval, fired, suppressed)
        .map_err(|e| wrap_resolution_error(eval, e, at, recursive))
}

/// A resolution failure as a `read` failure (S6): raw when
/// `recursive?`, else the `ReaderException` wrap at the live tracker
/// span (the port's readers are line-numbering, so `read` wraps —
/// probes `h-lnpr-*`; the cause keeps its data — probe
/// `h-lnpr-fnthrows`).
fn wrap_resolution_error(
    eval: &mut Evaluator,
    err: NativeError,
    span: Span,
    recursive: bool,
) -> NativeError {
    if recursive {
        return err;
    }
    let inner = err.into_inner();
    let mut outer = NativeError::new(
        ThrowClass::ReaderException,
        format!("{}: {}", inner.class.jvm_name(), inner.message),
    );
    outer.data = Some(reader_ex_data(span));
    outer.cause = Some(Throwable {
        id: eval.next_id(),
        class: inner.class,
        message: if inner.message.is_empty() {
            None
        } else {
            Some(Arc::from(inner.message.as_str()))
        },
        data: inner.data.map(Box::new),
        cause: inner.cause.map(Box::new),
    });
    outer
}

/// The S7 `read+string` capture (empty unless asked).
fn capture_text(parser: &Reader, mark: usize, capture: bool) -> String {
    if capture {
        parser.captured_since(mark)
    } else {
        String::new()
    }
}

/// Take the live parser out of its lock (S7): the hook runs user
/// code during the read, which may read the same reader again —
/// holding the (non-reentrant) lock across the read would
/// deadlock. The dummy left behind is never observed: every path
/// restores the advanced parser before returning, retrying, or
/// feeding. Returns whether the stream may grow (stdin).
/// The taken parser keeps its installed options: callers seed
/// once up front (per-take re-seeding would wipe the fired log
/// mid-discard — H2). `opts` only shapes the dummy.
pub(crate) fn take_parser(reader: &StreamReader, opts: &ReaderOpts) -> (Reader, bool) {
    let mut state = lock_state(reader);
    let grow = may_grow(&state);
    let parser = std::mem::replace(&mut state.parser, Reader::new("", opts));
    (parser, grow)
}

/// Restore a taken parser (see [`take_parser`]).
pub(crate) fn restore_parser(reader: &StreamReader, parser: Reader) {
    lock_state(reader).parser = parser;
}

/// Feed one line after an EOF (S7): the pre-attempt head returns
/// first (a truncated head survives retries), then the chunk.
fn feed_chunk(state: &mut StreamState, snapshot: Option<Reader>, chunk: &str) {
    if let Some(parser) = snapshot {
        state.parser = parser;
    }
    state.parser.extend(chunk);
}

/// Run one `read`: fresh options per call, stdin fed line by line
/// while the parser reports EOF, clean EOF honoring `eof-error?`
/// (the synthesized error sits at the live tracker position,
/// probe `r-exdata-eof`). A truncated head survives retries: the
/// parser is snapshotted before each attempt while more input may
/// arrive, then restored and extended with the next line (`Eof*`
/// only — anything else throws without feeding). The S7 hook rides
/// each attempt with the parser taken out of its lock (the hook
/// runs user code, which may read the same reader again);
/// `capture` brackets the consumed text for `read+string`.
/// Data readers resolve per call (S6): fired discards drain ahead
/// of every terminal exit, then the form resolves.
#[allow(clippy::too_many_arguments)]
fn run_read(
    eval: &mut Evaluator,
    reader: &StreamReader,
    eof_error: bool,
    eof_value: Value,
    recursive: bool,
    opts: &ReaderOpts,
    capture: bool,
) -> Result<(Value, String), NativeError> {
    check_reader_open(reader)?;
    let mark = lock_state(reader).parser.capture_mark();
    // Entry options, installed once: re-seeding per attempt would
    // wipe legit firings mid-discard on stdin retries (H2 —
    // `set_opts` clears the fired log).
    lock_state(reader).parser.set_opts(opts);
    loop {
        let (mut parser, grow) = take_parser(reader, opts);
        // Snapshot before the attempt consumes the head (string
        // readers never grow, so they never pay for the clone).
        let snapshot = grow.then(|| parser.clone());
        let outcome = {
            let mut hooks = LiveReadHooks::new(&mut *eval);
            parser.read_form(false, &mut hooks)
        };
        match outcome {
            Ok(Some(v)) => {
                let text = capture_text(&parser, mark, capture);
                // One guard episode (restore + extract), then the
                // guard drops before user code runs (H1).
                let (fired, at) = {
                    let mut state = lock_state(reader);
                    state.parser = parser;
                    take_drain(&mut state)
                };
                let form = finish_some(eval, v, fired, at, recursive, opts.suppress_read)?;
                return Ok((form, text));
            }
            Ok(None) => {
                if let Some(done) = finish_none_attempt(
                    eval,
                    reader,
                    parser,
                    mark,
                    capture,
                    eof_error,
                    &eof_value,
                    recursive,
                    opts.suppress_read,
                )? {
                    return Ok(done);
                }
            }
            Err(e) => {
                // `Err` shares the `Some` arm (100% line gate: a
                // lone `?` line would be uncoverable — `next_chunk`
                // fails only on live-stdin IO, which tests never
                // drive; same or-merge as `read_form`'s EOF arms).
                match finish_err_attempt(
                    eval,
                    reader,
                    parser,
                    snapshot,
                    &e,
                    recursive,
                    opts.suppress_read,
                ) {
                    Ok(Some(err)) | Err(err) => return Err(err),
                    Ok(None) => { /* retry */ }
                }
            }
        }
    }
}

/// Finish a no-form attempt (split from [`run_read`] for the
/// complexity ratchet — pure code motion, same guard episodes as
/// before): a fresh chunk extends the parser and retries (`None`
/// = retry), a true EOF finishes here.
#[allow(clippy::too_many_arguments)]
fn finish_none_attempt(
    eval: &mut Evaluator,
    reader: &StreamReader,
    parser: Reader,
    mark: usize,
    capture: bool,
    eof_error: bool,
    eof_value: &Value,
    recursive: bool,
    suppressed: bool,
) -> Result<Option<(Value, String)>, NativeError> {
    let mut state = lock_state(reader);
    state.parser = parser;
    match next_chunk(&mut state)? {
        Some(chunk) => {
            state.parser.extend(&chunk);
            Ok(None)
        }
        None => {
            // Capture before the drain: a reentrant
            // read mid-discard would skew the window.
            let text = capture_text(&state.parser, mark, capture);
            let (fired, at) = take_drain(&mut state);
            drop(state);
            let out =
                finish_clean_eof(eval, fired, at, eof_error, eof_value, recursive, suppressed)?;
            Ok(Some((out, text)))
        }
    }
}

/// Handle a failed read attempt (split from [`run_read`] for the
/// complexity ratchet — pure code motion, same guard episodes as
/// before): non-EOF errors and EOF-without-chunk finish here;
/// EOF-with-chunk feeds the parser and retries (`None` = retry).
#[allow(clippy::too_many_arguments)]
fn finish_err_attempt(
    eval: &mut Evaluator,
    reader: &StreamReader,
    parser: Reader,
    snapshot: Option<Reader>,
    e: &ReaderError,
    recursive: bool,
    suppressed: bool,
) -> Result<Option<NativeError>, NativeError> {
    if !is_eof_error(e) {
        let (fired, at) = {
            let mut state = lock_state(reader);
            state.parser = parser;
            take_drain(&mut state)
        };
        return Ok(Some(finish_read_error(
            eval, fired, at, e, recursive, suppressed,
        )));
    }
    let mut state = lock_state(reader);
    state.parser = parser;
    match next_chunk(&mut state)? {
        Some(chunk) => {
            feed_chunk(&mut state, snapshot, &chunk);
            Ok(None)
        }
        None => {
            let (fired, at) = take_drain(&mut state);
            drop(state);
            Ok(Some(finish_read_error(
                eval, fired, at, e, recursive, suppressed,
            )))
        }
    }
}

/// A read form: drain first, then resolve (S6). Runs unlocked
/// (see [`take_drain`]): the caller extracts `fired`/`at` under
/// one guard episode, then drops it before calling here.
fn finish_some(
    eval: &mut Evaluator,
    form: Value,
    fired: Vec<FiredTag>,
    at: Span,
    recursive: bool,
    suppressed: bool,
) -> Result<Value, NativeError> {
    run_drain(eval, fired, at, recursive, suppressed)?;
    crate::core::resolve_data_readers(eval, form, suppressed)
        .map_err(|e| wrap_resolution_error(eval, e, at, recursive))
}

/// A clean EOF: drain first, then honor `eof-error?` (the
/// synthesized error sits at the live tracker position, probe
/// `r-exdata-eof`). Runs unlocked (see [`take_drain`]).
fn finish_clean_eof(
    eval: &mut Evaluator,
    fired: Vec<FiredTag>,
    at: Span,
    eof_error: bool,
    eof_value: &Value,
    recursive: bool,
    suppressed: bool,
) -> Result<Value, NativeError> {
    run_drain(eval, fired, at, recursive, suppressed)?;
    if eof_error {
        let clean = ReaderError::Eof { span: at };
        return Err(map_reader_error(eval, &clean, recursive));
    }
    Ok(eof_value.clone())
}

/// A terminal read error: drain first (effects precede), then map
/// (always an error — the `read+string` capture has no `Ok` shape
/// for it, so this returns the error itself). Runs unlocked (see
/// [`take_drain`]).
fn finish_read_error(
    eval: &mut Evaluator,
    fired: Vec<FiredTag>,
    at: Span,
    err: &ReaderError,
    recursive: bool,
    suppressed: bool,
) -> NativeError {
    if let Err(e) = run_drain(eval, fired, at, recursive, suppressed) {
        return e;
    }
    map_reader_error(eval, err, recursive)
}

/// `read`: all five JVM arities over the bound reader (probes
/// `r-*`, `r-eof-outer`, `r-trunc-outer`, `r-recursive-outer`,
/// `r-exdata*`, `r-excause`).
fn native_read(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/read", args, &[0, 1, 2, 3, 4])?;
    let stream_value = pick_stream(eval, args);
    check_stream_type(&stream_value)?;
    // S7 gates (probes `g-*`, `h-order*`): the resolver cast,
    // then `:unknown`, then opts parsing.
    let seed = seed_read_gates(eval)?;
    let mut opts = ReaderOpts {
        read_cond: ReadCond::Deny,
        ..ReaderOpts::default()
    };
    let (eof_error, eof_value, recursive) = parse_read_tail(args, &mut opts)?;
    sync_reader_ns(eval, &mut opts);
    crate::core::snapshot_data_readers(eval, &mut opts)?;
    opts.read_eval = seed.read_eval;
    opts.suppress_read = seed.suppress_read;
    opts.resolver_bound = seed.resolver_bound;
    if matches!(stream_value, Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.io.Reader.read()\" because \"r\" is null".to_owned(),
        ));
    }
    let reader = as_reader(&stream_value).expect("nil and non-readers fail above");
    let (v, _) = run_read(eval, reader, eof_error, eof_value, recursive, &opts, false)?;
    Ok(v)
}

/// The `read+string` hint cast (S7; probe `p-strarg`): the
/// parameter type is `LineNumberingPushbackReader`, checked at
/// entry (nil slips through to the `getString` NPE below).
fn check_plus_stream_type(stream_value: &Value) -> Result<(), NativeError> {
    if as_reader(stream_value).is_none() {
        return Err(cce_err(stream_value, READER_CLASS));
    }
    Ok(())
}

/// `read+string`: `read` plus the consumed text (S7; probes
/// `p-*`): the same five arities, the hint cast at entry, the
/// nil-`getString` NPE ahead of opts parsing (probe
/// `p-nil-badopts`), and the Java-trimmed capture as the second
/// element.
fn native_read_plus_string(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/read+string", args, &[0, 1, 2, 3, 4])?;
    let stream_value = pick_stream(eval, args);
    if !matches!(stream_value, Value::Nil) {
        check_plus_stream_type(&stream_value)?;
    }
    if matches!(stream_value, Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"clojure.lang.LineNumberingPushbackReader.getString()\" because \"stream\" is null".to_owned(),
        ));
    }
    let seed = seed_read_gates(eval)?;
    let mut opts = ReaderOpts {
        read_cond: ReadCond::Deny,
        ..ReaderOpts::default()
    };
    let (eof_error, eof_value, recursive) = parse_read_tail(args, &mut opts)?;
    sync_reader_ns(eval, &mut opts);
    // Defer mode, like `read` (B2): without the snapshot, table
    // tags fail, `#inst` hits the stale N5 text, and default-fn
    // tags misroute.
    crate::core::snapshot_data_readers(eval, &mut opts)?;
    opts.read_eval = seed.read_eval;
    opts.suppress_read = seed.suppress_read;
    opts.resolver_bound = seed.resolver_bound;
    let reader = as_reader(&stream_value).expect("nil and non-readers fail above");
    let (v, text) = run_read(eval, reader, eof_error, eof_value, recursive, &opts, true)?;
    Ok(Value::Vector(clj_value::Vector::from_vec(vec![
        v,
        Value::Str(Arc::from(java_trim(&text).as_str())),
    ])))
}

/// `read-line`: one line from `*in*` (nil at EOF). Nil `*in*` is
/// the verbatim `Var.get()` NPE (probe `rl-nil-in`); non-readers
/// fail the `BufferedReader` cast (probes `rl-str-in`,
/// `rl-num-in`).
pub(crate) fn native_read_line(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.core/read-line", args, &[0])?;
    let stream_value = deref_core(eval, "*in*");
    if matches!(stream_value, Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            "Cannot invoke \"java.io.BufferedReader.readLine()\" because the return value of \"clojure.lang.Var.get()\" is null".to_owned(),
        ));
    }
    let Some(reader) = as_reader(&stream_value) else {
        return Err(cce_err(&stream_value, "java.io.BufferedReader"));
    };
    check_reader_open(reader)?;
    let mut state = lock_state(reader);
    Ok(stream_read_line(&mut state)?
        .map(|line| Value::Str(Arc::from(line.as_str())))
        .unwrap_or(Value::Nil))
}

/// `BufferedReader.readLine` on a stream reader (#205): one line
/// without the terminator (nil at EOF — the `read-line` body
/// minus the `*in*` lookup; closed reports the JDK text).
pub(crate) fn reader_read_line(r: &StreamReader) -> Result<Value, NativeError> {
    check_reader_open(r)?;
    let mut state = lock_state(r);
    Ok(stream_read_line(&mut state)?
        .map(|line| Value::Str(Arc::from(line.as_str())))
        .unwrap_or(Value::Nil))
}

/// The S-instream natives installed into `clojure.core`.
pub(crate) const READER_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("read", native_read),
    ("read-line", native_read_line),
    ("read+string", native_read_plus_string),
];

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    use std::io::Cursor;

    use super::*;
    use clj_value::{PrintOpts, pr_to_string};

    /// A scratch evaluator (ids only — the readers under test never
    /// evaluate).
    fn scratch() -> Evaluator {
        Evaluator::new_bare("user")
    }

    fn pr(v: &Value) -> String {
        pr_to_string(v, &PrintOpts::pr())
    }

    /// Run `run_read` over a string reader with scripted stdin lines
    /// (the stdin retry path without process stdin).
    fn run_scripted(
        eval: &mut Evaluator,
        first: &str,
        script: &[&str],
    ) -> Result<Value, NativeError> {
        let value = string_reader(eval, first);
        let reader = as_reader(&value).expect("a reader");
        lock_state(reader)
            .script_lines
            .extend(script.iter().map(|s| (*s).to_owned()));
        let opts = ReaderOpts {
            read_cond: ReadCond::Deny,
            ..ReaderOpts::default()
        };
        run_read(eval, reader, true, Value::Nil, false, &opts, false).map(|(v, _)| v)
    }

    #[test]
    fn constructors_pin_kinds() {
        // The only stdin-kind test: construction alone (driving it
        // would block on process stdin). The stdin parser starts
        // empty with line numbers on; the string parser starts over
        // its input.
        let mut eval = scratch();
        let stdin_value = stdin_reader(&mut eval);
        let stdin_ref = as_reader(&stdin_value).expect("a reader");
        let stdin_state = lock_state(stdin_ref);
        assert!(stdin_state.stdin.is_some());
        assert_eq!(
            (stdin_state.parser.line(), stdin_state.parser.col()),
            (1, 1)
        );
        drop(stdin_state);
        let str_value = string_reader(&mut eval, "a\n");
        let str_ref = as_reader(&str_value).expect("a reader");
        let mut str_state = lock_state(str_ref);
        assert!(str_state.stdin.is_none());
        assert_eq!(str_state.parser.read_line(), Some("a".to_owned()));
        assert!(as_reader(&Value::Nil).is_none());
    }

    #[test]
    fn pull_line_feeds_and_reports_eof() {
        // The stdin feed over in-memory buffers: fills extend the
        // parser (normalizing), EOF reports `false`.
        let mut eval = scratch();
        let value = string_reader(&mut eval, "");
        let reader = as_reader(&value).expect("a reader");
        let mut state = lock_state(reader);
        let mut src = Cursor::new(b"a\r\nb" as &[u8]);
        assert!(pull_line(&mut src, &mut state.parser).unwrap());
        assert!(!pull_line(&mut src, &mut state.parser).unwrap());
        assert_eq!(state.parser.read_line(), Some("a".to_owned()));
        assert_eq!(state.parser.read_line(), Some("b".to_owned()));
        assert_eq!(state.parser.read_line(), None);
    }

    #[test]
    fn more_input_drains_script_lines_first() {
        // The `read-line` feed over scripted lines (never process
        // stdin): a scripted line extends the parser, then the
        // string reader reports no more input.
        let mut eval = scratch();
        let value = string_reader(&mut eval, "");
        let reader = as_reader(&value).expect("a reader");
        let mut state = lock_state(reader);
        state.script_lines.push_back("a\n".to_owned());
        assert!(more_input(&mut state).unwrap());
        assert_eq!(state.parser.read_line(), Some("a".to_owned()));
        assert!(!more_input(&mut state).unwrap());
        assert_eq!(state.parser.read_line(), None);
    }

    #[test]
    fn stream_read_line_feeds_until_terminated() {
        // A buffered partial line is not the EOF tail while more
        // input may arrive: the feed loops until a terminator is
        // buffered, across as many chunks as it takes.
        let mut eval = scratch();
        let value = string_reader(&mut eval, "ab");
        let reader = as_reader(&value).expect("a reader");
        let mut state = lock_state(reader);
        state.script_lines.push_back("c".to_owned());
        state.script_lines.push_back("d\n".to_owned());
        assert_eq!(
            stream_read_line(&mut state).unwrap(),
            Some("abcd".to_owned())
        );
        assert_eq!(stream_read_line(&mut state).unwrap(), None);
    }

    #[test]
    fn slurp_reader_drains_script_lines() {
        // `slurp` over a reader with scripted lines keeps feeding
        // past the first chunk (the loop-continue arm).
        let mut eval = scratch();
        let value = string_reader(&mut eval, "a");
        let reader = as_reader(&value).expect("a reader");
        lock_state(reader).script_lines.push_back("b\n".to_owned());
        assert_eq!(slurp_reader(reader).unwrap(), "ab\n");

        // Direct stdin bulk path: CRLF/CR normalization + multi-chunk UTF-8/CR boundaries.
        let ext: Arc<dyn ExtensionValue> = Arc::new(StreamReader {
            id: 9001,
            state: Arc::new(Mutex::new(StreamState {
                parser: Reader::new("head:", &ReaderOpts::default()),
                stdin: Some(BufReader::new(Box::new(Cursor::new(b"a\r\nb\rc".to_vec())))),
                closed: false,
                script_lines: VecDeque::new(),
            })),
        });
        let bulk = Value::Extension(Arc::clone(&ext));
        assert_eq!(
            slurp_reader(as_reader(&bulk).unwrap()).unwrap(),
            "head:a\nb\nc"
        );
        assert_eq!(close_reader_value(&ext).unwrap().unwrap(), Value::Nil);
        // Single-chunk fast move and empty EOF.
        let mut fast = String::new();
        slurp_buf_reader(&mut Cursor::new(b"ok".as_slice()), &mut fast).unwrap();
        assert_eq!(fast, "ok");
        let mut empty = String::new();
        slurp_buf_reader(&mut Cursor::new(b"".as_slice()), &mut empty).unwrap();
        assert_eq!(empty, "");

        // Inputs past the old 8 KiB first-chunk edge (one buffer now): a
        // multi-byte char and a CRLF across that edge, a trailing `\r` at
        // EOF, `Interrupted`, and IO error / malformed UTF-8 coverage.
        let mut big = vec![b'x'; 8 * 1024 - 1];
        big.extend_from_slice("\r\n€".as_bytes()); // `\r` ends chunk 1, `\n€` starts chunk 2
        let mut big2 = vec![b'y'; 8 * 1024 - 1];
        big2.extend_from_slice("€\r".as_bytes()); // partial `€` straddles chunk edge + trailing `\r`
        struct RetryOnce(Vec<u8>, bool);
        impl Read for RetryOnce {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if !self.1 {
                    self.1 = true;
                    return Err(std::io::Error::from(std::io::ErrorKind::Interrupted));
                }
                let n = buf.len().min(self.0.len());
                buf[..n].copy_from_slice(&self.0[..n]);
                self.0.drain(..n);
                Ok(n)
            }
        }
        let mut multi = String::new();
        slurp_buf_reader(&mut RetryOnce(big, false), &mut multi).unwrap();
        assert_eq!(multi.len(), 8 * 1024 - 1 + 1 + 3);
        let mut multi2 = String::new();
        slurp_buf_reader(&mut Cursor::new(big2), &mut multi2).unwrap();
        assert!(multi2.ends_with("€\n"));

        // Truncated UTF-8 at EOF, corrupt UTF-8 at and past the old edge, and a read error.
        let mut trunc = vec![b'a'; 8 * 1024 - 1];
        trunc.push(0xE2); // 1st byte of 3-byte char at EOF
        assert_eq!(
            slurp_buf_reader(&mut Cursor::new(trunc), &mut String::new())
                .unwrap_err()
                .class,
            ThrowClass::IOException
        );
        let mut bad_tail = vec![b'a'; 8 * 1024 - 1];
        bad_tail.extend_from_slice(&[0xC3, 0x28]); // invalid continuation across 8K boundary
        assert_eq!(
            slurp_buf_reader(&mut Cursor::new(bad_tail), &mut String::new())
                .unwrap_err()
                .class,
            ThrowClass::IOException
        );
        let mut bad_mid = vec![b'a'; 8 * 1024];
        bad_mid.push(0xff);
        assert_eq!(
            slurp_buf_reader(&mut Cursor::new(bad_mid), &mut String::new())
                .unwrap_err()
                .class,
            ThrowClass::IOException
        );
        let mut bad_eof = vec![b'a'; 8 * 1024];
        bad_eof.push(0xC3); // incomplete UTF-8 ending 2nd chunk
        assert_eq!(
            slurp_buf_reader(&mut Cursor::new(bad_eof), &mut String::new())
                .unwrap_err()
                .class,
            ThrowClass::IOException
        );
        struct ErrRead;
        impl Read for ErrRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
        }
        assert_eq!(
            slurp_buf_reader(&mut ErrRead, &mut String::new())
                .unwrap_err()
                .class,
            ThrowClass::IOException
        );
    }

    #[test]
    fn stdin_retry_keeps_fired_discards() {
        // H2 (review round 1): a discard fired before an EOF
        // retry must survive the continuation — per-take
        // re-seeding used to wipe the fired log.
        let mut eval = crate::ns::boot();
        eval.eval_source("(def h2-calls (atom []))").expect("setup");
        eval.eval_source("(alter-var-root #'clojure.core/*data-readers* (constantly {'h2foo (fn [x] (do (swap! h2-calls conj :foo) :F))}))")
            .expect("table");
        let value = string_reader(&mut eval, "#_ #h2foo 1\n");
        let reader = as_reader(&value).expect("a reader");
        lock_state(reader)
            .script_lines
            .extend([":kept\n".to_owned()]);
        let mut opts = ReaderOpts {
            read_cond: ReadCond::Deny,
            ..ReaderOpts::default()
        };
        crate::core::snapshot_data_readers(&mut eval, &mut opts).expect("snapshot");
        let (v, _) =
            run_read(&mut eval, reader, true, Value::Nil, false, &opts, false).expect("read");
        assert_eq!(pr(&v), ":kept");
        let calls = eval.eval_source("@h2-calls").expect("calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(pr(&calls[0]), "[:foo]");
    }

    #[test]
    fn stdin_retry_keeps_truncated_head() {
        // A stdin form split across lines completes with its head
        // intact (the retry restores the snapshot, then extends).
        let mut eval = scratch();
        let v = run_scripted(&mut eval, "", &["(a\n", "b)\n"]).unwrap();
        assert_eq!(pr(&v), "(a b)");
    }

    #[test]
    fn stdin_retry_exhausted_throws_truncated_span() {
        // A partial head with no more lines throws the truncated
        // error (ex-data at the bumped EOF position, probe
        // `r-exdata` shape over fed input).
        let mut eval = scratch();
        let err = run_scripted(&mut eval, "", &["(a\n"]).unwrap_err();
        assert_eq!(err.class, ThrowClass::ReaderException);
        assert_eq!(
            err.message,
            "java.lang.RuntimeException: EOF while reading, starting at line 1"
        );
        let data = err.data.clone().expect("ex-data");
        assert!(matches!(data, Value::Map(_)), "ex-data is a map");
        assert_eq!(pr(&data), "#:clojure.error{:line 2, :column 1}");
    }

    #[test]
    fn stdin_retry_clean_eof_feeds_then_reads() {
        // Clean `None`s (empty, then whitespace-only) feed lines
        // until a form completes.
        let mut eval = scratch();
        let v = run_scripted(&mut eval, "", &["\n", "(b)\n"]).unwrap();
        assert_eq!(pr(&v), "(b)");
    }

    #[test]
    fn pull_line_rejects_non_utf8() {
        // Invalid UTF-8 is an `IOException` carrying the OS message.
        let mut eval = scratch();
        let value = string_reader(&mut eval, "");
        let reader = as_reader(&value).expect("a reader");
        let mut state = lock_state(reader);
        let mut src = Cursor::new(b"\xff" as &[u8]);
        let err = pull_line(&mut src, &mut state.parser).unwrap_err();
        assert_eq!(err.class, ThrowClass::IOException);
        let mentions_utf8 = err.message.contains("stream did not contain valid UTF-8");
        assert!(mentions_utf8, "{}", err.message);
    }

    #[test]
    fn pull_chunk_splits_at_char_boundaries() {
        // A buffer whose fills cut inside multi-byte chars: pulls
        // take the valid prefix and keep the split char buffered;
        // a fill holding only a partial char reads ahead by line.
        struct Chunky {
            data: Vec<u8>,
            pos: usize,
            fill: usize,
        }
        impl std::io::Read for Chunky {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let n = (self.data.len() - self.pos).min(out.len());
                out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
                self.pos += n;
                Ok(n)
            }
        }
        impl BufRead for Chunky {
            fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
                let end = (self.pos + self.fill).min(self.data.len());
                Ok(&self.data[self.pos..end])
            }
            fn consume(&mut self, amt: usize) {
                self.pos = (self.pos + amt).min(self.data.len());
            }
        }
        fn chunky(data: &str, fill: usize) -> Chunky {
            Chunky {
                data: data.as_bytes().to_vec(),
                pos: 0,
                fill,
            }
        }
        // "αβγ" = 6 bytes; fill 3 splits the second char.
        let mut src = chunky("αβγ", 3);
        assert_eq!(pull_chunk(&mut src).unwrap(), Some("α".to_owned()));
        assert_eq!(pull_chunk(&mut src).unwrap(), Some("β".to_owned()));
        assert_eq!(pull_chunk(&mut src).unwrap(), Some("γ".to_owned()));
        assert_eq!(pull_chunk(&mut src).unwrap(), None);
        // The `Read` side serves raw bytes in order.
        let mut src = chunky("αx", 1);
        let mut raw = [0u8; 2];
        assert_eq!(std::io::Read::read(&mut src, &mut raw).unwrap(), 2);
        assert_eq!(&raw, "α".as_bytes());
        // Fill 1: a lone lead byte reads ahead to the line end.
        let mut src = chunky("αx", 1);
        assert_eq!(pull_chunk(&mut src).unwrap(), Some("αx".to_owned()));
        assert_eq!(pull_chunk(&mut src).unwrap(), None);
        // Invalid bytes mid-fill report the read_line message.
        let mut src = Chunky {
            data: vec![b'a', 0xff],
            pos: 0,
            fill: 8,
        };
        let err = pull_chunk(&mut src).unwrap_err();
        assert_eq!(err.class, ThrowClass::IOException);
        assert!(err.message.contains("stream did not contain valid UTF-8"));
    }

    #[test]
    fn reader_debug_names_id() {
        // The `Debug` impl (an `ExtensionValue` bound) names the id.
        let mut eval = scratch();
        let value = string_reader(&mut eval, "x");
        let reader = as_reader(&value).expect("a reader");
        let id = reader.id;
        assert_eq!(format!("{reader:?}"), format!("StreamReader({id})"));
    }

    #[test]
    fn seed_bare_evaluator_allows() {
        // The missing-`*read-eval*` arm: a bare evaluator (no
        // vars installed) seeds `read_eval` true, matching the
        // true root.
        let eval = scratch();
        let seed = seed_read_gates(&eval).expect("bare seed allows");
        assert!(seed.read_eval);
        assert!(!seed.suppress_read);
        assert!(!seed.resolver_bound);
    }

    #[test]
    fn reader_eq_and_hash_use_id() {
        // Identity semantics, mirroring `Writer` (behavior tests
        // pin `=`/`distinct?`; the direct calls pin the trait fns).
        let mut eval = scratch();
        let first = string_reader(&mut eval, "x");
        let second = string_reader(&mut eval, "x");
        let a = as_reader(&first).expect("a reader");
        let b = as_reader(&second).expect("a reader");
        assert!(a.dyn_eq(a));
        assert!(!a.dyn_eq(b));
        let mut ha = DefaultHasher::new();
        a.dyn_hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.dyn_hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    #[test]
    fn apply_reader_maps_tolerates_missing_ns() {
        let eval = scratch();
        let mut opts = ReaderOpts::default();
        apply_reader_maps(&eval, "missing.ns", &mut opts, true);
        assert!(opts.aliases.is_empty());
        apply_reader_maps(&eval, "user", &mut opts, false);
        apply_reader_maps(&eval, "user", &mut opts, true);
    }

    #[test]
    fn more_input_and_feed_chunk_cover_stdin_and_snapshots() {
        let mut state = StreamState {
            parser: Reader::new("", &ReaderOpts::default()),
            stdin: Some(BufReader::new(Box::new(std::io::Cursor::new(
                b"42\n".to_vec(),
            )))),
            closed: false,
            script_lines: VecDeque::new(),
        };
        assert!(more_input(&mut state).unwrap());
        feed_chunk(&mut state, None, "1");
        feed_chunk(
            &mut state,
            Some(Reader::new("", &ReaderOpts::default())),
            "2",
        );

        struct BoomRead;
        impl std::io::Read for BoomRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
        }
        state.stdin = Some(BufReader::new(Box::new(BoomRead)));
        assert!(more_input(&mut state).is_err());
    }
}
