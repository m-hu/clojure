// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Dispatch macros (`reader-contract.md` §8–§9, §14–§18, §20).
//!
//! - `;`/`#!`: to end of line, sentinel. `#_`: discard one form.
//! - `#tag`: the ctor path (unread, read tag — non-symbol rejects —
//!   read form, then preserve/suppress ⇒ `TaggedLiteral`, dotted
//!   name ⇒ record deferral, defer mode (S6) ⇒ the consult (table
//!   keys, then `inst`/`uuid` validation, then the default-fn
//!   flag, else unknown tag — hits defer to `TaggedLiteral`, with
//!   the fired-discards log under `#_`), `inst`/`uuid` ⇒
//!   data-reader deferral, else unknown tag).
//! - `#=`: immediate deferral without consuming the form. `#<`:
//!   unreadable.
//! - `#:`: namespaced maps (auto/current-ns/alias resolution, key
//!   rewrite, evenness + duplicate rules).
//! - `##`: symbolic values (one recursive form, must be an
//!   `Inf`/`-Inf`/`NaN` symbol).
//! - `#?`/`#?@`: reader conditionals — allow-list checked before the
//!   body is touched; the `readCondDelimited` loop shape replicated
//!   exactly (match arm, then the ALWAYS-run suppressed discard;
//!   splices prepend list/vector items to the shared pending queue;
//!   top-level splicing rejected unless nothing matched).

use clj_value::{
    Keyword, List, Symbol, Value, java_debug, reader_conditional, tagged_literal, validate_inst,
    validate_uuid,
};

use crate::Reader;
use crate::error::ReaderError;
use crate::hooks::{EvalReadError, ReadHooks};
use crate::opts::ReadCond;
use crate::symbol::is_whitespace;

impl Reader {
    /// `DispatchReader`: `#` plus one char (EOF ⇒ `EofChar`); unknown
    /// chars (including non-ASCII) unread into the ctor path.
    pub(crate) fn dispatch_reader(
        &mut self,
        top: bool,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        let Some(ch) = self.stream.read() else {
            return Err(ReaderError::EofChar { span: self.span() });
        };
        match ch {
            '^' => self.meta_reader(hook),
            '#' => self.symbolic_reader(hook),
            '\'' => self.var_reader(hook),
            '"' => self.regex_reader(),
            '(' => self.fn_reader(hook),
            '{' => self.set_reader(hook),
            '=' => self.eval_reader(hook),
            '!' => self.comment_reader(),
            '<' => Err(ReaderError::Unreadable { span: self.span() }),
            '_' => self.discard_reader(hook),
            '?' => self.conditional_reader(top, hook),
            ':' => self.nsmap_reader(hook),
            _ => {
                self.stream.unread(ch);
                self.ctor_reader(hook)
            }
        }
    }

    /// `EvalReader` (S7): the `*read-eval*` gate first (entry span,
    /// probe `h-gate-lnpr`), then the operand, then the hook
    /// application (post-operand span, probe `h-div-lnpr`).
    /// Suppression never stops `#=` (probe `d-discard`).
    fn eval_reader(&mut self, hook: &mut dyn ReadHooks) -> Result<Option<Value>, ReaderError> {
        if !self.opts.read_eval {
            return Err(ReaderError::EvalFailed {
                span: self.span(),
                error: EvalReadError::Failed(clj_value::Throwable {
                    id: hook.fresh_id(),
                    class: clj_value::ThrowClass::RuntimeException,
                    message: Some(std::sync::Arc::from(
                        "EvalReader not allowed when *read-eval* is false.",
                    )),
                    data: None,
                    cause: None,
                }),
            });
        }
        let form = self.read_nested_value(hook)?;
        let at = self.span();
        match hook.eval_read(&form) {
            Ok(v) => Ok(Some(v)),
            Err(EvalReadError::NoHook) => Err(ReaderError::ReadEval { span: at }),
            Err(failed) => Err(ReaderError::EvalFailed {
                span: at,
                error: failed,
            }),
        }
    }

    /// `CommentReader` (shared by `;` and `#!`): skip to `\n` or EOF
    /// (the stream never yields `\r`), return the sentinel.
    pub(crate) fn comment_reader(&mut self) -> Result<Option<Value>, ReaderError> {
        while let Some(c) = self.stream.read() {
            if c == '\n' {
                break;
            }
        }
        Ok(None)
    }

    /// `DiscardReader`: read one form, drop it, return the sentinel.
    /// `DiscardReader`: read one form, drop it, return the sentinel.
    /// The depth flag marks tags for the fired-discards log (S6: the
    /// JVM invokes data readers for discarded forms). No `?` before
    /// the restore (like `cond_discard`): a failed discard must not
    /// stick the depth for the next read.
    pub(crate) fn discard_reader(
        &mut self,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        self.discarding += 1;
        let out = self.read_nested_value(hook);
        self.discarding -= 1;
        out?;
        Ok(None)
    }

    /// `VarReader`: `#'X` ⇒ `(var X)`.
    fn var_reader(&mut self, hook: &mut dyn ReadHooks) -> Result<Option<Value>, ReaderError> {
        let form = self.read_nested_value(hook)?;
        Ok(Some(Value::List(List::from_vec(vec![
            Value::Symbol(Symbol::new(None, "var")),
            form,
        ]))))
    }

    /// `CtorReader`: `#tag form` (see module docs). The tag read is
    /// `eofIsError`; a non-symbol tag rejects before the form is read;
    /// the form reads before any tag lookup (EOF there wins).
    fn ctor_reader(&mut self, hook: &mut dyn ReadHooks) -> Result<Option<Value>, ReaderError> {
        let tag = self.read_nested_value(hook)?;
        let Value::Symbol(sym) = tag else {
            return Err(ReaderError::TagNotSymbol { span: self.span() });
        };
        let checkpoint = self.fired.len();
        let form = self.read_nested_value(hook)?;
        // Suppressed reads (the cond discard loop) and preserved
        // conds produce `TaggedLiteral` without invoking anything.
        if self.suppress || (self.in_cond && self.opts.read_cond == ReadCond::Preserve) {
            return Ok(Some(tagged_literal(sym, form)));
        }
        // The dotted check is on the tag NAME, not the namespace
        // (both modes: record tags never consult the table).
        if sym.name().contains('.') {
            return Err(ReaderError::Record {
                span: self.span(),
                tag: sym.to_string(),
            });
        }
        if self.opts.data_readers {
            return self.consult_data_readers(&sym, form, checkpoint);
        }
        if sym.namespace().is_none() && (sym.name() == "inst" || sym.name() == "uuid") {
            return Err(ReaderError::DataReader {
                span: self.span(),
                tag: sym.to_string(),
            });
        }
        Err(ReaderError::UnknownTag {
            span: self.span(),
            tag: sym.to_string(),
        })
    }

    /// Defer-mode tag consult (S6): the table, then the `inst`/`uuid`
    /// defaults, then the default-fn flag, else the `UnknownTag`
    /// throw — all with span, exactly where the non-defer path
    /// throws. Invocation stays eval-side: table hits,
    /// default-fn tags, and validated `inst`/`uuid` payloads
    /// defer to `TaggedLiteral` for the eval walk (which resolves
    /// the defaults to natives — #103 — against the live table,
    /// so a mid-walk override install still wins).
    fn consult_data_readers(
        &mut self,
        sym: &Symbol,
        form: Value,
        checkpoint: usize,
    ) -> Result<Option<Value>, ReaderError> {
        // The borked table fails before any lookup (probe
        // `e-nonmap-inst`).
        if let Some(from_jvm) = self.opts.data_readers_borked.clone() {
            return Err(ReaderError::TableCce {
                span: self.span(),
                from_jvm,
            });
        }
        // Table hits defer for eval-side invocation (overrides skip
        // default validation — probe `e-override-novalid`).
        if self.opts.data_reader_keys.contains(sym) {
            return self.deferred_tag(sym, form, checkpoint);
        }
        // DIVERGENCE (#94): this `inst`/`uuid` arm is hardcoded —
        // upstream consults the `default-data-readers` VAR root
        // here, so replacing that root is a silent no-op in this
        // port (like `resolve_tag` eval-side).
        if sym.namespace().is_none() && matches!(sym.name(), "inst" | "uuid") {
            // String payloads validate inline (spanful); anything
            // else defers for the eval-side type dispatch (nil is
            // the NPE, the rest fail the `CharSequence` cast —
            // except `uuid`, whose non-strings fail as
            // `IllegalArgumentException`).
            if let Value::Str(s) = &form {
                let failure = if sym.name() == "inst" {
                    validate_inst(s).err()
                } else {
                    validate_uuid(s).err()
                };
                if let Some(e) = failure {
                    return Err(ReaderError::TagInvoke {
                        span: self.span(),
                        class: e.class,
                        message: e.message,
                    });
                }
                // Validated: this `TaggedLiteral` defers for the
                // eval walk, which resolves it through the
                // flipped `inst`/`uuid` defaults to a native
                // (#103) — re-validating idempotently on the
                // way (the reader must not construct eagerly:
                // the walk consults the live table).
                return Ok(Some(tagged_literal(sym.clone(), form)));
            }
            return self.deferred_tag(sym, form, checkpoint);
        }
        if self.opts.default_reader_fn {
            return self.deferred_tag(sym, form, checkpoint);
        }
        Err(ReaderError::UnknownTag {
            span: self.span(),
            tag: sym.to_string(),
        })
    }

    /// Defer one tag for eval-side invocation. Under `#_` the
    /// occurrence is also logged as fired (the evaluator invokes it
    /// for effect); entries logged while reading this form belong to
    /// nested tags, which the drain re-invokes through this entry's
    /// form — truncate them so each tag fires exactly once.
    fn deferred_tag(
        &mut self,
        sym: &Symbol,
        form: Value,
        checkpoint: usize,
    ) -> Result<Option<Value>, ReaderError> {
        if self.discarding > 0 {
            self.fired.truncate(checkpoint);
            self.fired.push(crate::FiredTag {
                tag: sym.clone(),
                form: form.clone(),
            });
        }
        Ok(Some(tagged_literal(sym.clone(), form)))
    }

    /// `NamespaceMapReader` (see module docs).
    fn nsmap_reader(&mut self, hook: &mut dyn ReadHooks) -> Result<Option<Value>, ReaderError> {
        let auto = match self.stream.read() {
            Some(':') => true,
            Some(c) => {
                self.stream.unread(c);
                false
            }
            // EOF: `unread(-1)` is skipped on the JVM; fall through
            // so the symbol read below reports `Eof`.
            None => false,
        };
        let (sym, next) = self.nsmap_head(auto, hook)?;
        // EOF here (including right after the symbol) means "must
        // specify a map", not `Eof`.
        if next != Some('{') {
            return Err(ReaderError::NsMapMap { span: self.span() });
        }
        let ns = self.resolve_nsmap_ns(auto, sym)?;
        self.nsmap_assemble(&ns, hook)
    }

    /// Skip whitespace chars, returning the first non-ws char (or EOF).
    fn skip_ws(&mut self, mut next: Option<char>) -> Option<char> {
        while next.is_some_and(is_whitespace) {
            next = self.stream.read();
        }
        next
    }

    /// Scan the ns-map head: optional alias symbol plus the char that
    /// follows it (whitespace skipped). A bad namespace errors here.
    fn nsmap_head(
        &mut self,
        auto: bool,
        hook: &mut dyn ReadHooks,
    ) -> Result<(Option<Value>, Option<char>), ReaderError> {
        let mut sym: Option<Value> = None;
        let mut next = self.stream.read();
        if next.is_some_and(is_whitespace) {
            if auto {
                // `#:: { }`: skip to `{`, else no namespace.
                next = self.skip_ws(next);
                if next != Some('{') {
                    if let Some(c) = next {
                        self.stream.unread(c);
                    }
                    return Err(ReaderError::NsMapNs { span: self.span() });
                }
            } else {
                if let Some(c) = next {
                    self.stream.unread(c);
                }
                return Err(ReaderError::NsMapNs { span: self.span() });
            }
        } else if next != Some('{') {
            if let Some(c) = next {
                self.stream.unread(c);
            }
            sym = Some(self.read_nested_value(hook)?);
            let head = self.stream.read();
            next = self.skip_ws(head);
        }
        Ok((sym, next))
    }

    /// Assemble the ns-map body: read `{...}`, check evenness, rewrite
    /// keys, build the checked map.
    fn nsmap_assemble(
        &mut self,
        ns: &str,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        let items = self.read_delimited('}', hook)?;
        if items.len() % 2 == 1 {
            return Err(ReaderError::NsMapOdd { span: self.span() });
        }
        let pairs = crate::coll::pair_up(items)
            .into_iter()
            .map(|(k, v)| (nsmap_key(k, ns), v))
            .collect();
        Self::checked_map(pairs, self.span()).map(|m| Some(Value::Map(m)))
    }

    /// Resolve the ns-map namespace (auto/current-ns/alias rules).
    fn resolve_nsmap_ns(&mut self, auto: bool, sym: Option<Value>) -> Result<String, ReaderError> {
        if auto {
            match sym {
                None => Ok(self.opts.current_ns.clone()),
                Some(Value::Symbol(sym)) if sym.namespace().is_none() => {
                    let name = sym.name().to_owned();
                    match self.opts.aliases.get(&name) {
                        Some(target) => Ok(target.clone()),
                        None => Err(ReaderError::NsMapAlias {
                            span: self.span(),
                            sym: sym.to_string(),
                        }),
                    }
                }
                Some(other) => Err(ReaderError::NsMapValidNs {
                    span: self.span(),
                    dbg: java_debug(&other),
                }),
            }
        } else {
            match sym {
                // `#:{}`: null sym ⇒ invalid namespace, rendering `null`.
                Some(Value::Symbol(sym)) if sym.namespace().is_none() => Ok(sym.name().to_owned()),
                Some(other) => Err(ReaderError::NsMapValidNs {
                    span: self.span(),
                    dbg: java_debug(&other),
                }),
                None => Err(ReaderError::NsMapValidNs {
                    span: self.span(),
                    dbg: "null".to_owned(),
                }),
            }
        }
    }

    /// `SymbolicValueReader`: one recursive form; non-symbols and
    /// unknown names reject with `##`-prefixed messages.
    fn symbolic_reader(&mut self, hook: &mut dyn ReadHooks) -> Result<Option<Value>, ReaderError> {
        let form = self.read_nested_value(hook)?;
        if let Value::Symbol(sym) = &form {
            return match (sym.namespace(), sym.name()) {
                (None, "Inf") => Ok(Some(Value::Double(f64::INFINITY))),
                (None, "-Inf") => Ok(Some(Value::Double(f64::NEG_INFINITY))),
                (None, "NaN") => Ok(Some(Value::Double(f64::NAN))),
                _ => Err(ReaderError::UnknownSymbolic {
                    span: self.span(),
                    sym: sym.to_string(),
                }),
            };
        }
        Err(ReaderError::BadSymbolic {
            span: self.span(),
            dbg: java_debug(&form),
        })
    }

    /// `ConditionalReader`: allow-list first (before touching the
    /// body), optional `@`, whitespace, `(`, then preserve (read the
    /// list raw) or the delimited loop — all under `in_cond`.
    fn conditional_reader(
        &mut self,
        top: bool,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        if !matches!(self.opts.read_cond, ReadCond::Allow | ReadCond::Preserve) {
            return Err(ReaderError::CondNotAllowed { span: self.span() });
        }
        // No EOF check between `@` and the whitespace skip (an EOF
        // there falls through to the check below).
        let mut ch = self.stream.read();
        let mut splicing = false;
        if ch == Some('@') {
            splicing = true;
            ch = self.stream.read();
        }
        ch = self.skip_ws(ch);
        let Some(ch) = ch else {
            return Err(ReaderError::EofChar { span: self.span() });
        };
        if ch != '(' {
            return Err(ReaderError::CondBody { span: self.span() });
        }
        let prev = self.in_cond;
        self.in_cond = true;
        // No `?` before the restore (like `cond_discard`): a failed
        // preserve read must not stick `in_cond` for the next read.
        let result = if self.opts.read_cond == ReadCond::Preserve {
            // The raw list (with line/col metas as usual). The
            // list reader returns a form on every success path,
            // so `expect` (whose line always executes) spells the
            // invariant coverably (100% line gate).
            self.list_reader(hook).map(|form| {
                Some(reader_conditional(
                    form.expect("the list reader always returns a form"),
                    splicing,
                ))
            })
        } else {
            self.read_cond_delimited(splicing, top, hook)
        };
        self.in_cond = prev;
        result
    }

    /// `readCondDelimited`: the loop shape, not "collect pairs" (see
    /// module docs). `top` is the calling frame's top-level flag.
    fn read_cond_delimited(
        &mut self,
        splicing: bool,
        top: bool,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        let firstline = self.firstline();
        let mut result: Option<Value> = None;
        loop {
            if result.is_none() && self.cond_match_arm(&mut result, firstline, hook)? {
                break;
            }
            // The discard runs ALWAYS — even right after a match.
            if self.cond_discard(firstline, hook)? {
                break;
            }
        }
        self.cond_finish(result, splicing, top)
    }

    /// One feature-match attempt of the cond loop: read a feature,
    /// validate it, and read its value on match. True finishes the
    /// loop (`)` where a feature was expected).
    fn cond_match_arm(
        &mut self,
        result: &mut Option<Value>,
        firstline: Option<u32>,
        hook: &mut dyn ReadHooks,
    ) -> Result<bool, ReaderError> {
        match self.read_inner(false, Some(')'), false, hook)? {
            crate::Form::Eof => Err(self.eof_delimited(firstline)),
            crate::Form::Finished => Ok(true),
            crate::Form::Value(f) => {
                if is_reserved_feature(&f) {
                    return Err(ReaderError::ReservedFeature {
                        span: self.span(),
                        kw: java_debug(&f),
                    });
                }
                let Value::Keyword(kw) = &f else {
                    return Err(ReaderError::BadFeature {
                        span: self.span(),
                        dbg: java_debug(&f),
                    });
                };
                if is_default(kw) || self.opts.features.contains(kw) {
                    match self.read_inner(false, Some(')'), false, hook)? {
                        crate::Form::Eof => Err(self.eof_delimited(firstline)),
                        crate::Form::Finished => Err(ReaderError::CondEven {
                            span: self.span(),
                            line: firstline,
                        }),
                        crate::Form::Value(v) => {
                            *result = Some(v);
                            Ok(false)
                        }
                    }
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// The ALWAYS-run suppressed discard of the cond loop. True
    /// finishes the loop.
    fn cond_discard(
        &mut self,
        firstline: Option<u32>,
        hook: &mut dyn ReadHooks,
    ) -> Result<bool, ReaderError> {
        let prev = self.suppress;
        self.suppress = true;
        let discarded = self.read_inner(false, Some(')'), false, hook);
        self.suppress = prev;
        match discarded? {
            crate::Form::Eof => Err(self.eof_delimited(firstline)),
            crate::Form::Finished => Ok(true),
            crate::Form::Value(_) => Ok(false),
        }
    }

    /// Finish the cond loop: no match ⇒ sentinel; splicing matches
    /// prepend to the pending queue (top-level splices reject).
    fn cond_finish(
        &mut self,
        result: Option<Value>,
        splicing: bool,
        top: bool,
    ) -> Result<Option<Value>, ReaderError> {
        let Some(result) = result else {
            // No match: the reader sentinel (recur).
            return Ok(None);
        };
        if !splicing {
            return Ok(Some(result));
        }
        let items = match &result {
            Value::List(l) => l.iter().cloned().collect::<Vec<_>>(),
            Value::Vector(v) => v.iter().cloned().collect::<Vec<_>>(),
            _ => {
                return Err(ReaderError::SpliceType { span: self.span() });
            }
        };
        if top {
            return Err(ReaderError::SpliceTopLevel { span: self.span() });
        }
        for v in items.into_iter().rev() {
            self.pending.push_front(v);
        }
        Ok(None)
    }
}

/// Rewrite one ns-map key: unqualified ⇒ prefix, `_`-ns ⇒ strip,
/// otherwise unchanged.
fn nsmap_key(key: Value, ns: &str) -> Value {
    match &key {
        Value::Keyword(kw) if kw.namespace().is_none() => {
            Value::Keyword(Keyword::new(Some(ns), kw.name()))
        }
        Value::Keyword(kw) if kw.namespace() == Some("_") => {
            Value::Keyword(Keyword::new(None, kw.name()))
        }
        Value::Symbol(sym) if sym.namespace().is_none() => {
            Value::Symbol(Symbol::new(Some(ns), sym.name()))
        }
        Value::Symbol(sym) if sym.namespace() == Some("_") => {
            Value::Symbol(Symbol::new(None, sym.name()))
        }
        _ => key,
    }
}

fn is_reserved_feature(f: &Value) -> bool {
    if let Value::Keyword(kw) = f {
        kw.namespace().is_none() && (kw.name() == "else" || kw.name() == "none")
    } else {
        false
    }
}

fn is_default(kw: &Keyword) -> bool {
    kw.namespace().is_none() && kw.name() == "default"
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use clj_value::{Symbol, ThrowClass, Throwable, Value};

    use crate::hooks::ScriptedHooks;
    use crate::{FiredTag, NoHooks, ReadCond, Reader, ReaderError, ReaderOpts, read_str};

    fn opts() -> ReaderOpts {
        ReaderOpts::default()
    }

    /// Defer-mode options: table keys plus the default-fn flag.
    fn defer_opts(keys: &[(&str, &str)], default_fn: bool) -> ReaderOpts {
        ReaderOpts {
            data_readers: true,
            data_reader_keys: keys
                .iter()
                .map(|(ns, name)| {
                    let ns = (!ns.is_empty()).then_some(*ns);
                    Symbol::new(ns, name)
                })
                .collect::<HashSet<_>>(),
            default_reader_fn: default_fn,
            ..ReaderOpts::default()
        }
    }

    /// Read every form plus the fired-discards log (S6 consult tests).
    fn read_defer(
        input: &str,
        opts: &ReaderOpts,
    ) -> (Result<Vec<Value>, ReaderError>, Vec<FiredTag>) {
        let mut reader = Reader::new(input, opts);
        let mut hooks = NoHooks;
        let mut forms = Vec::new();
        let mut failure: Option<ReaderError> = None;
        while failure.is_none() {
            match reader.read_form(false, &mut hooks) {
                Ok(Some(v)) => forms.push(v),
                Ok(None) => break,
                Err(e) => failure = Some(e),
            }
        }
        let out = match failure {
            Some(e) => Err(e),
            None => Ok(forms),
        };
        let fired = reader.take_fired();
        (out, fired)
    }

    /// Assert a value is the tagged literal `#tag form` (tag printed).
    fn assert_tagged(v: Value, tag: &str, form: &Value) {
        let Value::TaggedLiteral {
            tag: t, form: f, ..
        } = &v
        else {
            panic!("expected a tagged literal, got {v:?}");
        };
        assert_eq!(t.to_string(), tag);
        assert_eq!(f.as_ref(), form);
    }

    /// Assert a single-entry map holds one `Int(1)` pair keyed by a
    /// symbol (structural match: no `Value` equality needed).
    fn assert_single_sym_pair(v: Value, ns: Option<&str>, name: &str) {
        let Value::Map(m) = v else {
            panic!("expected a map, got {v:?}");
        };
        assert_eq!(m.size(), 1);
        for (k, val) in m.iter() {
            assert_eq!(val, &Value::Int(1));
            assert!(matches!(k, Value::Symbol(s) if s.namespace() == ns && s.name() == name));
        }
    }

    #[test]
    #[should_panic(expected = "expected a map")]
    fn assert_single_sym_pair_rejects_non_map() {
        // The helper's failure contract (also covers its guard line).
        assert_single_sym_pair(Value::Nil, None, "x");
    }

    /// A scripted application throwable (S7 unit pins).
    fn failed(id: u64, class: ThrowClass, message: &str) -> crate::EvalReadError {
        crate::EvalReadError::Failed(Throwable {
            id,
            class,
            message: Some(std::sync::Arc::from(message)),
            data: None,
            cause: None,
        })
    }

    #[test]
    fn eval_reader_denies_before_reading() {
        // S7 gate (probe `b-false`): the denial fires before the
        // operand is touched (even `#=`-at-EOF reports the gate),
        // at the entry span (probe `h-gate-lnpr`). #125: this
        // triple is the JVM's own (message + simple class + span),
        // pinned conformant by the corpus `#=(+ 1 2)` case.
        for input in ["#=(+ 1 2)", "#="] {
            let opts = ReaderOpts {
                read_eval: false,
                ..ReaderOpts::default()
            };
            let err = read_str(input, &opts, &mut NoHooks).unwrap_err();
            assert!(
                matches!(err, ReaderError::EvalFailed { .. }),
                "{input}: {err:?}"
            );
            assert_eq!(
                err.message(),
                "EvalReader not allowed when *read-eval* is false."
            );
            assert_eq!(err.jvm_class(), "RuntimeException");
            let span = err.span();
            assert_eq!((span.line, span.col), (1, 3), "{input}");
        }
    }

    #[test]
    fn eval_reader_applies_through_the_hook() {
        // S7 application: the hook answer becomes the form (the
        // operand reads first, so nested reads still run).
        let mut hooks = ScriptedHooks::new(Ok(Value::Int(42)), Vec::new());
        let v = read_str("#=(+ 1 2)", &opts(), &mut hooks).unwrap();
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn eval_reader_reports_hook_failures_post_operand() {
        // S7 failures surface with the post-operand span (probe
        // `h-div-lnpr`); suppression never stops `#=` (probe
        // `d-discard`).
        let opts = ReaderOpts {
            suppress_read: true,
            ..ReaderOpts::default()
        };
        let mut hooks = ScriptedHooks::new(
            Err(failed(7, ThrowClass::ArithmeticException, "Divide by zero")),
            Vec::new(),
        );
        let err = read_str("#=(/ 1 0)", &opts, &mut hooks).unwrap_err();
        assert!(matches!(err, ReaderError::EvalFailed { .. }));
        assert_eq!(err.message(), "Divide by zero");
        assert_eq!(err.jvm_class(), "ArithmeticException");
        let span = err.span();
        assert_eq!((span.line, span.col), (1, 10));
    }

    #[test]
    fn eval_reader_hookless_reports_read_eval() {
        // S7 pure-reader `#=`: allowed by the gate but nobody home.
        let err = read_str("#=1", &opts(), &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::ReadEval { .. }));
        assert_eq!(err.message(), "Eval reader #= requires an evaluator hook");
    }

    #[test]
    #[should_panic(expected = "expected a tagged literal")]
    fn assert_tagged_rejects_plain_values() {
        // The helper's failure contract (also covers its guard line).
        assert_tagged(Value::Int(1), "foo", &Value::Int(1));
    }

    #[test]
    fn automap_ws_then_nonbrace_rejects_namespace() {
        // `nsmap_head`, auto + whitespace: skip to `{`, else `NsMapNs`.
        for input in ["#:: x {}", "#:: "] {
            let err = read_str(input, &opts(), &mut NoHooks).unwrap_err();
            assert!(matches!(err, ReaderError::NsMapNs { .. }), "{input}");
            assert_eq!(err.message(), "Namespaced map must specify a namespace");
        }
    }

    #[test]
    fn automap_non_symbol_ns_rejected() {
        // `resolve_nsmap_ns`, auto + non-symbol head.
        let err = read_str("#::1 {}", &opts(), &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::NsMapValidNs { .. }));
        assert_eq!(
            err.message(),
            "Namespaced map must specify a valid namespace: 1"
        );
    }

    #[test]
    fn cond_eof_where_feature_expected() {
        // `cond_match_arm` head read hits EOF.
        let err = read_str("#?(", &opts(), &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::EofDelimited { .. }));
        assert_eq!(err.message(), "EOF while reading, starting at line 1");
    }

    #[test]
    fn cond_eof_where_value_expected() {
        // Feature matched, then EOF where its value goes.
        let err = read_str("#?(:clj", &opts(), &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::EofDelimited { line: 1, .. }));
    }

    #[test]
    fn cond_eof_in_discard_position() {
        // Unmatched feature, then EOF where its discarded value goes.
        let err = read_str("#?(:nothere", &opts(), &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::EofDelimited { line: 1, .. }));
    }

    #[test]
    fn preserve_cond_failure_restores_in_cond() {
        // One reused `Reader`: a failed preserve read must not
        // stick `in_cond`, else the next tag read returns
        // `TaggedLiteral` instead of the `DataReader` error.
        let opts = ReaderOpts {
            read_cond: ReadCond::Preserve,
            ..ReaderOpts::default()
        };
        let mut reader = Reader::new("#?(a b] #inst \"x\"", &opts);
        let err = reader.read_form(true, &mut NoHooks).unwrap_err();
        assert!(
            matches!(err, ReaderError::UnmatchedDelim { .. }),
            "got {err:?}"
        );
        let err = reader.read_form(true, &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::DataReader { .. }), "got {err:?}");
    }

    #[test]
    fn nsmap_prefixes_unqualified_symbol_keys() {
        // `nsmap_key` unqualified-symbol arm.
        let v = read_str("#:foo{a 1}", &opts(), &mut NoHooks).unwrap();
        assert_single_sym_pair(v, Some("foo"), "a");
    }

    #[test]
    fn nsmap_strips_underscore_symbol_ns() {
        // `nsmap_key` `_`-ns symbol arm.
        let v = read_str("#:foo{_/a 1}", &opts(), &mut NoHooks).unwrap();
        assert_single_sym_pair(v, None, "a");
    }

    #[test]
    fn defer_table_hit_defers_without_firing() {
        // Keys membership defers to `TaggedLiteral` (the eval walk
        // invokes); a plain read logs nothing.
        let (forms, fired) = read_defer("#foo 1", &defer_opts(&[("", "foo")], false));
        let forms = forms.unwrap();
        assert_eq!(forms.len(), 1);
        assert_tagged(forms[0].clone(), "foo", &Value::Int(1));
        assert!(fired.is_empty());
    }

    #[test]
    fn defer_discard_logs_fired_once_in_order() {
        // `#_` logs each occurrence oldest-first (probes
        // `e-discard-fires`, `e-discard-order`).
        let (forms, fired) =
            read_defer("#_ #a 1 #b 2", &defer_opts(&[("", "a"), ("", "b")], false));
        let forms = forms.unwrap();
        assert_eq!(forms.len(), 1);
        assert_tagged(forms[0].clone(), "b", &Value::Int(2));
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].tag.to_string(), "a");
        assert_eq!(fired[0].form, Value::Int(1));
    }

    #[test]
    fn defer_nested_discard_truncates_inner() {
        // The outer entry's form still holds the inner tag, which
        // the drain re-invokes — one log entry total.
        let (forms, fired) = read_defer(
            "#_ #foo #bar 1 :kept",
            &defer_opts(&[("", "foo"), ("", "bar")], false),
        );
        let forms = forms.unwrap();
        assert_eq!(
            forms,
            vec![Value::Keyword(clj_value::Keyword::new(None, "kept"))]
        );
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].tag.to_string(), "foo");
        assert_tagged(fired[0].form.clone(), "bar", &Value::Int(1));
    }

    #[test]
    fn defer_borked_table_fails_the_cast() {
        // Any tag over a non-lookupable table fails `ILookup`
        // (probes `e-nonmap`, `e-nonmap-inst`).
        let opts = ReaderOpts {
            data_readers_borked: Some("java.lang.Long".to_owned()),
            ..defer_opts(&[], false)
        };
        for input in ["#foo 1", "#inst \"2024-01-01\""] {
            let err = read_str(input, &opts, &mut NoHooks).unwrap_err();
            assert!(matches!(err, ReaderError::TableCce { .. }), "{input}");
            assert_eq!(err.jvm_class(), "ClassCastException");
            assert_eq!(
                err.message(),
                "class java.lang.Long cannot be cast to class clojure.lang.ILookup (java.lang.Long is in module java.base of loader 'bootstrap'; clojure.lang.ILookup is in unnamed module of loader 'app')"
            );
        }
    }

    #[test]
    fn defer_default_fn_flag_defers() {
        // The flag defers unknown tags (the eval walk calls
        // `[tag form]`); without it the same read throws.
        let (forms, _) = read_defer("#foo 1", &defer_opts(&[], true));
        assert_tagged(forms.unwrap()[0].clone(), "foo", &Value::Int(1));
        let err = read_str("#foo 1", &defer_opts(&[], false), &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::UnknownTag { .. }));
        assert_eq!(err.message(), "No reader function for tag foo");
        // The token read consumes EOF (the line bump), so the throw
        // site matches the JVM wrap position (probe `h-lnpr-wrap`).
        assert_eq!(err.span(), crate::Span::new(2, 1));
    }

    #[test]
    fn defer_dotted_tag_keeps_record_error() {
        // Dotted tags never consult (even with keys and the flag).
        let err = read_str(
            "#foo.bar [1]",
            &defer_opts(&[("", "foo.bar")], true),
            &mut NoHooks,
        )
        .unwrap_err();
        assert!(matches!(err, ReaderError::Record { .. }), "got {err:?}");
    }

    #[test]
    fn defer_validates_inst_payloads() {
        // Valid strings defer as tagged literals for the eval
        // walk (no fired entry even under `#_` — default
        // construction is pure, so the drain has nothing to
        // invoke); invalid ones throw the spanful validator
        // error (probes `d-inst`, `d-inst-bad`,
        // `g-discard-inst-bad`).
        let opts = defer_opts(&[], false);
        let (forms, fired) = read_defer("#_ #inst \"2024-01-01\" :kept", &opts);
        assert_eq!(forms.unwrap().len(), 1);
        assert!(fired.is_empty());
        let err = read_str("#inst \"bogus\"", &opts, &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::TagInvoke { .. }), "got {err:?}");
        assert_eq!(err.jvm_class(), "RuntimeException");
        assert_eq!(err.message(), "Unrecognized date/time syntax: bogus");
        assert_eq!(err.span(), crate::Span::new(1, 14));
    }

    #[test]
    fn defer_validates_uuid_payloads() {
        // Same split for `uuid` (probes `d-uuid`, `d-uuid-bad`).
        let opts = defer_opts(&[], false);
        let (forms, fired) = read_defer("#uuid \"550e8400-e29b-41d4-a716-446655440000\"", &opts);
        assert_tagged(
            forms.unwrap()[0].clone(),
            "uuid",
            &Value::Str("550e8400-e29b-41d4-a716-446655440000".into()),
        );
        assert!(fired.is_empty());
        let err = read_str("#uuid \"bogus\"", &opts, &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::TagInvoke { .. }), "got {err:?}");
        assert_eq!(err.jvm_class(), "IllegalArgumentException");
        assert_eq!(err.message(), "Invalid UUID string: bogus");
    }

    #[test]
    fn defer_non_string_defaults_defer() {
        // Non-string `inst`/`uuid` payloads defer for the eval-side
        // type dispatch (nil is the NPE, the rest fail the
        // `CharSequence` cast — except `uuid`, whose non-strings
        // fail as `IllegalArgumentException`).
        let (forms, fired) = read_defer("#inst 42", &defer_opts(&[], false));
        assert_tagged(forms.unwrap()[0].clone(), "inst", &Value::Int(42));
        assert!(fired.is_empty());
        let (forms, _) = read_defer("#uuid nil", &defer_opts(&[], false));
        assert_tagged(forms.unwrap()[0].clone(), "uuid", &Value::Nil);
    }

    #[test]
    fn defer_table_override_skips_validation() {
        // Keys win over the defaults without validating (probe
        // `e-override-novalid`).
        let (forms, fired) = read_defer("#inst \"bogus\"", &defer_opts(&[("", "inst")], false));
        assert_tagged(
            forms.unwrap()[0].clone(),
            "inst",
            &Value::Str("bogus".into()),
        );
        assert!(fired.is_empty());
    }

    #[test]
    fn defer_skips_consult_when_suppressed_or_preserved() {
        // Suppressed arms and preserved conds never consult (a
        // borked table stays silent — probes
        // `g-allow-nomatch-instbad`, `g-preserve-instbad`).
        let mut opts = defer_opts(&[], false);
        opts.data_readers_borked = Some("java.lang.Long".to_owned());
        let (forms, fired) = read_defer("#?(:cljs #foo 1) :kept", &opts);
        assert_eq!(forms.unwrap().len(), 1);
        assert!(fired.is_empty());
        opts.read_cond = ReadCond::Preserve;
        let (forms, fired) = read_defer("#?(:a #foo 1)", &opts);
        let forms = forms.unwrap();
        assert_eq!(forms.len(), 1);
        assert!(matches!(forms[0], Value::ReaderConditional { .. }));
        assert!(fired.is_empty());
    }

    #[test]
    fn defer_rejects_non_symbol_tags_and_eof_first() {
        // The tag read and form read precede the consult (like the
        // frozen path).
        let opts = defer_opts(&[("", "foo")], true);
        let err = read_str("#123 1", &opts, &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::TagNotSymbol { .. }));
        // ...through the multi-form helper too: nothing fired
        // before the failure, and the failure surfaces as `Err`.
        let (out, fired) = read_defer("#foo (unclosed", &opts);
        let err = out.unwrap_err();
        assert!(fired.is_empty());
        assert!(
            matches!(err, ReaderError::EofDelimited { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn fired_log_drains_and_clears() {
        // `take_fired` empties the log; `set_opts` starts a fresh
        // generation even with entries pending (the evaluator
        // drains after every read, so the clear is a no-op there).
        let opts = defer_opts(&[("", "foo")], false);
        let mut reader = Reader::new("#_ #foo 1 :kept", &opts);
        let mut hooks = NoHooks;
        assert!(reader.read_form(false, &mut hooks).unwrap().is_some());
        assert_eq!(reader.take_fired().len(), 1);
        assert!(reader.take_fired().is_empty());
        reader.extend(" #_ #foo 2");
        assert!(reader.read_form(false, &mut hooks).unwrap().is_none());
        reader.set_opts(&opts);
        assert!(reader.take_fired().is_empty());
    }

    #[test]
    fn discard_failure_restores_depth() {
        // A failed discard must not stick the depth: the next tag
        // read on the same reader defers without logging.
        let opts = defer_opts(&[("", "foo")], false);
        let mut reader = Reader::new("#_ (unclosed", &opts);
        let mut hooks = NoHooks;
        let err = reader.read_form(false, &mut hooks).unwrap_err();
        assert!(
            matches!(err, ReaderError::EofDelimited { .. }),
            "got {err:?}"
        );
        assert!(reader.take_fired().is_empty());
        reader.extend(" #foo 1");
        let form = reader.read_form(false, &mut hooks).unwrap().unwrap();
        assert_tagged(form.clone(), "foo", &Value::Int(1));
        assert!(reader.take_fired().is_empty());
    }
}
