// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Metadata, fn literals, and syntax-quote (`reader-contract.md`
//! §11–§13).
//!
//! - `^`: coerce the meta form (`Symbol`/`String` ⇒ `{:tag m}`,
//!   `Keyword` ⇒ `{m true}`, `Vector` ⇒ `{:param-tags m}`, `Map`
//!   as-is), attach to `Symbol`/`List`/`Vector`/`Map`/`Set` only,
//!   assoc `:line`/`:column` for lists unless present, merge over the
//!   object's existing meta.
//! - `#(...)`: fresh arg env per form (nested `#()` errors);
//!   `%`/`%&`/`%n` register `p{n}__{id}#`/`rest__{id}#` from the
//!   process-global id counter; expansion is `(fn* [args] form)` with
//!   gap args garg'd after the read.
//! - `` ` ``: fresh gensym env per form; specials quote unqualified;
//!   `s#` memoizes `s__{id}__auto__`; class/var/alias/current-ns
//!   resolution from the harness maps; unquote splices raw;
//!   collections expand to `apply`/`seq`/`concat` shapes; leftover
//!   meta wraps in `with-meta`.

use std::collections::BTreeMap;
use std::sync::Arc;

use clj_value::{Keyword, List, Symbol, ThrowClass, Throwable, Value, Vector};

use crate::Reader;
use crate::error::ReaderError;
use crate::hooks::{EvalReadError, ReadHooks, ResolverOp};
use crate::symbol::{interpret_token, read_token};

impl Reader {
    /// `MetaReader` (shared by `^` and dispatch `^`).
    pub(crate) fn meta_reader(
        &mut self,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        let meta_pos = self.meta_pos();
        let meta_form = self.read_nested_value(hook)?;
        let mut meta = coerce_meta(meta_form, self.span())?;
        let obj = self.read_nested_value(hook)?;
        if !supports_meta(&obj) {
            return Err(ReaderError::MetaTarget { span: self.span() });
        }
        if let (Some((line, col)), Value::List(_)) = (meta_pos, &obj) {
            meta = assoc_unless_present(meta, "line", Value::Int(i64::from(line)));
            meta = assoc_unless_present(meta, "column", Value::Int(i64::from(col)));
        }
        // Merge over the object's existing meta (insertion order is
        // canonicalized by the harness; values match exactly).
        let mut merged = obj.meta().cloned().unwrap_or_else(clj_value::new_map);
        for (k, v) in meta.iter() {
            merged = merged.insert(k.clone(), v.clone());
        }
        Ok(Some(obj.with_meta(Some(merged))))
    }

    /// `FnReader`.
    pub(crate) fn fn_reader(
        &mut self,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        if self.arg_env.is_some() {
            return Err(ReaderError::NestedFn { span: self.span() });
        }
        self.arg_env = Some(BTreeMap::new());
        self.stream.unread('(');
        let form = match self.read_nested_value(hook) {
            Ok(form) => form,
            Err(e) => {
                self.arg_env = None;
                return Err(e);
            }
        };
        let argsyms = self.arg_env.take().unwrap();
        let mut args: Vec<Value> = Vec::new();
        if let Some(&higharg) = argsyms.keys().next_back() {
            if higharg > 0 {
                for i in 1..=higharg {
                    match argsyms.get(&i) {
                        Some(sym) => args.push(Value::Symbol(sym.clone())),
                        // Gaps garg'd AFTER the read: fresh ids sort last.
                        None => args.push(Value::Symbol(garg(i))),
                    }
                }
            }
            if let Some(rest) = argsyms.get(&-1) {
                args.push(Value::Symbol(Symbol::new(None, "&")));
                args.push(Value::Symbol(rest.clone()));
            }
        }
        Ok(Some(Value::List(List::from_vec(vec![
            Value::Symbol(Symbol::new(None, "fn*")),
            Value::Vector(Vector::from_vec(args)),
            form,
        ]))))
    }

    /// `ArgReader`: outside `#()` a plain token; inside, `%`/`%&`/`%n`
    /// register args (anything else is `BadArg`; overflowing integers
    /// fail `Integer.parseInt` exactly like the JVM).
    pub(crate) fn arg_reader(
        &mut self,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        let token = read_token(&mut self.stream, '%');
        let span = self.span();
        if self.arg_env.is_none() {
            // The token keeps its `%` prefix, so it never starts
            // with `::` — `interpret_token` cannot eval-fail here
            // (its only `Err` is the auto-keyword resolver
            // consult). The error maps on a live line, keeping
            // every line covered.
            let opt = interpret_token(&token, &self.opts, hook)
                .map_err(|error| ReaderError::EvalFailed { span, error })?;
            return match opt {
                Some(v) => Ok(Some(v)),
                None => Err(ReaderError::InvalidToken { span, token }),
            };
        }
        // `%99999999999999999999`: `Integer.parseInt` NFE.
        let n = match arg_number(&token) {
            None => return Err(ReaderError::BadArg { span }),
            Some(Err(digits)) => {
                return Err(ReaderError::ArgNumber {
                    span,
                    token: digits,
                });
            }
            Some(Ok(n)) => n,
        };
        let sym = match self.arg_env.as_mut().unwrap().get(&n) {
            Some(sym) => sym.clone(),
            None => {
                let sym = garg(n);
                self.arg_env.as_mut().unwrap().insert(n, sym.clone());
                sym
            }
        };
        Ok(Some(Value::Symbol(sym)))
    }

    /// `SyntaxQuoteReader`: fresh gensym env per form (nesting
    /// rebinds), then [`syntax_quote`](Reader::syntax_quote).
    pub(crate) fn syntax_quote_reader(
        &mut self,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Value>, ReaderError> {
        let old = self.gensym_env.replace(std::collections::HashMap::new());
        let form = match self.read_nested_value(hook) {
            Ok(form) => form,
            Err(e) => {
                self.gensym_env = old;
                return Err(e);
            }
        };
        let quoted = self.syntax_quote(form, hook);
        self.gensym_env = old;
        quoted.map(Some)
    }

    /// `syntaxQuote` (see module docs).
    fn syntax_quote(
        &mut self,
        form: Value,
        hook: &mut dyn ReadHooks,
    ) -> Result<Value, ReaderError> {
        if let Value::Symbol(sym) = &form {
            let sym = sym.clone();
            // Specials quote without qualification.
            let quoted = if is_special(&sym) {
                Value::Symbol(sym)
            } else {
                self.syntax_quote_symbol(sym, hook)?
            };
            return self.with_sq_meta(form, quote_list(quoted), hook);
        }
        if is_unquote(&form) {
            // Raw second, skipping the meta wrap (early return).
            return Ok(second(&form));
        }
        if is_unquote_splicing(&form) {
            return Err(ReaderError::SpliceTop { span: self.span() });
        }
        let ret = self.sq_expand_form(form.clone(), hook)?;
        self.with_sq_meta(form, ret, hook)
    }

    /// Expand one non-symbol form under syntax-quote: collections to
    /// their `apply`/`seq`/`concat` shapes, plain atoms as-is, the
    /// rest quoted.
    fn sq_expand_form(
        &mut self,
        form: Value,
        hook: &mut dyn ReadHooks,
    ) -> Result<Value, ReaderError> {
        match &form {
            Value::Map(m) => {
                let mut keyvals = Vec::with_capacity(m.size() * 2);
                for (k, v) in m.iter() {
                    keyvals.push(k.clone());
                    keyvals.push(v.clone());
                }
                let items = self.sq_expand_list(keyvals, hook)?;
                Ok(apply_shape("hash-map", items))
            }
            Value::Vector(v) => {
                let items = self.sq_expand_list(v.iter().cloned().collect(), hook)?;
                Ok(apply_shape("vector", items))
            }
            Value::Set(s) => {
                let items = self.sq_expand_list(s.iter().cloned().collect(), hook)?;
                Ok(apply_shape("hash-set", items))
            }
            Value::List(l) if l.is_empty() => {
                Ok(Value::List(List::from_vec(vec![core_sym("list")])))
            }
            Value::List(l) => {
                let items = self.sq_expand_list(l.iter().cloned().collect(), hook)?;
                Ok(seq_concat_shape(items))
            }
            Value::Keyword(_)
            | Value::Int(_)
            | Value::Double(_)
            | Value::Char(_)
            | Value::Str(_) => Ok(form.clone()),
            _ => Ok(quote_list(form.clone())),
        }
    }

    /// One `*reader-resolver*` consult (S7): hook failures surface
    /// with the live span (post-form, like the JVM catch position).
    fn resolver_consult(
        &mut self,
        op: ResolverOp,
        sym: Option<&Symbol>,
        hook: &mut dyn ReadHooks,
    ) -> Result<Option<Symbol>, ReaderError> {
        hook.resolve(op, sym)
            .map_err(|error| ReaderError::EvalFailed {
                span: self.span(),
                error,
            })
    }

    /// A nil `currentNS()` under syntax-quote (S7): the JVM derefs
    /// `.name` unconditionally (probe `f-sq-currentnil`).
    fn current_ns_npe(&mut self, hook: &mut dyn ReadHooks) -> ReaderError {
        ReaderError::EvalFailed {
            span: self.span(),
            error: EvalReadError::Failed(Throwable {
                id: hook.fresh_id(),
                class: ThrowClass::NullPointerException,
                message: Some(Arc::from(
                    "Cannot read field \"name\" because the return value of \"clojure.lang.LispReader$Resolver.currentNS()\" is null",
                )),
                data: None,
                cause: None,
            }),
        }
    }

    /// Resolve one symbol under syntax-quote (gensym, `C.`, `.m`,
    /// class/alias/var/current-ns).
    #[allow(clippy::mutable_key_type)]
    fn syntax_quote_symbol(
        &mut self,
        mut sym: Symbol,
        hook: &mut dyn ReadHooks,
    ) -> Result<Value, ReaderError> {
        if sym.namespace().is_none() {
            if let Some(base) = sym.name().strip_suffix('#') {
                let env = self.gensym_env.as_mut().unwrap();
                if let Some(gs) = env.get(&sym) {
                    return Ok(Value::Symbol(gs.clone()));
                }
                let gs = Symbol::new(None, &format!("{base}__{}__auto__", crate::next_id()));
                env.insert(sym, gs.clone());
                return Ok(Value::Symbol(gs));
            }
            if let Some(base) = sym.name().strip_suffix('.') {
                // Resolve the class, then re-add the dot (the full
                // dotted name survives: `` `String. `` ⇒ `java.lang.String.`);
                // a bound resolver answers instead (probe `f-sq-dotclass`).
                if self.opts.resolver_bound {
                    let base_sym = Symbol::new(None, base);
                    let dotted = self
                        .resolver_consult(ResolverOp::ResolveClass, Some(&base_sym), hook)?
                        .map_or_else(|| base.to_owned(), |r| r.name().to_owned());
                    return Ok(Value::Symbol(Symbol::new(None, &format!("{dotted}."))));
                }
                let dotted = self
                    .opts
                    .class_map
                    .get(base)
                    .cloned()
                    .unwrap_or_else(|| base.to_owned());
                sym = Symbol::new(None, &format!("{dotted}."));
                return Ok(Value::Symbol(sym));
            }
            if sym.name().starts_with('.') {
                return Ok(Value::Symbol(sym));
            }
        }
        if sym.namespace().is_some() {
            return self.sq_qualified(sym, hook);
        }
        self.sq_plain(sym, hook)
    }

    /// Resolve a namespace-qualified symbol under syntax-quote
    /// (class namespace, else alias unless dotted). A bound resolver
    /// answers instead (class, then alias, else as-is — probes
    /// `f-sq-qual-*`); the maps are bypassed entirely.
    fn sq_qualified(
        &mut self,
        mut sym: Symbol,
        hook: &mut dyn ReadHooks,
    ) -> Result<Value, ReaderError> {
        let ns = sym.namespace().unwrap().to_owned();
        let name = sym.name().to_owned();
        if self.opts.resolver_bound {
            let ns_sym = Symbol::new(None, &ns);
            let mut resolved =
                self.resolver_consult(ResolverOp::ResolveClass, Some(&ns_sym), hook)?;
            if resolved.is_none() {
                resolved = self.resolver_consult(ResolverOp::ResolveAlias, Some(&ns_sym), hook)?;
            }
            if let Some(r) = resolved {
                let target = r.name().to_owned();
                sym = Symbol::new(Some(&target), &name);
            }
            return Ok(Value::Symbol(sym));
        }
        // A class namespace always rewrites (even `String/b.c` ⇒
        // `java.lang.String/b.c`); otherwise `Compiler.resolveSymbol`
        // leaves dotted names alone before trying the alias
        // (`str/b.c` stays `str/b.c`).
        if let Some(dotted) = self.opts.class_map.get(&ns) {
            sym = Symbol::new(Some(dotted), &name);
        } else {
            let target = if has_inner_dot(&name) {
                None
            } else {
                self.opts.aliases.get(&ns)
            };
            if let Some(target) = target {
                sym = Symbol::new(Some(target), &name);
            }
        }
        Ok(Value::Symbol(sym))
    }

    /// Resolve a plain unqualified symbol under syntax-quote
    /// (dotted ⇒ classname as-is, else class/refers/current-ns). A
    /// bound resolver answers instead — class, then var, else the
    /// current namespace (probes `f-sq-unqual-*`, `f-sq-dotted`,
    /// `f-sq-currentnil`); dottedness is not consulted.
    fn sq_plain(
        &mut self,
        mut sym: Symbol,
        hook: &mut dyn ReadHooks,
    ) -> Result<Value, ReaderError> {
        if self.opts.resolver_bound {
            let mut resolved = self.resolver_consult(ResolverOp::ResolveClass, Some(&sym), hook)?;
            if resolved.is_none() {
                resolved = self.resolver_consult(ResolverOp::ResolveVar, Some(&sym), hook)?;
            }
            match resolved {
                Some(r) => return Ok(Value::Symbol(r)),
                None => match self.resolver_consult(ResolverOp::CurrentNs, None, hook)? {
                    Some(c) => {
                        let current = c.name().to_owned();
                        let name = sym.name().to_owned();
                        sym = Symbol::new(Some(&current), &name);
                        return Ok(Value::Symbol(sym));
                    }
                    None => return Err(self.current_ns_npe(hook)),
                },
            }
        }
        let name = sym.name().to_owned();
        // Dotted unqualified names are classnames: quote as-is
        // (`` `clojure.core `` ⇒ `(quote clojure.core)`).
        if has_inner_dot(&name) {
            return Ok(Value::Symbol(sym));
        }
        if let Some(dotted) = self.opts.class_map.get(&name) {
            sym = Symbol::new(None, dotted);
        } else if let Some((ns, name)) = self.opts.refers.get(&name) {
            sym = Symbol::new(Some(ns), name);
        } else {
            let current = self.opts.current_ns.clone();
            sym = Symbol::new(Some(&current), &name);
        }
        Ok(Value::Symbol(sym))
    }

    /// `sqExpandList`.
    fn sq_expand_list(
        &mut self,
        items: Vec<Value>,
        hook: &mut dyn ReadHooks,
    ) -> Result<Vec<Value>, ReaderError> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if is_unquote(&item) {
                out.push(Value::List(List::from_vec(vec![
                    core_sym("list"),
                    second(&item),
                ])));
            } else if is_unquote_splicing(&item) {
                out.push(second(&item));
            } else {
                let quoted = self.syntax_quote(item, hook)?;
                out.push(Value::List(List::from_vec(vec![core_sym("list"), quoted])));
            }
        }
        Ok(out)
    }

    /// The trailing `with-meta` wrap (line/column filtered out).
    fn with_sq_meta(
        &mut self,
        form: Value,
        ret: Value,
        hook: &mut dyn ReadHooks,
    ) -> Result<Value, ReaderError> {
        let Some(meta) = form.meta() else {
            return Ok(ret);
        };
        let line = Value::Keyword(Keyword::new(None, "line"));
        let column = Value::Keyword(Keyword::new(None, "column"));
        let mut kept = Vec::new();
        for (k, v) in meta.iter() {
            if k != &line && k != &column {
                kept.push((k.clone(), v.clone()));
            }
        }
        if kept.is_empty() {
            return Ok(ret);
        }
        let mut meta_map = clj_value::new_map();
        for (k, v) in kept {
            meta_map = meta_map.insert(k, v);
        }
        let sq_meta = self.syntax_quote(Value::Map(meta_map), hook)?;
        Ok(Value::List(List::from_vec(vec![
            core_sym("with-meta"),
            ret,
            sq_meta,
        ])))
    }
}

/// Whether `^` may attach (Symbol/List/Vector/Map/Set — notably NOT
/// Keyword, String, or anything else).
fn supports_meta(v: &Value) -> bool {
    matches!(
        v,
        Value::Symbol(_) | Value::List(_) | Value::Vector(_) | Value::Map(_) | Value::Set(_)
    )
}

/// The `^` meta-form coercion.
fn coerce_meta(form: Value, span: crate::error::Span) -> Result<clj_value::Map, ReaderError> {
    let tag = Value::Keyword(Keyword::new(None, "tag"));
    match form {
        Value::Symbol(_) | Value::Str(_) => Ok(clj_value::new_map().insert(tag, form)),
        Value::Keyword(_) => Ok(clj_value::new_map().insert(form, Value::Bool(true))),
        Value::Vector(v) => Ok(clj_value::new_map().insert(
            Value::Keyword(Keyword::new(None, "param-tags")),
            Value::Vector(v),
        )),
        Value::Map(m) => Ok(m),
        _ => Err(ReaderError::BadMeta { span }),
    }
}

fn assoc_unless_present(map: clj_value::Map, key: &str, value: Value) -> clj_value::Map {
    let key = Value::Keyword(Keyword::new(None, key));
    if map.contains_key(&key) {
        map
    } else {
        map.insert(key, value)
    }
}

/// `garg(n)`: `p{n}__{id}#` / `rest__{id}#` from the global counter.
fn garg(n: i32) -> Symbol {
    let base = if n == -1 {
        "rest".to_owned()
    } else {
        format!("p{n}")
    };
    Symbol::new(None, &format!("{base}__{}#", crate::next_id()))
}

/// `argPat = %([&]|[1-9][0-9]*)?`: `None` rejects (`%0`, `%00`,
/// `%1x`, …); `Some(Err(digits))` is an overflowing integer (JVM
/// `Integer.parseInt` NFE); otherwise the arg number (`1` for bare
/// `%`, `-1` for `%&`).
fn arg_number(token: &str) -> Option<Result<i32, String>> {
    let rest = token.strip_prefix('%')?;
    if rest.is_empty() {
        return Some(Ok(1));
    }
    if rest == "&" {
        return Some(Ok(-1));
    }
    let mut chars = rest.chars();
    match chars.next() {
        Some('1'..='9') => {}
        _ => return None,
    }
    if !chars.all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(rest.parse::<i32>().map_err(|_| rest.to_owned()))
}

/// `Compiler.isSpecial` (exact set; `import*` qualified).
/// `Compiler.resolveSymbol`'s classname probe: a `.` past the first
/// char (`sym.name.indexOf('.') > 0`).
fn has_inner_dot(name: &str) -> bool {
    name.find('.').is_some_and(|i| i > 0)
}

fn is_special(sym: &Symbol) -> bool {
    const UNQUALIFIED: &[&str] = &[
        "def",
        "loop*",
        "recur",
        "if",
        "case*",
        "let*",
        "letfn*",
        "do",
        "fn*",
        "quote",
        "var",
        ".",
        "set!",
        "deftype*",
        "reify*",
        "try",
        "catch",
        "finally",
        "throw",
        "monitor-enter",
        "monitor-exit",
        "new",
        "&",
    ];
    match sym.namespace() {
        None => UNQUALIFIED.contains(&sym.name()),
        Some(ns) => ns == "clojure.core" && sym.name() == "import*",
    }
}

/// A non-empty list headed by `clojure.core/unquote`.
fn is_unquote(form: &Value) -> bool {
    if let Value::List(l) = form {
        return matches!(
            l.first(),
            Some(Value::Symbol(head))
                if head.namespace() == Some("clojure.core") && head.name() == "unquote"
        );
    }
    false
}

/// A non-empty list headed by `clojure.core/unquote-splicing`.
fn is_unquote_splicing(form: &Value) -> bool {
    if let Value::List(l) = form {
        return matches!(
            l.first(),
            Some(Value::Symbol(head))
                if head.namespace() == Some("clojure.core") && head.name() == "unquote-splicing"
        );
    }
    false
}

/// `RT.second` (`nil` when missing).
fn second(form: &Value) -> Value {
    match form {
        Value::List(l) => l.iter().nth(1).cloned().unwrap_or(Value::Nil),
        _ => Value::Nil,
    }
}

fn core_sym(name: &str) -> Value {
    Value::Symbol(Symbol::new(Some("clojure.core"), name))
}

fn quote_list(form: Value) -> Value {
    Value::List(List::from_vec(vec![
        Value::Symbol(Symbol::new(None, "quote")),
        form,
    ]))
}

/// `(apply {coll} (seq (concat ...)))`.
fn apply_shape(coll: &str, items: Vec<Value>) -> Value {
    let mut concat = vec![core_sym("concat")];
    concat.extend(items);
    Value::List(List::from_vec(vec![
        core_sym("apply"),
        core_sym(coll),
        Value::List(List::from_vec(vec![
            core_sym("seq"),
            Value::List(List::from_vec(concat)),
        ])),
    ]))
}

/// `(seq (concat ...))`.
fn seq_concat_shape(items: Vec<Value>) -> Value {
    let mut concat = vec![core_sym("concat")];
    concat.extend(items);
    Value::List(List::from_vec(vec![
        core_sym("seq"),
        Value::List(List::from_vec(concat)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::{has_inner_dot, is_special, second};
    use crate::{NoHooks, ReaderError, ReaderOpts, read_str};
    use clj_value::{Keyword, Symbol, Value};

    fn read_quoted(input: &str, opts: &ReaderOpts) -> (Option<String>, String) {
        let v = read_str(input, opts, &mut NoHooks).unwrap();
        let Value::List(l) = &v else {
            panic!("expected (quote sym), got {v:?}");
        };
        let items: Vec<&Value> = l.iter().collect();
        assert_eq!(items.len(), 2);
        let Value::Symbol(s) = items[1] else {
            panic!("expected quoted symbol, got {v:?}");
        };
        (s.namespace().map(str::to_owned), s.name().to_owned())
    }

    #[test]
    fn dotted_names_quote_as_is() {
        // `Compiler.resolveSymbol`: a dot past the first char means a
        // classname — no class/var/current-ns resolution applies.
        assert!(has_inner_dot("clojure.core"));
        assert!(has_inner_dot("b.c"));
        assert!(!has_inner_dot("plain"));
        assert!(!has_inner_dot(".leading"));
        let opts = ReaderOpts::default();
        assert_eq!(read_quoted("`a.b", &opts), (None, "a.b".to_owned()));
        assert_eq!(
            read_quoted("`a/b.c", &opts),
            (Some("a".to_owned()), "b.c".to_owned())
        );
        // ... but a class namespace still rewrites first.
        let mut opts = ReaderOpts::default();
        opts.class_map
            .insert("String".to_owned(), "java.lang.String".to_owned());
        opts.aliases
            .insert("str".to_owned(), "clojure.string".to_owned());
        assert_eq!(
            read_quoted("`String/b.c", &opts),
            (Some("java.lang.String".to_owned()), "b.c".to_owned())
        );
        // ... while an alias does not rewrite a dotted name.
        assert_eq!(
            read_quoted("`str/b.c", &opts),
            (Some("str".to_owned()), "b.c".to_owned())
        );
        assert_eq!(
            read_quoted("`str/join", &opts),
            (Some("clojure.string".to_owned()), "join".to_owned())
        );
    }

    #[test]
    fn specials_still_bypass_resolution() {
        assert!(is_special(&Symbol::new(None, "try")));
        assert!(!is_special(&Symbol::new(None, "fn")));
    }

    #[test]
    fn bare_percent_outside_fn_can_be_an_invalid_token() {
        // `arg_reader` outside `#()`: the token path rejects.
        let err = read_str("%a:", &ReaderOpts::default(), &mut NoHooks).unwrap_err();
        assert!(matches!(err, ReaderError::InvalidToken { .. }));
        assert_eq!(err.message(), "Invalid token: %a:");
    }

    #[test]
    fn meta_line_column_already_present_are_kept() {
        // `assoc_unless_present` keep arm: explicit
        // `:line`/`:column` survive on a list target.
        let v = read_str(
            "^{:line 5 :column 7} (a)",
            &ReaderOpts::default(),
            &mut NoHooks,
        )
        .unwrap();
        let meta = v.meta().cloned().expect("list target carries meta");
        let line = Value::Keyword(Keyword::new(None, "line"));
        let column = Value::Keyword(Keyword::new(None, "column"));
        let mut seen_line = false;
        let mut seen_column = false;
        for (k, val) in meta.iter() {
            if k == &line {
                assert_eq!(val, &Value::Int(5), ":line overwritten");
                seen_line = true;
            }
            if k == &column {
                assert_eq!(val, &Value::Int(7), ":column overwritten");
                seen_column = true;
            }
        }
        assert!(seen_line && seen_column);
    }

    #[test]
    fn second_of_non_list_is_nil() {
        // Direct pin: callers pre-guard with list checks, so the
        // fallback never fires through behavior.
        assert_eq!(second(&Value::Int(1)), Value::Nil);
    }

    #[test]
    #[should_panic(expected = "expected (quote sym)")]
    fn read_quoted_helper_rejects_non_lists() {
        let _ = read_quoted("42", &ReaderOpts::default());
    }

    #[test]
    #[should_panic(expected = "expected quoted symbol")]
    fn read_quoted_helper_rejects_non_symbol_second() {
        let _ = read_quoted("'(1 2)", &ReaderOpts::default());
    }

    #[test]
    #[should_panic(expected = "expected (quote sym)")]
    fn read_quoted_hooked_helper_rejects_non_lists() {
        use crate::hooks::ScriptedHooks;

        let mut hooks = ScriptedHooks::new(Err(crate::EvalReadError::NoHook), vec![]);
        let _ = read_quoted_hooked("42", &ReaderOpts::default(), &mut hooks);
    }

    #[test]
    #[should_panic(expected = "expected quoted symbol")]
    fn read_quoted_hooked_helper_rejects_non_symbol_second() {
        use crate::hooks::ScriptedHooks;

        let mut hooks = ScriptedHooks::new(Err(crate::EvalReadError::NoHook), vec![]);
        let _ = read_quoted_hooked("'(1 2)", &ReaderOpts::default(), &mut hooks);
    }

    /// Read `` `sym `` with a scripted resolver, returning the
    /// quoted symbol.
    fn read_quoted_hooked(
        input: &str,
        opts: &ReaderOpts,
        hooks: &mut crate::hooks::ScriptedHooks,
    ) -> (Option<String>, String) {
        let v = read_str(input, opts, hooks).unwrap();
        let Value::List(l) = &v else {
            panic!("expected (quote sym), got {v:?}");
        };
        let items: Vec<&Value> = l.iter().collect();
        assert_eq!(items.len(), 2);
        let Value::Symbol(s) = items[1] else {
            panic!("expected quoted symbol, got {v:?}");
        };
        (s.namespace().map(str::to_owned), s.name().to_owned())
    }

    fn sym(ns: Option<&str>, name: &str) -> Symbol {
        Symbol::new(ns, name)
    }

    #[test]
    fn syntax_quote_consults_the_resolver() {
        // S7 `*reader-resolver*` (probes `f-sq-*`): unqualified tries
        // class, then var, else qualifies with the current ns;
        // qualified tries class, then alias, else stays as-is; `C.`
        // re-dots the class answer. Answers contribute their NAME.
        use crate::hooks::{ResolverOp, ScriptedHooks};

        let opts = ReaderOpts {
            resolver_bound: true,
            ..ReaderOpts::default()
        };
        // Unqualified: class wins wholesale.
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(Some(sym(Some("C"), "D")))],
        );
        assert_eq!(
            read_quoted_hooked("`x", &opts, &mut hooks),
            (Some("C".to_owned()), "D".to_owned())
        );
        assert_eq!(
            hooks.calls,
            vec![(ResolverOp::ResolveClass, Some(sym(None, "x")))]
        );
        // Unqualified: var answers when class declines.
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(None), Ok(Some(sym(Some("V"), "W")))],
        );
        assert_eq!(
            read_quoted_hooked("`x", &opts, &mut hooks),
            (Some("V".to_owned()), "W".to_owned())
        );
        assert_eq!(
            hooks.calls,
            vec![
                (ResolverOp::ResolveClass, Some(sym(None, "x"))),
                (ResolverOp::ResolveVar, Some(sym(None, "x"))),
            ]
        );
        // Unqualified: current ns qualifies (dottedness ignored).
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(None), Ok(None), Ok(Some(sym(None, "cur-ns")))],
        );
        assert_eq!(
            read_quoted_hooked("`x", &opts, &mut hooks),
            (Some("cur-ns".to_owned()), "x".to_owned())
        );
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(None), Ok(None), Ok(Some(sym(None, "cur-ns")))],
        );
        assert_eq!(
            read_quoted_hooked("`clojure.core", &opts, &mut hooks),
            (Some("cur-ns".to_owned()), "clojure.core".to_owned())
        );
        // Qualified: class, then alias, else as-is.
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(Some(sym(Some("R"), "S")))],
        );
        assert_eq!(
            read_quoted_hooked("`String/foo", &opts, &mut hooks),
            (Some("S".to_owned()), "foo".to_owned())
        );
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(None), Ok(Some(sym(Some("X"), "Y")))],
        );
        assert_eq!(
            read_quoted_hooked("`alias/foo", &opts, &mut hooks),
            (Some("Y".to_owned()), "foo".to_owned())
        );
        let mut hooks =
            ScriptedHooks::new(Err(crate::EvalReadError::NoHook), vec![Ok(None), Ok(None)]);
        assert_eq!(
            read_quoted_hooked("`alias/foo", &opts, &mut hooks),
            (Some("alias".to_owned()), "foo".to_owned())
        );
        // `C.`: the class answer re-dots; nil stays as-is.
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(Some(sym(Some("R"), "S")))],
        );
        assert_eq!(
            read_quoted_hooked("`String.", &opts, &mut hooks),
            (None, "S.".to_owned())
        );
        let mut hooks = ScriptedHooks::new(Err(crate::EvalReadError::NoHook), vec![Ok(None)]);
        assert_eq!(
            read_quoted_hooked("`String.", &opts, &mut hooks),
            (None, "String.".to_owned())
        );
    }

    #[test]
    fn syntax_quote_nil_current_ns_is_an_npe() {
        // S7 (probe `f-sq-currentnil`): the JVM derefs `.name`
        // unconditionally.
        use crate::hooks::ScriptedHooks;

        let opts = ReaderOpts {
            resolver_bound: true,
            ..ReaderOpts::default()
        };
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Ok(None), Ok(None), Ok(None)],
        );
        let err = read_str("`x", &opts, &mut hooks).unwrap_err();
        assert!(matches!(err, ReaderError::EvalFailed { .. }));
        assert_eq!(
            err.message(),
            "Cannot read field \"name\" because the return value of \"clojure.lang.LispReader$Resolver.currentNS()\" is null"
        );
        assert_eq!(err.jvm_class(), "NullPointerException");
    }

    #[test]
    fn syntax_quote_reports_consult_failures() {
        // S7: a throwing consult fails the read (the live span rides
        // along, like the JVM catch position).
        use crate::hooks::ScriptedHooks;

        let opts = ReaderOpts {
            resolver_bound: true,
            ..ReaderOpts::default()
        };
        let mut hooks = ScriptedHooks::new(
            Err(crate::EvalReadError::NoHook),
            vec![Err(crate::EvalReadError::NoHook)],
        );
        let err = read_str("`x", &opts, &mut hooks).unwrap_err();
        assert!(matches!(err, ReaderError::EvalFailed { .. }));
        assert_eq!(err.message(), "Eval reader #= requires an evaluator hook");
    }
}
