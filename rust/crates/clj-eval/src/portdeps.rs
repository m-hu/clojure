// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Gap-124 slice 1: the `clojure.tools.deps.util.port` substrate —
//! the natives upstream tools.deps gets from `java.io`/`System`
//! (file reads, canonical paths, the working directory, env vars),
//! which the port deliberately lacks. Installed lazily by
//! [`install_port_ns`] (the gap-123 `install_async_ns` shape):
//! requiring the namespace installs the natives and marks it
//! loaded — never referred into `user`, never in `clojure.core`.
//!
//! `read-edn-string` mirrors `clojure.edn/read` over a whole
//! string: exactly one value (blank answers nil, a second value
//! throws), read errors throw, and every failure is an ex-info
//! carrying `:path` (`io-err` parity: the path absolutizes
//! `getAbsolutePath`-style — joined onto the cwd, never
//! normalized — and a missing path reports `"unknown path"`).
//! Read-eval is forced off: upstream `read-edn` reads through
//! `clojure.edn` (never `clojure.core/read`), so `#=` never
//! evaluates and the `*read-eval*`/`*suppress-read*`/resolver
//! gates are not consulted at all (no `seed_read_gates` call).

use std::sync::Arc;

use clj_reader::{ReadCond, Reader, ReaderOpts};
use clj_value::{Keyword, ThrowClass, Value, new_map};

use crate::core::cce_err;
use crate::error::NativeError;
use crate::eval::Evaluator;
use crate::ns::{NativeFnPtr, check_arity};

/// The port-substrate namespace (native-only, no source file).
pub(crate) const PORT_NS: &str = "clojure.tools.deps.util.port";

/// The missing-path report (upstream `io-err` says `"unknown path"`
/// when no `:path` opt arrives).
const UNKNOWN_PATH: &str = "unknown path";

/// The single-value refusal (upstream `read-edn` inner text,
/// verbatim — it is tools.deps wording, not reader wording).
const SINGLE_VALUE: &str = "Expected edn to contain a single value.";

/// A `read-edn-string` failure: ex-info naming the absolute path,
/// with the same path under `:path` (upstream `io-err` shape).
fn edn_err(inner: &str, abs: &str) -> NativeError {
    let mut err = NativeError::new(ThrowClass::ExceptionInfo, format!("{inner} ({abs})"));
    err.data = Some(Value::Map(new_map().insert(
        Value::Keyword(Keyword::new(None, "path")),
        Value::Str(Arc::from(abs)),
    )));
    err
}

/// The error-report path: `getAbsolutePath` parity (relative joins
/// the cwd lexically, absolute passes through) or `"unknown path"`.
fn edn_abs_path(path: Option<&str>) -> String {
    match path {
        None => UNKNOWN_PATH.to_owned(),
        Some(p) if std::path::Path::new(p).is_absolute() => p.to_owned(),
        Some(p) => std::env::current_dir()
            .unwrap_or_default()
            .join(p)
            .to_string_lossy()
            .into_owned(),
    }
}

/// Reader options for `read-edn-string`: `read-string` shape
/// (current ns, aliases, data-reader snapshot) with read-cond
/// denied and read-eval forced off (clojure.edn parity — deps.edn
/// never evaluates, and the entry gates stay unconsulted).
fn edn_reader_opts(eval: &mut Evaluator) -> Result<ReaderOpts, NativeError> {
    let cur = eval.current_ns_name();
    let mut opts = ReaderOpts {
        current_ns: cur.clone(),
        line_numbers: false,
        read_cond: ReadCond::Deny,
        read_eval: false,
        ..Default::default()
    };
    if let Some(ns) = eval.registry.ns_live_or_removed(&cur) {
        opts.aliases = ns.alias_map();
        opts.refers = ns.refer_map();
    }
    crate::core::snapshot_data_readers(eval, &mut opts)?;
    Ok(opts)
}

/// The `read-edn-string` source arg: nil is the `read-string`
/// bare NPE, anything else non-string fails the `String` cast.
fn edn_string_arg(args: &[Value]) -> Result<&str, NativeError> {
    let s_arg = &args[0];
    if matches!(s_arg, Value::Nil) {
        return Err(NativeError::new(
            ThrowClass::NullPointerException,
            String::new(),
        ));
    }
    let Value::Str(s) = s_arg else {
        return Err(cce_err(s_arg, "java.lang.String"));
    };
    Ok(s)
}

/// The `read-edn-string` report-path arg: missing or nil means
/// none (`"unknown path"` downstream), otherwise a string.
fn edn_path_arg(args: &[Value]) -> Result<Option<&str>, NativeError> {
    match args.get(1) {
        None | Some(Value::Nil) => Ok(None),
        Some(Value::Str(p)) => {
            let s: &str = p;
            Ok(Some(s))
        }
        Some(other) => Err(cce_err(other, "java.lang.String")),
    }
}

/// `read-edn-string`: one whole-string EDN read — blank answers
/// nil, a trailing second value throws, reader failures (including
/// unmatched delimiters and `#=`, which never evaluates here)
/// throw; every failure is an ex-info carrying `:path`. The two
/// `read_form` calls share one `Reader`, so a second value hides
/// behind no amount of comments or whitespace. Arity 2 takes the
/// `:path` report path (nil reports `"unknown path"`).
fn native_read_edn_string(eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity(
        "clojure.tools.deps.util.port/read-edn-string",
        args,
        &[1, 2],
    )?;
    let s = edn_string_arg(args)?;
    let abs = edn_abs_path(edn_path_arg(args)?);
    let opts = edn_reader_opts(eval)?;
    let text: &str = s;
    let mut reader = Reader::new(text, &opts);
    let first = {
        let mut hooks = crate::reader::LiveReadHooks::new(eval);
        reader.read_form(false, &mut hooks)
    };
    let form = match first {
        Ok(Some(v)) => v,
        Ok(None) => return Ok(Value::Nil),
        Err(e) => return Err(edn_err(&e.message(), &abs)),
    };
    let second = {
        let mut hooks = crate::reader::LiveReadHooks::new(eval);
        reader.read_form(false, &mut hooks)
    };
    match second {
        Ok(None) => {}
        Ok(Some(_)) => return Err(edn_err(SINGLE_VALUE, &abs)),
        Err(e) => return Err(edn_err(&e.message(), &abs)),
    }
    let fired = reader.take_fired();
    crate::core::drain_fired_tags(eval, fired, opts.suppress_read)?;
    crate::core::resolve_data_readers(eval, form, opts.suppress_read)
}

/// `file-exists?`: a path string answers whether anything exists
/// there (upstream `File/exists`).
fn native_file_exists(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.tools.deps.util.port/file-exists?", args, &[1])?;
    let Value::Str(path) = &args[0] else {
        return Err(cce_err(&args[0], "java.lang.String"));
    };
    let path_str: &str = path;
    Ok(Value::Bool(std::path::Path::new(path_str).exists()))
}

/// `canonical-path`: the absolute, normalized path string.
/// The 1-arity resolves `path` against the cwd; the 2-arity is
/// `(base, path)` (`java.io.File` parent/child order — the
/// `*the-dir*` context). Rust `fs::canonicalize` requires
/// existence while the JDK does not (spec A3), so the longest
/// existing ancestor prefix canonicalizes and the missing tail
/// rejoins lexically (`.` dropped, `..` popping — the JDK
/// normalizes those lexically too). Total: never throws.
/// Windows verbatim (`\\?\`) prefixes pass through unstripped:
/// comparisons stay canonical-vs-canonical, and stripping would
/// need platform-conditional branches the Linux-only coverage
/// gate cannot see.
fn canonical_path_str(base: Option<&str>, path: &str) -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let anchor = match base {
        Some(b) => {
            let bp = std::path::Path::new(b);
            if bp.is_absolute() {
                bp.to_path_buf()
            } else {
                cwd.join(bp)
            }
        }
        None => cwd,
    };
    let joined = {
        let p = std::path::Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            anchor.join(p)
        }
    };
    // The longest existing ancestor (`ancestors` ends at a root
    // that always exists, so no arm can miss) plus the missing
    // tail, leaf-first. `file_name` answers `None` for a trailing
    // `..`, which records explicitly (dropping it would resolve
    // `sub/..` to `sub/sub` instead of popping); a trailing `.`
    // drops here, which is the join's `.` rule applied early
    // (`components` normalizes `.` away, so no later stage could
    // see one anyway).
    let mut prefix = joined.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    for anc in joined.ancestors() {
        if anc.exists() {
            prefix = anc.to_path_buf();
            break;
        }
        if let Some(name) = anc.file_name() {
            tail.push(name.to_os_string());
        } else if matches!(
            anc.components().next_back(),
            Some(std::path::Component::ParentDir)
        ) {
            tail.push(std::ffi::OsString::from(".."));
        }
    }
    let out = std::fs::canonicalize(&prefix).unwrap_or(prefix);
    join_tail(out, &tail).to_string_lossy().into_owned()
}

/// Rejoin a missing tail (leaf-first) onto its canonical
/// prefix: `..` pops, the rest pushes (the JDK normalizes those
/// lexically too). No `.` arm: the walk drops `.` tails (see
/// above), and `file_name` never yields one, so `.` cannot reach
/// the join.
fn join_tail(mut out: std::path::PathBuf, tail: &[std::ffi::OsString]) -> std::path::PathBuf {
    for comp in tail.iter().rev() {
        if comp.to_string_lossy() == ".." {
            out.pop();
        } else {
            out.push(comp);
        }
    }
    out
}

/// The `canonical-path` native (see [`canonical_path_str`]).
fn native_canonical_path(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.tools.deps.util.port/canonical-path", args, &[1, 2])?;
    let (base, path) = match args.len() {
        1 => (None, &args[0]),
        _ => (Some(&args[0]), &args[1]),
    };
    let Value::Str(path) = path else {
        return Err(cce_err(path, "java.lang.String"));
    };
    let base = match base {
        None => None,
        Some(Value::Str(b)) => {
            let s: &str = b;
            Some(s)
        }
        Some(other) => return Err(cce_err(other, "java.lang.String")),
    };
    let path_str: &str = path;
    let out = canonical_path_str(base, path_str);
    Ok(Value::Str(Arc::from(out.as_str())))
}

/// `current-dir`: the process working directory
/// (upstream `user.dir`).
fn native_current_dir(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.tools.deps.util.port/current-dir", args, &[0])?;
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    Ok(Value::Str(Arc::from(cwd.as_str())))
}

/// `getenv`: the env var value, or nil when unset or
/// non-Unicode (upstream `System/getenv`, nil-able).
fn native_getenv(_eval: &mut Evaluator, args: &[Value]) -> Result<Value, NativeError> {
    check_arity("clojure.tools.deps.util.port/getenv", args, &[1])?;
    let Value::Str(name) = &args[0] else {
        return Err(cce_err(&args[0], "java.lang.String"));
    };
    let name_str: &str = name;
    match std::env::var(name_str) {
        Ok(v) => Ok(Value::Str(Arc::from(v.as_str()))),
        Err(_) => Ok(Value::Nil),
    }
}

/// The port-substrate natives (installed lazily by
/// [`install_port_ns`] — never referred into `user`).
pub(crate) const PORT_NATIVES: &[(&str, NativeFnPtr)] = &[
    ("read-edn-string", native_read_edn_string),
    ("file-exists?", native_file_exists),
    ("canonical-path", native_canonical_path),
    ("current-dir", native_current_dir),
    ("getenv", native_getenv),
];

/// Install the `clojure.tools.deps.util.port` namespace
/// (idempotent — reinstalling over itself is a `:reload` no-op).
pub(crate) fn install_port_ns(eval: &mut Evaluator) {
    eval.registry.ensure_ns(PORT_NS);
    for (name, f) in PORT_NATIVES {
        let func: Arc<crate::ext::NativeImpl> = Arc::new(*f);
        eval.install_native(PORT_NS, name, func);
    }
}
