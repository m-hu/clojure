// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Shared differential-corpus harness: EDN interpretation (the data
//! files are read with our own reader), the canonical printer
//! (mirroring `tests/corpus/driver.clj` exactly), gensym
//! normalization, FNV-1a file hashes, and outcome comparison.
//!
//! Canonicalization: map/set order (sorted by canonical rendering,
//! byte order — UTF-8 order equals codepoint order, matching the
//! driver's codepoint-vector sort), namespace-map lifting, and the
//! `^meta` prefix rules. Leaves render through the real printer, so
//! for order-deterministic values the canonical form equals `pr`
//! exactly.

//! Shared by `corpus` and `snapshot`, which use different subsets.
#![allow(dead_code)]

use std::collections::HashMap;

use clj_reader::{NoHooks, ReadCond, ReaderOpts, read_all};
use clj_value::{Keyword, PrintOpts, Symbol, Value, pr_to_string};

/// A corpus case (from `cases.edn`).
pub struct Case {
    pub id: String,
    pub input: String,
    pub read_cond: ReadCond,
    pub features: Vec<Keyword>,
    pub diverge: Option<String>,
}

/// The oracle resolution context (from `expectations.edn`).
pub struct Resolver {
    pub current_ns: String,
    pub aliases: HashMap<String, String>,
    pub refers: HashMap<String, (String, String)>,
    pub class_map: HashMap<String, String>,
}

/// One read form's oracle/snapshot renderings (`None` when
/// unrepresentable — lone-surrogate outputs).
pub struct OkForm {
    pub raw: Option<String>,
    pub canon: Option<String>,
    pub canon_meta: Option<String>,
}

/// An error triple.
#[derive(Debug, PartialEq, Eq)]
pub struct ErrInfo {
    pub class: String,
    pub message: String,
    pub line: Option<u32>,
    pub col: Option<u32>,
}

/// A read outcome (oracle `:jvm` or reviewed `:ours`).
pub enum Outcome {
    Ok(Vec<OkForm>),
    Err(ErrInfo),
}

// ---------------------------------------------------------------- EDN interpretation

fn kw_name(v: &Value) -> &str {
    match v {
        Value::Keyword(k) => {
            assert!(k.namespace().is_none(), "expected unqualified key, got {k}");
            k.name()
        }
        other => panic!("expected keyword, got {other:?}"),
    }
}

fn expect_string(v: &Value) -> String {
    match v {
        Value::Str(s) => s.to_string(),
        other => panic!("expected string, got {other:?}"),
    }
}

fn expect_int(v: &Value) -> i64 {
    match v {
        Value::Int(i) => *i,
        other => panic!("expected int, got {other:?}"),
    }
}

fn expect_vec(v: &Value) -> Vec<Value> {
    match v {
        Value::Vector(items) => items.iter().cloned().collect(),
        other => panic!("expected vector, got {other:?}"),
    }
}

fn map_get<'a>(m: &'a clj_value::Map, key: &str) -> Option<&'a Value> {
    m.get(&Value::Keyword(Keyword::new(None, key)))
}

fn expect_map(v: &Value) -> &clj_value::Map {
    match v {
        Value::Map(m) => m,
        other => panic!("expected map, got {other:?}"),
    }
}

fn parse_keyword(v: &Value) -> Keyword {
    match v {
        Value::Keyword(k) => k.clone(),
        other => panic!("expected keyword, got {other:?}"),
    }
}

/// Read one EDN data file with our own reader (single top-level form).
pub fn read_data_file(path: &std::path::Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap();
    let opts = ReaderOpts::default();
    let forms = read_all(&text, &opts, &mut NoHooks).unwrap();
    assert_eq!(forms.len(), 1, "{} must hold one form", path.display());
    forms.into_iter().next().unwrap()
}

/// Parse `cases.edn`.
pub fn parse_cases(v: &Value) -> Vec<Case> {
    match v {
        Value::Vector(items) => items
            .iter()
            .map(|c| {
                let m = expect_map(c);
                let opts = expect_map(map_get(m, "opts").unwrap());
                let read_cond = match kw_name(map_get(opts, "read-cond").unwrap()) {
                    "allow" => ReadCond::Allow,
                    "preserve" => ReadCond::Preserve,
                    "deny" => ReadCond::Deny,
                    other => panic!("bad read-cond {other}"),
                };
                let features = match map_get(opts, "features") {
                    Some(f) => expect_vec(f).iter().map(parse_keyword).collect(),
                    None => Vec::new(),
                };
                Case {
                    id: expect_string(map_get(m, "id").unwrap()),
                    input: expect_string(map_get(m, "input").unwrap()),
                    read_cond,
                    features,
                    diverge: map_get(m, "diverge").map(|d| kw_name(d).to_owned()),
                }
            })
            .collect(),
        other => panic!("expected cases vector, got {other:?}"),
    }
}

/// Parse the `:resolver` block of `expectations.edn`.
pub fn parse_resolver(v: &Value) -> Resolver {
    let m = expect_map(v);
    let r = expect_map(map_get(m, "resolver").unwrap());
    let str_map = |key: &str| {
        expect_map(map_get(r, key).unwrap())
            .iter()
            .map(|(k, v)| (expect_string(k), expect_string(v)))
            .collect()
    };
    let refers = expect_map(map_get(r, "refers").unwrap())
        .iter()
        .map(|(k, v)| {
            let pair = expect_vec(v);
            assert_eq!(pair.len(), 2);
            (
                expect_string(k),
                (expect_string(&pair[0]), expect_string(&pair[1])),
            )
        })
        .collect();
    Resolver {
        current_ns: expect_string(map_get(r, "current-ns").unwrap()),
        aliases: str_map("aliases"),
        refers,
        class_map: str_map("class-map"),
    }
}

/// Parse one `:jvm`/`:ours` outcome block.
pub fn parse_outcome(v: &Value) -> Outcome {
    let m = expect_map(v);
    if let Some(ok) = map_get(m, "ok") {
        let forms = expect_vec(ok)
            .iter()
            .map(|f| {
                let f = expect_map(f);
                let opt_str = |key: &str| match map_get(f, key).unwrap() {
                    Value::Str(s) => Some(s.to_string()),
                    Value::Keyword(_) => None,
                    other => panic!("bad rendering {other:?}"),
                };
                OkForm {
                    raw: opt_str("raw"),
                    canon: opt_str("canon"),
                    canon_meta: opt_str("canon-meta"),
                }
            })
            .collect();
        Outcome::Ok(forms)
    } else {
        let e = expect_map(map_get(m, "err").unwrap());
        let opt_u32 = |key: &str| match map_get(e, key).unwrap() {
            Value::Int(i) => Some(u32::try_from(*i).unwrap()),
            Value::Nil => None,
            other => panic!("bad position {other:?}"),
        };
        Outcome::Err(ErrInfo {
            class: expect_string(map_get(e, "class").unwrap()),
            message: expect_string(map_get(e, "message").unwrap()),
            line: opt_u32("line"),
            col: opt_u32("col"),
        })
    }
}

/// Parse the `:cases` block of `expectations.edn`: id → outcome.
pub fn parse_expectations(v: &Value) -> HashMap<String, Outcome> {
    let m = expect_map(v);
    let cases = expect_map(map_get(m, "cases").unwrap());
    cases
        .iter()
        .map(|(id, entry)| {
            let entry = expect_map(entry);
            (
                expect_string(id),
                parse_outcome(map_get(entry, "jvm").unwrap()),
            )
        })
        .collect()
}

/// Parse `ours.edn` (absent file → empty): id → outcome.
pub fn parse_ours(path: &std::path::Path) -> HashMap<String, Outcome> {
    if !path.exists() {
        return HashMap::new();
    }
    let v = read_data_file(path);
    expect_map(&v)
        .iter()
        .map(|(id, outcome)| (expect_string(id), parse_outcome(outcome)))
        .collect()
}

/// A full-file corpus entry (from `files.edn`).
pub struct FileEntry {
    pub path: String,
    pub count: usize,
    pub fnv_default: String,
    pub fnv_printmeta: String,
    pub exclude: Option<String>,
}

/// Parse `files.edn`.
pub fn parse_files(v: &Value) -> Vec<FileEntry> {
    match v {
        Value::Vector(items) => items
            .iter()
            .map(|f| {
                let m = expect_map(f);
                FileEntry {
                    path: expect_string(map_get(m, "path").unwrap()),
                    count: usize::try_from(map_get(m, "count").map(expect_int).unwrap_or(0))
                        .unwrap(),
                    fnv_default: map_get(m, "fnv-default")
                        .map(expect_string)
                        .unwrap_or_default(),
                    fnv_printmeta: map_get(m, "fnv-printmeta")
                        .map(expect_string)
                        .unwrap_or_default(),
                    exclude: map_get(m, "exclude").map(expect_string),
                }
            })
            .collect(),
        other => panic!("expected files vector, got {other:?}"),
    }
}

// ---------------------------------------------------------------- running

/// Reader options for a case (resolver maps installed; `:clj`
/// installed by the reader itself).
pub fn case_opts(case: &Case, resolver: &Resolver) -> ReaderOpts {
    ReaderOpts {
        read_cond: case.read_cond,
        features: case.features.iter().cloned().collect(),
        line_numbers: true,
        current_ns: resolver.current_ns.clone(),
        aliases: resolver.aliases.clone(),
        refers: resolver.refers.clone(),
        class_map: resolver.class_map.clone(),
        // The corpus drives the standalone reader: defer mode stays
        // off (S6 consults only on eval paths); the gate stays
        // closed like the driver's `(binding [*read-eval* false]
        // ...)` (#125: the `#=` case compares conformantly).
        read_eval: false,
        data_readers: false,
        data_reader_keys: std::collections::HashSet::new(),
        data_readers_borked: None,
        default_reader_fn: false,
        ..ReaderOpts::default()
    }
}

/// Run our reader over a case input, probe-shaped: the first form
/// with EOF errors, the rest EOF-terminated, one reader throughout
/// (so all-discarded inputs error like the oracle).
pub fn run_ours(input: &str, opts: &ReaderOpts) -> Result<Vec<Value>, clj_reader::ReaderError> {
    let mut reader = clj_reader::Reader::new(input, opts);
    let mut forms = Vec::new();
    if let Some(first) = reader.read_form(true, &mut NoHooks)? {
        forms.push(first);
    }
    while let Some(v) = reader.read_form(false, &mut NoHooks)? {
        forms.push(v);
    }
    Ok(forms)
}

/// Render our outcome (canons normalized by the caller).
pub fn our_outcome(values: Vec<Value>) -> Outcome {
    Outcome::Ok(
        values
            .into_iter()
            .map(|v| OkForm {
                raw: Some(pr_to_string(&v, &PrintOpts::pr())),
                canon: Some(canon(&v, false)),
                canon_meta: Some(canon(&v, true)),
            })
            .collect(),
    )
}

pub fn our_err(e: &clj_reader::ReaderError) -> Outcome {
    Outcome::Err(ErrInfo {
        class: e.jvm_class().to_owned(),
        message: e.message(),
        line: Some(e.span().line),
        col: Some(e.span().col),
    })
}

// ---------------------------------------------------------------- canonical printer

/// Canonical rendering (mirrors the driver's `canon`): leaves via
/// the real printer; collections traversed with sorted entries,
/// namespace-map lifting, and the `^meta` prefix rules.
pub fn canon(v: &Value, print_meta: bool) -> String {
    let mut out = String::new();
    append_canon_meta(&mut out, v, print_meta);
    append_canon_body(&mut out, v, print_meta);
    out
}

fn canon_meta(v: &Value, print_meta: bool) -> Option<&clj_value::Map> {
    if print_meta {
        v.meta().filter(map_nonempty)
    } else {
        None
    }
}

fn map_nonempty(m: &&clj_value::Map) -> bool {
    !m.is_empty()
}

/// `^meta` prefix: a lone truthy `:tag` prints bare, otherwise the map.
fn append_canon_meta(out: &mut String, v: &Value, print_meta: bool) {
    let Some(m) = canon_meta(v, print_meta) else {
        return;
    };
    let tag_key = Value::Keyword(Keyword::new(None, "tag"));
    out.push('^');
    match m.get(&tag_key) {
        Some(tag) if m.size() == 1 && tag.is_truthy() => out.push_str(&canon(tag, print_meta)),
        _ => out.push_str(&canon_map(m, print_meta)),
    }
    out.push(' ');
}

fn append_canon_body(out: &mut String, v: &Value, print_meta: bool) {
    match v {
        Value::Nil
        | Value::Bool(_)
        | Value::Int(_)
        | Value::Double(_)
        | Value::BigInt(_)
        | Value::BigInteger(_)
        | Value::BigDec(_)
        | Value::Ratio(_)
        | Value::Str(_)
        | Value::Char(_)
        | Value::Symbol(_)
        | Value::Keyword(_)
        | Value::Regex(_)
        // S9 scalars print canonically (tagged `#inst`/`#uuid`/bare URI).
        | Value::Uuid(_)
        | Value::Instant(_)
        | Value::Uri(_)
        // Evaluator-only values (the reader never emits them).
        | Value::HostType(_)
        | Value::MultiFn(_)
        | Value::Class(_)
        | Value::Array(_)
        | Value::PrimitiveVector(_)
        | Value::Var(_)
        | Value::Unbound(_)
        | Value::Transient(_)
        | Value::Throwable(_)
        | Value::Atom(_)
        | Value::Delay(_)
        | Value::Future(_)
        | Value::Promise(_)
        | Value::Agent(_)
        | Value::Volatile(_)
        | Value::Ref(_)
        // Executors are evaluator-only too (`(executor)` builds
        // them; the reader never emits them).
        | Value::Executor(_)
        | Value::MapEntry { .. }
        | Value::Extension(_) => out.push_str(&pr_to_string(v, &PrintOpts::pr())),
        Value::List(l) => {
            out.push('(');
            join_canon(out, l.iter(), " ", print_meta);
            out.push(')');
        }
        Value::Vector(items) => {
            out.push('[');
            join_canon(out, items.iter(), " ", print_meta);
            out.push(']');
        }
        Value::Map(m) => out.push_str(&canon_map(m, print_meta)),
        Value::Set(s) => append_canon_set(out, s, print_meta),
        Value::TaggedLiteral { tag, form, .. } => {
            out.push('#');
            out.push_str(&tag.to_string());
            out.push(' ');
            out.push_str(&canon(form, print_meta));
        }
        Value::ReaderConditional { form, splicing, .. } => {
            out.push_str(reader_cond_prefix(*splicing));
            out.push_str(&canon(form, print_meta));
        }
    }
}

fn reader_cond_prefix(splicing: bool) -> &'static str {
    if splicing { "#?@" } else { "#?" }
}

fn append_canon_set(out: &mut String, s: &clj_value::Set, print_meta: bool) {
    let mut members: Vec<String> = s.iter().map(|m| canon(m, print_meta)).collect();
    members.sort();
    out.push_str("#{");
    out.push_str(&members.join(" "));
    out.push('}');
}

fn join_canon<'a>(
    out: &mut String,
    items: impl Iterator<Item = &'a Value>,
    sep: &str,
    print_meta: bool,
) {
    let mut first = true;
    for v in items {
        if !first {
            out.push_str(sep);
        }
        first = false;
        out.push_str(&canon(v, print_meta));
    }
}

/// Canonical map rendering with lifting (mirrors `lift-ns` + sort).
fn canon_map(m: &clj_value::Map, print_meta: bool) -> String {
    let mut entries: Vec<(&Value, &Value)> = m.iter().collect();
    let lifted = lift_ns(&entries);
    if let Some((ns, stripped)) = &lifted {
        entries = stripped.iter().map(|(k, v)| (k, v)).collect();
        let mut rendered = sorted_entries(&entries, print_meta);
        rendered.sort();
        format!("#:{ns}{{{}}}", rendered.join(", "))
    } else {
        let mut rendered = sorted_entries(&entries, print_meta);
        rendered.sort();
        format!("{{{}}}", rendered.join(", "))
    }
}

fn sorted_entries(entries: &[(&Value, &Value)], print_meta: bool) -> Vec<String> {
    entries
        .iter()
        .map(|(k, v)| format!("{} {}", canon(k, print_meta), canon(v, print_meta)))
        .collect()
}

/// Mirror `lift-ns`: all keys qualified with one namespace.
fn lift_ns(entries: &[(&Value, &Value)]) -> Option<(String, Vec<(Value, Value)>)> {
    let ns = shared_ns(entries)?;
    Some((ns.to_owned(), lift_stripped(entries)))
}

fn shared_ns<'a>(entries: &'a [(&'a Value, &'a Value)]) -> Option<&'a str> {
    if entries.is_empty() {
        return None;
    }
    let mut ns: Option<&str> = None;
    for (k, _) in entries {
        let key_ns = key_namespace(k)?;
        ns = match ns {
            None => Some(key_ns),
            Some(n) if n == key_ns => Some(n),
            Some(_) => return None,
        };
    }
    ns
}

fn key_namespace(k: &Value) -> Option<&str> {
    match k {
        Value::Keyword(k) => k.namespace(),
        Value::Symbol(s) => s.namespace(),
        _ => None,
    }
}

fn lift_stripped(entries: &[(&Value, &Value)]) -> Vec<(Value, Value)> {
    let mut stripped = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        stripped.push((strip_ns_key(k), (*v).clone()));
    }
    stripped
}

fn strip_ns_key(k: &Value) -> Value {
    match k {
        Value::Keyword(k) => Value::Keyword(Keyword::new(None, k.name())),
        Value::Symbol(s) => Value::Symbol(Symbol::new(None, s.name())),
        _ => k.clone(),
    }
}

// ---------------------------------------------------------------- normalization + hashing

/// Normalize gensym ids (`__12#`, `__12__auto__` → `__N#`,
/// `__N__auto__`) on both sides before comparison.
pub fn normalize_gensyms(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if s[i..].starts_with("__") {
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 2 && s[j..].starts_with('#') {
                out.push_str("__N#");
                i = j + 1;
                continue;
            }
            if j > i + 2 && s[j..].starts_with("__auto__") {
                out.push_str("__N__auto__");
                i = j + 8;
                continue;
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// FNV-1a 64 over UTF-8 bytes, lowercase hex (matches the driver).
pub fn fnv1a_hex(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

// ---------------------------------------------------------------- comparison

/// Compare our outcome against an expected one (exact for `:jvm`,
/// snapshot for `:ours`); `Err` describes the first difference.
/// Compare one rendered pair after gensym normalization (`raw`
/// adds the raw renderings to the mismatch message, as before).
fn compare_rendered(
    i: usize,
    label: &str,
    x: &Option<String>,
    y: &Option<String>,
    raw: Option<(&Option<String>, &Option<String>)>,
) -> Result<(), String> {
    match (x, y) {
        (Some(xc), Some(yc)) => {
            let xc = normalize_gensyms(xc);
            let yc = normalize_gensyms(yc);
            if xc != yc {
                let mut msg = format!("form {i} {label}:\n  ours {xc:?}\n  want {yc:?}");
                if let Some((xr, yr)) = raw {
                    msg.push_str(&format!("\n  ours raw {xr:?}\n  want raw {yr:?}"));
                }
                return Err(msg);
            }
            Ok(())
        }
        _ => Err(format!("form {i}: unrepresentable rendering compared")),
    }
}

pub fn compare_outcomes(ours: &Outcome, want: &Outcome) -> Result<(), String> {
    match (ours, want) {
        (Outcome::Ok(a), Outcome::Ok(b)) => {
            if a.len() != b.len() {
                return Err(format!("form count: ours {} vs want {}", a.len(), b.len()));
            }
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                compare_rendered(i, "canon", &x.canon, &y.canon, Some((&x.raw, &y.raw)))?;
                compare_rendered(i, "canon-meta", &x.canon_meta, &y.canon_meta, None)?;
            }
            Ok(())
        }
        (Outcome::Err(a), Outcome::Err(b)) => {
            if a == b {
                Ok(())
            } else {
                Err(format!("error:\n  ours {a:?}\n  want {b:?}"))
            }
        }
        (Outcome::Ok(_), Outcome::Err(b)) => Err(format!("ours read ok, want error {b:?}")),
        (Outcome::Err(a), Outcome::Ok(_)) => Err(format!("ours error {a:?}, want ok")),
    }
}

#[cfg(test)]
mod tests {
    use super::{fnv1a_hex, normalize_gensyms};

    #[test]
    fn fnv_vectors() {
        assert_eq!(fnv1a_hex(""), "cbf29ce484222325");
        assert_eq!(fnv1a_hex("a"), "af63dc4c8601ec8c");
        assert_eq!(fnv1a_hex("foobar"), "85944171f73967e8");
    }

    #[test]
    fn gensym_normalization() {
        assert_eq!(normalize_gensyms("p1__12#"), "p1__N#");
        assert_eq!(normalize_gensyms("x__7__auto__"), "x__N__auto__");
        assert_eq!(normalize_gensyms("a__b #_1"), "a__b #_1");
        assert_eq!(
            normalize_gensyms("(fn* [p1__1#] p1__1#)"),
            "(fn* [p1__N#] p1__N#)"
        );
    }
}
