// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! S4 value types: [`Var`], [`Unbound`], [`Throwable`]/[`ThrowClass`],
//! and the [`ExtensionValue`] trait for host-defined values.
//!
//! Vars, unbound sentinels, and throwables all compare by identity
//! (their `id`), like the JVM — never by payload. Extension values
//! compare through the trait's [`dyn_eq`](ExtensionValue::dyn_eq).

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use crate::{Symbol, Value};

/// A namespace Var: `id` is its identity (equality and hashing use
/// the `id` only), `sym` its qualified name (used for printing).
#[derive(Clone, Debug)]
pub struct Var {
    pub id: u64,
    pub sym: Symbol,
}

/// The sentinel root of an unbound Var: `var_id` is its identity,
/// `sym` the Var's qualified name (used for printing).
#[derive(Clone, Debug)]
pub struct Unbound {
    pub sym: Symbol,
    pub var_id: u64,
}

/// The JVM exception classes this runtime can throw. `jvm_name` is
/// the class the printer reports and `catch` matches against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThrowClass {
    Exception,
    Error,
    RuntimeException,
    IllegalArgumentException,
    IllegalStateException,
    /// `IllegalMonitorStateException` (fence 4b F8b): a
    /// `RuntimeException` sibling of `IllegalStateException`
    /// (`javap` verdict; unmatched `monitor-exit`).
    IllegalMonitorStateException,
    UnsupportedOperationException,
    AssertionError,
    ArityException,
    NullPointerException,
    ClassCastException,
    IndexOutOfBoundsException,
    ArrayIndexOutOfBoundsException,
    ArrayStoreException,
    NegativeArraySizeException,
    OutOfMemoryError,
    ArithmeticException,
    ExceptionInfo,
    FileNotFoundException,
    IllegalAccessError,
    NumberFormatException,
    StringIndexOutOfBoundsException,
    IllegalFormatConversionException,
    UnknownFormatConversionException,
    MissingFormatConversionException,
    MissingFormatArgumentException,
    IllegalFormatPrecisionException,
    IllegalFormatCodePointException,
    FormatFlagsConversionMismatchException,
    DuplicateFormatFlagsException,
    IllegalFormatFlagsException,
    IllegalFormatWidthException,
    MissingFormatWidthException,
    IllegalFormatArgumentIndexException,
    IOException,
    /// `UnsupportedEncodingException` (#120 slice 2 `String`
    /// construction): a checked `IOException` for unknown charset
    /// names.
    UnsupportedEncodingException,
    CompilerException,
    ExecutionException,
    CancellationException,
    RejectedExecutionException,
    PatternSyntaxException,
    ReaderException,
    /// `ClassNotFoundException` (S16 `import`, wave1): a checked exception
    /// under `ReflectiveOperationException`.
    ClassNotFoundException,
    /// `NoSuchElementException` (S16 `getFirst`/`getLast` on empty
    /// lists): a `RuntimeException`.
    NoSuchElementException,
}

impl ThrowClass {
    /// The JVM binary name, e.g. `java.lang.IllegalArgumentException`.
    #[must_use]
    pub fn jvm_name(&self) -> &'static str {
        match self {
            Self::Exception => "java.lang.Exception",
            Self::Error => "java.lang.Error",
            Self::RuntimeException => "java.lang.RuntimeException",
            Self::IllegalArgumentException => "java.lang.IllegalArgumentException",
            Self::IllegalStateException => "java.lang.IllegalStateException",
            Self::IllegalMonitorStateException => "java.lang.IllegalMonitorStateException",
            Self::UnsupportedOperationException => "java.lang.UnsupportedOperationException",
            Self::AssertionError => "java.lang.AssertionError",
            Self::ArityException => "clojure.lang.ArityException",
            Self::NullPointerException => "java.lang.NullPointerException",
            Self::ClassCastException => "java.lang.ClassCastException",
            Self::IndexOutOfBoundsException => "java.lang.IndexOutOfBoundsException",
            Self::ArrayIndexOutOfBoundsException => "java.lang.ArrayIndexOutOfBoundsException",
            Self::NegativeArraySizeException => "java.lang.NegativeArraySizeException",
            // `ArithmeticException` lives in `jvm_name_rest` (which
            // has budget); the chain still matches every variant
            // exactly once.
            other => other.jvm_name_mid(),
        }
    }

    /// The JVM simple name (`Class.getSimpleName`): the binary
    /// name past the last package separator or outer class
    /// (`LispReader$ReaderException` ⇒ `ReaderException`). The
    /// reader reports it via `jvm_class()` (#125).
    #[must_use]
    pub fn simple_name(&self) -> &'static str {
        let full = self.jvm_name();
        let cut = full.rfind(['.', '$']).map_or(0, |i| i + 1);
        &full[cut..]
    }

    /// Format and I/O throw classes. Chained so each match stays under
    /// the complexity gate.
    fn jvm_name_mid(&self) -> &'static str {
        match self {
            Self::ExceptionInfo => "clojure.lang.ExceptionInfo",
            Self::FileNotFoundException => "java.io.FileNotFoundException",
            Self::UnsupportedEncodingException => "java.io.UnsupportedEncodingException",
            Self::IllegalAccessError => "java.lang.IllegalAccessError",
            Self::NumberFormatException => "java.lang.NumberFormatException",
            Self::StringIndexOutOfBoundsException => "java.lang.StringIndexOutOfBoundsException",
            Self::PatternSyntaxException => "java.util.regex.PatternSyntaxException",
            Self::IllegalFormatConversionException => "java.util.IllegalFormatConversionException",
            Self::UnknownFormatConversionException => "java.util.UnknownFormatConversionException",
            Self::MissingFormatConversionException => "java.util.MissingFormatConversionException",
            other => other.jvm_name_mid_tail(),
        }
    }

    /// Trailing format throw classes, split from
    /// [`ThrowClass::jvm_name_mid`] for the complexity budget.
    fn jvm_name_mid_tail(&self) -> &'static str {
        match self {
            Self::MissingFormatArgumentException => "java.util.MissingFormatArgumentException",
            Self::IllegalFormatPrecisionException => "java.util.IllegalFormatPrecisionException",
            Self::IllegalFormatCodePointException => "java.util.IllegalFormatCodePointException",
            Self::FormatFlagsConversionMismatchException => {
                "java.util.FormatFlagsConversionMismatchException"
            }
            other => other.jvm_name_rest(),
        }
    }

    /// Remaining throw classes (the S16 classes live in
    /// [`ThrowClass::jvm_name_s16`], split for the complexity
    /// budget).
    fn jvm_name_rest(&self) -> &'static str {
        match self {
            Self::ArithmeticException => "java.lang.ArithmeticException",
            Self::DuplicateFormatFlagsException => "java.util.DuplicateFormatFlagsException",
            Self::IllegalFormatFlagsException => "java.util.IllegalFormatFlagsException",
            Self::IllegalFormatWidthException => "java.util.IllegalFormatWidthException",
            Self::MissingFormatWidthException => "java.util.MissingFormatWidthException",
            Self::IllegalFormatArgumentIndexException => {
                "java.util.IllegalFormatArgumentIndexException"
            }
            Self::IOException => "java.io.IOException",
            Self::CompilerException => "clojure.lang.Compiler$CompilerException",
            Self::ExecutionException => "java.util.concurrent.ExecutionException",
            Self::CancellationException => "java.util.concurrent.CancellationException",
            Self::RejectedExecutionException => "java.util.concurrent.RejectedExecutionException",
            Self::ReaderException => "clojure.lang.LispReader$ReaderException",
            // The S16 classes plus anything upstream-handled (which
            // the tail rejects — the single caller partitions the
            // variants, so only a dispatch bug could send one here).
            other => other.jvm_name_s16(),
        }
    }

    /// The S16 throw classes (split from [`ThrowClass::jvm_name_rest`]
    /// for the complexity budget).
    fn jvm_name_s16(&self) -> &'static str {
        match self {
            Self::ClassNotFoundException => "java.lang.ClassNotFoundException",
            Self::NoSuchElementException => "java.util.NoSuchElementException",
            other => other.jvm_name_array(),
        }
    }

    /// Array exceptions plus the error class that arrived with
    /// the array slice (S15). Chained so each match stays under
    /// the complexity gate.
    fn jvm_name_array(&self) -> &'static str {
        match self {
            // Only `ArrayStoreException` and `OutOfMemoryError`
            // reach here: `ArrayIndexOutOfBoundsException` and
            // `NegativeArraySizeException` are answered directly
            // by `jvm_name`, like every other upstream variant.
            Self::ArrayStoreException => "java.lang.ArrayStoreException",
            Self::OutOfMemoryError => "java.lang.OutOfMemoryError",
            // All other variants are handled upstream (the single
            // caller partitions the variants).
            _ => unreachable!("throw-class partition"),
        }
    }

    /// Whether a `catch` on `catch_jvm_name` catches this class:
    /// the name itself or any superclass up the chain (`Throwable`
    /// catches everything).
    #[must_use]
    pub fn matches(&self, catch_jvm_name: &str) -> bool {
        let mut current = self.jvm_name();
        loop {
            if current == catch_jvm_name {
                return true;
            }
            match throw_parent(current) {
                Some(parent) => current = parent,
                None => return false,
            }
        }
    }
}

/// A JVM `ClassCastException` text with JDK module verbiage: same
/// modules combine (`A and B are in M`), different modules split
/// (`A is in M1; B is in M2`) — byte-pinned by probe39. Shared home
/// (moved from `clj-eval` for S6): the reader's spanful `TableCce`
/// and the tag validators format through this too.
#[must_use]
pub fn cce_message(from_jvm: &str, to_jvm: &str) -> String {
    fn module_of(cls: &str) -> &'static str {
        // S17: `java.sql` ships in its own module (oracle
        // `g-rs-argerr`).
        if cls.starts_with("java.sql.") {
            "module java.sql of loader 'platform'"
        } else if cls.starts_with("java.") || cls.starts_with('[') {
            // Array descriptors (`[I`, …) live in `java.base` too
            // (S15 oracle `d-ints-wrong`).
            "module java.base of loader 'bootstrap'"
        } else {
            "unnamed module of loader 'app'"
        }
    }
    let from_mod = module_of(from_jvm);
    let to_mod = module_of(to_jvm);
    if from_mod == to_mod {
        format!(
            "class {from_jvm} cannot be cast to class {to_jvm} ({from_jvm} and {to_jvm} are in {from_mod})"
        )
    } else {
        format!(
            "class {from_jvm} cannot be cast to class {to_jvm} ({from_jvm} is in {from_mod}; {to_jvm} is in {to_mod})"
        )
    }
}

/// The superclass of a [`ThrowClass::jvm_name`], or `None` for
/// `java.lang.Throwable` itself. Unknown names have no parent (they
/// never appear on the chain — `matches` only walks names produced
/// by [`ThrowClass::jvm_name`] and this function).
fn throw_parent(jvm_name: &str) -> Option<&'static str> {
    match jvm_name {
        "java.lang.Throwable" => None,
        "java.lang.Exception" | "java.lang.Error" => Some("java.lang.Throwable"),
        "java.lang.RuntimeException" => Some("java.lang.Exception"),
        "java.lang.NumberFormatException" => Some("java.lang.IllegalArgumentException"),
        "java.io.FileNotFoundException" => Some("java.io.IOException"),
        "java.io.IOException" => Some("java.lang.Exception"),
        "java.lang.ClassNotFoundException" => Some("java.lang.ReflectiveOperationException"),
        "java.lang.ReflectiveOperationException" => Some("java.lang.Exception"),
        "java.lang.AssertionError" => Some("java.lang.Error"),
        "java.lang.ArrayIndexOutOfBoundsException" => Some("java.lang.IndexOutOfBoundsException"),
        "java.lang.IllegalMonitorStateException" => Some("java.lang.RuntimeException"),
        "java.lang.IllegalArgumentException"
        | "java.lang.IllegalStateException"
        | "java.lang.UnsupportedOperationException"
        | "java.lang.NullPointerException"
        | "java.lang.ClassCastException"
        | "java.lang.IndexOutOfBoundsException"
        | "java.lang.NegativeArraySizeException"
        | "java.lang.ArithmeticException"
        | "clojure.lang.ArityException"
        | "clojure.lang.ExceptionInfo"
        | "clojure.lang.Compiler$CompilerException"
        | "clojure.lang.LispReader$ReaderException" => Some("java.lang.RuntimeException"),
        _ => throw_parent_concurrent(jvm_name),
    }
}

/// The `java.util.concurrent` branch of [`throw_parent`] (split
/// for the complexity budget; chains into [`throw_parent_rest`]).
fn throw_parent_concurrent(jvm_name: &str) -> Option<&'static str> {
    match jvm_name {
        "java.util.concurrent.ExecutionException" => Some("java.lang.Exception"),
        "java.util.concurrent.CancellationException" => Some("java.lang.IllegalStateException"),
        "java.util.concurrent.RejectedExecutionException" => Some("java.lang.RuntimeException"),
        _ => throw_parent_rest(jvm_name),
    }
}

/// The linkage-error branch plus the S16 `NoSuchElementException`
/// of [`throw_parent`] (split for the complexity budget; chains into
/// [`throw_parent_format`]). Also hosts the slice-2
/// `UnsupportedEncodingException` arm (moved out of
/// [`throw_parent`] proper to hold its baselined budget). The
/// reflective pair (`ClassNotFoundException`/`ReflectiveOperationException`)
/// lives in [`throw_parent`] proper — do not re-add it here (a
/// duplicate arm would be dead: `throw_parent` matches first).
fn throw_parent_rest(jvm_name: &str) -> Option<&'static str> {
    match jvm_name {
        "java.lang.LinkageError" => Some("java.lang.Error"),
        "java.lang.IllegalAccessError" => Some("java.lang.LinkageError"),
        "java.util.NoSuchElementException" => Some("java.lang.RuntimeException"),
        "java.io.UnsupportedEncodingException" => Some("java.io.IOException"),
        _ => throw_parent_format(jvm_name),
    }
}

/// The `java.util` formatting branch of [`throw_parent`] (split for
/// the complexity budget).
fn throw_parent_format(jvm_name: &str) -> Option<&'static str> {
    match jvm_name {
        "java.util.IllegalFormatConversionException"
        | "java.util.UnknownFormatConversionException"
        | "java.util.MissingFormatConversionException"
        | "java.util.MissingFormatArgumentException"
        | "java.util.IllegalFormatPrecisionException"
        | "java.util.IllegalFormatCodePointException"
        | "java.util.FormatFlagsConversionMismatchException"
        | "java.util.DuplicateFormatFlagsException"
        | "java.util.IllegalFormatFlagsException"
        | "java.util.IllegalFormatWidthException"
        | "java.util.MissingFormatWidthException"
        | "java.util.IllegalFormatArgumentIndexException" => {
            Some("java.util.IllegalFormatException")
        }
        "java.util.IllegalFormatException" => Some("java.lang.IllegalArgumentException"),
        "java.util.regex.PatternSyntaxException" => Some("java.lang.IllegalArgumentException"),
        _ => throw_parent_array(jvm_name),
    }
}

/// The array/error branch of [`throw_parent`] (S15, split for the
/// complexity budget): `AIOOBE` under `IOOBE`, store/size under
/// `RuntimeException`, `OOM` through `VirtualMachineError`.
fn throw_parent_array(jvm_name: &str) -> Option<&'static str> {
    match jvm_name {
        "java.lang.ArrayIndexOutOfBoundsException" => Some("java.lang.IndexOutOfBoundsException"),
        "java.lang.ArrayStoreException" | "java.lang.NegativeArraySizeException" => {
            Some("java.lang.RuntimeException")
        }
        "java.lang.OutOfMemoryError" => Some("java.lang.VirtualMachineError"),
        "java.lang.VirtualMachineError" => Some("java.lang.Error"),
        _ => None,
    }
}

/// A thrown value: `id` is its identity (equality and hashing use
/// the `id` only, like the JVM), `class` the exception class,
/// `message` its message, `data` the `ex-data` payload if any, and
/// `cause` the chained cause if any.
#[derive(Clone, Debug)]
pub struct Throwable {
    pub id: u64,
    pub class: ThrowClass,
    pub message: Option<Arc<str>>,
    pub data: Option<Box<Value>>,
    pub cause: Option<Box<Throwable>>,
}

/// A host-defined value embedded in [`Value::Extension`]. Object
/// safe: `Value`'s equality, hashing, comparison names, and printing
/// all dispatch through these methods.
pub trait ExtensionValue: std::fmt::Debug + Send + Sync {
    /// The JVM-style class name (used in compare errors).
    fn class_name(&self) -> String;

    /// The printed representation (used by the printer as-is).
    fn print(&self) -> String;

    /// Value equality against another extension value.
    fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool;

    /// Hash code, consistent with [`dyn_eq`](ExtensionValue::dyn_eq).
    fn dyn_hash(&self, state: &mut dyn Hasher);

    /// Downcasting support for [`dyn_eq`](ExtensionValue::dyn_eq).
    fn as_any(&self) -> &dyn Any;

    /// The `*print-dup*` constructor shape, when this extension has
    /// one (`None` renders the plain [`print`](ExtensionValue::print)
    /// fallback — the driver throws for undupable kinds before the
    /// engine ever asks). Only closures, native fns (`#=(munged. )`)
    /// and namespaces (`#=(find-ns name)`) override this.
    fn dup_shape(&self) -> Option<String> {
        None
    }

    /// The `str`/`print` display (used when not `readably`):
    /// `None` keeps [`print`](ExtensionValue::print) (the default
    /// — only `MathContext` overrides it, S2 oracle `wp-str`).
    fn print_str(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{ExtensionValue, ThrowClass, Throwable, Unbound, Var};
    use crate::{PrintOpts, Symbol, Value, compare, pr_to_string};
    use std::any::Any;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    fn pr(v: &Value) -> String {
        pr_to_string(v, &PrintOpts::pr())
    }

    fn hash_of(v: &Value) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn var_prints_reader_form() {
        let v = Value::Var(Var {
            id: 7,
            sym: Symbol::new(Some("user"), "x"),
        });
        assert_eq!(pr(&v), "#'user/x");
        let plain = Value::Var(Var {
            id: 1,
            sym: Symbol::new(None, "y"),
        });
        assert_eq!(pr(&plain), "#'y");
    }

    #[test]
    fn unbound_prints_object_form_with_hex_id() {
        let u = Value::Unbound(Unbound {
            sym: Symbol::new(Some("user"), "x"),
            var_id: 7,
        });
        assert_eq!(
            pr(&u),
            "#object[clojure.lang.Var$Unbound 0x7 \"Unbound: #'user/x\"]"
        );
        let big = Value::Unbound(Unbound {
            sym: Symbol::new(None, "y"),
            var_id: 255,
        });
        assert_eq!(
            pr(&big),
            "#object[clojure.lang.Var$Unbound 0xff \"Unbound: #'y\"]"
        );
    }

    #[test]
    fn throwable_prints_error_form() {
        let t = Value::Throwable(Throwable {
            id: 1,
            class: ThrowClass::Exception,
            message: Some(Arc::from("boom")),
            data: None,
            cause: None,
        });
        assert_eq!(
            pr(&t),
            "#error {:cause \"boom\" :data nil :via [{:type java.lang.Exception :message \"boom\"}] :trace []}"
        );
    }

    #[test]
    fn throwable_prints_via_chain_with_data() {
        let root = Throwable {
            id: 3,
            class: ThrowClass::IllegalArgumentException,
            message: Some(Arc::from("root \"cause\"")),
            data: None,
            cause: None,
        };
        let mid = Throwable {
            id: 2,
            class: ThrowClass::RuntimeException,
            message: Some(Arc::from("mid")),
            data: Some(Box::new(Value::Int(7))),
            cause: Some(Box::new(root)),
        };
        let top = Value::Throwable(Throwable {
            id: 1,
            class: ThrowClass::ExceptionInfo,
            message: Some(Arc::from("top")),
            data: Some(Box::new(Value::Vector(crate::Vector::from_vec(vec![
                Value::Int(1),
                Value::Str(Arc::from("s")),
            ])))),
            cause: Some(Box::new(mid)),
        });
        assert_eq!(
            pr(&top),
            concat!(
                "#error {:cause \"top\" :data [1 \"s\"] :via [",
                "{:type clojure.lang.ExceptionInfo :message \"top\" :data [1 \"s\"]} ",
                "{:type java.lang.RuntimeException :message \"mid\" :data 7} ",
                "{:type java.lang.IllegalArgumentException :message \"root \\\"cause\\\"\"}",
                "] :trace []}"
            )
        );
    }

    #[test]
    fn throw_class_jvm_names() {
        let cases = [
            (ThrowClass::Exception, "java.lang.Exception"),
            (ThrowClass::Error, "java.lang.Error"),
            (ThrowClass::RuntimeException, "java.lang.RuntimeException"),
            (
                ThrowClass::IllegalArgumentException,
                "java.lang.IllegalArgumentException",
            ),
            (
                ThrowClass::IllegalStateException,
                "java.lang.IllegalStateException",
            ),
            (
                ThrowClass::UnsupportedOperationException,
                "java.lang.UnsupportedOperationException",
            ),
            (ThrowClass::AssertionError, "java.lang.AssertionError"),
            (ThrowClass::ArityException, "clojure.lang.ArityException"),
            (
                ThrowClass::NullPointerException,
                "java.lang.NullPointerException",
            ),
            (
                ThrowClass::ClassCastException,
                "java.lang.ClassCastException",
            ),
            (
                ThrowClass::IndexOutOfBoundsException,
                "java.lang.IndexOutOfBoundsException",
            ),
            (
                ThrowClass::ArrayIndexOutOfBoundsException,
                "java.lang.ArrayIndexOutOfBoundsException",
            ),
            (
                ThrowClass::NegativeArraySizeException,
                "java.lang.NegativeArraySizeException",
            ),
            (
                ThrowClass::ArithmeticException,
                "java.lang.ArithmeticException",
            ),
            (ThrowClass::ExceptionInfo, "clojure.lang.ExceptionInfo"),
            (
                ThrowClass::FileNotFoundException,
                "java.io.FileNotFoundException",
            ),
            (
                ThrowClass::UnsupportedEncodingException,
                "java.io.UnsupportedEncodingException",
            ),
            (
                ThrowClass::IllegalAccessError,
                "java.lang.IllegalAccessError",
            ),
            (
                ThrowClass::NumberFormatException,
                "java.lang.NumberFormatException",
            ),
            (
                ThrowClass::CompilerException,
                "clojure.lang.Compiler$CompilerException",
            ),
            (
                ThrowClass::ExecutionException,
                "java.util.concurrent.ExecutionException",
            ),
            (
                ThrowClass::CancellationException,
                "java.util.concurrent.CancellationException",
            ),
            (
                ThrowClass::RejectedExecutionException,
                "java.util.concurrent.RejectedExecutionException",
            ),
            (
                ThrowClass::PatternSyntaxException,
                "java.util.regex.PatternSyntaxException",
            ),
        ];
        for (class, want) in cases {
            assert_eq!(class.jvm_name(), want);
        }
        assert_eq!(cases.len(), 24);
    }

    #[test]
    fn throw_class_matches_walks_parent_chain() {
        // (class, catch, expected)
        let cases = [
            (ThrowClass::Exception, "java.lang.Exception", true),
            (ThrowClass::Exception, "java.lang.Throwable", true),
            (ThrowClass::Exception, "java.lang.Error", false),
            (ThrowClass::Exception, "java.lang.RuntimeException", false),
            (ThrowClass::Error, "java.lang.Error", true),
            (ThrowClass::Error, "java.lang.Throwable", true),
            (ThrowClass::Error, "java.lang.Exception", false),
            (
                ThrowClass::RuntimeException,
                "java.lang.RuntimeException",
                true,
            ),
            (ThrowClass::RuntimeException, "java.lang.Exception", true),
            (ThrowClass::RuntimeException, "java.lang.Throwable", true),
            (ThrowClass::RuntimeException, "java.lang.Error", false),
            (
                ThrowClass::IllegalArgumentException,
                "java.lang.IllegalArgumentException",
                true,
            ),
            (
                ThrowClass::IllegalArgumentException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::IllegalArgumentException,
                "java.lang.Exception",
                true,
            ),
            (
                ThrowClass::IllegalArgumentException,
                "java.lang.Throwable",
                true,
            ),
            (
                ThrowClass::IllegalArgumentException,
                "java.lang.Error",
                false,
            ),
            (
                ThrowClass::IllegalArgumentException,
                "java.lang.IllegalStateException",
                false,
            ),
            (ThrowClass::AssertionError, "java.lang.AssertionError", true),
            (ThrowClass::AssertionError, "java.lang.Error", true),
            (ThrowClass::AssertionError, "java.lang.Throwable", true),
            (ThrowClass::AssertionError, "java.lang.Exception", false),
            (
                ThrowClass::AssertionError,
                "java.lang.RuntimeException",
                false,
            ),
            (
                ThrowClass::ExceptionInfo,
                "clojure.lang.ExceptionInfo",
                true,
            ),
            (
                ThrowClass::ExceptionInfo,
                "java.lang.RuntimeException",
                true,
            ),
            (ThrowClass::ExceptionInfo, "java.lang.Exception", true),
            (ThrowClass::ExceptionInfo, "java.lang.Throwable", true),
            (ThrowClass::ExceptionInfo, "java.lang.Error", false),
            (
                ThrowClass::ArityException,
                "clojure.lang.ArityException",
                true,
            ),
            (
                ThrowClass::ArityException,
                "java.lang.RuntimeException",
                true,
            ),
            (ThrowClass::ArityException, "java.lang.Exception", true),
            (ThrowClass::ArityException, "java.lang.Throwable", true),
            (ThrowClass::ArityException, "java.lang.Error", false),
            (
                ThrowClass::NullPointerException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::ClassCastException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::IndexOutOfBoundsException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::ArithmeticException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::IllegalStateException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::UnsupportedOperationException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::CompilerException,
                "clojure.lang.Compiler$CompilerException",
                true,
            ),
            (
                ThrowClass::CompilerException,
                "java.lang.RuntimeException",
                true,
            ),
            (ThrowClass::CompilerException, "java.lang.Error", false),
            (
                ThrowClass::ExecutionException,
                "java.util.concurrent.ExecutionException",
                true,
            ),
            (ThrowClass::ExecutionException, "java.lang.Exception", true),
            (ThrowClass::ExecutionException, "java.lang.Throwable", true),
            (
                ThrowClass::ExecutionException,
                "java.lang.RuntimeException",
                false,
            ),
            (ThrowClass::ExecutionException, "java.lang.Error", false),
            (
                ThrowClass::CancellationException,
                "java.util.concurrent.CancellationException",
                true,
            ),
            (
                ThrowClass::CancellationException,
                "java.lang.IllegalStateException",
                true,
            ),
            (
                ThrowClass::CancellationException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::CancellationException,
                "java.lang.Exception",
                true,
            ),
            (ThrowClass::CancellationException, "java.lang.Error", false),
            (
                ThrowClass::RejectedExecutionException,
                "java.util.concurrent.RejectedExecutionException",
                true,
            ),
            (
                ThrowClass::RejectedExecutionException,
                "java.lang.RuntimeException",
                true,
            ),
            (
                ThrowClass::RejectedExecutionException,
                "java.lang.Exception",
                true,
            ),
            (
                ThrowClass::RejectedExecutionException,
                "java.lang.Error",
                false,
            ),
            (
                ThrowClass::ArrayStoreException,
                "java.lang.ArrayStoreException",
                true,
            ),
            (
                ThrowClass::ArrayStoreException,
                "java.lang.RuntimeException",
                true,
            ),
            (ThrowClass::ArrayStoreException, "java.lang.Exception", true),
            (ThrowClass::ArrayStoreException, "java.lang.Throwable", true),
            (ThrowClass::ArrayStoreException, "java.lang.Error", false),
            // Unlike `ArrayIndexOutOfBoundsException`, the store
            // check is not an index check: it sits directly under
            // `RuntimeException`.
            (
                ThrowClass::ArrayStoreException,
                "java.lang.IndexOutOfBoundsException",
                false,
            ),
        ];
        for (class, catch, want) in cases {
            // Hoisted: assert args only run on failure (100% line gate).
            let name = format!("{:?}", class.jvm_name());
            assert_eq!(class.matches(catch), want, "{name} matches {catch}");
        }
    }

    #[test]
    fn everything_matches_throwable() {
        let all = [
            ThrowClass::Exception,
            ThrowClass::Error,
            ThrowClass::RuntimeException,
            ThrowClass::IllegalArgumentException,
            ThrowClass::IllegalStateException,
            ThrowClass::UnsupportedOperationException,
            ThrowClass::AssertionError,
            ThrowClass::ArityException,
            ThrowClass::NullPointerException,
            ThrowClass::ClassCastException,
            ThrowClass::IndexOutOfBoundsException,
            ThrowClass::ArrayIndexOutOfBoundsException,
            ThrowClass::NegativeArraySizeException,
            ThrowClass::ArithmeticException,
            ThrowClass::ExceptionInfo,
            ThrowClass::FileNotFoundException,
            ThrowClass::UnsupportedEncodingException,
            ThrowClass::IllegalAccessError,
            ThrowClass::NumberFormatException,
            ThrowClass::CompilerException,
            ThrowClass::ExecutionException,
            ThrowClass::CancellationException,
            ThrowClass::RejectedExecutionException,
        ];
        for class in all {
            assert!(class.matches("java.lang.Throwable"));
            assert!(class.matches(class.jvm_name()));
            assert!(!class.matches("java.lang.NoSuchClass"));
        }
        assert!(ThrowClass::FileNotFoundException.matches("java.io.IOException"));
        assert!(ThrowClass::FileNotFoundException.matches("java.lang.Exception"));
        assert!(!ThrowClass::FileNotFoundException.matches("java.lang.Error"));
        assert!(ThrowClass::UnsupportedEncodingException.matches("java.io.IOException"));
        assert!(ThrowClass::UnsupportedEncodingException.matches("java.lang.Exception"));
        assert!(!ThrowClass::UnsupportedEncodingException.matches("java.lang.Error"));
        assert!(!ThrowClass::UnsupportedEncodingException.matches("java.lang.RuntimeException"));
        assert!(ThrowClass::IllegalAccessError.matches("java.lang.LinkageError"));
        assert!(ThrowClass::IllegalAccessError.matches("java.lang.Error"));
        assert!(!ThrowClass::IllegalAccessError.matches("java.lang.Exception"));
        assert!(ThrowClass::NumberFormatException.matches("java.lang.IllegalArgumentException"));
        assert!(ThrowClass::NumberFormatException.matches("java.lang.RuntimeException"));
        assert!(ThrowClass::NumberFormatException.matches("java.lang.Exception"));
        assert!(!ThrowClass::NumberFormatException.matches("java.lang.Error"));
        assert!(
            ThrowClass::ArrayIndexOutOfBoundsException
                .matches("java.lang.IndexOutOfBoundsException")
        );
        assert!(ThrowClass::ArrayIndexOutOfBoundsException.matches("java.lang.RuntimeException"));
        assert!(ThrowClass::NegativeArraySizeException.matches("java.lang.RuntimeException"));
        assert!(!ThrowClass::NegativeArraySizeException.matches("java.lang.Error"));
    }

    fn throwable(id: u64, message: &str) -> Value {
        Value::Throwable(Throwable {
            id,
            class: ThrowClass::Exception,
            message: Some(Arc::from(message)),
            data: Some(Box::new(Value::Int(id as i64))),
            cause: None,
        })
    }

    #[test]
    fn new_variants_compare_by_identity() {
        let var_a = Value::Var(Var {
            id: 1,
            sym: Symbol::new(Some("u"), "a"),
        });
        let var_same_id = Value::Var(Var {
            id: 1,
            sym: Symbol::new(Some("other"), "b"),
        });
        let var_other_id = Value::Var(Var {
            id: 2,
            sym: Symbol::new(Some("u"), "a"),
        });
        assert_eq!(var_a, var_same_id);
        assert_eq!(hash_of(&var_a), hash_of(&var_same_id));
        assert_ne!(var_a, var_other_id);

        let unb_a = Value::Unbound(Unbound {
            sym: Symbol::new(Some("u"), "a"),
            var_id: 1,
        });
        let unb_same_id = Value::Unbound(Unbound {
            sym: Symbol::new(Some("other"), "b"),
            var_id: 1,
        });
        let unb_other_id = Value::Unbound(Unbound {
            sym: Symbol::new(Some("u"), "a"),
            var_id: 2,
        });
        assert_eq!(unb_a, unb_same_id);
        assert_eq!(hash_of(&unb_a), hash_of(&unb_same_id));
        assert_ne!(unb_a, unb_other_id);

        // Throwables never compare by message or data.
        let t1 = throwable(1, "boom");
        let t1_other_payload = Value::Throwable(Throwable {
            id: 1,
            class: ThrowClass::Error,
            message: Some(Arc::from("different")),
            data: None,
            cause: Some(Box::new(Throwable {
                id: 9,
                class: ThrowClass::Exception,
                message: Some(Arc::from("cause")),
                data: None,
                cause: None,
            })),
        });
        assert_eq!(t1, t1_other_payload);
        assert_eq!(hash_of(&t1), hash_of(&t1_other_payload));
        assert_ne!(t1, throwable(2, "boom"));

        // Cross-variant pairs are unequal, and unequal to old variants.
        assert_ne!(var_a, unb_a);
        assert_ne!(var_a, t1);
        assert_ne!(unb_a, t1);
        assert_ne!(var_a, Value::Int(1));
        assert_ne!(unb_a, Value::Nil);
        assert_ne!(t1, Value::Str(Arc::from("boom")));
    }

    #[test]
    fn new_variants_are_incomparable_but_nil_still_orders() {
        use std::cmp::Ordering;
        let var = Value::Var(Var {
            id: 1,
            sym: Symbol::new(None, "x"),
        });
        let unb = Value::Unbound(Unbound {
            sym: Symbol::new(None, "x"),
            var_id: 1,
        });
        let t = throwable(1, "boom");
        // Non-`Comparable` `k1` fails itself against `Comparable`;
        // a `String` `k1` fails the alien `k2` to `String` (#102).
        for (a, b, want) in [
            (
                &var,
                &var,
                "class clojure.lang.Var cannot be cast to class java.lang.Comparable (clojure.lang.Var is in unnamed module of loader 'app'; java.lang.Comparable is in module java.base of loader 'bootstrap')",
            ),
            (
                &var,
                &Value::Int(1),
                "class clojure.lang.Var cannot be cast to class java.lang.Comparable (clojure.lang.Var is in unnamed module of loader 'app'; java.lang.Comparable is in module java.base of loader 'bootstrap')",
            ),
            (
                &Value::Str(Arc::from("s")),
                &unb,
                "class clojure.lang.Var$Unbound cannot be cast to class java.lang.String (clojure.lang.Var$Unbound is in unnamed module of loader 'app'; java.lang.String is in module java.base of loader 'bootstrap')",
            ),
            (
                &t,
                &t,
                "class java.lang.Exception cannot be cast to class java.lang.Comparable (java.lang.Exception and java.lang.Comparable are in module java.base of loader 'bootstrap')",
            ),
            (
                &t,
                &var,
                "class java.lang.Exception cannot be cast to class java.lang.Comparable (java.lang.Exception and java.lang.Comparable are in module java.base of loader 'bootstrap')",
            ),
        ] {
            assert_eq!(compare(a, b).unwrap_err().message(), want);
        }
        // The (Nil, _)/(_, Nil) arms still win over the new variants.
        assert_eq!(compare(&Value::Nil, &var), Ok(Ordering::Less));
        assert_eq!(compare(&var, &Value::Nil), Ok(Ordering::Greater));
        assert_eq!(compare(&Value::Nil, &t), Ok(Ordering::Less));
        assert_eq!(compare(&t, &Value::Nil), Ok(Ordering::Greater));
    }

    /// A stub host type proving `Value` dispatches print, `=`, hash,
    /// and compare names through the trait.
    #[derive(Debug)]
    struct StubExt {
        n: i32,
    }

    impl ExtensionValue for StubExt {
        fn class_name(&self) -> String {
            "test.Stub".to_owned()
        }

        fn print(&self) -> String {
            format!("#test.Stub[{}]", self.n)
        }

        fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
            match other.as_any().downcast_ref::<Self>() {
                Some(o) => o.n == self.n,
                None => false,
            }
        }

        fn dyn_hash(&self, state: &mut dyn Hasher) {
            state.write_i32(self.n);
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Debug)]
    struct OtherExt;

    impl ExtensionValue for OtherExt {
        fn class_name(&self) -> String {
            "test.Other".to_owned()
        }

        fn print(&self) -> String {
            "#test.Other".to_owned()
        }

        fn dyn_eq(&self, other: &dyn ExtensionValue) -> bool {
            other.as_any().downcast_ref::<Self>().is_some()
        }

        fn dyn_hash(&self, state: &mut dyn Hasher) {
            state.write_u8(0x4f);
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn stub(n: i32) -> Value {
        Value::Extension(Arc::new(StubExt { n }))
    }

    #[test]
    fn extension_dispatches_through_value() {
        assert_eq!(pr(&stub(3)), "#test.Stub[3]");
        assert_eq!(stub(3), stub(3));
        assert_eq!(hash_of(&stub(3)), hash_of(&stub(3)));
        assert_ne!(stub(3), stub(4));
        // Different host types are unequal through the same variant.
        assert_ne!(stub(3), Value::Extension(Arc::new(OtherExt)));
        // Cross-variant pairs are unequal.
        assert_ne!(stub(3), Value::Int(3));
        assert_ne!(stub(3), Value::Str(Arc::from("#test.Stub[3]")));
        // Compare errors fail the host class (#102).
        let err = compare(&stub(3), &Value::Int(1)).unwrap_err();
        assert_eq!(
            err.message(),
            "class test.Stub cannot be cast to class java.lang.Comparable (test.Stub is in unnamed module of loader 'app'; java.lang.Comparable is in module java.base of loader 'bootstrap')"
        );
        let err = compare(&Value::Int(1), &stub(3)).unwrap_err();
        assert_eq!(
            err.message(),
            "class test.Stub cannot be cast to class java.lang.Number (test.Stub is in unnamed module of loader 'app'; java.lang.Number is in module java.base of loader 'bootstrap')"
        );
    }

    #[test]
    fn throw_class_jvm_names_format_tail() {
        let cases = [
            (
                ThrowClass::MissingFormatConversionException,
                "java.util.MissingFormatConversionException",
            ),
            (
                ThrowClass::IllegalFormatCodePointException,
                "java.util.IllegalFormatCodePointException",
            ),
            (ThrowClass::IOException, "java.io.IOException"),
        ];
        for (class, want) in cases {
            assert_eq!(class.jvm_name(), want);
        }
    }

    #[test]
    fn format_classes_match_through_format_parent() {
        let class = ThrowClass::IllegalFormatConversionException;
        assert!(class.matches("java.util.IllegalFormatConversionException"));
        assert!(class.matches("java.util.IllegalFormatException"));
        assert!(class.matches("java.lang.IllegalArgumentException"));
        assert!(class.matches("java.lang.RuntimeException"));
        assert!(class.matches("java.lang.Throwable"));
        assert!(!class.matches("java.lang.Error"));
        assert!(!class.matches("java.lang.NoSuchClass"));
        assert!(
            ThrowClass::MissingFormatWidthException.matches("java.util.IllegalFormatException")
        );
        assert!(ThrowClass::DuplicateFormatFlagsException.matches("java.lang.Throwable"));
    }

    #[test]
    fn unknown_throw_parents_are_none() {
        assert_eq!(super::throw_parent("java.lang.NoSuchClass"), None);
        assert_eq!(super::throw_parent_format("java.lang.NoSuchClass"), None);
    }

    #[test]
    fn other_extension_dispatches() {
        let other = Value::Extension(Arc::new(OtherExt));
        assert_eq!(pr(&other), "#test.Other");
        assert_eq!(other, other.clone());
        assert_eq!(hash_of(&other), hash_of(&other.clone()));
        let err = compare(&other, &Value::Int(1)).unwrap_err();
        assert_eq!(
            err.message(),
            "class test.Other cannot be cast to class java.lang.Comparable (test.Other is in unnamed module of loader 'app'; java.lang.Comparable is in module java.base of loader 'bootstrap')"
        );
    }

    #[test]
    fn class_not_found_names_and_catches() {
        // S7 `#=` symbol/class heads (probes `d-class-miss`,
        // `c-qualmiss`): the JVM binary name and the
        // `ReflectiveOperationException` catch chain.
        let class = ThrowClass::ClassNotFoundException;
        assert_eq!(class.jvm_name(), "java.lang.ClassNotFoundException");
        assert!(class.matches("java.lang.ClassNotFoundException"));
        assert!(class.matches("java.lang.ReflectiveOperationException"));
        assert!(class.matches("java.lang.Exception"));
        assert!(class.matches("java.lang.Throwable"));
        assert!(!class.matches("java.lang.RuntimeException"));
    }

    #[test]
    fn simple_name_strips_package_and_outer() {
        // #125: `jvm_class()` reports simple names (the oracle
        // records `getSimpleName`), so nested classes drop the
        // outer (`$`) as well as the package.
        assert_eq!(
            ThrowClass::RuntimeException.simple_name(),
            "RuntimeException"
        );
        assert_eq!(
            ThrowClass::PatternSyntaxException.simple_name(),
            "PatternSyntaxException"
        );
        assert_eq!(ThrowClass::ReaderException.simple_name(), "ReaderException");
        assert_eq!(
            ThrowClass::CompilerException.simple_name(),
            "CompilerException"
        );
    }

    #[test]
    #[should_panic(expected = "throw-class partition")]
    fn jvm_name_rest_rejects_upstream_variants() {
        // Defensive arm: the single caller partitions the variants,
        // so `jvm_name`-handled classes never arrive here (only a
        // dispatch bug could).
        let _ = ThrowClass::ExceptionInfo.jvm_name_rest();
    }
}
