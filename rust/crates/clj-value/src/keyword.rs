// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! Clojure keywords: like symbols but self-evaluating, with a leading
//! colon in their printed form. Globally interned (the JVM's
//! `Keyword.intern` table analog): every construction canonicalizes
//! its arcs, so `identical?` is pointer equality.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};

use crate::FxBuildHasher;

/// One interned keyword's canonical fields, shared behind one
/// `Arc` so a `Keyword` (inline in every `Value`) stays one word
/// (#254: `Value` shrinks to 32 bytes).
struct KeywordData {
    ns: Option<Arc<str>>,
    name: Arc<str>,
}

/// The intern tables (fence 4a): one shared [`KeywordData`] per
/// distinct keyword across every construction, like the JVM's
/// `KeywordTable` (strong refs — entries live as long as the
/// process, same as interned keywords on the JVM). Fx-hashed:
/// every keyword construction probes them. Plain keywords (most:
/// `:line` on every read list, `:macro` on every analyzed head)
/// probe by `&str`, allocating only on first interning; qualified
/// ones probe by their `(ns, name)` arc pair.
#[derive(Default)]
struct KeywordTable {
    plain: HashMap<Arc<str>, Arc<KeywordData>, FxBuildHasher>,
    qualified: HashMap<(Arc<str>, Arc<str>), Arc<KeywordData>, FxBuildHasher>,
}

static KEYWORD_TABLE: LazyLock<Mutex<KeywordTable>> =
    LazyLock::new(|| Mutex::new(KeywordTable::default()));

/// A plain keyword interned once per call site
/// (`static_keyword!("line")`): later uses clone the cached arcs
/// instead of probing the intern table under its lock. The cached
/// value came from [`Keyword::new`], so identity is canonical.
#[macro_export]
macro_rules! static_keyword {
    ($name:literal) => {{
        static KW: ::std::sync::LazyLock<$crate::Keyword> =
            ::std::sync::LazyLock::new(|| $crate::Keyword::new(None, $name));
        KW.clone()
    }};
}

/// A Clojure keyword such as `:ok` or `:user/id`.
#[derive(Clone)]
pub struct Keyword(Arc<KeywordData>);

impl fmt::Debug for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Keyword")
            .field("ns", &self.0.ns)
            .field("name", &self.0.name)
            .finish()
    }
}

impl Keyword {
    /// Build a keyword from namespace and name parts, canonicalizing
    /// through the intern table (every construction of an equal
    /// keyword shares one [`KeywordData`]).
    #[must_use]
    pub fn new(ns: Option<&str>, name: &str) -> Self {
        let mut table = KEYWORD_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(ns) = ns else {
            if let Some(data) = table.plain.get(name) {
                return Self(Arc::clone(data));
            }
            let data = Arc::new(KeywordData {
                ns: None,
                name: Arc::from(name),
            });
            table
                .plain
                .insert(Arc::clone(&data.name), Arc::clone(&data));
            return Self(data);
        };
        let data = table
            .qualified
            .entry((Arc::from(ns), Arc::from(name)))
            .or_insert_with_key(|k| {
                Arc::new(KeywordData {
                    ns: Some(Arc::clone(&k.0)),
                    name: Arc::clone(&k.1),
                })
            });
        Self(Arc::clone(data))
    }

    /// The namespace part, if any.
    #[must_use]
    pub fn namespace(&self) -> Option<&str> {
        self.0.ns.as_deref()
    }

    /// The name part.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Pointer identity for `identical?` (fence 4a): complete
    /// exactly when every construction interns (all of them route
    /// through [`new`](Keyword::new), so equal keywords share one
    /// [`KeywordData`]).
    #[inline]
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

// Manual impls over `(ns, name)`, mirroring `Symbol` (keywords never
// carry metadata, so these equal the derived ones; they are explicit
// so a future field cannot silently join the identity).
impl PartialEq for Keyword {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Every keyword is interned ([`new`](Keyword::new) is the only
        // constructor, and the table keeps each entry for the life of
        // the process), so pointer equality is exact: map-key scans and
        // `=` over keywords never compare strings (#235, #271).
        debug_assert_eq!(
            self.ptr_eq(other),
            self.0.ns == other.0.ns && self.0.name == other.0.name
        );
        self.ptr_eq(other)
    }
}

impl Eq for Keyword {}

impl Hash for Keyword {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.ns.hash(state);
        self.0.name.hash(state);
    }
}

impl PartialOrd for Keyword {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Keyword {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.0.ns.as_deref(), self.0.name.as_ref())
            .cmp(&(other.0.ns.as_deref(), other.0.name.as_ref()))
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(":")?;
        if let Some(ns) = &self.0.ns {
            write!(f, "{ns}/")?;
        }
        write!(f, "{}", self.0.name)
    }
}

#[cfg(test)]
mod tests {
    use super::Keyword;

    #[test]
    fn parts_and_display() {
        let plain = Keyword::new(None, "ok");
        assert_eq!(plain.namespace(), None);
        assert_eq!(plain.name(), "ok");
        assert_eq!(plain.to_string(), ":ok");

        let qualified = Keyword::new(Some("user"), "id");
        assert_eq!(qualified.to_string(), ":user/id");

        assert!(Keyword::new(None, "a") < Keyword::new(None, "b"));
        assert!(Keyword::new(None, "b") > Keyword::new(None, "a"));
    }

    #[test]
    fn equality_is_by_value() {
        assert_eq!(Keyword::new(None, "a"), Keyword::new(None, "a"));
        assert_ne!(Keyword::new(None, "a"), Keyword::new(Some("n"), "a"));
    }

    // Fence 4a: every construction interns — separately built
    // equal keywords share arcs (pointer identity).
    #[test]
    fn interning_shares_arcs() {
        let a = Keyword::new(None, "f4a-kw");
        let b = Keyword::new(None, "f4a-kw");
        assert!(a.ptr_eq(&b));
        assert!(a.ptr_eq(&a.clone()));
        let c = Keyword::new(Some("f4a-ns"), "f4a-kw");
        let d = Keyword::new(Some("f4a-ns"), "f4a-kw");
        assert!(c.ptr_eq(&d));
        assert!(!a.ptr_eq(&c));
        assert!(!c.ptr_eq(&Keyword::new(Some("f4a-ns"), "other")));
    }
}
