// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! The S4 namespace (one: `user`), var states, and the thread-binding
//! stack. Mirrors `Var.java`: a fresh var's root is its per-var
//! `Unbound` sentinel, `deref` never throws, and `hasRoot` is false
//! for ANY `Unbound` root (so `(def x x)` stays unbound, probe25).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

use clj_value::{
    BindCounts, ClassVal, FxBuildHasher, Keyword, Map, SymKey, Symbol, Unbound, Value, Var, new_map,
};

use crate::core_lazy::{LazyGroup, LazyIndex};
use crate::ext::NsHandle;

/// Lazy index segments: boot's `core.clj` plus the lazily loaded
/// libraries (a full table loads further libraries eagerly).
const LAZY_SEGMENTS: usize = 8;

/// Registry maps hash with Fx (not SipHash): these sit on the var-deref
/// hot path, and the keys are not attacker-controlled.
pub(crate) type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// A namespace's refers: simple name → target var symbol and the
/// var id pinned at install. Shared copy-on-write (#226): a fresh
/// namespace referring every public of another installs that
/// namespace's cached [`PublicsTable`] itself, and the reader reads
/// the table in place (`user`, `clojure.string` and each script's
/// `ns` refer all ~700 `clojure.core` publics on every run).
pub(crate) type ReferTable = FxHashMap<Arc<str>, (Symbol, u64)>;

/// Every public intern of a namespace as a [`ReferTable`] (`refer`
/// without `:only`/`:refer`/`:rename`), cached in the namespace until
/// a mapping or var state there changes.
#[derive(Clone, Debug)]
pub(crate) struct PublicsTable {
    pub(crate) table: Arc<ReferTable>,
    /// Some public shares its name with a default import (a refer
    /// drops the same-named import, so the table cannot be shared
    /// by a namespace holding the defaults).
    shadows_default_import: bool,
}

/// A var's evaluator-side state: the handle in [`Value::Var`] is
/// just `(id, sym)`; everything mutable lives here.
#[derive(Clone, Debug)]
pub struct VarState {
    /// The root (`None` = freshly interned, never set).
    pub root: Option<Value>,
    /// The var's metadata (`None` = cleared via `reset-meta!`/
    /// `alter-meta!`; fresh interns carry `:ns`/`:name`, `def`
    /// stamps the auto keys plus user meta).
    pub meta: Option<Map>,
    /// `^:dynamic`: required for `binding`/`set!`.
    pub dynamic: bool,
}

impl VarState {
    fn fresh() -> Self {
        Self {
            root: None,
            meta: None,
            dynamic: false,
        }
    }
}

/// One namespace: vars by id, qualified symbols to ids, aliases,
/// refers (simple name → qualified target), and imports (S16:
/// simple name → class). Var ids come from the registry counter,
/// never per-namespace, so ids stay unique across namespaces (and
/// double as the `#object` hex for `Unbound` sentinels). The
/// namespace `id` (also registry-issued, never reused while live)
/// is the `#object` hex for namespace values.
#[derive(Clone, Debug)]
pub struct Namespace {
    name: String,
    id: u64,
    vars: FxHashMap<u64, VarState>,
    ids: FxHashMap<Symbol, u64>,
    aliases: FxHashMap<String, (String, u64)>,
    refers: Arc<ReferTable>,
    /// Shared copy-on-write, initialized lazily: every namespace starts
    /// from the same default imports (see `classemu::default_import_map`),
    /// deferred until imports are inspected or mutated so minimal AOT
    /// startup skips instantiating default class descriptors.
    imports: OnceLock<Arc<FxHashMap<String, Arc<ClassVal>>>>,
    /// This namespace's publics as a refer table, built on demand and
    /// dropped by every intern, unmap, and var state write (privacy
    /// is var meta).
    publics_table: Option<PublicsTable>,
    meta: Option<Map>,
    /// This namespace as a value (every fresh var's `:ns` meta
    /// shares it, instead of minting one per var).
    handle: Arc<NsHandle>,
    /// The reader's view of the mappings, built on first read and
    /// dropped by every alias/refer/import change: `user` refers
    /// every `clojure.core` public, and each load used to rebuild it.
    reader_maps: Option<ReaderMaps>,
}

/// A namespace's mappings as the reader takes them (`::alias/kw`,
/// syntax-quote refers and class names), shared between reads.
#[derive(Clone, Debug)]
pub struct ReaderMaps {
    pub aliases: Arc<HashMap<String, String>>,
    pub refers: Arc<dyn clj_reader::opts::Refers>,
    pub class_map: Arc<HashMap<String, String>>,
}

/// The reader's class map over the default imports (shared by every
/// namespace still holding them).
fn default_class_map() -> Arc<HashMap<String, String>> {
    static MAP: std::sync::LazyLock<Arc<HashMap<String, String>>> =
        std::sync::LazyLock::new(|| Arc::new(class_map_of(&crate::classemu::default_import_map())));
    Arc::clone(&MAP)
}

fn class_map_of(imports: &FxHashMap<String, Arc<ClassVal>>) -> HashMap<String, String> {
    imports
        .iter()
        .map(|(simple, class)| (simple.clone(), class.name().to_owned()))
        .collect()
}

impl Namespace {
    /// A fresh namespace with a registry-issued id (`nil` meta,
    /// like the JVM) and the JVM default imports seeded (#74 —
    /// `RT.DEFAULT_IMPORTS`, flat like upstream: explicit imports
    /// conflict with same-named defaults exactly like explicit
    /// entries).
    #[must_use]
    pub fn new(name: &str, id: u64) -> Self {
        Self {
            name: name.to_owned(),
            id,
            vars: FxHashMap::default(),
            ids: FxHashMap::default(),
            aliases: FxHashMap::default(),
            refers: Arc::default(),
            imports: OnceLock::new(),
            publics_table: None,
            meta: None,
            handle: Arc::new(NsHandle {
                name: Arc::from(name),
                id,
            }),
            reader_maps: None,
        }
    }

    fn imports_map(&self) -> &Arc<FxHashMap<String, Arc<ClassVal>>> {
        self.imports
            .get_or_init(crate::classemu::default_import_map)
    }

    fn imports_mut(&mut self) -> &mut Arc<FxHashMap<String, Arc<ClassVal>>> {
        let _ = self.imports_map();
        self.imports.get_mut().expect("imports initialized")
    }

    /// [`ReaderMaps`] for this namespace (cached until a mapping
    /// changes).
    pub fn reader_maps(&mut self) -> ReaderMaps {
        if let Some(m) = &self.reader_maps {
            return m.clone();
        }
        let defaults = crate::classemu::default_import_map();
        let imports = self.imports_map();
        let maps = ReaderMaps {
            aliases: Arc::new(self.alias_map()),
            refers: self.refers.clone(),
            class_map: if Arc::ptr_eq(imports, &defaults) {
                default_class_map()
            } else {
                Arc::new(class_map_of(imports))
            },
        };
        self.reader_maps = Some(maps.clone());
        maps
    }

    /// This namespace's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// This namespace's registry id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Look up this namespace's own `name` without building a `Symbol`
    /// (the borrowed [`SymKey`] probe [`lookup`](Self::lookup) uses).
    #[must_use]
    pub fn lookup_str(&self, name: &str) -> Option<u64> {
        self.ids
            .get(&(Some(self.name.as_str()), name) as &dyn SymKey)
            .copied()
    }

    /// Look up a var id by symbol (unqualified resolves here).
    #[must_use]
    pub fn lookup(&self, sym: &Symbol) -> Option<u64> {
        // Borrowed probe: no qualified symbol is built (the analyzer
        // resolves through here per symbol).
        match sym.namespace() {
            None => self
                .ids
                .get(&(Some(self.name.as_str()), sym.name()) as &dyn SymKey)
                .copied(),
            Some(ns) if ns == self.name => self.ids.get(sym).copied(),
            Some(_) => None,
        }
    }

    /// Intern (find or create) a var in this namespace (the symbol
    /// must already name it). Creation stamps the `:ns`/`:name`
    /// meta (JVM `Var.intern`; `stamp` false — a program that cannot
    /// read var metadata — leaves none) and drops any refer or import
    /// of the same name (`def`/`intern` silently replace refers and
    /// imports, like `Namespace.reference` minus its `*err*` warning,
    /// which waits for S5D). Ids come from the registry counter.
    pub fn intern(&mut self, sym: &Symbol, next_id: &mut u64, stamp: bool) -> u64 {
        if let Some(id) = self.ids.get(sym) {
            return *id;
        }
        let id = *next_id;
        *next_id += 1;
        let mut state = VarState::fresh();
        if stamp {
            let ns_val: Arc<dyn clj_value::ExtensionValue> = self.handle.clone();
            static KEYS: std::sync::LazyLock<[Value; 2]> = std::sync::LazyLock::new(|| {
                [
                    Value::Keyword(clj_value::static_keyword!("ns")),
                    Value::Keyword(clj_value::static_keyword!("name")),
                ]
            });
            state.meta = Some(Map::from_distinct_keywords(
                &*KEYS,
                vec![
                    Value::Extension(ns_val),
                    Value::Symbol(Symbol::from_shared(None, Arc::clone(sym.name_arc()))),
                ],
            ));
        }
        self.vars.insert(id, state);
        self.ids.insert(sym.clone(), id);
        self.publics_table = None;
        let unrefer = self.drop_refer(sym.name());
        if self.drop_import(sym.name()) || unrefer {
            self.reader_maps = None;
        }
        id
    }

    /// Drop the refer of a simple name (`true` when there was one),
    /// copying a shared table only then.
    fn drop_refer(&mut self, name: &str) -> bool {
        self.refers.contains_key(name) && Arc::make_mut(&mut self.refers).remove(name).is_some()
    }

    /// Alias → target-namespace-name (for reader options).
    #[must_use]
    pub fn alias_map(&self) -> HashMap<String, String> {
        self.aliases
            .iter()
            .map(|(alias, (target, _))| (alias.clone(), target.clone()))
            .collect()
    }

    /// This namespace's metadata (`None` = `nil`, like the JVM).
    #[must_use]
    pub fn meta(&self) -> Option<&Map> {
        self.meta.as_ref()
    }

    /// Replace this namespace's metadata (`alter-meta!`/
    /// `reset-meta!`).
    pub fn set_meta(&mut self, meta: Option<Map>) {
        self.meta = meta;
    }

    /// Install an alias: fresh and identical-target installs are
    /// `Ok`; a conflicting target is `Err` with the old target
    /// (the `alias` error names it). The target id is pinned at
    /// install so `ns-aliases` shows the aliased object even after
    /// the target is removed and recreated.
    pub fn add_alias(&mut self, alias: &str, target: &str, target_id: u64) -> Result<(), String> {
        match self.aliases.get(alias) {
            None => {
                self.aliases
                    .insert(alias.to_owned(), (target.to_owned(), target_id));
                self.reader_maps = None;
                Ok(())
            }
            Some((old, _)) if old == target => Ok(()),
            Some((old, _)) => Err(old.clone()),
        }
    }

    /// Drop an alias (missing is a silent no-op).
    pub fn remove_alias(&mut self, alias: &str) {
        self.aliases.remove(alias);
        self.reader_maps = None;
    }

    /// Install a refer (replaces silently; callers keep interns —
    /// an intern of the same name must never reach this). The var
    /// id is pinned at install so `ns-refers`/`ns-resolve` see the
    /// referred var even after the target is re-interned. Any import
    /// of the same name goes (imports live under refers, like the
    /// JVM's single mapping).
    pub fn add_refer(&mut self, name: &str, target: Symbol, target_id: u64) {
        Arc::make_mut(&mut self.refers).insert(Arc::from(name), (target, target_id));
        self.drop_import(name);
        self.reader_maps = None;
    }

    /// Refer every public in `publics` except the `exclude`d simple
    /// names. A namespace with no mappings of its own yet (each `ns`
    /// form's `refer-clojure`) takes the table itself — or a trimmed
    /// copy when names are excluded; anything else installs entry by
    /// entry like [`add_refer`](Self::add_refer), interns winning.
    pub(crate) fn refer_all(&mut self, publics: &PublicsTable, exclude: &[&str]) {
        if Arc::ptr_eq(&self.refers, &publics.table) && exclude.is_empty() {
            return;
        }
        let fresh = self.ids.is_empty()
            && self.refers.is_empty()
            && !publics.shadows_default_import
            && self
                .imports
                .get()
                .is_none_or(|m| Arc::ptr_eq(m, &crate::classemu::default_import_map()));
        if fresh {
            self.refers = if exclude.is_empty() {
                Arc::clone(&publics.table)
            } else {
                let mut table = (*publics.table).clone();
                for name in exclude {
                    table.remove(*name);
                }
                Arc::new(table)
            };
            self.reader_maps = None;
            return;
        }
        for (name, (target, id)) in publics.table.iter() {
            if exclude.contains(&&**name)
                || self
                    .ids
                    .contains_key(&(Some(self.name.as_str()), &**name) as &dyn SymKey)
            {
                continue;
            }
            Arc::make_mut(&mut self.refers).insert(Arc::clone(name), (target.clone(), *id));
            self.drop_import(name);
        }
        self.reader_maps = None;
    }

    /// Drop the intern, refer, and import mappings for a simple name
    /// (the var state survives, like the JVM — only the mappings
    /// go; imports are mappings too).
    pub fn unmap(&mut self, name: &str) {
        let q = Symbol::new(Some(&self.name), name);
        self.ids.remove(&q);
        self.publics_table = None;
        self.drop_refer(name);
        self.drop_import(name);
        self.reader_maps = None;
    }

    /// Install an import: simple name → class (S16 `import`).
    pub fn import_class(&mut self, simple: &str, class: Arc<ClassVal>) {
        Arc::make_mut(self.imports_mut()).insert(simple.to_owned(), class);
        self.reader_maps = None;
    }

    /// Drop the import of a simple name (`true` when there was one),
    /// copying the shared table only then.
    fn drop_import(&mut self, name: &str) -> bool {
        if let Some(m) = self.imports.get() {
            if !m.contains_key(name) {
                return false;
            }
        } else if !crate::classemu::is_default_import(name) {
            return false;
        }
        Arc::make_mut(self.imports_mut()).remove(name).is_some()
    }

    /// The import for a simple name, if any.
    #[must_use]
    pub fn get_import(&self, simple: &str) -> Option<Arc<ClassVal>> {
        self.imports_map().get(simple).cloned()
    }

    /// Import mappings: simple name to class.
    pub fn imports(&self) -> impl Iterator<Item = (&str, &Arc<ClassVal>)> + '_ {
        self.imports_map().iter().map(|(n, c)| (n.as_str(), c))
    }

    /// Interned mappings: qualified symbol to var id.
    pub fn interns(&self) -> impl Iterator<Item = (&Symbol, u64)> + '_ {
        self.ids.iter().map(|(s, i)| (s, *i))
    }

    /// Refer mappings: simple name to qualified target and the
    /// var id pinned at install.
    pub fn refers(&self) -> impl Iterator<Item = (&str, &Symbol, u64)> + '_ {
        self.refers.iter().map(|(n, (s, i))| (&**n, s, *i))
    }

    /// Aliases: alias to target namespace name and the id pinned
    /// at install.
    pub fn aliases(&self) -> impl Iterator<Item = (&str, &str, u64)> + '_ {
        self.aliases
            .iter()
            .map(|(a, (t, i))| (a.as_str(), t.as_str(), *i))
    }

    /// The var's state, if interned.
    #[must_use]
    pub fn state(&self, id: u64) -> Option<&VarState> {
        self.vars.get(&id)
    }

    /// The var's state, mutably (any write may change privacy, so the
    /// publics table goes).
    pub fn state_mut(&mut self, id: u64) -> Option<&mut VarState> {
        self.publics_table = None;
        self.vars.get_mut(&id)
    }
}

/// Find or create a namespace in `namespaces`, issuing a fresh
/// id on creation (split out so `intern` can share it without
/// tripping the borrow checker).
fn ensure_ns_in<'a>(
    namespaces: &'a mut FxHashMap<String, Namespace>,
    next_ns_id: &mut u64,
    name: &str,
) -> &'a mut Namespace {
    // Probe first: hits (nearly every call) allocate no key.
    if !namespaces.contains_key(name) {
        let id = *next_ns_id;
        *next_ns_id += 1;
        namespaces.insert(name.to_owned(), Namespace::new(name, id));
    }
    namespaces.get_mut(name).expect("namespace ensured above")
}

/// The namespace registry (S5A): all namespaces plus the current
/// one. Var ids come from one global counter and are never reused
/// across namespaces, so `Value::Var`/`Unbound` identity and
/// [`Bindings`] stay sound.
#[derive(Clone, Debug, Default)]
pub struct NamespaceRegistry {
    namespaces: FxHashMap<String, Namespace>,
    current: String,
    next_id: u64,
    next_ns_id: u64,
    /// Each var's home namespace name (shared with the namespace's
    /// handle — state lookups clone it per write).
    by_id: FxHashMap<u64, Arc<str>>,
    removed: FxHashMap<String, Namespace>,
    /// Anonymous var states (S4b `with-local-vars`): ids with no
    /// namespace home, so `all-ns`/`ns-interns` never see them.
    /// [`state`](Self::state)/[`state_mut`](Self::state_mut)
    /// answer from here when `by_id` misses.
    anon: FxHashMap<u64, VarState>,
    /// Intern vars without the `:ns`/`:name` stamp (a Lite program
    /// that references no var-metadata reader, #271).
    bare_vars: bool,
}

impl NamespaceRegistry {
    /// An empty registry (boot ensures namespaces explicitly).
    /// Ids start at 1, like the S4 namespace counter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            next_ns_id: 1,
            ..Self::default()
        }
    }

    /// The current namespace name.
    #[must_use]
    pub fn current(&self) -> &str {
        &self.current
    }

    /// Switch the current namespace (it must exist).
    pub fn set_current(&mut self, name: &str) {
        self.current = name.to_owned();
    }

    /// Find or create a namespace (creation issues a fresh id).
    pub fn ensure_ns(&mut self, name: &str) -> &mut Namespace {
        let Self {
            namespaces,
            next_ns_id,
            ..
        } = self;
        ensure_ns_in(namespaces, next_ns_id, name)
    }

    /// Look up a namespace by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Namespace> {
        self.namespaces.get(name)
    }

    /// Look up a namespace by name, mutably.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Namespace> {
        self.namespaces.get_mut(name)
    }

    /// A live namespace's id (`None` when missing or removed).
    #[must_use]
    pub fn ns_id(&self, name: &str) -> Option<u64> {
        self.namespaces.get(name).map(Namespace::id)
    }

    /// All live namespaces as `(name, id)`, sorted by name (the
    /// JVM order is hash-unspecified; the port is deterministic).
    #[must_use]
    pub fn all_ns_sorted(&self) -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> = self
            .namespaces
            .iter()
            .map(|(n, ns)| (n.clone(), ns.id()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// A live or removed namespace by name (removed ones stay
    /// readable through stale handles, like the JVM's detached
    /// objects; only the latest removal is kept).
    #[must_use]
    pub fn ns_live_or_removed(&self, name: &str) -> Option<&Namespace> {
        self.namespaces.get(name).or_else(|| self.removed.get(name))
    }

    /// A live namespace's publics as a [`PublicsTable`] (`None` when
    /// missing), cached in the namespace. An intern's state always
    /// lives beside its mapping ([`Namespace::intern`] inserts both,
    /// and states are never dropped), so the namespace sees every
    /// privacy change.
    pub(crate) fn publics_table(&mut self, ns_name: &str) -> Option<PublicsTable> {
        let ns = self.namespaces.get_mut(ns_name)?;
        if let Some(cached) = &ns.publics_table {
            return Some(cached.clone());
        }
        let private = Value::Keyword(clj_value::static_keyword!("private"));
        let mut table =
            ReferTable::with_capacity_and_hasher(ns.ids.len(), FxBuildHasher::default());
        for (sym, &id) in &ns.ids {
            let private = ns
                .vars
                .get(&id)
                .and_then(|st| st.meta.as_ref())
                .and_then(|m| m.get(&private))
                .is_some_and(Value::is_truthy);
            if !private {
                table.insert(Arc::clone(sym.name_arc()), (sym.clone(), id));
            }
        }
        let defaults = crate::classemu::default_import_map();
        let publics = PublicsTable {
            shadows_default_import: defaults.keys().any(|k| table.contains_key(k.as_str())),
            table: Arc::new(table),
        };
        ns.publics_table = Some(publics.clone());
        Some(publics)
    }

    /// A live or removed namespace by name, mutably (name-based
    /// callers only; handle-pinned writers use
    /// [`NamespaceRegistry::ns_for_handle_mut`]).
    pub fn ns_live_or_removed_mut(&mut self, name: &str) -> Option<&mut Namespace> {
        if self.namespaces.contains_key(name) {
            self.namespaces.get_mut(name)
        } else {
            self.removed.get_mut(name)
        }
    }

    /// A namespace through a handle-pinned id: the live namespace
    /// when its generation matches, else the tombstone when it
    /// matches (stale handles read frozen mappings after
    /// remove+recreate, like the JVM's detached objects). `None`
    /// when neither generation matches (only the latest removal
    /// is kept, so older handles dangle).
    #[must_use]
    pub fn ns_for_handle(&self, name: &str, id: u64) -> Option<&Namespace> {
        // (Match guards, not let-chains: MSRV 1.85.)
        match self.namespaces.get(name) {
            Some(ns) if ns.id() == id => return Some(ns),
            _ => {}
        }
        self.removed.get(name).filter(|ns| ns.id() == id)
    }

    /// [`NamespaceRegistry::ns_for_handle`], mutably.
    pub fn ns_for_handle_mut(&mut self, name: &str, id: u64) -> Option<&mut Namespace> {
        match self.namespaces.get_mut(name) {
            Some(ns) if ns.id() == id => return Some(ns),
            _ => {}
        }
        self.removed.get_mut(name).filter(|ns| ns.id() == id)
    }

    /// Drop a namespace (aliases pointing at it dangle, like the
    /// JVM; var ids are never reused). The namespace survives as
    /// a tombstone, so stale handles, captured vars, and the
    /// removed current namespace's own unqualified resolution keep
    /// reading their frozen state; only qualified resolution
    /// treats the name as gone.
    pub fn remove_ns(&mut self, name: &str) -> bool {
        let Some(ns) = self.namespaces.remove(name) else {
            return false;
        };
        self.removed.insert(name.to_owned(), ns);
        true
    }

    /// Qualify an unqualified symbol into the current namespace;
    /// explicit namespaces pass through untouched.
    #[must_use]
    pub fn qualify(&self, sym: &Symbol) -> Symbol {
        self.qualify_in(&self.current, sym)
    }

    /// [`qualify`](Self::qualify) against an explicit namespace
    /// (the analyzer passes the `*ns*`-synced name).
    #[must_use]
    pub fn qualify_in(&self, ns_name: &str, sym: &Symbol) -> Symbol {
        if sym.namespace().is_some() {
            sym.clone()
        } else {
            Symbol::new(Some(ns_name), sym.name())
        }
    }

    /// The namespace a qualifier names: aliases win over real
    /// namespaces (probe34 `ns-alias-shadow`); `None` is genuinely
    /// unknown.
    #[must_use]
    pub fn namespace_target<'a>(&'a self, part: &'a str) -> Option<&'a str> {
        self.namespace_target_in(&self.current, part)
    }

    /// [`namespace_target`](Self::namespace_target) against an
    /// explicit namespace (`ns-resolve` takes aliases from its
    /// namespace argument, not the current one).
    #[must_use]
    pub fn namespace_target_in<'a>(&'a self, ns_name: &str, part: &'a str) -> Option<&'a str> {
        self.ns_live_or_removed(ns_name)
            .and_then(|ns| ns.aliases.get(part))
            .map(|(target, _)| target.as_str())
            .or_else(|| self.namespaces.contains_key(part).then_some(part))
    }

    /// Resolve a symbol to its qualified target: qualified symbols
    /// via alias-or-namespace, unqualified via the current
    /// namespace then refers. `None` is unresolvable (callers pick
    /// the error shape).
    #[must_use]
    pub fn resolve(&self, sym: &Symbol) -> Option<Symbol> {
        self.resolve_in(&self.current, sym)
    }

    /// [`resolve`](Self::resolve) against an explicit namespace.
    #[must_use]
    pub fn resolve_in(&self, ns_name: &str, sym: &Symbol) -> Option<Symbol> {
        match sym.namespace() {
            Some(part) => {
                let target = self.namespace_target_in(ns_name, part)?;
                // Live-or-removed: aliases dangle into tombstones
                // (upstream `namespaceFor` returns the alias object
                // liveness-blind; #81). Global names resolve live
                // only — `namespace_target_in` never yields a
                // removed one — so only alias targets read frozen
                // state here.
                let ns = self.ns_live_or_removed(target)?;
                let key: &dyn SymKey = &(Some(target), sym.name());
                ns.ids
                    .contains_key(key)
                    .then(|| Symbol::new(Some(target), sym.name()))
            }
            None => {
                // A removed current namespace still resolves its own
                // mappings (like the JVM's detached object — the
                // `*ns*` handle outlives the name).
                let home = self.ns_live_or_removed(ns_name)?;
                if home
                    .ids
                    .contains_key(&(Some(ns_name), sym.name()) as &dyn SymKey)
                {
                    return Some(Symbol::new(Some(ns_name), sym.name()));
                }
                // Refers validate live: a refer into a removed
                // namespace stops resolving (the JVM keeps the var
                // object alive — fenced in namespaces-contract —
                // but must never resurrect the namespace here).
                home.refers
                    .get(sym.name())
                    .filter(|(target, _)| {
                        target
                            .namespace()
                            .and_then(|ns| self.namespaces.get(ns))
                            .is_some_and(|ns| ns.ids.contains_key(target))
                    })
                    .map(|(target, _)| target.clone())
            }
        }
    }

    /// Resolve a symbol to its var id (`None` is unresolvable).
    #[must_use]
    pub fn lookup(&self, sym: &Symbol) -> Option<u64> {
        self.lookup_in(&self.current, sym)
    }

    /// [`lookup`](Self::lookup) against an explicit namespace (the
    /// analyzer passes the `*ns*`-synced name). Single pass:
    /// alias targets dangle into tombstones (#81), bare names
    /// stay live-only.
    #[must_use]
    pub fn lookup_in(&self, ns_name: &str, sym: &Symbol) -> Option<u64> {
        self.resolve_target_and_id(ns_name, sym).map(|(_, id)| id)
    }

    /// Resolve a symbol to its qualified target and var id under
    /// one borrow — single-locked through `SharedRegistry`, so no
    /// concurrent unmap can slip between resolution and id lookup
    /// (#86). Callers needing both a target and its id should
    /// prefer this; live-only snapshots and cross-namespace
    /// re-resolution keep their own reads.
    #[must_use]
    pub fn resolve_target_and_id(&self, ns_name: &str, sym: &Symbol) -> Option<(Symbol, u64)> {
        let target = self.resolve_in(ns_name, sym)?;
        let home = target.namespace()?;
        let id = self.ns_live_or_removed(home)?.lookup(&target)?;
        Some((target, id))
    }

    /// Intern (find or create) a var: unqualified symbols land in
    /// the current namespace, qualified ones name theirs (created
    /// when missing — internal callers always hit existing ones).
    pub fn intern(&mut self, sym: &Symbol) -> u64 {
        let q = self.qualify(sym);
        let Self {
            namespaces,
            next_id,
            next_ns_id,
            by_id,
            current,
            bare_vars,
            ..
        } = self;
        let ns_name = q.namespace().unwrap_or(current.as_str());
        let ns = ensure_ns_in(namespaces, next_ns_id, ns_name);
        let before = *next_id;
        let id = ns.intern(&q, next_id, !*bare_vars);
        // Only a fresh var needs its home recorded (re-interns hit).
        if *next_id != before {
            by_id.insert(id, Arc::clone(&ns.handle.name));
        }
        id
    }

    /// [`intern`](Self::intern) and set the root in the same step
    /// (boot installs every native this way).
    pub fn intern_with_root(&mut self, sym: &Symbol, root: Value) -> u64 {
        let q = &self.qualify(sym);
        let Self {
            namespaces,
            next_id,
            next_ns_id,
            by_id,
            current,
            bare_vars,
            ..
        } = self;
        let ns_name = q.namespace().unwrap_or(current.as_str());
        let ns = ensure_ns_in(namespaces, next_ns_id, ns_name);
        let before = *next_id;
        let id = ns.intern(q, next_id, !*bare_vars);
        if *next_id != before {
            by_id.insert(id, Arc::clone(&ns.handle.name));
        }
        ns.state_mut(id)
            .expect("freshly interned var has state")
            .root = Some(root);
        id
    }

    /// Intern each `(symbol, root)` into `ns_name` with its root set
    /// (boot's batched native installs: the namespace resolves once,
    /// and the var-home table grows once).
    pub(crate) fn intern_roots(&mut self, ns_name: &str, roots: Vec<(Symbol, Value)>) {
        let Self {
            namespaces,
            next_id,
            next_ns_id,
            by_id,
            bare_vars,
            ..
        } = self;
        let ns = ensure_ns_in(namespaces, next_ns_id, ns_name);
        by_id.reserve(roots.len());
        for (sym, root) in roots {
            let before = *next_id;
            let id = ns.intern(&sym, next_id, !*bare_vars);
            if *next_id != before {
                by_id.insert(id, Arc::clone(&ns.handle.name));
            }
            ns.state_mut(id).expect("interned var has state").root = Some(root);
        }
    }

    /// Pre-size a namespace's var tables (boot knows its native
    /// count).
    pub(crate) fn reserve_vars(&mut self, ns_name: &str, additional: usize) {
        if let Some(ns) = self.namespaces.get_mut(ns_name) {
            ns.vars.reserve(additional);
            ns.ids.reserve(additional);
        }
    }

    /// Intern a `def`-family target: qualify into the current
    /// namespace, then intern live-or-removed (a removed current
    /// namespace's tombstone absorbs the `def` — the JVM's
    /// detached `*ns*` object — instead of resurrecting the
    /// name); create only when the name is in neither map
    /// (T73 finding 4 round 2: `def`/`declare`/`defmacro` after
    /// `remove-ns` must not bring the namespace back).
    pub fn intern_def_target(&mut self, sym: &Symbol) -> u64 {
        let q = self.qualify(sym);
        let ns_name = q.namespace().unwrap_or(&self.current).to_owned();
        self.intern_in(&ns_name, &q)
            .unwrap_or_else(|| self.intern(&q))
    }

    /// Intern a var into a live or removed namespace (never
    /// creates: callers resolve through `the-ns` first; live wins
    /// over the tombstone by name). Returns `None` only when
    /// the namespace is neither live nor removed.
    pub fn intern_in(&mut self, ns_name: &str, sym: &Symbol) -> Option<u64> {
        let Self {
            namespaces,
            removed,
            next_id,
            by_id,
            bare_vars,
            ..
        } = self;
        let ns = namespaces
            .get_mut(ns_name)
            .or_else(|| removed.get_mut(ns_name))?;
        let id = ns.intern(sym, next_id, !*bare_vars);
        by_id.insert(id, Arc::clone(&ns.handle.name));
        Some(id)
    }

    /// Intern a var through a handle-pinned id (stale handles land
    /// in the tombstone after remove+recreate). Returns the var id
    /// and the home namespace id, or `None` when neither
    /// generation matches the pin.
    pub fn intern_in_pinned(
        &mut self,
        ns_name: &str,
        expected: u64,
        sym: &Symbol,
    ) -> Option<(u64, u64)> {
        let Self {
            namespaces,
            removed,
            next_id,
            by_id,
            bare_vars,
            ..
        } = self;
        let ns = match namespaces.get_mut(ns_name) {
            Some(ns) if ns.id() == expected => ns,
            _ => removed.get_mut(ns_name).filter(|ns| ns.id() == expected)?,
        };
        let home = ns.id();
        let id = ns.intern(sym, next_id, !*bare_vars);
        by_id.insert(id, Arc::clone(&ns.handle.name));
        Some((id, home))
    }

    /// The var's state, if interned (tombstones answer for vars
    /// whose namespace was removed, and for unmapped-then-kept
    /// states, like the JVM's surviving var objects).
    #[must_use]
    pub fn state(&self, id: u64) -> Option<&VarState> {
        self.by_id
            .get(&id)
            .and_then(|name| {
                self.namespaces
                    .get(&**name)
                    .and_then(|ns| ns.state(id))
                    .or_else(|| self.removed.get(&**name).and_then(|ns| ns.state(id)))
            })
            .or_else(|| self.anon.get(&id))
    }

    /// Allocate an anonymous dynamic var (S4b
    /// `with-local-vars`, the JVM `Var.create().setDynamic()`):
    /// a registry-issued id with no namespace home, no root,
    /// `dynamic` set, and the JVM create-time meta (`{:ns nil,
    /// :name nil}`, oracle `wv-meta`).
    pub fn fresh_anonymous(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let meta = new_map()
            .insert(Value::Keyword(Keyword::new(None, "ns")), Value::Nil)
            .insert(Value::Keyword(Keyword::new(None, "name")), Value::Nil);
        self.anon.insert(
            id,
            VarState {
                root: None,
                meta: Some(meta),
                dynamic: true,
            },
        );
        id
    }

    /// The var's state, mutably.
    pub fn state_mut(&mut self, id: u64) -> Option<&mut VarState> {
        if self.anon.contains_key(&id) {
            return self.anon.get_mut(&id);
        }
        let name = Arc::clone(self.by_id.get(&id)?);
        if self
            .namespaces
            .get(&*name)
            .and_then(|ns| ns.state(id))
            .is_some()
        {
            return self.namespaces.get_mut(&*name)?.state_mut(id);
        }
        self.removed.get_mut(&*name)?.state_mut(id)
    }

    /// `Var.isBound` (minus thread bindings): a root that is absent
    /// or ANY `Unbound` sentinel counts as unbound.
    #[must_use]
    pub fn has_root(&self, id: u64) -> bool {
        self.state(id)
            .is_some_and(|s| !matches!(s.root, None | Some(Value::Unbound(_))))
    }

    /// The var's root, cloned (`None` when uninterned or rootless).
    /// Narrower than [`state`](Self::state): one `Value` clone, no
    /// metadata clone.
    #[must_use]
    pub fn root_cloned(&self, id: u64) -> Option<Value> {
        self.state(id).and_then(|s| s.root.clone())
    }

    /// Whether the var is `^:dynamic` (missing counts as non-dynamic).
    /// Narrower than [`state`](Self::state): no clone at all.
    #[must_use]
    pub fn is_dynamic(&self, id: u64) -> bool {
        self.state(id).is_some_and(|s| s.dynamic)
    }

    /// Whether the var's metadata maps `key` to a truthy value
    /// (`:macro`, `:private`; missing or cleared metadata reads
    /// false). No clone at all.
    #[must_use]
    pub fn meta_flag(&self, id: u64, key: &Keyword) -> bool {
        self.state(id).is_some_and(|s| {
            s.meta
                .as_ref()
                .and_then(|m| m.get(&Value::Keyword(key.clone())))
                .is_some_and(Value::is_truthy)
        })
    }

    /// The var's metadata, cloned (`None` when uninterned or cleared).
    /// Narrower than [`state`](Self::state): no root clone.
    #[must_use]
    pub fn meta_cloned(&self, id: u64) -> Option<Map> {
        self.state(id).and_then(|s| s.meta.clone())
    }

    /// `Var.deref` (minus thread bindings): the root, or the var's
    /// own `Unbound` sentinel when rootless. Never throws.
    #[must_use]
    pub fn root_or_unbound(&self, id: u64, sym: &Symbol) -> Value {
        match self.root_cloned(id) {
            Some(v) => v,
            None => Value::Unbound(Unbound {
                sym: sym.clone(),
                var_id: id,
            }),
        }
    }
}

/// Per-lineage shared evaluator state (S6B, D1 spike choice): the
/// namespace registry plus the identity counters behind one mutex.
/// Cloning shares (`future` bodies fork an evaluator over the same
/// handle, so `def`s inside a body are globally visible and bodies
/// see every committed `def`, like the JVM's global namespaces).
/// Bindings stay per-evaluator (conveyed explicitly at fork, like
/// `bound-fn`); only the registry and counters are lineage-global.
///
/// This is spike option (a) — per-thread evaluator clones over
/// shared `Arc<Mutex<..>>` state. Message-passing (b) would
/// serialize every global lookup through the spawning thread, and
/// `thread::scope` borrowing (c) cannot serve detached `future`
/// threads (the spawner returns before the body finishes).
///
/// Lock discipline: one lock per method call, never spanning
/// evaluation (a body that `def`s while another thread holds the
/// lock would deadlock — `Mutex` is not reentrant), and the
/// `with_*` closures below must neither touch the registry nor
/// evaluate. Compound read-modify-write sequences (e.g.
/// `alter-var-root`'s read-call-install) are NOT atomic across
/// threads; single calls (`intern`, publishes) are.
/// The registry `current` namespace is lineage-global, but the
/// per-evaluator `*ns*` binding shadows it through
/// `current_ns_name`, so the global fallback is unobservable
/// while `*ns*` stays bound (boot binds it permanently).
#[derive(Clone, Debug)]
pub struct SharedRegistry {
    inner: Arc<Mutex<SharedInner>>,
    /// The lineage root generation: every `with_*` writer bumps it
    /// (those are the only paths that can touch a `VarState`), and
    /// the per-evaluator root cache reads it lock-free (see
    /// `Evaluator::deref_var_cached`).
    epoch: Arc<AtomicU64>,
    /// The next throwable/closure/native identity (lock-free: closure
    /// creation draws one per evaluation — #226).
    ids: Arc<AtomicU64>,
    /// Deferred vars (see `core_lazy`): allocated lazily on the first
    /// deferred library install so `BootMode::Lite` pays zero allocation.
    lazy: Arc<OnceLock<Box<[OnceLock<LazyIndex>; LAZY_SEGMENTS]>>>,
    /// Serializes lazy forcing (reentrant: a forced form may force
    /// another), so no thread observes a var mid-force as unbound.
    lazy_force: Arc<VarAlterLock>,
}

/// The lineage-global contents: the registry plus the evaluator
/// identity counters (throwable/closure/native ids, `fn` display
/// numbers, `gensym`, form numbers — shared so forked evaluators
/// never stamp colliding identities).
#[derive(Debug)]
struct SharedInner {
    registry: NamespaceRegistry,
    fn_seq: u64,
    gensym: u64,
    eval_seq: u64,
    alter_locks: FxHashMap<u64, Arc<VarAlterLock>>,
    protocol_fn_methods: FxHashMap<u64, crate::typeinternals::ProtocolFnInfo>,
    /// Unforced lazy `core.clj` groups, by each var they define.
    lazy_groups: FxHashMap<u64, Arc<LazyGroup>>,
    /// Native-backed vars' file forms, kept as their metadata
    /// source (forced by a metadata read only).
    meta_groups: FxHashMap<u64, Arc<LazyGroup>>,
    /// The same forms, kept for good: a native's slow path evaluates
    /// one into its interpreted definition (`core_lazy::interpreted`).
    native_sources: FxHashMap<u64, Arc<LazyGroup>>,
    /// Those interpreted definitions, once evaluated.
    fallbacks: FxHashMap<u64, Value>,
}

/// Per-var reentrant alter lock (T1192 FX1): mirrors the JVM's
/// per-var monitor for `alter-var-root` — read-f-install runs
/// under exclusion, same-thread reentry nests. Textbook
/// mutex+condvar reentrant lock; the loom mirror
/// (`clj-eval/tests/loom_var_lock.rs`) explores every schedule.
#[derive(Debug)]
struct VarAlterLock {
    state: Mutex<AlterOwner>,
    cvar: Condvar,
}

#[derive(Debug)]
struct AlterOwner {
    owner: Option<std::thread::ThreadId>,
    depth: u32,
}

impl VarAlterLock {
    fn new() -> Self {
        Self {
            state: Mutex::new(AlterOwner {
                owner: None,
                depth: 0,
            }),
            cvar: Condvar::new(),
        }
    }

    /// Take the lock (same-thread reentry nests).
    fn acquire(self: Arc<Self>) -> VarAlterGuard {
        let current = std::thread::current().id();
        let mut owner = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if matches!(&owner.owner, Some(held) if held == &current) {
                owner.depth += 1;
                break;
            }
            if owner.owner.is_some() {
                owner = self.cvar.wait(owner).unwrap_or_else(|e| e.into_inner());
            } else {
                owner.owner = Some(current);
                owner.depth = 1;
                break;
            }
        }
        drop(owner);
        VarAlterGuard { lock: self }
    }
}

/// RAII release for [`SharedRegistry::lock_var_for_alter`].
#[derive(Debug)]
pub(crate) struct VarAlterGuard {
    lock: Arc<VarAlterLock>,
}

impl Drop for VarAlterGuard {
    fn drop(&mut self) {
        let mut owner = self.lock.state.lock().unwrap_or_else(|e| e.into_inner());
        owner.depth -= 1;
        if owner.depth == 0 {
            owner.owner = None;
            self.lock.cvar.notify_one();
        }
    }
}

impl Default for SharedRegistry {
    /// A fresh lineage (same seeds as [`SharedRegistry::new`]).
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SharedInner {
                registry: NamespaceRegistry::new(),
                fn_seq: 0,
                gensym: 1,
                eval_seq: 0,
                alter_locks: FxHashMap::default(),
                protocol_fn_methods: FxHashMap::default(),
                lazy_groups: FxHashMap::default(),
                meta_groups: FxHashMap::default(),
                native_sources: FxHashMap::default(),
                fallbacks: FxHashMap::default(),
            })),
            epoch: Arc::new(AtomicU64::new(0)),
            ids: Arc::new(AtomicU64::new(1)),
            lazy: Arc::new(OnceLock::new()),
            lazy_force: Arc::new(VarAlterLock::new()),
        }
    }
}

/// RAII guard bracketing a locked root/namespace mutation: enters at
/// an odd epoch and advances to the next even generation on drop so
/// lock-free [`clj_value::FastVars`] caches fall back while mutations
/// run.
struct EpochGuard<'a>(&'a AtomicU64);

impl Drop for EpochGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

impl SharedRegistry {
    /// A fresh lineage: an empty registry with evaluator counters
    /// seeded exactly like [`Evaluator::new_bare`](crate::eval::Evaluator::new_bare).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock the lineage state (short critical sections only — poison
    /// is unreachable).
    fn lock(&self) -> MutexGuard<'_, SharedInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The current root generation (lock-free; odd while a locked
    /// root/namespace mutation is active).
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Share the atomic root-generation counter with lock-free
    /// program caches.
    #[must_use]
    pub fn epoch_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.epoch)
    }

    fn begin_mutation(&self) -> EpochGuard<'_> {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        EpochGuard(&self.epoch)
    }

    /// The next throwable/closure/native identity.
    pub fn next_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::Relaxed)
    }

    /// The next `fn` display sequence number.
    pub fn next_fn_seq(&self) -> u64 {
        let mut inner = self.lock();
        inner.fn_seq += 1;
        inner.fn_seq
    }

    /// Record a protocol method fn's cache identity (S14
    /// `__protocol_method*`, emitted by `defprotocol`).
    pub(crate) fn note_protocol_fn(&self, id: u64, info: crate::typeinternals::ProtocolFnInfo) {
        self.lock().protocol_fn_methods.insert(id, info);
    }

    /// A protocol method fn's cache identity, if registered
    /// (S14 `-cache-protocol-fn`).
    pub(crate) fn protocol_fn_info(&self, id: u64) -> Option<crate::typeinternals::ProtocolFnInfo> {
        self.lock().protocol_fn_methods.get(&id).cloned()
    }

    /// The next `gensym` counter value.
    pub fn next_gensym(&self) -> u64 {
        let mut inner = self.lock();
        let n = inner.gensym;
        inner.gensym += 1;
        n
    }

    /// Reset the `gensym` counter to 1 (boot restarts the
    /// user-visible counter after loading `core.clj`).
    pub fn reset_gensym(&self) {
        self.lock().gensym = 1;
    }

    /// The current top-level form number.
    #[must_use]
    pub fn eval_seq(&self) -> u64 {
        self.lock().eval_seq
    }

    /// Number the next top-level form, returning its number.
    pub fn bump_eval_seq(&self) -> u64 {
        let mut inner = self.lock();
        inner.eval_seq += 1;
        inner.eval_seq
    }

    /// The current namespace name.
    #[must_use]
    pub fn current(&self) -> String {
        self.lock().registry.current().to_owned()
    }

    /// Switch the current namespace (it must exist).
    pub fn set_current(&self, name: &str) {
        self.lock().registry.set_current(name);
    }

    /// Find or create a namespace, returning its id. Never the
    /// namespace itself: a clone copies every var, refer, and
    /// import table, and `refer` calls this once per referred var.
    pub fn ensure_ns(&self, name: &str) -> u64 {
        self.lock().registry.ensure_ns(name).id()
    }

    /// Look up a namespace by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Namespace> {
        self.lock().registry.get(name).cloned()
    }

    /// Look up a var id directly in a live namespace's interns
    /// (like [`get`](Self::get) + [`lookup`](Namespace::lookup),
    /// without cloning the namespace — the analyzer resolves
    /// through this per symbol).
    #[must_use]
    pub fn lookup_in_ns(&self, ns_name: &str, sym: &Symbol) -> Option<u64> {
        self.lock().registry.get(ns_name)?.lookup(sym)
    }

    /// [`lookup_in_ns`](Self::lookup_in_ns) for `ns_name/name` without
    /// building a `Symbol` (boot installs probe every core name).
    #[must_use]
    pub fn lookup_str(&self, ns_name: &str, name: &str) -> Option<u64> {
        self.lock().registry.get(ns_name)?.lookup_str(name)
    }

    /// Look up a var id directly in a live or removed namespace's
    /// interns (like [`ns_live_or_removed`](Self::ns_live_or_removed)
    /// + [`lookup`](Namespace::lookup), without the clone).
    #[must_use]
    pub fn lookup_in_live_or_removed(&self, name: &str, sym: &Symbol) -> Option<u64> {
        self.lock().registry.ns_live_or_removed(name)?.lookup(sym)
    }

    /// A live namespace's id (`None` when missing or removed).
    #[must_use]
    pub fn ns_id(&self, name: &str) -> Option<u64> {
        self.lock().registry.ns_id(name)
    }

    /// All live namespaces as `(name, id)`, sorted by name.
    #[must_use]
    pub fn all_ns_sorted(&self) -> Vec<(String, u64)> {
        self.lock().registry.all_ns_sorted()
    }

    /// A live or removed namespace by name (see
    /// [`NamespaceRegistry::ns_live_or_removed`]).
    #[must_use]
    pub fn ns_live_or_removed(&self, name: &str) -> Option<Namespace> {
        self.lock().registry.ns_live_or_removed(name).cloned()
    }

    /// A live namespace's publics as a shareable refer table (see
    /// [`NamespaceRegistry::publics_table`]).
    #[must_use]
    pub(crate) fn publics_table(&self, ns_name: &str) -> Option<PublicsTable> {
        self.lock().registry.publics_table(ns_name)
    }

    /// A live or removed namespace's import of a simple name (a read:
    /// no epoch bump; a removed `*ns*` keeps resolving its imports,
    /// like the JVM's detached namespace object).
    #[must_use]
    pub fn import_in(&self, ns_name: &str, simple: &str) -> Option<Arc<ClassVal>> {
        self.lock()
            .registry
            .ns_live_or_removed(ns_name)?
            .get_import(simple)
    }

    /// A live or removed namespace's id (no namespace clone).
    #[must_use]
    pub fn ns_live_or_removed_id(&self, name: &str) -> Option<u64> {
        self.lock()
            .registry
            .ns_live_or_removed(name)
            .map(Namespace::id)
    }

    /// A live or removed namespace's [`ReaderMaps`] (cached in the
    /// namespace; filling the cache bumps no epoch — it is not var
    /// state).
    #[must_use]
    pub fn reader_maps(&self, name: &str) -> Option<ReaderMaps> {
        self.lock()
            .registry
            .ns_live_or_removed_mut(name)
            .map(Namespace::reader_maps)
    }

    /// A namespace through a handle-pinned id (see
    /// [`NamespaceRegistry::ns_for_handle`]).
    #[must_use]
    pub fn ns_for_handle(&self, name: &str, id: u64) -> Option<Namespace> {
        self.lock().registry.ns_for_handle(name, id).cloned()
    }

    /// Drop a namespace (see [`NamespaceRegistry::remove_ns`]). A
    /// mutation like any other: by-name caches keyed on the epoch
    /// (`Evaluator::core_read`) must not outlive the namespace.
    pub fn remove_ns(&self, name: &str) -> bool {
        let mut inner = self.lock();
        let _epoch = self.begin_mutation();
        inner.registry.remove_ns(name)
    }

    /// Qualify an unqualified symbol against an explicit namespace.
    #[must_use]
    pub fn qualify_in(&self, ns_name: &str, sym: &Symbol) -> Symbol {
        self.lock().registry.qualify_in(ns_name, sym)
    }

    /// The namespace a qualifier names, against an explicit namespace.
    #[must_use]
    pub fn namespace_target_in(&self, ns_name: &str, part: &str) -> Option<String> {
        self.lock()
            .registry
            .namespace_target_in(ns_name, part)
            .map(str::to_owned)
    }

    /// Resolve a symbol to its qualified target, against an explicit namespace.
    #[must_use]
    pub fn resolve_in(&self, ns_name: &str, sym: &Symbol) -> Option<Symbol> {
        self.lock().registry.resolve_in(ns_name, sym)
    }

    /// Resolve a symbol to its var id (`None` is unresolvable).
    #[must_use]
    pub fn lookup(&self, sym: &Symbol) -> Option<u64> {
        self.lock().registry.lookup(sym)
    }

    /// [`lookup`](Self::lookup) against an explicit namespace (the
    /// analyzer passes the `*ns*`-synced name).
    #[must_use]
    pub fn lookup_in(&self, ns_name: &str, sym: &Symbol) -> Option<u64> {
        self.lock().registry.lookup_in(ns_name, sym)
    }

    /// Resolve a symbol to target and var id (see
    /// [`NamespaceRegistry::resolve_target_and_id`]; one lock).
    pub fn resolve_target_and_id(&self, ns_name: &str, sym: &Symbol) -> Option<(Symbol, u64)> {
        self.lock().registry.resolve_target_and_id(ns_name, sym)
    }

    /// Intern (find or create) a var (see
    /// [`NamespaceRegistry::intern`]).
    pub fn intern(&self, sym: &Symbol) -> u64 {
        self.lock().registry.intern(sym)
    }

    /// Intern a var and install its root under one lock (native
    /// installs: one lock and one generation bump instead of two).
    pub fn intern_with_root(&self, sym: &Symbol, root: Value) -> u64 {
        let mut inner = self.lock();
        let _epoch = self.begin_mutation();
        inner.registry.intern_with_root(sym, root)
    }

    /// [`NamespaceRegistry::intern_roots`] under one lock and one
    /// mutation epoch.
    pub(crate) fn intern_roots(&self, ns_name: &str, roots: Vec<(Symbol, Value)>) {
        let mut inner = self.lock();
        let _epoch = self.begin_mutation();
        inner.registry.intern_roots(ns_name, roots);
    }

    /// Pre-size a namespace's var tables (see
    /// [`NamespaceRegistry::reserve_vars`]).
    pub(crate) fn reserve_vars(&self, ns_name: &str, additional: usize) {
        self.lock().registry.reserve_vars(ns_name, additional);
    }

    /// Intern later vars without the `:ns`/`:name` metadata stamp (see
    /// [`Namespace::intern`]).
    pub(crate) fn set_bare_vars(&self, bare: bool) {
        self.lock().registry.bare_vars = bare;
    }

    /// Allocate an anonymous dynamic var (see
    /// [`NamespaceRegistry::fresh_anonymous`]).
    pub fn fresh_anonymous(&self) -> u64 {
        self.lock().registry.fresh_anonymous()
    }

    /// Intern a var into a live or removed namespace (see
    /// [`NamespaceRegistry::intern_in`]).
    pub fn intern_in(&self, ns_name: &str, sym: &Symbol) -> Option<u64> {
        self.lock().registry.intern_in(ns_name, sym)
    }

    /// Intern a `def`-family target (see
    /// [`NamespaceRegistry::intern_def_target`]).
    pub fn intern_def_target(&self, sym: &Symbol) -> u64 {
        self.lock().registry.intern_def_target(sym)
    }

    /// Intern a var through a handle-pinned id (see
    /// [`NamespaceRegistry::intern_in_pinned`]).
    pub fn intern_in_pinned(
        &self,
        ns_name: &str,
        expected: u64,
        sym: &Symbol,
    ) -> Option<(u64, u64)> {
        self.lock()
            .registry
            .intern_in_pinned(ns_name, expected, sym)
    }

    /// The var's state, if interned (an owned copy — mutate through
    /// [`with_state_mut`](Self::with_state_mut)).
    #[must_use]
    pub fn state(&self, id: u64) -> Option<VarState> {
        self.lock().registry.state(id).cloned()
    }

    /// The var's root, cloned (see
    /// [`NamespaceRegistry::root_cloned`]).
    #[must_use]
    pub fn root_cloned(&self, id: u64) -> Option<Value> {
        self.lock().registry.root_cloned(id)
    }

    /// Whether the var is `^:dynamic` (see
    /// [`NamespaceRegistry::is_dynamic`]).
    #[must_use]
    pub fn is_dynamic(&self, id: u64) -> bool {
        self.lock().registry.is_dynamic(id)
    }

    /// See [`NamespaceRegistry::meta_flag`] (one lock, no clone).
    #[must_use]
    pub fn meta_flag(&self, id: u64, key: &Keyword) -> bool {
        self.lock().registry.meta_flag(id, key)
    }

    /// The var's metadata, cloned (see
    /// [`NamespaceRegistry::meta_cloned`]).
    #[must_use]
    pub fn meta_cloned(&self, id: u64) -> Option<Map> {
        self.lock().registry.meta_cloned(id)
    }

    /// `Var.isBound` (minus thread bindings).
    #[must_use]
    pub fn has_root(&self, id: u64) -> bool {
        self.lock().registry.has_root(id)
    }

    /// `Var.deref` (minus thread bindings). Never throws.
    #[must_use]
    pub fn root_or_unbound(&self, id: u64, sym: &Symbol) -> Value {
        self.lock().registry.root_or_unbound(id, sym)
    }

    /// Acquire a var's alter lock (reentrant exclusion for
    /// `alter-var-root`'s read-f-install, like the JVM's var
    /// monitor). Blocks while another thread alters the same var;
    /// same-thread reentry nests. The table lookup holds the
    /// lineage lock only briefly — never across the wait.
    pub(crate) fn lock_var_for_alter(&self, id: u64) -> VarAlterGuard {
        let lock = {
            let mut inner = self.lock();
            inner
                .alter_locks
                .entry(id)
                .or_insert_with(|| Arc::new(VarAlterLock::new()))
                .clone()
        };
        lock.acquire()
    }

    /// Whether `id` is an unforced deferred var (lock-free).
    pub(crate) fn is_lazy(&self, id: u64) -> bool {
        self.lazy
            .get()
            .is_some_and(|segs| segs.iter().map_while(OnceLock::get).any(|ix| ix.test(id)))
    }

    /// Whether a lazy index segment is still free.
    pub(crate) fn lazy_room(&self) -> bool {
        self.lazy
            .get()
            .is_none_or(|segs| segs.iter().any(|seg| seg.get().is_none()))
    }

    /// Install one file's deferred groups in the next free segment
    /// (`false` when none is left; a later group claims a var an
    /// earlier one also defined, like a redefinition).
    pub(crate) fn install_lazy(&self, groups: Vec<LazyGroup>) -> bool {
        let mut ids = Vec::new();
        let mut inner = self.lock();
        for g in groups {
            let g = Arc::new(g);
            for &id in &g.ids {
                inner.lazy_groups.insert(id, Arc::clone(&g));
                ids.push(id);
            }
        }
        // Under the registry lock, so concurrent loads claim
        // distinct segments.
        let index = LazyIndex::new(&ids);
        let segs = self
            .lazy
            .get_or_init(|| Box::new(std::array::from_fn(|_| OnceLock::new())));
        segs.iter()
            .find(|seg| seg.get().is_none())
            .is_some_and(|seg| seg.set(index).is_ok())
    }

    /// Serialize a lazy force (see [`lazy_force`](Self::lazy_force)).
    pub(crate) fn lock_lazy_force(&self) -> VarAlterGuard {
        Arc::clone(&self.lazy_force).acquire()
    }

    /// Claim the unforced group defining `id`, unlisting all its
    /// vars (`None` once claimed). The index bits stay set until
    /// [`finish_lazy`](Self::finish_lazy), so other threads wait on
    /// the force lock instead of reading an unbound root.
    pub(crate) fn take_lazy(&self, id: u64) -> Option<Arc<LazyGroup>> {
        let mut inner = self.lock();
        let group = inner.lazy_groups.remove(&id)?;
        for other in &group.ids {
            inner.lazy_groups.remove(other);
        }
        Some(group)
    }

    /// Install metadata-source groups for native-backed vars.
    pub(crate) fn install_meta_lazy(&self, groups: Vec<LazyGroup>) {
        let mut inner = self.lock();
        for g in groups {
            let g = Arc::new(g);
            for &id in &g.ids {
                inner.meta_groups.insert(id, Arc::clone(&g));
                inner.native_sources.insert(id, Arc::clone(&g));
            }
        }
    }

    /// A native-backed var's file form (never consumed).
    pub(crate) fn native_source(&self, id: u64) -> Option<Arc<LazyGroup>> {
        self.lock().native_sources.get(&id).cloned()
    }

    /// A native-backed var's interpreted definition, once evaluated.
    pub(crate) fn fallback(&self, id: u64) -> Option<Value> {
        self.lock().fallbacks.get(&id).cloned()
    }

    /// Record [`fallback`](Self::fallback).
    pub(crate) fn set_fallback(&self, id: u64, f: Value) {
        self.lock().fallbacks.insert(id, f);
    }

    /// Claim `id`'s metadata-source group (`None` when absent).
    pub(crate) fn take_meta_lazy(&self, id: u64) -> Option<Arc<LazyGroup>> {
        self.lock().meta_groups.remove(&id)
    }

    /// Publish a forced group: its vars stop testing lazy.
    pub(crate) fn finish_lazy(&self, group: &LazyGroup) {
        let segs = self.lazy.get().into_iter().flat_map(|segs| segs.iter());
        for ix in segs.map_while(OnceLock::get) {
            for &id in &group.ids {
                ix.clear(id);
            }
        }
    }

    /// The gensym, top-level form, and `fn` display counters (lazy
    /// forcing restores them, so it never shifts user numbering).
    pub(crate) fn counters(&self) -> (u64, u64, u64) {
        let inner = self.lock();
        (inner.gensym, inner.eval_seq, inner.fn_seq)
    }

    /// Restore [`counters`](Self::counters).
    pub(crate) fn restore_counters(&self, (gensym, eval_seq, fn_seq): (u64, u64, u64)) {
        let mut inner = self.lock();
        inner.gensym = gensym;
        inner.eval_seq = eval_seq;
        inner.fn_seq = fn_seq;
    }

    /// Run `f` over a var's state (`None` when uninterned). The
    /// closure must neither touch the registry nor evaluate (the
    /// lock is held throughout). Bumps the root generation (the
    /// closure may have changed the root).
    pub fn with_state_mut<R>(&self, id: u64, f: impl FnOnce(&mut VarState) -> R) -> Option<R> {
        let mut inner = self.lock();
        let _epoch = self.begin_mutation();
        let state = inner.registry.state_mut(id)?;
        Some(f(state))
    }

    /// Run `f` over a live namespace (`None` when missing). Same
    /// closure discipline as [`with_state_mut`](Self::with_state_mut).
    /// Bumps the root generation (the closure sees `&mut` states).
    pub fn with_ns_mut<R>(&self, name: &str, f: impl FnOnce(&mut Namespace) -> R) -> Option<R> {
        let mut inner = self.lock();
        let _epoch = self.begin_mutation();
        let ns = inner.registry.get_mut(name)?;
        Some(f(ns))
    }

    /// Run `f` over a live or removed namespace (`None` when
    /// neither). Same closure discipline as
    /// [`with_state_mut`](Self::with_state_mut). Bumps the root
    /// generation (the closure sees `&mut` states).
    pub fn with_live_or_removed_mut<R>(
        &self,
        name: &str,
        f: impl FnOnce(&mut Namespace) -> R,
    ) -> Option<R> {
        let mut inner = self.lock();
        let _epoch = self.begin_mutation();
        let ns = inner.registry.ns_live_or_removed_mut(name)?;
        Some(f(ns))
    }

    /// Run `f` over a handle-pinned namespace (`None` when neither
    /// generation matches). Same closure discipline as
    /// [`with_state_mut`](Self::with_state_mut). Bumps the root
    /// generation (the closure sees `&mut` states).
    pub fn with_handle_mut<R>(
        &self,
        name: &str,
        id: u64,
        f: impl FnOnce(&mut Namespace) -> R,
    ) -> Option<R> {
        let mut inner = self.lock();
        let _epoch = self.begin_mutation();
        let ns = inner.registry.ns_for_handle_mut(name, id)?;
        Some(f(ns))
    }
}

/// Process-wide per-var active binding refcounts across all
/// [`Bindings`] stacks: initialized lazily only when requested via
/// [`bind_counts`], so interpreter runs avoid the 40 KB table unless
/// lock-free [`clj_value::FastVars`] caches are active.
static BIND_COUNTS: OnceLock<Arc<BindCounts>> = OnceLock::new();
static PRE_BINDS: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());

fn apply_bind_count(bc: &BindCounts, var_id: u64, n: usize, add: bool) {
    if add {
        bc.add(var_id, n);
    } else {
        bc.sub(var_id, n);
    }
}

fn update_pending_bind(pending: &mut Vec<(u64, usize)>, var_id: u64, n: usize, add: bool) {
    if add {
        match pending.iter_mut().find(|(id, _)| *id == var_id) {
            Some((_, count)) => *count += n,
            None => pending.push((var_id, n)),
        }
        return;
    }
    let pos = pending
        .iter()
        .position(|&(id, _)| id == var_id)
        .expect("live binding has pending refcount");
    pending[pos].1 -= n;
    if pending[pos].1 == 0 {
        pending.swap_remove(pos);
    }
}

fn track_bind_locked(
    counts: &OnceLock<Arc<BindCounts>>,
    pending: &mut Vec<(u64, usize)>,
    var_id: u64,
    n: usize,
    add: bool,
) {
    if let Some(bc) = counts.get() {
        apply_bind_count(bc, var_id, n, add);
        return;
    }
    update_pending_bind(pending, var_id, n, add);
}

fn track_bind(var_id: u64, n: usize, add: bool) {
    if let Some(bc) = BIND_COUNTS.get() {
        apply_bind_count(bc, var_id, n, add);
        return;
    }
    let mut pending = clj_value::unpoison(PRE_BINDS.lock());
    track_bind_locked(&BIND_COUNTS, &mut pending, var_id, n, add);
}

/// Shared atomic handle to [`BIND_COUNTS`], allocating and seeding it
/// from any active pre-init bindings on first use.
#[must_use]
pub fn bind_counts() -> Arc<BindCounts> {
    if let Some(bc) = BIND_COUNTS.get() {
        return Arc::clone(bc);
    }
    let mut pending = clj_value::unpoison(PRE_BINDS.lock());
    let bc = BIND_COUNTS.get_or_init(|| {
        let bc = Arc::new(BindCounts::new());
        for &(var_id, n) in pending.iter() {
            bc.add(var_id, n);
        }
        pending.clear();
        pending.shrink_to_fit();
        bc
    });
    Arc::clone(bc)
}

/// The thread-binding stack: one frame per `binding` /
/// `push-thread-bindings` / `with-bindings*` / `bound-fn`
/// invocation. Entries shadow by recency across frames; `set!`
/// writes the topmost entry for its var.
#[derive(Debug, Default)]
pub struct Bindings {
    frames: Vec<Vec<(Var, Value)>>,
    /// Bound-var refcounts: var id → live entry count (across all
    /// frames). Lets [`get`](Self::get) skip the frame scan when a
    /// var has no thread binding (the common case on the var-deref
    /// hot path).
    counts: FxHashMap<u64, usize>,
    /// The content version: fresh (process-unique) after every
    /// change, copied by clones, 0 for never-changed (empty)
    /// stacks — equal versions mean equal bindings, so the VM's
    /// per-site caches validate against it.
    version: u64,
}

impl Clone for Bindings {
    fn clone(&self) -> Self {
        for (&id, &n) in &self.counts {
            track_bind(id, n, true);
        }
        Self {
            frames: self.frames.clone(),
            counts: self.counts.clone(),
            version: self.version,
        }
    }
}

impl Drop for Bindings {
    fn drop(&mut self) {
        for (&id, &n) in &self.counts {
            track_bind(id, n, false);
        }
    }
}

/// The [`Bindings::version`] source.
static BINDINGS_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Drop one frame's refcounts (split out so [`Bindings::truncate`]
/// can drain frames and counts under separate borrows).
fn unbind_counts(counts: &mut FxHashMap<u64, usize>, frame: &[(Var, Value)]) {
    for (var, _) in frame {
        // Every live entry was counted at push, so the entry exists
        // with a positive count; a missing one would be a caller bug
        // (`and_modify` ignores it, like a double pop elsewhere).
        counts
            .entry(var.id)
            .and_modify(|n| *n = n.saturating_sub(1));
        if counts.get(&var.id).is_some_and(|n| *n == 0) {
            counts.remove(&var.id);
        }
    }
}

impl Bindings {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The frame depth (for restoring after a `binding` body).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// The content version (see the field).
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    fn bump(&mut self) {
        self.version = BINDINGS_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Push one frame (possibly empty, like the JVM).
    pub fn push_frame(&mut self, entries: Vec<(Var, Value)>) {
        for (var, _) in &entries {
            track_bind(var.id, 1, true);
            *self.counts.entry(var.id).or_default() += 1;
        }
        self.frames.push(entries);
        self.bump();
    }

    /// Pop one frame (`None` when the stack is empty —
    /// `pop-thread-bindings` raises then).
    pub fn pop_frame(&mut self) -> Option<Vec<(Var, Value)>> {
        let frame = self.frames.pop()?;
        unbind_counts(&mut self.counts, &frame);
        for (var, _) in &frame {
            track_bind(var.id, 1, false);
        }
        self.bump();
        Some(frame)
    }

    /// Pop back to an earlier depth (past-the-end depths are a
    /// lenient no-op, like `Vec::truncate`).
    pub fn truncate(&mut self, depth: usize) {
        if depth >= self.frames.len() {
            return;
        }
        self.bump();
        let Self { frames, counts, .. } = self;
        for frame in frames.drain(depth..) {
            unbind_counts(counts, &frame);
            for (var, _) in &frame {
                track_bind(var.id, 1, false);
            }
        }
    }

    /// Whether no thread binding is live (the refcounts drain to
    /// empty on pop, so this is exact even with pushed-but-empty
    /// frames — and a bare length check, so var-deref fast paths
    /// skip the map probe entirely).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// The visible value for a var, if bound.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Value> {
        if self.counts.is_empty() || !self.counts.contains_key(&id) {
            return None;
        }
        self.frames.iter().rev().find_map(|frame| {
            frame
                .iter()
                .rev()
                .find_map(|(var, v)| (var.id == id).then_some(v))
        })
    }

    /// Every entry oldest-first (for snapshots: later entries win
    /// per var when folded into a map).
    pub fn pairs(&self) -> impl Iterator<Item = (&Var, &Value)> + '_ {
        self.frames
            .iter()
            .flat_map(|frame| frame.iter().map(|(var, v)| (var, v)))
    }

    /// `set!`: replace the topmost entry for a var (false when no
    /// entry exists — the evaluator raises then).
    pub fn set_top(&mut self, id: u64, value: Value) -> bool {
        match self
            .frames
            .iter_mut()
            .rev()
            .find_map(|frame| frame.iter_mut().rev().find(|(var, _)| var.id == id))
        {
            Some((_, slot)) => {
                *slot = value;
                self.bump();
                true
            }
            None => false,
        }
    }
}

/// One frame's slots: immutable shared slots (the common case —
/// lock-free reads), a single value stored inline in the frame (the
/// `extend_one` fast path — one allocation instead of two, since the
/// hottest binder binds one name at a time), or patchable slots
/// behind a mutex (only for `letfn`/named-`fn` self-patching, where
/// closures captured the frame before its slots were installed).
#[derive(Clone, Debug)]
enum Slots {
    Shared(Arc<[Value]>),
    Inline(Value),
    Patchable(Arc<Mutex<Vec<Value>>>),
}

/// One linked frame: absolute slot indexes (`Local`) keep working
/// because `base` is the parent chain's length (`[base, len)` are
/// this frame's slots, `[0, base)` the parent's).
#[derive(Clone, Debug)]
struct Frame {
    parent: Option<Env>,
    base: usize,
    len: usize,
    slots: Slots,
}

/// The shared empty frame (every `Env::new` aliases it — no
/// allocation per top-level form).
static EMPTY_FRAME: std::sync::LazyLock<Arc<Frame>> = std::sync::LazyLock::new(|| {
    Arc::new(Frame {
        parent: None,
        base: 0,
        len: 0,
        slots: Slots::Shared(Arc::from([])),
    })
});

/// Local slots: a linked chain of immutable frames, shared
/// (`Arc`) so closures capture their frame. Extension allocates
/// only the new frame's slots (older frames are never cloned —
/// iteration N's closures keep observing iteration N's frame, the
/// old copy-on-`recur` promise); only `letfn`/named-`fn` patching
/// mutates, through a `Patchable` frame (before any observer
/// exists). Shared-frame reads take no lock; locks are never held
/// across evaluation.
#[derive(Clone, Debug)]
pub struct Env {
    frame: Arc<Frame>,
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame: Arc::clone(&EMPTY_FRAME),
        }
    }

    /// The slot count (O(1), no lock).
    #[must_use]
    pub fn len(&self) -> usize {
        self.frame.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a slot reference without cloning (#226: for borrow-path
    /// evaluation and fast integer ops). Returns `None` for
    /// `Patchable` frames or out-of-bounds indices.
    #[must_use]
    pub fn get_ref(&self, idx: usize) -> Option<&Value> {
        let mut frame: &Frame = &self.frame;
        while idx < frame.base {
            frame = &frame
                .parent
                .as_ref()
                .expect("parentless frame with base > 0")
                .frame;
        }
        match &frame.slots {
            Slots::Shared(slots) => slots.get(idx - frame.base),
            Slots::Inline(value) => {
                if idx == frame.base {
                    Some(value)
                } else {
                    None
                }
            }
            Slots::Patchable(_) => None,
        }
    }

    /// Read a slot (cloned; missing slots read `nil`, unreachable
    /// by construction). Shared frames read lock-free; only
    /// `Patchable` frames lock.
    #[must_use]
    pub fn get(&self, idx: usize) -> Value {
        // Pointer-chase through shared refs (no `Arc` bumps): an
        // index at or past `base` belongs to this frame (`[0,
        // base)` is the parent's), so the top frame usually answers
        // immediately. The root frame has base 0, so the descent
        // always lands (a missing parent here would be a caller bug
        // — fail loud, like an out-of-range index elsewhere).
        let mut frame: &Frame = &self.frame;
        while idx < frame.base {
            frame = &frame
                .parent
                .as_ref()
                .expect("parentless frame with base > 0")
                .frame;
        }
        match &frame.slots {
            Slots::Shared(slots) => slots.get(idx - frame.base).cloned().unwrap_or(Value::Nil),
            Slots::Inline(value) => {
                if idx == frame.base {
                    value.clone()
                } else {
                    Value::Nil
                }
            }
            Slots::Patchable(slots) => slots
                .lock()
                .ok()
                .and_then(|guard| guard.get(idx - frame.base).cloned())
                .unwrap_or(Value::Nil),
        }
    }

    /// Extend with fresh slots (the receiver is untouched — only
    /// the new frame allocates).
    #[must_use]
    pub fn extend(&self, values: Vec<Value>) -> Self {
        let base = self.len();
        let len = base + values.len();
        Self {
            frame: Arc::new(Frame {
                parent: Some(self.clone()),
                base,
                len,
                slots: Slots::Shared(values.into()),
            }),
        }
    }

    /// Rebind this frame's slots in place for `recur` when no
    /// observer can tell: the frame must be the flat all-symbols
    /// frame over `outer` (direct child at `outer`'s length — a
    /// trailing destructured frame in a nested init chain can hold
    /// the same slot count by coincidence, so the length check
    /// alone is unsound), with shared, uniquely owned slots of
    /// matching length. Any closure capturing this env — or a child
    /// of it, whose parent chain shares this frame — bumps the
    /// count and forces the fresh-frame fallback, preserving the
    /// copy-on-`recur` promise exactly (`true` = rebound in
    /// place). Anything unqualified leaves `values` untouched
    /// for the caller to bind fresh; a successful rebind drains
    /// it (the caller recycles the buffer — see `recur_pool`).
    pub fn try_rebind(&mut self, outer: &Env, values: &mut Vec<Value>) -> bool {
        // Each guard doubles as a cheap safety net: a captured
        // frame (or any shape surprise) falls through to the
        // fresh-frame path. Slots ride with their frame (never
        // shared alone — that `expect` never fires).
        let Some(frame) = Arc::get_mut(&mut self.frame) else {
            return false;
        };
        let flat_over_outer = frame.parent.as_ref().is_some_and(|parent| {
            Arc::ptr_eq(&parent.frame, &outer.frame) && frame.base == outer.len()
        });
        if flat_over_outer {
            let Slots::Shared(slots) = &mut frame.slots else {
                return false;
            };
            let dst = Arc::get_mut(slots).expect("slots ride with their frame");
            // Destructuring loops bind fewer values than slots (one
            // per pattern), so a mismatch falls back to a fresh
            // bind — only exact rebinds go in place. Values move
            // (never clone): each assignment drops the old slot
            // value, and the drain leaves the buffer for the
            // caller's pool (#235: a `clone_from_slice` here
            // cloned every slot per `recur`).
            if dst.len() == values.len() {
                for (slot, new_val) in dst.iter_mut().zip(values.drain(..)) {
                    *slot = new_val;
                }
                return true;
            }
        }
        false
    }

    /// Extend with one fresh slot (the single-name `bind_pattern`
    /// fast path — the value rides inline in the frame, so one
    /// allocation instead of two).
    #[must_use]
    pub fn extend_one(&self, value: Value) -> Self {
        let base = self.len();
        Self {
            frame: Arc::new(Frame {
                parent: Some(self.clone()),
                base,
                len: base + 1,
                slots: Slots::Inline(value),
            }),
        }
    }

    /// Extend with `count` `nil` slots for `letfn`/named-`fn`
    /// patching: the frame is `Patchable`, so [`set`](Self::set)
    /// installs each closure where its capturers observe it.
    #[must_use]
    pub fn extend_patchable(&self, count: usize) -> Self {
        let base = self.len();
        Self {
            frame: Arc::new(Frame {
                parent: Some(self.clone()),
                base,
                len: base + count,
                slots: Slots::Patchable(Arc::new(Mutex::new(vec![Value::Nil; count]))),
            }),
        }
    }

    /// Overwrite one slot (`letfn`/named-`fn` patching only, on a
    /// [`extend_patchable`](Self::extend_patchable) frame;
    /// `recur` rebinds a fresh frame instead since #171). Setting
    /// a live shared slot is a caller bug (debug-asserted, ignored
    /// in release); past-the-end writes are silent no-ops (the old
    /// out-of-range leniency).
    pub fn set(&self, idx: usize, value: Value) {
        // Descend like [`get`](Self::get) (the root's base 0 always
        // lands the loop).
        let mut frame: &Frame = &self.frame;
        while idx < frame.base {
            frame = &frame
                .parent
                .as_ref()
                .expect("parentless frame with base > 0")
                .frame;
        }
        // Past the whole chain (`[0, base)` is the parent's,
        // `[base, len)` this frame's): nothing to write.
        if idx >= frame.len {
            return;
        }
        if let Slots::Patchable(slots) = &frame.slots {
            // Poison is unreachable (nothing panics while holding a
            // frame lock); recover like `SharedRegistry::lock` so a
            // poisoned lock keeps the write instead of dropping it.
            let mut guard = clj_value::unpoison(slots.lock());
            guard[idx - frame.base] = value;
        } else {
            debug_assert!(false, "set on an immutable frame");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interned_roots_replace_existing_roots() {
        let mut reg = NamespaceRegistry::new();
        let sym = Symbol::new(Some("lib"), "f");
        reg.intern_roots("lib", vec![(sym.clone(), Value::Int(1))]);
        let id = reg.intern(&sym);
        reg.intern_roots("lib", vec![(sym.clone(), Value::Int(2))]);
        assert_eq!(reg.intern(&sym), id);
        assert_eq!(
            reg.state(id).and_then(|st| st.root.clone()),
            Some(Value::Int(2))
        );
    }

    fn sym(ns: Option<&str>, name: &str) -> Symbol {
        Symbol::new(ns, name)
    }

    /// Registry with `user` current, `lib` holding `v`, and `user`
    /// aliasing `q` → `lib` plus a `w` refer → `lib/v`.
    fn rigged() -> NamespaceRegistry {
        let mut reg = NamespaceRegistry::new();
        reg.ensure_ns("user");
        reg.set_current("user");
        reg.ensure_ns("lib");
        let v_id = reg.intern(&sym(Some("lib"), "v"));
        let lib_id = reg.ns_id("lib").unwrap();
        reg.get_mut("user")
            .unwrap()
            .add_alias("q", "lib", lib_id)
            .unwrap();
        reg.get_mut("user")
            .unwrap()
            .add_refer("w", sym(Some("lib"), "v"), v_id);
        reg
    }

    #[test]
    fn resolve_current_alias_refer() {
        let reg = rigged();
        assert_eq!(
            reg.resolve(&sym(Some("lib"), "v")),
            Some(sym(Some("lib"), "v"))
        );
        assert_eq!(
            reg.resolve(&sym(Some("q"), "v")),
            Some(sym(Some("lib"), "v"))
        );
        assert_eq!(reg.resolve(&sym(None, "w")), Some(sym(Some("lib"), "v")));
        assert_eq!(reg.resolve(&sym(None, "missing")), None);
        assert_eq!(reg.resolve(&sym(Some("lib"), "missing")), None);
        assert_eq!(reg.resolve(&sym(Some("nope"), "v")), None);
    }

    #[test]
    fn aliases_win_over_namespaces() {
        let mut reg = rigged();
        reg.ensure_ns("q");
        reg.intern(&sym(Some("q"), "v"));
        // `q/v` resolves through the alias (probe34
        // `ns-alias-shadow`), not the real `q` namespace.
        assert_eq!(
            reg.resolve(&sym(Some("q"), "v")),
            Some(sym(Some("lib"), "v"))
        );
        assert_eq!(reg.namespace_target("q"), Some("lib"));
    }

    #[test]
    fn resolve_target_and_id_shapes() {
        // #86: the single-locked helper resolves every shape —
        // aliases dangle into tombstones (#81) while bare removed
        // names and stale refers miss.
        let reg = rigged();
        let v = sym(Some("lib"), "v");
        let v_id = reg.lookup(&v).expect("live lib/v resolves");
        assert_eq!(
            reg.resolve_target_and_id("user", &sym(Some("q"), "v")),
            Some((v.clone(), v_id))
        );
        assert_eq!(
            reg.resolve_target_and_id("user", &sym(Some("q"), "missing")),
            None
        );
        assert_eq!(
            reg.resolve_target_and_id("user", &v),
            Some((v.clone(), v_id))
        );
        assert_eq!(
            reg.resolve_target_and_id("user", &sym(None, "w")),
            Some((v.clone(), v_id))
        );
        assert_eq!(
            reg.resolve_target_and_id("user", &sym(None, "missing")),
            None
        );
        let mut reg = rigged();
        assert!(reg.remove_ns("lib"));
        assert_eq!(
            reg.resolve_target_and_id("user", &sym(Some("q"), "v")),
            Some((v.clone(), v_id))
        );
        assert_eq!(
            reg.resolve_target_and_id("user", &sym(Some("q"), "missing")),
            None
        );
        assert_eq!(reg.resolve_target_and_id("user", &v), None);
        assert_eq!(reg.resolve_target_and_id("user", &sym(None, "w")), None);
    }

    #[test]
    fn dangling_alias_stays_namespace_shaped() {
        let mut reg = rigged();
        assert!(reg.remove_ns("lib"));
        // The alias still names a namespace (probe34
        // `ns-alias-removed`): it resolves to the detached object
        // (upstream `namespaceFor` returns the alias object
        // liveness-blind), so an existing var is found while a
        // missing one keeps the `No such var` shape (#81).
        assert_eq!(reg.namespace_target("q"), Some("lib"));
        assert_eq!(
            reg.resolve(&sym(Some("q"), "v")),
            Some(sym(Some("lib"), "v"))
        );
        assert_eq!(reg.resolve(&sym(Some("q"), "missing")), None);
        assert_eq!(reg.namespace_target("lib"), None);
        // Stale refers stop resolving instead of resurrecting
        // the namespace (fenced: the JVM keeps the var object).
        assert_eq!(reg.resolve(&sym(None, "w")), None);
        assert!(!reg.namespaces.contains_key("lib"));
    }

    #[test]
    fn var_ids_are_globally_unique() {
        let mut reg = rigged();
        let a = reg.intern(&sym(None, "a"));
        let b = reg.intern(&sym(Some("lib"), "b"));
        assert_ne!(a, b);
        assert!(reg.state(a).is_some());
        assert!(reg.state(b).is_some());
        assert!(reg.remove_ns("lib"));
        assert!(reg.state(a).is_some());
        // S5A2 tombstones supersede S5A1's drop: the state survives
        // removal (the JVM keeps the var object alive, so stale
        // handles and captured vars keep reading it) — only
        // resolution treats the name as gone.
        assert!(reg.state(b).is_some());
        assert_eq!(reg.resolve(&sym(Some("lib"), "b")), None);
    }

    #[test]
    fn qualify_uses_current() {
        let reg = rigged();
        assert_eq!(reg.qualify(&sym(None, "a")), sym(Some("user"), "a"));
        assert_eq!(reg.qualify(&sym(Some("lib"), "a")), sym(Some("lib"), "a"));
    }

    #[test]
    fn lookup_qualifies_here_and_rejects_foreign() {
        let reg = rigged();
        // Unqualified resolves here; foreign qualifiers never do.
        let lib = reg.get("lib").unwrap();
        assert_eq!(
            lib.lookup(&sym(None, "v")),
            lib.lookup(&sym(Some("lib"), "v"))
        );
        assert!(lib.lookup(&sym(None, "v")).is_some());
        assert_eq!(
            reg.get("user").unwrap().lookup(&sym(Some("lib"), "v")),
            None
        );
    }

    #[test]
    fn live_or_removed_mut_answers_tombstones() {
        let mut reg = rigged();
        assert!(reg.remove_ns("lib"));
        // Live names take the live branch; removed ones fall to the
        // tombstone.
        assert!(reg.ns_live_or_removed_mut("user").is_some());
        assert!(reg.ns_live_or_removed_mut("lib").is_some());
        assert!(reg.ns_live_or_removed_mut("missing").is_none());
    }

    #[test]
    fn env_empty_tracks_slots() {
        let env = Env::new();
        assert!(env.is_empty());
        assert!(!env.extend(vec![Value::Nil]).is_empty());
    }

    #[test]
    fn env_linked_frames_read_write() {
        // Empty env reads nil everywhere.
        let env = Env::new();
        assert!(env.is_empty());
        assert_eq!(env.len(), 0);
        assert_eq!(env.get(0), Value::Nil);
        assert_eq!(env.get_ref(0), None);
        assert_eq!(env.get(99), Value::Nil);
        assert_eq!(env.get_ref(99), None);
        assert_eq!(Env::default().len(), 0);

        // Shared extension: absolute indexing, receiver untouched.
        let e1 = env.extend(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(e1.len(), 2);
        assert_eq!(e1.get(0), Value::Int(1));
        assert_eq!(e1.get_ref(0), Some(&Value::Int(1)));
        assert_eq!(e1.get(1), Value::Int(2));
        assert_eq!(e1.get_ref(1), Some(&Value::Int(2)));
        assert_eq!(e1.get(2), Value::Nil);
        assert_eq!(e1.get_ref(2), None);
        assert_eq!(env.len(), 0);

        // `extend_one` chains; reads fall through to the parent.
        let e2 = e1.extend_one(Value::Int(3));
        assert_eq!(e2.len(), 3);
        assert_eq!(e2.get(0), Value::Int(1));
        assert_eq!(e2.get_ref(0), Some(&Value::Int(1)));
        assert_eq!(e2.get(2), Value::Int(3));
        assert_eq!(e2.get_ref(2), Some(&Value::Int(3)));
        assert_eq!(e2.get(9), Value::Nil);
        assert_eq!(e2.get_ref(9), None);

        // Empty extension keeps the base==len corner sound.
        let e_empty = e1.extend(vec![]);
        assert_eq!(e_empty.len(), 2);
        assert_eq!(e_empty.get(0), Value::Int(1));
        assert_eq!(e_empty.get_ref(0), Some(&Value::Int(1)));
        assert_eq!(e_empty.get(2), Value::Nil);
        assert_eq!(e_empty.get_ref(2), None);

        // Patchable frames: `set` is visible through the shared Arc.
        let p = e1.extend_patchable(2);
        assert_eq!(p.len(), 4);
        assert_eq!(p.get(2), Value::Nil);
        assert_eq!(p.get_ref(2), None);
        p.set(2, Value::Int(20));
        p.set(3, Value::Int(30));
        let p_alias = p.clone();
        assert_eq!(p_alias.get(2), Value::Int(20));
        assert_eq!(p_alias.get(3), Value::Int(30));
        // Parent slots still answer through a patchable child.
        assert_eq!(p_alias.get(0), Value::Int(1));
        // Setting through a shared child reaches the patchable parent.
        let child = p.extend_one(Value::Int(40));
        assert_eq!(child.len(), 5);
        child.set(2, Value::Int(21));
        assert_eq!(p.get(2), Value::Int(21));
        // Out-of-range writes are silent no-ops (old leniency).
        child.set(99, Value::Int(1));
        assert_eq!(child.get(99), Value::Nil);
    }

    #[test]
    #[should_panic(expected = "immutable frame")]
    fn env_set_shared_panics_in_debug() {
        let env = Env::new().extend_one(Value::Nil);
        env.set(0, Value::Int(1));
    }

    #[test]
    fn bindings_counts_track_push_pop_truncate() {
        let var = |id: u64| Var {
            id,
            sym: sym(None, "x"),
        };
        let mut b = Bindings::new();
        assert_eq!(b.get(1), None);
        b.push_frame(vec![(var(1), Value::Int(1)), (var(2), Value::Int(2))]);
        assert_eq!(b.get(1), Some(&Value::Int(1)));
        // A nested re-bind shadows, and its pop restores (refcounts
        // stack instead of dropping the outer entry).
        b.push_frame(vec![(var(1), Value::Int(10))]);
        assert_eq!(b.get(1), Some(&Value::Int(10)));
        b.pop_frame();
        assert_eq!(b.get(1), Some(&Value::Int(1)));
        // Past-the-end truncate is a lenient no-op.
        b.truncate(99);
        assert_eq!(b.get(2), Some(&Value::Int(2)));
        b.truncate(0);
        assert_eq!(b.get(1), None);
        assert_eq!(b.get(2), None);
        assert!(b.pop_frame().is_none());
        // `set_top` needs no count change.
        b.push_frame(vec![(var(3), Value::Int(3))]);
        assert!(b.set_top(3, Value::Int(30)));
        assert!(!b.set_top(99, Value::Int(0)));
        assert_eq!(b.get(3), Some(&Value::Int(30)));
    }

    #[test]
    fn bindings_is_empty_tracks_entries() {
        let var = |id: u64| Var {
            id,
            sym: sym(None, "x"),
        };
        let mut b = Bindings::new();
        assert!(b.is_empty());
        // Pushed-but-empty frames hold no entries (still empty).
        b.push_frame(vec![]);
        assert!(b.is_empty());
        b.push_frame(vec![(var(1), Value::Int(1))]);
        assert!(!b.is_empty());
        b.pop_frame();
        assert!(b.is_empty());
        b.pop_frame();
        assert!(b.is_empty());

        // Pending bindings accumulate before `bind_counts()` is first
        // called, seed into the global table, and track atomically afterward.
        let empty_clone = b.clone();
        drop(empty_clone);
        b.push_frame(vec![(var(2), Value::Int(2))]);
        let pre_clone = b.clone();
        b.pop_frame();
        let bc = bind_counts();
        assert!(bc.is_bound(2));
        drop(pre_clone);
        assert!(!bc.is_bound(2));

        let same_bc = bind_counts();
        assert!(Arc::ptr_eq(&bc, &same_bc));
        b.push_frame(vec![(var(3), Value::Int(3))]);
        assert!(bc.is_bound(3));
        let non_empty_clone = b.clone();
        b.truncate(0);
        assert!(bc.is_bound(3));
        drop(non_empty_clone);
        assert!(!bc.is_bound(3));

        // Cover post-lock OnceLock initialization race fallback directly.
        let seeded = OnceLock::new();
        let _ = seeded.set(Arc::new(BindCounts::new()));
        let mut dummy_pending = Vec::new();
        super::track_bind_locked(&seeded, &mut dummy_pending, 4, 1, true);
        assert!(seeded.get().unwrap().is_bound(4));
        super::track_bind_locked(&seeded, &mut dummy_pending, 4, 1, false);
        assert!(!seeded.get().unwrap().is_bound(4));
    }

    #[test]
    fn reserve_vars_and_intern_def_target_tolerate_missing_ns() {
        let mut reg = NamespaceRegistry::new();
        reg.ensure_ns("user");
        reg.reserve_vars("no.such.ns", 4);
        reg.reserve_vars("user", 4);
        let id = reg.intern_def_target(&sym(Some("brand.new.ns"), "v"));
        assert!(reg.ns_id("brand.new.ns").is_some());
        assert_eq!(reg.lookup(&sym(Some("brand.new.ns"), "v")), Some(id));
        // Lazy `imports`: lowercase interns skip `default_import_map`; dropping a
        // default import (`String`) materializes it, then initialized misses stay cheap.
        let mut ns = Namespace::new("lazy.imports", 1);
        assert!(!ns.drop_import("lower"));
        assert!(ns.drop_import("String"));
        assert!(!ns.drop_import("Missing"));
        assert!(ns.imports().any(|(n, _)| n == "Long"));
    }
}
