;; Copyright (c) Rich Hickey. All rights reserved.
;; The use and distribution terms for this software are covered by the
;; Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;; which can be found in the file epl-v10.html at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

;; gap-122 slices 1+E+A+D2+B: a faithful subset port of clojure.spec.alpha
;; (upstream spec.alpha-0.6.249, clojure/spec/alpha.clj) covering the
;; spec registry, `def`, predicate specs, `s/and`, `valid?`,
;; `conform`/`unform`, basic `explain`/`explain-str`, and (slice E)
;; regex specs: `s/cat`, `s/alt`, `s/*`, `s/+`, `s/?`, `s/&` over
;; the derivative engine, plus the `s/keys*` macro (slice F2
;; unblocks runtime use: `::kvs->map` + the fixed `gen` head).
;;
;; gap-122 slice A adds map specs (`s/keys`/`map-spec-impl` and
;; `s/merge`/`merge-spec-impl`; upstream 414-474, 503-509,
;; 822-908, 1195-1221), narrowed like slice 1 (no `:gen`, no
;; conformers; slice F2 restores `:gen`): `MapSpec`/`MergeSpec`
;; records, `pvalid?`, and the list-built `keys`/`merge` macros
;; (see docs/gap-122/slice-A.md).
;;
;; Port adaptations (see docs/gap-122/slice-1.md,
;; docs/gap-122/slice-A.md and docs/gap-122/slice-E.md for the full lists):
;; - No syntax-quote and no `::auto-resolve` anywhere in this file:
;;   the port reads a whole load batch before evaluating it, so both
;;   would qualify against the *requiring* namespace instead of this
;;   one. Macros build expansions with `list` (fully-qualified
;;   heads), and spec keywords are written out in full.
;; - Spec objects are `defrecord`s (PredSpec/AndSpec/RegexSpec/
;;   MapSpec/MergeSpec/TupleSpec/MultiSpec/NonconformingSpec) with
;;   the Spec protocol extended to them: the port's `reify` instances
;;   do not support metadata, while the JVM's do (`with-name` needs it).
;; - `specize` is a plain `cond` (not the Specize protocol): the
;;   port has no `Object`-default protocol dispatch, and set dispatch
;;   would miss sorted sets (exact-class keys). Behavior matches.
;; - `fn-sym` always yields nil (the port exposes no fn-name
;;   primitive), so bare-fn specs carry the `::unknown` form.
;; - `Exception.` construction is `ex-info` (same message, caught by
;;   `catch Exception` all the same).
;; - `->sym` reads the var's `:ns`/`:name` meta (no `(symbol var)`).
;; - `ns-qualify` uses `ns-name` (no `.name` interop, no `str` of
;;   namespaces).
;; - `gen*`/`with-gen*` are wired on every record (slice F2);
;;   every record carries the upstream `gfn` (Or/Multi/Map also
;;   the `id`), and `-impl` ctors keep short arities delegating
;;   with nil `gfn`, as upstream.
;; - `tagged-ret` is `(first {tag ret})`: the port cannot build a
;;   `MapEntry` object, but the entry of a single-entry map is one.
;; - `regex-spec-impl` keeps the `[re]` arity (nil `gfn`) for
;;   `maybe-spec`; the non-sequential `explain*` `:pred`
;;   is the literal `(clojure.core/fn [%] (clojure.core/or ...))`
;;   form (the deterministic value of upstream's `(res ...)` call).
;; - `keys*` keeps `let`/`fn`/core fns bare in its expansion
;;   (use-site refers; only the unqualified `let` is the special
;;   form) while spec/gen heads fully qualify. Slice F2 fixes the
;;   `gen` head to `clojure.spec.alpha/gen` (upstream's
;;   syntax-quote resolves it there, not to the gen alias) and
;;   registers `::kvs->map`, unblocking runtime use.
;;
;; gap-122 slice D2 adds `tuple`/`multi-spec`/`conformer`/
;; `nonconforming` (see docs/gap-122/slice-D2.md): `PredSpec`
;; carries the upstream `cpred?`/`unc` conformer flags,
;; `spec-impl`/`dt` grow the upstream conformer arities
;; (minus the `:gen` parameter), and the new `TupleSpec`/
;; `MultiSpec`/`NonconformingSpec` records mirror the upstream
;; `tuple-impl`/`multi-spec-impl`/`nonconforming` reifies.
;; Multi-spec dispatch-value/method lookup is
;; `clojure.core/get-method` plus the `__dispatch_fn*` glue
;; native (the port has no `.dispatchFn`/`.getMethod` interop).
;;
;; gap-122 slice B (see docs/gap-122/slice-B.md) appends the
;; `s/or`, `s/nilable`, `s/every`, `s/every-kv`,
;; `s/coll-of`, and `s/map-of` combinators below the slice-D2
;; code (`every-kv` reuses slice D2's `s/tuple`).
;; Further port adaptations: evaluated `Integer/MAX_VALUE`
;; reads are the literal 2147483647; the sampling step uses `quot`
;; (no `long`); the `every` opts map is built with an explicit
;; `apply hash-map`; `or` joins the `:refer-clojure :exclude`
;; list. (`tagged-ret` is slice E's `(first {tag ret})`, reused.)
;;
;; gap-122 slice F2 wires `gen*`/`with-gen*` on every record above
;; (plus `s/gen`, `s/with-gen`, `s/exercise`, `s/exercise-fn` and
;; the `:gen` kwargs of `s/spec`/`s/keys`/`s/every`; see
;; docs/gap-122/slice-F2.md). Records carry the upstream `gfn`
;; (Or/Multi/Map also the `id`); `-impl` ctors keep short arities
;; delegating with nil `gfn`, as upstream.
(ns clojure.spec.alpha
  "The spec library specifies the structure of data or functions and provides
  operations to validate, conform, explain, describe, and generate data based on
  the specs.

  Rationale: https://clojure.org/about/spec
  Guide: https://clojure.org/guides/spec"
  (:refer-clojure :exclude [and def + * cat keys merge or])
  (:require [clojure.spec.gen.alpha :as gen]))

(alias 'c 'clojure.core)

;; Upstream alpha.clj 44-50.
(defprotocol Spec
  (conform* [spec x])
  (unform* [spec y])
  (explain* [spec path via in x])
  (gen* [spec overrides path rmap])
  (with-gen* [spec gfn])
  (describe* [spec]))

;; Spec object records. Upstream uses `reify` (alpha.clj 913,
;; 1151); the port's reify instances carry no metadata while its
;; records do, and `with-name`/`spec-name` store the registered name
;; in metadata. AndSpec's `specs` field holds a delay (as upstream),
;; so forward references resolve at first conform, not at build.
;; Slice D2: PredSpec carries the upstream `cpred?`/`unc`
;; conformer flags (nil for plain predicate specs). Slice F2: every
;; record carries the upstream `gfn` (nil unless `with-gen`/`:gen`).
(defrecord PredSpec [form pred cpred? unc gfn])
(defrecord AndSpec [forms preds specs gfn])
;; Slice E (upstream alpha.clj 1697-1723): the Spec glide over a
;; regex op map. Upstream closes over `re` in a `reify`; the port
;; holds it in a record field (metadata-carrying, as above).
(defrecord RegexSpec [re gfn])

;; MapSpec fields mirror the upstream `map-spec-impl` argm (plus the
;; precomputed `k->s` table standing in for the `keys->specnames`
;; closure); MergeSpec mirrors `merge-spec-impl`'s `[forms preds]`.
(defrecord MapSpec [req-un opt-un keys-pred pred-exprs opt-keys req-specs req req-keys opt-specs pred-forms opt gfn id k->s])
(defrecord MergeSpec [forms preds gfn])

;; Forward declarations: the `extend-type` blocks below call fns
;; defined later in this file (the port resolves globals at analyze
;; time), as do `specize` and `def-impl` (-> `spec-impl`),
;; `maybe-spec`/`spec-impl` (-> `regex-spec-impl`), and MapSpec methods
;; (-> `registry`/`conform`/`pvalid?`/`explain-1`).
(declare spec-impl dt invalid? unform explain-pred-list regex-spec-impl re-conform op-unform re-explain re-gen op-describe regex-guard-form registry conform pvalid? explain-1 gensub inck recur-limit? and-spec-impl map-spec-impl merge-spec-impl or-spec-impl tuple-impl multi-spec-impl nilable-impl every-impl nonconforming k-gen or-k-gen and-k-gen empty-coll)

(extend-type PredSpec
  Spec
  (conform* [this x]
    (let [ret ((:pred this) x)]
      (if (:cpred? this)
        ret
        (if ret x :clojure.spec.alpha/invalid))))
  (unform* [this x]
    (if (:cpred? this)
      (if-let [unc (:unc this)]
        (unc x)
        (throw (ex-info "no unform fn for conformer" {})))
      x))
  (explain* [this path via in x]
    (when (invalid? (dt (:pred this) x (:form this) (:cpred? this)))
      [{:path path :pred (:form this) :val x :via via :in in}]))
  (gen* [this _ _ _]
    (if (:gfn this)
      ((:gfn this))
      (gen/gen-for-pred (:pred this))))
  (with-gen* [this gfn]
    (spec-impl (:form this) (:pred this) gfn (:cpred? this) (:unc this)))
  (describe* [this] (:form this)))

(extend-type AndSpec
  Spec
  (conform* [this x]
    (let [specs @(:specs this)]
      (loop [ret x i 0]
        (if (< i (count specs))
          (let [nret (conform* (specs i) ret)]
            (if (invalid? nret)
              :clojure.spec.alpha/invalid
              (recur nret (inc i))))
          ret))))
  (unform* [this x]
    (reduce #(unform %2 %1) x (reverse (:preds this))))
  (explain* [this path via in x]
    (explain-pred-list (:forms this) (:preds this) path via in x))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (gensub (first (:preds this)) overrides path rmap (first (:forms this)))))
  (with-gen* [this gfn]
    (and-spec-impl (:forms this) (:preds this) gfn))
  (describe* [this]
    (apply list 'clojure.spec.alpha/and (:forms this))))

(extend-type RegexSpec
  Spec
  (conform* [this x]
    (if (c/or (nil? x) (sequential? x))
      (re-conform (:re this) (seq x))
      :clojure.spec.alpha/invalid))
  (unform* [this x] (op-unform (:re this) x))
  (explain* [this path via in x]
    (if (c/or (nil? x) (sequential? x))
      (re-explain path via in (:re this) (seq x))
      [{:path path :pred regex-guard-form :val x :via via :in in}]))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (re-gen (:re this) overrides path rmap (op-describe (:re this)))))
  (with-gen* [this gfn]
    (regex-spec-impl (:re this) gfn))
  (describe* [this] (op-describe (:re this))))

;; Upstream alpha.clj 834-877/878-902/903-908. `(k->s k)` is
;; map-as-fn, as upstream's closure.
(extend-type MapSpec
  Spec
  (conform* [this m]
    (if ((:keys-pred this) m)
      (let [reg (registry)
            k->s (:k->s this)]
        (loop [ret m, [[k v] & ks :as keys] m]
          (if keys
            (let [sname (c/or (k->s k) k)]
              (if-let [s (get reg sname)]
                (let [cv (conform s v)]
                  (if (invalid? cv)
                    :clojure.spec.alpha/invalid
                    (recur (if (identical? cv v) ret (assoc ret k cv))
                           ks)))
                (recur ret ks)))
            ret)))
      :clojure.spec.alpha/invalid))
  (unform* [this m]
    (let [reg (registry)
          k->s (:k->s this)]
      (loop [ret m, [k & ks :as keys] (c/keys m)]
        (if keys
          (if (contains? reg (c/or (k->s k) k))
            (let [cv (get m k)
                  v (unform (c/or (k->s k) k) cv)]
              (recur (if (identical? cv v) ret (assoc ret k v))
                     ks))
            (recur ret ks))
          ret))))
  (explain* [this path via in x]
    (if-not (map? x)
      [{:path path :pred 'clojure.core/map? :val x :via via :in in}]
      (let [reg (registry)
            k->s (:k->s this)]
        (apply concat
               (when-let [probs (->> (map (fn [pred form] (when-not (pred x) form))
                                          (:pred-exprs this) (:pred-forms this))
                                     (keep identity)
                                     seq)]
                 (map #(identity {:path path :pred % :val x :via via :in in})
                      probs))
               (map (fn [[k v]]
                      (when-not (c/or (not (contains? reg (c/or (k->s k) k)))
                                      (pvalid? (c/or (k->s k) k) v k))
                        (explain-1 (c/or (k->s k) k) (c/or (k->s k) k)
                                   (conj path k) via (conj in k) v)))
                    (seq x))))))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (let [rmap (inck rmap (:id this))
            rgen (fn [k s] [k (gensub s overrides (conj path k) rmap k)])
            ogen (fn [k s]
                   (when-not (recur-limit? rmap (:id this) path k)
                     [k (gen/delay (gensub s overrides (conj path k) rmap k))]))
            reqs (map rgen (:req-keys this) (:req-specs this))
            opts (remove nil? (map ogen (:opt-keys this) (:opt-specs this)))]
        (when (every? identity (concat (map second reqs) (map second opts)))
          (gen/bind
           (gen/tuple
            (and-k-gen (:req this))
            (or-k-gen (:opt this))
            (and-k-gen (:req-un this))
            (or-k-gen (:opt-un this)))
           (fn [[req-ks opt-ks req-un-ks opt-un-ks]]
             (let [qks (flatten (concat req-ks opt-ks))
                   unqks (map (comp keyword name) (flatten (concat req-un-ks opt-un-ks)))]
               (->> (into reqs opts)
                    (filter #((set (concat qks unqks)) (first %)))
                    (apply concat)
                    (apply gen/hash-map)))))))))
  (with-gen* [this gfn]
    (map-spec-impl {:req-un (:req-un this) :opt-un (:opt-un this)
                    :keys-pred (:keys-pred this) :pred-exprs (:pred-exprs this)
                    :opt-keys (:opt-keys this) :req-specs (:req-specs this)
                    :req (:req this) :req-keys (:req-keys this)
                    :opt-specs (:opt-specs this) :pred-forms (:pred-forms this)
                    :opt (:opt this) :gfn gfn}))
  (describe* [this]
    (cons 'clojure.spec.alpha/keys
          (cond-> []
            (:req this) (conj :req (:req this))
            (:opt this) (conj :opt (:opt this))
            (:req-un this) (conj :req-un (:req-un this))
            (:opt-un this) (conj :opt-un (:opt-un this))))))

;; Upstream alpha.clj 1204-1221.
(extend-type MergeSpec
  Spec
  (conform* [this x]
    (let [ms (map #(dt %1 x %2) (:preds this) (:forms this))]
      (if (some invalid? ms)
        :clojure.spec.alpha/invalid
        (apply c/merge ms))))
  (unform* [this x]
    (apply c/merge (map #(unform % x) (reverse (:preds this)))))
  (explain* [this path via in x]
    (apply concat
           (map #(explain-1 %1 %2 path via in x)
                (:forms this) (:preds this))))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (gen/fmap
       #(apply c/merge %)
       (apply gen/tuple (map #(gensub %1 overrides path rmap %2)
                             (:preds this) (:forms this))))))
  (with-gen* [this gfn]
    (merge-spec-impl (:forms this) (:preds this) gfn))
  (describe* [this]
    (apply list 'clojure.spec.alpha/merge (:forms this))))

;; Upstream alpha.clj 52.
(defonce ^:private registry-ref (atom {}))

;; Upstream alpha.clj 54-58.
(defn- deep-resolve [reg k]
  (loop [spec k]
    (if (ident? spec)
      (recur (get reg spec))
      spec)))

;; Upstream alpha.clj 60-69.
(defn- reg-resolve
  "returns the spec/regex at end of alias chain starting with k, nil if not found, k if k not ident"
  [k]
  (if (ident? k)
    (let [reg @registry-ref
          spec (get reg k)]
      (if-not (ident? spec)
        spec
        (deep-resolve reg spec)))
    k))

;; Upstream alpha.clj 71-77 (throw via ex-info: same message, still
;; caught by `catch Exception`).
(defn- reg-resolve!
  "returns the spec/regex at end of alias chain starting with k, throws if not found, k if k not ident"
  [k]
  (if (ident? k)
    (c/or (reg-resolve k)
          (throw (ex-info (str "Unable to resolve spec: " k) {})))
    k))

;; Upstream alpha.clj 79-83 (`satisfies?`: the port has no Spec
;; interface class to `instance?` against).
(defn spec?
  "returns x if x is a spec object, else logical false"
  [x]
  (when (satisfies? Spec x)
    x))

;; Upstream alpha.clj 85-88.
(defn regex?
  "returns x if x is a (clojure.spec) regex op, else logical false"
  [x]
  (c/and (:clojure.spec.alpha/op x) x))

;; Upstream alpha.clj 90-96. The port's `with-meta` works on the
;; spec records; narrowed from `IObj` to `spec?` ( every non-ident,
;; non-regex value reaching here is a spec object).
(defn- with-name [spec name]
  (cond
   (ident? spec) spec
   (regex? spec) (assoc spec :clojure.spec.alpha/name name)
   (spec? spec) (with-meta spec (assoc (meta spec) :clojure.spec.alpha/name name))))

;; Upstream alpha.clj 98-108 (same narrowing as `with-name`).
(defn- spec-name [spec]
  (cond
   (ident? spec) spec
   (regex? spec) (:clojure.spec.alpha/name spec)
   (spec? spec) (:clojure.spec.alpha/name (meta spec))))

;; Upstream alpha.clj 110-119 (slice E wires the regex wrap).
(defn- maybe-spec
  "spec-or-k must be a spec, regex or resolvable kw/sym, else returns nil."
  [spec-or-k]
  (let [s (c/or (c/and (ident? spec-or-k) (reg-resolve spec-or-k))
                (spec? spec-or-k)
                (regex? spec-or-k)
                nil)]
    (if (regex? s)
      (with-name (regex-spec-impl s) (spec-name s))
      s)))

;; Upstream alpha.clj 121-126 (throw via ex-info, as `reg-resolve!`).
(defn- the-spec
  "spec-or-k must be a spec, regex or kw/sym, else returns nil. Throws if unresolvable kw/sym"
  [spec-or-k]
  (c/or (maybe-spec spec-or-k)
        (when (ident? spec-or-k)
          (throw (ex-info (str "Unable to resolve spec: " spec-or-k) {})))))

;; Spec coercion. Upstream (alpha.clj 128-160) extends the private
;; Specize protocol to Keyword/Symbol/Set/Object; the port dispatches
;; protocols on exact classes only (no `Object` default, and a sorted
;; set would miss an `IPersistentSet` entry), so this is the same
;; dispatch as a `cond`. `fn-sym` is folded to nil (see the file
;; header), which makes the Object branch unconditional.
(defn- specize
  ([s] (c/or (spec? s)
             (cond (ident? s) (specize (reg-resolve! s))
                   (set? s) (spec-impl s s)
                   :else (spec-impl :clojure.spec.alpha/unknown s))))
  ([s form] (c/or (spec? s)
                  (cond (ident? s) (specize (reg-resolve! s))
                        (set? s) (spec-impl form s)
                        :else (spec-impl form s)))))

;; Upstream alpha.clj 162-165.
(defn invalid?
  "tests the validity of a conform return value"
  [ret]
  (identical? :clojure.spec.alpha/invalid ret))

;; Upstream alpha.clj 167-171.
(defn conform
  "Given a spec and a value, returns :clojure.spec.alpha/invalid
if value does not match spec, else the (possibly destructured) value."
  [spec x]
  (conform* (specize spec) x))

;; Upstream alpha.clj 173-178.
(defn unform
  "Given a spec and a value created by or compliant with a call to
  'conform' with the same spec, returns a value with all conform
  destructuring undone."
  [spec x]
  (unform* (specize spec) x))

;; Upstream alpha.clj 180-184.
(defn form
  "returns the spec as data"
  [spec]
  (describe* (specize spec)))

;; Local postwalk (upstream uses clojure.walk, not ported yet).
(defn- spec-postwalk [f form]
  (let [walked (cond (seq? form) (apply list (map #(spec-postwalk f %) form))
                     (vector? form) (mapv #(spec-postwalk f %) form)
                     (record? form) (into form (map #(spec-postwalk f %) form))
                     (map? form) (into {} (map (fn [[k v]]
                                                 [(spec-postwalk f k) (spec-postwalk f v)])
                                               form))
                     (set? form) (into #{} (map #(spec-postwalk f %) form))
                     :else form)]
    (f walked)))

;; Upstream alpha.clj 186-203.
(defn abbrev [form]
  (cond
   (seq? form)
   (spec-postwalk (fn [form]
                    (cond
                     (c/and (symbol? form) (namespace form))
                     (-> form name symbol)

                     (c/and (seq? form) (= 'fn (first form)) (= '[%] (second form)))
                     (last form)

                     :else form))
                  form)

   (c/and (symbol? form) (namespace form))
   (-> form name symbol)

   :else form))

;; Upstream alpha.clj 205-208.
(defn describe
  "returns an abbreviated description of the spec as data"
  [spec]
  (abbrev (form spec)))

;; Upstream alpha.clj 218-223.
(defn explain-data* [spec path via in x]
  (let [probs (explain* (specize spec) path via in x)]
    (when-not (empty? probs)
      {:clojure.spec.alpha/problems probs
       :clojure.spec.alpha/spec spec
       :clojure.spec.alpha/value x})))

;; Upstream alpha.clj 225-232.
(defn explain-data
  "Given a spec and a value x which ought to conform, returns nil if x
  conforms, else a map with at least the key ::problems whose value is
  a collection of problem-maps, where problem-map has at least :path :pred and :val
  keys describing the predicate and the value that failed at that
  path."
  [spec x]
  (explain-data* spec [] (if-let [name (spec-name spec)] [name] []) [] x))

;; Upstream alpha.clj 234-257.
(defn explain-printer
  "Default printer for explain-data. nil indicates a successful validation."
  [ed]
  (if ed
    (let [problems (->> (:clojure.spec.alpha/problems ed)
                        (sort-by #(- (count (:in %))))
                        (sort-by #(- (count (:path %)))))]
      (doseq [{:keys [path pred val reason via in] :as prob} problems]
        (pr val)
        (print " - failed: ")
        (if reason (print reason) (pr (abbrev pred)))
        (when-not (empty? in)
          (print (str " in: " (pr-str in))))
        (when-not (empty? path)
          (print (str " at: " (pr-str path))))
        (when-not (empty? via)
          (print (str " spec: " (pr-str (last via)))))
        (doseq [[k v] prob]
          (when-not (#{:path :pred :val :reason :via :in} k)
            (print "\n\t" (pr-str k) " ")
            (pr v)))
        (newline)))
    (println "Success!")))

;; Upstream alpha.clj 259.
(def ^:dynamic *explain-out* explain-printer)

;; Upstream alpha.clj 261-265.
(defn explain-out
  "Prints explanation data (per 'explain-data') to *out* using the printer in *explain-out*,
   by default explain-printer."
  [ed]
  (*explain-out* ed))

;; Upstream alpha.clj 267-270.
(defn explain
  "Given a spec and a value that fails to conform, prints an explanation to *out*."
  [spec x]
  (explain-out (explain-data spec x)))

;; Upstream alpha.clj 272-276.
(defn explain-str
  "Given a spec and a value that fails to conform, returns an explanation as a string."
  [spec x]
  (with-out-str (explain spec x)))

;; Upstream alpha.clj 305-310. The port's `symbol` takes no var, so
;; the qualified symbol is rebuilt from the var's `:ns`/`:name` meta.
(defn- ->sym
  "Returns a symbol from a symbol or var"
  [x]
  (if (var? x)
    (let [m (meta x)]
      (symbol (str (ns-name (:ns m))) (name (:name m))))
    x))

;; Upstream alpha.clj 312-318 (postwalk-replace is the local
;; `spec-postwalk` over a one-entry table).
(defn- unfn [expr]
  (if (c/and (seq? expr)
             (symbol? (first expr))
             (= "fn*" (name (first expr))))
    (let [[[s] & form] (rest expr)]
      ;; Explicit `fn`, not `#()`: the reader would rewrite the
      ;; quoted `'%` table value into the arg placeholder.
      (conj (spec-postwalk (fn [x] (get {s '%} x x)) form) '[%] 'fn))
    expr))

;; Upstream alpha.clj 320-325.
(defn- res [form]
  (cond
   (keyword? form) form
   (symbol? form) (c/or (-> form resolve ->sym) form)
   (sequential? form) (spec-postwalk #(if (symbol? %) (res %) %) (unfn form))
   :else form))

;; Upstream alpha.clj 327-337. `assert` is bare (not `c/assert`):
;; the port only treats the unqualified and `clojure.core`-qualified
;; spellings as the special form.
(defn ^:skip-wiki def-impl
  "Do not call this directly, use 'def'"
  [k form spec]
  (assert (c/and (ident? k) (namespace k)) "k must be namespaced keyword or resolvable symbol")
  (if (nil? spec)
    (swap! registry-ref dissoc k)
    (let [spec (if (c/or (spec? spec) (regex? spec) (get @registry-ref spec))
                 spec
                 (spec-impl form spec))]
      (swap! registry-ref assoc k (with-name spec k))))
  k)

;; Upstream alpha.clj 339-345. The port has no `.name` interop and
;; `str` of a namespace renders the object rep, so both name reads go
;; through `ns-name`.
(defn- ns-qualify
  "Qualify symbol s by resolving it or using the current *ns*."
  [s]
  (if-let [ns-sym (some-> s namespace symbol)]
    (c/or (some-> (get (ns-aliases *ns*) ns-sym) ns-name str (symbol (name s)))
          s)
    (symbol (str (ns-name *ns*)) (str s))))

;; Upstream alpha.clj 347-354. Built with `list` (no syntax-quote:
;; see the file header).
(defmacro def
  "Given a namespace-qualified keyword or resolvable symbol k, and a
  spec, spec-name, predicate or regex-op makes an entry in the
  registry mapping k to the spec. Use nil to remove an entry in
  the registry for k."
  [k spec-form]
  (let [k (if (symbol? k) (ns-qualify k) k)]
    (list 'clojure.spec.alpha/def-impl (list 'quote k) (list 'quote (res spec-form)) spec-form)))

;; Upstream alpha.clj 356-359.
(defn registry
  "returns the registry map, prefer 'get-spec' to lookup a spec by name"
  []
  @registry-ref)

;; Upstream alpha.clj 361-364.
(defn get-spec
  "Returns spec registered for keyword/symbol/var k, or nil."
  [k]
  (get (registry) (if (keyword? k) k (->sym k))))

;; Upstream alpha.clj 366-383 (the `(when form …)` guard stays
;; dropped: pre-existing slice-1 narrowing, out of F2 scope).
(defmacro spec
  "Takes a single predicate form, e.g. can be the name of a predicate,
  like even?, or a fn literal like #(< % 42).

  Optionally takes :gen generator-fn, which must be a fn of no args that
  returns a test.check generator.

  Returns a spec."
  [form & {:keys [gen]}]
  (list 'clojure.spec.alpha/spec-impl (list 'quote (res form)) form gen nil))

;; Upstream alpha.clj 493-501 (built with `list`, as `def`).
(defmacro and
  "Takes predicate/spec-forms, e.g.

  (s/and even? #(< % 42))

  Returns a spec that returns the conformed value. Successive
  conformed values propagate through rest of predicates."
  [& pred-forms]
  (list 'clojure.spec.alpha/and-spec-impl
        (list 'quote (mapv res pred-forms))
        (vec pred-forms)))

;; Upstream alpha.clj 607-611 (built with `list`, as `def`).
(defmacro *
  "Returns a regex op that matches zero or more values matching
  pred. Produces a vector of matches iff there is at least one match"
  [pred-form]
  (list 'clojure.spec.alpha/rep-impl (list 'quote (res pred-form)) pred-form))

;; Upstream alpha.clj 613-617 (built with `list`, as `def`).
(defmacro +
  "Returns a regex op that matches one or more values matching
  pred. Produces a vector of matches"
  [pred-form]
  (list 'clojure.spec.alpha/rep+impl (list 'quote (res pred-form)) pred-form))

;; Upstream alpha.clj 619-623 (built with `list`, as `def`).
(defmacro ?
  "Returns a regex op that matches zero or one value matching
  pred. Produces a single value (not a collection) if matched."
  [pred-form]
  (list 'clojure.spec.alpha/maybe-impl pred-form (list 'quote (res pred-form))))

;; Upstream alpha.clj 625-640 (built with `list`, as `def`;
;; `assert` is bare, as in `def-impl`, while the `c/and` inside
;; stays: bare `and` would hit spec's own macro).
(defmacro alt
  "Takes key+pred pairs, e.g.

  (s/alt :even even? :small #(< % 42))

  Returns a regex op that returns a map entry containing the key of the
  first matching pred and the corresponding value. Thus the
  'key' and 'val' functions can be used to refer generically to the
  components of the tagged return"
  [& key-pred-forms]
  (let [pairs (partition 2 key-pred-forms)
        keys (mapv first pairs)
        pred-forms (mapv second pairs)
        pf (mapv res pred-forms)]
    (assert (c/and (even? (count key-pred-forms)) (every? keyword? keys)) "alt expects k1 p1 k2 p2..., where ks are keywords")
    (list 'clojure.spec.alpha/alt-impl keys pred-forms (list 'quote pf))))

;; Upstream alpha.clj 642-656 (built with `list`, as `alt`).
(defmacro cat
  "Takes key+pred pairs, e.g.

  (s/cat :e even? :o odd?)

  Returns a regex op that matches (all) values in sequence, returning a map
  containing the keys of each pred and the corresponding value."
  [& key-pred-forms]
  (let [pairs (partition 2 key-pred-forms)
        keys (mapv first pairs)
        pred-forms (mapv second pairs)
        pf (mapv res pred-forms)]
    (assert (c/and (even? (count key-pred-forms)) (every? keyword? keys)) "cat expects k1 p1 k2 p2..., where ks are keywords")
    (list 'clojure.spec.alpha/cat-impl keys pred-forms (list 'quote pf))))

;; Upstream alpha.clj 658-665 (built with `list`, as `def`).
(defmacro &
  "takes a regex op re, and predicates. Returns a regex-op that consumes
  input as per re but subjects the resulting value to the
  conjunction of the predicates, and any conforming they might perform."
  [re & preds]
  (let [pv (vec preds)]
    (list 'clojure.spec.alpha/amp-impl re (list 'quote (res re)) pv (list 'quote (mapv res pv)))))

;; Upstream alpha.clj 414-474. Built with `list` (no
;; syntax-quote: see the file header), `walk/postwalk` is the
;; local `spec-postwalk`, and the `c/assert` is bare `assert`.
(defmacro keys
  "Creates and returns a map validating spec. :req and :opt are both
  vectors of namespaced-qualified keywords. The validator will ensure
  the :req keys are present. The :opt keys serve as documentation and
  may be used by the generator.

  The :req key vector supports 'and' and 'or' for key groups:

  (s/keys :req [::x ::y (or ::secret (and ::user ::pwd))] :opt [::z])

  There are also -un versions of :req and :opt. These allow
  you to connect unqualified keys to specs.  In each case, fully
  qualified keywords are passed, which name the specs, but unqualified
  keys (with the same name component) are expected and checked at
  conform-time, and generated during gen:

  (s/keys :req-un [:my.ns/x :my.ns/y])

  The above says keys :x and :y are required, and will be validated
  and generated by specs (if they exist) named :my.ns/x :my.ns/y
  respectively.

  In addition, the values of *all* namespace-qualified keys will be validated
  (and possibly destructured) by any registered specs. Note: there is
  no support for inline value specification, by design.

  Optionally takes :gen generator-fn, which must be a fn of no args that
  returns a test.check generator."
  [& {:keys [req req-un opt opt-un gen]}]
  (let [unk (fn [k] (keyword (name k)))
        req-keys (filterv keyword? (flatten req))
        req-un-specs (filterv keyword? (flatten req-un))
        _ (assert (every? (fn [k] (c/and (keyword? k) (namespace k)))
                          (concat req-keys req-un-specs opt opt-un))
                  "all keys must be namespace-qualified keywords")
        req-specs (into req-keys req-un-specs)
        req-keys (into req-keys (map unk req-un-specs))
        opt-keys (into (vec opt) (map unk opt-un))
        opt-specs (into (vec opt) opt-un)
        gx (gensym)
        parse-req (fn [rk f]
                    (map (fn [x]
                           (if (keyword? x)
                             (list 'clojure.core/contains? gx (f x))
                             (spec-postwalk
                              (fn [y]
                                (if (keyword? y)
                                  (list 'clojure.core/contains? gx (f y))
                                  y))
                              x)))
                         rk))
        pred-exprs (into [(list 'clojure.core/map? gx)]
                         (parse-req req identity))
        pred-exprs (into pred-exprs (parse-req req-un unk))
        keys-pred (list 'fn* [gx] (apply list 'clojure.core/and pred-exprs))
        pred-exprs (mapv (fn [e] (list 'fn* [gx] e)) pred-exprs)
        pred-forms (spec-postwalk res pred-exprs)]
    (list 'clojure.spec.alpha/map-spec-impl
          {:req (list 'quote req)
           :opt (list 'quote opt)
           :req-un (list 'quote req-un)
           :opt-un (list 'quote opt-un)
           :req-keys (list 'quote req-keys)
           :req-specs (list 'quote req-specs)
           :opt-keys (list 'quote opt-keys)
           :opt-specs (list 'quote opt-specs)
           :pred-forms (list 'quote pred-forms)
           :pred-exprs pred-exprs
           :keys-pred keys-pred
           :gfn gen})))

;; Upstream alpha.clj 503-509 (passes nil `gfn`, as upstream).
;; Built with `list`, as `and`.
(defmacro merge
  "Takes map-validating specs (e.g. 'keys' specs) and
  returns a spec that returns a conformed map satisfying all of the
  specs.  Unlike 'and', merge can generate maps satisfying the
  union of the predicates."
  [& pred-forms]
  (list 'clojure.spec.alpha/merge-spec-impl
        (list 'quote (mapv res pred-forms))
        (vec pred-forms)))

;; Upstream alpha.clj 757-768 (slice D2 adds the `cpred?` arity:
;; conformer preds return their result as-is, only `::invalid`
;; fails). Throw via ex-info, as `reg-resolve!`.
(defn- dt
  ([pred x form] (dt pred x form nil))
  ([pred x form cpred?]
   (if pred
     (if-let [spec (the-spec pred)]
       (conform spec x)
       (if (ifn? pred)
         (if cpred?
           (pred x)
           (if (pred x) x :clojure.spec.alpha/invalid))
         (throw (ex-info (str (pr-str form) " is not a fn, expected predicate fn") {}))))
     x)))

;; Upstream alpha.clj 770-777.
(defn valid?
  "Helper function that returns true when x is valid for spec."
  ([spec x]
   (let [spec (specize spec)]
     (not (invalid? (conform* spec x)))))
  ([spec x form]
   (let [spec (specize spec form)]
     (not (invalid? (conform* spec x))))))

;; Upstream alpha.clj 779-784. Shared by slices A (`keys`/`merge`)
;; and D2 (multi-spec/tuple/conformer paths); `dt` carries the
;; `cpred?` arity. `::unknown` is written out (no auto-resolve,
;; per the file header).
(defn- pvalid?
  "internal helper function that returns true when x is valid for spec."
  ([pred x]
   (not (invalid? (dt pred x :clojure.spec.alpha/unknown))))
  ([pred x form]
   (not (invalid? (dt pred x form)))))

;; Upstream alpha.clj 786-793.
(defn- explain-1 [form pred path via in v]
  (let [pred (maybe-spec pred)]
    (if (spec? pred)
      (explain* pred path (if-let [name (spec-name pred)] (conj via name) via) in v)
      [{:path path :pred form :val v :via via :in in}])))

;; ---- gap-122 slice F2: generation ----

;; Upstream alpha.clj 26-30.
(def ^:dynamic *recursion-limit*
  "A soft limit on how many times a branching spec (or/alt/*/opt-keys/multi-spec)
  can be recursed through during generation. After this a
  non-recursive branch will be chosen."
  4)

;; Upstream alpha.clj 750-755 (`::recursion-limit` written out, per
;; the file header).
(defn- recur-limit? [rmap id path k]
  (c/and (> (get rmap id) (:clojure.spec.alpha/recursion-limit rmap))
         (contains? (set path) k)))

(defn- inck [m k]
  (assoc m k (inc (c/or (get m k) 0))))

;; Upstream alpha.clj 210-216 (`::gfn` written out).
(defn with-gen
  "Takes a spec and a no-arg, generator-returning fn and returns a version of that spec that uses that generator"
  [spec gen-fn]
  (let [spec (reg-resolve spec)]
    (if (regex? spec)
      (assoc spec :clojure.spec.alpha/gfn gen-fn)
      (with-gen* (specize spec) gen-fn))))

;; Upstream alpha.clj 279-290 (throw via ex-info, as `reg-resolve!`).
(defn- gensub
  [spec overrides path rmap form]
  (let [spec (specize spec)]
    (if-let [g (c/or (when-let [gfn (c/or (get overrides (c/or (spec-name spec) spec))
                                          (get overrides path))]
                       (gfn))
                     (gen* spec overrides path rmap))]
      (gen/such-that #(valid? spec %) g 100)
      (let [abbr (abbrev form)]
        (throw (ex-info (str "Unable to construct gen at: " path " for: " abbr)
                        {:clojure.spec.alpha/path path
                         :clojure.spec.alpha/form form
                         :clojure.spec.alpha/failure :no-gen}))))))

;; Upstream alpha.clj 292-303.
(defn gen
  "Given a spec, returns the generator for it, or throws if none can
  be constructed. Optionally an overrides map can be provided which
  should map spec names or paths (vectors of keywords) to no-arg
  generator-creating fns. These will be used instead of the generators at those
  names/paths. Note that parent generator (in the spec or overrides
  map) will supersede those of any subtrees. A generator for a regex
  op must always return a sequential collection (i.e. a generator for
  s/? should return either an empty sequence/vector or a
  sequence/vector with one item in it)"
  ([spec] (gen spec nil))
  ([spec overrides] (gensub spec overrides [] {:clojure.spec.alpha/recursion-limit *recursion-limit*} spec)))

;; Upstream alpha.clj 1866-1872.
(defn exercise
  "generates a number (default 10) of values compatible with spec and maps conform over them,
  returning a sequence of [val conformed-val] tuples. Optionally takes
  a generator overrides map as per gen"
  ([spec] (exercise spec 10))
  ([spec n] (exercise spec n nil))
  ([spec n overrides]
   (map #(vector % (conform spec %)) (gen/sample (gen spec overrides) n))))

;; Upstream alpha.clj 1874-1888 (throw via ex-info, as
;; `reg-resolve!`; full fspec use waits for slice G, but the fn
;; ports verbatim and works over `{:args …}` maps).
(defn exercise-fn
  "exercises the fn named by sym (a symbol) by applying it to
  n (default 10) generated samples of its args spec. When fspec is
  supplied its arg spec is used, and sym-or-f can be a fn.  Returns a
  sequence of tuples of [args ret]. "
  ([sym] (exercise-fn sym 10))
  ([sym n] (exercise-fn sym n (get-spec sym)))
  ([sym-or-f n fspec]
   (let [f (if (symbol? sym-or-f) (resolve sym-or-f) sym-or-f)]
     (if-let [arg-spec (c/and fspec (:args fspec))]
       (for [args (gen/sample (gen arg-spec) n)]
         [args (apply f args)])
       (throw (ex-info "No :args spec found, can't generate" {}))))))

;; Map key generators (upstream alpha.clj 795-819; `or-k-gen`/
;; `and-k-gen` are declared at the top, as upstream's line-793
;; declare).
(defn- k-gen
  "returns a generator for form f, which can be a keyword or a list
  starting with 'or or 'and."
  [f]
  (cond
   (keyword? f) (gen/return f)
   (= 'or (first f)) (or-k-gen 1 (rest f))
   (= 'and (first f)) (and-k-gen (rest f))))

(defn- or-k-gen
  "returns a tuple generator made up of generators for a random subset
  of min-count (default 0) to all elements in s."
  ([s] (or-k-gen 0 s))
  ([min-count s]
   (gen/bind (gen/tuple
              (gen/choose min-count (count s))
              (gen/shuffle (map k-gen s)))
             (fn [[n gens]]
               (apply gen/tuple (take n gens))))))

(defn- and-k-gen
  "returns a tuple generator made up of generators for every element
  in s."
  [s]
  (apply gen/tuple (map k-gen s)))

;; Upstream alpha.clj 1140-1150.
(defn- explain-pred-list
  [forms preds path via in x]
  (loop [ret x
         [form & forms] forms
         [pred & preds] preds]
    (when pred
      (let [nret (dt pred ret form)]
        (if (invalid? nret)
          (explain-1 form pred path via in ret)
          (recur nret forms preds))))))

;; Upstream alpha.clj 913-944. The conformer arities are
;; upstream's `[form pred gfn cpred?]` / `[form pred gfn cpred?
;; unc]`; slice 1's `[form pred]` convenience (nil `gfn`) stays for
;; `specize`/`def-impl`. Builds a PredSpec record; a regex pred
;; wraps via `regex-spec-impl` (slice E).
(defn ^:skip-wiki spec-impl
  "Do not call this directly, use 'spec'"
  ([form pred] (spec-impl form pred nil nil nil))
  ([form pred gfn cpred?] (spec-impl form pred gfn cpred? nil))
  ([form pred gfn cpred? unc]
   (cond
    (spec? pred) (cond-> pred gfn (with-gen gfn))
    (regex? pred) (regex-spec-impl pred gfn)
    (ident? pred) (cond-> (the-spec pred) gfn (with-gen gfn))
    :else (->PredSpec form pred cpred? unc gfn))))

;; Upstream alpha.clj 1151-1193. Builds an AndSpec record; the
;; per-count fast paths are one loop (behavior-identical). The
;; `[forms preds]` arity (nil `gfn`) stays for the `and` macro.
(defn ^:skip-wiki and-spec-impl
  "Do not call this directly, use 'and'"
  ([forms preds] (and-spec-impl forms preds nil))
  ([forms preds gfn]
   (->AndSpec forms preds (delay (mapv specize preds forms)) gfn)))

;; Upstream alpha.clj 1058-1059. The port cannot construct a
;; `MapEntry` object, so this takes the first entry of a
;; single-entry map (a port `MapEntry`: identical `pr`, `key`,
;; `val`, `nth`, `count`, `=`, `vector?`, destructuring).
(defn- tagged-ret [tag ret]
  (first {tag ret}))

;; Upstream alpha.clj 1128-1137.
(defn- and-preds [x preds forms]
  (loop [ret x
         [pred & preds] preds
         [form & forms] forms]
    (if pred
      (let [nret (dt pred ret form)]
        (if (invalid? nret)
          :clojure.spec.alpha/invalid
          ;;propagate conformed values
          (recur nret preds forms)))
      ret)))

;;;;;;;;;;;;;;;;;;;;;;; regex ;;;;;;;;;;;;;;;;;;;
;;See:
;; http://matt.might.net/articles/implementation-of-regular-expression-matching-in-scheme-with-derivatives/
;; http://www.ccs.neu.edu/home/turon/re-deriv.pdf

;; Upstream alpha.clj 1380-1383.
(defn- accept [x] {:clojure.spec.alpha/op :clojure.spec.alpha/accept :ret x})

(defn- accept? [{:keys [:clojure.spec.alpha/op]}]
  (= :clojure.spec.alpha/accept op))

;; Upstream alpha.clj 1385-1395.
(defn- pcat* [{[p1 & pr :as ps] :ps,  [k1 & kr :as ks] :ks, [f1 & fr :as forms] :forms, ret :ret, rep+ :rep+}]
  (when (every? identity ps)
    (if (accept? p1)
      (let [rp (:ret p1)
            ret (conj ret (if ks {k1 rp} rp))]
        (if pr
          (pcat* {:ps pr :ks kr :forms fr :ret ret})
          (accept ret)))
      {:clojure.spec.alpha/op :clojure.spec.alpha/pcat, :ps ps, :ret ret, :ks ks, :forms forms :rep+ rep+})))

(defn- pcat [& ps] (pcat* {:ps ps :ret []}))

;; Upstream alpha.clj 1397-1400.
(defn ^:skip-wiki cat-impl
  "Do not call this directly, use 'cat'"
  [ks ps forms]
  (pcat* {:ks ks, :ps ps, :forms forms, :ret {}}))

;; Upstream alpha.clj 1402-1407 (`random-uuid` for
;; `UUID/randomUUID`: same value type, port native).
(defn- rep* [p1 p2 ret splice form]
  (when p1
    (let [r {:clojure.spec.alpha/op :clojure.spec.alpha/rep, :p2 p2, :splice splice, :forms form :id (random-uuid)}]
      (if (accept? p1)
        (assoc r :p1 p2 :ret (conj ret (:ret p1)))
        (assoc r :p1 p1, :ret ret)))))

;; Upstream alpha.clj 1409-1411.
(defn ^:skip-wiki rep-impl
  "Do not call this directly, use '*'"
  [form p] (rep* p p [] false form))

;; Upstream alpha.clj 1413-1416 (the `` `[~form (* ~form)] `` form
;; vector is list-built, as the macros).
(defn ^:skip-wiki rep+impl
  "Do not call this directly, use '+'"
  [form p]
  (pcat* {:ps [p (rep* p p [] true form)] :forms (vector form (list 'clojure.spec.alpha/* form)) :ret [] :rep+ form}))

;; Upstream alpha.clj 1418-1421.
(defn ^:skip-wiki amp-impl
  "Do not call this directly, use '&'"
  [re re-form preds pred-forms]
  {:clojure.spec.alpha/op :clojure.spec.alpha/amp :p1 re :amp re-form :ps preds :forms pred-forms})

;; Upstream alpha.clj 1423-1430.
(defn- filter-alt [ps ks forms f]
  (if (c/or ks forms)
    (let [pks (->> (map vector ps
                        (c/or (seq ks) (repeat nil))
                        (c/or (seq forms) (repeat nil)))
                   (filter #(-> % first f)))]
      [(seq (map first pks)) (when ks (seq (map second pks))) (when forms (seq (map #(nth % 2) pks)))])
    [(seq (filter f ps)) ks forms]))

;; Upstream alpha.clj 1432-1442.
(defn- alt* [ps ks forms]
  (let [[[p1 & pr :as ps] [k1 :as ks] forms] (filter-alt ps ks forms identity)]
    (when ps
      (let [ret {:clojure.spec.alpha/op :clojure.spec.alpha/alt, :ps ps, :ks ks :forms forms}]
        (if (nil? pr)
          (if k1
            (if (accept? p1)
              (accept (tagged-ret k1 (:ret p1)))
              ret)
            p1)
          ret)))))

(defn- alts [& ps] (alt* ps nil nil))
(defn- alt2 [p1 p2] (if (c/and p1 p2) (alts p1 p2) (c/or p1 p2)))

;; Upstream alpha.clj 1447-1449 (`random-uuid`, as `rep*`).
(defn ^:skip-wiki alt-impl
  "Do not call this directly, use 'alt'"
  [ks ps forms] (assoc (alt* ps ks forms) :id (random-uuid)))

;; Upstream alpha.clj 1451-1453.
(defn ^:skip-wiki maybe-impl
  "Do not call this directly, use '?'"
  [p form] (assoc (alt* [p (accept :clojure.spec.alpha/nil)] nil [form :clojure.spec.alpha/nil]) :maybe form))

;; Upstream alpha.clj 1455-1459.
(defn- noret? [p1 pret]
  (c/or (= pret :clojure.spec.alpha/nil)
        (c/and (#{:clojure.spec.alpha/rep :clojure.spec.alpha/pcat} (:clojure.spec.alpha/op (reg-resolve! p1))) ;;hrm, shouldn't know these
               (empty? pret))
        nil))

(declare preturn)

;; Upstream alpha.clj 1463-1473. The `(next forms)` shift in the
;; `amp` branch is a verbatim upstream quirk (`preturn` below and
;; `op-explain` use the unshifted `forms`); it only affects the
;; `dt` "not a fn" message path.
(defn- accept-nil? [p]
  (let [{:keys [:clojure.spec.alpha/op ps p1 p2 forms] :as p} (reg-resolve! p)]
    (case op
          :clojure.spec.alpha/accept true
          nil nil
          :clojure.spec.alpha/amp (c/and (accept-nil? p1)
                       (let [ret (-> (preturn p1) (and-preds ps (next forms)))]
                         (not (invalid? ret))))
          :clojure.spec.alpha/rep (c/or (identical? p1 p2) (accept-nil? p1))
          :clojure.spec.alpha/pcat (every? accept-nil? ps)
          :clojure.spec.alpha/alt (c/some accept-nil? ps))))

(declare add-ret)

;; Upstream alpha.clj 1477-1490.
(defn- preturn [p]
  (let [{[p0 & pr :as ps] :ps, [k :as ks] :ks, :keys [:clojure.spec.alpha/op p1 ret forms] :as p} (reg-resolve! p)]
    (case op
          :clojure.spec.alpha/accept ret
          nil nil
          :clojure.spec.alpha/amp (let [pret (preturn p1)]
                  (if (noret? p1 pret)
                    :clojure.spec.alpha/nil
                    (and-preds pret ps forms)))
          :clojure.spec.alpha/rep (add-ret p1 ret k)
          :clojure.spec.alpha/pcat (add-ret p0 ret k)
          :clojure.spec.alpha/alt (let [[[p0] [k0]] (filter-alt ps ks forms accept-nil?)
                      r (if (nil? p0) :clojure.spec.alpha/nil (preturn p0))]
                  (if k0 (tagged-ret k0 r) r)))))

;; Upstream alpha.clj 1492-1511.
(defn- op-unform [p x]
  ;;(prn {:p p :x x})
  (let [{[p0 & pr :as ps] :ps, [k :as ks] :ks, :keys [:clojure.spec.alpha/op p1 ret forms rep+ maybe] :as p} (reg-resolve! p)
        kps (zipmap ks ps)]
    (case op
          :clojure.spec.alpha/accept [ret]
          nil [(unform p x)]
          :clojure.spec.alpha/amp (let [px (reduce #(unform %2 %1) x (reverse ps))]
                  (op-unform p1 px))
          :clojure.spec.alpha/rep (mapcat #(op-unform p1 %) x)
          :clojure.spec.alpha/pcat (if rep+
                   (mapcat #(op-unform p0 %) x)
                   (mapcat (fn [k]
                             (when (contains? x k)
                               (op-unform (kps k) (get x k))))
                           ks))
          :clojure.spec.alpha/alt (if maybe
                  [(unform p0 x)]
                  (let [[k v] x]
                    (op-unform (kps k) v))))))

;; Upstream alpha.clj 1513-1524.
(defn- add-ret [p r k]
  (let [{:keys [:clojure.spec.alpha/op ps splice] :as p} (reg-resolve! p)
        prop #(let [ret (preturn p)]
                (if (empty? ret) r ((if splice into conj) r (if k {k ret} ret))))]
    (case op
          nil r
          (:clojure.spec.alpha/alt :clojure.spec.alpha/accept :clojure.spec.alpha/amp)
          (let [ret (preturn p)]
            ;;(prn {:ret ret})
            (if (= ret :clojure.spec.alpha/nil) r (conj r (if k {k ret} ret))))

          (:clojure.spec.alpha/rep :clojure.spec.alpha/pcat) (prop))))

;; Upstream alpha.clj 1526-1544 (`(next forms)` in the `amp`
;; branch, as in `accept-nil?`).
(defn- deriv
  [p x]
  (let [{[p0 & pr :as ps] :ps, [k0 & kr :as ks] :ks, :keys [:clojure.spec.alpha/op p1 p2 ret splice forms amp] :as p} (reg-resolve! p)]
    (when p
      (case op
            :clojure.spec.alpha/accept nil
            nil (let [ret (dt p x p)]
                  (when-not (invalid? ret) (accept ret)))
            :clojure.spec.alpha/amp (when-let [p1 (deriv p1 x)]
                    (if (= :clojure.spec.alpha/accept (:clojure.spec.alpha/op p1))
                      (let [ret (-> (preturn p1) (and-preds ps (next forms)))]
                        (when-not (invalid? ret)
                          (accept ret)))
                      (amp-impl p1 amp ps forms)))
            :clojure.spec.alpha/pcat (alt2 (pcat* {:ps (cons (deriv p0 x) pr), :ks ks, :forms forms, :ret ret})
                         (when (accept-nil? p0) (deriv (pcat* {:ps pr, :ks kr, :forms (next forms), :ret (add-ret p0 ret k0)}) x)))
            :clojure.spec.alpha/alt (alt* (map #(deriv % x) ps) ks forms)
            :clojure.spec.alpha/rep (alt2 (rep* (deriv p1 x) p2 ret splice forms)
                        (when (accept-nil? p1) (deriv (rep* p2 p2 (add-ret p1 ret nil) splice forms) x)))))))

;; Upstream alpha.clj 1546-1560 (syntax-quoted op heads are
;; literal fully-qualified symbols, as the macros).
(defn- op-describe [p]
  (let [{:keys [:clojure.spec.alpha/op ps ks forms splice p1 rep+ maybe amp] :as p} (reg-resolve! p)]
    ;;(prn {:op op :ks ks :forms forms :p p})
    (when p
      (case op
            :clojure.spec.alpha/accept nil
            nil p
            :clojure.spec.alpha/amp (list* 'clojure.spec.alpha/& amp forms)
            :clojure.spec.alpha/pcat (if rep+
                     (list 'clojure.spec.alpha/+ rep+)
                     (cons 'clojure.spec.alpha/cat (mapcat vector (c/or (seq ks) (repeat :_)) forms)))
            :clojure.spec.alpha/alt (if maybe
                    (list 'clojure.spec.alpha/? maybe)
                    (cons 'clojure.spec.alpha/alt (mapcat vector ks forms)))
            :clojure.spec.alpha/rep (list (if splice 'clojure.spec.alpha/+ 'clojure.spec.alpha/*) forms)))))

;; Upstream alpha.clj 1562-1615.
(defn- op-explain [form p path via in input]
  ;;(prn {:form form :p p :path path :input input})
  (let [[x :as input] input
        {:keys [:clojure.spec.alpha/op ps ks forms splice p1 p2] :as p} (reg-resolve! p)
        via (if-let [name (spec-name p)] (conj via name) via)
        insufficient (fn [path form]
                       [{:path path
                         :reason "Insufficient input"
                         :pred form
                         :val ()
                         :via via
                         :in in}])]
    (when p
      (case op
            :clojure.spec.alpha/accept nil
            nil (if (empty? input)
                  (insufficient path form)
                  (explain-1 form p path via in x))
            :clojure.spec.alpha/amp (if (empty? input)
                    (if (accept-nil? p1)
                      (explain-pred-list forms ps path via in (preturn p1))
                      (insufficient path (:amp p)))
                    (if-let [p1 (deriv p1 x)]
                      (explain-pred-list forms ps path via in (preturn p1))
                      (op-explain (:amp p) p1 path via in input)))
            :clojure.spec.alpha/pcat (let [pkfs (map vector
                                   ps
                                   (c/or (seq ks) (repeat nil))
                                   (c/or (seq forms) (repeat nil)))
                         [pred k form] (if (= 1 (count pkfs))
                                         (first pkfs)
                                         (first (remove (fn [[p]] (accept-nil? p)) pkfs)))
                         path (if k (conj path k) path)
                         form (c/or form (op-describe pred))]
                     (if (c/and (empty? input) (not pred))
                       (insufficient path form)
                       (op-explain form pred path via in input)))
            :clojure.spec.alpha/alt (if (empty? input)
                    (insufficient path (op-describe p))
                    (apply concat
                           (map (fn [k form pred]
                                  (op-explain (c/or form (op-describe pred))
                                              pred
                                              (if k (conj path k) path)
                                              via
                                              in
                                              input))
                                (c/or (seq ks) (repeat nil))
                                (c/or (seq forms) (repeat nil))
                                ps)))
            :clojure.spec.alpha/rep (op-explain (if (identical? p1 p2)
                                forms
                                (op-describe p1))
                              p1 path via in input)))))

;; Upstream alpha.clj 1617-1655 (`::op`/`::gfn` written out, per
;; the file header).
(defn- re-gen [p overrides path rmap f]
  (let [origp p
        {:keys [:clojure.spec.alpha/op :ps :ks :p1 :p2 :forms :splice :ret :id :clojure.spec.alpha/gfn] :as p} (reg-resolve! p)
        rmap (if id (inck rmap id) rmap)
        ggens (fn [ps ks forms]
                (let [gen (fn [p k f]
                            (when-not (c/and rmap id k (recur-limit? rmap id path k))
                              (if id
                                (gen/delay (re-gen p overrides (if k (conj path k) path) rmap (c/or f p)))
                                (re-gen p overrides (if k (conj path k) path) rmap (c/or f p)))))]
                  (map gen ps (c/or (seq ks) (repeat nil)) (c/or (seq forms) (repeat nil)))))]
    (c/or (when-let [gfn (c/or (get overrides (spec-name origp))
                               (get overrides (spec-name p))
                               (get overrides path))]
            (case op
              (:clojure.spec.alpha/accept nil) (gen/fmap vector (gfn))
              (gfn)))
          (when gfn
            (gfn))
          (when p
            (case op
              :clojure.spec.alpha/accept (if (= ret :clojure.spec.alpha/nil)
                                           (gen/return [])
                                           (gen/return [ret]))
              nil (when-let [g (gensub p overrides path rmap f)]
                    (gen/fmap vector g))
              :clojure.spec.alpha/amp (re-gen p1 overrides path rmap (op-describe p1))
              :clojure.spec.alpha/pcat (let [gens (ggens ps ks forms)]
                                         (when (every? identity gens)
                                           (apply gen/cat gens)))
              :clojure.spec.alpha/alt (let [gens (remove nil? (ggens ps ks forms))]
                                        (when-not (empty? gens)
                                          (gen/one-of gens)))
              :clojure.spec.alpha/rep (if (recur-limit? rmap id [id] id)
                                        (gen/return [])
                                        (when-let [g (re-gen p2 overrides path rmap forms)]
                                          (gen/fmap #(apply concat %)
                                                    (gen/vector g)))))))))

;; Upstream alpha.clj 1658-1669.
(defn- re-conform [p [x & xs :as data]]
  ;;(prn {:p p :x x :xs xs})
  (if (empty? data)
    (if (accept-nil? p)
      (let [ret (preturn p)]
        (if (= ret :clojure.spec.alpha/nil)
          nil
          ret))
      :clojure.spec.alpha/invalid)
    (if-let [dp (deriv p x)]
      (recur dp xs)
      :clojure.spec.alpha/invalid)))

;; Upstream alpha.clj 1671-1695.
(defn- re-explain [path via in re input]
  (loop [p re [x & xs :as data] input i 0]
    ;;(prn {:p p :x x :xs xs :re re}) (prn)
    (if (empty? data)
      (if (accept-nil? p)
        nil ;;success
        (op-explain (op-describe p) p path via in nil))
      (if-let [dp (deriv p x)]
        (recur dp xs (inc i))
        (if (accept? p)
          (if (= (:clojure.spec.alpha/op p) :clojure.spec.alpha/pcat)
            (op-explain (op-describe p) p path via (conj in i) (seq data))
            [{:path path
              :reason "Extra input"
              :pred (op-describe re)
              :val data
              :via via
              :in (conj in i)}])
          (c/or (op-explain (op-describe p) p path via (conj in i) (seq data))
                [{:path path
                  :reason "Extra input"
                  :pred (op-describe p)
                  :val data
                  :via via
                  :in (conj in i)}]))))))

;; The deterministic value of upstream's `(res `#(c/or (nil? %)
;; (sequential? %)))` (regex-spec-impl explain*, alpha.clj 1717):
;; syntax-quote qualifies, `unfn` restores `(fn [%] ...)`, `res`
;; re-resolves the symbols via core refers. Abbreviates to
;; `(or (nil? %) (sequential? %))`.
(def ^:private regex-guard-form
  '(clojure.core/fn [%] (clojure.core/or (clojure.core/nil? %) (clojure.core/sequential? %))))

;; Upstream alpha.clj 1697-1723. Builds a RegexSpec record
;; (slice-1 rule 2); the `[re]` arity (nil `gfn`) stays for
;; `maybe-spec`. `gen*` without `gfn` reaches the `re-gen` stub
;; until the F2 regex chunk lands it.
(defn ^:skip-wiki regex-spec-impl
  "Do not call this directly, use 'spec' with a regex op argument"
  ([re] (regex-spec-impl re nil))
  ([re gfn]
   (->RegexSpec re gfn)))

;; Upstream alpha.clj 1794-1812 (built with `list`, as `def`;
;; `let`/`fn`/core fns stay bare for use-site resolution while
;; spec/gen heads fully qualify). Bare `gen` in the upstream
;; expansion syntax-quotes to `clojure.spec.alpha/gen` (s/gen),
;; not the gen alias. Slice F2 also registers `::kvs->map`
;; (below the `conformer` macro) so runtime use works.
(defmacro keys*
  "takes the same arguments as spec/keys and returns a regex op that matches sequences of key/values,
  converts them into a map, and conforms that map with a corresponding
  spec/keys call:

  user=> (s/conform (s/keys :req-un [::a ::c]) {:a 1 :c 2})
  {:a 1, :c 2}
  user=> (s/conform (s/keys* :req-un [::a ::c]) [:a 1 :c 2])
  {:a 1, :c 2}

  the resulting regex op can be composed into a larger regex:

  user=> (s/conform (s/cat :i1 integer? :m (s/keys* :req-un [::a ::c]) :i2 integer?) [42 :a 1 :c 2 :d 4 99])
  {:i1 42, :m {:a 1, :c 2, :d 4}, :i2 99}"
  [& kspecs]
  (let [mspec (gensym "mspec")
        m (gensym "m")]
    (list 'let [mspec (apply list 'clojure.spec.alpha/keys kspecs)]
          (list 'clojure.spec.alpha/with-gen
                (list 'clojure.spec.alpha/&
                      (list 'clojure.spec.alpha/*
                            (list 'clojure.spec.alpha/cat :clojure.spec.alpha/k 'keyword? :clojure.spec.alpha/v 'any?))
                      :clojure.spec.alpha/kvs->map
                      mspec)
                (list 'fn []
                      (list 'clojure.spec.gen.alpha/fmap
                            (list 'fn [m] (list 'apply 'concat m))
                            (list 'clojure.spec.alpha/gen mspec)))))))

;; Upstream alpha.clj 822-908. Builds a MapSpec record (see the
;; file header); `k->s` is precomputed here, standing in for the
;; `keys->specnames` closure. The `id` is a `random-uuid` minted
;; per construction (as upstream).
(defn ^:skip-wiki map-spec-impl
  "Do not call this directly, use 'keys'"
  [{:keys [req-un opt-un keys-pred pred-exprs opt-keys req-specs req req-keys opt-specs pred-forms opt gfn]}]
  (->MapSpec req-un opt-un keys-pred pred-exprs opt-keys req-specs req req-keys opt-specs pred-forms opt
             gfn (random-uuid)
             (zipmap (concat req-keys opt-keys) (concat req-specs opt-specs))))

;; Upstream alpha.clj 1195-1221. Builds a MergeSpec record. The
;; `[forms preds]` arity (nil `gfn`) stays for the `merge` macro.
(defn ^:skip-wiki merge-spec-impl
  "Do not call this directly, use 'merge'"
  ([forms preds] (merge-spec-impl forms preds nil))
  ([forms preds gfn]
   (->MergeSpec forms preds gfn)))

;; ---- gap-122 slice D2: tuple/multi-spec/conformer/nonconforming ----

;; Upstream alpha.clj 666-672, built with `list` (no syntax-quote:
;; see the file header). The expansion's `true` is `cpred?`; the
;; one-arg shape has no `unc`, so unforming it throws (upstream
;; alpha.clj 935).
(defmacro conformer
  "takes a predicate function with the semantics of conform i.e. it should return either a
  (possibly converted) value or :clojure.spec.alpha/invalid, and returns a
  spec that uses it as a predicate/conformer. Optionally takes a
  second fn that does unform of result of first"
  ([f] (list 'clojure.spec.alpha/spec-impl
             (list 'quote (list 'conformer (res f)))
             f nil true))
  ([f unf] (list 'clojure.spec.alpha/spec-impl
                 (list 'quote (list 'conformer (res f) (res unf)))
                 f nil true unf)))

;; Upstream alpha.clj 1793 (`::k`/`::v`/`::kvs->map` written out).
;; Registered after the `conformer` macro so `keys*` runtime use
;; works (slice F2 unblocks it).
(clojure.spec.alpha/def :clojure.spec.alpha/kvs->map
  (conformer #(zipmap (map :clojure.spec.alpha/k %) (map :clojure.spec.alpha/v %))
             #(map (fn [[k v]] {:clojure.spec.alpha/k k :clojure.spec.alpha/v v}) %)))

;; Upstream alpha.clj 694-700, built with `list` (as `def`).
;; `assert` is bare (not `c/assert`), as in `def-impl`.
(defmacro tuple
  "takes one or more preds and returns a spec for a tuple, a vector
  where each element conforms to the corresponding pred. Each element
  will be referred to in paths using its ordinal."
  [& preds]
  (assert (not (empty? preds)))
  (list 'clojure.spec.alpha/tuple-impl
        (list 'quote (mapv res preds))
        (vec preds)))

;; Upstream alpha.clj 385-412, built with `list` (as `def`).
;; `retag` is evaluated (a keyword or fn value, carried into
;; `describe`; only `gen*` consumes it, and generators are a later
;; slice).
(defmacro multi-spec
  "Takes the name of a spec/predicate-returning multimethod and a
  tag-restoring keyword or fn (retag).  Returns a spec that when
  conforming or explaining data will pass it to the multimethod to get
  an appropriate spec. You can e.g. use multi-spec to dynamically and
  extensibly associate specs with 'tagged' data (i.e. data where one
  of the fields indicates the shape of the rest of the structure).

  (defmulti mspec :tag)

  The methods should ignore their argument and return a predicate/spec:
  (defmethod mspec :int [_] (s/keys :req-un [::tag ::i]))

  retag is used during generation to retag generated values with
  matching tags. retag can either be a keyword, at which key the
  dispatch-tag will be assoc'ed, or a fn of generated value and
  dispatch-tag that should return an appropriately retagged value.

  Note that because the tags themselves comprise an open set,
  the tag key spec cannot enumerate the values, but can e.g.
  test for keyword?.

  Note also that the dispatch values of the multimethod will be
  included in the path, i.e. in reporting and gen overrides, even
  though those values are not evident in the spec.
"
  [mm retag]
  (list 'clojure.spec.alpha/multi-spec-impl
        (list 'quote (res mm))
        (list 'var mm)
        retag))

;; Slice-D2 spec object records (same record-for-reify shape as
;; PredSpec/AndSpec; slice F2 extends `gen*`/`with-gen*`).
;; TupleSpec's `specs` field holds a delay (as upstream), so
;; forward references resolve at first conform, not at build.
(defrecord TupleSpec [forms preds specs gfn])
;; MultiSpec's `mmvar` is derefed fresh on every operation (as
;; upstream), so late-installed methods apply; `retag` is
;; gen-only and rides along for `describe*`.
(defrecord MultiSpec [form mmvar retag gfn id])
;; NonconformingSpec's `spec` field holds a delay (as upstream).
(defrecord NonconformingSpec [spec])
;;;;;;;;;;;;;;;;;;;;;;; gap-122 slice B: combinators ;;;;;;;;;;;;;;;;;;;;;;;
;; `s/or`, `s/nilable`, `s/every`, `s/every-kv`,
;; `s/coll-of`, `s/map-of` (upstream alpha.clj, same 0.5.238
;; oracle as slice 1; `s/tuple` is slice D2's, reused by
;; `every-kv`). Everything below is slice-B code, appended
;; after the slice-D2 definitions so no forward `declare`s are
;; needed. (Slice F2 extends `gen*`/`with-gen*` throughout and
;; restores the upstream `gfn`/`id` ctor params.)

;; Upstream alpha.clj 36-42.
(def ^:dynamic *coll-check-limit*
  "The number of elements validated in a collection spec'ed with 'every'"
  101)

(def ^:dynamic *coll-error-limit*
  "The number of errors reported by explain in a collection spec'ed with 'every'"
  20)

;; Upstream alpha.clj 511-519 (`res-kind`). Takes the macro's raw
;; opts seq and flattens it to alternating key/value describe
;; data, resolving the `:kind` value. The seq-to-map conversion is
;; an explicit `apply hash-map`.
(defn- res-kind-pairs [opts]
  (let [m (apply hash-map opts)
        kind (:kind m)]
    (mapcat identity (if kind (assoc m :kind (res kind)) m))))

;; Upstream alpha.clj 1223-1241. The evaluated max bound is the
;; literal 2147483647 (no `Integer/MAX_VALUE` in the port);
;; `bounded-count` on a counted coll answers the full count, so
;; the check is exact all the same. The explain *data* keeps the
;; `Integer/MAX_VALUE` symbol, as upstream.
(defn- coll-prob [x kfn kform distinct count min-count max-count
                  path via in]
  (let [pred (c/or kfn c/coll?)
        kform (c/or kform 'clojure.core/coll?)]
    (cond
     (not (pvalid? pred x))
     (explain-1 kform pred path via in x)

     (c/and count (not= count (bounded-count count x)))
     [{:path path :pred (list 'clojure.core/= count (list 'clojure.core/count '%)) :val x :via via :in in}]

     (c/and (c/or min-count max-count)
            (not (<= (c/or min-count 0)
                     (bounded-count (if max-count (inc max-count) min-count) x)
                     (c/or max-count 2147483647))))
     [{:path path :pred (list 'clojure.core/<= (c/or min-count 0) (list 'clojure.core/count '%) (c/or max-count 'Integer/MAX_VALUE)) :val x :via via :in in}]

     (c/and distinct (not (empty? x)) (not (apply distinct? x)))
     [{:path path :pred 'distinct? :val x :via via :in in}])))

;; Slice-B spec records. As in slice 1 these are `defrecord`s
;; (not `reify`) so `with-name`/`spec-name` metadata works.
;; `EverySpec` keeps the full opts map (with the `::cpred` fn and
;; `::describe` data the `every` macro built) plus the lazily
;; specized element spec, the index-key fn, and the
;; collection-shape dispatch closure.
(defrecord OrSpec [keys forms preds specs kps gfn id])
(defrecord NilableSpec [form pred spec gfn])
(defrecord EverySpec [form pred opts spec kfn cfns gfn])

(extend-type OrSpec
  Spec
  (conform* [this x]
    (let [specs @(:specs this)
          keys (:keys this)]
      (loop [i 0]
        (if (< i (count specs))
          (let [ret (conform* (specs i) x)]
            (if (invalid? ret)
              (recur (inc i))
              (tagged-ret (keys i) ret)))
          :clojure.spec.alpha/invalid))))
  (unform* [this [k x]] (unform ((:kps this) k) x))
  (explain* [this path via in x]
    (when-not (pvalid? this x)
      (apply concat
             (map (fn [k form pred]
                    (when-not (pvalid? pred x)
                      (explain-1 form pred (conj path k) via in x)))
                  (:keys this) (:forms this) (:preds this)))))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (let [gen (fn [k p f]
                  (let [rmap (inck rmap (:id this))]
                    (when-not (recur-limit? rmap (:id this) path k)
                      (gen/delay
                       (gensub p overrides (conj path k) rmap f)))))
            gs (remove nil? (map gen (:keys this) (:preds this) (:forms this)))]
        (when-not (empty? gs)
          (gen/one-of gs)))))
  (with-gen* [this gfn]
    (or-spec-impl (:keys this) (:forms this) (:preds this) gfn))
  (describe* [this]
    (apply list 'clojure.spec.alpha/or (mapcat vector (:keys this) (:forms this)))))

(extend-type NilableSpec
  Spec
  (conform* [this x] (if (nil? x) nil (conform* @(:spec this) x)))
  (unform* [this x] (if (nil? x) nil (unform* @(:spec this) x)))
  (explain* [this path via in x]
    (when-not (c/or (pvalid? @(:spec this) x) (nil? x))
      (conj (explain-1 (:form this) (:pred this) (conj path :clojure.spec.alpha/pred) via in x)
            {:path (conj path :clojure.spec.alpha/nil) :pred 'nil? :val x :via via :in in})))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (gen/frequency
       [[1 (gen/delay (gen/return nil))]
        [9 (gen/delay (gensub (:pred this) overrides (conj path :clojure.spec.alpha/pred) rmap (:form this)))]])))
  (with-gen* [this gfn]
    (nilable-impl (:form this) (:pred this) gfn))
  (describe* [this] (list 'clojure.spec.alpha/nilable (res (:form this)))))

(extend-type TupleSpec
  Spec
  (conform* [this x]
    (let [specs @(:specs this)
          cnt (count (:preds this))]
      (if-not (c/and (vector? x)
                     (= (count x) cnt))
        :clojure.spec.alpha/invalid
        (loop [ret x i 0]
          (if (= i cnt)
            ret
            (let [v (x i)
                  cv (conform* (specs i) v)]
              (if (invalid? cv)
                :clojure.spec.alpha/invalid
                (recur (if (identical? cv v) ret (assoc ret i cv))
                       (inc i)))))))))
  (unform* [this x]
    (assert (c/and (vector? x)
                   (= (count x) (count (:preds this)))))
    (loop [ret x i 0]
      (if (= i (count x))
        ret
        (let [cv (x i)
              v (unform ((:preds this) i) cv)]
          (recur (if (identical? cv v) ret (assoc ret i v))
                 (inc i))))))
  (explain* [this path via in x]
    (cond
     (not (vector? x))
     [{:path path :pred 'clojure.core/vector? :val x :via via :in in}]

     (not= (count x) (count (:preds this)))
     [{:path path
       :pred (list 'clojure.core/= (list 'clojure.core/count '%) (count (:preds this)))
       :val x :via via :in in}]

     :else
     (apply concat
            (map (fn [i form pred]
                   (let [v (x i)]
                     (when-not (pvalid? pred v)
                       (explain-1 form pred (conj path i) via (conj in i) v))))
                 (range (count (:preds this))) (:forms this) (:preds this)))))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (let [gen (fn [i p f]
                  (gensub p overrides (conj path i) rmap f))
            gs (map gen (range (count (:preds this))) (:preds this) (:forms this))]
        (when (every? identity gs)
          (apply gen/tuple gs)))))
  (with-gen* [this gfn]
    (tuple-impl (:forms this) (:preds this) gfn))
  (describe* [this]
    (apply list 'clojure.spec.alpha/tuple (:forms this))))

(extend-type EverySpec
  Spec
  (conform* [this x]
    (let [opts (:opts this)
          spec @(:spec this)
          cpred (:clojure.spec.alpha/cpred opts)
          conform-all (:clojure.spec.alpha/conform-all opts)]
      (cond
       (not (cpred x)) :clojure.spec.alpha/invalid

       conform-all
       (let [[init add complete] ((:cfns this) x)]
         (loop [ret (init x) i 0 [v & vs :as vseq] (seq x)]
           (if vseq
             (let [cv (conform* spec v)]
               (if (invalid? cv)
                 :clojure.spec.alpha/invalid
                 (recur (add ret i v cv) (inc i) vs)))
             (complete ret))))

       :else
       (if (indexed? x)
         ;; `(quot ...)` for upstream's `(long (/ ...))`: the port
         ;; has no `long`; identical for non-negative counts.
         (let [step (max 1 (quot (count x) *coll-check-limit*))]
           (loop [i 0]
             (if (>= i (count x))
               x
               (if (valid? spec (nth x i))
                 (recur (c/+ i step))
                 :clojure.spec.alpha/invalid))))
         (let [limit *coll-check-limit*]
           (loop [i 0 [v & vs :as vseq] (seq x)]
             (cond
              (c/or (nil? vseq) (= i limit)) x
              (valid? spec v) (recur (inc i) vs)
              :else :clojure.spec.alpha/invalid)))))))
  (unform* [this x]
    (let [opts (:opts this)]
      (if (:clojure.spec.alpha/conform-all opts)
        (let [spec @(:spec this)
              [init add complete] ((:cfns this) x)]
          (loop [ret (init x) i 0 [v & vs :as vseq] (seq x)]
            (if (>= i (count x))
              (complete ret)
              (recur (add ret i v (unform* spec v)) (inc i) vs))))
        x)))
  (explain* [this path via in x]
    (let [opts (:opts this)
          form (:form this)
          pred (:pred this)
          spec @(:spec this)
          kfn (:kfn this)
          conform-all (:clojure.spec.alpha/conform-all opts)]
      (c/or (coll-prob x (:kind opts) (:clojure.spec.alpha/kind-form opts) (:distinct opts) (:count opts) (:min-count opts) (:max-count opts) path via in)
            (apply concat
                   ((if conform-all identity (partial take *coll-error-limit*))
                    (keep identity
                          (map (fn [i v]
                                 (let [k (kfn i v)]
                                   (when-not (valid? spec v)
                                     (explain-1 form pred path via (conj in k) v))))
                               (range) x)))))))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (let [opts (:opts this)
            form (:form this)
            pred (:pred this)
            conform-into (:into opts)
            kind (:kind opts)
            kind-form (:clojure.spec.alpha/kind-form opts)
            gen-max (c/or (:gen-max opts) 20)
            gen-into (if conform-into (empty conform-into) (get empty-coll kind-form))
            pgen (gensub pred overrides path rmap form)]
        (gen/bind
         (cond
          gen-into (gen/return gen-into)
          kind (gen/fmap #(if (empty? %) % (empty %))
                         (gensub kind overrides path rmap form))
          :else (gen/return []))
         (fn [init]
           (gen/fmap
            #(if (vector? init) % (into init %))
            (cond
             (:distinct opts)
             (if (:count opts)
               (gen/vector-distinct pgen {:num-elements (:count opts) :max-tries 100})
               (gen/vector-distinct pgen {:min-elements (c/or (:min-count opts) 0)
                                          :max-elements (c/or (:max-count opts) (max gen-max (c/* 2 (c/or (:min-count opts) 0))))
                                          :max-tries 100}))

             (:count opts)
             (gen/vector pgen (:count opts))

             (c/or (:min-count opts) (:max-count opts))
             (gen/vector pgen (c/or (:min-count opts) 0) (c/or (:max-count opts) (max gen-max (c/* 2 (c/or (:min-count opts) 0)))))

             :else
             (gen/vector pgen 0 gen-max))))))))
  (with-gen* [this gfn]
    (every-impl (:form this) (:pred this) (:opts this) gfn))
  (describe* [this]
    (let [opts (:opts this)]
      (c/or (:clojure.spec.alpha/describe opts)
            (apply list 'clojure.spec.alpha/every (res (:form this)) (mapcat identity opts))))))

;; Upstream alpha.clj 476-491. Built with `list` (no
;; syntax-quote: see the file header).
(defmacro or
  "Takes key+pred pairs, e.g.

  (s/or :even even? :small #(< % 42))

  Returns a destructuring spec that returns a map entry containing the
  key of the first matching pred and the corresponding value. Thus the
  'key' and 'val' functions can be used to refer generically to the
  components of the tagged return."
  [& key-pred-forms]
  (let [pairs (partition 2 key-pred-forms)
        keys (mapv first pairs)
        pred-forms (mapv second pairs)
        pf (mapv res pred-forms)]
    (assert (c/and (even? (count key-pred-forms)) (every? keyword? keys)) "spec/or expects k1 p1 k2 p2..., where ks are keywords")
    (list 'clojure.spec.alpha/or-spec-impl keys (list 'quote pf) pred-forms)))

;; Upstream alpha.clj 520-566. The opts seq is parsed with an
;; explicit `apply hash-map` (see `res-kind-pairs`); `:gen` is
;; threaded to `every-impl` as `gfn` (and stays dissoced from the
;; describe opts, as upstream). The `::cpred` fn form is emitted
;; with `list` (a `fn*` over a gensym, as upstream); the only
;; evaluation-shape change is the `Integer/MAX_VALUE` default
;; bound, which reads 2147483647 (the port has no
;; `Integer/MAX_VALUE`).
(defmacro every
  "takes a pred and validates collection elements against that pred.

  Note that 'every' does not do exhaustive checking, rather it samples
  *coll-check-limit* elements. Nor (as a result) does it do any
  conforming of elements. 'explain' will report at most *coll-error-limit*
  problems.  Thus 'every' should be suitable for potentially large
  collections.

  Takes several kwargs options that further constrain the collection:

  :kind - a pred that the collection type must satisfy, e.g. vector?
        (default nil) Note that if :kind is specified and :into is
        not, this pred must generate in order for every to generate.
  :count - specifies coll has exactly this count (default nil)
  :min-count, :max-count - coll has count (<= min-count count max-count) (defaults nil)
  :distinct - all the elements are distinct (default nil)

  And additional args that control gen

  :gen-max - the maximum coll size to generate (default 20)
  :into - one of [], (), {}, #{} - the default collection to generate into
      (default: empty coll as generated by :kind pred if supplied, else [])

  Optionally takes :gen generator-fn, which must be a fn of no args that
  returns a test.check generator

  See also - coll-of, every-kv
"
  [pred & opts]
  (let [m (apply hash-map opts)
        kind-expr (:kind m)
        count-expr (:count m)
        min-expr (:min-count m)
        max-expr (:max-count m)
        distinct-expr (:distinct m)
        desc-override (:clojure.spec.alpha/describe m)
        gx (gensym "every")
        cpreds (cond-> [(list (c/or kind-expr 'clojure.core/coll?) gx)]
                 count-expr (conj (list 'clojure.core/= count-expr
                                        (list 'clojure.core/bounded-count count-expr gx)))
                 (c/or min-expr max-expr)
                 (conj (list 'clojure.core/<=
                             (list 'clojure.core/or min-expr 0)
                             (list 'clojure.core/bounded-count
                                   (list 'if max-expr
                                         (list 'clojure.core/inc max-expr)
                                         min-expr)
                                   gx)
                             (list 'clojure.core/or max-expr 2147483647)))
                 distinct-expr
                 (conj (list 'clojure.core/or
                             (list 'clojure.core/empty? gx)
                             (list 'clojure.core/apply 'clojure.core/distinct? gx))))
        kind-form (res kind-expr)
        ;; A `::describe` override from `coll-of`/`map-of`/`every-kv`
        ;; is already a `(quote ...)` form; only the locally built
        ;; data needs quoting.
        desc-form (c/or desc-override
                        (list 'quote (apply list 'clojure.spec.alpha/every (res pred) (res-kind-pairs opts))))
        nopts (-> m
                  (dissoc :gen :clojure.spec.alpha/describe)
                  (assoc :clojure.spec.alpha/kind-form (list 'quote kind-form)
                         :clojure.spec.alpha/describe desc-form
                         :clojure.spec.alpha/cpred (list 'fn* [gx] (apply list 'clojure.core/and cpreds))))]
    (list 'clojure.spec.alpha/every-impl (list 'quote pred) pred nopts (:gen m))))

;; Upstream alpha.clj 568-577.
(defmacro every-kv
  "like 'every' but takes separate key and val preds and works on associative collections.

  Same options as 'every', :into defaults to {}

  See also - map-of"
  [kpred vpred & opts]
  (let [desc (apply list 'clojure.spec.alpha/every-kv (res kpred) (res vpred) (res-kind-pairs opts))
        gi (gensym "i")
        gv (gensym "v")]
    (apply list
           'clojure.spec.alpha/every
           (list 'clojure.spec.alpha/tuple kpred vpred)
           :clojure.spec.alpha/kfn (list 'clojure.core/fn [gi gv] (list 'clojure.core/nth gv 0))
           :into {}
           :clojure.spec.alpha/describe (list 'quote desc)
           opts)))

;; Upstream alpha.clj 579-590.
(defmacro coll-of
  "Returns a spec for a collection of items satisfying pred. Unlike
  'every', coll-of will exhaustively conform every value.

  Same options as 'every'. conform will produce a collection
  corresponding to :into if supplied, else will match the input collection,
  avoiding rebuilding when possible.

  See also - every, map-of"
  [pred & opts]
  (let [desc (apply list 'clojure.spec.alpha/coll-of (res pred) (res-kind-pairs opts))]
    (apply list
           'clojure.spec.alpha/every
           pred
           :clojure.spec.alpha/conform-all true
           :clojure.spec.alpha/describe (list 'quote desc)
           opts)))

;; Upstream alpha.clj 592-604. The `:kind map?` default precedes
;; the user opts so an explicit `:kind` wins, as upstream.
(defmacro map-of
  "Returns a spec for a map whose keys satisfy kpred and vals satisfy
  vpred. Unlike 'every-kv', map-of will exhaustively conform every
  value.

  Same options as 'every', :kind defaults to map?, with the addition of:

  :conform-keys - conform keys as well as values (default false)

  See also - every-kv"
  [kpred vpred & opts]
  (let [desc (apply list 'clojure.spec.alpha/map-of (res kpred) (res vpred) (res-kind-pairs opts))]
    (apply list
           'clojure.spec.alpha/every-kv
           kpred vpred
           :clojure.spec.alpha/conform-all true
           :kind 'clojure.core/map?
           :clojure.spec.alpha/describe (list 'quote desc)
           opts)))

;; Upstream alpha.clj 1860-1864.
(defmacro nilable
  "returns a spec that accepts nil and values satisfying pred"
  [pred]
  (let [pf (res pred)]
    (list 'clojure.spec.alpha/nilable-impl (list 'quote pf) pred)))

;; Upstream alpha.clj 1061-1066. The 2-/3-branch fast paths are
;; one loop (behavior-identical), as with `AndSpec`. The `id` is a
;; `random-uuid` minted per construction (as upstream); the short
;; arity (nil `gfn`) stays for the `or` macro.
(defn ^:skip-wiki or-spec-impl
  "Do not call this directly, use 'or'"
  ([keys forms preds] (or-spec-impl keys forms preds nil))
  ([keys forms preds gfn]
   (->OrSpec keys forms preds (delay (mapv specize preds forms)) (zipmap keys preds) gfn (random-uuid))))

;; Upstream alpha.clj 1834-1837. The `[form pred]` arity (nil `gfn`)
;; stays for the `nilable` macro.
(defn ^:skip-wiki nilable-impl
  "Do not call this directly, use 'nilable'"
  ([form pred] (nilable-impl form pred nil))
  ([form pred gfn]
   (->NilableSpec form pred (delay (specize pred form)) gfn)))


;; Upstream alpha.clj 962-974. These two helpers stay `id`/`tag`-free
;; (those live in `MultiSpec` `gen*`, slice F2).
;; The method-exists check is `get-method`, exactly like upstream's
;; `.getMethod` (hierarchy-aware, nil on miss, throws on
;; ambiguity): no try/catch, so method-body throws propagate.
;; The dispatch fn comes from the `__dispatch_fn*` glue native
;; (no `.dispatchFn` interop in the port).
(defn- multi-dval [mmvar x]
  ((clojure.core/__dispatch_fn* @mmvar) x))

(defn- multi-predx [mmvar x]
  (let [mm @mmvar
        dv ((clojure.core/__dispatch_fn* mm) x)]
    (c/and (get-method mm dv) (mm x))))

(extend-type MultiSpec
  Spec
  (conform* [this x]
    (if-let [pred (multi-predx (:mmvar this) x)]
      (dt pred x (:form this))
      :clojure.spec.alpha/invalid))
  (unform* [this x]
    (if-let [pred (multi-predx (:mmvar this) x)]
      (unform pred x)
      (throw (ex-info (str "No method of: " (:form this)
                           " for dispatch value: " (multi-dval (:mmvar this) x))
                      {}))))
  (explain* [this path via in x]
    (let [dv (multi-dval (:mmvar this) x)
          path (conj path dv)]
      (if-let [pred (multi-predx (:mmvar this) x)]
        (explain-1 (:form this) pred path via in x)
        [{:path path :pred (:form this) :val x :reason "no method" :via via :in in}])))
  (gen* [this overrides path rmap]
    (if (:gfn this)
      ((:gfn this))
      (let [tag (if (keyword? (:retag this))
                  #(assoc %1 (:retag this) %2)
                  (:retag this))
            gen (fn [[k f]]
                  (let [p (f nil)]
                    (let [rmap (inck rmap (:id this))]
                      (when-not (recur-limit? rmap (:id this) path k)
                        (gen/delay
                         (gen/fmap
                          #(tag % k)
                          (gensub p overrides (conj path k) rmap (list 'method (:form this) k))))))))
            gs (->> (methods @(:mmvar this))
                    (remove (fn [[k]] (invalid? k)))
                    (map gen)
                    (remove nil?))]
        (when (every? identity gs)
          (gen/one-of gs)))))
  (with-gen* [this gfn]
    (multi-spec-impl (:form this) (:mmvar this) (:retag this) gfn))
  (describe* [this]
    (list 'clojure.spec.alpha/multi-spec (:form this) (:retag this))))

(extend-type NonconformingSpec
  Spec
  (conform* [this x]
    (let [ret (conform* @(:spec this) x)]
      (if (invalid? ret)
        :clojure.spec.alpha/invalid
        x)))
  (unform* [this x] x)
  (explain* [this path via in x]
    (explain* @(:spec this) path via in x))
  (gen* [this overrides path rmap] (gen* @(:spec this) overrides path rmap))
  (with-gen* [this gfn] (nonconforming (with-gen* @(:spec this) gfn)))
  (describe* [this]
    (list 'clojure.spec.alpha/nonconforming (describe* @(:spec this)))))

;; Upstream alpha.clj 996-1059. Builds a TupleSpec record. The
;; `[forms preds]` arity (nil `gfn`) stays for the `tuple` macro.
(defn ^:skip-wiki tuple-impl
  "Do not call this directly, use 'tuple'"
  ([forms preds] (tuple-impl forms preds nil))
  ([forms preds gfn]
   (->TupleSpec forms preds (delay (mapv specize preds forms)) gfn)))

;; Upstream alpha.clj 946-994. Builds a MultiSpec record. The
;; `[form mmvar retag]` arity (nil `gfn`) stays for the
;; `multi-spec` macro; the `id` is a `random-uuid` minted per
;; construction (as upstream).
(defn ^:skip-wiki multi-spec-impl
  "Do not call this directly, use 'multi-spec'"
  ([form mmvar retag] (multi-spec-impl form mmvar retag nil))
  ([form mmvar retag gfn]
   (->MultiSpec form mmvar retag gfn (random-uuid))))

;; Upstream alpha.clj 1813-1832. Builds a NonconformingSpec record.
(defn ^:skip-wiki nonconforming
  "takes a spec and returns a spec that has the same properties except
  'conform' returns the original (not the conformed) value. Note, will specize regex ops."
  [spec]
  (->NonconformingSpec (delay (specize spec))))

;; Upstream alpha.clj 1243. Keys are the evaluated value of
;; upstream's syntax-quoted kind symbols (`res` of a `:kind` form
;; answers the same). Used by `EverySpec` `gen*` only.
(def ^:private empty-coll
  {'clojure.core/vector? [] 'clojure.core/set? #{} 'clojure.core/list? () 'clojure.core/map? {}})

;; Upstream alpha.clj 1245-1282. (`gen-into`/`gen-max` live in the
;; `gen*` body, computed from the stored opts, instead of closing
;; over the ctor locals as upstream.) The `count` opt shadows
;; `clojure.core/count` in this body, so every boolean connective
;; here is `c/and`/`c/or` (as upstream, which needs the same for
;; its shadowed `and`/`or`).
(defn ^:skip-wiki every-impl
  "Do not call this directly, use 'every', 'every-kv', 'coll-of' or 'map-of'"
  ([form pred opts] (every-impl form pred opts nil))
  ([form pred {conform-into :into
              describe-form :clojure.spec.alpha/describe
              :keys [kind :clojure.spec.alpha/kind-form count max-count min-count distinct gen-max :clojure.spec.alpha/kfn :clojure.spec.alpha/cpred
                     conform-keys :clojure.spec.alpha/conform-all]
              :or {gen-max 20}
              :as opts}
    gfn]
  (let [spec (delay (specize pred))
        kfn (c/or kfn (fn [i v] i))
        addcv (fn [ret i v cv] (conj ret cv))
        cfns (fn [x]
               ;; Returns a tuple of [init add complete] fns.
               (cond
                (c/and (vector? x) (c/or (not conform-into) (vector? conform-into)))
                [identity
                 (fn [ret i v cv]
                   (if (identical? v cv)
                     ret
                     (assoc ret i cv)))
                 identity]

                (c/and (map? x) (c/or (c/and kind (not conform-into)) (map? conform-into)))
                [(if conform-keys empty identity)
                 (fn [ret i v cv]
                   (if (c/and (identical? v cv) (not conform-keys))
                     ret
                     (assoc ret (nth (if conform-keys cv v) 0) (nth cv 1))))
                 identity]

                (c/or (list? conform-into) (seq? conform-into) (c/and (not conform-into) (c/or (list? x) (seq? x))))
                [(constantly ()) addcv reverse]

                :else [(fn [coll] (empty (c/or conform-into coll))) addcv identity]))]
    (->EverySpec form pred opts spec kfn cfns gfn))))

