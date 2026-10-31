;; Copyright (c) Rich Hickey. All rights reserved.
;; The use and distribution terms for this software are covered by the
;; Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;; which can be found in the file epl-v10.html at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

;; gap-122 slice 1: a faithful subset port of clojure.spec.alpha
;; (upstream spec.alpha-0.5.238, clojure/spec/alpha.clj) covering the
;; spec registry, `def`, predicate specs, `s/and`, `valid?`,
;; `conform`/`unform`, and basic `explain`/`explain-str`.
;;
;; Port adaptations (see docs/gap-122/slice-1.md for the full list):
;; - No syntax-quote and no `::auto-resolve` anywhere in this file:
;;   the port reads a whole load batch before evaluating it, so both
;;   would qualify against the *requiring* namespace instead of this
;;   one. Macros build expansions with `list` (fully-qualified
;;   heads), and spec keywords are written out in full.
;; - Spec objects are `defrecord`s (PredSpec/AndSpec) with the Spec
;;   protocol extended to them: the port's `reify` instances do not
;;   support metadata, while the JVM's do (`with-name` needs it).
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
;; - `gen*`/`with-gen*` are declared on the protocol but left
;;   unextended (generators are a later slice).
;;
;; gap-122 slice B (see docs/gap-122/slice-B.md) appends the
;; `s/or`, `s/nilable`, `s/tuple`, `s/every`, `s/every-kv`,
;; `s/coll-of`, and `s/map-of` combinators below the slice-1 code.
;; Further port adaptations: `tagged-ret` builds its entry via
;; `find` (no `MapEntry.` interop); evaluated `Integer/MAX_VALUE`
;; reads are the literal 2147483647; the sampling step uses `quot`
;; (no `long`); the `every` opts map is built with an explicit
;; `apply hash-map`; `or` joins the `:refer-clojure :exclude`
;; list.
(ns clojure.spec.alpha
  "The spec library specifies the structure of data or functions and provides
  operations to validate, conform, explain, describe, and generate data based on
  the specs.

  Rationale: https://clojure.org/about/spec
  Guide: https://clojure.org/guides/spec"
  (:refer-clojure :exclude [and def or]))

(alias 'c 'clojure.core)

;; Upstream alpha.clj 44-50. `gen*`/`with-gen*` are declared so later
;; slices extend them; slice 1 leaves them unimplemented.
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
(defrecord PredSpec [form pred])
(defrecord AndSpec [forms preds specs])

;; Forward declarations: the `extend-type` blocks below call fns
;; defined later in this file (the port resolves globals at analyze
;; time), as do `specize` and `def-impl` (-> `spec-impl`).
(declare spec-impl dt invalid? unform explain-pred-list)

(extend-type PredSpec
  Spec
  (conform* [this x]
    (let [ret ((:pred this) x)]
      (if ret x :clojure.spec.alpha/invalid)))
  (unform* [this x] x)
  (explain* [this path via in x]
    (when (invalid? (dt (:pred this) x (:form this)))
      [{:path path :pred (:form this) :val x :via via :in in}]))
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
  (describe* [this]
    (apply list 'clojure.spec.alpha/and (:forms this))))

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

;; Upstream alpha.clj 110-119. Regex ops cannot be built in slice 1,
;; so a regex reaching here is user data: fail loudly instead of
;; wrapping it (no `regex-spec-impl` yet).
(defn- maybe-spec
  "spec-or-k must be a spec, regex or resolvable kw/sym, else returns nil."
  [spec-or-k]
  (let [s (c/or (c/and (ident? spec-or-k) (reg-resolve spec-or-k))
                (spec? spec-or-k)
                (regex? spec-or-k)
                nil)]
    (if (regex? s)
      (throw (ex-info "regex specs are out of scope in gap-122 slice 1" {}))
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

;; Upstream alpha.clj 366-383, narrowed to `[form]`: the `:gen`
;; kwarg belongs to the generators slice.
(defmacro spec
  "Takes a single predicate form, e.g. can be the name of a predicate,
  like even?, or a fn literal like #(< % 42).

  Returns a spec."
  [form]
  (list 'clojure.spec.alpha/spec-impl (list 'quote (res form)) form))

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

;; Upstream alpha.clj 757-768, narrowed: `cpred?` is conformer
;; machinery (a later slice). Throw via ex-info, as `reg-resolve!`.
(defn- dt
  [pred x form]
  (if pred
    (if-let [spec (the-spec pred)]
      (conform spec x)
      (if (ifn? pred)
        (if (pred x) x :clojure.spec.alpha/invalid)
        (throw (ex-info (str (pr-str form) " is not a fn, expected predicate fn") {}))))
    x))

;; Upstream alpha.clj 770-777.
(defn valid?
  "Helper function that returns true when x is valid for spec."
  ([spec x]
   (let [spec (specize spec)]
     (not (invalid? (conform* spec x)))))
  ([spec x form]
   (let [spec (specize spec form)]
     (not (invalid? (conform* spec x))))))

;; Upstream alpha.clj 786-793.
(defn- explain-1 [form pred path via in v]
  (let [pred (maybe-spec pred)]
    (if (spec? pred)
      (explain* pred path (if-let [name (spec-name pred)] (conj via name) via) in v)
      [{:path path :pred form :val v :via via :in in}])))

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

;; Upstream alpha.clj 913-944, narrowed to `[form pred]` (`gfn` and
;; `cpred?`/`unc` are generator/conformer machinery). Builds a
;; PredSpec record; a regex pred fails loudly (no `regex-spec-impl`
;; yet).
(defn ^:skip-wiki spec-impl
  "Do not call this directly, use 'spec'"
  [form pred]
  (cond
   (spec? pred) pred
   (regex? pred) (throw (ex-info "regex specs are out of scope in gap-122 slice 1" {}))
   (ident? pred) (the-spec pred)
   :else (->PredSpec form pred)))

;; Upstream alpha.clj 1151-1193, narrowed to `[forms preds]` (same
;; reason as `spec-impl`). Builds an AndSpec record; the per-count
;; fast paths are one loop (behavior-identical).
(defn ^:skip-wiki and-spec-impl
  "Do not call this directly, use 'and'"
  [forms preds]
  (->AndSpec forms preds (delay (mapv specize preds forms))))

;;;;;;;;;;;;;;;;;;;;;;; gap-122 slice B: combinators ;;;;;;;;;;;;;;;;;;;;;;;
;; `s/or`, `s/nilable`, `s/tuple`, `s/every`, `s/every-kv`,
;; `s/coll-of`, `s/map-of` (upstream alpha.clj, same 0.5.238
;; oracle as slice 1). Everything below is slice-B code, appended
;; after the slice-1 definitions so no forward `declare`s are
;; needed. `gen*`/`with-gen*` stay unextended; every `-impl`
;; constructor drops the `gfn` param.

;; Upstream alpha.clj 36-42.
(def ^:dynamic *coll-check-limit*
  "The number of elements validated in a collection spec'ed with 'every'"
  101)

(def ^:dynamic *coll-error-limit*
  "The number of errors reported by explain in a collection spec'ed with 'every'"
  20)

;; Upstream alpha.clj 779-784.
(defn- pvalid?
  "internal helper function that returns true when x is valid for spec."
  ([pred x]
   (not (invalid? (dt pred x :clojure.spec.alpha/unknown))))
  ([pred x form]
   (not (invalid? (dt pred x form)))))

;; Upstream alpha.clj 1058-1059. The port has no `MapEntry.`
;; interop, so the tagged return is built with `find`: a real map
;; entry, printing as `[tag ret]` with working `key`/`val`/`nth`.
(defn- tagged-ret [tag ret]
  (find {tag ret} tag))

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
(defrecord OrSpec [keys forms preds specs kps])
(defrecord NilableSpec [form pred spec])
(defrecord TupleSpec [forms preds specs])
(defrecord EverySpec [form pred opts spec kfn cfns])

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
  (describe* [this] (list 'clojure.spec.alpha/nilable (res (:form this)))))

(extend-type TupleSpec
  Spec
  (conform* [this x]
    (let [specs @(:specs this)
          cnt (count (:preds this))]
      (if-not (c/and (vector? x) (= (count x) cnt))
        :clojure.spec.alpha/invalid
        (loop [ret x i 0]
          (if (= i cnt)
            ret
            (let [v (nth x i)
                  cv (conform* (specs i) v)]
              (if (invalid? cv)
                :clojure.spec.alpha/invalid
                (recur (if (identical? cv v) ret (assoc ret i cv))
                       (inc i)))))))))
  (unform* [this x]
    (assert (c/and (vector? x) (= (count x) (count (:preds this)))))
    (let [preds (:preds this)]
      (loop [ret x i 0]
        (if (= i (count x))
          ret
          (let [cv (nth x i)
                v (unform (nth preds i) cv)]
            (recur (if (identical? cv v) ret (assoc ret i v))
                   (inc i)))))))
  (explain* [this path via in x]
    (cond
     (not (vector? x))
     [{:path path :pred 'clojure.core/vector? :val x :via via :in in}]

     (not= (count x) (count (:preds this)))
     [{:path path :pred (list 'clojure.core/= (list 'clojure.core/count '%) (count (:preds this))) :val x :via via :in in}]

     :else
     (apply concat
            (map (fn [i form pred]
                   (let [v (nth x i)]
                     (when-not (pvalid? pred v)
                       (explain-1 form pred (conj path i) via (conj in i) v))))
                 (range (count (:preds this))) (:forms this) (:preds this)))))
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
                 (recur (+ i step))
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
;; accepted and dropped (generators are a later slice). The
;; `::cpred` fn form is emitted with `list` (a `fn*` over a
;; gensym, as upstream); the only evaluation-shape change is the
;; `Integer/MAX_VALUE` default bound, which reads 2147483647 (the
;; port has no `Integer/MAX_VALUE`).
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
    (list 'clojure.spec.alpha/every-impl (list 'quote pred) pred nopts)))

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

;; Upstream alpha.clj 694-700.
(defmacro tuple
  "takes one or more preds and returns a spec for a tuple, a vector
  where each element conforms to the corresponding pred. Each element
  will be referred to in paths using its ordinal."
  [& preds]
  (assert (not (empty? preds)))
  (list 'clojure.spec.alpha/tuple-impl (list 'quote (mapv res preds)) (vec preds)))

;; Upstream alpha.clj 1860-1864.
(defmacro nilable
  "returns a spec that accepts nil and values satisfying pred"
  [pred]
  (let [pf (res pred)]
    (list 'clojure.spec.alpha/nilable-impl (list 'quote pf) pred)))

;; Upstream alpha.clj 1061-1066, narrowed to `[keys forms preds]`
;; (no `id`/`gfn`: generator machinery). The 2-/3-branch fast
;; paths are one loop (behavior-identical), as with `AndSpec`.
(defn ^:skip-wiki or-spec-impl
  "Do not call this directly, use 'or'"
  [keys forms preds]
  (->OrSpec keys forms preds (delay (mapv specize preds forms)) (zipmap keys preds)))

;; Upstream alpha.clj 1834-1837, narrowed to `[form pred]`.
(defn ^:skip-wiki nilable-impl
  "Do not call this directly, use 'nilable'"
  [form pred]
  (->NilableSpec form pred (delay (specize pred form))))

;; Upstream alpha.clj 996-1001, narrowed to `[forms preds]`.
(defn ^:skip-wiki tuple-impl
  "Do not call this directly, use 'tuple'"
  [forms preds]
  (->TupleSpec forms preds (delay (mapv specize preds forms))))

;; Upstream alpha.clj 1245-1282, narrowed to `[form pred opts]`
;; (no `gfn`; `gen-into`/`gen-max` are generator machinery). The
;; `count` opt shadows `clojure.core/count` in this body, so every
;; boolean connective here is `c/and`/`c/or` (as upstream, which
;; needs the same for its shadowed `and`/`or`).
(defn ^:skip-wiki every-impl
  "Do not call this directly, use 'every', 'every-kv', 'coll-of' or 'map-of'"
  [form pred {conform-into :into
              describe-form :clojure.spec.alpha/describe
              :keys [kind :clojure.spec.alpha/kind-form count max-count min-count distinct gen-max :clojure.spec.alpha/kfn :clojure.spec.alpha/cpred
                     conform-keys :clojure.spec.alpha/conform-all]
              :or {gen-max 20}
              :as opts}]
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
    (->EverySpec form pred opts spec kfn cfns)))

