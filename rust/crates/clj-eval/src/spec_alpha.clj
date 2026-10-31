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
;;   Slice D2's TupleSpec/MultiSpec/NonconformingSpec records follow
;;   the same shape.
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
(ns clojure.spec.alpha
  "The spec library specifies the structure of data or functions and provides
  operations to validate, conform, explain, describe, and generate data based on
  the specs.

  Rationale: https://clojure.org/about/spec
  Guide: https://clojure.org/guides/spec"
  (:refer-clojure :exclude [and def]))

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
;; Slice D2: PredSpec carries the upstream `cpred?`/`unc`
;; conformer flags (nil for plain predicate specs).
(defrecord PredSpec [form pred cpred? unc])
(defrecord AndSpec [forms preds specs])

;; Forward declarations: the `extend-type` blocks below call fns
;; defined later in this file (the port resolves globals at analyze
;; time), as do `specize` and `def-impl` (-> `spec-impl`).
(declare spec-impl dt invalid? unform explain-pred-list)

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

;; Upstream alpha.clj 779-784. Slice D2: shared with slice A
;; (`keys`/`merge`) — see docs/gap-122/slice-D2.md for the overlap
;; note. `::unknown` is written out (no auto-resolve, per the file
;; header).
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

;; Upstream alpha.clj 913-944, narrowed (the `gfn` parameter is
;; generator machinery, a later slice). Slice D2 adds the
;; `cpred?`/`unc` conformer arities (upstream's `[form pred gfn
;; cpred?]` / `[form pred gfn cpred? unc]` minus `gfn`). Builds a
;; PredSpec record; a regex pred fails loudly (no `regex-spec-impl`
;; yet).
(defn ^:skip-wiki spec-impl
  "Do not call this directly, use 'spec'"
  ([form pred] (spec-impl form pred nil nil))
  ([form pred cpred?] (spec-impl form pred cpred? nil))
  ([form pred cpred? unc]
   (cond
    (spec? pred) pred
    (regex? pred) (throw (ex-info "regex specs are out of scope in gap-122 slice 1" {}))
    (ident? pred) (the-spec pred)
    :else (->PredSpec form pred cpred? unc))))

;; Upstream alpha.clj 1151-1193, narrowed to `[forms preds]` (same
;; reason as `spec-impl`). Builds an AndSpec record; the per-count
;; fast paths are one loop (behavior-identical).
(defn ^:skip-wiki and-spec-impl
  "Do not call this directly, use 'and'"
  [forms preds]
  (->AndSpec forms preds (delay (mapv specize preds forms))))

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
             (list 'quote (list 'clojure.spec.alpha/conformer (res f)))
             f true))
  ([f unf] (list 'clojure.spec.alpha/spec-impl
                 (list 'quote (list 'clojure.spec.alpha/conformer (res f) (res unf)))
                 f true unf)))

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
;; PredSpec/AndSpec; `gen*`/`with-gen*` stay unextended).
;; TupleSpec's `specs` field holds a delay (as upstream), so
;; forward references resolve at first conform, not at build.
(defrecord TupleSpec [forms preds specs])
;; MultiSpec's `mmvar` is derefed fresh on every operation (as
;; upstream), so late-installed methods apply; `retag` is
;; gen-only and rides along for `describe*`.
(defrecord MultiSpec [form mmvar retag])
;; NonconformingSpec's `spec` field holds a delay (as upstream).
(defrecord NonconformingSpec [spec])

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
  (describe* [this]
    (apply list 'clojure.spec.alpha/tuple (:forms this))))

;; Upstream alpha.clj 962-974, narrowed (no `id`/`tag`: gen-only).
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
  (describe* [this]
    (list 'clojure.spec.alpha/nonconforming (describe* @(:spec this)))))

;; Upstream alpha.clj 996-1059, narrowed to `[forms preds]` (the
;; `gfn` parameter is generator machinery). Builds a TupleSpec
;; record.
(defn ^:skip-wiki tuple-impl
  "Do not call this directly, use 'tuple'"
  [forms preds]
  (->TupleSpec forms preds (delay (mapv specize preds forms))))

;; Upstream alpha.clj 946-994, narrowed to `[form mmvar retag]`
;; (same reason as `tuple-impl`). Builds a MultiSpec record.
(defn ^:skip-wiki multi-spec-impl
  "Do not call this directly, use 'multi-spec'"
  [form mmvar retag]
  (->MultiSpec form mmvar retag))

;; Upstream alpha.clj 1813-1832, narrowed (no `gen*`/`with-gen*`).
;; Builds a NonconformingSpec record.
(defn ^:skip-wiki nonconforming
  "takes a spec and returns a spec that has the same properties except
  'conform' returns the original (not the conformed) value. Note, will specize regex ops."
  [spec]
  (->NonconformingSpec (delay (specize spec))))

