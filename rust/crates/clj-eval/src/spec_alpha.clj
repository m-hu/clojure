;; Copyright (c) Rich Hickey. All rights reserved.
;; The use and distribution terms for this software are covered by the
;; Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;; which can be found in the file epl-v10.html at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

;; gap-122 slices 1+E: a faithful subset port of clojure.spec.alpha
;; (upstream spec.alpha-0.6.249, clojure/spec/alpha.clj) covering the
;; spec registry, `def`, predicate specs, `s/and`, `valid?`,
;; `conform`/`unform`, basic `explain`/`explain-str`, and (slice E)
;; regex specs: `s/cat`, `s/alt`, `s/*`, `s/+`, `s/?`, `s/&` over
;; the derivative engine, plus the `s/keys*` macro (expansion only;
;; runtime use waits for slices A + D + F).
;;
;; Port adaptations (see docs/gap-122/slice-1.md and
;; docs/gap-122/slice-E.md for the full lists):
;; - No syntax-quote and no `::auto-resolve` anywhere in this file:
;;   the port reads a whole load batch before evaluating it, so both
;;   would qualify against the *requiring* namespace instead of this
;;   one. Macros build expansions with `list` (fully-qualified
;;   heads), and spec keywords are written out in full.
;; - Spec objects are `defrecord`s (PredSpec/AndSpec/RegexSpec) with
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
;; - `gen*`/`with-gen*` are declared on the protocol but left
;;   unextended on PredSpec/AndSpec (generators are a later slice);
;;   on RegexSpec both surface the `re-gen` stub (`ex-info`
;;   "generators slice pending"), which slice F replaces.
;; - `tagged-ret` is `(first {tag ret})`: the port cannot build a
;;   `MapEntry` object, but the entry of a single-entry map is one.
;; - `regex-spec-impl` narrows to `[re]` (`gfn` dropped, as the
;;   other `-impl` ctors); the non-sequential `explain*` `:pred`
;;   is the literal `(clojure.core/fn [%] (clojure.core/or ...))`
;;   form (the deterministic value of upstream's `(res ...)` call).
;; - `keys*` keeps `let`/`fn`/core fns bare in its expansion
;;   (use-site refers; only the unqualified `let` is the special
;;   form) while spec/gen heads fully qualify.
(ns clojure.spec.alpha
  "The spec library specifies the structure of data or functions and provides
  operations to validate, conform, explain, describe, and generate data based on
  the specs.

  Rationale: https://clojure.org/about/spec
  Guide: https://clojure.org/guides/spec"
  (:refer-clojure :exclude [and def + * cat]))

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
;; Slice E (upstream alpha.clj 1697-1723): the Spec glide over a
;; regex op map. Upstream closes over `re` in a `reify`; the port
;; holds it in a record field (metadata-carrying, as above).
(defrecord RegexSpec [re])

;; Forward declarations: the `extend-type` blocks below call fns
;; defined later in this file (the port resolves globals at analyze
;; time), as do `specize` and `def-impl` (-> `spec-impl`), and
;; `maybe-spec`/`spec-impl` (-> `regex-spec-impl`).
(declare spec-impl dt invalid? unform explain-pred-list regex-spec-impl re-conform op-unform re-explain re-gen op-describe regex-guard-form)

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
    (re-gen (:re this) overrides path rmap (op-describe (:re this))))
  (with-gen* [this gfn]
    (throw (ex-info "generators slice pending" {})))
  (describe* [this] (op-describe (:re this))))

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
;; PredSpec record; a regex pred wraps via `regex-spec-impl`
;; (slice E).
(defn ^:skip-wiki spec-impl
  "Do not call this directly, use 'spec'"
  [form pred]
  (cond
   (spec? pred) pred
   (regex? pred) (regex-spec-impl pred)
   (ident? pred) (the-spec pred)
   :else (->PredSpec form pred)))

;; Upstream alpha.clj 1151-1193, narrowed to `[forms preds]` (same
;; reason as `spec-impl`). Builds an AndSpec record; the per-count
;; fast paths are one loop (behavior-identical).
(defn ^:skip-wiki and-spec-impl
  "Do not call this directly, use 'and'"
  [forms preds]
  (->AndSpec forms preds (delay (mapv specize preds forms))))

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

;; Upstream alpha.clj 1617-1655. STUB: slice F replaces this body
;; with the real generator; until then both gen paths fail
;; honestly (see `RegexSpec`).
(defn- re-gen [p overrides path rmap f]
  (throw (ex-info "generators slice pending" {})))

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

;; Upstream alpha.clj 1697-1723, narrowed to `[re]` (slice-1 rule 8:
;; `gfn` dropped; `gen*`/`with-gen*` surface the `re-gen` stub
;; until slice F). Builds a RegexSpec record (slice-1 rule 2).
(defn ^:skip-wiki regex-spec-impl
  "Do not call this directly, use 'spec' with a regex op argument"
  [re]
  (->RegexSpec re))

;; Upstream alpha.clj 1794-1812 (built with `list`, as `def`;
;; `let`/`fn`/core fns stay bare for use-site resolution while
;; spec/gen heads fully qualify). `::kvs->map` itself needs
;; `conformer` (slice D), so only the macro lands here: runtime
;; use waits for slices A + D + F.
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
                            (list 'clojure.spec.gen.alpha/gen mspec)))))))

