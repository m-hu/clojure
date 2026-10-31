;; Copyright (c) Rich Hickey. All rights reserved.
;; The use and distribution terms for this software are covered by the
;; Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;; which can be found in the file epl-v10.html at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

;; gap-122 slice F1: a narrow port of clojure.spec.gen.alpha (upstream
;; spec.alpha-0.6.249, clojure/spec/gen/alpha.clj, 227 lines) over a
;; self-contained seeded RNG, covering the generator surface spec
;; itself uses. No test.check is ported; generator fns answer plain
;; values (no rose trees, no shrinking — that is slice I).
;;
;; Port adaptations (see docs/gap-122/slice-F1.md for the full list):
;; - No syntax-quote and no `::auto-resolve` anywhere in this file:
;;   the port reads a whole load batch before evaluating it, so both
;;   would qualify against the *requiring* namespace instead of this
;;   one (the spec_alpha.clj precedent).
;; - Core fns shadowed by generator defs are always referenced via
;;   the `c` alias (`c/map`, `c/double`, ...), so file order can stay
;;   bottom-up without surprises.
;; - `generator?` is private `satisfies?` over a one-method protocol
;;   (the port cannot `instance?` record classes).
;; - `RuntimeException.` construction is `ex-info` with identical
;;   messages (the spec_alpha.clj precedent).
;; - Seed streams do not match the JVM; `ratio`/`double` edge
;;   distributions are approximate; `uri?` has no builtin (no host
;;   URI constructor); `inst?` covers 1970-2100 only. All pinned by
;;   `rust/crates/clj-eval/tests/gen_alpha.rs`.
(ns clojure.spec.gen.alpha
  (:refer-clojure :exclude [boolean bytes cat list map not-empty set vector
                            char double int keyword symbol string uuid shuffle]))

(alias 'c 'clojure.core)

;; Seeded RNG: splitmix64 over one long of state, threaded immutably
;; like `random.clj`'s `rnd` (API shape mirrors `IRandom`: rand-long,
;; rand-double, split, split-n). The 64-bit mix constants have the
;; high bit set, so they are written as negative long literals.
;; Stream separators: draws mix the state with `+golden-gamma+`,
;; `split` children with `+split-1+`/`+split-2+` (3G/5G mod 2^64), so
;; the draw path and both child paths are disjoint by construction.
(def ^:private +golden-gamma+ -7046029254386353131)
(def ^:private +mix-1+ -4658895280553007687)
(def ^:private +mix-2+ -7723592293110705685)
(def ^:private +split-1+ -2691343689449507777)
(def ^:private +split-2+ 1663341875487337577)
(def ^:private +long-max+ 9223372036854775807)
(def ^:private +long-min+ -9223372036854775808)

(defn- mix64
  [z]
  (let [z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 30)) +mix-1+)
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) +mix-2+)]
    (bit-xor z (unsigned-bit-shift-right z 31))))

(defn- rand-long
  [rng]
  (mix64 (unchecked-add rng +golden-gamma+)))

(defn- rand-double
  [rng]
  (/ (c/double (unsigned-bit-shift-right (rand-long rng) 11)) 9007199254740992.0))

(defn- split
  [rng]
  [(mix64 (unchecked-add rng +split-1+))
   (mix64 (unchecked-add rng +split-2+))])

(defn- split-n
  [rng n]
  (loop [r rng i n acc []]
    (if (zero? i)
      acc
      (let [[r1 r2] (split r)]
        (recur r2 (dec i) (conj acc r1))))))

(defn- make-rng
  [seed]
  (mix64 (unchecked-add (long seed) +golden-gamma+)))

(defn- default-rng
  []
  (make-rng (rand-int 2147483647)))

;; Upstream `generators.cljc` calc-long, minus `Math/floor` (`long`
;; truncates toward zero, which is floor on the non-negative
;; product) and minus JVM saturation (`double_long` throws past 2^63,
;; so the extreme-width arm clamps in double space first, mapping the
;; `0.0 * +Inf` NaN to `lower`).
(defn- calc-long
  [factor lower upper]
  (let [width (-' upper lower -1)]
    (if (< width +long-max+)
      (+ lower (long (* factor width)))
      (let [d (+ (c/double lower) (* factor (c/double width)))
            c (min (c/double upper) (max (c/double lower) d))]
        (cond (not (= c c)) lower
              (>= c 9223372036854775808.0) upper
              :else (long c))))))

(defn- rand-range
  [rnd lower upper]
  (assert (<= lower upper))
  (calc-long (rand-double rnd) lower upper))

;; Upstream `generators.cljc:28-80`. A generator wraps a fn of
;; `[rng size]` answering a plain value (no rose tree in F1; every
;; `rose/*` call in the ported combinators is replaced by its
;; root-value equivalent).
(defprotocol GeneratorP
  (call-gen [g rnd size]))

(defrecord Generator [gen])

(extend-type Generator
  GeneratorP
  (call-gen [g rnd size] ((:gen g) rnd size)))

(defn- generator?
  [x]
  (satisfies? GeneratorP x))

(defn- make-gen
  [gfn]
  (->Generator gfn))

(defn- gen-pure
  [value]
  (make-gen (fn [_ _] value)))

(defn- gen-fmap
  [k g]
  (make-gen (fn [rnd size] (k (call-gen g rnd size)))))

(defn- gen-bind
  [g k]
  (make-gen
   (fn [rnd size]
     (let [[r1 r2] (split rnd)
           inner (call-gen g r1 size)]
       (call-gen (k inner) r2 size)))))

(defn- lazy-random-states
  [rr]
  (lazy-seq
   (let [[r1 r2] (split rr)]
     (cons r1 (lazy-random-states r2)))))

(defn- gen-tuple-gen
  [gens]
  (make-gen
   (fn [rnd size]
     (mapv #(call-gen %1 %2 size) gens (split-n rnd (count gens))))))

(defn return
  "Creates a generator that always returns `value`.
  You can think of this as the `constantly` of generators."
  [value]
  (gen-pure value))

(defn fmap
  "Returns a generator like `gen` but with values transformed by `f`."
  [f gen]
  (assert (generator? gen) "Second arg to fmap must be a generator")
  (gen-fmap f gen))

(defn bind
  "Creates a new generator that passes the result of `gen` into
  function `f`. `f` should return a new generator."
  [generator f]
  (assert (generator? generator) "First arg to bind must be a generator")
  (gen-bind generator f))

(defn sized
  "Creates a generator that depends on the size parameter.
  `sized-gen` is a function that takes an integer and returns
  a generator."
  [sized-gen]
  (make-gen
   (fn [rnd size]
     (call-gen (sized-gen size) rnd size))))

(defn resize
  "Creates a new generator with `size` always bound to `n`."
  [n generator]
  (assert (generator? generator) "Second arg to resize must be a generator")
  (make-gen
   (fn [rnd _]
     (call-gen generator rnd n))))

(defn- sample-seq
  ([generator] (sample-seq generator 200))
  ([generator max-size]
   (let [r (default-rng)
         size-seq (cycle (range 0 max-size))]
     (c/map #(call-gen generator %1 %2)
          (lazy-random-states r)
          size-seq))))

(defn sample
  "Return a sequence of `num-samples` (default 10)
  realized values from `generator`."
  ([generator]
   (sample generator 10))
  ([generator num-samples]
   (assert (generator? generator) "First arg to sample must be a generator")
   (take num-samples (sample-seq generator))))

(defn generate
  "Returns a single sample value from the generator.
  Optional args: size (default 30), seed."
  ([generator]
   (generate generator 30))
  ([generator size]
   (call-gen generator (default-rng) size))
  ([generator size seed]
   (call-gen generator (make-rng seed) size)))

(defn choose
  "Creates a generator that generates integers uniformly in the range
  `lower` to `upper`, inclusive."
  [lower upper]
  (let [lower (long lower)
        upper (long upper)]
    (make-gen
     (fn [rnd _]
       (rand-range rnd lower upper)))))

(defn one-of
  "Creates a generator that randomly chooses a value from the list of
  provided generators."
  [generators]
  (assert (every? generator? generators)
          "Arg to one-of must be a collection of generators")
  (assert (seq generators)
          "one-of cannot be called with an empty collection")
  (bind (choose 0 (dec (count generators)))
        #(nth generators %)))

(defn- pick
  [likelihoods n]
  (->> likelihoods
       (reductions + 0)
       (rest)
       (take-while #(<= % n))
       (count)))

(defn frequency
  "Creates a generator that chooses a generator from `pairs` based on
  the provided likelihoods."
  [pairs]
  (assert (every? (fn [[x g]] (and (number? x) (generator? g)))
                  pairs)
          "Arg to frequency must be a list of [num generator] pairs")
  (let [pairs (filter #(pos? (first %)) pairs)
        total (apply + (c/map first pairs))]
    (assert (seq pairs)
            "frequency must be called with at least one non-zero weight")
    (gen-bind (choose 0 (dec total))
              (fn [x]
                (let [idx (pick (c/map first pairs) x)]
                  (second (nth pairs idx)))))))

(defn elements
  "Creates a generator that randomly chooses an element from `coll`."
  [coll]
  (assert (seq coll) "elements cannot be called with an empty collection")
  (let [v (vec coll)]
    (gen-fmap v (choose 0 (dec (count v))))))

(defn- such-that-helper
  [pred gen {:keys [ex-fn max-tries]} rng size]
  (loop [tries-left max-tries
         rng rng
         size size]
    (if (zero? tries-left)
      (throw (ex-fn {:pred pred, :gen, gen :max-tries max-tries}))
      (let [[r1 r2] (split rng)
            value (call-gen gen r1 size)]
        (if (pred value)
          value
          (recur (dec tries-left) r2 (inc size)))))))

(def ^:private
  default-such-that-opts
  {:ex-fn (fn [{:keys [max-tries] :as arg}]
            (ex-info (str "Couldn't satisfy such-that predicate after "
                          max-tries " tries.")
                     arg))
   :max-tries 10})

(defn such-that
  "Creates a generator that generates values from `gen` that satisfy
  predicate `pred`. Tries `max-tries` times (default 10), growing the
  size on each retry, then throws."
  ([pred gen]
   (such-that pred gen 10))
  ([pred gen max-tries-or-opts]
   (let [opts (cond (integer? max-tries-or-opts)
                    {:max-tries max-tries-or-opts}

                    (map? max-tries-or-opts)
                    max-tries-or-opts

                    :else
                    (throw (ex-info "Bad argument to such-that!" {:max-tries-or-opts
                                                                  max-tries-or-opts})))
         opts (merge default-such-that-opts opts)]
     (assert (generator? gen) "Second arg to such-that must be a generator")
     (make-gen
      (fn [rand-seed size]
        (such-that-helper pred gen opts rand-seed size))))))

(defn not-empty
  "Modifies a generator so that it doesn't generate empty collections."
  [gen]
  (assert (generator? gen) "Arg to not-empty must be a generator")
  (such-that c/not-empty gen))

;; Size-bounded integer building blocks (upstream test.check
;; `nat`/`small-integer`; private here since `gen/alpha.clj` does not
;; re-export them).
(def ^:private nat
  (sized (fn [size] (choose 0 size))))

(def ^:private small-integer
  (sized (fn [size] (choose (- size) size))))

(defn tuple
  "Creates a generator that returns a vector, whose elements are
  chosen from the generators in the same position."
  [& generators]
  (assert (every? generator? generators)
          "Args to tuple must be generators")
  (gen-tuple-gen generators))

(defn vector
  "Creates a generator of vectors whose elements are chosen from
  `generator`. The count of the vector will be bounded by the `size`
  generator parameter."
  ([generator]
   (assert (generator? generator) "Arg to vector must be a generator")
   (gen-bind
    (sized #(choose 0 %))
    (fn [num-elements]
      (gen-tuple-gen (repeat num-elements generator)))))
  ([generator num-elements]
   (assert (generator? generator) "First arg to vector must be a generator")
   (apply tuple (repeat num-elements generator)))
  ([generator min-elements max-elements]
   (assert (generator? generator) "First arg to vector must be a generator")
   (gen-bind
    (choose min-elements max-elements)
    (fn [num-elements]
      (gen-tuple-gen (repeat num-elements generator))))))

(defn list
  "Like `vector`, but generates lists."
  [generator]
  (assert (generator? generator) "First arg to list must be a generator")
  (gen-bind (sized #(choose 0 %))
            (fn [num-elements]
              (fmap #(apply c/list %)
                    (gen-tuple-gen (repeat num-elements generator))))))

(defn- swap
  [coll [i1 i2]]
  (assoc coll i2 (coll i1) i1 (coll i2)))

(defn shuffle
  "Creates a generator that generates random permutations of `coll`.
  `coll` will be coerced to a vector."
  [coll]
  (let [coll (if (vector? coll) coll (vec coll))
        index-gen (choose 0 (dec (count coll)))]
    (fmap #(reduce swap coll %)
          (vector (tuple index-gen index-gen) 0 (* 2 (count coll))))))

;; Upstream `generators.cljc` coll-distinct-by, minus the ordered /
;; allows-dupes arms (only `set`/`map` use it in F1; `the-shuffle-fn`
;; moves with `vector-distinct` to F-remainder) and minus the
;; shrink-tree filters (generation already yields exact counts).
(defn- coll-distinct-by*
  [empty-coll key-fn gen rng size num-elements min-elements max-tries ex-fn]
  (loop [elts []
         s #{}
         rng rng
         size size
         tries 0]
    (cond (and (= max-tries tries)
               (< (count elts) min-elements))
          (throw (ex-fn {:gen gen
                         :max-tries max-tries
                         :num-elements num-elements}))

          (or (= max-tries tries)
              (= (count elts) num-elements))
          (into empty-coll elts)

          :else
          (let [[rng1 rng2] (split rng)
                value (call-gen gen rng1 size)
                k (key-fn value)]
            (if (contains? s k)
              (recur elts s rng2 (inc size) (inc tries))
              (recur (conj elts value)
                     (conj s k)
                     rng2
                     size
                     0))))))

;; NOTE: upstream takes these defaults via `:or` in the arglist
;; destructure; the port miscompiles calls to `:or`-defaulted fn
;; bindings (the callee arrives as its own first arg — pre-existing
;; port bug, visible on any `{:keys [e] :or {e (fn ...)}}` + `(e x)`),
;; so F1 merges a default map exactly like `such-that` does.
(def ^:private
  default-distinct-opts
  {:max-tries 10
   :ex-fn (fn [m] (ex-info "Couldn't generate enough distinct elements!" m))})

(defn- coll-distinct-by
  [empty-coll key-fn gen opts]
  (let [{:keys [num-elements min-elements max-elements max-tries ex-fn]}
        (merge default-distinct-opts opts)
        hard-min-elements (or num-elements min-elements 1)]
    (if num-elements
      (do (assert (and (nil? min-elements) (nil? max-elements)))
          (make-gen
           (fn [rng gen-size]
             (coll-distinct-by* empty-coll key-fn gen rng gen-size
                                num-elements hard-min-elements max-tries ex-fn))))
      (let [min-elements (or min-elements 0)]
        (gen-bind
         (if max-elements
           (choose min-elements max-elements)
           (sized #(choose min-elements (+ min-elements %))))
         (fn [num-elements]
           (make-gen
            (fn [rng gen-size]
              (coll-distinct-by* empty-coll key-fn gen rng gen-size
                                 num-elements hard-min-elements max-tries ex-fn)))))))))

(defn set
  "Generates a set of elements from the given generator.
  Available options: :num-elements :min-elements :max-elements
  :max-tries :ex-fn."
  ([gen] (set gen {}))
  ([gen opts]
   (assert (generator? gen) "First arg to set must be a generator!")
   (coll-distinct-by #{} identity gen opts)))

(defn map
  "Creates a generator that generates maps, with keys chosen from
  `key-gen` and values chosen from `val-gen`.
  Available options: :num-elements :min-elements :max-elements
  :max-tries :ex-fn."
  ([key-gen val-gen] (map key-gen val-gen {}))
  ([key-gen val-gen opts]
   (coll-distinct-by {} first (tuple key-gen val-gen) opts)))

(defn cat
  "Returns a generator of a sequence catenated from results of
  gens, each of which should generate something sequential."
  [& gens]
  (fmap #(apply concat %)
        (apply tuple gens)))

;; Large integers: upstream `generators.cljc` long->large-integer /
;; large-integer** / large-integer*, near-verbatim (`Long/MIN_VALUE`
;; and friends become long literals; `abs` is the port core fn,
;; probe-verified safe on `Long/MIN_VALUE`).
(def ^:private gen-raw-long
  (make-gen (fn [rnd _] (rand-long rnd))))

(defn- long->large-integer
  [bit-count x min max]
  (loop [res (-> x
                 (bit-shift-right (- 64 bit-count))
                 (cond-> (zero? min) (abs)))]
    (if (<= min res max)
      res
      (let [res' (- res)]
        (if (<= min res' max)
          res'
          (recur (bit-shift-right res 1)))))))

(defn- large-integer**
  [min max]
  (sized (fn [size]
           (let [size (c/max size 1)
                 max-bit-count (c/min size 64)]
             (fmap (fn [[bit-count x]]
                     (long->large-integer bit-count x min max))
                   (tuple (choose 1 max-bit-count)
                          gen-raw-long))))))

(defn large-integer*
  "Like large-integer, but accepts options:
    :min  the minimum integer (inclusive)
    :max  the maximum integer (inclusive)"
  [{:keys [min max]}]
  (let [min (or min +long-min+)
        max (or max +long-max+)]
    (assert (<= min max))
    (such-that #(<= min % max)
               (if (<= min 0 max)
                 (large-integer** min max)
                 (if (< max 0)
                   (fmap #(+ max %) (large-integer** (- min max) 0))
                   (fmap #(+ min %) (large-integer** 0 (- max min))))))))

;; Scalar prims mirror upstream `lazy-prims` (`gen/alpha.clj:99-120`):
;; each is a fn returning the generator value (built once in the
;; closing `let`, like the upstream delay), so callers write
;; `(boolean)`, `(string-alphanumeric)`, etc.
(let [g (large-integer* {})]
  (defn large-integer
    "Fn returning the large-integer generator."
    [& _]
    g))

;; Doubles: `double*` keeps the upstream frequency assembly
;; (`generators.cljc` double*), but `double-finite` is an approximate
;; sized small/large mix instead of the bit-level construction
;; (documented divergence 2; slice K bit-faithfulness).
(defn- uniform-double
  [lo hi]
  (make-gen
   (fn [rnd _]
     (let [f (rand-double rnd)
           d (+ lo (* f (- hi lo)))]
       (cond (not (= d d)) lo
             (> d hi) hi
             (< d lo) lo
             :else d)))))

(defn- double-finite
  [min max]
  (cond (and min max)
        (let [lo (c/double min) hi (c/double max)]
          (uniform-double lo hi))

        min
        (let [lo (c/double min)
              spread (c/max 1.0 (c/min (abs lo) 1000000.0))
              hi (c/min (+ lo spread) 1.7976931348623157e308)]
          (uniform-double lo hi))

        max
        (let [hi (c/double max)
              spread (c/max 1.0 (c/min (abs hi) 1000000.0))
              lo (c/max (- hi spread) -1.7976931348623157e308)]
          (uniform-double lo hi))

        :else
        (sized (fn [size]
                 (let [b (c/max 1 size)
                       bd (c/double b)]
                   (one-of [(fmap c/double (choose (- b) b))
                            (uniform-double (- bd) bd)
                            (uniform-double -1000000.0 1000000.0)]))))))

(defn double*
  "Generates a 64-bit floating point number. Options:
    :infinite? - whether +/- infinity can be generated (default true)
    :NaN?      - whether NaN can be generated (default true)
    :min       - minimum value (inclusive, default none)
    :max       - maximum value (inclusive, default none)"
  [{:keys [infinite? NaN? min max]
    :or {infinite? true, NaN? true}}]
  (let [frequency-arg (cond-> [[95 (double-finite min max)]]

                        (if (nil? min)
                          (or (nil? max) (<= 0.0 max))
                          (if (nil? max)
                            (<= min 0.0)
                            (<= min 0.0 max)))
                        (conj [1 (return 0.0)]
                              [1 (return (- 0.0))])

                        (and infinite? (nil? max))
                        (conj [1 (return ##Inf)])

                        (and infinite? (nil? min))
                        (conj [1 (return ##-Inf)])

                        NaN? (conj [1 (return ##NaN)]))]
    (if (= 1 (count frequency-arg))
      (-> frequency-arg first second)
      (frequency frequency-arg))))

(let [g (double* {})]
  (defn double
    "Fn returning the double generator."
    [& _]
    g))

;; Characters & strings: upstream `generators.cljc`, with
;; `clojure.string/join` replaced by `(apply str …)` (the port embeds
;; no `clojure.string`).
(let [g (fmap c/char (choose 0 255))]
  (defn char
    "Fn returning the char generator."
    [& _]
    g))

(let [g (fmap c/char (choose 32 126))]
  (defn char-ascii
    "Fn returning the char-ascii generator."
    [& _]
    g))

(let [g (fmap c/char
              (one-of [(choose 48 57)
                       (choose 65 90)
                       (choose 97 122)]))]
  (defn char-alphanumeric
    "Fn returning the char-alphanumeric generator."
    [& _]
    g))

(let [g (fmap c/char
              (one-of [(choose 65 90)
                       (choose 97 122)]))]
  (defn char-alpha
    "Fn returning the char-alpha generator."
    [& _]
    g))

(def ^:private char-symbol-special
  (elements [\* \+ \! \- \_ \? \.]))

(def ^:private char-symbol-noninitial
  (frequency [[14 (char-alphanumeric)]
              [7 char-symbol-special]
              [1 (return \: )]]))

(def ^:private char-symbol-initial
  (frequency [[2 (char-alpha)]
              [1 char-symbol-special]]))

(let [g (fmap #(apply str %) (vector (char)))]
  (defn string
    "Fn returning the string generator."
    [& _]
    g))

(let [g (fmap #(apply str %) (vector (char-ascii)))]
  (defn string-ascii
    "Fn returning the string-ascii generator."
    [& _]
    g))

(let [g (fmap #(apply str %) (vector (char-alphanumeric)))]
  (defn string-alphanumeric
    "Fn returning the string-alphanumeric generator."
    [& _]
    g))

(defn- digit?
  [d]
  (let [c (c/int d)]
    (and (<= 48 c) (<= c 57))))

(defn- +-or---digit?
  [c d]
  (c/boolean (and d
                  (or (= \+ c) (= \- c))
                  (digit? d))))

;; The two `string/replace` colon cleanups in upstream
;; `symbol-name-or-namespace`, written as a plain char loop (no
;; `clojure.string`, no regex literals in this file).
(defn- collapse-colons
  [s]
  (let [out (loop [chs (seq s) acc []]
              (if (nil? chs)
                acc
                (let [c (first chs)]
                  (recur (next chs)
                         (if (and (= \: c) (= \: (peek acc)))
                           acc
                           (conj acc c))))))
        out (loop [v out]
              (if (and (seq v) (= \: (peek v)))
                (recur (pop v))
                v))]
    (apply str out)))

(def ^:private symbol-name-or-namespace
  (->> (tuple char-symbol-initial (vector char-symbol-noninitial))
       (such-that (fn [[c [d]]] (not (+-or---digit? c d))))
       (fmap (fn [[c cs]]
               (collapse-colons (apply str (cons c cs)))))))

(defn- resize-symbolish
  [g]
  ;; Upstream scales by `n^0.6` (`Math/pow`, absent); `(quot n 4)`
  ;; matches it exactly at the default size 30 (documented sizing
  ;; approximation).
  (sized (fn [n] (resize (quot n 4) g))))

(let [g (->> symbol-name-or-namespace
             (fmap c/keyword)
             (resize-symbolish))]
  (defn keyword
    "Fn returning the keyword generator."
    [& _]
    g))

(let [g (->> (tuple symbol-name-or-namespace symbol-name-or-namespace)
             (fmap (fn [[ns name]] (c/keyword ns name)))
             (resize-symbolish))]
  (defn keyword-ns
    "Fn returning the keyword-ns generator."
    [& _]
    g))

(let [g (frequency [[100
                     (->> symbol-name-or-namespace
                          (fmap c/symbol)
                          (resize-symbolish))]
                    [1 (return '/)]])]
  (defn symbol
    "Fn returning the symbol generator."
    [& _]
    g))

(let [g (->> (tuple symbol-name-or-namespace symbol-name-or-namespace)
             (fmap (fn [[ns name]] (c/symbol ns name)))
             (resize-symbolish))]
  (defn symbol-ns
    "Fn returning the symbol-ns generator."
    [& _]
    g))

(let [g (fmap
         (fn [[a b]] (/ a b))
         (tuple small-integer (fmap inc nat)))]
  (defn ratio
    "Fn returning the ratio generator."
    [& _]
    g))

(def ^:private byte-gen
  (fmap byte (choose -128 127)))

(let [g (fmap byte-array (vector byte-gen))]
  (defn bytes
    "Fn returning the bytes generator."
    [& _]
    g))

(let [g (elements [false true])]
  (defn boolean
    "Fn returning the boolean generator."
    [& _]
    g))

(let [g small-integer]
  (defn int
    "Fn returning the int generator."
    [& _]
    g))

;; UUIDs: upstream builds `java.util.UUID.` from two masked longs;
;; the port has no such ctor, so F1 formats the two drawn longs as
;; 32 hex nibbles with version `4` / variant `[89ab]` and parses
;; (`parse-uuid` round-trip probe-verified).
(def ^:private +hex-chars+ "0123456789abcdef")

(defn- hex-nibble
  [n]
  (subs +hex-chars+ n (inc n)))

(defn- hex-of-long
  [x]
  (apply str
         (c/map #(hex-nibble (bit-and (unsigned-bit-shift-right x (* % 4)) 15))
                (range 15 -1 -1))))

(defn- uuid-string
  [x1 x2]
  (let [h1 (hex-of-long x1)
        h2 (hex-of-long x2)]
    (str (subs h1 0 8) "-"
         (subs h1 8 12) "-"
         "4" (subs h1 13 16) "-"
         (subs "89ab" (bit-and x2 3) (inc (bit-and x2 3))) (subs h2 1 4) "-"
         (subs h2 4 16))))

(let [g (make-gen
          (fn [rnd _]
            (let [[r1 r2] (split rnd)
                  x1 (rand-long r1)
                  x2 (rand-long r2)]
              (parse-uuid (uuid-string x1 x2)))))]
  (defn uuid
    "Fn returning the uuid generator."
    [& _]
    g))

;; Instants: upstream builds `(java.util.Date. millis)`; the port has
;; no millis→Instant constructor, so F1 formats millis as an ISO-8601
;; string (Hinnant civil-from-days + time-of-day split, hand-padded)
;; and calls the `inst` fn from `default-data-readers`. Range is
;; years 1970–2100 (documented divergence 4).
(defn- pad-left
  [w n]
  (let [s (str n)]
    (subs (str (apply str (repeat w "0")) s) (count s))))

(defn- civil-from-days
  [z]
  (let [z (+ z 719468)
        era (quot z 146097)
        doe (- z (* era 146097))
        yoe (quot (- (+ doe (quot doe 36524)) (+ (quot doe 1460) (quot doe 146096))) 365)
        y (+ yoe (* era 400))
        doy (- doe (+ (* 365 yoe) (- (quot yoe 4) (quot yoe 100))))
        mp (quot (+ (* 5 doy) 2) 153)
        d (+ (- doy (quot (+ (* 153 mp) 2) 5)) 1)
        m (if (< mp 10) (+ mp 3) (- mp 9))]
    [(if (<= m 2) (inc y) y) m d]))

(defn- millis->inst-str
  [ms]
  (let [days (quot ms 86400000)
        tod (- ms (* days 86400000))
        hours (quot tod 3600000)
        tod (- tod (* hours 3600000))
        minutes (quot tod 60000)
        tod (- tod (* minutes 60000))
        seconds (quot tod 1000)
        millis (- tod (* seconds 1000))
        [y m d] (civil-from-days days)]
    (str (pad-left 4 y) "-" (pad-left 2 m) "-" (pad-left 2 d)
         "T" (pad-left 2 hours) ":" (pad-left 2 minutes) ":"
         (pad-left 2 seconds) "." (pad-left 3 millis) "Z")))

(defn- millis->inst
  [ms]
  ((get default-data-readers 'inst) (millis->inst-str ms)))

;; Scalars, containers, and recursive generators: upstream
;; `generators.cljc` base-simple-type / container-type / recursive-gen
;; / any. `size-bounded-bigint` is replaced by `large-integer` (no
;; `BigInteger/shiftLeft` in the port); the `Math/pow`-based leaf
;; bound and the log-uniform pseudofactoring become integer
;; approximations (documented sizing approximations).
(defn- base-simple-type
  [double-gen char-gen string-gen]
  (one-of [(int) (large-integer) double-gen char-gen string-gen
           (ratio) (boolean) (keyword) (keyword-ns) (symbol) (symbol-ns) (uuid)]))

(let [g (base-simple-type (double) (char) (string))]
  (defn simple-type
    "Fn returning the simple-type generator."
    [& _]
    g))

(let [g (base-simple-type (double) (char-ascii) (string-ascii))]
  (defn simple-type-printable
    "Fn returning the simple-type-printable generator."
    [& _]
    g))

(defn- scale-half
  [g]
  (sized (fn [n] (resize (quot n 2) g))))

(defn- container-type
  [inner-type]
  (one-of [(vector inner-type)
           (list inner-type)
           (set inner-type)
           (scale-half (map inner-type inner-type))]))

(defn- size->max-leaf-count
  [size]
  ;; Upstream: `(long (Math/pow size 1.1))`; linear match at the
  ;; default size 30 (40 vs 42).
  (+ size (quot size 3)))

(defn- random-pseudofactoring
  [n rng]
  (if (<= n 2)
    [n]
    (let [[r1 r2] (split rng)
          n1 (+ 2 (rand-range r1 0 (- n 2)))
          n2 (quot n n1)]
      (if (and (< 1 n1) (< 1 n2))
        (cons n1 (random-pseudofactoring n2 r2))
        [n]))))

(defn- randomized
  [func]
  (make-gen (fn [rng size]
              (let [[r1 r2] (split rng)]
                (call-gen (func r1) r2 size)))))

(defn- recursive-gen
  [container-gen-fn scalar-gen]
  (assert (generator? scalar-gen)
          "Second arg to recursive-gen must be a generator")
  (sized (fn [size]
           (bind (choose 0 (size->max-leaf-count size))
                 (fn [max-leaf-count]
                   (randomized
                    (fn [rng]
                      (let [sizes (random-pseudofactoring max-leaf-count rng)
                            sized-scalar-gen (resize size scalar-gen)]
                        (reduce (fn [g size]
                                  (bind (choose 0 10)
                                        (fn [x]
                                          (if (zero? x)
                                            sized-scalar-gen
                                            (resize size
                                                    (container-gen-fn g))))))
                                sized-scalar-gen
                                sizes)))))))))

(let [g (recursive-gen container-type (simple-type))]
  (defn any
    "Fn returning the any generator."
    [& _]
    g))

(let [g (recursive-gen container-type (simple-type-printable))]
  (defn any-printable
    "Fn returning the any-printable generator."
    [& _]
    g))

(defn- qualified?
  [ident]
  (not (nil? (namespace ident))))

;; Upstream `gen/alpha.clj:131-189`, minus `uri?` (no host URI
;; constructor in the port — `gen-for-pred` returns nil for `uri?`,
;; documented divergence 3).
(def ^:private
  gen-builtins
  (delay
   (let [simple (simple-type-printable)]
     {any? (one-of [(return nil) (any-printable)])
      some? (such-that some? (any-printable))
      number? (one-of [(large-integer) (double)])
      integer? (large-integer)
      int? (large-integer)
      pos-int? (large-integer* {:min 1})
      neg-int? (large-integer* {:max -1})
      nat-int? (large-integer* {:min 0})
      float? (double)
      double? (double)
      boolean? (boolean)
      string? (string-alphanumeric)
      ident? (one-of [(keyword-ns) (symbol-ns)])
      simple-ident? (one-of [(keyword) (symbol)])
      qualified-ident? (such-that qualified? (one-of [(keyword-ns) (symbol-ns)]))
      keyword? (keyword-ns)
      simple-keyword? (keyword)
      qualified-keyword? (such-that qualified? (keyword-ns))
      symbol? (symbol-ns)
      simple-symbol? (symbol)
      qualified-symbol? (such-that qualified? (symbol-ns))
      uuid? (uuid)
      decimal? (fmap bigdec
                     (double* {:infinite? false :NaN? false}))
      inst? (fmap millis->inst (choose 0 4102444800000))
      seqable? (one-of [(return nil)
                        (list simple)
                        (vector simple)
                        (map simple simple)
                        (set simple)
                        (string-alphanumeric)])
      indexed? (vector simple)
      map? (map simple simple)
      vector? (vector simple)
      list? (list simple)
      seq? (list simple)
      char? (char)
      set? (set simple)
      nil? (return nil)
      false? (return false)
      true? (return true)
      zero? (return 0)
      rational? (one-of [(large-integer) (ratio)])
      coll? (one-of [(map simple simple)
                     (list simple)
                     (vector simple)
                     (set simple)])
      empty? (elements [nil '() [] {} #{}])
      associative? (one-of [(map simple simple) (vector simple)])
      sequential? (one-of [(list simple) (vector simple)])
      ratio? (such-that ratio? (ratio))
      bytes? (bytes)})))

(defn gen-for-pred
  "Given a predicate, returns a built-in generator if one exists."
  [pred]
  (if (set? pred)
    (elements pred)
    (get @gen-builtins pred)))

(defn gen-for-name
  "Resolves the qualified symbol `s` to a generator var. Fails like
  the upstream oracle: unqualified names trip the assert,
  non-generator vars and unresolvable vars throw `ex-info` with the
  oracle texts, unknown namespaces fail the `require`."
  [s]
  (let [n (namespace s)]
    (assert n)
    ;; Upstream `require` no-ops for loaded namespaces (including
    ;; `clojure.core`, which lives in `loaded-libs` on the JVM); the
    ;; port only tracks file-loaded libs, so skip boot namespaces
    ;; explicitly. (Superset: `user`-ns vars resolve here.)
    (let [ns-sym (c/symbol n)]
      (when-not (find-ns ns-sym)
        (require ns-sym)))
    (let [v (resolve s)]
      (if v
        (let [g @v]
          (if (generator? g)
            g
            (throw (ex-info (str "Var " s " is not a generator") {:var s}))))
        (throw (ex-info (str "Var " s " is not on the classpath") {:var s}))))))
