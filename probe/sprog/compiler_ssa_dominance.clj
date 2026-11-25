(ns compiler.ssa-dominance
  "SSA dominance: immediate dominators (Cooper-Harvey-Kennedy), dominance
   frontiers and phi-placement worklist.

   Time: O(n + m) DFS for reverse postorder plus CHK fixpoint (each pass
   O(m); converges in about 2 passes typically, O(n*m) worst) versus the
   original bitset dataflow (quadratic per iteration, cubic worst). DF
   keeps the original's exact single ordered up-pass (quadratic worst):
   textbook Cytron runners would be output-sensitive but compute
   different frontiers than the original on some loopy graphs, breaking
   the byte-identity stdout contract. Phi worklist O(|defs| * |DF|).
   Space: O(n + m). CHK is
   the standard practical-optimal dominator algorithm (faster than
   Lengauer-Tarjan on real graphs despite a weaker worst case). When
   unreachable nodes exist, an exact pure bitset fallback for the idoms
   preserves the original output byte for byte."
  (:refer-clojure :exclude [parse-long])
  (:require [clojure.string :as str]
            [clojure.set :as set]))

(def digit-values
  (zipmap "0123456789" (range 10)))

(def max-long
  "Largest signed 64-bit value; tokens beyond long range are invalid."
  9223372036854775807)

(def min-long
  "Smallest signed 64-bit value."
  -9223372036854775808)

(defn fit-long
  "Returns n when it fits in a signed 64-bit long, else nil."
  [n]
  (when (<= min-long n max-long) n))


(defn parse-long
  [s]
  (when (and (string? s) (seq s))
    (let [cs (seq s)
          [neg? rest-cs] (cond
                           (= (first cs) \-) [true (next cs)]
                           (= (first cs) \+) [false (next cs)]
                           :else [false cs])]
      (when (seq rest-cs)
        (loop [cs rest-cs
               acc 0]
          (if (nil? cs)
            (fit-long (if neg? (-' acc) acc))
            (let [d (get digit-values (first cs))]
              (when-not (nil? d)
                (recur (next cs) (+' (*' acc 10) d))))))))))

(defn error
  [msg]
  (binding [*out* *err*]
    (println msg))
  nil)

(defn reverse-postorder
  "Iterative DFS postorder from `root` over `succ` (1-based vector of
   vectors). Returns [rpo rpo-num]."
  [succ root n]
  (let [[_ post]
        (loop [visited #{root}
               stack [[root 0]]
               post []]
          (if (empty? stack)
            [visited post]
            (let [[u i] (peek stack)
                  vs (nth succ u)]
              (if (< i (count vs))
                (let [v (nth vs i)
                      stack (assoc stack (dec (count stack)) [u (inc i)])]
                  (if (contains? visited v)
                    (recur visited stack post)
                    (recur (conj visited v) (conj stack [v 0]) post)))
                (recur visited (pop stack) (conj post u))))))
        rpo (vec (reverse post))]
    [rpo (into {} (map-indexed (fn [idx node] [node idx]) rpo))]))

(defn intersect
  "Common dominator of b1,b2 via idom chains guided by `rpo-num`."
  [idom rpo-num b1 b2]
  (loop [f1 b1
         f2 b2]
    (if (= f1 f2)
      f1
      (if (> (get rpo-num f1) (get rpo-num f2))
        (recur (get idom f1) f2)
        (recur f1 (get idom f2))))))

(defn compute-idom
  "CHK immediate dominators. Returns map node -> idom (root maps to
   itself; unreachable absent)."
  [succ pred root n rpo rpo-num]
  (let [idom {root root}]
    (loop [idom idom]
      (let [[idom changed?]
            (reduce (fn [[idm ch] b]
                      (if (= b root)
                        [idm ch]
                        (let [procs (filter #(contains? idm %) (nth pred b))]
                          (if (empty? procs)
                            [idm ch]
                            (let [new-idom (reduce #(intersect idm rpo-num %1 %2)
                                                   (first procs) (rest procs))]
                              (if (= (get idm b) new-idom)
                                [idm ch]
                                [(assoc idm b new-idom) true]))))))
                    [idom false]
                    (rest rpo))]
        (if changed?
          (recur idom)
          idom)))))

(defn naive-df
  "Dominance frontiers exactly as the original computes them: a local
   edge pass, then ONE up-propagation sweep in ascending node order.
   That single ordered sweep is order-dependent and under-computes
   versus textbook Cytron DF on some loopy graphs (e.g. a defined loop
   header missing from its own frontier) -- but the stdout contract is
   byte-identity with the original, so this quirk is replicated, not
   fixed. `idomv` is the (n+1) idom vector (root->0). Returns vector
   (n+1) of sorted-sets."
  [succ idomv n]
  (let [df1 (reduce (fn [df x]
                      (reduce (fn [df2 y]
                                (if (not= (nth idomv y) x)
                                  (update df2 x conj y)
                                  df2))
                              df (nth succ x)))
                    (vec (repeat (inc n) (sorted-set)))
                    (range 1 (inc n)))]
    (reduce (fn [df x]
              (reduce (fn [df2 z]
                        (if (= (nth idomv z) x)
                          (reduce (fn [df3 y]
                                    (if (not= (nth idomv y) x)
                                      (update df3 x conj y)
                                      df3))
                                  df2 (seq (nth df2 z)))
                          df2))
                      df (range 1 (inc n))))
            df1 (range 1 (inc n)))))

(defn queue-from
  "Pure FIFO queue as [items read-index]; O(1) amortized ops."
  [coll]
  [(vec coll) 0])

(defn q-empty?
  [[v i]]
  (>= i (count v)))

(defn q-peek
  [[v i]]
  (nth v i))

(defn q-pop
  [[v i]]
  (let [j (inc i)]
    (if (and (> j 64) (> j (quot (count v) 2)))
      [(subvec v j) 0]
      [v j])))

(defn q-conj
  [[v i] x]
  [(conj v x) i])

(defn reachable-from
  "Set of nodes reachable from `root` over `succ`."
  [succ root]
  (loop [seen #{root}
         stack [root]]
    (if (empty? stack)
      seen
      (let [u (peek stack)
            stack (pop stack)
            [seen stack]
            (reduce (fn [[sn st] v]
                      (if (contains? sn v)
                        [sn st]
                        [(conj sn v) (conj st v)]))
                    [seen stack]
                    (nth succ u))]
        (recur seen stack)))))

(defn bitset-idom-df
  "Exact pure replication of the original bitset fixpoint plus naive DF
   loops. Used only when unreachable nodes exist, where the original
   output is defined by fixpoint artifacts (arbitrary idom for
   unreachable nodes) rather than dominance; CHK is undefined there.
   Returns [idom-vector df-vector]."
  [succ pred n]
  (let [all-nodes (into #{} (range 1 (inc n)))
        dom0 (reduce (fn [v i] (assoc v i all-nodes))
                     (assoc (vec (repeat (inc n) nil)) 1 #{1})
                     (range 2 (inc n)))
        dom-final
        (loop [dom dom0]
          (let [[dom changed?]
                (reduce (fn [[d ch] i]
                          (let [preds (nth pred i)]
                            (if (empty? preds)
                              [d ch]
                              (let [new-dom (conj (reduce (fn [acc p]
                                                            (set/intersection acc (nth d p)))
                                                          (nth d (first preds))
                                                          (rest preds))
                                                  i)]
                                (if (= new-dom (nth d i))
                                  [d ch]
                                  [(assoc d i new-dom) true])))))
                        [dom false]
                        (range 2 (inc n)))]
            (if changed?
              (recur dom)
              dom)))
        idomv
        (reduce (fn [iv i]
                  (let [dom-i (nth dom-final i)
                        sz (count dom-i)]
                    (reduce (fn [iv2 d]
                              (if (and (not= d i)
                                       (= sz (inc (count (nth dom-final d)))))
                                (assoc iv2 i d)
                                iv2))
                            iv (seq dom-i))))
                (vec (repeat (inc n) 0))
                (range 2 (inc n)))
        df (naive-df succ idomv n)]
    [idomv df]))

(defn phi-placement
  "Iterated dominance frontier of `defs`. Returns sorted-set."
  [df defs]
  (loop [phi (sorted-set)
         w (queue-from defs)]
    (if (q-empty? w)
      phi
      (let [x (q-peek w)
            w (q-pop w)
            [phi w]
            (reduce (fn [[ph qq] y]
                      (if (contains? ph y)
                        [ph qq]
                        [(conj ph y)
                         (if (contains? defs y) qq (q-conj qq y))]))
                    [phi w]
                    (nth df x))]
        (recur phi w)))))

(defn -main
  [& _args]
  (let [s (slurp *in*)]
    (when-not (str/blank? s)
      (let [toks (vec (remove str/blank? (str/split s #"\s+")))
            n (parse-long (get toks 0))
            m (parse-long (get toks 1))]
        (cond
          (or (nil? n) (nil? m))
          (error "invalid input: expected n m header")

          (or (< n 1) (neg? m))
          (error "invalid input: bad n m values")

          (< (count toks) (+' 2 (*' m 2) 1))
          (error "invalid input: truncated edge list")

          :else
          (let [pairs (take (* m 2) (drop 2 toks))
                tail (vec (drop (+ 2 (* m 2)) toks))
                edges (vec (for [[a b] (partition 2 pairs)]
                             [(parse-long a) (parse-long b)]))]
            (cond
              (some (fn [[u v]] (or (nil? u) (nil? v))) edges)
              (error "invalid input: bad edge")

              (some (fn [[u v]]
                      (or (< u 1) (> u n) (< v 1) (> v n)))
                    edges)
              (error "invalid input: edge out of range")

              :else
              (let [succ (reduce (fn [a [u v]] (update a u conj v))
                                 (vec (repeat (inc n) [])) edges)
                    pred (reduce (fn [a [u v]] (update a v conj u))
                                 (vec (repeat (inc n) [])) edges)
                    def-count (parse-long (get tail 0))]
                (cond
                  (nil? def-count)
                  (error "invalid input: expected def count")

                  (neg? def-count)
                  (error "invalid input: negative def count")

                  (< (count (rest tail)) def-count)
                  (error "invalid input: truncated def list")

                  :else
                  (let [def-vals (map parse-long (take def-count (rest tail)))]
                    (cond
                      (some nil? def-vals)
                      (error "invalid input: bad def")

                      (some #(or (< % 1) (> % n)) def-vals)
                      (error "invalid input: def out of range")

                      :else
                      (let [defs (into (sorted-set) def-vals)
                            reach (reachable-from succ 1)
                            connected? (every? reach (range 1 (inc n)))
                            [idomv df]
                            (if connected?
                              (let [[rpo rpo-num] (reverse-postorder succ 1 n)
                                    idom (compute-idom succ pred 1 n rpo rpo-num)
                                    ;; CHK maps root->root internally; the
                                    ;; original leaves idom[1] as 0, which
                                    ;; the DF comparisons depend on.
                                    idomv (assoc (reduce (fn [v [k val]]
                                                           (assoc v k val))
                                                         (vec (repeat (inc n) 0))
                                                         idom)
                                                 1 0)]
                                [idomv (naive-df succ idomv n)])
                              (bitset-idom-df succ pred n))
                            phi (phi-placement df defs)]
                        (println (str/join " "
                                           (for [i (range 2 (inc n))]
                                             (str "idom(" i ")=" (nth idomv i)))))
                        (if (empty? phi)
                          (println "NONE")
                          (println (str/join " " phi)))))))))))))))

(-main)
