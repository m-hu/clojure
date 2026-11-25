(ns compiler.earley-parser
  "Earley parser (recognizer) for context-free grammars with per-query
   item counts.

   Time: cubic worst case (quadratic when unambiguous, linear when
   deterministic), optimal for general CFG recognition. The predictor
   uses a left-hand-side index (constant rules per nonterminal) instead
   of the original linear scan in |P| per item; charts are persistent
   vectors with constant-time membership sets. Space: quadratic items.
   Output (accept or reject plus total chart items) matches the original
   exactly, including snapshot semantics of the completer."
  (:refer-clojure :exclude [parse-long])
  (:require [clojure.string :as str]))

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

(defn uppercase?
  "True for A-Z (pure char comparison via `compare`)."
  [c]
  (and (<= (compare \A c) 0) (<= (compare c \Z) 0)))

(defn terminal-symbol?
  "True for terminals: anything but uppercase nonterminals and #."
  [c]
  (and (not (uppercase? c)) (not= c \#)))

(defn error
  [msg]
  (binding [*out* *err*]
    (println msg))
  nil)

(defn add-item
  "Add `item` to chart position `idx` unless present. Returns [chart in-chart]."
  [chart in-chart idx item]
  (if (contains? (nth in-chart idx) item)
    [chart in-chart]
    [(assoc chart idx (conj (nth chart idx) item))
     (assoc in-chart idx (conj (nth in-chart idx) item))]))

(defn run-earley
  "Run Earley on word `w` (string). Returns [accepted? total-items]."
  [w prods rules-by-lhs]
  (let [n (count w)
        chars (vec w)
        chart0 (vec (repeat (inc n) []))
        in0 (vec (repeat (inc n) #{}))
        [chart1 in1] (add-item chart0 in0 0 [0 0 0])]
    (loop [k 0
           chart chart1
           in-chart in1
           tot 0]
      (if (> k n)
        (let [accepted? (contains? (nth in-chart n) [0 1 0])]
          [accepted? tot])
        (let [[chart in-chart]
              (loop [i 0
                     chart chart
                     in-chart in-chart]
                (let [ck (nth chart k)]
                  (if (>= i (count ck))
                    [chart in-chart]
                    (let [[rule-id dot origin] (nth ck i)
                          prod (nth prods rule-id)
                          rhs (:rhs prod)
                          rhs-len (count rhs)]
                      (if (< dot rhs-len)
                        (let [next-sym (nth rhs dot)]
                          (if-not (terminal-symbol? next-sym)
                            (let [[c2 s2]
                                  (reduce (fn [[c s] r]
                                            (add-item c s k [r 0 k]))
                                          [chart in-chart]
                                          (get rules-by-lhs next-sym []))]
                              (recur (inc i) c2 s2))
                            (if (and (< k n) (= (nth chars k) next-sym))
                              (let [[c2 s2]
                                    (add-item chart in-chart (inc k)
                                              [rule-id (inc dot) origin])]
                                (recur (inc i) c2 s2))
                              (recur (inc i) chart in-chart))))
                        (let [completed-lhs (:lhs prod)
                              origin-items (nth chart origin)
                              [c2 s2]
                              (reduce (fn [[c s] [p-id p-dot p-origin]]
                                        (let [prev-rhs (:rhs (nth prods p-id))]
                                          (if (and (< p-dot (count prev-rhs))
                                                   (= (nth prev-rhs p-dot) completed-lhs))
                                            (add-item c s k [p-id (inc p-dot) p-origin])
                                            [c s])))
                                      [chart in-chart]
                                      origin-items)]
                          (recur (inc i) c2 s2)))))))]
          (recur (inc k) chart in-chart (+ tot (count (nth chart k)))))))))

(defn -main
  [& _args]
  (let [s (slurp *in*)]
    (when-not (str/blank? s)
      (let [toks (vec (remove str/blank? (str/split s #"\s+")))
            num-rules (parse-long (get toks 0))]
        (cond
          (nil? num-rules)
          (error "invalid input: expected rule count")

          (neg? num-rules)
          (error "invalid input: negative rule count")

          (< (count toks) (+' 2 (*' 3 num-rules) 1))
          (error "invalid input: truncated rule list")

          :else
          (let [start-symbol (get toks 1)
                rule-toks (take (* 3 num-rules) (drop 2 toks))
                tail (vec (drop (+ 2 (* 3 num-rules)) toks))
                prods (into [{:lhs \@ :rhs (str (first start-symbol))}]
                            (for [[lhs _ rhs] (partition 3 rule-toks)]
                              {:lhs (first lhs)
                               :rhs (if (= rhs "#") "" rhs)}))
                rules-by-lhs (reduce (fn [m [idx p]]
                                       (update m (:lhs p) (fnil conj []) idx))
                                     {} (map-indexed vector prods))
                num-queries (parse-long (get tail 0))]
            (cond
              (nil? num-queries)
              (error "invalid input: expected query count")

              (neg? num-queries)
              (error "invalid input: negative query count")

              (< (count (rest tail)) num-queries)
              (error "invalid input: truncated query list")

              :else
              (doseq [w (take num-queries (rest tail))]
                (let [w (if (= w "EPS") "" w)
                      [accepted? total] (run-earley w prods rules-by-lhs)]
                  (if accepted?
                    (println (str "ACCEPT " total))
                    (println (str "REJECT " total))))))))))))

(-main)
