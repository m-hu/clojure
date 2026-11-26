(ns compiler.lexer-regex-nfa
  "Regex lexer via Thompson NFA construction plus on-the-fly simulation.

   Supports literals (letters/digits), concatenation (implicit), union (|),
   Kleene star (*) and parentheses. Time: O(m) to insert concatenation
   markers, O(m) shunting-yard to postfix, O(m) Thompson construction
   (m = pattern length), and O(m * n) NFA simulation per query of length
   n (each step closes epsilon moves and follows char edges). This is
   optimal for NFA simulation without powerset blowup. Space: O(m)
   states and transitions."
  (:refer-clojure :exclude [parse-long])
  (:require [clojure.string :as str]))

(defn digit?
  [c]
  (and (<= (compare \0 c) 0) (<= (compare c \9) 0)))

(defn upper?
  [c]
  (and (<= (compare \A c) 0) (<= (compare c \Z) 0)))

(defn lower?
  [c]
  (and (<= (compare \a c) 0) (<= (compare c \z) 0)))

(defn alnum?
  "ASCII letter or digit."
  [c]
  (or (digit? c) (upper? c) (lower? c)))

(def epsilon \u0000)

(defn get-prec
  [c]
  (case c
    \* 3
    \. 2
    \| 1
    0))

(defn insert-concat
  "Insert explicit concatenation operator \\. where juxtaposition implies it."
  [regex]
  (let [chars (vec regex)
        n (count chars)]
    (apply str
           (loop [i 0
                  out []]
             (if (>= i n)
               out
               (let [c1 (nth chars i)
                     out (conj out c1)]
                 (if (< (inc i) n)
                   (let [c2 (nth chars (inc i))
                         c1-can-end (or (alnum? c1) (= c1 \*) (= c1 \)))
                         c2-can-start (or (alnum? c2) (= c2 \())]
                     (recur (inc i) (if (and c1-can-end c2-can-start)
                                      (conj out \.)
                                      out)))
                   (recur (inc i) out))))))))

(defn to-postfix
  "Shunting-yard: infix regex (with \\.) to postfix string."
  [infix]
  (let [chars (vec infix)]
    (loop [cs chars
           out []
           stk []]
      (if (empty? cs)
        (apply str (into out (reverse stk)))
        (let [c (first cs)
              rest-cs (rest cs)]
          (cond
            (alnum? c)
            (recur rest-cs (conj out c) stk)

            (= c \()
            (recur rest-cs out (conj stk c))

            (= c \))
            (let [[out stk]
                  (loop [out out
                         stk stk]
                    (if (or (empty? stk) (= (peek stk) \())
                      [out stk]
                      (recur (conj out (peek stk)) (pop stk))))]
              (recur rest-cs out (if (empty? stk) stk (pop stk))))

            :else
            (let [[out stk]
                  (loop [out out
                         stk stk]
                    (if (or (empty? stk)
                            (= (peek stk) \()
                            (< (get-prec (peek stk)) (get-prec c)))
                      [out stk]
                      (recur (conj out (peek stk)) (pop stk))))]
              (recur rest-cs out (conj stk c)))))))))

(defn error
  [msg]
  (binding [*out* *err*]
    (println msg))
  nil)

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

(defn build-nfa
  "Thompson construction over postfix chars. Returns [trans start end] or
   nil when the postfix underflows (malformed pattern)."
  [postfix]
  (let [chars (vec postfix)]
    (loop [cs chars
           trans []
           stk []]
      (if (empty? cs)
        (when (= 1 (count stk))
          (let [[s e] (peek stk)]
            [trans s e]))
        (let [c (first cs)
              rest-cs (rest cs)]
          (cond
            (= c \.)
            (when (>= (count stk) 2)
              (let [f2 (peek stk)
                    f1 (nth stk (- (count stk) 2))
                    stk (pop (pop stk))
                    trans (update trans (nth f1 1) conj [epsilon (nth f2 0)])]
                (recur rest-cs trans (conj stk [(nth f1 0) (nth f2 1)]))))

            (= c \|)
            (when (>= (count stk) 2)
              (let [f2 (peek stk)
                    f1 (nth stk (- (count stk) 2))
                    stk (pop (pop stk))
                    s (count trans)
                    e (inc s)
                    trans (conj trans [] [])
                    trans (-> trans
                              (update s conj [epsilon (nth f1 0)])
                              (update s conj [epsilon (nth f2 0)])
                              (update (nth f1 1) conj [epsilon e])
                              (update (nth f2 1) conj [epsilon e]))]
                (recur rest-cs trans (conj stk [s e]))))

            (= c \*)
            (when (>= (count stk) 1)
              (let [f (peek stk)
                    stk (pop stk)
                    s (count trans)
                    e (inc s)
                    trans (conj trans [] [])
                    trans (-> trans
                              (update s conj [epsilon (nth f 0)])
                              (update s conj [epsilon e])
                              (update (nth f 1) conj [epsilon (nth f 0)])
                              (update (nth f 1) conj [epsilon e]))]
                (recur rest-cs trans (conj stk [s e]))))

            :else
            (let [s (count trans)
                  e (inc s)
                  trans (conj trans [] [])
                  trans (update trans s conj [c e])]
              (recur rest-cs trans (conj stk [s e])))))))))

(defn epsilon-closure
  "All states reachable from `start-set` via epsilon moves."
  [trans start-set]
  (loop [closure (set start-set)
         q (queue-from start-set)]
    (if (q-empty? q)
      closure
      (let [u (q-peek q)
            q (q-pop q)
            [closure q]
            (reduce (fn [[cl qq] [ch v]]
                      (if (and (= ch epsilon) (not (contains? cl v)))
                        [(conj cl v) (q-conj qq v)]
                        [cl qq]))
                    [closure q]
                    (nth trans u))]
        (recur closure q)))))

(defn matches?
  "NFA simulation of string `tc` from `start-state` to `end-state`."
  [trans start-state end-state tc]
  (let [chars (vec tc)]
    (loop [i 0
           curr (epsilon-closure trans [start-state])]
      (if (< i (count chars))
        (let [ch (nth chars i)
              next-states (reduce (fn [acc u]
                                    (reduce (fn [a [c v]]
                                              (if (= c ch) (conj a v) a))
                                            acc (nth trans u)))
                                  #{} curr)]
          (let [closed (epsilon-closure trans next-states)]
            (if (empty? closed)
              false
              (recur (inc i) closed))))
        (contains? curr end-state)))))

(defn -main
  [& _args]
  (let [s (slurp *in*)]
    (when-not (str/blank? s)
      (let [toks (vec (remove str/blank? (str/split s #"\s+")))
            pattern (get toks 0)
            num-cases (if (> (count toks) 1)
                        (parse-long (get toks 1))
                        0)
            num-cases (if (nil? num-cases) 0 num-cases)]
        (if (neg? num-cases)
          (error "invalid input: negative case count")
          (let [cases (take num-cases (drop 2 toks))]
            (if (< (count (drop 2 toks)) num-cases)
              (error "invalid input: truncated case list")
              (let [infix (insert-concat pattern)
                    postfix (to-postfix infix)
                    built (build-nfa postfix)]
                (if (nil? built)
                  (error "invalid input: malformed pattern")
                  (let [[trans start-state end-state] built]
                    (doseq [tc cases]
                      (if (matches? trans start-state end-state tc)
                        (println "MATCH")
                        (println "NO_MATCH")))))))))))))

(-main)
