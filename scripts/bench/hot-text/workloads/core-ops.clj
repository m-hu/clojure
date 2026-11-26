;; Core operations most programs reach that the other workloads skip:
;; reading all of stdin, the comparison and arithmetic ops over locals,
;; constants and captured values (tested and as values, inside loops long
;; enough for the loop tier), vector equality, hashing and ordering, the
;; filter / concat / drop-while walks, number parsing, comparator-sorted
;; collections.
(def input (slurp *in*))

(defn tally
  "Counts of each comparison over `n` iterations, against a param, a
  constant and a captured value, in test and value position."
  [n k]
  (let [u (- n 3)
        f (fn [i] [(< i u) (<= u i) (> i u) (>= u i) (= i u) (== u i) (compare i u) (+ i u) (- u i) (* i u) (min i u) (max u i)])]
    (loop [i 0 lt 0 le 0 gt 0 ge 0 eq 0 ne 0 acc 0 m 0 v []]
      (if (< i n)
        (recur (inc i)
               (if (< i k) (inc lt) lt)
               (if (<= i 7) (inc le) le)
               (if (> k i) (inc gt) gt)
               (if (>= 7 i) (inc ge) ge)
               (if (= i k) (inc eq) eq)
               (if (not= i 7) (inc ne) ne)
               (+ acc (* i 2) (- k i) (* 3 i) (min i k) (max 7 i) (compare i k) (compare 7 i))
               (if (== i 7) (inc m) (if (zero? (rem i 5)) (+ m 2) m))
               (if (and (pos? i) (not (neg? (- i k))) (some? v) (not (nil? k))) (conj v (- i)) v))
        [lt le gt ge eq ne acc m (count v) (f n) (f (- n)) (double acc) (long (/ acc 2))]))))

(defn vectors
  "Vector equality, hashing, ordering, and vectors as keys."
  [n]
  (let [pts (vec (for [i (range n)] [(rem i 7) (quot i 7)]))
        seen (set pts)
        by (group-by first pts)
        sorted (sort pts)
        cmp (sorted-set-by (fn [a b] (compare (second a) (second b))) [1 2] [3 1] [2 3])]
    [(count seen) (contains? seen [3 1]) (= (first sorted) [0 0]) (= [1 2] [1 2]) (not= [1 2] [2 1])
     (compare [1 2] [1 3]) (hash [1 2]) (count by) (first cmp) (get (zipmap pts (range)) [2 2])
     (= (seq [1 2]) [1 2]) (= '(1 2) [1 2])]))

(defn walks
  "The lazy filter, concat and while walks, and their eager forms."
  [n]
  (let [xs (vec (range (- n) n))]
    [(count (filter even? xs)) (count (filterv odd? xs)) (count (remove neg? xs))
     (count (concat xs (reverse xs))) (first (drop-while neg? xs)) (count (take-while neg? xs))
     (count (concat (filter pos? xs) (take 3 xs) [:x])) (vec (take 3 (drop-while #(< % 2) xs)))]))

(defn numbers
  "Number parsing and the double and ratio arithmetic it feeds."
  [n]
  (let [ds (mapv #(parse-double (str % ".25")) (range n))
        ls (mapv #(parse-long (str %)) (range n))
        rs (mapv #(/ % 3) (range 1 n))]
    [(reduce + ds) (reduce + ls) (reduce + rs) (apply max ds) (double (apply min rs))
     (quot (reduce * 1 (take 10 ls)) 7) (Math/sqrt (reduce + ds)) (Math/abs -2.5)]))

(defn sorted-colls
  "Comparator-sorted sets and maps, and range queries over them."
  [n]
  (let [ss (into (sorted-set-by >) (range n))
        sm (into (sorted-map-by (fn [a b] (compare b a))) (map (fn [i] [i (* i i)]) (range n)))]
    [(first ss) (count (subseq ss > 5)) (first sm) (count (rsubseq sm < 10)) (get sm 3) (contains? ss 4)]))

;; Each binary op over every operand shape the loop tier specializes: two
;; locals, a local and a constant (both orders), an upvalue (a value the
;; enclosing fn captured), and computed operands; tested and as values.
(defmacro shapes [op a b]
  `(let [up# ~b]
     (fn [n#]
       (loop [i# 1 acc# []]
         (if (< i# n#)
           (let [j# (inc i#)
                 vs# [(~op i# j#) (~op i# ~b) (~op ~a j#) (~op up# i#) (~op i# up#) (~op up# up#)
                      (~op (inc i#) (dec j#)) (~op i# (inc j#)) (~op (inc i#) j#) (~op up# (inc i#))]
                 ts# [(if (~op i# j#) 1 0) (if (~op i# ~b) 1 0) (if (~op ~a j#) 1 0) (if (~op up# i#) 1 0)
                      (if (~op (inc i#) j#) 1 0) (if (~op i# (inc j#)) 1 0) (if (~op up# (inc i#)) 1 0)]]
             (recur j# (if (< i# 3) (conj acc# vs# ts#) acc#)))
           acc#)))))

(def binary-ops
  [(shapes < 3 5) (shapes <= 3 5) (shapes > 3 5) (shapes >= 3 5) (shapes = 3 5) (shapes == 3 5)
   (shapes not= 3 5) (shapes + 3 5) (shapes - 3 5) (shapes * 3 5) (shapes min 3 5) (shapes max 3 5)
   (shapes compare 3 5) (shapes quot 3 5) (shapes rem 3 5) (shapes mod 3 5) (shapes bit-and 3 5)])

;; Unary ops and collection reads over a local, an upvalue and a computed
;; operand, in loops.
(defn unary [n]
  (let [v [1 2 3] m {:a 1 :b 2} s #{1 2}]
    (loop [i 0 acc 0 out []]
      (if (< i n)
        (let [k (rem i 3) w (conj v i)]
          (recur (inc i)
                 (+ acc (if (zero? k) 1 0) (if (pos? k) 1 0) (if (neg? (- k)) 1 0) (if (nil? (get m k)) 1 0)
                    (if (not (contains? s k)) 1 0) (if (contains? m :a) 1 0) (if (some? (get w k)) 1 0)
                    (nth v k) (nth w (inc k)) (count w) (second w) (get m :b) (long (/ i 2)) (- k)
                    (count v) (second v) (nth v 1) (if (even? i) 1 0) (if (odd? k) 1 0))
                 (if (< i 2) (conj out [(zero? k) (pos? i) (not (zero? k)) (- i) (nil? w) (empty? w) (first w)]) out)))
        [acc out]))))

;; Lazy walks consumed one item at a time.
(defn lazy-walks [n]
  (let [xs (vec (range (- n) n))]
    (loop [s (seq (concat (filter even? xs) (drop-while neg? xs) (take-while neg? xs) (remove odd? xs))) acc 0]
      (if s (recur (next s) (+ acc (first s))) acc))))

(println (count input) (tally 200 37) (vectors 120) (walks 100) (numbers 80) (sorted-colls 60)
         (mapv (fn [f] (count (f 100))) binary-ops) (unary 150) (lazy-walks 100))
